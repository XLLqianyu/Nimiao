//! Streaming against a real socket.
//!
//! Each test starts a hand-rolled HTTP/1.1 server on a loopback port and points
//! an [`OpenAIModel`] at it. No HTTP test crate is needed; the server reads one
//! request and answers it.

use model::openai::OpenAIModel;
use model::{Model, ModelEvent, ModelResponse};
use protocol::Item;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

/// One canned HTTP response.
struct Reply {
    status: &'static str,
    headers: &'static str,
    body: &'static str,
}

impl Reply {
    /// A successful completion carried by an event stream.
    fn sse(body: &'static str) -> Self {
        Self {
            status: "200 OK",
            headers: "Content-Type: text/event-stream\r\n",
            body,
        }
    }

    /// A successful completion delivered whole.
    fn json(body: &'static str) -> Self {
        Self {
            status: "200 OK",
            headers: "Content-Type: application/json\r\n",
            body,
        }
    }
}

/// A model bound to a server that answers with `replies` in order.
///
/// Returns the model and a counter of requests received so far.
async fn model_replying(replies: Vec<Reply>) -> (OpenAIModel, Arc<AtomicUsize>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base_url = format!("http://{}/v1", listener.local_addr().unwrap());
    let served = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&served);

    tokio::spawn(async move {
        for reply in replies {
            let (mut stream, _) = listener.accept().await.unwrap();
            read_request(&mut stream).await;
            counter.fetch_add(1, Ordering::SeqCst);
            write_response(&mut stream, reply).await;
        }
    });

    (
        OpenAIModel::new("test-key", &base_url, "test-model"),
        served,
    )
}

/// Reads one request: the head, then exactly `Content-Length` body bytes.
async fn read_request(stream: &mut TcpStream) -> String {
    let mut received = Vec::new();
    let mut buffer = [0u8; 1024];
    let head_end = loop {
        let read = stream.read(&mut buffer).await.unwrap();
        assert!(read > 0, "the client closed the connection mid-request");
        received.extend_from_slice(&buffer[..read]);
        if let Some(index) = received.windows(4).position(|part| part == b"\r\n\r\n") {
            break index + 4;
        }
    };

    let head = String::from_utf8_lossy(&received[..head_end]).to_ascii_lowercase();
    let content_length = head
        .lines()
        .find_map(|line| line.strip_prefix("content-length:"))
        .map(|value| value.trim().parse::<usize>().unwrap())
        .unwrap_or(0);

    while received.len() < head_end + content_length {
        let read = stream.read(&mut buffer).await.unwrap();
        assert!(read > 0, "the client closed the connection mid-body");
        received.extend_from_slice(&buffer[..read]);
    }

    String::from_utf8_lossy(&received).into_owned()
}

/// Writes one response and closes the connection.
async fn write_response(stream: &mut TcpStream, reply: Reply) {
    let response = format!(
        "HTTP/1.1 {}\r\n{}Content-Length: {}\r\nConnection: close\r\n\r\n{}",
        reply.status,
        reply.headers,
        reply.body.len(),
        reply.body
    );
    stream.write_all(response.as_bytes()).await.unwrap();
    stream.shutdown().await.unwrap();
}

/// Runs a streaming completion and collects the events it reported.
async fn stream_events(model: &OpenAIModel) -> (ModelResponse, Vec<ModelEvent>) {
    let mut events = Vec::new();
    let response = model
        .complete_stream(&[], &[], &mut |event| events.push(event))
        .await
        .unwrap();
    (response, events)
}

#[tokio::test]
async fn streams_text_deltas() {
    let (model, served) = model_replying(vec![Reply::sse(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"}}]}\n\n\
         data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hel\"}}]}\n\n\
         data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"}}]}\n\n\
         data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
         data: [DONE]\n\n",
    )])
    .await;

    let (response, events) = stream_events(&model).await;

    assert_eq!(served.load(Ordering::SeqCst), 1);
    assert_eq!(response.content.as_deref(), Some("Hello"));
    assert!(response.tool_calls.is_empty());
    assert_eq!(
        events,
        [
            ModelEvent::Text("Hel".to_string()),
            ModelEvent::Text("lo".to_string())
        ]
    );
}

#[tokio::test]
async fn streams_tool_calls() {
    let (model, _) = model_replying(vec![Reply::sse(
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"bash\",\"arguments\":\"\"}}]}}]}\n\n\
         data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"command\\\":\"}}]}}]}\n\n\
         data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"cargo test\\\"}\"}}]}}]}\n\n\
         data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n\
         data: [DONE]\n\n",
    )])
    .await;

    let (response, events) = stream_events(&model).await;

    assert!(events.is_empty(), "tool arguments are not reply text");
    assert_eq!(response.content, None);
    assert_eq!(response.tool_calls.len(), 1);
    assert_eq!(response.tool_calls[0].id, "call_1");
    assert_eq!(response.tool_calls[0].name, "bash");
    assert_eq!(
        response.tool_calls[0].arguments,
        serde_json::json!({"command": "cargo test"})
    );
}

#[tokio::test]
async fn falls_back_for_a_json_response() {
    let (model, _) = model_replying(vec![Reply::json(
        "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"whole reply\"}}]}",
    )])
    .await;

    let (response, events) = stream_events(&model).await;

    assert_eq!(response.content.as_deref(), Some("whole reply"));
    assert_eq!(
        events,
        [
            ModelEvent::Fallback,
            ModelEvent::Text("whole reply".to_string())
        ]
    );
}

#[tokio::test]
async fn falls_back_when_the_body_is_not_sse() {
    // The endpoint claims a stream but answers with plain JSON.
    let (model, _) = model_replying(vec![Reply {
        status: "200 OK",
        headers: "Content-Type: text/event-stream\r\n",
        body: "{\"choices\":[{\"message\":{\"content\":\"sneaky\"}}]}",
    }])
    .await;

    let (response, events) = stream_events(&model).await;

    assert_eq!(response.content.as_deref(), Some("sneaky"));
    assert_eq!(
        events,
        [ModelEvent::Fallback, ModelEvent::Text("sneaky".to_string())]
    );
}

#[tokio::test]
async fn falls_back_when_the_body_lacks_a_content_type() {
    let (model, _) = model_replying(vec![Reply {
        status: "200 OK",
        headers: "",
        body: "{\"choices\":[{\"message\":{\"content\":\"sniffed\"}}]}",
    }])
    .await;

    let (response, events) = stream_events(&model).await;

    assert_eq!(response.content.as_deref(), Some("sniffed"));
    assert_eq!(
        events,
        [
            ModelEvent::Fallback,
            ModelEvent::Text("sniffed".to_string())
        ]
    );
}

#[tokio::test]
async fn retries_without_stream_after_a_shape_rejection() {
    let (model, served) = model_replying(vec![
        Reply {
            status: "400 Bad Request",
            headers: "Content-Type: application/json\r\n",
            body: "{\"error\":{\"message\":\"unknown field 'stream'\"}}",
        },
        Reply::json("{\"choices\":[{\"message\":{\"content\":\"retried\"}}]}"),
    ])
    .await;

    let (response, events) = stream_events(&model).await;

    assert_eq!(served.load(Ordering::SeqCst), 2);
    assert_eq!(response.content.as_deref(), Some("retried"));
    assert_eq!(
        events,
        [
            ModelEvent::Fallback,
            ModelEvent::Text("retried".to_string())
        ]
    );
}

#[tokio::test]
async fn does_not_retry_an_auth_failure() {
    let (model, served) = model_replying(vec![Reply {
        status: "401 Unauthorized",
        headers: "Content-Type: application/json\r\n",
        body: "{\"error\":{\"message\":\"bad key\"}}",
    }])
    .await;

    let error = model
        .complete_stream(&[], &[], &mut |_| {})
        .await
        .unwrap_err();

    assert!(format!("{error:#}").contains("bad key"), "{error:#}");
    assert_eq!(served.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn reports_an_error_frame() {
    let (model, _) = model_replying(vec![Reply::sse(
        "data: {\"error\":{\"message\":\"overloaded\"}}\n\n",
    )])
    .await;

    let error = model
        .complete_stream(&[], &[], &mut |_| {})
        .await
        .unwrap_err();

    assert!(format!("{error:#}").contains("overloaded"), "{error:#}");
}

#[tokio::test]
async fn accepts_a_stream_that_ends_without_done() {
    let (model, _) = model_replying(vec![Reply::sse(
        "data: {\"choices\":[{\"delta\":{\"content\":\"cut\"}}]}\n\n",
    )])
    .await;

    let (response, events) = stream_events(&model).await;

    assert_eq!(response.content.as_deref(), Some("cut"));
    assert_eq!(events, [ModelEvent::Text("cut".to_string())]);
}

#[tokio::test]
async fn asks_for_an_event_stream() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base_url = format!("http://{}/v1", listener.local_addr().unwrap());
    let (sent, received) = tokio::sync::oneshot::channel();

    tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let request = read_request(&mut stream).await;
        write_response(
            &mut stream,
            Reply::sse("data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n"),
        )
        .await;
        let _ = sent.send(request);
    });

    let model = OpenAIModel::new("test-key", &base_url, "test-model");
    model.complete_stream(&[], &[], &mut |_| {}).await.unwrap();

    let request = received.await.unwrap().to_ascii_lowercase();
    assert!(
        request.starts_with("post /v1/chat/completions"),
        "{request}"
    );
    assert!(request.contains("accept: text/event-stream"), "{request}");
    assert!(request.contains("\"stream\":true"), "{request}");
}

#[tokio::test]
async fn reports_tool_definitions_and_history() {
    use protocol::ToolDefinition;

    let (model, _) = model_replying(vec![Reply::sse(
        "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n",
    )])
    .await;

    let history = vec![
        Item::UserMessage("hi".to_string()),
        Item::AssistantMessage("hello".to_string()),
    ];
    let tools = vec![ToolDefinition {
        name: "bash".to_string(),
        description: "run a shell command".to_string(),
        parameters: serde_json::json!({"type": "object"}),
    }];

    let response = model
        .complete_stream(&history, &tools, &mut |_| {})
        .await
        .unwrap();

    assert_eq!(response.content.as_deref(), Some("ok"));
}
