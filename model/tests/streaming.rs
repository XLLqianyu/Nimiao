//! Streaming against a real socket.
//!
//! Each test starts a hand-rolled HTTP/1.1 server on a loopback port and points
//! an [`OpenAIModel`] at it. No HTTP test crate is needed; the server reads one
//! request and answers it.

use model::openai::OpenAIModel;
use model::{Model, ModelEvent, ModelResponse, StopReason};
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
        OpenAIModel::new("test-key", &base_url, "test-model").unwrap(),
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

/// Writes a streamed body with `Transfer-Encoding: chunked`, one write per
/// piece, so frames reach the client in separate reads.
async fn write_chunked(stream: &mut TcpStream, pieces: &[&str], close: bool) {
    let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
                Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n";
    stream.write_all(head.as_bytes()).await.unwrap();
    for piece in pieces {
        let chunk = format!("{:X}\r\n{piece}\r\n", piece.len());
        stream.write_all(chunk.as_bytes()).await.unwrap();
        stream.flush().await.unwrap();
    }
    stream.write_all(b"0\r\n\r\n").await.unwrap();
    if close {
        stream.shutdown().await.unwrap();
    }
}

/// Spawns a server that hands every request to `handle`.
async fn serve<F, Fut>(handle: F) -> String
where
    F: Fn(TcpStream) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = ()> + Send,
{
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base_url = format!("http://{}/v1", listener.local_addr().unwrap());
    let handle = Arc::new(handle);
    tokio::spawn(async move {
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                return;
            };
            let handle = Arc::clone(&handle);
            tokio::spawn(async move {
                read_request(&mut stream).await;
                handle(stream).await;
            });
        }
    });
    base_url
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

/// Sends one completion and returns the request body the client built.
///
/// The body is parsed rather than matched as text, so the assertions below see
/// the message structure rather than how `serde_json` happened to space it.
async fn sent_body(history: &[Item], tools: &[protocol::ToolDefinition]) -> serde_json::Value {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base_url = format!("http://{}/v1", listener.local_addr().unwrap());
    let (sent, received) = tokio::sync::oneshot::channel();

    tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let request = read_request(&mut stream).await;
        write_response(&mut stream, Reply::sse("data: [DONE]\n\n")).await;
        let _ = sent.send(request);
    });

    let model = OpenAIModel::new("test-key", &base_url, "test-model").unwrap();
    model
        .complete_stream(history, tools, &mut |_| {})
        .await
        .unwrap();

    let request = received.await.unwrap();
    let body = request
        .split_once("\r\n\r\n")
        .expect("the request must carry a body")
        .1;
    serde_json::from_str(body).expect("the request body must be JSON")
}

fn call(id: &str, name: &str, arguments: serde_json::Value) -> Item {
    Item::ToolCall(protocol::ToolCall::new(id, name, arguments))
}

#[tokio::test]
async fn sends_text_and_a_call_as_one_assistant_message() {
    let history = vec![
        Item::UserMessage("hi".to_string()),
        Item::AssistantMessage("running it. ".to_string()),
        call("call_1", "bash", serde_json::json!({"command": "ls"})),
        Item::ToolResult {
            tool_call_id: "call_1".to_string(),
            content: "a.txt".to_string(),
            is_error: false,
        },
    ];

    let body = sent_body(&history, &[]).await;
    let messages = body["messages"].as_array().unwrap();

    assert_eq!(messages.len(), 3, "{messages:#?}");
    assert_eq!(messages[1]["role"], "assistant");
    assert_eq!(messages[1]["content"], "running it. ");
    let calls = messages[1]["tool_calls"].as_array().unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["id"], "call_1");
    assert_eq!(calls[0]["type"], "function");
    assert_eq!(calls[0]["function"]["name"], "bash");
    assert_eq!(messages[2]["role"], "tool");
    assert_eq!(messages[2]["tool_call_id"], "call_1");
}

#[tokio::test]
async fn sends_parallel_calls_in_one_assistant_message() {
    let history = vec![
        Item::UserMessage("hi".to_string()),
        call("call_a", "read_file", serde_json::json!({"path": "a"})),
        call("call_b", "read_file", serde_json::json!({"path": "b"})),
        Item::ToolResult {
            tool_call_id: "call_a".to_string(),
            content: "a".to_string(),
            is_error: false,
        },
        Item::ToolResult {
            tool_call_id: "call_b".to_string(),
            content: "b".to_string(),
            is_error: false,
        },
    ];

    let body = sent_body(&history, &[]).await;
    let messages = body["messages"].as_array().unwrap();

    assert_eq!(messages.len(), 4, "{messages:#?}");
    let assistants: Vec<_> = messages
        .iter()
        .filter(|message| message["role"] == "assistant")
        .collect();
    assert_eq!(assistants.len(), 1, "{messages:#?}");
    assert!(assistants[0]["content"].is_null(), "{messages:#?}");
    let calls = assistants[0]["tool_calls"].as_array().unwrap();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0]["id"], "call_a");
    assert_eq!(calls[1]["id"], "call_b");
}

#[tokio::test]
async fn keeps_text_with_parallel_calls_in_the_same_message() {
    let history = vec![
        Item::UserMessage("hi".to_string()),
        Item::AssistantMessage("looking. ".to_string()),
        call("call_a", "bash", serde_json::json!({"command": "one"})),
        call("call_b", "bash", serde_json::json!({"command": "two"})),
    ];

    let body = sent_body(&history, &[]).await;
    let messages = body["messages"].as_array().unwrap();

    assert_eq!(messages.len(), 2, "{messages:#?}");
    assert_eq!(messages[1]["content"], "looking. ");
    assert_eq!(messages[1]["tool_calls"].as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn keeps_an_empty_call_id_empty() {
    let history = vec![
        Item::UserMessage("hi".to_string()),
        call("", "bash", serde_json::json!({"command": "ls"})),
    ];

    let body = sent_body(&history, &[]).await;
    let messages = body["messages"].as_array().unwrap();

    // Fabricating an id would break the `tool_call_id` that answers it.
    assert_eq!(messages[1]["tool_calls"][0]["id"], "");
}

#[tokio::test]
async fn drops_an_assistant_message_with_nothing_in_it() {
    let history = vec![
        Item::UserMessage("hi".to_string()),
        Item::AssistantMessage("   ".to_string()),
        Item::AssistantMessage("ok".to_string()),
    ];

    let body = sent_body(&history, &[]).await;
    let messages = body["messages"].as_array().unwrap();

    assert_eq!(messages.len(), 2, "{messages:#?}");
    assert_eq!(messages[1]["role"], "assistant");
    assert_eq!(messages[1]["content"], "ok");
    assert!(messages[1].get("tool_calls").is_none());
}

#[tokio::test]
async fn sends_tools_beside_the_messages() {
    let tools = vec![protocol::ToolDefinition {
        name: "bash".to_string(),
        description: "run a shell command".to_string(),
        parameters: serde_json::json!({"type": "object"}),
    }];

    let body = sent_body(&[Item::UserMessage("hi".to_string())], &tools).await;

    assert_eq!(body["tools"][0]["function"]["name"], "bash");
    assert_eq!(body["messages"].as_array().unwrap().len(), 1);
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

    let model = OpenAIModel::new("test-key", &base_url, "test-model").unwrap();
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

#[tokio::test]
async fn reads_a_chunked_stream_piece_by_piece() {
    let base_url = serve(|mut stream| async move {
        write_chunked(
            &mut stream,
            &[
                "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n",
                "data: {\"choices\":[{\"delta\":{\"content\":\"b\"}}]}",
                "\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"c\"}}]}\n\n",
                "data: [DONE]\n\n",
            ],
            true,
        )
        .await;
    })
    .await;

    let model = OpenAIModel::new("test-key", &base_url, "test-model").unwrap();
    let (response, events) = stream_events(&model).await;

    assert_eq!(response.content.as_deref(), Some("abc"));
    assert_eq!(
        events,
        [
            ModelEvent::Text("a".to_string()),
            ModelEvent::Text("b".to_string()),
            ModelEvent::Text("c".to_string())
        ]
    );
}

#[tokio::test]
async fn returns_as_soon_as_done_arrives() {
    // The connection stays open after `[DONE]`, as a keep-alive proxy might
    // leave it. The call must not wait for a close that never comes.
    let base_url = serve(|mut stream| async move {
        let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
                    Transfer-Encoding: chunked\r\n\r\n";
        stream.write_all(head.as_bytes()).await.unwrap();
        let body = "data: {\"choices\":[{\"delta\":{\"content\":\"early\"}}]}\n\ndata: [DONE]\n\n";
        let chunk = format!("{:X}\r\n{body}\r\n", body.len());
        stream.write_all(chunk.as_bytes()).await.unwrap();
        stream.flush().await.unwrap();
        // Hold the connection open well past the assertion below.
        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
    })
    .await;

    let model = OpenAIModel::new("test-key", &base_url, "test-model").unwrap();
    let completed = tokio::time::timeout(std::time::Duration::from_secs(2), async {
        stream_events(&model).await
    })
    .await
    .expect("the call waited for the connection to close after [DONE]");

    assert_eq!(completed.0.content.as_deref(), Some("early"));
}

#[tokio::test]
async fn streams_parallel_tool_calls_and_keeps_index_order() {
    let (model, _) = model_replying(vec![Reply::sse(
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":1,\"id\":\"call_b\",\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\"}}]}}]}\n\n\
         data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_a\",\"function\":{\"name\":\"bash\",\"arguments\":\"{\\\"command\\\":\"}}]}}]}\n\n\
         data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":1,\"function\":{\"arguments\":\"\\\"b.txt\\\"}\"}}]}}]}\n\n\
         data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"ls\\\"}\"}}]}}]}\n\n\
         data: [DONE]\n\n",
    )])
    .await;

    let (response, _) = stream_events(&model).await;

    let calls = response.tool_calls;
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].id, "call_a");
    assert_eq!(calls[0].name, "bash");
    assert_eq!(calls[0].arguments, serde_json::json!({"command": "ls"}));
    assert_eq!(calls[1].id, "call_b");
    assert_eq!(calls[1].name, "read_file");
    assert_eq!(calls[1].arguments, serde_json::json!({"path": "b.txt"}));
}

#[tokio::test]
async fn reports_the_status_when_the_retry_also_fails() {
    let (model, served) = model_replying(vec![
        Reply {
            status: "400 Bad Request",
            headers: "Content-Type: application/json\r\n",
            body: "{\"error\":{\"message\":\"bad stream\"}}",
        },
        Reply {
            status: "500 Internal Server Error",
            headers: "Content-Type: application/json\r\n",
            body: "{\"error\":{\"message\":\"still broken\"}}",
        },
    ])
    .await;

    let error = model
        .complete_stream(&[], &[], &mut |_| {})
        .await
        .unwrap_err();
    let message = format!("{error:#}");

    assert_eq!(served.load(Ordering::SeqCst), 2, "the retry must happen");
    assert!(message.contains("500"), "{message}");
    assert!(message.contains("still broken"), "{message}");
}

#[tokio::test]
async fn retries_after_a_422_rejection() {
    let (model, served) = model_replying(vec![
        Reply {
            status: "422 Unprocessable Entity",
            headers: "Content-Type: application/json\r\n",
            body: "{\"error\":{\"message\":\"stream not supported\"}}",
        },
        Reply::json("{\"choices\":[{\"message\":{\"content\":\"fine\"}}]}"),
    ])
    .await;

    let (response, _) = stream_events(&model).await;

    assert_eq!(served.load(Ordering::SeqCst), 2);
    assert_eq!(response.content.as_deref(), Some("fine"));
}

#[tokio::test]
async fn does_not_retry_a_server_error() {
    let (model, served) = model_replying(vec![Reply {
        status: "503 Service Unavailable",
        headers: "Content-Type: text/plain\r\n",
        body: "try later",
    }])
    .await;

    let error = model
        .complete_stream(&[], &[], &mut |_| {})
        .await
        .unwrap_err();

    assert_eq!(served.load(Ordering::SeqCst), 1);
    assert!(format!("{error:#}").contains("503"), "{error:#}");
}

#[tokio::test]
async fn treats_a_content_type_parameter_as_a_stream() {
    let (model, _) = model_replying(vec![Reply {
        status: "200 OK",
        headers: "Content-Type: text/event-stream; charset=utf-8\r\n",
        body: "data: {\"choices\":[{\"delta\":{\"content\":\"utf8\"}}]}\n\ndata: [DONE]\n\n",
    }])
    .await;

    let (response, events) = stream_events(&model).await;

    assert_eq!(response.content.as_deref(), Some("utf8"));
    assert_eq!(events, [ModelEvent::Text("utf8".to_string())]);
}

#[tokio::test]
async fn treats_a_mixed_case_content_type_as_a_stream() {
    let (model, _) = model_replying(vec![Reply {
        status: "200 OK",
        headers: "Content-Type: Text/Event-Stream\r\n",
        body: "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n",
    }])
    .await;

    let (response, _) = stream_events(&model).await;

    assert_eq!(response.content.as_deref(), Some("ok"));
}

#[tokio::test]
async fn returns_an_empty_completion_for_an_empty_stream() {
    let (model, _) = model_replying(vec![Reply::sse("data: [DONE]\n\n")]).await;

    let (response, events) = stream_events(&model).await;

    assert_eq!(response.content, None);
    assert!(response.tool_calls.is_empty());
    assert!(events.is_empty());
}

#[tokio::test]
async fn reports_a_body_that_is_neither_stream_nor_json() {
    let (model, _) = model_replying(vec![Reply::json("<html>proxy error</html>")]).await;

    let error = model
        .complete_stream(&[], &[], &mut |_| {})
        .await
        .unwrap_err();
    let message = format!("{error:#}");

    assert!(message.contains("neither"), "{message}");
    assert!(message.contains("proxy error"), "{message}");
}

#[tokio::test]
async fn skips_frames_that_are_not_json_and_keeps_streaming() {
    let (model, _) = model_replying(vec![Reply::sse(
        "data: ping\n\n\
         data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n\
         data: [DONE]\n\n",
    )])
    .await;

    let (response, events) = stream_events(&model).await;

    assert_eq!(response.content.as_deref(), Some("ok"));
    assert_eq!(events, [ModelEvent::Text("ok".to_string())]);
}

#[tokio::test]
async fn ignores_a_delta_without_choices() {
    let (model, _) = model_replying(vec![Reply::sse(
        "data: {\"choices\":[],\"usage\":{\"total_tokens\":9}}\n\n\
         data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n\
         data: [DONE]\n\n",
    )])
    .await;

    let (response, events) = stream_events(&model).await;

    assert_eq!(response.content.as_deref(), Some("x"));
    assert_eq!(events, [ModelEvent::Text("x".to_string())]);
}

#[tokio::test]
async fn reports_a_length_finish_reason_from_a_stream() {
    let (model, _) = model_replying(vec![Reply::sse(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"cut\"}}]}\n\n\
         data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"length\"}]}\n\n\
         data: [DONE]\n\n",
    )])
    .await;

    let (response, events) = stream_events(&model).await;

    assert_eq!(response.stop_reason, Some(StopReason::Length));
    assert_eq!(events, [ModelEvent::Text("cut".to_string())]);
}

#[tokio::test]
async fn reports_a_content_filter_from_a_stream() {
    let (model, _) = model_replying(vec![Reply::sse(
        "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"content_filter\"}]}\n\n\
         data: [DONE]\n\n",
    )])
    .await;

    let (response, _) = stream_events(&model).await;

    assert_eq!(response.stop_reason, Some(StopReason::ContentFilter));
}

#[tokio::test]
async fn reports_a_tool_calls_finish_reason() {
    let (model, _) = model_replying(vec![Reply::sse(
        "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n\
         data: [DONE]\n\n",
    )])
    .await;

    let (response, _) = stream_events(&model).await;

    assert_eq!(response.stop_reason, Some(StopReason::ToolCalls));
}

#[tokio::test]
async fn leaves_the_stop_reason_unset_when_the_provider_omits_it() {
    let (model, _) = model_replying(vec![Reply::sse(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"}}]}\n\n\
         data: [DONE]\n\n",
    )])
    .await;

    let (response, _) = stream_events(&model).await;

    // Proxies commonly omit it; that must stay valid rather than fail.
    assert_eq!(response.stop_reason, None);
}

#[tokio::test]
async fn reports_a_finish_reason_from_a_whole_completion() {
    let (model, _) = model_replying(vec![Reply::json(
        "{\"choices\":[{\"message\":{\"content\":\"partial\"},\"finish_reason\":\"length\"}]}",
    )])
    .await;

    let (response, _) = stream_events(&model).await;

    assert_eq!(response.content.as_deref(), Some("partial"));
    assert_eq!(response.stop_reason, Some(StopReason::Length));
}

#[tokio::test]
async fn keeps_an_unknown_finish_reason_as_other() {
    let (model, _) = model_replying(vec![Reply::sse(
        "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"network_error\"}]}\n\n\
         data: [DONE]\n\n",
    )])
    .await;

    let (response, _) = stream_events(&model).await;

    assert_eq!(
        response.stop_reason,
        Some(StopReason::Other("network_error".to_string()))
    );
}

#[tokio::test]
async fn keeps_a_finished_tool_call_whose_arguments_are_not_json() {
    // The call is kept so the history stays answerable, but `{}` alone would
    // hide what the model actually said. The text it sent is kept beside it.
    let (model, _) = model_replying(vec![Reply::sse(
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"bash\",\"arguments\":\"not json\"}}]}}]}\n\n\
         data: [DONE]\n\n",
    )])
    .await;

    let (response, _) = stream_events(&model).await;

    assert_eq!(response.tool_calls.len(), 1);
    assert_eq!(response.tool_calls[0].name, "bash");
    assert_eq!(response.tool_calls[0].arguments, serde_json::json!({}));
    let error = response.tool_calls[0]
        .arguments_error
        .as_deref()
        .expect("the malformed text must be kept");
    assert!(error.contains("Received arguments:\nnot json"), "{error}");
}

#[tokio::test]
async fn keeps_a_finished_tool_call_whose_arguments_are_not_an_object() {
    // Valid JSON that is not an object cannot be replayed either: providers
    // require `arguments` to be a JSON object.
    let (model, _) = model_replying(vec![Reply::sse(
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"bash\",\"arguments\":\"[1,2]\"}}]}}]}\n\n\
         data: [DONE]\n\n",
    )])
    .await;

    let (response, _) = stream_events(&model).await;

    assert_eq!(response.tool_calls[0].arguments, serde_json::json!({}));
    let error = response.tool_calls[0].arguments_error.as_deref().unwrap();
    assert!(error.contains("must be a JSON object"), "{error}");
    assert!(error.contains("[1,2]"), "{error}");
}

#[tokio::test]
async fn accepts_a_call_with_no_arguments_text() {
    let (model, _) = model_replying(vec![Reply::sse(
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"bash\",\"arguments\":\"\"}}]}}]}\n\n\
         data: [DONE]\n\n",
    )])
    .await;

    let (response, _) = stream_events(&model).await;

    assert_eq!(response.tool_calls[0].arguments, serde_json::json!({}));
    assert!(response.tool_calls[0].arguments_error.is_none());
}

#[tokio::test]
async fn keeps_malformed_arguments_from_a_whole_completion() {
    let (model, _) = model_replying(vec![Reply::json(
        "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"tool_calls\":[\
           {\"id\":\"call_1\",\"function\":{\"name\":\"bash\",\"arguments\":\"{oops\"}}]}}]}",
    )])
    .await;

    let (response, _) = stream_events(&model).await;

    assert_eq!(response.tool_calls.len(), 1);
    assert_eq!(response.tool_calls[0].arguments, serde_json::json!({}));
    assert!(response.tool_calls[0]
        .arguments_error
        .as_deref()
        .unwrap()
        .contains("{oops"));
}

#[tokio::test]
async fn does_not_send_the_argument_error_on_the_wire() {
    let mut malformed = protocol::ToolCall::new("call_1", "bash", serde_json::json!({}));
    malformed.arguments_error = Some("received garbage".to_string());
    let history = vec![
        Item::UserMessage("hi".to_string()),
        Item::ToolCall(malformed),
    ];

    let body = sent_body(&history, &[]).await;
    let message = &body["messages"][1];

    assert_eq!(message["tool_calls"][0]["function"]["arguments"], "{}");
    assert!(message["tool_calls"][0].get("arguments_error").is_none());
}

#[tokio::test]
async fn gives_up_on_a_stalled_endpoint() {
    // The server accepts the request and then says nothing. Without a read
    // timeout the turn would wait forever.
    let base_url = serve(|mut stream| async move {
        // Hold the response open without sending anything: the request was
        // accepted, so only a read timeout can end the wait.
        tokio::time::sleep(std::time::Duration::from_secs(30)).await;
        let _ = stream.shutdown().await;
    })
    .await;

    let model = OpenAIModel::with_timeouts(
        "test-key",
        &base_url,
        "test-model",
        std::time::Duration::from_secs(5),
        std::time::Duration::from_millis(200),
    )
    .unwrap();

    let error = tokio::time::timeout(std::time::Duration::from_secs(10), async {
        model.complete_stream(&[], &[], &mut |_| {}).await
    })
    .await
    .expect("the read timeout should have fired")
    .expect_err("a silent endpoint is an error, not an empty reply");
    assert!(format!("{error:#}").contains("timed out"), "{error:#}");
}
