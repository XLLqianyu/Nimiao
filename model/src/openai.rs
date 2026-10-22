use crate::sse::SseDecoder;
use crate::{Model, ModelEvent, ModelResponse, ToolDefinition};
use async_trait::async_trait;
use protocol::{Item, ToolCall};
use reqwest::header::{ACCEPT, CONTENT_TYPE};
use serde_json::{json, Value};
use std::collections::BTreeMap;

/// Upper bound on bytes held while checking whether a claimed stream is one.
/// A whole JSON completion is well below this.
const MAX_SNIFF_BYTES: usize = 8 * 1024 * 1024;

pub struct OpenAIModel {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    model_name: String,
}

/// What reading an event stream produced.
#[derive(Debug)]
enum StreamOutcome {
    /// Deltas were applied and the completion is assembled.
    Streamed(ModelResponse),
    /// One frame carried a whole completion, as some proxies echo.
    WholeFrame(ModelResponse),
    /// The body is not an event stream after all; these are its bytes so far.
    NotSse(Vec<u8>),
}

impl OpenAIModel {
    pub fn new(api_key: &str, base_url: &str, model_name: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.to_string(),
            base_url: base_url.to_string(),
            model_name: model_name.to_string(),
        }
    }

    fn format_messages(history: &[Item]) -> Vec<serde_json::Value> {
        let mut messages = vec![];
        for item in history {
            match item {
                Item::UserMessage(t) => messages.push(json!({"role": "user", "content": t})),
                Item::AssistantMessage(t) => {
                    messages.push(json!({"role": "assistant", "content": t}))
                }
                Item::ToolCall(c) => {
                    messages.push(json!({
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": c.id,
                            "type": "function",
                            "function": {
                                "name": c.name,
                                "arguments": c.arguments.to_string()
                            }
                        }]
                    }));
                }
                Item::ToolResult {
                    tool_call_id,
                    content,
                    ..
                } => {
                    messages.push(json!({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": content
                    }));
                }
            }
        }
        messages
    }

    fn request_body(
        &self,
        history: &[Item],
        tools: &[ToolDefinition],
        stream: bool,
    ) -> serde_json::Value {
        let mut body = json!({
            "model": self.model_name,
            "messages": Self::format_messages(history),
        });

        if !tools.is_empty() {
            let openai_tools: Vec<_> = tools
                .iter()
                .map(|t| {
                    json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters
                        }
                    })
                })
                .collect();
            body["tools"] = json!(openai_tools);
        }

        if stream {
            body["stream"] = json!(true);
        }

        body
    }

    fn parse_completion(json_res: &serde_json::Value) -> anyhow::Result<ModelResponse> {
        let choice = json_res["choices"]
            .get(0)
            .ok_or_else(|| anyhow::anyhow!("Empty choices"))?;
        let message = &choice["message"];

        let content = message["content"].as_str().map(|s| s.to_string());
        let mut tool_calls = vec![];

        if let Some(tcs) = message["tool_calls"].as_array() {
            for tc in tcs {
                tool_calls.push(ToolCall {
                    id: tc["id"].as_str().unwrap_or_default().to_string(),
                    name: tc["function"]["name"]
                        .as_str()
                        .unwrap_or_default()
                        .to_string(),
                    arguments: serde_json::from_str(
                        tc["function"]["arguments"].as_str().unwrap_or("{}"),
                    )
                    .unwrap_or(json!({})),
                });
            }
        }

        Ok(ModelResponse {
            content,
            tool_calls,
        })
    }

    async fn send(
        &self,
        history: &[Item],
        tools: &[ToolDefinition],
        stream: bool,
    ) -> reqwest::Result<reqwest::Response> {
        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
        let mut request = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&self.request_body(history, tools, stream));
        if stream {
            request = request.header(ACCEPT, "text/event-stream");
        }
        request.send().await
    }

    /// Reads a response that was not an event stream as one JSON completion.
    fn parse_body(bytes: &[u8]) -> anyhow::Result<ModelResponse> {
        let value: serde_json::Value = serde_json::from_slice(bytes).map_err(|_| {
            let text = String::from_utf8_lossy(bytes);
            let mut chars = text.chars();
            let mut head: String = chars.by_ref().take(200).collect();
            if chars.next().is_some() {
                head.push('…');
            }
            anyhow::anyhow!("the endpoint returned neither an event stream nor JSON: {head}")
        })?;
        if let Some(error) = value.get("error") {
            anyhow::bail!("API Error: {error}");
        }
        Self::parse_completion(&value)
    }
}

#[async_trait]
impl Model for OpenAIModel {
    fn name(&self) -> &str {
        &self.model_name
    }

    async fn complete(
        &self,
        history: &[Item],
        tools: &[ToolDefinition],
    ) -> anyhow::Result<ModelResponse> {
        let res = self.send(history, tools, false).await?;

        if !res.status().is_success() {
            anyhow::bail!("API Error: {}", res.text().await?);
        }

        let json_res: serde_json::Value = res.json().await?;
        Self::parse_completion(&json_res)
    }

    async fn complete_stream(
        &self,
        history: &[Item],
        tools: &[ToolDefinition],
        on_event: &mut (dyn FnMut(ModelEvent) + Send),
    ) -> anyhow::Result<ModelResponse> {
        let mut res = self.send(history, tools, true).await?;

        let status = res.status();
        if !status.is_success() {
            let body = res.text().await.unwrap_or_default();
            // Only a shape rejection is worth retrying without `stream`.
            // Retrying an auth or rate-limit failure would just repeat it.
            if status.as_u16() != 400 && status.as_u16() != 422 {
                anyhow::bail!("API Error {status}: {body}");
            }
            let mut retried = self.send(history, tools, false).await?;
            if !retried.status().is_success() {
                let status = retried.status();
                let body = retried.text().await.unwrap_or_default();
                anyhow::bail!("API Error {status}: {body}");
            }
            return fall_back(&mut retried, Vec::new(), on_event).await;
        }

        let content_type = res
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .map(|value| value.trim_start().to_ascii_lowercase());
        let streams = content_type
            .as_deref()
            .is_none_or(|value| value.starts_with("text/event-stream"));
        if !streams {
            return fall_back(&mut res, Vec::new(), on_event).await;
        }

        match consume_stream(&mut res, on_event).await? {
            StreamOutcome::Streamed(response) | StreamOutcome::WholeFrame(response) => Ok(response),
            StreamOutcome::NotSse(prefix) => fall_back(&mut res, prefix, on_event).await,
        }
    }
}

/// Reads a response that was not a stream as one whole completion.
///
/// Says the endpoint fell back only once the body has parsed, so a broken
/// response reports its error without announcing a fallback first.
async fn fall_back(
    res: &mut reqwest::Response,
    bytes: Vec<u8>,
    on_event: &mut (dyn FnMut(ModelEvent) + Send),
) -> anyhow::Result<ModelResponse> {
    let bytes = read_body(res, bytes).await?;
    let response = OpenAIModel::parse_body(&bytes)?;
    on_event(ModelEvent::Fallback);
    emit_text(&response, on_event);
    Ok(response)
}

/// Drains the rest of a response body, keeping `bytes` as the start of it.
async fn read_body(res: &mut reqwest::Response, mut bytes: Vec<u8>) -> anyhow::Result<Vec<u8>> {
    while let Some(chunk) = res.chunk().await? {
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}

fn emit_text(response: &ModelResponse, on_event: &mut (dyn FnMut(ModelEvent) + Send)) {
    if let Some(text) = response.content.as_deref().filter(|text| !text.is_empty()) {
        on_event(ModelEvent::Text(text.to_string()));
    }
}

/// Reads an event stream to its end, applying frames as they arrive.
async fn consume_stream(
    res: &mut reqwest::Response,
    on_event: &mut (dyn FnMut(ModelEvent) + Send),
) -> anyhow::Result<StreamOutcome> {
    let mut decoder = SseDecoder::new();
    let mut state = StreamState::new();

    while let Some(chunk) = res.chunk().await? {
        if state.sniffing {
            state.raw.extend_from_slice(&chunk);
            if state.raw.len() > MAX_SNIFF_BYTES {
                anyhow::bail!(
                    "the endpoint claimed an event stream but sent {} bytes without a data frame",
                    state.raw.len()
                );
            }
        }
        for frame in decoder.feed(&chunk)? {
            // `[DONE]` ends the completion. Not every endpoint closes the
            // response after it, so stop here rather than wait for EOF.
            if frame.trim() == "[DONE]" {
                return Ok(StreamOutcome::Streamed(state.finish()));
            }
            if let Some(outcome) = state.apply(&frame, on_event)? {
                return Ok(outcome);
            }
        }
    }
    if let Some(frame) = decoder.finish() {
        if frame.trim() != "[DONE]" {
            if let Some(outcome) = state.apply(&frame, on_event)? {
                return Ok(outcome);
            }
        }
    }

    if state.sniffing {
        return Ok(StreamOutcome::NotSse(state.raw));
    }
    Ok(StreamOutcome::Streamed(state.finish()))
}

struct StreamState {
    /// Set until the first frame proves the body really is SSE.
    sniffing: bool,
    /// Bytes read so far, kept only while sniffing for a non-SSE fallback.
    raw: Vec<u8>,
    text: String,
    calls: ToolCallAccumulator,
}

impl StreamState {
    fn new() -> Self {
        Self {
            sniffing: true,
            raw: Vec::new(),
            text: String::new(),
            calls: ToolCallAccumulator::default(),
        }
    }

    fn apply(
        &mut self,
        frame: &str,
        on_event: &mut (dyn FnMut(ModelEvent) + Send),
    ) -> anyhow::Result<Option<StreamOutcome>> {
        let value = parse_object(frame);
        if let Some(error) = value.as_ref().and_then(|value| value.get("error")) {
            anyhow::bail!("API Error: {error}");
        }

        if self.sniffing {
            match value {
                Some(value) => {
                    self.sniffing = false;
                    self.raw.clear();
                    if let Some(response) = whole_completion(&value)? {
                        on_event(ModelEvent::Fallback);
                        emit_text(&response, on_event);
                        return Ok(Some(StreamOutcome::WholeFrame(response)));
                    }
                    self.apply_delta(&value, on_event);
                }
                // A junk frame amid valid SSE is skipped, but a body that
                // never produced a JSON frame is not a stream at all.
                None => return Ok(None),
            }
            return Ok(None);
        }

        if let Some(value) = value.filter(|frame| frame.get("choices").is_some()) {
            self.apply_delta(&value, on_event);
        }
        Ok(None)
    }

    fn apply_delta(&mut self, value: &Value, on_event: &mut (dyn FnMut(ModelEvent) + Send)) {
        let Some(choice) = value["choices"].get(0) else {
            return;
        };
        let delta = &choice["delta"];
        if let Some(content) = delta["content"].as_str() {
            if !content.is_empty() {
                self.text.push_str(content);
                on_event(ModelEvent::Text(content.to_string()));
            }
        }
        if let Some(calls) = delta["tool_calls"].as_array() {
            for call in calls {
                self.calls.push(call);
            }
        }
    }

    fn finish(self) -> ModelResponse {
        let content = if self.text.is_empty() {
            None
        } else {
            Some(self.text)
        };
        ModelResponse {
            content,
            tool_calls: self.calls.finish(),
        }
    }
}

fn parse_object(frame: &str) -> Option<Value> {
    serde_json::from_str::<Value>(frame)
        .ok()
        .filter(Value::is_object)
}

/// Recognizes a frame that holds a finished completion instead of a delta.
fn whole_completion(value: &Value) -> anyhow::Result<Option<ModelResponse>> {
    let first = &value["choices"][0];
    if first.get("message").is_some() && first.get("delta").is_none() {
        return Ok(Some(OpenAIModel::parse_completion(value)?));
    }
    Ok(None)
}

/// Collects `delta.tool_calls` fragments, which arrive split across frames.
#[derive(Debug, Default)]
struct ToolCallAccumulator {
    calls: BTreeMap<u64, PartialToolCall>,
}

#[derive(Debug, Default)]
struct PartialToolCall {
    id: String,
    name: String,
    arguments: String,
}

impl ToolCallAccumulator {
    fn push(&mut self, fragment: &Value) {
        let index = fragment["index"].as_u64().unwrap_or(0);
        let call = self.calls.entry(index).or_default();

        if call.id.is_empty() {
            if let Some(id) = fragment["id"].as_str().filter(|id| !id.is_empty()) {
                call.id = id.to_string();
            }
        }
        if call.name.is_empty() {
            if let Some(name) = fragment["function"]["name"]
                .as_str()
                .filter(|name| !name.is_empty())
            {
                call.name = name.to_string();
            }
        }
        if let Some(arguments) = fragment["function"]["arguments"].as_str() {
            call.arguments.push_str(arguments);
        }
    }

    fn finish(self) -> Vec<ToolCall> {
        self.calls
            .into_values()
            .map(|call| ToolCall {
                id: call.id,
                name: call.name,
                arguments: serde_json::from_str(&call.arguments).unwrap_or(json!({})),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apply(state: &mut StreamState, frame: &str, events: &mut Vec<ModelEvent>) {
        state.apply(frame, &mut |event| events.push(event)).unwrap();
    }

    #[test]
    fn accumulates_text_deltas() {
        let mut state = StreamState::new();
        let mut events = vec![];

        apply(
            &mut state,
            r#"{"choices":[{"index":0,"delta":{"content":"Hel"}}]}"#,
            &mut events,
        );
        apply(
            &mut state,
            r#"{"choices":[{"index":0,"delta":{"content":"lo"}}]}"#,
            &mut events,
        );
        apply(
            &mut state,
            r#"{"choices":[],"usage":{"total_tokens":3}}"#,
            &mut events,
        );

        assert_eq!(
            events,
            [
                ModelEvent::Text("Hel".to_string()),
                ModelEvent::Text("lo".to_string())
            ]
        );
        assert_eq!(state.finish().content.as_deref(), Some("Hello"));
    }

    #[test]
    fn merges_tool_call_fragments_by_index() {
        let mut state = StreamState::new();
        let mut events = vec![];

        for frame in [
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"bash","arguments":""}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"comm"}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"and\":\"ls\"}"}}]}}]}"#,
        ] {
            apply(&mut state, frame, &mut events);
        }

        let response = state.finish();
        assert!(events.is_empty());
        assert_eq!(response.content, None);
        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.tool_calls[0].id, "call_1");
        assert_eq!(response.tool_calls[0].name, "bash");
        assert_eq!(response.tool_calls[0].arguments, json!({"command": "ls"}));
    }

    #[test]
    fn orders_parallel_tool_calls_by_index() {
        let mut state = StreamState::new();

        apply(
            &mut state,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":1,"id":"b","function":{"name":"read_file","arguments":"{\"path\":\"b\"}"}}]}}]}"#,
            &mut vec![],
        );
        apply(
            &mut state,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"a","function":{"name":"read_file","arguments":"{\"path\":\"a\"}"}}]}}]}"#,
            &mut vec![],
        );

        let calls = state.finish().tool_calls;
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].id, "a");
        assert_eq!(calls[1].id, "b");
    }

    #[test]
    fn rejects_error_frames() {
        let mut state = StreamState::new();
        let error = state
            .apply(r#"{"error":{"message":"boom"}}"#, &mut |_| {})
            .unwrap_err();
        assert!(format!("{error:#}").contains("boom"));
    }

    #[test]
    fn accepts_a_whole_completion_in_one_frame() {
        let mut state = StreamState::new();
        let mut events = vec![];

        let outcome = state
            .apply(
                r#"{"choices":[{"message":{"content":"hi"}}]}"#,
                &mut |event| events.push(event),
            )
            .unwrap()
            .expect("a whole completion ends the stream");

        assert!(matches!(outcome, StreamOutcome::WholeFrame(_)));
        assert_eq!(
            events,
            [ModelEvent::Fallback, ModelEvent::Text("hi".to_string())]
        );
    }

    #[test]
    fn keeps_sniffing_past_an_unparseable_frame() {
        let mut state = StreamState::new();
        state.raw = b"junk".to_vec();
        let mut events = vec![];

        assert!(state.apply("not json", &mut |_| {}).unwrap().is_none());
        apply(
            &mut state,
            r#"{"choices":[{"delta":{"content":"ok"}}]}"#,
            &mut events,
        );

        assert!(
            state.raw.is_empty(),
            "sniffing ends at the first JSON frame"
        );
        assert_eq!(events, [ModelEvent::Text("ok".to_string())]);
        assert_eq!(state.finish().content.as_deref(), Some("ok"));
    }

    #[test]
    fn ignores_done_and_empty_frames_after_sniffing() {
        let mut state = StreamState::new();
        apply(
            &mut state,
            r#"{"choices":[{"delta":{"content":"ok"}}]}"#,
            &mut vec![],
        );

        assert!(state.apply("[DONE]", &mut |_| {}).unwrap().is_none());
        assert!(state.apply("", &mut |_| {}).unwrap().is_none());
        assert_eq!(state.finish().content.as_deref(), Some("ok"));
    }
}
