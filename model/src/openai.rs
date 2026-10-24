use crate::sse::SseDecoder;
use crate::{Model, ModelEvent, ModelResponse, StopReason, ToolDefinition};
use anyhow::Context;
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

/// One assistant step being collected: its text so far and every call it
/// declared, in the order the model produced them.
#[derive(Default)]
struct AssistantStep {
    text: String,
    calls: Vec<serde_json::Value>,
}

impl AssistantStep {
    fn push_text(&mut self, text: &str) {
        self.text.push_str(text);
    }

    fn push_call(&mut self, call: &ToolCall) {
        self.calls.push(json!({
            "id": call.id,
            "type": "function",
            "function": {
                "name": call.name,
                "arguments": call.arguments.to_string()
            }
        }));
    }
}

/// Writes the pending step, if any, and clears it.
///
/// A step with neither text nor calls is skipped: several providers refuse an
/// `assistant` message that carries neither, and there is nothing to say in it.
fn flush_step(messages: &mut Vec<serde_json::Value>, open: &mut Option<AssistantStep>) {
    let Some(step) = open.take() else {
        return;
    };
    let content = if step.text.is_empty() {
        serde_json::Value::Null
    } else {
        serde_json::Value::String(step.text)
    };
    if content.is_null() && step.calls.is_empty() {
        return;
    }

    let mut message = json!({"role": "assistant", "content": content});
    if !step.calls.is_empty() {
        message["tool_calls"] = serde_json::Value::Array(step.calls);
    }
    messages.push(message);
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

    /// Maps the flat history onto the wire, one `assistant` message per step.
    ///
    /// A step's text and every tool call it declared belong to a single
    /// `assistant` message: providers reject a call whose assistant message
    /// carries a different set of calls, and the ones that are strict reject an
    /// `assistant` message with neither content nor calls.
    fn format_messages(history: &[Item]) -> Vec<serde_json::Value> {
        let mut messages = vec![];
        let mut open: Option<AssistantStep> = None;

        for item in history {
            match item {
                Item::UserMessage(text) => {
                    flush_step(&mut messages, &mut open);
                    messages.push(json!({"role": "user", "content": text}));
                }
                Item::AssistantMessage(text) => {
                    // A step with no text still gets its message once a call
                    // arrives; an empty one on its own is dropped by the flush.
                    if text.trim().is_empty() {
                        continue;
                    }
                    open.get_or_insert_with(AssistantStep::default)
                        .push_text(text);
                }
                Item::ToolCall(call) => {
                    open.get_or_insert_with(AssistantStep::default)
                        .push_call(call);
                }
                Item::ToolResult {
                    tool_call_id,
                    content,
                    ..
                } => {
                    flush_step(&mut messages, &mut open);
                    messages.push(json!({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": content
                    }));
                }
            }
        }

        flush_step(&mut messages, &mut open);
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
                let (arguments, arguments_error) =
                    parse_arguments(tc["function"]["arguments"].as_str());
                tool_calls.push(ToolCall {
                    id: tc["id"].as_str().unwrap_or_default().to_string(),
                    name: tc["function"]["name"]
                        .as_str()
                        .unwrap_or_default()
                        .to_string(),
                    arguments,
                    arguments_error,
                });
            }
        }

        Ok(ModelResponse {
            content,
            tool_calls,
            stop_reason: map_stop_reason(&choice["finish_reason"]),
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
    while let Some(chunk) = res.chunk().await.context("reading the response body")? {
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

    while let Some(chunk) = res
        .chunk()
        .await
        .context("the response stream ended unexpectedly")?
    {
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
    /// The last `finish_reason` seen, kept across frames.
    stop_reason: Option<StopReason>,
}

impl StreamState {
    fn new() -> Self {
        Self {
            sniffing: true,
            raw: Vec::new(),
            text: String::new(),
            calls: ToolCallAccumulator::default(),
            stop_reason: None,
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
        // A finish reason may arrive on any frame of a stream, usually the
        // last one, so the latest value wins.
        self.stop_reason = map_stop_reason(&choice["finish_reason"]).or(self.stop_reason.take());
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
            stop_reason: self.stop_reason,
        }
    }
}

/// Maps a provider `finish_reason` onto a [`StopReason`].
///
/// `null`, a missing field, and an unknown string are all reported as `None`
/// or `Other`: this client must still work against proxies that do not report
/// a reason the way OpenAI does.
fn map_stop_reason(reason: &Value) -> Option<StopReason> {
    match reason.as_str()? {
        "stop" | "end" => Some(StopReason::Stop),
        "length" => Some(StopReason::Length),
        "tool_calls" | "function_call" => Some(StopReason::ToolCalls),
        "content_filter" => Some(StopReason::ContentFilter),
        other => Some(StopReason::Other(other.to_string())),
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
            .map(|call| {
                let (arguments, arguments_error) = parse_arguments(Some(&call.arguments));
                ToolCall {
                    id: call.id,
                    name: call.name,
                    arguments,
                    arguments_error,
                }
            })
            .collect()
    }
}

/// Parses a finished call's arguments text.
///
/// A finished call whose text is not a JSON object cannot be replayed as-is:
/// the provider requires `arguments` to be JSON. It becomes `{}` and the text
/// the model sent is kept beside it, so the tool result can quote it back
/// instead of the model silently seeing an empty object.
fn parse_arguments(text: Option<&str>) -> (Value, Option<String>) {
    let Some(text) = text.filter(|text| !text.trim().is_empty()) else {
        return (json!({}), None);
    };
    match serde_json::from_str::<Value>(text) {
        Ok(Value::Object(map)) => (Value::Object(map), None),
        Ok(other) => (
            json!({}),
            Some(format!(
                "tool arguments must be a JSON object, got {}\n\nReceived arguments:\n{text}",
                match other {
                    Value::Null => "null",
                    Value::Bool(_) => "a boolean",
                    Value::Number(_) => "a number",
                    Value::String(_) => "a string",
                    Value::Array(_) => "an array",
                    Value::Object(_) => unreachable!(),
                }
            )),
        ),
        Err(err) => (
            json!({}),
            Some(format!("{err}\n\nReceived arguments:\n{text}")),
        ),
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

    #[test]
    fn assembles_text_split_inside_a_character() {
        let mut state = StreamState::new();
        let mut events = vec![];
        let mut decoder = crate::sse::SseDecoder::new();

        let frames = "data: {\"choices\":[{\"delta\":{\"content\":\"你\"}}]}\n\n\
                      data: {\"choices\":[{\"delta\":{\"content\":\"好\"}}]}\n\n";
        // Split inside the second character's UTF-8 bytes.
        let split = frames.find('你').unwrap() + 1;
        for part in [&frames.as_bytes()[..split], &frames.as_bytes()[split..]] {
            for frame in decoder.feed(part).unwrap() {
                apply(&mut state, &frame, &mut events);
            }
        }

        assert_eq!(
            events,
            [
                ModelEvent::Text("你".to_string()),
                ModelEvent::Text("好".to_string())
            ]
        );
        assert_eq!(state.finish().content.as_deref(), Some("你好"));
    }

    #[test]
    fn treats_a_missing_tool_index_as_the_first_call() {
        let mut accumulator = ToolCallAccumulator::default();
        accumulator.push(&json!({
            "id": "call_1",
            "function": {"name": "bash", "arguments": "{\"command\":\"ls\"}"}
        }));

        let calls = accumulator.finish();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].arguments, json!({"command": "ls"}));
    }

    #[test]
    fn keeps_the_first_id_and_name_of_a_call() {
        let mut accumulator = ToolCallAccumulator::default();
        accumulator.push(&json!({"index": 0, "id": "first", "function": {"name": "bash"}}));
        // Later fragments usually omit both; a repeated one must not overwrite.
        accumulator.push(&json!({"index": 0, "id": "second", "function": {"name": "other"}}));

        let calls = accumulator.finish();
        assert_eq!(calls[0].id, "first");
        assert_eq!(calls[0].name, "bash");
    }

    #[test]
    fn treats_a_null_content_as_no_reply() {
        let mut state = StreamState::new();
        apply(
            &mut state,
            r#"{"choices":[{"delta":{"content":null}}]}"#,
            &mut vec![],
        );

        let response = state.finish();
        assert_eq!(response.content, None);
        assert!(response.tool_calls.is_empty());
    }

    #[test]
    fn joins_text_that_arrives_with_tool_calls() {
        let mut state = StreamState::new();
        let mut events = vec![];

        apply(
            &mut state,
            r#"{"choices":[{"delta":{"content":"calling ","tool_calls":[{"index":0,"id":"c","function":{"name":"bash","arguments":"{}"}}]}}]}"#,
            &mut events,
        );

        let response = state.finish();
        assert_eq!(response.content.as_deref(), Some("calling "));
        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(events, [ModelEvent::Text("calling ".to_string())]);
    }

    #[test]
    fn reads_a_whole_completion_with_tool_calls() {
        let mut state = StreamState::new();
        let mut events = vec![];

        state
            .apply(
                r#"{"choices":[{"message":{"content":null,"tool_calls":[{"id":"call_1","function":{"name":"bash","arguments":"{\"command\":\"ls\"}"}}]}}]}"#,
                &mut |event| events.push(event),
            )
            .unwrap();

        assert_eq!(events, [ModelEvent::Fallback]);
    }
}
