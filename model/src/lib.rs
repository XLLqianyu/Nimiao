use async_trait::async_trait;
use protocol::{Item, ToolCall};

pub mod openai;
pub mod sse;

pub use protocol::ToolDefinition;

/// Why the model stopped producing this completion.
///
/// `None` on a [`ModelResponse`] means the provider did not report a reason;
/// proxies commonly omit it, so that is not treated as an error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    /// The model finished on its own.
    Stop,
    /// The output token limit was hit; tool arguments may be truncated.
    Length,
    /// The model is waiting for tool results.
    ToolCalls,
    /// The provider removed the content.
    ContentFilter,
    /// A value this client does not model, kept for the message it came with.
    Other(String),
}

#[derive(Debug, Clone)]
pub struct ModelResponse {
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub stop_reason: Option<StopReason>,
}

/// Progress reported while a completion is being produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelEvent {
    /// A fragment of the reply text, in stream order.
    Text(String),
    /// The endpoint did not stream; this completion arrived whole.
    Fallback,
}

#[async_trait]
pub trait Model: Send + Sync {
    fn name(&self) -> &str;
    async fn complete(
        &self,
        history: &[Item],
        tools: &[ToolDefinition],
    ) -> anyhow::Result<ModelResponse>;

    /// Streams one completion, reporting text as it arrives.
    ///
    /// Implementations that only support one-shot completions use this default,
    /// which emits the finished reply as a single [`ModelEvent::Text`]. It never
    /// emits [`ModelEvent::Fallback`]: that event means streaming was attempted
    /// and did not work, which only an implementation that tries can know.
    async fn complete_stream(
        &self,
        history: &[Item],
        tools: &[ToolDefinition],
        on_event: &mut (dyn FnMut(ModelEvent) + Send),
    ) -> anyhow::Result<ModelResponse> {
        let response = self.complete(history, tools).await?;
        if let Some(text) = response.content.as_deref().filter(|text| !text.is_empty()) {
            on_event(ModelEvent::Text(text.to_string()));
        }
        Ok(response)
    }
}
