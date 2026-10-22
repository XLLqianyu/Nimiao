use async_trait::async_trait;
use protocol::{Item, ToolCall};

pub mod openai;
pub mod sse;

pub use protocol::ToolDefinition;

#[derive(Debug, Clone)]
pub struct ModelResponse {
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
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
