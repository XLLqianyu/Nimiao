use async_trait::async_trait;
use protocol::{Item, ToolCall};

pub mod openai;

pub use protocol::ToolDefinition;

#[derive(Debug, Clone)]
pub struct ModelResponse {
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
}

#[async_trait]
pub trait Model: Send + Sync {
    fn name(&self) -> &str;
    async fn complete(
        &self,
        history: &[Item],
        tools: &[ToolDefinition],
    ) -> anyhow::Result<ModelResponse>;
}
