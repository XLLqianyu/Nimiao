use serde::{Deserialize, Serialize};

/// A tool exposed to the model.
///
/// `parameters` holds a JSON Schema object exactly as it should be sent on the
/// wire; providers forward it verbatim.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Item {
    UserMessage(String),
    AssistantMessage(String),
    ToolCall(ToolCall),
    ToolResult {
        tool_call_id: String,
        content: String,
        is_error: bool,
    },
}
