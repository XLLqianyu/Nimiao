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
    /// Set when the model's arguments text was not a valid JSON object.
    ///
    /// `arguments` is then `{}`, the only value that can be replayed on the
    /// wire, and this holds the text the model actually sent so the tool
    /// result can tell it what went wrong. Never sent to the provider.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments_error: Option<String>,
}

impl ToolCall {
    /// A call whose arguments parsed cleanly.
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: serde_json::Value,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            arguments,
            arguments_error: None,
        }
    }
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
