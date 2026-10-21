//! Built-in tools for the Nimiao agent.
//!
//! Each tool implements the [`Tool`] trait: it advertises a JSON Schema
//! definition and executes a model-supplied argument object. Filesystem access
//! goes through [`ToolContext::sandbox`] and shell commands are screened by
//! `execpolicy` before they run.

pub mod bash;
pub mod edit_diff;
pub mod edit_file;
pub mod path_util;
pub mod read_file;
pub mod truncate;
pub mod write_file;

pub use bash::BashTool;
pub use edit_file::EditFileTool;
pub use read_file::ReadFileTool;
pub use write_file::WriteFileTool;

use async_trait::async_trait;
use protocol::ToolDefinition;
use sandbox::Sandbox;
use std::path::Path;

/// The result of a tool execution.
///
/// `is_error` marks outcomes the model should learn from (a failed command, an
/// unexpected exit status) while still delivering the content that explains
/// what happened.
#[derive(Debug, Clone)]
pub struct ToolOutput {
    pub content: String,
    pub is_error: bool,
}

impl ToolOutput {
    pub fn success(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: false,
        }
    }

    pub fn failure(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: true,
        }
    }
}

/// Execution environment shared by all tools.
#[derive(Debug, Clone)]
pub struct ToolContext {
    /// Filesystem boundary that every path must pass through.
    pub sandbox: Sandbox,
}

impl ToolContext {
    pub fn new(sandbox: Sandbox) -> Self {
        Self { sandbox }
    }

    /// The workspace root, also the working directory for shell commands.
    pub fn workspace_root(&self) -> &Path {
        self.sandbox.root()
    }
}

/// A capability the model can invoke.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Stable identifier used in tool calls.
    fn name(&self) -> &str;

    /// Description and parameter schema advertised to the model.
    fn definition(&self) -> ToolDefinition;

    /// Runs the tool.
    ///
    /// Returns `Err` when the tool itself could not run: invalid arguments, an
    /// inaccessible path, an unmatched edit. Returns
    /// [`ToolOutput::failure`] when the operation ran but failed in a way the
    /// model should see, such as a non-zero exit status.
    async fn execute(
        &self,
        args: serde_json::Value,
        ctx: &ToolContext,
    ) -> anyhow::Result<ToolOutput>;
}

/// Every built-in tool, ready to register with an agent loop.
pub fn builtin_tools() -> Vec<Box<dyn Tool>> {
    vec![
        Box::new(ReadFileTool),
        Box::new(WriteFileTool),
        Box::new(EditFileTool),
        Box::new(BashTool),
    ]
}

/// An ordered set of tools, looked up by name.
pub struct ToolRegistry {
    tools: Vec<Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new(tools: Vec<Box<dyn Tool>>) -> Self {
        Self { tools }
    }

    pub fn register(&mut self, tool: Box<dyn Tool>) {
        self.tools.push(tool);
    }

    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools
            .iter()
            .find(|tool| tool.name() == name)
            .map(|tool| tool.as_ref())
    }

    /// Schemas for every registered tool, in registration order.
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools.iter().map(|tool| tool.definition()).collect()
    }

    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

/// The tool set Nimiao starts with.
pub fn default_registry() -> ToolRegistry {
    ToolRegistry::new(builtin_tools())
}
