//! `write_file`: whole-file writer.

use crate::path_util::normalize_input_path;
use crate::{Tool, ToolContext, ToolOutput};
use anyhow::{Context, Result};
use async_trait::async_trait;
use protocol::ToolDefinition;
use serde::Deserialize;
use serde_json::json;

#[derive(Debug, Deserialize)]
struct WriteFileArgs {
    path: String,
    content: String,
}

/// Creates or overwrites a file, creating parent directories as needed.
pub struct WriteFileTool;

#[async_trait]
impl Tool for WriteFileTool {
    fn name(&self) -> &str {
        "write_file"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name().to_string(),
            description: "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. \
                          Parent directories are created automatically."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write (relative or absolute)"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file"
                    }
                },
                "required": ["path", "content"],
                "additionalProperties": false
            }),
        }
    }

    async fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> Result<ToolOutput> {
        let args: WriteFileArgs = serde_json::from_value(args)
            .map_err(|err| anyhow::anyhow!("Invalid arguments: {err}"))?;
        let display_path = args.path.clone();

        let resolved = ctx
            .sandbox
            .validate_path(normalize_input_path(&args.path))
            .with_context(|| format!("Cannot write '{display_path}'"))?;

        if let Some(parent) = resolved.parent() {
            tokio::fs::create_dir_all(parent).await.with_context(|| {
                format!("Cannot create parent directories for '{display_path}'")
            })?;
        }

        // Content is written verbatim: truncating a file silently would corrupt it.
        tokio::fs::write(&resolved, args.content.as_bytes())
            .await
            .with_context(|| format!("Cannot write '{display_path}'"))?;

        Ok(ToolOutput::success(format!(
            "Successfully wrote to {display_path}"
        )))
    }
}
