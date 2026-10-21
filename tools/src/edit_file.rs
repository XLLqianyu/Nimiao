//! `edit_file`: exact-text replacements in a single file.

use crate::edit_diff::{
    apply_edits_to_normalized_content, detect_line_ending, normalize_to_lf, restore_line_endings,
    split_bom, Edit,
};
use crate::path_util::normalize_input_path;
use crate::{Tool, ToolContext, ToolOutput};
use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use protocol::ToolDefinition;
use serde_json::{json, Value};

/// Edits a file by replacing unique text blocks.
pub struct EditFileTool;

#[async_trait]
impl Tool for EditFileTool {
    fn name(&self) -> &str {
        "edit_file"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name().to_string(),
            description: "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, \
                          non-overlapping region of the original file. If two changes affect the same block or nearby \
                          lines, merge them into one edit instead of emitting overlapping edits. Do not include large \
                          unchanged regions just to connect distant changes."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to edit (relative or absolute)"
                    },
                    "edits": {
                        "type": "array",
                        "description": "One or more targeted replacements. Each edit is matched against the original \
                                        file, not incrementally. Do not include overlapping or nested edits. If two \
                                        changes touch the same block or nearby lines, merge them into one edit instead.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "oldText": {
                                    "type": "string",
                                    "description": "Exact text for one targeted replacement. It must be unique in \
                                                    the original file and must not overlap with any other \
                                                    edits[].oldText in the same call."
                                },
                                "newText": {
                                    "type": "string",
                                    "description": "Replacement text for this targeted edit."
                                }
                            },
                            "required": ["oldText", "newText"],
                            "additionalProperties": false
                        }
                    }
                },
                "required": ["path", "edits"],
                "additionalProperties": false
            }),
        }
    }

    async fn execute(&self, args: Value, ctx: &ToolContext) -> Result<ToolOutput> {
        let (display_path, edits) = prepare_args(args)?;

        let resolved = ctx
            .sandbox
            .validate_path(normalize_input_path(&display_path))
            .with_context(|| format!("Cannot edit '{display_path}'"))?;

        let bytes = tokio::fs::read(&resolved)
            .await
            .with_context(|| format!("Cannot edit file: {display_path}"))?;
        let raw_content = String::from_utf8(bytes).map_err(|_| {
            anyhow::anyhow!("Cannot edit '{display_path}': the file is not valid UTF-8 text")
        })?;

        // The model never types an invisible BOM, so match without it and put
        // it back on write.
        let (has_bom, content) = split_bom(&raw_content);
        let line_ending = detect_line_ending(content);
        let normalized = normalize_to_lf(content);

        let applied = apply_edits_to_normalized_content(&normalized, &edits, &display_path)?;

        let mut final_content = String::with_capacity(applied.new_content.len() + 3);
        if has_bom {
            final_content.push('\u{feff}');
        }
        final_content.push_str(&restore_line_endings(&applied.new_content, line_ending));

        tokio::fs::write(&resolved, final_content.as_bytes())
            .await
            .with_context(|| format!("Cannot edit file: {display_path}"))?;

        Ok(ToolOutput::success(format!(
            "Successfully replaced {} block(s) in {display_path}.",
            edits.len()
        )))
    }
}

/// Accepts the shapes different models produce for the same intent.
///
/// Besides the documented `{ path, edits: [{ oldText, newText }] }`, this
/// handles a lone edit object, `edits` serialized as a JSON string, and the
/// older top-level `oldText` / `newText` pair.
fn prepare_args(args: Value) -> Result<(String, Vec<Edit>)> {
    let Value::Object(mut object) = args else {
        bail!("Invalid arguments: expected an object");
    };

    let path = match object.remove("path") {
        Some(Value::String(path)) => path,
        Some(_) => bail!("Invalid arguments: 'path' must be a string"),
        None => bail!("Invalid arguments: missing 'path'"),
    };

    let mut raw_edits: Vec<Value> = match object.remove("edits") {
        Some(Value::Array(items)) => items,
        Some(Value::Object(single)) => vec![Value::Object(single)],
        Some(Value::String(encoded)) => match serde_json::from_str::<Value>(&encoded) {
            Ok(Value::Array(items)) => items,
            Ok(Value::Object(single)) => vec![Value::Object(single)],
            _ => {
                bail!("Invalid arguments: 'edits' string must contain an edit or an array of edits")
            }
        },
        Some(Value::Null) | None => Vec::new(),
        Some(_) => bail!("Invalid arguments: 'edits' must be an array"),
    };

    // Legacy shape: one replacement spread across the top level.
    if let (Some(Value::String(old_text)), Some(Value::String(new_text))) =
        (object.remove("oldText"), object.remove("newText"))
    {
        raw_edits.push(json!({ "oldText": old_text, "newText": new_text }));
    }

    if raw_edits.is_empty() {
        bail!("Edit tool input is invalid. edits must contain at least one replacement.");
    }

    let mut edits = Vec::with_capacity(raw_edits.len());
    for (index, raw) in raw_edits.into_iter().enumerate() {
        let edit: Edit = serde_json::from_value(raw)
            .map_err(|err| anyhow::anyhow!("Invalid arguments: edits[{index}]: {err}"))?;
        edits.push(edit);
    }

    Ok((path, edits))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_a_normal_edit_array() {
        let args = json!({
            "path": "a.txt",
            "edits": [
                {"oldText": "a", "newText": "b"},
                {"oldText": "c", "newText": "d"}
            ]
        });
        let (path, edits) = prepare_args(args).unwrap();
        assert_eq!(path, "a.txt");
        assert_eq!(edits.len(), 2);
        assert_eq!(edits[1].new_text, "d");
    }

    #[test]
    fn accepts_a_single_edit_object() {
        let args = json!({
            "path": "a.txt",
            "edits": {"oldText": "a", "newText": "b"}
        });
        let (_, edits) = prepare_args(args).unwrap();
        assert_eq!(edits.len(), 1);
    }

    #[test]
    fn accepts_edits_encoded_as_a_string() {
        let args = json!({
            "path": "a.txt",
            "edits": "[{\"oldText\": \"a\", \"newText\": \"b\"}]"
        });
        let (_, edits) = prepare_args(args).unwrap();
        assert_eq!(edits.len(), 1);
    }

    #[test]
    fn accepts_the_legacy_top_level_shape() {
        let args = json!({"path": "a.txt", "oldText": "a", "newText": "b"});
        let (_, edits) = prepare_args(args).unwrap();
        assert_eq!(edits.len(), 1);
        assert_eq!(edits[0].old_text, "a");
    }

    #[test]
    fn rejects_missing_edits() {
        let args = json!({"path": "a.txt"});
        assert!(prepare_args(args).is_err());
    }

    #[test]
    fn rejects_missing_path() {
        let args = json!({"edits": [{"oldText": "a", "newText": "b"}]});
        assert!(prepare_args(args).is_err());
    }
}
