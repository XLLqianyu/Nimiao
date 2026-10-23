//! One-line summaries of tool calls for the plain and TUI renderers.

use serde_json::Value;

/// Longest tool-call preview, in characters.
const PREVIEW_LIMIT: usize = 60;

/// Summarizes the arguments a model passed to a tool.
///
/// Shell commands and paths identify a call best. Other tools fall back to the
/// first string in the argument object. `serde_json` keeps object keys sorted,
/// so that fallback would show `write_file`'s whole `content` before its
/// `path`; the explicit order above prevents it.
pub(crate) fn tool_preview(arguments: &Value) -> String {
    let Some(object) = arguments.as_object() else {
        return truncate(&arguments.to_string());
    };

    let preferred = ["command", "path"]
        .into_iter()
        .filter_map(|key| object.get(key).and_then(Value::as_str))
        .chain(object.values().filter_map(Value::as_str))
        .find(|value| !value.trim().is_empty());

    match preferred {
        Some(value) => truncate(&collapse(value)),
        None => truncate(&arguments.to_string()),
    }
}

/// Replaces control characters and whitespace runs with single spaces.
fn collapse(text: &str) -> String {
    text.chars()
        .map(|letter| if letter.is_control() { ' ' } else { letter })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Caps `text` at [`PREVIEW_LIMIT`] characters, marking the cut.
fn truncate(text: &str) -> String {
    let mut chars = text.chars();
    let mut head: String = chars.by_ref().take(PREVIEW_LIMIT).collect();
    if chars.next().is_some() {
        head.push('…');
    }
    head
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn previews_the_command() {
        assert_eq!(
            tool_preview(&json!({"command": "cargo test", "timeout": 60})),
            "cargo test"
        );
    }

    #[test]
    fn prefers_the_path_over_other_content() {
        let preview = tool_preview(&json!({"path": "src/main.rs", "content": "fn main() {}"}));
        assert_eq!(preview, "src/main.rs");
    }

    #[test]
    fn collapses_whitespace_and_control_characters() {
        assert_eq!(
            tool_preview(&json!({"command": "echo\n  hello\tworld"})),
            "echo hello world"
        );
        assert_eq!(tool_preview(&json!({"command": "a\u{1b}[31mb"})), "a [31mb");
    }

    #[test]
    fn truncates_long_values() {
        assert_eq!(
            tool_preview(&json!({"command": "x".repeat(100)})),
            format!("{}…", "x".repeat(PREVIEW_LIMIT))
        );
    }

    #[test]
    fn falls_back_to_the_whole_argument_object() {
        assert_eq!(tool_preview(&json!({"timeout": 30})), "{\"timeout\":30}");
    }

    #[test]
    fn ignores_an_empty_command_in_favour_of_a_path() {
        assert_eq!(
            tool_preview(&json!({"command": "  ", "path": "a.txt"})),
            "a.txt"
        );
    }
}
