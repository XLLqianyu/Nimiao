//! Nimiao command-line entry point.
//!
//! Reads an OpenAI-compatible endpoint from the environment, wires one model
//! to the built-in tool set, and serves a line-oriented REPL. The whole
//! session lives in one history vector, so each turn sees everything before
//! it. Replies are rendered as they stream in; tool calls appear as one status
//! line each.

use engine::{Agent, AgentEvent};
use model::openai::OpenAIModel;
use protocol::Item;
use sandbox::Sandbox;
use serde_json::Value;
use std::io::{self, Write};
use tokio::io::{AsyncBufReadExt, BufReader};
use tools::{default_registry, ToolContext};

/// Bright green prompt.
const PROMPT: &str = "\x1b[92m> \x1b[0m";

/// Bright blue label in front of model replies.
const AGENT_PREFIX: &str = "\x1b[94mAgent:\x1b[0m ";

/// Bright red label for non-fatal errors.
const ERROR_PREFIX: &str = "\x1b[91merror:\x1b[0m ";

/// Gray start for tool status lines.
const TOOL_PREFIX: &str = "\x1b[90m";

/// Gray label for notices, which go to stderr.
const NOTICE_PREFIX: &str = "\x1b[90mnote:\x1b[0m ";

/// Longest tool-call preview, in characters.
const PREVIEW_LIMIT: usize = 60;

#[tokio::main]
async fn main() {
    enable_ansi();

    let Some(api_key) = read_env("OPENAI_API_KEY") else {
        eprintln!("{ERROR_PREFIX}OPENAI_API_KEY is not set");
        std::process::exit(1);
    };
    let base_url =
        read_env("OPENAI_BASE_URL").unwrap_or_else(|| "https://api.openai.com/v1".to_string());
    let model_name = read_env("OPENAI_MODEL_NAME").unwrap_or_else(|| "gpt-4o".to_string());

    let sandbox = match Sandbox::new(".") {
        Ok(sandbox) => sandbox,
        Err(err) => {
            eprintln!("{ERROR_PREFIX}cannot open the workspace: {err}");
            std::process::exit(1);
        }
    };

    let ctx = ToolContext::new(sandbox);
    let root = ctx.workspace_root().to_path_buf();
    let registry = default_registry();

    let model = OpenAIModel::new(&api_key, &base_url, &model_name);
    let agent = Agent::new(model, registry, ctx);

    println!(
        "\x1b[1;36mNimiao\x1b[0m \x1b[90m·\x1b[0m {model_name} \x1b[90m·\x1b[0m {}",
        root.display()
    );
    println!("\x1b[90mType a request. `exit` or `quit` leaves.\x1b[0m");

    let mut history: Vec<Item> = Vec::new();
    let mut renderer = Renderer::new();
    let mut lines = BufReader::new(tokio::io::stdin()).lines();

    loop {
        print!("{PROMPT}");
        if let Err(err) = io::stdout().flush() {
            eprintln!("{ERROR_PREFIX}{err}");
            break;
        }

        let line = match lines.next_line().await {
            Ok(Some(line)) => line,
            // EOF: stdin was closed or the user pressed Ctrl+Z / Ctrl+D.
            Ok(None) => break,
            Err(err) => {
                eprintln!("{ERROR_PREFIX}cannot read input: {err}");
                break;
            }
        };

        let request = line.trim();
        if request.is_empty() {
            continue;
        }
        if request.eq_ignore_ascii_case("exit") || request.eq_ignore_ascii_case("quit") {
            break;
        }

        renderer.begin_turn();
        let result = {
            let mut sink = |event| renderer.on_event(event);
            agent
                .run_turn_stream(&mut history, request, &mut sink)
                .await
        };

        if let Err(err) = result {
            // A partial reply leaves the cursor mid-line; start the error on a
            // line of its own.
            renderer.end_line();
            eprintln!("{ERROR_PREFIX}{err:#}");
        } else {
            renderer.finish_turn();
        }
        if let Some(err) = renderer.take_error() {
            eprintln!("{ERROR_PREFIX}cannot write output: {err}");
            break;
        }
    }

    println!("\x1b[90mbye\x1b[0m");
}

/// Renders streamed events, remembering where the cursor is.
struct Renderer<O: Write, E: Write> {
    out: O,
    err: E,
    /// The cursor is at the start of a line.
    at_line_start: bool,
    /// The next text needs an `Agent:` label.
    label_pending: bool,
    /// The turn produced at least one text delta.
    wrote_text: bool,
    /// First failed write, kept so the REPL can stop cleanly.
    error: Option<io::Error>,
}

impl Renderer<io::Stdout, io::Stderr> {
    fn new() -> Self {
        Self::with_writers(io::stdout(), io::stderr())
    }
}

impl<O: Write, E: Write> Renderer<O, E> {
    fn with_writers(out: O, err: E) -> Self {
        Self {
            out,
            err,
            at_line_start: true,
            label_pending: true,
            wrote_text: false,
            error: None,
        }
    }

    /// Prepares for a new turn.
    fn begin_turn(&mut self) {
        self.end_line();
        self.label_pending = true;
        self.wrote_text = false;
    }

    fn on_event(&mut self, event: AgentEvent) {
        match event {
            AgentEvent::Text(delta) => {
                if self.label_pending {
                    self.write(AGENT_PREFIX);
                    self.label_pending = false;
                }
                self.write(&delta);
                self.at_line_start = delta.ends_with('\n');
                self.wrote_text = true;
                self.flush();
            }
            AgentEvent::ToolStarted { name, arguments } => {
                self.end_line();
                let preview = tool_preview(&arguments);
                self.write(&format!("{TOOL_PREFIX}> {name}({preview})\x1b[0m\n"));
                self.flush();
                self.label_pending = true;
            }
            AgentEvent::Notice(text) => {
                self.end_line();
                self.note(&text);
                self.label_pending = true;
            }
        }
    }

    /// Ends the open line, if any.
    fn end_line(&mut self) {
        if !self.at_line_start {
            self.write("\n");
            self.flush();
            self.at_line_start = true;
        }
    }

    /// Ends the turn: closes the line and fills in a missing reply.
    fn finish_turn(&mut self) {
        self.end_line();
        if !self.wrote_text && self.error.is_none() {
            self.write(&format!("{AGENT_PREFIX}\x1b[90m(no reply)\x1b[0m\n"));
            self.flush();
        }
    }

    fn take_error(&mut self) -> Option<io::Error> {
        self.error.take()
    }

    fn write(&mut self, text: &str) {
        if self.error.is_some() {
            return;
        }
        if let Err(err) = self.out.write_all(text.as_bytes()) {
            self.error = Some(err);
        }
    }

    fn flush(&mut self) {
        if self.error.is_some() {
            return;
        }
        if let Err(err) = self.out.flush() {
            self.error = Some(err);
        }
    }

    fn note(&mut self, text: &str) {
        if self.error.is_some() {
            return;
        }
        let line = format!("{NOTICE_PREFIX}\x1b[90m{text}\x1b[0m\n");
        if let Err(err) = self
            .err
            .write_all(line.as_bytes())
            .and_then(|()| self.err.flush())
        {
            self.error = Some(err);
        }
    }
}

/// Reads an environment variable, treating blank values as absent.
fn read_env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
}

/// One short, single-line summary of a tool call's arguments.
///
/// Shell commands and paths identify a call best. Other tools fall back to the
/// first string in the argument object. `serde_json` keeps object keys sorted,
/// so that fallback would show `write_file`'s whole `content` before its
/// `path`; the explicit order above prevents it.
fn tool_preview(arguments: &Value) -> String {
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

/// Turns on ANSI escape handling for the current console.
///
/// Windows Terminal renders escape sequences out of the box, but a legacy
/// console window needs `ENABLE_VIRTUAL_TERMINAL_PROCESSING` first.
#[cfg(windows)]
fn enable_ansi() {
    use std::ffi::c_void;

    const STD_OUTPUT_HANDLE: u32 = 0xFFFF_FFF5;
    const ENABLE_VIRTUAL_TERMINAL_PROCESSING: u32 = 0x0004;

    extern "system" {
        fn GetStdHandle(std_handle: u32) -> *mut c_void;
        fn GetConsoleMode(console_handle: *mut c_void, mode: *mut u32) -> i32;
        fn SetConsoleMode(console_handle: *mut c_void, mode: u32) -> i32;
    }

    // SAFETY: the calls only inspect the standard output handle owned by this
    // process. Any failure just means the console keeps its default mode.
    unsafe {
        let handle = GetStdHandle(STD_OUTPUT_HANDLE);
        if handle.is_null() {
            return;
        }
        let mut mode = 0u32;
        if GetConsoleMode(handle, &mut mode) != 0 {
            let _ = SetConsoleMode(handle, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
        }
    }
}

#[cfg(not(windows))]
fn enable_ansi() {}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::{Arc, Mutex};

    /// Renders into memory so the output can be asserted on.
    #[derive(Clone, Default)]
    struct Buffer(Arc<Mutex<Vec<u8>>>);

    impl Buffer {
        fn text(&self) -> String {
            String::from_utf8(self.0.lock().unwrap().clone()).unwrap()
        }
    }

    impl Write for Buffer {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    fn renderer() -> (Renderer<Buffer, Buffer>, Buffer, Buffer) {
        let out = Buffer::default();
        let err = Buffer::default();
        (Renderer::with_writers(out.clone(), err.clone()), out, err)
    }

    #[test]
    fn labels_the_first_text_and_not_each_delta() {
        let (mut renderer, out, _) = renderer();
        renderer.begin_turn();
        renderer.on_event(AgentEvent::Text("Hel".to_string()));
        renderer.on_event(AgentEvent::Text("lo".to_string()));
        renderer.finish_turn();

        let text = out.text();
        assert_eq!(text.matches(AGENT_PREFIX).count(), 1, "{text:?}");
        assert!(text.contains("Hello"), "{text:?}");
        assert!(text.ends_with('\n'), "{text:?}");
    }

    #[test]
    fn keeps_a_multi_line_reply_under_one_label() {
        let (mut renderer, out, _) = renderer();
        renderer.begin_turn();
        renderer.on_event(AgentEvent::Text("first\nsecond\n".to_string()));
        renderer.on_event(AgentEvent::Text("third".to_string()));
        renderer.finish_turn();

        let text = out.text();
        assert_eq!(text.matches(AGENT_PREFIX).count(), 1, "{text:?}");
        assert!(!text.contains("second\n\n"), "no blank line: {text:?}");
        assert!(text.ends_with("third\n"), "{text:?}");
    }

    #[test]
    fn closes_an_open_line_before_a_tool_status() {
        let (mut renderer, out, _) = renderer();
        renderer.begin_turn();
        renderer.on_event(AgentEvent::Text("thinking".to_string()));
        renderer.on_event(AgentEvent::ToolStarted {
            name: "bash".to_string(),
            arguments: json!({"command": "ls"}),
        });
        renderer.finish_turn();

        let text = out.text();
        assert!(text.contains("thinking\n"), "{text:?}");
        assert!(text.contains("> bash(ls)"), "{text:?}");
    }

    #[test]
    fn labels_text_again_after_a_tool_status() {
        let (mut renderer, out, _) = renderer();
        renderer.begin_turn();
        renderer.on_event(AgentEvent::Text("before".to_string()));
        renderer.on_event(AgentEvent::ToolStarted {
            name: "bash".to_string(),
            arguments: json!({"command": "ls"}),
        });
        renderer.on_event(AgentEvent::Text("after".to_string()));
        renderer.finish_turn();

        let text = out.text();
        assert_eq!(text.matches(AGENT_PREFIX).count(), 2, "{text:?}");
    }

    #[test]
    fn writes_notices_to_stderr_and_keeps_stdout_clean() {
        let (mut renderer, out, err) = renderer();
        renderer.begin_turn();
        renderer.on_event(AgentEvent::Text("partial".to_string()));
        renderer.on_event(AgentEvent::Notice("fell back".to_string()));
        renderer.on_event(AgentEvent::Text("more".to_string()));
        renderer.finish_turn();

        let out = out.text();
        assert!(!out.contains("fell back"), "{out:?}");
        assert!(out.contains("partial\n"), "the line is closed: {out:?}");
        assert!(err.text().contains("fell back"), "{:?}", err.text());
    }

    #[test]
    fn fills_in_a_missing_reply() {
        let (mut renderer, out, _) = renderer();
        renderer.begin_turn();
        renderer.finish_turn();

        assert!(out.text().contains("(no reply)"), "{:?}", out.text());
    }

    #[test]
    fn is_silent_when_a_write_failed() {
        let out = Buffer::default();
        let mut renderer = Renderer::with_writers(out.clone(), Buffer::default());
        renderer.begin_turn();
        renderer.write("x");
        renderer.error = Some(io::Error::other("disk full"));

        renderer.finish_turn();

        assert!(!out.text().contains("(no reply)"), "{:?}", out.text());
        assert!(renderer.take_error().is_some());
        assert!(
            renderer.take_error().is_none(),
            "the error is reported once"
        );
    }

    #[test]
    fn does_not_double_space_a_turn_boundary() {
        let (mut renderer, out, _) = renderer();
        renderer.begin_turn();
        renderer.on_event(AgentEvent::Text("one\n".to_string()));
        renderer.finish_turn();
        renderer.begin_turn();
        renderer.on_event(AgentEvent::Text("two".to_string()));
        renderer.finish_turn();

        let text = out.text();
        assert!(!text.contains("one\n\n"), "{text:?}");
        assert!(text.ends_with("two\n"), "{text:?}");
    }

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
        assert_eq!(tool_preview(&json!({"command": "a\x1b[31mb"})), "a [31mb");
    }

    #[test]
    fn truncates_long_values() {
        let preview = tool_preview(&json!({"command": "x".repeat(100)}));
        assert_eq!(preview.chars().count(), PREVIEW_LIMIT + 1);
        assert!(preview.ends_with('…'));
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
