//! Line-oriented streaming REPL used for redirected output and `--plain`.

use crate::tool_preview::tool_preview;
use engine::{Agent, AgentEvent};
use model::openai::OpenAIModel;
use protocol::Item;
use std::io::{self, Write};
use std::path::Path;
use tokio::io::{AsyncBufReadExt, BufReader};

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

/// Runs the legacy streaming REPL until EOF or an explicit quit.
pub(crate) async fn run(
    agent: Agent<OpenAIModel>,
    model_name: String,
    root: impl AsRef<Path>,
) -> anyhow::Result<()> {
    let root = root.as_ref();
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
        if let Err(error) = io::stdout().flush() {
            eprintln!("{ERROR_PREFIX}{error}");
            break;
        }

        let line = match lines.next_line().await {
            Ok(Some(line)) => line,
            Ok(None) => break,
            Err(error) => {
                eprintln!("{ERROR_PREFIX}cannot read input: {error}");
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

        if let Err(error) = result {
            renderer.end_line();
            eprintln!("{ERROR_PREFIX}{error:#}");
        } else {
            renderer.finish_turn();
        }
        if let Some(error) = renderer.take_error() {
            eprintln!("{ERROR_PREFIX}cannot write output: {error}");
            break;
        }
    }

    println!("\x1b[90mbye\x1b[0m");
    Ok(())
}

/// Renders streamed events, remembering where the cursor is.
struct Renderer<O: Write, E: Write> {
    out: O,
    err: E,
    at_line_start: bool,
    label_pending: bool,
    wrote_text: bool,
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
            AgentEvent::ToolFinished { .. } => {}
            AgentEvent::Notice(text) => {
                self.end_line();
                self.note(&text);
                self.label_pending = true;
            }
        }
    }

    fn end_line(&mut self) {
        if !self.at_line_start {
            self.write("\n");
            self.flush();
            self.at_line_start = true;
        }
    }

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
        if let Err(error) = self.out.write_all(text.as_bytes()) {
            self.error = Some(error);
        }
    }

    fn flush(&mut self) {
        if self.error.is_some() {
            return;
        }
        if let Err(error) = self.out.flush() {
            self.error = Some(error);
        }
    }

    fn note(&mut self, text: &str) {
        if self.error.is_some() {
            return;
        }
        let line = format!("{NOTICE_PREFIX}\x1b[90m{text}\x1b[0m\n");
        if let Err(error) = self
            .err
            .write_all(line.as_bytes())
            .and_then(|()| self.err.flush())
        {
            self.error = Some(error);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::{Arc, Mutex};

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
        renderer.on_event(AgentEvent::ToolFinished {
            name: "bash".to_string(),
            arguments: json!({"command": "ls"}),
            content: "done".to_string(),
            is_error: false,
        });
        renderer.finish_turn();

        let text = out.text();
        assert!(text.contains("thinking\n"), "{text:?}");
        assert!(text.contains("> bash(ls)"), "{text:?}");
        assert_eq!(text.matches("bash(ls)").count(), 1, "{text:?}");
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
}
