//! TUI state, input handling, queueing, and the short-lived turn task.

use crate::tool_preview::tool_preview;
use engine::{Agent, AgentEvent, TurnOutcome};
use model::Model;
use protocol::Item;
use ratatui::crossterm::event::{Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Line;
use ratatui_textarea::{CursorMove, TextArea, WrapMode};
use serde_json::Value;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, watch};
use unicode_width::UnicodeWidthChar;

const COMPOSER_MAX_ROWS: usize = 6;

/// A running agent together with the history it owns.
pub(crate) struct Session<M: Model> {
    pub(crate) agent: Agent<M>,
    pub(crate) history: Vec<Item>,
}

/// Messages produced by a turn task.
pub(crate) enum TurnMessage<M: Model> {
    Event(AgentEvent),
    Finished {
        session: Session<M>,
        result: anyhow::Result<TurnOutcome>,
    },
}

pub(crate) type MessageSender<M> = mpsc::UnboundedSender<TurnMessage<M>>;

pub(crate) struct ActiveTurn {
    pub(crate) cancel: watch::Sender<bool>,
    pub(crate) started: Instant,
    pub(crate) current_tool: Option<String>,
    pub(crate) interrupt_requested: bool,
    pub(crate) merge_on_interrupt: bool,
}

/// Scroll position shared by the live transcript and the detail page.
pub(crate) struct ViewportState {
    pub(crate) top: usize,
    pub(crate) follow: bool,
    pub(crate) max_top: usize,
    pub(crate) height: usize,
}

impl Default for ViewportState {
    fn default() -> Self {
        Self {
            top: 0,
            follow: true,
            max_top: 0,
            height: 1,
        }
    }
}

impl ViewportState {
    pub(crate) fn page_up(&mut self) {
        self.follow = false;
        self.top = self.top.saturating_sub(self.height.max(1));
    }

    pub(crate) fn page_down(&mut self) {
        self.top = self
            .top
            .saturating_add(self.height.max(1))
            .min(self.max_top);
        self.follow = self.top >= self.max_top;
    }

    pub(crate) fn line_up(&mut self) {
        self.follow = false;
        self.top = self.top.saturating_sub(1);
    }

    pub(crate) fn line_down(&mut self) {
        self.top = self.top.saturating_add(1).min(self.max_top);
        self.follow = self.top >= self.max_top;
    }

    pub(crate) fn half_up(&mut self) {
        self.follow = false;
        self.top = self.top.saturating_sub(self.height.max(1).div_ceil(2));
    }

    pub(crate) fn half_down(&mut self) {
        self.top = self
            .top
            .saturating_add(self.height.max(1).div_ceil(2))
            .min(self.max_top);
        self.follow = self.top >= self.max_top;
    }

    pub(crate) fn jump_top(&mut self) {
        self.follow = false;
        self.top = 0;
    }

    pub(crate) fn jump_bottom(&mut self) {
        self.follow = true;
        self.top = self.max_top;
    }

    pub(crate) fn resolve(&mut self, total: usize, height: usize) {
        self.height = height.max(1);
        self.max_top = total.saturating_sub(self.height);
        if self.follow {
            self.top = self.max_top;
        } else {
            self.top = self.top.min(self.max_top);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolStatus {
    Running,
    Ran,
    Failed,
    Interrupted,
}

pub(crate) struct ToolRow {
    pub(crate) name: String,
    pub(crate) arguments: Value,
    pub(crate) status: ToolStatus,
    pub(crate) content: String,
    pub(crate) is_error: bool,
    pub(crate) started: Instant,
    pub(crate) elapsed: Option<Duration>,
}

impl ToolRow {
    pub(crate) fn preview(&self) -> String {
        tool_preview(&self.arguments)
    }
}

pub(crate) enum CellBody {
    Header {
        version: String,
        model: String,
        workspace: String,
    },
    User(String),
    Assistant {
        text: String,
        interrupted: bool,
    },
    Tool(ToolRow),
    Notice(String),
    Error(String),
    Help(String),
}

pub(crate) struct CachedLines {
    pub(crate) width: u16,
    pub(crate) revision: u64,
    pub(crate) lines: Vec<Line<'static>>,
}

pub(crate) struct Cell {
    pub(crate) body: CellBody,
    pub(crate) revision: u64,
    pub(crate) main_cache: Option<CachedLines>,
    pub(crate) detail_cache: Option<CachedLines>,
}

impl Cell {
    pub(crate) fn new(body: CellBody) -> Self {
        Self {
            body,
            revision: 0,
            main_cache: None,
            detail_cache: None,
        }
    }

    pub(crate) fn bump(&mut self) {
        self.revision = self.revision.wrapping_add(1);
    }
}

pub(crate) struct TuiApp<M: Model> {
    pub(crate) session: Option<Session<M>>,
    pub(crate) cells: Vec<Cell>,
    pub(crate) composer: TextArea<'static>,
    pub(crate) queue: VecDeque<String>,
    pub(crate) queue_paused: bool,
    pub(crate) active: Option<ActiveTurn>,
    pub(crate) detail: Option<ViewportState>,
    pub(crate) live: ViewportState,
    pub(crate) quit: bool,
    assistant_open: bool,
    model_name: String,
    workspace: String,
}

impl<M: Model + 'static> TuiApp<M> {
    pub(crate) fn new(agent: Agent<M>, model_name: String, workspace: PathBuf) -> Self {
        let model = model_name.clone();
        let workspace = workspace.display().to_string();
        Self {
            session: Some(Session {
                agent,
                history: Vec::new(),
            }),
            cells: vec![Cell::new(CellBody::Header {
                version: env!("CARGO_PKG_VERSION").to_string(),
                model,
                workspace: workspace.clone(),
            })],
            composer: new_composer(),
            queue: VecDeque::new(),
            queue_paused: false,
            active: None,
            detail: None,
            live: ViewportState::default(),
            quit: false,
            assistant_open: false,
            model_name,
            workspace,
        }
    }

    pub(crate) fn handle_terminal(&mut self, event: Event, tx: &MessageSender<M>) {
        match event {
            Event::Key(key) if key.kind != KeyEventKind::Release => self.handle_key(key, tx),
            Event::Paste(text) => {
                self.composer
                    .insert_str(crate::tui::markdown::sanitize(&text));
            }
            Event::Resize(_, _) => {}
            _ => {}
        }
    }

    pub(crate) fn handle_turn(&mut self, message: TurnMessage<M>, tx: &MessageSender<M>) {
        match message {
            TurnMessage::Event(event) => self.apply_event(event),
            TurnMessage::Finished { session, result } => {
                if let Some(next) = self.finish_turn(session, result) {
                    self.start_turn(next, tx);
                }
            }
        }
    }

    pub(crate) fn input_closed(&mut self) {
        self.request_quit();
    }

    pub(crate) fn tick(&mut self) {}

    pub(crate) fn composer_text(&self) -> String {
        self.composer.lines().join("\n")
    }

    pub(crate) fn composer_height(&self, width: u16) -> u16 {
        let inner = usize::from(width.saturating_sub(2).max(1));
        let rows = self
            .composer_text()
            .split('\n')
            .map(|line| wrapped_rows(line, inner))
            .sum::<usize>()
            .clamp(1, COMPOSER_MAX_ROWS);
        (rows + 2) as u16
    }

    fn clear_composer(&mut self) {
        self.composer = new_composer();
    }

    fn set_composer(&mut self, text: String) {
        self.composer = new_composer();
        self.composer.insert_str(text);
        self.composer.move_cursor(CursorMove::Bottom);
        self.composer.move_cursor(CursorMove::End);
    }
}

fn new_composer() -> TextArea<'static> {
    let mut composer = TextArea::default();
    composer.set_wrap_mode(WrapMode::WordOrGlyph);
    composer.set_style(Style::default().fg(Color::White));
    composer.set_cursor_style(
        Style::default()
            .fg(Color::Black)
            .bg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
    );
    composer.set_placeholder_text("Send a request…");
    composer.set_placeholder_style(Style::default().fg(Color::DarkGray));
    composer
}

fn wrapped_rows(line: &str, width: usize) -> usize {
    if line.is_empty() {
        return 1;
    }
    let mut rows = 1;
    let mut used = 0;
    for ch in line.chars() {
        let char_width = UnicodeWidthChar::width(ch).unwrap_or(0);
        if used + char_width > width && used > 0 {
            rows += 1;
            used = 0;
        }
        used += char_width;
    }
    rows
}

impl<M: Model + 'static> TuiApp<M> {
    fn handle_key(&mut self, key: KeyEvent, tx: &MessageSender<M>) {
        if ctrl(key, 't') {
            if self.detail.is_some() {
                self.detail = None;
            } else {
                self.detail = Some(ViewportState::default());
            }
            return;
        }

        if let Some(detail) = self.detail.as_mut() {
            if ctrl(key, 'c') || (key.code == KeyCode::Char('q') && key.modifiers.is_empty()) {
                self.detail = None;
                return;
            }
            if ctrl(key, 'b') {
                detail.page_up();
            } else if ctrl(key, 'f') {
                detail.page_down();
            } else if ctrl(key, 'u') {
                detail.half_up();
            } else if ctrl(key, 'd') {
                detail.half_down();
            } else {
                match key.code {
                    KeyCode::Up | KeyCode::Char('k') => detail.line_up(),
                    KeyCode::Down | KeyCode::Char('j') => detail.line_down(),
                    KeyCode::PageUp => detail.page_up(),
                    KeyCode::PageDown => detail.page_down(),
                    KeyCode::Char(' ') if key.modifiers.contains(KeyModifiers::SHIFT) => {
                        detail.page_up();
                    }
                    KeyCode::Char(' ') if key.modifiers.is_empty() => detail.page_down(),
                    KeyCode::Home => detail.jump_top(),
                    KeyCode::End => detail.jump_bottom(),
                    _ => {}
                }
            }
            return;
        }

        if ctrl(key, 'c') {
            if self.active.is_some() {
                self.interrupt(true);
            } else {
                self.quit = true;
            }
            return;
        }
        if ctrl(key, 'd') {
            if self.active.is_none() && self.composer_text().is_empty() {
                self.quit = true;
            } else if self.active.is_none() {
                self.composer.input(key);
            }
            return;
        }
        if key.code == KeyCode::Esc && key.modifiers.is_empty() {
            if self.active.is_some() {
                self.interrupt(true);
            }
            return;
        }

        if key.code == KeyCode::PageUp && key.modifiers.is_empty() {
            self.live.page_up();
            return;
        }
        if key.code == KeyCode::PageDown && key.modifiers.is_empty() {
            self.live.page_down();
            return;
        }
        if key.code == KeyCode::Home && key.modifiers.contains(KeyModifiers::CONTROL) {
            self.live.jump_top();
            return;
        }
        if key.code == KeyCode::End && key.modifiers.contains(KeyModifiers::CONTROL) {
            self.live.jump_bottom();
            return;
        }

        if key.code == KeyCode::Enter && key.modifiers.is_empty() {
            self.submit(tx);
            return;
        }
        if key.code == KeyCode::Tab && key.modifiers.is_empty() {
            if self.active.is_some() {
                self.queue_current();
            } else {
                self.submit(tx);
            }
            return;
        }
        if (key.code == KeyCode::Enter
            && (key.modifiers.contains(KeyModifiers::SHIFT)
                || key.modifiers.contains(KeyModifiers::ALT)))
            || ctrl(key, 'j')
            || ctrl(key, 'm')
        {
            self.composer.insert_newline();
            return;
        }

        self.composer.input(key);
    }

    fn submit(&mut self, tx: &MessageSender<M>) {
        let raw = self.composer_text();
        let text = raw.trim().to_string();
        if text.is_empty() {
            if self.active.is_none() && self.queue_paused {
                if let Some(next) = self.queue.pop_front() {
                    self.queue_paused = false;
                    self.start_turn(next, tx);
                }
            }
            return;
        }

        if self.execute_command(&text) {
            self.clear_composer();
            return;
        }

        if self.active.is_some() {
            self.queue.push_back(text);
            self.clear_composer();
            return;
        }

        self.queue_paused = false;
        self.live.follow = true;
        self.start_turn(text, tx);
        self.clear_composer();
    }

    fn queue_current(&mut self) {
        let text = self.composer_text().trim().to_string();
        if text.is_empty() {
            return;
        }
        self.queue.push_back(text);
        self.clear_composer();
    }

    fn execute_command(&mut self, text: &str) -> bool {
        match text.to_ascii_lowercase().as_str() {
            "/quit" | "exit" | "quit" => {
                self.request_quit();
                true
            }
            "/help" => {
                self.push_cell(CellBody::Help(help_text()));
                true
            }
            "/clear" => {
                if self.active.is_some() {
                    self.push_notice("Interrupt the current turn before clearing.");
                } else {
                    self.clear_session();
                }
                true
            }
            _ => false,
        }
    }

    fn request_quit(&mut self) {
        if let Some(active) = &self.active {
            let _ = active.cancel.send(true);
        }
        self.quit = true;
    }

    fn interrupt(&mut self, merge: bool) {
        if let Some(active) = &mut self.active {
            active.interrupt_requested = true;
            active.merge_on_interrupt = merge;
            let _ = active.cancel.send(true);
        }
    }

    fn clear_session(&mut self) {
        if let Some(session) = &mut self.session {
            session.history.clear();
        }
        self.queue.clear();
        self.queue_paused = false;
        self.detail = None;
        self.live = ViewportState::default();
        self.assistant_open = false;
        self.cells = vec![Cell::new(CellBody::Header {
            version: env!("CARGO_PKG_VERSION").to_string(),
            model: self.model_name.clone(),
            workspace: self.workspace.clone(),
        })];
    }

    fn start_turn(&mut self, text: String, tx: &MessageSender<M>) {
        let Some(session) = self.session.take() else {
            self.push_error("the agent is not available");
            return;
        };

        self.finish_assistant(false);
        self.cells.push(Cell::new(CellBody::User(text.clone())));
        self.assistant_open = false;
        self.live.follow = true;

        let (cancel, cancel_rx) = watch::channel(false);
        self.active = Some(ActiveTurn {
            cancel,
            started: Instant::now(),
            current_tool: None,
            interrupt_requested: false,
            merge_on_interrupt: false,
        });

        let tx = tx.clone();
        tokio::spawn(async move {
            let Session { agent, mut history } = session;
            let mut cancel_rx = cancel_rx;
            let cancelled = async move {
                loop {
                    if *cancel_rx.borrow() {
                        return;
                    }
                    if cancel_rx.changed().await.is_err() {
                        return;
                    }
                }
            };
            let mut sink = |event| {
                let _ = tx.send(TurnMessage::Event(event));
            };
            let result = agent
                .run_turn_stream_cancellable(&mut history, text, &mut sink, cancelled)
                .await;
            let _ = tx.send(TurnMessage::Finished {
                session: Session { agent, history },
                result,
            });
        });
    }

    fn finish_turn(
        &mut self,
        session: Session<M>,
        result: anyhow::Result<TurnOutcome>,
    ) -> Option<String> {
        let active = self.active.take()?;
        self.session = Some(session);

        match result {
            Ok(TurnOutcome::Completed) => {
                self.finish_assistant(false);
            }
            Ok(TurnOutcome::Interrupted) => {
                let marked_assistant = self.finish_assistant(true);
                let marked_tool = self.interrupt_running_tool();
                if !marked_assistant && !marked_tool {
                    self.push_cell(CellBody::Notice("Interrupted".to_string()));
                }
                self.queue_paused = false;
                self.live.follow = true;
                if active.merge_on_interrupt {
                    self.merge_queue_and_draft();
                }
            }
            Err(error) => {
                self.finish_assistant(true);
                self.fail_running_tool("the turn failed");
                self.push_error(format!("{error:#}"));
                self.queue_paused = !self.queue.is_empty();
                self.live.follow = true;
            }
        }

        if !self.queue_paused && self.active.is_none() {
            self.queue.pop_front()
        } else {
            None
        }
    }

    fn merge_queue_and_draft(&mut self) {
        let had_queue = !self.queue.is_empty();
        let mut parts: Vec<String> = self.queue.drain(..).collect();
        let draft = self.composer_text().trim().to_string();
        if !draft.is_empty() {
            parts.push(draft);
        }
        if had_queue && !parts.is_empty() {
            self.set_composer(parts.join("\n"));
        }
    }

    fn apply_event(&mut self, event: AgentEvent) {
        match event {
            AgentEvent::Text(delta) => self.append_text(&delta),
            AgentEvent::ToolStarted { name, arguments } => {
                self.finish_assistant(false);
                if let Some(active) = &mut self.active {
                    active.current_tool = Some(name.clone());
                }
                self.cells.push(Cell::new(CellBody::Tool(ToolRow {
                    name,
                    arguments,
                    status: ToolStatus::Running,
                    content: String::new(),
                    is_error: false,
                    started: Instant::now(),
                    elapsed: None,
                })));
            }
            AgentEvent::ToolFinished {
                name,
                arguments,
                content,
                is_error,
            } => {
                if let Some(active) = &mut self.active {
                    active.current_tool = None;
                }
                let mut found = false;
                for cell in self.cells.iter_mut().rev() {
                    if let CellBody::Tool(tool) = &mut cell.body {
                        if tool.status == ToolStatus::Running {
                            tool.status = if is_error {
                                ToolStatus::Failed
                            } else {
                                ToolStatus::Ran
                            };
                            tool.content = content.clone();
                            tool.is_error = is_error;
                            tool.elapsed = Some(tool.started.elapsed());
                            cell.bump();
                            found = true;
                            break;
                        }
                    }
                }
                if !found {
                    self.cells.push(Cell::new(CellBody::Tool(ToolRow {
                        name,
                        arguments,
                        status: if is_error {
                            ToolStatus::Failed
                        } else {
                            ToolStatus::Ran
                        },
                        content,
                        is_error,
                        started: Instant::now(),
                        elapsed: Some(Duration::ZERO),
                    })));
                }
            }
            AgentEvent::Notice(text) => self.push_notice(text),
        }
    }

    fn append_text(&mut self, delta: &str) {
        if delta.is_empty() {
            return;
        }
        if !self.assistant_open {
            self.cells.push(Cell::new(CellBody::Assistant {
                text: String::new(),
                interrupted: false,
            }));
            self.assistant_open = true;
        }
        if let Some(cell) = self.cells.last_mut() {
            if let CellBody::Assistant { text, .. } = &mut cell.body {
                text.push_str(delta);
                cell.bump();
            }
        }
    }

    fn finish_assistant(&mut self, interrupted: bool) -> bool {
        if !self.assistant_open {
            return false;
        }
        self.assistant_open = false;
        let Some(cell) = self.cells.last_mut() else {
            return false;
        };
        let CellBody::Assistant {
            text,
            interrupted: was_interrupted,
        } = &mut cell.body
        else {
            return false;
        };
        if text.trim().is_empty() {
            return false;
        }
        if interrupted {
            *was_interrupted = true;
            cell.bump();
        }
        true
    }

    fn interrupt_running_tool(&mut self) -> bool {
        for cell in self.cells.iter_mut().rev() {
            if let CellBody::Tool(tool) = &mut cell.body {
                if tool.status == ToolStatus::Running {
                    tool.status = ToolStatus::Interrupted;
                    tool.elapsed = Some(tool.started.elapsed());
                    cell.bump();
                    return true;
                }
            }
        }
        false
    }

    fn fail_running_tool(&mut self, reason: &str) {
        for cell in self.cells.iter_mut().rev() {
            if let CellBody::Tool(tool) = &mut cell.body {
                if tool.status == ToolStatus::Running {
                    tool.status = ToolStatus::Failed;
                    tool.content = reason.to_string();
                    tool.is_error = true;
                    tool.elapsed = Some(tool.started.elapsed());
                    cell.bump();
                    return;
                }
            }
        }
    }

    fn push_notice(&mut self, text: impl Into<String>) {
        self.finish_assistant(false);
        self.push_cell(CellBody::Notice(text.into()));
    }

    fn push_error(&mut self, text: impl Into<String>) {
        self.finish_assistant(false);
        self.push_cell(CellBody::Error(text.into()));
    }

    fn push_cell(&mut self, body: CellBody) {
        self.cells.push(Cell::new(body));
    }
}

fn ctrl(key: KeyEvent, ch: char) -> bool {
    key.modifiers.contains(KeyModifiers::CONTROL)
        && matches!(key.code, KeyCode::Char(c) if c.eq_ignore_ascii_case(&ch))
}

fn help_text() -> String {
    [
        "Nimiao shortcuts",
        "Enter       send",
        "Tab         queue while running; send when idle",
        "Shift+Enter insert a newline",
        "Esc         interrupt the current turn",
        "Ctrl+T      open or close the detailed transcript",
        "PgUp/PgDn   scroll the live transcript",
        "Ctrl+Home   jump to the start",
        "Ctrl+End    jump to the latest output",
        "Ctrl+C      interrupt; quit when idle",
        "Ctrl+D      quit when the composer is empty",
        "",
        "Commands: /help  /clear  /quit",
    ]
    .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::render;
    use model::openai::OpenAIModel;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;
    use tools::{ToolContext, ToolRegistry};

    fn test_app() -> TuiApp<OpenAIModel> {
        let model = OpenAIModel::new("test", "local", "test-model").unwrap();
        let ctx = ToolContext::new(sandbox::Sandbox::current().unwrap());
        let agent = Agent::new(model, ToolRegistry::new(Vec::new()), ctx);
        TuiApp::new(agent, "test-model".to_string(), std::env::temp_dir())
    }

    fn active_turn() -> ActiveTurn {
        ActiveTurn {
            cancel: watch::channel(false).0,
            started: Instant::now(),
            current_tool: None,
            interrupt_requested: false,
            merge_on_interrupt: false,
        }
    }

    fn key(code: KeyCode, modifiers: KeyModifiers) -> KeyEvent {
        KeyEvent::new(code, modifiers)
    }

    fn screen(app: &mut TuiApp<OpenAIModel>, width: u16, height: u16) -> String {
        let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
        terminal
            .draw(|frame| render::draw(frame, app))
            .expect("draw the TUI");
        let buffer = terminal.backend().buffer();
        buffer
            .content
            .chunks(usize::from(width))
            .map(|row| {
                row.iter()
                    .map(|cell| cell.symbol())
                    .collect::<String>()
                    .trim_end()
                    .to_string()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn last_tool(app: &TuiApp<OpenAIModel>) -> &ToolRow {
        match &app.cells.last().expect("a cell").body {
            CellBody::Tool(tool) => tool,
            _ => panic!("expected a tool cell"),
        }
    }

    #[test]
    fn renders_at_common_and_tiny_terminal_sizes() {
        let mut app = test_app();
        let wide = screen(&mut app, 80, 24);
        assert!(wide.contains("Nimiao v"), "{wide}");
        assert!(wide.contains("Send a request…"), "{wide}");
        assert!(wide.contains('›'), "{wide}");

        app.cells.push(Cell::new(CellBody::Assistant {
            text: "# Report\n\n- first item".to_string(),
            interrupted: false,
        }));
        let large = screen(&mut app, 120, 40);
        assert!(large.contains("Report"), "{large}");
        assert!(large.contains("• first item"), "{large}");

        for (width, height) in [(24, 6), (12, 3), (1, 1)] {
            let _ = screen(&mut app, width, height);
        }
    }

    #[test]
    fn renders_the_queue_and_active_status() {
        let mut app = test_app();
        app.active = Some(active_turn());
        app.queue.push_back("first".to_string());
        app.queue.push_back("second".to_string());

        let screen = screen(&mut app, 80, 24);
        assert!(screen.contains("2 queued"), "{screen}");
        assert!(screen.contains("↳ first"), "{screen}");
        assert!(screen.contains("Working"), "{screen}");
    }

    #[test]
    fn tool_rows_follow_running_ran_failed_and_interrupted_states() {
        let mut app = test_app();
        let arguments = serde_json::json!({"command": "ls"});

        app.apply_event(AgentEvent::ToolStarted {
            name: "bash".to_string(),
            arguments: arguments.clone(),
        });
        assert_eq!(last_tool(&app).status, ToolStatus::Running);

        app.apply_event(AgentEvent::ToolFinished {
            name: "bash".to_string(),
            arguments: arguments.clone(),
            content: "ok".to_string(),
            is_error: false,
        });
        assert_eq!(last_tool(&app).status, ToolStatus::Ran);
        assert_eq!(last_tool(&app).content, "ok");
        assert!(last_tool(&app).elapsed.is_some());

        app.apply_event(AgentEvent::ToolStarted {
            name: "bash".to_string(),
            arguments: arguments.clone(),
        });
        app.apply_event(AgentEvent::ToolFinished {
            name: "bash".to_string(),
            arguments,
            content: "bad".to_string(),
            is_error: true,
        });
        assert_eq!(last_tool(&app).status, ToolStatus::Failed);
        assert!(last_tool(&app).is_error);

        app.apply_event(AgentEvent::ToolStarted {
            name: "bash".to_string(),
            arguments: serde_json::json!({"command": "sleep"}),
        });
        assert!(app.interrupt_running_tool());
        assert_eq!(last_tool(&app).status, ToolStatus::Interrupted);
    }

    #[test]
    fn detail_and_live_viewports_handle_their_documented_keys() {
        let mut app = test_app();
        let (tx, _rx) = mpsc::unbounded_channel();

        app.handle_key(key(KeyCode::Char('t'), KeyModifiers::CONTROL), &tx);
        let detail = app.detail.as_mut().expect("detail page");
        detail.resolve(30, 10);
        detail.jump_top();

        app.handle_key(key(KeyCode::Down, KeyModifiers::NONE), &tx);
        assert_eq!(app.detail.as_ref().unwrap().top, 1);
        app.handle_key(key(KeyCode::PageDown, KeyModifiers::NONE), &tx);
        assert_eq!(app.detail.as_ref().unwrap().top, 11);
        app.handle_key(key(KeyCode::Home, KeyModifiers::NONE), &tx);
        assert_eq!(app.detail.as_ref().unwrap().top, 0);
        app.handle_key(key(KeyCode::End, KeyModifiers::NONE), &tx);
        assert_eq!(app.detail.as_ref().unwrap().top, 20);
        assert!(app.detail.as_ref().unwrap().follow);
        app.handle_key(key(KeyCode::Char('q'), KeyModifiers::NONE), &tx);
        assert!(app.detail.is_none());

        app.live.resolve(30, 10);
        app.handle_key(key(KeyCode::PageUp, KeyModifiers::NONE), &tx);
        assert_eq!(app.live.top, 10);
        assert!(!app.live.follow);
        app.handle_key(key(KeyCode::Home, KeyModifiers::CONTROL), &tx);
        assert_eq!(app.live.top, 0);
        app.handle_key(key(KeyCode::End, KeyModifiers::CONTROL), &tx);
        assert_eq!(app.live.top, 20);
        assert!(app.live.follow);
    }

    #[test]
    fn ctrl_shortcuts_ignore_the_case_reported_by_the_platform() {
        let mut app = test_app();
        let (tx, _rx) = mpsc::unbounded_channel();

        app.handle_key(key(KeyCode::Char('T'), KeyModifiers::CONTROL), &tx);
        assert!(app.detail.is_some(), "uppercase Ctrl+T should open details");
        let detail_screen = screen(&mut app, 80, 24);
        assert!(detail_screen.contains("Transcript"), "{detail_screen}");

        let detail = app.detail.as_mut().expect("detail page");
        detail.resolve(30, 10);
        detail.jump_top();
        app.handle_key(key(KeyCode::Char('F'), KeyModifiers::CONTROL), &tx);
        assert_eq!(app.detail.as_ref().unwrap().top, 10);
        app.handle_key(key(KeyCode::Char('B'), KeyModifiers::CONTROL), &tx);
        assert_eq!(app.detail.as_ref().unwrap().top, 0);

        app.handle_key(key(KeyCode::Char('T'), KeyModifiers::CONTROL), &tx);
        assert!(
            app.detail.is_none(),
            "uppercase Ctrl+T should close details"
        );
        let main_screen = screen(&mut app, 80, 24);
        assert!(!main_screen.contains("Transcript"), "{main_screen}");
    }

    #[test]
    fn clear_only_runs_when_idle_and_help_is_added_inline() {
        let mut app = test_app();
        let (tx, _rx) = mpsc::unbounded_channel();

        app.apply_event(AgentEvent::Text("stale".to_string()));
        app.session
            .as_mut()
            .unwrap()
            .history
            .push(Item::UserMessage("stale".to_string()));
        app.queue.push_back("queued".to_string());
        app.composer.insert_str("/clear");
        app.submit(&tx);

        assert_eq!(app.cells.len(), 1);
        assert!(app.session.as_ref().unwrap().history.is_empty());
        assert!(app.queue.is_empty());
        assert!(app.composer_text().is_empty());

        app.composer.insert_str("/help");
        app.submit(&tx);
        assert!(matches!(
            &app.cells.last().unwrap().body,
            CellBody::Help(text) if text.contains("Nimiao shortcuts")
        ));

        app.active = Some(active_turn());
        app.composer.insert_str("/clear");
        app.submit(&tx);
        assert!(matches!(
            &app.cells.last().unwrap().body,
            CellBody::Notice(text) if text.contains("Interrupt")
        ));
        assert!(app.active.is_some());
    }

    #[test]
    fn a_successful_turn_starts_exactly_one_queued_message() {
        let mut app = test_app();
        let session = app.session.take().expect("session");
        app.active = Some(active_turn());
        app.queue.push_back("one".to_string());
        app.queue.push_back("two".to_string());

        let next = app.finish_turn(session, Ok(TurnOutcome::Completed));

        assert_eq!(next.as_deref(), Some("one"));
        assert_eq!(app.queue, ["two"]);
        assert!(!app.queue_paused);
    }

    #[test]
    fn an_error_pauses_the_queue_and_keeps_it() {
        let mut app = test_app();
        let session = app.session.take().expect("session");
        app.active = Some(active_turn());
        app.queue.push_back("later".to_string());

        let next = app.finish_turn(session, Err(anyhow::anyhow!("boom")));

        assert!(next.is_none());
        assert_eq!(app.queue, ["later"]);
        assert!(app.queue_paused);
        assert!(matches!(
            &app.cells.last().unwrap().body,
            CellBody::Error(text) if text.contains("boom")
        ));
    }

    #[test]
    fn interrupting_marks_partial_text_and_merges_queue_into_the_draft() {
        let mut app = test_app();
        let session = app.session.take().expect("session");
        let mut active = active_turn();
        active.merge_on_interrupt = true;
        app.active = Some(active);
        app.apply_event(AgentEvent::Text("partial".to_string()));
        app.queue.push_back("queued".to_string());
        app.composer.insert_str("draft");

        let next = app.finish_turn(session, Ok(TurnOutcome::Interrupted));

        assert!(next.is_none());
        assert!(app.queue.is_empty());
        assert_eq!(app.composer_text(), "queued\ndraft");
        assert!(matches!(
            &app.cells.last().unwrap().body,
            CellBody::Assistant {
                interrupted: true,
                ..
            }
        ));
    }

    #[test]
    fn strips_terminal_escapes_from_dynamic_text() {
        let mut app = test_app();
        let (tx, _rx) = mpsc::unbounded_channel();
        app.cells[0] = Cell::new(CellBody::Header {
            version: "1".to_string(),
            model: "model\u{1b}[31m".to_string(),
            workspace: "dir\u{1b}]0;title\u{7}".to_string(),
        });
        app.active = Some(active_turn());
        app.active.as_mut().unwrap().current_tool = Some("tool\u{1b}[31m".to_string());
        app.queue.push_back("queued\u{1b}[31m".to_string());
        app.apply_event(AgentEvent::ToolStarted {
            name: "bash\u{1b}[31m".to_string(),
            arguments: serde_json::json!({"command": "run"}),
        });

        let screen = screen(&mut app, 80, 24);
        assert!(!screen.contains('\u{1b}'), "{screen:?}");
        assert!(!screen.contains("[31m"), "{screen:?}");
        assert!(!screen.contains("]0;title"), "{screen:?}");

        app.handle_terminal(Event::Paste("a\u{1b}[31mb".to_string()), &tx);
        assert_eq!(app.composer_text(), "ab");
    }
}
