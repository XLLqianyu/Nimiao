//! Terminal setup, restoration, and the async UI event loop.

use super::app::{TuiApp, TurnMessage};
use super::render;
use crossterm::cursor;
use crossterm::event::{
    DisableBracketedPaste, EnableBracketedPaste, EventStream, KeyboardEnhancementFlags,
    PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, supports_keyboard_enhancement, EnterAlternateScreen,
    LeaveAlternateScreen,
};
use engine::Agent;
use futures_util::StreamExt;
use model::openai::OpenAIModel;
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;
use std::io;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::sync::mpsc;

const TICK: Duration = Duration::from_millis(100);

static TERMINAL_ACTIVE: AtomicBool = AtomicBool::new(false);
static KEYBOARD_ENHANCED: AtomicBool = AtomicBool::new(false);

/// Runs the full-screen client until the user quits or the input stream ends.
pub(super) async fn run(
    agent: Agent<OpenAIModel>,
    model_name: String,
    workspace: PathBuf,
) -> anyhow::Result<()> {
    let _guard = TerminalGuard::enter()?;
    install_panic_hook();

    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;

    let (tx, mut rx) = mpsc::unbounded_channel::<TurnMessage<OpenAIModel>>();
    let mut app = TuiApp::new(agent, model_name, workspace);
    let mut events = EventStream::new();
    let mut ticker = tokio::time::interval(TICK);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        terminal.draw(|frame| render::draw(frame, &mut app))?;
        if app.quit {
            break;
        }

        tokio::select! {
            _ = ticker.tick() => app.tick(),
            event = events.next() => match event {
                Some(Ok(event)) => app.handle_terminal(event, &tx),
                Some(Err(error)) => return Err(error.into()),
                None => app.input_closed(),
            },
            message = rx.recv() => {
                if let Some(message) = message {
                    app.handle_turn(message, &tx);
                }
            }
        }
    }

    Ok(())
}

/// Owns the terminal modes that must be undone on both normal exit and panic.
struct TerminalGuard;

impl TerminalGuard {
    fn enter() -> anyhow::Result<Self> {
        if TERMINAL_ACTIVE.swap(true, Ordering::SeqCst) {
            anyhow::bail!("the TUI terminal is already active");
        }

        let guard = Self;
        enable_raw_mode()?;
        execute!(io::stdout(), EnterAlternateScreen)?;
        execute!(io::stdout(), EnableBracketedPaste)?;

        if supports_keyboard_enhancement().unwrap_or(false)
            && execute!(
                io::stdout(),
                PushKeyboardEnhancementFlags(
                    KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
                        | KeyboardEnhancementFlags::REPORT_EVENT_TYPES
                )
            )
            .is_ok()
        {
            KEYBOARD_ENHANCED.store(true, Ordering::SeqCst);
        }

        Ok(guard)
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        restore_terminal();
    }
}

fn install_panic_hook() {
    static INSTALLED: AtomicBool = AtomicBool::new(false);
    if INSTALLED.swap(true, Ordering::SeqCst) {
        return;
    }

    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        restore_terminal();
        previous(info);
    }));
}

/// Restores the terminal at most once, regardless of how shutdown was reached.
fn restore_terminal() {
    if !TERMINAL_ACTIVE.swap(false, Ordering::SeqCst) {
        return;
    }

    let mut out = io::stdout();
    if KEYBOARD_ENHANCED.swap(false, Ordering::SeqCst) {
        let _ = execute!(out, PopKeyboardEnhancementFlags);
    }
    let _ = execute!(
        out,
        cursor::Show,
        DisableBracketedPaste,
        LeaveAlternateScreen
    );
    let _ = disable_raw_mode();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restore_is_idempotent() {
        TERMINAL_ACTIVE.store(false, Ordering::SeqCst);
        restore_terminal();
        restore_terminal();
        assert!(!TERMINAL_ACTIVE.load(Ordering::SeqCst));
    }
}
