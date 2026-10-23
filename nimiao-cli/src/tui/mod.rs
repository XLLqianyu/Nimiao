//! Full-screen ratatui client.
//!
//! The TUI owns presentation state only. It drives the existing `Agent` in a
//! short-lived turn task and deliberately has no protocol or service layer.

mod app;
mod markdown;
mod render;
mod terminal;

use engine::Agent;
use model::openai::OpenAIModel;
use std::path::PathBuf;

/// Starts the full-screen client.
pub(crate) async fn run(
    agent: Agent<OpenAIModel>,
    model_name: String,
    workspace: PathBuf,
) -> anyhow::Result<()> {
    terminal::run(agent, model_name, workspace).await
}
