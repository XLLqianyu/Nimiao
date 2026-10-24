//! Agent loop.
//!
//! One [`Agent`] binds a model to a tool set and an execution context. A turn
//! appends the user message to the conversation, asks the model for the next
//! step, runs whatever tools it asks for, and repeats until the model answers
//! without further tool calls.

use model::{Model, ModelEvent, ModelResponse, StopReason};
use protocol::{Item, ToolCall};
use std::future::Future;
use tools::{ToolContext, ToolRegistry};

/// Upper bound on tool-calling rounds in a single turn, so a model that never
/// stops asking for tools cannot loop forever.
const MAX_STEPS_PER_TURN: usize = 32;

/// Reported while a turn runs, for rendering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentEvent {
    /// A fragment of the reply text, in order.
    Text(String),
    /// A tool is about to run.
    ToolStarted {
        name: String,
        arguments: serde_json::Value,
    },
    /// A tool finished; `content` is the result recorded in history.
    ToolFinished {
        name: String,
        arguments: serde_json::Value,
        content: String,
        is_error: bool,
    },
    /// Something worth telling the user that is not part of the reply.
    Notice(String),
}

/// How a turn ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnOutcome {
    Completed,
    Interrupted,
}

/// A model bound to a tool set and an execution context.
pub struct Agent<M: Model> {
    model: M,
    tools: ToolRegistry,
    ctx: ToolContext,
}

impl<M: Model> Agent<M> {
    /// Builds an agent.
    ///
    /// Arguments are, in order: the model that produces replies, the registry
    /// of tools it may call, and the context every tool executes against.
    pub fn new(model: M, tools: ToolRegistry, ctx: ToolContext) -> Self {
        Self { model, tools, ctx }
    }

    pub fn tools(&self) -> &ToolRegistry {
        &self.tools
    }

    pub fn context(&self) -> &ToolContext {
        &self.ctx
    }

    /// Runs one user turn to completion and records every step in `history`,
    /// reporting progress to `on_event` as it happens.
    ///
    /// Returns after the model produced a reply with no tool calls. The final
    /// reply is the last [`Item::AssistantMessage`] appended to `history`.
    pub async fn run_turn_stream(
        &self,
        history: &mut Vec<Item>,
        input: impl Into<String>,
        on_event: &mut (dyn FnMut(AgentEvent) + Send),
    ) -> anyhow::Result<()> {
        self.run_turn_stream_cancellable(history, input, on_event, std::future::pending::<()>())
            .await?;
        Ok(())
    }

    /// Runs one user turn until it completes or `cancel` resolves.
    ///
    /// Cancellation is cooperative: the in-flight model request or tool future
    /// is dropped. Text already emitted for an unfinished model step is not
    /// written to history.
    pub async fn run_turn_stream_cancellable<C>(
        &self,
        history: &mut Vec<Item>,
        input: impl Into<String>,
        on_event: &mut (dyn FnMut(AgentEvent) + Send),
        cancel: C,
    ) -> anyhow::Result<TurnOutcome>
    where
        C: Future<Output = ()> + Send,
    {
        let mut cancel = std::pin::pin!(cancel);

        // Observe an already-completed cancellation future before touching
        // history. The ready arm makes the select return immediately when the
        // cancellation future is still pending.
        tokio::select! {
            biased;
            _ = &mut cancel => return Ok(TurnOutcome::Interrupted),
            _ = std::future::ready(()) => {}
        }

        history.push(Item::UserMessage(input.into()));

        let definitions = self.tools.definitions();
        // A non-streaming endpoint fails to stream on every tool round; say so
        // once per turn rather than once per round.
        let mut notified = false;

        for _ in 0..MAX_STEPS_PER_TURN {
            let mut sink = |event: ModelEvent| match event {
                ModelEvent::Text(text) => on_event(AgentEvent::Text(text)),
                ModelEvent::Fallback => {
                    if !notified {
                        notified = true;
                        on_event(AgentEvent::Notice(
                            "upstream did not stream; fell back to a one-shot reply".to_string(),
                        ));
                    }
                }
            };
            let ModelResponse {
                content,
                tool_calls,
                stop_reason,
            } = tokio::select! {
                biased;
                _ = &mut cancel => return Ok(TurnOutcome::Interrupted),
                response = self
                    .model
                    .complete_stream(history, &definitions, &mut sink) => response?,
            };

            if stop_reason == Some(StopReason::ContentFilter) {
                anyhow::bail!("Provider finish_reason: content_filter");
            }

            let text = content.filter(|text| !text.trim().is_empty());

            // A reply cut off at the token limit may carry calls whose
            // arguments were truncated mid-word. None of them is trustworthy,
            // so record every one as failed instead of running any.
            if stop_reason == Some(StopReason::Length) && !tool_calls.is_empty() {
                for call in tool_calls {
                    record_unexecuted(
                        history,
                        call,
                        "the response hit the output token limit, so its arguments may be truncated",
                        on_event,
                    );
                }
                anyhow::bail!("the reply was truncated before the tool calls were complete");
            }

            if stop_reason == Some(StopReason::Length) {
                on_event(AgentEvent::Notice(
                    "the reply hit the output token limit and was cut short".to_string(),
                ));
            }

            if tool_calls.is_empty() {
                let Some(text) = text else {
                    anyhow::bail!("the model returned an empty response");
                };
                history.push(Item::AssistantMessage(text));
                return Ok(TurnOutcome::Completed);
            }

            if let Some(text) = text {
                history.push(Item::AssistantMessage(text));
            }

            for call in tool_calls {
                tokio::select! {
                    biased;
                    _ = &mut cancel => return Ok(TurnOutcome::Interrupted),
                    () = self.run_tool(history, call, on_event) => {}
                }
            }
        }

        anyhow::bail!(
            "the model called tools {MAX_STEPS_PER_TURN} times without finishing the turn"
        )
    }

    /// Runs one user turn without reporting progress.
    pub async fn run_turn(
        &self,
        history: &mut Vec<Item>,
        input: impl Into<String>,
    ) -> anyhow::Result<()> {
        self.run_turn_stream(history, input, &mut |_| {}).await
    }

    /// Executes one tool call and records the call together with its result.
    async fn run_tool(
        &self,
        history: &mut Vec<Item>,
        call: ToolCall,
        on_event: &mut (dyn FnMut(AgentEvent) + Send),
    ) {
        let arguments = call.arguments.clone();
        on_event(AgentEvent::ToolStarted {
            name: call.name.clone(),
            arguments: arguments.clone(),
        });
        // Malformed arguments never reach the tool: it would only fail on a
        // schema it cannot see. The text the model sent goes back to it
        // instead, so it can retry with something parseable.
        let (content, is_error) = match call.arguments_error.as_deref() {
            Some(error) => (
                format!("Invalid arguments for tool \"{}\": {error}", call.name),
                true,
            ),
            None => match self.tools.get(&call.name) {
                Some(tool) => match tool.execute(arguments.clone(), &self.ctx).await {
                    Ok(output) => (output.content, output.is_error),
                    Err(err) => (format!("{err:#}"), true),
                },
                None => (format!("unknown tool '{}'", call.name), true),
            },
        };

        history.push(Item::ToolCall(call.clone()));
        history.push(Item::ToolResult {
            tool_call_id: call.id,
            content: content.clone(),
            is_error,
        });
        on_event(AgentEvent::ToolFinished {
            name: call.name,
            arguments,
            content,
            is_error,
        });
    }
}

/// Records a call the model declared but that must not run, with a failure
/// result explaining why.
///
/// The call still goes into history: a history in which a declared call has no
/// result is rejected by providers, so the failure result is what keeps the
/// transcript replayable.
fn record_unexecuted(
    history: &mut Vec<Item>,
    call: ToolCall,
    reason: &str,
    on_event: &mut (dyn FnMut(AgentEvent) + Send),
) {
    let content = format!(
        "Tool call \"{}\" was not executed: {reason}. Re-issue the tool call with complete arguments.",
        call.name
    );
    on_event(AgentEvent::ToolStarted {
        name: call.name.clone(),
        arguments: call.arguments.clone(),
    });
    history.push(Item::ToolCall(call.clone()));
    history.push(Item::ToolResult {
        tool_call_id: call.id.clone(),
        content: content.clone(),
        is_error: true,
    });
    on_event(AgentEvent::ToolFinished {
        name: call.name,
        arguments: call.arguments,
        content,
        is_error: true,
    });
}
