//! Agent loop.
//!
//! One [`Agent`] binds a model to a tool set and an execution context. A turn
//! appends the user message to the conversation, asks the model for the next
//! step, runs whatever tools it asks for, and repeats until the model answers
//! without further tool calls.
//!
//! # History shape
//!
//! History is kept in a shape a provider will accept back: every tool call an
//! assistant step declares is followed immediately by exactly one result, in
//! the order the calls were declared. A call that is never run — because the
//! turn was cancelled, because the reply was truncated, or because its
//! arguments did not parse — still gets a result saying so. A turn that ends
//! without the model's closing message leaves one behind that explains why, so
//! the next turn is never a user message answering a tool result.

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
        let turn_start = history.len() - 1;

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
            let response = tokio::select! {
                biased;
                _ = &mut cancel => {
                    close_open_batch(history, INTERRUPTED);
                    return Ok(TurnOutcome::Interrupted);
                }
                response = self
                    .model
                    .complete_stream(history, &definitions, &mut sink) => response,
            };
            let ModelResponse {
                content,
                tool_calls,
                stop_reason,
            } = match response {
                Ok(response) => response,
                // Nothing of this step reached history, so dropping the user
                // message leaves the conversation exactly as it was: the
                // caller can retry without pushing the same prompt twice.
                Err(error) if history.len() == turn_start + 1 => {
                    history.truncate(turn_start);
                    return Err(error);
                }
                Err(error) => {
                    close_unfinished_turn(history, "the model failed before answering", on_event);
                    return Err(error);
                }
            };

            if stop_reason == Some(StopReason::ContentFilter) {
                note_stopped(history, CONTENT_FILTERED);
                anyhow::bail!("Provider finish_reason: content_filter");
            }

            let text = content.filter(|text| !text.trim().is_empty());

            // A reply cut off at the token limit may carry calls whose
            // arguments were truncated mid-word. None of them is trustworthy,
            // so record every one as failed instead of running any.
            if stop_reason == Some(StopReason::Length) && !tool_calls.is_empty() {
                for call in tool_calls {
                    record_unexecuted(history, call, TRUNCATED, on_event);
                }
                note_stopped(history, TRUNCATED);
                anyhow::bail!("the reply was truncated before the tool calls were complete");
            }

            if stop_reason == Some(StopReason::Length) {
                on_event(AgentEvent::Notice(
                    "the reply hit the output token limit and was cut short".to_string(),
                ));
            }

            if tool_calls.is_empty() {
                let Some(text) = text else {
                    note_stopped(history, "the model returned an empty response");
                    anyhow::bail!("the model returned an empty response");
                };
                history.push(Item::AssistantMessage(text));
                return Ok(TurnOutcome::Completed);
            }

            if let Some(text) = text {
                history.push(Item::AssistantMessage(text));
            }

            // A batch is answered in full even when the turn is cancelled part
            // way through it: the calls that did not run still need results.
            let mut batch = tool_calls.into_iter();
            while let Some(call) = batch.next() {
                on_event(AgentEvent::ToolStarted {
                    name: call.name.clone(),
                    arguments: call.arguments.clone(),
                });
                let executed = tokio::select! {
                    biased;
                    _ = &mut cancel => None,
                    executed = self.invoke(&call) => Some(executed),
                };
                let Some((content, is_error)) = executed else {
                    let content = not_executed(&call, CANCELLED);
                    record_result(history, call, content, true, on_event);
                    for call in batch {
                        record_unexecuted(history, call, CANCELLED, on_event);
                    }
                    close_open_batch(history, INTERRUPTED);
                    return Ok(TurnOutcome::Interrupted);
                };
                record_result(history, call, content, is_error, on_event);
            }
        }

        close_unfinished_turn(
            history,
            &format!(
                "the model called tools {MAX_STEPS_PER_TURN} times without finishing the turn"
            ),
            on_event,
        );
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

    /// Executes one tool call, returning what to record and whether it failed.
    ///
    /// Kept free of history so a cancelled call can still be recorded by the
    /// caller: a future that writes history would be dropped mid-write.
    async fn invoke(&self, call: &ToolCall) -> (String, bool) {
        // Malformed arguments never reach the tool: it would only fail on a
        // schema it cannot see. The text the model sent goes back to it
        // instead, so it can retry with something parseable.
        if let Some(error) = call.arguments_error.as_deref() {
            return (
                format!("Invalid arguments for tool \"{}\": {error}", call.name),
                true,
            );
        }
        match self.tools.get(&call.name) {
            Some(tool) => match tool.execute(call.arguments.clone(), &self.ctx).await {
                Ok(output) => (output.content, output.is_error),
                Err(err) => (format!("{err:#}"), true),
            },
            None => (format!("unknown tool '{}'", call.name), true),
        }
    }
}

/// Records a call together with its result, which is what keeps the history
/// replayable: a declared call without one is rejected by providers.
fn record_result(
    history: &mut Vec<Item>,
    call: ToolCall,
    content: String,
    is_error: bool,
    on_event: &mut (dyn FnMut(AgentEvent) + Send),
) {
    history.push(Item::ToolCall(call.clone()));
    history.push(Item::ToolResult {
        tool_call_id: call.id,
        content: content.clone(),
        is_error,
    });
    on_event(AgentEvent::ToolFinished {
        name: call.name,
        arguments: call.arguments,
        content,
        is_error,
    });
}

/// Records a call the model declared but that must not run, with a failure
/// result explaining why.
fn record_unexecuted(
    history: &mut Vec<Item>,
    call: ToolCall,
    reason: &str,
    on_event: &mut (dyn FnMut(AgentEvent) + Send),
) {
    on_event(AgentEvent::ToolStarted {
        name: call.name.clone(),
        arguments: call.arguments.clone(),
    });
    let content = not_executed(&call, reason);
    record_result(history, call, content, true, on_event);
}

/// What the model is told about a call that did not run.
fn not_executed(call: &ToolCall, reason: &str) -> String {
    format!(
        "Tool call \"{}\" was not executed: {reason}. Re-issue the tool call with complete arguments.",
        call.name
    )
}

/// The reason recorded for calls a cancelled turn did not run.
const CANCELLED: &str = "the turn was cancelled";

/// The reason recorded for calls in a reply the token limit cut short.
const TRUNCATED: &str =
    "the response hit the output token limit, so its arguments may be truncated";

/// The reason recorded when a provider filtered the reply.
const CONTENT_FILTERED: &str = "the provider filtered the reply";

/// Said in place of the assistant turn that never arrived.
const INTERRUPTED: &str = "The turn was interrupted before I finished answering.";

/// Ends a turn that stopped without the model's closing message, so the next
/// user message does not follow a tool result.
fn close_unfinished_turn(
    history: &mut Vec<Item>,
    reason: &str,
    on_event: &mut (dyn FnMut(AgentEvent) + Send),
) {
    on_event(AgentEvent::Notice(reason.to_string()));
    note_stopped(history, reason);
}

/// Explains in history why a turn ended before the model finished, when the
/// last thing said needs an answer.
fn note_stopped(history: &mut Vec<Item>, reason: &str) {
    close_open_batch(history, &format!("I stopped: {reason}."));
}

/// Appends an assistant message when history ends on a tool result.
///
/// Some providers reject a user message that follows a tool result without an
/// assistant message between them. Nothing is added when the last item is
/// already something else, so this is safe to call on every early return.
fn close_open_batch(history: &mut Vec<Item>, text: &str) {
    if matches!(history.last(), Some(Item::ToolResult { .. })) {
        history.push(Item::AssistantMessage(text.to_string()));
    }
}
