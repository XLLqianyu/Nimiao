//! Agent loop.
//!
//! One [`Agent`] binds a model to a tool set and an execution context. A turn
//! appends the user message to the conversation, asks the model for the next
//! step, runs whatever tools it asks for, and repeats until the model answers
//! without further tool calls.

use model::{Model, ModelResponse};
use protocol::{Item, ToolCall};
use tools::{ToolContext, ToolRegistry};

/// Upper bound on tool-calling rounds in a single turn, so a model that never
/// stops asking for tools cannot loop forever.
const MAX_STEPS_PER_TURN: usize = 32;

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

    /// Runs one user turn to completion and records every step in `history`.
    ///
    /// Returns after the model produced a reply with no tool calls. The final
    /// reply is the last [`Item::AssistantMessage`] appended to `history`.
    pub async fn run_turn(
        &self,
        history: &mut Vec<Item>,
        input: impl Into<String>,
    ) -> anyhow::Result<()> {
        history.push(Item::UserMessage(input.into()));

        let definitions = self.tools.definitions();

        for _ in 0..MAX_STEPS_PER_TURN {
            let ModelResponse {
                content,
                tool_calls,
            } = self.model.complete(history, &definitions).await?;

            let text = content.filter(|text| !text.trim().is_empty());

            if tool_calls.is_empty() {
                let Some(text) = text else {
                    anyhow::bail!("the model returned an empty response");
                };
                history.push(Item::AssistantMessage(text));
                return Ok(());
            }

            if let Some(text) = text {
                history.push(Item::AssistantMessage(text));
            }

            for call in tool_calls {
                self.run_tool(history, call).await;
            }
        }

        anyhow::bail!(
            "the model called tools {MAX_STEPS_PER_TURN} times without finishing the turn"
        )
    }

    /// Executes one tool call and records the call together with its result.
    async fn run_tool(&self, history: &mut Vec<Item>, call: ToolCall) {
        let arguments = call.arguments;
        let (content, is_error) = match self.tools.get(&call.name) {
            Some(tool) => match tool.execute(arguments.clone(), &self.ctx).await {
                Ok(output) => (output.content, output.is_error),
                Err(err) => (format!("{err:#}"), true),
            },
            None => (format!("unknown tool '{}'", call.name), true),
        };

        history.push(Item::ToolCall(ToolCall {
            id: call.id.clone(),
            name: call.name,
            arguments,
        }));
        history.push(Item::ToolResult {
            tool_call_id: call.id,
            content,
            is_error,
        });
    }
}
