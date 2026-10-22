//! Agent-loop events, driven by a scripted model.

use async_trait::async_trait;
use engine::{Agent, AgentEvent};
use model::{Model, ModelEvent, ModelResponse};
use protocol::{Item, ToolCall, ToolDefinition};
use serde_json::json;
use std::sync::Mutex;
use tools::{Tool, ToolContext, ToolOutput, ToolRegistry};

/// A model that replays a fixed script of steps.
struct ScriptedModel {
    steps: Mutex<Vec<ModelResponse>>,
    /// Emit a fallback notice before every step's text.
    falls_back: bool,
}

impl ScriptedModel {
    fn new(steps: Vec<ModelResponse>) -> Self {
        Self {
            steps: Mutex::new(steps),
            falls_back: false,
        }
    }
}

#[async_trait]
impl Model for ScriptedModel {
    fn name(&self) -> &str {
        "scripted"
    }

    async fn complete(
        &self,
        _history: &[Item],
        _tools: &[ToolDefinition],
    ) -> anyhow::Result<ModelResponse> {
        panic!("the agent must stream, not call complete");
    }

    async fn complete_stream(
        &self,
        _history: &[Item],
        _tools: &[ToolDefinition],
        on_event: &mut (dyn FnMut(ModelEvent) + Send),
    ) -> anyhow::Result<ModelResponse> {
        let mut steps = self.steps.lock().unwrap();
        let step = steps.remove(0);
        if let Some(text) = step.content.as_deref() {
            for word in text.split_inclusive(' ') {
                on_event(ModelEvent::Text(word.to_string()));
            }
        }
        if self.falls_back {
            on_event(ModelEvent::Fallback);
        }
        Ok(step)
    }
}

/// A tool that echoes its argument and records that it ran.
struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name().to_string(),
            description: "echoes its arguments".to_string(),
            parameters: json!({"type": "object"}),
        }
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        _ctx: &ToolContext,
    ) -> anyhow::Result<ToolOutput> {
        Ok(ToolOutput::success(args.to_string()))
    }
}

fn agent(model: ScriptedModel) -> Agent<ScriptedModel> {
    let ctx = ToolContext::new(sandbox::Sandbox::current().unwrap());
    let registry = ToolRegistry::new(vec![Box::new(EchoTool)]);
    Agent::new(model, registry, ctx)
}

fn tool_response(text: Option<&str>, command: &str) -> ModelResponse {
    ModelResponse {
        content: text.map(str::to_string),
        tool_calls: vec![ToolCall {
            id: "call_1".to_string(),
            name: "echo".to_string(),
            arguments: json!({"command": command}),
        }],
    }
}

#[tokio::test]
async fn reports_text_then_tools_then_text() {
    let model = ScriptedModel::new(vec![
        tool_response(Some("running it. "), "ls"),
        ModelResponse {
            content: Some("done.".to_string()),
            tool_calls: vec![],
        },
    ]);
    let agent = agent(model);
    let mut history = vec![];
    let mut events = vec![];

    agent
        .run_turn_stream(&mut history, "go", &mut |event| events.push(event))
        .await
        .unwrap();

    assert_eq!(
        events,
        [
            AgentEvent::Text("running ".to_string()),
            AgentEvent::Text("it. ".to_string()),
            AgentEvent::ToolStarted {
                name: "echo".to_string(),
                arguments: json!({"command": "ls"}),
            },
            AgentEvent::Text("done.".to_string()),
        ]
    );
    assert!(matches!(history[0], Item::UserMessage(_)));
    assert!(matches!(history[1], Item::AssistantMessage(_)));
    assert!(matches!(history[2], Item::ToolCall(_)));
    assert!(matches!(history[3], Item::ToolResult { .. }));
    assert!(matches!(history[4], Item::AssistantMessage(_)));
}

#[tokio::test]
async fn announces_a_fallback_only_once_per_turn() {
    let mut first = tool_response(None, "ls");
    first.content = None;
    let model = ScriptedModel {
        steps: Mutex::new(vec![
            first,
            ModelResponse {
                content: Some("done.".to_string()),
                tool_calls: vec![],
            },
        ]),
        falls_back: true,
    };
    let agent = agent(model);
    let mut history = vec![];
    let mut events = vec![];

    agent
        .run_turn_stream(&mut history, "go", &mut |event| events.push(event))
        .await
        .unwrap();

    let notices = events
        .iter()
        .filter(|event| matches!(event, AgentEvent::Notice(_)))
        .count();
    assert_eq!(notices, 1, "{events:?}");
}

#[tokio::test]
async fn run_turn_works_without_a_sink() {
    let model = ScriptedModel::new(vec![ModelResponse {
        content: Some("hi".to_string()),
        tool_calls: vec![],
    }]);
    let agent = agent(model);
    let mut history = vec![];

    agent.run_turn(&mut history, "hello").await.unwrap();

    assert_eq!(history.len(), 2);
}

#[tokio::test]
async fn rejects_an_empty_reply() {
    let model = ScriptedModel::new(vec![ModelResponse {
        content: Some("   ".to_string()),
        tool_calls: vec![],
    }]);
    let agent = agent(model);
    let mut history = vec![];

    let error = agent.run_turn(&mut history, "hello").await.unwrap_err();

    assert!(format!("{error:#}").contains("empty"), "{error:#}");
}
