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
        let Some(step) = steps.first().cloned() else {
            anyhow::bail!("the script has no further steps for the agent");
        };
        steps.remove(0);
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

#[tokio::test]
async fn stops_at_the_step_limit() {
    // A model that always asks for a tool must not loop forever.
    let model = ScriptedModel::new(vec![tool_response(None, "ls"); 32]);
    let agent = agent(model);
    let mut history = vec![];
    let mut started = 0;

    let error = agent
        .run_turn_stream(&mut history, "go", &mut |event| {
            if matches!(event, AgentEvent::ToolStarted { .. }) {
                started += 1;
            }
        })
        .await
        .unwrap_err();

    assert_eq!(started, 32);
    assert!(format!("{error:#}").contains("32"), "{error:#}");
}

#[tokio::test]
async fn records_an_unknown_tool_as_an_error_result() {
    let model = ScriptedModel::new(vec![
        ModelResponse {
            content: None,
            tool_calls: vec![ToolCall {
                id: "call_1".to_string(),
                name: "missing".to_string(),
                arguments: json!({}),
            }],
        },
        ModelResponse {
            content: Some("done".to_string()),
            tool_calls: vec![],
        },
    ]);
    let agent = agent(model);
    let mut history = vec![];

    agent.run_turn(&mut history, "go").await.unwrap();

    match &history[2] {
        Item::ToolResult {
            content, is_error, ..
        } => {
            assert!(is_error);
            assert!(content.contains("unknown tool"), "{content}");
        }
        other => panic!("expected a tool result, got {other:?}"),
    }
}

#[tokio::test]
async fn records_a_failing_tool_as_an_error_result() {
    struct FailingTool;

    #[async_trait]
    impl Tool for FailingTool {
        fn name(&self) -> &str {
            "fail"
        }

        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: self.name().to_string(),
                description: "always fails".to_string(),
                parameters: json!({"type": "object"}),
            }
        }

        async fn execute(
            &self,
            _args: serde_json::Value,
            _ctx: &ToolContext,
        ) -> anyhow::Result<ToolOutput> {
            anyhow::bail!("the tool broke")
        }
    }

    let model = ScriptedModel::new(vec![
        ModelResponse {
            content: None,
            tool_calls: vec![ToolCall {
                id: "call_1".to_string(),
                name: "fail".to_string(),
                arguments: json!({}),
            }],
        },
        ModelResponse {
            content: Some("done".to_string()),
            tool_calls: vec![],
        },
    ]);
    let ctx = ToolContext::new(sandbox::Sandbox::current().unwrap());
    let agent = Agent::new(model, ToolRegistry::new(vec![Box::new(FailingTool)]), ctx);
    let mut history = vec![];

    agent.run_turn(&mut history, "go").await.unwrap();

    match &history[2] {
        Item::ToolResult {
            content, is_error, ..
        } => {
            assert!(is_error);
            assert!(content.contains("the tool broke"), "{content}");
        }
        other => panic!("expected a tool result, got {other:?}"),
    }
}

#[tokio::test]
async fn propagates_a_model_error_and_keeps_the_user_message() {
    let model = ScriptedModel::new(vec![]);
    let agent = agent(model);
    let mut history = vec![];

    let error = agent
        .run_turn_stream(&mut history, "go", &mut |_| {})
        .await
        .unwrap_err();

    assert!(format!("{error:#}").contains("script"), "{error:#}");
    assert!(matches!(history[0], Item::UserMessage(_)));
    assert_eq!(history.len(), 1);
}

#[tokio::test]
async fn records_text_that_arrives_with_tool_calls() {
    let model = ScriptedModel::new(vec![
        tool_response(Some("looking. "), "ls"),
        ModelResponse {
            content: Some("done.".to_string()),
            tool_calls: vec![],
        },
    ]);
    let agent = agent(model);
    let mut history = vec![];

    agent.run_turn(&mut history, "go").await.unwrap();

    match &history[1] {
        Item::AssistantMessage(text) => assert_eq!(text, "looking. "),
        other => panic!("expected an assistant message, got {other:?}"),
    }
}

#[tokio::test]
async fn runs_several_tool_calls_in_one_round() {
    let model = ScriptedModel::new(vec![
        ModelResponse {
            content: None,
            tool_calls: vec![
                ToolCall {
                    id: "call_1".to_string(),
                    name: "echo".to_string(),
                    arguments: json!({"command": "one"}),
                },
                ToolCall {
                    id: "call_2".to_string(),
                    name: "echo".to_string(),
                    arguments: json!({"command": "two"}),
                },
            ],
        },
        ModelResponse {
            content: Some("done".to_string()),
            tool_calls: vec![],
        },
    ]);
    let agent = agent(model);
    let mut history = vec![];
    let mut started = vec![];

    agent
        .run_turn_stream(&mut history, "go", &mut |event| {
            if let AgentEvent::ToolStarted { name, arguments } = event {
                started.push((name, arguments));
            }
        })
        .await
        .unwrap();

    assert_eq!(
        started,
        [
            ("echo".to_string(), json!({"command": "one"})),
            ("echo".to_string(), json!({"command": "two"})),
        ]
    );
    assert_eq!(history.len(), 6);
}
