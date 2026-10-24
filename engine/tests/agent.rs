//! Agent-loop events, driven by a scripted model.

use async_trait::async_trait;
use engine::{Agent, AgentEvent, TurnOutcome};
use model::{Model, ModelEvent, ModelResponse, StopReason};
use protocol::{Item, ToolCall, ToolDefinition};
use serde_json::json;
use std::sync::Mutex;
use std::time::Duration;
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
    agent_with_model(model)
}

fn agent_with_model<M: Model>(model: M) -> Agent<M> {
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
        stop_reason: Some(StopReason::ToolCalls),
    }
}

#[tokio::test]
async fn reports_text_then_tools_then_text() {
    let model = ScriptedModel::new(vec![
        tool_response(Some("running it. "), "ls"),
        ModelResponse {
            content: Some("done.".to_string()),
            tool_calls: vec![],
            stop_reason: None,
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
            AgentEvent::ToolFinished {
                name: "echo".to_string(),
                arguments: json!({"command": "ls"}),
                content: "{\"command\":\"ls\"}".to_string(),
                is_error: false,
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
                stop_reason: None,
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
        stop_reason: None,
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
        stop_reason: None,
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
async fn does_not_run_tool_calls_from_a_truncated_reply() {
    let mut step = tool_response(None, "ls");
    step.stop_reason = Some(StopReason::Length);
    let model = ScriptedModel::new(vec![step]);
    let agent = agent(model);
    let mut history = vec![];
    let mut finished = vec![];

    let error = agent
        .run_turn_stream(&mut history, "go", &mut |event| {
            if let AgentEvent::ToolFinished {
                name,
                content,
                is_error,
                ..
            } = event
            {
                finished.push((name, content, is_error));
            }
        })
        .await
        .unwrap_err();

    assert!(format!("{error:#}").contains("truncated"), "{error:#}");
    assert_eq!(finished.len(), 1, "{finished:?}");
    let (name, content, is_error) = &finished[0];
    assert_eq!(name, "echo");
    assert!(is_error);
    assert!(content.contains("output token limit"), "{content}");
    // The call is still recorded, with a failure result: dropping it would
    // leave the assistant message unanswered.
    assert_eq!(history.len(), 3, "{history:?}");
    assert!(matches!(history[1], Item::ToolCall(_)));
    assert!(matches!(
        history[2],
        Item::ToolResult { is_error: true, .. }
    ));
}

#[tokio::test]
async fn reports_a_content_filtered_reply_as_an_error() {
    let model = ScriptedModel::new(vec![ModelResponse {
        content: None,
        tool_calls: vec![],
        stop_reason: Some(StopReason::ContentFilter),
    }]);
    let agent = agent(model);
    let mut history = vec![];

    let error = agent
        .run_turn_stream(&mut history, "go", &mut |_| {})
        .await
        .unwrap_err();

    assert!(format!("{error:#}").contains("content_filter"), "{error:#}");
}

#[tokio::test]
async fn notices_a_reply_cut_short_without_tool_calls() {
    let model = ScriptedModel::new(vec![ModelResponse {
        content: Some("a long answer".to_string()),
        tool_calls: vec![],
        stop_reason: Some(StopReason::Length),
    }]);
    let agent = agent(model);
    let mut history = vec![];
    let mut events = vec![];

    agent
        .run_turn_stream(&mut history, "go", &mut |event| events.push(event))
        .await
        .unwrap();

    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::Notice(message) if message.contains("output token limit")
    )));
    assert!(matches!(history.last(), Some(Item::AssistantMessage(_))));
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
            stop_reason: None,
        },
        ModelResponse {
            content: Some("done".to_string()),
            tool_calls: vec![],
            stop_reason: None,
        },
    ]);
    let agent = agent(model);
    let mut history = vec![];
    let mut events = vec![];

    agent
        .run_turn_stream(&mut history, "go", &mut |event| events.push(event))
        .await
        .unwrap();

    match &history[2] {
        Item::ToolResult {
            content, is_error, ..
        } => {
            assert!(is_error);
            assert!(content.contains("unknown tool"), "{content}");
        }
        other => panic!("expected a tool result, got {other:?}"),
    }
    let finished = events
        .iter()
        .find_map(|event| match event {
            AgentEvent::ToolFinished {
                name,
                content,
                is_error,
                ..
            } => Some((name.as_str(), content.as_str(), *is_error)),
            _ => None,
        })
        .expect("tool completion event");
    assert_eq!(finished.0, "missing");
    assert!(finished.1.contains("unknown tool"), "{}", finished.1);
    assert!(finished.2);
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
            stop_reason: None,
        },
        ModelResponse {
            content: Some("done".to_string()),
            tool_calls: vec![],
            stop_reason: None,
        },
    ]);
    let ctx = ToolContext::new(sandbox::Sandbox::current().unwrap());
    let agent = Agent::new(model, ToolRegistry::new(vec![Box::new(FailingTool)]), ctx);
    let mut history = vec![];
    let mut events = vec![];

    agent
        .run_turn_stream(&mut history, "go", &mut |event| events.push(event))
        .await
        .unwrap();

    match &history[2] {
        Item::ToolResult {
            content, is_error, ..
        } => {
            assert!(is_error);
            assert!(content.contains("the tool broke"), "{content}");
        }
        other => panic!("expected a tool result, got {other:?}"),
    }
    let finished = events
        .iter()
        .find_map(|event| match event {
            AgentEvent::ToolFinished {
                name,
                content,
                is_error,
                ..
            } => Some((name.as_str(), content.as_str(), *is_error)),
            _ => None,
        })
        .expect("tool completion event");
    assert_eq!(finished.0, "fail");
    assert!(finished.1.contains("the tool broke"), "{}", finished.1);
    assert!(finished.2);
}

#[tokio::test]
async fn cancellation_before_the_request_leaves_history_untouched() {
    let agent = agent(ScriptedModel::new(vec![]));
    let mut history = vec![];
    let mut events = vec![];

    let outcome = agent
        .run_turn_stream_cancellable(
            &mut history,
            "go",
            &mut |event| events.push(event),
            std::future::ready(()),
        )
        .await
        .unwrap();

    assert_eq!(outcome, TurnOutcome::Interrupted);
    assert!(history.is_empty());
    assert!(events.is_empty());
}

#[tokio::test]
async fn cancellation_while_streaming_keeps_partial_text_out_of_history() {
    struct HangingModel {
        cancel: Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
    }

    #[async_trait]
    impl Model for HangingModel {
        fn name(&self) -> &str {
            "hanging"
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
            on_event(ModelEvent::Text("partial".to_string()));
            if let Some(sender) = self.cancel.lock().unwrap().take() {
                let _ = sender.send(());
            }
            loop {
                tokio::time::sleep(Duration::from_secs(60)).await;
            }
        }
    }

    let (cancel_tx, cancel_rx) = tokio::sync::oneshot::channel();
    let model = HangingModel {
        cancel: Mutex::new(Some(cancel_tx)),
    };
    let agent = agent_with_model(model);
    let mut history = vec![];
    let mut events = vec![];

    let outcome = tokio::time::timeout(
        Duration::from_secs(1),
        agent.run_turn_stream_cancellable(
            &mut history,
            "go",
            &mut |event| events.push(event),
            async move {
                let _ = cancel_rx.await;
            },
        ),
    )
    .await
    .expect("cancellation completed the turn")
    .unwrap();

    assert_eq!(outcome, TurnOutcome::Interrupted);
    assert_eq!(events, [AgentEvent::Text("partial".to_string())]);
    assert_eq!(history.len(), 1);
    assert!(matches!(history[0], Item::UserMessage(_)));
}

#[tokio::test]
async fn cancellation_during_a_tool_leaves_no_tool_items_in_history() {
    struct HangingTool {
        cancel: Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
    }

    #[async_trait]
    impl Tool for HangingTool {
        fn name(&self) -> &str {
            "echo"
        }

        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: self.name().to_string(),
                description: "hangs until canceled".to_string(),
                parameters: json!({"type": "object"}),
            }
        }

        async fn execute(
            &self,
            _args: serde_json::Value,
            _ctx: &ToolContext,
        ) -> anyhow::Result<ToolOutput> {
            if let Some(sender) = self.cancel.lock().unwrap().take() {
                let _ = sender.send(());
            }
            loop {
                tokio::time::sleep(Duration::from_secs(60)).await;
            }
        }
    }

    let (cancel_tx, cancel_rx) = tokio::sync::oneshot::channel();
    let model = ScriptedModel::new(vec![tool_response(None, "work")]);
    let ctx = ToolContext::new(sandbox::Sandbox::current().unwrap());
    let agent = Agent::new(
        model,
        ToolRegistry::new(vec![Box::new(HangingTool {
            cancel: Mutex::new(Some(cancel_tx)),
        })]),
        ctx,
    );
    let mut history = vec![];
    let mut events = vec![];

    let outcome = tokio::time::timeout(
        Duration::from_secs(1),
        agent.run_turn_stream_cancellable(
            &mut history,
            "go",
            &mut |event| events.push(event),
            async move {
                let _ = cancel_rx.await;
            },
        ),
    )
    .await
    .expect("tool cancellation completed the turn")
    .unwrap();

    assert_eq!(outcome, TurnOutcome::Interrupted);
    assert_eq!(
        events,
        [AgentEvent::ToolStarted {
            name: "echo".to_string(),
            arguments: json!({"command": "work"}),
        }]
    );
    assert_eq!(history.len(), 1);
    assert!(matches!(history[0], Item::UserMessage(_)));
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
            stop_reason: None,
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
            stop_reason: None,
        },
        ModelResponse {
            content: Some("done".to_string()),
            tool_calls: vec![],
            stop_reason: None,
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
