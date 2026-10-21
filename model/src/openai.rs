use crate::{Model, ModelResponse, ToolDefinition};
use async_trait::async_trait;
use protocol::{Item, ToolCall};
use serde_json::json;

pub struct OpenAIModel {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    model_name: String,
}

impl OpenAIModel {
    pub fn new(api_key: &str, base_url: &str, model_name: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.to_string(),
            base_url: base_url.to_string(),
            model_name: model_name.to_string(),
        }
    }

    fn format_messages(history: &[Item]) -> Vec<serde_json::Value> {
        let mut messages = vec![];
        for item in history {
            match item {
                Item::UserMessage(t) => messages.push(json!({"role": "user", "content": t})),
                Item::AssistantMessage(t) => {
                    messages.push(json!({"role": "assistant", "content": t}))
                }
                Item::ToolCall(c) => {
                    messages.push(json!({
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": c.id,
                            "type": "function",
                            "function": {
                                "name": c.name,
                                "arguments": c.arguments.to_string()
                            }
                        }]
                    }));
                }
                Item::ToolResult {
                    tool_call_id,
                    content,
                    ..
                } => {
                    messages.push(json!({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": content
                    }));
                }
            }
        }
        messages
    }
}

#[async_trait]
impl Model for OpenAIModel {
    fn name(&self) -> &str {
        &self.model_name
    }

    async fn complete(
        &self,
        history: &[Item],
        tools: &[ToolDefinition],
    ) -> anyhow::Result<ModelResponse> {
        let messages = Self::format_messages(history);
        let mut body = json!({
            "model": self.model_name,
            "messages": messages,
        });

        if !tools.is_empty() {
            let openai_tools: Vec<_> = tools
                .iter()
                .map(|t| {
                    json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters
                        }
                    })
                })
                .collect();
            body["tools"] = json!(openai_tools);
        }

        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
        let res = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?;

        if !res.status().is_success() {
            anyhow::bail!("API Error: {}", res.text().await?);
        }

        let json_res: serde_json::Value = res.json().await?;
        let choice = json_res["choices"]
            .get(0)
            .ok_or_else(|| anyhow::anyhow!("Empty choices"))?;
        let message = &choice["message"];

        let content = message["content"].as_str().map(|s| s.to_string());
        let mut tool_calls = vec![];

        if let Some(tcs) = message["tool_calls"].as_array() {
            for tc in tcs {
                tool_calls.push(ToolCall {
                    id: tc["id"].as_str().unwrap_or_default().to_string(),
                    name: tc["function"]["name"]
                        .as_str()
                        .unwrap_or_default()
                        .to_string(),
                    arguments: serde_json::from_str(
                        tc["function"]["arguments"].as_str().unwrap_or("{}"),
                    )
                    .unwrap_or(json!({})),
                });
            }
        }

        Ok(ModelResponse {
            content,
            tool_calls,
        })
    }
}
