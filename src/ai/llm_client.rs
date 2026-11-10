//! Universal OpenAI-compatible LLM Client for ShrestiDB
//!
//! Connects to any OpenAI-compatible API endpoint:
//! - OpenAI (https://api.openai.com/v1)
//! - Google Gemini (via OpenAI compatibility: https://generativelanguage.googleapis.com/v1beta/openai)
//! - Local Ollama (http://localhost:11434/v1)
//! - Groq / DeepSeek / Together / Anthropic via proxy

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMConfig {
    /// Base URL (e.g. "https://api.openai.com/v1", "http://localhost:11434/v1", "https://generativelanguage.googleapis.com/v1beta/openai")
    pub endpoint: String,
    /// API key (empty for local providers like Ollama)
    pub api_key: String,
    /// Model name (e.g. "gemini-1.5-flash", "gpt-4o-mini", "llama3.2", "qwen2.5-coder")
    pub model: String,
    /// Temperature for sampling
    pub temperature: f32,
}

impl Default for LLMConfig {
    fn default() -> Self {
        // Read configuration from environment variables if present
        let endpoint = std::env::var("SHRESTIDB_AI_ENDPOINT")
            .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
        let api_key = std::env::var("SHRESTIDB_AI_API_KEY")
            .unwrap_or_default();
        let model = std::env::var("SHRESTIDB_AI_MODEL")
            .unwrap_or_else(|_| "gpt-4o-mini".to_string());

        Self {
            endpoint,
            api_key,
            model,
            temperature: 0.1,
        }
    }
}

#[derive(Debug, Serialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    temperature: f32,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAIMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIResponse {
    choices: Vec<OpenAIChoice>,
}

#[derive(Debug, Deserialize)]
struct OpenAIChoice {
    message: OpenAIMessage,
}

pub struct LLMClient {
    config: LLMConfig,
    client: reqwest::Client,
}

impl LLMClient {
    pub fn new(config: LLMConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
        }
    }

    /// Check if this client is configured with active credentials or a local endpoint
    pub fn is_configured(&self) -> bool {
        !self.config.api_key.trim().is_empty() || self.config.endpoint.contains("localhost") || self.config.endpoint.contains("127.0.0.1")
    }

    /// Send a chat completion request to the provider
    pub async fn chat_completion(
        &self,
        system_prompt: &str,
        user_message: &str,
        history: &[crate::ai::ChatMessage],
    ) -> Result<String, String> {
        let mut messages = Vec::new();
        messages.push(OpenAIMessage {
            role: "system".to_string(),
            content: system_prompt.to_string(),
        });

        for msg in history {
            messages.push(OpenAIMessage {
                role: msg.role.clone(),
                content: msg.content.clone(),
            });
        }

        messages.push(OpenAIMessage {
            role: "user".to_string(),
            content: user_message.to_string(),
        });

        let req_body = OpenAIRequest {
            model: self.config.model.clone(),
            messages,
            temperature: self.config.temperature,
        };

        let url = format!("{}/chat/completions", self.config.endpoint.trim_end_matches('/'));

        let mut req = self.client.post(&url)
            .json(&req_body)
            .header("Content-Type", "application/json");

        if !self.config.api_key.is_empty() {
            req = req.header("Authorization", format!("Bearer {}", self.config.api_key));
        }

        let resp = req.send().await.map_err(|e| format!("HTTP request failed: {e}"))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let err_text = resp.text().await.unwrap_or_default();
            return Err(format!("Model provider returned error {status}: {err_text}"));
        }

        let data: OpenAIResponse = resp.json().await.map_err(|e| format!("Failed to parse response: {e}"))?;

        data.choices
            .first()
            .map(|c| c.message.content.clone())
            .ok_or_else(|| "No choices returned by model provider".to_string())
    }
}
