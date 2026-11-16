//! AI Agent and Natural Language SQL Engine for ShrestiDB
//!
//! Provides schema-aware natural language to SQL translation, query explanation,
//! and interactive chat capabilities directly within the database engine and Studio.

pub mod llm_client;
pub use llm_client::{LLMClient, LLMConfig};

use std::sync::Arc;
use serde::{Deserialize, Serialize};

use crate::execution::QueryExecutor;

/// A single message in the chat conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String, // "user", "assistant", or "system"
    pub content: String,
}

/// Request sent to the /api/chat endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub message: String,
    #[serde(default)]
    pub history: Vec<ChatMessage>,
}

/// Query execution result returned alongside AI response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResultView {
    pub success: bool,
    pub columns: Vec<String>,
    pub rows: Vec<Vec<String>>,
    pub affected_rows: Option<usize>,
    pub duration_us: u128,
    pub error: Option<String>,
}

/// Response returned to the Studio chat UI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub response: String,
    pub sql: Option<String>,
    pub results: Option<QueryResultView>,
    pub suggestions: Vec<String>,
    #[serde(default)]
    pub provider: String,
}

/// Schema summary representing the live database state
#[derive(Debug, Clone)]
pub struct SchemaSummary {
    pub tables: Vec<TableSummary>,
}

#[derive(Debug, Clone)]
pub struct TableSummary {
    pub name: String,
    pub columns: Vec<(String, String)>, // (col_name, data_type)
    pub approx_rows: usize,
}

impl SchemaSummary {
    /// Extract schema summary from executor catalog and storage
    pub fn from_executor(executor: &QueryExecutor) -> Self {
        let catalog = executor.catalog.read();
        let mut tables = Vec::new();

        for schema in catalog.tables.values() {
            let approx_rows = executor.oltp.store.table(schema.table_id as u64).row_count();
            let columns = schema
                .columns
                .iter()
                .map(|c| (c.name.clone(), format!("{:?}", c.data_type)))
                .collect();

            tables.push(TableSummary {
                name: schema.name.clone(),
                columns,
                approx_rows,
            });
        }
        tables.sort_by(|a, b| a.name.cmp(&b.name));

        SchemaSummary { tables }
    }

    /// Render schema description for prompts and explanations
    pub fn describe(&self) -> String {
        if self.tables.is_empty() {
            return "No tables currently registered in the database catalog.".to_string();
        }

        let mut out = String::from("Current Database Schema:\n");
        for t in &self.tables {
            let col_list = t
                .columns
                .iter()
                .map(|(n, dt)| format!("{n} {dt}"))
                .collect::<Vec<_>>()
                .join(", ");
            out.push_str(&format!("• Table '{}' (~{} rows): ({})\n", t.name, t.approx_rows, col_list));
        }
        out
    }
}

/// The Studio AI Agent
pub struct AIAgent {
    executor: Arc<QueryExecutor>,
    llm: LLMClient,
}

impl AIAgent {
    pub fn new(executor: Arc<QueryExecutor>) -> Self {
        Self {
            executor,
            llm: LLMClient::new(LLMConfig::default()),
        }
    }

    pub fn with_llm_config(executor: Arc<QueryExecutor>, config: LLMConfig) -> Self {
        Self {
            executor,
            llm: LLMClient::new(config),
        }
    }

    /// Process an incoming prompt: natural language via LLM, direct SQL execution, or schema commands
    pub async fn process_message(&self, req: &ChatRequest) -> ChatResponse {
        let schema = SchemaSummary::from_executor(&self.executor);
        let user_prompt = req.message.trim();
        let p_lower = user_prompt.to_lowercase();

        // 1. System schema inspection shortcut: \dt, tables, schema
        if p_lower == "tables"
            || p_lower == "\\dt"
            || p_lower == "schema"
            || p_lower == "show schema"
            || p_lower == "show tables"
            || p_lower == "list tables"
        {
            return ChatResponse {
                response: format!("### Current Database Catalog\n\n{}", schema.describe()),
                sql: None,
                results: None,
                suggestions: schema.tables.iter().map(|t| format!("SELECT * FROM {} LIMIT 5;", t.name)).collect(),
                provider: "System Catalog".to_string(),
            };
        }

        // 2. Direct raw SQL execution: if the user typed valid SQL directly, execute it
        if let Ok(stmt) = crate::sql::parser::SQLParser::parse(user_prompt.trim_end_matches(';')) {
            let sql = if user_prompt.ends_with(';') {
                user_prompt.to_string()
            } else {
                format!("{user_prompt};")
            };
            let results = self.try_execute_sql(&sql);
            let suggestions = self.generate_suggestions(&schema, Some(&sql));
            let explanation = match stmt {
                crate::sql::parser::SQLStatement::Select(_) => "Executed SQL query:".to_string(),
                crate::sql::parser::SQLStatement::Insert(_) => "Executed INSERT statement:".to_string(),
                crate::sql::parser::SQLStatement::Update(_) => "Executed UPDATE statement:".to_string(),
                crate::sql::parser::SQLStatement::Delete(_) => "Executed DELETE statement:".to_string(),
                crate::sql::parser::SQLStatement::CreateTable(_) => "Executed DDL: Created table successfully.".to_string(),
                crate::sql::parser::SQLStatement::CreateIndex(_) => "Executed DDL: Created index successfully.".to_string(),
                crate::sql::parser::SQLStatement::Analyze(_) => "Analyzed table statistics successfully.".to_string(),
            };

            return ChatResponse {
                response: explanation,
                sql: Some(sql),
                results,
                suggestions,
                provider: "Direct SQL".to_string(),
            };
        }

        // 3. Natural language query: delegate to configured LLM (Ollama / OpenAI / Gemini)
        if self.llm.is_configured() {
            let system_prompt = format!(
                "You are an AI SQL agent for ShrestiDB, a learned and vector database.\n\
                 {}\n\
                 Supported SQL syntax:\n\
                 - SELECT with WHERE, ORDER BY, GROUP BY, LIMIT, JOIN\n\
                 - Vector operators: <=> (cosine distance), <-> (L2 distance), <#> (inner product)\n\
                 - CREATE TABLE ... (id INT PRIMARY KEY, ..., embedding VECTOR(N))\n\
                 - INSERT INTO ... VALUES (...)\n\n\
                 Always include any SQL query you generate inside standard markdown ```sql ... ``` code blocks.\n\
                 Keep explanations concise and helpful.",
                schema.describe()
            );

            match self.llm.chat_completion(&system_prompt, user_prompt, &req.history).await {
                Ok(llm_response) => {
                    let extracted_sql = Self::extract_sql_from_markdown(&llm_response);
                    let results = extracted_sql.as_deref().and_then(|sql| self.try_execute_sql(sql));
                    let suggestions = self.generate_suggestions(&schema, extracted_sql.as_deref());

                    return ChatResponse {
                        response: llm_response,
                        sql: extracted_sql,
                        results,
                        suggestions,
                        provider: format!("{} ({})", self.llm.provider_name(), self.llm.model_name()),
                    };
                }
                Err(err) => {
                    let suggestions = schema.tables.iter().map(|t| format!("SELECT * FROM {} LIMIT 5;", t.name)).collect();
                    let response = format!(
                        "⚠️ **AI Provider Offline**\n\n\
                         Could not reach **{}** at `{}` (model: `{}`).\n\n\
                         **Error:**\n```\n{}\n```\n\n\
                         **To use local AI with Ollama:**\n\
                         1. Start Ollama with your model in a terminal:\n\
                         ```bash\n\
                         ollama run {}\n\
                         ```\n\
                         2. Once running, ask your question again!\n\n\
                         *Tip: You can also enter direct SQL statements (e.g. `SELECT * FROM accounts;`) here anytime.*",
                        self.llm.provider_name(),
                        self.llm.endpoint(),
                        self.llm.model_name(),
                        err,
                        self.llm.model_name()
                    );

                    return ChatResponse {
                        response,
                        sql: None,
                        results: None,
                        suggestions,
                        provider: format!("{} (Offline)", self.llm.provider_name()),
                    };
                }
            }
        }

        // 4. No LLM provider configured
        let suggestions = schema.tables.iter().map(|t| format!("SELECT * FROM {} LIMIT 5;", t.name)).collect();
        let response = format!(
            "ℹ️ **AI Model Not Configured**\n\n\
             No AI model provider is currently active.\n\n\
             **To enable local AI with Ollama:**\n\
             ```bash\n\
             ollama run llama3.2\n\
             cargo run --bin shrestidb-studio\n\
             ```\n\n\
             **To use OpenAI / Cloud LLM:**\n\
             ```bash\n\
             SHRESTIDB_AI_ENDPOINT=\"https://api.openai.com/v1\" \\\n\
             SHRESTIDB_AI_API_KEY=\"sk-...\" \\\n\
             SHRESTIDB_AI_MODEL=\"gpt-4o-mini\" \\\n\
             cargo run --bin shrestidb-studio\n\
             ```\n\n\
             *Tip: You can run SQL queries directly here anytime.*"
        );

        ChatResponse {
            response,
            sql: None,
            results: None,
            suggestions,
            provider: "Unconfigured".to_string(),
        }
    }

    /// Helper to extract SQL from markdown code fences ```sql ... ```
    fn extract_sql_from_markdown(text: &str) -> Option<String> {
        if let Some(start) = text.find("```sql") {
            let rest = &text[start + 6..];
            if let Some(end) = rest.find("```") {
                let sql = rest[..end].trim();
                if !sql.is_empty() {
                    return Some(sql.to_string());
                }
            }
        } else if let Some(start) = text.find("```") {
            let rest = &text[start + 3..];
            if let Some(end) = rest.find("```") {
                let sql = rest[..end].trim();
                if sql.to_lowercase().starts_with("select")
                    || sql.to_lowercase().starts_with("create")
                    || sql.to_lowercase().starts_with("insert")
                {
                    return Some(sql.to_string());
                }
            }
        }
        None
    }

    /// Try executing a SQL statement and formatting the output
    fn try_execute_sql(&self, sql: &str) -> Option<QueryResultView> {
        let start = std::time::Instant::now();
        match self.executor.execute_sql(sql) {
            Ok(raw_rows) => {
                let duration_us = start.elapsed().as_micros();
                let mut columns = Vec::new();
                if let Ok(stmt) = crate::sql::parser::SQLParser::parse(sql) {
                    if let crate::sql::parser::SQLStatement::Select(select) = stmt {
                        columns = self.executor.resolve_select_columns(&select);
                    }
                }
                if columns.is_empty() && !raw_rows.is_empty() {
                    columns = (1..=raw_rows[0].len()).map(|i| format!("col_{i}")).collect();
                }

                Some(QueryResultView {
                    success: true,
                    columns,
                    rows: raw_rows,
                    affected_rows: None,
                    duration_us,
                    error: None,
                })
            }
            Err(e) => {
                let duration_us = start.elapsed().as_micros();
                Some(QueryResultView {
                    success: false,
                    columns: Vec::new(),
                    rows: Vec::new(),
                    affected_rows: None,
                    duration_us,
                    error: Some(e.to_string()),
                })
            }
        }
    }

    /// Generate intelligent contextual follow-up chips
    fn generate_suggestions(&self, schema: &SchemaSummary, last_sql: Option<&str>) -> Vec<String> {
        let mut suggestions = Vec::new();

        if let Some(table) = schema.tables.first() {
            let has_vector = table.columns.iter().find(|(_, dt)| dt.to_lowercase().contains("vector"));

            if let Some((v_col, _)) = has_vector {
                suggestions.push(format!("Find items similar to [1.0, 0.0, 0.0] in {}", table.name));
                suggestions.push(format!("CREATE INDEX idx_{}_hnsw ON {}({});", table.name, table.name, v_col));
            }

            suggestions.push(format!("Count all rows in {}", table.name));
            suggestions.push(format!("SELECT * FROM {} LIMIT 5;", table.name));
        }

        if last_sql.map(|s| !s.contains("ANALYZE")).unwrap_or(true) {
            if let Some(table) = schema.tables.first() {
                suggestions.push(format!("ANALYZE {};", table.name));
            }
        }

        suggestions.truncate(4);
        suggestions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::catalog::Catalog;

    #[tokio::test]
    async fn test_ai_agent_schema_inspection() {
        let executor = Arc::new(QueryExecutor::new(Catalog::new()));
        let agent = AIAgent::new(executor);

        let req = ChatRequest {
            message: "tables".to_string(),
            history: vec![],
        };
        let resp = agent.process_message(&req).await;
        assert_eq!(resp.provider, "System Catalog");
        assert!(resp.response.contains("No tables currently registered"));
    }

    #[tokio::test]
    async fn test_ai_agent_direct_sql_vector_search() {
        let executor = Arc::new(QueryExecutor::new(Catalog::new()));

        // Setup items table with VECTOR column
        executor.execute_sql("CREATE TABLE items (id INT PRIMARY KEY, title TEXT, embedding VECTOR(3))").unwrap();
        executor.execute_sql("INSERT INTO items VALUES (1, 'apple', '[1.0, 0.0, 0.0]')").unwrap();
        executor.execute_sql("INSERT INTO items VALUES (2, 'banana', '[0.0, 1.0, 0.0]')").unwrap();

        let agent = AIAgent::new(executor);

        // Direct SQL executed in chat
        let req = ChatRequest {
            message: "SELECT * FROM items ORDER BY embedding <=> '[0.9, 0.1, 0.0]' LIMIT 2;".to_string(),
            history: vec![],
        };
        let resp = agent.process_message(&req).await;

        assert_eq!(resp.provider, "Direct SQL");
        assert!(resp.sql.is_some());
        assert!(resp.results.is_some());
        let res = resp.results.unwrap();
        assert!(res.success);
        assert_eq!(res.rows.len(), 2);
        assert_eq!(res.rows[0][0], "1"); // 'apple' is nearest to [0.9, 0.1, 0.0]
    }

    #[tokio::test]
    async fn test_ai_agent_direct_sql_ddl() {
        let executor = Arc::new(QueryExecutor::new(Catalog::new()));
        let agent = AIAgent::new(executor);

        let req = ChatRequest {
            message: "CREATE TABLE zip_codes (id INT PRIMARY KEY, zip TEXT, city TEXT, coordinates VECTOR(2));".to_string(),
            history: vec![],
        };
        let resp = agent.process_message(&req).await;
        assert_eq!(resp.provider, "Direct SQL");
        assert!(resp.sql.is_some());
        assert!(resp.response.contains("Created table successfully"));
    }

    #[tokio::test]
    async fn test_ai_agent_offline_diagnostic() {
        let executor = Arc::new(QueryExecutor::new(Catalog::new()));
        // Point to a non-existent port to simulate offline Ollama
        let config = LLMConfig {
            endpoint: "http://127.0.0.1:54321/v1".to_string(),
            api_key: "".to_string(),
            model: "llama3.2".to_string(),
            temperature: 0.1,
        };
        let agent = AIAgent::with_llm_config(executor, config);

        let req = ChatRequest {
            message: "how many users are there?".to_string(),
            history: vec![],
        };
        let resp = agent.process_message(&req).await;

        // Must NOT invent fake SQL or pretend to be an AI rule engine
        assert!(resp.provider.contains("Offline"));
        assert!(resp.sql.is_none());
        assert!(resp.response.contains("AI Provider Offline"));
        assert!(resp.response.contains("ollama run llama3.2"));
    }
}
