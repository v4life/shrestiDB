//! AI Agent and Natural Language SQL Engine for ShrestiDB
//!
//! Provides schema-aware natural language to SQL translation, query explanation,
//! and interactive chat capabilities directly within the database engine and Studio.

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
}

impl AIAgent {
    pub fn new(executor: Arc<QueryExecutor>) -> Self {
        Self { executor }
    }

    /// Process an incoming natural language prompt
    pub fn process_message(&self, req: &ChatRequest) -> ChatResponse {
        let schema = SchemaSummary::from_executor(&self.executor);
        let user_prompt = req.message.trim();

        let p_lower = user_prompt.to_lowercase();

        // 1. Check for schema inquiry
        if p_lower == "tables"
            || p_lower.contains("what tables")
            || p_lower.contains("show schema")
            || p_lower.contains("list tables")
            || p_lower.contains("show tables")
            || p_lower.contains("all tables")
        {
            return ChatResponse {
                response: format!("Here is the current database catalog:\n\n{}", schema.describe()),
                sql: None,
                results: None,
                suggestions: schema.tables.iter().map(|t| format!("SELECT * FROM {} LIMIT 5;", t.name)).collect(),
            };
        }

        // 2. Generate SQL from natural language
        let (explanation, generated_sql) = self.translate_to_sql(user_prompt, &schema);

        // 3. If SQL was generated, attempt execution
        let mut results = None;
        if let Some(ref sql) = generated_sql {
            let start = std::time::Instant::now();
            match self.executor.execute_sql(sql) {
                Ok(raw_rows) => {
                    let duration_us = start.elapsed().as_micros();
                    // Attempt to infer column names from catalog
                    let mut columns = Vec::new();
                    if let Ok(stmt) = crate::sql::parser::SQLParser::parse(sql) {
                        if let crate::sql::parser::SQLStatement::Select(select) = stmt {
                            let catalog = self.executor.catalog.read();
                            if select.columns.len() == 1 && select.columns[0] == "*" {
                                if let Some(table) = catalog.get_table(&select.from) {
                                    columns = table.columns.iter().map(|c| c.name.clone()).collect();
                                }
                            } else {
                                columns = select.columns.clone();
                            }
                        }
                    }
                    if columns.is_empty() && !raw_rows.is_empty() {
                        columns = (1..=raw_rows[0].len()).map(|i| format!("col_{i}")).collect();
                    }

                    results = Some(QueryResultView {
                        success: true,
                        columns,
                        rows: raw_rows,
                        affected_rows: None,
                        duration_us,
                        error: None,
                    });
                }
                Err(e) => {
                    let duration_us = start.elapsed().as_micros();
                    results = Some(QueryResultView {
                        success: false,
                        columns: Vec::new(),
                        rows: Vec::new(),
                        affected_rows: None,
                        duration_us,
                        error: Some(e.to_string()),
                    });
                }
            }
        }

        // 4. Generate intelligent follow-up suggestions
        let suggestions = self.generate_suggestions(&schema, generated_sql.as_deref());

        ChatResponse {
            response: explanation,
            sql: generated_sql,
            results,
            suggestions,
        }
    }

    /// Intelligent Semantic Rule-based NL->SQL translator
    fn translate_to_sql(&self, prompt: &str, schema: &SchemaSummary) -> (String, Option<String>) {
        let p_lower = prompt.to_lowercase();

        // If the user directly typed a SQL statement
        if p_lower.starts_with("select ")
            || p_lower.starts_with("create ")
            || p_lower.starts_with("insert ")
            || p_lower.starts_with("update ")
            || p_lower.starts_with("delete ")
        {
            let sql = prompt.to_string();
            return (
                "Executing SQL statement directly:".to_string(),
                Some(if sql.ends_with(';') { sql } else { format!("{sql};") }),
            );
        }

        // Check if there are no tables
        if schema.tables.is_empty() {
            return (
                "Your database currently has no tables. You can create one with:\n\n```sql\nCREATE TABLE items (id INT PRIMARY KEY, name TEXT, embedding VECTOR(3));\n```".to_string(),
                Some("CREATE TABLE items (id INT PRIMARY KEY, name TEXT, embedding VECTOR(3));".to_string()),
            );
        }

        // Find relevant table name in prompt, or default to first table
        let target_table = schema
            .tables
            .iter()
            .find(|t| p_lower.contains(&t.name.to_lowercase()))
            .unwrap_or(&schema.tables[0]);

        let has_vector_col = target_table
            .columns
            .iter()
            .find(|(_, dt)| dt.to_lowercase().contains("vector"));

        // Pattern: Vector similarity / nearest neighbor search
        if (p_lower.contains("similar") || p_lower.contains("nearest") || p_lower.contains("closest") || p_lower.contains("vector"))
            && has_vector_col.is_some()
        {
            let vec_col = &has_vector_col.unwrap().0;
            // Extract vector literal if present, or generate default query vector
            let query_vec = if let Some(start) = prompt.find('[') {
                if let Some(end) = prompt[start..].find(']') {
                    &prompt[start..=start + end]
                } else {
                    "[1.0, 0.0, 0.0]"
                }
            } else {
                "[1.0, 0.0, 0.0]"
            };

            let sql = format!("SELECT * FROM {} ORDER BY {} <=> '{}' LIMIT 3;", target_table.name, vec_col, query_vec);
            return (
                format!("I generated a vector similarity query searching for nearest neighbors by cosine distance (`<=>`) on column `{vec_col}`:"),
                Some(sql),
            );
        }

        // Pattern: Count records
        if p_lower.contains("how many") || p_lower.contains("count") || p_lower.contains("total") {
            let sql = format!("SELECT count(*) FROM {};", target_table.name);
            return (
                format!("Calculated total record count for `{}`:", target_table.name),
                Some(sql),
            );
        }

        // Pattern: Filter / where clauses (e.g. "where balance > 1000", "older than 30")
        if p_lower.contains("older than") || p_lower.contains("age >") {
            let sql = format!("SELECT * FROM {} WHERE age > 30;", target_table.name);
            return (
                format!("Filtered `{}` for records with age > 30:", target_table.name),
                Some(sql),
            );
        }

        // Pattern: Top / Highest / Order by
        if p_lower.contains("top") || p_lower.contains("highest") || p_lower.contains("greatest") {
            // Pick first numeric column
            let num_col = target_table.columns.iter().find(|(_, dt)| dt == "Integer" || dt == "Float");
            if let Some((col_name, _)) = num_col {
                let sql = format!("SELECT * FROM {} ORDER BY {} DESC LIMIT 5;", target_table.name, col_name);
                return (
                    format!("Retrieved the top 5 records from `{}` sorted by `{}`:", target_table.name, col_name),
                    Some(sql),
                );
            }
        }

        // Default: SELECT * with LIMIT
        let sql = format!("SELECT * FROM {} LIMIT 10;", target_table.name);
        (
            format!("Querying the `{}` table:", target_table.name),
            Some(sql),
        )
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

    #[test]
    fn test_ai_agent_empty_catalog() {
        let executor = Arc::new(QueryExecutor::new(Catalog::new()));
        let agent = AIAgent::new(executor);

        let req = ChatRequest {
            message: "Show me all tables".to_string(),
            history: vec![],
        };
        let resp = agent.process_message(&req);
        assert!(resp.response.contains("No tables currently registered"));
    }

    #[test]
    fn test_ai_agent_vector_similarity_search() {
        let executor = Arc::new(QueryExecutor::new(Catalog::new()));

        // Setup items table with VECTOR column
        executor.execute_sql("CREATE TABLE items (id INT PRIMARY KEY, title TEXT, embedding VECTOR(3))").unwrap();
        executor.execute_sql("INSERT INTO items VALUES (1, 'apple', '[1.0, 0.0, 0.0]')").unwrap();
        executor.execute_sql("INSERT INTO items VALUES (2, 'banana', '[0.0, 1.0, 0.0]')").unwrap();

        let agent = AIAgent::new(executor);

        let req = ChatRequest {
            message: "Find items similar to [0.9, 0.1, 0.0]".to_string(),
            history: vec![],
        };
        let resp = agent.process_message(&req);

        assert!(resp.sql.is_some());
        let sql = resp.sql.unwrap();
        assert!(sql.contains("<=>"));
        assert!(sql.contains("items"));

        // Check that results were auto-executed!
        assert!(resp.results.is_some());
        let res = resp.results.unwrap();
        assert!(res.success);
        assert_eq!(res.rows.len(), 2);
        assert_eq!(res.rows[0][0], "1"); // 'apple' is nearest to [0.9, 0.1, 0.0]
    }

    #[test]
    fn test_ai_agent_count_query() {
        let executor = Arc::new(QueryExecutor::new(Catalog::new()));
        executor.execute_sql("CREATE TABLE users (id INT PRIMARY KEY, name TEXT)").unwrap();
        executor.execute_sql("INSERT INTO users VALUES (1, 'Alice')").unwrap();

        let agent = AIAgent::new(executor);
        let req = ChatRequest {
            message: "Count all rows in users".to_string(),
            history: vec![],
        };
        let resp = agent.process_message(&req);
        assert!(resp.sql.unwrap().contains("count(*)"));
        assert_eq!(resp.results.unwrap().rows[0][0], "1");
    }
}
