//! SQL parser
//!
//! Parses SQL text into `SQLStatement` using `sqlparser` — a real,
//! standards-compliant recursive-descent SQL parser — instead of a
//! whitespace-split tokenizer. Column lists, WHERE predicates, and
//! INSERT/UPDATE values are still flattened to their rendered string form:
//! there's no expression-tree evaluator downstream of this yet (see
//! `Binder`/`QueryExecutor`), so a richer `Expr` representation here
//! wouldn't be consumed by anything. What changes is that parsing itself is
//! now correct — real operator precedence, string/identifier quoting,
//! multi-clause statements — rather than a token split that couldn't
//! handle any of that.

use crate::error::{DatabaseError, Result};
use serde::{Deserialize, Serialize};
use sqlparser::ast::{self, Statement as AstStatement};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser as SqlDialectParser;

/// SQL statement types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SQLStatement {
    Select(SelectStatement),
    Insert(InsertStatement),
    Update(UpdateStatement),
    Delete(DeleteStatement),
    CreateTable(CreateTableStatement),
}

/// SELECT statement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectStatement {
    /// Column expressions as written, or `["*"]` for `SELECT *`.
    pub columns: Vec<String>,
    pub from: String,
    /// Names of any tables joined to `from` (in `JOIN` order). Join
    /// conditions/types aren't captured yet — there's no operator that
    /// consumes them downstream.
    pub join_tables: Vec<String>,
    pub where_clause: Option<String>,
    pub order_by: Option<String>,
    pub limit: Option<usize>,
}

/// INSERT statement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsertStatement {
    pub table: String,
    pub columns: Vec<String>,
    pub values: Vec<Vec<String>>,
}

/// UPDATE statement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateStatement {
    pub table: String,
    pub assignments: Vec<(String, String)>,
    pub where_clause: Option<String>,
}

/// DELETE statement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteStatement {
    pub table: String,
    pub where_clause: Option<String>,
}

/// CREATE TABLE statement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateTableStatement {
    pub name: String,
    pub columns: Vec<(String, String)>, // (name, type)
}

/// SQL Parser
pub struct SQLParser;

impl SQLParser {
    /// Parse a single SQL statement.
    pub fn parse(sql: &str) -> Result<SQLStatement> {
        let statements = SqlDialectParser::parse_sql(&GenericDialect {}, sql)
            .map_err(|e| DatabaseError::ParseError(e.to_string()))?;

        let stmt = statements
            .into_iter()
            .next()
            .ok_or_else(|| DatabaseError::ParseError("Empty SQL statement".to_string()))?;

        Self::convert(stmt)
    }

    fn convert(stmt: AstStatement) -> Result<SQLStatement> {
        match stmt {
            AstStatement::Query(query) => Self::convert_query(*query),
            AstStatement::Insert(insert) => Self::convert_insert(insert),
            AstStatement::Update(update) => Self::convert_update(update),
            AstStatement::Delete(delete) => Self::convert_delete(delete),
            AstStatement::CreateTable(create) => Self::convert_create_table(create),
            other => Err(DatabaseError::ParseError(format!(
                "Unsupported statement: {other}"
            ))),
        }
    }

    fn convert_query(query: ast::Query) -> Result<SQLStatement> {
        let select = match *query.body {
            ast::SetExpr::Select(select) => select,
            other => {
                return Err(DatabaseError::ParseError(format!(
                    "Only simple SELECT queries are supported, got: {other}"
                )))
            }
        };

        let columns = select
            .projection
            .iter()
            .map(|item| match item {
                ast::SelectItem::Wildcard(_) | ast::SelectItem::QualifiedWildcard(_, _) => {
                    "*".to_string()
                }
                other => other.to_string(),
            })
            .collect();

        let from = select
            .from
            .first()
            .map(|t| Self::table_factor_name(&t.relation))
            .unwrap_or_default();
        let join_tables = select
            .from
            .first()
            .map(|t| t.joins.iter().map(|j| Self::table_factor_name(&j.relation)).collect())
            .unwrap_or_default();

        let where_clause = select.selection.as_ref().map(|e| e.to_string());
        let order_by = query.order_by.as_ref().map(|o| o.to_string());
        let limit = query.limit_clause.as_ref().and_then(Self::extract_limit);

        Ok(SQLStatement::Select(SelectStatement {
            columns,
            from,
            join_tables,
            where_clause,
            order_by,
            limit,
        }))
    }

    fn table_factor_name(factor: &ast::TableFactor) -> String {
        match factor {
            ast::TableFactor::Table { name, .. } => name.to_string(),
            other => other.to_string(),
        }
    }

    fn extract_limit(clause: &ast::LimitClause) -> Option<usize> {
        match clause {
            ast::LimitClause::LimitOffset { limit, .. } => {
                limit.as_ref().and_then(Self::expr_as_usize)
            }
            _ => None,
        }
    }

    fn expr_as_usize(expr: &ast::Expr) -> Option<usize> {
        match expr {
            ast::Expr::Value(v) => match &v.value {
                ast::Value::Number(n, _) => n.parse().ok(),
                _ => None,
            },
            _ => None,
        }
    }

    fn convert_insert(insert: ast::Insert) -> Result<SQLStatement> {
        let table = match &insert.table {
            ast::TableObject::TableName(name) => name.to_string(),
            other => other.to_string(),
        };
        let columns = insert.columns.iter().map(|c| c.to_string()).collect();

        let values = match insert.source {
            Some(query) => match *query.body {
                ast::SetExpr::Values(values) => values
                    .rows
                    .into_iter()
                    .map(|row| row.content.into_iter().map(|e| e.to_string()).collect())
                    .collect(),
                _ => Vec::new(),
            },
            None => Vec::new(),
        };

        Ok(SQLStatement::Insert(InsertStatement {
            table,
            columns,
            values,
        }))
    }

    fn convert_update(update: ast::Update) -> Result<SQLStatement> {
        let table = Self::table_factor_name(&update.table.relation);
        let assignments = update
            .assignments
            .into_iter()
            .map(|a| (a.target.to_string(), a.value.to_string()))
            .collect();
        let where_clause = update.selection.as_ref().map(|e| e.to_string());

        Ok(SQLStatement::Update(UpdateStatement {
            table,
            assignments,
            where_clause,
        }))
    }

    fn convert_delete(delete: ast::Delete) -> Result<SQLStatement> {
        let tables = match &delete.from {
            ast::FromTable::WithFromKeyword(tables) => tables,
            ast::FromTable::WithoutKeyword(tables) => tables,
        };
        let table = tables
            .first()
            .map(|t| Self::table_factor_name(&t.relation))
            .ok_or_else(|| DatabaseError::ParseError("DELETE requires a table".to_string()))?;
        let where_clause = delete.selection.as_ref().map(|e| e.to_string());

        Ok(SQLStatement::Delete(DeleteStatement { table, where_clause }))
    }

    fn convert_create_table(create: ast::CreateTable) -> Result<SQLStatement> {
        let name = create.name.to_string();
        let columns = create
            .columns
            .into_iter()
            .map(|c| (c.name.to_string(), c.data_type.to_string()))
            .collect();

        Ok(SQLStatement::CreateTable(CreateTableStatement { name, columns }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_select() {
        let sql = "SELECT id, name FROM users WHERE age > 18";
        let stmt = SQLParser::parse(sql).unwrap();
        match stmt {
            SQLStatement::Select(s) => {
                assert_eq!(s.from, "users");
                assert_eq!(s.columns, vec!["id".to_string(), "name".to_string()]);
                assert!(s.where_clause.is_some());
            }
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_parse_select_star_with_order_and_limit() {
        let sql = "SELECT * FROM users ORDER BY age DESC LIMIT 10";
        let stmt = SQLParser::parse(sql).unwrap();
        match stmt {
            SQLStatement::Select(s) => {
                assert_eq!(s.columns, vec!["*".to_string()]);
                assert_eq!(s.from, "users");
                assert_eq!(s.limit, Some(10));
                assert!(s.order_by.is_some());
            }
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_parse_select_with_string_literal_predicate() {
        // A naive whitespace-split parser can't handle a quoted string
        // containing spaces or reserved words; a real parser can.
        let sql = "SELECT * FROM users WHERE name = 'John Smith' AND active = true";
        let stmt = SQLParser::parse(sql).unwrap();
        match stmt {
            SQLStatement::Select(s) => {
                let clause = s.where_clause.unwrap();
                assert!(clause.contains("John Smith"));
            }
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_parse_insert() {
        let sql = "INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)";
        let stmt = SQLParser::parse(sql).unwrap();
        match stmt {
            SQLStatement::Insert(i) => {
                assert_eq!(i.table, "users");
                assert_eq!(i.columns, vec!["id", "name", "age"]);
                assert_eq!(i.values.len(), 1);
                assert_eq!(i.values[0].len(), 3);
            }
            _ => panic!("Expected INSERT statement"),
        }
    }

    #[test]
    fn test_parse_update() {
        let sql = "UPDATE users SET age = 31, name = 'Alicia' WHERE id = 1";
        let stmt = SQLParser::parse(sql).unwrap();
        match stmt {
            SQLStatement::Update(u) => {
                assert_eq!(u.table, "users");
                assert_eq!(u.assignments.len(), 2);
                assert!(u.where_clause.is_some());
            }
            _ => panic!("Expected UPDATE statement"),
        }
    }

    #[test]
    fn test_parse_delete() {
        let sql = "DELETE FROM users WHERE id = 1";
        let stmt = SQLParser::parse(sql).unwrap();
        match stmt {
            SQLStatement::Delete(d) => {
                assert_eq!(d.table, "users");
                assert!(d.where_clause.is_some());
            }
            _ => panic!("Expected DELETE statement"),
        }
    }

    #[test]
    fn test_parse_create_table() {
        let sql = "CREATE TABLE users (id INT, name VARCHAR(50), age INT)";
        let stmt = SQLParser::parse(sql).unwrap();
        match stmt {
            SQLStatement::CreateTable(c) => {
                assert_eq!(c.name, "users");
                assert_eq!(c.columns.len(), 3);
                assert_eq!(c.columns[0].0, "id");
            }
            _ => panic!("Expected CREATE TABLE statement"),
        }
    }

    #[test]
    fn test_parse_invalid_sql_errors() {
        assert!(SQLParser::parse("SELECT * FRM users").is_err());
        assert!(SQLParser::parse("").is_err());
    }
}
