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
    CreateIndex(CreateIndexStatement),
    Analyze(AnalyzeStatement),
}

/// One `JOIN` clause: the joined table and its `ON` condition, rendered as
/// a flattened string like `where_clause` (e.g. `"users.id = orders.user_id"`,
/// or `"u.id = o.user_id"` if the query used aliases — see `alias`) — there's
/// no expression tree kept around, just like everywhere else in this AST.
/// `USING`/`NATURAL` joins aren't specially handled: their condition comes
/// through as `None`, same as an explicit `CROSS JOIN`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoinClause {
    pub table: String,
    /// The table's alias (`JOIN orders AS o` / `JOIN orders o`), if any.
    /// `condition` refers to this table by the alias when one is given —
    /// see `execution::executor::QueryExecutor::merge_schemas`.
    pub alias: Option<String>,
    pub condition: Option<String>,
}

/// SELECT statement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectStatement {
    /// Column expressions as written, or `["*"]` for `SELECT *`.
    pub columns: Vec<String>,
    pub from: String,
    /// The FROM table's alias (`FROM users AS u` / `FROM users u`), if any.
    pub from_alias: Option<String>,
    /// Any tables joined to `from`, in `JOIN` order.
    pub joins: Vec<JoinClause>,
    pub where_clause: Option<String>,
    /// `GROUP BY` column names. Empty means no `GROUP BY` — `GROUP BY ALL`
    /// (Snowflake/DuckDB/ClickHouse syntax) isn't recognized and also
    /// comes through empty.
    pub group_by: Vec<String>,
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

/// One column definition in a CREATE TABLE statement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnDef {
    pub name: String,
    /// The declared type, rendered as written (e.g. `"INT"`, `"VARCHAR(50)"`)
    /// — not parsed further here; see `execution::executor` for how this
    /// gets mapped onto the catalog's coarser `DataType`.
    pub data_type: String,
    pub nullable: bool,
    pub primary_key: bool,
}

/// CREATE TABLE statement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateTableStatement {
    pub name: String,
    pub columns: Vec<ColumnDef>,
}

/// `CREATE INDEX <name> ON <table>(<column>)`. Only a single indexed
/// column is supported — if the statement names more than one, every
/// column after the first is silently dropped rather than rejected; this
/// is a deliberate MVP scope boundary, not an oversight.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateIndexStatement {
    pub name: String,
    pub table: String,
    pub column: String,
}

/// `ANALYZE [TABLE] <table>` — scans every column of `table` and builds a
/// real `optimizer::cardinality::ColumnDistribution` for each (see
/// `execution::executor::QueryExecutor::execute_analyze`). Column-scoped
/// `ANALYZE <table> (col1, col2)` (Postgres) or `FOR COLUMNS` (Hive)
/// syntax both parse fine via `sqlparser` but aren't distinguished here —
/// this always analyzes every column, a deliberate MVP scope boundary
/// (analyzing a whole table is cheap enough at this project's scale that
/// a column-subset optimization isn't worth the complexity yet), not an
/// oversight.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyzeStatement {
    pub table: String,
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
            AstStatement::CreateIndex(create) => Self::convert_create_index(create),
            AstStatement::Analyze(analyze) => Self::convert_analyze(analyze),
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
        let from_alias = select.from.first().and_then(|t| Self::table_factor_alias(&t.relation));
        let joins = select
            .from
            .first()
            .map(|t| {
                t.joins
                    .iter()
                    .map(|j| JoinClause {
                        table: Self::table_factor_name(&j.relation),
                        alias: Self::table_factor_alias(&j.relation),
                        condition: Self::join_condition(&j.join_operator),
                    })
                    .collect()
            })
            .unwrap_or_default();

        let where_clause = select.selection.as_ref().map(|e| e.to_string());
        let group_by = match &select.group_by {
            ast::GroupByExpr::Expressions(exprs, _) => exprs.iter().map(|e| e.to_string()).collect(),
            ast::GroupByExpr::All(_) => Vec::new(), // not recognized; see field docs
        };
        let order_by = query.order_by.as_ref().map(|o| o.to_string());
        let limit = query.limit_clause.as_ref().and_then(Self::extract_limit);

        Ok(SQLStatement::Select(SelectStatement {
            columns,
            from,
            from_alias,
            joins,
            where_clause,
            group_by,
            order_by,
            limit,
        }))
    }

    /// Extract the `ON <expr>` condition from a join operator, regardless
    /// of join type (`INNER`/`LEFT`/`RIGHT`/...). `None` for `USING`,
    /// `NATURAL`, or no constraint (`CROSS JOIN`).
    fn join_condition(op: &ast::JoinOperator) -> Option<String> {
        use ast::JoinOperator::*;
        let constraint = match op {
            Join(c) | Inner(c) | Left(c) | LeftOuter(c) | Right(c) | RightOuter(c) | FullOuter(c)
            | CrossJoin(c) | Semi(c) | LeftSemi(c) => Some(c),
            _ => None,
        }?;
        match constraint {
            ast::JoinConstraint::On(expr) => Some(expr.to_string()),
            _ => None,
        }
    }

    fn table_factor_name(factor: &ast::TableFactor) -> String {
        match factor {
            ast::TableFactor::Table { name, .. } => name.to_string(),
            other => other.to_string(),
        }
    }

    /// The table's alias, if the query gave it one (`FROM t AS a` / `FROM t a`).
    fn table_factor_alias(factor: &ast::TableFactor) -> Option<String> {
        match factor {
            ast::TableFactor::Table { alias, .. } => alias.as_ref().map(|a| a.name.value.clone()),
            _ => None,
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
            .map(|c| {
                let mut nullable = true;
                let mut primary_key = false;
                for opt in &c.options {
                    match &opt.option {
                        ast::ColumnOption::NotNull => nullable = false,
                        ast::ColumnOption::Null => nullable = true,
                        ast::ColumnOption::PrimaryKey(_) => {
                            primary_key = true;
                            nullable = false; // a primary key is implicitly NOT NULL
                        }
                        _ => {}
                    }
                }
                ColumnDef {
                    name: c.name.to_string(),
                    data_type: c.data_type.to_string(),
                    nullable,
                    primary_key,
                }
            })
            .collect();

        Ok(SQLStatement::CreateTable(CreateTableStatement { name, columns }))
    }

    fn convert_create_index(create: ast::CreateIndex) -> Result<SQLStatement> {
        let name = create
            .name
            .map(|n| n.to_string())
            .ok_or_else(|| DatabaseError::ParseError("CREATE INDEX requires a name".to_string()))?;
        let table = create.table_name.to_string();
        let column = create
            .columns
            .first()
            .map(|c| c.column.expr.to_string())
            .ok_or_else(|| DatabaseError::ParseError("CREATE INDEX requires a column".to_string()))?;

        Ok(SQLStatement::CreateIndex(CreateIndexStatement { name, table, column }))
    }

    fn convert_analyze(analyze: ast::Analyze) -> Result<SQLStatement> {
        let table = analyze
            .table_name
            .map(|n| n.to_string())
            .ok_or_else(|| DatabaseError::ParseError("ANALYZE requires a table".to_string()))?;

        Ok(SQLStatement::Analyze(AnalyzeStatement { table }))
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
    fn test_parse_group_by() {
        let sql = "SELECT user_id, COUNT(*) FROM orders GROUP BY user_id";
        let stmt = SQLParser::parse(sql).unwrap();
        match stmt {
            SQLStatement::Select(s) => {
                assert_eq!(s.group_by, vec!["user_id".to_string()]);
            }
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_parse_select_without_group_by_is_empty() {
        let sql = "SELECT * FROM users";
        let stmt = SQLParser::parse(sql).unwrap();
        match stmt {
            SQLStatement::Select(s) => assert!(s.group_by.is_empty()),
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
        let sql = "CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(50), age INT NOT NULL)";
        let stmt = SQLParser::parse(sql).unwrap();
        match stmt {
            SQLStatement::CreateTable(c) => {
                assert_eq!(c.name, "users");
                assert_eq!(c.columns.len(), 3);
                assert_eq!(c.columns[0].name, "id");
                assert!(c.columns[0].primary_key);
                assert!(!c.columns[0].nullable);
                assert!(!c.columns[1].primary_key);
                assert!(c.columns[1].nullable);
                assert!(!c.columns[2].nullable);
            }
            _ => panic!("Expected CREATE TABLE statement"),
        }
    }

    #[test]
    fn test_parse_create_index() {
        let sql = "CREATE INDEX idx_users_name ON users (name)";
        let stmt = SQLParser::parse(sql).unwrap();
        match stmt {
            SQLStatement::CreateIndex(c) => {
                assert_eq!(c.name, "idx_users_name");
                assert_eq!(c.table, "users");
                assert_eq!(c.column, "name");
            }
            _ => panic!("Expected CREATE INDEX statement"),
        }
    }

    #[test]
    fn test_parse_invalid_sql_errors() {
        assert!(SQLParser::parse("SELECT * FRM users").is_err());
        assert!(SQLParser::parse("").is_err());
    }
}
