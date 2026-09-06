//! Semantic analysis and binding
//!
//! Resolves table/column references in a parsed `SQLStatement` against the
//! catalog before it reaches the planner, catching typos and schema
//! mismatches (unknown table, unknown column, wrong arity in an INSERT)
//! instead of letting them surface later as a confusing execution failure.
//!
//! Projected columns and assignment targets are strings (see
//! `sql::parser`'s doc comment), which can be a bare column name or an
//! arbitrary rendered expression (`COUNT(*)`, `a + b`, `col AS alias`, ...).
//! Only bare identifiers are checked against the schema — there's no
//! expression-tree binder downstream yet to validate the rest against.

use crate::error::{DatabaseError, Result};
use crate::execution::catalog::{Catalog, TableSchema};
use crate::sql::parser::SQLStatement;

/// Binder for semantic analysis
pub struct Binder {
    pub catalog: Catalog,
}

impl Binder {
    pub fn new(catalog: Catalog) -> Self {
        Binder { catalog }
    }

    /// Bind a SQL statement against the catalog: table and (simple) column
    /// references must exist, and CREATE TABLE must not collide with an
    /// existing table.
    pub fn bind(&self, stmt: &SQLStatement) -> Result<()> {
        match stmt {
            SQLStatement::Select(s) => {
                let table = self.require_table(&s.from)?;
                for col in &s.columns {
                    self.check_column(table, col)?;
                }
                Ok(())
            }
            SQLStatement::Insert(i) => {
                let table = self.require_table(&i.table)?;
                for col in &i.columns {
                    self.check_column(table, col)?;
                }
                let expected_arity = if i.columns.is_empty() {
                    table.columns.len()
                } else {
                    i.columns.len()
                };
                for row in &i.values {
                    if row.len() != expected_arity {
                        return Err(DatabaseError::BindingError(format!(
                            "INSERT into '{}' expects {} values, got {}",
                            table.name,
                            expected_arity,
                            row.len()
                        )));
                    }
                }
                Ok(())
            }
            SQLStatement::Update(u) => {
                let table = self.require_table(&u.table)?;
                for (col, _) in &u.assignments {
                    self.check_column(table, col)?;
                }
                Ok(())
            }
            SQLStatement::Delete(d) => {
                self.require_table(&d.table)?;
                Ok(())
            }
            SQLStatement::CreateTable(c) => {
                if self.catalog.get_table(&c.name).is_some() {
                    return Err(DatabaseError::BindingError(format!(
                        "Table '{}' already exists",
                        c.name
                    )));
                }
                Ok(())
            }
        }
    }

    fn require_table(&self, name: &str) -> Result<&TableSchema> {
        self.catalog
            .get_table(name)
            .ok_or_else(|| DatabaseError::BindingError(format!("Table '{name}' does not exist")))
    }

    fn check_column(&self, table: &TableSchema, name: &str) -> Result<()> {
        if name == "*" || !Self::is_simple_identifier(name) {
            // Wildcard, or a non-trivial expression (function call, alias,
            // arithmetic, ...) that this level can't meaningfully resolve.
            return Ok(());
        }
        if table.get_column(name).is_some() {
            Ok(())
        } else {
            Err(DatabaseError::BindingError(format!(
                "Column '{name}' does not exist on table '{}'",
                table.name
            )))
        }
    }

    fn is_simple_identifier(s: &str) -> bool {
        !s.is_empty() && s.chars().all(|c| c.is_alphanumeric() || c == '_')
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::catalog::{Column, DataType, TableSchema};
    use crate::sql::parser::SQLParser;

    fn catalog_with_users() -> Catalog {
        let mut catalog = Catalog::new();
        let mut schema = TableSchema::new(1, "users".to_string());
        schema.add_column(Column {
            id: 1,
            name: "id".to_string(),
            data_type: DataType::Integer,
            nullable: false,
            primary_key: true,
        });
        schema.add_column(Column {
            id: 2,
            name: "name".to_string(),
            data_type: DataType::String,
            nullable: false,
            primary_key: false,
        });
        catalog.register_table(schema);
        catalog
    }

    #[test]
    fn test_bind_select_success() {
        let binder = Binder::new(catalog_with_users());
        let stmt = SQLParser::parse("SELECT id, name FROM users").unwrap();
        assert!(binder.bind(&stmt).is_ok());
    }

    #[test]
    fn test_bind_select_unknown_table() {
        let binder = Binder::new(catalog_with_users());
        let stmt = SQLParser::parse("SELECT * FROM ghosts").unwrap();
        assert!(binder.bind(&stmt).is_err());
    }

    #[test]
    fn test_bind_select_unknown_column() {
        let binder = Binder::new(catalog_with_users());
        let stmt = SQLParser::parse("SELECT nope FROM users").unwrap();
        assert!(binder.bind(&stmt).is_err());
    }

    #[test]
    fn test_bind_select_star_and_expressions_skip_column_check() {
        let binder = Binder::new(catalog_with_users());
        assert!(binder.bind(&SQLParser::parse("SELECT * FROM users").unwrap()).is_ok());
        assert!(binder
            .bind(&SQLParser::parse("SELECT COUNT(*) FROM users").unwrap())
            .is_ok());
    }

    #[test]
    fn test_bind_insert_arity_mismatch() {
        let binder = Binder::new(catalog_with_users());
        let stmt = SQLParser::parse("INSERT INTO users (id, name) VALUES (1)").unwrap();
        assert!(binder.bind(&stmt).is_err());
    }

    #[test]
    fn test_bind_insert_success() {
        let binder = Binder::new(catalog_with_users());
        let stmt = SQLParser::parse("INSERT INTO users (id, name) VALUES (1, 'Alice')").unwrap();
        assert!(binder.bind(&stmt).is_ok());
    }

    #[test]
    fn test_bind_update_unknown_column() {
        let binder = Binder::new(catalog_with_users());
        let stmt = SQLParser::parse("UPDATE users SET nope = 1 WHERE id = 1").unwrap();
        assert!(binder.bind(&stmt).is_err());
    }

    #[test]
    fn test_bind_create_table_duplicate() {
        let binder = Binder::new(catalog_with_users());
        let stmt = SQLParser::parse("CREATE TABLE users (id INT)").unwrap();
        assert!(binder.bind(&stmt).is_err());
    }

    #[test]
    fn test_bind_create_table_new() {
        let binder = Binder::new(catalog_with_users());
        let stmt = SQLParser::parse("CREATE TABLE orders (id INT)").unwrap();
        assert!(binder.bind(&stmt).is_ok());
    }
}
