//! Catalog for schema and metadata management

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Column definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Column {
    pub id: u32,
    pub name: String,
    pub data_type: DataType,
    pub nullable: bool,
    pub primary_key: bool,
}

/// Data types
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum DataType {
    Integer,
    Float,
    String,
    Boolean,
    Timestamp,
    /// Fixed-dimension floating-point vector for embeddings / similarity
    /// search. `Vector(384)` stores exactly 384 `f64` components —
    /// analogous to pgvector's `vector(384)`.
    Vector(usize),
}

/// Table schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableSchema {
    pub table_id: u32,
    pub name: String,
    pub columns: Vec<Column>,
}

impl TableSchema {
    pub fn new(table_id: u32, name: String) -> Self {
        TableSchema {
            table_id,
            name,
            columns: Vec::new(),
        }
    }

    pub fn add_column(&mut self, column: Column) {
        self.columns.push(column);
    }

    pub fn get_column(&self, name: &str) -> Option<&Column> {
        self.columns.iter().find(|c| c.name == name)
    }
}

/// Catalog for managing schemas and metadata
pub struct Catalog {
    pub tables: HashMap<u32, TableSchema>,
    pub table_names: HashMap<String, u32>,
}

impl Catalog {
    pub fn new() -> Self {
        Catalog {
            tables: HashMap::new(),
            table_names: HashMap::new(),
        }
    }

    /// Register a table schema
    pub fn register_table(&mut self, schema: TableSchema) {
        self.table_names
            .insert(schema.name.clone(), schema.table_id);
        self.tables.insert(schema.table_id, schema);
    }

    /// Get table schema by name
    pub fn get_table(&self, name: &str) -> Option<&TableSchema> {
        self.table_names
            .get(name)
            .and_then(|id| self.tables.get(id))
    }

    /// Get table schema by ID
    pub fn get_table_by_id(&self, table_id: u32) -> Option<&TableSchema> {
        self.tables.get(&table_id)
    }
}

impl Default for Catalog {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_catalog() {
        let mut catalog = Catalog::new();
        let mut schema = TableSchema::new(1, "users".to_string());
        schema.add_column(Column {
            id: 1,
            name: "id".to_string(),
            data_type: DataType::Integer,
            nullable: false,
            primary_key: true,
        });

        catalog.register_table(schema);

        let table = catalog.get_table("users");
        assert!(table.is_some());
        assert_eq!(table.unwrap().columns.len(), 1);
    }
}
