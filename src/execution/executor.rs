//! Query execution engine
//!
//! `execute_sql` is the real end-to-end entry point: parse -> bind -> (for
//! SELECT) plan -> execute against the OLTP engine, or (for INSERT/UPDATE/
//! DELETE) write rows directly. `execute(&PhysicalPlan)` compiles a
//! `Scan`/`Filter` plan into operators that actually read through
//! `OLTPEngine`'s MVCC snapshot. DML statements return the number of rows
//! affected, as `Ok(vec![vec![n.to_string()]])`.
//!
//! `CREATE TABLE` registers the parsed schema into both the catalog and the
//! OLTP store, so a table created this way can immediately be inserted
//! into. `Aggregate` nodes (COUNT/SUM/AVG/MIN/MAX with no GROUP BY) execute
//! for real, and so does `Join` — a nested-loop join whose `ON` condition
//! must compare two real columns (see `merge_schemas`); anything it can't
//! recognize that way errors rather than silently returning an unfiltered
//! cross product. There is no `GROUP BY`, and a `JOIN` condition using a
//! table alias (rather than the table's real name) won't resolve.

use crate::error::{DatabaseError, Result};
use crate::execution::aggregate;
use crate::execution::catalog::{Catalog, Column, DataType, TableSchema};
use crate::execution::mvcc_store::WriteOp;
use crate::execution::oltp::OLTPEngine;
use crate::execution::operators::{Tuple, Value};
use crate::execution::row_codec;
use crate::execution::transaction::TransactionId;
use crate::optimizer::planner::{LogicalPlanNode, PhysicalPlan, QueryPlanner};
use crate::sql::binder::Binder;
use crate::sql::parser::{
    CreateTableStatement, DeleteStatement, InsertStatement, SQLParser, SQLStatement, UpdateStatement,
};
use parking_lot::RwLock;

/// Query executor: owns the catalog, the OLTP engine, and a query planner,
/// and ties them together into a real (if still partial) SQL execution
/// path.
///
/// `catalog` is behind a lock (unlike a plain field) so `CREATE TABLE` can
/// register a new schema through `&self`, matching how everything else
/// here (`OLTPEngine`, `LockManager`, `MVCCStore`) uses interior mutability
/// rather than requiring `&mut self` — that keeps `QueryExecutor` usable
/// the same way those are: shared behind an `Arc` across threads.
pub struct QueryExecutor {
    pub catalog: RwLock<Catalog>,
    pub oltp: OLTPEngine,
    pub planner: QueryPlanner,
}

impl QueryExecutor {
    pub fn new(catalog: Catalog) -> Self {
        QueryExecutor {
            catalog: RwLock::new(catalog),
            oltp: OLTPEngine::new(),
            planner: QueryPlanner::new(),
        }
    }

    /// Parse, bind, and run a SQL statement end to end.
    pub fn execute_sql(&self, sql: &str) -> Result<Vec<Vec<String>>> {
        let stmt = SQLParser::parse(sql)?;
        {
            let catalog = self.catalog.read();
            Binder::new(&catalog).bind(&stmt)?;
        }

        match stmt {
            SQLStatement::Select(_) => {
                let plan = self.planner.plan(sql);
                self.execute(&plan)
            }
            SQLStatement::Insert(insert) => {
                let affected = self.execute_insert(insert)?;
                Ok(vec![vec![affected.to_string()]])
            }
            SQLStatement::Update(update) => {
                let affected = self.execute_update(update)?;
                Ok(vec![vec![affected.to_string()]])
            }
            SQLStatement::Delete(delete) => {
                let affected = self.execute_delete(delete)?;
                Ok(vec![vec![affected.to_string()]])
            }
            SQLStatement::CreateTable(create) => {
                self.execute_create_table(create)?;
                Ok(Vec::new())
            }
        }
    }

    /// Execute a query plan: compiles `Scan`/`Filter`/`Aggregate`/`Join`
    /// nodes into operators reading through the OLTP engine's MVCC
    /// snapshot. `Join` does a nested-loop join with a single-comparison
    /// `ON` condition into a merged row set — see `merge_schemas` for how
    /// column names disambiguate. A condition this executor can't
    /// recognize as comparing two real columns errors rather than silently
    /// degrading to an unfiltered cross product.
    pub fn execute(&self, plan: &PhysicalPlan) -> Result<Vec<Vec<String>>> {
        let mut current: Option<(TableSchema, Vec<Tuple>)> = None;

        for node in &plan.nodes {
            match node {
                LogicalPlanNode::Scan { table_name, .. } => {
                    current = Some(self.scan_table_tuples(table_name)?);
                }
                LogicalPlanNode::Filter { predicate, .. } => {
                    let (schema, tuples) = current
                        .take()
                        .ok_or_else(|| DatabaseError::ExecutionError("Filter with no input".to_string()))?;
                    let filtered = tuples
                        .into_iter()
                        .filter(|t| row_codec::evaluate_predicate(predicate, &schema, t).unwrap_or(true))
                        .collect();
                    current = Some((schema, filtered));
                }
                LogicalPlanNode::Join { right_table, condition, .. } => {
                    let (left_schema, left_tuples) = current
                        .take()
                        .ok_or_else(|| DatabaseError::ExecutionError("JOIN with no input".to_string()))?;
                    let (right_schema, right_tuples) = self.scan_table_tuples(right_table)?;
                    let merged_schema = Self::merge_schemas(&left_schema, &right_schema);

                    if let Some(cond) = condition {
                        let (left_tok, _, right_tok) = row_codec::split_comparison(cond).ok_or_else(|| {
                            DatabaseError::ExecutionError(format!("Unsupported JOIN condition: '{cond}'"))
                        })?;
                        let is_column = |tok: &str| merged_schema.columns.iter().any(|c| c.name == tok);
                        if !is_column(&left_tok) || !is_column(&right_tok) {
                            return Err(DatabaseError::ExecutionError(format!(
                                "JOIN condition must compare two columns (e.g. 'a.id = b.a_id'), got: '{cond}'"
                            )));
                        }
                    }

                    let mut merged_tuples = Vec::new();
                    for l in &left_tuples {
                        for r in &right_tuples {
                            let mut values = l.values.clone();
                            values.extend(r.values.clone());
                            let merged = Tuple { values };
                            let keep = match condition {
                                Some(cond) => {
                                    row_codec::evaluate_predicate(cond, &merged_schema, &merged).unwrap_or(true)
                                }
                                None => true, // CROSS JOIN (or USING/NATURAL, not specially resolved)
                            };
                            if keep {
                                merged_tuples.push(merged);
                            }
                        }
                    }
                    current = Some((merged_schema, merged_tuples));
                }
                LogicalPlanNode::Aggregate { columns, .. } => {
                    let (schema, tuples) = current
                        .take()
                        .ok_or_else(|| DatabaseError::ExecutionError("Aggregate with no input".to_string()))?;

                    let mut output = Vec::with_capacity(columns.len());
                    for col in columns {
                        let (func, arg) = aggregate::parse_aggregate(col).ok_or_else(|| {
                            DatabaseError::ExecutionError(format!("Not a recognized aggregate: '{col}'"))
                        })?;
                        output.push(aggregate::compute_aggregate(func, arg.as_deref(), &schema, &tuples));
                    }
                    current = Some((schema, vec![Tuple { values: output }]));
                }
            }
        }

        let tuples = current.map(|(_, t)| t).unwrap_or_default();
        Ok(tuples
            .into_iter()
            .map(|t| t.values.iter().map(row_codec::value_to_string).collect())
            .collect())
    }

    /// Read a table's full row set at a fresh snapshot (used by `Scan` and,
    /// for its right-hand side, `Join`).
    fn scan_table_tuples(&self, table_name: &str) -> Result<(TableSchema, Vec<Tuple>)> {
        let schema = self.catalog.read().get_table(table_name).cloned().ok_or_else(|| {
            DatabaseError::ExecutionError(format!("Unknown table '{table_name}'"))
        })?;
        let table_id = schema.table_id as u64;
        let rows = self.oltp.with_read_snapshot(|tx| self.oltp.scan_table(tx, table_id));
        let tuples = rows
            .into_iter()
            .filter_map(|(_, bytes)| bincode::deserialize::<Tuple>(&bytes).ok())
            .collect();
        Ok((schema, tuples))
    }

    /// Build the schema for a joined row: `left`'s columns followed by
    /// `right`'s, each renamed to `"<table>.<column>"` so (a) two tables
    /// with a same-named column don't collide, and (b) an `ON` condition
    /// like `"users.id = orders.user_id"` — rendered by `sql::parser`
    /// exactly that way for a qualified reference — resolves directly
    /// against these names. A condition using a table *alias* rather than
    /// its real name won't resolve; there's no alias tracking here.
    fn merge_schemas(left: &TableSchema, right: &TableSchema) -> TableSchema {
        let mut merged = TableSchema::new(0, format!("{}_{}", left.name, right.name));
        let mut next_id = 1u32;
        for (table, col) in left
            .columns
            .iter()
            .map(|c| (left, c))
            .chain(right.columns.iter().map(|c| (right, c)))
        {
            merged.add_column(Column {
                id: next_id,
                name: format!("{}.{}", table.name, col.name),
                data_type: col.data_type,
                nullable: col.nullable,
                primary_key: false,
            });
            next_id += 1;
        }
        merged
    }

    /// Execute an INSERT by writing rows directly to the OLTP engine.
    /// Returns the number of rows inserted.
    ///
    /// The row id is derived from the table's primary-key column: nothing
    /// in this codebase allocates a surrogate row id, so a table without a
    /// primary key can't be inserted into through this path.
    fn execute_insert(&self, insert: InsertStatement) -> Result<usize> {
        let schema = self.catalog.read().get_table(&insert.table).cloned().ok_or_else(|| {
            DatabaseError::ExecutionError(format!("Unknown table '{}'", insert.table))
        })?;
        let pk_index = schema.columns.iter().position(|c| c.primary_key).ok_or_else(|| {
            DatabaseError::ExecutionError(format!(
                "Table '{}' has no primary key; INSERT execution needs one to derive a row id",
                schema.name
            ))
        })?;

        let table_id = schema.table_id as u64;
        self.oltp.create_table(table_id);

        let tx = self.oltp.begin();
        for row in &insert.values {
            let ordered = match Self::order_insert_row(&insert, &schema, row) {
                Ok(v) => v,
                Err(e) => {
                    self.oltp.abort(tx);
                    return Err(e);
                }
            };

            let row_id = match &ordered[pk_index] {
                Value::Integer(i) => *i as u64,
                other => {
                    self.oltp.abort(tx);
                    return Err(DatabaseError::ExecutionError(format!(
                        "Primary key must be an integer, got {other:?}"
                    )));
                }
            };

            let data = match bincode::serialize(&Tuple { values: ordered }) {
                Ok(d) => d,
                Err(e) => {
                    self.oltp.abort(tx);
                    return Err(DatabaseError::SerializationError(e.to_string()));
                }
            };

            if let Err(e) = self.oltp.write(tx, WriteOp::Insert { table_id, row_id, data }) {
                self.oltp.abort(tx);
                return Err(e);
            }
        }
        let affected = insert.values.len();
        self.oltp.commit(tx);
        Ok(affected)
    }

    /// Execute an UPDATE: scans the table within one transaction, applies
    /// the SET assignments to every row matching WHERE (all rows if there's
    /// no WHERE), and writes each changed row back under its existing row
    /// id — unlike INSERT, there's no id to derive here, `scan_table`
    /// already hands back each row's real id. Returns the number of rows
    /// updated.
    fn execute_update(&self, update: UpdateStatement) -> Result<usize> {
        let schema = self.catalog.read().get_table(&update.table).cloned().ok_or_else(|| {
            DatabaseError::ExecutionError(format!("Unknown table '{}'", update.table))
        })?;
        let pk_index = schema.columns.iter().position(|c| c.primary_key);

        // Resolve assignment targets up front so a typo, or an attempt to
        // change the primary key's value, fails before any writes happen.
        let mut assignments = Vec::with_capacity(update.assignments.len());
        for (col_name, raw_value) in &update.assignments {
            let idx = schema
                .columns
                .iter()
                .position(|c| &c.name == col_name)
                .ok_or_else(|| DatabaseError::ExecutionError(format!("Unknown column '{col_name}'")))?;
            if Some(idx) == pk_index {
                return Err(DatabaseError::ExecutionError(
                    "Updating the primary key column is not supported".to_string(),
                ));
            }
            assignments.push((idx, row_codec::parse_value(raw_value, schema.columns[idx].data_type)));
        }

        let table_id = schema.table_id as u64;
        self.oltp.create_table(table_id);

        let tx = self.oltp.begin();
        let rows = self.oltp.scan_table(tx, table_id);

        let mut affected = 0usize;
        for (row_id, bytes) in rows {
            let Ok(mut tuple) = bincode::deserialize::<Tuple>(&bytes) else {
                continue; // unreadable row: skip rather than fail the whole statement
            };

            let matches = match &update.where_clause {
                Some(predicate) => row_codec::evaluate_predicate(predicate, &schema, &tuple).unwrap_or(true),
                None => true,
            };
            if !matches {
                continue;
            }

            for (idx, value) in &assignments {
                tuple.values[*idx] = value.clone();
            }

            let data = match bincode::serialize(&tuple) {
                Ok(d) => d,
                Err(e) => {
                    self.oltp.abort(tx);
                    return Err(DatabaseError::SerializationError(e.to_string()));
                }
            };

            if let Err(e) = self.oltp.write(tx, WriteOp::Update { table_id, row_id, data }) {
                self.oltp.abort(tx);
                return Err(e);
            }
            affected += 1;
        }

        self.oltp.commit(tx);
        Ok(affected)
    }

    /// Execute a DELETE: scans the table within one transaction and deletes
    /// every row matching WHERE (all rows if there's no WHERE). Returns the
    /// number of rows deleted.
    fn execute_delete(&self, delete: DeleteStatement) -> Result<usize> {
        let schema = self.catalog.read().get_table(&delete.table).cloned().ok_or_else(|| {
            DatabaseError::ExecutionError(format!("Unknown table '{}'", delete.table))
        })?;
        let table_id = schema.table_id as u64;
        self.oltp.create_table(table_id);

        let tx = self.oltp.begin();
        let rows = self.oltp.scan_table(tx, table_id);

        let mut affected = 0usize;
        for (row_id, bytes) in rows {
            let Ok(tuple) = bincode::deserialize::<Tuple>(&bytes) else {
                continue;
            };

            let matches = match &delete.where_clause {
                Some(predicate) => row_codec::evaluate_predicate(predicate, &schema, &tuple).unwrap_or(true),
                None => true,
            };
            if !matches {
                continue;
            }

            if let Err(e) = self.oltp.write(tx, WriteOp::Delete { table_id, row_id }) {
                self.oltp.abort(tx);
                return Err(e);
            }
            affected += 1;
        }

        self.oltp.commit(tx);
        Ok(affected)
    }

    /// Execute a CREATE TABLE: registers the parsed schema in the catalog
    /// and creates the matching (empty) table in the OLTP store, so an
    /// INSERT against it works immediately without a separate step.
    fn execute_create_table(&self, create: CreateTableStatement) -> Result<()> {
        let mut catalog = self.catalog.write();
        if catalog.get_table(&create.name).is_some() {
            return Err(DatabaseError::ExecutionError(format!(
                "Table '{}' already exists",
                create.name
            )));
        }

        let table_id = catalog.tables.keys().copied().max().unwrap_or(0) + 1;
        let mut schema = TableSchema::new(table_id, create.name.clone());
        for (i, col) in create.columns.iter().enumerate() {
            schema.add_column(Column {
                id: (i + 1) as u32,
                name: col.name.clone(),
                data_type: Self::map_ddl_type(&col.data_type),
                nullable: col.nullable,
                primary_key: col.primary_key,
            });
        }
        catalog.register_table(schema);
        drop(catalog);

        self.oltp.create_table(table_id as u64);
        Ok(())
    }

    /// Map a rendered DDL type string (e.g. `"VARCHAR(50)"`, `"BIGINT"`) onto
    /// the catalog's coarser `DataType`. A heuristic substring match, since
    /// `DataType` doesn't track length/precision at all — good enough given
    /// the executor only branches on which of the 5 variants a column is.
    fn map_ddl_type(raw: &str) -> DataType {
        let upper = raw.to_uppercase();
        if upper.contains("BOOL") {
            DataType::Boolean
        } else if upper.contains("TIMESTAMP") || upper.contains("DATE") {
            DataType::Timestamp
        } else if upper.contains("INT") {
            DataType::Integer
        } else if upper.contains("FLOAT")
            || upper.contains("DOUBLE")
            || upper.contains("DECIMAL")
            || upper.contains("NUMERIC")
            || upper.contains("REAL")
        {
            DataType::Float
        } else {
            DataType::String
        }
    }

    /// Map one INSERT value row (raw strings, in either explicit-column or
    /// schema-column order) into typed `Value`s in schema column order.
    fn order_insert_row(insert: &InsertStatement, schema: &TableSchema, row: &[String]) -> Result<Vec<Value>> {
        if insert.columns.is_empty() {
            return Ok(schema
                .columns
                .iter()
                .zip(row)
                .map(|(c, v)| row_codec::parse_value(v, c.data_type))
                .collect());
        }

        let mut ordered = vec![Value::Null; schema.columns.len()];
        for (col_name, raw) in insert.columns.iter().zip(row) {
            let idx = schema
                .columns
                .iter()
                .position(|c| &c.name == col_name)
                .ok_or_else(|| DatabaseError::ExecutionError(format!("Unknown column '{col_name}'")))?;
            ordered[idx] = row_codec::parse_value(raw, schema.columns[idx].data_type);
        }
        Ok(ordered)
    }

    /// Begin a new transaction against the OLTP engine.
    pub fn begin_transaction(&self) -> TransactionId {
        self.oltp.begin()
    }

    /// Transactional point read of a row, honoring 2PL locking and MVCC
    /// snapshot isolation.
    pub fn read_row(
        &self,
        tx_id: TransactionId,
        table_id: u64,
        row_id: u64,
    ) -> Result<Option<Vec<u8>>> {
        self.oltp.read(tx_id, table_id, row_id)
    }

    /// Buffer a transactional write; not visible to other transactions until
    /// `commit_transaction`.
    pub fn write_row(&self, tx_id: TransactionId, op: WriteOp) -> Result<()> {
        self.oltp.write(tx_id, op)
    }

    pub fn commit_transaction(&self, tx_id: TransactionId) {
        self.oltp.commit(tx_id);
    }

    pub fn abort_transaction(&self, tx_id: TransactionId) {
        self.oltp.abort(tx_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::catalog::{Column, DataType};
    use crate::execution::mvcc_store::WriteOp;

    #[test]
    fn test_executor_creation() {
        let catalog = Catalog::new();
        let executor = QueryExecutor::new(catalog);
        assert_eq!(executor.catalog.read().tables.len(), 0);
    }

    #[test]
    fn test_executor_transactional_read_write() {
        let executor = QueryExecutor::new(Catalog::new());
        executor.oltp.create_table(1);

        let tx = executor.begin_transaction();
        executor
            .write_row(
                tx,
                WriteOp::Insert {
                    table_id: 1,
                    row_id: 1,
                    data: b"row".to_vec(),
                },
            )
            .unwrap();
        assert_eq!(executor.read_row(tx, 1, 1).unwrap(), Some(b"row".to_vec()));
        executor.commit_transaction(tx);

        let tx2 = executor.begin_transaction();
        assert_eq!(executor.read_row(tx2, 1, 1).unwrap(), Some(b"row".to_vec()));
    }

    fn users_catalog() -> Catalog {
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
        schema.add_column(Column {
            id: 3,
            name: "age".to_string(),
            data_type: DataType::Integer,
            nullable: false,
            primary_key: false,
        });
        catalog.register_table(schema);
        catalog
    }

    #[test]
    fn test_insert_then_select_end_to_end() {
        let executor = QueryExecutor::new(users_catalog());

        executor
            .execute_sql("INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)")
            .unwrap();
        executor
            .execute_sql("INSERT INTO users (id, name, age) VALUES (2, 'Bob', 15)")
            .unwrap();

        let rows = executor.execute_sql("SELECT * FROM users").unwrap();
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_select_with_where_filters_rows() {
        let executor = QueryExecutor::new(users_catalog());
        executor
            .execute_sql("INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)")
            .unwrap();
        executor
            .execute_sql("INSERT INTO users (id, name, age) VALUES (2, 'Bob', 15)")
            .unwrap();

        let rows = executor
            .execute_sql("SELECT * FROM users WHERE age > 18")
            .unwrap();
        assert_eq!(rows.len(), 1);
        assert!(rows[0].contains(&"Alice".to_string()));
    }

    #[test]
    fn test_select_unknown_table_errors() {
        let executor = QueryExecutor::new(users_catalog());
        assert!(executor.execute_sql("SELECT * FROM ghosts").is_err());
    }

    #[test]
    fn test_insert_without_primary_key_column_fails() {
        let mut catalog = Catalog::new();
        let mut schema = TableSchema::new(1, "logs".to_string());
        schema.add_column(Column {
            id: 1,
            name: "message".to_string(),
            data_type: DataType::String,
            nullable: false,
            primary_key: false,
        });
        catalog.register_table(schema);
        let executor = QueryExecutor::new(catalog);

        assert!(executor
            .execute_sql("INSERT INTO logs (message) VALUES ('hi')")
            .is_err());
    }

    #[test]
    fn test_select_on_table_with_no_rows_yet_is_empty() {
        let executor = QueryExecutor::new(users_catalog());
        let rows = executor.execute_sql("SELECT * FROM users").unwrap();
        assert!(rows.is_empty());
    }

    fn seed_users(executor: &QueryExecutor) {
        executor
            .execute_sql("INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)")
            .unwrap();
        executor
            .execute_sql("INSERT INTO users (id, name, age) VALUES (2, 'Bob', 15)")
            .unwrap();
    }

    #[test]
    fn test_update_with_where_changes_matching_rows_only() {
        let executor = QueryExecutor::new(users_catalog());
        seed_users(&executor);

        let result = executor
            .execute_sql("UPDATE users SET age = 31 WHERE name = 'Alice'")
            .unwrap();
        assert_eq!(result, vec![vec!["1".to_string()]]); // 1 row affected

        let rows = executor.execute_sql("SELECT * FROM users").unwrap();
        let alice = rows.iter().find(|r| r.contains(&"Alice".to_string())).unwrap();
        assert!(alice.contains(&"31".to_string()));
        let bob = rows.iter().find(|r| r.contains(&"Bob".to_string())).unwrap();
        assert!(bob.contains(&"15".to_string())); // untouched
    }

    #[test]
    fn test_update_without_where_changes_all_rows() {
        let executor = QueryExecutor::new(users_catalog());
        seed_users(&executor);

        let result = executor.execute_sql("UPDATE users SET age = 0").unwrap();
        assert_eq!(result, vec![vec!["2".to_string()]]);

        let rows = executor.execute_sql("SELECT * FROM users").unwrap();
        assert!(rows.iter().all(|r| r.contains(&"0".to_string())));
    }

    #[test]
    fn test_update_primary_key_column_is_rejected() {
        let executor = QueryExecutor::new(users_catalog());
        seed_users(&executor);
        assert!(executor.execute_sql("UPDATE users SET id = 99 WHERE id = 1").is_err());
    }

    #[test]
    fn test_delete_with_where_removes_matching_rows_only() {
        let executor = QueryExecutor::new(users_catalog());
        seed_users(&executor);

        let result = executor
            .execute_sql("DELETE FROM users WHERE age < 18")
            .unwrap();
        assert_eq!(result, vec![vec!["1".to_string()]]);

        let rows = executor.execute_sql("SELECT * FROM users").unwrap();
        assert_eq!(rows.len(), 1);
        assert!(rows[0].contains(&"Alice".to_string()));
    }

    #[test]
    fn test_delete_without_where_removes_all_rows() {
        let executor = QueryExecutor::new(users_catalog());
        seed_users(&executor);

        let result = executor.execute_sql("DELETE FROM users").unwrap();
        assert_eq!(result, vec![vec!["2".to_string()]]);

        let rows = executor.execute_sql("SELECT * FROM users").unwrap();
        assert!(rows.is_empty());
    }

    #[test]
    fn test_update_unknown_table_errors() {
        let executor = QueryExecutor::new(users_catalog());
        assert!(executor.execute_sql("UPDATE ghosts SET x = 1").is_err());
    }

    #[test]
    fn test_create_table_then_insert_and_select_end_to_end() {
        let executor = QueryExecutor::new(Catalog::new());

        executor
            .execute_sql("CREATE TABLE items (id INT PRIMARY KEY, label VARCHAR(50), price FLOAT)")
            .unwrap();

        assert!(executor.catalog.read().get_table("items").is_some());

        executor
            .execute_sql("INSERT INTO items (id, label, price) VALUES (1, 'Widget', 9.99)")
            .unwrap();

        let rows = executor.execute_sql("SELECT * FROM items").unwrap();
        assert_eq!(rows.len(), 1);
        assert!(rows[0].contains(&"Widget".to_string()));
    }

    #[test]
    fn test_create_table_maps_column_types() {
        let executor = QueryExecutor::new(Catalog::new());
        executor
            .execute_sql(
                "CREATE TABLE things (id INT PRIMARY KEY, active BOOLEAN, created TIMESTAMP, note TEXT)",
            )
            .unwrap();

        let catalog = executor.catalog.read();
        let schema = catalog.get_table("things").unwrap();
        assert_eq!(schema.get_column("id").unwrap().data_type, DataType::Integer);
        assert_eq!(schema.get_column("active").unwrap().data_type, DataType::Boolean);
        assert_eq!(schema.get_column("created").unwrap().data_type, DataType::Timestamp);
        assert_eq!(schema.get_column("note").unwrap().data_type, DataType::String);
        assert!(schema.get_column("id").unwrap().primary_key);
        assert!(!schema.get_column("active").unwrap().primary_key);
    }

    #[test]
    fn test_create_table_duplicate_name_errors() {
        let executor = QueryExecutor::new(users_catalog());
        assert!(executor
            .execute_sql("CREATE TABLE users (id INT PRIMARY KEY)")
            .is_err());
    }

    #[test]
    fn test_select_count_star() {
        let executor = QueryExecutor::new(users_catalog());
        seed_users(&executor);

        let rows = executor.execute_sql("SELECT COUNT(*) FROM users").unwrap();
        assert_eq!(rows, vec![vec!["2".to_string()]]);
    }

    #[test]
    fn test_select_aggregate_respects_where_clause() {
        let executor = QueryExecutor::new(users_catalog());
        seed_users(&executor); // Alice 30, Bob 15

        let rows = executor
            .execute_sql("SELECT COUNT(*) FROM users WHERE age > 18")
            .unwrap();
        assert_eq!(rows, vec![vec!["1".to_string()]]);
    }

    #[test]
    fn test_select_multiple_aggregates() {
        let executor = QueryExecutor::new(users_catalog());
        seed_users(&executor); // ages 30, 15

        let rows = executor
            .execute_sql("SELECT COUNT(*), SUM(age), AVG(age), MIN(age), MAX(age) FROM users")
            .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][0], "2"); // count
        assert_eq!(rows[0][1], "45"); // sum
        assert_eq!(rows[0][3], "15"); // min
        assert_eq!(rows[0][4], "30"); // max
    }

    #[test]
    fn test_select_aggregate_on_empty_table() {
        let executor = QueryExecutor::new(users_catalog());
        let rows = executor.execute_sql("SELECT COUNT(*) FROM users").unwrap();
        assert_eq!(rows, vec![vec!["0".to_string()]]);
    }

    #[test]
    fn test_join_condition_on_nonexistent_column_errors() {
        let mut catalog = users_catalog();
        let orders_schema = TableSchema::new(2, "orders".to_string()); // no columns at all
        catalog.register_table(orders_schema);
        let executor = QueryExecutor::new(catalog);

        assert!(executor
            .execute_sql("SELECT * FROM users JOIN orders ON users.id = orders.user_id")
            .is_err());
    }

    fn users_and_orders_catalog() -> Catalog {
        let mut catalog = users_catalog();
        let mut orders = TableSchema::new(2, "orders".to_string());
        orders.add_column(Column {
            id: 1,
            name: "id".to_string(),
            data_type: DataType::Integer,
            nullable: false,
            primary_key: true,
        });
        orders.add_column(Column {
            id: 2,
            name: "user_id".to_string(),
            data_type: DataType::Integer,
            nullable: false,
            primary_key: false,
        });
        orders.add_column(Column {
            id: 3,
            name: "total".to_string(),
            data_type: DataType::Float,
            nullable: false,
            primary_key: false,
        });
        catalog.register_table(orders);
        catalog
    }

    #[test]
    fn test_join_with_on_condition_matches_correct_rows() {
        let executor = QueryExecutor::new(users_and_orders_catalog());
        seed_users(&executor); // ids 1 (Alice), 2 (Bob)

        executor
            .execute_sql("INSERT INTO orders (id, user_id, total) VALUES (100, 1, 9.5)")
            .unwrap();
        executor
            .execute_sql("INSERT INTO orders (id, user_id, total) VALUES (101, 2, 4.0)")
            .unwrap();
        executor
            .execute_sql("INSERT INTO orders (id, user_id, total) VALUES (102, 1, 2.0)")
            .unwrap();

        let rows = executor
            .execute_sql("SELECT * FROM users JOIN orders ON users.id = orders.user_id")
            .unwrap();

        assert_eq!(rows.len(), 3); // Alice has 2 orders, Bob has 1
        let alice_orders = rows.iter().filter(|r| r.contains(&"Alice".to_string())).count();
        assert_eq!(alice_orders, 2);
    }

    #[test]
    fn test_join_with_where_filters_after_joining() {
        let executor = QueryExecutor::new(users_and_orders_catalog());
        seed_users(&executor);
        executor
            .execute_sql("INSERT INTO orders (id, user_id, total) VALUES (100, 1, 9.5)")
            .unwrap();
        executor
            .execute_sql("INSERT INTO orders (id, user_id, total) VALUES (101, 2, 4.0)")
            .unwrap();

        let rows = executor
            .execute_sql(
                "SELECT * FROM users JOIN orders ON users.id = orders.user_id WHERE users.age > 18",
            )
            .unwrap();

        assert_eq!(rows.len(), 1);
        assert!(rows[0].contains(&"Alice".to_string()));
    }

    #[test]
    fn test_join_with_no_matches_returns_empty() {
        let executor = QueryExecutor::new(users_and_orders_catalog());
        seed_users(&executor);
        // No orders inserted at all.

        let rows = executor
            .execute_sql("SELECT * FROM users JOIN orders ON users.id = orders.user_id")
            .unwrap();
        assert!(rows.is_empty());
    }
}
