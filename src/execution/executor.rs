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
//! must compare two real columns (see `merge_schemas`), qualified by either
//! the table's real name or its alias if the query gave it one; anything it
//! can't recognize that way errors rather than silently returning an
//! unfiltered cross product.
//!
//! `new()` is in-memory only, same as always. `open(path)` is the durable
//! entry point: every CREATE TABLE and every committed write is logged to
//! a WAL at `path` before it takes effect (see `execution::oltp` and
//! `execution::wal`), and `open` replays whatever's already in that file
//! to rebuild catalog + row state before returning — so a restart doesn't
//! lose data.

use std::collections::HashMap;
use std::path::Path;

use crate::error::{DatabaseError, Result};
use crate::execution::aggregate;
use crate::execution::catalog::{Catalog, Column, DataType, TableSchema};
use crate::execution::mvcc_store::WriteOp;
use crate::execution::oltp::OLTPEngine;
use crate::execution::operators::{Tuple, Value};
use crate::execution::recovery::RecoveryManager;
use crate::execution::row_codec;
use crate::execution::secondary_index::SecondaryIndex;
use crate::execution::transaction::TransactionId;
use crate::execution::wal::WriteAheadLog;
use crate::optimizer::planner::{LogicalPlanNode, PhysicalPlan, QueryPlanner};
use crate::sql::binder::Binder;
use crate::sql::parser::{
    CreateIndexStatement, CreateTableStatement, DeleteStatement, InsertStatement, SQLParser, SQLStatement,
    UpdateStatement,
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
    /// Secondary indexes created via `CREATE INDEX`, keyed by (table id,
    /// indexed column name). Live at this layer rather than inside
    /// `MVCCTable`/`OLTPEngine` because building and maintaining one needs
    /// schema knowledge (which byte offset in a deserialized `Tuple` holds
    /// the indexed column) that those lower, byte-oriented layers
    /// deliberately don't have.
    secondary_indexes: RwLock<HashMap<(u64, String), SecondaryIndex>>,
}

impl QueryExecutor {
    /// In-memory only: nothing here survives a restart. What every
    /// existing test uses.
    pub fn new(catalog: Catalog) -> Self {
        QueryExecutor {
            catalog: RwLock::new(catalog),
            oltp: OLTPEngine::new(),
            planner: QueryPlanner::new(),
            secondary_indexes: RwLock::new(HashMap::new()),
        }
    }

    /// Durable: opens (creating if needed) a WAL at `path`, replays
    /// whatever's already logged there to rebuild the catalog and row
    /// state, then returns an executor where every future CREATE TABLE and
    /// every future commit is logged to that file before it takes effect.
    /// Any `CREATE INDEX`es logged in the WAL are rebuilt by backfill-
    /// scanning their table's now-recovered rows (see
    /// `rebuild_secondary_index`) — the index structure itself isn't what
    /// gets persisted, just the fact that it should exist.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let (wal, records) = WriteAheadLog::open(path)?;
        let mut catalog = Catalog::new();
        let oltp = OLTPEngine::with_wal(wal);
        let index_specs = RecoveryManager::recover(records, &mut catalog, &oltp);

        let executor = QueryExecutor {
            catalog: RwLock::new(catalog),
            oltp,
            planner: QueryPlanner::new(),
            secondary_indexes: RwLock::new(HashMap::new()),
        };

        for (table_id, column) in index_specs {
            executor.rebuild_secondary_index(table_id, &column)?;
        }

        Ok(executor)
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
            SQLStatement::CreateIndex(create) => {
                self.execute_create_index(create)?;
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
    ///
    /// A `Scan` immediately followed by a `Filter` comparing the primary
    /// key to a literal (`id = 5`, `id > 10`, ...) is compiled into an
    /// index-accelerated lookup (see `try_indexed_scan`) instead of a full
    /// table scan — this is the one place the learned PGM index built on
    /// `MVCCTable` (see `execution::mvcc_store`) actually gets used by a
    /// query, rather than sitting proven only by its own unit tests.
    /// Anything else still scans the whole table; that's always correct,
    /// just not accelerated.
    pub fn execute(&self, plan: &PhysicalPlan) -> Result<Vec<Vec<String>>> {
        let mut current: Option<(TableSchema, Vec<Tuple>)> = None;
        // The qualifier a subsequent Join should prefix `current`'s columns
        // with. `Some` right after a Scan (its alias, or the real table name
        // if none was given); taken (leaving `None`) the first time it's
        // consumed by a Join, since a merged schema's columns are already
        // qualified and must not be prefixed again on a later join in a
        // chain — see `merge_schemas`.
        let mut current_qualifier: Option<String> = None;
        let mut nodes = plan.nodes.iter().peekable();

        while let Some(node) = nodes.next() {
            match node {
                LogicalPlanNode::Scan { table_name, alias, .. } => {
                    if let Some(LogicalPlanNode::Filter { predicate, .. }) = nodes.peek() {
                        if let Some(indexed) = self.try_indexed_scan(table_name, predicate)? {
                            nodes.next(); // the Filter is already applied by the index lookup
                            current = Some(indexed);
                            current_qualifier = Some(alias.clone().unwrap_or_else(|| table_name.clone()));
                            continue;
                        }
                    }
                    current = Some(self.scan_table_tuples(table_name)?);
                    current_qualifier = Some(alias.clone().unwrap_or_else(|| table_name.clone()));
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
                LogicalPlanNode::Join { right_table, right_alias, condition, .. } => {
                    let (left_schema, left_tuples) = current
                        .take()
                        .ok_or_else(|| DatabaseError::ExecutionError("JOIN with no input".to_string()))?;
                    let (right_schema, right_tuples) = self.scan_table_tuples(right_table)?;
                    let left_prefix = current_qualifier.take();
                    let right_prefix = right_alias.clone().unwrap_or_else(|| right_table.clone());
                    let merged_schema =
                        Self::merge_schemas(&left_schema, left_prefix.as_deref(), &right_schema, &right_prefix);

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
                LogicalPlanNode::Aggregate { columns, group_by, .. } => {
                    let (schema, tuples) = current
                        .take()
                        .ok_or_else(|| DatabaseError::ExecutionError("Aggregate with no input".to_string()))?;

                    // Resolve each GROUP BY column to its position in the
                    // row up front, then partition rows into groups by
                    // their key values. No GROUP BY means one group
                    // holding every row -- the pre-existing ungrouped
                    // behavior.
                    let group_indices = group_by
                        .iter()
                        .map(|g| {
                            schema.columns.iter().position(|c| &c.name == g).ok_or_else(|| {
                                DatabaseError::ExecutionError(format!("Unknown GROUP BY column '{g}'"))
                            })
                        })
                        .collect::<Result<Vec<usize>>>()?;

                    let groups: Vec<(Vec<Value>, Vec<Tuple>)> = if group_indices.is_empty() {
                        vec![(Vec::new(), tuples)]
                    } else {
                        let mut groups: Vec<(Vec<Value>, Vec<Tuple>)> = Vec::new();
                        for tuple in tuples {
                            let key: Vec<Value> = group_indices.iter().map(|&i| tuple.values[i].clone()).collect();
                            match groups.iter_mut().find(|(k, _)| k == &key) {
                                Some((_, rows)) => rows.push(tuple),
                                None => groups.push((key, vec![tuple])),
                            }
                        }
                        groups
                    };

                    let mut output_rows = Vec::with_capacity(groups.len());
                    for (key, group_tuples) in &groups {
                        let mut output = Vec::with_capacity(columns.len());
                        for col in columns {
                            if let Some(pos) = group_by.iter().position(|g| g == col) {
                                output.push(key[pos].clone());
                            } else {
                                let (func, arg) = aggregate::parse_aggregate(col).ok_or_else(|| {
                                    DatabaseError::ExecutionError(format!("Not a recognized aggregate: '{col}'"))
                                })?;
                                output.push(aggregate::compute_aggregate(func, arg.as_deref(), &schema, group_tuples));
                            }
                        }
                        output_rows.push(Tuple { values: output });
                    }
                    current = Some((schema, output_rows));
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

    /// If `predicate` is a recognized comparison of `table_name`'s primary
    /// key against a literal (`id = 5`, `id > 10`, ...), use the learned
    /// PGM index (`MVCCTable::index_range`) to fetch only the candidate
    /// rows instead of scanning the whole table. Each candidate is still
    /// confirmed with a real point read — the index can hand back ids for
    /// since-deleted rows, and that read (not the index) is what actually
    /// decides visibility. `Ok(None)` for anything not shaped this way
    /// (non-PK column, `!=`, an unparseable predicate, ...): the caller
    /// falls back to a full scan, which is always correct, just slower.
    fn try_indexed_scan(&self, table_name: &str, predicate: &str) -> Result<Option<(TableSchema, Vec<Tuple>)>> {
        let schema = self.catalog.read().get_table(table_name).cloned().ok_or_else(|| {
            DatabaseError::ExecutionError(format!("Unknown table '{table_name}'"))
        })?;
        let Some((left, op, right)) = row_codec::split_comparison(predicate) else {
            return Ok(None);
        };

        if let Some(result) = self.try_pk_index_scan(&schema, &left, &op, &right)? {
            return Ok(Some(result));
        }
        self.try_secondary_index_scan(&schema, &left, &op, &right, predicate)
    }

    /// Primary-key path: uses `MVCCTable::index_range` (the learned PGM
    /// index over row ids, which — because a row id is always its
    /// primary-key value — doubles as a PK index). No re-verification of
    /// candidates against the predicate is needed here: a row's PK can
    /// never change (`execute_update` rejects that), so a candidate row id
    /// in `[min, max]` is definitionally correct, not just probably so.
    fn try_pk_index_scan(
        &self,
        schema: &TableSchema,
        left: &str,
        op: &str,
        right: &str,
    ) -> Result<Option<(TableSchema, Vec<Tuple>)>> {
        let Some(pk_col) = schema.columns.iter().find(|c| c.primary_key) else {
            return Ok(None);
        };
        if left != pk_col.name {
            return Ok(None);
        }
        let Value::Integer(pk_value) = row_codec::parse_value(right, pk_col.data_type) else {
            // This system only ever assigns integer primary keys (see
            // execute_insert), so a non-integer literal here can't match
            // anything -- but that's a scan-and-find-nothing answer, not
            // a shape this index path is equipped to give directly.
            return Ok(None);
        };
        let pk_value = pk_value as f64;

        let (min, max) = match op {
            "=" => (pk_value, pk_value),
            ">" => (pk_value + 1.0, f64::MAX),
            ">=" => (pk_value, f64::MAX),
            "<" => (f64::MIN, pk_value - 1.0),
            "<=" => (f64::MIN, pk_value),
            _ => return Ok(None), // e.g. "!=" has no useful index range
        };

        let table_id = schema.table_id as u64;
        let candidates = match self.oltp.store.get_table(table_id) {
            Some(table) => table.index_range(min, max),
            None => return Ok(Some((schema.clone(), Vec::new()))), // registered but never written to
        };
        let candidates: std::collections::HashSet<u64> = candidates.into_iter().collect();

        let tuples = self.oltp.with_read_snapshot(|tx| {
            candidates
                .into_iter()
                .filter_map(|row_id| self.oltp.read(tx, table_id, row_id).ok().flatten())
                .filter_map(|bytes| bincode::deserialize::<Tuple>(&bytes).ok())
                .collect::<Vec<_>>()
        });

        Ok(Some((schema.clone(), tuples)))
    }

    /// Secondary-index path: uses a `SecondaryIndex` registered via
    /// `CREATE INDEX`, if one exists for `left`'s column on this table.
    /// Unlike the PK path, a candidate here genuinely can be stale — an
    /// `UPDATE` adds a new index entry for a row's new value but never
    /// removes the old one (see `secondary_index` module docs), so a
    /// candidate's *current* value might not actually match anymore.
    /// Every candidate is therefore re-checked with the exact predicate
    /// before being included, not just assumed correct because the index
    /// produced it.
    fn try_secondary_index_scan(
        &self,
        schema: &TableSchema,
        left: &str,
        op: &str,
        right: &str,
        full_predicate: &str,
    ) -> Result<Option<(TableSchema, Vec<Tuple>)>> {
        let table_id = schema.table_id as u64;
        let candidates = {
            let indexes = self.secondary_indexes.read();
            let Some(index) = indexes.get(&(table_id, left.to_string())) else {
                return Ok(None);
            };
            let Some(col) = schema.columns.iter().find(|c| c.name == left) else {
                return Ok(None);
            };
            let literal = row_codec::parse_value(right, col.data_type);
            use std::ops::Bound;
            match op {
                "=" => index.equals(&literal),
                ">" => index.range(Bound::Excluded(literal), Bound::Unbounded),
                ">=" => index.range(Bound::Included(literal), Bound::Unbounded),
                "<" => index.range(Bound::Unbounded, Bound::Excluded(literal)),
                "<=" => index.range(Bound::Unbounded, Bound::Included(literal)),
                _ => return Ok(None), // e.g. "!=" has no useful index range
            }
        };
        let candidates: std::collections::HashSet<u64> = candidates.into_iter().collect();

        let tuples = self.oltp.with_read_snapshot(|tx| {
            candidates
                .into_iter()
                .filter_map(|row_id| self.oltp.read(tx, table_id, row_id).ok().flatten())
                .filter_map(|bytes| bincode::deserialize::<Tuple>(&bytes).ok())
                .filter(|tuple| row_codec::evaluate_predicate(full_predicate, schema, tuple).unwrap_or(false))
                .collect::<Vec<_>>()
        });

        Ok(Some((schema.clone(), tuples)))
    }

    /// Build the schema for a joined row: `left`'s columns followed by
    /// `right`'s. `right`'s columns are always renamed to
    /// `"<right_prefix>.<column>"`, where `right_prefix` is the joined
    /// table's alias if the query gave it one, or its real name otherwise
    /// — matching how `sql::parser` renders a qualified reference in the
    /// `ON` condition (`"o.user_id"` for `JOIN orders o`, `"orders.user_id"`
    /// for a plain `JOIN orders`), so the condition resolves directly
    /// against these names either way.
    ///
    /// `left_prefix` is `Some(qualifier)` when `left` is a single raw table
    /// scan not yet qualified (the same alias-or-name rule), and `None`
    /// when `left` is itself already the merged output of an earlier join
    /// in a chain — its columns are already `"qualifier.column"` and must
    /// be left alone rather than re-prefixed a second time.
    fn merge_schemas(left: &TableSchema, left_prefix: Option<&str>, right: &TableSchema, right_prefix: &str) -> TableSchema {
        let mut merged = TableSchema::new(0, format!("{}_{}", left.name, right.name));
        let mut next_id = 1u32;
        for col in &left.columns {
            let name = match left_prefix {
                Some(prefix) => format!("{prefix}.{}", col.name),
                None => col.name.clone(),
            };
            merged.add_column(Column {
                id: next_id,
                name,
                data_type: col.data_type,
                nullable: col.nullable,
                primary_key: false,
            });
            next_id += 1;
        }
        for col in &right.columns {
            merged.add_column(Column {
                id: next_id,
                name: format!("{right_prefix}.{}", col.name),
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
        // Secondary indexes are only updated once the transaction actually
        // commits (below) -- updating them eagerly here would leave stale
        // entries behind for a row that turned out to be aborted.
        let mut indexed_rows: Vec<(u64, Vec<Value>)> = Vec::new();
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

            indexed_rows.push((row_id, ordered.clone()));

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
        self.oltp.commit(tx)?;

        for (row_id, values) in indexed_rows {
            self.index_row_in_secondary_indexes(table_id, row_id, &schema, &values);
        }

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
        // See execute_insert: secondary indexes are only updated once the
        // transaction actually commits, below.
        let mut indexed_rows: Vec<(u64, Vec<Value>)> = Vec::new();
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
            indexed_rows.push((row_id, tuple.values.clone()));

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

        self.oltp.commit(tx)?;

        for (row_id, values) in indexed_rows {
            self.index_row_in_secondary_indexes(table_id, row_id, &schema, &values);
        }

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

        self.oltp.commit(tx)?;
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
        self.oltp.log_schema_change(&schema)?;

        catalog.register_table(schema);
        drop(catalog);

        self.oltp.create_table(table_id as u64);
        Ok(())
    }

    /// Execute a CREATE INDEX: logs the (table, column) to the WAL (so a
    /// restart knows to rebuild it — see `open`), then builds it now by
    /// backfill-scanning the table's current rows.
    fn execute_create_index(&self, create: CreateIndexStatement) -> Result<()> {
        let table_id = {
            let catalog = self.catalog.read();
            let schema = catalog.get_table(&create.table).ok_or_else(|| {
                DatabaseError::ExecutionError(format!("Unknown table '{}'", create.table))
            })?;
            if !schema.columns.iter().any(|c| c.name == create.column) {
                return Err(DatabaseError::ExecutionError(format!(
                    "Unknown column '{}' on table '{}'",
                    create.column, create.table
                )));
            }
            schema.table_id as u64
        };

        self.oltp.log_index_change(table_id, &create.column)?;
        self.rebuild_secondary_index(table_id, &create.column)
    }

    /// (Re)build a secondary index for `table_id`'s `column` from whatever
    /// rows currently exist, replacing any previous index for that column.
    fn rebuild_secondary_index(&self, table_id: u64, column: &str) -> Result<()> {
        let schema = self
            .catalog
            .read()
            .get_table_by_id(table_id as u32)
            .cloned()
            .ok_or_else(|| DatabaseError::ExecutionError(format!("Unknown table id {table_id}")))?;
        let col_idx = schema.columns.iter().position(|c| c.name == column).ok_or_else(|| {
            DatabaseError::ExecutionError(format!("Unknown column '{column}'"))
        })?;

        let rows = self.oltp.with_read_snapshot(|tx| self.oltp.scan_table(tx, table_id));
        let mut index = SecondaryIndex::new();
        for (row_id, bytes) in rows {
            if let Ok(tuple) = bincode::deserialize::<Tuple>(&bytes) {
                if let Some(value) = tuple.values.get(col_idx) {
                    index.insert(value.clone(), row_id);
                }
            }
        }

        self.secondary_indexes.write().insert((table_id, column.to_string()), index);
        Ok(())
    }

    /// Add `row`'s value for every column with a registered secondary
    /// index. Called after a successful INSERT or UPDATE (an UPDATE only
    /// adds the *new* value's entry — see `secondary_index`'s module docs
    /// for why the old one is safely left stale rather than removed).
    fn index_row_in_secondary_indexes(&self, table_id: u64, row_id: u64, schema: &TableSchema, row: &[Value]) {
        let mut indexes = self.secondary_indexes.write();
        if indexes.is_empty() {
            return;
        }
        for (idx, col) in schema.columns.iter().enumerate() {
            if let Some(index) = indexes.get_mut(&(table_id, col.name.clone())) {
                if let Some(value) = row.get(idx) {
                    index.insert(value.clone(), row_id);
                }
            }
        }
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

    pub fn commit_transaction(&self, tx_id: TransactionId) -> Result<()> {
        self.oltp.commit(tx_id)
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
        executor.commit_transaction(tx).unwrap();

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

    #[test]
    fn test_join_with_aliases_on_both_tables() {
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
            .execute_sql("SELECT * FROM users u JOIN orders o ON u.id = o.user_id")
            .unwrap();

        assert_eq!(rows.len(), 3); // Alice has 2 orders, Bob has 1
        let alice_orders = rows.iter().filter(|r| r.contains(&"Alice".to_string())).count();
        assert_eq!(alice_orders, 2);
    }

    #[test]
    fn test_join_with_alias_only_on_joined_table() {
        let executor = QueryExecutor::new(users_and_orders_catalog());
        seed_users(&executor);
        executor
            .execute_sql("INSERT INTO orders (id, user_id, total) VALUES (100, 1, 9.5)")
            .unwrap();

        // FROM table unaliased, joined table aliased -- the condition must
        // resolve "users.id" (real name) against "o.user_id" (alias).
        let rows = executor
            .execute_sql("SELECT * FROM users JOIN orders o ON users.id = o.user_id")
            .unwrap();
        assert_eq!(rows.len(), 1);
        assert!(rows[0].contains(&"Alice".to_string()));
    }

    #[test]
    fn test_join_with_alias_and_where_on_aliased_column() {
        let executor = QueryExecutor::new(users_and_orders_catalog());
        seed_users(&executor); // Alice age 30, Bob age 15
        executor
            .execute_sql("INSERT INTO orders (id, user_id, total) VALUES (100, 1, 9.5)")
            .unwrap();
        executor
            .execute_sql("INSERT INTO orders (id, user_id, total) VALUES (101, 2, 4.0)")
            .unwrap();

        let rows = executor
            .execute_sql("SELECT * FROM users u JOIN orders o ON u.id = o.user_id WHERE u.age > 18")
            .unwrap();

        assert_eq!(rows.len(), 1);
        assert!(rows[0].contains(&"Alice".to_string()));
    }

    #[test]
    fn test_data_survives_restart() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        {
            let executor = QueryExecutor::open(&wal_path).unwrap();
            executor
                .execute_sql("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(50), age INT)")
                .unwrap();
            executor
                .execute_sql("INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)")
                .unwrap();
            executor
                .execute_sql("INSERT INTO users (id, name, age) VALUES (2, 'Bob', 15)")
                .unwrap();
            // Dropped here -- simulates the process exiting.
        }

        let reopened = QueryExecutor::open(&wal_path).unwrap();
        let rows = reopened.execute_sql("SELECT * FROM users").unwrap();
        assert_eq!(rows.len(), 2);

        // New writes after recovery must still work, with commit
        // timestamps past everything replayed.
        reopened
            .execute_sql("INSERT INTO users (id, name, age) VALUES (3, 'Carol', 40)")
            .unwrap();
        let rows = reopened.execute_sql("SELECT * FROM users").unwrap();
        assert_eq!(rows.len(), 3);
    }

    #[test]
    fn test_updates_and_deletes_survive_restart() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        {
            let executor = QueryExecutor::open(&wal_path).unwrap();
            executor
                .execute_sql("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(50), age INT)")
                .unwrap();
            executor
                .execute_sql("INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)")
                .unwrap();
            executor
                .execute_sql("INSERT INTO users (id, name, age) VALUES (2, 'Bob', 15)")
                .unwrap();
            executor
                .execute_sql("UPDATE users SET age = 31 WHERE id = 1")
                .unwrap();
            executor.execute_sql("DELETE FROM users WHERE id = 2").unwrap();
        }

        let reopened = QueryExecutor::open(&wal_path).unwrap();
        let rows = reopened.execute_sql("SELECT * FROM users").unwrap();
        assert_eq!(rows.len(), 1);
        assert!(rows[0].contains(&"31".to_string()));
    }

    #[test]
    fn test_in_memory_executor_has_no_wal_file() {
        // new() stays fully in-memory -- no path argument, nothing written
        // anywhere. This is what every other test in this file relies on.
        let executor = QueryExecutor::new(users_catalog());
        assert!(executor.execute_sql("SELECT * FROM users").is_ok());
    }

    #[test]
    fn test_indexed_point_lookup_on_primary_key() {
        let executor = QueryExecutor::new(users_catalog());
        seed_users(&executor); // id 1 = Alice, id 2 = Bob

        let rows = executor.execute_sql("SELECT * FROM users WHERE id = 1").unwrap();
        assert_eq!(rows.len(), 1);
        assert!(rows[0].contains(&"Alice".to_string()));

        // No match: empty, not an error.
        let rows = executor.execute_sql("SELECT * FROM users WHERE id = 999").unwrap();
        assert!(rows.is_empty());
    }

    #[test]
    fn test_indexed_range_lookup_on_primary_key() {
        let executor = QueryExecutor::new(users_catalog());
        executor
            .execute_sql("INSERT INTO users (id, name, age) VALUES (1, 'A', 10)")
            .unwrap();
        executor
            .execute_sql("INSERT INTO users (id, name, age) VALUES (2, 'B', 20)")
            .unwrap();
        executor
            .execute_sql("INSERT INTO users (id, name, age) VALUES (3, 'C', 30)")
            .unwrap();

        let rows = executor.execute_sql("SELECT * FROM users WHERE id > 1").unwrap();
        assert_eq!(rows.len(), 2);
        assert!(!rows.iter().any(|r| r.contains(&"A".to_string())));

        let rows = executor.execute_sql("SELECT * FROM users WHERE id <= 2").unwrap();
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_indexed_lookup_after_delete_excludes_deleted_row() {
        // The index never removes a deleted row's id -- correctness must
        // come from the point-read visibility check the index path still
        // does per candidate, not from the index itself being accurate.
        let executor = QueryExecutor::new(users_catalog());
        seed_users(&executor);
        executor.execute_sql("DELETE FROM users WHERE id = 1").unwrap();

        let rows = executor.execute_sql("SELECT * FROM users WHERE id = 1").unwrap();
        assert!(rows.is_empty());

        let rows = executor.execute_sql("SELECT * FROM users WHERE id >= 1").unwrap();
        assert_eq!(rows.len(), 1);
        assert!(rows[0].contains(&"Bob".to_string()));
    }

    #[test]
    fn test_indexed_lookup_sees_update() {
        let executor = QueryExecutor::new(users_catalog());
        seed_users(&executor);
        executor
            .execute_sql("UPDATE users SET age = 99 WHERE id = 1")
            .unwrap();

        let rows = executor.execute_sql("SELECT * FROM users WHERE id = 1").unwrap();
        assert_eq!(rows.len(), 1);
        assert!(rows[0].contains(&"99".to_string()));
    }

    #[test]
    fn test_non_primary_key_predicate_still_correct_without_index() {
        // "name = ..." isn't the PK, so this must fall back to a full
        // scan -- proving that path is still wired correctly too.
        let executor = QueryExecutor::new(users_catalog());
        seed_users(&executor);

        let rows = executor
            .execute_sql("SELECT * FROM users WHERE name = 'Bob'")
            .unwrap();
        assert_eq!(rows.len(), 1);
        assert!(rows[0].contains(&"Bob".to_string()));
    }

    fn seed_orders(executor: &QueryExecutor) {
        // user 1: two orders (10.0, 5.0); user 2: one order (20.0)
        executor
            .execute_sql("INSERT INTO orders (id, user_id, total) VALUES (100, 1, 10.0)")
            .unwrap();
        executor
            .execute_sql("INSERT INTO orders (id, user_id, total) VALUES (101, 1, 5.0)")
            .unwrap();
        executor
            .execute_sql("INSERT INTO orders (id, user_id, total) VALUES (102, 2, 20.0)")
            .unwrap();
    }

    #[test]
    fn test_group_by_count() {
        let executor = QueryExecutor::new(users_and_orders_catalog());
        seed_orders(&executor);

        let rows = executor
            .execute_sql("SELECT user_id, COUNT(*) FROM orders GROUP BY user_id")
            .unwrap();
        assert_eq!(rows.len(), 2); // two distinct user_ids

        let user1_row = rows.iter().find(|r| r[0] == "1").unwrap();
        assert_eq!(user1_row[1], "2");
        let user2_row = rows.iter().find(|r| r[0] == "2").unwrap();
        assert_eq!(user2_row[1], "1");
    }

    #[test]
    fn test_group_by_sum() {
        let executor = QueryExecutor::new(users_and_orders_catalog());
        seed_orders(&executor);

        let rows = executor
            .execute_sql("SELECT user_id, SUM(total) FROM orders GROUP BY user_id")
            .unwrap();
        assert_eq!(rows.len(), 2);

        let user1_row = rows.iter().find(|r| r[0] == "1").unwrap();
        assert_eq!(user1_row[1], "15");
        let user2_row = rows.iter().find(|r| r[0] == "2").unwrap();
        assert_eq!(user2_row[1], "20");
    }

    #[test]
    fn test_group_by_respects_where_clause() {
        let executor = QueryExecutor::new(users_and_orders_catalog());
        seed_orders(&executor);

        let rows = executor
            .execute_sql("SELECT user_id, COUNT(*) FROM orders WHERE total > 8 GROUP BY user_id")
            .unwrap();
        // Only the 10.0 and 20.0 orders qualify -- one per user.
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|r| r[1] == "1"));
    }

    #[test]
    fn test_group_by_on_table_with_no_rows_is_empty() {
        let executor = QueryExecutor::new(users_and_orders_catalog());
        let rows = executor
            .execute_sql("SELECT user_id, COUNT(*) FROM orders GROUP BY user_id")
            .unwrap();
        assert!(rows.is_empty());
    }

    #[test]
    fn test_group_by_without_aggregate_acts_like_distinct() {
        let executor = QueryExecutor::new(users_and_orders_catalog());
        seed_orders(&executor);

        let rows = executor
            .execute_sql("SELECT user_id FROM orders GROUP BY user_id")
            .unwrap();
        let mut ids: Vec<&String> = rows.iter().map(|r| &r[0]).collect();
        ids.sort();
        assert_eq!(ids, vec!["1", "2"]);
    }

    #[test]
    fn test_where_and() {
        let executor = QueryExecutor::new(users_catalog());
        seed_users(&executor); // Alice 30, Bob 15

        let rows = executor
            .execute_sql("SELECT * FROM users WHERE age > 18 AND name = 'Alice'")
            .unwrap();
        assert_eq!(rows.len(), 1);
        assert!(rows[0].contains(&"Alice".to_string()));
    }

    #[test]
    fn test_where_or() {
        let executor = QueryExecutor::new(users_catalog());
        seed_users(&executor); // Alice 30, Bob 15

        let rows = executor
            .execute_sql("SELECT * FROM users WHERE name = 'Alice' OR name = 'Bob'")
            .unwrap();
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_where_parenthesized_and_or() {
        let executor = QueryExecutor::new(users_catalog());
        seed_users(&executor); // Alice 30, Bob 15
        executor
            .execute_sql("INSERT INTO users (id, name, age) VALUES (3, 'Carol', 40)")
            .unwrap();

        // Only Alice and Carol are old enough; of those, only rows named
        // Alice or Carol qualify -- Bob is excluded by age regardless.
        let rows = executor
            .execute_sql("SELECT * FROM users WHERE age > 18 AND (name = 'Alice' OR name = 'Carol')")
            .unwrap();
        assert_eq!(rows.len(), 2);
        assert!(!rows.iter().any(|r| r.contains(&"Bob".to_string())));
    }

    #[test]
    fn test_where_and_still_uses_index_when_predicate_matches() {
        // "id = 1" alone would use the index; wrapped in a compound
        // predicate it can't (try_indexed_scan only recognizes a single
        // comparison) -- this must still be correct via the scan fallback.
        let executor = QueryExecutor::new(users_catalog());
        seed_users(&executor);

        let rows = executor
            .execute_sql("SELECT * FROM users WHERE id = 1 AND age > 18")
            .unwrap();
        assert_eq!(rows.len(), 1);
        assert!(rows[0].contains(&"Alice".to_string()));
    }

    #[test]
    fn test_delete_with_compound_where() {
        let executor = QueryExecutor::new(users_catalog());
        seed_users(&executor); // Alice 30, Bob 15
        executor
            .execute_sql("INSERT INTO users (id, name, age) VALUES (3, 'Carol', 40)")
            .unwrap();

        let result = executor
            .execute_sql("DELETE FROM users WHERE age > 18 AND name = 'Carol'")
            .unwrap();
        assert_eq!(result, vec![vec!["1".to_string()]]);

        let rows = executor.execute_sql("SELECT * FROM users").unwrap();
        assert_eq!(rows.len(), 2);
        assert!(!rows.iter().any(|r| r.contains(&"Carol".to_string())));
    }

    #[test]
    fn test_create_index_backfills_existing_rows() {
        let executor = QueryExecutor::new(users_catalog());
        seed_users(&executor); // Alice 30, Bob 15 -- inserted before the index exists

        executor.execute_sql("CREATE INDEX idx_name ON users (name)").unwrap();

        let rows = executor.execute_sql("SELECT * FROM users WHERE name = 'Bob'").unwrap();
        assert_eq!(rows.len(), 1);
        assert!(rows[0].contains(&"Bob".to_string()));
    }

    #[test]
    fn test_secondary_index_equality_and_range() {
        let executor = QueryExecutor::new(users_catalog());
        executor.execute_sql("CREATE INDEX idx_age ON users (age)").unwrap();
        seed_users(&executor); // Alice 30, Bob 15
        executor
            .execute_sql("INSERT INTO users (id, name, age) VALUES (3, 'Carol', 40)")
            .unwrap();

        let rows = executor.execute_sql("SELECT * FROM users WHERE age = 30").unwrap();
        assert_eq!(rows.len(), 1);
        assert!(rows[0].contains(&"Alice".to_string()));

        let rows = executor.execute_sql("SELECT * FROM users WHERE age > 18").unwrap();
        assert_eq!(rows.len(), 2);
        assert!(!rows.iter().any(|r| r.contains(&"Bob".to_string())));
    }

    #[test]
    fn test_secondary_index_no_matches_is_empty_not_error() {
        let executor = QueryExecutor::new(users_catalog());
        executor.execute_sql("CREATE INDEX idx_name ON users (name)").unwrap();
        seed_users(&executor);

        let rows = executor.execute_sql("SELECT * FROM users WHERE name = 'Zed'").unwrap();
        assert!(rows.is_empty());
    }

    #[test]
    fn test_secondary_index_reflects_update_to_indexed_column() {
        // The stale old-value entry the index leaves behind after an
        // UPDATE (see secondary_index module docs) must not cause the row
        // to wrongly appear under its old value, and it must be findable
        // under its new value.
        let executor = QueryExecutor::new(users_catalog());
        executor.execute_sql("CREATE INDEX idx_name ON users (name)").unwrap();
        seed_users(&executor); // Alice 30, Bob 15

        executor
            .execute_sql("UPDATE users SET name = 'Robert' WHERE name = 'Bob'")
            .unwrap();

        let rows = executor.execute_sql("SELECT * FROM users WHERE name = 'Bob'").unwrap();
        assert!(rows.is_empty(), "stale index entry must not resurrect the old value");

        let rows = executor.execute_sql("SELECT * FROM users WHERE name = 'Robert'").unwrap();
        assert_eq!(rows.len(), 1);
        assert!(rows[0].contains(&"15".to_string()));
    }

    #[test]
    fn test_secondary_index_excludes_deleted_row() {
        let executor = QueryExecutor::new(users_catalog());
        executor.execute_sql("CREATE INDEX idx_name ON users (name)").unwrap();
        seed_users(&executor); // Alice 30, Bob 15

        executor.execute_sql("DELETE FROM users WHERE name = 'Bob'").unwrap();

        let rows = executor.execute_sql("SELECT * FROM users WHERE name = 'Bob'").unwrap();
        assert!(rows.is_empty());
    }

    #[test]
    fn test_create_index_unknown_table_errors() {
        let executor = QueryExecutor::new(users_catalog());
        assert!(executor.execute_sql("CREATE INDEX idx ON ghosts (name)").is_err());
    }

    #[test]
    fn test_secondary_index_survives_restart() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        {
            let executor = QueryExecutor::open(&wal_path).unwrap();
            executor
                .execute_sql("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(50), age INT)")
                .unwrap();
            executor
                .execute_sql("INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)")
                .unwrap();
            executor.execute_sql("CREATE INDEX idx_name ON users (name)").unwrap();
            executor
                .execute_sql("INSERT INTO users (id, name, age) VALUES (2, 'Bob', 15)")
                .unwrap();
            // Dropped here -- simulates the process exiting.
        }

        let reopened = QueryExecutor::open(&wal_path).unwrap();
        // Both the pre- and post-CREATE-INDEX rows must be findable: the
        // index is rebuilt by backfill scan over the fully-recovered
        // table, not by replaying inserts against a stale index snapshot.
        let rows = reopened.execute_sql("SELECT * FROM users WHERE name = 'Alice'").unwrap();
        assert_eq!(rows.len(), 1);
        let rows = reopened.execute_sql("SELECT * FROM users WHERE name = 'Bob'").unwrap();
        assert_eq!(rows.len(), 1);
    }
}
