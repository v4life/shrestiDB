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
use crate::execution::lock_manager::LockKey;
use crate::execution::mvcc_store::WriteOp;
use crate::execution::oltp::OLTPEngine;
use crate::execution::operators::{Tuple, Value};
use crate::execution::recovery::RecoveryManager;
use crate::execution::row_codec;
use crate::execution::secondary_index::SecondaryIndex;
use crate::execution::transaction::TransactionId;
use crate::execution::wal::WriteAheadLog;
use crate::optimizer::cardinality::ColumnDistribution;
use crate::optimizer::planner::{LogicalPlanNode, PhysicalPlan, QueryPlanner};
use crate::sql::binder::Binder;
use crate::sql::parser::{
    AnalyzeStatement, CreateIndexStatement, CreateTableStatement, DeleteStatement, EquiMatch, InsertStatement,
    JoinKind, SQLParser, SQLStatement, UpdateStatement,
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
    /// Real per-column value distributions built by `ANALYZE <table>`
    /// (`execute_analyze`), keyed by `(table_name, column_name)` — see
    /// `optimizer::cardinality::ColumnDistribution`'s docs. Unlike
    /// `secondary_indexes`, this is a disposable, rebuildable cache, not
    /// data: nothing here is WAL-logged, and it goes stale after further
    /// writes until `ANALYZE` runs again, the same staleness story every
    /// production database's `ANALYZE` has.
    stats: RwLock<HashMap<(String, String), ColumnDistribution>>,
}

/// A statement parsed (and, for `SELECT`, planned) once by `QueryExecutor::prepare`,
/// ready to be run repeatedly by `execute_prepared` with different bound
/// `params` — see `prepare`'s doc comment for why this exists.
pub struct PreparedStatement {
    kind: PreparedKind,
    /// The number of bound parameters `execute_prepared` requires: the
    /// highest `$N` index referenced, or the count of `?` occurrences.
    param_count: usize,
}

enum PreparedKind {
    Select(PhysicalPlan),
    Insert(InsertStatement),
    Update(UpdateStatement),
    Delete(DeleteStatement),
}

/// A hashable, total-equality view of a `Value`, for hash-join keys.
/// `Value` can't derive `Hash`/`Eq` itself — blocked by the `Float(f64)`
/// variant — so this hashes a float's bit pattern instead of its numeric
/// value, the standard approach for hash-keyed joins. Two `NaN`s with the
/// same bit pattern hash-match under this scheme, unlike IEEE float
/// comparison — the usual convention for a *key*, not a correctness gap
/// for any join key a real query would actually use.
#[derive(PartialEq, Eq, Hash)]
enum JoinHashKey {
    Integer(i64),
    Float(u64),
    String(String),
    Boolean(bool),
    Null,
}

impl JoinHashKey {
    fn from_value(value: &Value) -> JoinHashKey {
        match value {
            Value::Integer(i) => JoinHashKey::Integer(*i),
            Value::Float(f) => JoinHashKey::Float(f.to_bits()),
            Value::String(s) => JoinHashKey::String(s.clone()),
            Value::Boolean(b) => JoinHashKey::Boolean(*b),
            Value::Null => JoinHashKey::Null,
        }
    }
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
            stats: RwLock::new(HashMap::new()),
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
            stats: RwLock::new(HashMap::new()),
        };

        for (table_id, column) in index_specs {
            executor.rebuild_secondary_index(table_id, &column)?;
        }

        Ok(executor)
    }

    /// Parse, bind, and run a SQL statement end to end. Rejects a `?`/`$N`
    /// placeholder outright rather than letting it silently reach
    /// `row_codec::parse_value` and become `Value::Null` (that function's
    /// existing soft-failure-on-unparseable-literal behavior, which would
    /// otherwise silently corrupt every value using this pattern instead of
    /// erroring) — `prepare`/`execute_prepared` is what placeholders are
    /// for; this entry point never binds parameters.
    pub fn execute_sql(&self, sql: &str) -> Result<Vec<Vec<String>>> {
        let stmt = SQLParser::parse(sql)?;
        {
            let catalog = self.catalog.read();
            Binder::new(&catalog).bind(&stmt)?;
        }
        if Self::scan_statement_placeholders(&stmt)? > 0 {
            return Err(DatabaseError::ExecutionError(
                "this statement contains a ?/$N placeholder -- use prepare()/execute_prepared() to bind parameters, execute_sql() cannot".to_string(),
            ));
        }

        match stmt {
            SQLStatement::Select(select) => {
                let plan = self.planner.plan_select(&select, &self.stats.read())?;
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
            SQLStatement::Analyze(analyze) => {
                self.execute_analyze(&analyze)?;
                Ok(Vec::new())
            }
        }
    }

    /// Parse, bind, and plan/validate `sql` once, returning a
    /// `PreparedStatement` that `execute_prepared` can run repeatedly with
    /// different bound `params` — each execution substitutes parameters
    /// into a clone of the cached, already-parsed (and, for `SELECT`,
    /// already-planned) statement rather than re-running `sqlparser`'s
    /// tokenizer and recursive-descent parser on raw SQL text every time.
    /// That re-parse is real, measured cost: it's most of why ShrestiDB's
    /// bulk-load path was ~10x slower than SQLite's in
    /// `examples/vs_sqlite.rs`, which used a real prepared statement on
    /// SQLite's side and `execute_sql` in a loop on this side.
    ///
    /// `CREATE TABLE`/`CREATE INDEX` can't be prepared (there's nothing to
    /// meaningfully re-execute with different parameters) — use
    /// `execute_sql` for those.
    pub fn prepare(&self, sql: &str) -> Result<PreparedStatement> {
        let stmt = SQLParser::parse(sql)?;
        {
            let catalog = self.catalog.read();
            Binder::new(&catalog).bind(&stmt)?;
        }
        let param_count = Self::scan_statement_placeholders(&stmt)?;

        let kind = match stmt {
            SQLStatement::Select(select) => {
                let plan = self.planner.plan_select(&select, &self.stats.read())?;
                PreparedKind::Select(plan)
            }
            SQLStatement::Insert(insert) => PreparedKind::Insert(insert),
            SQLStatement::Update(update) => PreparedKind::Update(update),
            SQLStatement::Delete(delete) => PreparedKind::Delete(delete),
            SQLStatement::CreateTable(_) | SQLStatement::CreateIndex(_) | SQLStatement::Analyze(_) => {
                return Err(DatabaseError::ExecutionError(
                    "CREATE and ANALYZE statements cannot be prepared; use execute_sql".to_string(),
                ));
            }
        };

        Ok(PreparedStatement { kind, param_count })
    }

    /// Run a statement `prepare`d earlier, substituting `params` for its
    /// `?`/`$N` placeholders. `params.len()` must be at least the number
    /// `prepare` determined the statement needs (the highest `$N` index,
    /// or the count of `?` occurrences) — extra trailing params beyond
    /// what's referenced are accepted and ignored, same as most driver
    /// APIs, rather than treated as an error.
    pub fn execute_prepared(&self, stmt: &PreparedStatement, params: &[Value]) -> Result<Vec<Vec<String>>> {
        if params.len() < stmt.param_count {
            return Err(DatabaseError::ExecutionError(format!(
                "prepared statement expects {} parameter(s), got {}",
                stmt.param_count,
                params.len()
            )));
        }

        let mut next = 0usize;
        match &stmt.kind {
            PreparedKind::Select(plan) => {
                let mut bound_plan = plan.clone();
                for node in &mut bound_plan.nodes {
                    match node {
                        LogicalPlanNode::Filter { predicate, split, .. } => {
                            // When the predicate was already split at
                            // plan time (a single comparison -- see
                            // LogicalPlanNode::Filter::split's docs),
                            // substitute each token directly instead of
                            // re-tokenizing the whole string: the shape
                            // is already known, so there's nothing left
                            // to parse, only values to fill in.
                            if let Some((left, op, right)) = split {
                                *left = row_codec::substitute_token(left, params, &mut next)?;
                                *right = row_codec::substitute_token(right, params, &mut next)?;
                                *predicate = format!("{left} {op} {right}");
                            } else {
                                *predicate = row_codec::substitute_placeholders(predicate, params, &mut next)?;
                            }
                        }
                        LogicalPlanNode::Join { condition: Some(cond), .. } => {
                            *cond = row_codec::substitute_placeholders(cond, params, &mut next)?;
                        }
                        _ => {}
                    }
                }
                self.execute(&bound_plan)
            }
            PreparedKind::Insert(insert) => {
                let mut bound = insert.clone();
                for row in &mut bound.values {
                    for value in row {
                        *value = row_codec::substitute_placeholders(value, params, &mut next)?;
                    }
                }
                let affected = self.execute_insert(bound)?;
                Ok(vec![vec![affected.to_string()]])
            }
            PreparedKind::Update(update) => {
                let mut bound = update.clone();
                for (_, raw) in &mut bound.assignments {
                    *raw = row_codec::substitute_placeholders(raw, params, &mut next)?;
                }
                if let Some(predicate) = &mut bound.where_clause {
                    *predicate = row_codec::substitute_placeholders(predicate, params, &mut next)?;
                }
                let affected = self.execute_update(bound)?;
                Ok(vec![vec![affected.to_string()]])
            }
            PreparedKind::Delete(delete) => {
                let mut bound = delete.clone();
                if let Some(predicate) = &mut bound.where_clause {
                    *predicate = row_codec::substitute_placeholders(predicate, params, &mut next)?;
                }
                let affected = self.execute_delete(bound)?;
                Ok(vec![vec![affected.to_string()]])
            }
        }
    }

    /// Parse, bind, and plan `sql` (a `SELECT`) without executing it,
    /// returning the resulting `PhysicalPlan` — the same role `EXPLAIN`
    /// plays in most SQL databases, as a Rust API rather than new SQL
    /// syntax (out of scope for now). `estimated_rows` reflects real
    /// `ANALYZE`'d statistics when available (`execute_analyze`) or the
    /// planner's fixed defaults otherwise — see `optimizer::planner`'s
    /// module doc.
    pub fn explain(&self, sql: &str) -> Result<PhysicalPlan> {
        match SQLParser::parse(sql)? {
            SQLStatement::Select(select) => self.planner.plan_select(&select, &self.stats.read()),
            _ => Err(DatabaseError::ExecutionError("explain() only supports SELECT".to_string())),
        }
    }

    /// Walk every placeholder-eligible field of `stmt` (`WHERE` clauses,
    /// `JOIN` conditions, `UPDATE` assignment right-hand sides, `INSERT`
    /// values — never column/table names, which aren't parameterizable in
    /// standard prepared-statement semantics either) and return how many
    /// bound parameters executing it would need. This exact field order is
    /// what `execute_prepared`'s substitution walk must also follow, so a
    /// positional `?`'s binding position agrees between the two — see
    /// `row_codec::substitute_placeholders`'s doc comment.
    fn scan_statement_placeholders(stmt: &SQLStatement) -> Result<usize> {
        let mut positional = 0usize;
        let mut indexed = 0usize;
        match stmt {
            SQLStatement::Select(select) => {
                // Joins before the WHERE clause -- matching the physical
                // plan's actual node order (Scan, Join*, Filter, ...),
                // since execute_prepared's substitution walk visits plan
                // nodes in that order, not source-text order. A `?`'s
                // binding position must agree between the two walks.
                for join in &select.joins {
                    if let Some(condition) = &join.condition {
                        row_codec::scan_placeholders(condition, &mut positional, &mut indexed)?;
                    }
                }
                if let Some(predicate) = &select.where_clause {
                    row_codec::scan_placeholders(predicate, &mut positional, &mut indexed)?;
                }
            }
            SQLStatement::Insert(insert) => {
                for row in &insert.values {
                    for value in row {
                        row_codec::scan_placeholders(value, &mut positional, &mut indexed)?;
                    }
                }
            }
            SQLStatement::Update(update) => {
                for (_, raw) in &update.assignments {
                    row_codec::scan_placeholders(raw, &mut positional, &mut indexed)?;
                }
                if let Some(predicate) = &update.where_clause {
                    row_codec::scan_placeholders(predicate, &mut positional, &mut indexed)?;
                }
            }
            SQLStatement::Delete(delete) => {
                if let Some(predicate) = &delete.where_clause {
                    row_codec::scan_placeholders(predicate, &mut positional, &mut indexed)?;
                }
            }
            SQLStatement::CreateTable(_) | SQLStatement::CreateIndex(_) | SQLStatement::Analyze(_) => {}
        }
        Ok(positional.max(indexed))
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
                    if let Some(LogicalPlanNode::Filter { predicate, split, .. }) = nodes.peek() {
                        // Reuse the plan-time split when available --
                        // avoids re-tokenizing `predicate` a second time
                        // here on top of whatever already produced it
                        // (see LogicalPlanNode::Filter::split's docs).
                        let indexed = match split {
                            Some((left, op, right)) => self.try_indexed_scan_split(table_name, left, op, right)?,
                            None => self.try_indexed_scan(table_name, predicate)?,
                        };
                        if let Some(indexed) = indexed {
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
                    // Compiled once, outside the loop -- see
                    // row_codec::CompiledPredicate's docs on why that
                    // matters for anything past a handful of rows.
                    //
                    // A `None` here used to fall open (`unwrap_or(true)`,
                    // every row kept) -- meant as a conservative default
                    // for a genuine structural parse failure, but
                    // `CompiledPredicate`'s tiny grammar (comparisons,
                    // `AND`/`OR`) also returns `None` for any real,
                    // well-formed SQL this engine just doesn't implement
                    // yet -- `LIKE`, `IN`, `BETWEEN`, and so on. A
                    // "conservative" default that means `WHERE name LIKE
                    // 'A%'` silently returns *every* row is the opposite
                    // of conservative. Now a hard error instead -- the
                    // same "unsupported, not silently wrong" discipline
                    // already applied to `HAVING`/`ORDER BY`/etc.
                    // elsewhere in this codebase's history.
                    let compiled = row_codec::CompiledPredicate::compile(predicate, &schema).ok_or_else(|| {
                        DatabaseError::ExecutionError(format!("Unsupported WHERE clause: '{predicate}'"))
                    })?;
                    // A per-row `None` here (distinct from the compile
                    // failure above) means this specific row's values
                    // couldn't be compared for some structural reason --
                    // excluded, not kept, matching how `NULL`/`Unknown`
                    // is already excluded (`row_codec::Tri`) rather than
                    // given the old "give up and keep it" treatment.
                    let filtered = tuples.into_iter().filter(|t| compiled.eval(t).unwrap_or(false)).collect();
                    current = Some((schema, filtered));
                }
                LogicalPlanNode::Join { right_table, right_alias, condition, equi_match, kind, .. } => {
                    let (left_schema, left_tuples) = current
                        .take()
                        .ok_or_else(|| DatabaseError::ExecutionError("JOIN with no input".to_string()))?;
                    let (right_schema, right_tuples) = self.scan_table_tuples(right_table)?;
                    let right_cols = right_schema.columns.len();
                    let left_prefix = current_qualifier.take();
                    let right_prefix = right_alias.clone().unwrap_or_else(|| right_table.clone());
                    let merged_schema =
                        Self::merge_schemas(&left_schema, left_prefix.as_deref(), &right_schema, &right_prefix);
                    let left_len = left_schema.columns.len();

                    // USING/NATURAL has no ON expression to flatten (see
                    // sql::parser::EquiMatch's docs) -- this parser has no
                    // catalog access, so resolving which columns it
                    // actually means has to happen here, the earliest
                    // point the real schemas exist. Once resolved, it
                    // becomes an ordinary condition string and flows
                    // through the exact same logic below as a real ON
                    // clause would -- including hash-join eligibility.
                    // An earlier version of this engine discarded
                    // USING/NATURAL entirely, so both silently executed
                    // as an unfiltered CROSS JOIN.
                    let condition: Option<String> = match equi_match {
                        Some(EquiMatch::Using(cols)) => Some(Self::resolve_using_condition(
                            cols,
                            &left_schema,
                            left_prefix.as_deref(),
                            &right_schema,
                            &right_prefix,
                            right_table,
                        )?),
                        Some(EquiMatch::Natural) => Self::resolve_natural_condition(
                            &left_schema,
                            left_prefix.as_deref(),
                            &right_schema,
                            &right_prefix,
                        )?,
                        None => condition.clone(),
                    };
                    let condition = &condition;

                    // Resolve the condition once: which two merged-schema
                    // column indices it compares, and -- the shape hash
                    // join needs -- whether it's a plain equality between
                    // one left-side and one right-side column. Anything
                    // else (no condition at all, a non-equality operator,
                    // both operands landing on the same side) falls back
                    // to the nested loop below, which is always correct
                    // regardless of shape.
                    let mut hash_key_indices: Option<(usize, usize)> = None; // (idx into left_tuples, idx into right_tuples)
                    if let Some(cond) = condition {
                        let (left_tok, op, right_tok) = row_codec::split_comparison(cond).ok_or_else(|| {
                            DatabaseError::ExecutionError(format!("Unsupported JOIN condition: '{cond}'"))
                        })?;
                        let col_idx = |tok: &str| merged_schema.columns.iter().position(|c| c.name == tok);
                        let (a, b) = match (col_idx(&left_tok), col_idx(&right_tok)) {
                            (Some(a), Some(b)) => (a, b),
                            _ => {
                                return Err(DatabaseError::ExecutionError(format!(
                                    "JOIN condition must compare two columns (e.g. 'a.id = b.a_id'), got: '{cond}'"
                                )));
                            }
                        };
                        if op == "=" {
                            hash_key_indices = match (a < left_len, b < left_len) {
                                (true, false) => Some((a, b - left_len)),
                                (false, true) => Some((b, a - left_len)),
                                _ => None, // both operands on the same side: not hash-joinable
                            };
                        }
                    }

                    let merged_tuples = if let Some((left_key_idx, right_key_idx)) = hash_key_indices {
                        Self::hash_join(&left_tuples, left_key_idx, &right_tuples, right_key_idx, *kind, left_len, right_cols)
                    } else {
                        // Compiled once before the nested loop -- re-parsing
                        // the condition string on every one of
                        // left_tuples.len() * right_tuples.len() pairs is
                        // exactly the cost that made this join 350-500x
                        // slower than SQLite/Postgres on the same query
                        // (see row_codec::CompiledPredicate's docs).
                        //
                        // `None` here must mean "no condition at all"
                        // (a genuine `CROSS JOIN`) -- never "a condition
                        // was given but couldn't compile," which an
                        // earlier version conflated with the real cross
                        // join case via `Option::and_then`, so `JOIN ...
                        // ON a.x LIKE b.y` silently ran as an unfiltered
                        // cross join (every pair kept) instead of
                        // erroring on the unsupported operator. `map` +
                        // `transpose` keeps that distinction: `Some(cond)`
                        // that fails to compile is now a hard error, not
                        // absorbed into the "no condition" case.
                        let compiled_condition = condition
                            .as_ref()
                            .map(|cond| {
                                row_codec::CompiledPredicate::compile(cond, &merged_schema).ok_or_else(|| {
                                    DatabaseError::ExecutionError(format!("Unsupported JOIN condition: '{cond}'"))
                                })
                            })
                            .transpose()?;

                        // A row pair's values are only cloned into a
                        // merged Tuple once it's known to match -- cloning
                        // (and, for a String column, heap-allocating)
                        // every pair regardless of match was real, wasted
                        // cost. See CompiledPredicate::eval_split.
                        //
                        // left_matched/right_matched track which rows on
                        // each side ever matched at least one row on the
                        // other -- needed for LEFT/RIGHT/FULL OUTER to
                        // emit a NULL-padded row for the ones that never
                        // did (see the loop below), on top of the ordinary
                        // matched pairs.
                        let mut left_matched = vec![false; left_tuples.len()];
                        let mut right_matched = vec![false; right_tuples.len()];
                        let mut merged_tuples = Vec::new();
                        for (li, l) in left_tuples.iter().enumerate() {
                            for (ri, r) in right_tuples.iter().enumerate() {
                                let keep = match &compiled_condition {
                                    // A per-pair `None` (the condition
                                    // compiled fine, but this specific
                                    // pair's values couldn't be compared)
                                    // excludes the pair rather than
                                    // keeping it -- same reasoning as the
                                    // Filter node's per-row default.
                                    Some(compiled) => compiled.eval_split(l, left_len, r).unwrap_or(false),
                                    None => true, // genuine CROSS JOIN -- no condition was given at all
                                };
                                if keep {
                                    left_matched[li] = true;
                                    right_matched[ri] = true;
                                    let mut values = l.values.clone();
                                    values.extend(r.values.clone());
                                    merged_tuples.push(Tuple { values });
                                }
                            }
                        }
                        Self::append_unmatched(&mut merged_tuples, *kind, &left_tuples, &left_matched, &right_tuples, &right_matched, left_len, right_cols);
                        merged_tuples
                    };
                    current = Some((merged_schema, merged_tuples));
                }
                LogicalPlanNode::Aggregate { columns, group_by, having, .. } => {
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

                    // HAVING filters *groups*, evaluated after aggregation
                    // -- unlike WHERE, it can reference an aggregate call
                    // directly (`HAVING COUNT(*) > 1`), which isn't a real
                    // schema column. `aggregate::extract_calls` rewrites
                    // every such call to a bare placeholder identifier
                    // (see its docs for why: `CompiledPredicate` splits
                    // `(`/`)` into their own tokens, so `"COUNT(*)"` isn't
                    // one atomic token to it without this), so what's left
                    // is an ordinary comparison expression `CompiledPredicate`
                    // already knows how to parse. Built once, before the
                    // per-group loop, against a synthetic schema covering
                    // every GROUP BY key plus every placeholder -- then
                    // each group's placeholder values are computed the
                    // same way `columns`' aggregate outputs are below.
                    let having_plan = having
                        .as_deref()
                        .map(|expr| {
                            let (rewritten, having_aggs) = aggregate::extract_calls(expr);

                            let mut synth_schema = TableSchema::new(schema.table_id, schema.name.clone());
                            for g in group_by {
                                if let Some(col) = schema.columns.iter().find(|c| &c.name == g) {
                                    synth_schema.add_column(col.clone());
                                }
                            }
                            for (placeholder, func, _) in &having_aggs {
                                let data_type =
                                    if *func == aggregate::AggregateFn::Count { DataType::Integer } else { DataType::Float };
                                synth_schema.add_column(Column {
                                    id: 0,
                                    name: placeholder.clone(),
                                    data_type,
                                    nullable: true,
                                    primary_key: false,
                                });
                            }

                            // A `None` here used to fall open (every
                            // group kept) -- the same class of bug fixed
                            // for `WHERE` above: a `HAVING` clause using
                            // an operator this grammar doesn't implement
                            // is a real, well-formed clause, not a
                            // structural parse failure safe to shrug off.
                            let compiled = row_codec::CompiledPredicate::compile(&rewritten, &synth_schema)
                                .ok_or_else(|| DatabaseError::ExecutionError(format!("Unsupported HAVING clause: '{expr}'")))?;
                            Ok::<_, DatabaseError>((compiled, having_aggs))
                        })
                        .transpose()?;

                    let mut output_rows = Vec::with_capacity(groups.len());
                    for (key, group_tuples) in &groups {
                        if let Some((compiled, having_aggs)) = &having_plan {
                            let mut synth_values = key.clone();
                            for (_, func, arg) in having_aggs {
                                synth_values.push(aggregate::compute_aggregate(*func, arg.as_deref(), &schema, group_tuples));
                            }
                            let synth_tuple = Tuple { values: synth_values };
                            // A per-group `None` (compiled fine, this
                            // group's synthetic values couldn't be
                            // compared) excludes the group -- same
                            // reasoning as Filter's per-row default.
                            if !compiled.eval(&synth_tuple).unwrap_or(false) {
                                continue;
                            }
                        }

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
                LogicalPlanNode::Project { columns, .. } => {
                    let (schema, tuples) = current
                        .take()
                        .ok_or_else(|| DatabaseError::ExecutionError("Project with no input".to_string()))?;

                    // Resolved once, before touching any row -- the same
                    // name must resolve for every row anyway, so there's
                    // nothing to gain from re-resolving it per tuple (and
                    // an unknown column should fail the whole statement,
                    // not just skip silently for some rows).
                    let indices: Vec<usize> = columns
                        .iter()
                        .map(|col| {
                            schema.columns.iter().position(|c| &c.name == col).ok_or_else(|| {
                                DatabaseError::ExecutionError(format!("Unknown column '{col}' in SELECT list"))
                            })
                        })
                        .collect::<Result<Vec<usize>>>()?;

                    let mut projected_schema = TableSchema::new(schema.table_id, schema.name.clone());
                    for &idx in &indices {
                        projected_schema.add_column(schema.columns[idx].clone());
                    }
                    let projected_tuples = tuples
                        .into_iter()
                        .map(|t| Tuple { values: indices.iter().map(|&i| t.values[i].clone()).collect() })
                        .collect();
                    current = Some((projected_schema, projected_tuples));
                }
                LogicalPlanNode::Sort { keys, .. } => {
                    let (schema, mut tuples) = current
                        .take()
                        .ok_or_else(|| DatabaseError::ExecutionError("Sort with no input".to_string()))?;

                    // Resolved once, like Project -- an ORDER BY column
                    // that doesn't exist fails the whole statement rather
                    // than being silently skipped mid-sort. Ordering by an
                    // aggregate expression itself (ORDER BY COUNT(*)) hits
                    // this same error: Aggregate's output schema isn't
                    // (yet) renamed to its own result columns, only a
                    // GROUP BY key's name survives unchanged from the
                    // pre-aggregation schema -- a known, documented scope
                    // limit, not a silent wrong order.
                    let resolved: Vec<(usize, bool)> = keys
                        .iter()
                        .map(|(col, ascending)| {
                            schema
                                .columns
                                .iter()
                                .position(|c| &c.name == col)
                                .map(|idx| (idx, *ascending))
                                .ok_or_else(|| DatabaseError::ExecutionError(format!("Unknown column '{col}' in ORDER BY")))
                        })
                        .collect::<Result<Vec<_>>>()?;

                    tuples.sort_by(|a, b| {
                        for &(idx, ascending) in &resolved {
                            let ord = row_codec::compare_for_sort(&a.values[idx], &b.values[idx]);
                            let ord = if ascending { ord } else { ord.reverse() };
                            if ord != std::cmp::Ordering::Equal {
                                return ord;
                            }
                        }
                        std::cmp::Ordering::Equal
                    });
                    current = Some((schema, tuples));
                }
                LogicalPlanNode::Distinct { .. } => {
                    let (schema, tuples) = current
                        .take()
                        .ok_or_else(|| DatabaseError::ExecutionError("Distinct with no input".to_string()))?;

                    // Order-preserving: keeps the first occurrence of each
                    // distinct row, so a Sort that already ran (Distinct
                    // is always placed after it -- see this node's docs)
                    // stays honored rather than scrambled by a HashSet's
                    // iteration order. Value doesn't implement Eq/Hash on
                    // its own (Float(f64) can't) -- JoinHashKey already
                    // solves exactly this for a single join column; here
                    // it's reused per-value and collected into a Vec to
                    // key a whole row instead of one column.
                    let mut seen: std::collections::HashSet<Vec<JoinHashKey>> = std::collections::HashSet::new();
                    let deduped: Vec<Tuple> = tuples
                        .into_iter()
                        .filter(|t| {
                            let key: Vec<JoinHashKey> = t.values.iter().map(JoinHashKey::from_value).collect();
                            seen.insert(key)
                        })
                        .collect();
                    current = Some((schema, deduped));
                }
                LogicalPlanNode::Limit { limit, offset, .. } => {
                    let (schema, tuples) = current
                        .take()
                        .ok_or_else(|| DatabaseError::ExecutionError("Limit with no input".to_string()))?;
                    // OFFSET first, standard SQL order -- skip() rather
                    // than an index-based split since offset may exceed
                    // the row count (an empty result, not an error).
                    let mut tuples: Vec<Tuple> = tuples.into_iter().skip(*offset).collect();
                    tuples.truncate(*limit);
                    current = Some((schema, tuples));
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
    ///
    /// Thin wrapper around `try_indexed_scan_split` for a caller that only
    /// has the flattened predicate string, not an already-split one (see
    /// that function's docs for why a caller would ever have one).
    fn try_indexed_scan(&self, table_name: &str, predicate: &str) -> Result<Option<(TableSchema, Vec<Tuple>)>> {
        let Some((left, op, right)) = row_codec::split_comparison(predicate) else {
            return Ok(None);
        };
        self.try_indexed_scan_split(table_name, &left, &op, &right)
    }

    /// Same as `try_indexed_scan`, but for a caller that already has
    /// `predicate` split into `(left, op, right)` — `execute`'s `Scan`
    /// arm, when the following `Filter` node's plan-time `split` (see
    /// `LogicalPlanNode::Filter::split`) is available, so it doesn't have
    /// to tokenize the same string `try_indexed_scan` would otherwise
    /// tokenize all over again.
    fn try_indexed_scan_split(
        &self,
        table_name: &str,
        left: &str,
        op: &str,
        right: &str,
    ) -> Result<Option<(TableSchema, Vec<Tuple>)>> {
        let schema = self.catalog.read().get_table(table_name).cloned().ok_or_else(|| {
            DatabaseError::ExecutionError(format!("Unknown table '{table_name}'"))
        })?;

        if let Some(tuples) = self.try_pk_index_scan(&schema, left, op, right)? {
            return Ok(Some((schema, tuples)));
        }
        let full_predicate = format!("{left} {op} {right}");
        let tuples = self.try_secondary_index_scan(&schema, left, op, right, &full_predicate)?;
        Ok(tuples.map(|tuples| (schema, tuples)))
    }

    /// Primary-key path: uses `MVCCTable::index_range` (the learned PGM
    /// index over row ids, which — because a row id is always its
    /// primary-key value — doubles as a PK index). No re-verification of
    /// candidates against the predicate is needed here: a row's PK can
    /// never change (`execute_update` rejects that), so a candidate row id
    /// in `[min, max]` is definitionally correct, not just probably so.
    ///
    /// Returns just the matching tuples, not `schema` alongside them —
    /// `schema` is only ever read here, never needed back: the caller
    /// (`try_indexed_scan_split`) already owns it and pairs it with
    /// whichever path actually produced tuples itself, rather than this
    /// function cloning its own copy just to hand back what the caller
    /// already had.
    fn try_pk_index_scan(&self, schema: &TableSchema, left: &str, op: &str, right: &str) -> Result<Option<Vec<Tuple>>> {
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
            None => return Ok(Some(Vec::new())), // registered but never written to
        };
        let candidates: std::collections::HashSet<u64> = candidates.into_iter().collect();

        let tuples = self.oltp.with_read_snapshot(|tx| {
            candidates
                .into_iter()
                .filter_map(|row_id| self.oltp.read(tx, table_id, row_id).ok().flatten())
                .filter_map(|bytes| bincode::deserialize::<Tuple>(&bytes).ok())
                .collect::<Vec<_>>()
        });

        Ok(Some(tuples))
    }

    /// Candidate row ids from a `SecondaryIndex` registered via `CREATE
    /// INDEX`, if one exists for `left`'s column on this table and `op`
    /// has a useful index range (`!=` doesn't). Just row ids -- no I/O,
    /// no deserialization -- shared between `try_secondary_index_scan`
    /// (`SELECT`, which fetches/deserializes/filters them into `Tuple`s)
    /// and `candidate_rows_for_write` (`UPDATE`/`DELETE`, which already
    /// does its own fetch/deserialize/re-verify on whatever candidate set
    /// it's given, full-scan or indexed).
    ///
    /// A candidate here genuinely can be stale — an `UPDATE` adds a new
    /// index entry for a row's new value but never removes the old one
    /// (see `secondary_index` module docs), so a candidate's *current*
    /// value might not actually match anymore. Every caller of this is
    /// responsible for re-checking the real predicate before treating a
    /// candidate as a match, not assuming correct because the index
    /// produced it.
    fn secondary_index_candidates(&self, table_id: u64, schema: &TableSchema, left: &str, op: &str, right: &str) -> Option<Vec<u64>> {
        let indexes = self.secondary_indexes.read();
        let index = indexes.get(&(table_id, left.to_string()))?;
        let col = schema.columns.iter().find(|c| c.name == left)?;
        let literal = row_codec::parse_value(right, col.data_type);
        use std::ops::Bound;
        Some(match op {
            "=" => index.equals(&literal),
            ">" => index.range(Bound::Excluded(literal), Bound::Unbounded),
            ">=" => index.range(Bound::Included(literal), Bound::Unbounded),
            "<" => index.range(Bound::Unbounded, Bound::Excluded(literal)),
            "<=" => index.range(Bound::Unbounded, Bound::Included(literal)),
            _ => return None, // e.g. "!=" has no useful index range
        })
    }

    /// Secondary-index path for `SELECT`: uses `secondary_index_candidates`
    /// (see its docs, including why every candidate is re-checked below
    /// rather than trusted outright), then fetches, deserializes, and
    /// filters them into real `Tuple`s.
    ///
    /// Returns just the matching tuples, not `schema` -- see
    /// `try_pk_index_scan`'s docs for why.
    fn try_secondary_index_scan(
        &self,
        schema: &TableSchema,
        left: &str,
        op: &str,
        right: &str,
        full_predicate: &str,
    ) -> Result<Option<Vec<Tuple>>> {
        let table_id = schema.table_id as u64;
        let Some(candidates) = self.secondary_index_candidates(table_id, schema, left, op, right) else {
            return Ok(None);
        };
        let candidates: std::collections::HashSet<u64> = candidates.into_iter().collect();
        let compiled = row_codec::CompiledPredicate::compile(full_predicate, schema);

        let tuples = self.oltp.with_read_snapshot(|tx| {
            candidates
                .into_iter()
                .filter_map(|row_id| self.oltp.read(tx, table_id, row_id).ok().flatten())
                .filter_map(|bytes| bincode::deserialize::<Tuple>(&bytes).ok())
                .filter(|tuple| compiled.as_ref().and_then(|c| c.eval(tuple)).unwrap_or(false))
                .collect::<Vec<_>>()
        });

        Ok(Some(tuples))
    }

    /// A column's own name, stripping any `"qualifier."` prefix a prior
    /// merge may have added (see `merge_schemas`) -- `"u.id"` and `"id"`
    /// both give `"id"`. Needed because `USING`/`NATURAL` match by plain
    /// column name, but `left_schema` may already be a merged, qualified
    /// schema from an earlier join in the chain by the time a later
    /// `USING`/`NATURAL` join runs, while `right_schema` (a fresh, single
    /// table scan) never is.
    fn base_column_name(name: &str) -> &str {
        name.rsplit('.').next().unwrap_or(name)
    }

    /// Build the merged-schema token pair (e.g. `("u.id", "o.user_id")`)
    /// for an equi-join on `left_col`/`right_col`, matching exactly how
    /// `merge_schemas` will name them -- `left_col`'s name gets
    /// `left_prefix` applied only when `left` isn't already a merged,
    /// qualified schema (same rule `merge_schemas` itself follows).
    fn equi_join_tokens(left_col: &Column, left_prefix: Option<&str>, right_col: &Column, right_prefix: &str) -> (String, String) {
        let left_token = match left_prefix {
            Some(prefix) => format!("{prefix}.{}", left_col.name),
            None => left_col.name.clone(),
        };
        (left_token, format!("{right_prefix}.{}", right_col.name))
    }

    /// Resolve `JOIN ... USING (cols)` into an ordinary `"<left> = <right>"`
    /// condition string, the same shape a real `ON` clause would produce
    /// -- see this function's call site for why that's what lets the rest
    /// of `execute`'s `Join` arm (hash-join eligibility included) run
    /// completely unchanged from here. Only a single `USING` column is
    /// supported: this engine's join execution has no composite/multi-
    /// column hash key, so more than one is a clear error rather than a
    /// silent partial match (joining on only the first column) or a
    /// silent fallback to an unfiltered cross join.
    fn resolve_using_condition(
        cols: &[String],
        left_schema: &TableSchema,
        left_prefix: Option<&str>,
        right_schema: &TableSchema,
        right_prefix: &str,
        right_table: &str,
    ) -> Result<String> {
        let [col] = cols else {
            return Err(DatabaseError::ExecutionError(format!(
                "JOIN ... USING with more than one column isn't supported yet (got {}); use an explicit ON condition instead",
                cols.len()
            )));
        };
        let left_col = left_schema
            .columns
            .iter()
            .find(|c| Self::base_column_name(&c.name) == col.as_str())
            .ok_or_else(|| DatabaseError::ExecutionError(format!("USING column '{col}' not found on the left side of the join")))?;
        let right_col = right_schema
            .columns
            .iter()
            .find(|c| &c.name == col)
            .ok_or_else(|| DatabaseError::ExecutionError(format!("USING column '{col}' not found in table '{right_table}'")))?;
        let (left_token, right_token) = Self::equi_join_tokens(left_col, left_prefix, right_col, right_prefix);
        Ok(format!("{left_token} = {right_token}"))
    }

    /// Resolve `NATURAL JOIN` into an ordinary `"<left> = <right>"`
    /// condition string -- see `resolve_using_condition`'s docs for why
    /// that shape, and the same "exactly one column" scope limit (real
    /// `NATURAL JOIN` matches on every shared column name; this engine
    /// only ever matches on one). `Ok(None)` when the two tables share no
    /// column names at all: real `NATURAL JOIN` semantics degrade to a
    /// plain, unfiltered `CROSS JOIN` in that case -- not a bug, and not
    /// this function's problem to reject.
    fn resolve_natural_condition(
        left_schema: &TableSchema,
        left_prefix: Option<&str>,
        right_schema: &TableSchema,
        right_prefix: &str,
    ) -> Result<Option<String>> {
        let shared: Vec<(&Column, &Column)> = left_schema
            .columns
            .iter()
            .filter_map(|lc| {
                let base = Self::base_column_name(&lc.name);
                right_schema.columns.iter().find(|rc| rc.name == base).map(|rc| (lc, rc))
            })
            .collect();
        match shared.as_slice() {
            [] => Ok(None),
            [(left_col, right_col)] => {
                let (left_token, right_token) = Self::equi_join_tokens(left_col, left_prefix, right_col, right_prefix);
                Ok(Some(format!("{left_token} = {right_token}")))
            }
            _ => Err(DatabaseError::ExecutionError(format!(
                "NATURAL JOIN with more than one shared column isn't supported yet ({} shared columns found); use an explicit ON condition instead",
                shared.len()
            ))),
        }
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

    /// Equi-join `left`/`right` on `left_key_idx`/`right_key_idx` (indices
    /// into each side's own tuples, not the merged schema): build a hash
    /// table on the smaller side, probe with the larger — O(n+m) instead
    /// of the nested loop's O(n*m). Only called from `execute`'s `Join`
    /// arm once the condition is known to be a plain equality between one
    /// column on each side; anything else (no condition, a non-equality
    /// operator, both operands resolving to the same side) stays on the
    /// nested-loop path, which is always correct regardless of shape.
    /// Output preserves the schema's expected `left ++ right` column
    /// order regardless of which side ends up as the hash table.
    /// `left_cols`/`right_cols` (each side's own column count, not the
    /// merged schema's) are only used for `NULL`-padding unmatched rows
    /// when `kind` calls for it — see `append_unmatched`.
    fn hash_join(
        left: &[Tuple],
        left_key_idx: usize,
        right: &[Tuple],
        right_key_idx: usize,
        kind: JoinKind,
        left_cols: usize,
        right_cols: usize,
    ) -> Vec<Tuple> {
        if left.len() <= right.len() {
            Self::hash_join_build_probe(left, left_key_idx, right, right_key_idx, false, kind, left_cols, right_cols)
        } else {
            Self::hash_join_build_probe(right, right_key_idx, left, left_key_idx, true, kind, left_cols, right_cols)
        }
    }

    /// `swapped` is true when `build`/`probe` are actually the join's
    /// right/left sides (the build side is picked by size in `hash_join`,
    /// not by which is logically "left") — controls whether a match's
    /// merged row is built `probe ++ build` or `build ++ probe` so the
    /// output always ends up `left ++ right` either way, and which of
    /// `build`/`probe`'s match-tracking corresponds to the join's
    /// logical left/right side for `append_unmatched`.
    fn hash_join_build_probe(
        build: &[Tuple],
        build_key_idx: usize,
        probe: &[Tuple],
        probe_key_idx: usize,
        swapped: bool,
        kind: JoinKind,
        left_cols: usize,
        right_cols: usize,
    ) -> Vec<Tuple> {
        // A NULL join-key value never equals anything -- not even
        // another NULL (SQL's NULL means "unknown", and two unknowns
        // aren't known to be equal). Build-side rows with a NULL key are
        // deliberately never inserted, so a NULL-keyed probe row (or
        // one that coincidentally shares its hash bucket) can never find
        // a match for either -- both stay correctly unmatched (and, for
        // an outer join, NULL-padded by append_unmatched below) rather
        // than spuriously matching every other NULL on the same side.
        let mut table: HashMap<JoinHashKey, Vec<usize>> = HashMap::new();
        for (i, tuple) in build.iter().enumerate() {
            if matches!(tuple.values[build_key_idx], Value::Null) {
                continue;
            }
            table.entry(JoinHashKey::from_value(&tuple.values[build_key_idx])).or_default().push(i);
        }

        // Indexed by position in `build`/`probe` -- a one-to-many or
        // many-to-many match still only needs each row marked once.
        let mut build_matched = vec![false; build.len()];
        let mut probe_matched = vec![false; probe.len()];
        let mut merged = Vec::new();
        for (pi, p) in probe.iter().enumerate() {
            if matches!(p.values[probe_key_idx], Value::Null) {
                continue;
            }
            let Some(indices) = table.get(&JoinHashKey::from_value(&p.values[probe_key_idx])) else {
                continue;
            };
            for &bi in indices {
                build_matched[bi] = true;
                probe_matched[pi] = true;
                let b = &build[bi];
                let mut values = if swapped { p.values.clone() } else { b.values.clone() };
                values.extend_from_slice(if swapped { &b.values } else { &p.values });
                merged.push(Tuple { values });
            }
        }

        let (left_tuples, left_matched, right_tuples, right_matched) =
            if swapped { (probe, &probe_matched, build, &build_matched) } else { (build, &build_matched, probe, &probe_matched) };
        Self::append_unmatched(&mut merged, kind, left_tuples, left_matched, right_tuples, right_matched, left_cols, right_cols);
        merged
    }

    /// Shared by both join strategies: given which rows on each side
    /// matched at least one row on the other (`left_matched`/
    /// `right_matched`, indexed the same as `left_tuples`/`right_tuples`),
    /// append the `NULL`-padded rows `kind` requires for the ones that
    /// never matched -- every left row for `Left`/`FullOuter`, every
    /// right row for `Right`/`FullOuter`. A no-op for `Inner` (an
    /// unmatched row is just dropped, the pre-existing behavior). Output
    /// rows keep the schema's `left ++ right` column order.
    fn append_unmatched(
        merged: &mut Vec<Tuple>,
        kind: JoinKind,
        left_tuples: &[Tuple],
        left_matched: &[bool],
        right_tuples: &[Tuple],
        right_matched: &[bool],
        left_cols: usize,
        right_cols: usize,
    ) {
        if matches!(kind, JoinKind::Left | JoinKind::FullOuter) {
            for (i, l) in left_tuples.iter().enumerate() {
                if !left_matched[i] {
                    let mut values = l.values.clone();
                    values.extend(std::iter::repeat(Value::Null).take(right_cols));
                    merged.push(Tuple { values });
                }
            }
        }
        if matches!(kind, JoinKind::Right | JoinKind::FullOuter) {
            for (i, r) in right_tuples.iter().enumerate() {
                if !right_matched[i] {
                    let mut values: Vec<Value> = std::iter::repeat(Value::Null).take(left_cols).collect();
                    values.extend(r.values.clone());
                    merged.push(Tuple { values });
                }
            }
        }
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

    /// Row source for `UPDATE`/`DELETE`'s initial candidate scan: the full
    /// table by default, or — when `where_clause` reduces to `<column>
    /// <op> <literal>` against an indexed column — a narrower candidate
    /// set from that index instead, skipping the full-table
    /// bincode-deserialize-and-filter pass entirely.
    ///
    /// `SELECT` already had both of these accelerations
    /// (`try_pk_index_scan`/`try_secondary_index_scan`), but
    /// `UPDATE`/`DELETE` never reused either — first found via the
    /// primary-key case, while building a workload that does many
    /// single-row `UPDATE ... WHERE id = ?` calls and it being
    /// unexpectedly slow: every one of them deserialized and
    /// predicate-checked *every* row in the table, no matter how narrow
    /// the `WHERE` clause was. The same gap existed identically for a
    /// `CREATE INDEX`-ed column, just never separately noticed until
    /// checked for directly.
    ///
    /// Two tiers, tried in order:
    /// 1. Exact `<primary key column> = <literal>` — a direct point read
    ///    by row id (a row id always *is* its table's primary-key value;
    ///    see `execution::mvcc_store::MVCCTable`'s docs). No
    ///    re-verification concern here: a row's primary key can never
    ///    change (`execute_update` rejects that), so this candidate is
    ///    definitionally correct, not just probably so — same reasoning
    ///    as `try_pk_index_scan`'s.
    /// 2. `secondary_index_candidates` for any other indexed column and
    ///    any of `=`/`>`/`>=`/`<`/`<=`. These candidates genuinely can be
    ///    stale (see that function's docs) — but that's not a new risk
    ///    introduced here: both callers already re-verify every candidate
    ///    against the real predicate (and, for `UPDATE`, re-read under an
    ///    exclusive lock) before writing anything, exactly as they did
    ///    against a full scan's candidates before either tier existed. A
    ///    wrong or stale candidate from either tier can therefore never
    ///    produce a wrong result, only wasted-or-not-wasted work — which
    ///    is also why neither tier needs to match every shape `SELECT`'s
    ///    versions handle (a compound `AND`/`OR` `WHERE`, for instance):
    ///    the full-scan fallback below stays correct regardless.
    fn candidate_rows_for_write(
        &self,
        tx: TransactionId,
        table_id: u64,
        schema: &TableSchema,
        where_clause: &Option<String>,
    ) -> Vec<(u64, Vec<u8>)> {
        if let Some(predicate) = where_clause {
            if let Some((left, op, right)) = row_codec::split_comparison(predicate) {
                if op == "=" {
                    if let Some(pk_col) = schema.columns.iter().find(|c| c.primary_key) {
                        if left == pk_col.name {
                            if let Value::Integer(pk_value) = row_codec::parse_value(&right, pk_col.data_type) {
                                return match self.oltp.read(tx, table_id, pk_value as u64) {
                                    Ok(Some(bytes)) => vec![(pk_value as u64, bytes)],
                                    _ => Vec::new(),
                                };
                            }
                        }
                    }
                }
                if let Some(candidates) = self.secondary_index_candidates(table_id, schema, &left, &op, &right) {
                    return candidates
                        .into_iter()
                        .filter_map(|row_id| match self.oltp.read(tx, table_id, row_id) {
                            Ok(Some(bytes)) => Some((row_id, bytes)),
                            _ => None,
                        })
                        .collect();
                }
            }
        }
        self.oltp.scan_table(tx, table_id)
    }

    /// Execute an UPDATE: scans the table within one transaction, applies
    /// the SET assignments to every row matching WHERE (all rows if there's
    /// no WHERE), and writes each changed row back under its existing row
    /// id — unlike INSERT, there's no id to derive here, `scan_table`
    /// already hands back each row's real id. Returns the number of rows
    /// updated.
    ///
    /// A self-referential assignment (`SET balance = balance + amount`,
    /// via `row_codec::CompiledAssignment`) is **not** safe against a
    /// concurrent UPDATE touching the same row, and switching a caller
    /// from a client-side read-then-write to this in one statement doesn't
    /// fix that by itself: `scan_table` reads every candidate row under
    /// this transaction's MVCC snapshot with no lock at all (see
    /// `OLTPEngine::read`'s docs on why reads are lock-free by design), so
    /// a value read that way is stale the instant a concurrent transaction
    /// commits a change to the same row. What actually closes the race is
    /// what happens next, for each row that snapshot-read looked like a
    /// candidate for: this acquires that row's exclusive lock *before*
    /// re-reading it, then computes and writes from that re-read value,
    /// not the snapshot one. Once the lock is held, no concurrent
    /// transaction can also be a writer on this row until this one commits
    /// or aborts, and `OLTPEngine::commit` applies a transaction's writes
    /// to the store *before* releasing its locks -- so the re-read is
    /// guaranteed to reflect every prior holder's committed write, however
    /// recent. The initial snapshot-read match is still what decides which
    /// rows to even attempt this for (locking every row in the table for a
    /// narrow `WHERE` would be its own regression); the `WHERE` clause is
    /// re-checked against the re-read value too, in case a concurrent
    /// commit changed a column it depends on in between.

    fn execute_update(&self, update: UpdateStatement) -> Result<usize> {
        let schema = self.catalog.read().get_table(&update.table).cloned().ok_or_else(|| {
            DatabaseError::ExecutionError(format!("Unknown table '{}'", update.table))
        })?;
        let pk_index = schema.columns.iter().position(|c| c.primary_key);

        // Resolve assignment targets up front so a typo, or an attempt to
        // change the primary key's value, fails before any writes happen.
        // Each assignment's RHS is compiled once here rather than
        // re-parsed per row -- see row_codec::CompiledAssignment, which
        // also fixes what was, until now, a silent-corruption bug: an
        // arithmetic RHS like "s_qty - 1" isn't a valid literal, so
        // parse_value used to fall back to Value::Null for every matched
        // row rather than erroring or actually computing the decrement.
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
            let compiled = row_codec::CompiledAssignment::compile(raw_value, &schema, schema.columns[idx].data_type)
                .ok_or_else(|| DatabaseError::ExecutionError(format!("Unsupported SET expression: '{col_name} = {raw_value}'")))?;
            assignments.push((idx, compiled));
        }

        let table_id = schema.table_id as u64;
        self.oltp.create_table(table_id);

        let tx = self.oltp.begin();
        let rows = self.candidate_rows_for_write(tx, table_id, &schema, &update.where_clause);
        // A `None` here must mean "no WHERE clause at all" (update every
        // row, correct) -- never "a WHERE clause was given but couldn't
        // compile," which used to fall open the same way Filter's WHERE
        // did (see that node's docs): `UPDATE t SET x = 1 WHERE name
        // LIKE 'A%'` would silently update *every* row, not just the
        // matching ones. `map` + `transpose` keeps the two apart -- a
        // present-but-uncompilable clause is now a hard error, raised
        // before the transaction touches anything (this runs before the
        // loop below acquires a single lock).
        let compiled_where = update
            .where_clause
            .as_ref()
            .map(|p| {
                row_codec::CompiledPredicate::compile(p, &schema)
                    .ok_or_else(|| DatabaseError::ExecutionError(format!("Unsupported WHERE clause: '{p}'")))
            })
            .transpose()
            .map_err(|e| {
                self.oltp.abort(tx);
                e
            })?;

        let mut affected = 0usize;
        // See execute_insert: secondary indexes are only updated once the
        // transaction actually commits, below.
        let mut indexed_rows: Vec<(u64, Vec<Value>)> = Vec::new();
        for (row_id, bytes) in rows {
            let Ok(tuple) = bincode::deserialize::<Tuple>(&bytes) else {
                continue; // unreadable row: skip rather than fail the whole statement
            };

            // Cheap candidacy check against this transaction's snapshot --
            // avoids locking every row in the table for a narrow UPDATE.
            // A row that passes here still gets its value re-read and
            // re-checked below, under its own lock, before anything is
            // actually written -- see this method's doc comment. A
            // per-row `None` (compiled fine, this row's values couldn't
            // be compared) excludes the row, same reasoning as Filter's
            // per-row default.
            let matches = compiled_where.as_ref().map(|c| c.eval(&tuple).unwrap_or(false)).unwrap_or(true);
            if !matches {
                continue;
            }

            if let Err(e) = self.oltp.locks.acquire_exclusive(tx, LockKey::new(table_id, row_id)) {
                self.oltp.abort(tx);
                return Err(e);
            }

            // Re-read the row's latest *committed* value now that its
            // lock is held, and re-check WHERE against that value -- see
            // this method's doc comment for why this, not the snapshot
            // read above, is what the assignments below must be computed
            // from.
            let Some(table) = self.oltp.store.get_table(table_id) else {
                continue;
            };
            let Some(latest_bytes) = table.read(row_id, self.oltp.current_commit_ts()) else {
                continue; // deleted by a concurrent transaction since our scan
            };
            let Ok(mut tuple) = bincode::deserialize::<Tuple>(&latest_bytes) else {
                continue;
            };
            let still_matches = compiled_where.as_ref().map(|c| c.eval(&tuple).unwrap_or(false)).unwrap_or(true);
            if !still_matches {
                continue;
            }

            // Evaluate every assignment against this row's pre-statement
            // snapshot before writing any of them -- "SET a = b, b = a"
            // must swap using both original values, not have the second
            // assignment see the first one's already-written result.
            let original = tuple.clone();
            for (idx, compiled) in &assignments {
                tuple.values[*idx] = compiled.eval(&original);
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
        let rows = self.candidate_rows_for_write(tx, table_id, &schema, &delete.where_clause);
        // See execute_update's identical construction for why `None`
        // must mean "no WHERE clause" and not "uncompilable WHERE
        // clause" -- the same bug here meant `DELETE FROM t WHERE name
        // LIKE 'A%'` silently deleted *every* row in the table.
        let compiled_where = delete
            .where_clause
            .as_ref()
            .map(|p| {
                row_codec::CompiledPredicate::compile(p, &schema)
                    .ok_or_else(|| DatabaseError::ExecutionError(format!("Unsupported WHERE clause: '{p}'")))
            })
            .transpose()
            .map_err(|e| {
                self.oltp.abort(tx);
                e
            })?;

        let mut affected = 0usize;
        for (row_id, bytes) in rows {
            let Ok(tuple) = bincode::deserialize::<Tuple>(&bytes) else {
                continue;
            };

            let matches = compiled_where.as_ref().map(|c| c.eval(&tuple).unwrap_or(false)).unwrap_or(true);
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

    /// The PGM segment-fitting tolerance `execute_analyze` builds each
    /// numeric column's distribution with. Looser than the tight bounds
    /// used elsewhere in this codebase for exact point-lookup indexes
    /// (`mvcc_store`'s PK index uses 8) -- a cardinality estimate doesn't
    /// need `search`'s bounded-error guarantee, only a reasonable CDF
    /// shape, so trading a little accuracy for fewer segments is a fair
    /// exchange here.
    const ANALYZE_PGM_ERROR_BOUND: usize = 16;

    /// Execute `ANALYZE <table>`: scans the table's current committed
    /// rows once and builds a real `ColumnDistribution` for every column,
    /// replacing whatever was on record for this table before. Not
    /// WAL-logged (see `stats`'s doc comment) — a restart needs a fresh
    /// `ANALYZE`, the same staleness story every production database's
    /// statistics have.
    fn execute_analyze(&self, analyze: &AnalyzeStatement) -> Result<()> {
        let schema = self.catalog.read().get_table(&analyze.table).cloned().ok_or_else(|| {
            DatabaseError::ExecutionError(format!("Unknown table '{}'", analyze.table))
        })?;
        let table_id = schema.table_id as u64;

        let rows = self.oltp.with_read_snapshot(|tx| self.oltp.scan_table(tx, table_id));
        let tuples: Vec<Tuple> =
            rows.into_iter().filter_map(|(_, bytes)| bincode::deserialize::<Tuple>(&bytes).ok()).collect();

        let mut new_stats = HashMap::new();
        for (idx, col) in schema.columns.iter().enumerate() {
            match col.data_type {
                DataType::Integer | DataType::Float | DataType::Timestamp => {
                    let values: Vec<f64> = tuples
                        .iter()
                        .filter_map(|t| match t.values.get(idx) {
                            Some(Value::Integer(i)) => Some(*i as f64),
                            Some(Value::Float(f)) => Some(*f),
                            _ => None,
                        })
                        .collect();
                    if !values.is_empty() {
                        new_stats.insert(
                            (analyze.table.clone(), col.name.clone()),
                            ColumnDistribution::build_numeric(values, Self::ANALYZE_PGM_ERROR_BOUND),
                        );
                    }
                }
                DataType::String | DataType::Boolean => {
                    let values: Vec<String> = tuples
                        .iter()
                        .filter_map(|t| match t.values.get(idx) {
                            Some(Value::String(s)) => Some(s.clone()),
                            Some(Value::Boolean(b)) => Some(b.to_string()),
                            _ => None,
                        })
                        .collect();
                    if !values.is_empty() {
                        new_stats.insert(
                            (analyze.table.clone(), col.name.clone()),
                            ColumnDistribution::build_categorical(values),
                        );
                    }
                }
            }
        }

        let mut stats = self.stats.write();
        stats.retain(|(table, _), _| table != &analyze.table);
        stats.extend(new_stats);
        Ok(())
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
    ///
    /// Each value must actually parse as a real literal (or `NULL`) of
    /// its column's declared type -- `row_codec::parse_literal_checked`,
    /// not the looser `parse_value`, which silently defaulted an
    /// unparseable value to `Value::Null` instead of erroring. Verified
    /// empirically before this was fixed: `INSERT INTO items (id, price)
    /// VALUES (1, ABS(-9.99))` silently inserted `price = NULL` rather
    /// than rejecting the unsupported expression -- the same root cause,
    /// and the same fix, as `CompiledAssignment`'s (see that type's docs).
    fn order_insert_row(insert: &InsertStatement, schema: &TableSchema, row: &[String]) -> Result<Vec<Value>> {
        let checked = |raw: &str, data_type: DataType| {
            row_codec::parse_literal_checked(raw, data_type)
                .ok_or_else(|| DatabaseError::ExecutionError(format!("Unsupported value expression: '{raw}'")))
        };

        if insert.columns.is_empty() {
            return schema.columns.iter().zip(row).map(|(c, v)| checked(v, c.data_type)).collect();
        }

        let mut ordered = vec![Value::Null; schema.columns.len()];
        for (col_name, raw) in insert.columns.iter().zip(row) {
            let idx = schema
                .columns
                .iter()
                .position(|c| &c.name == col_name)
                .ok_or_else(|| DatabaseError::ExecutionError(format!("Unknown column '{col_name}'")))?;
            ordered[idx] = checked(raw, schema.columns[idx].data_type)?;
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

    /// Confirms the real SQL UPDATE path is safe against lost updates
    /// under real concurrency -- execute_update re-reads a row's latest
    /// committed value under its exclusive lock before computing the new
    /// one, unlike OLTPEngine's raw primitives used naively (see
    /// oltp::tests::test_raw_primitives_naive_read_then_write_can_lose_updates).
    #[test]
    fn test_concurrent_sql_updates_dont_lose_writes() {
        use std::sync::Arc;
        use std::thread;

        let executor = Arc::new(QueryExecutor::new(Catalog::new()));
        executor.execute_sql("CREATE TABLE counters (id INT PRIMARY KEY, val INT)").unwrap();
        executor.execute_sql("INSERT INTO counters (id, val) VALUES (1, 0)").unwrap();

        const THREADS: usize = 8;
        const INCREMENTS_PER_THREAD: usize = 25;
        let mut handles = Vec::new();
        for _ in 0..THREADS {
            let executor = executor.clone();
            handles.push(thread::spawn(move || {
                for _ in 0..INCREMENTS_PER_THREAD {
                    executor.execute_sql("UPDATE counters SET val = val + 1 WHERE id = 1").unwrap();
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        // SELECT * so column order is unambiguous (id, val) -- SELECT val
        // alone silently returns every column, a separate real bug found
        // while debugging this (see the "always returns every column"
        // finding reported separately).
        let rows = executor.execute_sql("SELECT * FROM counters WHERE id = 1").unwrap();
        let final_val: i64 = rows[0][1].parse().unwrap();
        let expected = (THREADS * INCREMENTS_PER_THREAD) as i64;
        eprintln!("diag: real SQL UPDATE increments -- expected {expected}, got {final_val}");
        assert_eq!(final_val, expected, "the SQL UPDATE path lost concurrent updates");
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
    fn test_execute_sql_rejects_bare_placeholder() {
        // Before this check existed, "?" reaching parse_value silently
        // became Value::Null -- the exact silent-corruption shape this
        // guards against. execute_sql never binds parameters, so this
        // must error, not quietly write NULL.
        let executor = QueryExecutor::new(users_catalog());
        assert!(executor
            .execute_sql("INSERT INTO users (id, name, age) VALUES (1, ?, 30)")
            .is_err());
    }

    #[test]
    fn test_prepared_insert_executed_repeatedly_with_different_params() {
        let executor = QueryExecutor::new(users_catalog());
        let stmt = executor.prepare("INSERT INTO users (id, name, age) VALUES (?, ?, ?)").unwrap();

        executor.execute_prepared(&stmt, &[Value::Integer(1), Value::String("Alice".to_string()), Value::Integer(30)]).unwrap();
        executor.execute_prepared(&stmt, &[Value::Integer(2), Value::String("Bob".to_string()), Value::Integer(15)]).unwrap();

        let rows = executor.execute_sql("SELECT * FROM users").unwrap();
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().any(|r| r.contains(&"Alice".to_string()) && r.contains(&"30".to_string())));
        assert!(rows.iter().any(|r| r.contains(&"Bob".to_string()) && r.contains(&"15".to_string())));
    }

    #[test]
    fn test_prepared_select_with_where_placeholder() {
        let executor = QueryExecutor::new(users_catalog());
        seed_users(&executor); // Alice 30, Bob 15

        let stmt = executor.prepare("SELECT * FROM users WHERE age > ?").unwrap();

        let rows = executor.execute_prepared(&stmt, &[Value::Integer(18)]).unwrap();
        assert_eq!(rows.len(), 1);
        assert!(rows[0].contains(&"Alice".to_string()));

        // Same prepared statement, different bound value -- proves the
        // cached plan's Filter predicate is substituted fresh each call,
        // not baked in from the first execution.
        let rows = executor.execute_prepared(&stmt, &[Value::Integer(10)]).unwrap();
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_prepared_select_pk_lookup_uses_indexed_scan_and_rebinds_correctly() {
        // WHERE id = ? (id is the primary key) exercises the full
        // optimized path together: execute_prepared substitutes the
        // Filter's plan-time split (LogicalPlanNode::Filter::split)
        // directly instead of re-tokenizing, and execute's Scan arm
        // reuses that same split to reach try_indexed_scan_split without
        // tokenizing the rebuilt predicate string a second time either.
        let executor = QueryExecutor::new(users_catalog());
        seed_users(&executor); // id 1 = Alice, id 2 = Bob

        let stmt = executor.prepare("SELECT * FROM users WHERE id = ?").unwrap();

        let rows = executor.execute_prepared(&stmt, &[Value::Integer(1)]).unwrap();
        assert_eq!(rows.len(), 1);
        assert!(rows[0].contains(&"Alice".to_string()));

        // Re-execute with a different bound value -- proves the cached,
        // already-split plan node is substituted fresh each call, not
        // reused stale from the first execution.
        let rows = executor.execute_prepared(&stmt, &[Value::Integer(2)]).unwrap();
        assert_eq!(rows.len(), 1);
        assert!(rows[0].contains(&"Bob".to_string()));

        // No match: empty, not an error -- same as the non-prepared path.
        let rows = executor.execute_prepared(&stmt, &[Value::Integer(999)]).unwrap();
        assert!(rows.is_empty());
    }

    #[test]
    fn test_prepared_select_join_and_where_placeholders_bind_in_plan_order() {
        // Regression test for an ordering bug caught while implementing
        // this: the physical plan visits Join nodes before the Filter
        // node, but source SQL text has WHERE after JOIN -- scanning
        // placeholders in source order at prepare() time while
        // substituting in plan order at execute_prepared() time would
        // silently swap which bound value lands in which slot. Both must
        // walk in the same order (plan order: joins, then filter).
        let executor = QueryExecutor::new(users_catalog());
        executor.execute_sql("CREATE TABLE orders (id INT PRIMARY KEY, user_id INT, total FLOAT)").unwrap();
        seed_users(&executor); // ids 1 (Alice, age 30), 2 (Bob, age 15)
        executor.execute_sql("INSERT INTO orders (id, user_id, total) VALUES (100, 1, 9.5)").unwrap();
        executor.execute_sql("INSERT INTO orders (id, user_id, total) VALUES (101, 2, 4.0)").unwrap();

        let stmt = executor
            .prepare("SELECT * FROM users JOIN orders ON users.id = orders.user_id WHERE users.age > ?")
            .unwrap();

        let rows = executor.execute_prepared(&stmt, &[Value::Integer(18)]).unwrap();
        assert_eq!(rows.len(), 1);
        assert!(rows[0].contains(&"Alice".to_string()));
    }

    #[test]
    fn test_prepared_update_and_delete_with_placeholders() {
        let executor = QueryExecutor::new(users_catalog());
        seed_users(&executor); // Alice 30, Bob 15

        let update_stmt = executor.prepare("UPDATE users SET age = ? WHERE name = ?").unwrap();
        executor
            .execute_prepared(&update_stmt, &[Value::Integer(31), Value::String("Alice".to_string())])
            .unwrap();
        let rows = executor.execute_sql("SELECT * FROM users").unwrap();
        let alice = rows.iter().find(|r| r.contains(&"Alice".to_string())).unwrap();
        assert!(alice.contains(&"31".to_string()));

        let delete_stmt = executor.prepare("DELETE FROM users WHERE name = ?").unwrap();
        let affected = executor.execute_prepared(&delete_stmt, &[Value::String("Bob".to_string())]).unwrap();
        assert_eq!(affected, vec![vec!["1".to_string()]]);
        let rows = executor.execute_sql("SELECT * FROM users").unwrap();
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn test_prepared_statement_wrong_param_count_errors() {
        let executor = QueryExecutor::new(users_catalog());
        let stmt = executor.prepare("SELECT * FROM users WHERE age > ?").unwrap();
        assert!(executor.execute_prepared(&stmt, &[]).is_err());
    }

    #[test]
    fn test_prepare_rejects_mixed_placeholder_styles() {
        let executor = QueryExecutor::new(users_catalog());
        assert!(executor.prepare("SELECT * FROM users WHERE age > ? AND name = $1").is_err());
    }

    #[test]
    fn test_prepare_create_table_is_rejected() {
        let executor = QueryExecutor::new(Catalog::new());
        assert!(executor.prepare("CREATE TABLE t (id INT PRIMARY KEY)").is_err());
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
    fn test_update_set_with_unsupported_function_call_errors_instead_of_writing_null() {
        // Verified empirically before the fix: this silently wrote
        // Alice's age to NULL instead of erroring.
        let executor = QueryExecutor::new(users_catalog());
        seed_users(&executor); // Alice 30, Bob 15
        let err = executor.execute_sql("UPDATE users SET age = ABS(age) WHERE name = 'Alice'").unwrap_err();
        assert!(err.to_string().contains("Unsupported SET expression"));

        let rows = executor.execute_sql("SELECT * FROM users").unwrap();
        let alice = rows.iter().find(|r| r.contains(&"Alice".to_string())).unwrap();
        assert!(alice.contains(&"30".to_string()), "Alice's age must be untouched, not silently NULLed");
    }

    #[test]
    fn test_update_set_string_column_with_unsupported_function_call_errors_instead_of_writing_garbage() {
        // Verified empirically before the fix: this silently wrote the
        // literal, unparsed text "UPPER(name)" into Alice's name column.
        let executor = QueryExecutor::new(users_catalog());
        seed_users(&executor); // Alice 30, Bob 15
        let err = executor.execute_sql("UPDATE users SET name = UPPER(name) WHERE name = 'Alice'").unwrap_err();
        assert!(err.to_string().contains("Unsupported SET expression"));

        let rows = executor.execute_sql("SELECT * FROM users").unwrap();
        assert!(
            rows.iter().any(|r| r.contains(&"Alice".to_string())),
            "Alice's name must be untouched, not overwritten with the unparsed expression text"
        );
    }

    #[test]
    fn test_update_arithmetic_expression_increments_column() {
        // Before CompiledAssignment existed, "age + 1" wasn't a valid
        // literal, so this silently wrote NULL to every matched row
        // instead of erroring -- this is the case that bug was in.
        let executor = QueryExecutor::new(users_catalog());
        seed_users(&executor); // Alice 30, Bob 15

        let result = executor.execute_sql("UPDATE users SET age = age + 1 WHERE name = 'Alice'").unwrap();
        assert_eq!(result, vec![vec!["1".to_string()]]);

        let rows = executor.execute_sql("SELECT * FROM users").unwrap();
        let alice = rows.iter().find(|r| r.contains(&"Alice".to_string())).unwrap();
        assert!(alice.contains(&"31".to_string()));
        let bob = rows.iter().find(|r| r.contains(&"Bob".to_string())).unwrap();
        assert!(bob.contains(&"15".to_string())); // untouched
    }

    #[test]
    fn test_update_arithmetic_expression_without_where_updates_every_row() {
        let executor = QueryExecutor::new(users_catalog());
        seed_users(&executor); // Alice 30, Bob 15

        executor.execute_sql("UPDATE users SET age = age - 5").unwrap();

        let rows = executor.execute_sql("SELECT * FROM users").unwrap();
        let alice = rows.iter().find(|r| r.contains(&"Alice".to_string())).unwrap();
        assert!(alice.contains(&"25".to_string()));
        let bob = rows.iter().find(|r| r.contains(&"Bob".to_string())).unwrap();
        assert!(bob.contains(&"10".to_string()));
    }

    #[test]
    fn test_update_multiple_assignments_use_pre_statement_snapshot() {
        // "SET a = b, b = a" must swap using the row's original values --
        // if the second assignment saw the first one's already-written
        // result, both columns would end up equal to the original b.
        let executor = QueryExecutor::new(Catalog::new());
        executor.execute_sql("CREATE TABLE pair (id INT PRIMARY KEY, a INT, b INT)").unwrap();
        executor.execute_sql("INSERT INTO pair (id, a, b) VALUES (1, 10, 20)").unwrap();

        executor.execute_sql("UPDATE pair SET a = b, b = a WHERE id = 1").unwrap();

        let rows = executor.execute_sql("SELECT * FROM pair").unwrap();
        assert_eq!(rows, vec![vec!["1".to_string(), "20".to_string(), "10".to_string()]]);
    }

    #[test]
    fn test_concurrent_arithmetic_updates_do_not_lose_updates() {
        // The actual proof of execute_update's lock-before-read fix: before
        // it, concurrent "value = value + 1" updates from multiple threads
        // could each read the same pre-update value under their own MVCC
        // snapshot and compute their increment from it, so whichever
        // committed last would silently overwrite the other's. With the
        // fix, every UPDATE re-reads and re-locks the row before computing
        // its new value, so the final total must be *exactly* the number
        // of increments applied -- not "usually" or "close to", every
        // single run, since the fix removes the race by construction
        // rather than just narrowing the window.
        let executor = QueryExecutor::new(Catalog::new());
        executor.execute_sql("CREATE TABLE counter (id INT PRIMARY KEY, value INT)").unwrap();
        executor.execute_sql("INSERT INTO counter (id, value) VALUES (1, 0)").unwrap();

        const THREADS: usize = 8;
        const INCREMENTS_PER_THREAD: usize = 50;

        std::thread::scope(|scope| {
            for _ in 0..THREADS {
                let executor = &executor;
                scope.spawn(move || {
                    for _ in 0..INCREMENTS_PER_THREAD {
                        // A lock-timeout abort under real contention is
                        // expected and not what this test is checking --
                        // retry it, same as a real client would on a
                        // transient serialization failure.
                        loop {
                            if executor.execute_sql("UPDATE counter SET value = value + 1 WHERE id = 1").is_ok() {
                                break;
                            }
                        }
                    }
                });
            }
        });

        let rows = executor.execute_sql("SELECT * FROM counter").unwrap();
        assert_eq!(rows[0][1], (THREADS * INCREMENTS_PER_THREAD).to_string());
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
    fn test_update_by_primary_key_equality_uses_the_fast_path_correctly() {
        // WHERE id = <literal> is exactly the shape candidate_rows_for_write
        // accelerates (a direct point read instead of a full table scan) --
        // this asserts the fast path produces the same correct result as
        // the general one, not just that it's fast.
        let executor = QueryExecutor::new(users_catalog());
        seed_users(&executor);

        let result = executor.execute_sql("UPDATE users SET age = 99 WHERE id = 1").unwrap();
        assert_eq!(result, vec![vec!["1".to_string()]]);

        let rows = executor.execute_sql("SELECT * FROM users").unwrap();
        let alice = rows.iter().find(|r| r.contains(&"Alice".to_string())).unwrap();
        assert!(alice.contains(&"99".to_string()));
        let bob = rows.iter().find(|r| r.contains(&"Bob".to_string())).unwrap();
        assert!(bob.contains(&"15".to_string())); // untouched
    }

    #[test]
    fn test_update_by_primary_key_equality_on_nonexistent_id_affects_nothing() {
        let executor = QueryExecutor::new(users_catalog());
        seed_users(&executor);

        let result = executor.execute_sql("UPDATE users SET age = 99 WHERE id = 999").unwrap();
        assert_eq!(result, vec![vec!["0".to_string()]]);
    }

    #[test]
    fn test_delete_by_primary_key_equality_uses_the_fast_path_correctly() {
        let executor = QueryExecutor::new(users_catalog());
        seed_users(&executor);

        let result = executor.execute_sql("DELETE FROM users WHERE id = 1").unwrap();
        assert_eq!(result, vec![vec!["1".to_string()]]);

        let rows = executor.execute_sql("SELECT * FROM users").unwrap();
        assert_eq!(rows.len(), 1);
        assert!(rows[0].contains(&"Bob".to_string()));
    }

    #[test]
    fn test_delete_by_primary_key_equality_on_nonexistent_id_affects_nothing() {
        let executor = QueryExecutor::new(users_catalog());
        seed_users(&executor);

        let result = executor.execute_sql("DELETE FROM users WHERE id = 999").unwrap();
        assert_eq!(result, vec![vec!["0".to_string()]]);
        assert_eq!(executor.execute_sql("SELECT * FROM users").unwrap().len(), 2);
    }

    #[test]
    fn test_update_by_primary_key_is_not_a_full_table_scan() {
        // Real regression coverage for the bug candidate_rows_for_write
        // fixes: UPDATE/DELETE used to scan and bincode-deserialize every
        // row in the table on every call, regardless of how narrow WHERE
        // was. 3,000 single-row `UPDATE ... WHERE id = ?` calls against a
        // 3,000-row table is O(n) total with the point-read fast path
        // (each call touches one row) but O(n^2) with a full scan (each
        // call touches all n) -- at this n the two are seconds apart, not
        // a close call a generous bound might flake on.
        let executor = QueryExecutor::new(users_catalog());
        let insert_stmt = executor.prepare("INSERT INTO users (id, name, age) VALUES (?, ?, ?)").unwrap();
        let n = 3000;
        for id in 0..n {
            executor
                .execute_prepared(&insert_stmt, &[Value::Integer(id), Value::String(format!("user{id}")), Value::Integer(20)])
                .unwrap();
        }

        let update_stmt = executor.prepare("UPDATE users SET age = ? WHERE id = ?").unwrap();
        let start = std::time::Instant::now();
        for id in 0..n {
            executor.execute_prepared(&update_stmt, &[Value::Integer(21), Value::Integer(id)]).unwrap();
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_secs_f64() < 2.0,
            "{n} single-row UPDATEs by primary key took {elapsed:.2?} -- looks like the full-scan path again, not the point-read fast path"
        );
    }

    // ── Unsupported WHERE/JOIN-condition operators must hard-error, not fall open ──
    //
    // A whole class of severe, silent bugs: an earlier version of Filter/
    // HAVING/JOIN/UPDATE/DELETE treated "this predicate didn't compile"
    // (CompiledPredicate::compile returned None) as "keep everything" --
    // a default meant for a genuine structural parse failure, but
    // CompiledPredicate's tiny grammar (comparisons plus AND/OR) returns
    // that same None for any real, well-formed SQL it doesn't implement:
    // LIKE, IN, BETWEEN, and so on. `WHERE name LIKE 'A%'` silently
    // returned every row instead of erroring -- and for DELETE
    // specifically, silently deleted every row in the table (verified
    // empirically before this fix: DELETE FROM users WHERE name LIKE
    // 'A%' removed both seeded rows, not just the matching one).

    #[test]
    fn test_select_with_like_where_errors_instead_of_returning_everything() {
        let executor = QueryExecutor::new(users_catalog());
        seed_users(&executor);
        let err = executor.execute_sql("SELECT * FROM users WHERE name LIKE 'A%'").unwrap_err();
        assert!(err.to_string().contains("Unsupported WHERE clause"));
    }

    #[test]
    fn test_select_with_in_where_errors_instead_of_returning_everything() {
        let executor = QueryExecutor::new(users_catalog());
        seed_users(&executor);
        let err = executor.execute_sql("SELECT * FROM users WHERE id IN (1)").unwrap_err();
        assert!(err.to_string().contains("Unsupported WHERE clause"));
    }

    #[test]
    fn test_select_with_between_where_errors_instead_of_returning_everything() {
        let executor = QueryExecutor::new(users_catalog());
        seed_users(&executor);
        let err = executor.execute_sql("SELECT * FROM users WHERE age BETWEEN 20 AND 40").unwrap_err();
        assert!(err.to_string().contains("Unsupported WHERE clause"));
    }

    #[test]
    fn test_delete_with_like_where_errors_instead_of_deleting_everything() {
        let executor = QueryExecutor::new(users_catalog());
        seed_users(&executor); // Alice 30, Bob 15
        let err = executor.execute_sql("DELETE FROM users WHERE name LIKE 'A%'").unwrap_err();
        assert!(err.to_string().contains("Unsupported WHERE clause"));

        // Nothing was deleted -- the error must be raised before any row
        // is touched, not partway through.
        let rows = executor.execute_sql("SELECT * FROM users").unwrap();
        assert_eq!(rows.len(), 2, "DELETE must not have removed anything after erroring on the WHERE clause");
    }

    #[test]
    fn test_update_with_like_where_errors_instead_of_updating_everything() {
        let executor = QueryExecutor::new(users_catalog());
        seed_users(&executor); // Alice 30, Bob 15
        let err = executor.execute_sql("UPDATE users SET age = 0 WHERE name LIKE 'A%'").unwrap_err();
        assert!(err.to_string().contains("Unsupported WHERE clause"));

        let rows = executor.execute_sql("SELECT * FROM users").unwrap();
        assert!(rows.iter().any(|r| r.contains(&"30".to_string())), "Alice's age must be untouched");
        assert!(rows.iter().any(|r| r.contains(&"15".to_string())), "Bob's age must be untouched");
    }

    #[test]
    fn test_having_with_like_errors_instead_of_keeping_every_group() {
        let executor = QueryExecutor::new(users_and_orders_catalog());
        seed_orders(&executor);
        let err = executor
            .execute_sql("SELECT user_id, COUNT(*) FROM orders GROUP BY user_id HAVING user_id LIKE '1'")
            .unwrap_err();
        assert!(err.to_string().contains("Unsupported HAVING clause"));
    }

    #[test]
    fn test_join_with_like_condition_errors_instead_of_running_as_cross_join() {
        let executor = QueryExecutor::new(Catalog::new());
        executor.execute_sql("CREATE TABLE a (id INT PRIMARY KEY, tag TEXT)").unwrap();
        executor.execute_sql("CREATE TABLE b (id INT PRIMARY KEY, tag TEXT)").unwrap();
        executor.execute_sql("INSERT INTO a (id, tag) VALUES (1, 'x')").unwrap();
        executor.execute_sql("INSERT INTO b (id, tag) VALUES (1, 'x')").unwrap();

        let err = executor.execute_sql("SELECT * FROM a JOIN b ON a.tag LIKE b.tag").unwrap_err();
        assert!(err.to_string().contains("Unsupported JOIN condition"));
    }

    #[test]
    fn test_select_with_no_where_clause_still_returns_every_row() {
        // Regression guard for the fix above: `None` (genuinely no WHERE
        // clause) must still mean "keep everything" -- only a *present*,
        // uncompilable clause should error.
        let executor = QueryExecutor::new(users_catalog());
        seed_users(&executor);
        let rows = executor.execute_sql("SELECT * FROM users").unwrap();
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_delete_with_no_where_clause_still_deletes_every_row() {
        let executor = QueryExecutor::new(users_catalog());
        seed_users(&executor);
        let result = executor.execute_sql("DELETE FROM users").unwrap();
        assert_eq!(result, vec![vec!["2".to_string()]]);
    }

    #[test]
    fn test_join_with_no_condition_still_runs_as_a_real_cross_join() {
        let executor = QueryExecutor::new(Catalog::new());
        executor.execute_sql("CREATE TABLE a (id INT PRIMARY KEY)").unwrap();
        executor.execute_sql("CREATE TABLE b (id INT PRIMARY KEY)").unwrap();
        executor.execute_sql("INSERT INTO a (id) VALUES (1)").unwrap();
        executor.execute_sql("INSERT INTO a (id) VALUES (2)").unwrap();
        executor.execute_sql("INSERT INTO b (id) VALUES (10)").unwrap();

        let rows = executor.execute_sql("SELECT * FROM a JOIN b").unwrap();
        assert_eq!(rows.len(), 2, "a genuine CROSS JOIN (no ON clause at all) must still produce the full cross product");
    }

    #[test]
    fn test_update_unknown_table_errors() {
        let executor = QueryExecutor::new(users_catalog());
        assert!(executor.execute_sql("UPDATE ghosts SET x = 1").is_err());
    }

    #[test]
    fn test_insert_with_unsupported_expression_value_errors_instead_of_writing_null() {
        // Verified empirically before the fix: this silently inserted
        // price = NULL instead of erroring on the unsupported expression.
        let executor = QueryExecutor::new(Catalog::new());
        executor.execute_sql("CREATE TABLE items (id INT PRIMARY KEY, price FLOAT)").unwrap();
        let err = executor.execute_sql("INSERT INTO items (id, price) VALUES (1, ABS(-9.99))").unwrap_err();
        assert!(err.to_string().contains("Unsupported value expression"));

        let rows = executor.execute_sql("SELECT * FROM items").unwrap();
        assert!(rows.is_empty(), "the row must not have been inserted at all, not inserted with a NULL price");
    }

    #[test]
    fn test_insert_with_explicit_null_value_still_works() {
        let executor = QueryExecutor::new(Catalog::new());
        executor.execute_sql("CREATE TABLE items (id INT PRIMARY KEY, price FLOAT)").unwrap();
        executor.execute_sql("INSERT INTO items (id, price) VALUES (1, NULL)").unwrap();
        let rows = executor.execute_sql("SELECT * FROM items").unwrap();
        assert_eq!(rows, vec![vec!["1".to_string(), "NULL".to_string()]]);
    }

    #[test]
    fn test_insert_with_unquoted_text_for_string_column_errors_instead_of_storing_it() {
        let executor = QueryExecutor::new(Catalog::new());
        executor.execute_sql("CREATE TABLE items (id INT PRIMARY KEY, label VARCHAR(50))").unwrap();
        let err = executor.execute_sql("INSERT INTO items (id, label) VALUES (1, UPPER(x))").unwrap_err();
        assert!(err.to_string().contains("Unsupported value expression"));
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

    /// `users_and_orders_catalog` plus a `payments` table referencing
    /// `orders`, for a genuine 3-way `JOIN ... JOIN ...` chain -- no test
    /// exercised one before this, despite the alias/qualifier-tracking
    /// fix (`current_qualifier` in `execute`) that a chained join relies
    /// on having landed separately, and despite join reordering
    /// (`optimizer::join_reorder`) now depending on a 3-way chain
    /// actually working correctly.
    fn users_orders_and_payments_catalog() -> Catalog {
        let mut catalog = users_and_orders_catalog();
        let mut payments = TableSchema::new(3, "payments".to_string());
        payments.add_column(Column {
            id: 1,
            name: "id".to_string(),
            data_type: DataType::Integer,
            nullable: false,
            primary_key: true,
        });
        payments.add_column(Column {
            id: 2,
            name: "order_id".to_string(),
            data_type: DataType::Integer,
            nullable: false,
            primary_key: false,
        });
        payments.add_column(Column {
            id: 3,
            name: "method".to_string(),
            data_type: DataType::String,
            nullable: false,
            primary_key: false,
        });
        catalog.register_table(payments);
        catalog
    }

    #[test]
    fn test_three_way_join_matches_correct_rows() {
        let executor = QueryExecutor::new(users_orders_and_payments_catalog());
        seed_users(&executor); // ids 1 (Alice), 2 (Bob)
        executor.execute_sql("INSERT INTO orders (id, user_id, total) VALUES (100, 1, 9.5)").unwrap();
        executor.execute_sql("INSERT INTO orders (id, user_id, total) VALUES (101, 2, 4.0)").unwrap();
        executor.execute_sql("INSERT INTO payments (id, order_id, method) VALUES (1000, 100, 'card')").unwrap();
        executor.execute_sql("INSERT INTO payments (id, order_id, method) VALUES (1001, 100, 'cash')").unwrap();
        executor.execute_sql("INSERT INTO payments (id, order_id, method) VALUES (1002, 101, 'card')").unwrap();

        let rows = executor
            .execute_sql(
                "SELECT * FROM users u JOIN orders o ON u.id = o.user_id JOIN payments p ON o.id = p.order_id",
            )
            .unwrap();

        // Alice's order 100 has 2 payments, Bob's order 101 has 1.
        assert_eq!(rows.len(), 3);
        let alice_payments = rows.iter().filter(|r| r.contains(&"Alice".to_string())).count();
        assert_eq!(alice_payments, 2);
        let bob_payments = rows.iter().filter(|r| r.contains(&"Bob".to_string())).count();
        assert_eq!(bob_payments, 1);
    }

    #[test]
    fn test_select_explicit_column_list_projects_only_those_columns() {
        // Previously select.columns was only ever consulted to detect an
        // aggregate query and otherwise discarded, so a plain non-aggregate
        // SELECT with an explicit column list silently behaved exactly
        // like SELECT * -- returning id/name/age instead of just name.
        let executor = QueryExecutor::new(users_catalog());
        seed_users(&executor); // Alice (30), Bob (15)

        let rows = executor.execute_sql("SELECT name FROM users WHERE age > 18").unwrap();
        assert_eq!(rows, vec![vec!["Alice".to_string()]]);
    }

    #[test]
    fn test_select_column_list_respects_requested_order() {
        let executor = QueryExecutor::new(users_catalog());
        seed_users(&executor); // Alice, id=1, age=30

        // Table column order is id, name, age -- SELECT lists them
        // reversed, and the output must follow the SELECT list, not the
        // table's own column order.
        let rows = executor.execute_sql("SELECT age, name, id FROM users WHERE id = 1").unwrap();
        assert_eq!(rows, vec![vec!["30".to_string(), "Alice".to_string(), "1".to_string()]]);
    }

    #[test]
    fn test_select_column_list_on_indexed_pk_lookup_still_projects() {
        // WHERE id = <literal> is fused into an indexed scan (try_pk_index_scan)
        // by execute()'s Scan arm -- confirms the Project node downstream
        // of that fast path still correctly narrows the columns, not just
        // the full-scan path.
        let executor = QueryExecutor::new(users_catalog());
        seed_users(&executor);

        let rows = executor.execute_sql("SELECT name FROM users WHERE id = 1").unwrap();
        assert_eq!(rows, vec![vec!["Alice".to_string()]]);
    }

    #[test]
    fn test_select_qualified_columns_after_join_projects_correctly() {
        let executor = QueryExecutor::new(users_and_orders_catalog());
        seed_users(&executor); // ids 1 (Alice), 2 (Bob)
        executor.execute_sql("INSERT INTO orders (id, user_id, total) VALUES (100, 1, 9.5)").unwrap();

        let rows = executor
            .execute_sql("SELECT users.name, orders.total FROM users JOIN orders ON users.id = orders.user_id")
            .unwrap();
        assert_eq!(rows, vec![vec!["Alice".to_string(), "9.5".to_string()]]);
    }

    #[test]
    fn test_order_by_sorts_ascending_by_default() {
        // select.order_by used to be parsed and then never consulted --
        // rows came back in arbitrary storage order regardless.
        let executor = QueryExecutor::new(users_catalog());
        seed_users(&executor); // Alice (30), Bob (15)

        let rows = executor.execute_sql("SELECT name FROM users ORDER BY age").unwrap();
        assert_eq!(rows, vec![vec!["Bob".to_string()], vec!["Alice".to_string()]]);
    }

    #[test]
    fn test_order_by_desc_reverses_order() {
        let executor = QueryExecutor::new(users_catalog());
        seed_users(&executor);

        let rows = executor.execute_sql("SELECT name FROM users ORDER BY age DESC").unwrap();
        assert_eq!(rows, vec![vec!["Alice".to_string()], vec!["Bob".to_string()]]);
    }

    #[test]
    fn test_order_by_column_not_in_select_list_still_works() {
        // Real SQL allows sorting by a column that isn't projected --
        // Sort must run before Project drops it, not after.
        let executor = QueryExecutor::new(users_catalog());
        seed_users(&executor);

        let rows = executor.execute_sql("SELECT name FROM users ORDER BY age DESC").unwrap();
        assert_eq!(rows, vec![vec!["Alice".to_string()], vec!["Bob".to_string()]]);
    }

    #[test]
    fn test_limit_truncates_result() {
        let executor = QueryExecutor::new(users_catalog());
        seed_users(&executor);

        let rows = executor.execute_sql("SELECT * FROM users LIMIT 1").unwrap();
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn test_order_by_then_limit_returns_the_correct_top_n() {
        let executor = QueryExecutor::new(users_catalog());
        seed_users(&executor); // Alice (30), Bob (15)
        executor.execute_sql("INSERT INTO users (id, name, age) VALUES (3, 'Carol', 45)").unwrap();

        let rows = executor.execute_sql("SELECT name FROM users ORDER BY age DESC LIMIT 2").unwrap();
        assert_eq!(rows, vec![vec!["Carol".to_string()], vec!["Alice".to_string()]]);
    }

    #[test]
    fn test_order_by_multi_column_breaks_ties_with_second_key() {
        let executor = QueryExecutor::new(Catalog::new());
        executor.execute_sql("CREATE TABLE t (id INT PRIMARY KEY, grp INT, val INT)").unwrap();
        executor.execute_sql("INSERT INTO t (id, grp, val) VALUES (1, 1, 20)").unwrap();
        executor.execute_sql("INSERT INTO t (id, grp, val) VALUES (2, 1, 10)").unwrap();
        executor.execute_sql("INSERT INTO t (id, grp, val) VALUES (3, 2, 5)").unwrap();

        let rows = executor.execute_sql("SELECT id FROM t ORDER BY grp, val").unwrap();
        assert_eq!(rows, vec![vec!["2".to_string()], vec!["1".to_string()], vec!["3".to_string()]]);
    }

    #[test]
    fn test_order_by_sorts_null_first() {
        let executor = QueryExecutor::new(Catalog::new());
        executor.execute_sql("CREATE TABLE t (id INT PRIMARY KEY, val INT)").unwrap();
        executor.execute_sql("INSERT INTO t (id, val) VALUES (1, 5)").unwrap();
        executor.execute_sql("INSERT INTO t (id, val) VALUES (2, NULL)").unwrap();

        let rows = executor.execute_sql("SELECT id FROM t ORDER BY val").unwrap();
        assert_eq!(rows, vec![vec!["2".to_string()], vec!["1".to_string()]]);
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
    fn test_left_join_hash_path_keeps_unmatched_left_row_with_nulls() {
        // a.id=2 has no matching b row -- LEFT JOIN must still emit it
        // once, with NULLs for b's columns, not drop it the way INNER
        // JOIN (the only behavior this engine had before) would.
        let executor = QueryExecutor::new(Catalog::new());
        executor.execute_sql("CREATE TABLE a (id INT PRIMARY KEY, tag INT)").unwrap();
        executor.execute_sql("CREATE TABLE b (id INT PRIMARY KEY, tag INT)").unwrap();
        executor.execute_sql("INSERT INTO a (id, tag) VALUES (1, 1)").unwrap();
        executor.execute_sql("INSERT INTO a (id, tag) VALUES (2, 99)").unwrap();
        executor.execute_sql("INSERT INTO b (id, tag) VALUES (10, 1)").unwrap();

        let rows = executor.execute_sql("SELECT * FROM a LEFT JOIN b ON a.tag = b.tag").unwrap();
        assert_eq!(rows.len(), 2);
        let matched = rows.iter().find(|r| r[0] == "1").unwrap();
        assert_eq!(matched, &vec!["1".to_string(), "1".to_string(), "10".to_string(), "1".to_string()]);
        let unmatched = rows.iter().find(|r| r[0] == "2").unwrap();
        assert_eq!(unmatched, &vec!["2".to_string(), "99".to_string(), "NULL".to_string(), "NULL".to_string()]);
    }

    #[test]
    fn test_right_join_hash_path_keeps_unmatched_right_row_with_nulls() {
        // Mirror of the LEFT JOIN case: b.id=20 has no matching a row.
        let executor = QueryExecutor::new(Catalog::new());
        executor.execute_sql("CREATE TABLE a (id INT PRIMARY KEY, tag INT)").unwrap();
        executor.execute_sql("CREATE TABLE b (id INT PRIMARY KEY, tag INT)").unwrap();
        executor.execute_sql("INSERT INTO a (id, tag) VALUES (1, 1)").unwrap();
        executor.execute_sql("INSERT INTO b (id, tag) VALUES (10, 1)").unwrap();
        executor.execute_sql("INSERT INTO b (id, tag) VALUES (20, 99)").unwrap();

        let rows = executor.execute_sql("SELECT * FROM a RIGHT JOIN b ON a.tag = b.tag").unwrap();
        assert_eq!(rows.len(), 2);
        let matched = rows.iter().find(|r| r[2] == "10").unwrap();
        assert_eq!(matched, &vec!["1".to_string(), "1".to_string(), "10".to_string(), "1".to_string()]);
        let unmatched = rows.iter().find(|r| r[2] == "20").unwrap();
        assert_eq!(unmatched, &vec!["NULL".to_string(), "NULL".to_string(), "20".to_string(), "99".to_string()]);
    }

    #[test]
    fn test_full_outer_join_keeps_unmatched_rows_on_both_sides() {
        let executor = QueryExecutor::new(Catalog::new());
        executor.execute_sql("CREATE TABLE a (id INT PRIMARY KEY, tag INT)").unwrap();
        executor.execute_sql("CREATE TABLE b (id INT PRIMARY KEY, tag INT)").unwrap();
        executor.execute_sql("INSERT INTO a (id, tag) VALUES (1, 1)").unwrap();
        executor.execute_sql("INSERT INTO a (id, tag) VALUES (2, 99)").unwrap(); // unmatched left
        executor.execute_sql("INSERT INTO b (id, tag) VALUES (10, 1)").unwrap();
        executor.execute_sql("INSERT INTO b (id, tag) VALUES (20, 88)").unwrap(); // unmatched right

        let rows = executor.execute_sql("SELECT * FROM a FULL OUTER JOIN b ON a.tag = b.tag").unwrap();
        assert_eq!(rows.len(), 3); // 1 matched pair + 1 unmatched left + 1 unmatched right
        assert!(rows.iter().any(|r| r == &vec!["1".to_string(), "1".to_string(), "10".to_string(), "1".to_string()]));
        assert!(rows.iter().any(|r| r == &vec!["2".to_string(), "99".to_string(), "NULL".to_string(), "NULL".to_string()]));
        assert!(rows.iter().any(|r| r == &vec!["NULL".to_string(), "NULL".to_string(), "20".to_string(), "88".to_string()]));
    }

    #[test]
    fn test_left_join_nested_loop_path_keeps_unmatched_left_row_with_nulls() {
        // A non-equality condition ("<") isn't hash-joinable, so this
        // exercises the nested-loop path's own unmatched-row tracking,
        // not hash_join_build_probe's -- both paths need this to work.
        let executor = QueryExecutor::new(Catalog::new());
        executor.execute_sql("CREATE TABLE a (id INT PRIMARY KEY, val INT)").unwrap();
        executor.execute_sql("CREATE TABLE b (id INT PRIMARY KEY, val INT)").unwrap();
        executor.execute_sql("INSERT INTO a (id, val) VALUES (1, 1)").unwrap();
        executor.execute_sql("INSERT INTO a (id, val) VALUES (2, 100)").unwrap(); // matches nothing (no b.val > 100)
        executor.execute_sql("INSERT INTO b (id, val) VALUES (10, 5)").unwrap();

        let rows = executor.execute_sql("SELECT * FROM a LEFT JOIN b ON a.val < b.val").unwrap();
        assert_eq!(rows.len(), 2);
        let matched = rows.iter().find(|r| r[0] == "1").unwrap();
        assert_eq!(matched, &vec!["1".to_string(), "1".to_string(), "10".to_string(), "5".to_string()]);
        let unmatched = rows.iter().find(|r| r[0] == "2").unwrap();
        assert_eq!(unmatched, &vec!["2".to_string(), "100".to_string(), "NULL".to_string(), "NULL".to_string()]);
    }

    #[test]
    fn test_inner_join_default_still_drops_unmatched_rows() {
        // Same data shape as the LEFT JOIN hash-path test above, but
        // plain JOIN -- confirms adding outer-join support didn't change
        // INNER JOIN's pre-existing (and still correct) behavior.
        let executor = QueryExecutor::new(Catalog::new());
        executor.execute_sql("CREATE TABLE a (id INT PRIMARY KEY, tag INT)").unwrap();
        executor.execute_sql("CREATE TABLE b (id INT PRIMARY KEY, tag INT)").unwrap();
        executor.execute_sql("INSERT INTO a (id, tag) VALUES (1, 1)").unwrap();
        executor.execute_sql("INSERT INTO a (id, tag) VALUES (2, 99)").unwrap();
        executor.execute_sql("INSERT INTO b (id, tag) VALUES (10, 1)").unwrap();

        let rows = executor.execute_sql("SELECT * FROM a JOIN b ON a.tag = b.tag").unwrap();
        assert_eq!(rows.len(), 1);
        assert!(!rows.iter().any(|r| r.contains(&"NULL".to_string())));
    }

    #[test]
    fn test_using_join_matches_correctly_not_a_cross_product() {
        // An earlier version of this engine discarded USING entirely,
        // silently executing it as an unfiltered CROSS JOIN -- 2 a-rows x
        // 3 b-rows = 6 rows, instead of the 2 rows a real equi-join on id
        // produces here.
        let executor = QueryExecutor::new(Catalog::new());
        executor.execute_sql("CREATE TABLE a (id INT PRIMARY KEY, val INT)").unwrap();
        executor.execute_sql("CREATE TABLE b (id INT PRIMARY KEY, val INT)").unwrap();
        executor.execute_sql("INSERT INTO a (id, val) VALUES (1, 10)").unwrap();
        executor.execute_sql("INSERT INTO a (id, val) VALUES (2, 20)").unwrap();
        executor.execute_sql("INSERT INTO b (id, val) VALUES (1, 100)").unwrap();
        executor.execute_sql("INSERT INTO b (id, val) VALUES (2, 200)").unwrap();
        executor.execute_sql("INSERT INTO b (id, val) VALUES (3, 300)").unwrap();

        let rows = executor.execute_sql("SELECT * FROM a JOIN b USING (id)").unwrap();
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().any(|r| r == &vec!["1".to_string(), "10".to_string(), "1".to_string(), "100".to_string()]));
        assert!(rows.iter().any(|r| r == &vec!["2".to_string(), "20".to_string(), "2".to_string(), "200".to_string()]));
    }

    #[test]
    fn test_natural_join_matches_on_shared_column() {
        // Same silent-CROSS-JOIN bug as USING, for NATURAL JOIN's
        // automatic shared-column matching (here: `id`, since `val`
        // differs between the two rows -- a real NATURAL JOIN wouldn't
        // match on `id` alone if `val` is also shared and different, but
        // this table shape only shares `id`).
        let executor = QueryExecutor::new(Catalog::new());
        executor.execute_sql("CREATE TABLE a (id INT PRIMARY KEY, a_only INT)").unwrap();
        executor.execute_sql("CREATE TABLE b (id INT PRIMARY KEY, b_only INT)").unwrap();
        executor.execute_sql("INSERT INTO a (id, a_only) VALUES (1, 10)").unwrap();
        executor.execute_sql("INSERT INTO a (id, a_only) VALUES (2, 20)").unwrap();
        executor.execute_sql("INSERT INTO b (id, b_only) VALUES (1, 100)").unwrap();
        executor.execute_sql("INSERT INTO b (id, b_only) VALUES (3, 300)").unwrap();

        let rows = executor.execute_sql("SELECT * FROM a NATURAL JOIN b").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0], vec!["1".to_string(), "10".to_string(), "1".to_string(), "100".to_string()]);
    }

    #[test]
    fn test_natural_join_with_no_shared_columns_is_a_real_cross_join() {
        // Not a bug: standard NATURAL JOIN semantics degrade to a plain
        // CROSS JOIN when the two tables share no column names at all.
        let executor = QueryExecutor::new(Catalog::new());
        executor.execute_sql("CREATE TABLE a (a_id INT PRIMARY KEY)").unwrap();
        executor.execute_sql("CREATE TABLE b (b_id INT PRIMARY KEY)").unwrap();
        executor.execute_sql("INSERT INTO a (a_id) VALUES (1)").unwrap();
        executor.execute_sql("INSERT INTO a (a_id) VALUES (2)").unwrap();
        executor.execute_sql("INSERT INTO b (b_id) VALUES (10)").unwrap();

        let rows = executor.execute_sql("SELECT * FROM a NATURAL JOIN b").unwrap();
        assert_eq!(rows.len(), 2); // 2 x 1, every combination
    }

    #[test]
    fn test_using_join_with_multiple_columns_errors_loudly() {
        // Composite-key USING isn't supported (this engine's join
        // execution has no multi-column hash key) -- must be a clear
        // error, not a silent partial match on just the first column.
        let executor = QueryExecutor::new(Catalog::new());
        executor.execute_sql("CREATE TABLE a (id INT PRIMARY KEY, tag INT)").unwrap();
        executor.execute_sql("CREATE TABLE b (id INT PRIMARY KEY, tag INT)").unwrap();
        let err = executor.execute_sql("SELECT * FROM a JOIN b USING (id, tag)").unwrap_err();
        assert!(err.to_string().contains("more than one column"));
    }

    #[test]
    fn test_natural_join_with_multiple_shared_columns_errors_loudly() {
        let executor = QueryExecutor::new(Catalog::new());
        executor.execute_sql("CREATE TABLE a (id INT PRIMARY KEY, tag INT)").unwrap();
        executor.execute_sql("CREATE TABLE b (id INT PRIMARY KEY, tag INT)").unwrap();
        let err = executor.execute_sql("SELECT * FROM a NATURAL JOIN b").unwrap_err();
        assert!(err.to_string().contains("more than one shared column"));
    }

    #[test]
    fn test_using_join_as_second_join_in_chain_matches_qualified_left_column() {
        // By the time this second USING join runs, `left_schema` is
        // already the merged, qualified output of the first join (its
        // `id` column is really named "a.id") -- exercises
        // base_column_name's qualifier-stripping, not just the simple
        // unqualified-first-join case the other tests cover.
        let executor = QueryExecutor::new(Catalog::new());
        executor.execute_sql("CREATE TABLE a (id INT PRIMARY KEY, val INT)").unwrap();
        executor.execute_sql("CREATE TABLE b (id INT PRIMARY KEY, val INT)").unwrap();
        executor.execute_sql("CREATE TABLE c (id INT PRIMARY KEY, val INT)").unwrap();
        executor.execute_sql("INSERT INTO a (id, val) VALUES (1, 10)").unwrap();
        executor.execute_sql("INSERT INTO b (id, val) VALUES (1, 100)").unwrap();
        executor.execute_sql("INSERT INTO c (id, val) VALUES (1, 1000)").unwrap();
        executor.execute_sql("INSERT INTO c (id, val) VALUES (2, 2000)").unwrap();

        let rows = executor.execute_sql("SELECT * FROM a JOIN b ON a.id = b.id JOIN c USING (id)").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(
            rows[0],
            vec!["1".to_string(), "10".to_string(), "1".to_string(), "100".to_string(), "1".to_string(), "1000".to_string()]
        );
    }

    #[test]
    fn test_where_excludes_null_comparisons_instead_of_including_them() {
        // SQL: `NULL > 100` is UNKNOWN, and WHERE excludes UNKNOWN same
        // as FALSE. An earlier version's Filter arm treated the
        // "couldn't evaluate" fallback (.unwrap_or(true)) as covering
        // this too, so a NULL-valued row was *kept* instead of excluded.
        let executor = QueryExecutor::new(Catalog::new());
        executor.execute_sql("CREATE TABLE t (id INT PRIMARY KEY, fk INT)").unwrap();
        executor.execute_sql("INSERT INTO t (id, fk) VALUES (1, NULL)").unwrap();
        executor.execute_sql("INSERT INTO t (id, fk) VALUES (2, 5)").unwrap();

        let rows = executor.execute_sql("SELECT * FROM t WHERE fk > 100").unwrap();
        assert!(rows.is_empty());
    }

    #[test]
    fn test_nested_loop_join_null_key_matches_nothing_not_everything() {
        // Non-equality condition forces the nested-loop path (eval_split,
        // not the hash-join's separate JoinHashKey mechanism). a.fk is
        // NULL, so `a.fk < c.fk` is UNKNOWN for every c row -- must match
        // none of them, not (the old, broken behavior) every one of them.
        let executor = QueryExecutor::new(Catalog::new());
        executor.execute_sql("CREATE TABLE a (id INT PRIMARY KEY, fk INT)").unwrap();
        executor.execute_sql("CREATE TABLE c (id INT PRIMARY KEY, fk INT)").unwrap();
        executor.execute_sql("INSERT INTO a (id, fk) VALUES (1, NULL)").unwrap();
        executor.execute_sql("INSERT INTO c (id, fk) VALUES (50, 5)").unwrap();
        executor.execute_sql("INSERT INTO c (id, fk) VALUES (51, 999)").unwrap();

        let rows = executor.execute_sql("SELECT * FROM a JOIN c ON a.fk < c.fk").unwrap();
        assert!(rows.is_empty());
    }

    #[test]
    fn test_hash_join_null_key_never_matches_another_null() {
        // The hash-join path doesn't go through row_codec::compare at
        // all -- it hashes join-key values directly (JoinHashKey), which
        // treated Value::Null as an ordinary, matchable key equal to
        // itself. Two independent NULLs on the join column spuriously
        // matched each other, even though real SQL NULL never equals
        // NULL. Reachable in practice via a chained LEFT JOIN's own
        // NULL-padding flowing into a further equi-join, exercised here
        // directly instead.
        let executor = QueryExecutor::new(Catalog::new());
        executor.execute_sql("CREATE TABLE a (id INT PRIMARY KEY, fk INT)").unwrap();
        executor.execute_sql("CREATE TABLE b (id INT PRIMARY KEY, fk INT)").unwrap();
        executor.execute_sql("INSERT INTO a (id, fk) VALUES (1, NULL)").unwrap();
        executor.execute_sql("INSERT INTO b (id, fk) VALUES (10, NULL)").unwrap();

        let rows = executor.execute_sql("SELECT * FROM a JOIN b ON a.fk = b.fk").unwrap();
        assert!(rows.is_empty());
    }

    #[test]
    fn test_left_join_null_padded_row_correctly_stays_unmatched_in_a_later_join() {
        // The exact chained scenario the bug was found in: a LEFT JOIN's
        // own NULL-padding (b.a_id is NULL for every a-row, since b is
        // empty) must not spuriously match a third table's NULL-valued
        // row on a later join.
        let executor = QueryExecutor::new(Catalog::new());
        executor.execute_sql("CREATE TABLE a (id INT PRIMARY KEY, fk INT)").unwrap();
        executor.execute_sql("CREATE TABLE b (id INT PRIMARY KEY, a_id INT)").unwrap();
        executor.execute_sql("CREATE TABLE c (id INT PRIMARY KEY, fk INT)").unwrap();
        executor.execute_sql("INSERT INTO a (id, fk) VALUES (1, 100)").unwrap();
        executor.execute_sql("INSERT INTO a (id, fk) VALUES (2, 200)").unwrap();
        executor.execute_sql("INSERT INTO c (id, fk) VALUES (50, NULL)").unwrap();

        let rows = executor
            .execute_sql("SELECT * FROM a LEFT JOIN b ON a.id = b.a_id JOIN c ON b.a_id = c.fk")
            .unwrap();
        assert!(rows.is_empty());
    }

    #[test]
    fn test_hash_join_many_to_many_matching() {
        // users.id/orders.user_id are both unique-on-one-side (a PK and a
        // non-unique FK), so existing JOIN tests never exercise a key
        // that's duplicated on *both* sides of the hash table -- exactly
        // the shape that would expose a bug in hash_join_build_probe's
        // per-key Vec<&Tuple> bucket if it only ever handled a single
        // match per key. Two rows on each side sharing tag=1 must produce
        // all four combinations.
        let executor = QueryExecutor::new(Catalog::new());
        executor.execute_sql("CREATE TABLE a (id INT PRIMARY KEY, tag INT)").unwrap();
        executor.execute_sql("CREATE TABLE b (id INT PRIMARY KEY, tag INT)").unwrap();
        executor.execute_sql("INSERT INTO a (id, tag) VALUES (1, 1)").unwrap();
        executor.execute_sql("INSERT INTO a (id, tag) VALUES (2, 1)").unwrap();
        executor.execute_sql("INSERT INTO b (id, tag) VALUES (10, 1)").unwrap();
        executor.execute_sql("INSERT INTO b (id, tag) VALUES (20, 1)").unwrap();

        let rows = executor.execute_sql("SELECT * FROM a JOIN b ON a.tag = b.tag").unwrap();
        assert_eq!(rows.len(), 4); // 2 x 2 -- every a paired with every b sharing the tag
    }

    #[test]
    fn test_hash_join_picks_smaller_side_either_direction() {
        // hash_join builds the hash table on whichever side is smaller,
        // which changes which of build/probe is "left" -- exercise both
        // directions (left smaller, then right smaller) to prove the
        // swapped-output-ordering logic in hash_join_build_probe is
        // correct both ways, not just the common case.
        // Distinct id values on each side (99 vs 7) so a left/right
        // column-order bug would actually change the result, rather than
        // both orderings coincidentally producing the same row.
        let executor = QueryExecutor::new(Catalog::new());
        executor.execute_sql("CREATE TABLE small (id INT PRIMARY KEY, tag INT)").unwrap();
        executor.execute_sql("CREATE TABLE big (id INT PRIMARY KEY, tag INT)").unwrap();
        executor.execute_sql("INSERT INTO small (id, tag) VALUES (99, 5)").unwrap();
        executor.execute_sql("INSERT INTO big (id, tag) VALUES (7, 5)").unwrap();
        for i in 1..=4 {
            executor.execute_sql(&format!("INSERT INTO big (id, tag) VALUES ({}, 0)", i + 10)).unwrap();
        }

        // small (1 row) JOIN big (5 rows) -- small is the build side.
        let rows = executor.execute_sql("SELECT * FROM small JOIN big ON small.tag = big.tag").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0], vec!["99".to_string(), "5".to_string(), "7".to_string(), "5".to_string()]);

        // big (5 rows) JOIN small (1 row) -- small is still the build
        // side (picked by size, not join order), but output column order
        // must still be big-then-small since big is now the "left" table.
        let rows = executor.execute_sql("SELECT * FROM big JOIN small ON big.tag = small.tag").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0], vec!["7".to_string(), "5".to_string(), "99".to_string(), "5".to_string()]);
    }

    #[test]
    fn test_join_non_equality_condition_still_uses_nested_loop() {
        // A ">" condition between two bare columns is structurally valid
        // (passes the "compares two real columns" check) but isn't
        // hash-joinable -- must still fall back to the nested loop and
        // produce correct results, not be silently dropped or mishandled.
        let executor = QueryExecutor::new(Catalog::new());
        executor.execute_sql("CREATE TABLE a (id INT PRIMARY KEY, val INT)").unwrap();
        executor.execute_sql("CREATE TABLE b (id INT PRIMARY KEY, val INT)").unwrap();
        executor.execute_sql("INSERT INTO a (id, val) VALUES (1, 10)").unwrap();
        executor.execute_sql("INSERT INTO b (id, val) VALUES (1, 3)").unwrap();
        executor.execute_sql("INSERT INTO b (id, val) VALUES (2, 20)").unwrap();

        let rows = executor.execute_sql("SELECT * FROM a JOIN b ON a.val > b.val").unwrap();
        assert_eq!(rows.len(), 1); // a.val=10 > b.val=3, but not > b.val=20
        assert!(rows[0].contains(&"3".to_string()));
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
    fn test_limit_offset_skips_then_caps() {
        let executor = QueryExecutor::new(users_and_orders_catalog());
        seed_orders(&executor); // ids 100, 101, 102

        let rows = executor
            .execute_sql("SELECT id FROM orders ORDER BY id LIMIT 1 OFFSET 1")
            .unwrap();
        assert_eq!(rows, vec![vec!["101".to_string()]]);
    }

    #[test]
    fn test_offset_without_limit_skips_and_keeps_the_rest() {
        let executor = QueryExecutor::new(users_and_orders_catalog());
        seed_orders(&executor);

        let rows = executor.execute_sql("SELECT id FROM orders ORDER BY id OFFSET 1").unwrap();
        assert_eq!(rows, vec![vec!["101".to_string()], vec!["102".to_string()]]);
    }

    #[test]
    fn test_offset_past_every_row_is_an_empty_result_not_an_error() {
        let executor = QueryExecutor::new(users_and_orders_catalog());
        seed_orders(&executor);

        let rows = executor.execute_sql("SELECT id FROM orders ORDER BY id OFFSET 100").unwrap();
        assert!(rows.is_empty());
    }

    #[test]
    fn test_select_distinct_removes_duplicate_rows() {
        let executor = QueryExecutor::new(users_and_orders_catalog());
        seed_orders(&executor); // user_id: 1, 1, 2 -- one duplicate

        let rows = executor.execute_sql("SELECT DISTINCT user_id FROM orders").unwrap();
        let mut ids: Vec<&String> = rows.iter().map(|r| &r[0]).collect();
        ids.sort();
        assert_eq!(ids, vec!["1", "2"]);
    }

    #[test]
    fn test_select_all_does_not_deduplicate() {
        let executor = QueryExecutor::new(users_and_orders_catalog());
        seed_orders(&executor);

        // The explicit opposite of DISTINCT -- must keep every row,
        // duplicates included.
        let rows = executor.execute_sql("SELECT ALL user_id FROM orders").unwrap();
        assert_eq!(rows.len(), 3);
    }

    #[test]
    fn test_select_distinct_star_dedups_on_full_row() {
        let executor = QueryExecutor::new(users_and_orders_catalog());
        seed_orders(&executor);

        // No Project node for `SELECT *` -- Distinct must still work
        // directly against whatever's current at that point in the plan.
        // Every row's primary key differs here, so nothing actually
        // collapses; this asserts DISTINCT * doesn't error or drop rows
        // it shouldn't.
        let rows = executor.execute_sql("SELECT DISTINCT * FROM orders").unwrap();
        assert_eq!(rows.len(), 3);
    }

    #[test]
    fn test_select_distinct_preserves_order_by_result() {
        let executor = QueryExecutor::new(users_and_orders_catalog());
        seed_orders(&executor);

        let rows = executor
            .execute_sql("SELECT DISTINCT user_id FROM orders ORDER BY user_id DESC")
            .unwrap();
        assert_eq!(rows, vec![vec!["2".to_string()], vec!["1".to_string()]]);
    }

    #[test]
    fn test_having_filters_groups_by_aggregate_value() {
        let executor = QueryExecutor::new(users_and_orders_catalog());
        seed_orders(&executor); // user 1: 2 orders; user 2: 1 order

        let rows = executor
            .execute_sql("SELECT user_id, COUNT(*) FROM orders GROUP BY user_id HAVING COUNT(*) > 1")
            .unwrap();
        assert_eq!(rows, vec![vec!["1".to_string(), "2".to_string()]]);
    }

    #[test]
    fn test_having_can_reference_an_aggregate_not_in_the_select_list() {
        let executor = QueryExecutor::new(users_and_orders_catalog());
        seed_orders(&executor); // user 1: total 15.0; user 2: total 20.0

        let rows = executor
            .execute_sql("SELECT user_id FROM orders GROUP BY user_id HAVING SUM(total) > 15")
            .unwrap();
        assert_eq!(rows, vec![vec!["2".to_string()]]);
    }

    #[test]
    fn test_having_can_combine_group_by_key_and_aggregate_with_and() {
        let executor = QueryExecutor::new(users_and_orders_catalog());
        seed_orders(&executor);

        let rows = executor
            .execute_sql("SELECT user_id, COUNT(*) FROM orders GROUP BY user_id HAVING user_id = 1 AND COUNT(*) > 1")
            .unwrap();
        assert_eq!(rows, vec![vec!["1".to_string(), "2".to_string()]]);
    }

    #[test]
    fn test_having_without_group_by_or_aggregate_select_is_a_hard_error() {
        let executor = QueryExecutor::new(users_and_orders_catalog());
        seed_orders(&executor);

        let err = executor.execute_sql("SELECT * FROM orders HAVING total > 10");
        assert!(err.is_err());
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
    fn test_update_by_secondary_indexed_equality_uses_the_index_correctly() {
        // WHERE name = <literal> on an indexed, non-PK column is exactly
        // the shape candidate_rows_for_write's second tier (secondary
        // index candidates) accelerates -- asserts the same correct
        // result as the general full-scan path, not just that it's fast.
        let executor = QueryExecutor::new(users_catalog());
        executor.execute_sql("CREATE INDEX idx_name ON users (name)").unwrap();
        seed_users(&executor); // Alice 30, Bob 15

        let result = executor.execute_sql("UPDATE users SET age = 99 WHERE name = 'Alice'").unwrap();
        assert_eq!(result, vec![vec!["1".to_string()]]);

        let rows = executor.execute_sql("SELECT * FROM users").unwrap();
        let alice = rows.iter().find(|r| r.contains(&"Alice".to_string())).unwrap();
        assert!(alice.contains(&"99".to_string()));
        let bob = rows.iter().find(|r| r.contains(&"Bob".to_string())).unwrap();
        assert!(bob.contains(&"15".to_string())); // untouched
    }

    #[test]
    fn test_update_by_secondary_index_does_not_act_on_a_stale_candidate() {
        // secondary_index_candidates can be stale -- an UPDATE adds a new
        // index entry for a row's new value but never removes the old one
        // (see secondary_index module docs). This asserts
        // candidate_rows_for_write's re-verification (unchanged from the
        // full-scan path's own re-check) still catches that, rather than
        // the new acceleration trusting a stale candidate into a wrong
        // write.
        let executor = QueryExecutor::new(users_catalog());
        executor.execute_sql("CREATE INDEX idx_name ON users (name)").unwrap();
        seed_users(&executor); // Alice 30, Bob 15

        // Bob's old name ("Bob") stays in the index as a stale entry
        // after this rename.
        executor.execute_sql("UPDATE users SET name = 'Robert' WHERE name = 'Bob'").unwrap();

        // A second UPDATE against the now-stale "Bob" entry must affect
        // nothing -- the row it points at no longer actually has that name.
        let result = executor.execute_sql("UPDATE users SET age = 0 WHERE name = 'Bob'").unwrap();
        assert_eq!(result, vec![vec!["0".to_string()]]);

        let rows = executor.execute_sql("SELECT * FROM users WHERE name = 'Robert'").unwrap();
        assert_eq!(rows.len(), 1);
        assert!(rows[0].contains(&"15".to_string()), "Robert's age must be unchanged by the no-op UPDATE against the stale 'Bob' entry");
    }

    #[test]
    fn test_delete_by_secondary_indexed_equality_uses_the_index_correctly() {
        let executor = QueryExecutor::new(users_catalog());
        executor.execute_sql("CREATE INDEX idx_name ON users (name)").unwrap();
        seed_users(&executor); // Alice 30, Bob 15

        let result = executor.execute_sql("DELETE FROM users WHERE name = 'Alice'").unwrap();
        assert_eq!(result, vec![vec!["1".to_string()]]);

        let rows = executor.execute_sql("SELECT * FROM users").unwrap();
        assert_eq!(rows.len(), 1);
        assert!(rows[0].contains(&"Bob".to_string()));
    }

    #[test]
    fn test_update_by_secondary_indexed_range_uses_the_index_correctly() {
        let executor = QueryExecutor::new(users_catalog());
        executor.execute_sql("CREATE INDEX idx_age ON users (age)").unwrap();
        seed_users(&executor); // Alice 30, Bob 15

        let result = executor.execute_sql("UPDATE users SET age = 0 WHERE age > 20").unwrap();
        assert_eq!(result, vec![vec!["1".to_string()]]);

        let rows = executor.execute_sql("SELECT * FROM users").unwrap();
        let alice = rows.iter().find(|r| r.contains(&"Alice".to_string())).unwrap();
        assert!(alice.contains(&"0".to_string()));
        let bob = rows.iter().find(|r| r.contains(&"Bob".to_string())).unwrap();
        assert!(bob.contains(&"15".to_string())); // untouched
    }

    #[test]
    fn test_update_by_secondary_index_is_not_a_full_table_scan() {
        // Real regression coverage, same shape as
        // test_update_by_primary_key_is_not_a_full_table_scan: 3,000
        // single-row UPDATEs by an indexed, non-PK column against a
        // 3,000-row table is O(n) total with the index (each call touches
        // one row's candidate set) but O(n^2) with a full scan.
        let executor = QueryExecutor::new(users_catalog());
        executor.execute_sql("CREATE INDEX idx_name ON users (name)").unwrap();
        let insert_stmt = executor.prepare("INSERT INTO users (id, name, age) VALUES (?, ?, ?)").unwrap();
        let n = 3000;
        for id in 0..n {
            executor
                .execute_prepared(&insert_stmt, &[Value::Integer(id), Value::String(format!("user{id}")), Value::Integer(20)])
                .unwrap();
        }

        let update_stmt = executor.prepare("UPDATE users SET age = ? WHERE name = ?").unwrap();
        let start = std::time::Instant::now();
        for id in 0..n {
            executor
                .execute_prepared(&update_stmt, &[Value::Integer(21), Value::String(format!("user{id}"))])
                .unwrap();
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_secs_f64() < 2.0,
            "{n} single-row UPDATEs by an indexed column took {elapsed:.2?} -- looks like the full-scan path again"
        );
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
    fn test_analyze_unknown_table_errors() {
        let executor = QueryExecutor::new(users_catalog());
        assert!(executor.execute_sql("ANALYZE ghosts").is_err());
    }

    #[test]
    fn test_analyze_numeric_and_categorical_columns() {
        let executor = QueryExecutor::new(Catalog::new());
        executor
            .execute_sql("CREATE TABLE orders (id INT PRIMARY KEY, status VARCHAR(20), amount INT)")
            .unwrap();
        for i in 1..=950 {
            executor
                .execute_sql(&format!("INSERT INTO orders (id, status, amount) VALUES ({i}, 'shipped', {i})"))
                .unwrap();
        }
        for i in 951..=1000 {
            executor
                .execute_sql(&format!("INSERT INTO orders (id, status, amount) VALUES ({i}, 'cancelled', {i})"))
                .unwrap();
        }

        executor.execute_sql("ANALYZE orders").unwrap();

        let stats = executor.stats.read();
        assert!(stats.contains_key(&("orders".to_string(), "status".to_string())));
        assert!(stats.contains_key(&("orders".to_string(), "amount".to_string())));
        // "id" is the primary key -- still a real INT column, still analyzed.
        assert!(stats.contains_key(&("orders".to_string(), "id".to_string())));
    }

    #[test]
    fn test_analyze_then_select_still_returns_correct_rows() {
        // ANALYZE only changes plan *estimates* -- it must never change
        // what a query actually returns.
        let executor = QueryExecutor::new(Catalog::new());
        executor.execute_sql("CREATE TABLE orders (id INT PRIMARY KEY, status VARCHAR(20))").unwrap();
        executor.execute_sql("INSERT INTO orders (id, status) VALUES (1, 'shipped')").unwrap();
        executor.execute_sql("INSERT INTO orders (id, status) VALUES (2, 'cancelled')").unwrap();
        executor.execute_sql("ANALYZE orders").unwrap();

        let rows = executor.execute_sql("SELECT * FROM orders WHERE status = 'cancelled'").unwrap();
        assert_eq!(rows.len(), 1);
        assert!(rows[0].contains(&"2".to_string()));
    }

    #[test]
    fn test_analyze_makes_plan_estimate_dramatically_more_accurate_than_fixed_heuristic() {
        // The actual proof this feature exists for. Deliberately a
        // numeric range predicate, not a categorical equality one: a
        // categorical column's equality estimate assumes uniformity
        // across *distinct values* (1/distinct_count -- see
        // ColumnDistribution::estimate_selectivity's docs), which for a
        // column with only 2 distinct values (a status-like column) is
        // 1/2 = 0.5 regardless of the real skew -- no better than the
        // fixed heuristic it's replacing, an honest limitation, not a
        // bug (see cardinality.rs's own
        // test_categorical_equality_selectivity_uses_real_distinct_count).
        // A numeric column's *range* estimate has no such limitation: it
        // comes from a real fitted CDF (row_codec::pgm's PGMIndex), not a
        // distinct-value count, so this is the case that actually
        // demonstrates the win.
        let executor = QueryExecutor::new(Catalog::new());
        executor.execute_sql("CREATE TABLE orders (id INT PRIMARY KEY, amount INT)").unwrap();
        for i in 1..=950 {
            executor.execute_sql(&format!("INSERT INTO orders (id, amount) VALUES ({i}, 10)")).unwrap();
        }
        for i in 951..=1000 {
            executor.execute_sql(&format!("INSERT INTO orders (id, amount) VALUES ({i}, 999)")).unwrap();
        }
        let ground_truth = executor.execute_sql("SELECT * FROM orders WHERE amount >= 999").unwrap().len();
        assert_eq!(ground_truth, 50); // sanity: the data really is this skewed

        // Before ANALYZE: the same fixed 0.5 selectivity this planner
        // has always used, applied to the assumed 1000-row table size --
        // 500 estimated rows against a true 50. 10x wrong.
        let before = executor.explain("SELECT * FROM orders WHERE amount >= 999").unwrap().estimated_rows;
        assert_eq!(before, 500);

        executor.execute_sql("ANALYZE orders").unwrap();

        // After ANALYZE: a real, data-driven estimate close to the true 50.
        let after = executor.explain("SELECT * FROM orders WHERE amount >= 999").unwrap().estimated_rows;
        assert!(
            (after as i64 - 50).abs() < 20,
            "post-ANALYZE estimate {after} should be close to the true {ground_truth}, not the fixed-heuristic {before}"
        );
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
