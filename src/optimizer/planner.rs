//! Query planner with learned optimization
//!
//! Generates query plans by parsing the query and consulting the cost model
//! and join orderer already held on this planner (previously constructed
//! but never used — `plan()` ignored its input and returned a hardcoded
//! single-table scan regardless of what was asked). Row-count estimates
//! are real when `stats` — a map of real
//! `optimizer::cardinality::ColumnDistribution`s, built by
//! `QueryExecutor::execute_analyze` from an actual scan of a table's
//! values (see that type's docs) — has something for the table being
//! queried: a bare scan's row count comes from any analyzed column's
//! `total_rows` (every column gets scanned together, so any one of them
//! gives the whole table's real size), and a `WHERE`-clause estimate uses
//! a real selectivity when the filter reduces to a single
//! `<column> <literal>` comparison against a table with no `JOIN` before
//! it and a distribution on record for that column. Both fall back to
//! the old fixed constants (`ASSUMED_TABLE_ROWS`,
//! `DEFAULT_FILTER_SELECTIVITY`) whenever real stats aren't available or
//! don't apply — no stats yet, a compound `AND`/`OR` predicate, a `JOIN`
//! in the way, a literal on the wrong side — graceful degradation, not a
//! new failure mode. A real base row count matters for more than just
//! looking accurate on its own: a Filter's selectivity, however real,
//! produces a meaningless final row count if it's multiplied against a
//! fake base — the two only add up to a real number together.
//!
//! Multi-table queries also get a real decision, not just a real
//! estimate: `JOIN` clauses are handed to
//! `join_reorder::JoinOrderer::find_optimal_order`, which searches for
//! the cheapest order that's provably safe to run (see that function's
//! docs) using real distinct-value counts when `stats` has them, falling
//! back to source order otherwise. The executor has no independent
//! notion of join order — it walks this planner's `Join` nodes in
//! whatever sequence they're emitted — so this is a real reordering, not
//! a number that gets computed and then ignored.

use crate::error::{DatabaseError, Result};
use crate::execution::aggregate;
use crate::execution::operators::Value;
use crate::execution::row_codec;
use crate::optimizer::cardinality::ColumnDistribution;
use crate::optimizer::cost_model::{CostModel, OperatorCost, OperatorType};
use crate::optimizer::join_reorder::JoinOrderer;
use crate::sql::parser::{EquiMatch, JoinKind, SQLParser, SQLStatement, SelectStatement};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Logical query plan node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogicalPlanNode {
    Scan {
        table_id: u32,
        table_name: String,
        /// The table's alias in the query (`FROM t AS a`), if any — used to
        /// qualify this table's columns when it's joined to another (see
        /// `execution::executor::QueryExecutor::merge_schemas`).
        alias: Option<String>,
        rows: usize,
    },
    Filter {
        predicate: String,
        /// `predicate` pre-split into `(left, op, right)` — see
        /// `row_codec::split_comparison` — when it's a single comparison;
        /// `None` for a compound `AND`/`OR` predicate, or anything else
        /// `split_comparison` can't reduce to one triple. Computed once,
        /// here, at plan time (itself just one tokenize, reused for the
        /// life of a prepared statement) so neither
        /// `QueryExecutor::execute_prepared`'s parameter substitution nor
        /// `execute`'s indexed-scan eligibility check has to re-tokenize
        /// this same string from scratch on every single execution — see
        /// both call sites.
        split: Option<(String, String, String)>,
        rows: usize,
    },
    Join {
        right_table: String,
        /// The joined table's alias, if any — same role as `Scan::alias`.
        right_alias: Option<String>,
        /// The `ON` condition, flattened to a string (see `sql::parser`'s
        /// `JoinClause`). `None` when the join was unconditional (`CROSS
        /// JOIN`), or written as `USING`/`NATURAL` (see `equi_match`).
        condition: Option<String>,
        /// `USING`/`NATURAL`, when the join was written that way instead
        /// of `ON` — see `sql::parser::EquiMatch`'s docs for why this
        /// isn't resolved into `condition` before execution time.
        equi_match: Option<EquiMatch>,
        left_rows: usize,
        right_rows: usize,
        /// `INNER`/`LEFT`/`RIGHT`/`FULL OUTER`, from `sql::parser::JoinClause::kind`
        /// -- the executor's `Join` arm reads this to decide whether an
        /// unmatched row on either side is dropped (`Inner`) or emitted
        /// once with `NULL`s on the other side.
        kind: JoinKind,
    },
    Aggregate {
        /// The projected columns: each is either a recognized aggregate
        /// call (see `execution::aggregate::parse_aggregate`) or, when
        /// `group_by` is non-empty, one of the grouping columns —
        /// re-parsed/looked-up at execution time rather than duplicating a
        /// typed spec here.
        columns: Vec<String>,
        /// `GROUP BY` column names. Empty means a single ungrouped
        /// aggregate — the whole input collapses to one output row.
        group_by: Vec<String>,
        /// The flattened `HAVING <expr>` clause, if any — filters groups
        /// *after* aggregation, unlike `Filter`'s `WHERE` which runs
        /// before. An earlier version of this planner had no field for
        /// this at all: `select.having` was parsed and then never stored
        /// anywhere, so `GROUP BY x HAVING COUNT(*) > 1` silently returned
        /// every group instead of just the ones matching the predicate.
        having: Option<String>,
        rows: usize,
    },
    /// Keep only `columns`, in this order, dropping everything else from
    /// each row — what `SELECT <col1>, <col2>` actually means. An earlier
    /// version of this planner had no node for this at all: `select.columns`
    /// was consulted only to detect an aggregate query (see `Aggregate`
    /// above) and otherwise silently discarded, so a plain, non-aggregate
    /// `SELECT` with an explicit column list always returned every column
    /// — identical to `SELECT *` regardless of what was actually asked
    /// for. Never emitted for `SELECT *` (nothing to drop) or an
    /// aggregate query (`Aggregate` already produces exactly the
    /// requested output shape on its own).
    Project {
        /// Resolved against whatever the current schema's column names
        /// are at execution time (see `execution::executor::execute`'s
        /// `Project` arm) — bare names for a single-table query, or
        /// `"qualifier.column"` after a `JOIN`, matching how every other
        /// column reference in this codebase's flattened predicates
        /// already works. An unresolvable name is a hard error, not a
        /// silently dropped column.
        columns: Vec<String>,
        rows: usize,
    },
    /// `ORDER BY <col> [ASC|DESC], ...` — an earlier version of this
    /// planner parsed `select.order_by` into a real string and then never
    /// consulted it anywhere; rows came back in whatever order the
    /// storage layer happened to produce them, `ORDER BY` clause or not.
    /// Placed after `Filter`/`Join`/`Aggregate` but before `Project`, so
    /// a sort key that isn't in the `SELECT` list (`SELECT name FROM t
    /// ORDER BY age`) is still available to sort by — real SQL allows
    /// that, and narrowing columns first would break it.
    Sort {
        /// `(column, ascending)` pairs in priority order — see
        /// `row_codec::parse_order_by`, which produced this list, and
        /// `row_codec::compare_for_sort` for the actual per-value
        /// ordering `execution::executor::execute`'s `Sort` arm uses.
        keys: Vec<(String, bool)>,
        rows: usize,
    },
    /// `SELECT DISTINCT` — same story again: `select.distinct` was parsed
    /// from `sqlparser`'s AST and then never stored on `SelectStatement`
    /// at all, so `SELECT DISTINCT user_id FROM orders` returned every
    /// row, duplicates included, identical to plain `SELECT`. Placed
    /// after `Project` (or after `Sort` when there's no `Project` node,
    /// i.e. `SELECT DISTINCT *`) so it dedups the query's actual final
    /// row shape, not a wider pre-projection one — two rows that only
    /// differ in a column `SELECT` doesn't return must still collapse to
    /// one. Placed before `Limit`, so `LIMIT` caps the deduplicated set.
    Distinct { rows: usize },
    /// `LIMIT <n>` — same story as `Sort`: parsed into `select.limit` and
    /// then silently ignored everywhere. Always the last node in a plan
    /// when present, so it caps whatever `Sort`/`Project`/`Aggregate`/`Distinct`
    /// already produced rather than racing them.
    Limit { limit: usize, rows: usize },
}

/// Physical query plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicalPlan {
    pub nodes: Vec<LogicalPlanNode>,
    pub estimated_cost: f64,
    pub estimated_rows: usize,
}

/// Query planner
pub struct QueryPlanner {
    pub cost_model: CostModel,
    pub join_orderer: JoinOrderer,
}

impl QueryPlanner {
    pub fn new() -> Self {
        QueryPlanner {
            cost_model: CostModel::new(),
            join_orderer: JoinOrderer::new(),
        }
    }

    /// Assumed row count for a scanned table. There's no catalog or
    /// table-statistics wiring into the planner yet, so this is a fixed
    /// placeholder rather than a real number — same role the old hardcoded
    /// `1000` played, just no longer silently ignoring the actual query.
    const ASSUMED_TABLE_ROWS: usize = 1000;
    /// Default join selectivity used when estimating a joined table's
    /// output cardinality, absent any real join-key statistics.
    const DEFAULT_JOIN_SELECTIVITY: f64 = 0.1;
    /// Default filter selectivity used when a WHERE clause is present.
    /// The learned cardinality estimator isn't used here: it needs
    /// structured predicates, and a WHERE clause is currently just a
    /// flattened string (see `sql::parser`) with nothing to extract them
    /// from honestly.
    const DEFAULT_FILTER_SELECTIVITY: f64 = 0.5;

    /// Generate a query plan for `query`, with no real column statistics
    /// available (equivalent to `plan_select(select, &HashMap::new())`).
    /// Parses `query` itself; a caller that already has a parsed
    /// `SelectStatement` in hand (`execute_sql`, `QueryExecutor::prepare`)
    /// should call `plan_select` directly instead — re-parsing a string
    /// that was just parsed one call up is exactly the wasted-work
    /// pattern `row_codec::CompiledPredicate` existed to eliminate on the
    /// predicate side; this is the same fix on the planning side.
    pub fn plan(&self, query: &str) -> Result<PhysicalPlan> {
        match SQLParser::parse(query) {
            Ok(SQLStatement::Select(select)) => self.plan_select(&select, &HashMap::new()),
            // Not a SELECT, or failed to parse: nothing to build a read
            // plan for yet (writes and DDL don't have a plan shape here).
            _ => Ok(PhysicalPlan {
                nodes: vec![LogicalPlanNode::Scan {
                    table_id: 0,
                    table_name: String::new(),
                    alias: None,
                    rows: 0,
                }],
                estimated_cost: 0.0,
                estimated_rows: 0,
            }),
        }
    }

    /// The actual plan-building logic, over an already-parsed `select` --
    /// see `plan`'s doc comment for why this is the one to call when a
    /// parsed statement already exists. `stats` is keyed by
    /// `(table_name, column_name)`; an empty map degrades exactly to this
    /// planner's old always-constant behavior.
    pub fn plan_select(&self, select: &SelectStatement, stats: &HashMap<(String, String), ColumnDistribution>) -> Result<PhysicalPlan> {
        // Any analyzed column's total_rows is the whole table's real row
        // count as of that ANALYZE (every column gets scanned together),
        // not just that one column's -- so the first stats entry found
        // for this table gives a real base row count for the bare scan,
        // in place of the fixed ASSUMED_TABLE_ROWS placeholder. Without
        // this, a Filter's real selectivity would still be multiplied
        // against a fake base count, and the result would be no more
        // meaningful than the constant it replaced.
        // Maps each table's qualifier as it appears in JOIN conditions
        // (its alias if it has one, else its real name) back to the real
        // table name `stats` is keyed by -- needed because a condition
        // like "u.id = o.user_id" only ever names aliases, never the
        // underlying table, once a query uses them.
        let mut qualifier_to_table: HashMap<String, String> = HashMap::new();
        let from_qualifier = select.from_alias.clone().unwrap_or_else(|| select.from.clone());
        qualifier_to_table.insert(from_qualifier.clone(), select.from.clone());
        for join in &select.joins {
            let q = join.alias.clone().unwrap_or_else(|| join.table.clone());
            qualifier_to_table.insert(q, join.table.clone());
        }
        let row_count_of = |qualifier: &str| -> usize {
            qualifier_to_table
                .get(qualifier)
                .and_then(|table| stats.iter().find(|((t, _), _)| t == table).map(|(_, dist)| dist.total_rows()))
                .unwrap_or(Self::ASSUMED_TABLE_ROWS)
        };
        let distinct_count_of = |qualifier: &str, column: &str| -> Option<usize> {
            let table = qualifier_to_table.get(qualifier)?;
            stats.get(&(table.clone(), column.to_string())).map(|dist| dist.distinct_count())
        };

        let mut rows = row_count_of(&from_qualifier);
        let mut nodes = vec![LogicalPlanNode::Scan {
            table_id: 0,
            table_name: select.from.clone(),
            alias: select.from_alias.clone(),
            rows,
        }];

        // Real join ordering: search for the cheapest order that's
        // provably resolvable (every join's condition can be evaluated
        // against whatever's already been accumulated at that point),
        // falling back to source order otherwise -- see
        // `join_reorder::JoinOrderer::find_optimal_order`'s docs. The
        // executor just walks `nodes` in whatever order they're emitted
        // here, so this loop's order IS the execution order.
        let planned_joins = self.join_orderer.find_optimal_order(
            &from_qualifier,
            rows,
            &select.joins,
            row_count_of,
            distinct_count_of,
            Self::DEFAULT_JOIN_SELECTIVITY,
        );
        for planned in &planned_joins {
            let join = &select.joins[planned.join_index];
            nodes.push(LogicalPlanNode::Join {
                right_table: join.table.clone(),
                right_alias: join.alias.clone(),
                condition: join.condition.clone(),
                equi_match: join.equi_match.clone(),
                left_rows: planned.left_rows,
                right_rows: planned.right_rows,
                kind: join.kind,
            });
            rows = planned.output_rows;
        }

        if let Some(predicate) = select.where_clause.clone() {
            // Split once, reused two ways below: the cardinality estimate
            // (when applicable) and the Filter node's own `split` field
            // (see that field's docs for why -- avoiding a second
            // tokenize of this same string at execute time, on every
            // single prepared-statement execution).
            let split = row_codec::split_comparison(&predicate);

            // A real, data-driven estimate only when the filter is a
            // single <column> <op> <literal> comparison against a table
            // nothing has been JOINed to yet (a post-join filter's
            // columns live in a merged, qualified schema this planner
            // doesn't resolve) and a distribution is on record for that
            // exact column. Every other shape keeps the fixed default it
            // always used.
            let real_estimate = if select.joins.is_empty() {
                split.clone().and_then(|(left, op, right)| {
                    let dist = stats.get(&(select.from.clone(), left))?;
                    let value = literal_for_distribution(&right, dist)?;
                    dist.estimate_row_count(&op, &value, rows)
                })
            } else {
                None
            };
            rows = real_estimate.unwrap_or_else(|| (rows as f64 * Self::DEFAULT_FILTER_SELECTIVITY) as usize);
            nodes.push(LogicalPlanNode::Filter { predicate, split, rows });
        }

        // Treat this as an aggregate query when every projected column is
        // either a recognized aggregate call or (with GROUP BY) one of the
        // grouping columns -- anything else (a plain column that's neither)
        // is left as a plain, if not fully correct, row-returning plan
        // instead of pretending to aggregate.
        let is_aggregate_query = !select.columns.is_empty()
            && select
                .columns
                .iter()
                .all(|c| aggregate::parse_aggregate(c).is_some() || select.group_by.contains(c));

        if is_aggregate_query {
            rows = if select.group_by.is_empty() {
                1 // no GROUP BY: aggregation always collapses to one row
            } else {
                // No real cardinality data to estimate distinct groups
                // from; sqrt(rows) is a common rough heuristic, not a
                // measurement -- good enough since nothing depends on it
                // for correctness, only the cost estimate.
                ((rows as f64).sqrt().ceil() as usize).max(1)
            };
            nodes.push(LogicalPlanNode::Aggregate {
                columns: select.columns.clone(),
                group_by: select.group_by.clone(),
                having: select.having.clone(),
                rows,
            });
        } else if select.having.is_some() {
            // HAVING only means something over groups -- a query that
            // isn't recognized as an aggregate query has no Aggregate node
            // for it to attach to. Erroring here (rather than silently
            // dropping select.having, which is exactly the bug this field
            // exists to fix) matches how an unbound `?`/`$N` placeholder
            // reaching execute_sql is also a hard error instead of a
            // silent no-op.
            return Err(DatabaseError::ExecutionError(
                "HAVING requires GROUP BY or an all-aggregate SELECT list".to_string(),
            ));
        }

        // ORDER BY: after Filter/Join/Aggregate, before Project -- a sort
        // key that isn't in the SELECT list (SELECT name FROM t ORDER BY
        // age) must still be available to sort by, and Project would
        // already have dropped it.
        if let Some(order_by) = &select.order_by {
            let keys = row_codec::parse_order_by(order_by);
            if !keys.is_empty() {
                nodes.push(LogicalPlanNode::Sort { keys, rows });
            }
        }

        if !is_aggregate_query && !(select.columns.len() == 1 && select.columns[0] == "*") {
            // A real column list, not SELECT * -- see Project's docs for
            // why this needs its own node rather than being silently
            // ignored the way it used to be.
            nodes.push(LogicalPlanNode::Project { columns: select.columns.clone(), rows });
        }

        if select.distinct {
            // No real cardinality data on how many rows collapse -- rows
            // carries through unchanged, an intentional overestimate
            // (same "don't invent a number nothing backs" stance as
            // ORDER BY's rows pass-through above) rather than a guessed
            // reduction.
            nodes.push(LogicalPlanNode::Distinct { rows });
        }

        // LIMIT: always last -- caps whatever Sort/Project/Aggregate/Distinct
        // already produced rather than racing any of them.
        if let Some(limit) = select.limit {
            rows = rows.min(limit);
            nodes.push(LogicalPlanNode::Limit { limit, rows });
        }

        let estimated_cost = self.cost_model.estimate_total_cost(&Self::to_operator_costs(&nodes));

        Ok(PhysicalPlan {
            nodes,
            estimated_cost,
            estimated_rows: rows,
        })
    }

    /// Map plan nodes to the cost model's operator representation so
    /// `CostModel` (built, but never previously consulted by `plan()`)
    /// actually drives the plan's cost estimate.
    fn to_operator_costs(nodes: &[LogicalPlanNode]) -> Vec<OperatorCost> {
        let mut input_rows = 0usize;
        nodes
            .iter()
            .map(|node| {
                let (op_type, output_rows) = match node {
                    LogicalPlanNode::Scan { rows, .. } => (OperatorType::TableScan, *rows),
                    LogicalPlanNode::Filter { rows, .. } => (OperatorType::Filter, *rows),
                    LogicalPlanNode::Join {
                        left_rows,
                        right_rows,
                        ..
                    } => (OperatorType::HashJoin, (*left_rows).max(*right_rows)),
                    LogicalPlanNode::Aggregate { rows, .. } => (OperatorType::Aggregate, *rows),
                    LogicalPlanNode::Project { rows, .. } => (OperatorType::Filter, *rows),
                    LogicalPlanNode::Sort { rows, .. } => (OperatorType::Sort, *rows),
                    LogicalPlanNode::Distinct { rows, .. } => (OperatorType::Aggregate, *rows),
                    LogicalPlanNode::Limit { rows, .. } => (OperatorType::Limit, *rows),
                };
                let cost = OperatorCost::new(
                    op_type,
                    if input_rows == 0 { output_rows } else { input_rows },
                    output_rows,
                    1.0,
                );
                input_rows = output_rows;
                cost
            })
            .collect()
    }
}

impl Default for QueryPlanner {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse `raw` (a flattened literal token from `sql::parser`, e.g. `"18"`
/// or `"'cancelled'"`) into the `Value` shape `dist` expects — a bare
/// number for a `Numeric` distribution, a (quote-stripped) string for a
/// `Categorical` one. `None` if `raw` doesn't parse as the shape `dist`
/// needs (e.g. a non-numeric literal against a `Numeric` column), same
/// "can't evaluate this way" convention used throughout this codebase.
fn literal_for_distribution(raw: &str, dist: &ColumnDistribution) -> Option<Value> {
    match dist {
        ColumnDistribution::Numeric { .. } => raw.trim().parse::<f64>().ok().map(Value::Float),
        ColumnDistribution::Categorical { .. } => {
            let s = raw.trim();
            let s = s.strip_prefix('\'').and_then(|s| s.strip_suffix('\'')).unwrap_or(s);
            Some(Value::String(s.to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_planner_creation() {
        let planner = QueryPlanner::new();
        let plan = planner.plan("SELECT * FROM table1").unwrap();
        assert!(!plan.nodes.is_empty());
    }

    #[test]
    fn test_plan_reflects_real_table_name() {
        let planner = QueryPlanner::new();
        let plan = planner.plan("SELECT * FROM orders").unwrap();
        match &plan.nodes[0] {
            LogicalPlanNode::Scan { table_name, .. } => assert_eq!(table_name, "orders"),
            other => panic!("expected a Scan node, got {other:?}"),
        }
    }

    #[test]
    fn test_plan_emits_filter_node_for_where_clause() {
        let planner = QueryPlanner::new();
        let plan = planner.plan("SELECT * FROM users WHERE age > 18").unwrap();
        assert!(plan
            .nodes
            .iter()
            .any(|n| matches!(n, LogicalPlanNode::Filter { .. })));
        // The filter should narrow the estimate below a bare scan.
        assert!(plan.estimated_rows < QueryPlanner::ASSUMED_TABLE_ROWS);
    }

    #[test]
    fn test_plan_emits_join_node_for_join_query() {
        let planner = QueryPlanner::new();
        let plan = planner.plan("SELECT * FROM users JOIN orders ON users.id = orders.user_id").unwrap();
        assert!(plan
            .nodes
            .iter()
            .any(|n| matches!(n, LogicalPlanNode::Join { .. })));
    }

    #[test]
    fn test_plan_non_select_falls_back_to_empty_plan() {
        let planner = QueryPlanner::new();
        let plan = planner.plan("DELETE FROM users WHERE id = 1").unwrap();
        assert_eq!(plan.estimated_rows, 0);
    }

    #[test]
    fn test_plan_emits_aggregate_node_when_all_columns_are_aggregates() {
        let planner = QueryPlanner::new();
        let plan = planner.plan("SELECT COUNT(*), AVG(age) FROM users").unwrap();
        assert!(plan
            .nodes
            .iter()
            .any(|n| matches!(n, LogicalPlanNode::Aggregate { .. })));
        assert_eq!(plan.estimated_rows, 1);
    }

    #[test]
    fn test_plan_emits_project_node_for_explicit_column_list() {
        // Previously select.columns was only ever consulted to detect an
        // aggregate query -- a plain "SELECT name, age" got no projection
        // node at all, and execute() returned every column, same as
        // SELECT *.
        let planner = QueryPlanner::new();
        let plan = planner.plan("SELECT name, age FROM users").unwrap();
        match plan.nodes.iter().find(|n| matches!(n, LogicalPlanNode::Project { .. })) {
            Some(LogicalPlanNode::Project { columns, .. }) => {
                assert_eq!(columns, &vec!["name".to_string(), "age".to_string()]);
            }
            _ => panic!("expected a Project node"),
        }
    }

    #[test]
    fn test_plan_does_not_emit_project_node_for_select_star() {
        let planner = QueryPlanner::new();
        let plan = planner.plan("SELECT * FROM users").unwrap();
        assert!(!plan.nodes.iter().any(|n| matches!(n, LogicalPlanNode::Project { .. })));
    }

    #[test]
    fn test_plan_emits_sort_node_for_order_by() {
        // Previously select.order_by was parsed into a real string and
        // then never consulted anywhere -- no Sort node at all.
        let planner = QueryPlanner::new();
        let plan = planner.plan("SELECT * FROM users ORDER BY age DESC").unwrap();
        match plan.nodes.iter().find(|n| matches!(n, LogicalPlanNode::Sort { .. })) {
            Some(LogicalPlanNode::Sort { keys, .. }) => {
                assert_eq!(keys, &vec![("age".to_string(), false)]);
            }
            _ => panic!("expected a Sort node"),
        }
    }

    #[test]
    fn test_plan_emits_sort_node_for_multi_column_order_by() {
        let planner = QueryPlanner::new();
        let plan = planner.plan("SELECT * FROM users ORDER BY age, name DESC").unwrap();
        match plan.nodes.iter().find(|n| matches!(n, LogicalPlanNode::Sort { .. })) {
            Some(LogicalPlanNode::Sort { keys, .. }) => {
                assert_eq!(keys, &vec![("age".to_string(), true), ("name".to_string(), false)]);
            }
            _ => panic!("expected a Sort node"),
        }
    }

    #[test]
    fn test_plan_emits_limit_node_for_limit_clause() {
        // Previously select.limit was parsed and then never consulted --
        // no Limit node at all.
        let planner = QueryPlanner::new();
        let plan = planner.plan("SELECT * FROM users LIMIT 5").unwrap();
        match plan.nodes.iter().find(|n| matches!(n, LogicalPlanNode::Limit { .. })) {
            Some(LogicalPlanNode::Limit { limit, .. }) => assert_eq!(*limit, 5),
            _ => panic!("expected a Limit node"),
        }
    }

    #[test]
    fn test_plan_omits_sort_and_limit_nodes_when_absent() {
        let planner = QueryPlanner::new();
        let plan = planner.plan("SELECT * FROM users").unwrap();
        assert!(!plan.nodes.iter().any(|n| matches!(n, LogicalPlanNode::Sort { .. })));
        assert!(!plan.nodes.iter().any(|n| matches!(n, LogicalPlanNode::Limit { .. })));
    }

    #[test]
    fn test_plan_does_not_emit_aggregate_for_mixed_columns() {
        let planner = QueryPlanner::new();
        // "name" alongside COUNT(*) with no GROUP BY at all is invalid --
        // must not be misdetected as a pure aggregate query.
        let plan = planner.plan("SELECT name, COUNT(*) FROM users").unwrap();
        assert!(!plan
            .nodes
            .iter()
            .any(|n| matches!(n, LogicalPlanNode::Aggregate { .. })));
    }

    #[test]
    fn test_plan_emits_aggregate_node_for_group_by() {
        let planner = QueryPlanner::new();
        let plan = planner.plan("SELECT user_id, COUNT(*) FROM orders GROUP BY user_id").unwrap();
        match plan.nodes.iter().find(|n| matches!(n, LogicalPlanNode::Aggregate { .. })) {
            Some(LogicalPlanNode::Aggregate { columns, group_by, .. }) => {
                assert_eq!(columns, &vec!["user_id".to_string(), "COUNT(*)".to_string()]);
                assert_eq!(group_by, &vec!["user_id".to_string()]);
            }
            _ => panic!("expected an Aggregate node"),
        }
    }

    #[test]
    fn test_plan_rejects_group_by_with_ungrouped_plain_column() {
        let planner = QueryPlanner::new();
        // "name" is neither an aggregate call nor a GROUP BY column --
        // still not a valid aggregate query shape even with GROUP BY present.
        let plan = planner.plan("SELECT name, COUNT(*) FROM orders GROUP BY user_id").unwrap();
        assert!(!plan
            .nodes
            .iter()
            .any(|n| matches!(n, LogicalPlanNode::Aggregate { .. })));
    }

    #[test]
    fn test_plan_carries_having_onto_the_aggregate_node() {
        let planner = QueryPlanner::new();
        let plan = planner
            .plan("SELECT user_id, COUNT(*) FROM orders GROUP BY user_id HAVING COUNT(*) > 1")
            .unwrap();
        match plan.nodes.iter().find(|n| matches!(n, LogicalPlanNode::Aggregate { .. })) {
            Some(LogicalPlanNode::Aggregate { having, .. }) => {
                assert_eq!(having.as_deref(), Some("COUNT(*) > 1"));
            }
            _ => panic!("expected an Aggregate node"),
        }
    }

    #[test]
    fn test_plan_rejects_having_without_group_by_or_aggregate_select() {
        let planner = QueryPlanner::new();
        let err = planner.plan("SELECT * FROM users HAVING age > 18");
        assert!(err.is_err());
    }

    #[test]
    fn test_plan_emits_distinct_node_for_select_distinct() {
        let planner = QueryPlanner::new();
        let plan = planner.plan("SELECT DISTINCT user_id FROM orders").unwrap();
        assert!(plan.nodes.iter().any(|n| matches!(n, LogicalPlanNode::Distinct { .. })));
    }

    #[test]
    fn test_plan_omits_distinct_node_for_plain_select() {
        let planner = QueryPlanner::new();
        let plan = planner.plan("SELECT user_id FROM orders").unwrap();
        assert!(!plan.nodes.iter().any(|n| matches!(n, LogicalPlanNode::Distinct { .. })));
    }

    #[test]
    fn test_different_queries_yield_different_costs() {
        let planner = QueryPlanner::new();
        let scan_only = planner.plan("SELECT * FROM users").unwrap();
        let filtered = planner.plan("SELECT * FROM users WHERE age > 18").unwrap();
        // A different query shape should actually change the estimate now,
        // instead of every query getting the same hardcoded 100.0/1000.
        assert_ne!(scan_only.estimated_cost, filtered.estimated_cost);
        assert_ne!(scan_only.estimated_rows, filtered.estimated_rows);
    }
}
