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

use crate::execution::aggregate;
use crate::execution::operators::Value;
use crate::execution::row_codec;
use crate::optimizer::cardinality::ColumnDistribution;
use crate::optimizer::cost_model::{CostModel, OperatorCost, OperatorType};
use crate::optimizer::join_reorder::JoinOrderer;
use crate::sql::parser::{SQLParser, SQLStatement, SelectStatement};
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
        rows: usize,
    },
    Join {
        right_table: String,
        /// The joined table's alias, if any — same role as `Scan::alias`.
        right_alias: Option<String>,
        /// The `ON` condition, flattened to a string (see `sql::parser`'s
        /// `JoinClause`). `None` means an unconditional join (`CROSS JOIN`,
        /// or a `USING`/`NATURAL` join — those aren't specially resolved,
        /// so they degrade to the same thing as `CROSS JOIN`).
        condition: Option<String>,
        left_rows: usize,
        right_rows: usize,
        join_type: String,
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
        rows: usize,
    },
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
    pub fn plan(&self, query: &str) -> PhysicalPlan {
        match SQLParser::parse(query) {
            Ok(SQLStatement::Select(select)) => self.plan_select(&select, &HashMap::new()),
            // Not a SELECT, or failed to parse: nothing to build a read
            // plan for yet (writes and DDL don't have a plan shape here).
            _ => PhysicalPlan {
                nodes: vec![LogicalPlanNode::Scan {
                    table_id: 0,
                    table_name: String::new(),
                    alias: None,
                    rows: 0,
                }],
                estimated_cost: 0.0,
                estimated_rows: 0,
            },
        }
    }

    /// The actual plan-building logic, over an already-parsed `select` --
    /// see `plan`'s doc comment for why this is the one to call when a
    /// parsed statement already exists. `stats` is keyed by
    /// `(table_name, column_name)`; an empty map degrades exactly to this
    /// planner's old always-constant behavior.
    pub fn plan_select(&self, select: &SelectStatement, stats: &HashMap<(String, String), ColumnDistribution>) -> PhysicalPlan {
        // Any analyzed column's total_rows is the whole table's real row
        // count as of that ANALYZE (every column gets scanned together),
        // not just that one column's -- so the first stats entry found
        // for this table gives a real base row count for the bare scan,
        // in place of the fixed ASSUMED_TABLE_ROWS placeholder. Without
        // this, a Filter's real selectivity would still be multiplied
        // against a fake base count, and the result would be no more
        // meaningful than the constant it replaced.
        let mut rows = stats
            .iter()
            .find(|((table, _), _)| table == &select.from)
            .map(|(_, dist)| dist.total_rows())
            .unwrap_or(Self::ASSUMED_TABLE_ROWS);
        let mut nodes = vec![LogicalPlanNode::Scan {
            table_id: 0,
            table_name: select.from.clone(),
            alias: select.from_alias.clone(),
            rows,
        }];

        for join in &select.joins {
            let (left_rows, right_rows) = (rows, Self::ASSUMED_TABLE_ROWS);
            rows = self.join_orderer.estimate_join_cardinality(
                left_rows,
                right_rows,
                Self::DEFAULT_JOIN_SELECTIVITY,
            );
            nodes.push(LogicalPlanNode::Join {
                right_table: join.table.clone(),
                right_alias: join.alias.clone(),
                condition: join.condition.clone(),
                left_rows,
                right_rows,
                join_type: "inner".to_string(),
            });
        }

        if let Some(predicate) = select.where_clause.clone() {
            // A real, data-driven estimate only when the filter is a
            // single <column> <op> <literal> comparison (see
            // row_codec::split_comparison) against a table nothing has
            // been JOINed to yet (a post-join filter's columns live in a
            // merged, qualified schema this planner doesn't resolve) and
            // a distribution is on record for that exact column. Every
            // other shape keeps the fixed default it always used.
            let real_estimate = if select.joins.is_empty() {
                row_codec::split_comparison(&predicate).and_then(|(left, op, right)| {
                    let dist = stats.get(&(select.from.clone(), left))?;
                    let value = literal_for_distribution(&right, dist)?;
                    dist.estimate_row_count(&op, &value, rows)
                })
            } else {
                None
            };
            rows = real_estimate.unwrap_or_else(|| (rows as f64 * Self::DEFAULT_FILTER_SELECTIVITY) as usize);
            nodes.push(LogicalPlanNode::Filter { predicate, rows });
        }

        // Treat this as an aggregate query when every projected column is
        // either a recognized aggregate call or (with GROUP BY) one of the
        // grouping columns -- anything else (a plain column that's neither)
        // is left as a plain, if not fully correct, row-returning plan
        // instead of pretending to aggregate.
        if !select.columns.is_empty()
            && select
                .columns
                .iter()
                .all(|c| aggregate::parse_aggregate(c).is_some() || select.group_by.contains(c))
        {
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
                rows,
            });
        }

        let estimated_cost = self.cost_model.estimate_total_cost(&Self::to_operator_costs(&nodes));

        PhysicalPlan {
            nodes,
            estimated_cost,
            estimated_rows: rows,
        }
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
        let plan = planner.plan("SELECT * FROM table1");
        assert!(!plan.nodes.is_empty());
    }

    #[test]
    fn test_plan_reflects_real_table_name() {
        let planner = QueryPlanner::new();
        let plan = planner.plan("SELECT * FROM orders");
        match &plan.nodes[0] {
            LogicalPlanNode::Scan { table_name, .. } => assert_eq!(table_name, "orders"),
            other => panic!("expected a Scan node, got {other:?}"),
        }
    }

    #[test]
    fn test_plan_emits_filter_node_for_where_clause() {
        let planner = QueryPlanner::new();
        let plan = planner.plan("SELECT * FROM users WHERE age > 18");
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
        let plan = planner.plan("SELECT * FROM users JOIN orders ON users.id = orders.user_id");
        assert!(plan
            .nodes
            .iter()
            .any(|n| matches!(n, LogicalPlanNode::Join { .. })));
    }

    #[test]
    fn test_plan_non_select_falls_back_to_empty_plan() {
        let planner = QueryPlanner::new();
        let plan = planner.plan("DELETE FROM users WHERE id = 1");
        assert_eq!(plan.estimated_rows, 0);
    }

    #[test]
    fn test_plan_emits_aggregate_node_when_all_columns_are_aggregates() {
        let planner = QueryPlanner::new();
        let plan = planner.plan("SELECT COUNT(*), AVG(age) FROM users");
        assert!(plan
            .nodes
            .iter()
            .any(|n| matches!(n, LogicalPlanNode::Aggregate { .. })));
        assert_eq!(plan.estimated_rows, 1);
    }

    #[test]
    fn test_plan_does_not_emit_aggregate_for_mixed_columns() {
        let planner = QueryPlanner::new();
        // "name" alongside COUNT(*) with no GROUP BY at all is invalid --
        // must not be misdetected as a pure aggregate query.
        let plan = planner.plan("SELECT name, COUNT(*) FROM users");
        assert!(!plan
            .nodes
            .iter()
            .any(|n| matches!(n, LogicalPlanNode::Aggregate { .. })));
    }

    #[test]
    fn test_plan_emits_aggregate_node_for_group_by() {
        let planner = QueryPlanner::new();
        let plan = planner.plan("SELECT user_id, COUNT(*) FROM orders GROUP BY user_id");
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
        let plan = planner.plan("SELECT name, COUNT(*) FROM orders GROUP BY user_id");
        assert!(!plan
            .nodes
            .iter()
            .any(|n| matches!(n, LogicalPlanNode::Aggregate { .. })));
    }

    #[test]
    fn test_different_queries_yield_different_costs() {
        let planner = QueryPlanner::new();
        let scan_only = planner.plan("SELECT * FROM users");
        let filtered = planner.plan("SELECT * FROM users WHERE age > 18");
        // A different query shape should actually change the estimate now,
        // instead of every query getting the same hardcoded 100.0/1000.
        assert_ne!(scan_only.estimated_cost, filtered.estimated_cost);
        assert_ne!(scan_only.estimated_rows, filtered.estimated_rows);
    }
}
