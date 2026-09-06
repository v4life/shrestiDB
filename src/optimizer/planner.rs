//! Query planner with learned optimization
//!
//! Generates query plans by parsing the query and consulting the cost model
//! and join orderer already held on this planner (previously constructed
//! but never used — `plan()` ignored its input and returned a hardcoded
//! single-table scan regardless of what was asked). Row-count estimates for
//! a bare scan still use a fixed assumed table size: there's no catalog or
//! table-statistics wiring into the planner yet, so there's no real number
//! to use instead. The learned cardinality estimator similarly isn't
//! consulted here yet — it expects structured `QueryPredicate`s, and a
//! WHERE clause is currently just a flattened string (see `sql::parser`),
//! so there's no honest way to build one without fabricating fake
//! structured input.

use crate::optimizer::cardinality::LearnedCardinalityEstimator;
use crate::optimizer::cost_model::{CostModel, OperatorCost, OperatorType};
use crate::optimizer::join_reorder::JoinOrderer;
use crate::sql::parser::{SQLParser, SQLStatement};
use serde::{Deserialize, Serialize};

/// Logical query plan node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogicalPlanNode {
    Scan {
        table_id: u32,
        table_name: String,
        rows: usize,
    },
    Filter {
        predicate: String,
        rows: usize,
    },
    Join {
        left_rows: usize,
        right_rows: usize,
        join_type: String,
    },
    Aggregate {
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
    pub cardinality_estimator: LearnedCardinalityEstimator,
    pub cost_model: CostModel,
    pub join_orderer: JoinOrderer,
}

impl QueryPlanner {
    pub fn new() -> Self {
        QueryPlanner {
            cardinality_estimator: LearnedCardinalityEstimator::new(10),
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

    /// Generate a query plan for `query`.
    pub fn plan(&self, query: &str) -> PhysicalPlan {
        let select = match SQLParser::parse(query) {
            Ok(SQLStatement::Select(select)) => select,
            // Not a SELECT, or failed to parse: nothing to build a read
            // plan for yet (writes and DDL don't have a plan shape here).
            _ => {
                return PhysicalPlan {
                    nodes: vec![LogicalPlanNode::Scan {
                        table_id: 0,
                        table_name: String::new(),
                        rows: 0,
                    }],
                    estimated_cost: 0.0,
                    estimated_rows: 0,
                };
            }
        };

        let mut rows = Self::ASSUMED_TABLE_ROWS;
        let mut nodes = vec![LogicalPlanNode::Scan {
            table_id: 0,
            table_name: select.from.clone(),
            rows,
        }];

        for _joined_table in &select.join_tables {
            let (left_rows, right_rows) = (rows, Self::ASSUMED_TABLE_ROWS);
            rows = self.join_orderer.estimate_join_cardinality(
                left_rows,
                right_rows,
                Self::DEFAULT_JOIN_SELECTIVITY,
            );
            nodes.push(LogicalPlanNode::Join {
                left_rows,
                right_rows,
                join_type: "inner".to_string(),
            });
        }

        if let Some(predicate) = select.where_clause.clone() {
            rows = (rows as f64 * Self::DEFAULT_FILTER_SELECTIVITY) as usize;
            nodes.push(LogicalPlanNode::Filter { predicate, rows });
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
                    LogicalPlanNode::Aggregate { rows } => (OperatorType::Aggregate, *rows),
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
