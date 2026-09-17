//! Real join reordering for query optimization
//!
//! `find_optimal_order` used to ignore the `cost_model` it held and just
//! `.sort()` table IDs — and was never actually called from anywhere but
//! its own tests, since nothing upstream ever produced real table IDs
//! either (`optimizer::planner`'s `Scan` node hardcoded `table_id: 0`).
//! This is the real version: given a `FROM` table and its `JOIN`
//! clauses, each carrying its real `ON` condition, it searches for the
//! order with the lowest estimated cost among every order that's
//! *provably resolvable*: at the point each join would run, every
//! qualifier (alias or table name) its condition references must
//! already be present in the accumulated left side.
//!
//! That check isn't a nice-to-have. This engine's executor is a
//! left-deep chain with no expression resolver of its own — it walks
//! `PhysicalPlan::nodes` in whatever order they appear and resolves each
//! `Join` node's condition against whatever's already been accumulated
//! (see `execution::executor::QueryExecutor::execute`'s `Join` arm) — so
//! a reordering that violated this would make a working query fail to
//! resolve its own condition, or (for a self-join sharing a bare,
//! unqualified column name) silently resolve it against the wrong side.
//! Source order is always one valid ordering — a query wouldn't
//! otherwise execute correctly today — so there's always at least one
//! candidate, and reordering never turns a working query into a broken
//! one.
//!
//! A join whose condition can't be resolved this way at all — most
//! commonly, an unqualified column reference — makes the whole query's
//! order fixed at source order. Partial reordering, applied only to the
//! joins this planner does understand while silently leaving others
//! pinned in place, would be a subtler and harder-to-predict result than
//! just declining to reorder anything for that query.

use crate::execution::row_codec;
use crate::optimizer::cost_model::{CostModel, OperatorCost, OperatorType};
use crate::sql::parser::JoinClause;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// One join, chosen and costed by `find_optimal_order`: which original
/// join (`join_index`, an index into the `joins` slice passed to it)
/// goes at this position in the chain, and the row-count estimates that
/// position implies.
#[derive(Debug, Clone, PartialEq)]
pub struct PlannedJoin {
    pub join_index: usize,
    pub left_rows: usize,
    pub right_rows: usize,
    pub output_rows: usize,
}

/// How a join's `ON` condition relates to the qualifiers that could be
/// known at the point it might run — resolved once up front rather than
/// re-parsed for every candidate ordering the search considers.
#[derive(Debug, Clone)]
enum JoinShape {
    /// No condition at all (`CROSS JOIN`, or a `USING`/`NATURAL` join —
    /// see `sql::parser::JoinClause`'s docs, which aren't specially
    /// resolved either): no ordering constraint, and no real selectivity
    /// to estimate from — the output is the full cross product.
    Cross,
    /// A comparison between one column on this join's own table
    /// (`own_column`) and one column on a table qualified
    /// `existing_qualifier` (`existing_column`) — that qualifier must
    /// already be known before this join can run.
    Compare { op: String, own_column: String, existing_qualifier: String, existing_column: String },
    /// The condition doesn't reduce to two qualified columns (most
    /// commonly, one side is a bare, unqualified column name). This
    /// planner can't safely tell which side it belongs to, so reordering
    /// is declined for the whole query rather than guessed at.
    Unresolvable,
}

fn own_qualifier(join: &JoinClause) -> String {
    join.alias.clone().unwrap_or_else(|| join.table.clone())
}

fn resolve_shape(own: &str, condition: &Option<String>) -> JoinShape {
    let Some(cond) = condition else { return JoinShape::Cross };
    let Some((left, op, right)) = row_codec::split_comparison(cond) else {
        return JoinShape::Unresolvable;
    };
    if !left.contains('.') || !right.contains('.') {
        return JoinShape::Unresolvable;
    }
    let split = |tok: &str| -> (String, String) {
        let mut parts = tok.splitn(2, '.');
        let q = parts.next().unwrap_or_default().to_string();
        let c = parts.next().unwrap_or_default().to_string();
        (q, c)
    };
    let (lq, lc) = split(&left);
    let (rq, rc) = split(&right);
    match (lq == own, rq == own) {
        (true, false) => JoinShape::Compare { op, own_column: lc, existing_qualifier: rq, existing_column: rc },
        (false, true) => JoinShape::Compare { op, own_column: rc, existing_qualifier: lq, existing_column: lc },
        // Neither side (or, for a malformed self-join, both) names this
        // join's own qualifier -- can't tell which operand is "ours".
        _ => JoinShape::Unresolvable,
    }
}

/// Everything a single `find_optimal_order` call's search needs, bundled
/// to keep the recursive search's own signature manageable.
struct SearchContext<'a> {
    from_rows: usize,
    joins: &'a [JoinClause],
    shapes: &'a [JoinShape],
    row_count_of: &'a dyn Fn(&str) -> usize,
    distinct_count_of: &'a dyn Fn(&str, &str) -> Option<usize>,
    default_join_selectivity: f64,
}

/// Join reorderer with a real (if fixed-formula, not adaptive — see
/// `CostModel`'s own docs) cost model, now actually consulted by a real
/// decision instead of sitting unused.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoinOrderer {
    pub cost_model: CostModel,
}

impl JoinOrderer {
    pub fn new() -> Self {
        JoinOrderer { cost_model: CostModel::new() }
    }

    /// Fixed-selectivity join-cardinality estimate, applied uniformly —
    /// used whenever a real, stats-driven one
    /// (`estimate_join_cardinality_with_stats`) isn't available.
    pub fn estimate_join_cardinality(&self, left_rows: usize, right_rows: usize, join_selectivity: f64) -> usize {
        (left_rows as f64 * right_rows as f64 * join_selectivity) as usize
    }

    /// A real equi-join cardinality estimate when both sides' distinct
    /// join-column counts are known: the same containment/uniformity
    /// assumption most production optimizers make absent a
    /// most-common-values list — each distinct value on the smaller
    /// domain is assumed to match roughly `rows / distinct` rows on the
    /// other side, giving `left_rows * right_rows / max(left_distinct,
    /// right_distinct)`. Falls back to `estimate_join_cardinality` with
    /// `default_selectivity` when either side's distinct count is
    /// unavailable (no `ANALYZE` yet for that column).
    pub fn estimate_join_cardinality_with_stats(
        &self,
        left_rows: usize,
        right_rows: usize,
        left_distinct: Option<usize>,
        right_distinct: Option<usize>,
        default_selectivity: f64,
    ) -> usize {
        match (left_distinct, right_distinct) {
            (Some(l), Some(r)) if l > 0 && r > 0 => {
                ((left_rows as f64 * right_rows as f64) / (l.max(r) as f64)) as usize
            }
            _ => self.estimate_join_cardinality(left_rows, right_rows, default_selectivity),
        }
    }

    /// Find the lowest-cost, provably-resolvable order to run `joins` in
    /// against `from_qualifier` (the `FROM` table's alias or name),
    /// starting from `from_rows` rows already accumulated. Resolving a
    /// qualifier back to a real table name, and looking up any real
    /// per-column distinct-value count, is `row_count_of`/
    /// `distinct_count_of`'s job — only the caller (`optimizer::planner`)
    /// holds the `stats` map and the qualifier-to-table mapping for this
    /// specific query.
    ///
    /// Bounded to at most 8 joins searched exhaustively (a handful of
    /// tables, the realistic case for this engine) — beyond that, or
    /// whenever any join's condition can't be safely resolved by
    /// qualifier, this returns source order costed as-is rather than
    /// searching. That's always correct, just not necessarily cheapest;
    /// a real DP/branch-and-bound search for larger join counts is a
    /// diminishing-returns build for a query shape this engine doesn't
    /// realistically see.
    pub fn find_optimal_order(
        &self,
        from_qualifier: &str,
        from_rows: usize,
        joins: &[JoinClause],
        row_count_of: impl Fn(&str) -> usize,
        distinct_count_of: impl Fn(&str, &str) -> Option<usize>,
        default_join_selectivity: f64,
    ) -> Vec<PlannedJoin> {
        if joins.is_empty() {
            return Vec::new();
        }

        let shapes: Vec<JoinShape> = joins.iter().map(|j| resolve_shape(&own_qualifier(j), &j.condition)).collect();
        let source_order: Vec<usize> = (0..joins.len()).collect();

        let ctx = SearchContext {
            from_rows,
            joins,
            shapes: &shapes,
            row_count_of: &row_count_of,
            distinct_count_of: &distinct_count_of,
            default_join_selectivity,
        };

        let searchable = joins.len() <= 8 && !shapes.iter().any(|s| matches!(s, JoinShape::Unresolvable));
        let chosen = if searchable {
            let mut known: HashSet<String> = HashSet::new();
            known.insert(from_qualifier.to_string());
            let mut used = vec![false; joins.len()];
            let mut order = Vec::with_capacity(joins.len());
            let mut best: Option<(f64, Vec<usize>)> = None;
            self.search_step(&ctx, &mut known, &mut used, &mut order, &mut best);
            best.map(|(_, order)| order).unwrap_or(source_order)
        } else {
            source_order
        };

        self.replay(&ctx, &chosen).0
    }

    /// Depth-first search over valid orderings of join indices: a join
    /// is only tried once every qualifier its condition depends on
    /// (`JoinShape::Compare::existing_qualifier`) is already in `known`.
    /// Each complete ordering is costed via `replay` and kept if it beats
    /// the best found so far — strictly better, not merely equal, so a
    /// tie keeps whichever complete ordering the search reaches first
    /// (source order, tried first at every branch, when nothing else
    /// actually costs less).
    fn search_step(
        &self,
        ctx: &SearchContext,
        known: &mut HashSet<String>,
        used: &mut [bool],
        order: &mut Vec<usize>,
        best: &mut Option<(f64, Vec<usize>)>,
    ) {
        if order.len() == ctx.joins.len() {
            let (_, cost) = self.replay(ctx, order);
            if best.as_ref().map_or(true, |(best_cost, _)| cost < *best_cost) {
                *best = Some((cost, order.clone()));
            }
            return;
        }

        for i in 0..ctx.joins.len() {
            if used[i] {
                continue;
            }
            let ready = match &ctx.shapes[i] {
                JoinShape::Cross => true,
                JoinShape::Compare { existing_qualifier, .. } => known.contains(existing_qualifier),
                JoinShape::Unresolvable => false, // find_optimal_order already excludes this path
            };
            if !ready {
                continue;
            }

            used[i] = true;
            order.push(i);
            let q = own_qualifier(&ctx.joins[i]);
            let newly_known = known.insert(q.clone());
            self.search_step(ctx, known, used, order, best);
            if newly_known {
                known.remove(&q);
            }
            order.pop();
            used[i] = false;
        }
    }

    /// Replay a fixed `order` (indices into `ctx.joins`) to compute both
    /// the resulting `PlannedJoin` chain and its total estimated cost —
    /// used both to score a candidate ordering during search and, once,
    /// to produce `find_optimal_order`'s real return value for whichever
    /// ordering won (or for source order, when the search was skipped).
    fn replay(&self, ctx: &SearchContext, order: &[usize]) -> (Vec<PlannedJoin>, f64) {
        let mut current_rows = ctx.from_rows;
        let mut planned = Vec::with_capacity(order.len());
        let mut total_cost = 0.0;

        for &idx in order {
            let join = &ctx.joins[idx];
            let right_qualifier = own_qualifier(join);
            let right_rows = (ctx.row_count_of)(&right_qualifier);

            let output_rows = match &ctx.shapes[idx] {
                JoinShape::Cross => current_rows.saturating_mul(right_rows),
                JoinShape::Compare { op, own_column, existing_qualifier, existing_column } if op == "=" => {
                    let left_distinct = (ctx.distinct_count_of)(existing_qualifier, existing_column);
                    let right_distinct = (ctx.distinct_count_of)(&right_qualifier, own_column);
                    self.estimate_join_cardinality_with_stats(
                        current_rows,
                        right_rows,
                        left_distinct,
                        right_distinct,
                        ctx.default_join_selectivity,
                    )
                }
                // A non-equality two-column comparison: the
                // distinct-count formula above only holds for equi-joins,
                // so fall back to the fixed default like before.
                JoinShape::Compare { .. } | JoinShape::Unresolvable => {
                    self.estimate_join_cardinality(current_rows, right_rows, ctx.default_join_selectivity)
                }
            };

            let op_cost = OperatorCost::new(OperatorType::HashJoin, current_rows.max(right_rows), output_rows, 1.0);
            total_cost += self.cost_model.estimate_cost(&op_cost);
            planned.push(PlannedJoin { join_index: idx, left_rows: current_rows, right_rows, output_rows });
            current_rows = output_rows;
        }

        (planned, total_cost)
    }
}

impl Default for JoinOrderer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn join(table: &str, alias: Option<&str>, condition: Option<&str>) -> JoinClause {
        JoinClause { table: table.to_string(), alias: alias.map(str::to_string), condition: condition.map(str::to_string) }
    }

    #[test]
    fn test_join_cardinality_estimation() {
        let orderer = JoinOrderer::new();
        let card = orderer.estimate_join_cardinality(100, 50, 0.1);
        assert_eq!(card, 500);
    }

    #[test]
    fn test_stats_based_cardinality_uses_distinct_counts_not_fixed_selectivity() {
        let orderer = JoinOrderer::new();
        // 1000 rows on each side; real formula is
        // left_rows*right_rows/max(left_distinct,right_distinct) =
        // 1_000_000/500 = 2000 -- nowhere near the fixed-selectivity
        // guess of 1000*1000*0.1 = 100_000, proving the real path (not
        // the fallback) is what actually ran.
        let real = orderer.estimate_join_cardinality_with_stats(1000, 1000, Some(500), Some(10), 0.1);
        assert_eq!(real, 2_000);
        let fixed = orderer.estimate_join_cardinality(1000, 1000, 0.1);
        assert_eq!(fixed, 100_000);
        assert_ne!(real, fixed);
    }

    #[test]
    fn test_stats_based_cardinality_falls_back_without_distinct_counts() {
        let orderer = JoinOrderer::new();
        let est = orderer.estimate_join_cardinality_with_stats(100, 50, None, Some(10), 0.1);
        assert_eq!(est, orderer.estimate_join_cardinality(100, 50, 0.1));
    }

    #[test]
    fn test_find_optimal_order_single_join_keeps_it() {
        let orderer = JoinOrderer::new();
        let joins = vec![join("orders", Some("o"), Some("u.id = o.user_id"))];
        let planned = orderer.find_optimal_order("u", 100, &joins, |_| 1000, |_, _| None, 0.1);
        assert_eq!(planned.len(), 1);
        assert_eq!(planned[0].join_index, 0);
    }

    #[test]
    fn test_find_optimal_order_picks_more_selective_join_first() {
        let orderer = JoinOrderer::new();
        // Two joins off the same root table `u`: joining `big` first
        // multiplies u's 100 rows by big's 100_000 nearly unfiltered
        // (distinct=2, barely selective) before ever joining the cheap,
        // highly selective `small` (distinct=100_000, i.e. nearly 1:1) --
        // a real cost-aware search should prefer `small` first.
        let joins = vec![
            join("big", Some("b"), Some("u.id = b.u_id")),
            join("small", Some("s"), Some("u.id = s.u_id")),
        ];
        let row_count_of = |q: &str| match q {
            "u" => 100,
            "b" => 100_000,
            "s" => 100,
            _ => 1000,
        };
        let distinct_count_of = |q: &str, _c: &str| match q {
            "u" => Some(100),
            "b" => Some(2),      // low cardinality on the join column: huge fan-out
            "s" => Some(100),    // near 1:1: highly selective
            _ => None,
        };
        let planned = orderer.find_optimal_order("u", 100, &joins, row_count_of, distinct_count_of, 0.1);
        assert_eq!(planned.len(), 2);
        // join_index 1 is `small` -- should be scheduled first.
        assert_eq!(planned[0].join_index, 1, "expected the highly selective join to be scheduled first");
        assert_eq!(planned[1].join_index, 0);
    }

    #[test]
    fn test_find_optimal_order_respects_dependency_chain() {
        let orderer = JoinOrderer::new();
        // `p` depends on `o` (its condition references o.id), which in
        // turn depends on `u`. Even though nothing here makes `o` look
        // artificially cheap to schedule first, `p` can never be
        // scheduled before `o` -- the search must never propose an
        // order that puts p ahead of o.
        let joins = vec![
            join("payments", Some("p"), Some("o.id = p.order_id")),
            join("orders", Some("o"), Some("u.id = o.user_id")),
        ];
        let planned = orderer.find_optimal_order("u", 100, &joins, |_| 500, |_, _| None, 0.1);
        let position_of = |join_index: usize| planned.iter().position(|pj| pj.join_index == join_index).unwrap();
        assert!(position_of(1) < position_of(0), "orders (index 1) must be scheduled before payments (index 0)");
    }

    #[test]
    fn test_find_optimal_order_falls_back_to_source_order_on_unqualified_condition() {
        let orderer = JoinOrderer::new();
        // An unqualified column on one side -- can't safely tell which
        // table it belongs to, so the whole query's order stays fixed.
        let joins = vec![
            join("orders", Some("o"), Some("id = o.user_id")),
            join("payments", Some("p"), Some("o.id = p.order_id")),
        ];
        let planned = orderer.find_optimal_order("u", 100, &joins, |_| 500, |_, _| None, 0.1);
        assert_eq!(planned.iter().map(|pj| pj.join_index).collect::<Vec<_>>(), vec![0, 1]);
    }

    #[test]
    fn test_find_optimal_order_cross_join_has_no_dependency_and_full_cardinality() {
        let orderer = JoinOrderer::new();
        let joins = vec![join("logs", Some("l"), None)];
        let planned = orderer.find_optimal_order("u", 10, &joins, |_| 5, |_, _| None, 0.1);
        assert_eq!(planned[0].output_rows, 50); // full cross product, not selectivity-reduced
    }

    #[test]
    fn test_find_optimal_order_empty_joins_returns_empty() {
        let orderer = JoinOrderer::new();
        let planned = orderer.find_optimal_order("u", 100, &[], |_| 1000, |_, _| None, 0.1);
        assert!(planned.is_empty());
    }
}
