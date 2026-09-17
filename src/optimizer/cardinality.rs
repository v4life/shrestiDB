//! Real, data-driven cardinality estimation
//!
//! Replaces two previously-decorative modules: this file's old
//! `LearnedCardinalityEstimator` — a "neural net" whose weights were
//! hardcoded to `0.1` at construction, whose `train()` was a literal
//! no-op (`// In production, this would do backpropagation` — it never
//! did), and whose `estimate_selectivity` never actually read the real
//! `min_value`/`max_value`/`distinct_values`/`null_count` statistics it
//! was given, only the *length* of the stats list, as padding — and
//! `optimizer::statistics`'s `StatisticsCollector`, a plain registry
//! with no scanning logic behind it, used nowhere outside its own test.
//! Neither was ever wired into a real decision; replacing one fixed
//! constant with another, data-blind one dressed as ML would have been
//! worse than what was there before, not better.
//!
//! `ColumnDistribution` is built from an actual scan of a column's
//! committed values (`QueryExecutor::execute_analyze`, driven by
//! `ANALYZE <table>`), not synthesized. For a numeric column it reuses
//! this codebase's own proven learned-index technique — `PGMIndex`,
//! already shown to beat a B-tree by up to 36.8x on lookups — as a
//! piecewise-linear model of the column's empirical CDF:
//! `PGMIndex::predicted_rank(x)` estimates how many rows have a value
//! `<= x`, whether or not `x` was ever actually in the data, which is
//! exactly what a range predicate's selectivity needs. For a categorical
//! (string/boolean) column, where a CDF isn't meaningful, only a real
//! distinct-value count is kept, giving equality selectivity
//! (`1 / distinct_count`) — the same assumption most production query
//! optimizers make absent a most-common-values list.

use crate::execution::operators::Value;
use crate::index::pgm::PGMIndex;
use serde::{Deserialize, Serialize};

/// A single column's real, observed value distribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ColumnDistribution {
    /// Integer/Float/Timestamp columns: a real PGM model over the
    /// column's actual sorted values, usable for both equality and
    /// range selectivity.
    Numeric { model: PGMIndex, total_rows: usize, distinct_count: usize },
    /// String/Boolean columns: no CDF (a `PGMIndex` operates on `f64`
    /// keys), so only a real distinct-value count — equality selectivity
    /// only; range predicates on a categorical column fall back to the
    /// caller's default, same as an unavailable distribution would.
    Categorical { total_rows: usize, distinct_count: usize },
}

impl ColumnDistribution {
    /// Build from a numeric column's actual values (any order; sorted
    /// internally). `error_bound` is the same PGM segment-fitting
    /// tolerance `PGMIndex::build` takes elsewhere in this codebase.
    pub fn build_numeric(mut values: Vec<f64>, error_bound: usize) -> ColumnDistribution {
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let total_rows = values.len();
        let distinct_count = count_distinct(&values).max(1);
        ColumnDistribution::Numeric { model: PGMIndex::build(values, error_bound), total_rows, distinct_count }
    }

    /// Build from a categorical column's actual values.
    pub fn build_categorical(mut values: Vec<String>) -> ColumnDistribution {
        let total_rows = values.len();
        values.sort();
        values.dedup();
        ColumnDistribution::Categorical { total_rows, distinct_count: values.len().max(1) }
    }

    /// Selectivity of `column <op> value` — `op` one of `=`, `!=`/`<>`,
    /// `<`, `<=`, `>`, `>=`. `None` when this distribution can't answer
    /// it at all (an unrecognized operator; a range operator against a
    /// `Categorical` distribution; a `value` whose type doesn't match
    /// this distribution's), matching this codebase's existing
    /// "can't evaluate -> caller falls back to a default" convention
    /// (see `row_codec::evaluate_predicate`'s docs) rather than a
    /// fabricated number.
    ///
    /// `<=` and `<` (and `>=`/`>`) are *not* treated as interchangeable:
    /// an earlier version of this method did, on the reasoning that one
    /// exact-match boundary is negligible next to the estimate's own
    /// error margin — wrong for exactly the skewed, few-distinct-values
    /// column this estimator exists to handle well (e.g. 950 rows at one
    /// value, 50 at another): querying `>= <the common value>` landed
    /// squarely on that value's entire duplicate run, and conflating
    /// `>=`/`>` there was off by the whole run, not a rounding error.
    /// `<=`/`>` use the learned model's `predicted_rank` directly; `<`/`>=`
    /// need "count strictly less than `v`", which `predicted_rank` alone
    /// can't distinguish from "count `<= v`" at a duplicate-heavy boundary
    /// — a first attempt tried `range_search(v, v).len()` for that
    /// correction and it was *also* wrong: `range_search` starts from
    /// wherever `search`'s bounded lookup happens to land inside a
    /// duplicate run and only scans forward, silently undercounting
    /// occurrences before that point. Since `model.keys` already holds
    /// every real value PGM's own point-lookup path (`bounded_search`)
    /// needs anyway, a real binary search (`partition_point`, exact,
    /// `O(log n)`) is what actually gives a correct count here — used
    /// only for this one boundary correction, not as a replacement for
    /// the learned estimate everywhere else.
    pub fn estimate_selectivity(&self, op: &str, value: &Value) -> Option<f64> {
        match self {
            ColumnDistribution::Numeric { model, total_rows, distinct_count } => {
                if *total_rows == 0 {
                    return Some(1.0);
                }
                let v = match value {
                    Value::Integer(i) => *i as f64,
                    Value::Float(f) => *f,
                    _ => return None,
                };
                let n = *total_rows as f64;
                let rank_le = model.predicted_rank(v) as f64; // count of values <= v
                Some(match op {
                    "=" => (1.0 / *distinct_count as f64).clamp(0.0, 1.0),
                    "!=" | "<>" => (1.0 - 1.0 / *distinct_count as f64).clamp(0.0, 1.0),
                    "<=" => (rank_le / n).clamp(0.0, 1.0),
                    ">" => (1.0 - rank_le / n).clamp(0.0, 1.0),
                    "<" | ">=" => {
                        let rank_lt = model.keys.partition_point(|&x| x < v) as f64;
                        if op == "<" {
                            (rank_lt / n).clamp(0.0, 1.0)
                        } else {
                            (1.0 - rank_lt / n).clamp(0.0, 1.0)
                        }
                    }
                    _ => return None,
                })
            }
            ColumnDistribution::Categorical { distinct_count, .. } => match op {
                "=" => Some((1.0 / *distinct_count as f64).clamp(0.0, 1.0)),
                "!=" | "<>" => Some((1.0 - 1.0 / *distinct_count as f64).clamp(0.0, 1.0)),
                _ => None,
            },
        }
    }

    /// `estimate_selectivity` applied to `current_total_rows` — deliberately
    /// a caller-supplied count, not this distribution's own `total_rows`
    /// (an `ANALYZE`-time snapshot that may be stale by now): the
    /// *selectivity ratio* comes from the analyzed distribution, but the
    /// row count it's applied to should reflect the table's current
    /// size wherever the caller already tracks that.
    pub fn estimate_row_count(&self, op: &str, value: &Value, current_total_rows: usize) -> Option<usize> {
        self.estimate_selectivity(op, value).map(|s| (current_total_rows as f64 * s) as usize)
    }

    /// The table's row count as of `ANALYZE` time — an `ANALYZE`-time
    /// snapshot, same staleness caveat as everywhere else this type
    /// carries `total_rows`. Useful as a real base row count for a bare
    /// scan's estimate (any analyzed column's `total_rows` is the whole
    /// table's row count at that `ANALYZE`, not just that column's) —
    /// see `optimizer::planner::plan_select`.
    pub fn total_rows(&self) -> usize {
        match self {
            ColumnDistribution::Numeric { total_rows, .. } => *total_rows,
            ColumnDistribution::Categorical { total_rows, .. } => *total_rows,
        }
    }

    /// The column's real, observed distinct-value count as of `ANALYZE`
    /// time — the same staleness caveat as `total_rows`. Used by
    /// `join_reorder::JoinOrderer` for a real equi-join cardinality
    /// estimate instead of a fixed selectivity.
    pub fn distinct_count(&self) -> usize {
        match self {
            ColumnDistribution::Numeric { distinct_count, .. } => *distinct_count,
            ColumnDistribution::Categorical { distinct_count, .. } => *distinct_count,
        }
    }
}

fn count_distinct(sorted_values: &[f64]) -> usize {
    let mut count = 0;
    let mut last: Option<f64> = None;
    for &v in sorted_values {
        if last != Some(v) {
            count += 1;
            last = Some(v);
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numeric_equality_selectivity_uses_real_distinct_count() {
        // 4 distinct values, uniformly spread -- equality selectivity
        // should be close to 1/4, not a fixed, data-blind constant.
        let values: Vec<f64> = (0..1000).map(|i| (i % 4) as f64).collect();
        let dist = ColumnDistribution::build_numeric(values, 4);
        let selectivity = dist.estimate_selectivity("=", &Value::Float(2.0)).unwrap();
        assert!((selectivity - 0.25).abs() < 0.01, "selectivity {selectivity} should be close to 0.25");
    }

    #[test]
    fn test_numeric_range_selectivity_reflects_real_skew() {
        // 950 values at 0.0, 50 values at 1.0 -- a skewed distribution a
        // fixed 0.5-selectivity heuristic would badly misjudge.
        let mut values = vec![0.0; 950];
        values.extend(vec![1.0; 50]);
        let dist = ColumnDistribution::build_numeric(values, 4);
        let selectivity = dist.estimate_selectivity(">=", &Value::Float(1.0)).unwrap();
        assert!(
            (selectivity - 0.05).abs() < 0.02,
            "selectivity {selectivity} should be close to the real 0.05, not a fixed 0.5"
        );
    }

    #[test]
    fn test_categorical_equality_selectivity_uses_real_distinct_count() {
        let values: Vec<String> =
            std::iter::repeat("shipped".to_string()).take(950).chain(std::iter::repeat("cancelled".to_string()).take(50)).collect();
        let dist = ColumnDistribution::build_categorical(values);
        let rare = dist.estimate_selectivity("=", &Value::String("cancelled".to_string())).unwrap();
        // Real selectivity of the rare value is 50/1000 = 0.05, but this
        // estimator (like most production optimizers without a
        // most-common-values list) assumes uniformity across distinct
        // values: 1/2 = 0.5. Still a massive, real improvement over a
        // fixed 0.5-for-everything heuristic once there are more than a
        // couple of distinct values -- see the executor-level test for
        // the actually-skewed multi-value case.
        assert!((rare - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_categorical_range_operator_returns_none() {
        let dist = ColumnDistribution::build_categorical(vec!["a".to_string(), "b".to_string()]);
        assert!(dist.estimate_selectivity("<", &Value::String("a".to_string())).is_none());
    }

    #[test]
    fn test_unrecognized_operator_returns_none() {
        let dist = ColumnDistribution::build_numeric(vec![1.0, 2.0, 3.0], 1);
        assert!(dist.estimate_selectivity("LIKE", &Value::Float(1.0)).is_none());
    }

    #[test]
    fn test_estimate_row_count_uses_caller_supplied_total_not_analyze_time_snapshot() {
        let dist = ColumnDistribution::build_numeric(vec![1.0, 2.0, 3.0, 4.0], 1);
        // Analyzed at 4 rows, but the caller says the table now has 400 --
        // the row-count estimate should scale with the *current* count.
        let count = dist.estimate_row_count("=", &Value::Float(2.0), 400).unwrap();
        assert_eq!(count, 100); // 1/4 selectivity * 400 current rows
    }
}
