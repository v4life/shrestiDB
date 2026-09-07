//! Secondary indexes on non-primary-key columns.
//!
//! Unlike the learned (PGM) index over row ids in `execution::mvcc_store`
//! — which works because a row id *is* its primary-key value, a
//! one-to-one mapping the same PGM structure that predicts array
//! positions can serve directly — a secondary index on an arbitrary
//! column is fundamentally a different shape: many rows can share the
//! same value, so it needs a real value -> row-ids multimap, not a
//! position predictor. `DynamicPGMIndex` doesn't fit that shape (its
//! `range_search` returns the keys it was given, not arbitrary payloads
//! attached to them), so this is a plain sorted `BTreeMap`, not a learned
//! structure. That's a deliberate, honest scope boundary: this is a real
//! secondary index, just not one this codebase currently has a learned
//! version of.
//!
//! Like the PK index, this only ever grows — deleting a row never removes
//! its id from here — so `equals`/`range`'s results are candidates the
//! caller must still confirm against the real version chain, exactly the
//! same contract `MVCCTable::index_range` already has.

use std::collections::BTreeMap;
use std::ops::Bound;

use crate::execution::operators::Value;

/// A `Value` wrapper with a total order, so it can key a `BTreeMap`.
/// `Value` itself only has `PartialEq`/no `Ord` (floats have no total
/// order via the standard trait, because of NaN). Comparisons across
/// different `Value` variants shouldn't occur in practice — every value
/// indexed for one column comes from that column's single declared type
/// (see `row_codec::parse_value`) — but still need to resolve to *some*
/// total order for `BTreeMap`'s sake, so mismatched variants fall back to
/// a fixed, arbitrary rank ordering.
#[derive(Debug, Clone, PartialEq)]
struct IndexKey(Value);

impl Eq for IndexKey {}

impl PartialOrd for IndexKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for IndexKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        match (&self.0, &other.0) {
            (Value::Integer(a), Value::Integer(b)) => a.cmp(b),
            (Value::Float(a), Value::Float(b)) => a.total_cmp(b),
            (Value::String(a), Value::String(b)) => a.cmp(b),
            (Value::Boolean(a), Value::Boolean(b)) => a.cmp(b),
            (Value::Null, Value::Null) => Ordering::Equal,
            (a, b) => variant_rank(a).cmp(&variant_rank(b)),
        }
    }
}

fn variant_rank(v: &Value) -> u8 {
    match v {
        Value::Null => 0,
        Value::Boolean(_) => 1,
        Value::Integer(_) => 2,
        Value::Float(_) => 3,
        Value::String(_) => 4,
    }
}

fn map_bound(b: Bound<Value>) -> Bound<IndexKey> {
    match b {
        Bound::Included(v) => Bound::Included(IndexKey(v)),
        Bound::Excluded(v) => Bound::Excluded(IndexKey(v)),
        Bound::Unbounded => Bound::Unbounded,
    }
}

/// A value -> row-ids multimap for one (table, column), maintained
/// incrementally as rows are inserted (see `QueryExecutor::execute_insert`).
#[derive(Debug, Default)]
pub struct SecondaryIndex {
    entries: BTreeMap<IndexKey, Vec<u64>>,
}

impl SecondaryIndex {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, value: Value, row_id: u64) {
        self.entries.entry(IndexKey(value)).or_default().push(row_id);
    }

    /// Row ids whose indexed value equals `value` exactly.
    pub fn equals(&self, value: &Value) -> Vec<u64> {
        self.entries.get(&IndexKey(value.clone())).cloned().unwrap_or_default()
    }

    /// Row ids whose indexed value falls within `[lower, upper]` bounds
    /// (each `Bound::Unbounded`, `Included`, or `Excluded` — see
    /// `std::ops::Bound`).
    pub fn range(&self, lower: Bound<Value>, upper: Bound<Value>) -> Vec<u64> {
        self.entries
            .range((map_bound(lower), map_bound(upper)))
            .flat_map(|(_, ids)| ids.iter().copied())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_equals_finds_matching_rows() {
        let mut index = SecondaryIndex::new();
        index.insert(Value::String("Bob".to_string()), 1);
        index.insert(Value::String("Alice".to_string()), 2);
        index.insert(Value::String("Bob".to_string()), 3); // duplicate value, different row

        let mut ids = index.equals(&Value::String("Bob".to_string()));
        ids.sort();
        assert_eq!(ids, vec![1, 3]);
        assert!(index.equals(&Value::String("Carol".to_string())).is_empty());
    }

    #[test]
    fn test_range_on_integers() {
        let mut index = SecondaryIndex::new();
        for (value, row_id) in [(10, 1), (20, 2), (30, 3), (40, 4)] {
            index.insert(Value::Integer(value), row_id);
        }

        let mut ids = index.range(Bound::Included(Value::Integer(20)), Bound::Included(Value::Integer(30)));
        ids.sort();
        assert_eq!(ids, vec![2, 3]);

        let mut ids = index.range(Bound::Excluded(Value::Integer(20)), Bound::Unbounded);
        ids.sort();
        assert_eq!(ids, vec![3, 4]);
    }

    #[test]
    fn test_empty_index_returns_empty() {
        let index = SecondaryIndex::new();
        assert!(index.equals(&Value::Integer(1)).is_empty());
        assert!(index.range(Bound::Unbounded, Bound::Unbounded).is_empty());
    }
}
