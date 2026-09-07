//! Multi-Version Concurrency Control (MVCC) store
//!
//! Each row has a chain of immutable versions tagged with [begin_ts, end_ts).
//! A version is visible to a transaction whose snapshot_ts falls inside that range.
//!
//! Write protocol (used by OLTPEngine):
//!   1. Buffer writes in the active transaction's write_set.
//!   2. On commit: get commit_ts from global counter, apply write_set atomically.
//!   3. On abort:  discard write_set (nothing touches the store).
//!
//! This means the store only ever sees committed data — no "dirty read" possible.

use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};

use crate::index::pgm::DynamicPGMIndex;

pub const TS_INFINITY: u64 = u64::MAX;

/// PGM error bound for a table's row-id index — how far off a predicted
/// position can be before the bounded search around it must widen.
const PK_INDEX_ERROR_BOUND: usize = 8;
/// How many buffered inserts a table's row-id index absorbs before
/// merging them into its base PGM segments (see `DynamicPGMIndex`).
const PK_INDEX_BUFFER_CAPACITY: usize = 64;

// ── Version ───────────────────────────────────────────────────────────────────

/// One immutable snapshot of a row.
#[derive(Debug, Clone)]
pub struct RowVersion {
    /// Serialised row bytes.
    pub data:     Vec<u8>,
    /// Commit timestamp of the transaction that created this version.
    pub begin_ts: u64,
    /// Commit timestamp of the transaction that deleted/replaced this version.
    /// `TS_INFINITY` means the version is still alive.
    pub end_ts:   u64,
}

impl RowVersion {
    /// Is this version visible to a snapshot taken at `snap_ts`?
    #[inline]
    pub fn visible_at(&self, snap_ts: u64) -> bool {
        self.begin_ts <= snap_ts && snap_ts < self.end_ts
    }
}

// ── Version chain ─────────────────────────────────────────────────────────────

#[derive(Debug, Default)]
struct VersionChain {
    /// Versions ordered newest-first so the common case (read latest) is O(1).
    versions: Vec<RowVersion>,
}

impl VersionChain {
    fn read(&self, snap_ts: u64) -> Option<Vec<u8>> {
        self.versions.iter().find(|v| v.visible_at(snap_ts)).map(|v| v.data.clone())
    }

    /// Append a new live version (INSERT / UPDATE new value).
    fn push_version(&mut self, begin_ts: u64, data: Vec<u8>) {
        self.versions.insert(0, RowVersion { data, begin_ts, end_ts: TS_INFINITY });
    }

    /// Expire the current live version (DELETE / UPDATE old value).
    fn expire_live(&mut self, end_ts: u64) {
        for v in &mut self.versions {
            if v.end_ts == TS_INFINITY {
                v.end_ts = end_ts;
                break;
            }
        }
    }

    /// Drop every version that ended at or before `horizon` — the oldest
    /// snapshot timestamp any currently active transaction could still be
    /// reading from. No snapshot at or after `horizon` can ever see a
    /// version whose `end_ts <= horizon` (see `RowVersion::visible_at`), so
    /// this is safe regardless of why the caller chose to sweep now.
    /// Returns the number of versions removed.
    fn prune(&mut self, horizon: u64) -> usize {
        let before = self.versions.len();
        self.versions.retain(|v| v.end_ts > horizon);
        before - self.versions.len()
    }
}

// ── Buffered write ops (pending inside a transaction) ─────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WriteOp {
    Insert { table_id: u64, row_id: u64, data: Vec<u8> },
    Update { table_id: u64, row_id: u64, data: Vec<u8> },
    Delete { table_id: u64, row_id: u64 },
}

impl WriteOp {
    pub fn table_row(&self) -> (u64, u64) {
        match self {
            WriteOp::Insert { table_id, row_id, .. } => (*table_id, *row_id),
            WriteOp::Update { table_id, row_id, .. } => (*table_id, *row_id),
            WriteOp::Delete { table_id, row_id }     => (*table_id, *row_id),
        }
    }
}

// ── MVCC table ────────────────────────────────────────────────────────────────

pub type RowId = u64;

/// One table's version-chain storage.
///
/// `pk_index` is a learned (PGM) index over this table's row ids — which
/// are always a row's primary-key value (see `QueryExecutor::execute_insert`,
/// the only place a row id is ever assigned), so this doubles as a PK
/// index without needing a separate structure. It only ever grows: a
/// deleted row's id is never removed from it, and `range_ids`'/`contains_id`'s
/// callers must treat what it returns as *candidates* to be confirmed
/// against the actual version chain (via `read`), not final answers — a
/// stale or not-yet-committed-in-this-view id is simply filtered out at
/// that point, same as a row a full scan would have also had to check.
#[derive(Default)]
pub struct MVCCTable {
    rows: RwLock<HashMap<RowId, VersionChain>>,
    pk_index: Mutex<Option<DynamicPGMIndex>>,
}

impl MVCCTable {
    /// Point-read at `snap_ts`.
    pub fn read(&self, row_id: RowId, snap_ts: u64) -> Option<Vec<u8>> {
        self.rows.read().get(&row_id)?.read(snap_ts)
    }

    /// Full-table scan: all rows visible at `snap_ts`.
    pub fn scan(&self, snap_ts: u64) -> Vec<(RowId, Vec<u8>)> {
        self.rows.read()
            .iter()
            .filter_map(|(id, chain)| chain.read(snap_ts).map(|d| (*id, d)))
            .collect()
    }

    /// Row ids in `[min, max]` per the learned index — see the struct docs
    /// for why these are candidates, not a final answer.
    pub fn index_range(&self, min: f64, max: f64) -> Vec<RowId> {
        match self.pk_index.lock().as_ref() {
            Some(index) => index.range_search(min, max).into_iter().map(|k| k as RowId).collect(),
            None => Vec::new(),
        }
    }

    fn index_insert(&self, row_id: RowId) {
        let mut guard = self.pk_index.lock();
        match guard.as_mut() {
            Some(index) => index.insert(row_id as f64),
            None => {
                *guard = Some(DynamicPGMIndex::new(
                    vec![row_id as f64],
                    PK_INDEX_ERROR_BOUND,
                    PK_INDEX_BUFFER_CAPACITY,
                ))
            }
        }
    }

    /// Apply a committed INSERT.
    pub(crate) fn apply_insert(&self, row_id: RowId, commit_ts: u64, data: Vec<u8>) {
        self.rows.write().entry(row_id).or_default().push_version(commit_ts, data);
        self.index_insert(row_id);
    }

    /// Apply a committed UPDATE (expire old + push new).
    pub(crate) fn apply_update(&self, row_id: RowId, commit_ts: u64, data: Vec<u8>) {
        let mut rows = self.rows.write();
        let chain = rows.entry(row_id).or_default();
        chain.expire_live(commit_ts);
        chain.push_version(commit_ts, data);
    }

    /// Apply a committed DELETE (expire live version).
    pub(crate) fn apply_delete(&self, row_id: RowId, commit_ts: u64) {
        if let Some(chain) = self.rows.write().get_mut(&row_id) {
            chain.expire_live(commit_ts);
        }
    }

    /// Row count (for diagnostics / TPC-C data load verification).
    pub fn row_count(&self) -> usize {
        self.rows.read().len()
    }

    /// Number of versions (live + dead) currently held for one row (for
    /// diagnostics and tests — e.g. verifying pruning actually bounds
    /// chain growth).
    pub fn version_count(&self, row_id: RowId) -> usize {
        self.rows.read().get(&row_id).map(|c| c.versions.len()).unwrap_or(0)
    }

    /// Sweep dead versions out of one row's chain. See `VersionChain::prune`
    /// for the safety argument. Returns the number removed.
    pub(crate) fn prune_row(&self, row_id: RowId, horizon: u64) -> usize {
        self.rows
            .write()
            .get_mut(&row_id)
            .map(|chain| chain.prune(horizon))
            .unwrap_or(0)
    }
}

// ── MVCC store (all tables) ───────────────────────────────────────────────────

#[derive(Default)]
pub struct MVCCStore {
    tables: RwLock<HashMap<u64, Arc<MVCCTable>>>,
}

impl MVCCStore {
    pub fn new() -> Self { MVCCStore::default() }

    /// Create a table if it doesn't exist (idempotent).
    pub fn create_table(&self, table_id: u64) {
        self.tables.write().entry(table_id).or_default();
    }

    /// Get a table handle (panics if table doesn't exist — call create_table first).
    pub fn table(&self, table_id: u64) -> Arc<MVCCTable> {
        self.tables.read().get(&table_id)
            .cloned()
            .unwrap_or_else(|| panic!("Table {} not created", table_id))
    }

    /// Non-panicking table lookup, for callers (like a query executor) that
    /// may legitimately be asked to read a table that's registered in the
    /// catalog but has never had a row written to it yet in this store.
    pub fn get_table(&self, table_id: u64) -> Option<Arc<MVCCTable>> {
        self.tables.read().get(&table_id).cloned()
    }

    /// Check if a row exists at a given snapshot (for write-write conflict detection).
    pub fn row_exists(&self, table_id: u64, row_id: u64, snap_ts: u64) -> bool {
        self.tables.read()
            .get(&table_id)
            .and_then(|t| t.read(row_id, snap_ts))
            .is_some()
    }

    /// Apply a committed write_set atomically.
    pub fn apply_write_set(&self, ops: &[WriteOp], commit_ts: u64) {
        let tables = self.tables.read();
        for op in ops {
            let (table_id, row_id) = op.table_row();
            if let Some(table) = tables.get(&table_id) {
                match op {
                    WriteOp::Insert { data, .. } => table.apply_insert(row_id, commit_ts, data.clone()),
                    WriteOp::Update { data, .. } => table.apply_update(row_id, commit_ts, data.clone()),
                    WriteOp::Delete { .. }       => table.apply_delete(row_id, commit_ts),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_range_finds_inserted_rows() {
        let table = MVCCTable::default();
        for id in [10u64, 20, 30, 40, 50] {
            table.apply_insert(id, 1, format!("row{id}").into_bytes());
        }

        let mut ids = table.index_range(15.0, 45.0);
        ids.sort();
        assert_eq!(ids, vec![20, 30, 40]);
    }

    #[test]
    fn test_index_range_on_empty_table_is_empty() {
        let table = MVCCTable::default();
        assert!(table.index_range(0.0, 100.0).is_empty());
    }

    #[test]
    fn test_index_range_includes_recently_inserted_unflushed_rows() {
        // Fewer inserts than PK_INDEX_BUFFER_CAPACITY, so these all sit in
        // the index's write buffer rather than its rebuilt base segments --
        // range_search must still find them.
        let table = MVCCTable::default();
        table.apply_insert(1, 1, b"a".to_vec());
        table.apply_insert(2, 1, b"b".to_vec());

        let mut ids = table.index_range(0.0, 10.0);
        ids.sort();
        assert_eq!(ids, vec![1, 2]);
    }
}
