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

    /// Real allocated heap memory this table's primary-key index (the
    /// live `DynamicPGMIndex` every real INSERT/read actually goes
    /// through, not a structure built separately for measurement) is
    /// currently using — see `index::pgm::DynamicPGMIndex::heap_bytes`.
    /// `0` before this table's first insert (no index built yet).
    pub fn pk_index_heap_bytes(&self) -> usize {
        self.pk_index.lock().as_ref().map(|i| i.heap_bytes()).unwrap_or(0)
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

    // ── Retention benchmark: learned predictor vs fixed-interval sweeping ──
    //
    // Real comparison of `execution::learned_retention::RetentionPredictor`
    // (the scheduler actually wired into `OLTPEngine::commit`, see
    // `oltp.rs`) against fixed-interval sweeping, under a realistic skewed
    // write workload. Drives `MVCCTable` directly with the same
    // `apply_insert`/`apply_update`/`prune_row` primitives `OLTPEngine`
    // itself calls, rather than adding a pluggable-policy seam to the real
    // commit path just to run a benchmark -- that path is hot and already
    // correct, not worth the risk for this.
    //
    // Lives here (an in-crate test) rather than as an `examples/` binary
    // because `apply_insert`/`apply_update`/`prune_row` are `pub(crate)` on
    // purpose: they bypass the WAL and lock manager entirely, and staying
    // reachable only from inside the crate is what keeps raw MVCC mutation
    // behind the transactional path for every real caller.
    mod retention_bloat_benchmark {
        use super::*;
        use crate::execution::learned_retention::RetentionPredictor;
        use std::collections::HashMap;

        const NUM_ROWS: u64 = 500;
        const NUM_HOT_ROWS: u64 = 20;
        const TOTAL_COMMITS: u64 = 20_000;
        /// The snapshot horizon (below which pruning is safe -- see
        /// `VersionChain::prune`'s docs) advances only every this-many
        /// commits, modeling a long-running reader that trails a few
        /// commits behind rather than always being fully caught up.
        /// Without that lag *any* prune attempt would always find
        /// something to remove and every policy would look identical.
        const HORIZON_LAG_EVERY: u64 = 5;

        /// Deterministic, dependency-free "random enough" stream --
        /// reproducible across runs (no RNG crate pulled in just for this),
        /// which matters since every policy must see the identical write
        /// sequence to be a fair comparison.
        struct Lcg(u64);
        impl Lcg {
            fn next(&mut self) -> u64 {
                self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                self.0 >> 33
            }
        }

        /// 80% of writes land on the 20 "hot" rows, the rest spread across
        /// the remaining 480 "cold" ones -- a realistic skew, not a
        /// uniform one; real OLTP access patterns concentrate like this.
        fn build_write_sequence() -> Vec<u64> {
            let mut rng = Lcg(0xC0FFEE);
            (0..TOTAL_COMMITS)
                .map(|_| {
                    if rng.next() % 100 < 80 {
                        rng.next() % NUM_HOT_ROWS
                    } else {
                        NUM_HOT_ROWS + (rng.next() % (NUM_ROWS - NUM_HOT_ROWS))
                    }
                })
                .collect()
        }

        struct Outcome {
            label: String,
            end_of_run_bloat: usize,
            sweep_attempts: usize,
            wasted_sweeps: usize,
        }

        fn horizon_at(commit_ts: u64) -> u64 {
            commit_ts.saturating_sub(1) / HORIZON_LAG_EVERY
        }

        fn apply_write(table: &MVCCTable, row_id: u64, commit_ts: u64) {
            if table.version_count(row_id) == 0 {
                table.apply_insert(row_id, commit_ts, vec![0u8]);
            } else {
                table.apply_update(row_id, commit_ts, vec![0u8]);
            }
        }

        fn total_bloat(table: &MVCCTable) -> usize {
            (0..NUM_ROWS).map(|r| table.version_count(r).saturating_sub(1)).sum()
        }

        fn run_learned(writes: &[u64]) -> Outcome {
            let table = MVCCTable::default();
            let predictor = RetentionPredictor::new();
            let mut sweep_attempts = 0usize;
            let mut wasted_sweeps = 0usize;

            for (i, &row_id) in writes.iter().enumerate() {
                let commit_ts = i as u64 + 1;
                apply_write(&table, row_id, commit_ts);

                predictor.record_write(1, row_id);
                if predictor.should_prune(1, row_id) {
                    sweep_attempts += 1;
                    let removed = table.prune_row(row_id, horizon_at(commit_ts));
                    if removed == 0 {
                        wasted_sweeps += 1;
                    }
                    predictor.record_prune_result(1, row_id, removed);
                }
            }

            println!("  [diagnostic] predictor.has_model() at end of run: {}", predictor.has_model());

            Outcome {
                label: "learned (RetentionPredictor)".to_string(),
                end_of_run_bloat: total_bloat(&table),
                sweep_attempts,
                wasted_sweeps,
            }
        }

        fn run_fixed_interval(writes: &[u64], interval: u64) -> Outcome {
            let table = MVCCTable::default();
            let mut writes_since_prune: HashMap<u64, u64> = HashMap::new();
            let mut sweep_attempts = 0usize;
            let mut wasted_sweeps = 0usize;

            for (i, &row_id) in writes.iter().enumerate() {
                let commit_ts = i as u64 + 1;
                apply_write(&table, row_id, commit_ts);

                let count = writes_since_prune.entry(row_id).or_insert(0);
                *count += 1;
                if *count >= interval {
                    *count = 0;
                    sweep_attempts += 1;
                    let removed = table.prune_row(row_id, horizon_at(commit_ts));
                    if removed == 0 {
                        wasted_sweeps += 1;
                    }
                }
            }

            Outcome {
                label: format!("fixed interval = {interval}"),
                end_of_run_bloat: total_bloat(&table),
                sweep_attempts,
                wasted_sweeps,
            }
        }

        fn print_outcome(o: &Outcome) {
            println!(
                "  {:<32} end-of-run bloat: {:>6}   sweep attempts: {:>6}   wasted: {:>6} ({:>5.1}%)",
                o.label,
                o.end_of_run_bloat,
                o.sweep_attempts,
                o.wasted_sweeps,
                100.0 * o.wasted_sweeps as f64 / o.sweep_attempts.max(1) as f64
            );
        }

        #[test]
        fn test_retention_bloat_comparison() {
            let writes = build_write_sequence();

            let sweep_every_write = run_fixed_interval(&writes, 1);
            let fixed_4 = run_fixed_interval(&writes, 4);
            let fixed_8 = run_fixed_interval(&writes, 8);
            let fixed_32 = run_fixed_interval(&writes, 32);
            let fixed_128 = run_fixed_interval(&writes, 128);
            let learned = run_learned(&writes);

            println!("\n=== MVCC Retention: Learned Predictor vs Fixed-Interval Sweeping ===");
            println!(
                "{NUM_ROWS} rows ({NUM_HOT_ROWS} hot, {} cold), {TOTAL_COMMITS} commits, \
                 80% of writes hit the hot set, horizon lags by up to {HORIZON_LAG_EVERY} commits\n",
                NUM_ROWS - NUM_HOT_ROWS
            );
            for o in [&sweep_every_write, &fixed_4, &fixed_8, &fixed_32, &fixed_128, &learned] {
                print_outcome(o);
            }

            // The real claim under test: sweeping every write (the
            // "always safe, never smart" extreme) can't be beaten on
            // bloat by anything that sweeps less often -- so the
            // meaningful bar for the learned predictor isn't "beats
            // everything," it's "gets close to sweep-every-write's bloat
            // without paying its wasted-sweep cost."
            assert!(
                learned.wasted_sweeps < sweep_every_write.wasted_sweeps,
                "learned predictor's wasted sweeps ({}) should be well below sweeping every write ({})",
                learned.wasted_sweeps,
                sweep_every_write.wasted_sweeps
            );
            assert!(
                learned.end_of_run_bloat < fixed_128.end_of_run_bloat,
                "learned predictor's bloat ({}) should be well below the too-sparse fixed_128 policy's ({})",
                learned.end_of_run_bloat,
                fixed_128.end_of_run_bloat
            );
        }
    }
}
