//! OLTP engine: wires `TransactionManager`, `LockManager`, and `MVCCStore`
//! together into a single transactional read/write API.
//!
//! Protocol per transaction:
//!   1. `begin()` records a snapshot timestamp (the current commit watermark).
//!   2. `read()` acquires a shared lock, then reads via the write set
//!      (read-your-own-writes) or, failing that, the version visible at the
//!      transaction's snapshot.
//!   3. `write()` acquires an exclusive lock and buffers the op — nothing is
//!      applied to the MVCC store until commit, so other transactions never
//!      see uncommitted writes.
//!   4. `commit()` assigns a commit timestamp, durably logs the write set to
//!      the WAL (if one is configured — see `with_wal`) and only then
//!      applies it and releases all locks. If the WAL write fails, the
//!      transaction is aborted rather than applied: an in-memory-only
//!      "commit" that isn't actually durable would be a correctness bug,
//!      not a degraded feature. `abort()` discards the write set and
//!      releases all locks.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use crate::error::Result;
use crate::execution::learned_retention::RetentionPredictor;
use crate::execution::lock_manager::{LockKey, LockManager};
use crate::execution::mvcc_store::{MVCCStore, WriteOp};
use crate::execution::transaction::{TransactionId, TransactionManager};
use crate::execution::wal::{WalRecord, WriteAheadLog};

pub struct OLTPEngine {
    pub transactions: TransactionManager,
    pub locks: LockManager,
    pub store: MVCCStore,
    pub retention: RetentionPredictor,
    /// `None` means no durability: writes only ever live in memory (the
    /// original behavior, still what every in-process test uses). `Some`
    /// means every commit is logged to disk before it takes effect.
    wal: Option<WriteAheadLog>,
    commit_clock: AtomicU64,
    write_sets: Mutex<HashMap<u64, Vec<WriteOp>>>,
    snapshots: Mutex<HashMap<u64, u64>>,
}

impl OLTPEngine {
    pub fn new() -> Self {
        OLTPEngine {
            transactions: TransactionManager::new(),
            locks: LockManager::new(),
            store: MVCCStore::new(),
            retention: RetentionPredictor::new(),
            wal: None,
            commit_clock: AtomicU64::new(0),
            write_sets: Mutex::new(HashMap::new()),
            snapshots: Mutex::new(HashMap::new()),
        }
    }

    /// A durable engine: every commit is logged to `wal` before it takes
    /// effect. The caller (see `QueryExecutor::open`) is responsible for
    /// having already replayed `wal`'s existing records into this engine's
    /// store (and a catalog) before handing it out for new queries.
    pub fn with_wal(wal: WriteAheadLog) -> Self {
        OLTPEngine {
            wal: Some(wal),
            ..Self::new()
        }
    }

    /// Log a schema change (CREATE TABLE) to the WAL, if one is configured.
    /// Called by `QueryExecutor::execute_create_table` — schema changes
    /// aren't part of a transaction's write set, so they don't go through
    /// `commit()`.
    pub fn log_schema_change(&self, schema: &crate::execution::catalog::TableSchema) -> Result<()> {
        match &self.wal {
            Some(wal) => wal.append(&WalRecord::CreateTable(schema.clone())),
            None => Ok(()),
        }
    }

    /// Log a CREATE INDEX, if a WAL is configured. Called by
    /// `QueryExecutor::execute_create_index` — same reasoning as
    /// `log_schema_change`: this is a catalog-level event, not part of any
    /// transaction's write set, so it doesn't go through `commit()`.
    pub fn log_index_change(&self, table_id: u64, column: &str) -> Result<()> {
        match &self.wal {
            Some(wal) => wal.append(&WalRecord::CreateIndex { table_id, column: column.to_string() }),
            None => Ok(()),
        }
    }

    /// Advance the commit clock to at least `ts` (used after replaying a
    /// WAL on startup, so the first new commit gets a timestamp past
    /// everything already replayed).
    pub fn restore_commit_clock(&self, ts: u64) {
        self.commit_clock.fetch_max(ts, Ordering::SeqCst);
    }

    /// The latest commit timestamp as of right now -- a read at this
    /// timestamp (not `mvcc_store::TS_INFINITY`; see `RowVersion::visible_at`'s
    /// strict `<`, which a value equal to `end_ts` on a still-live version
    /// would fail) sees every write committed before this call returns.
    /// For a caller that already holds a row's exclusive lock and wants
    /// its truly current value rather than an older transaction snapshot
    /// — see `QueryExecutor::execute_update`'s doc comment for why that
    /// distinction matters.
    pub fn current_commit_ts(&self) -> u64 {
        self.commit_clock.load(Ordering::SeqCst)
    }

    /// Create a table in the underlying MVCC store (idempotent).
    pub fn create_table(&self, table_id: u64) {
        self.store.create_table(table_id);
    }

    /// Begin a new transaction, snapshotting the current commit watermark.
    pub fn begin(&self) -> TransactionId {
        let tx_id = self.transactions.begin();
        let snap_ts = self.commit_clock.load(Ordering::SeqCst);
        self.snapshots.lock().unwrap().insert(tx_id.0, snap_ts);
        self.write_sets.lock().unwrap().insert(tx_id.0, Vec::new());
        tx_id
    }

    /// Run `f` against a fresh read-only snapshot, then end the
    /// transaction. A read-only caller can't forget to clean up this way —
    /// an unended transaction's snapshot would sit in `min_active_snapshot`
    /// forever, blocking the learned retention sweep from ever pruning
    /// anything at or after it.
    pub fn with_read_snapshot<T>(&self, f: impl FnOnce(TransactionId) -> T) -> T {
        let tx_id = self.begin();
        let result = f(tx_id);
        self.abort(tx_id);
        result
    }

    /// Full-table read at `tx_id`'s snapshot: every row visible to it. An
    /// unknown table (registered in the catalog but never written to in
    /// this store) reads as empty rather than erroring — that's a
    /// legitimate state for a freshly created table.
    pub fn scan_table(&self, tx_id: TransactionId, table_id: u64) -> Vec<(u64, Vec<u8>)> {
        let snap_ts = *self.snapshots.lock().unwrap().get(&tx_id.0).unwrap_or(&0);
        match self.store.get_table(table_id) {
            Some(table) => table.scan(snap_ts),
            None => Vec::new(),
        }
    }

    /// Transactional point read: checks the transaction's own uncommitted
    /// writes first, then falls back to the MVCC snapshot.
    ///
    /// Reads never take a lock. That's the point of MVCC: a reader is
    /// satisfied entirely by the last version committed at or before its
    /// snapshot, so it never has to wait on a concurrent writer (and a
    /// writer never has to wait on a reader). Locking is only needed
    /// between writers, to serialize conflicting writes to the same row —
    /// see `write()`.
    pub fn read(&self, tx_id: TransactionId, table_id: u64, row_id: u64) -> Result<Option<Vec<u8>>> {
        if let Some(ops) = self.write_sets.lock().unwrap().get(&tx_id.0) {
            for op in ops.iter().rev() {
                if op.table_row() == (table_id, row_id) {
                    return Ok(match op {
                        WriteOp::Insert { data, .. } | WriteOp::Update { data, .. } => {
                            Some(data.clone())
                        }
                        WriteOp::Delete { .. } => None,
                    });
                }
            }
        }

        let snap_ts = *self.snapshots.lock().unwrap().get(&tx_id.0).unwrap_or(&0);
        Ok(self.store.table(table_id).read(row_id, snap_ts))
    }

    /// Buffer a write under an exclusive lock. Invisible to every other
    /// transaction (and to fresh reads from the store) until commit.
    pub fn write(&self, tx_id: TransactionId, op: WriteOp) -> Result<()> {
        let (table_id, row_id) = op.table_row();
        self.locks
            .acquire_exclusive(tx_id, LockKey::new(table_id, row_id))?;
        self.write_sets
            .lock()
            .unwrap()
            .entry(tx_id.0)
            .or_default()
            .push(op);
        Ok(())
    }

    /// Commit: assign a commit timestamp, apply the buffered write set
    /// atomically, release all locks, then let the learned retention
    /// predictor decide whether each written row's chain is worth sweeping
    /// now (see `learned_retention.rs`). The sweep itself is always safe —
    /// it only drops versions no active snapshot could still need — so a
    /// wrong prediction costs efficiency, never correctness.
    pub fn commit(&self, tx_id: TransactionId) -> Result<()> {
        let ops = self.write_sets.lock().unwrap().remove(&tx_id.0).unwrap_or_default();
        let commit_ts = self.commit_clock.fetch_add(1, Ordering::SeqCst) + 1;

        if let Some(wal) = &self.wal {
            if !ops.is_empty() {
                if let Err(e) = wal.append(&WalRecord::Commit { commit_ts, ops: ops.clone() }) {
                    // Not durable: treat this as an abort, not a commit.
                    self.transactions.abort(tx_id);
                    self.locks.release_all(tx_id);
                    self.snapshots.lock().unwrap().remove(&tx_id.0);
                    return Err(e);
                }
            }
        }

        self.store.apply_write_set(&ops, commit_ts);
        self.transactions.commit(tx_id);
        self.locks.release_all(tx_id);
        self.snapshots.lock().unwrap().remove(&tx_id.0);

        let horizon = self.min_active_snapshot();
        for op in &ops {
            let (table_id, row_id) = op.table_row();
            self.retention.record_write(table_id, row_id);
            if self.retention.should_prune(table_id, row_id) {
                let removed = self.store.table(table_id).prune_row(row_id, horizon);
                self.retention.record_prune_result(table_id, row_id, removed);
            }
        }

        Ok(())
    }

    /// Abort: discard the buffered write set (nothing ever touched the
    /// store) and release all locks.
    pub fn abort(&self, tx_id: TransactionId) {
        self.write_sets.lock().unwrap().remove(&tx_id.0);
        self.transactions.abort(tx_id);
        self.locks.release_all(tx_id);
        self.snapshots.lock().unwrap().remove(&tx_id.0);
    }

    /// The oldest snapshot timestamp any currently active transaction could
    /// still be reading from — the horizon below which pruning is safe. No
    /// active transaction means nothing needs history older than "now".
    fn min_active_snapshot(&self) -> u64 {
        self.snapshots
            .lock()
            .unwrap()
            .values()
            .copied()
            .min()
            .unwrap_or_else(|| self.commit_clock.load(Ordering::SeqCst))
    }
}

impl Default for OLTPEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    fn insert(table_id: u64, row_id: u64, val: &str) -> WriteOp {
        WriteOp::Insert {
            table_id,
            row_id,
            data: val.as_bytes().to_vec(),
        }
    }

    #[test]
    fn test_read_your_own_writes() {
        let engine = OLTPEngine::new();
        engine.create_table(1);
        let tx = engine.begin();

        assert_eq!(engine.read(tx, 1, 1).unwrap(), None);
        engine.write(tx, insert(1, 1, "hello")).unwrap();
        assert_eq!(engine.read(tx, 1, 1).unwrap(), Some(b"hello".to_vec()));

        engine.commit(tx).unwrap();
    }

    #[test]
    fn test_snapshot_isolation_hides_uncommitted_and_later_writes() {
        let engine = OLTPEngine::new();
        engine.create_table(1);

        let reader = engine.begin();

        let writer = engine.begin();
        engine.write(writer, insert(1, 1, "v1")).unwrap();
        // Not visible to `reader`: uncommitted, and `reader`'s snapshot predates the commit.
        assert_eq!(engine.read(reader, 1, 1).unwrap(), None);
        engine.commit(writer).unwrap();

        // Still not visible: `reader`'s snapshot was taken before the commit.
        assert_eq!(engine.read(reader, 1, 1).unwrap(), None);

        // A transaction started after the commit sees it.
        let late_reader = engine.begin();
        assert_eq!(engine.read(late_reader, 1, 1).unwrap(), Some(b"v1".to_vec()));
    }

    #[test]
    fn test_abort_discards_writes() {
        let engine = OLTPEngine::new();
        engine.create_table(1);

        let tx = engine.begin();
        engine.write(tx, insert(1, 1, "should not persist")).unwrap();
        engine.abort(tx);

        let checker = engine.begin();
        assert_eq!(engine.read(checker, 1, 1).unwrap(), None);
    }

    #[test]
    fn test_learned_pruning_bounds_chain_growth() {
        let engine = OLTPEngine::new();
        engine.create_table(1);

        // Repeatedly update the same row. Without pruning, expire_live never
        // removes anything and the chain would grow by one dead version per
        // update forever. The learned predictor should trigger sweeps once
        // it (or the bootstrap fallback) decides it's worth it, keeping the
        // chain from growing without bound.
        for i in 0..100 {
            let tx = engine.begin();
            engine
                .write(
                    tx,
                    WriteOp::Update {
                        table_id: 1,
                        row_id: 1,
                        data: format!("v{i}").into_bytes(),
                    },
                )
                .unwrap();
            engine.commit(tx).unwrap();
        }

        let versions = engine.store.table(1).version_count(1);
        assert!(
            versions < 100,
            "expected pruning to bound chain growth, got {versions} versions after 100 updates"
        );
        // The final value must still be readable.
        let checker = engine.begin();
        assert_eq!(engine.read(checker, 1, 1).unwrap(), Some(b"v99".to_vec()));
    }

    #[test]
    fn test_pruning_never_breaks_a_long_lived_reader_snapshot() {
        let engine = OLTPEngine::new();
        engine.create_table(1);

        let tx0 = engine.begin();
        engine.write(tx0, insert(1, 1, "v0")).unwrap();
        engine.commit(tx0).unwrap();

        // Reader starts a snapshot right after v0 commits, but doesn't read yet.
        let reader = engine.begin();

        // Many subsequent commits to the same row, aggressively hammering
        // the row so the predictor is very likely to trigger pruning sweeps
        // along the way, while `reader`'s snapshot is still active.
        for i in 1..50 {
            let tx = engine.begin();
            engine
                .write(
                    tx,
                    WriteOp::Update {
                        table_id: 1,
                        row_id: 1,
                        data: format!("v{i}").into_bytes(),
                    },
                )
                .unwrap();
            engine.commit(tx).unwrap();
        }

        // `reader`'s snapshot predates every one of those updates, so it
        // must still see "v0" — the min_active_snapshot horizon must have
        // protected that version from every sweep that ran while it was open.
        assert_eq!(engine.read(reader, 1, 1).unwrap(), Some(b"v0".to_vec()));
    }

    #[test]
    fn test_concurrent_writers_to_same_row_serialize_via_lock() {
        let engine = Arc::new(OLTPEngine::new());
        engine.create_table(1);

        let tx1 = engine.begin();
        engine.write(tx1, insert(1, 1, "from tx1")).unwrap();

        let engine2 = engine.clone();
        let handle = thread::spawn(move || {
            let tx2 = engine2.begin();
            // tx1 holds the exclusive lock on row 1; this must block until
            // tx1 releases it (commit) or time out (deadlock detection).
            engine2.write(tx2, insert(1, 1, "from tx2"))
        });

        thread::sleep(Duration::from_millis(10));
        engine.commit(tx1).unwrap(); // releases the lock, unblocking tx2's writer

        assert!(handle.join().unwrap().is_ok());
    }

    /// Documents a real property of the raw API, not a bug: OLTPEngine's
    /// read()/write()/commit() primitives, used the naive way (read once,
    /// compute, write, no re-read under lock), are vulnerable to lost
    /// updates under real concurrency. read() never locks (by MVCC
    /// design), so there's a window between a thread's read and its
    /// write()'s lock acquisition where another thread can fully commit
    /// -- a naive caller's write then overwrites that commit with a value
    /// computed from stale data. Contrast with
    /// executor::tests::test_concurrent_sql_updates_dont_lose_writes:
    /// the SQL UPDATE path is safe because execute_update re-reads the
    /// row's latest committed value under lock before computing the new
    /// one -- that discipline lives in the caller, not this primitive.
    #[test]
    fn test_raw_primitives_naive_read_then_write_can_lose_updates() {
        let engine = Arc::new(OLTPEngine::new());
        engine.create_table(1);
        let tx0 = engine.begin();
        engine.write(tx0, insert(1, 1, "0")).unwrap();
        engine.commit(tx0).unwrap();

        const THREADS: usize = 8;
        const INCREMENTS_PER_THREAD: usize = 25;
        let mut handles = Vec::new();
        for _ in 0..THREADS {
            let engine = engine.clone();
            handles.push(thread::spawn(move || {
                for _ in 0..INCREMENTS_PER_THREAD {
                    let tx = engine.begin();
                    let current: i64 = engine
                        .read(tx, 1, 1)
                        .unwrap()
                        .and_then(|b| String::from_utf8(b).ok())
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0);
                    // NAIVE: no re-read under lock -- compute directly
                    // from the value read before acquiring any lock.
                    engine
                        .write(tx, WriteOp::Update { table_id: 1, row_id: 1, data: (current + 1).to_string().into_bytes() })
                        .unwrap();
                    engine.commit(tx).unwrap();
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        let checker = engine.begin();
        let final_val: i64 = engine
            .read(checker, 1, 1)
            .unwrap()
            .and_then(|b| String::from_utf8(b).ok())
            .and_then(|s| s.parse().ok())
            .unwrap();
        let expected = (THREADS * INCREMENTS_PER_THREAD) as i64;
        assert!(
            final_val < expected,
            "expected naive read-then-write to lose some updates under real concurrency \
             (got the full {expected} -- either this machine got extraordinarily lucky with \
             thread scheduling, or the primitive's documented lack of protection has changed)"
        );
    }
}
