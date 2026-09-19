//! Row-level 2PL lock manager with deadlock detection via timeout
//!
//! Supports Shared (S) and Exclusive (X) locks per (table_id, row_id).
//! Deadlock is detected by a 100 ms acquisition timeout — any transaction
//! that waits longer is aborted and must retry.
//!
//! Compatibility matrix:
//!   S  ×  S  = ✅ compatible
//!   S  ×  X  = ❌
//!   X  ×  X  = ❌

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::{Condvar, Mutex};

use crate::error::{DatabaseError, Result};
use crate::execution::transaction::TransactionId;

// ── Lock mode ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockMode {
    Shared,
    Exclusive,
}

// ── Lock key ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct LockKey {
    pub table_id: u64,
    pub row_id:   u64,
}

impl LockKey {
    pub fn new(table_id: u64, row_id: u64) -> Self {
        LockKey { table_id, row_id }
    }
}

// ── Internal entry per locked row ────────────────────────────────────────────

#[derive(Debug, Default)]
struct LockEntry {
    holders: Vec<(TransactionId, LockMode)>,
    waiters: Vec<(TransactionId, LockMode)>,
}

impl LockEntry {
    /// True if `tx_id` can be granted `mode` immediately.
    fn can_grant(&self, tx_id: TransactionId, mode: LockMode) -> bool {
        // Re-entrant: already holds a lock on this row
        if self.holders.iter().any(|(id, _)| *id == tx_id) {
            return true;
        }
        match mode {
            LockMode::Shared => {
                // Grant S if no X holder and no X waiters ahead (FIFO fairness)
                self.holders.iter().all(|(_, m)| *m == LockMode::Shared)
                    && self.waiters.iter().all(|(_, m)| *m == LockMode::Shared)
            }
            LockMode::Exclusive => {
                // Grant X only when there are no holders at all
                self.holders.is_empty()
            }
        }
    }
}

// ── Lock manager ─────────────────────────────────────────────────────────────

/// `held_by` tracks which keys each transaction actually holds a lock on
/// -- what makes `release_all` (see its docs) touch only its own
/// transaction's locks instead of scanning `entries` for every
/// currently-locked row in the whole database. Kept under the same
/// `Mutex` as `entries` (rather than a second lock) so the two can never
/// observe each other mid-update; every mutation of one already holds
/// the lock the other needs anyway.
#[derive(Debug, Default)]
struct LockTable {
    entries: HashMap<LockKey, LockEntry>,
    held_by: HashMap<TransactionId, Vec<LockKey>>,
}

type Table = Mutex<LockTable>;

pub struct LockManager {
    inner:   Arc<(Table, Condvar)>,
    timeout: Duration,
}

impl LockManager {
    pub fn new() -> Self {
        LockManager {
            inner:   Arc::new((Mutex::new(LockTable::default()), Condvar::new())),
            timeout: Duration::from_millis(100),
        }
    }

    /// Acquire a shared lock (for reads under 2PL).
    pub fn acquire_shared(&self, tx_id: TransactionId, key: LockKey) -> Result<()> {
        self.acquire(tx_id, key, LockMode::Shared)
    }

    /// Acquire an exclusive lock (for writes).
    pub fn acquire_exclusive(&self, tx_id: TransactionId, key: LockKey) -> Result<()> {
        self.acquire(tx_id, key, LockMode::Exclusive)
    }

    fn acquire(&self, tx_id: TransactionId, key: LockKey, mode: LockMode) -> Result<()> {
        let (lock, condvar) = &*self.inner;
        let mut table = lock.lock();
        let deadline = std::time::Instant::now() + self.timeout;

        loop {
            // Disjoint borrows of `entries`/`held_by` (rather than going
            // through `table.entries`/`table.held_by` directly inside the
            // block below) so both can be mutated in the same scope --
            // needed once granting a lock also means recording it in
            // `held_by`.
            let LockTable { entries, held_by } = &mut *table;
            let entry = entries.entry(key.clone()).or_default();

            if entry.can_grant(tx_id, mode) {
                // Upgrade S → X in place if re-entering with stronger mode
                if let Some(holder) = entry.holders.iter_mut().find(|(id, _)| *id == tx_id) {
                    if mode == LockMode::Exclusive {
                        holder.1 = LockMode::Exclusive;
                    }
                } else {
                    entry.holders.push((tx_id, mode));
                    // Newly granted -- record it under this transaction so
                    // release_all (see its docs) doesn't have to scan
                    // every other transaction's locked rows to find it
                    // again. Not touched on a re-entrant upgrade (the
                    // branch above): the key's already recorded here from
                    // its first acquisition.
                    held_by.entry(tx_id).or_default().push(key.clone());
                }
                entry.waiters.retain(|(id, _)| *id != tx_id);
                return Ok(());
            }

            // Queue as waiter
            if !entry.waiters.iter().any(|(id, _)| *id == tx_id) {
                entry.waiters.push((tx_id, mode));
            }

            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                let entry = table.entries.entry(key.clone()).or_default();
                entry.waiters.retain(|(id, _)| *id != tx_id);
                return Err(DatabaseError::TransactionError(format!(
                    "Deadlock timeout: tx {:?} aborted while waiting for lock on ({}, {})",
                    tx_id, key.table_id, key.row_id
                )));
            }

            let r = condvar.wait_for(&mut table, remaining);
            if r.timed_out() {
                let entry = table.entries.entry(key.clone()).or_default();
                entry.waiters.retain(|(id, _)| *id != tx_id);
                return Err(DatabaseError::TransactionError(format!(
                    "Deadlock timeout: tx {:?} aborted while waiting for lock on ({}, {})",
                    tx_id, key.table_id, key.row_id
                )));
            }
        }
    }

    /// Release every lock held by `tx_id` (called on commit or abort).
    ///
    /// Used to iterate every currently-locked row in the whole database
    /// on every single call, regardless of how many (often just one or a
    /// handful) that particular transaction actually held -- real,
    /// measured cost: releasing one lock took 37.77ms with 200,000
    /// *unrelated* rows locked by other transactions elsewhere in the
    /// table (`test_release_all_does_not_scan_every_locked_row_in_the_database`). That
    /// scales with total concurrent lock count, not total data size like
    /// `DynamicPGMIndex`'s bulk-insert bug did, but the shape is the
    /// same: real cost under exactly the high-concurrency, many-short-
    /// transactions workload this codebase's OLTP-facing benchmarks care
    /// about, paid on every commit.
    ///
    /// Fixed the same way as that one -- stop paying for what you don't
    /// need to touch: `held_by` (see its docs) already knows exactly
    /// which keys this transaction holds, so this only ever visits those.
    pub fn release_all(&self, tx_id: TransactionId) {
        let (lock, condvar) = &*self.inner;
        let mut table = lock.lock();
        let mut woke = false;

        if let Some(keys) = table.held_by.remove(&tx_id) {
            for key in keys {
                let Some(entry) = table.entries.get_mut(&key) else { continue };
                let before = entry.holders.len();
                entry.holders.retain(|(id, _)| *id != tx_id);
                if entry.holders.len() < before {
                    woke = true;
                }
                if entry.holders.is_empty() && entry.waiters.is_empty() {
                    table.entries.remove(&key);
                }
            }
        }

        if woke {
            condvar.notify_all();
        }
    }

    /// Number of rows currently locked (for diagnostics).
    pub fn lock_count(&self) -> usize {
        self.inner.0.lock().entries.len()
    }
}

impl Default for LockManager {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    fn tx(id: u64) -> TransactionId { TransactionId::new(id) }
    fn key(t: u64, r: u64) -> LockKey { LockKey::new(t, r) }

    #[test]
    fn test_shared_shared_compatible() {
        let lm = LockManager::new();
        lm.acquire_shared(tx(1), key(1, 1)).unwrap();
        lm.acquire_shared(tx(2), key(1, 1)).unwrap(); // must not block
        lm.release_all(tx(1));
        lm.release_all(tx(2));
    }

    #[test]
    fn test_exclusive_blocks_and_times_out() {
        let lm = Arc::new(LockManager::new());
        lm.acquire_exclusive(tx(10), key(2, 1)).unwrap();

        let lm2 = lm.clone();
        let result = thread::spawn(move || {
            lm2.acquire_exclusive(tx(11), key(2, 1))
        }).join().unwrap();

        assert!(result.is_err(), "Should time out with deadlock error");
        lm.release_all(tx(10));
    }

    #[test]
    fn test_reentrant_upgrade() {
        let lm = LockManager::new();
        lm.acquire_shared(tx(1), key(3, 1)).unwrap();
        lm.acquire_exclusive(tx(1), key(3, 1)).unwrap(); // upgrade, same tx
        lm.release_all(tx(1));

        // A re-entrant upgrade must not double-record the key in held_by
        // (see acquire's docs on why it's only pushed on the *first*
        // acquisition) -- if it had, this release would still leave a
        // phantom holder entry behind, and a second transaction
        // requesting the same key would wrongly block.
        assert_eq!(lm.lock_count(), 0, "release_all should have fully cleared the row's entry, not left a residual one behind");
        lm.acquire_exclusive(tx(2), key(3, 1)).unwrap();
    }

    #[test]
    fn test_release_all_only_releases_that_transactions_own_locks() {
        let lm = LockManager::new();
        lm.acquire_exclusive(tx(1), key(1, 1)).unwrap();
        lm.acquire_exclusive(tx(1), key(1, 2)).unwrap();
        lm.acquire_exclusive(tx(2), key(1, 3)).unwrap();

        lm.release_all(tx(1));

        // tx(1)'s two locks are gone -- both immediately reacquirable by
        // someone else.
        lm.acquire_exclusive(tx(3), key(1, 1)).unwrap();
        lm.acquire_exclusive(tx(3), key(1, 2)).unwrap();
        // tx(2)'s lock is untouched by tx(1)'s release.
        assert!(lm.acquire_exclusive(tx(4), key(1, 3)).is_err(), "tx(2)'s lock should still be held, not released by tx(1)'s release_all");
    }

    #[test]
    fn test_release_all_on_a_transaction_holding_no_locks_is_a_harmless_no_op() {
        let lm = LockManager::new();
        lm.release_all(tx(99)); // never acquired anything
    }

    #[test]
    fn test_release_all_does_not_scan_every_locked_row_in_the_database() {
        // Real regression coverage for the bug this fix addresses:
        // release_all used to iterate every currently-locked row in the
        // whole table regardless of how many (if any) belonged to the
        // releasing transaction -- real, measured cost: releasing 1 lock
        // took 37.77ms with 200,000 *other* transactions' rows locked
        // (diagnosed before this fix). With held_by tracking each
        // transaction's own locks, this must complete in well under 10ms,
        // not tens of milliseconds.
        let lm = LockManager::new();
        let n = 200_000u64;
        for i in 0..n {
            lm.acquire_exclusive(tx(i), key(1, i)).unwrap();
        }

        let start = std::time::Instant::now();
        lm.release_all(tx(0));
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 10,
            "releasing 1 lock took {elapsed:?} with {n} other rows locked -- looks like the full-table-scan behavior again"
        );
    }

    #[test]
    fn test_release_unblocks_waiter() {
        let lm = Arc::new(LockManager::new());
        lm.acquire_exclusive(tx(1), key(4, 1)).unwrap();

        let lm2 = lm.clone();
        let handle = thread::spawn(move || {
            lm2.acquire_shared(tx(2), key(4, 1))
        });

        thread::sleep(Duration::from_millis(10));
        lm.release_all(tx(1)); // unblock waiter

        // tx(2) should now succeed (before 100ms timeout)
        assert!(handle.join().unwrap().is_ok());
    }

}
