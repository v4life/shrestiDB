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

type Table = Mutex<HashMap<LockKey, LockEntry>>;

pub struct LockManager {
    inner:   Arc<(Table, Condvar)>,
    timeout: Duration,
}

impl LockManager {
    pub fn new() -> Self {
        LockManager {
            inner:   Arc::new((Mutex::new(HashMap::new()), Condvar::new())),
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
            let entry = table.entry(key.clone()).or_default();

            if entry.can_grant(tx_id, mode) {
                // Upgrade S → X in place if re-entering with stronger mode
                if let Some(holder) = entry.holders.iter_mut().find(|(id, _)| *id == tx_id) {
                    if mode == LockMode::Exclusive {
                        holder.1 = LockMode::Exclusive;
                    }
                } else {
                    entry.holders.push((tx_id, mode));
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
                let entry = table.entry(key.clone()).or_default();
                entry.waiters.retain(|(id, _)| *id != tx_id);
                return Err(DatabaseError::Other(format!(
                    "Deadlock timeout: tx {:?} aborted while waiting for lock on ({}, {})",
                    tx_id, key.table_id, key.row_id
                )));
            }

            let r = condvar.wait_for(&mut table, remaining);
            if r.timed_out() {
                let entry = table.entry(key.clone()).or_default();
                entry.waiters.retain(|(id, _)| *id != tx_id);
                return Err(DatabaseError::Other(format!(
                    "Deadlock timeout: tx {:?} aborted while waiting for lock on ({}, {})",
                    tx_id, key.table_id, key.row_id
                )));
            }
        }
    }

    /// Release every lock held by `tx_id` (called on commit or abort).
    pub fn release_all(&self, tx_id: TransactionId) {
        let (lock, condvar) = &*self.inner;
        let mut table = lock.lock();
        let mut woke = false;

        for entry in table.values_mut() {
            let before = entry.holders.len();
            entry.holders.retain(|(id, _)| *id != tx_id);
            entry.waiters.retain(|(id, _)| *id != tx_id);
            if entry.holders.len() < before {
                woke = true;
            }
        }
        table.retain(|_, e| !e.holders.is_empty() || !e.waiters.is_empty());

        if woke {
            condvar.notify_all();
        }
    }

    /// Number of rows currently locked (for diagnostics).
    pub fn lock_count(&self) -> usize {
        self.inner.0.lock().len()
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
