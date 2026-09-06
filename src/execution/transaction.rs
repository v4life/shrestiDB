//! Transaction management with MVCC

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Transaction ID
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct TransactionId(pub u64);

impl TransactionId {
    pub fn new(id: u64) -> Self {
        TransactionId(id)
    }
}

/// Transaction state
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TransactionState {
    Running,
    Committed,
    Aborted,
}

/// MVCC transaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transaction {
    pub id: TransactionId,
    pub state: TransactionState,
    pub start_timestamp: u64,
    pub end_timestamp: Option<u64>,
}

impl Transaction {
    pub fn new(id: TransactionId, start_timestamp: u64) -> Self {
        Transaction {
            id,
            state: TransactionState::Running,
            start_timestamp,
            end_timestamp: None,
        }
    }
}

/// Transaction manager
pub struct TransactionManager {
    next_tx_id: Arc<AtomicU64>,
    pub transactions: std::sync::RwLock<std::collections::HashMap<u64, Transaction>>,
}

impl TransactionManager {
    pub fn new() -> Self {
        TransactionManager {
            next_tx_id: Arc::new(AtomicU64::new(1)),
            transactions: std::sync::RwLock::new(std::collections::HashMap::new()),
        }
    }

    /// Start a new transaction
    pub fn begin(&self) -> TransactionId {
        let id = self.next_tx_id.fetch_add(1, Ordering::SeqCst);
        let tx = Transaction::new(TransactionId::new(id), id);
        self.transactions.write().unwrap().insert(id, tx);
        TransactionId::new(id)
    }

    /// Commit a transaction
    pub fn commit(&self, tx_id: TransactionId) {
        if let Ok(mut txs) = self.transactions.write() {
            if let Some(tx) = txs.get_mut(&tx_id.0) {
                tx.state = TransactionState::Committed;
                tx.end_timestamp = Some(self.next_tx_id.load(Ordering::SeqCst));
            }
        }
    }

    /// Abort a transaction
    pub fn abort(&self, tx_id: TransactionId) {
        if let Ok(mut txs) = self.transactions.write() {
            if let Some(tx) = txs.get_mut(&tx_id.0) {
                tx.state = TransactionState::Aborted;
            }
        }
    }
}

impl Default for TransactionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transaction_lifecycle() {
        let tm = TransactionManager::new();
        let tx_id = tm.begin();
        tm.commit(tx_id);

        let txs = tm.transactions.read().unwrap();
        assert_eq!(txs[&tx_id.0].state, TransactionState::Committed);
    }
}
