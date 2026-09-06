//! Query execution engine

use crate::error::Result;
use crate::execution::catalog::Catalog;
use crate::execution::mvcc_store::WriteOp;
use crate::execution::oltp::OLTPEngine;
use crate::execution::transaction::TransactionId;
use crate::optimizer::planner::PhysicalPlan;

/// Query executor.
///
/// `oltp` provides transactional, lock- and MVCC-backed row access
/// (`begin_transaction` / `read_row` / `write_row` / `commit_transaction` /
/// `abort_transaction`). `execute()` itself is still a plan-execution stub —
/// there is no compiler from `PhysicalPlan` into operators bound to real
/// storage yet, so it does not route through `oltp`.
pub struct QueryExecutor {
    pub catalog: Catalog,
    pub oltp: OLTPEngine,
}

impl QueryExecutor {
    pub fn new(catalog: Catalog) -> Self {
        QueryExecutor {
            catalog,
            oltp: OLTPEngine::new(),
        }
    }

    /// Execute a query plan
    pub fn execute(&self, _plan: &PhysicalPlan) -> Result<Vec<Vec<String>>> {
        // Simplified: return empty results
        Ok(Vec::new())
    }

    /// Begin a new transaction against the OLTP engine.
    pub fn begin_transaction(&self) -> TransactionId {
        self.oltp.begin()
    }

    /// Transactional point read of a row, honoring 2PL locking and MVCC
    /// snapshot isolation.
    pub fn read_row(
        &self,
        tx_id: TransactionId,
        table_id: u64,
        row_id: u64,
    ) -> Result<Option<Vec<u8>>> {
        self.oltp.read(tx_id, table_id, row_id)
    }

    /// Buffer a transactional write; not visible to other transactions until
    /// `commit_transaction`.
    pub fn write_row(&self, tx_id: TransactionId, op: WriteOp) -> Result<()> {
        self.oltp.write(tx_id, op)
    }

    pub fn commit_transaction(&self, tx_id: TransactionId) {
        self.oltp.commit(tx_id);
    }

    pub fn abort_transaction(&self, tx_id: TransactionId) {
        self.oltp.abort(tx_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::mvcc_store::WriteOp;

    #[test]
    fn test_executor_creation() {
        let catalog = Catalog::new();
        let executor = QueryExecutor::new(catalog);
        assert_eq!(executor.catalog.tables.len(), 0);
    }

    #[test]
    fn test_executor_transactional_read_write() {
        let executor = QueryExecutor::new(Catalog::new());
        executor.oltp.create_table(1);

        let tx = executor.begin_transaction();
        executor
            .write_row(
                tx,
                WriteOp::Insert {
                    table_id: 1,
                    row_id: 1,
                    data: b"row".to_vec(),
                },
            )
            .unwrap();
        assert_eq!(executor.read_row(tx, 1, 1).unwrap(), Some(b"row".to_vec()));
        executor.commit_transaction(tx);

        let tx2 = executor.begin_transaction();
        assert_eq!(executor.read_row(tx2, 1, 1).unwrap(), Some(b"row".to_vec()));
    }
}
