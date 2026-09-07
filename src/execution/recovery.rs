//! Crash recovery: replay a WAL to rebuild catalog + row state on startup.

use crate::execution::catalog::Catalog;
use crate::execution::oltp::OLTPEngine;
use crate::execution::wal::WalRecord;

pub struct RecoveryManager;

impl RecoveryManager {
    /// Replay `records` (in log order, as read by `WriteAheadLog::open`)
    /// into `catalog` and `oltp`. Restores the commit clock past the
    /// highest replayed timestamp, so a new transaction after recovery
    /// never reuses one.
    ///
    /// Returns the (table id, column) pairs any `CREATE INDEX`es named —
    /// rebuilding the actual index structure needs schema-aware
    /// deserialization (see `QueryExecutor::rebuild_secondary_index`),
    /// which this layer, deliberately, doesn't have.
    pub fn recover(records: Vec<WalRecord>, catalog: &mut Catalog, oltp: &OLTPEngine) -> Vec<(u64, String)> {
        let mut max_commit_ts = 0u64;
        let mut index_specs = Vec::new();

        for record in records {
            match record {
                WalRecord::CreateTable(schema) => {
                    let table_id = schema.table_id as u64;
                    oltp.create_table(table_id);
                    catalog.register_table(schema);
                }
                WalRecord::CreateIndex { table_id, column } => {
                    index_specs.push((table_id, column));
                }
                WalRecord::Commit { commit_ts, ops } => {
                    oltp.store.apply_write_set(&ops, commit_ts);
                    max_commit_ts = max_commit_ts.max(commit_ts);
                }
            }
        }

        oltp.restore_commit_clock(max_commit_ts);
        index_specs
    }
}
