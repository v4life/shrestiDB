//! Write-Ahead Log (WAL) for durability
//!
//! A minimal but real WAL: every CREATE TABLE and every committed
//! transaction's write set is appended as a length-prefixed bincode record
//! and fsynced before the caller (see `OLTPEngine::commit`) is told the
//! write succeeded — so once a caller sees success, that data survives a
//! crash. Records are replayed in order on startup (see
//! `execution::recovery`) to rebuild state before any new query runs.
//!
//! Framing: each record is `[4-byte little-endian length][bincode payload]`.
//! A torn trailing record (the process crashed mid-append, so the payload
//! is shorter than its length prefix promised, or doesn't deserialize) is
//! treated as "nothing to recover from that record" and replay stops
//! there, rather than failing startup — the write it represents was never
//! fully durable (the fsync that would have made it so never completed),
//! so losing it is correct, not a bug.

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::execution::catalog::TableSchema;
use crate::execution::mvcc_store::WriteOp;

/// One durable event: either a new table's schema, or a transaction's
/// committed write set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WalRecord {
    CreateTable(TableSchema),
    Commit { commit_ts: u64, ops: Vec<WriteOp> },
}

pub struct WriteAheadLog {
    file: Mutex<File>,
    #[allow(dead_code)]
    path: PathBuf,
}

impl WriteAheadLog {
    /// Open (creating if needed) the WAL at `path`, returning it along with
    /// every record already in it, in log order — the caller replays these
    /// (see `execution::recovery::RecoveryManager::recover`) before
    /// accepting new writes.
    pub fn open(path: impl AsRef<Path>) -> Result<(Self, Vec<WalRecord>)> {
        let path = path.as_ref().to_path_buf();
        let records = if path.exists() { Self::read_all(&path)? } else { Vec::new() };

        let file = OpenOptions::new().create(true).append(true).open(&path)?;

        Ok((WriteAheadLog { file: Mutex::new(file), path }, records))
    }

    /// Append `record`, fsynced before returning. An error here means the
    /// write is NOT durable — callers must treat that as the write
    /// failing, not succeeding (see `OLTPEngine::commit`, which aborts the
    /// transaction rather than applying it if this fails).
    pub fn append(&self, record: &WalRecord) -> Result<()> {
        let bytes = bincode::serialize(record)
            .map_err(|e| crate::error::DatabaseError::SerializationError(e.to_string()))?;
        let mut file = self.file.lock().unwrap();
        file.write_all(&(bytes.len() as u32).to_le_bytes())?;
        file.write_all(&bytes)?;
        file.sync_data()?;
        Ok(())
    }

    fn read_all(path: &Path) -> Result<Vec<WalRecord>> {
        let mut file = File::open(path)?;
        let mut records = Vec::new();

        loop {
            let mut len_buf = [0u8; 4];
            if file.read_exact(&mut len_buf).is_err() {
                break; // clean EOF, or a torn length prefix -- either way, done
            }
            let len = u32::from_le_bytes(len_buf) as usize;
            let mut payload = vec![0u8; len];
            if file.read_exact(&mut payload).is_err() {
                break; // torn trailing record from a crash mid-append
            }
            match bincode::deserialize::<WalRecord>(&payload) {
                Ok(record) => records.push(record),
                Err(_) => break, // corrupt trailing record: same treatment
            }
        }

        Ok(records)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::catalog::{Column, DataType};

    #[test]
    fn test_append_and_reopen_replays_records() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.wal");

        {
            let (wal, existing) = WriteAheadLog::open(&path).unwrap();
            assert!(existing.is_empty());
            wal.append(&WalRecord::Commit {
                commit_ts: 1,
                ops: vec![WriteOp::Insert { table_id: 1, row_id: 1, data: b"hello".to_vec() }],
            })
            .unwrap();
            wal.append(&WalRecord::Commit {
                commit_ts: 2,
                ops: vec![WriteOp::Insert { table_id: 1, row_id: 2, data: b"world".to_vec() }],
            })
            .unwrap();
        }

        let (_wal, records) = WriteAheadLog::open(&path).unwrap();
        assert_eq!(records.len(), 2);
    }

    #[test]
    fn test_create_table_record_roundtrips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.wal");

        let mut schema = TableSchema::new(1, "users".to_string());
        schema.add_column(Column {
            id: 1,
            name: "id".to_string(),
            data_type: DataType::Integer,
            nullable: false,
            primary_key: true,
        });

        {
            let (wal, _) = WriteAheadLog::open(&path).unwrap();
            wal.append(&WalRecord::CreateTable(schema)).unwrap();
        }

        let (_wal, records) = WriteAheadLog::open(&path).unwrap();
        assert_eq!(records.len(), 1);
        match &records[0] {
            WalRecord::CreateTable(s) => assert_eq!(s.name, "users"),
            other => panic!("expected CreateTable record, got {other:?}"),
        }
    }

    #[test]
    fn test_truncated_tail_record_is_dropped_not_fatal() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.wal");

        {
            let (wal, _) = WriteAheadLog::open(&path).unwrap();
            wal.append(&WalRecord::Commit { commit_ts: 1, ops: vec![] }).unwrap();
        }
        // Simulate a crash mid-append: a length prefix with no payload behind it.
        {
            let mut f = std::fs::OpenOptions::new().append(true).open(&path).unwrap();
            f.write_all(&100u32.to_le_bytes()).unwrap(); // promises 100 bytes, delivers none
        }

        let (_wal, records) = WriteAheadLog::open(&path).unwrap();
        assert_eq!(records.len(), 1); // the torn record is dropped, not fatal
    }
}
