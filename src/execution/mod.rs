//! Query Execution Layer
//!
//! Vectorized execution engine with transaction support.

pub mod catalog;
pub mod operators;
pub mod executor;
pub mod transaction;
pub mod wal;
pub mod recovery;
pub mod lock_manager;
pub mod mvcc_store;
pub mod oltp;
pub mod learned_retention;
pub mod row_codec;
pub mod aggregate;

pub use catalog::Catalog;
pub use executor::QueryExecutor;
pub use transaction::TransactionManager;
pub use wal::WriteAheadLog;
pub use lock_manager::LockManager;
pub use mvcc_store::MVCCStore;
pub use oltp::OLTPEngine;
pub use learned_retention::RetentionPredictor;
