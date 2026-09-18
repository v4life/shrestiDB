//! ShrestiDB
//!
//! A database kernel with ML-driven indexing and query optimization —
//! see `index::pgm`/`index::rmi` (real, measured lookup speedups over
//! this crate's own B-Tree) and `optimizer::cardinality`/
//! `optimizer::join_reorder` (real, `ANALYZE`-driven cardinality
//! estimation and join ordering). This crate used to also claim
//! "adaptive buffer management" here: a `storage` module (disk-backed
//! pages, a Markov-chain buffer pool) existed, but nothing in the real
//! read/write path (`execution::oltp`, an in-memory `MVCCStore` plus a
//! plain append-only WAL — see `execution::wal`) ever used it. Removed
//! rather than left as unconnected code implying a capability the
//! engine doesn't have.

pub mod compute;
pub mod distributed;
pub mod error;
pub mod execution;
pub mod index;
pub mod ml;
pub mod optimizer;
pub mod sql;

pub use error::{DatabaseError, Result};

/// Database kernel version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
