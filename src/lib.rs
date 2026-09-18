//! ShrestiDB
//!
//! A database kernel with ML-driven indexing and query optimization —
//! see `index::pgm`/`index::rmi` (real, measured lookup speedups over
//! this crate's own real B+Tree) and `optimizer::cardinality`/
//! `optimizer::join_reorder` (real, `ANALYZE`-driven cardinality
//! estimation and join ordering). This crate used to also claim
//! "adaptive buffer management" here: a `storage` module (disk-backed
//! pages, a Markov-chain buffer pool) existed, but nothing in the real
//! read/write path (`execution::oltp`, an in-memory `MVCCStore` plus a
//! plain append-only WAL — see `execution::wal`) ever used it. A
//! separate `ml` module (`LinearRegression`, `PolynomialRegression`,
//! `NeuralNetwork`, `ARModel`, `ModelTrainer`) existed alongside the
//! real learned components above with the same problem, one level
//! worse: most of it wasn't just unused but non-functional even in
//! isolation (`PolynomialRegression::fit` ignored its `y` argument
//! entirely and returned a hardcoded placeholder; `NeuralNetwork`'s
//! weights were hardcoded and `ModelTrainer::train` was a literal
//! no-op, the same shape as the fake cardinality "neural network"
//! already found and replaced elsewhere in this codebase's history).
//! `git log` confirms it was never wired into anything outside itself
//! since the commit that added it. Both removed rather than left as
//! unconnected code implying capabilities the engine doesn't have.

pub mod compute;
pub mod distributed;
pub mod error;
pub mod execution;
pub mod index;
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
