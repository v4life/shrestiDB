//! Learned Index Structures
//!
//! Real, measured learned indexes (PGM, RMI) and this crate's own B-Tree
//! baseline they're compared against — see `examples/learned_index_demo.rs`.
//! A `hybrid_router` module (routing between a learned model and a
//! B-Tree fallback by data distribution) used to live here; nothing in
//! the real query execution path ever used it (the live PK index is
//! `DynamicPGMIndex`, used directly — see `execution::mvcc_store`), so
//! it was removed rather than left implying a capability that isn't
//! actually wired in.

pub mod btree;
pub mod models;
pub mod pgm;
pub mod rmi;

pub use pgm::{DynamicPGMIndex, PGMIndex, PGMSegment};
pub use rmi::RMIIndex;
