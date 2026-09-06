//! Hybrid index router that switches between learned models and B-Tree

use crate::index::btree::BTree;
use crate::index::pgm::{DynamicPGMIndex, PGMIndex};
use crate::index::rmi::RMIIndex;
use serde::{Deserialize, Serialize};

/// Hybrid index that dynamically switches between learned and traditional indexes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HybridIndex {
    /// Learned RMI index
    RMI(RMIIndex),
    /// Learned PGM index
    PGM(PGMIndex),
    /// Dynamic PGM index with write buffer
    DynamicPGM(DynamicPGMIndex),
    /// Fallback B+ Tree
    BTree(BTree),
}

impl HybridIndex {
    /// Search using the active index strategy
    pub fn search(&self, key: f64) -> Option<usize> {
        match self {
            HybridIndex::RMI(rmi) => rmi.find_exact(key, 16),
            HybridIndex::PGM(pgm) => pgm.search(key),
            HybridIndex::DynamicPGM(dpgm) => {
                if dpgm.contains(key) {
                    Some(0)
                } else {
                    None
                }
            }
            HybridIndex::BTree(btree) => btree.search(key),
        }
    }

    /// Get the index type name
    pub fn index_type(&self) -> &'static str {
        match self {
            HybridIndex::RMI(_) => "RMI",
            HybridIndex::PGM(_) => "PGM",
            HybridIndex::DynamicPGM(_) => "DynamicPGM",
            HybridIndex::BTree(_) => "BTree",
        }
    }
}
