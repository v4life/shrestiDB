//! B+ Tree implementation — the real "traditional index" baseline PGM/RMI
//! are measured against elsewhere in this codebase.
//!
//! An earlier version of this file was not a B+ Tree at all: `insert`'s own
//! comment said "Simplified insert: just add to root" — every key ever
//! inserted lived in one `BTreeNode`, so `search` was really just
//! `Vec::binary_search_by` over a flat, unsplit array wearing B-tree-shaped
//! names. Every "PGM/RMI beats a B-Tree by Nx" number this project has ever
//! printed (including the historical 36.8x figure) was measured against
//! that, not against a real multi-level tree with the node-fanout and
//! pointer-chasing overhead the comparison is supposed to be about — an
//! honesty gap discovered while trying to build a fair memory-footprint
//! comparison and worth fixing before adding anything on top of it.
//!
//! This is now a real B+Tree: internal nodes split at `order` keys, a
//! promoted separator key propagates upward (copied from the leaf level,
//! moved at internal levels — the standard B+Tree distinction), and the
//! root grows a new level on overflow. Range scans use recursive descent
//! with child-range pruning rather than the leaf-sibling linked list a
//! production B+Tree would use — this index isn't on any live query path
//! (`execution::mvcc_store::MVCCTable` uses `DynamicPGMIndex` directly), so
//! it only needs to be a fair, real baseline, not a fully production-grade
//! one.

use crate::error::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
enum Node {
    Leaf {
        keys: Vec<f64>,
        values: Vec<usize>,
    },
    Internal {
        /// `keys[i]` separates `children[i]` (strictly less) from
        /// `children[i + 1]` (greater or equal) — `children.len()` is
        /// always `keys.len() + 1`.
        keys: Vec<f64>,
        children: Vec<Box<Node>>,
    },
}

impl Node {
    fn search(&self, key: f64) -> Option<usize> {
        match self {
            Node::Leaf { keys, values } => keys
                .binary_search_by(|k| k.partial_cmp(&key).unwrap())
                .ok()
                .map(|i| values[i]),
            Node::Internal { keys, children } => {
                let idx = keys.partition_point(|&k| k <= key);
                children[idx].search(key)
            }
        }
    }

    /// Inserts `(key, value)` into this subtree. `Some((separator,
    /// right_sibling))` means this node overflowed past `order` keys and
    /// split -- the caller (the parent, or `BTree::insert` for the root)
    /// is responsible for inserting `separator` and `right_sibling` into
    /// itself, which may cascade into another split one level up.
    fn insert(&mut self, key: f64, value: usize, order: usize) -> Option<(f64, Box<Node>)> {
        match self {
            Node::Leaf { keys, values } => {
                let pos = keys.partition_point(|&k| k < key);
                if pos < keys.len() && keys[pos] == key {
                    values[pos] = value; // key already present: update, don't duplicate
                    return None;
                }
                keys.insert(pos, key);
                values.insert(pos, value);
                if keys.len() < order {
                    return None;
                }

                let mid = keys.len() / 2;
                let right_keys = keys.split_off(mid);
                let right_values = values.split_off(mid);
                // B+Tree leaves: the separator is copied (the key still
                // lives in the right leaf too), unlike an internal split.
                let separator = right_keys[0];
                Some((separator, Box::new(Node::Leaf { keys: right_keys, values: right_values })))
            }
            Node::Internal { keys, children } => {
                let idx = keys.partition_point(|&k| k <= key);
                let split = children[idx].insert(key, value, order);
                let Some((separator, right_child)) = split else {
                    return None;
                };

                keys.insert(idx, separator);
                children.insert(idx + 1, right_child);
                if keys.len() < order {
                    return None;
                }

                let mid = keys.len() / 2;
                let promoted = keys[mid];
                let right_keys = keys.split_off(mid + 1);
                keys.truncate(mid); // drops `promoted` itself from the left side
                let right_children = children.split_off(mid + 1);
                Some((promoted, Box::new(Node::Internal { keys: right_keys, children: right_children })))
            }
        }
    }

    /// Recursive descent with range pruning: a child is skipped only when
    /// its whole key range provably can't overlap `[min, max]`, using the
    /// separators on either side of it in `keys`.
    fn range_search(&self, min: f64, max: f64, out: &mut Vec<(f64, usize)>) {
        match self {
            Node::Leaf { keys, values } => {
                for (k, v) in keys.iter().zip(values.iter()) {
                    if *k >= min && *k <= max {
                        out.push((*k, *v));
                    }
                }
            }
            Node::Internal { keys, children } => {
                for (i, child) in children.iter().enumerate() {
                    let child_min = if i == 0 { f64::NEG_INFINITY } else { keys[i - 1] };
                    let child_max = if i == keys.len() { f64::INFINITY } else { keys[i] };
                    if child_max < min || child_min > max {
                        continue;
                    }
                    child.range_search(min, max, out);
                }
            }
        }
    }

    /// Real heap-allocated bytes this subtree occupies: each `Vec`'s
    /// allocated *capacity* (not `len` -- capacity is what's actually
    /// resident, same reasoning as everywhere else this method's siblings
    /// on `PGMIndex`/`RMIIndex` compute it) times its element size, plus
    /// this node's own heap slot (it lives inside a `Box`, so its `size_of`
    /// is real allocated memory, not stack space).
    fn heap_bytes(&self) -> usize {
        std::mem::size_of::<Node>()
            + match self {
                Node::Leaf { keys, values } => {
                    keys.capacity() * std::mem::size_of::<f64>() + values.capacity() * std::mem::size_of::<usize>()
                }
                Node::Internal { keys, children } => {
                    keys.capacity() * std::mem::size_of::<f64>()
                        + children.capacity() * std::mem::size_of::<Box<Node>>()
                        + children.iter().map(|c| c.heap_bytes()).sum::<usize>()
                }
            }
    }

    fn height(&self) -> usize {
        match self {
            Node::Leaf { .. } => 1,
            Node::Internal { children, .. } => 1 + children[0].height(),
        }
    }
}

/// Real B+Tree implementation, used as this codebase's traditional-index
/// baseline (see this module's docs for why that matters).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BTree {
    root: Box<Node>,
    /// Max keys per node before it splits; also the max children an
    /// internal node holds once full (`order` keys -> `order + 1`
    /// children before the next insert forces a split).
    order: usize,
}

impl BTree {
    /// `order` must be at least 3 -- a smaller value can't split
    /// meaningfully (a 2-key-max node splits into two 1-key nodes with
    /// nothing left to promote a search on). Real B+Trees commonly use
    /// orders in the hundreds (sized to a disk page); this one defaults to
    /// whatever the caller passes, same as before this rewrite.
    pub fn new(order: usize) -> Self {
        BTree {
            root: Box::new(Node::Leaf { keys: Vec::new(), values: Vec::new() }),
            order: order.max(3),
        }
    }

    pub fn search(&self, key: f64) -> Option<usize> {
        self.root.search(key)
    }

    pub fn insert(&mut self, key: f64, value: usize) -> Result<()> {
        if let Some((separator, right)) = self.root.insert(key, value, self.order) {
            let placeholder = Node::Leaf { keys: Vec::new(), values: Vec::new() };
            let old_root = std::mem::replace(self.root.as_mut(), placeholder);
            self.root = Box::new(Node::Internal {
                keys: vec![separator],
                children: vec![Box::new(old_root), right],
            });
        }
        Ok(())
    }

    /// All `(key, value)` pairs with `min <= key <= max`, in ascending
    /// order.
    pub fn range_search(&self, min: f64, max: f64) -> Vec<(f64, usize)> {
        let mut out = Vec::new();
        if min <= max {
            self.root.range_search(min, max, &mut out);
        }
        out
    }

    /// Real allocated heap memory this tree occupies -- see
    /// `Node::heap_bytes`'s docs for the accounting method, which
    /// `index::pgm::PGMIndex::heap_bytes`/`index::rmi::RMIIndex::heap_bytes`
    /// mirror so the three are directly comparable.
    pub fn heap_bytes(&self) -> usize {
        std::mem::size_of::<BTree>() + self.root.heap_bytes()
    }

    /// Number of levels from root to leaf, inclusive -- a single-leaf,
    /// unsplit tree has height 1.
    pub fn height(&self) -> usize {
        self.root.height()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_btree_insert_search() -> Result<()> {
        let mut btree = BTree::new(3);
        btree.insert(5.0, 10)?;
        btree.insert(3.0, 6)?;

        assert_eq!(btree.search(5.0), Some(10));
        assert_eq!(btree.search(3.0), Some(6));
        Ok(())
    }

    #[test]
    fn test_btree_actually_splits_past_order() -> Result<()> {
        // The bug this rewrite fixes: an earlier version never split at
        // all, so height stayed 1 forever regardless of how many keys
        // went in. order=3 means a 3rd key in one node forces a split.
        let mut btree = BTree::new(3);
        for k in [1.0, 2.0, 3.0, 4.0, 5.0] {
            btree.insert(k, k as usize)?;
        }
        assert!(btree.height() > 1, "tree with 5 keys at order 3 must have split at least once");
        Ok(())
    }

    #[test]
    fn test_btree_search_after_many_splits() -> Result<()> {
        let mut btree = BTree::new(4);
        let keys: Vec<f64> = (0..500).map(|i| i as f64).collect();
        for &k in &keys {
            btree.insert(k, k as usize)?;
        }
        for &k in &keys {
            assert_eq!(btree.search(k), Some(k as usize), "lookup for {k} failed after splitting");
        }
        assert_eq!(btree.search(-1.0), None);
        assert_eq!(btree.search(999.0), None);
        Ok(())
    }

    #[test]
    fn test_btree_insert_updates_existing_key_rather_than_duplicating() -> Result<()> {
        let mut btree = BTree::new(4);
        btree.insert(1.0, 100)?;
        btree.insert(1.0, 200)?;
        assert_eq!(btree.search(1.0), Some(200));
        Ok(())
    }

    #[test]
    fn test_btree_range_search_after_splits() -> Result<()> {
        let mut btree = BTree::new(4);
        for k in 0..200 {
            btree.insert(k as f64, k as usize)?;
        }
        let mut results = btree.range_search(50.0, 60.0);
        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let keys: Vec<f64> = results.iter().map(|(k, _)| *k).collect();
        assert_eq!(keys, (50..=60).map(|k| k as f64).collect::<Vec<_>>());
        Ok(())
    }

    #[test]
    fn test_btree_out_of_order_range_is_empty() -> Result<()> {
        let mut btree = BTree::new(4);
        btree.insert(1.0, 1)?;
        assert!(btree.range_search(10.0, 5.0).is_empty());
        Ok(())
    }

    #[test]
    fn test_btree_height_grows_with_more_keys() -> Result<()> {
        let mut small = BTree::new(4);
        small.insert(1.0, 1)?;
        assert_eq!(small.height(), 1);

        let mut big = BTree::new(4);
        for k in 0..1000 {
            big.insert(k as f64, k as usize)?;
        }
        assert!(big.height() > small.height(), "1000 keys must produce a taller tree than 1 key");
        Ok(())
    }

    #[test]
    fn test_btree_heap_bytes_grows_with_key_count() -> Result<()> {
        let mut small = BTree::new(32);
        small.insert(1.0, 1)?;
        let small_bytes = small.heap_bytes();

        let mut big = BTree::new(32);
        for k in 0..10_000 {
            big.insert(k as f64, k as usize)?;
        }
        let big_bytes = big.heap_bytes();

        assert!(big_bytes > small_bytes, "10,000 keys must occupy more heap than 1");
        Ok(())
    }
}
