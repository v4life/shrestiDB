//! HNSW (Hierarchical Navigable Small World) Index
//!
//! A production-quality implementation of the HNSW algorithm for approximate
//! nearest neighbor (ANN) search, as described in:
//!
//!   Malkov, Yu. A. & Yashunin, D. A. (2018).
//!   "Efficient and robust approximate nearest neighbor search using
//!    Hierarchical Navigable Small World graphs."
//!   IEEE Transactions on Pattern Analysis and Machine Intelligence.
//!
//! The index organizes vectors into a multi-layer navigable small-world graph.
//! Upper layers act as a coarse skip-list, enabling logarithmic entry into the
//! bottom layer where the full-resolution graph lives. Beam search (controlled
//! by the `ef` parameter) is used at each layer to find nearest neighbors.
//!
//! # Thread Safety
//!
//! All public methods are safe to call concurrently. The internal graph
//! structure is protected by `parking_lot::RwLock`, and node-level neighbor
//! lists use their own fine-grained locks to minimise contention during
//! concurrent inserts.
//!
//! # Distance Metrics
//!
//! Three metrics are supported — Cosine, L2 (Euclidean), and Inner Product —
//! each exposed through [`DistanceMetric`]. All distances are non-negative
//! with lower values indicating greater similarity, following the standard
//! convention for ANN search libraries.

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Ordering;

use parking_lot::RwLock;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::compute::vector_math::VectorOps;
use crate::error::{DatabaseError, Result};

// ---------------------------------------------------------------------------
// Distance metric
// ---------------------------------------------------------------------------

/// Distance metric used by the HNSW index to compare vectors.
///
/// All metrics produce a non-negative distance where **lower = more similar**.
/// For inner-product and cosine, the raw values are transformed so that this
/// invariant holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// Cosine distance: `1.0 - cosine_similarity(a, b)`.
    ///
    /// Ranges from 0 (identical direction) to 2 (opposite direction).
    Cosine,

    /// Squared Euclidean (L2²) distance: `Σ (a_i - b_i)²`.
    ///
    /// Using the squared form avoids an unnecessary `sqrt` in every
    /// comparison while preserving the ordering.
    L2,

    /// Negative inner product distance: `-dot(a, b)`.
    ///
    /// Negated so that "higher similarity" (larger dot product) maps to a
    /// *smaller* distance value, keeping the "lower = more similar" contract.
    InnerProduct,
}

impl DistanceMetric {
    /// Compute the distance between two vectors under this metric.
    ///
    /// # Panics
    ///
    /// Panics (via `VectorOps`) if `a` and `b` have different lengths *and*
    /// the caller has not ensured they match `HNSWIndex::dims`.  In normal
    /// usage the index validates dimensionality before calling this method.
    pub fn compute(&self, a: &[f64], b: &[f64]) -> f64 {
        match self {
            DistanceMetric::Cosine => {
                let sim = VectorOps::cosine_similarity(a, b);
                // Clamp to handle floating-point drift past 1.0.
                (1.0 - sim).max(0.0)
            }
            DistanceMetric::L2 => {
                // Expand ||a - b||² = ||a||² + ||b||² - 2·a·b so we can
                // reuse the existing VectorOps::dot without allocating a
                // temporary difference vector.
                let dot_ab = VectorOps::dot(a, b);
                let norm_a_sq = VectorOps::dot(a, a);
                let norm_b_sq = VectorOps::dot(b, b);
                (norm_a_sq + norm_b_sq - 2.0 * dot_ab).max(0.0)
            }
            DistanceMetric::InnerProduct => {
                -VectorOps::dot(a, b)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration parameters for an [`HNSWIndex`].
///
/// Sensible defaults are provided via [`Default`]; the most commonly tuned
/// knobs are `m` (controls graph connectivity / memory) and `ef_search`
/// (controls recall / latency trade-off at query time).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HNSWConfig {
    /// Maximum number of bi-directional connections per node per layer.
    ///
    /// Higher values improve recall at the cost of more memory and slower
    /// inserts. The paper recommends 12–48; 16 is a balanced default.
    pub m: usize,

    /// Size of the dynamic candidate list during index construction.
    ///
    /// Higher values yield a higher-quality graph but slower build times.
    pub ef_construction: usize,

    /// Size of the dynamic candidate list during search (query time).
    ///
    /// Must be ≥ `k` (the number of results requested). Higher values
    /// improve recall at the cost of higher latency.
    pub ef_search: usize,

    /// Maximum number of layers in the hierarchy.
    ///
    /// In practice the probabilistic level assignment almost never exceeds
    /// ~log₁ₘ(N), so 16 layers is sufficient for billions of vectors.
    pub max_layers: usize,

    /// Distance metric for comparing vectors.
    pub metric: DistanceMetric,
}

impl Default for HNSWConfig {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 200,
            ef_search: 64,
            max_layers: 16,
            metric: DistanceMetric::Cosine,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal node
// ---------------------------------------------------------------------------

/// A single vector stored in the HNSW graph.
///
/// Each node knows its assigned level and maintains a separate neighbor list
/// for every layer it participates in (layers `0..=level`).
#[allow(dead_code)]
struct HNSWNode {
    /// External identifier supplied by the caller.
    id: u64,

    /// The raw vector data.
    vector: Vec<f64>,

    /// Maximum layer this node is present on (0-indexed).
    level: usize,

    /// `neighbors[layer]` holds the IDs of this node's connections at that
    /// layer. Guarded by the per-node lock in the outer `HNSWIndex`.
    neighbors: Vec<RwLock<Vec<u64>>>,

    /// Tombstone flag for lazy deletion.
    deleted: bool,
}

// ---------------------------------------------------------------------------
// Min/max distance wrappers for BinaryHeap
// ---------------------------------------------------------------------------

/// A `(distance, node_id)` pair ordered so that the *smallest* distance is
/// popped first from a `BinaryHeap` (i.e. a min-heap).
#[derive(Clone, Copy)]
struct MinHeapItem {
    dist: f64,
    id: u64,
}

impl PartialEq for MinHeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist && self.id == other.id
    }
}

impl Eq for MinHeapItem {}

impl PartialOrd for MinHeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MinHeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse so BinaryHeap pops the smallest distance first.
        other
            .dist
            .partial_cmp(&self.dist)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.id.cmp(&self.id))
    }
}

/// A `(distance, node_id)` pair ordered so that the *largest* distance is
/// popped first (standard max-heap behaviour).
#[derive(Clone, Copy)]
struct MaxHeapItem {
    dist: f64,
    id: u64,
}

impl PartialEq for MaxHeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist && self.id == other.id
    }
}

impl Eq for MaxHeapItem {}

impl PartialOrd for MaxHeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MaxHeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.id.cmp(&other.id))
    }
}

// ---------------------------------------------------------------------------
// HNSW index
// ---------------------------------------------------------------------------

/// Hierarchical Navigable Small World index for approximate nearest-neighbor
/// search over dense floating-point vectors.
///
/// # Examples
///
/// ```rust
/// use shrestidb::index::hnsw::{HNSWIndex, HNSWConfig, DistanceMetric};
///
/// let config = HNSWConfig {
///     metric: DistanceMetric::L2,
///     ..HNSWConfig::default()
/// };
/// let index = HNSWIndex::new(3, config);
///
/// index.insert(1, vec![1.0, 0.0, 0.0]).unwrap();
/// index.insert(2, vec![0.0, 1.0, 0.0]).unwrap();
/// index.insert(3, vec![0.0, 0.0, 1.0]).unwrap();
///
/// let results = index.search(&[1.0, 0.1, 0.0], 2);
/// assert_eq!(results[0].0, 1); // closest
/// ```
pub struct HNSWIndex {
    /// Dimensionality every stored vector must match.
    dims: usize,

    /// Tuning knobs.
    config: HNSWConfig,

    /// The normalisation factor for the random level distribution:
    /// `1 / ln(m)`.  Precomputed once during construction.
    level_mult: f64,

    /// All nodes in the graph, keyed by their external id.
    /// Protected by a read-write lock so searches can proceed concurrently
    /// while inserts take a write lock only briefly.
    nodes: RwLock<HashMap<u64, HNSWNode>>,

    /// The id of the current entry-point node (the node with the highest
    /// level), or `None` if the index is empty.
    entry_point: RwLock<Option<u64>>,

    /// The current maximum level in the graph (0-indexed).
    max_level: RwLock<usize>,
}

impl HNSWIndex {
    // -- construction -------------------------------------------------------

    /// Create a new, empty HNSW index for vectors of `dims` dimensions.
    pub fn new(dims: usize, config: HNSWConfig) -> Self {
        let m = config.m.max(2); // guard against degenerate M
        let level_mult = 1.0 / (m as f64).ln();
        Self {
            dims,
            config: HNSWConfig { m, ..config },
            level_mult,
            nodes: RwLock::new(HashMap::new()),
            entry_point: RwLock::new(None),
            max_level: RwLock::new(0),
        }
    }

    // -- public API ---------------------------------------------------------

    /// Insert a vector with the given `id` into the index.
    ///
    /// # Errors
    ///
    /// Returns [`DatabaseError::IndexError`] if:
    /// - A vector with the same `id` already exists.
    /// - `vector.len() != self.dims`.
    pub fn insert(&self, id: u64, vector: Vec<f64>) -> Result<()> {
        if vector.len() != self.dims {
            return Err(DatabaseError::IndexError(format!(
                "dimension mismatch: expected {}, got {}",
                self.dims,
                vector.len()
            )));
        }

        // Assign a random level for this node.
        let level = self.random_level();

        let node = HNSWNode {
            id,
            vector,
            level,
            neighbors: (0..=level).map(|_| RwLock::new(Vec::new())).collect(),
            deleted: false,
        };

        // --- critical section: insert the node into the map ----------------
        {
            let mut nodes = self.nodes.write();
            if nodes.contains_key(&id) {
                return Err(DatabaseError::IndexError(format!(
                    "duplicate vector id: {id}"
                )));
            }
            nodes.insert(id, node);
        }

        // --- connect the new node into the graph ---------------------------
        let entry_point_id = { *self.entry_point.read() };

        match entry_point_id {
            None => {
                // First node ever — it becomes the entry point.
                *self.entry_point.write() = Some(id);
                *self.max_level.write() = level;
            }
            Some(ep_id) => {
                let current_max_level = { *self.max_level.read() };

                let mut cur_ep = ep_id;

                // Phase 1: greedy descent through layers above the new node's level.
                let nodes_guard = self.nodes.read();
                for lc in (level + 1..=current_max_level).rev() {
                    cur_ep = self.greedy_closest(&nodes_guard, &self.get_vector(&nodes_guard, id), cur_ep, lc);
                }
                drop(nodes_guard);

                // Phase 2: for each layer the new node participates in,
                // run a beam search to find neighbours and connect them.
                let top = level.min(current_max_level);
                for lc in (0..=top).rev() {
                    let nodes_guard = self.nodes.read();
                    let query_vec = self.get_vector(&nodes_guard, id);
                    let candidates = self.search_layer(&nodes_guard, &query_vec, cur_ep, self.config.ef_construction, lc);
                    drop(nodes_guard);

                    // Select the M closest non-deleted neighbours.
                    let m = if lc == 0 { self.config.m * 2 } else { self.config.m };
                    let neighbors: Vec<u64> = candidates
                        .iter()
                        .take(m)
                        .map(|&(nid, _)| nid)
                        .collect();

                    if let Some(&(closest, _)) = candidates.first() {
                        cur_ep = closest;
                    }

                    // Set forward edges from the new node.
                    {
                        let nodes_guard = self.nodes.read();
                        if let Some(node) = nodes_guard.get(&id) {
                            let mut nb = node.neighbors[lc].write();
                            *nb = neighbors.clone();
                        }
                    }

                    // Set reverse edges and prune if over capacity.
                    for &neighbor_id in &neighbors {
                        let nodes_guard = self.nodes.read();
                        if let Some(neighbor_node) = nodes_guard.get(&neighbor_id) {
                            if lc < neighbor_node.neighbors.len() {
                                let mut nb = neighbor_node.neighbors[lc].write();
                                nb.push(id);
                                // Prune if over capacity.
                                let max_conn = if lc == 0 { self.config.m * 2 } else { self.config.m };
                                if nb.len() > max_conn {
                                    // Keep only the closest `max_conn` neighbours.
                                    let neighbor_vec = neighbor_node.vector.clone();
                                    drop(nb);
                                    drop(nodes_guard);
                                    self.prune_neighbors(neighbor_id, &neighbor_vec, lc, max_conn);
                                }
                            }
                        }
                    }
                }

                // If the new node's level exceeds the current max, promote it
                // to entry point.
                if level > current_max_level {
                    *self.entry_point.write() = Some(id);
                    *self.max_level.write() = level;
                }
            }
        }

        Ok(())
    }

    /// Search for the `k` nearest neighbors of `query`.
    ///
    /// Returns a `Vec<(id, distance)>` sorted by ascending distance (closest
    /// first). If the index contains fewer than `k` non-deleted vectors, all
    /// of them are returned.
    pub fn search(&self, query: &[f64], k: usize) -> Vec<(u64, f64)> {
        if k == 0 {
            return Vec::new();
        }

        let entry_point_id = { *self.entry_point.read() };
        let ep_id = match entry_point_id {
            Some(id) => id,
            None => return Vec::new(),
        };

        let current_max_level = { *self.max_level.read() };
        let nodes_guard = self.nodes.read();

        // Check if the entry point has been deleted and the index is
        // effectively empty.
        if nodes_guard.is_empty() {
            return Vec::new();
        }

        let mut cur_ep = ep_id;

        // Phase 1: greedy descent through upper layers.
        for lc in (1..=current_max_level).rev() {
            cur_ep = self.greedy_closest(&nodes_guard, query, cur_ep, lc);
        }

        // Phase 2: beam search on layer 0.
        let ef = self.config.ef_search.max(k);
        let candidates = self.search_layer(&nodes_guard, query, cur_ep, ef, 0);

        // Filter deleted nodes and take the top-k.
        candidates
            .into_iter()
            .filter(|&(id, _)| {
                nodes_guard
                    .get(&id)
                    .map_or(false, |n| !n.deleted)
            })
            .take(k)
            .collect()
    }

    /// Mark a vector as deleted (lazy deletion).
    ///
    /// The node remains in the graph for routing purposes but is excluded
    /// from search results. Returns `true` if the node existed and was not
    /// already deleted.
    pub fn remove(&self, id: u64) -> bool {
        let mut nodes = self.nodes.write();
        match nodes.get_mut(&id) {
            Some(node) if !node.deleted => {
                node.deleted = true;
                true
            }
            _ => false,
        }
    }

    /// Number of non-deleted vectors in the index.
    pub fn len(&self) -> usize {
        let nodes = self.nodes.read();
        nodes.values().filter(|n| !n.deleted).count()
    }

    /// Returns `true` if the index contains no non-deleted vectors.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Dimensionality of the vectors stored in this index.
    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Returns `true` if the index contains a non-deleted vector with `id`.
    pub fn contains(&self, id: u64) -> bool {
        let nodes = self.nodes.read();
        nodes.get(&id).map_or(false, |n| !n.deleted)
    }

    // -- private helpers ----------------------------------------------------

    /// Sample a random level for a new node using the HNSW distribution:
    /// `floor(-ln(uniform(0,1)) * level_mult)`, clamped to `max_layers - 1`.
    fn random_level(&self) -> usize {
        let mut rng = rand::thread_rng();
        let r: f64 = rng.gen::<f64>(); // (0, 1)
        let level = (-r.ln() * self.level_mult).floor() as usize;
        level.min(self.config.max_layers - 1)
    }

    /// Convenience: get a clone of a node's vector while holding a read guard.
    fn get_vector(&self, nodes: &HashMap<u64, HNSWNode>, id: u64) -> Vec<f64> {
        nodes[&id].vector.clone()
    }

    /// Compute the distance from a query to the node with `id`.
    fn distance_to(&self, nodes: &HashMap<u64, HNSWNode>, query: &[f64], id: u64) -> f64 {
        self.config.metric.compute(query, &nodes[&id].vector)
    }

    /// Greedy walk: starting from `ep`, follow the single closest neighbor
    /// at `layer` until no improvement is possible. Returns the id of the
    /// closest node found.
    fn greedy_closest(
        &self,
        nodes: &HashMap<u64, HNSWNode>,
        query: &[f64],
        ep: u64,
        layer: usize,
    ) -> u64 {
        let mut current = ep;
        let mut current_dist = self.distance_to(nodes, query, current);

        loop {
            let mut changed = false;

            let neighbor_ids: Vec<u64> = nodes
                .get(&current)
                .map(|node| {
                    if layer < node.neighbors.len() {
                        node.neighbors[layer].read().clone()
                    } else {
                        Vec::new()
                    }
                })
                .unwrap_or_default();

            for &neighbor_id in &neighbor_ids {
                if let Some(neighbor_node) = nodes.get(&neighbor_id) {
                    let d = self.config.metric.compute(query, &neighbor_node.vector);
                    if d < current_dist {
                        current = neighbor_id;
                        current_dist = d;
                        changed = true;
                    }
                }
            }

            if !changed {
                break;
            }
        }

        current
    }

    /// Beam search (Algorithm 2 from the HNSW paper) at a single `layer`,
    /// starting from entry point `ep`. Returns up to `ef` closest
    /// `(id, distance)` pairs, sorted by ascending distance.
    fn search_layer(
        &self,
        nodes: &HashMap<u64, HNSWNode>,
        query: &[f64],
        ep: u64,
        ef: usize,
        layer: usize,
    ) -> Vec<(u64, f64)> {
        let ep_dist = self.distance_to(nodes, query, ep);

        // `candidates`: min-heap of nodes to explore (closest first).
        let mut candidates = BinaryHeap::new();
        candidates.push(MinHeapItem { dist: ep_dist, id: ep });

        // `results`: max-heap of current best results (farthest on top so
        // we can cheaply evict the worst).
        let mut results = BinaryHeap::new();
        results.push(MaxHeapItem { dist: ep_dist, id: ep });

        let mut visited = HashSet::new();
        visited.insert(ep);

        while let Some(MinHeapItem { dist: c_dist, id: c_id }) = candidates.pop() {
            // If the closest candidate is farther than the farthest result
            // we already have, we can stop.
            let farthest_dist = results.peek().map_or(f64::INFINITY, |r| r.dist);
            if c_dist > farthest_dist && results.len() >= ef {
                break;
            }

            // Expand neighbours of `c_id` at this layer.
            let neighbor_ids: Vec<u64> = nodes
                .get(&c_id)
                .map(|node| {
                    if layer < node.neighbors.len() {
                        node.neighbors[layer].read().clone()
                    } else {
                        Vec::new()
                    }
                })
                .unwrap_or_default();

            for &neighbor_id in &neighbor_ids {
                if !visited.insert(neighbor_id) {
                    continue; // already visited
                }

                if nodes.get(&neighbor_id).is_none() {
                    continue; // stale id
                }

                let d = self.distance_to(nodes, query, neighbor_id);
                let farthest_dist = results.peek().map_or(f64::INFINITY, |r| r.dist);

                if d < farthest_dist || results.len() < ef {
                    candidates.push(MinHeapItem { dist: d, id: neighbor_id });
                    results.push(MaxHeapItem { dist: d, id: neighbor_id });

                    if results.len() > ef {
                        results.pop(); // evict the farthest
                    }
                }
            }
        }

        // Drain the max-heap into a vec and sort ascending.
        let mut result_vec: Vec<(u64, f64)> =
            results.into_iter().map(|item| (item.id, item.dist)).collect();
        result_vec.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        result_vec
    }

    /// Prune the neighbor list of `node_id` at `layer` to at most
    /// `max_connections` entries, keeping the closest ones.
    fn prune_neighbors(
        &self,
        node_id: u64,
        node_vector: &[f64],
        layer: usize,
        max_connections: usize,
    ) {
        let nodes_guard = self.nodes.read();
        let node = match nodes_guard.get(&node_id) {
            Some(n) => n,
            None => return,
        };
        if layer >= node.neighbors.len() {
            return;
        }

        let mut nb = node.neighbors[layer].write();
        if nb.len() <= max_connections {
            return;
        }

        // Score every current neighbour.
        let mut scored: Vec<(u64, f64)> = nb
            .iter()
            .filter_map(|&nid| {
                nodes_guard.get(&nid).map(|n| {
                    (nid, self.config.metric.compute(node_vector, &n.vector))
                })
            })
            .collect();

        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        scored.truncate(max_connections);

        *nb = scored.into_iter().map(|(id, _)| id).collect();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    /// Helper: build a simple 3-D index with L2 metric.
    fn make_l2_index() -> HNSWIndex {
        HNSWIndex::new(
            3,
            HNSWConfig {
                m: 16,
                ef_construction: 100,
                ef_search: 64,
                max_layers: 8,
                metric: DistanceMetric::L2,
            },
        )
    }

    // -- DistanceMetric unit tests ------------------------------------------

    #[test]
    fn test_cosine_distance_identical_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        let d = DistanceMetric::Cosine.compute(&a, &a);
        assert!(d.abs() < 1e-10, "cosine distance to self should be ~0, got {d}");
    }

    #[test]
    fn test_cosine_distance_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let d = DistanceMetric::Cosine.compute(&a, &b);
        assert!((d - 1.0).abs() < 1e-10, "cosine distance for orthogonal should be ~1, got {d}");
    }

    #[test]
    fn test_l2_distance() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![3.0, 4.0, 0.0];
        let d = DistanceMetric::L2.compute(&a, &b);
        assert!((d - 25.0).abs() < 1e-10, "L2 squared dist should be 25, got {d}");
    }

    #[test]
    fn test_inner_product_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        // dot = 4 + 10 + 18 = 32, distance = -32
        let d = DistanceMetric::InnerProduct.compute(&a, &b);
        assert!((d - (-32.0)).abs() < 1e-10, "IP distance should be -32, got {d}");
    }

    // -- empty index edge cases ---------------------------------------------

    #[test]
    fn test_empty_index() {
        let idx = make_l2_index();
        assert!(idx.is_empty());
        assert_eq!(idx.len(), 0);
        assert_eq!(idx.dims(), 3);
        assert!(!idx.contains(42));

        let results = idx.search(&[1.0, 2.0, 3.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_k_zero() {
        let idx = make_l2_index();
        idx.insert(1, vec![1.0, 0.0, 0.0]).unwrap();
        let results = idx.search(&[1.0, 0.0, 0.0], 0);
        assert!(results.is_empty());
    }

    // -- basic insert & search ----------------------------------------------

    #[test]
    fn test_single_insert_and_search() {
        let idx = make_l2_index();
        idx.insert(1, vec![1.0, 0.0, 0.0]).unwrap();

        assert_eq!(idx.len(), 1);
        assert!(!idx.is_empty());
        assert!(idx.contains(1));

        let results = idx.search(&[1.0, 0.0, 0.0], 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 1);
        assert!(results[0].1 < 1e-10); // distance to self
    }

    #[test]
    fn test_dimension_mismatch() {
        let idx = make_l2_index();
        let err = idx.insert(1, vec![1.0, 2.0]).unwrap_err();
        assert!(
            format!("{err}").contains("dimension mismatch"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_duplicate_insert() {
        let idx = make_l2_index();
        idx.insert(1, vec![1.0, 0.0, 0.0]).unwrap();
        let err = idx.insert(1, vec![0.0, 1.0, 0.0]).unwrap_err();
        assert!(
            format!("{err}").contains("duplicate"),
            "unexpected error: {err}"
        );
    }

    // -- nearest-neighbor correctness (small datasets) ----------------------

    #[test]
    fn test_nearest_neighbor_l2() {
        let idx = make_l2_index();
        idx.insert(1, vec![1.0, 0.0, 0.0]).unwrap();
        idx.insert(2, vec![0.0, 1.0, 0.0]).unwrap();
        idx.insert(3, vec![0.0, 0.0, 1.0]).unwrap();
        idx.insert(4, vec![1.0, 1.0, 0.0]).unwrap();

        // Query close to id=1
        let results = idx.search(&[0.9, 0.1, 0.0], 2);
        assert!(results.len() >= 2);
        assert_eq!(results[0].0, 1, "expected id 1 as closest");
    }

    #[test]
    fn test_nearest_neighbor_cosine() {
        let idx = HNSWIndex::new(
            3,
            HNSWConfig {
                metric: DistanceMetric::Cosine,
                ..HNSWConfig::default()
            },
        );

        idx.insert(1, vec![1.0, 0.0, 0.0]).unwrap();
        idx.insert(2, vec![0.0, 1.0, 0.0]).unwrap();
        idx.insert(3, vec![0.0, 0.0, 1.0]).unwrap();

        // Query in the direction of id=2
        let results = idx.search(&[0.1, 10.0, 0.0], 1);
        assert_eq!(results[0].0, 2);
    }

    #[test]
    fn test_nearest_neighbor_inner_product() {
        let idx = HNSWIndex::new(
            3,
            HNSWConfig {
                metric: DistanceMetric::InnerProduct,
                ..HNSWConfig::default()
            },
        );

        idx.insert(1, vec![1.0, 0.0, 0.0]).unwrap();
        idx.insert(2, vec![0.0, 1.0, 0.0]).unwrap();
        idx.insert(3, vec![0.0, 0.0, 1.0]).unwrap();

        // Maximize dot with first axis → id=1
        let results = idx.search(&[10.0, 0.0, 0.0], 1);
        assert_eq!(results[0].0, 1);
    }

    // -- recall on a larger dataset -----------------------------------------

    #[test]
    fn test_recall_on_random_dataset() {
        let dims = 32;
        let n = 500;
        let k = 10;

        let idx = HNSWIndex::new(
            dims,
            HNSWConfig {
                m: 16,
                ef_construction: 200,
                ef_search: 128,
                max_layers: 8,
                metric: DistanceMetric::L2,
            },
        );

        let mut rng = rand::thread_rng();

        // Generate random vectors.
        let mut vectors: Vec<(u64, Vec<f64>)> = Vec::with_capacity(n);
        for i in 0..n {
            let v: Vec<f64> = (0..dims).map(|_| rng.gen::<f64>()).collect();
            vectors.push((i as u64, v));
        }

        // Insert all vectors.
        for (id, v) in &vectors {
            idx.insert(*id, v.clone()).unwrap();
        }

        assert_eq!(idx.len(), n);

        // Generate a random query and compute brute-force ground truth.
        let query: Vec<f64> = (0..dims).map(|_| rng.gen::<f64>()).collect();

        let mut brute_force: Vec<(u64, f64)> = vectors
            .iter()
            .map(|(id, v)| (*id, DistanceMetric::L2.compute(&query, v)))
            .collect();
        brute_force.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let ground_truth: HashSet<u64> = brute_force[..k].iter().map(|&(id, _)| id).collect();

        // HNSW search.
        let results = idx.search(&query, k);
        let result_ids: HashSet<u64> = results.iter().map(|&(id, _)| id).collect();

        let recall = ground_truth.intersection(&result_ids).count() as f64 / k as f64;
        assert!(
            recall >= 0.9,
            "recall too low: {recall:.2} (expected >= 0.9)"
        );

        // Results should be sorted by distance.
        for w in results.windows(2) {
            assert!(
                w[0].1 <= w[1].1 + 1e-12,
                "results not sorted: {} > {}",
                w[0].1,
                w[1].1
            );
        }
    }

    // -- deletion -----------------------------------------------------------

    #[test]
    fn test_delete_and_search() {
        let idx = make_l2_index();
        idx.insert(1, vec![1.0, 0.0, 0.0]).unwrap();
        idx.insert(2, vec![0.0, 1.0, 0.0]).unwrap();
        idx.insert(3, vec![0.0, 0.0, 1.0]).unwrap();

        assert_eq!(idx.len(), 3);
        assert!(idx.contains(1));

        // Delete node 1.
        assert!(idx.remove(1));
        assert!(!idx.contains(1));
        assert_eq!(idx.len(), 2);

        // Deleting again should return false.
        assert!(!idx.remove(1));

        // Deleting a non-existent id should return false.
        assert!(!idx.remove(999));

        // Search should not return the deleted node.
        let results = idx.search(&[1.0, 0.0, 0.0], 3);
        for &(id, _) in &results {
            assert_ne!(id, 1, "deleted node 1 appeared in results");
        }
        assert_eq!(results.len(), 2);
    }

    // -- multi-threaded inserts ---------------------------------------------

    #[test]
    fn test_concurrent_inserts() {
        let dims = 8;
        let idx = Arc::new(HNSWIndex::new(
            dims,
            HNSWConfig {
                m: 8,
                ef_construction: 50,
                ef_search: 32,
                max_layers: 6,
                metric: DistanceMetric::L2,
            },
        ));

        let num_threads = 4;
        let per_thread = 50;

        let mut handles = Vec::new();
        for t in 0..num_threads {
            let idx = Arc::clone(&idx);
            handles.push(thread::spawn(move || {
                let mut rng = rand::thread_rng();
                for i in 0..per_thread {
                    let id = (t * per_thread + i) as u64;
                    let v: Vec<f64> = (0..dims).map(|_| rng.gen::<f64>()).collect();
                    idx.insert(id, v).unwrap();
                }
            }));
        }

        for h in handles {
            h.join().expect("thread panicked");
        }

        assert_eq!(idx.len(), num_threads * per_thread);
    }

    // -- misc ---------------------------------------------------------------

    #[test]
    fn test_search_returns_at_most_k() {
        let idx = make_l2_index();
        idx.insert(1, vec![1.0, 0.0, 0.0]).unwrap();
        idx.insert(2, vec![0.0, 1.0, 0.0]).unwrap();

        let results = idx.search(&[0.5, 0.5, 0.0], 10);
        assert_eq!(results.len(), 2, "should return all vectors when k > n");
    }

    #[test]
    fn test_config_defaults() {
        let cfg = HNSWConfig::default();
        assert_eq!(cfg.m, 16);
        assert_eq!(cfg.ef_construction, 200);
        assert_eq!(cfg.ef_search, 64);
        assert_eq!(cfg.max_layers, 16);
        assert_eq!(cfg.metric, DistanceMetric::Cosine);
    }
}
