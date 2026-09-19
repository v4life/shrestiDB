//! Piecewise Geometric Model (PGM) Index
//!
//! Partitions sorted keys into segments, each covered by a linear model.
//! Provides both static (PGMIndex) and dynamic (DynamicPGMIndex) structures.

use crate::compute::simd_ops::SIMDSearch;
use crate::index::models::LinearModel;
use serde::{Deserialize, Serialize};

/// Segment in a PGM index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PGMSegment {
    pub model: LinearModel,
    pub start_key: f64,
    pub end_key: f64,
    pub start_pos: usize,
    pub end_pos: usize,
}

impl PGMSegment {
    pub fn new(
        model: LinearModel,
        start_key: f64,
        end_key: f64,
        start_pos: usize,
        end_pos: usize,
    ) -> Self {
        PGMSegment {
            model,
            start_key,
            end_key,
            start_pos,
            end_pos,
        }
    }

    pub fn predict_position(&self, key: f64) -> usize {
        let predicted = self.model.predict(key) as usize;
        predicted.clamp(self.start_pos, self.end_pos)
    }
}

/// Piecewise Geometric Model Index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PGMIndex {
    pub segments: Vec<PGMSegment>,
    pub keys: Vec<f64>,
    pub error_bound: usize,
}

impl PGMIndex {
    pub fn new(segments: Vec<PGMSegment>, keys: Vec<f64>, error_bound: usize) -> Self {
        PGMIndex {
            segments,
            keys,
            error_bound,
        }
    }

    /// Build PGM index from sorted keys in O(n) time.
    ///
    /// Streams through the keys once, keeping a fixed anchor at the start of the
    /// current segment and maintaining the interval of slopes still consistent
    /// with every point seen so far (each new point only tightens the interval,
    /// since the anchor never moves). When the interval goes empty, the segment
    /// is closed with the midpoint slope and a new segment starts at that key.
    /// This avoids refitting OLS over the growing segment on every candidate
    /// extension, which is what made the previous version quadratic.
    pub fn build(keys: Vec<f64>, error_bound: usize) -> Self {
        let n = keys.len();

        if n == 0 {
            return PGMIndex {
                segments: Vec::new(),
                keys,
                error_bound,
            };
        }

        let eps = error_bound as f64;
        let mut segments = Vec::new();
        let mut seg_start = 0usize;
        let mut min_slope = f64::NEG_INFINITY;
        let mut max_slope = f64::INFINITY;

        let close_segment = |seg_start: usize,
                              seg_end: usize,
                              min_slope: f64,
                              max_slope: f64,
                              keys: &[f64]| {
            let slope = if min_slope.is_finite() && max_slope.is_finite() {
                (min_slope + max_slope) / 2.0
            } else {
                0.0
            };
            let origin_x = keys[seg_start];
            let origin_y = seg_start as f64;
            let model = LinearModel::new(slope, origin_y - slope * origin_x);
            PGMSegment::new(model, keys[seg_start], keys[seg_end], seg_start, seg_end)
        };

        for i in 1..n {
            let dx = keys[i] - keys[seg_start];
            let dy = (i - seg_start) as f64;

            if dx == 0.0 {
                // Duplicate key: predicted position is always the anchor's
                // position regardless of slope, so it only fits if dy <= eps.
                if dy > eps {
                    segments.push(close_segment(seg_start, i - 1, min_slope, max_slope, &keys));
                    seg_start = i;
                    min_slope = f64::NEG_INFINITY;
                    max_slope = f64::INFINITY;
                }
                continue;
            }

            let s_lower = (dy - eps) / dx;
            let s_upper = (dy + eps) / dx;
            let new_min = min_slope.max(s_lower);
            let new_max = max_slope.min(s_upper);

            if new_min > new_max {
                segments.push(close_segment(seg_start, i - 1, min_slope, max_slope, &keys));
                seg_start = i;
                min_slope = f64::NEG_INFINITY;
                max_slope = f64::INFINITY;
            } else {
                min_slope = new_min;
                max_slope = new_max;
            }
        }

        segments.push(close_segment(seg_start, n - 1, min_slope, max_slope, &keys));

        PGMIndex {
            segments,
            keys,
            error_bound,
        }
    }

    /// Find segment covering the given key using binary search over segment start keys
    pub fn find_segment(&self, key: f64) -> Option<&PGMSegment> {
        if self.segments.is_empty() {
            return None;
        }

        let seg_idx = match self
            .segments
            .binary_search_by(|s| s.start_key.partial_cmp(&key).unwrap())
        {
            Ok(idx) => idx,
            Err(0) => 0,
            Err(idx) => idx - 1,
        };

        let seg = &self.segments[seg_idx];
        if key >= seg.start_key && key <= seg.end_key {
            Some(seg)
        } else {
            // Check next segment if within precision boundary
            if seg_idx + 1 < self.segments.len() {
                let next_seg = &self.segments[seg_idx + 1];
                if key >= next_seg.start_key && key <= next_seg.end_key {
                    return Some(next_seg);
                }
            }
            None
        }
    }

    /// Search using PGM segments and SIMD-accelerated bounded search
    pub fn search(&self, key: f64) -> Option<usize> {
        let segment = self.find_segment(key)?;
        let predicted_pos = segment.predict_position(key);

        SIMDSearch::bounded_search(&self.keys, key, predicted_pos, self.error_bound)
    }

    /// An estimate of how many build-time keys are `<= key`, whether or
    /// not `key` was actually present in that data — unlike `search`,
    /// which only succeeds for an exact match (a `key` absent from the
    /// data returns `None`, discarding the very position `find_segment`'s
    /// linear model already computed for it). Meant for CDF/selectivity
    /// estimation (`optimizer::cardinality::ColumnDistribution`), not
    /// point lookup: `error_bound`'s accuracy guarantee only covers keys
    /// that were actually in the build set, so for any other key this is
    /// a genuine estimate, not the bounded one `search` gives, and can be
    /// off by roughly a segment's worth of rows even for a key close to
    /// one that was in the data.
    ///
    /// Always in `[0, keys.len()]`: `0` means "at or before everything",
    /// `keys.len()` means "at or after everything" — so
    /// `predicted_rank(key) as f64 / keys.len() as f64` is directly a
    /// selectivity estimate for `<= key`.
    pub fn predicted_rank(&self, key: f64) -> usize {
        if self.keys.is_empty() {
            return 0;
        }
        if key < self.keys[0] {
            return 0;
        }
        if key >= *self.keys.last().unwrap() {
            return self.keys.len();
        }

        match self.find_segment(key) {
            Some(segment) => segment.predict_position(key),
            // key falls in a real gap between two segments' covered
            // ranges -- no data exists at exactly this value, and it's
            // not before/after everything either (that's handled above).
            // The rank is the position right after whichever segment
            // precedes it: every one of that segment's keys is <= key,
            // none of the next segment's are.
            None => match self.segments.binary_search_by(|s| s.start_key.partial_cmp(&key).unwrap()) {
                Ok(idx) => self.segments[idx].start_pos,
                Err(0) => 0,
                Err(idx) => self.segments[idx - 1].end_pos + 1,
            },
        }
    }

    /// Range search returning positions of all keys in [min_key, max_key]
    pub fn range_search(&self, min_key: f64, max_key: f64) -> Vec<usize> {
        if self.keys.is_empty() || min_key > max_key {
            return Vec::new();
        }

        let start_pos = match self.search(min_key) {
            Some(pos) => pos,
            None => {
                match self.keys.binary_search_by(|k| k.partial_cmp(&min_key).unwrap()) {
                    Ok(idx) => idx,
                    Err(idx) => idx,
                }
            }
        };

        let mut results = Vec::new();
        let mut curr = start_pos;
        while curr < self.keys.len() && self.keys[curr] <= max_key {
            results.push(curr);
            curr += 1;
        }

        results
    }

    /// Real allocated heap memory this index occupies: each `Vec`'s
    /// allocated *capacity* (not `len` -- capacity is what's actually
    /// resident) times its element size. `PGMSegment` stores a `LinearModel`
    /// (2 `f64`s) plus two more `f64`s and two `usize`s per segment,
    /// regardless of how many keys that segment covers -- the actual
    /// mechanism behind PGM's real memory advantage over a B-Tree: a run of
    /// N keys that fits one linear model within `error_bound` costs one
    /// fixed-size segment no matter how large N is, where a B-Tree's node
    /// count scales with N directly. Mirrored by
    /// `index::btree::BTree::heap_bytes`/`index::rmi::RMIIndex::heap_bytes`
    /// so the three are directly, consistently comparable.
    pub fn heap_bytes(&self) -> usize {
        std::mem::size_of::<PGMIndex>()
            + self.segments.capacity() * std::mem::size_of::<PGMSegment>()
            + self.keys.capacity() * std::mem::size_of::<f64>()
    }

    /// Insert a key and rebuild index
    pub fn insert_with_rebuild(&mut self, key: f64) {
        let insert_idx = match self.keys.binary_search_by(|k| k.partial_cmp(&key).unwrap()) {
            Ok(idx) => idx,
            Err(idx) => idx,
        };
        self.keys.insert(insert_idx, key);
        *self = Self::build(self.keys.clone(), self.error_bound);
    }
}

/// Dynamic PGM Index supporting high-throughput insertions without immediate full rebuilds
///
/// `insert` used to keep `write_buffer` sorted at all times (a
/// `binary_search` to find the insertion point, then `Vec::insert` to
/// shift everything after it into place) and `flush_buffer` merged it
/// into `base` and called `PGMIndex::build` — an O(`base.len()`) full
/// rebuild — every time the buffer reached a *fixed* `buffer_capacity`
/// (64, set by every real caller — see `execution::mvcc_store::MVCCTable`,
/// the live PK index every `INSERT` goes through). With a fixed
/// threshold, the number of flushes for `n` total inserts is `n / 64`,
/// each costing `O(current base size)` — total bulk-insert cost
/// `O(n^2 / 64)`. Real and measured: `examples/monotonic_ingestion_niche.rs`
/// showed real SQL insert throughput dropping from ~108,000 rows/sec at
/// 20,000 rows to ~19,700 rows/sec at 500,000 rows on identical per-row
/// work, purely from this.
///
/// Fixed two ways together (fixing only one reintroduces the other's
/// cost, see each field's docs):
/// 1. `write_buffer` is now unsorted between flushes — `insert` just
///    appends (amortized O(1), same as any `Vec::push`) instead of
///    finding-and-shifting into sorted position. It's sorted once, in
///    `flush_buffer`, right before the merge that already needed sorted
///    input.
/// 2. The flush threshold now grows with `base`'s size instead of
///    staying fixed at its construction-time value (still the *floor*,
///    for a small or fresh index) — see `buffer_capacity`'s docs. Fewer,
///    larger flushes as the index grows is the same amortized argument
///    behind `Vec`'s own geometric growth: total flush cost across `n`
///    inserts becomes `O(n)`, not `O(n^2)`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicPGMIndex {
    /// Immutable/base PGM index
    pub base: PGMIndex,
    /// Write buffer (L0) absorbing writes — unsorted between flushes (see
    /// this struct's docs); sorted once, in `flush_buffer`, immediately
    /// before it's needed sorted for the merge.
    pub write_buffer: Vec<f64>,
    /// Current capacity of `write_buffer` before triggering a flush.
    /// Recomputed after every flush as `max(min_buffer_capacity,
    /// base.keys.len() / BUFFER_GROWTH_DIVISOR)` — grows with the index
    /// instead of staying fixed, so flush frequency drops as `base`
    /// grows (see this struct's docs for why a fixed threshold makes
    /// bulk insertion quadratic). Bounded well below `base.keys.len()`
    /// itself (not doubling-style growth, which would let the buffer —
    /// and therefore the linear-scan cost `contains`/`range_search` pay
    /// against it — grow to a large fraction of the whole index) so read
    /// latency doesn't pay for faster bulk writes.
    pub buffer_capacity: usize,
    /// The smallest `buffer_capacity` is ever allowed to shrink back to —
    /// fixed at whatever `new` was constructed with, so a small or
    /// freshly-created index doesn't flush after every single insert
    /// just because `base` is still tiny.
    min_buffer_capacity: usize,
    /// Error bound for segments
    pub error_bound: usize,
}

/// `buffer_capacity` grows to `base.len() / BUFFER_GROWTH_DIVISOR` after
/// each flush — the buffer never exceeds roughly this fraction of the
/// index's total size, keeping `contains`/`range_search`'s linear scan
/// over it cheap relative to the whole index even as bulk inserts get
/// the full benefit of needing far fewer, larger flushes.
const BUFFER_GROWTH_DIVISOR: usize = 8;

impl DynamicPGMIndex {
    /// Create a new dynamic PGM index. `buffer_capacity` is the initial —
    /// and minimum — flush threshold; see `DynamicPGMIndex::buffer_capacity`'s
    /// docs for how it grows from here.
    pub fn new(keys: Vec<f64>, error_bound: usize, buffer_capacity: usize) -> Self {
        let base = PGMIndex::build(keys, error_bound);
        DynamicPGMIndex {
            base,
            write_buffer: Vec::with_capacity(buffer_capacity),
            buffer_capacity,
            min_buffer_capacity: buffer_capacity,
            error_bound,
        }
    }

    /// Insert key into the dynamic index. Appends to `write_buffer`
    /// unsorted (amortized O(1) — see this struct's docs for why this
    /// changed from a sorted insert) and flushes once the buffer reaches
    /// its current (grown, not fixed) capacity.
    pub fn insert(&mut self, key: f64) {
        self.write_buffer.push(key);

        if self.write_buffer.len() >= self.buffer_capacity {
            self.flush_buffer();
        }
    }

    /// Flush write buffer by merging with base keys and rebuilding segments
    pub fn flush_buffer(&mut self) {
        if self.write_buffer.is_empty() {
            return;
        }

        // write_buffer is unsorted between flushes (see this struct's
        // docs) -- the merge below needs both inputs sorted, and this is
        // the one place that's actually required, not on every insert.
        self.write_buffer.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Merge two sorted vectors in O(N + M)
        let mut merged = Vec::with_capacity(self.base.keys.len() + self.write_buffer.len());
        let mut i = 0;
        let mut j = 0;

        while i < self.base.keys.len() && j < self.write_buffer.len() {
            if self.base.keys[i] <= self.write_buffer[j] {
                merged.push(self.base.keys[i]);
                i += 1;
            } else {
                merged.push(self.write_buffer[j]);
                j += 1;
            }
        }

        while i < self.base.keys.len() {
            merged.push(self.base.keys[i]);
            i += 1;
        }

        while j < self.write_buffer.len() {
            merged.push(self.write_buffer[j]);
            j += 1;
        }

        self.write_buffer.clear();
        self.base = PGMIndex::build(merged, self.error_bound);
        self.buffer_capacity = (self.base.keys.len() / BUFFER_GROWTH_DIVISOR).max(self.min_buffer_capacity);
    }

    /// Check if key exists in either write buffer or base PGM. A linear
    /// scan over `write_buffer` now, not a `binary_search` -- it's
    /// unsorted between flushes (see this struct's docs), and bounded by
    /// `buffer_capacity`'s growth policy to stay a small fraction of the
    /// index's total size, not the whole thing.
    pub fn contains(&self, key: f64) -> bool {
        if self.write_buffer.iter().any(|&k| k == key) {
            return true;
        }
        self.base.search(key).is_some()
    }

    /// Range search across write buffer and base PGM. Already did its own
    /// linear filter over `write_buffer` plus a final sort of the
    /// combined results even before `write_buffer` became unsorted
    /// between flushes (see this struct's docs) -- unaffected by that
    /// change, unlike `contains`, which used to rely on it being sorted.
    pub fn range_search(&self, min_key: f64, max_key: f64) -> Vec<f64> {
        let mut results = Vec::new();

        // From base PGM
        let base_indices = self.base.range_search(min_key, max_key);
        for idx in base_indices {
            results.push(self.base.keys[idx]);
        }

        // From write buffer
        for &k in &self.write_buffer {
            if k >= min_key && k <= max_key {
                results.push(k);
            }
        }

        results.sort_by(|a, b| a.partial_cmp(b).unwrap());
        results
    }

    /// Total number of keys
    pub fn len(&self) -> usize {
        self.base.keys.len() + self.write_buffer.len()
    }

    /// Whether index is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Real allocated heap memory: the base `PGMIndex` (`base.heap_bytes()`
    /// already counts `base`'s own struct size, so it isn't repeated here)
    /// plus this struct's two `usize` fields and the write buffer's
    /// allocated capacity. The write buffer holds raw `f64` keys (no
    /// segments yet -- it's flushed and re-fit into `base` at
    /// `buffer_capacity`), so a dynamic index with a large pending buffer
    /// will (correctly) look less space-efficient than its base alone
    /// until the next flush.
    pub fn heap_bytes(&self) -> usize {
        self.base.heap_bytes()
            + std::mem::size_of::<usize>() * 2 // buffer_capacity, error_bound
            + self.write_buffer.capacity() * std::mem::size_of::<f64>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pgm_build() {
        let keys = vec![1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0];
        let pgm = PGMIndex::build(keys, 1);
        assert!(!pgm.segments.is_empty());
    }

    #[test]
    fn test_pgm_search() {
        let keys = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let pgm = PGMIndex::build(keys, 1);
        let result = pgm.search(3.0);
        assert_eq!(result, Some(2));
    }

    #[test]
    fn test_predicted_rank_on_present_key_is_close_to_real_index() {
        let keys = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let pgm = PGMIndex::build(keys, 1);
        // key=5.0 is the 5th element (0-indexed position 4) -- an
        // in-data key's rank should land within the error bound of its
        // real position, same guarantee `search` gives for exact match.
        let rank = pgm.predicted_rank(5.0);
        assert!((rank as i64 - 4).abs() <= 1, "rank {rank} should be close to real index 4");
    }

    #[test]
    fn test_predicted_rank_between_real_values_falls_between_their_ranks() {
        let keys = vec![10.0, 20.0, 30.0, 80.0, 90.0, 100.0];
        let pgm = PGMIndex::build(keys, 1);
        // 50.0 was never in the data (a real gap between 30 and 80) --
        // its rank must still land between the ranks of its neighbors,
        // not be thrown away the way search() would (None).
        let rank_30 = pgm.predicted_rank(30.0);
        let rank_50 = pgm.predicted_rank(50.0);
        let rank_80 = pgm.predicted_rank(80.0);
        assert!(rank_30 <= rank_50, "rank(30)={rank_30} should be <= rank(50)={rank_50}");
        assert!(rank_50 <= rank_80, "rank(50)={rank_50} should be <= rank(80)={rank_80}");
    }

    #[test]
    fn test_predicted_rank_before_and_after_all_data() {
        let keys = vec![10.0, 20.0, 30.0];
        let pgm = PGMIndex::build(keys, 1);
        assert_eq!(pgm.predicted_rank(0.0), 0);
        assert_eq!(pgm.predicted_rank(100.0), 3);
    }

    #[test]
    fn test_predicted_rank_on_heavily_duplicated_column() {
        // A "status"-column shape: a handful of distinct values, each
        // repeated many times -- exactly the case duplicate handling in
        // build() exists for (pgm.rs's close_segment forcing a new
        // segment when a duplicate run's spread exceeds error_bound).
        let mut keys = vec![0.0; 950]; // "shipped"
        keys.extend(vec![1.0; 50]); // "cancelled"
        let pgm = PGMIndex::build(keys, 4);

        // Before the only two distinct values: rank 0.
        assert_eq!(pgm.predicted_rank(-1.0), 0);
        // At or past the last (largest) value: rank is the full count.
        assert_eq!(pgm.predicted_rank(1.0), 1000);
        // Strictly between the two distinct values (never present):
        // every "shipped" row is <= it, no "cancelled" row is.
        let rank_between = pgm.predicted_rank(0.5);
        assert!(
            (rank_between as i64 - 950).abs() <= 4,
            "rank(0.5)={rank_between} should be close to the 950 'shipped' rows"
        );
    }

    #[test]
    fn test_pgm_range_search() {
        let keys = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let pgm = PGMIndex::build(keys, 2);
        let range = pgm.range_search(20.0, 45.0);
        assert_eq!(range, vec![1, 2, 3]);
    }

    #[test]
    fn test_dynamic_pgm_insert_and_contains() {
        let initial_keys = vec![10.0, 20.0, 30.0];
        let mut dpgm = DynamicPGMIndex::new(initial_keys, 2, 3);

        // Insert into write buffer
        dpgm.insert(15.0);
        dpgm.insert(25.0);
        assert!(dpgm.contains(15.0));
        assert!(dpgm.contains(20.0));
        assert!(dpgm.contains(25.0));
        assert!(!dpgm.contains(99.0));

        // 3rd insert triggers flush & merge
        dpgm.insert(5.0);
        assert_eq!(dpgm.len(), 6);
        assert!(dpgm.contains(5.0));
        assert!(dpgm.contains(10.0));
        assert!(dpgm.contains(15.0));
        assert!(dpgm.contains(20.0));
        assert!(dpgm.contains(25.0));
        assert!(dpgm.contains(30.0));
    }

    #[test]
    fn test_dynamic_pgm_range() {
        let mut dpgm = DynamicPGMIndex::new(vec![1.0, 10.0, 20.0], 2, 4);
        dpgm.insert(5.0);
        dpgm.insert(15.0);

        let res = dpgm.range_search(4.0, 16.0);
        assert_eq!(res, vec![5.0, 10.0, 15.0]);
    }

    #[test]
    fn test_dynamic_pgm_buffer_capacity_grows_with_base_size_not_fixed() {
        // The actual fix: a fixed flush threshold makes bulk-insert cost
        // O(n^2) (see this struct's docs) -- buffer_capacity must actually
        // grow as base grows, not stay pinned at its construction-time
        // value forever.
        let mut dpgm = DynamicPGMIndex::new(Vec::new(), 4, 64);
        let initial_capacity = dpgm.buffer_capacity;
        assert_eq!(initial_capacity, 64);

        for i in 0..50_000 {
            dpgm.insert(i as f64);
        }

        assert!(
            dpgm.buffer_capacity > initial_capacity,
            "buffer_capacity should have grown past its initial 64 after 50,000 inserts, still {}",
            dpgm.buffer_capacity
        );
    }

    #[test]
    fn test_dynamic_pgm_buffer_capacity_never_shrinks_below_its_floor() {
        // A tiny index (few flushes, small base) must not flush on every
        // single insert just because base.len()/BUFFER_GROWTH_DIVISOR
        // rounds down to something smaller than the original floor.
        let mut dpgm = DynamicPGMIndex::new(Vec::new(), 4, 64);
        for i in 0..10 {
            dpgm.insert(i as f64);
        }
        assert_eq!(dpgm.buffer_capacity, 64, "a small index shouldn't shrink its flush threshold below the floor");
    }

    #[test]
    fn test_dynamic_pgm_correct_after_many_flush_cycles() {
        // Exercises the unsorted-write-buffer change (insert no longer
        // keeps it sorted; flush_buffer sorts it once before merging) and
        // the growing buffer_capacity together, across many real flush
        // cycles -- contains() and range_search() must still be correct
        // for every key, not just the ones in the still-unflushed buffer
        // at the end.
        let mut dpgm = DynamicPGMIndex::new(Vec::new(), 4, 64);
        let n = 20_000;
        // Inserted out of order on purpose -- insert() no longer requires
        // (or produces) sorted arrival order now that write_buffer isn't
        // kept sorted between flushes.
        let mut keys: Vec<i64> = (0..n).collect();
        let mut rng_state: u64 = 0xC0FFEE;
        for i in (1..keys.len()).rev() {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let j = (rng_state >> 33) as usize % (i + 1);
            keys.swap(i, j);
        }
        for &k in &keys {
            dpgm.insert(k as f64);
        }

        assert_eq!(dpgm.len(), n as usize);
        for k in [0i64, 1, n / 2, n - 1] {
            assert!(dpgm.contains(k as f64), "missing key {k} after {n} out-of-order inserts across many flush cycles");
        }
        assert!(!dpgm.contains(-1.0));
        assert!(!dpgm.contains(n as f64));

        let range = dpgm.range_search(100.0, 105.0);
        assert_eq!(range, vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0]);
    }

    #[test]
    fn test_dynamic_pgm_bulk_insert_is_not_quadratic() {
        // Real regression coverage for the O(n^2/64) bug this fix
        // addresses: 200,000 inserts into a growing index. With the old
        // fixed-64-threshold behavior (200,000/64 ~= 3,125 full rebuilds,
        // each averaging ~100,000 keys) this took long enough to be a
        // real, measured problem (examples/monotonic_ingestion_niche.rs
        // showed real SQL insert throughput dropping ~5x between 20,000
        // and 500,000 rows because of it). With a growing threshold, this
        // must complete in well under a second, not tens of seconds.
        let mut dpgm = DynamicPGMIndex::new(Vec::new(), 4, 64);
        let n = 200_000;
        let start = std::time::Instant::now();
        for i in 0..n {
            dpgm.insert(i as f64);
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_secs_f64() < 1.0,
            "{n} inserts took {elapsed:.2?} -- looks like the old fixed-threshold O(n^2) behavior again"
        );
    }

    #[test]
    fn test_pgm_heap_bytes_grows_with_key_count() {
        let small = PGMIndex::build(vec![1.0, 2.0, 3.0], 4);
        let big = PGMIndex::build((0..100_000).map(|i| i as f64).collect(), 4);
        assert!(big.heap_bytes() > small.heap_bytes());
    }

    #[test]
    fn test_pgm_heap_bytes_far_smaller_than_one_segment_per_key() {
        // The actual mechanism behind PGM's real memory advantage: a long
        // run of keys that fits one linear model costs one fixed-size
        // segment, not one entry per key. A perfectly linear sequence
        // (build() keeps extending one segment indefinitely as long as
        // every point stays within error_bound of the fitted line) should
        // collapse to a small, roughly constant number of segments
        // regardless of how many keys go in.
        let pgm = PGMIndex::build((0..1_000_000).map(|i| i as f64).collect(), 4);
        assert!(
            pgm.segments.len() < 100,
            "a perfectly linear 1M-key sequence should need well under 100 segments, got {}",
            pgm.segments.len()
        );
    }
}

