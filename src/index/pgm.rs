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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicPGMIndex {
    /// Immutable/base PGM index
    pub base: PGMIndex,
    /// Sorted write buffer (L0) absorbing writes
    pub write_buffer: Vec<f64>,
    /// Capacity of write buffer before triggering a linear merge & rebuild
    pub buffer_capacity: usize,
    /// Error bound for segments
    pub error_bound: usize,
}

impl DynamicPGMIndex {
    /// Create a new dynamic PGM index
    pub fn new(keys: Vec<f64>, error_bound: usize, buffer_capacity: usize) -> Self {
        let base = PGMIndex::build(keys, error_bound);
        DynamicPGMIndex {
            base,
            write_buffer: Vec::with_capacity(buffer_capacity),
            buffer_capacity,
            error_bound,
        }
    }

    /// Insert key into the dynamic index
    pub fn insert(&mut self, key: f64) {
        let idx = match self
            .write_buffer
            .binary_search_by(|k| k.partial_cmp(&key).unwrap())
        {
            Ok(i) => i,
            Err(i) => i,
        };
        self.write_buffer.insert(idx, key);

        if self.write_buffer.len() >= self.buffer_capacity {
            self.flush_buffer();
        }
    }

    /// Flush write buffer by merging with base keys and rebuilding segments
    pub fn flush_buffer(&mut self) {
        if self.write_buffer.is_empty() {
            return;
        }

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
    }

    /// Check if key exists in either write buffer or base PGM
    pub fn contains(&self, key: f64) -> bool {
        if self
            .write_buffer
            .binary_search_by(|k| k.partial_cmp(&key).unwrap())
            .is_ok()
        {
            return true;
        }
        self.base.search(key).is_some()
    }

    /// Range search across write buffer and base PGM
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
}

