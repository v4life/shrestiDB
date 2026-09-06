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

    /// Build PGM index from sorted keys
    pub fn build(keys: Vec<f64>, error_bound: usize) -> Self {
        let mut segments = Vec::new();

        if keys.is_empty() {
            return PGMIndex {
                segments,
                keys,
                error_bound,
            };
        }

        let mut i = 0;

        while i < keys.len() {
            let start_pos = i;
            let start_key = keys[i];
            let mut j = (i + 1).min(keys.len());

            // Find segment end with bounded error
            while j < keys.len() {
                let segment_keys = &keys[i..=j];
                let segment_positions: Vec<f64> = (0..=j - i).map(|x| x as f64).collect();

                if let Some(model) = LinearModel::fit(
                    segment_keys,
                    &segment_positions
                        .iter()
                        .map(|x| *x + start_pos as f64)
                        .collect::<Vec<_>>(),
                ) {
                    // Check error
                    let mut max_error = 0usize;
                    for (k, pos) in segment_keys.iter().zip(&segment_positions) {
                        let predicted = model.predict(*k) as usize;
                        let actual = (*pos as usize) + start_pos;
                        let error = (predicted as i32 - actual as i32).abs() as usize;
                        max_error = max_error.max(error);
                    }

                    if max_error <= error_bound {
                        j += 1;
                        continue;
                    }
                }
                break;
            }

            let end_pos = j - 1;
            let end_key = keys[end_pos];

            if let Some(model) = LinearModel::fit(
                &keys[i..=end_pos],
                &(0..=end_pos - i)
                    .map(|x| (x + start_pos) as f64)
                    .collect::<Vec<_>>(),
            ) {
                let segment = PGMSegment::new(model, start_key, end_key, start_pos, end_pos);
                segments.push(segment);
            }

            i = end_pos + 1;
        }

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

