//! SIMD-accelerated search, comparison, and filtering operations
//!
//! Provides AVX2 hardware-accelerated kernels on x86_64 with safe portable
//! 4-wide chunked fallbacks on any architecture.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// SIMD search and vector operations
pub struct SIMDSearch;

impl SIMDSearch {
    /// SIMD-accelerated binary search.
    /// For sub-slices of size <= 16, switches to 4-wide vectorized linear search
    /// to avoid branch misprediction penalties on narrow search windows.
    pub fn binary_search(arr: &[f64], target: f64) -> Option<usize> {
        if arr.is_empty() {
            return None;
        }

        let mut low = 0;
        let mut high = arr.len();

        // Binary search until the range is small enough for SIMD scan
        while high - low > 16 {
            let mid = low + (high - low) / 2;
            if arr[mid] < target {
                low = mid + 1;
            } else {
                high = mid;
            }
        }

        // Vectorized scan within [low..high]
        let sub = &arr[low..high];
        Self::simd_find(sub, target).map(|idx| low + idx)
    }

    /// SIMD-accelerated bounded search around a predicted position.
    /// Designed specifically for learned index structures (PGM / RMI) where the target
    /// key is guaranteed to be within [predicted_pos - error_bound, predicted_pos + error_bound].
    pub fn bounded_search(
        arr: &[f64],
        target: f64,
        predicted_pos: usize,
        error_bound: usize,
    ) -> Option<usize> {
        if arr.is_empty() {
            return None;
        }

        let start = predicted_pos.saturating_sub(error_bound).min(arr.len() - 1);
        let end = (predicted_pos + error_bound + 1).min(arr.len());

        let window = &arr[start..end];
        if window.len() <= 32 {
            Self::simd_find(window, target).map(|idx| start + idx)
        } else {
            Self::binary_search(window, target).map(|idx| start + idx)
        }
    }

    /// Find exact index of target in array using 4-wide vector comparisons
    pub fn simd_find(arr: &[f64], target: f64) -> Option<usize> {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { Self::find_avx2(arr, target) };
            }
        }

        Self::find_portable(arr, target)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn find_avx2(arr: &[f64], target: f64) -> Option<usize> {
        let len = arr.len();
        let target_vec = _mm256_set1_pd(target);
        let chunks = len / 4;

        for c in 0..chunks {
            let ptr = arr.as_ptr().add(c * 4);
            let val = _mm256_loadu_pd(ptr);
            let cmp = _mm256_cmp_pd(val, target_vec, _CMP_EQ_OQ);
            let mask = _mm256_movemask_pd(cmp);
            if mask != 0 {
                let bit_idx = mask.trailing_zeros() as usize;
                return Some(c * 4 + bit_idx);
            }
        }

        // Tail elements
        for i in (chunks * 4)..len {
            if arr[i] == target {
                return Some(i);
            }
        }

        None
    }

    fn find_portable(arr: &[f64], target: f64) -> Option<usize> {
        let chunks = arr.chunks_exact(4);
        let remainder = chunks.remainder();
        let mut offset = 0;

        for chunk in chunks {
            if chunk[0] == target {
                return Some(offset);
            }
            if chunk[1] == target {
                return Some(offset + 1);
            }
            if chunk[2] == target {
                return Some(offset + 2);
            }
            if chunk[3] == target {
                return Some(offset + 3);
            }
            offset += 4;
        }

        for (i, &val) in remainder.iter().enumerate() {
            if val == target {
                return Some(offset + i);
            }
        }

        None
    }

    /// Generic filter using predicate
    pub fn filter<F>(arr: &[f64], predicate: F) -> Vec<f64>
    where
        F: Fn(f64) -> bool,
    {
        arr.iter().filter(|x| predicate(**x)).copied().collect()
    }

    /// High-throughput vectorized range filter [min_val, max_val]
    pub fn filter_range(arr: &[f64], min_val: f64, max_val: f64) -> Vec<f64> {
        let mut result = Vec::with_capacity(arr.len() / 2);
        let chunks = arr.chunks_exact(4);
        let remainder = chunks.remainder();

        for chunk in chunks {
            let m0 = chunk[0] >= min_val && chunk[0] <= max_val;
            let m1 = chunk[1] >= min_val && chunk[1] <= max_val;
            let m2 = chunk[2] >= min_val && chunk[2] <= max_val;
            let m3 = chunk[3] >= min_val && chunk[3] <= max_val;

            if m0 { result.push(chunk[0]); }
            if m1 { result.push(chunk[1]); }
            if m2 { result.push(chunk[2]); }
            if m3 { result.push(chunk[3]); }
        }

        for &x in remainder {
            if x >= min_val && x <= max_val {
                result.push(x);
            }
        }

        result
    }

    /// SIMD compare operation
    pub fn compare(arr1: &[f64], arr2: &[f64]) -> Vec<bool> {
        let min_len = arr1.len().min(arr2.len());
        let mut result = Vec::with_capacity(min_len);

        let chunks1 = arr1[..min_len].chunks_exact(4);
        let chunks2 = arr2[..min_len].chunks_exact(4);

        for (c1, c2) in chunks1.zip(chunks2) {
            result.push(c1[0] == c2[0]);
            result.push(c1[1] == c2[1]);
            result.push(c1[2] == c2[2]);
            result.push(c1[3] == c2[3]);
        }

        let rem1 = &arr1[(min_len / 4 * 4)..min_len];
        let rem2 = &arr2[(min_len / 4 * 4)..min_len];
        for (&a, &b) in rem1.iter().zip(rem2) {
            result.push(a == b);
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_binary_search() {
        let arr = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert_eq!(SIMDSearch::binary_search(&arr, 3.0), Some(2));
        assert_eq!(SIMDSearch::binary_search(&arr, 1.0), Some(0));
        assert_eq!(SIMDSearch::binary_search(&arr, 10.0), Some(9));
        assert_eq!(SIMDSearch::binary_search(&arr, 11.0), None);
    }

    #[test]
    fn test_simd_bounded_search() {
        let arr = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        // Target 40.0, predicted position 3, bound 2
        assert_eq!(SIMDSearch::bounded_search(&arr, 40.0, 3, 2), Some(3));
        assert_eq!(SIMDSearch::bounded_search(&arr, 99.0, 3, 2), None);
    }

    #[test]
    fn test_simd_filter_range() {
        let arr = vec![1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0];
        let res = SIMDSearch::filter_range(&arr, 5.0, 20.0);
        assert_eq!(res, vec![5.0, 10.0, 15.0, 20.0]);
    }

    #[test]
    fn test_simd_compare() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 9.0, 3.0, 4.0, 0.0];
        let res = SIMDSearch::compare(&a, &b);
        assert_eq!(res, vec![true, false, true, true, false]);
    }
}
