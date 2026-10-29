//! Vectorized mathematical operations
//!
//! Provides AVX2-accelerated kernels on x86_64 with unrolled 4-lane accumulators
//! for high ILP (instruction-level parallelism) and SIMD throughput.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// High-performance vector operations
pub struct VectorOps;

impl VectorOps {
    /// Element-wise addition: c[i] = a[i] + b[i]
    pub fn add(a: &[f64], b: &[f64]) -> Vec<f64> {
        let min_len = a.len().min(b.len());
        let mut result = Vec::with_capacity(min_len);

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    result.set_len(min_len);
                    Self::add_avx2(a, b, &mut result);
                    return result;
                }
            }
        }

        Self::add_portable(a, b, &mut result);
        result
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn add_avx2(a: &[f64], b: &[f64], result: &mut [f64]) {
        let len = result.len();
        let chunks = len / 4;

        for c in 0..chunks {
            let offset = c * 4;
            let va = _mm256_loadu_pd(a.as_ptr().add(offset));
            let vb = _mm256_loadu_pd(b.as_ptr().add(offset));
            let vr = _mm256_add_pd(va, vb);
            _mm256_storeu_pd(result.as_mut_ptr().add(offset), vr);
        }

        for i in (chunks * 4)..len {
            result[i] = a[i] + b[i];
        }
    }

    fn add_portable(a: &[f64], b: &[f64], result: &mut Vec<f64>) {
        let min_len = a.len().min(b.len());
        let chunks_a = a[..min_len].chunks_exact(4);
        let chunks_b = b[..min_len].chunks_exact(4);

        for (ca, cb) in chunks_a.zip(chunks_b) {
            result.push(ca[0] + cb[0]);
            result.push(ca[1] + cb[1]);
            result.push(ca[2] + cb[2]);
            result.push(ca[3] + cb[3]);
        }

        let tail_start = min_len / 4 * 4;
        for i in tail_start..min_len {
            result.push(a[i] + b[i]);
        }
    }

    /// Element-wise multiplication: c[i] = a[i] * b[i]
    pub fn multiply(a: &[f64], b: &[f64]) -> Vec<f64> {
        let min_len = a.len().min(b.len());
        let mut result = Vec::with_capacity(min_len);

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    result.set_len(min_len);
                    Self::mul_avx2(a, b, &mut result);
                    return result;
                }
            }
        }

        Self::mul_portable(a, b, &mut result);
        result
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn mul_avx2(a: &[f64], b: &[f64], result: &mut [f64]) {
        let len = result.len();
        let chunks = len / 4;

        for c in 0..chunks {
            let offset = c * 4;
            let va = _mm256_loadu_pd(a.as_ptr().add(offset));
            let vb = _mm256_loadu_pd(b.as_ptr().add(offset));
            let vr = _mm256_mul_pd(va, vb);
            _mm256_storeu_pd(result.as_mut_ptr().add(offset), vr);
        }

        for i in (chunks * 4)..len {
            result[i] = a[i] * b[i];
        }
    }

    fn mul_portable(a: &[f64], b: &[f64], result: &mut Vec<f64>) {
        let min_len = a.len().min(b.len());
        let chunks_a = a[..min_len].chunks_exact(4);
        let chunks_b = b[..min_len].chunks_exact(4);

        for (ca, cb) in chunks_a.zip(chunks_b) {
            result.push(ca[0] * cb[0]);
            result.push(ca[1] * cb[1]);
            result.push(ca[2] * cb[2]);
            result.push(ca[3] * cb[3]);
        }

        let tail_start = min_len / 4 * 4;
        for i in tail_start..min_len {
            result.push(a[i] * b[i]);
        }
    }

    /// Dot product with unrolled accumulators to break latency chains
    pub fn dot(a: &[f64], b: &[f64]) -> f64 {
        let min_len = a.len().min(b.len());

        let chunks_a = a[..min_len].chunks_exact(4);
        let chunks_b = b[..min_len].chunks_exact(4);

        let mut acc0 = 0.0;
        let mut acc1 = 0.0;
        let mut acc2 = 0.0;
        let mut acc3 = 0.0;

        for (ca, cb) in chunks_a.zip(chunks_b) {
            acc0 += ca[0] * cb[0];
            acc1 += ca[1] * cb[1];
            acc2 += ca[2] * cb[2];
            acc3 += ca[3] * cb[3];
        }

        let mut sum = (acc0 + acc1) + (acc2 + acc3);
        let tail_start = min_len / 4 * 4;
        for i in tail_start..min_len {
            sum += a[i] * b[i];
        }

        sum
    }

    /// Vector L2 norm
    pub fn norm(v: &[f64]) -> f64 {
        Self::dot(v, v).sqrt()
    }

    /// Cosine similarity between two vectors
    pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
        let dot = Self::dot(a, b);
        let norm_a = Self::norm(a);
        let norm_b = Self::norm(b);
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }

    /// Cosine distance: 1 - cosine_similarity. Range [0, 2], lower = more similar.
    /// This is the metric used by the `<=>` operator (pgvector-compatible).
    pub fn cosine_distance(a: &[f64], b: &[f64]) -> f64 {
        1.0 - Self::cosine_similarity(a, b)
    }

    /// Euclidean (L2) distance between two vectors.
    /// This is the metric used by the `<->` operator (pgvector-compatible).
    pub fn l2_distance(a: &[f64], b: &[f64]) -> f64 {
        let min_len = a.len().min(b.len());
        let chunks_a = a[..min_len].chunks_exact(4);
        let chunks_b = b[..min_len].chunks_exact(4);

        let mut acc0 = 0.0;
        let mut acc1 = 0.0;
        let mut acc2 = 0.0;
        let mut acc3 = 0.0;

        for (ca, cb) in chunks_a.zip(chunks_b) {
            let d0 = ca[0] - cb[0];
            let d1 = ca[1] - cb[1];
            let d2 = ca[2] - cb[2];
            let d3 = ca[3] - cb[3];
            acc0 += d0 * d0;
            acc1 += d1 * d1;
            acc2 += d2 * d2;
            acc3 += d3 * d3;
        }
        let mut sum = (acc0 + acc1) + (acc2 + acc3);

        let tail_start = min_len / 4 * 4;
        for i in tail_start..min_len {
            let d = a[i] - b[i];
            sum += d * d;
        }

        sum.sqrt()
    }

    /// Negative inner product distance: -dot(a, b).
    /// This is the metric used by the `<#>` operator (pgvector-compatible).
    /// More negative = more similar, so negating makes lower = more similar.
    pub fn inner_product_distance(a: &[f64], b: &[f64]) -> f64 {
        -Self::dot(a, b)
    }

    /// L2-normalize a vector in place, returning a new normalized vector.
    pub fn normalize(v: &[f64]) -> Vec<f64> {
        let n = Self::norm(v);
        if n == 0.0 {
            return vec![0.0; v.len()];
        }
        v.iter().map(|x| x / n).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_add() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![4.0, 5.0, 6.0, 7.0, 8.0];
        let result = VectorOps::add(&a, &b);
        assert_eq!(result, vec![5.0, 7.0, 9.0, 11.0, 13.0]);
    }

    #[test]
    fn test_vector_multiply() {
        let a = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![3.0, 4.0, 5.0, 6.0, 7.0];
        let result = VectorOps::multiply(&a, &b);
        assert_eq!(result, vec![6.0, 12.0, 20.0, 30.0, 42.0]);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![4.0, 5.0, 6.0, 7.0, 8.0];
        // 4 + 10 + 18 + 28 + 40 = 100
        let result = VectorOps::dot(&a, &b);
        assert_eq!(result, 100.0);
    }

    #[test]
    fn test_norm_and_cosine() {
        let a = vec![3.0, 4.0];
        assert_eq!(VectorOps::norm(&a), 5.0);

        let b = vec![6.0, 8.0];
        let sim = VectorOps::cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_distance() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((VectorOps::cosine_distance(&a, &b) - 0.0).abs() < 1e-6);

        let c = vec![0.0, 1.0, 0.0];
        assert!((VectorOps::cosine_distance(&a, &c) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_distance() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![3.0, 4.0, 0.0];
        assert!((VectorOps::l2_distance(&a, &b) - 5.0).abs() < 1e-6);

        // Distance to self is 0
        assert!((VectorOps::l2_distance(&a, &a) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_inner_product_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        // dot = 4 + 10 + 18 = 32, negative = -32
        assert!((VectorOps::inner_product_distance(&a, &b) - (-32.0)).abs() < 1e-6);
    }

    #[test]
    fn test_normalize() {
        let v = vec![3.0, 4.0];
        let n = VectorOps::normalize(&v);
        assert!((n[0] - 0.6).abs() < 1e-6);
        assert!((n[1] - 0.8).abs() < 1e-6);
        assert!((VectorOps::norm(&n) - 1.0).abs() < 1e-6);
    }
}
