//! Integration tests for Dynamic PGM Index and SIMD operations

use shresti::index::pgm::DynamicPGMIndex;
use shresti::compute::simd_ops::SIMDSearch;
use shresti::compute::vector_math::VectorOps;

#[test]
fn test_dynamic_pgm_bulk_insertion() {
    let mut dpgm = DynamicPGMIndex::new(vec![100.0, 200.0, 300.0], 4, 10);

    for i in 0..50 {
        dpgm.insert((i * 10) as f64);
    }

    assert_eq!(dpgm.len(), 53);

    for i in 0..50 {
        assert!(dpgm.contains((i * 10) as f64), "Should contain {}", i * 10);
    }
}

#[test]
fn test_dynamic_pgm_range_query() {
    let mut dpgm = DynamicPGMIndex::new(vec![10.0, 30.0, 50.0, 70.0], 2, 5);
    dpgm.insert(20.0);
    dpgm.insert(40.0);
    dpgm.insert(60.0);

    let res = dpgm.range_search(25.0, 65.0);
    assert_eq!(res, vec![30.0, 40.0, 50.0, 60.0]);
}

#[test]
fn test_simd_search_and_filter() {
    let data: Vec<f64> = (0..100).map(|x| x as f64).collect();

    // Binary search
    assert_eq!(SIMDSearch::binary_search(&data, 42.0), Some(42));
    assert_eq!(SIMDSearch::binary_search(&data, 999.0), None);

    // Bounded search
    assert_eq!(SIMDSearch::bounded_search(&data, 50.0, 48, 5), Some(50));

    // Vector range filter
    let filtered = SIMDSearch::filter_range(&data, 10.0, 15.0);
    assert_eq!(filtered, vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
}

#[test]
fn test_vector_math_kernels() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];

    let sum = VectorOps::add(&a, &b);
    assert_eq!(sum, vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

    let prod = VectorOps::multiply(&a, &b);
    assert_eq!(prod, vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);

    let dot = VectorOps::dot(&a, &b);
    assert_eq!(dot, 72.0);
}
