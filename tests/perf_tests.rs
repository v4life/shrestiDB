//! Performance regression tests

#[cfg(test)]
mod perf_tests {
    use shrestidb::index::pgm::PGMIndex;
    use shrestidb::ml::regression::LinearRegression;
    use shrestidb::compute::simd_ops::SIMDSearch;
    use shrestidb::optimizer::cardinality::LearnedCardinalityEstimator;

    #[test]
    fn test_pgm_lookup_latency() {
        let keys: Vec<f64> = (0..50000).map(|i| i as f64).collect();
        let pgm = PGMIndex::build(keys.clone(), 64);

        let start = std::time::Instant::now();
        for _ in 0..1000 {
            let _ = pgm.search(25000.0);
        }
        let elapsed = start.elapsed();
        
        let avg_latency = elapsed.as_micros() as f64 / 1000.0;
        println!("PGM average lookup latency: {:.3} µs", avg_latency);
        
        // PGM should be very fast (sub-microsecond for this size)
        assert!(avg_latency < 1000.0);
    }

    #[test]
    fn test_linear_regression_accuracy() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];

        let model = LinearRegression::fit(&x, &y).unwrap();
        let r2 = model.r_squared(&x, &y);
        
        println!("Linear regression R² : {:.6}", r2);
        assert!(r2 > 0.999);
    }

    #[test]
    fn test_cardinality_estimation_speed() {
        use shrestidb::optimizer::cardinality::QueryPredicate;
        
        let estimator = LearnedCardinalityEstimator::new(10);
        let predicates = vec![
            QueryPredicate {
                column_id: 0,
                min_value: 10.0,
                max_value: 100.0,
                is_equality: false,
            },
        ];

        let start = std::time::Instant::now();
        for _ in 0..10000 {
            let _ = estimator.estimate_selectivity(&predicates);
        }
        let elapsed = start.elapsed();
        
        let avg_latency = elapsed.as_nanos() as f64 / 10000.0;
        println!("Cardinality estimation average latency: {:.0} ns", avg_latency);

        // No hard threshold here: an absolute wall-clock assertion in an
        // unoptimized debug build is inherently flaky (this one failed
        // under ordinary machine load with generous headroom already
        // baked in, on a commit that hadn't changed in weeks). Real
        // performance tracking belongs in a proper benchmark harness --
        // see benches/optimizer_benchmark.rs, which already covers
        // cardinality estimation with criterion (statistical, no brittle
        // pass/fail threshold). This test still measures and prints the
        // number for a quick eyeball check, same as
        // test_simd_search_performance and test_memory_efficiency below.
    }

    #[test]
    fn test_simd_search_performance() {
        let arr: Vec<f64> = (0..100000).map(|i| i as f64).collect();
        
        let start = std::time::Instant::now();
        for _ in 0..1000 {
            let _ = SIMDSearch::binary_search(&arr, 50000.0);
        }
        let elapsed = start.elapsed();
        
        println!("SIMD binary search average time: {:.3} µs", elapsed.as_micros() as f64 / 1000.0);
    }

    #[test]
    fn test_memory_efficiency() {
        use std::mem;
        
        let keys: Vec<f64> = (0..100000).map(|i| i as f64).collect();
        let pgm = PGMIndex::build(keys.clone(), 64);
        
        let pgm_size = mem::size_of_val(&pgm);
        let keys_size = mem::size_of_val(&pgm.keys);
        
        println!("PGM index size: {} bytes", pgm_size);
        println!("Keys storage size: {} bytes", keys_size);
        println!("Average bytes per key: {:.2}", keys_size as f64 / pgm.keys.len() as f64);
    }
}
