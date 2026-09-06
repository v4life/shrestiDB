//! Example TPC-H benchmark workload
//!
//! Simulates the TPC-H analytical query workload.

use shresti::index::pgm::PGMIndex;
use shresti::optimizer::cardinality::LearnedCardinalityEstimator;
use shresti::optimizer::cost_model::CostModel;
use std::time::Instant;

fn main() {
    println!("=== TPC-H Workload Simulation ===");
    println!();

    // Simulate a large fact table
    let order_keys: Vec<f64> = (0..1_000_000)
        .map(|i| (i as f64 * 1.337) % 1_000_000.0)
        .collect();

    println!("Building learned index on {} order records...", order_keys.len());
    let start = Instant::now();
    let mut sorted_keys = order_keys.clone();
    sorted_keys.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let index = PGMIndex::build(sorted_keys.clone(), 64);
    let build_time = start.elapsed();
    println!("Index built in {:?}", build_time);
    println!();

    // Simulate Q1: Simple range query
    println!("Query 1: SELECT * FROM orders WHERE orderkey BETWEEN 100000 AND 200000");
    let estimator = LearnedCardinalityEstimator::new(10);
    let start = Instant::now();
    let estimated_rows = estimator.estimate_row_count(1_000_000, &vec![]);
    let est_time = start.elapsed();
    println!("  Estimated rows: {}", estimated_rows);
    println!("  Estimation time: {:?}", est_time);
    println!();

    // Simulate Q6: Aggregation query
    println!("Query 6: SELECT sum(extendedprice * discount) FROM orders WHERE ...");
    let cost_model = CostModel::new();
    use shresti::optimizer::cost_model::{OperatorCost, OperatorType};
    let plan = vec![
        OperatorCost::new(OperatorType::TableScan, 1_000_000, 1_000_000, 1.0),
        OperatorCost::new(OperatorType::Filter, 1_000_000, 500_000, 0.5),
        OperatorCost::new(OperatorType::Aggregate, 500_000, 1, 0.000002),
    ];
    let total_cost = cost_model.estimate_total_cost(&plan);
    println!("  Estimated cost: {:.2}", total_cost);
    println!();

    // Index lookup performance
    println!("Index Lookup Performance:");
    let search_keys: Vec<f64> = sorted_keys.iter().step_by(1000).copied().collect();
    let start = Instant::now();
    for key in &search_keys {
        let _ = index.search(*key);
    }
    let lookup_time = start.elapsed();
    let avg_latency = lookup_time.as_micros() as f64 / search_keys.len() as f64;
    println!("  Total lookups: {}", search_keys.len());
    println!("  Total time: {:?}", lookup_time);
    println!("  Average latency: {:.3} µs", avg_latency);
    println!();

    println!("=== Summary ===");
    println!("Index type: PGM (Piecewise Geometric Model)");
    println!("Dataset size: {} records", order_keys.len());
    println!("Index build time: {:?}", build_time);
    println!("Average lookup latency: {:.3} µs", avg_latency);
}
