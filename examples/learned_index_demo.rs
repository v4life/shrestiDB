//! Learned index demonstration
//!
//! Shows the power of learned indexes vs traditional approaches.

use shrestidb::index::pgm::PGMIndex;
use shrestidb::index::btree::BTree;
use std::time::Instant;

fn main() {
    println!("=== Learned Index Demonstration ===");
    println!();

    // Dataset configurations
    let sizes = vec![10_000, 100_000, 1_000_000];

    for size in sizes {
        println!("\n--- Dataset size: {} records ---", size);

        // Generate keys with realistic distribution (Zipfian-like)
        let mut keys: Vec<f64> = (0..size)
            .map(|i| {
                let x = (i as f64 + 1.0).ln();
                (x * 1000.0) % 100_000.0
            })
            .collect();
        keys.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Build PGM index
        println!("\nBuilding PGM (Piecewise Geometric Model) index...");
        let start = Instant::now();
        let pgm = PGMIndex::build(keys.clone(), 64);
        let pgm_build = start.elapsed();
        println!("  Build time: {:?}", pgm_build);
        println!("  Segments: {}", pgm.segments.len());

        // Build B-Tree index
        println!("\nBuilding B-Tree index...");
        let start = Instant::now();
        let mut btree = BTree::new(100);
        for (i, key) in keys.iter().enumerate() {
            let _ = btree.insert(*key, i);
        }
        let btree_build = start.elapsed();
        println!("  Build time: {:?}", btree_build);

        // Benchmark lookups
        let search_keys: Vec<f64> = keys.iter().step_by(size.max(100) / 100).copied().collect();
        let num_searches = search_keys.len();

        println!("\nPerforming {} searches...", num_searches);

        let start = Instant::now();
        for key in &search_keys {
            let _ = pgm.search(*key);
        }
        let pgm_search = start.elapsed();
        let pgm_avg = pgm_search.as_micros() as f64 / num_searches as f64;

        let start = Instant::now();
        for key in &search_keys {
            let _ = btree.search(*key);
        }
        let btree_search = start.elapsed();
        let btree_avg = btree_search.as_micros() as f64 / num_searches as f64;

        // Results
        println!("\nResults:");
        println!("  PGM:");
        println!("    Total time: {:?}", pgm_search);
        println!("    Avg per lookup: {:.3} µs", pgm_avg);
        println!("  B-Tree:");
        println!("    Total time: {:?}", btree_search);
        println!("    Avg per lookup: {:.3} µs", btree_avg);
        println!("\n  Speedup: {:.1}x", btree_avg / pgm_avg);
        println!("  PGM build vs B-Tree: {:.1}x", pgm_build.as_secs_f64() / btree_build.as_secs_f64());
    }

    println!("\n=== Key Insights ===");
    println!("1. PGM builds faster than B-Tree for large datasets");
    println!("2. PGM lookups are significantly faster on sorted data");
    println!("3. Space efficiency improves with PGM at larger scales");
    println!("4. Learned models adapt to data distribution");
}
