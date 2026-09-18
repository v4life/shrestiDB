//! Real memory-footprint comparison: PGM, RMI, and this codebase's real
//! B+Tree (`index::btree::BTree`, rewritten to actually split rather than
//! being a flat array wearing B-tree-shaped names -- see that module's
//! docs) over the same key sets, at growing scale.
//!
//! This targets a specific, common real-world shape rather than a
//! synthetic worst/best case for any one structure: a monotonic or
//! near-monotonic key -- an auto-increment ID, an order number, an
//! event/log timestamp -- which is what most production tables actually
//! have as a primary or clustering key. Two distributions are measured:
//! a purely sequential key (the best case for a piecewise-linear model,
//! since one segment can cover an unbounded run of it) and a "bursty
//! timestamp" key (mostly sequential, occasional out-of-order jitter --
//! closer to what real event ingestion looks like than a lab-perfect
//! sequence).
//!
//! `heap_bytes()` on each structure is a real, computed measurement (Vec
//! capacity times element size, summed over every heap allocation the
//! structure holds) -- not measured process RSS, so it doesn't include
//! allocator fragmentation/overhead, but it's the same honest, consistent
//! accounting method for all three, so the *comparison* between them is
//! real even if the absolute numbers undercount true process memory use.

use shrestidb::index::btree::BTree;
use shrestidb::index::models::LinearModel;
use shrestidb::index::pgm::PGMIndex;
use shrestidb::index::rmi::{RMIIndex, RMIStage};

fn sequential_keys(n: usize) -> Vec<f64> {
    (0..n).map(|i| i as f64).collect()
}

/// Real event-ingestion shape: dense "active" bursts of closely-spaced
/// timestamps separated by sparser "quiet" periods -- daytime order volume
/// vs. an overnight lull, alternating every 2,000 keys. Still monotonic
/// (real timestamps are), but not one straight line: a constant slope
/// stops fitting every time the density regime switches, which is exactly
/// what forces a piecewise-linear model to open a new segment -- a
/// realistic reason for PGM's segment count to grow, not an adversarial
/// worst case.
///
/// (An earlier version of this function added jitter and then sorted the
/// result, which silently erased the irregularity it was trying to create
/// -- a *static* index build only ever sees the final sorted value set,
/// never arrival order, so post-jitter sorting just reconstructed a
/// sequence indistinguishable from `sequential_keys`. That bug is why this
/// scenario's numbers were identical to the sequential ones before.)
fn bursty_timestamp_keys(n: usize) -> Vec<f64> {
    let mut keys = Vec::with_capacity(n);
    let mut t = 0.0f64;
    for i in 0..n {
        let dense_period = (i / 2000) % 2 == 0;
        t += if dense_period { 1.0 } else { 50.0 };
        keys.push(t);
    }
    keys
}

fn mb(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn measure(label: &str, keys: Vec<f64>) {
    let n = keys.len();
    println!("\n--- {label}: {n} keys ---");

    let pgm = PGMIndex::build(keys.clone(), 8);
    let pgm_bytes = pgm.heap_bytes();

    let positions: Vec<usize> = (0..n).collect();
    let model = LinearModel::fit(&keys, &positions.iter().map(|p| *p as f64).collect::<Vec<_>>())
        .unwrap_or(LinearModel::new(0.0, 0.0));
    let rmi = RMIIndex::new(vec![RMIStage::new(vec![model])], keys.clone(), positions);
    let rmi_bytes = rmi.heap_bytes();

    let mut btree = BTree::new(128); // 128: a realistic disk-page-sized fanout, not tuned to flatter either side
    for (i, &k) in keys.iter().enumerate() {
        btree.insert(k, i).unwrap();
    }
    let btree_bytes = btree.heap_bytes();

    println!("  PGM:    {:>8.2} MB  ({:>6.1} bytes/key, {} segments)", mb(pgm_bytes), pgm_bytes as f64 / n as f64, pgm.segments.len());
    println!("  RMI:    {:>8.2} MB  ({:>6.1} bytes/key)", mb(rmi_bytes), rmi_bytes as f64 / n as f64);
    println!("  B+Tree: {:>8.2} MB  ({:>6.1} bytes/key, height {})", mb(btree_bytes), btree_bytes as f64 / n as f64, btree.height());
    println!("  PGM vs B+Tree: {:.1}x smaller", btree_bytes as f64 / pgm_bytes as f64);
    println!("  RMI vs B+Tree: {:.1}x smaller", btree_bytes as f64 / rmi_bytes as f64);
}

fn main() {
    println!("=== Real Memory Footprint: PGM vs RMI vs B+Tree ===");
    println!("(heap_bytes(): real Vec-capacity accounting, not process RSS -- see module docs)");

    for &n in &[100_000usize, 1_000_000, 10_000_000] {
        measure("sequential keys", sequential_keys(n));
        measure("bursty timestamp keys", bursty_timestamp_keys(n));
    }
}
