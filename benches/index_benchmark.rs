use criterion::{black_box, criterion_group, criterion_main, Criterion};
use shrestidb::index::btree::BTree;
use shrestidb::index::pgm::PGMIndex;
use shrestidb::index::rmi::{RMIIndex, RMIStage};
use shrestidb::index::models::LinearModel;

/// Generate synthetic keys with skewed distribution
fn generate_skewed_keys(count: usize) -> Vec<f64> {
    let mut keys = Vec::new();
    for i in 0..count {
        // Zipfian-like distribution (many small values, few large)
        let key = ((i as f64 + 1.0).ln() * 1000.0) % 100000.0;
        keys.push(key);
    }
    keys.sort_by(|a, b| a.partial_cmp(b).unwrap());
    keys
}

/// Generate uniformly distributed keys
fn generate_uniform_keys(count: usize) -> Vec<f64> {
    (0..count).map(|i| i as f64).collect()
}

/// Generate random keys
fn generate_random_keys(count: usize) -> Vec<f64> {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    
    let hasher = RandomState::new();
    let mut keys = Vec::new();
    for i in 0..count {
        let mut h = hasher.build_hasher();
        h.write_usize(i);
        let key = (h.finish() % 100000) as f64;
        keys.push(key);
    }
    keys.sort_by(|a, b| a.partial_cmp(b).unwrap());
    keys
}

fn benchmark_btree_lookup(c: &mut Criterion) {
    let keys = generate_skewed_keys(10000);
    let mut btree = BTree::new(100);
    
    for key in &keys {
        let _ = btree.insert(*key, keys.iter().position(|k| k == key).unwrap());
    }

    let search_keys: Vec<f64> = keys.iter().step_by(10).copied().collect();

    c.bench_function("btree_lookup_skewed_10k", |b| {
        b.iter(|| {
            for key in &search_keys {
                let _ = btree.search(black_box(*key));
            }
        });
    });
}

fn benchmark_pgm_lookup(c: &mut Criterion) {
    let keys = generate_skewed_keys(10000);
    let pgm = PGMIndex::build(keys.clone(), 32);
    let search_keys: Vec<f64> = keys.iter().step_by(10).copied().collect();

    c.bench_function("pgm_lookup_skewed_10k", |b| {
        b.iter(|| {
            for key in &search_keys {
                let _ = pgm.search(black_box(*key));
            }
        });
    });
}

fn benchmark_rmi_lookup(c: &mut Criterion) {
    let keys = generate_skewed_keys(10000);
    let positions: Vec<usize> = (0..keys.len()).collect();
    let model = LinearModel::fit(&keys, &positions.iter().map(|p| *p as f64).collect::<Vec<_>>())
        .unwrap();
    let stage = RMIStage::new(vec![model]);
    let rmi = RMIIndex::new(vec![stage], keys.clone(), positions);
    let search_keys: Vec<f64> = keys.iter().step_by(10).copied().collect();

    c.bench_function("rmi_lookup_skewed_10k", |b| {
        b.iter(|| {
            for key in &search_keys {
                let _ = rmi.find_exact(black_box(*key), 16);
            }
        });
    });
}

fn benchmark_uniform_distribution(c: &mut Criterion) {
    let keys = generate_uniform_keys(10000);
    let pgm = PGMIndex::build(keys.clone(), 32);
    let search_keys: Vec<f64> = keys.iter().step_by(10).copied().collect();

    c.bench_function("pgm_lookup_uniform_10k", |b| {
        b.iter(|| {
            for key in &search_keys {
                let _ = pgm.search(black_box(*key));
            }
        });
    });
}

fn benchmark_random_distribution(c: &mut Criterion) {
    let keys = generate_random_keys(10000);
    let pgm = PGMIndex::build(keys.clone(), 32);
    let search_keys: Vec<f64> = keys.iter().step_by(10).copied().collect();

    c.bench_function("pgm_lookup_random_10k", |b| {
        b.iter(|| {
            for key in &search_keys {
                let _ = pgm.search(black_box(*key));
            }
        });
    });
}

fn benchmark_index_build_time(c: &mut Criterion) {
    let keys = generate_skewed_keys(10000);

    c.bench_function("pgm_build_10k", |b| {
        b.iter(|| {
            PGMIndex::build(black_box(keys.clone()), 32);
        });
    });
}

criterion_group!(
    benches,
    benchmark_btree_lookup,
    benchmark_pgm_lookup,
    benchmark_rmi_lookup,
    benchmark_uniform_distribution,
    benchmark_random_distribution,
    benchmark_index_build_time
);
criterion_main!(benches);
