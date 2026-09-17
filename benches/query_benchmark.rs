use criterion::{black_box, criterion_group, criterion_main, Criterion};
use shrestidb::index::pgm::PGMIndex;
use shrestidb::optimizer::cardinality::ColumnDistribution;
use shrestidb::optimizer::cost_model::{CostModel, OperatorCost, OperatorType};
use shrestidb::execution::catalog::{Catalog, TableSchema, Column, DataType};
use shrestidb::execution::operators::Value;

fn generate_test_keys(count: usize) -> Vec<f64> {
    (0..count).map(|i| (i as f64 * 1.5) % 100000.0).collect()
}

fn benchmark_end_to_end_query_optimization(c: &mut Criterion) {
    // Real ColumnDistribution now -- see optimizer::cardinality's module
    // doc for why the old "LearnedCardinalityEstimator" this benchmark
    // used to measure was replaced.
    let dist = ColumnDistribution::build_numeric(generate_test_keys(100000), 64);
    let cost_model = CostModel::new();

    let plan = vec![
        OperatorCost::new(OperatorType::TableScan, 100000, 100000, 1.0),
        OperatorCost::new(OperatorType::Filter, 100000, 50000, 0.5),
        OperatorCost::new(OperatorType::Aggregate, 50000, 1000, 0.02),
    ];

    c.bench_function("e2e_optimization_pipeline", |b| {
        b.iter(|| {
            let _card = dist.estimate_row_count(">=", &Value::Float(50000.0), 100000);
            let _cost = cost_model.estimate_total_cost(black_box(&plan));
        });
    });
}

fn benchmark_catalog_lookup(c: &mut Criterion) {
    let mut catalog = Catalog::new();
    let mut schema = TableSchema::new(1, "users".to_string());
    
    for i in 0..20 {
        schema.add_column(Column {
            id: i,
            name: format!("col_{}", i),
            data_type: DataType::Integer,
            nullable: false,
            primary_key: i == 0,
        });
    }
    
    catalog.register_table(schema);

    c.bench_function("catalog_table_lookup", |b| {
        b.iter(|| {
            let _ = catalog.get_table(black_box("users"));
        });
    });
}

fn benchmark_index_scan_vs_full_scan(c: &mut Criterion) {
    let keys = generate_test_keys(100000);
    let pgm = PGMIndex::build(keys.clone(), 64);

    let index_scan = vec![
        OperatorCost::new(OperatorType::IndexScan, 100000, 1000, 0.01),
    ];

    let full_scan = vec![
        OperatorCost::new(OperatorType::TableScan, 100000, 1000, 1.0),
        OperatorCost::new(OperatorType::Filter, 100000, 1000, 0.01),
    ];

    let cost_model = CostModel::new();

    c.bench_function("index_scan_performance", |b| {
        b.iter(|| {
            let index_cost = cost_model.estimate_total_cost(&index_scan);
            let full_scan_cost = cost_model.estimate_total_cost(&full_scan);
            let _speedup = full_scan_cost / index_cost;
        });
    });
}

criterion_group!(
    benches,
    benchmark_end_to_end_query_optimization,
    benchmark_catalog_lookup,
    benchmark_index_scan_vs_full_scan
);
criterion_main!(benches);
