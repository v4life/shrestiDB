use criterion::{black_box, criterion_group, criterion_main, Criterion};
use shrestidb::execution::operators::Value;
use shrestidb::optimizer::cardinality::ColumnDistribution;
use shrestidb::optimizer::cost_model::{CostModel, OperatorCost, OperatorType};
use shrestidb::optimizer::join_reorder::JoinOrderer;
use shrestidb::sql::parser::JoinClause;

fn benchmark_cardinality_estimation(c: &mut Criterion) {
    // Real ColumnDistribution now -- see optimizer::cardinality's module
    // doc for why the old "LearnedCardinalityEstimator" this benchmark
    // used to measure was replaced (fixed weights, never actually read
    // the stats it was given).
    let values: Vec<f64> = (0..1000).map(|i| i as f64).collect();
    let dist = ColumnDistribution::build_numeric(values, 32);

    c.bench_function("cardinality_estimate_range_predicate", |b| {
        b.iter(|| {
            dist.estimate_selectivity(black_box(">="), black_box(&Value::Float(300.0)));
        });
    });
}

fn benchmark_cardinality_categorical(c: &mut Criterion) {
    let values: Vec<String> = (0..1000).map(|i| format!("category_{}", i % 20)).collect();
    let dist = ColumnDistribution::build_categorical(values);

    c.bench_function("cardinality_estimate_categorical_equality", |b| {
        b.iter(|| {
            dist.estimate_selectivity(black_box("="), black_box(&Value::String("category_0".to_string())));
        });
    });
}

fn benchmark_cost_model_estimation(c: &mut Criterion) {
    let cost_model = CostModel::new();
    let operator = OperatorCost::new(OperatorType::TableScan, 10000, 5000, 0.5);

    c.bench_function("cost_model_operator_estimation", |b| {
        b.iter(|| {
            cost_model.estimate_cost(black_box(&operator));
        });
    });
}

fn benchmark_plan_cost_comparison(c: &mut Criterion) {
    let cost_model = CostModel::new();

    let plan_a = vec![
        OperatorCost::new(OperatorType::TableScan, 10000, 10000, 1.0),
        OperatorCost::new(OperatorType::Filter, 10000, 5000, 0.5),
        OperatorCost::new(OperatorType::HashJoin, 5000, 2500, 0.5),
    ];

    let plan_b = vec![
        OperatorCost::new(OperatorType::IndexScan, 10000, 5000, 0.5),
        OperatorCost::new(OperatorType::HashJoin, 5000, 2500, 0.5),
    ];

    c.bench_function("cost_model_plan_comparison", |b| {
        b.iter(|| {
            let _cost_a = cost_model.estimate_total_cost(black_box(&plan_a));
            let _cost_b = cost_model.estimate_total_cost(black_box(&plan_b));
        });
    });
}

fn benchmark_join_ordering(c: &mut Criterion) {
    // The real search (see join_reorder::JoinOrderer::find_optimal_order):
    // a 5-way chain, each join off the root table `t0`, with distinct
    // per-table selectivity so the search actually has a decision to
    // make (not just replaying one trivially-valid order).
    let orderer = JoinOrderer::new();
    let joins: Vec<JoinClause> = (1..=5)
        .map(|i| JoinClause {
            table: format!("t{i}"),
            alias: Some(format!("t{i}")),
            condition: Some(format!("t0.id = t{i}.t0_id")),
            equi_match: None,
            kind: shrestidb::sql::parser::JoinKind::Inner,
        })
        .collect();
    let row_count_of = |q: &str| if q == "t0" { 10_000 } else { 1_000 };
    let distinct_count_of = |q: &str, _c: &str| match q {
        "t0" => Some(10_000),
        "t1" => Some(10),
        "t2" => Some(100),
        "t3" => Some(1_000),
        "t4" => Some(500),
        "t5" => Some(50),
        _ => None,
    };

    c.bench_function("join_reorder_5_tables", |b| {
        b.iter(|| {
            orderer.find_optimal_order(black_box("t0"), black_box(10_000), &joins, row_count_of, distinct_count_of, 0.1);
        });
    });
}

criterion_group!(
    benches,
    benchmark_cardinality_estimation,
    benchmark_cardinality_categorical,
    benchmark_cost_model_estimation,
    benchmark_plan_cost_comparison,
    benchmark_join_ordering
);
criterion_main!(benches);
