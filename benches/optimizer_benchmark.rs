use criterion::{black_box, criterion_group, criterion_main, Criterion};
use shresti::optimizer::cardinality::{
    LearnedCardinalityEstimator, QueryPredicate, ColumnStats,
};
use shresti::optimizer::cost_model::{CostModel, OperatorCost, OperatorType};
use shresti::optimizer::join_reorder::JoinOrderer;

fn benchmark_cardinality_estimation(c: &mut Criterion) {
    let mut estimator = LearnedCardinalityEstimator::new(10);
    
    // Add column statistics
    estimator.update_column_stats(ColumnStats {
        column_id: 0,
        min_value: 0.0,
        max_value: 1000.0,
        distinct_values: 100,
        null_count: 0,
    });

    let predicates = vec![
        QueryPredicate {
            column_id: 0,
            min_value: 100.0,
            max_value: 500.0,
            is_equality: false,
        },
    ];

    c.bench_function("cardinality_estimate_single_predicate", |b| {
        b.iter(|| {
            estimator.estimate_selectivity(black_box(&predicates));
        });
    });
}

fn benchmark_cardinality_multiple_predicates(c: &mut Criterion) {
    let mut estimator = LearnedCardinalityEstimator::new(10);
    
    for i in 0..10 {
        estimator.update_column_stats(ColumnStats {
            column_id: i,
            min_value: 0.0,
            max_value: 1000.0,
            distinct_values: 100,
            null_count: 0,
        });
    }

    let predicates = vec![
        QueryPredicate {
            column_id: 0,
            min_value: 100.0,
            max_value: 500.0,
            is_equality: false,
        },
        QueryPredicate {
            column_id: 1,
            min_value: 200.0,
            max_value: 800.0,
            is_equality: false,
        },
        QueryPredicate {
            column_id: 2,
            min_value: 300.0,
            max_value: 900.0,
            is_equality: true,
        },
    ];

    c.bench_function("cardinality_estimate_multiple_predicates", |b| {
        b.iter(|| {
            estimator.estimate_selectivity(black_box(&predicates));
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
    let orderer = JoinOrderer::new();
    let tables = vec![1, 2, 3, 4, 5];

    c.bench_function("join_reorder_5_tables", |b| {
        b.iter(|| {
            orderer.find_optimal_order(black_box(tables.clone()));
        });
    });
}

criterion_group!(
    benches,
    benchmark_cardinality_estimation,
    benchmark_cardinality_multiple_predicates,
    benchmark_cost_model_estimation,
    benchmark_plan_cost_comparison,
    benchmark_join_ordering
);
criterion_main!(benches);
