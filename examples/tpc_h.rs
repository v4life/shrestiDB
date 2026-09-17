//! TPC-H-lite: real range-scan, aggregation, and join queries run through
//! the actual `QueryExecutor` -- replacing the previous version of this
//! file, which never touched `execute_sql` at all and only benchmarked the
//! raw PGM index's build/lookup in isolation, disconnected from any real
//! query.
//!
//! Explicitly not spec-compliant TPC-H: three columns across two tables,
//! not the real schema's eight tables, at a scale chosen to keep the join
//! query's runtime reasonable rather than the spec's GB-scale, and no
//! comparison baseline -- there's no "Traditional DB" anywhere in this
//! codebase to benchmark against, so this reports ShrestiDB's own real,
//! measured numbers only, not a fabricated speedup over something that was
//! never actually run.
//!
//! In-memory (no WAL): TPC-H is read-heavy and analytical, where what
//! matters is scan/join/aggregate throughput, not commit durability --
//! unlike `examples/oltp.rs`'s TPC-C-lite benchmark, which is specifically
//! WAL-backed because durability is part of what TPC-C's numbers measure.
//!
//! The join query's scale is deliberately modest: `evaluate_predicate`
//! re-parses its predicate string on every row pair rather than caching a
//! parsed expression tree (see `execution::row_codec`), so a nested-loop
//! join's cost includes that re-parsing cost on every one of its
//! `left * right` comparisons. Worth optimizing if join-heavy workloads at
//! real TPC-H scale ever matter here, but out of scope for this benchmark
//! to fix -- it just picks a scale that stays fast today.

use shrestidb::execution::catalog::Catalog;
use shrestidb::execution::executor::QueryExecutor;
use std::time::Instant;

const NUM_ORDERS: i64 = 20_000;
const NUM_CUSTOMERS: i64 = 2_000;
// Nested-loop join cost is O(left * right) with a predicate re-parse per
// pair (see module doc) -- kept far smaller than the scan/aggregation
// tables so the join query finishes in a reasonable time.
const JOIN_ORDERS: i64 = 3_000;
const JOIN_CUSTOMERS: i64 = 300;

fn main() {
    println!("=== TPC-H-lite ===");
    println!();

    let executor = QueryExecutor::new(Catalog::new());
    executor
        .execute_sql("CREATE TABLE orders (o_orderkey INT PRIMARY KEY, o_custkey INT, o_totalprice FLOAT)")
        .unwrap();
    executor
        .execute_sql("CREATE TABLE customer (c_custkey INT PRIMARY KEY, c_name VARCHAR(50))")
        .unwrap();

    println!("Loading {NUM_ORDERS} orders and {NUM_CUSTOMERS} customers...");
    let start = Instant::now();
    for i in 1..=NUM_ORDERS {
        let custkey = 1 + (i % NUM_CUSTOMERS);
        let price = 100.0 + (i as f64 * 1.337) % 5000.0;
        executor
            .execute_sql(&format!(
                "INSERT INTO orders (o_orderkey, o_custkey, o_totalprice) VALUES ({i}, {custkey}, {price})"
            ))
            .unwrap();
    }
    for i in 1..=NUM_CUSTOMERS {
        executor
            .execute_sql(&format!("INSERT INTO customer (c_custkey, c_name) VALUES ({i}, 'Customer{i}')"))
            .unwrap();
    }
    let load_time = start.elapsed();
    println!("Loaded in {load_time:?}");
    println!();

    println!("Range Scan: SELECT * FROM orders WHERE o_orderkey >= {}", NUM_ORDERS / 2);
    let start = Instant::now();
    let rows = executor
        .execute_sql(&format!("SELECT * FROM orders WHERE o_orderkey >= {}", NUM_ORDERS / 2))
        .unwrap();
    let range_scan_time = start.elapsed();
    println!("  {} rows in {:?} (PK-index-accelerated)", rows.len(), range_scan_time);
    println!();

    println!("Aggregation: SELECT COUNT(*), SUM(o_totalprice) FROM orders");
    let start = Instant::now();
    let rows = executor.execute_sql("SELECT COUNT(*), SUM(o_totalprice) FROM orders").unwrap();
    let agg_time = start.elapsed();
    println!("  count={}, sum={} in {:?} (full scan)", rows[0][0], rows[0][1], agg_time);
    println!();

    // Separate, smaller tables for the join -- see module doc on why.
    executor
        .execute_sql("CREATE TABLE join_orders (o_orderkey INT PRIMARY KEY, o_custkey INT, o_totalprice FLOAT)")
        .unwrap();
    executor
        .execute_sql("CREATE TABLE join_customer (c_custkey INT PRIMARY KEY, c_name VARCHAR(50))")
        .unwrap();
    for i in 1..=JOIN_ORDERS {
        let custkey = 1 + (i % JOIN_CUSTOMERS);
        let price = 100.0 + (i as f64 * 1.337) % 5000.0;
        executor
            .execute_sql(&format!(
                "INSERT INTO join_orders (o_orderkey, o_custkey, o_totalprice) VALUES ({i}, {custkey}, {price})"
            ))
            .unwrap();
    }
    for i in 1..=JOIN_CUSTOMERS {
        executor
            .execute_sql(&format!("INSERT INTO join_customer (c_custkey, c_name) VALUES ({i}, 'Customer{i}')"))
            .unwrap();
    }

    println!(
        "Join Query: SELECT * FROM join_orders JOIN join_customer ON join_orders.o_custkey = join_customer.c_custkey"
    );
    let start = Instant::now();
    let rows = executor
        .execute_sql(
            "SELECT * FROM join_orders JOIN join_customer ON join_orders.o_custkey = join_customer.c_custkey",
        )
        .unwrap();
    let join_time = start.elapsed();
    println!(
        "  {} rows ({JOIN_ORDERS} orders x {JOIN_CUSTOMERS} customers, nested loop) in {:?}",
        rows.len(),
        join_time
    );
    println!();

    println!("=== Summary ===");
    println!("Load:       {NUM_ORDERS} orders + {NUM_CUSTOMERS} customers in {load_time:?}");
    println!("Range Scan: {range_scan_time:?} ({NUM_ORDERS} orders, PK-indexed)");
    println!("Aggregation: {agg_time:?} ({NUM_ORDERS} orders, full scan)");
    println!("Join Query: {join_time:?} ({JOIN_ORDERS} x {JOIN_CUSTOMERS}, nested loop)");
}
