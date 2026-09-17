//! TPC-H-lite: real range-scan, aggregation, and join queries run through
//! the actual `QueryExecutor` -- replacing the previous version of this
//! file, which never touched `execute_sql` at all and only benchmarked the
//! raw PGM index's build/lookup in isolation, disconnected from any real
//! query.
//!
//! Explicitly not spec-compliant TPC-H: three columns across two tables,
//! not the real schema's eight tables, and no comparison baseline --
//! there's no "Traditional DB" anywhere in this codebase to benchmark
//! against, so this reports ShrestiDB's own real, measured numbers only,
//! not a fabricated speedup over something that was never actually run.
//!
//! In-memory (no WAL): TPC-H is read-heavy and analytical, where what
//! matters is scan/join/aggregate throughput, not commit durability --
//! unlike `examples/oltp.rs`'s TPC-C-lite benchmark, which is specifically
//! WAL-backed because durability is part of what TPC-C's numbers measure.
//!
//! The join runs at the same scale as the scan/aggregation tables now --
//! it didn't used to. Its `o_custkey = c_custkey` condition is a plain
//! equality between two bare columns, so `execution::executor` runs it as
//! a hash join (build a hash table on the smaller side, probe with the
//! larger) rather than a nested loop: O(orders + customers), not
//! O(orders * customers). Before that existed, this benchmark's join ran
//! on separate tables 6-7x smaller than the scan/aggregation ones,
//! specifically because the nested loop it had no alternative to was too
//! slow to run at matching scale -- a real, measured 496.8ms at 3,000 x
//! 300 even after the predicate-recompilation fix
//! (`row_codec::CompiledPredicate`) that preceded hash join. That
//! constraint is gone; this file no longer needs a second, smaller set of
//! tables just for the join query.

use shrestidb::execution::catalog::Catalog;
use shrestidb::execution::executor::QueryExecutor;
use shrestidb::execution::operators::Value;
use std::time::Instant;

const NUM_ORDERS: i64 = 20_000;
const NUM_CUSTOMERS: i64 = 2_000;

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
    let orders_stmt = executor
        .prepare("INSERT INTO orders (o_orderkey, o_custkey, o_totalprice) VALUES (?, ?, ?)")
        .unwrap();
    for i in 1..=NUM_ORDERS {
        let custkey = 1 + (i % NUM_CUSTOMERS);
        let price = 100.0 + (i as f64 * 1.337) % 5000.0;
        executor
            .execute_prepared(&orders_stmt, &[Value::Integer(i), Value::Integer(custkey), Value::Float(price)])
            .unwrap();
    }
    let customer_stmt = executor.prepare("INSERT INTO customer (c_custkey, c_name) VALUES (?, ?)").unwrap();
    for i in 1..=NUM_CUSTOMERS {
        executor.execute_prepared(&customer_stmt, &[Value::Integer(i), Value::String(format!("Customer{i}"))]).unwrap();
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

    println!("Join Query: SELECT * FROM orders JOIN customer ON orders.o_custkey = customer.c_custkey");
    let start = Instant::now();
    let rows = executor
        .execute_sql("SELECT * FROM orders JOIN customer ON orders.o_custkey = customer.c_custkey")
        .unwrap();
    let join_time = start.elapsed();
    println!("  {} rows ({NUM_ORDERS} orders x {NUM_CUSTOMERS} customers, hash join) in {:?}", rows.len(), join_time);
    println!();

    println!("=== Summary ===");
    println!("Load:        {load_time:?} ({NUM_ORDERS} orders + {NUM_CUSTOMERS} customers)");
    println!("Range Scan:  {range_scan_time:?} ({NUM_ORDERS} orders, PK-indexed)");
    println!("Aggregation: {agg_time:?} ({NUM_ORDERS} orders, full scan)");
    println!("Join Query:  {join_time:?} ({NUM_ORDERS} x {NUM_CUSTOMERS}, hash join)");
}
