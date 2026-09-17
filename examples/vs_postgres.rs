//! ShrestiDB vs Postgres: a real comparison, honestly framed
//!
//! Same TPC-H-lite workload as `examples/vs_sqlite.rs` (see that file's
//! module doc for the load/join fairness notes -- they apply here too),
//! against a real local Postgres over its normal client/server protocol
//! (`host=localhost port=5432`, `NoTls`).
//!
//! Unlike the SQLite comparison, this one is **not** apples-to-apples on
//! its own terms, and pretending otherwise would be dishonest: every
//! query against Postgres pays a real client/server round trip (even
//! over loopback, that's socket I/O and protocol serialization) that
//! ShrestiDB and SQLite -- both in-process, in-memory, same address space
//! as the benchmark itself -- never pay. Postgres is also a durable,
//! WAL-backed, disk-oriented system by default here (this doesn't disable
//! that), while this run of ShrestiDB is deliberately in-memory, same as
//! `vs_sqlite.rs`, to isolate execution cost from fsync cost. So: real
//! numbers, from a real local Postgres, but the delta includes IPC
//! overhead and durability Postgres pays and the other two don't -- not a
//! clean measurement of query execution alone. Take it as "where do we
//! stand against a real client/server RDBMS as actually deployed", not
//! "how fast is our execution engine relative to Postgres's".
//!
//! Needs a reachable Postgres with a database the connecting user can
//! create tables in. Set `SHRESTIDB_PG_URL` to override the default
//! (`host=localhost port=5432 user=<$USER> dbname=<$USER>`, matching a
//! default local Postgres.app / Homebrew install).

use postgres::{Client, NoTls};
use shrestidb::execution::catalog::Catalog;
use shrestidb::execution::executor::QueryExecutor;
use shrestidb::execution::operators::Value;
use std::time::{Duration, Instant};

const NUM_ORDERS: i64 = 20_000;
const NUM_CUSTOMERS: i64 = 2_000;
const JOIN_ORDERS: i64 = 3_000;
const JOIN_CUSTOMERS: i64 = 300;

fn pg_url() -> String {
    if let Ok(url) = std::env::var("SHRESTIDB_PG_URL") {
        return url;
    }
    let user = std::env::var("USER").unwrap_or_else(|_| "postgres".to_string());
    format!("host=localhost port=5432 user={user} dbname={user}")
}

fn pg_query(client: &mut Client, sql: &str) -> Vec<Vec<String>> {
    client
        .query(sql, &[])
        .unwrap()
        .iter()
        .map(|row| {
            (0..row.len())
                .map(|i| match row.columns()[i].type_() {
                    &postgres::types::Type::INT8 => row.get::<_, i64>(i).to_string(),
                    &postgres::types::Type::INT4 => row.get::<_, i32>(i).to_string(),
                    &postgres::types::Type::FLOAT8 => row.get::<_, f64>(i).to_string(),
                    &postgres::types::Type::TEXT | &postgres::types::Type::VARCHAR => row.get::<_, String>(i),
                    other => format!("<unhandled type {other:?}>"),
                })
                .collect()
        })
        .collect()
}

fn ratio(a: Duration, b: Duration) -> String {
    format!("{:.2}x", b.as_secs_f64() / a.as_secs_f64())
}

fn main() {
    println!("=== ShrestiDB vs Postgres (TPC-H-lite) ===");
    println!("Connecting: {}", pg_url());
    let mut pg = Client::connect(&pg_url(), NoTls).expect(
        "couldn't connect to Postgres -- set SHRESTIDB_PG_URL, or start a local Postgres on port 5432",
    );
    println!();

    let shresti = QueryExecutor::new(Catalog::new());
    shresti
        .execute_sql("CREATE TABLE orders (o_orderkey INT PRIMARY KEY, o_custkey INT, o_totalprice FLOAT)")
        .unwrap();
    shresti
        .execute_sql("CREATE TABLE customer (c_custkey INT PRIMARY KEY, c_name VARCHAR(50))")
        .unwrap();

    for table in ["orders", "customer", "join_orders", "join_customer"] {
        pg.execute(&format!("DROP TABLE IF EXISTS {table}"), &[]).unwrap();
    }
    pg.execute(
        "CREATE TABLE orders (o_orderkey BIGINT PRIMARY KEY, o_custkey BIGINT, o_totalprice DOUBLE PRECISION)",
        &[],
    )
    .unwrap();
    pg.execute("CREATE TABLE customer (c_custkey BIGINT PRIMARY KEY, c_name TEXT)", &[]).unwrap();

    println!("--- Load: {NUM_ORDERS} orders + {NUM_CUSTOMERS} customers ---");

    let start = Instant::now();
    {
        let orders_stmt = shresti
            .prepare("INSERT INTO orders (o_orderkey, o_custkey, o_totalprice) VALUES (?, ?, ?)")
            .unwrap();
        for i in 1..=NUM_ORDERS {
            let custkey = 1 + (i % NUM_CUSTOMERS);
            let price = 100.0 + (i as f64 * 1.337) % 5000.0;
            shresti
                .execute_prepared(&orders_stmt, &[Value::Integer(i), Value::Integer(custkey), Value::Float(price)])
                .unwrap();
        }
        let customer_stmt = shresti.prepare("INSERT INTO customer (c_custkey, c_name) VALUES (?, ?)").unwrap();
        for i in 1..=NUM_CUSTOMERS {
            shresti
                .execute_prepared(&customer_stmt, &[Value::Integer(i), Value::String(format!("Customer{i}"))])
                .unwrap();
        }
    }
    let shresti_load = start.elapsed();
    println!("  ShrestiDB (in-process, prepared statement):    {shresti_load:?}");

    let start = Instant::now();
    {
        let stmt = pg.prepare("INSERT INTO orders (o_orderkey, o_custkey, o_totalprice) VALUES ($1, $2, $3)").unwrap();
        for i in 1..=NUM_ORDERS {
            let custkey = 1 + (i % NUM_CUSTOMERS);
            let price = 100.0 + (i as f64 * 1.337) % 5000.0;
            pg.execute(&stmt, &[&i, &custkey, &price]).unwrap();
        }
        let stmt = pg.prepare("INSERT INTO customer (c_custkey, c_name) VALUES ($1, $2)").unwrap();
        for i in 1..=NUM_CUSTOMERS {
            pg.execute(&stmt, &[&i, &format!("Customer{i}")]).unwrap();
        }
    }
    let pg_load = start.elapsed();
    println!("  Postgres (prepared statement, client/server round trip per row): {pg_load:?}");
    println!("  ShrestiDB/Postgres: {}", ratio(pg_load, shresti_load));
    println!();

    println!("--- Range Scan: WHERE o_orderkey >= {} ---", NUM_ORDERS / 2);
    let start = Instant::now();
    let shresti_rows = shresti
        .execute_sql(&format!("SELECT * FROM orders WHERE o_orderkey >= {}", NUM_ORDERS / 2))
        .unwrap();
    let shresti_scan = start.elapsed();
    let start = Instant::now();
    let pg_rows = pg_query(&mut pg, &format!("SELECT * FROM orders WHERE o_orderkey >= {}", NUM_ORDERS / 2));
    let pg_scan = start.elapsed();
    println!("  ShrestiDB: {} rows in {:?} (PK-index-accelerated, in-process)", shresti_rows.len(), shresti_scan);
    println!("  Postgres:  {} rows in {:?} (PK B-tree, over the wire)", pg_rows.len(), pg_scan);
    println!("  ShrestiDB/Postgres: {}", ratio(pg_scan, shresti_scan));
    println!();

    println!("--- Aggregation: SELECT COUNT(*), SUM(o_totalprice) FROM orders ---");
    let start = Instant::now();
    let shresti_rows = shresti.execute_sql("SELECT COUNT(*), SUM(o_totalprice) FROM orders").unwrap();
    let shresti_agg = start.elapsed();
    let start = Instant::now();
    let pg_rows = pg_query(&mut pg, "SELECT COUNT(*), SUM(o_totalprice) FROM orders");
    let pg_agg = start.elapsed();
    println!("  ShrestiDB: {:?} in {:?}", shresti_rows[0], shresti_agg);
    println!("  Postgres:  {:?} in {:?}", pg_rows[0], pg_agg);
    println!("  ShrestiDB/Postgres: {}", ratio(pg_agg, shresti_agg));
    println!();

    shresti
        .execute_sql("CREATE TABLE join_orders (o_orderkey INT PRIMARY KEY, o_custkey INT, o_totalprice FLOAT)")
        .unwrap();
    shresti
        .execute_sql("CREATE TABLE join_customer (c_custkey INT PRIMARY KEY, c_name VARCHAR(50))")
        .unwrap();
    pg.execute(
        "CREATE TABLE join_orders (o_orderkey BIGINT PRIMARY KEY, o_custkey BIGINT, o_totalprice DOUBLE PRECISION)",
        &[],
    )
    .unwrap();
    pg.execute("CREATE TABLE join_customer (c_custkey BIGINT PRIMARY KEY, c_name TEXT)", &[]).unwrap();
    {
        let shresti_stmt = shresti
            .prepare("INSERT INTO join_orders (o_orderkey, o_custkey, o_totalprice) VALUES (?, ?, ?)")
            .unwrap();
        let stmt =
            pg.prepare("INSERT INTO join_orders (o_orderkey, o_custkey, o_totalprice) VALUES ($1, $2, $3)").unwrap();
        for i in 1..=JOIN_ORDERS {
            let custkey = 1 + (i % JOIN_CUSTOMERS);
            let price = 100.0 + (i as f64 * 1.337) % 5000.0;
            shresti
                .execute_prepared(&shresti_stmt, &[Value::Integer(i), Value::Integer(custkey), Value::Float(price)])
                .unwrap();
            pg.execute(&stmt, &[&i, &custkey, &price]).unwrap();
        }
        let shresti_stmt = shresti.prepare("INSERT INTO join_customer (c_custkey, c_name) VALUES (?, ?)").unwrap();
        let stmt = pg.prepare("INSERT INTO join_customer (c_custkey, c_name) VALUES ($1, $2)").unwrap();
        for i in 1..=JOIN_CUSTOMERS {
            shresti
                .execute_prepared(&shresti_stmt, &[Value::Integer(i), Value::String(format!("Customer{i}"))])
                .unwrap();
            pg.execute(&stmt, &[&i, &format!("Customer{i}")]).unwrap();
        }
    }

    println!("--- Join: join_orders JOIN join_customer ON o_custkey = c_custkey ({JOIN_ORDERS} x {JOIN_CUSTOMERS}) ---");
    let start = Instant::now();
    let shresti_rows = shresti
        .execute_sql("SELECT * FROM join_orders JOIN join_customer ON join_orders.o_custkey = join_customer.c_custkey")
        .unwrap();
    let shresti_join = start.elapsed();
    let start = Instant::now();
    let pg_rows = pg_query(
        &mut pg,
        "SELECT * FROM join_orders JOIN join_customer ON join_orders.o_custkey = join_customer.c_custkey",
    );
    let pg_join = start.elapsed();
    println!("  ShrestiDB: {} rows in {:?} (always nested loop)", shresti_rows.len(), shresti_join);
    println!("  Postgres:  {} rows in {:?} (planner's choice)", pg_rows.len(), pg_join);
    println!("  ShrestiDB/Postgres: {}", ratio(pg_join, shresti_join));
    println!();

    for table in ["orders", "customer", "join_orders", "join_customer"] {
        pg.execute(&format!("DROP TABLE IF EXISTS {table}"), &[]).unwrap();
    }

    println!("=== Summary (ShrestiDB time / Postgres time -- over 1.0x means ShrestiDB is slower) ===");
    println!("Load:        {}", ratio(pg_load, shresti_load));
    println!("Range Scan:  {}", ratio(pg_scan, shresti_scan));
    println!("Aggregation: {}", ratio(pg_agg, shresti_agg));
    println!("Join:        {}", ratio(pg_join, shresti_join));
}
