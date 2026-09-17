//! ShrestiDB vs SQLite: a real, in-process comparison
//!
//! Runs the same TPC-H-lite workload (see `examples/tpc_h.rs`) against
//! both engines: identical schema, identical data, identical query text
//! (modulo trivial dialect differences), both running fully in-memory (no
//! WAL on ShrestiDB's side, `:memory:` on SQLite's — neither touches
//! disk, so this measures execution, not fsync cost). Both engines
//! materialize every result row into owned `String`s before the clock
//! stops, so neither is being timed on lazy iteration while the other
//! does real work.
//!
//! Two things are *not* forced to parity, deliberately, because doing so
//! would misrepresent one engine or the other rather than compare them
//! honestly:
//!
//! - **The join query's plan.** Neither table has a secondary index on
//!   the join column on either engine. SQLite's query planner is free to
//!   pick whatever strategy it wants for that; ShrestiDB always does a
//!   nested loop (see `execution::executor`). That's not a handicap
//!   applied to either side — it's genuinely how each engine would run
//!   this query today.
//! - **The load phase.** SQLite is loaded through a real prepared,
//!   parameterized statement — the idiomatic, fast way any real SQLite
//!   user would bulk-insert. ShrestiDB has no prepared-statement or
//!   parameter-binding API at all yet: every `execute_sql` call re-parses
//!   a full SQL string from scratch (see `sql::parser`), even in a tight
//!   insert loop. Giving ShrestiDB a synthetic advantage by hand-rolling
//!   some faster non-public insert path would hide that gap rather than
//!   report it — so the load numbers below include ShrestiDB's real
//!   per-statement parse cost, and that asymmetry is the point, not an
//!   oversight.

use rusqlite::Connection;
use shrestidb::execution::catalog::Catalog;
use shrestidb::execution::executor::QueryExecutor;
use std::time::{Duration, Instant};

const NUM_ORDERS: i64 = 20_000;
const NUM_CUSTOMERS: i64 = 2_000;
const JOIN_ORDERS: i64 = 3_000;
const JOIN_CUSTOMERS: i64 = 300;

fn sqlite_query(conn: &Connection, sql: &str) -> Vec<Vec<String>> {
    let mut stmt = conn.prepare(sql).unwrap();
    let col_count = stmt.column_count();
    let rows = stmt
        .query_map([], |row| {
            (0..col_count)
                .map(|i| {
                    let value: rusqlite::types::Value = row.get(i)?;
                    Ok(match value {
                        rusqlite::types::Value::Null => "NULL".to_string(),
                        rusqlite::types::Value::Integer(v) => v.to_string(),
                        rusqlite::types::Value::Real(v) => v.to_string(),
                        rusqlite::types::Value::Text(v) => v,
                        rusqlite::types::Value::Blob(_) => "<blob>".to_string(),
                    })
                })
                .collect::<rusqlite::Result<Vec<String>>>()
        })
        .unwrap();
    rows.map(|r| r.unwrap()).collect()
}

fn ratio(a: Duration, b: Duration) -> String {
    format!("{:.2}x", b.as_secs_f64() / a.as_secs_f64())
}

fn main() {
    println!("=== ShrestiDB vs SQLite (TPC-H-lite) ===");
    println!();

    let shresti = QueryExecutor::new(Catalog::new());
    shresti
        .execute_sql("CREATE TABLE orders (o_orderkey INT PRIMARY KEY, o_custkey INT, o_totalprice FLOAT)")
        .unwrap();
    shresti
        .execute_sql("CREATE TABLE customer (c_custkey INT PRIMARY KEY, c_name VARCHAR(50))")
        .unwrap();

    let sqlite = Connection::open_in_memory().unwrap();
    sqlite
        .execute(
            "CREATE TABLE orders (o_orderkey INTEGER PRIMARY KEY, o_custkey INTEGER, o_totalprice REAL)",
            [],
        )
        .unwrap();
    sqlite
        .execute("CREATE TABLE customer (c_custkey INTEGER PRIMARY KEY, c_name TEXT)", [])
        .unwrap();

    println!("--- Load: {NUM_ORDERS} orders + {NUM_CUSTOMERS} customers ---");

    let start = Instant::now();
    for i in 1..=NUM_ORDERS {
        let custkey = 1 + (i % NUM_CUSTOMERS);
        let price = 100.0 + (i as f64 * 1.337) % 5000.0;
        shresti
            .execute_sql(&format!(
                "INSERT INTO orders (o_orderkey, o_custkey, o_totalprice) VALUES ({i}, {custkey}, {price})"
            ))
            .unwrap();
    }
    for i in 1..=NUM_CUSTOMERS {
        shresti
            .execute_sql(&format!("INSERT INTO customer (c_custkey, c_name) VALUES ({i}, 'Customer{i}')"))
            .unwrap();
    }
    let shresti_load = start.elapsed();
    println!("  ShrestiDB (re-parses SQL text per statement): {shresti_load:?}");

    let start = Instant::now();
    {
        let mut stmt = sqlite
            .prepare("INSERT INTO orders (o_orderkey, o_custkey, o_totalprice) VALUES (?1, ?2, ?3)")
            .unwrap();
        for i in 1..=NUM_ORDERS {
            let custkey = 1 + (i % NUM_CUSTOMERS);
            let price = 100.0 + (i as f64 * 1.337) % 5000.0;
            stmt.execute(rusqlite::params![i, custkey, price]).unwrap();
        }
        let mut stmt = sqlite.prepare("INSERT INTO customer (c_custkey, c_name) VALUES (?1, ?2)").unwrap();
        for i in 1..=NUM_CUSTOMERS {
            stmt.execute(rusqlite::params![i, format!("Customer{i}")]).unwrap();
        }
    }
    let sqlite_load = start.elapsed();
    println!("  SQLite (prepared, parameterized statement):   {sqlite_load:?}");
    println!("  ShrestiDB/SQLite: {}", ratio(sqlite_load, shresti_load));
    println!();

    println!("--- Range Scan: WHERE o_orderkey >= {} ---", NUM_ORDERS / 2);
    let start = Instant::now();
    let shresti_rows = shresti
        .execute_sql(&format!("SELECT * FROM orders WHERE o_orderkey >= {}", NUM_ORDERS / 2))
        .unwrap();
    let shresti_scan = start.elapsed();
    let start = Instant::now();
    let sqlite_rows = sqlite_query(&sqlite, &format!("SELECT * FROM orders WHERE o_orderkey >= {}", NUM_ORDERS / 2));
    let sqlite_scan = start.elapsed();
    println!("  ShrestiDB: {} rows in {:?} (PK-index-accelerated)", shresti_rows.len(), shresti_scan);
    println!("  SQLite:    {} rows in {:?} (rowid B-tree)", sqlite_rows.len(), sqlite_scan);
    println!("  ShrestiDB/SQLite: {}", ratio(sqlite_scan, shresti_scan));
    println!();

    println!("--- Aggregation: SELECT COUNT(*), SUM(o_totalprice) FROM orders ---");
    let start = Instant::now();
    let shresti_rows = shresti.execute_sql("SELECT COUNT(*), SUM(o_totalprice) FROM orders").unwrap();
    let shresti_agg = start.elapsed();
    let start = Instant::now();
    let sqlite_rows = sqlite_query(&sqlite, "SELECT COUNT(*), SUM(o_totalprice) FROM orders");
    let sqlite_agg = start.elapsed();
    println!("  ShrestiDB: {:?} in {:?}", shresti_rows[0], shresti_agg);
    println!("  SQLite:    {:?} in {:?}", sqlite_rows[0], sqlite_agg);
    println!("  ShrestiDB/SQLite: {}", ratio(sqlite_agg, shresti_agg));
    println!();

    // Separate, smaller tables for the join -- see examples/tpc_h.rs's
    // module doc on why (ShrestiDB's predicate re-parse-per-row cost).
    shresti
        .execute_sql("CREATE TABLE join_orders (o_orderkey INT PRIMARY KEY, o_custkey INT, o_totalprice FLOAT)")
        .unwrap();
    shresti
        .execute_sql("CREATE TABLE join_customer (c_custkey INT PRIMARY KEY, c_name VARCHAR(50))")
        .unwrap();
    sqlite
        .execute(
            "CREATE TABLE join_orders (o_orderkey INTEGER PRIMARY KEY, o_custkey INTEGER, o_totalprice REAL)",
            [],
        )
        .unwrap();
    sqlite
        .execute("CREATE TABLE join_customer (c_custkey INTEGER PRIMARY KEY, c_name TEXT)", [])
        .unwrap();
    {
        let mut sqlite_orders_stmt = sqlite
            .prepare("INSERT INTO join_orders (o_orderkey, o_custkey, o_totalprice) VALUES (?1, ?2, ?3)")
            .unwrap();
        for i in 1..=JOIN_ORDERS {
            let custkey = 1 + (i % JOIN_CUSTOMERS);
            let price = 100.0 + (i as f64 * 1.337) % 5000.0;
            shresti
                .execute_sql(&format!(
                    "INSERT INTO join_orders (o_orderkey, o_custkey, o_totalprice) VALUES ({i}, {custkey}, {price})"
                ))
                .unwrap();
            sqlite_orders_stmt.execute(rusqlite::params![i, custkey, price]).unwrap();
        }
        let mut sqlite_customer_stmt =
            sqlite.prepare("INSERT INTO join_customer (c_custkey, c_name) VALUES (?1, ?2)").unwrap();
        for i in 1..=JOIN_CUSTOMERS {
            shresti
                .execute_sql(&format!("INSERT INTO join_customer (c_custkey, c_name) VALUES ({i}, 'Customer{i}')"))
                .unwrap();
            sqlite_customer_stmt.execute(rusqlite::params![i, format!("Customer{i}")]).unwrap();
        }
    }

    println!("--- Join: join_orders JOIN join_customer ON o_custkey = c_custkey ({JOIN_ORDERS} x {JOIN_CUSTOMERS}) ---");
    let start = Instant::now();
    let shresti_rows = shresti
        .execute_sql("SELECT * FROM join_orders JOIN join_customer ON join_orders.o_custkey = join_customer.c_custkey")
        .unwrap();
    let shresti_join = start.elapsed();
    let start = Instant::now();
    let sqlite_rows = sqlite_query(
        &sqlite,
        "SELECT * FROM join_orders JOIN join_customer ON join_orders.o_custkey = join_customer.c_custkey",
    );
    let sqlite_join = start.elapsed();
    println!("  ShrestiDB: {} rows in {:?} (always nested loop)", shresti_rows.len(), shresti_join);
    println!("  SQLite:    {} rows in {:?} (planner's choice)", sqlite_rows.len(), sqlite_join);
    println!("  ShrestiDB/SQLite: {}", ratio(sqlite_join, shresti_join));
    println!();

    println!("=== Summary (ShrestiDB time / SQLite time -- over 1.0x means ShrestiDB is slower) ===");
    println!("Load:        {}", ratio(sqlite_load, shresti_load));
    println!("Range Scan:  {}", ratio(sqlite_scan, shresti_scan));
    println!("Aggregation: {}", ratio(sqlite_agg, shresti_agg));
    println!("Join:        {}", ratio(sqlite_join, shresti_join));
}
