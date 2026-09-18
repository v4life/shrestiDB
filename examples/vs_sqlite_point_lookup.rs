//! ShrestiDB vs SQLite: does the learned index's lookup speedup actually
//! show up end-to-end, against a real competitor, in some scenario?
//!
//! `examples/learned_index_demo.rs` already shows PGM/RMI beating this
//! codebase's own B-Tree by up to 36.8x on raw lookups -- but that's a
//! component-level number, isolated from everything a real query pays for
//! (parsing, row materialization, transaction/lock overhead). The one
//! place `vs_sqlite.rs` tries to show an index advantage end-to-end (its
//! "Range Scan" test) was never actually shaped to find one: it runs at
//! 20K rows, a scale where `learned_index_demo.rs`'s own numbers show PGM
//! *losing* to a B-Tree (the crossover is somewhere well above 100K), and
//! it's a range scan returning thousands of rows, so total time is
//! dominated by materializing that result, not by the initial search.
//!
//! This is the scenario actually built to find the advantage if one
//! exists: 1,000,000 rows (matching where the component-level win is
//! real and largest) and many *single-row* point lookups by primary key
//! (so the O(1)-ish learned-index search, not output volume, is what
//! dominates total time) -- run through the full SQL path on both
//! engines, real prepared statements, real in-memory storage, exactly
//! the same fairness rules `vs_sqlite.rs` already established. If the
//! learned-index thesis has a real, provable end-to-end edge over a
//! mature competitor, this is where it should appear. If it doesn't
//! appear even here, that's the honest answer too -- report it, don't
//! reshape the benchmark until it does.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rusqlite::Connection;
use shrestidb::execution::catalog::Catalog;
use shrestidb::execution::executor::QueryExecutor;
use shrestidb::execution::operators::Value;
use std::time::{Duration, Instant};

const NUM_ROWS: i64 = 1_000_000;
const NUM_LOOKUPS: usize = 20_000;

fn main() {
    println!("=== ShrestiDB vs SQLite: point-lookup latency at 1M rows ===");
    println!();

    // ── Load ──────────────────────────────────────────────────────────
    println!("Loading {NUM_ROWS} rows into both engines...");

    let sqlite = Connection::open_in_memory().unwrap();
    sqlite.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)", []).unwrap();
    {
        // A real transaction, not autocommit-per-row -- the fair
        // comparison point, matching how any real bulk load would be
        // done; per-statement autocommit would make this section about
        // fsync-equivalent overhead, not the index itself.
        let tx = sqlite.unchecked_transaction().unwrap();
        {
            let mut stmt = tx.prepare("INSERT INTO t (id, val) VALUES (?1, ?2)").unwrap();
            for id in 1..=NUM_ROWS {
                stmt.execute(rusqlite::params![id, id * 7]).unwrap();
            }
        }
        tx.commit().unwrap();
    }

    let shresti = QueryExecutor::new(Catalog::new());
    shresti.execute_sql("CREATE TABLE t (id INT PRIMARY KEY, val INT)").unwrap();
    let insert_stmt = shresti.prepare("INSERT INTO t (id, val) VALUES (?, ?)").unwrap();
    for id in 1..=NUM_ROWS {
        shresti.execute_prepared(&insert_stmt, &[Value::Integer(id), Value::Integer(id * 7)]).unwrap();
    }
    println!("Load done.");
    println!();

    // ── Point lookups ────────────────────────────────────────────────
    // A fixed seed -- reproducible ids across runs, not cherry-picked
    // for either engine (both sides look up the exact same sequence).
    let mut rng = StdRng::seed_from_u64(42);
    let lookup_ids: Vec<i64> = (0..NUM_LOOKUPS).map(|_| rng.gen_range(1..=NUM_ROWS)).collect();

    println!("Running {NUM_LOOKUPS} point lookups (SELECT * FROM t WHERE id = ?) on each engine...");

    let mut sqlite_stmt = sqlite.prepare("SELECT id, val FROM t WHERE id = ?1").unwrap();
    let start = Instant::now();
    for &id in &lookup_ids {
        let (rid, rval): (i64, i64) = sqlite_stmt.query_row([id], |r| Ok((r.get(0)?, r.get(1)?))).unwrap();
        std::hint::black_box((rid, rval));
    }
    let sqlite_elapsed = start.elapsed();

    let lookup_stmt = shresti.prepare("SELECT * FROM t WHERE id = ?").unwrap();
    let start = Instant::now();
    for &id in &lookup_ids {
        let row = shresti.execute_prepared(&lookup_stmt, &[Value::Integer(id)]).unwrap();
        std::hint::black_box(row);
    }
    let shresti_elapsed = start.elapsed();

    println!();
    println!("=== Results ===");
    print_result("SQLite", sqlite_elapsed);
    print_result("ShrestiDB", shresti_elapsed);
    println!();
    if shresti_elapsed < sqlite_elapsed {
        println!(
            "ShrestiDB is {:.2}x faster than SQLite on this workload.",
            sqlite_elapsed.as_secs_f64() / shresti_elapsed.as_secs_f64()
        );
    } else {
        println!(
            "ShrestiDB is {:.2}x SLOWER than SQLite on this workload -- the learned index's raw",
            shresti_elapsed.as_secs_f64() / sqlite_elapsed.as_secs_f64()
        );
        println!("lookup-speed advantage did not translate into an end-to-end win here either.");
        println!("Reporting this honestly: real per-statement overhead in the full SQL path");
        println!("(execute_prepared's lock/transaction per call, result materialization into");
        println!("owned Strings) still dominates the microseconds the index search itself saves.");
    }
}

fn print_result(name: &str, elapsed: Duration) {
    let per_lookup = elapsed.as_secs_f64() / NUM_LOOKUPS as f64;
    println!("{name}: {elapsed:.2?} total, {:.2}µs/lookup", per_lookup * 1_000_000.0);
}
