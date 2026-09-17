//! TPC-C-lite: a durability-inclusive OLTP benchmark
//!
//! Inspired by TPC-C's New-Order and Payment transactions — its two
//! heaviest and most common transaction types (45% and 43% of the real
//! mix) — run concurrently across threads against a single shared,
//! WAL-backed `QueryExecutor`. This is explicitly *not* a spec-compliant
//! TPC-C run: no Order-Status/Delivery/Stock-Level transactions, no
//! per-district O_ID sequences or per-warehouse stock (stock is one shared
//! table here), no terminal think-time, and no official audited metric —
//! just the same transaction shapes and the same concurrent-write
//! contention pattern, against the real engine (MVCC + `LockManager` +
//! fsync'd WAL), not a placeholder loop.
//!
//! One real limitation this surfaced rather than hid: there's no
//! server-side expression evaluation in `UPDATE ... SET` (see
//! `execution::row_codec` — assignment values are parsed as literals, not
//! expressions) and no multi-statement SQL transaction, so `s_qty = s_qty -
//! 1` isn't expressible directly. Both transactions below do the
//! textbook-naive thing instead: `SELECT` the current value, compute the
//! new one in Rust, then `UPDATE` to that literal — which means, under
//! real concurrent contention on the same warehouse/customer row (which
//! this benchmark deliberately creates, with only a handful of warehouses
//! shared across all threads), a classic lost update is possible: the
//! `LockManager` only serializes the two `UPDATE`s themselves, not the
//! read-compute-write gap between the `SELECT` and the `UPDATE`. That's a
//! real, honest gap in this system today (no read-modify-write primitive,
//! no explicit `BEGIN`/`COMMIT` spanning statements), not a bug in this
//! benchmark — worth fixing separately if atomic increments matter.

use shrestidb::execution::executor::QueryExecutor;
use rand::Rng;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::time::Instant;

const NUM_WAREHOUSES: i64 = 4;
const CUSTOMERS_PER_WAREHOUSE: i64 = 50;
const NUM_STOCK_ITEMS: i64 = 100;
const NUM_THREADS: usize = 4;
const TRANSACTIONS_PER_THREAD: usize = 250;

#[derive(Default)]
struct Counters {
    new_order_ok: AtomicU64,
    new_order_failed: AtomicU64,
    payment_ok: AtomicU64,
    payment_failed: AtomicU64,
}

fn setup(executor: &QueryExecutor) {
    executor
        .execute_sql("CREATE TABLE warehouse (w_id INT PRIMARY KEY, w_balance FLOAT)")
        .unwrap();
    executor
        .execute_sql("CREATE TABLE customer (c_id INT PRIMARY KEY, c_w_id INT, c_balance FLOAT)")
        .unwrap();
    executor
        .execute_sql("CREATE TABLE stock (s_id INT PRIMARY KEY, s_qty INT)")
        .unwrap();
    executor
        .execute_sql("CREATE TABLE orders (o_id INT PRIMARY KEY, o_w_id INT, o_c_id INT, o_amount FLOAT)")
        .unwrap();

    for w_id in 1..=NUM_WAREHOUSES {
        executor
            .execute_sql(&format!("INSERT INTO warehouse (w_id, w_balance) VALUES ({w_id}, 0.0)"))
            .unwrap();
    }
    for c_id in 1..=(NUM_WAREHOUSES * CUSTOMERS_PER_WAREHOUSE) {
        let c_w_id = ((c_id - 1) % NUM_WAREHOUSES) + 1;
        executor
            .execute_sql(&format!(
                "INSERT INTO customer (c_id, c_w_id, c_balance) VALUES ({c_id}, {c_w_id}, 0.0)"
            ))
            .unwrap();
    }
    for s_id in 1..=NUM_STOCK_ITEMS {
        executor
            .execute_sql(&format!("INSERT INTO stock (s_id, s_qty) VALUES ({s_id}, 1000)"))
            .unwrap();
    }
}

/// Decrement a random stock item's quantity and record a new order — the
/// write-heavy transaction TPC-C weights at 45% of its mix.
fn new_order(executor: &QueryExecutor, next_order_id: &AtomicI64, rng: &mut impl Rng) -> bool {
    let w_id = rng.gen_range(1..=NUM_WAREHOUSES);
    let c_id = rng.gen_range(1..=(NUM_WAREHOUSES * CUSTOMERS_PER_WAREHOUSE));
    let s_id = rng.gen_range(1..=NUM_STOCK_ITEMS);
    let qty = rng.gen_range(1..=10);

    let Ok(rows) = executor.execute_sql(&format!("SELECT s_qty FROM stock WHERE s_id = {s_id}")) else {
        return false;
    };
    let Some(current_qty) = rows.first().and_then(|r| r[0].parse::<i64>().ok()) else {
        return false;
    };
    let new_qty = (current_qty - qty).max(0);
    if executor
        .execute_sql(&format!("UPDATE stock SET s_qty = {new_qty} WHERE s_id = {s_id}"))
        .is_err()
    {
        return false;
    }

    let order_id = next_order_id.fetch_add(1, Ordering::Relaxed);
    let amount = qty as f64 * 9.99;
    executor
        .execute_sql(&format!(
            "INSERT INTO orders (o_id, o_w_id, o_c_id, o_amount) VALUES ({order_id}, {w_id}, {c_id}, {amount})"
        ))
        .is_ok()
}

/// Move money from a customer to their warehouse's balance — TPC-C weights
/// this at 43% of its mix, and it's the transaction most likely to
/// contend here since only `NUM_WAREHOUSES` rows absorb every thread's
/// writes to the warehouse side.
fn payment(executor: &QueryExecutor, rng: &mut impl Rng) -> bool {
    let w_id = rng.gen_range(1..=NUM_WAREHOUSES);
    let c_id = rng.gen_range(1..=(NUM_WAREHOUSES * CUSTOMERS_PER_WAREHOUSE));
    let amount: f64 = rng.gen_range(1.0..500.0);

    let Ok(rows) = executor.execute_sql(&format!("SELECT c_balance FROM customer WHERE c_id = {c_id}")) else {
        return false;
    };
    let Some(current_balance) = rows.first().and_then(|r| r[0].parse::<f64>().ok()) else {
        return false;
    };
    if executor
        .execute_sql(&format!(
            "UPDATE customer SET c_balance = {} WHERE c_id = {c_id}",
            current_balance + amount
        ))
        .is_err()
    {
        return false;
    }

    let Ok(rows) = executor.execute_sql(&format!("SELECT w_balance FROM warehouse WHERE w_id = {w_id}")) else {
        return false;
    };
    let Some(current_w_balance) = rows.first().and_then(|r| r[0].parse::<f64>().ok()) else {
        return false;
    };
    executor
        .execute_sql(&format!(
            "UPDATE warehouse SET w_balance = {} WHERE w_id = {w_id}",
            current_w_balance + amount
        ))
        .is_ok()
}

fn main() {
    println!("=== TPC-C-lite ===");
    println!(
        "{NUM_WAREHOUSES} warehouses, {} customers, {NUM_STOCK_ITEMS} stock items, \
         {NUM_THREADS} threads x {TRANSACTIONS_PER_THREAD} transactions (New-Order/Payment, ~50/50)",
        NUM_WAREHOUSES * CUSTOMERS_PER_WAREHOUSE
    );
    println!();

    let dir = tempfile::tempdir().expect("failed to create temp dir for WAL");
    let executor = QueryExecutor::open(dir.path().join("tpcc.wal")).unwrap();
    setup(&executor);

    let next_order_id = AtomicI64::new(1);
    let counters = Counters::default();

    let start = Instant::now();
    std::thread::scope(|scope| {
        for _ in 0..NUM_THREADS {
            let executor = &executor;
            let next_order_id = &next_order_id;
            let counters = &counters;
            scope.spawn(move || {
                let mut rng = rand::thread_rng();
                for _ in 0..TRANSACTIONS_PER_THREAD {
                    if rng.gen_bool(0.5) {
                        match new_order(executor, next_order_id, &mut rng) {
                            true => counters.new_order_ok.fetch_add(1, Ordering::Relaxed),
                            false => counters.new_order_failed.fetch_add(1, Ordering::Relaxed),
                        };
                    } else {
                        match payment(executor, &mut rng) {
                            true => counters.payment_ok.fetch_add(1, Ordering::Relaxed),
                            false => counters.payment_failed.fetch_add(1, Ordering::Relaxed),
                        };
                    }
                }
            });
        }
    });
    let elapsed = start.elapsed();

    let new_order_ok = counters.new_order_ok.load(Ordering::Relaxed);
    let new_order_failed = counters.new_order_failed.load(Ordering::Relaxed);
    let payment_ok = counters.payment_ok.load(Ordering::Relaxed);
    let payment_failed = counters.payment_failed.load(Ordering::Relaxed);
    let total_ok = new_order_ok + payment_ok;
    let total = total_ok + new_order_failed + payment_failed;

    println!("Elapsed: {elapsed:?}");
    println!("New-Order: {new_order_ok} committed, {new_order_failed} failed");
    println!("Payment:   {payment_ok} committed, {payment_failed} failed");
    println!("Total:     {total_ok}/{total} committed");
    println!();
    println!(
        "tpmC (New-Order committed / minute): {:.1}",
        new_order_ok as f64 / elapsed.as_secs_f64() * 60.0
    );
    println!(
        "Overall throughput: {:.1} committed transactions/sec",
        total_ok as f64 / elapsed.as_secs_f64()
    );
}
