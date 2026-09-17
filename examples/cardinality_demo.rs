//! Real, data-driven cardinality estimation vs. a fixed constant
//!
//! Every query planner needs a selectivity estimate for a `WHERE` clause
//! to decide how much work a plan will do. Before `ANALYZE` existed, this
//! project's planner used one hardcoded constant (0.5) for every filter,
//! regardless of the actual data. This example shows exactly what that
//! costs on realistic, skewed data, and what a real, `ANALYZE`-driven
//! estimate looks like instead — reusing this project's own proven
//! learned-index technique (`PGMIndex`, already shown elsewhere in this
//! repo to beat a B-tree by up to 36.8x on lookups) as a piecewise-linear
//! model of the column's empirical CDF, rather than a synthetic
//! stand-in. See `optimizer::cardinality::ColumnDistribution` for the
//! full mechanism, and that module's doc comment for why an earlier
//! "LearnedCardinalityEstimator" in this codebase was replaced rather
//! than reused — it was never actually learned (fixed weights, a `train`
//! that never trained, and it never read the real statistics it was
//! given).
//!
//! The skew here is realistic, not cherry-picked to make the point look
//! better than it is: a 95/5 split is an ordinary "most orders ship,
//! some get cancelled" shape, not an extreme outlier.

use shrestidb::execution::catalog::Catalog;
use shrestidb::execution::executor::QueryExecutor;
use shrestidb::execution::operators::Value;

// The fixed heuristic this replaces always outputs exactly
// ASSUMED_TABLE_ROWS(1000) * DEFAULT_FILTER_SELECTIVITY(0.5) = 500,
// regardless of the real data -- so these are deliberately chosen to
// make the true count land somewhere else, not coincide with that
// constant the way a smaller/differently-shaped demo accidentally did
// during development (500 orders out of 10,000 at 5% is *also* 500,
// which would make the old heuristic look right by pure coincidence).
const NUM_ORDERS: i64 = 100_000;
const CANCELLED_FRACTION: f64 = 0.05;

fn main() {
    println!("=== Real cardinality estimation vs. a fixed constant ===");
    println!();

    let executor = QueryExecutor::new(Catalog::new());
    executor.execute_sql("CREATE TABLE orders (id INT PRIMARY KEY, amount INT)").unwrap();

    let cancelled_count = (NUM_ORDERS as f64 * CANCELLED_FRACTION) as i64;
    let shipped_count = NUM_ORDERS - cancelled_count;
    println!("Loading {NUM_ORDERS} orders: {shipped_count} 'shipped' (amount=10), {cancelled_count} 'cancelled' (amount=999)...");
    let insert_stmt = executor.prepare("INSERT INTO orders (id, amount) VALUES (?, ?)").unwrap();
    for i in 1..=shipped_count {
        executor.execute_prepared(&insert_stmt, &[Value::Integer(i), Value::Integer(10)]).unwrap();
    }
    for i in (shipped_count + 1)..=NUM_ORDERS {
        executor.execute_prepared(&insert_stmt, &[Value::Integer(i), Value::Integer(999)]).unwrap();
    }
    println!();

    let query = "SELECT * FROM orders WHERE amount >= 999";
    println!("Query: {query}");
    println!();

    let before = executor.explain(query).unwrap();
    println!("Plan estimate BEFORE ANALYZE (fixed 0.5 constant, every filter, regardless of data):");
    println!("  estimated_rows = {}", before.estimated_rows);
    println!();

    println!("Running ANALYZE orders...");
    executor.execute_sql("ANALYZE orders").unwrap();
    println!();

    let after = executor.explain(query).unwrap();
    println!("Plan estimate AFTER ANALYZE (real PGM-fitted CDF over the actual 'amount' values):");
    println!("  estimated_rows = {}", after.estimated_rows);
    println!();

    let ground_truth = executor.execute_sql(query).unwrap().len();
    println!("Ground truth (actual row count for this query): {ground_truth}");
    println!();

    println!("=== Summary ===");
    let before_error = (before.estimated_rows as f64 - ground_truth as f64).abs();
    let after_error = (after.estimated_rows as f64 - ground_truth as f64).abs();
    println!("Before ANALYZE: estimated {} vs. true {ground_truth} -- off by {before_error:.0} rows", before.estimated_rows);
    println!("After ANALYZE:  estimated {} vs. true {ground_truth} -- off by {after_error:.0} rows", after.estimated_rows);
    println!(
        "Fixed-constant error was {:.1}x larger than the real, ANALYZE-driven estimate's error.",
        before_error / after_error.max(1.0)
    );
}
