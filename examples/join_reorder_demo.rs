//! Real, data-driven join reordering vs. a cost model with no real signal
//!
//! `optimizer::join_reorder::JoinOrderer::find_optimal_order` searches
//! for the cheapest *provably resolvable* order to run a chain of joins
//! in (see that function's docs for what "provably resolvable" means and
//! why it's non-negotiable, not just a nice-to-have). Before `ANALYZE`
//! runs, every join's real distinct-value counts are unknown, so the
//! search has no real signal to act on -- every candidate order costs
//! the same by its own estimate, and it keeps the query's original
//! (here, deliberately worst-case) order. After `ANALYZE`, real per-
//! column distinct counts are available, the search finds a genuinely
//! cheaper order, and the executor -- which just walks whatever `Join`
//! nodes the planner emitted, in that order -- actually runs it that
//! way. This example measures the real wall-clock difference, not just
//! a different number printed.
//!
//! The shape: `accounts` (small) is joined to two tables, `transactions`
//! (a near-1:1 match on `account_id` -- cheap, barely expands the row
//! count) and `tags` (a *low-cardinality* join key, `region_id`, with
//! only 2 distinct values across many rows -- expensive, each account
//! matches thousands of tag rows). The query is written with the
//! expensive join FIRST, source-SQL order -- exactly the case that
//! benefits from being pushed later, since the cheap join should run
//! while the accumulated row count is still small, not after it's
//! already been blown up by the fan-out join.

use shrestidb::execution::catalog::Catalog;
use shrestidb::execution::executor::QueryExecutor;
use shrestidb::execution::operators::Value;
use std::time::Instant;

const NUM_ACCOUNTS: i64 = 200;
const NUM_TAGS: i64 = 10_000;

fn main() {
    println!("=== Real join reordering vs. a cost model with no real signal ===");
    println!();

    let executor = QueryExecutor::new(Catalog::new());
    executor.execute_sql("CREATE TABLE accounts (id INT PRIMARY KEY, region_id INT)").unwrap();
    executor.execute_sql("CREATE TABLE tags (id INT PRIMARY KEY, region_id INT)").unwrap();
    executor.execute_sql("CREATE TABLE transactions (id INT PRIMARY KEY, account_id INT)").unwrap();

    println!("Loading {NUM_ACCOUNTS} accounts, {NUM_ACCOUNTS} transactions (1 per account), {NUM_TAGS} tags...");
    println!("(accounts.region_id and tags.region_id each alternate between 2 values -- a real,");
    println!(" low-cardinality join key, the shape that makes tags a wide fan-out join.)");
    println!();

    let insert_account = executor.prepare("INSERT INTO accounts (id, region_id) VALUES (?, ?)").unwrap();
    for i in 1..=NUM_ACCOUNTS {
        let region = (i % 2) + 1;
        executor.execute_prepared(&insert_account, &[Value::Integer(i), Value::Integer(region)]).unwrap();
    }
    let insert_transaction = executor.prepare("INSERT INTO transactions (id, account_id) VALUES (?, ?)").unwrap();
    for i in 1..=NUM_ACCOUNTS {
        executor.execute_prepared(&insert_transaction, &[Value::Integer(1000 + i), Value::Integer(i)]).unwrap();
    }
    let insert_tag = executor.prepare("INSERT INTO tags (id, region_id) VALUES (?, ?)").unwrap();
    for i in 1..=NUM_TAGS {
        let region = (i % 2) + 1;
        executor.execute_prepared(&insert_tag, &[Value::Integer(i), Value::Integer(region)]).unwrap();
    }

    // Deliberately worst-case source order: the wide fan-out join
    // (tags) listed first, the cheap near-1:1 join (transactions) second.
    let query = "SELECT * FROM accounts a JOIN tags t ON a.region_id = t.region_id JOIN transactions x ON a.id = x.account_id";
    println!("Query (source order lists the expensive join first):");
    println!("  {query}");
    println!();

    let plan_before = executor.explain(query).unwrap();
    println!("Plan BEFORE ANALYZE (no real distinct counts -- every order costs the same, so the");
    println!("search keeps source order): {}", describe_join_order(&plan_before));
    println!("  estimated_cost = {:.1}", plan_before.estimated_cost);

    let start = Instant::now();
    let rows_before = executor.execute_sql(query).unwrap();
    let elapsed_before = start.elapsed();
    println!("  actual rows = {}, wall time = {elapsed_before:.2?}", rows_before.len());
    println!();

    println!("Running ANALYZE on all three tables...");
    executor.execute_sql("ANALYZE accounts").unwrap();
    executor.execute_sql("ANALYZE tags").unwrap();
    executor.execute_sql("ANALYZE transactions").unwrap();
    println!();

    let plan_after = executor.explain(query).unwrap();
    println!("Plan AFTER ANALYZE (real distinct counts -- transactions' near-1:1 join is genuinely");
    println!("cheaper to run first): {}", describe_join_order(&plan_after));
    println!("  estimated_cost = {:.1}", plan_after.estimated_cost);

    let start = Instant::now();
    let rows_after = executor.execute_sql(query).unwrap();
    let elapsed_after = start.elapsed();
    println!("  actual rows = {}, wall time = {elapsed_after:.2?}", rows_after.len());
    println!();

    println!("=== Summary ===");
    println!("Row count identical both times: {} (reordering changed performance, not correctness)", rows_before.len() == rows_after.len());
    println!("Estimated cost dropped {:.1}x ({:.1} -> {:.1})", plan_before.estimated_cost / plan_after.estimated_cost.max(1.0), plan_before.estimated_cost, plan_after.estimated_cost);
    println!(
        "Real wall time: {elapsed_before:.2?} before ANALYZE vs. {elapsed_after:.2?} after -- {:.1}x faster once the search had a real signal to act on.",
        elapsed_before.as_secs_f64() / elapsed_after.as_secs_f64().max(0.000_001)
    );
    println!();
    println!("Honest note: the wall-time speedup is smaller than the estimated-cost ratio.");
    println!("Both orders end up materializing and stringifying the same final ~1,000,000-row");
    println!("result -- that fixed cost doesn't depend on join order, so it dilutes the real,");
    println!("order-dependent difference in join-algorithm work. The cost model doesn't account");
    println!("for output materialization at all -- a real gap, not something to paper over.");
}

fn describe_join_order(plan: &shrestidb::optimizer::planner::PhysicalPlan) -> String {
    use shrestidb::optimizer::planner::LogicalPlanNode;
    let order: Vec<&str> = plan
        .nodes
        .iter()
        .filter_map(|n| match n {
            LogicalPlanNode::Join { right_table, .. } => Some(right_table.as_str()),
            _ => None,
        })
        .collect();
    format!("accounts -> {}", order.join(" -> "))
}
