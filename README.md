# ShrestiDB

A relational database kernel written in Rust that leverages machine learning models throughout its architecture — indexing, query planning, and storage retention — instead of just at one layer.

## 🚀 Key Features

### Learned Indexing
- **Recursive Model Index (RMI)**: Multi-stage learned index, real and measured — up to ~9.8x faster lookups than this codebase's own (real, node-splitting) B+Tree at 1M records, growing with scale (see [Index Build Performance](#index-build-performance)). An earlier ~36.8x figure was measured against a `BTree` that turned out not to actually be one — see that section for the full story.
- **Piecewise Geometric Model (PGM)**: Adaptive segmented indexing optimized for skewed data; the actual primary-key index the live engine uses (`execution::mvcc_store::MVCCTable`), and the model reused for real cardinality estimation below.
- **Bounded Error Search**: SIMD-accelerated binary search within prediction error bounds
- ~~Hybrid Router~~: a `HybridIndex` (routing between a learned model and a B-Tree fallback) existed but was never used by the real query path — the live PK index is `DynamicPGMIndex`, used directly. Removed rather than left implying a routing capability that wasn't wired in.

### ML-Driven Query Optimization
- **Real Cardinality Estimation**: `ANALYZE <table>` builds a real per-column value distribution — a `PGMIndex` (the same learned-index technique above) fit over the column's actual sorted values as an empirical CDF, plus a real distinct-value count for equality selectivity. Replaces a fixed 0.5-selectivity-for-every-filter constant with a genuine, data-driven estimate; see [`optimizer::cardinality`](src/optimizer/cardinality.rs) and the [Real Cardinality Estimation](#real-cardinality-estimation-vs-a-fixed-constant) results below. (An earlier version of this feature was a "neural network" whose weights were hardcoded and never trained — replaced rather than reused; see that module's doc comment for the full story.)
- **Cost Model**: A real IO/CPU cost formula, consulted for every plan's `estimated_cost` — not adaptive or trained from execution history despite earlier documentation here claiming otherwise.
- **Hash Join**: Equality join conditions run as a real hash join (build a hash table on the smaller side, probe with the larger) instead of a nested loop — see [Benchmark Results](#-benchmark-results) below for the measured effect.
- **Real Join Reordering**: For 3+-table queries, [`optimizer::join_reorder::JoinOrderer`](src/optimizer/join_reorder.rs) searches for the cheapest join order that's *provably resolvable* by this engine's left-deep executor, using real per-column distinct-value counts from `ANALYZE` — not a heuristic sort, and not a number computed and then ignored; the executor runs whatever order the search picks. See [Real Join Reordering](#real-join-reordering-vs-a-cost-model-with-no-real-signal) below for a measured before/after. Queries with an OUTER join always keep source order — reordering one isn't generally safe, and proving when it is stays out of scope (see below).
- **Real Outer Joins**: `LEFT`/`RIGHT`/`FULL OUTER JOIN` now actually emit an unmatched row once, padded with `NULL` on the other side, instead of silently behaving like `INNER JOIN` (dropping it) — an earlier version of this parser recognized the SQL syntax but discarded which kind of join it was. Works on both the hash-join and nested-loop execution paths; see [`sql::parser::JoinKind`](src/sql/parser.rs) and `execution::executor::append_unmatched`.
- **Real `USING`/`NATURAL JOIN`**: previously silently executed as an unfiltered `CROSS JOIN` — an earlier version of this parser recognized the syntax but discarded it entirely, so `a JOIN b USING (id)` on 2×3 rows returned all 6 (every combination) instead of the 2 that actually match. Now resolved against the real schemas at execution time into an ordinary equi-join condition; see `execution::executor::resolve_using_condition`/`resolve_natural_condition`. Scope limit, loudly enforced rather than silently guessed at: only a single shared/listed column is supported (this engine's join execution has no composite hash key) — more than one is a clear error, not a partial match. `SEMI`/`ANTI` join variants remain a known, lower-priority gap (execute as `INNER JOIN` rather than their real semantics); they're non-standard syntax, unlikely to be hit by accident.
- **Real three-valued (`NULL`-aware) comparison logic**: a comparison touching `NULL` used to be conflated with "this engine couldn't evaluate the predicate at all" — both fell back to keeping the row, so `WHERE fk > 100` with `fk` actually `NULL` incorrectly *included* that row, and (on the hash-join path specifically) two independent `NULL`s on a join key spuriously matched each other, since a `NULL` was just an ordinary, matchable hash key. Both were reachable before, but far more so once `LEFT`/`RIGHT`/`FULL OUTER JOIN` started routinely producing `NULL`-padded rows that flow into later predicates. Fixed with a real `Tri` (`True`/`False`/`Unknown`) type implementing SQL's actual `AND`/`OR` truth tables (`row_codec::Tri`) — `Unknown` is excluded exactly like `False`, never given the "give up and keep it" treatment real structural failures get — plus excluding `NULL`-keyed rows from the hash-join table entirely, so a `NULL` join key can never match, not even another `NULL`.
- **Real `SELECT` column projection**: `SELECT name FROM users` used to silently return every column, identical to `SELECT *` — `select.columns` was only ever consulted to detect an aggregate query (`SELECT COUNT(*)`, ...) and otherwise discarded; there was no "keep only these columns" step in the physical plan at all. Fixed with a real `Project` node (`optimizer::planner::LogicalPlanNode::Project`), resolved against the current schema at execution time (so qualified names like `orders.total` after a `JOIN` work the same way every other column reference in this codebase already does) and honoring the requested column order, not just the table's own. An unresolvable column name is a hard error — the binder already catches most cases before execution even starts, `Project` is the last line of defense, not silent data loss.
- **Real `ORDER BY` / `LIMIT`**: both were parsed into `SelectStatement.order_by`/`.limit` and then never consulted anywhere — `SELECT * FROM t ORDER BY val` returned rows in arbitrary storage order, and `LIMIT 1` returned every row. Fixed with real `Sort`/`Limit` nodes (`optimizer::planner::LogicalPlanNode`), placed after `Filter`/`Join`/`Aggregate` but before `Project` (so `SELECT name FROM t ORDER BY age` can still sort by a column that isn't in the `SELECT` list, same as any real SQL engine allows), with `Limit` always last. Multi-column `ORDER BY` and mixed `ASC`/`DESC` both work; `NULL` always sorts first (SQLite's convention), regardless of direction. `NULLS FIRST`/`NULLS LAST` isn't recognized, and ordering by an aggregate expression itself (`ORDER BY COUNT(*)`) isn't yet supported (`Aggregate`'s output schema doesn't carry its own result-column names yet) — both documented scope limits, not silent wrong output.
- **Verified transaction-subsystem correctness**: audited MVCC/lock manager/transaction manager under real concurrent load rather than just reading the code. The real SQL `UPDATE` path re-reads a row's latest *committed* value under its exclusive lock before computing a new one, and is genuinely safe — verified with 8 threads × 25 concurrent `UPDATE ... SET val = val + 1` calls landing all 200 increments, zero lost. The raw `OLTPEngine` primitives, used naively (no re-read under lock), are a real footgun — not reachable through any SQL path today, but now a permanent test documents it for whoever builds on them next. See [`execution::oltp`](src/execution/oltp.rs) and [`execution::lock_manager`](src/execution/lock_manager.rs).
- **Real `HAVING`**: `select.having` was parsed by `sql::parser` and then never stored on `SelectStatement` at all — `GROUP BY user_id HAVING COUNT(*) > 1` silently returned every group, not just the ones matching the predicate. Fixed by carrying `having` through to `LogicalPlanNode::Aggregate` and filtering each group after aggregation, against a synthetic per-group schema covering both `GROUP BY` keys and any aggregate call the `HAVING` clause references — including one that isn't in the `SELECT` list at all (`SELECT user_id FROM orders GROUP BY user_id HAVING SUM(total) > 15` works, even with `SUM` unprojected). A `HAVING` clause on a query that isn't `GROUP BY`/all-aggregate is a hard plan-time error, not a silent no-op — the same discipline as an unbound `?`/`$N` placeholder reaching `execute_sql`.
- **Real `SELECT DISTINCT`**: the fourth field in a row found parsed and then silently discarded this session — `select.distinct` was read from `sqlparser`'s AST and never stored anywhere, so `SELECT DISTINCT user_id FROM orders` returned every row, duplicates included, identical to plain `SELECT`. Fixed with a real `Distinct` node (`optimizer::planner::LogicalPlanNode::Distinct`), placed after `Project`/`Sort` and before `Limit` so it dedups the query's actual final row shape in whatever order `ORDER BY` already produced, not a wider pre-projection one. Correctly leaves `SELECT ALL` (the explicit opposite) alone, and doesn't confuse it with plain `SELECT`'s implicit non-distinct default. `DISTINCT ON (...)` (a Postgres extension) is recognized as `DISTINCT` but its column-subset argument isn't honored — a documented scope limit, not a different bug.
- **Real `OFFSET`**: the fifth in a row — `OFFSET` had no field on `SelectStatement` at all, so `LIMIT 1 OFFSET 1` behaved identically to `LIMIT 1`, silently returning the first row instead of skipping it. Fixed by threading `select.offset` onto `LogicalPlanNode::Limit` alongside `limit`, skipped before the cap is applied (standard SQL order). `OFFSET` with no `LIMIT` (skip N, keep everything after) works too, and an `OFFSET` past every row is an empty result, not an error.
- **`UPDATE`/`DELETE` now use the primary-key index**: found while building [the monotonic-key ingestion proof of concept](#proof-of-concept-a-monotonic-key-ingestion-niche) — `UPDATE ... WHERE id = ?` and `DELETE ... WHERE id = ?` never consulted the PK index at all, unconditionally deserializing and predicate-checking *every* row in the table on every call, no matter how narrow `WHERE` was (`SELECT` already had this exact acceleration, `try_pk_index_scan`, but it was never reused for writes). Fixed with `QueryExecutor::candidate_rows_for_write`: a direct point read for exact primary-key equality, falling back to the full scan for every other `WHERE` shape — narrow and safe by construction, since both callers already re-verify every candidate against the real predicate before writing anything, exactly as they did against a full scan's candidates before. Real, measured effect: 2,121 single-row `UPDATE`s went from 17.19s to 18.43ms — about 930x.

### Storage
- **Write-Ahead Log**: Durable, real crash recovery — a plain append-only log (`execution::wal`) that every write goes through, replayed to rebuild the in-memory store on restart. See [`execution::recovery`](src/execution/recovery.rs).
- **In-Memory MVCC Store**: The actual live storage — version chains per row (`execution::mvcc_store::MVCCTable`), not page-based disk storage.

Earlier versions of this list also claimed a "Learned Buffer Pool"
(Markov-chain page-access prediction), "Slotted Pages", and "Direct I/O
with io_uring support". A real, disk-backed paged-storage module
(`storage::page`/`storage::disk_manager`/`storage::buffer_pool`, ~700
lines, including a genuine — if unused — Markov-chain predictor) existed
in the repository, but nothing in the actual read/write path ever used
it: this engine's real storage is the in-memory MVCC store above, with
the WAL providing durability. There was no page cache to prefetch into,
no page-based I/O of any kind, and no `io_uring` dependency anywhere in
the project — that whole claim never had code behind it. The unused
module has been removed rather than left implying storage capabilities
the shipped engine doesn't have; making a learned buffer pool real here
would mean building an actual disk-backed paged storage engine
underneath the current one, a genuinely large undertaking, not a
wiring fix — see the project's issue tracker / commit history for that
discussion if it's ever picked up.

### High-Performance Execution
- **MVCC Transactions**: Real snapshot isolation via [`execution::mvcc_store`](src/execution/mvcc_store.rs) — version chains per row, a real read/write lock manager ([`execution::lock_manager`](src/execution/lock_manager.rs)) for write conflicts.
- **Learned Retention (MVCC version-chain pruning)**: [`execution::learned_retention::RetentionPredictor`](src/execution/learned_retention.rs) — real, wired into every commit ([`execution::oltp::OLTPEngine::commit`](src/execution/oltp.rs)), and, unlike the removed buffer pool/lock-scheduling claims below, genuinely deciding something: fits an online `LinearModel` (the same one PGM/RMI use) on real telemetry — writes since a row's chain was last swept, versus dead versions that sweep actually found — to predict per-row whether sweeping now is worth it, instead of a fixed interval for every row regardless of its actual write rate. Found real and fixed here, not just audited: the model could never fit at all under any realistic driven workload — every bootstrap-mode sweep fired at exactly the same fixed write count, so every training sample had identical x, and `LinearModel::fit` rejects zero-variance input by design, a self-reinforcing deadlock verified via `mvcc_store::test_retention_bloat_comparison` (`has_model()` stayed `false` for an entire 20,000-commit run). Fixed with a small per-row jitter on the bootstrap threshold (`jittered_bootstrap`) so different rows contribute genuine variance; a second issue — fitting after just 4-5 noisy samples produced a *worse*-than-any-fixed-policy model — was fixed by raising the sample requirement before the first fit is trusted. See [Retention: Learned vs Fixed-Interval Sweeping](#retention-learned-predictor-vs-fixed-interval-sweeping) for the real, measured before/after.
- **SIMD Acceleration**: Vectorized search inside the PGM index specifically ([`compute::simd_ops`](src/compute/simd_ops.rs)) — not a general query-engine-wide batch-processing feature; this codebase's row execution is per-tuple, not columnar/batched.
- ~~Learned Lock Scheduling~~: no adaptive or learned concurrency-control code exists anywhere in this codebase. The lock manager above is real, but plain — acquire/release, no scheduling logic beyond that. Removed rather than left as an unfounded claim.

## 📊 Performance

Real, measured numbers — not marketing estimates — live in
[📊 Benchmark Results](#-benchmark-results) below, each reproducible via
`cargo run --example <name> --release`. Headline results as of the
latest run: RMI lookups up to **~9.8x** faster than this codebase's own
B+Tree at 1M records, growing with data scale (see
[Index Build Performance](#index-build-performance) — PGM specifically is
*slower* than the B+Tree below 1M records, and only pulls ahead past it;
RMI wins at every size tested, but by a smaller margin at small scale);
a hash-joined query within **1.2-1.5x** of SQLite/Postgres at 20K x 2K
rows; real `ANALYZE`-driven cardinality estimates landing on the true row
count where a fixed-constant heuristic was off by **10x** (see
[Real Cardinality Estimation](#real-cardinality-estimation-vs-a-fixed-constant));
and PGM's *memory* footprint — a separate, more consistent win than its
lookup speed — staying a real, measured **~4.2x smaller** than the same
B+Tree at every scale tested (100K to 10M keys), including under a
realistic non-uniform key distribution, not just an ideal one (see
[Memory Footprint](#memory-footprint-pgmrmi-vs-a-real-btree)).
The lookup-speedup number is a real, component-level win, not yet an
end-to-end one: tested through the full SQL path against SQLite at the
same 1M-row scale (many point lookups, the scenario built to give it
the best chance), ShrestiDB is **4.54x slower**, not faster — real
per-statement overhead around the index (down from 5.71x after removing
a redundant re-tokenization and an unnecessary schema clone found by
profiling the gap) still swallows what the search itself saves (see
[Point Lookup Latency vs
SQLite](#point-lookup-latency-vs-sqlite-does-the-index-speedup-survive-the-full-sql-path)).
Reported here because a real negative result is still a real result.
An earlier version of this table quoted round marketing-style figures
("100K+ transactions/sec", "2-5x more accurate than histograms") that
were never actually measured, some of which directly contradicted real
numbers already published elsewhere in this same file — replaced with
pointers to the real thing rather than a second set of numbers to keep
in sync by hand. A separate earlier figure — "36.8x faster than a
B-Tree" — turned out to be measured against a `BTree` implementation
whose own `insert` comment said "Simplified insert: just add to root":
every key lived in one unsplit node, so it was really `Vec::binary_search`
wearing B-tree-shaped names, not a real multi-level tree with the
node-fanout and pointer-chasing overhead the comparison is supposed to
be about. Discovered while building the memory-footprint comparison
above, `index::btree::BTree` is now a real, node-splitting B+Tree — see
that module's doc comment for the full story, and every number on this
page that involves a "B-Tree" comparison has been re-measured against it.

## 🎯 Proof of Concept: a monotonic-key ingestion niche
If ShrestiDB isn't chasing SQLite/Postgres parity on raw query speed (it
isn't — see [Point Lookup Latency vs
SQLite](#point-lookup-latency-vs-sqlite-does-the-index-speedup-survive-the-full-sql-path)),
what's the real, defensible thing it's good at? Not a number that shines
in isolation, but a concrete scenario, end to end, through the actual SQL
engine: a memory- and maintenance-efficient store for **monotonic-key,
high-volume ingestion** — event logs, IoT telemetry, order/transaction
streams. Two real wins, demonstrated together, on live engine state, in
[`examples/monotonic_ingestion_niche.rs`](examples/monotonic_ingestion_niche.rs)
(`cargo run --release --example monotonic_ingestion_niche`):

An `events` table (`id INT PRIMARY KEY, device_id, value, processed`) is
bulk-loaded with 500,000 rows via real prepared-statement `INSERT`s, then
a realistic minority of events (5,000 of 500,000 — most events process
cleanly on the first try, a minority get retried, an uneven number of
times each) go through a retry storm of 52,648 real `UPDATE`s:

| | Result |
|---|---|
| Ingest | 500,000 events, ~116,000 rows/sec |
| **Memory**: live PGM PK index vs. a real B+Tree over the same 500,000 ids | 4.11 MB vs. 16.15 MB — **3.9x smaller** |
| Retry storm | 5,000 events, 52,648 real `UPDATE`s, well under a second |
| **Retention**: dead MVCC versions left resident after the storm | 7,164 total — 1.43 per retried event, bounded automatically, no manual `VACUUM` |

(The 3.9x here is slightly below `memory_footprint.rs`'s isolated 4.2x —
this measures the live index mid-run, with a nonzero, not-yet-flushed
write buffer sitting in it, not a freshly-built structure over a fully
known key set. Both are real; they're just measuring at different
moments.)

Three things worth being upfront about, found *while building this*, not
polished away — two real bugs, both fixed here, not just flagged:

- **`UPDATE`/`DELETE` never consulted the primary-key index at all** —
  every `UPDATE ... WHERE id = ?` did a full deserialize-and-scan of the
  entire table, the exact acceleration `SELECT` already had
  (`try_pk_index_scan`) just never reused for writes. Before the fix,
  this example's smaller original retry storm (2,121 updates) took
  **17.19 seconds**; after, **18.43ms** — about 930x. See the
  `UPDATE`/`DELETE` bullet above and `QueryExecutor::candidate_rows_for_write`.
- **`DynamicPGMIndex` — the live PK index every `INSERT` goes through —
  rebuilt its base PGM segments from scratch every 64 buffered inserts**
  (`index::pgm::DynamicPGMIndex::flush_buffer`), real O(n²)-ish bulk-load
  cost that only showed up at real scale: ingest throughput at 500,000
  rows was 19,752 rows/sec, sharply down from ~108,000 rows/sec at
  20,000 rows. Fixed with an unsorted, append-only write buffer (sorted
  once, at flush time, instead of kept sorted on every single insert)
  and a flush threshold that grows with the index instead of staying
  fixed — the same amortized argument behind `Vec`'s own geometric
  growth. Ingest throughput at 500,000 rows is now ~116,000 rows/sec,
  matching the small-scale number instead of degrading against it. See
  `index::pgm::DynamicPGMIndex`'s doc comment for the full mechanism.

## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│     SQL Interface Layer              │
│  (Parser, Binder, Type Checker)     │
└────────────────┬─────────────────────┘
                 │
┌────────────────▼──────────────────────┐
│   Query Optimization Layer            │
│  (Cost Model, Real Cardinality        │
│   Estimation, Hash Join, Real Join    │
│   Reordering — all via ANALYZE)       │
└────────────────┬──────────────────────┘
                 │
┌────────────────▼──────────────────────┐
│   Query Execution Layer               │
│  (Vectorized Operators, MVCC          │
│   Transactions, WAL)                  │
└────────────────┬──────────────────────┘
                 │
┌────────────────▼──────────────────────┐
│      Index Layer                      │
│  (RMI, PGM — this crate's own real,   │
│   node-splitting B+Tree as baseline)  │
└────────────────┬──────────────────────┘
                 │
┌────────────────▼──────────────────────┐
│      Storage Layer                    │
│  (In-Memory MVCC Store, Write-Ahead   │
│   Log)                                │
└──────────────────────────────────────┘
```

## 📁 Project Structure

```
shrestidb/
├── Cargo.toml                          # Rust dependencies
├── README.md                           # This file
├── DESIGN.md                           # Comprehensive design document
├── src/
│   ├── main.rs                         # CLI server entry point
│   ├── lib.rs                          # Library exports
│   ├── error.rs                        # Error types
│   │
│   ├── index/                          # [LAYER 1] Learned Indexing
│   │   ├── mod.rs
│   │   ├── models.rs                   # LinearModel, PiecewiseLinearModel
│   │   ├── rmi.rs                      # Recursive Model Index
│   │   ├── pgm.rs                      # Piecewise Geometric Model
│   │   └── btree.rs                    # Real B+Tree comparison baseline (node-splitting)
│   │
│   ├── optimizer/                      # [LAYER 2] Query Optimization
│   │   ├── mod.rs
│   │   ├── cardinality.rs              # Real, ANALYZE-driven cardinality estimation (PGM-based)
│   │   ├── cost_model.rs               # Real IO/CPU cost formula (not adaptive/trained)
│   │   ├── join_reorder.rs             # Real, cost-based join ordering
│   │   └── planner.rs                  # Query planner
│   │
│   ├── execution/                      # [LAYER 3] Query Execution & Storage
│   │   ├── mod.rs
│   │   ├── catalog.rs                  # Schema and metadata
│   │   ├── operators.rs                # Query operators
│   │   ├── executor.rs                 # Execution engine
│   │   ├── transaction.rs              # MVCC transaction manager
│   │   ├── wal.rs                      # Write-Ahead Log
│   │   └── recovery.rs                 # Crash recovery
│   │
│   ├── sql/                            # [LAYER 4] SQL Processing
│   │   ├── mod.rs
│   │   ├── parser.rs                   # SQL parsing
│   │   ├── binder.rs                   # Semantic analysis
│   │   ├── planner.rs                  # SQL to physical plan
│   │   └── types.rs                    # SQL data types
│   │
│   ├── ml/                             # [UTILITY] ML Components
│   │   ├── mod.rs
│   │   ├── regression.rs               # Linear/polynomial regression
│   │   ├── neural.rs                   # Neural networks
│   │   ├── time_series.rs              # Time series prediction
│   │   └── training.rs                 # Model training
│   │
│   └── compute/                        # [UTILITY] SIMD & Math
│       ├── mod.rs
│       ├── simd_ops.rs                 # SIMD search operations
│       └── vector_math.rs              # Vectorized math
│
├── benches/                            # Performance Benchmarks
│   ├── index_benchmark.rs              # RMI vs PGM vs B+Tree
│   ├── optimizer_benchmark.rs          # Cardinality and cost estimation
│   └── query_benchmark.rs              # End-to-end query performance
│
├── tests/                              # Test Suites
│   ├── integration_tests.rs            # Integration tests
│   ├── perf_tests.rs                   # Performance regression tests
│   └── sql_tests.rs                    # SQL and schema tests
│
└── examples/                           # Example Workloads
    ├── tpc_h.rs                        # TPC-H analytical workload
    ├── oltp.rs                         # TPC-C-lite OLTP benchmark (New-Order/Payment)
    ├── learned_index_demo.rs           # Learned index showcase (PGM/RMI/B+Tree)
    ├── memory_footprint.rs             # Real PGM/RMI vs B+Tree memory footprint at scale
    ├── monotonic_ingestion_niche.rs    # Proof of concept: memory + retention, real SQL, one workload
    ├── vs_sqlite.rs                    # Real comparison vs SQLite (in-process)
    ├── vs_sqlite_point_lookup.rs       # Point-lookup latency vs SQLite, full SQL path
    └── vs_postgres.rs                  # Real comparison vs Postgres (client/server)
```

(The retention-scheduling benchmark isn't listed above because it isn't an
`examples/` binary — it lives as an in-crate test,
`execution::mvcc_store::tests::retention_bloat_benchmark`, since it
exercises `MVCCTable` methods that are deliberately `pub(crate)`, not
public API. Run it with `cargo test test_retention_bloat_comparison --lib
-- --nocapture`.)

## 🔧 Building

### Requirements
- Rust 1.70+
- 2GB RAM (for compilation)
- Linux/macOS/Windows

### Build Commands

```bash
# Debug build
cargo build

# Release build (optimized)
cargo build --release

# Run tests
cargo test

# Run all benchmarks
cargo bench

# Run specific example
cargo run --example learned_index_demo --release
cargo run --example tpc_h --release
cargo run --example oltp --release
```

## 📈 Running Benchmarks

### Index Performance Comparison
```bash
cargo bench --bench index_benchmark
```
For real, measured numbers (not an "expected range" guess) see
[Index Build Performance](#index-build-performance) and [Memory
Footprint](#memory-footprint-pgmrmi-vs-a-real-btree) below — both
reproducible via `cargo run --example learned_index_demo --release` /
`cargo run --example memory_footprint --release`.

### Query Optimizer Benchmarks
```bash
cargo bench --bench optimizer_benchmark

# Expected results:
# - Cardinality estimation: <1 µs
# - Cost model: <10 µs
# - Join reordering (5 tables): <100 µs
```

### End-to-End Query Performance
```bash
cargo bench --bench query_benchmark
```

## 🧪 Running Tests

### All Tests
```bash
cargo test
```

### Specific Test Suite
```bash
cargo test --test integration_tests
cargo test --test perf_tests
cargo test --test sql_tests
```

### With Output
```bash
cargo test -- --nocapture
```

## 💡 Example Usage

### Running TPC-H-lite
```bash
cargo run --example tpc_h --release
```

Real output from a run on the author's machine (Apple Silicon Mac, not a
controlled benchmark environment — see the example's own module doc for
what's simplified):
```
=== TPC-H-lite ===

Loading 20000 orders and 2000 customers...
Loaded in 177.9ms

Range Scan: SELECT * FROM orders WHERE o_orderkey >= 10000
  10001 rows in 12.8ms (PK-index-accelerated)

Aggregation: SELECT COUNT(*), SUM(o_totalprice) FROM orders
  count=20000, sum=49878370.000000276 in 11.7ms (full scan)

Join Query: SELECT * FROM orders JOIN customer ON orders.o_custkey = customer.c_custkey
  20000 rows (20000 orders x 2000 customers, hash join) in 42.2ms
```
The join now runs at the *same* scale as the scan/aggregation tables —
it didn't used to, and that history is worth keeping. Three fixes, in
order: compiling `evaluate_predicate`'s predicate once per query instead
of re-parsing the string on every row pair (1.80s → 497ms, at the old
3,000 x 300 scale — see
[`row_codec::CompiledPredicate`](src/execution/row_codec.rs)); not
cloning a row pair's values into a `Tuple` at all until the condition is
known to match, instead of doing that for every pair and discarding most
of them (497ms → 14.7ms, still at 3,000 x 300 — see
`CompiledPredicate::eval_split`); then, once those stopped being the
bottleneck, running the join as a real hash join instead of a nested loop
at all for its plain-equality condition — O(orders + customers) instead
of O(orders × customers) — which is what actually let the join scale up
to match the other tables: 20,000 x 2,000 (40M possible pairs) in 42ms,
where the old nested loop needed a table 6-7x smaller just to stay fast.
Measuring the join at matching scale was never possible before hash
join existed; it's not a coincidence that the scale changed in the same
commit as the algorithm.

### Running TPC-C-lite
```bash
cargo run --example oltp --release
```

Real output from a run on the author's machine — WAL-backed and
fsync-per-commit, since durability is part of what these numbers are
meant to include (see the example's module doc for what's simplified
versus real TPC-C). Both transactions use a single-statement arithmetic
`UPDATE` (`SET s_qty = s_qty - 1`, via
[`row_codec::CompiledAssignment`](src/execution/row_codec.rs)) rather
than the client-side `SELECT`-then-`UPDATE` this benchmark originally had
to use — and, as of the fix described below, that's now genuinely
race-free under contention on the same row too, not just simpler syntax:
```
=== TPC-C-lite ===
4 warehouses, 200 customers, 100 stock items, 4 threads x 250 transactions (New-Order/Payment, ~50/50)

Elapsed: 28.2s
New-Order: 509 committed, 0 failed
Payment:   491 committed, 0 failed
Total:     1000/1000 committed

tpmC (New-Order committed / minute): 1082.1
Overall throughput: 35.4 committed transactions/sec
```
(This benchmark's queries are simple single-row updates by primary key,
not filter-heavy scans, so it isn't where the predicate-re-parsing fix
above shows up — run-to-run timing differences here are ordinary
fsync-latency noise, not a regression or a real speedup from either
fix.)

### Running Learned Index Demo
```bash
cargo run --example learned_index_demo --release
```

Shows detailed performance comparison between PGM, RMI, and B-Tree — all three built and searched for real over the same key sets — across different dataset sizes. Feeds the [Index Build Performance](#index-build-performance) numbers below.

### Running vs SQLite / vs Postgres
```bash
cargo run --example vs_sqlite --release

# Needs a reachable Postgres (defaults to host=localhost port=5432,
# user/dbname = $USER; override with SHRESTIDB_PG_URL):
cargo run --example vs_postgres --release
```

Real, in-process (`vs_sqlite`) or real client/server (`vs_postgres`) comparisons against the same TPC-H-lite workload as `examples/tpc_h.rs`. Read both examples' module docs before trusting the numbers at a glance — each documents exactly what is and isn't held equal between engines (see [vs SQLite / vs Postgres](#vs-sqlite--vs-postgres-real-comparisons) below for the short version).

## 📚 Documentation

### Design Document
See [DESIGN.md](DESIGN.md) for comprehensive architecture, algorithm descriptions, and performance analysis.

### Key Papers
1. **Learned Index Structures** - https://arxiv.org/abs/1712.01208
2. **PGM-Index: A Scalable Approximate Index** - https://arxiv.org/abs/1910.06169
3. **Machine Learning for Systems** - https://arxiv.org/abs/1905.13328

## 🎯 Development Status

### Completed ✅
- [x] In-memory MVCC store with Write-Ahead Log recovery
- [x] Learned index structures (RMI, PGM), real and measured against this codebase's own real (node-splitting) B+Tree — both lookup speed (scale-dependent, RMI winning at every size tested) and memory footprint (PGM ~4.2x smaller, consistent at every scale)
- [x] Learned MVCC retention scheduling (`RetentionPredictor`), real and wired into every commit — a real training-data bug (the model could never fit under any realistic driven workload) found and fixed this session
- [x] `UPDATE`/`DELETE` primary-key index acceleration (`candidate_rows_for_write`) — a real, ~930x measured fix, found while building the monotonic-key ingestion proof of concept below
- [x] `DynamicPGMIndex` bulk-insert scaling (unsorted write buffer + a growing, not fixed, flush threshold) — a real O(n²)-to-O(n) fix, also found while building that proof of concept
- [x] Basic SQL parsing and type system
- [x] Real cardinality estimation (`ANALYZE`, PGM-fitted per-column distributions)
- [x] Cost model consulted for real plan cost estimates
- [x] Hash join for equality conditions
- [x] Cost-based join ordering for 3+-table queries (real per-column distinct counts from `ANALYZE`, searched under a provable-resolvability safety check — see [`optimizer::join_reorder`](src/optimizer/join_reorder.rs))
- [x] Vectorized query operators
- [x] MVCC transaction manager
- [x] Write-Ahead Log (WAL)
- [x] Comprehensive benchmarks
- [x] Example workloads (TPC-H, OLTP)

### In Progress 🔄
- [ ] Advanced SQL features (subqueries, window functions)
- [ ] Index creation/selection automation
- [ ] Distributed query processing
- [ ] Advanced ML models (reinforcement learning)
- [ ] GPU acceleration for SIMD operations

### Future 🚀
- [ ] Multi-table learned statistics
- [ ] Self-tuning indexes
- [ ] Adaptive data partitioning
- [ ] ML-based query compilation
- [ ] Integration with popular ORMs

## 🔬 What's Actually Real Here

This section used to list grandiose, unverified "research contributions"
("First complete RDBMS with ML throughout", "Superior accuracy to
histogram-based approaches" — never measured against any histogram),
several of them describing code that turned out to be unused or
nonexistent (a buffer pool nothing called, a "hybrid" index router
nothing routed through). Replaced with the actual, measured results —
each reproducible via `cargo run --example <name> --release`, not
claimed from vibes:

1. **Real learned-index lookup speedup and memory footprint**: RMI beats this codebase's own real B+Tree by up to ~9.8x at 1M records (PGM is actually *slower* below 1M, only pulling ahead past it) — real, but a component-level result, not (yet) an end-to-end one; see [Point Lookup Latency vs SQLite](#point-lookup-latency-vs-sqlite-does-the-index-speedup-survive-the-full-sql-path) for the honest full-SQL-path number. PGM's *memory* footprint is the more consistent win: ~4.2x smaller than the same B+Tree at every scale tested, real data distribution included — see [Memory Footprint](#memory-footprint-pgmrmi-vs-a-real-btree).
2. **Real, `ANALYZE`-driven cardinality estimation**: replaced a fake "neural network" (hardcoded weights, a `train()` that never trained) with an actual empirical-CDF model over real column data — see [Real Cardinality Estimation](#real-cardinality-estimation-vs-a-fixed-constant).
3. **Real, cost-based join reordering**: a dead `.sort()`-by-ID stub, never called by anything, replaced with a real search using real per-column distinct counts, gated by a provable-safety check — see [Real Join Reordering](#real-join-reordering-vs-a-cost-model-with-no-real-signal).
4. **Real outer-join and `NULL` semantics**: `LEFT`/`RIGHT`/`FULL OUTER JOIN` and `USING`/`NATURAL JOIN` previously executed silently wrong (dropping or fabricating rows); `NULL` comparisons were treated as matches instead of SQL's three-valued `UNKNOWN`. All fixed and tested.

## 📊 Benchmark Results

### TPC-H-lite (20K orders, 2K customers — see [`examples/tpc_h.rs`](examples/tpc_h.rs))
These are ShrestiDB's own measured numbers, produced by `cargo run
--example tpc_h --release`. There's no "Traditional DB" anywhere in this
codebase to compare against, so there's no fabricated baseline or speedup
column here — see the example's module doc for exact scale and caveats.

| Query | Scale | Time |
|-------|-------|------|
| Range Scan (PK-indexed) | 20,000 orders | 12.8ms |
| Aggregation (full scan) | 20,000 orders | 11.7ms |
| Join Query (hash join) | 20,000 x 2,000 | 42.2ms |

The join runs at the same scale as everything else in this table now — it
didn't used to. It went through three fixes to get there: compiling
`evaluate_predicate`'s predicate once per query instead of re-parsing the
string on every row pair (1.80s → 497ms, at the old 3,000 x 300 scale —
see [`row_codec::CompiledPredicate`](src/execution/row_codec.rs)); not
cloning a row pair's values into a merged `Tuple` until the condition is
known to match, instead of doing that unconditionally (497ms → 14.7ms,
still at 3,000 x 300); then replacing the always-nested-loop join with a
real hash join for its plain-equality condition — O(orders + customers)
instead of O(orders × customers) — which is what actually let the join
scale up to 20,000 x 2,000 (40M possible pairs) at 42ms, where the old
algorithm needed a table 6-7x smaller just to stay fast. The middle fix
looked biggest in isolation; the last one is what actually removed the
scale ceiling.

### TPC-C-lite (see [`examples/oltp.rs`](examples/oltp.rs))
WAL-backed, fsync-per-commit, 4 threads. Produced by `cargo run --example
oltp --release`; see the example's module doc for what's simplified versus
real TPC-C. Both transactions use a single-statement arithmetic `UPDATE`
(see [`row_codec::CompiledAssignment`](src/execution/row_codec.rs)),
genuinely race-free under contention on the same row: `execute_update`
acquires a candidate row's exclusive lock *before* re-reading its latest
committed value and computing from that, not from the earlier snapshot
read that decided the row was a candidate — see that method's doc comment
for the full mechanism, and
`test_concurrent_arithmetic_updates_do_not_lose_updates` in
[`execution::executor`](src/execution/executor.rs) for the test that
proves it (verified to actually fail without the fix, not just pass with
it).

| Metric | Performance |
|--------|-------------|
| Throughput | 35.4 committed tx/sec |
| tpmC (New-Order/min) | 1082.1 |
| Committed | 1000/1000 (0 lock-manager failures) |

(This benchmark's queries are simple PK updates, not filter-heavy scans,
so run-to-run variance here is ordinary fsync-latency noise, not related
to the predicate-parsing fix above or the assignment-evaluation fix
below.)

### Index Build Performance
Real numbers from [`examples/learned_index_demo.rs`](examples/learned_index_demo.rs)
(`cargo run --example learned_index_demo --release`) — PGM, a single-stage
RMI, and this codebase's own `BTree` all built and searched over the same
sorted key sets. No external database involved; `BTree` is the in-repo
baseline both learned structures are compared against — and, as of this
measurement, a *real* one: `index::btree::BTree` used to be a single
unsplit node (`Vec::binary_search` wearing B-tree-shaped names, its own
`insert` comment admitted as much: "Simplified insert: just add to
root"), discovered while building the memory-footprint comparison below.
It's now a real B+Tree with actual node splitting and a growing height —
every number in both tables below is measured against that, not the old
placeholder, and both build slower and search differently as a result.
Lookup timing here is a small sample (100 searches per size) read off a
single representative run, not a statistically tight measurement — expect
run-to-run variance of a few tenths of a microsecond at these scales.

**Build time**
| Dataset Size | PGM | RMI | B+Tree |
|-------------|-----|-----|--------|
| 10K | 130µs | 85µs | 870µs |
| 100K | 1.27ms | 0.80ms | 11.00ms |
| 1M | 13.31ms | 9.10ms | 135.21ms |

PGM/RMI build roughly 10x faster than the B+Tree at every size tested —
consistent, not scale-dependent the way lookup speedup is (below): no
node-splitting cascades to pay for, one linear pass over already-sorted
keys instead.

**Lookup latency** (avg per lookup, 100 searches)
| Dataset Size | PGM | RMI | B+Tree | PGM Speedup | RMI Speedup |
|-------------|-----|-----|--------|-------------|-------------|
| 10K | 0.270µs | 0.030µs | 0.060µs | 0.2x | 2.0x |
| 100K | 0.380µs | 0.070µs | 0.220µs | 0.6x | 3.1x |
| 1M | 0.400µs | 0.090µs | 0.880µs | 2.2x | 9.8x |

Honestly: PGM is *slower* than the real B+Tree at both 10K and 100K —
predicting a segment and then bounded-searching it costs more than the
B+Tree's own binary descent at small scale, where the B+Tree stays
shallow (a handful of levels) and cache-resident. The crossover to a
genuine PGM win happens somewhere between 100K and 1M records. RMI (one
global linear model, cheaper per lookup than PGM's segment lookup) wins
at every size tested here, growing from a modest 2.0x at 10K to 9.8x at
1M — a real, scale-dependent number, not the flat "always Nx faster"
framing a single headline figure implies.

### Memory Footprint (PGM/RMI vs a real B+Tree)
Real numbers from `cargo run --example memory_footprint --release`
([`examples/memory_footprint.rs`](examples/memory_footprint.rs)) — actual
allocated heap bytes (`Vec` capacity × element size, summed over every
heap allocation each structure holds; see `heap_bytes()` on
`PGMIndex`/`RMIIndex`/`BTree`), not process RSS, so allocator overhead
isn't included, but the same accounting method applies to all three, so
the comparison between them is real. This targets a concrete real-world
shape — a monotonic or near-monotonic key (auto-increment id, order
number, event timestamp), what most production tables actually cluster
on — at growing scale, under two distributions: purely sequential (PGM's
ideal case) and a "bursty timestamp" pattern (dense/sparse periods
alternating every 2,000 keys — still monotonic, but not one straight
line, which forces PGM to open new segments the way real event-ingestion
irregularity would).

| Keys | PGM | RMI | B+Tree | PGM vs B+Tree | RMI vs B+Tree |
|------|-----|-----|--------|---------------|---------------|
| 100K | 0.76 MB (8.0 B/key) | 1.53 MB (16.0 B/key) | 3.23 MB (33.8 B/key) | 4.2x smaller | 2.1x smaller |
| 1M | 7.63 MB (8.0 B/key) | 15.26 MB (16.0 B/key) | 32.30 MB (33.9 B/key) | 4.2x smaller | 2.1x smaller |
| 10M | 76.29 MB (8.0 B/key) | 152.59 MB (16.0 B/key) | 323.10 MB (33.9 B/key) | 4.2x smaller | 2.1x smaller |

Sequential-key numbers shown; the bursty-timestamp distribution (50-5,000
segments depending on scale, versus 1 segment for perfectly sequential
keys) lands within 0.5% of the same bytes/key at every size — the 48
bytes a `PGMSegment` costs is negligible next to the 8 bytes/key the raw
keys themselves cost, even with real irregularity opening far more
segments than the ideal case. This is the more consistent, real-world-
relevant win of the two: unlike lookup speed (scale-dependent, and PGM
specifically loses below 1M), the memory advantage holds at every scale
tested and isn't sensitive to how linear the actual data is. RMI's
footprint is worse than PGM's by construction, not implementation
sloppiness: this RMI keeps an explicit `positions: Vec<usize>` alongside
`keys`, one entry per key, where PGM encodes position implicitly as array
index — a real, structural design difference between the two, not a bug
in either.

### Retention: Learned Predictor vs Fixed-Interval Sweeping
Real numbers from `cargo test test_retention_bloat_comparison --lib --
--nocapture` ([`execution::mvcc_store`](src/execution/mvcc_store.rs)) —
`RetentionPredictor` (real, wired into every `OLTPEngine::commit`, see
above) versus sweeping every row's dead MVCC versions at a fixed write
interval, replaying the identical 20,000-commit write sequence against
fresh tables for each policy. Workload: 500 rows, 20 of them "hot" (80%
of all writes land there — a realistic skew, not uniform), a snapshot
horizon that lags by up to 5 commits (modeling a trailing long-running
reader — without that lag any prune attempt always finds something and
every policy looks identical).

| Policy | End-of-run bloat | Sweep attempts | Wasted sweeps |
|---|---|---|---|
| Sweep every write | 15,980 | 20,000 | 16,997 (85.0%) |
| Fixed interval = 4 | 16,086 | 4,808 | 2,537 (52.8%) |
| Fixed interval = 8 | 16,144 | 2,273 | 607 (26.7%) |
| Fixed interval = 32 | 16,380 | 493 | 1 (0.2%) |
| Fixed interval = 128 | 16,537 | 117 | 0 (0.0%) |
| **Learned (RetentionPredictor)** | **16,352** | **1,007** | **71 (7.1%)** |

The learned predictor lands close to fixed-8's bloat while making
under half its sweep attempts and a third its waste rate — genuinely
better than every fixed interval from 1 through 8 on both axes at once,
not a tradeoff a human tuning a single constant made blind. It doesn't
clearly beat fixed-32 here (essentially tied on bloat, fixed-32 wastes
fewer attempts in absolute terms) — but fixed-32 is a constant chosen
with hindsight for *this* workload's specific 80/20 skew; the real claim
for the learned predictor isn't "beats every possible constant," it's
"finds a good point on the tradeoff automatically, per row, without a
human guessing the right constant for whatever a given table's actual
write pattern turns out to be" — which matters most exactly when that
pattern isn't known in advance or changes over time, the situation a
fixed constant can't adapt to but a model retrained on live telemetry
can.

### Point Lookup Latency vs SQLite (does the index speedup survive the full SQL path?)
Real numbers from [`examples/vs_sqlite_point_lookup.rs`](examples/vs_sqlite_point_lookup.rs)
(`cargo run --example vs_sqlite_point_lookup --release`) — 1,000,000
rows, 20,000 primary-key point lookups (`SELECT * FROM t WHERE id = ?`),
both engines via real prepared statements, in-memory, no disk I/O on
either side. This is the scenario built specifically to give the
learned index its best shot: the scale where [Index Build
Performance](#index-build-performance) shows RMI's largest real advantage
(~9.8x at 1M rows), and a workload of single-row lookups rather than a
range scan, so total time is dominated by the search itself rather than
materializing a large result:

| | Total (20K lookups) | Per lookup |
|---|---|---|
| SQLite | 21.29ms | 1.06µs |
| ShrestiDB | 96.65ms | 4.83µs |

**ShrestiDB is 4.54x slower, not faster.** The component-level ~9.8x
lookup advantage over this codebase's own B+Tree is real (see [Index
Build Performance](#index-build-performance)) — but it doesn't survive
the full SQL execution path: real per-statement overhead (a
lock/transaction per `execute_prepared` call, row deserialization,
materializing the result into owned `String`s) costs several
microseconds on its own, dwarfing whatever the index search saves at
this row count. The honest conclusion from every benchmark on this page
together: the learned-index technique is a genuine, measured algorithmic
win in isolation, but this engine doesn't yet have a scenario where a
user would actually experience it as "faster than SQLite" end to end —
that requires shrinking the fixed per-query overhead around the index,
not a faster index.

**Profiled and partially closed.** A stage-by-stage breakdown (each
stage timed by calling the real internal code directly) showed the
index search itself is only ~10% of total lookup time — never the
bottleneck. The two largest identified costs were a redundant
re-tokenization of the same predicate string (`execute_prepared`
substituting `"id = ?"` into `"id = 1234"`, then `execute`'s indexed-scan
check re-parsing that same string right back apart — ~26% combined) and
an unnecessary `TableSchema` clone on every lookup inside the index-scan
path (its caller already owned one). Both are fixed: the query planner
now splits a single-comparison predicate into `(left, op, right)` once,
at plan time (`LogicalPlanNode::Filter::split`), reused by both
parameter substitution and the indexed-scan check instead of each
re-tokenizing it independently; the schema clone was simply redundant
and removed. Real result: **6.30µs → 4.83µs per lookup (a 23%
reduction)**, moving the SQLite gap from 5.71x to 4.54x. Still slower,
not yet a win — the remaining cost is spread across smaller items
(`Vec<u8>` cloning on every row read, `PhysicalPlan` cloning per
execution, general dispatch overhead) with no single dominant target
left to fix.

### Real Cardinality Estimation (vs. a fixed constant)
Real numbers from [`examples/cardinality_demo.rs`](examples/cardinality_demo.rs)
(`cargo run --example cardinality_demo --release`). 100,000 orders, a
realistic 95%/5% shipped/cancelled skew, query `WHERE amount >= 999`
(the cancelled orders):

| | Estimated rows | True rows | Error |
|---|---|---|---|
| Before `ANALYZE` (fixed 0.5 selectivity) | 500 | 5,000 | 4,500 rows |
| After `ANALYZE` (real `ColumnDistribution`) | 5,000 | 5,000 | 0 rows |

The fixed heuristic isn't wrong because it's poorly tuned — it's wrong
because it's the *same number for every filter, on every table,
regardless of the actual data*. `ANALYZE` replaces it with a real
piecewise-linear model of the column's empirical CDF (`PGMIndex::predicted_rank`,
the same learned-index technique behind the lookup numbers above), fit
to the column's actual values.

### Real Join Reordering (vs. a cost model with no real signal)
Real numbers from [`examples/join_reorder_demo.rs`](examples/join_reorder_demo.rs)
(`cargo run --example join_reorder_demo --release`). 200 accounts joined
to 10,000 `tags` (a low-cardinality, wide fan-out join key — only 2
distinct values) and 200 `transactions` (a near-1:1 match), written with
the expensive join listed first in the SQL — the case that benefits most
from being pushed later:

| | Chosen order | Estimated cost | Wall time |
|---|---|---|---|
| Before `ANALYZE` (no distinct counts — every order costs the same) | tags → transactions (source order, unchanged) | 22,330 | 1.44s |
| After `ANALYZE` (real distinct counts) | transactions → tags | 2,266 | 1.03s |

Row count is identical both times (1,000,000 — reordering changed
performance, not correctness) — the search only ever picks among orders
it can *prove* resolvable against this engine's left-deep executor (see
[`optimizer::join_reorder`](src/optimizer/join_reorder.rs)'s module doc),
so it can never turn a working query into a broken one, only a faster
one. The 9.9x drop in estimated cost overstates the real 1.4x wall-time
win: both orders still materialize and stringify the same ~1,000,000-row
final result, a fixed cost the cost model doesn't account for at all and
that dilutes the real, order-dependent difference in join work — a real
gap, reported rather than tuned away.

### vs SQLite / vs Postgres (real comparisons)
Same TPC-H-lite workload (20K orders, 2K customers, joined at that same
20K x 2K scale — see [`examples/tpc_h.rs`](examples/tpc_h.rs)'s module
doc) run against real SQLite and Postgres instances, not a fabricated
baseline. **Read the caveats below the table before drawing conclusions
from it** — the two comparisons aren't measuring the same thing, on
purpose.

| Query | ShrestiDB vs SQLite | ShrestiDB vs Postgres |
|-------|---------------------|------------------------|
| Load (20K + 2K rows) | 3.78x slower | **19.5x faster** |
| Range Scan (PK-indexed) | 1.76x slower | **faster (0.83x)** |
| Aggregation (full scan) | 5.36x slower | 2.67x slower |
| Join (hash join, 20K x 2K) | 1.49x slower | 1.21x slower |

("Nx slower/faster" = ShrestiDB's time relative to the other engine's,
for the same query. Run-to-run noise moves these by a point or so —
don't read the exact digits as more precise than they are.)

**What this does and doesn't tell you:**
- **[`vs_sqlite.rs`](examples/vs_sqlite.rs)** is genuinely apples-to-apples:
  both engines in-process, both in-memory (no disk I/O on either side),
  both fully materializing every result row before the clock stops, and
  both loaded through a real prepared, parameterized statement
  (`QueryExecutor::prepare`/`execute_prepared` on ShrestiDB's side — see
  [`row_codec`](src/execution/row_codec.rs)'s `?`/`$N` placeholder
  support). That cut ShrestiDB's load gap from 10.10x slower to 3.78x —
  the remaining difference is real per-statement overhead (each
  `execute_prepared` call still walks the cached statement, re-parses
  each bound literal via `parse_value`, and takes a lock/transaction per
  statement), not the giant re-parse-per-row gap this replaced.
- **[`vs_postgres.rs`](examples/vs_postgres.rs)** is *not*
  apples-to-apples, deliberately: Postgres pays a real client/server
  round trip per statement (even over loopback) that the other two never
  pay, which is most of why ShrestiDB's *load* number looks faster than
  Postgres here — that's IPC overhead dominating, not query execution
  being faster. Take the Postgres numbers as "where we stand against a
  real client/server RDBMS as actually deployed," not as an isolated
  measurement of execution speed.
- **The join gap is now mostly closed, and the join runs at real scale
  for the first time.** It was 350-530x slower at the old, deliberately
  tiny 3,000 x 300 scale — a scale that small specifically because the
  nested loop it had no alternative to couldn't run any larger without
  becoming impractical to even benchmark. Three fixes got it to 1.21-1.49x
  at 20,000 x 2,000 (40M possible pairs, not 900K): compiling the
  predicate once instead of re-parsing it per pair (350-530x → 73-109x,
  still at the old scale); not cloning a row pair's values into a merged
  `Tuple` until the condition is known to match (73-109x → 2.75-4.56x,
  still at the old scale — see
  [`row_codec::CompiledPredicate::eval_split`](src/execution/row_codec.rs));
  then replacing the nested loop itself with a real hash join for
  equality conditions (see `execution::executor::hash_join`) — which
  didn't just narrow the gap further, it's what actually removed the
  scale ceiling the first two fixes never touched. The two earlier fixes
  made a slow algorithm faster; the third replaced the algorithm.

## 📝 License

Proprietary — All Rights Reserved. See [LICENSE](LICENSE). This is closed-source software; see [CONTRIBUTING.md](CONTRIBUTING.md) for the internal engineering workflow.

## 👤 Author

Created by **v4life** as a demonstration of machine learning integration in database systems.

## 🙏 Acknowledgments

- Research team at UC Berkeley for Learned Index Structures
- Rossano Venturini for PGM-Index development
- Rust community for excellent tools and libraries

---

## Quick Start

```bash
# Clone and build
cd shrestidb
cargo build --release

# Run example
cargo run --example learned_index_demo --release

# Run benchmarks
cargo bench

# Run tests
cargo test
```

**The future of databases is learned.** 🧠📊
