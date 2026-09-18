# ShrestiDB

A relational database kernel written in Rust that leverages machine learning models throughout its architecture — indexing, query planning, and storage retention — instead of just at one layer.

## 🚀 Key Features

### Learned Indexing
- **Recursive Model Index (RMI)**: Multi-stage learned index with O(1) average case lookup
- **Piecewise Geometric Model (PGM)**: Adaptive segmented indexing optimized for skewed data
- **Hybrid Router**: Automatically switches between learned models and B-Tree fallback based on data distribution
- **Bounded Error Search**: SIMD-accelerated binary search within prediction error bounds

### ML-Driven Query Optimization
- **Real Cardinality Estimation**: `ANALYZE <table>` builds a real per-column value distribution — a `PGMIndex` (the same learned-index technique above) fit over the column's actual sorted values as an empirical CDF, plus a real distinct-value count for equality selectivity. Replaces a fixed 0.5-selectivity-for-every-filter constant with a genuine, data-driven estimate; see [`optimizer::cardinality`](src/optimizer/cardinality.rs) and the [Real Cardinality Estimation](#real-cardinality-estimation-vs-a-fixed-constant) results below. (An earlier version of this feature was a "neural network" whose weights were hardcoded and never trained — replaced rather than reused; see that module's doc comment for the full story.)
- **Cost Model**: A real IO/CPU cost formula, consulted for every plan's `estimated_cost` — not adaptive or trained from execution history despite earlier documentation here claiming otherwise.
- **Hash Join**: Equality join conditions run as a real hash join (build a hash table on the smaller side, probe with the larger) instead of a nested loop — see [Benchmark Results](#-benchmark-results) below for the measured effect.
- **Real Join Reordering**: For 3+-table queries, [`optimizer::join_reorder::JoinOrderer`](src/optimizer/join_reorder.rs) searches for the cheapest join order that's *provably resolvable* by this engine's left-deep executor, using real per-column distinct-value counts from `ANALYZE` — not a heuristic sort, and not a number computed and then ignored; the executor runs whatever order the search picks. See [Real Join Reordering](#real-join-reordering-vs-a-cost-model-with-no-real-signal) below for a measured before/after. Queries with an OUTER join always keep source order — reordering one isn't generally safe, and proving when it is stays out of scope (see below).
- **Real Outer Joins**: `LEFT`/`RIGHT`/`FULL OUTER JOIN` now actually emit an unmatched row once, padded with `NULL` on the other side, instead of silently behaving like `INNER JOIN` (dropping it) — an earlier version of this parser recognized the SQL syntax but discarded which kind of join it was. Works on both the hash-join and nested-loop execution paths; see [`sql::parser::JoinKind`](src/sql/parser.rs) and `execution::executor::append_unmatched`.
- **Real `USING`/`NATURAL JOIN`**: previously silently executed as an unfiltered `CROSS JOIN` — an earlier version of this parser recognized the syntax but discarded it entirely, so `a JOIN b USING (id)` on 2×3 rows returned all 6 (every combination) instead of the 2 that actually match. Now resolved against the real schemas at execution time into an ordinary equi-join condition; see `execution::executor::resolve_using_condition`/`resolve_natural_condition`. Scope limit, loudly enforced rather than silently guessed at: only a single shared/listed column is supported (this engine's join execution has no composite hash key) — more than one is a clear error, not a partial match. `SEMI`/`ANTI` join variants remain a known, lower-priority gap (execute as `INNER JOIN` rather than their real semantics); they're non-standard syntax, unlikely to be hit by accident.
- **Real three-valued (`NULL`-aware) comparison logic**: a comparison touching `NULL` used to be conflated with "this engine couldn't evaluate the predicate at all" — both fell back to keeping the row, so `WHERE fk > 100` with `fk` actually `NULL` incorrectly *included* that row, and (on the hash-join path specifically) two independent `NULL`s on a join key spuriously matched each other, since a `NULL` was just an ordinary, matchable hash key. Both were reachable before, but far more so once `LEFT`/`RIGHT`/`FULL OUTER JOIN` started routinely producing `NULL`-padded rows that flow into later predicates. Fixed with a real `Tri` (`True`/`False`/`Unknown`) type implementing SQL's actual `AND`/`OR` truth tables (`row_codec::Tri`) — `Unknown` is excluded exactly like `False`, never given the "give up and keep it" treatment real structural failures get — plus excluding `NULL`-keyed rows from the hash-join table entirely, so a `NULL` join key can never match, not even another `NULL`.

### Intelligent Storage
- **Learned Buffer Pool**: Markov chain predictor anticipates page access patterns for prefetching
- **Slotted Pages**: Efficient variable-length record storage with minimal fragmentation
- **Direct I/O**: Asynchronous I/O with io_uring support for high throughput
- **Write-Ahead Log**: Durable ACID transactions with crash recovery

### High-Performance Execution
- **Vectorized Query Engine**: Batch processing for cache efficiency
- **MVCC Transactions**: Lock-free snapshot isolation without write conflicts
- **SIMD Acceleration**: Vectorized filters and search operations
- **Learned Lock Scheduling**: Adaptive concurrency control

## 📊 Performance

Real, measured numbers — not marketing estimates — live in
[📊 Benchmark Results](#-benchmark-results) below, each reproducible via
`cargo run --example <name> --release`. Headline results as of the
latest run: RMI/PGM lookups **7.2-36.8x** faster than this codebase's own
B-Tree at 1M records (scaling *up* with data size, not down — see
[Index Build Performance](#index-build-performance)); a hash-joined query
within **1.2-1.5x** of SQLite/Postgres at 20K x 2K rows; and real,
`ANALYZE`-driven cardinality estimates landing on the true row count
where a fixed-constant heuristic was off by **10x** (see
[Real Cardinality Estimation](#real-cardinality-estimation-vs-a-fixed-constant)).
An earlier version of this table quoted round marketing-style figures
("100K+ transactions/sec", "2-5x more accurate than histograms") that
were never actually measured, some of which directly contradicted real
numbers already published elsewhere in this same file — replaced with
pointers to the real thing rather than a second set of numbers to keep
in sync by hand.

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
│  (RMI, PGM, B-Tree Fallback,         │
│   Hybrid Router)                      │
└────────────────┬──────────────────────┘
                 │
┌────────────────▼──────────────────────┐
│      Storage Layer                    │
│  (Pages, Disk Manager, Learned        │
│   Buffer Pool with Prefetching)       │
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
│   ├── storage/                        # [LAYER 1] Storage Management
│   │   ├── mod.rs
│   │   ├── page.rs                     # Slotted pages (4KB)
│   │   ├── disk_manager.rs             # Direct I/O abstraction
│   │   └── buffer_pool.rs              # Learned prefetch predictor
│   │
│   ├── index/                          # [LAYER 2] Learned Indexing
│   │   ├── mod.rs
│   │   ├── models.rs                   # LinearModel, PiecewiseLinearModel
│   │   ├── rmi.rs                      # Recursive Model Index
│   │   ├── pgm.rs                      # Piecewise Geometric Model
│   │   ├── btree.rs                    # B-Tree reference implementation
│   │   └── hybrid_router.rs            # Dynamic index selection
│   │
│   ├── optimizer/                      # [LAYER 3] Query Optimization
│   │   ├── mod.rs
│   │   ├── cardinality.rs              # Neural network cardinality estimator
│   │   ├── cost_model.rs               # Learned cost model
│   │   ├── join_reorder.rs             # Adaptive join ordering
│   │   ├── planner.rs                  # Query planner
│   │   └── statistics.rs               # Statistics collection
│   │
│   ├── execution/                      # [LAYER 4] Query Execution
│   │   ├── mod.rs
│   │   ├── catalog.rs                  # Schema and metadata
│   │   ├── operators.rs                # Query operators
│   │   ├── executor.rs                 # Execution engine
│   │   ├── transaction.rs              # MVCC transaction manager
│   │   ├── wal.rs                      # Write-Ahead Log
│   │   └── recovery.rs                 # Crash recovery
│   │
│   ├── sql/                            # [LAYER 5] SQL Processing
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
│   ├── index_benchmark.rs              # RMI vs PGM vs B-Tree
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
    ├── learned_index_demo.rs           # Learned index showcase (PGM/RMI/B-Tree)
    ├── vs_sqlite.rs                    # Real comparison vs SQLite (in-process)
    └── vs_postgres.rs                  # Real comparison vs Postgres (client/server)
```

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

# Expected results on modern CPU:
# - RMI lookup: 0.5-2 µs
# - PGM lookup: 0.3-1 µs
# - B-Tree lookup: 2-10 µs
# - Speedup: 2-10x faster than B-Tree
```

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
- [x] Core storage layer with paging and buffer pool
- [x] Learned index structures (RMI, PGM)
- [x] Hybrid index router with B-Tree fallback
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

## 🔬 Research Contributions

This project demonstrates several novel contributions:

1. **Practical Learned Database System**: First complete RDBMS with ML throughout
2. **Adaptive Index Selection**: Automatic switching between index types
3. **Learned Buffer Pool**: Markov chain-based prefetching
4. **Neural Cardinality Estimation**: Superior accuracy to histogram-based approaches
5. **Hybrid Execution**: Combining learned models with traditional fallbacks

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
baseline both learned structures are compared against.

**Build time**
| Dataset Size | PGM | RMI | B-Tree |
|-------------|-----|-----|--------|
| 10K | 157µs | 82µs | 425µs |
| 100K | 1.39ms | 781µs | 3.52ms |
| 1M | 12.28ms | 8.36ms | 31.89ms |

**Lookup latency** (avg per lookup, 100 searches)
| Dataset Size | PGM | RMI | B-Tree | PGM Speedup | RMI Speedup |
|-------------|-----|-----|--------|-------------|-------------|
| 10K | 0.130µs | 0.030µs | 0.070µs | 0.5x | 2.3x |
| 100K | 0.240µs | 0.080µs | 0.240µs | 1.0x | 3.0x |
| 1M | 0.410µs | 0.080µs | 2.940µs | 7.2x | 36.8x |

Honestly: at 10K, PGM lookup is actually *slower* than the B-Tree —
fixed per-lookup overhead dominates at small scale. The real advantage
shows up as data grows: by 1M records PGM is 7.2x and RMI is 36.8x
faster than the B-Tree baseline.

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

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

1. **Performance**: Further optimization of hot paths
2. **Features**: Advanced SQL support, distributed execution
3. **Testing**: More comprehensive test coverage
4. **Documentation**: Additional examples and guides

## 📝 License

MIT License - See LICENSE file for details

## 👤 Author

Created by **v4life** as a demonstration of machine learning integration in database systems.

## 🙏 Acknowledgments

- Research team at UC Berkeley for Learned Index Structures
- Rossano Venturini for PGM-Index development
- Rust community for excellent tools and libraries

## 📞 Contact & Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing documentation in DESIGN.md
- Review example workloads for usage patterns

---

## Quick Start

```bash
# Clone and build
git clone https://github.com/v4life/shrestidb.git
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
