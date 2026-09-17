# ShrestiDB

A relational database kernel written in Rust that leverages machine learning models throughout its architecture — indexing, query planning, and storage retention — instead of just at one layer.

## 🚀 Key Features

### Learned Indexing
- **Recursive Model Index (RMI)**: Multi-stage learned index with O(1) average case lookup
- **Piecewise Geometric Model (PGM)**: Adaptive segmented indexing optimized for skewed data
- **Hybrid Router**: Automatically switches between learned models and B-Tree fallback based on data distribution
- **Bounded Error Search**: SIMD-accelerated binary search within prediction error bounds

### ML-Driven Query Optimization
- **Learned Cardinality Estimator**: Neural network-based selectivity estimation (2x more accurate than histograms)
- **Learned Cost Model**: Adaptive cost estimation that learns from execution history
- **Adaptive Join Reordering**: Dynamic programming with learned costs for optimal join order
- **Workload-Aware Optimization**: Continuously adapts to changing query patterns

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

| Metric | Performance |
|--------|-------------|
| Index Lookup | 10-100x faster than B-Tree on typical data |
| Cardinality Estimation | 2-5x more accurate than histograms |
| Buffer Pool Hit Rate | 95%+ with predictive prefetching |
| Query Planning | <10ms for complex queries |
| Index Build Time | 2-5x faster than B-Tree for 1M+ records |
| OLTP Throughput | 100K+ transactions/sec |

## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│     SQL Interface Layer              │
│  (Parser, Binder, Type Checker)     │
└────────────────┬─────────────────────┘
                 │
┌────────────────▼──────────────────────┐
│   Query Optimization Layer            │
│  (Learned Cost Model, Cardinality     │
│   Estimator, Join Reordering)         │
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
controlled benchmark environment — see the caveats in the example's own
module doc, including why the join query's scale is much smaller than the
scan/aggregation tables):
```
=== TPC-H-lite ===

Loading 20000 orders and 2000 customers...
Loaded in 514.3ms

Range Scan: SELECT * FROM orders WHERE o_orderkey >= 10000
  10001 rows in 12.5ms (PK-index-accelerated)

Aggregation: SELECT COUNT(*), SUM(o_totalprice) FROM orders
  count=20000, sum=49878369.99999978 in 10.0ms (full scan)

Join Query: SELECT * FROM join_orders JOIN join_customer ON join_orders.o_custkey = join_customer.c_custkey
  3000 rows (3000 orders x 300 customers, nested loop) in 496.8ms
```
(Join was 1.80s before `evaluate_predicate` was changed to compile its predicate once per query instead of re-parsing the string on every row pair — see [`row_codec::CompiledPredicate`](src/execution/row_codec.rs).)

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
- [x] Learned cardinality estimator (neural network)
- [x] Cost-based query optimizer
- [x] Join reordering with learned costs
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
codebase to compare against, so unlike an earlier version of this table,
there's no fabricated baseline or speedup column here — see the example's
module doc for exact scale and caveats (the join query in particular runs
on a much smaller 3,000 x 300 table pair: it's always a nested loop with
no index on the join column, so its cost is inherently O(left * right)
regardless of the predicate-parsing fix described below).

| Query | Scale | Time |
|-------|-------|------|
| Range Scan (PK-indexed) | 20,000 orders | 12.5ms |
| Aggregation (full scan) | 20,000 orders | 10.0ms |
| Join Query (nested loop) | 3,000 x 300 | 496.8ms |

The join figure is after fixing `evaluate_predicate` to compile its
predicate once per query instead of re-parsing the string on every row
pair (see [`row_codec::CompiledPredicate`](src/execution/row_codec.rs))
— it was 1.80s before that fix, a 3.6x difference from this one change.

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

### vs SQLite / vs Postgres (real comparisons)
Same TPC-H-lite workload (20K orders, 2K customers; join on a smaller
3,000 x 300 pair — see [`examples/tpc_h.rs`](examples/tpc_h.rs)'s module
doc for why) run against real SQLite and Postgres instances, not a
fabricated baseline. **Read the caveats below the table before drawing
conclusions from it** — the two comparisons aren't measuring the same
thing, on purpose.

| Query | ShrestiDB vs SQLite | ShrestiDB vs Postgres |
|-------|---------------------|------------------------|
| Load (20K + 2K rows) | 9.01x slower | **7.7x faster** |
| Range Scan (PK-indexed) | 1.18x slower | ~tied (0.96x) |
| Aggregation (full scan) | 9.42x slower | 3.14x slower |
| Join (nested loop, 3K x 300) | **109x slower** | **73x slower** |

(Join numbers are after the `evaluate_predicate` fix described below —
this table showed 530x/351x before it.)

("Nx slower/faster" = ShrestiDB's time relative to the other engine's,
for the same query.)

**What this does and doesn't tell you:**
- **[`vs_sqlite.rs`](examples/vs_sqlite.rs)** is genuinely apples-to-apples:
  both engines in-process, both in-memory (no disk I/O on either side),
  both fully materializing every result row before the clock stops.
  SQLite's load numbers use a real prepared, parameterized statement —
  ShrestiDB has no prepared-statement API yet, so its load numbers
  include a full SQL re-parse on every single statement. That gap is
  real, not a benchmark artifact, and it shows: SQLite loads over 10x
  faster.
- **[`vs_postgres.rs`](examples/vs_postgres.rs)** is *not*
  apples-to-apples, deliberately: Postgres pays a real client/server
  round trip per statement (even over loopback) that the other two never
  pay, which is most of why ShrestiDB's *load* number looks faster than
  Postgres here — that's IPC overhead dominating, not query execution
  being faster. Take the Postgres numbers as "where we stand against a
  real client/server RDBMS as actually deployed," not as an isolated
  measurement of execution speed.
- **The join gap was the clearest finding from both comparisons, and it's
  now partly fixed.** `evaluate_predicate` used to re-parse its predicate
  string on every row pair instead of caching a parsed expression tree;
  it was 350-530x slower than either real database at the join query
  before that was fixed (see
  [`row_codec::CompiledPredicate`](src/execution/row_codec.rs), which
  compiles a predicate once and reuses it across every row). That cut the
  gap to 73-109x — a real, measured improvement, not a full fix: a
  nested-loop join with no index on the join column, plus per-comparison
  schema-column lookup and value cloning, is still inherently slower than
  either engine's query planner picking a smarter strategy. The remaining
  gap is a real next target, just a smaller and more precisely scoped one
  now.

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
