# ShrestiDB ⚡

[![Build & Test](https://img.shields.io/badge/tests-346%20passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-Proprietary-blue.svg)]()
[![Rust](https://img.shields.io/badge/rust-2021%20edition-orange.svg)]()

**ShrestiDB** is an ACID-compliant, relational database kernel written in Rust that integrates machine learning models directly into indexing, query optimization, and storage management:

- **Learned Indexing (PGM & RMI)**: Up to **~8.2x – 9.8x faster** key lookups than a node-splitting B+Tree at scale, while using **~4x less memory**.
- **Real-Time Web Studio**: Built-in browser GUI for interactive query execution, catalog exploration, and live index benchmarking.
- **Interactive CLI & REPL**: Terminal shell with arrow-key history, persistent command storage (`~/.shrestidb_history`), and formatted ASCII data grids.
- **MVCC & Write-Ahead Logging (WAL)**: Snapshot isolation with row-level 2PL lock management, deadlock timeouts, and file-backed crash recovery.
- **Learned MVCC Retention GC**: Online linear models predict per-row version bloat to adaptively sweep dead tuple versions without manual `VACUUM`.
- **Data-Driven Query Optimizer**: Empirical CDF cardinality estimation via PGM models and dynamic cost-based join reordering.

---

## 🚀 Quick Start

### 1. Interactive CLI Shell

Launch the interactive REPL with full arrow-key history and tabular formatting:

```bash
cargo run --bin shrestidb
```

Once in the shell:
```sql
shrestidb> CREATE TABLE accounts (id INT PRIMARY KEY, name TEXT, balance FLOAT);
OK (took 19.52ms)

shrestidb> INSERT INTO accounts VALUES (1, 'Veer', 5000.0);
Query OK, 1 row affected (took 22.48ms)

shrestidb> INSERT INTO accounts VALUES (2, 'Alice', 3200.5);
Query OK, 1 row affected (took 25.25ms)

shrestidb> SELECT * FROM accounts WHERE balance > 3000;
+----+-------+---------+
| id | name  | balance |
+----+-------+---------+
| 1  | Veer  | 5000    |
| 2  | Alice | 3200.5  |
+----+-------+---------+
(2 rows, took 247.33µs)

shrestidb> \d
+----------+------------+---------+
| Table ID | Table Name | Columns |
+----------+------------+---------+
| 1        | accounts   | 3       |
+----------+------------+---------+
(1 table)

shrestidb> \q
Bye!
```

#### REPL Shortcuts & Meta-Commands:
| Command | Description |
|---|---|
| `↑` / `↓` | Cycle backward / forward through persistent command history |
| `Ctrl + R` | Interactive reverse search through past queries |
| `\d` / `SHOW TABLES` | List all tables registered in the catalog |
| `\d <table>` / `DESCRIBE <table>` | Show column schema, data types, nullability, and primary keys |
| `\status` | Display database version, WAL file path, and active table count |
| `\q` / `exit` / `quit` | Exit the REPL |

---

### 2. Browser Web Studio (GUI)

ShrestiDB includes an embedded Web Studio with a SQL editor, table explorer, and visual benchmark runner:

```bash
cargo run --bin shrestidb -- --studio
```
*(Or launch directly on a custom port: `cargo run --bin shrestidb-studio -- --port 8080`)*

Then open your browser to **[http://localhost:8080/](http://localhost:8080/)**:

- **SQL Editor**: Write SQL with quick query templates and press **`⌘ + Enter`** to run.
- **Live Catalog**: Browse tables and view row counts in real-time.
- **Learned Index Benchmark Tab**: Benchmark our Piecewise Geometric Model (PGM) against a standard B-Tree on 10K, 50K, or 100K keys with animated performance graphs.

---

### 3. Non-Interactive / Scripting Execution (`-e`)

Run one-shot queries directly from terminal scripts, Makefiles, or CI pipelines:

```bash
cargo run --bin shrestidb -- -q -w data.wal -e "SELECT name, balance FROM accounts WHERE id = 1;"
```

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────┐
│               Interface Layer                          │
│   • CLI REPL (Rustyline, history, formatted tables)    │
│   • Web Studio (Tokio HTTP, embedded SPA, REST API)   │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│               SQL & Planning Layer                     │
│   • Parser (sqlparser) & Semantic Binder               │
│   • Query Planner (Physical & Logical AST)             │
│   • Cost Model & Cost-Based Join Reordering            │
│   • Data-Driven Cardinality Estimator (PGM empirical) │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│              Query Execution Engine                    │
│   • Vectorized SIMD Operators (filter, project, limit) │
│   • Hash Join & Nested Loop Joins                      │
│   • Aggregations (COUNT, SUM, AVG, MIN, MAX)           │
│   • Prepared Statements ($N / ? parameters)            │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│                  Index Layer                           │
│   • Piecewise Geometric Model (DynamicPGMIndex)        │
│   • Recursive Model Index (RMIIndex)                   │
│   • Secondary B+Tree Indexes (CREATE INDEX)            │
│   • SIMD-accelerated Bounded Binary Search             │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│              Storage & Durability Layer                │
│   • In-Memory MVCC Store (Row Version Chains)          │
│   • Row-Level 2PL Lock Manager (Timeout Deadlock Det.) │
│   • Write-Ahead Log (WAL) with Replay-Based Recovery   │
│   • Learned Retention Predictor (Adaptive Version GC)  │
└────────────────────────────────────────────────────────┘
```

---

## ⚡ Core Innovations & Features

### 1. Learned Index Structures (PGM & RMI)
Traditional B-Trees incur $O(\log N)$ pointer-chasing latency across tree levels. ShrestiDB models data positions as a continuous function $f(key) \to position$:
- **Piecewise Geometric Model (PGM)**: Splits sorted data into piecewise linear segments with bounded error $\epsilon$. Point searches locate the target segment and run SIMD-accelerated bounded search in $O(\log \epsilon)$.
- **Dynamic PGM**: Combines a static PGM base with an append-only write buffer, amortizing bulk-load costs through geometric capacity expansion.
- **Memory Footprint**: Segments store only model slope and intercept, reducing index memory by **~4x** compared to node-splitting B+Trees.

### 2. Learned MVCC Retention GC
MVCC creates version chains that bloat memory if dead versions are not pruned. Instead of relying on periodic coarse-grained manual `VACUUM` commands:
- ShrestiDB fits an online regression model tracking write counts versus dead tuples collected.
- It adaptively triggers per-row version chain pruning only when predicted reclaimable memory justifies the sweep cost.

### 3. PGM-Driven Cardinality Estimation
Replaces arbitrary static 50% selectivity constants with an empirical Cumulative Distribution Function (CDF) built by fitting a PGM model over sorted column values during `ANALYZE <table>`:
- Yields true data-distribution-aware selectivity estimates for range and equality predicates.
- Directly powers the dynamic programming join reordering engine (`JoinOrderer`).

### 4. Durability & Transaction Isolation
- **Snapshot Isolation**: Transactions read consistent MVCC snapshots at `BEGIN` timestamp without acquiring read locks on clean reads.
- **Row-Level 2PL**: Exclusive locks on write operations with a 100ms cycle-detection timeout to prevent deadlocks.
- **Write-Ahead Logging**: Every `CREATE TABLE`, `INSERT`, `UPDATE`, and `DELETE` is logged to an append-only WAL before committing, guaranteeing durability across restarts.

---

## 📊 Measured Benchmark Results

All benchmarks are reproducible via `cargo run --release --example <name>`:

### 1. Index Lookup Performance (PGM / RMI vs B+Tree)
*Reproducible with `cargo run --release --example learned_index_demo` and `examples/memory_footprint.rs`*:

| Dataset Size | PGM Lookup | RMI Lookup | Real B+Tree | PGM Speedup | Index Memory |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **10,000** | 0.55 µs | 0.18 µs | 0.10 µs | 0.5x | ~4x smaller |
| **100,000** | 0.41 µs | 0.22 µs | 0.30 µs | 0.7x | ~4.1x smaller |
| **1,000,000** | **0.44 µs** | **0.37 µs** | **3.61 µs** | **8.2x – 9.8x** | **~4.2x smaller** |

> **Key takeaway**: While B-Trees perform well on tiny datasets fitting entirely in L1/L2 cache, learned models scale flatly at $O(\log \epsilon)$, overtaking B-Trees by **8x – 10x** at 1M+ keys while consuming **75% less RAM**.

---

### 2. High-Volume Monotonic Ingestion
*Reproducible with `cargo run --release --example monotonic_ingestion_niche`*:

| Metric | Measured Result |
|---|---|
| **Bulk Ingestion Throughput** | **116,000 rows/sec** (500,000 events) |
| **Live PGM PK Index Memory** | **4.11 MB** vs B+Tree **16.15 MB** (**3.9x smaller**) |
| **Retry Storm Update Throughput** | 52,648 updates in < 1 second |
| **Dead MVCC Residual Bloat** | Bounded at **1.43 versions/row** automatically |

---

### 3. Join Reordering & Hash Joins
*Reproducible with `cargo run --release --example vs_postgres` and `examples/vs_sqlite.rs`*:

- **Hash Join**: Evaluates equality join conditions in a single pass over hash partitions instead of nested loops.
- **Equi-join on 20K × 2K rows**: Within **1.2x – 1.5x** of SQLite and Postgres latency.

---

## 📁 Repository Structure

```
shrestidb/
├── Cargo.toml                          # Crate configuration & binaries
├── README.md                           # Documentation & quick start
├── DESIGN.md                           # Architectural design specification
│
├── src/
│   ├── main.rs                         # Interactive CLI REPL & command runner
│   ├── lib.rs                          # ShrestiDB library root
│   ├── error.rs                        # DatabaseError & Result types
│   │
│   ├── bin/                            # Binary entry points
│   │   ├── db_server.rs                # Standalone background daemon
│   │   └── studio.rs                   # Standalone Web Studio runner
│   │
│   ├── studio.rs                       # Embedded Tokio HTTP server & SPA UI
│   │
│   ├── index/                          # [LAYER 1] Learned Indexing
│   │   ├── mod.rs
│   │   ├── models.rs                   # Linear regression segment models
│   │   ├── pgm.rs                      # Static & Dynamic PGM Index
│   │   ├── rmi.rs                      # Multi-stage Recursive Model Index
│   │   └── btree.rs                    # Node-splitting B+Tree comparison baseline
│   │
│   ├── optimizer/                      # [LAYER 2] Query Optimization
│   │   ├── mod.rs
│   │   ├── cardinality.rs              # PGM-based empirical CDF cardinality
│   │   ├── cost_model.rs               # CPU/IO cost evaluation
│   │   ├── join_reorder.rs             # Cost-based optimal join orderer
│   │   └── planner.rs                  # Logical & physical plan generator
│   │
│   ├── execution/                      # [LAYER 3] Query Engine & Storage
│   │   ├── mod.rs
│   │   ├── catalog.rs                  # Table schemas, types, and metadata
│   │   ├── executor.rs                 # SQL execution coordinator
│   │   ├── operators.rs                # Tuple streams, values, and operators
│   │   ├── mvcc_store.rs               # Multi-version row chains
│   │   ├── oltp.rs                     # Transactional read/write coordinator
│   │   ├── lock_manager.rs             # Row-level 2PL with deadlock detection
│   │   ├── wal.rs                      # Append-only Write-Ahead Log
│   │   ├── recovery.rs                 # Replay-based crash recovery
│   │   ├── secondary_index.rs          # Secondary index integration
│   │   ├── aggregate.rs                # GroupBy and aggregation functions
│   │   ├── row_codec.rs                # Row encoding and compiled predicates
│   │   └── learned_retention.rs        # ML-driven MVCC garbage collection
│   │
│   ├── sql/                            # [LAYER 4] SQL Processing
│   │   ├── mod.rs
│   │   ├── parser.rs                   # SQL parser wrapper around sqlparser
│   │   ├── binder.rs                   # Catalog validation and binding
│   │   └── types.rs                    # AST representations
│   │
│   └── compute/                        # [UTILITY] SIMD Acceleration
│       ├── mod.rs
│       ├── simd_ops.rs                 # AVX2/portable SIMD bounded search
│       └── vector_math.rs              # Vectorized array math
│
├── examples/                           # End-to-end demonstrations
│   ├── learned_index_demo.rs           # PGM vs B-Tree microbenchmark
│   ├── memory_footprint.rs             # Live RAM comparison
│   ├── monotonic_ingestion_niche.rs    # Telemetry ingestion benchmark
│   ├── vs_postgres.rs                  # PostgreSQL comparative benchmark
│   ├── vs_sqlite.rs                    # SQLite comparative benchmark
│   └── cardinality_demo.rs             # Learned cardinality estimation demo
│
└── tests/                              # Comprehensive test suite (346 tests)
    ├── dynamic_pgm_tests.rs
    ├── integration_tests.rs
    ├── perf_tests.rs
    └── sql_tests.rs
```

---

## 🧪 Testing & Validation

Run the complete test suite:

```bash
cargo test --lib
```

All **346 unit and integration tests** validate:
- Learned index bounded prediction accuracy and segment splits
- Strict predicate type safety (unsupported expressions error cleanly rather than causing silent corruption)
- Concurrent transaction correctness and 2PL lock release verification
- File-backed WAL crash recovery across restarts

---

## 📜 License

Copyright © 2026 Shresti Corp. All rights reserved.
