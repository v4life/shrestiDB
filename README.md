# ShrestiDB ⚡

[![Build & Test](https://img.shields.io/badge/tests-377%20passed-brightgreen.svg)]()
[![Rust](https://img.shields.io/badge/rust-2021%20edition-orange.svg)]()
[![Docker](https://img.shields.io/badge/docker-ready-2496ED.svg)]()
[![License](https://img.shields.io/badge/license-Proprietary-blue.svg)]()

**ShrestiDB** is an ACID-compliant, hybrid relational and vector database kernel written in Rust. It integrates **learned machine learning models** directly into indexing, query optimization, and storage retention:

* **Learned Indexing (PGM & RMI)**: Up to **8x – 10x faster** lookups than a traditional B+Tree at scale, while using **~4x less memory**.
* **Unified Relational + Vector Engine**: Native `VECTOR(N)` support, SIMD distance operators (`<=>`, `<->`, `<#>`), and an HNSW graph index executed inside classical SQL pipelines.
* **MVCC & ACID Durability**: Multi-version concurrency control with snapshot isolation, row-level 2PL lock management, and write-ahead logging (WAL) for instant crash recovery.
* **Learned Storage Retention (Auto-GC)**: Adaptive online regression models predict dead version bloat to automatically sweep dead tuples without manual `VACUUM`.
* **Zero-Dependency Web Studio & AI Agent**: Self-contained browser management console and natural language SQL assistant running on an asynchronous Tokio HTTP server.

> 📖 **Deep Dive**: For full technical design, component diagrams, and algorithmic workflows, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## 📊 Measured Benchmark Results

*Reproducible with `cargo run --release --example learned_index_demo`*:

| Dataset Size | PGM Lookup | RMI Lookup | Real B+Tree | PGM Speedup | Index Memory |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **10,000** | 0.55 µs | 0.18 µs | 0.10 µs | 0.5x | ~4x smaller |
| **100,000** | 0.41 µs | 0.22 µs | 0.30 µs | 0.7x | ~4.1x smaller |
| **1,000,000** | **0.44 µs** | **0.37 µs** | **3.61 µs** | **8.2x – 9.8x** | **~4.2x smaller** |

While B-Trees perform well on tiny datasets fitting in L1/L2 cache, learned models scale flatly at $O(\log \epsilon)$, overtaking B-Trees by **8x – 10x** at 1M+ keys while consuming **75% less RAM**.

---

## 🚀 Quick Start

### 1. Interactive CLI Shell

Launch the interactive REPL with arrow-key history and formatted ASCII tables:

```bash
cargo run --bin shrestidb
```

```sql
shrestidb> CREATE TABLE accounts (id INT PRIMARY KEY, name TEXT, balance FLOAT);
OK (took 19.52ms)

shrestidb> INSERT INTO accounts VALUES (1, 'Veer', 5000.0);
Query OK, 1 row affected (took 22.48ms)

shrestidb> SELECT * FROM accounts WHERE balance >= 3000;
+----+------+---------+
| id | name | balance |
+----+------+---------+
| 1  | Veer | 5000    |
+----+------+---------+
(1 row, took 210.15µs)
```

### 2. Browser Web Studio & AI Console

Start the embedded Web Studio (binds to `0.0.0.0:8080`):

```bash
cargo run --bin shrestidb-studio
```

Open **[http://localhost:8080/](http://localhost:8080/)**:
* **SQL Editor**: Execute queries with instant schema completion (`⌘ + Enter` to run).
* **AI SQL Agent**: Chat with your database in plain English via local Ollama or cloud LLMs.
* **Live Benchmark Visualizer**: Benchmark learned PGM models vs B-Trees on live generated datasets.

---

## ⚡ Unified SQL: Relational + Vector Similarity

Query structured relational data and high-dimensional embeddings in a single statement:

```sql
-- 1. Create a table with metadata and vector embeddings
CREATE TABLE documents (
    id INT PRIMARY KEY,
    title TEXT,
    embedding VECTOR(3)
);

-- 2. Insert records
INSERT INTO documents VALUES (1, 'Deep Learning Paper', '[0.95, 0.10, 0.05]');
INSERT INTO documents VALUES (2, 'Database Kernels', '[0.10, 0.90, 0.20]');
INSERT INTO documents VALUES (3, 'AI Systems', '[0.85, 0.20, 0.15]');

-- 3. Nearest-neighbor vector similarity search using Cosine distance (<=>)
SELECT title, embedding <=> '[0.90, 0.15, 0.05]' AS distance
FROM documents
ORDER BY distance
LIMIT 2;
```

#### Vector Distance Operators:
* `<=>` **Cosine Distance** ($1 - \cos \theta$) — Semantic search, text embeddings, RAG
* `<->` **L2 Euclidean Distance** ($\|\vec{u} - \vec{v}\|_2$) — Spatial coordinates, geometric search
* `<#>` **Negative Dot Product** ($-\vec{u} \cdot \vec{v}$) — Normalized unit vector scoring

---

## 🐳 Cloud & Docker Deployment

ShrestiDB is container-ready for AWS ECS, GCP Cloud Run, Fly.io, and Kubernetes with native IP rate limiting and public demo protection:

```bash
# Build production Docker image
docker build -t shrestidb .

# Run container with persistent storage and rate limiting (default 60 req/min)
docker run -p 8080:8080 \
  -e SHRESTIDB_RATE_LIMIT=60 \
  -e SHRESTIDB_DEMO_MODE=1 \
  -v shrestidb_data:/data shrestidb
```

Health probes for orchestrators:
```bash
curl -i http://localhost:8080/health
# HTTP/1.1 200 OK
```

---

## 🧪 Testing

Run the test suite:

```bash
cargo test --lib
```

All **377 unit and integration tests** pass, validating:
* Learned index bounded prediction accuracy and segment splits
* SIMD vector distance calculations (`<=>`, `<->`, `<#>`) and HNSW recall
* Multi-table chained join predicate partitioning and schema projection
* Concurrent transaction correctness and 2PL lock release verification
* File-backed WAL crash recovery across restarts

---

## 📜 License

Copyright © 2026 Shresti Corp. All rights reserved.
