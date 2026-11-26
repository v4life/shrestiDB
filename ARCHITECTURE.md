# ShrestiDB Architecture & System Design

This document details the internal architecture, engine layers, storage mechanisms, and algorithmic workflows powering **ShrestiDB**.

---

## 1. System Overview

ShrestiDB is an embedded, unified database engine that combines:
1. **Classical Relational Execution** (ANSI SQL, multi-table hash joins, MVCC transactions, ACID durability).
2. **Learned Machine Learning Indexes** (Piecewise Geometric Models & Recursive Model Indexes).
3. **High-Dimensional Vector Search** (HNSW proximity graph with SIMD distance kernels).
4. **Self-Contained Management & AI** (Embedded Tokio async HTTP Web Studio and local AI agent).

```mermaid
flowchart TD
    subgraph Client ["Client Interfaces"]
        CLI["CLI REPL (Rustyline, history)"]
        Browser["Web Studio UI (Pure Tokio HTTP / SPA)"]
        AI["AI Natural Language Agent (Ollama / Cloud LLM)"]
    end

    subgraph SQL ["SQL & Query Planning Layer"]
        Parser["SQL Parser (sqlparser-rs)"]
        Binder["Catalog Binder & Type Checker"]
        Planner["Logical & Physical Planner"]
        JoinOrderer["Dynamic Programming Join Reorderer"]
        CardEst["Empirical CDF Cardinality Estimator (PGM)"]
    end

    subgraph Exec ["Query Execution Engine"]
        ExecEngine["Volcano / Vectorized Iterator"]
        HashJoin["Partitioned Hash Join Operator"]
        NLJoin["Compiled Nested Loop Join"]
        Agg["GroupBy & Aggregate Compute (COUNT, SUM, AVG)"]
        SortLimit["Sort (Scalar / Vector Similarity) & Limit"]
    end

    subgraph Indexes ["Unified Index Layer"]
        PGM["Learned PGM Index (Piecewise Geometric Models)"]
        RMI["Recursive Model Index (Two-Stage Regression)"]
        HNSW["HNSW Graph Index (Vector Similarity)"]
        BTree["Node-Splitting B+Tree (Baseline)"]
        SIMD["SIMD Compute (AVX2 / NEON Search & Metrics)"]
    end

    subgraph Storage ["Storage & Durability Layer"]
        MVCC["MVCC Store (Version Chains & Epoch Timestamps)"]
        LockMgr["Row-Level 2PL Lock Manager (Deadlock Detection)"]
        WAL["Append-Only Write-Ahead Log (CRC32 Checksums)"]
        GC["Learned Retention GC (Version Bloat Sweeper)"]
    end

    CLI --> Parser
    Browser --> Parser
    AI --> Parser

    Parser --> Binder --> Planner
    CardEst -.-> JoinOrderer
    JoinOrderer --> Planner
    Planner --> ExecEngine

    ExecEngine --> HashJoin
    ExecEngine --> NLJoin
    ExecEngine --> Agg
    ExecEngine --> SortLimit

    ExecEngine --> Indexes
    Indexes --> SIMD

    ExecEngine --> Storage
    Storage --> WAL
    MVCC --> GC
```

---

## 2. Layer Deep Dives

### 2.1 SQL & Optimization Layer

```mermaid
sequenceDiagram
    autonumber
    participant Client
    participant Parser
    participant Binder
    participant Planner
    participant Engine as Execution Engine

    Client->>Parser: SQL Query (e.g. SELECT ... JOIN ... ORDER BY embedding <=> '[...]')
    Parser->>Binder: AST (SelectStatement, Joins, Projections)
    Binder->>Binder: Resolve table schemas, column aliases, & vector dimensions
    Binder->>Planner: Validated AST
    Planner->>Planner: Cost evaluation & Join reordering (JoinOrderer)
    Planner->>Engine: Physical Plan (Scan -> Join -> Filter -> Sort -> Project)
    Engine->>Client: Tabular Tuple Stream
```

* **Parser & Binder**: Wraps standard SQL AST. Resolves table aliases (`accounts a`), multi-column aliases (`z.zip_code AS address_zip_code`), and unqualified identifiers across join schemas.
* **Learned Cardinality Estimator**: During `ANALYZE <table>`, builds an empirical Cumulative Distribution Function (CDF) by fitting a PGM model over sorted column keys, avoiding inaccurate static 50% selectivity assumptions.
* **Dynamic Programming Join Reordering**: Uses PGM cardinality estimates to determine the optimal equi-join order, placing small build tables on the left of hash partitions.

---

### 2.2 Learned Index Architecture (PGM & RMI)

Traditional B-Trees incur $O(\log N)$ branch mispredictions across tree nodes. ShrestiDB models index lookup as a numeric regression problem:

$$f(\text{key}) \to \text{physical record position}$$

```mermaid
flowchart LR
    Key(["Search Key: k"]) --> SegmentLookup["Segment Binary Search (O(log S))"]
    SegmentLookup --> LinearModel["Linear Model: pos = slope * k + intercept"]
    LinearModel --> BoundedRange["Predicted Range: [pos - ε, pos + ε]"]
    BoundedRange --> SIMDSearch["SIMD Bounded Binary Search (O(log ε))"]
    SIMDSearch --> ExactRecord["Exact Tuple Record ID"]
```

* **Piecewise Geometric Model (PGM)**: Partitions sorted keys into optimal linear segments with bounded error $\epsilon$. Because segments store only slope and intercept $(m, c)$, index size is reduced by **~4x** compared to a node-splitting B+Tree.
* **Dynamic PGM**: Pairs a frozen, optimal PGM base with an append-only write buffer, amortizing bulk-load rebuilds geometrically.
* **Performance**: With $S$ segments where $S \ll N$, lookup complexity is $O(\log S) + O(\log \epsilon)$, achieving sub-microsecond point queries on million-row tables.

---

### 2.3 Vector Database Engine (HNSW & SIMD)

For high-dimensional embeddings (`VECTOR(N)`), ShrestiDB incorporates a multi-layer **Hierarchical Navigable Small World (HNSW)** graph index:

```mermaid
flowchart TD
    subgraph Layer2 ["Layer 2 (Sparse Graph - Express Navigation)"]
        L2_A["Node A"] --- L2_B["Node B"]
    end

    subgraph Layer1 ["Layer 1 (Medium Density Graph)"]
        L1_A["Node A"] --- L1_C["Node C"]
        L1_C --- L1_B["Node B"]
    end

    subgraph Layer0 ["Layer 0 (Dense Ground Layer - All Vectors)"]
        L0_A["Node A"] --- L0_D["Node D"]
        L0_D --- L0_C["Node C"]
        L0_C --- L0_E["Node E"]
        L0_E --- L0_B["Node B"]
    end

    Entry(["Query Vector Q"]) --> Layer2
    Layer2 -->|Greedy Search| Layer1
    Layer1 -->|Greedy Search| Layer0
    Layer0 --> K_Nearest(["K Nearest Neighbors"])
```

* **Distance Metrics**:
  - `<=>` **Cosine Distance**: $1 - \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|}$ (semantic and text search)
  - `<->` **L2 Euclidean Distance**: $\|\vec{u} - \vec{v}\|_2$ (spatial and geometric search)
  - `<#>` **Negative Dot Product**: $-\vec{u} \cdot \vec{v}$ (normalized unit embeddings)
* **Hardware SIMD**: Evaluates dot products and Euclidean norms using portable 8-wide f32 SIMD vector registers (AVX2 / NEON).

---

### 2.4 Transactional Storage & MVCC Lifecycle

```mermaid
stateDiagram-v2
    [*] --> ActiveTransaction: BEGIN (Alloc Tx ID & Read Timestamp)
    ActiveTransaction --> Acquire2PL: Write Operation (INSERT/UPDATE/DELETE)
    Acquire2PL --> CheckDeadlock: Cycle Detection Timeout (100ms)
    CheckDeadlock --> ActiveTransaction: Lock Acquired
    CheckDeadlock --> Abort: Timeout / Deadlock Detected
    ActiveTransaction --> AppendWAL: COMMIT
    AppendWAL --> ApplyVersionChain: Flush to Disk with CRC32
    ApplyVersionChain --> [*]: Clean Exit
    
    state MVCC_GC {
        [*] --> VersionChainInspection
        VersionChainInspection --> LearnedRetentionModel: Predict Bloat Rate
        LearnedRetentionModel --> SweepDeadTuples: Benefit > Sweep Cost
        SweepDeadTuples --> [*]: Reclaim Memory
    }
```

* **Snapshot Isolation (MVCC)**: Queries read row versions visible as of their `BEGIN` transaction timestamp. Read transactions execute lock-free.
* **Row-Level 2PL**: Exclusive write locks guard mutated records with a 100ms cycle-detection timeout to prevent deadlocks.
* **Crash Recovery (WAL)**: All table operations log binary records to disk before acknowledgment. On boot, the replay engine rebuilds catalogs, indexes, and row tables deterministically.
* **Learned Retention GC**: An online linear model predicts version churn, triggering row-chain compaction when reclaimable space exceeds sweep overhead, eliminating manual `VACUUM`.
