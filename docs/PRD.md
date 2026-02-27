# Silica — Product Requirements Document

> A production-grade, embedded-first relational database written in Zig.

**Version:** 0.1 (Draft)
**Date:** February 27, 2026
**Author:** Yusa

---

## 1. Overview

### 1.1 Vision

Silica is a lightweight, high-performance relational database engine written entirely in Zig. Inspired by SQLite's simplicity and embeddability, Silica aims to deliver a production-ready storage engine that can be used as an embedded library or as a standalone client-server process. By leveraging Zig's manual memory control, comptime capabilities, and C ABI compatibility, Silica targets use cases where predictable performance, minimal dependencies, and a small binary footprint are critical.

### 1.2 Why Zig?

- **No hidden allocations or control flow.** Every allocation is explicit, making it straightforward to reason about memory usage and lifetime — essential for a database engine.
- **Comptime metaprogramming.** Schema validation, query plan optimization hints, and wire protocol serialization can be partially evaluated at compile time.
- **C ABI interop with zero overhead.** The embedded library can be consumed from C, C++, Python, Node.js, Go, and any language with C FFI — just like SQLite.
- **No garbage collector.** Eliminates GC pauses that can cause unpredictable latency spikes during transactions.
- **Cross-compilation out of the box.** A single build step can produce binaries for Linux, macOS, Windows, and WASI.

### 1.3 Project Goals

| Priority | Goal |
|----------|------|
| P0 | Production-grade embedded relational database with ACID guarantees |
| P0 | Standard SQL subset support (DDL, DML, basic joins, aggregations) |
| P1 | High performance for OLTP workloads with predictable latency |
| P1 | Single-file database format with crash recovery |
| P2 | Optional client-server mode with a wire protocol |
| P3 | Extensibility (custom functions, virtual tables) |

---

## 2. Target Users & Use Cases

### 2.1 Primary Users

- **Application developers** who need an embedded database (mobile apps, desktop apps, CLI tools, IoT devices).
- **Systems programmers** building infrastructure that requires a reliable local data store.
- **Zig ecosystem developers** looking for a native database library without C dependency wrappers.

### 2.2 Key Use Cases

1. **Embedded application storage** — Local-first apps that store structured data in a single file (e.g., note-taking apps, configuration stores, local caches).
2. **Edge / IoT data collection** — Resource-constrained environments where SQLite's C codebase is hard to audit or cross-compile.
3. **Testing & prototyping** — Drop-in embedded DB for integration tests without requiring a server process.
4. **Lightweight server database** — Small-to-medium services (< 100 concurrent connections) that don't need a full-scale RDBMS.

---

## 3. Architecture

### 3.1 High-Level Component Diagram

```
┌──────────────────────────────────────────────────┐
│                   Client Layer                    │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────┐ │
│  │ Zig API     │  │ C API (FFI)  │  │ Wire    │ │
│  │ (Embedded)  │  │ (Embedded)   │  │ Protocol│ │
│  └──────┬──────┘  └──────┬───────┘  └────┬────┘ │
│         └────────────────┼────────────────┘      │
├──────────────────────────┼───────────────────────┤
│                    SQL Frontend                   │
│  ┌──────────┐  ┌─────────┴──────┐  ┌──────────┐ │
│  │ Tokenizer│→ │     Parser     │→ │ Semantic │ │
│  │ (Lexer)  │  │ (AST Builder)  │  │ Analyzer │ │
│  └──────────┘  └────────────────┘  └────┬─────┘ │
├─────────────────────────────────────────┼────────┤
│                 Query Engine             │        │
│  ┌──────────┐  ┌────────────────┐  ┌────┴─────┐ │
│  │ Query    │→ │   Optimizer    │→ │  VM /    │ │
│  │ Planner  │  │ (Rule-based)   │  │ Executor │ │
│  └──────────┘  └────────────────┘  └────┬─────┘ │
├─────────────────────────────────────────┼────────┤
│              Transaction Manager         │        │
│  ┌──────────┐  ┌────────────────┐  ┌────┴─────┐ │
│  │   WAL    │  │  Lock Manager  │  │  MVCC    │ │
│  │  Writer  │  │  (Page-level)  │  │ (future) │ │
│  └──────────┘  └────────────────┘  └────┬─────┘ │
├─────────────────────────────────────────┼────────┤
│                Storage Engine            │        │
│  ┌──────────┐  ┌────────────────┐  ┌────┴─────┐ │
│  │  B+Tree  │  │  Page Manager  │  │  Buffer  │ │
│  │  Index   │  │  (Pager)       │  │  Pool    │ │
│  └──────────┘  └────────────────┘  └──────────┘ │
├──────────────────────────────────────────────────┤
│                    OS Layer                       │
│  ┌──────────┐  ┌────────────────┐  ┌──────────┐ │
│  │  File    │  │  Memory-mapped  │  │  fsync   │ │
│  │  I/O     │  │  I/O (optional) │  │  Control │ │
│  └──────────┘  └────────────────┘  └──────────┘ │
└──────────────────────────────────────────────────┘
```

### 3.2 Component Responsibilities

#### 3.2.1 Storage Engine (Phase 1 — Milestone 1–2)

The storage engine is the foundation. It manages how data is physically stored on disk and retrieved into memory.

**Page Manager (Pager)**
- Fixed-size pages (default 4096 bytes, configurable: 512 – 65536).
- Each database is a single file composed of sequential pages.
- Page types: metadata, internal B+Tree node, leaf node, overflow, freelist.
- Page header contains: page type, page number, checksum (CRC32C), free space pointer.

**B+Tree**
- Primary data structure for both table storage (clustered index on rowid) and secondary indexes.
- Variable-length keys and values with overflow page support for large payloads.
- Operations: insert, delete, point lookup, range scan (forward/backward cursors).
- Page splits and merges maintain balance invariants.
- Leaf pages are doubly linked for efficient range scans.

**Buffer Pool**
- LRU-based page cache with configurable size (default: 2000 pages ≈ 8 MB).
- Dirty page tracking for write-back.
- Pin/unpin semantics to prevent eviction of pages in active use.
- Optional direct I/O bypass for bulk loading.

**File Format**
```
┌─────────────────────────────────┐
│  Page 0: Database Header        │
│  - Magic bytes ("SLCA")         │
│  - Format version               │
│  - Page size                    │
│  - Total page count             │
│  - Freelist head pointer        │
│  - Schema version               │
│  - WAL mode flag                │
│  - Reserved region              │
├─────────────────────────────────┤
│  Page 1: Schema Table Root      │
│  (B+Tree root for sqlite_master │
│   equivalent)                   │
├─────────────────────────────────┤
│  Page 2..N: Data & Index Pages  │
└─────────────────────────────────┘
```

#### 3.2.2 SQL Frontend (Phase 2 — Milestone 3–4)

**Tokenizer**
- Hand-written lexer (no external parser generators).
- Token types: keywords, identifiers, literals (integer, float, string, blob), operators, punctuation.
- Support for quoted identifiers (`"column"`) and string escaping.

**Parser**
- Recursive descent parser producing a typed AST.
- Supported statement types (initial scope):

| Category | Statements |
|----------|-----------|
| DDL | `CREATE TABLE`, `DROP TABLE`, `CREATE INDEX`, `DROP INDEX` |
| DML | `SELECT`, `INSERT`, `UPDATE`, `DELETE` |
| Transaction | `BEGIN`, `COMMIT`, `ROLLBACK` |
| Utility | `EXPLAIN`, `PRAGMA` |

- SELECT support scope:
  - `WHERE` clauses with comparison, `AND`/`OR`, `IN`, `BETWEEN`, `IS NULL`
  - `ORDER BY`, `LIMIT`, `OFFSET`
  - `GROUP BY`, `HAVING`
  - Aggregate functions: `COUNT`, `SUM`, `AVG`, `MIN`, `MAX`
  - `INNER JOIN`, `LEFT JOIN` (initial; `RIGHT`/`FULL` later)
  - Subqueries in `WHERE` and `FROM` clauses
  - `DISTINCT`, `UNION`, `UNION ALL`

**Semantic Analyzer**
- Type checking and name resolution against the schema catalog.
- Validates column references, table existence, and type compatibility.
- Resolves `*` expansion and implicit column references.

#### 3.2.3 Query Engine (Phase 2 — Milestone 4–5)

**Query Planner**
- Translates the validated AST into a logical plan (relational algebra tree).
- Logical operators: Scan, Filter, Project, Join, Sort, Aggregate, Limit.

**Optimizer (Rule-Based)**
- Predicate pushdown (push filters below joins).
- Index selection (choose index vs. full table scan based on selectivity heuristics).
- Join order optimization for queries with ≤ 4 tables (exhaustive search); greedy heuristic for more.
- Constant folding and expression simplification.

**Executor**
- Volcano-model (iterator-based, pull) execution.
- Each operator implements `open()`, `next()`, `close()`.
- Produces `Row` tuples that flow up the operator tree.
- Memory-aware: external merge sort for large `ORDER BY`, hash-based grouping with spill-to-disk.

#### 3.2.4 Transaction Manager (Phase 3 — Milestone 5–6)

**WAL (Write-Ahead Log)**
- All modifications are written to a WAL file before being applied to the main database.
- WAL format: sequence of frames, each containing page number + page data + checksum.
- Checkpoint process merges WAL frames back into the main database file.
- Supports `NORMAL` and `FULL` sync modes (trade durability vs. performance).

**Concurrency Control**
- Phase 1: Single-writer, multiple-reader using WAL-based isolation.
  - Readers see a consistent snapshot from the start of their transaction.
  - Writers acquire an exclusive lock; concurrent writes are serialized.
- Phase 2 (future): MVCC for improved read-write concurrency.

**ACID Properties**
| Property | Implementation |
|----------|---------------|
| Atomicity | WAL ensures all-or-nothing; rollback discards uncommitted WAL frames |
| Consistency | Schema validation + constraint checks (NOT NULL, UNIQUE, PK, FK) |
| Isolation | Snapshot isolation via WAL read markers |
| Durability | fsync on WAL commit; checkpoint flushes to main DB file |

#### 3.2.5 Client-Server Mode (Phase 4 — Milestone 7)

**Wire Protocol**
- Custom binary protocol over TCP (inspired by PostgreSQL's wire protocol).
- Message types: Query, Parse, Bind, Execute, Describe, Close, Sync, Error, RowData, CommandComplete.
- TLS support via system libraries.

**Connection Handling**
- Thread-per-connection model (leveraging Zig's lightweight threading).
- Connection pooling support in client library.
- Authentication: password-based (SHA-256 challenge-response).

---

## 4. Data Types

### 4.1 Core Types

| SQL Type | Internal Representation | Size |
|----------|------------------------|------|
| `INTEGER` | i64 | 1–8 bytes (varint) |
| `REAL` | f64 | 8 bytes |
| `TEXT` | UTF-8 `[]const u8` | variable |
| `BLOB` | Raw `[]const u8` | variable |
| `BOOLEAN` | i64 (0 or 1) | 1 byte (varint) |
| `NULL` | Tag only | 0 bytes |

### 4.2 Type Affinity

Follow SQLite-style type affinity rules for flexibility:
- Column types map to an affinity (INTEGER, REAL, TEXT, BLOB, NUMERIC).
- Values are stored in the most compact representation that preserves information.

---

## 5. Constraints & Integrity

| Constraint | Phase |
|-----------|-------|
| `PRIMARY KEY` (single column, implicit rowid) | Phase 1 |
| `NOT NULL` | Phase 1 |
| `UNIQUE` | Phase 2 |
| `DEFAULT` values | Phase 2 |
| `CHECK` expressions | Phase 3 |
| `FOREIGN KEY` with `ON DELETE`/`ON UPDATE` actions | Phase 3 |
| Composite `PRIMARY KEY` | Phase 2 |

---

## 6. API Design

### 6.1 Zig Embedded API (Primary)

```zig
const silica = @import("silica");

// Open or create a database
var db = try silica.open("app.db", .{
    .page_size = 4096,
    .cache_size = 2000,
    .wal_mode = true,
});
defer db.close();

// Execute DDL
try db.exec("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT)");

// Prepared statements with parameter binding
var stmt = try db.prepare("SELECT id, name FROM users WHERE email = ?");
defer stmt.deinit();

try stmt.bind(.{ "alice@example.com" });
var rows = try stmt.execute();

while (try rows.next()) |row| {
    const id = row.get(i64, 0);
    const name = row.get([]const u8, 1);
    // ...
}
```

### 6.2 C API (FFI)

```c
silica *db;
int rc = silica_open("app.db", &db);

silica_stmt *stmt;
rc = silica_prepare(db, "SELECT * FROM users WHERE id = ?", -1, &stmt, NULL);
silica_bind_int64(stmt, 1, 42);

while (silica_step(stmt) == SILICA_ROW) {
    const char *name = silica_column_text(stmt, 1);
    // ...
}

silica_finalize(stmt);
silica_close(db);
```

---

## 7. Non-Functional Requirements

### 7.1 Performance Targets

| Metric | Target | Benchmark |
|--------|--------|-----------|
| Point lookup (by PK) | < 5 µs (cached) | 1M random reads |
| Sequential insert | > 100K rows/sec | 1M row bulk insert |
| Range scan throughput | > 500K rows/sec | 100-byte rows, cached |
| Database open time | < 10 ms | 1 GB database |
| Binary size (embedded, stripped) | < 2 MB | Release build |
| Memory overhead (idle) | < 1 MB + cache | Embedded mode |

### 7.2 Reliability

- Crash recovery: database must be recoverable after power loss or process kill at any point during a transaction.
- Fuzz testing: tokenizer, parser, and B+Tree operations must pass 72+ hours of continuous fuzzing with no crashes.
- No undefined behavior: zero `@intCast` / `@ptrCast` UB; all edge cases handled explicitly.

### 7.3 Compatibility

- Zig version: 0.14.x stable (track latest stable).
- OS support: Linux (primary), macOS, Windows. WASI as stretch goal.
- No external C dependencies for the core engine.

---

## 8. Milestone Plan

### Phase 1: Storage Foundation (Weeks 1–6)

**Milestone 1 — Page Manager & File Format** (Weeks 1–2)
- [ ] Define and implement the database file header (Page 0)
- [ ] Implement page read/write with checksums
- [ ] Freelist management (allocate/free pages)
- [ ] Basic test suite: create DB, write pages, reopen and verify

**Milestone 2 — B+Tree & Buffer Pool** (Weeks 3–6)
- [ ] B+Tree insert, delete, point lookup
- [ ] Leaf page splits and merges
- [ ] Range scan cursors (forward/backward)
- [ ] Overflow pages for large values
- [ ] LRU buffer pool with dirty page tracking
- [ ] Comprehensive B+Tree fuzz tests

### Phase 2: SQL Layer (Weeks 7–14)

**Milestone 3 — Tokenizer & Parser** (Weeks 7–9)
- [ ] Hand-written tokenizer with full SQL token coverage
- [ ] Recursive descent parser → typed AST
- [ ] DDL: `CREATE TABLE`, `DROP TABLE`
- [ ] DML: basic `SELECT`, `INSERT`, `UPDATE`, `DELETE`
- [ ] Parser error recovery with meaningful error messages

**Milestone 4 — Semantic Analysis & Execution** (Weeks 10–14)
- [ ] Schema catalog (in-memory schema table backed by page 1)
- [ ] Name resolution and type checking
- [ ] Query planner: AST → logical plan → physical plan
- [ ] Volcano-model executor: Scan, Filter, Project, Sort, Limit
- [ ] `WHERE` clause evaluation with index selection
- [ ] `JOIN` execution (nested loop + index lookup)
- [ ] Aggregate functions and `GROUP BY`

### Phase 3: Transactions & ACID (Weeks 15–20)

**Milestone 5 — WAL & Crash Recovery** (Weeks 15–17)
- [ ] WAL file format and frame writer
- [ ] Read-path WAL integration (check WAL before main DB)
- [ ] Checkpoint process (WAL → main DB)
- [ ] Crash recovery tests (simulate kill at every write point)

**Milestone 6 — Concurrency & Constraints** (Weeks 18–20)
- [ ] Single-writer / multiple-reader lock protocol
- [ ] Snapshot isolation for readers
- [ ] `UNIQUE` constraint enforcement via index
- [ ] `FOREIGN KEY` basic support
- [ ] Savepoints and nested transactions

### Phase 4: Client-Server & Polish (Weeks 21–26)

**Milestone 7 — Wire Protocol & Server** (Weeks 21–23)
- [ ] Binary wire protocol implementation
- [ ] TCP server with connection management
- [ ] Client library (Zig)
- [ ] Authentication (SHA-256 challenge-response)

**Milestone 8 — Production Readiness** (Weeks 24–26)
- [ ] `EXPLAIN` and `PRAGMA` support
- [ ] Comprehensive benchmark suite
- [ ] Fuzz testing campaign (72+ hours, all components)
- [ ] Documentation: API reference, architecture guide, getting started
- [ ] CI/CD pipeline: build matrix (Linux/macOS/Windows), test, fuzz, bench
- [ ] Package: `build.zig` for consumers, C header generation

---

## 9. Testing Strategy

### 9.1 Test Pyramid

| Level | Scope | Tool |
|-------|-------|------|
| Unit | Individual functions (B+Tree ops, tokenizer, page read/write) | Zig built-in `test` |
| Integration | Multi-component flows (SQL → parse → plan → execute → storage) | Custom test harness |
| Fuzz | Tokenizer, parser, B+Tree insert/delete sequences | Zig's built-in fuzzing + AFL |
| Crash | WAL recovery under simulated power failure | Custom crash injector |
| Compatibility | SQL test suites (adapted from SQLite's test corpus) | SQL script runner |
| Performance | Regression benchmarks on every commit | Zig `std.time` + custom bench framework |

### 9.2 Correctness Invariants

- B+Tree: all keys in a node are sorted; parent keys correctly partition children; leaf chain is consistent.
- WAL: after crash recovery, database content matches the last committed transaction.
- Concurrency: no torn reads; readers always see a consistent snapshot.

---

## 10. Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| B+Tree edge cases (splits during concurrent reads) | Data corruption | Medium | Extensive fuzz testing; formal invariant checks after every operation |
| WAL recovery bugs | Data loss | High (if untested) | Crash injection test suite; test every combination of failure points |
| SQL compatibility gaps surprise users | Adoption friction | Medium | Document supported SQL subset clearly; provide `EXPLAIN` for transparency |
| Zig language breaking changes | Build failures | Low (targeting stable) | Pin Zig version; CI tests on nightly for early warning |
| Scope creep (adding too many SQL features) | Delayed release | High | Strict phase gating; MVP = Phase 1–2 with basic transactions |

---

## 11. Future Roadmap (Post v1.0)

These are explicitly **out of scope** for the initial release but inform architectural decisions:

- **MVCC** — Multi-version concurrency control for true read-write parallelism.
- **Full-text search** — Built-in FTS with inverted index (virtual table).
- **JSON support** — `JSON` column type with path-based query operators.
- **Replication** — WAL-based streaming replication for read replicas.
- **WASM target** — Compile to WebAssembly for browser-based usage.
- **Virtual tables** — Plugin interface for custom storage backends.
- **Window functions** — `ROW_NUMBER`, `RANK`, `LAG`, `LEAD`, etc.

---

## 12. Success Criteria

The project is considered successful at v1.0 when:

1. **Functional completeness** — All Phase 1–3 milestones are complete and passing tests.
2. **Reliability** — 72+ hours of fuzz testing with zero crashes; crash recovery tests pass 100%.
3. **Performance** — Meets or exceeds all targets in Section 7.1 on reference hardware.
4. **Usability** — A developer can `@import("silica")` in their Zig project and run SQL queries within 5 minutes using the documentation.
5. **Code quality** — Zero known undefined behavior; all public APIs documented; test coverage > 80% by line.

---

## Appendix A: Reference Projects

| Project | Relevance |
|---------|-----------|
| [SQLite](https://sqlite.org) | Primary inspiration for embedded design, file format concepts, SQL subset |
| [rqlite](https://github.com/rqlite/rqlite) | Distributed SQLite — reference for client-server layering |
| [DuckDB](https://duckdb.org) | Modern embedded analytical DB — reference for API design |
| [TigerBeetle](https://tigerbeetle.com) | Production Zig database — reference for Zig-specific patterns, io_uring usage |
| [Chiselstore](https://github.com/nickel-lang/chiselstore) | Rust-based SQLite-like — reference for B+Tree implementation |
| [CMU DB Course (BusTub)](https://15445.courses.cs.cmu.edu) | Educational DB — reference for component architecture |

## Appendix B: Glossary

| Term | Definition |
|------|-----------|
| **B+Tree** | A self-balancing tree data structure optimized for disk-based storage where all values reside in leaf nodes |
| **WAL** | Write-Ahead Log — a technique where changes are written to a log before being applied to the main data file |
| **Buffer Pool** | An in-memory cache of database pages that reduces disk I/O |
| **Pager** | The component responsible for reading and writing fixed-size pages to/from disk |
| **Volcano Model** | A query execution model where each operator produces one row at a time via an iterator interface |
| **MVCC** | Multi-Version Concurrency Control — allows multiple transactions to read consistent snapshots without blocking writers |
