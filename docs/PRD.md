# Silica — Product Requirements Document

> A full-featured relational database engine written in Zig — dual-mode (embedded + client-server), MVCC concurrency, full SQL:2016, streaming replication.

**Version:** 0.2
**Date:** February 28, 2026
**Author:** Yusa

---

## 1. Overview

### 1.1 Vision

Silica is a production-grade relational database engine written entirely in Zig. It operates as both an embedded library and a standalone client-server process — two equal deployment modes from a single codebase. Silica targets the feature set of PostgreSQL-class systems: full SQL:2016 support, MVCC with concurrent writers, streaming replication, views, triggers, stored functions, window functions, CTEs, JSON, and full-text search.

By leveraging Zig's manual memory control, comptime capabilities, and C ABI compatibility, Silica delivers predictable performance with zero hidden allocations — making it suitable for everything from mobile apps to multi-tenant server deployments.

### 1.2 Why Zig?

- **No hidden allocations or control flow.** Every allocation is explicit, making it straightforward to reason about memory usage and lifetime — essential for a database engine.
- **Comptime metaprogramming.** Schema validation, wire protocol serialization, and query plan optimization hints can be partially evaluated at compile time.
- **C ABI interop with zero overhead.** The embedded library can be consumed from C, C++, Python, Node.js, Go, and any language with C FFI.
- **No garbage collector.** Eliminates GC pauses that can cause unpredictable latency spikes during transactions.
- **Cross-compilation out of the box.** A single build step can produce binaries for Linux, macOS, Windows, and WASI.
- **Safety without runtime cost.** Zig's checked arithmetic, slice bounds checking, and explicit error handling catch bugs at development time without runtime overhead in release builds.

### 1.3 Project Goals

| Priority | Goal |
|----------|------|
| P0 | Full ACID-compliant relational database with MVCC |
| P0 | Complete SQL:2016 core conformance (DDL, DML, views, CTEs, window functions, JSON) |
| P0 | Dual-mode: embedded library AND client-server with identical feature parity |
| P1 | MVCC with concurrent writers and full isolation levels |
| P1 | High performance for OLTP workloads with predictable latency |
| P1 | WAL-based streaming replication for read replicas |
| P2 | Extensibility: stored functions, triggers, virtual tables, custom types |
| P2 | Full-text search with inverted index |
| P3 | Role-based access control (RBAC) and row-level security |

---

## 2. Target Users & Use Cases

### 2.1 Primary Users

- **Application developers** who need an embedded database (mobile apps, desktop apps, CLI tools, IoT devices).
- **Backend engineers** running Silica as a primary server database for web applications and microservices.
- **Systems programmers** building infrastructure that requires a reliable, high-performance data store.
- **Zig ecosystem developers** looking for a native database library without C dependency wrappers.
- **DevOps / SRE teams** who need a lightweight, operationally simple database with replication.

### 2.2 Key Use Cases

1. **Embedded application storage** — Local-first apps that store structured data in a single file (note-taking apps, configuration stores, local caches).
2. **Primary server database** — Web applications and APIs using Silica as the main OLTP database with full SQL, transactions, and replication.
3. **Edge / IoT data collection** — Resource-constrained environments where a small binary footprint and cross-compilation matter.
4. **Testing & prototyping** — Drop-in embedded DB for integration tests without requiring a server process.
5. **Replicated read-heavy workloads** — Primary with streaming replicas for read scaling and high availability.
6. **Multi-tenant SaaS** — Schema-per-tenant isolation with connection pooling and RBAC.

---

## 3. Architecture

### 3.1 High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                       Client Layer                           │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ Zig API     │  │ C API (FFI)  │  │ Wire Protocol     │  │
│  │ (Embedded)  │  │ (Embedded)   │  │ (TCP + TLS)       │  │
│  └──────┬──────┘  └──────┬───────┘  └────────┬──────────┘  │
│         └────────────────┼─────────────────────┘            │
├──────────────────────────┼──────────────────────────────────┤
│                   Connection Manager                         │
│  ┌──────────┐  ┌─────────┴──────┐  ┌────────────────────┐  │
│  │ Auth &   │  │   Session      │  │ Connection Pool    │  │
│  │ RBAC     │  │   Manager      │  │ (server mode)      │  │
│  └──────────┘  └────────────────┘  └────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                      SQL Frontend                            │
│  ┌──────────┐  ┌────────────────┐  ┌────────────────────┐  │
│  │ Tokenizer│→ │     Parser     │→ │ Semantic Analyzer  │  │
│  │ (Lexer)  │  │ (AST Builder)  │  │ (type check, bind) │  │
│  └──────────┘  └────────────────┘  └─────────┬──────────┘  │
├──────────────────────────────────────────────┼──────────────┤
│                    Query Engine               │              │
│  ┌──────────┐  ┌────────────────┐  ┌─────────┴──────────┐  │
│  │ Query    │→ │  Optimizer     │→ │    Executor        │  │
│  │ Planner  │  │ (Cost-based)   │  │ (Volcano + JIT)    │  │
│  └──────────┘  └────────────────┘  └─────────┬──────────┘  │
├──────────────────────────────────────────────┼──────────────┤
│                  Catalog & Schema             │              │
│  ┌──────────┐  ┌────────────────┐  ┌─────────┴──────────┐  │
│  │ System   │  │    Views &     │  │ Triggers &         │  │
│  │ Tables   │  │  Materialized  │  │ Stored Functions   │  │
│  └──────────┘  └────────────────┘  └─────────┬──────────┘  │
├──────────────────────────────────────────────┼──────────────┤
│                Transaction Manager            │              │
│  ┌──────────┐  ┌────────────────┐  ┌─────────┴──────────┐  │
│  │   WAL    │  │  Lock Manager  │  │      MVCC          │  │
│  │  Writer  │  │ (Row + Table)  │  │ (Multi-version)    │  │
│  └──────────┘  └────────────────┘  └─────────┬──────────┘  │
├──────────────────────────────────────────────┼──────────────┤
│                  Storage Engine               │              │
│  ┌──────────┐  ┌────────────────┐  ┌─────────┴──────────┐  │
│  │  B+Tree  │  │  Page Manager  │  │     Buffer Pool    │  │
│  │  Index   │  │  (Pager)       │  │     (LRU + clock)  │  │
│  └──────────┘  └────────────────┘  └────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                   Replication Layer                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ WAL Sender   │  │ WAL Receiver │  │ Replication Slot │  │
│  │ (Primary)    │  │ (Replica)    │  │ Manager          │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                       OS Layer                               │
│  ┌──────────┐  ┌────────────────┐  ┌────────────────────┐  │
│  │  File    │  │  io_uring /    │  │  fsync / fdatasync │  │
│  │  I/O     │  │  epoll / kqueue│  │  Control           │  │
│  └──────────┘  └────────────────┘  └────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Component Responsibilities

#### 3.2.1 Storage Engine

The storage engine manages how data is physically stored on disk and retrieved into memory.

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
- Supports partial indexes and expression indexes.

**Buffer Pool**
- LRU-based page cache with configurable size (default: 2000 pages ≈ 8 MB).
- Dirty page tracking for write-back.
- Pin/unpin semantics to prevent eviction of pages in active use.
- Clock sweep eviction as alternative to pure LRU for better concurrent performance.
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
│  - MVCC epoch                   │
│  - Replication LSN              │
│  - Reserved region              │
├─────────────────────────────────┤
│  Page 1: Schema Table Root      │
│  (B+Tree root for system        │
│   catalog tables)               │
├─────────────────────────────────┤
│  Page 2..N: Data & Index Pages  │
└─────────────────────────────────┘
```

#### 3.2.2 SQL Frontend

**Tokenizer**
- Hand-written lexer (no external parser generators).
- Token types: keywords (full SQL:2016 keyword set), identifiers, literals (integer, float, string, blob, date/time), operators, punctuation.
- Support for quoted identifiers (`"column"`), dollar-quoted strings (`$$body$$`), and string escaping.
- Unicode identifier support.

**Parser**
- Recursive descent parser with Pratt precedence for expressions, producing a typed AST.
- Full statement support:

| Category | Statements |
|----------|-----------|
| DDL | `CREATE/ALTER/DROP TABLE`, `CREATE/DROP INDEX`, `CREATE/DROP VIEW`, `CREATE/DROP SCHEMA` |
| DML | `SELECT`, `INSERT`, `UPDATE`, `DELETE`, `MERGE`, `UPSERT` (INSERT ON CONFLICT) |
| Transaction | `BEGIN`, `COMMIT`, `ROLLBACK`, `SAVEPOINT`, `RELEASE SAVEPOINT` |
| DCL | `GRANT`, `REVOKE`, `CREATE/DROP ROLE` |
| Functions | `CREATE/DROP FUNCTION`, `CREATE/DROP TRIGGER` |
| Utility | `EXPLAIN`, `ANALYZE`, `VACUUM`, `REINDEX`, `SET`, `SHOW` |
| Replication | `CREATE/DROP PUBLICATION`, `CREATE/DROP SUBSCRIPTION` |

- SELECT full support:
  - `WHERE` clauses with all comparison operators, `AND`/`OR`/`NOT`, `IN`, `BETWEEN`, `IS NULL`/`IS NOT NULL`
  - `ORDER BY` (ASC/DESC, NULLS FIRST/LAST), `LIMIT`, `OFFSET`, `FETCH FIRST`
  - `GROUP BY`, `HAVING`, `GROUP BY ROLLUP/CUBE/GROUPING SETS`
  - All aggregate functions: `COUNT`, `SUM`, `AVG`, `MIN`, `MAX`, `STRING_AGG`, `ARRAY_AGG`, `BOOL_AND`, `BOOL_OR`
  - All JOIN types: `INNER`, `LEFT`, `RIGHT`, `FULL OUTER`, `CROSS`, `LATERAL`, `NATURAL`
  - Subqueries in `WHERE`, `FROM`, `SELECT`, `HAVING` (correlated and uncorrelated)
  - `DISTINCT`, `DISTINCT ON`
  - Set operations: `UNION`, `UNION ALL`, `INTERSECT`, `EXCEPT`
  - `WITH` (CTE) and `WITH RECURSIVE`
  - Window functions: `ROW_NUMBER`, `RANK`, `DENSE_RANK`, `LAG`, `LEAD`, `FIRST_VALUE`, `LAST_VALUE`, `NTH_VALUE`, `NTILE`, `PERCENT_RANK`, `CUME_DIST` with `OVER (PARTITION BY ... ORDER BY ... ROWS/RANGE BETWEEN ...)`
  - `CASE WHEN ... THEN ... ELSE ... END`
  - `CAST`, `::` type cast operator
  - `EXISTS`, `ANY`, `ALL`, `SOME`
  - `LIKE`, `ILIKE`, `SIMILAR TO`, regex match (`~`, `~*`)
  - `COALESCE`, `NULLIF`, `GREATEST`, `LEAST`
  - Array constructors and subscript access
  - JSON operators (`->`, `->>`, `@>`, `<@`, `?`, `?|`, `?&`, `#>`, `#>>`)

**Semantic Analyzer**
- Type checking and name resolution against the system catalog.
- Validates column references, table existence, and type compatibility.
- Resolves `*` expansion and implicit column references.
- View expansion (inline view definitions into the query tree).
- CTE scope resolution.
- Function overload resolution.
- Implicit type coercion rules.

#### 3.2.3 Query Engine

**Query Planner**
- Translates the validated AST into a logical plan (relational algebra tree).
- Logical operators: Scan, Filter, Project, Join, Sort, Aggregate, Limit, WindowAgg, SetOp, CTE, Values, Materialize.
- Prepared statement plan caching.

**Optimizer (Cost-Based)**
- Statistics collection via `ANALYZE` (histograms, distinct counts, null fractions, correlation).
- Cost model: estimates I/O cost + CPU cost for each operator.
- Transformations:
  - Predicate pushdown (push filters below joins).
  - Index selection based on selectivity estimates.
  - Join order optimization: dynamic programming for ≤ 8 tables, greedy heuristic for more.
  - Join algorithm selection: nested loop, hash join, merge join.
  - Subquery decorrelation (convert correlated subqueries to joins where possible).
  - Constant folding and expression simplification.
  - Common subexpression elimination.
  - Materialized view matching (when applicable).
  - Partition pruning (when partitioning is implemented).

**Executor**
- Volcano-model (iterator-based, pull) execution.
- Each operator implements `open()`, `next()`, `close()`.
- Produces `Row` tuples that flow up the operator tree.
- Physical operators: SeqScan, IndexScan, IndexOnlyScan, BitmapScan, NestedLoopJoin, HashJoin, MergeJoin, HashAggregate, SortAggregate, WindowAgg, Sort (in-memory + external merge), Limit, Materialize, CTE Scan, Values, Insert, Update, Delete.
- Memory-aware: external merge sort for large `ORDER BY`, hash-based grouping with spill-to-disk.
- Expression evaluation with compiled (comptime-optimized) paths for common patterns.

#### 3.2.4 Catalog & Schema

**System Tables**
- `silica_tables` — all tables and their metadata.
- `silica_columns` — column definitions, types, constraints, defaults.
- `silica_indexes` — index definitions and root page pointers.
- `silica_views` — view definitions (stored SQL).
- `silica_functions` — stored function definitions.
- `silica_triggers` — trigger definitions and binding.
- `silica_sequences` — sequence generators.
- `silica_constraints` — CHECK, FK, UNIQUE constraint definitions.
- `silica_statistics` — table and column statistics for the optimizer.
- `silica_roles` — roles and permissions.
- `silica_schemas` — schema (namespace) definitions.

**Views**
- Regular views: stored query definitions, expanded inline during planning.
- Materialized views: pre-computed result sets stored as tables, refreshable via `REFRESH MATERIALIZED VIEW`.
- Updatable views: simple views (single table, no aggregates) support INSERT/UPDATE/DELETE.
- `WITH CHECK OPTION` for view-based data validation.

**Triggers**
- Row-level and statement-level triggers.
- `BEFORE`, `AFTER`, `INSTEAD OF` timing.
- Trigger functions written in Silica's stored function language.
- `OLD` and `NEW` row references in row-level triggers.
- Trigger conditions via `WHEN` clause.

**Stored Functions**
- Silica Function Language (SFL): a simple imperative language for server-side logic.
  - Variables, control flow (`IF`/`ELSE`, `LOOP`, `WHILE`, `FOR`), `RETURN`.
  - SQL statement execution within functions.
  - Exception handling (`BEGIN ... EXCEPTION WHEN ... THEN ... END`).
- Scalar functions (return a single value) and set-returning functions (return rows).
- `IMMUTABLE`, `STABLE`, `VOLATILE` function volatility categories for optimizer hints.

#### 3.2.5 Transaction Manager

**WAL (Write-Ahead Log)**
- All modifications are written to a WAL file before being applied to the main database.
- WAL format: sequence of frames, each containing page number + page data + checksum + transaction ID.
- Checkpoint process merges WAL frames back into the main database file.
- Supports `NORMAL`, `FULL`, and `OFF` sync modes.
- WAL serves double duty as replication stream source.

**MVCC (Multi-Version Concurrency Control)**
- Every row version is tagged with `(xmin, xmax)` — the transaction IDs that created and deleted it.
- Readers never block writers; writers never block readers.
- Concurrent writers: row-level conflict detection. Two transactions modifying the same row → second one waits or aborts depending on isolation level.
- Visibility rules per isolation level determine which row versions a transaction can see.
- Garbage collection (VACUUM) reclaims dead row versions no longer visible to any active transaction.

**Isolation Levels**
| Level | Behavior |
|-------|----------|
| `READ UNCOMMITTED` | See uncommitted changes from other transactions (dirty reads allowed). Silica treats this the same as READ COMMITTED. |
| `READ COMMITTED` (default) | Each statement sees a fresh snapshot. No dirty reads. Possible non-repeatable reads and phantom reads. |
| `REPEATABLE READ` | Transaction sees a single snapshot taken at the start. No dirty or non-repeatable reads. Phantom reads prevented via predicate locking. |
| `SERIALIZABLE` | Full serializability via SSI (Serializable Snapshot Isolation). Detects and aborts transactions that would violate serial execution order. |

**Lock Manager**
- Row-level locks: shared (FOR SHARE) and exclusive (FOR UPDATE).
- Table-level locks: ACCESS SHARE, ROW SHARE, ROW EXCLUSIVE, SHARE, SHARE ROW EXCLUSIVE, EXCLUSIVE, ACCESS EXCLUSIVE.
- Deadlock detection via wait-for graph cycle detection (background thread, configurable timeout).
- Advisory locks for application-level coordination.

**Savepoints**
- `SAVEPOINT name` — create a named savepoint within a transaction.
- `ROLLBACK TO SAVEPOINT name` — undo changes since the savepoint.
- `RELEASE SAVEPOINT name` — discard a savepoint (merge into parent transaction).
- Nested savepoints supported.

#### 3.2.6 Client-Server Mode

**Wire Protocol**
- PostgreSQL-compatible wire protocol (v3) for broad client library compatibility.
  - Silica can be used as a drop-in replacement with existing PostgreSQL client drivers (psycopg2, pg, node-postgres, etc.).
- Message types: StartupMessage, AuthenticationOk, Query, Parse, Bind, Execute, Describe, Close, Sync, ErrorResponse, RowDescription, DataRow, CommandComplete, ReadyForQuery.
- Extended query protocol: prepared statements with separate parse/bind/execute phases.
- TLS support via system libraries.
- COPY protocol for bulk data import/export.

**Connection Manager**
- Async I/O event loop (io_uring on Linux, kqueue on macOS, IOCP on Windows).
- Connection pool with configurable max connections (default: 100).
- Per-connection session state: current schema, transaction state, prepared statements, runtime parameters.
- Graceful shutdown: drain active connections, wait for in-flight transactions.

**Authentication & Authorization**
- Password-based: SCRAM-SHA-256 (preferred), MD5 (legacy compatibility).
- Certificate-based: mutual TLS client certificates.
- `pg_hba.conf`-style host-based access control file.
- Role-based access control:
  - `CREATE ROLE`, `DROP ROLE`, `ALTER ROLE`.
  - `GRANT`/`REVOKE` on tables, schemas, functions, sequences.
  - Role inheritance and membership (`GRANT role TO role`).
- Row-level security (RLS):
  - `CREATE POLICY` with `USING` (read filter) and `WITH CHECK` (write filter) expressions.
  - `ALTER TABLE ... ENABLE ROW LEVEL SECURITY`.

#### 3.2.7 Replication

**WAL-Based Streaming Replication**
- Primary streams WAL records to replicas in real-time over a persistent TCP connection.
- Replication slots: track each replica's consumption position, prevent WAL recycling before replica has caught up.
- Synchronous replication (optional): primary waits for at least N replicas to confirm write before committing.
- Asynchronous replication (default): primary does not wait; replicas lag by network + apply time.

**Replica Capabilities**
- Hot standby: replicas accept read-only queries while applying WAL.
- Replica promotion: `SELECT pg_promote()` or external trigger to promote a replica to primary.
- Cascading replication: replicas can feed WAL to other replicas.
- Replication lag monitoring: `pg_stat_replication` equivalent system view.

**Replication Protocol**
- Replication connection handshake: `START_REPLICATION SLOT slot_name LOGICAL|PHYSICAL LSN lsn`.
- Keepalive messages to detect broken connections.
- Feedback messages from replica to primary (flush position, apply position).

---

## 4. Data Types

### 4.1 Core Types

| SQL Type | Internal Representation | Size |
|----------|------------------------|------|
| `SMALLINT` | i16 | 2 bytes |
| `INTEGER` | i32 | 4 bytes |
| `BIGINT` | i64 | 8 bytes |
| `SERIAL` / `BIGSERIAL` | i32 / i64 + auto-increment sequence | 4 / 8 bytes |
| `REAL` | f32 | 4 bytes |
| `DOUBLE PRECISION` | f64 | 8 bytes |
| `NUMERIC(p,s)` / `DECIMAL(p,s)` | Fixed-point BCD or 128-bit integer | variable |
| `BOOLEAN` | bool | 1 byte |
| `TEXT` | UTF-8 `[]const u8` | variable |
| `VARCHAR(n)` | UTF-8 `[]const u8` with length constraint | variable |
| `CHAR(n)` | UTF-8 `[]const u8`, space-padded | variable |
| `BYTEA` / `BLOB` | Raw `[]const u8` | variable |
| `DATE` | i32 (days since 2000-01-01) | 4 bytes |
| `TIME` | i64 (microseconds since midnight) | 8 bytes |
| `TIMESTAMP` | i64 (microseconds since 2000-01-01 00:00:00) | 8 bytes |
| `TIMESTAMP WITH TIME ZONE` | i64 (microseconds, UTC-normalized) | 8 bytes |
| `INTERVAL` | Composite: months (i32) + days (i32) + microseconds (i64) | 16 bytes |
| `UUID` | u128 | 16 bytes |
| `JSON` | UTF-8 text (validated on input) | variable |
| `JSONB` | Binary decomposed JSON (tree structure) | variable |
| `ARRAY` | Typed array of any base type | variable |
| `NULL` | Tag only | 0 bytes |

### 4.2 Type System

- **Strict typing** with implicit coercions for safe conversions (e.g., INTEGER → BIGINT, REAL → DOUBLE PRECISION).
- **Explicit CAST** required for potentially lossy conversions (e.g., DOUBLE → INTEGER, TEXT → INTEGER).
- **Domain types**: `CREATE DOMAIN` for user-defined type aliases with constraints.
- **Enum types**: `CREATE TYPE name AS ENUM ('val1', 'val2', ...)`.
- **Composite types**: `CREATE TYPE name AS (field1 type1, field2 type2, ...)`.

---

## 5. Constraints & Integrity

| Constraint | Description |
|-----------|-------------|
| `PRIMARY KEY` | Single or composite. Unique, NOT NULL, clustered index. |
| `NOT NULL` | Column must have a value. |
| `UNIQUE` | Enforced via unique index. Supports partial unique constraints (`WHERE` clause). |
| `DEFAULT` | Column default values (constants, expressions, function calls like `NOW()`). |
| `CHECK` | Arbitrary boolean expressions validated on INSERT/UPDATE. |
| `FOREIGN KEY` | References another table's PRIMARY KEY or UNIQUE columns. Actions: `CASCADE`, `SET NULL`, `SET DEFAULT`, `RESTRICT`, `NO ACTION`. Supports `ON DELETE` and `ON UPDATE`. |
| `EXCLUSION` | Ensures no two rows satisfy a given predicate (e.g., range overlap). Uses GiST index. |
| `GENERATED` | `GENERATED ALWAYS AS (expr) STORED` — computed columns stored on disk. |

---

## 6. Index Types

| Index Type | Structure | Use Case |
|-----------|-----------|----------|
| B+Tree (default) | Balanced tree with sorted keys | Equality, range, ordering, prefix matching |
| Hash | Hash table | Equality-only lookups (fast point queries) |
| GiST | Generalized Search Tree | Geometric data, full-text search, range types, exclusion constraints |
| GIN | Generalized Inverted Index | Full-text search, JSONB containment, array element search |

**Index Features:**
- Partial indexes: `CREATE INDEX ... WHERE condition` — index only rows matching the condition.
- Expression indexes: `CREATE INDEX ... ON table (expression)` — index computed values.
- Covering indexes (INCLUDE): `CREATE INDEX ... ON table (col1) INCLUDE (col2, col3)` — index-only scans.
- Concurrent index creation: `CREATE INDEX CONCURRENTLY` — builds index without blocking writes.
- Multi-column indexes with independent sort directions per column.

---

## 7. Full-Text Search

- `TSVECTOR` type: processed document representation (lexemes with positional info).
- `TSQUERY` type: search query representation (lexemes with boolean operators).
- `to_tsvector(config, text)` and `to_tsquery(config, text)` functions.
- Match operator: `tsvector @@ tsquery`.
- Ranking functions: `ts_rank`, `ts_rank_cd`.
- Configurable text search configurations: dictionaries, stemmers, stop words.
- GIN index support for fast full-text search.
- Headline generation: `ts_headline` for result snippet display.

---

## 8. JSON Support

- `JSON` type: stores text, validates on input.
- `JSONB` type: binary decomposed storage, supports indexing and efficient querying.
- Operators:
  - `->` (get element by key/index, returns JSON)
  - `->>` (get element as text)
  - `#>` (get nested element by path)
  - `#>>` (get nested element as text)
  - `@>` (contains), `<@` (contained by)
  - `?` (key exists), `?|` (any key exists), `?&` (all keys exist)
  - `||` (concatenation)
  - `-` (delete key), `#-` (delete path)
- Functions: `jsonb_build_object`, `jsonb_build_array`, `jsonb_each`, `jsonb_array_elements`, `jsonb_set`, `jsonb_insert`, `jsonb_typeof`, `jsonb_strip_nulls`, `jsonb_path_query` (SQL/JSON path).
- GIN index on JSONB for `@>`, `?`, `?|`, `?&` operators.
- SQL/JSON standard path language support.

---

## 9. API Design

### 9.1 Zig Embedded API

```zig
const silica = @import("silica");

// Open or create a database
var db = try silica.open("app.db", .{
    .page_size = 4096,
    .cache_size = 2000,
    .wal_mode = true,
    .default_isolation = .read_committed,
});
defer db.close();

// Execute DDL
try db.exec("CREATE TABLE users (id SERIAL PRIMARY KEY, name TEXT NOT NULL, email TEXT UNIQUE, data JSONB)");

// Transaction with explicit isolation level
var tx = try db.begin(.{ .isolation = .serializable });
errdefer tx.rollback();

// Prepared statements with parameter binding
var stmt = try tx.prepare("SELECT id, name, data->>'age' FROM users WHERE email = $1");
defer stmt.deinit();

var rows = try stmt.query(.{"alice@example.com"});
while (try rows.next()) |row| {
    const id = row.get(i32, 0);
    const name = row.get([]const u8, 1);
    const age = row.getOptional([]const u8, 2); // nullable
    _ = .{ id, name, age };
}

try tx.commit();
```

### 9.2 C API (FFI)

```c
silica_db *db;
int rc = silica_open("app.db", &db, NULL);

silica_tx *tx;
rc = silica_begin(db, SILICA_ISO_READ_COMMITTED, &tx);

silica_stmt *stmt;
rc = silica_prepare(tx, "SELECT * FROM users WHERE id = $1", -1, &stmt);
silica_bind_int32(stmt, 1, 42);

while (silica_step(stmt) == SILICA_ROW) {
    const char *name = silica_column_text(stmt, 1);
    int name_len = silica_column_bytes(stmt, 1);
    // ...
}

silica_finalize(stmt);
silica_commit(tx);
silica_close(db);
```

### 9.3 Server Mode CLI

```bash
# Start server
silica server --data-dir /var/lib/silica --port 5433 --max-connections 200

# Connect with any PostgreSQL-compatible client
psql -h localhost -p 5433 -U admin -d mydb

# Or use the built-in CLI
silica connect --host localhost --port 5433 --user admin --db mydb

# Embedded mode (single-file, no server)
silica shell app.db
silica --tui app.db
```

---

## 10. Non-Functional Requirements

### 10.1 Performance Targets

| Metric | Target | Benchmark |
|--------|--------|-----------|
| Point lookup (by PK, cached) | < 5 us | 1M random reads |
| Sequential insert | > 150K rows/sec | 1M row bulk insert |
| Range scan throughput | > 500K rows/sec | 100-byte rows, cached |
| TPC-C throughput (server mode) | > 10K tpmC | 10 warehouses, 50 connections |
| Database open time | < 10 ms | 1 GB database |
| Concurrent transactions | > 500 active | Mixed OLTP workload |
| Replication lag (async) | < 100 ms (p99) | Sustained 10K writes/sec |
| Binary size (embedded, stripped) | < 4 MB | Release build |
| Binary size (server, stripped) | < 8 MB | Release build with replication |
| Memory overhead (idle) | < 2 MB + cache | Embedded mode |

### 10.2 Reliability

- Crash recovery: database must be recoverable after power loss or process kill at any point during a transaction.
- Fuzz testing: tokenizer, parser, B+Tree operations, wire protocol must pass 72+ hours of continuous fuzzing with no crashes.
- No undefined behavior: zero `@intCast` / `@ptrCast` UB; all edge cases handled explicitly.
- Replication correctness: replica state must be byte-identical to primary after applying all WAL records.
- MVCC correctness: no anomalies at any isolation level (verified by jepsen-style testing).

### 10.3 Compatibility

- Zig version: track latest stable (currently 0.15.x).
- OS support: Linux (primary), macOS, Windows. WASI as stretch goal.
- No external C dependencies for the core engine.
- PostgreSQL wire protocol v3 compatibility: standard psql, psycopg2, node-pg, pgx clients must connect and execute queries.
- SQL:2016 core conformance (with documented deviations).

---

## 11. Milestone Plan

### Phase 1: Storage Foundation ✅

**Milestone 1 — Page Manager & File Format**
- [x] Database file header (Page 0) with magic, version, page size
- [x] Page read/write with CRC32C checksums
- [x] Freelist management (allocate/free pages)
- [x] Test suite: create DB, write pages, reopen and verify

**Milestone 2 — B+Tree & Buffer Pool**
- [x] B+Tree insert, delete, point lookup
- [x] Leaf page splits and merges
- [x] Range scan cursors (forward/backward)
- [x] Overflow pages for large values
- [x] LRU buffer pool with dirty page tracking
- [x] Comprehensive B+Tree fuzz tests

### Phase 2: SQL Core ✅

**Milestone 3 — Tokenizer & Parser**
- [x] Hand-written tokenizer with full SQL token coverage
- [x] Recursive descent parser with Pratt precedence → typed AST
- [x] DDL: `CREATE TABLE`, `DROP TABLE`, `CREATE INDEX`
- [x] DML: `SELECT`, `INSERT`, `UPDATE`, `DELETE`
- [x] Expressions: arithmetic, comparison, boolean, CASE, CAST, LIKE, IN, BETWEEN, IS NULL
- [x] JOINs: INNER, LEFT, RIGHT, FULL, CROSS
- [x] Subqueries, GROUP BY, HAVING, ORDER BY, LIMIT/OFFSET
- [x] Parser error recovery with meaningful error messages

**Milestone 4 — Semantic Analysis & Basic Execution**
- [x] Schema catalog (B+Tree backed on page 1)
- [x] Name resolution and type checking
- [x] Query planner: AST → logical plan
- [x] Rule-based optimizer: predicate pushdown, constant folding, index selection
- [x] Volcano-model executor: Scan, Filter, Project, Sort, Limit, Aggregate, Join
- [x] `WHERE` clause evaluation with index selection
- [x] Database engine: full SQL pipeline integration

### Phase 3: WAL & Basic Transactions ✅

**Milestone 5 — WAL & Crash Recovery**
- [x] WAL file format and frame writer
- [x] WAL commit, rollback, checkpoint
- [x] Read-path WAL integration (check WAL before main DB)
- [x] Buffer pool WAL routing
- [x] Crash recovery: scan frames, validate checksums, promote committed transactions
- [x] Engine WAL mode integration

### Phase 4: MVCC & Full Transactions

**Milestone 6 — MVCC Core**
- [ ] Tuple header: add `xmin`, `xmax`, `cid`, `ctid` fields to row format
- [ ] Transaction ID allocation and management (XID counter, wraparound handling)
- [ ] Visibility map: per-tuple visibility check based on transaction snapshot
- [ ] Snapshot isolation: capture active transaction set at statement/transaction start
- [ ] READ COMMITTED isolation: per-statement snapshots
- [ ] REPEATABLE READ isolation: per-transaction snapshots
- [ ] Row-level locking: `SELECT ... FOR UPDATE`, `SELECT ... FOR SHARE`
- [ ] Concurrent writer conflict detection: wait or abort on row conflict
- [ ] Dead tuple tracking for vacuum

**Milestone 7 — VACUUM & SSI**
- [ ] VACUUM: reclaim dead row versions, update visibility map, update free space map
- [ ] Auto-vacuum daemon (background process, configurable thresholds)
- [ ] Free space map (FSM): track available space per page for efficient inserts
- [ ] SERIALIZABLE isolation via SSI (Serializable Snapshot Isolation)
  - Read/write dependency tracking (rw-antidependencies)
  - Cycle detection in serialization graph
  - Transaction abort on serialization failure
- [ ] Deadlock detection via wait-for graph
- [ ] Savepoints: `SAVEPOINT`, `ROLLBACK TO`, `RELEASE`

### Phase 5: Advanced SQL

**Milestone 8 — Views, CTEs, Set Operations**
- [x] `CREATE VIEW` / `DROP VIEW` — stored query definitions
- [x] View expansion in query planner
- [ ] Updatable views (single-table, no aggregates)
- [ ] `WITH CHECK OPTION`
- [x] Common Table Expressions (`WITH ... AS`)
- [ ] Recursive CTEs (`WITH RECURSIVE`)
- [x] Set operations: `UNION`, `UNION ALL`, `INTERSECT`, `EXCEPT`
- [x] `DISTINCT ON` support

**Milestone 9 — Window Functions**
- [ ] `OVER (PARTITION BY ... ORDER BY ...)` clause parsing and planning
- [ ] Window frame specification: `ROWS`, `RANGE`, `GROUPS` with `BETWEEN ... AND ...`
- [ ] Ranking functions: `ROW_NUMBER`, `RANK`, `DENSE_RANK`, `NTILE`
- [ ] Value functions: `LAG`, `LEAD`, `FIRST_VALUE`, `LAST_VALUE`, `NTH_VALUE`
- [ ] Distribution functions: `PERCENT_RANK`, `CUME_DIST`
- [ ] Aggregate functions as window functions (e.g., `SUM(...) OVER (...)`)
- [ ] Multiple window definitions in a single query
- [ ] `WINDOW` clause for named window definitions
- [ ] WindowAgg executor operator with partition tracking

**Milestone 10 — Advanced Data Types**
- [ ] `DATE`, `TIME`, `TIMESTAMP`, `TIMESTAMP WITH TIME ZONE` types with arithmetic
- [ ] `INTERVAL` type with date/time arithmetic
- [ ] `NUMERIC(p,s)` / `DECIMAL` fixed-point type
- [ ] `UUID` type with `gen_random_uuid()` function
- [ ] `SERIAL` / `BIGSERIAL` via implicit sequence creation
- [ ] `ARRAY` type: construction, subscript access, `ANY`/`ALL` with arrays, `unnest()`
- [ ] `ENUM` types: `CREATE TYPE ... AS ENUM`
- [ ] Domain types: `CREATE DOMAIN` with constraints
- [ ] Type coercion rules and implicit/explicit cast matrix

### Phase 6: JSON & Full-Text Search

**Milestone 11 — JSON/JSONB**
- [ ] `JSON` type: text storage with validation
- [ ] `JSONB` type: binary decomposed storage format
- [ ] JSON operators: `->`, `->>`, `#>`, `#>>`, `@>`, `<@`, `?`, `?|`, `?&`, `||`, `-`, `#-`
- [ ] JSON functions: `jsonb_build_object`, `jsonb_build_array`, `jsonb_each`, `jsonb_array_elements`, `jsonb_set`, `jsonb_typeof`, `jsonb_strip_nulls`, `jsonb_path_query`
- [ ] GIN index support for JSONB containment and existence queries
- [ ] SQL/JSON path language (basic support)

**Milestone 12 — Full-Text Search**
- [ ] `TSVECTOR` and `TSQUERY` types
- [ ] `to_tsvector(config, text)` and `to_tsquery(config, text)` functions
- [ ] Match operator `@@` with GIN index acceleration
- [ ] Ranking: `ts_rank`, `ts_rank_cd`
- [ ] Text search configurations: default English dictionary, stemming, stop words
- [ ] `ts_headline` for search result snippets
- [ ] GIN index for full-text search

### Phase 7: Stored Functions & Triggers

**Milestone 13 — Stored Functions**
- [ ] `CREATE FUNCTION ... LANGUAGE sfl` — Silica Function Language
- [ ] SFL: variables, assignment, IF/ELSE, LOOP, WHILE, FOR, RETURN
- [ ] SFL: execute SQL statements within functions
- [ ] SFL: exception handling (BEGIN ... EXCEPTION WHEN ... THEN ... END)
- [ ] Scalar functions and set-returning functions (RETURNS TABLE/SETOF)
- [ ] Function volatility categories: IMMUTABLE, STABLE, VOLATILE
- [ ] Function overloading by argument types
- [ ] Built-in function library: string, math, date/time, array, type conversion

**Milestone 14 — Triggers**
- [ ] `CREATE TRIGGER` — row-level and statement-level
- [ ] `BEFORE`, `AFTER`, `INSTEAD OF` timing
- [ ] `OLD` and `NEW` row references
- [ ] Trigger conditions: `WHEN (condition)`
- [ ] `INSERT`, `UPDATE`, `DELETE`, `TRUNCATE` events
- [ ] `UPDATE OF column_list` — trigger only on specific column changes
- [ ] Trigger execution order (alphabetical by name)
- [ ] `DROP TRIGGER`, `ALTER TRIGGER ... ENABLE/DISABLE`

### Phase 8: Client-Server & Wire Protocol

**Milestone 15 — Wire Protocol**
- [ ] PostgreSQL wire protocol v3 implementation
- [ ] Simple query protocol: Query → RowDescription → DataRow* → CommandComplete → ReadyForQuery
- [ ] Extended query protocol: Parse → Bind → Describe → Execute → Close → Sync
- [ ] Prepared statement management (named and unnamed)
- [ ] Parameter binding with type OIDs
- [ ] Error and notice message formatting (SQLSTATE codes)
- [ ] COPY protocol (COPY TO/FROM with CSV, binary formats)
- [ ] SSL/TLS negotiation

**Milestone 16 — Server & Connection Management**
- [ ] TCP server with async I/O event loop (io_uring / kqueue / IOCP)
- [ ] Connection manager with configurable max connections
- [ ] Per-connection session state (schema search path, transaction state, runtime parameters)
- [ ] Authentication: SCRAM-SHA-256, MD5, trust
- [ ] `pg_hba.conf`-style host-based access control
- [ ] Graceful shutdown (drain connections, wait for transactions)
- [ ] `silica server` CLI with configuration file support
- [ ] Logging: structured log output (JSON), configurable log levels

**Milestone 17 — Authorization (RBAC)**
- [ ] `CREATE ROLE` / `DROP ROLE` / `ALTER ROLE`
- [ ] `GRANT` / `REVOKE` on tables, schemas, functions, sequences
- [ ] Role inheritance and membership
- [ ] Row-level security: `CREATE POLICY`, `ENABLE ROW LEVEL SECURITY`
- [ ] `information_schema` views for introspection
- [ ] Default privileges for new objects

### Phase 9: Streaming Replication

**Milestone 18 — WAL Sender & Receiver**
- [ ] WAL sender process on primary: streams WAL records over TCP
- [ ] WAL receiver process on replica: receives and applies WAL records
- [ ] Replication slots: track consumer position, prevent WAL recycling
- [ ] Replication protocol: START_REPLICATION, keepalive, feedback messages
- [ ] Hot standby: read-only queries on replicas while applying WAL
- [ ] Conflict resolution for hot standby (cancel conflicting queries on replica)

**Milestone 19 — Replication Operations**
- [ ] Synchronous replication: configurable `synchronous_standby_names`
- [ ] Replica promotion: `SELECT silica_promote()`
- [ ] Cascading replication: replica-to-replica WAL forwarding
- [ ] Base backup: `silica_basebackup` for initial replica provisioning
- [ ] Monitoring: `silica_stat_replication` system view
- [ ] Replication lag metrics and alerting hooks
- [ ] Switchover procedure: controlled primary/replica swap

### Phase 10: Cost-Based Optimizer & Performance

**Milestone 20 — Statistics & Cost Model**
- [ ] `ANALYZE` command: collect table and column statistics
- [ ] Histograms: equi-depth histograms for value distribution
- [ ] Statistics storage: distinct count, null fraction, average width, correlation, most common values
- [ ] Cost model: I/O cost + CPU cost estimation per operator
- [ ] Selectivity estimation: equality, range, LIKE, IN, IS NULL predicates
- [ ] Join selectivity estimation
- [ ] Auto-analyze: trigger after N rows changed

**Milestone 21 — Advanced Optimization**
- [ ] Join order optimization: dynamic programming for ≤ 8 tables
- [ ] Join algorithm selection: nested loop vs hash join vs merge join
- [ ] Subquery decorrelation (correlated → join transformation)
- [ ] Common subexpression elimination
- [ ] Expression index matching
- [ ] Partial index matching
- [ ] Covering index / index-only scan selection
- [ ] `EXPLAIN ANALYZE` with runtime statistics

### Phase 11: Additional Index Types

**Milestone 22 — Hash, GiST, GIN Indexes**
- [ ] Hash index: fast equality-only lookups
- [ ] GiST index framework: pluggable for geometric, range, FTS data
- [ ] GIN index framework: inverted index for multi-valued columns (JSONB, arrays, FTS)
- [ ] `CREATE INDEX CONCURRENTLY` — non-blocking index builds
- [ ] Bitmap index scans: combine multiple index results efficiently
- [ ] Multi-column index support with mixed sort directions

### Phase 12: Production Readiness

**Milestone 23 — Operational Tools**
- [ ] `EXPLAIN` and `EXPLAIN ANALYZE` with multiple output formats (text, JSON, YAML)
- [ ] `VACUUM` (manual and auto) with progress reporting
- [ ] `REINDEX` for index rebuilding
- [ ] `pg_stat_activity` equivalent: active queries, wait events, locks
- [ ] `pg_locks` equivalent: lock monitoring
- [ ] Configuration system: `SET`/`SHOW`/`RESET` runtime parameters
- [ ] `silica.conf` configuration file with hot-reload for applicable parameters

**Milestone 24 — Testing & Certification**
- [ ] Comprehensive benchmark suite (TPC-C, TPC-H subset, custom OLTP)
- [ ] Fuzz testing campaign (72+ hours: tokenizer, parser, B+Tree, wire protocol, MVCC)
- [ ] Jepsen-style consistency testing for isolation levels and replication
- [ ] Crash injection tests: power failure at every write point
- [ ] SQL conformance test suite (adapted from PostgreSQL regression tests)
- [ ] Performance regression CI: detect slowdowns on every commit

**Milestone 25 — Documentation & Packaging**
- [ ] API reference: Zig embedded API, C FFI API
- [ ] Architecture guide: internal design document
- [ ] Getting started guide: embedded and server modes
- [ ] SQL reference: all supported statements, functions, types, operators
- [ ] Operations guide: backup, restore, replication setup, monitoring, tuning
- [ ] CI/CD pipeline: build matrix (Linux/macOS/Windows), test, fuzz, bench, release
- [ ] Package: `build.zig` for Zig consumers, C header generation, system packages (deb, rpm, brew)

---

## 12. Built-in Functions (Summary)

### String Functions
`length`, `lower`, `upper`, `trim`, `ltrim`, `rtrim`, `substring`, `position`, `replace`, `repeat`, `reverse`, `left`, `right`, `lpad`, `rpad`, `concat`, `concat_ws`, `split_part`, `regexp_match`, `regexp_replace`, `starts_with`, `encode`, `decode`, `md5`, `chr`, `ascii`, `format`

### Math Functions
`abs`, `ceil`, `floor`, `round`, `trunc`, `mod`, `power`, `sqrt`, `log`, `ln`, `exp`, `sign`, `random`, `setseed`, `pi`, `degrees`, `radians`, `greatest`, `least`

### Date/Time Functions
`now`, `current_timestamp`, `current_date`, `current_time`, `clock_timestamp`, `date_trunc`, `date_part`, `extract`, `age`, `make_date`, `make_time`, `make_timestamp`, `make_interval`, `to_char`, `to_date`, `to_timestamp`

### Aggregate Functions
`count`, `sum`, `avg`, `min`, `max`, `string_agg`, `array_agg`, `bool_and`, `bool_or`, `bit_and`, `bit_or`, `every`, `percentile_cont`, `percentile_disc`, `mode`

### System Functions
`version`, `current_user`, `current_schema`, `current_database`, `pg_table_size`, `pg_total_relation_size`, `pg_database_size`, `txid_current`, `gen_random_uuid`, `pg_advisory_lock`, `pg_try_advisory_lock`

---

## 13. Testing Strategy

### 13.1 Test Pyramid

| Level | Scope | Tool |
|-------|-------|------|
| Unit | Individual functions (B+Tree ops, tokenizer, page read/write, MVCC visibility) | Zig built-in `test` |
| Integration | Multi-component flows (SQL → parse → plan → execute → storage → MVCC) | Custom test harness |
| Fuzz | Tokenizer, parser, B+Tree, wire protocol, SQL random generation | Zig's built-in fuzzing + custom |
| Crash | WAL recovery, MVCC consistency under simulated power failure | Custom crash injector |
| Concurrency | Multi-threaded transaction isolation, deadlock detection, replication | Thread-based test harness |
| Consistency | Isolation level correctness (jepsen-style) | Custom checker |
| Compatibility | SQL conformance suites (adapted from PostgreSQL regression tests) | SQL script runner |
| Performance | Regression benchmarks on every commit | Zig `std.time` + custom bench framework |
| Replication | WAL streaming, failover, promotion, lag | Multi-process test harness |

### 13.2 Correctness Invariants

- **B+Tree**: all keys sorted; parent keys correctly partition children; leaf chain consistent; no orphan pages.
- **WAL**: after crash recovery, database matches the last committed transaction.
- **MVCC**: no dirty reads, no lost updates. Each isolation level provides its documented guarantees — no weaker.
- **Replication**: after full WAL replay, replica is byte-identical to primary.
- **Concurrency**: no torn reads; serializable transactions produce a result equivalent to some serial execution order.
- **Constraints**: no committed row violates any active constraint (PK, FK, UNIQUE, CHECK, NOT NULL).

---

## 14. Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| MVCC complexity (visibility bugs, XID wraparound) | Data corruption, phantom anomalies | High | Formal model of visibility rules; jepsen-style testing; wraparound protection (freeze old tuples) |
| B+Tree edge cases during concurrent access | Data corruption | Medium | Extensive fuzz testing; formal invariant checks; latch protocol for concurrent B+Tree access |
| WAL recovery bugs | Data loss | High (if untested) | Crash injection test suite; test every combination of failure points |
| Replication divergence | Silent data inconsistency | Medium | Checksum comparison between primary and replica; automated consistency checks |
| Wire protocol incompatibilities | Client library failures | Medium | Test against real PostgreSQL client libraries (psycopg2, node-pg, pgx, JDBC); fuzz protocol parser |
| SSI false positive aborts | Reduced throughput under serializable | Medium | Tune conflict detection granularity; benchmark abort rates |
| SQL compatibility gaps | User frustration, migration friction | Medium | Comprehensive SQL conformance test suite; document deviations clearly |
| Zig language breaking changes | Build failures | Low | Pin Zig version; CI tests on nightly for early warning |
| Scope (full RDBMS is massive) | Extended timeline | High | Strict phase gating; each phase delivers a usable increment; prioritize correctness over features |

---

## 15. Future Roadmap (Post v1.0)

These are explicitly **out of scope** for the initial release but inform architectural decisions:

- **Logical replication** — Change data capture (CDC) for selective table replication and heterogeneous targets.
- **Partitioning** — Range, list, and hash partitioning with partition pruning in the optimizer.
- **Parallel query execution** — Intra-query parallelism for scans, joins, and aggregates.
- **Columnar storage** — Hybrid row/columnar storage for analytical queries.
- **WASM target** — Compile to WebAssembly for browser-based and edge usage.
- **Extensions API** — Dynamically loadable extensions for custom types, functions, and index methods.
- **Connection pooler** — Built-in connection pooler (PgBouncer equivalent).
- **Online schema changes** — `ALTER TABLE` without exclusive locks.
- **Distributed transactions** — Two-phase commit for multi-node deployments.
- **Query result caching** — Transparent caching of query results with invalidation.

---

## 16. Success Criteria

The project is considered successful at v1.0 when:

1. **SQL conformance** — Passes a comprehensive SQL:2016 core conformance test suite with documented deviations.
2. **ACID correctness** — All four isolation levels produce correct results under concurrent load, verified by jepsen-style testing.
3. **Replication** — Streaming replication works with automatic failover; replica is always consistent with primary.
4. **Performance** — Meets or exceeds all targets in Section 10.1 on reference hardware (4-core, 16 GB RAM, NVMe SSD).
5. **Compatibility** — Standard PostgreSQL clients (psql, psycopg2, node-pg, pgx) connect and operate without modification.
6. **Reliability** — 72+ hours of fuzz testing with zero crashes; crash recovery tests pass 100%; zero known data corruption bugs.
7. **Dual-mode** — Both embedded and server modes provide identical SQL and transaction semantics.
8. **Usability** — A developer can start the server and run queries within 5 minutes; embed in a Zig project with `@import("silica")` in 2 minutes.
9. **Code quality** — Zero known undefined behavior; all public APIs documented; test coverage > 80% by line.

---

## Appendix A: Reference Projects

| Project | Relevance |
|---------|-----------|
| [PostgreSQL](https://postgresql.org) | Primary reference for SQL semantics, MVCC, wire protocol, replication, system catalog design |
| [SQLite](https://sqlite.org) | Embedded design inspiration, file format concepts, testing methodology (billions of tests) |
| [CockroachDB](https://cockroachlabs.com) | Reference for serializable isolation (SSI), modern SQL engine design |
| [DuckDB](https://duckdb.org) | Modern embedded DB — API design, vectorized execution ideas |
| [TigerBeetle](https://tigerbeetle.com) | Production Zig database — Zig-specific patterns, io_uring usage, deterministic simulation testing |
| [Neon](https://neon.tech) | PostgreSQL-compatible — reference for storage/compute separation, WAL-based replication |
| [CMU DB Course (BusTub)](https://15445.courses.cs.cmu.edu) | Educational DB — reference for component architecture, MVCC, query processing |

## Appendix B: Glossary

| Term | Definition |
|------|-----------|
| **B+Tree** | A self-balancing tree data structure optimized for disk-based storage where all values reside in leaf nodes |
| **WAL** | Write-Ahead Log — a technique where changes are written to a log before being applied to the main data file |
| **Buffer Pool** | An in-memory cache of database pages that reduces disk I/O |
| **Pager** | The component responsible for reading and writing fixed-size pages to/from disk |
| **Volcano Model** | A query execution model where each operator produces one row at a time via an iterator interface |
| **MVCC** | Multi-Version Concurrency Control — allows multiple transactions to read consistent snapshots without blocking writers |
| **SSI** | Serializable Snapshot Isolation — a technique for providing serializable isolation without traditional locking, using dependency tracking |
| **XID** | Transaction ID — a monotonically increasing integer assigned to each transaction, used for MVCC visibility checks |
| **GiST** | Generalized Search Tree — an extensible index framework that supports various data types and query predicates |
| **GIN** | Generalized Inverted Index — an index type optimized for values that contain multiple elements (arrays, JSONB, full-text) |
| **LSN** | Log Sequence Number — a position in the WAL stream, used to track replication progress |
| **TSVECTOR** | A processed representation of a text document for full-text search, containing lexemes with positional information |
| **CTE** | Common Table Expression — a named temporary result set defined within a WITH clause |
| **SFL** | Silica Function Language — the built-in procedural language for stored functions and triggers |

## Appendix C: Wire Protocol Compatibility Matrix

Silica targets compatibility with PostgreSQL wire protocol v3. The following client libraries are tested:

| Client Library | Language | Status |
|---------------|----------|--------|
| psql | CLI | Target |
| psycopg2 / psycopg3 | Python | Target |
| node-postgres (pg) | Node.js | Target |
| pgx | Go | Target |
| JDBC (PostgreSQL driver) | Java | Target |
| Npgsql | .NET | Target |
| rust-postgres / sqlx | Rust | Target |
| Zig client (native) | Zig | Built-in |
