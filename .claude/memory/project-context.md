# Silica — Project Context

## Overview
- **Type**: Production-grade embedded relational database engine
- **Language**: Zig 0.15.x (stable)
- **Inspired by**: SQLite (simplicity, embeddability, single-file format)
- **Author**: Yusa

## Current Phase: Phase 7 — Stored Functions & Triggers (Milestone 13 complete, Milestone 14 pending)

### Completed Phases
- **Phase 1**: Storage Foundation ✅ (v0.1.0)
- **Phase 2**: SQL Layer ✅ (Tokenizer, Parser, AST, Analyzer, Catalog, Planner, Optimizer, Executor, Engine)
- **Phase 3**: WAL & Basic Transactions ✅
- **Phase 4**: MVCC & Full Transactions ✅
  - Milestone 6: MVCC Core ✅ (TupleHeader, TransactionManager, Snapshot, visibility, locks)
  - Milestone 7: VACUUM & SSI ✅ (VACUUM, Auto-vacuum, Savepoints, FSM, Deadlock Detection, SSI)

### Current: Phase 5 — Advanced SQL
- **Milestone 8**: Views & CTEs ✅
  - Regular/materialized/updatable views
  - CTEs (WITH, WITH RECURSIVE)
  - Set operations (UNION/INTERSECT/EXCEPT)
  - DISTINCT/DISTINCT ON
- **Milestone 9**: Window Functions ✅
  - All ranking/value/distribution functions
  - WINDOW clause named definitions
  - 36 integration tests
- **Milestone 10**: Advanced Data Types ✅
  - [x] DATE/TIME/TIMESTAMP types
  - [x] INTERVAL type
  - [x] NUMERIC/DECIMAL fixed-point
  - [x] UUID type
  - [x] SERIAL/BIGSERIAL types
  - [x] ENUM types (CREATE TYPE AS ENUM, DROP TYPE)
  - [x] ARRAY type (constructor, subscript, ANY/ALL operators, unnest())
  - [x] DOMAIN types (CREATE DOMAIN with constraints)
  - [x] Type coercion (all types castable to/from text)

### Current: Phase 6 — JSON & Full-Text Search
- **Milestone 11**: JSON/JSONB ✅
  - [x] JSON/JSONB keywords in tokenizer
  - [x] type_json/type_jsonb in AST DataType
  - [x] .json/.jsonb in catalog ColumnType (tags 0x0D, 0x0E)
  - [x] JSON/JSONB type in parser (CREATE TABLE, CAST)
  - [x] JSON validation with RFC 8259 parsing
  - [x] JSON operators (10 operators: ->, ->>, @>, <@, ?, ?|, ?&, #>, #>>, #-)
  - [x] JSON serialization using std.json.fmt
  - [x] Integration with engine, CLI, TUI, vacuum
  - [ ] JSONB binary storage format (deferred)
  - [ ] JSON functions (jsonb_build_object, jsonb_build_array, etc.) (future)
  - [ ] GIN index for JSONB (future)

- **Milestone 12**: Full-Text Search ✅
  - [x] TSVECTOR/TSQUERY data types (tags 0x0F, 0x10)
  - [x] to_tsvector() function with stemming and stop word filtering
  - [x] to_tsquery() function with query parsing
  - [x] @@ match operator (tsvector @@ tsquery)
  - [x] ts_rank/ts_rank_cd ranking functions with normalization
  - [x] ts_headline for highlighted snippets
  - [x] Porter stemmer implementation
  - [x] English stop words (33 common words)
  - [x] Integration with engine, CLI, TUI, vacuum
  - [ ] GIN index for TSVECTOR (deferred to future)

### Current: Phase 7 — Stored Functions & Triggers
- **Milestone 13**: Stored Functions ✅
  - [x] Tokenizer (13A): Function keywords (CREATE/DROP FUNCTION, RETURNS, LANGUAGE, AS, VOLATILE/STABLE/IMMUTABLE, OR REPLACE, IF EXISTS)
  - [x] AST (13B): CreateFunctionStmt, FunctionParam, FunctionReturnType, FunctionVolatility
  - [x] Parser (13C): CREATE FUNCTION/DROP FUNCTION DDL
  - [x] Catalog (13D): Function storage with 'func:' key prefix, serialization/deserialization
  - [x] Analyzer (13E): Function signature validation
  - [x] Planner (13F): CREATE/DROP FUNCTION planning (PlanType.transaction)
  - [x] Executor (13G): evalFunctionCall with catalog parameter, parameter binding
  - [x] Engine (13H): CREATE/DROP FUNCTION DDL integration, ProjectOp catalog threading
  - [x] 14 catalog tests (all return types, volatility, parameters)
  - [x] 9 analyzer tests (signature validation)
  - [x] 10 planner tests (DDL planning)
  - [x] 1 executor test (error path: unknown function without catalog)
  - [x] 1 engine integration test (CREATE/DROP/IF EXISTS)
  - **KNOWN LIMITATIONS** (discovered during STABILIZATION 2026-03-10):
    - SQL-language functions return body text instead of executing expressions
    - Functions in WHERE/ORDER BY clauses fail (FilterOp/SortOp lack catalog)
    - NULL parameter handling incorrect (returns text instead of NULL)
    - Nested function calls don't work (return unevaluated body)
  - **REQUIRED FOR FULL EXECUTION**:
    - Proper evalFunctionCall implementation that evaluates parsed body
    - Catalog threading through FilterOp, SortOp operators
    - Type conversion from evaluated result to proper Value variant

- **Milestone 14**: Triggers ✅ (DDL complete, execution deferred)
  - [x] Tokenizer (14A): Trigger keywords (TRIGGER, BEFORE, AFTER, INSTEAD, OF, EACH, STATEMENT, OLD, NEW, ENABLE, DISABLE, TRUNCATE)
  - [x] AST (14B): CreateTriggerStmt, DropTriggerStmt, AlterTriggerStmt, TriggerTiming, TriggerEvent, TriggerLevel
  - [x] Parser (14C): CREATE TRIGGER/DROP TRIGGER/ALTER TRIGGER DDL with full syntax
  - [x] Catalog (14D): Trigger storage with 'trigger:' key prefix, serialization/deserialization (13 tests)
  - [x] Analyzer (14E): Trigger definition validation (14 tests)
  - [x] Planner (14F): CREATE/DROP/ALTER TRIGGER planning (12 tests)
  - [x] Engine (14H): CREATE/DROP/ALTER TRIGGER DDL integration (2 integration tests)
  - Tests: 7 tokenizer + 6 AST + 14 parser + 13 catalog + 14 analyzer + 12 planner + 2 engine = 68 tests
  - **BUG FIX (c25f4a0)**: Fixed DuplicateKey in OR REPLACE and ALTER operations (all 4 disabled tests re-enabled)
  - **LIMITATION**: Trigger execution NOT yet implemented — definitions stored in catalog only
  - **FUTURE WORK**: Trigger firing mechanism, OLD/NEW row references, WHEN condition evaluation, execution order

## Architecture Layers
1. Client Layer (Zig API, C FFI, Wire Protocol)
2. SQL Frontend (Tokenizer → Parser → Semantic Analyzer)
3. Query Engine (Planner → Optimizer → Executor)
4. Transaction Manager (WAL, Locks, MVCC future)
5. Storage Engine (B+Tree, Page Manager, Buffer Pool)
6. OS Layer (File I/O, mmap optional, fsync)

## Test Coverage Status (as of 2026-03-10)
- Total tests: 1618 (all passing, 0 disabled)
- tokenizer.zig: 75 tests (includes JSON/JSONB keywords + 5 JSON operators + @@ operator + 13 function keywords)
- parser.zig: 160 tests (includes JSON/JSONB, JSON operators, ANY/ALL, window, SERIAL, ENUM, DOMAIN, CREATE/DROP FUNCTION with 11 tests)
- executor.zig: 246 tests (includes JSON/JSONB cast, JSON operators with 20 tests, ANY/ALL eval, INTERVAL, TSVECTOR/TSQUERY with 16 type tests, to_tsvector/to_tsquery with 13 tests, @@ operator with 9 tests [1 disabled due to bug #1], ts_rank with 10 tests, ts_rank_cd with 7 tests, Porter stemmer with 10 tests, stop words with 4 tests, FTS integration with 3 tests, FTS edge cases with 10 tests, ts_headline with 10 tests, user-defined function error path with 1 test)
- engine.zig: 419 tests (includes JSON/JSONB CRUD, window functions, ENUM, DOMAIN, temporal types, views, CTEs, set ops, SSI, VACUUM, savepoints, unnest(), ts_rank/ts_rank_cd integration tests, ts_headline integration tests, division by zero with proper cleanup, 2 stabilization edge case tests: ORDER BY sorting, complex WHERE expressions, CREATE/DROP FUNCTION integration test, CREATE/DROP/ALTER TRIGGER integration tests with 4 tests)
- STABILIZATION session findings (2026-03-10 20:00 UTC):
  - **BUG FIX**: GitHub issue #1 resolved (DuplicateKey in OR REPLACE/ALTER operations)
  - Root cause: Catalog didn't delete old entry before insert in createFunction/createTrigger/alterTrigger
  - 4 tests re-enabled: function OR REPLACE, trigger ALTER, 2 engine integration tests
  - All 1618 tests passing (1569 main + 49 sailor)
  - Milestone 13 SQL function execution limitations documented in engine.zig
  - Comprehensive error handling verification (1165 defer cleanup sites, proper errdefer patterns)

## Performance Targets
- Point lookup (PK, cached): < 5 µs
- Sequential insert: > 100K rows/sec
- Range scan: > 500K rows/sec
- DB open: < 10 ms (1 GB)
- Binary size: < 2 MB
- Memory idle: < 1 MB + cache

## Key File Format
- Page size: 4096 bytes (default, configurable 512-65536)
- Magic bytes: "SLCA"
- Single-file database
- Page types: header (0x01), internal (0x02), leaf (0x03), overflow (0x04), free (0x05)
- Page header: 16 bytes (type, flags, cell_count, page_id, free_offset, checksum)
- DB header: 64 bytes (magic, version, page_size, page_count, freelist_head, schema_version, wal_mode)

## Implemented Files
- `build.zig` — Build system (Zig 0.15 API, library + CLI targets, sailor dep)
- `build.zig.zon` — Package metadata (with sailor dependency)
- `src/main.zig` — Entry point with module imports
- `src/cli.zig` — CLI entry point (sailor.arg, color, fmt integration)
- `src/tui.zig` — TUI database browser (sailor.tui)
- `src/util/checksum.zig` — CRC32C using std.hash.crc.Crc32Iscsi
- `src/util/varint.zig` — LEB128 unsigned varint encode/decode
- `src/storage/page.zig` — Pager with header, read/write, freelist
- `src/storage/buffer_pool.zig` — LRU buffer pool with pin/unpin, dirty tracking, WAL integration
- `src/storage/btree.zig` — B+Tree with slotted-page layout, splits, merges, cursor, overflow
- `src/storage/overflow.zig` — Overflow page chain management
- `src/storage/fuzz.zig` — B+Tree fuzz tests
- `src/sql/tokenizer.zig` — Hand-written SQL lexer
- `src/sql/ast.zig` — AST node definitions with arena allocator
- `src/sql/parser.zig` — Recursive descent parser with Pratt precedence
- `src/sql/catalog.zig` — Schema catalog (B+Tree backed)
- `src/sql/analyzer.zig` — Semantic analysis, name resolution, type checking
- `src/sql/planner.zig` — AST → logical plan tree
- `src/sql/optimizer.zig` — Rule-based plan optimization
- `src/sql/engine.zig` — Database integration layer, full SQL pipeline
- `src/tx/wal.zig` — Write-Ahead Log (frame writer, commit, checkpoint, recovery)
- `src/tx/mvcc.zig` — MVCC visibility, snapshots, TransactionManager
- `src/tx/lock.zig` — Lock manager (row-level + table-level locks, conflict detection)

## Test Summary (1084 tests total: 1035 main + 49 fuzz)
- `tokenizer.zig`: 54 | `ast.zig`: 11 | `parser.zig`: 104 | `catalog.zig`: 30
- `analyzer.zig`: 46 | `planner.zig`: 50 | `optimizer.zig`: 16 | `executor.zig`: 68
- `btree.zig`: 53 | `fuzz.zig`: 12 | `overflow.zig`: 18 | `page.zig`: 24
- `buffer_pool.zig`: 23 | `checksum.zig`: 12 | `varint.zig`: 19
- `wal.zig`: 23 | `mvcc.zig`: 69 | `lock.zig`: 50
- `vacuum.zig`: 46 | `fsm.zig`: 21
- `engine.zig`: 284 | `cli.zig`: 30 | `tui.zig`: 18

## Current Phase: Phase 2 — SQL Layer + Phase 3 — Transactions

### Milestone 3 — Tokenizer & Parser
- [x] Tokenizer (3A) — hand-written lexer, SQL keyword recognition (53 tests)
- [x] Parser (3B) — recursive descent → AST (78 tests)
- [x] DDL statements (3C) — included in 3B
- [x] DML statements (3D) — included in 3B
- [ ] Parser error recovery (3E)

### Milestone 4 — Semantic Analysis & Execution
- [x] Schema catalog (4A) — B+Tree backed, serialization
- [x] Semantic analyzer (4B) — name resolution, type checking
- [x] Query planner + optimizer (5A) — logical plan, predicate pushdown
- [x] Volcano-model executor (5B) — all operators, expression eval
- [x] Database engine (5C) — full SQL pipeline, Database.open/exec/close
- [x] WHERE with index selection (4E) — secondary B+Tree indexes for PK columns
- [ ] JOIN execution (4F)

### Milestone 5 — WAL & Transactions (Phase 3)
- [x] WAL module (5A) — frame writer, commit, rollback, checkpoint, recovery
- [x] WAL integration with buffer pool and engine
- [ ] Read-path WAL deeper integration (5B)
- [ ] Checkpoint process (5C)
- [ ] Crash recovery tests (5D)

### Milestone 6 — MVCC Core (Phase 4)
- [x] TupleHeader, TupleFlags, Snapshot — core data structures
- [x] TransactionManager — begin/commit/abort, snapshot, CID management
- [x] isTupleVisible / isTupleVisibleWithTm — visibility rules
- [x] Versioned row format (0xAA prefix) — serialize/deserialize
- [x] Engine integration — transaction-aware exec, BEGIN/COMMIT/ROLLBACK
- [x] Aborted txn visibility fix — TM-based commit/abort status lookup
- [x] RR/SERIALIZABLE snapshot ownership fix — no double-free
- [x] Lock manager — row-level locking (shared/exclusive), table-level locks (7 modes), conflict detection
- [x] Lock-engine integration — DML acquires exclusive row locks, conflict via xmax check, released on commit/rollback

### Milestone 7 — VACUUM & SSI (Phase 4)
- [x] VACUUM — dead tuple reclamation, freeze old tuples, FSM updates
- [x] Auto-vacuum daemon — configurable thresholds (PostgreSQL-style), per-table stats, engine integration
- [x] Free Space Map (FSM) — per-page free space tracking, disk persistence
- [x] SERIALIZABLE isolation via SSI — rw-antidependency tracking, cycle detection, txn abort
- [x] Deadlock detection — wait-for graph with cycle detection
- [x] Savepoints — SAVEPOINT, ROLLBACK TO, RELEASE with CID management

### Phase 5 — Advanced SQL (In Progress)
- [x] CREATE VIEW / DROP VIEW — parsing, catalog, execution (17 tests)
- [x] CTEs (WITH) — parsing, analysis, planning, execution (11 tests)
- [x] Set operations (UNION/INTERSECT/EXCEPT) — full pipeline (16 tests)
- [x] SELECT DISTINCT / DISTINCT ON — executor with hash-based dedup (10 tests)
- [x] WITH RECURSIVE (recursive CTEs) — iterative fixed-point eval (11 tests)
- [x] Updatable views (INSERT/UPDATE/DELETE through views) — 16 tests
- [x] WITH CHECK OPTION (LOCAL/CASCADED) — enforced on INSERT/UPDATE
- [ ] Window functions (ROW_NUMBER, RANK, etc.) — Milestone 9
- [ ] Advanced data types (DATE/TIME/TIMESTAMP, NUMERIC, UUID, ARRAY, ENUM) — Milestone 10
