# Silica — Project Context

## Overview
- **Type**: Production-grade embedded relational database engine
- **Language**: Zig 0.15.x (stable)
- **Inspired by**: SQLite (simplicity, embeddability, single-file format)
- **Author**: Yusa

## Current Phase: Phase 1 — Storage Foundation (Weeks 1-6)

### Milestone 1 — Page Manager & File Format (Weeks 1-2)
- [x] Utilities: CRC32C (src/util/checksum.zig), varint (src/util/varint.zig)
- [x] Database file header (Page 0) — Magic: "SLCA"
- [x] Page read/write with CRC32C checksums
- [x] Freelist management (allocate/free pages)
- [x] Basic test suite: create DB, write pages, reopen and verify

### Milestone 2 — B+Tree & Buffer Pool (Weeks 3-6)
- [x] LRU buffer pool with dirty page tracking (2A)
- [x] B+Tree insert, delete, point lookup with leaf/internal splits (2B)
- [x] Leaf/internal merges, underflow handling, root shrink (2C)
- [x] Range scan cursors (forward/backward) with seek (2D)
- [x] Overflow pages for large values (2E)
- [x] Comprehensive B+Tree fuzz tests (2F)

## Architecture Layers
1. Client Layer (Zig API, C FFI, Wire Protocol)
2. SQL Frontend (Tokenizer → Parser → Semantic Analyzer)
3. Query Engine (Planner → Optimizer → Executor)
4. Transaction Manager (WAL, Locks, MVCC future)
5. Storage Engine (B+Tree, Page Manager, Buffer Pool)
6. OS Layer (File I/O, mmap optional, fsync)

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

## Test Summary (679 tests total: 630 library + 49 CLI/TUI)
- `parser.zig`: 78 tests
- `tokenizer.zig`: 53 tests
- `ast.zig`: 10 tests
- `btree.zig`: 53 tests
- `fuzz.zig`: 12 tests
- `overflow.zig`: 18 tests
- `buffer_pool.zig`: 19 tests (incl. 4 WAL integration)
- `page.zig`: 24 tests
- `checksum.zig`: 12 tests
- `varint.zig`: 19 tests
- `catalog.zig`: 17 tests
- `analyzer.zig`: 28 tests
- `planner.zig`: 29 tests
- `optimizer.zig`: 14 tests
- `executor.zig`: 35 tests
- `engine.zig`: 124 tests (incl. 8 WAL + 14 error handling + 31 MVCC + 8 lock integration)
- `lock.zig`: 24 tests
- `wal.zig`: 15 tests
- `mvcc.zig`: 44 tests
- `cli.zig`: 31 tests
- `tui.zig`: 18 tests

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
