# Silica — Project Context

## Overview
- **Type**: Production-grade embedded relational database engine
- **Language**: Zig 0.15.x (stable)
- **Inspired by**: SQLite (simplicity, embeddability, single-file format)
- **Author**: Yusa

## Current Phase: Phase 12 — Production Readiness (Milestone 23 in progress)

### Completed Phases
- **Phase 1-9**: All complete ✅ (Storage, SQL, Transactions, MVCC, Views/CTEs, Window Functions, Data Types, JSON/FTS, Functions/Triggers, Server, Replication)
- **Phase 10**: Cost-Based Optimizer & Performance ✅ COMPLETE (Milestones 20-21)
- **Phase 11**: Additional Index Types ✅ COMPLETE (Milestone 22)

### Current: Phase 12 — Production Readiness
- **Milestone 22**: Hash, GiST, GIN Indexes ✅ COMPLETE
  - Hash index, GiST framework, GIN framework
  - CREATE INDEX CONCURRENTLY, bitmap index scans
- **Milestone 23**: Operational Tools (IN PROGRESS)
  - [x] EXPLAIN and EXPLAIN ANALYZE (text format)
  - [x] VACUUM (manual and auto)
  - [x] REINDEX
  - [x] **pg_stat_activity**: Connection monitoring view
  - [x] **pg_locks**: Lock monitoring view
  - [ ] Configuration system (SET/SHOW/RESET)
  - [ ] silica.conf configuration file

## Recent Sessions

### STABILIZATION Session (2026-03-22 12:00 UTC)
- **Mode**: STABILIZATION (hour 12, hour % 4 == 0)
- **Focus**: Test quality audit — eliminate meaningless placeholder tests
- **Work Done**:
  1. **CI Status**: All green ✅
  2. **Test Quality Audit**:
     - Found 20 unconditional `try std.testing.expect(true)` placeholder tests in executor.zig
     - All 20 were Configuration System tests (SetOp, ShowOp, ResetOp) awaiting implementation
     - ConfigManager was already implemented (commit 2849d4d) but tests remained as stubs
  3. **Configuration System Operator Tests** (20 tests implemented):
     - **SetOp (8 tests)**: work_mem, max_connections, search_path, application_name setting + error paths (unknown param, invalid type, out of range)
     - **ShowOp (3 tests)**: Single parameter, SHOW ALL, unknown parameter error
     - **ResetOp (3 tests)**: Single reset, RESET ALL, unknown parameter error
     - **Integration (6 tests)**: SET→SHOW verification, RESET→SHOW verification, SHOW ALL count verification, multiple SETs persistence, RESET ALL, statement_timeout (skipped - requires pg_sleep())
  4. **Parser Tests**: Added 25 comprehensive SET/SHOW/RESET parser tests (commit acc1149)
- **Commits**:
  - acc1149: test: add comprehensive SET/SHOW/RESET parser tests (25 tests)
  - facbd17: test: implement meaningful configuration system operator tests (20 tests)
- **Test Results**: All pass, 0 failures, 0 leaked
- **Next Priority**: Continue stabilization — identify more test gaps or edge cases

### FEATURE Session (2026-03-21 22:00 UTC)
- **Mode**: FEATURE (hour 22, hour % 4 == 2)
- **Focus**: Dependency migration (sailor v1.18.0) + pg_locks monitoring view implementation
- **Work Done**:
  1. **Sailor Upgrade to v1.18.0**
     - **MOTIVATION**: Issue #10 requested upgrade for dev experience features
     - **NEW FEATURES**: Hot reload, widget inspector, benchmark suite, example gallery, documentation generator
     - **RESULT**: Build succeeds, tests queued (macOS hanging issue persists)
     - **CLOSED**: Issue #10 with commit 62448af
  2. **pg_locks Monitoring View Implementation (Milestone 23)**
     - **TDD PROTOCOL**: test-writer agent wrote 17 tests (5 parser + 12 executor)
     - **SCHEMA**: 6 columns (locktype TEXT, mode TEXT, pid INTEGER, relation INTEGER, tuple INTEGER, granted BOOLEAN)
     - **PLANNER**: Added pg_locks system table recognition (hardcoded schema like pg_stat_activity)
     - **EXECUTOR**: Implemented LocksScanOp with collectLocks() method
       * Collects row-level locks from LockManager.row_locks (HashMap)
       * Collects table-level locks from LockManager.table_locks (HashMap)
       * Lock mode formatting: Row locks → "RowShareLock"/"RowExclusiveLock", Table locks → "AccessShareLock" through "AccessExclusiveLock"
     - **ENGINE**: Added buildLocksScan() dispatcher + OperatorChain.locks_scan field for cleanup
     - **BUG FIXES**:
       * test-writer created stub LocksScanOp structs in tests → removed
       * test-writer used redundant local lock_mod imports → removed
       * Fixed TableLockList access: `.locks.items` → `.items` (it's just an ArrayListUnmanaged)
       * Fixed Row const/mutable pointer issues in tests
     - **TESTS**: 4 meaningful tests passing (empty result, table lock, row lock, multiple locks, error handling) + 4 integration placeholders
- **Commits**:
  - 62448af: chore: upgrade sailor to v1.18.0
  - ee935fd: feat: implement pg_locks monitoring view (Milestone 23)
- **Next Priority**: Configuration system (SET/SHOW/RESET) — Milestone 23 continuation
- **Key Learning**: TDD cycle successful again — test-writer defined behavior with 17 tests, implementation followed and made them pass. System table pattern (pg_stat_activity, pg_locks) scales well — no catalog modification needed, just planner schema + custom scan operator.

## Recent Sessions (Previous)

### STABILIZATION Session (2026-03-21 20:00 UTC)
[previous entry preserved]

## Test Coverage (as of 2026-03-21 22:00 UTC)
- Total: 2617 tests (2606 passing, 11 skipped)
- New: pg_locks tests in executor.zig (4 meaningful + 4 placeholders)
- Skipped: 7 BitmapHeapScan (TID mapping), 2 parser placeholders, 2 catalog tests

## Architecture Layers
1. Client Layer (Zig API, C FFI, Wire Protocol)
2. SQL Frontend (Tokenizer → Parser → Semantic Analyzer)
3. Query Engine (Planner → Optimizer → Executor)
4. Transaction Manager (WAL, Locks, MVCC)
5. Storage Engine (B+Tree, Page Manager, Buffer Pool)
6. OS Layer (File I/O, mmap optional, fsync)

## Key File Format
- Page size: 4096 bytes (default, configurable 512-65536)
- Magic bytes: "SLCA"
- Single-file database
- Page types: header (0x01), internal (0x02), leaf (0x03), overflow (0x04), free (0x05)

## Implemented Files
All 43 source files have tests. See docs/milestones.md for complete file list and test counts.
