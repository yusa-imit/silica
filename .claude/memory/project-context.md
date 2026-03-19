# Silica — Project Context

## Overview
- **Type**: Production-grade embedded relational database engine
- **Language**: Zig 0.15.x (stable)
- **Inspired by**: SQLite (simplicity, embeddability, single-file format)
- **Author**: Yusa

## Current Phase: Phase 10 — Advanced Optimization (Milestone 21 in progress)

### Completed Phases
- **Phase 1-9**: All complete ✅ (Storage, SQL, Transactions, MVCC, Views/CTEs, Window Functions, Data Types, JSON/FTS, Functions/Triggers, Server, Replication)

### Current: Phase 10 — Cost-Based Optimizer & Performance
- **Milestone 20**: Statistics & Cost Model ✅ COMPLETE
  - ANALYZE command, histograms, selectivity estimation, I/O+CPU cost model
- **Milestone 21**: Advanced Optimization IN PROGRESS
  - [x] Join reordering (simplified two-table)
  - [x] Hash/merge join selection with proper join key extraction
  - [x] EXPLAIN/EXPLAIN ANALYZE
  - [ ] Subquery decorrelation (blocked: requires Database handle threading)
  - [ ] Index-only scans (infrastructure added, full implementation deferred)

## Recent Sessions

### STABILIZATION Session (2026-03-20 00:00 UTC)
- **Mode**: STABILIZATION (hour 00, hour % 4 == 0)
- **Focus**: Test coverage expansion for recently added features
- **Work Done**:
  1. Committed uncommitted hash index catalog support (c34da3b)
     - AST, Catalog, Parser, Engine changes for CREATE INDEX USING HASH|BTREE
     - Backward-compatible serialization format
     - 3 integration tests in engine.zig
  2. Added 7 comprehensive catalog serialization tests (ac2d852)
     - Hash index serialization/deserialization
     - Unique hash index round-trip
     - Unique btree index round-trip
     - **Critical**: Backward compatibility test for old catalog format
     - Multiple indexes with mixed types
     - Hash index with INCLUDE columns
     - Empty index name edge case
- **Test Coverage**:
  - Total: 2391/2394 tests passing (3 skipped from previous sessions)
  - catalog.zig: +7 tests for IndexInfo.index_type and is_unique fields
  - Backward compatibility verified: old databases deserialize with defaults
- **Commits**: c34da3b (feat), ac2d852 (test)
- **Key Learning**: TDD protocol enforced — test-writer agent produced comprehensive edge case tests before any catalog format was used in production

### STABILIZATION Session (2026-03-19 16:00 UTC)
- **Mode**: STABILIZATION (hour 16, hour % 4 == 0)
- **Focus**: Code quality audit and edge case testing
- **Discovery**: Critical bug in HashJoinOp join key extraction (commit 6581dac)
  - Bug: extractJoinKeys ignored table prefixes in qualified column names
  - Impact: Joins with ambiguous column names (e.g., `t1.id = t2.id` where both tables have "id") failed
  - Root cause 1: getColumnRef returned only name.name, ignoring name.prefix
  - Root cause 2: Hash table built BEFORE extracting join keys, causing hash mismatch
- **Solution**:
  - Modified getColumnRef to return full ast.Name (with optional prefix)
  - Enhanced findColumnIndex to try qualified match first, fall back to unqualified
  - Refactored HashJoinOp.next() to rebuild hash table after extracting join keys
- **Tests**:
  - Added 3 comprehensive tests via test-writer agent:
    1. Qualified names with ambiguous columns (single equi-join)
    2. Mixed qualified/unqualified names (employees.dept_id = departments.dept_id)
    3. Multiple qualified columns in AND condition (composite join keys)
  - All 2253 tests passing (2250 previous + 3 new)
- **Commit**: 64ed780 fix(executor): handle qualified column names in HashJoinOp
- **Key Learning**: Recent feature additions (Milestone 21B HashJoinOp improvements) had subtle bug that only manifested with qualified column names — comprehensive edge case testing caught it before production

## Test Coverage (as of 2026-03-20 00:00 UTC)
- Total: 2391 tests (2391 passing, 3 skipped from previous sessions)
- executor.zig: 14 HashJoinOp tests (including qualified column name edge cases)
- catalog.zig: 7 new hash index serialization tests (backward compatibility verified)
- optimizer.zig: 51 tests (join reordering, filter pushdown, constant folding)
- cost.zig: 33 tests (cost estimation edge cases)
- selectivity.zig: 25 tests (selectivity estimation edge cases)
- Test quality: All modules have comprehensive edge case coverage, no meaningless always-passing tests found

## Architecture Layers
1. Client Layer (Zig API, C FFI, Wire Protocol)
2. SQL Frontend (Tokenizer → Parser → Semantic Analyzer)
3. Query Engine (Planner → Optimizer → Executor)
4. Transaction Manager (WAL, Locks, MVCC)
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
All 43 source files have tests. See docs/milestones.md for complete file list and test counts.
