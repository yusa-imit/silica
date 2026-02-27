# Silica — Project Context

## Overview
- **Type**: Production-grade embedded relational database engine
- **Language**: Zig 0.14.x (stable)
- **Inspired by**: SQLite (simplicity, embeddability, single-file format)
- **Author**: Yusa

## Current Phase: Phase 1 — Storage Foundation (Weeks 1-6)

### Milestone 1 — Page Manager & File Format (Weeks 1-2)
- [ ] Database file header (Page 0) — Magic: "SLCA"
- [ ] Page read/write with CRC32C checksums
- [ ] Freelist management (allocate/free pages)
- [ ] Basic test suite: create DB, write pages, reopen and verify

### Milestone 2 — B+Tree & Buffer Pool (Weeks 3-6)
- [ ] B+Tree insert, delete, point lookup
- [ ] Leaf page splits and merges
- [ ] Range scan cursors (forward/backward)
- [ ] Overflow pages for large values
- [ ] LRU buffer pool with dirty page tracking
- [ ] Comprehensive B+Tree fuzz tests

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
- Page types: metadata, internal B+Tree, leaf, overflow, freelist
