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
- `build.zig` — Build system (Zig 0.15 API)
- `build.zig.zon` — Package metadata
- `src/main.zig` — Entry point with module imports
- `src/util/checksum.zig` — CRC32C using std.hash.crc.Crc32Iscsi
- `src/util/varint.zig` — LEB128 unsigned varint encode/decode
- `src/storage/page.zig` — Pager with header, read/write, freelist
- `src/storage/buffer_pool.zig` — LRU buffer pool with pin/unpin, dirty tracking
- `src/storage/btree.zig` — B+Tree with slotted-page layout, insert/delete/get, splits, merges, cursor, overflow support
- `src/storage/overflow.zig` — Overflow page chain management (write/read/free)
- `src/storage/fuzz.zig` — Comprehensive B+Tree fuzz tests (12 tests)

## Test Summary (147 tests total)
- `btree.zig`: 46 tests — CRUD, splits, merges, underflow, cursors, overflow insert/get/delete/cursor
- `fuzz.zig`: 12 tests — random insert/delete, small pages, overflow mix, reinsert, cursor consistency, seek, multi-page-size, grow-shrink, duplicates
- `overflow.zig`: 18 tests — chain write/read/free, single/multi-page, prefix, boundaries, min page size
- `buffer_pool.zig`: 15 tests — fetch/unpin, LRU eviction, dirty flush, pool size=1, stress pin cycles
- `page.zig`: 24 tests — header/alloc/free, checksums, freelist chains, max/min page sizes, persistence
- `checksum.zig`: 12 tests — known values, incremental hashing, bit flip detection
- `varint.zig`: 19 tests — roundtrip all ranges, boundary values, overflow detection, bit patterns

## Next Phase: Phase 2 — SQL Layer (Weeks 7-14)

### Milestone 3 — Tokenizer & Parser
- [ ] Tokenizer (3A) — hand-written lexer, SQL keyword recognition
- [ ] Parser (3B) — recursive descent → AST
- [ ] DDL statements (3C) — CREATE TABLE, DROP TABLE
- [ ] DML statements (3D) — SELECT, INSERT, UPDATE, DELETE
- [ ] Parser error recovery (3E)
