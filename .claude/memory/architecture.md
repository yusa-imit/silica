# Silica — Architecture

## Concurrency Limitations (CRITICAL — Session 40 Finding)

**Silica v0.7.0 does NOT support concurrent connections.**

### Per-Connection Resource Isolation

Each `Database.open()` creates isolated instances:
1. **Buffer Pool** (engine.zig:743-746) — Separate in-memory page cache per connection
2. **WAL** (engine.zig:751-756) — **CRITICAL BUG**: Multiple Wal instances write to same file without synchronization!
3. **Transaction Manager** — Correctly shared via global registry

### Concurrency Bugs

1. **WAL Corruption**: Multiple connections write interleaved frames to `db-wal` → corrupt checksums, lost frames
2. **Stale Cache**: Connection A modifies page → Connection B serves old cached copy from its buffer pool
3. **Rollback Hazard**: Connection A's rollback truncates WAL, may discard Connection B's committed data

### Impact

- Jepsen-style concurrent tests fail with data loss (expected 1000, found 995)
- UPDATE/DELETE in concurrent transactions cause NoRows errors (issue #20)
- Multi-connection workloads are UNSAFE

### Fix Required (Milestone 26+)

1. **Shared Buffer Pool** with proper locking (like PostgreSQL's shared_buffers)
2. **Single WAL Manager** or serialized WAL writes
3. **Multi-version storage** for true MVCC

**Current Status**: Single-connection mode only. Multi-connection support deferred to Milestone 26.

## Layered Architecture

```
┌─────────────────────────────────────────┐
│            Client Layer                  │
│  Zig API (embedded) | C FFI | Wire Proto│
├─────────────────────────────────────────┤
│            SQL Frontend                  │
│  Tokenizer → Parser → Semantic Analyzer │
├─────────────────────────────────────────┤
│            Query Engine                  │
│  Planner → Optimizer → Executor (Volcano)│
├─────────────────────────────────────────┤
│         Transaction Manager              │
│  WAL Writer | Lock Manager | MVCC (future)│
├─────────────────────────────────────────┤
│           Storage Engine                 │
│  B+Tree | Page Manager | Buffer Pool    │
├─────────────────────────────────────────┤
│             OS Layer                     │
│  File I/O | mmap (optional) | fsync     │
└─────────────────────────────────────────┘
```

## Module Dependencies (Build Order)

```
util (checksum, varint) → storage (page, btree, buffer_pool) → tx (wal, lock) → sql (tokenizer, parser, analyzer) → query (planner, optimizer, executor) → server (wire, connection)
```

## Key Interfaces (To Be Defined)

### Pager Interface
- `readPage(page_num: u32) -> *Page`
- `writePage(page_num: u32, data: []const u8) -> void`
- `allocPage() -> u32`
- `freePage(page_num: u32) -> void`

### B+Tree Interface
- `insert(key: []const u8, value: []const u8) -> void`
- `delete(key: []const u8) -> bool`
- `get(key: []const u8) -> ?[]const u8`
- `cursor() -> Cursor` (range scans)

### Buffer Pool Interface
- `fetchPage(page_num: u32) -> *BufferFrame`
- `unpinPage(page_num: u32, dirty: bool) -> void`
- `flushAll() -> void`

## Dependency Migrations (Silica v1.0.0+)

### zuda LRUCache (Session 46 — COMPLETED)
**Completed**: BufferPool LRU eviction replaced with `zuda.containers.cache.LRUCache(u32, u32, AutoContext, null)`
- **Removed**: Manual doubly-linked list (prev/next fields, lru_head/lru_tail/lru_size)
- **Benefit**: Production-tested LRU implementation, ~30 LOC reduction
- **Impact**: All 2262 tests pass, zero regressions

## File Format

```
Page 0: Database Header
  - Magic: "SLCA" (4 bytes)
  - Format version: u32
  - Page size: u32
  - Total page count: u32
  - Freelist head: u32
  - Schema version: u32
  - WAL mode flag: u8
  - Reserved: padding to page_size

Page 1: Schema table root (B+Tree)
Page 2..N: Data & Index pages
```
