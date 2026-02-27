# Silica — Architecture

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
