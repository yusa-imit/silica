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

## GIN Index Native Storage Wiring — Architect Design (Session 469, 2026-07-16)

**Problem**: `src/storage/gin_index.zig` (GIN posting-list struct: insert/delete/search, inline + posting-tree pages) and `src/storage/gist_index.zig` are fully implemented but **never invoked** anywhere outside their own files. `CREATE INDEX ... USING GIN` records `idx_type=.gin` in the catalog but the root page is initialized and used as a plain B+Tree page — DML (`insertIndexEntries`/`deleteIndexEntries`, engine.zig ~4923-5008) and scans (`IndexScanOp`, `BitmapIndexScanOp`, executor.zig ~8285/12263) all route `.gin`/`.gist` through `BTree.init(...).insert/delete/get`. Containment/full-text predicates (`@>`, `?`, `?|`, `?&`, `@@`, array `&&`) are only ever evaluated as per-row scalar checks during a full scan (`evalContainment`/`jsonContains`, executor.zig ~2260-2336) — never index-assisted. Confirmed via direct grep/read, not speculation (see [[next-priorities]]).

**Root blocker — row identity mismatch**: Silica is index-organized (rows live in a data B+Tree keyed by `row_key`, which is an 8-byte big-endian int for rowid tables but arbitrary bytes for PK tables). `gin_index.zig`'s posting lists store `ItemPointer{page_id, tuple_offset}` — a *physical* heap TID silica doesn't have. The existing `BitmapIndexScanOp`/`BitmapHeapScanOp` "TID" path is a known-broken placeholder (hashes row_key into a fake TID, ~8 tests skipped, **no MVCC visibility check** — do not build on it, violates rule 15).

**Resolution (Option A, adopted)**: restrict native GIN to rowid tables (row_key is exactly 8 bytes); pack the row_key bit-for-bit into the existing `ItemPointer` u64 field (ignore its page_id/offset semantics, treat as opaque 8-byte handle). Text/composite-PK tables keep the B+Tree fallback forever. This needs zero low-level page-format change to `gin_index.zig`.

**7-step phased plan** (each step TDD, green, B+Tree fallback intact until final cutover — gated behind a new `IndexInfo.gin_opclass` field that no existing catalog sets, so steps 1-6 are invisible to current behavior):
1. ✅ DONE (session 469, commit 36e7b3b) `catalog.zig`: add `IndexInfo.gin_opclass` enum (none/array_ops/jsonb_ops/tsvector_ops), backward-compatible optional-byte serialization (mirror existing pattern at catalog.zig:502-510).
2. **PARTIAL (session 471)** `gin_index.zig`: `array_ops` DONE — `ArrayOpsOpClass` (gin_index.zig ~171-317, 20 tests ~1394-1770) implements compare/extractValue/extractQuery/consistent. Input wire format is a local, independent reimplementation (storage layer must not import sql layer) of the tag+payload scheme used by executor.zig's `serializeValue`/`deserializeValue` (0x00 null … 0x0C array … 0x10 tsquery) — `column_value`/`query_value` must be a 0x0C array; each element's raw tag+payload span becomes one GIN key. `compare()` is deliberately plain lexicographic byte comparison (not type-aware/numeric) since GIN's entry-tree only needs a consistent total order, not semantic magnitude ordering — this is the reusable pattern for the remaining two opclasses. Code-reviewed (session 471): fixed one real issue (unvalidated `count` field could attempt a multi-GB allocation from a 5-byte malformed input — now bounds-checked against remaining buffer length before allocating, test added). Known follow-ups (not blockers, tracked for a future test-quality/stabilization pass): (a) no recursion depth limit on nested-array (0x0C-in-0x0C) parsing in `valueSpanLen` — mirrors a pre-existing identical gap in executor.zig's `deserializeValue`, now duplicated in a second place; (b) test suite only exercises integer/text element types, not boolean/date/time/timestamp/interval/numeric/uuid/tsvector/tsquery/nested-array elements (arithmetic manually verified correct for all, just no regression protection). **`jsonb_ops` DONE (session 472, commit 6a936ef)**: `JsonbOpsOpClass` (gin_index.zig, right after `ArrayOpsOpClass`'s closing brace, before the "GIN Tree Structure" section; 15 tests right after the array_ops tests) supports only `@>` (strategy 0) — `?`/`?|`/`?&` deliberately out of scope since they need a plain-text/text-array query wire format the current single-shape `OpClass.extractQuery(allocator, query_value)` signature can't cleanly express alongside `@>`'s jsonb-shaped query_value (would need an interface change, e.g. a strategy parameter on extractQuery, to do properly — don't fake it). JSON/JSONB columns are stored as raw `Value.text` (confirmed via executor.zig:3474-3485, `type_json`/`type_jsonb` → `Value{.text=...}`) — no dedicated json tag in the tag+payload scheme, so `column_value`/`query_value` here is tag 0x03 (text) wire format wrapping the JSON text. A recursive walker shared verbatim by extractValue/extractQuery emits: object key-exists entries (`0x01`+keylen+key) and key+scalar-value entries (`0x02`+keylen+key+scalarEncode(value)) at every nesting depth, array scalar-element entries (`0x03`+scalarEncode(value)) at every nesting depth, and one entry for a bare scalar document root — this flattens structure (no path/depth encoding), which is why it's lossy and step 7's FilterOp recheck is mandatory, not optional (soundness argument: recursive containment implies the contained document's entries are a subset of the containing document's flattened entry set, even though depth isn't tracked). `scalarEncode` is a small local tag scheme independent of executor.zig's. Also extracted `ArrayOpsOpClass.compare`'s lexicographic byte comparison into a shared `lexCompareKeys` free function reused by both opclasses. TDD: test-writer wrote 18 failing tests first; code-reviewer found and fixed one real bug — entry buffers leaked when `ArrayList.append`'s internal growth allocation failed mid-walk (buffer was allocated before the failed append, so the top-level `errdefer` — which only frees already-appended `entries.items` — never saw it); fixed by freeing the buffer specifically on that append's own failure at all 4 call sites. `tsvector_ops` still NOT started — needs the actual TSVECTOR/TSQUERY lexeme wire format investigated first (not yet read as of session 472).
3. `gin_index.zig`: make `GIN.search` strategy-aware (intersect for `@>`/`?&`, union for `&&`/`?|`) + dedup sorted-merge.
4. `engine.zig` CREATE INDEX: add `.gin` native page-init arm + opclass resolution from column type; error clearly on unsupported column/PK type (don't silently fall back for *new* indexes).
5. `engine.zig` `insertIndexEntries`/`deleteIndexEntries`: route `.gin` to native `GIN.insert/delete` only when `gin_opclass != .none && row_key.len == 8`; pass whole column value (not `valueToIndexKey`) so `GIN.insert`'s existing extractValue loop fans out multi-posting-per-row.
6. `executor.zig`: new `GinIndexScanOp` modeled on `IndexScanOp` (NOT the broken bitmap path) — reuses `IndexScanOp`'s exact MVCC visibility check verbatim (isVersionedRow/TupleHeader.deserialize/isTupleVisibleWithTm) per candidate row after `GIN.search` + row_key unpack + data-tree fetch.
7. **Cutover step** — planner: add `extractGinPredicate` (recognizes `@>`/`?`/`?|`/`?&`/`@@`/`&&`) in `tryBuildIndexScan` (engine.zig ~3952-4010, NOT optimizer.zig — that's confirmed to only do pushdown/join-algo, index selection lives in engine.zig), gated on native GIN index existing; **must wrap the GinIndexScanOp in the existing FilterOp as a correctness recheck** since `consistent` can be lossy — never let this recheck be "optimized away" later.

**NOT safe to do incrementally** (do first, atomically, if attempted):
- Single-page GIN entry tree has a hard capacity ceiling (`insertNewEntry`, gin_index.zig ~905-986, returns `error.PageFull` on a full root — no internal-node split exists). Converting to a real multi-level entry tree changes root_page_id semantics and page layout — must happen *before* step 4 ships any native index to disk, or requires a full on-disk migration afterward. Either scope initial release to low-cardinality columns, or do the entry-tree split as its own prerequisite effort.
- REINDEX (`rebuildIndex`, engine.zig ~8810-8878) has no GIN branch today — migrating an existing `.gin` (opclass=none, B+Tree-fallback) index to native GIN requires teaching REINDEX to init a native root + set gin_opclass + fan out via GIN.insert. Do this after step 6, as its own commit; until then native GIN is "new indexes only" (acceptable scope).

Full agent transcript/reasoning available via session notes if a future session needs the complete file:line citation list; this summary has the load-bearing decisions and file:line anchors needed to start implementation at step 1.

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
