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
2. ✅ **DONE (session 474, commits 5b76b1c/816938e)** `gin_index.zig`: all three opclasses complete. `array_ops` (session 471) — `ArrayOpsOpClass` (gin_index.zig ~171-317, 20 tests ~1394-1770) implements compare/extractValue/extractQuery/consistent. Input wire format is a local, independent reimplementation (storage layer must not import sql layer) of the tag+payload scheme used by executor.zig's `serializeValue`/`deserializeValue` (0x00 null … 0x0C array … 0x10 tsquery) — `column_value`/`query_value` must be a 0x0C array; each element's raw tag+payload span becomes one GIN key. `compare()` is deliberately plain lexicographic byte comparison (not type-aware/numeric) since GIN's entry-tree only needs a consistent total order, not semantic magnitude ordering — this is the reusable pattern for the other opclasses. Known follow-ups (not blockers, tracked for a future test-quality/stabilization pass): (a) no recursion depth limit on nested-array (0x0C-in-0x0C) parsing in `valueSpanLen` — mirrors a pre-existing identical gap in executor.zig's `deserializeValue`; (b) test suite only exercises integer/text element types, not boolean/date/time/timestamp/interval/numeric/uuid/tsvector/tsquery/nested-array elements. `jsonb_ops` (session 472, commit 6a936ef) — `JsonbOpsOpClass` supports only `@>` (strategy 0) — `?`/`?|`/`?&` deliberately out of scope, need an `extractQuery` interface change to express cleanly. JSON/JSONB columns are stored as raw `Value.text` — `column_value`/`query_value` is tag 0x03 (text) wire format wrapping JSON text. A recursive walker shared by extractValue/extractQuery flattens structure (no path/depth encoding) — lossy by design, so step 7's FilterOp recheck is mandatory, not optional. `tsvector_ops` (session 474, commit 5b76b1c, fixed 816938e) — `TsvectorOpsOpClass` for `@@` (match) support: `extractValue` uses tag 0x0F (space-separated lexemes), `extractQuery` uses tag 0x10 (space-ampersand-space joined lexemes), `consistent` is strategy-0 AND-only. Both extractValue/extractQuery skip empty lexemes from leading/trailing/consecutive separators (816938e fixed a gap where extractQuery didn't mirror extractValue's guard). 38 tests total added across both commits, all green.
3. ✅ **DONE (session 475, commit 80f6ed9)** `gin_index.zig`: `GIN.search` is now strategy-aware — strategy 1 (`&&`/`?|`, overlaps/OR) returns the deduplicated union of all posting lists; every other strategy (0 = `@>`/`?&`, contains-all/AND) still does the shortest-list-driven intersection. Bug found: the pre-existing code always fell through to the AND-intersection path regardless of strategy, so overlaps queries silently returned wrong (too-few) results. Fixed with a dedicated union branch gated on `strategy == 1`, plus 3 new regression tests (overlaps-union, contains-intersection regression guard, overlaps-dedup).
4. ✅ **DONE (session 479, commit 3cefe02)** `engine.zig` CREATE INDEX: `resolveGinOpClass` maps column type → opclass (array→array_ops, json/jsonb→jsonb_ops, tsvector→tsvector_ops, else none); when non-none, root page is initialized via new `gin_index.initEntryTreeLeafPage` (extracted from `GIN.getOrCreateRootFrame`) instead of `btree_mod.initLeafPage`. Deviates from the original plan's "error clearly on unsupported column/PK type" — unsupported types silently keep `gin_opclass = .none` and B+Tree-fallback page init instead, matching pre-existing GIN-on-scalar-column behavior (no behavior change for existing scalar-GIN users). Composite/text-PK tables aren't rejected at CREATE INDEX time either; they get `gin_opclass != .none` but fall back to B+Tree at DML time per step 5's `row_key.len == 8` gate. Revisit if this silent-fallback scope ever needs tightening.
5. ✅ **DONE (session 479, commit 3cefe02)** `engine.zig` `insertIndexEntries`/`deleteIndexEntries`: both now take an added `row_key: []const u8` param (all 5 call sites updated — UPSERT conflict path, batch UPDATE, batch DELETE, TRUNCATE-via-DELETE-all). `.gin` case routes to native `GIN.insert`/`GIN.delete` when `gin_opclass != .none and row_key.len == 8`, passing `executor_mod.serializeValueBytes(vals[idx.column_index])` (new helper, wraps `serializeValue` for a single `Value`) as the column bytes and `ItemPointer.fromU64(row_key as big-endian u64)` as the TID; otherwise falls back to the pre-existing B+Tree insert/delete on `idx_key`. 13 new tests in engine.zig covering catalog opclass wiring, native insert/search round-trip, multi-key fan-out, and delete-removes-posting — all green (4472/4502 passed repo-wide, 30 skipped, 0 failed).
6. ✅ **DONE (session 480, commit e00b942)** `executor.zig`: `GinIndexScanOp` modeled on `IndexScanOp` (NOT the broken bitmap path) — reuses `IndexScanOp`'s exact MVCC visibility check verbatim (isVersionedRow/TupleHeader.deserialize/isTupleVisibleWithTm) per candidate row after `GIN.search` + row_key unpack + data-tree fetch. 7 tests in executor.zig cover basic/multiple/no-match, MVCC filtering, orphaned entries, strategy 1 (OR).
7. ✅ **DONE (session 483, commit 24d3cd5)** **Cutover step — scoped to jsonb_ops `@>` and tsvector_ops `@@` only.** `extractGinPredicate` + `tryBuildGinIndexScan` added in `engine.zig` (right after `tryBuildIndexScan`/`extractEqualityPredicate` — planner index selection lives in engine.zig, not optimizer.zig, confirmed). Wired into `buildFilter`: `tryBuildIndexScan` tried first, then `tryBuildGinIndexScan`; when the latter succeeds its `RowIterator` is **always** wrapped in a `FilterOp` recheck of the original predicate (never optimized away, since GIN `consistent` is lossy by design for jsonb_ops/array_ops). `array_ops @>` is **intentionally excluded** from the cutover: `evalJsonContains` (executor.zig) only accepts `Value.text`/`Value.blob`, not `Value.array` — confirmed via a throwaway probe test that `labels @> ARRAY[10]` on an `INTEGER[]` column already threw `TypeError` *before* this cutover (full-scan FilterOp path), so wiring array_ops into GinIndexScanOp would not fix anything and the mandatory recheck would still TypeError on every row. A regression test (`"GIN cutover — array_ops @> is not routed through native GIN scan"`) locks in that this pre-existing gap is unchanged, not newly introduced. **Follow-up for a future session**: teach `evalJsonContains` (or a sibling function) to handle `Value.array` directly, then extend the cutover's opclass switch to include `array_ops`.
   - **Bug found and fixed in the same commit**: `peekIsDataType()` (parser.zig) was missing `.kw_tsvector`/`.kw_tsquery` — `CREATE TABLE t (col TSVECTOR)` always failed to parse (`expect(.right_paren)` error) even though the tokenizer keyword table, `parseDataType()`'s switch, `catalog.zig`'s type mapping, and `executor.zig`'s CAST/eval support for `.type_tsvector`/`.type_tsquery` were all already complete — a one-line parser gap that made the entire tsvector_ops opclass (steps 2-6) unreachable from SQL until now. Fixed by adding both tokens to `peekIsDataType`'s condition list.
   - RHS-is-constant check: `tryBuildGinIndexScan` evaluates `pred.rhs` via `evalExpr(allocator, rhs, &empty_row, null)` (same pattern `buildLimit` uses for LIMIT/OFFSET exprs) — if the RHS references a column, `evalExpr` returns `ColumnNotFound` which is caught and treated as "not applicable," falling back to full scan. This lets `to_tsquery('term')` (a function call, not a literal AST node) work as a GIN search key without needing a new literal-extraction special case.
   - GIN native storage wiring (all 7 steps) is now **feature-complete for jsonb_ops and tsvector_ops**; array_ops needs the `evalJsonContains` array-support follow-up above before its cutover can be safely enabled.

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
