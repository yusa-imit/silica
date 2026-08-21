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
7. ✅ **DONE (session 483, commit 24d3cd5; array_ops completed session 484, commit 330d7de)** **Cutover step — now covers all three opclasses: jsonb_ops `@>`, array_ops `@>`, tsvector_ops `@@`.** `extractGinPredicate` + `tryBuildGinIndexScan` added in `engine.zig` (right after `tryBuildIndexScan`/`extractEqualityPredicate` — planner index selection lives in engine.zig, not optimizer.zig, confirmed). Wired into `buildFilter`: `tryBuildIndexScan` tried first, then `tryBuildGinIndexScan`; when the latter succeeds its `RowIterator` is **always** wrapped in a `FilterOp` recheck of the original predicate (never optimized away, since GIN `consistent` is lossy by design). Session 483 scoped the cutover to jsonb_ops/tsvector_ops only, since `evalJsonContains` (executor.zig) rejected `Value.array` with `TypeError`. Session 484 closed that gap: `evalJsonContains` now special-cases `left == .array or right == .array` (both must be arrays, else `TypeError`) and delegates to new helpers `valueArrayContains`/`valueContainsElement` — set-containment semantics (order/duplicates don't matter, right's elements must each match some left element via `Value.eql`, recursing into nested arrays) operating directly on `Value`, not JSON-text round-tripping (arrays were never JSON-serialized to begin with). `tryBuildGinIndexScan`'s opclass switch now includes `.array_ops => if (pred.op == .json_contains) 0 else return null` (same strategy 0 as jsonb_ops). The old regression test documenting the gap (`"GIN cutover — array_ops @> is not routed..."`) was replaced with `"GIN cutover — array_ops @> uses native GIN scan and excludes non-matching rows"`, which asserts actual correct row filtering.
   - **Bug found and fixed in commit 24d3cd5 (session 483)**: `peekIsDataType()` (parser.zig) was missing `.kw_tsvector`/`.kw_tsquery` — `CREATE TABLE t (col TSVECTOR)` always failed to parse (`expect(.right_paren)` error) even though the tokenizer keyword table, `parseDataType()`'s switch, `catalog.zig`'s type mapping, and `executor.zig`'s CAST/eval support for `.type_tsvector`/`.type_tsquery` were all already complete — a one-line parser gap that made the entire tsvector_ops opclass (steps 2-6) unreachable from SQL until now. Fixed by adding both tokens to `peekIsDataType`'s condition list.
   - RHS-is-constant check: `tryBuildGinIndexScan` evaluates `pred.rhs` via `evalExpr(allocator, rhs, &empty_row, null)` (same pattern `buildLimit` uses for LIMIT/OFFSET exprs) — if the RHS references a column, `evalExpr` returns `ColumnNotFound` which is caught and treated as "not applicable," falling back to full scan. This lets `to_tsquery('term')` (a function call, not a literal AST node) work as a GIN search key without needing a new literal-extraction special case.
   - GIN native storage wiring (all 7 steps) is now **feature-complete for jsonb_ops, array_ops, and tsvector_ops end-to-end via SQL**. No further follow-up work is tracked for this feature.

**NOT safe to do incrementally** (do first, atomically, if attempted):
- Single-page GIN entry tree has a hard capacity ceiling (`insertNewEntry`, gin_index.zig ~905-986, returns `error.PageFull` on a full root — no internal-node split exists). Converting to a real multi-level entry tree changes root_page_id semantics and page layout — must happen *before* step 4 ships any native index to disk, or requires a full on-disk migration afterward. Either scope initial release to low-cardinality columns, or do the entry-tree split as its own prerequisite effort.
- REINDEX (`rebuildIndex`, engine.zig ~8810-8878) has no GIN branch today — migrating an existing `.gin` (opclass=none, B+Tree-fallback) index to native GIN requires teaching REINDEX to init a native root + set gin_opclass + fan out via GIN.insert. Do this after step 6, as its own commit; until then native GIN is "new indexes only" (acceptable scope).

Full agent transcript/reasoning available via session notes if a future session needs the complete file:line citation list; this summary has the load-bearing decisions and file:line anchors needed to start implementation at step 1.

## Index-Only Scan — Architect Design (Session 487, 2026-08-22)

**Problem**: `src/sql/index_entry.zig` (committed session 487, commit 9d7a358) is a storage-agnostic encode/decode wire format for covering-index B+Tree leaf entries, but nothing writes or reads it yet. Secondary index B+Tree leaf values are `row_key`-only (confirmed `engine.zig:5068-5069` `insertIndexEntries`), so `INCLUDE (...)` columns (fully parsed + catalog-persisted already: `ast.CreateIndexStmt.included_columns`, `catalog.IndexInfo.included_columns`) are currently decorative — a covering-index query still needs a heap fetch, defeating the point.

**Key finding — existing scaffolding is half-wired and currently a dead/broken no-op, not a blank slate**: `planner.zig:186` `PlanNode.Scan.index_only: bool` and `optimizer.zig:172-195` `optimizeScan` (calling `indexCoversColumns` at optimizer.zig:428-445) already exist and can set `index_only = true` — but (a) `scan.columns` fed into `indexCoversColumns` is always the *full table schema* (planner.zig:1040-1049), not query-referenced columns, so the check essentially never passes for a realistic partial-covering index, and (b) `engine.zig`'s `buildTableScan`/`tryBuildIndexScan` never read `scan.index_only` at all (zero hits outside planner/optimizer). 4 commented-out acceptance tests already exist at `executor.zig:18121-18260` specifying the intended end-to-end semantics — adapt, don't write from scratch.

**No B+Tree/page-format change needed** — unlike the GIN cutover, `BTree.insert(key, value)` already handles arbitrary-length values via the existing `overflow.zig` mechanism, so wider covering-index leaf values need no new storage capability.

**MVCC**: `DecodedIndexEntry.header` is the same `mvcc_mod.TupleHeader` type used everywhere; visibility check is exactly `mvcc_mod.isTupleVisibleWithTm(decoded.header, ctx.snapshot, ctx.current_xid, ctx.current_cid, ctx.tm)` — identical to `IndexScanOp.next()`'s existing call (executor.zig:8352). Every DML path writes the index entry's header from the *same* `TupleHeader` value used for the heap row in the same call, and every UPDATE unconditionally rewrites all secondary index entries (engine.zig:5921-5938) regardless of whether the indexed column changed — so covering entries can't go stale relative to the heap row.

**Backward compatibility gate**: new `IndexInfo.covering_storage: bool = false` catalog field (mirrors `gin_opclass`'s exact backward-compatible trailing-byte pattern, catalog.zig:380-381). Defaults `false` for every pre-existing serialized index row — **this flag, not `included_columns.len > 0`**, gates index-only scan selection. An old `INCLUDE`-clause index has non-empty `included_columns` already but row_key-only leaves; using column-presence as the trigger would misparse legacy leaves as covering entries. `optimizeScan` requires `idx.covering_storage == true` *and* the column-coverage check.

**Scope**: `covering_storage` only ever settable for `index_type == .btree` (composite index *keys* aren't even supported today — `IndexInfo` is single-column-key only, engine.zig ~8194-8196 only uses `ci.columns[0]`). `.hash`/`.gist`/`.gin` covering storage explicitly out of scope.

**7-step phased plan** (each step TDD, green; steps 1-5 invisible to existing behavior since nothing sets `covering_storage = true` until step 4, and nothing reads it until step 6 — same gating discipline as the GIN plan):
1. `catalog.zig`: add `IndexInfo.covering_storage: bool = false`, backward-compatible trailing-byte serialize/deserialize (mirror `gin_opclass`'s exact pattern at catalog.zig:380-381 and the conditional-read logic ~483-565). Tests: round-trip with/without the byte present (old-format fixture), default-false for pre-existing catalogs.
2. `optimizer.zig`: **real prerequisite, not polish** — add a "required referenced columns" collection pass (new function collecting `column_ref`s from `Project`/`Filter`/`Sort`/`Aggregate`/`Window`/`Join.on_condition` for a given table/alias, generalizing the existing `exprMentionsTable`-style walk at optimizer.zig:448-458) and feed its output into `indexCoversColumns` instead of the full table schema. Likely needs a small structural change to `Optimizer.optimize()` (precompute a `table/alias → []ColumnRef` map once, pass down) since the current recursive-descent shape gives leaf-level `optimizeScan` no ancestor context. Pure, storage-independent, unit-testable on constructed `PlanNode` trees; still a no-op end-to-end (nothing downstream reads `index_only` yet).
3. `engine.zig` DML wiring: thread `header: TupleHeader` into `insertIndexEntries` at all 3 call sites (INSERT engine.zig:4636/4603, batch UPDATE engine.zig:5938/5880, `ON CONFLICT DO UPDATE` engine.zig:4783 — see bug note below, this path has no header computed today and needs `TupleHeader.forInsert(txn.xid, cid)` added). Inside the `.btree` branch, when `idx.covering_storage`, resolve `included_columns` → `vals[]` and call `index_entry.encodeIndexEntry` instead of the plain row_key dupe; unchanged when `!idx.covering_storage`. `deleteIndexEntries` needs no change (delete-by-key, value format irrelevant). Still no SQL-visible behavior change (`covering_storage` always false until step 4) — test via direct unit calls with a manually-constructed `covering_storage=true` `IndexInfo`.
4. `engine.zig` CREATE INDEX: set `covering_storage = included_columns.len > 0 and index_type == .btree` when building `IndexInfo` from `CreateIndexStmt`. First step where real SQL (`CREATE INDEX ... INCLUDE (...)`) writes real covering entries end-to-end — still invisible to query results (nothing reads covering data yet). Tests: create, insert, directly decode leaf bytes via `index_entry.decodeIndexEntry` to confirm covering data + correct MVCC header.
5. `executor.zig`: add `IndexScanOp.covering: bool` fast path (skip heap fetch, decode via `index_entry.decodeIndexEntry`, visibility-check the decoded header) for the point-lookup case, and a new `IndexOnlyScanOp` (modeled on `ScanOp` executor.zig:8092-8131, cursor over the index B+Tree instead of data B+Tree, decoding each leaf via `decodeIndexEntry`) for the full/ordered-scan case — same "sibling struct for a different iteration shape" precedent as `GinIndexScanOp` vs `IndexScanOp`. Built/tested directly (construct the op, feed a pre-populated covering index) without planner involvement — still no visible behavior change.
6. **Cutover**: `optimizer.zig` adds `and idx.covering_storage` to `optimizeScan`'s condition; `tryBuildIndexScan` (engine.zig:4021, WHERE-equality path) builds the covering `IndexScanOp` variant when `idx_info.covering_storage` and column-coverage both hold; `buildTableScan` (engine.zig:3916, full/ordered-scan path) builds `IndexOnlyScanOp` when `scan.index_only`. **This is the only step that changes observable query behavior.** Uncomment/adapt the 4 skipped acceptance tests at executor.zig:18121-18260 (covered SELECT uses index; SELECT * or partial coverage falls back to heap scan — falls out naturally from the column-coverage check, no special-case needed; WHERE-equality covering scan; results match equivalent heap-scan exactly).
7. Backward-compat regression test: hand-construct an `IndexInfo` simulating a pre-step-4 on-disk catalog (`included_columns` non-empty, `covering_storage = false`) and confirm the planner never selects index-only scan for it.

**Explicitly out of scope / not safe incrementally**: REINDEX has no path to upgrade an existing `covering_storage=false` INCLUDE index into a covering one (same "new indexes only" scope call as the GIN plan made for its own REINDEX gap) — until taught to do so, such indexes are permanently heap-fetch-only unless dropped and recreated. Composite index keys and non-btree covering storage are structurally impossible per the CREATE INDEX gating in step 4, not just unimplemented.

**Bug found during this review, independent of index-only scan — flagged, not fixed by the architect (design-only task)**: `ON CONFLICT DO UPDATE` (engine.zig:4777) writes the heap row via plain `serializeRow` instead of `serializeVersionedRow` — no `TupleHeader` at all, unlike every other INSERT/UPDATE path in the file. The updated row falls into the "legacy, no header, always visible" fallback branch in scan operators, silently bypassing snapshot isolation for that row until it's next touched by a header-writing path. This violates rule 15 ("MVCC visibility is sacred") independent of anything in this plan — track and fix separately, must be fixed before step 3 above (which adds a `TupleHeader.forInsert` at this same call site for index-entry purposes) since the two fixes touch the same code path.

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
