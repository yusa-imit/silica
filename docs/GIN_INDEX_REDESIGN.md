# GIN Index Architectural Redesign

## Problem Statement

Currently, 5 out of 27 GIN index tests are skipped with "TODO: GIN architectural issues - needs redesign". The primary issue is that **search operations return empty results** even when matching tuples exist.

### Affected Tests
1. `test "GIN insert single value with single key"` (line 993)
2. `test "GIN insert single value with multiple keys"` (line 998)
3. `test "GIN insert common key in multiple rows"` (line 1054)
4. `test "GIN handles array with many elements"` (line 1114)
5. `test "GIN posting tree split when exceeding inline threshold"` (line 1119)

## Root Cause Analysis

### Current Architecture

GIN (Generalized Inverted Index) uses a two-level structure:
1. **Entry Tree**: B+Tree mapping `indexed_value → posting_list` (or `posting_tree_root_page`)
2. **Posting List/Tree**: Compact list or B+Tree of `tuple_ids` (ItemPointer)

```
Entry Tree (B+Tree):
  key="apple" → posting_list=[tid1, tid2, tid3]
  key="banana" → posting_tree_root_page=42

Posting Tree (B+Tree of tuple_ids):
  Page 42: [tid1, tid5, tid8, tid12, ...]
```

### Suspected Issues

1. **Entry Tree Corruption During Insert**
   - Problem: Multi-key inserts may corrupt entry tree structure
   - Evidence: Tests with multiple keys per value fail
   - Hypothesis: Incorrect B+Tree split/merge logic when inserting multiple entries

2. **Posting List Encoding Bugs**
   - Problem: Varint-encoded delta list may have off-by-one errors
   - Evidence: Single-key tests also fail (not just multi-key)
   - Hypothesis: `appendToPostingList()` or `readPostingList()` has encoding mismatch

3. **Search Path Bugs**
   - Problem: Search may not correctly traverse entry tree
   - Evidence: Inserts don't crash, but searches return empty
   - Hypothesis: Key comparison or leaf node traversal is incorrect

4. **OpClass Consistency Function**
   - Problem: `consistent()` check may be too strict
   - Evidence: Even simple contains checks fail
   - Hypothesis: ArrayInt32OpClass may require ALL keys to have results (AND instead of OR)

## Diagnostic Steps

### Step 1: Validate Entry Tree Integrity

Add debug logging to `insert()` to verify:
- Keys are inserted into entry tree
- Entry tree can be read back immediately after insert
- Leaf nodes contain expected key-value pairs

```zig
pub fn debugDumpEntryTree(self: *GinIndex) !void {
    // Walk entry tree and print all keys + posting info
    var cursor = try self.initCursor();
    defer cursor.deinit();

    while (try cursor.next()) |entry| {
        std.debug.print("Key: {s}, posting_info: 0x{x}\n", .{entry.key, entry.posting_info});
    }
}
```

### Step 2: Validate Posting List Encoding

Add test for posting list round-trip:
```zig
test "GIN posting list encode/decode round-trip" {
    const tids = [_]ItemPointer{
        .{ .page_id = 1, .tuple_offset = 0 },
        .{ .page_id = 1, .tuple_offset = 1 },
        .{ .page_id = 2, .tuple_offset = 0 },
    };

    // Encode to bytes
    var buf: [1024]u8 = undefined;
    const encoded = try encodePostingList(&buf, &tids);

    // Decode back
    const decoded = try decodePostingList(allocator, encoded);
    defer allocator.free(decoded);

    try std.testing.expectEqualSlices(ItemPointer, &tids, decoded);
}
```

### Step 3: Add Search Instrumentation

Modify `search()` to log each step:
```zig
pub fn search(self: *GinIndex, query_keys: []const []const u8) ![]const ItemPointer {
    std.debug.print("[GIN Search] Query keys: {any}\n", .{query_keys});

    var posting_lists = ArrayList([]const ItemPointer).init(self.allocator);
    defer posting_lists.deinit();

    for (query_keys) |key| {
        std.debug.print("[GIN Search] Looking up key: {s}\n", .{key});
        const posting_list = self.lookupPostingList(key) catch |err| {
            std.debug.print("[GIN Search] Lookup failed: {}\n", .{err});
            return err;
        };
        std.debug.print("[GIN Search] Found {} tuples\n", .{posting_list.len});
        try posting_lists.append(posting_list);
    }

    // ... rest of search logic
}
```

## Proposed Solution

### Option A: Fix In-Place (Incremental)

1. **Add extensive logging** to isolate which component fails
2. **Fix posting list encoding** if varint delta encoding is broken
3. **Fix entry tree traversal** if B+Tree search is incorrect
4. **Fix OpClass.consistent()** if logic is too strict

**Pros**: Preserves existing architecture, minimal code churn
**Cons**: May not address fundamental design issues

### Option B: Redesign from Scratch (Clean Slate)

1. **Simplify posting list format**: Use fixed-size `u64` instead of varint deltas
2. **Separate entry tree from B+Tree**: Use custom page format optimized for GIN
3. **Implement reference PostgreSQL GIN**: Follow pg_gin.c more closely

**Pros**: Opportunity to fix design flaws, cleaner implementation
**Cons**: Large code changes, may introduce new bugs

### Option C: Incremental Redesign (Hybrid)

1. **Phase 1**: Fix posting list encoding (use fixed u64, defer varint optimization)
2. **Phase 2**: Add comprehensive tests for each component (entry tree, posting list, search)
3. **Phase 3**: Optimize with varint deltas once correctness is proven

**Pros**: Balances correctness with performance, testable milestones
**Cons**: Multiple phases required

## Recommended Approach

**Option C (Incremental Redesign)** is recommended:

1. **Immediate**: Add diagnostic logging to existing code (Step 1-3 above)
2. **Phase 1**: ✅ **COMPLETE** (Session 302) — Replace varint delta encoding with fixed `u64` tuple IDs
   - Commit: 3659a1f
   - Format changed from `[tid0 u64][delta1 varint][delta2 varint]...` to `[tid0 u64][tid1 u64][tid2 u64]...`
   - MAX_INLINE_TUPLES reduced from 1000 to 16 (128 bytes / 8 bytes per tuple)
3. **Phase 2**: ✅ **COMPLETE** (Session 304) — Write unit tests for each GIN function
   - Added 8 comprehensive unit tests for posting list operations
   - Tests cover: ItemPointer serialization, sortedness enforcement, capacity limits, error handling
   - All tests target low-level functions: `insertNewEntry()`, `appendToPostingList()`, `readInlinePostingList()`
   - Tests use explicit page setup to verify byte-level correctness
4. **Phase 3**: Re-enable skipped tests one-by-one, fixing issues as they arise
5. **Phase 4**: Optimize posting list with varint deltas (optional, for v1.1)

## Testing Strategy

### Unit Tests (Phase 2 Complete — 8 tests added)
- ✅ `ItemPointer.toU64()` / `fromU64()` round-trip (3 tests: normal, max, zero)
- ✅ `insertNewEntry()` with valid posting list structure verification
- ✅ `appendToPostingList()` sortedness enforcement
- ✅ `appendToPostingList()` capacity limit (MAX_INLINE_TUPLES = 16)
- ✅ `readInlinePostingList()` empty list handling
- ✅ `readInlinePostingList()` corrupted tuple count rejection
- Future: `findEntry` in entry tree
- Future: `insertEntry` with key collisions
- Future: `deleteEntry` with posting list compaction

### Integration Tests (Existing but Failing)
- Single-key insert + search
- Multi-key insert + search
- Posting tree split threshold

### Fuzz Tests (Future)
- Random insert/delete/search sequences
- Corrupt posting list recovery
- Entry tree integrity after random operations

## Success Criteria

1. All 5 skipped GIN tests pass
2. Zero search failures on valid data
3. GIN fuzz tests pass for 10,000 operations
4. Performance benchmark: 10,000 inserts + 1,000 searches < 500ms

## References

- PostgreSQL GIN implementation: `src/backend/access/gin/`
- GIN paper: "Generalized Search Trees for Database Systems" (Hellerstein et al.)
- Current implementation: `src/storage/gin_index.zig`

## Assignee

- **Owner**: Future stabilization session (Session X90, X95, etc.)
- **Estimated effort**: 3-5 stabilization cycles
- **Blocking**: JSON/JSONB full-text search features (Phase 6)
- **Priority**: Medium (GIN works for some cases, just not multi-key scenarios)
