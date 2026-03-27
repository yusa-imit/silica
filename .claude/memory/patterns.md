# Silica — Verified Code Patterns

## Allocator Patterns

### Transaction-scoped arena
```zig
var arena = std.heap.ArenaAllocator.init(backing_allocator);
defer arena.deinit();
const alloc = arena.allocator();
// All transaction allocations use `alloc` — freed on commit/rollback
```

### Testing with leak detection
```zig
test "no leaks" {
    const allocator = std.testing.allocator;
    var obj = try MyStruct.init(allocator);
    defer obj.deinit();
    // ...
}
```

## Resource Safety Patterns

### Pin/Unpin for buffer pool pages
```zig
const frame = try pool.fetchPage(page_num);
defer pool.unpinPage(page_num, false); // unpin on exit
// use frame.data...
```

### File handle safety
```zig
const file = try std.fs.cwd().openFile(path, .{});
defer file.close();
```

## Database Patterns

### Page checksum (CRC32C) — VERIFIED
```zig
const Crc32c = std.hash.crc.Crc32Iscsi;  // Zig 0.15 API
const content = buf[PAGE_HEADER_SIZE..page_size];
const crc = Crc32c.hash(content);
std.mem.writeInt(u32, buf[12..16], crc, .little);
```

### Varint encoding — VERIFIED
```zig
const varint = @import("util/varint.zig");
var buf: [varint.max_encoded_len]u8 = undefined;
const n = try varint.encode(value, &buf);
const result = try varint.decode(buf[0..n]);
// result.value, result.bytes_read
```

### Pager page alloc/write/read — VERIFIED
```zig
var pager = try Pager.init(allocator, path, .{});
defer pager.deinit();

const page_id = try pager.allocPage();
const buf = try pager.allocPageBuf();
defer pager.freePageBuf(buf);

@memset(buf, 0);
const header = PageHeader{ .page_type = .leaf, .page_id = page_id };
header.serialize(buf[0..PAGE_HEADER_SIZE]);
try pager.writePage(page_id, buf);  // auto-checksums
try pager.readPage(page_id, buf);   // auto-verifies checksum
```

### Zig 0.15 Build Pattern — VERIFIED
```zig
const mod = b.addModule("silica", .{
    .root_source_file = b.path("src/main.zig"),
    .target = target,
    .optimize = optimize,
});
const lib = b.addLibrary(.{ .name = "silica", .root_module = mod, .linkage = .static });
const tests = b.addTest(.{ .root_module = mod });
```

### Buffer Pool pin/unpin with zuda LRUCache — VERIFIED
```zig
const buffer_pool = @import("storage/buffer_pool.zig");
var pool = try buffer_pool.BufferPool.init(allocator, &pager, 0); // 0 = default 2000
defer pool.deinit();

const frame = try pool.fetchPage(page_id);
defer pool.unpinPage(page_id, false); // unpin on exit, dirty=false
// Use frame.data[0..page_size]
frame.markDirty(); // or pass dirty=true to unpinPage

// For new pages:
const new_pid = try pager.allocPage();
const new_frame = try pool.fetchNewPage(new_pid);
// new_frame is auto-dirty, write content, then unpin
pool.unpinPage(new_pid, true);
try pool.flushAll();

// Internal LRU management: BufferPool uses zuda.containers.cache.LRUCache(u32, u32, AutoContext, null)
// - Unpinned frames are tracked: page_id -> frame_index
// - getEvictableFrame() finds oldest unpinned frame (LRU end of iterator)
// - Error handling: lru.put() failures in unpinPage() are caught and ignored (cleanup path)
```

### B+Tree usage — VERIFIED
```zig
const btree = @import("storage/btree.zig");

// Create a root leaf page on disk first
const root_id = try pager.allocPage();
{
    const raw = try pager.allocPageBuf();
    defer pager.freePageBuf(raw);
    btree.initLeafPage(raw, pager.page_size, root_id);
    try pager.writePage(root_id, raw);
}

var tree = btree.BTree.init(&pool, root_id);
try tree.insert("key", "value");
const val = try tree.get(allocator, "key"); // returns owned slice or null
if (val) |v| { defer allocator.free(v); }
try tree.delete("key");
// NOTE: tree.root_page_id may change after insert (root split)
```

### B+Tree page layout — VERIFIED
- Leaf: `[PageHeader 16B][prev_leaf 4B][next_leaf 4B][cell_ptrs...] ... [cells←]`
- Internal: `[PageHeader 16B][right_child 4B][cell_ptrs...] ... [cells←]`
- Leaf cell: `[key_len varint][key_data][value_len varint][value_data]`
- Internal cell: `[left_child u32 LE][key_len varint][key_data]`
- Cell pointers are u16 offsets, sorted by key order
- Cells grow from end of page backward (slotted page design)

### @memcpy aliasing — IMPORTANT
Zig's `@memcpy` panics if src and dst overlap. Use:
- `std.mem.copyForwards` when dst < src (e.g., deleting/shifting left)
- `std.mem.copyBackwards` when dst > src (e.g., inserting/shifting right)

### Lock Manager usage — VERIFIED
```zig
const lock_mod = @import("tx/lock.zig");
const LockManager = lock_mod.LockManager;
const LockTarget = lock_mod.LockTarget;
const LockMode = lock_mod.LockMode;
const TableLockMode = lock_mod.TableLockMode;

var lm = LockManager.init(allocator);
defer lm.deinit();

// Row-level lock
const target = LockTarget{ .table_page_id = 5, .row_key = 100 };
try lm.acquireRowLock(xid, target, .exclusive);
defer lm.releaseRowLock(xid, target);

// Table-level lock (implicit on DML)
try lm.acquireTableLock(xid, table_page_id, .row_exclusive);

// Release all locks on transaction end
lm.releaseAllLocks(xid);

// Check for conflicts before acquiring
if (lm.hasConflict(xid, target, .exclusive)) {
    return error.LockConflict;
}
```

### Lock Manager conflict resolution — VERIFIED
- Shared locks: multiple holders allowed, compatible with each other
- Exclusive locks: single holder only, conflicts with all modes
- Lock upgrade: shared→exclusive only if sole holder, otherwise conflict
- Table locks: follow PostgreSQL 7-mode conflict matrix
- releaseAllLocks: iterates both row_locks and table_locks hash maps
- Custom hash map context for LockTarget (table_page_id + row_key composite key)

### Zig 0.15 ArrayList pattern — VERIFIED
```zig
// Zig 0.15: ArrayList initialized without .init()
var list = std.ArrayList(u8){}; // NOT ArrayList.init(allocator)
defer list.deinit(allocator); // deinit takes allocator parameter

// For appending, must pass allocator explicitly:
try list.append(allocator, value);

// Convert to owned slice:
const slice = try list.toOwnedSlice(allocator);
defer allocator.free(slice);
```

### Thread Mutex pattern (Zig 0.15) — VERIFIED
```zig
var mutex = std.Thread.Mutex{}; // Initialize with .{}
defer {
    mutex.lock();
    defer mutex.unlock();
    // Protected code
}
// OR one-liner:
{
    mutex.lock();
    defer mutex.unlock();
    // Protected code
}
```

## Stress Testing Patterns (Milestone 19C/D)

### Concurrent operations stress test
```zig
test "concurrent stress test" {
    const allocator = std.testing.allocator;
    var manager = SlotManager.init(allocator);
    defer manager.deinit();

    const num_threads = 10;
    const ops_per_thread = 20;
    var threads: [num_threads]std.Thread = undefined;

    // Spawn threads performing create/activate/update/drop operations
    for (&threads) |*thread| {
        thread.* = try std.Thread.spawn(.{}, workerFunction, .{&manager});
    }

    // Wait for all threads
    for (threads) |thread| {
        thread.join();
    }

    // Verify final state consistency
}
```

### High-locality workload stress test
```zig
test "sequential page reuse stress test" {
    const allocator = std.testing.allocator;
    var pool = try BufferPool.init(allocator, 10);  // Small pool
    defer pool.deinit();

    const num_rounds = 20;
    const pages_per_round = 10;

    // Repeatedly access same page range to verify LRU correctness
    for (0..num_rounds) |round| {
        for (0..pages_per_round) |i| {
            const page_num = (round * pages_per_round) + i;
            // Fetch, use, unpin
        }
    }
}
```

### Bulk write capacity stress test
```zig
test "many frames in single commit" {
    const allocator = std.testing.allocator;
    var wal = try Wal.init(allocator, "test.wal", 4096);
    defer wal.deinit();

    const num_frames = 100;

    // Write many frames before commit
    for (0..num_frames) |i| {
        try wal.writeFrame(page_data, @intCast(i));
    }

    try wal.commit();  // Verify bulk commit works
}
```

## GiST Index Pattern — VERIFIED

### Operator Class Interface
```zig
pub const OpClass = struct {
    consistent: *const fn (allocator: std.mem.Allocator, entry_pred: []const u8, query: []const u8, strategy: u8) Error!bool,
    union_fn: *const fn (allocator: std.mem.Allocator, entries: []const []const u8) Error![]u8,
    penalty: *const fn (allocator: std.mem.Allocator, current_pred: []const u8, new_pred: []const u8) Error!u64,
    picksplit: *const fn (allocator: std.mem.Allocator, entries: []const []const u8) Error!struct { group_a: []usize, group_b: []usize },
    same: *const fn (allocator: std.mem.Allocator, pred_a: []const u8, pred_b: []const u8) Error!bool,
};

// Example: Int4RangeOpClass for [lo, hi) integer ranges
pub const Int4RangeOpClass = struct {
    pub fn consistent(allocator: std.mem.Allocator, entry_pred: []const u8, query: []const u8, strategy: u8) Error!bool {
        // strategy 0 = contains, 1 = overlaps
    }
    pub fn getOpClass() OpClass { return .{ .consistent = consistent, ... }; }
};

// Usage
var gist = try GiST.init(allocator, &pool, root_page_id, Int4RangeOpClass.getOpClass());
try gist.insert(predicate, tuple_id);
const results = try gist.search(query, 1); // strategy=1 for overlap
defer allocator.free(results);
```

### GiST Page Layout
- Leaf: `[PageHeader 16B][entry_count u16][reserved 2B][entry_0_pred_size u16][entry_0_tuple_id u32]...[predicates←]`
- Internal: `[PageHeader 16B][child_count u16][reserved 2B][child_0_pred_size u16][child_0_page_id u32]...[predicates←]`
- Entry header: `[pred_size u16][tuple_id/child_id u32]` = 6 bytes fixed
- Predicates: variable-length, stored from page end backward (slotted layout like B+Tree)

<!-- Add new patterns as they are verified through implementation -->

## Zig 0.15.2 Specific Patterns

### Thread sleep — VERIFIED
```zig
// CORRECT (Zig 0.15.2)
std.Thread.sleep(100); // 100 nanoseconds

// INCORRECT (old API)
std.time.sleep(100); // Does not exist in 0.15.2
```

### Stress test patterns
```zig
test "concurrent stress test" {
    const allocator = std.testing.allocator;
    var shared_state = try MyStruct.init(allocator);
    defer shared_state.deinit();

    const worker = struct {
        fn run(state: *MyStruct) void {
            var i: usize = 0;
            while (i < 50) : (i += 1) {
                state.doOperation() catch {};
                std.Thread.sleep(1000); // Small delay between ops
            }
        }
    }.run;

    // Launch N threads
    var threads: [8]std.Thread = undefined;
    for (&threads) |*thread| {
        thread.* = try std.Thread.spawn(.{}, worker, .{&shared_state});
    }
    for (threads) |thread| {
        thread.join();
    }
}
```

### Sequential lifecycle stress test
```zig
test "sequential lifecycle stress" {
    const allocator = std.testing.allocator;
    var coordinator = try Coordinator.init(allocator);
    defer coordinator.deinit();

    var i: usize = 0;
    while (i < 100) : (i += 1) {
        try coordinator.start();
        try coordinator.doWork();
        try coordinator.complete();
        try coordinator.cleanup(); // CRITICAL: clean up between iterations
    }
}
```
