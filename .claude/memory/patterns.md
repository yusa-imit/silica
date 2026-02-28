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

### Buffer Pool pin/unpin — VERIFIED
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

<!-- Add new patterns as they are verified through implementation -->
