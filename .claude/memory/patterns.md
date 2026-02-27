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

<!-- Add new patterns as they are verified through implementation -->
