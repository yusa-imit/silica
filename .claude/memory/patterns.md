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

### Page checksum (CRC32C)
```zig
const data = page.data[0..page.size - 4]; // exclude checksum field
const checksum = std.hash.crc.Crc32c.hash(data);
std.mem.writeInt(u32, page.data[page.size - 4..], checksum, .little);
```

### Varint encoding (for compact integer storage)
```zig
// Encode: smaller values use fewer bytes
// Decode: read until MSB is 0
```

<!-- Add new patterns as they are verified through implementation -->
