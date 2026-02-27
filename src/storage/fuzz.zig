//! Comprehensive fuzz tests for the B+Tree storage engine.
//!
//! These tests exercise the B+Tree with pseudo-random operation sequences,
//! diverse key/value sizes, and multiple page sizes to uncover edge cases
//! that deterministic tests might miss.
//!
//! Each test uses a deterministic seed for reproducibility.

const std = @import("std");
const page_mod = @import("page.zig");
const buffer_pool_mod = @import("buffer_pool.zig");
const btree_mod = @import("btree.zig");
const overflow_mod = @import("overflow.zig");

const Pager = page_mod.Pager;
const BufferPool = buffer_pool_mod.BufferPool;
const BTree = btree_mod.BTree;

// ── Test Helpers ─────────────────────────────────────────────────────────

/// Create a B+Tree backed by a temporary file for testing.
/// Uses heap allocation to avoid pointer invalidation from struct moves.
const TestTree = struct {
    pager: *Pager,
    pool: *BufferPool,
    tree: BTree,
    path: []const u8,
    allocator: std.mem.Allocator,

    fn init(allocator: std.mem.Allocator, path: []const u8, page_size: u32, pool_size: u32) !TestTree {
        const pager = try allocator.create(Pager);
        pager.* = try Pager.init(allocator, path, .{ .page_size = page_size });

        const root_id = try pager.allocPage();
        {
            const raw = try pager.allocPageBuf();
            defer pager.freePageBuf(raw);
            btree_mod.initLeafPage(raw, pager.page_size, root_id);
            try pager.writePage(root_id, raw);
        }

        const pool = try allocator.create(BufferPool);
        pool.* = try BufferPool.init(allocator, pager, pool_size);
        const tree = BTree.init(pool, root_id);

        return .{
            .pager = pager,
            .pool = pool,
            .tree = tree,
            .path = path,
            .allocator = allocator,
        };
    }

    fn deinit(self: *TestTree) void {
        self.pool.deinit();
        self.pager.deinit();
        self.allocator.destroy(self.pool);
        self.allocator.destroy(self.pager);
        std.fs.cwd().deleteFile(self.path) catch {};
    }
};

/// Generate a key string from an integer. Zero-padded for lexicographic order.
fn makeKey(buf: []u8, i: u32) []const u8 {
    return std.fmt.bufPrint(buf, "k{d:0>8}", .{i}) catch unreachable;
}

/// Generate a value of a given length filled with a deterministic byte pattern.
fn makeValue(allocator: std.mem.Allocator, seed: u32, len: usize) ![]u8 {
    const buf = try allocator.alloc(u8, len);
    var rng = std.Random.DefaultPrng.init(seed);
    const random = rng.random();
    for (buf) |*b| {
        b.* = random.intRangeAtMost(u8, 32, 126); // printable ASCII
    }
    return buf;
}

/// Verify that the B+Tree contains exactly the expected set of keys, and
/// that a forward cursor scan returns them in sorted order.
fn verifyTreeContents(
    allocator: std.mem.Allocator,
    tree: *BTree,
    expected_keys: *std.StringHashMap([]const u8),
) !void {
    // 1. Point lookups: every expected key must exist with correct value
    var it = expected_keys.iterator();
    while (it.next()) |entry| {
        const val = try tree.get(allocator, entry.key_ptr.*);
        if (val) |v| {
            defer allocator.free(v);
            std.testing.expectEqualSlices(u8, entry.value_ptr.*, v) catch |err| {
                std.debug.print("Value mismatch for key: {s}\n", .{entry.key_ptr.*});
                return err;
            };
        } else {
            std.debug.print("Missing key: {s}\n", .{entry.key_ptr.*});
            return error.TestUnexpectedResult;
        }
    }

    // 2. Forward cursor scan: must produce exactly expected_keys.count() entries in sorted order
    var cursor = btree_mod.Cursor.init(allocator, tree);
    defer cursor.deinit();
    try cursor.seekFirst();

    var scan_count: usize = 0;
    var prev_key: ?[]u8 = null;
    defer if (prev_key) |pk| allocator.free(pk);

    while (try cursor.next()) |entry| {
        defer allocator.free(entry.key);
        defer allocator.free(entry.value);
        scan_count += 1;

        // Verify sorted order
        if (prev_key) |pk| {
            if (std.mem.order(u8, pk, entry.key) != .lt) {
                std.debug.print("Sort violation: {s} >= {s}\n", .{ pk, entry.key });
                return error.TestUnexpectedResult;
            }
            allocator.free(pk);
        }
        prev_key = try allocator.dupe(u8, entry.key);

        // Verify this key exists in expected set
        if (!expected_keys.contains(entry.key)) {
            std.debug.print("Unexpected key in scan: {s}\n", .{entry.key});
            return error.TestUnexpectedResult;
        }
    }

    try std.testing.expectEqual(expected_keys.count(), scan_count);
}

// ── Fuzz Test 1: Random insert/delete sequences ─────────────────────────

test "fuzz: random insert-delete sequences" {
    const allocator = std.testing.allocator;
    const path = "test_fuzz_insert_delete.db";

    var tt = try TestTree.init(allocator, path, 4096, 500);
    defer tt.deinit();

    // Track which keys are currently in the tree and their values
    var live_keys = std.StringHashMap([]const u8).init(allocator);
    defer {
        var it = live_keys.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            allocator.free(entry.value_ptr.*);
        }
        live_keys.deinit();
    }

    var rng = std.Random.DefaultPrng.init(0xDEAD_BEEF);
    const random = rng.random();
    var key_buf: [20]u8 = undefined;

    const OPS = 500;
    for (0..OPS) |_| {
        const op = random.intRangeAtMost(u8, 0, 2);

        if (op <= 1 or live_keys.count() == 0) {
            // INSERT — weighted 2:1 towards inserts to grow the tree
            const key_id = random.intRangeAtMost(u32, 0, 999);
            const k = makeKey(&key_buf, key_id);

            if (live_keys.contains(k)) continue; // skip duplicates

            const val_len = random.intRangeAtMost(usize, 0, 50);
            const v = try makeValue(allocator, key_id, val_len);

            try tt.tree.insert(k, v);

            const owned_key = try allocator.dupe(u8, k);
            try live_keys.put(owned_key, v);
        } else {
            // DELETE — pick a random live key
            const count = live_keys.count();
            var skip = random.intRangeLessThan(usize, 0, count);
            var it = live_keys.iterator();
            var target_key: []const u8 = undefined;
            while (it.next()) |entry| {
                if (skip == 0) {
                    target_key = entry.key_ptr.*;
                    break;
                }
                skip -= 1;
            }

            try tt.tree.delete(target_key);

            const removed = live_keys.fetchRemove(target_key).?;
            allocator.free(removed.key);
            allocator.free(removed.value);
        }
    }

    // Verify final state
    try verifyTreeContents(allocator, &tt.tree, &live_keys);
}

// ── Fuzz Test 2: Random insert-delete with small pages ──────────────────

test "fuzz: random operations with 512-byte pages" {
    const allocator = std.testing.allocator;
    const path = "test_fuzz_small_pages.db";

    var tt = try TestTree.init(allocator, path, 512, 500);
    defer tt.deinit();

    var live_keys = std.StringHashMap([]const u8).init(allocator);
    defer {
        var it = live_keys.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            allocator.free(entry.value_ptr.*);
        }
        live_keys.deinit();
    }

    var rng = std.Random.DefaultPrng.init(0xCAFE_BABE);
    const random = rng.random();
    var key_buf: [20]u8 = undefined;

    // Small pages mean more splits/merges per operation
    const OPS = 300;
    for (0..OPS) |_| {
        const op = random.intRangeAtMost(u8, 0, 2);

        if (op <= 1 or live_keys.count() == 0) {
            const key_id = random.intRangeAtMost(u32, 0, 499);
            const k = makeKey(&key_buf, key_id);
            if (live_keys.contains(k)) continue;

            const val_len = random.intRangeAtMost(usize, 0, 30);
            const v = try makeValue(allocator, key_id, val_len);

            try tt.tree.insert(k, v);
            const owned_key = try allocator.dupe(u8, k);
            try live_keys.put(owned_key, v);
        } else {
            const count = live_keys.count();
            var skip = random.intRangeLessThan(usize, 0, count);
            var it = live_keys.iterator();
            var target_key: []const u8 = undefined;
            while (it.next()) |entry| {
                if (skip == 0) {
                    target_key = entry.key_ptr.*;
                    break;
                }
                skip -= 1;
            }

            try tt.tree.delete(target_key);
            const removed = live_keys.fetchRemove(target_key).?;
            allocator.free(removed.key);
            allocator.free(removed.value);
        }
    }

    try verifyTreeContents(allocator, &tt.tree, &live_keys);
}

// ── Fuzz Test 3: Mixed overflow and inline values ───────────────────────

test "fuzz: mixed overflow and inline values" {
    const allocator = std.testing.allocator;
    const path = "test_fuzz_overflow_mix.db";

    var tt = try TestTree.init(allocator, path, 4096, 500);
    defer tt.deinit();

    var live_keys = std.StringHashMap([]const u8).init(allocator);
    defer {
        var it = live_keys.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            allocator.free(entry.value_ptr.*);
        }
        live_keys.deinit();
    }

    var rng = std.Random.DefaultPrng.init(0xF00D_FACE);
    const random = rng.random();
    var key_buf: [20]u8 = undefined;

    // Mix of tiny (0-10), medium (50-200), and overflow-sized (1000-5000) values
    const OPS = 200;
    for (0..OPS) |_| {
        const op = random.intRangeAtMost(u8, 0, 3);

        if (op <= 2 or live_keys.count() == 0) {
            const key_id = random.intRangeAtMost(u32, 0, 499);
            const k = makeKey(&key_buf, key_id);
            if (live_keys.contains(k)) continue;

            // Vary value sizes: 25% tiny, 50% medium, 25% overflow
            const size_class = random.intRangeAtMost(u8, 0, 3);
            const val_len: usize = switch (size_class) {
                0 => random.intRangeAtMost(usize, 0, 10),
                1, 2 => random.intRangeAtMost(usize, 50, 200),
                3 => random.intRangeAtMost(usize, 1000, 5000),
                else => unreachable,
            };

            const v = try makeValue(allocator, key_id, val_len);

            try tt.tree.insert(k, v);
            const owned_key = try allocator.dupe(u8, k);
            try live_keys.put(owned_key, v);
        } else {
            const count = live_keys.count();
            var skip = random.intRangeLessThan(usize, 0, count);
            var it = live_keys.iterator();
            var target_key: []const u8 = undefined;
            while (it.next()) |entry| {
                if (skip == 0) {
                    target_key = entry.key_ptr.*;
                    break;
                }
                skip -= 1;
            }

            try tt.tree.delete(target_key);
            const removed = live_keys.fetchRemove(target_key).?;
            allocator.free(removed.key);
            allocator.free(removed.value);
        }
    }

    try verifyTreeContents(allocator, &tt.tree, &live_keys);
}

// ── Fuzz Test 4: Insert-delete-reinsert stress ──────────────────────────

test "fuzz: insert all, delete all, reinsert all" {
    const allocator = std.testing.allocator;
    const path = "test_fuzz_reinsert.db";

    var tt = try TestTree.init(allocator, path, 4096, 500);
    defer tt.deinit();

    var key_buf: [20]u8 = undefined;
    const N: u32 = 200;

    // Phase 1: Insert N keys
    for (0..N) |i| {
        const k = makeKey(&key_buf, @intCast(i));
        const v = try makeValue(allocator, @intCast(i), 20);
        defer allocator.free(v);
        try tt.tree.insert(k, v);
    }

    // Verify all present
    for (0..N) |i| {
        const k = makeKey(&key_buf, @intCast(i));
        const val = try tt.tree.get(allocator, k);
        try std.testing.expect(val != null);
        allocator.free(val.?);
    }

    // Phase 2: Delete all in random order
    var delete_order: [N]u32 = undefined;
    for (0..N) |i| delete_order[i] = @intCast(i);
    var rng = std.Random.DefaultPrng.init(0xBAAD_F00D);
    rng.random().shuffle(u32, &delete_order);

    for (delete_order) |id| {
        const k = makeKey(&key_buf, id);
        try tt.tree.delete(k);
    }

    // Verify all gone
    for (0..N) |i| {
        const k = makeKey(&key_buf, @intCast(i));
        const val = try tt.tree.get(allocator, k);
        try std.testing.expect(val == null);
    }

    // Phase 3: Reinsert all with different values
    for (0..N) |i| {
        const k = makeKey(&key_buf, @intCast(i));
        const v = try makeValue(allocator, @intCast(i + 1000), 30);
        defer allocator.free(v);
        try tt.tree.insert(k, v);
    }

    // Verify all present with new values
    for (0..N) |i| {
        const k = makeKey(&key_buf, @intCast(i));
        const val = try tt.tree.get(allocator, k);
        try std.testing.expect(val != null);
        const expected = try makeValue(allocator, @intCast(i + 1000), 30);
        defer allocator.free(expected);
        try std.testing.expectEqualSlices(u8, expected, val.?);
        allocator.free(val.?);
    }
}

// ── Fuzz Test 5: Cursor consistency after random mutations ──────────────

test "fuzz: cursor scan matches point lookups after random ops" {
    const allocator = std.testing.allocator;
    const path = "test_fuzz_cursor_consistency.db";

    var tt = try TestTree.init(allocator, path, 1024, 500);
    defer tt.deinit();

    var live_keys = std.StringHashMap([]const u8).init(allocator);
    defer {
        var it = live_keys.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            allocator.free(entry.value_ptr.*);
        }
        live_keys.deinit();
    }

    var rng = std.Random.DefaultPrng.init(0x1234_5678);
    const random = rng.random();
    var key_buf: [20]u8 = undefined;

    // Build a tree with random operations
    for (0..400) |_| {
        const op = random.intRangeAtMost(u8, 0, 2);

        if (op <= 1 or live_keys.count() == 0) {
            const key_id = random.intRangeAtMost(u32, 0, 799);
            const k = makeKey(&key_buf, key_id);
            if (live_keys.contains(k)) continue;

            const val_len = random.intRangeAtMost(usize, 1, 40);
            const v = try makeValue(allocator, key_id, val_len);
            try tt.tree.insert(k, v);
            const owned_key = try allocator.dupe(u8, k);
            try live_keys.put(owned_key, v);
        } else {
            const count = live_keys.count();
            var skip = random.intRangeLessThan(usize, 0, count);
            var it = live_keys.iterator();
            var target_key: []const u8 = undefined;
            while (it.next()) |entry| {
                if (skip == 0) {
                    target_key = entry.key_ptr.*;
                    break;
                }
                skip -= 1;
            }
            try tt.tree.delete(target_key);
            const removed = live_keys.fetchRemove(target_key).?;
            allocator.free(removed.key);
            allocator.free(removed.value);
        }
    }

    // Verify: forward scan produces sorted keys matching all live keys
    try verifyTreeContents(allocator, &tt.tree, &live_keys);

    // Also verify backward scan produces correct reverse order
    var cursor = btree_mod.Cursor.init(allocator, &tt.tree);
    defer cursor.deinit();
    try cursor.seekLast();

    var backward_count: usize = 0;
    var next_key: ?[]u8 = null;
    defer if (next_key) |nk| allocator.free(nk);

    while (try cursor.prev()) |entry| {
        defer allocator.free(entry.key);
        defer allocator.free(entry.value);
        backward_count += 1;

        // Verify reverse sorted order
        if (next_key) |nk| {
            if (std.mem.order(u8, entry.key, nk) != .lt) {
                std.debug.print("Backward sort violation: {s} >= {s}\n", .{ entry.key, nk });
                return error.TestUnexpectedResult;
            }
            allocator.free(nk);
        }
        next_key = try allocator.dupe(u8, entry.key);
    }

    try std.testing.expectEqual(live_keys.count(), backward_count);
}

// ── Fuzz Test 6: Seek cursor accuracy ───────────────────────────────────

test "fuzz: cursor seek finds correct positions" {
    const allocator = std.testing.allocator;
    const path = "test_fuzz_cursor_seek.db";

    var tt = try TestTree.init(allocator, path, 2048, 500);
    defer tt.deinit();

    // Insert keys with gaps: 0, 3, 6, 9, ... to leave room for seek targets
    var key_buf: [20]u8 = undefined;
    var val_buf: [20]u8 = undefined;

    for (0..100) |i| {
        const id: u32 = @intCast(i * 3);
        const k = makeKey(&key_buf, id);
        const v = std.fmt.bufPrint(&val_buf, "val{d:0>5}", .{id}) catch unreachable;
        try tt.tree.insert(k, v);
    }

    // Seek to various positions — both existing and non-existing keys
    var rng = std.Random.DefaultPrng.init(0xAAAA_BBBB);
    const random = rng.random();

    for (0..100) |_| {
        const seek_id = random.intRangeAtMost(u32, 0, 310);
        const seek_key = makeKey(&key_buf, seek_id);

        var cursor = btree_mod.Cursor.init(allocator, &tt.tree);
        defer cursor.deinit();
        try cursor.seek(seek_key);

        const entry = try cursor.next();
        if (entry) |e| {
            defer allocator.free(e.key);
            defer allocator.free(e.value);

            // The returned key must be >= seek_key
            const order = std.mem.order(u8, e.key, seek_key);
            if (order == .lt) {
                std.debug.print("Seek returned key < target: {s} < {s}\n", .{ e.key, seek_key });
                return error.TestUnexpectedResult;
            }

            // It must also be the smallest key >= seek_key
            // (i.e., the next multiple of 3 >= seek_id)
            const expected_id = ((seek_id + 2) / 3) * 3; // round up to next multiple of 3
            if (expected_id < 300) {
                var expected_buf: [20]u8 = undefined;
                const expected_key = makeKey(&expected_buf, expected_id);
                try std.testing.expectEqualSlices(u8, expected_key, e.key);
            }
        } else {
            // No entry means seek_key is past all keys
            // All keys are 0,3,...,297 so seek_id >= 298 should produce null
            try std.testing.expect(seek_id >= 298);
        }
    }
}

// ── Fuzz Test 7: Multiple page sizes ────────────────────────────────────

test "fuzz: correctness across page sizes" {
    const allocator = std.testing.allocator;
    const page_sizes = [_]u32{ 512, 1024, 2048, 4096 };

    for (page_sizes) |ps| {
        var path_buf: [64]u8 = undefined;
        const path = std.fmt.bufPrint(&path_buf, "test_fuzz_pagesize_{d}.db", .{ps}) catch unreachable;

        var tt = try TestTree.init(allocator, path, ps, 500);
        defer tt.deinit();

        var key_buf: [20]u8 = undefined;

        // Insert 150 keys
        const N: u32 = 150;
        for (0..N) |i| {
            const k = makeKey(&key_buf, @intCast(i));
            const v = try makeValue(allocator, @intCast(i), 15);
            defer allocator.free(v);
            try tt.tree.insert(k, v);
        }

        // Delete every 3rd key
        for (0..N) |i| {
            if (i % 3 == 0) {
                const k = makeKey(&key_buf, @intCast(i));
                try tt.tree.delete(k);
            }
        }

        // Verify remaining keys
        for (0..N) |i| {
            const k = makeKey(&key_buf, @intCast(i));
            const val = try tt.tree.get(allocator, k);
            if (i % 3 == 0) {
                try std.testing.expect(val == null);
            } else {
                try std.testing.expect(val != null);
                const expected = try makeValue(allocator, @intCast(i), 15);
                defer allocator.free(expected);
                try std.testing.expectEqualSlices(u8, expected, val.?);
                allocator.free(val.?);
            }
        }

        // Cursor scan count should match surviving keys
        var cursor = btree_mod.Cursor.init(allocator, &tt.tree);
        defer cursor.deinit();
        try cursor.seekFirst();

        var count: usize = 0;
        while (try cursor.next()) |entry| {
            allocator.free(entry.key);
            allocator.free(entry.value);
            count += 1;
        }
        try std.testing.expectEqual(@as(usize, N - N / 3), count);
    }
}

// ── Fuzz Test 8: Sequential insert then random delete ───────────────────

test "fuzz: sequential insert, random delete pattern" {
    const allocator = std.testing.allocator;
    const path = "test_fuzz_seq_insert_rand_delete.db";

    var tt = try TestTree.init(allocator, path, 4096, 500);
    defer tt.deinit();

    var key_buf: [20]u8 = undefined;
    const N: u32 = 300;

    // Insert 0..N-1 in order
    for (0..N) |i| {
        const k = makeKey(&key_buf, @intCast(i));
        const v = try makeValue(allocator, @intCast(i), 25);
        defer allocator.free(v);
        try tt.tree.insert(k, v);
    }

    // Delete in random order
    var order: [N]u32 = undefined;
    for (0..N) |i| order[i] = @intCast(i);
    var rng = std.Random.DefaultPrng.init(0xFACE_1234);
    rng.random().shuffle(u32, &order);

    // Delete first half
    const half = N / 2;
    var deleted = std.AutoHashMap(u32, void).init(allocator);
    defer deleted.deinit();

    for (order[0..half]) |id| {
        const k = makeKey(&key_buf, id);
        try tt.tree.delete(k);
        try deleted.put(id, {});
    }

    // Verify exactly the remaining half exists
    for (0..N) |i| {
        const k = makeKey(&key_buf, @intCast(i));
        const val = try tt.tree.get(allocator, k);
        if (deleted.contains(@intCast(i))) {
            try std.testing.expect(val == null);
        } else {
            try std.testing.expect(val != null);
            allocator.free(val.?);
        }
    }

    // Cursor should return exactly half entries
    var cursor = btree_mod.Cursor.init(allocator, &tt.tree);
    defer cursor.deinit();
    try cursor.seekFirst();

    var count: usize = 0;
    while (try cursor.next()) |entry| {
        allocator.free(entry.key);
        allocator.free(entry.value);
        count += 1;
    }
    try std.testing.expectEqual(@as(usize, half), count);
}

// ── Fuzz Test 9: Reverse insert order stress ────────────────────────────

test "fuzz: reverse order inserts with small pages" {
    const allocator = std.testing.allocator;
    const path = "test_fuzz_reverse_small.db";

    var tt = try TestTree.init(allocator, path, 512, 500);
    defer tt.deinit();

    var key_buf: [20]u8 = undefined;
    const N: u32 = 200;

    // Insert in reverse order — worst case for right-heavy splits
    var i: u32 = N;
    while (i > 0) {
        i -= 1;
        const k = makeKey(&key_buf, i);
        const v = try makeValue(allocator, i, 10);
        defer allocator.free(v);
        try tt.tree.insert(k, v);
    }

    // Delete odd-numbered keys
    for (0..N) |j| {
        if (j % 2 == 1) {
            const k = makeKey(&key_buf, @intCast(j));
            try tt.tree.delete(k);
        }
    }

    // Verify even keys remain
    for (0..N) |j| {
        const k = makeKey(&key_buf, @intCast(j));
        const val = try tt.tree.get(allocator, k);
        if (j % 2 == 1) {
            try std.testing.expect(val == null);
        } else {
            try std.testing.expect(val != null);
            allocator.free(val.?);
        }
    }
}

// ── Fuzz Test 10: Overflow values with small pages ──────────────────────

test "fuzz: overflow values with 512-byte pages" {
    const allocator = std.testing.allocator;
    const path = "test_fuzz_overflow_small_pages.db";

    var tt = try TestTree.init(allocator, path, 512, 500);
    defer tt.deinit();

    var live_keys = std.StringHashMap([]const u8).init(allocator);
    defer {
        var it = live_keys.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            allocator.free(entry.value_ptr.*);
        }
        live_keys.deinit();
    }

    var rng = std.Random.DefaultPrng.init(0x9999_AAAA);
    const random = rng.random();
    var key_buf: [20]u8 = undefined;

    // With 512-byte pages, even moderate values trigger overflow
    const OPS = 100;
    for (0..OPS) |_| {
        const op = random.intRangeAtMost(u8, 0, 2);

        if (op <= 1 or live_keys.count() == 0) {
            const key_id = random.intRangeAtMost(u32, 0, 199);
            const k = makeKey(&key_buf, key_id);
            if (live_keys.contains(k)) continue;

            // Mix: some inline, some overflow for 512B pages
            const val_len = random.intRangeAtMost(usize, 10, 500);
            const v = try makeValue(allocator, key_id, val_len);

            try tt.tree.insert(k, v);
            const owned_key = try allocator.dupe(u8, k);
            try live_keys.put(owned_key, v);
        } else {
            const count = live_keys.count();
            var skip = random.intRangeLessThan(usize, 0, count);
            var it = live_keys.iterator();
            var target_key: []const u8 = undefined;
            while (it.next()) |entry| {
                if (skip == 0) {
                    target_key = entry.key_ptr.*;
                    break;
                }
                skip -= 1;
            }
            try tt.tree.delete(target_key);
            const removed = live_keys.fetchRemove(target_key).?;
            allocator.free(removed.key);
            allocator.free(removed.value);
        }
    }

    try verifyTreeContents(allocator, &tt.tree, &live_keys);
}

// ── Fuzz Test 11: Rapid grow-shrink cycles ──────────────────────────────

test "fuzz: grow-shrink cycles" {
    const allocator = std.testing.allocator;
    const path = "test_fuzz_grow_shrink.db";

    var tt = try TestTree.init(allocator, path, 2048, 500);
    defer tt.deinit();

    var key_buf: [20]u8 = undefined;
    var next_id: u32 = 0;

    // Track live key IDs in a hash set
    var live = std.AutoHashMap(u32, void).init(allocator);
    defer live.deinit();

    // 5 cycles of grow-then-shrink
    for (0..5) |cycle| {
        // Grow: insert 100 keys
        for (0..100) |_| {
            const k = makeKey(&key_buf, next_id);
            const v = try makeValue(allocator, next_id, 20);
            defer allocator.free(v);
            try tt.tree.insert(k, v);
            try live.put(next_id, {});
            next_id += 1;
        }

        // Shrink: delete 80 keys chosen pseudo-randomly
        var rng = std.Random.DefaultPrng.init(next_id +% @as(u32, @intCast(cycle)));
        const random = rng.random();
        var deleted_count: usize = 0;
        const target_deletes: usize = 80;

        // Collect current live keys into a temporary buffer for shuffled deletion
        const live_count = live.count();
        const live_arr = try allocator.alloc(u32, live_count);
        defer allocator.free(live_arr);
        {
            var it = live.keyIterator();
            var idx: usize = 0;
            while (it.next()) |key_ptr| {
                live_arr[idx] = key_ptr.*;
                idx += 1;
            }
        }
        random.shuffle(u32, live_arr);

        const to_delete = @min(target_deletes, live_arr.len);
        for (live_arr[0..to_delete]) |id| {
            const k = makeKey(&key_buf, id);
            try tt.tree.delete(k);
            _ = live.remove(id);
            deleted_count += 1;
        }
    }

    // Verify all remaining keys
    var it = live.keyIterator();
    while (it.next()) |id_ptr| {
        const k = makeKey(&key_buf, id_ptr.*);
        const val = try tt.tree.get(allocator, k);
        try std.testing.expect(val != null);
        allocator.free(val.?);
    }
}

// ── Fuzz Test 12: Duplicate key rejection under stress ──────────────────

test "fuzz: duplicate key rejection is consistent" {
    const allocator = std.testing.allocator;
    const path = "test_fuzz_duplicates.db";

    var tt = try TestTree.init(allocator, path, 4096, 500);
    defer tt.deinit();

    var key_buf: [20]u8 = undefined;
    const N: u32 = 200;

    // Insert keys
    for (0..N) |i| {
        const k = makeKey(&key_buf, @intCast(i));
        const v = try makeValue(allocator, @intCast(i), 15);
        defer allocator.free(v);
        try tt.tree.insert(k, v);
    }

    // Try to insert every key again — all must fail with DuplicateKey
    for (0..N) |i| {
        const k = makeKey(&key_buf, @intCast(i));
        const result = tt.tree.insert(k, "new_value");
        try std.testing.expectError(error.DuplicateKey, result);
    }

    // Delete some, then verify re-insert works for deleted ones
    for (0..N) |i| {
        if (i % 4 == 0) {
            const k = makeKey(&key_buf, @intCast(i));
            try tt.tree.delete(k);
        }
    }

    for (0..N) |i| {
        const k = makeKey(&key_buf, @intCast(i));
        if (i % 4 == 0) {
            // Was deleted — should accept insert
            const v = try makeValue(allocator, @intCast(i + 5000), 15);
            defer allocator.free(v);
            try tt.tree.insert(k, v);
        } else {
            // Still exists — should reject
            const result = tt.tree.insert(k, "new_value");
            try std.testing.expectError(error.DuplicateKey, result);
        }
    }
}
