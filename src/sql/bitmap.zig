//! Bitmap Index Scan Support
//!
//! Provides RowKeySet — a sorted, deduped set of owned row_key byte slices
//! used by bitmap index scans to combine multiple index results efficiently
//! via two-pointer set operations (intersect/unionOf).
//!
//! Row keys are arbitrary-length byte sequences (typically 8-byte big-endian
//! u64 for rowid tables, or variable-length for text/composite-key tables).

const std = @import("std");
const Allocator = std.mem.Allocator;

// ── Tests ────────────────────────────────────────────────────────────────────

const testing = std.testing;

test "RowKeySet.fromOwnedUnsorted sorts and dedupes input" {
    const allocator = testing.allocator;

    // Create unsorted input with duplicates
    var items = try allocator.alloc([]u8, 5);
    defer allocator.free(items);

    items[0] = try allocator.dupe(u8, "charlie");
    items[1] = try allocator.dupe(u8, "alice");
    items[2] = try allocator.dupe(u8, "charlie"); // duplicate
    items[3] = try allocator.dupe(u8, "bob");
    items[4] = try allocator.dupe(u8, "alice"); // duplicate

    var set = try RowKeySet.fromOwnedUnsorted(allocator, items);
    defer set.deinit(allocator);

    // Should be sorted and deduped: alice, bob, charlie (3 unique items)
    try testing.expectEqual(@as(usize, 3), set.items.len);
    try testing.expectEqualSlices(u8, "alice", set.items[0]);
    try testing.expectEqualSlices(u8, "bob", set.items[1]);
    try testing.expectEqualSlices(u8, "charlie", set.items[2]);
}

test "RowKeySet.fromOwnedUnsorted with empty input" {
    const allocator = testing.allocator;

    const items = try allocator.alloc([]u8, 0);
    defer allocator.free(items);

    var set = try RowKeySet.fromOwnedUnsorted(allocator, items);
    defer set.deinit(allocator);

    try testing.expectEqual(@as(usize, 0), set.items.len);
}

test "RowKeySet.fromOwnedUnsorted with single item" {
    const allocator = testing.allocator;

    var items = try allocator.alloc([]u8, 1);
    defer allocator.free(items);

    items[0] = try allocator.dupe(u8, "single");

    var set = try RowKeySet.fromOwnedUnsorted(allocator, items);
    defer set.deinit(allocator);

    try testing.expectEqual(@as(usize, 1), set.items.len);
    try testing.expectEqualSlices(u8, "single", set.items[0]);
}

test "RowKeySet.intersect with partial overlap" {
    const allocator = testing.allocator;

    // Create set A: [alice, bob, charlie, david]
    var items_a = try allocator.alloc([]u8, 4);
    defer allocator.free(items_a);
    items_a[0] = try allocator.dupe(u8, "alice");
    items_a[1] = try allocator.dupe(u8, "bob");
    items_a[2] = try allocator.dupe(u8, "charlie");
    items_a[3] = try allocator.dupe(u8, "david");

    var set_a = try RowKeySet.fromOwnedUnsorted(allocator, items_a);
    defer set_a.deinit(allocator);

    // Create set B: [bob, charlie, eve, frank]
    var items_b = try allocator.alloc([]u8, 4);
    defer allocator.free(items_b);
    items_b[0] = try allocator.dupe(u8, "bob");
    items_b[1] = try allocator.dupe(u8, "charlie");
    items_b[2] = try allocator.dupe(u8, "eve");
    items_b[3] = try allocator.dupe(u8, "frank");

    var set_b = try RowKeySet.fromOwnedUnsorted(allocator, items_b);
    defer set_b.deinit(allocator);

    // Intersect: should be [bob, charlie]
    var result = try RowKeySet.intersect(allocator, set_a, set_b);
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 2), result.items.len);
    try testing.expectEqualSlices(u8, "bob", result.items[0]);
    try testing.expectEqualSlices(u8, "charlie", result.items[1]);
}

test "RowKeySet.intersect with disjoint sets" {
    const allocator = testing.allocator;

    // Create set A: [alice, bob]
    var items_a = try allocator.alloc([]u8, 2);
    defer allocator.free(items_a);
    items_a[0] = try allocator.dupe(u8, "alice");
    items_a[1] = try allocator.dupe(u8, "bob");

    var set_a = try RowKeySet.fromOwnedUnsorted(allocator, items_a);
    defer set_a.deinit(allocator);

    // Create set B: [charlie, david]
    var items_b = try allocator.alloc([]u8, 2);
    defer allocator.free(items_b);
    items_b[0] = try allocator.dupe(u8, "charlie");
    items_b[1] = try allocator.dupe(u8, "david");

    var set_b = try RowKeySet.fromOwnedUnsorted(allocator, items_b);
    defer set_b.deinit(allocator);

    // Intersect: should be empty
    var result = try RowKeySet.intersect(allocator, set_a, set_b);
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 0), result.items.len);
}

test "RowKeySet.intersect where one set is empty" {
    const allocator = testing.allocator;

    // Create set A: [alice, bob]
    var items_a = try allocator.alloc([]u8, 2);
    defer allocator.free(items_a);
    items_a[0] = try allocator.dupe(u8, "alice");
    items_a[1] = try allocator.dupe(u8, "bob");

    var set_a = try RowKeySet.fromOwnedUnsorted(allocator, items_a);
    defer set_a.deinit(allocator);

    // Create empty set B
    const items_b = try allocator.alloc([]u8, 0);
    defer allocator.free(items_b);

    var set_b = try RowKeySet.fromOwnedUnsorted(allocator, items_b);
    defer set_b.deinit(allocator);

    // Intersect: should be empty
    var result = try RowKeySet.intersect(allocator, set_a, set_b);
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 0), result.items.len);
}

test "RowKeySet.unionOf with partial overlap" {
    const allocator = testing.allocator;

    // Create set A: [alice, bob, charlie]
    var items_a = try allocator.alloc([]u8, 3);
    defer allocator.free(items_a);
    items_a[0] = try allocator.dupe(u8, "alice");
    items_a[1] = try allocator.dupe(u8, "bob");
    items_a[2] = try allocator.dupe(u8, "charlie");

    var set_a = try RowKeySet.fromOwnedUnsorted(allocator, items_a);
    defer set_a.deinit(allocator);

    // Create set B: [bob, charlie, david, eve]
    var items_b = try allocator.alloc([]u8, 4);
    defer allocator.free(items_b);
    items_b[0] = try allocator.dupe(u8, "bob");
    items_b[1] = try allocator.dupe(u8, "charlie");
    items_b[2] = try allocator.dupe(u8, "david");
    items_b[3] = try allocator.dupe(u8, "eve");

    var set_b = try RowKeySet.fromOwnedUnsorted(allocator, items_b);
    defer set_b.deinit(allocator);

    // Union: should be [alice, bob, charlie, david, eve]
    var result = try RowKeySet.unionOf(allocator, set_a, set_b);
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 5), result.items.len);
    try testing.expectEqualSlices(u8, "alice", result.items[0]);
    try testing.expectEqualSlices(u8, "bob", result.items[1]);
    try testing.expectEqualSlices(u8, "charlie", result.items[2]);
    try testing.expectEqualSlices(u8, "david", result.items[3]);
    try testing.expectEqualSlices(u8, "eve", result.items[4]);
}

test "RowKeySet.unionOf with disjoint sets" {
    const allocator = testing.allocator;

    // Create set A: [alice, bob]
    var items_a = try allocator.alloc([]u8, 2);
    defer allocator.free(items_a);
    items_a[0] = try allocator.dupe(u8, "alice");
    items_a[1] = try allocator.dupe(u8, "bob");

    var set_a = try RowKeySet.fromOwnedUnsorted(allocator, items_a);
    defer set_a.deinit(allocator);

    // Create set B: [charlie, david]
    var items_b = try allocator.alloc([]u8, 2);
    defer allocator.free(items_b);
    items_b[0] = try allocator.dupe(u8, "charlie");
    items_b[1] = try allocator.dupe(u8, "david");

    var set_b = try RowKeySet.fromOwnedUnsorted(allocator, items_b);
    defer set_b.deinit(allocator);

    // Union: should be [alice, bob, charlie, david]
    var result = try RowKeySet.unionOf(allocator, set_a, set_b);
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 4), result.items.len);
    try testing.expectEqualSlices(u8, "alice", result.items[0]);
    try testing.expectEqualSlices(u8, "bob", result.items[1]);
    try testing.expectEqualSlices(u8, "charlie", result.items[2]);
    try testing.expectEqualSlices(u8, "david", result.items[3]);
}

test "RowKeySet.unionOf where one set is empty" {
    const allocator = testing.allocator;

    // Create set A: [alice, bob]
    var items_a = try allocator.alloc([]u8, 2);
    defer allocator.free(items_a);
    items_a[0] = try allocator.dupe(u8, "alice");
    items_a[1] = try allocator.dupe(u8, "bob");

    var set_a = try RowKeySet.fromOwnedUnsorted(allocator, items_a);
    defer set_a.deinit(allocator);

    // Create empty set B
    const items_b = try allocator.alloc([]u8, 0);
    defer allocator.free(items_b);

    var set_b = try RowKeySet.fromOwnedUnsorted(allocator, items_b);
    defer set_b.deinit(allocator);

    // Union: should equal set A
    var result = try RowKeySet.unionOf(allocator, set_a, set_b);
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 2), result.items.len);
    try testing.expectEqualSlices(u8, "alice", result.items[0]);
    try testing.expectEqualSlices(u8, "bob", result.items[1]);
}

test "RowKeySet.deinit does not leak memory" {
    const allocator = testing.allocator;

    // Create set with multiple items
    var items = try allocator.alloc([]u8, 3);
    defer allocator.free(items);

    items[0] = try allocator.dupe(u8, "alice");
    items[1] = try allocator.dupe(u8, "bob");
    items[2] = try allocator.dupe(u8, "charlie");

    var set = try RowKeySet.fromOwnedUnsorted(allocator, items);

    // Deinit will be called here; testing.allocator detects leaks on test end
    set.deinit(allocator);
}

test "RowKeySet with binary row_keys (not just text)" {
    const allocator = testing.allocator;

    // Create binary row_keys (simulating 8-byte big-endian u64 format)
    var items = try allocator.alloc([]u8, 3);
    defer allocator.free(items);

    // Row key 1: 0x0000000000000001
    items[0] = try allocator.dupe(u8, "\x00\x00\x00\x00\x00\x00\x00\x01");
    // Row key 2: 0x0000000000000002
    items[1] = try allocator.dupe(u8, "\x00\x00\x00\x00\x00\x00\x00\x02");
    // Row key 3: 0x0000000000000002 (duplicate)
    items[2] = try allocator.dupe(u8, "\x00\x00\x00\x00\x00\x00\x00\x02");

    var set = try RowKeySet.fromOwnedUnsorted(allocator, items);
    defer set.deinit(allocator);

    // Should be 2 unique items, sorted lexicographically
    try testing.expectEqual(@as(usize, 2), set.items.len);
    try testing.expectEqualSlices(u8, "\x00\x00\x00\x00\x00\x00\x00\x01", set.items[0]);
    try testing.expectEqualSlices(u8, "\x00\x00\x00\x00\x00\x00\x00\x02", set.items[1]);
}

test "RowKeySet.intersect creates fresh owned copies" {
    const allocator = testing.allocator;

    // Create set A
    var items_a = try allocator.alloc([]u8, 2);
    defer allocator.free(items_a);
    items_a[0] = try allocator.dupe(u8, "alice");
    items_a[1] = try allocator.dupe(u8, "bob");

    var set_a = try RowKeySet.fromOwnedUnsorted(allocator, items_a);
    defer set_a.deinit(allocator);

    // Create set B
    var items_b = try allocator.alloc([]u8, 2);
    defer allocator.free(items_b);
    items_b[0] = try allocator.dupe(u8, "bob");
    items_b[1] = try allocator.dupe(u8, "charlie");

    var set_b = try RowKeySet.fromOwnedUnsorted(allocator, items_b);
    defer set_b.deinit(allocator);

    // Intersect
    var result = try RowKeySet.intersect(allocator, set_a, set_b);
    defer result.deinit(allocator);

    // Result should have fresh owned copies (different pointers than originals)
    // but equal byte contents
    try testing.expectEqual(@as(usize, 1), result.items.len);
    try testing.expectEqualSlices(u8, "bob", result.items[0]);
    // Verify it's a fresh copy by checking pointer inequality
    try testing.expect(result.items[0].ptr != set_a.items[1].ptr);
}

test "RowKeySet.unionOf preserves original sets unchanged" {
    const allocator = testing.allocator;

    // Create set A
    var items_a = try allocator.alloc([]u8, 2);
    defer allocator.free(items_a);
    items_a[0] = try allocator.dupe(u8, "alice");
    items_a[1] = try allocator.dupe(u8, "bob");

    var set_a = try RowKeySet.fromOwnedUnsorted(allocator, items_a);
    defer set_a.deinit(allocator);

    // Create set B
    var items_b = try allocator.alloc([]u8, 2);
    defer allocator.free(items_b);
    items_b[0] = try allocator.dupe(u8, "bob");
    items_b[1] = try allocator.dupe(u8, "charlie");

    var set_b = try RowKeySet.fromOwnedUnsorted(allocator, items_b);
    defer set_b.deinit(allocator);

    // Store original lengths and contents
    const orig_a_len = set_a.items.len;
    const orig_b_len = set_b.items.len;

    // Union should not modify originals
    var result = try RowKeySet.unionOf(allocator, set_a, set_b);
    defer result.deinit(allocator);

    // Verify originals unchanged
    try testing.expectEqual(orig_a_len, set_a.items.len);
    try testing.expectEqual(orig_b_len, set_b.items.len);
    try testing.expectEqualSlices(u8, "alice", set_a.items[0]);
    try testing.expectEqualSlices(u8, "bob", set_a.items[1]);
}

// ── Type Definition ─────────────────────────────────────────────────────────

fn lessThanRowKey(_: void, a: []u8, b: []u8) bool {
    return std.mem.lessThan(u8, a, b);
}

/// RowKeySet — a sorted, deduped set of owned row_key byte slices.
pub const RowKeySet = struct {
    items: [][]u8,

    /// Takes ownership of items slice and all contained row_key slices,
    /// sorts them lexicographically, removes duplicates (freeing dropped copies),
    /// and returns a new sorted, deduped RowKeySet.
    pub fn fromOwnedUnsorted(allocator: Allocator, items: [][]u8) !RowKeySet {
        std.mem.sort([]u8, items, {}, lessThanRowKey);

        var write: usize = 0;
        var read: usize = 0;
        while (read < items.len) : (read += 1) {
            if (write > 0 and std.mem.eql(u8, items[write - 1], items[read])) {
                allocator.free(items[read]);
                continue;
            }
            items[write] = items[read];
            write += 1;
        }

        const deduped = try allocator.alloc([]u8, write);
        @memcpy(deduped, items[0..write]);
        return RowKeySet{ .items = deduped };
    }

    /// Computes the intersection of two RowKeySets via two-pointer algorithm.
    /// Returns a new RowKeySet with fresh owned copies of common elements.
    /// Does not consume a or b.
    pub fn intersect(allocator: Allocator, a: RowKeySet, b: RowKeySet) !RowKeySet {
        var result = std.ArrayList([]u8).empty;
        errdefer {
            for (result.items) |item| allocator.free(item);
            result.deinit(allocator);
        }

        var i: usize = 0;
        var j: usize = 0;
        while (i < a.items.len and j < b.items.len) {
            const order = std.mem.order(u8, a.items[i], b.items[j]);
            switch (order) {
                .lt => i += 1,
                .gt => j += 1,
                .eq => {
                    try result.append(allocator, try allocator.dupe(u8, a.items[i]));
                    i += 1;
                    j += 1;
                },
            }
        }

        return RowKeySet{ .items = try result.toOwnedSlice(allocator) };
    }

    /// Computes the union of two RowKeySets via two-pointer algorithm.
    /// Returns a new RowKeySet with fresh owned copies of union elements, deduped.
    /// Does not consume a or b.
    pub fn unionOf(allocator: Allocator, a: RowKeySet, b: RowKeySet) !RowKeySet {
        var result = std.ArrayList([]u8).empty;
        errdefer {
            for (result.items) |item| allocator.free(item);
            result.deinit(allocator);
        }

        var i: usize = 0;
        var j: usize = 0;
        while (i < a.items.len and j < b.items.len) {
            const order = std.mem.order(u8, a.items[i], b.items[j]);
            switch (order) {
                .lt => {
                    try result.append(allocator, try allocator.dupe(u8, a.items[i]));
                    i += 1;
                },
                .gt => {
                    try result.append(allocator, try allocator.dupe(u8, b.items[j]));
                    j += 1;
                },
                .eq => {
                    try result.append(allocator, try allocator.dupe(u8, a.items[i]));
                    i += 1;
                    j += 1;
                },
            }
        }
        while (i < a.items.len) : (i += 1) {
            try result.append(allocator, try allocator.dupe(u8, a.items[i]));
        }
        while (j < b.items.len) : (j += 1) {
            try result.append(allocator, try allocator.dupe(u8, b.items[j]));
        }

        return RowKeySet{ .items = try result.toOwnedSlice(allocator) };
    }

    /// Frees all owned row_key copies and internal storage.
    pub fn deinit(self: *RowKeySet, allocator: Allocator) void {
        for (self.items) |item| allocator.free(item);
        allocator.free(self.items);
        self.items = &.{};
    }
};
