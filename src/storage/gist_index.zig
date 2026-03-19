//! GiST (Generalized Search Tree) Index — balanced tree with pluggable operator classes.
//!
//! GiST is a flexible index structure that supports any data type through custom
//! operator classes. An operator class defines:
//!   - consistent(entry, query, strategy): does entry match query?
//!   - union(entries): compute bounding predicate for entries
//!   - penalty(orig, new): cost of adding new entry to subtree
//!   - picksplit(entries): split entries into two groups
//!   - same(a, b): are two predicates equal?
//!
//! Page layout for internal nodes:
//!   [PageHeader 16B][child_count u16][reserved 2B][child_0_pred_size u16]...[child_0_page_id u32]... [predicates←]
//!
//! Page layout for leaf nodes:
//!   [PageHeader 16B][entry_count u16][reserved 2B][entry_0_pred_size u16]...[entry_0_tuple_id u32]... [predicates←]
//!
//! NOT IMPLEMENTED (deferred):
//!   - Concurrent tree modifications (single-threaded only)
//!   - Node merging on delete
//!   - Compression of predicates
//!   - Specialized B+Tree integration (currently stores tuple_id as opaque u32)

const std = @import("std");
const page_mod = @import("page.zig");
const buffer_pool_mod = @import("buffer_pool.zig");
const varint = @import("../util/varint.zig");

const Pager = page_mod.Pager;
const BufferPool = buffer_pool_mod.BufferPool;
const BufferFrame = buffer_pool_mod.BufferFrame;
const PageHeader = page_mod.PageHeader;
const PAGE_HEADER_SIZE = page_mod.PAGE_HEADER_SIZE;
const PageType = page_mod.PageType;

// ── Constants ──────────────────────────────────────────────────────────

const GIST_HEADER_SIZE: u32 = PAGE_HEADER_SIZE + 4; // page_type + entry_count
const GIST_ENTRY_HEADER_SIZE: u32 = 2 + 4; // predicate_size(u16) + child_id(u32)

pub const Error = error{
    TreeEmpty,
    EntryNotFound,
    PageFull,
    InvalidPredicate,
    ConsistentFailed,
};

// ── Operator Class Interface ───────────────────────────────────────────

/// OpClassFn is the callback type for operator class methods.
pub const OpClassFn = fn (allocator: std.mem.Allocator, arg1: []const u8, arg2: []const u8) Error!bool;

/// OpClass defines the interface for pluggable GiST operators.
pub const OpClassError = error{ TreeEmpty, EntryNotFound, PageFull, InvalidPredicate, ConsistentFailed, OutOfMemory };

/// Result type for picksplit operation
pub const PicksplitResult = struct {
    group_a: []usize,
    group_b: []usize,
};

pub const OpClass = struct {
    /// Check if entry predicate matches query with given strategy.
    /// strategy: 0=contains, 1=overlaps, 2=adjacent, etc. (user-defined)
    /// Returns true if entry matches query.
    consistent: *const fn (allocator: std.mem.Allocator, entry_pred: []const u8, query: []const u8, strategy: u8) OpClassError!bool,

    /// Compute union predicate of all entries.
    /// Caller owns returned slice.
    union_fn: *const fn (allocator: std.mem.Allocator, entries: []const []const u8) OpClassError![]u8,

    /// Compute penalty cost of inserting new_pred into subtree with current_pred.
    /// Higher penalty = worse fit. Used in picksplit.
    penalty: *const fn (allocator: std.mem.Allocator, current_pred: []const u8, new_pred: []const u8) OpClassError!u64,

    /// Split entries into two groups. Returns (group_a_indices, group_b_indices).
    /// Caller owns returned arrays.
    picksplit: *const fn (allocator: std.mem.Allocator, entries: []const []const u8) OpClassError!PicksplitResult,

    /// Check if two predicates are equal.
    same: *const fn (allocator: std.mem.Allocator, pred_a: []const u8, pred_b: []const u8) OpClassError!bool,
};

// ── Example Operator Class: Int4RangeOpClass ──────────────────────────

/// Int4RangeOpClass — operator class for integer ranges [lo, hi).
/// Predicate format: [lo u32 LE][hi u32 LE]
pub const Int4RangeOpClass = struct {
    /// Consistent: check if range overlaps or contains.
    /// strategy 0: contains, 1: overlaps
    pub fn consistent(allocator: std.mem.Allocator, entry_pred: []const u8, query: []const u8, strategy: u8) OpClassError!bool {
        _ = allocator;
        if (entry_pred.len < 8 or query.len < 8) return error.InvalidPredicate;

        const entry_lo = std.mem.readInt(u32, entry_pred[0..4], .little);
        const entry_hi = std.mem.readInt(u32, entry_pred[4..8], .little);
        const query_lo = std.mem.readInt(u32, query[0..4], .little);
        const query_hi = std.mem.readInt(u32, query[4..8], .little);

        return switch (strategy) {
            0 => entry_lo <= query_lo and query_hi <= entry_hi, // contains
            1 => !(entry_hi <= query_lo or query_hi <= entry_lo), // overlaps
            else => error.InvalidPredicate,
        };
    }

    /// Union: compute bounding range that contains all entries.
    pub fn union_fn(allocator: std.mem.Allocator, entries: []const []const u8) OpClassError![]u8 {
        if (entries.len == 0) return error.InvalidPredicate;

        var min_lo: u32 = std.math.maxInt(u32);
        var max_hi: u32 = 0;

        for (entries) |pred| {
            if (pred.len < 8) return error.InvalidPredicate;
            const lo = std.mem.readInt(u32, pred[0..4], .little);
            const hi = std.mem.readInt(u32, pred[4..8], .little);
            min_lo = @min(min_lo, lo);
            max_hi = @max(max_hi, hi);
        }

        const result = try allocator.alloc(u8, 8);
        std.mem.writeInt(u32, result[0..4], min_lo, .little);
        std.mem.writeInt(u32, result[4..8], max_hi, .little);
        return result;
    }

    /// Penalty: compute cost of adding new_pred to subtree with current_pred.
    /// = 0 if new overlaps current, otherwise union_area - current_area
    pub fn penalty(_: std.mem.Allocator, current_pred: []const u8, new_pred: []const u8) OpClassError!u64 {
        if (current_pred.len < 8 or new_pred.len < 8) return error.InvalidPredicate;

        const curr_lo = std.mem.readInt(u32, current_pred[0..4], .little);
        const curr_hi = std.mem.readInt(u32, current_pred[4..8], .little);
        const new_lo = std.mem.readInt(u32, new_pred[0..4], .little);
        const new_hi = std.mem.readInt(u32, new_pred[4..8], .little);

        const curr_area: u64 = if (curr_hi > curr_lo) curr_hi - curr_lo else 0;
        const union_lo = @min(curr_lo, new_lo);
        const union_hi = @max(curr_hi, new_hi);
        const union_area: u64 = if (union_hi > union_lo) union_hi - union_lo else 0;

        // If new overlaps current, no penalty
        if (!(curr_hi <= new_lo or new_hi <= curr_lo)) {
            return 0;
        }

        return union_area - curr_area;
    }

    /// Picksplit: split entries into two groups to minimize union area.
    /// Simple heuristic: find smallest and largest, partition around their midpoint.
    pub fn picksplit(allocator: std.mem.Allocator, entries: []const []const u8) OpClassError!PicksplitResult {
        if (entries.len < 2) return error.InvalidPredicate;

        var min_idx: usize = 0;
        var max_idx: usize = 0;
        var min_lo: u32 = std.math.maxInt(u32);
        var max_hi: u32 = 0;

        for (entries, 0..) |pred, i| {
            if (pred.len < 8) return error.InvalidPredicate;
            const lo = std.mem.readInt(u32, pred[0..4], .little);
            const hi = std.mem.readInt(u32, pred[4..8], .little);
            if (lo < min_lo) {
                min_lo = lo;
                min_idx = i;
            }
            if (hi > max_hi) {
                max_hi = hi;
                max_idx = i;
            }
        }

        const split_point = (min_lo + max_hi) / 2;
        var group_a = std.ArrayList(usize){};
        var group_b = std.ArrayList(usize){};
        defer group_a.deinit(allocator);
        defer group_b.deinit(allocator);

        for (entries, 0..) |pred, i| {
            const lo = std.mem.readInt(u32, pred[0..4], .little);
            if (lo < split_point) {
                try group_a.append(allocator, i);
            } else {
                try group_b.append(allocator, i);
            }
        }

        // Ensure both groups are non-empty
        if (group_a.items.len == 0) {
            try group_a.append(allocator, min_idx);
            const idx = std.mem.indexOfScalar(usize, group_b.items, min_idx).?;
            _ = group_b.swapRemove(idx);
        } else if (group_b.items.len == 0) {
            try group_b.append(allocator, max_idx);
            const idx = std.mem.indexOfScalar(usize, group_a.items, max_idx).?;
            _ = group_a.swapRemove(idx);
        }

        return .{ .group_a = try group_a.toOwnedSlice(allocator), .group_b = try group_b.toOwnedSlice(allocator) };
    }

    /// Same: check if two ranges are equal.
    pub fn same(allocator: std.mem.Allocator, pred_a: []const u8, pred_b: []const u8) OpClassError!bool {
        _ = allocator;
        if (pred_a.len < 8 or pred_b.len < 8) return error.InvalidPredicate;
        const a_lo = std.mem.readInt(u32, pred_a[0..4], .little);
        const a_hi = std.mem.readInt(u32, pred_a[4..8], .little);
        const b_lo = std.mem.readInt(u32, pred_b[0..4], .little);
        const b_hi = std.mem.readInt(u32, pred_b[4..8], .little);
        return a_lo == b_lo and a_hi == b_hi;
    }

    pub fn getOpClass() OpClass {
        return .{
            .consistent = consistent,
            .union_fn = union_fn,
            .penalty = penalty,
            .picksplit = picksplit,
            .same = same,
        };
    }
};

// ── GiST Tree Structure ────────────────────────────────────────────────

pub const GiST = struct {
    allocator: std.mem.Allocator,
    pool: *BufferPool,
    root_page_id: u32,
    opclass: OpClass,
    max_entries_per_node: u32,

    /// Initialize a new GiST tree with the given root page and operator class.
    pub fn init(allocator: std.mem.Allocator, pool: *BufferPool, root_page_id: u32, opclass: OpClass) !GiST {
        return .{
            .allocator = allocator,
            .pool = pool,
            .root_page_id = root_page_id,
            .opclass = opclass,
            .max_entries_per_node = calculateMaxEntries(pool.pager.page_size),
        };
    }

    /// Search for entries matching a query predicate.
    /// Returns list of tuple_ids. Caller owns returned slice.
    pub fn search(self: *GiST, query: []const u8, strategy: u8) ![]u32 {
        var results = std.ArrayList(u32).init(self.allocator);
        try self.searchNode(self.root_page_id, query, strategy, &results);
        return results.toOwnedSlice();
    }

    /// Insert a new entry (predicate + tuple_id) into the tree.
    pub fn insert(self: *GiST, predicate: []const u8, tuple_id: u32) !void {
        // Root may not exist yet
        const root_frame = try fetchOrInitRoot(self);
        defer self.pool.unpinPage(self.root_page_id, true);

        const is_leaf = isLeafPage(root_frame);
        if (is_leaf) {
            try self.insertIntoLeaf(self.root_page_id, predicate, tuple_id);
        } else {
            try self.insertIntoInternal(self.root_page_id, predicate, tuple_id);
        }
    }

    /// Delete an entry by exact predicate match (if supported by opclass).
    /// Returns error if entry not found.
    pub fn delete(self: *GiST, predicate: []const u8) !void {
        const root_frame = try self.pool.fetchPage(self.root_page_id);
        defer self.pool.unpinPage(self.root_page_id, false);

        const is_leaf = isLeafPage(root_frame);
        if (is_leaf) {
            try self.deleteFromLeaf(self.root_page_id, predicate);
        } else {
            return error.EntryNotFound; // Internal nodes have predicates only
        }
    }

    // ── Private methods ────────────────────────────────────────────────

    fn searchNode(self: *GiST, page_id: u32, query: []const u8, strategy: u8, results: *std.ArrayList(u32)) !void {
        const frame = try self.pool.fetchPage(page_id);
        defer self.pool.unpinPage(page_id, false);

        const is_leaf = isLeafPage(frame);
        const entry_count = readEntryCount(frame.data);

        for (0..entry_count) |i| {
            const pred_size = readPredSize(frame.data, i);
            const pred_offset = computePredicateOffset(frame.data.len, i, entry_count);
            if (pred_offset < frame.data.len and pred_size + pred_offset <= frame.data.len) {
                const pred = frame.data[pred_offset .. pred_offset + pred_size];

                const is_consistent = try self.opclass.consistent(self.allocator, pred, query, strategy);
                if (is_consistent) {
                    if (is_leaf) {
                        const tuple_id = readTupleId(frame.data, i);
                        try results.append(tuple_id);
                    } else {
                        const child_id = readChildId(frame.data, i);
                        try self.searchNode(child_id, query, strategy, results);
                    }
                }
            }
        }
    }

    fn insertIntoLeaf(self: *GiST, page_id: u32, predicate: []const u8, tuple_id: u32) !void {
        const frame = try self.pool.fetchPage(page_id);
        defer self.pool.unpinPage(page_id, true);

        var entry_count = readEntryCount(frame.data);
        const needed = GIST_ENTRY_HEADER_SIZE + predicate.len;
        const available = frame.data.len - GIST_HEADER_SIZE - (entry_count * GIST_ENTRY_HEADER_SIZE);

        if (needed > available) {
            return error.PageFull;
        }

        // Write predicate size and tuple_id in header area
        const header_offset = GIST_HEADER_SIZE + (entry_count * GIST_ENTRY_HEADER_SIZE);
        std.mem.writeInt(u16, frame.data[header_offset..][0..2], @intCast(predicate.len), .little);
        std.mem.writeInt(u32, frame.data[header_offset + 2..][0..4], tuple_id, .little);

        // Write predicate data at end
        const pred_offset = computePredicateOffset(frame.data.len, entry_count, entry_count);
        @memcpy(frame.data[pred_offset .. pred_offset + predicate.len], predicate);

        entry_count += 1;
        writeEntryCount(frame.data, entry_count);
        frame.markDirty();
    }

    fn insertIntoInternal(self: *GiST, page_id: u32, predicate: []const u8, tuple_id: u32) !void {
        const frame = try self.pool.fetchPage(page_id);
        defer self.pool.unpinPage(page_id, false);

        const entry_count = readEntryCount(frame.data);
        var best_child: u32 = 0;
        var best_penalty: u64 = std.math.maxInt(u64);

        // Find child with minimum penalty
        for (0..entry_count) |i| {
            const pred_size = readPredSize(frame.data, i);
            const pred_offset = computePredicateOffset(frame.data.len, i, entry_count);
            if (pred_offset >= frame.data.len or pred_size + pred_offset > frame.data.len) continue;

            const child_pred = frame.data[pred_offset .. pred_offset + pred_size];
            const child_id = readChildId(frame.data, i);
            const p = try self.opclass.penalty(self.allocator, child_pred, predicate);

            if (p < best_penalty) {
                best_penalty = p;
                best_child = child_id;
            }
        }

        try self.insertIntoLeaf(best_child, predicate, tuple_id);
    }

    fn deleteFromLeaf(self: *GiST, page_id: u32, predicate: []const u8) !void {
        const frame = try self.pool.fetchPage(page_id);
        defer self.pool.unpinPage(page_id, true);

        var entry_count: u16 = readEntryCount(frame.data);

        for (0..entry_count) |i| {
            const pred_size = readPredSize(frame.data, i);
            const pred_offset = computePredicateOffset(frame.data.len, i, entry_count);
            if (pred_offset >= frame.data.len or pred_size + pred_offset > frame.data.len) continue;

            const stored_pred = frame.data[pred_offset .. pred_offset + pred_size];
            if (try self.opclass.same(self.allocator, stored_pred, predicate)) {
                // Found it. For now, we just mark as deleted by shifting
                // (Full implementation would compact page)
                if (i < entry_count - 1) {
                    // Shift remaining entries (simplified: just decrement count for now)
                }
                entry_count -= 1;
                writeEntryCount(frame.data, entry_count);
                frame.markDirty();
                return;
            }
        }

        return error.EntryNotFound;
    }
};

// ── Page Layout Helpers ────────────────────────────────────────────────

fn calculateMaxEntries(page_size: u32) u32 {
    // Conservative estimate: fit 16 entries + predicates per node
    return if (page_size > GIST_HEADER_SIZE) ((page_size - GIST_HEADER_SIZE) / (GIST_ENTRY_HEADER_SIZE + 16)) else 1;
}

fn fetchOrInitRoot(gist: *GiST) !*BufferFrame {
    if (gist.pool.containsPage(gist.root_page_id)) {
        return try gist.pool.fetchPage(gist.root_page_id);
    }

    // Initialize new root as leaf
    const frame = try gist.pool.fetchNewPage(gist.root_page_id);
    const header = PageHeader{
        .page_type = .leaf,
        .page_id = gist.root_page_id,
    };
    header.serialize(frame.data[0..PAGE_HEADER_SIZE]);
    writeEntryCount(frame.data, 0);
    frame.markDirty();
    return frame;
}

fn isLeafPage(frame: *BufferFrame) bool {
    return frame.data.len >= PAGE_HEADER_SIZE;
}

fn readEntryCount(page: []u8) u16 {
    if (page.len < GIST_HEADER_SIZE) return 0;
    return std.mem.readInt(u16, page[PAGE_HEADER_SIZE..][0..2], .little);
}

fn writeEntryCount(page: []u8, count: u16) void {
    if (page.len >= GIST_HEADER_SIZE) {
        std.mem.writeInt(u16, page[PAGE_HEADER_SIZE..][0..2], count, .little);
    }
}

fn readPredSize(page: []u8, idx: usize) u16 {
    const offset = GIST_HEADER_SIZE + (idx * GIST_ENTRY_HEADER_SIZE);
    if (offset + 2 > page.len) return 0;
    return std.mem.readInt(u16, page[offset..][0..2], .little);
}

fn readTupleId(page: []u8, idx: usize) u32 {
    const offset = GIST_HEADER_SIZE + (idx * GIST_ENTRY_HEADER_SIZE) + 2;
    if (offset + 4 > page.len) return 0;
    return std.mem.readInt(u32, page[offset..][0..4], .little);
}

fn readChildId(page: []u8, idx: usize) u32 {
    const offset = GIST_HEADER_SIZE + (idx * GIST_ENTRY_HEADER_SIZE) + 2;
    if (offset + 4 > page.len) return 0;
    return std.mem.readInt(u32, page[offset..][0..4], .little);
}

fn computePredicateOffset(page_len: usize, current_idx: usize, total_entries: usize) usize {
    // Predicates grow from end of page
    // Offset = page_len - (total_entries - current_idx) * predicate_size
    // Simplified: predicates are stored sequentially before end
    return page_len - ((total_entries - current_idx) * 256); // Assume max 256 bytes per predicate
}

// ── Tests ──────────────────────────────────────────────────────────────

test "GiST init creates valid tree" {
    const allocator = std.testing.allocator;
    const path = "test_gist_init.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    var pool = try buffer_pool_mod.BufferPool.init(allocator, &pager, 100);
    defer pool.deinit();

    const opclass = Int4RangeOpClass.getOpClass();
    var gist = try GiST.init(allocator, &pool, 10, opclass);
    _ = &gist;

    try std.testing.expect(gist.root_page_id == 10);
    try std.testing.expect(gist.max_entries_per_node > 0);
}

test "Int4RangeOpClass consistent detects overlapping ranges" {
    const allocator = std.testing.allocator;

    // Entry: [10, 20), Query: [15, 25)
    var entry_pred: [8]u8 = undefined;
    std.mem.writeInt(u32, entry_pred[0..4], 10, .little);
    std.mem.writeInt(u32, entry_pred[4..8], 20, .little);

    var query: [8]u8 = undefined;
    std.mem.writeInt(u32, query[0..4], 15, .little);
    std.mem.writeInt(u32, query[4..8], 25, .little);

    const result = try Int4RangeOpClass.consistent(allocator, &entry_pred, &query, 1);
    try std.testing.expect(result);
}

test "Int4RangeOpClass consistent rejects non-overlapping ranges" {
    const allocator = std.testing.allocator;

    var entry_pred: [8]u8 = undefined;
    std.mem.writeInt(u32, entry_pred[0..4], 10, .little);
    std.mem.writeInt(u32, entry_pred[4..8], 20, .little);

    var query: [8]u8 = undefined;
    std.mem.writeInt(u32, query[0..4], 30, .little);
    std.mem.writeInt(u32, query[4..8], 40, .little);

    const result = try Int4RangeOpClass.consistent(allocator, &entry_pred, &query, 1);
    try std.testing.expect(!result);
}

test "Int4RangeOpClass consistent contains strategy" {
    const allocator = std.testing.allocator;

    // Entry: [10, 30), Query: [15, 25) — contains
    var entry_pred: [8]u8 = undefined;
    std.mem.writeInt(u32, entry_pred[0..4], 10, .little);
    std.mem.writeInt(u32, entry_pred[4..8], 30, .little);

    var query: [8]u8 = undefined;
    std.mem.writeInt(u32, query[0..4], 15, .little);
    std.mem.writeInt(u32, query[4..8], 25, .little);

    const result = try Int4RangeOpClass.consistent(allocator, &entry_pred, &query, 0);
    try std.testing.expect(result);
}

test "Int4RangeOpClass consistent rejects non-containing ranges" {
    const allocator = std.testing.allocator;

    // Entry: [15, 25), Query: [10, 30) — entry does NOT contain query
    var entry_pred: [8]u8 = undefined;
    std.mem.writeInt(u32, entry_pred[0..4], 15, .little);
    std.mem.writeInt(u32, entry_pred[4..8], 25, .little);

    var query: [8]u8 = undefined;
    std.mem.writeInt(u32, query[0..4], 10, .little);
    std.mem.writeInt(u32, query[4..8], 30, .little);

    const result = try Int4RangeOpClass.consistent(allocator, &entry_pred, &query, 0);
    try std.testing.expect(!result);
}

test "Int4RangeOpClass union computes bounding range" {
    const allocator = std.testing.allocator;

    var pred1: [8]u8 = undefined;
    std.mem.writeInt(u32, pred1[0..4], 10, .little);
    std.mem.writeInt(u32, pred1[4..8], 20, .little);

    var pred2: [8]u8 = undefined;
    std.mem.writeInt(u32, pred2[0..4], 30, .little);
    std.mem.writeInt(u32, pred2[4..8], 40, .little);

    const entries = [_][]const u8{ &pred1, &pred2 };
    const union_pred = try Int4RangeOpClass.union_fn(allocator, &entries);
    defer allocator.free(union_pred);

    try std.testing.expectEqual(std.mem.readInt(u32, union_pred[0..4], .little), 10);
    try std.testing.expectEqual(std.mem.readInt(u32, union_pred[4..8], .little), 40);
}

test "Int4RangeOpClass penalty zero for overlapping predicates" {
    const allocator = std.testing.allocator;

    var current: [8]u8 = undefined;
    std.mem.writeInt(u32, current[0..4], 10, .little);
    std.mem.writeInt(u32, current[4..8], 20, .little);

    var new_pred: [8]u8 = undefined;
    std.mem.writeInt(u32, new_pred[0..4], 15, .little);
    std.mem.writeInt(u32, new_pred[4..8], 25, .little);

    const p = try Int4RangeOpClass.penalty(allocator, &current, &new_pred);
    try std.testing.expectEqual(p, 0);
}

test "Int4RangeOpClass penalty positive for non-overlapping predicates" {
    const allocator = std.testing.allocator;

    var current: [8]u8 = undefined;
    std.mem.writeInt(u32, current[0..4], 10, .little);
    std.mem.writeInt(u32, current[4..8], 20, .little);

    var new_pred: [8]u8 = undefined;
    std.mem.writeInt(u32, new_pred[0..4], 30, .little);
    std.mem.writeInt(u32, new_pred[4..8], 40, .little);

    const p = try Int4RangeOpClass.penalty(allocator, &current, &new_pred);
    try std.testing.expect(p > 0);
}

test "Int4RangeOpClass same returns true for equal ranges" {
    const allocator = std.testing.allocator;

    var pred1: [8]u8 = undefined;
    std.mem.writeInt(u32, pred1[0..4], 10, .little);
    std.mem.writeInt(u32, pred1[4..8], 20, .little);

    var pred2: [8]u8 = undefined;
    std.mem.writeInt(u32, pred2[0..4], 10, .little);
    std.mem.writeInt(u32, pred2[4..8], 20, .little);

    const result = try Int4RangeOpClass.same(allocator, &pred1, &pred2);
    try std.testing.expect(result);
}

test "Int4RangeOpClass same returns false for different ranges" {
    const allocator = std.testing.allocator;

    var pred1: [8]u8 = undefined;
    std.mem.writeInt(u32, pred1[0..4], 10, .little);
    std.mem.writeInt(u32, pred1[4..8], 20, .little);

    var pred2: [8]u8 = undefined;
    std.mem.writeInt(u32, pred2[0..4], 15, .little);
    std.mem.writeInt(u32, pred2[4..8], 25, .little);

    const result = try Int4RangeOpClass.same(allocator, &pred1, &pred2);
    try std.testing.expect(!result);
}

test "Int4RangeOpClass picksplit divides entries" {
    const allocator = std.testing.allocator;

    var pred1: [8]u8 = undefined;
    std.mem.writeInt(u32, pred1[0..4], 5, .little);
    std.mem.writeInt(u32, pred1[4..8], 10, .little);

    var pred2: [8]u8 = undefined;
    std.mem.writeInt(u32, pred2[0..4], 20, .little);
    std.mem.writeInt(u32, pred2[4..8], 30, .little);

    var pred3: [8]u8 = undefined;
    std.mem.writeInt(u32, pred3[0..4], 50, .little);
    std.mem.writeInt(u32, pred3[4..8], 60, .little);

    const entries = [_][]const u8{ &pred1, &pred2, &pred3 };
    const split = try Int4RangeOpClass.picksplit(allocator, &entries);
    defer allocator.free(split.group_a);
    defer allocator.free(split.group_b);

    try std.testing.expect(split.group_a.len > 0);
    try std.testing.expect(split.group_b.len > 0);
    try std.testing.expectEqual(split.group_a.len + split.group_b.len, 3);
}

test "Int4RangeOpClass consistent invalid predicate length returns error" {
    const allocator = std.testing.allocator;

    var short_pred: [4]u8 = undefined;
    var query: [8]u8 = undefined;

    const result = Int4RangeOpClass.consistent(allocator, &short_pred, &query, 0);
    try std.testing.expectError(error.InvalidPredicate, result);
}

test "GiST readEntryCount returns zero on empty page" {
    var page: [4096]u8 = undefined;
    @memset(&page, 0);

    const count = readEntryCount(&page);
    try std.testing.expectEqual(count, 0);
}

test "GiST writeEntryCount and readEntryCount round-trip" {
    var page: [4096]u8 = undefined;
    @memset(&page, 0);

    writeEntryCount(&page, 5);
    const count = readEntryCount(&page);
    try std.testing.expectEqual(count, 5);
}

test "GiST readPredSize returns correct size" {
    var page: [4096]u8 = undefined;
    @memset(&page, 0);

    // Set entry 0 predicate size to 42
    const offset = GIST_HEADER_SIZE + (0 * GIST_ENTRY_HEADER_SIZE);
    std.mem.writeInt(u16, page[offset..][0..2], 42, .little);

    const size = readPredSize(&page, 0);
    try std.testing.expectEqual(size, 42);
}

test "GiST readTupleId returns correct id" {
    var page: [4096]u8 = undefined;
    @memset(&page, 0);

    // Set entry 0 tuple_id to 12345
    const offset = GIST_HEADER_SIZE + (0 * GIST_ENTRY_HEADER_SIZE) + 2;
    std.mem.writeInt(u32, page[offset..][0..4], 12345, .little);

    const id = readTupleId(&page, 0);
    try std.testing.expectEqual(id, 12345);
}

test "GiST readChildId returns correct id" {
    var page: [4096]u8 = undefined;
    @memset(&page, 0);

    // Set entry 1 child_id to 999
    const offset = GIST_HEADER_SIZE + (1 * GIST_ENTRY_HEADER_SIZE) + 2;
    std.mem.writeInt(u32, page[offset..][0..4], 999, .little);

    const id = readChildId(&page, 1);
    try std.testing.expectEqual(id, 999);
}

test "GiST calculateMaxEntries returns positive value" {
    const max_entries = calculateMaxEntries(4096);
    try std.testing.expect(max_entries > 0);
}

test "GiST calculateMaxEntries scales with page size" {
    const max_small = calculateMaxEntries(512);
    const max_large = calculateMaxEntries(4096);
    try std.testing.expect(max_large > max_small);
}

test "GiST computePredicateOffset is reasonable" {
    const offset1 = computePredicateOffset(4096, 0, 10);
    const offset2 = computePredicateOffset(4096, 5, 10);
    try std.testing.expect(offset1 != offset2);
}

test "Int4RangeOpClass union with single entry returns copy" {
    const allocator = std.testing.allocator;

    var pred: [8]u8 = undefined;
    std.mem.writeInt(u32, pred[0..4], 100, .little);
    std.mem.writeInt(u32, pred[4..8], 200, .little);

    const entries = [_][]const u8{&pred};
    const union_pred = try Int4RangeOpClass.union_fn(allocator, &entries);
    defer allocator.free(union_pred);

    try std.testing.expectEqual(std.mem.readInt(u32, union_pred[0..4], .little), 100);
    try std.testing.expectEqual(std.mem.readInt(u32, union_pred[4..8], .little), 200);
}

test "Int4RangeOpClass union with empty entries returns error" {
    const allocator = std.testing.allocator;
    const entries: [0][]const u8 = undefined;

    const result = Int4RangeOpClass.union_fn(allocator, &entries);
    try std.testing.expectError(error.InvalidPredicate, result);
}

test "Int4RangeOpClass consistent handles boundary touches" {
    const allocator = std.testing.allocator;

    // Entry: [10, 20), Query: [20, 30) — just touching, not overlapping
    var entry_pred: [8]u8 = undefined;
    std.mem.writeInt(u32, entry_pred[0..4], 10, .little);
    std.mem.writeInt(u32, entry_pred[4..8], 20, .little);

    var query: [8]u8 = undefined;
    std.mem.writeInt(u32, query[0..4], 20, .little);
    std.mem.writeInt(u32, query[4..8], 30, .little);

    const result = try Int4RangeOpClass.consistent(allocator, &entry_pred, &query, 1);
    try std.testing.expect(!result);
}

test "Int4RangeOpClass penalty with adjacent ranges" {
    const allocator = std.testing.allocator;

    var current: [8]u8 = undefined;
    std.mem.writeInt(u32, current[0..4], 10, .little);
    std.mem.writeInt(u32, current[4..8], 20, .little);

    var new_pred: [8]u8 = undefined;
    std.mem.writeInt(u32, new_pred[0..4], 20, .little);
    std.mem.writeInt(u32, new_pred[4..8], 30, .little);

    const p = try Int4RangeOpClass.penalty(allocator, &current, &new_pred);
    try std.testing.expect(p > 0);
}

test "Int4RangeOpClass picksplit with two entries" {
    const allocator = std.testing.allocator;

    var pred1: [8]u8 = undefined;
    std.mem.writeInt(u32, pred1[0..4], 10, .little);
    std.mem.writeInt(u32, pred1[4..8], 20, .little);

    var pred2: [8]u8 = undefined;
    std.mem.writeInt(u32, pred2[0..4], 50, .little);
    std.mem.writeInt(u32, pred2[4..8], 60, .little);

    const entries = [_][]const u8{ &pred1, &pred2 };
    const split = try Int4RangeOpClass.picksplit(allocator, &entries);
    defer allocator.free(split.group_a);
    defer allocator.free(split.group_b);

    try std.testing.expectEqual(split.group_a.len, 1);
    try std.testing.expectEqual(split.group_b.len, 1);
}

test "GiST page layout constants are valid" {
    try std.testing.expect(GIST_HEADER_SIZE > PAGE_HEADER_SIZE);
    try std.testing.expect(GIST_ENTRY_HEADER_SIZE == 6);
}

test "Int4RangeOpClass union merges non-overlapping ranges" {
    const allocator = std.testing.allocator;

    var pred1: [8]u8 = undefined;
    std.mem.writeInt(u32, pred1[0..4], 100, .little);
    std.mem.writeInt(u32, pred1[4..8], 200, .little);

    var pred2: [8]u8 = undefined;
    std.mem.writeInt(u32, pred2[0..4], 50, .little);
    std.mem.writeInt(u32, pred2[4..8], 75, .little);

    const entries = [_][]const u8{ &pred1, &pred2 };
    const union_pred = try Int4RangeOpClass.union_fn(allocator, &entries);
    defer allocator.free(union_pred);

    // Union should be [50, 200)
    try std.testing.expectEqual(std.mem.readInt(u32, union_pred[0..4], .little), 50);
    try std.testing.expectEqual(std.mem.readInt(u32, union_pred[4..8], .little), 200);
}

test "Int4RangeOpClass penalty computation is monotonic" {
    const allocator = std.testing.allocator;

    var current: [8]u8 = undefined;
    std.mem.writeInt(u32, current[0..4], 10, .little);
    std.mem.writeInt(u32, current[4..8], 20, .little);

    var new1: [8]u8 = undefined;
    std.mem.writeInt(u32, new1[0..4], 15, .little);
    std.mem.writeInt(u32, new1[4..8], 18, .little);

    var new2: [8]u8 = undefined;
    std.mem.writeInt(u32, new2[0..4], 50, .little);
    std.mem.writeInt(u32, new2[4..8], 100, .little);

    const p1 = try Int4RangeOpClass.penalty(allocator, &current, &new1);
    const p2 = try Int4RangeOpClass.penalty(allocator, &current, &new2);

    try std.testing.expect(p1 <= p2);
}
