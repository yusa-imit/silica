//! VACUUM — Dead tuple reclamation for MVCC.
//!
//! Scans table B+Trees for dead row versions that are no longer visible
//! to any active transaction, and removes them. This reclaims storage
//! space and prevents table bloat from accumulated dead tuples.
//!
//! A row version is "dead" when:
//!   1. Its xmin is aborted (inserting transaction rolled back), OR
//!   2. Its xmax is committed AND xmax < vacuum_horizon
//!      (deleting transaction committed and no active transaction can see it)
//!
//! The vacuum_horizon is the oldest active transaction's XID. Any row
//! version deleted by a transaction older than this is invisible to all
//! current and future transactions.

const std = @import("std");
const Allocator = std.mem.Allocator;
const mvcc_mod = @import("mvcc.zig");
const TupleHeader = mvcc_mod.TupleHeader;
const TransactionManager = mvcc_mod.TransactionManager;
const btree_mod = @import("../storage/btree.zig");
const BTree = btree_mod.BTree;
const Cursor = btree_mod.Cursor;
const buffer_pool_mod = @import("../storage/buffer_pool.zig");
const BufferPool = buffer_pool_mod.BufferPool;
const page_mod = @import("../storage/page.zig");
const Pager = page_mod.Pager;
const catalog_mod = @import("../sql/catalog.zig");
const Catalog = catalog_mod.Catalog;
const executor_mod = @import("../sql/executor.zig");
const Value = executor_mod.Value;

/// Result of a VACUUM operation on a single table.
pub const VacuumResult = struct {
    /// Number of dead tuples removed.
    tuples_removed: u64 = 0,
    /// Number of tuples frozen (xmin set to FROZEN_XID).
    tuples_frozen: u64 = 0,
    /// Number of tuples scanned.
    tuples_scanned: u64 = 0,
};

/// Determine if a versioned row is dead and should be vacuumed.
///
/// A tuple is dead if:
///   - xmin is aborted (row was inserted by a rolled-back transaction)
///   - xmin is committed AND xmax is committed AND xmax < vacuum_horizon
///     (row was deleted and no active transaction can see it)
pub fn isDeadTuple(header: TupleHeader, tm: *TransactionManager, vacuum_horizon: u32) bool {
    // Case 1: Inserting transaction aborted → dead
    if (header.flags.xmin_aborted) return true;
    if (!header.flags.xmin_committed) {
        // No hint flags — check TM
        if (tm.isAborted(header.xmin)) return true;
        // If xmin is still active, we can't touch this tuple
        if (!tm.isCommitted(header.xmin)) return false;
    }

    // xmin is committed — check xmax
    if (header.xmax == mvcc_mod.INVALID_XID) return false; // Still live

    // Case 2: Deleting transaction aborted → not dead (delete was rolled back)
    if (header.flags.xmax_aborted) return false;
    if (!header.flags.xmax_committed) {
        if (tm.isAborted(header.xmax)) return false;
        if (!tm.isCommitted(header.xmax)) return false; // Delete still in progress
    }

    // xmax is committed — dead if xmax < vacuum_horizon
    // (no active transaction can see the pre-delete version)
    return header.xmax < vacuum_horizon;
}

/// Check if a tuple can be frozen (xmin set to FROZEN_XID for permanent visibility).
///
/// A tuple can be frozen when:
///   - It is committed (xmin committed)
///   - It is not deleted (xmax == INVALID_XID)
///   - Its xmin < vacuum_horizon (all transactions can see it)
///   - It is not already frozen
pub fn canFreezeTuple(header: TupleHeader, tm: *TransactionManager, vacuum_horizon: u32) bool {
    if (header.isFrozen()) return false;
    if (header.xmax != mvcc_mod.INVALID_XID) return false;

    // Check xmin is committed
    if (header.flags.xmin_aborted) return false;
    if (!header.flags.xmin_committed and !tm.isCommitted(header.xmin)) return false;

    return header.xmin < vacuum_horizon;
}

/// Vacuum a single table: scan all rows, remove dead tuples, freeze old ones.
///
/// This directly modifies the B+Tree by:
///   1. Deleting dead tuples (aborted inserts, committed deletes below horizon)
///   2. Freezing old committed tuples (replacing xmin with FROZEN_XID)
///   3. Maintaining secondary indexes for deleted rows
///
/// Returns statistics about the operation.
pub fn vacuumTable(
    allocator: Allocator,
    pool: *BufferPool,
    table_root_page_id: u32,
    tm: *TransactionManager,
    table_info: *catalog_mod.TableInfo,
) !VacuumResult {
    var result = VacuumResult{};
    const vacuum_horizon = tm.getVacuumHorizon();

    var tree = BTree.init(pool, table_root_page_id);

    // Phase 1: Scan and collect dead tuple keys
    var dead_keys = std.ArrayListUnmanaged([]u8){};
    defer {
        for (dead_keys.items) |k| allocator.free(k);
        dead_keys.deinit(allocator);
    }

    // Also collect keys of tuples to freeze, with their row data
    const FreezeEntry = struct { key: []u8, data: []u8 };
    var freeze_entries = std.ArrayListUnmanaged(FreezeEntry){};
    defer {
        for (freeze_entries.items) |f| {
            allocator.free(f.key);
            allocator.free(f.data);
        }
        freeze_entries.deinit(allocator);
    }

    // Dead tuple values for index maintenance
    var dead_values = std.ArrayListUnmanaged([]Value){};
    defer {
        for (dead_values.items) |vals| {
            for (vals) |v| v.free(allocator);
            allocator.free(vals);
        }
        dead_values.deinit(allocator);
    }

    {
        var cursor = Cursor.init(allocator, &tree);
        defer cursor.deinit();
        try cursor.seekFirst();

        while (try cursor.next()) |entry| {
            defer allocator.free(entry.value);
            result.tuples_scanned += 1;

            // Only process MVCC-versioned rows
            if (!mvcc_mod.isVersionedRow(entry.value)) {
                allocator.free(entry.key);
                continue;
            }

            const header = TupleHeader.deserialize(
                entry.value[1..][0..mvcc_mod.TUPLE_HEADER_SIZE],
            );

            if (isDeadTuple(header, tm, vacuum_horizon)) {
                // Collect key for deletion
                try dead_keys.append(allocator, entry.key);

                // Deserialize values for index maintenance
                const values = executor_mod.deserializeRow(
                    allocator,
                    entry.value[mvcc_mod.MVCC_ROW_OVERHEAD..],
                ) catch {
                    continue;
                };
                try dead_values.append(allocator, values);

                result.tuples_removed += 1;
            } else if (canFreezeTuple(header, tm, vacuum_horizon)) {
                // Create frozen version of the row
                var frozen_data = try allocator.dupe(u8, entry.value);
                errdefer allocator.free(frozen_data);

                // Modify the header in-place: set xmin to FROZEN_XID, set committed flag
                var new_header = header;
                new_header.xmin = mvcc_mod.FROZEN_XID;
                new_header.flags.xmin_committed = true;
                new_header.flags.xmin_aborted = false;
                new_header.serialize(frozen_data[1..][0..mvcc_mod.TUPLE_HEADER_SIZE]);

                try freeze_entries.append(allocator, .{
                    .key = entry.key,
                    .data = frozen_data,
                });

                result.tuples_frozen += 1;
            } else {
                allocator.free(entry.key);
            }
        }
    }

    // Phase 2: Delete dead tuples from B+Tree and indexes
    for (dead_keys.items, 0..) |key, idx| {
        // Remove index entries for this dead row
        if (idx < dead_values.items.len) {
            deleteIndexEntries(allocator, pool, table_info, dead_values.items[idx]);
        }

        tree.delete(key) catch {};
    }

    // Phase 3: Freeze old tuples (delete + re-insert with frozen header)
    for (freeze_entries.items) |entry| {
        tree.delete(entry.key) catch {};
        tree.insert(entry.key, entry.data) catch {};
    }

    // Update table root page ID if it changed due to B+Tree operations
    if (tree.root_page_id != table_root_page_id) {
        table_info.data_root_page_id = tree.root_page_id;
    }

    return result;
}

/// Remove index entries for a deleted row's values.
fn deleteIndexEntries(
    allocator: Allocator,
    pool: *BufferPool,
    table_info: *catalog_mod.TableInfo,
    values: []const Value,
) void {
    for (table_info.indexes) |idx| {
        if (idx.column_index >= values.len) continue;

        const idx_key = valueToIndexKey(allocator, values[idx.column_index]) catch continue;
        defer allocator.free(idx_key);

        var idx_tree = BTree.init(pool, idx.root_page_id);
        idx_tree.delete(idx_key) catch {};
    }
}

/// Encode a Value as an index key for B+Tree storage.
fn valueToIndexKey(allocator: Allocator, val: Value) ![]u8 {
    return switch (val) {
        .integer => |i| {
            const buf = try allocator.alloc(u8, 8);
            const unsigned: u64 = @bitCast(i);
            const flipped = unsigned ^ (@as(u64, 1) << 63);
            std.mem.writeInt(u64, buf[0..8], flipped, .big);
            return buf;
        },
        .text => |t| try allocator.dupe(u8, t),
        .real => |r| {
            const buf = try allocator.alloc(u8, 8);
            const bits: u64 = @bitCast(r);
            const flipped = if (r >= 0) bits ^ (@as(u64, 1) << 63) else ~bits;
            std.mem.writeInt(u64, buf[0..8], flipped, .big);
            return buf;
        },
        .boolean => |b| {
            const buf = try allocator.alloc(u8, 1);
            buf[0] = if (b) 1 else 0;
            return buf;
        },
        .null_value => try allocator.alloc(u8, 0),
        .blob => |b| try allocator.dupe(u8, b),
    };
}

// ── Tests ──────────────────────────────────────────────────────────────

test "isDeadTuple — aborted xmin with hint flag" {
    const allocator = std.testing.allocator;
    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    const header = TupleHeader{
        .xmin = 5,
        .xmax = mvcc_mod.INVALID_XID,
        .cid = 0,
        .flags = .{ .xmin_aborted = true },
    };
    try std.testing.expect(isDeadTuple(header, &tm, 10));
}

test "isDeadTuple — aborted xmin via TM lookup" {
    const allocator = std.testing.allocator;
    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    const xid = try tm.begin(.read_committed);
    try tm.abort(xid);

    const header = TupleHeader{
        .xmin = xid,
        .xmax = mvcc_mod.INVALID_XID,
        .cid = 0,
        .flags = .{}, // no hints
    };
    try std.testing.expect(isDeadTuple(header, &tm, 10));
}

test "isDeadTuple — committed delete below horizon" {
    const allocator = std.testing.allocator;
    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    const xid1 = try tm.begin(.read_committed);
    const xid2 = try tm.begin(.read_committed);
    try tm.commit(xid1);
    try tm.commit(xid2);

    const header = TupleHeader{
        .xmin = xid1,
        .xmax = xid2,
        .cid = 0,
        .flags = .{ .xmin_committed = true, .xmax_committed = true },
    };

    // vacuum_horizon = next_xid (no active txns), both below → dead
    const horizon = tm.getVacuumHorizon();
    try std.testing.expect(isDeadTuple(header, &tm, horizon));
}

test "isDeadTuple — committed delete above horizon" {
    const allocator = std.testing.allocator;
    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    const xid1 = try tm.begin(.read_committed);
    const xid2 = try tm.begin(.read_committed);
    try tm.commit(xid1);
    // xid2 still active — horizon = xid2

    var header = TupleHeader.forInsert(xid1, 0);
    header.flags.xmin_committed = true;
    header.markDeleted(xid2, 0);
    // xmax = xid2 which is still active → not dead
    const horizon = tm.getVacuumHorizon();
    try std.testing.expect(!isDeadTuple(header, &tm, horizon));

    try tm.commit(xid2);
}

test "isDeadTuple — live tuple (no xmax)" {
    const allocator = std.testing.allocator;
    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    const header = TupleHeader{
        .xmin = 3,
        .xmax = mvcc_mod.INVALID_XID,
        .cid = 0,
        .flags = .{ .xmin_committed = true },
    };
    try std.testing.expect(!isDeadTuple(header, &tm, 10));
}

test "isDeadTuple — aborted delete (xmax aborted)" {
    const allocator = std.testing.allocator;
    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    const header = TupleHeader{
        .xmin = 3,
        .xmax = 5,
        .cid = 0,
        .flags = .{ .xmin_committed = true, .xmax_aborted = true },
    };
    // Delete was rolled back → tuple is live, not dead
    try std.testing.expect(!isDeadTuple(header, &tm, 10));
}

test "isDeadTuple — active xmin (in-progress insert)" {
    const allocator = std.testing.allocator;
    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    const xid = try tm.begin(.read_committed);

    const header = TupleHeader{
        .xmin = xid,
        .xmax = mvcc_mod.INVALID_XID,
        .cid = 0,
        .flags = .{}, // no hints, active in TM
    };
    // Active insert — cannot vacuum
    try std.testing.expect(!isDeadTuple(header, &tm, 10));

    try tm.commit(xid);
}

test "isDeadTuple — active delete (xmax active)" {
    const allocator = std.testing.allocator;
    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    const xid = try tm.begin(.read_committed);

    const header = TupleHeader{
        .xmin = 3,
        .xmax = xid,
        .cid = 0,
        .flags = .{ .xmin_committed = true }, // no xmax hints
    };
    // Delete in progress — not dead yet
    try std.testing.expect(!isDeadTuple(header, &tm, 10));

    try tm.commit(xid);
}

test "canFreezeTuple — eligible for freezing" {
    const allocator = std.testing.allocator;
    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    const header = TupleHeader{
        .xmin = 3,
        .xmax = mvcc_mod.INVALID_XID,
        .cid = 0,
        .flags = .{ .xmin_committed = true },
    };
    // xmin=3 < vacuum_horizon=10, committed, not deleted → can freeze
    try std.testing.expect(canFreezeTuple(header, &tm, 10));
}

test "canFreezeTuple — already frozen" {
    const allocator = std.testing.allocator;
    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    const header = TupleHeader{
        .xmin = mvcc_mod.FROZEN_XID,
        .xmax = mvcc_mod.INVALID_XID,
        .cid = 0,
        .flags = .{},
    };
    try std.testing.expect(!canFreezeTuple(header, &tm, 10));
}

test "canFreezeTuple — deleted tuple" {
    const allocator = std.testing.allocator;
    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    const header = TupleHeader{
        .xmin = 3,
        .xmax = 5,
        .cid = 0,
        .flags = .{ .xmin_committed = true },
    };
    // Has xmax → can't freeze (it's being deleted)
    try std.testing.expect(!canFreezeTuple(header, &tm, 10));
}

test "canFreezeTuple — xmin above horizon" {
    const allocator = std.testing.allocator;
    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    const header = TupleHeader{
        .xmin = 15,
        .xmax = mvcc_mod.INVALID_XID,
        .cid = 0,
        .flags = .{ .xmin_committed = true },
    };
    // xmin=15 >= vacuum_horizon=10 → can't freeze yet
    try std.testing.expect(!canFreezeTuple(header, &tm, 10));
}

test "canFreezeTuple — aborted xmin" {
    const allocator = std.testing.allocator;
    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    const header = TupleHeader{
        .xmin = 3,
        .xmax = mvcc_mod.INVALID_XID,
        .cid = 0,
        .flags = .{ .xmin_aborted = true },
    };
    // Aborted insert → should be dead, not frozen
    try std.testing.expect(!canFreezeTuple(header, &tm, 10));
}

/// Helper: allocate and initialize a root leaf page for a fresh B+Tree.
fn initTestTree(pager: *Pager, pool: *BufferPool) !BTree {
    const root_id = try pager.allocPage();
    const raw = try pager.allocPageBuf();
    defer pager.freePageBuf(raw);
    btree_mod.initLeafPage(raw, pager.page_size, root_id);
    try pager.writePage(root_id, raw);
    return BTree.init(pool, root_id);
}

test "vacuumTable — removes dead tuples from aborted transactions" {
    const allocator = std.testing.allocator;

    // Set up storage: pager + buffer pool
    const test_path = "test_vacuum_dead_aborted.db";
    defer std.fs.cwd().deleteFile(test_path) catch {};

    const pager = try allocator.create(Pager);
    defer allocator.destroy(pager);
    pager.* = try Pager.init(allocator, test_path, .{});
    defer pager.deinit();

    const pool = try allocator.create(BufferPool);
    defer allocator.destroy(pool);
    pool.* = try BufferPool.init(allocator, pager, 100);
    defer pool.deinit();

    // Create a simple table in the B+Tree
    var tree = try initTestTree(pager, pool);

    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    // Insert some rows: 2 committed, 1 from aborted transaction
    const xid1 = try tm.begin(.read_committed);
    const xid2 = try tm.begin(.read_committed);

    // Row 1: committed insert (xid1)
    {
        const header = TupleHeader.forInsert(xid1, 0);
        const vals = &[_]Value{.{ .integer = 100 }};
        const data = try mvcc_mod.serializeVersionedRow(allocator, header, vals);
        defer allocator.free(data);
        var key_buf: [8]u8 = undefined;
        std.mem.writeInt(u64, &key_buf, 1, .big);
        try tree.insert(&key_buf, data);
    }

    // Row 2: aborted insert (xid2)
    {
        const header = TupleHeader.forInsert(xid2, 0);
        const vals = &[_]Value{.{ .integer = 200 }};
        const data = try mvcc_mod.serializeVersionedRow(allocator, header, vals);
        defer allocator.free(data);
        var key_buf: [8]u8 = undefined;
        std.mem.writeInt(u64, &key_buf, 2, .big);
        try tree.insert(&key_buf, data);
    }

    // Row 3: committed insert (xid1)
    {
        const header = TupleHeader.forInsert(xid1, 1);
        const vals = &[_]Value{.{ .integer = 300 }};
        const data = try mvcc_mod.serializeVersionedRow(allocator, header, vals);
        defer allocator.free(data);
        var key_buf: [8]u8 = undefined;
        std.mem.writeInt(u64, &key_buf, 3, .big);
        try tree.insert(&key_buf, data);
    }

    try tm.commit(xid1);
    try tm.abort(xid2);

    // Create a minimal TableInfo (no indexes)
    var table_info = catalog_mod.TableInfo{
        .name = "test",
        .columns = &.{},
        .table_constraints = &.{},
        .data_root_page_id = tree.root_page_id,
    };

    // Vacuum should remove the aborted row
    const result = try vacuumTable(allocator, pool, tree.root_page_id, &tm, &table_info);

    try std.testing.expectEqual(@as(u64, 3), result.tuples_scanned);
    try std.testing.expectEqual(@as(u64, 1), result.tuples_removed);
    // The 2 committed rows should be frozen (xid1 < vacuum_horizon)
    try std.testing.expectEqual(@as(u64, 2), result.tuples_frozen);
}

test "vacuumTable — removes committed deletes below horizon" {
    const allocator = std.testing.allocator;

    const test_path = "test_vacuum_dead_deleted.db";
    defer std.fs.cwd().deleteFile(test_path) catch {};

    const pager = try allocator.create(Pager);
    defer allocator.destroy(pager);
    pager.* = try Pager.init(allocator, test_path, .{});
    defer pager.deinit();

    const pool = try allocator.create(BufferPool);
    defer allocator.destroy(pool);
    pool.* = try BufferPool.init(allocator, pager, 100);
    defer pool.deinit();

    var tree = try initTestTree(pager, pool);

    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    const xid1 = try tm.begin(.read_committed);
    const xid2 = try tm.begin(.read_committed);

    // Insert a row (xid1), then delete it (xid2)
    {
        var header = TupleHeader.forInsert(xid1, 0);
        header.flags.xmin_committed = true;
        header.markDeleted(xid2, 0);
        header.flags.xmax_committed = true;

        const vals = &[_]Value{.{ .integer = 42 }};
        const data = try mvcc_mod.serializeVersionedRow(allocator, header, vals);
        defer allocator.free(data);
        var key_buf: [8]u8 = undefined;
        std.mem.writeInt(u64, &key_buf, 1, .big);
        try tree.insert(&key_buf, data);
    }

    try tm.commit(xid1);
    try tm.commit(xid2);

    var table_info = catalog_mod.TableInfo{
        .name = "test",
        .columns = &.{},
        .table_constraints = &.{},
        .data_root_page_id = tree.root_page_id,
    };

    // Both xid1 and xid2 committed, no active txns → horizon = next_xid
    // xmax (xid2) < horizon → dead
    const result = try vacuumTable(allocator, pool, tree.root_page_id, &tm, &table_info);

    try std.testing.expectEqual(@as(u64, 1), result.tuples_scanned);
    try std.testing.expectEqual(@as(u64, 1), result.tuples_removed);
    try std.testing.expectEqual(@as(u64, 0), result.tuples_frozen);
}

test "vacuumTable — freezes old committed tuples" {
    const allocator = std.testing.allocator;

    const test_path = "test_vacuum_freeze.db";
    defer std.fs.cwd().deleteFile(test_path) catch {};

    const pager = try allocator.create(Pager);
    defer allocator.destroy(pager);
    pager.* = try Pager.init(allocator, test_path, .{});
    defer pager.deinit();

    const pool = try allocator.create(BufferPool);
    defer allocator.destroy(pool);
    pool.* = try BufferPool.init(allocator, pager, 100);
    defer pool.deinit();

    var tree = try initTestTree(pager, pool);

    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    const xid1 = try tm.begin(.read_committed);

    // Insert a committed, live row
    {
        var header = TupleHeader.forInsert(xid1, 0);
        header.flags.xmin_committed = true;

        const vals = &[_]Value{.{ .integer = 99 }};
        const data = try mvcc_mod.serializeVersionedRow(allocator, header, vals);
        defer allocator.free(data);
        var key_buf: [8]u8 = undefined;
        std.mem.writeInt(u64, &key_buf, 1, .big);
        try tree.insert(&key_buf, data);
    }

    try tm.commit(xid1);

    var table_info = catalog_mod.TableInfo{
        .name = "test",
        .columns = &.{},
        .table_constraints = &.{},
        .data_root_page_id = tree.root_page_id,
    };

    const result = try vacuumTable(allocator, pool, tree.root_page_id, &tm, &table_info);

    try std.testing.expectEqual(@as(u64, 1), result.tuples_scanned);
    try std.testing.expectEqual(@as(u64, 0), result.tuples_removed);
    try std.testing.expectEqual(@as(u64, 1), result.tuples_frozen);

    // Verify the tuple was frozen: read it back and check xmin
    var key_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &key_buf, 1, .big);

    const val = try tree.get(allocator, &key_buf);
    defer if (val) |v| allocator.free(v);

    if (val) |v| {
        try std.testing.expect(mvcc_mod.isVersionedRow(v));
        const hdr = TupleHeader.deserialize(v[1..][0..mvcc_mod.TUPLE_HEADER_SIZE]);
        try std.testing.expectEqual(mvcc_mod.FROZEN_XID, hdr.xmin);
        try std.testing.expect(hdr.flags.xmin_committed);
    } else {
        return error.TestUnexpectedResult;
    }
}

test "vacuumTable — skips legacy (non-MVCC) rows" {
    const allocator = std.testing.allocator;

    const test_path = "test_vacuum_legacy.db";
    defer std.fs.cwd().deleteFile(test_path) catch {};

    const pager = try allocator.create(Pager);
    defer allocator.destroy(pager);
    pager.* = try Pager.init(allocator, test_path, .{});
    defer pager.deinit();

    const pool = try allocator.create(BufferPool);
    defer allocator.destroy(pool);
    pool.* = try BufferPool.init(allocator, pager, 100);
    defer pool.deinit();

    var tree = try initTestTree(pager, pool);

    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    // Insert a legacy (non-MVCC) row
    {
        const vals = &[_]Value{.{ .integer = 77 }};
        const data = try executor_mod.serializeRow(allocator, vals);
        defer allocator.free(data);
        var key_buf: [8]u8 = undefined;
        std.mem.writeInt(u64, &key_buf, 1, .big);
        try tree.insert(&key_buf, data);
    }

    var table_info = catalog_mod.TableInfo{
        .name = "test",
        .columns = &.{},
        .table_constraints = &.{},
        .data_root_page_id = tree.root_page_id,
    };

    const result = try vacuumTable(allocator, pool, tree.root_page_id, &tm, &table_info);

    // Legacy rows should be scanned but not removed or frozen
    try std.testing.expectEqual(@as(u64, 1), result.tuples_scanned);
    try std.testing.expectEqual(@as(u64, 0), result.tuples_removed);
    try std.testing.expectEqual(@as(u64, 0), result.tuples_frozen);
}

test "vacuumTable — empty table" {
    const allocator = std.testing.allocator;

    const test_path = "test_vacuum_empty.db";
    defer std.fs.cwd().deleteFile(test_path) catch {};

    const pager = try allocator.create(Pager);
    defer allocator.destroy(pager);
    pager.* = try Pager.init(allocator, test_path, .{});
    defer pager.deinit();

    const pool = try allocator.create(BufferPool);
    defer allocator.destroy(pool);
    pool.* = try BufferPool.init(allocator, pager, 100);
    defer pool.deinit();

    const tree = try initTestTree(pager, pool);

    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    var table_info = catalog_mod.TableInfo{
        .name = "test",
        .columns = &.{},
        .table_constraints = &.{},
        .data_root_page_id = tree.root_page_id,
    };

    const result = try vacuumTable(allocator, pool, tree.root_page_id, &tm, &table_info);

    try std.testing.expectEqual(@as(u64, 0), result.tuples_scanned);
    try std.testing.expectEqual(@as(u64, 0), result.tuples_removed);
    try std.testing.expectEqual(@as(u64, 0), result.tuples_frozen);
}
