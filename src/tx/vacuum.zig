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
const fsm_mod = @import("../storage/fsm.zig");
const FreeSpaceMap = fsm_mod.FreeSpaceMap;

/// Result of a VACUUM operation on a single table.
pub const VacuumResult = struct {
    /// Number of dead tuples removed.
    tuples_removed: u64 = 0,
    /// Number of tuples frozen (xmin set to FROZEN_XID).
    tuples_frozen: u64 = 0,
    /// Number of tuples scanned.
    tuples_scanned: u64 = 0,
    /// Number of leaf pages with updated free space info.
    pages_updated: u64 = 0,
    /// Total free space reclaimed (bytes, estimated via FSM categories).
    free_space_bytes: u64 = 0,
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
    fsm_opt: ?*FreeSpaceMap,
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

    // Phase 4: Update Free Space Map for modified leaf pages
    if (fsm_opt) |fsm| {
        updateFsmForTree(pool, &tree, fsm) catch {};
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

/// Walk a B+Tree's leaf pages and update FSM entries with current free space.
fn updateFsmForTree(pool: *BufferPool, tree: *BTree, fsm: *FreeSpaceMap) !void {
    const page_size = pool.pager.page_size;
    var page_id = tree.root_page_id;

    // Traverse to the leftmost leaf
    while (true) {
        const frame = try pool.fetchPage(page_id);
        defer pool.unpinPage(page_id, false);
        const header = page_mod.PageHeader.deserialize(frame.data[0..page_mod.PAGE_HEADER_SIZE]);

        if (header.page_type == .leaf) {
            break;
        } else if (header.page_type == .internal) {
            if (header.cell_count == 0) {
                page_id = btree_mod.getRightChild(frame.data);
            } else {
                const cell = btree_mod.readInternalCell(frame.data, page_size, 0);
                page_id = cell.left_child;
            }
        } else {
            return;
        }
    }

    // Walk leaf pages via next_leaf pointers
    while (page_id != 0) {
        const frame = try pool.fetchPage(page_id);
        defer pool.unpinPage(page_id, false);
        const header = page_mod.PageHeader.deserialize(frame.data[0..page_mod.PAGE_HEADER_SIZE]);

        if (header.page_type != .leaf) break;

        const free_space = btree_mod.leafFreeSpace(frame.data, page_size, header.cell_count);
        try fsm.update(page_id, free_space);

        // Get next leaf
        const next_leaf = btree_mod.getNextLeaf(frame.data);
        page_id = next_leaf;
    }
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
    const result = try vacuumTable(allocator, pool, tree.root_page_id, &tm, &table_info, null);

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
    const result = try vacuumTable(allocator, pool, tree.root_page_id, &tm, &table_info, null);

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

    const result = try vacuumTable(allocator, pool, tree.root_page_id, &tm, &table_info, null);

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

    const result = try vacuumTable(allocator, pool, tree.root_page_id, &tm, &table_info, null);

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

    const result = try vacuumTable(allocator, pool, tree.root_page_id, &tm, &table_info, null);

    try std.testing.expectEqual(@as(u64, 0), result.tuples_scanned);
    try std.testing.expectEqual(@as(u64, 0), result.tuples_removed);
    try std.testing.expectEqual(@as(u64, 0), result.tuples_frozen);
}

test "isDeadTuple — horizon boundary" {
    const allocator = std.testing.allocator;
    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    const xid1 = try tm.begin(.read_committed);
    const xid2 = try tm.begin(.read_committed);
    try tm.commit(xid1);
    try tm.commit(xid2);

    const vacuum_horizon: u32 = xid2; // horizon = xid2

    // xmax == horizon → NOT dead (xmax must be LESS THAN horizon)
    {
        const h = TupleHeader{
            .xmin = xid1,
            .xmax = xid2,
            .cid = 0,
            .flags = .{ .xmin_committed = true, .xmax_committed = true },
        };
        try std.testing.expect(!isDeadTuple(h, &tm, vacuum_horizon));
    }

    // xmax == horizon - 1 → dead
    {
        const h = TupleHeader{
            .xmin = xid1,
            .xmax = xid1, // xid1 < xid2 = horizon
            .cid = 0,
            .flags = .{ .xmin_committed = true, .xmax_committed = true },
        };
        try std.testing.expect(isDeadTuple(h, &tm, vacuum_horizon));
    }
}

test "isDeadTuple — already frozen tuple with xmax" {
    const allocator = std.testing.allocator;
    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    const xid1 = try tm.begin(.read_committed);
    try tm.commit(xid1);

    // Frozen xmin, committed xmax below horizon
    const h = TupleHeader{
        .xmin = mvcc_mod.FROZEN_XID,
        .xmax = xid1,
        .cid = 0,
        .flags = .{ .xmin_committed = true, .xmax_committed = true },
    };

    // xmin is FROZEN_XID (= BOOTSTRAP_XID = 1), which isCommitted returns true for
    // xmax is committed and < horizon → dead
    try std.testing.expect(isDeadTuple(h, &tm, tm.getVacuumHorizon()));
}

test "canFreezeTuple — already frozen returns false" {
    const allocator = std.testing.allocator;
    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    const h = TupleHeader{
        .xmin = mvcc_mod.FROZEN_XID,
        .xmax = mvcc_mod.INVALID_XID,
        .cid = 0,
        .flags = .{ .xmin_committed = true },
    };
    try std.testing.expect(!canFreezeTuple(h, &tm, 100));
}

test "canFreezeTuple — deleted tuple cannot be frozen" {
    const allocator = std.testing.allocator;
    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    const xid1 = try tm.begin(.read_committed);
    try tm.commit(xid1);

    // xmax is set (tuple is deleted) → cannot freeze
    const h = TupleHeader{
        .xmin = xid1,
        .xmax = xid1,
        .cid = 0,
        .flags = .{ .xmin_committed = true, .xmax_committed = true },
    };
    try std.testing.expect(!canFreezeTuple(h, &tm, tm.getVacuumHorizon()));
}

test "canFreezeTuple — aborted xmin cannot be frozen" {
    const allocator = std.testing.allocator;
    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    const xid1 = try tm.begin(.read_committed);
    try tm.abort(xid1);

    const h = TupleHeader{
        .xmin = xid1,
        .xmax = mvcc_mod.INVALID_XID,
        .cid = 0,
        .flags = .{ .xmin_aborted = true },
    };
    try std.testing.expect(!canFreezeTuple(h, &tm, tm.getVacuumHorizon()));
}

test "canFreezeTuple — xmin at horizon cannot be frozen" {
    const allocator = std.testing.allocator;
    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    const xid1 = try tm.begin(.read_committed);
    try tm.commit(xid1);

    // xmin == vacuum_horizon → cannot freeze (must be strictly less than)
    const h = TupleHeader{
        .xmin = xid1,
        .xmax = mvcc_mod.INVALID_XID,
        .cid = 0,
        .flags = .{ .xmin_committed = true },
    };
    // Set horizon to exactly xid1
    try std.testing.expect(!canFreezeTuple(h, &tm, xid1));
}

test "isDeadTuple — active xmin means not dead" {
    const allocator = std.testing.allocator;
    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    const xid1 = try tm.begin(.read_committed);

    // No hint flags, xmin is active → TM says not committed → not dead
    const h = TupleHeader{
        .xmin = xid1,
        .xmax = mvcc_mod.INVALID_XID,
        .cid = 0,
        .flags = .{},
    };
    try std.testing.expect(!isDeadTuple(h, &tm, tm.getVacuumHorizon()));

    try tm.commit(xid1);
}

test "isDeadTuple — xmax aborted means not dead" {
    const allocator = std.testing.allocator;
    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    const xid1 = try tm.begin(.read_committed);
    const xid2 = try tm.begin(.read_committed);
    try tm.commit(xid1);
    try tm.abort(xid2);

    // xmin committed, xmax aborted (without hint flags, TM consulted)
    const h = TupleHeader{
        .xmin = xid1,
        .xmax = xid2,
        .cid = 0,
        .flags = .{ .xmin_committed = true }, // no xmax flags
    };
    try std.testing.expect(!isDeadTuple(h, &tm, tm.getVacuumHorizon()));
}

test "vacuumTable — all tuples dead" {
    const allocator = std.testing.allocator;

    const test_path = "test_vacuum_all_dead.db";
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

    // Insert 3 rows, all deleted
    var i: u64 = 1;
    while (i <= 3) : (i += 1) {
        var header = TupleHeader.forInsert(xid1, 0);
        header.flags.xmin_committed = true;
        header.markDeleted(xid2, 0);
        header.flags.xmax_committed = true;

        const vals = &[_]Value{.{ .integer = @as(i64, @intCast(i)) }};
        const data = try mvcc_mod.serializeVersionedRow(allocator, header, vals);
        defer allocator.free(data);
        var key_buf: [8]u8 = undefined;
        std.mem.writeInt(u64, &key_buf, i, .big);
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

    const result = try vacuumTable(allocator, pool, tree.root_page_id, &tm, &table_info, null);

    try std.testing.expectEqual(@as(u64, 3), result.tuples_scanned);
    try std.testing.expectEqual(@as(u64, 3), result.tuples_removed);
    try std.testing.expectEqual(@as(u64, 0), result.tuples_frozen);
}

test "vacuumTable — all tuples live and freezable" {
    const allocator = std.testing.allocator;

    const test_path = "test_vacuum_all_live.db";
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

    // Insert 3 live rows (no xmax)
    var i: u64 = 1;
    while (i <= 3) : (i += 1) {
        var header = TupleHeader.forInsert(xid1, 0);
        header.flags.xmin_committed = true;

        const vals = &[_]Value{.{ .integer = @as(i64, @intCast(i * 10)) }};
        const data = try mvcc_mod.serializeVersionedRow(allocator, header, vals);
        defer allocator.free(data);
        var key_buf: [8]u8 = undefined;
        std.mem.writeInt(u64, &key_buf, i, .big);
        try tree.insert(&key_buf, data);
    }

    try tm.commit(xid1);

    var table_info = catalog_mod.TableInfo{
        .name = "test",
        .columns = &.{},
        .table_constraints = &.{},
        .data_root_page_id = tree.root_page_id,
    };

    const result = try vacuumTable(allocator, pool, tree.root_page_id, &tm, &table_info, null);

    try std.testing.expectEqual(@as(u64, 3), result.tuples_scanned);
    try std.testing.expectEqual(@as(u64, 0), result.tuples_removed);
    try std.testing.expectEqual(@as(u64, 3), result.tuples_frozen);

    // Verify all tuples are now frozen
    var j: u64 = 1;
    while (j <= 3) : (j += 1) {
        var key_buf: [8]u8 = undefined;
        std.mem.writeInt(u64, &key_buf, j, .big);
        const val = try tree.get(allocator, &key_buf);
        defer if (val) |v| allocator.free(v);
        if (val) |v| {
            const hdr = TupleHeader.deserialize(v[1..][0..mvcc_mod.TUPLE_HEADER_SIZE]);
            try std.testing.expectEqual(mvcc_mod.FROZEN_XID, hdr.xmin);
        } else {
            return error.TestUnexpectedResult;
        }
    }
}

test "vacuum updates FSM after dead tuple removal" {
    const allocator = std.testing.allocator;

    const test_path = "test_vacuum_fsm.db";
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

    var fsm = FreeSpaceMap.init(allocator, pager.page_size);
    defer fsm.deinit();

    // Insert rows: 3 committed + 3 from aborted transaction
    const xid1 = try tm.begin(.read_committed);
    const xid2 = try tm.begin(.read_committed);

    // 3 committed rows
    for (1..4) |i| {
        const header = TupleHeader.forInsert(xid1, @intCast(i - 1));
        const vals = &[_]Value{.{ .integer = @intCast(i * 100) }};
        const data = try mvcc_mod.serializeVersionedRow(allocator, header, vals);
        defer allocator.free(data);
        var key_buf: [8]u8 = undefined;
        std.mem.writeInt(u64, &key_buf, @as(u64, @intCast(i)), .big);
        try tree.insert(&key_buf, data);
    }

    // 3 aborted rows
    for (4..7) |i| {
        const header = TupleHeader.forInsert(xid2, @intCast(i - 4));
        const vals = &[_]Value{.{ .integer = @intCast(i * 100) }};
        const data = try mvcc_mod.serializeVersionedRow(allocator, header, vals);
        defer allocator.free(data);
        var key_buf: [8]u8 = undefined;
        std.mem.writeInt(u64, &key_buf, @as(u64, @intCast(i)), .big);
        try tree.insert(&key_buf, data);
    }

    try tm.commit(xid1);
    try tm.abort(xid2);

    // FSM should have no entries before vacuum
    try std.testing.expectEqual(@as(u8, 0), fsm.getCategory(tree.root_page_id));

    var table_info = catalog_mod.TableInfo{
        .name = "test",
        .columns = &.{},
        .table_constraints = &.{},
        .data_root_page_id = tree.root_page_id,
    };
    const result = try vacuumTable(allocator, pool, tree.root_page_id, &tm, &table_info, &fsm);

    // Should have removed 3 aborted rows
    try std.testing.expectEqual(@as(u64, 3), result.tuples_removed);

    // FSM should now have entries for leaf pages with free space
    try std.testing.expect(fsm.trackedPages() > 0);

    // The leaf page should have more free space after removing 3 rows
    const cat = fsm.getCategory(tree.root_page_id);
    try std.testing.expect(cat > 0);
}

test "vacuum FSM reflects free space correctly" {
    const allocator = std.testing.allocator;

    const test_path = "test_vacuum_fsm2.db";
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

    var fsm = FreeSpaceMap.init(allocator, pager.page_size);
    defer fsm.deinit();

    // All rows committed — vacuum should freeze but not remove
    const xid1 = try tm.begin(.read_committed);

    for (1..6) |i| {
        const header = TupleHeader.forInsert(xid1, @intCast(i - 1));
        const vals = &[_]Value{.{ .integer = @intCast(i * 100) }};
        const data = try mvcc_mod.serializeVersionedRow(allocator, header, vals);
        defer allocator.free(data);
        var key_buf: [8]u8 = undefined;
        std.mem.writeInt(u64, &key_buf, @as(u64, @intCast(i)), .big);
        try tree.insert(&key_buf, data);
    }

    try tm.commit(xid1);

    var table_info = catalog_mod.TableInfo{
        .name = "test",
        .columns = &.{},
        .table_constraints = &.{},
        .data_root_page_id = tree.root_page_id,
    };
    const result = try vacuumTable(allocator, pool, tree.root_page_id, &tm, &table_info, &fsm);

    // No removals, only freezing
    try std.testing.expectEqual(@as(u64, 0), result.tuples_removed);
    try std.testing.expectEqual(@as(u64, 5), result.tuples_frozen);

    // FSM should track the leaf page (it has free space since not full)
    try std.testing.expect(fsm.trackedPages() > 0);
    try std.testing.expect(fsm.totalFreeSpace() > 0);
}
