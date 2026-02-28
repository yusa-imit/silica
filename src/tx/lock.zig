//! Lock Manager — Row-level and table-level locking for MVCC transactions.
//!
//! Implements PostgreSQL-compatible locking semantics:
//!   - Row-level locks: SHARED (FOR SHARE) and EXCLUSIVE (FOR UPDATE, DML)
//!   - Table-level locks: 7 modes (ACCESS SHARE → ACCESS EXCLUSIVE)
//!   - Conflict detection: prevents concurrent writers from modifying the same row
//!   - Lock release on transaction end (commit or rollback)
//!
//! Lock acquisition protocol:
//!   1. Table-level lock is acquired first (implicit on DML/SELECT)
//!   2. Row-level locks are acquired during tuple access
//!   3. All locks are released together on transaction end
//!
//! Currently single-threaded but designed for future multi-threaded use.

const std = @import("std");
const Allocator = std.mem.Allocator;

// ── Lock Modes ─────────────────────────────────────────────────────────

/// Row-level lock modes.
pub const LockMode = enum(u8) {
    /// Shared row lock — FOR SHARE (multiple holders allowed)
    shared,
    /// Exclusive row lock — FOR UPDATE, INSERT, UPDATE, DELETE (single holder)
    exclusive,

    /// Check if this mode conflicts with another mode.
    pub fn conflictsWith(self: LockMode, other: LockMode) bool {
        return switch (self) {
            .shared => other == .exclusive,
            .exclusive => true, // exclusive conflicts with all modes
        };
    }
};

/// Table-level lock modes (PostgreSQL-compatible).
pub const TableLockMode = enum(u8) {
    access_share,        // SELECT
    row_share,           // SELECT FOR SHARE
    row_exclusive,       // INSERT/UPDATE/DELETE
    share,               // CREATE INDEX
    share_row_exclusive, // ALTER TABLE (some variants)
    exclusive,           // VACUUM, some schema changes
    access_exclusive,    // DDL (DROP TABLE, ALTER TABLE ADD COLUMN)

    /// Check if this mode conflicts with another mode using PostgreSQL conflict matrix.
    pub fn conflictsWith(self: TableLockMode, other: TableLockMode) bool {
        const self_level: u8 = @intFromEnum(self);
        const other_level: u8 = @intFromEnum(other);

        // Conflict matrix (rows = self, columns = other)
        // AS  RS  RE  S   SRE  E   AE
        const matrix = [7][7]bool{
            // ACCESS SHARE
            .{ false, false, false, false, false, false, true },
            // ROW SHARE
            .{ false, false, false, false, false, true, true },
            // ROW EXCLUSIVE
            .{ false, false, false, true, true, true, true },
            // SHARE
            .{ false, false, true, false, true, true, true },
            // SHARE ROW EXCLUSIVE
            .{ false, false, true, true, true, true, true },
            // EXCLUSIVE
            .{ false, true, true, true, true, true, true },
            // ACCESS EXCLUSIVE
            .{ true, true, true, true, true, true, true },
        };

        return matrix[self_level][other_level];
    }
};

// ── Lock Target ────────────────────────────────────────────────────────

/// Identifies a specific row to be locked.
pub const LockTarget = struct {
    /// Root page ID of the table's B+Tree.
    table_page_id: u32,
    /// Row key in the B+Tree.
    row_key: u64,

    pub fn hash(self: LockTarget) u64 {
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(std.mem.asBytes(&self.table_page_id));
        hasher.update(std.mem.asBytes(&self.row_key));
        return hasher.final();
    }

    pub fn eql(a: LockTarget, b: LockTarget) bool {
        return a.table_page_id == b.table_page_id and a.row_key == b.row_key;
    }
};

// ── Lock Info ──────────────────────────────────────────────────────────

/// Information about a row-level lock.
pub const RowLockInfo = struct {
    mode: LockMode,
    /// XIDs holding this lock (shared locks can have multiple holders).
    holders: std.ArrayListUnmanaged(u32),

    pub fn init() RowLockInfo {
        return .{
            .mode = .shared,
            .holders = .{},
        };
    }

    pub fn deinit(self: *RowLockInfo, allocator: Allocator) void {
        self.holders.deinit(allocator);
    }

    /// Check if a specific XID holds this lock.
    pub fn isHeldBy(self: RowLockInfo, xid: u32) bool {
        for (self.holders.items) |holder| {
            if (holder == xid) return true;
        }
        return false;
    }

    /// Add a holder to this lock.
    pub fn addHolder(self: *RowLockInfo, allocator: Allocator, xid: u32) !void {
        // Check if already holding
        if (self.isHeldBy(xid)) return;
        try self.holders.append(allocator, xid);
    }

    /// Remove a holder from this lock. Returns true if lock is now empty.
    pub fn removeHolder(self: *RowLockInfo, xid: u32) bool {
        var idx: usize = 0;
        while (idx < self.holders.items.len) {
            if (self.holders.items[idx] == xid) {
                _ = self.holders.orderedRemove(idx);
                break;
            }
            idx += 1;
        }
        return self.holders.items.len == 0;
    }
};

/// Information about a table-level lock.
pub const TableLockEntry = struct {
    xid: u32,
    mode: TableLockMode,
};

pub const TableLockList = std.ArrayListUnmanaged(TableLockEntry);

// ── Lock Manager ───────────────────────────────────────────────────────

pub const LockError = error{
    LockConflict,
    DeadlockDetected,
    OutOfMemory,
};

/// Custom hash map context for LockTarget.
const LockTargetContext = struct {
    pub fn hash(_: LockTargetContext, key: LockTarget) u64 {
        return key.hash();
    }

    pub fn eql(_: LockTargetContext, a: LockTarget, b: LockTarget) bool {
        return a.eql(b);
    }
};

pub const LockManager = struct {
    allocator: Allocator,
    /// Row-level locks: LockTarget → RowLockInfo
    row_locks: std.HashMapUnmanaged(
        LockTarget,
        RowLockInfo,
        LockTargetContext,
        std.hash_map.default_max_load_percentage,
    ),
    /// Table-level locks: table_page_id → list of (xid, mode)
    table_locks: std.AutoHashMapUnmanaged(u32, TableLockList),

    pub fn init(allocator: Allocator) LockManager {
        return .{
            .allocator = allocator,
            .row_locks = .{},
            .table_locks = .{},
        };
    }

    pub fn deinit(self: *LockManager) void {
        // Free row lock holders
        var row_it = self.row_locks.iterator();
        while (row_it.next()) |entry| {
            entry.value_ptr.deinit(self.allocator);
        }
        self.row_locks.deinit(self.allocator);

        // Free table lock lists
        var table_it = self.table_locks.iterator();
        while (table_it.next()) |entry| {
            entry.value_ptr.deinit(self.allocator);
        }
        self.table_locks.deinit(self.allocator);
    }

    // ── Row-level Locks ────────────────────────────────────────────

    /// Acquire a row-level lock. Returns error if a conflicting lock exists.
    pub fn acquireRowLock(
        self: *LockManager,
        xid: u32,
        target: LockTarget,
        mode: LockMode,
    ) LockError!void {
        // Check for existing lock
        if (self.row_locks.getPtr(target)) |existing| {
            // Check if current transaction already holds this lock
            if (existing.isHeldBy(xid)) {
                // Lock upgrade: shared → exclusive
                if (existing.mode == .shared and mode == .exclusive) {
                    // Can only upgrade if we are the sole holder
                    if (existing.holders.items.len == 1) {
                        existing.mode = .exclusive;
                        return;
                    } else {
                        return LockError.LockConflict;
                    }
                }
                // Already holds compatible lock
                return;
            }

            // Check conflict with existing lock
            if (existing.mode.conflictsWith(mode) or mode.conflictsWith(existing.mode)) {
                return LockError.LockConflict;
            }

            // Compatible lock (both shared) — add this XID as holder
            try existing.addHolder(self.allocator, xid);
        } else {
            // No existing lock — create new entry
            var info = RowLockInfo.init();
            info.mode = mode;
            try info.addHolder(self.allocator, xid);
            try self.row_locks.put(self.allocator, target, info);
        }
    }

    /// Release a specific row-level lock.
    pub fn releaseRowLock(self: *LockManager, xid: u32, target: LockTarget) void {
        if (self.row_locks.getPtr(target)) |info| {
            const is_empty = info.removeHolder(xid);
            if (is_empty) {
                var removed = self.row_locks.fetchRemove(target).?;
                removed.value.deinit(self.allocator);
            }
        }
    }

    /// Check if a row has a conflicting lock for the given mode.
    /// Returns true if acquiring the lock would conflict.
    pub fn hasConflict(self: *LockManager, xid: u32, target: LockTarget, mode: LockMode) bool {
        const existing = self.row_locks.get(target) orelse return false;

        // If we already hold it, check for upgrade scenario
        if (existing.isHeldBy(xid)) {
            // Shared → exclusive upgrade with other holders is conflict
            if (existing.mode == .shared and mode == .exclusive and existing.holders.items.len > 1) {
                return true;
            }
            return false;
        }

        // Check mode conflict
        return existing.mode.conflictsWith(mode) or mode.conflictsWith(existing.mode);
    }

    /// Get the first lock holder for a row (for conflict reporting).
    /// Returns null if no lock exists.
    pub fn getRowLockHolder(self: *LockManager, target: LockTarget) ?u32 {
        const info = self.row_locks.get(target) orelse return null;
        if (info.holders.items.len == 0) return null;
        return info.holders.items[0];
    }

    // ── Table-level Locks ──────────────────────────────────────────

    /// Acquire a table-level lock.
    pub fn acquireTableLock(
        self: *LockManager,
        xid: u32,
        table_page_id: u32,
        mode: TableLockMode,
    ) LockError!void {
        const gop = try self.table_locks.getOrPut(self.allocator, table_page_id);
        if (!gop.found_existing) {
            gop.value_ptr.* = .{};
        }

        const list = gop.value_ptr;

        // Check if this XID already holds a lock on this table
        for (list.items) |entry| {
            if (entry.xid == xid) {
                // Already holds a lock — allow if same or stronger mode
                // For simplicity, we don't do automatic upgrade here
                return;
            }
        }

        // Check for conflicts with other transactions
        for (list.items) |entry| {
            if (entry.xid != xid) {
                if (entry.mode.conflictsWith(mode)) {
                    return LockError.LockConflict;
                }
            }
        }

        // Add the lock
        try list.append(self.allocator, .{ .xid = xid, .mode = mode });
    }

    /// Release a table-level lock.
    pub fn releaseTableLock(self: *LockManager, xid: u32, table_page_id: u32) void {
        if (self.table_locks.getPtr(table_page_id)) |list| {
            var idx: usize = 0;
            while (idx < list.items.len) {
                if (list.items[idx].xid == xid) {
                    _ = list.orderedRemove(idx);
                    // Don't break — a txn could theoretically hold multiple locks
                } else {
                    idx += 1;
                }
            }

            // Clean up empty lists
            if (list.items.len == 0) {
                list.deinit(self.allocator);
                _ = self.table_locks.remove(table_page_id);
            }
        }
    }

    // ── Batch Operations ───────────────────────────────────────────

    /// Release ALL locks held by a transaction (called on commit/rollback).
    pub fn releaseAllLocks(self: *LockManager, xid: u32) void {
        // Release all row locks
        var row_targets_to_remove = std.ArrayListUnmanaged(LockTarget){};
        defer row_targets_to_remove.deinit(self.allocator);

        var row_it = self.row_locks.iterator();
        while (row_it.next()) |entry| {
            const is_empty = entry.value_ptr.removeHolder(xid);
            if (is_empty) {
                row_targets_to_remove.append(self.allocator, entry.key_ptr.*) catch continue;
            }
        }

        for (row_targets_to_remove.items) |target| {
            var removed = self.row_locks.fetchRemove(target).?;
            removed.value.deinit(self.allocator);
        }

        // Release all table locks
        var table_ids_to_clean = std.ArrayListUnmanaged(u32){};
        defer table_ids_to_clean.deinit(self.allocator);

        var table_it = self.table_locks.iterator();
        while (table_it.next()) |entry| {
            const table_id = entry.key_ptr.*;
            const list = entry.value_ptr;

            var idx: usize = 0;
            while (idx < list.items.len) {
                if (list.items[idx].xid == xid) {
                    _ = list.orderedRemove(idx);
                } else {
                    idx += 1;
                }
            }

            if (list.items.len == 0) {
                table_ids_to_clean.append(self.allocator, table_id) catch continue;
            }
        }

        for (table_ids_to_clean.items) |table_id| {
            if (self.table_locks.fetchRemove(table_id)) |kv| {
                var list_copy = kv.value;
                list_copy.deinit(self.allocator);
            }
        }
    }

    // ── Monitoring ─────────────────────────────────────────────────

    /// Count total active row locks.
    pub fn activeRowLockCount(self: *LockManager) usize {
        return self.row_locks.count();
    }

    /// Count total active table locks.
    pub fn activeTableLockCount(self: *LockManager) usize {
        var count: usize = 0;
        var it = self.table_locks.iterator();
        while (it.next()) |entry| {
            count += entry.value_ptr.items.len;
        }
        return count;
    }
};

// ── Tests ──────────────────────────────────────────────────────────────

test "LockMode conflictsWith" {
    try std.testing.expect(LockMode.shared.conflictsWith(.exclusive));
    try std.testing.expect(!LockMode.shared.conflictsWith(.shared));
    try std.testing.expect(LockMode.exclusive.conflictsWith(.shared));
    try std.testing.expect(LockMode.exclusive.conflictsWith(.exclusive));
}

test "TableLockMode conflictsWith — ACCESS SHARE" {
    const as = TableLockMode.access_share;
    try std.testing.expect(!as.conflictsWith(.access_share));
    try std.testing.expect(!as.conflictsWith(.row_share));
    try std.testing.expect(!as.conflictsWith(.row_exclusive));
    try std.testing.expect(!as.conflictsWith(.share));
    try std.testing.expect(!as.conflictsWith(.share_row_exclusive));
    try std.testing.expect(!as.conflictsWith(.exclusive));
    try std.testing.expect(as.conflictsWith(.access_exclusive));
}

test "TableLockMode conflictsWith — ACCESS EXCLUSIVE" {
    const ae = TableLockMode.access_exclusive;
    try std.testing.expect(ae.conflictsWith(.access_share));
    try std.testing.expect(ae.conflictsWith(.row_share));
    try std.testing.expect(ae.conflictsWith(.row_exclusive));
    try std.testing.expect(ae.conflictsWith(.share));
    try std.testing.expect(ae.conflictsWith(.share_row_exclusive));
    try std.testing.expect(ae.conflictsWith(.exclusive));
    try std.testing.expect(ae.conflictsWith(.access_exclusive));
}

test "TableLockMode conflictsWith — ROW EXCLUSIVE" {
    const re = TableLockMode.row_exclusive;
    try std.testing.expect(!re.conflictsWith(.access_share));
    try std.testing.expect(!re.conflictsWith(.row_share));
    try std.testing.expect(!re.conflictsWith(.row_exclusive));
    try std.testing.expect(re.conflictsWith(.share));
    try std.testing.expect(re.conflictsWith(.share_row_exclusive));
    try std.testing.expect(re.conflictsWith(.exclusive));
    try std.testing.expect(re.conflictsWith(.access_exclusive));
}

test "LockTarget hash and equality" {
    const t1 = LockTarget{ .table_page_id = 5, .row_key = 100 };
    const t2 = LockTarget{ .table_page_id = 5, .row_key = 100 };
    const t3 = LockTarget{ .table_page_id = 5, .row_key = 101 };
    const t4 = LockTarget{ .table_page_id = 6, .row_key = 100 };

    try std.testing.expect(t1.eql(t2));
    try std.testing.expect(!t1.eql(t3));
    try std.testing.expect(!t1.eql(t4));

    // Hash equality for equal targets
    try std.testing.expectEqual(t1.hash(), t2.hash());
}

test "RowLockInfo — basic operations" {
    const allocator = std.testing.allocator;
    var info = RowLockInfo.init();
    defer info.deinit(allocator);

    try std.testing.expect(!info.isHeldBy(1));

    try info.addHolder(allocator, 1);
    try std.testing.expect(info.isHeldBy(1));
    try std.testing.expectEqual(@as(usize, 1), info.holders.items.len);

    try info.addHolder(allocator, 2);
    try std.testing.expect(info.isHeldBy(2));
    try std.testing.expectEqual(@as(usize, 2), info.holders.items.len);

    // Add same holder again — should be idempotent
    try info.addHolder(allocator, 1);
    try std.testing.expectEqual(@as(usize, 2), info.holders.items.len);

    const empty1 = info.removeHolder(1);
    try std.testing.expect(!empty1);
    try std.testing.expect(!info.isHeldBy(1));
    try std.testing.expectEqual(@as(usize, 1), info.holders.items.len);

    const empty2 = info.removeHolder(2);
    try std.testing.expect(empty2);
    try std.testing.expectEqual(@as(usize, 0), info.holders.items.len);
}

test "LockManager — init and deinit" {
    const allocator = std.testing.allocator;
    var lm = LockManager.init(allocator);
    defer lm.deinit();

    try std.testing.expectEqual(@as(usize, 0), lm.activeRowLockCount());
    try std.testing.expectEqual(@as(usize, 0), lm.activeTableLockCount());
}

test "LockManager — acquire and release row lock" {
    const allocator = std.testing.allocator;
    var lm = LockManager.init(allocator);
    defer lm.deinit();

    const target = LockTarget{ .table_page_id = 5, .row_key = 100 };

    try lm.acquireRowLock(1, target, .shared);
    try std.testing.expectEqual(@as(usize, 1), lm.activeRowLockCount());
    try std.testing.expect(!lm.hasConflict(1, target, .shared));

    lm.releaseRowLock(1, target);
    try std.testing.expectEqual(@as(usize, 0), lm.activeRowLockCount());
}

test "LockManager — shared row locks multiple holders" {
    const allocator = std.testing.allocator;
    var lm = LockManager.init(allocator);
    defer lm.deinit();

    const target = LockTarget{ .table_page_id = 5, .row_key = 100 };

    try lm.acquireRowLock(1, target, .shared);
    try lm.acquireRowLock(2, target, .shared);
    try lm.acquireRowLock(3, target, .shared);

    try std.testing.expectEqual(@as(usize, 1), lm.activeRowLockCount());

    const info = lm.row_locks.get(target).?;
    try std.testing.expectEqual(@as(usize, 3), info.holders.items.len);
    try std.testing.expect(info.isHeldBy(1));
    try std.testing.expect(info.isHeldBy(2));
    try std.testing.expect(info.isHeldBy(3));

    lm.releaseRowLock(2, target);
    try std.testing.expectEqual(@as(usize, 1), lm.activeRowLockCount());

    lm.releaseRowLock(1, target);
    lm.releaseRowLock(3, target);
    try std.testing.expectEqual(@as(usize, 0), lm.activeRowLockCount());
}

test "LockManager — exclusive row lock conflict" {
    const allocator = std.testing.allocator;
    var lm = LockManager.init(allocator);
    defer lm.deinit();

    const target = LockTarget{ .table_page_id = 5, .row_key = 100 };

    try lm.acquireRowLock(1, target, .exclusive);

    // Another txn tries exclusive — conflict
    try std.testing.expectError(error.LockConflict, lm.acquireRowLock(2, target, .exclusive));

    // Another txn tries shared — conflict
    try std.testing.expectError(error.LockConflict, lm.acquireRowLock(2, target, .shared));

    // Same txn re-acquires — no conflict
    try lm.acquireRowLock(1, target, .exclusive);

    lm.releaseRowLock(1, target);
}

test "LockManager — shared + exclusive conflict" {
    const allocator = std.testing.allocator;
    var lm = LockManager.init(allocator);
    defer lm.deinit();

    const target = LockTarget{ .table_page_id = 5, .row_key = 100 };

    try lm.acquireRowLock(1, target, .shared);
    try std.testing.expectError(error.LockConflict, lm.acquireRowLock(2, target, .exclusive));

    lm.releaseRowLock(1, target);

    // After release, exclusive should succeed
    try lm.acquireRowLock(2, target, .exclusive);
    lm.releaseRowLock(2, target);
}

test "LockManager — lock upgrade shared to exclusive" {
    const allocator = std.testing.allocator;
    var lm = LockManager.init(allocator);
    defer lm.deinit();

    const target = LockTarget{ .table_page_id = 5, .row_key = 100 };

    // Acquire shared
    try lm.acquireRowLock(1, target, .shared);

    // Upgrade to exclusive (sole holder)
    try lm.acquireRowLock(1, target, .exclusive);

    const info = lm.row_locks.get(target).?;
    try std.testing.expectEqual(LockMode.exclusive, info.mode);
    try std.testing.expectEqual(@as(usize, 1), info.holders.items.len);

    lm.releaseRowLock(1, target);
}

test "LockManager — lock upgrade fails with multiple holders" {
    const allocator = std.testing.allocator;
    var lm = LockManager.init(allocator);
    defer lm.deinit();

    const target = LockTarget{ .table_page_id = 5, .row_key = 100 };

    try lm.acquireRowLock(1, target, .shared);
    try lm.acquireRowLock(2, target, .shared);

    // Txn 1 tries to upgrade — should fail (other holder present)
    try std.testing.expectError(error.LockConflict, lm.acquireRowLock(1, target, .exclusive));

    lm.releaseRowLock(1, target);
    lm.releaseRowLock(2, target);
}

test "LockManager — release all locks for transaction" {
    const allocator = std.testing.allocator;
    var lm = LockManager.init(allocator);
    defer lm.deinit();

    const t1 = LockTarget{ .table_page_id = 5, .row_key = 100 };
    const t2 = LockTarget{ .table_page_id = 5, .row_key = 101 };
    const t3 = LockTarget{ .table_page_id = 6, .row_key = 200 };

    try lm.acquireRowLock(1, t1, .exclusive);
    try lm.acquireRowLock(1, t2, .shared);
    try lm.acquireRowLock(2, t2, .shared);
    try lm.acquireRowLock(2, t3, .exclusive);

    try std.testing.expectEqual(@as(usize, 3), lm.activeRowLockCount());

    lm.releaseAllLocks(1);

    // t1 should be gone, t2 should still exist (held by 2), t3 still exists
    try std.testing.expectEqual(@as(usize, 2), lm.activeRowLockCount());
    try std.testing.expect(lm.row_locks.get(t1) == null);
    try std.testing.expectEqual(@as(?u32, 2), lm.getRowLockHolder(t2));
    try std.testing.expectEqual(@as(?u32, 2), lm.getRowLockHolder(t3));

    lm.releaseAllLocks(2);
    try std.testing.expectEqual(@as(usize, 0), lm.activeRowLockCount());
}

test "LockManager — table lock acquire and release" {
    const allocator = std.testing.allocator;
    var lm = LockManager.init(allocator);
    defer lm.deinit();

    try lm.acquireTableLock(1, 5, .row_exclusive);
    try std.testing.expectEqual(@as(usize, 1), lm.activeTableLockCount());

    lm.releaseTableLock(1, 5);
    try std.testing.expectEqual(@as(usize, 0), lm.activeTableLockCount());
}

test "LockManager — table lock conflict" {
    const allocator = std.testing.allocator;
    var lm = LockManager.init(allocator);
    defer lm.deinit();

    try lm.acquireTableLock(1, 5, .row_exclusive);

    // Another txn tries SHARE — conflicts
    try std.testing.expectError(error.LockConflict, lm.acquireTableLock(2, 5, .share));

    // Same txn re-acquires — no conflict
    try lm.acquireTableLock(1, 5, .row_exclusive);

    lm.releaseTableLock(1, 5);

    // After release, SHARE should succeed
    try lm.acquireTableLock(2, 5, .share);
    lm.releaseTableLock(2, 5);
}

test "LockManager — table lock compatible modes" {
    const allocator = std.testing.allocator;
    var lm = LockManager.init(allocator);
    defer lm.deinit();

    try lm.acquireTableLock(1, 5, .access_share);
    try lm.acquireTableLock(2, 5, .access_share);
    try lm.acquireTableLock(3, 5, .row_share);

    // All compatible — should succeed
    try std.testing.expectEqual(@as(usize, 3), lm.activeTableLockCount());

    lm.releaseTableLock(1, 5);
    lm.releaseTableLock(2, 5);
    lm.releaseTableLock(3, 5);
    try std.testing.expectEqual(@as(usize, 0), lm.activeTableLockCount());
}

test "LockManager — hasConflict returns correct results" {
    const allocator = std.testing.allocator;
    var lm = LockManager.init(allocator);
    defer lm.deinit();

    const target = LockTarget{ .table_page_id = 5, .row_key = 100 };

    // No lock — no conflict
    try std.testing.expect(!lm.hasConflict(1, target, .shared));
    try std.testing.expect(!lm.hasConflict(1, target, .exclusive));

    try lm.acquireRowLock(1, target, .shared);

    // Same txn — no conflict
    try std.testing.expect(!lm.hasConflict(1, target, .shared));
    try std.testing.expect(!lm.hasConflict(1, target, .exclusive));

    // Different txn — shared compatible, exclusive conflicts
    try std.testing.expect(!lm.hasConflict(2, target, .shared));
    try std.testing.expect(lm.hasConflict(2, target, .exclusive));

    lm.releaseRowLock(1, target);

    try lm.acquireRowLock(1, target, .exclusive);

    // Different txn — both conflict with exclusive
    try std.testing.expect(lm.hasConflict(2, target, .shared));
    try std.testing.expect(lm.hasConflict(2, target, .exclusive));

    lm.releaseRowLock(1, target);
}

test "LockManager — getRowLockHolder" {
    const allocator = std.testing.allocator;
    var lm = LockManager.init(allocator);
    defer lm.deinit();

    const target = LockTarget{ .table_page_id = 5, .row_key = 100 };

    try std.testing.expectEqual(@as(?u32, null), lm.getRowLockHolder(target));

    try lm.acquireRowLock(10, target, .exclusive);
    try std.testing.expectEqual(@as(?u32, 10), lm.getRowLockHolder(target));

    lm.releaseRowLock(10, target);

    try lm.acquireRowLock(20, target, .shared);
    try lm.acquireRowLock(30, target, .shared);

    // Should return first holder
    const holder = lm.getRowLockHolder(target).?;
    try std.testing.expect(holder == 20 or holder == 30);

    lm.releaseAllLocks(20);
    lm.releaseAllLocks(30);
}

test "LockManager — acquire after release no stale locks" {
    const allocator = std.testing.allocator;
    var lm = LockManager.init(allocator);
    defer lm.deinit();

    const target = LockTarget{ .table_page_id = 5, .row_key = 100 };

    try lm.acquireRowLock(1, target, .exclusive);
    lm.releaseRowLock(1, target);

    // Should be able to acquire with different txn
    try lm.acquireRowLock(2, target, .exclusive);
    try std.testing.expectEqual(@as(?u32, 2), lm.getRowLockHolder(target));

    lm.releaseRowLock(2, target);
}

test "LockManager — multiple rows locked by same transaction" {
    const allocator = std.testing.allocator;
    var lm = LockManager.init(allocator);
    defer lm.deinit();

    const t1 = LockTarget{ .table_page_id = 5, .row_key = 100 };
    const t2 = LockTarget{ .table_page_id = 5, .row_key = 101 };
    const t3 = LockTarget{ .table_page_id = 5, .row_key = 102 };

    try lm.acquireRowLock(1, t1, .exclusive);
    try lm.acquireRowLock(1, t2, .shared);
    try lm.acquireRowLock(1, t3, .exclusive);

    try std.testing.expectEqual(@as(usize, 3), lm.activeRowLockCount());

    try std.testing.expectEqual(@as(?u32, 1), lm.getRowLockHolder(t1));
    try std.testing.expectEqual(@as(?u32, 1), lm.getRowLockHolder(t2));
    try std.testing.expectEqual(@as(?u32, 1), lm.getRowLockHolder(t3));

    lm.releaseAllLocks(1);
    try std.testing.expectEqual(@as(usize, 0), lm.activeRowLockCount());
}

test "LockManager — different tables same row key no conflict" {
    const allocator = std.testing.allocator;
    var lm = LockManager.init(allocator);
    defer lm.deinit();

    const t1 = LockTarget{ .table_page_id = 5, .row_key = 100 };
    const t2 = LockTarget{ .table_page_id = 6, .row_key = 100 };

    try lm.acquireRowLock(1, t1, .exclusive);
    try lm.acquireRowLock(2, t2, .exclusive);

    // Both should succeed — different tables
    try std.testing.expectEqual(@as(usize, 2), lm.activeRowLockCount());

    lm.releaseAllLocks(1);
    lm.releaseAllLocks(2);
}

test "LockManager — activeRowLockCount and activeTableLockCount" {
    const allocator = std.testing.allocator;
    var lm = LockManager.init(allocator);
    defer lm.deinit();

    const r1 = LockTarget{ .table_page_id = 5, .row_key = 100 };
    const r2 = LockTarget{ .table_page_id = 5, .row_key = 101 };

    try lm.acquireRowLock(1, r1, .exclusive);
    try lm.acquireRowLock(1, r2, .shared);
    try lm.acquireTableLock(1, 5, .row_exclusive);
    try lm.acquireTableLock(2, 6, .access_share);

    try std.testing.expectEqual(@as(usize, 2), lm.activeRowLockCount());
    try std.testing.expectEqual(@as(usize, 2), lm.activeTableLockCount());

    lm.releaseAllLocks(1);

    try std.testing.expectEqual(@as(usize, 0), lm.activeRowLockCount());
    try std.testing.expectEqual(@as(usize, 1), lm.activeTableLockCount());

    lm.releaseAllLocks(2);
    try std.testing.expectEqual(@as(usize, 0), lm.activeTableLockCount());
}

test "LockManager — hasConflict upgrade scenario with multiple holders" {
    const allocator = std.testing.allocator;
    var lm = LockManager.init(allocator);
    defer lm.deinit();

    const target = LockTarget{ .table_page_id = 5, .row_key = 100 };

    try lm.acquireRowLock(1, target, .shared);
    try lm.acquireRowLock(2, target, .shared);

    // Txn 1 tries to upgrade — should detect conflict
    try std.testing.expect(lm.hasConflict(1, target, .exclusive));

    lm.releaseAllLocks(1);
    lm.releaseAllLocks(2);
}
