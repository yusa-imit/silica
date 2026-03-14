const std = @import("std");
const Allocator = std.mem.Allocator;

/// Standby mode coordinator - manages read-only access on replicas
/// while WAL is being applied in the background
pub const StandbyCoordinator = struct {
    allocator: Allocator,
    mode: StandbyMode,
    wal_apply_in_progress: bool,
    read_only_txns: std.ArrayListUnmanaged(u64), // List of active read-only transaction IDs
    mutex: std.Thread.Mutex,

    pub const StandbyMode = enum {
        /// Not in standby mode (this is the primary)
        disabled,
        /// Hot standby: read-only queries allowed
        hot,
        /// Warm standby: no queries allowed (future feature)
        warm,
    };

    pub const Error = error{
        NotInStandbyMode,
        WriteOperationNotAllowed,
        ConflictWithRecovery,
        OutOfMemory,
    };

    /// Initialize standby coordinator
    pub fn init(allocator: Allocator, mode: StandbyMode) Error!StandbyCoordinator {
        return StandbyCoordinator{
            .allocator = allocator,
            .mode = mode,
            .wal_apply_in_progress = false,
            .read_only_txns = .{},
            .mutex = std.Thread.Mutex{},
        };
    }

    /// Clean up resources
    pub fn deinit(self: *StandbyCoordinator) void {
        self.read_only_txns.deinit(self.allocator);
    }

    /// Check if standby mode is enabled
    pub fn isStandbyMode(self: *const StandbyCoordinator) bool {
        return self.mode != .disabled;
    }

    /// Check if hot standby is enabled (read-only queries allowed)
    pub fn isHotStandby(self: *const StandbyCoordinator) bool {
        return self.mode == .hot;
    }

    /// Register a new read-only transaction
    pub fn registerReadOnlyTxn(self: *StandbyCoordinator, txn_id: u64) Error!void {
        if (!self.isHotStandby()) {
            return Error.NotInStandbyMode;
        }

        self.mutex.lock();
        defer self.mutex.unlock();

        try self.read_only_txns.append(self.allocator, txn_id);
    }

    /// Unregister a read-only transaction when it commits/aborts
    pub fn unregisterReadOnlyTxn(self: *StandbyCoordinator, txn_id: u64) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Find and remove the transaction
        for (self.read_only_txns.items, 0..) |id, i| {
            if (id == txn_id) {
                _ = self.read_only_txns.swapRemove(i);
                break;
            }
        }
    }

    /// Check if a write operation is allowed (only on primary)
    pub fn checkWriteAllowed(self: *const StandbyCoordinator) Error!void {
        if (self.isStandbyMode()) {
            return Error.WriteOperationNotAllowed;
        }
    }

    /// Notify that WAL apply is starting
    pub fn beginWalApply(self: *StandbyCoordinator) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.wal_apply_in_progress = true;
    }

    /// Notify that WAL apply has finished
    pub fn endWalApply(self: *StandbyCoordinator) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.wal_apply_in_progress = false;
    }

    /// Check if there are active read-only transactions
    /// Used by WAL receiver to determine if it needs to wait
    pub fn hasActiveReadOnlyTxns(self: *StandbyCoordinator) bool {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.read_only_txns.items.len > 0;
    }

    /// Get count of active read-only transactions
    pub fn getActiveReadOnlyTxnCount(self: *const StandbyCoordinator) usize {
        return self.read_only_txns.items.len;
    }

    /// Check if a query conflicts with WAL apply
    /// Returns true if the query should be canceled
    pub fn checkConflict(self: *StandbyCoordinator, _: u64) Error!bool {
        // Basic implementation: no conflicts yet
        // Future: check if WAL apply is touching data the query is reading
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.wal_apply_in_progress) {
            // Simplified conflict detection: if WAL is being applied,
            // there's a potential conflict
            // Production version would check actual data dependencies
            return true;
        }

        return false;
    }

    /// Cancel a conflicting query
    pub fn cancelConflictingQuery(self: *StandbyCoordinator, txn_id: u64) void {
        self.unregisterReadOnlyTxn(txn_id);
        // TODO: Signal the connection handler to abort this transaction
    }
};

// ============================================================================
// Tests
// ============================================================================

test "StandbyCoordinator init disabled mode" {
    var coord = try StandbyCoordinator.init(std.testing.allocator, .disabled);
    defer coord.deinit();

    try std.testing.expect(!coord.isStandbyMode());
    try std.testing.expect(!coord.isHotStandby());
}

test "StandbyCoordinator init hot standby mode" {
    var coord = try StandbyCoordinator.init(std.testing.allocator, .hot);
    defer coord.deinit();

    try std.testing.expect(coord.isStandbyMode());
    try std.testing.expect(coord.isHotStandby());
}

test "StandbyCoordinator register and unregister read-only txn" {
    var coord = try StandbyCoordinator.init(std.testing.allocator, .hot);
    defer coord.deinit();

    try coord.registerReadOnlyTxn(100);
    try std.testing.expectEqual(@as(usize, 1), coord.getActiveReadOnlyTxnCount());

    try coord.registerReadOnlyTxn(101);
    try std.testing.expectEqual(@as(usize, 2), coord.getActiveReadOnlyTxnCount());

    coord.unregisterReadOnlyTxn(100);
    try std.testing.expectEqual(@as(usize, 1), coord.getActiveReadOnlyTxnCount());

    coord.unregisterReadOnlyTxn(101);
    try std.testing.expectEqual(@as(usize, 0), coord.getActiveReadOnlyTxnCount());
}

test "StandbyCoordinator write operations not allowed in standby mode" {
    var coord = try StandbyCoordinator.init(std.testing.allocator, .hot);
    defer coord.deinit();

    const result = coord.checkWriteAllowed();
    try std.testing.expectError(StandbyCoordinator.Error.WriteOperationNotAllowed, result);
}

test "StandbyCoordinator write operations allowed on primary" {
    var coord = try StandbyCoordinator.init(std.testing.allocator, .disabled);
    defer coord.deinit();

    try coord.checkWriteAllowed(); // Should not error
}

test "StandbyCoordinator WAL apply tracking" {
    var coord = try StandbyCoordinator.init(std.testing.allocator, .hot);
    defer coord.deinit();

    try std.testing.expect(!coord.wal_apply_in_progress);

    coord.beginWalApply();
    try std.testing.expect(coord.wal_apply_in_progress);

    coord.endWalApply();
    try std.testing.expect(!coord.wal_apply_in_progress);
}

test "StandbyCoordinator conflict detection during WAL apply" {
    var coord = try StandbyCoordinator.init(std.testing.allocator, .hot);
    defer coord.deinit();

    // No conflict when WAL apply is not in progress
    const conflict1 = try coord.checkConflict(100);
    try std.testing.expect(!conflict1);

    // Conflict detected during WAL apply
    coord.beginWalApply();
    const conflict2 = try coord.checkConflict(100);
    try std.testing.expect(conflict2);

    coord.endWalApply();
    const conflict3 = try coord.checkConflict(100);
    try std.testing.expect(!conflict3);
}

test "StandbyCoordinator register txn fails in disabled mode" {
    var coord = try StandbyCoordinator.init(std.testing.allocator, .disabled);
    defer coord.deinit();

    const result = coord.registerReadOnlyTxn(100);
    try std.testing.expectError(StandbyCoordinator.Error.NotInStandbyMode, result);
}

test "StandbyCoordinator hasActiveReadOnlyTxns" {
    var coord = try StandbyCoordinator.init(std.testing.allocator, .hot);
    defer coord.deinit();

    try std.testing.expect(!coord.hasActiveReadOnlyTxns());

    try coord.registerReadOnlyTxn(100);
    try std.testing.expect(coord.hasActiveReadOnlyTxns());

    coord.unregisterReadOnlyTxn(100);
    try std.testing.expect(!coord.hasActiveReadOnlyTxns());
}

test "StandbyCoordinator unregister non-existent txn (no-op)" {
    var coord = try StandbyCoordinator.init(std.testing.allocator, .hot);
    defer coord.deinit();

    try coord.registerReadOnlyTxn(100);
    coord.unregisterReadOnlyTxn(999); // Non-existent, should be no-op
    try std.testing.expectEqual(@as(usize, 1), coord.getActiveReadOnlyTxnCount());
}

test "StandbyCoordinator multiple unregister same txn" {
    var coord = try StandbyCoordinator.init(std.testing.allocator, .hot);
    defer coord.deinit();

    try coord.registerReadOnlyTxn(100);
    coord.unregisterReadOnlyTxn(100);
    coord.unregisterReadOnlyTxn(100); // Second unregister is no-op
    try std.testing.expectEqual(@as(usize, 0), coord.getActiveReadOnlyTxnCount());
}

test "StandbyCoordinator warm standby mode (future feature)" {
    var coord = try StandbyCoordinator.init(std.testing.allocator, .warm);
    defer coord.deinit();

    try std.testing.expect(coord.isStandbyMode());
    try std.testing.expect(!coord.isHotStandby());

    // Should not allow read-only transactions in warm standby
    const result = coord.registerReadOnlyTxn(100);
    try std.testing.expectError(StandbyCoordinator.Error.NotInStandbyMode, result);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

test "StandbyCoordinator concurrent txn registration" {
    var coord = try StandbyCoordinator.init(std.testing.allocator, .hot);
    defer coord.deinit();

    // Register many transactions concurrently (simulated)
    try coord.registerReadOnlyTxn(1);
    try coord.registerReadOnlyTxn(2);
    try coord.registerReadOnlyTxn(3);
    try coord.registerReadOnlyTxn(4);
    try coord.registerReadOnlyTxn(5);

    try std.testing.expectEqual(@as(usize, 5), coord.getActiveReadOnlyTxnCount());

    // Unregister in different order
    coord.unregisterReadOnlyTxn(3);
    coord.unregisterReadOnlyTxn(1);
    coord.unregisterReadOnlyTxn(5);

    try std.testing.expectEqual(@as(usize, 2), coord.getActiveReadOnlyTxnCount());
}

test "StandbyCoordinator large transaction ID" {
    var coord = try StandbyCoordinator.init(std.testing.allocator, .hot);
    defer coord.deinit();

    const large_txn_id: u64 = 0xFFFFFFFFFFFFFFFF;
    try coord.registerReadOnlyTxn(large_txn_id);
    try std.testing.expectEqual(@as(usize, 1), coord.getActiveReadOnlyTxnCount());

    coord.unregisterReadOnlyTxn(large_txn_id);
    try std.testing.expectEqual(@as(usize, 0), coord.getActiveReadOnlyTxnCount());
}

test "StandbyCoordinator multiple WAL apply cycles" {
    var coord = try StandbyCoordinator.init(std.testing.allocator, .hot);
    defer coord.deinit();

    // Cycle 1
    coord.beginWalApply();
    try std.testing.expect(coord.wal_apply_in_progress);
    coord.endWalApply();
    try std.testing.expect(!coord.wal_apply_in_progress);

    // Cycle 2
    coord.beginWalApply();
    try std.testing.expect(coord.wal_apply_in_progress);
    coord.endWalApply();
    try std.testing.expect(!coord.wal_apply_in_progress);

    // Cycle 3
    coord.beginWalApply();
    try std.testing.expect(coord.wal_apply_in_progress);
    coord.endWalApply();
    try std.testing.expect(!coord.wal_apply_in_progress);
}

test "StandbyCoordinator register duplicate transaction ID" {
    var coord = try StandbyCoordinator.init(std.testing.allocator, .hot);
    defer coord.deinit();

    try coord.registerReadOnlyTxn(100);
    try coord.registerReadOnlyTxn(100); // Duplicate allowed (same txn could have multiple snapshots)
    try std.testing.expectEqual(@as(usize, 2), coord.getActiveReadOnlyTxnCount());

    coord.unregisterReadOnlyTxn(100); // Removes first instance
    try std.testing.expectEqual(@as(usize, 1), coord.getActiveReadOnlyTxnCount());
}

test "StandbyCoordinator cancelConflictingQuery" {
    var coord = try StandbyCoordinator.init(std.testing.allocator, .hot);
    defer coord.deinit();

    try coord.registerReadOnlyTxn(100);
    try std.testing.expectEqual(@as(usize, 1), coord.getActiveReadOnlyTxnCount());

    coord.cancelConflictingQuery(100);
    try std.testing.expectEqual(@as(usize, 0), coord.getActiveReadOnlyTxnCount());
}

test "StandbyCoordinator empty state operations" {
    var coord = try StandbyCoordinator.init(std.testing.allocator, .hot);
    defer coord.deinit();

    // Operations on empty state should be no-op or return expected values
    try std.testing.expect(!coord.hasActiveReadOnlyTxns());
    try std.testing.expectEqual(@as(usize, 0), coord.getActiveReadOnlyTxnCount());
    coord.unregisterReadOnlyTxn(999); // No-op
    coord.cancelConflictingQuery(999); // No-op
}

test "StandbyCoordinator conflict check with zero txn ID" {
    var coord = try StandbyCoordinator.init(std.testing.allocator, .hot);
    defer coord.deinit();

    const conflict = try coord.checkConflict(0);
    try std.testing.expect(!conflict);
}

test "StandbyCoordinator register after unregister same txn" {
    var coord = try StandbyCoordinator.init(std.testing.allocator, .hot);
    defer coord.deinit();

    try coord.registerReadOnlyTxn(100);
    coord.unregisterReadOnlyTxn(100);
    try coord.registerReadOnlyTxn(100); // Re-register same ID
    try std.testing.expectEqual(@as(usize, 1), coord.getActiveReadOnlyTxnCount());
}
