// Synchronous Replication Coordinator for Silica
//
// Manages synchronous commit semantics, tracking which standbys are synchronous
// and coordinating commit acknowledgments from replicas.

const std = @import("std");
const Allocator = std.mem.Allocator;
const protocol = @import("protocol.zig");
const LSN = protocol.LSN;

/// Synchronous replication mode
pub const SyncMode = enum(u8) {
    /// No synchronous replication (async only)
    off = 0,
    /// Wait for at least one synchronous standby
    on = 1,
    /// Wait for all synchronous standbys (quorum)
    quorum = 2,
};

/// Standby synchronization state
pub const StandbyState = struct {
    /// Standby name
    name: []const u8,
    /// Is this standby synchronous?
    is_sync: bool,
    /// Last confirmed flush LSN
    flush_lsn: LSN,
    /// Last update timestamp (microseconds)
    last_update: i64,
};

/// Synchronous replication coordinator
pub const SyncCoordinator = struct {
    /// Memory allocator
    allocator: Allocator,
    /// Synchronous replication mode
    mode: SyncMode,
    /// Synchronous standby names (comma-separated)
    sync_standby_names: []const u8,
    /// Connected standbys (name -> StandbyState)
    standbys: std.StringHashMap(StandbyState),
    /// Mutex for thread-safe access
    mutex: std.Thread.Mutex,
    /// Condition variable for commit waiters
    commit_cond: std.Thread.Condition,

    /// Initialize synchronous replication coordinator
    pub fn init(allocator: Allocator, mode: SyncMode, sync_standby_names: []const u8) !SyncCoordinator {
        const names_copy = try allocator.dupe(u8, sync_standby_names);
        return .{
            .allocator = allocator,
            .mode = mode,
            .sync_standby_names = names_copy,
            .standbys = std.StringHashMap(StandbyState).init(allocator),
            .mutex = .{},
            .commit_cond = .{},
        };
    }

    /// Deinitialize coordinator
    pub fn deinit(self: *SyncCoordinator) void {
        var it = self.standbys.iterator();
        while (it.next()) |entry| {
            // entry.key_ptr.* and entry.value_ptr.name are the same pointer, free only once
            self.allocator.free(entry.key_ptr.*);
        }
        self.standbys.deinit();
        self.allocator.free(self.sync_standby_names);
    }

    /// Register a standby connection
    pub fn registerStandby(self: *SyncCoordinator, name: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        // If already registered, remove old entry to avoid leak
        if (self.standbys.fetchRemove(name)) |entry| {
            self.allocator.free(entry.key);
        }

        const name_copy = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(name_copy);

        const is_sync = self.isSyncStandby(name);
        const state = StandbyState{
            .name = name_copy, // Share the same allocation for key and state.name
            .is_sync = is_sync,
            .flush_lsn = 0,
            .last_update = std.time.microTimestamp(),
        };

        try self.standbys.put(name_copy, state);
    }

    /// Unregister a standby connection
    pub fn unregisterStandby(self: *SyncCoordinator, name: []const u8) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.standbys.fetchRemove(name)) |entry| {
            // entry.key and entry.value.name are the same pointer, free only once
            self.allocator.free(entry.key);
        }
    }

    /// Update standby flush LSN
    pub fn updateFlushLSN(self: *SyncCoordinator, name: []const u8, flush_lsn: LSN) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const state = self.standbys.getPtr(name) orelse return error.StandbyNotFound;
        state.flush_lsn = flush_lsn;
        state.last_update = std.time.microTimestamp();

        // Wake up waiting commits
        self.commit_cond.broadcast();
    }

    /// Wait for synchronous standbys to acknowledge WAL flush up to target_lsn
    pub fn waitForSync(self: *SyncCoordinator, target_lsn: LSN, timeout_ms: u64) !void {
        if (self.mode == .off) return; // No synchronous replication

        self.mutex.lock();
        defer self.mutex.unlock();

        const deadline = std.time.microTimestamp() + @as(i64, @intCast(timeout_ms * 1000));

        while (true) {
            // Check if synchronous condition is met
            if (try self.isSyncConditionMetLocked(target_lsn)) {
                return;
            }

            // Check timeout
            const now = std.time.microTimestamp();
            if (now >= deadline) {
                return error.SyncTimeout;
            }

            // Wait for updates (with timeout)
            const wait_us = @as(u64, @intCast(deadline - now));
            _ = self.commit_cond.timedWait(&self.mutex, wait_us) catch {};
        }
    }

    /// Check if synchronous condition is met (caller must hold lock)
    fn isSyncConditionMetLocked(self: *SyncCoordinator, target_lsn: LSN) !bool {
        var sync_count: u32 = 0;
        var confirmed_count: u32 = 0;

        var it = self.standbys.iterator();
        while (it.next()) |entry| {
            const state = entry.value_ptr;
            if (state.is_sync) {
                sync_count += 1;
                if (state.flush_lsn >= target_lsn) {
                    confirmed_count += 1;
                }
            }
        }

        return switch (self.mode) {
            .off => true,
            .on => confirmed_count > 0, // At least one sync standby confirmed
            .quorum => confirmed_count == sync_count, // All sync standbys confirmed
        };
    }

    /// Check if a standby name is in the synchronous list
    fn isSyncStandby(self: *SyncCoordinator, name: []const u8) bool {
        // Simple comma-separated matching (production: support FIRST N, ANY N, etc.)
        var it = std.mem.splitScalar(u8, self.sync_standby_names, ',');
        while (it.next()) |sync_name| {
            const trimmed = std.mem.trim(u8, sync_name, " \t");
            if (std.mem.eql(u8, trimmed, name)) {
                return true;
            }
        }
        return false;
    }

    /// Get current synchronous standby count
    pub fn getSyncStandbyCount(self: *SyncCoordinator) u32 {
        self.mutex.lock();
        defer self.mutex.unlock();

        var count: u32 = 0;
        var it = self.standbys.iterator();
        while (it.next()) |entry| {
            if (entry.value_ptr.is_sync) {
                count += 1;
            }
        }
        return count;
    }

    /// Get minimum flush LSN among synchronous standbys
    pub fn getMinSyncFlushLSN(self: *SyncCoordinator) LSN {
        self.mutex.lock();
        defer self.mutex.unlock();

        var min_lsn: LSN = std.math.maxInt(LSN);
        var found = false;

        var it = self.standbys.iterator();
        while (it.next()) |entry| {
            const state = entry.value_ptr;
            if (state.is_sync) {
                found = true;
                if (state.flush_lsn < min_lsn) {
                    min_lsn = state.flush_lsn;
                }
            }
        }

        return if (found) min_lsn else 0;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "SyncCoordinator: init and deinit" {
    const allocator = std.testing.allocator;
    var coord = try SyncCoordinator.init(allocator, .on, "replica1,replica2");
    defer coord.deinit();

    try std.testing.expectEqual(SyncMode.on, coord.mode);
    try std.testing.expectEqualStrings("replica1,replica2", coord.sync_standby_names);
}

test "SyncCoordinator: register and unregister standby" {
    const allocator = std.testing.allocator;
    var coord = try SyncCoordinator.init(allocator, .on, "replica1");
    defer coord.deinit();

    try coord.registerStandby("replica1");
    try std.testing.expectEqual(@as(u32, 1), coord.getSyncStandbyCount());

    coord.unregisterStandby("replica1");
    try std.testing.expectEqual(@as(u32, 0), coord.getSyncStandbyCount());
}

test "SyncCoordinator: isSyncStandby matching" {
    const allocator = std.testing.allocator;
    var coord = try SyncCoordinator.init(allocator, .on, "replica1, replica2");
    defer coord.deinit();

    try std.testing.expect(coord.isSyncStandby("replica1"));
    try std.testing.expect(coord.isSyncStandby("replica2"));
    try std.testing.expect(!coord.isSyncStandby("replica3"));
}

test "SyncCoordinator: updateFlushLSN and getMinSyncFlushLSN" {
    const allocator = std.testing.allocator;
    var coord = try SyncCoordinator.init(allocator, .on, "replica1,replica2");
    defer coord.deinit();

    try coord.registerStandby("replica1");
    try coord.registerStandby("replica2");

    try coord.updateFlushLSN("replica1", 1000);
    try coord.updateFlushLSN("replica2", 2000);

    const min_lsn = coord.getMinSyncFlushLSN();
    try std.testing.expectEqual(@as(LSN, 1000), min_lsn);
}

test "SyncCoordinator: waitForSync with mode off" {
    const allocator = std.testing.allocator;
    var coord = try SyncCoordinator.init(allocator, .off, "");
    defer coord.deinit();

    // Should return immediately when mode is off
    try coord.waitForSync(1000, 100);
}

test "SyncCoordinator: waitForSync mode on (at least one)" {
    const allocator = std.testing.allocator;
    var coord = try SyncCoordinator.init(allocator, .on, "replica1,replica2");
    defer coord.deinit();

    try coord.registerStandby("replica1");
    try coord.registerStandby("replica2");

    // Update one standby to confirm
    try coord.updateFlushLSN("replica1", 1000);

    // Should succeed immediately (at least one confirmed)
    try coord.waitForSync(1000, 100);
}

test "SyncCoordinator: waitForSync mode quorum (all)" {
    const allocator = std.testing.allocator;
    var coord = try SyncCoordinator.init(allocator, .quorum, "replica1,replica2");
    defer coord.deinit();

    try coord.registerStandby("replica1");
    try coord.registerStandby("replica2");

    // Update both standbys
    try coord.updateFlushLSN("replica1", 1000);
    try coord.updateFlushLSN("replica2", 1000);

    // Should succeed (all confirmed)
    try coord.waitForSync(1000, 100);
}

test "SyncCoordinator: waitForSync timeout" {
    const allocator = std.testing.allocator;
    var coord = try SyncCoordinator.init(allocator, .on, "replica1");
    defer coord.deinit();

    try coord.registerStandby("replica1");

    // No flush LSN update → should timeout
    const result = coord.waitForSync(1000, 50);
    try std.testing.expectError(error.SyncTimeout, result);
}

test "SyncCoordinator: async standby not counted for sync" {
    const allocator = std.testing.allocator;
    var coord = try SyncCoordinator.init(allocator, .on, "replica1");
    defer coord.deinit();

    try coord.registerStandby("replica1");
    try coord.registerStandby("replica2"); // Not in sync list

    try std.testing.expectEqual(@as(u32, 1), coord.getSyncStandbyCount());

    // Update async standby
    try coord.updateFlushLSN("replica2", 1000);

    // Sync condition not met (only async standby confirmed)
    const result = coord.waitForSync(1000, 50);
    try std.testing.expectError(error.SyncTimeout, result);
}

test "SyncCoordinator: unregister standby updates count" {
    const allocator = std.testing.allocator;
    var coord = try SyncCoordinator.init(allocator, .on, "replica1,replica2");
    defer coord.deinit();

    try coord.registerStandby("replica1");
    try coord.registerStandby("replica2");
    try std.testing.expectEqual(@as(u32, 2), coord.getSyncStandbyCount());

    coord.unregisterStandby("replica1");
    try std.testing.expectEqual(@as(u32, 1), coord.getSyncStandbyCount());
}

test "SyncCoordinator: getMinSyncFlushLSN with no sync standbys" {
    const allocator = std.testing.allocator;
    var coord = try SyncCoordinator.init(allocator, .on, "replica1");
    defer coord.deinit();

    // No standbys registered
    const min_lsn = coord.getMinSyncFlushLSN();
    try std.testing.expectEqual(@as(LSN, 0), min_lsn);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

test "SyncCoordinator: empty sync_standby_names" {
    const allocator = std.testing.allocator;
    var coord = try SyncCoordinator.init(allocator, .on, "");
    defer coord.deinit();

    try coord.registerStandby("replica1");
    try std.testing.expectEqual(@as(u32, 0), coord.getSyncStandbyCount());

    // No sync standbys, so waitForSync should timeout
    const result = coord.waitForSync(1000, 50);
    try std.testing.expectError(error.SyncTimeout, result);
}

test "SyncCoordinator: updateFlushLSN for non-existent standby" {
    const allocator = std.testing.allocator;
    var coord = try SyncCoordinator.init(allocator, .on, "replica1");
    defer coord.deinit();

    try coord.registerStandby("replica1");

    // Try to update non-existent standby
    const result = coord.updateFlushLSN("replica2", 1000);
    try std.testing.expectError(error.StandbyNotFound, result);
}

test "SyncCoordinator: register same standby twice" {
    const allocator = std.testing.allocator;
    var coord = try SyncCoordinator.init(allocator, .on, "replica1");
    defer coord.deinit();

    try coord.registerStandby("replica1");
    try std.testing.expectEqual(@as(u32, 1), coord.getSyncStandbyCount());

    // Register again - should replace existing
    try coord.registerStandby("replica1");
    try std.testing.expectEqual(@as(u32, 1), coord.getSyncStandbyCount());
}

test "SyncCoordinator: unregister non-existent standby" {
    const allocator = std.testing.allocator;
    var coord = try SyncCoordinator.init(allocator, .on, "replica1");
    defer coord.deinit();

    // Unregister standby that was never registered (should be no-op)
    coord.unregisterStandby("replica1");
    try std.testing.expectEqual(@as(u32, 0), coord.getSyncStandbyCount());
}

test "SyncCoordinator: getMinSyncFlushLSN with mixed sync and async standbys" {
    const allocator = std.testing.allocator;
    var coord = try SyncCoordinator.init(allocator, .on, "replica1,replica2");
    defer coord.deinit();

    try coord.registerStandby("replica1"); // sync
    try coord.registerStandby("replica2"); // sync
    try coord.registerStandby("replica3"); // async (not in sync list)

    try coord.updateFlushLSN("replica1", 1000);
    try coord.updateFlushLSN("replica2", 2000);
    try coord.updateFlushLSN("replica3", 500); // Lower LSN but async

    // Should return min of sync standbys only (1000), not async (500)
    const min_lsn = coord.getMinSyncFlushLSN();
    try std.testing.expectEqual(@as(LSN, 1000), min_lsn);
}

test "SyncCoordinator: standby names with whitespace" {
    const allocator = std.testing.allocator;
    var coord = try SyncCoordinator.init(allocator, .on, "  replica1  ,  replica2  ");
    defer coord.deinit();

    // Should match after trimming whitespace
    try std.testing.expect(coord.isSyncStandby("replica1"));
    try std.testing.expect(coord.isSyncStandby("replica2"));
}

test "SyncCoordinator: very large LSN values" {
    const allocator = std.testing.allocator;
    var coord = try SyncCoordinator.init(allocator, .on, "replica1");
    defer coord.deinit();

    try coord.registerStandby("replica1");

    const max_lsn = std.math.maxInt(LSN);
    try coord.updateFlushLSN("replica1", max_lsn);

    const min_lsn = coord.getMinSyncFlushLSN();
    try std.testing.expectEqual(max_lsn, min_lsn);

    // Should succeed with max LSN
    try coord.waitForSync(max_lsn, 100);
}

test "SyncCoordinator: waitForSync with partial quorum" {
    const allocator = std.testing.allocator;
    var coord = try SyncCoordinator.init(allocator, .quorum, "replica1,replica2");
    defer coord.deinit();

    try coord.registerStandby("replica1");
    try coord.registerStandby("replica2");

    // Only one standby confirms (not enough for quorum)
    try coord.updateFlushLSN("replica1", 1000);

    const result = coord.waitForSync(1000, 50);
    try std.testing.expectError(error.SyncTimeout, result);
}

test "SyncCoordinator: mode quorum with no sync standbys" {
    const allocator = std.testing.allocator;
    var coord = try SyncCoordinator.init(allocator, .quorum, "");
    defer coord.deinit();

    try coord.registerStandby("replica1"); // async

    try coord.updateFlushLSN("replica1", 1000);

    // No sync standbys, quorum should be met (0 == 0)
    try coord.waitForSync(1000, 100);
}
