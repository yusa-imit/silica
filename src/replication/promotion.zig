// Replica Promotion for Silica
//
// Handles promotion of a standby replica to primary.
// Used for failover and switchover scenarios.

const std = @import("std");
const Allocator = std.mem.Allocator;
const receiver = @import("receiver.zig");
const standby = @import("standby.zig");
const protocol = @import("protocol.zig");
const LSN = protocol.LSN;

/// Promotion errors
pub const Error = error{
    /// Not a standby (already primary)
    NotStandby,
    /// Promotion already in progress
    PromotionInProgress,
    /// Failed to stop WAL receiver
    ReceiverStopFailed,
    /// Failed to transition standby coordinator
    StandbyTransitionFailed,
} || Allocator.Error;

/// Promotion state
pub const PromotionState = enum(u8) {
    /// No promotion in progress
    idle = 0,
    /// Promotion started
    in_progress = 1,
    /// Promotion completed successfully
    completed = 2,
    /// Promotion failed
    failed = 3,
};

/// Replica promotion coordinator
pub const PromotionCoordinator = struct {
    /// Memory allocator
    allocator: Allocator,
    /// Current promotion state
    state: PromotionState,
    /// WAL receiver (to be stopped during promotion)
    wal_receiver: ?*receiver.WalReceiver,
    /// Standby coordinator (to be transitioned to read-write)
    standby_coordinator: ?*standby.StandbyCoordinator,
    /// New timeline ID after promotion
    new_timeline_id: u32,
    /// Promotion start timestamp
    start_time: i64,
    /// Promotion end timestamp
    end_time: i64,
    /// Last error message (if failed)
    error_message: ?[]const u8,

    /// Initialize promotion coordinator
    pub fn init(allocator: Allocator) PromotionCoordinator {
        return .{
            .allocator = allocator,
            .state = .idle,
            .wal_receiver = null,
            .standby_coordinator = null,
            .new_timeline_id = 0,
            .start_time = 0,
            .end_time = 0,
            .error_message = null,
        };
    }

    /// Deinitialize coordinator
    pub fn deinit(self: *PromotionCoordinator) void {
        if (self.error_message) |msg| {
            self.allocator.free(msg);
        }
    }

    /// Set WAL receiver reference
    pub fn setWalReceiver(self: *PromotionCoordinator, recv: *receiver.WalReceiver) void {
        self.wal_receiver = recv;
    }

    /// Set standby coordinator reference
    pub fn setStandbyCoordinator(self: *PromotionCoordinator, coord: *standby.StandbyCoordinator) void {
        self.standby_coordinator = coord;
    }

    /// Promote standby to primary
    pub fn promote(self: *PromotionCoordinator, new_timeline_id: u32) !void {
        if (self.state == .in_progress) {
            return Error.PromotionInProgress;
        }

        // Check if we're actually a standby
        if (self.standby_coordinator == null or self.wal_receiver == null) {
            return Error.NotStandby;
        }

        self.state = .in_progress;
        self.start_time = std.time.microTimestamp();
        self.new_timeline_id = new_timeline_id;

        errdefer {
            self.state = .failed;
            self.end_time = std.time.microTimestamp();
        }

        // Step 1: Stop WAL receiver
        if (self.wal_receiver) |recv| {
            recv.disconnect();
        }

        // Step 2: Transition standby coordinator to read-write (disabled mode)
        if (self.standby_coordinator) |coord| {
            try coord.transitionToReadWrite();
        }

        // Step 3: Mark promotion as completed
        self.state = .completed;
        self.end_time = std.time.microTimestamp();
    }

    /// Get current promotion state
    pub fn getState(self: *PromotionCoordinator) PromotionState {
        return self.state;
    }

    /// Get promotion duration in microseconds (if completed or failed)
    pub fn getDuration(self: *PromotionCoordinator) ?i64 {
        if (self.start_time == 0) return null;
        const end = if (self.end_time > 0) self.end_time else std.time.microTimestamp();
        return end - self.start_time;
    }

    /// Check if promotion is complete
    pub fn isComplete(self: *PromotionCoordinator) bool {
        return self.state == .completed;
    }

    /// Reset promotion state (for testing)
    pub fn reset(self: *PromotionCoordinator) void {
        self.state = .idle;
        self.start_time = 0;
        self.end_time = 0;
        self.new_timeline_id = 0;
        if (self.error_message) |msg| {
            self.allocator.free(msg);
            self.error_message = null;
        }
    }
};

// ============================================================================
// Tests
// ============================================================================

test "PromotionCoordinator: init and deinit" {
    const allocator = std.testing.allocator;
    var coord = PromotionCoordinator.init(allocator);
    defer coord.deinit();

    try std.testing.expectEqual(PromotionState.idle, coord.state);
    try std.testing.expect(coord.wal_receiver == null);
    try std.testing.expect(coord.standby_coordinator == null);
}

test "PromotionCoordinator: setWalReceiver and setStandbyCoordinator" {
    const allocator = std.testing.allocator;
    var coord = PromotionCoordinator.init(allocator);
    defer coord.deinit();

    const recv_config = receiver.Config{
        .primary_conninfo = "conninfo",
        .slot_name = "slot1",
    };
    var recv = try receiver.WalReceiver.init(allocator, recv_config);
    defer recv.deinit();

    var standby_coord = try standby.StandbyCoordinator.init(allocator, .hot);
    defer standby_coord.deinit();

    coord.setWalReceiver(&recv);
    coord.setStandbyCoordinator(&standby_coord);

    try std.testing.expect(coord.wal_receiver != null);
    try std.testing.expect(coord.standby_coordinator != null);
}

test "PromotionCoordinator: promote from standby" {
    const allocator = std.testing.allocator;
    var coord = PromotionCoordinator.init(allocator);
    defer coord.deinit();

    const recv_config = receiver.Config{
        .primary_conninfo = "conninfo",
        .slot_name = "slot1",
    };
    var recv = try receiver.WalReceiver.init(allocator, recv_config);
    defer recv.deinit();

    var standby_coord = try standby.StandbyCoordinator.init(allocator, .hot);
    defer standby_coord.deinit();

    coord.setWalReceiver(&recv);
    coord.setStandbyCoordinator(&standby_coord);

    // Promote to timeline 2
    try coord.promote(2);

    try std.testing.expect(coord.isComplete());
    try std.testing.expectEqual(PromotionState.completed, coord.state);
    try std.testing.expectEqual(@as(u32, 2), coord.new_timeline_id);
    try std.testing.expect(coord.getDuration() != null);
}

test "PromotionCoordinator: promote when not standby" {
    const allocator = std.testing.allocator;
    var coord = PromotionCoordinator.init(allocator);
    defer coord.deinit();

    // No WAL receiver or standby coordinator set
    const result = coord.promote(2);
    try std.testing.expectError(Error.NotStandby, result);
}

test "PromotionCoordinator: promotion already in progress" {
    const allocator = std.testing.allocator;
    var coord = PromotionCoordinator.init(allocator);
    defer coord.deinit();

    const recv_config = receiver.Config{
        .primary_conninfo = "conninfo",
        .slot_name = "slot1",
    };
    var recv = try receiver.WalReceiver.init(allocator, recv_config);
    defer recv.deinit();

    var standby_coord = try standby.StandbyCoordinator.init(allocator, .hot);
    defer standby_coord.deinit();

    coord.setWalReceiver(&recv);
    coord.setStandbyCoordinator(&standby_coord);

    // Set state to in_progress manually
    coord.state = .in_progress;

    const result = coord.promote(2);
    try std.testing.expectError(Error.PromotionInProgress, result);
}

test "PromotionCoordinator: getDuration returns null before promotion" {
    const allocator = std.testing.allocator;
    var coord = PromotionCoordinator.init(allocator);
    defer coord.deinit();

    try std.testing.expect(coord.getDuration() == null);
}

test "PromotionCoordinator: reset clears state" {
    const allocator = std.testing.allocator;
    var coord = PromotionCoordinator.init(allocator);
    defer coord.deinit();

    const recv_config = receiver.Config{
        .primary_conninfo = "conninfo",
        .slot_name = "slot1",
    };
    var recv = try receiver.WalReceiver.init(allocator, recv_config);
    defer recv.deinit();

    var standby_coord = try standby.StandbyCoordinator.init(allocator, .hot);
    defer standby_coord.deinit();

    coord.setWalReceiver(&recv);
    coord.setStandbyCoordinator(&standby_coord);

    try coord.promote(2);
    try std.testing.expect(coord.isComplete());

    coord.reset();
    try std.testing.expectEqual(PromotionState.idle, coord.state);
    try std.testing.expectEqual(@as(i64, 0), coord.start_time);
    try std.testing.expectEqual(@as(i64, 0), coord.end_time);
}

test "PromotionCoordinator: promotion stops WAL receiver" {
    const allocator = std.testing.allocator;
    var coord = PromotionCoordinator.init(allocator);
    defer coord.deinit();

    const recv_config = receiver.Config{
        .primary_conninfo = "conninfo",
        .slot_name = "slot1",
    };
    var recv = try receiver.WalReceiver.init(allocator, recv_config);
    defer recv.deinit();

    var standby_coord = try standby.StandbyCoordinator.init(allocator, .hot);
    defer standby_coord.deinit();

    coord.setWalReceiver(&recv);
    coord.setStandbyCoordinator(&standby_coord);

    // Simulate connection
    recv.connected = true;

    try coord.promote(2);

    // WAL receiver should be disconnected
    try std.testing.expect(!recv.connected);
}
