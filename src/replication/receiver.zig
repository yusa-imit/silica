// WAL Receiver Process for Silica
//
// Receives and applies WAL records from primary server.
// Runs on replica servers to maintain synchronized copy.

const std = @import("std");
const Allocator = std.mem.Allocator;
const protocol = @import("protocol.zig");
const LSN = protocol.LSN;
const BackendMessage = protocol.BackendMessage;
const FrontendMessage = protocol.FrontendMessage;

/// WAL Receiver errors
pub const Error = error{
    /// Connection lost
    ConnectionLost,
    /// Invalid WAL data
    InvalidWalData,
    /// LSN mismatch
    LsnMismatch,
    /// Protocol error
    ProtocolError,
    /// Apply failed
    ApplyFailed,
} || Allocator.Error || std.fs.File.WriteError || std.fs.File.ReadError;

/// WAL Receiver configuration
pub const Config = struct {
    /// Primary server connection string
    primary_conninfo: []const u8,
    /// Replication slot name on primary
    slot_name: []const u8,
    /// Status update interval in milliseconds
    status_interval_ms: u64 = 10_000,
    /// Maximum retry attempts for connection
    max_retries: u32 = 10,
    /// Retry delay in milliseconds
    retry_delay_ms: u64 = 1000,
};

/// WAL Receiver state
pub const WalReceiver = struct {
    /// Memory allocator
    allocator: Allocator,
    /// Configuration
    config: Config,
    /// Last received LSN (write_lsn)
    write_lsn: LSN,
    /// Last flushed LSN (flush_lsn)
    flush_lsn: LSN,
    /// Last applied LSN (apply_lsn)
    apply_lsn: LSN,
    /// Last status update timestamp
    last_status_update: i64,
    /// Connection established flag
    connected: bool,
    /// WAL file for writing received data
    wal_file: ?std.fs.File,
    /// Apply buffer
    apply_buffer: std.ArrayList(u8),

    pub fn init(allocator: Allocator, config: Config) !WalReceiver {
        return .{
            .allocator = allocator,
            .config = config,
            .write_lsn = 0,
            .flush_lsn = 0,
            .apply_lsn = 0,
            .last_status_update = std.time.microTimestamp(),
            .connected = false,
            .wal_file = null,
            .apply_buffer = std.ArrayList(u8).init(allocator),
        };
    }

    pub fn deinit(self: *WalReceiver) void {
        if (self.wal_file) |*file| {
            file.close();
        }
        self.apply_buffer.deinit();
    }

    /// Connect to primary and start replication
    pub fn connect(self: *WalReceiver, start_lsn: LSN) !void {
        // TODO: Actual TCP connection to primary
        // For now, just mark as connected
        self.connected = true;
        self.write_lsn = start_lsn;
        self.flush_lsn = start_lsn;
        self.apply_lsn = start_lsn;
    }

    /// Disconnect from primary
    pub fn disconnect(self: *WalReceiver) void {
        self.connected = false;
    }

    /// Process received WAL data message
    pub fn processWalData(
        self: *WalReceiver,
        wal_start: LSN,
        wal_end: LSN,
        data: []const u8,
    ) !void {
        // Verify LSN continuity
        if (wal_start != self.write_lsn) {
            return Error.LsnMismatch;
        }

        // Write to WAL buffer
        try self.apply_buffer.appendSlice(data);
        self.write_lsn = wal_end;

        // TODO: Actual WAL file writing and application
        // For now, just update positions
        try self.flushWal();
        try self.applyWal();
    }

    /// Process keepalive message
    pub fn processKeepalive(
        self: *WalReceiver,
        wal_end: LSN,
        reply_requested: bool,
    ) !bool {
        _ = wal_end;
        return reply_requested;
    }

    /// Flush WAL data to disk
    fn flushWal(self: *WalReceiver) !void {
        // TODO: Actual file flush
        self.flush_lsn = self.write_lsn;
    }

    /// Apply WAL data to database
    fn applyWal(self: *WalReceiver) !void {
        // TODO: Actual WAL application
        self.apply_lsn = self.flush_lsn;
        self.apply_buffer.clearRetainingCapacity();
    }

    /// Create standby status update message
    pub fn createStatusUpdate(self: *WalReceiver, reply_requested: bool) FrontendMessage {
        self.last_status_update = std.time.microTimestamp();
        return .{
            .standby_status = .{
                .write_lsn = self.write_lsn,
                .flush_lsn = self.flush_lsn,
                .apply_lsn = self.apply_lsn,
                .client_timestamp = self.last_status_update,
                .reply_requested = reply_requested,
            },
        };
    }

    /// Check if status update should be sent
    pub fn shouldSendStatus(self: *WalReceiver) bool {
        const now = std.time.microTimestamp();
        const elapsed_us = now - self.last_status_update;
        const threshold_us = @as(i64, @intCast(self.config.status_interval_ms)) * 1000;
        return elapsed_us >= threshold_us;
    }

    /// Create IDENTIFY_SYSTEM message
    pub fn createIdentifySystemMessage() FrontendMessage {
        return .{ .identify_system = {} };
    }

    /// Create START_REPLICATION message
    pub fn createStartReplicationMessage(
        allocator: Allocator,
        slot_name: []const u8,
        start_lsn: LSN,
    ) !FrontendMessage {
        const slot_name_copy = try allocator.dupe(u8, slot_name);
        return .{
            .start_replication = .{
                .slot_name = slot_name_copy,
                .start_lsn = start_lsn,
            },
        };
    }

    /// Create CREATE_REPLICATION_SLOT message
    pub fn createCreateSlotMessage(
        allocator: Allocator,
        slot_name: []const u8,
        temporary: bool,
    ) !FrontendMessage {
        const slot_name_copy = try allocator.dupe(u8, slot_name);
        return .{
            .create_slot = .{
                .slot_name = slot_name_copy,
                .temporary = temporary,
            },
        };
    }

    /// Create DROP_REPLICATION_SLOT message
    pub fn createDropSlotMessage(
        allocator: Allocator,
        slot_name: []const u8,
    ) !FrontendMessage {
        const slot_name_copy = try allocator.dupe(u8, slot_name);
        return .{
            .drop_slot = .{
                .slot_name = slot_name_copy,
            },
        };
    }

    /// Get current replication lag in bytes
    pub fn getReplicationLag(self: *WalReceiver, primary_wal_end: LSN) i64 {
        if (primary_wal_end < self.apply_lsn) {
            return 0; // Replica ahead (shouldn't happen)
        }
        return @as(i64, @intCast(primary_wal_end - self.apply_lsn));
    }
};

// ============================================================================
// Tests
// ============================================================================

test "WalReceiver init and deinit" {
    const allocator = std.testing.allocator;

    const config = Config{
        .primary_conninfo = "host=primary port=5432",
        .slot_name = "test-slot",
    };

    var receiver = try WalReceiver.init(allocator, config);
    defer receiver.deinit();

    try std.testing.expectEqual(@as(LSN, 0), receiver.write_lsn);
    try std.testing.expectEqual(@as(LSN, 0), receiver.flush_lsn);
    try std.testing.expectEqual(@as(LSN, 0), receiver.apply_lsn);
    try std.testing.expectEqual(false, receiver.connected);
}

test "WalReceiver connect" {
    const allocator = std.testing.allocator;

    const config = Config{
        .primary_conninfo = "host=primary",
        .slot_name = "test-slot",
    };

    var receiver = try WalReceiver.init(allocator, config);
    defer receiver.deinit();

    try receiver.connect(1000);

    try std.testing.expectEqual(true, receiver.connected);
    try std.testing.expectEqual(@as(LSN, 1000), receiver.write_lsn);
    try std.testing.expectEqual(@as(LSN, 1000), receiver.flush_lsn);
    try std.testing.expectEqual(@as(LSN, 1000), receiver.apply_lsn);
}

test "WalReceiver disconnect" {
    const allocator = std.testing.allocator;

    const config = Config{
        .primary_conninfo = "host=primary",
        .slot_name = "test-slot",
    };

    var receiver = try WalReceiver.init(allocator, config);
    defer receiver.deinit();

    try receiver.connect(0);
    receiver.disconnect();

    try std.testing.expectEqual(false, receiver.connected);
}

test "WalReceiver process WAL data" {
    const allocator = std.testing.allocator;

    const config = Config{
        .primary_conninfo = "host=primary",
        .slot_name = "test-slot",
    };

    var receiver = try WalReceiver.init(allocator, config);
    defer receiver.deinit();

    try receiver.connect(1000);

    const data = "test wal data";
    try receiver.processWalData(1000, 1000 + data.len, data);

    try std.testing.expectEqual(@as(LSN, 1000 + data.len), receiver.write_lsn);
    try std.testing.expectEqual(@as(LSN, 1000 + data.len), receiver.flush_lsn);
    try std.testing.expectEqual(@as(LSN, 1000 + data.len), receiver.apply_lsn);
}

test "WalReceiver process WAL data with LSN mismatch" {
    const allocator = std.testing.allocator;

    const config = Config{
        .primary_conninfo = "host=primary",
        .slot_name = "test-slot",
    };

    var receiver = try WalReceiver.init(allocator, config);
    defer receiver.deinit();

    try receiver.connect(1000);

    const data = "test data";
    const result = receiver.processWalData(2000, 2000 + data.len, data);

    try std.testing.expectError(Error.LsnMismatch, result);
}

test "WalReceiver process keepalive" {
    const allocator = std.testing.allocator;

    const config = Config{
        .primary_conninfo = "host=primary",
        .slot_name = "test-slot",
    };

    var receiver = try WalReceiver.init(allocator, config);
    defer receiver.deinit();

    const reply_needed = try receiver.processKeepalive(5000, true);
    try std.testing.expectEqual(true, reply_needed);

    const no_reply = try receiver.processKeepalive(5000, false);
    try std.testing.expectEqual(false, no_reply);
}

test "WalReceiver create status update" {
    const allocator = std.testing.allocator;

    const config = Config{
        .primary_conninfo = "host=primary",
        .slot_name = "test-slot",
    };

    var receiver = try WalReceiver.init(allocator, config);
    defer receiver.deinit();

    receiver.write_lsn = 1000;
    receiver.flush_lsn = 800;
    receiver.apply_lsn = 600;

    const msg = receiver.createStatusUpdate(true);

    try std.testing.expectEqual(@as(LSN, 1000), msg.standby_status.write_lsn);
    try std.testing.expectEqual(@as(LSN, 800), msg.standby_status.flush_lsn);
    try std.testing.expectEqual(@as(LSN, 600), msg.standby_status.apply_lsn);
    try std.testing.expectEqual(true, msg.standby_status.reply_requested);
}

test "WalReceiver should send status" {
    const allocator = std.testing.allocator;

    const config = Config{
        .primary_conninfo = "host=primary",
        .slot_name = "test-slot",
        .status_interval_ms = 100,
    };

    var receiver = try WalReceiver.init(allocator, config);
    defer receiver.deinit();

    // Initially should not send
    try std.testing.expectEqual(false, receiver.shouldSendStatus());

    // Wait for interval
    std.time.sleep(110 * std.time.ns_per_ms);

    // Now should send
    try std.testing.expectEqual(true, receiver.shouldSendStatus());

    // After creating status update, should reset
    _ = receiver.createStatusUpdate(false);
    try std.testing.expectEqual(false, receiver.shouldSendStatus());
}

test "WalReceiver create identify system message" {
    const msg = WalReceiver.createIdentifySystemMessage();
    try std.testing.expectEqual(FrontendMessage.identify_system, msg);
}

test "WalReceiver create start replication message" {
    const allocator = std.testing.allocator;

    var msg = try WalReceiver.createStartReplicationMessage(allocator, "test-slot", 5000);
    defer allocator.free(msg.start_replication.slot_name);

    try std.testing.expectEqualStrings("test-slot", msg.start_replication.slot_name);
    try std.testing.expectEqual(@as(LSN, 5000), msg.start_replication.start_lsn);
}

test "WalReceiver create slot messages" {
    const allocator = std.testing.allocator;

    var create_msg = try WalReceiver.createCreateSlotMessage(allocator, "test-slot", true);
    defer allocator.free(create_msg.create_slot.slot_name);

    try std.testing.expectEqualStrings("test-slot", create_msg.create_slot.slot_name);
    try std.testing.expectEqual(true, create_msg.create_slot.temporary);

    var drop_msg = try WalReceiver.createDropSlotMessage(allocator, "test-slot");
    defer allocator.free(drop_msg.drop_slot.slot_name);

    try std.testing.expectEqualStrings("test-slot", drop_msg.drop_slot.slot_name);
}

test "WalReceiver get replication lag" {
    const allocator = std.testing.allocator;

    const config = Config{
        .primary_conninfo = "host=primary",
        .slot_name = "test-slot",
    };

    var receiver = try WalReceiver.init(allocator, config);
    defer receiver.deinit();

    receiver.apply_lsn = 1000;

    const lag = receiver.getReplicationLag(5000);
    try std.testing.expectEqual(@as(i64, 4000), lag);

    // Replica ahead (shouldn't happen but handle gracefully)
    const no_lag = receiver.getReplicationLag(500);
    try std.testing.expectEqual(@as(i64, 0), no_lag);
}

test "WalReceiver continuous WAL application" {
    const allocator = std.testing.allocator;

    const config = Config{
        .primary_conninfo = "host=primary",
        .slot_name = "test-slot",
    };

    var receiver = try WalReceiver.init(allocator, config);
    defer receiver.deinit();

    try receiver.connect(0);

    // Apply multiple chunks
    try receiver.processWalData(0, 100, "chunk1");
    try std.testing.expectEqual(@as(LSN, 100), receiver.apply_lsn);

    try receiver.processWalData(100, 250, "chunk2");
    try std.testing.expectEqual(@as(LSN, 250), receiver.apply_lsn);

    try receiver.processWalData(250, 300, "chunk3");
    try std.testing.expectEqual(@as(LSN, 300), receiver.apply_lsn);
}

// Edge case tests

test "WalReceiver — very large replication lag" {
    const allocator = std.testing.allocator;

    const config = Config{
        .primary_conninfo = "host=primary",
        .slot_name = "test-slot",
    };

    var receiver = try WalReceiver.init(allocator, config);
    defer receiver.deinit();

    receiver.apply_lsn = 1000;

    // Test with very large primary WAL end (near u64 max)
    const large_lsn: LSN = std.math.maxInt(u64) - 100;
    const lag = receiver.getReplicationLag(large_lsn);
    try std.testing.expectEqual(@as(i64, @as(i64, @intCast(large_lsn)) - 1000), lag);
}

test "WalReceiver — empty primary conninfo" {
    const allocator = std.testing.allocator;

    const config = Config{
        .primary_conninfo = "",
        .slot_name = "test-slot",
    };

    var receiver = try WalReceiver.init(allocator, config);
    defer receiver.deinit();

    try std.testing.expectEqualStrings("", receiver.primary_conninfo);
}

test "WalReceiver — multiple disconnect calls" {
    const allocator = std.testing.allocator;

    const config = Config{
        .primary_conninfo = "host=primary",
        .slot_name = "test-slot",
    };

    var receiver = try WalReceiver.init(allocator, config);
    defer receiver.deinit();

    try receiver.connect(0);
    try std.testing.expectEqual(true, receiver.connected);

    receiver.disconnect();
    try std.testing.expectEqual(false, receiver.connected);

    // Second disconnect should be no-op
    receiver.disconnect();
    try std.testing.expectEqual(false, receiver.connected);
}

test "WalReceiver — process WAL data with zero-length data" {
    const allocator = std.testing.allocator;

    const config = Config{
        .primary_conninfo = "host=primary",
        .slot_name = "test-slot",
    };

    var receiver = try WalReceiver.init(allocator, config);
    defer receiver.deinit();

    try receiver.connect(0);

    // Process empty chunk
    try receiver.processWalData(0, 0, "");
    try std.testing.expectEqual(@as(LSN, 0), receiver.apply_lsn);
}

test "WalReceiver — very long slot name" {
    const allocator = std.testing.allocator;

    // 1024-byte slot name
    var long_slot_name: [1024]u8 = undefined;
    @memset(&long_slot_name, 's');
    const slot_name_str = long_slot_name[0..];

    var msg = try WalReceiver.createStartReplicationMessage(allocator, slot_name_str, 1000);
    defer allocator.free(msg.start_replication.slot_name);

    try std.testing.expectEqualStrings(slot_name_str, msg.start_replication.slot_name);
    try std.testing.expectEqual(@as(LSN, 1000), msg.start_replication.start_lsn);
}

test "WalReceiver — zero status interval" {
    const allocator = std.testing.allocator;

    const config = Config{
        .primary_conninfo = "host=primary",
        .slot_name = "test-slot",
        .status_interval_ms = 0,
    };

    var receiver = try WalReceiver.init(allocator, config);
    defer receiver.deinit();

    // With zero interval, should always return true
    try std.testing.expectEqual(true, receiver.shouldSendStatus());
}

test "WalReceiver — keepalive with reply not requested" {
    const allocator = std.testing.allocator;

    const config = Config{
        .primary_conninfo = "host=primary",
        .slot_name = "test-slot",
    };

    var receiver = try WalReceiver.init(allocator, config);
    defer receiver.deinit();

    const reply_requested = receiver.processKeepalive(5000, false);
    try std.testing.expectEqual(false, reply_requested);
}

test "WalReceiver — process WAL data updates all LSN fields" {
    const allocator = std.testing.allocator;

    const config = Config{
        .primary_conninfo = "host=primary",
        .slot_name = "test-slot",
    };

    var receiver = try WalReceiver.init(allocator, config);
    defer receiver.deinit();

    try receiver.connect(0);

    try receiver.processWalData(0, 1000, "test-data");

    // All LSN fields should be updated
    try std.testing.expectEqual(@as(LSN, 1000), receiver.write_lsn);
    try std.testing.expectEqual(@as(LSN, 1000), receiver.flush_lsn);
    try std.testing.expectEqual(@as(LSN, 1000), receiver.apply_lsn);
}
