// WAL Sender Process for Silica
//
// Streams WAL records from primary to replica over TCP.
// Runs on the primary server and sends WAL data to connected replicas.

const std = @import("std");
const Allocator = std.mem.Allocator;
const protocol = @import("protocol.zig");
const slot = @import("slot.zig");
const LSN = protocol.LSN;
const SlotState = protocol.SlotState;
const BackendMessage = protocol.BackendMessage;
const FrontendMessage = protocol.FrontendMessage;

/// WAL Sender errors
pub const Error = error{
    /// Slot not found
    SlotNotFound,
    /// Slot already active
    SlotAlreadyActive,
    /// Invalid LSN (before restart_lsn)
    InvalidLSN,
    /// WAL data not available (already recycled)
    WalDataNotAvailable,
    /// Connection closed
    ConnectionClosed,
    /// Protocol error
    ProtocolError,
} || Allocator.Error || std.fs.File.WriteError || std.fs.File.ReadError;

/// WAL Sender configuration
pub const Config = struct {
    /// Maximum WAL data chunk size in bytes
    max_chunk_size: u32 = 8192,
    /// Keepalive interval in milliseconds
    keepalive_interval_ms: u64 = 10_000,
    /// Timeout for replica feedback in milliseconds
    feedback_timeout_ms: u64 = 60_000,
};

/// WAL Sender state
pub const WalSender = struct {
    /// Memory allocator
    allocator: Allocator,
    /// Configuration
    config: Config,
    /// Replication slot manager
    slot_manager: *slot.SlotManager,
    /// Current slot name
    slot_name: ?[]const u8,
    /// Current streaming LSN (next position to send)
    current_lsn: LSN,
    /// Last keepalive timestamp
    last_keepalive: i64,
    /// WAL file handle (for reading WAL data)
    wal_file: ?std.fs.File,
    /// System identifier
    system_id: []const u8,
    /// Timeline ID
    timeline_id: u32,
    /// Current WAL end position
    wal_end: LSN,

    pub fn init(
        allocator: Allocator,
        slot_manager: *slot.SlotManager,
        system_id: []const u8,
        timeline_id: u32,
        config: Config,
    ) !WalSender {
        const system_id_copy = try allocator.dupe(u8, system_id);
        return .{
            .allocator = allocator,
            .config = config,
            .slot_manager = slot_manager,
            .slot_name = null,
            .current_lsn = 0,
            .last_keepalive = std.time.microTimestamp(),
            .wal_file = null,
            .system_id = system_id_copy,
            .timeline_id = timeline_id,
            .wal_end = 0,
        };
    }

    pub fn deinit(self: *WalSender) void {
        if (self.wal_file) |*file| {
            file.close();
        }
        if (self.slot_name) |name| {
            self.allocator.free(name);
        }
        self.allocator.free(self.system_id);
    }

    /// Start replication from specified LSN
    pub fn startReplication(self: *WalSender, slot_name: []const u8, start_lsn: LSN) !void {
        // Activate slot
        try self.slot_manager.activateSlot(slot_name);

        // Get slot info to validate start_lsn
        const slot_info = try self.slot_manager.getSlot(slot_name);
        if (start_lsn < slot_info.restart_lsn) {
            return Error.InvalidLSN;
        }

        // Set current slot
        if (self.slot_name) |old_name| {
            self.allocator.free(old_name);
        }
        self.slot_name = try self.allocator.dupe(u8, slot_name);
        self.current_lsn = start_lsn;
    }

    /// Stop replication and deactivate slot
    pub fn stopReplication(self: *WalSender) !void {
        if (self.slot_name) |name| {
            try self.slot_manager.deactivateSlot(name);
            self.allocator.free(name);
            self.slot_name = null;
        }
    }

    /// Process standby status update from replica
    pub fn processStandbyStatus(
        self: *WalSender,
        write_lsn: LSN,
        flush_lsn: LSN,
        apply_lsn: LSN,
    ) !void {
        if (self.slot_name) |name| {
            // Update slot's confirmed flush LSN
            try self.slot_manager.updateSlotLSN(name, null, flush_lsn);
        }
        _ = write_lsn;
        _ = apply_lsn;
    }

    /// Read WAL data chunk at current LSN
    /// Returns null if no data available (end of WAL reached)
    pub fn readWalChunk(self: *WalSender, buf: []u8) !?usize {
        if (self.current_lsn >= self.wal_end) {
            return null; // No data available
        }

        const chunk_size = @min(buf.len, self.config.max_chunk_size);
        const available = self.wal_end - self.current_lsn;
        const to_read = @min(chunk_size, @as(usize, @intCast(available)));

        // TODO: Actual WAL file reading implementation
        // For now, return empty chunk
        _ = to_read;
        return 0;
    }

    /// Create WAL_DATA message with data at current LSN
    pub fn createWalDataMessage(
        self: *WalSender,
        allocator: Allocator,
        data: []const u8,
    ) !BackendMessage {
        const wal_start = self.current_lsn;
        const wal_end = self.current_lsn + data.len;
        const data_copy = try allocator.dupe(u8, data);

        // Advance current LSN
        self.current_lsn = wal_end;

        return .{
            .wal_data = .{
                .wal_start = wal_start,
                .wal_end = wal_end,
                .server_timestamp = std.time.microTimestamp(),
                .data = data_copy,
            },
        };
    }

    /// Create KEEPALIVE message
    pub fn createKeepaliveMessage(self: *WalSender, reply_requested: bool) BackendMessage {
        self.last_keepalive = std.time.microTimestamp();
        return .{
            .keepalive = .{
                .wal_end = self.wal_end,
                .server_timestamp = self.last_keepalive,
                .reply_requested = reply_requested,
            },
        };
    }

    /// Create SYSTEM_IDENTIFICATION message
    pub fn createSystemInfoMessage(self: *WalSender, allocator: Allocator, database_name: []const u8) !BackendMessage {
        const system_id_copy = try allocator.dupe(u8, self.system_id);
        const db_name_copy = try allocator.dupe(u8, database_name);
        return .{
            .system_info = .{
                .system_id = system_id_copy,
                .timeline_id = self.timeline_id,
                .wal_position = self.wal_end,
                .database_name = db_name_copy,
            },
        };
    }

    /// Update WAL end position (called when WAL is written)
    pub fn updateWalEnd(self: *WalSender, new_wal_end: LSN) void {
        self.wal_end = new_wal_end;
    }

    /// Check if keepalive should be sent
    pub fn shouldSendKeepalive(self: *WalSender) bool {
        const now = std.time.microTimestamp();
        const elapsed_us = now - self.last_keepalive;
        const threshold_us = @as(i64, @intCast(self.config.keepalive_interval_ms)) * 1000;
        return elapsed_us >= threshold_us;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "WalSender init and deinit" {
    const allocator = std.testing.allocator;

    var slot_mgr = slot.SlotManager.init(allocator);
    defer slot_mgr.deinit();

    var sender = try WalSender.init(
        allocator,
        &slot_mgr,
        "test-system",
        1,
        .{},
    );
    defer sender.deinit();

    try std.testing.expectEqual(@as(?[]const u8, null), sender.slot_name);
    try std.testing.expectEqual(@as(LSN, 0), sender.current_lsn);
    try std.testing.expectEqualStrings("test-system", sender.system_id);
    try std.testing.expectEqual(@as(u32, 1), sender.timeline_id);
}

test "WalSender start replication" {
    const allocator = std.testing.allocator;

    var slot_mgr = slot.SlotManager.init(allocator);
    defer slot_mgr.deinit();

    // Create slot
    try slot_mgr.createSlot("test-slot", false);
    defer slot_mgr.dropSlot("test-slot") catch {};

    var sender = try WalSender.init(allocator, &slot_mgr, "system", 1, .{});
    defer sender.deinit();

    // Start replication
    try sender.startReplication("test-slot", 0);
    defer sender.stopReplication() catch {};

    try std.testing.expect(sender.slot_name != null);
    try std.testing.expectEqualStrings("test-slot", sender.slot_name.?);
    try std.testing.expectEqual(@as(LSN, 0), sender.current_lsn);

    // Verify slot is active
    const slot_info = try slot_mgr.getSlot("test-slot");
    try std.testing.expectEqual(SlotState.active, slot_info.state);
}

test "WalSender stop replication" {
    const allocator = std.testing.allocator;

    var slot_mgr = slot.SlotManager.init(allocator);
    defer slot_mgr.deinit();

    try slot_mgr.createSlot("test-slot", false);
    defer slot_mgr.dropSlot("test-slot") catch {};

    var sender = try WalSender.init(allocator, &slot_mgr, "system", 1, .{});
    defer sender.deinit();

    try sender.startReplication("test-slot", 0);
    try sender.stopReplication();

    try std.testing.expectEqual(@as(?[]const u8, null), sender.slot_name);

    // Verify slot is inactive
    const slot_info = try slot_mgr.getSlot("test-slot");
    try std.testing.expectEqual(SlotState.inactive, slot_info.state);
}

test "WalSender process standby status" {
    const allocator = std.testing.allocator;

    var slot_mgr = slot.SlotManager.init(allocator);
    defer slot_mgr.deinit();

    try slot_mgr.createSlot("test-slot", false);
    defer slot_mgr.dropSlot("test-slot") catch {};

    var sender = try WalSender.init(allocator, &slot_mgr, "system", 1, .{});
    defer sender.deinit();

    try sender.startReplication("test-slot", 0);
    defer sender.stopReplication() catch {};

    // Process status update
    try sender.processStandbyStatus(1024, 1024, 1024);

    // Verify slot LSN updated
    const slot_info = try slot_mgr.getSlot("test-slot");
    try std.testing.expectEqual(@as(LSN, 1024), slot_info.confirmed_flush_lsn);
}

test "WalSender create WAL data message" {
    const allocator = std.testing.allocator;

    var slot_mgr = slot.SlotManager.init(allocator);
    defer slot_mgr.deinit();

    var sender = try WalSender.init(allocator, &slot_mgr, "system", 1, .{});
    defer sender.deinit();

    sender.current_lsn = 1000;

    const data = "test wal data";
    const msg = try sender.createWalDataMessage(allocator, data);
    defer allocator.free(msg.wal_data.data);

    try std.testing.expectEqual(@as(LSN, 1000), msg.wal_data.wal_start);
    try std.testing.expectEqual(@as(LSN, 1000 + data.len), msg.wal_data.wal_end);
    try std.testing.expectEqualStrings(data, msg.wal_data.data);

    // Verify current LSN advanced
    try std.testing.expectEqual(@as(LSN, 1000 + data.len), sender.current_lsn);
}

test "WalSender create keepalive message" {
    const allocator = std.testing.allocator;

    var slot_mgr = slot.SlotManager.init(allocator);
    defer slot_mgr.deinit();

    var sender = try WalSender.init(allocator, &slot_mgr, "system", 1, .{});
    defer sender.deinit();

    sender.wal_end = 5000;

    const msg = sender.createKeepaliveMessage(true);

    try std.testing.expectEqual(@as(LSN, 5000), msg.keepalive.wal_end);
    try std.testing.expectEqual(true, msg.keepalive.reply_requested);
}

test "WalSender create system info message" {
    const allocator = std.testing.allocator;

    var slot_mgr = slot.SlotManager.init(allocator);
    defer slot_mgr.deinit();

    var sender = try WalSender.init(allocator, &slot_mgr, "test-system-id", 42, .{});
    defer sender.deinit();

    sender.wal_end = 8192;

    const msg = try sender.createSystemInfoMessage(allocator, "testdb");
    defer {
        allocator.free(msg.system_info.system_id);
        allocator.free(msg.system_info.database_name);
    }

    try std.testing.expectEqualStrings("test-system-id", msg.system_info.system_id);
    try std.testing.expectEqual(@as(u32, 42), msg.system_info.timeline_id);
    try std.testing.expectEqual(@as(LSN, 8192), msg.system_info.wal_position);
    try std.testing.expectEqualStrings("testdb", msg.system_info.database_name);
}

test "WalSender update WAL end" {
    const allocator = std.testing.allocator;

    var slot_mgr = slot.SlotManager.init(allocator);
    defer slot_mgr.deinit();

    var sender = try WalSender.init(allocator, &slot_mgr, "system", 1, .{});
    defer sender.deinit();

    try std.testing.expectEqual(@as(LSN, 0), sender.wal_end);

    sender.updateWalEnd(4096);
    try std.testing.expectEqual(@as(LSN, 4096), sender.wal_end);

    sender.updateWalEnd(8192);
    try std.testing.expectEqual(@as(LSN, 8192), sender.wal_end);
}

test "WalSender should send keepalive" {
    const allocator = std.testing.allocator;

    var slot_mgr = slot.SlotManager.init(allocator);
    defer slot_mgr.deinit();

    var sender = try WalSender.init(allocator, &slot_mgr, "system", 1, .{ .keepalive_interval_ms = 100 });
    defer sender.deinit();

    // Initially should not send
    try std.testing.expectEqual(false, sender.shouldSendKeepalive());

    // Wait for interval
    std.Thread.sleep(110 * std.time.ns_per_ms);

    // Now should send
    try std.testing.expectEqual(true, sender.shouldSendKeepalive());

    // After creating keepalive, should reset
    _ = sender.createKeepaliveMessage(false);
    try std.testing.expectEqual(false, sender.shouldSendKeepalive());
}

test "WalSender start replication with invalid LSN" {
    const allocator = std.testing.allocator;

    var slot_mgr = slot.SlotManager.init(allocator);
    defer slot_mgr.deinit();

    try slot_mgr.createSlot("test-slot", false);
    defer slot_mgr.dropSlot("test-slot") catch {};

    // Set restart LSN to 1000
    try slot_mgr.activateSlot("test-slot");
    try slot_mgr.updateSlotLSN("test-slot", 1000, null);
    try slot_mgr.deactivateSlot("test-slot");

    var sender = try WalSender.init(allocator, &slot_mgr, "system", 1, .{});
    defer sender.deinit();

    // Try to start from LSN before restart_lsn
    const result = sender.startReplication("test-slot", 500);
    try std.testing.expectError(Error.InvalidLSN, result);
}

test "WalSender read WAL chunk when no data" {
    const allocator = std.testing.allocator;

    var slot_mgr = slot.SlotManager.init(allocator);
    defer slot_mgr.deinit();

    var sender = try WalSender.init(allocator, &slot_mgr, "system", 1, .{});
    defer sender.deinit();

    sender.current_lsn = 1000;
    sender.wal_end = 1000;

    var buf: [1024]u8 = undefined;
    const result = try sender.readWalChunk(&buf);

    try std.testing.expectEqual(@as(?usize, null), result);
}

// Edge case tests

test "WalSender — very large LSN values" {
    const allocator = std.testing.allocator;

    var slot_mgr = slot.SlotManager.init(allocator);
    defer slot_mgr.deinit();

    var sender = try WalSender.init(allocator, &slot_mgr, "system", 1, .{});
    defer sender.deinit();

    // Test with LSN near u64 max
    const large_lsn: LSN = std.math.maxInt(u64) - 1000;
    sender.updateWalEnd(large_lsn);
    try std.testing.expectEqual(large_lsn, sender.wal_end);

    // Create WAL data message with large LSN
    sender.current_lsn = large_lsn - 100;
    const msg = try sender.createWalDataMessage(allocator, "test");
    defer allocator.free(msg.wal_data.data);
    try std.testing.expectEqual(large_lsn - 100, msg.wal_data.wal_start);
    try std.testing.expectEqual(large_lsn - 100 + 4, msg.wal_data.wal_end);
}

test "WalSender — very long system ID and database name" {
    const allocator = std.testing.allocator;

    var slot_mgr = slot.SlotManager.init(allocator);
    defer slot_mgr.deinit();

    // 1024-byte system ID
    var long_system_id: [1024]u8 = undefined;
    @memset(&long_system_id, 'x');
    const system_id_str = long_system_id[0..];

    var sender = try WalSender.init(allocator, &slot_mgr, system_id_str, 1, .{});
    defer sender.deinit();

    // 1024-byte database name
    var long_db_name: [1024]u8 = undefined;
    @memset(&long_db_name, 'd');
    const db_name_str = long_db_name[0..];

    const msg = try sender.createSystemInfoMessage(allocator, db_name_str);
    defer {
        allocator.free(msg.system_info.system_id);
        allocator.free(msg.system_info.database_name);
    }

    try std.testing.expectEqualStrings(system_id_str, msg.system_info.system_id);
    try std.testing.expectEqualStrings(db_name_str, msg.system_info.database_name);
}

test "WalSender — zero keepalive interval" {
    const allocator = std.testing.allocator;

    var slot_mgr = slot.SlotManager.init(allocator);
    defer slot_mgr.deinit();

    var sender = try WalSender.init(allocator, &slot_mgr, "system", 1, .{ .keepalive_interval_ms = 0 });
    defer sender.deinit();

    // With zero interval, should always return true
    try std.testing.expectEqual(true, sender.shouldSendKeepalive());
}

test "WalSender — multiple consecutive keepalive calls" {
    const allocator = std.testing.allocator;

    var slot_mgr = slot.SlotManager.init(allocator);
    defer slot_mgr.deinit();

    var sender = try WalSender.init(allocator, &slot_mgr, "system", 1, .{ .keepalive_interval_ms = 50 });
    defer sender.deinit();

    // Wait for interval
    std.Thread.sleep(60 * std.time.ns_per_ms);

    // First call should trigger
    try std.testing.expectEqual(true, sender.shouldSendKeepalive());
    _ = sender.createKeepaliveMessage(false);

    // Immediately after, should not trigger
    try std.testing.expectEqual(false, sender.shouldSendKeepalive());

    // Call again immediately
    try std.testing.expectEqual(false, sender.shouldSendKeepalive());
}

test "WalSender — empty WAL data chunk" {
    const allocator = std.testing.allocator;

    var slot_mgr = slot.SlotManager.init(allocator);
    defer slot_mgr.deinit();

    var sender = try WalSender.init(allocator, &slot_mgr, "system", 1, .{});
    defer sender.deinit();

    // Create message with empty data
    sender.current_lsn = 0;
    const msg = try sender.createWalDataMessage(allocator, "");
    defer allocator.free(msg.wal_data.data);
    try std.testing.expectEqual(@as(usize, 0), msg.wal_data.data.len);
    try std.testing.expectEqual(@as(LSN, 0), msg.wal_data.wal_start);
    try std.testing.expectEqual(@as(LSN, 0), msg.wal_data.wal_end);
}

test "WalSender — stop replication when not active" {
    const allocator = std.testing.allocator;

    var slot_mgr = slot.SlotManager.init(allocator);
    defer slot_mgr.deinit();

    var sender = try WalSender.init(allocator, &slot_mgr, "system", 1, .{});
    defer sender.deinit();

    // Stop without starting (should be no-op)
    try sender.stopReplication();
    try std.testing.expectEqual(@as(?[]const u8, null), sender.slot_name);
}
