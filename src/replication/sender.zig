// WAL Sender Process for Silica
//
// Streams WAL records from primary to replica over TCP.
// Runs on the primary server and sends WAL data to connected replicas.

const std = @import("std");
const Allocator = std.mem.Allocator;
const protocol = @import("protocol.zig");
const slot = @import("slot.zig");
const sync = @import("sync.zig");
const wal_mod = @import("../tx/wal.zig");
const page_mod = @import("../storage/page.zig");
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
    /// Synchronous replication coordinator (optional)
    sync_coordinator: ?*sync.SyncCoordinator,
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
    /// Optional TCP stream for transport (phase 2+)
    stream: ?std.net.Stream = null,
    /// Optional real Wal for phase 3+ WAL streaming (null = stub mode)
    wal: ?*wal_mod.Wal = null,
    /// Optional mutex for thread-safe WAL access (phase 3+)
    wal_mutex: ?*std.Thread.Mutex = null,

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
            .sync_coordinator = null,
            .slot_name = null,
            .current_lsn = 0,
            .last_keepalive = std.time.microTimestamp(),
            .wal_file = null,
            .system_id = system_id_copy,
            .timeline_id = timeline_id,
            .wal_end = 0,
        };
    }

    /// Set synchronous replication coordinator
    pub fn setSyncCoordinator(self: *WalSender, coordinator: *sync.SyncCoordinator) void {
        self.sync_coordinator = coordinator;
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

        // Register with synchronous replication coordinator
        if (self.sync_coordinator) |coord| {
            try coord.registerStandby(slot_name);
        }
    }

    /// Stop replication and deactivate slot
    pub fn stopReplication(self: *WalSender) !void {
        if (self.slot_name) |name| {
            // Unregister from synchronous replication coordinator
            if (self.sync_coordinator) |coord| {
                coord.unregisterStandby(name);
            }

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

            // Update synchronous replication coordinator
            if (self.sync_coordinator) |coord| {
                try coord.updateFlushLSN(name, flush_lsn);
            }
        }
        _ = write_lsn;
        _ = apply_lsn;
    }

    /// Pack Lsn into u64: (checkpoint_seq << 32) | frame_index
    fn packLsn(lsn: wal_mod.Lsn) LSN {
        return (@as(LSN, @intCast(lsn.checkpoint_seq)) << 32) |
               (@as(LSN, @intCast(lsn.frame_index)));
    }

    /// Unpack u64 into Lsn: checkpoint_seq = val >> 32, frame_index = val & 0xFFFFFFFF
    fn unpackLsn(val: LSN) wal_mod.Lsn {
        return .{
            .checkpoint_seq = @intCast(val >> 32),
            .frame_index = @intCast(val & 0xFFFFFFFF),
        };
    }

    /// Read WAL data chunk at current LSN
    /// Returns null if no data available (end of WAL reached)
    pub fn readWalChunk(self: *WalSender, buf: []u8) !?usize {
        // Phase 3: if no real WAL is wired in, use stub behavior
        if (self.wal == null) {
            if (self.current_lsn >= self.wal_end) {
                return null; // No data available
            }

            const chunk_size = @min(buf.len, self.config.max_chunk_size);
            const available = self.wal_end - self.current_lsn;
            const to_read = @min(chunk_size, @as(usize, @intCast(available)));

            // TODO: Actual WAL file reading implementation
            // For now, return empty chunk (stub)
            _ = to_read;
            return 0;
        }

        // Real WAL streaming path (phase 3+)
        // Lock mutex if provided, otherwise proceed unlocked
        if (self.wal_mutex) |mutex| {
            mutex.lock();
            defer mutex.unlock();
        }

        return try self.readWalChunkLocked(buf);
    }

    /// Helper: reads WAL chunk assuming wal_mutex (if needed) is already held
    fn readWalChunkLocked(self: *WalSender, buf: []u8) !?usize {
        const wal = self.wal.?;

        // Unpack current_lsn
        const start_lsn = unpackLsn(self.current_lsn);

        // Refresh wal_end from the current committed frontier
        const current_lsn_from_wal = wal.currentLsn();
        self.wal_end = packLsn(current_lsn_from_wal);

        // If already at end, return null (caught up)
        if (self.current_lsn >= self.wal_end) {
            return null;
        }

        // Read frames from WAL
        const result = wal.readRawFrames(start_lsn, buf) catch |err| {
            return switch (err) {
                error.LsnFromEarlierEpoch => Error.WalDataNotAvailable,
                else => err,
            };
        };

        // Pack the next_lsn back into u64
        self.current_lsn = packLsn(result.next_lsn);

        // Return bytes_read as Some (or wrap in optional if needed)
        // Note: result.bytes_read can be 0 if we're caught up, but we already
        // checked current_lsn >= wal_end above, so this shouldn't happen.
        // However, due to timing, after refresh we might be exactly at end.
        return if (result.bytes_read > 0) result.bytes_read else null;
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

test "WalSender — setSyncCoordinator" {
    const allocator = std.testing.allocator;

    var slot_mgr = slot.SlotManager.init(allocator);
    defer slot_mgr.deinit();

    var sender = try WalSender.init(allocator, &slot_mgr, "system", 1, .{});
    defer sender.deinit();

    var coordinator = try sync.SyncCoordinator.init(allocator, .on, "replica1");
    defer coordinator.deinit();

    sender.setSyncCoordinator(&coordinator);
    try std.testing.expect(sender.sync_coordinator != null);
}

test "WalSender — register/unregister standby with coordinator" {
    const allocator = std.testing.allocator;

    var slot_mgr = slot.SlotManager.init(allocator);
    defer slot_mgr.deinit();

    var sender = try WalSender.init(allocator, &slot_mgr, "system", 1, .{});
    defer sender.deinit();

    var coordinator = try sync.SyncCoordinator.init(allocator, .on, "replica1");
    defer coordinator.deinit();
    sender.setSyncCoordinator(&coordinator);

    // Create slot
    try slot_mgr.createSlot("replica1", false);

    // Start replication (should register with coordinator)
    try sender.startReplication("replica1", 0);
    try std.testing.expectEqual(@as(u32, 1), coordinator.getSyncStandbyCount());

    // Stop replication (should unregister)
    try sender.stopReplication();
    try std.testing.expectEqual(@as(u32, 0), coordinator.getSyncStandbyCount());
}

test "WalSender — processStandbyStatus updates coordinator" {
    const allocator = std.testing.allocator;

    var slot_mgr = slot.SlotManager.init(allocator);
    defer slot_mgr.deinit();

    var sender = try WalSender.init(allocator, &slot_mgr, "system", 1, .{});
    defer sender.deinit();

    var coordinator = try sync.SyncCoordinator.init(allocator, .on, "replica1");
    defer coordinator.deinit();
    sender.setSyncCoordinator(&coordinator);

    // Create slot and start replication
    try slot_mgr.createSlot("replica1", false);
    try sender.startReplication("replica1", 0);

    // Process standby status (should update flush LSN in coordinator)
    try sender.processStandbyStatus(1000, 1000, 1000);

    const min_lsn = coordinator.getMinSyncFlushLSN();
    try std.testing.expectEqual(@as(LSN, 1000), min_lsn);
}

test "WalSender — works without coordinator (backward compat)" {
    const allocator = std.testing.allocator;

    var slot_mgr = slot.SlotManager.init(allocator);
    defer slot_mgr.deinit();

    var sender = try WalSender.init(allocator, &slot_mgr, "system", 1, .{});
    defer sender.deinit();

    // Create slot and start replication without coordinator
    try slot_mgr.createSlot("replica1", false);
    try sender.startReplication("replica1", 0);

    // Should work fine without coordinator
    try sender.processStandbyStatus(1000, 1000, 1000);
    try sender.stopReplication();
}

// ============================================================================
// Error Path Tests
// ============================================================================

test "WalSender — SlotNotFound error" {
    const allocator = std.testing.allocator;

    var slot_mgr = slot.SlotManager.init(allocator);
    defer slot_mgr.deinit();

    var sender = try WalSender.init(allocator, &slot_mgr, "system", 1, .{});
    defer sender.deinit();

    // Try to start replication with non-existent slot
    const result = sender.startReplication("nonexistent-slot", 0);
    try std.testing.expectError(Error.SlotNotFound, result);
}

test "WalSender — SlotAlreadyActive error" {
    const allocator = std.testing.allocator;

    var slot_mgr = slot.SlotManager.init(allocator);
    defer slot_mgr.deinit();

    var sender = try WalSender.init(allocator, &slot_mgr, "system", 1, .{});
    defer sender.deinit();

    try slot_mgr.createSlot("test-slot", false);
    defer slot_mgr.dropSlot("test-slot") catch {};

    // Activate the slot externally
    try slot_mgr.activateSlot("test-slot");

    // Try to start replication on already active slot
    const result = sender.startReplication("test-slot", 0);
    try std.testing.expectError(slot.SlotError.SlotInUse, result);
}

test "WalSender — stopReplication when not active is no-op" {
    const allocator = std.testing.allocator;

    var slot_mgr = slot.SlotManager.init(allocator);
    defer slot_mgr.deinit();

    var sender = try WalSender.init(allocator, &slot_mgr, "system", 1, .{});
    defer sender.deinit();

    // Stop without starting should be no-op (doesn't error)
    try sender.stopReplication();
    try std.testing.expect(sender.slot_name == null);
}

test "WalSender — processStandbyStatus when not active is no-op" {
    const allocator = std.testing.allocator;

    var slot_mgr = slot.SlotManager.init(allocator);
    defer slot_mgr.deinit();

    var sender = try WalSender.init(allocator, &slot_mgr, "system", 1, .{});
    defer sender.deinit();

    // Process status without active replication should be no-op (doesn't error)
    try sender.processStandbyStatus(1000, 1000, 1000);
    try std.testing.expect(sender.slot_name == null);
}

test "WalSender — LSN ordering validation" {
    const allocator = std.testing.allocator;

    var slot_mgr = slot.SlotManager.init(allocator);
    defer slot_mgr.deinit();

    var sender = try WalSender.init(allocator, &slot_mgr, "system", 1, .{});
    defer sender.deinit();

    try slot_mgr.createSlot("test-slot", false);
    defer slot_mgr.dropSlot("test-slot") catch {};

    try sender.startReplication("test-slot", 0);

    // Write LSN > Flush LSN > Apply LSN should succeed
    try sender.processStandbyStatus(3000, 2000, 1000);

    // Slot confirmed_flush_lsn should be updated to flush_lsn
    const slot_info = try slot_mgr.getSlot("test-slot");
    try std.testing.expectEqual(@as(LSN, 2000), slot_info.confirmed_flush_lsn);
}

// ============================================================================
// Phase 3: Real WAL Streaming Tests
// ============================================================================

test "Phase 3: regression guard — readWalChunk with null wal unchanged behavior" {
    const allocator = std.testing.allocator;

    var slot_mgr = slot.SlotManager.init(allocator);
    defer slot_mgr.deinit();

    var sender = try WalSender.init(allocator, &slot_mgr, "system", 1, .{});
    defer sender.deinit();

    // Explicitly verify wal is null (regression guard)
    try std.testing.expectEqual(@as(?*wal_mod.Wal, null), sender.wal);

    // With wal = null, readWalChunk should behave like the old stub:
    // When current_lsn >= wal_end, return null; otherwise return 0 (no actual data)
    sender.current_lsn = 100;
    sender.wal_end = 100;

    var buf: [1024]u8 = undefined;
    const result = try sender.readWalChunk(&buf);

    // Must return null (caught up)
    try std.testing.expectEqual(@as(?usize, null), result);

    // Try again with data available (but wal=null means stub behavior)
    sender.wal_end = 200;
    const result2 = try sender.readWalChunk(&buf);

    // Stub always returns 0 when wal=null, even if current_lsn < wal_end
    try std.testing.expectEqual(@as(?usize, 0), result2);
}

test "Phase 3: real single-threaded round-trip — write frames, read via readWalChunk" {
    const allocator = std.testing.allocator;
    const wal_path = "/tmp/test_sender_phase3_wal.db";
    defer std.fs.cwd().deleteFile(wal_path) catch {};
    defer std.fs.cwd().deleteFile(wal_path ++ "-wal") catch {};

    // Create a real Wal
    var wal = try wal_mod.Wal.init(allocator, wal_path, 4096);
    defer wal.deinit();

    // Create a WalSender with wal pointer
    var slot_mgr = slot.SlotManager.init(allocator);
    defer slot_mgr.deinit();

    var sender = try WalSender.init(allocator, &slot_mgr, "system", 1, .{ .max_chunk_size = 8192 });
    defer sender.deinit();

    // Wire in the real wal (phase 3: these fields will exist)
    sender.wal = &wal;
    sender.wal_mutex = null; // no concurrency for this test

    // Write 2 frames with known content and commit
    var page_data1: [4096]u8 = undefined;
    @memset(&page_data1, 0x11);
    try wal.writeFrame(42, &page_data1);

    var page_data2: [4096]u8 = undefined;
    @memset(&page_data2, 0x22);
    try wal.writeFrame(43, &page_data2);

    try wal.commit(2); // 2 pages in database

    // Initialize sender at LSN 0
    sender.current_lsn = 0;

    // Allocate buffer large enough for 2 frames
    const frame_size = wal_mod.WAL_FRAME_HEADER_SIZE + 4096;
    var buf = try allocator.alloc(u8, frame_size * 3);
    defer allocator.free(buf);

    // Read first chunk (should get at least 1 frame)
    const chunk1 = try sender.readWalChunk(buf);
    try std.testing.expect(chunk1 != null);
    try std.testing.expect(chunk1.? > 0);

    // Verify frame header was read correctly
    const fh1 = wal_mod.WalFrameHeader.deserialize(buf[0..wal_mod.WAL_FRAME_HEADER_SIZE]);
    try std.testing.expectEqual(@as(u32, 42), fh1.page_id); // First frame should be page 42

    // Verify page data matches what we wrote
    const expected_page1 = buf[wal_mod.WAL_FRAME_HEADER_SIZE..][0..4096];
    try std.testing.expectEqualSlices(u8, &page_data1, expected_page1);

    // current_lsn should have advanced
    try std.testing.expect(sender.current_lsn > 0);

    // Continue reading (may get second frame or need another call)
    const chunk2 = try sender.readWalChunk(buf);
    if (chunk2 != null and chunk2.? > 0) {
        // Got second frame
        const fh2 = wal_mod.WalFrameHeader.deserialize(buf[0..wal_mod.WAL_FRAME_HEADER_SIZE]);
        try std.testing.expectEqual(@as(u32, 43), fh2.page_id);
        const expected_page2 = buf[wal_mod.WAL_FRAME_HEADER_SIZE..][0..4096];
        try std.testing.expectEqualSlices(u8, &page_data2, expected_page2);
    }

    // When caught up, should return null
    const caught_up = try sender.readWalChunk(buf);
    try std.testing.expectEqual(@as(?usize, null), caught_up);
}

test "Phase 3: caught-up returns null" {
    const allocator = std.testing.allocator;
    const wal_path = "/tmp/test_sender_caughtup_wal.db";
    defer std.fs.cwd().deleteFile(wal_path) catch {};
    defer std.fs.cwd().deleteFile(wal_path ++ "-wal") catch {};

    var wal = try wal_mod.Wal.init(allocator, wal_path, 4096);
    defer wal.deinit();

    // Write and commit 1 frame
    var page_data: [4096]u8 = undefined;
    @memset(&page_data, 0xAA);
    try wal.writeFrame(10, &page_data);
    try wal.commit(1);

    var slot_mgr = slot.SlotManager.init(allocator);
    defer slot_mgr.deinit();

    var sender = try WalSender.init(allocator, &slot_mgr, "system", 1, .{});
    defer sender.deinit();

    sender.wal = &wal;
    sender.wal_mutex = null;

    // Get current LSN and set sender to that position
    const current = wal.currentLsn();
    // Pack LSN into u64: (checkpoint_seq << 32) | frame_index
    sender.current_lsn = (@as(LSN, @intCast(current.checkpoint_seq)) << 32) |
                         (@as(LSN, @intCast(current.frame_index)));

    // Allocate minimum buffer
    var buf: [4096 + 24]u8 = undefined;

    // Should return null (already at end)
    const result = try sender.readWalChunk(&buf);
    try std.testing.expectEqual(@as(?usize, null), result);
}

test "Phase 3: checkpoint-recycled data maps to WalDataNotAvailable" {
    const allocator = std.testing.allocator;
    const wal_path = "/tmp/test_sender_checkpoint_wal.db";
    defer std.fs.cwd().deleteFile(wal_path) catch {};
    defer std.fs.cwd().deleteFile(wal_path ++ "-wal") catch {};

    // Initialize pager for checkpoint
    const pager = try allocator.create(page_mod.Pager);
    defer allocator.destroy(pager);
    pager.* = try page_mod.Pager.init(allocator, wal_path, .{ .page_size = 4096 });
    defer pager.deinit();
    pager.page_count = 21; // Must be >= page_id + 1 for page 20

    var wal = try wal_mod.Wal.init(allocator, wal_path, 4096);
    defer wal.deinit();

    // Write frame in epoch 0
    var page_data: [4096]u8 = undefined;
    @memset(&page_data, 0xBB);
    try wal.writeFrame(20, &page_data);
    try wal.commit(1);

    // Get the LSN at frame 0 (before checkpoint)
    const pre_checkpoint_lsn = wal.lsnAtFrame(0);
    try std.testing.expectEqual(@as(u32, 0), pre_checkpoint_lsn.checkpoint_seq);

    // Perform checkpoint (increments epoch, truncates old data)
    try wal.checkpoint(pager);

    // Now checkpoint_seq has incremented
    try std.testing.expect(wal.header.checkpoint_seq > 0);

    var slot_mgr = slot.SlotManager.init(allocator);
    defer slot_mgr.deinit();

    var sender = try WalSender.init(allocator, &slot_mgr, "system", 1, .{});
    defer sender.deinit();

    sender.wal = &wal;
    sender.wal_mutex = null;

    // Try to read from the old epoch LSN (checkpoint already recycled it)
    // Pack old LSN into u64
    sender.current_lsn = (0 << 32) | 0; // epoch 0, frame 0

    var buf: [4096 + 24]u8 = undefined;

    // Should return error.WalDataNotAvailable (from error.LsnFromEarlierEpoch)
    const result = sender.readWalChunk(&buf);
    try std.testing.expectError(Error.WalDataNotAvailable, result);
}

test "Phase 3: concurrency — reader and writer threads with mutex" {
    const allocator = std.testing.allocator;
    const wal_path = "/tmp/test_sender_concurrent_wal.db";
    defer std.fs.cwd().deleteFile(wal_path) catch {};
    defer std.fs.cwd().deleteFile(wal_path ++ "-wal") catch {};

    // Create shared WAL and mutex
    var wal = try wal_mod.Wal.init(allocator, wal_path, 4096);
    defer wal.deinit();

    var wal_mutex = std.Thread.Mutex{};

    // Create sender with WAL pointer and mutex
    var slot_mgr = slot.SlotManager.init(allocator);
    defer slot_mgr.deinit();

    var sender = try WalSender.init(allocator, &slot_mgr, "system", 1, .{ .max_chunk_size = 8192 });
    defer sender.deinit();

    sender.wal = &wal;
    sender.wal_mutex = &wal_mutex;
    sender.current_lsn = 0;

    // Thread-safe counters for verification
    var bytes_read_total: usize = 0;
    var frames_read: u32 = 0;

    // Writer thread: writes 3 frames
    const writer_thread = try std.Thread.spawn(.{}, struct {
        fn run(w: *wal_mod.Wal, m: *std.Thread.Mutex) !void {
            // Write frame 1
            {
                m.lock();
                defer m.unlock();
                var page_data: [4096]u8 = undefined;
                @memset(&page_data, 0x11);
                try w.writeFrame(100, &page_data);
            }

            std.Thread.sleep(10 * std.time.ns_per_ms); // Small delay between writes

            // Write frame 2 and commit
            {
                m.lock();
                defer m.unlock();
                var page_data: [4096]u8 = undefined;
                @memset(&page_data, 0x22);
                try w.writeFrame(101, &page_data);
                try w.commit(2);
            }

            std.Thread.sleep(10 * std.time.ns_per_ms);

            // Write frame 3 and commit
            {
                m.lock();
                defer m.unlock();
                var page_data: [4096]u8 = undefined;
                @memset(&page_data, 0x33);
                try w.writeFrame(102, &page_data);
                try w.commit(3);
            }
        }
    }.run, .{ &wal, &wal_mutex });

    // Reader thread: reads until caught up
    const reader_thread = try std.Thread.spawn(.{}, struct {
        fn run(s: *WalSender, total_bytes: *usize, frames: *u32) !void {
            const frame_size = wal_mod.WAL_FRAME_HEADER_SIZE + 4096;
            const buf = try std.testing.allocator.alloc(u8, frame_size * 5);
            defer std.testing.allocator.free(buf);

            // Read in a loop until caught up
            var attempts: u32 = 0;
            while (attempts < 50) : (attempts += 1) {
                const chunk = try s.readWalChunk(buf);
                if (chunk == null) {
                    // Caught up
                    std.Thread.sleep(5 * std.time.ns_per_ms);
                    continue;
                }
                if (chunk.? > 0) {
                    total_bytes.* += chunk.?;
                    // Count complete frames
                    frames.* += @intCast(chunk.? / frame_size);
                }
                std.Thread.sleep(2 * std.time.ns_per_ms);
            }
        }
    }.run, .{ &sender, &bytes_read_total, &frames_read });

    // Wait for both threads to complete
    writer_thread.join();
    reader_thread.join();

    // Verify we read the frames
    try std.testing.expect(frames_read >= 2); // At least 2 frames should be readable
    try std.testing.expect(bytes_read_total > 0); // Should have read some bytes
}
