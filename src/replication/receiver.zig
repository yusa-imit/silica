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
const wal_mod = @import("../tx/wal.zig");
const page_mod = @import("../storage/page.zig");

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
    /// Optional TCP stream for transport (phase 2+)
    stream: ?std.net.Stream = null,
    /// Optional local WAL for phase 4+ real frame application
    local_wal: ?*wal_mod.Wal = null,
    /// Optional local Pager for phase 4+ real frame application
    local_pager: ?*page_mod.Pager = null,

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
            .apply_buffer = std.ArrayList(u8){},
        };
    }

    pub fn deinit(self: *WalReceiver) void {
        if (self.wal_file) |*file| {
            file.close();
        }
        self.apply_buffer.deinit(self.allocator);
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
        try self.apply_buffer.appendSlice(self.allocator, data);
        self.write_lsn = wal_end;

        // Phase 4+: If local_wal is set, apply frames to local storage
        if (self.local_wal != null) {
            const local_wal = self.local_wal.?;
            const frame_size = wal_mod.WAL_FRAME_HEADER_SIZE + local_wal.page_size;

            // Verify data is whole frames
            if (data.len % frame_size != 0) {
                return Error.InvalidWalData;
            }

            // Parse and apply each frame
            var offset: usize = 0;
            while (offset < data.len) : (offset += frame_size) {
                const frame_bytes = data[offset..][0..frame_size];
                try local_wal.appendRawFrame(frame_bytes);
            }
        }

        try self.flushWal();
        try self.applyWal();
    }

    /// Process keepalive message
    pub fn processKeepalive(
        _: *WalReceiver,
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
        // Phase 4+: When local_wal and local_pager are set, checkpoint after applying frames
        // to ensure committed changes are written to the main database file
        if (self.local_wal != null and self.local_pager != null) {
            try self.local_wal.?.checkpoint(self.local_pager.?);
        }

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
    std.Thread.sleep(110 * std.time.ns_per_ms);

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

    const msg = try WalReceiver.createStartReplicationMessage(allocator, "test-slot", 5000);
    defer allocator.free(msg.start_replication.slot_name);

    try std.testing.expectEqualStrings("test-slot", msg.start_replication.slot_name);
    try std.testing.expectEqual(@as(LSN, 5000), msg.start_replication.start_lsn);
}

test "WalReceiver create slot messages" {
    const allocator = std.testing.allocator;

    const create_msg = try WalReceiver.createCreateSlotMessage(allocator, "test-slot", true);
    defer allocator.free(create_msg.create_slot.slot_name);

    try std.testing.expectEqualStrings("test-slot", create_msg.create_slot.slot_name);
    try std.testing.expectEqual(true, create_msg.create_slot.temporary);

    const drop_msg = try WalReceiver.createDropSlotMessage(allocator, "test-slot");
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

    // Test with very large primary WAL end (but within i64 range for lag calculation)
    const large_lsn: LSN = std.math.maxInt(i64) - 100;
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

    try std.testing.expectEqualStrings("", receiver.config.primary_conninfo);
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

    const msg = try WalReceiver.createStartReplicationMessage(allocator, slot_name_str, 1000);
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

// Phase 4: Real WAL Application Tests
// ============================================================================

test "Phase 4: receiver applies real WAL frames to local Wal and Pager" {
    const allocator = std.testing.allocator;
    const src_path = "test_receiver_phase4_src.db";
    defer std.fs.cwd().deleteFile(src_path) catch {};
    defer std.fs.cwd().deleteFile(src_path ++ "-wal") catch {};

    const dst_path = "test_receiver_phase4_dst.db";
    defer std.fs.cwd().deleteFile(dst_path) catch {};
    defer std.fs.cwd().deleteFile(dst_path ++ "-wal") catch {};

    // Source side: create Wal, write pages, commit
    var src_wal = try wal_mod.Wal.init(allocator, src_path, 4096);
    defer src_wal.deinit();

    var page_data1: [4096]u8 = undefined;
    @memset(&page_data1, 0x11);
    page_data1[0] = 0x03; // PageType.leaf
    try src_wal.writeFrame(42, &page_data1);

    var page_data2: [4096]u8 = undefined;
    @memset(&page_data2, 0x22);
    page_data2[0] = 0x03; // PageType.leaf
    try src_wal.writeFrame(43, &page_data2);

    try src_wal.commit(2);

    // Read raw frames back
    const frame_size = wal_mod.WAL_FRAME_HEADER_SIZE + 4096;
    var frame_buf = try allocator.alloc(u8, frame_size * 3);
    defer allocator.free(frame_buf);

    const start_lsn = src_wal.lsnAtFrame(0);
    const read_result = try src_wal.readRawFrames(start_lsn, frame_buf);
    try std.testing.expect(read_result.bytes_read > 0);

    // Destination side: create Pager and Wal
    var dst_pager = try page_mod.Pager.init(allocator, dst_path, .{ .page_size = 4096 });
    defer dst_pager.deinit();

    var dst_wal = try wal_mod.Wal.init(allocator, dst_path, 4096);
    defer dst_wal.deinit();

    // Create receiver with local_wal and local_pager set
    const config = Config{
        .primary_conninfo = "host=primary",
        .slot_name = "test-slot",
    };

    var receiver = try WalReceiver.init(allocator, config);
    defer receiver.deinit();

    receiver.local_wal = &dst_wal;
    receiver.local_pager = &dst_pager;

    try receiver.connect(0);

    // Process the raw frame bytes
    try receiver.processWalData(0, read_result.bytes_read, frame_buf[0..read_result.bytes_read]);

    // After processWalData with commit frames, checkpoint should have run
    // So pages should be in the main database file (via dst_pager)
    var read_buf: [4096]u8 = undefined;
    try dst_pager.readPage(42, &read_buf);
    // Compare non-checksum bytes (0-11 and 16-4095); bytes 12-15 are recomputed checksum
    try std.testing.expectEqualSlices(u8, page_data1[0..12], read_buf[0..12]);
    try std.testing.expectEqualSlices(u8, page_data1[16..], read_buf[16..]);

    try dst_pager.readPage(43, &read_buf);
    try std.testing.expectEqualSlices(u8, page_data2[0..12], read_buf[0..12]);
    try std.testing.expectEqualSlices(u8, page_data2[16..], read_buf[16..]);
}

test "Phase 4: receiver with non-commit frame does not checkpoint yet" {
    const allocator = std.testing.allocator;
    const src_path = "test_receiver_phase4_noncommit_src.db";
    defer std.fs.cwd().deleteFile(src_path) catch {};
    defer std.fs.cwd().deleteFile(src_path ++ "-wal") catch {};

    const dst_path = "test_receiver_phase4_noncommit_dst.db";
    defer std.fs.cwd().deleteFile(dst_path) catch {};
    defer std.fs.cwd().deleteFile(dst_path ++ "-wal") catch {};

    // Source side: write 2 frames and commit
    var src_wal = try wal_mod.Wal.init(allocator, src_path, 4096);
    defer src_wal.deinit();

    var page_data1: [4096]u8 = undefined;
    @memset(&page_data1, 0x33);
    try src_wal.writeFrame(100, &page_data1);

    var page_data2: [4096]u8 = undefined;
    @memset(&page_data2, 0x44);
    try src_wal.writeFrame(101, &page_data2);

    try src_wal.commit(2);

    // Read raw frames
    const frame_size = wal_mod.WAL_FRAME_HEADER_SIZE + 4096;
    var frame_buf = try allocator.alloc(u8, frame_size * 2);
    defer allocator.free(frame_buf);

    const start_lsn = src_wal.lsnAtFrame(0);
    const read_result = try src_wal.readRawFrames(start_lsn, frame_buf);
    try std.testing.expect(read_result.bytes_read > 0);

    // Extract just the first frame (non-commit)
    const first_frame = frame_buf[0..frame_size];

    // Destination side
    var dst_pager = try page_mod.Pager.init(allocator, dst_path, .{ .page_size = 4096 });
    defer dst_pager.deinit();

    var dst_wal = try wal_mod.Wal.init(allocator, dst_path, 4096);
    defer dst_wal.deinit();

    const config = Config{
        .primary_conninfo = "host=primary",
        .slot_name = "test-slot",
    };

    var receiver = try WalReceiver.init(allocator, config);
    defer receiver.deinit();

    receiver.local_wal = &dst_wal;
    receiver.local_pager = &dst_pager;

    try receiver.connect(0);

    // Process only the first (non-commit) frame
    try receiver.processWalData(0, first_frame.len, first_frame);

    // After processing non-commit frame, checkpoint should NOT have run
    // So page should NOT be in the main database file yet
    // (It would be in WAL pending, but not in the main DB)
    try std.testing.expectEqual(@as(u32, 0), dst_wal.committed_frame_count);

    // Verify pager has unchanged page_count (no checkpoint occurred)
    // pager starts with 2 pages (header + schema root), unchanged since no checkpoint ran
    try std.testing.expectEqual(@as(u32, 2), dst_pager.page_count);
}

test "Phase 4: regression guard — receiver with null local_wal/local_pager unchanged behavior" {
    const allocator = std.testing.allocator;

    const config = Config{
        .primary_conninfo = "host=primary",
        .slot_name = "test-slot",
    };

    var receiver = try WalReceiver.init(allocator, config);
    defer receiver.deinit();

    // Explicitly verify local_wal and local_pager are null (regression guard)
    try std.testing.expectEqual(@as(?*wal_mod.Wal, null), receiver.local_wal);
    try std.testing.expectEqual(@as(?*page_mod.Pager, null), receiver.local_pager);

    // With both null, processWalData should behave like old stub:
    // Just update LSNs and clear buffer, no actual WAL/Pager application
    try receiver.connect(0);

    const data = "stub-wal-data";
    try receiver.processWalData(0, data.len, data);

    // LSNs should be updated
    try std.testing.expectEqual(@as(LSN, @intCast(data.len)), receiver.write_lsn);
    try std.testing.expectEqual(@as(LSN, @intCast(data.len)), receiver.apply_lsn);

    // Buffer should be cleared
    try std.testing.expectEqual(@as(usize, 0), receiver.apply_buffer.items.len);
}
