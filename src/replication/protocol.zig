// Replication Protocol for Silica
//
// Implements streaming replication protocol for WAL transmission between
// primary and replica servers. Based on PostgreSQL's replication protocol
// with simplifications for Silica's architecture.

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Log Sequence Number (LSN) — monotonically increasing WAL position
/// Represents byte offset in the WAL stream
pub const LSN = u64;

/// Replication slot state
pub const SlotState = enum(u8) {
    inactive = 0,
    active = 1,
    reserved = 2,
};

/// Replication mode
pub const ReplicationMode = enum(u8) {
    /// Asynchronous: primary doesn't wait for replica acknowledgment
    async = 0,
    /// Synchronous: primary waits for at least one replica to acknowledge
    sync = 1,
};

/// Frontend (sender) messages — sent by replica to primary
pub const FrontendMessage = union(enum) {
    /// START_REPLICATION: begin streaming WAL from specified LSN
    /// Format: START_REPLICATION slot_name start_lsn
    start_replication: struct {
        slot_name: []const u8,
        start_lsn: LSN,
    },

    /// STANDBY_STATUS_UPDATE: replica reports progress
    /// Sent periodically to acknowledge received WAL
    standby_status: struct {
        /// Last WAL position written to disk
        write_lsn: LSN,
        /// Last WAL position flushed to disk
        flush_lsn: LSN,
        /// Last WAL position applied
        apply_lsn: LSN,
        /// Client timestamp (microseconds since epoch)
        client_timestamp: i64,
        /// Request immediate reply from primary
        reply_requested: bool,
    },

    /// CREATE_REPLICATION_SLOT: create a new replication slot
    create_slot: struct {
        slot_name: []const u8,
        /// Temporary slot (deleted when connection closes)
        temporary: bool,
    },

    /// DROP_REPLICATION_SLOT: remove a replication slot
    drop_slot: struct {
        slot_name: []const u8,
    },

    /// IDENTIFY_SYSTEM: request system identification
    identify_system: void,

    /// BASE_BACKUP: request base backup for initial replica setup
    base_backup: void,
};

/// Backend (receiver) messages — sent by primary to replica
pub const BackendMessage = union(enum) {
    /// COPYBOTH: streaming mode established
    copyboth_response: void,

    /// WAL_DATA: WAL record stream
    /// Format: 'w' + wal_start + wal_end + server_timestamp + data
    wal_data: struct {
        /// WAL start position for this chunk
        wal_start: LSN,
        /// WAL end position (exclusive)
        wal_end: LSN,
        /// Server timestamp (microseconds since epoch)
        server_timestamp: i64,
        /// WAL data bytes
        data: []const u8,
    },

    /// KEEPALIVE: heartbeat message
    /// Sent when no WAL data available to ensure connection stays alive
    keepalive: struct {
        /// Current WAL end position on primary
        wal_end: LSN,
        /// Server timestamp (microseconds since epoch)
        server_timestamp: i64,
        /// Request immediate reply from replica
        reply_requested: bool,
    },

    /// SYSTEM_IDENTIFICATION: response to IDENTIFY_SYSTEM
    system_info: struct {
        system_id: []const u8,
        timeline_id: u32,
        wal_position: LSN,
        database_name: []const u8,
    },

    /// ERROR_RESPONSE: replication error
    error_response: struct {
        message: []const u8,
    },
};

/// Replication slot information
pub const ReplicationSlot = struct {
    /// Unique slot name
    name: []const u8,
    /// Slot state
    state: SlotState,
    /// Restart LSN — oldest WAL position needed by this slot
    restart_lsn: LSN,
    /// Confirmed flush LSN — last position acknowledged by replica
    confirmed_flush_lsn: LSN,
    /// Temporary slot (deleted when connection closes)
    temporary: bool,
    /// Creation timestamp
    created_at: i64,
    /// Active since timestamp (null if inactive)
    active_since: ?i64,

    allocator: Allocator,

    pub fn init(allocator: Allocator, name: []const u8, temporary: bool) !ReplicationSlot {
        const name_copy = try allocator.dupe(u8, name);
        return .{
            .name = name_copy,
            .state = .inactive,
            .restart_lsn = 0,
            .confirmed_flush_lsn = 0,
            .temporary = temporary,
            .created_at = std.time.microTimestamp(),
            .active_since = null,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ReplicationSlot) void {
        self.allocator.free(self.name);
    }
};

/// Message type tags for wire protocol
pub const MessageTag = enum(u8) {
    // Frontend (replica → primary)
    start_replication = 'S',
    standby_status = 's',
    create_slot = 'C',
    drop_slot = 'D',
    identify_system = 'I',
    base_backup = 'B',

    // Backend (primary → replica)
    copyboth = 'W',
    wal_data = 'w',
    keepalive = 'k',
    system_info = 'i',
    error_response = 'E',
};

/// Serialize frontend message to bytes
pub fn serializeFrontendMessage(allocator: Allocator, msg: FrontendMessage) ![]u8 {
    var buf = std.ArrayList(u8){};
    errdefer buf.deinit(allocator);
    const writer = buf.writer(allocator);

    switch (msg) {
        .start_replication => |sr| {
            try writer.writeByte(@intFromEnum(MessageTag.start_replication));
            try writer.writeInt(u32, @intCast(sr.slot_name.len), .little);
            try writer.writeAll(sr.slot_name);
            try writer.writeInt(u64, sr.start_lsn, .little);
        },
        .standby_status => |ss| {
            try writer.writeByte(@intFromEnum(MessageTag.standby_status));
            try writer.writeInt(u64, ss.write_lsn, .little);
            try writer.writeInt(u64, ss.flush_lsn, .little);
            try writer.writeInt(u64, ss.apply_lsn, .little);
            try writer.writeInt(i64, ss.client_timestamp, .little);
            try writer.writeByte(if (ss.reply_requested) 1 else 0);
        },
        .create_slot => |cs| {
            try writer.writeByte(@intFromEnum(MessageTag.create_slot));
            try writer.writeInt(u32, @intCast(cs.slot_name.len), .little);
            try writer.writeAll(cs.slot_name);
            try writer.writeByte(if (cs.temporary) 1 else 0);
        },
        .drop_slot => |ds| {
            try writer.writeByte(@intFromEnum(MessageTag.drop_slot));
            try writer.writeInt(u32, @intCast(ds.slot_name.len), .little);
            try writer.writeAll(ds.slot_name);
        },
        .identify_system => {
            try writer.writeByte(@intFromEnum(MessageTag.identify_system));
        },
        .base_backup => {
            try writer.writeByte(@intFromEnum(MessageTag.base_backup));
        },
    }

    return buf.toOwnedSlice(allocator);
}

/// Serialize backend message to bytes
pub fn serializeBackendMessage(allocator: Allocator, msg: BackendMessage) ![]u8 {
    var buf = std.ArrayList(u8){};
    errdefer buf.deinit(allocator);
    const writer = buf.writer(allocator);

    switch (msg) {
        .copyboth_response => {
            try writer.writeByte(@intFromEnum(MessageTag.copyboth));
        },
        .wal_data => |wd| {
            try writer.writeByte(@intFromEnum(MessageTag.wal_data));
            try writer.writeInt(u64, wd.wal_start, .little);
            try writer.writeInt(u64, wd.wal_end, .little);
            try writer.writeInt(i64, wd.server_timestamp, .little);
            try writer.writeInt(u32, @intCast(wd.data.len), .little);
            try writer.writeAll(wd.data);
        },
        .keepalive => |ka| {
            try writer.writeByte(@intFromEnum(MessageTag.keepalive));
            try writer.writeInt(u64, ka.wal_end, .little);
            try writer.writeInt(i64, ka.server_timestamp, .little);
            try writer.writeByte(if (ka.reply_requested) 1 else 0);
        },
        .system_info => |si| {
            try writer.writeByte(@intFromEnum(MessageTag.system_info));
            try writer.writeInt(u32, @intCast(si.system_id.len), .little);
            try writer.writeAll(si.system_id);
            try writer.writeInt(u32, si.timeline_id, .little);
            try writer.writeInt(u64, si.wal_position, .little);
            try writer.writeInt(u32, @intCast(si.database_name.len), .little);
            try writer.writeAll(si.database_name);
        },
        .error_response => |er| {
            try writer.writeByte(@intFromEnum(MessageTag.error_response));
            try writer.writeInt(u32, @intCast(er.message.len), .little);
            try writer.writeAll(er.message);
        },
    }

    return buf.toOwnedSlice(allocator);
}

// ── Tests ────────────────────────────────────────────────────────────

const testing = std.testing;

test "LSN arithmetic" {
    const lsn1: LSN = 1000;
    const lsn2: LSN = 2000;
    try testing.expect(lsn2 > lsn1);
    try testing.expectEqual(@as(LSN, 1000), lsn2 - lsn1);
}

test "ReplicationSlot init and deinit" {
    const slot = try ReplicationSlot.init(testing.allocator, "test_slot", false);
    var mutable_slot = slot;
    defer mutable_slot.deinit();

    try testing.expectEqualStrings("test_slot", slot.name);
    try testing.expectEqual(SlotState.inactive, slot.state);
    try testing.expectEqual(false, slot.temporary);
    try testing.expect(slot.created_at > 0);
    try testing.expectEqual(@as(?i64, null), slot.active_since);
}

test "serialize START_REPLICATION" {
    const msg = FrontendMessage{
        .start_replication = .{
            .slot_name = "slot1",
            .start_lsn = 42,
        },
    };
    const bytes = try serializeFrontendMessage(testing.allocator, msg);
    defer testing.allocator.free(bytes);

    try testing.expectEqual(@as(u8, 'S'), bytes[0]);
    // slot_name length (4 bytes) + "slot1" (5 bytes) + start_lsn (8 bytes) = 17 bytes total
    try testing.expect(bytes.len >= 14);
}

test "serialize STANDBY_STATUS_UPDATE" {
    const msg = FrontendMessage{
        .standby_status = .{
            .write_lsn = 100,
            .flush_lsn = 90,
            .apply_lsn = 80,
            .client_timestamp = 123456789,
            .reply_requested = true,
        },
    };
    const bytes = try serializeFrontendMessage(testing.allocator, msg);
    defer testing.allocator.free(bytes);

    try testing.expectEqual(@as(u8, 's'), bytes[0]);
    // 1 (tag) + 8 (write) + 8 (flush) + 8 (apply) + 8 (timestamp) + 1 (reply) = 34 bytes
    try testing.expectEqual(@as(usize, 34), bytes.len);
}

test "serialize CREATE_SLOT" {
    const msg = FrontendMessage{
        .create_slot = .{
            .slot_name = "new_slot",
            .temporary = true,
        },
    };
    const bytes = try serializeFrontendMessage(testing.allocator, msg);
    defer testing.allocator.free(bytes);

    try testing.expectEqual(@as(u8, 'C'), bytes[0]);
}

test "serialize DROP_SLOT" {
    const msg = FrontendMessage{
        .drop_slot = .{
            .slot_name = "old_slot",
        },
    };
    const bytes = try serializeFrontendMessage(testing.allocator, msg);
    defer testing.allocator.free(bytes);

    try testing.expectEqual(@as(u8, 'D'), bytes[0]);
}

test "serialize IDENTIFY_SYSTEM" {
    const msg = FrontendMessage{ .identify_system = {} };
    const bytes = try serializeFrontendMessage(testing.allocator, msg);
    defer testing.allocator.free(bytes);

    try testing.expectEqual(@as(u8, 'I'), bytes[0]);
    try testing.expectEqual(@as(usize, 1), bytes.len);
}

test "serialize COPYBOTH_RESPONSE" {
    const msg = BackendMessage{ .copyboth_response = {} };
    const bytes = try serializeBackendMessage(testing.allocator, msg);
    defer testing.allocator.free(bytes);

    try testing.expectEqual(@as(u8, 'W'), bytes[0]);
    try testing.expectEqual(@as(usize, 1), bytes.len);
}

test "serialize WAL_DATA" {
    const wal_bytes = [_]u8{ 0x01, 0x02, 0x03, 0x04 };
    const msg = BackendMessage{
        .wal_data = .{
            .wal_start = 1000,
            .wal_end = 1004,
            .server_timestamp = 999999,
            .data = &wal_bytes,
        },
    };
    const bytes = try serializeBackendMessage(testing.allocator, msg);
    defer testing.allocator.free(bytes);

    try testing.expectEqual(@as(u8, 'w'), bytes[0]);
    // 1 (tag) + 8 (start) + 8 (end) + 8 (timestamp) + 4 (data_len) + 4 (data) = 33 bytes
    try testing.expectEqual(@as(usize, 33), bytes.len);
}

test "serialize KEEPALIVE" {
    const msg = BackendMessage{
        .keepalive = .{
            .wal_end = 5000,
            .server_timestamp = 111111,
            .reply_requested = false,
        },
    };
    const bytes = try serializeBackendMessage(testing.allocator, msg);
    defer testing.allocator.free(bytes);

    try testing.expectEqual(@as(u8, 'k'), bytes[0]);
    // 1 (tag) + 8 (wal_end) + 8 (timestamp) + 1 (reply) = 18 bytes
    try testing.expectEqual(@as(usize, 18), bytes.len);
}

test "serialize SYSTEM_INFO" {
    const msg = BackendMessage{
        .system_info = .{
            .system_id = "silica-001",
            .timeline_id = 1,
            .wal_position = 2000,
            .database_name = "testdb",
        },
    };
    const bytes = try serializeBackendMessage(testing.allocator, msg);
    defer testing.allocator.free(bytes);

    try testing.expectEqual(@as(u8, 'i'), bytes[0]);
}

test "serialize ERROR_RESPONSE" {
    const msg = BackendMessage{
        .error_response = .{
            .message = "Replication slot not found",
        },
    };
    const bytes = try serializeBackendMessage(testing.allocator, msg);
    defer testing.allocator.free(bytes);

    try testing.expectEqual(@as(u8, 'E'), bytes[0]);
}

// ── Edge Case Tests ──────────────────────────────────────────────────

test "START_REPLICATION with empty slot name" {
    const msg = FrontendMessage{
        .start_replication = .{
            .slot_name = "",
            .start_lsn = 0,
        },
    };
    const bytes = try serializeFrontendMessage(testing.allocator, msg);
    defer testing.allocator.free(bytes);

    try testing.expectEqual(@as(u8, 'S'), bytes[0]);
    // 1 (tag) + 4 (slot_name_len=0) + 0 (empty name) + 8 (start_lsn) = 13 bytes
    try testing.expectEqual(@as(usize, 13), bytes.len);
}

test "START_REPLICATION with max LSN" {
    const msg = FrontendMessage{
        .start_replication = .{
            .slot_name = "slot",
            .start_lsn = std.math.maxInt(u64),
        },
    };
    const bytes = try serializeFrontendMessage(testing.allocator, msg);
    defer testing.allocator.free(bytes);

    try testing.expectEqual(@as(u8, 'S'), bytes[0]);
    // Verify LSN is serialized correctly at max value
    const lsn_offset = 1 + 4 + 4; // tag + len + "slot"
    const lsn = std.mem.readInt(u64, bytes[lsn_offset..][0..8], .little);
    try testing.expectEqual(std.math.maxInt(u64), lsn);
}

test "STANDBY_STATUS_UPDATE with zero LSNs" {
    const msg = FrontendMessage{
        .standby_status = .{
            .write_lsn = 0,
            .flush_lsn = 0,
            .apply_lsn = 0,
            .client_timestamp = 0,
            .reply_requested = false,
        },
    };
    const bytes = try serializeFrontendMessage(testing.allocator, msg);
    defer testing.allocator.free(bytes);

    try testing.expectEqual(@as(u8, 's'), bytes[0]);
    try testing.expectEqual(@as(usize, 34), bytes.len);
}

test "STANDBY_STATUS_UPDATE with negative timestamp" {
    const msg = FrontendMessage{
        .standby_status = .{
            .write_lsn = 100,
            .flush_lsn = 100,
            .apply_lsn = 100,
            .client_timestamp = -123456789,
            .reply_requested = true,
        },
    };
    const bytes = try serializeFrontendMessage(testing.allocator, msg);
    defer testing.allocator.free(bytes);

    try testing.expectEqual(@as(u8, 's'), bytes[0]);
    // Verify negative timestamp is preserved
    const ts_offset = 1 + 8 + 8 + 8; // tag + write + flush + apply
    const ts = std.mem.readInt(i64, bytes[ts_offset..][0..8], .little);
    try testing.expectEqual(@as(i64, -123456789), ts);
}

test "CREATE_SLOT with very long slot name" {
    var long_name: [1024]u8 = undefined;
    @memset(&long_name, 'a');
    const msg = FrontendMessage{
        .create_slot = .{
            .slot_name = &long_name,
            .temporary = false,
        },
    };
    const bytes = try serializeFrontendMessage(testing.allocator, msg);
    defer testing.allocator.free(bytes);

    try testing.expectEqual(@as(u8, 'C'), bytes[0]);
    // 1 (tag) + 4 (len) + 1024 (name) + 1 (temporary) = 1030 bytes
    try testing.expectEqual(@as(usize, 1030), bytes.len);
}

test "WAL_DATA with empty data" {
    const msg = BackendMessage{
        .wal_data = .{
            .wal_start = 1000,
            .wal_end = 1000, // start == end means no data
            .server_timestamp = 0,
            .data = &[_]u8{},
        },
    };
    const bytes = try serializeBackendMessage(testing.allocator, msg);
    defer testing.allocator.free(bytes);

    try testing.expectEqual(@as(u8, 'w'), bytes[0]);
    // 1 (tag) + 8 (start) + 8 (end) + 8 (timestamp) + 4 (data_len=0) + 0 (data) = 29 bytes
    try testing.expectEqual(@as(usize, 29), bytes.len);
}

test "WAL_DATA with max LSN values" {
    const wal_bytes = [_]u8{0xFF};
    const msg = BackendMessage{
        .wal_data = .{
            .wal_start = std.math.maxInt(u64) - 1,
            .wal_end = std.math.maxInt(u64),
            .server_timestamp = std.math.maxInt(i64),
            .data = &wal_bytes,
        },
    };
    const bytes = try serializeBackendMessage(testing.allocator, msg);
    defer testing.allocator.free(bytes);

    try testing.expectEqual(@as(u8, 'w'), bytes[0]);
    try testing.expect(bytes.len > 0);
}

test "KEEPALIVE with negative timestamp" {
    const msg = BackendMessage{
        .keepalive = .{
            .wal_end = 0,
            .server_timestamp = std.math.minInt(i64),
            .reply_requested = true,
        },
    };
    const bytes = try serializeBackendMessage(testing.allocator, msg);
    defer testing.allocator.free(bytes);

    try testing.expectEqual(@as(u8, 'k'), bytes[0]);
    try testing.expectEqual(@as(usize, 18), bytes.len);
}

test "SYSTEM_INFO with empty strings" {
    const msg = BackendMessage{
        .system_info = .{
            .system_id = "",
            .timeline_id = 0,
            .wal_position = 0,
            .database_name = "",
        },
    };
    const bytes = try serializeBackendMessage(testing.allocator, msg);
    defer testing.allocator.free(bytes);

    try testing.expectEqual(@as(u8, 'i'), bytes[0]);
    // 1 (tag) + 4 (system_id_len=0) + 0 (empty) + 4 (timeline) + 8 (wal_pos) + 4 (db_len=0) + 0 (empty) = 21 bytes
    try testing.expectEqual(@as(usize, 21), bytes.len);
}

test "ERROR_RESPONSE with empty message" {
    const msg = BackendMessage{
        .error_response = .{
            .message = "",
        },
    };
    const bytes = try serializeBackendMessage(testing.allocator, msg);
    defer testing.allocator.free(bytes);

    try testing.expectEqual(@as(u8, 'E'), bytes[0]);
    // 1 (tag) + 4 (message_len=0) + 0 (empty) = 5 bytes
    try testing.expectEqual(@as(usize, 5), bytes.len);
}

test "ReplicationSlot temporary flag" {
    const slot = try ReplicationSlot.init(testing.allocator, "temp_slot", true);
    var mutable_slot = slot;
    defer mutable_slot.deinit();

    try testing.expectEqual(true, slot.temporary);
}

test "ReplicationSlot LSN updates" {
    var slot = try ReplicationSlot.init(testing.allocator, "lsn_test", false);
    defer slot.deinit();

    // Update LSNs
    slot.restart_lsn = 1000;
    slot.confirmed_flush_lsn = 500;

    try testing.expectEqual(@as(LSN, 1000), slot.restart_lsn);
    try testing.expectEqual(@as(LSN, 500), slot.confirmed_flush_lsn);
}

test "ReplicationSlot state transitions" {
    var slot = try ReplicationSlot.init(testing.allocator, "state_test", false);
    defer slot.deinit();

    try testing.expectEqual(SlotState.inactive, slot.state);

    slot.state = .active;
    slot.active_since = std.time.microTimestamp();
    try testing.expectEqual(SlotState.active, slot.state);
    try testing.expect(slot.active_since != null);

    slot.state = .reserved;
    try testing.expectEqual(SlotState.reserved, slot.state);
}
