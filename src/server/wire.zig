//! PostgreSQL Wire Protocol v3 Implementation
//!
//! Reference: https://www.postgresql.org/docs/current/protocol.html
//!
//! Message Format:
//!   [type: u8] [length: i32 (network order)] [payload...]
//!
//! Frontend (Client) Messages:
//!   'Q' - Query (simple query)
//!   'P' - Parse (extended query)
//!   'B' - Bind
//!   'D' - Describe
//!   'E' - Execute
//!   'C' - Close
//!   'S' - Sync
//!   'X' - Terminate
//!
//! Backend (Server) Messages:
//!   'R' - Authentication
//!   'K' - BackendKeyData
//!   'Z' - ReadyForQuery
//!   'T' - RowDescription
//!   'D' - DataRow
//!   'C' - CommandComplete
//!   'E' - ErrorResponse
//!   'N' - NoticeResponse
//!   '1' - ParseComplete
//!   '2' - BindComplete
//!   's' - PortalSuspended
//!   'n' - NoData
//!   'S' - ParameterStatus
//!   't' - ParameterDescription

const std = @import("std");
const Allocator = std.mem.Allocator;

// ── Message Types ──────────────────────────────────────────────────────

/// Frontend message types (client → server)
pub const FrontendMessageType = enum(u8) {
    query = 'Q',
    parse = 'P',
    bind = 'B',
    describe = 'D',
    execute = 'E',
    close = 'C',
    sync = 'S',
    terminate = 'X',
    password = 'p',
    _,
};

/// Backend message types (server → client)
pub const BackendMessageType = enum(u8) {
    authentication = 'R',
    backend_key_data = 'K',
    ready_for_query = 'Z',
    row_description = 'T',
    data_row = 'D',
    command_complete = 'C',
    error_response = 'E',
    notice_response = 'N',
    parse_complete = '1',
    bind_complete = '2',
    close_complete = '3',
    portal_suspended = 's',
    no_data = 'n',
    parameter_status = 'S',
    parameter_description = 't',
    _,
};

/// Transaction status indicator (for ReadyForQuery)
pub const TransactionStatus = enum(u8) {
    idle = 'I', // not in a transaction block
    in_transaction = 'T', // in a transaction block
    failed_transaction = 'E', // in a failed transaction block (queries ignored until rollback)
};

/// Authentication request types
pub const AuthenticationType = enum(i32) {
    ok = 0,
    cleartext_password = 3,
    md5_password = 5,
    scram_sha_256 = 10,
    _,
};

// ── Message Structures ──────────────────────────────────────────────────

/// Query message (simple query protocol)
pub const Query = struct {
    query: []const u8,

    pub fn parse(payload: []const u8) !Query {
        // Payload is null-terminated query string
        if (payload.len == 0) return error.InvalidMessage;
        if (payload[payload.len - 1] != 0) return error.InvalidMessage;
        return Query{ .query = payload[0 .. payload.len - 1] };
    }

    pub fn write(self: Query, writer: anytype) !void {
        try writer.writeByte(@intFromEnum(FrontendMessageType.query));
        const len: i32 = @intCast(4 + self.query.len + 1); // length + query + null
        try writer.writeInt(i32, len, .big);
        try writer.writeAll(self.query);
        try writer.writeByte(0); // null terminator
    }
};

/// Parse message (extended query protocol)
pub const Parse = struct {
    statement_name: []const u8,
    query: []const u8,
    param_types: []const i32, // OID array

    pub fn parse(payload: []const u8, allocator: Allocator) !Parse {
        var offset: usize = 0;

        // Statement name (null-terminated)
        const stmt_name_end = std.mem.indexOfScalarPos(u8, payload, offset, 0) orelse return error.InvalidMessage;
        const stmt_name = payload[offset..stmt_name_end];
        offset = stmt_name_end + 1;

        // Query string (null-terminated)
        const query_end = std.mem.indexOfScalarPos(u8, payload, offset, 0) orelse return error.InvalidMessage;
        const query = payload[offset..query_end];
        offset = query_end + 1;

        // Parameter type OIDs
        if (offset + 2 > payload.len) return error.InvalidMessage;
        const param_count = std.mem.readInt(i16, payload[offset..][0..2], .big);
        offset += 2;

        // Validate param_count is non-negative
        if (param_count < 0) return error.InvalidMessage;

        const param_types = try allocator.alloc(i32, @intCast(param_count));
        errdefer allocator.free(param_types);

        for (param_types) |*oid| {
            if (offset + 4 > payload.len) return error.InvalidMessage;
            oid.* = std.mem.readInt(i32, payload[offset..][0..4], .big);
            offset += 4;
        }

        return Parse{
            .statement_name = stmt_name,
            .query = query,
            .param_types = param_types,
        };
    }

    pub fn deinit(self: Parse, allocator: Allocator) void {
        allocator.free(self.param_types);
    }
};

/// Bind message (extended query protocol)
pub const Bind = struct {
    portal_name: []const u8,
    statement_name: []const u8,
    param_formats: []const i16, // 0 = text, 1 = binary
    param_values: []const []const u8,
    result_formats: []const i16,

    pub fn parse(payload: []const u8, allocator: Allocator) !Bind {
        var offset: usize = 0;

        // Portal name (null-terminated)
        const portal_name_end = std.mem.indexOfScalarPos(u8, payload, offset, 0) orelse return error.InvalidMessage;
        const portal_name = payload[offset..portal_name_end];
        offset = portal_name_end + 1;

        // Statement name (null-terminated)
        const stmt_name_end = std.mem.indexOfScalarPos(u8, payload, offset, 0) orelse return error.InvalidMessage;
        const stmt_name = payload[offset..stmt_name_end];
        offset = stmt_name_end + 1;

        // Parameter formats
        if (offset + 2 > payload.len) return error.InvalidMessage;
        const format_count = std.mem.readInt(i16, payload[offset..][0..2], .big);
        offset += 2;

        // Validate format_count is non-negative
        if (format_count < 0) return error.InvalidMessage;

        const param_formats = try allocator.alloc(i16, @intCast(format_count));
        errdefer allocator.free(param_formats);

        for (param_formats) |*fmt| {
            if (offset + 2 > payload.len) return error.InvalidMessage;
            fmt.* = std.mem.readInt(i16, payload[offset..][0..2], .big);
            offset += 2;
        }

        // Parameter values
        if (offset + 2 > payload.len) return error.InvalidMessage;
        const param_count = std.mem.readInt(i16, payload[offset..][0..2], .big);
        offset += 2;

        // Validate param_count is non-negative
        if (param_count < 0) return error.InvalidMessage;

        const param_values = try allocator.alloc([]const u8, @intCast(param_count));
        errdefer allocator.free(param_values);

        for (param_values) |*val| {
            if (offset + 4 > payload.len) return error.InvalidMessage;
            const val_len = std.mem.readInt(i32, payload[offset..][0..4], .big);
            offset += 4;

            if (val_len == -1) {
                // NULL value
                val.* = &[_]u8{};
            } else if (val_len < 0) {
                // Invalid negative length (other than -1 which means NULL)
                return error.InvalidMessage;
            } else {
                const len: usize = @intCast(val_len);
                if (offset + len > payload.len) return error.InvalidMessage;
                val.* = payload[offset .. offset + len];
                offset += len;
            }
        }

        // Result column formats
        if (offset + 2 > payload.len) return error.InvalidMessage;
        const result_format_count = std.mem.readInt(i16, payload[offset..][0..2], .big);
        offset += 2;

        // Validate result_format_count is non-negative
        if (result_format_count < 0) return error.InvalidMessage;

        const result_formats = try allocator.alloc(i16, @intCast(result_format_count));
        errdefer allocator.free(result_formats);

        for (result_formats) |*fmt| {
            if (offset + 2 > payload.len) return error.InvalidMessage;
            fmt.* = std.mem.readInt(i16, payload[offset..][0..2], .big);
            offset += 2;
        }

        return Bind{
            .portal_name = portal_name,
            .statement_name = stmt_name,
            .param_formats = param_formats,
            .param_values = param_values,
            .result_formats = result_formats,
        };
    }

    pub fn deinit(self: Bind, allocator: Allocator) void {
        allocator.free(self.param_formats);
        allocator.free(self.param_values);
        allocator.free(self.result_formats);
    }
};

/// Execute message (execute portal)
pub const Execute = struct {
    portal_name: []const u8,
    max_rows: i32,

    pub fn parse(payload: []const u8) !Execute {
        var stream = std.io.fixedBufferStream(payload);
        const reader = stream.reader();

        // Portal name (null-terminated string)
        const name_end = std.mem.indexOfScalar(u8, payload, 0) orelse return error.InvalidMessage;
        const portal_name = payload[0..name_end];

        // Seek past the portal name and null terminator
        try reader.skipBytes(name_end + 1, .{});

        // Maximum rows (0 = unlimited)
        const max_rows = try reader.readInt(i32, .big);

        return Execute{
            .portal_name = portal_name,
            .max_rows = max_rows,
        };
    }
};

/// Close message (close statement or portal)
pub const Close = struct {
    close_type: u8, // 'S' for statement, 'P' for portal
    name: []const u8,

    pub fn parse(payload: []const u8) !Close {
        if (payload.len < 2) return error.InvalidMessage;

        // Close type ('S' or 'P')
        const close_type = payload[0];

        // Name (null-terminated string)
        const name_start = 1;
        const name_end = std.mem.indexOfScalar(u8, payload[name_start..], 0) orelse return error.InvalidMessage;
        const name = payload[name_start .. name_start + name_end];

        return Close{
            .close_type = close_type,
            .name = name,
        };
    }
};

/// RowDescription message (column metadata)
pub const RowDescription = struct {
    pub const Field = struct {
        name: []const u8,
        table_oid: i32,
        column_attr_number: i16,
        type_oid: i32,
        type_size: i16,
        type_modifier: i32,
        format_code: i16, // 0 = text, 1 = binary
    };

    fields: []const Field,

    pub fn write(self: RowDescription, writer: anytype, allocator: Allocator) !void {
        try writer.writeByte(@intFromEnum(BackendMessageType.row_description));

        // Calculate total length
        var total_len: i32 = 4 + 2; // length field + field count
        for (self.fields) |field| {
            total_len += @intCast(field.name.len + 1 + 4 + 2 + 4 + 2 + 4 + 2);
        }
        try writer.writeInt(i32, total_len, .big);

        // Field count
        try writer.writeInt(i16, @intCast(self.fields.len), .big);

        // Fields
        for (self.fields) |field| {
            try writer.writeAll(field.name);
            try writer.writeByte(0); // null terminator
            try writer.writeInt(i32, field.table_oid, .big);
            try writer.writeInt(i16, field.column_attr_number, .big);
            try writer.writeInt(i32, field.type_oid, .big);
            try writer.writeInt(i16, field.type_size, .big);
            try writer.writeInt(i32, field.type_modifier, .big);
            try writer.writeInt(i16, field.format_code, .big);
        }

        _ = allocator;
    }

    pub fn deinit(self: RowDescription, allocator: Allocator) void {
        allocator.free(self.fields);
    }
};

/// DataRow message (single row of query results)
pub const DataRow = struct {
    columns: []const []const u8,

    pub fn write(self: DataRow, writer: anytype) !void {
        try writer.writeByte(@intFromEnum(BackendMessageType.data_row));

        // Calculate total length
        var total_len: i32 = 4 + 2; // length field + column count
        for (self.columns) |col| {
            total_len += 4 + @as(i32, @intCast(col.len));
        }
        try writer.writeInt(i32, total_len, .big);

        // Column count
        try writer.writeInt(i16, @intCast(self.columns.len), .big);

        // Column values
        for (self.columns) |col| {
            try writer.writeInt(i32, @intCast(col.len), .big);
            try writer.writeAll(col);
        }
    }
};

/// CommandComplete message (query finished)
pub const CommandComplete = struct {
    tag: []const u8, // e.g., "SELECT 5", "INSERT 0 3", "UPDATE 2"

    pub fn write(self: CommandComplete, writer: anytype) !void {
        try writer.writeByte(@intFromEnum(BackendMessageType.command_complete));
        const len: i32 = @intCast(4 + self.tag.len + 1);
        try writer.writeInt(i32, len, .big);
        try writer.writeAll(self.tag);
        try writer.writeByte(0); // null terminator
    }
};

/// ReadyForQuery message (ready for next command)
pub const ReadyForQuery = struct {
    status: TransactionStatus,

    pub fn write(self: ReadyForQuery, writer: anytype) !void {
        try writer.writeByte(@intFromEnum(BackendMessageType.ready_for_query));
        try writer.writeInt(i32, 5, .big); // length = 4 + 1
        try writer.writeByte(@intFromEnum(self.status));
    }
};

/// ErrorResponse message
pub const ErrorResponse = struct {
    pub const Field = struct {
        code: u8, // 'S' = severity, 'C' = SQLSTATE, 'M' = message, etc.
        value: []const u8,
    };

    fields: []const Field,

    pub fn write(self: ErrorResponse, writer: anytype) !void {
        try writer.writeByte(@intFromEnum(BackendMessageType.error_response));

        // Calculate total length
        var total_len: i32 = 4 + 1; // length field + terminator
        for (self.fields) |field| {
            total_len += 1 + @as(i32, @intCast(field.value.len)) + 1;
        }
        try writer.writeInt(i32, total_len, .big);

        // Fields
        for (self.fields) |field| {
            try writer.writeByte(field.code);
            try writer.writeAll(field.value);
            try writer.writeByte(0); // null terminator
        }

        try writer.writeByte(0); // final terminator
    }

    pub fn deinit(self: ErrorResponse, allocator: Allocator) void {
        allocator.free(self.fields);
    }
};

/// Authentication message
pub const Authentication = struct {
    auth_type: AuthenticationType,
    salt: ?[4]u8, // for MD5

    pub fn write(self: Authentication, writer: anytype) !void {
        try writer.writeByte(@intFromEnum(BackendMessageType.authentication));

        if (self.salt) |salt| {
            try writer.writeInt(i32, 12, .big); // length = 4 + 4 + 4
            try writer.writeInt(i32, @intFromEnum(self.auth_type), .big);
            try writer.writeAll(&salt);
        } else {
            try writer.writeInt(i32, 8, .big); // length = 4 + 4
            try writer.writeInt(i32, @intFromEnum(self.auth_type), .big);
        }
    }
};

// ── Message Reader ──────────────────────────────────────────────────────

/// Read a message from a reader
pub fn readMessage(reader: anytype, allocator: Allocator) !struct {
    msg_type: u8,
    payload: []u8,
} {
    // Read message type
    const msg_type = try reader.readByte();

    // Read message length (includes itself, so subtract 4)
    const length = try reader.readInt(i32, .big);
    if (length < 4) return error.InvalidMessage;
    const payload_len: usize = @intCast(length - 4);

    // Read payload
    const payload = try allocator.alloc(u8, payload_len);
    errdefer allocator.free(payload);

    const n = try reader.readAll(payload);
    if (n != payload_len) return error.UnexpectedEof;

    return .{
        .msg_type = msg_type,
        .payload = payload,
    };
}

// ── Tests ──────────────────────────────────────────────────────────────

test "Query message serialization" {
    const allocator = std.testing.allocator;
    const query = Query{ .query = "SELECT 1" };

    var buf = std.ArrayListUnmanaged(u8){};
    defer buf.deinit(allocator);

    try query.write(buf.writer(allocator));

    // Expected: 'Q' + length(4) + "SELECT 1\0"
    try std.testing.expectEqual(@as(u8, 'Q'), buf.items[0]);
    const len = std.mem.readInt(i32, buf.items[1..5], .big);
    try std.testing.expectEqual(@as(i32, 13), len); // 4 + 8 + 1
    try std.testing.expectEqualStrings("SELECT 1", buf.items[5 .. buf.items.len - 1]);
    try std.testing.expectEqual(@as(u8, 0), buf.items[buf.items.len - 1]);
}

test "Query message parsing" {
    const payload = "SELECT 1\x00";
    const query = try Query.parse(payload);
    try std.testing.expectEqualStrings("SELECT 1", query.query);
}

test "RowDescription serialization" {
    const allocator = std.testing.allocator;

    const fields = try allocator.alloc(RowDescription.Field, 2);
    defer allocator.free(fields);

    fields[0] = .{
        .name = "id",
        .table_oid = 16384,
        .column_attr_number = 1,
        .type_oid = 23, // INT4
        .type_size = 4,
        .type_modifier = -1,
        .format_code = 0,
    };
    fields[1] = .{
        .name = "name",
        .table_oid = 16384,
        .column_attr_number = 2,
        .type_oid = 25, // TEXT
        .type_size = -1,
        .type_modifier = -1,
        .format_code = 0,
    };

    const row_desc = RowDescription{ .fields = fields };

    var buf = std.ArrayListUnmanaged(u8){};
    defer buf.deinit(allocator);

    try row_desc.write(buf.writer(allocator), allocator);

    // Verify message type
    try std.testing.expectEqual(@as(u8, 'T'), buf.items[0]);
    // Verify field count
    const field_count = std.mem.readInt(i16, buf.items[5..7], .big);
    try std.testing.expectEqual(@as(i16, 2), field_count);
}

test "DataRow serialization" {
    const allocator = std.testing.allocator;

    const columns = &[_][]const u8{ "1", "Alice" };
    const data_row = DataRow{ .columns = columns };

    var buf = std.ArrayListUnmanaged(u8){};
    defer buf.deinit(allocator);

    try data_row.write(buf.writer(allocator));

    // Verify message type
    try std.testing.expectEqual(@as(u8, 'D'), buf.items[0]);
    // Verify column count
    const col_count = std.mem.readInt(i16, buf.items[5..7], .big);
    try std.testing.expectEqual(@as(i16, 2), col_count);
}

test "CommandComplete serialization" {
    const allocator = std.testing.allocator;
    const cmd_complete = CommandComplete{ .tag = "SELECT 5" };

    var buf = std.ArrayListUnmanaged(u8){};
    defer buf.deinit(allocator);

    try cmd_complete.write(buf.writer(allocator));

    // Verify message type
    try std.testing.expectEqual(@as(u8, 'C'), buf.items[0]);
    // Verify tag
    const tag_start = 5;
    const tag_end = buf.items.len - 1;
    try std.testing.expectEqualStrings("SELECT 5", buf.items[tag_start..tag_end]);
}

test "ReadyForQuery serialization" {
    const allocator = std.testing.allocator;
    const ready = ReadyForQuery{ .status = .idle };

    var buf = std.ArrayListUnmanaged(u8){};
    defer buf.deinit(allocator);

    try ready.write(buf.writer(allocator));

    try std.testing.expectEqual(@as(u8, 'Z'), buf.items[0]);
    try std.testing.expectEqual(@as(u8, 'I'), buf.items[5]);
}

test "ErrorResponse serialization" {
    const allocator = std.testing.allocator;

    const fields = try allocator.alloc(ErrorResponse.Field, 3);
    defer allocator.free(fields);

    fields[0] = .{ .code = 'S', .value = "ERROR" };
    fields[1] = .{ .code = 'C', .value = "42P01" };
    fields[2] = .{ .code = 'M', .value = "table does not exist" };

    const err_resp = ErrorResponse{ .fields = fields };

    var buf = std.ArrayListUnmanaged(u8){};
    defer buf.deinit(allocator);

    try err_resp.write(buf.writer(allocator));

    try std.testing.expectEqual(@as(u8, 'E'), buf.items[0]);
    // Verify fields are present
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "ERROR") != null);
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "42P01") != null);
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "table does not exist") != null);
}

test "Parse message parsing" {
    const allocator = std.testing.allocator;

    // Construct payload: stmt_name + query + param_count + oids
    var payload = std.ArrayListUnmanaged(u8){};
    defer payload.deinit(allocator);

    try payload.appendSlice(allocator, "stmt1\x00");
    try payload.appendSlice(allocator, "SELECT $1\x00");
    try payload.writer(allocator).writeInt(i16, 1, .big); // 1 param
    try payload.writer(allocator).writeInt(i32, 23, .big); // INT4 OID

    const parse_msg = try Parse.parse(payload.items, allocator);
    defer parse_msg.deinit(allocator);

    try std.testing.expectEqualStrings("stmt1", parse_msg.statement_name);
    try std.testing.expectEqualStrings("SELECT $1", parse_msg.query);
    try std.testing.expectEqual(@as(usize, 1), parse_msg.param_types.len);
    try std.testing.expectEqual(@as(i32, 23), parse_msg.param_types[0]);
}

test "Bind message parsing" {
    const allocator = std.testing.allocator;

    var payload = std.ArrayListUnmanaged(u8){};
    defer payload.deinit(allocator);

    try payload.appendSlice(allocator, "portal1\x00");
    try payload.appendSlice(allocator, "stmt1\x00");
    try payload.writer(allocator).writeInt(i16, 1, .big); // 1 format
    try payload.writer(allocator).writeInt(i16, 0, .big); // text format
    try payload.writer(allocator).writeInt(i16, 1, .big); // 1 param value
    try payload.writer(allocator).writeInt(i32, 2, .big); // length
    try payload.appendSlice(allocator, "42");
    try payload.writer(allocator).writeInt(i16, 1, .big); // 1 result format
    try payload.writer(allocator).writeInt(i16, 0, .big); // text format

    const bind_msg = try Bind.parse(payload.items, allocator);
    defer bind_msg.deinit(allocator);

    try std.testing.expectEqualStrings("portal1", bind_msg.portal_name);
    try std.testing.expectEqualStrings("stmt1", bind_msg.statement_name);
    try std.testing.expectEqual(@as(usize, 1), bind_msg.param_values.len);
    try std.testing.expectEqualStrings("42", bind_msg.param_values[0]);
}

test "readMessage helper" {
    const allocator = std.testing.allocator;

    // Construct a Query message: 'Q' + length + "SELECT 1\0"
    var buf = std.ArrayListUnmanaged(u8){};
    defer buf.deinit(allocator);

    try buf.append(allocator, 'Q');
    try buf.writer(allocator).writeInt(i32, 13, .big); // 4 + 9
    try buf.appendSlice(allocator, "SELECT 1\x00");

    var stream = std.io.fixedBufferStream(buf.items);
    const msg = try readMessage(stream.reader(), allocator);
    defer allocator.free(msg.payload);

    try std.testing.expectEqual(@as(u8, 'Q'), msg.msg_type);
    try std.testing.expectEqualStrings("SELECT 1\x00", msg.payload);
}

// ── Edge Case & Error Path Tests ───────────────────────────────────────

test "Query parse - empty payload" {
    const result = Query.parse(&[_]u8{});
    try std.testing.expectError(error.InvalidMessage, result);
}

test "Query parse - missing null terminator" {
    const result = Query.parse("SELECT 1");
    try std.testing.expectError(error.InvalidMessage, result);
}

test "Parse message - truncated statement name" {
    const allocator = std.testing.allocator;
    const payload = "stmt1"; // missing null terminator
    const result = Parse.parse(payload, allocator);
    try std.testing.expectError(error.InvalidMessage, result);
}

test "Parse message - truncated query" {
    const allocator = std.testing.allocator;
    const payload = "stmt1\x00SELECT"; // query missing null terminator
    const result = Parse.parse(payload, allocator);
    try std.testing.expectError(error.InvalidMessage, result);
}

test "Parse message - truncated param count" {
    const allocator = std.testing.allocator;
    const payload = "stmt1\x00SELECT 1\x00"; // missing param count
    const result = Parse.parse(payload, allocator);
    try std.testing.expectError(error.InvalidMessage, result);
}

test "Parse message - truncated param types" {
    const allocator = std.testing.allocator;
    var payload = std.ArrayListUnmanaged(u8){};
    defer payload.deinit(allocator);

    try payload.appendSlice(allocator, "stmt1\x00");
    try payload.appendSlice(allocator, "SELECT $1\x00");
    try payload.writer(allocator).writeInt(i16, 2, .big); // claims 2 params
    try payload.writer(allocator).writeInt(i32, 23, .big); // only 1 param OID

    const result = Parse.parse(payload.items, allocator);
    try std.testing.expectError(error.InvalidMessage, result);
}

test "Bind message - truncated portal name" {
    const allocator = std.testing.allocator;
    const payload = "portal1"; // missing null terminator
    const result = Bind.parse(payload, allocator);
    try std.testing.expectError(error.InvalidMessage, result);
}

test "Bind message - truncated statement name" {
    const allocator = std.testing.allocator;
    const payload = "portal1\x00stmt1"; // stmt missing null terminator
    const result = Bind.parse(payload, allocator);
    try std.testing.expectError(error.InvalidMessage, result);
}

test "Bind message - truncated format count" {
    const allocator = std.testing.allocator;
    const payload = "portal1\x00stmt1\x00"; // missing format count
    const result = Bind.parse(payload, allocator);
    try std.testing.expectError(error.InvalidMessage, result);
}

test "Bind message - truncated param formats" {
    const allocator = std.testing.allocator;
    var payload = std.ArrayListUnmanaged(u8){};
    defer payload.deinit(allocator);

    try payload.appendSlice(allocator, "portal1\x00");
    try payload.appendSlice(allocator, "stmt1\x00");
    try payload.writer(allocator).writeInt(i16, 2, .big); // claims 2 formats
    try payload.writer(allocator).writeInt(i16, 0, .big); // only 1 format

    const result = Bind.parse(payload.items, allocator);
    try std.testing.expectError(error.InvalidMessage, result);
}

test "Bind message - truncated param value count" {
    const allocator = std.testing.allocator;
    var payload = std.ArrayListUnmanaged(u8){};
    defer payload.deinit(allocator);

    try payload.appendSlice(allocator, "portal1\x00");
    try payload.appendSlice(allocator, "stmt1\x00");
    try payload.writer(allocator).writeInt(i16, 1, .big); // 1 format
    try payload.writer(allocator).writeInt(i16, 0, .big); // text format
    // missing param value count

    const result = Bind.parse(payload.items, allocator);
    try std.testing.expectError(error.InvalidMessage, result);
}

test "Bind message - truncated param value length" {
    const allocator = std.testing.allocator;
    var payload = std.ArrayListUnmanaged(u8){};
    defer payload.deinit(allocator);

    try payload.appendSlice(allocator, "portal1\x00");
    try payload.appendSlice(allocator, "stmt1\x00");
    try payload.writer(allocator).writeInt(i16, 1, .big); // 1 format
    try payload.writer(allocator).writeInt(i16, 0, .big); // text format
    try payload.writer(allocator).writeInt(i16, 1, .big); // 1 param value
    // missing param value length

    const result = Bind.parse(payload.items, allocator);
    try std.testing.expectError(error.InvalidMessage, result);
}

test "Bind message - truncated param value data" {
    const allocator = std.testing.allocator;
    var payload = std.ArrayListUnmanaged(u8){};
    defer payload.deinit(allocator);

    try payload.appendSlice(allocator, "portal1\x00");
    try payload.appendSlice(allocator, "stmt1\x00");
    try payload.writer(allocator).writeInt(i16, 1, .big); // 1 format
    try payload.writer(allocator).writeInt(i16, 0, .big); // text format
    try payload.writer(allocator).writeInt(i16, 1, .big); // 1 param value
    try payload.writer(allocator).writeInt(i32, 10, .big); // length 10
    try payload.appendSlice(allocator, "short"); // only 5 bytes

    const result = Bind.parse(payload.items, allocator);
    try std.testing.expectError(error.InvalidMessage, result);
}

test "Bind message - NULL param value" {
    const allocator = std.testing.allocator;
    var payload = std.ArrayListUnmanaged(u8){};
    defer payload.deinit(allocator);

    try payload.appendSlice(allocator, "portal1\x00");
    try payload.appendSlice(allocator, "stmt1\x00");
    try payload.writer(allocator).writeInt(i16, 1, .big); // 1 format
    try payload.writer(allocator).writeInt(i16, 0, .big); // text format
    try payload.writer(allocator).writeInt(i16, 1, .big); // 1 param value
    try payload.writer(allocator).writeInt(i32, -1, .big); // NULL value
    try payload.writer(allocator).writeInt(i16, 1, .big); // 1 result format
    try payload.writer(allocator).writeInt(i16, 0, .big); // text format

    const bind_msg = try Bind.parse(payload.items, allocator);
    defer bind_msg.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 1), bind_msg.param_values.len);
    try std.testing.expectEqual(@as(usize, 0), bind_msg.param_values[0].len); // NULL represented as empty slice
}

test "Bind message - truncated result format count" {
    const allocator = std.testing.allocator;
    var payload = std.ArrayListUnmanaged(u8){};
    defer payload.deinit(allocator);

    try payload.appendSlice(allocator, "portal1\x00");
    try payload.appendSlice(allocator, "stmt1\x00");
    try payload.writer(allocator).writeInt(i16, 1, .big); // 1 format
    try payload.writer(allocator).writeInt(i16, 0, .big); // text format
    try payload.writer(allocator).writeInt(i16, 1, .big); // 1 param value
    try payload.writer(allocator).writeInt(i32, 2, .big); // length
    try payload.appendSlice(allocator, "42");
    // missing result format count

    const result = Bind.parse(payload.items, allocator);
    try std.testing.expectError(error.InvalidMessage, result);
}

test "Bind message - truncated result formats" {
    const allocator = std.testing.allocator;
    var payload = std.ArrayListUnmanaged(u8){};
    defer payload.deinit(allocator);

    try payload.appendSlice(allocator, "portal1\x00");
    try payload.appendSlice(allocator, "stmt1\x00");
    try payload.writer(allocator).writeInt(i16, 1, .big); // 1 format
    try payload.writer(allocator).writeInt(i16, 0, .big); // text format
    try payload.writer(allocator).writeInt(i16, 1, .big); // 1 param value
    try payload.writer(allocator).writeInt(i32, 2, .big); // length
    try payload.appendSlice(allocator, "42");
    try payload.writer(allocator).writeInt(i16, 2, .big); // claims 2 result formats
    try payload.writer(allocator).writeInt(i16, 0, .big); // only 1 format

    const result = Bind.parse(payload.items, allocator);
    try std.testing.expectError(error.InvalidMessage, result);
}

test "readMessage - invalid length (too small)" {
    const allocator = std.testing.allocator;

    var buf = std.ArrayListUnmanaged(u8){};
    defer buf.deinit(allocator);

    try buf.append(allocator, 'Q');
    try buf.writer(allocator).writeInt(i32, 3, .big); // invalid: less than 4

    var stream = std.io.fixedBufferStream(buf.items);
    const result = readMessage(stream.reader(), allocator);
    try std.testing.expectError(error.InvalidMessage, result);
}

test "readMessage - unexpected EOF" {
    const allocator = std.testing.allocator;

    var buf = std.ArrayListUnmanaged(u8){};
    defer buf.deinit(allocator);

    try buf.append(allocator, 'Q');
    try buf.writer(allocator).writeInt(i32, 100, .big); // claims 100 bytes
    try buf.appendSlice(allocator, "short"); // only 5 bytes

    var stream = std.io.fixedBufferStream(buf.items);
    const result = readMessage(stream.reader(), allocator);
    try std.testing.expectError(error.UnexpectedEof, result);
}

test "Authentication message - with salt" {
    const allocator = std.testing.allocator;
    const auth = Authentication{
        .auth_type = .md5_password,
        .salt = [4]u8{ 0x12, 0x34, 0x56, 0x78 },
    };

    var buf = std.ArrayListUnmanaged(u8){};
    defer buf.deinit(allocator);

    try auth.write(buf.writer(allocator));

    try std.testing.expectEqual(@as(u8, 'R'), buf.items[0]);
    const len = std.mem.readInt(i32, buf.items[1..5], .big);
    try std.testing.expectEqual(@as(i32, 12), len);
    const auth_type = std.mem.readInt(i32, buf.items[5..9], .big);
    try std.testing.expectEqual(@as(i32, 5), auth_type); // MD5
    try std.testing.expectEqual([4]u8{ 0x12, 0x34, 0x56, 0x78 }, buf.items[9..13].*);
}

test "Authentication message - without salt" {
    const allocator = std.testing.allocator;
    const auth = Authentication{
        .auth_type = .ok,
        .salt = null,
    };

    var buf = std.ArrayListUnmanaged(u8){};
    defer buf.deinit(allocator);

    try auth.write(buf.writer(allocator));

    try std.testing.expectEqual(@as(u8, 'R'), buf.items[0]);
    const len = std.mem.readInt(i32, buf.items[1..5], .big);
    try std.testing.expectEqual(@as(i32, 8), len);
    const auth_type = std.mem.readInt(i32, buf.items[5..9], .big);
    try std.testing.expectEqual(@as(i32, 0), auth_type); // OK
}

test "DataRow - empty columns" {
    const allocator = std.testing.allocator;
    const data_row = DataRow{ .columns = &[_][]const u8{} };

    var buf = std.ArrayListUnmanaged(u8){};
    defer buf.deinit(allocator);

    try data_row.write(buf.writer(allocator));

    try std.testing.expectEqual(@as(u8, 'D'), buf.items[0]);
    const col_count = std.mem.readInt(i16, buf.items[5..7], .big);
    try std.testing.expectEqual(@as(i16, 0), col_count);
}

test "RowDescription - zero fields" {
    const allocator = std.testing.allocator;
    const row_desc = RowDescription{ .fields = &[_]RowDescription.Field{} };

    var buf = std.ArrayListUnmanaged(u8){};
    defer buf.deinit(allocator);

    try row_desc.write(buf.writer(allocator), allocator);

    try std.testing.expectEqual(@as(u8, 'T'), buf.items[0]);
    const field_count = std.mem.readInt(i16, buf.items[5..7], .big);
    try std.testing.expectEqual(@as(i16, 0), field_count);
}

test "ErrorResponse - empty fields" {
    const allocator = std.testing.allocator;
    const err_resp = ErrorResponse{ .fields = &[_]ErrorResponse.Field{} };

    var buf = std.ArrayListUnmanaged(u8){};
    defer buf.deinit(allocator);

    try err_resp.write(buf.writer(allocator));

    try std.testing.expectEqual(@as(u8, 'E'), buf.items[0]);
    // Should have final terminator
    try std.testing.expectEqual(@as(u8, 0), buf.items[buf.items.len - 1]);
}

test "Query write - empty query string" {
    const allocator = std.testing.allocator;
    const query = Query{ .query = "" };

    var buf = std.ArrayListUnmanaged(u8){};
    defer buf.deinit(allocator);

    try query.write(buf.writer(allocator));

    try std.testing.expectEqual(@as(u8, 'Q'), buf.items[0]);
    const len = std.mem.readInt(i32, buf.items[1..5], .big);
    try std.testing.expectEqual(@as(i32, 5), len); // 4 + 1 (null terminator only)
}

test "CommandComplete - empty tag" {
    const allocator = std.testing.allocator;
    const cmd_complete = CommandComplete{ .tag = "" };

    var buf = std.ArrayListUnmanaged(u8){};
    defer buf.deinit(allocator);

    try cmd_complete.write(buf.writer(allocator));

    try std.testing.expectEqual(@as(u8, 'C'), buf.items[0]);
    const len = std.mem.readInt(i32, buf.items[1..5], .big);
    try std.testing.expectEqual(@as(i32, 5), len); // 4 + 1 (null terminator only)
}
