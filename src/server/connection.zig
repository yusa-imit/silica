//! Connection Handler for PostgreSQL Wire Protocol
//!
//! Implements the simple query protocol:
//!   Client sends Query message
//!   Server responds with:
//!     RowDescription (if query returns rows)
//!     DataRow* (zero or more data rows)
//!     CommandComplete
//!     ReadyForQuery
//!
//! Reference: https://www.postgresql.org/docs/current/protocol-flow.html#PROTOCOL-FLOW-SIMPLE-QUERY

const std = @import("std");
const Allocator = std.mem.Allocator;
const wire = @import("wire.zig");
const engine = @import("../sql/engine.zig");
const Database = engine.Database;
const QueryResult = engine.QueryResult;
const Value = @import("../sql/executor.zig").Value;

/// Connection state
pub const Connection = struct {
    allocator: Allocator,
    db: *Database,
    transaction_status: wire.TransactionStatus,

    pub fn init(allocator: Allocator, db: *Database) Connection {
        return .{
            .allocator = allocator,
            .db = db,
            .transaction_status = .idle,
        };
    }

    pub fn deinit(self: *Connection) void {
        _ = self;
    }

    /// Handle simple query protocol
    /// Reads a Query message, executes it, and writes the response
    pub fn handleSimpleQuery(
        self: *Connection,
        query_msg: wire.Query,
        writer: anytype,
    ) !void {
        // Execute query
        var result = self.db.execSQL(query_msg.query) catch |err| {
            // Send error response
            try self.sendError(writer, err);
            return;
        };
        defer result.close();

        // Determine if query returns rows
        if (result.columns) |columns| {
            // Send RowDescription
            try self.sendRowDescription(writer, columns);

            // Send DataRow for each row
            while (try result.next()) |row| {
                try self.sendDataRow(writer, row, columns);
            }
        }

        // Send CommandComplete
        const tag = try self.getCommandTag(&result);
        const cmd_complete = wire.CommandComplete{ .tag = tag };
        try cmd_complete.write(writer);

        // Send ReadyForQuery
        const ready = wire.ReadyForQuery{ .status = self.transaction_status };
        try ready.write(writer);
    }

    /// Send RowDescription message
    fn sendRowDescription(
        self: *Connection,
        writer: anytype,
        columns: []const []const u8,
    ) !void {
        const fields = try self.allocator.alloc(wire.RowDescription.Field, columns.len);
        defer self.allocator.free(fields);

        for (columns, 0..) |col_name, i| {
            fields[i] = .{
                .name = col_name,
                .table_oid = 0, // Not implemented yet
                .column_attr_number = @intCast(i + 1),
                .type_oid = 25, // TEXT OID (default for now)
                .type_size = -1, // variable length
                .type_modifier = -1,
                .format_code = 0, // text format
            };
        }

        const row_desc = wire.RowDescription{ .fields = fields };
        try row_desc.write(writer, self.allocator);
    }

    /// Send DataRow message
    fn sendDataRow(
        self: *Connection,
        writer: anytype,
        row: engine.Row,
        columns: []const []const u8,
    ) !void {
        const col_values = try self.allocator.alloc([]const u8, columns.len);
        defer {
            for (col_values) |val| {
                self.allocator.free(val);
            }
            self.allocator.free(col_values);
        }

        // Convert each column value to text
        for (columns, 0..) |col_name, i| {
            const value = row.get(col_name) orelse Value{ .null = {} };
            col_values[i] = try self.valueToText(value);
        }

        const data_row = wire.DataRow{ .columns = col_values };
        try data_row.write(writer);
    }

    /// Send ErrorResponse message
    fn sendError(self: *Connection, writer: anytype, err: anyerror) !void {
        const fields = try self.allocator.alloc(wire.ErrorResponse.Field, 3);
        defer self.allocator.free(fields);

        fields[0] = .{ .code = 'S', .value = "ERROR" };
        fields[1] = .{ .code = 'C', .value = try self.getSQLState(err) };
        fields[2] = .{ .code = 'M', .value = try self.getErrorMessage(err) };

        const err_resp = wire.ErrorResponse{ .fields = fields };
        try err_resp.write(writer);

        // Send ReadyForQuery after error
        const ready = wire.ReadyForQuery{ .status = self.transaction_status };
        try ready.write(writer);
    }

    /// Convert Value to text representation
    fn valueToText(self: *Connection, value: Value) ![]const u8 {
        var buf = std.ArrayListUnmanaged(u8){};
        errdefer buf.deinit(self.allocator);

        const writer = buf.writer(self.allocator);

        switch (value) {
            .null => return try self.allocator.dupe(u8, "NULL"),
            .integer => |i| try std.fmt.format(writer, "{d}", .{i}),
            .real => |r| try std.fmt.format(writer, "{d}", .{r}),
            .text => |t| return try self.allocator.dupe(u8, t),
            .blob => |b| {
                // Return hex-encoded blob
                for (b) |byte| {
                    try std.fmt.format(writer, "{x:0>2}", .{byte});
                }
            },
            .boolean => |b| try std.fmt.format(writer, "{}", .{b}),
            .date => |d| try std.fmt.format(writer, "{d}", .{d}),
            .time => |t| try std.fmt.format(writer, "{d}", .{t}),
            .timestamp => |ts| try std.fmt.format(writer, "{d}", .{ts}),
            .interval => |iv| {
                // Format: "P<months>M<days>DT<seconds>S" or simplified
                try std.fmt.format(writer, "{d} months {d} days {d} micros", .{ iv.months, iv.days, iv.micros });
            },
            .numeric => |n| {
                // Format numeric value
                const scale = n.scale;
                const abs_val = if (n.value < 0) -n.value else n.value;
                const integer_part = @divTrunc(abs_val, std.math.pow(i128, 10, @intCast(scale)));
                const fractional_part = @mod(abs_val, std.math.pow(i128, 10, @intCast(scale)));

                if (n.value < 0) try writer.writeByte('-');
                try std.fmt.format(writer, "{d}", .{integer_part});
                if (scale > 0) {
                    try writer.writeByte('.');
                    try std.fmt.format(writer, "{d:0>[1]}", .{ fractional_part, scale });
                }
            },
            .uuid => |u| {
                // Format: "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
                try std.fmt.format(writer, "{x:0>2}{x:0>2}{x:0>2}{x:0>2}-{x:0>2}{x:0>2}-{x:0>2}{x:0>2}-{x:0>2}{x:0>2}-{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}", .{
                    u[0],  u[1],  u[2],  u[3],  u[4],  u[5],  u[6],  u[7],
                    u[8],  u[9],  u[10], u[11], u[12], u[13], u[14], u[15],
                });
            },
            .json, .jsonb => |j| return try self.allocator.dupe(u8, j),
            .tsvector, .tsquery => |t| return try self.allocator.dupe(u8, t),
        }

        return try buf.toOwnedSlice(self.allocator);
    }

    /// Get command tag for CommandComplete message
    fn getCommandTag(self: *Connection, result: *QueryResult) ![]const u8 {
        _ = self;
        if (result.rows_affected) |affected| {
            // For INSERT/UPDATE/DELETE, include row count
            // For now, just return a generic tag
            if (affected > 0) {
                return "SELECT"; // Placeholder
            }
        }
        return "OK";
    }

    /// Map error to SQLSTATE code
    fn getSQLState(self: *Connection, err: anyerror) ![]const u8 {
        _ = self;
        return switch (err) {
            error.ParseError => "42601", // syntax_error
            error.TableNotFound => "42P01", // undefined_table
            error.ColumnNotFound => "42703", // undefined_column
            error.DivisionByZero => "22012", // division_by_zero
            error.OutOfMemory => "53200", // out_of_memory
            else => "XX000", // internal_error
        };
    }

    /// Get human-readable error message
    fn getErrorMessage(self: *Connection, err: anyerror) ![]const u8 {
        _ = self;
        return switch (err) {
            error.ParseError => "syntax error",
            error.TableNotFound => "table does not exist",
            error.ColumnNotFound => "column does not exist",
            error.DivisionByZero => "division by zero",
            error.OutOfMemory => "out of memory",
            else => "internal error",
        };
    }
};

// ── Tests ──────────────────────────────────────────────────────────

test "Connection init/deinit" {
    const allocator = std.testing.allocator;

    // Create temporary database
    const db_path = "test_connection.db";
    defer std.fs.cwd().deleteFile(db_path) catch {};

    var db = try Database.init(allocator, db_path);
    defer db.deinit();

    var conn = Connection.init(allocator, &db);
    defer conn.deinit();

    try std.testing.expectEqual(wire.TransactionStatus.idle, conn.transaction_status);
}

test "valueToText - integer" {
    const allocator = std.testing.allocator;

    const db_path = "test_value_int.db";
    defer std.fs.cwd().deleteFile(db_path) catch {};

    var db = try Database.init(allocator, db_path);
    defer db.deinit();

    var conn = Connection.init(allocator, &db);
    defer conn.deinit();

    const value = Value{ .integer = 42 };
    const text = try conn.valueToText(value);
    defer allocator.free(text);

    try std.testing.expectEqualStrings("42", text);
}

test "valueToText - text" {
    const allocator = std.testing.allocator;

    const db_path = "test_value_text.db";
    defer std.fs.cwd().deleteFile(db_path) catch {};

    var db = try Database.init(allocator, db_path);
    defer db.deinit();

    var conn = Connection.init(allocator, &db);
    defer conn.deinit();

    const value = Value{ .text = "hello" };
    const text = try conn.valueToText(value);
    defer allocator.free(text);

    try std.testing.expectEqualStrings("hello", text);
}

test "valueToText - null" {
    const allocator = std.testing.allocator;

    const db_path = "test_value_null.db";
    defer std.fs.cwd().deleteFile(db_path) catch {};

    var db = try Database.init(allocator, db_path);
    defer db.deinit();

    var conn = Connection.init(allocator, &db);
    defer conn.deinit();

    const value = Value{ .null = {} };
    const text = try conn.valueToText(value);
    defer allocator.free(text);

    try std.testing.expectEqualStrings("NULL", text);
}

test "getSQLState - error mapping" {
    const allocator = std.testing.allocator;

    const db_path = "test_sqlstate.db";
    defer std.fs.cwd().deleteFile(db_path) catch {};

    var db = try Database.init(allocator, db_path);
    defer db.deinit();

    var conn = Connection.init(allocator, &db);
    defer conn.deinit();

    const sqlstate = try conn.getSQLState(error.TableNotFound);
    try std.testing.expectEqualStrings("42P01", sqlstate);
}

test "getCommandTag - OK" {
    const allocator = std.testing.allocator;

    const db_path = "test_cmdtag.db";
    defer std.fs.cwd().deleteFile(db_path) catch {};

    var db = try Database.init(allocator, db_path);
    defer db.deinit();

    var conn = Connection.init(allocator, &db);
    defer conn.deinit();

    var result = try db.execSQL("CREATE TABLE test (id INTEGER)");
    defer result.close();

    const tag = try conn.getCommandTag(&result);
    try std.testing.expectEqualStrings("OK", tag);
}
