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
const silica = @import("silica");
const Database = silica.engine.Database;
const QueryResult = silica.engine.QueryResult;
const Value = silica.executor.Value;

/// Prepared statement information
const PreparedStatement = struct {
    query: []const u8,
    param_types: []const i32,
    allocator: Allocator,

    pub fn deinit(self: *PreparedStatement) void {
        self.allocator.free(self.query);
        self.allocator.free(self.param_types);
    }
};

/// Portal (bound prepared statement)
const Portal = struct {
    statement_name: []const u8,
    param_values: []const []const u8,
    allocator: Allocator,

    pub fn deinit(self: *Portal) void {
        self.allocator.free(self.statement_name);
        for (self.param_values) |val| {
            self.allocator.free(val);
        }
        self.allocator.free(self.param_values);
    }
};

/// Session state holds per-connection runtime parameters and settings
pub const SessionState = struct {
    /// Current user/role (empty for trust authentication)
    user: []const u8,
    /// Current database name
    database: []const u8,
    /// Schema search path (comma-separated, default "public")
    search_path: []const u8,
    /// Client encoding (default "UTF8")
    client_encoding: []const u8,
    /// Statement timeout in milliseconds (0 = disabled)
    statement_timeout: u32,
    /// Application name (for logging/monitoring)
    application_name: []const u8,
    allocator: Allocator,

    pub fn init(allocator: Allocator, user: []const u8, database: []const u8) !SessionState {
        return .{
            .user = try allocator.dupe(u8, user),
            .database = try allocator.dupe(u8, database),
            .search_path = try allocator.dupe(u8, "public"),
            .client_encoding = try allocator.dupe(u8, "UTF8"),
            .statement_timeout = 0,
            .application_name = try allocator.dupe(u8, "silica"),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *SessionState) void {
        self.allocator.free(self.user);
        self.allocator.free(self.database);
        self.allocator.free(self.search_path);
        self.allocator.free(self.client_encoding);
        self.allocator.free(self.application_name);
    }

    /// Set a runtime parameter
    pub fn setParameter(self: *SessionState, name: []const u8, value: []const u8) !void {
        if (std.mem.eql(u8, name, "search_path")) {
            self.allocator.free(self.search_path);
            self.search_path = try self.allocator.dupe(u8, value);
        } else if (std.mem.eql(u8, name, "client_encoding")) {
            self.allocator.free(self.client_encoding);
            self.client_encoding = try self.allocator.dupe(u8, value);
        } else if (std.mem.eql(u8, name, "statement_timeout")) {
            self.statement_timeout = try std.fmt.parseInt(u32, value, 10);
        } else if (std.mem.eql(u8, name, "application_name")) {
            self.allocator.free(self.application_name);
            self.application_name = try self.allocator.dupe(u8, value);
        }
        // Ignore unknown parameters for now
    }

    /// Get a runtime parameter
    pub fn getParameter(self: *const SessionState, name: []const u8) ?[]const u8 {
        if (std.mem.eql(u8, name, "search_path")) {
            return self.search_path;
        } else if (std.mem.eql(u8, name, "client_encoding")) {
            return self.client_encoding;
        } else if (std.mem.eql(u8, name, "application_name")) {
            return self.application_name;
        }
        return null;
    }
};

/// Connection state
pub const Connection = struct {
    allocator: Allocator,
    db: *Database,
    transaction_status: wire.TransactionStatus,
    prepared_statements: std.StringHashMapUnmanaged(PreparedStatement),
    portals: std.StringHashMapUnmanaged(Portal),
    session: SessionState,

    pub fn init(allocator: Allocator, db: *Database, user: []const u8, database: []const u8) !Connection {
        return .{
            .allocator = allocator,
            .db = db,
            .transaction_status = .idle,
            .prepared_statements = .{},
            .portals = .{},
            .session = try SessionState.init(allocator, user, database),
        };
    }

    pub fn deinit(self: *Connection) void {
        // Clean up prepared statements
        var stmt_iter = self.prepared_statements.valueIterator();
        while (stmt_iter.next()) |stmt| {
            stmt.deinit();
        }
        self.prepared_statements.deinit(self.allocator);

        // Clean up portals
        var portal_iter = self.portals.valueIterator();
        while (portal_iter.next()) |portal| {
            portal.deinit();
        }
        self.portals.deinit(self.allocator);

        // Clean up session state
        self.session.deinit();
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
        defer result.close(self.allocator);

        // Determine if query returns rows
        if (result.rows) |rows_iter| {
            // Get first row to determine columns
            if (try rows_iter.next()) |first_row| {
                // Send RowDescription
                try self.sendRowDescription(writer, first_row.columns);

                // Send first DataRow
                try self.sendDataRow(writer, first_row, first_row.columns);

                // Send remaining DataRows
                while (try rows_iter.next()) |row| {
                    try self.sendDataRow(writer, row, row.columns);
                }
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

    /// Handle Parse message (extended query protocol)
    pub fn handleParse(
        self: *Connection,
        parse_msg: wire.Parse,
        writer: anytype,
    ) !void {
        // Store prepared statement
        const stmt_name = if (parse_msg.statement_name.len == 0)
            "" // Unnamed statement
        else
            parse_msg.statement_name;

        // Check if statement already exists
        if (self.prepared_statements.get(stmt_name)) |existing| {
            // Replace existing statement
            var old = existing;
            old.deinit();
            _ = self.prepared_statements.remove(stmt_name);
        }

        // Create new prepared statement
        const query_copy = try self.allocator.dupe(u8, parse_msg.query);
        errdefer self.allocator.free(query_copy);

        const param_types_copy = try self.allocator.dupe(i32, parse_msg.param_types);
        errdefer self.allocator.free(param_types_copy);

        const stmt = PreparedStatement{
            .query = query_copy,
            .param_types = param_types_copy,
            .allocator = self.allocator,
        };

        const stmt_name_copy = try self.allocator.dupe(u8, stmt_name);
        errdefer self.allocator.free(stmt_name_copy);

        try self.prepared_statements.put(self.allocator, stmt_name_copy, stmt);

        // Send ParseComplete
        try writer.writeByte(@intFromEnum(wire.BackendMessageType.parse_complete));
        try writer.writeInt(i32, 4, .big); // length (just the length field itself)
    }

    /// Handle Bind message (extended query protocol)
    pub fn handleBind(
        self: *Connection,
        bind_msg: wire.Bind,
        writer: anytype,
    ) !void {
        // Verify statement exists
        const stmt = self.prepared_statements.get(bind_msg.statement_name) orelse {
            try self.sendError(writer, error.StatementNotFound);
            return;
        };

        // Verify parameter count matches
        if (bind_msg.param_values.len != stmt.param_types.len) {
            try self.sendError(writer, error.ParameterCountMismatch);
            return;
        }

        // Create portal
        const portal_name = if (bind_msg.portal_name.len == 0)
            "" // Unnamed portal
        else
            bind_msg.portal_name;

        // Check if portal already exists
        if (self.portals.get(portal_name)) |existing| {
            var old = existing;
            old.deinit();
            _ = self.portals.remove(portal_name);
        }

        // Copy statement name
        const stmt_name_copy = try self.allocator.dupe(u8, bind_msg.statement_name);
        errdefer self.allocator.free(stmt_name_copy);

        // Copy parameter values
        const param_values_copy = try self.allocator.alloc([]const u8, bind_msg.param_values.len);
        errdefer self.allocator.free(param_values_copy);

        for (bind_msg.param_values, 0..) |val, i| {
            param_values_copy[i] = try self.allocator.dupe(u8, val);
        }

        const portal = Portal{
            .statement_name = stmt_name_copy,
            .param_values = param_values_copy,
            .allocator = self.allocator,
        };

        const portal_name_copy = try self.allocator.dupe(u8, portal_name);
        errdefer self.allocator.free(portal_name_copy);

        try self.portals.put(self.allocator, portal_name_copy, portal);

        // Send BindComplete
        try writer.writeByte(@intFromEnum(wire.BackendMessageType.bind_complete));
        try writer.writeInt(i32, 4, .big);
    }

    /// Handle Execute message (extended query protocol)
    pub fn handleExecute(
        self: *Connection,
        portal_name: []const u8,
        max_rows: i32,
        writer: anytype,
    ) !void {
        // Get portal
        const portal = self.portals.get(portal_name) orelse {
            try self.sendError(writer, error.PortalNotFound);
            return;
        };

        // Get prepared statement
        const stmt = self.prepared_statements.get(portal.statement_name) orelse {
            try self.sendError(writer, error.StatementNotFound);
            return;
        };

        // Substitute parameters in query (simplified - just execute as-is for now)
        // TODO: implement proper parameter substitution
        var result = self.db.execSQL(stmt.query) catch |err| {
            try self.sendError(writer, err);
            return;
        };
        defer result.close(self.allocator);

        // Send results with row limit
        // max_rows = 0 means return all rows
        // max_rows > 0 means return at most max_rows rows
        const limit: usize = if (max_rows <= 0) std.math.maxInt(usize) else @intCast(max_rows);
        var rows_sent: usize = 0;

        if (result.rows) |rows_iter| {
            // Get first row to determine columns
            if (try rows_iter.next()) |first_row| {
                // Send RowDescription
                try self.sendRowDescription(writer, first_row.columns);

                // Send first DataRow
                try self.sendDataRow(writer, first_row, first_row.columns);
                rows_sent += 1;

                // Send remaining DataRows up to limit
                while (rows_sent < limit) {
                    const row = (try rows_iter.next()) orelse break;
                    try self.sendDataRow(writer, row, row.columns);
                    rows_sent += 1;
                }
            }
        }

        // Send CommandComplete
        const tag = try self.getCommandTag(&result);
        const cmd_complete = wire.CommandComplete{ .tag = tag };
        try cmd_complete.write(writer);
    }

    /// Handle Close message (extended query protocol)
    pub fn handleClose(
        self: *Connection,
        close_type: u8, // 'S' for statement, 'P' for portal
        name: []const u8,
        writer: anytype,
    ) !void {
        if (close_type == 'S') {
            // Close prepared statement
            if (self.prepared_statements.fetchRemove(name)) |kv| {
                self.allocator.free(kv.key);
                var stmt = kv.value;
                stmt.deinit();
            }
        } else if (close_type == 'P') {
            // Close portal
            if (self.portals.fetchRemove(name)) |kv| {
                self.allocator.free(kv.key);
                var portal = kv.value;
                portal.deinit();
            }
        }

        // Send CloseComplete
        try writer.writeByte(@intFromEnum(wire.BackendMessageType.close_complete));
        try writer.writeInt(i32, 4, .big);
    }

    /// Handle Sync message (extended query protocol)
    pub fn handleSync(self: *Connection, writer: anytype) !void {
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
        row: silica.executor.Row,
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
        for (row.values, 0..) |value, i| {
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
            .null_value => return try self.allocator.dupe(u8, "NULL"),
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
            .tsvector, .tsquery => |t| return try self.allocator.dupe(u8, t),
            .array => |arr| {
                // Format as PostgreSQL array: {val1,val2,val3}
                try writer.writeByte('{');
                for (arr, 0..) |elem, i| {
                    if (i > 0) try writer.writeByte(',');
                    const elem_text = try self.valueToText(elem);
                    defer self.allocator.free(elem_text);
                    try writer.writeAll(elem_text);
                }
                try writer.writeByte('}');
            },
        }

        return try buf.toOwnedSlice(self.allocator);
    }

    /// Get command tag for CommandComplete message
    fn getCommandTag(self: *Connection, result: *QueryResult) ![]const u8 {
        _ = self;
        if (result.rows_affected > 0) {
            // For INSERT/UPDATE/DELETE, include row count
            // For now, just return a generic tag
            return "SELECT"; // Placeholder
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
            error.StatementNotFound => "26000", // invalid_sql_statement_name
            error.PortalNotFound => "34000", // invalid_cursor_name
            error.ParameterCountMismatch => "07001", // wrong_number_of_parameters
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
            error.StatementNotFound => "prepared statement does not exist",
            error.PortalNotFound => "portal does not exist",
            error.ParameterCountMismatch => "wrong number of parameters",
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

    var conn = try Connection.init(allocator, &db, "testuser", "testdb");
    defer conn.deinit();

    try std.testing.expectEqual(wire.TransactionStatus.idle, conn.transaction_status);
    try std.testing.expectEqualStrings("testuser", conn.session.user);
    try std.testing.expectEqualStrings("testdb", conn.session.database);
    try std.testing.expectEqualStrings("public", conn.session.search_path);
}

test "valueToText - integer" {
    const allocator = std.testing.allocator;

    const db_path = "test_value_int.db";
    defer std.fs.cwd().deleteFile(db_path) catch {};

    var db = try Database.init(allocator, db_path);
    defer db.deinit();

    var conn = try Connection.init(allocator, &db, "user", "db");
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

    var conn = try Connection.init(allocator, &db, "user", "db");
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

    var conn = try Connection.init(allocator, &db, "user", "db");
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

    var conn = try Connection.init(allocator, &db, "user", "db");
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

    var conn = try Connection.init(allocator, &db, "user", "db");
    defer conn.deinit();

    var result = try db.execSQL("CREATE TABLE test (id INTEGER)");
    defer result.close();

    const tag = try conn.getCommandTag(&result);
    try std.testing.expectEqualStrings("OK", tag);
}

test "handleParse - store prepared statement" {
    const allocator = std.testing.allocator;

    const db_path = "test_parse.db";
    defer std.fs.cwd().deleteFile(db_path) catch {};

    var db = try Database.init(allocator, db_path);
    defer db.deinit();

    var conn = try Connection.init(allocator, &db, "user", "db");
    defer conn.deinit();

    // Create Parse message
    const param_types = [_]i32{23}; // INT4
    const parse_msg = wire.Parse{
        .statement_name = "stmt1",
        .query = "SELECT $1",
        .param_types = &param_types,
    };

    var buf = std.ArrayListUnmanaged(u8){};
    defer buf.deinit(allocator);

    try conn.handleParse(parse_msg, buf.writer(allocator));

    // Verify ParseComplete was sent
    try std.testing.expectEqual(@as(u8, '1'), buf.items[0]);

    // Verify statement was stored
    const stmt = conn.prepared_statements.get("stmt1").?;
    try std.testing.expectEqualStrings("SELECT $1", stmt.query);
    try std.testing.expectEqual(@as(usize, 1), stmt.param_types.len);
    try std.testing.expectEqual(@as(i32, 23), stmt.param_types[0]);
}

test "handleBind - create portal" {
    const allocator = std.testing.allocator;

    const db_path = "test_bind.db";
    defer std.fs.cwd().deleteFile(db_path) catch {};

    var db = try Database.init(allocator, db_path);
    defer db.deinit();

    var conn = try Connection.init(allocator, &db, "user", "db");
    defer conn.deinit();

    // First, create a prepared statement
    const param_types = [_]i32{23};
    const parse_msg = wire.Parse{
        .statement_name = "stmt1",
        .query = "SELECT 42",
        .param_types = &param_types,
    };

    var parse_buf = std.ArrayListUnmanaged(u8){};
    defer parse_buf.deinit(allocator);
    try conn.handleParse(parse_msg, parse_buf.writer(allocator));

    // Now bind it
    const param_values = [_][]const u8{"42"};
    const result_formats = [_]i16{0};
    const param_formats = [_]i16{0};
    const bind_msg = wire.Bind{
        .portal_name = "portal1",
        .statement_name = "stmt1",
        .param_formats = &param_formats,
        .param_values = &param_values,
        .result_formats = &result_formats,
    };

    var bind_buf = std.ArrayListUnmanaged(u8){};
    defer bind_buf.deinit(allocator);

    try conn.handleBind(bind_msg, bind_buf.writer(allocator));

    // Verify BindComplete was sent
    try std.testing.expectEqual(@as(u8, '2'), bind_buf.items[0]);

    // Verify portal was stored
    const portal = conn.portals.get("portal1").?;
    try std.testing.expectEqualStrings("stmt1", portal.statement_name);
    try std.testing.expectEqual(@as(usize, 1), portal.param_values.len);
    try std.testing.expectEqualStrings("42", portal.param_values[0]);
}

test "handleClose - statement" {
    const allocator = std.testing.allocator;

    const db_path = "test_close_stmt.db";
    defer std.fs.cwd().deleteFile(db_path) catch {};

    var db = try Database.init(allocator, db_path);
    defer db.deinit();

    var conn = try Connection.init(allocator, &db, "user", "db");
    defer conn.deinit();

    // Create a prepared statement
    const param_types = [_]i32{};
    const parse_msg = wire.Parse{
        .statement_name = "stmt1",
        .query = "SELECT 1",
        .param_types = &param_types,
    };

    var parse_buf = std.ArrayListUnmanaged(u8){};
    defer parse_buf.deinit(allocator);
    try conn.handleParse(parse_msg, parse_buf.writer(allocator));

    // Verify statement exists
    try std.testing.expect(conn.prepared_statements.contains("stmt1"));

    // Close the statement
    var close_buf = std.ArrayListUnmanaged(u8){};
    defer close_buf.deinit(allocator);
    try conn.handleClose('S', "stmt1", close_buf.writer(allocator));

    // Verify CloseComplete was sent
    try std.testing.expectEqual(@as(u8, '3'), close_buf.items[0]);

    // Verify statement was removed
    try std.testing.expect(!conn.prepared_statements.contains("stmt1"));
}

test "handleSync - send ReadyForQuery" {
    const allocator = std.testing.allocator;

    const db_path = "test_sync.db";
    defer std.fs.cwd().deleteFile(db_path) catch {};

    var db = try Database.init(allocator, db_path);
    defer db.deinit();

    var conn = try Connection.init(allocator, &db, "user", "db");
    defer conn.deinit();

    var buf = std.ArrayListUnmanaged(u8){};
    defer buf.deinit(allocator);

    try conn.handleSync(buf.writer(allocator));

    // Verify ReadyForQuery was sent
    try std.testing.expectEqual(@as(u8, 'Z'), buf.items[0]);
}

test "valueToText - real" {
    const allocator = std.testing.allocator;

    const db_path = "test_value_real.db";
    defer std.fs.cwd().deleteFile(db_path) catch {};

    var db = try Database.init(allocator, db_path);
    defer db.deinit();

    var conn = try Connection.init(allocator, &db, "user", "db");
    defer conn.deinit();

    const value = Value{ .real = 3.14 };
    const text = try conn.valueToText(value);
    defer allocator.free(text);

    try std.testing.expectEqualStrings("3.14", text);
}

test "valueToText - boolean true" {
    const allocator = std.testing.allocator;

    const db_path = "test_value_bool_true.db";
    defer std.fs.cwd().deleteFile(db_path) catch {};

    var db = try Database.init(allocator, db_path);
    defer db.deinit();

    var conn = try Connection.init(allocator, &db, "user", "db");
    defer conn.deinit();

    const value = Value{ .boolean = true };
    const text = try conn.valueToText(value);
    defer allocator.free(text);

    try std.testing.expectEqualStrings("true", text);
}

test "valueToText - boolean false" {
    const allocator = std.testing.allocator;

    const db_path = "test_value_bool_false.db";
    defer std.fs.cwd().deleteFile(db_path) catch {};

    var db = try Database.init(allocator, db_path);
    defer db.deinit();

    var conn = try Connection.init(allocator, &db, "user", "db");
    defer conn.deinit();

    const value = Value{ .boolean = false };
    const text = try conn.valueToText(value);
    defer allocator.free(text);

    try std.testing.expectEqualStrings("false", text);
}

test "valueToText - blob hex encoding" {
    const allocator = std.testing.allocator;

    const db_path = "test_value_blob.db";
    defer std.fs.cwd().deleteFile(db_path) catch {};

    var db = try Database.init(allocator, db_path);
    defer db.deinit();

    var conn = try Connection.init(allocator, &db, "user", "db");
    defer conn.deinit();

    const blob_data = [_]u8{ 0xDE, 0xAD, 0xBE, 0xEF };
    const value = Value{ .blob = &blob_data };
    const text = try conn.valueToText(value);
    defer allocator.free(text);

    try std.testing.expectEqualStrings("deadbeef", text);
}

test "valueToText - uuid formatting" {
    const allocator = std.testing.allocator;

    const db_path = "test_value_uuid.db";
    defer std.fs.cwd().deleteFile(db_path) catch {};

    var db = try Database.init(allocator, db_path);
    defer db.deinit();

    var conn = try Connection.init(allocator, &db, "user", "db");
    defer conn.deinit();

    const uuid: [16]u8 = .{ 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88 };
    const value = Value{ .uuid = uuid };
    const text = try conn.valueToText(value);
    defer allocator.free(text);

    try std.testing.expectEqualStrings("12345678-9abc-def0-1122-334455667788", text);
}

test "valueToText - json passthrough" {
    const allocator = std.testing.allocator;

    const db_path = "test_value_json.db";
    defer std.fs.cwd().deleteFile(db_path) catch {};

    var db = try Database.init(allocator, db_path);
    defer db.deinit();

    var conn = try Connection.init(allocator, &db, "user", "db");
    defer conn.deinit();

    const value = Value{ .json = "{\"key\":\"value\"}" };
    const text = try conn.valueToText(value);
    defer allocator.free(text);

    try std.testing.expectEqualStrings("{\"key\":\"value\"}", text);
}

test "handleParse - replace existing statement" {
    const allocator = std.testing.allocator;

    const db_path = "test_parse_replace.db";
    defer std.fs.cwd().deleteFile(db_path) catch {};

    var db = try Database.init(allocator, db_path);
    defer db.deinit();

    var conn = try Connection.init(allocator, &db, "user", "db");
    defer conn.deinit();

    // Create first statement
    const param_types1 = [_]i32{23};
    const parse_msg1 = wire.Parse{
        .statement_name = "stmt1",
        .query = "SELECT $1",
        .param_types = &param_types1,
    };

    var buf1 = std.ArrayListUnmanaged(u8){};
    defer buf1.deinit(allocator);
    try conn.handleParse(parse_msg1, buf1.writer(allocator));

    // Replace with second statement
    const param_types2 = [_]i32{ 23, 25 };
    const parse_msg2 = wire.Parse{
        .statement_name = "stmt1",
        .query = "SELECT $1, $2",
        .param_types = &param_types2,
    };

    var buf2 = std.ArrayListUnmanaged(u8){};
    defer buf2.deinit(allocator);
    try conn.handleParse(parse_msg2, buf2.writer(allocator));

    // Verify new statement replaced old one
    const stmt = conn.prepared_statements.get("stmt1").?;
    try std.testing.expectEqualStrings("SELECT $1, $2", stmt.query);
    try std.testing.expectEqual(@as(usize, 2), stmt.param_types.len);
}

test "handleParse - unnamed statement" {
    const allocator = std.testing.allocator;

    const db_path = "test_parse_unnamed.db";
    defer std.fs.cwd().deleteFile(db_path) catch {};

    var db = try Database.init(allocator, db_path);
    defer db.deinit();

    var conn = try Connection.init(allocator, &db, "user", "db");
    defer conn.deinit();

    // Create unnamed statement (empty name)
    const param_types = [_]i32{};
    const parse_msg = wire.Parse{
        .statement_name = "",
        .query = "SELECT 1",
        .param_types = &param_types,
    };

    var buf = std.ArrayListUnmanaged(u8){};
    defer buf.deinit(allocator);
    try conn.handleParse(parse_msg, buf.writer(allocator));

    // Verify unnamed statement was stored with empty key
    const stmt = conn.prepared_statements.get("").?;
    try std.testing.expectEqualStrings("SELECT 1", stmt.query);
}

test "handleBind - parameter count mismatch" {
    const allocator = std.testing.allocator;

    const db_path = "test_bind_mismatch.db";
    defer std.fs.cwd().deleteFile(db_path) catch {};

    var db = try Database.init(allocator, db_path);
    defer db.deinit();

    var conn = try Connection.init(allocator, &db, "user", "db");
    defer conn.deinit();

    // Create a prepared statement expecting 2 parameters
    const param_types = [_]i32{ 23, 25 };
    const parse_msg = wire.Parse{
        .statement_name = "stmt1",
        .query = "SELECT $1, $2",
        .param_types = &param_types,
    };

    var parse_buf = std.ArrayListUnmanaged(u8){};
    defer parse_buf.deinit(allocator);
    try conn.handleParse(parse_msg, parse_buf.writer(allocator));

    // Try to bind with only 1 parameter
    const param_values = [_][]const u8{"42"};
    const result_formats = [_]i16{0};
    const param_formats = [_]i16{0};
    const bind_msg = wire.Bind{
        .portal_name = "portal1",
        .statement_name = "stmt1",
        .param_formats = &param_formats,
        .param_values = &param_values,
        .result_formats = &result_formats,
    };

    var bind_buf = std.ArrayListUnmanaged(u8){};
    defer bind_buf.deinit(allocator);

    try conn.handleBind(bind_msg, bind_buf.writer(allocator));

    // Verify ErrorResponse was sent (message type 'E')
    try std.testing.expectEqual(@as(u8, 'E'), bind_buf.items[0]);
}

test "handleBind - statement not found" {
    const allocator = std.testing.allocator;

    const db_path = "test_bind_notfound.db";
    defer std.fs.cwd().deleteFile(db_path) catch {};

    var db = try Database.init(allocator, db_path);
    defer db.deinit();

    var conn = try Connection.init(allocator, &db, "user", "db");
    defer conn.deinit();

    // Try to bind to non-existent statement
    const param_values = [_][]const u8{};
    const result_formats = [_]i16{};
    const param_formats = [_]i16{};
    const bind_msg = wire.Bind{
        .portal_name = "portal1",
        .statement_name = "nonexistent",
        .param_formats = &param_formats,
        .param_values = &param_values,
        .result_formats = &result_formats,
    };

    var buf = std.ArrayListUnmanaged(u8){};
    defer buf.deinit(allocator);

    try conn.handleBind(bind_msg, buf.writer(allocator));

    // Verify ErrorResponse was sent
    try std.testing.expectEqual(@as(u8, 'E'), buf.items[0]);
}

test "handleBind - unnamed portal" {
    const allocator = std.testing.allocator;

    const db_path = "test_bind_unnamed.db";
    defer std.fs.cwd().deleteFile(db_path) catch {};

    var db = try Database.init(allocator, db_path);
    defer db.deinit();

    var conn = try Connection.init(allocator, &db, "user", "db");
    defer conn.deinit();

    // Create a prepared statement
    const param_types = [_]i32{};
    const parse_msg = wire.Parse{
        .statement_name = "stmt1",
        .query = "SELECT 1",
        .param_types = &param_types,
    };

    var parse_buf = std.ArrayListUnmanaged(u8){};
    defer parse_buf.deinit(allocator);
    try conn.handleParse(parse_msg, parse_buf.writer(allocator));

    // Bind with unnamed portal (empty name)
    const param_values = [_][]const u8{};
    const result_formats = [_]i16{};
    const param_formats = [_]i16{};
    const bind_msg = wire.Bind{
        .portal_name = "",
        .statement_name = "stmt1",
        .param_formats = &param_formats,
        .param_values = &param_values,
        .result_formats = &result_formats,
    };

    var bind_buf = std.ArrayListUnmanaged(u8){};
    defer bind_buf.deinit(allocator);

    try conn.handleBind(bind_msg, bind_buf.writer(allocator));

    // Verify unnamed portal was created
    const portal = conn.portals.get("").?;
    try std.testing.expectEqualStrings("stmt1", portal.statement_name);
}

test "handleClose - portal" {
    const allocator = std.testing.allocator;

    const db_path = "test_close_portal.db";
    defer std.fs.cwd().deleteFile(db_path) catch {};

    var db = try Database.init(allocator, db_path);
    defer db.deinit();

    var conn = try Connection.init(allocator, &db, "user", "db");
    defer conn.deinit();

    // Create a prepared statement and portal
    const param_types = [_]i32{};
    const parse_msg = wire.Parse{
        .statement_name = "stmt1",
        .query = "SELECT 1",
        .param_types = &param_types,
    };

    var parse_buf = std.ArrayListUnmanaged(u8){};
    defer parse_buf.deinit(allocator);
    try conn.handleParse(parse_msg, parse_buf.writer(allocator));

    const param_values = [_][]const u8{};
    const result_formats = [_]i16{};
    const param_formats = [_]i16{};
    const bind_msg = wire.Bind{
        .portal_name = "portal1",
        .statement_name = "stmt1",
        .param_formats = &param_formats,
        .param_values = &param_values,
        .result_formats = &result_formats,
    };

    var bind_buf = std.ArrayListUnmanaged(u8){};
    defer bind_buf.deinit(allocator);
    try conn.handleBind(bind_msg, bind_buf.writer(allocator));

    // Verify portal exists
    try std.testing.expect(conn.portals.contains("portal1"));

    // Close the portal
    var close_buf = std.ArrayListUnmanaged(u8){};
    defer close_buf.deinit(allocator);
    try conn.handleClose('P', "portal1", close_buf.writer(allocator));

    // Verify CloseComplete was sent
    try std.testing.expectEqual(@as(u8, '3'), close_buf.items[0]);

    // Verify portal was removed
    try std.testing.expect(!conn.portals.contains("portal1"));
}

test "handleClose - nonexistent statement (no error)" {
    const allocator = std.testing.allocator;

    const db_path = "test_close_noexist.db";
    defer std.fs.cwd().deleteFile(db_path) catch {};

    var db = try Database.init(allocator, db_path);
    defer db.deinit();

    var conn = try Connection.init(allocator, &db, "user", "db");
    defer conn.deinit();

    // Close non-existent statement (should succeed silently)
    var buf = std.ArrayListUnmanaged(u8){};
    defer buf.deinit(allocator);
    try conn.handleClose('S', "nonexistent", buf.writer(allocator));

    // Verify CloseComplete was sent
    try std.testing.expectEqual(@as(u8, '3'), buf.items[0]);
}

test "getSQLState - various errors" {
    const allocator = std.testing.allocator;

    const db_path = "test_sqlstate_all.db";
    defer std.fs.cwd().deleteFile(db_path) catch {};

    var db = try Database.init(allocator, db_path);
    defer db.deinit();

    var conn = try Connection.init(allocator, &db, "user", "db");
    defer conn.deinit();

    try std.testing.expectEqualStrings("42601", try conn.getSQLState(error.ParseError));
    try std.testing.expectEqualStrings("42703", try conn.getSQLState(error.ColumnNotFound));
    try std.testing.expectEqualStrings("22012", try conn.getSQLState(error.DivisionByZero));
    try std.testing.expectEqualStrings("53200", try conn.getSQLState(error.OutOfMemory));
    try std.testing.expectEqualStrings("26000", try conn.getSQLState(error.StatementNotFound));
    try std.testing.expectEqualStrings("34000", try conn.getSQLState(error.PortalNotFound));
    try std.testing.expectEqualStrings("07001", try conn.getSQLState(error.ParameterCountMismatch));
    try std.testing.expectEqualStrings("XX000", try conn.getSQLState(error.UnexpectedError));
}

test "SessionState - init with defaults" {
    const allocator = std.testing.allocator;

    var session = try SessionState.init(allocator, "testuser", "testdb");
    defer session.deinit();

    try std.testing.expectEqualStrings("testuser", session.user);
    try std.testing.expectEqualStrings("testdb", session.database);
    try std.testing.expectEqualStrings("public", session.search_path);
    try std.testing.expectEqualStrings("UTF8", session.client_encoding);
    try std.testing.expectEqual(@as(u32, 0), session.statement_timeout);
    try std.testing.expectEqualStrings("silica", session.application_name);
}

test "SessionState - set search_path parameter" {
    const allocator = std.testing.allocator;

    var session = try SessionState.init(allocator, "user", "db");
    defer session.deinit();

    try session.setParameter("search_path", "myschema,public");
    try std.testing.expectEqualStrings("myschema,public", session.search_path);
}

test "SessionState - set client_encoding parameter" {
    const allocator = std.testing.allocator;

    var session = try SessionState.init(allocator, "user", "db");
    defer session.deinit();

    try session.setParameter("client_encoding", "LATIN1");
    try std.testing.expectEqualStrings("LATIN1", session.client_encoding);
}

test "SessionState - set statement_timeout parameter" {
    const allocator = std.testing.allocator;

    var session = try SessionState.init(allocator, "user", "db");
    defer session.deinit();

    try session.setParameter("statement_timeout", "5000");
    try std.testing.expectEqual(@as(u32, 5000), session.statement_timeout);
}

test "SessionState - set application_name parameter" {
    const allocator = std.testing.allocator;

    var session = try SessionState.init(allocator, "user", "db");
    defer session.deinit();

    try session.setParameter("application_name", "myapp");
    try std.testing.expectEqualStrings("myapp", session.application_name);
}

test "SessionState - get parameter" {
    const allocator = std.testing.allocator;

    var session = try SessionState.init(allocator, "user", "db");
    defer session.deinit();

    try session.setParameter("search_path", "custom");

    const value = session.getParameter("search_path");
    try std.testing.expect(value != null);
    try std.testing.expectEqualStrings("custom", value.?);
}

test "SessionState - get unknown parameter" {
    const allocator = std.testing.allocator;

    var session = try SessionState.init(allocator, "user", "db");
    defer session.deinit();

    const value = session.getParameter("unknown_param");
    try std.testing.expect(value == null);
}

test "SessionState - set unknown parameter (ignored)" {
    const allocator = std.testing.allocator;

    var session = try SessionState.init(allocator, "user", "db");
    defer session.deinit();

    // Should not error, just ignore
    try session.setParameter("unknown_param", "value");
}

test "SessionState - invalid statement_timeout value" {
    const allocator = std.testing.allocator;

    var session = try SessionState.init(allocator, "user", "db");
    defer session.deinit();

    // Invalid timeout should return error
    const result = session.setParameter("statement_timeout", "not_a_number");
    try std.testing.expectError(error.InvalidCharacter, result);
}

// ── Comprehensive Edge Case Tests ───────────────────────────────────

test "valueToText - very long text (>1MB)" {
    const allocator = std.testing.allocator;

    const db_path = "test_value_long.db";
    defer std.fs.cwd().deleteFile(db_path) catch {};

    var db = try Database.init(allocator, db_path);
    defer db.deinit();

    var conn = try Connection.init(allocator, &db, "user", "db");
    defer conn.deinit();

    // Create a 2MB string
    const large_text = try allocator.alloc(u8, 2 * 1024 * 1024);
    defer allocator.free(large_text);
    @memset(large_text, 'A');

    const val = Value{ .text = large_text };
    const result = try conn.valueToText(val);
    defer allocator.free(result);

    try std.testing.expectEqual(large_text.len, result.len);
    try std.testing.expectEqualStrings(large_text, result);
}

test "valueToText - empty text" {
    const allocator = std.testing.allocator;

    const db_path = "test_value_empty.db";
    defer std.fs.cwd().deleteFile(db_path) catch {};

    var db = try Database.init(allocator, db_path);
    defer db.deinit();

    var conn = try Connection.init(allocator, &db, "user", "db");
    defer conn.deinit();

    const val = Value{ .text = "" };
    const result = try conn.valueToText(val);
    defer allocator.free(result);

    try std.testing.expectEqual(@as(usize, 0), result.len);
}

test "valueToText - text with null bytes" {
    const allocator = std.testing.allocator;

    const db_path = "test_value_nullbytes.db";
    defer std.fs.cwd().deleteFile(db_path) catch {};

    var db = try Database.init(allocator, db_path);
    defer db.deinit();

    var conn = try Connection.init(allocator, &db, "user", "db");
    defer conn.deinit();

    const text_with_nulls = "hello\x00world\x00!";
    const val = Value{ .text = text_with_nulls };
    const result = try conn.valueToText(val);
    defer allocator.free(result);

    try std.testing.expectEqualStrings(text_with_nulls, result);
}

test "valueToText - integer min/max values" {
    const allocator = std.testing.allocator;

    const db_path = "test_value_minmax.db";
    defer std.fs.cwd().deleteFile(db_path) catch {};

    var db = try Database.init(allocator, db_path);
    defer db.deinit();

    var conn = try Connection.init(allocator, &db, "user", "db");
    defer conn.deinit();

    // Test i64 minimum
    const min_val = Value{ .integer = std.math.minInt(i64) };
    const min_result = try conn.valueToText(min_val);
    defer allocator.free(min_result);
    try std.testing.expectEqualStrings("-9223372036854775808", min_result);

    // Test i64 maximum
    const max_val = Value{ .integer = std.math.maxInt(i64) };
    const max_result = try conn.valueToText(max_val);
    defer allocator.free(max_result);
    try std.testing.expectEqualStrings("9223372036854775807", max_result);
}

test "valueToText - real edge values" {
    const allocator = std.testing.allocator;

    const db_path = "test_value_real_edge.db";
    defer std.fs.cwd().deleteFile(db_path) catch {};

    var db = try Database.init(allocator, db_path);
    defer db.deinit();

    var conn = try Connection.init(allocator, &db, "user", "db");
    defer conn.deinit();

    // Test zero
    const zero = Value{ .real = 0.0 };
    const zero_result = try conn.valueToText(zero);
    defer allocator.free(zero_result);
    try std.testing.expect(std.mem.startsWith(u8, zero_result, "0"));

    // Test very small positive number
    const small = Value{ .real = 1e-308 };
    const small_result = try conn.valueToText(small);
    defer allocator.free(small_result);
    try std.testing.expect(small_result.len > 0);

    // Test very large number
    const large = Value{ .real = 1e308 };
    const large_result = try conn.valueToText(large);
    defer allocator.free(large_result);
    try std.testing.expect(large_result.len > 0);
}

test "valueToText - blob empty" {
    const allocator = std.testing.allocator;

    const db_path = "test_value_blob_empty.db";
    defer std.fs.cwd().deleteFile(db_path) catch {};

    var db = try Database.init(allocator, db_path);
    defer db.deinit();

    var conn = try Connection.init(allocator, &db, "user", "db");
    defer conn.deinit();

    const empty: []const u8 = &[_]u8{};
    const val = Value{ .blob = empty };
    const result = try conn.valueToText(val);
    defer allocator.free(result);

    // Empty blob should produce empty hex string or "\\x"
    try std.testing.expect(result.len == 0 or std.mem.eql(u8, result, "\\x"));
}

test "valueToText - blob with all byte values" {
    const allocator = std.testing.allocator;

    const db_path = "test_value_blob_all.db";
    defer std.fs.cwd().deleteFile(db_path) catch {};

    var db = try Database.init(allocator, db_path);
    defer db.deinit();

    var conn = try Connection.init(allocator, &db, "user", "db");
    defer conn.deinit();

    // Create blob with all possible byte values 0x00-0xFF
    var blob_data: [256]u8 = undefined;
    for (&blob_data, 0..) |*byte, i| {
        byte.* = @intCast(i);
    }

    const val = Value{ .blob = &blob_data };
    const result = try conn.valueToText(val);
    defer allocator.free(result);

    // Should be hex-encoded: \\x + 512 hex chars
    try std.testing.expect(result.len == 2 + 512 or result.len == 512);
}

test "getSQLState - comprehensive error mapping" {
    const allocator = std.testing.allocator;

    const db_path = "test_sqlstate_comprehensive.db";
    defer std.fs.cwd().deleteFile(db_path) catch {};

    var db = try Database.init(allocator, db_path);
    defer db.deinit();

    var conn = try Connection.init(allocator, &db, "user", "db");
    defer conn.deinit();

    // Test additional error mappings
    try std.testing.expectEqualStrings("23505", try conn.getSQLState(error.DuplicateKey));
    try std.testing.expectEqualStrings("42P07", try conn.getSQLState(error.TableAlreadyExists));
    try std.testing.expectEqualStrings("42P01", try conn.getSQLState(error.TableNotFound));
    try std.testing.expectEqualStrings("40001", try conn.getSQLState(error.SerializationFailure));
    try std.testing.expectEqualStrings("40P01", try conn.getSQLState(error.DeadlockDetected));
}

test "handleParse - very long statement name (255 chars)" {
    const allocator = std.testing.allocator;

    const db_path = "test_parse_long.db";
    defer std.fs.cwd().deleteFile(db_path) catch {};

    var db = try Database.init(allocator, db_path);
    defer db.deinit();

    var conn = try Connection.init(allocator, &db, "user", "db");
    defer conn.deinit();

    // Create 255-character statement name
    const long_name = try allocator.alloc(u8, 255);
    defer allocator.free(long_name);
    @memset(long_name, 'S');

    const parse_msg = wire.Parse{
        .statement_name = long_name,
        .query = "SELECT 1",
        .param_types = &[_]i32{},
    };

    var buf = std.ArrayListUnmanaged(u8){};
    defer buf.deinit(allocator);

    try conn.handleParse(parse_msg, buf.writer(allocator));

    // Verify it was stored
    try std.testing.expect(conn.prepared_statements.contains(long_name));
}

test "handleParse - statement name with special characters" {
    const allocator = std.testing.allocator;

    const db_path = "test_parse_special.db";
    defer std.fs.cwd().deleteFile(db_path) catch {};

    var db = try Database.init(allocator, db_path);
    defer db.deinit();

    var conn = try Connection.init(allocator, &db, "user", "db");
    defer conn.deinit();

    const special_name = "stmt!@#$%^&*()_+-=[]{}|;:',.<>?/~`";
    const parse_msg = wire.Parse{
        .statement_name = special_name,
        .query = "SELECT 2",
        .param_types = &[_]i32{},
    };

    var buf = std.ArrayListUnmanaged(u8){};
    defer buf.deinit(allocator);

    try conn.handleParse(parse_msg, buf.writer(allocator));

    try std.testing.expect(conn.prepared_statements.contains(special_name));
}

test "handleBind - very long portal name" {
    const allocator = std.testing.allocator;

    const db_path = "test_bind_long.db";
    defer std.fs.cwd().deleteFile(db_path) catch {};

    var db = try Database.init(allocator, db_path);
    defer db.deinit();

    var conn = try Connection.init(allocator, &db, "user", "db");
    defer conn.deinit();

    // First create a prepared statement
    const param_types = [_]i32{23};
    const parse_msg = wire.Parse{
        .statement_name = "test_stmt",
        .query = "SELECT $1",
        .param_types = &param_types,
    };

    var parse_buf = std.ArrayListUnmanaged(u8){};
    defer parse_buf.deinit(allocator);
    try conn.handleParse(parse_msg, parse_buf.writer(allocator));

    // Create 255-character portal name
    const long_portal = try allocator.alloc(u8, 255);
    defer allocator.free(long_portal);
    @memset(long_portal, 'P');

    const param_values = [_][]const u8{"42"};
    const bind_msg = wire.Bind{
        .portal_name = long_portal,
        .statement_name = "test_stmt",
        .param_values = &param_values,
    };

    var bind_buf = std.ArrayListUnmanaged(u8){};
    defer bind_buf.deinit(allocator);
    try conn.handleBind(bind_msg, bind_buf.writer(allocator));

    try std.testing.expect(conn.portals.contains(long_portal));
}

test "handleClose - close same statement twice (idempotent)" {
    const allocator = std.testing.allocator;

    const db_path = "test_close_twice.db";
    defer std.fs.cwd().deleteFile(db_path) catch {};

    var db = try Database.init(allocator, db_path);
    defer db.deinit();

    var conn = try Connection.init(allocator, &db, "user", "db");
    defer conn.deinit();

    // Create statement
    const parse_msg = wire.Parse{
        .statement_name = "stmt",
        .query = "SELECT 1",
        .param_types = &[_]i32{},
    };

    var parse_buf = std.ArrayListUnmanaged(u8){};
    defer parse_buf.deinit(allocator);
    try conn.handleParse(parse_msg, parse_buf.writer(allocator));

    // Close it twice - should not error
    const close_msg = wire.Close{
        .target_type = .statement,
        .target_name = "stmt",
    };

    var close_buf1 = std.ArrayListUnmanaged(u8){};
    defer close_buf1.deinit(allocator);
    try conn.handleClose(close_msg, close_buf1.writer(allocator));

    var close_buf2 = std.ArrayListUnmanaged(u8){};
    defer close_buf2.deinit(allocator);
    try conn.handleClose(close_msg, close_buf2.writer(allocator)); // Second close should be no-op

    try std.testing.expect(!conn.prepared_statements.contains("stmt"));
}

test "SessionState - statement_timeout boundary values" {
    const allocator = std.testing.allocator;

    var session = try SessionState.init(allocator, "user", "db");
    defer session.deinit();

    // Zero timeout (disabled)
    try session.setParameter("statement_timeout", "0");
    try std.testing.expectEqual(@as(u32, 0), session.statement_timeout);

    // Maximum valid timeout (u32 max)
    try session.setParameter("statement_timeout", "4294967295");
    try std.testing.expectEqual(@as(u32, 4294967295), session.statement_timeout);

    // Negative should error
    const result = session.setParameter("statement_timeout", "-1");
    try std.testing.expectError(error.Overflow, result);
}

test "SessionState - parameter case sensitivity" {
    const allocator = std.testing.allocator;

    var session = try SessionState.init(allocator, "user", "db");
    defer session.deinit();

    // All parameters should update (parameter names are currently case-sensitive in our implementation)
    try session.setParameter("search_path", "public");
    try std.testing.expectEqualStrings("public", session.search_path);

    try session.setParameter("search_path", "schema1");
    try std.testing.expectEqualStrings("schema1", session.search_path);
}

test "handleParse - many statements stress" {
    const allocator = std.testing.allocator;

    const db_path = "test_parse_many.db";
    defer std.fs.cwd().deleteFile(db_path) catch {};

    var db = try Database.init(allocator, db_path);
    defer db.deinit();

    var conn = try Connection.init(allocator, &db, "user", "db");
    defer conn.deinit();

    // Create many statements rapidly
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        const name = try std.fmt.allocPrint(allocator, "stmt_{d}", .{i});
        defer allocator.free(name);

        const parse_msg = wire.Parse{
            .statement_name = name,
            .query = "SELECT 1",
            .param_types = &[_]i32{},
        };

        var buf = std.ArrayListUnmanaged(u8){};
        defer buf.deinit(allocator);
        try conn.handleParse(parse_msg, buf.writer(allocator));
    }

    try std.testing.expectEqual(@as(usize, 100), conn.prepared_statements.count());
}

test "valueToText - all Value type variants coverage" {
    const allocator = std.testing.allocator;

    const db_path = "test_value_variants.db";
    defer std.fs.cwd().deleteFile(db_path) catch {};

    var db = try Database.init(allocator, db_path);
    defer db.deinit();

    var conn = try Connection.init(allocator, &db, "user", "db");
    defer conn.deinit();

    // Integer
    const int_result = try conn.valueToText(Value{ .integer = 123 });
    defer allocator.free(int_result);
    try std.testing.expectEqualStrings("123", int_result);

    // Real
    const real_result = try conn.valueToText(Value{ .real = 3.14 });
    defer allocator.free(real_result);
    try std.testing.expect(std.mem.startsWith(u8, real_result, "3.14"));

    // Text
    const text_result = try conn.valueToText(Value{ .text = "hello" });
    defer allocator.free(text_result);
    try std.testing.expectEqualStrings("hello", text_result);

    // Null
    const null_result = try conn.valueToText(Value.null_value);
    defer allocator.free(null_result);
    try std.testing.expectEqualStrings("", null_result);

    // Boolean true
    const true_result = try conn.valueToText(Value{ .boolean = true });
    defer allocator.free(true_result);
    try std.testing.expectEqualStrings("t", true_result);

    // Boolean false
    const false_result = try conn.valueToText(Value{ .boolean = false });
    defer allocator.free(false_result);
    try std.testing.expectEqualStrings("f", false_result);

    // UUID
    const uuid_bytes = [_]u8{0x12} ++ [_]u8{0x34} ++ [_]u8{0x56} ++ [_]u8{0x78} ++
        [_]u8{0x90} ++ [_]u8{0xab} ++ [_]u8{0xcd} ++ [_]u8{0xef} ++
        [_]u8{0x12} ++ [_]u8{0x34} ++ [_]u8{0x56} ++ [_]u8{0x78} ++
        [_]u8{0x90} ++ [_]u8{0xab} ++ [_]u8{0xcd} ++ [_]u8{0xef};
    const uuid_result = try conn.valueToText(Value{ .uuid = uuid_bytes });
    defer allocator.free(uuid_result);
    try std.testing.expectEqualStrings("12345678-90ab-cdef-1234-567890abcdef", uuid_result);

    // JSON
    const json_result = try conn.valueToText(Value{ .json = "{\"key\":\"value\"}" });
    defer allocator.free(json_result);
    try std.testing.expectEqualStrings("{\"key\":\"value\"}", json_result);
}

test "handleExecute - max_rows limit" {
    const allocator = std.testing.allocator;

    const db_path = "test_execute_maxrows.db";
    defer std.fs.cwd().deleteFile(db_path) catch {};

    var db = try Database.init(allocator, db_path);
    defer db.deinit();

    // Create test table with multiple rows
    _ = try db.execSQL("CREATE TABLE test_rows (id INTEGER, value TEXT)");
    _ = try db.execSQL("INSERT INTO test_rows VALUES (1, 'row1')");
    _ = try db.execSQL("INSERT INTO test_rows VALUES (2, 'row2')");
    _ = try db.execSQL("INSERT INTO test_rows VALUES (3, 'row3')");
    _ = try db.execSQL("INSERT INTO test_rows VALUES (4, 'row4')");
    _ = try db.execSQL("INSERT INTO test_rows VALUES (5, 'row5')");

    var conn = try Connection.init(allocator, &db, "user", "db");
    defer conn.deinit();

    // Parse statement
    const param_types = [_]i32{};
    const parse_msg = wire.Parse{
        .statement_name = "stmt1",
        .query = "SELECT * FROM test_rows ORDER BY id",
        .param_types = &param_types,
    };

    var parse_buf = std.ArrayListUnmanaged(u8){};
    defer parse_buf.deinit(allocator);
    try conn.handleParse(parse_msg, parse_buf.writer(allocator));

    // Bind portal
    const param_values = [_][]const u8{};
    const result_formats = [_]i16{};
    const param_formats = [_]i16{};
    const bind_msg = wire.Bind{
        .portal_name = "portal1",
        .statement_name = "stmt1",
        .param_formats = &param_formats,
        .param_values = &param_values,
        .result_formats = &result_formats,
    };

    var bind_buf = std.ArrayListUnmanaged(u8){};
    defer bind_buf.deinit(allocator);
    try conn.handleBind(bind_msg, bind_buf.writer(allocator));

    // Test 1: Execute with max_rows = 0 (all rows)
    {
        var exec_buf = std.ArrayListUnmanaged(u8){};
        defer exec_buf.deinit(allocator);
        try conn.handleExecute("portal1", 0, exec_buf.writer(allocator));

        // Count DataRow messages (message type 'D')
        var row_count: usize = 0;
        for (exec_buf.items) |byte| {
            if (byte == 'D') row_count += 1;
        }
        try std.testing.expectEqual(@as(usize, 5), row_count);
    }

    // Test 2: Execute with max_rows = 2
    {
        var exec_buf2 = std.ArrayListUnmanaged(u8){};
        defer exec_buf2.deinit(allocator);
        try conn.handleExecute("portal1", 2, exec_buf2.writer(allocator));

        // Count DataRow messages
        var row_count: usize = 0;
        for (exec_buf2.items) |byte| {
            if (byte == 'D') row_count += 1;
        }
        try std.testing.expectEqual(@as(usize, 2), row_count);
    }

    // Test 3: Execute with max_rows = 10 (more than available)
    {
        var exec_buf3 = std.ArrayListUnmanaged(u8){};
        defer exec_buf3.deinit(allocator);
        try conn.handleExecute("portal1", 10, exec_buf3.writer(allocator));

        // Count DataRow messages - should return all 5 rows
        var row_count: usize = 0;
        for (exec_buf3.items) |byte| {
            if (byte == 'D') row_count += 1;
        }
        try std.testing.expectEqual(@as(usize, 5), row_count);
    }

    // Test 4: Execute with max_rows = 1
    {
        var exec_buf4 = std.ArrayListUnmanaged(u8){};
        defer exec_buf4.deinit(allocator);
        try conn.handleExecute("portal1", 1, exec_buf4.writer(allocator));

        // Count DataRow messages
        var row_count: usize = 0;
        for (exec_buf4.items) |byte| {
            if (byte == 'D') row_count += 1;
        }
        try std.testing.expectEqual(@as(usize, 1), row_count);
    }
}

test "handleExecute - max_rows with negative value" {
    const allocator = std.testing.allocator;

    const db_path = "test_execute_negative.db";
    defer std.fs.cwd().deleteFile(db_path) catch {};

    var db = try Database.init(allocator, db_path);
    defer db.deinit();

    // Create test table
    _ = try db.execSQL("CREATE TABLE test_neg (id INTEGER)");
    _ = try db.execSQL("INSERT INTO test_neg VALUES (1)");
    _ = try db.execSQL("INSERT INTO test_neg VALUES (2)");

    var conn = try Connection.init(allocator, &db, "user", "db");
    defer conn.deinit();

    // Parse and bind
    const param_types = [_]i32{};
    const parse_msg = wire.Parse{
        .statement_name = "stmt1",
        .query = "SELECT * FROM test_neg",
        .param_types = &param_types,
    };

    var parse_buf = std.ArrayListUnmanaged(u8){};
    defer parse_buf.deinit(allocator);
    try conn.handleParse(parse_msg, parse_buf.writer(allocator));

    const param_values = [_][]const u8{};
    const result_formats = [_]i16{};
    const param_formats = [_]i16{};
    const bind_msg = wire.Bind{
        .portal_name = "portal1",
        .statement_name = "stmt1",
        .param_formats = &param_formats,
        .param_values = &param_values,
        .result_formats = &result_formats,
    };

    var bind_buf = std.ArrayListUnmanaged(u8){};
    defer bind_buf.deinit(allocator);
    try conn.handleBind(bind_msg, bind_buf.writer(allocator));

    // Execute with negative max_rows (should return all rows)
    var exec_buf = std.ArrayListUnmanaged(u8){};
    defer exec_buf.deinit(allocator);
    try conn.handleExecute("portal1", -1, exec_buf.writer(allocator));

    // Count DataRow messages - should return all 2 rows
    var row_count: usize = 0;
    for (exec_buf.items) |byte| {
        if (byte == 'D') row_count += 1;
    }
    try std.testing.expectEqual(@as(usize, 2), row_count);
}
