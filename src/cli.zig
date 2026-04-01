const std = @import("std");
const sailor = @import("sailor");
const silica = @import("silica");

const tui_mod = @import("tui.zig");
const server_mod = @import("server/server.zig");

const version = "1.0.0";

const executor = silica.executor;
const Value = executor.Value;
const Row = executor.Row;
const Database = silica.engine.Database;
const QueryResult = silica.engine.QueryResult;

// ── Output Mode ───────────────────────────────────────────────────────

pub const OutputMode = enum {
    table,
    csv,
    json,
    jsonl,
    plain,
};

// ── CLI Flags ──────────────────────────────────────────────────────────

const CliFlags = [_]sailor.arg.FlagDef{
    .{ .name = "help", .short = 'h', .type = .bool, .help = "Show this help message" },
    .{ .name = "version", .short = 'v', .type = .bool, .help = "Show version information" },
    .{ .name = "header", .type = .bool, .help = "Show column headers in output", .default = "true" },
    .{ .name = "csv", .type = .bool, .help = "Output in CSV format" },
    .{ .name = "json", .type = .bool, .help = "Output in JSON format" },
    .{ .name = "mode", .short = 'm', .type = .string, .help = "Output mode: table, csv, json, jsonl, plain" },
    .{ .name = "tui", .short = 't', .type = .bool, .help = "Launch TUI database browser" },
    .{ .name = "host", .type = .string, .help = "Server host (default: 127.0.0.1)" },
    .{ .name = "port", .short = 'p', .type = .string, .help = "Server port (default: 5433)" },
    .{ .name = "max-connections", .type = .string, .help = "Maximum connections (default: 100)" },
};

// ── Server Mode ───────────────────────────────────────────────────────

fn runServer(
    allocator: std.mem.Allocator,
    stdout: anytype,
    stderr: anytype,
    db_path: []const u8,
    arg_parser: anytype,
) !void {
    // Parse server configuration
    const host = arg_parser.getString("host", "127.0.0.1");
    const port_str = arg_parser.getString("port", "5433");
    const max_conn_str = arg_parser.getString("max-connections", "100");

    const port = std.fmt.parseInt(u16, port_str, 10) catch {
        printError(stderr, "Invalid port number");
        stderr.flush() catch {};
        std.process.exit(1);
    };

    const max_connections = std.fmt.parseInt(usize, max_conn_str, 10) catch {
        printError(stderr, "Invalid max-connections value");
        stderr.flush() catch {};
        std.process.exit(1);
    };

    // Create server
    var server = server_mod.Server.init(allocator, .{
        .host = host,
        .port = port,
        .max_connections = max_connections,
        .database_path = db_path,
    }) catch {
        printError(stderr, "Failed to initialize server");
        stderr.flush() catch {};
        std.process.exit(1);
    };
    defer server.deinit();

    stdout.print("Silica v{s} — PostgreSQL wire protocol server\n", .{version}) catch {};
    stdout.print("Database: {s}\n", .{db_path}) catch {};
    stdout.print("Listening on {s}:{}\n", .{ host, port }) catch {};
    stdout.print("Max connections: {}\n", .{max_connections}) catch {};
    stdout.writeAll("Press Ctrl+C to stop.\n\n") catch {};
    stdout.flush() catch {};

    // Start server (blocks until shutdown)
    server.start() catch {
        printError(stderr, "Server error");
        stderr.flush() catch {};
        std.process.exit(1);
    };
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var stderr_buf: [4096]u8 = undefined;
    var stdout_buf: [4096]u8 = undefined;
    var stderr_writer = std.fs.File.stderr().writer(&stderr_buf);
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buf);
    const stderr = &stderr_writer.interface;
    const stdout = &stdout_writer.interface;

    // Skip argv[0] (program name)
    const all_args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, all_args);
    const args = if (all_args.len > 0) all_args[1..] else all_args[0..0];

    var arg_parser = sailor.arg.Parser(&CliFlags).init(allocator);
    defer arg_parser.deinit();

    arg_parser.parse(args) catch |err| {
        printError(stderr, switch (err) {
            error.UnknownFlag => "Unknown flag. Use --help for usage information.",
            error.MissingValue => "Flag requires a value.",
            error.MissingRequiredFlag => "Required flag missing.",
            error.InvalidValue => "Invalid flag value.",
            else => "Failed to parse arguments.",
        });
        stderr.flush() catch {};
        std.process.exit(1);
    };

    if (arg_parser.getBool("help", false)) {
        printUsage(stdout);
        stdout.flush() catch {};
        return;
    }

    if (arg_parser.getBool("version", false)) {
        stdout.print("silica {s}\n", .{version}) catch {};
        stdout.flush() catch {};
        return;
    }

    // First positional argument is either "server" or database path
    if (arg_parser.positional.items.len == 0) {
        printUsage(stdout);
        stdout.flush() catch {};
        return;
    }

    const first_arg = arg_parser.positional.items[0];

    // Check if running in server mode
    if (std.mem.eql(u8, first_arg, "server")) {
        if (arg_parser.positional.items.len < 2) {
            printError(stderr, "Server mode requires database path: silica server <database>");
            stderr.flush() catch {};
            std.process.exit(1);
        }
        const db_path = arg_parser.positional.items[1];
        return runServer(allocator, stdout, stderr, db_path, &arg_parser);
    }

    const db_path = first_arg;

    // Determine initial output mode from flags
    var mode: OutputMode = .table;
    const mode_str = arg_parser.getString("mode", "");
    if (mode_str.len > 0) {
        mode = parseModeString(mode_str) orelse {
            printError(stderr, "Invalid mode. Use: table, csv, json, jsonl, plain");
            stderr.flush() catch {};
            std.process.exit(1);
        };
    } else if (arg_parser.getBool("csv", false)) {
        mode = .csv;
    } else if (arg_parser.getBool("json", false)) {
        mode = .json;
    }

    // Open database
    var db = Database.open(allocator, db_path, .{}) catch {
        printError(stderr, "Failed to open database.");
        stderr.flush() catch {};
        std.process.exit(1);
    };
    defer db.close();

    // Launch TUI mode if --tui flag is set
    if (arg_parser.getBool("tui", false)) {
        tui_mod.run(allocator, &db, db_path) catch {
            printError(stderr, "TUI initialization failed.");
            stderr.flush() catch {};
            std.process.exit(1);
        };
        return;
    }

    // Print banner
    stdout.print("Silica v{s} — interactive SQL shell\n", .{version}) catch {};
    stdout.print("Database: {s}\n", .{db_path}) catch {};
    stdout.writeAll("Type .help for usage hints, .quit to exit.\n\n") catch {};
    stdout.flush() catch {};

    // Simple stdin-based REPL loop
    // NOTE: sailor.repl has Zig 0.15 compat bugs (https://github.com/yusa-imit/sailor/issues/1)
    const stdin = std.fs.File.stdin();
    var stdin_buf: [8192]u8 = undefined;
    var reader = stdin.reader(&stdin_buf);

    var input_buf = std.ArrayListUnmanaged(u8){};
    defer input_buf.deinit(allocator);

    var is_continuation = false;
    var show_timer = true; // Default: show query execution time

    while (true) {
        const prompt = if (is_continuation) "   ...> " else "silica> ";
        stdout.writeAll(prompt) catch {};
        stdout.flush() catch {};

        const line = reader.interface.takeDelimiter('\n') catch {
            printError(stderr, "Failed to read input.");
            stderr.flush() catch {};
            continue;
        };

        if (line == null) {
            stdout.writeAll("\nBye!\n") catch {};
            stdout.flush() catch {};
            break;
        }

        const trimmed = std.mem.trim(u8, line.?, " \t\r\n");
        if (trimmed.len == 0) continue;

        // Handle dot-commands (only on first line, not continuation)
        if (!is_continuation and trimmed[0] == '.') {
            const result = handleDotCommand(allocator, &db, trimmed, &mode, &show_timer, stdout, stderr);
            stdout.flush() catch {};
            stderr.flush() catch {};
            if (result == .quit) break;
            continue;
        }

        // Accumulate multi-line input
        if (input_buf.items.len > 0) {
            input_buf.append(allocator, '\n') catch continue;
        }
        input_buf.appendSlice(allocator, trimmed) catch continue;

        // Check if statement is complete (ends with ';')
        const accumulated = std.mem.trimRight(u8, input_buf.items, " \t\r\n");
        if (accumulated.len == 0) continue;

        if (accumulated[accumulated.len - 1] != ';') {
            is_continuation = true;
            continue;
        }

        // Execute SQL via engine
        execAndDisplay(allocator, &db, input_buf.items, mode, show_timer, stdout, stderr);
        stdout.flush() catch {};
        stderr.flush() catch {};
        input_buf.clearRetainingCapacity();
        is_continuation = false;
    }
}

// ── SQL Execution ──────────────────────────────────────────────────────

/// Execute SQL via the Database engine and display results.
fn execAndDisplay(allocator: std.mem.Allocator, db: *Database, sql: []const u8, mode: OutputMode, show_timer: bool, stdout: anytype, stderr: anytype) void {
    // Start timer for query execution (if enabled)
    var timer = if (show_timer) std.time.Timer.start() catch {
        // If timer fails, continue without timing
        execAndDisplayWithoutTiming(allocator, db, sql, mode, stdout, stderr);
        return;
    } else null;

    var result = db.exec(sql) catch |err| {
        const msg = switch (err) {
            error.ParseError => "SQL parse error.",
            error.AnalysisError => "Semantic analysis error.",
            error.PlanError => "Query planning error.",
            error.ExecutionError => "Execution error.",
            error.TableNotFound => "Table not found.",
            error.TableAlreadyExists => "Table already exists.",
            error.InvalidData => "Invalid data.",
            else => "Database error.",
        };
        printError(stderr, msg);
        return;
    };
    defer result.close(allocator);

    if (result.rows != null) {
        // SELECT — format rows
        displayRows(allocator, &result, mode, stdout, stderr);
    } else if (result.message.len > 0) {
        stdout.writeAll(result.message) catch {};
        stdout.writeByte('\n') catch {};
    }

    if (result.rows_affected > 0) {
        stdout.print("Rows affected: {d}\n", .{result.rows_affected}) catch {};
    }

    // Display query execution time (if timer enabled)
    if (timer) |*t| {
        const elapsed_ns = t.read();
        const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
        stdout.print("Query time: {d:.3} ms\n", .{elapsed_ms}) catch {};
    }
}

/// Fallback for when timer is unavailable (executes without timing).
fn execAndDisplayWithoutTiming(allocator: std.mem.Allocator, db: *Database, sql: []const u8, mode: OutputMode, stdout: anytype, stderr: anytype) void {
    var result = db.exec(sql) catch |err| {
        const msg = switch (err) {
            error.ParseError => "SQL parse error.",
            error.AnalysisError => "Semantic analysis error.",
            error.PlanError => "Query planning error.",
            error.ExecutionError => "Execution error.",
            error.TableNotFound => "Table not found.",
            error.TableAlreadyExists => "Table already exists.",
            error.InvalidData => "Invalid data.",
            else => "Database error.",
        };
        printError(stderr, msg);
        return;
    };
    defer result.close(allocator);

    if (result.rows != null) {
        // SELECT — format rows
        displayRows(allocator, &result, mode, stdout, stderr);
    } else if (result.message.len > 0) {
        stdout.writeAll(result.message) catch {};
        stdout.writeByte('\n') catch {};
    }

    if (result.rows_affected > 0) {
        stdout.print("Rows affected: {d}\n", .{result.rows_affected}) catch {};
    }
}

/// Drain all rows from a QueryResult and display them in the given output mode.
fn displayRows(allocator: std.mem.Allocator, result: *QueryResult, mode: OutputMode, stdout: anytype, stderr: anytype) void {
    _ = stderr;

    // Collect all rows (we need them all for table mode column widths)
    var all_rows = std.ArrayListUnmanaged(Row){};
    defer {
        for (all_rows.items) |*row| row.deinit();
        all_rows.deinit(allocator);
    }

    var col_names: ?[]const []const u8 = null;
    var row_count: usize = 0;

    while (true) {
        const maybe_row = result.rows.?.next() catch break;
        if (maybe_row) |row| {
            if (col_names == null and row.columns.len > 0) {
                // Copy column names from first row
                const names = allocator.alloc([]const u8, row.columns.len) catch break;
                for (row.columns, 0..) |col, i| {
                    names[i] = allocator.dupe(u8, col) catch break;
                }
                col_names = names;
            }
            all_rows.append(allocator, row) catch break;
            row_count += 1;
        } else break;
    }

    defer {
        if (col_names) |names| {
            for (names) |n| allocator.free(n);
            allocator.free(names);
        }
    }

    if (col_names == null) return;
    const headers = col_names.?;

    switch (mode) {
        .table => formatTable(allocator, headers, all_rows.items, stdout),
        .csv => formatCsv(headers, all_rows.items, allocator, stdout),
        .json => formatJson(headers, all_rows.items, allocator, stdout),
        .jsonl => formatJsonl(headers, all_rows.items, allocator, stdout),
        .plain => formatPlain(headers, all_rows.items, allocator, stdout),
    }
}

// ── Value Formatting ───────────────────────────────────────────────────

/// Convert a Value to its display string. Caller owns returned memory.
fn valueToText(allocator: std.mem.Allocator, val: Value) ?[]const u8 {
    return switch (val) {
        .integer => |v| std.fmt.allocPrint(allocator, "{d}", .{v}) catch null,
        .real => |v| std.fmt.allocPrint(allocator, "{d}", .{v}) catch null,
        .text => |v| allocator.dupe(u8, v) catch null,
        .blob => |v| blk: {
            // Format blob as hex string
            const hex_len = 2 + v.len * 2 + 1; // X' + hex + '
            const hex_buf = allocator.alloc(u8, hex_len) catch break :blk null;
            hex_buf[0] = 'X';
            hex_buf[1] = '\'';
            for (v, 0..) |byte, i| {
                const digits = "0123456789abcdef";
                hex_buf[2 + i * 2] = digits[byte >> 4];
                hex_buf[2 + i * 2 + 1] = digits[byte & 0x0f];
            }
            hex_buf[hex_len - 1] = '\'';
            break :blk hex_buf;
        },
        .boolean => |v| allocator.dupe(u8, if (v) "TRUE" else "FALSE") catch null,
        .date => |v| executor.formatDate(allocator, v) catch null,
        .time => |v| executor.formatTime(allocator, v) catch null,
        .timestamp => |v| executor.formatTimestamp(allocator, v) catch null,
        .interval => |v| executor.formatInterval(allocator, v) catch null,
        .numeric => |v| executor.formatNumeric(allocator, v) catch null,
        .uuid => |v| executor.formatUuid(allocator, v) catch null,
        .array => |v| executor.formatArray(allocator, v) catch null,
        .tsvector => |v| allocator.dupe(u8, v) catch null,
        .tsquery => |v| allocator.dupe(u8, v) catch null,
        .null_value => allocator.dupe(u8, "NULL") catch null,
    };
}

// ── Table Format ───────────────────────────────────────────────────────

fn formatTable(allocator: std.mem.Allocator, headers: []const []const u8, rows: []Row, writer: anytype) void {
    var table = sailor.fmt.Table.init(allocator, headers, .{}) catch return;
    defer table.deinit();

    // Convert all rows to string slices
    var str_rows = std.ArrayListUnmanaged([]const []const u8){};
    defer {
        for (str_rows.items) |row| {
            for (row) |cell| allocator.free(cell);
            allocator.free(row);
        }
        str_rows.deinit(allocator);
    }

    for (rows) |row| {
        const cells = allocator.alloc([]const u8, row.values.len) catch continue;
        var valid = true;
        for (row.values, 0..) |val, i| {
            cells[i] = valueToText(allocator, val) orelse {
                valid = false;
                // Free already allocated cells
                for (cells[0..i]) |c| allocator.free(c);
                break;
            };
        }
        if (!valid) {
            allocator.free(cells);
            continue;
        }
        table.addRow(cells) catch {
            for (cells) |c| allocator.free(c);
            allocator.free(cells);
            continue;
        };
        str_rows.append(allocator, cells) catch {
            for (cells) |c| allocator.free(c);
            allocator.free(cells);
            continue;
        };
    }

    table.render(writer) catch {};
}

// ── CSV Format ────────────────────────────────────────────────────────

fn formatCsv(headers: []const []const u8, rows: []Row, allocator: std.mem.Allocator, writer: anytype) void {
    const WriterType = @TypeOf(writer);
    var csv = sailor.fmt.Csv(WriterType).init(writer, .{});

    // Write headers
    for (headers) |h| {
        csv.writeField(h) catch return;
    }
    csv.endRow() catch return;

    // Write rows
    for (rows) |row| {
        for (row.values) |val| {
            const text = valueToText(allocator, val) orelse "NULL";
            defer if (text.ptr != "NULL".ptr) allocator.free(text);
            csv.writeField(text) catch return;
        }
        csv.endRow() catch return;
    }
}

// ── JSON Format ────────────────────────────────────────────────────────

fn formatJson(headers: []const []const u8, rows: []Row, allocator: std.mem.Allocator, writer: anytype) void {
    const WriterType = @TypeOf(writer);
    var arr = sailor.fmt.JsonArray(WriterType).init(writer) catch return;

    for (rows) |row| {
        var obj = arr.beginObject() catch return;
        for (headers, 0..) |h, i| {
            if (i < row.values.len) {
                writeJsonValue(&obj, h, row.values[i], allocator);
            }
        }
        obj.end() catch return;
    }

    arr.end() catch return;
    writer.writeByte('\n') catch {};
}

fn formatJsonl(headers: []const []const u8, rows: []Row, allocator: std.mem.Allocator, writer: anytype) void {
    const WriterType = @TypeOf(writer);
    for (rows) |row| {
        var obj = sailor.fmt.JsonObject(WriterType).init(writer) catch return;
        for (headers, 0..) |h, i| {
            if (i < row.values.len) {
                writeJsonValue(&obj, h, row.values[i], allocator);
            }
        }
        obj.end() catch return;
        writer.writeByte('\n') catch {};
    }
}

fn writeJsonValue(obj: anytype, key: []const u8, val: Value, allocator: std.mem.Allocator) void {
    switch (val) {
        .integer => |v| obj.addNumber(key, v) catch {},
        .real => |v| obj.addNumber(key, v) catch {},
        .text => |v| obj.addString(key, v) catch {},
        .boolean => |v| obj.addBool(key, v) catch {},
        .null_value => obj.addNull(key) catch {},
        .date => |v| {
            const s = executor.formatDate(allocator, v) catch return;
            defer allocator.free(s);
            obj.addString(key, s) catch {};
        },
        .time => |v| {
            const s = executor.formatTime(allocator, v) catch return;
            defer allocator.free(s);
            obj.addString(key, s) catch {};
        },
        .timestamp => |v| {
            const s = executor.formatTimestamp(allocator, v) catch return;
            defer allocator.free(s);
            obj.addString(key, s) catch {};
        },
        .interval => |v| {
            const s = executor.formatInterval(allocator, v) catch return;
            defer allocator.free(s);
            obj.addString(key, s) catch {};
        },
        .numeric => |v| {
            const s = executor.formatNumeric(allocator, v) catch return;
            defer allocator.free(s);
            obj.addString(key, s) catch {};
        },
        .uuid => |v| {
            const s = executor.formatUuid(allocator, v) catch return;
            defer allocator.free(s);
            obj.addString(key, s) catch {};
        },
        .array => |v| {
            const s = executor.formatArray(allocator, v) catch return;
            defer allocator.free(s);
            obj.addString(key, s) catch {};
        },
        .tsvector => |v| obj.addString(key, v) catch {},
        .tsquery => |v| obj.addString(key, v) catch {},
        .blob => |v| {
            // Format blob as hex string
            const hex = allocator.alloc(u8, v.len * 2) catch return;
            defer allocator.free(hex);
            const digits = "0123456789abcdef";
            for (v, 0..) |byte, i| {
                hex[i * 2] = digits[byte >> 4];
                hex[i * 2 + 1] = digits[byte & 0x0f];
            }
            obj.addString(key, hex) catch {};
        },
    }
}

// ── Plain Format ──────────────────────────────────────────────────────

fn formatPlain(headers: []const []const u8, rows: []Row, allocator: std.mem.Allocator, writer: anytype) void {
    for (rows) |row| {
        for (headers, 0..) |h, i| {
            if (i < row.values.len) {
                const text = valueToText(allocator, row.values[i]) orelse "NULL";
                defer if (text.ptr != "NULL".ptr) allocator.free(text);
                writer.print("{s} = {s}\n", .{ h, text }) catch {};
            }
        }
        if (rows.len > 1) writer.writeByte('\n') catch {};
    }
}

// ── Parse Mode String ──────────────────────────────────────────────────

fn parseModeString(s: []const u8) ?OutputMode {
    if (std.mem.eql(u8, s, "table")) return .table;
    if (std.mem.eql(u8, s, "csv")) return .csv;
    if (std.mem.eql(u8, s, "json")) return .json;
    if (std.mem.eql(u8, s, "jsonl")) return .jsonl;
    if (std.mem.eql(u8, s, "plain")) return .plain;
    return null;
}

// ── Process SQL (parse-only, for testing without a DB) ──────────────

fn processSQL(allocator: std.mem.Allocator, sql: []const u8, stdout: anytype, stderr: anytype) void {
    var arena = silica.ast.AstArena.init(allocator);
    defer arena.deinit();

    var sql_parser = silica.parser.Parser.init(allocator, sql, &arena) catch {
        printError(stderr, "Failed to initialize parser.");
        return;
    };
    defer sql_parser.deinit();

    var stmt_count: usize = 0;
    while (true) {
        const stmt = sql_parser.parseStatement() catch {
            for (sql_parser.errors.items) |err| {
                printSQLError(stderr, sql, err);
            }
            return;
        };

        if (stmt == null) break;

        stmt_count += 1;
        printStmtInfo(stdout, stmt.?);
    }

    if (stmt_count == 0) {
        printError(stderr, "No SQL statement found.");
    }
}

/// Print info about a parsed statement.
fn printStmtInfo(writer: anytype, stmt: silica.ast.Stmt) void {
    switch (stmt) {
        .select => |s| {
            writer.print("Parsed: SELECT ({d} columns", .{s.columns.len}) catch {};
            if (s.from != null) writer.writeAll(", FROM") catch {};
            if (s.where != null) writer.writeAll(", WHERE") catch {};
            if (s.joins.len > 0) writer.print(", {d} JOINs", .{s.joins.len}) catch {};
            if (s.group_by.len > 0) writer.writeAll(", GROUP BY") catch {};
            if (s.order_by.len > 0) writer.writeAll(", ORDER BY") catch {};
            if (s.limit != null) writer.writeAll(", LIMIT") catch {};
            writer.writeAll(")\n") catch {};
        },
        .insert => |s| {
            writer.print("Parsed: INSERT INTO {s} ({d} row(s))\n", .{ s.table, s.values.len }) catch {};
        },
        .update => |s| {
            writer.print("Parsed: UPDATE {s} ({d} assignment(s))\n", .{ s.table, s.assignments.len }) catch {};
        },
        .delete => |s| {
            writer.print("Parsed: DELETE FROM {s}\n", .{s.table}) catch {};
        },
        .create_table => |s| {
            writer.print("Parsed: CREATE TABLE {s} ({d} columns)\n", .{ s.name, s.columns.len }) catch {};
        },
        .drop_table => |s| {
            writer.print("Parsed: DROP TABLE {s}\n", .{s.name}) catch {};
        },
        .create_index => |s| {
            writer.print("Parsed: CREATE INDEX {s} ON {s}\n", .{ s.name, s.table }) catch {};
        },
        .drop_index => |s| {
            writer.print("Parsed: DROP INDEX {s}\n", .{s.name}) catch {};
        },
        .transaction => |t| {
            switch (t) {
                .begin => writer.writeAll("Parsed: BEGIN\n") catch {},
                .commit => writer.writeAll("Parsed: COMMIT\n") catch {},
                .rollback => writer.writeAll("Parsed: ROLLBACK\n") catch {},
                .savepoint => |n| writer.print("Parsed: SAVEPOINT {s}\n", .{n}) catch {},
                .release => |n| writer.print("Parsed: RELEASE {s}\n", .{n}) catch {},
            }
        },
        .explain => writer.writeAll("Parsed: EXPLAIN\n") catch {},
        .vacuum => |v| {
            if (v.table_name) |name| {
                writer.print("Parsed: VACUUM {s}\n", .{name}) catch {};
            } else {
                writer.writeAll("Parsed: VACUUM\n") catch {};
            }
        },
        .analyze => |a| {
            if (a.table_name) |name| {
                writer.print("Parsed: ANALYZE {s}\n", .{name}) catch {};
            } else {
                writer.writeAll("Parsed: ANALYZE\n") catch {};
            }
        },
        .create_view => |v| {
            writer.print("Parsed: CREATE VIEW {s}\n", .{v.name}) catch {};
        },
        .drop_view => |v| {
            writer.print("Parsed: DROP VIEW {s}\n", .{v.name}) catch {};
        },
        .create_type => |t| {
            writer.print("Parsed: CREATE TYPE {s} AS ENUM ({d} values)\n", .{ t.name, t.values.len }) catch {};
        },
        .drop_type => |t| {
            writer.print("Parsed: DROP TYPE {s}\n", .{t.name}) catch {};
        },
        .create_domain => |d| {
            writer.print("Parsed: CREATE DOMAIN {s} AS {any}\n", .{ d.name, d.base_type }) catch {};
        },
        .drop_domain => |d| {
            writer.print("Parsed: DROP DOMAIN {s}\n", .{d.name}) catch {};
        },
        .create_function => |f| {
            writer.print("Parsed: CREATE FUNCTION {s} ({d} params)\n", .{ f.name, f.parameters.len }) catch {};
        },
        .drop_function => |f| {
            writer.print("Parsed: DROP FUNCTION {s}\n", .{f.name}) catch {};
        },
        .create_trigger => |t| {
            writer.print("Parsed: CREATE TRIGGER {s} ON {s}\n", .{ t.name, t.table_name }) catch {};
        },
        .drop_trigger => |t| {
            writer.print("Parsed: DROP TRIGGER {s}\n", .{t.name}) catch {};
        },
        .alter_trigger => |t| {
            writer.print("Parsed: ALTER TRIGGER {s}\n", .{t.name}) catch {};
        },
        .create_role => |r| {
            writer.print("Parsed: CREATE ROLE {s}\n", .{r.name}) catch {};
        },
        .drop_role => |r| {
            writer.print("Parsed: DROP ROLE {s}\n", .{r.name}) catch {};
        },
        .alter_role => |r| {
            writer.print("Parsed: ALTER ROLE {s}\n", .{r.name}) catch {};
        },
        .grant => |g| {
            writer.print("Parsed: GRANT {d} privilege(s) ON {s} TO {s}\n", .{ g.privileges.len, g.object_name, g.grantee }) catch {};
        },
        .revoke => |r| {
            writer.print("Parsed: REVOKE {d} privilege(s) ON {s} FROM {s}\n", .{ r.privileges.len, r.object_name, r.grantee }) catch {};
        },
        .grant_role => |g| {
            writer.print("Parsed: GRANT {s} TO {d} member(s)\n", .{ g.role, g.members.len }) catch {};
        },
        .revoke_role => |r| {
            writer.print("Parsed: REVOKE {s} FROM {d} member(s)\n", .{ r.role, r.members.len }) catch {};
        },
        .create_policy => |p| {
            writer.print("Parsed: CREATE POLICY {s} ON {s}\n", .{ p.policy_name, p.table_name }) catch {};
        },
        .drop_policy => |p| {
            writer.print("Parsed: DROP POLICY {s} ON {s}\n", .{ p.policy_name, p.table_name }) catch {};
        },
        .alter_table_rls => |a| {
            writer.print("Parsed: ALTER TABLE {s} (RLS)\n", .{a.table_name}) catch {};
        },
        .reindex => |r| {
            switch (r) {
                .index => |idx| writer.print("Parsed: REINDEX INDEX {s}\n", .{idx}) catch {},
                .table => |tbl| writer.print("Parsed: REINDEX TABLE {s}\n", .{tbl}) catch {},
                .database => writer.writeAll("Parsed: REINDEX DATABASE\n") catch {},
            }
        },
        .set => |s| {
            writer.print("Parsed: SET {s} = {s}\n", .{ s.parameter, s.value }) catch {};
        },
        .show => |s| {
            if (s.parameter) |p| {
                writer.print("Parsed: SHOW {s}\n", .{p}) catch {};
            } else {
                writer.writeAll("Parsed: SHOW ALL\n") catch {};
            }
        },
        .reset => |r| {
            if (r.parameter) |p| {
                writer.print("Parsed: RESET {s}\n", .{p}) catch {};
            } else {
                writer.writeAll("Parsed: RESET ALL\n") catch {};
            }
        },
    }
}

// ── Error Formatting ───────────────────────────────────────────────────

fn printSQLError(writer: anytype, sql: []const u8, err: silica.parser.ParseError) void {
    var line_num: usize = 1;
    var col_num: usize = 1;
    var line_start: usize = 0;
    for (sql[0..@min(err.token.start, sql.len)], 0..) |c, i| {
        if (c == '\n') {
            line_num += 1;
            col_num = 1;
            line_start = i + 1;
        } else {
            col_num += 1;
        }
    }

    var line_end = err.token.start;
    while (line_end < sql.len and sql[line_end] != '\n') : (line_end += 1) {}

    sailor.color.writeStyled(writer, sailor.color.semantic.err, "error") catch {};
    writer.print(" (line {d}, col {d}): {s}\n", .{ line_num, col_num, err.message }) catch {};

    if (line_start < sql.len) {
        writer.writeAll("  ") catch {};
        writer.writeAll(sql[line_start..line_end]) catch {};
        writer.writeAll("\n") catch {};

        writer.writeAll("  ") catch {};
        for (0..col_num - 1) |_| {
            writer.writeAll(" ") catch {};
        }
        sailor.color.writeStyled(writer, sailor.color.semantic.err, "^") catch {};
        writer.writeAll("\n") catch {};
    }
}

// ── Dot Commands ───────────────────────────────────────────────────────

/// List all tables in the database
/// List all open database connections
fn showDatabases(db: *Database, stdout: anytype) void {
    // Silica currently supports only one database connection at a time
    // Display the current database path in SQLite-compatible format
    stdout.writeAll("seq  name             file\n") catch {};
    stdout.writeAll("---  ---------------  ") catch {};
    stdout.writeAll("------------------------------------------------------------\n") catch {};
    stdout.print("0    main             {s}\n", .{db.db_path}) catch {};
}

fn listTables(allocator: std.mem.Allocator, db: *Database, stdout: anytype, stderr: anytype) void {
    // Use catalog API to get table names directly
    var catalog = &db.catalog;
    const names = catalog.listTables(allocator) catch |err| {
        const msg = switch (err) {
            error.OutOfMemory => "Out of memory.",
            else => "Failed to list tables.",
        };
        printError(stderr, msg);
        return;
    };
    defer {
        for (names) |name| allocator.free(name);
        allocator.free(names);
    }

    if (names.len == 0) {
        stdout.writeAll("No tables found.\n") catch {};
        return;
    }

    // Sort table names alphabetically
    std.mem.sort([]const u8, names, {}, struct {
        fn lessThan(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.order(u8, a, b) == .lt;
        }
    }.lessThan);

    for (names) |name| {
        stdout.print("{s}\n", .{name}) catch {};
    }
}

/// Show indexes for all tables or a specific table
fn showIndexes(allocator: std.mem.Allocator, db: *Database, table_name: ?[]const u8, stdout: anytype, stderr: anytype) void {
    var catalog = &db.catalog;

    if (table_name) |name| {
        // Show indexes for specific table
        const trimmed = std.mem.trim(u8, name, " \t");
        if (trimmed.len == 0) {
            printError(stderr, "Table name cannot be empty.");
            return;
        }

        const table_info = catalog.getTable(trimmed) catch |err| {
            const msg = switch (err) {
                error.TableNotFound => "Table not found.",
                error.OutOfMemory => "Out of memory.",
                else => "Failed to retrieve table info.",
            };
            printError(stderr, msg);
            return;
        };
        defer table_info.deinit(catalog.allocator);

        if (table_info.indexes.len == 0) {
            stdout.print("No indexes found for table '{s}'.\n", .{trimmed}) catch {};
            return;
        }

        for (table_info.indexes) |idx| {
            const index_type = @tagName(idx.index_type);
            const unique_str = if (idx.is_unique) " UNIQUE" else "";
            const state_str = switch (idx.state) {
                .valid => "",
                .building => " (BUILDING)",
                .invalid => " (INVALID)",
            };

            if (idx.index_name.len > 0) {
                stdout.print("{s} ({s}{s} on {s}.{s}){s}\n", .{
                    idx.index_name,
                    index_type,
                    unique_str,
                    trimmed,
                    idx.column_name,
                    state_str,
                }) catch {};
            } else {
                // Auto-generated index (e.g., from PRIMARY KEY or UNIQUE constraint)
                stdout.print("(auto) ({s}{s} on {s}.{s}){s}\n", .{
                    index_type,
                    unique_str,
                    trimmed,
                    idx.column_name,
                    state_str,
                }) catch {};
            }
        }
    } else {
        // Show indexes for all tables
        const names = catalog.listTables(allocator) catch |err| {
            const msg = switch (err) {
                error.OutOfMemory => "Out of memory.",
                else => "Failed to list tables.",
            };
            printError(stderr, msg);
            return;
        };
        defer {
            for (names) |n| allocator.free(n);
            allocator.free(names);
        }

        if (names.len == 0) {
            stdout.writeAll("No tables found.\n") catch {};
            return;
        }

        // Sort table names alphabetically
        std.mem.sort([]const u8, names, {}, struct {
            fn lessThan(_: void, a: []const u8, b: []const u8) bool {
                return std.mem.order(u8, a, b) == .lt;
            }
        }.lessThan);

        var has_indexes = false;
        for (names) |name| {
            const table_info = catalog.getTable(name) catch continue;
            defer table_info.deinit(catalog.allocator);

            if (table_info.indexes.len == 0) continue;

            has_indexes = true;
            for (table_info.indexes) |idx| {
                const index_type = @tagName(idx.index_type);
                const unique_str = if (idx.is_unique) " UNIQUE" else "";
                const state_str = switch (idx.state) {
                    .valid => "",
                    .building => " (BUILDING)",
                    .invalid => " (INVALID)",
                };

                if (idx.index_name.len > 0) {
                    stdout.print("{s} ({s}{s} on {s}.{s}){s}\n", .{
                        idx.index_name,
                        index_type,
                        unique_str,
                        name,
                        idx.column_name,
                        state_str,
                    }) catch {};
                } else {
                    // Auto-generated index
                    stdout.print("(auto) ({s}{s} on {s}.{s}){s}\n", .{
                        index_type,
                        unique_str,
                        name,
                        idx.column_name,
                        state_str,
                    }) catch {};
                }
            }
        }

        if (!has_indexes) {
            stdout.writeAll("No indexes found.\n") catch {};
        }
    }
}

/// Show schema (CREATE TABLE statements) for all tables or a specific table
fn showSchema(allocator: std.mem.Allocator, db: *Database, table_name: ?[]const u8, stdout: anytype, stderr: anytype) void {
    var catalog = &db.catalog;

    if (table_name) |name| {
        // Show schema for specific table
        const trimmed = std.mem.trim(u8, name, " \t");
        if (trimmed.len == 0) {
            printError(stderr, "Table name cannot be empty.");
            return;
        }
        showTableSchema(allocator, catalog, trimmed, stdout, stderr);
    } else {
        // Show schema for all tables
        const names = catalog.listTables(allocator) catch |err| {
            const msg = switch (err) {
                error.OutOfMemory => "Out of memory.",
                else => "Failed to list tables.",
            };
            printError(stderr, msg);
            return;
        };
        defer {
            for (names) |name| allocator.free(name);
            allocator.free(names);
        }

        if (names.len == 0) {
            stdout.writeAll("No tables found.\n") catch {};
            return;
        }

        // Sort table names alphabetically
        std.mem.sort([]const u8, names, {}, struct {
            fn lessThan(_: void, a: []const u8, b: []const u8) bool {
                return std.mem.order(u8, a, b) == .lt;
            }
        }.lessThan);

        for (names, 0..) |name, i| {
            if (i > 0) stdout.writeAll("\n") catch {};
            showTableSchema(allocator, catalog, name, stdout, stderr);
        }
    }
}

/// Helper to show schema for a single table
fn showTableSchema(allocator: std.mem.Allocator, catalog: *silica.catalog.Catalog, table_name: []const u8, stdout: anytype, stderr: anytype) void {
    const table = catalog.getTable(table_name) catch |err| {
        const msg = switch (err) {
            error.TableNotFound => "Table not found.",
            error.OutOfMemory => "Out of memory.",
            else => "Failed to get table schema.",
        };
        printError(stderr, msg);
        return;
    };
    defer table.deinit(allocator);

    // Generate CREATE TABLE statement
    stdout.print("CREATE TABLE {s} (\n", .{table_name}) catch {};

    // Columns
    for (table.columns, 0..) |col, i| {
        stdout.writeAll("  ") catch {};
        stdout.print("{s} ", .{col.name}) catch {};

        // Column type
        const type_name = switch (col.column_type) {
            .integer => "INTEGER",
            .real => "REAL",
            .text => "TEXT",
            .blob => "BLOB",
            .boolean => "BOOLEAN",
            .date => "DATE",
            .time => "TIME",
            .timestamp => "TIMESTAMP",
            .interval => "INTERVAL",
            .numeric => "NUMERIC",
            .uuid => "UUID",
            .array => "ARRAY",
            .json => "JSON",
            .jsonb => "JSONB",
            .tsvector => "TSVECTOR",
            .tsquery => "TSQUERY",
            .untyped => "",
        };
        if (type_name.len > 0) {
            stdout.writeAll(type_name) catch {};
        }

        // Column constraints
        if (col.flags.primary_key) {
            stdout.writeAll(" PRIMARY KEY") catch {};
        }
        if (col.flags.not_null) {
            stdout.writeAll(" NOT NULL") catch {};
        }
        if (col.flags.unique) {
            stdout.writeAll(" UNIQUE") catch {};
        }
        if (col.flags.autoincrement) {
            stdout.writeAll(" AUTOINCREMENT") catch {};
        }

        // Comma for all but last column (unless there are table constraints)
        const is_last_column = (i == table.columns.len - 1);
        const has_table_constraints = table.table_constraints.len > 0;
        if (!is_last_column or has_table_constraints) {
            stdout.writeAll(",") catch {};
        }
        stdout.writeAll("\n") catch {};
    }

    // Table-level constraints
    for (table.table_constraints, 0..) |constraint, i| {
        stdout.writeAll("  ") catch {};
        switch (constraint) {
            .primary_key => |cols| {
                stdout.writeAll("PRIMARY KEY (") catch {};
                for (cols, 0..) |col_name, j| {
                    stdout.writeAll(col_name) catch {};
                    if (j < cols.len - 1) stdout.writeAll(", ") catch {};
                }
                stdout.writeAll(")") catch {};
            },
            .unique => |cols| {
                stdout.writeAll("UNIQUE (") catch {};
                for (cols, 0..) |col_name, j| {
                    stdout.writeAll(col_name) catch {};
                    if (j < cols.len - 1) stdout.writeAll(", ") catch {};
                }
                stdout.writeAll(")") catch {};
            },
        }
        if (i < table.table_constraints.len - 1) {
            stdout.writeAll(",") catch {};
        }
        stdout.writeAll("\n") catch {};
    }

    stdout.writeAll(");\n") catch {};

    // Show indexes if any (skip auto-generated indexes with empty names)
    if (table.indexes.len > 0) {
        for (table.indexes) |idx| {
            // Skip auto-generated indexes (empty name)
            if (idx.index_name.len == 0) continue;

            const index_type = switch (idx.index_type) {
                .btree => "BTREE",
                .hash => "HASH",
                .gist => "GIST",
                .gin => "GIN",
            };
            const unique_str = if (idx.is_unique) "UNIQUE " else "";
            stdout.print("CREATE {s}INDEX {s} ON {s} USING {s} ({s});\n", .{
                unique_str,
                idx.index_name,
                table_name,
                index_type,
                idx.column_name,
            }) catch {};
        }
    }
}

/// Dump entire database as SQL text (CREATE TABLE + INSERT statements)
fn dumpDatabase(allocator: std.mem.Allocator, db: *Database, stdout: anytype, stderr: anytype) void {
    var catalog = &db.catalog;

    // Get all table names
    const names = catalog.listTables(allocator) catch |err| {
        const msg = switch (err) {
            error.OutOfMemory => "Out of memory.",
            else => "Failed to list tables.",
        };
        printError(stderr, msg);
        return;
    };
    defer {
        for (names) |name| allocator.free(name);
        allocator.free(names);
    }

    if (names.len == 0) {
        stdout.writeAll("-- No tables found.\n") catch {};
        return;
    }

    // Sort table names alphabetically
    std.mem.sort([]const u8, names, {}, struct {
        fn lessThan(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.order(u8, a, b) == .lt;
        }
    }.lessThan);

    stdout.writeAll("-- Silica database dump\n") catch {};
    stdout.writeAll("BEGIN TRANSACTION;\n\n") catch {};

    // Dump each table
    for (names) |table_name| {
        dumpTable(allocator, db, catalog, table_name, stdout, stderr);
        stdout.writeAll("\n") catch {};
    }

    stdout.writeAll("COMMIT;\n") catch {};
}

/// Read and execute SQL statements from a file
fn readAndExecuteFile(allocator: std.mem.Allocator, db: *Database, filename: []const u8, mode: OutputMode, show_timer: bool, stdout: anytype, stderr: anytype) void {
    // Open file
    const file = std.fs.cwd().openFile(filename, .{}) catch |err| {
        const msg = switch (err) {
            error.FileNotFound => "File not found.",
            error.AccessDenied => "Access denied.",
            error.IsDir => "Path is a directory.",
            else => "Failed to open file.",
        };
        printError(stderr, msg);
        return;
    };
    defer file.close();

    // Read file contents
    const max_size = 100 * 1024 * 1024; // 100 MB limit
    const content = file.readToEndAlloc(allocator, max_size) catch |err| {
        const msg = switch (err) {
            error.OutOfMemory => "Out of memory.",
            error.FileTooBig => "File too large (max 100 MB).",
            else => "Failed to read file.",
        };
        printError(stderr, msg);
        return;
    };
    defer allocator.free(content);

    // Split content into statements by lines first to handle comments
    var statement_buf = std.ArrayListUnmanaged(u8){};
    defer statement_buf.deinit(allocator);
    var statement_count: usize = 0;

    var line_iter = std.mem.splitScalar(u8, content, '\n');
    while (line_iter.next()) |line| {
        const trimmed_line = std.mem.trim(u8, line, " \t\r");

        // Skip empty lines
        if (trimmed_line.len == 0) continue;

        // Skip comment-only lines
        if (std.mem.startsWith(u8, trimmed_line, "--")) continue;

        // Append line to statement buffer
        if (statement_buf.items.len > 0) {
            statement_buf.append(allocator, '\n') catch continue;
        }
        statement_buf.appendSlice(allocator, trimmed_line) catch continue;

        // Check if statement is complete (ends with semicolon)
        if (std.mem.endsWith(u8, trimmed_line, ";")) {
            const statement = statement_buf.items;
            if (statement.len > 0) {
                statement_count += 1;
                execAndDisplay(allocator, db, statement, mode, show_timer, stdout, stderr);
            }
            statement_buf.clearRetainingCapacity();
        }
    }

    // Check for unterminated statement
    if (statement_buf.items.len > 0) {
        const remaining = std.mem.trim(u8, statement_buf.items, " \t\r\n");
        if (remaining.len > 0) {
            printError(stderr, "Warning: Unterminated statement at end of file (missing ';')");
        }
    }

    stdout.print("Executed {} statement(s) from {s}\n", .{ statement_count, filename }) catch {};
}

/// Dump a single table (CREATE TABLE + CREATE INDEX + INSERT statements)
fn dumpTable(allocator: std.mem.Allocator, db: *Database, catalog: *silica.catalog.Catalog, table_name: []const u8, stdout: anytype, stderr: anytype) void {
    // Get table schema
    const table = catalog.getTable(table_name) catch |err| {
        const msg = switch (err) {
            error.TableNotFound => "Table not found.",
            error.OutOfMemory => "Out of memory.",
            else => "Failed to get table schema.",
        };
        printError(stderr, msg);
        return;
    };
    defer table.deinit(allocator);

    // Generate CREATE TABLE statement
    stdout.print("CREATE TABLE {s} (\n", .{table_name}) catch {};

    // Columns
    for (table.columns, 0..) |col, i| {
        stdout.writeAll("  ") catch {};
        stdout.print("{s} ", .{col.name}) catch {};

        // Column type
        const type_name = switch (col.column_type) {
            .integer => "INTEGER",
            .real => "REAL",
            .text => "TEXT",
            .blob => "BLOB",
            .boolean => "BOOLEAN",
            .date => "DATE",
            .time => "TIME",
            .timestamp => "TIMESTAMP",
            .interval => "INTERVAL",
            .numeric => "NUMERIC",
            .uuid => "UUID",
            .array => "ARRAY",
            .json => "JSON",
            .jsonb => "JSONB",
            .tsvector => "TSVECTOR",
            .tsquery => "TSQUERY",
            .untyped => "",
        };
        if (type_name.len > 0) {
            stdout.writeAll(type_name) catch {};
        }

        // Column constraints
        if (col.flags.primary_key) {
            stdout.writeAll(" PRIMARY KEY") catch {};
        }
        if (col.flags.not_null) {
            stdout.writeAll(" NOT NULL") catch {};
        }
        if (col.flags.unique) {
            stdout.writeAll(" UNIQUE") catch {};
        }
        if (col.flags.autoincrement) {
            stdout.writeAll(" AUTOINCREMENT") catch {};
        }

        // Comma for all but last column (unless there are table constraints)
        const is_last_column = (i == table.columns.len - 1);
        const has_table_constraints = table.table_constraints.len > 0;
        if (!is_last_column or has_table_constraints) {
            stdout.writeAll(",") catch {};
        }
        stdout.writeAll("\n") catch {};
    }

    // Table-level constraints
    for (table.table_constraints, 0..) |constraint, i| {
        stdout.writeAll("  ") catch {};
        switch (constraint) {
            .primary_key => |cols| {
                stdout.writeAll("PRIMARY KEY (") catch {};
                for (cols, 0..) |col_name, j| {
                    stdout.writeAll(col_name) catch {};
                    if (j < cols.len - 1) stdout.writeAll(", ") catch {};
                }
                stdout.writeAll(")") catch {};
            },
            .unique => |cols| {
                stdout.writeAll("UNIQUE (") catch {};
                for (cols, 0..) |col_name, j| {
                    stdout.writeAll(col_name) catch {};
                    if (j < cols.len - 1) stdout.writeAll(", ") catch {};
                }
                stdout.writeAll(")") catch {};
            },
        }
        if (i < table.table_constraints.len - 1) {
            stdout.writeAll(",") catch {};
        }
        stdout.writeAll("\n") catch {};
    }

    stdout.writeAll(");\n") catch {};

    // Show indexes (skip auto-generated indexes with empty names)
    if (table.indexes.len > 0) {
        for (table.indexes) |idx| {
            // Skip auto-generated indexes (empty name)
            if (idx.index_name.len == 0) continue;

            const index_type = switch (idx.index_type) {
                .btree => "BTREE",
                .hash => "HASH",
                .gist => "GIST",
                .gin => "GIN",
            };
            const unique_str = if (idx.is_unique) "UNIQUE " else "";
            stdout.print("CREATE {s}INDEX {s} ON {s} USING {s} ({s});\n", .{
                unique_str,
                idx.index_name,
                table_name,
                index_type,
                idx.column_name,
            }) catch {};
        }
    }

    // Dump data as INSERT statements
    const sql = std.fmt.allocPrint(allocator, "SELECT * FROM {s};", .{table_name}) catch {
        printError(stderr, "Out of memory.");
        return;
    };
    defer allocator.free(sql);

    var result = db.exec(sql) catch |err| {
        const msg = switch (err) {
            error.OutOfMemory => "Out of memory.",
            else => "Failed to query table data.",
        };
        printError(stderr, msg);
        return;
    };
    defer result.close(allocator);

    // Generate INSERT statements
    if (result.rows) |*rows| {
        while (true) {
            const maybe_row = rows.next() catch break;
            if (maybe_row) |row| {
                defer {
                    // Free row resources
                    for (row.values) |v| v.free(allocator);
                    allocator.free(row.values);
                    allocator.free(row.columns);
                }

                // Build INSERT statement
                stdout.print("INSERT INTO {s} VALUES (", .{table_name}) catch {};

                for (row.values, 0..) |value, i| {
                    switch (value) {
                        .null_value => stdout.writeAll("NULL") catch {},
                        .integer => |v| stdout.print("{}", .{v}) catch {},
                        .real => |v| stdout.print("{d}", .{v}) catch {},
                        .text => |v| {
                            // Escape single quotes in text
                            stdout.writeAll("'") catch {};
                            for (v) |ch| {
                                if (ch == '\'') {
                                    stdout.writeAll("''") catch {};
                                } else {
                                    stdout.writeByte(ch) catch {};
                                }
                            }
                            stdout.writeAll("'") catch {};
                        },
                        .blob => |v| {
                            // Output as hex string
                            stdout.writeAll("X'") catch {};
                            for (v) |byte| {
                                stdout.print("{X:0>2}", .{byte}) catch {};
                            }
                            stdout.writeAll("'") catch {};
                        },
                        .boolean => |v| {
                            const bool_str = if (v) "TRUE" else "FALSE";
                            stdout.writeAll(bool_str) catch {};
                        },
                        else => {
                            // For other types (date, time, timestamp, numeric, uuid, array, etc.)
                            // Use NULL placeholder for now
                            stdout.writeAll("NULL") catch {};
                        },
                    }

                    if (i < row.values.len - 1) {
                        stdout.writeAll(", ") catch {};
                    }
                }

                stdout.writeAll(");\n") catch {};
            } else break;
        }
    }
}

const DotCommandResult = enum { ok, quit };

fn handleDotCommand(allocator: std.mem.Allocator, db: *Database, cmd: []const u8, mode: *OutputMode, show_timer: *bool, stdout: anytype, stderr: anytype) DotCommandResult {
    if (std.mem.eql(u8, cmd, ".quit") or std.mem.eql(u8, cmd, ".exit")) {
        stdout.writeAll("Bye!\n") catch {};
        return .quit;
    } else if (std.mem.eql(u8, cmd, ".help")) {
        stdout.writeAll(
            \\.help               Show this help
            \\.quit               Exit the shell
            \\.exit               Exit the shell
            \\.mode MODE          Set output mode (table, csv, json, jsonl, plain)
            \\.mode               Show current output mode
            \\.timer on|off       Enable or disable query execution timing
            \\.timer              Show current timer setting
            \\.databases          List database connections
            \\.tables             List all tables
            \\.indexes [TABLE]    List all indexes or indexes for a specific table
            \\.schema [TABLE]     Show CREATE TABLE statements for all tables or specific table
            \\.dump               Dump database as SQL text (CREATE TABLE + INSERT statements)
            \\.read FILENAME      Read and execute SQL from file
            \\
        ) catch {};
    } else if (std.mem.startsWith(u8, cmd, ".timer")) {
        const rest = std.mem.trimLeft(u8, cmd[6..], " \t");
        if (rest.len == 0) {
            // Show current timer setting
            const setting = if (show_timer.*) "on" else "off";
            stdout.print("Timer: {s}\n", .{setting}) catch {};
        } else if (std.mem.eql(u8, rest, "on")) {
            show_timer.* = true;
            stdout.writeAll("Timer enabled\n") catch {};
        } else if (std.mem.eql(u8, rest, "off")) {
            show_timer.* = false;
            stdout.writeAll("Timer disabled\n") catch {};
        } else {
            printError(stderr, "Usage: .timer on|off");
        }
    } else if (std.mem.startsWith(u8, cmd, ".mode")) {
        const rest = std.mem.trimLeft(u8, cmd[5..], " \t");
        if (rest.len == 0) {
            // Show current mode
            const mode_name = @tagName(mode.*);
            stdout.print("Current mode: {s}\n", .{mode_name}) catch {};
        } else {
            if (parseModeString(rest)) |new_mode| {
                mode.* = new_mode;
                stdout.print("Output mode set to: {s}\n", .{@tagName(new_mode)}) catch {};
            } else {
                printError(stderr, "Invalid mode. Use: table, csv, json, jsonl, plain");
            }
        }
    } else if (std.mem.eql(u8, cmd, ".databases")) {
        showDatabases(db, stdout);
    } else if (std.mem.eql(u8, cmd, ".tables")) {
        listTables(allocator, db, stdout, stderr);
    } else if (std.mem.startsWith(u8, cmd, ".indexes")) {
        const rest = std.mem.trimLeft(u8, cmd[8..], " \t");
        const table_name = if (rest.len > 0) rest else null;
        showIndexes(allocator, db, table_name, stdout, stderr);
    } else if (std.mem.startsWith(u8, cmd, ".schema")) {
        const rest = std.mem.trimLeft(u8, cmd[7..], " \t");
        const table_name = if (rest.len > 0) rest else null;
        showSchema(allocator, db, table_name, stdout, stderr);
    } else if (std.mem.eql(u8, cmd, ".dump")) {
        dumpDatabase(allocator, db, stdout, stderr);
    } else if (std.mem.startsWith(u8, cmd, ".read")) {
        const rest = std.mem.trimLeft(u8, cmd[5..], " \t");
        if (rest.len == 0) {
            printError(stderr, "Usage: .read FILENAME");
        } else {
            readAndExecuteFile(allocator, db, rest, mode.*, show_timer.*, stdout, stderr);
        }
    } else {
        printError(stderr, "Unknown command. Type .help for usage hints.");
    }
    return .ok;
}

// ── Utility Functions ──────────────────────────────────────────────────

fn sqlHighlighter(buf: []const u8, writer: std.io.AnyWriter) anyerror!void {
    var tok = silica.tokenizer.Tokenizer.init(buf);
    var last_end: usize = 0;

    while (true) {
        const t = tok.next();
        if (t.type == .eof) break;

        if (t.start > last_end) {
            try writer.writeAll(buf[last_end..t.start]);
        }

        const text = buf[t.start .. t.start + t.len];

        if (t.type.isKeyword()) {
            try sailor.color.writeStyled(writer, .{
                .fg = .{ .basic = .blue },
                .attrs = .{ .bold = true },
            }, text);
        } else if (t.type == .string_literal) {
            try sailor.color.writeStyled(writer, .{
                .fg = .{ .basic = .green },
            }, text);
        } else if (t.type == .integer_literal or t.type == .float_literal) {
            try sailor.color.writeStyled(writer, .{
                .fg = .{ .basic = .cyan },
            }, text);
        } else {
            try writer.writeAll(text);
        }

        last_end = t.start + t.len;
    }

    if (last_end < buf.len) {
        try writer.writeAll(buf[last_end..]);
    }
}

const Validation = enum { complete, incomplete };

fn sqlValidator(buf: []const u8) Validation {
    const trimmed = std.mem.trimRight(u8, buf, " \t\r\n");
    if (trimmed.len == 0) return .complete;
    if (trimmed[trimmed.len - 1] == ';') return .complete;
    if (trimmed[0] == '.') return .complete;
    return .incomplete;
}

fn sqlCompleter(buf: []const u8, allocator: std.mem.Allocator) anyerror![]const []const u8 {
    const trimmed = std.mem.trimRight(u8, buf, " \t");
    if (trimmed.len == 0) return &.{};

    var word_start = trimmed.len;
    while (word_start > 0 and isWordChar(trimmed[word_start - 1])) : (word_start -= 1) {}

    const prefix = trimmed[word_start..];
    if (prefix.len == 0) return &.{};

    var results = std.ArrayListUnmanaged([]const u8){};

    for (sql_keywords) |kw| {
        if (kw.len >= prefix.len and std.ascii.startsWithIgnoreCase(kw, prefix)) {
            results.append(allocator, kw) catch continue;
        }
    }

    return results.toOwnedSlice(allocator) catch &.{};
}

fn isWordChar(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '_';
}

fn buildHistoryPath(allocator: std.mem.Allocator) ?[]const u8 {
    const home = std.process.getEnvVarOwned(allocator, "HOME") catch return null;
    defer allocator.free(home);
    return std.fmt.allocPrint(allocator, "{s}/.silica_history", .{home}) catch null;
}

fn printUsage(writer: anytype) void {
    writer.writeAll(
        \\Usage: silica [OPTIONS] <database>
        \\       silica server [OPTIONS] <database>
        \\
        \\A lightweight, embedded relational database engine.
        \\
        \\Arguments:
        \\  <database>    Path to the database file
        \\
        \\Commands:
        \\  server        Start PostgreSQL wire protocol server
        \\
        \\
    ) catch {};
    sailor.arg.Parser(&CliFlags).writeHelp(writer) catch {};
    writer.writeAll(
        \\
        \\Examples:
        \\  silica mydb.db                  Open database in interactive mode
        \\  silica --tui mydb.db            Open TUI database browser
        \\  silica --csv mydb.db            Open with CSV output format
        \\  silica -m json mydb.db          Open with JSON output format
        \\  silica server mydb.db           Start server on 127.0.0.1:5433
        \\  silica server --port 5432 db    Start server on custom port
        \\
    ) catch {};
}

fn printError(writer: anytype, message: []const u8) void {
    sailor.color.writeStyled(writer, sailor.color.semantic.err, "error") catch {};
    writer.writeAll(": ") catch {};
    writer.writeAll(message) catch {};
    writer.writeAll("\n") catch {};
}

/// SQL keywords for tab completion.
const sql_keywords = [_][]const u8{
    "SELECT",     "FROM",       "WHERE",       "INSERT",       "INTO",
    "VALUES",     "UPDATE",     "SET",         "DELETE",       "CREATE",
    "TABLE",      "DROP",       "INDEX",       "ALTER",        "ADD",
    "COLUMN",     "RENAME",     "PRIMARY",     "KEY",          "UNIQUE",
    "NOT",        "NULL",       "DEFAULT",     "CHECK",        "FOREIGN",
    "REFERENCES", "BEGIN",      "COMMIT",      "ROLLBACK",     "SAVEPOINT",
    "RELEASE",    "EXPLAIN",    "ORDER",       "BY",           "ASC",
    "DESC",       "LIMIT",      "OFFSET",      "GROUP",        "HAVING",
    "DISTINCT",   "ALL",        "UNION",       "EXCEPT",       "INTERSECT",
    "JOIN",       "INNER",      "LEFT",        "RIGHT",        "FULL",
    "OUTER",      "CROSS",      "NATURAL",     "ON",           "AS",
    "AND",        "OR",         "IN",          "BETWEEN",      "IS",
    "LIKE",       "CASE",       "WHEN",        "THEN",         "ELSE",
    "END",        "CAST",       "COUNT",       "SUM",          "AVG",
    "MIN",        "MAX",        "INTEGER",     "INT",          "REAL",
    "TEXT",       "BLOB",       "BOOLEAN",     "VARCHAR",      "TRUE",
    "FALSE",      "IF",         "EXISTS",      "AUTOINCREMENT", "TRANSACTION",
    "VACUUM",     "ANALYZE",    "REINDEX",     "VIEW",         "TRIGGER",
    "FUNCTION",   "WITH",       "RECURSIVE",   "WINDOW",       "PARTITION",
    "OVER",       "ROW_NUMBER", "RANK",        "DENSE_RANK",   "LAG",
    "LEAD",       "FIRST_VALUE", "LAST_VALUE", "ROWS",         "RANGE",
    "UNBOUNDED",  "PRECEDING",  "FOLLOWING",   "CURRENT",      "GRANT",
    "REVOKE",     "ROLE",       "POLICY",      "CONCURRENTLY", "MATERIALIZED",
    "TO",         "WITHOUT",    "ROWID",       "STRICT",       "TEMP",
    "TEMPORARY",  "REPLACE",    "CONSTRAINT",  "CASCADE",      "RESTRICT",
    "ACTION",     "NO",         "OF",          "ENUM",         "DOMAIN",
    "RETURNS",    "LANGUAGE",   "IMMUTABLE",   "STABLE",       "VOLATILE",
    "BEFORE",     "AFTER",      "INSTEAD",     "EACH",         "STATEMENT",
    "OLD",        "NEW",        "ENABLE",      "DISABLE",      "TRUNCATE",
    "GLOB",       "ANY",        "ROW",         "ISOLATION",    "READ",
    "COMMITTED",  "REPEATABLE", "SERIALIZABLE", "PRAGMA",      "SHOW",
    "RESET",      "DATE",       "TIME",        "TIMESTAMP",    "INTERVAL",
    "NUMERIC",    "DECIMAL",    "UUID",        "SERIAL",       "BIGSERIAL",
    "ARRAY",      "JSON",       "JSONB",       "TSVECTOR",     "TSQUERY",
};

// ── Tests ────────────────────────────────────────────────────

test {
    _ = tui_mod;
}

test "printUsage does not error" {
    var buf: [4096]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    printUsage(&w);
    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "Usage: silica") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "<database>") != null);
}

test "printError formats error message" {
    var buf: [256]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    printError(&w, "something went wrong");
    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "error") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "something went wrong") != null);
}

test "version string is set" {
    try std.testing.expectEqualStrings("1.0.0", version);
}

test "sqlValidator complete with semicolon" {
    try std.testing.expectEqual(Validation.complete, sqlValidator("SELECT 1;"));
    try std.testing.expectEqual(Validation.complete, sqlValidator("SELECT * FROM t;  "));
}

test "sqlValidator incomplete without semicolon" {
    try std.testing.expectEqual(Validation.incomplete, sqlValidator("SELECT 1"));
    try std.testing.expectEqual(Validation.incomplete, sqlValidator("SELECT *"));
}

test "sqlValidator complete for dot-commands" {
    try std.testing.expectEqual(Validation.complete, sqlValidator(".help"));
    try std.testing.expectEqual(Validation.complete, sqlValidator(".quit"));
}

test "sqlValidator empty input is complete" {
    try std.testing.expectEqual(Validation.complete, sqlValidator(""));
    try std.testing.expectEqual(Validation.complete, sqlValidator("   "));
}

test "sqlHighlighter does not error on valid SQL" {
    var buf: [1024]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    try sqlHighlighter("SELECT * FROM users WHERE id = 1;", fbs.writer().any());
    const output = fbs.getWritten();
    try std.testing.expect(output.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, output, "users") != null);
}

test "sqlHighlighter empty input" {
    var buf: [64]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    try sqlHighlighter("", fbs.writer().any());
    try std.testing.expectEqual(@as(usize, 0), fbs.getWritten().len);
}

test "sqlCompleter matches keywords" {
    const allocator = std.testing.allocator;
    const results = try sqlCompleter("SEL", allocator);
    defer allocator.free(results);
    try std.testing.expect(results.len >= 1);
    var found = false;
    for (results) |r| {
        if (std.mem.eql(u8, r, "SELECT")) found = true;
    }
    try std.testing.expect(found);
}

test "sqlCompleter no match" {
    const allocator = std.testing.allocator;
    const results = try sqlCompleter("ZZZZZ", allocator);
    defer allocator.free(results);
    try std.testing.expectEqual(@as(usize, 0), results.len);
}

test "sqlCompleter empty input" {
    const allocator = std.testing.allocator;
    const results = try sqlCompleter("", allocator);
    try std.testing.expectEqual(@as(usize, 0), results.len);
}

test "handleDotCommand help" {
    const allocator = std.testing.allocator;
    const path = ":memory:";
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();

    var buf: [2048]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;
    var show_timer = true;
    const result = handleDotCommand(allocator, &db, ".help", &mode, &show_timer, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);
    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, ".help") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, ".quit") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, ".mode") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, ".databases") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, ".read") != null);
}

test "handleDotCommand databases" {
    const allocator = std.testing.allocator;
    const path = "test_databases.db";
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();
    defer std.fs.cwd().deleteFile(path) catch {};

    var buf: [4096]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;

    var show_timer = true;
    const result = handleDotCommand(allocator, &db, ".databases", &mode, &show_timer, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);
    const output = fbs.getWritten();

    // Verify output contains SQLite-compatible format
    try std.testing.expect(std.mem.indexOf(u8, output, "seq") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "name") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "file") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "0") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "main") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, path) != null);
}

test "handleDotCommand databases - memory database" {
    const allocator = std.testing.allocator;
    const path = ":memory:";
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();

    var buf: [4096]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;

    var show_timer = true;
    const result = handleDotCommand(allocator, &db, ".databases", &mode, &show_timer, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);
    const output = fbs.getWritten();

    // Verify output shows :memory: database
    try std.testing.expect(std.mem.indexOf(u8, output, ":memory:") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "main") != null);
}

test "handleDotCommand quit" {
    const allocator = std.testing.allocator;
    const path = ":memory:";
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();

    var buf: [256]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;
    var show_timer = true;
    const result = handleDotCommand(allocator, &db, ".quit", &mode, &show_timer, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.quit, result);
    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "Bye!") != null);
}

test "handleDotCommand unknown" {
    const allocator = std.testing.allocator;
    const path = ":memory:";
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();

    var buf: [256]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;
    var show_timer = true;
    const result = handleDotCommand(allocator, &db, ".foobar", &mode, &show_timer, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);
    const eoutput = efbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, eoutput, "Unknown command") != null);
}

test "handleDotCommand mode set" {
    const allocator = std.testing.allocator;
    const path = ":memory:";
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();

    var buf: [256]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;
    var show_timer = true;
    _ = handleDotCommand(allocator, &db, ".mode csv", &mode, &show_timer, &w, &ew);
    try std.testing.expectEqual(OutputMode.csv, mode);
    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "csv") != null);
}

test "handleDotCommand mode show" {
    const allocator = std.testing.allocator;
    const path = ":memory:";
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();

    var buf: [256]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .json;
    var show_timer = true;
    _ = handleDotCommand(allocator, &db, ".mode", &mode, &show_timer, &w, &ew);
    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "json") != null);
}

test "handleDotCommand mode invalid" {
    const allocator = std.testing.allocator;
    const path = ":memory:";
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();

    var buf: [256]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;
    var show_timer = true;
    _ = handleDotCommand(allocator, &db, ".mode foobar", &mode, &show_timer, &w, &ew);
    try std.testing.expectEqual(OutputMode.table, mode); // unchanged
    const eoutput = efbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, eoutput, "Invalid mode") != null);
}

test "handleDotCommand tables" {
    const allocator = std.testing.allocator;
    const path = ":memory:";
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();

    // Create a test table
    _ = db.exec("CREATE TABLE users (id INTEGER, name TEXT);") catch return error.SkipZigTest;

    var buf: [1024]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;

    var show_timer = true;
    const result = handleDotCommand(allocator, &db, ".tables", &mode, &show_timer, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);
    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "users") != null);
}

test "handleDotCommand schema - all tables" {
    const allocator = std.testing.allocator;
    const path = ":memory:";
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();

    // Create test tables
    _ = db.exec("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL);") catch return error.SkipZigTest;
    _ = db.exec("CREATE TABLE posts (id INTEGER, title TEXT, UNIQUE(title));") catch return error.SkipZigTest;

    var buf: [4096]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;

    var show_timer = true;
    const result = handleDotCommand(allocator, &db, ".schema", &mode, &show_timer, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);
    const output = fbs.getWritten();

    // Verify both tables appear
    try std.testing.expect(std.mem.indexOf(u8, output, "CREATE TABLE posts") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "CREATE TABLE users") != null);

    // Verify column types and constraints
    try std.testing.expect(std.mem.indexOf(u8, output, "id INTEGER PRIMARY KEY") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "name TEXT NOT NULL") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "UNIQUE (title)") != null);
}

test "handleDotCommand schema - specific table" {
    const allocator = std.testing.allocator;
    const path = ":memory:";
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();

    // Create test tables
    _ = db.exec("CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT UNIQUE);") catch return error.SkipZigTest;
    _ = db.exec("CREATE TABLE posts (id INTEGER, content TEXT);") catch return error.SkipZigTest;

    var buf: [2048]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;

    var show_timer = true;
    const result = handleDotCommand(allocator, &db, ".schema users", &mode, &show_timer, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);
    const output = fbs.getWritten();

    // Verify only users table appears
    try std.testing.expect(std.mem.indexOf(u8, output, "CREATE TABLE users") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "posts") == null);

    // Verify constraints
    try std.testing.expect(std.mem.indexOf(u8, output, "PRIMARY KEY") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "AUTOINCREMENT") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "UNIQUE") != null);
}

test "handleDotCommand schema - with composite primary key" {
    const allocator = std.testing.allocator;
    const path = ":memory:";
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();

    // Create table with composite primary key
    _ = db.exec("CREATE TABLE user_roles (user_id INTEGER, role_id INTEGER, PRIMARY KEY (user_id, role_id));") catch return error.SkipZigTest;

    var buf: [2048]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;

    var show_timer = true;
    const result = handleDotCommand(allocator, &db, ".schema user_roles", &mode, &show_timer, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);
    const output = fbs.getWritten();

    // Verify composite primary key
    try std.testing.expect(std.mem.indexOf(u8, output, "PRIMARY KEY (user_id, role_id)") != null);
}

test "handleDotCommand schema - table not found" {
    const allocator = std.testing.allocator;
    const path = ":memory:";
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();

    var buf: [256]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;

    var show_timer = true;
    const result = handleDotCommand(allocator, &db, ".schema nonexistent", &mode, &show_timer, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);
    const eoutput = efbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, eoutput, "Table not found") != null);
}

test "handleDotCommand schema - no tables" {
    const allocator = std.testing.allocator;
    const path = "test_schema_no_tables.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();

    var buf: [256]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;

    var show_timer = true;
    const result = handleDotCommand(allocator, &db, ".schema", &mode, &show_timer, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);
    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "No tables found") != null);
}

test "handleDotCommand indexes - named index" {
    const allocator = std.testing.allocator;
    const path = "test_indexes_named.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();

    // Create table and index
    _ = db.exec("CREATE TABLE users (id INTEGER, email TEXT);") catch return error.SkipZigTest;
    _ = db.exec("CREATE INDEX idx_email ON users(email);") catch return error.SkipZigTest;

    var buf: [2048]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;

    var show_timer = true;
    const result = handleDotCommand(allocator, &db, ".indexes users", &mode, &show_timer, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);
    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "idx_email") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "email") != null);
}

test "handleDotCommand indexes - all tables" {
    const allocator = std.testing.allocator;
    const path = "test_indexes_all.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();

    // Create tables with indexes
    _ = db.exec("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);") catch return error.SkipZigTest;
    _ = db.exec("CREATE INDEX idx_name ON users(name);") catch return error.SkipZigTest;
    _ = db.exec("CREATE TABLE posts (id INTEGER, title TEXT);") catch return error.SkipZigTest;
    _ = db.exec("CREATE UNIQUE INDEX idx_title ON posts(title);") catch return error.SkipZigTest;

    var buf: [4096]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;

    var show_timer = true;
    const result = handleDotCommand(allocator, &db, ".indexes", &mode, &show_timer, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);
    const output = fbs.getWritten();
    // Verify both indexes appear
    try std.testing.expect(std.mem.indexOf(u8, output, "idx_name") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "idx_title") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "UNIQUE") != null);
}

test "handleDotCommand indexes - no indexes" {
    const allocator = std.testing.allocator;
    const path = "test_indexes_none.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();

    // Create table without indexes
    _ = db.exec("CREATE TABLE plain (id INTEGER, data TEXT);") catch return error.SkipZigTest;

    var buf: [1024]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;

    var show_timer = true;
    const result = handleDotCommand(allocator, &db, ".indexes plain", &mode, &show_timer, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);
    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "No indexes found") != null);
}

test "handleDotCommand indexes - table not found" {
    const allocator = std.testing.allocator;
    const path = "test_indexes_notfound.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();

    var buf: [256]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;

    var show_timer = true;
    const result = handleDotCommand(allocator, &db, ".indexes nonexistent", &mode, &show_timer, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);
    const eoutput = efbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, eoutput, "Table not found") != null);
}

test "handleDotCommand indexes - no tables" {
    const allocator = std.testing.allocator;
    const path = "test_indexes_no_tables.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();

    var buf: [256]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;

    var show_timer = true;
    const result = handleDotCommand(allocator, &db, ".indexes", &mode, &show_timer, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);
    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "No tables found") != null);
}

test "handleDotCommand dump - basic table" {
    const allocator = std.testing.allocator;
    const path = "test_dump_basic.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();

    // Create table and insert data
    var result1 = db.exec("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL);") catch return error.SkipZigTest;
    defer result1.close(allocator);
    var result2 = db.exec("INSERT INTO users (id, name) VALUES (1, 'Alice'), (2, 'Bob');") catch return error.SkipZigTest;
    defer result2.close(allocator);

    // Run .dump
    var buf: [2048]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;

    var show_timer = true;
    const result = handleDotCommand(allocator, &db, ".dump", &mode, &show_timer, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);
    const output = fbs.getWritten();

    // Verify output contains expected SQL
    try std.testing.expect(std.mem.indexOf(u8, output, "BEGIN TRANSACTION") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "CREATE TABLE users") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "id INTEGER PRIMARY KEY") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "name TEXT NOT NULL") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "INSERT INTO users VALUES (1, 'Alice')") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "INSERT INTO users VALUES (2, 'Bob')") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "COMMIT") != null);
}

test "handleDotCommand dump - empty database" {
    const allocator = std.testing.allocator;
    const path = "test_dump_empty.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();

    var buf: [256]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;

    var show_timer = true;
    const result = handleDotCommand(allocator, &db, ".dump", &mode, &show_timer, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);
    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "No tables found") != null);
}

test "handleDotCommand dump - with indexes" {
    const allocator = std.testing.allocator;
    const path = "test_dump_indexes.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();

    // Create table with index
    var result1 = db.exec("CREATE TABLE products (id INTEGER PRIMARY KEY, sku TEXT UNIQUE);") catch return error.SkipZigTest;
    defer result1.close(allocator);
    var result2 = db.exec("CREATE INDEX idx_sku ON products (sku);") catch return error.SkipZigTest;
    defer result2.close(allocator);
    var result3 = db.exec("INSERT INTO products (id, sku) VALUES (1, 'ABC123');") catch return error.SkipZigTest;
    defer result3.close(allocator);

    var buf: [2048]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;

    var show_timer = true;
    const result = handleDotCommand(allocator, &db, ".dump", &mode, &show_timer, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);
    const output = fbs.getWritten();

    // Verify CREATE TABLE and CREATE INDEX statements
    try std.testing.expect(std.mem.indexOf(u8, output, "CREATE TABLE products") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "CREATE INDEX idx_sku ON products") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "INSERT INTO products VALUES (1, 'ABC123')") != null);
}

test "processSQL parses valid SQL" {
    var buf: [1024]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    processSQL(std.testing.allocator, "SELECT 1;", &w, &ew);
    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "Parsed: SELECT") != null);
}

test "processSQL reports parse errors" {
    var buf: [256]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [1024]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    processSQL(std.testing.allocator, "INVALID SYNTAX;", &w, &ew);
    const eoutput = efbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, eoutput, "error") != null);
}

test "buildHistoryPath returns path" {
    const allocator = std.testing.allocator;
    const path = buildHistoryPath(allocator);
    if (path) |p| {
        defer allocator.free(p);
        try std.testing.expect(std.mem.endsWith(u8, p, ".silica_history"));
    }
}

test "isWordChar" {
    try std.testing.expect(isWordChar('a'));
    try std.testing.expect(isWordChar('Z'));
    try std.testing.expect(isWordChar('0'));
    try std.testing.expect(isWordChar('_'));
    try std.testing.expect(!isWordChar(' '));
    try std.testing.expect(!isWordChar(';'));
}

test "printStmtInfo formats statement types" {
    {
        var buf: [256]u8 = undefined;
        var fbs = std.io.fixedBufferStream(&buf);
        var w = fbs.writer();
        printStmtInfo(&w, .{ .select = .{ .columns = &.{.all_columns} } });
        const output = fbs.getWritten();
        try std.testing.expect(std.mem.indexOf(u8, output, "SELECT") != null);
    }
    {
        var buf: [256]u8 = undefined;
        var fbs = std.io.fixedBufferStream(&buf);
        var w = fbs.writer();
        printStmtInfo(&w, .{ .insert = .{ .table = "users" } });
        const output = fbs.getWritten();
        try std.testing.expect(std.mem.indexOf(u8, output, "INSERT INTO users") != null);
    }
    {
        var buf: [256]u8 = undefined;
        var fbs = std.io.fixedBufferStream(&buf);
        var w = fbs.writer();
        printStmtInfo(&w, .{ .create_table = .{ .name = "test" } });
        const output = fbs.getWritten();
        try std.testing.expect(std.mem.indexOf(u8, output, "CREATE TABLE test") != null);
    }
    {
        var buf: [256]u8 = undefined;
        var fbs = std.io.fixedBufferStream(&buf);
        var w = fbs.writer();
        printStmtInfo(&w, .{ .drop_table = .{ .name = "test" } });
        const output = fbs.getWritten();
        try std.testing.expect(std.mem.indexOf(u8, output, "DROP TABLE test") != null);
    }
    {
        var buf: [256]u8 = undefined;
        var fbs = std.io.fixedBufferStream(&buf);
        var w = fbs.writer();
        printStmtInfo(&w, .{ .transaction = .{ .begin = .{} } });
        const output = fbs.getWritten();
        try std.testing.expect(std.mem.indexOf(u8, output, "BEGIN") != null);
    }
}

test "valueToText converts all types" {
    const allocator = std.testing.allocator;

    // Integer
    {
        const text = valueToText(allocator, .{ .integer = 42 }).?;
        defer allocator.free(text);
        try std.testing.expectEqualStrings("42", text);
    }
    // Real
    {
        const text = valueToText(allocator, .{ .real = 3.14 }).?;
        defer allocator.free(text);
        // Just check it starts with "3.14"
        try std.testing.expect(std.mem.startsWith(u8, text, "3.14"));
    }
    // Text
    {
        const text = valueToText(allocator, .{ .text = "hello" }).?;
        defer allocator.free(text);
        try std.testing.expectEqualStrings("hello", text);
    }
    // Boolean
    {
        const t = valueToText(allocator, .{ .boolean = true }).?;
        defer allocator.free(t);
        try std.testing.expectEqualStrings("TRUE", t);
        const f = valueToText(allocator, .{ .boolean = false }).?;
        defer allocator.free(f);
        try std.testing.expectEqualStrings("FALSE", f);
    }
    // Null
    {
        const text = valueToText(allocator, .null_value).?;
        defer allocator.free(text);
        try std.testing.expectEqualStrings("NULL", text);
    }
}

test "parseModeString" {
    try std.testing.expectEqual(OutputMode.table, parseModeString("table").?);
    try std.testing.expectEqual(OutputMode.csv, parseModeString("csv").?);
    try std.testing.expectEqual(OutputMode.json, parseModeString("json").?);
    try std.testing.expectEqual(OutputMode.jsonl, parseModeString("jsonl").?);
    try std.testing.expectEqual(OutputMode.plain, parseModeString("plain").?);
    try std.testing.expect(parseModeString("bogus") == null);
}

test "formatTable renders bordered table" {
    const allocator = std.testing.allocator;

    const headers = &[_][]const u8{ "id", "name" };

    // Create mock rows manually
    var rows: [2]Row = undefined;
    const vals1 = try allocator.alloc(Value, 2);
    defer allocator.free(vals1);
    vals1[0] = .{ .integer = 1 };
    vals1[1] = .{ .text = "Alice" };
    const cols1 = try allocator.alloc([]const u8, 2);
    defer allocator.free(cols1);
    cols1[0] = "id";
    cols1[1] = "name";
    rows[0] = .{ .columns = cols1, .values = vals1, .allocator = allocator };

    const vals2 = try allocator.alloc(Value, 2);
    defer allocator.free(vals2);
    vals2[0] = .{ .integer = 2 };
    vals2[1] = .{ .text = "Bob" };
    const cols2 = try allocator.alloc([]const u8, 2);
    defer allocator.free(cols2);
    cols2[0] = "id";
    cols2[1] = "name";
    rows[1] = .{ .columns = cols2, .values = vals2, .allocator = allocator };

    var out_buf: [4096]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&out_buf);
    var w = fbs.writer();
    formatTable(allocator, headers, &rows, &w);
    const output = fbs.getWritten();

    try std.testing.expect(output.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, output, "Alice") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "Bob") != null);
    // Should have table borders
    try std.testing.expect(std.mem.indexOf(u8, output, "│") != null or std.mem.indexOf(u8, output, "|") != null);
}

test "formatCsv renders CSV output" {
    const allocator = std.testing.allocator;

    const headers = &[_][]const u8{ "id", "name" };

    var rows: [1]Row = undefined;
    const vals = try allocator.alloc(Value, 2);
    defer allocator.free(vals);
    vals[0] = .{ .integer = 1 };
    vals[1] = .{ .text = "Alice" };
    const cols = try allocator.alloc([]const u8, 2);
    defer allocator.free(cols);
    cols[0] = "id";
    cols[1] = "name";
    rows[0] = .{ .columns = cols, .values = vals, .allocator = allocator };

    var out_buf: [4096]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&out_buf);
    var w = fbs.writer();
    formatCsv(headers, &rows, allocator, &w);
    const output = fbs.getWritten();

    try std.testing.expect(std.mem.indexOf(u8, output, "id,name") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "1,Alice") != null);
}

test "formatJson renders JSON array" {
    const allocator = std.testing.allocator;

    const headers = &[_][]const u8{ "id", "name" };

    var rows: [1]Row = undefined;
    const vals = try allocator.alloc(Value, 2);
    defer allocator.free(vals);
    vals[0] = .{ .integer = 1 };
    vals[1] = .{ .text = "Alice" };
    const cols = try allocator.alloc([]const u8, 2);
    defer allocator.free(cols);
    cols[0] = "id";
    cols[1] = "name";
    rows[0] = .{ .columns = cols, .values = vals, .allocator = allocator };

    var out_buf: [4096]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&out_buf);
    var w = fbs.writer();
    formatJson(headers, &rows, allocator, &w);
    const output = fbs.getWritten();

    try std.testing.expect(std.mem.indexOf(u8, output, "[") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "\"id\":1") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "\"name\":\"Alice\"") != null);
}

test "formatJsonl renders JSONL output" {
    const allocator = std.testing.allocator;

    const headers = &[_][]const u8{"val"};

    var rows: [2]Row = undefined;
    const vals1 = try allocator.alloc(Value, 1);
    defer allocator.free(vals1);
    vals1[0] = .null_value;
    const cols1 = try allocator.alloc([]const u8, 1);
    defer allocator.free(cols1);
    cols1[0] = "val";
    rows[0] = .{ .columns = cols1, .values = vals1, .allocator = allocator };

    const vals2 = try allocator.alloc(Value, 1);
    defer allocator.free(vals2);
    vals2[0] = .{ .boolean = true };
    const cols2 = try allocator.alloc([]const u8, 1);
    defer allocator.free(cols2);
    cols2[0] = "val";
    rows[1] = .{ .columns = cols2, .values = vals2, .allocator = allocator };

    var out_buf: [4096]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&out_buf);
    var w = fbs.writer();
    formatJsonl(headers, &rows, allocator, &w);
    const output = fbs.getWritten();

    try std.testing.expect(std.mem.indexOf(u8, output, "\"val\":null") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "\"val\":true") != null);
}

test "formatPlain renders key-value pairs" {
    const allocator = std.testing.allocator;

    const headers = &[_][]const u8{ "id", "name" };

    var rows: [1]Row = undefined;
    const vals = try allocator.alloc(Value, 2);
    defer allocator.free(vals);
    vals[0] = .{ .integer = 42 };
    vals[1] = .{ .text = "World" };
    const cols = try allocator.alloc([]const u8, 2);
    defer allocator.free(cols);
    cols[0] = "id";
    cols[1] = "name";
    rows[0] = .{ .columns = cols, .values = vals, .allocator = allocator };

    var out_buf: [4096]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&out_buf);
    var w = fbs.writer();
    formatPlain(headers, &rows, allocator, &w);
    const output = fbs.getWritten();

    try std.testing.expect(std.mem.indexOf(u8, output, "id = 42") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "name = World") != null);
}

test "handleDotCommand .read executes SQL from file" {
    const allocator = std.testing.allocator;

    // Create test database
    const path = "test_read_command.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var db = Database.open(allocator, path, .{}) catch unreachable;
    defer db.close();

    // Create a SQL file with multiple statements
    const sql_file = "test_script.sql";
    defer std.fs.cwd().deleteFile(sql_file) catch {};

    const sql_content =
        \\CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);
        \\INSERT INTO users (id, name) VALUES (1, 'Alice');
        \\INSERT INTO users (id, name) VALUES (2, 'Bob');
    ;

    const file = try std.fs.cwd().createFile(sql_file, .{});
    defer file.close();
    try file.writeAll(sql_content);

    // Execute .read command
    var out_buf: [4096]u8 = undefined;
    var err_buf: [4096]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&out_buf);
    var ebs = std.io.fixedBufferStream(&err_buf);
    var w = fbs.writer();
    var ew = ebs.writer();

    var mode = OutputMode.table;
    var show_timer = true;
    const result = handleDotCommand(allocator, &db, ".read test_script.sql", &mode, &show_timer, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "Executed 3 statement(s)") != null);

    // Verify data was inserted
    var query_result = db.exec("SELECT COUNT(*) FROM users;") catch unreachable;
    defer query_result.close(allocator);

    if (query_result.rows) |*rows| {
        const row = rows.next() catch unreachable;
        if (row) |r| {
            defer {
                for (r.values) |v| v.free(allocator);
                allocator.free(r.values);
                allocator.free(r.columns);
            }
            try std.testing.expectEqual(Value{ .integer = 2 }, r.values[0]);
        }
    }
}

test "handleDotCommand .read handles file not found" {
    const allocator = std.testing.allocator;

    const path = "test_read_not_found.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var db = Database.open(allocator, path, .{}) catch unreachable;
    defer db.close();

    var out_buf: [4096]u8 = undefined;
    var err_buf: [4096]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&out_buf);
    var ebs = std.io.fixedBufferStream(&err_buf);
    var w = fbs.writer();
    var ew = ebs.writer();

    var mode = OutputMode.table;
    var show_timer = true;
    const result = handleDotCommand(allocator, &db, ".read nonexistent.sql", &mode, &show_timer, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const error_output = ebs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, error_output, "File not found") != null);
}

test "handleDotCommand .read skips SQL comments" {
    const allocator = std.testing.allocator;

    const path = "test_read_comments.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var db = Database.open(allocator, path, .{}) catch unreachable;
    defer db.close();

    const sql_file = "test_comments.sql";
    defer std.fs.cwd().deleteFile(sql_file) catch {};

    const sql_content =
        \\-- This is a comment
        \\CREATE TABLE test (id INTEGER);
        \\-- Another comment
        \\INSERT INTO test VALUES (1);
    ;

    const file = try std.fs.cwd().createFile(sql_file, .{});
    defer file.close();
    try file.writeAll(sql_content);

    var out_buf: [4096]u8 = undefined;
    var err_buf: [4096]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&out_buf);
    var ebs = std.io.fixedBufferStream(&err_buf);
    var w = fbs.writer();
    var ew = ebs.writer();

    var mode = OutputMode.table;
    var show_timer = true;
    const result = handleDotCommand(allocator, &db, ".read test_comments.sql", &mode, &show_timer, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    // Should execute 2 statements (CREATE TABLE + INSERT), not 4 (skipped comments)
    try std.testing.expect(std.mem.indexOf(u8, output, "Executed 2 statement(s)") != null);
}

test "handleDotCommand .read requires filename" {
    const allocator = std.testing.allocator;

    const path = "test_read_no_arg.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var db = Database.open(allocator, path, .{}) catch unreachable;
    defer db.close();

    var out_buf: [4096]u8 = undefined;
    var err_buf: [4096]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&out_buf);
    var ebs = std.io.fixedBufferStream(&err_buf);
    var w = fbs.writer();
    var ew = ebs.writer();

    var mode = OutputMode.table;
    var show_timer = true;
    const result = handleDotCommand(allocator, &db, ".read", &mode, &show_timer, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const error_output = ebs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, error_output, "Usage: .read FILENAME") != null);
}

test "handleDotCommand .timer on" {
    const allocator = std.testing.allocator;

    const path = "test_timer_on.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var db = Database.open(allocator, path, .{}) catch unreachable;
    defer db.close();

    var out_buf: [4096]u8 = undefined;
    var err_buf: [4096]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&out_buf);
    var ebs = std.io.fixedBufferStream(&err_buf);
    var w = fbs.writer();
    var ew = ebs.writer();

    var mode = OutputMode.table;
    var show_timer = false; // Start disabled

    const result = handleDotCommand(allocator, &db, ".timer on", &mode, &show_timer, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);
    try std.testing.expectEqual(true, show_timer); // Should be enabled now

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "Timer enabled") != null);
}

test "handleDotCommand .timer off" {
    const allocator = std.testing.allocator;

    const path = "test_timer_off.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var db = Database.open(allocator, path, .{}) catch unreachable;
    defer db.close();

    var out_buf: [4096]u8 = undefined;
    var err_buf: [4096]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&out_buf);
    var ebs = std.io.fixedBufferStream(&err_buf);
    var w = fbs.writer();
    var ew = ebs.writer();

    var mode = OutputMode.table;
    var show_timer = true; // Start enabled

    const result = handleDotCommand(allocator, &db, ".timer off", &mode, &show_timer, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);
    try std.testing.expectEqual(false, show_timer); // Should be disabled now

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "Timer disabled") != null);
}

test "handleDotCommand .timer shows current setting" {
    const allocator = std.testing.allocator;

    const path = "test_timer_show.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var db = Database.open(allocator, path, .{}) catch unreachable;
    defer db.close();

    var out_buf: [4096]u8 = undefined;
    var err_buf: [4096]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&out_buf);
    var ebs = std.io.fixedBufferStream(&err_buf);
    var w = fbs.writer();
    var ew = ebs.writer();

    var mode = OutputMode.table;
    var show_timer = true;

    const result = handleDotCommand(allocator, &db, ".timer", &mode, &show_timer, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "Timer: on") != null);
}

test "handleDotCommand .timer invalid argument" {
    const allocator = std.testing.allocator;

    const path = "test_timer_invalid.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var db = Database.open(allocator, path, .{}) catch unreachable;
    defer db.close();

    var out_buf: [4096]u8 = undefined;
    var err_buf: [4096]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&out_buf);
    var ebs = std.io.fixedBufferStream(&err_buf);
    var w = fbs.writer();
    var ew = ebs.writer();

    var mode = OutputMode.table;
    var show_timer = true;

    const result = handleDotCommand(allocator, &db, ".timer foobar", &mode, &show_timer, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);
    try std.testing.expectEqual(true, show_timer); // Should remain unchanged

    const error_output = ebs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, error_output, "Usage: .timer on|off") != null);
}

test "handleDotCommand .help includes .timer" {
    const allocator = std.testing.allocator;

    const path = "test_help_timer.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var db = Database.open(allocator, path, .{}) catch unreachable;
    defer db.close();

    var out_buf: [8192]u8 = undefined;
    var err_buf: [4096]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&out_buf);
    var ebs = std.io.fixedBufferStream(&err_buf);
    var w = fbs.writer();
    var ew = ebs.writer();

    var mode = OutputMode.table;
    var show_timer = true;
    const result = handleDotCommand(allocator, &db, ".help", &mode, &show_timer, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, ".timer") != null);
}

// Import wire_fuzz tests
test {
    _ = @import("server/wire_fuzz.zig");
}
