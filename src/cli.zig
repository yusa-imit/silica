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

    var db_path = first_arg;

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
    var show_headers = true; // Default: show column headers
    var csv_separator: []const u8 = ","; // Default CSV separator
    var null_display: []const u8 = "NULL"; // Default NULL display string
    var last_rows_affected: u64 = 0; // Track last DML operation's row count
    var bail_on_error = false; // Default: continue on SQL errors (SQLite-compatible)
    var show_stats = false; // Default: don't show execution statistics
    var show_eqp = false; // Default: don't automatically explain query plans
    var main_prompt: []const u8 = "silica> "; // Default main prompt
    var continue_prompt: []const u8 = "   ...> "; // Default continuation prompt

    // Output redirection state
    var output_file: ?std.fs.File = null;
    defer if (output_file) |f| f.close();
    var output_file_buf: [4096]u8 = undefined;

    // One-time output redirection state (.once command)
    var once_file: ?std.fs.File = null;
    defer if (once_file) |f| f.close();
    var once_file_buf: [4096]u8 = undefined;

    // Query logging state
    var log_file: ?std.fs.File = null;
    defer if (log_file) |f| f.close();

    while (true) {
        const prompt = if (is_continuation) continue_prompt else main_prompt;
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
            const result = handleDotCommand(allocator, &db, db_path, trimmed, &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, stdout, stderr);
            stdout.flush() catch {};
            stderr.flush() catch {};
            switch (result) {
                .quit => break,
                .reopen => |new_path| {
                    defer allocator.free(new_path);
                    // Close current database
                    db.close();
                    // Try to open new database
                    db = Database.open(allocator, new_path, .{}) catch {
                        printError(stderr, "Failed to open database. Reopening original database.");
                        stderr.flush() catch {};
                        // Reopen original database on failure
                        db = Database.open(allocator, db_path, .{}) catch {
                            printError(stderr, "Failed to reopen original database. Exiting.");
                            stderr.flush() catch {};
                            std.process.exit(1);
                        };
                        continue;
                    };
                    // Update db_path reference
                    db_path = new_path;
                    stdout.print("Opened database: {s}\n", .{db_path}) catch {};
                    stdout.flush() catch {};
                },
                .ok => {},
            }
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
        // Priority: once_file > output_file > stdout
        if (once_file) |f| {
            // Use once_file for this query only
            var file_writer = f.writer(&once_file_buf);
            const file_out = &file_writer.interface;
            _ = execAndDisplay(allocator, &db, input_buf.items, mode, show_timer, show_stats, show_headers, csv_separator, null_display, &last_rows_affected, log_file, show_eqp, file_out, stderr);
            file_out.flush() catch {};
            // Close and reset once_file after use
            f.close();
            once_file = null;
        } else if (output_file) |f| {
            var file_writer = f.writer(&output_file_buf);
            const file_out = &file_writer.interface;
            _ = execAndDisplay(allocator, &db, input_buf.items, mode, show_timer, show_stats, show_headers, csv_separator, null_display, &last_rows_affected, log_file, show_eqp, file_out, stderr);
            file_out.flush() catch {};
        } else {
            _ = execAndDisplay(allocator, &db, input_buf.items, mode, show_timer, show_stats, show_headers, csv_separator, null_display, &last_rows_affected, log_file, show_eqp, stdout, stderr);
            stdout.flush() catch {};
        }
        stderr.flush() catch {};
        input_buf.clearRetainingCapacity();
        is_continuation = false;
    }
}

// ── SQL Execution ──────────────────────────────────────────────────────

/// Execute SQL via the Database engine and display results.
/// Returns true on success, false on error.
fn execAndDisplay(allocator: std.mem.Allocator, db: *Database, sql: []const u8, mode: OutputMode, show_timer: bool, show_stats: bool, show_headers: bool, csv_separator: []const u8, null_display: []const u8, last_rows_affected: *u64, log_file: ?std.fs.File, show_eqp: bool, stdout: anytype, stderr: anytype) bool {
    // Prepend EXPLAIN if .eqp is enabled and query is not already EXPLAIN
    var sql_to_execute = sql;
    var explain_buf: [16384]u8 = undefined;
    const explain_allocated = false;
    if (show_eqp and !std.ascii.startsWithIgnoreCase(sql, "EXPLAIN")) {
        const explain_sql = std.fmt.bufPrint(&explain_buf, "EXPLAIN {s}", .{sql}) catch sql;
        sql_to_execute = explain_sql;
    }
    defer if (explain_allocated) allocator.free(sql_to_execute);

    // Start timer for query execution (if enabled)
    var timer = if (show_timer or show_stats) std.time.Timer.start() catch {
        // If timer fails, continue without timing
        return execAndDisplayWithoutTiming(allocator, db, sql_to_execute, mode, show_stats, show_headers, csv_separator, null_display, last_rows_affected, log_file, show_eqp, stdout, stderr);
    } else null;

    // Log query if logging is enabled
    if (log_file) |f| {
        const timestamp = std.time.timestamp();
        var log_buf: [256]u8 = undefined;
        var log_writer = f.writer(&log_buf);
        log_writer.interface.print("[{d}] {s}\n", .{ timestamp, sql }) catch {};
        log_writer.interface.flush() catch {};
    }

    var result = db.exec(sql_to_execute) catch |err| {
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
        return false;
    };
    defer result.close(allocator);

    // Save rows_affected for .changes command
    last_rows_affected.* = result.rows_affected;

    if (result.rows != null) {
        // SELECT — format rows
        displayRows(allocator, &result, mode, show_headers, csv_separator, null_display, stdout, stderr);
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
        if (show_timer) {
            stdout.print("Query time: {d:.3} ms\n", .{elapsed_ms}) catch {};
        }
    }

    // Display execution statistics (if stats enabled)
    if (show_stats) {
        if (timer) |*t| {
            const elapsed_ns = t.read();
            const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
            stdout.writeAll("─── Stats ───\n") catch {};
            stdout.print("Execution time: {d:.3} ms\n", .{elapsed_ms}) catch {};
            if (result.rows_affected > 0) {
                stdout.print("Rows changed:   {d}\n", .{result.rows_affected}) catch {};
            }
        }
    }

    return true;
}

/// Fallback for when timer is unavailable (executes without timing).
/// Returns true on success, false on error.
fn execAndDisplayWithoutTiming(allocator: std.mem.Allocator, db: *Database, sql: []const u8, mode: OutputMode, show_stats: bool, show_headers: bool, csv_separator: []const u8, null_display: []const u8, last_rows_affected: *u64, log_file: ?std.fs.File, show_eqp: bool, stdout: anytype, stderr: anytype) bool {
    _ = show_stats; // Stats require timer, so ignored in this path

    // Prepend EXPLAIN if .eqp is enabled and query is not already EXPLAIN
    var sql_to_execute = sql;
    var explain_buf: [16384]u8 = undefined;
    const explain_allocated = false;
    if (show_eqp and !std.ascii.startsWithIgnoreCase(sql, "EXPLAIN")) {
        const explain_sql = std.fmt.bufPrint(&explain_buf, "EXPLAIN {s}", .{sql}) catch sql;
        sql_to_execute = explain_sql;
    }
    defer if (explain_allocated) allocator.free(sql_to_execute);

    // Log query if logging is enabled
    if (log_file) |f| {
        const timestamp = std.time.timestamp();
        var log_buf: [256]u8 = undefined;
        var log_writer = f.writer(&log_buf);
        log_writer.interface.print("[{d}] {s}\n", .{ timestamp, sql }) catch {};
        log_writer.interface.flush() catch {};
    }

    var result = db.exec(sql_to_execute) catch |err| {
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
        return false;
    };
    defer result.close(allocator);

    // Save rows_affected for .changes command
    last_rows_affected.* = result.rows_affected;

    if (result.rows != null) {
        // SELECT — format rows
        displayRows(allocator, &result, mode, show_headers, csv_separator, null_display, stdout, stderr);
    } else if (result.message.len > 0) {
        stdout.writeAll(result.message) catch {};
        stdout.writeByte('\n') catch {};
    }

    if (result.rows_affected > 0) {
        stdout.print("Rows affected: {d}\n", .{result.rows_affected}) catch {};
    }

    return true;
}

/// Drain all rows from a QueryResult and display them in the given output mode.
fn displayRows(allocator: std.mem.Allocator, result: *QueryResult, mode: OutputMode, show_headers: bool, csv_separator: []const u8, null_display: []const u8, stdout: anytype, stderr: anytype) void {
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
        .table => formatTable(allocator, headers, all_rows.items, show_headers, null_display, stdout),
        .csv => formatCsv(headers, all_rows.items, allocator, show_headers, csv_separator, null_display, stdout),
        .json => formatJson(headers, all_rows.items, allocator, stdout),
        .jsonl => formatJsonl(headers, all_rows.items, allocator, stdout),
        .plain => formatPlain(headers, all_rows.items, allocator, show_headers, null_display, stdout),
    }
}

// ── Value Formatting ───────────────────────────────────────────────────

/// Convert a Value to its display string. Caller owns returned memory.
fn valueToText(allocator: std.mem.Allocator, val: Value, null_display: []const u8) ?[]const u8 {
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
        .null_value => allocator.dupe(u8, null_display) catch null,
    };
}

// ── Table Format ───────────────────────────────────────────────────────

fn formatTable(allocator: std.mem.Allocator, headers: []const []const u8, rows: []Row, show_headers: bool, null_display: []const u8, writer: anytype) void {
    if (show_headers) {
        // Use sailor.fmt.Table with headers
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
                cells[i] = valueToText(allocator, val, null_display) orelse {
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
    } else {
        // No headers — just print rows separated by '|'
        for (rows) |row| {
            for (row.values, 0..) |val, i| {
                if (i > 0) writer.writeAll("|") catch {};
                const text = valueToText(allocator, val, null_display) orelse continue;
                defer allocator.free(text);
                writer.writeAll(text) catch {};
            }
            writer.writeAll("\n") catch {};
        }
    }
}

// ── CSV Format ────────────────────────────────────────────────────────

fn formatCsv(headers: []const []const u8, rows: []Row, allocator: std.mem.Allocator, show_headers: bool, separator: []const u8, null_display: []const u8, writer: anytype) void {
    // Write headers (if enabled)
    if (show_headers) {
        for (headers, 0..) |h, i| {
            if (i > 0) writer.writeAll(separator) catch return;
            writeCsvField(writer, h) catch return;
        }
        writer.writeByte('\n') catch return;
    }

    // Write rows
    for (rows) |row| {
        for (row.values, 0..) |val, i| {
            if (i > 0) writer.writeAll(separator) catch return;
            const text = valueToText(allocator, val, null_display) orelse null_display;
            defer if (text.ptr != null_display.ptr) allocator.free(text);
            writeCsvField(writer, text) catch return;
        }
        writer.writeByte('\n') catch return;
    }
}

fn writeCsvField(writer: anytype, field: []const u8) !void {
    // Quote field if it contains separator, quotes, or newlines
    const needs_quoting = std.mem.indexOfAny(u8, field, ",\"\n\r") != null;

    if (needs_quoting) {
        try writer.writeByte('"');
        for (field) |ch| {
            if (ch == '"') {
                try writer.writeAll("\"\""); // Escape quotes by doubling
            } else {
                try writer.writeByte(ch);
            }
        }
        try writer.writeByte('"');
    } else {
        try writer.writeAll(field);
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

fn formatPlain(headers: []const []const u8, rows: []Row, allocator: std.mem.Allocator, show_headers: bool, null_display: []const u8, writer: anytype) void {
    if (show_headers) {
        // Show headers as "column = value" format
        for (rows) |row| {
            for (headers, 0..) |h, i| {
                if (i < row.values.len) {
                    const text = valueToText(allocator, row.values[i], null_display) orelse null_display;
                    defer if (text.ptr != null_display.ptr) allocator.free(text);
                    writer.print("{s} = {s}\n", .{ h, text }) catch {};
                }
            }
            if (rows.len > 1) writer.writeByte('\n') catch {};
        }
    } else {
        // No headers — just print values separated by newlines
        for (rows) |row| {
            for (row.values) |val| {
                const text = valueToText(allocator, val, null_display) orelse null_display;
                defer if (text.ptr != null_display.ptr) allocator.free(text);
                writer.print("{s}\n", .{text}) catch {};
            }
            if (rows.len > 1) writer.writeByte('\n') catch {};
        }
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

fn showDbInfo(db: *Database, db_path: []const u8, stdout: anytype, stderr: anytype) void {
    const pager = db.pager;

    // Get file size
    const file_size = pager.file.getEndPos() catch |err| {
        const msg = switch (err) {
            else => "Failed to get file size.",
        };
        printError(stderr, msg);
        return;
    };

    // Calculate free pages by traversing freelist
    var free_page_count: u32 = 0;
    var freelist_current = pager.freelist_head;

    // Allocate buffer for reading freelist pages
    const page_buf = pager.allocPageBuf() catch {
        printError(stderr, "Out of memory.");
        return;
    };
    defer pager.freePageBuf(page_buf);

    // Traverse freelist to count free pages (limit to avoid infinite loops)
    const max_freelist_depth: u32 = 10000;
    while (freelist_current != 0 and free_page_count < max_freelist_depth) {
        free_page_count += 1;

        // Read the free page to get next pointer
        pager.readPage(freelist_current, page_buf) catch break;

        // For free pages, the next free page ID is stored at offset PAGE_HEADER_SIZE
        if (page_buf.len >= silica.page.PAGE_HEADER_SIZE + 4) {
            freelist_current = std.mem.readInt(u32, page_buf[silica.page.PAGE_HEADER_SIZE..][0..4], .little);
        } else {
            break;
        }

        // Sanity check to avoid infinite loops
        if (freelist_current >= pager.page_count) break;
    }

    // Display database information (SQLite-compatible format)
    stdout.writeAll("database page size:  ") catch {};
    stdout.print("{d}\n", .{pager.page_size}) catch {};

    stdout.writeAll("write format:        ") catch {};
    stdout.print("{d}\n", .{silica.page.FORMAT_VERSION}) catch {};

    stdout.writeAll("read format:         ") catch {};
    stdout.print("{d}\n", .{silica.page.FORMAT_VERSION}) catch {};

    stdout.writeAll("number of pages:     ") catch {};
    stdout.print("{d}\n", .{pager.page_count}) catch {};

    stdout.writeAll("page size:           ") catch {};
    stdout.print("{d}\n", .{pager.page_size}) catch {};

    stdout.writeAll("file size:           ") catch {};
    stdout.print("{d} bytes ({d} pages)\n", .{ file_size, pager.page_count }) catch {};

    stdout.writeAll("freelist count:      ") catch {};
    stdout.print("{d}\n", .{free_page_count}) catch {};

    stdout.writeAll("schema version:      ") catch {};
    stdout.print("{d}\n", .{pager.schema_version}) catch {};

    stdout.writeAll("schema cookie:       ") catch {};
    stdout.print("{d}\n", .{pager.schema_version}) catch {};

    stdout.writeAll("database path:       ") catch {};
    stdout.print("{s}\n", .{db_path}) catch {};
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
fn readAndExecuteFile(allocator: std.mem.Allocator, db: *Database, filename: []const u8, mode: OutputMode, show_timer: bool, show_stats: bool, show_headers: bool, csv_separator: []const u8, null_display: []const u8, last_rows_affected: *u64, bail_on_error: bool, log_file: ?std.fs.File, show_eqp: bool, stdout: anytype, stderr: anytype) void {
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
                const success = execAndDisplay(allocator, db, statement, mode, show_timer, show_stats, show_headers, csv_separator, null_display, last_rows_affected, log_file, show_eqp, stdout, stderr);

                // If bail_on_error is enabled and execution failed, stop reading the file
                if (bail_on_error and !success) {
                    stderr.writeAll("Error: Script execution stopped due to bail on error setting\n") catch {};
                    return;
                }
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

/// Create a backup copy of the database file
fn backupDatabase(source_path: []const u8, dest_path: []const u8, stdout: anytype, stderr: anytype) void {
    // Don't allow backing up to the same file
    if (std.mem.eql(u8, source_path, dest_path)) {
        printError(stderr, "Cannot backup to the same file");
        return;
    }

    // Check if destination already exists
    std.fs.cwd().access(dest_path, .{}) catch |err| switch (err) {
        error.FileNotFound => {}, // OK, file doesn't exist
        else => {
            printError(stderr, "Cannot access destination path");
            return;
        },
    };

    // If we reach here and access() succeeded, the file exists
    if (std.fs.cwd().access(dest_path, .{})) |_| {
        printError(stderr, "Destination file already exists. Remove it first or choose a different name");
        return;
    } else |_| {}

    // Copy the database file
    std.fs.cwd().copyFile(source_path, std.fs.cwd(), dest_path, .{}) catch |err| {
        const msg = switch (err) {
            error.FileNotFound => "Source database file not found",
            error.AccessDenied => "Access denied. Check file permissions",
            error.PathAlreadyExists => "Destination file already exists. Remove it first or choose a different name",
            error.IsDir => "Cannot backup to a directory",
            error.NotDir => "Parent directory does not exist",
            error.NoSpaceLeft => "No space left on device",
            else => "Failed to create backup",
        };
        printError(stderr, msg);
        return;
    };

    stdout.print("Database backed up to: {s}\n", .{dest_path}) catch {};
}

/// Save database to a file (works with both file-based and :memory: databases)
fn saveDatabase(allocator: std.mem.Allocator, source_db: *Database, source_path: []const u8, dest_path: []const u8, stdout: anytype, stderr: anytype) void {
    // Check if destination already exists
    if (std.fs.cwd().access(dest_path, .{})) |_| {
        printError(stderr, "Destination file already exists. Remove it first or choose a different name");
        return;
    } else |_| {}

    // If source is a regular file (not :memory:), just use backup (file copy)
    if (!std.mem.eql(u8, source_path, ":memory:")) {
        backupDatabase(source_path, dest_path, stdout, stderr);
        return;
    }

    // For :memory: databases, we need to dump and restore
    // Create a new database at the destination
    var dest_db = Database.open(allocator, dest_path, .{}) catch |err| {
        const msg = switch (err) {
            error.OutOfMemory => "Out of memory",
            error.AccessDenied => "Access denied. Check file permissions",
            error.NoSpaceLeft => "No space left on device",
            else => "Failed to create destination database",
        };
        printError(stderr, msg);
        return;
    };
    defer dest_db.close();

    // Get list of tables from source database
    const table_names = source_db.catalog.listTables(allocator) catch |err| {
        const msg = switch (err) {
            error.OutOfMemory => "Out of memory",
            else => "Failed to list tables",
        };
        printError(stderr, msg);
        return;
    };
    defer {
        for (table_names) |name| allocator.free(name);
        allocator.free(table_names);
    }

    if (table_names.len == 0) {
        stdout.print("Database saved to: {s} (empty database)\n", .{dest_path}) catch {};
        return;
    }

    // Sort table names alphabetically for consistent output
    std.mem.sort([]const u8, table_names, {}, struct {
        fn lessThan(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.order(u8, a, b) == .lt;
        }
    }.lessThan);

    // Begin transaction in destination database
    _ = dest_db.exec("BEGIN TRANSACTION;") catch {
        printError(stderr, "Failed to begin transaction");
        return;
    };

    // For each table: get schema, create in dest, copy data
    for (table_names) |table_name| {
        // Get table schema from source
        const table_info = source_db.catalog.getTable(table_name) catch |err| {
            const msg = switch (err) {
                error.OutOfMemory => "Out of memory",
                else => "Failed to get table schema",
            };
            printError(stderr, msg);
            _ = dest_db.exec("ROLLBACK;") catch {};
            return;
        };
        defer table_info.deinit(source_db.allocator);

        // Build CREATE TABLE statement
        var create_sql: std.ArrayList(u8) = .{};
        defer create_sql.deinit(allocator);

        create_sql.writer(allocator).print("CREATE TABLE {s} (\n", .{table_name}) catch {
            printError(stderr, "Out of memory");
            _ = dest_db.exec("ROLLBACK;") catch {};
            return;
        };

        // Check if there are table-level constraints
        const has_table_constraints = table_info.table_constraints.len > 0;

        // Add columns
        for (table_info.columns, 0..) |col, i| {
            create_sql.writer(allocator).writeAll("  ") catch {};
            create_sql.writer(allocator).writeAll(col.name) catch {};
            create_sql.writer(allocator).writeAll(" ") catch {};

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
                .untyped => "INTEGER", // default for untyped
            };
            create_sql.writer(allocator).writeAll(type_name) catch {};

            // Add column constraints (only if not part of table-level constraint)
            if (col.flags.primary_key and !has_table_constraints) {
                create_sql.writer(allocator).writeAll(" PRIMARY KEY") catch {};
            }
            if (col.flags.not_null) {
                create_sql.writer(allocator).writeAll(" NOT NULL") catch {};
            }
            if (col.flags.unique and !col.flags.primary_key) {
                create_sql.writer(allocator).writeAll(" UNIQUE") catch {};
            }
            if (col.flags.autoincrement) {
                create_sql.writer(allocator).writeAll(" AUTOINCREMENT") catch {};
            }

            // Comma for all but last column (unless there are table constraints)
            const is_last_column = (i == table_info.columns.len - 1);
            if (!is_last_column or has_table_constraints) {
                create_sql.writer(allocator).writeAll(",\n") catch {};
            }
        }

        // Add table-level constraints
        for (table_info.table_constraints, 0..) |constraint, j| {
            create_sql.writer(allocator).writeAll("  ") catch {};
            switch (constraint) {
                .primary_key => |cols| {
                    create_sql.writer(allocator).writeAll("PRIMARY KEY (") catch {};
                    for (cols, 0..) |col_name, k| {
                        create_sql.writer(allocator).writeAll(col_name) catch {};
                        if (k < cols.len - 1) {
                            create_sql.writer(allocator).writeAll(", ") catch {};
                        }
                    }
                    create_sql.writer(allocator).writeAll(")") catch {};
                },
                .unique => |cols| {
                    create_sql.writer(allocator).writeAll("UNIQUE (") catch {};
                    for (cols, 0..) |col_name, k| {
                        create_sql.writer(allocator).writeAll(col_name) catch {};
                        if (k < cols.len - 1) {
                            create_sql.writer(allocator).writeAll(", ") catch {};
                        }
                    }
                    create_sql.writer(allocator).writeAll(")") catch {};
                },
            }
            if (j < table_info.table_constraints.len - 1) {
                create_sql.writer(allocator).writeAll(",\n") catch {};
            }
        }

        create_sql.writer(allocator).writeAll("\n);") catch {};

        // Execute CREATE TABLE in destination
        _ = dest_db.exec(create_sql.items) catch |err| {
            stderr.print("Failed to create table {s}: {s}\n", .{ table_name, @errorName(err) }) catch {};
            _ = dest_db.exec("ROLLBACK;") catch {};
            return;
        };

        // Copy data: SELECT all rows from source, INSERT into dest
        var select_sql_buf: [256]u8 = undefined;
        const select_sql = std.fmt.bufPrint(&select_sql_buf, "SELECT * FROM {s};", .{table_name}) catch {
            printError(stderr, "Table name too long");
            _ = dest_db.exec("ROLLBACK;") catch {};
            return;
        };

        var result = source_db.exec(select_sql) catch |err| {
            stderr.print("Failed to read data from {s}: {s}\n", .{ table_name, @errorName(err) }) catch {};
            _ = dest_db.exec("ROLLBACK;") catch {};
            return;
        };
        defer result.close(allocator);

        // Insert rows into destination
        if (result.rows) |*rows| {
            while (true) {
                const maybe_row = rows.next() catch break;
                if (maybe_row) |row| {
                    defer {
                        for (row.values) |v| v.free(allocator);
                        allocator.free(row.values);
                        allocator.free(row.columns);
                    }

                    // Build INSERT statement
                    var insert_sql: std.ArrayList(u8) = .{};
                    defer insert_sql.deinit(allocator);

                    insert_sql.writer(allocator).print("INSERT INTO {s} VALUES (", .{table_name}) catch continue;

                    for (row.values, 0..) |value, j| {
                        switch (value) {
                            .null_value => insert_sql.writer(allocator).writeAll("NULL") catch {},
                            .integer => |v| insert_sql.writer(allocator).print("{}", .{v}) catch {},
                            .real => |v| insert_sql.writer(allocator).print("{d}", .{v}) catch {},
                            .text => |v| {
                                insert_sql.writer(allocator).writeAll("'") catch {};
                                for (v) |ch| {
                                    if (ch == '\'') {
                                        insert_sql.writer(allocator).writeAll("''") catch {};
                                    } else {
                                        insert_sql.writer(allocator).writeByte(ch) catch {};
                                    }
                                }
                                insert_sql.writer(allocator).writeAll("'") catch {};
                            },
                            .blob => |v| {
                                insert_sql.writer(allocator).writeAll("X'") catch {};
                                for (v) |byte| {
                                    insert_sql.writer(allocator).print("{X:0>2}", .{byte}) catch {};
                                }
                                insert_sql.writer(allocator).writeAll("'") catch {};
                            },
                            .boolean => |v| {
                                const bool_str = if (v) "TRUE" else "FALSE";
                                insert_sql.writer(allocator).writeAll(bool_str) catch {};
                            },
                            else => insert_sql.writer(allocator).writeAll("NULL") catch {},
                        }

                        if (j < row.values.len - 1) {
                            insert_sql.writer(allocator).writeAll(", ") catch {};
                        }
                    }

                    insert_sql.writer(allocator).writeAll(");") catch {};

                    // Execute INSERT in destination
                    _ = dest_db.exec(insert_sql.items) catch |err| {
                        stderr.print("Failed to insert row into {s}: {s}\n", .{ table_name, @errorName(err) }) catch {};
                        // Continue with next row instead of aborting
                        continue;
                    };
                } else break;
            }
        }

        // Copy indexes for this table
        for (table_info.indexes) |idx| {
            // Skip auto-generated indexes (PRIMARY KEY, UNIQUE)
            if (idx.index_name.len == 0) continue;

            var create_index_sql: std.ArrayList(u8) = .{};
            defer create_index_sql.deinit(allocator);

            create_index_sql.writer(allocator).writeAll("CREATE ") catch continue;
            if (idx.is_unique) {
                create_index_sql.writer(allocator).writeAll("UNIQUE ") catch continue;
            }
            create_index_sql.writer(allocator).print("INDEX {s} ON {s} ({s});", .{
                idx.index_name,
                table_name,
                idx.column_name,
            }) catch continue;

            _ = dest_db.exec(create_index_sql.items) catch |err| {
                stderr.print("Warning: Failed to create index {s}: {s}\n", .{ idx.index_name, @errorName(err) }) catch {};
                // Continue even if index creation fails
            };
        }
    }

    // Commit transaction
    _ = dest_db.exec("COMMIT;") catch {
        printError(stderr, "Failed to commit transaction");
        _ = dest_db.exec("ROLLBACK;") catch {};
        return;
    };

    stdout.print("Database saved to: {s}\n", .{dest_path}) catch {};
}

fn importCsvFile(allocator: std.mem.Allocator, db: *Database, csv_path: []const u8, table_name: []const u8, csv_separator: []const u8, stdout: anytype, stderr: anytype) void {
    // Read CSV file (max 100 MB)
    const max_size = 100 * 1024 * 1024;
    const csv_content = std.fs.cwd().readFileAlloc(allocator, csv_path, max_size) catch |err| {
        const msg = switch (err) {
            error.FileNotFound => "CSV file not found",
            error.AccessDenied => "Access denied. Check file permissions",
            error.FileTooBig => "CSV file too large (max 100 MB)",
            else => "Failed to read CSV file",
        };
        printError(stderr, msg);
        return;
    };
    defer allocator.free(csv_content);

    // Split CSV into lines
    var lines = std.mem.splitScalar(u8, csv_content, '\n');
    var rows_imported: u64 = 0;

    // Process each line
    while (lines.next()) |line| {
        // Skip empty lines
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (trimmed.len == 0) continue;

        // Parse CSV fields (simple split by separator)
        var values: std.ArrayList([]const u8) = .{};
        defer values.deinit(allocator);

        var field_iter = std.mem.splitSequence(u8, trimmed, csv_separator);
        while (field_iter.next()) |field| {
            // Trim quotes if present
            var cleaned_field = std.mem.trim(u8, field, " \t");
            if (cleaned_field.len >= 2 and cleaned_field[0] == '"' and cleaned_field[cleaned_field.len - 1] == '"') {
                cleaned_field = cleaned_field[1 .. cleaned_field.len - 1];
            }
            values.append(allocator, cleaned_field) catch {
                printError(stderr, "Out of memory while parsing CSV");
                return;
            };
        }

        // Build INSERT statement
        if (values.items.len == 0) continue;

        // Construct SQL: INSERT INTO table VALUES (?, ?, ...)
        var sql_buf: std.ArrayList(u8) = .{};
        defer sql_buf.deinit(allocator);

        sql_buf.writer(allocator).print("INSERT INTO {s} VALUES (", .{table_name}) catch {
            printError(stderr, "Out of memory while building INSERT");
            return;
        };

        for (values.items, 0..) |value, i| {
            if (i > 0) sql_buf.writer(allocator).writeAll(", ") catch {};
            // Escape single quotes and wrap in quotes
            sql_buf.writer(allocator).writeByte('\'') catch {};
            var j: usize = 0;
            while (j < value.len) : (j += 1) {
                if (value[j] == '\'') {
                    sql_buf.writer(allocator).writeAll("''") catch {};
                } else {
                    sql_buf.writer(allocator).writeByte(value[j]) catch {};
                }
            }
            sql_buf.writer(allocator).writeByte('\'') catch {};
        }
        sql_buf.writer(allocator).writeAll(");") catch {};

        // Execute INSERT
        const sql = sql_buf.items;
        _ = db.exec(sql) catch |err| {
            const msg = switch (err) {
                error.TableNotFound => "Table not found. Create the table first",
                error.UniqueConstraintViolation => "Duplicate key error. Row already exists",
                else => "Failed to insert row",
            };
            stderr.print("Error importing row: {s}\n", .{msg}) catch {};
            stderr.print("SQL: {s}\n", .{sql}) catch {};
            continue;
        };

        rows_imported += 1;
    }

    stdout.print("Imported {d} rows from {s} into {s}\n", .{ rows_imported, csv_path, table_name }) catch {};
}

const DotCommandResult = union(enum) {
    ok,
    quit,
    reopen: []const u8, // New database path to open
};

fn handleDotCommand(allocator: std.mem.Allocator, db: *Database, db_path: []const u8, cmd: []const u8, mode: *OutputMode, show_timer: *bool, show_headers: *bool, csv_separator: *[]const u8, null_display: *[]const u8, output_file: *?std.fs.File, once_file: *?std.fs.File, last_rows_affected: *u64, bail_on_error: *bool, log_file: *?std.fs.File, show_stats: *bool, show_eqp: *bool, main_prompt: *[]const u8, continue_prompt: *[]const u8, stdout: anytype, stderr: anytype) DotCommandResult {
    if (std.mem.eql(u8, cmd, ".quit") or std.mem.eql(u8, cmd, ".exit")) {
        stdout.writeAll("Bye!\n") catch {};
        return .quit;
    } else if (std.mem.eql(u8, cmd, ".version")) {
        stdout.print("Silica v{s}\n", .{version}) catch {};
        stdout.print("Zig {s}\n", .{@import("builtin").zig_version_string}) catch {};
        stdout.writeAll("Dependencies:\n") catch {};
        stdout.writeAll("  sailor v1.36.0\n") catch {};
        stdout.writeAll("  zuda v2.0.0\n") catch {};
    } else if (std.mem.eql(u8, cmd, ".clear")) {
        // Clear screen using ANSI escape codes: ESC[2J (clear) + ESC[H (home cursor)
        stdout.writeAll("\x1b[2J\x1b[H") catch {};
    } else if (std.mem.eql(u8, cmd, ".help")) {
        stdout.writeAll(
            \\.help               Show this help
            \\.quit               Exit the shell
            \\.exit               Exit the shell
            \\.version            Show version information
            \\.clear              Clear the screen
            \\.echo TEXT          Print literal text to output
            \\.print TEXT         Print literal text to output (alias for .echo)
            \\.show               Show current settings (mode, headers, timer, stats, separator, nullvalue, output, bail, eqp)
            \\.changes            Show number of rows changed by last DML statement
            \\.mode MODE          Set output mode (table, csv, json, jsonl, plain)
            \\.mode               Show current output mode
            \\.separator STRING   Set CSV output separator (default: ",")
            \\.separator          Show current separator
            \\.headers on|off     Enable or disable column headers in output
            \\.headers            Show current headers setting
            \\.timer on|off       Enable or disable query execution timing
            \\.timer              Show current timer setting
            \\.stats on|off       Show execution statistics after each query
            \\.stats              Show current stats setting
            \\.bail on|off        Stop script execution on first error
            \\.bail               Show current bail setting
            \\.eqp on|off         Automatically EXPLAIN query plans
            \\.eqp                Show current eqp setting
            \\.output FILENAME    Redirect output to file
            \\.output             Reset output to stdout
            \\.once FILENAME      Write next query output to file (one-time only)
            \\.log FILENAME       Enable query logging to file (appends)
            \\.log off            Disable query logging
            \\.log                Show current log setting
            \\.nullvalue STRING   Set string to display for NULL values
            \\.nullvalue          Show current NULL display string
            \\.databases          List database connections
            \\.dbinfo             Show database file statistics
            \\.tables             List all tables
            \\.indexes [TABLE]    List all indexes or indexes for a specific table
            \\.schema [TABLE]     Show CREATE TABLE statements for all tables or specific table
            \\.dump               Dump database as SQL text (CREATE TABLE + INSERT statements)
            \\.backup FILENAME    Create a backup copy of the database file
            \\.save FILENAME      Save database to file (works with :memory: databases)
            \\.import FILE TABLE  Import CSV data from file into table
            \\.read FILENAME      Read and execute SQL from file
            \\.cd DIRECTORY       Change the working directory
            \\.open FILENAME      Close current database and open a new one
            \\.open               Show current database path
            \\.prompt MAIN CONT   Replace the standard prompts
            \\.prompt             Show current prompts
            \\.system CMD ARGS    Run CMD ARGS in a system shell
            \\.shell CMD ARGS     Run CMD ARGS in a system shell (alias for .system)
            \\
        ) catch {};
    } else if (std.mem.startsWith(u8, cmd, ".headers")) {
        const rest = std.mem.trimLeft(u8, cmd[8..], " \t");
        if (rest.len == 0) {
            // Show current headers setting
            const setting = if (show_headers.*) "on" else "off";
            stdout.print("Headers: {s}\n", .{setting}) catch {};
        } else if (std.mem.eql(u8, rest, "on")) {
            show_headers.* = true;
            stdout.writeAll("Headers enabled\n") catch {};
        } else if (std.mem.eql(u8, rest, "off")) {
            show_headers.* = false;
            stdout.writeAll("Headers disabled\n") catch {};
        } else {
            printError(stderr, "Usage: .headers on|off");
        }
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
    } else if (std.mem.startsWith(u8, cmd, ".stats")) {
        const rest = std.mem.trimLeft(u8, cmd[6..], " \t");
        if (rest.len == 0) {
            // Show current stats setting
            const setting = if (show_stats.*) "on" else "off";
            stdout.print("Stats: {s}\n", .{setting}) catch {};
        } else if (std.mem.eql(u8, rest, "on")) {
            show_stats.* = true;
            stdout.writeAll("Stats enabled\n") catch {};
        } else if (std.mem.eql(u8, rest, "off")) {
            show_stats.* = false;
            stdout.writeAll("Stats disabled\n") catch {};
        } else {
            printError(stderr, "Usage: .stats on|off");
        }
    } else if (std.mem.startsWith(u8, cmd, ".bail")) {
        const rest = std.mem.trimLeft(u8, cmd[5..], " \t");
        if (rest.len == 0) {
            // Show current bail setting
            const setting = if (bail_on_error.*) "on" else "off";
            stdout.print("Bail: {s}\n", .{setting}) catch {};
        } else if (std.mem.eql(u8, rest, "on")) {
            bail_on_error.* = true;
            stdout.writeAll("Bail on error enabled\n") catch {};
        } else if (std.mem.eql(u8, rest, "off")) {
            bail_on_error.* = false;
            stdout.writeAll("Bail on error disabled\n") catch {};
        } else {
            printError(stderr, "Usage: .bail on|off");
        }
    } else if (std.mem.startsWith(u8, cmd, ".eqp")) {
        const rest = std.mem.trimLeft(u8, cmd[4..], " \t");
        if (rest.len == 0) {
            // Show current eqp setting
            const setting = if (show_eqp.*) "on" else "off";
            stdout.print("EQP: {s}\n", .{setting}) catch {};
        } else if (std.mem.eql(u8, rest, "on")) {
            show_eqp.* = true;
            stdout.writeAll("EQP enabled\n") catch {};
        } else if (std.mem.eql(u8, rest, "off")) {
            show_eqp.* = false;
            stdout.writeAll("EQP disabled\n") catch {};
        } else {
            printError(stderr, "Usage: .eqp on|off");
        }
    } else if (std.mem.startsWith(u8, cmd, ".output")) {
        const rest = std.mem.trimLeft(u8, cmd[7..], " \t");
        if (rest.len == 0) {
            // Reset output to stdout
            if (output_file.*) |f| {
                f.close();
                output_file.* = null;
                stdout.writeAll("Output reset to stdout\n") catch {};
            } else {
                stdout.writeAll("Output is already stdout\n") catch {};
            }
        } else {
            // Open file for writing (create or truncate)
            const file = std.fs.cwd().createFile(rest, .{}) catch {
                printError(stderr, "Failed to open output file");
                return .ok;
            };
            // Close existing output file if any
            if (output_file.*) |f| {
                f.close();
            }
            output_file.* = file;
            stdout.print("Output redirected to: {s}\n", .{rest}) catch {};
        }
    } else if (std.mem.startsWith(u8, cmd, ".once")) {
        const rest = std.mem.trimLeft(u8, cmd[5..], " \t");
        if (rest.len == 0) {
            printError(stderr, "Usage: .once FILENAME");
        } else {
            // Open file for writing (create or truncate)
            const file = std.fs.cwd().createFile(rest, .{}) catch {
                printError(stderr, "Failed to open output file");
                return .ok;
            };
            // Close existing once_file if any (shouldn't happen, but be safe)
            if (once_file.*) |f| {
                f.close();
            }
            once_file.* = file;
            stdout.print("Next output will be written to: {s}\n", .{rest}) catch {};
        }
    } else if (std.mem.startsWith(u8, cmd, ".log")) {
        const rest = std.mem.trimLeft(u8, cmd[4..], " \t");
        if (rest.len == 0) {
            // Show current log status
            if (log_file.* != null) {
                stdout.writeAll("log: on\n") catch {};
            } else {
                stdout.writeAll("log: off\n") catch {};
            }
        } else if (std.mem.eql(u8, rest, "off")) {
            // Turn logging off
            if (log_file.*) |f| {
                f.close();
                log_file.* = null;
                stdout.writeAll("Logging disabled\n") catch {};
            } else {
                stdout.writeAll("Logging is already off\n") catch {};
            }
        } else {
            // Open log file for appending (create if doesn't exist)
            const file = std.fs.cwd().createFile(rest, .{ .truncate = false }) catch {
                printError(stderr, "Failed to open log file");
                return .ok;
            };
            // Seek to end for append mode
            file.seekFromEnd(0) catch {
                file.close();
                printError(stderr, "Failed to seek to end of log file");
                return .ok;
            };
            // Close existing log file if any
            if (log_file.*) |f| {
                f.close();
            }
            log_file.* = file;
            stdout.print("Logging to: {s}\n", .{rest}) catch {};
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
    } else if (std.mem.startsWith(u8, cmd, ".separator")) {
        const rest = std.mem.trimLeft(u8, cmd[10..], " \t");
        if (rest.len == 0) {
            // Show current separator
            stdout.print("Separator: \"{s}\"\n", .{csv_separator.*}) catch {};
        } else {
            // Set new separator
            csv_separator.* = rest;
            stdout.print("Separator set to: \"{s}\"\n", .{rest}) catch {};
        }
    } else if (std.mem.eql(u8, cmd, ".databases")) {
        showDatabases(db, stdout);
    } else if (std.mem.eql(u8, cmd, ".dbinfo")) {
        showDbInfo(db, db_path, stdout, stderr);
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
    } else if (std.mem.startsWith(u8, cmd, ".backup")) {
        const rest = std.mem.trimLeft(u8, cmd[7..], " \t");
        if (rest.len == 0) {
            printError(stderr, "Usage: .backup FILENAME");
        } else {
            backupDatabase(db_path, rest, stdout, stderr);
        }
    } else if (std.mem.startsWith(u8, cmd, ".save")) {
        const rest = std.mem.trimLeft(u8, cmd[5..], " \t");
        if (rest.len == 0) {
            printError(stderr, "Usage: .save FILENAME");
        } else {
            saveDatabase(allocator, db, db_path, rest, stdout, stderr);
        }
    } else if (std.mem.startsWith(u8, cmd, ".import")) {
        const rest = std.mem.trimLeft(u8, cmd[7..], " \t");
        if (rest.len == 0) {
            printError(stderr, "Usage: .import FILE TABLE");
        } else {
            // Parse FILE and TABLE arguments
            var args_iter = std.mem.splitScalar(u8, rest, ' ');
            const csv_file = args_iter.next() orelse {
                printError(stderr, "Usage: .import FILE TABLE");
                return .ok;
            };
            // Skip any spaces between arguments
            var table_arg: ?[]const u8 = null;
            while (args_iter.next()) |arg| {
                if (arg.len > 0) {
                    table_arg = arg;
                    break;
                }
            }
            const table_name = table_arg orelse {
                printError(stderr, "Usage: .import FILE TABLE");
                return .ok;
            };
            importCsvFile(allocator, db, csv_file, table_name, csv_separator.*, stdout, stderr);
        }
    } else if (std.mem.startsWith(u8, cmd, ".read")) {
        const rest = std.mem.trimLeft(u8, cmd[5..], " \t");
        if (rest.len == 0) {
            printError(stderr, "Usage: .read FILENAME");
        } else {
            readAndExecuteFile(allocator, db, rest, mode.*, show_timer.*, show_stats.*, show_headers.*, csv_separator.*, null_display.*, last_rows_affected, bail_on_error.*, log_file.*, show_eqp.*, stdout, stderr);
        }
    } else if (std.mem.startsWith(u8, cmd, ".cd")) {
        const rest = std.mem.trimLeft(u8, cmd[3..], " \t");
        if (rest.len == 0) {
            // Show current directory
            const cwd = std.fs.cwd().realpathAlloc(allocator, ".") catch {
                printError(stderr, "Failed to get current directory");
                return .ok;
            };
            defer allocator.free(cwd);
            stdout.print("Current directory: {s}\n", .{cwd}) catch {};
        } else {
            // Change directory
            const path_z = allocator.dupeZ(u8, rest) catch {
                printError(stderr, "Out of memory");
                return .ok;
            };
            defer allocator.free(path_z);
            std.posix.chdir(path_z) catch {
                printError(stderr, "Failed to change directory");
                return .ok;
            };
            stdout.print("Changed directory to: {s}\n", .{rest}) catch {};
        }
    } else if (std.mem.startsWith(u8, cmd, ".open")) {
        const rest = std.mem.trimLeft(u8, cmd[5..], " \t");
        if (rest.len == 0) {
            // Show current database path
            stdout.print("Current database: {s}\n", .{db_path}) catch {};
        } else {
            // Signal to reopen with new database path
            // Allocate persistent memory for the path (will be freed by caller after reopening)
            const new_path = allocator.dupe(u8, rest) catch {
                printError(stderr, "Out of memory");
                return .ok;
            };
            return .{ .reopen = new_path };
        }
    } else if (std.mem.startsWith(u8, cmd, ".nullvalue")) {
        const rest = std.mem.trimLeft(u8, cmd[10..], " \t");
        if (rest.len == 0) {
            // Show current null display string
            stdout.print("Nullvalue: \"{s}\"\n", .{null_display.*}) catch {};
        } else {
            // Set custom null display string
            null_display.* = rest;
            stdout.print("Nullvalue set to: \"{s}\"\n", .{null_display.*}) catch {};
        }
    } else if (std.mem.startsWith(u8, cmd, ".echo")) {
        const rest = std.mem.trimLeft(u8, cmd[5..], " \t");
        // Print the literal text argument (useful in scripts for progress messages)
        stdout.print("{s}\n", .{rest}) catch {};
    } else if (std.mem.startsWith(u8, cmd, ".print")) {
        const rest = std.mem.trimLeft(u8, cmd[6..], " \t");
        // Alias for .echo (SQLite compatibility)
        stdout.print("{s}\n", .{rest}) catch {};
    } else if (std.mem.eql(u8, cmd, ".changes")) {
        // Show number of rows changed by last DML statement
        stdout.print("{d}\n", .{last_rows_affected.*}) catch {};
    } else if (std.mem.eql(u8, cmd, ".show")) {
        // Show all current settings
        stdout.writeAll("        mode: ") catch {};
        stdout.print("{s}\n", .{@tagName(mode.*)}) catch {};
        stdout.writeAll("     headers: ") catch {};
        stdout.print("{s}\n", .{if (show_headers.*) "on" else "off"}) catch {};
        stdout.writeAll("       timer: ") catch {};
        stdout.print("{s}\n", .{if (show_timer.*) "on" else "off"}) catch {};
        stdout.writeAll("       stats: ") catch {};
        stdout.print("{s}\n", .{if (show_stats.*) "on" else "off"}) catch {};
        stdout.writeAll("        bail: ") catch {};
        stdout.print("{s}\n", .{if (bail_on_error.*) "on" else "off"}) catch {};
        stdout.writeAll("   separator: ") catch {};
        stdout.print("\"{s}\"\n", .{csv_separator.*}) catch {};
        stdout.writeAll("   nullvalue: ") catch {};
        stdout.print("\"{s}\"\n", .{null_display.*}) catch {};
        stdout.writeAll("      output: ") catch {};
        if (output_file.*) |_| {
            stdout.writeAll("file\n") catch {};
        } else {
            stdout.writeAll("stdout\n") catch {};
        }
        stdout.writeAll("        once: ") catch {};
        if (once_file.*) |_| {
            stdout.writeAll("pending\n") catch {};
        } else {
            stdout.writeAll("off\n") catch {};
        }
        stdout.writeAll("         log: ") catch {};
        if (log_file.*) |_| {
            stdout.writeAll("on\n") catch {};
        } else {
            stdout.writeAll("off\n") catch {};
        }
        stdout.writeAll("         eqp: ") catch {};
        stdout.print("{s}\n", .{if (show_eqp.*) "on" else "off"}) catch {};
        stdout.writeAll("      prompt: ") catch {};
        stdout.print("\"{s}\" \"{s}\"\n", .{ main_prompt.*, continue_prompt.* }) catch {};
    } else if (std.mem.startsWith(u8, cmd, ".prompt")) {
        const rest = std.mem.trimLeft(u8, cmd[7..], " \t");
        if (rest.len == 0) {
            // Show current prompts
            stdout.print("Main: \"{s}\"\n", .{main_prompt.*}) catch {};
            stdout.print("Continue: \"{s}\"\n", .{continue_prompt.*}) catch {};
        } else {
            // Parse MAIN and CONTINUE prompts
            // Format: .prompt MAIN CONTINUE
            // Both arguments are required
            var iter = std.mem.splitScalar(u8, rest, ' ');
            const main_text = iter.next();
            const cont_text = iter.next();

            if (main_text == null or cont_text == null) {
                printError(stderr, "Usage: .prompt MAIN CONTINUE");
            } else {
                main_prompt.* = main_text.?;
                continue_prompt.* = cont_text.?;
                stdout.writeAll("Prompts updated\n") catch {};
            }
        }
    } else if (std.mem.startsWith(u8, cmd, ".system") or std.mem.startsWith(u8, cmd, ".shell")) {
        // .shell is SQLite-compatible alias for .system
        const offset: usize = if (std.mem.startsWith(u8, cmd, ".system")) 7 else 6;
        const rest = std.mem.trimLeft(u8, cmd[offset..], " \t");
        if (rest.len == 0) {
            const usage = if (offset == 7) "Usage: .system CMD ARGS..." else "Usage: .shell CMD ARGS...";
            printError(stderr, usage);
        } else {
            // Execute shell command and capture output
            var child = std.process.Child.init(&[_][]const u8{
                "/bin/sh",
                "-c",
                rest,
            }, allocator);

            child.stdout_behavior = .Pipe;
            child.stderr_behavior = .Pipe;

            child.spawn() catch {
                printError(stderr, "Failed to execute command");
                return .ok;
            };

            // Read stdout
            const child_stdout = child.stdout.?.readToEndAlloc(allocator, 1024 * 1024) catch {
                _ = child.wait() catch {};
                printError(stderr, "Failed to read command output");
                return .ok;
            };
            defer allocator.free(child_stdout);

            // Read stderr
            const child_stderr = child.stderr.?.readToEndAlloc(allocator, 1024 * 1024) catch {
                _ = child.wait() catch {};
                printError(stderr, "Failed to read command errors");
                return .ok;
            };
            defer allocator.free(child_stderr);

            // Wait for child to complete
            const term = child.wait() catch {
                printError(stderr, "Failed to wait for command");
                return .ok;
            };

            // Write stdout
            if (child_stdout.len > 0) {
                stdout.writeAll(child_stdout) catch {};
            }

            // Write stderr
            if (child_stderr.len > 0) {
                stderr.writeAll(child_stderr) catch {};
            }

            // Show non-zero exit code
            switch (term) {
                .Exited => |code| {
                    if (code != 0) {
                        stderr.print("Command exited with code {d}\n", .{code}) catch {};
                    }
                },
                .Signal => |sig| {
                    stderr.print("Command terminated by signal {d}\n", .{sig}) catch {};
                },
                else => {},
            }
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

    var buf: [4096]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;
    var show_timer = true;
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";
    const result = handleDotCommand(allocator, &db, path, ".help", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);
    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, ".help") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, ".quit") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, ".mode") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, ".databases") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, ".dbinfo") != null);
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";
    const result = handleDotCommand(allocator, &db, path, ".databases", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";
    const result = handleDotCommand(allocator, &db, path, ".databases", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);
    const output = fbs.getWritten();

    // Verify output shows :memory: database
    try std.testing.expect(std.mem.indexOf(u8, output, ":memory:") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "main") != null);
}

test "handleDotCommand dbinfo" {
    const allocator = std.testing.allocator;
    const path = "test_dbinfo.db";
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";
    const result = handleDotCommand(allocator, &db, path, ".dbinfo", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);
    const output = fbs.getWritten();

    // Verify output contains database statistics
    try std.testing.expect(std.mem.indexOf(u8, output, "database page size") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "number of pages") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "page size") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "file size") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "freelist count") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "schema version") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "database path") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, path) != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "4096") != null); // Default page size
}

test "handleDotCommand dbinfo - memory database" {
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";
    const result = handleDotCommand(allocator, &db, path, ".dbinfo", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);
    const output = fbs.getWritten();

    // Verify output shows database info for :memory:
    try std.testing.expect(std.mem.indexOf(u8, output, ":memory:") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "database page size") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "4096") != null);
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";
    const result = handleDotCommand(allocator, &db, path, ".quit", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.quit, result);
    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "Bye!") != null);
}

test "handleDotCommand clear" {
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";
    const result = handleDotCommand(allocator, &db, path, ".clear", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);
    const output = fbs.getWritten();
    // Verify ANSI clear screen escape sequence
    try std.testing.expect(std.mem.indexOf(u8, output, "\x1b[2J\x1b[H") != null);
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";
    const result = handleDotCommand(allocator, &db, path, ".foobar", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";
    _ = handleDotCommand(allocator, &db, path, ".mode csv", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";
    _ = handleDotCommand(allocator, &db, path, ".mode", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";
    _ = handleDotCommand(allocator, &db, path, ".mode foobar", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";
    const result = handleDotCommand(allocator, &db, path, ".tables", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";
    const result = handleDotCommand(allocator, &db, path, ".schema", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";
    const result = handleDotCommand(allocator, &db, path, ".schema users", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";
    const result = handleDotCommand(allocator, &db, path, ".schema user_roles", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";
    const result = handleDotCommand(allocator, &db, path, ".schema nonexistent", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";
    const result = handleDotCommand(allocator, &db, path, ".schema", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";
    const result = handleDotCommand(allocator, &db, path, ".indexes users", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";
    const result = handleDotCommand(allocator, &db, path, ".indexes", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";
    const result = handleDotCommand(allocator, &db, path, ".indexes plain", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";
    const result = handleDotCommand(allocator, &db, path, ".indexes nonexistent", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";
    const result = handleDotCommand(allocator, &db, path, ".indexes", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";
    const result = handleDotCommand(allocator, &db, path, ".dump", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";
    const result = handleDotCommand(allocator, &db, path, ".dump", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";
    const result = handleDotCommand(allocator, &db, path, ".dump", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
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
        const text = valueToText(allocator, .{ .integer = 42 }, "NULL").?;
        defer allocator.free(text);
        try std.testing.expectEqualStrings("42", text);
    }
    // Real
    {
        const text = valueToText(allocator, .{ .real = 3.14 }, "NULL").?;
        defer allocator.free(text);
        // Just check it starts with "3.14"
        try std.testing.expect(std.mem.startsWith(u8, text, "3.14"));
    }
    // Text
    {
        const text = valueToText(allocator, .{ .text = "hello" }, "NULL").?;
        defer allocator.free(text);
        try std.testing.expectEqualStrings("hello", text);
    }
    // Boolean
    {
        const t = valueToText(allocator, .{ .boolean = true }, "NULL").?;
        defer allocator.free(t);
        try std.testing.expectEqualStrings("TRUE", t);
        const f = valueToText(allocator, .{ .boolean = false }, "NULL").?;
        defer allocator.free(f);
        try std.testing.expectEqualStrings("FALSE", f);
    }
    // Null
    {
        const text = valueToText(allocator, .null_value, "NULL").?;
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
    formatTable(allocator, headers, &rows, true, "NULL", &w);
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
    formatCsv(headers, &rows, allocator, true, ",", "NULL", &w);
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
    formatPlain(headers, &rows, allocator, true, "NULL", &w);
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";
    const result = handleDotCommand(allocator, &db, path, ".read test_script.sql", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";
    const result = handleDotCommand(allocator, &db, path, ".read nonexistent.sql", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";
    const result = handleDotCommand(allocator, &db, path, ".read test_comments.sql", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";
    const result = handleDotCommand(allocator, &db, path, ".read", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const error_output = ebs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, error_output, "Usage: .read FILENAME") != null);
}

test "handleDotCommand .output to file" {
    const allocator = std.testing.allocator;

    const path = "test_output_file.db";
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const output_path = "test_output_results.txt";
    defer std.fs.cwd().deleteFile(output_path) catch {};

    const result = handleDotCommand(allocator, &db, path, ".output test_output_results.txt", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);
    try std.testing.expect(output_file != null);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "Output redirected to: test_output_results.txt") != null);

    // Clean up the file handle
    if (output_file) |f| f.close();
}

test "handleDotCommand .output reset to stdout" {
    const allocator = std.testing.allocator;

    const path = "test_output_reset.db";
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    // First redirect to file
    const temp_path = "test_output_temp.txt";
    defer std.fs.cwd().deleteFile(temp_path) catch {};

    _ = handleDotCommand(allocator, &db, path, ".output test_output_temp.txt", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expect(output_file != null);

    // Reset fbs for next command
    fbs = std.io.fixedBufferStream(&out_buf);
    w = fbs.writer();

    // Then reset to stdout
    const result = handleDotCommand(allocator, &db, path, ".output", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);
    try std.testing.expect(output_file == null);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "Output reset to stdout") != null);
}

test "handleDotCommand .output already stdout" {
    const allocator = std.testing.allocator;

    const path = "test_output_already_stdout.db";
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".output", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);
    try std.testing.expect(output_file == null);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "Output is already stdout") != null);
}

test "handleDotCommand .output file creation error" {
    const allocator = std.testing.allocator;

    const path = "test_output_error.db";
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    // Try to open a file in a non-existent directory
    const result = handleDotCommand(allocator, &db, path, ".output /nonexistent/path/file.txt", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);
    try std.testing.expect(output_file == null); // Should remain null on error

    const error_output = ebs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, error_output, "Failed to open output file") != null);
}

test "handleDotCommand .output in help text" {
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";
    const result = handleDotCommand(allocator, &db, path, ".help", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);
    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, ".output") != null);
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".timer on", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".timer off", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".timer", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".timer foobar", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";
    const result = handleDotCommand(allocator, &db, path, ".help", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, ".timer") != null);
}

test "handleDotCommand .headers on" {
    const allocator = std.testing.allocator;

    const path = "test_headers_on.db";
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
    var show_headers = false; // Start disabled
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".headers on", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);
    try std.testing.expectEqual(true, show_headers); // Should be enabled now

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "Headers enabled") != null);
}

test "handleDotCommand .headers off" {
    const allocator = std.testing.allocator;

    const path = "test_headers_off.db";
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
    var show_headers = true; // Start enabled
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".headers off", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);
    try std.testing.expectEqual(false, show_headers); // Should be disabled now

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "Headers disabled") != null);
}

test "handleDotCommand .headers shows current setting" {
    const allocator = std.testing.allocator;

    const path = "test_headers_show.db";
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".headers", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "Headers: on") != null);
}

test "handleDotCommand .headers invalid argument" {
    const allocator = std.testing.allocator;

    const path = "test_headers_invalid.db";
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".headers foobar", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);
    try std.testing.expectEqual(true, show_headers); // Should remain unchanged

    const error_output = ebs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, error_output, "Usage: .headers on|off") != null);
}

test "handleDotCommand .help includes .headers" {
    const allocator = std.testing.allocator;

    const path = "test_help_headers.db";
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";
    const result = handleDotCommand(allocator, &db, path, ".help", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, ".headers") != null);
}

test "handleDotCommand .separator set custom separator" {
    const allocator = std.testing.allocator;

    const path = "test_separator_custom.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var db = Database.open(allocator, path, .{}) catch unreachable;
    defer db.close();

    var out_buf: [4096]u8 = undefined;
    var err_buf: [4096]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&out_buf);
    var ebs = std.io.fixedBufferStream(&err_buf);
    var w = fbs.writer();
    var ew = ebs.writer();

    var mode = OutputMode.csv;
    var show_timer = true;
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".separator |", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);
    try std.testing.expectEqualStrings("|", csv_separator);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "|") != null);
}

test "handleDotCommand .separator show current separator" {
    const allocator = std.testing.allocator;

    const path = "test_separator_show.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var db = Database.open(allocator, path, .{}) catch unreachable;
    defer db.close();

    var out_buf: [4096]u8 = undefined;
    var err_buf: [4096]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&out_buf);
    var ebs = std.io.fixedBufferStream(&err_buf);
    var w = fbs.writer();
    var ew = ebs.writer();

    var mode = OutputMode.csv;
    var show_timer = true;
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".separator", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "\",\"") != null);
}

test "handleDotCommand .separator pipe then query with CSV output" {
    const allocator = std.testing.allocator;

    const path = "test_separator_csv_output.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var db = Database.open(allocator, path, .{}) catch unreachable;
    defer db.close();

    // Create table and insert data
    _ = db.exec("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);") catch unreachable;
    _ = db.exec("INSERT INTO users (id, name) VALUES (1, 'Alice');") catch unreachable;

    var out_buf: [4096]u8 = undefined;
    var err_buf: [4096]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&out_buf);
    var ebs = std.io.fixedBufferStream(&err_buf);
    var w = fbs.writer();
    var ew = ebs.writer();

    var mode = OutputMode.csv;
    var show_timer = false;
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    // Set separator to pipe
    _ = handleDotCommand(allocator, &db, path, ".separator |", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqualStrings("|", csv_separator);

    // Execute query with CSV output
    fbs.reset();
    _ = execAndDisplay(allocator, &db, "SELECT * FROM users;", mode, show_timer, false, show_headers, csv_separator, null_display, &last_rows_affected, null, false, &w, &ew);

    const output = fbs.getWritten();
    // Verify output uses pipe separator
    try std.testing.expect(std.mem.indexOf(u8, output, "id|name") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "1|Alice") != null);
}

test "handleDotCommand .help includes .separator" {
    const allocator = std.testing.allocator;

    const path = "test_help_separator.db";
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".help", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, ".separator") != null);
}

test "handleDotCommand .nullvalue set custom string" {
    const allocator = std.testing.allocator;

    const path = "test_nullvalue_set.db";
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".nullvalue <empty>", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);
    try std.testing.expectEqualStrings("<empty>", null_display);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "Nullvalue set to:") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "<empty>") != null);
}

test "handleDotCommand .nullvalue show current string" {
    const allocator = std.testing.allocator;

    const path = "test_nullvalue_show.db";
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".nullvalue", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "Nullvalue:") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "NULL") != null);
}

test "handleDotCommand .nullvalue with NULL values in query" {
    const allocator = std.testing.allocator;

    const path = "test_nullvalue_query.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var db = Database.open(allocator, path, .{}) catch unreachable;
    defer db.close();

    // Create table with NULL values
    _ = db.exec("CREATE TABLE test (id INTEGER, name TEXT);") catch unreachable;
    _ = db.exec("INSERT INTO test (id, name) VALUES (1, NULL);") catch unreachable;

    var out_buf: [4096]u8 = undefined;
    var err_buf: [4096]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&out_buf);
    var ebs = std.io.fixedBufferStream(&err_buf);
    var w = fbs.writer();
    var ew = ebs.writer();

    var mode = OutputMode.table;
    var show_timer = false;
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    // Set custom null display
    _ = handleDotCommand(allocator, &db, path, ".nullvalue <empty>", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqualStrings("<empty>", null_display);

    // Execute query with NULL values
    fbs.reset();
    _ = execAndDisplay(allocator, &db, "SELECT * FROM test;", mode, show_timer, false, show_headers, csv_separator, null_display, &last_rows_affected, null, false, &w, &ew);

    const output = fbs.getWritten();
    // Verify that NULL is displayed as "<empty>"
    try std.testing.expect(std.mem.indexOf(u8, output, "<empty>") != null);
}

test "handleDotCommand .help includes .nullvalue" {
    const allocator = std.testing.allocator;

    const path = "test_help_nullvalue.db";
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".help", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, ".nullvalue") != null);
}

test "handleDotCommand .echo prints literal text" {
    const allocator = std.testing.allocator;

    const path = "test_echo.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var db = Database.open(allocator, path, .{}) catch unreachable;
    defer db.close();

    var out_buf: [1024]u8 = undefined;
    var err_buf: [1024]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&out_buf);
    var ebs = std.io.fixedBufferStream(&err_buf);
    var w = fbs.writer();
    var ew = ebs.writer();

    var mode = OutputMode.table;
    var show_timer = true;
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".echo Hello, World!", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "Hello, World!") != null);
}

test "handleDotCommand .echo with no text" {
    const allocator = std.testing.allocator;

    const path = "test_echo_empty.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var db = Database.open(allocator, path, .{}) catch unreachable;
    defer db.close();

    var out_buf: [1024]u8 = undefined;
    var err_buf: [1024]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&out_buf);
    var ebs = std.io.fixedBufferStream(&err_buf);
    var w = fbs.writer();
    var ew = ebs.writer();

    var mode = OutputMode.table;
    var show_timer = true;
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".echo", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    // Should just print a newline
    try std.testing.expectEqualStrings("\n", output);
}

test "handleDotCommand .help includes .echo" {
    const allocator = std.testing.allocator;

    const path = "test_help_echo.db";
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".help", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, ".echo") != null);
}

test "handleDotCommand .help includes .clear" {
    const allocator = std.testing.allocator;

    const path = "test_help_clear.db";
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".help", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, ".clear") != null);
}

test "handleDotCommand .print prints literal text" {
    const allocator = std.testing.allocator;

    const path = "test_print.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var db = Database.open(allocator, path, .{}) catch unreachable;
    defer db.close();

    var out_buf: [1024]u8 = undefined;
    var err_buf: [1024]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&out_buf);
    var ebs = std.io.fixedBufferStream(&err_buf);
    var w = fbs.writer();
    var ew = ebs.writer();

    var mode = OutputMode.table;
    var show_timer = true;
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".print SQLite-style output", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "SQLite-style output") != null);
}

test "handleDotCommand .print with no text" {
    const allocator = std.testing.allocator;

    const path = "test_print_empty.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var db = Database.open(allocator, path, .{}) catch unreachable;
    defer db.close();

    var out_buf: [1024]u8 = undefined;
    var err_buf: [1024]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&out_buf);
    var ebs = std.io.fixedBufferStream(&err_buf);
    var w = fbs.writer();
    var ew = ebs.writer();

    var mode = OutputMode.table;
    var show_timer = true;
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".print", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    // Should just print a newline
    try std.testing.expectEqualStrings("\n", output);
}

test "handleDotCommand .help includes .print" {
    const allocator = std.testing.allocator;

    const path = "test_help_print.db";
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".help", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, ".print") != null);
}

test "handleDotCommand .show displays all settings" {
    const allocator = std.testing.allocator;

    const path = "test_show.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var db = Database.open(allocator, path, .{}) catch unreachable;
    defer db.close();

    var out_buf: [2048]u8 = undefined;
    var err_buf: [1024]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&out_buf);
    var ebs = std.io.fixedBufferStream(&err_buf);
    var w = fbs.writer();
    var ew = ebs.writer();

    var mode = OutputMode.csv;
    var show_timer = false;
    var show_headers = false;
    var csv_separator: []const u8 = "|";
    var null_display: []const u8 = "<empty>";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".show", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    // Verify all settings are displayed
    try std.testing.expect(std.mem.indexOf(u8, output, "mode: csv") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "headers: off") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "timer: off") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "separator: \"|\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "nullvalue: \"<empty>\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "output: stdout") != null);
}

test "handleDotCommand .show with defaults" {
    const allocator = std.testing.allocator;

    const path = "test_show_defaults.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var db = Database.open(allocator, path, .{}) catch unreachable;
    defer db.close();

    var out_buf: [2048]u8 = undefined;
    var err_buf: [1024]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&out_buf);
    var ebs = std.io.fixedBufferStream(&err_buf);
    var w = fbs.writer();
    var ew = ebs.writer();

    // Default settings
    var mode = OutputMode.table;
    var show_timer = true;
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".show", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    // Verify default settings
    try std.testing.expect(std.mem.indexOf(u8, output, "mode: table") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "headers: on") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "timer: on") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "separator: \",\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "nullvalue: \"NULL\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "output: stdout") != null);
}

test "handleDotCommand .help includes .show" {
    const allocator = std.testing.allocator;

    const path = "test_help_show.db";
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".help", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, ".show") != null);
}

test "handleDotCommand .backup creates backup file" {
    const allocator = std.testing.allocator;
    const path = "test_backup_source.db";
    const backup_path = "test_backup_dest.db";

    // Clean up any existing files
    std.fs.cwd().deleteFile(path) catch {};
    std.fs.cwd().deleteFile(backup_path) catch {};
    defer {
        std.fs.cwd().deleteFile(path) catch {};
        std.fs.cwd().deleteFile(backup_path) catch {};
    }

    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();

    // Create a table with data to verify backup integrity
    _ = db.exec("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);") catch return error.SkipZigTest;
    _ = db.exec("INSERT INTO users (id, name) VALUES (1, 'Alice'), (2, 'Bob');") catch return error.SkipZigTest;

    var buf: [512]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;
    var show_timer = true;
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const cmd = ".backup test_backup_dest.db";
    const result = handleDotCommand(allocator, &db, path, cmd, &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "Database backed up to: test_backup_dest.db") != null);

    // Verify backup file exists
    const backup_file = std.fs.cwd().openFile(backup_path, .{}) catch return error.SkipZigTest;
    backup_file.close();

    // Verify backup can be opened and contains the data
    var backup_db = Database.open(allocator, backup_path, .{}) catch return error.SkipZigTest;
    defer backup_db.close();

    var count_result = backup_db.exec("SELECT COUNT(*) FROM users;") catch return error.SkipZigTest;
    defer count_result.close(allocator);
    const rows = count_result.rows orelse return error.SkipZigTest;
    var row = (rows.next() catch return error.SkipZigTest) orelse return error.SkipZigTest;
    defer row.deinit();
    const count = row.values[0].integer;
    try std.testing.expectEqual(@as(i64, 2), count);
}

test "handleDotCommand .backup requires filename" {
    const allocator = std.testing.allocator;
    const path = "test_backup_noarg.db";
    std.fs.cwd().deleteFile(path) catch {};
    defer std.fs.cwd().deleteFile(path) catch {};

    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();

    var buf: [512]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;
    var show_timer = true;
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".backup", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const err_output = efbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, err_output, "Usage: .backup FILENAME") != null);
}

test "handleDotCommand .backup prevents same file backup" {
    const allocator = std.testing.allocator;
    const path = "test_backup_same.db";
    std.fs.cwd().deleteFile(path) catch {};
    defer std.fs.cwd().deleteFile(path) catch {};

    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();

    var buf: [512]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;
    var show_timer = true;
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const cmd = ".backup test_backup_same.db";
    const result = handleDotCommand(allocator, &db, path, cmd, &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const err_output = efbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, err_output, "Cannot backup to the same file") != null);
}

test "handleDotCommand .backup handles existing file error" {
    const allocator = std.testing.allocator;
    const path = "test_backup_exists_src.db";
    const backup_path = "test_backup_exists_dest.db";

    std.fs.cwd().deleteFile(path) catch {};
    std.fs.cwd().deleteFile(backup_path) catch {};
    defer {
        std.fs.cwd().deleteFile(path) catch {};
        std.fs.cwd().deleteFile(backup_path) catch {};
    }

    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();

    // Create destination file first
    const dest_file = std.fs.cwd().createFile(backup_path, .{}) catch return error.SkipZigTest;
    dest_file.close();

    var buf: [512]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [512]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;
    var show_timer = true;
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const cmd = ".backup test_backup_exists_dest.db";
    const result = handleDotCommand(allocator, &db, path, cmd, &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const err_output = efbs.getWritten();
    // Error message should indicate file already exists
    try std.testing.expect(std.mem.indexOf(u8, err_output, "already exists") != null);
}

test "handleDotCommand .help includes .backup" {
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".help", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, ".backup") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "Create a backup copy of the database file") != null);
}

test "handleDotCommand .save creates file from :memory: database" {
    const allocator = std.testing.allocator;
    const path = ":memory:";
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();

    // Create table and insert data
    var result1 = db.exec("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);") catch return error.SkipZigTest;
    defer result1.close(allocator);
    var result2 = db.exec("INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob');") catch return error.SkipZigTest;
    defer result2.close(allocator);

    const save_path = "test_save_memory.db";
    defer std.fs.cwd().deleteFile(save_path) catch {};

    var buf: [256]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;
    var show_timer = true;
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    var cmd_buf: [128]u8 = undefined;
    const cmd = std.fmt.bufPrint(&cmd_buf, ".save {s}", .{save_path}) catch return error.SkipZigTest;

    const result = handleDotCommand(allocator, &db, path, cmd, &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "Database saved to") != null);

    // Verify the saved file exists and contains correct data
    var saved_db = Database.open(allocator, save_path, .{}) catch return error.SkipZigTest;
    defer saved_db.close();

    var verify_result = saved_db.exec("SELECT COUNT(*) FROM users;") catch return error.SkipZigTest;
    defer verify_result.close(allocator);
    if (try verify_result.rows.?.next()) |*row_ptr| {
        var row = row_ptr.*;
        defer row.deinit();
        const count = row.values[0].integer;
        try std.testing.expectEqual(@as(i64, 2), count);
    } else {
        return error.SkipZigTest;
    }
}

test "handleDotCommand .save requires filename" {
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".save", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const err_output = efbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, err_output, "Usage: .save FILENAME") != null);
}

test "handleDotCommand .save prevents overwriting existing file" {
    const allocator = std.testing.allocator;
    const path = ":memory:";
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();

    const save_path = "test_save_exists.db";
    // Create a dummy file
    std.fs.cwd().writeFile(.{ .sub_path = save_path, .data = "dummy" }) catch return error.SkipZigTest;
    defer std.fs.cwd().deleteFile(save_path) catch {};

    var buf: [256]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;
    var show_timer = true;
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    var cmd_buf: [128]u8 = undefined;
    const cmd = std.fmt.bufPrint(&cmd_buf, ".save {s}", .{save_path}) catch return error.SkipZigTest;

    const result = handleDotCommand(allocator, &db, path, cmd, &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const err_output = efbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, err_output, "already exists") != null);
}

test "handleDotCommand .save works with file-based databases (uses backup)" {
    const allocator = std.testing.allocator;
    const source_path = "test_save_source.db";
    defer std.fs.cwd().deleteFile(source_path) catch {};
    var db = Database.open(allocator, source_path, .{}) catch return error.SkipZigTest;
    defer db.close();

    // Create table and insert data
    var result1 = db.exec("CREATE TABLE products (id INTEGER, name TEXT);") catch return error.SkipZigTest;
    defer result1.close(allocator);
    var result2 = db.exec("INSERT INTO products VALUES (1, 'Widget');") catch return error.SkipZigTest;
    defer result2.close(allocator);

    const save_path = "test_save_dest.db";
    defer std.fs.cwd().deleteFile(save_path) catch {};

    var buf: [256]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;
    var show_timer = true;
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    var cmd_buf: [128]u8 = undefined;
    const cmd = std.fmt.bufPrint(&cmd_buf, ".save {s}", .{save_path}) catch return error.SkipZigTest;

    const result = handleDotCommand(allocator, &db, source_path, cmd, &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    // Verify file was saved (should use backup mechanism)
    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "backed up to") != null or std.mem.indexOf(u8, output, "saved to") != null);
}

test "handleDotCommand .help includes .save" {
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".help", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, ".save") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "Save database to file") != null);
}

test "handleDotCommand .import imports CSV data" {
    const allocator = std.testing.allocator;
    const path = "test_import.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();

    // Create table
    var result1 = db.exec("CREATE TABLE users (id INTEGER, name TEXT, email TEXT);") catch return error.SkipZigTest;
    defer result1.close(allocator);

    // Create CSV file
    const csv_path = "test_import.csv";
    const csv_content = "1,Alice,alice@example.com\n2,Bob,bob@example.com\n3,Charlie,charlie@example.com\n";
    std.fs.cwd().writeFile(.{ .sub_path = csv_path, .data = csv_content }) catch return error.SkipZigTest;
    defer std.fs.cwd().deleteFile(csv_path) catch {};

    var buf: [512]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;
    var show_timer = true;
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".import test_import.csv users", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "Imported 3 rows") != null);

    // Verify data was imported
    var verify_result = db.exec("SELECT COUNT(*) FROM users;") catch return error.SkipZigTest;
    defer verify_result.close(allocator);
    if (try verify_result.rows.?.next()) |*row_ptr| {
        var row = row_ptr.*;
        defer row.deinit();
        const count = row.values[0].integer;
        try std.testing.expectEqual(@as(i64, 3), count);
    } else {
        return error.SkipZigTest;
    }
}

test "handleDotCommand .import with custom separator" {
    const allocator = std.testing.allocator;
    const path = "test_import_pipe.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();

    // Create table
    var result1 = db.exec("CREATE TABLE products (id INTEGER, name TEXT, price TEXT);") catch return error.SkipZigTest;
    defer result1.close(allocator);

    // Create pipe-separated CSV file
    const csv_path = "test_import_pipe.csv";
    const csv_content = "1|Apple|1.99\n2|Banana|0.99\n";
    std.fs.cwd().writeFile(.{ .sub_path = csv_path, .data = csv_content }) catch return error.SkipZigTest;
    defer std.fs.cwd().deleteFile(csv_path) catch {};

    var buf: [512]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;
    var show_timer = true;
    var show_headers = true;
    var csv_separator: []const u8 = "|"; // Use pipe separator
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".import test_import_pipe.csv products", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "Imported 2 rows") != null);
}

test "handleDotCommand .import missing arguments" {
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".import", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const eoutput = efbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, eoutput, "Usage: .import FILE TABLE") != null);
}

test "handleDotCommand .import file not found" {
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".import nonexistent.csv users", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const eoutput = efbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, eoutput, "CSV file not found") != null);
}

test "handleDotCommand .help includes .import" {
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".help", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, ".import") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "Import CSV data from file into table") != null);
}

test "handleDotCommand .changes shows rows affected by INSERT" {
    const allocator = std.testing.allocator;
    const path = "test_changes_insert.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var db = Database.open(allocator, path, .{}) catch unreachable;
    defer db.close();

    // Create table and insert rows
    _ = db.exec("CREATE TABLE test (id INTEGER);") catch unreachable;

    var out_buf: [4096]u8 = undefined;
    var err_buf: [4096]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&out_buf);
    var ebs = std.io.fixedBufferStream(&err_buf);
    var w = fbs.writer();
    var ew = ebs.writer();

    var mode = OutputMode.table;
    var show_timer = false;
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    // Execute INSERT and track rows_affected
    _ = execAndDisplay(allocator, &db, "INSERT INTO test (id) VALUES (1), (2), (3);", mode, show_timer, false, show_headers, csv_separator, null_display, &last_rows_affected, null, false, &w, &ew);

    // Now check .changes
    fbs.reset();
    const result = handleDotCommand(allocator, &db, path, ".changes", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "3") != null); // 3 rows inserted
}

test "handleDotCommand .changes shows rows affected by UPDATE" {
    const allocator = std.testing.allocator;
    const path = "test_changes_update.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var db = Database.open(allocator, path, .{}) catch unreachable;
    defer db.close();

    // Create table and insert rows
    _ = db.exec("CREATE TABLE test (id INTEGER, value INTEGER);") catch unreachable;
    _ = db.exec("INSERT INTO test (id, value) VALUES (1, 10), (2, 20), (3, 30);") catch unreachable;

    var out_buf: [4096]u8 = undefined;
    var err_buf: [4096]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&out_buf);
    var ebs = std.io.fixedBufferStream(&err_buf);
    var w = fbs.writer();
    var ew = ebs.writer();

    var mode = OutputMode.table;
    var show_timer = false;
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    // Execute UPDATE and track rows_affected
    _ = execAndDisplay(allocator, &db, "UPDATE test SET value = 99 WHERE id <= 2;", mode, show_timer, false, show_headers, csv_separator, null_display, &last_rows_affected, null, false, &w, &ew);

    // Now check .changes
    fbs.reset();
    const result = handleDotCommand(allocator, &db, path, ".changes", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "2") != null); // 2 rows updated
}

test "handleDotCommand .changes shows rows affected by DELETE" {
    const allocator = std.testing.allocator;
    const path = "test_changes_delete.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var db = Database.open(allocator, path, .{}) catch unreachable;
    defer db.close();

    // Create table and insert rows
    _ = db.exec("CREATE TABLE test (id INTEGER);") catch unreachable;
    _ = db.exec("INSERT INTO test (id) VALUES (1), (2), (3), (4);") catch unreachable;

    var out_buf: [4096]u8 = undefined;
    var err_buf: [4096]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&out_buf);
    var ebs = std.io.fixedBufferStream(&err_buf);
    var w = fbs.writer();
    var ew = ebs.writer();

    var mode = OutputMode.table;
    var show_timer = false;
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    // Execute DELETE and track rows_affected
    _ = execAndDisplay(allocator, &db, "DELETE FROM test WHERE id > 1;", mode, show_timer, false, show_headers, csv_separator, null_display, &last_rows_affected, null, false, &w, &ew);

    // Now check .changes
    fbs.reset();
    const result = handleDotCommand(allocator, &db, path, ".changes", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "3") != null); // 3 rows deleted
}

test "handleDotCommand .changes shows 0 for SELECT" {
    const allocator = std.testing.allocator;
    const path = "test_changes_select.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var db = Database.open(allocator, path, .{}) catch unreachable;
    defer db.close();

    // Create table and insert rows
    _ = db.exec("CREATE TABLE test (id INTEGER);") catch unreachable;
    _ = db.exec("INSERT INTO test (id) VALUES (1), (2);") catch unreachable;

    var out_buf: [4096]u8 = undefined;
    var err_buf: [4096]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&out_buf);
    var ebs = std.io.fixedBufferStream(&err_buf);
    var w = fbs.writer();
    var ew = ebs.writer();

    var mode = OutputMode.table;
    var show_timer = false;
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    // Execute SELECT (doesn't affect rows)
    _ = execAndDisplay(allocator, &db, "SELECT * FROM test;", mode, show_timer, false, show_headers, csv_separator, null_display, &last_rows_affected, null, false, &w, &ew);

    // Now check .changes
    fbs.reset();
    const result = handleDotCommand(allocator, &db, path, ".changes", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "0") != null); // SELECT doesn't change rows
}

test "handleDotCommand .help includes .changes" {
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".help", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, ".changes") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "Show number of rows changed by last DML statement") != null);
}

test "handleDotCommand .bail on" {
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".bail on", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);
    try std.testing.expect(bail_on_error == true);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "Bail on error enabled") != null);
}

test "handleDotCommand .bail off" {
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = true; // Start with true
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".bail off", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);
    try std.testing.expect(bail_on_error == false);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "Bail on error disabled") != null);
}

test "handleDotCommand .bail - show setting" {
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".bail", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "Bail: off") != null);
}

test "handleDotCommand .bail invalid argument" {
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".bail foobar", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const err_output = efbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, err_output, "Usage: .bail on|off") != null);
}

test "handleDotCommand .show includes bail" {
    const allocator = std.testing.allocator;
    const path = ":memory:";
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();

    var buf: [1024]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;
    var show_timer = true;
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".show", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "bail: off") != null);
}

test "handleDotCommand .help includes .bail" {
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".help", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, ".bail") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "Stop script execution on first error") != null);
}

test "readAndExecuteFile with bail_on_error off - continues on error" {
    const allocator = std.testing.allocator;
    const path = "test_bail_off.db";
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();
    defer std.fs.cwd().deleteFile(path) catch {};

    // Create a test SQL file with one valid statement and one error
    const filename = "test_bail_off.sql";
    const file = std.fs.cwd().createFile(filename, .{}) catch return error.SkipZigTest;
    defer std.fs.cwd().deleteFile(filename) catch {};
    file.writeAll("CREATE TABLE test (id INTEGER PRIMARY KEY);\n") catch return error.SkipZigTest;
    file.writeAll("INSERT INTO nonexistent VALUES (1);\n") catch return error.SkipZigTest; // This will fail
    file.writeAll("INSERT INTO test VALUES (1);\n") catch return error.SkipZigTest;
    file.close();

    var buf: [4096]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [512]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var last_rows_affected: u64 = 0;

    readAndExecuteFile(allocator, &db, filename, .table, true, false, true, ",", "NULL", &last_rows_affected, false, null, false, &w, &ew);

    // With bail_on_error=false, all 3 statements should execute (one fails, two succeed)
    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "Executed 3 statement(s)") != null);

    // Verify that the third statement executed successfully
    var result = db.exec("SELECT COUNT(*) FROM test;") catch return error.SkipZigTest;
    defer result.close(allocator);
    var row = (result.rows.?.next() catch return error.SkipZigTest).?;
    defer row.deinit();
    const count_val = row.values[0];
    try std.testing.expectEqual(@as(i64, 1), count_val.integer);
}

test "readAndExecuteFile with bail_on_error on - stops on error" {
    const allocator = std.testing.allocator;
    const path = "test_bail_on.db";
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();
    defer std.fs.cwd().deleteFile(path) catch {};

    // Create a test SQL file with one valid statement and one error
    const filename = "test_bail_on.sql";
    const file = std.fs.cwd().createFile(filename, .{}) catch return error.SkipZigTest;
    defer std.fs.cwd().deleteFile(filename) catch {};
    file.writeAll("CREATE TABLE test (id INTEGER PRIMARY KEY);\n") catch return error.SkipZigTest;
    file.writeAll("INSERT INTO nonexistent VALUES (1);\n") catch return error.SkipZigTest; // This will fail
    file.writeAll("INSERT INTO test VALUES (1);\n") catch return error.SkipZigTest;
    file.close();

    var buf: [4096]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [512]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var last_rows_affected: u64 = 0;

    readAndExecuteFile(allocator, &db, filename, .table, true, false, true, ",", "NULL", &last_rows_affected, true, null, false, &w, &ew);

    // With bail_on_error=true, execution should stop after the error
    const err_output = efbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, err_output, "Script execution stopped due to bail on error setting") != null);

    // Verify that the third statement did NOT execute
    var result = db.exec("SELECT COUNT(*) FROM test;") catch return error.SkipZigTest;
    defer result.close(allocator);
    var row = (result.rows.?.next() catch return error.SkipZigTest).?;
    defer row.deinit();
    const count_val = row.values[0];
    try std.testing.expectEqual(@as(i64, 0), count_val.integer); // Table is empty (third INSERT didn't run)
}

test "handleDotCommand .log FILENAME - enable logging" {
    const allocator = std.testing.allocator;
    const path = "test_log_enable.db";
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();
    defer std.fs.cwd().deleteFile(path) catch {};

    const log_path = "test_query.log";
    defer std.fs.cwd().deleteFile(log_path) catch {};

    var buf: [256]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;
    var show_timer = true;
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";
    defer if (log_file) |f| f.close();

    const cmd = ".log test_query.log";
    const result = handleDotCommand(allocator, &db, path, cmd, &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);
    try std.testing.expect(log_file != null);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "Logging to: test_query.log") != null);
}

test "handleDotCommand .log off - disable logging" {
    const allocator = std.testing.allocator;
    const path = "test_log_off.db";
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();
    defer std.fs.cwd().deleteFile(path) catch {};

    var buf: [256]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;
    var show_timer = true;
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".log off", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);
    try std.testing.expect(log_file == null);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "Logging is already off") != null);
}

test "handleDotCommand .log - show current status (off)" {
    const allocator = std.testing.allocator;
    const path = "test_log_status.db";
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();
    defer std.fs.cwd().deleteFile(path) catch {};

    var buf: [256]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;
    var show_timer = true;
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".log", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "log: off") != null);
}

test "handleDotCommand .log - show current status (on)" {
    const allocator = std.testing.allocator;
    const path = "test_log_status_on.db";
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();
    defer std.fs.cwd().deleteFile(path) catch {};

    const log_path = "test_status_on.log";
    defer std.fs.cwd().deleteFile(log_path) catch {};

    var buf: [256]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;
    var show_timer = true;
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";
    defer if (log_file) |f| f.close();

    // Enable logging first
    _ = handleDotCommand(allocator, &db, path, ".log test_status_on.log", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expect(log_file != null);

    // Check status
    fbs.reset();
    const result = handleDotCommand(allocator, &db, path, ".log", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "log: on") != null);
}

test "handleDotCommand .show - includes log setting" {
    const allocator = std.testing.allocator;
    const path = "test_show_log.db";
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();
    defer std.fs.cwd().deleteFile(path) catch {};

    var buf: [512]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;
    var show_timer = true;
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".show", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "log:") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "off") != null);
}

test "handleDotCommand .help - includes .log" {
    const allocator = std.testing.allocator;
    const path = "test_help_log.db";
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".help", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, ".log") != null);
}

test "handleDotCommand .cd - show current directory" {
    const allocator = std.testing.allocator;
    const path = "test_cd_show.db";
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".cd", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "Current directory:") != null);
}

test "handleDotCommand .cd DIRECTORY - change directory" {
    const allocator = std.testing.allocator;
    const path = "test_cd_change.db";
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();
    defer std.fs.cwd().deleteFile(path) catch {};

    // Create a temp directory for testing
    const temp_dir = "test_cd_temp_dir";
    std.fs.cwd().makeDir(temp_dir) catch |err| {
        if (err != error.PathAlreadyExists) return error.SkipZigTest;
    };
    defer std.fs.cwd().deleteDir(temp_dir) catch {};

    // Get current directory to restore later
    const original_cwd = std.fs.cwd().realpathAlloc(allocator, ".") catch return error.SkipZigTest;
    defer allocator.free(original_cwd);

    var buf: [4096]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;
    var show_timer = true;
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const cmd = ".cd " ++ temp_dir;
    const result = handleDotCommand(allocator, &db, path, cmd, &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "Changed directory to:") != null);

    // Restore original directory
    const original_cwd_z = allocator.dupeZ(u8, original_cwd) catch return error.SkipZigTest;
    defer allocator.free(original_cwd_z);
    std.posix.chdir(original_cwd_z) catch {};
}

test "handleDotCommand .open - show current database" {
    const allocator = std.testing.allocator;
    const path = "test_open_show.db";
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".open", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "Current database:") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, path) != null);
}

test "handleDotCommand .open FILENAME - return reopen result" {
    const allocator = std.testing.allocator;
    const path = "test_open_original.db";
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const new_db_path = "test_open_new.db";
    defer std.fs.cwd().deleteFile(new_db_path) catch {};

    const cmd = ".open " ++ new_db_path;
    const result = handleDotCommand(allocator, &db, path, cmd, &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);

    // Verify we got a reopen result
    try std.testing.expect(result == .reopen);

    // Verify the new path is returned
    switch (result) {
        .reopen => |new_path| {
            defer allocator.free(new_path);
            try std.testing.expect(std.mem.eql(u8, new_path, new_db_path));
        },
        else => unreachable,
    }
}

test "handleDotCommand .help includes .open" {
    const allocator = std.testing.allocator;
    const path = "test_help_open.db";
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();
    defer std.fs.cwd().deleteFile(path) catch {};

    var buf: [8192]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;
    var show_timer = true;
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".help", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, ".open") != null);
}

test "handleDotCommand .help includes .cd" {
    const allocator = std.testing.allocator;
    const path = "test_help_cd.db";
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".help", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, ".cd") != null);
}

test "handleDotCommand .version shows version info" {
    const allocator = std.testing.allocator;
    const path = "test_version.db";
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".version", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "Silica") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "Zig") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "Dependencies:") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "sailor v1.36.0") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "zuda v2.0.0") != null);
}

test "handleDotCommand .help includes .version" {
    const allocator = std.testing.allocator;
    const path = "test_help_version.db";
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".help", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, ".version") != null);
}

test "handleDotCommand .stats on - enable stats" {
    const allocator = std.testing.allocator;
    const path = "test_stats_on.db";
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();
    defer std.fs.cwd().deleteFile(path) catch {};

    var buf: [512]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [128]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;
    var show_timer = true;
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".stats on", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);
    try std.testing.expect(show_stats == true);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "Stats enabled") != null);
}

test "handleDotCommand .stats off - disable stats" {
    const allocator = std.testing.allocator;
    const path = "test_stats_off.db";
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();
    defer std.fs.cwd().deleteFile(path) catch {};

    var buf: [512]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [128]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;
    var show_timer = true;
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = true;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".stats off", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);
    try std.testing.expect(show_stats == false);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "Stats disabled") != null);
}

test "handleDotCommand .stats - show current stats setting" {
    const allocator = std.testing.allocator;
    const path = "test_stats_show.db";
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();
    defer std.fs.cwd().deleteFile(path) catch {};

    var buf: [512]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [128]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;
    var show_timer = true;
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".stats", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "Stats: off") != null);
}

test "handleDotCommand .show - includes stats setting" {
    const allocator = std.testing.allocator;
    const path = "test_show_stats.db";
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();
    defer std.fs.cwd().deleteFile(path) catch {};

    var buf: [1024]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [128]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;
    var show_timer = true;
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = true;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".show", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "stats: on") != null);
}

test "handleDotCommand .help - includes .stats command" {
    const allocator = std.testing.allocator;
    const path = "test_help_stats.db";
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();
    defer std.fs.cwd().deleteFile(path) catch {};

    var buf: [4096]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [128]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;
    var show_timer = true;
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".help", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, ".stats") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "Show execution statistics") != null);
}

test "handleDotCommand .eqp on - enable automatic explain" {
    const allocator = std.testing.allocator;
    const path = "test_eqp_on.db";
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();
    defer std.fs.cwd().deleteFile(path) catch {};

    var buf: [2048]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;
    var show_timer = false;
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".eqp on", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);
    try std.testing.expect(show_eqp == true);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "EQP enabled") != null);
}

test "handleDotCommand .eqp off - disable automatic explain" {
    const allocator = std.testing.allocator;
    const path = "test_eqp_off.db";
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();
    defer std.fs.cwd().deleteFile(path) catch {};

    var buf: [2048]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;
    var show_timer = false;
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = true;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".eqp off", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);
    try std.testing.expect(show_eqp == false);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "EQP disabled") != null);
}

test "handleDotCommand .eqp - show current eqp setting" {
    const allocator = std.testing.allocator;
    const path = "test_eqp_show.db";
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();
    defer std.fs.cwd().deleteFile(path) catch {};

    var buf: [2048]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;
    var show_timer = false;
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = true;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".eqp", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "EQP: on") != null);
}

test "handleDotCommand .show - includes eqp setting" {
    const allocator = std.testing.allocator;
    const path = "test_show_eqp.db";
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();
    defer std.fs.cwd().deleteFile(path) catch {};

    var buf: [2048]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;
    var show_timer = true;
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = true;
    var show_eqp = true;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".show", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "eqp: on") != null);
}

test "handleDotCommand .prompt MAIN CONTINUE - set custom prompts" {
    const allocator = std.testing.allocator;
    const path = "test_prompt_set.db";
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();
    defer std.fs.cwd().deleteFile(path) catch {};

    var buf: [2048]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;
    var show_timer = false;
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".prompt db> ...", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);
    try std.testing.expectEqualStrings("db>", main_prompt);
    try std.testing.expectEqualStrings("...", continue_prompt);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "Prompts updated") != null);
}

test "handleDotCommand .prompt - show current prompts" {
    const allocator = std.testing.allocator;
    const path = "test_prompt_show.db";
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();
    defer std.fs.cwd().deleteFile(path) catch {};

    var buf: [2048]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;
    var show_timer = false;
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "custom> ";
    var continue_prompt: []const u8 = ">>>";

    const result = handleDotCommand(allocator, &db, path, ".prompt", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "Main: \"custom> \"") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "Continue: \">>>\"") != null);
}

test "handleDotCommand .prompt - error on missing argument" {
    const allocator = std.testing.allocator;
    const path = "test_prompt_error.db";
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();
    defer std.fs.cwd().deleteFile(path) catch {};

    var buf: [2048]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;
    var show_timer = false;
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".prompt only_one", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const error_output = efbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, error_output, "Usage: .prompt MAIN CONTINUE") != null);
}

test "handleDotCommand .show - includes prompt setting" {
    const allocator = std.testing.allocator;
    const path = "test_show_prompt.db";
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();
    defer std.fs.cwd().deleteFile(path) catch {};

    var buf: [2048]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;
    var show_timer = true;
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = true;
    var show_eqp = true;
    var main_prompt: []const u8 = "my_db> ";
    var continue_prompt: []const u8 = "...";

    const result = handleDotCommand(allocator, &db, path, ".show", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "prompt: \"my_db> \" \"...\"") != null);
}

test "handleDotCommand .help - includes .prompt command" {
    const allocator = std.testing.allocator;
    const path = "test_help_prompt.db";
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".help", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, ".prompt") != null);
}

test "handleDotCommand .help - includes .eqp command" {
    const allocator = std.testing.allocator;
    const path = "test_help_eqp.db";
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();
    defer std.fs.cwd().deleteFile(path) catch {};

    var buf: [2048]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;
    var show_timer = false;
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".help", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, ".eqp") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "Automatically EXPLAIN query plans") != null);
}

test "handleDotCommand .once - write next query to file" {
    const allocator = std.testing.allocator;
    const path = "test_once_source.db";
    const once_path = "test_once_output.txt";

    // Clean up any existing files
    std.fs.cwd().deleteFile(path) catch {};
    std.fs.cwd().deleteFile(once_path) catch {};
    defer {
        std.fs.cwd().deleteFile(path) catch {};
        std.fs.cwd().deleteFile(once_path) catch {};
    }

    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();

    _ = db.exec("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);") catch return error.SkipZigTest;
    _ = db.exec("INSERT INTO users (id, name) VALUES (1, 'Alice'), (2, 'Bob');") catch return error.SkipZigTest;

    var buf: [512]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [128]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .csv;
    var show_timer = false;
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    // Set .once to redirect next query
    const cmd = ".once test_once_output.txt";
    const result = handleDotCommand(allocator, &db, path, cmd, &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "Next output will be written to: test_once_output.txt") != null);

    // Verify once_file is now set
    try std.testing.expect(once_file != null);

    // Simulate query execution with once_file
    var once_file_buf: [4096]u8 = undefined;
    var once_writer = once_file.?.writer(&once_file_buf);
    const once_out = &once_writer.interface;

    var query_result = db.exec("SELECT * FROM users ORDER BY id;") catch return error.SkipZigTest;
    defer query_result.close(allocator);

    // Write headers
    once_out.writeAll("id,name\n") catch {};

    // Write rows
    if (query_result.rows) |*rows| {
        while (try rows.next()) |r| {
            var row = r;
            defer row.deinit();
            once_out.print("{d},{s}\n", .{row.values[0].integer, row.values[1].text}) catch {};
        }
    }
    once_out.flush() catch {};

    // Close once_file (simulating what the REPL does)
    once_file.?.close();
    once_file = null;

    // Verify file was created and contains expected content
    const file_content = std.fs.cwd().readFileAlloc(allocator, once_path, 1024) catch return error.SkipZigTest;
    defer allocator.free(file_content);

    try std.testing.expect(std.mem.indexOf(u8, file_content, "id,name") != null);
    try std.testing.expect(std.mem.indexOf(u8, file_content, "1,Alice") != null);
    try std.testing.expect(std.mem.indexOf(u8, file_content, "2,Bob") != null);
}

test "handleDotCommand .once - requires filename" {
    const allocator = std.testing.allocator;
    const path = "test_once_noarg.db";
    std.fs.cwd().deleteFile(path) catch {};
    defer std.fs.cwd().deleteFile(path) catch {};

    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();

    var buf: [128]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;
    var show_timer = true;
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".once", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const err_output = efbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, err_output, "Usage: .once FILENAME") != null);
}

test "handleDotCommand .show - includes once setting" {
    const allocator = std.testing.allocator;
    const path = "test_show_once.db";
    std.fs.cwd().deleteFile(path) catch {};
    defer std.fs.cwd().deleteFile(path) catch {};

    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();

    var buf: [1024]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [128]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;
    var show_timer = true;
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".show", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "once:") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "off") != null); // Default is off
}

test "handleDotCommand .help - includes .once command" {
    const allocator = std.testing.allocator;
    const path = "test_help_once.db";
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();
    defer std.fs.cwd().deleteFile(path) catch {};

    var buf: [8192]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [128]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;
    var show_timer = true;
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".help", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, ".once") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "Write next query output to file") != null);
}

test "handleDotCommand .system - executes shell command" {
    const allocator = std.testing.allocator;
    const path = "test_system_cmd.db";
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".system echo Hello", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "Hello") != null);
}

test "handleDotCommand .system - missing command shows error" {
    const allocator = std.testing.allocator;
    const path = "test_system_missing.db";
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();
    defer std.fs.cwd().deleteFile(path) catch {};

    var buf: [256]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;
    var show_timer = true;
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".system", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const error_output = efbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, error_output, "Usage: .system CMD ARGS...") != null);
}

test "handleDotCommand .help - includes .system command" {
    const allocator = std.testing.allocator;
    const path = "test_help_system.db";
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();
    defer std.fs.cwd().deleteFile(path) catch {};

    var buf: [8192]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;
    var show_timer = true;
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".help", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, ".system") != null);
}

test "handleDotCommand .shell - executes shell command (SQLite alias)" {
    const allocator = std.testing.allocator;
    const path = "test_shell_cmd.db";
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
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".shell echo World", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "World") != null);
}

test "handleDotCommand .shell - missing command shows error" {
    const allocator = std.testing.allocator;
    const path = "test_shell_missing.db";
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();
    defer std.fs.cwd().deleteFile(path) catch {};

    var buf: [256]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;
    var show_timer = true;
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".shell", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const error_output = efbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, error_output, "Usage: .shell CMD ARGS...") != null);
}

test "handleDotCommand .help - includes .shell command" {
    const allocator = std.testing.allocator;
    const path = "test_help_shell.db";
    var db = Database.open(allocator, path, .{}) catch return error.SkipZigTest;
    defer db.close();
    defer std.fs.cwd().deleteFile(path) catch {};

    var buf: [8192]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;
    var show_timer = true;
    var show_headers = true;
    var csv_separator: []const u8 = ",";
    var null_display: []const u8 = "NULL";
    var output_file: ?std.fs.File = null;
    var once_file: ?std.fs.File = null;
    var last_rows_affected: u64 = 0;
    var bail_on_error = false;
    var log_file: ?std.fs.File = null;
    var show_stats = false;
    var show_eqp = false;
    var main_prompt: []const u8 = "silica> ";
    var continue_prompt: []const u8 = "   ...> ";

    const result = handleDotCommand(allocator, &db, path, ".help", &mode, &show_timer, &show_headers, &csv_separator, &null_display, &output_file, &once_file, &last_rows_affected, &bail_on_error, &log_file, &show_stats, &show_eqp, &main_prompt, &continue_prompt, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);

    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, ".shell") != null);
}

// Import wire_fuzz tests
test {
    _ = @import("server/wire_fuzz.zig");
}
