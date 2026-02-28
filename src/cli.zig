const std = @import("std");
const sailor = @import("sailor");
const silica = @import("silica");

const tui_mod = @import("tui.zig");

const version = "0.4.0";

const Value = silica.executor.Value;
const Row = silica.executor.Row;
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
};

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

    // First positional argument is the database path
    if (arg_parser.positional.items.len == 0) {
        printUsage(stdout);
        stdout.flush() catch {};
        return;
    }

    const db_path = arg_parser.positional.items[0];

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
        tui_mod.run(allocator, &db, db_path) catch |err| {
            printError(stderr, switch (err) {
                error.NotATty => "TUI mode requires a terminal.",
                error.TerminalSizeUnavailable => "Cannot determine terminal size.",
                else => "TUI initialization failed.",
            });
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
            const result = handleDotCommand(trimmed, &mode, stdout, stderr);
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
        execAndDisplay(allocator, &db, input_buf.items, mode, stdout, stderr);
        stdout.flush() catch {};
        stderr.flush() catch {};
        input_buf.clearRetainingCapacity();
        is_continuation = false;
    }
}

// ── SQL Execution ──────────────────────────────────────────────────────

/// Execute SQL via the Database engine and display results.
fn execAndDisplay(allocator: std.mem.Allocator, db: *Database, sql: []const u8, mode: OutputMode, stdout: anytype, stderr: anytype) void {
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

const DotCommandResult = enum { ok, quit };

fn handleDotCommand(cmd: []const u8, mode: *OutputMode, stdout: anytype, stderr: anytype) DotCommandResult {
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
            \\.tables             List tables (not yet implemented)
            \\.schema             Show schema (not yet implemented)
            \\
        ) catch {};
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
    } else if (std.mem.eql(u8, cmd, ".tables") or std.mem.eql(u8, cmd, ".schema")) {
        stdout.writeAll("Not yet implemented.\n") catch {};
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
        \\
        \\A lightweight, embedded relational database engine.
        \\
        \\Arguments:
        \\  <database>    Path to the database file
        \\
        \\
    ) catch {};
    sailor.arg.Parser(&CliFlags).writeHelp(writer) catch {};
    writer.writeAll(
        \\
        \\Examples:
        \\  silica mydb.db              Open database in interactive mode
        \\  silica --tui mydb.db        Open TUI database browser
        \\  silica --csv mydb.db        Open with CSV output format
        \\  silica -m json mydb.db      Open with JSON output format
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
    try std.testing.expectEqualStrings("0.4.0", version);
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
    var buf: [2048]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;
    const result = handleDotCommand(".help", &mode, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);
    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, ".help") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, ".quit") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, ".mode") != null);
}

test "handleDotCommand quit" {
    var buf: [256]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;
    const result = handleDotCommand(".quit", &mode, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.quit, result);
    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "Bye!") != null);
}

test "handleDotCommand unknown" {
    var buf: [256]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;
    const result = handleDotCommand(".foobar", &mode, &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);
    const eoutput = efbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, eoutput, "Unknown command") != null);
}

test "handleDotCommand mode set" {
    var buf: [256]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;
    _ = handleDotCommand(".mode csv", &mode, &w, &ew);
    try std.testing.expectEqual(OutputMode.csv, mode);
    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "csv") != null);
}

test "handleDotCommand mode show" {
    var buf: [256]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .json;
    _ = handleDotCommand(".mode", &mode, &w, &ew);
    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "json") != null);
}

test "handleDotCommand mode invalid" {
    var buf: [256]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    var mode: OutputMode = .table;
    _ = handleDotCommand(".mode foobar", &mode, &w, &ew);
    try std.testing.expectEqual(OutputMode.table, mode); // unchanged
    const eoutput = efbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, eoutput, "Invalid mode") != null);
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
