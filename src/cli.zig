const std = @import("std");
const sailor = @import("sailor");
const silica = @import("silica");

const version = "0.2.0";

const CliFlags = [_]sailor.arg.FlagDef{
    .{ .name = "help", .short = 'h', .type = .bool, .help = "Show this help message" },
    .{ .name = "version", .short = 'v', .type = .bool, .help = "Show version information" },
    .{ .name = "header", .type = .bool, .help = "Show column headers in output", .default = "true" },
    .{ .name = "csv", .type = .bool, .help = "Output in CSV format" },
    .{ .name = "json", .type = .bool, .help = "Output in JSON format" },
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

    // Build history file path: ~/.silica_history
    const history_path = buildHistoryPath(allocator);
    defer if (history_path) |p| allocator.free(p);

    // Initialize REPL
    var repl = sailor.repl.Repl.init(allocator, .{
        .prompt = "silica> ",
        .continuation_prompt = "   ...> ",
        .history_file = history_path,
        .history_size = 1000,
        .highlighter = sqlHighlighter,
        .validator = sqlValidator,
        .completer = sqlCompleter,
    }) catch {
        printError(stderr, "Failed to initialize REPL.");
        stderr.flush() catch {};
        std.process.exit(1);
    };
    defer repl.deinit();

    // Print banner
    stdout.print("Silica v{s} — interactive SQL shell\n", .{version}) catch {};
    stdout.print("Database: {s}\n", .{db_path}) catch {};
    stdout.writeAll("Type .help for usage hints, .quit to exit.\n\n") catch {};
    stdout.flush() catch {};

    // REPL loop
    var input_buf = std.ArrayListUnmanaged(u8){};
    defer input_buf.deinit(allocator);

    while (true) {
        const line = repl.readLine(stdout) catch |err| switch (err) {
            error.EndOfStream => {
                stdout.writeAll("\nBye!\n") catch {};
                stdout.flush() catch {};
                break;
            },
            else => {
                printError(stderr, "Failed to read input.");
                stderr.flush() catch {};
                continue;
            },
        };

        if (line) |input| {
            defer allocator.free(input);

            const trimmed = std.mem.trim(u8, input, " \t\r\n");
            if (trimmed.len == 0) continue;

            // Handle dot-commands
            if (trimmed[0] == '.') {
                const result = handleDotCommand(trimmed, stdout, stderr);
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
                continue;
            }

            // Parse and display
            processSQL(allocator, input_buf.items, stdout, stderr);
            stdout.flush() catch {};
            stderr.flush() catch {};
            input_buf.clearRetainingCapacity();
        } else {
            stdout.writeAll("\nBye!\n") catch {};
            stdout.flush() catch {};
            break;
        }
    }
}

/// Process SQL input: parse and display result.
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

/// Print info about a parsed statement (temporary — until executor is ready).
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

/// Print a SQL parse error with context.
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

/// Handle dot-commands like .help, .quit, .tables
const DotCommandResult = enum { ok, quit };

fn handleDotCommand(cmd: []const u8, stdout: anytype, stderr: anytype) DotCommandResult {
    if (std.mem.eql(u8, cmd, ".quit") or std.mem.eql(u8, cmd, ".exit")) {
        stdout.writeAll("Bye!\n") catch {};
        return .quit;
    } else if (std.mem.eql(u8, cmd, ".help")) {
        stdout.writeAll(
            \\.help       Show this help
            \\.quit       Exit the shell
            \\.exit       Exit the shell
            \\.tables     List tables (not yet implemented)
            \\.schema     Show schema (not yet implemented)
            \\
        ) catch {};
    } else if (std.mem.eql(u8, cmd, ".tables") or std.mem.eql(u8, cmd, ".schema")) {
        stdout.writeAll("Not yet implemented (requires query executor).\n") catch {};
    } else {
        printError(stderr, "Unknown command. Type .help for usage hints.");
    }
    return .ok;
}

/// SQL syntax highlighting callback for the REPL.
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

/// SQL input validator for multi-line support.
fn sqlValidator(buf: []const u8) sailor.repl.Validation {
    const trimmed = std.mem.trimRight(u8, buf, " \t\r\n");
    if (trimmed.len == 0) return .complete;
    if (trimmed[trimmed.len - 1] == ';') return .complete;
    if (trimmed[0] == '.') return .complete;
    return .incomplete;
}

/// SQL keyword completion callback.
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
        \\  silica --csv mydb.db        Open with CSV output format
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
    try std.testing.expectEqualStrings("0.2.0", version);
}

test "sqlValidator complete with semicolon" {
    try std.testing.expectEqual(sailor.repl.Validation.complete, sqlValidator("SELECT 1;"));
    try std.testing.expectEqual(sailor.repl.Validation.complete, sqlValidator("SELECT * FROM t;  "));
}

test "sqlValidator incomplete without semicolon" {
    try std.testing.expectEqual(sailor.repl.Validation.incomplete, sqlValidator("SELECT 1"));
    try std.testing.expectEqual(sailor.repl.Validation.incomplete, sqlValidator("SELECT *"));
}

test "sqlValidator complete for dot-commands" {
    try std.testing.expectEqual(sailor.repl.Validation.complete, sqlValidator(".help"));
    try std.testing.expectEqual(sailor.repl.Validation.complete, sqlValidator(".quit"));
}

test "sqlValidator empty input is complete" {
    try std.testing.expectEqual(sailor.repl.Validation.complete, sqlValidator(""));
    try std.testing.expectEqual(sailor.repl.Validation.complete, sqlValidator("   "));
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
    var buf: [1024]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    const result = handleDotCommand(".help", &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);
    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, ".help") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, ".quit") != null);
}

test "handleDotCommand quit" {
    var buf: [256]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    var ebuf: [256]u8 = undefined;
    var efbs = std.io.fixedBufferStream(&ebuf);
    var ew = efbs.writer();
    const result = handleDotCommand(".quit", &w, &ew);
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
    const result = handleDotCommand(".foobar", &w, &ew);
    try std.testing.expectEqual(DotCommandResult.ok, result);
    const eoutput = efbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, eoutput, "Unknown command") != null);
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
    // Test SELECT
    {
        var buf: [256]u8 = undefined;
        var fbs = std.io.fixedBufferStream(&buf);
        var w = fbs.writer();
        printStmtInfo(&w, .{ .select = .{ .columns = &.{.all_columns} } });
        const output = fbs.getWritten();
        try std.testing.expect(std.mem.indexOf(u8, output, "SELECT") != null);
    }

    // Test INSERT
    {
        var buf: [256]u8 = undefined;
        var fbs = std.io.fixedBufferStream(&buf);
        var w = fbs.writer();
        printStmtInfo(&w, .{ .insert = .{ .table = "users" } });
        const output = fbs.getWritten();
        try std.testing.expect(std.mem.indexOf(u8, output, "INSERT INTO users") != null);
    }

    // Test CREATE TABLE
    {
        var buf: [256]u8 = undefined;
        var fbs = std.io.fixedBufferStream(&buf);
        var w = fbs.writer();
        printStmtInfo(&w, .{ .create_table = .{ .name = "test" } });
        const output = fbs.getWritten();
        try std.testing.expect(std.mem.indexOf(u8, output, "CREATE TABLE test") != null);
    }

    // Test DROP TABLE
    {
        var buf: [256]u8 = undefined;
        var fbs = std.io.fixedBufferStream(&buf);
        var w = fbs.writer();
        printStmtInfo(&w, .{ .drop_table = .{ .name = "test" } });
        const output = fbs.getWritten();
        try std.testing.expect(std.mem.indexOf(u8, output, "DROP TABLE test") != null);
    }

    // Test BEGIN
    {
        var buf: [256]u8 = undefined;
        var fbs = std.io.fixedBufferStream(&buf);
        var w = fbs.writer();
        printStmtInfo(&w, .{ .transaction = .{ .begin = .{} } });
        const output = fbs.getWritten();
        try std.testing.expect(std.mem.indexOf(u8, output, "BEGIN") != null);
    }
}
