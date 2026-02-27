const std = @import("std");
const sailor = @import("sailor");

const version = "0.1.0";

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

    var parser = sailor.arg.Parser(&CliFlags).init(allocator);
    defer parser.deinit();

    parser.parse(args) catch |err| {
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

    if (parser.getBool("help", false)) {
        printUsage(stdout);
        stdout.flush() catch {};
        return;
    }

    if (parser.getBool("version", false)) {
        stdout.print("silica {s}\n", .{version}) catch {};
        stdout.flush() catch {};
        return;
    }

    // First positional argument is the database path
    if (parser.positional.items.len == 0) {
        printUsage(stdout);
        stdout.flush() catch {};
        return;
    }

    const db_path = parser.positional.items[0];
    _ = db_path;

    // TODO: Interactive SQL shell (requires tokenizer/parser — Phase 2)
    printError(stderr, "Interactive SQL shell not yet implemented. Coming in Phase 2.");
    stderr.flush() catch {};
    std.process.exit(1);
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
    try std.testing.expectEqualStrings("0.1.0", version);
}
