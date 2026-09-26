//! Mechanical Tiger Style checks for `zig build tidy`: line length, function
//! length against a ratchet baseline, and missing `//!` module headers.

const std = @import("std");

/// Aliased once per file per Tiger Style — never call `std.debug.assert`
/// inline. Compiled out in ReleaseFast/ReleaseSmall: a programmer-error
/// tripwire, not a guard against malformed source files (those are handled
/// as `error.InvalidBaselineLine` or simply produce a violation).
const assert = std.debug.assert;

/// Documents a condition that is legitimately sometimes true, as distinct
/// from `assert`'s "always". One-line no-op, kept local until zuda ships it.
fn maybe(ok: bool) void {
    _ = ok;
}

/// Category of a tidy violation. Kept small and exhaustive on purpose — the
/// ban-list checks (catch-unreachable-without-SAFETY, debug-print-in-lib,
/// time-in-lib, usize-in-wire-format) are deferred to a follow-up part and
/// are not represented here yet.
pub const Kind = enum { line_too_long, function_too_long, missing_module_header };

/// A single mechanical-check failure, pinned to a file and (where
/// applicable) a line and/or a named function.
pub const Violation = struct {
    path: []const u8,
    /// 1-based line number; 0 for whole-file violations (missing header).
    line: u32,
    kind: Kind,
    /// Function name for `function_too_long`; empty for every other kind.
    name: []const u8,
    /// Measured line length (line_too_long) or function line span
    /// (function_too_long); 0 for missing_module_header.
    count: u32,
};

/// One line of the checked-in tidy baseline: a pre-existing violation the
/// ratchet still tolerates, up to (but not beyond) `limit`.
pub const BaselineEntry = struct {
    kind: Kind,
    path: []const u8,
    /// Function name; empty for non-function kinds (e.g. missing header).
    name: []const u8,
    /// Maximum count this entry still covers; a violation whose count grows
    /// past `limit` is no longer baseline-covered.
    limit: u32,
};

/// Flags every line of `source` longer than `max_cols` columns. Pure,
/// synchronous, no I/O — `path` only labels the produced `Violation`s.
pub fn checkLineLength(
    gpa: std.mem.Allocator,
    path: []const u8,
    source: []const u8,
    max_cols: u32,
    out: *std.ArrayList(Violation),
) !void {
    assert(max_cols > 0);
    assert(path.len > 0);
    maybe(source.len == 0);

    // Bounded by construction: `splitScalar` can never yield more lines than
    // `source.len + 1` (one per byte, plus the empty tail), so there is no
    // separate `lines_max` constant to name here.
    var line_no: u32 = 0;
    var appended: u32 = 0;
    var it = std.mem.splitScalar(u8, source, '\n');
    while (it.next()) |raw_line| {
        line_no += 1;
        const line = stripTrailingCr(raw_line);
        const cols: u32 = @intCast(line.len);
        if (cols > max_cols) {
            try out.append(gpa, .{
                .path = path,
                .line = line_no,
                .kind = .line_too_long,
                .name = "",
                .count = cols,
            });
            appended += 1;
        }
    }

    assert(appended <= line_no);
}

/// Strips one trailing `\r` (CRLF line endings), if present.
fn stripTrailingCr(line: []const u8) []const u8 {
    if (std.mem.endsWith(u8, line, "\r")) return line[0 .. line.len - 1];
    return line;
}

/// Flags every top-level function whose body (from the `fn` line through its
/// own matching closing brace, inclusive) spans more than `max_lines` lines.
/// Must track brace depth so nested blocks (`if`, `for`, struct literals,
/// etc.) inside the body don't end the function early.
pub fn checkFunctionLength(
    gpa: std.mem.Allocator,
    path: []const u8,
    source: []const u8,
    max_lines: u32,
    out: *std.ArrayList(Violation),
) !void {
    assert(max_lines > 0);
    assert(path.len > 0);
    maybe(source.len == 0);

    var lines = std.ArrayList([]const u8){};
    defer lines.deinit(gpa);
    var split = std.mem.splitScalar(u8, source, '\n');
    while (split.next()) |line| try lines.append(gpa, line);
    const lines_total: u32 = @intCast(lines.items.len);

    var appended: u32 = 0;
    var i: u32 = 0;
    while (i < lines_total) : (i += 1) {
        const name = matchFnStart(lines.items[i]) orelse continue;
        const start_line = i + 1; // 1-based.
        const end_line = functionEndLine(lines.items, i);
        const count = end_line - start_line + 1;

        if (count > max_lines) {
            try out.append(gpa, .{
                .path = path,
                .line = start_line,
                .kind = .function_too_long,
                .name = name,
                .count = count,
            });
            appended += 1;
        }
        i = end_line - 1; // Resume scanning after this function's own close.
    }

    assert(appended <= lines_total);
}

/// Returns the top-level function name starting at zero-indented `line`, or
/// `null` if `line` is not a `(pub )?(extern )?(inline )?fn <name>` start —
/// nested/local `fn`s (indented) never match.
fn matchFnStart(line: []const u8) ?[]const u8 {
    if (line.len == 0 or line[0] == ' ' or line[0] == '\t') return null;

    var rest = skipKeyword(line, "pub");
    rest = skipKeyword(rest, "extern");
    rest = skipKeyword(rest, "inline");
    if (!std.mem.startsWith(u8, rest, "fn")) return null;
    rest = rest["fn".len..];
    if (rest.len == 0 or (rest[0] != ' ' and rest[0] != '\t')) return null;
    rest = std.mem.trimLeft(u8, rest, " \t");

    var end: usize = 0;
    while (end < rest.len and isWordChar(rest[end])) end += 1;
    if (end == 0) return null;
    return rest[0..end];
}

/// Consumes `keyword` followed by required whitespace from the front of
/// `line`, returning `line` unchanged if `keyword` is not there.
fn skipKeyword(line: []const u8, keyword: []const u8) []const u8 {
    if (!std.mem.startsWith(u8, line, keyword)) return line;
    const after = line[keyword.len..];
    if (after.len == 0 or (after[0] != ' ' and after[0] != '\t')) return line;
    return std.mem.trimLeft(u8, after, " \t");
}

fn isWordChar(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '_';
}

/// Textual brace-depth scan from `lines[start_index]` (the `fn` line) to the
/// 1-based line where depth first returns to 0 after opening. Known
/// simplification: braces inside string/char literals or comments are not
/// excluded (matches the kingdom `tidy-auditor` agent's grep-based approach).
fn functionEndLine(lines: []const []const u8, start_index: u32) u32 {
    assert(start_index < lines.len);

    var depth: i64 = 0;
    var seen_open = false;
    var j = start_index;
    while (j < lines.len) : (j += 1) {
        for (lines[j]) |c| {
            if (c == '{') {
                depth += 1;
                seen_open = true;
            } else if (c == '}') {
                depth -= 1;
            }
        }
        if (seen_open and depth <= 0) return j + 1;
    }

    assert(j == lines.len);
    return @intCast(lines.len); // Unterminated function: EOF ends it.
}

/// Flags `source` when its first non-blank line does not start with `//!`.
/// An empty file, or one containing only blank lines, also counts as
/// missing a header.
pub fn checkModuleHeader(
    gpa: std.mem.Allocator,
    path: []const u8,
    source: []const u8,
    out: *std.ArrayList(Violation),
) !void {
    assert(path.len > 0);
    maybe(source.len == 0);

    var has_header = false;
    var it = std.mem.splitScalar(u8, source, '\n');
    while (it.next()) |raw_line| {
        const trimmed = std.mem.trim(u8, raw_line, " \t\r");
        if (trimmed.len == 0) continue;
        has_header = std.mem.startsWith(u8, trimmed, "//!");
        break;
    }

    var appended = false;
    if (!has_header) {
        try out.append(gpa, .{
            .path = path,
            .line = 0,
            .kind = .missing_module_header,
            .name = "",
            .count = 0,
        });
        appended = true;
    }

    assert(appended == !has_header);
}

/// Parses the checked-in baseline file format:
///   function_length:<path>:<function_name>:<max_lines>
///   missing_module_header:<path>
/// `#`-prefixed comment lines and blank lines are ignored.
pub fn parseBaseline(gpa: std.mem.Allocator, text: []const u8) !std.ArrayList(BaselineEntry) {
    maybe(text.len == 0);

    var entries = std.ArrayList(BaselineEntry){};
    errdefer entries.deinit(gpa);

    var lines_seen: u32 = 0;
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |raw_line| {
        const line = std.mem.trim(u8, raw_line, " \t\r");
        if (line.len == 0 or line[0] == '#') continue;

        const entry = try parseBaselineLine(line);
        try entries.append(gpa, entry);
        lines_seen += 1;
    }

    assert(entries.items.len == lines_seen);
    return entries;
}

/// Parses one non-blank, non-comment baseline line into an entry, or returns
/// `error.InvalidBaselineLine` for anything that does not match either the
/// `function_length:<path>:<name>:<limit>` or `missing_module_header:<path>`
/// shape — never `unreachable` on user-editable input.
fn parseBaselineLine(line: []const u8) error{InvalidBaselineLine}!BaselineEntry {
    assert(line.len > 0);

    const fields_max = 4;
    var fields: [fields_max][]const u8 = undefined;
    var count: u32 = 0;
    var it = std.mem.splitScalar(u8, line, ':');
    while (it.next()) |field| {
        if (count == fields_max) return error.InvalidBaselineLine;
        fields[count] = field;
        count += 1;
    }
    assert(count <= fields_max);

    if (count == 2 and std.mem.eql(u8, fields[0], "missing_module_header")) {
        return .{ .kind = .missing_module_header, .path = fields[1], .name = "", .limit = 0 };
    }
    if (count == 4 and std.mem.eql(u8, fields[0], "function_length")) {
        const limit = std.fmt.parseInt(u32, fields[3], 10) catch return error.InvalidBaselineLine;
        return .{ .kind = .function_too_long, .path = fields[1], .name = fields[2], .limit = limit };
    }
    return error.InvalidBaselineLine;
}

/// Filters `violations` down to the ones NOT covered by `baseline` — the
/// ones that should fail the build:
///   - `function_too_long` is covered iff a matching (path, name) baseline
///     entry exists with `limit >= violation.count` (grown-beyond-baseline
///     is NOT covered, i.e. still fails: the ratchet).
///   - `missing_module_header` is covered iff a matching (path) baseline
///     entry exists with `kind == .missing_module_header`.
///   - `line_too_long` is never baseline-covered (hard check, no ratchet).
pub fn unbaselined(
    gpa: std.mem.Allocator,
    violations: []const Violation,
    baseline: []const BaselineEntry,
) !std.ArrayList(Violation) {
    maybe(violations.len == 0);
    maybe(baseline.len == 0);

    var kept = std.ArrayList(Violation){};
    errdefer kept.deinit(gpa);

    for (violations) |violation| {
        if (isCovered(violation, baseline)) continue;
        try kept.append(gpa, violation);
    }

    assert(kept.items.len <= violations.len);
    return kept;
}

/// Exhaustive on `Kind` by construction (a `switch` with no `else`) so a
/// future fourth `Kind` fails the build here instead of silently passing.
fn isCovered(violation: Violation, baseline: []const BaselineEntry) bool {
    return switch (violation.kind) {
        .line_too_long => false,
        .missing_module_header => isMissingHeaderCovered(violation, baseline),
        .function_too_long => isFunctionLengthCovered(violation, baseline),
    };
}

fn isMissingHeaderCovered(violation: Violation, baseline: []const BaselineEntry) bool {
    assert(violation.kind == .missing_module_header);
    for (baseline) |entry| {
        if (entry.kind != .missing_module_header) continue;
        if (std.mem.eql(u8, entry.path, violation.path)) return true;
    }
    return false;
}

fn isFunctionLengthCovered(violation: Violation, baseline: []const BaselineEntry) bool {
    assert(violation.kind == .function_too_long);
    for (baseline) |entry| {
        if (entry.kind != .function_too_long) continue;
        if (!std.mem.eql(u8, entry.path, violation.path)) continue;
        if (!std.mem.eql(u8, entry.name, violation.name)) continue;
        if (entry.limit >= violation.count) return true;
    }
    return false;
}

// ── `zig build tidy` CLI ─────────────────────────────────────────────────

/// Every directory nesting depth under `--src` gets counted against this
/// bound before the walk gives up — a defensive limit against a pathological
/// (e.g. cyclic-symlink) tree, not a realistic ceiling for `src/`.
const dirs_max: u32 = 4096;

/// Per-file read cap, comfortably above the largest tracked file
/// (`engine.zig`, ~1.9 MB as of this writing).
const file_bytes_max: usize = 32 * 1024 * 1024;

/// Per-baseline-file read cap — it is checked in, hand-edited, small.
const baseline_bytes_max: usize = 16 * 1024 * 1024;

const max_cols_default: u32 = 100;
const max_lines_default: u32 = 70;

const Options = struct {
    src_dir: []const u8,
    baseline_path: []const u8,
};

/// Parses `--src <dir> --baseline <file>`. Both flags are required —
/// options are always explicit, never defaulted, per Tiger Style.
fn parseArgs(args: []const []const u8) error{InvalidArgs}!Options {
    assert(args.len >= 1); // argv[0] (the program name) is always present.

    var src_dir: ?[]const u8 = null;
    var baseline_path: ?[]const u8 = null;

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--src")) {
            i += 1;
            if (i >= args.len) return error.InvalidArgs;
            src_dir = args[i];
        } else if (std.mem.eql(u8, arg, "--baseline")) {
            i += 1;
            if (i >= args.len) return error.InvalidArgs;
            baseline_path = args[i];
        } else {
            return error.InvalidArgs;
        }
    }

    const options = Options{
        .src_dir = src_dir orelse return error.InvalidArgs,
        .baseline_path = baseline_path orelse return error.InvalidArgs,
    };
    assert(options.src_dir.len > 0);
    assert(options.baseline_path.len > 0);
    return options;
}

/// Recursively collects every `*.zig` file under `root`, skipping
/// dot-directories, using an explicit directory stack instead of recursion
/// (Tiger Style: no recursion on attacker- or at-least-filesystem-shaped
/// input). All returned paths and the scratch stack live in `arena`.
fn collectZigFiles(arena: std.mem.Allocator, root: []const u8) ![][]const u8 {
    assert(root.len > 0);

    var files = std.ArrayList([]const u8){};
    var stack = std.ArrayList([]const u8){};
    try stack.append(arena, root);

    var dirs_visited: u32 = 0;
    while (stack.pop()) |dir_path| {
        dirs_visited += 1;
        assert(dirs_visited <= dirs_max);

        var dir = try std.fs.cwd().openDir(dir_path, .{ .iterate = true });
        defer dir.close();

        var it = dir.iterate();
        while (try it.next()) |entry| {
            if (entry.name.len > 0 and entry.name[0] == '.') continue;
            const full = try std.fs.path.join(arena, &.{ dir_path, entry.name });
            if (entry.kind == .directory) {
                try stack.append(arena, full);
            } else if (entry.kind == .file and std.mem.endsWith(u8, entry.name, ".zig")) {
                try files.append(arena, full);
            }
        }
    }

    assert(dirs_visited >= 1);
    return files.items;
}

/// Reads `path` and runs all three mechanical checks against it, appending
/// any violations to `out`. `arena` backs the file-content and line-index
/// scratch that the checks themselves allocate.
fn checkFile(
    gpa: std.mem.Allocator,
    arena: std.mem.Allocator,
    path: []const u8,
    out: *std.ArrayList(Violation),
) !void {
    assert(path.len > 0);
    assert(std.mem.endsWith(u8, path, ".zig"));

    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    const source = try file.readToEndAlloc(arena, file_bytes_max);
    assert(source.len <= file_bytes_max);

    try checkLineLength(gpa, path, source, max_cols_default, out);
    try checkFunctionLength(gpa, path, source, max_lines_default, out);
    try checkModuleHeader(gpa, path, source, out);
}

/// Reads the baseline file at `path`; a missing file is an empty baseline,
/// not an error — `zig build tidy` must work before the file is created.
fn readBaselineFile(arena: std.mem.Allocator, path: []const u8) ![]const u8 {
    assert(path.len > 0);

    const file = std.fs.cwd().openFile(path, .{}) catch |err| switch (err) {
        error.FileNotFound => return "",
        else => |e| return e,
    };
    defer file.close();

    const text = try file.readToEndAlloc(arena, baseline_bytes_max);
    assert(text.len <= baseline_bytes_max);
    return text;
}

fn printViolation(stderr: *std.Io.Writer, violation: Violation) !void {
    switch (violation.kind) {
        .line_too_long => try stderr.print(
            "{s}:{d}: line_too_long: {d} columns\n",
            .{ violation.path, violation.line, violation.count },
        ),
        .function_too_long => try stderr.print(
            "{s}:{d}: function_too_long: {s} spans {d} lines\n",
            .{ violation.path, violation.line, violation.name, violation.count },
        ),
        .missing_module_header => try stderr.print(
            "{s}: missing_module_header: no leading //! doc comment\n",
            .{violation.path},
        ),
    }
}

/// Walks `options.src_dir`, checks every `*.zig` file found, filters the
/// result through `options.baseline_path`, and prints every remaining
/// violation to `stderr`. Returns the count that should fail the build.
fn run(gpa: std.mem.Allocator, arena: std.mem.Allocator, options: Options, stderr: *std.Io.Writer) !u32 {
    assert(options.src_dir.len > 0);
    assert(options.baseline_path.len > 0);

    const files = try collectZigFiles(arena, options.src_dir);

    var violations = std.ArrayList(Violation){};
    defer violations.deinit(gpa);
    for (files) |path| try checkFile(gpa, arena, path, &violations);

    const baseline_text = try readBaselineFile(arena, options.baseline_path);
    var baseline = try parseBaseline(arena, baseline_text);
    defer baseline.deinit(arena);

    var kept = try unbaselined(gpa, violations.items, baseline.items);
    defer kept.deinit(gpa);

    for (kept.items) |violation| try printViolation(stderr, violation);

    assert(kept.items.len <= violations.items.len);
    return @intCast(kept.items.len);
}

pub fn main() !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const gpa = gpa_state.allocator();

    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var stderr_buf: [4096]u8 = undefined;
    var stderr_writer = std.fs.File.stderr().writer(&stderr_buf);
    const stderr = &stderr_writer.interface;

    const args = try std.process.argsAlloc(arena);
    const options = parseArgs(args) catch |err| switch (err) {
        error.InvalidArgs => {
            try stderr.print("usage: tidy --src <dir> --baseline <file>\n", .{});
            try stderr.flush();
            std.process.exit(2);
        },
    };

    const violation_count = try run(gpa, arena, options, stderr);
    try stderr.flush();

    if (violation_count > 0) std.process.exit(1);
}

// ── Tests ──────────────────────────────────────────────────────────────

const testing = std.testing;

test "checkLineLength flags a line over max_cols but not one at exactly max_cols" {
    const allocator = testing.allocator;
    const line_100 = "a" ** 100;
    const line_101 = "b" ** 101;
    const src = line_100 ++ "\n" ++ line_101 ++ "\n";

    var out = std.ArrayList(Violation){};
    defer out.deinit(allocator);

    try checkLineLength(allocator, "src/example.zig", src, 100, &out);

    try testing.expectEqual(@as(usize, 1), out.items.len);
    try testing.expectEqual(@as(u32, 2), out.items[0].line);
    try testing.expectEqual(Kind.line_too_long, out.items[0].kind);
    try testing.expectEqual(@as(u32, 101), out.items[0].count);
    try testing.expectEqualStrings("", out.items[0].name);
}

test "checkLineLength reports zero violations when every line fits" {
    const allocator = testing.allocator;
    const src = "short line\n" ++ ("z" ** 100) ++ "\ntiny\n";

    var out = std.ArrayList(Violation){};
    defer out.deinit(allocator);

    try checkLineLength(allocator, "src/example.zig", src, 100, &out);

    try testing.expectEqual(@as(usize, 0), out.items.len);
}

test "checkFunctionLength flags a function spanning more than max_lines" {
    const allocator = testing.allocator;
    const src =
        \\pub fn tooLong() void {
        \\    const a = 1;
        \\    const b = 2;
        \\    const c = 3;
        \\    const d = 4;
        \\    const e = 5;
        \\}
    ;

    var out = std.ArrayList(Violation){};
    defer out.deinit(allocator);

    try checkFunctionLength(allocator, "src/example.zig", src, 5, &out);

    try testing.expectEqual(@as(usize, 1), out.items.len);
    try testing.expectEqual(Kind.function_too_long, out.items[0].kind);
    try testing.expectEqualStrings("tooLong", out.items[0].name);
    try testing.expectEqual(@as(u32, 1), out.items[0].line);
    try testing.expectEqual(@as(u32, 7), out.items[0].count);
}

test "checkFunctionLength allows a function at exactly max_lines" {
    const allocator = testing.allocator;
    const src =
        \\pub fn exact() void {
        \\    const a = 1;
        \\    const b = 2;
        \\}
    ;

    var out = std.ArrayList(Violation){};
    defer out.deinit(allocator);

    // src is exactly 4 lines; max_lines == 4 must be allowed (max_lines+1 is not).
    try checkFunctionLength(allocator, "src/example.zig", src, 4, &out);

    try testing.expectEqual(@as(usize, 0), out.items.len);
}

test "checkFunctionLength flags only the offending function among several" {
    const allocator = testing.allocator;
    const src =
        \\pub fn functionA() void {
        \\    const a = 1;
        \\    const b = 2;
        \\    const c = 3;
        \\    const d = 4;
        \\    const e = 5;
        \\}
        \\
        \\pub fn functionB() void {
        \\    const a = 1;
        \\}
    ;

    var out = std.ArrayList(Violation){};
    defer out.deinit(allocator);

    try checkFunctionLength(allocator, "src/example.zig", src, 5, &out);

    try testing.expectEqual(@as(usize, 1), out.items.len);
    try testing.expectEqualStrings("functionA", out.items[0].name);
    try testing.expectEqual(@as(u32, 7), out.items[0].count);
}

test "checkFunctionLength tracks brace depth through nested blocks" {
    const allocator = testing.allocator;
    // The function body contains an `if` block, a struct literal, and a
    // `for` loop, each opening and closing its own brace before the
    // function's own closing brace on the final line. A naive "stop at the
    // first `}`" scanner would end the function at line 4 (count 4) instead
    // of line 11 (count 11).
    const src =
        \\pub fn withNesting() void {
        \\    if (true) {
        \\        const x = 1;
        \\    }
        \\    const s = struct {
        \\        field: i32 = 0,
        \\    }{};
        \\    for (0..3) |i| {
        \\        _ = i;
        \\    }
        \\}
    ;

    var out = std.ArrayList(Violation){};
    defer out.deinit(allocator);

    try checkFunctionLength(allocator, "src/example.zig", src, 5, &out);

    try testing.expectEqual(@as(usize, 1), out.items.len);
    try testing.expectEqualStrings("withNesting", out.items[0].name);
    try testing.expectEqual(@as(u32, 11), out.items[0].count);
}

test "checkModuleHeader accepts a source starting with a //! doc comment" {
    const src = "//! Does something useful.\nconst std = @import(\"std\");\n";

    var out = std.ArrayList(Violation){};
    defer out.deinit(testing.allocator);

    try checkModuleHeader(testing.allocator, "src/example.zig", src, &out);

    try testing.expectEqual(@as(usize, 0), out.items.len);
}

test "checkModuleHeader flags a source whose first line is not a //! doc comment" {
    const allocator = testing.allocator;
    const variants = [_][]const u8{
        "\nconst std = @import(\"std\");\n", // blank first line
        "// plain comment, not a doc comment\nconst std = @import(\"std\");\n",
        "const std = @import(\"std\");\n", // code as the first line
    };

    for (variants) |src| {
        var out = std.ArrayList(Violation){};
        defer out.deinit(allocator);

        try checkModuleHeader(allocator, "src/example.zig", src, &out);

        try testing.expectEqual(@as(usize, 1), out.items.len);
        try testing.expectEqual(Kind.missing_module_header, out.items[0].kind);
        try testing.expectEqual(@as(u32, 0), out.items[0].line);
        try testing.expectEqual(@as(u32, 0), out.items[0].count);
        try testing.expectEqualStrings("", out.items[0].name);
    }
}

test "parseBaseline round-trips both entry kinds around comments and blank lines" {
    const allocator = testing.allocator;
    const text =
        \\# tidy baseline -- do not add entries casually
        \\function_length:src/sql/engine.zig:parseSelect:120
        \\
        \\missing_module_header:src/legacy/old.zig
    ;

    var entries = try parseBaseline(allocator, text);
    defer entries.deinit(allocator);

    try testing.expectEqual(@as(usize, 2), entries.items.len);

    try testing.expectEqual(Kind.function_too_long, entries.items[0].kind);
    try testing.expectEqualStrings("src/sql/engine.zig", entries.items[0].path);
    try testing.expectEqualStrings("parseSelect", entries.items[0].name);
    try testing.expectEqual(@as(u32, 120), entries.items[0].limit);

    try testing.expectEqual(Kind.missing_module_header, entries.items[1].kind);
    try testing.expectEqualStrings("src/legacy/old.zig", entries.items[1].path);
    try testing.expectEqualStrings("", entries.items[1].name);
    try testing.expectEqual(@as(u32, 0), entries.items[1].limit);
}

test "unbaselined filters a function_too_long violation within its baseline limit" {
    const allocator = testing.allocator;
    const violations = [_]Violation{
        .{ .path = "src/sql/engine.zig", .line = 42, .kind = .function_too_long, .name = "parseSelect", .count = 100 },
    };
    const baseline = [_]BaselineEntry{
        .{ .kind = .function_too_long, .path = "src/sql/engine.zig", .name = "parseSelect", .limit = 120 },
    };

    var result = try unbaselined(allocator, &violations, &baseline);
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 0), result.items.len);
}

test "unbaselined keeps a function_too_long violation that grew past its baseline limit" {
    const allocator = testing.allocator;
    const violations = [_]Violation{
        .{ .path = "src/sql/engine.zig", .line = 42, .kind = .function_too_long, .name = "parseSelect", .count = 130 },
    };
    const baseline = [_]BaselineEntry{
        .{ .kind = .function_too_long, .path = "src/sql/engine.zig", .name = "parseSelect", .limit = 120 },
    };

    var result = try unbaselined(allocator, &violations, &baseline);
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 1), result.items.len);
    try testing.expectEqualStrings("src/sql/engine.zig", result.items[0].path);
    try testing.expectEqualStrings("parseSelect", result.items[0].name);
    try testing.expectEqual(@as(u32, 130), result.items[0].count);
    try testing.expectEqual(Kind.function_too_long, result.items[0].kind);
}

test "unbaselined keeps a function_too_long violation with no matching baseline entry" {
    const allocator = testing.allocator;
    const violations = [_]Violation{
        .{ .path = "src/sql/engine.zig", .line = 7, .kind = .function_too_long, .name = "brandNewOffender", .count = 90 },
    };
    const baseline = [_]BaselineEntry{
        .{ .kind = .function_too_long, .path = "src/sql/engine.zig", .name = "parseSelect", .limit = 120 },
    };

    var result = try unbaselined(allocator, &violations, &baseline);
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 1), result.items.len);
    try testing.expectEqualStrings("brandNewOffender", result.items[0].name);
}

test "unbaselined filters a missing_module_header violation with a matching path entry" {
    const allocator = testing.allocator;
    const violations = [_]Violation{
        .{ .path = "src/legacy/old.zig", .line = 0, .kind = .missing_module_header, .name = "", .count = 0 },
    };
    const baseline = [_]BaselineEntry{
        .{ .kind = .missing_module_header, .path = "src/legacy/old.zig", .name = "", .limit = 0 },
    };

    var result = try unbaselined(allocator, &violations, &baseline);
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 0), result.items.len);
}

test "unbaselined never covers line_too_long regardless of baseline contents" {
    const allocator = testing.allocator;
    const violations = [_]Violation{
        .{ .path = "src/util/varint.zig", .line = 5, .kind = .line_too_long, .name = "", .count = 115 },
    };
    // Baseline entries for the same path, under other kinds, must not leak
    // coverage onto the hard line-length check.
    const baseline = [_]BaselineEntry{
        .{ .kind = .function_too_long, .path = "src/util/varint.zig", .name = "", .limit = 999 },
        .{ .kind = .missing_module_header, .path = "src/util/varint.zig", .name = "", .limit = 0 },
    };

    var result = try unbaselined(allocator, &violations, &baseline);
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 1), result.items.len);
    try testing.expectEqual(Kind.line_too_long, result.items[0].kind);
    try testing.expectEqual(@as(u32, 115), result.items[0].count);
}
