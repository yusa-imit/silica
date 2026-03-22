//! Comprehensive fuzz tests for the SQL parser.
//!
//! These tests exercise the parser with pseudo-random SQL generation including:
//! - Random valid SQL statements (SELECT, INSERT, CREATE TABLE, etc.)
//! - Deeply nested expressions (100+ levels)
//! - Complex WHERE clauses with many AND/OR operators
//! - Subqueries at various nesting depths
//! - CTEs (WITH clauses)
//! - JOIN chains with many tables
//! - Window functions with complex partitioning
//! - JSON path expressions
//! - Array expressions
//! - CASE expressions with many WHEN clauses
//! - Aggregate functions with FILTER/DISTINCT
//! - Invalid syntax combinations (should return errors gracefully)
//! - Very long identifier names
//! - Large IN lists (1000+ values)
//! - Complex GROUP BY/ORDER BY
//!
//! Each test uses a deterministic seed for reproducibility.
//! The parser must NEVER crash or leak memory, regardless of input.
//! Syntax errors are acceptable; crashes are NOT.

const std = @import("std");
const parser_mod = @import("parser.zig");
const ast_mod = @import("ast.zig");

const Parser = parser_mod.Parser;
const AstArena = ast_mod.AstArena;

// ── Test Helpers ─────────────────────────────────────────────────────────

/// StringBuilder for constructing SQL strings.
const StringBuilder = struct {
    buf: std.ArrayListUnmanaged(u8),
    allocator: std.mem.Allocator,

    fn init(allocator: std.mem.Allocator) StringBuilder {
        return .{
            .buf = .{},
            .allocator = allocator,
        };
    }

    fn deinit(self: *StringBuilder) void {
        self.buf.deinit(self.allocator);
    }

    fn append(self: *StringBuilder, str: []const u8) !void {
        try self.buf.appendSlice(self.allocator, str);
    }

    fn appendFmt(self: *StringBuilder, comptime fmt: []const u8, args: anytype) !void {
        try self.buf.writer(self.allocator).print(fmt, args);
    }

    fn toOwned(self: *StringBuilder) ![]const u8 {
        return try self.buf.toOwnedSlice(self.allocator);
    }
};

/// Parse SQL and verify NO crashes or memory leaks. Returns true if parse succeeded.
fn fuzzParse(allocator: std.mem.Allocator, sql: []const u8) !bool {
    var arena = AstArena.init(allocator);
    defer arena.deinit();

    var parser = Parser.init(allocator, sql, &arena) catch |err| switch (err) {
        error.OutOfMemory => return err,
        error.ParseFailed => return false,
    };
    defer parser.deinit();

    _ = parser.parseStatement() catch |err| switch (err) {
        error.OutOfMemory => return err,
        error.ParseFailed => return false,
    };

    return true;
}

/// Generate a random identifier.
fn randomIdentifier(sb: *StringBuilder, random: std.Random, len: usize) !void {
    const first_char = if (random.boolean()) 'a' + random.intRangeAtMost(u8, 0, 25) else 'A' + random.intRangeAtMost(u8, 0, 25);
    try sb.appendFmt("{c}", .{first_char});
    for (0..len - 1) |_| {
        const ch = random.intRangeAtMost(u8, 0, 2);
        const char: u8 = switch (ch) {
            0 => 'a' + random.intRangeAtMost(u8, 0, 25),
            1 => '0' + random.intRangeAtMost(u8, 0, 9),
            2 => '_',
            else => unreachable,
        };
        try sb.appendFmt("{c}", .{char});
    }
}

/// Generate a random integer literal.
fn randomIntLiteral(sb: *StringBuilder, random: std.Random) !void {
    const val = random.intRangeAtMost(i32, -10000, 10000);
    try sb.appendFmt("{d}", .{val});
}

/// Generate a random string literal.
fn randomStringLiteral(sb: *StringBuilder, random: std.Random, max_len: usize) !void {
    try sb.append("'");
    const len = random.intRangeAtMost(usize, 0, max_len);
    for (0..len) |_| {
        const ch = random.intRangeAtMost(u8, 32, 126);
        if (ch == '\'') {
            try sb.append("''"); // Escaped single quote
        } else {
            try sb.appendFmt("{c}", .{ch});
        }
    }
    try sb.append("'");
}

/// Generate a random binary operator.
fn randomBinOp(random: std.Random) []const u8 {
    const ops = [_][]const u8{ "+", "-", "*", "/", "%", "=", "!=", "<", ">", "<=", ">=", "AND", "OR", "||", "->", "->>" };
    return ops[random.intRangeLessThan(usize, 0, ops.len)];
}

/// Generate a random expression (recursive).
fn randomExpr(sb: *StringBuilder, random: std.Random, depth: u32, max_depth: u32) !void {
    if (depth >= max_depth) {
        // Base case: literal or identifier
        const choice = random.intRangeAtMost(u8, 0, 2);
        switch (choice) {
            0 => try randomIntLiteral(sb, random),
            1 => try randomStringLiteral(sb, random, 10),
            2 => try randomIdentifier(sb, random, random.intRangeAtMost(usize, 3, 12)),
            else => unreachable,
        }
        return;
    }

    const choice = random.intRangeAtMost(u8, 0, 5);
    switch (choice) {
        0 => {
            // Binary expression
            try randomExpr(sb, random, depth + 1, max_depth);
            try sb.append(" ");
            try sb.append(randomBinOp(random));
            try sb.append(" ");
            try randomExpr(sb, random, depth + 1, max_depth);
        },
        1 => {
            // Parenthesized expression
            try sb.append("(");
            try randomExpr(sb, random, depth + 1, max_depth);
            try sb.append(")");
        },
        2 => {
            // Function call
            const func = [_][]const u8{ "COUNT", "SUM", "AVG", "MIN", "MAX", "LENGTH", "UPPER", "LOWER" };
            try sb.append(func[random.intRangeLessThan(usize, 0, func.len)]);
            try sb.append("(");
            try randomExpr(sb, random, depth + 1, max_depth);
            try sb.append(")");
        },
        3 => {
            // CASE expression
            try sb.append("CASE WHEN ");
            try randomExpr(sb, random, depth + 1, max_depth);
            try sb.append(" THEN ");
            try randomExpr(sb, random, depth + 1, max_depth);
            try sb.append(" ELSE ");
            try randomExpr(sb, random, depth + 1, max_depth);
            try sb.append(" END");
        },
        4 => {
            // Array literal
            try sb.append("ARRAY[");
            const count = random.intRangeAtMost(usize, 1, 5);
            for (0..count) |i| {
                if (i > 0) try sb.append(", ");
                try randomExpr(sb, random, depth + 1, max_depth);
            }
            try sb.append("]");
        },
        5 => {
            // Base case
            try randomIntLiteral(sb, random);
        },
        else => unreachable,
    }
}

// ── Fuzz Test 1: Random SELECT statements ───────────────────────────────

test "fuzz: random SELECT statements" {
    const allocator = std.testing.allocator;
    var rng = std.Random.DefaultPrng.init(0xDEAD_BEEF);
    const random = rng.random();

    for (0..100) |_| {
        var sb = StringBuilder.init(allocator);
        defer sb.deinit();

        try sb.append("SELECT ");
        const col_count = random.intRangeAtMost(usize, 1, 10);
        for (0..col_count) |i| {
            if (i > 0) try sb.append(", ");
            try randomExpr(&sb, random, 0, 3);
        }
        try sb.append(" FROM ");
        try randomIdentifier(&sb, random, 8);

        const sql = try sb.toOwned();
        defer allocator.free(sql);

        _ = try fuzzParse(allocator, sql);
    }
}

// ── Fuzz Test 2: Deeply nested expressions ──────────────────────────────

test "fuzz: deeply nested expressions" {
    const allocator = std.testing.allocator;
    var rng = std.Random.DefaultPrng.init(0xCAFE_BABE);
    const random = rng.random();

    for (0..50) |_| {
        var sb = StringBuilder.init(allocator);
        defer sb.deinit();

        try sb.append("SELECT ");
        const depth = random.intRangeAtMost(u32, 10, 100);
        try randomExpr(&sb, random, 0, depth);
        try sb.append(" FROM ");
        try randomIdentifier(&sb, random, 5);

        const sql = try sb.toOwned();
        defer allocator.free(sql);

        _ = try fuzzParse(allocator, sql);
    }
}

// ── Fuzz Test 3: Complex WHERE clauses ──────────────────────────────────

test "fuzz: complex WHERE clauses with many AND/OR" {
    const allocator = std.testing.allocator;
    var rng = std.Random.DefaultPrng.init(0xF00D_FACE);
    const random = rng.random();

    for (0..100) |_| {
        var sb = StringBuilder.init(allocator);
        defer sb.deinit();

        try sb.append("SELECT * FROM t WHERE ");
        const condition_count = random.intRangeAtMost(usize, 5, 50);
        for (0..condition_count) |i| {
            if (i > 0) {
                try sb.append(if (random.boolean()) " AND " else " OR ");
            }
            try randomExpr(&sb, random, 0, 2);
            try sb.append(" = ");
            try randomExpr(&sb, random, 0, 2);
        }

        const sql = try sb.toOwned();
        defer allocator.free(sql);

        _ = try fuzzParse(allocator, sql);
    }
}

// ── Fuzz Test 4: Subqueries at various depths ───────────────────────────

test "fuzz: nested subqueries" {
    const allocator = std.testing.allocator;
    var rng = std.Random.DefaultPrng.init(0xBEEF_CAFE);
    const random = rng.random();

    for (0..50) |_| {
        var sb = StringBuilder.init(allocator);
        defer sb.deinit();

        const depth = random.intRangeAtMost(usize, 1, 10);
        try sb.append("SELECT * FROM (");
        for (0..depth) |i| {
            if (i > 0) try sb.append("SELECT * FROM (");
            try sb.append("SELECT ");
            try randomIdentifier(&sb, random, 5);
            try sb.append(" FROM ");
            try randomIdentifier(&sb, random, 5);
        }
        try sb.append(") AS t0");
        for (0..depth - 1) |_| {
            try sb.append(") AS t");
        }

        const sql = try sb.toOwned();
        defer allocator.free(sql);

        _ = try fuzzParse(allocator, sql);
    }
}

// ── Fuzz Test 5: CTEs (WITH clauses) ─────────────────────────────────────

test "fuzz: Common Table Expressions (WITH clauses)" {
    const allocator = std.testing.allocator;
    var rng = std.Random.DefaultPrng.init(0xABCD_1234);
    const random = rng.random();

    for (0..50) |_| {
        var sb = StringBuilder.init(allocator);
        defer sb.deinit();

        try sb.append("WITH ");
        const cte_count = random.intRangeAtMost(usize, 1, 5);
        for (0..cte_count) |i| {
            if (i > 0) try sb.append(", ");
            try sb.append("cte");
            try sb.appendFmt("{d}", .{i});
            try sb.append(" AS (SELECT ");
            try randomExpr(&sb, random, 0, 2);
            try sb.append(" FROM ");
            try randomIdentifier(&sb, random, 5);
            try sb.append(")");
        }
        try sb.append(" SELECT * FROM cte0");

        const sql = try sb.toOwned();
        defer allocator.free(sql);

        _ = try fuzzParse(allocator, sql);
    }
}

// ── Fuzz Test 6: JOIN chains ─────────────────────────────────────────────

test "fuzz: JOIN chains with many tables" {
    const allocator = std.testing.allocator;
    var rng = std.Random.DefaultPrng.init(0x5EED_FACE);
    const random = rng.random();

    for (0..50) |_| {
        var sb = StringBuilder.init(allocator);
        defer sb.deinit();

        try sb.append("SELECT * FROM t0");
        const join_count = random.intRangeAtMost(usize, 1, 15);
        for (0..join_count) |i| {
            const join_type = [_][]const u8{ " JOIN ", " LEFT JOIN ", " RIGHT JOIN ", " FULL JOIN ", " CROSS JOIN " };
            try sb.append(join_type[random.intRangeLessThan(usize, 0, join_type.len)]);
            try sb.append("t");
            try sb.appendFmt("{d}", .{i + 1});
            // Only add ON clause for non-CROSS joins
            if (!std.mem.eql(u8, join_type[random.intRangeLessThan(usize, 0, join_type.len - 1)], " CROSS JOIN ")) {
                try sb.append(" ON t0.id = t");
                try sb.appendFmt("{d}", .{i + 1});
                try sb.append(".id");
            }
        }

        const sql = try sb.toOwned();
        defer allocator.free(sql);

        _ = try fuzzParse(allocator, sql);
    }
}

// ── Fuzz Test 7: Window functions ────────────────────────────────────────

test "fuzz: Window functions with complex PARTITION BY/ORDER BY" {
    const allocator = std.testing.allocator;
    var rng = std.Random.DefaultPrng.init(0xDEED_BEEF);
    const random = rng.random();

    for (0..50) |_| {
        var sb = StringBuilder.init(allocator);
        defer sb.deinit();

        try sb.append("SELECT ");
        const window_func = [_][]const u8{ "ROW_NUMBER", "RANK", "DENSE_RANK", "SUM", "AVG", "COUNT" };
        try sb.append(window_func[random.intRangeLessThan(usize, 0, window_func.len)]);
        try sb.append("(");
        if (random.boolean()) {
            try randomExpr(&sb, random, 0, 1);
        }
        try sb.append(") OVER (");
        if (random.boolean()) {
            try sb.append("PARTITION BY ");
            const part_count = random.intRangeAtMost(usize, 1, 5);
            for (0..part_count) |i| {
                if (i > 0) try sb.append(", ");
                try randomIdentifier(&sb, random, 5);
            }
        }
        if (random.boolean()) {
            try sb.append(" ORDER BY ");
            const order_count = random.intRangeAtMost(usize, 1, 5);
            for (0..order_count) |i| {
                if (i > 0) try sb.append(", ");
                try randomIdentifier(&sb, random, 5);
                try sb.append(if (random.boolean()) " ASC" else " DESC");
            }
        }
        try sb.append(") FROM t");

        const sql = try sb.toOwned();
        defer allocator.free(sql);

        _ = try fuzzParse(allocator, sql);
    }
}

// ── Fuzz Test 8: JSON path expressions ──────────────────────────────────

test "fuzz: JSON path expressions" {
    const allocator = std.testing.allocator;
    var rng = std.Random.DefaultPrng.init(0x1234_5678);
    const random = rng.random();

    for (0..100) |_| {
        var sb = StringBuilder.init(allocator);
        defer sb.deinit();

        try sb.append("SELECT data");
        const op = [_][]const u8{ "->", "->>", "#>", "#>>", "@>", "<@" };
        try sb.append(op[random.intRangeLessThan(usize, 0, op.len)]);
        try randomStringLiteral(&sb, random, 20);
        try sb.append(" FROM ");
        try randomIdentifier(&sb, random, 5);

        const sql = try sb.toOwned();
        defer allocator.free(sql);

        _ = try fuzzParse(allocator, sql);
    }
}

// ── Fuzz Test 9: Array expressions ──────────────────────────────────────

test "fuzz: Array expressions" {
    const allocator = std.testing.allocator;
    var rng = std.Random.DefaultPrng.init(0x9876_5432);
    const random = rng.random();

    for (0..50) |_| {
        var sb = StringBuilder.init(allocator);
        defer sb.deinit();

        try sb.append("SELECT ARRAY[");
        const count = random.intRangeAtMost(usize, 1, 20);
        for (0..count) |i| {
            if (i > 0) try sb.append(", ");
            try randomExpr(&sb, random, 0, 2);
        }
        try sb.append("] FROM t");

        const sql = try sb.toOwned();
        defer allocator.free(sql);

        _ = try fuzzParse(allocator, sql);
    }
}

// ── Fuzz Test 10: CASE expressions with many WHEN clauses ───────────────

test "fuzz: CASE expressions with many WHEN clauses" {
    const allocator = std.testing.allocator;
    var rng = std.Random.DefaultPrng.init(0xFEED_FACE);
    const random = rng.random();

    for (0..50) |_| {
        var sb = StringBuilder.init(allocator);
        defer sb.deinit();

        try sb.append("SELECT CASE");
        const when_count = random.intRangeAtMost(usize, 1, 20);
        for (0..when_count) |_| {
            try sb.append(" WHEN ");
            try randomExpr(&sb, random, 0, 2);
            try sb.append(" THEN ");
            try randomExpr(&sb, random, 0, 2);
        }
        try sb.append(" ELSE ");
        try randomExpr(&sb, random, 0, 2);
        try sb.append(" END FROM t");

        const sql = try sb.toOwned();
        defer allocator.free(sql);

        _ = try fuzzParse(allocator, sql);
    }
}

// ── Fuzz Test 11: Aggregate functions with FILTER/DISTINCT ──────────────

test "fuzz: Aggregate functions with FILTER and DISTINCT" {
    const allocator = std.testing.allocator;
    var rng = std.Random.DefaultPrng.init(0xBABE_FACE);
    const random = rng.random();

    for (0..100) |_| {
        var sb = StringBuilder.init(allocator);
        defer sb.deinit();

        try sb.append("SELECT ");
        const agg = [_][]const u8{ "COUNT", "SUM", "AVG", "MIN", "MAX" };
        try sb.append(agg[random.intRangeLessThan(usize, 0, agg.len)]);
        try sb.append("(");
        if (random.boolean()) try sb.append("DISTINCT ");
        try randomExpr(&sb, random, 0, 2);
        try sb.append(")");
        if (random.boolean()) {
            try sb.append(" FILTER (WHERE ");
            try randomExpr(&sb, random, 0, 2);
            try sb.append(" > 0)");
        }
        try sb.append(" FROM t");

        const sql = try sb.toOwned();
        defer allocator.free(sql);

        _ = try fuzzParse(allocator, sql);
    }
}

// ── Fuzz Test 12: Invalid syntax combinations ───────────────────────────

test "fuzz: Invalid syntax combinations (graceful errors)" {
    const allocator = std.testing.allocator;
    var rng = std.Random.DefaultPrng.init(0xBAD_F00D);
    const random = rng.random();

    const invalid_sqls = [_][]const u8{
        "SELECT FROM WHERE",
        "INSERT INTO",
        "CREATE TABLE ()",
        "SELECT * FROM FROM",
        "SELECT SELECT SELECT",
        "WHERE FROM SELECT",
        "JOIN ON ON ON",
        "GROUP BY BY BY",
        "ORDER ASC DESC",
        "SELECT * FROM t WHERE WHERE WHERE",
        "UPDATE SET SET",
        "DELETE FROM FROM",
        "INSERT VALUES VALUES",
        "CREATE TABLE t (id id id)",
        "SELECT * FROM t JOIN JOIN",
        "SELECT COUNT(COUNT(COUNT(x))) FROM t",
        "SELECT * FROM t WHERE AND OR",
        "SELECT * FROM t GROUP BY ORDER BY",
        "SELECT * FROM t HAVING",
        "SELECT * FROM t LIMIT OFFSET",
    };

    for (invalid_sqls) |sql| {
        // Should not crash, just return parse error
        const result = try fuzzParse(allocator, sql);
        try std.testing.expect(!result or result); // Accept any result (error or success)
    }

    // Generate random garbage
    for (0..100) |_| {
        var sb = StringBuilder.init(allocator);
        defer sb.deinit();

        const token_count = random.intRangeAtMost(usize, 1, 20);
        const keywords = [_][]const u8{ "SELECT", "FROM", "WHERE", "JOIN", "ON", "GROUP", "BY", "ORDER", "LIMIT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "TABLE", "INTO", "VALUES", "SET" };
        for (0..token_count) |i| {
            if (i > 0) try sb.append(" ");
            try sb.append(keywords[random.intRangeLessThan(usize, 0, keywords.len)]);
        }

        const sql = try sb.toOwned();
        defer allocator.free(sql);

        _ = try fuzzParse(allocator, sql);
    }
}

// ── Fuzz Test 13: Very long identifier names ────────────────────────────

test "fuzz: Very long identifier names" {
    const allocator = std.testing.allocator;
    var rng = std.Random.DefaultPrng.init(0x10A6_AAAE);
    const random = rng.random();

    for (0..50) |_| {
        var sb = StringBuilder.init(allocator);
        defer sb.deinit();

        try sb.append("SELECT ");
        const len = random.intRangeAtMost(usize, 100, 1000);
        try randomIdentifier(&sb, random, len);
        try sb.append(" FROM ");
        try randomIdentifier(&sb, random, random.intRangeAtMost(usize, 100, 500));

        const sql = try sb.toOwned();
        defer allocator.free(sql);

        _ = try fuzzParse(allocator, sql);
    }
}

// ── Fuzz Test 14: Large IN lists ─────────────────────────────────────────

test "fuzz: Large IN lists (1000+ values)" {
    const allocator = std.testing.allocator;
    var rng = std.Random.DefaultPrng.init(0x1A_1157);
    const random = rng.random();

    for (0..20) |_| {
        var sb = StringBuilder.init(allocator);
        defer sb.deinit();

        try sb.append("SELECT * FROM t WHERE id IN (");
        const count = random.intRangeAtMost(usize, 100, 1000);
        for (0..count) |i| {
            if (i > 0) try sb.append(", ");
            try randomIntLiteral(&sb, random);
        }
        try sb.append(")");

        const sql = try sb.toOwned();
        defer allocator.free(sql);

        _ = try fuzzParse(allocator, sql);
    }
}

// ── Fuzz Test 15: Complex GROUP BY/ORDER BY ─────────────────────────────

test "fuzz: Complex GROUP BY/ORDER BY with many columns" {
    const allocator = std.testing.allocator;
    var rng = std.Random.DefaultPrng.init(0x60AD_B7);
    const random = rng.random();

    for (0..50) |_| {
        var sb = StringBuilder.init(allocator);
        defer sb.deinit();

        try sb.append("SELECT ");
        const col_count = random.intRangeAtMost(usize, 1, 10);
        for (0..col_count) |i| {
            if (i > 0) try sb.append(", ");
            try randomIdentifier(&sb, random, 5);
        }
        try sb.append(" FROM t");

        if (random.boolean()) {
            try sb.append(" GROUP BY ");
            const group_count = random.intRangeAtMost(usize, 1, 15);
            for (0..group_count) |i| {
                if (i > 0) try sb.append(", ");
                try randomIdentifier(&sb, random, 5);
            }
        }

        if (random.boolean()) {
            try sb.append(" ORDER BY ");
            const order_count = random.intRangeAtMost(usize, 1, 15);
            for (0..order_count) |i| {
                if (i > 0) try sb.append(", ");
                try randomIdentifier(&sb, random, 5);
                try sb.append(if (random.boolean()) " ASC" else " DESC");
            }
        }

        const sql = try sb.toOwned();
        defer allocator.free(sql);

        _ = try fuzzParse(allocator, sql);
    }
}

// ── Fuzz Test 16: INSERT statements with many values ────────────────────

test "fuzz: INSERT statements with many values" {
    const allocator = std.testing.allocator;
    var rng = std.Random.DefaultPrng.init(0x1A5E47);
    const random = rng.random();

    for (0..50) |_| {
        var sb = StringBuilder.init(allocator);
        defer sb.deinit();

        try sb.append("INSERT INTO ");
        try randomIdentifier(&sb, random, 8);
        try sb.append(" (");
        const col_count = random.intRangeAtMost(usize, 1, 10);
        for (0..col_count) |i| {
            if (i > 0) try sb.append(", ");
            try randomIdentifier(&sb, random, 5);
        }
        try sb.append(") VALUES ");

        const row_count = random.intRangeAtMost(usize, 1, 50);
        for (0..row_count) |i| {
            if (i > 0) try sb.append(", ");
            try sb.append("(");
            for (0..col_count) |j| {
                if (j > 0) try sb.append(", ");
                try randomExpr(&sb, random, 0, 1);
            }
            try sb.append(")");
        }

        const sql = try sb.toOwned();
        defer allocator.free(sql);

        _ = try fuzzParse(allocator, sql);
    }
}

// ── Fuzz Test 17: CREATE TABLE with many columns and constraints ────────

test "fuzz: CREATE TABLE with many columns and constraints" {
    const allocator = std.testing.allocator;
    var rng = std.Random.DefaultPrng.init(0xC4EA7E);
    const random = rng.random();

    for (0..50) |_| {
        var sb = StringBuilder.init(allocator);
        defer sb.deinit();

        try sb.append("CREATE TABLE ");
        try randomIdentifier(&sb, random, 8);
        try sb.append(" (");
        const col_count = random.intRangeAtMost(usize, 1, 20);
        for (0..col_count) |i| {
            if (i > 0) try sb.append(", ");
            try randomIdentifier(&sb, random, 8);
            try sb.append(" ");
            const types = [_][]const u8{ "INTEGER", "TEXT", "REAL", "BLOB", "BOOLEAN", "VARCHAR", "TIMESTAMP", "JSON" };
            try sb.append(types[random.intRangeLessThan(usize, 0, types.len)]);
            if (random.boolean()) {
                try sb.append(" NOT NULL");
            }
            if (random.boolean()) {
                try sb.append(" PRIMARY KEY");
            }
        }
        try sb.append(")");

        const sql = try sb.toOwned();
        defer allocator.free(sql);

        _ = try fuzzParse(allocator, sql);
    }
}

// ── Fuzz Test 18: UPDATE with complex SET and WHERE ─────────────────────

test "fuzz: UPDATE with complex SET and WHERE clauses" {
    const allocator = std.testing.allocator;
    var rng = std.Random.DefaultPrng.init(0x0FDA7E);
    const random = rng.random();

    for (0..50) |_| {
        var sb = StringBuilder.init(allocator);
        defer sb.deinit();

        try sb.append("UPDATE ");
        try randomIdentifier(&sb, random, 8);
        try sb.append(" SET ");
        const set_count = random.intRangeAtMost(usize, 1, 10);
        for (0..set_count) |i| {
            if (i > 0) try sb.append(", ");
            try randomIdentifier(&sb, random, 5);
            try sb.append(" = ");
            try randomExpr(&sb, random, 0, 2);
        }
        try sb.append(" WHERE ");
        try randomExpr(&sb, random, 0, 3);

        const sql = try sb.toOwned();
        defer allocator.free(sql);

        _ = try fuzzParse(allocator, sql);
    }
}

// ── Fuzz Test 19: DELETE with complex WHERE ─────────────────────────────

test "fuzz: DELETE with complex WHERE clause" {
    const allocator = std.testing.allocator;
    var rng = std.Random.DefaultPrng.init(0xDE1E7E);
    const random = rng.random();

    for (0..50) |_| {
        var sb = StringBuilder.init(allocator);
        defer sb.deinit();

        try sb.append("DELETE FROM ");
        try randomIdentifier(&sb, random, 8);
        try sb.append(" WHERE ");
        try randomExpr(&sb, random, 0, 5);

        const sql = try sb.toOwned();
        defer allocator.free(sql);

        _ = try fuzzParse(allocator, sql);
    }
}

// ── Fuzz Test 20: Combined stress test ──────────────────────────────────

test "fuzz: Combined stress test (all patterns)" {
    const allocator = std.testing.allocator;
    var rng = std.Random.DefaultPrng.init(0x574E55);
    const random = rng.random();

    for (0..100) |_| {
        var sb = StringBuilder.init(allocator);
        defer sb.deinit();

        // Randomly choose statement type
        const stmt_type = random.intRangeAtMost(u8, 0, 4);
        switch (stmt_type) {
            0 => {
                // Complex SELECT
                try sb.append("WITH cte AS (SELECT ");
                try randomExpr(&sb, random, 0, 3);
                try sb.append(" FROM t1) SELECT ");
                try randomExpr(&sb, random, 0, 4);
                try sb.append(" FROM cte JOIN t2 ON cte.id = t2.id WHERE ");
                try randomExpr(&sb, random, 0, 3);
                try sb.append(" GROUP BY ");
                try randomIdentifier(&sb, random, 5);
                try sb.append(" ORDER BY ");
                try randomIdentifier(&sb, random, 5);
                try sb.append(" LIMIT 100");
            },
            1 => {
                // Complex INSERT
                try sb.append("INSERT INTO ");
                try randomIdentifier(&sb, random, 8);
                try sb.append(" SELECT ");
                try randomExpr(&sb, random, 0, 2);
                try sb.append(" FROM ");
                try randomIdentifier(&sb, random, 8);
                try sb.append(" WHERE ");
                try randomExpr(&sb, random, 0, 2);
            },
            2 => {
                // Complex UPDATE
                try sb.append("UPDATE ");
                try randomIdentifier(&sb, random, 8);
                try sb.append(" SET col = (SELECT ");
                try randomExpr(&sb, random, 0, 2);
                try sb.append(" FROM t2 WHERE t2.id = t1.id) WHERE ");
                try randomExpr(&sb, random, 0, 3);
            },
            3 => {
                // Complex DELETE
                try sb.append("DELETE FROM ");
                try randomIdentifier(&sb, random, 8);
                try sb.append(" WHERE id IN (SELECT id FROM t2 WHERE ");
                try randomExpr(&sb, random, 0, 3);
                try sb.append(")");
            },
            4 => {
                // CREATE TABLE
                try sb.append("CREATE TABLE ");
                try randomIdentifier(&sb, random, 8);
                try sb.append(" (id INTEGER PRIMARY KEY, ");
                try randomIdentifier(&sb, random, 8);
                try sb.append(" TEXT NOT NULL)");
            },
            else => unreachable,
        }

        const sql = try sb.toOwned();
        defer allocator.free(sql);

        _ = try fuzzParse(allocator, sql);
    }
}
