//! Comprehensive fuzz tests for the SQL tokenizer.
//!
//! These tests exercise the tokenizer with pseudo-random inputs including:
//! - Random byte sequences (valid/invalid UTF-8)
//! - Edge cases: very long tokens, unclosed strings, malformed numbers
//! - Unicode identifiers and operators
//! - Mixed valid/invalid token sequences
//! - Operator fuzzing (valid/invalid combinations)
//!
//! Each test uses a deterministic seed for reproducibility.
//! The tokenizer must NEVER crash or leak memory, regardless of input.
//! Syntax errors are acceptable; crashes are NOT.

const std = @import("std");
const tokenizer_mod = @import("tokenizer.zig");

const Tokenizer = tokenizer_mod.Tokenizer;
const TokenType = tokenizer_mod.TokenType;
const Token = tokenizer_mod.Token;

// ── Test Helpers ─────────────────────────────────────────────────────────

/// Generate a random printable ASCII string of given length.
fn randomPrintableAscii(allocator: std.mem.Allocator, random: std.Random, len: usize) ![]u8 {
    const buf = try allocator.alloc(u8, len);
    for (buf) |*b| {
        b.* = random.intRangeAtMost(u8, 32, 126); // Space to ~
    }
    return buf;
}

/// Generate a random byte sequence (possibly invalid UTF-8).
fn randomBytes(allocator: std.mem.Allocator, random: std.Random, len: usize) ![]u8 {
    const buf = try allocator.alloc(u8, len);
    for (buf) |*b| {
        b.* = random.int(u8);
    }
    return buf;
}

/// Generate a random SQL keyword (case variations).
fn randomKeyword(random: std.Random) []const u8 {
    const keywords = [_][]const u8{
        "SELECT", "select", "SeLeCt",
        "INSERT", "insert", "InSeRt",
        "UPDATE", "update", "UpDaTe",
        "DELETE", "delete", "DeLeTe",
        "CREATE", "create", "CrEaTe",
        "DROP", "drop", "DrOp",
        "TABLE", "table", "TaBlE",
        "WHERE", "where", "WhErE",
        "FROM", "from", "FrOm",
        "JOIN", "join", "JoIn",
    };
    return keywords[random.intRangeLessThan(usize, 0, keywords.len)];
}

/// Generate a random operator string.
fn randomOperator(random: std.Random) []const u8 {
    const ops = [_][]const u8{
        "=",  "==", "!=", "<>", "<", ">", "<=", ">=",
        "+",  "-",  "*",  "/",  "%", "||", "<<", ">>",
        "&",  "|",  "~",  "->", "->>", "@>", "<@",
        "?",  "?|", "?&", "#>", "#>>", "#-", "@@",
    };
    return ops[random.intRangeLessThan(usize, 0, ops.len)];
}

/// Tokenize input and verify NO crashes or memory leaks. Return token count.
fn fuzzTokenize(allocator: std.mem.Allocator, input: []const u8) !usize {
    var tokenizer = Tokenizer.init(input);
    const tokens = tokenizer.tokenize(allocator) catch |err| switch (err) {
        error.OutOfMemory => return err,
        else => return 0, // Other errors are acceptable (syntax errors)
    };
    defer allocator.free(tokens);
    return tokens.len;
}

// ── Fuzz Test 1: Random printable ASCII sequences ───────────────────────

test "fuzz: random printable ASCII input" {
    const allocator = std.testing.allocator;
    var rng = std.Random.DefaultPrng.init(0xDEAD_BEEF);
    const random = rng.random();

    // Generate 100 random ASCII strings of varying lengths
    for (0..100) |_| {
        const len = random.intRangeAtMost(usize, 0, 500);
        const input = try randomPrintableAscii(allocator, random, len);
        defer allocator.free(input);

        // Must not crash
        _ = try fuzzTokenize(allocator, input);
    }
}

// ── Fuzz Test 2: Invalid UTF-8 sequences ─────────────────────────────────

test "fuzz: invalid UTF-8 byte sequences" {
    const allocator = std.testing.allocator;
    var rng = std.Random.DefaultPrng.init(0xCAFE_BABE);
    const random = rng.random();

    // Generate 100 random byte sequences (some invalid UTF-8)
    for (0..100) |_| {
        const len = random.intRangeAtMost(usize, 0, 300);
        const input = try randomBytes(allocator, random, len);
        defer allocator.free(input);

        // Tokenizer must handle invalid UTF-8 gracefully
        _ = try fuzzTokenize(allocator, input);
    }
}

// ── Fuzz Test 3: Very long identifiers ──────────────────────────────────

test "fuzz: very long identifiers" {
    const allocator = std.testing.allocator;
    var rng = std.Random.DefaultPrng.init(0xF00D_FACE);
    const random = rng.random();

    for (0..50) |_| {
        const len = random.intRangeAtMost(usize, 1000, 10_000);
        const input = try allocator.alloc(u8, len + 1);
        defer allocator.free(input);

        // Start with valid identifier character
        input[0] = 'a';
        for (input[1..]) |*b| {
            const ch = random.intRangeAtMost(u8, 0, 2);
            b.* = switch (ch) {
                0 => 'a' + random.intRangeAtMost(u8, 0, 25),
                1 => '0' + random.intRangeAtMost(u8, 0, 9),
                2 => '_',
                else => unreachable,
            };
        }

        var tokenizer = Tokenizer.init(input);
        const tok = tokenizer.next();
        // Should tokenize as identifier or possibly truncate/reject gracefully
        try std.testing.expect(tok.type == .identifier or tok.type == .invalid or tok.type == .eof);
    }
}

// ── Fuzz Test 4: Unclosed string literals ───────────────────────────────

test "fuzz: unclosed string literals" {
    const allocator = std.testing.allocator;
    var rng = std.Random.DefaultPrng.init(0xBAAD_F00D);
    const random = rng.random();

    for (0..100) |_| {
        const len = random.intRangeAtMost(usize, 1, 200);
        const input = try allocator.alloc(u8, len + 1);
        defer allocator.free(input);

        input[0] = '\''; // Opening quote
        for (input[1..]) |*b| {
            // Fill with random printable ASCII, but NO closing quote
            b.* = random.intRangeAtMost(u8, 32, 126);
            if (b.* == '\'') b.* = '"'; // Replace any accidental quotes
        }

        var tokenizer = Tokenizer.init(input);
        const tok = tokenizer.next();
        // Should be invalid (unterminated string)
        try std.testing.expectEqual(TokenType.invalid, tok.type);
    }
}

// ── Fuzz Test 5: Invalid escape sequences ───────────────────────────────

test "fuzz: string literals with random escape sequences" {
    const allocator = std.testing.allocator;
    var rng = std.Random.DefaultPrng.init(0x1234_5678);
    const random = rng.random();

    for (0..100) |_| {
        var buf: std.ArrayListUnmanaged(u8) = .{};
        defer buf.deinit(allocator);

        try buf.append(allocator, '\''); // Opening quote
        const inner_len = random.intRangeAtMost(usize, 0, 100);
        for (0..inner_len) |_| {
            const ch = random.intRangeAtMost(u8, 32, 126);
            try buf.append(allocator, ch);
        }
        try buf.append(allocator, '\''); // Closing quote

        var tokenizer = Tokenizer.init(buf.items);
        const tok = tokenizer.next();
        // Should tokenize as string_literal (SQL uses '' for escaping)
        try std.testing.expect(tok.type == .string_literal or tok.type == .invalid);
    }
}

// ── Fuzz Test 6: Very long string literals ──────────────────────────────

test "fuzz: very long string literals" {
    const allocator = std.testing.allocator;
    var rng = std.Random.DefaultPrng.init(0x9999_AAAA);
    const random = rng.random();

    for (0..20) |_| {
        const len = random.intRangeAtMost(usize, 5000, 50_000);
        const input = try allocator.alloc(u8, len + 2);
        defer allocator.free(input);

        input[0] = '\'';
        for (input[1 .. len + 1]) |*b| {
            b.* = random.intRangeAtMost(u8, 32, 126);
            if (b.* == '\'') b.* = '"'; // Avoid closing early
        }
        input[len + 1] = '\'';

        var tokenizer = Tokenizer.init(input);
        const tok = tokenizer.next();
        try std.testing.expectEqual(TokenType.string_literal, tok.type);
        try std.testing.expectEqual(@as(u32, @intCast(len + 2)), tok.len);
    }
}

// ── Fuzz Test 7: Number format edge cases ────────────────────────────────

test "fuzz: malformed number literals" {
    const allocator = std.testing.allocator;
    var rng = std.Random.DefaultPrng.init(0xAAAA_BBBB);
    const random = rng.random();

    const cases = [_][]const u8{
        "0x",        // Hex with no digits
        "0xG",       // Hex with invalid char
        "1e",        // Exponent with no digits
        "1e+",       // Exponent with sign but no digits
        "1.2.3",     // Multiple dots
        "..5",       // Double dot
        ".e5",       // Dot-exponent without digits
        "123abc",    // Number followed by identifier
        "0b101",     // Binary (not supported by current tokenizer)
        "0o777",     // Octal (not supported)
    };

    for (cases) |input| {
        var tokenizer = Tokenizer.init(input);
        _ = tokenizer.next(); // Must not crash
    }

    // Random combinations
    for (0..100) |_| {
        var buf: std.ArrayListUnmanaged(u8) = .{};
        defer buf.deinit(allocator);

        // Generate a number-like string with random characters
        const len = random.intRangeAtMost(usize, 1, 20);
        for (0..len) |_| {
            const ch = random.intRangeAtMost(u8, 0, 5);
            try buf.append(allocator, switch (ch) {
                0 => '0' + random.intRangeAtMost(u8, 0, 9),
                1 => '.',
                2 => 'e',
                3 => 'E',
                4 => '+',
                5 => '-',
                else => unreachable,
            });
        }

        var tokenizer = Tokenizer.init(buf.items);
        _ = tokenizer.next(); // Must not crash
    }
}

// ── Fuzz Test 8: Unicode identifiers (non-ASCII) ─────────────────────────

test "fuzz: unicode identifiers" {
    const allocator = std.testing.allocator;
    var rng = std.Random.DefaultPrng.init(0xFACE_1234);
    const random = rng.random();

    // Mix ASCII identifiers with Unicode characters
    for (0..50) |_| {
        var buf: std.ArrayListUnmanaged(u8) = .{};
        defer buf.deinit(allocator);

        // Start with valid ASCII identifier
        try buf.append(allocator, 'x');

        const len = random.intRangeAtMost(usize, 1, 50);
        for (0..len) |_| {
            const choice = random.intRangeAtMost(u8, 0, 3);
            if (choice == 0) {
                // Insert a multi-byte UTF-8 sequence
                try buf.append(allocator, 0xC3); // Start of 2-byte sequence
                try buf.append(allocator, random.intRangeAtMost(u8, 0x80, 0xBF));
            } else {
                // ASCII alphanumeric
                const ch = random.intRangeAtMost(u8, 0, 2);
                try buf.append(allocator, switch (ch) {
                    0 => 'a' + random.intRangeAtMost(u8, 0, 25),
                    1 => '0' + random.intRangeAtMost(u8, 0, 9),
                    2 => '_',
                    else => unreachable,
                });
            }
        }

        // Tokenizer currently expects ASCII identifiers — non-ASCII may be invalid
        var tokenizer = Tokenizer.init(buf.items);
        _ = tokenizer.next(); // Must not crash
    }
}

// ── Fuzz Test 9: Mixed valid/invalid token sequences ────────────────────

test "fuzz: mixed valid and invalid tokens" {
    const allocator = std.testing.allocator;
    var rng = std.Random.DefaultPrng.init(0x5555_6666);
    const random = rng.random();

    for (0..100) |_| {
        var buf: std.ArrayListUnmanaged(u8) = .{};
        defer buf.deinit(allocator);

        const num_tokens = random.intRangeAtMost(usize, 5, 30);
        for (0..num_tokens) |_| {
            const token_type = random.intRangeAtMost(u8, 0, 7);
            switch (token_type) {
                0 => try buf.appendSlice(allocator, randomKeyword(random)),
                1 => try buf.appendSlice(allocator, randomOperator(random)),
                2 => {
                    // Random number
                    const num = random.intRangeAtMost(u32, 0, 999999);
                    const num_str = try std.fmt.allocPrint(allocator, "{d}", .{num});
                    defer allocator.free(num_str);
                    try buf.appendSlice(allocator, num_str);
                },
                3 => try buf.appendSlice(allocator, "'string'"),
                4 => try buf.append(allocator, '('),
                5 => try buf.append(allocator, ')'),
                6 => try buf.append(allocator, ','),
                7 => try buf.append(allocator, ';'),
                else => unreachable,
            }

            // Add random whitespace or no whitespace
            if (random.boolean()) {
                const ws = random.intRangeAtMost(u8, 0, 3);
                try buf.append(allocator, switch (ws) {
                    0 => ' ',
                    1 => '\t',
                    2 => '\n',
                    3 => '\r',
                    else => unreachable,
                });
            }
        }

        _ = try fuzzTokenize(allocator, buf.items);
    }
}

// ── Fuzz Test 10: Operator sequences (valid and invalid) ────────────────

test "fuzz: operator sequences" {
    const allocator = std.testing.allocator;
    var rng = std.Random.DefaultPrng.init(0x7777_8888);
    const random = rng.random();

    // Test sequences of operator characters
    for (0..200) |_| {
        var buf: std.ArrayListUnmanaged(u8) = .{};
        defer buf.deinit(allocator);

        const len = random.intRangeAtMost(usize, 1, 10);
        for (0..len) |_| {
            const ch = random.intRangeAtMost(u8, 0, 14);
            try buf.append(allocator, switch (ch) {
                0 => '=',
                1 => '!',
                2 => '<',
                3 => '>',
                4 => '+',
                5 => '-',
                6 => '*',
                7 => '/',
                8 => '%',
                9 => '&',
                10 => '|',
                11 => '~',
                12 => '@',
                13 => '?',
                14 => '#',
                else => unreachable,
            });
        }

        var tokenizer = Tokenizer.init(buf.items);
        // Tokenize all — some may be invalid, but must not crash
        while (true) {
            const tok = tokenizer.next();
            if (tok.type == .eof) break;
        }
    }
}

// ── Fuzz Test 11: Keyword fuzzing (partial/misspelled) ──────────────────

test "fuzz: partial and misspelled keywords" {
    const allocator = std.testing.allocator;
    var rng = std.Random.DefaultPrng.init(0x3333_4444);
    const random = rng.random();

    const base_keywords = [_][]const u8{
        "SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP",
        "TABLE", "INDEX", "WHERE", "FROM", "JOIN", "ORDER", "GROUP",
    };

    for (0..100) |_| {
        const base = base_keywords[random.intRangeLessThan(usize, 0, base_keywords.len)];
        var buf: std.ArrayListUnmanaged(u8) = .{};
        defer buf.deinit(allocator);

        // Apply mutations
        for (base) |ch| {
            const mutation = random.intRangeAtMost(u8, 0, 10);
            if (mutation == 0) {
                // Skip character (truncate keyword)
                continue;
            } else if (mutation == 1) {
                // Duplicate character
                try buf.append(allocator, ch);
                try buf.append(allocator, ch);
            } else if (mutation == 2) {
                // Insert random character
                try buf.append(allocator, ch);
                try buf.append(allocator, random.intRangeAtMost(u8, 'a', 'z'));
            } else {
                // Keep as-is
                try buf.append(allocator, ch);
            }
        }

        var tokenizer = Tokenizer.init(buf.items);
        const tok = tokenizer.next();
        // Should be either a keyword or identifier, not crash
        try std.testing.expect(tok.type.isKeyword() or tok.type == .identifier or tok.type == .invalid);
    }
}

// ── Fuzz Test 12: Edge cases (empty, whitespace-only, special chars) ────

test "fuzz: edge case inputs" {
    _ = std.testing.allocator;

    const cases = [_][]const u8{
        "",                   // Empty
        " ",                  // Single space
        "   \t\n\r  ",        // Whitespace only
        "\x00",               // Null byte
        "\x00\x00\x00",       // Multiple nulls
        ";;;;;;;",            // Only semicolons
        "((((((",             // Only left parens
        "))))))",             // Only right parens
        ",,,,,,",             // Only commas
        "......",             // Only dots
        "''",                 // Empty string literal
        "\"\"",               // Empty quoted identifier
        "'",                  // Single quote
        "\"",                 // Single double quote
        "--",                 // Comment start only
        "-- comment\n",       // Line comment with newline
        "/*",                 // Unclosed block comment
        "/* unclosed",        // Unclosed block comment with text
        "/* nested /* */",    // Partially nested comment
    };

    for (cases) |input| {
        var tokenizer = Tokenizer.init(input);
        // Tokenize all tokens — must not crash
        while (true) {
            const tok = tokenizer.next();
            if (tok.type == .eof) break;
        }
    }
}

// ── Fuzz Test 13: Deeply nested comments ─────────────────────────────────

test "fuzz: deeply nested block comments" {
    const allocator = std.testing.allocator;
    var rng = std.Random.DefaultPrng.init(0x2222_3333);
    const random = rng.random();

    for (0..50) |_| {
        var buf: std.ArrayListUnmanaged(u8) = .{};
        defer buf.deinit(allocator);

        const depth = random.intRangeAtMost(usize, 1, 20);

        // Open nested comments
        for (0..depth) |_| {
            try buf.appendSlice(allocator, "/* ");
        }

        // Add some content
        try buf.appendSlice(allocator,"content");

        // Close nested comments
        for (0..depth) |_| {
            try buf.appendSlice(allocator, " */");
        }

        var tokenizer = Tokenizer.init(buf.items);
        const tok = tokenizer.next();
        // All comments consumed, next token should be EOF
        try std.testing.expectEqual(TokenType.eof, tok.type);
    }
}

// ── Fuzz Test 14: Random SQL-like statements ─────────────────────────────

test "fuzz: random SQL-like statements" {
    const allocator = std.testing.allocator;
    var rng = std.Random.DefaultPrng.init(0x4444_5555);
    const random = rng.random();

    for (0..100) |_| {
        var buf: std.ArrayListUnmanaged(u8) = .{};
        defer buf.deinit(allocator);

        // Generate a pseudo-SQL statement
        try buf.appendSlice(allocator,randomKeyword(random));
        try buf.append(allocator, ' ');

        const num_parts = random.intRangeAtMost(usize, 3, 15);
        for (0..num_parts) |_| {
            const part_type = random.intRangeAtMost(u8, 0, 5);
            switch (part_type) {
                0 => try buf.appendSlice(allocator, randomKeyword(random)),
                1 => {
                    // Identifier
                    const id_len = random.intRangeAtMost(usize, 1, 10);
                    for (0..id_len) |_| {
                        try buf.append(allocator, 'a' + random.intRangeAtMost(u8, 0, 25));
                    }
                },
                2 => {
                    // Number
                    const num = random.intRangeAtMost(u32, 0, 9999);
                    const num_str = try std.fmt.allocPrint(allocator, "{d}", .{num});
                    defer allocator.free(num_str);
                    try buf.appendSlice(allocator, num_str);
                },
                3 => try buf.appendSlice(allocator, randomOperator(random)),
                4 => try buf.append(allocator, '('),
                5 => try buf.append(allocator, ')'),
                else => unreachable,
            }

            if (random.boolean()) {
                try buf.append(allocator, ' ');
            }
        }
        try buf.append(allocator, ';');

        _ = try fuzzTokenize(allocator, buf.items);
    }
}

// ── Fuzz Test 15: Quoted identifier edge cases ──────────────────────────

test "fuzz: quoted identifier edge cases" {
    const allocator = std.testing.allocator;
    var rng = std.Random.DefaultPrng.init(0x6666_7777);
    const random = rng.random();

    for (0..100) |_| {
        var buf: std.ArrayListUnmanaged(u8) = .{};
        defer buf.deinit(allocator);

        try buf.append(allocator, '"'); // Opening quote
        const len = random.intRangeAtMost(usize, 0, 50);
        for (0..len) |_| {
            const ch = random.intRangeAtMost(u8, 32, 126);
            try buf.append(allocator, ch);
            // Escape internal quotes with ""
            if (ch == '"' and random.boolean()) {
                try buf.append(allocator, '"');
            }
        }

        // Randomly close or leave unclosed
        if (random.boolean()) {
            try buf.append(allocator, '"');
        }

        var tokenizer = Tokenizer.init(buf.items);
        const tok = tokenizer.next();
        // Should be quoted_identifier or invalid (if unclosed)
        try std.testing.expect(tok.type == .quoted_identifier or tok.type == .invalid);
    }
}

// ── Fuzz Test 16: Blob literal fuzzing ──────────────────────────────────

test "fuzz: blob literal edge cases" {
    const allocator = std.testing.allocator;
    var rng = std.Random.DefaultPrng.init(0x8888_9999);
    const random = rng.random();

    for (0..100) |_| {
        var buf: std.ArrayListUnmanaged(u8) = .{};
        defer buf.deinit(allocator);

        // X' or x'
        if (random.boolean()) {
            try buf.append(allocator, 'X');
        } else {
            try buf.append(allocator, 'x');
        }
        try buf.append(allocator, '\'');

        // Generate hex-like content (may be invalid)
        const len = random.intRangeAtMost(usize, 0, 50);
        for (0..len) |_| {
            const hex_valid = random.boolean();
            if (hex_valid) {
                const hex_chars = "0123456789ABCDEFabcdef";
                try buf.append(allocator, hex_chars[random.intRangeLessThan(usize, 0, hex_chars.len)]);
            } else {
                // Invalid hex character
                try buf.append(allocator, 'G' + random.intRangeAtMost(u8, 0, 10));
            }
        }

        // Randomly close or leave unclosed
        if (random.boolean()) {
            try buf.append(allocator, '\'');
        }

        var tokenizer = Tokenizer.init(buf.items);
        const tok = tokenizer.next();
        // Should be blob_literal or invalid
        try std.testing.expect(tok.type == .blob_literal or tok.type == .invalid);
    }
}

// ── Fuzz Test 17: JSON operator fuzzing ─────────────────────────────────

test "fuzz: JSON operator sequences" {
    const allocator = std.testing.allocator;
    var rng = std.Random.DefaultPrng.init(0xBBBB_CCCC);
    const random = rng.random();

    const json_op_chars = [_]u8{ '-', '>', '@', '?', '#', '|', '&' };

    for (0..200) |_| {
        var buf: std.ArrayListUnmanaged(u8) = .{};
        defer buf.deinit(allocator);

        const len = random.intRangeAtMost(usize, 1, 5);
        for (0..len) |_| {
            const ch = json_op_chars[random.intRangeLessThan(usize, 0, json_op_chars.len)];
            try buf.append(allocator, ch);
        }

        var tokenizer = Tokenizer.init(buf.items);
        // Tokenize all — some combinations are valid JSON ops, some invalid
        while (true) {
            const tok = tokenizer.next();
            if (tok.type == .eof) break;
        }
    }
}

// ── Fuzz Test 18: Large token stream ─────────────────────────────────────

test "fuzz: large token stream (1000+ tokens)" {
    const allocator = std.testing.allocator;
    var rng = std.Random.DefaultPrng.init(0xDDDD_EEEE);
    const random = rng.random();

    var buf: std.ArrayListUnmanaged(u8) = .{};
    defer buf.deinit(allocator);

    // Generate 1000 random tokens
    for (0..1000) |_| {
        const token_type = random.intRangeAtMost(u8, 0, 6);
        switch (token_type) {
            0 => try buf.appendSlice(allocator, randomKeyword(random)),
            1 => {
                const num = random.intRangeAtMost(u32, 0, 99999);
                const num_str = try std.fmt.allocPrint(allocator, "{d}", .{num});
                defer allocator.free(num_str);
                try buf.appendSlice(allocator, num_str);
            },
            2 => try buf.appendSlice(allocator, "'str'"),
            3 => try buf.appendSlice(allocator, randomOperator(random)),
            4 => try buf.append(allocator, '('),
            5 => try buf.append(allocator, ')'),
            6 => try buf.append(allocator, ','),
            else => unreachable,
        }
        try buf.append(allocator, ' ');
    }

    const token_count = try fuzzTokenize(allocator, buf.items);
    // Should produce many tokens without crashing
    try std.testing.expect(token_count > 500);
}

// ── Fuzz Test 19: Comment fuzzing (line and block) ──────────────────────

test "fuzz: random comment patterns" {
    const allocator = std.testing.allocator;
    var rng = std.Random.DefaultPrng.init(0xFFFF_0000);
    const random = rng.random();

    for (0..100) |_| {
        var buf: std.ArrayListUnmanaged(u8) = .{};
        defer buf.deinit(allocator);

        const num_comments = random.intRangeAtMost(usize, 1, 10);
        for (0..num_comments) |_| {
            const comment_type = random.boolean();
            if (comment_type) {
                // Line comment
                try buf.appendSlice(allocator, "-- ");
                const len = random.intRangeAtMost(usize, 0, 50);
                for (0..len) |_| {
                    try buf.append(allocator, random.intRangeAtMost(u8, 32, 126));
                }
                try buf.append(allocator, '\n');
            } else {
                // Block comment
                try buf.appendSlice(allocator, "/* ");
                const len = random.intRangeAtMost(usize, 0, 50);
                for (0..len) |_| {
                    try buf.append(allocator, random.intRangeAtMost(u8, 32, 126));
                }
                // Randomly close or leave unclosed
                if (random.boolean()) {
                    try buf.appendSlice(allocator, " */");
                }
            }

            // Add some tokens between comments
            if (random.boolean()) {
                try buf.appendSlice(allocator, " SELECT ");
            }
        }

        _ = try fuzzTokenize(allocator, buf.items);
    }
}

// ── Fuzz Test 20: Stress test with all patterns combined ────────────────

test "fuzz: combined stress test" {
    const allocator = std.testing.allocator;
    var rng = std.Random.DefaultPrng.init(0x1111_2222);
    const random = rng.random();

    // Generate 50 complex inputs combining all edge cases
    for (0..50) |_| {
        var buf: std.ArrayListUnmanaged(u8) = .{};
        defer buf.deinit(allocator);

        const num_parts = random.intRangeAtMost(usize, 10, 50);
        for (0..num_parts) |_| {
            const part_type = random.intRangeAtMost(u8, 0, 11);
            switch (part_type) {
                0 => try buf.appendSlice(allocator, randomKeyword(random)),
                1 => try buf.appendSlice(allocator, randomOperator(random)),
                2 => {
                    // Random number
                    const num = random.intRangeAtMost(u32, 0, 999999);
                    const num_str = try std.fmt.allocPrint(allocator, "{d}", .{num});
                    defer allocator.free(num_str);
                    try buf.appendSlice(allocator, num_str);
                },
                3 => {
                    // Random string literal
                    try buf.append(allocator, '\'');
                    const len = random.intRangeAtMost(usize, 0, 20);
                    for (0..len) |_| {
                        const ch = random.intRangeAtMost(u8, 32, 126);
                        try buf.append(allocator, if (ch == '\'') '"' else ch);
                    }
                    if (random.boolean()) try buf.append(allocator, '\''); // Randomly close
                },
                4 => {
                    // Quoted identifier
                    try buf.append(allocator, '"');
                    const len = random.intRangeAtMost(usize, 0, 20);
                    for (0..len) |_| {
                        try buf.append(allocator, random.intRangeAtMost(u8, 'a', 'z'));
                    }
                    if (random.boolean()) try buf.append(allocator, '"'); // Randomly close
                },
                5 => try buf.append(allocator, '('),
                6 => try buf.append(allocator, ')'),
                7 => try buf.append(allocator, ','),
                8 => try buf.append(allocator, ';'),
                9 => {
                    // Line comment
                    try buf.appendSlice(allocator, "-- comment\n");
                },
                10 => {
                    // Block comment
                    try buf.appendSlice(allocator, "/* comment */");
                },
                11 => {
                    // Random whitespace
                    const ws = random.intRangeAtMost(usize, 1, 5);
                    for (0..ws) |_| try buf.append(allocator, ' ');
                },
                else => unreachable,
            }
        }

        _ = try fuzzTokenize(allocator, buf.items);
    }
}
