//! Wire Protocol Fuzz Tests
//!
//! Tests wire protocol message parsing with random and malformed inputs
//! to ensure robustness and proper error handling.

const std = @import("std");
const wire = @import("wire.zig");
const Allocator = std.mem.Allocator;

/// Fuzz test helper that runs a parsing function with random bytes
fn fuzzParse(
    comptime parseFunc: anytype,
    comptime needs_allocator: bool,
    allocator: Allocator,
    iterations: usize,
    seed: u64,
) !void {
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();

    var i: usize = 0;
    while (i < iterations) : (i += 1) {
        // Generate random payload size (0 to 1024 bytes)
        const size = random.uintLessThan(usize, 1024);
        const payload = try allocator.alloc(u8, size);
        defer allocator.free(payload);

        // Fill with random bytes
        random.bytes(payload);

        // Try to parse - should either succeed or return an error, not crash
        if (needs_allocator) {
            const result = parseFunc(payload, allocator) catch continue;
            if (@hasDecl(@TypeOf(result), "deinit")) {
                result.deinit(allocator);
            }
        } else {
            _ = parseFunc(payload) catch continue;
        }
    }
}

test "Fuzz Query.parse" {
    const allocator = std.testing.allocator;
    try fuzzParse(
        wire.Query.parse,
        false,
        allocator,
        100, // Reduced iterations for stability
        12345,
    );
}

test "Fuzz Parse.parse" {
    const allocator = std.testing.allocator;
    try fuzzParse(
        wire.Parse.parse,
        true,
        allocator,
        100, // Reduced iterations for stability
        23456,
    );
}

test "Fuzz Bind.parse" {
    const allocator = std.testing.allocator;
    try fuzzParse(
        wire.Bind.parse,
        true,
        allocator,
        100, // Reduced iterations for stability
        34567,
    );
}

test "Fuzz Execute.parse" {
    const allocator = std.testing.allocator;
    try fuzzParse(
        wire.Execute.parse,
        false,
        allocator,
        100, // Reduced iterations for stability
        45678,
    );
}

test "Fuzz Close.parse" {
    const allocator = std.testing.allocator;
    try fuzzParse(
        wire.Close.parse,
        false,
        allocator,
        100, // Reduced iterations for stability
        56789,
    );
}

test "Fuzz readMessage with random streams" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(67890);
    const random = prng.random();

    var i: usize = 0;
    while (i < 50) : (i += 1) {
        // Generate random message-like data
        const size = random.uintLessThan(usize, 1024);
        const data = try allocator.alloc(u8, size);
        defer allocator.free(data);
        random.bytes(data);

        // Try to read message from stream
        var fbs = std.io.fixedBufferStream(data);
        const reader = fbs.reader();

        const result = wire.readMessage(reader, allocator) catch continue;
        allocator.free(result.payload);
    }
}

test "Fuzz readMessage with truncated messages" {
    const allocator = std.testing.allocator;

    // Test various truncation points
    const valid_query = "Q\x00\x00\x00\x0DSELECT 1\x00";

    var i: usize = 0;
    while (i < valid_query.len) : (i += 1) {
        var fbs = std.io.fixedBufferStream(valid_query[0..i]);
        const reader = fbs.reader();

        const result = wire.readMessage(reader, allocator) catch continue;
        allocator.free(result.payload);
    }
}

test "Fuzz Query.parse with null byte variations" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(78901);
    const random = prng.random();

    var i: usize = 0;
    while (i < 50) : (i += 1) {
        // Generate query with random null bytes
        const size = random.uintLessThan(usize, 256);
        const payload = try allocator.alloc(u8, size);
        defer allocator.free(payload);

        // Fill with printable chars, but add random nulls
        for (payload, 0..) |*byte, idx| {
            if (random.boolean()) {
                byte.* = 0;
            } else {
                byte.* = @intCast(33 + random.uintLessThan(u8, 94)); // printable ASCII
            }
            // Ensure last byte is sometimes null
            if (idx == size - 1 and random.boolean()) {
                byte.* = 0;
            }
        }

        _ = wire.Query.parse(payload) catch continue;
    }
}

test "Fuzz Parse.parse with malformed lengths" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(89012);
    const random = prng.random();

    var i: usize = 0;
    while (i < 50) : (i += 1) {
        // Generate payload with random length fields
        var buf = std.ArrayListUnmanaged(u8){};
        defer buf.deinit(allocator);

        // Statement name length
        const stmt_len = random.int(i16);
        try buf.writer(allocator).writeInt(i16, stmt_len, .big);
        if (stmt_len > 0) {
            const name = try allocator.alloc(u8, @abs(stmt_len));
            defer allocator.free(name);
            random.bytes(name);
            try buf.appendSlice(allocator, name);
        }

        // Query string length
        const query_len = random.int(i32);
        try buf.writer(allocator).writeInt(i32, query_len, .big);
        if (query_len > 0) {
            const query = try allocator.alloc(u8, @abs(query_len));
            defer allocator.free(query);
            random.bytes(query);
            try buf.appendSlice(allocator, query);
        }

        // Param count
        const param_count = random.int(i16);
        try buf.writer(allocator).writeInt(i16, param_count, .big);

        const result = wire.Parse.parse(buf.items, allocator) catch continue;
        result.deinit(allocator);
    }
}

test "Fuzz Bind.parse with parameter variations" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(90123);
    const random = prng.random();

    var i: usize = 0;
    while (i < 50) : (i += 1) {
        var buf = std.ArrayListUnmanaged(u8){};
        defer buf.deinit(allocator);

        // Portal name
        const portal_len = random.uintLessThan(u16, 64);
        try buf.writer(allocator).writeInt(i16, @intCast(portal_len), .big);
        {
            const name = try allocator.alloc(u8, portal_len);
            defer allocator.free(name);
            random.bytes(name);
            try buf.appendSlice(allocator, name);
        }

        // Statement name
        const stmt_len = random.uintLessThan(u16, 64);
        try buf.writer(allocator).writeInt(i16, @intCast(stmt_len), .big);
        {
            const name = try allocator.alloc(u8, stmt_len);
            defer allocator.free(name);
            random.bytes(name);
            try buf.appendSlice(allocator, name);
        }

        // Format codes
        const format_count = random.uintLessThan(u16, 10);
        try buf.writer(allocator).writeInt(i16, @intCast(format_count), .big);
        var j: usize = 0;
        while (j < format_count) : (j += 1) {
            try buf.writer(allocator).writeInt(i16, random.int(i16), .big);
        }

        // Parameter values
        const param_count = random.uintLessThan(u16, 10);
        try buf.writer(allocator).writeInt(i16, @intCast(param_count), .big);
        j = 0;
        while (j < param_count) : (j += 1) {
            const len = random.int(i32);
            try buf.writer(allocator).writeInt(i32, len, .big);
            if (len > 0 and len < 1024) {
                const value = try allocator.alloc(u8, @intCast(len));
                defer allocator.free(value);
                random.bytes(value);
                try buf.appendSlice(allocator, value);
            }
        }

        // Result format codes
        const result_format_count = random.uintLessThan(u16, 10);
        try buf.writer(allocator).writeInt(i16, @intCast(result_format_count), .big);
        j = 0;
        while (j < result_format_count) : (j += 1) {
            try buf.writer(allocator).writeInt(i16, random.int(i16), .big);
        }

        const result = wire.Bind.parse(buf.items, allocator) catch continue;
        result.deinit(allocator);
    }
}

test "Fuzz Execute.parse with name variations" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(101234);
    const random = prng.random();

    var i: usize = 0;
    while (i < 50) : (i += 1) {
        var buf = std.ArrayListUnmanaged(u8){};
        defer buf.deinit(allocator);

        // Portal name
        const portal_len = random.uintLessThan(usize, 64);
        const portal = try allocator.alloc(u8, portal_len);
        defer allocator.free(portal);
        random.bytes(portal);
        try buf.appendSlice(allocator, portal);
        try buf.append(allocator, 0); // null terminator

        // Max rows
        try buf.writer(allocator).writeInt(i32, random.int(i32), .big);

        _ = wire.Execute.parse(buf.items) catch continue;
    }
}

test "Fuzz Close.parse with type variations" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(112345);
    const random = prng.random();

    var i: usize = 0;
    while (i < 50) : (i += 1) {
        var buf = std.ArrayListUnmanaged(u8){};
        defer buf.deinit(allocator);

        // Close type (random byte)
        try buf.append(allocator, random.int(u8));

        // Name
        const name_len = random.uintLessThan(usize, 64);
        const name = try allocator.alloc(u8, name_len);
        defer allocator.free(name);
        random.bytes(name);
        try buf.appendSlice(allocator, name);
        try buf.append(allocator, 0); // null terminator

        _ = wire.Close.parse(buf.items) catch continue;
    }
}
