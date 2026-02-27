//! Variable-length integer encoding/decoding.
//!
//! Encodes unsigned 64-bit integers using 1-9 bytes, where smaller values
//! use fewer bytes. Each byte stores 7 bits of data; the MSB indicates
//! whether more bytes follow (1 = more, 0 = last byte).
//!
//! This is the standard LEB128 unsigned encoding, commonly used in
//! database page formats to minimize storage overhead for small integers.

const std = @import("std");

pub const max_encoded_len = 10; // u64 worst case: ceil(64/7) = 10 bytes

pub const Error = error{
    Overflow,
    EndOfBuffer,
};

/// Encode a u64 value into the provided buffer.
/// Returns the number of bytes written (1-10).
pub fn encode(value: u64, buf: []u8) Error!usize {
    if (buf.len == 0) return Error.EndOfBuffer;

    var v = value;
    var i: usize = 0;

    while (true) {
        if (i >= buf.len) return Error.EndOfBuffer;
        const byte: u8 = @truncate(v & 0x7F);
        v >>= 7;
        if (v == 0) {
            buf[i] = byte;
            return i + 1;
        }
        buf[i] = byte | 0x80;
        i += 1;
    }
}

/// Decode a u64 value from the provided buffer.
/// Returns the decoded value and the number of bytes consumed.
pub fn decode(buf: []const u8) Error!struct { value: u64, bytes_read: usize } {
    if (buf.len == 0) return Error.EndOfBuffer;

    var result: u64 = 0;
    var shift: u6 = 0;

    for (buf, 0..) |byte, i| {
        if (i >= max_encoded_len) return Error.Overflow;

        const payload: u64 = byte & 0x7F;

        // Check for overflow: if shift is 63, payload can only be 0 or 1
        if (shift >= 64) return Error.Overflow;
        if (shift == 63 and payload > 1) return Error.Overflow;

        result |= payload << shift;
        shift = @intCast(@min(@as(u8, shift) + 7, 63));

        if (byte & 0x80 == 0) {
            return .{ .value = result, .bytes_read = i + 1 };
        }
    }

    return Error.EndOfBuffer;
}

/// Returns the number of bytes needed to encode the given value.
pub fn encodedLen(value: u64) usize {
    if (value == 0) return 1;
    // Number of bits needed, divided by 7, rounded up
    const bits = 64 - @clz(value);
    return (bits + 6) / 7;
}

// ── Tests ──────────────────────────────────────────────────────────────

test "encode/decode zero" {
    var buf: [max_encoded_len]u8 = undefined;
    const n = try encode(0, &buf);
    try std.testing.expectEqual(@as(usize, 1), n);
    try std.testing.expectEqual(@as(u8, 0x00), buf[0]);

    const result = try decode(buf[0..n]);
    try std.testing.expectEqual(@as(u64, 0), result.value);
    try std.testing.expectEqual(@as(usize, 1), result.bytes_read);
}

test "encode/decode small values" {
    var buf: [max_encoded_len]u8 = undefined;

    // Single byte values (0-127)
    for (0..128) |v| {
        const value: u64 = @intCast(v);
        const n = try encode(value, &buf);
        try std.testing.expectEqual(@as(usize, 1), n);
        const result = try decode(buf[0..n]);
        try std.testing.expectEqual(value, result.value);
        try std.testing.expectEqual(@as(usize, 1), result.bytes_read);
    }
}

test "encode/decode 128 (two bytes)" {
    var buf: [max_encoded_len]u8 = undefined;
    const n = try encode(128, &buf);
    try std.testing.expectEqual(@as(usize, 2), n);
    try std.testing.expectEqual(@as(u8, 0x80), buf[0]); // 0 with continuation
    try std.testing.expectEqual(@as(u8, 0x01), buf[1]); // 1

    const result = try decode(buf[0..n]);
    try std.testing.expectEqual(@as(u64, 128), result.value);
}

test "encode/decode 300" {
    var buf: [max_encoded_len]u8 = undefined;
    const n = try encode(300, &buf);
    try std.testing.expectEqual(@as(usize, 2), n);

    const result = try decode(buf[0..n]);
    try std.testing.expectEqual(@as(u64, 300), result.value);
}

test "encode/decode u32 max" {
    var buf: [max_encoded_len]u8 = undefined;
    const value: u64 = std.math.maxInt(u32);
    const n = try encode(value, &buf);
    const result = try decode(buf[0..n]);
    try std.testing.expectEqual(value, result.value);
}

test "encode/decode u64 max" {
    var buf: [max_encoded_len]u8 = undefined;
    const value: u64 = std.math.maxInt(u64);
    const n = try encode(value, &buf);
    const result = try decode(buf[0..n]);
    try std.testing.expectEqual(value, result.value);
}

test "encode/decode powers of two" {
    var buf: [max_encoded_len]u8 = undefined;
    var value: u64 = 1;
    for (0..63) |_| {
        const n = try encode(value, &buf);
        const result = try decode(buf[0..n]);
        try std.testing.expectEqual(value, result.value);
        value <<= 1;
    }
}

test "encodedLen correctness" {
    try std.testing.expectEqual(@as(usize, 1), encodedLen(0));
    try std.testing.expectEqual(@as(usize, 1), encodedLen(127));
    try std.testing.expectEqual(@as(usize, 2), encodedLen(128));
    try std.testing.expectEqual(@as(usize, 2), encodedLen(16383));
    try std.testing.expectEqual(@as(usize, 3), encodedLen(16384));
    try std.testing.expectEqual(@as(usize, 5), encodedLen(std.math.maxInt(u32)));
    try std.testing.expectEqual(@as(usize, 10), encodedLen(std.math.maxInt(u64)));
}

test "encodedLen matches actual encoded length" {
    var buf: [max_encoded_len]u8 = undefined;
    const test_values = [_]u64{ 0, 1, 127, 128, 255, 256, 16383, 16384, 1 << 20, 1 << 32, 1 << 48, std.math.maxInt(u64) };
    for (test_values) |value| {
        const actual = try encode(value, &buf);
        try std.testing.expectEqual(actual, encodedLen(value));
    }
}

test "decode empty buffer returns EndOfBuffer" {
    const result = decode("");
    try std.testing.expectError(Error.EndOfBuffer, result);
}

test "encode empty buffer returns EndOfBuffer" {
    var buf: [0]u8 = undefined;
    const result = encode(42, &buf);
    try std.testing.expectError(Error.EndOfBuffer, result);
}

test "encode buffer too small returns EndOfBuffer" {
    var buf: [1]u8 = undefined;
    const result = encode(128, &buf); // needs 2 bytes
    try std.testing.expectError(Error.EndOfBuffer, result);
}

test "decode truncated varint returns EndOfBuffer" {
    // A byte with continuation bit set but no following byte
    const buf = [_]u8{0x80};
    const result = decode(&buf);
    try std.testing.expectError(Error.EndOfBuffer, result);
}

test "roundtrip boundary values" {
    var buf: [max_encoded_len]u8 = undefined;
    const boundaries = [_]u64{
        0,
        1,
        0x7F,       // 127 — max 1-byte
        0x80,       // 128 — min 2-byte
        0x3FFF,     // 16383 — max 2-byte
        0x4000,     // 16384 — min 3-byte
        0x1FFFFF,   // max 3-byte
        0x200000,   // min 4-byte
        0xFFFFFFF,  // max 4-byte
        0x10000000, // min 5-byte
        std.math.maxInt(u32),
        std.math.maxInt(u64),
    };
    for (boundaries) |value| {
        const n = try encode(value, &buf);
        const result = try decode(buf[0..n]);
        try std.testing.expectEqual(value, result.value);
        try std.testing.expectEqual(n, result.bytes_read);
    }
}
