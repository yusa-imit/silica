//! Index Entry Encoder/Decoder
//!
//! Provides encode/decode for covering index (INCLUDE) B+Tree leaf entries.
//! Layout: [row_key_len:u16 LE][row_key bytes][TupleHeader:12B][serialized values]
//! (the values blob comes from executor.serializeRow, which embeds its own col_count)
//!
//! This module is storage-agnostic and focuses only on the wire format for
//! index-only scans — it does not interact with the B+Tree, pager, or engine.

const std = @import("std");
const Allocator = std.mem.Allocator;
const executor_mod = @import("executor.zig");
const mvcc_mod = @import("../tx/mvcc.zig");

const Value = executor_mod.Value;
const TupleHeader = mvcc_mod.TupleHeader;
const TUPLE_HEADER_SIZE = mvcc_mod.TUPLE_HEADER_SIZE;

/// Errors returned by index entry operations.
pub const IndexEntryError = error{
    InvalidIndexEntry,
    OutOfMemory,
};

/// Decoded index entry with owned memory.
pub const DecodedIndexEntry = struct {
    row_key: []u8,       // owned, caller frees
    header: TupleHeader,
    values: []Value,     // owned, caller frees (each Value.free + slice free)
};

/// Encode a covering index entry: [row_key_len:u16][row_key][header:12B][values].
/// The values blob is produced by `executor_mod.serializeRow`, which already
/// prefixes its own col_count — no separate col_count field is added here.
/// Returns an allocated byte buffer that the caller owns.
pub fn encodeIndexEntry(allocator: Allocator, row_key: []const u8, header: TupleHeader, included_vals: []const Value) IndexEntryError![]u8 {
    if (row_key.len > std.math.maxInt(u16)) return IndexEntryError.InvalidIndexEntry;

    const values_bytes = executor_mod.serializeRow(allocator, included_vals) catch return IndexEntryError.OutOfMemory;
    defer allocator.free(values_bytes);

    const total_len = 2 + row_key.len + TUPLE_HEADER_SIZE + values_bytes.len;
    const buf = try allocator.alloc(u8, total_len);
    errdefer allocator.free(buf);

    var pos: usize = 0;
    std.mem.writeInt(u16, buf[pos..][0..2], @intCast(row_key.len), .little);
    pos += 2;

    @memcpy(buf[pos..][0..row_key.len], row_key);
    pos += row_key.len;

    var header_buf: [TUPLE_HEADER_SIZE]u8 = undefined;
    header.serialize(&header_buf);
    @memcpy(buf[pos..][0..TUPLE_HEADER_SIZE], &header_buf);
    pos += TUPLE_HEADER_SIZE;

    @memcpy(buf[pos..][0..values_bytes.len], values_bytes);
    pos += values_bytes.len;

    std.debug.assert(pos == total_len);
    return buf;
}

/// Decode a covering index entry from bytes.
/// Returns owned DecodedIndexEntry; caller must free row_key, each value in values array, and the values array itself.
pub fn decodeIndexEntry(allocator: Allocator, bytes: []const u8) IndexEntryError!DecodedIndexEntry {
    if (bytes.len < 2) return IndexEntryError.InvalidIndexEntry;
    const row_key_len = std.mem.readInt(u16, bytes[0..2], .little);

    const row_key_end = 2 + @as(usize, row_key_len);
    if (bytes.len < row_key_end) return IndexEntryError.InvalidIndexEntry;

    const row_key = allocator.dupe(u8, bytes[2..row_key_end]) catch return IndexEntryError.OutOfMemory;
    errdefer allocator.free(row_key);

    const header_end = row_key_end + TUPLE_HEADER_SIZE;
    if (bytes.len < header_end) return IndexEntryError.InvalidIndexEntry;

    const header = TupleHeader.deserialize(bytes[row_key_end..header_end][0..TUPLE_HEADER_SIZE]);

    const values = executor_mod.deserializeRow(allocator, bytes[header_end..]) catch |err| {
        return switch (err) {
            error.OutOfMemory => IndexEntryError.OutOfMemory,
            else => IndexEntryError.InvalidIndexEntry,
        };
    };

    return DecodedIndexEntry{
        .row_key = row_key,
        .header = header,
        .values = values,
    };
}

// ── Tests ────────────────────────────────────────────────────────────────────

const testing = std.testing;

test "encode/decode round-trip with mixed value types" {
    const allocator = testing.allocator;

    // Create a realistic 8-byte big-endian row_key (standard format in this codebase)
    var row_key_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &row_key_buf, 12345, .big);
    const row_key = &row_key_buf;

    // Create a TupleHeader
    const header = TupleHeader{
        .xmin = 100,
        .xmax = mvcc_mod.INVALID_XID,
        .cid = 0,
        .flags = .{},
    };

    // Create included values: mix of types (integer, text, null, real)
    const included_vals = [_]Value{
        .{ .integer = 42 },
        .{ .text = "hello world" },
        .null_value,
        .{ .real = 3.14159 },
    };

    // Encode
    const encoded = try encodeIndexEntry(allocator, row_key, header, &included_vals);
    defer allocator.free(encoded);

    // Decode
    const decoded = try decodeIndexEntry(allocator, encoded);
    defer {
        allocator.free(decoded.row_key);
        for (decoded.values) |v| v.free(allocator);
        allocator.free(decoded.values);
    }

    // Verify row_key matches
    try testing.expectEqualSlices(u8, row_key, decoded.row_key);

    // Verify header matches exactly
    try testing.expectEqual(header.xmin, decoded.header.xmin);
    try testing.expectEqual(header.xmax, decoded.header.xmax);
    try testing.expectEqual(header.cid, decoded.header.cid);
    try testing.expectEqual(@as(u8, @bitCast(header.flags)), @as(u8, @bitCast(decoded.header.flags)));

    // Verify values match
    try testing.expectEqual(@as(usize, 4), decoded.values.len);
    try testing.expectEqual(@as(i64, 42), decoded.values[0].integer);
    try testing.expectEqualStrings("hello world", decoded.values[1].text);
    try testing.expectEqual(Value.null_value, decoded.values[2]);
    try testing.expectEqual(@as(f64, 3.14159), decoded.values[3].real);
}

test "encode/decode round-trip with empty included values" {
    const allocator = testing.allocator;

    // Create a row_key
    var row_key_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &row_key_buf, 999, .big);
    const row_key = &row_key_buf;

    const header = TupleHeader{
        .xmin = 50,
        .xmax = mvcc_mod.INVALID_XID,
        .cid = 1,
        .flags = .{},
    };

    const included_vals = [_]Value{};

    // Encode
    const encoded = try encodeIndexEntry(allocator, row_key, header, &included_vals);
    defer allocator.free(encoded);

    // Decode
    const decoded = try decodeIndexEntry(allocator, encoded);
    defer {
        allocator.free(decoded.row_key);
        allocator.free(decoded.values);
    }

    // Verify row_key matches
    try testing.expectEqualSlices(u8, row_key, decoded.row_key);

    // Verify header matches
    try testing.expectEqual(header.xmin, decoded.header.xmin);
    try testing.expectEqual(header.xmax, decoded.header.xmax);

    // Verify values array is empty
    try testing.expectEqual(@as(usize, 0), decoded.values.len);
}

test "encode/decode with standard 8-byte big-endian row_key format" {
    const allocator = testing.allocator;

    // Use standard 8-byte big-endian u64 row_key
    var row_key_buf: [8]u8 = undefined;
    const expected_row_id: u64 = 0xDEADBEEFCAFEBABE;
    std.mem.writeInt(u64, &row_key_buf, expected_row_id, .big);
    const row_key = &row_key_buf;

    const header = TupleHeader{
        .xmin = 200,
        .xmax = mvcc_mod.INVALID_XID,
        .cid = 5,
        .flags = .{},
    };

    const included_vals = [_]Value{
        .{ .integer = 999 },
    };

    const encoded = try encodeIndexEntry(allocator, row_key, header, &included_vals);
    defer allocator.free(encoded);

    const decoded = try decodeIndexEntry(allocator, encoded);
    defer {
        allocator.free(decoded.row_key);
        for (decoded.values) |v| v.free(allocator);
        allocator.free(decoded.values);
    }

    // Verify row_key exactly matches the 8-byte big-endian encoding
    try testing.expectEqualSlices(u8, row_key, decoded.row_key);
    try testing.expectEqual(@as(usize, 8), decoded.row_key.len);

    // Verify we can read it back as big-endian u64
    const recovered_row_id = std.mem.readInt(u64, decoded.row_key[0..8], .big);
    try testing.expectEqual(expected_row_id, recovered_row_id);
}

test "encode/decode tuple header with xmax != INVALID_XID (dead tuple)" {
    const allocator = testing.allocator;

    var row_key_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &row_key_buf, 777, .big);
    const row_key = &row_key_buf;

    // Header with xmax set (simulating a deleted/updated tuple)
    const header = TupleHeader{
        .xmin = 100,
        .xmax = 150, // deleted by transaction 150
        .cid = 2,
        .flags = .{ .xmax_committed = true },
    };

    const included_vals = [_]Value{
        .{ .text = "deleted row" },
    };

    const encoded = try encodeIndexEntry(allocator, row_key, header, &included_vals);
    defer allocator.free(encoded);

    const decoded = try decodeIndexEntry(allocator, encoded);
    defer {
        allocator.free(decoded.row_key);
        for (decoded.values) |v| v.free(allocator);
        allocator.free(decoded.values);
    }

    // Verify xmax is preserved correctly
    try testing.expectEqual(@as(u32, 150), decoded.header.xmax);
    try testing.expectEqual(@as(u32, 100), decoded.header.xmin);

    // Verify flags are preserved
    try testing.expectEqual(true, decoded.header.flags.xmax_committed);
}

test "decode truncated buffer before row_key_len" {
    const allocator = testing.allocator;

    // Buffer too short to contain even row_key_len (u16 = 2 bytes)
    const truncated = [_]u8{};
    const result = decodeIndexEntry(allocator, &truncated);

    try testing.expectError(error.InvalidIndexEntry, result);
}

test "decode truncated buffer in row_key bytes" {
    const allocator = testing.allocator;

    // Buffer declares row_key_len=10 but only provides 5 bytes
    var buf: [7]u8 = undefined;
    std.mem.writeInt(u16, buf[0..2], 10, .little); // says row_key is 10 bytes
    @memcpy(buf[2..7], "short"); // but only 5 bytes follow
    const result = decodeIndexEntry(allocator, &buf);

    try testing.expectError(error.InvalidIndexEntry, result);
}

test "decode truncated buffer before TupleHeader" {
    const allocator = testing.allocator;

    // Buffer has row_key_len + row_key, but truncated before header
    var buf: [10]u8 = undefined;
    std.mem.writeInt(u16, buf[0..2], 4, .little); // row_key_len=4
    @memcpy(buf[2..6], "test"); // 4-byte row_key
    // buf[6..10] = incomplete, doesn't have full 12-byte TupleHeader

    const result = decodeIndexEntry(allocator, &buf);

    try testing.expectError(error.InvalidIndexEntry, result);
}

test "decode truncated buffer before col_count" {
    const allocator = testing.allocator;

    // Buffer has row_key + header, but truncated before col_count
    const row_key_len: u16 = 8;
    const total_header_start = 2 + row_key_len; // offset to TupleHeader
    const col_count_start = total_header_start + TUPLE_HEADER_SIZE;

    var buf = try allocator.alloc(u8, col_count_start + 1); // 1 byte short
    defer allocator.free(buf);

    std.mem.writeInt(u16, buf[0..2], row_key_len, .little);
    @memset(buf[2..][0..row_key_len], 0xAA);

    // Write TupleHeader
    var header_buf: [TUPLE_HEADER_SIZE]u8 = undefined;
    const header = TupleHeader.forInsert(100, 0);
    header.serialize(&header_buf);
    @memcpy(buf[total_header_start..][0..TUPLE_HEADER_SIZE], &header_buf);

    // buf is 1 byte short of col_count — should error
    const result = decodeIndexEntry(allocator, buf);
    try testing.expectError(error.InvalidIndexEntry, result);
}

test "encode/decode with various value types including nested array" {
    const allocator = testing.allocator;

    var row_key_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &row_key_buf, 111, .big);
    const row_key = &row_key_buf;

    const header = TupleHeader{
        .xmin = 1000,
        .xmax = mvcc_mod.INVALID_XID,
        .cid = 3,
        .flags = .{},
    };

    // Create an array of integers
    const arr_elems = try allocator.alloc(Value, 3);
    arr_elems[0] = .{ .integer = 10 };
    arr_elems[1] = .{ .integer = 20 };
    arr_elems[2] = .{ .integer = 30 };

    const included_vals = try allocator.alloc(Value, 3);
    included_vals[0] = .{ .integer = 5 };
    included_vals[1] = .{ .array = arr_elems };
    included_vals[2] = .{ .text = try allocator.dupe(u8, "after_array") };

    defer {
        for (included_vals) |v| v.free(allocator);
        allocator.free(included_vals);
    }

    const encoded = try encodeIndexEntry(allocator, row_key, header, included_vals);
    defer allocator.free(encoded);

    const decoded = try decodeIndexEntry(allocator, encoded);
    defer {
        allocator.free(decoded.row_key);
        for (decoded.values) |v| v.free(allocator);
        allocator.free(decoded.values);
    }

    // Verify structure
    try testing.expectEqual(@as(usize, 3), decoded.values.len);
    try testing.expectEqual(@as(i64, 5), decoded.values[0].integer);
    try testing.expectEqual(@as(usize, 3), decoded.values[1].array.len);
    try testing.expectEqual(@as(i64, 10), decoded.values[1].array[0].integer);
    try testing.expectEqual(@as(i64, 20), decoded.values[1].array[1].integer);
    try testing.expectEqual(@as(i64, 30), decoded.values[1].array[2].integer);
    try testing.expectEqualStrings("after_array", decoded.values[2].text);
}

test "encode/decode with variable-length row_key (not 8 bytes)" {
    const allocator = testing.allocator;

    // Composite key: 12 bytes (e.g., partition_id + row_id)
    const row_key = "composite_pk";
    const header = TupleHeader.forInsert(50, 0);

    const included_vals = [_]Value{
        .{ .text = "data" },
    };

    const encoded = try encodeIndexEntry(allocator, row_key, header, &included_vals);
    defer allocator.free(encoded);

    const decoded = try decodeIndexEntry(allocator, encoded);
    defer {
        allocator.free(decoded.row_key);
        for (decoded.values) |v| v.free(allocator);
        allocator.free(decoded.values);
    }

    // Verify row_key matches exactly (including length)
    try testing.expectEqualSlices(u8, row_key, decoded.row_key);
    try testing.expectEqual(@as(usize, 12), decoded.row_key.len);
}

test "encode/decode memory safety: multiple round trips" {
    const allocator = testing.allocator;

    var row_key_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &row_key_buf, 555, .big);
    const row_key = &row_key_buf;

    const header = TupleHeader.forInsert(777, 0);
    const included_vals = [_]Value{
        .{ .integer = 1 },
        .{ .text = "test" },
    };

    // First round-trip
    const encoded1 = try encodeIndexEntry(allocator, row_key, header, &included_vals);
    defer allocator.free(encoded1);

    const decoded1 = try decodeIndexEntry(allocator, encoded1);
    defer {
        allocator.free(decoded1.row_key);
        for (decoded1.values) |v| v.free(allocator);
        allocator.free(decoded1.values);
    }

    // Re-encode the decoded entry
    const encoded2 = try encodeIndexEntry(allocator, decoded1.row_key, decoded1.header, decoded1.values);
    defer allocator.free(encoded2);

    const decoded2 = try decodeIndexEntry(allocator, encoded2);
    defer {
        allocator.free(decoded2.row_key);
        for (decoded2.values) |v| v.free(allocator);
        allocator.free(decoded2.values);
    }

    // Verify idempotence: second round-trip should match first
    try testing.expectEqualSlices(u8, decoded1.row_key, decoded2.row_key);
    try testing.expectEqual(decoded1.header.xmin, decoded2.header.xmin);
    try testing.expectEqual(decoded1.values.len, decoded2.values.len);
}
