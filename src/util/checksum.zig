//! CRC32C checksum utilities for page integrity verification.
//!
//! Uses the Castagnoli polynomial (iSCSI) which has better error detection
//! properties than CRC32 and is hardware-accelerated on modern CPUs.

const std = @import("std");

const Crc32c = std.hash.crc.Crc32Iscsi;

/// Compute CRC32C checksum over the given data.
pub fn crc32c(data: []const u8) u32 {
    return Crc32c.hash(data);
}

/// Compute CRC32C checksum incrementally, starting from a previous value.
pub fn crc32cUpdate(prev: u32, data: []const u8) u32 {
    var h = Crc32c.init();
    // Reverse the finalization of the previous value to get internal state
    h.crc = prev ^ 0xFFFFFFFF;
    h.update(data);
    return h.final();
}

/// Verify that data matches an expected CRC32C checksum.
pub fn verify(data: []const u8, expected: u32) bool {
    return crc32c(data) == expected;
}

// ── Tests ──────────────────────────────────────────────────────────────

test "crc32c empty input" {
    try std.testing.expectEqual(@as(u32, 0), crc32c(""));
}

test "crc32c known values" {
    // CRC32C of "123456789" is 0xE3069283
    try std.testing.expectEqual(@as(u32, 0xE3069283), crc32c("123456789"));
}

test "crc32c single byte" {
    const result = crc32c("a");
    try std.testing.expect(result != 0);
}

test "crc32c different inputs produce different checksums" {
    const a = crc32c("hello");
    const b = crc32c("world");
    try std.testing.expect(a != b);
}

test "crc32c verify success" {
    const data = "silica database engine";
    const expected = crc32c(data);
    try std.testing.expect(verify(data, expected));
}

test "crc32c verify failure" {
    try std.testing.expect(!verify("data", 0xDEADBEEF));
}

test "crc32c incremental matches single-shot" {
    const full_data = "hello world";
    const full = crc32c(full_data);

    const part1 = "hello ";
    const part2 = "world";
    const incremental = crc32cUpdate(crc32c(part1), part2);

    try std.testing.expectEqual(full, incremental);
}

test "crc32c page-sized buffer" {
    var buf: [4096]u8 = undefined;
    @memset(&buf, 0xAB);
    const result = crc32c(&buf);
    try std.testing.expect(result != 0);
    try std.testing.expect(verify(&buf, result));
}

test "crc32c three-part incremental" {
    const full_data = "hello world test";
    const full = crc32c(full_data);

    const part1 = "hello ";
    const part2 = "world ";
    const part3 = "test";
    var incremental = crc32c(part1);
    incremental = crc32cUpdate(incremental, part2);
    incremental = crc32cUpdate(incremental, part3);

    try std.testing.expectEqual(full, incremental);
}

test "crc32c single bit change detection" {
    var buf: [64]u8 = undefined;
    @memset(&buf, 0x00);
    const original_checksum = crc32c(&buf);

    // Flip each bit in byte 0 and verify checksum changes
    for (0..8) |bit| {
        buf[0] ^= @as(u8, 1) << @intCast(bit);
        const modified_checksum = crc32c(&buf);
        try std.testing.expect(modified_checksum != original_checksum);
        // Reset the bit
        buf[0] ^= @as(u8, 1) << @intCast(bit);
    }
}

test "crc32c all 0xFF buffer" {
    var buf: [4096]u8 = undefined;
    @memset(&buf, 0xFF);
    const result = crc32c(&buf);
    try std.testing.expect(result != 0);
    try std.testing.expect(verify(&buf, result));
}

test "crc32c empty update preserves value" {
    const initial = crc32c("hello");
    const after_empty_update = crc32cUpdate(initial, "");
    try std.testing.expectEqual(initial, after_empty_update);
}
