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
