//! Statistics — table and column statistics for the query optimizer.
//!
//! Statistics are stored in the catalog with a `stats:` key prefix.
//! Key format: `stats:<table_name>:<column_name>` for column stats
//!             `stats:<table_name>` for table stats
//!
//! Serialization format for table statistics:
//!   [row_count: u64][last_analyze_time: i64]
//!
//! Serialization format for column statistics:
//!   [distinct_count: u64][null_fraction: f64][avg_width: f64]
//!   [correlation: f64][mcv_count: u16]
//!   for each MCV:
//!     [value_len: u16][value_bytes...][frequency: f64]
//!   [histogram_bucket_count: u16]
//!   for each bucket:
//!     [lower_len: u16][lower_bytes...][upper_len: u16][upper_bytes...][count: u64]

const std = @import("std");
const Allocator = std.mem.Allocator;

// ── Statistics Structures ───────────────────────────────────────────────

/// Table-level statistics.
pub const TableStats = struct {
    /// Total number of rows in the table.
    row_count: u64,
    /// Unix timestamp (microseconds) of last ANALYZE.
    last_analyze_time: i64,

    pub fn init(row_count: u64) TableStats {
        const now = std.time.microTimestamp();
        return .{
            .row_count = row_count,
            .last_analyze_time = now,
        };
    }
};

/// Most Common Value — a value and its frequency in the column.
pub const MostCommonValue = struct {
    /// Serialized value bytes (tag + data).
    value: []const u8,
    /// Fraction of rows with this value (0.0 to 1.0).
    frequency: f64,

    pub fn deinit(self: MostCommonValue, allocator: Allocator) void {
        allocator.free(self.value);
    }
};

/// Histogram bucket for equi-depth histograms.
pub const HistogramBucket = struct {
    /// Lower bound value (inclusive).
    lower: []const u8,
    /// Upper bound value (inclusive).
    upper: []const u8,
    /// Number of rows in this bucket.
    count: u64,

    pub fn deinit(self: HistogramBucket, allocator: Allocator) void {
        allocator.free(self.lower);
        allocator.free(self.upper);
    }
};

/// Column-level statistics.
pub const ColumnStats = struct {
    /// Number of distinct values in this column.
    distinct_count: u64,
    /// Fraction of rows with NULL values (0.0 to 1.0).
    null_fraction: f64,
    /// Average storage width in bytes.
    avg_width: f64,
    /// Correlation coefficient with physical row order (-1.0 to 1.0).
    /// Values close to 1.0 or -1.0 indicate good correlation with index scans.
    correlation: f64,
    /// Most common values and their frequencies.
    most_common_values: []const MostCommonValue,
    /// Equi-depth histogram buckets.
    histogram_buckets: []const HistogramBucket,

    pub fn deinit(self: ColumnStats, allocator: Allocator) void {
        for (self.most_common_values) |mcv| mcv.deinit(allocator);
        allocator.free(self.most_common_values);
        for (self.histogram_buckets) |bucket| bucket.deinit(allocator);
        allocator.free(self.histogram_buckets);
    }
};

// ── Serialization ───────────────────────────────────────────────────────

/// Serialize table statistics to bytes for catalog storage.
pub fn serializeTableStats(allocator: Allocator, stats: TableStats) ![]u8 {
    var buf = std.ArrayListUnmanaged(u8){};
    errdefer buf.deinit(allocator);

    const writer = buf.writer(allocator);
    try writer.writeInt(u64, stats.row_count, .little);
    try writer.writeInt(i64, stats.last_analyze_time, .little);

    return try buf.toOwnedSlice(allocator);
}

/// Deserialize table statistics from catalog bytes.
pub fn deserializeTableStats(data: []const u8) !TableStats {
    if (data.len < 16) return error.InvalidData;

    var pos: usize = 0;
    const row_count = std.mem.readInt(u64, data[pos..][0..8], .little);
    pos += 8;
    const last_analyze_time = std.mem.readInt(i64, data[pos..][0..8], .little);

    return TableStats{
        .row_count = row_count,
        .last_analyze_time = last_analyze_time,
    };
}

/// Serialize column statistics to bytes for catalog storage.
pub fn serializeColumnStats(allocator: Allocator, stats: ColumnStats) ![]u8 {
    var buf = std.ArrayListUnmanaged(u8){};
    errdefer buf.deinit(allocator);

    const writer = buf.writer(allocator);

    // Basic stats
    try writer.writeInt(u64, stats.distinct_count, .little);
    try writer.writeAll(std.mem.asBytes(&stats.null_fraction));
    try writer.writeAll(std.mem.asBytes(&stats.avg_width));
    try writer.writeAll(std.mem.asBytes(&stats.correlation));

    // Most common values
    try writer.writeInt(u16, @intCast(stats.most_common_values.len), .little);
    for (stats.most_common_values) |mcv| {
        try writer.writeInt(u16, @intCast(mcv.value.len), .little);
        try writer.writeAll(mcv.value);
        try writer.writeAll(std.mem.asBytes(&mcv.frequency));
    }

    // Histogram buckets
    try writer.writeInt(u16, @intCast(stats.histogram_buckets.len), .little);
    for (stats.histogram_buckets) |bucket| {
        try writer.writeInt(u16, @intCast(bucket.lower.len), .little);
        try writer.writeAll(bucket.lower);
        try writer.writeInt(u16, @intCast(bucket.upper.len), .little);
        try writer.writeAll(bucket.upper);
        try writer.writeInt(u64, bucket.count, .little);
    }

    return try buf.toOwnedSlice(allocator);
}

/// Deserialize column statistics from catalog bytes.
pub fn deserializeColumnStats(allocator: Allocator, data: []const u8) !ColumnStats {
    var pos: usize = 0;

    // Minimum size: 8 (distinct_count) + 8 (null_fraction) + 8 (avg_width) + 8 (correlation) + 2 (mcv_count) + 2 (bucket_count) = 36 bytes
    if (data.len < 36) return error.InvalidData;

    const distinct_count = std.mem.readInt(u64, data[pos..][0..8], .little);
    pos += 8;

    // Read f64 values using bytesAsValue to handle potential misalignment
    const null_fraction = std.mem.bytesAsValue(f64, data[pos..][0..8]).*;
    pos += 8;
    const avg_width = std.mem.bytesAsValue(f64, data[pos..][0..8]).*;
    pos += 8;
    const correlation = std.mem.bytesAsValue(f64, data[pos..][0..8]).*;
    pos += 8;

    // Most common values
    if (pos + 2 > data.len) return error.InvalidData;
    const mcv_count = std.mem.readInt(u16, data[pos..][0..2], .little);
    pos += 2;

    var mcvs = std.ArrayListUnmanaged(MostCommonValue){};
    errdefer {
        for (mcvs.items) |mcv| mcv.deinit(allocator);
        mcvs.deinit(allocator);
    }

    for (0..mcv_count) |_| {
        if (pos + 2 > data.len) return error.InvalidData;
        const value_len = std.mem.readInt(u16, data[pos..][0..2], .little);
        pos += 2;

        if (pos + value_len > data.len) return error.InvalidData;
        const value = try allocator.dupe(u8, data[pos..][0..value_len]);
        pos += value_len;

        if (pos + 8 > data.len) return error.InvalidData;
        const frequency = std.mem.bytesAsValue(f64, data[pos..][0..8]).*;
        pos += 8;

        try mcvs.append(allocator, .{ .value = value, .frequency = frequency });
    }

    // Histogram buckets
    if (pos + 2 > data.len) return error.InvalidData;
    const bucket_count = std.mem.readInt(u16, data[pos..][0..2], .little);
    pos += 2;

    var buckets = std.ArrayListUnmanaged(HistogramBucket){};
    errdefer {
        for (buckets.items) |bucket| bucket.deinit(allocator);
        buckets.deinit(allocator);
    }

    for (0..bucket_count) |_| {
        if (pos + 2 > data.len) return error.InvalidData;
        const lower_len = std.mem.readInt(u16, data[pos..][0..2], .little);
        pos += 2;

        if (pos + lower_len > data.len) return error.InvalidData;
        const lower = try allocator.dupe(u8, data[pos..][0..lower_len]);
        pos += lower_len;

        if (pos + 2 > data.len) return error.InvalidData;
        const upper_len = std.mem.readInt(u16, data[pos..][0..2], .little);
        pos += 2;

        if (pos + upper_len > data.len) return error.InvalidData;
        const upper = try allocator.dupe(u8, data[pos..][0..upper_len]);
        pos += upper_len;

        if (pos + 8 > data.len) return error.InvalidData;
        const count = std.mem.readInt(u64, data[pos..][0..8], .little);
        pos += 8;

        try buckets.append(allocator, .{ .lower = lower, .upper = upper, .count = count });
    }

    return ColumnStats{
        .distinct_count = distinct_count,
        .null_fraction = null_fraction,
        .avg_width = avg_width,
        .correlation = correlation,
        .most_common_values = try mcvs.toOwnedSlice(allocator),
        .histogram_buckets = try buckets.toOwnedSlice(allocator),
    };
}

// ── Tests ───────────────────────────────────────────────────────────────

const testing = std.testing;

test "TableStats init and serialize/deserialize" {
    const alloc = testing.allocator;

    const stats = TableStats.init(1000);
    try testing.expect(stats.row_count == 1000);
    try testing.expect(stats.last_analyze_time > 0);

    const data = try serializeTableStats(alloc, stats);
    defer alloc.free(data);

    const deserialized = try deserializeTableStats(data);
    try testing.expectEqual(stats.row_count, deserialized.row_count);
    try testing.expectEqual(stats.last_analyze_time, deserialized.last_analyze_time);
}

test "ColumnStats serialize/deserialize with empty MCVs and histograms" {
    const alloc = testing.allocator;

    const stats = ColumnStats{
        .distinct_count = 500,
        .null_fraction = 0.1,
        .avg_width = 12.5,
        .correlation = 0.85,
        .most_common_values = &.{},
        .histogram_buckets = &.{},
    };

    const data = try serializeColumnStats(alloc, stats);
    defer alloc.free(data);

    var deserialized = try deserializeColumnStats(alloc, data);
    defer deserialized.deinit(alloc);

    try testing.expectEqual(stats.distinct_count, deserialized.distinct_count);
    try testing.expectEqual(stats.null_fraction, deserialized.null_fraction);
    try testing.expectEqual(stats.avg_width, deserialized.avg_width);
    try testing.expectEqual(stats.correlation, deserialized.correlation);
    try testing.expectEqual(@as(usize, 0), deserialized.most_common_values.len);
    try testing.expectEqual(@as(usize, 0), deserialized.histogram_buckets.len);
}

test "ColumnStats serialize/deserialize with MCVs" {
    const alloc = testing.allocator;

    const value1 = try alloc.dupe(u8, &[_]u8{ 0x01, 0x2A, 0x00, 0x00, 0x00 }); // integer 42
    defer alloc.free(value1);
    const value2 = try alloc.dupe(u8, &[_]u8{ 0x01, 0x64, 0x00, 0x00, 0x00 }); // integer 100
    defer alloc.free(value2);

    const mcvs = [_]MostCommonValue{
        .{ .value = value1, .frequency = 0.15 },
        .{ .value = value2, .frequency = 0.10 },
    };

    const stats = ColumnStats{
        .distinct_count = 200,
        .null_fraction = 0.05,
        .avg_width = 8.0,
        .correlation = -0.3,
        .most_common_values = &mcvs,
        .histogram_buckets = &.{},
    };

    const data = try serializeColumnStats(alloc, stats);
    defer alloc.free(data);

    var deserialized = try deserializeColumnStats(alloc, data);
    defer deserialized.deinit(alloc);

    try testing.expectEqual(stats.distinct_count, deserialized.distinct_count);
    try testing.expectEqual(@as(usize, 2), deserialized.most_common_values.len);
    try testing.expectEqualSlices(u8, value1, deserialized.most_common_values[0].value);
    try testing.expectEqual(@as(f64, 0.15), deserialized.most_common_values[0].frequency);
}

test "ColumnStats serialize/deserialize with histogram" {
    const alloc = testing.allocator;

    const lower1 = try alloc.dupe(u8, &[_]u8{ 0x01, 0x00, 0x00, 0x00, 0x00 }); // integer 0
    defer alloc.free(lower1);
    const upper1 = try alloc.dupe(u8, &[_]u8{ 0x01, 0x63, 0x00, 0x00, 0x00 }); // integer 99
    defer alloc.free(upper1);

    const buckets = [_]HistogramBucket{
        .{ .lower = lower1, .upper = upper1, .count = 100 },
    };

    const stats = ColumnStats{
        .distinct_count = 100,
        .null_fraction = 0.0,
        .avg_width = 4.0,
        .correlation = 1.0,
        .most_common_values = &.{},
        .histogram_buckets = &buckets,
    };

    const data = try serializeColumnStats(alloc, stats);
    defer alloc.free(data);

    var deserialized = try deserializeColumnStats(alloc, data);
    defer deserialized.deinit(alloc);

    try testing.expectEqual(@as(usize, 1), deserialized.histogram_buckets.len);
    try testing.expectEqualSlices(u8, lower1, deserialized.histogram_buckets[0].lower);
    try testing.expectEqualSlices(u8, upper1, deserialized.histogram_buckets[0].upper);
    try testing.expectEqual(@as(u64, 100), deserialized.histogram_buckets[0].count);
}

test "TableStats deserialization with invalid data" {
    try testing.expectError(error.InvalidData, deserializeTableStats(&[_]u8{0x01}));
    try testing.expectError(error.InvalidData, deserializeTableStats(&[_]u8{}));
}

test "ColumnStats deserialization with invalid data" {
    const alloc = testing.allocator;

    // Too short
    try testing.expectError(error.InvalidData, deserializeColumnStats(alloc, &[_]u8{0x01}));

    // Truncated MCV section
    var buf: [40]u8 = undefined;
    std.mem.writeInt(u64, buf[0..8], 100, .little);
    @memset(buf[8..40], 0);
    std.mem.writeInt(u16, buf[32..34], 5, .little); // 5 MCVs claimed but no data
    try testing.expectError(error.InvalidData, deserializeColumnStats(alloc, &buf));
}
