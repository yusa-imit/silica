const std = @import("std");
const silica = @import("silica");

/// Benchmark result for a single operation
pub const BenchmarkResult = struct {
    name: []const u8,
    iterations: usize,
    total_ns: u64,
    mean_ns: u64,
    min_ns: u64,
    max_ns: u64,
    stddev_ns: f64,
    ops_per_sec: f64,

    pub fn print(self: BenchmarkResult, writer: anytype) !void {
        try writer.print("Benchmark: {s}\n", .{self.name});
        try writer.print("  Iterations: {d}\n", .{self.iterations});
        try writer.print("  Mean: {d:.2} µs\n", .{@as(f64, @floatFromInt(self.mean_ns)) / 1000.0});
        try writer.print("  Min: {d:.2} µs\n", .{@as(f64, @floatFromInt(self.min_ns)) / 1000.0});
        try writer.print("  Max: {d:.2} µs\n", .{@as(f64, @floatFromInt(self.max_ns)) / 1000.0});
        try writer.print("  Stddev: {d:.2} µs\n", .{self.stddev_ns / 1000.0});
        try writer.print("  Ops/sec: {d:.0}\n", .{self.ops_per_sec});
        try writer.print("\n", .{});
    }
};

/// Run a benchmark function multiple times and collect statistics
pub fn runBenchmark(
    allocator: std.mem.Allocator,
    comptime name: []const u8,
    comptime iterations: usize,
    comptime warmup: usize,
    bench_fn: anytype,
) !BenchmarkResult {
    var timings = try allocator.alloc(u64, iterations);
    defer allocator.free(timings);

    // Warmup runs
    var i: usize = 0;
    while (i < warmup) : (i += 1) {
        try bench_fn();
    }

    // Measured runs
    i = 0;
    while (i < iterations) : (i += 1) {
        const start = std.time.nanoTimestamp();
        try bench_fn();
        const end = std.time.nanoTimestamp();
        timings[i] = @intCast(end - start);
    }

    // Calculate statistics
    var min: u64 = std.math.maxInt(u64);
    var max: u64 = 0;
    var sum: u64 = 0;

    for (timings) |timing| {
        if (timing < min) min = timing;
        if (timing > max) max = timing;
        sum += timing;
    }

    const mean = sum / iterations;

    // Calculate standard deviation
    var variance_sum: f64 = 0.0;
    for (timings) |timing| {
        const diff = @as(f64, @floatFromInt(timing)) - @as(f64, @floatFromInt(mean));
        variance_sum += diff * diff;
    }
    const stddev = @sqrt(variance_sum / @as(f64, @floatFromInt(iterations)));

    const ops_per_sec = 1_000_000_000.0 / @as(f64, @floatFromInt(mean));

    return BenchmarkResult{
        .name = name,
        .iterations = iterations,
        .total_ns = sum,
        .mean_ns = mean,
        .min_ns = min,
        .max_ns = max,
        .stddev_ns = stddev,
        .ops_per_sec = ops_per_sec,
    };
}

/// Benchmark suite to group related benchmarks
pub const BenchmarkSuite = struct {
    name: []const u8,
    results: std.ArrayList(BenchmarkResult),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, name: []const u8) BenchmarkSuite {
        return .{
            .name = name,
            .results = std.ArrayList(BenchmarkResult).fromOwnedSlice(&.{}),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *BenchmarkSuite) void {
        self.results.deinit();
    }

    pub fn add(self: *BenchmarkSuite, result: BenchmarkResult) !void {
        try self.results.append(result);
    }

    pub fn print(self: BenchmarkSuite, writer: anytype) !void {
        try writer.print("=== Benchmark Suite: {s} ===\n\n", .{self.name});
        for (self.results.items) |result| {
            try result.print(writer);
        }
        try writer.print("Total benchmarks: {d}\n", .{self.results.items.len});
    }

    /// Check if benchmark meets target requirements from PRD
    pub fn checkTarget(result: BenchmarkResult, target_us: f64) bool {
        const mean_us = @as(f64, @floatFromInt(result.mean_ns)) / 1000.0;
        return mean_us < target_us;
    }
};

test "benchmark harness basic functionality" {
    const allocator = std.testing.allocator;

    var suite = BenchmarkSuite.init(allocator, "Test Suite");
    defer suite.deinit();

    const result = try runBenchmark(allocator, "increment counter", 100, 10, struct {
        fn bench() !void {
            var cnt: usize = 0;
            cnt += 1;
            // Simulate some work
            var i: usize = 0;
            while (i < 100) : (i += 1) {
                std.mem.doNotOptimizeAway(i);
            }
        }
    }.bench);

    try suite.add(result);

    // Verify result fields are populated
    try std.testing.expect(result.iterations == 100);
    try std.testing.expect(result.mean_ns > 0);
    try std.testing.expect(result.min_ns > 0);
    try std.testing.expect(result.max_ns >= result.min_ns);
    try std.testing.expect(result.ops_per_sec > 0);
}
