//! Cost Model — estimates query execution cost for plan optimization.
//!
//! This module implements PostgreSQL-style cost estimation:
//!   - Sequential scan cost: (N_pages * seq_page_cost) + (N_tuples * cpu_tuple_cost)
//!   - Index scan cost: (N_index_pages * random_page_cost) + (N_tuples * cpu_index_tuple_cost)
//!   - Join cost: (left_cost + right_cost) + (N_outer * N_inner * cpu_join_cost)
//!   - Sort cost: (N * log2(N) * cpu_sort_cost)
//!
//! Cost units are arbitrary but consistent — relative costs guide plan selection.

const std = @import("std");
const catalog_mod = @import("catalog.zig");
const selectivity_mod = @import("selectivity.zig");
const ast = @import("ast.zig");

const TableStats = @import("stats.zig").TableStats;
const ColumnStats = @import("stats.zig").ColumnStats;

// ── Cost Parameters ───────────────────────────────────────────────────

/// Cost configuration — default values based on PostgreSQL defaults.
pub const CostConfig = struct {
    /// Cost to fetch a sequential page (buffer pool hit or disk read).
    seq_page_cost: f64 = 1.0,

    /// Cost to fetch a random page (index lookup).
    random_page_cost: f64 = 4.0,

    /// CPU cost to process one tuple (row).
    cpu_tuple_cost: f64 = 0.01,

    /// CPU cost to process one index tuple.
    cpu_index_tuple_cost: f64 = 0.005,

    /// CPU cost per operator evaluation in a qual (WHERE clause).
    cpu_operator_cost: f64 = 0.0025,

    /// CPU cost per JOIN tuple pair comparison.
    cpu_join_cost: f64 = 0.01,

    /// CPU cost per sort comparison.
    cpu_sort_cost: f64 = 0.01,

    /// Effective cache size (in pages) — affects caching assumptions.
    effective_cache_size: u32 = 16384, // 64 MB for 4KB pages

    /// Page size in bytes (for I/O cost calculations).
    page_size: u32 = 4096,
};

// ── Cost Estimation ───────────────────────────────────────────────────

/// Cost estimator for query plans.
pub const CostEstimator = struct {
    config: CostConfig,

    pub fn init(config: CostConfig) CostEstimator {
        return .{ .config = config };
    }

    /// Estimate sequential scan cost.
    ///
    /// Cost = (N_pages * seq_page_cost) + (N_tuples * cpu_tuple_cost)
    ///        + (N_tuples * N_quals * cpu_operator_cost)
    ///
    /// If avg_row_size is not provided, assumes 100 bytes per row.
    pub fn estimateSeqScan(
        self: *const CostEstimator,
        row_count: u64,
        avg_row_size: ?u32,
        num_quals: u32,
    ) f64 {
        const n_tuples: f64 = @floatFromInt(row_count);
        const n_quals: f64 = @floatFromInt(num_quals);

        // Estimate page count based on row count and avg row size
        const avg_size: u32 = avg_row_size orelse 100;
        const rows_per_page = self.config.page_size / avg_size;
        const estimated_pages = if (rows_per_page > 0)
            (row_count + rows_per_page - 1) / rows_per_page
        else
            row_count; // fallback for very large rows
        const n_pages: f64 = @floatFromInt(estimated_pages);

        const io_cost = n_pages * self.config.seq_page_cost;
        const cpu_cost = n_tuples * self.config.cpu_tuple_cost;
        const qual_cost = n_tuples * n_quals * self.config.cpu_operator_cost;

        return io_cost + cpu_cost + qual_cost;
    }

    /// Estimate index scan cost.
    ///
    /// Cost = (N_index_pages * random_page_cost)
    ///        + (N_tuples * cpu_index_tuple_cost)
    ///        + (N_tuples * N_quals * cpu_operator_cost)
    pub fn estimateIndexScan(
        self: *const CostEstimator,
        n_index_pages: u32,
        n_tuples: u32,
        num_quals: u32,
    ) f64 {
        const n_pages: f64 = @floatFromInt(n_index_pages);
        const n_rows: f64 = @floatFromInt(n_tuples);
        const n_quals: f64 = @floatFromInt(num_quals);

        const io_cost = n_pages * self.config.random_page_cost;
        const cpu_cost = n_rows * self.config.cpu_index_tuple_cost;
        const qual_cost = n_rows * n_quals * self.config.cpu_operator_cost;

        return io_cost + cpu_cost + qual_cost;
    }

    /// Estimate nested loop join cost.
    ///
    /// Cost = outer_cost + inner_cost
    ///        + (N_outer * N_inner * cpu_join_cost)
    pub fn estimateNestedLoopJoin(
        self: *const CostEstimator,
        outer_cost: f64,
        inner_cost: f64,
        n_outer: u32,
        n_inner: u32,
    ) f64 {
        const outer_rows: f64 = @floatFromInt(n_outer);
        const inner_rows: f64 = @floatFromInt(n_inner);

        const join_cost = outer_rows * inner_rows * self.config.cpu_join_cost;
        return outer_cost + inner_cost + join_cost;
    }

    /// Estimate hash join cost.
    ///
    /// Cost = outer_cost + inner_cost
    ///        + (N_outer * cpu_operator_cost) [build hash table]
    ///        + (N_inner * cpu_operator_cost) [probe hash table]
    pub fn estimateHashJoin(
        self: *const CostEstimator,
        outer_cost: f64,
        inner_cost: f64,
        n_outer: u32,
        n_inner: u32,
    ) f64 {
        const outer_rows: f64 = @floatFromInt(n_outer);
        const inner_rows: f64 = @floatFromInt(n_inner);

        const build_cost = outer_rows * self.config.cpu_operator_cost;
        const probe_cost = inner_rows * self.config.cpu_operator_cost;

        return outer_cost + inner_cost + build_cost + probe_cost;
    }

    /// Estimate sort cost.
    ///
    /// Cost = N * log2(N) * cpu_sort_cost
    pub fn estimateSort(
        self: *const CostEstimator,
        input_cost: f64,
        n_tuples: u32,
    ) f64 {
        if (n_tuples == 0) return input_cost;

        const n: f64 = @floatFromInt(n_tuples);
        const log_n = @log2(n);
        const sort_cost = n * log_n * self.config.cpu_sort_cost;

        return input_cost + sort_cost;
    }

    /// Estimate aggregate cost.
    ///
    /// Cost = input_cost + (N_tuples * cpu_operator_cost)
    pub fn estimateAggregate(
        self: *const CostEstimator,
        input_cost: f64,
        n_tuples: u32,
    ) f64 {
        const n: f64 = @floatFromInt(n_tuples);
        return input_cost + (n * self.config.cpu_operator_cost);
    }

    /// Estimate limit cost (nearly free, just row count reduction).
    ///
    /// Cost = input_cost * (limit / n_tuples)
    pub fn estimateLimit(
        self: *const CostEstimator,
        input_cost: f64,
        limit: u32,
        n_tuples: u32,
    ) f64 {
        _ = self;
        if (n_tuples == 0) return input_cost;

        const limit_f: f64 = @floatFromInt(limit);
        const n: f64 = @floatFromInt(n_tuples);

        // If limit > n_tuples, return full cost
        if (limit_f >= n) return input_cost;

        // Otherwise proportional cost
        return input_cost * (limit_f / n);
    }
};

// ── Tests ─────────────────────────────────────────────────────────────

test "CostEstimator: init with default config" {
    const estimator = CostEstimator.init(.{});
    try std.testing.expectEqual(@as(f64, 1.0), estimator.config.seq_page_cost);
    try std.testing.expectEqual(@as(f64, 4.0), estimator.config.random_page_cost);
}

test "CostEstimator: sequential scan cost" {
    const estimator = CostEstimator.init(.{});

    // 10000 rows, avg row size 100 bytes
    // Pages = ceil(10000 / (4096/100)) = ceil(10000 / 40) = 250 pages
    // Cost = (250 * 1.0) [I/O] + (10000 * 0.01) [CPU] + (10000 * 2 * 0.0025) [quals]
    //      = 250 + 100 + 50 = 400
    const cost = estimator.estimateSeqScan(10000, 100, 2);
    try std.testing.expectApproxEqAbs(@as(f64, 400.0), cost, 0.01);
}

test "CostEstimator: index scan cost" {
    const estimator = CostEstimator.init(.{});

    // Index scan fetching 100 tuples with 10 index pages
    // Cost = (10 * 4.0) [I/O] + (100 * 0.005) [CPU] + (100 * 1 * 0.0025) [quals]
    //      = 40 + 0.5 + 0.25 = 40.75
    const cost = estimator.estimateIndexScan(10, 100, 1);
    try std.testing.expectApproxEqAbs(@as(f64, 40.75), cost, 0.01);
}

test "CostEstimator: nested loop join cost" {
    const estimator = CostEstimator.init(.{});

    // Outer: 1000 rows, cost 100
    // Inner: 10 rows, cost 5
    // Join cost = 100 + 5 + (1000 * 10 * 0.01) = 105 + 100 = 205
    const cost = estimator.estimateNestedLoopJoin(100.0, 5.0, 1000, 10);
    try std.testing.expectApproxEqAbs(@as(f64, 205.0), cost, 0.01);
}

test "CostEstimator: hash join cost" {
    const estimator = CostEstimator.init(.{});

    // Outer: 1000 rows, cost 100
    // Inner: 1000 rows, cost 100
    // Build + probe = (1000 * 0.0025) + (1000 * 0.0025) = 5.0
    // Total = 100 + 100 + 5.0 = 205.0
    const cost = estimator.estimateHashJoin(100.0, 100.0, 1000, 1000);
    try std.testing.expectApproxEqAbs(@as(f64, 205.0), cost, 0.01);
}

test "CostEstimator: sort cost" {
    const estimator = CostEstimator.init(.{});

    // Sort 1000 rows: 1000 * log2(1000) * 0.01 ≈ 1000 * 9.97 * 0.01 ≈ 99.7
    const cost = estimator.estimateSort(50.0, 1000);
    try std.testing.expectApproxEqAbs(@as(f64, 149.7), cost, 1.0);
}

test "CostEstimator: aggregate cost" {
    const estimator = CostEstimator.init(.{});

    // Aggregate 5000 rows: input_cost + (5000 * 0.0025) = 100 + 12.5 = 112.5
    const cost = estimator.estimateAggregate(100.0, 5000);
    try std.testing.expectApproxEqAbs(@as(f64, 112.5), cost, 0.01);
}

test "CostEstimator: limit cost" {
    const estimator = CostEstimator.init(.{});

    // LIMIT 100 from 10000 rows: 200 * (100 / 10000) = 2.0
    const cost = estimator.estimateLimit(200.0, 100, 10000);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), cost, 0.01);
}

test "CostEstimator: limit larger than n_tuples" {
    const estimator = CostEstimator.init(.{});

    // LIMIT 5000 from 1000 rows: should return full cost
    const cost = estimator.estimateLimit(100.0, 5000, 1000);
    try std.testing.expectApproxEqAbs(@as(f64, 100.0), cost, 0.01);
}

test "CostEstimator: zero tuples edge cases" {
    const estimator = CostEstimator.init(.{});

    // Zero tuples in scan (0 pages estimated)
    const scan_cost = estimator.estimateSeqScan(0, 100, 0);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), scan_cost, 0.01);

    // Zero tuples in sort
    const sort_cost = estimator.estimateSort(10.0, 0);
    try std.testing.expectApproxEqAbs(@as(f64, 10.0), sort_cost, 0.01);

    // Zero tuples in limit
    const limit_cost = estimator.estimateLimit(10.0, 5, 0);
    try std.testing.expectApproxEqAbs(@as(f64, 10.0), limit_cost, 0.01);
}

test "CostEstimator: custom config" {
    const custom_config = CostConfig{
        .seq_page_cost = 2.0,
        .random_page_cost = 8.0,
        .cpu_tuple_cost = 0.02,
    };
    const estimator = CostEstimator.init(custom_config);

    // 1000 rows, avg row size 50 bytes
    // Pages = ceil(1000 / (4096/50)) = ceil(1000 / 81) = 13 pages
    // Cost = (13 * 2.0) + (1000 * 0.02) + (1000 * 0 * 0.0025) = 26 + 20 + 0 = 46
    const cost = estimator.estimateSeqScan(1000, 50, 0);
    try std.testing.expectApproxEqAbs(@as(f64, 46.0), cost, 0.01);
}

test "CostEstimator: hash join cheaper than nested loop for large tables" {
    const estimator = CostEstimator.init(.{});

    // Large tables: nested loop is expensive
    const nested_loop_cost = estimator.estimateNestedLoopJoin(100.0, 100.0, 10000, 10000);
    const hash_join_cost = estimator.estimateHashJoin(100.0, 100.0, 10000, 10000);

    // Nested loop: 100 + 100 + (10000 * 10000 * 0.01) = 1,000,200
    // Hash join: 100 + 100 + (10000 * 0.0025) + (10000 * 0.0025) = 250
    try std.testing.expect(hash_join_cost < nested_loop_cost);
}

// ── Edge Case Tests ──────────────────────────────────────────────────

test "CostEstimator: null avg_row_size uses default" {
    const estimator = CostEstimator.init(.{});

    // No avg_row_size provided, should use default 100 bytes
    const cost = estimator.estimateSeqScan(1000, null, 0);
    // Pages = ceil(1000 / (4096/100)) = ceil(1000 / 40) = 25
    // Cost = (25 * 1.0) + (1000 * 0.01) = 35
    try std.testing.expectApproxEqAbs(@as(f64, 35.0), cost, 0.01);
}

test "CostEstimator: very large row size" {
    const estimator = CostEstimator.init(.{});

    // Row size larger than page size
    const cost = estimator.estimateSeqScan(100, 8192, 0);
    // rows_per_page = 4096 / 8192 = 0 → fallback: 100 pages
    // Cost = (100 * 1.0) + (100 * 0.01) = 101
    try std.testing.expectApproxEqAbs(@as(f64, 101.0), cost, 0.01);
}

test "CostEstimator: very small row size" {
    const estimator = CostEstimator.init(.{});

    // Very small rows (10 bytes each)
    const cost = estimator.estimateSeqScan(10000, 10, 0);
    // rows_per_page = 4096 / 10 = 409
    // pages = ceil(10000 / 409) = 25
    // Cost = (25 * 1.0) + (10000 * 0.01) = 125
    try std.testing.expectApproxEqAbs(@as(f64, 125.0), cost, 1.0);
}

test "CostEstimator: index scan with zero tuples" {
    const estimator = CostEstimator.init(.{});

    const cost = estimator.estimateIndexScan(5, 0, 2);
    // Cost = (5 * 4.0) + (0 * 0.005) + (0 * 2 * 0.0025) = 20
    try std.testing.expectApproxEqAbs(@as(f64, 20.0), cost, 0.01);
}

test "CostEstimator: index scan with many quals" {
    const estimator = CostEstimator.init(.{});

    // Many filter predicates increase CPU cost
    const cost = estimator.estimateIndexScan(10, 100, 10);
    // Cost = (10 * 4.0) + (100 * 0.005) + (100 * 10 * 0.0025)
    //      = 40 + 0.5 + 2.5 = 43
    try std.testing.expectApproxEqAbs(@as(f64, 43.0), cost, 0.01);
}

test "CostEstimator: nested loop join with zero outer rows" {
    const estimator = CostEstimator.init(.{});

    const cost = estimator.estimateNestedLoopJoin(50.0, 30.0, 0, 1000);
    // Join cost is 0 since no outer rows
    // Total = 50 + 30 + 0 = 80
    try std.testing.expectApproxEqAbs(@as(f64, 80.0), cost, 0.01);
}

test "CostEstimator: nested loop join with zero inner rows" {
    const estimator = CostEstimator.init(.{});

    const cost = estimator.estimateNestedLoopJoin(50.0, 30.0, 1000, 0);
    // Join cost is 0 since no inner rows
    // Total = 50 + 30 + 0 = 80
    try std.testing.expectApproxEqAbs(@as(f64, 80.0), cost, 0.01);
}

test "CostEstimator: hash join with very large outer table" {
    const estimator = CostEstimator.init(.{});

    // Large outer table → expensive hash build
    const cost = estimator.estimateHashJoin(1000.0, 100.0, 1000000, 1000);
    // Build = 1000000 * 0.0025 = 2500
    // Probe = 1000 * 0.0025 = 2.5
    // Total = 1000 + 100 + 2500 + 2.5 = 3602.5
    try std.testing.expectApproxEqAbs(@as(f64, 3602.5), cost, 0.01);
}

test "CostEstimator: sort single row" {
    const estimator = CostEstimator.init(.{});

    // Sorting 1 row: log2(1) = 0
    const cost = estimator.estimateSort(10.0, 1);
    try std.testing.expectApproxEqAbs(@as(f64, 10.0), cost, 0.01);
}

test "CostEstimator: sort very large dataset" {
    const estimator = CostEstimator.init(.{});

    // Sort 1 million rows: 1M * log2(1M) * 0.01 ≈ 1M * 19.93 * 0.01 ≈ 199,300
    const cost = estimator.estimateSort(1000.0, 1000000);
    try std.testing.expectApproxEqAbs(@as(f64, 200300.0), cost, 500.0);
}

test "CostEstimator: aggregate with zero tuples" {
    const estimator = CostEstimator.init(.{});

    const cost = estimator.estimateAggregate(50.0, 0);
    try std.testing.expectApproxEqAbs(@as(f64, 50.0), cost, 0.01);
}

test "CostEstimator: limit equal to tuple count" {
    const estimator = CostEstimator.init(.{});

    // LIMIT 100 from 100 rows → full cost
    const cost = estimator.estimateLimit(200.0, 100, 100);
    try std.testing.expectApproxEqAbs(@as(f64, 200.0), cost, 0.01);
}

test "CostEstimator: limit very small fraction" {
    const estimator = CostEstimator.init(.{});

    // LIMIT 10 from 1,000,000 rows → very small fraction
    const cost = estimator.estimateLimit(100000.0, 10, 1000000);
    // Cost = 100000 * (10 / 1000000) = 1.0
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), cost, 0.01);
}

test "CostEstimator: comparison - seq scan vs index scan selectivity" {
    const estimator = CostEstimator.init(.{});

    // Table: 100,000 rows, avg size 100 bytes
    // Pages = ceil(100000 / (4096/100)) = ceil(100000 / 40) = 2500
    // Seq scan cost = (2500 * 1.0) + (100000 * 0.01) + (100000 * 1 * 0.0025)
    //               = 2500 + 1000 + 250 = 3750
    const seq_cost = estimator.estimateSeqScan(100000, 100, 1);

    // Index scan fetching 1% of rows (highly selective) = 1000 rows
    // Assume index depth 3, so 3 random pages + ceil(1000 / 40) = 3 + 25 = 28 pages
    // Index scan cost = (28 * 4.0) + (1000 * 0.005) + (1000 * 1 * 0.0025)
    //                 = 112 + 5 + 2.5 = 119.5
    const index_cost = estimator.estimateIndexScan(28, 1000, 1);

    // For highly selective queries (1%), index scan should be cheaper
    try std.testing.expect(index_cost < seq_cost);
}

