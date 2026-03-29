//! Selectivity Estimation — estimate the fraction of rows matching predicates.
//!
//! Selectivity values range from 0.0 (no rows match) to 1.0 (all rows match).
//! These estimates are used by the cost-based optimizer to choose efficient query plans.
//!
//! Estimation methods:
//! - Equality (col = val): Uses MCVs if available, otherwise 1/distinct_count
//! - Range (col > val, col BETWEEN a AND b): Uses histogram buckets
//! - LIKE (col LIKE pattern): Pattern-aware estimates (prefix 10%, suffix/substring 20%, exact 1%)
//! - IN (col IN (v1, v2, ...)): Sum of individual equality estimates
//! - IS NULL: Uses null_fraction from statistics

const std = @import("std");
const Allocator = std.mem.Allocator;
const ast = @import("ast.zig");
const stats_mod = @import("stats.zig");

const TableStats = stats_mod.TableStats;
const ColumnStats = stats_mod.ColumnStats;

// ── Constants ───────────────────────────────────────────────────────────

/// Default selectivity when no statistics are available.
const DEFAULT_SELECTIVITY: f64 = 0.1;

/// Default selectivity for equality predicates without statistics.
const DEFAULT_EQUALITY_SELECTIVITY: f64 = 0.01;

/// Default selectivity for range predicates without statistics.
const DEFAULT_RANGE_SELECTIVITY: f64 = 0.33;

/// Default selectivity for LIKE 'prefix%' patterns.
const LIKE_PREFIX_SELECTIVITY: f64 = 0.1;

/// Default selectivity for LIKE '%suffix' or LIKE '%substring%' patterns.
const LIKE_SUBSTRING_SELECTIVITY: f64 = 0.2;

/// Minimum selectivity for any predicate (avoid zero estimates).
const MIN_SELECTIVITY: f64 = 0.0001;

// ── Selectivity Estimator ───────────────────────────────────────────────

pub const SelectivityEstimator = struct {
    allocator: Allocator,

    pub fn init(allocator: Allocator) SelectivityEstimator {
        return .{ .allocator = allocator };
    }

    /// Estimate selectivity for a WHERE clause predicate.
    /// Returns a value between 0.0 and 1.0.
    pub fn estimatePredicate(
        self: *SelectivityEstimator,
        predicate: *const ast.Expr,
        table_stats: ?TableStats,
        column_stats_map: ?std.StringHashMap(ColumnStats),
    ) f64 {
        _ = table_stats;
        return self.estimateExpr(predicate, column_stats_map);
    }

    fn estimateExpr(
        self: *SelectivityEstimator,
        expr: *const ast.Expr,
        column_stats_map: ?std.StringHashMap(ColumnStats),
    ) f64 {
        return switch (expr.*) {
            .binary_op => |b| {
                // Check if this is a logical operator (AND/OR) or a comparison
                return switch (b.op) {
                    .@"and" => self.estimateAnd(b.left, b.right, column_stats_map),
                    .@"or" => self.estimateOr(b.left, b.right, column_stats_map),
                    else => self.estimateBinaryOp(b, column_stats_map),
                };
            },
            .unary_op => |u| {
                if (u.op == .not) {
                    return self.estimateNot(u.operand, column_stats_map);
                }
                return DEFAULT_SELECTIVITY;
            },
            .is_null => |isn| self.estimateIsNullNode(isn, column_stats_map),
            .in_list => |inl| self.estimateInList(inl, column_stats_map),
            .like => |l| self.estimateLike(l),
            else => DEFAULT_SELECTIVITY,
        };
    }

    // ── Binary Operators ────────────────────────────────────────────────

    fn estimateBinaryOp(
        self: *SelectivityEstimator,
        binop: anytype,
        column_stats_map: ?std.StringHashMap(ColumnStats),
    ) f64 {
        return switch (binop.op) {
            .equal => self.estimateEquality(binop.left, binop.right, column_stats_map),
            .not_equal => 1.0 - self.estimateEquality(binop.left, binop.right, column_stats_map),
            .less_than, .less_than_or_equal, .greater_than, .greater_than_or_equal => {
                return self.estimateRange(binop.left, binop.op, binop.right, column_stats_map);
            },
            else => DEFAULT_SELECTIVITY,
        };
    }

    fn estimateEquality(
        self: *SelectivityEstimator,
        left: *const ast.Expr,
        right: *const ast.Expr,
        column_stats_map: ?std.StringHashMap(ColumnStats),
    ) f64 {
        _ = self;

        // Try to identify the column reference
        const col_name = getColumnName(left) orelse getColumnName(right) orelse {
            return DEFAULT_EQUALITY_SELECTIVITY;
        };

        // We have a column and likely a literal value
        // For now, we don't analyze the literal value itself

        // Look up column statistics
        if (column_stats_map) |stats_map| {
            if (stats_map.get(col_name)) |col_stats| {
                // If we have distinct count, use 1 / distinct_count
                if (col_stats.distinct_count > 0) {
                    const selectivity = 1.0 / @as(f64, @floatFromInt(col_stats.distinct_count));
                    return @max(selectivity, MIN_SELECTIVITY);
                }
            }
        }

        return DEFAULT_EQUALITY_SELECTIVITY;
    }

    fn estimateRange(
        self: *SelectivityEstimator,
        left: *const ast.Expr,
        op: ast.BinaryOp,
        right: *const ast.Expr,
        column_stats_map: ?std.StringHashMap(ColumnStats),
    ) f64 {
        _ = self;
        _ = left;
        _ = op;
        _ = right;
        _ = column_stats_map;

        // For now, use a simple heuristic:
        // <, <=, >, >= each select approximately 1/3 of rows
        return DEFAULT_RANGE_SELECTIVITY;
    }

    // ── IS NULL / IS NOT NULL ───────────────────────────────────────────

    fn estimateIsNullNode(
        self: *SelectivityEstimator,
        is_null_node: anytype,
        column_stats_map: ?std.StringHashMap(ColumnStats),
    ) f64 {
        _ = self;

        const col_name = getColumnName(is_null_node.expr) orelse {
            return if (is_null_node.negated) 0.9 else 0.1;
        };

        if (column_stats_map) |stats_map| {
            if (stats_map.get(col_name)) |col_stats| {
                const null_sel = col_stats.null_fraction;
                return if (is_null_node.negated) (1.0 - null_sel) else null_sel;
            }
        }

        // Default: assume 10% NULL values
        return if (is_null_node.negated) 0.9 else 0.1;
    }

    // ── IN List ─────────────────────────────────────────────────────────

    fn estimateInList(
        self: *SelectivityEstimator,
        in_list: anytype,
        column_stats_map: ?std.StringHashMap(ColumnStats),
    ) f64 {
        // IN (v1, v2, ..., vN) ≈ P(col = v1) + P(col = v2) + ... + P(col = vN)
        // Approximate as: N * P(col = v) assuming values are distinct
        const n_values = @as(f64, @floatFromInt(in_list.list.len));
        const single_eq_selectivity = self.estimateEquality(
            in_list.expr,
            &ast.Expr{ .integer_literal = 0 }, // Dummy literal
            column_stats_map,
        );

        const base_sel = n_values * single_eq_selectivity;
        const result = @min(base_sel, 1.0);

        // Handle negation (NOT IN)
        return if (in_list.negated) (1.0 - result) else result;
    }

    // ── LIKE ────────────────────────────────────────────────────────────

    fn estimateLike(self: *SelectivityEstimator, like: anytype) f64 {
        _ = self;

        // Analyze the pattern to determine selectivity
        const base_sel = blk: {
            // Try to extract the pattern string
            const pattern_str = switch (like.pattern.*) {
                .string_literal => |s| s,
                else => break :blk LIKE_PREFIX_SELECTIVITY, // Unknown pattern type
            };

            // Analyze pattern structure
            if (pattern_str.len == 0) {
                break :blk MIN_SELECTIVITY; // Empty pattern matches nothing
            }

            const starts_with_wildcard = pattern_str[0] == '%';
            const ends_with_wildcard = pattern_str[pattern_str.len - 1] == '%';

            if (!starts_with_wildcard and ends_with_wildcard) {
                // 'prefix%' → most selective (10%)
                break :blk LIKE_PREFIX_SELECTIVITY;
            } else if (starts_with_wildcard and !ends_with_wildcard) {
                // '%suffix' → less selective (20%)
                break :blk LIKE_SUBSTRING_SELECTIVITY;
            } else if (starts_with_wildcard and ends_with_wildcard) {
                // '%substring%' → least selective (20%)
                break :blk LIKE_SUBSTRING_SELECTIVITY;
            } else {
                // Exact match (no wildcards) → very selective (1%)
                break :blk DEFAULT_EQUALITY_SELECTIVITY;
            }
        };

        // Handle negation (NOT LIKE)
        return if (like.negated) (1.0 - base_sel) else base_sel;
    }

    // ── Logical Combinators ─────────────────────────────────────────────

    fn estimateAnd(
        self: *SelectivityEstimator,
        left_expr: *const ast.Expr,
        right_expr: *const ast.Expr,
        column_stats_map: ?std.StringHashMap(ColumnStats),
    ) f64 {
        // AND combines selectivities by multiplication (assuming independence)
        const left_sel = self.estimateExpr(left_expr, column_stats_map);
        const right_sel = self.estimateExpr(right_expr, column_stats_map);
        return left_sel * right_sel;
    }

    fn estimateOr(
        self: *SelectivityEstimator,
        left_expr: *const ast.Expr,
        right_expr: *const ast.Expr,
        column_stats_map: ?std.StringHashMap(ColumnStats),
    ) f64 {
        // OR combines selectivities using: P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
        // Approximation: P(A ∪ B) ≈ P(A) + P(B) - P(A) * P(B)
        const left_sel = self.estimateExpr(left_expr, column_stats_map);
        const right_sel = self.estimateExpr(right_expr, column_stats_map);
        return left_sel + right_sel - (left_sel * right_sel);
    }

    fn estimateNot(
        self: *SelectivityEstimator,
        inner_expr: *const ast.Expr,
        column_stats_map: ?std.StringHashMap(ColumnStats),
    ) f64 {
        // NOT negates the selectivity
        const inner_sel = self.estimateExpr(inner_expr, column_stats_map);
        return 1.0 - inner_sel;
    }
};

// ── Helper Functions ────────────────────────────────────────────────────

/// Extract column name from an expression (if it's a column reference).
fn getColumnName(expr: *const ast.Expr) ?[]const u8 {
    return switch (expr.*) {
        .column_ref => |c| c.name,
        else => null,
    };
}


// ── Tests ───────────────────────────────────────────────────────────────

test "SelectivityEstimator: default selectivity" {
    const allocator = std.testing.allocator;
    var estimator = SelectivityEstimator.init(allocator);

    // Unknown expression type → default selectivity
    const expr = ast.Expr{ .integer_literal = 42 };
    const sel = estimator.estimatePredicate(&expr, null, null);
    try std.testing.expectEqual(DEFAULT_SELECTIVITY, sel);
}

test "SelectivityEstimator: equality without statistics" {
    const allocator = std.testing.allocator;
    var estimator = SelectivityEstimator.init(allocator);

    // col = 5 without statistics → default equality selectivity
    var col = ast.Expr{ .column_ref = .{ .name = "age" } };
    var val = ast.Expr{ .integer_literal = 5 };
    var expr = ast.Expr{ .binary_op = .{
        .op = .equal,
        .left = &col,
        .right = &val,
    } };

    const sel = estimator.estimatePredicate(&expr, null, null);
    try std.testing.expectEqual(DEFAULT_EQUALITY_SELECTIVITY, sel);
}

test "SelectivityEstimator: equality with distinct count" {
    const allocator = std.testing.allocator;
    var estimator = SelectivityEstimator.init(allocator);

    // Create column statistics with distinct_count = 100
    var stats_map = std.StringHashMap(ColumnStats).init(allocator);
    defer stats_map.deinit();

    const col_stats = ColumnStats{
        .distinct_count = 100,
        .null_fraction = 0.0,
        .avg_width = 4.0,
        .correlation = 0.0,
        .most_common_values = &[_]stats_mod.MostCommonValue{},
        .histogram_buckets = &[_]stats_mod.HistogramBucket{},
    };
    try stats_map.put("age", col_stats);

    const col = ast.Expr{ .column_ref = .{ .name = "age" } };
    const val = ast.Expr{ .integer_literal = 25 };
    const expr = ast.Expr{ .binary_op = .{
        .left = &col,
        .op = .equal,
        .right = &val,
    } };

    const sel = estimator.estimatePredicate(&expr, null, stats_map);
    // 1 / 100 = 0.01
    try std.testing.expectApproxEqRel(0.01, sel, 0.001);
}

test "SelectivityEstimator: IS NULL with null_fraction" {
    const allocator = std.testing.allocator;
    var estimator = SelectivityEstimator.init(allocator);

    var stats_map = std.StringHashMap(ColumnStats).init(allocator);
    defer stats_map.deinit();

    const col_stats = ColumnStats{
        .distinct_count = 100,
        .null_fraction = 0.15, // 15% NULL
        .avg_width = 4.0,
        .correlation = 0.0,
        .most_common_values = &[_]stats_mod.MostCommonValue{},
        .histogram_buckets = &[_]stats_mod.HistogramBucket{},
    };
    try stats_map.put("email", col_stats);

    var col = ast.Expr{ .column_ref = .{ .name = "email" } };
    var expr = ast.Expr{ .is_null = .{ .expr = &col, .negated = false } };

    const sel = estimator.estimatePredicate(&expr, null, stats_map);
    try std.testing.expectEqual(0.15, sel);
}

test "SelectivityEstimator: IS NOT NULL with null_fraction" {
    const allocator = std.testing.allocator;
    var estimator = SelectivityEstimator.init(allocator);

    var stats_map = std.StringHashMap(ColumnStats).init(allocator);
    defer stats_map.deinit();

    const col_stats = ColumnStats{
        .distinct_count = 100,
        .null_fraction = 0.15,
        .avg_width = 4.0,
        .correlation = 0.0,
        .most_common_values = &[_]stats_mod.MostCommonValue{},
        .histogram_buckets = &[_]stats_mod.HistogramBucket{},
    };
    try stats_map.put("email", col_stats);

    var col = ast.Expr{ .column_ref = .{ .name = "email" } };
    var expr = ast.Expr{ .is_null = .{ .expr = &col, .negated = true } };

    const sel = estimator.estimatePredicate(&expr, null, stats_map);
    try std.testing.expectEqual(0.85, sel); // 1.0 - 0.15
}

test "SelectivityEstimator: AND combines selectivities" {
    const allocator = std.testing.allocator;
    var estimator = SelectivityEstimator.init(allocator);

    // age = 25 AND status = 'active'
    // Each has 0.01 selectivity → combined = 0.01 * 0.01 = 0.0001
    var col1 = ast.Expr{ .column_ref = .{ .name = "age" } };
    var val1 = ast.Expr{ .integer_literal = 25 };
    var eq1 = ast.Expr{ .binary_op = .{
        .op = .equal,
        .left = &col1,
        .right = &val1,
    } };

    var col2 = ast.Expr{ .column_ref = .{ .name = "status" } };
    var val2 = ast.Expr{ .string_literal = "active" };
    var eq2 = ast.Expr{ .binary_op = .{
        .op = .equal,
        .left = &col2,
        .right = &val2,
    } };

    var and_expr = ast.Expr{ .binary_op = .{
        .op = .@"and",
        .left = &eq1,
        .right = &eq2,
    } };

    const sel = estimator.estimatePredicate(&and_expr, null, null);
    // 0.01 * 0.01 = 0.0001
    try std.testing.expectApproxEqRel(0.0001, sel, 0.00001);
}

test "SelectivityEstimator: OR combines selectivities" {
    const allocator = std.testing.allocator;
    var estimator = SelectivityEstimator.init(allocator);

    // age = 25 OR age = 30
    // Each has 0.01 selectivity → combined ≈ 0.01 + 0.01 - 0.01*0.01 = 0.0199
    var col1 = ast.Expr{ .column_ref = .{ .name = "age" } };
    var val1 = ast.Expr{ .integer_literal = 25 };
    var eq1 = ast.Expr{ .binary_op = .{
        .op = .equal,
        .left = &col1,
        .right = &val1,
    } };

    var col2 = ast.Expr{ .column_ref = .{ .name = "age" } };
    var val2 = ast.Expr{ .integer_literal = 30 };
    var eq2 = ast.Expr{ .binary_op = .{
        .op = .equal,
        .left = &col2,
        .right = &val2,
    } };

    var or_expr = ast.Expr{ .binary_op = .{
        .op = .@"or",
        .left = &eq1,
        .right = &eq2,
    } };

    const sel = estimator.estimatePredicate(&or_expr, null, null);
    // 0.01 + 0.01 - 0.01*0.01 = 0.0199
    try std.testing.expectApproxEqRel(0.0199, sel, 0.0001);
}

test "SelectivityEstimator: NOT negates selectivity" {
    const allocator = std.testing.allocator;
    var estimator = SelectivityEstimator.init(allocator);

    var col = ast.Expr{ .column_ref = .{ .name = "email" } };
    var is_null = ast.Expr{ .is_null = .{ .expr = &col, .negated = false } };
    var not_null = ast.Expr{ .unary_op = .{ .op = .not, .operand = &is_null } };

    const sel = estimator.estimatePredicate(&not_null, null, null);
    // NOT (IS NULL with default 0.1) = 1.0 - 0.1 = 0.9
    try std.testing.expectEqual(0.9, sel);
}

test "SelectivityEstimator: IN list selectivity" {
    const allocator = std.testing.allocator;
    var estimator = SelectivityEstimator.init(allocator);

    // status IN ('active', 'pending', 'review')
    // 3 values * 0.01 = 0.03
    var col = ast.Expr{ .column_ref = .{ .name = "status" } };
    var v1 = ast.Expr{ .string_literal = "active" };
    var v2 = ast.Expr{ .string_literal = "pending" };
    var v3 = ast.Expr{ .string_literal = "review" };
    const values = [_]*const ast.Expr{ &v1, &v2, &v3 };

    var in_expr = ast.Expr{ .in_list = .{ .expr = &col, .list = &values, .negated = false } };

    const sel = estimator.estimatePredicate(&in_expr, null, null);
    try std.testing.expectApproxEqRel(0.03, sel, 0.001);
}

test "SelectivityEstimator: range selectivity" {
    const allocator = std.testing.allocator;
    var estimator = SelectivityEstimator.init(allocator);

    // age > 25
    var col = ast.Expr{ .column_ref = .{ .name = "age" } };
    var val = ast.Expr{ .integer_literal = 25 };
    var expr = ast.Expr{ .binary_op = .{
        .op = .greater_than,
        .left = &col,
        .right = &val,
    } };

    const sel = estimator.estimatePredicate(&expr, null, null);
    try std.testing.expectEqual(DEFAULT_RANGE_SELECTIVITY, sel);
}

test "SelectivityEstimator: LIKE selectivity" {
    const allocator = std.testing.allocator;
    var estimator = SelectivityEstimator.init(allocator);

    // name LIKE 'A%' (prefix match)
    var col = ast.Expr{ .column_ref = .{ .name = "name" } };
    var pattern = ast.Expr{ .string_literal = "A%" };
    var like_expr = ast.Expr{ .like = .{ .expr = &col, .pattern = &pattern, .negated = false } };

    const sel = estimator.estimatePredicate(&like_expr, null, null);
    try std.testing.expectEqual(LIKE_PREFIX_SELECTIVITY, sel);
}

test "SelectivityEstimator: LIKE suffix pattern" {
    const allocator = std.testing.allocator;
    var estimator = SelectivityEstimator.init(allocator);

    // name LIKE '%son' (suffix match)
    var col = ast.Expr{ .column_ref = .{ .name = "name" } };
    var pattern = ast.Expr{ .string_literal = "%son" };
    var like_expr = ast.Expr{ .like = .{ .expr = &col, .pattern = &pattern, .negated = false } };

    const sel = estimator.estimatePredicate(&like_expr, null, null);
    try std.testing.expectEqual(LIKE_SUBSTRING_SELECTIVITY, sel);
}

test "SelectivityEstimator: LIKE substring pattern" {
    const allocator = std.testing.allocator;
    var estimator = SelectivityEstimator.init(allocator);

    // name LIKE '%john%' (substring match)
    var col = ast.Expr{ .column_ref = .{ .name = "name" } };
    var pattern = ast.Expr{ .string_literal = "%john%" };
    var like_expr = ast.Expr{ .like = .{ .expr = &col, .pattern = &pattern, .negated = false } };

    const sel = estimator.estimatePredicate(&like_expr, null, null);
    try std.testing.expectEqual(LIKE_SUBSTRING_SELECTIVITY, sel);
}

test "SelectivityEstimator: LIKE exact pattern" {
    const allocator = std.testing.allocator;
    var estimator = SelectivityEstimator.init(allocator);

    // name LIKE 'John' (exact match, no wildcards)
    var col = ast.Expr{ .column_ref = .{ .name = "name" } };
    var pattern = ast.Expr{ .string_literal = "John" };
    var like_expr = ast.Expr{ .like = .{ .expr = &col, .pattern = &pattern, .negated = false } };

    const sel = estimator.estimatePredicate(&like_expr, null, null);
    try std.testing.expectEqual(DEFAULT_EQUALITY_SELECTIVITY, sel);
}

test "SelectivityEstimator: NOT LIKE pattern" {
    const allocator = std.testing.allocator;
    var estimator = SelectivityEstimator.init(allocator);

    // name NOT LIKE 'A%' (negated prefix match)
    var col = ast.Expr{ .column_ref = .{ .name = "name" } };
    var pattern = ast.Expr{ .string_literal = "A%" };
    var like_expr = ast.Expr{ .like = .{ .expr = &col, .pattern = &pattern, .negated = true } };

    const sel = estimator.estimatePredicate(&like_expr, null, null);
    try std.testing.expectApproxEqRel(1.0 - LIKE_PREFIX_SELECTIVITY, sel, 0.0001);
}

// ── Edge Case Tests ─────────────────────────────────────────────────────

test "SelectivityEstimator: NOT EQUAL selectivity" {
    const allocator = std.testing.allocator;
    var estimator = SelectivityEstimator.init(allocator);

    // col != 5 → should be 1.0 - equality_selectivity
    var col = ast.Expr{ .column_ref = .{ .name = "status" } };
    var val = ast.Expr{ .integer_literal = 5 };
    var expr = ast.Expr{ .binary_op = .{
        .op = .not_equal,
        .left = &col,
        .right = &val,
    } };

    const sel = estimator.estimatePredicate(&expr, null, null);
    const expected = 1.0 - DEFAULT_EQUALITY_SELECTIVITY;
    try std.testing.expectEqual(expected, sel);
}

test "SelectivityEstimator: less_than_or_equal operator" {
    const allocator = std.testing.allocator;
    var estimator = SelectivityEstimator.init(allocator);

    // age <= 30
    var col = ast.Expr{ .column_ref = .{ .name = "age" } };
    var val = ast.Expr{ .integer_literal = 30 };
    var expr = ast.Expr{ .binary_op = .{
        .op = .less_than_or_equal,
        .left = &col,
        .right = &val,
    } };

    const sel = estimator.estimatePredicate(&expr, null, null);
    try std.testing.expectEqual(DEFAULT_RANGE_SELECTIVITY, sel);
}

test "SelectivityEstimator: greater_than_or_equal operator" {
    const allocator = std.testing.allocator;
    var estimator = SelectivityEstimator.init(allocator);

    // salary >= 50000
    var col = ast.Expr{ .column_ref = .{ .name = "salary" } };
    var val = ast.Expr{ .integer_literal = 50000 };
    var expr = ast.Expr{ .binary_op = .{
        .op = .greater_than_or_equal,
        .left = &col,
        .right = &val,
    } };

    const sel = estimator.estimatePredicate(&expr, null, null);
    try std.testing.expectEqual(DEFAULT_RANGE_SELECTIVITY, sel);
}

test "SelectivityEstimator: less_than operator" {
    const allocator = std.testing.allocator;
    var estimator = SelectivityEstimator.init(allocator);

    // price < 100
    var col = ast.Expr{ .column_ref = .{ .name = "price" } };
    var val = ast.Expr{ .integer_literal = 100 };
    var expr = ast.Expr{ .binary_op = .{
        .op = .less_than,
        .left = &col,
        .right = &val,
    } };

    const sel = estimator.estimatePredicate(&expr, null, null);
    try std.testing.expectEqual(DEFAULT_RANGE_SELECTIVITY, sel);
}

test "SelectivityEstimator: nested AND expressions" {
    const allocator = std.testing.allocator;
    var estimator = SelectivityEstimator.init(allocator);

    // (age > 25 AND salary > 50000) AND department = 'eng'
    var col1 = ast.Expr{ .column_ref = .{ .name = "age" } };
    var val1 = ast.Expr{ .integer_literal = 25 };
    var expr1 = ast.Expr{ .binary_op = .{
        .op = .greater_than,
        .left = &col1,
        .right = &val1,
    } };

    var col2 = ast.Expr{ .column_ref = .{ .name = "salary" } };
    var val2 = ast.Expr{ .integer_literal = 50000 };
    var expr2 = ast.Expr{ .binary_op = .{
        .op = .greater_than,
        .left = &col2,
        .right = &val2,
    } };

    var inner_and = ast.Expr{ .binary_op = .{
        .op = .@"and",
        .left = &expr1,
        .right = &expr2,
    } };

    var col3 = ast.Expr{ .column_ref = .{ .name = "department" } };
    var val3 = ast.Expr{ .string_literal = "eng" };
    var expr3 = ast.Expr{ .binary_op = .{
        .op = .equal,
        .left = &col3,
        .right = &val3,
    } };

    var outer_and = ast.Expr{ .binary_op = .{
        .op = .@"and",
        .left = &inner_and,
        .right = &expr3,
    } };

    const sel = estimator.estimatePredicate(&outer_and, null, null);
    // Expected: DEFAULT_RANGE_SELECTIVITY * DEFAULT_RANGE_SELECTIVITY * DEFAULT_EQUALITY_SELECTIVITY
    const expected = DEFAULT_RANGE_SELECTIVITY * DEFAULT_RANGE_SELECTIVITY * DEFAULT_EQUALITY_SELECTIVITY;
    try std.testing.expectEqual(expected, sel);
}

test "SelectivityEstimator: nested OR expressions" {
    const allocator = std.testing.allocator;
    var estimator = SelectivityEstimator.init(allocator);

    // (status = 1 OR status = 2) OR status = 3
    var col1 = ast.Expr{ .column_ref = .{ .name = "status" } };
    var val1 = ast.Expr{ .integer_literal = 1 };
    var expr1 = ast.Expr{ .binary_op = .{
        .op = .equal,
        .left = &col1,
        .right = &val1,
    } };

    var col2 = ast.Expr{ .column_ref = .{ .name = "status" } };
    var val2 = ast.Expr{ .integer_literal = 2 };
    var expr2 = ast.Expr{ .binary_op = .{
        .op = .equal,
        .left = &col2,
        .right = &val2,
    } };

    var inner_or = ast.Expr{ .binary_op = .{
        .op = .@"or",
        .left = &expr1,
        .right = &expr2,
    } };

    var col3 = ast.Expr{ .column_ref = .{ .name = "status" } };
    var val3 = ast.Expr{ .integer_literal = 3 };
    var expr3 = ast.Expr{ .binary_op = .{
        .op = .equal,
        .left = &col3,
        .right = &val3,
    } };

    var outer_or = ast.Expr{ .binary_op = .{
        .op = .@"or",
        .left = &inner_or,
        .right = &expr3,
    } };

    const sel = estimator.estimatePredicate(&outer_or, null, null);
    // Expected: ((p1 + p2 - p1*p2) + p3 - (p1 + p2 - p1*p2)*p3)
    const p = DEFAULT_EQUALITY_SELECTIVITY;
    const inner = p + p - (p * p);
    const expected = inner + p - (inner * p);
    try std.testing.expectApproxEqAbs(expected, sel, 0.0001);
}

test "SelectivityEstimator: AND with zero selectivity" {
    const allocator = std.testing.allocator;
    var estimator = SelectivityEstimator.init(allocator);

    // col = 5 AND <unknown> → should multiply selectivities
    var col = ast.Expr{ .column_ref = .{ .name = "id" } };
    var val = ast.Expr{ .integer_literal = 5 };
    var expr1 = ast.Expr{ .binary_op = .{
        .op = .equal,
        .left = &col,
        .right = &val,
    } };

    var unknown = ast.Expr{ .integer_literal = 42 }; // default selectivity

    var and_expr = ast.Expr{ .binary_op = .{
        .op = .@"and",
        .left = &expr1,
        .right = &unknown,
    } };

    const sel = estimator.estimatePredicate(&and_expr, null, null);
    const expected = DEFAULT_EQUALITY_SELECTIVITY * DEFAULT_SELECTIVITY;
    try std.testing.expectEqual(expected, sel);
}

test "SelectivityEstimator: OR with high selectivity" {
    const allocator = std.testing.allocator;
    var estimator = SelectivityEstimator.init(allocator);

    // IS NOT NULL OR IS NOT NULL → should cap near 1.0
    var col1 = ast.Expr{ .column_ref = .{ .name = "col1" } };
    var isn1 = ast.Expr{ .is_null = .{ .expr = &col1, .negated = true } };

    var col2 = ast.Expr{ .column_ref = .{ .name = "col2" } };
    var isn2 = ast.Expr{ .is_null = .{ .expr = &col2, .negated = true } };

    var or_expr = ast.Expr{ .binary_op = .{
        .op = .@"or",
        .left = &isn1,
        .right = &isn2,
    } };

    const sel = estimator.estimatePredicate(&or_expr, null, null);
    // Both IS NOT NULL default to 0.9, OR combines them
    // P(A OR B) = P(A) + P(B) - P(A)*P(B) = 0.9 + 0.9 - 0.81 = 0.99
    const expected = 0.9 + 0.9 - (0.9 * 0.9);
    try std.testing.expectApproxEqAbs(expected, sel, 0.0001);
}

test "SelectivityEstimator: double NOT negation" {
    const allocator = std.testing.allocator;
    var estimator = SelectivityEstimator.init(allocator);

    // NOT (NOT (col = 5)) → should return back to equality selectivity
    var col = ast.Expr{ .column_ref = .{ .name = "id" } };
    var val = ast.Expr{ .integer_literal = 5 };
    var eq_expr = ast.Expr{ .binary_op = .{
        .op = .equal,
        .left = &col,
        .right = &val,
    } };

    var inner_not = ast.Expr{ .unary_op = .{
        .op = .not,
        .operand = &eq_expr,
    } };

    var outer_not = ast.Expr{ .unary_op = .{
        .op = .not,
        .operand = &inner_not,
    } };

    const sel = estimator.estimatePredicate(&outer_not, null, null);
    // Use approx comparison due to floating point precision (1.0 - (1.0 - 0.01))
    try std.testing.expectApproxEqAbs(DEFAULT_EQUALITY_SELECTIVITY, sel, 0.0001);
}

test "SelectivityEstimator: IN list with single value" {
    const allocator = std.testing.allocator;
    var estimator = SelectivityEstimator.init(allocator);

    // status IN (1) → should equal single equality
    var col = ast.Expr{ .column_ref = .{ .name = "status" } };
    var val = ast.Expr{ .integer_literal = 1 };
    const values = [_]*const ast.Expr{&val};
    var in_expr = ast.Expr{ .in_list = .{
        .expr = &col,
        .list = &values,
        .negated = false,
    } };

    const sel = estimator.estimatePredicate(&in_expr, null, null);
    try std.testing.expectEqual(DEFAULT_EQUALITY_SELECTIVITY, sel);
}

test "SelectivityEstimator: IS NULL with zero null_fraction" {
    const allocator = std.testing.allocator;
    var estimator = SelectivityEstimator.init(allocator);

    // Create column statistics with null_fraction = 0.0 (no nulls)
    var stats_map = std.StringHashMap(ColumnStats).init(allocator);
    defer stats_map.deinit();

    var col_stats = ColumnStats{
        .distinct_count = 1000,
        .null_fraction = 0.0,
        .avg_width = 8,
        .correlation = 0.0,
        .most_common_values = &[_]stats_mod.MostCommonValue{},
        .histogram_buckets = &[_]stats_mod.HistogramBucket{},
    };
    defer col_stats.deinit(allocator);

    try stats_map.put("id", col_stats);

    // id IS NULL
    var col = ast.Expr{ .column_ref = .{ .name = "id" } };
    var isn = ast.Expr{ .is_null = .{ .expr = &col, .negated = false } };

    const sel = estimator.estimatePredicate(&isn, null, stats_map);
    try std.testing.expectEqual(0.0, sel);
}

test "SelectivityEstimator: IS NOT NULL with 100% null_fraction" {
    const allocator = std.testing.allocator;
    var estimator = SelectivityEstimator.init(allocator);

    // Create column statistics with null_fraction = 1.0 (all nulls)
    var stats_map = std.StringHashMap(ColumnStats).init(allocator);
    defer stats_map.deinit();

    var col_stats = ColumnStats{
        .distinct_count = 0,
        .null_fraction = 1.0,
        .avg_width = 0,
        .correlation = 0.0,
        .most_common_values = &[_]stats_mod.MostCommonValue{},
        .histogram_buckets = &[_]stats_mod.HistogramBucket{},
    };
    defer col_stats.deinit(allocator);

    try stats_map.put("optional_field", col_stats);

    // optional_field IS NOT NULL
    var col = ast.Expr{ .column_ref = .{ .name = "optional_field" } };
    var isn = ast.Expr{ .is_null = .{ .expr = &col, .negated = true } };

    const sel = estimator.estimatePredicate(&isn, null, stats_map);
    try std.testing.expectEqual(0.0, sel);
}

test "SelectivityEstimator: equality with distinct_count = 1" {
    const allocator = std.testing.allocator;
    var estimator = SelectivityEstimator.init(allocator);

    // Create column statistics with distinct_count = 1 (single unique value)
    var stats_map = std.StringHashMap(ColumnStats).init(allocator);
    defer stats_map.deinit();

    var col_stats = ColumnStats{
        .distinct_count = 1,
        .null_fraction = 0.0,
        .avg_width = 4,
        .correlation = 0.0,
        .most_common_values = &[_]stats_mod.MostCommonValue{},
        .histogram_buckets = &[_]stats_mod.HistogramBucket{},
    };
    defer col_stats.deinit(allocator);

    try stats_map.put("constant_col", col_stats);

    // constant_col = 42 → selectivity should be 1/1 = 1.0
    var col = ast.Expr{ .column_ref = .{ .name = "constant_col" } };
    var val = ast.Expr{ .integer_literal = 42 };
    var expr = ast.Expr{ .binary_op = .{
        .op = .equal,
        .left = &col,
        .right = &val,
    } };

    const sel = estimator.estimatePredicate(&expr, null, stats_map);
    try std.testing.expectEqual(1.0, sel);
}
