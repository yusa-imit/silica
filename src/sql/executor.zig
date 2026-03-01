//! Volcano-model query executor — iterator-based, pull execution engine.
//!
//! Each operator implements `open()`, `next()`, `close()` and produces
//! Row tuples that flow up the operator tree. The executor translates
//! optimized LogicalPlan nodes into a tree of iterators.
//!
//! Supported operators:
//!   Scan     — full table scan via B+Tree cursor
//!   Filter   — predicate evaluation (WHERE)
//!   Project  — column selection / expression computation
//!   Sort     — in-memory ORDER BY
//!   Limit    — row count restriction with OFFSET
//!   Aggregate — GROUP BY with aggregate functions
//!   NestedLoopJoin — nested loop join (all join types)
//!   Values   — literal row set (INSERT VALUES)

const std = @import("std");
const Allocator = std.mem.Allocator;
const ast = @import("ast.zig");
const catalog_mod = @import("catalog.zig");
const planner_mod = @import("planner.zig");
const btree_mod = @import("../storage/btree.zig");
const buffer_pool_mod = @import("../storage/buffer_pool.zig");
const page_mod = @import("../storage/page.zig");

const mvcc_mod = @import("../tx/mvcc.zig");

const BTree = btree_mod.BTree;
const Cursor = btree_mod.Cursor;
const BufferPool = buffer_pool_mod.BufferPool;
const Pager = page_mod.Pager;
const Catalog = catalog_mod.Catalog;
const ColumnInfo = catalog_mod.ColumnInfo;
const ColumnType = catalog_mod.ColumnType;
const TableInfo = catalog_mod.TableInfo;
const PlanNode = planner_mod.PlanNode;
const LogicalPlan = planner_mod.LogicalPlan;
const PlanType = planner_mod.PlanType;
const AggFunc = planner_mod.AggFunc;
const TupleHeader = mvcc_mod.TupleHeader;
const Snapshot = mvcc_mod.Snapshot;

// ── MVCC Context ──────────────────────────────────────────────────────────

/// MVCC context passed to scan operators for visibility filtering.
/// When null/disabled, all rows are visible (legacy/auto-commit mode).
pub const MvccContext = struct {
    snapshot: Snapshot,
    current_xid: u32,
    current_cid: u16,
    /// When true, rows carry MVCC headers and need visibility checks.
    enabled: bool = true,
    /// Optional reference to TransactionManager for commit/abort status lookup.
    /// Enables correct visibility for tuples without hint flags (e.g., aborted txns).
    tm: ?*mvcc_mod.TransactionManager = null,
};

// ── Value Type ──────────────────────────────────────────────────────────

/// A runtime value in the executor.
pub const Value = union(enum) {
    integer: i64,
    real: f64,
    text: []const u8,
    blob: []const u8,
    boolean: bool,
    null_value,

    /// Compare two values. Returns .lt, .eq, or .gt.
    /// NULLs sort last (greater than any non-null value).
    pub fn compare(a: Value, b: Value) std.math.Order {
        // NULL handling: NULL > everything
        if (a == .null_value and b == .null_value) return .eq;
        if (a == .null_value) return .gt;
        if (b == .null_value) return .lt;

        // Same type comparison
        return switch (a) {
            .integer => |av| switch (b) {
                .integer => |bv| std.math.order(av, bv),
                .real => |bv| std.math.order(@as(f64, @floatFromInt(av)), bv),
                else => .lt, // integers < text/blob/bool
            },
            .real => |av| switch (b) {
                .integer => |bv| std.math.order(av, @as(f64, @floatFromInt(bv))),
                .real => |bv| std.math.order(av, bv),
                else => .lt,
            },
            .text => |av| switch (b) {
                .text => |bv| std.mem.order(u8, av, bv),
                else => .gt, // text > numbers
            },
            .blob => |av| switch (b) {
                .blob => |bv| std.mem.order(u8, av, bv),
                else => .gt,
            },
            .boolean => |av| switch (b) {
                .boolean => |bv| {
                    if (av == bv) return .eq;
                    if (!av and bv) return .lt;
                    return .gt;
                },
                else => .gt,
            },
            .null_value => unreachable,
        };
    }

    /// Check if two values are equal.
    pub fn eql(a: Value, b: Value) bool {
        return a.compare(b) == .eq;
    }

    /// Check if value is truthy (for boolean evaluation).
    pub fn isTruthy(self: Value) bool {
        return switch (self) {
            .integer => |v| v != 0,
            .real => |v| v != 0.0,
            .text => |v| v.len > 0,
            .blob => |v| v.len > 0,
            .boolean => |v| v,
            .null_value => false,
        };
    }

    /// Convert to integer if possible.
    pub fn toInteger(self: Value) ?i64 {
        return switch (self) {
            .integer => |v| v,
            .real => |v| @intFromFloat(v),
            .boolean => |v| if (v) @as(i64, 1) else 0,
            .text => |v| std.fmt.parseInt(i64, v, 10) catch null,
            else => null,
        };
    }

    /// Convert to float if possible.
    pub fn toReal(self: Value) ?f64 {
        return switch (self) {
            .integer => |v| @floatFromInt(v),
            .real => |v| v,
            .boolean => |v| if (v) @as(f64, 1.0) else 0.0,
            .text => |v| std.fmt.parseFloat(f64, v) catch null,
            else => null,
        };
    }

    /// Duplicate a value, allocating copies of text/blob data.
    pub fn dupe(self: Value, allocator: Allocator) !Value {
        return switch (self) {
            .text => |v| .{ .text = try allocator.dupe(u8, v) },
            .blob => |v| .{ .blob = try allocator.dupe(u8, v) },
            else => self,
        };
    }

    /// Free heap-allocated value data.
    pub fn free(self: Value, allocator: Allocator) void {
        switch (self) {
            .text => |v| allocator.free(v),
            .blob => |v| allocator.free(v),
            else => {},
        }
    }
};

// ── Row ─────────────────────────────────────────────────────────────────

/// A row is a sequence of named column values.
pub const Row = struct {
    /// Column names (not owned — references schema or plan).
    columns: []const []const u8,
    /// Column values (owned by this row).
    values: []Value,
    allocator: Allocator,

    pub fn deinit(self: *Row) void {
        for (self.values) |v| v.free(self.allocator);
        self.allocator.free(self.values);
        self.allocator.free(self.columns);
    }

    /// Look up a column value by name (case-insensitive).
    pub fn getColumn(self: *const Row, name: []const u8) ?Value {
        for (self.columns, 0..) |col, i| {
            if (std.ascii.eqlIgnoreCase(col, name)) return self.values[i];
        }
        return null;
    }

    /// Look up a column value by qualified name (table.column).
    pub fn getQualifiedColumn(self: *const Row, table: []const u8, column: []const u8) ?Value {
        // Try "table.column" format first
        for (self.columns, 0..) |col, i| {
            // Check if column name is "table.column"
            if (std.mem.indexOf(u8, col, ".")) |dot_pos| {
                const col_table = col[0..dot_pos];
                const col_name = col[dot_pos + 1 ..];
                if (std.ascii.eqlIgnoreCase(col_table, table) and std.ascii.eqlIgnoreCase(col_name, column)) {
                    return self.values[i];
                }
            }
        }
        // Fall back to unqualified lookup
        return self.getColumn(column);
    }

    /// Create a deep copy of this row.
    pub fn clone(self: *const Row, allocator: Allocator) !Row {
        const cols = try allocator.alloc([]const u8, self.columns.len);
        errdefer allocator.free(cols);
        for (self.columns, 0..) |c, i| {
            cols[i] = c;
        }

        const vals = try allocator.alloc(Value, self.values.len);
        errdefer allocator.free(vals);
        var inited: usize = 0;
        errdefer for (vals[0..inited]) |v| v.free(allocator);
        for (self.values, 0..) |v, i| {
            vals[i] = try v.dupe(allocator);
            inited += 1;
        }

        return .{
            .columns = cols,
            .values = vals,
            .allocator = allocator,
        };
    }
};

// ── Row Serialization ───────────────────────────────────────────────────

/// Serialize a row's values into bytes for B+Tree storage.
/// Format: [col_count: u16] for each col: [type_tag: u8][value_data...]
///   integer: 8 bytes (i64 little-endian)
///   real: 8 bytes (f64 little-endian)
///   text: [len: u32][bytes...]
///   blob: [len: u32][bytes...]
///   boolean: 1 byte (0 or 1)
///   null: 0 bytes (tag only)
pub fn serializeRow(allocator: Allocator, values: []const Value) ![]u8 {
    var size: usize = 2; // col_count
    for (values) |v| {
        size += 1; // type tag
        switch (v) {
            .integer => size += 8,
            .real => size += 8,
            .text => |t| size += 4 + t.len,
            .blob => |b| size += 4 + b.len,
            .boolean => size += 1,
            .null_value => {},
        }
    }

    const buf = try allocator.alloc(u8, size);
    errdefer allocator.free(buf);
    var pos: usize = 0;

    std.mem.writeInt(u16, buf[pos..][0..2], @intCast(values.len), .little);
    pos += 2;

    for (values) |v| {
        switch (v) {
            .integer => |i| {
                buf[pos] = 0x01;
                pos += 1;
                std.mem.writeInt(i64, buf[pos..][0..8], i, .little);
                pos += 8;
            },
            .real => |r| {
                buf[pos] = 0x02;
                pos += 1;
                std.mem.writeInt(u64, buf[pos..][0..8], @bitCast(r), .little);
                pos += 8;
            },
            .text => |t| {
                buf[pos] = 0x03;
                pos += 1;
                std.mem.writeInt(u32, buf[pos..][0..4], @intCast(t.len), .little);
                pos += 4;
                @memcpy(buf[pos..][0..t.len], t);
                pos += t.len;
            },
            .blob => |b| {
                buf[pos] = 0x04;
                pos += 1;
                std.mem.writeInt(u32, buf[pos..][0..4], @intCast(b.len), .little);
                pos += 4;
                @memcpy(buf[pos..][0..b.len], b);
                pos += b.len;
            },
            .boolean => |b| {
                buf[pos] = 0x05;
                pos += 1;
                buf[pos] = if (b) 1 else 0;
                pos += 1;
            },
            .null_value => {
                buf[pos] = 0x00;
                pos += 1;
            },
        }
    }

    std.debug.assert(pos == size);
    return buf;
}

/// Deserialize a row's values from B+Tree storage bytes.
pub fn deserializeRow(allocator: Allocator, data: []const u8) ![]Value {
    if (data.len < 2) return error.InvalidRowData;
    var pos: usize = 0;

    const col_count = std.mem.readInt(u16, data[pos..][0..2], .little);
    pos += 2;

    const values = try allocator.alloc(Value, col_count);
    var inited: usize = 0;
    errdefer {
        for (values[0..inited]) |v| v.free(allocator);
        allocator.free(values);
    }

    for (values) |*v| {
        if (pos >= data.len) return error.InvalidRowData;
        const tag = data[pos];
        pos += 1;

        switch (tag) {
            0x01 => { // integer
                if (pos + 8 > data.len) return error.InvalidRowData;
                v.* = .{ .integer = std.mem.readInt(i64, data[pos..][0..8], .little) };
                pos += 8;
            },
            0x02 => { // real
                if (pos + 8 > data.len) return error.InvalidRowData;
                v.* = .{ .real = @bitCast(std.mem.readInt(u64, data[pos..][0..8], .little)) };
                pos += 8;
            },
            0x03 => { // text
                if (pos + 4 > data.len) return error.InvalidRowData;
                const len = std.mem.readInt(u32, data[pos..][0..4], .little);
                pos += 4;
                if (pos + len > data.len) return error.InvalidRowData;
                v.* = .{ .text = try allocator.dupe(u8, data[pos..][0..len]) };
                pos += len;
            },
            0x04 => { // blob
                if (pos + 4 > data.len) return error.InvalidRowData;
                const len = std.mem.readInt(u32, data[pos..][0..4], .little);
                pos += 4;
                if (pos + len > data.len) return error.InvalidRowData;
                v.* = .{ .blob = try allocator.dupe(u8, data[pos..][0..len]) };
                pos += len;
            },
            0x05 => { // boolean
                if (pos >= data.len) return error.InvalidRowData;
                v.* = .{ .boolean = data[pos] != 0 };
                pos += 1;
            },
            0x00 => { // null
                v.* = .null_value;
            },
            else => return error.InvalidRowData,
        }
        inited += 1;
    }

    return values;
}

// ── Expression Evaluator ────────────────────────────────────────────────

pub const EvalError = error{
    OutOfMemory,
    TypeError,
    DivisionByZero,
    ColumnNotFound,
    UnsupportedExpression,
};

/// Evaluate an AST expression against a row, producing a Value.
pub fn evalExpr(allocator: Allocator, expr: *const ast.Expr, row: *const Row) EvalError!Value {
    switch (expr.*) {
        .integer_literal => |v| return .{ .integer = v },
        .float_literal => |v| return .{ .real = v },
        .string_literal => |v| return .{ .text = try allocator.dupe(u8, v) },
        .boolean_literal => |v| return .{ .boolean = v },
        .null_literal => return .null_value,

        .column_ref => |ref| {
            if (ref.prefix) |table| {
                return (row.getQualifiedColumn(table, ref.name) orelse
                    return EvalError.ColumnNotFound).dupe(allocator) catch return EvalError.OutOfMemory;
            }
            return (row.getColumn(ref.name) orelse
                return EvalError.ColumnNotFound).dupe(allocator) catch return EvalError.OutOfMemory;
        },

        .paren => |inner| return evalExpr(allocator, inner, row),

        .unary_op => |op| {
            const operand = try evalExpr(allocator, op.operand, row);
            defer operand.free(allocator);
            return evalUnaryOp(op.op, operand);
        },

        .binary_op => |op| {
            const left = try evalExpr(allocator, op.left, row);
            defer left.free(allocator);
            const right = try evalExpr(allocator, op.right, row);
            defer right.free(allocator);
            return evalBinaryOp(allocator, op.op, left, right);
        },

        .is_null => |is| {
            const val = try evalExpr(allocator, is.expr, row);
            defer val.free(allocator);
            const result = val == .null_value;
            return .{ .boolean = if (is.negated) !result else result };
        },

        .between => |bt| {
            const val = try evalExpr(allocator, bt.expr, row);
            defer val.free(allocator);
            const low = try evalExpr(allocator, bt.low, row);
            defer low.free(allocator);
            const high = try evalExpr(allocator, bt.high, row);
            defer high.free(allocator);
            const in_range = val.compare(low) != .lt and val.compare(high) != .gt;
            return .{ .boolean = if (bt.negated) !in_range else in_range };
        },

        .in_list => |il| {
            const val = try evalExpr(allocator, il.expr, row);
            defer val.free(allocator);
            var found = false;
            for (il.list) |item| {
                const item_val = try evalExpr(allocator, item, row);
                defer item_val.free(allocator);
                if (val.eql(item_val)) {
                    found = true;
                    break;
                }
            }
            return .{ .boolean = if (il.negated) !found else found };
        },

        .like => |lk| {
            const val = try evalExpr(allocator, lk.expr, row);
            defer val.free(allocator);
            const pat = try evalExpr(allocator, lk.pattern, row);
            defer pat.free(allocator);
            const text_val = switch (val) {
                .text => |t| t,
                else => return .{ .boolean = false },
            };
            const pattern = switch (pat) {
                .text => |t| t,
                else => return .{ .boolean = false },
            };
            const matches = likeMatch(text_val, pattern);
            return .{ .boolean = if (lk.negated) !matches else matches };
        },

        .case_expr => |ce| {
            if (ce.operand) |operand| {
                const op_val = try evalExpr(allocator, operand, row);
                defer op_val.free(allocator);
                for (ce.when_clauses) |wc| {
                    const when_val = try evalExpr(allocator, wc.condition, row);
                    defer when_val.free(allocator);
                    if (op_val.eql(when_val)) {
                        return evalExpr(allocator, wc.result, row);
                    }
                }
            } else {
                for (ce.when_clauses) |wc| {
                    const cond = try evalExpr(allocator, wc.condition, row);
                    defer cond.free(allocator);
                    if (cond.isTruthy()) {
                        return evalExpr(allocator, wc.result, row);
                    }
                }
            }
            if (ce.else_expr) |else_e| {
                return evalExpr(allocator, else_e, row);
            }
            return .null_value;
        },

        .cast => |c| {
            const val = try evalExpr(allocator, c.expr, row);
            defer val.free(allocator);
            return evalCast(allocator, val, c.target_type);
        },

        .function_call => |fc| {
            return evalFunctionCall(allocator, fc, row);
        },

        // Unsupported in row-level evaluation (aggregates handled in AggregateExecutor)
        .blob_literal,
        .subquery,
        .bind_parameter,
        => return EvalError.UnsupportedExpression,
    }
}

fn evalUnaryOp(op: ast.UnaryOp, operand: Value) Value {
    return switch (op) {
        .negate => switch (operand) {
            .integer => |v| .{ .integer = -v },
            .real => |v| .{ .real = -v },
            else => .null_value,
        },
        .not => .{ .boolean = !operand.isTruthy() },
        .bitwise_not => switch (operand) {
            .integer => |v| .{ .integer = ~v },
            else => .null_value,
        },
    };
}

fn evalBinaryOp(allocator: Allocator, op: ast.BinaryOp, left: Value, right: Value) EvalError!Value {
    // NULL propagation for most ops
    if (left == .null_value or right == .null_value) {
        return switch (op) {
            .@"and" => blk: {
                // FALSE AND NULL = FALSE
                if (left == .boolean and !left.boolean) break :blk Value{ .boolean = false };
                if (right == .boolean and !right.boolean) break :blk Value{ .boolean = false };
                break :blk Value.null_value;
            },
            .@"or" => blk: {
                // TRUE OR NULL = TRUE
                if (left == .boolean and left.boolean) break :blk Value{ .boolean = true };
                if (right == .boolean and right.boolean) break :blk Value{ .boolean = true };
                break :blk Value.null_value;
            },
            else => .null_value,
        };
    }

    return switch (op) {
        // Arithmetic
        .add => evalArithmetic(left, right, .add),
        .subtract => evalArithmetic(left, right, .sub),
        .multiply => evalArithmetic(left, right, .mul),
        .divide => {
            if (right == .integer and right.integer == 0) return EvalError.DivisionByZero;
            if (right == .real and right.real == 0.0) return EvalError.DivisionByZero;
            return evalArithmetic(left, right, .div);
        },
        .modulo => {
            if (right == .integer and right.integer == 0) return EvalError.DivisionByZero;
            return switch (left) {
                .integer => |a| switch (right) {
                    .integer => |b| Value{ .integer = @mod(a, b) },
                    else => .null_value,
                },
                else => .null_value,
            };
        },

        // Comparison
        .equal => .{ .boolean = left.eql(right) },
        .not_equal => .{ .boolean = !left.eql(right) },
        .less_than => .{ .boolean = left.compare(right) == .lt },
        .greater_than => .{ .boolean = left.compare(right) == .gt },
        .less_than_or_equal => .{ .boolean = left.compare(right) != .gt },
        .greater_than_or_equal => .{ .boolean = left.compare(right) != .lt },

        // Logical
        .@"and" => .{ .boolean = left.isTruthy() and right.isTruthy() },
        .@"or" => .{ .boolean = left.isTruthy() or right.isTruthy() },

        // String concatenation
        .concat => blk: {
            const l = switch (left) {
                .text => |t| t,
                else => break :blk Value.null_value,
            };
            const r = switch (right) {
                .text => |t| t,
                else => break :blk Value.null_value,
            };
            const result = allocator.alloc(u8, l.len + r.len) catch return EvalError.OutOfMemory;
            @memcpy(result[0..l.len], l);
            @memcpy(result[l.len..], r);
            break :blk Value{ .text = result };
        },

        // Bitwise
        .bitwise_and => switch (left) {
            .integer => |a| switch (right) {
                .integer => |b| Value{ .integer = a & b },
                else => .null_value,
            },
            else => .null_value,
        },
        .bitwise_or => switch (left) {
            .integer => |a| switch (right) {
                .integer => |b| Value{ .integer = a | b },
                else => .null_value,
            },
            else => .null_value,
        },
        .left_shift => switch (left) {
            .integer => |a| switch (right) {
                .integer => |b| Value{ .integer = if (b >= 0 and b < 64) a << @intCast(b) else 0 },
                else => .null_value,
            },
            else => .null_value,
        },
        .right_shift => switch (left) {
            .integer => |a| switch (right) {
                .integer => |b| Value{ .integer = if (b >= 0 and b < 64) a >> @intCast(b) else 0 },
                else => .null_value,
            },
            else => .null_value,
        },
    };
}

const ArithOp = enum { add, sub, mul, div };

fn evalArithmetic(left: Value, right: Value, op: ArithOp) Value {
    // Try integer arithmetic first
    if (left == .integer and right == .integer) {
        const a = left.integer;
        const b = right.integer;
        return .{ .integer = switch (op) {
            .add => a +% b,
            .sub => a -% b,
            .mul => a *% b,
            .div => @divTrunc(a, b),
        } };
    }

    // Fall back to float
    const a = left.toReal() orelse return .null_value;
    const b = right.toReal() orelse return .null_value;
    return .{ .real = switch (op) {
        .add => a + b,
        .sub => a - b,
        .mul => a * b,
        .div => a / b,
    } };
}

fn evalCast(allocator: Allocator, val: Value, target: ast.DataType) EvalError!Value {
    return switch (target) {
        .type_integer, .type_int => .{ .integer = val.toInteger() orelse return .null_value },
        .type_real => .{ .real = val.toReal() orelse return .null_value },
        .type_text, .type_varchar => blk: {
            const s = switch (val) {
                .integer => |v| std.fmt.allocPrint(allocator, "{d}", .{v}) catch return EvalError.OutOfMemory,
                .real => |v| std.fmt.allocPrint(allocator, "{d}", .{v}) catch return EvalError.OutOfMemory,
                .boolean => |v| (allocator.dupe(u8, if (v) "true" else "false") catch return EvalError.OutOfMemory),
                .text => |v| (allocator.dupe(u8, v) catch return EvalError.OutOfMemory),
                .null_value => return .null_value,
                .blob => return .null_value,
            };
            break :blk Value{ .text = s };
        },
        .type_boolean => .{ .boolean = val.isTruthy() },
        .type_blob => .null_value,
    };
}

fn evalFunctionCall(allocator: Allocator, fc: anytype, row: *const Row) EvalError!Value {
    // Aggregate functions: look up result by column name from the aggregate output row.
    // When an Aggregate operator has already computed COUNT/SUM/etc., the result is
    // stored as a named column. The Project operator just needs to retrieve it.
    if (isAggregateFuncName(fc.name)) {
        // Build the expected column name for this aggregate
        const col_name = aggResultColName(fc);
        for (row.columns, 0..) |c, i| {
            if (std.ascii.eqlIgnoreCase(c, col_name)) {
                return row.values[i].dupe(allocator) catch return EvalError.OutOfMemory;
            }
        }
        // If not found by exact name, the aggregate wasn't computed — fall through
        return EvalError.UnsupportedExpression;
    }

    // Built-in scalar functions
    if (std.ascii.eqlIgnoreCase(fc.name, "abs")) {
        if (fc.args.len != 1) return EvalError.TypeError;
        const arg = try evalExpr(allocator, fc.args[0], row);
        defer arg.free(allocator);
        return switch (arg) {
            .integer => |v| Value{ .integer = if (v < 0) -v else v },
            .real => |v| Value{ .real = @abs(v) },
            else => .null_value,
        };
    }
    if (std.ascii.eqlIgnoreCase(fc.name, "length")) {
        if (fc.args.len != 1) return EvalError.TypeError;
        const arg = try evalExpr(allocator, fc.args[0], row);
        defer arg.free(allocator);
        return switch (arg) {
            .text => |v| Value{ .integer = @intCast(v.len) },
            .blob => |v| Value{ .integer = @intCast(v.len) },
            else => .null_value,
        };
    }
    if (std.ascii.eqlIgnoreCase(fc.name, "upper")) {
        if (fc.args.len != 1) return EvalError.TypeError;
        const arg = try evalExpr(allocator, fc.args[0], row);
        defer arg.free(allocator);
        return switch (arg) {
            .text => |v| blk: {
                const upper = allocator.alloc(u8, v.len) catch return EvalError.OutOfMemory;
                for (v, 0..) |c, i| upper[i] = std.ascii.toUpper(c);
                break :blk Value{ .text = upper };
            },
            else => .null_value,
        };
    }
    if (std.ascii.eqlIgnoreCase(fc.name, "lower")) {
        if (fc.args.len != 1) return EvalError.TypeError;
        const arg = try evalExpr(allocator, fc.args[0], row);
        defer arg.free(allocator);
        return switch (arg) {
            .text => |v| blk: {
                const lower = allocator.alloc(u8, v.len) catch return EvalError.OutOfMemory;
                for (v, 0..) |c, i| lower[i] = std.ascii.toLower(c);
                break :blk Value{ .text = lower };
            },
            else => .null_value,
        };
    }
    if (std.ascii.eqlIgnoreCase(fc.name, "coalesce")) {
        for (fc.args) |arg| {
            const val = try evalExpr(allocator, arg, row);
            if (val != .null_value) return val;
            val.free(allocator);
        }
        return .null_value;
    }
    if (std.ascii.eqlIgnoreCase(fc.name, "typeof")) {
        if (fc.args.len != 1) return EvalError.TypeError;
        const arg = try evalExpr(allocator, fc.args[0], row);
        defer arg.free(allocator);
        const type_name: []const u8 = switch (arg) {
            .integer => "integer",
            .real => "real",
            .text => "text",
            .blob => "blob",
            .boolean => "boolean",
            .null_value => "null",
        };
        return Value{ .text = allocator.dupe(u8, type_name) catch return EvalError.OutOfMemory };
    }
    return EvalError.UnsupportedExpression;
}

/// Check if a function name is an aggregate function.
fn isAggregateFuncName(name: []const u8) bool {
    const agg_names = [_][]const u8{ "count", "sum", "avg", "min", "max" };
    for (agg_names) |n| {
        if (std.ascii.eqlIgnoreCase(name, n)) return true;
    }
    return false;
}

/// Build the column name that the AggregateOp would use for this function call.
fn aggResultColName(fc: anytype) []const u8 {
    // COUNT(*) → "count(*)", COUNT(x) → "count", SUM(x) → "sum", etc.
    if (std.ascii.eqlIgnoreCase(fc.name, "count")) {
        if (fc.args.len > 0 and fc.args[0].* == .column_ref and
            std.mem.eql(u8, fc.args[0].column_ref.name, "*"))
        {
            return "count(*)";
        }
        return "count";
    }
    if (std.ascii.eqlIgnoreCase(fc.name, "sum")) return "sum";
    if (std.ascii.eqlIgnoreCase(fc.name, "avg")) return "avg";
    if (std.ascii.eqlIgnoreCase(fc.name, "min")) return "min";
    if (std.ascii.eqlIgnoreCase(fc.name, "max")) return "max";
    return fc.name;
}

/// SQL LIKE pattern matching (% = any string, _ = any char).
fn likeMatch(text: []const u8, pattern: []const u8) bool {
    var ti: usize = 0;
    var pi: usize = 0;
    var star_pi: ?usize = null;
    var star_ti: usize = 0;

    while (ti < text.len or pi < pattern.len) {
        if (pi < pattern.len) {
            if (pattern[pi] == '%') {
                star_pi = pi;
                star_ti = ti;
                pi += 1;
                continue;
            }
            if (ti < text.len) {
                if (pattern[pi] == '_' or std.ascii.toLower(pattern[pi]) == std.ascii.toLower(text[ti])) {
                    ti += 1;
                    pi += 1;
                    continue;
                }
            }
        }
        if (star_pi) |sp| {
            pi = sp + 1;
            star_ti += 1;
            ti = star_ti;
            if (ti > text.len) return false;
        } else {
            return false;
        }
    }
    return true;
}

// ── Executor Interface ──────────────────────────────────────────────────

/// Error type for executor operations.
pub const ExecError = error{
    OutOfMemory,
    TypeError,
    DivisionByZero,
    ColumnNotFound,
    UnsupportedExpression,
    TableNotFound,
    InvalidRowData,
    StorageError,
    ExecutionError,
};

/// The Volcano-model iterator interface.
/// Each operator produces rows one at a time via next().
pub const RowIterator = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        next: *const fn (ptr: *anyopaque) ExecError!?Row,
        close: *const fn (ptr: *anyopaque) void,
    };

    pub fn next(self: RowIterator) ExecError!?Row {
        return self.vtable.next(self.ptr);
    }

    pub fn close(self: RowIterator) void {
        self.vtable.close(self.ptr);
    }
};

// ── Scan Operator ───────────────────────────────────────────────────────

/// Full table scan via B+Tree cursor.
pub const ScanOp = struct {
    allocator: Allocator,
    tree: BTree,
    cursor: ?Cursor = null,
    col_names: []const []const u8,
    opened: bool = false,
    /// MVCC context for visibility filtering (null = all rows visible).
    mvcc_ctx: ?MvccContext = null,

    /// Create a ScanOp. After placement on the heap, call initCursor() to
    /// set up the cursor with a stable pointer to the tree.
    pub fn init(allocator: Allocator, pool: *BufferPool, data_root_page_id: u32, col_names: []const []const u8) ScanOp {
        return .{
            .allocator = allocator,
            .tree = BTree.init(pool, data_root_page_id),
            .col_names = col_names,
        };
    }

    /// Must be called after the ScanOp is at its final heap location.
    pub fn initCursor(self: *ScanOp) void {
        self.cursor = Cursor.init(self.allocator, &self.tree);
    }

    pub fn open(self: *ScanOp) ExecError!void {
        if (self.cursor == null) self.initCursor();
        self.cursor.?.seekFirst() catch return ExecError.StorageError;
        self.opened = true;
    }

    pub fn next(self: *ScanOp) ExecError!?Row {
        if (!self.opened) try self.open();

        while (true) {
            const entry = self.cursor.?.next() catch return ExecError.StorageError;
            if (entry == null) return null;

            defer self.allocator.free(entry.?.key);

            // MVCC visibility check: deserialize header and filter invisible tuples
            if (self.mvcc_ctx) |ctx| {
                if (ctx.enabled and mvcc_mod.isVersionedRow(entry.?.value)) {
                    const header = TupleHeader.deserialize(entry.?.value[1..][0..mvcc_mod.TUPLE_HEADER_SIZE]);
                    if (!mvcc_mod.isTupleVisibleWithTm(header, ctx.snapshot, ctx.current_xid, ctx.current_cid, ctx.tm)) {
                        // Tuple not visible — skip it
                        self.allocator.free(entry.?.value);
                        continue;
                    }
                    // Visible: deserialize column data (skip MVCC header)
                    const values = deserializeRow(self.allocator, entry.?.value[mvcc_mod.MVCC_ROW_OVERHEAD..]) catch {
                        self.allocator.free(entry.?.value);
                        return ExecError.InvalidRowData;
                    };
                    self.allocator.free(entry.?.value);
                    errdefer {
                        for (values) |v| v.free(self.allocator);
                        self.allocator.free(values);
                    }

                    const cols = self.allocator.alloc([]const u8, self.col_names.len) catch return ExecError.OutOfMemory;
                    for (self.col_names, 0..) |c, i| cols[i] = c;
                    return Row{ .columns = cols, .values = values, .allocator = self.allocator };
                }
            }

            defer self.allocator.free(entry.?.value);

            // No MVCC context or legacy row: return all rows.
            // Still need to detect MVCC format and skip the overhead for committed rows.
            const row_bytes = if (mvcc_mod.isVersionedRow(entry.?.value))
                entry.?.value[mvcc_mod.MVCC_ROW_OVERHEAD..]
            else
                entry.?.value;
            const values = deserializeRow(self.allocator, row_bytes) catch return ExecError.InvalidRowData;
            errdefer {
                for (values) |v| v.free(self.allocator);
                self.allocator.free(values);
            }

            // Build column names
            const cols = self.allocator.alloc([]const u8, self.col_names.len) catch return ExecError.OutOfMemory;
            for (self.col_names, 0..) |c, i| cols[i] = c;

            return Row{
                .columns = cols,
                .values = values,
                .allocator = self.allocator,
            };
        }
    }

    pub fn close(self: *ScanOp) void {
        if (self.cursor) |*c| c.deinit();
    }

    pub fn iterator(self: *ScanOp) RowIterator {
        return .{
            .ptr = self,
            .vtable = &.{
                .next = @ptrCast(&ScanOp.next),
                .close = @ptrCast(&ScanOp.close),
            },
        };
    }
};

// ── Index Scan Operator ─────────────────────────────────────────────────

/// Index-based point lookup: uses a secondary index B+Tree to find
/// matching row keys, then fetches full rows from the data B+Tree.
/// Returns at most one row for an equality lookup.
pub const IndexScanOp = struct {
    allocator: Allocator,
    pool: *BufferPool,
    data_root_page_id: u32,
    index_root_page_id: u32,
    lookup_key: []const u8,
    col_names: []const []const u8,
    exhausted: bool = false,
    /// MVCC context for visibility filtering (null = all rows visible).
    mvcc_ctx: ?MvccContext = null,

    pub fn init(
        allocator: Allocator,
        pool: *BufferPool,
        data_root_page_id: u32,
        index_root_page_id: u32,
        lookup_key: []const u8,
        col_names: []const []const u8,
    ) IndexScanOp {
        return .{
            .allocator = allocator,
            .pool = pool,
            .data_root_page_id = data_root_page_id,
            .index_root_page_id = index_root_page_id,
            .lookup_key = lookup_key,
            .col_names = col_names,
        };
    }

    pub fn next(self: *IndexScanOp) ExecError!?Row {
        if (self.exhausted) return null;
        self.exhausted = true;

        // Look up the index to find the row key
        var idx_tree = BTree.init(self.pool, self.index_root_page_id);
        const row_key = idx_tree.get(self.allocator, self.lookup_key) catch return ExecError.StorageError;
        if (row_key == null) return null; // No matching index entry
        defer self.allocator.free(row_key.?);

        // Fetch the actual row from the data B+Tree
        var data_tree = BTree.init(self.pool, self.data_root_page_id);
        const row_data = data_tree.get(self.allocator, row_key.?) catch return ExecError.StorageError;
        if (row_data == null) return null; // Orphaned index entry
        defer self.allocator.free(row_data.?);

        // MVCC visibility check
        if (self.mvcc_ctx) |ctx| {
            if (ctx.enabled and mvcc_mod.isVersionedRow(row_data.?)) {
                const header = TupleHeader.deserialize(row_data.?[1..][0..mvcc_mod.TUPLE_HEADER_SIZE]);
                if (!mvcc_mod.isTupleVisible(header, ctx.snapshot, ctx.current_xid, ctx.current_cid)) {
                    return null; // Tuple not visible
                }
                // Visible: deserialize column data (skip MVCC header)
                const values = deserializeRow(self.allocator, row_data.?[mvcc_mod.MVCC_ROW_OVERHEAD..]) catch return ExecError.InvalidRowData;
                errdefer {
                    for (values) |v| v.free(self.allocator);
                    self.allocator.free(values);
                }
                const cols = self.allocator.alloc([]const u8, self.col_names.len) catch return ExecError.OutOfMemory;
                for (self.col_names, 0..) |c, i| cols[i] = c;
                return Row{ .columns = cols, .values = values, .allocator = self.allocator };
            }
        }

        // No MVCC context or legacy row — still detect MVCC format and skip overhead
        const row_bytes = if (mvcc_mod.isVersionedRow(row_data.?))
            row_data.?[mvcc_mod.MVCC_ROW_OVERHEAD..]
        else
            row_data.?;
        const values = deserializeRow(self.allocator, row_bytes) catch return ExecError.InvalidRowData;
        errdefer {
            for (values) |v| v.free(self.allocator);
            self.allocator.free(values);
        }

        const cols = self.allocator.alloc([]const u8, self.col_names.len) catch return ExecError.OutOfMemory;
        for (self.col_names, 0..) |c, i| cols[i] = c;

        return Row{
            .columns = cols,
            .values = values,
            .allocator = self.allocator,
        };
    }

    pub fn close(_: *IndexScanOp) void {}

    pub fn iterator(self: *IndexScanOp) RowIterator {
        return .{
            .ptr = self,
            .vtable = &.{
                .next = @ptrCast(&IndexScanOp.next),
                .close = @ptrCast(&IndexScanOp.close),
            },
        };
    }
};

// ── Filter Operator ─────────────────────────────────────────────────────

/// Applies a predicate to filter rows from its input.
pub const FilterOp = struct {
    allocator: Allocator,
    input: RowIterator,
    predicate: *const ast.Expr,

    pub fn init(allocator: Allocator, input: RowIterator, predicate: *const ast.Expr) FilterOp {
        return .{
            .allocator = allocator,
            .input = input,
            .predicate = predicate,
        };
    }

    pub fn next(self: *FilterOp) ExecError!?Row {
        while (true) {
            var row = try self.input.next() orelse return null;
            const val = evalExpr(self.allocator, self.predicate, &row) catch |err| {
                row.deinit();
                return switch (err) {
                    error.OutOfMemory => ExecError.OutOfMemory,
                    error.TypeError => ExecError.TypeError,
                    error.DivisionByZero => ExecError.DivisionByZero,
                    error.ColumnNotFound => ExecError.ColumnNotFound,
                    error.UnsupportedExpression => ExecError.UnsupportedExpression,
                };
            };
            defer val.free(self.allocator);

            if (val.isTruthy()) return row;
            row.deinit();
        }
    }

    pub fn close(self: *FilterOp) void {
        self.input.close();
    }

    pub fn iterator(self: *FilterOp) RowIterator {
        return .{
            .ptr = self,
            .vtable = &.{
                .next = @ptrCast(&FilterOp.next),
                .close = @ptrCast(&FilterOp.close),
            },
        };
    }
};

// ── Project Operator ────────────────────────────────────────────────────

/// Selects/computes output columns from each input row.
pub const ProjectOp = struct {
    allocator: Allocator,
    input: RowIterator,
    columns: []const PlanNode.ProjectColumn,

    pub fn init(allocator: Allocator, input: RowIterator, columns: []const PlanNode.ProjectColumn) ProjectOp {
        return .{
            .allocator = allocator,
            .input = input,
            .columns = columns,
        };
    }

    pub fn next(self: *ProjectOp) ExecError!?Row {
        var row = try self.input.next() orelse return null;
        defer row.deinit();

        const vals = self.allocator.alloc(Value, self.columns.len) catch return ExecError.OutOfMemory;
        var inited: usize = 0;
        errdefer {
            for (vals[0..inited]) |v| v.free(self.allocator);
            self.allocator.free(vals);
        }

        const col_names = self.allocator.alloc([]const u8, self.columns.len) catch return ExecError.OutOfMemory;
        errdefer self.allocator.free(col_names);

        for (self.columns, 0..) |col, i| {
            vals[i] = evalExpr(self.allocator, col.expr, &row) catch |err| {
                // When an aggregate function has an alias (e.g., SUM(x) AS total),
                // the AggregateOp stores the result under the alias name. Try looking
                // up by alias before reporting an error.
                if (err == error.UnsupportedExpression) {
                    if (col.alias) |alias| {
                        if (row.getColumn(alias)) |v| {
                            vals[i] = v.dupe(self.allocator) catch return ExecError.OutOfMemory;
                            inited += 1;
                            col_names[i] = col.alias orelse exprColumnName(col.expr);
                            continue;
                        }
                    }
                }
                return switch (err) {
                    error.OutOfMemory => ExecError.OutOfMemory,
                    error.TypeError => ExecError.TypeError,
                    error.DivisionByZero => ExecError.DivisionByZero,
                    error.ColumnNotFound => ExecError.ColumnNotFound,
                    error.UnsupportedExpression => ExecError.UnsupportedExpression,
                };
            };
            inited += 1;
            col_names[i] = col.alias orelse exprColumnName(col.expr);
        }

        return Row{
            .columns = col_names,
            .values = vals,
            .allocator = self.allocator,
        };
    }

    pub fn close(self: *ProjectOp) void {
        self.input.close();
    }

    pub fn iterator(self: *ProjectOp) RowIterator {
        return .{
            .ptr = self,
            .vtable = &.{
                .next = @ptrCast(&ProjectOp.next),
                .close = @ptrCast(&ProjectOp.close),
            },
        };
    }
};

/// Extract a display name from an expression (for unnamed columns).
fn exprColumnName(expr: *const ast.Expr) []const u8 {
    return switch (expr.*) {
        .column_ref => |ref| ref.name,
        .function_call => |fc| fc.name,
        .integer_literal => "?column?",
        .float_literal => "?column?",
        .string_literal => "?column?",
        else => "?column?",
    };
}

// ── Limit Operator ──────────────────────────────────────────────────────

/// Restricts output to a maximum number of rows, with optional offset.
pub const LimitOp = struct {
    allocator: Allocator,
    input: RowIterator,
    limit_count: ?u64,
    offset_count: u64,
    returned: u64 = 0,
    skipped: u64 = 0,

    pub fn init(allocator: Allocator, input: RowIterator, limit_count: ?u64, offset_count: u64) LimitOp {
        return .{
            .allocator = allocator,
            .input = input,
            .limit_count = limit_count,
            .offset_count = offset_count,
        };
    }

    pub fn next(self: *LimitOp) ExecError!?Row {
        // Skip offset rows
        while (self.skipped < self.offset_count) {
            var row = try self.input.next() orelse return null;
            row.deinit();
            self.skipped += 1;
        }

        // Check limit
        if (self.limit_count) |lim| {
            if (self.returned >= lim) return null;
        }

        const row = try self.input.next() orelse return null;
        self.returned += 1;
        return row;
    }

    pub fn close(self: *LimitOp) void {
        self.input.close();
    }

    pub fn iterator(self: *LimitOp) RowIterator {
        return .{
            .ptr = self,
            .vtable = &.{
                .next = @ptrCast(&LimitOp.next),
                .close = @ptrCast(&LimitOp.close),
            },
        };
    }
};

// ── Sort Operator ───────────────────────────────────────────────────────

/// In-memory sort. Materializes all input rows, sorts, then emits.
pub const SortOp = struct {
    allocator: Allocator,
    input: RowIterator,
    order_by: []const ast.OrderByItem,
    rows: std.ArrayListUnmanaged(Row) = .{},
    index: usize = 0,
    materialized: bool = false,

    pub fn init(allocator: Allocator, input: RowIterator, order_by: []const ast.OrderByItem) SortOp {
        return .{
            .allocator = allocator,
            .input = input,
            .order_by = order_by,
        };
    }

    fn materialize(self: *SortOp) ExecError!void {
        // Collect all rows
        while (true) {
            const row = try self.input.next() orelse break;
            self.rows.append(self.allocator, row) catch return ExecError.OutOfMemory;
        }

        // Sort using order_by expressions
        const ctx = SortContext{ .order_by = self.order_by, .allocator = self.allocator };
        std.sort.block(Row, self.rows.items, ctx, SortContext.lessThan);

        self.materialized = true;
    }

    pub fn next(self: *SortOp) ExecError!?Row {
        if (!self.materialized) try self.materialize();

        if (self.index >= self.rows.items.len) return null;
        const row = self.rows.items[self.index];
        self.index += 1;
        // Transfer ownership — don't deinit here
        return row;
    }

    pub fn close(self: *SortOp) void {
        // Free any remaining rows that haven't been consumed
        for (self.rows.items[self.index..]) |*row| row.deinit();
        self.rows.deinit(self.allocator);
        self.input.close();
    }

    pub fn iterator(self: *SortOp) RowIterator {
        return .{
            .ptr = self,
            .vtable = &.{
                .next = @ptrCast(&SortOp.next),
                .close = @ptrCast(&SortOp.close),
            },
        };
    }

    const SortContext = struct {
        order_by: []const ast.OrderByItem,
        allocator: Allocator,

        fn lessThan(ctx: SortContext, a: Row, b: Row) bool {
            for (ctx.order_by) |ob| {
                const av = evalExpr(ctx.allocator, ob.expr, &a) catch Value.null_value;
                defer av.free(ctx.allocator);
                const bv = evalExpr(ctx.allocator, ob.expr, &b) catch Value.null_value;
                defer bv.free(ctx.allocator);

                const order = av.compare(bv);
                if (order == .eq) continue;

                return switch (ob.direction) {
                    .asc => order == .lt,
                    .desc => order == .gt,
                };
            }
            return false;
        }
    };
};

// ── Aggregate Operator ──────────────────────────────────────────────────

/// GROUP BY with aggregate functions. Materializes all input, groups, then emits.
pub const AggregateOp = struct {
    allocator: Allocator,
    input: RowIterator,
    group_by: []const *const ast.Expr,
    aggregates: []const planner_mod.PlanNode.AggregateExpr,
    result_rows: std.ArrayListUnmanaged(Row) = .{},
    index: usize = 0,
    materialized: bool = false,

    pub fn init(
        allocator: Allocator,
        input: RowIterator,
        group_by: []const *const ast.Expr,
        aggregates: []const planner_mod.PlanNode.AggregateExpr,
    ) AggregateOp {
        return .{
            .allocator = allocator,
            .input = input,
            .group_by = group_by,
            .aggregates = aggregates,
        };
    }

    fn materialize(self: *AggregateOp) ExecError!void {
        // Collect all input rows
        var input_rows = std.ArrayListUnmanaged(Row){};
        defer {
            for (input_rows.items) |*r| r.deinit();
            input_rows.deinit(self.allocator);
        }

        while (true) {
            const row = try self.input.next() orelse break;
            input_rows.append(self.allocator, row) catch return ExecError.OutOfMemory;
        }

        if (input_rows.items.len == 0 and self.group_by.len == 0) {
            // No rows + no GROUP BY = produce one row with aggregate defaults
            try self.emitAggregateRow(&.{});
        } else if (self.group_by.len == 0) {
            // No GROUP BY = all rows in one group
            try self.emitAggregateRow(input_rows.items);
        } else {
            // Group rows by GROUP BY key
            // Simple approach: sort by group key, then scan for group boundaries
            const ctx = GroupContext{ .group_by = self.group_by, .allocator = self.allocator };
            std.sort.block(Row, input_rows.items, ctx, GroupContext.lessThan);

            var group_start: usize = 0;
            for (input_rows.items[1..], 1..) |_, i| {
                if (!groupKeysEqual(self.allocator, self.group_by, &input_rows.items[group_start], &input_rows.items[i])) {
                    try self.emitAggregateRow(input_rows.items[group_start..i]);
                    group_start = i;
                }
            }
            try self.emitAggregateRow(input_rows.items[group_start..]);
        }

        self.materialized = true;
    }

    fn emitAggregateRow(self: *AggregateOp, group: []const Row) ExecError!void {
        const total_cols = self.group_by.len + self.aggregates.len;
        const vals = self.allocator.alloc(Value, total_cols) catch return ExecError.OutOfMemory;
        var inited: usize = 0;
        errdefer {
            for (vals[0..inited]) |v| v.free(self.allocator);
            self.allocator.free(vals);
        }

        const col_names = self.allocator.alloc([]const u8, total_cols) catch return ExecError.OutOfMemory;
        errdefer self.allocator.free(col_names);

        // Group by columns from first row in group
        for (self.group_by, 0..) |gb_expr, i| {
            if (group.len > 0) {
                vals[i] = evalExpr(self.allocator, gb_expr, &group[0]) catch .null_value;
            } else {
                vals[i] = .null_value;
            }
            inited += 1;
            col_names[i] = exprColumnName(gb_expr);
        }

        // Aggregate columns
        for (self.aggregates, 0..) |agg, i| {
            const idx = self.group_by.len + i;
            vals[idx] = self.computeAggregate(agg, group);
            inited += 1;
            col_names[idx] = agg.alias orelse aggFuncName(agg.func);
        }

        self.result_rows.append(self.allocator, Row{
            .columns = col_names,
            .values = vals,
            .allocator = self.allocator,
        }) catch return ExecError.OutOfMemory;
    }

    fn computeAggregate(self: *AggregateOp, agg: planner_mod.PlanNode.AggregateExpr, group: []const Row) Value {
        switch (agg.func) {
            .count_star => return .{ .integer = @intCast(group.len) },
            .count => {
                var count: i64 = 0;
                for (group) |*row| {
                    if (agg.arg) |arg_expr| {
                        const val = evalExpr(self.allocator, arg_expr, row) catch continue;
                        defer val.free(self.allocator);
                        if (val != .null_value) count += 1;
                    }
                }
                return .{ .integer = count };
            },
            .sum => {
                var int_sum: i64 = 0;
                var float_sum: f64 = 0;
                var has_float = false;
                var has_value = false;
                for (group) |*row| {
                    if (agg.arg) |arg_expr| {
                        const val = evalExpr(self.allocator, arg_expr, row) catch continue;
                        defer val.free(self.allocator);
                        switch (val) {
                            .integer => |v| {
                                int_sum += v;
                                has_value = true;
                            },
                            .real => |v| {
                                float_sum += v;
                                has_float = true;
                                has_value = true;
                            },
                            else => {},
                        }
                    }
                }
                if (!has_value) return .null_value;
                if (has_float) return .{ .real = float_sum + @as(f64, @floatFromInt(int_sum)) };
                return .{ .integer = int_sum };
            },
            .avg => {
                var sum: f64 = 0;
                var count: f64 = 0;
                for (group) |*row| {
                    if (agg.arg) |arg_expr| {
                        const val = evalExpr(self.allocator, arg_expr, row) catch continue;
                        defer val.free(self.allocator);
                        if (val.toReal()) |v| {
                            sum += v;
                            count += 1;
                        }
                    }
                }
                if (count == 0) return .null_value;
                return .{ .real = sum / count };
            },
            .min => {
                var min_val: ?Value = null;
                for (group) |*row| {
                    if (agg.arg) |arg_expr| {
                        const val = evalExpr(self.allocator, arg_expr, row) catch continue;
                        if (val == .null_value) {
                            val.free(self.allocator);
                            continue;
                        }
                        if (min_val) |current| {
                            if (val.compare(current) == .lt) {
                                current.free(self.allocator);
                                min_val = val;
                            } else {
                                val.free(self.allocator);
                            }
                        } else {
                            min_val = val;
                        }
                    }
                }
                if (min_val) |v| {
                    defer {
                        var mv = v;
                        mv.free(self.allocator);
                    }
                    return v.dupe(self.allocator) catch .null_value;
                }
                return .null_value;
            },
            .max => {
                var max_val: ?Value = null;
                for (group) |*row| {
                    if (agg.arg) |arg_expr| {
                        const val = evalExpr(self.allocator, arg_expr, row) catch continue;
                        if (val == .null_value) {
                            val.free(self.allocator);
                            continue;
                        }
                        if (max_val) |current| {
                            if (val.compare(current) == .gt) {
                                current.free(self.allocator);
                                max_val = val;
                            } else {
                                val.free(self.allocator);
                            }
                        } else {
                            max_val = val;
                        }
                    }
                }
                if (max_val) |v| {
                    defer {
                        var mv = v;
                        mv.free(self.allocator);
                    }
                    return v.dupe(self.allocator) catch .null_value;
                }
                return .null_value;
            },
        }
    }

    pub fn next(self: *AggregateOp) ExecError!?Row {
        if (!self.materialized) try self.materialize();
        if (self.index >= self.result_rows.items.len) return null;
        const row = self.result_rows.items[self.index];
        self.index += 1;
        return row;
    }

    pub fn close(self: *AggregateOp) void {
        for (self.result_rows.items[self.index..]) |*r| r.deinit();
        self.result_rows.deinit(self.allocator);
        self.input.close();
    }

    pub fn iterator(self: *AggregateOp) RowIterator {
        return .{
            .ptr = self,
            .vtable = &.{
                .next = @ptrCast(&AggregateOp.next),
                .close = @ptrCast(&AggregateOp.close),
            },
        };
    }

    const GroupContext = struct {
        group_by: []const *const ast.Expr,
        allocator: Allocator,

        fn lessThan(ctx: GroupContext, a: Row, b: Row) bool {
            for (ctx.group_by) |expr| {
                const av = evalExpr(ctx.allocator, expr, &a) catch Value.null_value;
                defer av.free(ctx.allocator);
                const bv = evalExpr(ctx.allocator, expr, &b) catch Value.null_value;
                defer bv.free(ctx.allocator);
                const order = av.compare(bv);
                if (order == .eq) continue;
                return order == .lt;
            }
            return false;
        }
    };
};

fn groupKeysEqual(allocator: Allocator, group_by: []const *const ast.Expr, a: *const Row, b: *const Row) bool {
    for (group_by) |expr| {
        const av = evalExpr(allocator, expr, a) catch Value.null_value;
        defer av.free(allocator);
        const bv = evalExpr(allocator, expr, b) catch Value.null_value;
        defer bv.free(allocator);
        if (!av.eql(bv)) return false;
    }
    return true;
}

fn aggFuncName(func: AggFunc) []const u8 {
    return switch (func) {
        .count => "count",
        .sum => "sum",
        .avg => "avg",
        .min => "min",
        .max => "max",
        .count_star => "count(*)",
    };
}

// ── Nested Loop Join ────────────────────────────────────────────────────

/// Nested loop join — iterates all combinations of left and right rows.
pub const NestedLoopJoinOp = struct {
    allocator: Allocator,
    left: RowIterator,
    right: RowIterator,
    join_type: ast.JoinType,
    on_condition: ?*const ast.Expr,
    left_row: ?Row = null,
    right_rows: std.ArrayListUnmanaged(Row) = .{},
    right_index: usize = 0,
    right_materialized: bool = false,
    left_matched: bool = false,

    pub fn init(
        allocator: Allocator,
        left: RowIterator,
        right: RowIterator,
        join_type: ast.JoinType,
        on_condition: ?*const ast.Expr,
    ) NestedLoopJoinOp {
        return .{
            .allocator = allocator,
            .left = left,
            .right = right,
            .join_type = join_type,
            .on_condition = on_condition,
        };
    }

    fn materializeRight(self: *NestedLoopJoinOp) ExecError!void {
        while (true) {
            const row = try self.right.next() orelse break;
            self.right_rows.append(self.allocator, row) catch return ExecError.OutOfMemory;
        }
        self.right_materialized = true;
    }

    pub fn next(self: *NestedLoopJoinOp) ExecError!?Row {
        if (!self.right_materialized) try self.materializeRight();

        while (true) {
            // Get current left row
            if (self.left_row == null) {
                self.left_row = try self.left.next();
                if (self.left_row == null) return null;
                self.right_index = 0;
                self.left_matched = false;
            }

            // Try each right row
            while (self.right_index < self.right_rows.items.len) {
                const right_row = &self.right_rows.items[self.right_index];
                self.right_index += 1;

                // Build combined row
                var combined = try combineRows(self.allocator, &self.left_row.?, right_row);

                // Check join condition
                if (self.on_condition) |cond| {
                    const val = evalExpr(self.allocator, cond, &combined) catch {
                        combined.deinit();
                        continue;
                    };
                    defer val.free(self.allocator);
                    if (!val.isTruthy()) {
                        combined.deinit();
                        continue;
                    }
                }

                self.left_matched = true;
                return combined;
            }

            // Exhausted right side for current left row
            if (!self.left_matched and (self.join_type == .left or self.join_type == .full)) {
                // Emit left row with NULLs for right columns
                const result = try leftOuterRow(self.allocator, &self.left_row.?, self.right_rows.items);
                self.left_row.?.deinit();
                self.left_row = null;
                return result;
            }

            self.left_row.?.deinit();
            self.left_row = null;
        }
    }

    pub fn close(self: *NestedLoopJoinOp) void {
        if (self.left_row) |*lr| lr.deinit();
        for (self.right_rows.items) |*r| r.deinit();
        self.right_rows.deinit(self.allocator);
        self.left.close();
        self.right.close();
    }

    pub fn iterator(self: *NestedLoopJoinOp) RowIterator {
        return .{
            .ptr = self,
            .vtable = &.{
                .next = @ptrCast(&NestedLoopJoinOp.next),
                .close = @ptrCast(&NestedLoopJoinOp.close),
            },
        };
    }
};

fn combineRows(allocator: Allocator, left: *const Row, right: *const Row) ExecError!Row {
    const total = left.columns.len + right.columns.len;
    const cols = allocator.alloc([]const u8, total) catch return ExecError.OutOfMemory;
    errdefer allocator.free(cols);
    const vals = allocator.alloc(Value, total) catch return ExecError.OutOfMemory;
    var inited: usize = 0;
    errdefer {
        for (vals[0..inited]) |v| v.free(allocator);
        allocator.free(vals);
    }

    for (left.columns, 0..) |c, i| {
        cols[i] = c;
        vals[i] = left.values[i].dupe(allocator) catch return ExecError.OutOfMemory;
        inited += 1;
    }
    for (right.columns, 0..) |c, i| {
        cols[left.columns.len + i] = c;
        vals[left.columns.len + i] = right.values[i].dupe(allocator) catch return ExecError.OutOfMemory;
        inited += 1;
    }

    return Row{
        .columns = cols,
        .values = vals,
        .allocator = allocator,
    };
}

fn leftOuterRow(allocator: Allocator, left: *const Row, right_sample: []const Row) ExecError!Row {
    const right_cols = if (right_sample.len > 0) right_sample[0].columns.len else 0;
    const total = left.columns.len + right_cols;
    const cols = allocator.alloc([]const u8, total) catch return ExecError.OutOfMemory;
    errdefer allocator.free(cols);
    const vals = allocator.alloc(Value, total) catch return ExecError.OutOfMemory;
    var inited: usize = 0;
    errdefer {
        for (vals[0..inited]) |v| v.free(allocator);
        allocator.free(vals);
    }

    for (left.columns, 0..) |c, i| {
        cols[i] = c;
        vals[i] = left.values[i].dupe(allocator) catch return ExecError.OutOfMemory;
        inited += 1;
    }
    if (right_sample.len > 0) {
        for (right_sample[0].columns, 0..) |c, i| {
            cols[left.columns.len + i] = c;
            vals[left.columns.len + i] = .null_value;
            inited += 1;
        }
    }

    return Row{
        .columns = cols,
        .values = vals,
        .allocator = allocator,
    };
}

// ── Values Operator ─────────────────────────────────────────────────────

/// Produces literal rows from INSERT VALUES.
pub const ValuesOp = struct {
    allocator: Allocator,
    col_names: []const []const u8,
    rows: []const []const *const ast.Expr,
    index: usize = 0,

    pub fn init(allocator: Allocator, col_names: []const []const u8, rows: []const []const *const ast.Expr) ValuesOp {
        return .{
            .allocator = allocator,
            .col_names = col_names,
            .rows = rows,
        };
    }

    pub fn next(self: *ValuesOp) ExecError!?Row {
        if (self.index >= self.rows.len) return null;

        const exprs = self.rows[self.index];
        self.index += 1;

        // Create an empty row for evaluation context
        const empty_row = Row{
            .columns = &.{},
            .values = &.{},
            .allocator = self.allocator,
        };

        const vals = self.allocator.alloc(Value, exprs.len) catch return ExecError.OutOfMemory;
        var inited: usize = 0;
        errdefer {
            for (vals[0..inited]) |v| v.free(self.allocator);
            self.allocator.free(vals);
        }

        for (exprs, 0..) |expr, i| {
            vals[i] = evalExpr(self.allocator, expr, &empty_row) catch |err| {
                return switch (err) {
                    error.OutOfMemory => ExecError.OutOfMemory,
                    error.TypeError => ExecError.TypeError,
                    error.DivisionByZero => ExecError.DivisionByZero,
                    error.ColumnNotFound => ExecError.ColumnNotFound,
                    error.UnsupportedExpression => ExecError.UnsupportedExpression,
                };
            };
            inited += 1;
        }

        const cols = self.allocator.alloc([]const u8, self.col_names.len) catch return ExecError.OutOfMemory;
        for (self.col_names, 0..) |c, i| cols[i] = c;

        return Row{
            .columns = cols,
            .values = vals,
            .allocator = self.allocator,
        };
    }

    pub fn close(_: *ValuesOp) void {}

    pub fn iterator(self: *ValuesOp) RowIterator {
        return .{
            .ptr = self,
            .vtable = &.{
                .next = @ptrCast(&ValuesOp.next),
                .close = @ptrCast(&ValuesOp.close),
            },
        };
    }
};

// ── Empty Operator ──────────────────────────────────────────────────────

/// Produces no rows (used for DDL results).
pub const EmptyOp = struct {
    pub fn next(_: *EmptyOp) ExecError!?Row {
        return null;
    }

    pub fn close(_: *EmptyOp) void {}

    pub fn iterator(self: *EmptyOp) RowIterator {
        return .{
            .ptr = self,
            .vtable = &.{
                .next = @ptrCast(&EmptyOp.next),
                .close = @ptrCast(&EmptyOp.close),
            },
        };
    }
};

// ── Materialized Operator ────────────────────────────────────────────────

/// Serves pre-materialized rows (used for view expansion).
pub const MaterializedOp = struct {
    allocator: Allocator,
    col_names: []const []const u8,
    rows: []const []Value,
    index: usize = 0,

    pub fn init(allocator: Allocator, col_names: []const []const u8, rows: []const []Value) MaterializedOp {
        return .{
            .allocator = allocator,
            .col_names = col_names,
            .rows = rows,
        };
    }

    pub fn next(self: *MaterializedOp) ExecError!?Row {
        if (self.index >= self.rows.len) return null;

        const source_vals = self.rows[self.index];
        self.index += 1;

        // Duplicate values for the returned row (Row.deinit will free them)
        const vals = self.allocator.alloc(Value, source_vals.len) catch return ExecError.OutOfMemory;
        var inited: usize = 0;
        errdefer {
            for (vals[0..inited]) |v| v.free(self.allocator);
            self.allocator.free(vals);
        }
        for (source_vals, 0..) |v, i| {
            vals[i] = v.dupe(self.allocator) catch return ExecError.OutOfMemory;
            inited += 1;
        }

        const cols = self.allocator.alloc([]const u8, self.col_names.len) catch return ExecError.OutOfMemory;
        for (self.col_names, 0..) |c, i| cols[i] = c;

        return Row{
            .columns = cols,
            .values = vals,
            .allocator = self.allocator,
        };
    }

    pub fn close(self: *MaterializedOp) void {
        for (self.rows) |vals| {
            for (vals) |v| v.free(self.allocator);
            self.allocator.free(vals);
        }
        self.allocator.free(self.rows);
        for (self.col_names) |name| self.allocator.free(@constCast(name));
        self.allocator.free(self.col_names);
    }

    pub fn iterator(self: *MaterializedOp) RowIterator {
        return .{
            .ptr = self,
            .vtable = &.{
                .next = @ptrCast(&MaterializedOp.next),
                .close = @ptrCast(&MaterializedOp.close),
            },
        };
    }
};

// ── Set Operation Operator ───────────────────────────────────────────────

/// Executes UNION, UNION ALL, INTERSECT, EXCEPT between two row iterators.
/// For UNION/INTERSECT/EXCEPT, uses a hash set of serialized rows for dedup.
pub const SetOpOp = struct {
    allocator: Allocator,
    left: RowIterator,
    right: RowIterator,
    op: ast.SetOpType,
    /// Phase: true = reading from left, false = reading from right
    reading_left: bool = true,
    /// For deduplication (UNION, INTERSECT, EXCEPT):
    /// Stores serialized row keys that have been seen/materialized.
    seen: std.StringHashMapUnmanaged(void) = .{},
    /// For INTERSECT/EXCEPT: materialized right side rows (serialized keys).
    right_set: ?std.StringHashMapUnmanaged(void) = null,
    /// Track all allocated keys for cleanup.
    allocated_keys: std.ArrayListUnmanaged([]u8) = .{},
    initialized: bool = false,

    pub fn init(allocator: Allocator, left: RowIterator, right: RowIterator, op: ast.SetOpType) SetOpOp {
        return .{
            .allocator = allocator,
            .left = left,
            .right = right,
            .op = op,
        };
    }

    /// Serialize a row's values into a comparable key for dedup.
    fn rowKey(self: *SetOpOp, row: *const Row) ![]u8 {
        return serializeRow(self.allocator, row.values);
    }

    /// Materialize all rows from the right side into a hash set.
    fn materializeRight(self: *SetOpOp) !void {
        var right_set = std.StringHashMapUnmanaged(void){};
        while (try self.right.next()) |*r| {
            var row = r.*;
            defer row.deinit();
            const key = try self.rowKey(&row);
            self.allocated_keys.append(self.allocator, key) catch return ExecError.OutOfMemory;
            right_set.put(self.allocator, key, {}) catch return ExecError.OutOfMemory;
        }
        self.right_set = right_set;
    }

    pub fn next(self: *SetOpOp) ExecError!?Row {
        // Initialize right side for INTERSECT/EXCEPT
        if (!self.initialized) {
            self.initialized = true;
            if (self.op == .intersect or self.op == .except) {
                try self.materializeRight();
            }
        }

        switch (self.op) {
            .union_all => return self.nextUnionAll(),
            .@"union" => return self.nextUnion(),
            .intersect => return self.nextIntersect(),
            .except => return self.nextExcept(),
        }
    }

    fn nextUnionAll(self: *SetOpOp) ExecError!?Row {
        if (self.reading_left) {
            if (try self.left.next()) |row| return row;
            self.reading_left = false;
        }
        return self.right.next();
    }

    fn nextUnion(self: *SetOpOp) ExecError!?Row {
        // Read from left, then right, skipping duplicates
        while (self.reading_left) {
            var row = try self.left.next() orelse {
                self.reading_left = false;
                break;
            };
            const key = self.rowKey(&row) catch {
                row.deinit();
                return ExecError.OutOfMemory;
            };
            if (self.seen.contains(key)) {
                self.allocator.free(key);
                row.deinit();
                continue;
            }
            self.allocated_keys.append(self.allocator, key) catch {
                self.allocator.free(key);
                row.deinit();
                return ExecError.OutOfMemory;
            };
            self.seen.put(self.allocator, key, {}) catch {
                row.deinit();
                return ExecError.OutOfMemory;
            };
            return row;
        }

        // Right side
        while (true) {
            var row = try self.right.next() orelse return null;
            const key = self.rowKey(&row) catch {
                row.deinit();
                return ExecError.OutOfMemory;
            };
            if (self.seen.contains(key)) {
                self.allocator.free(key);
                row.deinit();
                continue;
            }
            self.allocated_keys.append(self.allocator, key) catch {
                self.allocator.free(key);
                row.deinit();
                return ExecError.OutOfMemory;
            };
            self.seen.put(self.allocator, key, {}) catch {
                row.deinit();
                return ExecError.OutOfMemory;
            };
            return row;
        }
    }

    fn nextIntersect(self: *SetOpOp) ExecError!?Row {
        const rs = self.right_set orelse return null;
        while (true) {
            var row = try self.left.next() orelse return null;
            const key = self.rowKey(&row) catch {
                row.deinit();
                return ExecError.OutOfMemory;
            };
            defer self.allocator.free(key);
            if (rs.contains(key)) {
                // Also deduplicate: don't emit same row twice
                if (!self.seen.contains(key)) {
                    const key_dup = self.allocator.dupe(u8, key) catch {
                        row.deinit();
                        return ExecError.OutOfMemory;
                    };
                    self.allocated_keys.append(self.allocator, key_dup) catch {
                        self.allocator.free(key_dup);
                        row.deinit();
                        return ExecError.OutOfMemory;
                    };
                    self.seen.put(self.allocator, key_dup, {}) catch {
                        row.deinit();
                        return ExecError.OutOfMemory;
                    };
                    return row;
                }
                row.deinit();
                continue;
            }
            row.deinit();
        }
    }

    fn nextExcept(self: *SetOpOp) ExecError!?Row {
        const rs = self.right_set orelse return null;
        while (true) {
            var row = try self.left.next() orelse return null;
            const key = self.rowKey(&row) catch {
                row.deinit();
                return ExecError.OutOfMemory;
            };
            defer self.allocator.free(key);
            if (!rs.contains(key)) {
                // Also deduplicate within left side
                if (!self.seen.contains(key)) {
                    const key_dup = self.allocator.dupe(u8, key) catch {
                        row.deinit();
                        return ExecError.OutOfMemory;
                    };
                    self.allocated_keys.append(self.allocator, key_dup) catch {
                        self.allocator.free(key_dup);
                        row.deinit();
                        return ExecError.OutOfMemory;
                    };
                    self.seen.put(self.allocator, key_dup, {}) catch {
                        row.deinit();
                        return ExecError.OutOfMemory;
                    };
                    return row;
                }
                row.deinit();
                continue;
            }
            row.deinit();
        }
    }

    pub fn close(self: *SetOpOp) void {
        self.left.close();
        self.right.close();
        self.seen.deinit(self.allocator);
        if (self.right_set) |*rs| rs.deinit(self.allocator);
        for (self.allocated_keys.items) |key| self.allocator.free(key);
        self.allocated_keys.deinit(self.allocator);
    }

    pub fn iterator(self: *SetOpOp) RowIterator {
        return .{
            .ptr = self,
            .vtable = &.{
                .next = @ptrCast(&SetOpOp.next),
                .close = @ptrCast(&SetOpOp.close),
            },
        };
    }
};

// ── Execution Result ────────────────────────────────────────────────────

/// Result of executing a SQL statement.
pub const ExecResult = struct {
    /// For SELECT: iterator that produces rows.
    iterator: ?RowIterator = null,
    /// For DML: number of rows affected.
    rows_affected: u64 = 0,
    /// Human-readable message (for DDL, etc.).
    message: []const u8 = "",
};

// ── Tests ───────────────────────────────────────────────────────────────

test "Value compare" {
    const a = Value{ .integer = 42 };
    const b = Value{ .integer = 100 };
    const c = Value{ .integer = 42 };
    const n: Value = .null_value;

    try std.testing.expectEqual(std.math.Order.lt, a.compare(b));
    try std.testing.expectEqual(std.math.Order.gt, b.compare(a));
    try std.testing.expectEqual(std.math.Order.eq, a.compare(c));
    try std.testing.expect(a.eql(c));
    try std.testing.expect(!a.eql(b));

    // NULL comparisons
    try std.testing.expectEqual(std.math.Order.gt, n.compare(a));
    try std.testing.expectEqual(std.math.Order.lt, a.compare(n));
    try std.testing.expectEqual(std.math.Order.eq, n.compare(n));
}

test "Value isTruthy" {
    try std.testing.expect(Value.isTruthy(.{ .integer = 1 }));
    try std.testing.expect(!Value.isTruthy(.{ .integer = 0 }));
    try std.testing.expect(Value.isTruthy(.{ .boolean = true }));
    try std.testing.expect(!Value.isTruthy(.{ .boolean = false }));
    try std.testing.expect(!Value.isTruthy(.null_value));
    try std.testing.expect(Value.isTruthy(.{ .text = "hello" }));
    try std.testing.expect(!Value.isTruthy(.{ .text = "" }));
}

test "Value toInteger and toReal" {
    try std.testing.expectEqual(@as(?i64, 42), (Value{ .integer = 42 }).toInteger());
    try std.testing.expectEqual(@as(?i64, 3), (Value{ .real = 3.7 }).toInteger());
    try std.testing.expectEqual(@as(?i64, 1), (Value{ .boolean = true }).toInteger());
    const null_val: Value = .null_value;
    try std.testing.expectEqual(@as(?i64, null), null_val.toInteger());

    try std.testing.expectEqual(@as(?f64, 42.0), (Value{ .integer = 42 }).toReal());
    try std.testing.expectEqual(@as(?f64, 3.14), (Value{ .real = 3.14 }).toReal());
}

test "Value dupe and free" {
    const allocator = std.testing.allocator;
    const original = Value{ .text = "hello" };
    const duped = try original.dupe(allocator);
    defer duped.free(allocator);

    try std.testing.expectEqualStrings("hello", duped.text);
    // Verify it's a different allocation
    try std.testing.expect(original.text.ptr != duped.text.ptr);
}

test "Row getColumn" {
    const allocator = std.testing.allocator;
    const cols = try allocator.alloc([]const u8, 2);
    defer allocator.free(cols);
    cols[0] = "id";
    cols[1] = "name";

    const vals = try allocator.alloc(Value, 2);
    defer allocator.free(vals);
    vals[0] = .{ .integer = 1 };
    vals[1] = .{ .text = "Alice" };

    const row = Row{ .columns = cols, .values = vals, .allocator = allocator };

    try std.testing.expectEqual(@as(i64, 1), row.getColumn("id").?.integer);
    try std.testing.expectEqualStrings("Alice", row.getColumn("name").?.text);
    try std.testing.expectEqual(@as(?Value, null), row.getColumn("age"));
}

test "serialize and deserialize row" {
    const allocator = std.testing.allocator;
    const values = [_]Value{
        .{ .integer = 42 },
        .{ .text = "hello" },
        .{ .boolean = true },
        .null_value,
        .{ .real = 3.14 },
    };

    const data = try serializeRow(allocator, &values);
    defer allocator.free(data);

    const result = try deserializeRow(allocator, data);
    defer {
        for (result) |v| v.free(allocator);
        allocator.free(result);
    }

    try std.testing.expectEqual(@as(usize, 5), result.len);
    try std.testing.expectEqual(@as(i64, 42), result[0].integer);
    try std.testing.expectEqualStrings("hello", result[1].text);
    try std.testing.expect(result[2].boolean);
    try std.testing.expectEqual(Value.null_value, result[3]);
    try std.testing.expectEqual(@as(f64, 3.14), result[4].real);
}

test "evalExpr literals" {
    const allocator = std.testing.allocator;
    const empty_row = Row{ .columns = &.{}, .values = &.{}, .allocator = allocator };

    const int_expr = ast.Expr{ .integer_literal = 42 };
    const v1 = try evalExpr(allocator, &int_expr, &empty_row);
    defer v1.free(allocator);
    try std.testing.expectEqual(@as(i64, 42), v1.integer);

    const str_expr = ast.Expr{ .string_literal = "hello" };
    const v2 = try evalExpr(allocator, &str_expr, &empty_row);
    defer v2.free(allocator);
    try std.testing.expectEqualStrings("hello", v2.text);

    const bool_expr = ast.Expr{ .boolean_literal = true };
    const v3 = try evalExpr(allocator, &bool_expr, &empty_row);
    try std.testing.expect(v3.boolean);

    const null_expr = ast.Expr{ .null_literal = {} };
    const v4 = try evalExpr(allocator, &null_expr, &empty_row);
    try std.testing.expectEqual(Value.null_value, v4);
}

test "evalExpr column reference" {
    const allocator = std.testing.allocator;
    const cols = try allocator.alloc([]const u8, 2);
    defer allocator.free(cols);
    cols[0] = "id";
    cols[1] = "name";

    const vals = try allocator.alloc(Value, 2);
    defer allocator.free(vals);
    vals[0] = .{ .integer = 1 };
    vals[1] = .{ .text = "Alice" };

    const row = Row{ .columns = cols, .values = vals, .allocator = allocator };

    const ref_expr = ast.Expr{ .column_ref = .{ .name = "id" } };
    const v = try evalExpr(allocator, &ref_expr, &row);
    defer v.free(allocator);
    try std.testing.expectEqual(@as(i64, 1), v.integer);

    const bad_ref = ast.Expr{ .column_ref = .{ .name = "nonexistent" } };
    try std.testing.expectError(EvalError.ColumnNotFound, evalExpr(allocator, &bad_ref, &row));
}

test "evalExpr binary arithmetic" {
    const allocator = std.testing.allocator;
    const empty_row = Row{ .columns = &.{}, .values = &.{}, .allocator = allocator };

    const left = ast.Expr{ .integer_literal = 10 };
    const right = ast.Expr{ .integer_literal = 3 };
    const add_expr = ast.Expr{ .binary_op = .{ .op = .add, .left = &left, .right = &right } };
    const v = try evalExpr(allocator, &add_expr, &empty_row);
    try std.testing.expectEqual(@as(i64, 13), v.integer);

    const sub_expr = ast.Expr{ .binary_op = .{ .op = .subtract, .left = &left, .right = &right } };
    const v2 = try evalExpr(allocator, &sub_expr, &empty_row);
    try std.testing.expectEqual(@as(i64, 7), v2.integer);

    const mul_expr = ast.Expr{ .binary_op = .{ .op = .multiply, .left = &left, .right = &right } };
    const v3 = try evalExpr(allocator, &mul_expr, &empty_row);
    try std.testing.expectEqual(@as(i64, 30), v3.integer);

    const div_expr = ast.Expr{ .binary_op = .{ .op = .divide, .left = &left, .right = &right } };
    const v4 = try evalExpr(allocator, &div_expr, &empty_row);
    try std.testing.expectEqual(@as(i64, 3), v4.integer);

    const zero = ast.Expr{ .integer_literal = 0 };
    const div_zero = ast.Expr{ .binary_op = .{ .op = .divide, .left = &left, .right = &zero } };
    try std.testing.expectError(EvalError.DivisionByZero, evalExpr(allocator, &div_zero, &empty_row));
}

test "evalExpr comparison" {
    const allocator = std.testing.allocator;
    const empty_row = Row{ .columns = &.{}, .values = &.{}, .allocator = allocator };

    const left = ast.Expr{ .integer_literal = 10 };
    const right = ast.Expr{ .integer_literal = 20 };

    const lt = ast.Expr{ .binary_op = .{ .op = .less_than, .left = &left, .right = &right } };
    const v = try evalExpr(allocator, &lt, &empty_row);
    try std.testing.expect(v.boolean);

    const eq = ast.Expr{ .binary_op = .{ .op = .equal, .left = &left, .right = &left } };
    const v2 = try evalExpr(allocator, &eq, &empty_row);
    try std.testing.expect(v2.boolean);

    const neq = ast.Expr{ .binary_op = .{ .op = .not_equal, .left = &left, .right = &right } };
    const v3 = try evalExpr(allocator, &neq, &empty_row);
    try std.testing.expect(v3.boolean);
}

test "evalExpr logical AND/OR" {
    const allocator = std.testing.allocator;
    const empty_row = Row{ .columns = &.{}, .values = &.{}, .allocator = allocator };

    const t = ast.Expr{ .boolean_literal = true };
    const f = ast.Expr{ .boolean_literal = false };

    const and_expr = ast.Expr{ .binary_op = .{ .op = .@"and", .left = &t, .right = &f } };
    const v = try evalExpr(allocator, &and_expr, &empty_row);
    try std.testing.expect(!v.boolean);

    const or_expr = ast.Expr{ .binary_op = .{ .op = .@"or", .left = &t, .right = &f } };
    const v2 = try evalExpr(allocator, &or_expr, &empty_row);
    try std.testing.expect(v2.boolean);
}

test "evalExpr IS NULL / IS NOT NULL" {
    const allocator = std.testing.allocator;
    const empty_row = Row{ .columns = &.{}, .values = &.{}, .allocator = allocator };

    const null_expr = ast.Expr{ .null_literal = {} };
    const is_null = ast.Expr{ .is_null = .{ .expr = &null_expr } };
    const v = try evalExpr(allocator, &is_null, &empty_row);
    try std.testing.expect(v.boolean);

    const is_not_null = ast.Expr{ .is_null = .{ .expr = &null_expr, .negated = true } };
    const v2 = try evalExpr(allocator, &is_not_null, &empty_row);
    try std.testing.expect(!v2.boolean);

    const int_expr = ast.Expr{ .integer_literal = 42 };
    const not_null = ast.Expr{ .is_null = .{ .expr = &int_expr } };
    const v3 = try evalExpr(allocator, &not_null, &empty_row);
    try std.testing.expect(!v3.boolean);
}

test "evalExpr BETWEEN" {
    const allocator = std.testing.allocator;
    const empty_row = Row{ .columns = &.{}, .values = &.{}, .allocator = allocator };

    const val = ast.Expr{ .integer_literal = 5 };
    const low = ast.Expr{ .integer_literal = 1 };
    const high = ast.Expr{ .integer_literal = 10 };

    const between = ast.Expr{ .between = .{ .expr = &val, .low = &low, .high = &high } };
    const v = try evalExpr(allocator, &between, &empty_row);
    try std.testing.expect(v.boolean);

    const out = ast.Expr{ .integer_literal = 15 };
    const not_between = ast.Expr{ .between = .{ .expr = &out, .low = &low, .high = &high } };
    const v2 = try evalExpr(allocator, &not_between, &empty_row);
    try std.testing.expect(!v2.boolean);
}

test "evalExpr IN list" {
    const allocator = std.testing.allocator;
    const empty_row = Row{ .columns = &.{}, .values = &.{}, .allocator = allocator };

    const val = ast.Expr{ .integer_literal = 3 };
    const item1 = ast.Expr{ .integer_literal = 1 };
    const item2 = ast.Expr{ .integer_literal = 3 };
    const item3 = ast.Expr{ .integer_literal = 5 };
    const list = [_]*const ast.Expr{ &item1, &item2, &item3 };

    const in_expr = ast.Expr{ .in_list = .{ .expr = &val, .list = &list } };
    const v = try evalExpr(allocator, &in_expr, &empty_row);
    try std.testing.expect(v.boolean);

    const val2 = ast.Expr{ .integer_literal = 4 };
    const not_in = ast.Expr{ .in_list = .{ .expr = &val2, .list = &list } };
    const v2 = try evalExpr(allocator, &not_in, &empty_row);
    try std.testing.expect(!v2.boolean);
}

test "evalExpr LIKE" {
    const allocator = std.testing.allocator;
    const empty_row = Row{ .columns = &.{}, .values = &.{}, .allocator = allocator };

    const text = ast.Expr{ .string_literal = "hello world" };
    const pat1 = ast.Expr{ .string_literal = "hello%" };
    const like1 = ast.Expr{ .like = .{ .expr = &text, .pattern = &pat1 } };
    const v1 = try evalExpr(allocator, &like1, &empty_row);
    defer v1.free(allocator);
    try std.testing.expect(v1.boolean);

    const pat2 = ast.Expr{ .string_literal = "h_llo%" };
    const like2 = ast.Expr{ .like = .{ .expr = &text, .pattern = &pat2 } };
    const v2 = try evalExpr(allocator, &like2, &empty_row);
    defer v2.free(allocator);
    try std.testing.expect(v2.boolean);

    const pat3 = ast.Expr{ .string_literal = "goodbye%" };
    const like3 = ast.Expr{ .like = .{ .expr = &text, .pattern = &pat3 } };
    const v3 = try evalExpr(allocator, &like3, &empty_row);
    defer v3.free(allocator);
    try std.testing.expect(!v3.boolean);
}

test "evalExpr unary negation" {
    const allocator = std.testing.allocator;
    const empty_row = Row{ .columns = &.{}, .values = &.{}, .allocator = allocator };

    const inner = ast.Expr{ .integer_literal = 42 };
    const neg = ast.Expr{ .unary_op = .{ .op = .negate, .operand = &inner } };
    const v = try evalExpr(allocator, &neg, &empty_row);
    try std.testing.expectEqual(@as(i64, -42), v.integer);
}

test "evalExpr NOT" {
    const allocator = std.testing.allocator;
    const empty_row = Row{ .columns = &.{}, .values = &.{}, .allocator = allocator };

    const inner = ast.Expr{ .boolean_literal = true };
    const not_expr = ast.Expr{ .unary_op = .{ .op = .not, .operand = &inner } };
    const v = try evalExpr(allocator, &not_expr, &empty_row);
    try std.testing.expect(!v.boolean);
}

test "LIKE pattern matching" {
    try std.testing.expect(likeMatch("hello", "hello"));
    try std.testing.expect(likeMatch("hello", "%"));
    try std.testing.expect(likeMatch("hello", "h%"));
    try std.testing.expect(likeMatch("hello", "%o"));
    try std.testing.expect(likeMatch("hello", "h_llo"));
    try std.testing.expect(likeMatch("hello", "%ll%"));
    try std.testing.expect(!likeMatch("hello", "world"));
    try std.testing.expect(!likeMatch("hello", "h_lo"));
    try std.testing.expect(likeMatch("", ""));
    try std.testing.expect(likeMatch("", "%"));
    try std.testing.expect(!likeMatch("", "_"));
}

test "FilterOp filters rows" {
    const allocator = std.testing.allocator;

    // Create a simple in-memory data source
    var data = InMemorySource.init(allocator, &.{ "id", "name" });
    try data.addRow(&.{ Value{ .integer = 1 }, Value{ .text = "Alice" } });
    try data.addRow(&.{ Value{ .integer = 2 }, Value{ .text = "Bob" } });
    try data.addRow(&.{ Value{ .integer = 3 }, Value{ .text = "Charlie" } });
    defer data.deinit();

    // Filter: id > 1
    const id_ref = ast.Expr{ .column_ref = .{ .name = "id" } };
    const one = ast.Expr{ .integer_literal = 1 };
    const predicate = ast.Expr{ .binary_op = .{ .op = .greater_than, .left = &id_ref, .right = &one } };

    var filter = FilterOp.init(allocator, data.iterator(), &predicate);
    defer filter.close();

    var row1 = (try filter.next()).?;
    defer row1.deinit();
    try std.testing.expectEqual(@as(i64, 2), row1.getColumn("id").?.integer);

    var row2 = (try filter.next()).?;
    defer row2.deinit();
    try std.testing.expectEqual(@as(i64, 3), row2.getColumn("id").?.integer);

    try std.testing.expectEqual(@as(?Row, null), try filter.next());
}

test "ProjectOp selects columns" {
    const allocator = std.testing.allocator;

    var data = InMemorySource.init(allocator, &.{ "id", "name", "age" });
    try data.addRow(&.{ Value{ .integer = 1 }, Value{ .text = "Alice" }, Value{ .integer = 30 } });
    defer data.deinit();

    const name_ref = ast.Expr{ .column_ref = .{ .name = "name" } };
    const cols = [_]PlanNode.ProjectColumn{
        .{ .expr = &name_ref, .alias = "user_name" },
    };

    var proj = ProjectOp.init(allocator, data.iterator(), &cols);
    defer proj.close();

    var row = (try proj.next()).?;
    defer row.deinit();
    try std.testing.expectEqual(@as(usize, 1), row.columns.len);
    try std.testing.expectEqualStrings("user_name", row.columns[0]);
    try std.testing.expectEqualStrings("Alice", row.values[0].text);

    try std.testing.expectEqual(@as(?Row, null), try proj.next());
}

test "LimitOp with limit and offset" {
    const allocator = std.testing.allocator;

    var data = InMemorySource.init(allocator, &.{"id"});
    try data.addRow(&.{Value{ .integer = 1 }});
    try data.addRow(&.{Value{ .integer = 2 }});
    try data.addRow(&.{Value{ .integer = 3 }});
    try data.addRow(&.{Value{ .integer = 4 }});
    try data.addRow(&.{Value{ .integer = 5 }});
    defer data.deinit();

    var limit = LimitOp.init(allocator, data.iterator(), 2, 1);
    defer limit.close();

    var r1 = (try limit.next()).?;
    defer r1.deinit();
    try std.testing.expectEqual(@as(i64, 2), r1.getColumn("id").?.integer);

    var r2 = (try limit.next()).?;
    defer r2.deinit();
    try std.testing.expectEqual(@as(i64, 3), r2.getColumn("id").?.integer);

    try std.testing.expectEqual(@as(?Row, null), try limit.next());
}

test "SortOp sorts rows" {
    const allocator = std.testing.allocator;

    var data = InMemorySource.init(allocator, &.{"val"});
    try data.addRow(&.{Value{ .integer = 3 }});
    try data.addRow(&.{Value{ .integer = 1 }});
    try data.addRow(&.{Value{ .integer = 2 }});
    defer data.deinit();

    const val_ref = ast.Expr{ .column_ref = .{ .name = "val" } };
    const order = [_]ast.OrderByItem{
        .{ .expr = &val_ref, .direction = .asc },
    };

    var sort = SortOp.init(allocator, data.iterator(), &order);
    defer sort.close();

    var r1 = (try sort.next()).?;
    defer r1.deinit();
    try std.testing.expectEqual(@as(i64, 1), r1.getColumn("val").?.integer);

    var r2 = (try sort.next()).?;
    defer r2.deinit();
    try std.testing.expectEqual(@as(i64, 2), r2.getColumn("val").?.integer);

    var r3 = (try sort.next()).?;
    defer r3.deinit();
    try std.testing.expectEqual(@as(i64, 3), r3.getColumn("val").?.integer);

    try std.testing.expectEqual(@as(?Row, null), try sort.next());
}

test "SortOp descending" {
    const allocator = std.testing.allocator;

    var data = InMemorySource.init(allocator, &.{"val"});
    try data.addRow(&.{Value{ .integer = 1 }});
    try data.addRow(&.{Value{ .integer = 3 }});
    try data.addRow(&.{Value{ .integer = 2 }});
    defer data.deinit();

    const val_ref = ast.Expr{ .column_ref = .{ .name = "val" } };
    const order = [_]ast.OrderByItem{
        .{ .expr = &val_ref, .direction = .desc },
    };

    var sort = SortOp.init(allocator, data.iterator(), &order);
    defer sort.close();

    var r1 = (try sort.next()).?;
    defer r1.deinit();
    try std.testing.expectEqual(@as(i64, 3), r1.getColumn("val").?.integer);

    var r2 = (try sort.next()).?;
    defer r2.deinit();
    try std.testing.expectEqual(@as(i64, 2), r2.getColumn("val").?.integer);

    var r3 = (try sort.next()).?;
    defer r3.deinit();
    try std.testing.expectEqual(@as(i64, 1), r3.getColumn("val").?.integer);
}

test "AggregateOp count_star" {
    const allocator = std.testing.allocator;

    var data = InMemorySource.init(allocator, &.{"id"});
    try data.addRow(&.{Value{ .integer = 1 }});
    try data.addRow(&.{Value{ .integer = 2 }});
    try data.addRow(&.{Value{ .integer = 3 }});
    defer data.deinit();

    const aggs = [_]planner_mod.PlanNode.AggregateExpr{
        .{ .func = .count_star, .alias = "cnt" },
    };

    var agg = AggregateOp.init(allocator, data.iterator(), &.{}, &aggs);
    defer agg.close();

    var row = (try agg.next()).?;
    defer row.deinit();
    try std.testing.expectEqual(@as(i64, 3), row.getColumn("cnt").?.integer);

    try std.testing.expectEqual(@as(?Row, null), try agg.next());
}

test "AggregateOp sum and avg" {
    const allocator = std.testing.allocator;

    var data = InMemorySource.init(allocator, &.{"val"});
    try data.addRow(&.{Value{ .integer = 10 }});
    try data.addRow(&.{Value{ .integer = 20 }});
    try data.addRow(&.{Value{ .integer = 30 }});
    defer data.deinit();

    const val_ref = ast.Expr{ .column_ref = .{ .name = "val" } };
    const aggs = [_]planner_mod.PlanNode.AggregateExpr{
        .{ .func = .sum, .arg = &val_ref, .alias = "total" },
        .{ .func = .avg, .arg = &val_ref, .alias = "average" },
    };

    var agg = AggregateOp.init(allocator, data.iterator(), &.{}, &aggs);
    defer agg.close();

    var row = (try agg.next()).?;
    defer row.deinit();
    try std.testing.expectEqual(@as(i64, 60), row.getColumn("total").?.integer);
    try std.testing.expectEqual(@as(f64, 20.0), row.getColumn("average").?.real);
}

test "AggregateOp min and max" {
    const allocator = std.testing.allocator;

    var data = InMemorySource.init(allocator, &.{"val"});
    try data.addRow(&.{Value{ .integer = 30 }});
    try data.addRow(&.{Value{ .integer = 10 }});
    try data.addRow(&.{Value{ .integer = 20 }});
    defer data.deinit();

    const val_ref = ast.Expr{ .column_ref = .{ .name = "val" } };
    const aggs = [_]planner_mod.PlanNode.AggregateExpr{
        .{ .func = .min, .arg = &val_ref, .alias = "min_val" },
        .{ .func = .max, .arg = &val_ref, .alias = "max_val" },
    };

    var agg = AggregateOp.init(allocator, data.iterator(), &.{}, &aggs);
    defer agg.close();

    var row = (try agg.next()).?;
    defer row.deinit();
    try std.testing.expectEqual(@as(i64, 10), row.getColumn("min_val").?.integer);
    try std.testing.expectEqual(@as(i64, 30), row.getColumn("max_val").?.integer);
}

test "AggregateOp with GROUP BY" {
    const allocator = std.testing.allocator;

    var data = InMemorySource.init(allocator, &.{ "dept", "salary" });
    try data.addRow(&.{ Value{ .text = "eng" }, Value{ .integer = 100 } });
    try data.addRow(&.{ Value{ .text = "eng" }, Value{ .integer = 200 } });
    try data.addRow(&.{ Value{ .text = "hr" }, Value{ .integer = 150 } });
    defer data.deinit();

    const dept_ref = ast.Expr{ .column_ref = .{ .name = "dept" } };
    const salary_ref = ast.Expr{ .column_ref = .{ .name = "salary" } };
    const group_by = [_]*const ast.Expr{&dept_ref};
    const aggs = [_]planner_mod.PlanNode.AggregateExpr{
        .{ .func = .sum, .arg = &salary_ref, .alias = "total" },
    };

    var agg = AggregateOp.init(allocator, data.iterator(), &group_by, &aggs);
    defer agg.close();

    var r1 = (try agg.next()).?;
    defer r1.deinit();
    // Sorted by group key — "eng" first
    try std.testing.expectEqualStrings("eng", r1.getColumn("dept").?.text);
    try std.testing.expectEqual(@as(i64, 300), r1.getColumn("total").?.integer);

    var r2 = (try agg.next()).?;
    defer r2.deinit();
    try std.testing.expectEqualStrings("hr", r2.getColumn("dept").?.text);
    try std.testing.expectEqual(@as(i64, 150), r2.getColumn("total").?.integer);

    try std.testing.expectEqual(@as(?Row, null), try agg.next());
}

test "ValuesOp produces literal rows" {
    const allocator = std.testing.allocator;

    const e1 = ast.Expr{ .integer_literal = 1 };
    const e2 = ast.Expr{ .string_literal = "Alice" };
    const e3 = ast.Expr{ .integer_literal = 2 };
    const e4 = ast.Expr{ .string_literal = "Bob" };
    const row1 = [_]*const ast.Expr{ &e1, &e2 };
    const row2 = [_]*const ast.Expr{ &e3, &e4 };
    const rows = [_][]const *const ast.Expr{ &row1, &row2 };

    var values = ValuesOp.init(allocator, &.{ "id", "name" }, &rows);
    defer values.close();

    var r1 = (try values.next()).?;
    defer r1.deinit();
    try std.testing.expectEqual(@as(i64, 1), r1.getColumn("id").?.integer);
    try std.testing.expectEqualStrings("Alice", r1.getColumn("name").?.text);

    var r2 = (try values.next()).?;
    defer r2.deinit();
    try std.testing.expectEqual(@as(i64, 2), r2.getColumn("id").?.integer);
    try std.testing.expectEqualStrings("Bob", r2.getColumn("name").?.text);

    try std.testing.expectEqual(@as(?Row, null), try values.next());
}

test "NestedLoopJoinOp inner join" {
    const allocator = std.testing.allocator;

    var left = InMemorySource.init(allocator, &.{ "users.id", "users.name" });
    try left.addRow(&.{ Value{ .integer = 1 }, Value{ .text = "Alice" } });
    try left.addRow(&.{ Value{ .integer = 2 }, Value{ .text = "Bob" } });
    defer left.deinit();

    var right = InMemorySource.init(allocator, &.{ "orders.user_id", "orders.amount" });
    try right.addRow(&.{ Value{ .integer = 1 }, Value{ .integer = 100 } });
    try right.addRow(&.{ Value{ .integer = 1 }, Value{ .integer = 200 } });
    try right.addRow(&.{ Value{ .integer = 3 }, Value{ .integer = 50 } });
    defer right.deinit();

    // ON users.id = orders.user_id
    const left_ref = ast.Expr{ .column_ref = .{ .name = "id", .prefix = "users" } };
    const right_ref = ast.Expr{ .column_ref = .{ .name = "user_id", .prefix = "orders" } };
    const cond = ast.Expr{ .binary_op = .{ .op = .equal, .left = &left_ref, .right = &right_ref } };

    var join = NestedLoopJoinOp.init(allocator, left.iterator(), right.iterator(), .inner, &cond);
    defer join.close();

    var r1 = (try join.next()).?;
    defer r1.deinit();
    try std.testing.expectEqual(@as(i64, 1), r1.getQualifiedColumn("users", "id").?.integer);
    try std.testing.expectEqual(@as(i64, 100), r1.getQualifiedColumn("orders", "amount").?.integer);

    var r2 = (try join.next()).?;
    defer r2.deinit();
    try std.testing.expectEqual(@as(i64, 1), r2.getQualifiedColumn("users", "id").?.integer);
    try std.testing.expectEqual(@as(i64, 200), r2.getQualifiedColumn("orders", "amount").?.integer);

    // Bob (id=2) has no matching orders, so no row emitted
    try std.testing.expectEqual(@as(?Row, null), try join.next());
}

test "EmptyOp produces no rows" {
    var empty = EmptyOp{};
    try std.testing.expectEqual(@as(?Row, null), try empty.next());
}

test "evalExpr CASE expression" {
    const allocator = std.testing.allocator;
    const empty_row = Row{ .columns = &.{}, .values = &.{}, .allocator = allocator };

    // CASE WHEN true THEN 1 ELSE 0 END
    const when_cond = ast.Expr{ .boolean_literal = true };
    const then_val = ast.Expr{ .integer_literal = 1 };
    const else_val = ast.Expr{ .integer_literal = 0 };
    const when_clauses = [_]ast.WhenClause{
        .{ .condition = &when_cond, .result = &then_val },
    };
    const case_expr = ast.Expr{ .case_expr = .{ .when_clauses = &when_clauses, .else_expr = &else_val } };
    const v = try evalExpr(allocator, &case_expr, &empty_row);
    defer v.free(allocator);
    try std.testing.expectEqual(@as(i64, 1), v.integer);
}

test "evalExpr NULL arithmetic propagation" {
    const allocator = std.testing.allocator;
    const empty_row = Row{ .columns = &.{}, .values = &.{}, .allocator = allocator };

    const int_val = ast.Expr{ .integer_literal = 5 };
    const null_val = ast.Expr{ .null_literal = {} };
    const add_null = ast.Expr{ .binary_op = .{ .op = .add, .left = &int_val, .right = &null_val } };
    const v = try evalExpr(allocator, &add_null, &empty_row);
    try std.testing.expectEqual(Value.null_value, v);
}

test "AggregateOp empty input no GROUP BY" {
    const allocator = std.testing.allocator;

    var data = InMemorySource.init(allocator, &.{"val"});
    defer data.deinit();

    const val_ref = ast.Expr{ .column_ref = .{ .name = "val" } };
    const aggs = [_]planner_mod.PlanNode.AggregateExpr{
        .{ .func = .count_star, .alias = "cnt" },
        .{ .func = .sum, .arg = &val_ref, .alias = "total" },
    };

    var agg = AggregateOp.init(allocator, data.iterator(), &.{}, &aggs);
    defer agg.close();

    // With no GROUP BY and no rows, should emit one row with defaults
    var row = (try agg.next()).?;
    defer row.deinit();
    try std.testing.expectEqual(@as(i64, 0), row.getColumn("cnt").?.integer);
    try std.testing.expectEqual(Value.null_value, row.getColumn("total").?);
}

test "evalExpr string concatenation" {
    const allocator = std.testing.allocator;
    const empty_row = Row{ .columns = &.{}, .values = &.{}, .allocator = allocator };

    const left = ast.Expr{ .string_literal = "hello" };
    const right = ast.Expr{ .string_literal = " world" };
    const concat = ast.Expr{ .binary_op = .{ .op = .concat, .left = &left, .right = &right } };
    const v = try evalExpr(allocator, &concat, &empty_row);
    defer v.free(allocator);
    try std.testing.expectEqualStrings("hello world", v.text);
}

test "evalExpr CAST" {
    const allocator = std.testing.allocator;
    const empty_row = Row{ .columns = &.{}, .values = &.{}, .allocator = allocator };

    const int_val = ast.Expr{ .integer_literal = 42 };
    const cast_expr = ast.Expr{ .cast = .{ .expr = &int_val, .target_type = .type_text } };
    const v = try evalExpr(allocator, &cast_expr, &empty_row);
    defer v.free(allocator);
    try std.testing.expectEqualStrings("42", v.text);
}

// ── Test Helper: In-Memory Row Source ───────────────────────────────────

/// A simple in-memory row source for testing operators without storage.
const InMemorySource = struct {
    allocator: Allocator,
    col_names: []const []const u8,
    rows: std.ArrayListUnmanaged([]Value),
    index: usize = 0,

    fn init(allocator: Allocator, col_names: []const []const u8) InMemorySource {
        return .{
            .allocator = allocator,
            .col_names = col_names,
            .rows = .{},
        };
    }

    fn addRow(self: *InMemorySource, values: []const Value) !void {
        const row_vals = try self.allocator.alloc(Value, values.len);
        for (values, 0..) |v, i| {
            row_vals[i] = try v.dupe(self.allocator);
        }
        try self.rows.append(self.allocator, row_vals);
    }

    fn deinit(self: *InMemorySource) void {
        for (self.rows.items) |row_vals| {
            for (row_vals) |v| v.free(self.allocator);
            self.allocator.free(row_vals);
        }
        self.rows.deinit(self.allocator);
    }

    fn nextFn(ptr: *anyopaque) ExecError!?Row {
        const self: *InMemorySource = @ptrCast(@alignCast(ptr));
        if (self.index >= self.rows.items.len) return null;

        const row_vals = self.rows.items[self.index];
        self.index += 1;

        const vals = self.allocator.alloc(Value, row_vals.len) catch return ExecError.OutOfMemory;
        errdefer self.allocator.free(vals);
        var inited: usize = 0;
        errdefer for (vals[0..inited]) |v| v.free(self.allocator);

        for (row_vals, 0..) |v, i| {
            vals[i] = v.dupe(self.allocator) catch return ExecError.OutOfMemory;
            inited += 1;
        }

        const cols = self.allocator.alloc([]const u8, self.col_names.len) catch return ExecError.OutOfMemory;
        for (self.col_names, 0..) |c, i| cols[i] = c;

        return Row{
            .columns = cols,
            .values = vals,
            .allocator = self.allocator,
        };
    }

    fn closeFn(_: *anyopaque) void {}

    fn iterator(self: *InMemorySource) RowIterator {
        return .{
            .ptr = self,
            .vtable = &.{
                .next = &InMemorySource.nextFn,
                .close = &InMemorySource.closeFn,
            },
        };
    }
};

// ── MaterializedOp Tests ────────────────────────────────────────────────

test "MaterializedOp with multiple rows" {
    const allocator = std.testing.allocator;

    // Build col_names (owned by MaterializedOp — freed in close)
    const col_names = try allocator.alloc([]const u8, 2);
    col_names[0] = try allocator.dupe(u8, "id");
    col_names[1] = try allocator.dupe(u8, "name");

    // Build rows (owned by MaterializedOp — freed in close)
    const rows = try allocator.alloc([]Value, 2);
    rows[0] = try allocator.alloc(Value, 2);
    rows[0][0] = Value{ .integer = 1 };
    rows[0][1] = Value{ .text = try allocator.dupe(u8, "alice") };
    rows[1] = try allocator.alloc(Value, 2);
    rows[1][0] = Value{ .integer = 2 };
    rows[1][1] = Value{ .text = try allocator.dupe(u8, "bob") };

    var op = MaterializedOp.init(allocator, col_names, rows);

    // Row 1
    var r1 = (try op.next()).?;
    defer r1.deinit();
    try std.testing.expectEqual(@as(i64, 1), r1.values[0].integer);
    try std.testing.expectEqualStrings("alice", r1.values[1].text);
    try std.testing.expectEqualStrings("id", r1.columns[0]);
    try std.testing.expectEqualStrings("name", r1.columns[1]);

    // Row 2
    var r2 = (try op.next()).?;
    defer r2.deinit();
    try std.testing.expectEqual(@as(i64, 2), r2.values[0].integer);
    try std.testing.expectEqualStrings("bob", r2.values[1].text);

    // No more rows
    try std.testing.expect(try op.next() == null);

    op.close();
}

test "MaterializedOp with empty rows" {
    const allocator = std.testing.allocator;

    const col_names = try allocator.alloc([]const u8, 1);
    col_names[0] = try allocator.dupe(u8, "x");

    const rows = try allocator.alloc([]Value, 0);

    var op = MaterializedOp.init(allocator, col_names, rows);

    try std.testing.expect(try op.next() == null);

    op.close();
}

test "MaterializedOp single row with null value" {
    const allocator = std.testing.allocator;

    const col_names = try allocator.alloc([]const u8, 2);
    col_names[0] = try allocator.dupe(u8, "a");
    col_names[1] = try allocator.dupe(u8, "b");

    const rows = try allocator.alloc([]Value, 1);
    rows[0] = try allocator.alloc(Value, 2);
    rows[0][0] = Value{ .integer = 42 };
    rows[0][1] = .null_value;

    var op = MaterializedOp.init(allocator, col_names, rows);

    var r = (try op.next()).?;
    defer r.deinit();
    try std.testing.expectEqual(@as(i64, 42), r.values[0].integer);
    try std.testing.expect(r.values[1] == .null_value);

    try std.testing.expect(try op.next() == null);

    op.close();
}

test "MaterializedOp iterator interface" {
    const allocator = std.testing.allocator;

    const col_names = try allocator.alloc([]const u8, 1);
    col_names[0] = try allocator.dupe(u8, "val");

    const rows = try allocator.alloc([]Value, 2);
    rows[0] = try allocator.alloc(Value, 1);
    rows[0][0] = Value{ .integer = 10 };
    rows[1] = try allocator.alloc(Value, 1);
    rows[1][0] = Value{ .integer = 20 };

    var op = MaterializedOp.init(allocator, col_names, rows);
    var iter = op.iterator();

    var r1 = (try iter.next()).?;
    defer r1.deinit();
    try std.testing.expectEqual(@as(i64, 10), r1.values[0].integer);

    var r2 = (try iter.next()).?;
    defer r2.deinit();
    try std.testing.expectEqual(@as(i64, 20), r2.values[0].integer);

    try std.testing.expect(try iter.next() == null);

    iter.close();
}

test "MaterializedOp with text values duplicates correctly" {
    const allocator = std.testing.allocator;

    const col_names = try allocator.alloc([]const u8, 1);
    col_names[0] = try allocator.dupe(u8, "name");

    const rows = try allocator.alloc([]Value, 1);
    rows[0] = try allocator.alloc(Value, 1);
    rows[0][0] = Value{ .text = try allocator.dupe(u8, "hello") };

    var op = MaterializedOp.init(allocator, col_names, rows);

    var r = (try op.next()).?;
    // The returned row has a duplicated text value (independent memory)
    try std.testing.expectEqualStrings("hello", r.values[0].text);
    r.deinit();

    // After consuming all rows, close frees the source data
    try std.testing.expect(try op.next() == null);
    op.close();
}
