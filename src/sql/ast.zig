const std = @import("std");
const Allocator = std.mem.Allocator;
const tokenizer = @import("tokenizer.zig");
const TokenType = tokenizer.TokenType;

/// A name that may be qualified with a table/schema prefix.
/// Stored as byte offsets into the original SQL source.
pub const Name = struct {
    /// The identifier text (e.g., "id", "users")
    name: []const u8,
    /// Optional table/schema prefix (e.g., "t1" in "t1.id")
    prefix: ?[]const u8 = null,
};

/// SQL data type as declared in DDL.
pub const DataType = enum {
    type_integer,
    type_int,
    type_real,
    type_text,
    type_blob,
    type_boolean,
    type_varchar,
};

/// Column constraint in a CREATE TABLE statement.
pub const ColumnConstraint = union(enum) {
    primary_key: struct {
        autoincrement: bool = false,
    },
    not_null,
    unique,
    default: *const Expr,
    check: *const Expr,
    foreign_key: struct {
        table: []const u8,
        column: ?[]const u8 = null,
        on_delete: ?ForeignKeyAction = null,
        on_update: ?ForeignKeyAction = null,
    },
};

pub const ForeignKeyAction = enum {
    cascade,
    restrict,
    set_null,
    set_default,
    no_action,
};

/// Column definition in a CREATE TABLE statement.
pub const ColumnDef = struct {
    name: []const u8,
    data_type: ?DataType = null,
    constraints: []const ColumnConstraint = &.{},
};

/// Table constraint in a CREATE TABLE statement.
pub const TableConstraint = union(enum) {
    primary_key: struct {
        columns: []const []const u8,
    },
    unique: struct {
        columns: []const []const u8,
    },
    check: struct {
        expr: *const Expr,
    },
    foreign_key: struct {
        columns: []const []const u8,
        ref_table: []const u8,
        ref_columns: []const []const u8,
        on_delete: ?ForeignKeyAction = null,
        on_update: ?ForeignKeyAction = null,
    },
};

/// ORDER BY direction.
pub const OrderDirection = enum { asc, desc };

/// A single ORDER BY item.
pub const OrderByItem = struct {
    expr: *const Expr,
    direction: OrderDirection = .asc,
};

/// JOIN type.
pub const JoinType = enum {
    inner,
    left,
    right,
    full,
    cross,
};

/// A single JOIN clause.
pub const JoinClause = struct {
    join_type: JoinType = .inner,
    table: *const TableRef,
    on_condition: ?*const Expr = null,
};

/// Table reference in FROM clause.
pub const TableRef = union(enum) {
    /// Simple table name with optional alias
    table_name: struct {
        name: []const u8,
        alias: ?[]const u8 = null,
    },
    /// Subquery with alias
    subquery: struct {
        select: *const SelectStmt,
        alias: []const u8,
    },
};

/// SELECT result column.
pub const ResultColumn = union(enum) {
    /// All columns: *
    all_columns,
    /// All columns from a table: t.*
    table_all_columns: []const u8,
    /// Expression with optional alias: expr AS alias
    expr: struct {
        value: *const Expr,
        alias: ?[]const u8 = null,
    },
};

/// SQL expression node (recursive).
pub const Expr = union(enum) {
    /// Integer literal
    integer_literal: i64,
    /// Float literal
    float_literal: f64,
    /// String literal (without quotes)
    string_literal: []const u8,
    /// Blob literal
    blob_literal: []const u8,
    /// Boolean literal
    boolean_literal: bool,
    /// NULL literal
    null_literal,
    /// Column reference (optionally qualified)
    column_ref: Name,
    /// Unary operation: NOT expr, -expr
    unary_op: struct {
        op: UnaryOp,
        operand: *const Expr,
    },
    /// Binary operation: expr op expr
    binary_op: struct {
        op: BinaryOp,
        left: *const Expr,
        right: *const Expr,
    },
    /// Function call: func(args...)
    function_call: struct {
        name: []const u8,
        args: []const *const Expr,
        distinct: bool = false,
    },
    /// expr BETWEEN low AND high
    between: struct {
        expr: *const Expr,
        low: *const Expr,
        high: *const Expr,
        negated: bool = false,
    },
    /// expr IN (values...) or expr IN (subquery)
    in_list: struct {
        expr: *const Expr,
        list: []const *const Expr,
        negated: bool = false,
    },
    /// expr IS NULL / expr IS NOT NULL
    is_null: struct {
        expr: *const Expr,
        negated: bool = false,
    },
    /// expr LIKE pattern
    like: struct {
        expr: *const Expr,
        pattern: *const Expr,
        negated: bool = false,
    },
    /// CASE [expr] WHEN ... THEN ... [ELSE ...] END
    case_expr: struct {
        operand: ?*const Expr = null,
        when_clauses: []const WhenClause,
        else_expr: ?*const Expr = null,
    },
    /// CAST(expr AS type)
    cast: struct {
        expr: *const Expr,
        target_type: DataType,
    },
    /// Parenthesized expression
    paren: *const Expr,
    /// Subquery expression (scalar subquery)
    subquery: *const SelectStmt,
    /// Bind parameter: ?
    bind_parameter: u32,
};

pub const WhenClause = struct {
    condition: *const Expr,
    result: *const Expr,
};

pub const UnaryOp = enum {
    negate,
    not,
    bitwise_not,
};

pub const BinaryOp = enum {
    // Arithmetic
    add,
    subtract,
    multiply,
    divide,
    modulo,

    // Comparison
    equal,
    not_equal,
    less_than,
    greater_than,
    less_than_or_equal,
    greater_than_or_equal,

    // Logical
    @"and",
    @"or",

    // String
    concat,

    // Bitwise
    bitwise_and,
    bitwise_or,
    left_shift,
    right_shift,
};

/// SELECT statement.
pub const SelectStmt = struct {
    distinct: bool = false,
    columns: []const ResultColumn = &.{},
    from: ?*const TableRef = null,
    joins: []const JoinClause = &.{},
    where: ?*const Expr = null,
    group_by: []const *const Expr = &.{},
    having: ?*const Expr = null,
    order_by: []const OrderByItem = &.{},
    limit: ?*const Expr = null,
    offset: ?*const Expr = null,
};

/// INSERT statement.
pub const InsertStmt = struct {
    table: []const u8,
    columns: ?[]const []const u8 = null,
    values: []const []const *const Expr = &.{},
};

/// UPDATE statement.
pub const UpdateStmt = struct {
    table: []const u8,
    assignments: []const Assignment = &.{},
    where: ?*const Expr = null,
};

pub const Assignment = struct {
    column: []const u8,
    value: *const Expr,
};

/// DELETE statement.
pub const DeleteStmt = struct {
    table: []const u8,
    where: ?*const Expr = null,
};

/// CREATE TABLE statement.
pub const CreateTableStmt = struct {
    if_not_exists: bool = false,
    name: []const u8,
    columns: []const ColumnDef = &.{},
    table_constraints: []const TableConstraint = &.{},
    without_rowid: bool = false,
    strict: bool = false,
};

/// DROP TABLE statement.
pub const DropTableStmt = struct {
    if_exists: bool = false,
    name: []const u8,
};

/// CREATE INDEX statement.
pub const CreateIndexStmt = struct {
    if_not_exists: bool = false,
    unique: bool = false,
    name: []const u8,
    table: []const u8,
    columns: []const OrderByItem = &.{},
};

/// DROP INDEX statement.
pub const DropIndexStmt = struct {
    if_exists: bool = false,
    name: []const u8,
};

/// Transaction control statement.
pub const TransactionStmt = union(enum) {
    begin: struct {
        mode: TransactionMode = .deferred,
    },
    commit,
    rollback: struct {
        savepoint: ?[]const u8 = null,
    },
    savepoint: []const u8,
    release: []const u8,
};

pub const TransactionMode = enum {
    deferred,
    immediate,
    exclusive,
};

/// EXPLAIN statement.
pub const ExplainStmt = struct {
    stmt: *const Stmt,
};

/// VACUUM statement — reclaim dead tuples from a table (or all tables).
pub const VacuumStmt = struct {
    /// Optional table name to vacuum. null = vacuum all tables.
    table_name: ?[]const u8 = null,
};

/// Top-level SQL statement.
pub const Stmt = union(enum) {
    select: SelectStmt,
    insert: InsertStmt,
    update: UpdateStmt,
    delete: DeleteStmt,
    create_table: CreateTableStmt,
    drop_table: DropTableStmt,
    create_index: CreateIndexStmt,
    drop_index: DropIndexStmt,
    transaction: TransactionStmt,
    explain: ExplainStmt,
    vacuum: VacuumStmt,

    pub fn deinit(self: *const Stmt, allocator: Allocator) void {
        _ = self;
        _ = allocator;
        // AST nodes are allocated in an arena; no individual deallocation needed.
    }
};

/// Arena-based AST allocator. All AST nodes live for the duration of a parse.
pub const AstArena = struct {
    arena: std.heap.ArenaAllocator,

    pub fn init(backing: Allocator) AstArena {
        return .{
            .arena = std.heap.ArenaAllocator.init(backing),
        };
    }

    pub fn deinit(self: *AstArena) void {
        self.arena.deinit();
    }

    pub fn allocator(self: *AstArena) Allocator {
        return self.arena.allocator();
    }

    /// Create a single AST node.
    pub fn create(self: *AstArena, comptime T: type, value: T) !*const T {
        const ptr = try self.arena.allocator().create(T);
        ptr.* = value;
        return ptr;
    }

    /// Duplicate a slice into the arena.
    pub fn dupeSlice(self: *AstArena, comptime T: type, items: []const T) ![]const T {
        return self.arena.allocator().dupe(T, items);
    }
};

// Tests

test "AstArena basic allocation" {
    var ast_arena = AstArena.init(std.testing.allocator);
    defer ast_arena.deinit();

    const expr = try ast_arena.create(Expr, .{ .integer_literal = 42 });
    try std.testing.expectEqual(@as(i64, 42), expr.integer_literal);
}

test "AstArena creates expression tree" {
    var ast_arena = AstArena.init(std.testing.allocator);
    defer ast_arena.deinit();

    const left = try ast_arena.create(Expr, .{ .integer_literal = 1 });
    const right = try ast_arena.create(Expr, .{ .integer_literal = 2 });
    const add = try ast_arena.create(Expr, .{ .binary_op = .{
        .op = .add,
        .left = left,
        .right = right,
    } });

    try std.testing.expectEqual(BinaryOp.add, add.binary_op.op);
    try std.testing.expectEqual(@as(i64, 1), add.binary_op.left.integer_literal);
    try std.testing.expectEqual(@as(i64, 2), add.binary_op.right.integer_literal);
}

test "SelectStmt default values" {
    const stmt = SelectStmt{};
    try std.testing.expect(!stmt.distinct);
    try std.testing.expectEqual(@as(usize, 0), stmt.columns.len);
    try std.testing.expect(stmt.from == null);
    try std.testing.expect(stmt.where == null);
}

test "Stmt union select" {
    var ast_arena = AstArena.init(std.testing.allocator);
    defer ast_arena.deinit();

    const col_ref = try ast_arena.create(Expr, .{ .column_ref = .{ .name = "id" } });
    const columns = try ast_arena.dupeSlice(ResultColumn, &.{
        .{ .expr = .{ .value = col_ref } },
    });

    const stmt = Stmt{ .select = .{
        .columns = columns,
        .from = try ast_arena.create(TableRef, .{
            .table_name = .{ .name = "users" },
        }),
    } };

    switch (stmt) {
        .select => |s| {
            try std.testing.expectEqual(@as(usize, 1), s.columns.len);
            try std.testing.expectEqualStrings("users", s.from.?.table_name.name);
        },
        else => unreachable,
    }
}

test "CreateTableStmt with columns" {
    const stmt = CreateTableStmt{
        .name = "users",
        .columns = &.{
            .{
                .name = "id",
                .data_type = .type_integer,
                .constraints = &.{
                    .{ .primary_key = .{} },
                },
            },
            .{
                .name = "name",
                .data_type = .type_text,
                .constraints = &.{
                    .not_null,
                },
            },
        },
    };

    try std.testing.expectEqualStrings("users", stmt.name);
    try std.testing.expectEqual(@as(usize, 2), stmt.columns.len);
    try std.testing.expectEqualStrings("id", stmt.columns[0].name);
    try std.testing.expectEqual(DataType.type_integer, stmt.columns[0].data_type.?);
}

test "Expr unary and binary ops" {
    var ast_arena = AstArena.init(std.testing.allocator);
    defer ast_arena.deinit();

    const inner = try ast_arena.create(Expr, .{ .integer_literal = 5 });
    const neg = try ast_arena.create(Expr, .{ .unary_op = .{
        .op = .negate,
        .operand = inner,
    } });

    try std.testing.expectEqual(UnaryOp.negate, neg.unary_op.op);
    try std.testing.expectEqual(@as(i64, 5), neg.unary_op.operand.integer_literal);
}

test "InsertStmt with values" {
    const stmt = InsertStmt{
        .table = "users",
        .columns = &.{ "id", "name" },
    };

    try std.testing.expectEqualStrings("users", stmt.table);
    try std.testing.expectEqual(@as(usize, 2), stmt.columns.?.len);
}

test "DataType enum" {
    const t: DataType = .type_integer;
    try std.testing.expect(t == .type_integer);
}

test "BinaryOp covers all arithmetic" {
    const ops = [_]BinaryOp{ .add, .subtract, .multiply, .divide, .modulo };
    try std.testing.expectEqual(@as(usize, 5), ops.len);
}

test "ColumnConstraint variants" {
    const c1: ColumnConstraint = .{ .primary_key = .{ .autoincrement = true } };
    const c2: ColumnConstraint = .not_null;
    const c3: ColumnConstraint = .unique;

    switch (c1) {
        .primary_key => |pk| try std.testing.expect(pk.autoincrement),
        else => unreachable,
    }
    switch (c2) {
        .not_null => {},
        else => unreachable,
    }
    switch (c3) {
        .unique => {},
        else => unreachable,
    }
}
