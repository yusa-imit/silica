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
    type_date,
    type_time,
    type_timestamp,
    type_interval,
    type_numeric,
    type_decimal,
    type_uuid,
    type_serial,
    type_bigserial,
    type_array,
    type_json,
    type_jsonb,
    type_tsvector,
    type_tsquery,
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
    /// Table function call: func(args...) with optional alias
    table_function: struct {
        name: []const u8,
        args: []const *const Expr,
        alias: ?[]const u8 = null,
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

/// Window function call payload: func(...) OVER (PARTITION BY ... ORDER BY ... frame)
pub const WindowFunctionExpr = struct {
    /// Function name (e.g., "row_number", "rank", "sum")
    name: []const u8,
    /// Function arguments (empty for ROW_NUMBER, RANK, etc.)
    args: []const *const Expr = &.{},
    /// DISTINCT in function args (e.g., COUNT(DISTINCT x) OVER ...)
    distinct: bool = false,
    /// PARTITION BY expressions
    partition_by: []const *const Expr = &.{},
    /// ORDER BY within the window
    order_by: []const OrderByItem = &.{},
    /// Optional frame specification
    frame: ?*const WindowFrameSpec = null,
    /// Named window reference (e.g., OVER w) — resolved by planner
    window_name: ?[]const u8 = null,
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
    /// EXISTS (subquery) — returns true if subquery has any rows
    exists: struct {
        subquery: *const SelectStmt,
        negated: bool = false,
    },
    /// Window function call: func(...) OVER (PARTITION BY ... ORDER BY ... frame)
    window_function: WindowFunctionExpr,
    /// ARRAY[expr, expr, ...] constructor
    array_constructor: []const *const Expr,
    /// expr[index] — array subscript access (1-based)
    array_subscript: struct {
        array: *const Expr,
        index: *const Expr,
    },
    /// expr op ANY(array) — true if comparison holds for any array element
    any: struct {
        expr: *const Expr,
        op: BinaryOp,
        array: *const Expr,
    },
    /// expr op ALL(array) — true if comparison holds for all array elements
    all: struct {
        expr: *const Expr,
        op: BinaryOp,
        array: *const Expr,
    },
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

    // JSON operators
    json_extract, // ->
    json_extract_text, // ->>
    json_contains, // @>
    json_contained_by, // <@
    json_key_exists, // ?
    json_any_key_exists, // ?|
    json_all_keys_exist, // ?&
    json_path_extract, // #>
    json_path_extract_text, // #>>
    json_delete_path, // #-

    // Full-text search
    ts_match, // @@
};

/// Window frame mode: ROWS, RANGE, or GROUPS.
pub const WindowFrameMode = enum {
    rows,
    range,
    groups,
};

/// Window frame bound specification.
pub const WindowFrameBound = union(enum) {
    /// UNBOUNDED PRECEDING
    unbounded_preceding,
    /// UNBOUNDED FOLLOWING
    unbounded_following,
    /// CURRENT ROW
    current_row,
    /// <expr> PRECEDING
    expr_preceding: *const Expr,
    /// <expr> FOLLOWING
    expr_following: *const Expr,
};

/// Window frame specification: ROWS/RANGE/GROUPS BETWEEN ... AND ...
pub const WindowFrameSpec = struct {
    mode: WindowFrameMode = .range,
    start: WindowFrameBound = .unbounded_preceding,
    end: WindowFrameBound = .current_row,
};

/// Named window definition in WINDOW clause.
pub const WindowDef = struct {
    name: []const u8,
    partition_by: []const *const Expr = &.{},
    order_by: []const OrderByItem = &.{},
    frame: ?*const WindowFrameSpec = null,
};

/// A single CTE definition in a WITH clause.
pub const CteDefinition = struct {
    /// CTE name (used as table reference in main query)
    name: []const u8,
    /// The SELECT query defining the CTE
    select: *const SelectStmt,
    /// Optional explicit column names for the CTE
    column_names: []const []const u8 = &.{},
};

/// Set operation type for UNION, INTERSECT, EXCEPT.
pub const SetOpType = enum {
    @"union",
    union_all,
    intersect,
    except,
};

/// A chained set operation: current SELECT <op> next SELECT.
pub const SetOperation = struct {
    op: SetOpType,
    right: *const SelectStmt,
};

/// SELECT statement.
pub const SelectStmt = struct {
    /// Common Table Expressions (WITH ... AS clauses)
    ctes: []const CteDefinition = &.{},
    /// Whether WITH RECURSIVE was used
    recursive: bool = false,
    distinct: bool = false,
    /// DISTINCT ON expressions — when non-empty, dedup is based on these expressions only.
    /// The first row per unique combination of these expressions is returned.
    distinct_on: []const *const Expr = &.{},
    columns: []const ResultColumn = &.{},
    from: ?*const TableRef = null,
    joins: []const JoinClause = &.{},
    where: ?*const Expr = null,
    group_by: []const *const Expr = &.{},
    having: ?*const Expr = null,
    order_by: []const OrderByItem = &.{},
    limit: ?*const Expr = null,
    offset: ?*const Expr = null,
    /// Named window definitions (WINDOW w AS (...), ...)
    window_defs: []const WindowDef = &.{},
    /// Optional set operation chaining (UNION, INTERSECT, EXCEPT)
    set_operation: ?*const SetOperation = null,
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
    /// Build index without blocking concurrent writes to the table
    concurrently: bool = false,
    name: []const u8,
    table: []const u8,
    columns: []const OrderByItem = &.{},
    /// Non-indexed columns included in the index for covering (index-only) scans
    included_columns: []const []const u8 = &.{},
    /// Index type: "btree" or "hash" (null means default to btree)
    index_type: ?[]const u8 = null,
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
    /// If true, execute the query and show actual runtime statistics.
    analyze: bool = false,
};

/// VACUUM statement — reclaim dead tuples from a table (or all tables).
pub const VacuumStmt = struct {
    /// Optional table name to vacuum. null = vacuum all tables.
    table_name: ?[]const u8 = null,
};

/// ANALYZE statement: collect table statistics for query optimization.
pub const AnalyzeStmt = struct {
    /// Optional table name to analyze. null = analyze all tables.
    table_name: ?[]const u8 = null,
};

/// WITH CHECK OPTION type for updatable views.
pub const CheckOption = enum {
    none,
    local,
    cascaded,
};

/// CREATE VIEW statement.
pub const CreateViewStmt = struct {
    name: []const u8,
    /// The SELECT query that defines the view.
    select: SelectStmt,
    /// CREATE OR REPLACE VIEW
    or_replace: bool = false,
    /// IF NOT EXISTS
    if_not_exists: bool = false,
    /// Optional column aliases for the view.
    column_names: []const []const u8 = &.{},
    /// WITH [LOCAL | CASCADED] CHECK OPTION
    check_option: CheckOption = .none,
};

/// DROP VIEW statement.
pub const DropViewStmt = struct {
    name: []const u8,
    if_exists: bool = false,
};

/// CREATE TYPE AS ENUM statement.
pub const CreateTypeStmt = struct {
    name: []const u8,
    /// The enum values (e.g., 'happy', 'sad').
    values: []const []const u8,
};

/// DROP TYPE statement.
pub const DropTypeStmt = struct {
    name: []const u8,
    if_exists: bool = false,
};

/// CREATE DOMAIN statement.
pub const CreateDomainStmt = struct {
    name: []const u8,
    base_type: DataType,
    constraint: ?*const Expr = null,
};

/// DROP DOMAIN statement.
pub const DropDomainStmt = struct {
    name: []const u8,
    if_exists: bool = false,
};

/// Function parameter definition.
pub const FunctionParam = struct {
    name: []const u8,
    data_type: DataType,
};

/// Function return type specification.
pub const FunctionReturn = union(enum) {
    /// RETURNS type_name
    scalar: DataType,
    /// RETURNS TABLE(col1 type1, col2 type2, ...)
    table: []const ColumnDef,
    /// RETURNS SETOF type_name
    setof: DataType,
};

/// Function volatility category.
pub const FunctionVolatility = enum {
    immutable, // Same input always gives same output, no side effects
    stable,    // Same input gives same output within a transaction
    vol,       // Output can vary, has side effects (volatile)
};

/// CREATE FUNCTION statement.
pub const CreateFunctionStmt = struct {
    name: []const u8,
    parameters: []const FunctionParam = &.{},
    return_type: FunctionReturn,
    language: []const u8, // e.g., "sfl" for Silica Function Language
    body: []const u8,     // Function body (SQL or SFL code)
    volatility: FunctionVolatility = .vol,
    or_replace: bool = false,
};

/// DROP FUNCTION statement.
pub const DropFunctionStmt = struct {
    name: []const u8,
    /// Optional parameter types for overload resolution
    param_types: []const DataType = &.{},
    if_exists: bool = false,
};

/// Trigger timing: when the trigger fires relative to the event.
pub const TriggerTiming = enum {
    before,     // BEFORE INSERT/UPDATE/DELETE
    after,      // AFTER INSERT/UPDATE/DELETE
    instead_of, // INSTEAD OF (for views)
};

/// Trigger event: what operation causes the trigger to fire.
pub const TriggerEvent = enum {
    insert,
    update,
    delete,
    truncate,
};

/// Trigger level: per-row or per-statement.
pub const TriggerLevel = enum {
    row,       // FOR EACH ROW
    statement, // FOR EACH STATEMENT
};

/// CREATE TRIGGER statement.
pub const CreateTriggerStmt = struct {
    name: []const u8,
    table_name: []const u8,
    timing: TriggerTiming,
    event: TriggerEvent,
    /// Columns for UPDATE OF clause (empty for other events)
    update_columns: []const []const u8 = &.{},
    level: TriggerLevel = .row,
    /// WHEN (condition) clause
    when_condition: ?*const Expr = null,
    /// Trigger body (SQL statements to execute)
    body: []const u8,
    or_replace: bool = false,
};

/// DROP TRIGGER statement.
pub const DropTriggerStmt = struct {
    name: []const u8,
    table_name: ?[]const u8 = null, // Optional; some DBs require it
    if_exists: bool = false,
};

/// ALTER TRIGGER statement (ENABLE/DISABLE).
pub const AlterTriggerStmt = struct {
    name: []const u8,
    table_name: ?[]const u8 = null,
    enable: bool, // true = ENABLE, false = DISABLE
};

/// Role options for CREATE/ALTER ROLE.
pub const RoleOptions = struct {
    login: ?bool = null, // LOGIN or NOLOGIN
    superuser: ?bool = null, // SUPERUSER or NOSUPERUSER
    createdb: ?bool = null, // CREATEDB or NOCREATEDB
    createrole: ?bool = null, // CREATEROLE or NOCREATEROLE
    inherit: ?bool = null, // INHERIT or NOINHERIT
    password: ?[]const u8 = null, // PASSWORD 'password'
    valid_until: ?[]const u8 = null, // VALID UNTIL 'timestamp'
};

/// CREATE ROLE statement.
pub const CreateRoleStmt = struct {
    name: []const u8,
    options: RoleOptions = .{},
    or_replace: bool = false,
};

/// DROP ROLE statement.
pub const DropRoleStmt = struct {
    name: []const u8,
    if_exists: bool = false,
};

/// ALTER ROLE statement.
pub const AlterRoleStmt = struct {
    name: []const u8,
    options: RoleOptions = .{},
};

/// Privilege type for GRANT/REVOKE.
pub const Privilege = enum {
    select,
    insert,
    update,
    delete,
    all,
};

/// Object type for GRANT/REVOKE.
pub const ObjectType = enum {
    table,
    schema,
    function,
    sequence,
    database,
};

/// GRANT statement.
pub const GrantStmt = struct {
    privileges: []const Privilege,
    object_type: ObjectType,
    object_name: []const u8,
    grantee: []const u8, // role name
    with_grant_option: bool = false,
};

/// REVOKE statement.
pub const RevokeStmt = struct {
    privileges: []const Privilege,
    object_type: ObjectType,
    object_name: []const u8,
    grantee: []const u8, // role name
};

/// GRANT role membership statement: GRANT role TO member
pub const GrantRoleStmt = struct {
    role: []const u8, // role being granted
    members: []const []const u8, // users/roles receiving the role
    with_admin_option: bool = false,
};

/// REVOKE role membership statement: REVOKE role FROM member
pub const RevokeRoleStmt = struct {
    role: []const u8, // role being revoked
    members: []const []const u8, // users/roles losing the role
};

/// Policy command type for CREATE POLICY
pub const PolicyCommand = enum {
    all, // default: applies to all commands
    select,
    insert,
    update,
    delete,
};

/// Policy type for CREATE POLICY
pub const PolicyType = enum {
    permissive, // default: OR logic with other policies
    restrictive, // AND logic with other policies
};

/// CREATE POLICY statement: CREATE POLICY name ON table [AS {PERMISSIVE|RESTRICTIVE}] [FOR {ALL|SELECT|INSERT|UPDATE|DELETE}] [USING (qual)] [WITH CHECK (with_check)]
pub const CreatePolicyStmt = struct {
    policy_name: []const u8,
    table_name: []const u8,
    policy_type: PolicyType = .permissive,
    command: PolicyCommand = .all,
    using_expr: ?Expr = null, // USING clause (for SELECT, UPDATE, DELETE)
    with_check_expr: ?Expr = null, // WITH CHECK clause (for INSERT, UPDATE)
};

/// DROP POLICY statement: DROP POLICY [IF EXISTS] name ON table
pub const DropPolicyStmt = struct {
    policy_name: []const u8,
    table_name: []const u8,
    if_exists: bool = false,
};

/// ALTER TABLE ENABLE/DISABLE ROW LEVEL SECURITY statement
pub const AlterTableRLSStmt = struct {
    table_name: []const u8,
    enable: bool, // true = ENABLE, false = DISABLE
    force: bool = false, // FORCE ROW LEVEL SECURITY (applies even to table owner)
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
    analyze: AnalyzeStmt,
    vacuum: VacuumStmt,
    create_view: CreateViewStmt,
    drop_view: DropViewStmt,
    create_type: CreateTypeStmt,
    drop_type: DropTypeStmt,
    create_domain: CreateDomainStmt,
    drop_domain: DropDomainStmt,
    create_function: CreateFunctionStmt,
    drop_function: DropFunctionStmt,
    create_trigger: CreateTriggerStmt,
    drop_trigger: DropTriggerStmt,
    alter_trigger: AlterTriggerStmt,
    create_role: CreateRoleStmt,
    drop_role: DropRoleStmt,
    alter_role: AlterRoleStmt,
    grant: GrantStmt,
    revoke: RevokeStmt,
    grant_role: GrantRoleStmt,
    revoke_role: RevokeRoleStmt,
    create_policy: CreatePolicyStmt,
    drop_policy: DropPolicyStmt,
    alter_table_rls: AlterTableRLSStmt,

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

test "SetOpType enum and SetOperation struct" {
    var ast_arena = AstArena.init(std.testing.allocator);
    defer ast_arena.deinit();

    const right_select = try ast_arena.create(SelectStmt, .{});
    const set_op = try ast_arena.create(SetOperation, .{
        .op = .@"union",
        .right = right_select,
    });

    const stmt = SelectStmt{
        .set_operation = set_op,
    };
    try std.testing.expect(stmt.set_operation != null);
    try std.testing.expectEqual(SetOpType.@"union", stmt.set_operation.?.op);

    // Test all set op types
    try std.testing.expect(SetOpType.union_all != SetOpType.@"union");
    try std.testing.expect(SetOpType.intersect != SetOpType.except);
}

test "WindowFunction expression" {
    var ast_arena = AstArena.init(std.testing.allocator);
    defer ast_arena.deinit();

    // ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC)
    const dept_ref = try ast_arena.create(Expr, .{ .column_ref = .{ .name = "dept" } });
    const salary_ref = try ast_arena.create(Expr, .{ .column_ref = .{ .name = "salary" } });
    const partition_by = try ast_arena.dupeSlice(*const Expr, &.{dept_ref});
    const order_by = try ast_arena.dupeSlice(OrderByItem, &.{
        .{ .expr = salary_ref, .direction = .desc },
    });

    const wf = try ast_arena.create(Expr, .{ .window_function = .{
        .name = "row_number",
        .partition_by = partition_by,
        .order_by = order_by,
    } });

    try std.testing.expectEqualStrings("row_number", wf.window_function.name);
    try std.testing.expectEqual(@as(usize, 0), wf.window_function.args.len);
    try std.testing.expectEqual(@as(usize, 1), wf.window_function.partition_by.len);
    try std.testing.expectEqual(@as(usize, 1), wf.window_function.order_by.len);
    try std.testing.expectEqual(OrderDirection.desc, wf.window_function.order_by[0].direction);
    try std.testing.expect(wf.window_function.frame == null);
}

test "WindowFrameSpec" {
    var ast_arena = AstArena.init(std.testing.allocator);
    defer ast_arena.deinit();

    // ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    const frame = try ast_arena.create(WindowFrameSpec, .{
        .mode = .rows,
        .start = .unbounded_preceding,
        .end = .current_row,
    });

    try std.testing.expectEqual(WindowFrameMode.rows, frame.mode);
    try std.testing.expectEqual(WindowFrameBound.unbounded_preceding, frame.start);
    try std.testing.expectEqual(WindowFrameBound.current_row, frame.end);

    // ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING
    const one = try ast_arena.create(Expr, .{ .integer_literal = 1 });
    const frame2 = try ast_arena.create(WindowFrameSpec, .{
        .mode = .rows,
        .start = .{ .expr_preceding = one },
        .end = .{ .expr_following = one },
    });

    try std.testing.expectEqual(WindowFrameMode.rows, frame2.mode);
    switch (frame2.start) {
        .expr_preceding => |e| try std.testing.expectEqual(@as(i64, 1), e.integer_literal),
        else => unreachable,
    }
}

test "WindowDef named window" {
    const def = WindowDef{
        .name = "w",
    };
    try std.testing.expectEqualStrings("w", def.name);
    try std.testing.expectEqual(@as(usize, 0), def.partition_by.len);
    try std.testing.expectEqual(@as(usize, 0), def.order_by.len);
    try std.testing.expect(def.frame == null);
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

test "JSON binary operators" {
    // Verify JSON operators are valid BinaryOp variants by using them in a switch
    const op: BinaryOp = .json_extract;
    switch (op) {
        .json_extract,
        .json_extract_text,
        .json_contains,
        .json_contained_by,
        .json_key_exists,
        .json_any_key_exists,
        .json_all_keys_exist,
        .json_path_extract,
        .json_path_extract_text,
        .json_delete_path,
        => {},
        else => {},
    }

    // Test that all JSON operators can be assigned
    const ops = [_]BinaryOp{
        .json_extract,
        .json_extract_text,
        .json_contains,
        .json_contained_by,
        .json_key_exists,
        .json_any_key_exists,
        .json_all_keys_exist,
        .json_path_extract,
        .json_path_extract_text,
        .json_delete_path,
    };
    try std.testing.expectEqual(@as(usize, 10), ops.len);
}

test "CreateFunctionStmt basic structure" {
    var arena = AstArena.init(std.testing.allocator);
    defer arena.deinit();

    const params = try arena.dupeSlice(FunctionParam, &.{
        .{ .name = "x", .data_type = .type_integer },
        .{ .name = "y", .data_type = .type_integer },
    });

    const stmt = CreateFunctionStmt{
        .name = "add_numbers",
        .parameters = params,
        .return_type = .{ .scalar = .type_integer },
        .language = "sfl",
        .body = "RETURN x + y;",
        .volatility = .immutable,
    };

    try std.testing.expectEqualStrings("add_numbers", stmt.name);
    try std.testing.expectEqual(@as(usize, 2), stmt.parameters.len);
    try std.testing.expectEqualStrings("x", stmt.parameters[0].name);
    try std.testing.expectEqual(DataType.type_integer, stmt.parameters[0].data_type);
    try std.testing.expectEqual(FunctionVolatility.immutable, stmt.volatility);
}

test "FunctionReturn variants" {
    // Test scalar return
    const scalar_ret = FunctionReturn{ .scalar = .type_text };
    try std.testing.expectEqual(DataType.type_text, scalar_ret.scalar);

    // Test setof return
    const setof_ret = FunctionReturn{ .setof = .type_integer };
    try std.testing.expectEqual(DataType.type_integer, setof_ret.setof);

    // Table return is tested via its presence in the union
    var arena = AstArena.init(std.testing.allocator);
    defer arena.deinit();

    const cols = try arena.dupeSlice(ColumnDef, &.{
        .{ .name = "id", .data_type = .type_integer },
    });
    const table_ret = FunctionReturn{ .table = cols };
    try std.testing.expectEqual(@as(usize, 1), table_ret.table.len);
}

test "DropFunctionStmt with overload resolution" {
    var arena = AstArena.init(std.testing.allocator);
    defer arena.deinit();

    const param_types = try arena.dupeSlice(DataType, &.{
        .type_integer,
        .type_text,
    });

    const stmt = DropFunctionStmt{
        .name = "process",
        .param_types = param_types,
        .if_exists = true,
    };

    try std.testing.expectEqualStrings("process", stmt.name);
    try std.testing.expectEqual(@as(usize, 2), stmt.param_types.len);
    try std.testing.expect(stmt.if_exists);
}

test "CreateTriggerStmt row-level AFTER INSERT" {
    const stmt = CreateTriggerStmt{
        .name = "audit_insert",
        .table_name = "users",
        .timing = .after,
        .event = .insert,
        .level = .row,
        .body = "INSERT INTO audit_log VALUES (NEW.id, NOW());",
    };

    try std.testing.expectEqualStrings("audit_insert", stmt.name);
    try std.testing.expectEqualStrings("users", stmt.table_name);
    try std.testing.expectEqual(TriggerTiming.after, stmt.timing);
    try std.testing.expectEqual(TriggerEvent.insert, stmt.event);
    try std.testing.expectEqual(TriggerLevel.row, stmt.level);
    try std.testing.expect(!stmt.or_replace);
}

test "CreateTriggerStmt UPDATE OF with columns" {
    var arena = AstArena.init(std.testing.allocator);
    defer arena.deinit();

    const update_cols = try arena.dupeSlice([]const u8, &.{ "name", "email" });

    const stmt = CreateTriggerStmt{
        .name = "check_update",
        .table_name = "users",
        .timing = .before,
        .event = .update,
        .update_columns = update_cols,
        .level = .row,
        .body = "SELECT validate_email(NEW.email);",
    };

    try std.testing.expectEqual(@as(usize, 2), stmt.update_columns.len);
    try std.testing.expectEqualStrings("name", stmt.update_columns[0]);
    try std.testing.expectEqualStrings("email", stmt.update_columns[1]);
}

test "CreateTriggerStmt INSTEAD OF for views" {
    const stmt = CreateTriggerStmt{
        .name = "view_insert",
        .table_name = "user_view",
        .timing = .instead_of,
        .event = .insert,
        .level = .row,
        .body = "INSERT INTO users (id, name) VALUES (NEW.id, NEW.name);",
    };

    try std.testing.expectEqual(TriggerTiming.instead_of, stmt.timing);
}

test "DropTriggerStmt with table name" {
    const stmt = DropTriggerStmt{
        .name = "audit_trigger",
        .table_name = "users",
        .if_exists = true,
    };

    try std.testing.expectEqualStrings("audit_trigger", stmt.name);
    try std.testing.expectEqualStrings("users", stmt.table_name.?);
    try std.testing.expect(stmt.if_exists);
}

test "AlterTriggerStmt ENABLE" {
    const stmt = AlterTriggerStmt{
        .name = "check_constraint",
        .table_name = "orders",
        .enable = true,
    };

    try std.testing.expectEqualStrings("check_constraint", stmt.name);
    try std.testing.expect(stmt.enable);
}

test "CreateRoleStmt basic" {
    const stmt = CreateRoleStmt{
        .name = "admin",
        .options = .{
            .login = true,
            .superuser = true,
        },
    };

    try std.testing.expectEqualStrings("admin", stmt.name);
    try std.testing.expectEqual(true, stmt.options.login.?);
    try std.testing.expectEqual(true, stmt.options.superuser.?);
}

test "CreateRoleStmt with password" {
    const stmt = CreateRoleStmt{
        .name = "app_user",
        .options = .{
            .login = true,
            .password = "secret123",
            .valid_until = "2025-12-31",
        },
    };

    try std.testing.expectEqualStrings("app_user", stmt.name);
    try std.testing.expectEqualStrings("secret123", stmt.options.password.?);
    try std.testing.expectEqualStrings("2025-12-31", stmt.options.valid_until.?);
}

test "CreateRoleStmt with all options" {
    const stmt = CreateRoleStmt{
        .name = "test_role",
        .options = .{
            .login = true,
            .superuser = false,
            .createdb = true,
            .createrole = false,
            .inherit = true,
            .password = "pass",
            .valid_until = "2026-01-01",
        },
        .or_replace = true,
    };

    try std.testing.expectEqualStrings("test_role", stmt.name);
    try std.testing.expectEqual(true, stmt.options.login.?);
    try std.testing.expectEqual(false, stmt.options.superuser.?);
    try std.testing.expectEqual(true, stmt.options.createdb.?);
    try std.testing.expectEqual(false, stmt.options.createrole.?);
    try std.testing.expectEqual(true, stmt.options.inherit.?);
    try std.testing.expect(stmt.or_replace);
}

test "DropRoleStmt basic" {
    const stmt = DropRoleStmt{
        .name = "old_user",
    };

    try std.testing.expectEqualStrings("old_user", stmt.name);
    try std.testing.expectEqual(false, stmt.if_exists);
}

test "DropRoleStmt with IF EXISTS" {
    const stmt = DropRoleStmt{
        .name = "maybe_role",
        .if_exists = true,
    };

    try std.testing.expectEqualStrings("maybe_role", stmt.name);
    try std.testing.expect(stmt.if_exists);
}

test "AlterRoleStmt modify options" {
    const stmt = AlterRoleStmt{
        .name = "user1",
        .options = .{
            .login = false,
            .password = "new_pass",
        },
    };

    try std.testing.expectEqualStrings("user1", stmt.name);
    try std.testing.expectEqual(false, stmt.options.login.?);
    try std.testing.expectEqualStrings("new_pass", stmt.options.password.?);
}

test "GrantStmt SELECT on table" {
    const privileges = [_]Privilege{.select};
    const stmt = GrantStmt{
        .privileges = &privileges,
        .object_type = .table,
        .object_name = "users",
        .grantee = "alice",
        .with_grant_option = false,
    };

    try std.testing.expectEqual(@as(usize, 1), stmt.privileges.len);
    try std.testing.expectEqual(Privilege.select, stmt.privileges[0]);
    try std.testing.expectEqual(ObjectType.table, stmt.object_type);
    try std.testing.expectEqualStrings("users", stmt.object_name);
    try std.testing.expectEqualStrings("alice", stmt.grantee);
    try std.testing.expectEqual(false, stmt.with_grant_option);
}

test "GrantStmt ALL PRIVILEGES with grant option" {
    const privileges = [_]Privilege{.all};
    const stmt = GrantStmt{
        .privileges = &privileges,
        .object_type = .database,
        .object_name = "mydb",
        .grantee = "admin",
        .with_grant_option = true,
    };

    try std.testing.expectEqual(@as(usize, 1), stmt.privileges.len);
    try std.testing.expectEqual(Privilege.all, stmt.privileges[0]);
    try std.testing.expectEqual(ObjectType.database, stmt.object_type);
    try std.testing.expectEqualStrings("mydb", stmt.object_name);
    try std.testing.expectEqualStrings("admin", stmt.grantee);
    try std.testing.expectEqual(true, stmt.with_grant_option);
}

test "RevokeStmt multiple privileges" {
    const privileges = [_]Privilege{ .select, .insert, .update };
    const stmt = RevokeStmt{
        .privileges = &privileges,
        .object_type = .table,
        .object_name = "orders",
        .grantee = "bob",
    };

    try std.testing.expectEqual(@as(usize, 3), stmt.privileges.len);
    try std.testing.expectEqual(Privilege.select, stmt.privileges[0]);
    try std.testing.expectEqual(Privilege.insert, stmt.privileges[1]);
    try std.testing.expectEqual(Privilege.update, stmt.privileges[2]);
    try std.testing.expectEqual(ObjectType.table, stmt.object_type);
    try std.testing.expectEqualStrings("orders", stmt.object_name);
    try std.testing.expectEqualStrings("bob", stmt.grantee);
}

test "GrantRoleStmt single member" {
    const members = [_][]const u8{"user1"};
    const stmt = GrantRoleStmt{
        .role = "admin",
        .members = &members,
        .with_admin_option = false,
    };

    try std.testing.expectEqualStrings("admin", stmt.role);
    try std.testing.expectEqual(@as(usize, 1), stmt.members.len);
    try std.testing.expectEqualStrings("user1", stmt.members[0]);
    try std.testing.expect(!stmt.with_admin_option);
}

test "GrantRoleStmt multiple members with admin option" {
    const members = [_][]const u8{ "user1", "user2", "user3" };
    const stmt = GrantRoleStmt{
        .role = "editor",
        .members = &members,
        .with_admin_option = true,
    };

    try std.testing.expectEqualStrings("editor", stmt.role);
    try std.testing.expectEqual(@as(usize, 3), stmt.members.len);
    try std.testing.expectEqualStrings("user1", stmt.members[0]);
    try std.testing.expectEqualStrings("user2", stmt.members[1]);
    try std.testing.expectEqualStrings("user3", stmt.members[2]);
    try std.testing.expect(stmt.with_admin_option);
}

test "RevokeRoleStmt single member" {
    const members = [_][]const u8{"user1"};
    const stmt = RevokeRoleStmt{
        .role = "admin",
        .members = &members,
    };

    try std.testing.expectEqualStrings("admin", stmt.role);
    try std.testing.expectEqual(@as(usize, 1), stmt.members.len);
    try std.testing.expectEqualStrings("user1", stmt.members[0]);
}

test "RevokeRoleStmt multiple members" {
    const members = [_][]const u8{ "alice", "bob" };
    const stmt = RevokeRoleStmt{
        .role = "viewer",
        .members = &members,
    };

    try std.testing.expectEqualStrings("viewer", stmt.role);
    try std.testing.expectEqual(@as(usize, 2), stmt.members.len);
    try std.testing.expectEqualStrings("alice", stmt.members[0]);
    try std.testing.expectEqualStrings("bob", stmt.members[1]);
}

test "CreatePolicyStmt default permissive all" {
    const stmt = CreatePolicyStmt{
        .policy_name = "policy1",
        .table_name = "users",
        .using_expr = null,
        .with_check_expr = null,
    };

    try std.testing.expectEqualStrings("policy1", stmt.policy_name);
    try std.testing.expectEqualStrings("users", stmt.table_name);
    try std.testing.expectEqual(PolicyType.permissive, stmt.policy_type);
    try std.testing.expectEqual(PolicyCommand.all, stmt.command);
    try std.testing.expect(stmt.using_expr == null);
    try std.testing.expect(stmt.with_check_expr == null);
}

test "CreatePolicyStmt restrictive select with using" {
    const using_expr = Expr{ .integer_literal = 1 };
    const stmt = CreatePolicyStmt{
        .policy_name = "select_policy",
        .table_name = "documents",
        .policy_type = .restrictive,
        .command = .select,
        .using_expr = using_expr,
        .with_check_expr = null,
    };

    try std.testing.expectEqualStrings("select_policy", stmt.policy_name);
    try std.testing.expectEqualStrings("documents", stmt.table_name);
    try std.testing.expectEqual(PolicyType.restrictive, stmt.policy_type);
    try std.testing.expectEqual(PolicyCommand.select, stmt.command);
    try std.testing.expect(stmt.using_expr != null);
    try std.testing.expect(stmt.with_check_expr == null);
}

test "CreatePolicyStmt insert with check" {
    const check_expr = Expr{ .boolean_literal = true };
    const stmt = CreatePolicyStmt{
        .policy_name = "insert_check",
        .table_name = "posts",
        .command = .insert,
        .with_check_expr = check_expr,
    };

    try std.testing.expectEqualStrings("insert_check", stmt.policy_name);
    try std.testing.expectEqualStrings("posts", stmt.table_name);
    try std.testing.expectEqual(PolicyCommand.insert, stmt.command);
    try std.testing.expect(stmt.using_expr == null);
    try std.testing.expect(stmt.with_check_expr != null);
}

test "DropPolicyStmt without if exists" {
    const stmt = DropPolicyStmt{
        .policy_name = "old_policy",
        .table_name = "accounts",
    };

    try std.testing.expectEqualStrings("old_policy", stmt.policy_name);
    try std.testing.expectEqualStrings("accounts", stmt.table_name);
    try std.testing.expect(!stmt.if_exists);
}

test "DropPolicyStmt with if exists" {
    const stmt = DropPolicyStmt{
        .policy_name = "maybe_policy",
        .table_name = "logs",
        .if_exists = true,
    };

    try std.testing.expectEqualStrings("maybe_policy", stmt.policy_name);
    try std.testing.expectEqualStrings("logs", stmt.table_name);
    try std.testing.expect(stmt.if_exists);
}

test "AlterTableRLSStmt enable" {
    const stmt = AlterTableRLSStmt{
        .table_name = "sensitive_data",
        .enable = true,
    };

    try std.testing.expectEqualStrings("sensitive_data", stmt.table_name);
    try std.testing.expect(stmt.enable);
    try std.testing.expect(!stmt.force);
}

test "AlterTableRLSStmt disable" {
    const stmt = AlterTableRLSStmt{
        .table_name = "public_data",
        .enable = false,
    };

    try std.testing.expectEqualStrings("public_data", stmt.table_name);
    try std.testing.expect(!stmt.enable);
    try std.testing.expect(!stmt.force);
}

test "AlterTableRLSStmt force enable" {
    const stmt = AlterTableRLSStmt{
        .table_name = "audit_log",
        .enable = true,
        .force = true,
    };

    try std.testing.expectEqualStrings("audit_log", stmt.table_name);
    try std.testing.expect(stmt.enable);
    try std.testing.expect(stmt.force);
}

test "AnalyzeStmt with table name" {
    const stmt = AnalyzeStmt{
        .table_name = "users",
    };

    try std.testing.expectEqualStrings("users", stmt.table_name.?);
}

test "AnalyzeStmt all tables" {
    const stmt = AnalyzeStmt{};

    try std.testing.expect(stmt.table_name == null);
}

test "ExplainStmt basic explain" {
    const select_stmt = SelectStmt{
        .columns = &[_]ResultColumn{},
        .from = null,
    };
    const stmt = Stmt{ .select = select_stmt };
    const explain_stmt = ExplainStmt{
        .stmt = &stmt,
        .analyze = false,
    };

    try std.testing.expect(!explain_stmt.analyze);
}

test "ExplainStmt with analyze option" {
    const select_stmt = SelectStmt{
        .columns = &[_]ResultColumn{},
        .from = null,
    };
    const stmt = Stmt{ .select = select_stmt };
    const explain_stmt = ExplainStmt{
        .stmt = &stmt,
        .analyze = true,
    };

    try std.testing.expect(explain_stmt.analyze);
}

test "EXISTS expression basic" {
    const subquery = SelectStmt{
        .columns = &[_]ResultColumn{},
        .from = null,
    };
    const exists_expr = Expr{
        .exists = .{
            .subquery = &subquery,
            .negated = false,
        },
    };

    try std.testing.expect(!exists_expr.exists.negated);
}

test "NOT EXISTS expression" {
    const subquery = SelectStmt{
        .columns = &[_]ResultColumn{},
        .from = null,
    };
    const not_exists_expr = Expr{
        .exists = .{
            .subquery = &subquery,
            .negated = true,
        },
    };

    try std.testing.expect(not_exists_expr.exists.negated);
}
