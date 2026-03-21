//! Query Planner — translates validated AST into a logical plan (relational algebra tree).
//!
//! The planner converts SQL statements into a tree of logical operators:
//!   SELECT → Project(Filter(Scan(...)))
//!   WHERE  → Filter node wrapping a Scan
//!   JOIN   → Join node combining two child plans
//!   ORDER BY → Sort node
//!   GROUP BY → Aggregate node
//!   LIMIT/OFFSET → Limit node
//!
//! The output is a LogicalPlan tree that can be fed to the optimizer and executor.

const std = @import("std");
const Allocator = std.mem.Allocator;
const ast = @import("ast.zig");
const catalog_mod = @import("catalog.zig");
const analyzer_mod = @import("analyzer.zig");

const ColumnInfo = catalog_mod.ColumnInfo;
const ColumnType = catalog_mod.ColumnType;
const TableInfo = catalog_mod.TableInfo;
const SchemaProvider = analyzer_mod.SchemaProvider;

// ── Value Type ────────────────────────────────────────────────────────

/// Runtime value types in the query engine.
pub const ValueType = enum {
    integer,
    real,
    text,
    blob,
    boolean,
    date,
    time,
    timestamp,
    interval,
    numeric,
    uuid,
    array,
    json,
    jsonb,
    tsvector,
    tsquery,
    null_type,

    pub fn fromColumnType(ct: ColumnType) ValueType {
        return switch (ct) {
            .integer => .integer,
            .real => .real,
            .text => .text,
            .blob => .blob,
            .boolean => .boolean,
            .date => .date,
            .time => .time,
            .timestamp => .timestamp,
            .interval => .interval,
            .numeric => .numeric,
            .uuid => .uuid,
            .array => .array,
            .json => .json,
            .jsonb => .jsonb,
            .tsvector => .tsvector,
            .tsquery => .tsquery,
            .untyped => .text, // default untyped to text
        };
    }

    pub fn toColumnType(self: ValueType) ColumnType {
        return switch (self) {
            .integer => .integer,
            .real => .real,
            .text => .text,
            .blob => .blob,
            .boolean => .boolean,
            .date => .date,
            .time => .time,
            .timestamp => .timestamp,
            .interval => .interval,
            .numeric => .numeric,
            .uuid => .uuid,
            .array => .array,
            .json => .json,
            .jsonb => .jsonb,
            .tsvector => .tsvector,
            .tsquery => .tsquery,
            .null_type => .untyped,
        };
    }
};

// ── Column Reference ──────────────────────────────────────────────────

/// A resolved column reference in the logical plan.
pub const ColumnRef = struct {
    table: []const u8,
    column: []const u8,
    col_type: ValueType = .text,
};

// ── Aggregate Function ────────────────────────────────────────────────

pub const AggFunc = enum {
    count,
    sum,
    avg,
    min,
    max,
    count_star,
};

// ── Logical Plan Node ─────────────────────────────────────────────────

/// A node in the logical plan tree (relational algebra operator).
pub const PlanNode = union(enum) {
    /// Table scan — read all rows from a table.
    scan: Scan,
    /// Table function scan — generate rows from a table-valued function (e.g., unnest).
    table_function_scan: TableFunctionScan,
    /// Filter — apply a predicate (WHERE clause).
    filter: Filter,
    /// Project — select/compute output columns.
    project: Project,
    /// Join — combine rows from two inputs.
    join: Join,
    /// Sort — order rows by specified columns.
    sort: Sort,
    /// Aggregate — group and aggregate rows.
    aggregate: Aggregate,
    /// Limit — restrict output row count.
    limit: Limit,
    /// Values — literal row set (for INSERT VALUES).
    values: Values,
    /// Set operation — UNION, UNION ALL, INTERSECT, EXCEPT.
    set_op: SetOp,
    /// Distinct — eliminate duplicate rows (SELECT DISTINCT / DISTINCT ON).
    distinct: Distinct,
    /// Window — compute window functions over the input rows.
    window: Window,
    /// Empty — produces no rows (e.g., for DDL results).
    empty: Empty,

    pub const SetOp = struct {
        op: ast.SetOpType,
        left: *const PlanNode,
        right: *const PlanNode,
    };

    pub const Scan = struct {
        table: []const u8,
        alias: ?[]const u8 = null,
        columns: []const ColumnRef = &.{},
        /// Index-only scan flag: if true, all required columns are in the index, no heap fetch needed
        index_only: bool = false,
    };

    pub const TableFunctionScan = struct {
        function_name: []const u8,
        args: []const *const ast.Expr,
        alias: ?[]const u8 = null,
    };

    pub const Filter = struct {
        input: *const PlanNode,
        predicate: *const ast.Expr,
    };

    pub const Project = struct {
        input: *const PlanNode,
        columns: []const ProjectColumn,
    };

    pub const ProjectColumn = struct {
        expr: *const ast.Expr,
        alias: ?[]const u8 = null,
    };

    pub const JoinAlgorithm = enum {
        /// Nested loop join — O(n*m) complexity, works for any join condition
        nested_loop,
        /// Hash join — O(n+m) complexity, requires equi-join condition
        hash,
        /// Merge join — O(n+m) complexity, requires sorted inputs on join keys
        merge,
    };

    pub const Join = struct {
        left: *const PlanNode,
        right: *const PlanNode,
        join_type: ast.JoinType = .inner,
        on_condition: ?*const ast.Expr = null,
        /// Join algorithm chosen by optimizer (default: nested_loop)
        algorithm: JoinAlgorithm = .nested_loop,
    };

    pub const Sort = struct {
        input: *const PlanNode,
        order_by: []const ast.OrderByItem,
    };

    pub const Aggregate = struct {
        input: *const PlanNode,
        group_by: []const *const ast.Expr,
        aggregates: []const AggregateExpr = &.{},
    };

    pub const AggregateExpr = struct {
        func: AggFunc,
        arg: ?*const ast.Expr = null,
        alias: ?[]const u8 = null,
        distinct: bool = false,
    };

    pub const Limit = struct {
        input: *const PlanNode,
        limit_expr: ?*const ast.Expr = null,
        offset_expr: ?*const ast.Expr = null,
    };

    pub const Distinct = struct {
        input: *const PlanNode,
        /// For DISTINCT ON: the expressions to compare for uniqueness.
        /// Empty slice means plain DISTINCT (compare all columns).
        on_exprs: []const *const ast.Expr = &.{},
    };

    pub const Window = struct {
        input: *const PlanNode,
        /// Window function expressions from SELECT columns.
        /// Each entry is a pointer to a window_function Expr node.
        funcs: []const *const ast.Expr,
        /// Aliases for each window function (from SELECT AS clause).
        aliases: []const ?[]const u8,
    };

    pub const Values = struct {
        table: []const u8,
        columns: ?[]const []const u8 = null,
        rows: []const []const *const ast.Expr,
    };

    pub const Empty = struct {
        /// What kind of DDL/DML this represents (for display).
        description: []const u8 = "",
    };
};

// ── Logical Plan ──────────────────────────────────────────────────────

/// A planned CTE: name + its logical plan + optional explicit column names.
pub const CtePlan = struct {
    name: []const u8,
    plan: *const PlanNode,
    column_names: []const []const u8 = &.{},
    /// True if this CTE is recursive (WITH RECURSIVE and self-referencing).
    recursive: bool = false,
};

/// The complete logical plan for a SQL statement.
pub const LogicalPlan = struct {
    root: *const PlanNode,
    plan_type: PlanType,
    /// Planned CTEs (in definition order; each may reference earlier ones).
    ctes: []const CtePlan = &.{},
};

pub const PlanType = enum {
    select_query,
    insert,
    update,
    delete,
    create_table,
    drop_table,
    create_index,
    drop_index,
    transaction,
    explain,
};

// ── Planner ───────────────────────────────────────────────────────────

pub const PlanError = error{
    OutOfMemory,
    UnsupportedStatement,
    InvalidPlan,
};

/// Translates a validated AST into a logical plan tree.
pub const Planner = struct {
    arena: *ast.AstArena,
    schema: SchemaProvider,
    /// CTE names currently in scope (for planTableRef to recognize CTE references).
    cte_names: std.StringHashMapUnmanaged(void) = .{},

    pub fn init(arena: *ast.AstArena, schema: SchemaProvider) Planner {
        return .{
            .arena = arena,
            .schema = schema,
        };
    }

    /// Plan a top-level SQL statement.
    pub fn plan(self: *Planner, stmt: ast.Stmt) PlanError!LogicalPlan {
        return switch (stmt) {
            .select => |s| self.planSelect(s),
            .insert => |s| self.planInsert(s),
            .update => |s| self.planUpdate(s),
            .delete => |s| self.planDelete(s),
            .create_table => |s| self.planCreateTable(s),
            .drop_table => |s| self.planDropTable(s),
            .create_index => |s| self.planCreateIndex(s),
            .drop_index => |s| self.planDropIndex(s),
            .transaction => self.planTransaction(),
            .vacuum => self.planTransaction(), // handled early in engine, should never reach planner
            .analyze => self.planTransaction(), // Milestone 20A: ANALYZE command
            .create_view => self.planTransaction(), // handled early in engine
            .drop_view => self.planTransaction(), // handled early in engine
            .create_type => self.planTransaction(), // handled early in engine
            .drop_type => self.planTransaction(), // handled early in engine
            .create_domain => self.planTransaction(), // handled early in engine
            .drop_domain => self.planTransaction(), // handled early in engine
            .create_function => self.planTransaction(), // handled early in engine
            .drop_function => self.planTransaction(), // handled early in engine
            .create_trigger => self.planTransaction(), // TODO: Milestone 14F
            .drop_trigger => self.planTransaction(),   // TODO: Milestone 14F
            .alter_trigger => self.planTransaction(),  // TODO: Milestone 14F
            .create_role => self.planTransaction(),    // TODO: Milestone 17A
            .drop_role => self.planTransaction(),      // TODO: Milestone 17A
            .alter_role => self.planTransaction(),     // TODO: Milestone 17A
            .grant => self.planTransaction(),          // TODO: Milestone 17B
            .revoke => self.planTransaction(),         // TODO: Milestone 17B
            .grant_role => self.planTransaction(),     // TODO: Milestone 17C
            .revoke_role => self.planTransaction(),    // TODO: Milestone 17C
            .create_policy => self.planTransaction(),  // TODO: Milestone 17D
            .drop_policy => self.planTransaction(),    // TODO: Milestone 17D
            .alter_table_rls => self.planTransaction(), // TODO: Milestone 17D
            .reindex => self.planTransaction(), // TODO: Milestone 23
            .explain => |s| self.planExplain(s),
        };
    }

    // ── SELECT planning ──────────────────────────────────────────────

    fn planSelect(self: *Planner, stmt: ast.SelectStmt) PlanError!LogicalPlan {
        // Build up the plan bottom-up:
        // CTEs → SelectBody → [SetOp(body, right_body)] → Sort → Limit

        // 0. Plan CTEs (in definition order)
        const alloc = self.arena.allocator();
        var cte_plans = std.ArrayListUnmanaged(CtePlan){};
        for (stmt.ctes) |cte| {
            // For recursive CTEs, register name BEFORE planning so self-reference resolves
            if (stmt.recursive) {
                self.cte_names.put(alloc, cte.name, {}) catch return error.OutOfMemory;
            }
            // Plan each CTE's inner SELECT
            const cte_inner = try self.planSelect(cte.select.*);
            // A CTE is recursive if WITH RECURSIVE is set and the inner
            // SELECT has a UNION ALL set operation (anchor UNION ALL recursive).
            const is_recursive = stmt.recursive and cte.select.set_operation != null;
            cte_plans.append(alloc, .{
                .name = cte.name,
                .plan = cte_inner.root,
                .column_names = cte.column_names,
                .recursive = is_recursive,
            }) catch return error.OutOfMemory;
            // Register CTE name so later CTEs and the main query recognize it
            if (!stmt.recursive) {
                self.cte_names.put(alloc, cte.name, {}) catch return error.OutOfMemory;
            }
        }

        // 1. Plan the left SELECT body (without Project — see below)
        var node = try self.planSelectBody(&stmt);

        // 2. Handle set operations (UNION, INTERSECT, EXCEPT)
        if (stmt.set_operation) |set_op| {
            // For set ops, both sides need Project before the set op to match column counts
            const left_proj_cols = try self.buildProjectColumns(stmt.columns);
            node = try self.createNode(.{ .project = .{
                .input = node,
                .columns = left_proj_cols,
            } });
            const right_node = try self.planSetOpRight(set_op.right);
            node = try self.createNode(.{ .set_op = .{
                .op = set_op.op,
                .left = node,
                .right = right_node,
            } });
        }

        // 3. ORDER BY → Sort (before Project so it can reference non-selected columns)
        if (stmt.order_by.len > 0) {
            node = try self.createNode(.{ .sort = .{
                .input = node,
                .order_by = stmt.order_by,
            } });
        }

        // 3.5. Project — applied after Sort so ORDER BY can access all columns
        // (for set ops, Project was already applied above before the set op)
        if (stmt.set_operation == null) {
            const proj_cols = try self.buildProjectColumns(stmt.columns);
            node = try self.createNode(.{ .project = .{
                .input = node,
                .columns = proj_cols,
            } });
        }

        // 4. DISTINCT / DISTINCT ON → Distinct (after project, before limit)
        if (stmt.distinct) {
            node = try self.createNode(.{ .distinct = .{
                .input = node,
                .on_exprs = stmt.distinct_on,
            } });
        }

        // 4. LIMIT/OFFSET → Limit
        if (stmt.limit != null or stmt.offset != null) {
            node = try self.createNode(.{ .limit = .{
                .input = node,
                .limit_expr = stmt.limit,
                .offset_expr = stmt.offset,
            } });
        }

        return .{
            .root = node,
            .plan_type = .select_query,
            .ctes = if (cte_plans.items.len > 0)
                cte_plans.toOwnedSlice(alloc) catch return error.OutOfMemory
            else
                &.{},
        };
    }

    /// Plan the body of a SELECT statement (FROM, JOINs, WHERE, GROUP BY, HAVING, Project).
    fn planSelectBody(self: *Planner, stmt: *const ast.SelectStmt) PlanError!*const PlanNode {
        // 1. FROM clause → base scan or joins
        var node: *const PlanNode = if (stmt.from) |from|
            try self.planTableRef(from)
        else
            try self.createNode(.{ .empty = .{ .description = "dual" } });

        // 2. JOINs
        for (stmt.joins) |join_clause| {
            const right = try self.planTableRef(join_clause.table);
            node = try self.createNode(.{ .join = .{
                .left = node,
                .right = right,
                .join_type = join_clause.join_type,
                .on_condition = join_clause.on_condition,
            } });
        }

        // 3. WHERE → Filter
        if (stmt.where) |where_expr| {
            node = try self.createNode(.{ .filter = .{
                .input = node,
                .predicate = where_expr,
            } });
        }

        // 4. GROUP BY → Aggregate
        const aggs = try self.extractAggregates(stmt.columns);
        if (stmt.group_by.len > 0 or aggs.len > 0) {
            node = try self.createNode(.{ .aggregate = .{
                .input = node,
                .group_by = stmt.group_by,
                .aggregates = aggs,
            } });
        }

        // 5. HAVING → Filter on aggregated result
        if (stmt.having) |having_expr| {
            node = try self.createNode(.{ .filter = .{
                .input = node,
                .predicate = having_expr,
            } });
        }

        // 6. Window functions → Window node (before project)
        const win_result = try self.extractWindowFunctions(stmt.columns, stmt.window_defs);
        if (win_result.funcs.len > 0) {
            node = try self.createNode(.{ .window = .{
                .input = node,
                .funcs = win_result.funcs,
                .aliases = win_result.aliases,
            } });
        }

        // Note: Project is NOT added here — it's added in planSelect/planSetOpRight
        // after Sort, so ORDER BY can reference columns not in the SELECT list.

        return node;
    }

    /// Plan the right side of a set operation, recursively handling chained set ops.
    fn planSetOpRight(self: *Planner, stmt: *const ast.SelectStmt) PlanError!*const PlanNode {
        var node = try self.planSelectBody(stmt);

        // Add Project for set op branches (column count must match across sides)
        const proj_cols = try self.buildProjectColumns(stmt.columns);
        node = try self.createNode(.{ .project = .{
            .input = node,
            .columns = proj_cols,
        } });

        // Handle chained set operations on the right side
        if (stmt.set_operation) |set_op| {
            const right_node = try self.planSetOpRight(set_op.right);
            node = try self.createNode(.{ .set_op = .{
                .op = set_op.op,
                .left = node,
                .right = right_node,
            } });
        }

        return node;
    }

    fn planTableRef(self: *Planner, ref: *const ast.TableRef) PlanError!*const PlanNode {
        return switch (ref.*) {
            .table_name => |tn| {
                var columns: []const ColumnRef = &.{};
                const alloc = self.arena.allocator();

                // Check CTE scope first — CTE columns are resolved at execution time
                if (self.cte_names.contains(tn.name)) {
                    return self.createNode(.{ .scan = .{
                        .table = tn.name,
                        .alias = tn.alias,
                        .columns = &.{}, // columns resolved at execution time
                    } });
                }

                // Check for system tables (pg_stat_activity)
                if (std.mem.eql(u8, tn.name, "pg_stat_activity")) {
                    // Define pg_stat_activity column schema
                    var cols = std.ArrayListUnmanaged(ColumnRef){};
                    cols.append(alloc, .{ .table = "pg_stat_activity", .column = "pid", .col_type = .integer }) catch return error.OutOfMemory;
                    cols.append(alloc, .{ .table = "pg_stat_activity", .column = "usename", .col_type = .text }) catch return error.OutOfMemory;
                    cols.append(alloc, .{ .table = "pg_stat_activity", .column = "application_name", .col_type = .text }) catch return error.OutOfMemory;
                    cols.append(alloc, .{ .table = "pg_stat_activity", .column = "client_addr", .col_type = .text }) catch return error.OutOfMemory;
                    cols.append(alloc, .{ .table = "pg_stat_activity", .column = "query", .col_type = .text }) catch return error.OutOfMemory;
                    cols.append(alloc, .{ .table = "pg_stat_activity", .column = "state", .col_type = .text }) catch return error.OutOfMemory;
                    cols.append(alloc, .{ .table = "pg_stat_activity", .column = "query_start", .col_type = .timestamp }) catch return error.OutOfMemory;
                    cols.append(alloc, .{ .table = "pg_stat_activity", .column = "state_change", .col_type = .timestamp }) catch return error.OutOfMemory;
                    columns = cols.toOwnedSlice(alloc) catch return error.OutOfMemory;

                    return self.createNode(.{ .scan = .{
                        .table = tn.name,
                        .alias = tn.alias,
                        .columns = columns,
                    } });
                }

                // Resolve column metadata from schema
                if (self.schema.getTable(alloc, tn.name)) |info| {
                    var cols = std.ArrayListUnmanaged(ColumnRef){};
                    for (info.columns) |col| {
                        cols.append(alloc, .{
                            .table = tn.name,
                            .column = col.name,
                            .col_type = ValueType.fromColumnType(col.column_type),
                        }) catch return error.OutOfMemory;
                    }
                    columns = cols.toOwnedSlice(alloc) catch return error.OutOfMemory;
                }

                return self.createNode(.{ .scan = .{
                    .table = tn.name,
                    .alias = tn.alias,
                    .columns = columns,
                } });
            },
            .subquery => |sq| {
                // Plan the subquery as a nested SELECT
                const sub_plan = try self.planSelect(sq.select.*);
                return sub_plan.root;
            },
            .table_function => |tf| {
                // Table function call — create a special scan node for table-valued functions
                // For now, we'll use a scan node with table name as the function name
                // and store the args for executor to process
                return self.createNode(.{ .table_function_scan = .{
                    .function_name = tf.name,
                    .args = tf.args,
                    .alias = tf.alias,
                } });
            },
        };
    }

    fn buildProjectColumns(self: *Planner, columns: []const ast.ResultColumn) PlanError![]const PlanNode.ProjectColumn {
        const alloc = self.arena.allocator();
        var result = std.ArrayListUnmanaged(PlanNode.ProjectColumn){};

        for (columns) |col| {
            switch (col) {
                .all_columns => {
                    // SELECT * — will be expanded during execution
                    const star = self.arena.create(ast.Expr, .{ .column_ref = .{ .name = "*" } }) catch
                        return error.OutOfMemory;
                    result.append(alloc, .{ .expr = star }) catch return error.OutOfMemory;
                },
                .table_all_columns => |table| {
                    // t.* — will be expanded during execution
                    const star = self.arena.create(ast.Expr, .{
                        .column_ref = .{ .name = "*", .prefix = table },
                    }) catch return error.OutOfMemory;
                    result.append(alloc, .{ .expr = star }) catch return error.OutOfMemory;
                },
                .expr => |e| {
                    result.append(alloc, .{
                        .expr = e.value,
                        .alias = e.alias,
                    }) catch return error.OutOfMemory;
                },
            }
        }

        return result.toOwnedSlice(alloc) catch return error.OutOfMemory;
    }

    fn extractAggregates(self: *Planner, columns: []const ast.ResultColumn) PlanError![]const PlanNode.AggregateExpr {
        const alloc = self.arena.allocator();
        var aggs = std.ArrayListUnmanaged(PlanNode.AggregateExpr){};

        for (columns) |col| {
            switch (col) {
                .expr => |e| {
                    if (self.isAggregateExpr(e.value)) {
                        const agg = self.exprToAggregate(e.value, e.alias);
                        if (agg) |a| {
                            aggs.append(alloc, a) catch return error.OutOfMemory;
                        }
                    }
                },
                else => {},
            }
        }

        return aggs.toOwnedSlice(alloc) catch return error.OutOfMemory;
    }

    const WindowExtractResult = struct {
        funcs: []const *const ast.Expr,
        aliases: []const ?[]const u8,
    };

    fn extractWindowFunctions(self: *Planner, columns: []const ast.ResultColumn, window_defs: []const ast.WindowDef) PlanError!WindowExtractResult {
        const alloc = self.arena.allocator();
        var funcs = std.ArrayListUnmanaged(*const ast.Expr){};
        var aliases = std.ArrayListUnmanaged(?[]const u8){};

        for (columns) |col| {
            switch (col) {
                .expr => |e| {
                    if (e.value.* == .window_function) {
                        const resolved = try self.resolveWindowRef(e.value, window_defs);
                        funcs.append(alloc, resolved) catch return error.OutOfMemory;
                        aliases.append(alloc, e.alias) catch return error.OutOfMemory;
                    }
                },
                else => {},
            }
        }

        return .{
            .funcs = funcs.toOwnedSlice(alloc) catch return error.OutOfMemory,
            .aliases = aliases.toOwnedSlice(alloc) catch return error.OutOfMemory,
        };
    }

    /// Resolve a window function expression that references a named window definition.
    /// If the expression has a window_name, merge the named definition's spec into it.
    fn resolveWindowRef(self: *Planner, expr: *const ast.Expr, window_defs: []const ast.WindowDef) PlanError!*const ast.Expr {
        const wf = expr.window_function;
        const win_name = wf.window_name orelse return expr;

        // Look up the named window definition
        for (window_defs) |def| {
            if (std.mem.eql(u8, def.name, win_name)) {
                // Create a new expression with the named definition's spec merged in.
                // Inline spec (if any) overrides the named definition.
                return self.arena.create(ast.Expr, .{ .window_function = .{
                    .name = wf.name,
                    .args = wf.args,
                    .distinct = wf.distinct,
                    .partition_by = if (wf.partition_by.len > 0) wf.partition_by else def.partition_by,
                    .order_by = if (wf.order_by.len > 0) wf.order_by else def.order_by,
                    .frame = if (wf.frame != null) wf.frame else def.frame,
                    .window_name = null,
                } }) catch return error.OutOfMemory;
            }
        }

        // Named window not found — return as-is (analyzer should catch this)
        return expr;
    }

    fn isAggregateExpr(_: *Planner, expr: *const ast.Expr) bool {
        return switch (expr.*) {
            .function_call => |fc| isAggregateName(fc.name),
            else => false,
        };
    }

    fn exprToAggregate(_: *Planner, expr: *const ast.Expr, alias: ?[]const u8) ?PlanNode.AggregateExpr {
        return switch (expr.*) {
            .function_call => |fc| {
                var func = aggFuncFromName(fc.name) orelse return null;
                // Distinguish COUNT(*) from COUNT(expr)
                if (func == .count and fc.args.len > 0 and
                    fc.args[0].* == .column_ref and std.mem.eql(u8, fc.args[0].column_ref.name, "*"))
                {
                    func = .count_star;
                }
                return .{
                    .func = func,
                    .arg = if (func == .count_star) null else if (fc.args.len > 0) fc.args[0] else null,
                    .alias = alias,
                    .distinct = fc.distinct,
                };
            },
            else => null,
        };
    }

    // ── INSERT planning ──────────────────────────────────────────────

    fn planInsert(self: *Planner, stmt: ast.InsertStmt) PlanError!LogicalPlan {
        const node = try self.createNode(.{ .values = .{
            .table = stmt.table,
            .columns = stmt.columns,
            .rows = stmt.values,
        } });
        return .{ .root = node, .plan_type = .insert };
    }

    // ── UPDATE planning ──────────────────────────────────────────────

    fn planUpdate(self: *Planner, stmt: ast.UpdateStmt) PlanError!LogicalPlan {
        // UPDATE → Scan → Filter(WHERE) → Project(assignments)
        var node: *const PlanNode = try self.createNode(.{ .scan = .{
            .table = stmt.table,
        } });

        if (stmt.where) |where_expr| {
            node = try self.createNode(.{ .filter = .{
                .input = node,
                .predicate = where_expr,
            } });
        }

        // Wrap assignments as project columns
        const alloc = self.arena.allocator();
        var proj_cols = std.ArrayListUnmanaged(PlanNode.ProjectColumn){};
        for (stmt.assignments) |assignment| {
            proj_cols.append(alloc, .{
                .expr = assignment.value,
                .alias = assignment.column,
            }) catch return error.OutOfMemory;
        }

        node = try self.createNode(.{ .project = .{
            .input = node,
            .columns = proj_cols.toOwnedSlice(alloc) catch return error.OutOfMemory,
        } });

        return .{ .root = node, .plan_type = .update };
    }

    // ── DELETE planning ──────────────────────────────────────────────

    fn planDelete(self: *Planner, stmt: ast.DeleteStmt) PlanError!LogicalPlan {
        // DELETE → Scan → Filter(WHERE)
        var node: *const PlanNode = try self.createNode(.{ .scan = .{
            .table = stmt.table,
        } });

        if (stmt.where) |where_expr| {
            node = try self.createNode(.{ .filter = .{
                .input = node,
                .predicate = where_expr,
            } });
        }

        return .{ .root = node, .plan_type = .delete };
    }

    // ── DDL planning (pass-through) ─────────────────────────────────

    fn planCreateTable(self: *Planner, stmt: ast.CreateTableStmt) PlanError!LogicalPlan {
        const desc = std.fmt.allocPrint(self.arena.allocator(), "CREATE TABLE {s}", .{stmt.name}) catch
            return error.OutOfMemory;
        const node = try self.createNode(.{ .empty = .{ .description = desc } });
        return .{ .root = node, .plan_type = .create_table };
    }

    fn planDropTable(self: *Planner, stmt: ast.DropTableStmt) PlanError!LogicalPlan {
        const desc = std.fmt.allocPrint(self.arena.allocator(), "DROP TABLE {s}", .{stmt.name}) catch
            return error.OutOfMemory;
        const node = try self.createNode(.{ .empty = .{ .description = desc } });
        return .{ .root = node, .plan_type = .drop_table };
    }

    fn planCreateIndex(self: *Planner, stmt: ast.CreateIndexStmt) PlanError!LogicalPlan {
        const desc = std.fmt.allocPrint(self.arena.allocator(), "CREATE INDEX {s} ON {s}", .{ stmt.name, stmt.table }) catch
            return error.OutOfMemory;
        const node = try self.createNode(.{ .empty = .{ .description = desc } });
        return .{ .root = node, .plan_type = .create_index };
    }

    fn planDropIndex(self: *Planner, stmt: ast.DropIndexStmt) PlanError!LogicalPlan {
        const desc = std.fmt.allocPrint(self.arena.allocator(), "DROP INDEX {s}", .{stmt.name}) catch
            return error.OutOfMemory;
        const node = try self.createNode(.{ .empty = .{ .description = desc } });
        return .{ .root = node, .plan_type = .drop_index };
    }

    fn planTransaction(self: *Planner) PlanError!LogicalPlan {
        const node = try self.createNode(.{ .empty = .{ .description = "TRANSACTION" } });
        return .{ .root = node, .plan_type = .transaction };
    }

    fn planExplain(self: *Planner, stmt: ast.ExplainStmt) PlanError!LogicalPlan {
        const inner = try self.plan(stmt.stmt.*);
        _ = inner;
        const node = try self.createNode(.{ .empty = .{ .description = "EXPLAIN" } });
        return .{ .root = node, .plan_type = .explain };
    }

    // ── Helper ───────────────────────────────────────────────────────

    fn createNode(self: *Planner, value: PlanNode) PlanError!*const PlanNode {
        return self.arena.create(PlanNode, value) catch return error.OutOfMemory;
    }
};

// ── Aggregate Function Helpers ────────────────────────────────────────

fn isAggregateName(name: []const u8) bool {
    return aggFuncFromName(name) != null;
}

fn aggFuncFromName(name: []const u8) ?AggFunc {
    var buf: [16]u8 = undefined;
    const len = @min(name.len, buf.len);
    for (name[0..len], 0..) |c, i| {
        buf[i] = std.ascii.toLower(c);
    }
    const lower = buf[0..len];

    if (std.mem.eql(u8, lower, "count")) return .count;
    if (std.mem.eql(u8, lower, "sum")) return .sum;
    if (std.mem.eql(u8, lower, "avg")) return .avg;
    if (std.mem.eql(u8, lower, "min")) return .min;
    if (std.mem.eql(u8, lower, "max")) return .max;
    return null;
}

// ── Plan Display (for EXPLAIN) ────────────────────────────────────────

/// Format a logical plan as indented text (for EXPLAIN output).
pub fn formatPlan(node: *const PlanNode, writer: anytype, depth: usize) !void {
    for (0..depth) |_| try writer.writeAll("  ");

    switch (node.*) {
        .scan => |s| {
            try writer.print("Scan: {s}", .{s.table});
            if (s.alias) |a| try writer.print(" AS {s}", .{a});
            try writer.writeAll("\n");
        },
        .table_function_scan => |tfs| {
            try writer.print("TableFunction: {s}({d} args)", .{ tfs.function_name, tfs.args.len });
            if (tfs.alias) |a| try writer.print(" AS {s}", .{a});
            try writer.writeAll("\n");
        },
        .filter => |f| {
            try writer.writeAll("Filter\n");
            try formatPlan(f.input, writer, depth + 1);
        },
        .project => |p| {
            try writer.print("Project ({d} columns)\n", .{p.columns.len});
            try formatPlan(p.input, writer, depth + 1);
        },
        .join => |j| {
            try writer.print("Join ({s})\n", .{@tagName(j.join_type)});
            try formatPlan(j.left, writer, depth + 1);
            try formatPlan(j.right, writer, depth + 1);
        },
        .sort => |s| {
            try writer.print("Sort ({d} keys)\n", .{s.order_by.len});
            try formatPlan(s.input, writer, depth + 1);
        },
        .aggregate => |a| {
            try writer.print("Aggregate ({d} groups, {d} aggs)\n", .{ a.group_by.len, a.aggregates.len });
            try formatPlan(a.input, writer, depth + 1);
        },
        .limit => |l| {
            try writer.writeAll("Limit");
            if (l.offset_expr != null) try writer.writeAll("+Offset");
            try writer.writeAll("\n");
            try formatPlan(l.input, writer, depth + 1);
        },
        .set_op => |s| {
            const op_name = switch (s.op) {
                .@"union" => "Union",
                .union_all => "Union All",
                .intersect => "Intersect",
                .except => "Except",
            };
            try writer.print("SetOp: {s}\n", .{op_name});
            try formatPlan(s.left, writer, depth + 1);
            try formatPlan(s.right, writer, depth + 1);
        },
        .distinct => |d| {
            if (d.on_exprs.len > 0) {
                try writer.print("Distinct On ({d} exprs)\n", .{d.on_exprs.len});
            } else {
                try writer.writeAll("Distinct\n");
            }
            try formatPlan(d.input, writer, depth + 1);
        },
        .window => |w| {
            try writer.print("Window ({d} funcs)\n", .{w.funcs.len});
            try formatPlan(w.input, writer, depth + 1);
        },
        .values => |v| {
            try writer.print("Values: {s} ({d} rows)\n", .{ v.table, v.rows.len });
        },
        .empty => |e| {
            try writer.print("Empty: {s}\n", .{e.description});
        },
    }
}

// ── Tests ─────────────────────────────────────────────────────────────

const testing = std.testing;
const parser_mod = @import("parser.zig");

/// Test helper: create a MemorySchema with a "users" table.
fn testSchema(alloc: Allocator) analyzer_mod.MemorySchema {
    var schema = analyzer_mod.MemorySchema.init(alloc);
    schema.addTable("users", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{ .primary_key = true, .not_null = true } },
        .{ .name = "name", .column_type = .text, .flags = .{ .not_null = true } },
        .{ .name = "email", .column_type = .text, .flags = .{} },
        .{ .name = "age", .column_type = .integer, .flags = .{} },
    });
    schema.addTable("orders", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{ .primary_key = true, .not_null = true } },
        .{ .name = "user_id", .column_type = .integer, .flags = .{ .not_null = true } },
        .{ .name = "amount", .column_type = .real, .flags = .{} },
    });
    return schema;
}

/// Helper: parse SQL and plan it.
fn parseAndPlan(alloc: Allocator, sql: []const u8, arena: *ast.AstArena, schema: *analyzer_mod.MemorySchema) !LogicalPlan {
    var p = try parser_mod.Parser.init(alloc, sql, arena);
    defer p.deinit();

    const stmt = try p.parseStatement();
    if (stmt == null) return error.InvalidPlan;

    var planner = Planner.init(arena, schema.provider());
    return planner.plan(stmt.?);
}

test "plan simple SELECT" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator, "SELECT * FROM users;", &arena, &schema);
    try testing.expectEqual(PlanType.select_query, plan.plan_type);

    // Should be: Project → Scan
    switch (plan.root.*) {
        .project => |p| {
            try testing.expect(p.columns.len >= 1);
            switch (p.input.*) {
                .scan => |s| try testing.expectEqualStrings("users", s.table),
                else => return error.InvalidPlan,
            }
        },
        else => return error.InvalidPlan,
    }
}

test "plan SELECT with WHERE" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator, "SELECT id, name FROM users WHERE age > 18;", &arena, &schema);

    // Should be: Project → Filter → Scan
    switch (plan.root.*) {
        .project => |p| {
            try testing.expectEqual(@as(usize, 2), p.columns.len);
            switch (p.input.*) {
                .filter => |f| {
                    switch (f.input.*) {
                        .scan => |s| try testing.expectEqualStrings("users", s.table),
                        else => return error.InvalidPlan,
                    }
                },
                else => return error.InvalidPlan,
            }
        },
        else => return error.InvalidPlan,
    }
}

test "plan SELECT with ORDER BY" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator, "SELECT * FROM users ORDER BY name ASC;", &arena, &schema);

    // Should be: Project → Sort → Scan (Sort before Project so ORDER BY can access all columns)
    switch (plan.root.*) {
        .project => |p| {
            switch (p.input.*) {
                .sort => |s| {
                    try testing.expectEqual(@as(usize, 1), s.order_by.len);
                },
                else => return error.InvalidPlan,
            }
        },
        else => return error.InvalidPlan,
    }
}

test "plan SELECT with LIMIT" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator, "SELECT * FROM users LIMIT 10;", &arena, &schema);

    // Should be: Limit → Project → Scan
    switch (plan.root.*) {
        .limit => |l| {
            try testing.expect(l.limit_expr != null);
            try testing.expect(l.offset_expr == null);
        },
        else => return error.InvalidPlan,
    }
}

test "plan SELECT with LIMIT and OFFSET" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator, "SELECT * FROM users LIMIT 10 OFFSET 5;", &arena, &schema);

    switch (plan.root.*) {
        .limit => |l| {
            try testing.expect(l.limit_expr != null);
            try testing.expect(l.offset_expr != null);
        },
        else => return error.InvalidPlan,
    }
}

test "plan SELECT with JOIN" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(
        testing.allocator,
        "SELECT u.name, o.amount FROM users u JOIN orders o ON u.id = o.user_id;",
        &arena,
        &schema,
    );

    // Should be: Project → Join → (Scan, Scan)
    switch (plan.root.*) {
        .project => |p| {
            try testing.expectEqual(@as(usize, 2), p.columns.len);
            switch (p.input.*) {
                .join => |j| {
                    try testing.expectEqual(ast.JoinType.inner, j.join_type);
                    try testing.expect(j.on_condition != null);
                    switch (j.left.*) {
                        .scan => |s| try testing.expectEqualStrings("users", s.table),
                        else => return error.InvalidPlan,
                    }
                    switch (j.right.*) {
                        .scan => |s| try testing.expectEqualStrings("orders", s.table),
                        else => return error.InvalidPlan,
                    }
                },
                else => return error.InvalidPlan,
            }
        },
        else => return error.InvalidPlan,
    }
}

test "plan SELECT with GROUP BY" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(
        testing.allocator,
        "SELECT name, COUNT(id) FROM users GROUP BY name;",
        &arena,
        &schema,
    );

    // Should be: Project → Aggregate → Scan
    switch (plan.root.*) {
        .project => |proj| {
            switch (proj.input.*) {
                .aggregate => |a| {
                    try testing.expectEqual(@as(usize, 1), a.group_by.len);
                    try testing.expectEqual(@as(usize, 1), a.aggregates.len);
                    try testing.expectEqual(AggFunc.count, a.aggregates[0].func);
                },
                else => return error.InvalidPlan,
            }
        },
        else => return error.InvalidPlan,
    }
}

test "plan SELECT with WHERE, ORDER BY, LIMIT" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(
        testing.allocator,
        "SELECT id, name FROM users WHERE age > 21 ORDER BY name LIMIT 5;",
        &arena,
        &schema,
    );

    // Should be: Limit → Project → Sort → Filter → Scan
    switch (plan.root.*) {
        .limit => |l| {
            switch (l.input.*) {
                .project => |p| {
                    switch (p.input.*) {
                        .sort => |s| {
                            switch (s.input.*) {
                                .filter => |f| {
                                    switch (f.input.*) {
                                        .scan => {},
                                        else => return error.InvalidPlan,
                                    }
                                },
                                else => return error.InvalidPlan,
                            }
                        },
                        else => return error.InvalidPlan,
                    }
                },
                else => return error.InvalidPlan,
            }
        },
        else => return error.InvalidPlan,
    }
}

test "plan INSERT" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(
        testing.allocator,
        "INSERT INTO users (id, name) VALUES (1, 'Alice');",
        &arena,
        &schema,
    );

    try testing.expectEqual(PlanType.insert, plan.plan_type);
    switch (plan.root.*) {
        .values => |v| {
            try testing.expectEqualStrings("users", v.table);
            try testing.expectEqual(@as(usize, 2), v.columns.?.len);
            try testing.expectEqual(@as(usize, 1), v.rows.len);
        },
        else => return error.InvalidPlan,
    }
}

test "plan UPDATE" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(
        testing.allocator,
        "UPDATE users SET name = 'Bob' WHERE id = 1;",
        &arena,
        &schema,
    );

    try testing.expectEqual(PlanType.update, plan.plan_type);
    // Should be: Project → Filter → Scan
    switch (plan.root.*) {
        .project => |p| {
            try testing.expectEqual(@as(usize, 1), p.columns.len);
            try testing.expectEqualStrings("name", p.columns[0].alias.?);
            switch (p.input.*) {
                .filter => |f| {
                    switch (f.input.*) {
                        .scan => |s| try testing.expectEqualStrings("users", s.table),
                        else => return error.InvalidPlan,
                    }
                },
                else => return error.InvalidPlan,
            }
        },
        else => return error.InvalidPlan,
    }
}

test "plan DELETE" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(
        testing.allocator,
        "DELETE FROM users WHERE id = 1;",
        &arena,
        &schema,
    );

    try testing.expectEqual(PlanType.delete, plan.plan_type);
    // Should be: Filter → Scan
    switch (plan.root.*) {
        .filter => |f| {
            switch (f.input.*) {
                .scan => |s| try testing.expectEqualStrings("users", s.table),
                else => return error.InvalidPlan,
            }
        },
        else => return error.InvalidPlan,
    }
}

test "plan DELETE without WHERE" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator, "DELETE FROM users;", &arena, &schema);

    try testing.expectEqual(PlanType.delete, plan.plan_type);
    switch (plan.root.*) {
        .scan => |s| try testing.expectEqualStrings("users", s.table),
        else => return error.InvalidPlan,
    }
}

test "plan CREATE TABLE" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(
        testing.allocator,
        "CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT);",
        &arena,
        &schema,
    );

    try testing.expectEqual(PlanType.create_table, plan.plan_type);
    switch (plan.root.*) {
        .empty => |e| try testing.expect(std.mem.indexOf(u8, e.description, "products") != null),
        else => return error.InvalidPlan,
    }
}

test "plan DROP TABLE" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator, "DROP TABLE users;", &arena, &schema);

    try testing.expectEqual(PlanType.drop_table, plan.plan_type);
    switch (plan.root.*) {
        .empty => |e| try testing.expect(std.mem.indexOf(u8, e.description, "users") != null),
        else => return error.InvalidPlan,
    }
}

test "plan CREATE INDEX" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(
        testing.allocator,
        "CREATE INDEX idx_name ON users (name);",
        &arena,
        &schema,
    );

    try testing.expectEqual(PlanType.create_index, plan.plan_type);
    switch (plan.root.*) {
        .empty => |e| {
            try testing.expect(std.mem.indexOf(u8, e.description, "idx_name") != null);
            try testing.expect(std.mem.indexOf(u8, e.description, "users") != null);
        },
        else => return error.InvalidPlan,
    }
}

test "plan DROP INDEX" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator, "DROP INDEX idx_name;", &arena, &schema);
    try testing.expectEqual(PlanType.drop_index, plan.plan_type);
}

test "plan BEGIN" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator, "BEGIN;", &arena, &schema);
    try testing.expectEqual(PlanType.transaction, plan.plan_type);
}

test "plan COMMIT" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator, "COMMIT;", &arena, &schema);
    try testing.expectEqual(PlanType.transaction, plan.plan_type);
}

test "plan LEFT JOIN" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(
        testing.allocator,
        "SELECT * FROM users LEFT JOIN orders ON users.id = orders.user_id;",
        &arena,
        &schema,
    );

    switch (plan.root.*) {
        .project => |p| {
            switch (p.input.*) {
                .join => |j| try testing.expectEqual(ast.JoinType.left, j.join_type),
                else => return error.InvalidPlan,
            }
        },
        else => return error.InvalidPlan,
    }
}

test "formatPlan simple select" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator, "SELECT * FROM users;", &arena, &schema);

    var buf: [1024]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    try formatPlan(plan.root, &w, 0);
    const output = fbs.getWritten();
    try testing.expect(std.mem.indexOf(u8, output, "Project") != null);
    try testing.expect(std.mem.indexOf(u8, output, "Scan: users") != null);
}

test "formatPlan with filter and sort" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(
        testing.allocator,
        "SELECT name FROM users WHERE age > 18 ORDER BY name;",
        &arena,
        &schema,
    );

    var buf: [2048]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    try formatPlan(plan.root, &w, 0);
    const output = fbs.getWritten();
    try testing.expect(std.mem.indexOf(u8, output, "Sort") != null);
    try testing.expect(std.mem.indexOf(u8, output, "Filter") != null);
    try testing.expect(std.mem.indexOf(u8, output, "Scan") != null);
}

test "plan SELECT with aggregate functions" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(
        testing.allocator,
        "SELECT name, SUM(age), AVG(age) FROM users GROUP BY name;",
        &arena,
        &schema,
    );

    switch (plan.root.*) {
        .project => |proj| {
            switch (proj.input.*) {
                .aggregate => |a| {
                    try testing.expectEqual(@as(usize, 1), a.group_by.len);
                    try testing.expectEqual(@as(usize, 2), a.aggregates.len);
                    try testing.expectEqual(AggFunc.sum, a.aggregates[0].func);
                    try testing.expectEqual(AggFunc.avg, a.aggregates[1].func);
                },
                else => return error.InvalidPlan,
            }
        },
        else => return error.InvalidPlan,
    }
}

test "plan SELECT without FROM" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator, "SELECT 1 + 2;", &arena, &schema);

    switch (plan.root.*) {
        .project => |p| {
            try testing.expectEqual(@as(usize, 1), p.columns.len);
            switch (p.input.*) {
                .empty => {},
                else => return error.InvalidPlan,
            }
        },
        else => return error.InvalidPlan,
    }
}

test "plan SELECT with HAVING" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(
        testing.allocator,
        "SELECT name, COUNT(id) FROM users GROUP BY name HAVING COUNT(id) > 1;",
        &arena,
        &schema,
    );

    // Should be: Project → Filter(HAVING) → Aggregate → Scan
    switch (plan.root.*) {
        .project => |proj| {
            switch (proj.input.*) {
                .filter => |f| {
                    switch (f.input.*) {
                        .aggregate => {},
                        else => return error.InvalidPlan,
                    }
                },
                else => return error.InvalidPlan,
            }
        },
        else => return error.InvalidPlan,
    }
}

test "plan INSERT with multiple rows" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(
        testing.allocator,
        "INSERT INTO users (id, name) VALUES (1, 'Alice'), (2, 'Bob');",
        &arena,
        &schema,
    );

    try testing.expectEqual(PlanType.insert, plan.plan_type);
    switch (plan.root.*) {
        .values => |v| {
            try testing.expectEqual(@as(usize, 2), v.rows.len);
        },
        else => return error.InvalidPlan,
    }
}

test "plan UPDATE without WHERE" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator, "UPDATE users SET name = 'X';", &arena, &schema);

    try testing.expectEqual(PlanType.update, plan.plan_type);
    // Should be: Project → Scan (no filter)
    switch (plan.root.*) {
        .project => |p| {
            switch (p.input.*) {
                .scan => {},
                else => return error.InvalidPlan,
            }
        },
        else => return error.InvalidPlan,
    }
}

test "ValueType.fromColumnType" {
    try testing.expectEqual(ValueType.integer, ValueType.fromColumnType(.integer));
    try testing.expectEqual(ValueType.real, ValueType.fromColumnType(.real));
    try testing.expectEqual(ValueType.text, ValueType.fromColumnType(.text));
    try testing.expectEqual(ValueType.blob, ValueType.fromColumnType(.blob));
    try testing.expectEqual(ValueType.boolean, ValueType.fromColumnType(.boolean));
    try testing.expectEqual(ValueType.text, ValueType.fromColumnType(.untyped));
}

test "ValueType.toColumnType" {
    try testing.expectEqual(ColumnType.integer, ValueType.integer.toColumnType());
    try testing.expectEqual(ColumnType.real, ValueType.real.toColumnType());
    try testing.expectEqual(ColumnType.text, ValueType.text.toColumnType());
    try testing.expectEqual(ColumnType.blob, ValueType.blob.toColumnType());
    try testing.expectEqual(ColumnType.boolean, ValueType.boolean.toColumnType());
    try testing.expectEqual(ColumnType.untyped, ValueType.null_type.toColumnType());
}

test "ValueType fromColumnType and toColumnType roundtrip" {
    // integer -> ValueType.integer -> ColumnType.integer
    try testing.expectEqual(ColumnType.integer, ValueType.fromColumnType(.integer).toColumnType());
    try testing.expectEqual(ColumnType.real, ValueType.fromColumnType(.real).toColumnType());
    try testing.expectEqual(ColumnType.text, ValueType.fromColumnType(.text).toColumnType());
    try testing.expectEqual(ColumnType.blob, ValueType.fromColumnType(.blob).toColumnType());
    try testing.expectEqual(ColumnType.boolean, ValueType.fromColumnType(.boolean).toColumnType());
    // untyped -> text -> text (not a round-trip back to untyped, by design)
    try testing.expectEqual(ColumnType.text, ValueType.fromColumnType(.untyped).toColumnType());
}

test "aggFuncFromName" {
    try testing.expectEqual(AggFunc.count, aggFuncFromName("COUNT").?);
    try testing.expectEqual(AggFunc.sum, aggFuncFromName("SUM").?);
    try testing.expectEqual(AggFunc.avg, aggFuncFromName("avg").?);
    try testing.expectEqual(AggFunc.min, aggFuncFromName("Min").?);
    try testing.expectEqual(AggFunc.max, aggFuncFromName("MAX").?);
    try testing.expect(aggFuncFromName("unknown") == null);
}

test "Scan with schema columns" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator, "SELECT * FROM users;", &arena, &schema);

    switch (plan.root.*) {
        .project => |p| {
            switch (p.input.*) {
                .scan => |s| {
                    try testing.expectEqual(@as(usize, 4), s.columns.len);
                    try testing.expectEqualStrings("id", s.columns[0].column);
                    try testing.expectEqualStrings("name", s.columns[1].column);
                    try testing.expectEqual(ValueType.integer, s.columns[0].col_type);
                    try testing.expectEqual(ValueType.text, s.columns[1].col_type);
                },
                else => return error.InvalidPlan,
            }
        },
        else => return error.InvalidPlan,
    }
}

// ── CTE Planning Tests ──────────────────────────────────────────────

test "plan simple CTE" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "WITH cte AS (SELECT 1) SELECT * FROM cte;",
        &arena, &schema);

    // Plan should have one CTE
    try testing.expectEqual(@as(usize, 1), plan.ctes.len);
    try testing.expectEqualStrings("cte", plan.ctes[0].name);

    // Main query scans the CTE (no schema columns — resolved at execution)
    switch (plan.root.*) {
        .project => |p| {
            switch (p.input.*) {
                .scan => |s| {
                    try testing.expectEqualStrings("cte", s.table);
                    try testing.expectEqual(@as(usize, 0), s.columns.len);
                },
                else => return error.InvalidPlan,
            }
        },
        else => return error.InvalidPlan,
    }
}

test "plan multiple CTEs" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "WITH a AS (SELECT 1), b AS (SELECT 2) SELECT * FROM a;",
        &arena, &schema);

    try testing.expectEqual(@as(usize, 2), plan.ctes.len);
    try testing.expectEqualStrings("a", plan.ctes[0].name);
    try testing.expectEqualStrings("b", plan.ctes[1].name);
}

test "plan CTE with column aliases" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "WITH cte(x, y) AS (SELECT 1, 2) SELECT * FROM cte;",
        &arena, &schema);

    try testing.expectEqual(@as(usize, 1), plan.ctes.len);
    try testing.expectEqual(@as(usize, 2), plan.ctes[0].column_names.len);
    try testing.expectEqualStrings("x", plan.ctes[0].column_names[0]);
    try testing.expectEqualStrings("y", plan.ctes[0].column_names[1]);
}

test "plan CTE referencing real table" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "WITH active_users AS (SELECT id, name FROM users) SELECT * FROM active_users;",
        &arena, &schema);

    try testing.expectEqual(@as(usize, 1), plan.ctes.len);
    try testing.expectEqualStrings("active_users", plan.ctes[0].name);

    // CTE inner plan should scan the real table
    switch (plan.ctes[0].plan.*) {
        .project => |p| {
            switch (p.input.*) {
                .scan => |s| try testing.expectEqualStrings("users", s.table),
                else => return error.InvalidPlan,
            }
        },
        else => return error.InvalidPlan,
    }
}

// ── Set Operation Planning Tests ─────────────────────────────────────

test "plan UNION" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "SELECT id FROM users UNION SELECT id FROM orders;",
        &arena, &schema);

    // Root should be SetOp(union, Project(Scan(users)), Project(Scan(orders)))
    switch (plan.root.*) {
        .set_op => |s| {
            try testing.expectEqual(ast.SetOpType.@"union", s.op);
            switch (s.left.*) {
                .project => |p| {
                    switch (p.input.*) {
                        .scan => |sc| try testing.expectEqualStrings("users", sc.table),
                        else => return error.InvalidPlan,
                    }
                },
                else => return error.InvalidPlan,
            }
            switch (s.right.*) {
                .project => |p| {
                    switch (p.input.*) {
                        .scan => |sc| try testing.expectEqualStrings("orders", sc.table),
                        else => return error.InvalidPlan,
                    }
                },
                else => return error.InvalidPlan,
            }
        },
        else => return error.InvalidPlan,
    }
}

test "plan UNION ALL" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "SELECT id FROM users UNION ALL SELECT id FROM orders;",
        &arena, &schema);

    switch (plan.root.*) {
        .set_op => |s| try testing.expectEqual(ast.SetOpType.union_all, s.op),
        else => return error.InvalidPlan,
    }
}

test "plan INTERSECT" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "SELECT id FROM users INTERSECT SELECT id FROM orders;",
        &arena, &schema);

    switch (plan.root.*) {
        .set_op => |s| try testing.expectEqual(ast.SetOpType.intersect, s.op),
        else => return error.InvalidPlan,
    }
}

test "plan EXCEPT" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "SELECT id FROM users EXCEPT SELECT id FROM orders;",
        &arena, &schema);

    switch (plan.root.*) {
        .set_op => |s| try testing.expectEqual(ast.SetOpType.except, s.op),
        else => return error.InvalidPlan,
    }
}

test "plan UNION with ORDER BY and LIMIT" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "SELECT id FROM users UNION SELECT id FROM orders ORDER BY id LIMIT 5;",
        &arena, &schema);

    // Should be: Limit → Sort → SetOp(union, ...)
    switch (plan.root.*) {
        .limit => |l| {
            switch (l.input.*) {
                .sort => |s| {
                    switch (s.input.*) {
                        .set_op => |so| try testing.expectEqual(ast.SetOpType.@"union", so.op),
                        else => return error.InvalidPlan,
                    }
                },
                else => return error.InvalidPlan,
            }
        },
        else => return error.InvalidPlan,
    }
}

test "plan chained set operations" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "SELECT id FROM users UNION SELECT id FROM orders EXCEPT SELECT id FROM users;",
        &arena, &schema);

    // Root: SetOp(union, left, SetOp(except, ...))
    switch (plan.root.*) {
        .set_op => |s| {
            try testing.expectEqual(ast.SetOpType.@"union", s.op);
            switch (s.right.*) {
                .set_op => |inner| {
                    try testing.expectEqual(ast.SetOpType.except, inner.op);
                },
                else => return error.InvalidPlan,
            }
        },
        else => return error.InvalidPlan,
    }
}

test "formatPlan set operation" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "SELECT id FROM users UNION ALL SELECT id FROM orders;",
        &arena, &schema);

    var buf: [2048]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    try formatPlan(plan.root, &w, 0);
    const output = fbs.getWritten();
    try testing.expect(std.mem.indexOf(u8, output, "SetOp: Union All") != null);
    try testing.expect(std.mem.indexOf(u8, output, "Scan: users") != null);
    try testing.expect(std.mem.indexOf(u8, output, "Scan: orders") != null);
}

test "plan SELECT DISTINCT" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "SELECT DISTINCT name FROM users;",
        &arena, &schema);

    // Should be: Distinct → Project → Scan
    switch (plan.root.*) {
        .distinct => |d| {
            try testing.expectEqual(@as(usize, 0), d.on_exprs.len);
            switch (d.input.*) {
                .project => |p| {
                    try testing.expectEqual(@as(usize, 1), p.columns.len);
                },
                else => return error.InvalidPlan,
            }
        },
        else => return error.InvalidPlan,
    }
}

test "plan SELECT DISTINCT with ORDER BY" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "SELECT DISTINCT name, age FROM users ORDER BY name;",
        &arena, &schema);

    // Should be: Distinct → Project → Sort → Scan
    switch (plan.root.*) {
        .distinct => |d| {
            try testing.expectEqual(@as(usize, 0), d.on_exprs.len);
            switch (d.input.*) {
                .project => |p| {
                    switch (p.input.*) {
                        .sort => |s| {
                            try testing.expectEqual(@as(usize, 1), s.order_by.len);
                        },
                        else => return error.InvalidPlan,
                    }
                },
                else => return error.InvalidPlan,
            }
        },
        else => return error.InvalidPlan,
    }
}

test "plan SELECT DISTINCT ON" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "SELECT DISTINCT ON (name) name, age FROM users ORDER BY name, age;",
        &arena, &schema);

    // Should be: Distinct On → Project → Sort → Scan
    switch (plan.root.*) {
        .distinct => |d| {
            try testing.expectEqual(@as(usize, 1), d.on_exprs.len);
            switch (d.input.*) {
                .project => |p| {
                    switch (p.input.*) {
                        .sort => {},
                        else => return error.InvalidPlan,
                    }
                },
                else => return error.InvalidPlan,
            }
        },
        else => return error.InvalidPlan,
    }
}

test "plan DISTINCT with LIMIT" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "SELECT DISTINCT name FROM users LIMIT 5;",
        &arena, &schema);

    // Should be: Limit → Distinct → Project → Scan
    switch (plan.root.*) {
        .limit => |l| {
            switch (l.input.*) {
                .distinct => |d| {
                    try testing.expectEqual(@as(usize, 0), d.on_exprs.len);
                },
                else => return error.InvalidPlan,
            }
        },
        else => return error.InvalidPlan,
    }
}

test "formatPlan distinct" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "SELECT DISTINCT name FROM users;",
        &arena, &schema);

    var buf: [2048]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    try formatPlan(plan.root, &w, 0);
    const output = fbs.getWritten();
    try testing.expect(std.mem.indexOf(u8, output, "Distinct\n") != null);
}

test "formatPlan distinct on" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "SELECT DISTINCT ON (name) name, age FROM users ORDER BY name;",
        &arena, &schema);

    var buf: [2048]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var w = fbs.writer();
    try formatPlan(plan.root, &w, 0);
    const output = fbs.getWritten();
    try testing.expect(std.mem.indexOf(u8, output, "Distinct On (1 exprs)") != null);
}

test "plan recursive CTE" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "WITH RECURSIVE cnt(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM cnt WHERE x < 5) SELECT * FROM cnt;",
        &arena, &schema);

    // Should have one CTE marked as recursive
    try testing.expectEqual(@as(usize, 1), plan.ctes.len);
    try testing.expectEqualStrings("cnt", plan.ctes[0].name);
    try testing.expect(plan.ctes[0].recursive);
    try testing.expectEqual(@as(usize, 1), plan.ctes[0].column_names.len);
    try testing.expectEqualStrings("x", plan.ctes[0].column_names[0]);

    // Plan root should be a set_op (UNION ALL) with anchor and recursive parts
    switch (plan.ctes[0].plan.*) {
        .set_op => |s| {
            try testing.expectEqual(ast.SetOpType.union_all, s.op);
        },
        else => return error.InvalidPlan,
    }
}

test "plan non-recursive CTE with WITH RECURSIVE keyword" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    // WITH RECURSIVE without UNION ALL — CTE is not actually recursive
    const plan = try parseAndPlan(testing.allocator,
        "WITH RECURSIVE cte AS (SELECT 1) SELECT * FROM cte;",
        &arena, &schema);

    try testing.expectEqual(@as(usize, 1), plan.ctes.len);
    try testing.expect(!plan.ctes[0].recursive); // No set_operation → not recursive
}

// ── Stored Function Planning Tests ───────────────────────────────────

test "plan CREATE FUNCTION scalar" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "CREATE FUNCTION add(x INTEGER, y INTEGER) RETURNS INTEGER LANGUAGE sfl AS 'RETURN x + y;';",
        &arena, &schema);

    // CREATE FUNCTION is a DDL statement handled in engine, returns transaction plan
    try testing.expectEqual(PlanType.transaction, plan.plan_type);
}

test "plan CREATE FUNCTION with OR REPLACE" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "CREATE OR REPLACE FUNCTION add(x INTEGER) RETURNS INTEGER LANGUAGE sfl AS 'RETURN x;';",
        &arena, &schema);

    try testing.expectEqual(PlanType.transaction, plan.plan_type);
}

test "plan CREATE FUNCTION table return" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "CREATE FUNCTION get_users() RETURNS TABLE (id INTEGER, name TEXT) LANGUAGE sfl AS 'SELECT 1, ''test'';';",
        &arena, &schema);

    try testing.expectEqual(PlanType.transaction, plan.plan_type);
}

test "plan CREATE FUNCTION setof return" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "CREATE FUNCTION generate_series(start INTEGER, stop INTEGER) RETURNS SETOF INTEGER LANGUAGE sfl AS 'RETURN start;';",
        &arena, &schema);

    try testing.expectEqual(PlanType.transaction, plan.plan_type);
}

test "plan CREATE FUNCTION with volatility IMMUTABLE" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "CREATE FUNCTION pi() RETURNS REAL LANGUAGE sfl IMMUTABLE AS 'RETURN 3.14159;';",
        &arena, &schema);

    try testing.expectEqual(PlanType.transaction, plan.plan_type);
}

test "plan CREATE FUNCTION with volatility STABLE" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "CREATE FUNCTION current_user() RETURNS TEXT LANGUAGE sfl STABLE AS 'RETURN ''admin'';';",
        &arena, &schema);

    try testing.expectEqual(PlanType.transaction, plan.plan_type);
}

test "plan CREATE FUNCTION with volatility VOLATILE" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "CREATE FUNCTION random() RETURNS REAL LANGUAGE sfl VOLATILE AS 'RETURN 0.5;';",
        &arena, &schema);

    try testing.expectEqual(PlanType.transaction, plan.plan_type);
}

test "plan DROP FUNCTION simple" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "DROP FUNCTION add;",
        &arena, &schema);

    try testing.expectEqual(PlanType.transaction, plan.plan_type);
}

test "plan DROP FUNCTION with parameter types" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "DROP FUNCTION add(INTEGER, INTEGER);",
        &arena, &schema);

    try testing.expectEqual(PlanType.transaction, plan.plan_type);
}

test "plan DROP FUNCTION IF EXISTS" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "DROP FUNCTION IF EXISTS nonexistent;",
        &arena, &schema);

    try testing.expectEqual(PlanType.transaction, plan.plan_type);
}

// ── Milestone 14F: Trigger Planner Tests ──────────────────────────────

test "plan CREATE TRIGGER BEFORE INSERT" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "CREATE TRIGGER audit_insert BEFORE INSERT ON users FOR EACH ROW AS 'INSERT INTO audit VALUES (NEW.id)';",
        &arena, &schema);

    try testing.expectEqual(PlanType.transaction, plan.plan_type);
}

test "plan CREATE TRIGGER AFTER UPDATE" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "CREATE TRIGGER update_timestamp AFTER UPDATE ON products FOR EACH ROW AS 'UPDATE products SET modified_at = NOW() WHERE id = NEW.id';",
        &arena, &schema);

    try testing.expectEqual(PlanType.transaction, plan.plan_type);
}

test "plan CREATE TRIGGER BEFORE DELETE" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "CREATE TRIGGER archive_user BEFORE DELETE ON users FOR EACH ROW AS 'INSERT INTO archived_users SELECT * FROM users WHERE id = OLD.id';",
        &arena, &schema);

    try testing.expectEqual(PlanType.transaction, plan.plan_type);
}

test "plan CREATE TRIGGER with UPDATE OF columns" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "CREATE TRIGGER price_change AFTER UPDATE OF price, discount ON products FOR EACH ROW AS 'INSERT INTO price_history VALUES (OLD.price, NEW.price)';",
        &arena, &schema);

    try testing.expectEqual(PlanType.transaction, plan.plan_type);
}

test "plan CREATE TRIGGER with WHEN clause" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "CREATE TRIGGER high_salary_alert AFTER INSERT ON employees FOR EACH ROW WHEN (NEW.salary > 100000) AS 'INSERT INTO alerts VALUES (NEW.id)';",
        &arena, &schema);

    try testing.expectEqual(PlanType.transaction, plan.plan_type);
}

test "plan CREATE TRIGGER statement-level" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "CREATE TRIGGER check_batch AFTER INSERT ON orders FOR EACH STATEMENT AS 'SELECT validate_batch()';",
        &arena, &schema);

    try testing.expectEqual(PlanType.transaction, plan.plan_type);
}

test "plan CREATE TRIGGER INSTEAD OF (for views)" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "CREATE TRIGGER view_insert INSTEAD OF INSERT ON user_view FOR EACH ROW AS 'INSERT INTO users VALUES (NEW.id, NEW.name)';",
        &arena, &schema);

    try testing.expectEqual(PlanType.transaction, plan.plan_type);
}

test "plan CREATE TRIGGER TRUNCATE event" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "CREATE TRIGGER log_truncate AFTER TRUNCATE ON sensitive_data FOR EACH STATEMENT AS 'INSERT INTO security_log VALUES (NOW())';",
        &arena, &schema);

    try testing.expectEqual(PlanType.transaction, plan.plan_type);
}

test "plan DROP TRIGGER simple" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "DROP TRIGGER audit_insert ON users;",
        &arena, &schema);

    try testing.expectEqual(PlanType.transaction, plan.plan_type);
}

test "plan DROP TRIGGER IF EXISTS" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "DROP TRIGGER IF EXISTS nonexistent ON users;",
        &arena, &schema);

    try testing.expectEqual(PlanType.transaction, plan.plan_type);
}

test "plan ALTER TRIGGER ENABLE" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "ALTER TRIGGER audit_insert ON users ENABLE;",
        &arena, &schema);

    try testing.expectEqual(PlanType.transaction, plan.plan_type);
}

test "plan ALTER TRIGGER DISABLE" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "ALTER TRIGGER update_timestamp ON products DISABLE;",
        &arena, &schema);

    try testing.expectEqual(PlanType.transaction, plan.plan_type);
}

test "plan GRANT SELECT" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "GRANT SELECT ON TABLE users TO alice;",
        &arena, &schema);

    try testing.expectEqual(PlanType.transaction, plan.plan_type);
}

test "plan GRANT multiple privileges" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "GRANT SELECT, INSERT, UPDATE ON TABLE products TO bob;",
        &arena, &schema);

    try testing.expectEqual(PlanType.transaction, plan.plan_type);
}

test "plan GRANT ALL PRIVILEGES" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "GRANT ALL PRIVILEGES ON TABLE admin_data TO superuser;",
        &arena, &schema);

    try testing.expectEqual(PlanType.transaction, plan.plan_type);
}

test "plan GRANT with WITH GRANT OPTION" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "GRANT DELETE ON TABLE logs TO manager WITH GRANT OPTION;",
        &arena, &schema);

    try testing.expectEqual(PlanType.transaction, plan.plan_type);
}

test "plan REVOKE SELECT" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "REVOKE SELECT ON TABLE users FROM alice;",
        &arena, &schema);

    try testing.expectEqual(PlanType.transaction, plan.plan_type);
}

test "plan REVOKE multiple privileges" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "REVOKE INSERT, UPDATE, DELETE ON TABLE restricted FROM user1;",
        &arena, &schema);

    try testing.expectEqual(PlanType.transaction, plan.plan_type);
}

test "plan REVOKE ALL PRIVILEGES" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "REVOKE ALL PRIVILEGES ON TABLE sensitive FROM guest;",
        &arena, &schema);

    try testing.expectEqual(PlanType.transaction, plan.plan_type);
}

test "plan CREATE POLICY basic SELECT" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "CREATE POLICY view_policy ON users FOR SELECT USING (id = current_user_id());",
        &arena, &schema);

    try testing.expectEqual(PlanType.transaction, plan.plan_type);
}

test "plan CREATE POLICY PERMISSIVE" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "CREATE POLICY allow_read ON data AS PERMISSIVE FOR SELECT USING (public = true);",
        &arena, &schema);

    try testing.expectEqual(PlanType.transaction, plan.plan_type);
}

test "plan CREATE POLICY RESTRICTIVE" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "CREATE POLICY block_sensitive ON logs AS RESTRICTIVE FOR ALL USING (level != 'DEBUG');",
        &arena, &schema);

    try testing.expectEqual(PlanType.transaction, plan.plan_type);
}

test "plan CREATE POLICY INSERT with WITH CHECK" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "CREATE POLICY insert_check ON posts FOR INSERT WITH CHECK (author_id = current_user_id());",
        &arena, &schema);

    try testing.expectEqual(PlanType.transaction, plan.plan_type);
}

test "plan CREATE POLICY UPDATE with both clauses" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "CREATE POLICY update_own ON comments FOR UPDATE USING (user_id = current_user()) WITH CHECK (user_id = current_user());",
        &arena, &schema);

    try testing.expectEqual(PlanType.transaction, plan.plan_type);
}

test "plan CREATE POLICY ALL command" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "CREATE POLICY all_access ON public_table FOR ALL USING (true);",
        &arena, &schema);

    try testing.expectEqual(PlanType.transaction, plan.plan_type);
}

test "plan DROP POLICY simple" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "DROP POLICY old_policy ON users;",
        &arena, &schema);

    try testing.expectEqual(PlanType.transaction, plan.plan_type);
}

test "plan DROP POLICY IF EXISTS" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "DROP POLICY IF EXISTS maybe_policy ON accounts;",
        &arena, &schema);

    try testing.expectEqual(PlanType.transaction, plan.plan_type);
}

test "plan ALTER TABLE ENABLE RLS" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "ALTER TABLE sensitive_data ENABLE ROW LEVEL SECURITY;",
        &arena, &schema);

    try testing.expectEqual(PlanType.transaction, plan.plan_type);
}

test "plan ALTER TABLE DISABLE RLS" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "ALTER TABLE public_data DISABLE ROW LEVEL SECURITY;",
        &arena, &schema);

    try testing.expectEqual(PlanType.transaction, plan.plan_type);
}

test "plan ALTER TABLE FORCE RLS" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "ALTER TABLE admin_logs FORCE ROW LEVEL SECURITY;",
        &arena, &schema);

    try testing.expectEqual(PlanType.transaction, plan.plan_type);
}

test "plan ALTER TABLE NO FORCE RLS" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try parseAndPlan(testing.allocator,
        "ALTER TABLE normal_table NO FORCE ROW LEVEL SECURITY;",
        &arena, &schema);

    try testing.expectEqual(PlanType.transaction, plan.plan_type);
}
