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
    null_type,

    pub fn fromColumnType(ct: ColumnType) ValueType {
        return switch (ct) {
            .integer => .integer,
            .real => .real,
            .text => .text,
            .blob => .blob,
            .boolean => .boolean,
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
    /// Empty — produces no rows (e.g., for DDL results).
    empty: Empty,

    pub const Scan = struct {
        table: []const u8,
        alias: ?[]const u8 = null,
        columns: []const ColumnRef = &.{},
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

    pub const Join = struct {
        left: *const PlanNode,
        right: *const PlanNode,
        join_type: ast.JoinType = .inner,
        on_condition: ?*const ast.Expr = null,
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
            .create_view => self.planTransaction(), // handled early in engine
            .drop_view => self.planTransaction(), // handled early in engine
            .explain => |s| self.planExplain(s),
        };
    }

    // ── SELECT planning ──────────────────────────────────────────────

    fn planSelect(self: *Planner, stmt: ast.SelectStmt) PlanError!LogicalPlan {
        // Build up the plan bottom-up:
        // CTEs → Scan → Join → Filter → Aggregate → Having → Project → Sort → Limit

        // 0. Plan CTEs (in definition order)
        const alloc = self.arena.allocator();
        var cte_plans = std.ArrayListUnmanaged(CtePlan){};
        for (stmt.ctes) |cte| {
            // Plan each CTE's inner SELECT
            const cte_inner = try self.planSelect(cte.select.*);
            cte_plans.append(alloc, .{
                .name = cte.name,
                .plan = cte_inner.root,
                .column_names = cte.column_names,
            }) catch return error.OutOfMemory;
            // Register CTE name so planTableRef recognizes it
            self.cte_names.put(alloc, cte.name, {}) catch return error.OutOfMemory;
        }

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

        // 4. GROUP BY → Aggregate (also add aggregate node when select contains aggregate functions)
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

        // 6. SELECT columns → Project
        const proj_cols = try self.buildProjectColumns(stmt.columns);
        node = try self.createNode(.{ .project = .{
            .input = node,
            .columns = proj_cols,
        } });

        // 7. ORDER BY → Sort
        if (stmt.order_by.len > 0) {
            node = try self.createNode(.{ .sort = .{
                .input = node,
                .order_by = stmt.order_by,
            } });
        }

        // 8. LIMIT/OFFSET → Limit
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

    // Should be: Sort → Project → Scan
    switch (plan.root.*) {
        .sort => |s| {
            try testing.expectEqual(@as(usize, 1), s.order_by.len);
            switch (s.input.*) {
                .project => {},
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

    // Should be: Limit → Sort → Project → Filter → Scan
    switch (plan.root.*) {
        .limit => |l| {
            switch (l.input.*) {
                .sort => |s| {
                    switch (s.input.*) {
                        .project => |p| {
                            switch (p.input.*) {
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
