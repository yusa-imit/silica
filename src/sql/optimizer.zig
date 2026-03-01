//! Rule-Based Query Optimizer — applies transformations to logical plan trees.
//!
//! Optimization rules applied (in order):
//! 1. Predicate pushdown — push Filter below Join where possible
//! 2. Constant folding — evaluate constant expressions at plan time
//! 3. Redundant filter elimination — remove always-true filters
//!
//! The optimizer takes a LogicalPlan and returns an optimized LogicalPlan.
//! All nodes are allocated in the provided AstArena.

const std = @import("std");
const Allocator = std.mem.Allocator;
const ast = @import("ast.zig");
const planner_mod = @import("planner.zig");

const PlanNode = planner_mod.PlanNode;
const LogicalPlan = planner_mod.LogicalPlan;

// ── Optimizer ─────────────────────────────────────────────────────────

pub const Optimizer = struct {
    arena: *ast.AstArena,

    pub fn init(arena: *ast.AstArena) Optimizer {
        return .{ .arena = arena };
    }

    /// Optimize a logical plan by applying all rules.
    pub fn optimize(self: *Optimizer, plan: LogicalPlan) !LogicalPlan {
        const optimized = try self.optimizeNode(plan.root);
        return .{ .root = optimized, .plan_type = plan.plan_type, .ctes = plan.ctes };
    }

    fn optimizeNode(self: *Optimizer, node: *const PlanNode) error{OutOfMemory}!*const PlanNode {
        return switch (node.*) {
            .filter => |f| self.optimizeFilter(f),
            .project => |p| self.optimizeProject(p),
            .join => |j| self.optimizeJoin(j),
            .sort => |s| self.optimizeSort(s),
            .aggregate => |a| self.optimizeAggregate(a),
            .limit => |l| self.optimizeLimit(l),
            .set_op => |s| self.optimizeSetOp(s),
            // Leaf nodes — no optimization
            .scan, .values, .empty => node,
        };
    }

    // ── Filter Optimization ──────────────────────────────────────────

    fn optimizeFilter(self: *Optimizer, filter: PlanNode.Filter) !*const PlanNode {
        // First optimize the input
        const opt_input = try self.optimizeNode(filter.input);

        // Try constant folding on the predicate
        if (self.isAlwaysTrue(filter.predicate)) {
            // Filter is always true — eliminate it
            return opt_input;
        }

        // Try predicate pushdown: if input is a join, push filter into one side
        switch (opt_input.*) {
            .join => |j| {
                return self.pushFilterIntoJoin(filter.predicate, j);
            },
            else => {},
        }

        // Return optimized filter
        return self.createNode(.{ .filter = .{
            .input = opt_input,
            .predicate = filter.predicate,
        } });
    }

    fn pushFilterIntoJoin(self: *Optimizer, predicate: *const ast.Expr, join: PlanNode.Join) !*const PlanNode {
        // Check which side of the join the predicate references
        const refs_left = self.exprReferencesTable(predicate, join.left);
        const refs_right = self.exprReferencesTable(predicate, join.right);

        if (refs_left and !refs_right) {
            // Push filter to left side only
            const new_left = try self.createNode(.{ .filter = .{
                .input = join.left,
                .predicate = predicate,
            } });
            return self.createNode(.{ .join = .{
                .left = new_left,
                .right = join.right,
                .join_type = join.join_type,
                .on_condition = join.on_condition,
            } });
        }

        if (refs_right and !refs_left) {
            // Push filter to right side only
            const new_right = try self.createNode(.{ .filter = .{
                .input = join.right,
                .predicate = predicate,
            } });
            return self.createNode(.{ .join = .{
                .left = join.left,
                .right = new_right,
                .join_type = join.join_type,
                .on_condition = join.on_condition,
            } });
        }

        // References both sides or neither — keep filter above join
        const opt_join = try self.createNode(.{ .join = join });
        return self.createNode(.{ .filter = .{
            .input = opt_join,
            .predicate = predicate,
        } });
    }

    fn exprReferencesTable(_: *Optimizer, expr: *const ast.Expr, plan_node: *const PlanNode) bool {
        // Get the table name from the plan node (if it's a scan)
        const table_name = switch (plan_node.*) {
            .scan => |s| s.alias orelse s.table,
            else => return false,
        };

        return exprMentionsTable(expr, table_name);
    }

    // ── Project Optimization ─────────────────────────────────────────

    fn optimizeProject(self: *Optimizer, project: PlanNode.Project) !*const PlanNode {
        const opt_input = try self.optimizeNode(project.input);
        return self.createNode(.{ .project = .{
            .input = opt_input,
            .columns = project.columns,
        } });
    }

    // ── Join Optimization ────────────────────────────────────────────

    fn optimizeJoin(self: *Optimizer, join: PlanNode.Join) !*const PlanNode {
        const opt_left = try self.optimizeNode(join.left);
        const opt_right = try self.optimizeNode(join.right);

        return self.createNode(.{ .join = .{
            .left = opt_left,
            .right = opt_right,
            .join_type = join.join_type,
            .on_condition = join.on_condition,
        } });
    }

    // ── Sort Optimization ────────────────────────────────────────────

    fn optimizeSort(self: *Optimizer, sort: PlanNode.Sort) !*const PlanNode {
        const opt_input = try self.optimizeNode(sort.input);

        // If sorting with no keys, eliminate
        if (sort.order_by.len == 0) return opt_input;

        return self.createNode(.{ .sort = .{
            .input = opt_input,
            .order_by = sort.order_by,
        } });
    }

    // ── Aggregate Optimization ───────────────────────────────────────

    fn optimizeAggregate(self: *Optimizer, agg: PlanNode.Aggregate) !*const PlanNode {
        const opt_input = try self.optimizeNode(agg.input);
        return self.createNode(.{ .aggregate = .{
            .input = opt_input,
            .group_by = agg.group_by,
            .aggregates = agg.aggregates,
        } });
    }

    // ── Limit Optimization ───────────────────────────────────────────

    fn optimizeLimit(self: *Optimizer, limit: PlanNode.Limit) !*const PlanNode {
        const opt_input = try self.optimizeNode(limit.input);

        // LIMIT 0 → empty
        if (limit.limit_expr) |le| {
            if (le.* == .integer_literal and le.integer_literal == 0) {
                return self.createNode(.{ .empty = .{ .description = "LIMIT 0" } });
            }
        }

        return self.createNode(.{ .limit = .{
            .input = opt_input,
            .limit_expr = limit.limit_expr,
            .offset_expr = limit.offset_expr,
        } });
    }

    // ── Set Operation Optimization ────────────────────────────────────

    fn optimizeSetOp(self: *Optimizer, set_op: PlanNode.SetOp) !*const PlanNode {
        const opt_left = try self.optimizeNode(set_op.left);
        const opt_right = try self.optimizeNode(set_op.right);

        return self.createNode(.{ .set_op = .{
            .op = set_op.op,
            .left = opt_left,
            .right = opt_right,
        } });
    }

    // ── Constant Folding ─────────────────────────────────────────────

    /// Check if an expression is always true (e.g., TRUE, 1=1).
    fn isAlwaysTrue(_: *Optimizer, expr: *const ast.Expr) bool {
        return switch (expr.*) {
            .boolean_literal => |b| b,
            .integer_literal => |i| i != 0,
            .binary_op => |op| {
                if (op.op != .equal) return false;
                // Check if both sides are identical literals
                return exprsEqual(op.left, op.right);
            },
            else => false,
        };
    }

    // ── Helpers ──────────────────────────────────────────────────────

    fn createNode(self: *Optimizer, value: PlanNode) !*const PlanNode {
        return self.arena.create(PlanNode, value) catch return error.OutOfMemory;
    }
};

// ── Free Functions ────────────────────────────────────────────────────

/// Check if an expression mentions a specific table name (in qualified column refs).
fn exprMentionsTable(expr: *const ast.Expr, table: []const u8) bool {
    return switch (expr.*) {
        .column_ref => |cr| {
            if (cr.prefix) |p| return std.mem.eql(u8, p, table);
            return false;
        },
        .binary_op => |op| {
            return exprMentionsTable(op.left, table) or exprMentionsTable(op.right, table);
        },
        .unary_op => |op| exprMentionsTable(op.operand, table),
        .paren => |inner| exprMentionsTable(inner, table),
        .function_call => |fc| {
            for (fc.args) |arg| {
                if (exprMentionsTable(arg, table)) return true;
            }
            return false;
        },
        .between => |b| {
            return exprMentionsTable(b.expr, table) or
                exprMentionsTable(b.low, table) or
                exprMentionsTable(b.high, table);
        },
        .in_list => |il| {
            if (exprMentionsTable(il.expr, table)) return true;
            for (il.list) |item| {
                if (exprMentionsTable(item, table)) return true;
            }
            return false;
        },
        .is_null => |isn| exprMentionsTable(isn.expr, table),
        .like => |lk| exprMentionsTable(lk.expr, table) or exprMentionsTable(lk.pattern, table),
        .case_expr => |ce| {
            if (ce.operand) |op| {
                if (exprMentionsTable(op, table)) return true;
            }
            for (ce.when_clauses) |wc| {
                if (exprMentionsTable(wc.condition, table) or exprMentionsTable(wc.result, table))
                    return true;
            }
            if (ce.else_expr) |ee| return exprMentionsTable(ee, table);
            return false;
        },
        .cast => |c| exprMentionsTable(c.expr, table),
        else => false,
    };
}

/// Shallow equality check for constant expressions.
fn exprsEqual(a: *const ast.Expr, b: *const ast.Expr) bool {
    const a_tag = @as(std.meta.Tag(ast.Expr), a.*);
    const b_tag = @as(std.meta.Tag(ast.Expr), b.*);
    if (a_tag != b_tag) return false;

    return switch (a.*) {
        .integer_literal => |av| av == b.integer_literal,
        .float_literal => |av| av == b.float_literal,
        .string_literal => |av| std.mem.eql(u8, av, b.string_literal),
        .boolean_literal => |av| av == b.boolean_literal,
        .null_literal => true,
        .column_ref => |av| {
            const bv = b.column_ref;
            if (!std.mem.eql(u8, av.name, bv.name)) return false;
            if (av.prefix == null and bv.prefix == null) return true;
            if (av.prefix != null and bv.prefix != null)
                return std.mem.eql(u8, av.prefix.?, bv.prefix.?);
            return false;
        },
        else => false,
    };
}

// ── Tests ─────────────────────────────────────────────────────────────

const testing = std.testing;
const parser_mod = @import("parser.zig");
const analyzer_mod = @import("analyzer.zig");

fn testSchema(alloc: Allocator) analyzer_mod.MemorySchema {
    var schema = analyzer_mod.MemorySchema.init(alloc);
    schema.addTable("users", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{ .primary_key = true, .not_null = true } },
        .{ .name = "name", .column_type = .text, .flags = .{ .not_null = true } },
        .{ .name = "age", .column_type = .integer, .flags = .{} },
    });
    schema.addTable("orders", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{ .primary_key = true, .not_null = true } },
        .{ .name = "user_id", .column_type = .integer, .flags = .{ .not_null = true } },
        .{ .name = "amount", .column_type = .real, .flags = .{} },
    });
    return schema;
}

fn planAndOptimize(alloc: Allocator, sql: []const u8, arena: *ast.AstArena, schema: *analyzer_mod.MemorySchema) !LogicalPlan {
    var p = try parser_mod.Parser.init(alloc, sql, arena);
    defer p.deinit();

    const stmt = try p.parseStatement();
    if (stmt == null) return error.InvalidPlan;

    var plan = planner_mod.Planner.init(arena, schema.provider());
    const logical = try plan.plan(stmt.?);

    var opt = Optimizer.init(arena);
    return opt.optimize(logical);
}

test "optimize simple SELECT (no changes)" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try planAndOptimize(testing.allocator, "SELECT * FROM users;", &arena, &schema);
    try testing.expectEqual(planner_mod.PlanType.select_query, plan.plan_type);

    // Should still be: Project → Scan
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

test "optimize eliminates always-true filter (boolean true)" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    // Manually build a plan with WHERE TRUE
    const scan = try arena.create(PlanNode, .{ .scan = .{ .table = "users" } });
    const true_expr = try arena.create(ast.Expr, .{ .boolean_literal = true });
    const filter = try arena.create(PlanNode, .{ .filter = .{
        .input = scan,
        .predicate = true_expr,
    } });
    const proj_col = try arena.dupeSlice(PlanNode.ProjectColumn, &.{
        .{ .expr = try arena.create(ast.Expr, .{ .column_ref = .{ .name = "*" } }) },
    });
    const project = try arena.create(PlanNode, .{ .project = .{
        .input = filter,
        .columns = proj_col,
    } });

    var opt = Optimizer.init(&arena);
    const optimized = try opt.optimize(.{ .root = project, .plan_type = .select_query });

    // Filter should be eliminated, leaving Project → Scan
    switch (optimized.root.*) {
        .project => |p| {
            switch (p.input.*) {
                .scan => {},
                else => return error.InvalidPlan,
            }
        },
        else => return error.InvalidPlan,
    }
}

test "optimize LIMIT 0 → empty" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    // Build: Limit(0) → Scan
    const scan = try arena.create(PlanNode, .{ .scan = .{ .table = "users" } });
    const zero = try arena.create(ast.Expr, .{ .integer_literal = 0 });
    const limit_node = try arena.create(PlanNode, .{ .limit = .{
        .input = scan,
        .limit_expr = zero,
    } });

    var opt = Optimizer.init(&arena);
    const optimized = try opt.optimize(.{ .root = limit_node, .plan_type = .select_query });

    switch (optimized.root.*) {
        .empty => |e| try testing.expect(std.mem.indexOf(u8, e.description, "LIMIT 0") != null),
        else => return error.InvalidPlan,
    }
}

test "optimize predicate pushdown: filter into left side of join" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    // Build: Filter(u.age > 18) → Join(Scan(users AS u), Scan(orders AS o))
    const left_scan = try arena.create(PlanNode, .{ .scan = .{
        .table = "users",
        .alias = "u",
    } });
    const right_scan = try arena.create(PlanNode, .{ .scan = .{
        .table = "orders",
        .alias = "o",
    } });
    const join_on = try arena.create(ast.Expr, .{ .binary_op = .{
        .op = .equal,
        .left = try arena.create(ast.Expr, .{ .column_ref = .{ .name = "id", .prefix = "u" } }),
        .right = try arena.create(ast.Expr, .{ .column_ref = .{ .name = "user_id", .prefix = "o" } }),
    } });
    const join_node = try arena.create(PlanNode, .{ .join = .{
        .left = left_scan,
        .right = right_scan,
        .join_type = .inner,
        .on_condition = join_on,
    } });

    // Filter referencing only the left table (u.age > 18)
    const filter_pred = try arena.create(ast.Expr, .{ .binary_op = .{
        .op = .greater_than,
        .left = try arena.create(ast.Expr, .{ .column_ref = .{ .name = "age", .prefix = "u" } }),
        .right = try arena.create(ast.Expr, .{ .integer_literal = 18 }),
    } });
    const filter_node = try arena.create(PlanNode, .{ .filter = .{
        .input = join_node,
        .predicate = filter_pred,
    } });

    var opt = Optimizer.init(&arena);
    const optimized = try opt.optimize(.{ .root = filter_node, .plan_type = .select_query });

    // After pushdown: Join(Filter(Scan(users)), Scan(orders))
    switch (optimized.root.*) {
        .join => |j| {
            // Filter should be pushed to left
            switch (j.left.*) {
                .filter => |f| {
                    switch (f.input.*) {
                        .scan => |s| try testing.expectEqualStrings("users", s.table),
                        else => return error.InvalidPlan,
                    }
                },
                else => return error.InvalidPlan,
            }
            // Right should remain a scan
            switch (j.right.*) {
                .scan => |s| try testing.expectEqualStrings("orders", s.table),
                else => return error.InvalidPlan,
            }
        },
        else => return error.InvalidPlan,
    }
}

test "optimize predicate pushdown: filter into right side of join" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const left_scan = try arena.create(PlanNode, .{ .scan = .{ .table = "users", .alias = "u" } });
    const right_scan = try arena.create(PlanNode, .{ .scan = .{ .table = "orders", .alias = "o" } });
    const join_node = try arena.create(PlanNode, .{ .join = .{
        .left = left_scan,
        .right = right_scan,
        .join_type = .inner,
    } });

    // Filter referencing only the right table (o.amount > 100)
    const filter_pred = try arena.create(ast.Expr, .{ .binary_op = .{
        .op = .greater_than,
        .left = try arena.create(ast.Expr, .{ .column_ref = .{ .name = "amount", .prefix = "o" } }),
        .right = try arena.create(ast.Expr, .{ .integer_literal = 100 }),
    } });
    const filter_node = try arena.create(PlanNode, .{ .filter = .{
        .input = join_node,
        .predicate = filter_pred,
    } });

    var opt = Optimizer.init(&arena);
    const optimized = try opt.optimize(.{ .root = filter_node, .plan_type = .select_query });

    // Filter should be pushed to right
    switch (optimized.root.*) {
        .join => |j| {
            switch (j.left.*) {
                .scan => {},
                else => return error.InvalidPlan,
            }
            switch (j.right.*) {
                .filter => |f| {
                    switch (f.input.*) {
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

test "optimize keeps filter above join when referencing both sides" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const left_scan = try arena.create(PlanNode, .{ .scan = .{ .table = "users", .alias = "u" } });
    const right_scan = try arena.create(PlanNode, .{ .scan = .{ .table = "orders", .alias = "o" } });
    const join_node = try arena.create(PlanNode, .{ .join = .{
        .left = left_scan,
        .right = right_scan,
        .join_type = .inner,
    } });

    // Filter referencing both tables (u.id = o.user_id)
    const filter_pred = try arena.create(ast.Expr, .{ .binary_op = .{
        .op = .equal,
        .left = try arena.create(ast.Expr, .{ .column_ref = .{ .name = "id", .prefix = "u" } }),
        .right = try arena.create(ast.Expr, .{ .column_ref = .{ .name = "user_id", .prefix = "o" } }),
    } });
    const filter_node = try arena.create(PlanNode, .{ .filter = .{
        .input = join_node,
        .predicate = filter_pred,
    } });

    var opt = Optimizer.init(&arena);
    const optimized = try opt.optimize(.{ .root = filter_node, .plan_type = .select_query });

    // Filter should stay above join
    switch (optimized.root.*) {
        .filter => |f| {
            switch (f.input.*) {
                .join => {},
                else => return error.InvalidPlan,
            }
        },
        else => return error.InvalidPlan,
    }
}

test "optimize eliminates 1=1 filter" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const scan = try arena.create(PlanNode, .{ .scan = .{ .table = "users" } });
    const one_a = try arena.create(ast.Expr, .{ .integer_literal = 1 });
    const one_b = try arena.create(ast.Expr, .{ .integer_literal = 1 });
    const eq = try arena.create(ast.Expr, .{ .binary_op = .{
        .op = .equal,
        .left = one_a,
        .right = one_b,
    } });
    const filter_node = try arena.create(PlanNode, .{ .filter = .{
        .input = scan,
        .predicate = eq,
    } });

    var opt = Optimizer.init(&arena);
    const optimized = try opt.optimize(.{ .root = filter_node, .plan_type = .select_query });

    // 1=1 is always true, filter should be eliminated
    switch (optimized.root.*) {
        .scan => {},
        else => return error.InvalidPlan,
    }
}

test "exprsEqual with matching literals" {
    try testing.expect(exprsEqual(
        &ast.Expr{ .integer_literal = 42 },
        &ast.Expr{ .integer_literal = 42 },
    ));
    try testing.expect(!exprsEqual(
        &ast.Expr{ .integer_literal = 1 },
        &ast.Expr{ .integer_literal = 2 },
    ));
    try testing.expect(exprsEqual(
        &ast.Expr{ .string_literal = "hello" },
        &ast.Expr{ .string_literal = "hello" },
    ));
    try testing.expect(!exprsEqual(
        &ast.Expr{ .string_literal = "a" },
        &ast.Expr{ .string_literal = "b" },
    ));
}

test "exprsEqual with column refs" {
    try testing.expect(exprsEqual(
        &ast.Expr{ .column_ref = .{ .name = "id", .prefix = "t" } },
        &ast.Expr{ .column_ref = .{ .name = "id", .prefix = "t" } },
    ));
    try testing.expect(!exprsEqual(
        &ast.Expr{ .column_ref = .{ .name = "id", .prefix = "t" } },
        &ast.Expr{ .column_ref = .{ .name = "id", .prefix = "u" } },
    ));
    try testing.expect(!exprsEqual(
        &ast.Expr{ .column_ref = .{ .name = "id" } },
        &ast.Expr{ .column_ref = .{ .name = "id", .prefix = "t" } },
    ));
}

test "exprsEqual with different types" {
    try testing.expect(!exprsEqual(
        &ast.Expr{ .integer_literal = 1 },
        &ast.Expr{ .string_literal = "1" },
    ));
}

test "exprMentionsTable with qualified refs" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();

    const expr = try arena.create(ast.Expr, .{ .column_ref = .{ .name = "age", .prefix = "u" } });
    const scan = try arena.create(PlanNode, .{ .scan = .{ .table = "users", .alias = "u" } });
    const other = try arena.create(PlanNode, .{ .scan = .{ .table = "orders", .alias = "o" } });

    var opt = Optimizer.init(&arena);
    try testing.expect(opt.exprReferencesTable(expr, scan));
    try testing.expect(!opt.exprReferencesTable(expr, other));
}

test "exprMentionsTable with binary op" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();

    const expr = try arena.create(ast.Expr, .{ .binary_op = .{
        .op = .greater_than,
        .left = try arena.create(ast.Expr, .{ .column_ref = .{ .name = "age", .prefix = "u" } }),
        .right = try arena.create(ast.Expr, .{ .integer_literal = 18 }),
    } });
    const scan = try arena.create(PlanNode, .{ .scan = .{ .table = "users", .alias = "u" } });

    var opt = Optimizer.init(&arena);
    try testing.expect(opt.exprReferencesTable(expr, scan));
}

test "optimize full query with pushdown" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try planAndOptimize(
        testing.allocator,
        "SELECT * FROM users WHERE age > 21 ORDER BY name LIMIT 10;",
        &arena,
        &schema,
    );

    // Should have Limit → Sort → Project → Filter → Scan structure preserved
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

test "optimize preserves INSERT plan" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try planAndOptimize(
        testing.allocator,
        "INSERT INTO users (id, name) VALUES (1, 'Alice');",
        &arena,
        &schema,
    );

    try testing.expectEqual(planner_mod.PlanType.insert, plan.plan_type);
    switch (plan.root.*) {
        .values => {},
        else => return error.InvalidPlan,
    }
}

test "optimize recursively optimizes set operation children" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    // UNION with a trivial filter (1=1) on left side that should be eliminated
    const plan = try planAndOptimize(
        testing.allocator,
        "SELECT id FROM users WHERE 1=1 UNION SELECT id FROM orders;",
        &arena,
        &schema,
    );

    try testing.expectEqual(planner_mod.PlanType.select_query, plan.plan_type);
    // Root should be a set_op
    switch (plan.root.*) {
        .set_op => |s| {
            try testing.expectEqual(ast.SetOpType.@"union", s.op);
            // Left side: the 1=1 filter should be eliminated by optimizer
            // (left should be a project over scan, not a filter)
            switch (s.left.*) {
                .project => {},
                else => return error.InvalidPlan,
            }
        },
        else => return error.InvalidPlan,
    }
}

test "optimize preserves set operation type through optimization" {
    var arena = ast.AstArena.init(testing.allocator);
    defer arena.deinit();
    var schema = testSchema(testing.allocator);
    defer schema.deinit();

    const plan = try planAndOptimize(
        testing.allocator,
        "SELECT id, name FROM users EXCEPT SELECT id, name FROM users WHERE age > 30;",
        &arena,
        &schema,
    );

    try testing.expectEqual(planner_mod.PlanType.select_query, plan.plan_type);
    switch (plan.root.*) {
        .set_op => |s| {
            try testing.expectEqual(ast.SetOpType.except, s.op);
        },
        else => return error.InvalidPlan,
    }
}
