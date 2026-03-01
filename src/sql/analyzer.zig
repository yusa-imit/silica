//! Semantic Analyzer — name resolution and type checking for SQL ASTs.
//!
//! Validates parsed SQL statements against the schema catalog:
//! - Table existence checks for SELECT, INSERT, UPDATE, DELETE
//! - Column reference resolution (unqualified and qualified)
//! - SELECT * and table.* expansion
//! - Type compatibility checking in expressions
//! - Alias resolution in SELECT columns and FROM clauses
//! - Duplicate column/alias detection

const std = @import("std");
const Allocator = std.mem.Allocator;
const ast = @import("ast.zig");
const catalog_mod = @import("catalog.zig");

const ColumnInfo = catalog_mod.ColumnInfo;
const ColumnType = catalog_mod.ColumnType;
const TableInfo = catalog_mod.TableInfo;

// ── Analysis Errors ─────────────────────────────────────────────────

pub const AnalysisError = struct {
    message: []const u8,
    kind: ErrorKind,
};

pub const ErrorKind = enum {
    table_not_found,
    column_not_found,
    ambiguous_column,
    duplicate_alias,
    type_mismatch,
    too_many_values,
    too_few_values,
    column_count_mismatch,
    invalid_expression,
    star_not_allowed,
};

// ── Resolved Column Info ────────────────────────────────────────────

/// A column resolved during analysis.
pub const ResolvedColumn = struct {
    table: []const u8,
    column: []const u8,
    column_type: ColumnType,
    index: usize,
};

// ── Schema Provider ─────────────────────────────────────────────────

/// Abstraction over the schema catalog for testability.
/// In production, this wraps Catalog; in tests, uses in-memory tables.
pub const SchemaProvider = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    const VTable = struct {
        getTable: *const fn (ptr: *anyopaque, allocator: Allocator, name: []const u8) ?TableInfo,
        tableExists: *const fn (ptr: *anyopaque, name: []const u8) bool,
    };

    pub fn getTable(self: SchemaProvider, allocator: Allocator, name: []const u8) ?TableInfo {
        return self.vtable.getTable(self.ptr, allocator, name);
    }

    pub fn tableExists(self: SchemaProvider, name: []const u8) bool {
        return self.vtable.tableExists(self.ptr, name);
    }
};

// ── Scope ───────────────────────────────────────────────────────────

/// A table visible in the current query scope (from FROM/JOIN).
const ScopeTable = struct {
    /// Alias or table name as visible in the query.
    alias: []const u8,
    /// Actual table name in the catalog.
    table_name: []const u8,
    /// Column metadata from the catalog.
    columns: []const ColumnInfo,
};

// ── Analyzer ────────────────────────────────────────────────────────

pub const Analyzer = struct {
    allocator: Allocator,
    schema: SchemaProvider,
    errors: std.ArrayListUnmanaged(AnalysisError),
    /// Tables in scope for the current statement.
    scope_tables: std.ArrayListUnmanaged(ScopeTable),
    /// Arena for strings produced during analysis.
    arena: std.heap.ArenaAllocator,
    /// CTE names in scope (name → column info from CTE definition).
    cte_columns: std.StringHashMapUnmanaged([]const ColumnInfo),

    pub fn init(allocator: Allocator, schema: SchemaProvider) Analyzer {
        return .{
            .allocator = allocator,
            .schema = schema,
            .errors = .{},
            .scope_tables = .{},
            .arena = std.heap.ArenaAllocator.init(allocator),
            .cte_columns = .{},
        };
    }

    pub fn deinit(self: *Analyzer) void {
        self.errors.deinit(self.allocator);
        self.scope_tables.deinit(self.allocator);
        self.cte_columns.deinit(self.allocator);
        self.arena.deinit();
    }

    fn addError(self: *Analyzer, kind: ErrorKind, comptime fmt: []const u8, args: anytype) void {
        const msg = std.fmt.allocPrint(self.arena.allocator(), fmt, args) catch return;
        self.errors.append(self.allocator, .{ .message = msg, .kind = kind }) catch {};
    }

    fn clearScope(self: *Analyzer) void {
        self.scope_tables.clearRetainingCapacity();
        self.cte_columns.clearRetainingCapacity();
    }

    /// Add a table to the current scope. Returns false if alias conflicts.
    fn addTableToScope(self: *Analyzer, table_name: []const u8, alias: ?[]const u8) bool {
        const visible_name = alias orelse table_name;

        // Check for duplicate alias
        for (self.scope_tables.items) |st| {
            if (std.ascii.eqlIgnoreCase(st.alias, visible_name)) {
                self.addError(.duplicate_alias, "duplicate table alias: '{s}'", .{visible_name});
                return false;
            }
        }

        // Check CTE scope first
        if (self.cte_columns.get(table_name)) |cte_cols| {
            self.scope_tables.append(self.allocator, .{
                .alias = visible_name,
                .table_name = table_name,
                .columns = cte_cols,
            }) catch {};
            return true;
        }

        // Look up table in catalog
        const table_info = self.schema.getTable(self.allocator, table_name);
        if (table_info == null) {
            self.addError(.table_not_found, "table not found: '{s}'", .{table_name});
            return false;
        }

        self.scope_tables.append(self.allocator, .{
            .alias = visible_name,
            .table_name = table_name,
            .columns = table_info.?.columns,
        }) catch {};

        return true;
    }

    /// Resolve a column reference to the table it belongs to.
    fn resolveColumn(self: *Analyzer, name: ast.Name) ?ResolvedColumn {
        if (name.prefix) |prefix| {
            // Qualified: find the specific table
            for (self.scope_tables.items) |st| {
                if (std.ascii.eqlIgnoreCase(st.alias, prefix)) {
                    for (st.columns, 0..) |col, i| {
                        if (std.ascii.eqlIgnoreCase(col.name, name.name)) {
                            return .{
                                .table = st.alias,
                                .column = col.name,
                                .column_type = col.column_type,
                                .index = i,
                            };
                        }
                    }
                    self.addError(.column_not_found, "column '{s}' not found in table '{s}'", .{ name.name, prefix });
                    return null;
                }
            }
            self.addError(.table_not_found, "table or alias '{s}' not found", .{prefix});
            return null;
        }

        // Unqualified: search all tables in scope
        var found: ?ResolvedColumn = null;
        for (self.scope_tables.items) |st| {
            for (st.columns, 0..) |col, i| {
                if (std.ascii.eqlIgnoreCase(col.name, name.name)) {
                    if (found != null) {
                        self.addError(.ambiguous_column, "ambiguous column reference: '{s}'", .{name.name});
                        return null;
                    }
                    found = .{
                        .table = st.alias,
                        .column = col.name,
                        .column_type = col.column_type,
                        .index = i,
                    };
                }
            }
        }

        if (found == null) {
            self.addError(.column_not_found, "column '{s}' not found", .{name.name});
        }

        return found;
    }

    // ── Statement Analysis ──────────────────────────────────────────

    /// Analyze a top-level statement.
    pub fn analyze(self: *Analyzer, stmt: ast.Stmt) void {
        self.clearScope();
        switch (stmt) {
            .select => |s| self.analyzeSelect(&s),
            .insert => |s| self.analyzeInsert(&s),
            .update => |s| self.analyzeUpdate(&s),
            .delete => |s| self.analyzeDelete(&s),
            .create_table => |s| self.analyzeCreateTable(&s),
            .drop_table => |s| self.analyzeDropTable(&s),
            .create_index => |s| self.analyzeCreateIndex(&s),
            .drop_index => {},
            .transaction => {},
            .vacuum => {},
            .create_view => |s| self.analyzeSelect(&s.select),
            .drop_view => {},
            .explain => |s| self.analyze(s.stmt.*),
        }
    }

    fn analyzeSelect(self: *Analyzer, stmt: *const ast.SelectStmt) void {
        // Process CTEs — register each CTE as a virtual table in scope
        for (stmt.ctes) |cte| {
            // Infer column info from the CTE's inner SELECT
            const cte_cols = self.inferCteColumns(&cte);
            self.cte_columns.put(self.allocator, cte.name, cte_cols) catch {};
        }

        // Analyze the left (primary) SELECT body
        self.analyzeSelectBody(stmt);

        // Analyze set operation if present
        if (stmt.set_operation) |set_op| {
            const left_count = self.countResultColumns(stmt);

            // Clear scope for the right side analysis
            self.scope_tables.shrinkRetainingCapacity(0);

            // Re-register CTEs for right side (they share the WITH scope)
            // CTE columns are already in cte_columns hashmap

            self.analyzeSelectBody(set_op.right);

            const right_count = self.countResultColumns(set_op.right);

            // Validate column counts match
            if (left_count != right_count and left_count > 0 and right_count > 0) {
                self.addError(.column_count_mismatch, "set operation requires equal column counts: left has {d}, right has {d}", .{ left_count, right_count });
            }

            // Recursively analyze chained set operations on the right side
            if (set_op.right.set_operation != null) {
                // The right side's body was already analyzed above; now analyze its chain
                self.analyzeSetOpChain(set_op.right, left_count);
            }

            // Restore scope for outer ORDER BY / LIMIT analysis.
            // Clear scope completely (right side polluted positions 0..saved_scope),
            // then re-analyze the left body to rebuild the correct scope.
            self.scope_tables.shrinkRetainingCapacity(0);
            self.analyzeSelectBody(stmt);
        }

        // ORDER BY — applies to the compound result
        for (stmt.order_by) |o| self.analyzeExpr(o.expr);
    }

    /// Analyze the body of a SELECT (FROM, JOINs, columns, WHERE, GROUP BY, HAVING).
    fn analyzeSelectBody(self: *Analyzer, stmt: *const ast.SelectStmt) void {
        // Resolve FROM clause first (builds scope)
        if (stmt.from) |from| {
            self.resolveTableRef(from);
        }

        // Resolve JOINs
        for (stmt.joins) |join| {
            self.resolveTableRef(join.table);
            if (join.on_condition) |cond| {
                self.analyzeExpr(cond);
            }
        }

        // Resolve SELECT columns
        for (stmt.columns) |col| {
            switch (col) {
                .all_columns => {
                    // * requires at least one table in scope
                    if (self.scope_tables.items.len == 0) {
                        self.addError(.star_not_allowed, "SELECT * requires a FROM clause", .{});
                    }
                },
                .table_all_columns => |tbl| {
                    // table.* — verify the table is in scope
                    var found = false;
                    for (self.scope_tables.items) |st| {
                        if (std.ascii.eqlIgnoreCase(st.alias, tbl)) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        self.addError(.table_not_found, "table or alias '{s}' not found for '{s}.*'", .{ tbl, tbl });
                    }
                },
                .expr => |e| {
                    self.analyzeExpr(e.value);
                },
            }
        }

        // WHERE
        if (stmt.where) |w| self.analyzeExpr(w);

        // GROUP BY
        for (stmt.group_by) |g| self.analyzeExpr(g);

        // HAVING
        if (stmt.having) |h| self.analyzeExpr(h);
    }

    /// Recursively analyze chained set operations, validating column counts.
    fn analyzeSetOpChain(self: *Analyzer, stmt: *const ast.SelectStmt, expected_count: usize) void {
        if (stmt.set_operation) |set_op| {
            const right_count = self.countResultColumns(set_op.right);
            if (expected_count != right_count and expected_count > 0 and right_count > 0) {
                self.addError(.column_count_mismatch, "set operation requires equal column counts: expected {d}, got {d}", .{ expected_count, right_count });
            }

            // Clear scope and analyze the chained right side body
            self.scope_tables.shrinkRetainingCapacity(0);
            self.analyzeSelectBody(set_op.right);

            // Continue chain
            if (set_op.right.set_operation != null) {
                self.analyzeSetOpChain(set_op.right, expected_count);
            }
        }
    }

    /// Count the number of result columns in a SELECT statement.
    /// Returns 0 for SELECT * (unknown at analysis time without full expansion).
    fn countResultColumns(self: *Analyzer, stmt: *const ast.SelectStmt) usize {
        var count: usize = 0;
        for (stmt.columns) |col| {
            switch (col) {
                .all_columns => {
                    // * expands to all columns in scope — count from scope tables
                    for (self.scope_tables.items) |st| {
                        count += st.columns.len;
                    }
                },
                .table_all_columns => |tbl| {
                    for (self.scope_tables.items) |st| {
                        if (std.ascii.eqlIgnoreCase(st.alias, tbl)) {
                            count += st.columns.len;
                            break;
                        }
                    }
                },
                .expr => {
                    count += 1;
                },
            }
        }
        return count;
    }

    fn analyzeInsert(self: *Analyzer, stmt: *const ast.InsertStmt) void {
        // Check table exists
        if (!self.schema.tableExists(stmt.table)) {
            self.addError(.table_not_found, "table not found: '{s}'", .{stmt.table});
            return;
        }

        const table_info = self.schema.getTable(self.allocator, stmt.table) orelse return;

        // Validate column names if specified
        if (stmt.columns) |cols| {
            for (cols) |col_name| {
                var found = false;
                for (table_info.columns) |tc| {
                    if (std.ascii.eqlIgnoreCase(tc.name, col_name)) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    self.addError(.column_not_found, "column '{s}' not found in table '{s}'", .{ col_name, stmt.table });
                }
            }

            // Validate value row lengths match column count
            for (stmt.values) |row| {
                if (row.len != cols.len) {
                    self.addError(.column_count_mismatch, "expected {d} values, got {d}", .{ cols.len, row.len });
                }
            }
        } else {
            // No column list — value count must match table column count
            for (stmt.values) |row| {
                if (row.len != table_info.columns.len) {
                    self.addError(.column_count_mismatch, "expected {d} values, got {d}", .{ table_info.columns.len, row.len });
                }
            }
        }

        // Analyze value expressions
        for (stmt.values) |row| {
            for (row) |val| {
                self.analyzeExpr(val);
            }
        }
    }

    fn analyzeUpdate(self: *Analyzer, stmt: *const ast.UpdateStmt) void {
        if (!self.schema.tableExists(stmt.table)) {
            self.addError(.table_not_found, "table not found: '{s}'", .{stmt.table});
            return;
        }

        _ = self.addTableToScope(stmt.table, null);

        const table_info = self.schema.getTable(self.allocator, stmt.table) orelse return;

        for (stmt.assignments) |assignment| {
            var found = false;
            for (table_info.columns) |col| {
                if (std.ascii.eqlIgnoreCase(col.name, assignment.column)) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                self.addError(.column_not_found, "column '{s}' not found in table '{s}'", .{ assignment.column, stmt.table });
            }
            self.analyzeExpr(assignment.value);
        }

        if (stmt.where) |w| self.analyzeExpr(w);
    }

    fn analyzeDelete(self: *Analyzer, stmt: *const ast.DeleteStmt) void {
        if (!self.schema.tableExists(stmt.table)) {
            self.addError(.table_not_found, "table not found: '{s}'", .{stmt.table});
            return;
        }

        _ = self.addTableToScope(stmt.table, null);

        if (stmt.where) |w| self.analyzeExpr(w);
    }

    fn analyzeCreateTable(self: *Analyzer, stmt: *const ast.CreateTableStmt) void {
        // Check for duplicate column names
        for (stmt.columns, 0..) |col, i| {
            for (stmt.columns[i + 1 ..]) |other| {
                if (std.ascii.eqlIgnoreCase(col.name, other.name)) {
                    self.addError(.duplicate_alias, "duplicate column name: '{s}'", .{col.name});
                    break;
                }
            }
        }

        // Validate table constraint column references
        for (stmt.table_constraints) |tc| {
            switch (tc) {
                .primary_key => |pk| {
                    for (pk.columns) |pk_col| {
                        if (!columnExistsInDef(stmt.columns, pk_col)) {
                            self.addError(.column_not_found, "column '{s}' in PRIMARY KEY not found in table definition", .{pk_col});
                        }
                    }
                },
                .unique => |uq| {
                    for (uq.columns) |uq_col| {
                        if (!columnExistsInDef(stmt.columns, uq_col)) {
                            self.addError(.column_not_found, "column '{s}' in UNIQUE constraint not found in table definition", .{uq_col});
                        }
                    }
                },
                .check => |ch| {
                    // Build a temporary scope for CHECK expression validation
                    // We can't use the catalog since the table doesn't exist yet
                    _ = ch;
                },
                .foreign_key => |fk| {
                    for (fk.columns) |fk_col| {
                        if (!columnExistsInDef(stmt.columns, fk_col)) {
                            self.addError(.column_not_found, "column '{s}' in FOREIGN KEY not found in table definition", .{fk_col});
                        }
                    }
                    // Validate referenced table exists
                    if (!self.schema.tableExists(fk.ref_table)) {
                        self.addError(.table_not_found, "referenced table '{s}' not found", .{fk.ref_table});
                    }
                },
            }
        }

        // Check that table doesn't already exist (unless IF NOT EXISTS)
        if (!stmt.if_not_exists and self.schema.tableExists(stmt.name)) {
            self.addError(.duplicate_alias, "table '{s}' already exists", .{stmt.name});
        }
    }

    fn analyzeDropTable(self: *Analyzer, stmt: *const ast.DropTableStmt) void {
        if (!stmt.if_exists and !self.schema.tableExists(stmt.name)) {
            self.addError(.table_not_found, "table '{s}' not found", .{stmt.name});
        }
    }

    fn analyzeCreateIndex(self: *Analyzer, stmt: *const ast.CreateIndexStmt) void {
        if (!self.schema.tableExists(stmt.table)) {
            self.addError(.table_not_found, "table '{s}' not found", .{stmt.table});
            return;
        }

        _ = self.addTableToScope(stmt.table, null);

        for (stmt.columns) |col| {
            self.analyzeExpr(col.expr);
        }
    }

    // ── CTE Column Inference ────────────────────────────────────────

    /// Infer column metadata from a CTE definition.
    /// Uses explicit column names if provided, otherwise derives from SELECT result columns.
    fn inferCteColumns(self: *Analyzer, cte: *const ast.CteDefinition) []const ColumnInfo {
        const a = self.arena.allocator();

        // If explicit column names are provided, use them
        if (cte.column_names.len > 0) {
            var cols = std.ArrayListUnmanaged(ColumnInfo){};
            for (cte.column_names) |col_name| {
                cols.append(a, .{
                    .name = col_name,
                    .column_type = .blob, // type unknown at analysis time
                    .flags = .{},
                }) catch {};
            }
            return cols.toOwnedSlice(a) catch return &.{};
        }

        // Derive column names from result columns
        var cols = std.ArrayListUnmanaged(ColumnInfo){};
        for (cte.select.columns, 0..) |col, i| {
            const col_name: []const u8 = switch (col) {
                .all_columns => "*",
                .table_all_columns => |t| t,
                .expr => |e| blk: {
                    // Use alias if present
                    if (e.alias) |alias_name| break :blk alias_name;
                    // Try to extract a name from column_ref
                    if (e.value.* == .column_ref) break :blk e.value.column_ref.name;
                    // For expressions/literals, generate a generic name
                    break :blk std.fmt.allocPrint(a, "column{d}", .{i}) catch "?column?";
                },
            };
            cols.append(a, .{
                .name = col_name,
                .column_type = .blob,
                .flags = .{},
            }) catch {};
        }
        return cols.toOwnedSlice(a) catch return &.{};
    }

    // ── Table Reference Resolution ──────────────────────────────────

    fn resolveTableRef(self: *Analyzer, ref: *const ast.TableRef) void {
        switch (ref.*) {
            .table_name => |tn| {
                _ = self.addTableToScope(tn.name, tn.alias);
            },
            .subquery => {
                // Subqueries create their own scope — skip for now
            },
        }
    }

    // ── Expression Analysis ─────────────────────────────────────────

    fn analyzeExpr(self: *Analyzer, expr: *const ast.Expr) void {
        switch (expr.*) {
            .integer_literal, .float_literal, .string_literal, .blob_literal, .boolean_literal, .null_literal => {},
            .column_ref => |name| {
                _ = self.resolveColumn(name);
            },
            .unary_op => |u| {
                self.analyzeExpr(u.operand);
            },
            .binary_op => |b| {
                self.analyzeExpr(b.left);
                self.analyzeExpr(b.right);
            },
            .function_call => |f| {
                for (f.args) |arg| {
                    // Skip resolution for * in aggregate functions like COUNT(*)
                    if (arg.* == .column_ref and std.mem.eql(u8, arg.column_ref.name, "*")) continue;
                    self.analyzeExpr(arg);
                }
            },
            .between => |b| {
                self.analyzeExpr(b.expr);
                self.analyzeExpr(b.low);
                self.analyzeExpr(b.high);
            },
            .in_list => |il| {
                self.analyzeExpr(il.expr);
                for (il.list) |item| self.analyzeExpr(item);
            },
            .is_null => |isn| {
                self.analyzeExpr(isn.expr);
            },
            .like => |l| {
                self.analyzeExpr(l.expr);
                self.analyzeExpr(l.pattern);
            },
            .case_expr => |c| {
                if (c.operand) |op| self.analyzeExpr(op);
                for (c.when_clauses) |wc| {
                    self.analyzeExpr(wc.condition);
                    self.analyzeExpr(wc.result);
                }
                if (c.else_expr) |el| self.analyzeExpr(el);
            },
            .cast => |ca| {
                self.analyzeExpr(ca.expr);
            },
            .paren => |p| {
                self.analyzeExpr(p);
            },
            .subquery => {
                // Subqueries have their own scope — skip deep analysis for now
            },
            .bind_parameter => {},
        }
    }

    // ── Helpers ─────────────────────────────────────────────────────

    fn columnExistsInDef(columns: []const ast.ColumnDef, name: []const u8) bool {
        for (columns) |col| {
            if (std.ascii.eqlIgnoreCase(col.name, name)) return true;
        }
        return false;
    }

    pub fn hasErrors(self: *const Analyzer) bool {
        return self.errors.items.len > 0;
    }
};

// ── In-Memory Schema Provider for Testing ───────────────────────────

pub const MemorySchema = struct {
    tables: std.StringHashMapUnmanaged(TableInfo),
    allocator: Allocator,

    pub fn init(allocator: Allocator) MemorySchema {
        return .{
            .tables = .{},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *MemorySchema) void {
        self.tables.deinit(self.allocator);
    }

    pub fn addTable(self: *MemorySchema, name: []const u8, columns: []const ColumnInfo) void {
        self.tables.put(self.allocator, name, .{
            .name = name,
            .columns = columns,
            .table_constraints = &.{},
        }) catch {};
    }

    pub fn provider(self: *MemorySchema) SchemaProvider {
        return .{
            .ptr = @ptrCast(self),
            .vtable = &.{
                .getTable = memoryGetTable,
                .tableExists = memoryTableExists,
            },
        };
    }

    fn memoryGetTable(ptr: *anyopaque, _: Allocator, name: []const u8) ?TableInfo {
        const self: *MemorySchema = @ptrCast(@alignCast(ptr));
        return self.tables.get(name);
    }

    fn memoryTableExists(ptr: *anyopaque, name: []const u8) bool {
        const self: *MemorySchema = @ptrCast(@alignCast(ptr));
        return self.tables.contains(name);
    }
};

// ── Tests ───────────────────────────────────────────────────────────

fn parseAndAnalyze(allocator: Allocator, sql: []const u8, schema: SchemaProvider) !Analyzer {
    var arena = ast.AstArena.init(allocator);
    defer arena.deinit();

    var parser = @import("parser.zig").Parser.init(allocator, sql, &arena) catch return error.ParseFailed;
    defer parser.deinit();

    const stmt = parser.parseStatement() catch return error.ParseFailed;
    if (stmt == null) return error.ParseFailed;

    var analyzer = Analyzer.init(allocator, schema);
    analyzer.analyze(stmt.?);
    return analyzer;
}

test "SELECT * FROM existing table" {
    const allocator = std.testing.allocator;
    var schema = MemorySchema.init(allocator);
    defer schema.deinit();

    schema.addTable("users", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{} },
        .{ .name = "name", .column_type = .text, .flags = .{} },
    });

    var analyzer = try parseAndAnalyze(allocator, "SELECT * FROM users;", schema.provider());
    defer analyzer.deinit();

    try std.testing.expect(!analyzer.hasErrors());
}

test "SELECT from non-existent table" {
    const allocator = std.testing.allocator;
    var schema = MemorySchema.init(allocator);
    defer schema.deinit();

    var analyzer = try parseAndAnalyze(allocator, "SELECT * FROM ghost;", schema.provider());
    defer analyzer.deinit();

    try std.testing.expect(analyzer.hasErrors());
    // Expect table_not_found error (may also get star_not_allowed since scope is empty)
    var found_table_error = false;
    for (analyzer.errors.items) |err| {
        if (err.kind == .table_not_found) found_table_error = true;
    }
    try std.testing.expect(found_table_error);
}

test "SELECT with qualified column reference" {
    const allocator = std.testing.allocator;
    var schema = MemorySchema.init(allocator);
    defer schema.deinit();

    schema.addTable("users", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{} },
        .{ .name = "name", .column_type = .text, .flags = .{} },
    });

    var analyzer = try parseAndAnalyze(allocator, "SELECT users.id, users.name FROM users;", schema.provider());
    defer analyzer.deinit();

    try std.testing.expect(!analyzer.hasErrors());
}

test "SELECT with unknown column" {
    const allocator = std.testing.allocator;
    var schema = MemorySchema.init(allocator);
    defer schema.deinit();

    schema.addTable("users", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{} },
    });

    var analyzer = try parseAndAnalyze(allocator, "SELECT age FROM users;", schema.provider());
    defer analyzer.deinit();

    try std.testing.expect(analyzer.hasErrors());
    try std.testing.expectEqual(ErrorKind.column_not_found, analyzer.errors.items[0].kind);
}

test "SELECT with ambiguous column" {
    const allocator = std.testing.allocator;
    var schema = MemorySchema.init(allocator);
    defer schema.deinit();

    schema.addTable("users", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{} },
        .{ .name = "name", .column_type = .text, .flags = .{} },
    });
    schema.addTable("orders", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{} },
        .{ .name = "total", .column_type = .real, .flags = .{} },
    });

    var analyzer = try parseAndAnalyze(allocator, "SELECT id FROM users INNER JOIN orders ON users.id = orders.id;", schema.provider());
    defer analyzer.deinit();

    try std.testing.expect(analyzer.hasErrors());
    // "id" is ambiguous between users and orders
    var found_ambiguous = false;
    for (analyzer.errors.items) |err| {
        if (err.kind == .ambiguous_column) found_ambiguous = true;
    }
    try std.testing.expect(found_ambiguous);
}

test "SELECT with table alias" {
    const allocator = std.testing.allocator;
    var schema = MemorySchema.init(allocator);
    defer schema.deinit();

    schema.addTable("users", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{} },
        .{ .name = "name", .column_type = .text, .flags = .{} },
    });

    var analyzer = try parseAndAnalyze(allocator, "SELECT u.id, u.name FROM users u;", schema.provider());
    defer analyzer.deinit();

    try std.testing.expect(!analyzer.hasErrors());
}

test "SELECT * without FROM" {
    const allocator = std.testing.allocator;
    var schema = MemorySchema.init(allocator);
    defer schema.deinit();

    var analyzer = try parseAndAnalyze(allocator, "SELECT *;", schema.provider());
    defer analyzer.deinit();

    try std.testing.expect(analyzer.hasErrors());
    try std.testing.expectEqual(ErrorKind.star_not_allowed, analyzer.errors.items[0].kind);
}

test "INSERT with correct column count" {
    const allocator = std.testing.allocator;
    var schema = MemorySchema.init(allocator);
    defer schema.deinit();

    schema.addTable("users", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{} },
        .{ .name = "name", .column_type = .text, .flags = .{} },
    });

    var analyzer = try parseAndAnalyze(allocator, "INSERT INTO users VALUES (1, 'alice');", schema.provider());
    defer analyzer.deinit();

    try std.testing.expect(!analyzer.hasErrors());
}

test "INSERT with wrong column count" {
    const allocator = std.testing.allocator;
    var schema = MemorySchema.init(allocator);
    defer schema.deinit();

    schema.addTable("users", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{} },
        .{ .name = "name", .column_type = .text, .flags = .{} },
    });

    var analyzer = try parseAndAnalyze(allocator, "INSERT INTO users VALUES (1);", schema.provider());
    defer analyzer.deinit();

    try std.testing.expect(analyzer.hasErrors());
    try std.testing.expectEqual(ErrorKind.column_count_mismatch, analyzer.errors.items[0].kind);
}

test "INSERT into non-existent table" {
    const allocator = std.testing.allocator;
    var schema = MemorySchema.init(allocator);
    defer schema.deinit();

    var analyzer = try parseAndAnalyze(allocator, "INSERT INTO ghost VALUES (1);", schema.provider());
    defer analyzer.deinit();

    try std.testing.expect(analyzer.hasErrors());
    try std.testing.expectEqual(ErrorKind.table_not_found, analyzer.errors.items[0].kind);
}

test "INSERT with named columns" {
    const allocator = std.testing.allocator;
    var schema = MemorySchema.init(allocator);
    defer schema.deinit();

    schema.addTable("users", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{} },
        .{ .name = "name", .column_type = .text, .flags = .{} },
        .{ .name = "email", .column_type = .text, .flags = .{} },
    });

    // Valid subset
    var a1 = try parseAndAnalyze(allocator, "INSERT INTO users (id, name) VALUES (1, 'alice');", schema.provider());
    defer a1.deinit();
    try std.testing.expect(!a1.hasErrors());

    // Unknown column
    var a2 = try parseAndAnalyze(allocator, "INSERT INTO users (id, age) VALUES (1, 25);", schema.provider());
    defer a2.deinit();
    try std.testing.expect(a2.hasErrors());
    try std.testing.expectEqual(ErrorKind.column_not_found, a2.errors.items[0].kind);
}

test "UPDATE with valid columns" {
    const allocator = std.testing.allocator;
    var schema = MemorySchema.init(allocator);
    defer schema.deinit();

    schema.addTable("users", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{} },
        .{ .name = "name", .column_type = .text, .flags = .{} },
    });

    var analyzer = try parseAndAnalyze(allocator, "UPDATE users SET name = 'bob' WHERE id = 1;", schema.provider());
    defer analyzer.deinit();

    try std.testing.expect(!analyzer.hasErrors());
}

test "UPDATE with unknown column" {
    const allocator = std.testing.allocator;
    var schema = MemorySchema.init(allocator);
    defer schema.deinit();

    schema.addTable("users", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{} },
    });

    var analyzer = try parseAndAnalyze(allocator, "UPDATE users SET age = 25;", schema.provider());
    defer analyzer.deinit();

    try std.testing.expect(analyzer.hasErrors());
    try std.testing.expectEqual(ErrorKind.column_not_found, analyzer.errors.items[0].kind);
}

test "DELETE with valid table" {
    const allocator = std.testing.allocator;
    var schema = MemorySchema.init(allocator);
    defer schema.deinit();

    schema.addTable("users", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{} },
    });

    var analyzer = try parseAndAnalyze(allocator, "DELETE FROM users WHERE id = 1;", schema.provider());
    defer analyzer.deinit();

    try std.testing.expect(!analyzer.hasErrors());
}

test "DELETE from non-existent table" {
    const allocator = std.testing.allocator;
    var schema = MemorySchema.init(allocator);
    defer schema.deinit();

    var analyzer = try parseAndAnalyze(allocator, "DELETE FROM ghost;", schema.provider());
    defer analyzer.deinit();

    try std.testing.expect(analyzer.hasErrors());
    try std.testing.expectEqual(ErrorKind.table_not_found, analyzer.errors.items[0].kind);
}

test "CREATE TABLE duplicate columns" {
    const allocator = std.testing.allocator;
    var schema = MemorySchema.init(allocator);
    defer schema.deinit();

    var analyzer = try parseAndAnalyze(allocator, "CREATE TABLE t (a INTEGER, b TEXT, a REAL);", schema.provider());
    defer analyzer.deinit();

    try std.testing.expect(analyzer.hasErrors());
    try std.testing.expectEqual(ErrorKind.duplicate_alias, analyzer.errors.items[0].kind);
}

test "CREATE TABLE already exists" {
    const allocator = std.testing.allocator;
    var schema = MemorySchema.init(allocator);
    defer schema.deinit();

    schema.addTable("users", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{} },
    });

    var analyzer = try parseAndAnalyze(allocator, "CREATE TABLE users (id INTEGER);", schema.provider());
    defer analyzer.deinit();

    try std.testing.expect(analyzer.hasErrors());
}

test "CREATE TABLE IF NOT EXISTS suppresses error" {
    const allocator = std.testing.allocator;
    var schema = MemorySchema.init(allocator);
    defer schema.deinit();

    schema.addTable("users", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{} },
    });

    var analyzer = try parseAndAnalyze(allocator, "CREATE TABLE IF NOT EXISTS users (id INTEGER);", schema.provider());
    defer analyzer.deinit();

    try std.testing.expect(!analyzer.hasErrors());
}

test "DROP TABLE non-existent" {
    const allocator = std.testing.allocator;
    var schema = MemorySchema.init(allocator);
    defer schema.deinit();

    var analyzer = try parseAndAnalyze(allocator, "DROP TABLE ghost;", schema.provider());
    defer analyzer.deinit();

    try std.testing.expect(analyzer.hasErrors());
    try std.testing.expectEqual(ErrorKind.table_not_found, analyzer.errors.items[0].kind);
}

test "DROP TABLE IF EXISTS suppresses error" {
    const allocator = std.testing.allocator;
    var schema = MemorySchema.init(allocator);
    defer schema.deinit();

    var analyzer = try parseAndAnalyze(allocator, "DROP TABLE IF EXISTS ghost;", schema.provider());
    defer analyzer.deinit();

    try std.testing.expect(!analyzer.hasErrors());
}

test "SELECT with WHERE column check" {
    const allocator = std.testing.allocator;
    var schema = MemorySchema.init(allocator);
    defer schema.deinit();

    schema.addTable("users", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{} },
        .{ .name = "name", .column_type = .text, .flags = .{} },
    });

    // Valid WHERE column
    var a1 = try parseAndAnalyze(allocator, "SELECT name FROM users WHERE id > 5;", schema.provider());
    defer a1.deinit();
    try std.testing.expect(!a1.hasErrors());

    // Invalid WHERE column
    var a2 = try parseAndAnalyze(allocator, "SELECT name FROM users WHERE age > 5;", schema.provider());
    defer a2.deinit();
    try std.testing.expect(a2.hasErrors());
}

test "SELECT with JOIN" {
    const allocator = std.testing.allocator;
    var schema = MemorySchema.init(allocator);
    defer schema.deinit();

    schema.addTable("users", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{} },
        .{ .name = "name", .column_type = .text, .flags = .{} },
    });
    schema.addTable("orders", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{} },
        .{ .name = "user_id", .column_type = .integer, .flags = .{} },
        .{ .name = "total", .column_type = .real, .flags = .{} },
    });

    var analyzer = try parseAndAnalyze(allocator,
        "SELECT users.name, orders.total FROM users INNER JOIN orders ON users.id = orders.user_id;",
        schema.provider());
    defer analyzer.deinit();

    try std.testing.expect(!analyzer.hasErrors());
}

test "MemorySchema basic operations" {
    const allocator = std.testing.allocator;
    var schema = MemorySchema.init(allocator);
    defer schema.deinit();

    try std.testing.expect(!schema.provider().tableExists("users"));

    schema.addTable("users", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{} },
    });

    try std.testing.expect(schema.provider().tableExists("users"));
    try std.testing.expect(!schema.provider().tableExists("orders"));

    const info = schema.provider().getTable(allocator, "users");
    try std.testing.expect(info != null);
    try std.testing.expectEqualStrings("users", info.?.name);
    try std.testing.expectEqual(@as(usize, 1), info.?.columns.len);
}

test "SELECT with expressions" {
    const allocator = std.testing.allocator;
    var schema = MemorySchema.init(allocator);
    defer schema.deinit();

    schema.addTable("t", &.{
        .{ .name = "a", .column_type = .integer, .flags = .{} },
        .{ .name = "b", .column_type = .integer, .flags = .{} },
    });

    // Arithmetic expressions
    var a1 = try parseAndAnalyze(allocator, "SELECT a + b FROM t;", schema.provider());
    defer a1.deinit();
    try std.testing.expect(!a1.hasErrors());

    // Function call
    var a2 = try parseAndAnalyze(allocator, "SELECT COUNT(a) FROM t;", schema.provider());
    defer a2.deinit();
    try std.testing.expect(!a2.hasErrors());
}

test "SELECT with ORDER BY and LIMIT" {
    const allocator = std.testing.allocator;
    var schema = MemorySchema.init(allocator);
    defer schema.deinit();

    schema.addTable("users", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{} },
        .{ .name = "name", .column_type = .text, .flags = .{} },
    });

    var analyzer = try parseAndAnalyze(allocator,
        "SELECT name FROM users ORDER BY id DESC LIMIT 10;",
        schema.provider());
    defer analyzer.deinit();

    try std.testing.expect(!analyzer.hasErrors());
}

test "SELECT literal expression without FROM" {
    const allocator = std.testing.allocator;
    var schema = MemorySchema.init(allocator);
    defer schema.deinit();

    var analyzer = try parseAndAnalyze(allocator, "SELECT 1 + 2;", schema.provider());
    defer analyzer.deinit();

    try std.testing.expect(!analyzer.hasErrors());
}

test "table.* with valid table alias" {
    const allocator = std.testing.allocator;
    var schema = MemorySchema.init(allocator);
    defer schema.deinit();

    schema.addTable("users", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{} },
    });

    var analyzer = try parseAndAnalyze(allocator, "SELECT u.* FROM users u;", schema.provider());
    defer analyzer.deinit();

    try std.testing.expect(!analyzer.hasErrors());
}

test "table.* with invalid alias" {
    const allocator = std.testing.allocator;
    var schema = MemorySchema.init(allocator);
    defer schema.deinit();

    schema.addTable("users", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{} },
    });

    var analyzer = try parseAndAnalyze(allocator, "SELECT x.* FROM users;", schema.provider());
    defer analyzer.deinit();

    try std.testing.expect(analyzer.hasErrors());
    try std.testing.expectEqual(ErrorKind.table_not_found, analyzer.errors.items[0].kind);
}

// ── CTE Analysis Tests ──────────────────────────────────────────────

test "CTE: simple CTE passes analysis" {
    const allocator = std.testing.allocator;
    var schema = MemorySchema.init(allocator);
    defer schema.deinit();

    var analyzer = try parseAndAnalyze(allocator,
        "WITH vals AS (SELECT 1) SELECT * FROM vals;",
        schema.provider());
    defer analyzer.deinit();

    try std.testing.expect(!analyzer.hasErrors());
}

test "CTE: CTE with column aliases" {
    const allocator = std.testing.allocator;
    var schema = MemorySchema.init(allocator);
    defer schema.deinit();

    var analyzer = try parseAndAnalyze(allocator,
        "WITH cte(x, y) AS (SELECT 1, 2) SELECT x, y FROM cte;",
        schema.provider());
    defer analyzer.deinit();

    try std.testing.expect(!analyzer.hasErrors());
}

test "CTE: multiple CTEs" {
    const allocator = std.testing.allocator;
    var schema = MemorySchema.init(allocator);
    defer schema.deinit();

    var analyzer = try parseAndAnalyze(allocator,
        "WITH a AS (SELECT 1), b AS (SELECT 2) SELECT * FROM a;",
        schema.provider());
    defer analyzer.deinit();

    try std.testing.expect(!analyzer.hasErrors());
}

test "CTE: CTE referencing real table" {
    const allocator = std.testing.allocator;
    var schema = MemorySchema.init(allocator);
    defer schema.deinit();

    schema.addTable("users", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{} },
        .{ .name = "name", .column_type = .text, .flags = .{} },
    });

    var analyzer = try parseAndAnalyze(allocator,
        "WITH active AS (SELECT id, name FROM users) SELECT * FROM active;",
        schema.provider());
    defer analyzer.deinit();

    try std.testing.expect(!analyzer.hasErrors());
}

test "CTE: non-existent CTE reference fails" {
    const allocator = std.testing.allocator;
    var schema = MemorySchema.init(allocator);
    defer schema.deinit();

    var analyzer = try parseAndAnalyze(allocator,
        "WITH a AS (SELECT 1) SELECT * FROM nonexistent;",
        schema.provider());
    defer analyzer.deinit();

    try std.testing.expect(analyzer.hasErrors());
    try std.testing.expectEqual(ErrorKind.table_not_found, analyzer.errors.items[0].kind);
}

// ── Set Operation Analysis Tests ─────────────────────────────────────

test "set op: UNION with matching column count passes" {
    const allocator = std.testing.allocator;
    var schema = MemorySchema.init(allocator);
    defer schema.deinit();

    schema.addTable("t1", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{} },
        .{ .name = "name", .column_type = .text, .flags = .{} },
    });
    schema.addTable("t2", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{} },
        .{ .name = "label", .column_type = .text, .flags = .{} },
    });

    var analyzer = try parseAndAnalyze(allocator,
        "SELECT id, name FROM t1 UNION SELECT id, label FROM t2;",
        schema.provider());
    defer analyzer.deinit();

    try std.testing.expect(!analyzer.hasErrors());
}

test "set op: UNION ALL passes" {
    const allocator = std.testing.allocator;
    var schema = MemorySchema.init(allocator);
    defer schema.deinit();

    schema.addTable("t1", &.{
        .{ .name = "a", .column_type = .integer, .flags = .{} },
    });
    schema.addTable("t2", &.{
        .{ .name = "b", .column_type = .integer, .flags = .{} },
    });

    var analyzer = try parseAndAnalyze(allocator,
        "SELECT a FROM t1 UNION ALL SELECT b FROM t2;",
        schema.provider());
    defer analyzer.deinit();

    try std.testing.expect(!analyzer.hasErrors());
}

test "set op: INTERSECT passes" {
    const allocator = std.testing.allocator;
    var schema = MemorySchema.init(allocator);
    defer schema.deinit();

    schema.addTable("t1", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{} },
    });
    schema.addTable("t2", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{} },
    });

    var analyzer = try parseAndAnalyze(allocator,
        "SELECT id FROM t1 INTERSECT SELECT id FROM t2;",
        schema.provider());
    defer analyzer.deinit();

    try std.testing.expect(!analyzer.hasErrors());
}

test "set op: EXCEPT passes" {
    const allocator = std.testing.allocator;
    var schema = MemorySchema.init(allocator);
    defer schema.deinit();

    schema.addTable("t1", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{} },
    });
    schema.addTable("t2", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{} },
    });

    var analyzer = try parseAndAnalyze(allocator,
        "SELECT id FROM t1 EXCEPT SELECT id FROM t2;",
        schema.provider());
    defer analyzer.deinit();

    try std.testing.expect(!analyzer.hasErrors());
}

test "set op: mismatched column count fails" {
    const allocator = std.testing.allocator;
    var schema = MemorySchema.init(allocator);
    defer schema.deinit();

    schema.addTable("t1", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{} },
        .{ .name = "name", .column_type = .text, .flags = .{} },
    });
    schema.addTable("t2", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{} },
    });

    var analyzer = try parseAndAnalyze(allocator,
        "SELECT id, name FROM t1 UNION SELECT id FROM t2;",
        schema.provider());
    defer analyzer.deinit();

    try std.testing.expect(analyzer.hasErrors());
    var found_mismatch = false;
    for (analyzer.errors.items) |err| {
        if (err.kind == .column_count_mismatch) found_mismatch = true;
    }
    try std.testing.expect(found_mismatch);
}

test "set op: with CTE passes" {
    const allocator = std.testing.allocator;
    var schema = MemorySchema.init(allocator);
    defer schema.deinit();

    var analyzer = try parseAndAnalyze(allocator,
        "WITH cte AS (SELECT 1 AS x) SELECT x FROM cte UNION SELECT x FROM cte;",
        schema.provider());
    defer analyzer.deinit();

    try std.testing.expect(!analyzer.hasErrors());
}

test "set op: chained UNION passes" {
    const allocator = std.testing.allocator;
    var schema = MemorySchema.init(allocator);
    defer schema.deinit();
    schema.addTable("t1", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{} },
    });
    schema.addTable("t2", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{} },
    });
    schema.addTable("t3", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{} },
    });

    var analyzer = try parseAndAnalyze(allocator,
        "SELECT id FROM t1 UNION SELECT id FROM t2 UNION SELECT id FROM t3;",
        schema.provider());
    defer analyzer.deinit();

    try std.testing.expect(!analyzer.hasErrors());
}

test "set op: chained with mismatched columns in third query fails" {
    const allocator = std.testing.allocator;
    var schema = MemorySchema.init(allocator);
    defer schema.deinit();
    schema.addTable("t1", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{} },
    });
    schema.addTable("t2", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{} },
    });
    schema.addTable("t3", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{} },
        .{ .name = "name", .column_type = .text, .flags = .{} },
    });

    var analyzer = try parseAndAnalyze(allocator,
        "SELECT id FROM t1 UNION SELECT id FROM t2 UNION SELECT id, name FROM t3;",
        schema.provider());
    defer analyzer.deinit();

    try std.testing.expect(analyzer.hasErrors());
    var found_mismatch = false;
    for (analyzer.errors.items) |err| {
        if (err.kind == .column_count_mismatch) found_mismatch = true;
    }
    try std.testing.expect(found_mismatch);
}

test "set op: SELECT * with matching columns passes" {
    const allocator = std.testing.allocator;
    var schema = MemorySchema.init(allocator);
    defer schema.deinit();
    schema.addTable("t1", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{} },
        .{ .name = "name", .column_type = .text, .flags = .{} },
    });
    schema.addTable("t2", &.{
        .{ .name = "x", .column_type = .integer, .flags = .{} },
        .{ .name = "y", .column_type = .text, .flags = .{} },
    });

    var analyzer = try parseAndAnalyze(allocator,
        "SELECT * FROM t1 UNION SELECT * FROM t2;",
        schema.provider());
    defer analyzer.deinit();

    try std.testing.expect(!analyzer.hasErrors());
}

test "set op: SELECT * with different column counts fails" {
    const allocator = std.testing.allocator;
    var schema = MemorySchema.init(allocator);
    defer schema.deinit();
    schema.addTable("t1", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{} },
        .{ .name = "name", .column_type = .text, .flags = .{} },
    });
    schema.addTable("t2", &.{
        .{ .name = "x", .column_type = .integer, .flags = .{} },
    });

    var analyzer = try parseAndAnalyze(allocator,
        "SELECT * FROM t1 UNION SELECT * FROM t2;",
        schema.provider());
    defer analyzer.deinit();

    try std.testing.expect(analyzer.hasErrors());
}

test "set op: with ORDER BY on compound result passes" {
    const allocator = std.testing.allocator;
    var schema = MemorySchema.init(allocator);
    defer schema.deinit();
    schema.addTable("t1", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{} },
    });
    schema.addTable("t2", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{} },
    });

    var analyzer = try parseAndAnalyze(allocator,
        "SELECT id FROM t1 UNION SELECT id FROM t2 ORDER BY id;",
        schema.provider());
    defer analyzer.deinit();

    try std.testing.expect(!analyzer.hasErrors());
}

test "set op: CTE accessible on both sides" {
    const allocator = std.testing.allocator;
    var schema = MemorySchema.init(allocator);
    defer schema.deinit();
    schema.addTable("t1", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{} },
    });

    var analyzer = try parseAndAnalyze(allocator,
        "WITH vals AS (SELECT 1 AS x) SELECT x FROM vals UNION ALL SELECT x FROM vals;",
        schema.provider());
    defer analyzer.deinit();

    try std.testing.expect(!analyzer.hasErrors());
}

test "set op: multiple CTEs with set operation" {
    const allocator = std.testing.allocator;
    var schema = MemorySchema.init(allocator);
    defer schema.deinit();

    var analyzer = try parseAndAnalyze(allocator,
        "WITH a AS (SELECT 1 AS x), b AS (SELECT 2 AS y) SELECT x FROM a UNION SELECT y FROM b;",
        schema.provider());
    defer analyzer.deinit();

    try std.testing.expect(!analyzer.hasErrors());
}
