const std = @import("std");
const Allocator = std.mem.Allocator;
const tokenizer = @import("tokenizer.zig");
const Token = tokenizer.Token;
const TokenType = tokenizer.TokenType;
const Tokenizer = tokenizer.Tokenizer;
const ast = @import("ast.zig");

/// Parser error information.
pub const ParseError = struct {
    message: []const u8,
    token: Token,
};

/// Recursive descent SQL parser.
/// Produces a typed AST from a token stream. Uses arena allocation for all AST nodes.
pub const Parser = struct {
    source: []const u8,
    tokens: []const Token,
    pos: u32,
    arena: *ast.AstArena,
    /// Allocator for parser infrastructure (token list, error list).
    infra_alloc: Allocator,
    errors: std.ArrayListUnmanaged(ParseError),
    /// Counter for bind parameters (? placeholders) in current statement.
    bind_param_index: u32,

    pub const Error = error{
        ParseFailed,
        OutOfMemory,
    };

    /// Initialize a parser. `infra_alloc` is used for parser-internal data;
    /// all AST nodes are allocated in the arena.
    pub fn init(infra_alloc: Allocator, source: []const u8, arena: *ast.AstArena) Error!Parser {
        var tok = Tokenizer.init(source);
        var token_list = std.ArrayListUnmanaged(Token){};
        while (true) {
            const t = tok.next();
            token_list.append(infra_alloc, t) catch return error.OutOfMemory;
            if (t.type == .eof) break;
        }

        return .{
            .source = source,
            .tokens = token_list.toOwnedSlice(infra_alloc) catch return error.OutOfMemory,
            .pos = 0,
            .arena = arena,
            .infra_alloc = infra_alloc,
            .errors = .{},
            .bind_param_index = 0,
        };
    }

    pub fn deinit(self: *Parser) void {
        self.infra_alloc.free(self.tokens);
        self.errors.deinit(self.infra_alloc);
    }

    /// Arena allocator for AST nodes.
    fn alloc(self: *Parser) Allocator {
        return self.arena.allocator();
    }

    // ── Token helpers ─────────────────────────────────────────────

    fn peek(self: *const Parser) Token {
        if (self.pos < self.tokens.len) return self.tokens[self.pos];
        return .{ .type = .eof, .start = @intCast(self.source.len), .len = 0 };
    }

    fn advance(self: *Parser) Token {
        const t = self.peek();
        if (self.pos < self.tokens.len) self.pos += 1;
        return t;
    }

    fn check(self: *const Parser, tt: TokenType) bool {
        return self.peek().type == tt;
    }

    /// Look ahead by `offset` tokens from current position.
    fn checkAhead(self: *const Parser, tt: TokenType, offset: u32) bool {
        const idx = self.pos + offset;
        if (idx < self.tokens.len) return self.tokens[idx].type == tt;
        return tt == .eof;
    }

    fn match(self: *Parser, tt: TokenType) bool {
        if (self.check(tt)) {
            _ = self.advance();
            return true;
        }
        return false;
    }

    fn expect(self: *Parser, tt: TokenType) Error!Token {
        if (self.check(tt)) return self.advance();
        try self.addError(self.peek(), expectedTokenMsg(tt));
        return error.ParseFailed;
    }

    fn lexeme(self: *const Parser, t: Token) []const u8 {
        return t.lexeme(self.source);
    }

    fn addError(self: *Parser, t: Token, msg: []const u8) Error!void {
        self.errors.append(self.infra_alloc, .{ .message = msg, .token = t }) catch return error.OutOfMemory;
    }

    fn expectedTokenMsg(tt: TokenType) []const u8 {
        return switch (tt) {
            .left_paren => "expected '('",
            .right_paren => "expected ')'",
            .comma => "expected ','",
            .semicolon => "expected ';'",
            .kw_from => "expected 'FROM'",
            .kw_into => "expected 'INTO'",
            .kw_values => "expected 'VALUES'",
            .kw_table => "expected 'TABLE'",
            .kw_set => "expected 'SET'",
            .kw_on => "expected 'ON'",
            .kw_as => "expected 'AS'",
            .kw_then => "expected 'THEN'",
            .kw_end => "expected 'END'",
            .kw_by => "expected 'BY'",
            .kw_key => "expected 'KEY'",
            .kw_null => "expected 'NULL'",
            .kw_not => "expected 'NOT'",
            .kw_exists => "expected 'EXISTS'",
            .kw_action => "expected 'ACTION'",
            .kw_rowid => "expected 'ROWID'",
            .kw_join => "expected 'JOIN'",
            .kw_between => "expected 'BETWEEN'",
            .kw_and => "expected 'AND'",
            .kw_references => "expected 'REFERENCES'",
            .kw_index => "expected 'INDEX'",
            .kw_savepoint => "expected 'SAVEPOINT'",
            .kw_delete => "expected 'DELETE'",
            .kw_update => "expected 'UPDATE'",
            .kw_begin => "expected 'BEGIN'",
            .kw_commit => "expected 'COMMIT'",
            .kw_rollback => "expected 'ROLLBACK'",
            .kw_explain => "expected 'EXPLAIN'",
            .kw_vacuum => "expected 'VACUUM'",
            .kw_reindex => "expected 'REINDEX'",
            .kw_select => "expected 'SELECT'",
            .kw_insert => "expected 'INSERT'",
            .kw_over => "expected 'OVER'",
            .kw_row => "expected 'ROW'",
            .identifier => "expected identifier",
            .equals => "expected '='",
            .integer_literal => "expected integer",
            else => "unexpected token",
        };
    }

    /// Expect an identifier or a keyword used as an identifier.
    fn expectIdentifier(self: *Parser) Error![]const u8 {
        const t = self.peek();
        if (t.type == .identifier or t.type == .quoted_identifier) {
            _ = self.advance();
            const text = self.lexeme(t);
            if (t.type == .quoted_identifier) {
                return text[1 .. text.len - 1];
            }
            return text;
        }
        // Allow type keywords as identifiers (common in SQL)
        if (t.type.isKeyword()) {
            _ = self.advance();
            return self.lexeme(t);
        }
        try self.addError(t, "expected identifier");
        return error.ParseFailed;
    }

    // ── Public API ────────────────────────────────────────────────

    /// Parse a single SQL statement. Returns null if at EOF.
    pub fn parseStatement(self: *Parser) Error!?ast.Stmt {
        while (self.match(.semicolon)) {}

        if (self.check(.eof)) return null;

        // Reset bind parameter counter for each statement
        self.bind_param_index = 0;

        const t = self.peek();
        const stmt: ast.Stmt = switch (t.type) {
            .kw_select => .{ .select = try self.parseSelect() },
            .kw_with => .{ .select = try self.parseSelect() },
            .kw_insert => .{ .insert = try self.parseInsert() },
            .kw_update => .{ .update = try self.parseUpdate() },
            .kw_delete => .{ .delete = try self.parseDelete() },
            .kw_create => try self.parseCreate(),
            .kw_drop => try self.parseDrop(),
            .kw_alter => try self.parseAlter(),
            .kw_begin => .{ .transaction = try self.parseBegin() },
            .kw_commit => .{ .transaction = self.parseCommit() },
            .kw_rollback => .{ .transaction = try self.parseRollback() },
            .kw_savepoint => .{ .transaction = try self.parseSavepoint() },
            .kw_release => .{ .transaction = try self.parseRelease() },
            .kw_explain => .{ .explain = try self.parseExplain() },
            .kw_analyze => .{ .analyze = self.parseAnalyze() },
            .kw_vacuum => .{ .vacuum = self.parseVacuum() },
            .kw_reindex => .{ .reindex = try self.parseReindex() },
            .kw_grant => try self.parseGrant(),
            .kw_revoke => try self.parseRevoke(),
            .kw_set => .{ .set = try self.parseSet() },
            .kw_show => .{ .show = try self.parseShow() },
            .kw_reset => .{ .reset = try self.parseReset() },
            else => {
                try self.addError(t, "expected statement");
                return error.ParseFailed;
            },
        };

        _ = self.match(.semicolon);
        return stmt;
    }

    // ── SELECT ────────────────────────────────────────────────────

    fn parseSelect(self: *Parser) Error!ast.SelectStmt {
        var stmt = ast.SelectStmt{};
        const a = self.alloc();

        // Parse optional WITH clause (CTEs) — applies to the whole compound select
        var ctes: []const ast.CteDefinition = &.{};
        var recursive = false;
        if (self.match(.kw_with)) {
            if (self.match(.kw_recursive)) {
                recursive = true;
            }
            var cte_list = std.ArrayListUnmanaged(ast.CteDefinition){};
            while (true) {
                cte_list.append(a, try self.parseCteDefinition()) catch return error.OutOfMemory;
                if (!self.match(.comma)) break;
            }
            ctes = cte_list.toOwnedSlice(a) catch return error.OutOfMemory;
        }

        stmt.ctes = ctes;
        stmt.recursive = recursive;

        // Parse the core SELECT body (without ORDER BY / LIMIT for compound queries)
        stmt = try self.parseSelectBody(stmt);

        // Check for set operations: UNION [ALL], INTERSECT, EXCEPT
        if (self.peekIsSetOpKeyword()) {
            const set_op_type = try self.parseSetOpKeyword();
            // Parse the right-hand SELECT body (no CTEs, no ORDER BY / LIMIT)
            var right_stmt = ast.SelectStmt{};
            right_stmt = try self.parseSelectBody(right_stmt);

            // Chain further set operations on the right side
            while (self.peekIsSetOpKeyword()) {
                const next_op = try self.parseSetOpKeyword();
                var next_stmt = ast.SelectStmt{};
                next_stmt = try self.parseSelectBody(next_stmt);
                const next_right = self.arena.create(ast.SelectStmt, next_stmt) catch return error.OutOfMemory;
                right_stmt.set_operation = self.arena.create(ast.SetOperation, .{
                    .op = next_op,
                    .right = next_right,
                }) catch return error.OutOfMemory;
            }

            const right_ptr = self.arena.create(ast.SelectStmt, right_stmt) catch return error.OutOfMemory;
            stmt.set_operation = self.arena.create(ast.SetOperation, .{
                .op = set_op_type,
                .right = right_ptr,
            }) catch return error.OutOfMemory;
        }

        // ORDER BY and LIMIT/OFFSET apply to the entire compound query
        if (self.match(.kw_order)) {
            _ = try self.expect(.kw_by);
            var items = std.ArrayListUnmanaged(ast.OrderByItem){};
            while (true) {
                const expr = try self.parseExpr(0);
                var dir: ast.OrderDirection = .asc;
                if (self.match(.kw_desc)) {
                    dir = .desc;
                } else {
                    _ = self.match(.kw_asc);
                }
                items.append(a, .{ .expr = expr, .direction = dir }) catch return error.OutOfMemory;
                if (!self.match(.comma)) break;
            }
            stmt.order_by = items.toOwnedSlice(a) catch return error.OutOfMemory;
        }

        if (self.match(.kw_limit)) {
            stmt.limit = try self.parseExpr(0);
            if (self.match(.kw_offset)) {
                stmt.offset = try self.parseExpr(0);
            }
        }

        return stmt;
    }

    /// Parse the body of a SELECT (without CTEs, ORDER BY, LIMIT).
    fn parseSelectBody(self: *Parser, base: ast.SelectStmt) Error!ast.SelectStmt {
        var stmt = base;
        const a = self.alloc();

        _ = try self.expect(.kw_select);

        if (self.match(.kw_distinct)) {
            // Check for DISTINCT ON (expr, ...)
            if (self.match(.kw_on)) {
                _ = try self.expect(.left_paren);
                var exprs = std.ArrayListUnmanaged(*const ast.Expr){};
                while (true) {
                    exprs.append(a, try self.parseExpr(0)) catch return error.OutOfMemory;
                    if (!self.match(.comma)) break;
                }
                _ = try self.expect(.right_paren);
                stmt.distinct = true;
                stmt.distinct_on = exprs.toOwnedSlice(a) catch return error.OutOfMemory;
            } else {
                stmt.distinct = true;
            }
        } else {
            _ = self.match(.kw_all);
        }

        stmt.columns = try self.parseResultColumns();

        if (self.match(.kw_from)) {
            stmt.from = try self.parseTableRef();

            var joins = std.ArrayListUnmanaged(ast.JoinClause){};
            while (self.peekIsJoinKeyword()) {
                joins.append(a, try self.parseJoin()) catch return error.OutOfMemory;
            }
            if (joins.items.len > 0) {
                stmt.joins = joins.toOwnedSlice(a) catch return error.OutOfMemory;
            }
        }

        if (self.match(.kw_where)) {
            stmt.where = try self.parseExpr(0);
        }

        if (self.match(.kw_group)) {
            _ = try self.expect(.kw_by);
            var exprs = std.ArrayListUnmanaged(*const ast.Expr){};
            while (true) {
                exprs.append(a, try self.parseExpr(0)) catch return error.OutOfMemory;
                if (!self.match(.comma)) break;
            }
            stmt.group_by = exprs.toOwnedSlice(a) catch return error.OutOfMemory;

            if (self.match(.kw_having)) {
                stmt.having = try self.parseExpr(0);
            }
        }

        // Parse optional WINDOW clause: WINDOW name AS (...), ...
        if (self.match(.kw_window)) {
            var defs = std.ArrayListUnmanaged(ast.WindowDef){};
            while (true) {
                const win_name = try self.expectIdentifier();
                _ = try self.expect(.kw_as);
                _ = try self.expect(.left_paren);

                // Parse PARTITION BY
                var partition_by = std.ArrayListUnmanaged(*const ast.Expr){};
                if (self.match(.kw_partition)) {
                    _ = try self.expect(.kw_by);
                    while (true) {
                        partition_by.append(a, try self.parseExpr(0)) catch return error.OutOfMemory;
                        if (!self.match(.comma)) break;
                    }
                }

                // Parse ORDER BY
                var order_by = std.ArrayListUnmanaged(ast.OrderByItem){};
                if (self.match(.kw_order)) {
                    _ = try self.expect(.kw_by);
                    while (true) {
                        const expr = try self.parseExpr(0);
                        var dir: ast.OrderDirection = .asc;
                        if (self.match(.kw_desc)) {
                            dir = .desc;
                        } else {
                            _ = self.match(.kw_asc);
                        }
                        order_by.append(a, .{ .expr = expr, .direction = dir }) catch return error.OutOfMemory;
                        if (!self.match(.comma)) break;
                    }
                }

                // Parse frame specification
                var frame: ?*const ast.WindowFrameSpec = null;
                if (self.check(.kw_rows) or self.check(.kw_range) or self.check(.kw_groups)) {
                    frame = try self.parseFrameSpec();
                }

                _ = try self.expect(.right_paren);

                defs.append(a, .{
                    .name = win_name,
                    .partition_by = partition_by.toOwnedSlice(a) catch return error.OutOfMemory,
                    .order_by = order_by.toOwnedSlice(a) catch return error.OutOfMemory,
                    .frame = frame,
                }) catch return error.OutOfMemory;

                if (!self.match(.comma)) break;
            }
            stmt.window_defs = defs.toOwnedSlice(a) catch return error.OutOfMemory;
        }

        return stmt;
    }

    /// Check if the next token is a set operation keyword.
    fn peekIsSetOpKeyword(self: *Parser) bool {
        const t = self.peek().type;
        return t == .kw_union or t == .kw_intersect or t == .kw_except;
    }

    /// Consume and return the set operation type.
    fn parseSetOpKeyword(self: *Parser) Error!ast.SetOpType {
        if (self.match(.kw_union)) {
            if (self.match(.kw_all)) return .union_all;
            return .@"union";
        }
        if (self.match(.kw_intersect)) return .intersect;
        if (self.match(.kw_except)) return .except;
        try self.addError(self.peek(), "expected UNION, INTERSECT, or EXCEPT");
        return error.ParseFailed;
    }

    /// Parse a single CTE definition: name [(col1, col2, ...)] AS (SELECT ...)
    fn parseCteDefinition(self: *Parser) Error!ast.CteDefinition {
        const a = self.alloc();
        const name = try self.expectIdentifier();

        // Optional column aliases come before AS: name(col1, col2) AS (SELECT ...)
        // If next token is '(' it must be column aliases (AS hasn't appeared yet)
        var col_names = std.ArrayListUnmanaged([]const u8){};
        if (self.check(.left_paren)) {
            _ = self.advance(); // (
            while (true) {
                col_names.append(a, try self.expectIdentifier()) catch return error.OutOfMemory;
                if (!self.match(.comma)) break;
            }
            _ = try self.expect(.right_paren);
        }

        _ = try self.expect(.kw_as);
        _ = try self.expect(.left_paren);
        const select = try self.parseSelect();
        _ = try self.expect(.right_paren);

        const select_ptr = self.arena.create(ast.SelectStmt, select) catch return error.OutOfMemory;

        return .{
            .name = name,
            .select = select_ptr,
            .column_names = if (col_names.items.len > 0)
                col_names.toOwnedSlice(a) catch return error.OutOfMemory
            else
                &.{},
        };
    }

    fn parseResultColumns(self: *Parser) Error![]const ast.ResultColumn {
        const a = self.alloc();
        var cols = std.ArrayListUnmanaged(ast.ResultColumn){};
        while (true) {
            cols.append(a, try self.parseResultColumn()) catch return error.OutOfMemory;
            if (!self.match(.comma)) break;
        }
        return cols.toOwnedSlice(a) catch return error.OutOfMemory;
    }

    fn parseResultColumn(self: *Parser) Error!ast.ResultColumn {
        if (self.match(.star)) return .all_columns;

        // table.*
        if ((self.peek().type == .identifier or self.peek().type == .quoted_identifier) and
            self.pos + 2 < self.tokens.len and
            self.tokens[self.pos + 1].type == .dot and
            self.tokens[self.pos + 2].type == .star)
        {
            const name = self.lexeme(self.advance());
            _ = self.advance(); // dot
            _ = self.advance(); // star
            return .{ .table_all_columns = name };
        }

        const expr = try self.parseExpr(0);
        var alias: ?[]const u8 = null;
        if (self.match(.kw_as)) {
            alias = try self.expectIdentifier();
        } else if (self.peek().type == .identifier) {
            alias = self.lexeme(self.advance());
        }
        return .{ .expr = .{ .value = expr, .alias = alias } };
    }

    fn parseTableRef(self: *Parser) Error!*const ast.TableRef {
        if (self.match(.left_paren)) {
            if (self.check(.kw_select)) {
                const select = try self.parseSelect();
                _ = try self.expect(.right_paren);
                _ = self.match(.kw_as);
                const alias = try self.expectIdentifier();
                const sel_ptr = self.arena.create(ast.SelectStmt, select) catch return error.OutOfMemory;
                return self.arena.create(ast.TableRef, .{
                    .subquery = .{ .select = sel_ptr, .alias = alias },
                }) catch return error.OutOfMemory;
            }
            try self.addError(self.peek(), "expected SELECT in subquery");
            return error.ParseFailed;
        }

        const name = try self.expectIdentifier();

        // Check for function call: name(...)
        if (self.match(.left_paren)) {
            var args = std.ArrayListUnmanaged(*const ast.Expr){};
            const a = self.arena.allocator();

            if (!self.check(.right_paren)) {
                while (true) {
                    const arg = try self.parseExpr(0);
                    args.append(a, arg) catch return error.OutOfMemory;
                    if (!self.match(.comma)) break;
                }
            }
            _ = try self.expect(.right_paren);

            var alias: ?[]const u8 = null;
            if (self.match(.kw_as)) {
                alias = try self.expectIdentifier();
            } else if (self.peek().type == .identifier and !self.peekIsClauseKeyword()) {
                alias = self.lexeme(self.advance());
            }

            return self.arena.create(ast.TableRef, .{
                .table_function = .{
                    .name = name,
                    .args = args.toOwnedSlice(a) catch return error.OutOfMemory,
                    .alias = alias,
                },
            }) catch return error.OutOfMemory;
        }

        // Regular table name
        var alias: ?[]const u8 = null;
        if (self.match(.kw_as)) {
            alias = try self.expectIdentifier();
        } else if (self.peek().type == .identifier and !self.peekIsClauseKeyword()) {
            alias = self.lexeme(self.advance());
        }
        return self.arena.create(ast.TableRef, .{
            .table_name = .{ .name = name, .alias = alias },
        }) catch return error.OutOfMemory;
    }

    fn peekIsClauseKeyword(self: *const Parser) bool {
        const t = self.peek().type;
        return t == .kw_where or t == .kw_order or t == .kw_group or
            t == .kw_having or t == .kw_limit or t == .kw_join or
            t == .kw_inner or t == .kw_left or t == .kw_right or
            t == .kw_full or t == .kw_cross or t == .kw_natural or
            t == .kw_on or t == .kw_set;
    }

    fn peekIsJoinKeyword(self: *const Parser) bool {
        const t = self.peek().type;
        return t == .kw_join or t == .kw_inner or t == .kw_left or
            t == .kw_right or t == .kw_full or t == .kw_cross or t == .kw_natural;
    }

    fn parseJoin(self: *Parser) Error!ast.JoinClause {
        var join_type: ast.JoinType = .inner;

        if (self.match(.kw_natural)) {}

        if (self.match(.kw_inner)) {
            join_type = .inner;
        } else if (self.match(.kw_left)) {
            _ = self.match(.kw_outer);
            join_type = .left;
        } else if (self.match(.kw_right)) {
            _ = self.match(.kw_outer);
            join_type = .right;
        } else if (self.match(.kw_full)) {
            _ = self.match(.kw_outer);
            join_type = .full;
        } else if (self.match(.kw_cross)) {
            join_type = .cross;
        }

        _ = try self.expect(.kw_join);

        const table = try self.parseTableRef();
        var on_condition: ?*const ast.Expr = null;
        if (self.match(.kw_on)) {
            on_condition = try self.parseExpr(0);
        }

        return .{
            .join_type = join_type,
            .table = table,
            .on_condition = on_condition,
        };
    }

    // ── INSERT ────────────────────────────────────────────────────

    fn parseInsert(self: *Parser) Error!ast.InsertStmt {
        const a = self.alloc();
        _ = try self.expect(.kw_insert);
        _ = try self.expect(.kw_into);

        const table = try self.expectIdentifier();

        var columns: ?[]const []const u8 = null;
        if (self.match(.left_paren)) {
            var cols = std.ArrayListUnmanaged([]const u8){};
            while (true) {
                cols.append(a, try self.expectIdentifier()) catch return error.OutOfMemory;
                if (!self.match(.comma)) break;
            }
            _ = try self.expect(.right_paren);
            columns = cols.toOwnedSlice(a) catch return error.OutOfMemory;
        }

        _ = try self.expect(.kw_values);

        var rows = std.ArrayListUnmanaged([]const *const ast.Expr){};
        while (true) {
            _ = try self.expect(.left_paren);
            var vals = std.ArrayListUnmanaged(*const ast.Expr){};
            while (true) {
                vals.append(a, try self.parseExpr(0)) catch return error.OutOfMemory;
                if (!self.match(.comma)) break;
            }
            _ = try self.expect(.right_paren);
            rows.append(a, vals.toOwnedSlice(a) catch return error.OutOfMemory) catch return error.OutOfMemory;
            if (!self.match(.comma)) break;
        }

        return .{
            .table = table,
            .columns = columns,
            .values = rows.toOwnedSlice(a) catch return error.OutOfMemory,
        };
    }

    // ── UPDATE ────────────────────────────────────────────────────

    fn parseUpdate(self: *Parser) Error!ast.UpdateStmt {
        const a = self.alloc();
        _ = try self.expect(.kw_update);
        const table = try self.expectIdentifier();
        _ = try self.expect(.kw_set);

        var assignments = std.ArrayListUnmanaged(ast.Assignment){};
        while (true) {
            const col = try self.expectIdentifier();
            _ = try self.expect(.equals);
            const val = try self.parseExpr(0);
            assignments.append(a, .{ .column = col, .value = val }) catch return error.OutOfMemory;
            if (!self.match(.comma)) break;
        }

        var where: ?*const ast.Expr = null;
        if (self.match(.kw_where)) {
            where = try self.parseExpr(0);
        }

        return .{
            .table = table,
            .assignments = assignments.toOwnedSlice(a) catch return error.OutOfMemory,
            .where = where,
        };
    }

    // ── DELETE ────────────────────────────────────────────────────

    fn parseDelete(self: *Parser) Error!ast.DeleteStmt {
        _ = try self.expect(.kw_delete);
        _ = try self.expect(.kw_from);
        const table = try self.expectIdentifier();

        var where: ?*const ast.Expr = null;
        if (self.match(.kw_where)) {
            where = try self.parseExpr(0);
        }

        return .{ .table = table, .where = where };
    }

    // ── CREATE ────────────────────────────────────────────────────

    fn parseCreate(self: *Parser) Error!ast.Stmt {
        _ = try self.expect(.kw_create);

        // CREATE OR REPLACE (VIEW, FUNCTION, or TRIGGER)
        if (self.check(.kw_or)) {
            // Peek ahead to determine if it's VIEW, FUNCTION, or TRIGGER
            if (self.checkAhead(.kw_view, 2)) {
                return .{ .create_view = try self.parseCreateView(true) };
            }
            if (self.checkAhead(.kw_function, 2)) {
                return .{ .create_function = try self.parseCreateFunction(true) };
            }
            if (self.checkAhead(.kw_trigger, 2)) {
                return .{ .create_trigger = try self.parseCreateTrigger(true) };
            }
            if (self.checkAhead(.kw_role, 2)) {
                return .{ .create_role = try self.parseCreateRole(true) };
            }
            try self.addError(self.peek(), "expected VIEW, FUNCTION, TRIGGER, or ROLE after CREATE OR REPLACE");
            return error.ParseFailed;
        }
        if (self.check(.kw_view)) {
            return .{ .create_view = try self.parseCreateView(false) };
        }
        if (self.check(.kw_function)) {
            return .{ .create_function = try self.parseCreateFunction(false) };
        }
        if (self.match(.kw_unique)) {
            return .{ .create_index = try self.parseCreateIndex(true) };
        }
        if (self.check(.kw_index)) {
            return .{ .create_index = try self.parseCreateIndex(false) };
        }
        if (self.check(.kw_table) or self.check(.kw_temp) or self.check(.kw_temporary)) {
            return .{ .create_table = try self.parseCreateTable() };
        }
        if (self.check(.kw_type)) {
            return .{ .create_type = try self.parseCreateType() };
        }
        if (self.check(.kw_domain)) {
            return .{ .create_domain = try self.parseCreateDomain() };
        }
        if (self.check(.kw_trigger)) {
            return .{ .create_trigger = try self.parseCreateTrigger(false) };
        }
        if (self.check(.kw_role)) {
            return .{ .create_role = try self.parseCreateRole(false) };
        }
        if (self.check(.kw_policy)) {
            return .{ .create_policy = try self.parseCreatePolicy() };
        }

        try self.addError(self.peek(), "expected TABLE, VIEW, INDEX, TYPE, DOMAIN, FUNCTION, TRIGGER, ROLE, or POLICY after CREATE");
        return error.ParseFailed;
    }

    fn parseCreateTable(self: *Parser) Error!ast.CreateTableStmt {
        const a = self.alloc();
        _ = self.match(.kw_temp);
        _ = self.match(.kw_temporary);

        _ = try self.expect(.kw_table);

        var if_not_exists = false;
        if (self.match(.kw_if)) {
            _ = try self.expect(.kw_not);
            _ = try self.expect(.kw_exists);
            if_not_exists = true;
        }

        const name = try self.expectIdentifier();
        _ = try self.expect(.left_paren);

        var columns = std.ArrayListUnmanaged(ast.ColumnDef){};
        var table_constraints = std.ArrayListUnmanaged(ast.TableConstraint){};

        while (!self.check(.right_paren) and !self.check(.eof)) {
            if (self.check(.kw_primary) or self.check(.kw_unique) or
                self.check(.kw_check) or self.check(.kw_foreign) or
                self.check(.kw_constraint))
            {
                table_constraints.append(a, try self.parseTableConstraint()) catch return error.OutOfMemory;
            } else {
                columns.append(a, try self.parseColumnDef()) catch return error.OutOfMemory;
            }
            if (!self.match(.comma)) break;
        }

        _ = try self.expect(.right_paren);

        var without_rowid = false;
        var strict = false;
        while (self.check(.kw_without) or self.check(.kw_strict)) {
            if (self.match(.kw_without)) {
                _ = try self.expect(.kw_rowid);
                without_rowid = true;
            } else if (self.match(.kw_strict)) {
                strict = true;
            }
            _ = self.match(.comma);
        }

        return .{
            .if_not_exists = if_not_exists,
            .name = name,
            .columns = columns.toOwnedSlice(a) catch return error.OutOfMemory,
            .table_constraints = table_constraints.toOwnedSlice(a) catch return error.OutOfMemory,
            .without_rowid = without_rowid,
            .strict = strict,
        };
    }

    fn parseColumnDef(self: *Parser) Error!ast.ColumnDef {
        const a = self.alloc();
        const name = try self.expectIdentifier();

        var data_type: ?ast.DataType = null;
        if (self.peekIsDataType()) {
            data_type = self.parseDataType();
        }

        var constraints = std.ArrayListUnmanaged(ast.ColumnConstraint){};
        while (self.peekIsColumnConstraint()) {
            constraints.append(a, try self.parseColumnConstraint()) catch return error.OutOfMemory;
        }

        return .{
            .name = name,
            .data_type = data_type,
            .constraints = if (constraints.items.len > 0)
                constraints.toOwnedSlice(a) catch return error.OutOfMemory
            else
                &.{},
        };
    }

    fn peekIsDataType(self: *const Parser) bool {
        const t = self.peek().type;
        return t == .kw_integer or t == .kw_int or t == .kw_real or
            t == .kw_text or t == .kw_blob or t == .kw_boolean or t == .kw_varchar or
            t == .kw_date or t == .kw_time or t == .kw_timestamp or t == .kw_interval or
            t == .kw_numeric or t == .kw_decimal or t == .kw_uuid or
            t == .kw_serial or t == .kw_bigserial or t == .kw_array or
            t == .kw_json or t == .kw_jsonb;
    }

    fn parseDataType(self: *Parser) ?ast.DataType {
        const t = self.peek().type;

        // Standalone ARRAY keyword (e.g., column_name ARRAY)
        if (t == .kw_array) {
            _ = self.advance();
            return .type_array;
        }

        const dt: ?ast.DataType = switch (t) {
            .kw_integer => .type_integer,
            .kw_int => .type_int,
            .kw_real => .type_real,
            .kw_text => .type_text,
            .kw_blob => .type_blob,
            .kw_boolean => .type_boolean,
            .kw_varchar => .type_varchar,
            .kw_date => .type_date,
            .kw_time => .type_time,
            .kw_timestamp => .type_timestamp,
            .kw_interval => .type_interval,
            .kw_numeric => .type_numeric,
            .kw_decimal => .type_decimal,
            .kw_uuid => .type_uuid,
            .kw_serial => .type_serial,
            .kw_bigserial => .type_bigserial,
            .kw_json => .type_json,
            .kw_jsonb => .type_jsonb,
            .kw_tsvector => .type_tsvector,
            .kw_tsquery => .type_tsquery,
            else => null,
        };
        if (dt != null) {
            _ = self.advance();
            if (self.match(.left_paren)) {
                _ = self.match(.integer_literal);
                // Handle NUMERIC(precision, scale) — consume comma and second integer
                if (self.match(.comma)) {
                    _ = self.match(.integer_literal);
                }
                _ = self.match(.right_paren);
            }
            // Support INTEGER[] and INTEGER ARRAY syntax → maps to type_array
            if (self.peek().type == .left_bracket) {
                _ = self.advance(); // consume [
                _ = self.match(.right_bracket); // consume ]
                return .type_array;
            }
            if (self.peek().type == .kw_array) {
                _ = self.advance(); // consume ARRAY
                return .type_array;
            }
        }
        return dt;
    }

    fn peekIsColumnConstraint(self: *const Parser) bool {
        const t = self.peek().type;
        return t == .kw_primary or t == .kw_not or t == .kw_unique or
            t == .kw_default or t == .kw_check or t == .kw_references or
            t == .kw_constraint;
    }

    fn parseColumnConstraint(self: *Parser) Error!ast.ColumnConstraint {
        if (self.match(.kw_constraint)) {
            _ = try self.expectIdentifier();
        }

        if (self.match(.kw_primary)) {
            _ = try self.expect(.kw_key);
            var autoincrement = false;
            _ = self.match(.kw_asc);
            _ = self.match(.kw_desc);
            if (self.match(.kw_autoincrement)) {
                autoincrement = true;
            }
            return .{ .primary_key = .{ .autoincrement = autoincrement } };
        }

        if (self.match(.kw_not)) {
            _ = try self.expect(.kw_null);
            return .not_null;
        }

        if (self.match(.kw_unique)) {
            return .unique;
        }

        if (self.match(.kw_default)) {
            if (self.match(.left_paren)) {
                const expr = try self.parseExpr(0);
                _ = try self.expect(.right_paren);
                return .{ .default = expr };
            }
            const expr = try self.parsePrimary();
            return .{ .default = expr };
        }

        if (self.match(.kw_check)) {
            _ = try self.expect(.left_paren);
            const expr = try self.parseExpr(0);
            _ = try self.expect(.right_paren);
            return .{ .check = expr };
        }

        if (self.match(.kw_references)) {
            const table = try self.expectIdentifier();
            var column: ?[]const u8 = null;
            if (self.match(.left_paren)) {
                column = try self.expectIdentifier();
                _ = try self.expect(.right_paren);
            }
            var on_delete: ?ast.ForeignKeyAction = null;
            var on_update: ?ast.ForeignKeyAction = null;
            while (self.match(.kw_on)) {
                if (self.match(.kw_delete)) {
                    on_delete = try self.parseForeignKeyAction();
                } else if (self.match(.kw_update)) {
                    on_update = try self.parseForeignKeyAction();
                } else {
                    try self.addError(self.peek(), "expected DELETE or UPDATE after ON");
                    return error.ParseFailed;
                }
            }
            return .{ .foreign_key = .{
                .table = table,
                .column = column,
                .on_delete = on_delete,
                .on_update = on_update,
            } };
        }

        try self.addError(self.peek(), "expected column constraint");
        return error.ParseFailed;
    }

    fn parseForeignKeyAction(self: *Parser) Error!ast.ForeignKeyAction {
        if (self.match(.kw_cascade)) return .cascade;
        if (self.match(.kw_restrict)) return .restrict;
        if (self.match(.kw_set)) {
            if (self.match(.kw_null)) return .set_null;
            if (self.match(.kw_default)) return .set_default;
            try self.addError(self.peek(), "expected NULL or DEFAULT after SET");
            return error.ParseFailed;
        }
        if (self.match(.kw_no)) {
            _ = try self.expect(.kw_action);
            return .no_action;
        }
        try self.addError(self.peek(), "expected foreign key action");
        return error.ParseFailed;
    }

    fn parseTableConstraint(self: *Parser) Error!ast.TableConstraint {
        const a = self.alloc();

        if (self.match(.kw_constraint)) {
            _ = try self.expectIdentifier();
        }

        if (self.match(.kw_primary)) {
            _ = try self.expect(.kw_key);
            _ = try self.expect(.left_paren);
            var cols = std.ArrayListUnmanaged([]const u8){};
            while (true) {
                cols.append(a, try self.expectIdentifier()) catch return error.OutOfMemory;
                if (!self.match(.comma)) break;
            }
            _ = try self.expect(.right_paren);
            return .{ .primary_key = .{
                .columns = cols.toOwnedSlice(a) catch return error.OutOfMemory,
            } };
        }

        if (self.match(.kw_unique)) {
            _ = try self.expect(.left_paren);
            var cols = std.ArrayListUnmanaged([]const u8){};
            while (true) {
                cols.append(a, try self.expectIdentifier()) catch return error.OutOfMemory;
                if (!self.match(.comma)) break;
            }
            _ = try self.expect(.right_paren);
            return .{ .unique = .{
                .columns = cols.toOwnedSlice(a) catch return error.OutOfMemory,
            } };
        }

        if (self.match(.kw_check)) {
            _ = try self.expect(.left_paren);
            const expr = try self.parseExpr(0);
            _ = try self.expect(.right_paren);
            return .{ .check = .{ .expr = expr } };
        }

        if (self.match(.kw_foreign)) {
            _ = try self.expect(.kw_key);
            _ = try self.expect(.left_paren);
            var cols = std.ArrayListUnmanaged([]const u8){};
            while (true) {
                cols.append(a, try self.expectIdentifier()) catch return error.OutOfMemory;
                if (!self.match(.comma)) break;
            }
            _ = try self.expect(.right_paren);
            _ = try self.expect(.kw_references);
            const ref_table = try self.expectIdentifier();
            _ = try self.expect(.left_paren);
            var ref_cols = std.ArrayListUnmanaged([]const u8){};
            while (true) {
                ref_cols.append(a, try self.expectIdentifier()) catch return error.OutOfMemory;
                if (!self.match(.comma)) break;
            }
            _ = try self.expect(.right_paren);

            var on_delete: ?ast.ForeignKeyAction = null;
            var on_update: ?ast.ForeignKeyAction = null;
            while (self.match(.kw_on)) {
                if (self.match(.kw_delete)) {
                    on_delete = try self.parseForeignKeyAction();
                } else if (self.match(.kw_update)) {
                    on_update = try self.parseForeignKeyAction();
                }
            }

            return .{ .foreign_key = .{
                .columns = cols.toOwnedSlice(a) catch return error.OutOfMemory,
                .ref_table = ref_table,
                .ref_columns = ref_cols.toOwnedSlice(a) catch return error.OutOfMemory,
                .on_delete = on_delete,
                .on_update = on_update,
            } };
        }

        try self.addError(self.peek(), "expected table constraint");
        return error.ParseFailed;
    }

    fn parseCreateIndex(self: *Parser, unique: bool) Error!ast.CreateIndexStmt {
        const a = self.alloc();
        _ = try self.expect(.kw_index);

        // Check for CONCURRENTLY keyword (parsed as identifier)
        var concurrently = false;
        if (self.check(.identifier)) {
            const peek_lexeme = self.lexeme(self.peek());
            if (std.ascii.eqlIgnoreCase(peek_lexeme, "concurrently")) {
                _ = self.advance();
                concurrently = true;
            }
        }

        var if_not_exists = false;
        if (self.match(.kw_if)) {
            _ = try self.expect(.kw_not);
            _ = try self.expect(.kw_exists);
            if_not_exists = true;
        }

        const name = try self.expectIdentifier();
        _ = try self.expect(.kw_on);
        const table = try self.expectIdentifier();
        _ = try self.expect(.left_paren);

        var cols = std.ArrayListUnmanaged(ast.OrderByItem){};
        while (true) {
            const expr = try self.parseExpr(0);
            var dir: ast.OrderDirection = .asc;
            if (self.match(.kw_desc)) {
                dir = .desc;
            } else {
                _ = self.match(.kw_asc);
            }
            cols.append(a, .{ .expr = expr, .direction = dir }) catch return error.OutOfMemory;
            if (!self.match(.comma)) break;
        }
        _ = try self.expect(.right_paren);

        // Parse optional INCLUDE clause
        var included_cols = std.ArrayListUnmanaged([]const u8){};
        if (self.match(.kw_include)) {
            _ = try self.expect(.left_paren);

            // Parse comma-separated column identifiers
            while (true) {
                const col_name = try self.expectIdentifier();
                included_cols.append(a, col_name) catch return error.OutOfMemory;
                if (!self.match(.comma)) break;
            }
            _ = try self.expect(.right_paren);

            // Validate INCLUDE list is not empty
            if (included_cols.items.len == 0) {
                try self.addError(self.peek(), "INCLUDE clause cannot be empty");
                return error.ParseFailed;
            }

            // Validate no duplicates within INCLUDE list
            for (included_cols.items, 0..) |col1, i| {
                for (included_cols.items[i + 1 ..]) |col2| {
                    if (std.mem.eql(u8, col1, col2)) {
                        const msg = try std.fmt.allocPrint(a, "duplicate column in INCLUDE: {s}", .{col1});
                        try self.addError(self.peek(), msg);
                        return error.ParseFailed;
                    }
                }
            }

            // Validate no overlap between indexed columns and INCLUDE columns
            // Extract column names from indexed expressions
            const indexed_cols = cols.toOwnedSlice(a) catch return error.OutOfMemory;
            for (indexed_cols) |idx_item| {
                if (idx_item.expr.* == .column_ref) {
                    const idx_col_name = idx_item.expr.column_ref.name;
                    for (included_cols.items) |inc_col| {
                        if (std.mem.eql(u8, idx_col_name, inc_col)) {
                            const msg = try std.fmt.allocPrint(a, "column {s} appears in both index and INCLUDE", .{inc_col});
                            try self.addError(self.peek(), msg);
                            return error.ParseFailed;
                        }
                    }
                }
            }

            // Parse optional USING clause
            var index_type: ?[]const u8 = null;
            if (self.match(.kw_using)) {
                const type_token = try self.expectIdentifier();
                index_type = type_token;
            }

            return .{
                .if_not_exists = if_not_exists,
                .unique = unique,
                .concurrently = concurrently,
                .name = name,
                .table = table,
                .columns = indexed_cols,
                .included_columns = included_cols.toOwnedSlice(a) catch return error.OutOfMemory,
                .index_type = index_type,
            };
        }

        // Parse optional USING clause
        var index_type: ?[]const u8 = null;
        if (self.match(.kw_using)) {
            const type_token = try self.expectIdentifier();
            index_type = type_token;
        }

        return .{
            .if_not_exists = if_not_exists,
            .unique = unique,
            .concurrently = concurrently,
            .name = name,
            .table = table,
            .columns = cols.toOwnedSlice(a) catch return error.OutOfMemory,
            .index_type = index_type,
        };
    }

    // ── DROP ──────────────────────────────────────────────────────

    fn parseDrop(self: *Parser) Error!ast.Stmt {
        _ = try self.expect(.kw_drop);
        if (self.check(.kw_table)) return .{ .drop_table = try self.parseDropTable() };
        if (self.check(.kw_view)) return .{ .drop_view = try self.parseDropView() };
        if (self.check(.kw_index)) return .{ .drop_index = try self.parseDropIndex() };
        if (self.check(.kw_type)) return .{ .drop_type = try self.parseDropType() };
        if (self.check(.kw_domain)) return .{ .drop_domain = try self.parseDropDomain() };
        if (self.check(.kw_function)) return .{ .drop_function = try self.parseDropFunction() };
        if (self.check(.kw_trigger)) return .{ .drop_trigger = try self.parseDropTrigger() };
        if (self.check(.kw_role)) return .{ .drop_role = try self.parseDropRole() };
        if (self.check(.kw_policy)) return .{ .drop_policy = try self.parseDropPolicy() };
        try self.addError(self.peek(), "expected TABLE, VIEW, INDEX, TYPE, DOMAIN, FUNCTION, TRIGGER, ROLE, or POLICY after DROP");
        return error.ParseFailed;
    }

    fn parseDropTable(self: *Parser) Error!ast.DropTableStmt {
        _ = try self.expect(.kw_table);
        var if_exists = false;
        if (self.match(.kw_if)) {
            _ = try self.expect(.kw_exists);
            if_exists = true;
        }
        return .{ .if_exists = if_exists, .name = try self.expectIdentifier() };
    }

    fn parseDropIndex(self: *Parser) Error!ast.DropIndexStmt {
        _ = try self.expect(.kw_index);
        var if_exists = false;
        if (self.match(.kw_if)) {
            _ = try self.expect(.kw_exists);
            if_exists = true;
        }
        return .{ .if_exists = if_exists, .name = try self.expectIdentifier() };
    }

    // ── ALTER ─────────────────────────────────────────────────────

    fn parseAlter(self: *Parser) Error!ast.Stmt {
        _ = try self.expect(.kw_alter);
        if (self.check(.kw_trigger)) return .{ .alter_trigger = try self.parseAlterTrigger() };
        if (self.check(.kw_role)) return .{ .alter_role = try self.parseAlterRole() };
        if (self.check(.kw_table)) return try self.parseAlterTable();
        try self.addError(self.peek(), "expected TRIGGER, ROLE, or TABLE after ALTER");
        return error.ParseFailed;
    }

    fn parseAlterTrigger(self: *Parser) Error!ast.AlterTriggerStmt {
        _ = try self.expect(.kw_trigger);

        const name = try self.expectIdentifier();

        // Optional ON table_name (some SQL dialects require it, some don't)
        var table_name: ?[]const u8 = null;
        if (self.match(.kw_on)) {
            table_name = try self.expectIdentifier();
        }

        // Parse ENABLE or DISABLE
        const enable: bool = if (self.match(.kw_enable))
            true
        else if (self.match(.kw_disable))
            false
        else {
            try self.addError(self.peek(), "expected ENABLE or DISABLE");
            return error.ParseFailed;
        };

        return .{
            .name = name,
            .table_name = table_name,
            .enable = enable,
        };
    }

    // ── CREATE ROLE / DROP ROLE / ALTER ROLE ──────────────────────────────────

    fn parseCreateRole(self: *Parser, or_kw_seen: bool) Error!ast.CreateRoleStmt {
        var or_replace = false;

        if (or_kw_seen) {
            // CREATE OR REPLACE ROLE
            _ = try self.expect(.kw_or);
            _ = try self.expect(.kw_replace);
            or_replace = true;
        }

        _ = try self.expect(.kw_role);

        const name = try self.expectIdentifier();

        // Parse optional WITH keyword
        _ = self.match(.kw_with);

        // Parse role options
        var options = ast.RoleOptions{};
        while (true) {
            if (self.match(.kw_login)) {
                options.login = true;
            } else if (self.match(.kw_nologin)) {
                options.login = false;
            } else if (self.match(.kw_superuser)) {
                options.superuser = true;
            } else if (self.match(.kw_nosuperuser)) {
                options.superuser = false;
            } else if (self.match(.kw_createdb)) {
                options.createdb = true;
            } else if (self.match(.kw_nocreatedb)) {
                options.createdb = false;
            } else if (self.match(.kw_createrole)) {
                options.createrole = true;
            } else if (self.match(.kw_nocreaterole)) {
                options.createrole = false;
            } else if (self.match(.kw_inherit)) {
                options.inherit = true;
            } else if (self.match(.kw_noinherit)) {
                options.inherit = false;
            } else if (self.match(.kw_password)) {
                options.password = try self.parseStringLiteral();
            } else if (self.match(.kw_valid)) {
                _ = try self.expect(.kw_until);
                options.valid_until = try self.parseStringLiteral();
            } else {
                // No more role options
                break;
            }
        }

        return .{
            .name = name,
            .options = options,
            .or_replace = or_replace,
        };
    }

    fn parseDropRole(self: *Parser) Error!ast.DropRoleStmt {
        _ = try self.expect(.kw_role);
        var if_exists = false;
        if (self.match(.kw_if)) {
            _ = try self.expect(.kw_exists);
            if_exists = true;
        }
        return .{ .if_exists = if_exists, .name = try self.expectIdentifier() };
    }

    fn parseAlterRole(self: *Parser) Error!ast.AlterRoleStmt {
        _ = try self.expect(.kw_role);

        const name = try self.expectIdentifier();

        // Parse optional WITH keyword
        _ = self.match(.kw_with);

        // Parse role options (same as CREATE ROLE)
        var options = ast.RoleOptions{};
        while (true) {
            if (self.match(.kw_login)) {
                options.login = true;
            } else if (self.match(.kw_nologin)) {
                options.login = false;
            } else if (self.match(.kw_superuser)) {
                options.superuser = true;
            } else if (self.match(.kw_nosuperuser)) {
                options.superuser = false;
            } else if (self.match(.kw_createdb)) {
                options.createdb = true;
            } else if (self.match(.kw_nocreatedb)) {
                options.createdb = false;
            } else if (self.match(.kw_createrole)) {
                options.createrole = true;
            } else if (self.match(.kw_nocreaterole)) {
                options.createrole = false;
            } else if (self.match(.kw_inherit)) {
                options.inherit = true;
            } else if (self.match(.kw_noinherit)) {
                options.inherit = false;
            } else if (self.match(.kw_password)) {
                options.password = try self.parseStringLiteral();
            } else if (self.match(.kw_valid)) {
                _ = try self.expect(.kw_until);
                options.valid_until = try self.parseStringLiteral();
            } else {
                // No more role options
                break;
            }
        }

        return .{
            .name = name,
            .options = options,
        };
    }

    // ── GRANT / REVOKE ────────────────────────────────────────────

    fn parseGrant(self: *Parser) Error!ast.Stmt {
        const a = self.alloc();
        _ = try self.expect(.kw_grant);

        // Distinguish between role membership and object privileges:
        // GRANT role TO user1, user2 (no keywords after GRANT)
        // vs
        // GRANT SELECT, INSERT ON table TO user (keywords after GRANT)

        const is_role_grant = !self.check(.kw_all) and
            !self.check(.kw_select) and
            !self.check(.kw_insert) and
            !self.check(.kw_update) and
            !self.check(.kw_delete);

        if (is_role_grant) {
            return .{ .grant_role = try self.parseGrantRole() };
        }

        // Parse privileges
        var privileges = std.ArrayListUnmanaged(ast.Privilege){};
        if (self.match(.kw_all)) {
            // GRANT ALL [PRIVILEGES]
            _ = self.match(.kw_privileges);
            privileges.append(a, .all) catch return error.OutOfMemory;
        } else {
            // GRANT privilege [, privilege ...]
            while (true) {
                const priv: ast.Privilege = if (self.match(.kw_select))
                    .select
                else if (self.match(.kw_insert))
                    .insert
                else if (self.match(.kw_update))
                    .update
                else if (self.match(.kw_delete))
                    .delete
                else {
                    try self.addError(self.peek(), "expected SELECT, INSERT, UPDATE, DELETE, or ALL");
                    return error.ParseFailed;
                };
                privileges.append(a, priv) catch return error.OutOfMemory;
                if (!self.match(.comma)) break;
            }
        }

        // ON object_type object_name
        _ = try self.expect(.kw_on);

        const object_type: ast.ObjectType = if (self.match(.kw_table))
            .table
        else if (self.check(.identifier)) // Default to table if just name
            .table
        else {
            try self.addError(self.peek(), "expected TABLE or object name");
            return error.ParseFailed;
        };

        const object_name = try self.expectIdentifier();

        // TO role_name
        _ = try self.expect(.kw_to);
        const grantee = try self.expectIdentifier();

        // Optional WITH GRANT OPTION
        var with_grant_option = false;
        if (self.match(.kw_with)) {
            _ = try self.expect(.kw_grant);
            _ = try self.expect(.kw_option);
            with_grant_option = true;
        }

        return .{ .grant = .{
            .privileges = privileges.toOwnedSlice(a) catch return error.OutOfMemory,
            .object_type = object_type,
            .object_name = object_name,
            .grantee = grantee,
            .with_grant_option = with_grant_option,
        } };
    }

    fn parseGrantRole(self: *Parser) Error!ast.GrantRoleStmt {
        const a = self.alloc();
        // Already consumed GRANT keyword

        // Parse role name
        const role = try self.expectIdentifier();

        // TO member [, member ...]
        _ = try self.expect(.kw_to);

        var members = std.ArrayListUnmanaged([]const u8){};
        while (true) {
            const member = try self.expectIdentifier();
            members.append(a, member) catch return error.OutOfMemory;
            if (!self.match(.comma)) break;
        }

        // Optional WITH ADMIN OPTION
        var with_admin_option = false;
        if (self.match(.kw_with)) {
            _ = try self.expect(.kw_admin);
            _ = try self.expect(.kw_option);
            with_admin_option = true;
        }

        return .{
            .role = role,
            .members = members.toOwnedSlice(a) catch return error.OutOfMemory,
            .with_admin_option = with_admin_option,
        };
    }

    fn parseRevoke(self: *Parser) Error!ast.Stmt {
        const a = self.alloc();
        _ = try self.expect(.kw_revoke);

        // Distinguish between role membership and object privileges (same as GRANT)
        const is_role_revoke = !self.check(.kw_all) and
            !self.check(.kw_select) and
            !self.check(.kw_insert) and
            !self.check(.kw_update) and
            !self.check(.kw_delete);

        if (is_role_revoke) {
            return .{ .revoke_role = try self.parseRevokeRole() };
        }

        // Parse privileges (same as GRANT)
        var privileges = std.ArrayListUnmanaged(ast.Privilege){};
        if (self.match(.kw_all)) {
            _ = self.match(.kw_privileges);
            privileges.append(a, .all) catch return error.OutOfMemory;
        } else {
            while (true) {
                const priv: ast.Privilege = if (self.match(.kw_select))
                    .select
                else if (self.match(.kw_insert))
                    .insert
                else if (self.match(.kw_update))
                    .update
                else if (self.match(.kw_delete))
                    .delete
                else {
                    try self.addError(self.peek(), "expected SELECT, INSERT, UPDATE, DELETE, or ALL");
                    return error.ParseFailed;
                };
                privileges.append(a, priv) catch return error.OutOfMemory;
                if (!self.match(.comma)) break;
            }
        }

        // ON object_type object_name
        _ = try self.expect(.kw_on);

        const object_type: ast.ObjectType = if (self.match(.kw_table))
            .table
        else if (self.check(.identifier))
            .table
        else {
            try self.addError(self.peek(), "expected TABLE or object name");
            return error.ParseFailed;
        };

        const object_name = try self.expectIdentifier();

        // FROM role_name (not TO!)
        if (self.match(.kw_to)) {
            try self.addError(self.peek(), "expected FROM (not TO) in REVOKE statement");
            return error.ParseFailed;
        }
        _ = try self.expect(.kw_from);
        const grantee = try self.expectIdentifier();

        return .{ .revoke = .{
            .privileges = privileges.toOwnedSlice(a) catch return error.OutOfMemory,
            .object_type = object_type,
            .object_name = object_name,
            .grantee = grantee,
        } };
    }

    fn parseRevokeRole(self: *Parser) Error!ast.RevokeRoleStmt {
        const a = self.alloc();
        // Already consumed REVOKE keyword

        // Parse role name
        const role = try self.expectIdentifier();

        // FROM member [, member ...]
        _ = try self.expect(.kw_from);

        var members = std.ArrayListUnmanaged([]const u8){};
        while (true) {
            const member = try self.expectIdentifier();
            members.append(a, member) catch return error.OutOfMemory;
            if (!self.match(.comma)) break;
        }

        return .{
            .role = role,
            .members = members.toOwnedSlice(a) catch return error.OutOfMemory,
        };
    }

    // ── CREATE POLICY / DROP POLICY / ALTER TABLE RLS ───────────

    fn parseCreatePolicy(self: *Parser) Error!ast.CreatePolicyStmt {
        _ = try self.expect(.kw_policy);

        const policy_name = try self.expectIdentifier();

        _ = try self.expect(.kw_on);
        const table_name = try self.expectIdentifier();

        // Optional AS {PERMISSIVE | RESTRICTIVE}
        var policy_type: ast.PolicyType = .permissive;
        if (self.match(.kw_as)) {
            if (self.match(.kw_permissive)) {
                policy_type = .permissive;
            } else if (self.match(.kw_restrictive)) {
                policy_type = .restrictive;
            } else {
                try self.addError(self.peek(), "expected PERMISSIVE or RESTRICTIVE after AS");
                return error.ParseFailed;
            }
        }

        // Optional FOR {ALL | SELECT | INSERT | UPDATE | DELETE}
        var command: ast.PolicyCommand = .all;
        if (self.match(.kw_for)) {
            if (self.match(.kw_all)) {
                command = .all;
            } else if (self.match(.kw_select)) {
                command = .select;
            } else if (self.match(.kw_insert)) {
                command = .insert;
            } else if (self.match(.kw_update)) {
                command = .update;
            } else if (self.match(.kw_delete)) {
                command = .delete;
            } else {
                try self.addError(self.peek(), "expected ALL, SELECT, INSERT, UPDATE, or DELETE after FOR");
                return error.ParseFailed;
            }
        }

        // Optional USING (qual)
        var using_expr: ?ast.Expr = null;
        if (self.match(.kw_using)) {
            _ = try self.expect(.left_paren);
            const expr = try self.parseExpr(0);
            using_expr = expr.*;
            _ = try self.expect(.right_paren);
        }

        // Optional WITH CHECK (with_check)
        var with_check_expr: ?ast.Expr = null;
        if (self.match(.kw_with)) {
            _ = try self.expect(.kw_check);
            _ = try self.expect(.left_paren);
            const expr = try self.parseExpr(0);
            with_check_expr = expr.*;
            _ = try self.expect(.right_paren);
        }

        return .{
            .policy_name = policy_name,
            .table_name = table_name,
            .policy_type = policy_type,
            .command = command,
            .using_expr = using_expr,
            .with_check_expr = with_check_expr,
        };
    }

    fn parseDropPolicy(self: *Parser) Error!ast.DropPolicyStmt {
        _ = try self.expect(.kw_policy);

        var if_exists = false;
        if (self.match(.kw_if)) {
            _ = try self.expect(.kw_exists);
            if_exists = true;
        }

        const policy_name = try self.expectIdentifier();

        _ = try self.expect(.kw_on);
        const table_name = try self.expectIdentifier();

        return .{
            .policy_name = policy_name,
            .table_name = table_name,
            .if_exists = if_exists,
        };
    }

    fn parseAlterTable(self: *Parser) Error!ast.Stmt {
        _ = try self.expect(.kw_table);
        const table_name = try self.expectIdentifier();

        // Check for ROW LEVEL SECURITY keywords
        if (self.match(.kw_enable)) {
            return try self.parseAlterTableRLS(table_name, true);
        } else if (self.match(.kw_disable)) {
            return try self.parseAlterTableRLS(table_name, false);
        } else if (self.match(.kw_force)) {
            _ = try self.expect(.kw_row);
            _ = try self.expect(.kw_level);
            _ = try self.expect(.kw_security);
            return .{ .alter_table_rls = .{
                .table_name = table_name,
                .enable = true,
                .force = true,
            } };
        } else if (self.match(.kw_no)) {
            _ = try self.expect(.kw_force);
            _ = try self.expect(.kw_row);
            _ = try self.expect(.kw_level);
            _ = try self.expect(.kw_security);
            return .{ .alter_table_rls = .{
                .table_name = table_name,
                .enable = false,
                .force = false,
            } };
        }

        try self.addError(self.peek(), "expected ENABLE, DISABLE, FORCE, or NO after ALTER TABLE");
        return error.ParseFailed;
    }

    fn parseAlterTableRLS(self: *Parser, table_name: []const u8, enable: bool) Error!ast.Stmt {
        // Already consumed ENABLE or DISABLE

        var force = false;
        if (self.match(.kw_force)) {
            force = true;
        }

        _ = try self.expect(.kw_row);
        _ = try self.expect(.kw_level);
        _ = try self.expect(.kw_security);

        return .{ .alter_table_rls = .{
            .table_name = table_name,
            .enable = enable,
            .force = force,
        } };
    }

    // ── CREATE VIEW / DROP VIEW ──────────────────────────────────

    fn parseCreateView(self: *Parser, or_kw_seen: bool) Error!ast.CreateViewStmt {
        const a = self.alloc();
        var or_replace = false;

        if (or_kw_seen) {
            // CREATE OR REPLACE VIEW
            _ = try self.expect(.kw_or);
            _ = try self.expect(.kw_replace);
            or_replace = true;
        }

        _ = try self.expect(.kw_view);

        var if_not_exists = false;
        if (self.match(.kw_if)) {
            _ = try self.expect(.kw_not);
            _ = try self.expect(.kw_exists);
            if_not_exists = true;
        }

        const name = try self.expectIdentifier();

        // Optional column list: CREATE VIEW v (col1, col2) AS ...
        var column_names = std.ArrayListUnmanaged([]const u8){};
        if (self.match(.left_paren)) {
            while (!self.check(.right_paren) and !self.check(.eof)) {
                const col = try self.expectIdentifier();
                column_names.append(a, col) catch return error.OutOfMemory;
                if (!self.match(.comma)) break;
            }
            _ = try self.expect(.right_paren);
        }

        _ = try self.expect(.kw_as);

        const select = try self.parseSelect();

        // Optional: WITH [LOCAL | CASCADED] CHECK OPTION
        var check_option: ast.CheckOption = .none;
        if (self.check(.kw_with)) {
            // Peek ahead to distinguish WITH CHECK OPTION from other WITH clauses
            if (self.checkAhead(.kw_check, 1) or
                self.checkAhead(.kw_local, 1) or
                self.checkAhead(.kw_cascaded, 1))
            {
                _ = self.advance(); // consume WITH
                if (self.match(.kw_local)) {
                    check_option = .local;
                } else if (self.match(.kw_cascaded)) {
                    check_option = .cascaded;
                } else {
                    // WITH CHECK OPTION (default is CASCADED per SQL standard)
                    check_option = .cascaded;
                }
                _ = try self.expect(.kw_check);
                _ = try self.expect(.kw_option);
            }
        }

        return .{
            .name = name,
            .select = select,
            .or_replace = or_replace,
            .if_not_exists = if_not_exists,
            .column_names = column_names.toOwnedSlice(a) catch return error.OutOfMemory,
            .check_option = check_option,
        };
    }

    fn parseDropView(self: *Parser) Error!ast.DropViewStmt {
        _ = try self.expect(.kw_view);
        var if_exists = false;
        if (self.match(.kw_if)) {
            _ = try self.expect(.kw_exists);
            if_exists = true;
        }
        return .{ .if_exists = if_exists, .name = try self.expectIdentifier() };
    }

    // ── CREATE TYPE / DROP TYPE ───────────────────────────────────

    fn parseCreateType(self: *Parser) Error!ast.CreateTypeStmt {
        const a = self.alloc();
        _ = try self.expect(.kw_type);
        const name = try self.expectIdentifier();
        _ = try self.expect(.kw_as);
        _ = try self.expect(.kw_enum);
        _ = try self.expect(.left_paren);

        var values = std.ArrayListUnmanaged([]const u8){};

        // Parse enum values (string literals)
        while (true) {
            const tok = self.peek();
            if (tok.type != .string_literal) {
                try self.addError(tok, "expected string literal for enum value");
                return error.ParseFailed;
            }
            values.append(a, tok.lexeme(self.source)) catch return error.OutOfMemory;
            _ = self.advance();

            if (!self.match(.comma)) break;
        }

        _ = try self.expect(.right_paren);

        return .{ .name = name, .values = values.toOwnedSlice(a) catch return error.OutOfMemory };
    }

    fn parseDropType(self: *Parser) Error!ast.DropTypeStmt {
        _ = try self.expect(.kw_type);
        var if_exists = false;
        if (self.match(.kw_if)) {
            _ = try self.expect(.kw_exists);
            if_exists = true;
        }
        return .{ .if_exists = if_exists, .name = try self.expectIdentifier() };
    }

    fn parseCreateDomain(self: *Parser) Error!ast.CreateDomainStmt {
        _ = try self.expect(.kw_domain);
        const name = try self.expectIdentifier();
        _ = try self.expect(.kw_as);
        const base_type = self.parseDataType() orelse {
            try self.addError(self.peek(), "expected data type after AS");
            return error.ParseFailed;
        };

        var constraint: ?*const ast.Expr = null;
        if (self.match(.kw_check)) {
            _ = try self.expect(.left_paren);
            constraint = try self.parseExpr(0);
            _ = try self.expect(.right_paren);
        }

        return .{ .name = name, .base_type = base_type, .constraint = constraint };
    }

    fn parseDropDomain(self: *Parser) Error!ast.DropDomainStmt {
        _ = try self.expect(.kw_domain);
        var if_exists = false;
        if (self.match(.kw_if)) {
            _ = try self.expect(.kw_exists);
            if_exists = true;
        }
        return .{ .if_exists = if_exists, .name = try self.expectIdentifier() };
    }

    // ── CREATE FUNCTION / DROP FUNCTION ──────────────────────────

    fn parseCreateFunction(self: *Parser, or_kw_seen: bool) Error!ast.CreateFunctionStmt {
        const a = self.alloc();
        var or_replace = false;

        if (or_kw_seen) {
            // CREATE OR REPLACE FUNCTION
            _ = try self.expect(.kw_or);
            _ = try self.expect(.kw_replace);
            or_replace = true;
        }

        _ = try self.expect(.kw_function);

        const name = try self.expectIdentifier();

        // Parse parameter list: (param1 type1, param2 type2, ...)
        _ = try self.expect(.left_paren);
        var params = std.ArrayListUnmanaged(ast.FunctionParam){};
        while (!self.check(.right_paren) and !self.check(.eof)) {
            const param_name = try self.expectIdentifier();
            const param_type = self.parseDataType() orelse {
                try self.addError(self.peek(), "expected data type for parameter");
                return error.ParseFailed;
            };
            params.append(a, .{ .name = param_name, .data_type = param_type }) catch return error.OutOfMemory;
            if (!self.match(.comma)) break;
        }
        _ = try self.expect(.right_paren);

        // Parse RETURNS clause
        _ = try self.expect(.kw_returns);
        const return_type = try self.parseFunctionReturn();

        // Parse optional LANGUAGE clause
        var language: []const u8 = "sfl"; // default to SFL
        if (self.match(.kw_language)) {
            language = try self.expectIdentifier();
        }

        // Parse optional volatility category
        var volatility: ast.FunctionVolatility = .vol;
        if (self.match(.kw_immutable)) {
            volatility = .immutable;
        } else if (self.match(.kw_stable)) {
            volatility = .stable;
        } else if (self.match(.kw_volatile)) {
            volatility = .vol;
        }

        // Parse AS 'body' or AS $$ body $$
        _ = try self.expect(.kw_as);
        const body = try self.parseStringLiteral();

        return .{
            .name = name,
            .parameters = params.toOwnedSlice(a) catch return error.OutOfMemory,
            .return_type = return_type,
            .language = language,
            .body = body,
            .volatility = volatility,
            .or_replace = or_replace,
        };
    }

    fn parseFunctionReturn(self: *Parser) Error!ast.FunctionReturn {
        const a = self.alloc();

        // RETURNS SETOF type_name
        if (self.match(.kw_setof)) {
            const element_type = self.parseDataType() orelse {
                try self.addError(self.peek(), "expected data type after SETOF");
                return error.ParseFailed;
            };
            return .{ .setof = element_type };
        }

        // RETURNS TABLE(col1 type1, col2 type2, ...)
        if (self.match(.kw_table)) {
            _ = try self.expect(.left_paren);
            var columns = std.ArrayListUnmanaged(ast.ColumnDef){};
            while (!self.check(.right_paren) and !self.check(.eof)) {
                const col_name = try self.expectIdentifier();
                const col_type = self.parseDataType() orelse {
                    try self.addError(self.peek(), "expected data type for column");
                    return error.ParseFailed;
                };
                columns.append(a, .{
                    .name = col_name,
                    .data_type = col_type,
                    .constraints = &.{},
                }) catch return error.OutOfMemory;
                if (!self.match(.comma)) break;
            }
            _ = try self.expect(.right_paren);
            return .{ .table = columns.toOwnedSlice(a) catch return error.OutOfMemory };
        }

        // RETURNS type_name (scalar)
        const scalar_type = self.parseDataType() orelse {
            try self.addError(self.peek(), "expected return type after RETURNS");
            return error.ParseFailed;
        };
        return .{ .scalar = scalar_type };
    }

    fn parseDropFunction(self: *Parser) Error!ast.DropFunctionStmt {
        const a = self.alloc();
        _ = try self.expect(.kw_function);

        var if_exists = false;
        if (self.match(.kw_if)) {
            _ = try self.expect(.kw_exists);
            if_exists = true;
        }

        const name = try self.expectIdentifier();

        // Optional parameter type list for overload resolution: DROP FUNCTION foo(INTEGER, TEXT)
        var param_types = std.ArrayListUnmanaged(ast.DataType){};
        if (self.match(.left_paren)) {
            while (!self.check(.right_paren) and !self.check(.eof)) {
                const ptype = self.parseDataType() orelse {
                    try self.addError(self.peek(), "expected data type");
                    return error.ParseFailed;
                };
                param_types.append(a, ptype) catch return error.OutOfMemory;
                if (!self.match(.comma)) break;
            }
            _ = try self.expect(.right_paren);
        }

        return .{
            .name = name,
            .param_types = param_types.toOwnedSlice(a) catch return error.OutOfMemory,
            .if_exists = if_exists,
        };
    }

    fn parseStringLiteral(self: *Parser) Error![]const u8 {
        const t = self.advance();
        if (t.type != .string_literal) {
            try self.addError(t, "expected string literal");
            return error.ParseFailed;
        }
        return self.lexeme(t);
    }

    // ── CREATE TRIGGER / DROP TRIGGER ───────────────────────────────

    fn parseCreateTrigger(self: *Parser, or_kw_seen: bool) Error!ast.CreateTriggerStmt {
        const a = self.alloc();
        var or_replace = false;

        if (or_kw_seen) {
            // CREATE OR REPLACE TRIGGER
            _ = try self.expect(.kw_or);
            _ = try self.expect(.kw_replace);
            or_replace = true;
        }

        _ = try self.expect(.kw_trigger);

        const name = try self.expectIdentifier();

        // Parse timing: BEFORE, AFTER, or INSTEAD OF
        const timing: ast.TriggerTiming = if (self.match(.kw_before))
            .before
        else if (self.match(.kw_after))
            .after
        else if (self.match(.kw_instead)) blk: {
            _ = try self.expect(.kw_of);
            break :blk .instead_of;
        } else {
            try self.addError(self.peek(), "expected BEFORE, AFTER, or INSTEAD OF");
            return error.ParseFailed;
        };

        // Parse event: INSERT, UPDATE, DELETE, or TRUNCATE
        const event: ast.TriggerEvent = if (self.match(.kw_insert))
            .insert
        else if (self.match(.kw_update))
            .update
        else if (self.match(.kw_delete))
            .delete
        else if (self.match(.kw_truncate))
            .truncate
        else {
            try self.addError(self.peek(), "expected INSERT, UPDATE, DELETE, or TRUNCATE");
            return error.ParseFailed;
        };

        // Parse UPDATE OF column_list (optional, only for UPDATE events)
        var update_columns = std.ArrayListUnmanaged([]const u8){};
        if (event == .update and self.match(.kw_of)) {
            while (true) {
                const col = try self.expectIdentifier();
                update_columns.append(a, col) catch return error.OutOfMemory;
                if (!self.match(.comma)) break;
            }
        }

        // Parse ON table_name
        _ = try self.expect(.kw_on);
        const table_name = try self.expectIdentifier();

        // Parse FOR EACH [ROW | STATEMENT] (default ROW)
        var level: ast.TriggerLevel = .row;
        if (self.match(.kw_for)) {
            _ = try self.expect(.kw_each);
            if (self.match(.kw_row)) {
                level = .row;
            } else if (self.match(.kw_statement)) {
                level = .statement;
            } else {
                try self.addError(self.peek(), "expected ROW or STATEMENT after FOR EACH");
                return error.ParseFailed;
            }
        }

        // Parse WHEN (condition) clause (optional)
        var when_condition: ?*const ast.Expr = null;
        if (self.match(.kw_when)) {
            _ = try self.expect(.left_paren);
            when_condition = try self.parseExpr(0);
            _ = try self.expect(.right_paren);
        }

        // Parse trigger body: AS 'body' or just 'body' as string
        // For simplicity, we expect AS 'body' similar to functions
        _ = try self.expect(.kw_as);
        const body = try self.parseStringLiteral();

        return .{
            .name = name,
            .table_name = table_name,
            .timing = timing,
            .event = event,
            .update_columns = update_columns.toOwnedSlice(a) catch return error.OutOfMemory,
            .level = level,
            .when_condition = when_condition,
            .body = body,
            .or_replace = or_replace,
        };
    }

    fn parseDropTrigger(self: *Parser) Error!ast.DropTriggerStmt {
        _ = try self.expect(.kw_trigger);

        var if_exists = false;
        if (self.match(.kw_if)) {
            _ = try self.expect(.kw_exists);
            if_exists = true;
        }

        const name = try self.expectIdentifier();

        // Optional ON table_name (some SQL dialects require it, some don't)
        var table_name: ?[]const u8 = null;
        if (self.match(.kw_on)) {
            table_name = try self.expectIdentifier();
        }

        return .{
            .name = name,
            .table_name = table_name,
            .if_exists = if_exists,
        };
    }

    // ── Transaction ───────────────────────────────────────────────

    fn parseBegin(self: *Parser) Error!ast.TransactionStmt {
        _ = try self.expect(.kw_begin);
        var mode: ast.TransactionMode = .deferred;
        if (self.match(.kw_deferred)) {
            mode = .deferred;
        } else if (self.match(.kw_immediate)) {
            mode = .immediate;
        } else if (self.match(.kw_exclusive)) {
            mode = .exclusive;
        }
        _ = self.match(.kw_transaction);

        // Parse optional ISOLATION LEVEL clause
        var isolation_level: ?ast.IsolationLevel = null;
        if (self.match(.kw_isolation)) {
            _ = try self.expect(.kw_level);
            if (self.match(.kw_read)) {
                _ = try self.expect(.kw_committed);
                isolation_level = .read_committed;
            } else if (self.match(.kw_repeatable)) {
                _ = try self.expect(.kw_read);
                isolation_level = .repeatable_read;
            } else if (self.match(.kw_serializable)) {
                isolation_level = .serializable;
            } else {
                try self.addError(self.peek(), "Expected READ COMMITTED, REPEATABLE READ, or SERIALIZABLE");
                return Error.ParseFailed;
            }
        }

        return .{ .begin = .{ .mode = mode, .isolation_level = isolation_level } };
    }

    fn parseCommit(self: *Parser) ast.TransactionStmt {
        _ = self.advance();
        _ = self.match(.kw_transaction);
        return .commit;
    }

    fn parseRollback(self: *Parser) Error!ast.TransactionStmt {
        _ = try self.expect(.kw_rollback);
        _ = self.match(.kw_transaction);
        if (self.match(.kw_to)) {
            _ = self.match(.kw_savepoint);
            return .{ .rollback = .{ .savepoint = try self.expectIdentifier() } };
        }
        return .{ .rollback = .{} };
    }

    fn parseSavepoint(self: *Parser) Error!ast.TransactionStmt {
        _ = try self.expect(.kw_savepoint);
        return .{ .savepoint = try self.expectIdentifier() };
    }

    fn parseRelease(self: *Parser) Error!ast.TransactionStmt {
        _ = try self.expect(.kw_release);
        _ = self.match(.kw_savepoint);
        return .{ .release = try self.expectIdentifier() };
    }

    // ── EXPLAIN ───────────────────────────────────────────────────

    fn parseExplain(self: *Parser) Error!ast.ExplainStmt {
        _ = try self.expect(.kw_explain);

        // Check for optional ANALYZE keyword
        const analyze = self.match(.kw_analyze);

        const inner = try self.parseStatement() orelse {
            try self.addError(self.peek(), "expected statement after EXPLAIN");
            return error.ParseFailed;
        };
        const stmt_ptr = self.arena.create(ast.Stmt, inner) catch return error.OutOfMemory;
        return .{ .stmt = stmt_ptr, .analyze = analyze };
    }

    // ── VACUUM ────────────────────────────────────────────────────

    fn parseVacuum(self: *Parser) ast.VacuumStmt {
        _ = self.advance(); // consume VACUUM keyword
        // Optional table name
        const table_name = if (self.check(.identifier))
            self.advance().lexeme(self.source)
        else if (self.peek().type.isKeyword())
            self.advance().lexeme(self.source) // allow keywords as table names
        else
            null;
        return .{ .table_name = table_name };
    }

    fn parseAnalyze(self: *Parser) ast.AnalyzeStmt {
        _ = self.advance(); // consume ANALYZE keyword
        // Optional table name
        const table_name = if (self.check(.identifier))
            self.advance().lexeme(self.source)
        else if (self.peek().type.isKeyword())
            self.advance().lexeme(self.source) // allow keywords as table names
        else
            null;
        return .{ .table_name = table_name };
    }

    fn parseReindex(self: *Parser) Error!ast.ReindexStmt {
        _ = self.advance(); // consume REINDEX keyword

        // REINDEX INDEX <name> | REINDEX TABLE <name> | REINDEX DATABASE
        const next = self.peek();

        if (self.match(.kw_index)) {
            // REINDEX INDEX <index_name>
            const name = try self.expect(.identifier);
            return .{ .index = name.lexeme(self.source) };
        } else if (self.match(.kw_table)) {
            // REINDEX TABLE <table_name>
            const name = try self.expect(.identifier);
            return .{ .table = name.lexeme(self.source) };
        } else if (self.match(.kw_database)) {
            // REINDEX DATABASE
            return .{ .database = {} };
        } else {
            try self.addError(next, "expected INDEX, TABLE, or DATABASE after REINDEX");
            return error.ParseFailed;
        }
    }

    // ── Configuration statements ──────────────────────────────────

    fn parseSet(self: *Parser) Error!ast.SetStmt {
        _ = self.advance(); // consume SET keyword

        // Get parameter name
        const parameter = try self.expectIdentifier();

        // Expect = or TO
        if (!self.match(.equals) and !self.match(.kw_to)) {
            try self.addError(self.peek(), "expected '=' or 'TO' after parameter name");
            return error.ParseFailed;
        }

        // Get value (can be identifier, string literal, or integer)
        const value_token = self.peek();
        const value = switch (value_token.type) {
            .string_literal => blk: {
                _ = self.advance();
                const text = self.lexeme(value_token);
                // Strip quotes from string literals
                break :blk if (text.len >= 2) text[1 .. text.len - 1] else text;
            },
            .integer_literal => blk: {
                _ = self.advance();
                break :blk self.lexeme(value_token);
            },
            .identifier => blk: {
                _ = self.advance();
                break :blk self.lexeme(value_token);
            },
            else => {
                try self.addError(value_token, "expected value after '=' or 'TO'");
                return error.ParseFailed;
            },
        };

        return .{
            .parameter = parameter,
            .value = value,
        };
    }

    fn parseShow(self: *Parser) Error!ast.ShowStmt {
        _ = self.advance(); // consume SHOW keyword

        // Check for SHOW ALL
        if (self.match(.kw_all)) {
            return .{ .parameter = null };
        }

        // Otherwise expect a parameter name
        const param_token = self.peek();
        if (param_token.type != .identifier) {
            try self.addError(param_token, "expected parameter name or 'ALL' after SHOW");
            return error.ParseFailed;
        }

        const parameter = try self.expectIdentifier();
        return .{ .parameter = parameter };
    }

    fn parseReset(self: *Parser) Error!ast.ResetStmt {
        _ = self.advance(); // consume RESET keyword

        // Check for RESET ALL
        if (self.match(.kw_all)) {
            return .{ .parameter = null };
        }

        // Otherwise expect a parameter name
        const param_token = self.peek();
        if (param_token.type != .identifier) {
            try self.addError(param_token, "expected parameter name or 'ALL' after RESET");
            return error.ParseFailed;
        }

        const parameter = try self.expectIdentifier();
        return .{ .parameter = parameter };
    }

    // ── Expression parser (Pratt / precedence climbing) ──────────

    fn parseExpr(self: *Parser, min_prec: u8) Error!*const ast.Expr {
        var left = try self.parsePrimary();

        // Postfix subscript: expr[index]
        while (self.check(.left_bracket)) {
            _ = self.advance(); // consume [
            const index = try self.parseExpr(0);
            _ = try self.expect(.right_bracket);
            left = self.arena.create(ast.Expr, .{ .array_subscript = .{
                .array = left,
                .index = index,
            } }) catch return error.OutOfMemory;
        }

        while (true) {
            const prec = self.currentPrecedence();
            if (prec < min_prec) break;

            if (self.check(.kw_is)) {
                left = try self.parseIsExpr(left);
                continue;
            }
            if (self.check(.kw_not) and self.peekNot()) {
                left = try self.parseNotInfix(left);
                continue;
            }
            if (self.check(.kw_between)) {
                left = try self.parseBetweenExpr(left, false);
                continue;
            }
            if (self.check(.kw_in)) {
                left = try self.parseInExpr(left, false);
                continue;
            }
            if (self.check(.kw_like)) {
                left = try self.parseLikeExpr(left, false);
                continue;
            }

            const op = self.currentBinaryOp() orelse break;
            _ = self.advance();

            // Check for ANY/ALL after comparison operator
            if (self.check(.kw_any)) {
                _ = self.advance(); // consume ANY
                _ = try self.expect(.left_paren);
                const array = try self.parseExpr(0);
                _ = try self.expect(.right_paren);
                left = self.arena.create(ast.Expr, .{ .any = .{
                    .expr = left,
                    .op = op,
                    .array = array,
                } }) catch return error.OutOfMemory;
                continue;
            }
            if (self.check(.kw_all)) {
                _ = self.advance(); // consume ALL
                _ = try self.expect(.left_paren);
                const array = try self.parseExpr(0);
                _ = try self.expect(.right_paren);
                left = self.arena.create(ast.Expr, .{ .all = .{
                    .expr = left,
                    .op = op,
                    .array = array,
                } }) catch return error.OutOfMemory;
                continue;
            }

            const right = try self.parseExpr(prec + 1);
            left = self.arena.create(ast.Expr, .{ .binary_op = .{
                .op = op,
                .left = left,
                .right = right,
            } }) catch return error.OutOfMemory;
        }

        return left;
    }

    fn parsePrimary(self: *Parser) Error!*const ast.Expr {
        const t = self.peek();

        switch (t.type) {
            .integer_literal => {
                _ = self.advance();
                const val = std.fmt.parseInt(i64, self.lexeme(t), 10) catch 0;
                return self.arena.create(ast.Expr, .{ .integer_literal = val }) catch return error.OutOfMemory;
            },
            .float_literal => {
                _ = self.advance();
                const val = std.fmt.parseFloat(f64, self.lexeme(t)) catch 0.0;
                return self.arena.create(ast.Expr, .{ .float_literal = val }) catch return error.OutOfMemory;
            },
            .string_literal => {
                _ = self.advance();
                const text = self.lexeme(t);
                const inner = if (text.len >= 2) text[1 .. text.len - 1] else text;
                return self.arena.create(ast.Expr, .{ .string_literal = inner }) catch return error.OutOfMemory;
            },
            .blob_literal => {
                _ = self.advance();
                return self.arena.create(ast.Expr, .{ .blob_literal = self.lexeme(t) }) catch return error.OutOfMemory;
            },
            .kw_true => {
                _ = self.advance();
                return self.arena.create(ast.Expr, .{ .boolean_literal = true }) catch return error.OutOfMemory;
            },
            .kw_false => {
                _ = self.advance();
                return self.arena.create(ast.Expr, .{ .boolean_literal = false }) catch return error.OutOfMemory;
            },
            .kw_null => {
                _ = self.advance();
                return self.arena.create(ast.Expr, .null_literal) catch return error.OutOfMemory;
            },
            .placeholder => {
                // Bind parameter placeholder: ?
                _ = self.advance();
                const idx = self.bind_param_index;
                self.bind_param_index += 1;
                return self.arena.create(ast.Expr, .{ .bind_parameter = idx }) catch return error.OutOfMemory;
            },
            .minus => {
                _ = self.advance();
                const operand = try self.parsePrimary();
                return self.arena.create(ast.Expr, .{ .unary_op = .{
                    .op = .negate,
                    .operand = operand,
                } }) catch return error.OutOfMemory;
            },
            .plus => {
                // Unary plus — just skip it
                _ = self.advance();
                return self.parsePrimary();
            },
            .kw_not => {
                _ = self.advance();
                // Check for NOT EXISTS
                if (self.check(.kw_exists)) {
                    _ = self.advance(); // consume EXISTS
                    _ = try self.expect(.left_paren);
                    const sel = try self.parseSelect();
                    _ = try self.expect(.right_paren);
                    const sel_ptr = self.arena.create(ast.SelectStmt, sel) catch return error.OutOfMemory;
                    return self.arena.create(ast.Expr, .{ .exists = .{
                        .subquery = sel_ptr,
                        .negated = true,
                    } }) catch return error.OutOfMemory;
                }
                const operand = try self.parseExpr(12);
                return self.arena.create(ast.Expr, .{ .unary_op = .{
                    .op = .not,
                    .operand = operand,
                } }) catch return error.OutOfMemory;
            },
            .bitwise_not => {
                _ = self.advance();
                const operand = try self.parsePrimary();
                return self.arena.create(ast.Expr, .{ .unary_op = .{
                    .op = .bitwise_not,
                    .operand = operand,
                } }) catch return error.OutOfMemory;
            },
            .kw_exists => {
                _ = self.advance(); // consume EXISTS
                _ = try self.expect(.left_paren);
                const sel = try self.parseSelect();
                _ = try self.expect(.right_paren);
                const sel_ptr = self.arena.create(ast.SelectStmt, sel) catch return error.OutOfMemory;
                return self.arena.create(ast.Expr, .{ .exists = .{
                    .subquery = sel_ptr,
                    .negated = false,
                } }) catch return error.OutOfMemory;
            },
            .left_paren => {
                _ = self.advance();
                if (self.check(.kw_select)) {
                    const sel = try self.parseSelect();
                    _ = try self.expect(.right_paren);
                    const sel_ptr = self.arena.create(ast.SelectStmt, sel) catch return error.OutOfMemory;
                    return self.arena.create(ast.Expr, .{ .subquery = sel_ptr }) catch return error.OutOfMemory;
                }
                const inner = try self.parseExpr(0);
                _ = try self.expect(.right_paren);
                return self.arena.create(ast.Expr, .{ .paren = inner }) catch return error.OutOfMemory;
            },
            .kw_array => return self.parseArrayConstructor(),
            .kw_case => return self.parseCaseExpr(),
            .kw_cast => return self.parseCastExpr(),
            .kw_count, .kw_sum, .kw_avg, .kw_min, .kw_max => return self.parseFunctionCall(),
            // Window function keywords — always followed by ()
            .kw_row_number,
            .kw_rank,
            .kw_dense_rank,
            .kw_ntile,
            .kw_lag,
            .kw_lead,
            .kw_first_value,
            .kw_last_value,
            .kw_nth_value,
            .kw_percent_rank,
            .kw_cume_dist,
            => return self.parseFunctionCall(),
            .identifier, .quoted_identifier => {
                if (self.pos + 1 < self.tokens.len and self.tokens[self.pos + 1].type == .left_paren) {
                    return self.parseFunctionCall();
                }
                return self.parseColumnRef();
            },
            .star => {
                _ = self.advance();
                return self.arena.create(ast.Expr, .{ .column_ref = .{ .name = "*" } }) catch return error.OutOfMemory;
            },
            else => {
                // Allow SQL keywords to be used as column names / identifiers
                // (e.g., "temp", "name", "type", "key", "value" are common column names
                // that happen to be SQL keywords).
                if (t.type.isKeyword()) {
                    if (self.pos + 1 < self.tokens.len and self.tokens[self.pos + 1].type == .left_paren) {
                        return self.parseFunctionCall();
                    }
                    return self.parseColumnRef();
                }
                try self.addError(t, "expected expression");
                return error.ParseFailed;
            },
        }
    }

    fn parseColumnRef(self: *Parser) Error!*const ast.Expr {
        const t = self.advance();
        var name_text = self.lexeme(t);
        if (t.type == .quoted_identifier and name_text.len >= 2) {
            name_text = name_text[1 .. name_text.len - 1];
        }

        if (self.match(.dot)) {
            const col_token = self.advance();
            var col_text = self.lexeme(col_token);
            if (col_token.type == .quoted_identifier and col_text.len >= 2) {
                col_text = col_text[1 .. col_text.len - 1];
            }
            return self.arena.create(ast.Expr, .{ .column_ref = .{
                .name = col_text,
                .prefix = name_text,
            } }) catch return error.OutOfMemory;
        }

        return self.arena.create(ast.Expr, .{ .column_ref = .{ .name = name_text } }) catch return error.OutOfMemory;
    }

    fn parseFunctionCall(self: *Parser) Error!*const ast.Expr {
        const a = self.alloc();
        const name_token = self.advance();
        const name = self.lexeme(name_token);
        _ = try self.expect(.left_paren);

        var distinct = false;
        var args = std.ArrayListUnmanaged(*const ast.Expr){};

        if (self.match(.star)) {
            const star_expr = self.arena.create(ast.Expr, .{ .column_ref = .{ .name = "*" } }) catch return error.OutOfMemory;
            args.append(a, star_expr) catch return error.OutOfMemory;
        } else if (!self.check(.right_paren)) {
            if (self.match(.kw_distinct)) {
                distinct = true;
            }
            while (true) {
                args.append(a, try self.parseExpr(0)) catch return error.OutOfMemory;
                if (!self.match(.comma)) break;
            }
        }

        _ = try self.expect(.right_paren);

        // Check for OVER clause — converts to window function
        if (self.check(.kw_over)) {
            return self.parseWindowSpec(name, args.toOwnedSlice(a) catch return error.OutOfMemory, distinct);
        }

        return self.arena.create(ast.Expr, .{ .function_call = .{
            .name = name,
            .args = args.toOwnedSlice(a) catch return error.OutOfMemory,
            .distinct = distinct,
        } }) catch return error.OutOfMemory;
    }

    /// Parse OVER (...) or OVER window_name after a function call.
    fn parseWindowSpec(self: *Parser, name: []const u8, func_args: []const *const ast.Expr, distinct: bool) Error!*const ast.Expr {
        const a = self.alloc();
        _ = try self.expect(.kw_over);

        // OVER window_name — named window reference
        if (self.check(.identifier)) {
            const win_name = try self.expectIdentifier();
            return self.arena.create(ast.Expr, .{ .window_function = .{
                .name = name,
                .args = func_args,
                .distinct = distinct,
                .window_name = win_name,
            } }) catch return error.OutOfMemory;
        }

        _ = try self.expect(.left_paren);

        // Parse PARTITION BY
        var partition_by = std.ArrayListUnmanaged(*const ast.Expr){};
        if (self.match(.kw_partition)) {
            _ = try self.expect(.kw_by);
            while (true) {
                partition_by.append(a, try self.parseExpr(0)) catch return error.OutOfMemory;
                if (!self.match(.comma)) break;
            }
        }

        // Parse ORDER BY
        var order_by = std.ArrayListUnmanaged(ast.OrderByItem){};
        if (self.match(.kw_order)) {
            _ = try self.expect(.kw_by);
            while (true) {
                const expr = try self.parseExpr(0);
                var dir: ast.OrderDirection = .asc;
                if (self.match(.kw_desc)) {
                    dir = .desc;
                } else {
                    _ = self.match(.kw_asc);
                }
                order_by.append(a, .{ .expr = expr, .direction = dir }) catch return error.OutOfMemory;
                if (!self.match(.comma)) break;
            }
        }

        // Parse frame specification: ROWS | RANGE | GROUPS [BETWEEN ... AND ...]
        var frame: ?*const ast.WindowFrameSpec = null;
        if (self.check(.kw_rows) or self.check(.kw_range) or self.check(.kw_groups)) {
            frame = try self.parseFrameSpec();
        }

        _ = try self.expect(.right_paren);

        return self.arena.create(ast.Expr, .{ .window_function = .{
            .name = name,
            .args = func_args,
            .distinct = distinct,
            .partition_by = partition_by.toOwnedSlice(a) catch return error.OutOfMemory,
            .order_by = order_by.toOwnedSlice(a) catch return error.OutOfMemory,
            .frame = frame,
        } }) catch return error.OutOfMemory;
    }

    /// Parse ROWS/RANGE/GROUPS BETWEEN ... AND ... frame specification.
    fn parseFrameSpec(self: *Parser) Error!*const ast.WindowFrameSpec {
        // Parse frame mode
        const mode: ast.WindowFrameMode = if (self.match(.kw_rows))
            .rows
        else if (self.match(.kw_range))
            .range
        else if (self.match(.kw_groups))
            .groups
        else {
            try self.addError(self.peek(), "expected ROWS, RANGE, or GROUPS");
            return error.ParseFailed;
        };

        if (self.match(.kw_between)) {
            // BETWEEN start AND end
            const start = try self.parseFrameBound();
            _ = try self.expect(.kw_and);
            const end = try self.parseFrameBound();
            return self.arena.create(ast.WindowFrameSpec, .{
                .mode = mode,
                .start = start,
                .end = end,
            }) catch return error.OutOfMemory;
        }

        // Single bound (start only, end defaults to CURRENT ROW)
        const start = try self.parseFrameBound();
        return self.arena.create(ast.WindowFrameSpec, .{
            .mode = mode,
            .start = start,
            .end = .current_row,
        }) catch return error.OutOfMemory;
    }

    /// Parse a single frame bound: UNBOUNDED PRECEDING/FOLLOWING, CURRENT ROW, or <expr> PRECEDING/FOLLOWING.
    fn parseFrameBound(self: *Parser) Error!ast.WindowFrameBound {
        if (self.match(.kw_unbounded)) {
            if (self.match(.kw_preceding)) return .unbounded_preceding;
            if (self.match(.kw_following)) return .unbounded_following;
            try self.addError(self.peek(), "expected PRECEDING or FOLLOWING after UNBOUNDED");
            return error.ParseFailed;
        }

        if (self.match(.kw_current)) {
            _ = try self.expect(.kw_row);
            return .current_row;
        }

        // <expr> PRECEDING or <expr> FOLLOWING
        const expr = try self.parseExpr(0);
        if (self.match(.kw_preceding)) return .{ .expr_preceding = expr };
        if (self.match(.kw_following)) return .{ .expr_following = expr };

        try self.addError(self.peek(), "expected PRECEDING or FOLLOWING");
        return error.ParseFailed;
    }

    fn parseCaseExpr(self: *Parser) Error!*const ast.Expr {
        const a = self.alloc();
        _ = try self.expect(.kw_case);

        var operand: ?*const ast.Expr = null;
        if (!self.check(.kw_when)) {
            operand = try self.parseExpr(0);
        }

        var when_clauses = std.ArrayListUnmanaged(ast.WhenClause){};
        while (self.match(.kw_when)) {
            const condition = try self.parseExpr(0);
            _ = try self.expect(.kw_then);
            const result = try self.parseExpr(0);
            when_clauses.append(a, .{ .condition = condition, .result = result }) catch return error.OutOfMemory;
        }

        var else_expr: ?*const ast.Expr = null;
        if (self.match(.kw_else)) {
            else_expr = try self.parseExpr(0);
        }

        _ = try self.expect(.kw_end);

        return self.arena.create(ast.Expr, .{ .case_expr = .{
            .operand = operand,
            .when_clauses = when_clauses.toOwnedSlice(a) catch return error.OutOfMemory,
            .else_expr = else_expr,
        } }) catch return error.OutOfMemory;
    }

    fn parseArrayConstructor(self: *Parser) Error!*const ast.Expr {
        _ = try self.expect(.kw_array);
        _ = try self.expect(.left_bracket);

        const a = self.alloc();
        var elements = std.ArrayListUnmanaged(*const ast.Expr){};
        defer elements.deinit(a);

        if (!self.check(.right_bracket)) {
            elements.append(a, try self.parseExpr(0)) catch return error.OutOfMemory;
            while (self.match(.comma)) {
                elements.append(a, try self.parseExpr(0)) catch return error.OutOfMemory;
            }
        }
        _ = try self.expect(.right_bracket);

        const elems = elements.toOwnedSlice(a) catch return error.OutOfMemory;
        return self.arena.create(ast.Expr, .{ .array_constructor = elems }) catch return error.OutOfMemory;
    }

    fn parseCastExpr(self: *Parser) Error!*const ast.Expr {
        _ = try self.expect(.kw_cast);
        _ = try self.expect(.left_paren);
        const expr = try self.parseExpr(0);
        _ = try self.expect(.kw_as);
        const target_type = self.parseDataType() orelse {
            try self.addError(self.peek(), "expected type name in CAST");
            return error.ParseFailed;
        };
        _ = try self.expect(.right_paren);

        return self.arena.create(ast.Expr, .{ .cast = .{
            .expr = expr,
            .target_type = target_type,
        } }) catch return error.OutOfMemory;
    }

    fn parseIsExpr(self: *Parser, left: *const ast.Expr) Error!*const ast.Expr {
        _ = try self.expect(.kw_is);
        const negated = self.match(.kw_not);
        _ = try self.expect(.kw_null);
        return self.arena.create(ast.Expr, .{ .is_null = .{
            .expr = left,
            .negated = negated,
        } }) catch return error.OutOfMemory;
    }

    fn peekNot(self: *const Parser) bool {
        if (self.pos + 1 >= self.tokens.len) return false;
        const next = self.tokens[self.pos + 1].type;
        return next == .kw_between or next == .kw_in or next == .kw_like;
    }

    fn parseNotInfix(self: *Parser, left: *const ast.Expr) Error!*const ast.Expr {
        _ = try self.expect(.kw_not);
        if (self.check(.kw_between)) return self.parseBetweenExpr(left, true);
        if (self.check(.kw_in)) return self.parseInExpr(left, true);
        if (self.check(.kw_like)) return self.parseLikeExpr(left, true);
        try self.addError(self.peek(), "expected BETWEEN, IN, or LIKE after NOT");
        return error.ParseFailed;
    }

    fn parseBetweenExpr(self: *Parser, expr: *const ast.Expr, negated: bool) Error!*const ast.Expr {
        _ = try self.expect(.kw_between);
        const low = try self.parseExpr(6);
        _ = try self.expect(.kw_and);
        const high = try self.parseExpr(6);
        return self.arena.create(ast.Expr, .{ .between = .{
            .expr = expr,
            .low = low,
            .high = high,
            .negated = negated,
        } }) catch return error.OutOfMemory;
    }

    fn parseInExpr(self: *Parser, expr: *const ast.Expr, negated: bool) Error!*const ast.Expr {
        const a = self.alloc();
        _ = try self.expect(.kw_in);
        _ = try self.expect(.left_paren);

        var list = std.ArrayListUnmanaged(*const ast.Expr){};
        if (!self.check(.right_paren)) {
            while (true) {
                list.append(a, try self.parseExpr(0)) catch return error.OutOfMemory;
                if (!self.match(.comma)) break;
            }
        }
        _ = try self.expect(.right_paren);

        return self.arena.create(ast.Expr, .{ .in_list = .{
            .expr = expr,
            .list = list.toOwnedSlice(a) catch return error.OutOfMemory,
            .negated = negated,
        } }) catch return error.OutOfMemory;
    }

    fn parseLikeExpr(self: *Parser, expr: *const ast.Expr, negated: bool) Error!*const ast.Expr {
        _ = try self.expect(.kw_like);
        const pattern = try self.parseExpr(0);
        return self.arena.create(ast.Expr, .{ .like = .{
            .expr = expr,
            .pattern = pattern,
            .negated = negated,
        } }) catch return error.OutOfMemory;
    }

    // ── Operator precedence ───────────────────────────────────────

    fn currentPrecedence(self: *const Parser) u8 {
        const t = self.peek().type;
        return switch (t) {
            .kw_or => 1,
            .kw_and => 2,
            .kw_not => if (self.peekNot()) 3 else 0,
            .kw_is => 4,
            .kw_between, .kw_in, .kw_like, .kw_glob => 4,
            .equals, .not_equals, .less_than, .greater_than, .less_than_or_equal, .greater_than_or_equal => 5,
            .bitwise_and, .bitwise_or => 6,
            .left_shift, .right_shift => 7,
            .plus, .minus => 8,
            .star, .slash, .percent => 9,
            .concat => 10,
            // JSON operators have high precedence (11) - similar to member access
            .json_extract, .json_extract_text => 11,
            .json_contains, .json_contained_by => 11,
            // NOTE: .json_key_exists (?) is now used as bind parameter, not binary op
            .json_any_key_exists, .json_all_keys_exist => 11,
            .json_path_extract, .json_path_extract_text, .json_delete_path => 11,
            // Full-text search operator @@ has comparison precedence (7)
            .ts_match => 7,
            else => 0,
        };
    }

    fn currentBinaryOp(self: *const Parser) ?ast.BinaryOp {
        return switch (self.peek().type) {
            .plus => .add,
            .minus => .subtract,
            .star => .multiply,
            .slash => .divide,
            .percent => .modulo,
            .equals => .equal,
            .not_equals => .not_equal,
            .less_than => .less_than,
            .greater_than => .greater_than,
            .less_than_or_equal => .less_than_or_equal,
            .greater_than_or_equal => .greater_than_or_equal,
            .kw_and => .@"and",
            .kw_or => .@"or",
            .concat => .concat,
            .bitwise_and => .bitwise_and,
            .bitwise_or => .bitwise_or,
            .left_shift => .left_shift,
            .right_shift => .right_shift,
            // JSON operators
            .json_extract => .json_extract,
            .json_extract_text => .json_extract_text,
            .json_contains => .json_contains,
            .json_contained_by => .json_contained_by,
            // NOTE: .json_key_exists (?) is now used as bind parameter, not binary op
            .json_any_key_exists => .json_any_key_exists,
            .json_all_keys_exist => .json_all_keys_exist,
            .json_path_extract => .json_path_extract,
            .json_path_extract_text => .json_path_extract_text,
            .json_delete_path => .json_delete_path,
            // Full-text search
            .ts_match => .ts_match,
            else => null,
        };
    }
};

// ══════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════

const TestParseResult = struct {
    stmt: ast.Stmt,
    arena: ast.AstArena,
    parser: Parser,

    pub fn deinit(self: *TestParseResult) void {
        self.parser.deinit();
        self.arena.deinit();
    }
};

fn testParseWithArena(sql: []const u8) !TestParseResult {
    var ast_arena = ast.AstArena.init(std.testing.allocator);
    errdefer ast_arena.deinit();

    var p = try Parser.init(std.testing.allocator, sql, &ast_arena);
    errdefer p.deinit();

    const stmt = try p.parseStatement() orelse return error.ParseFailed;
    return .{ .stmt = stmt, .arena = ast_arena, .parser = p };
}

// ── SELECT tests ──────────────────────────────────────────────

test "parse simple SELECT *" {
    var r = try testParseWithArena("SELECT * FROM users");
    defer r.deinit();
    const sel = r.stmt.select;
    try std.testing.expectEqual(@as(usize, 1), sel.columns.len);
    try std.testing.expect(sel.columns[0] == .all_columns);
    try std.testing.expectEqualStrings("users", sel.from.?.table_name.name);
}

test "parse SELECT with columns" {
    var r = try testParseWithArena("SELECT id, name, email FROM users");
    defer r.deinit();
    const sel = r.stmt.select;
    try std.testing.expectEqual(@as(usize, 3), sel.columns.len);
    const first = sel.columns[0].expr.value.column_ref;
    try std.testing.expectEqualStrings("id", first.name);
}

test "parse SELECT with alias" {
    var r = try testParseWithArena("SELECT id AS user_id FROM users");
    defer r.deinit();
    try std.testing.expectEqualStrings("user_id", r.stmt.select.columns[0].expr.alias.?);
}

test "parse SELECT DISTINCT" {
    var r = try testParseWithArena("SELECT DISTINCT name FROM users");
    defer r.deinit();
    try std.testing.expect(r.stmt.select.distinct);
    try std.testing.expectEqual(@as(usize, 0), r.stmt.select.distinct_on.len);
}

test "parse SELECT DISTINCT ON single column" {
    var r = try testParseWithArena("SELECT DISTINCT ON (category) category, name, price FROM products");
    defer r.deinit();
    try std.testing.expect(r.stmt.select.distinct);
    try std.testing.expectEqual(@as(usize, 1), r.stmt.select.distinct_on.len);
    try std.testing.expectEqualStrings("category", r.stmt.select.distinct_on[0].column_ref.name);
    try std.testing.expectEqual(@as(usize, 3), r.stmt.select.columns.len);
}

test "parse SELECT DISTINCT ON multiple columns" {
    var r = try testParseWithArena("SELECT DISTINCT ON (dept, role) * FROM employees ORDER BY dept, role, salary DESC");
    defer r.deinit();
    try std.testing.expect(r.stmt.select.distinct);
    try std.testing.expectEqual(@as(usize, 2), r.stmt.select.distinct_on.len);
    try std.testing.expectEqualStrings("dept", r.stmt.select.distinct_on[0].column_ref.name);
    try std.testing.expectEqualStrings("role", r.stmt.select.distinct_on[1].column_ref.name);
    try std.testing.expectEqual(@as(usize, 3), r.stmt.select.order_by.len);
}

test "parse SELECT with WHERE" {
    var r = try testParseWithArena("SELECT * FROM users WHERE id = 1");
    defer r.deinit();
    const w = r.stmt.select.where.?.binary_op;
    try std.testing.expectEqual(ast.BinaryOp.equal, w.op);
}

test "parse SELECT with ORDER BY" {
    var r = try testParseWithArena("SELECT * FROM users ORDER BY name ASC, id DESC");
    defer r.deinit();
    const sel = r.stmt.select;
    try std.testing.expectEqual(@as(usize, 2), sel.order_by.len);
    try std.testing.expectEqual(ast.OrderDirection.asc, sel.order_by[0].direction);
    try std.testing.expectEqual(ast.OrderDirection.desc, sel.order_by[1].direction);
}

test "parse SELECT with LIMIT and OFFSET" {
    var r = try testParseWithArena("SELECT * FROM users LIMIT 10 OFFSET 20");
    defer r.deinit();
    try std.testing.expectEqual(@as(i64, 10), r.stmt.select.limit.?.integer_literal);
    try std.testing.expectEqual(@as(i64, 20), r.stmt.select.offset.?.integer_literal);
}

test "parse SELECT with GROUP BY and HAVING" {
    var r = try testParseWithArena("SELECT department, COUNT(*) FROM employees GROUP BY department HAVING COUNT(*) > 5");
    defer r.deinit();
    try std.testing.expectEqual(@as(usize, 1), r.stmt.select.group_by.len);
    try std.testing.expect(r.stmt.select.having != null);
}

test "parse SELECT with JOIN" {
    var r = try testParseWithArena("SELECT * FROM users INNER JOIN orders ON users.id = orders.user_id");
    defer r.deinit();
    try std.testing.expectEqual(@as(usize, 1), r.stmt.select.joins.len);
    try std.testing.expectEqual(ast.JoinType.inner, r.stmt.select.joins[0].join_type);
    try std.testing.expect(r.stmt.select.joins[0].on_condition != null);
}

test "parse SELECT with LEFT JOIN" {
    var r = try testParseWithArena("SELECT * FROM a LEFT JOIN b ON a.id = b.a_id");
    defer r.deinit();
    try std.testing.expectEqual(ast.JoinType.left, r.stmt.select.joins[0].join_type);
}

test "parse SELECT with table alias" {
    var r = try testParseWithArena("SELECT u.name FROM users u");
    defer r.deinit();
    try std.testing.expectEqualStrings("u", r.stmt.select.from.?.table_name.alias.?);
}

test "parse SELECT with subquery in FROM" {
    var r = try testParseWithArena("SELECT * FROM (SELECT id FROM t) sub");
    defer r.deinit();
    try std.testing.expect(r.stmt.select.from.?.* == .subquery);
    try std.testing.expectEqualStrings("sub", r.stmt.select.from.?.subquery.alias);
}

test "parse SELECT table.*" {
    var r = try testParseWithArena("SELECT t.* FROM t");
    defer r.deinit();
    try std.testing.expect(r.stmt.select.columns[0] == .table_all_columns);
    try std.testing.expectEqualStrings("t", r.stmt.select.columns[0].table_all_columns);
}

// ── INSERT tests ──────────────────────────────────────────────

test "parse INSERT with columns" {
    var r = try testParseWithArena("INSERT INTO users (id, name) VALUES (1, 'Alice')");
    defer r.deinit();
    const ins = r.stmt.insert;
    try std.testing.expectEqualStrings("users", ins.table);
    try std.testing.expectEqual(@as(usize, 2), ins.columns.?.len);
    try std.testing.expectEqual(@as(usize, 1), ins.values.len);
    try std.testing.expectEqual(@as(usize, 2), ins.values[0].len);
}

test "parse INSERT without columns" {
    var r = try testParseWithArena("INSERT INTO users VALUES (1, 'Bob', 'bob@example.com')");
    defer r.deinit();
    try std.testing.expect(r.stmt.insert.columns == null);
    try std.testing.expectEqual(@as(usize, 3), r.stmt.insert.values[0].len);
}

test "parse INSERT multiple rows" {
    var r = try testParseWithArena("INSERT INTO t (a) VALUES (1), (2), (3)");
    defer r.deinit();
    try std.testing.expectEqual(@as(usize, 3), r.stmt.insert.values.len);
}

// ── UPDATE tests ──────────────────────────────────────────────

test "parse UPDATE" {
    var r = try testParseWithArena("UPDATE users SET name = 'Charlie' WHERE id = 1");
    defer r.deinit();
    const upd = r.stmt.update;
    try std.testing.expectEqualStrings("users", upd.table);
    try std.testing.expectEqual(@as(usize, 1), upd.assignments.len);
    try std.testing.expect(upd.where != null);
}

test "parse UPDATE multiple assignments" {
    var r = try testParseWithArena("UPDATE t SET a = 1, b = 2, c = 3");
    defer r.deinit();
    try std.testing.expectEqual(@as(usize, 3), r.stmt.update.assignments.len);
}

// ── DELETE tests ──────────────────────────────────────────────

test "parse DELETE" {
    var r = try testParseWithArena("DELETE FROM users WHERE id = 1");
    defer r.deinit();
    try std.testing.expectEqualStrings("users", r.stmt.delete.table);
    try std.testing.expect(r.stmt.delete.where != null);
}

test "parse DELETE without WHERE" {
    var r = try testParseWithArena("DELETE FROM users");
    defer r.deinit();
    try std.testing.expect(r.stmt.delete.where == null);
}

// ── CREATE TABLE tests ────────────────────────────────────────

test "parse CREATE TABLE" {
    var r = try testParseWithArena(
        \\CREATE TABLE users (
        \\  id INTEGER PRIMARY KEY AUTOINCREMENT,
        \\  name TEXT NOT NULL,
        \\  email TEXT UNIQUE
        \\)
    );
    defer r.deinit();
    const ct = r.stmt.create_table;
    try std.testing.expectEqualStrings("users", ct.name);
    try std.testing.expectEqual(@as(usize, 3), ct.columns.len);
    try std.testing.expect(!ct.if_not_exists);
    try std.testing.expectEqualStrings("id", ct.columns[0].name);
    try std.testing.expectEqual(ast.DataType.type_integer, ct.columns[0].data_type.?);
    try std.testing.expect(ct.columns[0].constraints[0].primary_key.autoincrement);
    try std.testing.expect(ct.columns[1].constraints[0] == .not_null);
}

test "parse CREATE TABLE IF NOT EXISTS" {
    var r = try testParseWithArena("CREATE TABLE IF NOT EXISTS t (id INTEGER)");
    defer r.deinit();
    try std.testing.expect(r.stmt.create_table.if_not_exists);
}

test "parse CREATE TABLE with table constraints" {
    var r = try testParseWithArena(
        \\CREATE TABLE t (
        \\  a INTEGER,
        \\  b INTEGER,
        \\  PRIMARY KEY (a, b)
        \\)
    );
    defer r.deinit();
    const ct = r.stmt.create_table;
    try std.testing.expectEqual(@as(usize, 2), ct.columns.len);
    try std.testing.expectEqual(@as(usize, 1), ct.table_constraints.len);
    try std.testing.expectEqual(@as(usize, 2), ct.table_constraints[0].primary_key.columns.len);
}

test "parse CREATE TABLE WITHOUT ROWID" {
    var r = try testParseWithArena("CREATE TABLE t (id INTEGER PRIMARY KEY) WITHOUT ROWID");
    defer r.deinit();
    try std.testing.expect(r.stmt.create_table.without_rowid);
}

test "parse CREATE TABLE STRICT" {
    var r = try testParseWithArena("CREATE TABLE t (id INTEGER PRIMARY KEY) STRICT");
    defer r.deinit();
    try std.testing.expect(r.stmt.create_table.strict);
}

test "parse CREATE TABLE with DEFAULT" {
    var r = try testParseWithArena("CREATE TABLE t (x INTEGER DEFAULT 0)");
    defer r.deinit();
    try std.testing.expect(r.stmt.create_table.columns[0].constraints[0] == .default);
}

test "parse CREATE TABLE with FOREIGN KEY" {
    var r = try testParseWithArena(
        \\CREATE TABLE orders (
        \\  id INTEGER PRIMARY KEY,
        \\  user_id INTEGER REFERENCES users(id) ON DELETE CASCADE
        \\)
    );
    defer r.deinit();
    const ct = r.stmt.create_table;
    const fk = ct.columns[1].constraints[0].foreign_key;
    try std.testing.expectEqualStrings("users", fk.table);
    try std.testing.expectEqualStrings("id", fk.column.?);
    try std.testing.expectEqual(ast.ForeignKeyAction.cascade, fk.on_delete.?);
}

// ── DROP tests ────────────────────────────────────────────────

test "parse DROP TABLE" {
    var r = try testParseWithArena("DROP TABLE users");
    defer r.deinit();
    try std.testing.expectEqualStrings("users", r.stmt.drop_table.name);
    try std.testing.expect(!r.stmt.drop_table.if_exists);
}

test "parse DROP TABLE IF EXISTS" {
    var r = try testParseWithArena("DROP TABLE IF EXISTS users");
    defer r.deinit();
    try std.testing.expect(r.stmt.drop_table.if_exists);
}

test "parse DROP INDEX" {
    var r = try testParseWithArena("DROP INDEX idx_name");
    defer r.deinit();
    try std.testing.expectEqualStrings("idx_name", r.stmt.drop_index.name);
}

// ── CREATE INDEX tests ────────────────────────────────────────

test "parse CREATE INDEX" {
    var r = try testParseWithArena("CREATE INDEX idx_email ON users (email)");
    defer r.deinit();
    const ci = r.stmt.create_index;
    try std.testing.expectEqualStrings("idx_email", ci.name);
    try std.testing.expectEqualStrings("users", ci.table);
    try std.testing.expect(!ci.unique);
}

test "parse CREATE UNIQUE INDEX" {
    var r = try testParseWithArena("CREATE UNIQUE INDEX idx_u ON t (a, b DESC)");
    defer r.deinit();
    const ci = r.stmt.create_index;
    try std.testing.expect(ci.unique);
    try std.testing.expectEqual(@as(usize, 2), ci.columns.len);
    try std.testing.expectEqual(ast.OrderDirection.desc, ci.columns[1].direction);
}

test "parse CREATE INDEX with INCLUDE clause" {
    var r = try testParseWithArena("CREATE INDEX idx_name ON users (name) INCLUDE (email, created_at)");
    defer r.deinit();
    const ci = r.stmt.create_index;
    try std.testing.expectEqualStrings("idx_name", ci.name);
    try std.testing.expectEqualStrings("users", ci.table);
    try std.testing.expectEqual(@as(usize, 1), ci.columns.len);
    try std.testing.expectEqual(@as(usize, 2), ci.included_columns.len);
    try std.testing.expectEqualStrings("email", ci.included_columns[0]);
    try std.testing.expectEqualStrings("created_at", ci.included_columns[1]);
}

test "parse CREATE UNIQUE INDEX with INCLUDE clause" {
    var r = try testParseWithArena("CREATE UNIQUE INDEX idx_user_email ON users (email) INCLUDE (name)");
    defer r.deinit();
    const ci = r.stmt.create_index;
    try std.testing.expect(ci.unique);
    try std.testing.expectEqual(@as(usize, 1), ci.columns.len);
    try std.testing.expectEqual(@as(usize, 1), ci.included_columns.len);
    try std.testing.expectEqualStrings("name", ci.included_columns[0]);
}

test "parse CREATE INDEX INCLUDE with multiple columns" {
    var r = try testParseWithArena("CREATE INDEX idx_composite ON orders (user_id, created_at DESC) INCLUDE (total, status, notes)");
    defer r.deinit();
    const ci = r.stmt.create_index;
    try std.testing.expectEqual(@as(usize, 2), ci.columns.len);
    try std.testing.expectEqual(@as(usize, 3), ci.included_columns.len);
    try std.testing.expectEqualStrings("total", ci.included_columns[0]);
    try std.testing.expectEqualStrings("status", ci.included_columns[1]);
    try std.testing.expectEqualStrings("notes", ci.included_columns[2]);
}

test "parse CREATE INDEX with empty INCLUDE should fail" {
    const r = testParseWithArena("CREATE INDEX idx_bad ON users (name) INCLUDE ()");
    try std.testing.expectError(error.ParseFailed, r);
}

test "parse CREATE INDEX with duplicate column in INCLUDE should fail" {
    const r = testParseWithArena("CREATE INDEX idx_dup ON users (name) INCLUDE (name, email)");
    try std.testing.expectError(error.ParseFailed, r);
}

test "parse CREATE INDEX with duplicate across index and INCLUDE should fail" {
    const r = testParseWithArena("CREATE INDEX idx_dup ON users (name, email) INCLUDE (email)");
    try std.testing.expectError(error.ParseFailed, r);
}

// ── CREATE INDEX CONCURRENTLY tests ────────────────────────────

test "parse CREATE INDEX CONCURRENTLY" {
    var r = try testParseWithArena("CREATE INDEX CONCURRENTLY idx_email ON users (email)");
    defer r.deinit();
    const ci = r.stmt.create_index;
    try std.testing.expectEqualStrings("idx_email", ci.name);
    try std.testing.expectEqualStrings("users", ci.table);
    try std.testing.expect(!ci.unique);
    try std.testing.expect(ci.concurrently);
}

test "parse CREATE UNIQUE INDEX CONCURRENTLY" {
    var r = try testParseWithArena("CREATE UNIQUE INDEX CONCURRENTLY idx_u ON t (a)");
    defer r.deinit();
    const ci = r.stmt.create_index;
    try std.testing.expect(ci.unique);
    try std.testing.expect(ci.concurrently);
}

test "parse CREATE INDEX CONCURRENTLY with multiple columns" {
    var r = try testParseWithArena("CREATE INDEX CONCURRENTLY idx_composite ON orders (user_id, created_at DESC)");
    defer r.deinit();
    const ci = r.stmt.create_index;
    try std.testing.expect(ci.concurrently);
    try std.testing.expectEqual(@as(usize, 2), ci.columns.len);
}

test "parse CREATE INDEX CONCURRENTLY IF NOT EXISTS" {
    var r = try testParseWithArena("CREATE INDEX CONCURRENTLY IF NOT EXISTS idx ON t (col)");
    defer r.deinit();
    const ci = r.stmt.create_index;
    try std.testing.expect(ci.concurrently);
    try std.testing.expect(ci.if_not_exists);
}

test "parse CREATE INDEX CONCURRENTLY with USING clause" {
    var r = try testParseWithArena("CREATE INDEX CONCURRENTLY idx_hash ON users (email) USING hash");
    defer r.deinit();
    const ci = r.stmt.create_index;
    try std.testing.expect(ci.concurrently);
    try std.testing.expectEqualStrings("hash", ci.index_type.?);
}

test "parse CREATE INDEX CONCURRENTLY with INCLUDE clause" {
    var r = try testParseWithArena("CREATE INDEX CONCURRENTLY idx_name ON users (name) INCLUDE (email)");
    defer r.deinit();
    const ci = r.stmt.create_index;
    try std.testing.expect(ci.concurrently);
    try std.testing.expectEqual(@as(usize, 1), ci.columns.len);
    try std.testing.expectEqual(@as(usize, 1), ci.included_columns.len);
}

test "parse CREATE UNIQUE INDEX CONCURRENTLY with INCLUDE and USING" {
    var r = try testParseWithArena("CREATE UNIQUE INDEX CONCURRENTLY idx_u ON users (email) INCLUDE (name) USING btree");
    defer r.deinit();
    const ci = r.stmt.create_index;
    try std.testing.expect(ci.unique);
    try std.testing.expect(ci.concurrently);
    try std.testing.expectEqualStrings("btree", ci.index_type.?);
    try std.testing.expectEqual(@as(usize, 1), ci.included_columns.len);
}

test "parse CREATE INDEX without CONCURRENTLY flag is false" {
    var r = try testParseWithArena("CREATE INDEX idx_email ON users (email)");
    defer r.deinit();
    const ci = r.stmt.create_index;
    try std.testing.expect(!ci.concurrently);
}

test "parse CREATE INDEX CONCURRENTLY with btree type" {
    var r = try testParseWithArena("CREATE INDEX CONCURRENTLY idx_btree ON t (col) USING btree");
    defer r.deinit();
    const ci = r.stmt.create_index;
    try std.testing.expect(ci.concurrently);
    try std.testing.expectEqualStrings("btree", ci.index_type.?);
}

// ── Transaction tests ─────────────────────────────────────────

test "parse BEGIN" {
    var r = try testParseWithArena("BEGIN");
    defer r.deinit();
    try std.testing.expectEqual(ast.TransactionMode.deferred, r.stmt.transaction.begin.mode);
}

test "parse BEGIN IMMEDIATE" {
    var r = try testParseWithArena("BEGIN IMMEDIATE TRANSACTION");
    defer r.deinit();
    try std.testing.expectEqual(ast.TransactionMode.immediate, r.stmt.transaction.begin.mode);
}

test "parse COMMIT" {
    var r = try testParseWithArena("COMMIT");
    defer r.deinit();
    try std.testing.expect(r.stmt.transaction == .commit);
}

test "parse ROLLBACK" {
    var r = try testParseWithArena("ROLLBACK");
    defer r.deinit();
    try std.testing.expect(r.stmt.transaction.rollback.savepoint == null);
}

test "parse ROLLBACK TO SAVEPOINT" {
    var r = try testParseWithArena("ROLLBACK TO SAVEPOINT sp1");
    defer r.deinit();
    try std.testing.expectEqualStrings("sp1", r.stmt.transaction.rollback.savepoint.?);
}

test "parse SAVEPOINT" {
    var r = try testParseWithArena("SAVEPOINT sp1");
    defer r.deinit();
    try std.testing.expectEqualStrings("sp1", r.stmt.transaction.savepoint);
}

test "parse RELEASE" {
    var r = try testParseWithArena("RELEASE SAVEPOINT sp1");
    defer r.deinit();
    try std.testing.expectEqualStrings("sp1", r.stmt.transaction.release);
}

// ── Expression tests ──────────────────────────────────────────

test "parse arithmetic precedence" {
    var r = try testParseWithArena("SELECT 1 + 2 * 3");
    defer r.deinit();
    const expr = r.stmt.select.columns[0].expr.value;
    const add = expr.binary_op;
    try std.testing.expectEqual(ast.BinaryOp.add, add.op);
    try std.testing.expectEqual(@as(i64, 1), add.left.integer_literal);
    try std.testing.expectEqual(ast.BinaryOp.multiply, add.right.binary_op.op);
}

test "parse comparison with AND/OR precedence" {
    var r = try testParseWithArena("SELECT * FROM t WHERE a = 1 OR b = 2 AND c = 3");
    defer r.deinit();
    // AND > OR, so: a = 1 OR (b = 2 AND c = 3)
    try std.testing.expectEqual(ast.BinaryOp.@"or", r.stmt.select.where.?.binary_op.op);
}

test "parse BETWEEN" {
    var r = try testParseWithArena("SELECT * FROM t WHERE x BETWEEN 1 AND 10");
    defer r.deinit();
    try std.testing.expect(r.stmt.select.where.?.* == .between);
    try std.testing.expect(!r.stmt.select.where.?.between.negated);
}

test "parse NOT BETWEEN" {
    var r = try testParseWithArena("SELECT * FROM t WHERE x NOT BETWEEN 1 AND 10");
    defer r.deinit();
    try std.testing.expect(r.stmt.select.where.?.between.negated);
}

test "parse IN list" {
    var r = try testParseWithArena("SELECT * FROM t WHERE id IN (1, 2, 3)");
    defer r.deinit();
    const w = r.stmt.select.where.?;
    try std.testing.expect(w.* == .in_list);
    try std.testing.expectEqual(@as(usize, 3), w.in_list.list.len);
}

test "parse IS NULL" {
    var r = try testParseWithArena("SELECT * FROM t WHERE x IS NULL");
    defer r.deinit();
    try std.testing.expect(r.stmt.select.where.?.* == .is_null);
    try std.testing.expect(!r.stmt.select.where.?.is_null.negated);
}

test "parse IS NOT NULL" {
    var r = try testParseWithArena("SELECT * FROM t WHERE x IS NOT NULL");
    defer r.deinit();
    try std.testing.expect(r.stmt.select.where.?.is_null.negated);
}

test "parse LIKE" {
    var r = try testParseWithArena("SELECT * FROM t WHERE name LIKE '%alice%'");
    defer r.deinit();
    try std.testing.expect(r.stmt.select.where.?.* == .like);
    try std.testing.expectEqualStrings("%alice%", r.stmt.select.where.?.like.pattern.string_literal);
}

test "parse NOT LIKE" {
    var r = try testParseWithArena("SELECT * FROM t WHERE name NOT LIKE 'test%'");
    defer r.deinit();
    try std.testing.expect(r.stmt.select.where.?.like.negated);
}

test "parse CASE expression" {
    var r = try testParseWithArena("SELECT CASE WHEN x > 0 THEN 'positive' ELSE 'negative' END FROM t");
    defer r.deinit();
    const expr = r.stmt.select.columns[0].expr.value;
    try std.testing.expect(expr.* == .case_expr);
    try std.testing.expectEqual(@as(usize, 1), expr.case_expr.when_clauses.len);
    try std.testing.expect(expr.case_expr.else_expr != null);
}

test "parse CAST" {
    var r = try testParseWithArena("SELECT CAST(x AS INTEGER) FROM t");
    defer r.deinit();
    const expr = r.stmt.select.columns[0].expr.value;
    try std.testing.expect(expr.* == .cast);
    try std.testing.expectEqual(ast.DataType.type_integer, expr.cast.target_type);
}

test "parse function call COUNT(*)" {
    var r = try testParseWithArena("SELECT COUNT(*) FROM t");
    defer r.deinit();
    const expr = r.stmt.select.columns[0].expr.value;
    try std.testing.expect(expr.* == .function_call);
    try std.testing.expectEqualStrings("COUNT", expr.function_call.name);
}

test "parse COUNT DISTINCT" {
    var r = try testParseWithArena("SELECT COUNT(DISTINCT name) FROM t");
    defer r.deinit();
    try std.testing.expect(r.stmt.select.columns[0].expr.value.function_call.distinct);
}

test "parse negative number" {
    var r = try testParseWithArena("SELECT -42");
    defer r.deinit();
    const expr = r.stmt.select.columns[0].expr.value;
    try std.testing.expectEqual(ast.UnaryOp.negate, expr.unary_op.op);
    try std.testing.expectEqual(@as(i64, 42), expr.unary_op.operand.integer_literal);
}

test "parse string literal" {
    var r = try testParseWithArena("SELECT 'hello world'");
    defer r.deinit();
    try std.testing.expectEqualStrings("hello world", r.stmt.select.columns[0].expr.value.string_literal);
}

test "parse boolean literals" {
    var r = try testParseWithArena("SELECT TRUE, FALSE");
    defer r.deinit();
    try std.testing.expect(r.stmt.select.columns[0].expr.value.boolean_literal);
    try std.testing.expect(!r.stmt.select.columns[1].expr.value.boolean_literal);
}

test "parse NULL literal" {
    var r = try testParseWithArena("SELECT NULL");
    defer r.deinit();
    try std.testing.expect(r.stmt.select.columns[0].expr.value.* == .null_literal);
}

test "parse parenthesized expression" {
    var r = try testParseWithArena("SELECT (1 + 2) * 3");
    defer r.deinit();
    const mul = r.stmt.select.columns[0].expr.value.binary_op;
    try std.testing.expectEqual(ast.BinaryOp.multiply, mul.op);
    try std.testing.expect(mul.left.* == .paren);
}

test "parse qualified column reference" {
    var r = try testParseWithArena("SELECT t.id FROM t");
    defer r.deinit();
    const col = r.stmt.select.columns[0].expr.value.column_ref;
    try std.testing.expectEqualStrings("id", col.name);
    try std.testing.expectEqualStrings("t", col.prefix.?);
}

test "parse EXPLAIN" {
    var r = try testParseWithArena("EXPLAIN SELECT * FROM t");
    defer r.deinit();
    try std.testing.expect(r.stmt == .explain);
    try std.testing.expect(r.stmt.explain.stmt.* == .select);
    try std.testing.expect(!r.stmt.explain.analyze);
}

test "parse EXPLAIN ANALYZE" {
    var r = try testParseWithArena("EXPLAIN ANALYZE SELECT * FROM t");
    defer r.deinit();
    try std.testing.expect(r.stmt == .explain);
    try std.testing.expect(r.stmt.explain.stmt.* == .select);
    try std.testing.expect(r.stmt.explain.analyze);
}

test "parse EXPLAIN ANALYZE INSERT" {
    var r = try testParseWithArena("EXPLAIN ANALYZE INSERT INTO users (name) VALUES ('Alice')");
    defer r.deinit();
    try std.testing.expect(r.stmt == .explain);
    try std.testing.expect(r.stmt.explain.stmt.* == .insert);
    try std.testing.expect(r.stmt.explain.analyze);
}

test "parse multiple statements" {
    var ast_arena = ast.AstArena.init(std.testing.allocator);
    defer ast_arena.deinit();
    var p = try Parser.init(std.testing.allocator, "SELECT 1; SELECT 2;", &ast_arena);
    defer p.deinit();

    try std.testing.expect((try p.parseStatement()) != null);
    try std.testing.expect((try p.parseStatement()) != null);
    try std.testing.expect((try p.parseStatement()) == null);
}

test "parse error on invalid input" {
    var ast_arena = ast.AstArena.init(std.testing.allocator);
    defer ast_arena.deinit();
    var p = try Parser.init(std.testing.allocator, "FOOBAR", &ast_arena);
    defer p.deinit();

    const result = p.parseStatement();
    try std.testing.expect(result == error.ParseFailed);
    try std.testing.expect(p.errors.items.len > 0);
}

test "parse NOT IN" {
    var r = try testParseWithArena("SELECT * FROM t WHERE x NOT IN (1, 2)");
    defer r.deinit();
    try std.testing.expect(r.stmt.select.where.?.in_list.negated);
}

test "parse multiple JOINs" {
    var r = try testParseWithArena("SELECT * FROM a JOIN b ON a.id = b.a_id LEFT JOIN c ON b.id = c.b_id");
    defer r.deinit();
    try std.testing.expectEqual(@as(usize, 2), r.stmt.select.joins.len);
    try std.testing.expectEqual(ast.JoinType.inner, r.stmt.select.joins[0].join_type);
    try std.testing.expectEqual(ast.JoinType.left, r.stmt.select.joins[1].join_type);
}

test "parse nested function calls" {
    var r = try testParseWithArena("SELECT MAX(ABS(x)) FROM t");
    defer r.deinit();
    const outer = r.stmt.select.columns[0].expr.value.function_call;
    try std.testing.expectEqualStrings("MAX", outer.name);
    const inner = outer.args[0].function_call;
    try std.testing.expectEqualStrings("ABS", inner.name);
}

// ── Stabilization: Parser Error & Edge Case Tests ─────────────────────

test "parse error on empty string" {
    var ast_arena = ast.AstArena.init(std.testing.allocator);
    defer ast_arena.deinit();
    var p = try Parser.init(std.testing.allocator, "", &ast_arena);
    defer p.deinit();
    const result = try p.parseStatement();
    try std.testing.expect(result == null);
}

test "parse error on just semicolons" {
    var ast_arena = ast.AstArena.init(std.testing.allocator);
    defer ast_arena.deinit();
    var p = try Parser.init(std.testing.allocator, ";;;", &ast_arena);
    defer p.deinit();
    // Should return null (no statement to parse) or succeed silently
    const result = try p.parseStatement();
    try std.testing.expect(result == null);
}

test "parse error on incomplete SELECT" {
    var ast_arena = ast.AstArena.init(std.testing.allocator);
    defer ast_arena.deinit();
    var p = try Parser.init(std.testing.allocator, "SELECT", &ast_arena);
    defer p.deinit();
    const result = p.parseStatement();
    try std.testing.expect(result == error.ParseFailed);
    try std.testing.expect(p.errors.items.len > 0);
}

test "parse error on incomplete INSERT" {
    var ast_arena = ast.AstArena.init(std.testing.allocator);
    defer ast_arena.deinit();
    var p = try Parser.init(std.testing.allocator, "INSERT INTO", &ast_arena);
    defer p.deinit();
    const result = p.parseStatement();
    try std.testing.expect(result == error.ParseFailed);
    try std.testing.expect(p.errors.items.len > 0);
}

test "parse error on incomplete CREATE TABLE" {
    var ast_arena = ast.AstArena.init(std.testing.allocator);
    defer ast_arena.deinit();
    var p = try Parser.init(std.testing.allocator, "CREATE TABLE", &ast_arena);
    defer p.deinit();
    const result = p.parseStatement();
    try std.testing.expect(result == error.ParseFailed);
}

test "parse error on missing closing parenthesis" {
    var ast_arena = ast.AstArena.init(std.testing.allocator);
    defer ast_arena.deinit();
    var p = try Parser.init(std.testing.allocator, "SELECT (1 + 2", &ast_arena);
    defer p.deinit();
    const result = p.parseStatement();
    try std.testing.expect(result == error.ParseFailed);
}

test "parse deeply nested parentheses" {
    var r = try testParseWithArena("SELECT ((((1 + 2)))) FROM t");
    defer r.deinit();
    try std.testing.expectEqual(@as(usize, 1), r.stmt.select.columns.len);
}

test "parse SELECT with all clause types combined" {
    var r = try testParseWithArena(
        "SELECT DISTINCT a, COUNT(b) FROM t1 INNER JOIN t2 ON t1.id = t2.fk WHERE a > 5 GROUP BY a HAVING COUNT(b) > 1 ORDER BY a DESC LIMIT 10 OFFSET 20",
    );
    defer r.deinit();
    const sel = r.stmt.select;
    try std.testing.expect(sel.distinct);
    try std.testing.expect(sel.where != null);
    try std.testing.expect(sel.group_by.len > 0);
    try std.testing.expect(sel.having != null);
    try std.testing.expect(sel.order_by.len > 0);
    try std.testing.expect(sel.limit != null);
    try std.testing.expect(sel.offset != null);
    try std.testing.expectEqual(@as(usize, 1), sel.joins.len);
}

test "parse keyword as table alias" {
    // Common pattern: using a keyword as an alias
    var r = try testParseWithArena("SELECT t.name FROM users t");
    defer r.deinit();
    try std.testing.expect(r.stmt.select.from.?.table_name.alias != null);
    try std.testing.expectEqualStrings("t", r.stmt.select.from.?.table_name.alias.?);
}

test "parse multiple column aliases" {
    var r = try testParseWithArena("SELECT a AS x, b AS y, c AS z FROM t");
    defer r.deinit();
    const cols = r.stmt.select.columns;
    try std.testing.expectEqual(@as(usize, 3), cols.len);
    try std.testing.expectEqualStrings("x", cols[0].expr.alias.?);
    try std.testing.expectEqualStrings("y", cols[1].expr.alias.?);
    try std.testing.expectEqualStrings("z", cols[2].expr.alias.?);
}

test "parse CREATE TABLE with multiple constraints" {
    var r = try testParseWithArena(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY NOT NULL, customer_id INTEGER NOT NULL, amount REAL DEFAULT 0.0, status TEXT DEFAULT 'pending')",
    );
    defer r.deinit();
    const create = r.stmt.create_table;
    try std.testing.expectEqual(@as(usize, 4), create.columns.len);
    try std.testing.expectEqualStrings("orders", create.name);
}

test "parse INSERT with many rows" {
    var r = try testParseWithArena(
        "INSERT INTO t VALUES (1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')",
    );
    defer r.deinit();
    try std.testing.expectEqual(@as(usize, 4), r.stmt.insert.values.len);
}

test "parse UPDATE with multiple SET assignments" {
    var r = try testParseWithArena(
        "UPDATE users SET name = 'Alice', age = 30, active = TRUE WHERE id = 1",
    );
    defer r.deinit();
    try std.testing.expectEqual(@as(usize, 3), r.stmt.update.assignments.len);
    try std.testing.expect(r.stmt.update.where != null);
}

test "parse complex WHERE with mixed operators" {
    var r = try testParseWithArena(
        "SELECT * FROM t WHERE (a > 1 AND b < 10) OR (c = 'x' AND d IS NOT NULL)",
    );
    defer r.deinit();
    try std.testing.expect(r.stmt.select.where != null);
    // The top-level expression should be an OR
    try std.testing.expect(r.stmt.select.where.?.binary_op.op == .@"or");
}

// ── CTE (WITH ... AS) tests ──────────────────────────────────

test "parse simple CTE" {
    var r = try testParseWithArena(
        "WITH cte AS (SELECT 1) SELECT * FROM cte",
    );
    defer r.deinit();
    const sel = r.stmt.select;
    try std.testing.expectEqual(@as(usize, 1), sel.ctes.len);
    try std.testing.expectEqualStrings("cte", sel.ctes[0].name);
    try std.testing.expect(!sel.recursive);
    try std.testing.expectEqual(@as(usize, 0), sel.ctes[0].column_names.len);
    // Main query references CTE
    try std.testing.expectEqualStrings("cte", sel.from.?.table_name.name);
}

test "parse CTE with column aliases" {
    var r = try testParseWithArena(
        "WITH cte(x, y) AS (SELECT 1, 2) SELECT x, y FROM cte",
    );
    defer r.deinit();
    const sel = r.stmt.select;
    try std.testing.expectEqual(@as(usize, 1), sel.ctes.len);
    try std.testing.expectEqualStrings("cte", sel.ctes[0].name);
    try std.testing.expectEqual(@as(usize, 2), sel.ctes[0].column_names.len);
    try std.testing.expectEqualStrings("x", sel.ctes[0].column_names[0]);
    try std.testing.expectEqualStrings("y", sel.ctes[0].column_names[1]);
}

test "parse multiple CTEs" {
    var r = try testParseWithArena(
        "WITH a AS (SELECT 1), b AS (SELECT 2) SELECT * FROM a, b",
    );
    defer r.deinit();
    const sel = r.stmt.select;
    try std.testing.expectEqual(@as(usize, 2), sel.ctes.len);
    try std.testing.expectEqualStrings("a", sel.ctes[0].name);
    try std.testing.expectEqualStrings("b", sel.ctes[1].name);
    try std.testing.expect(!sel.recursive);
}

test "parse WITH RECURSIVE flag" {
    // RECURSIVE flag is parsed even without actual recursion (UNION ALL support is separate)
    var r = try testParseWithArena(
        "WITH RECURSIVE cnt(x) AS (SELECT 1) SELECT x FROM cnt",
    );
    defer r.deinit();
    const sel = r.stmt.select;
    try std.testing.expect(sel.recursive);
    try std.testing.expectEqual(@as(usize, 1), sel.ctes.len);
    try std.testing.expectEqualStrings("cnt", sel.ctes[0].name);
    try std.testing.expectEqual(@as(usize, 1), sel.ctes[0].column_names.len);
    try std.testing.expectEqualStrings("x", sel.ctes[0].column_names[0]);
}

test "parse CTE without FROM" {
    var r = try testParseWithArena(
        "WITH vals AS (SELECT 42) SELECT * FROM vals",
    );
    defer r.deinit();
    const sel = r.stmt.select;
    try std.testing.expectEqual(@as(usize, 1), sel.ctes.len);
    // CTE inner select has no FROM
    try std.testing.expect(sel.ctes[0].select.from == null);
}

// ── Set operation tests ──────────────────────────────────────

test "parse UNION" {
    var r = try testParseWithArena(
        "SELECT id FROM users UNION SELECT id FROM orders",
    );
    defer r.deinit();
    const sel = r.stmt.select;
    try std.testing.expectEqualStrings("users", sel.from.?.table_name.name);
    try std.testing.expect(sel.set_operation != null);
    try std.testing.expectEqual(ast.SetOpType.@"union", sel.set_operation.?.op);
    try std.testing.expectEqualStrings("orders", sel.set_operation.?.right.from.?.table_name.name);
}

test "parse UNION ALL" {
    var r = try testParseWithArena(
        "SELECT id FROM t1 UNION ALL SELECT id FROM t2",
    );
    defer r.deinit();
    const sel = r.stmt.select;
    try std.testing.expect(sel.set_operation != null);
    try std.testing.expectEqual(ast.SetOpType.union_all, sel.set_operation.?.op);
}

test "parse INTERSECT" {
    var r = try testParseWithArena(
        "SELECT id FROM t1 INTERSECT SELECT id FROM t2",
    );
    defer r.deinit();
    const sel = r.stmt.select;
    try std.testing.expect(sel.set_operation != null);
    try std.testing.expectEqual(ast.SetOpType.intersect, sel.set_operation.?.op);
}

test "parse EXCEPT" {
    var r = try testParseWithArena(
        "SELECT id FROM t1 EXCEPT SELECT id FROM t2",
    );
    defer r.deinit();
    const sel = r.stmt.select;
    try std.testing.expect(sel.set_operation != null);
    try std.testing.expectEqual(ast.SetOpType.except, sel.set_operation.?.op);
}

test "parse chained set operations" {
    var r = try testParseWithArena(
        "SELECT id FROM t1 UNION SELECT id FROM t2 INTERSECT SELECT id FROM t3",
    );
    defer r.deinit();
    const sel = r.stmt.select;
    // First op: UNION
    try std.testing.expect(sel.set_operation != null);
    try std.testing.expectEqual(ast.SetOpType.@"union", sel.set_operation.?.op);
    // The right side of UNION has an INTERSECT chain
    const right = sel.set_operation.?.right;
    try std.testing.expect(right.set_operation != null);
    try std.testing.expectEqual(ast.SetOpType.intersect, right.set_operation.?.op);
    try std.testing.expectEqualStrings("t3", right.set_operation.?.right.from.?.table_name.name);
}

test "parse set operation with ORDER BY and LIMIT" {
    var r = try testParseWithArena(
        "SELECT id FROM t1 UNION ALL SELECT id FROM t2 ORDER BY id LIMIT 10",
    );
    defer r.deinit();
    const sel = r.stmt.select;
    try std.testing.expect(sel.set_operation != null);
    try std.testing.expectEqual(ast.SetOpType.union_all, sel.set_operation.?.op);
    // ORDER BY and LIMIT belong to the outer (left) statement
    try std.testing.expectEqual(@as(usize, 1), sel.order_by.len);
    try std.testing.expect(sel.limit != null);
}

test "parse set operation with WHERE on both sides" {
    var r = try testParseWithArena(
        "SELECT id FROM t1 WHERE id > 5 UNION SELECT id FROM t2 WHERE id < 10",
    );
    defer r.deinit();
    const sel = r.stmt.select;
    try std.testing.expect(sel.where != null);
    try std.testing.expect(sel.set_operation != null);
    try std.testing.expect(sel.set_operation.?.right.where != null);
}

test "parse set operation with CTE" {
    var r = try testParseWithArena(
        "WITH cte AS (SELECT 1) SELECT * FROM cte UNION SELECT * FROM cte",
    );
    defer r.deinit();
    const sel = r.stmt.select;
    try std.testing.expectEqual(@as(usize, 1), sel.ctes.len);
    try std.testing.expect(sel.set_operation != null);
}

// ── CREATE VIEW / DROP VIEW tests ────────────────────────────

test "parse CREATE VIEW" {
    var r = try testParseWithArena("CREATE VIEW user_names AS SELECT id, name FROM users");
    defer r.deinit();
    const cv = r.stmt.create_view;
    try std.testing.expectEqualStrings("user_names", cv.name);
    try std.testing.expect(!cv.or_replace);
    try std.testing.expect(!cv.if_not_exists);
    try std.testing.expectEqual(@as(usize, 0), cv.column_names.len);
    try std.testing.expectEqual(@as(usize, 2), cv.select.columns.len);
}

test "parse CREATE OR REPLACE VIEW" {
    var r = try testParseWithArena("CREATE OR REPLACE VIEW v AS SELECT 1");
    defer r.deinit();
    const cv = r.stmt.create_view;
    try std.testing.expectEqualStrings("v", cv.name);
    try std.testing.expect(cv.or_replace);
    try std.testing.expect(!cv.if_not_exists);
}

test "parse CREATE VIEW IF NOT EXISTS" {
    var r = try testParseWithArena("CREATE VIEW IF NOT EXISTS v AS SELECT 1");
    defer r.deinit();
    const cv = r.stmt.create_view;
    try std.testing.expectEqualStrings("v", cv.name);
    try std.testing.expect(!cv.or_replace);
    try std.testing.expect(cv.if_not_exists);
}

test "parse CREATE VIEW with column aliases" {
    var r = try testParseWithArena("CREATE VIEW v (a, b, c) AS SELECT x, y, z FROM t");
    defer r.deinit();
    const cv = r.stmt.create_view;
    try std.testing.expectEqualStrings("v", cv.name);
    try std.testing.expectEqual(@as(usize, 3), cv.column_names.len);
    try std.testing.expectEqualStrings("a", cv.column_names[0]);
    try std.testing.expectEqualStrings("b", cv.column_names[1]);
    try std.testing.expectEqualStrings("c", cv.column_names[2]);
    try std.testing.expectEqual(@as(usize, 3), cv.select.columns.len);
}

test "parse CREATE VIEW with complex SELECT" {
    var r = try testParseWithArena(
        "CREATE VIEW active_users AS SELECT id, name FROM users WHERE active = 1 ORDER BY name",
    );
    defer r.deinit();
    const cv = r.stmt.create_view;
    try std.testing.expectEqualStrings("active_users", cv.name);
    try std.testing.expect(cv.select.where != null);
    try std.testing.expectEqual(@as(usize, 1), cv.select.order_by.len);
}

test "parse DROP VIEW" {
    var r = try testParseWithArena("DROP VIEW my_view");
    defer r.deinit();
    const dv = r.stmt.drop_view;
    try std.testing.expectEqualStrings("my_view", dv.name);
    try std.testing.expect(!dv.if_exists);
}

test "parse DROP VIEW IF EXISTS" {
    var r = try testParseWithArena("DROP VIEW IF EXISTS my_view");
    defer r.deinit();
    const dv = r.stmt.drop_view;
    try std.testing.expectEqualStrings("my_view", dv.name);
    try std.testing.expect(dv.if_exists);
}

test "parse CREATE VIEW WITH CHECK OPTION (default cascaded)" {
    var r = try testParseWithArena("CREATE VIEW v AS SELECT * FROM t WHERE x > 0 WITH CHECK OPTION");
    defer r.deinit();
    const cv = r.stmt.create_view;
    try std.testing.expectEqualStrings("v", cv.name);
    try std.testing.expectEqual(ast.CheckOption.cascaded, cv.check_option);
    try std.testing.expect(cv.select.where != null);
}

test "parse CREATE VIEW WITH LOCAL CHECK OPTION" {
    var r = try testParseWithArena("CREATE VIEW v AS SELECT id FROM t WITH LOCAL CHECK OPTION");
    defer r.deinit();
    const cv = r.stmt.create_view;
    try std.testing.expectEqual(ast.CheckOption.local, cv.check_option);
}

test "parse CREATE VIEW WITH CASCADED CHECK OPTION" {
    var r = try testParseWithArena("CREATE VIEW v AS SELECT id FROM t WITH CASCADED CHECK OPTION");
    defer r.deinit();
    const cv = r.stmt.create_view;
    try std.testing.expectEqual(ast.CheckOption.cascaded, cv.check_option);
}

test "parse CREATE VIEW without CHECK OPTION" {
    var r = try testParseWithArena("CREATE VIEW v AS SELECT * FROM t");
    defer r.deinit();
    const cv = r.stmt.create_view;
    try std.testing.expectEqual(ast.CheckOption.none, cv.check_option);
}

// ── Window Function tests ─────────────────────────────────────

test "parse ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC)" {
    var r = try testParseWithArena("SELECT ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC) FROM emp");
    defer r.deinit();
    const sel = r.stmt.select;
    try std.testing.expectEqual(@as(usize, 1), sel.columns.len);
    const col_expr = sel.columns[0].expr.value;
    const wf = col_expr.window_function;
    try std.testing.expectEqualStrings("ROW_NUMBER", wf.name);
    try std.testing.expectEqual(@as(usize, 0), wf.args.len);
    try std.testing.expectEqual(@as(usize, 1), wf.partition_by.len);
    try std.testing.expectEqual(@as(usize, 1), wf.order_by.len);
    try std.testing.expectEqual(ast.OrderDirection.desc, wf.order_by[0].direction);
    try std.testing.expect(wf.frame == null);
}

test "parse SUM(x) OVER (ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)" {
    var r = try testParseWithArena("SELECT SUM(x) OVER (ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM t");
    defer r.deinit();
    const wf = r.stmt.select.columns[0].expr.value.window_function;
    try std.testing.expectEqualStrings("SUM", wf.name);
    try std.testing.expectEqual(@as(usize, 1), wf.args.len);
    try std.testing.expect(wf.frame != null);
    try std.testing.expectEqual(ast.WindowFrameMode.rows, wf.frame.?.mode);
    try std.testing.expectEqual(ast.WindowFrameBound.unbounded_preceding, wf.frame.?.start);
    try std.testing.expectEqual(ast.WindowFrameBound.current_row, wf.frame.?.end);
}

test "parse RANK() OVER (ORDER BY score)" {
    var r = try testParseWithArena("SELECT RANK() OVER (ORDER BY score) FROM t");
    defer r.deinit();
    const wf = r.stmt.select.columns[0].expr.value.window_function;
    try std.testing.expectEqualStrings("RANK", wf.name);
    try std.testing.expectEqual(@as(usize, 0), wf.partition_by.len);
    try std.testing.expectEqual(@as(usize, 1), wf.order_by.len);
}

test "parse LAG(salary, 1) OVER (PARTITION BY dept ORDER BY id)" {
    var r = try testParseWithArena("SELECT LAG(salary, 1) OVER (PARTITION BY dept ORDER BY id) FROM emp");
    defer r.deinit();
    const wf = r.stmt.select.columns[0].expr.value.window_function;
    try std.testing.expectEqualStrings("LAG", wf.name);
    try std.testing.expectEqual(@as(usize, 2), wf.args.len);
    try std.testing.expectEqual(@as(usize, 1), wf.partition_by.len);
}

test "parse ROWS BETWEEN N PRECEDING AND N FOLLOWING" {
    var r = try testParseWithArena("SELECT AVG(val) OVER (ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING) FROM t");
    defer r.deinit();
    const wf = r.stmt.select.columns[0].expr.value.window_function;
    const frame = wf.frame.?;
    try std.testing.expectEqual(ast.WindowFrameMode.rows, frame.mode);
    switch (frame.start) {
        .expr_preceding => |e| try std.testing.expectEqual(@as(i64, 2), e.integer_literal),
        else => return error.ParseFailed,
    }
    switch (frame.end) {
        .expr_following => |e| try std.testing.expectEqual(@as(i64, 2), e.integer_literal),
        else => return error.ParseFailed,
    }
}

test "parse RANGE UNBOUNDED PRECEDING (short form)" {
    var r = try testParseWithArena("SELECT SUM(x) OVER (ORDER BY id RANGE UNBOUNDED PRECEDING) FROM t");
    defer r.deinit();
    const wf = r.stmt.select.columns[0].expr.value.window_function;
    const frame = wf.frame.?;
    try std.testing.expectEqual(ast.WindowFrameMode.range, frame.mode);
    try std.testing.expectEqual(ast.WindowFrameBound.unbounded_preceding, frame.start);
    try std.testing.expectEqual(ast.WindowFrameBound.current_row, frame.end);
}

test "parse DENSE_RANK with empty OVER()" {
    var r = try testParseWithArena("SELECT DENSE_RANK() OVER () FROM t");
    defer r.deinit();
    const wf = r.stmt.select.columns[0].expr.value.window_function;
    try std.testing.expectEqualStrings("DENSE_RANK", wf.name);
    try std.testing.expectEqual(@as(usize, 0), wf.partition_by.len);
    try std.testing.expectEqual(@as(usize, 0), wf.order_by.len);
    try std.testing.expect(wf.frame == null);
}

test "parse aggregate as window function: COUNT(*) OVER ()" {
    var r = try testParseWithArena("SELECT COUNT(*) OVER () FROM t");
    defer r.deinit();
    const wf = r.stmt.select.columns[0].expr.value.window_function;
    try std.testing.expectEqualStrings("COUNT", wf.name);
    try std.testing.expectEqual(@as(usize, 1), wf.args.len);
}

test "parse multiple window functions in SELECT" {
    var r = try testParseWithArena("SELECT ROW_NUMBER() OVER (ORDER BY id), RANK() OVER (ORDER BY score) FROM t");
    defer r.deinit();
    const sel = r.stmt.select;
    try std.testing.expectEqual(@as(usize, 2), sel.columns.len);
    const wf1 = sel.columns[0].expr.value.window_function;
    const wf2 = sel.columns[1].expr.value.window_function;
    try std.testing.expectEqualStrings("ROW_NUMBER", wf1.name);
    try std.testing.expectEqualStrings("RANK", wf2.name);
}

test "parse GROUPS frame mode" {
    var r = try testParseWithArena("SELECT SUM(x) OVER (ORDER BY id GROUPS BETWEEN 1 PRECEDING AND 1 FOLLOWING) FROM t");
    defer r.deinit();
    const frame = r.stmt.select.columns[0].expr.value.window_function.frame.?;
    try std.testing.expectEqual(ast.WindowFrameMode.groups, frame.mode);
}

test "parse WINDOW clause with named window definition" {
    var r = try testParseWithArena("SELECT ROW_NUMBER() OVER w FROM t WINDOW w AS (PARTITION BY dept ORDER BY salary DESC)");
    defer r.deinit();
    const sel = r.stmt.select;
    // Check window function references named window
    const wf = sel.columns[0].expr.value.window_function;
    try std.testing.expectEqualStrings("ROW_NUMBER", wf.name);
    try std.testing.expectEqualStrings("w", wf.window_name.?);
    try std.testing.expectEqual(@as(usize, 0), wf.partition_by.len);
    // Check window definition
    try std.testing.expectEqual(@as(usize, 1), sel.window_defs.len);
    try std.testing.expectEqualStrings("w", sel.window_defs[0].name);
    try std.testing.expectEqual(@as(usize, 1), sel.window_defs[0].partition_by.len);
    try std.testing.expectEqual(@as(usize, 1), sel.window_defs[0].order_by.len);
    try std.testing.expectEqual(ast.OrderDirection.desc, sel.window_defs[0].order_by[0].direction);
}

test "parse WINDOW clause with multiple named windows" {
    var r = try testParseWithArena("SELECT ROW_NUMBER() OVER w1, SUM(x) OVER w2 FROM t WINDOW w1 AS (ORDER BY id), w2 AS (PARTITION BY dept)");
    defer r.deinit();
    const sel = r.stmt.select;
    // Two window functions referencing different named windows
    try std.testing.expectEqualStrings("w1", sel.columns[0].expr.value.window_function.window_name.?);
    try std.testing.expectEqualStrings("w2", sel.columns[1].expr.value.window_function.window_name.?);
    // Two window definitions
    try std.testing.expectEqual(@as(usize, 2), sel.window_defs.len);
    try std.testing.expectEqualStrings("w1", sel.window_defs[0].name);
    try std.testing.expectEqualStrings("w2", sel.window_defs[1].name);
}

test "parse WINDOW clause with frame spec" {
    var r = try testParseWithArena("SELECT SUM(x) OVER w FROM t WINDOW w AS (ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)");
    defer r.deinit();
    const def = r.stmt.select.window_defs[0];
    try std.testing.expect(def.frame != null);
    try std.testing.expectEqual(ast.WindowFrameMode.rows, def.frame.?.mode);
    try std.testing.expectEqual(ast.WindowFrameBound.unbounded_preceding, def.frame.?.start);
    try std.testing.expectEqual(ast.WindowFrameBound.current_row, def.frame.?.end);
}

test "parse CREATE TABLE with SERIAL column" {
    var r = try testParseWithArena("CREATE TABLE t (id SERIAL, name TEXT)");
    defer r.deinit();
    const ct = r.stmt.create_table;
    try std.testing.expectEqual(@as(usize, 2), ct.columns.len);
    try std.testing.expectEqual(ast.DataType.type_serial, ct.columns[0].data_type.?);
    try std.testing.expectEqual(ast.DataType.type_text, ct.columns[1].data_type.?);
}

test "parse CREATE TABLE with BIGSERIAL column" {
    var r = try testParseWithArena("CREATE TABLE t (id BIGSERIAL, payload TEXT)");
    defer r.deinit();
    const ct = r.stmt.create_table;
    try std.testing.expectEqual(@as(usize, 2), ct.columns.len);
    try std.testing.expectEqual(ast.DataType.type_bigserial, ct.columns[0].data_type.?);
}

test "parse CAST to SERIAL" {
    var r = try testParseWithArena("SELECT CAST(x AS SERIAL) FROM t");
    defer r.deinit();
    const expr = r.stmt.select.columns[0].expr.value;
    try std.testing.expect(expr.* == .cast);
    try std.testing.expectEqual(ast.DataType.type_serial, expr.cast.target_type);
}

test "parse ARRAY[1, 2, 3] constructor" {
    var r = try testParseWithArena("SELECT ARRAY[1, 2, 3] FROM t");
    defer r.deinit();
    const expr = r.stmt.select.columns[0].expr.value;
    try std.testing.expect(expr.* == .array_constructor);
    const elements = expr.array_constructor;
    try std.testing.expectEqual(@as(usize, 3), elements.len);
    try std.testing.expectEqual(@as(i64, 1), elements[0].integer_literal);
    try std.testing.expectEqual(@as(i64, 2), elements[1].integer_literal);
    try std.testing.expectEqual(@as(i64, 3), elements[2].integer_literal);
}

test "parse array subscript col[1]" {
    var r = try testParseWithArena("SELECT col[1] FROM t");
    defer r.deinit();
    const expr = r.stmt.select.columns[0].expr.value;
    try std.testing.expect(expr.* == .array_subscript);
    const sub = expr.array_subscript;
    try std.testing.expect(sub.array.* == .column_ref);
    try std.testing.expectEqualStrings("col", sub.array.column_ref.name);
    try std.testing.expect(sub.index.* == .integer_literal);
    try std.testing.expectEqual(@as(i64, 1), sub.index.integer_literal);
}

test "parse CREATE TABLE with INTEGER[] array column" {
    var r = try testParseWithArena("CREATE TABLE t (tags INTEGER[])");
    defer r.deinit();
    const ct = r.stmt.create_table;
    try std.testing.expectEqual(@as(usize, 1), ct.columns.len);
    try std.testing.expectEqual(ast.DataType.type_array, ct.columns[0].data_type.?);
}

test "parse CREATE TABLE with INTEGER ARRAY column" {
    var r = try testParseWithArena("CREATE TABLE t (tags INTEGER ARRAY)");
    defer r.deinit();
    const ct = r.stmt.create_table;
    try std.testing.expectEqual(@as(usize, 1), ct.columns.len);
    try std.testing.expectEqual(ast.DataType.type_array, ct.columns[0].data_type.?);
}

test "parse nested array constructor ARRAY[ARRAY[1,2], ARRAY[3,4]]" {
    var r = try testParseWithArena("SELECT ARRAY[ARRAY[1,2], ARRAY[3,4]] FROM t");
    defer r.deinit();
    const expr = r.stmt.select.columns[0].expr.value;
    try std.testing.expect(expr.* == .array_constructor);
    const elements = expr.array_constructor;
    try std.testing.expectEqual(@as(usize, 2), elements.len);
    try std.testing.expect(elements[0].* == .array_constructor);
    try std.testing.expect(elements[1].* == .array_constructor);
}

test "parse array subscript with expression index col[id + 1]" {
    var r = try testParseWithArena("SELECT col[id + 1] FROM t");
    defer r.deinit();
    const expr = r.stmt.select.columns[0].expr.value;
    try std.testing.expect(expr.* == .array_subscript);
    const sub = expr.array_subscript;
    try std.testing.expect(sub.index.* == .binary_op);
}

test "parse CAST to ARRAY" {
    var r = try testParseWithArena("SELECT CAST(x AS ARRAY) FROM t");
    defer r.deinit();
    const expr = r.stmt.select.columns[0].expr.value;
    try std.testing.expect(expr.* == .cast);
    try std.testing.expectEqual(ast.DataType.type_array, expr.cast.target_type);
}

test "parse ANY with array" {
    var r = try testParseWithArena("SELECT 5 = ANY(ARRAY[1, 2, 5]) FROM t");
    defer r.deinit();
    const expr = r.stmt.select.columns[0].expr.value;
    try std.testing.expect(expr.* == .any);
    const any_expr = expr.any;
    try std.testing.expect(any_expr.expr.* == .integer_literal);
    try std.testing.expectEqual(@as(i64, 5), any_expr.expr.integer_literal);
    try std.testing.expectEqual(ast.BinaryOp.equal, any_expr.op);
    try std.testing.expect(any_expr.array.* == .array_constructor);
}

test "parse ALL with array" {
    var r = try testParseWithArena("SELECT x > ALL(ARRAY[1, 2, 3]) FROM t");
    defer r.deinit();
    const expr = r.stmt.select.columns[0].expr.value;
    try std.testing.expect(expr.* == .all);
    const all_expr = expr.all;
    try std.testing.expect(all_expr.expr.* == .column_ref);
    try std.testing.expectEqual(ast.BinaryOp.greater_than, all_expr.op);
    try std.testing.expect(all_expr.array.* == .array_constructor);
}

test "parse ANY with column array" {
    var r = try testParseWithArena("SELECT 'foo' = ANY(tags) FROM t");
    defer r.deinit();
    const expr = r.stmt.select.columns[0].expr.value;
    try std.testing.expect(expr.* == .any);
    const any_expr = expr.any;
    try std.testing.expect(any_expr.expr.* == .string_literal);
    try std.testing.expectEqualStrings("foo", any_expr.expr.string_literal);
    try std.testing.expect(any_expr.array.* == .column_ref);
    try std.testing.expectEqualStrings("tags", any_expr.array.column_ref.name);
}

// ── ENUM type tests ───────────────────────────────────────────

test "parse CREATE TYPE AS ENUM" {
    var r = try testParseWithArena("CREATE TYPE mood AS ENUM ('happy', 'sad', 'neutral')");
    defer r.deinit();
    const ct = r.stmt.create_type;
    try std.testing.expectEqualStrings("mood", ct.name);
    try std.testing.expectEqual(@as(usize, 3), ct.values.len);
    try std.testing.expectEqualStrings("'happy'", ct.values[0]);
    try std.testing.expectEqualStrings("'sad'", ct.values[1]);
    try std.testing.expectEqualStrings("'neutral'", ct.values[2]);
}

test "parse CREATE TYPE with single enum value" {
    var r = try testParseWithArena("CREATE TYPE status AS ENUM ('active')");
    defer r.deinit();
    const ct = r.stmt.create_type;
    try std.testing.expectEqualStrings("status", ct.name);
    try std.testing.expectEqual(@as(usize, 1), ct.values.len);
    try std.testing.expectEqualStrings("'active'", ct.values[0]);
}

test "parse DROP TYPE" {
    var r = try testParseWithArena("DROP TYPE mood");
    defer r.deinit();
    const dt = r.stmt.drop_type;
    try std.testing.expectEqualStrings("mood", dt.name);
    try std.testing.expect(!dt.if_exists);
}

test "parse DROP TYPE IF EXISTS" {
    var r = try testParseWithArena("DROP TYPE IF EXISTS mood");
    defer r.deinit();
    const dt = r.stmt.drop_type;
    try std.testing.expectEqualStrings("mood", dt.name);
    try std.testing.expect(dt.if_exists);
}

// ── JSON type tests ───────────────────────────────────────────

test "parse CREATE TABLE with JSON column" {
    var r = try testParseWithArena("CREATE TABLE t (data JSON)");
    defer r.deinit();
    const ct = r.stmt.create_table;
    try std.testing.expectEqual(@as(usize, 1), ct.columns.len);
    try std.testing.expectEqual(ast.DataType.type_json, ct.columns[0].data_type.?);
}

test "parse CREATE TABLE with JSONB column" {
    var r = try testParseWithArena("CREATE TABLE t (metadata JSONB)");
    defer r.deinit();
    const ct = r.stmt.create_table;
    try std.testing.expectEqual(@as(usize, 1), ct.columns.len);
    try std.testing.expectEqual(ast.DataType.type_jsonb, ct.columns[0].data_type.?);
}

test "parse CAST to JSON" {
    var r = try testParseWithArena("SELECT CAST('{}' AS JSON) FROM t");
    defer r.deinit();
    const expr = r.stmt.select.columns[0].expr.value;
    try std.testing.expect(expr.* == .cast);
    try std.testing.expectEqual(ast.DataType.type_json, expr.cast.target_type);
}

test "parse CAST to JSONB" {
    var r = try testParseWithArena("SELECT CAST('[1,2,3]' AS JSONB) FROM t");
    defer r.deinit();
    const expr = r.stmt.select.columns[0].expr.value;
    try std.testing.expect(expr.* == .cast);
    try std.testing.expectEqual(ast.DataType.type_jsonb, expr.cast.target_type);
}

test "parse JSON extract operator (->)" {
    var r = try testParseWithArena("SELECT data -> 'name' FROM users");
    defer r.deinit();
    const expr = r.stmt.select.columns[0].expr.value;
    try std.testing.expect(expr.* == .binary_op);
    try std.testing.expectEqual(ast.BinaryOp.json_extract, expr.binary_op.op);
}

test "parse JSON extract text operator (->>)" {
    var r = try testParseWithArena("SELECT data ->> 'name' FROM users");
    defer r.deinit();
    const expr = r.stmt.select.columns[0].expr.value;
    try std.testing.expect(expr.* == .binary_op);
    try std.testing.expectEqual(ast.BinaryOp.json_extract_text, expr.binary_op.op);
}

test "parse JSON contains operator (@>)" {
    var r = try testParseWithArena("SELECT * FROM users WHERE data @> '{\"age\":30}'");
    defer r.deinit();
    const where_expr = r.stmt.select.where.?;
    try std.testing.expect(where_expr.* == .binary_op);
    try std.testing.expectEqual(ast.BinaryOp.json_contains, where_expr.binary_op.op);
}

test "parse JSON contained by operator (<@)" {
    var r = try testParseWithArena("SELECT * FROM users WHERE '{\"a\":1}' <@ data");
    defer r.deinit();
    const where_expr = r.stmt.select.where.?;
    try std.testing.expect(where_expr.* == .binary_op);
    try std.testing.expectEqual(ast.BinaryOp.json_contained_by, where_expr.binary_op.op);
}

// NOTE: Disabled because ? is now used for bind parameters, not JSON key exists operator.
// If we need JSON key exists, we should use ?| or ?& operators with explicit syntax,
// or use a different bind parameter syntax (e.g., $1, $2 like PostgreSQL).
//test "parse JSON key exists operator (?)" {
//    var r = try testParseWithArena("SELECT * FROM users WHERE data ? 'name'");
//    defer r.deinit();
//    const where_expr = r.stmt.select.where.?;
//    try std.testing.expect(where_expr.* == .binary_op);
//    try std.testing.expectEqual(ast.BinaryOp.json_key_exists, where_expr.binary_op.op);
//}

test "parse JSON any key exists operator (?|)" {
    var r = try testParseWithArena("SELECT * FROM users WHERE data ?| ARRAY['a','b']");
    defer r.deinit();
    const where_expr = r.stmt.select.where.?;
    try std.testing.expect(where_expr.* == .binary_op);
    try std.testing.expectEqual(ast.BinaryOp.json_any_key_exists, where_expr.binary_op.op);
}

test "parse JSON all keys exist operator (?&)" {
    var r = try testParseWithArena("SELECT * FROM users WHERE data ?& ARRAY['a','b']");
    defer r.deinit();
    const where_expr = r.stmt.select.where.?;
    try std.testing.expect(where_expr.* == .binary_op);
    try std.testing.expectEqual(ast.BinaryOp.json_all_keys_exist, where_expr.binary_op.op);
}

test "parse JSON path extract operator (#>)" {
    var r = try testParseWithArena("SELECT data #> '{a,b}' FROM users");
    defer r.deinit();
    const expr = r.stmt.select.columns[0].expr.value;
    try std.testing.expect(expr.* == .binary_op);
    try std.testing.expectEqual(ast.BinaryOp.json_path_extract, expr.binary_op.op);
}

test "parse JSON path extract text operator (#>>)" {
    var r = try testParseWithArena("SELECT data #>> '{a,b}' FROM users");
    defer r.deinit();
    const expr = r.stmt.select.columns[0].expr.value;
    try std.testing.expect(expr.* == .binary_op);
    try std.testing.expectEqual(ast.BinaryOp.json_path_extract_text, expr.binary_op.op);
}

test "parse JSON delete path operator (#-)" {
    var r = try testParseWithArena("SELECT data #- '{a}' FROM users");
    defer r.deinit();
    const expr = r.stmt.select.columns[0].expr.value;
    try std.testing.expect(expr.* == .binary_op);
    try std.testing.expectEqual(ast.BinaryOp.json_delete_path, expr.binary_op.op);
}

test "parse chained JSON operators" {
    var r = try testParseWithArena("SELECT data -> 'user' ->> 'name' FROM users");
    defer r.deinit();
    const expr = r.stmt.select.columns[0].expr.value;
    // Should parse as: (data -> 'user') ->> 'name'
    try std.testing.expect(expr.* == .binary_op);
    try std.testing.expectEqual(ast.BinaryOp.json_extract_text, expr.binary_op.op);
    try std.testing.expect(expr.binary_op.left.* == .binary_op);
    try std.testing.expectEqual(ast.BinaryOp.json_extract, expr.binary_op.left.binary_op.op);
}

// ── CREATE FUNCTION / DROP FUNCTION tests ────────────────────────

test "parse CREATE FUNCTION with scalar return" {
    var r = try testParseWithArena(
        \\CREATE FUNCTION add(a INTEGER, b INTEGER)
        \\RETURNS INTEGER
        \\LANGUAGE sfl
        \\AS 'a + b'
    );
    defer r.deinit();
    try std.testing.expect(r.stmt == .create_function);
    const func = r.stmt.create_function;
    try std.testing.expectEqualStrings("add", func.name);
    try std.testing.expectEqual(@as(usize, 2), func.parameters.len);
    try std.testing.expectEqualStrings("a", func.parameters[0].name);
    try std.testing.expectEqual(ast.DataType.type_integer, func.parameters[0].data_type);
    try std.testing.expectEqualStrings("b", func.parameters[1].name);
    try std.testing.expectEqual(ast.DataType.type_integer, func.parameters[1].data_type);
    try std.testing.expect(func.return_type == .scalar);
    try std.testing.expectEqual(ast.DataType.type_integer, func.return_type.scalar);
    try std.testing.expectEqualStrings("sfl", func.language);
    try std.testing.expectEqualStrings("'a + b'", func.body);
    try std.testing.expectEqual(ast.FunctionVolatility.vol, func.volatility);
    try std.testing.expectEqual(false, func.or_replace);
}

test "parse CREATE FUNCTION with SETOF return" {
    var r = try testParseWithArena(
        \\CREATE FUNCTION get_numbers(n INTEGER)
        \\RETURNS SETOF INTEGER
        \\AS 'SELECT generate_series(1, n)'
    );
    defer r.deinit();
    try std.testing.expect(r.stmt == .create_function);
    const func = r.stmt.create_function;
    try std.testing.expectEqualStrings("get_numbers", func.name);
    try std.testing.expect(func.return_type == .setof);
    try std.testing.expectEqual(ast.DataType.type_integer, func.return_type.setof);
}

test "parse CREATE FUNCTION with TABLE return" {
    var r = try testParseWithArena(
        \\CREATE FUNCTION user_info()
        \\RETURNS TABLE(id INTEGER, name TEXT)
        \\AS 'SELECT id, name FROM users'
    );
    defer r.deinit();
    try std.testing.expect(r.stmt == .create_function);
    const func = r.stmt.create_function;
    try std.testing.expectEqualStrings("user_info", func.name);
    try std.testing.expect(func.return_type == .table);
    try std.testing.expectEqual(@as(usize, 2), func.return_type.table.len);
    try std.testing.expectEqualStrings("id", func.return_type.table[0].name);
    try std.testing.expectEqual(ast.DataType.type_integer, func.return_type.table[0].data_type);
    try std.testing.expectEqualStrings("name", func.return_type.table[1].name);
    try std.testing.expectEqual(ast.DataType.type_text, func.return_type.table[1].data_type);
}

test "parse CREATE OR REPLACE FUNCTION" {
    var r = try testParseWithArena(
        \\CREATE OR REPLACE FUNCTION double(x INTEGER)
        \\RETURNS INTEGER
        \\AS 'x * 2'
    );
    defer r.deinit();
    try std.testing.expect(r.stmt == .create_function);
    const func = r.stmt.create_function;
    try std.testing.expectEqualStrings("double", func.name);
    try std.testing.expectEqual(true, func.or_replace);
}

test "parse CREATE FUNCTION with IMMUTABLE" {
    var r = try testParseWithArena(
        \\CREATE FUNCTION square(x INTEGER)
        \\RETURNS INTEGER
        \\IMMUTABLE
        \\AS 'x * x'
    );
    defer r.deinit();
    try std.testing.expect(r.stmt == .create_function);
    const func = r.stmt.create_function;
    try std.testing.expectEqual(ast.FunctionVolatility.immutable, func.volatility);
}

test "parse CREATE FUNCTION with STABLE" {
    var r = try testParseWithArena(
        \\CREATE FUNCTION current_user_id()
        \\RETURNS INTEGER
        \\STABLE
        \\AS 'SELECT id FROM users WHERE username = current_user()'
    );
    defer r.deinit();
    try std.testing.expect(r.stmt == .create_function);
    const func = r.stmt.create_function;
    try std.testing.expectEqual(ast.FunctionVolatility.stable, func.volatility);
}

test "parse CREATE FUNCTION with VOLATILE" {
    var r = try testParseWithArena(
        \\CREATE FUNCTION random_value()
        \\RETURNS INTEGER
        \\VOLATILE
        \\AS 'SELECT random()'
    );
    defer r.deinit();
    try std.testing.expect(r.stmt == .create_function);
    const func = r.stmt.create_function;
    try std.testing.expectEqual(ast.FunctionVolatility.vol, func.volatility);
}

test "parse CREATE FUNCTION with no parameters" {
    var r = try testParseWithArena(
        \\CREATE FUNCTION get_timestamp()
        \\RETURNS TIMESTAMP
        \\AS 'SELECT NOW()'
    );
    defer r.deinit();
    try std.testing.expect(r.stmt == .create_function);
    const func = r.stmt.create_function;
    try std.testing.expectEqualStrings("get_timestamp", func.name);
    try std.testing.expectEqual(@as(usize, 0), func.parameters.len);
    try std.testing.expectEqual(ast.DataType.type_timestamp, func.return_type.scalar);
}

test "parse DROP FUNCTION" {
    var r = try testParseWithArena("DROP FUNCTION add");
    defer r.deinit();
    try std.testing.expect(r.stmt == .drop_function);
    const func = r.stmt.drop_function;
    try std.testing.expectEqualStrings("add", func.name);
    try std.testing.expectEqual(@as(usize, 0), func.param_types.len);
    try std.testing.expectEqual(false, func.if_exists);
}

test "parse DROP FUNCTION IF EXISTS" {
    var r = try testParseWithArena("DROP FUNCTION IF EXISTS add");
    defer r.deinit();
    try std.testing.expect(r.stmt == .drop_function);
    const func = r.stmt.drop_function;
    try std.testing.expectEqualStrings("add", func.name);
    try std.testing.expectEqual(true, func.if_exists);
}

test "parse DROP FUNCTION with parameter types" {
    var r = try testParseWithArena("DROP FUNCTION add(INTEGER, INTEGER)");
    defer r.deinit();
    try std.testing.expect(r.stmt == .drop_function);
    const func = r.stmt.drop_function;
    try std.testing.expectEqualStrings("add", func.name);
    try std.testing.expectEqual(@as(usize, 2), func.param_types.len);
    try std.testing.expectEqual(ast.DataType.type_integer, func.param_types[0]);
    try std.testing.expectEqual(ast.DataType.type_integer, func.param_types[1]);
}

test "parse DROP FUNCTION IF EXISTS with parameter types" {
    var r = try testParseWithArena("DROP FUNCTION IF EXISTS format(TEXT, INTEGER)");
    defer r.deinit();
    try std.testing.expect(r.stmt == .drop_function);
    const func = r.stmt.drop_function;
    try std.testing.expectEqualStrings("format", func.name);
    try std.testing.expectEqual(@as(usize, 2), func.param_types.len);
    try std.testing.expectEqual(ast.DataType.type_text, func.param_types[0]);
    try std.testing.expectEqual(ast.DataType.type_integer, func.param_types[1]);
    try std.testing.expectEqual(true, func.if_exists);
}

test "parse CREATE TRIGGER AFTER INSERT" {
    var r = try testParseWithArena("CREATE TRIGGER audit_log AFTER INSERT ON users FOR EACH ROW AS 'INSERT INTO audit VALUES (NEW.id)'");
    defer r.deinit();
    try std.testing.expect(r.stmt == .create_trigger);
    const trig = r.stmt.create_trigger;
    try std.testing.expectEqualStrings("audit_log", trig.name);
    try std.testing.expectEqualStrings("users", trig.table_name);
    try std.testing.expectEqual(ast.TriggerTiming.after, trig.timing);
    try std.testing.expectEqual(ast.TriggerEvent.insert, trig.event);
    try std.testing.expectEqual(ast.TriggerLevel.row, trig.level);
    try std.testing.expectEqual(false, trig.or_replace);
}

test "parse CREATE TRIGGER BEFORE UPDATE OF columns" {
    var r = try testParseWithArena("CREATE TRIGGER validate_email BEFORE UPDATE OF email, name ON users FOR EACH ROW AS 'SELECT check_email(NEW.email)'");
    defer r.deinit();
    try std.testing.expect(r.stmt == .create_trigger);
    const trig = r.stmt.create_trigger;
    try std.testing.expectEqualStrings("validate_email", trig.name);
    try std.testing.expectEqual(ast.TriggerEvent.update, trig.event);
    try std.testing.expectEqual(@as(usize, 2), trig.update_columns.len);
    try std.testing.expectEqualStrings("email", trig.update_columns[0]);
    try std.testing.expectEqualStrings("name", trig.update_columns[1]);
}

test "parse CREATE TRIGGER INSTEAD OF for views" {
    var r = try testParseWithArena("CREATE TRIGGER view_insert INSTEAD OF INSERT ON user_view FOR EACH ROW AS 'INSERT INTO users VALUES (NEW.id, NEW.name)'");
    defer r.deinit();
    try std.testing.expect(r.stmt == .create_trigger);
    const trig = r.stmt.create_trigger;
    try std.testing.expectEqual(ast.TriggerTiming.instead_of, trig.timing);
    try std.testing.expectEqual(ast.TriggerEvent.insert, trig.event);
}

test "parse CREATE TRIGGER with WHEN condition" {
    var r = try testParseWithArena("CREATE TRIGGER check_balance BEFORE UPDATE ON accounts FOR EACH ROW WHEN (NEW.balance < 0) AS 'SELECT RAISE(ABORT)'");
    defer r.deinit();
    try std.testing.expect(r.stmt == .create_trigger);
    const trig = r.stmt.create_trigger;
    try std.testing.expect(trig.when_condition != null);
}

test "parse CREATE OR REPLACE TRIGGER" {
    var r = try testParseWithArena("CREATE OR REPLACE TRIGGER audit_update AFTER UPDATE ON users FOR EACH ROW AS 'INSERT INTO audit VALUES (OLD.id, NEW.id)'");
    defer r.deinit();
    try std.testing.expect(r.stmt == .create_trigger);
    const trig = r.stmt.create_trigger;
    try std.testing.expectEqual(true, trig.or_replace);
}

test "parse CREATE TRIGGER FOR EACH STATEMENT" {
    var r = try testParseWithArena("CREATE TRIGGER cascade_delete AFTER DELETE ON departments FOR EACH STATEMENT AS 'DELETE FROM employees WHERE dept_id = OLD.id'");
    defer r.deinit();
    try std.testing.expect(r.stmt == .create_trigger);
    const trig = r.stmt.create_trigger;
    try std.testing.expectEqual(ast.TriggerLevel.statement, trig.level);
}

test "parse DROP TRIGGER" {
    var r = try testParseWithArena("DROP TRIGGER audit_log");
    defer r.deinit();
    try std.testing.expect(r.stmt == .drop_trigger);
    const trig = r.stmt.drop_trigger;
    try std.testing.expectEqualStrings("audit_log", trig.name);
    try std.testing.expectEqual(false, trig.if_exists);
}

test "parse DROP TRIGGER IF EXISTS with table name" {
    var r = try testParseWithArena("DROP TRIGGER IF EXISTS audit_log ON users");
    defer r.deinit();
    try std.testing.expect(r.stmt == .drop_trigger);
    const trig = r.stmt.drop_trigger;
    try std.testing.expectEqualStrings("audit_log", trig.name);
    try std.testing.expectEqualStrings("users", trig.table_name.?);
    try std.testing.expectEqual(true, trig.if_exists);
}

test "parse CREATE TRIGGER TRUNCATE event" {
    var r = try testParseWithArena("CREATE TRIGGER log_truncate AFTER TRUNCATE ON sensitive_data FOR EACH STATEMENT AS 'INSERT INTO security_log VALUES (NOW())'");
    defer r.deinit();
    try std.testing.expect(r.stmt == .create_trigger);
    const trig = r.stmt.create_trigger;
    try std.testing.expectEqual(ast.TriggerEvent.truncate, trig.event);
}

test "parse ALTER TRIGGER ENABLE" {
    var r = try testParseWithArena("ALTER TRIGGER check_constraint ENABLE");
    defer r.deinit();
    try std.testing.expect(r.stmt == .alter_trigger);
    const trig = r.stmt.alter_trigger;
    try std.testing.expectEqualStrings("check_constraint", trig.name);
    try std.testing.expect(trig.enable);
    try std.testing.expect(trig.table_name == null);
}

test "parse ALTER TRIGGER DISABLE" {
    var r = try testParseWithArena("ALTER TRIGGER audit_log DISABLE");
    defer r.deinit();
    try std.testing.expect(r.stmt == .alter_trigger);
    const trig = r.stmt.alter_trigger;
    try std.testing.expectEqualStrings("audit_log", trig.name);
    try std.testing.expect(!trig.enable);
    try std.testing.expect(trig.table_name == null);
}

test "parse ALTER TRIGGER ENABLE with table name" {
    var r = try testParseWithArena("ALTER TRIGGER check_constraint ON orders ENABLE");
    defer r.deinit();
    try std.testing.expect(r.stmt == .alter_trigger);
    const trig = r.stmt.alter_trigger;
    try std.testing.expectEqualStrings("check_constraint", trig.name);
    try std.testing.expectEqualStrings("orders", trig.table_name.?);
    try std.testing.expect(trig.enable);
}

test "parse ALTER TRIGGER DISABLE with table name" {
    var r = try testParseWithArena("ALTER TRIGGER audit_log ON users DISABLE");
    defer r.deinit();
    try std.testing.expect(r.stmt == .alter_trigger);
    const trig = r.stmt.alter_trigger;
    try std.testing.expectEqualStrings("audit_log", trig.name);
    try std.testing.expectEqualStrings("users", trig.table_name.?);
    try std.testing.expect(!trig.enable);
}

test "parse CREATE ROLE basic" {
    var r = try testParseWithArena("CREATE ROLE admin");
    defer r.deinit();
    try std.testing.expect(r.stmt == .create_role);
    const role = r.stmt.create_role;
    try std.testing.expectEqualStrings("admin", role.name);
    try std.testing.expect(role.options.login == null);
}

test "parse CREATE ROLE with LOGIN" {
    var r = try testParseWithArena("CREATE ROLE app_user WITH LOGIN");
    defer r.deinit();
    try std.testing.expect(r.stmt == .create_role);
    const role = r.stmt.create_role;
    try std.testing.expectEqualStrings("app_user", role.name);
    try std.testing.expectEqual(true, role.options.login.?);
}

test "parse CREATE ROLE with NOLOGIN" {
    var r = try testParseWithArena("CREATE ROLE group_role NOLOGIN");
    defer r.deinit();
    try std.testing.expect(r.stmt == .create_role);
    const role = r.stmt.create_role;
    try std.testing.expectEqualStrings("group_role", role.name);
    try std.testing.expectEqual(false, role.options.login.?);
}

test "parse CREATE ROLE with multiple options" {
    var r = try testParseWithArena("CREATE ROLE superadmin WITH LOGIN SUPERUSER CREATEDB");
    defer r.deinit();
    try std.testing.expect(r.stmt == .create_role);
    const role = r.stmt.create_role;
    try std.testing.expectEqualStrings("superadmin", role.name);
    try std.testing.expectEqual(true, role.options.login.?);
    try std.testing.expectEqual(true, role.options.superuser.?);
    try std.testing.expectEqual(true, role.options.createdb.?);
}

test "parse CREATE ROLE with PASSWORD" {
    var r = try testParseWithArena("CREATE ROLE user1 LOGIN PASSWORD 'secret123'");
    defer r.deinit();
    try std.testing.expect(r.stmt == .create_role);
    const role = r.stmt.create_role;
    try std.testing.expectEqualStrings("user1", role.name);
    try std.testing.expectEqual(true, role.options.login.?);
    try std.testing.expectEqualStrings("'secret123'", role.options.password.?);
}

test "parse CREATE ROLE with VALID UNTIL" {
    var r = try testParseWithArena("CREATE ROLE temp_user LOGIN VALID UNTIL '2025-12-31'");
    defer r.deinit();
    try std.testing.expect(r.stmt == .create_role);
    const role = r.stmt.create_role;
    try std.testing.expectEqualStrings("temp_user", role.name);
    try std.testing.expectEqual(true, role.options.login.?);
    try std.testing.expectEqualStrings("'2025-12-31'", role.options.valid_until.?);
}

test "parse CREATE ROLE with all options" {
    var r = try testParseWithArena("CREATE ROLE full_role WITH LOGIN SUPERUSER CREATEDB CREATEROLE INHERIT PASSWORD 'pass' VALID UNTIL '2026-01-01'");
    defer r.deinit();
    try std.testing.expect(r.stmt == .create_role);
    const role = r.stmt.create_role;
    try std.testing.expectEqualStrings("full_role", role.name);
    try std.testing.expectEqual(true, role.options.login.?);
    try std.testing.expectEqual(true, role.options.superuser.?);
    try std.testing.expectEqual(true, role.options.createdb.?);
    try std.testing.expectEqual(true, role.options.createrole.?);
    try std.testing.expectEqual(true, role.options.inherit.?);
    try std.testing.expectEqualStrings("'pass'", role.options.password.?);
    try std.testing.expectEqualStrings("'2026-01-01'", role.options.valid_until.?);
}

test "parse CREATE ROLE with negative options" {
    var r = try testParseWithArena("CREATE ROLE restricted WITH NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT");
    defer r.deinit();
    try std.testing.expect(r.stmt == .create_role);
    const role = r.stmt.create_role;
    try std.testing.expectEqualStrings("restricted", role.name);
    try std.testing.expectEqual(false, role.options.login.?);
    try std.testing.expectEqual(false, role.options.superuser.?);
    try std.testing.expectEqual(false, role.options.createdb.?);
    try std.testing.expectEqual(false, role.options.createrole.?);
    try std.testing.expectEqual(false, role.options.inherit.?);
}

test "parse CREATE OR REPLACE ROLE" {
    var r = try testParseWithArena("CREATE OR REPLACE ROLE admin LOGIN");
    defer r.deinit();
    try std.testing.expect(r.stmt == .create_role);
    const role = r.stmt.create_role;
    try std.testing.expectEqualStrings("admin", role.name);
    try std.testing.expect(role.or_replace);
    try std.testing.expectEqual(true, role.options.login.?);
}

test "parse DROP ROLE" {
    var r = try testParseWithArena("DROP ROLE old_user");
    defer r.deinit();
    try std.testing.expect(r.stmt == .drop_role);
    const role = r.stmt.drop_role;
    try std.testing.expectEqualStrings("old_user", role.name);
    try std.testing.expect(!role.if_exists);
}

test "parse DROP ROLE IF EXISTS" {
    var r = try testParseWithArena("DROP ROLE IF EXISTS maybe_role");
    defer r.deinit();
    try std.testing.expect(r.stmt == .drop_role);
    const role = r.stmt.drop_role;
    try std.testing.expectEqualStrings("maybe_role", role.name);
    try std.testing.expect(role.if_exists);
}

test "parse ALTER ROLE basic" {
    var r = try testParseWithArena("ALTER ROLE user1 WITH LOGIN");
    defer r.deinit();
    try std.testing.expect(r.stmt == .alter_role);
    const role = r.stmt.alter_role;
    try std.testing.expectEqualStrings("user1", role.name);
    try std.testing.expectEqual(true, role.options.login.?);
}

test "parse ALTER ROLE change PASSWORD" {
    var r = try testParseWithArena("ALTER ROLE app_user PASSWORD 'new_secret'");
    defer r.deinit();
    try std.testing.expect(r.stmt == .alter_role);
    const role = r.stmt.alter_role;
    try std.testing.expectEqualStrings("app_user", role.name);
    try std.testing.expectEqualStrings("'new_secret'", role.options.password.?);
}

test "parse ALTER ROLE multiple options" {
    var r = try testParseWithArena("ALTER ROLE user2 WITH NOLOGIN NOSUPERUSER");
    defer r.deinit();
    try std.testing.expect(r.stmt == .alter_role);
    const role = r.stmt.alter_role;
    try std.testing.expectEqualStrings("user2", role.name);
    try std.testing.expectEqual(false, role.options.login.?);
    try std.testing.expectEqual(false, role.options.superuser.?);
}

test "parse GRANT SELECT on table" {
    var r = try testParseWithArena("GRANT SELECT ON users TO alice");
    defer r.deinit();
    try std.testing.expect(r.stmt == .grant);
    const grant = r.stmt.grant;
    try std.testing.expectEqual(@as(usize, 1), grant.privileges.len);
    try std.testing.expectEqual(ast.Privilege.select, grant.privileges[0]);
    try std.testing.expectEqual(ast.ObjectType.table, grant.object_type);
    try std.testing.expectEqualStrings("users", grant.object_name);
    try std.testing.expectEqualStrings("alice", grant.grantee);
    try std.testing.expectEqual(false, grant.with_grant_option);
}

test "parse GRANT ALL PRIVILEGES" {
    var r = try testParseWithArena("GRANT ALL PRIVILEGES ON TABLE employees TO bob");
    defer r.deinit();
    try std.testing.expect(r.stmt == .grant);
    const grant = r.stmt.grant;
    try std.testing.expectEqual(@as(usize, 1), grant.privileges.len);
    try std.testing.expectEqual(ast.Privilege.all, grant.privileges[0]);
    try std.testing.expectEqual(ast.ObjectType.table, grant.object_type);
    try std.testing.expectEqualStrings("employees", grant.object_name);
    try std.testing.expectEqualStrings("bob", grant.grantee);
}

test "parse GRANT multiple privileges" {
    var r = try testParseWithArena("GRANT SELECT, INSERT, UPDATE ON orders TO clerk");
    defer r.deinit();
    try std.testing.expect(r.stmt == .grant);
    const grant = r.stmt.grant;
    try std.testing.expectEqual(@as(usize, 3), grant.privileges.len);
    try std.testing.expectEqual(ast.Privilege.select, grant.privileges[0]);
    try std.testing.expectEqual(ast.Privilege.insert, grant.privileges[1]);
    try std.testing.expectEqual(ast.Privilege.update, grant.privileges[2]);
}

test "parse GRANT with grant option" {
    var r = try testParseWithArena("GRANT ALL ON mydb TO admin WITH GRANT OPTION");
    defer r.deinit();
    try std.testing.expect(r.stmt == .grant);
    const grant = r.stmt.grant;
    try std.testing.expectEqual(ast.Privilege.all, grant.privileges[0]);
    try std.testing.expectEqualStrings("mydb", grant.object_name);
    try std.testing.expectEqualStrings("admin", grant.grantee);
    try std.testing.expectEqual(true, grant.with_grant_option);
}

test "parse REVOKE SELECT" {
    var r = try testParseWithArena("REVOKE SELECT ON products FROM charlie");
    defer r.deinit();
    try std.testing.expect(r.stmt == .revoke);
    const revoke = r.stmt.revoke;
    try std.testing.expectEqual(@as(usize, 1), revoke.privileges.len);
    try std.testing.expectEqual(ast.Privilege.select, revoke.privileges[0]);
    try std.testing.expectEqual(ast.ObjectType.table, revoke.object_type);
    try std.testing.expectEqualStrings("products", revoke.object_name);
    try std.testing.expectEqualStrings("charlie", revoke.grantee);
}

test "parse REVOKE ALL PRIVILEGES" {
    var r = try testParseWithArena("REVOKE ALL PRIVILEGES ON TABLE inventory FROM guest");
    defer r.deinit();
    try std.testing.expect(r.stmt == .revoke);
    const revoke = r.stmt.revoke;
    try std.testing.expectEqual(@as(usize, 1), revoke.privileges.len);
    try std.testing.expectEqual(ast.Privilege.all, revoke.privileges[0]);
    try std.testing.expectEqualStrings("inventory", revoke.object_name);
    try std.testing.expectEqualStrings("guest", revoke.grantee);
}

test "parse REVOKE multiple privileges" {
    var r = try testParseWithArena("REVOKE INSERT, UPDATE, DELETE ON logs FROM intern");
    defer r.deinit();
    try std.testing.expect(r.stmt == .revoke);
    const revoke = r.stmt.revoke;
    try std.testing.expectEqual(@as(usize, 3), revoke.privileges.len);
    try std.testing.expectEqual(ast.Privilege.insert, revoke.privileges[0]);
    try std.testing.expectEqual(ast.Privilege.update, revoke.privileges[1]);
    try std.testing.expectEqual(ast.Privilege.delete, revoke.privileges[2]);
}

// ── GRANT/REVOKE Role Membership ─────────────────────────────────

test "parse GRANT role to single member" {
    var r = try testParseWithArena("GRANT manager TO alice");
    defer r.deinit();
    try std.testing.expect(r.stmt == .grant_role);
    const grant = r.stmt.grant_role;
    try std.testing.expectEqualStrings("manager", grant.role);
    try std.testing.expectEqual(@as(usize, 1), grant.members.len);
    try std.testing.expectEqualStrings("alice", grant.members[0]);
    try std.testing.expectEqual(false, grant.with_admin_option);
}

test "parse GRANT role to multiple members" {
    var r = try testParseWithArena("GRANT admin TO alice, bob, charlie");
    defer r.deinit();
    try std.testing.expect(r.stmt == .grant_role);
    const grant = r.stmt.grant_role;
    try std.testing.expectEqualStrings("admin", grant.role);
    try std.testing.expectEqual(@as(usize, 3), grant.members.len);
    try std.testing.expectEqualStrings("alice", grant.members[0]);
    try std.testing.expectEqualStrings("bob", grant.members[1]);
    try std.testing.expectEqualStrings("charlie", grant.members[2]);
    try std.testing.expectEqual(false, grant.with_admin_option);
}

test "parse GRANT role WITH ADMIN OPTION" {
    var r = try testParseWithArena("GRANT superuser TO alice WITH ADMIN OPTION");
    defer r.deinit();
    try std.testing.expect(r.stmt == .grant_role);
    const grant = r.stmt.grant_role;
    try std.testing.expectEqualStrings("superuser", grant.role);
    try std.testing.expectEqual(@as(usize, 1), grant.members.len);
    try std.testing.expectEqualStrings("alice", grant.members[0]);
    try std.testing.expectEqual(true, grant.with_admin_option);
}

test "parse GRANT role multiple members WITH ADMIN OPTION" {
    var r = try testParseWithArena("GRANT dba TO alice, bob WITH ADMIN OPTION");
    defer r.deinit();
    try std.testing.expect(r.stmt == .grant_role);
    const grant = r.stmt.grant_role;
    try std.testing.expectEqualStrings("dba", grant.role);
    try std.testing.expectEqual(@as(usize, 2), grant.members.len);
    try std.testing.expectEqualStrings("alice", grant.members[0]);
    try std.testing.expectEqualStrings("bob", grant.members[1]);
    try std.testing.expectEqual(true, grant.with_admin_option);
}

test "parse REVOKE role from single member" {
    var r = try testParseWithArena("REVOKE manager FROM alice");
    defer r.deinit();
    try std.testing.expect(r.stmt == .revoke_role);
    const revoke = r.stmt.revoke_role;
    try std.testing.expectEqualStrings("manager", revoke.role);
    try std.testing.expectEqual(@as(usize, 1), revoke.members.len);
    try std.testing.expectEqualStrings("alice", revoke.members[0]);
}

test "parse REVOKE role from multiple members" {
    var r = try testParseWithArena("REVOKE admin FROM alice, bob, charlie");
    defer r.deinit();
    try std.testing.expect(r.stmt == .revoke_role);
    const revoke = r.stmt.revoke_role;
    try std.testing.expectEqualStrings("admin", revoke.role);
    try std.testing.expectEqual(@as(usize, 3), revoke.members.len);
    try std.testing.expectEqualStrings("alice", revoke.members[0]);
    try std.testing.expectEqualStrings("bob", revoke.members[1]);
    try std.testing.expectEqualStrings("charlie", revoke.members[2]);
}

// ── Row-Level Security Tests ──────────────────────────────────

test "parse CREATE POLICY simple" {
    var r = try testParseWithArena("CREATE POLICY policy1 ON users");
    defer r.deinit();
    try std.testing.expect(r.stmt == .create_policy);
    const policy = r.stmt.create_policy;
    try std.testing.expectEqualStrings("policy1", policy.policy_name);
    try std.testing.expectEqualStrings("users", policy.table_name);
    try std.testing.expectEqual(ast.PolicyType.permissive, policy.policy_type);
    try std.testing.expectEqual(ast.PolicyCommand.all, policy.command);
    try std.testing.expect(policy.using_expr == null);
    try std.testing.expect(policy.with_check_expr == null);
}

test "parse CREATE POLICY with AS PERMISSIVE" {
    var r = try testParseWithArena("CREATE POLICY p1 ON t1 AS PERMISSIVE");
    defer r.deinit();
    try std.testing.expect(r.stmt == .create_policy);
    const policy = r.stmt.create_policy;
    try std.testing.expectEqual(ast.PolicyType.permissive, policy.policy_type);
}

test "parse CREATE POLICY with AS RESTRICTIVE" {
    var r = try testParseWithArena("CREATE POLICY p1 ON t1 AS RESTRICTIVE");
    defer r.deinit();
    try std.testing.expect(r.stmt == .create_policy);
    const policy = r.stmt.create_policy;
    try std.testing.expectEqual(ast.PolicyType.restrictive, policy.policy_type);
}

test "parse CREATE POLICY FOR SELECT" {
    var r = try testParseWithArena("CREATE POLICY p1 ON t1 FOR SELECT");
    defer r.deinit();
    try std.testing.expect(r.stmt == .create_policy);
    const policy = r.stmt.create_policy;
    try std.testing.expectEqual(ast.PolicyCommand.select, policy.command);
}

test "parse CREATE POLICY FOR INSERT" {
    var r = try testParseWithArena("CREATE POLICY p1 ON t1 FOR INSERT");
    defer r.deinit();
    try std.testing.expect(r.stmt == .create_policy);
    const policy = r.stmt.create_policy;
    try std.testing.expectEqual(ast.PolicyCommand.insert, policy.command);
}

test "parse CREATE POLICY FOR UPDATE" {
    var r = try testParseWithArena("CREATE POLICY p1 ON t1 FOR UPDATE");
    defer r.deinit();
    try std.testing.expect(r.stmt == .create_policy);
    const policy = r.stmt.create_policy;
    try std.testing.expectEqual(ast.PolicyCommand.update, policy.command);
}

test "parse CREATE POLICY FOR DELETE" {
    var r = try testParseWithArena("CREATE POLICY p1 ON t1 FOR DELETE");
    defer r.deinit();
    try std.testing.expect(r.stmt == .create_policy);
    const policy = r.stmt.create_policy;
    try std.testing.expectEqual(ast.PolicyCommand.delete, policy.command);
}

test "parse CREATE POLICY with USING" {
    var r = try testParseWithArena("CREATE POLICY p1 ON t1 USING (user_id = 123)");
    defer r.deinit();
    try std.testing.expect(r.stmt == .create_policy);
    const policy = r.stmt.create_policy;
    try std.testing.expect(policy.using_expr != null);
    try std.testing.expect(policy.with_check_expr == null);
}

test "parse CREATE POLICY with WITH CHECK" {
    var r = try testParseWithArena("CREATE POLICY p1 ON t1 WITH CHECK (status = 'active')");
    defer r.deinit();
    try std.testing.expect(r.stmt == .create_policy);
    const policy = r.stmt.create_policy;
    try std.testing.expect(policy.using_expr == null);
    try std.testing.expect(policy.with_check_expr != null);
}

test "parse CREATE POLICY with USING and WITH CHECK" {
    var r = try testParseWithArena("CREATE POLICY p1 ON t1 USING (user_id = 1) WITH CHECK (role = 'admin')");
    defer r.deinit();
    try std.testing.expect(r.stmt == .create_policy);
    const policy = r.stmt.create_policy;
    try std.testing.expect(policy.using_expr != null);
    try std.testing.expect(policy.with_check_expr != null);
}

test "parse CREATE POLICY full syntax" {
    var r = try testParseWithArena("CREATE POLICY admin_policy ON documents AS RESTRICTIVE FOR SELECT USING (owner_id = current_user_id())");
    defer r.deinit();
    try std.testing.expect(r.stmt == .create_policy);
    const policy = r.stmt.create_policy;
    try std.testing.expectEqualStrings("admin_policy", policy.policy_name);
    try std.testing.expectEqualStrings("documents", policy.table_name);
    try std.testing.expectEqual(ast.PolicyType.restrictive, policy.policy_type);
    try std.testing.expectEqual(ast.PolicyCommand.select, policy.command);
    try std.testing.expect(policy.using_expr != null);
}

test "parse DROP POLICY simple" {
    var r = try testParseWithArena("DROP POLICY policy1 ON users");
    defer r.deinit();
    try std.testing.expect(r.stmt == .drop_policy);
    const drop = r.stmt.drop_policy;
    try std.testing.expectEqualStrings("policy1", drop.policy_name);
    try std.testing.expectEqualStrings("users", drop.table_name);
    try std.testing.expect(!drop.if_exists);
}

test "parse DROP POLICY IF EXISTS" {
    var r = try testParseWithArena("DROP POLICY IF EXISTS old_policy ON accounts");
    defer r.deinit();
    try std.testing.expect(r.stmt == .drop_policy);
    const drop = r.stmt.drop_policy;
    try std.testing.expectEqualStrings("old_policy", drop.policy_name);
    try std.testing.expectEqualStrings("accounts", drop.table_name);
    try std.testing.expect(drop.if_exists);
}

test "parse ALTER TABLE ENABLE ROW LEVEL SECURITY" {
    var r = try testParseWithArena("ALTER TABLE users ENABLE ROW LEVEL SECURITY");
    defer r.deinit();
    try std.testing.expect(r.stmt == .alter_table_rls);
    const alter = r.stmt.alter_table_rls;
    try std.testing.expectEqualStrings("users", alter.table_name);
    try std.testing.expect(alter.enable);
    try std.testing.expect(!alter.force);
}

test "parse ALTER TABLE DISABLE ROW LEVEL SECURITY" {
    var r = try testParseWithArena("ALTER TABLE logs DISABLE ROW LEVEL SECURITY");
    defer r.deinit();
    try std.testing.expect(r.stmt == .alter_table_rls);
    const alter = r.stmt.alter_table_rls;
    try std.testing.expectEqualStrings("logs", alter.table_name);
    try std.testing.expect(!alter.enable);
    try std.testing.expect(!alter.force);
}

test "parse ALTER TABLE ENABLE FORCE ROW LEVEL SECURITY" {
    var r = try testParseWithArena("ALTER TABLE sensitive ENABLE FORCE ROW LEVEL SECURITY");
    defer r.deinit();
    try std.testing.expect(r.stmt == .alter_table_rls);
    const alter = r.stmt.alter_table_rls;
    try std.testing.expectEqualStrings("sensitive", alter.table_name);
    try std.testing.expect(alter.enable);
    try std.testing.expect(alter.force);
}

test "parse ALTER TABLE FORCE ROW LEVEL SECURITY" {
    var r = try testParseWithArena("ALTER TABLE audit FORCE ROW LEVEL SECURITY");
    defer r.deinit();
    try std.testing.expect(r.stmt == .alter_table_rls);
    const alter = r.stmt.alter_table_rls;
    try std.testing.expectEqualStrings("audit", alter.table_name);
    try std.testing.expect(alter.enable);
    try std.testing.expect(alter.force);
}

test "parse ALTER TABLE NO FORCE ROW LEVEL SECURITY" {
    var r = try testParseWithArena("ALTER TABLE public_data NO FORCE ROW LEVEL SECURITY");
    defer r.deinit();
    try std.testing.expect(r.stmt == .alter_table_rls);
    const alter = r.stmt.alter_table_rls;
    try std.testing.expectEqualStrings("public_data", alter.table_name);
    try std.testing.expect(!alter.enable);
    try std.testing.expect(!alter.force);
}

// ── RLS Parser Edge Cases ───────────────────────────────────────────────

test "parse CREATE POLICY with complex USING expression" {
    var r = try testParseWithArena("CREATE POLICY complex ON t USING (a > 10 AND b < 20 OR c = 'value')");
    defer r.deinit();
    try std.testing.expect(r.stmt == .create_policy);
    const policy = r.stmt.create_policy;
    try std.testing.expectEqualStrings("complex", policy.policy_name);
    try std.testing.expectEqualStrings("t", policy.table_name);
    try std.testing.expect(policy.using_expr != null);
}

test "parse CREATE POLICY with nested function calls in WITH CHECK" {
    var r = try testParseWithArena("CREATE POLICY func_check ON t WITH CHECK (validate(user_id, get_role()))");
    defer r.deinit();
    try std.testing.expect(r.stmt == .create_policy);
    const policy = r.stmt.create_policy;
    try std.testing.expect(policy.with_check_expr != null);
}

test "parse CREATE POLICY with all clauses combined" {
    var r = try testParseWithArena("CREATE POLICY full ON t AS PERMISSIVE FOR UPDATE USING (owner = me()) WITH CHECK (status IN ('active', 'pending'))");
    defer r.deinit();
    try std.testing.expect(r.stmt == .create_policy);
    const policy = r.stmt.create_policy;
    try std.testing.expectEqualStrings("full", policy.policy_name);
    try std.testing.expectEqual(ast.PolicyType.permissive, policy.policy_type);
    try std.testing.expectEqual(ast.PolicyCommand.update, policy.command);
    try std.testing.expect(policy.using_expr != null);
    try std.testing.expect(policy.with_check_expr != null);
}

test "parse CREATE POLICY minimal (only name and table)" {
    var r = try testParseWithArena("CREATE POLICY minimal ON users");
    defer r.deinit();
    try std.testing.expect(r.stmt == .create_policy);
    const policy = r.stmt.create_policy;
    try std.testing.expectEqualStrings("minimal", policy.policy_name);
    try std.testing.expectEqualStrings("users", policy.table_name);
    try std.testing.expectEqual(ast.PolicyType.permissive, policy.policy_type);
    try std.testing.expectEqual(ast.PolicyCommand.all, policy.command);
    try std.testing.expect(policy.using_expr == null);
    try std.testing.expect(policy.with_check_expr == null);
}

test "parse CREATE POLICY with quoted identifiers" {
    var r = try testParseWithArena("CREATE POLICY \"my-policy\" ON \"user-table\"");
    defer r.deinit();
    try std.testing.expect(r.stmt == .create_policy);
    const policy = r.stmt.create_policy;
    try std.testing.expectEqualStrings("my-policy", policy.policy_name);
    try std.testing.expectEqualStrings("user-table", policy.table_name);
}

test "parse DROP POLICY with quoted identifiers" {
    var r = try testParseWithArena("DROP POLICY \"old-policy\" ON \"legacy-table\"");
    defer r.deinit();
    try std.testing.expect(r.stmt == .drop_policy);
    const drop = r.stmt.drop_policy;
    try std.testing.expectEqualStrings("old-policy", drop.policy_name);
    try std.testing.expectEqualStrings("legacy-table", drop.table_name);
}

test "parse ALTER TABLE with quoted table name" {
    var r = try testParseWithArena("ALTER TABLE \"my-table\" ENABLE ROW LEVEL SECURITY");
    defer r.deinit();
    try std.testing.expect(r.stmt == .alter_table_rls);
    const alter = r.stmt.alter_table_rls;
    try std.testing.expectEqualStrings("my-table", alter.table_name);
    try std.testing.expect(alter.enable);
}

test "parse CREATE POLICY FOR INSERT with WITH CHECK (no USING)" {
    // INSERT policies typically only have WITH CHECK, not USING
    var r = try testParseWithArena("CREATE POLICY ins ON t FOR INSERT WITH CHECK (dept = 'sales')");
    defer r.deinit();
    try std.testing.expect(r.stmt == .create_policy);
    const policy = r.stmt.create_policy;
    try std.testing.expectEqual(ast.PolicyCommand.insert, policy.command);
    try std.testing.expect(policy.using_expr == null);
    try std.testing.expect(policy.with_check_expr != null);
}

test "parse CREATE POLICY with subquery in USING" {
    // DISABLED: Parser does not yet support IN (SELECT ...) syntax
    // Only IN (value_list) is currently implemented. Subquery support in IN expressions
    // requires adding a new AST node type (in_subquery) and updating parseInExpr().
    // See: src/sql/ast.zig (no in_subquery node), src/sql/parser.zig:2900 (parseInExpr)
    return error.SkipZigTest;

    // var r = try testParseWithArena("CREATE POLICY sub ON t USING (user_id IN (SELECT id FROM allowed))");
    // defer r.deinit();
    // try std.testing.expect(r.stmt == .create_policy);
    // const policy = r.stmt.create_policy;
    // try std.testing.expect(policy.using_expr != null);
}

test "parse CREATE POLICY with boolean literal in WITH CHECK" {
    var r = try testParseWithArena("CREATE POLICY always ON t WITH CHECK (true)");
    defer r.deinit();
    try std.testing.expect(r.stmt == .create_policy);
    const policy = r.stmt.create_policy;
    try std.testing.expect(policy.with_check_expr != null);
}

test "parse ANALYZE with table name" {
    var r = try testParseWithArena("ANALYZE users");
    defer r.deinit();
    try std.testing.expect(r.stmt == .analyze);
    const analyze = r.stmt.analyze;
    try std.testing.expectEqualStrings("users", analyze.table_name.?);
}

test "parse ANALYZE without table (all tables)" {
    var r = try testParseWithArena("ANALYZE");
    defer r.deinit();
    try std.testing.expect(r.stmt == .analyze);
    const analyze = r.stmt.analyze;
    try std.testing.expect(analyze.table_name == null);
}

test "parse ANALYZE case insensitive" {
    var r = try testParseWithArena("analyze products");
    defer r.deinit();
    try std.testing.expect(r.stmt == .analyze);
    try std.testing.expectEqualStrings("products", r.stmt.analyze.table_name.?);
}

test "parse REINDEX INDEX" {
    var r = try testParseWithArena("REINDEX INDEX idx_users_email");
    defer r.deinit();
    try std.testing.expect(r.stmt == .reindex);
    const reindex = r.stmt.reindex;
    try std.testing.expect(reindex == .index);
    try std.testing.expectEqualStrings("idx_users_email", reindex.index);
}

test "parse REINDEX TABLE" {
    var r = try testParseWithArena("REINDEX TABLE users");
    defer r.deinit();
    try std.testing.expect(r.stmt == .reindex);
    const reindex = r.stmt.reindex;
    try std.testing.expect(reindex == .table);
    try std.testing.expectEqualStrings("users", reindex.table);
}

test "parse REINDEX DATABASE" {
    var r = try testParseWithArena("REINDEX DATABASE");
    defer r.deinit();
    try std.testing.expect(r.stmt == .reindex);
    const reindex = r.stmt.reindex;
    try std.testing.expect(reindex == .database);
}

test "parse REINDEX case insensitive" {
    var r = try testParseWithArena("reindex index idx_test");
    defer r.deinit();
    try std.testing.expect(r.stmt == .reindex);
    try std.testing.expectEqualStrings("idx_test", r.stmt.reindex.index);
}

test "parse REINDEX missing target should fail" {
    const result = testParseWithArena("REINDEX");
    try std.testing.expectError(error.ParseFailed, result);
}

test "parse REINDEX INDEX without name should fail" {
    const result = testParseWithArena("REINDEX INDEX");
    try std.testing.expectError(error.ParseFailed, result);
}

test "parse REINDEX TABLE without name should fail" {
    const result = testParseWithArena("REINDEX TABLE");
    try std.testing.expectError(error.ParseFailed, result);
}

test "parse EXISTS subquery in WHERE" {
    var r = try testParseWithArena("SELECT * FROM users WHERE EXISTS (SELECT 1 FROM orders WHERE orders.user_id = users.id)");
    defer r.deinit();
    try std.testing.expect(r.stmt == .select);
    const select_stmt = r.stmt.select;
    try std.testing.expect(select_stmt.where != null);
    try std.testing.expect(select_stmt.where.?.* == .exists);
    try std.testing.expect(!select_stmt.where.?.exists.negated);
    // Verify subquery is properly parsed
    try std.testing.expect(select_stmt.where.?.exists.subquery.columns.len == 1);
}

test "parse NOT EXISTS subquery in WHERE" {
    var r = try testParseWithArena("SELECT * FROM users WHERE NOT EXISTS (SELECT 1 FROM orders WHERE orders.user_id = users.id)");
    defer r.deinit();
    try std.testing.expect(r.stmt == .select);
    const select_stmt = r.stmt.select;
    try std.testing.expect(select_stmt.where != null);
    try std.testing.expect(select_stmt.where.?.* == .exists);
    try std.testing.expect(select_stmt.where.?.exists.negated);
}

test "parse EXISTS with complex subquery" {
    var r = try testParseWithArena("SELECT name FROM products WHERE EXISTS (SELECT * FROM reviews WHERE reviews.product_id = products.id AND rating > 4)");
    defer r.deinit();
    try std.testing.expect(r.stmt == .select);
    const select_stmt = r.stmt.select;
    try std.testing.expect(select_stmt.where != null);
    try std.testing.expect(select_stmt.where.?.* == .exists);
    try std.testing.expect(!select_stmt.where.?.exists.negated);
}

test "parse EXISTS in SELECT list" {
    var r = try testParseWithArena("SELECT id, EXISTS (SELECT 1 FROM orders WHERE user_id = users.id) AS has_orders FROM users");
    defer r.deinit();
    try std.testing.expect(r.stmt == .select);
    const select_stmt = r.stmt.select;
    try std.testing.expect(select_stmt.columns.len == 2);
    try std.testing.expect(select_stmt.columns[1] == .expr);
    try std.testing.expect(select_stmt.columns[1].expr.value.* == .exists);
    try std.testing.expectEqualStrings("has_orders", select_stmt.columns[1].expr.alias.?);
}

test "parse multiple EXISTS in WHERE with AND" {
    var r = try testParseWithArena("SELECT * FROM users WHERE EXISTS (SELECT 1 FROM orders) AND NOT EXISTS (SELECT 1 FROM reviews)");
    defer r.deinit();
    try std.testing.expect(r.stmt == .select);
    const select_stmt = r.stmt.select;
    try std.testing.expect(select_stmt.where != null);
    try std.testing.expect(select_stmt.where.?.* == .binary_op);
}

// ── pg_stat_activity (monitoring views) tests ──────────────────────

test "parse SELECT * FROM pg_stat_activity" {
    var r = try testParseWithArena("SELECT * FROM pg_stat_activity");
    defer r.deinit();
    try std.testing.expect(r.stmt == .select);
    const select_stmt = r.stmt.select;
    try std.testing.expect(select_stmt.from != null);
    try std.testing.expectEqualStrings("pg_stat_activity", select_stmt.from.?.table_name.name);
    try std.testing.expectEqual(@as(usize, 1), select_stmt.columns.len);
    try std.testing.expect(select_stmt.columns[0] == .all_columns);
}

test "parse SELECT specific columns FROM pg_stat_activity" {
    var r = try testParseWithArena("SELECT pid, usename, query FROM pg_stat_activity");
    defer r.deinit();
    try std.testing.expect(r.stmt == .select);
    const select_stmt = r.stmt.select;
    try std.testing.expectEqual(@as(usize, 3), select_stmt.columns.len);
    try std.testing.expectEqualStrings("pid", select_stmt.columns[0].expr.value.column_ref.name);
    try std.testing.expectEqualStrings("usename", select_stmt.columns[1].expr.value.column_ref.name);
    try std.testing.expectEqualStrings("query", select_stmt.columns[2].expr.value.column_ref.name);
}

test "parse SELECT FROM pg_stat_activity with WHERE state = 'active'" {
    var r = try testParseWithArena("SELECT * FROM pg_stat_activity WHERE state = 'active'");
    defer r.deinit();
    try std.testing.expect(r.stmt == .select);
    const select_stmt = r.stmt.select;
    try std.testing.expect(select_stmt.where != null);
    try std.testing.expectEqualStrings("pg_stat_activity", select_stmt.from.?.table_name.name);
    // WHERE clause is binary_op (state = 'active')
    try std.testing.expect(select_stmt.where.?.* == .binary_op);
}

test "parse SELECT FROM pg_stat_activity with WHERE usename = 'postgres'" {
    var r = try testParseWithArena("SELECT pid, query FROM pg_stat_activity WHERE usename = 'postgres'");
    defer r.deinit();
    try std.testing.expect(r.stmt == .select);
    const select_stmt = r.stmt.select;
    try std.testing.expectEqual(@as(usize, 2), select_stmt.columns.len);
    try std.testing.expect(select_stmt.where != null);
    try std.testing.expect(select_stmt.where.?.* == .binary_op);
}

test "parse SELECT FROM pg_stat_activity with complex WHERE" {
    var r = try testParseWithArena(
        "SELECT * FROM pg_stat_activity WHERE state = 'active' AND query IS NOT NULL"
    );
    defer r.deinit();
    try std.testing.expect(r.stmt == .select);
    const select_stmt = r.stmt.select;
    try std.testing.expect(select_stmt.where != null);
    try std.testing.expect(select_stmt.where.?.* == .binary_op);
}

test "parse SELECT FROM pg_stat_activity ORDER BY query_start" {
    var r = try testParseWithArena("SELECT * FROM pg_stat_activity ORDER BY query_start DESC");
    defer r.deinit();
    try std.testing.expect(r.stmt == .select);
    const select_stmt = r.stmt.select;
    try std.testing.expectEqual(@as(usize, 1), select_stmt.order_by.len);
}

test "parse SELECT FROM pg_stat_activity with LIMIT" {
    var r = try testParseWithArena("SELECT * FROM pg_stat_activity LIMIT 10");
    defer r.deinit();
    try std.testing.expect(r.stmt == .select);
    const select_stmt = r.stmt.select;
    try std.testing.expect(select_stmt.limit != null);
}

test "parse SELECT pid AS connection_id FROM pg_stat_activity" {
    var r = try testParseWithArena("SELECT pid AS connection_id, state AS status FROM pg_stat_activity");
    defer r.deinit();
    try std.testing.expect(r.stmt == .select);
    const select_stmt = r.stmt.select;
    try std.testing.expectEqual(@as(usize, 2), select_stmt.columns.len);
    try std.testing.expectEqualStrings("connection_id", select_stmt.columns[0].expr.alias.?);
    try std.testing.expectEqualStrings("status", select_stmt.columns[1].expr.alias.?);
}

// ── pg_locks (lock monitoring view) tests ──────────────────────

test "parse SELECT * FROM pg_locks" {
    var r = try testParseWithArena("SELECT * FROM pg_locks");
    defer r.deinit();
    try std.testing.expect(r.stmt == .select);
    const select_stmt = r.stmt.select;
    try std.testing.expect(select_stmt.from != null);
    try std.testing.expectEqualStrings("pg_locks", select_stmt.from.?.table_name.name);
    try std.testing.expectEqual(@as(usize, 1), select_stmt.columns.len);
    try std.testing.expect(select_stmt.columns[0] == .all_columns);
}

test "parse SELECT specific columns FROM pg_locks" {
    var r = try testParseWithArena("SELECT locktype, mode, pid FROM pg_locks");
    defer r.deinit();
    try std.testing.expect(r.stmt == .select);
    const select_stmt = r.stmt.select;
    try std.testing.expectEqual(@as(usize, 3), select_stmt.columns.len);
    try std.testing.expectEqualStrings("locktype", select_stmt.columns[0].expr.value.column_ref.name);
    try std.testing.expectEqualStrings("mode", select_stmt.columns[1].expr.value.column_ref.name);
    try std.testing.expectEqualStrings("pid", select_stmt.columns[2].expr.value.column_ref.name);
}

test "parse SELECT FROM pg_locks with WHERE locktype = 'relation'" {
    var r = try testParseWithArena("SELECT * FROM pg_locks WHERE locktype = 'relation'");
    defer r.deinit();
    try std.testing.expect(r.stmt == .select);
    const select_stmt = r.stmt.select;
    try std.testing.expect(select_stmt.where != null);
    try std.testing.expectEqualStrings("pg_locks", select_stmt.from.?.table_name.name);
    try std.testing.expect(select_stmt.where.?.* == .binary_op);
}

test "parse SELECT FROM pg_locks ORDER BY pid" {
    var r = try testParseWithArena("SELECT * FROM pg_locks ORDER BY pid");
    defer r.deinit();
    try std.testing.expect(r.stmt == .select);
    const select_stmt = r.stmt.select;
    try std.testing.expectEqual(@as(usize, 1), select_stmt.order_by.len);
}

test "parse SELECT FROM pg_locks with LIMIT" {
    var r = try testParseWithArena("SELECT * FROM pg_locks LIMIT 5");
    defer r.deinit();
    try std.testing.expect(r.stmt == .select);
    const select_stmt = r.stmt.select;
    try std.testing.expect(select_stmt.limit != null);
}

// ── Configuration System Parser Tests ────────────────────────────────

test "parse SET with equals syntax" {
    var r = try testParseWithArena("SET work_mem = '4MB'");
    defer r.deinit();

    switch (r.stmt) {
        .set => |s| {
            try std.testing.expectEqualStrings("work_mem", s.parameter);
            try std.testing.expectEqualStrings("4MB", s.value);
        },
        else => return error.TestUnexpectedStmtType,
    }
}

test "parse SET with TO syntax" {
    var r = try testParseWithArena("SET search_path TO public");
    defer r.deinit();

    switch (r.stmt) {
        .set => |s| {
            try std.testing.expectEqualStrings("search_path", s.parameter);
            try std.testing.expectEqualStrings("public", s.value);
        },
        else => return error.TestUnexpectedStmtType,
    }
}

test "parse SET integer parameter" {
    var r = try testParseWithArena("SET max_connections = 100");
    defer r.deinit();

    switch (r.stmt) {
        .set => |s| {
            try std.testing.expectEqualStrings("max_connections", s.parameter);
            try std.testing.expectEqualStrings("100", s.value);
        },
        else => return error.TestUnexpectedStmtType,
    }
}

test "parse SET with quoted value" {
    var r = try testParseWithArena("SET application_name = 'my_app'");
    defer r.deinit();

    switch (r.stmt) {
        .set => |s| {
            try std.testing.expectEqualStrings("application_name", s.parameter);
            try std.testing.expectEqualStrings("my_app", s.value);
        },
        else => return error.TestUnexpectedStmtType,
    }
}

test "parse SHOW single parameter" {
    var r = try testParseWithArena("SHOW work_mem");
    defer r.deinit();

    switch (r.stmt) {
        .show => |s| {
            try std.testing.expect(s.parameter != null);
            try std.testing.expectEqualStrings("work_mem", s.parameter.?);
        },
        else => return error.TestUnexpectedStmtType,
    }
}

test "parse SHOW ALL" {
    var r = try testParseWithArena("SHOW ALL");
    defer r.deinit();

    switch (r.stmt) {
        .show => |s| {
            try std.testing.expect(s.parameter == null);
        },
        else => return error.TestUnexpectedStmtType,
    }
}

test "parse RESET single parameter" {
    var r = try testParseWithArena("RESET work_mem");
    defer r.deinit();

    switch (r.stmt) {
        .reset => |s| {
            try std.testing.expect(s.parameter != null);
            try std.testing.expectEqualStrings("work_mem", s.parameter.?);
        },
        else => return error.TestUnexpectedStmtType,
    }
}

test "parse RESET ALL" {
    var r = try testParseWithArena("RESET ALL");
    defer r.deinit();

    switch (r.stmt) {
        .reset => |s| {
            try std.testing.expect(s.parameter == null);
        },
        else => return error.TestUnexpectedStmtType,
    }
}

test "parse SET without value fails" {
    const source = "SET work_mem";
    var ast_arena = ast.AstArena.init(std.testing.allocator);
    defer ast_arena.deinit();
    var parser = try Parser.init(std.testing.allocator, source, &ast_arena);
    defer parser.deinit();

    const result = parser.parseStatement();
    try std.testing.expectError(error.ParseFailed, result);
}

test "parse SHOW without parameter fails" {
    const source = "SHOW";
    var ast_arena = ast.AstArena.init(std.testing.allocator);
    defer ast_arena.deinit();
    var parser = try Parser.init(std.testing.allocator, source, &ast_arena);
    defer parser.deinit();

    const result = parser.parseStatement();
    try std.testing.expectError(error.ParseFailed, result);
}

test "parse RESET without parameter fails" {
    const source = "RESET";
    var ast_arena = ast.AstArena.init(std.testing.allocator);
    defer ast_arena.deinit();
    var parser = try Parser.init(std.testing.allocator, source, &ast_arena);
    defer parser.deinit();

    const result = parser.parseStatement();
    try std.testing.expectError(error.ParseFailed, result);
}

// ── Extended SET/SHOW/RESET Parser Tests ──────────────────────────────

test "parse SET with size value (4MB)" {
    var r = try testParseWithArena("SET work_mem = '4MB'");
    defer r.deinit();

    switch (r.stmt) {
        .set => |s| {
            try std.testing.expectEqualStrings("work_mem", s.parameter);
            try std.testing.expectEqualStrings("4MB", s.value);
        },
        else => return error.TestUnexpectedStmtType,
    }
}

test "parse SET with size value (1GB)" {
    var r = try testParseWithArena("SET shared_buffers = '1GB'");
    defer r.deinit();

    switch (r.stmt) {
        .set => |s| {
            try std.testing.expectEqualStrings("shared_buffers", s.parameter);
            try std.testing.expectEqualStrings("1GB", s.value);
        },
        else => return error.TestUnexpectedStmtType,
    }
}

test "parse SET max_connections to integer" {
    var r = try testParseWithArena("SET max_connections = 200");
    defer r.deinit();

    switch (r.stmt) {
        .set => |s| {
            try std.testing.expectEqualStrings("max_connections", s.parameter);
            try std.testing.expectEqualStrings("200", s.value);
        },
        else => return error.TestUnexpectedStmtType,
    }
}

test "parse SET application_name to string" {
    var r = try testParseWithArena("SET application_name = 'test'");
    defer r.deinit();

    switch (r.stmt) {
        .set => |s| {
            try std.testing.expectEqualStrings("application_name", s.parameter);
            try std.testing.expectEqualStrings("test", s.value);
        },
        else => return error.TestUnexpectedStmtType,
    }
}

test "parse SET search_path to identifier" {
    var r = try testParseWithArena("SET search_path = public");
    defer r.deinit();

    switch (r.stmt) {
        .set => |s| {
            try std.testing.expectEqualStrings("search_path", s.parameter);
            try std.testing.expectEqualStrings("public", s.value);
        },
        else => return error.TestUnexpectedStmtType,
    }
}

test "parse SET search_path with quoted value" {
    var r = try testParseWithArena("SET search_path = 'public, private'");
    defer r.deinit();

    switch (r.stmt) {
        .set => |s| {
            try std.testing.expectEqualStrings("search_path", s.parameter);
            try std.testing.expectEqualStrings("public, private", s.value);
        },
        else => return error.TestUnexpectedStmtType,
    }
}

test "parse SET with unquoted comma-separated value" {
    var r = try testParseWithArena("SET search_path TO public");
    defer r.deinit();

    switch (r.stmt) {
        .set => |s| {
            try std.testing.expectEqualStrings("search_path", s.parameter);
            try std.testing.expectEqualStrings("public", s.value);
        },
        else => return error.TestUnexpectedStmtType,
    }
}

test "parse SET parameter with zero integer value" {
    var r = try testParseWithArena("SET statement_timeout = 0");
    defer r.deinit();

    switch (r.stmt) {
        .set => |s| {
            try std.testing.expectEqualStrings("statement_timeout", s.parameter);
            try std.testing.expectEqualStrings("0", s.value);
        },
        else => return error.TestUnexpectedStmtType,
    }
}

test "parse SET parameter with large integer value" {
    var r = try testParseWithArena("SET effective_cache_size = 2147483647");
    defer r.deinit();

    switch (r.stmt) {
        .set => |s| {
            try std.testing.expectEqualStrings("effective_cache_size", s.parameter);
            try std.testing.expectEqualStrings("2147483647", s.value);
        },
        else => return error.TestUnexpectedStmtType,
    }
}

test "parse SET with empty string value" {
    var r = try testParseWithArena("SET datestyle = ''");
    defer r.deinit();

    switch (r.stmt) {
        .set => |s| {
            try std.testing.expectEqualStrings("datestyle", s.parameter);
            try std.testing.expectEqualStrings("", s.value);
        },
        else => return error.TestUnexpectedStmtType,
    }
}

test "parse SHOW work_mem" {
    var r = try testParseWithArena("SHOW work_mem");
    defer r.deinit();

    switch (r.stmt) {
        .show => |s| {
            try std.testing.expect(s.parameter != null);
            try std.testing.expectEqualStrings("work_mem", s.parameter.?);
        },
        else => return error.TestUnexpectedStmtType,
    }
}

test "parse SHOW max_connections" {
    var r = try testParseWithArena("SHOW max_connections");
    defer r.deinit();

    switch (r.stmt) {
        .show => |s| {
            try std.testing.expect(s.parameter != null);
            try std.testing.expectEqualStrings("max_connections", s.parameter.?);
        },
        else => return error.TestUnexpectedStmtType,
    }
}

test "parse SHOW search_path" {
    var r = try testParseWithArena("SHOW search_path");
    defer r.deinit();

    switch (r.stmt) {
        .show => |s| {
            try std.testing.expect(s.parameter != null);
            try std.testing.expectEqualStrings("search_path", s.parameter.?);
        },
        else => return error.TestUnexpectedStmtType,
    }
}

test "parse SHOW application_name" {
    var r = try testParseWithArena("SHOW application_name");
    defer r.deinit();

    switch (r.stmt) {
        .show => |s| {
            try std.testing.expect(s.parameter != null);
            try std.testing.expectEqualStrings("application_name", s.parameter.?);
        },
        else => return error.TestUnexpectedStmtType,
    }
}

test "parse SHOW ALL returns null parameter" {
    var r = try testParseWithArena("SHOW ALL");
    defer r.deinit();

    switch (r.stmt) {
        .show => |s| {
            try std.testing.expect(s.parameter == null);
        },
        else => return error.TestUnexpectedStmtType,
    }
}

test "parse RESET work_mem" {
    var r = try testParseWithArena("RESET work_mem");
    defer r.deinit();

    switch (r.stmt) {
        .reset => |s| {
            try std.testing.expect(s.parameter != null);
            try std.testing.expectEqualStrings("work_mem", s.parameter.?);
        },
        else => return error.TestUnexpectedStmtType,
    }
}

test "parse RESET max_connections" {
    var r = try testParseWithArena("RESET max_connections");
    defer r.deinit();

    switch (r.stmt) {
        .reset => |s| {
            try std.testing.expect(s.parameter != null);
            try std.testing.expectEqualStrings("max_connections", s.parameter.?);
        },
        else => return error.TestUnexpectedStmtType,
    }
}

test "parse RESET search_path" {
    var r = try testParseWithArena("RESET search_path");
    defer r.deinit();

    switch (r.stmt) {
        .reset => |s| {
            try std.testing.expect(s.parameter != null);
            try std.testing.expectEqualStrings("search_path", s.parameter.?);
        },
        else => return error.TestUnexpectedStmtType,
    }
}

test "parse RESET application_name" {
    var r = try testParseWithArena("RESET application_name");
    defer r.deinit();

    switch (r.stmt) {
        .reset => |s| {
            try std.testing.expect(s.parameter != null);
            try std.testing.expectEqualStrings("application_name", s.parameter.?);
        },
        else => return error.TestUnexpectedStmtType,
    }
}

test "parse RESET ALL returns null parameter" {
    var r = try testParseWithArena("RESET ALL");
    defer r.deinit();

    switch (r.stmt) {
        .reset => |s| {
            try std.testing.expect(s.parameter == null);
        },
        else => return error.TestUnexpectedStmtType,
    }
}

test "parse SET statement with TO keyword" {
    var r = try testParseWithArena("SET max_connections TO 500");
    defer r.deinit();

    switch (r.stmt) {
        .set => |s| {
            try std.testing.expectEqualStrings("max_connections", s.parameter);
            try std.testing.expectEqualStrings("500", s.value);
        },
        else => return error.TestUnexpectedStmtType,
    }
}

test "parse SET statement with TO and string value" {
    var r = try testParseWithArena("SET application_name TO 'myapp'");
    defer r.deinit();

    switch (r.stmt) {
        .set => |s| {
            try std.testing.expectEqualStrings("application_name", s.parameter);
            try std.testing.expectEqualStrings("myapp", s.value);
        },
        else => return error.TestUnexpectedStmtType,
    }
}

test "parse SET with numeric string parameter (numeric identifier)" {
    var r = try testParseWithArena("SET lc_time = 'C'");
    defer r.deinit();

    switch (r.stmt) {
        .set => |s| {
            try std.testing.expectEqualStrings("lc_time", s.parameter);
            try std.testing.expectEqualStrings("C", s.value);
        },
        else => return error.TestUnexpectedStmtType,
    }
}

test "SET parameter name is correctly stored" {
    var r = try testParseWithArena("SET enable_seqscan = off");
    defer r.deinit();

    switch (r.stmt) {
        .set => |s| {
            try std.testing.expectEqualStrings("enable_seqscan", s.parameter);
        },
        else => return error.TestUnexpectedStmtType,
    }
}

test "SHOW parameter name is correctly stored" {
    var r = try testParseWithArena("SHOW idle_in_transaction_session_timeout");
    defer r.deinit();

    switch (r.stmt) {
        .show => |s| {
            try std.testing.expect(s.parameter != null);
            try std.testing.expectEqualStrings("idle_in_transaction_session_timeout", s.parameter.?);
        },
        else => return error.TestUnexpectedStmtType,
    }
}

test "RESET parameter name is correctly stored" {
    var r = try testParseWithArena("RESET idle_in_transaction_session_timeout");
    defer r.deinit();

    switch (r.stmt) {
        .reset => |s| {
            try std.testing.expect(s.parameter != null);
            try std.testing.expectEqualStrings("idle_in_transaction_session_timeout", s.parameter.?);
        },
        else => return error.TestUnexpectedStmtType,
    }
}

test "parse SET with single-quoted numeric string" {
    var r = try testParseWithArena("SET random_page_cost = '4.0'");
    defer r.deinit();

    switch (r.stmt) {
        .set => |s| {
            try std.testing.expectEqualStrings("random_page_cost", s.parameter);
            try std.testing.expectEqualStrings("4.0", s.value);
        },
        else => return error.TestUnexpectedStmtType,
    }
}

test "parse SET with identifier value containing underscore" {
    var r = try testParseWithArena("SET log_statement = 'all'");
    defer r.deinit();

    switch (r.stmt) {
        .set => |s| {
            try std.testing.expectEqualStrings("log_statement", s.parameter);
            try std.testing.expectEqualStrings("all", s.value);
        },
        else => return error.TestUnexpectedStmtType,
    }
}

test "SHOW returns correct statement type" {
    var r = try testParseWithArena("SHOW timezone");
    defer r.deinit();

    try std.testing.expect(r.stmt == .show);
}

test "SET returns correct statement type" {
    var r = try testParseWithArena("SET timezone = 'UTC'");
    defer r.deinit();

    try std.testing.expect(r.stmt == .set);
}

test "RESET returns correct statement type" {
    var r = try testParseWithArena("RESET timezone");
    defer r.deinit();

    try std.testing.expect(r.stmt == .reset);
}

test "bind parameter in WHERE clause" {
    var r = try testParseWithArena("SELECT * FROM test WHERE id = ?");
    defer r.deinit();

    // Verify it's a SELECT statement
    try std.testing.expect(r.stmt == .select);

    // Verify WHERE clause exists
    const sel = r.stmt.select;
    try std.testing.expect(sel.where != null);

    // Verify WHERE clause is a binary op (id = ?)
    const where = sel.where.?;
    try std.testing.expect(where.* == .binary_op);

    // Right side should be bind_parameter with index 0
    const right = where.binary_op.right;
    try std.testing.expect(right.* == .bind_parameter);
    try std.testing.expectEqual(@as(u32, 0), right.bind_parameter);
}

test "multiple bind parameters in INSERT" {
    var r = try testParseWithArena("INSERT INTO test (a, b, c) VALUES (?, ?, ?)");
    defer r.deinit();

    // Verify it's an INSERT statement
    try std.testing.expect(r.stmt == .insert);

    // Verify values are bind parameters with indices 0, 1, 2
    const ins = r.stmt.insert;
    try std.testing.expectEqual(@as(usize, 1), ins.values.len); // One row
    try std.testing.expectEqual(@as(usize, 3), ins.values[0].len); // Three columns

    for (ins.values[0], 0..) |expr, i| {
        try std.testing.expect(expr.* == .bind_parameter);
        try std.testing.expectEqual(@as(u32, @intCast(i)), expr.bind_parameter);
    }
}

test "bind parameters reset between statements" {
    // Parse first statement with 2 params
    var r1 = try testParseWithArena("SELECT * FROM t WHERE a = ? AND b = ?");
    defer r1.deinit();

    const sel1 = r1.stmt.select;
    const where1 = sel1.where.?;
    // Left side of AND: a = ? (param index 0)
    const left1 = where1.binary_op.left.binary_op.right;
    try std.testing.expectEqual(@as(u32, 0), left1.bind_parameter);
    // Right side of AND: b = ? (param index 1)
    const right1 = where1.binary_op.right.binary_op.right;
    try std.testing.expectEqual(@as(u32, 1), right1.bind_parameter);

    // Parse second statement - indices should reset to 0
    var r2 = try testParseWithArena("INSERT INTO t (x) VALUES (?)");
    defer r2.deinit();

    const ins = r2.stmt.insert;
    try std.testing.expectEqual(@as(u32, 0), ins.values[0][0].bind_parameter);
}
