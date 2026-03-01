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
            .kw_select => "expected 'SELECT'",
            .kw_insert => "expected 'INSERT'",
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

        const t = self.peek();
        const stmt: ast.Stmt = switch (t.type) {
            .kw_select => .{ .select = try self.parseSelect() },
            .kw_with => .{ .select = try self.parseSelect() },
            .kw_insert => .{ .insert = try self.parseInsert() },
            .kw_update => .{ .update = try self.parseUpdate() },
            .kw_delete => .{ .delete = try self.parseDelete() },
            .kw_create => try self.parseCreate(),
            .kw_drop => try self.parseDrop(),
            .kw_begin => .{ .transaction = try self.parseBegin() },
            .kw_commit => .{ .transaction = self.parseCommit() },
            .kw_rollback => .{ .transaction = try self.parseRollback() },
            .kw_savepoint => .{ .transaction = try self.parseSavepoint() },
            .kw_release => .{ .transaction = try self.parseRelease() },
            .kw_explain => .{ .explain = try self.parseExplain() },
            .kw_vacuum => .{ .vacuum = self.parseVacuum() },
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

        // CREATE OR REPLACE VIEW
        if (self.check(.kw_or)) {
            return .{ .create_view = try self.parseCreateView(true) };
        }
        if (self.check(.kw_view)) {
            return .{ .create_view = try self.parseCreateView(false) };
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

        try self.addError(self.peek(), "expected TABLE, VIEW, or INDEX after CREATE");
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
            t == .kw_text or t == .kw_blob or t == .kw_boolean or t == .kw_varchar;
    }

    fn parseDataType(self: *Parser) ?ast.DataType {
        const t = self.peek().type;
        const dt: ?ast.DataType = switch (t) {
            .kw_integer => .type_integer,
            .kw_int => .type_int,
            .kw_real => .type_real,
            .kw_text => .type_text,
            .kw_blob => .type_blob,
            .kw_boolean => .type_boolean,
            .kw_varchar => .type_varchar,
            else => null,
        };
        if (dt != null) {
            _ = self.advance();
            if (self.match(.left_paren)) {
                _ = self.match(.integer_literal);
                _ = self.match(.right_paren);
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

        return .{
            .if_not_exists = if_not_exists,
            .unique = unique,
            .name = name,
            .table = table,
            .columns = cols.toOwnedSlice(a) catch return error.OutOfMemory,
        };
    }

    // ── DROP ──────────────────────────────────────────────────────

    fn parseDrop(self: *Parser) Error!ast.Stmt {
        _ = try self.expect(.kw_drop);
        if (self.check(.kw_table)) return .{ .drop_table = try self.parseDropTable() };
        if (self.check(.kw_view)) return .{ .drop_view = try self.parseDropView() };
        if (self.check(.kw_index)) return .{ .drop_index = try self.parseDropIndex() };
        try self.addError(self.peek(), "expected TABLE, VIEW, or INDEX after DROP");
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

        return .{
            .name = name,
            .select = select,
            .or_replace = or_replace,
            .if_not_exists = if_not_exists,
            .column_names = column_names.toOwnedSlice(a) catch return error.OutOfMemory,
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
        return .{ .begin = .{ .mode = mode } };
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
        const inner = try self.parseStatement() orelse {
            try self.addError(self.peek(), "expected statement after EXPLAIN");
            return error.ParseFailed;
        };
        const stmt_ptr = self.arena.create(ast.Stmt, inner) catch return error.OutOfMemory;
        return .{ .stmt = stmt_ptr };
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

    // ── Expression parser (Pratt / precedence climbing) ──────────

    fn parseExpr(self: *Parser, min_prec: u8) Error!*const ast.Expr {
        var left = try self.parsePrimary();

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
            .kw_case => return self.parseCaseExpr(),
            .kw_cast => return self.parseCastExpr(),
            .kw_count, .kw_sum, .kw_avg, .kw_min, .kw_max => return self.parseFunctionCall(),
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

        return self.arena.create(ast.Expr, .{ .function_call = .{
            .name = name,
            .args = args.toOwnedSlice(a) catch return error.OutOfMemory,
            .distinct = distinct,
        } }) catch return error.OutOfMemory;
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
