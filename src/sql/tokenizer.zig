const std = @import("std");

/// SQL token types
pub const TokenType = enum {
    // Literals
    integer_literal,
    float_literal,
    string_literal,
    blob_literal,

    // Identifiers
    identifier,
    quoted_identifier,

    // Operators
    plus,
    minus,
    star,
    slash,
    percent,
    equals,
    not_equals, // <> or !=
    less_than,
    greater_than,
    less_than_or_equal,
    greater_than_or_equal,
    concat, // ||
    bitwise_and,
    bitwise_or,
    bitwise_not, // ~
    left_shift, // <<
    right_shift, // >>
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
    ts_match, // @@

    // Punctuation
    left_paren,
    right_paren,
    left_bracket,
    right_bracket,
    comma,
    semicolon,
    dot,

    // Keywords — DDL
    kw_create,
    kw_table,
    kw_drop,
    kw_index,
    kw_alter,
    kw_add,
    kw_column,
    kw_rename,
    kw_to,
    kw_if,
    kw_exists,
    kw_primary,
    kw_key,
    kw_unique,
    kw_not,
    kw_null,
    kw_default,
    kw_check,
    kw_foreign,
    kw_references,
    kw_autoincrement,
    kw_constraint,
    kw_cascade,
    kw_restrict,
    kw_set,
    kw_action,
    kw_no,
    kw_on,
    kw_delete,
    kw_update,
    kw_without,
    kw_rowid,
    kw_strict,
    kw_temp,
    kw_temporary,
    kw_view,
    kw_replace,
    kw_with,
    kw_option,
    kw_recursive,
    kw_materialized,
    kw_local,
    kw_cascaded,
    kw_type,
    kw_enum,
    kw_domain,

    // Keywords — DML
    kw_select,
    kw_from,
    kw_where,
    kw_insert,
    kw_into,
    kw_values,
    kw_as,
    kw_and,
    kw_or,
    kw_in,
    kw_between,
    kw_is,
    kw_like,
    kw_glob,
    kw_order,
    kw_by,
    kw_asc,
    kw_desc,
    kw_limit,
    kw_offset,
    kw_group,
    kw_having,
    kw_distinct,
    kw_all,
    kw_any,
    kw_union,
    kw_except,
    kw_intersect,
    kw_join,
    kw_inner,
    kw_left,
    kw_right,
    kw_full,
    kw_outer,
    kw_cross,
    kw_natural,
    kw_case,
    kw_when,
    kw_then,
    kw_else,
    kw_end,
    kw_cast,

    // Keywords — Aggregate functions
    kw_count,
    kw_sum,
    kw_avg,
    kw_min,
    kw_max,

    // Keywords — Window functions
    kw_over,
    kw_partition,
    kw_rows,
    kw_range,
    kw_groups,
    kw_unbounded,
    kw_preceding,
    kw_following,
    kw_current,
    kw_row,
    kw_window,
    kw_row_number,
    kw_rank,
    kw_dense_rank,
    kw_ntile,
    kw_lag,
    kw_lead,
    kw_first_value,
    kw_last_value,
    kw_nth_value,
    kw_percent_rank,
    kw_cume_dist,

    // Keywords — Transaction
    kw_begin,
    kw_commit,
    kw_rollback,
    kw_savepoint,
    kw_release,
    kw_transaction,
    kw_deferred,
    kw_immediate,
    kw_exclusive,

    // Keywords — Utility
    kw_explain,
    kw_pragma,
    kw_vacuum,

    // Keywords — Types
    kw_integer,
    kw_int,
    kw_real,
    kw_text,
    kw_blob,
    kw_boolean,
    kw_varchar,
    kw_date,
    kw_time,
    kw_timestamp,
    kw_interval,
    kw_numeric,
    kw_decimal,
    kw_uuid,
    kw_serial,
    kw_bigserial,
    kw_array,
    kw_json,
    kw_jsonb,
    kw_tsvector,
    kw_tsquery,

    // Keywords — Values
    kw_true,
    kw_false,

    // Special
    eof,
    invalid,

    pub fn isKeyword(self: TokenType) bool {
        return @intFromEnum(self) >= @intFromEnum(TokenType.kw_create) and
            @intFromEnum(self) <= @intFromEnum(TokenType.kw_false);
    }
};

/// A single token with its location in the source
pub const Token = struct {
    type: TokenType,
    /// Byte offset into the source where this token starts
    start: u32,
    /// Length of this token in bytes
    len: u32,

    pub fn lexeme(self: Token, source: []const u8) []const u8 {
        return source[self.start..][0..self.len];
    }
};

/// Hand-written SQL tokenizer (lexer)
pub const Tokenizer = struct {
    source: []const u8,
    pos: u32,

    pub fn init(source: []const u8) Tokenizer {
        return .{
            .source = source,
            .pos = 0,
        };
    }

    pub fn next(self: *Tokenizer) Token {
        self.skipWhitespaceAndComments();

        if (self.pos >= self.source.len) {
            return .{ .type = .eof, .start = self.pos, .len = 0 };
        }

        const start = self.pos;
        const ch = self.source[self.pos];

        // Single-character tokens
        switch (ch) {
            '(' => return self.singleChar(.left_paren),
            ')' => return self.singleChar(.right_paren),
            '[' => return self.singleChar(.left_bracket),
            ']' => return self.singleChar(.right_bracket),
            ',' => return self.singleChar(.comma),
            ';' => return self.singleChar(.semicolon),
            '.' => {
                // Check for .123 float literal
                if (self.pos + 1 < self.source.len and isDigit(self.source[self.pos + 1])) {
                    return self.scanNumber();
                }
                return self.singleChar(.dot);
            },
            '+' => return self.singleChar(.plus),
            '-' => {
                self.pos += 1;
                if (self.pos < self.source.len and self.source[self.pos] == '>') {
                    self.pos += 1;
                    if (self.pos < self.source.len and self.source[self.pos] == '>') {
                        self.pos += 1;
                        return .{ .type = .json_extract_text, .start = start, .len = 3 };
                    }
                    return .{ .type = .json_extract, .start = start, .len = 2 };
                }
                return .{ .type = .minus, .start = start, .len = 1 };
            },
            '*' => return self.singleChar(.star),
            '/' => return self.singleChar(.slash),
            '%' => return self.singleChar(.percent),
            '~' => return self.singleChar(.bitwise_not),
            '&' => return self.singleChar(.bitwise_and),
            '=' => {
                self.pos += 1;
                if (self.pos < self.source.len and self.source[self.pos] == '=') {
                    self.pos += 1;
                    return .{ .type = .equals, .start = start, .len = 2 };
                }
                return .{ .type = .equals, .start = start, .len = 1 };
            },
            '!' => {
                self.pos += 1;
                if (self.pos < self.source.len and self.source[self.pos] == '=') {
                    self.pos += 1;
                    return .{ .type = .not_equals, .start = start, .len = 2 };
                }
                return .{ .type = .invalid, .start = start, .len = 1 };
            },
            '<' => {
                self.pos += 1;
                if (self.pos < self.source.len) {
                    switch (self.source[self.pos]) {
                        '=' => {
                            self.pos += 1;
                            return .{ .type = .less_than_or_equal, .start = start, .len = 2 };
                        },
                        '>' => {
                            self.pos += 1;
                            return .{ .type = .not_equals, .start = start, .len = 2 };
                        },
                        '<' => {
                            self.pos += 1;
                            return .{ .type = .left_shift, .start = start, .len = 2 };
                        },
                        '@' => {
                            self.pos += 1;
                            return .{ .type = .json_contained_by, .start = start, .len = 2 };
                        },
                        else => {},
                    }
                }
                return .{ .type = .less_than, .start = start, .len = 1 };
            },
            '@' => {
                self.pos += 1;
                if (self.pos < self.source.len) {
                    switch (self.source[self.pos]) {
                        '>' => {
                            self.pos += 1;
                            return .{ .type = .json_contains, .start = start, .len = 2 };
                        },
                        '@' => {
                            self.pos += 1;
                            return .{ .type = .ts_match, .start = start, .len = 2 };
                        },
                        else => {},
                    }
                }
                return .{ .type = .invalid, .start = start, .len = 1 };
            },
            '>' => {
                self.pos += 1;
                if (self.pos < self.source.len) {
                    switch (self.source[self.pos]) {
                        '=' => {
                            self.pos += 1;
                            return .{ .type = .greater_than_or_equal, .start = start, .len = 2 };
                        },
                        '>' => {
                            self.pos += 1;
                            return .{ .type = .right_shift, .start = start, .len = 2 };
                        },
                        else => {},
                    }
                }
                return .{ .type = .greater_than, .start = start, .len = 1 };
            },
            '|' => {
                self.pos += 1;
                if (self.pos < self.source.len and self.source[self.pos] == '|') {
                    self.pos += 1;
                    return .{ .type = .concat, .start = start, .len = 2 };
                }
                return .{ .type = .bitwise_or, .start = start, .len = 1 };
            },
            '?' => {
                self.pos += 1;
                if (self.pos < self.source.len) {
                    switch (self.source[self.pos]) {
                        '|' => {
                            self.pos += 1;
                            return .{ .type = .json_any_key_exists, .start = start, .len = 2 };
                        },
                        '&' => {
                            self.pos += 1;
                            return .{ .type = .json_all_keys_exist, .start = start, .len = 2 };
                        },
                        else => {},
                    }
                }
                return .{ .type = .json_key_exists, .start = start, .len = 1 };
            },
            '#' => {
                self.pos += 1;
                if (self.pos < self.source.len) {
                    switch (self.source[self.pos]) {
                        '>' => {
                            self.pos += 1;
                            if (self.pos < self.source.len and self.source[self.pos] == '>') {
                                self.pos += 1;
                                return .{ .type = .json_path_extract_text, .start = start, .len = 3 };
                            }
                            return .{ .type = .json_path_extract, .start = start, .len = 2 };
                        },
                        '-' => {
                            self.pos += 1;
                            return .{ .type = .json_delete_path, .start = start, .len = 2 };
                        },
                        else => {},
                    }
                }
                return .{ .type = .invalid, .start = start, .len = 1 };
            },
            '\'' => return self.scanString(),
            '"' => return self.scanQuotedIdentifier(),
            else => {},
        }

        // Numbers (including hex: 0x...)
        if (isDigit(ch)) {
            if (ch == '0' and self.pos + 1 < self.source.len) {
                const next_ch = self.source[self.pos + 1];
                if (next_ch == 'x' or next_ch == 'X') {
                    return self.scanHexNumber();
                }
            }
            return self.scanNumber();
        }

        // Blob literal: X'...' or x'...'
        if ((ch == 'X' or ch == 'x') and self.pos + 1 < self.source.len and self.source[self.pos + 1] == '\'') {
            return self.scanBlobLiteral();
        }

        // Identifiers and keywords
        if (isIdentStart(ch)) {
            return self.scanIdentifierOrKeyword();
        }

        // Unknown character
        self.pos += 1;
        return .{ .type = .invalid, .start = start, .len = 1 };
    }

    /// Collect all tokens into a slice (caller owns memory)
    pub fn tokenize(self: *Tokenizer, allocator: std.mem.Allocator) ![]Token {
        var tokens: std.ArrayListUnmanaged(Token) = .{};
        errdefer tokens.deinit(allocator);

        while (true) {
            const tok = self.next();
            try tokens.append(allocator, tok);
            if (tok.type == .eof) break;
        }

        return tokens.toOwnedSlice(allocator);
    }

    // --- Private scanning methods ---

    fn singleChar(self: *Tokenizer, token_type: TokenType) Token {
        const start = self.pos;
        self.pos += 1;
        return .{ .type = token_type, .start = start, .len = 1 };
    }

    fn skipWhitespaceAndComments(self: *Tokenizer) void {
        while (self.pos < self.source.len) {
            const ch = self.source[self.pos];
            if (ch == ' ' or ch == '\t' or ch == '\n' or ch == '\r') {
                self.pos += 1;
                continue;
            }
            // Line comment: -- ...
            if (ch == '-' and self.pos + 1 < self.source.len and self.source[self.pos + 1] == '-') {
                self.pos += 2;
                while (self.pos < self.source.len and self.source[self.pos] != '\n') {
                    self.pos += 1;
                }
                continue;
            }
            // Block comment: /* ... */
            if (ch == '/' and self.pos + 1 < self.source.len and self.source[self.pos + 1] == '*') {
                self.pos += 2;
                var depth: u32 = 1;
                while (self.pos + 1 < self.source.len and depth > 0) {
                    if (self.source[self.pos] == '/' and self.source[self.pos + 1] == '*') {
                        depth += 1;
                        self.pos += 2;
                    } else if (self.source[self.pos] == '*' and self.source[self.pos + 1] == '/') {
                        depth -= 1;
                        self.pos += 2;
                    } else {
                        self.pos += 1;
                    }
                }
                continue;
            }
            break;
        }
    }

    fn scanString(self: *Tokenizer) Token {
        const start = self.pos;
        self.pos += 1; // skip opening '

        while (self.pos < self.source.len) {
            if (self.source[self.pos] == '\'') {
                self.pos += 1;
                // Escaped quote: '' → literal '
                if (self.pos < self.source.len and self.source[self.pos] == '\'') {
                    self.pos += 1;
                    continue;
                }
                return .{ .type = .string_literal, .start = start, .len = self.pos - start };
            }
            self.pos += 1;
        }

        // Unterminated string
        return .{ .type = .invalid, .start = start, .len = self.pos - start };
    }

    fn scanQuotedIdentifier(self: *Tokenizer) Token {
        const start = self.pos;
        self.pos += 1; // skip opening "

        while (self.pos < self.source.len) {
            if (self.source[self.pos] == '"') {
                self.pos += 1;
                // Escaped quote: "" → literal "
                if (self.pos < self.source.len and self.source[self.pos] == '"') {
                    self.pos += 1;
                    continue;
                }
                return .{ .type = .quoted_identifier, .start = start, .len = self.pos - start };
            }
            self.pos += 1;
        }

        // Unterminated quoted identifier
        return .{ .type = .invalid, .start = start, .len = self.pos - start };
    }

    fn scanNumber(self: *Tokenizer) Token {
        const start = self.pos;
        var is_float = false;

        // Leading dot (.123)
        if (self.source[self.pos] == '.') {
            is_float = true;
            self.pos += 1;
            while (self.pos < self.source.len and isDigit(self.source[self.pos])) {
                self.pos += 1;
            }
        } else {
            // Integer part
            while (self.pos < self.source.len and isDigit(self.source[self.pos])) {
                self.pos += 1;
            }
            // Fractional part
            if (self.pos < self.source.len and self.source[self.pos] == '.') {
                is_float = true;
                self.pos += 1;
                while (self.pos < self.source.len and isDigit(self.source[self.pos])) {
                    self.pos += 1;
                }
            }
        }

        // Exponent part (e/E)
        if (self.pos < self.source.len and (self.source[self.pos] == 'e' or self.source[self.pos] == 'E')) {
            is_float = true;
            self.pos += 1;
            if (self.pos < self.source.len and (self.source[self.pos] == '+' or self.source[self.pos] == '-')) {
                self.pos += 1;
            }
            if (self.pos >= self.source.len or !isDigit(self.source[self.pos])) {
                return .{ .type = .invalid, .start = start, .len = self.pos - start };
            }
            while (self.pos < self.source.len and isDigit(self.source[self.pos])) {
                self.pos += 1;
            }
        }

        const token_type: TokenType = if (is_float) .float_literal else .integer_literal;
        return .{ .type = token_type, .start = start, .len = self.pos - start };
    }

    fn scanHexNumber(self: *Tokenizer) Token {
        const start = self.pos;
        self.pos += 2; // skip 0x

        if (self.pos >= self.source.len or !isHexDigit(self.source[self.pos])) {
            return .{ .type = .invalid, .start = start, .len = self.pos - start };
        }

        while (self.pos < self.source.len and isHexDigit(self.source[self.pos])) {
            self.pos += 1;
        }

        return .{ .type = .integer_literal, .start = start, .len = self.pos - start };
    }

    fn scanBlobLiteral(self: *Tokenizer) Token {
        const start = self.pos;
        self.pos += 2; // skip X'

        while (self.pos < self.source.len and self.source[self.pos] != '\'') {
            if (!isHexDigit(self.source[self.pos])) {
                // Invalid character in blob literal
                while (self.pos < self.source.len and self.source[self.pos] != '\'') {
                    self.pos += 1;
                }
                if (self.pos < self.source.len) self.pos += 1;
                return .{ .type = .invalid, .start = start, .len = self.pos - start };
            }
            self.pos += 1;
        }

        if (self.pos >= self.source.len) {
            return .{ .type = .invalid, .start = start, .len = self.pos - start };
        }

        self.pos += 1; // skip closing '
        return .{ .type = .blob_literal, .start = start, .len = self.pos - start };
    }

    fn scanIdentifierOrKeyword(self: *Tokenizer) Token {
        const start = self.pos;
        self.pos += 1;
        while (self.pos < self.source.len and isIdentCont(self.source[self.pos])) {
            self.pos += 1;
        }
        const len = self.pos - start;
        const text = self.source[start..][0..len];

        const kw = lookupKeyword(text);
        return .{ .type = kw orelse .identifier, .start = start, .len = len };
    }

    // --- Character classification ---

    fn isDigit(ch: u8) bool {
        return ch >= '0' and ch <= '9';
    }

    fn isHexDigit(ch: u8) bool {
        return (ch >= '0' and ch <= '9') or
            (ch >= 'a' and ch <= 'f') or
            (ch >= 'A' and ch <= 'F');
    }

    fn isIdentStart(ch: u8) bool {
        return (ch >= 'a' and ch <= 'z') or
            (ch >= 'A' and ch <= 'Z') or
            ch == '_';
    }

    fn isIdentCont(ch: u8) bool {
        return isIdentStart(ch) or isDigit(ch);
    }
};

/// Case-insensitive keyword lookup
fn lookupKeyword(text: []const u8) ?TokenType {
    // Convert to lowercase for comparison (stack buffer, max keyword length 12)
    var lower: [16]u8 = undefined;
    if (text.len > lower.len) return null;

    for (text, 0..) |ch, i| {
        lower[i] = if (ch >= 'A' and ch <= 'Z') ch + 32 else ch;
    }
    const key = lower[0..text.len];

    const map = std.StaticStringMap(TokenType).initComptime(.{
        // DDL
        .{ "create", .kw_create },
        .{ "table", .kw_table },
        .{ "drop", .kw_drop },
        .{ "index", .kw_index },
        .{ "alter", .kw_alter },
        .{ "add", .kw_add },
        .{ "column", .kw_column },
        .{ "rename", .kw_rename },
        .{ "to", .kw_to },
        .{ "if", .kw_if },
        .{ "exists", .kw_exists },
        .{ "primary", .kw_primary },
        .{ "key", .kw_key },
        .{ "unique", .kw_unique },
        .{ "not", .kw_not },
        .{ "null", .kw_null },
        .{ "default", .kw_default },
        .{ "check", .kw_check },
        .{ "foreign", .kw_foreign },
        .{ "references", .kw_references },
        .{ "autoincrement", .kw_autoincrement },
        .{ "constraint", .kw_constraint },
        .{ "cascade", .kw_cascade },
        .{ "restrict", .kw_restrict },
        .{ "set", .kw_set },
        .{ "action", .kw_action },
        .{ "no", .kw_no },
        .{ "on", .kw_on },
        .{ "delete", .kw_delete },
        .{ "update", .kw_update },
        .{ "without", .kw_without },
        .{ "rowid", .kw_rowid },
        .{ "strict", .kw_strict },
        .{ "temp", .kw_temp },
        .{ "temporary", .kw_temporary },
        .{ "view", .kw_view },
        .{ "replace", .kw_replace },
        .{ "with", .kw_with },
        .{ "option", .kw_option },
        .{ "recursive", .kw_recursive },
        .{ "materialized", .kw_materialized },
        .{ "local", .kw_local },
        .{ "cascaded", .kw_cascaded },
        .{ "type", .kw_type },
        .{ "enum", .kw_enum },
        .{ "domain", .kw_domain },
        // DML
        .{ "select", .kw_select },
        .{ "from", .kw_from },
        .{ "where", .kw_where },
        .{ "insert", .kw_insert },
        .{ "into", .kw_into },
        .{ "values", .kw_values },
        .{ "as", .kw_as },
        .{ "and", .kw_and },
        .{ "or", .kw_or },
        .{ "in", .kw_in },
        .{ "between", .kw_between },
        .{ "is", .kw_is },
        .{ "like", .kw_like },
        .{ "glob", .kw_glob },
        .{ "order", .kw_order },
        .{ "by", .kw_by },
        .{ "asc", .kw_asc },
        .{ "desc", .kw_desc },
        .{ "limit", .kw_limit },
        .{ "offset", .kw_offset },
        .{ "group", .kw_group },
        .{ "having", .kw_having },
        .{ "distinct", .kw_distinct },
        .{ "all", .kw_all },
        .{ "any", .kw_any },
        .{ "union", .kw_union },
        .{ "except", .kw_except },
        .{ "intersect", .kw_intersect },
        .{ "join", .kw_join },
        .{ "inner", .kw_inner },
        .{ "left", .kw_left },
        .{ "right", .kw_right },
        .{ "full", .kw_full },
        .{ "outer", .kw_outer },
        .{ "cross", .kw_cross },
        .{ "natural", .kw_natural },
        .{ "case", .kw_case },
        .{ "when", .kw_when },
        .{ "then", .kw_then },
        .{ "else", .kw_else },
        .{ "end", .kw_end },
        .{ "cast", .kw_cast },
        // Aggregates
        .{ "count", .kw_count },
        .{ "sum", .kw_sum },
        .{ "avg", .kw_avg },
        .{ "min", .kw_min },
        .{ "max", .kw_max },
        // Window functions
        .{ "over", .kw_over },
        .{ "partition", .kw_partition },
        .{ "rows", .kw_rows },
        .{ "range", .kw_range },
        .{ "groups", .kw_groups },
        .{ "unbounded", .kw_unbounded },
        .{ "preceding", .kw_preceding },
        .{ "following", .kw_following },
        .{ "current", .kw_current },
        .{ "row", .kw_row },
        .{ "window", .kw_window },
        .{ "row_number", .kw_row_number },
        .{ "rank", .kw_rank },
        .{ "dense_rank", .kw_dense_rank },
        .{ "ntile", .kw_ntile },
        .{ "lag", .kw_lag },
        .{ "lead", .kw_lead },
        .{ "first_value", .kw_first_value },
        .{ "last_value", .kw_last_value },
        .{ "nth_value", .kw_nth_value },
        .{ "percent_rank", .kw_percent_rank },
        .{ "cume_dist", .kw_cume_dist },
        // Transaction
        .{ "begin", .kw_begin },
        .{ "commit", .kw_commit },
        .{ "rollback", .kw_rollback },
        .{ "savepoint", .kw_savepoint },
        .{ "release", .kw_release },
        .{ "transaction", .kw_transaction },
        .{ "deferred", .kw_deferred },
        .{ "immediate", .kw_immediate },
        .{ "exclusive", .kw_exclusive },
        // Utility
        .{ "explain", .kw_explain },
        .{ "pragma", .kw_pragma },
        .{ "vacuum", .kw_vacuum },
        // Types
        .{ "integer", .kw_integer },
        .{ "int", .kw_int },
        .{ "real", .kw_real },
        .{ "text", .kw_text },
        .{ "blob", .kw_blob },
        .{ "boolean", .kw_boolean },
        .{ "varchar", .kw_varchar },
        .{ "date", .kw_date },
        .{ "time", .kw_time },
        .{ "timestamp", .kw_timestamp },
        .{ "interval", .kw_interval },
        .{ "numeric", .kw_numeric },
        .{ "decimal", .kw_decimal },
        .{ "uuid", .kw_uuid },
        .{ "serial", .kw_serial },
        .{ "bigserial", .kw_bigserial },
        .{ "array", .kw_array },
        .{ "json", .kw_json },
        .{ "jsonb", .kw_jsonb },
        .{ "tsvector", .kw_tsvector },
        .{ "tsquery", .kw_tsquery },
        // Values
        .{ "true", .kw_true },
        .{ "false", .kw_false },
    });

    return map.get(key);
}

// ============================================================
// Tests
// ============================================================

fn expectTokens(source: []const u8, expected: []const TokenType) !void {
    var tokenizer = Tokenizer.init(source);
    for (expected) |exp| {
        const tok = tokenizer.next();
        try std.testing.expectEqual(exp, tok.type);
    }
    const eof = tokenizer.next();
    try std.testing.expectEqual(TokenType.eof, eof.type);
}

fn expectSingleToken(source: []const u8, expected_type: TokenType, expected_lexeme: []const u8) !void {
    var tokenizer = Tokenizer.init(source);
    const tok = tokenizer.next();
    try std.testing.expectEqual(expected_type, tok.type);
    try std.testing.expectEqualStrings(expected_lexeme, tok.lexeme(source));
    try std.testing.expectEqual(TokenType.eof, tokenizer.next().type);
}

test "empty input" {
    var tokenizer = Tokenizer.init("");
    const tok = tokenizer.next();
    try std.testing.expectEqual(TokenType.eof, tok.type);
}

test "whitespace only" {
    var tokenizer = Tokenizer.init("   \t\n\r  ");
    const tok = tokenizer.next();
    try std.testing.expectEqual(TokenType.eof, tok.type);
}

test "single punctuation" {
    try expectSingleToken("(", .left_paren, "(");
    try expectSingleToken(")", .right_paren, ")");
    try expectSingleToken(",", .comma, ",");
    try expectSingleToken(";", .semicolon, ";");
    try expectSingleToken(".", .dot, ".");
}

test "arithmetic operators" {
    try expectSingleToken("+", .plus, "+");
    try expectSingleToken("-", .minus, "-");
    try expectSingleToken("*", .star, "*");
    try expectSingleToken("/", .slash, "/");
    try expectSingleToken("%", .percent, "%");
}

test "comparison operators" {
    try expectSingleToken("=", .equals, "=");
    try expectSingleToken("==", .equals, "==");
    try expectSingleToken("!=", .not_equals, "!=");
    try expectSingleToken("<>", .not_equals, "<>");
    try expectSingleToken("<", .less_than, "<");
    try expectSingleToken(">", .greater_than, ">");
    try expectSingleToken("<=", .less_than_or_equal, "<=");
    try expectSingleToken(">=", .greater_than_or_equal, ">=");
}

test "bitwise and shift operators" {
    try expectSingleToken("&", .bitwise_and, "&");
    try expectSingleToken("|", .bitwise_or, "|");
    try expectSingleToken("~", .bitwise_not, "~");
    try expectSingleToken("<<", .left_shift, "<<");
    try expectSingleToken(">>", .right_shift, ">>");
    try expectSingleToken("||", .concat, "||");
}

test "integer literals" {
    try expectSingleToken("0", .integer_literal, "0");
    try expectSingleToken("42", .integer_literal, "42");
    try expectSingleToken("1234567890", .integer_literal, "1234567890");
}

test "float literals" {
    try expectSingleToken("3.14", .float_literal, "3.14");
    try expectSingleToken(".5", .float_literal, ".5");
    try expectSingleToken("1.", .float_literal, "1.");
    try expectSingleToken("1e10", .float_literal, "1e10");
    try expectSingleToken("1.5e-3", .float_literal, "1.5e-3");
    try expectSingleToken("2E+4", .float_literal, "2E+4");
}

test "hex integer" {
    try expectSingleToken("0xFF", .integer_literal, "0xFF");
    try expectSingleToken("0x1a2B", .integer_literal, "0x1a2B");
}

test "string literals" {
    try expectSingleToken("'hello'", .string_literal, "'hello'");
    try expectSingleToken("''", .string_literal, "''");
    try expectSingleToken("'it''s'", .string_literal, "'it''s'");
}

test "unterminated string" {
    try expectSingleToken("'hello", .invalid, "'hello");
}

test "blob literals" {
    try expectSingleToken("X'FF'", .blob_literal, "X'FF'");
    try expectSingleToken("x'0123456789abcdef'", .blob_literal, "x'0123456789abcdef'");
}

test "quoted identifiers" {
    try expectSingleToken("\"my column\"", .quoted_identifier, "\"my column\"");
    try expectSingleToken("\"has\"\"quotes\"", .quoted_identifier, "\"has\"\"quotes\"");
}

test "identifiers" {
    try expectSingleToken("foo", .identifier, "foo");
    try expectSingleToken("_bar", .identifier, "_bar");
    try expectSingleToken("table1", .identifier, "table1");
}

test "keywords case insensitive" {
    try expectSingleToken("SELECT", .kw_select, "SELECT");
    try expectSingleToken("select", .kw_select, "select");
    try expectSingleToken("Select", .kw_select, "Select");
    try expectSingleToken("INSERT", .kw_insert, "INSERT");
    try expectSingleToken("CREATE", .kw_create, "CREATE");
    try expectSingleToken("TABLE", .kw_table, "TABLE");
}

test "all DDL keywords" {
    try expectSingleToken("create", .kw_create, "create");
    try expectSingleToken("table", .kw_table, "table");
    try expectSingleToken("drop", .kw_drop, "drop");
    try expectSingleToken("index", .kw_index, "index");
    try expectSingleToken("alter", .kw_alter, "alter");
    try expectSingleToken("primary", .kw_primary, "primary");
    try expectSingleToken("key", .kw_key, "key");
    try expectSingleToken("unique", .kw_unique, "unique");
    try expectSingleToken("not", .kw_not, "not");
    try expectSingleToken("null", .kw_null, "null");
    try expectSingleToken("default", .kw_default, "default");
    try expectSingleToken("foreign", .kw_foreign, "foreign");
    try expectSingleToken("references", .kw_references, "references");
    try expectSingleToken("autoincrement", .kw_autoincrement, "autoincrement");
}

test "all DML keywords" {
    try expectSingleToken("select", .kw_select, "select");
    try expectSingleToken("from", .kw_from, "from");
    try expectSingleToken("where", .kw_where, "where");
    try expectSingleToken("insert", .kw_insert, "insert");
    try expectSingleToken("into", .kw_into, "into");
    try expectSingleToken("values", .kw_values, "values");
    try expectSingleToken("and", .kw_and, "and");
    try expectSingleToken("or", .kw_or, "or");
    try expectSingleToken("order", .kw_order, "order");
    try expectSingleToken("by", .kw_by, "by");
    try expectSingleToken("join", .kw_join, "join");
    try expectSingleToken("inner", .kw_inner, "inner");
    try expectSingleToken("left", .kw_left, "left");
    try expectSingleToken("group", .kw_group, "group");
    try expectSingleToken("having", .kw_having, "having");
    try expectSingleToken("distinct", .kw_distinct, "distinct");
    try expectSingleToken("limit", .kw_limit, "limit");
    try expectSingleToken("offset", .kw_offset, "offset");
}

test "aggregate keywords" {
    try expectSingleToken("count", .kw_count, "count");
    try expectSingleToken("sum", .kw_sum, "sum");
    try expectSingleToken("avg", .kw_avg, "avg");
    try expectSingleToken("min", .kw_min, "min");
    try expectSingleToken("max", .kw_max, "max");
}

test "transaction keywords" {
    try expectSingleToken("begin", .kw_begin, "begin");
    try expectSingleToken("commit", .kw_commit, "commit");
    try expectSingleToken("rollback", .kw_rollback, "rollback");
    try expectSingleToken("savepoint", .kw_savepoint, "savepoint");
    try expectSingleToken("release", .kw_release, "release");
    try expectSingleToken("transaction", .kw_transaction, "transaction");
}

test "type keywords" {
    try expectSingleToken("integer", .kw_integer, "integer");
    try expectSingleToken("int", .kw_int, "int");
    try expectSingleToken("real", .kw_real, "real");
    try expectSingleToken("text", .kw_text, "text");
    try expectSingleToken("blob", .kw_blob, "blob");
    try expectSingleToken("boolean", .kw_boolean, "boolean");
    try expectSingleToken("varchar", .kw_varchar, "varchar");
}

test "value keywords" {
    try expectSingleToken("true", .kw_true, "true");
    try expectSingleToken("false", .kw_false, "false");
}

test "line comments" {
    try expectTokens("-- this is a comment", &.{});
    try expectTokens("42 -- comment\n 7", &.{ .integer_literal, .integer_literal });
}

test "block comments" {
    try expectTokens("/* comment */", &.{});
    try expectTokens("42 /* comment */ 7", &.{ .integer_literal, .integer_literal });
}

test "nested block comments" {
    try expectTokens("/* outer /* inner */ still comment */", &.{});
}

test "simple SELECT statement" {
    const sql = "SELECT id, name FROM users WHERE age > 18;";
    try expectTokens(sql, &.{
        .kw_select, .identifier, .comma,     .identifier, .kw_from,
        .identifier, .kw_where,  .identifier, .greater_than, .integer_literal,
        .semicolon,
    });
}

test "CREATE TABLE statement" {
    const sql = "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL);";
    try expectTokens(sql, &.{
        .kw_create,  .kw_table,   .identifier, .left_paren, .identifier,
        .kw_integer, .kw_primary, .kw_key,     .comma,      .identifier,
        .kw_text,    .kw_not,     .kw_null,    .right_paren, .semicolon,
    });
}

test "INSERT statement" {
    const sql = "INSERT INTO users (name, age) VALUES ('Alice', 30);";
    try expectTokens(sql, &.{
        .kw_insert,      .kw_into,          .identifier,  .left_paren,
        .identifier,     .comma,            .identifier,  .right_paren,
        .kw_values,      .left_paren,       .string_literal, .comma,
        .integer_literal, .right_paren, .semicolon,
    });
}

test "complex WHERE clause" {
    const sql = "SELECT * FROM t WHERE a >= 1 AND b <= 10 OR c IN (1, 2, 3)";
    try expectTokens(sql, &.{
        .kw_select,            .star,                 .kw_from,
        .identifier,           .kw_where,             .identifier,
        .greater_than_or_equal, .integer_literal,     .kw_and,
        .identifier,           .less_than_or_equal,   .integer_literal,
        .kw_or,                .identifier,           .kw_in,
        .left_paren,           .integer_literal,      .comma,
        .integer_literal,      .comma,                .integer_literal,
        .right_paren,
    });
}

test "JOIN query" {
    const sql = "SELECT a.id FROM a INNER JOIN b ON a.id = b.id";
    try expectTokens(sql, &.{
        .kw_select,  .identifier, .dot,       .identifier, .kw_from,
        .identifier, .kw_inner,   .kw_join,   .identifier, .kw_on,
        .identifier, .dot,        .identifier, .equals,    .identifier,
        .dot,        .identifier,
    });
}

test "string concatenation" {
    const sql = "'hello' || ' ' || 'world'";
    try expectTokens(sql, &.{
        .string_literal, .concat, .string_literal, .concat, .string_literal,
    });
}

test "negative number expression" {
    // -5 is tokenized as minus + integer, not a negative literal
    const sql = "-5";
    try expectTokens(sql, &.{ .minus, .integer_literal });
}

test "dot-prefixed float" {
    const sql = "SELECT .5 FROM t";
    try expectTokens(sql, &.{ .kw_select, .float_literal, .kw_from, .identifier });
}

test "tokenize collects all tokens" {
    const sql = "SELECT 1;";
    var tokenizer = Tokenizer.init(sql);
    const tokens = try tokenizer.tokenize(std.testing.allocator);
    defer std.testing.allocator.free(tokens);

    try std.testing.expectEqual(@as(usize, 4), tokens.len);
    try std.testing.expectEqual(TokenType.kw_select, tokens[0].type);
    try std.testing.expectEqual(TokenType.integer_literal, tokens[1].type);
    try std.testing.expectEqual(TokenType.semicolon, tokens[2].type);
    try std.testing.expectEqual(TokenType.eof, tokens[3].type);
}

test "token lexeme" {
    const sql = "SELECT name FROM users";
    var tokenizer = Tokenizer.init(sql);

    const tok1 = tokenizer.next();
    try std.testing.expectEqualStrings("SELECT", tok1.lexeme(sql));

    const tok2 = tokenizer.next();
    try std.testing.expectEqualStrings("name", tok2.lexeme(sql));

    const tok3 = tokenizer.next();
    try std.testing.expectEqualStrings("FROM", tok3.lexeme(sql));

    const tok4 = tokenizer.next();
    try std.testing.expectEqualStrings("users", tok4.lexeme(sql));
}

test "isKeyword" {
    try std.testing.expect(TokenType.kw_select.isKeyword());
    try std.testing.expect(TokenType.kw_false.isKeyword());
    try std.testing.expect(!TokenType.identifier.isKeyword());
    try std.testing.expect(!TokenType.integer_literal.isKeyword());
    try std.testing.expect(!TokenType.eof.isKeyword());
}

test "invalid character" {
    var tokenizer = Tokenizer.init("@");
    const tok = tokenizer.next();
    try std.testing.expectEqual(TokenType.invalid, tok.type);
}

test "EXPLAIN and PRAGMA" {
    try expectSingleToken("explain", .kw_explain, "explain");
    try expectSingleToken("pragma", .kw_pragma, "pragma");
    try expectSingleToken("vacuum", .kw_vacuum, "vacuum");
}

test "VIEW and CTE keywords" {
    try expectSingleToken("view", .kw_view, "view");
    try expectSingleToken("VIEW", .kw_view, "VIEW");
    try expectSingleToken("replace", .kw_replace, "replace");
    try expectSingleToken("with", .kw_with, "with");
    try expectSingleToken("WITH", .kw_with, "WITH");
    try expectSingleToken("option", .kw_option, "option");
    try expectSingleToken("recursive", .kw_recursive, "recursive");
    try expectSingleToken("materialized", .kw_materialized, "materialized");
}

test "CASE expression tokens" {
    const sql = "CASE WHEN x = 1 THEN 'one' ELSE 'other' END";
    try expectTokens(sql, &.{
        .kw_case,        .kw_when,        .identifier, .equals,
        .integer_literal, .kw_then,       .string_literal,
        .kw_else,        .string_literal, .kw_end,
    });
}

test "multiline SQL" {
    const sql =
        \\SELECT
        \\  id,
        \\  name
        \\FROM
        \\  users
        \\WHERE
        \\  active = 1
    ;
    try expectTokens(sql, &.{
        .kw_select,  .identifier,      .comma,   .identifier,
        .kw_from,    .identifier,      .kw_where, .identifier,
        .equals,     .integer_literal,
    });
}

test "consecutive tokens without whitespace" {
    try expectTokens("(1+2)*3", &.{
        .left_paren, .integer_literal, .plus, .integer_literal,
        .right_paren, .star, .integer_literal,
    });
}

test "bang alone is invalid" {
    try expectSingleToken("!", .invalid, "!");
}

test "invalid hex number" {
    // 0x with no digits
    var tokenizer = Tokenizer.init("0x ");
    const tok = tokenizer.next();
    try std.testing.expectEqual(TokenType.invalid, tok.type);
}

test "invalid exponent" {
    // 1e with no digits
    var tokenizer = Tokenizer.init("1e ");
    const tok = tokenizer.next();
    try std.testing.expectEqual(TokenType.invalid, tok.type);
}

test "blob literal with invalid chars" {
    var tokenizer = Tokenizer.init("X'GG'");
    const tok = tokenizer.next();
    try std.testing.expectEqual(TokenType.invalid, tok.type);
}

test "unterminated blob literal" {
    var tokenizer = Tokenizer.init("X'FF");
    const tok = tokenizer.next();
    try std.testing.expectEqual(TokenType.invalid, tok.type);
}

test "unterminated quoted identifier" {
    var tokenizer = Tokenizer.init("\"unclosed");
    const tok = tokenizer.next();
    try std.testing.expectEqual(TokenType.invalid, tok.type);
}

test "BETWEEN expression" {
    const sql = "x BETWEEN 1 AND 10";
    try expectTokens(sql, &.{
        .identifier, .kw_between, .integer_literal, .kw_and, .integer_literal,
    });
}

test "CAST expression" {
    const sql = "CAST(x AS INTEGER)";
    try expectTokens(sql, &.{
        .kw_cast, .left_paren, .identifier, .kw_as, .kw_integer, .right_paren,
    });
}

test "IS NULL / IS NOT NULL" {
    try expectTokens("x IS NULL", &.{ .identifier, .kw_is, .kw_null });
    try expectTokens("x IS NOT NULL", &.{ .identifier, .kw_is, .kw_not, .kw_null });
}

test "GROUP BY HAVING" {
    const sql = "SELECT dept, COUNT(*) FROM emp GROUP BY dept HAVING COUNT(*) > 5";
    try expectTokens(sql, &.{
        .kw_select,        .identifier, .comma,    .kw_count,
        .left_paren,       .star,       .right_paren, .kw_from,
        .identifier,       .kw_group,   .kw_by,    .identifier,
        .kw_having,        .kw_count,   .left_paren, .star,
        .right_paren,      .greater_than, .integer_literal,
    });
}

test "UNION ALL" {
    try expectTokens("SELECT 1 UNION ALL SELECT 2", &.{
        .kw_select, .integer_literal, .kw_union, .kw_all,
        .kw_select, .integer_literal,
    });
}

test "ORDER BY ASC DESC" {
    try expectTokens("ORDER BY a ASC, b DESC", &.{
        .kw_order, .kw_by, .identifier, .kw_asc, .comma, .identifier, .kw_desc,
    });
}

test "LIMIT OFFSET" {
    try expectTokens("LIMIT 10 OFFSET 20", &.{
        .kw_limit, .integer_literal, .kw_offset, .integer_literal,
    });
}

test "window function keywords" {
    try expectSingleToken("over", .kw_over, "over");
    try expectSingleToken("OVER", .kw_over, "OVER");
    try expectSingleToken("partition", .kw_partition, "partition");
    try expectSingleToken("rows", .kw_rows, "rows");
    try expectSingleToken("range", .kw_range, "range");
    try expectSingleToken("groups", .kw_groups, "groups");
    try expectSingleToken("unbounded", .kw_unbounded, "unbounded");
    try expectSingleToken("preceding", .kw_preceding, "preceding");
    try expectSingleToken("following", .kw_following, "following");
    try expectSingleToken("current", .kw_current, "current");
    try expectSingleToken("row", .kw_row, "row");
    try expectSingleToken("window", .kw_window, "window");
    try expectSingleToken("row_number", .kw_row_number, "row_number");
    try expectSingleToken("rank", .kw_rank, "rank");
    try expectSingleToken("dense_rank", .kw_dense_rank, "dense_rank");
    try expectSingleToken("ntile", .kw_ntile, "ntile");
    try expectSingleToken("lag", .kw_lag, "lag");
    try expectSingleToken("lead", .kw_lead, "lead");
    try expectSingleToken("first_value", .kw_first_value, "first_value");
    try expectSingleToken("last_value", .kw_last_value, "last_value");
    try expectSingleToken("nth_value", .kw_nth_value, "nth_value");
    try expectSingleToken("percent_rank", .kw_percent_rank, "percent_rank");
    try expectSingleToken("cume_dist", .kw_cume_dist, "cume_dist");
}

test "window function expression tokens" {
    const sql = "ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC)";
    try expectTokens(sql, &.{
        .kw_row_number, .left_paren,  .right_paren,  .kw_over,
        .left_paren,    .kw_partition, .kw_by,        .identifier,
        .kw_order,      .kw_by,        .identifier,   .kw_desc,
        .right_paren,
    });
}

test "window frame specification tokens" {
    const sql = "SUM(x) OVER (ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)";
    try expectTokens(sql, &.{
        .kw_sum,       .left_paren,   .identifier,    .right_paren,
        .kw_over,      .left_paren,   .kw_rows,       .kw_between,
        .kw_unbounded, .kw_preceding, .kw_and,        .kw_current,
        .kw_row,       .right_paren,
    });
}

test "SERIAL and BIGSERIAL keywords" {
    try expectSingleToken("serial", .kw_serial, "serial");
    try expectSingleToken("bigserial", .kw_bigserial, "bigserial");
    try expectSingleToken("SERIAL", .kw_serial, "SERIAL");
    try expectSingleToken("BIGSERIAL", .kw_bigserial, "BIGSERIAL");
}

test "SERIAL in CREATE TABLE context" {
    const sql = "CREATE TABLE t (id SERIAL, name TEXT)";
    try expectTokens(sql, &.{
        .kw_create, .kw_table, .identifier, .left_paren,
        .identifier, .kw_serial, .comma,
        .identifier, .kw_text, .right_paren,
    });
}

test "ARRAY keyword" {
    try expectSingleToken("array", .kw_array, "array");
    try expectSingleToken("ARRAY", .kw_array, "ARRAY");
}

test "array brackets" {
    try expectSingleToken("[", .left_bracket, "[");
    try expectSingleToken("]", .right_bracket, "]");
}

test "ARRAY constructor expression tokens" {
    const sql = "ARRAY[1,2,3]";
    try expectTokens(sql, &.{
        .kw_array, .left_bracket, .integer_literal, .comma,
        .integer_literal, .comma, .integer_literal, .right_bracket,
    });
}

test "array subscript expression tokens" {
    const sql = "col[1]";
    try expectTokens(sql, &.{
        .identifier, .left_bracket, .integer_literal, .right_bracket,
    });
}

test "CREATE TABLE with ARRAY column type" {
    const sql = "CREATE TABLE t (tags INTEGER[])";
    try expectTokens(sql, &.{
        .kw_create, .kw_table, .identifier, .left_paren,
        .identifier, .kw_integer, .left_bracket, .right_bracket,
        .right_paren,
    });
}

test "TYPE and ENUM keywords" {
    try expectSingleToken("type", .kw_type, "type");
    try expectSingleToken("TYPE", .kw_type, "TYPE");
    try expectSingleToken("enum", .kw_enum, "enum");
    try expectSingleToken("ENUM", .kw_enum, "ENUM");
}

test "CREATE TYPE AS ENUM tokens" {
    const sql = "CREATE TYPE mood AS ENUM ('happy', 'sad')";
    try expectTokens(sql, &.{
        .kw_create,     .kw_type,   .identifier, .kw_as,
        .kw_enum,       .left_paren, .string_literal, .comma,
        .string_literal, .right_paren,
    });
}

test "JSON and JSONB keywords" {
    try expectSingleToken("json", .kw_json, "json");
    try expectSingleToken("JSON", .kw_json, "JSON");
    try expectSingleToken("jsonb", .kw_jsonb, "jsonb");
    try expectSingleToken("JSONB", .kw_jsonb, "JSONB");
}

test "CREATE TABLE with JSON column type" {
    const sql = "CREATE TABLE t (data JSON, metadata JSONB)";
    try expectTokens(sql, &.{
        .kw_create, .kw_table, .identifier, .left_paren,
        .identifier, .kw_json, .comma,
        .identifier, .kw_jsonb, .right_paren,
    });
}

test "JSON navigation operators" {
    // -> operator
    try expectSingleToken("->", .json_extract, "->");
    // ->> operator
    try expectSingleToken("->>", .json_extract_text, "->>");
    // Mixed with identifiers
    const sql = "data -> 'key'";
    try expectTokens(sql, &.{ .identifier, .json_extract, .string_literal });
    const sql2 = "data ->> 'key'";
    try expectTokens(sql2, &.{ .identifier, .json_extract_text, .string_literal });
}

test "JSON containment operators" {
    // @> operator
    try expectSingleToken("@>", .json_contains, "@>");
    // <@ operator
    try expectSingleToken("<@", .json_contained_by, "<@");
    // Mixed with identifiers
    const sql = "data @> '{\"key\":1}'";
    try expectTokens(sql, &.{ .identifier, .json_contains, .string_literal });
    const sql2 = "data <@ '{\"key\":1}'";
    try expectTokens(sql2, &.{ .identifier, .json_contained_by, .string_literal });
}

test "JSON existence operators" {
    // ? operator
    try expectSingleToken("?", .json_key_exists, "?");
    // ?| operator
    try expectSingleToken("?|", .json_any_key_exists, "?|");
    // ?& operator
    try expectSingleToken("?&", .json_all_keys_exist, "?&");
    // Mixed with identifiers
    const sql = "data ? 'key'";
    try expectTokens(sql, &.{ .identifier, .json_key_exists, .string_literal });
    const sql2 = "data ?| ARRAY['a','b']";
    try expectTokens(sql2, &.{ .identifier, .json_any_key_exists, .kw_array, .left_bracket, .string_literal, .comma, .string_literal, .right_bracket });
}

test "JSON path operators" {
    // #> operator
    try expectSingleToken("#>", .json_path_extract, "#>");
    // #>> operator
    try expectSingleToken("#>>", .json_path_extract_text, "#>>");
    // #- operator
    try expectSingleToken("#-", .json_delete_path, "#-");
    // Mixed with identifiers
    const sql = "data #> '{a,b}'";
    try expectTokens(sql, &.{ .identifier, .json_path_extract, .string_literal });
    const sql2 = "data #>> '{a,b}'";
    try expectTokens(sql2, &.{ .identifier, .json_path_extract_text, .string_literal });
    const sql3 = "data #- '{a}'";
    try expectTokens(sql3, &.{ .identifier, .json_delete_path, .string_literal });
}

test "full-text search @@ operator" {
    // @@ operator
    try expectSingleToken("@@", .ts_match, "@@");
    // Mixed with identifiers
    const sql = "doc @@ query";
    try expectTokens(sql, &.{ .identifier, .ts_match, .identifier });
    const sql2 = "to_tsvector('text') @@ to_tsquery('search')";
    try expectTokens(sql2, &.{
        .identifier,
        .left_paren,
        .string_literal,
        .right_paren,
        .ts_match,
        .identifier,
        .left_paren,
        .string_literal,
        .right_paren,
    });
}

test "JSON operators in SELECT queries" {
    const sql = "SELECT data -> 'name' FROM users WHERE data @> '{\"age\":30}'";
    try expectTokens(sql, &.{
        .kw_select,
        .identifier,
        .json_extract,
        .string_literal,
        .kw_from,
        .identifier,
        .kw_where,
        .identifier,
        .json_contains,
        .string_literal,
    });
}
