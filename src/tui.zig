const std = @import("std");
const sailor = @import("sailor");
const silica = @import("silica");
const tui = sailor.tui;

const Database = silica.engine.Database;
const QueryResult = silica.engine.QueryResult;
const executor = silica.executor;
const Value = executor.Value;
const Row = executor.Row;
const Catalog = silica.catalog.Catalog;
// Note: CompletionPopup from sailor has a bug with border_type field
// Using our own CompletionItem struct instead
const CompletionItem = struct {
    text: []const u8,
    description: ?[]const u8 = null,
};

// SQL keyword help text for tooltips
fn getSqlKeywordHelp(keyword: []const u8) ?[]const u8 {
    const upper = std.ascii.allocUpperString(std.heap.page_allocator, keyword) catch return null;
    defer std.heap.page_allocator.free(upper);

    const help_map = std.StaticStringMap([]const u8).initComptime(.{
        .{ "SELECT", "Retrieve rows from tables" },
        .{ "INSERT", "Add new rows to a table" },
        .{ "UPDATE", "Modify existing rows" },
        .{ "DELETE", "Remove rows from a table" },
        .{ "CREATE", "Create database objects (TABLE, INDEX, VIEW)" },
        .{ "DROP", "Remove database objects" },
        .{ "ALTER", "Modify table structure" },
        .{ "FROM", "Specify source table(s)" },
        .{ "WHERE", "Filter rows with conditions" },
        .{ "JOIN", "Combine rows from multiple tables" },
        .{ "INNER", "Return matching rows from both tables" },
        .{ "LEFT", "Return all left table rows + matches" },
        .{ "RIGHT", "Return all right table rows + matches" },
        .{ "FULL", "Return all rows with matches where available" },
        .{ "ON", "Specify join condition" },
        .{ "GROUP", "Aggregate rows by column values" },
        .{ "HAVING", "Filter grouped results" },
        .{ "ORDER", "Sort result rows" },
        .{ "BY", "Specify sort/group column(s)" },
        .{ "LIMIT", "Restrict number of rows returned" },
        .{ "OFFSET", "Skip N rows before returning results" },
        .{ "DISTINCT", "Remove duplicate rows" },
        .{ "AS", "Rename column or table (alias)" },
        .{ "AND", "Combine conditions (both must be true)" },
        .{ "OR", "Combine conditions (either can be true)" },
        .{ "NOT", "Negate a condition" },
        .{ "NULL", "Represents missing or unknown value" },
        .{ "IS", "Test for NULL values" },
        .{ "IN", "Match any value in a list" },
        .{ "BETWEEN", "Match values in a range" },
        .{ "LIKE", "Match text patterns (% and _ wildcards)" },
        .{ "EXISTS", "Check if subquery returns rows" },
        .{ "CASE", "Conditional expression (if-then-else)" },
        .{ "WHEN", "Condition in CASE expression" },
        .{ "THEN", "Result when WHEN condition is true" },
        .{ "ELSE", "Default result in CASE expression" },
        .{ "END", "Close CASE or compound statement" },
        .{ "UNION", "Combine results from multiple queries" },
        .{ "INTERSECT", "Return common rows from queries" },
        .{ "EXCEPT", "Return rows from first query not in second" },
        .{ "PRIMARY", "Uniquely identifies rows (PRIMARY KEY)" },
        .{ "FOREIGN", "References another table (FOREIGN KEY)" },
        .{ "KEY", "Constraint or index" },
        .{ "UNIQUE", "Prevent duplicate values in column" },
        .{ "CHECK", "Validate column values with condition" },
        .{ "DEFAULT", "Specify default column value" },
        .{ "INDEX", "Speed up queries on column(s)" },
        .{ "VIEW", "Virtual table from a query" },
        .{ "TRANSACTION", "Atomic unit of work (BEGIN...COMMIT)" },
        .{ "BEGIN", "Start transaction" },
        .{ "COMMIT", "Save transaction changes" },
        .{ "ROLLBACK", "Discard transaction changes" },
        .{ "SAVEPOINT", "Create rollback point within transaction" },
        .{ "RELEASE", "Remove savepoint" },
        .{ "WITH", "Define CTE (Common Table Expression)" },
        .{ "RECURSIVE", "Enable recursive CTE" },
        .{ "OVER", "Define window for window function" },
        .{ "PARTITION", "Divide rows into groups for window" },
        .{ "ROWS", "Define window frame by row count" },
        .{ "RANGE", "Define window frame by value range" },
        .{ "ANALYZE", "Collect table statistics for optimizer" },
        .{ "EXPLAIN", "Show query execution plan" },
        .{ "VACUUM", "Reclaim storage and optimize database" },
        .{ "REINDEX", "Rebuild index from scratch" },
    });

    return help_map.get(upper);
}

// Get table metadata tooltip (shows column info)
// Returns a thread-local static buffer — valid until next call
fn getTableHelp(db: *Database, table_name: []const u8) ?[]const u8 {
    // Thread-local static buffer for tooltip text (avoid allocation)
    const ThreadLocal = struct {
        var buf: [512]u8 = undefined;
    };

    // Query catalog for table metadata
    const table_info = db.catalog.getTable(table_name) catch return null;
    const col_count = table_info.columns.len;

    // Show column names and types (up to 3 columns)
    var col_info_buf: [256]u8 = undefined;
    var col_info_len: usize = 0;

    const max_cols_to_show = @min(col_count, 3);
    for (table_info.columns[0..max_cols_to_show], 0..) |col, i| {
        const type_name = @tagName(col.column_type);
        const col_text = if (i == 0)
            std.fmt.bufPrint(col_info_buf[col_info_len..], "{s}: {s}", .{ col.name, type_name })
        else
            std.fmt.bufPrint(col_info_buf[col_info_len..], ", {s}: {s}", .{ col.name, type_name });

        if (col_text) |text| {
            col_info_len += text.len;
        } else |_| break;
    }

    const col_info = col_info_buf[0..col_info_len];
    const more_cols = if (col_count > 3) "..." else "";

    // Format: "Table: <name> | <col_count> columns | <col1>: <type1>, <col2>: <type2>..."
    const help_text = std.fmt.bufPrint(
        &ThreadLocal.buf,
        "Table: {s} | {d} column{s} | {s}{s}",
        .{
            table_name,
            col_count,
            if (col_count == 1) "" else "s",
            col_info,
            more_cols,
        },
    ) catch return null;

    return help_text;
}

// ── Focus Pane ───────────────────────────────────────────────────────

const Pane = enum { schema, results, input };

// ── Application State ────────────────────────────────────────────────

const App = struct {
    allocator: std.mem.Allocator,
    db: *Database,
    db_path: []const u8,
    should_quit: bool = false,
    focus: Pane = .input,

    // Schema tree (flat list with indented columns)
    schema_items: std.ArrayListUnmanaged([]const u8) = .{}, // display items (table names + indented columns)
    schema_table_indices: std.ArrayListUnmanaged(usize) = .{}, // indices into schema_items that are table names
    schema_selected: usize = 0,
    schema_offset: usize = 0,

    // SQL input
    input_text: std.ArrayListUnmanaged(u8) = .{},
    input_cursor: usize = 0,

    // Query results
    result_columns: std.ArrayListUnmanaged([]const u8) = .{},
    result_rows: std.ArrayListUnmanaged([]const []const u8) = .{},
    result_selected: usize = 0,
    result_offset: usize = 0,
    result_message: []const u8 = "",

    // Status
    status_left: []const u8 = "",
    status_right: []const u8 = "",

    // Autocomplete
    completion_items: std.ArrayListUnmanaged(CompletionItem) = .{},
    completion_selected: usize = 0,
    completion_visible: bool = false,
    completion_prefix: []const u8 = "",

    fn init(allocator: std.mem.Allocator, db: *Database, db_path: []const u8) App {
        return .{
            .allocator = allocator,
            .db = db,
            .db_path = db_path,
        };
    }

    fn deinit(self: *App) void {
        self.clearSchema();
        self.clearResults();
        self.clearCompletions();
        self.input_text.deinit(self.allocator);
        if (self.status_left.len > 0) self.allocator.free(self.status_left);
        if (self.status_right.len > 0) self.allocator.free(self.status_right);
    }

    fn clearSchema(self: *App) void {
        for (self.schema_items.items) |item| self.allocator.free(item);
        self.schema_items.deinit(self.allocator);
        self.schema_items = .{};
        self.schema_table_indices.deinit(self.allocator);
        self.schema_table_indices = .{};
    }

    fn clearResults(self: *App) void {
        for (self.result_columns.items) |col| self.allocator.free(col);
        self.result_columns.deinit(self.allocator);
        self.result_columns = .{};

        for (self.result_rows.items) |row| {
            for (row) |cell| self.allocator.free(cell);
            self.allocator.free(row);
        }
        self.result_rows.deinit(self.allocator);
        self.result_rows = .{};

        if (self.result_message.len > 0) {
            self.allocator.free(self.result_message);
            self.result_message = "";
        }

        self.result_selected = 0;
        self.result_offset = 0;
    }

    fn clearCompletions(self: *App) void {
        for (self.completion_items.items) |item| {
            self.allocator.free(item.text);
            if (item.description) |desc| self.allocator.free(desc);
        }
        self.completion_items.deinit(self.allocator);
        self.completion_items = .{};
        if (self.completion_prefix.len > 0) {
            self.allocator.free(self.completion_prefix);
            self.completion_prefix = "";
        }
        self.completion_selected = 0;
        self.completion_visible = false;
    }

    fn buildCompletions(self: *App, prefix: []const u8) void {
        self.clearCompletions();

        const sql_keywords = [_][]const u8{
            "SELECT", "FROM", "WHERE", "INSERT", "INTO", "VALUES", "UPDATE", "SET",
            "DELETE", "CREATE", "TABLE", "INDEX", "DROP", "ALTER", "PRIMARY", "KEY",
            "FOREIGN", "REFERENCES", "UNIQUE", "NOT", "NULL", "DEFAULT", "CHECK",
            "AND", "OR", "IN", "LIKE", "BETWEEN", "IS", "AS", "ON", "JOIN", "LEFT",
            "RIGHT", "INNER", "OUTER", "GROUP", "BY", "HAVING", "ORDER", "ASC", "DESC",
            "LIMIT", "OFFSET", "DISTINCT", "COUNT", "SUM", "AVG", "MIN", "MAX",
            "INTEGER", "REAL", "TEXT", "BLOB", "BOOLEAN", "DATE", "TIME", "TIMESTAMP",
        };

        // Filter keywords by prefix
        for (sql_keywords) |kw| {
            if (std.ascii.startsWithIgnoreCase(kw, prefix)) {
                const text = self.allocator.dupe(u8, kw) catch continue;
                const item = CompletionItem{
                    .text = text,
                    .description = self.allocator.dupe(u8, "keyword") catch null,
                };
                self.completion_items.append(self.allocator, item) catch {
                    self.allocator.free(text);
                    continue;
                };
            }
        }

        // Add table names
        const tables = self.db.catalog.listTables(self.allocator) catch &[_][]const u8{};
        defer self.allocator.free(tables);

        for (tables) |table_name| {
            if (std.ascii.startsWithIgnoreCase(table_name, prefix)) {
                const text = self.allocator.dupe(u8, table_name) catch continue;
                const item = CompletionItem{
                    .text = text,
                    .description = self.allocator.dupe(u8, "table") catch null,
                };
                self.completion_items.append(self.allocator, item) catch {
                    self.allocator.free(text);
                    continue;
                };
            }
        }

        if (self.completion_items.items.len > 0) {
            self.completion_prefix = self.allocator.dupe(u8, prefix) catch "";
            self.completion_selected = 0;
            self.completion_visible = true;
        }
    }

    fn hideCompletions(self: *App) void {
        self.completion_visible = false;
    }

    fn selectNextCompletion(self: *App) void {
        if (self.completion_items.items.len == 0) return;
        self.completion_selected = (self.completion_selected + 1) % self.completion_items.items.len;
    }

    fn selectPrevCompletion(self: *App) void {
        if (self.completion_items.items.len == 0) return;
        if (self.completion_selected == 0) {
            self.completion_selected = self.completion_items.items.len - 1;
        } else {
            self.completion_selected -= 1;
        }
    }

    fn refreshSchema(self: *App) void {
        self.clearSchema();

        const tables = self.db.catalog.listTables(self.allocator) catch return;
        defer self.allocator.free(tables);

        for (tables) |table_name| {
            // Record that this index is a table name
            self.schema_table_indices.append(self.allocator, self.schema_items.items.len) catch continue;

            const name = self.allocator.dupe(u8, table_name) catch continue;
            self.schema_items.append(self.allocator, name) catch {
                self.allocator.free(name);
                _ = self.schema_table_indices.pop();
                continue;
            };

            // Add indented column entries
            const table_info = self.db.catalog.getTable(table_name) catch continue;
            defer table_info.deinit(self.allocator);

            for (table_info.columns) |col| {
                const label = formatColumnLabel(self.allocator, col) catch continue;
                self.schema_items.append(self.allocator, label) catch {
                    self.allocator.free(label);
                    continue;
                };
            }
        }
    }

    fn executeSQL(self: *App) void {
        if (self.input_text.items.len == 0) return;

        self.clearResults();

        var result = self.db.exec(self.input_text.items) catch |err| {
            const msg = switch (err) {
                error.ParseError => "SQL parse error.",
                error.AnalysisError => "Semantic analysis error.",
                error.PlanError => "Query planning error.",
                error.ExecutionError => "Execution error.",
                error.TableNotFound => "Table not found.",
                error.TableAlreadyExists => "Table already exists.",
                error.InvalidData => "Invalid data.",
                else => "Database error.",
            };
            self.result_message = self.allocator.dupe(u8, msg) catch "";
            self.updateStatus();
            return;
        };
        defer result.close(self.allocator);

        if (result.rows != null) {
            // Collect result rows
            var row_count: usize = 0;
            while (true) {
                const maybe_row = result.rows.?.next() catch break;
                if (maybe_row) |row| {
                    // Copy column names from first row
                    if (self.result_columns.items.len == 0 and row.columns.len > 0) {
                        for (row.columns) |col| {
                            const c = self.allocator.dupe(u8, col) catch continue;
                            self.result_columns.append(self.allocator, c) catch {
                                self.allocator.free(c);
                                continue;
                            };
                        }
                    }

                    // Convert values to strings
                    const cells = self.allocator.alloc([]const u8, row.values.len) catch continue;
                    var valid = true;
                    for (row.values, 0..) |val, i| {
                        cells[i] = valueToString(self.allocator, val) catch {
                            valid = false;
                            for (cells[0..i]) |c| self.allocator.free(c);
                            break;
                        };
                    }
                    if (!valid) {
                        self.allocator.free(cells);
                        continue;
                    }
                    self.result_rows.append(self.allocator, cells) catch {
                        for (cells) |c| self.allocator.free(c);
                        self.allocator.free(cells);
                        continue;
                    };
                    row_count += 1;
                } else break;
            }

            self.result_message = std.fmt.allocPrint(self.allocator, "{d} row(s) returned", .{row_count}) catch "";
        } else if (result.message.len > 0) {
            self.result_message = self.allocator.dupe(u8, result.message) catch "";
            // Refresh schema after DDL
            self.refreshSchema();
        }

        if (result.rows_affected > 0) {
            if (self.result_message.len > 0) self.allocator.free(self.result_message);
            self.result_message = std.fmt.allocPrint(self.allocator, "{d} row(s) affected", .{result.rows_affected}) catch "";
            // Refresh schema after DML
            self.refreshSchema();
        }

        self.updateStatus();
    }

    fn updateStatus(self: *App) void {
        if (self.status_left.len > 0) self.allocator.free(self.status_left);
        self.status_left = std.fmt.allocPrint(self.allocator, " {s} | {d} table(s)", .{
            self.db_path,
            self.schema_table_indices.items.len,
        }) catch "";

        if (self.status_right.len > 0) self.allocator.free(self.status_right);
        if (self.result_message.len > 0) {
            self.status_right = std.fmt.allocPrint(self.allocator, "{s} ", .{self.result_message}) catch "";
        } else {
            self.status_right = self.allocator.dupe(u8, "Ready ") catch "";
        }
    }

    fn handleKey(self: *App, byte: u8) void {
        // Ctrl+C / Ctrl+Q quit
        if (byte == 3 or byte == 17) {
            self.should_quit = true;
            return;
        }

        // Tab switches focus
        if (byte == 9) {
            self.focus = switch (self.focus) {
                .schema => .results,
                .results => .input,
                .input => .schema,
            };
            return;
        }

        switch (self.focus) {
            .schema => self.handleSchemaKey(byte),
            .results => self.handleResultsKey(byte),
            .input => self.handleInputKey(byte),
        }
    }

    fn handleEscapeSequence(self: *App, b2: ?u8, b3: ?u8) void {
        if (b2 == null or b2.? != '[') return;
        if (b3 == null) return;

        switch (self.focus) {
            .schema => {
                if (b3.? == 'A') { // Up
                    if (self.schema_selected > 0) self.schema_selected -= 1;
                } else if (b3.? == 'B') { // Down
                    if (self.schema_selected + 1 < self.schema_items.items.len) self.schema_selected += 1;
                }
            },
            .results => {
                if (b3.? == 'A') { // Up
                    if (self.result_selected > 0) self.result_selected -= 1;
                } else if (b3.? == 'B') { // Down
                    if (self.result_selected + 1 < self.result_rows.items.len) self.result_selected += 1;
                }
            },
            .input => {
                // If completion popup is showing, arrow up/down navigates it
                if (self.completion_visible) {
                    if (b3.? == 'A') { // Up
                        self.selectPrevCompletion();
                        return;
                    } else if (b3.? == 'B') { // Down
                        self.selectNextCompletion();
                        return;
                    }
                }

                // Otherwise, left/right moves cursor
                if (b3.? == 'C') { // Right
                    if (self.input_cursor < self.input_text.items.len) self.input_cursor += 1;
                } else if (b3.? == 'D') { // Left
                    if (self.input_cursor > 0) self.input_cursor -= 1;
                }
            },
        }
    }

    fn handleSchemaKey(self: *App, byte: u8) void {
        if (byte == '\r' or byte == '\n') {
            // Enter: find which table the selected index belongs to
            const table = schemaTableForIndex(self.schema_table_indices.items, self.schema_items.items, self.schema_selected);
            if (table) |table_name| {
                self.input_text.clearRetainingCapacity();
                self.input_cursor = 0;
                const sql = std.fmt.allocPrint(self.allocator, "SELECT * FROM {s} LIMIT 100;", .{table_name}) catch return;
                defer self.allocator.free(sql);
                self.input_text.appendSlice(self.allocator, sql) catch return;
                self.input_cursor = self.input_text.items.len;
                self.executeSQL();
            }
        }
    }

    fn handleResultsKey(self: *App, byte: u8) void {
        _ = self;
        _ = byte;
        // Results pane: arrow keys handled in handleEscapeSequence
    }

    fn handleInputKey(self: *App, byte: u8) void {
        // Ctrl+Space (0): trigger completion or hide if already showing
        if (byte == 0) {
            if (self.completion_visible) {
                self.hideCompletions();
            } else {
                const prefix = self.getCurrentWord();
                if (prefix.len > 0) {
                    self.buildCompletions(prefix);
                } else {
                    // Show all completions if no prefix
                    self.buildCompletions("");
                }
            }
            return;
        }

        // If completion popup is showing, handle navigation
        if (self.completion_visible) {
            // Ctrl+N: next item
            if (byte == 14) {
                self.selectNextCompletion();
                return;
            }

            // Ctrl+P: previous item
            if (byte == 16) {
                self.selectPrevCompletion();
                return;
            }

            // Enter: insert selected completion
            if (byte == '\r' or byte == '\n') {
                if (self.completion_selected < self.completion_items.items.len) {
                    const item = self.completion_items.items[self.completion_selected];

                    // Delete the prefix and insert the completion
                    const prefix_len = self.completion_prefix.len;
                    if (prefix_len <= self.input_cursor) {
                        const cursor_start = self.input_cursor - prefix_len;

                        // Remove prefix characters
                        var i: usize = 0;
                        while (i < prefix_len and cursor_start < self.input_text.items.len) : (i += 1) {
                            _ = self.input_text.orderedRemove(cursor_start);
                        }
                        self.input_cursor = cursor_start;

                        // Insert completion text
                        for (item.text) |ch| {
                            self.input_text.insert(self.allocator, self.input_cursor, ch) catch break;
                            self.input_cursor += 1;
                        }
                    }

                    self.hideCompletions();
                }
                return;
            }

            // Escape: hide completion
            if (byte == 27) {
                self.hideCompletions();
                return;
            }
        } else {
            // No popup showing, normal Enter executes SQL
            if (byte == '\r' or byte == '\n') {
                self.executeSQL();
                return;
            }
        }

        if (byte == 127 or byte == 8) {
            // Backspace
            if (self.input_cursor > 0) {
                _ = self.input_text.orderedRemove(self.input_cursor - 1);
                self.input_cursor -= 1;
                self.hideCompletions(); // Hide on edit
            }
            return;
        }

        // Ctrl+U: clear input
        if (byte == 21) {
            self.input_text.clearRetainingCapacity();
            self.input_cursor = 0;
            self.hideCompletions();
            return;
        }

        // Ctrl+A: move to start
        if (byte == 1) {
            self.input_cursor = 0;
            return;
        }

        // Ctrl+E: move to end
        if (byte == 5) {
            self.input_cursor = self.input_text.items.len;
            return;
        }

        // Printable character
        if (byte >= 32 and byte < 127) {
            self.input_text.insert(self.allocator, self.input_cursor, byte) catch return;
            self.input_cursor += 1;
            self.hideCompletions(); // Hide on edit
        }
    }

    fn getCurrentWord(self: *const App) []const u8 {
        if (self.input_text.items.len == 0 or self.input_cursor == 0) return "";

        // Find word start (scan backwards from cursor)
        var start: usize = self.input_cursor;
        while (start > 0) {
            const ch = self.input_text.items[start - 1];
            if (!std.ascii.isAlphanumeric(ch) and ch != '_') break;
            start -= 1;
        }

        return self.input_text.items[start..self.input_cursor];
    }
};

// ── Schema Helpers ───────────────────────────────────────────────────

const ColumnInfo = silica.catalog.ColumnInfo;
const ColumnType = silica.catalog.ColumnType;

/// Find the table name for a given flat-list index.
/// Uses schema_table_indices to find the nearest table header at or before `idx`.
fn schemaTableForIndex(table_indices: []const usize, items: []const []const u8, idx: usize) ?[]const u8 {
    if (table_indices.len == 0 or items.len == 0) return null;
    if (idx >= items.len) return null;

    // Binary search for the largest table index <= idx
    var best: ?usize = null;
    for (table_indices) |ti| {
        if (ti <= idx) {
            best = ti;
        } else {
            break; // table_indices are in ascending order
        }
    }

    if (best) |b| return items[b];
    return null;
}

/// Format a column label for display in the schema sidebar.
/// Example: "  id INTEGER PK" or "  name TEXT NOT NULL"
fn formatColumnLabel(allocator: std.mem.Allocator, col: ColumnInfo) ![]const u8 {
    const type_str = switch (col.column_type) {
        .integer => "INTEGER",
        .real => "REAL",
        .text => "TEXT",
        .blob => "BLOB",
        .boolean => "BOOLEAN",
        .date => "DATE",
        .time => "TIME",
        .timestamp => "TIMESTAMP",
        .interval => "INTERVAL",
        .numeric => "NUMERIC",
        .uuid => "UUID",
        .array => "ARRAY",
        .json => "JSON",
        .jsonb => "JSONB",
        .tsvector => "TSVECTOR",
        .tsquery => "TSQUERY",
        .untyped => "",
    };

    // Build suffix for constraints
    var suffix_buf: [64]u8 = undefined;
    var suffix_len: usize = 0;

    if (col.flags.primary_key) {
        const pk = " PK";
        @memcpy(suffix_buf[suffix_len..][0..pk.len], pk);
        suffix_len += pk.len;
    }
    if (col.flags.not_null) {
        const nn = " NN";
        @memcpy(suffix_buf[suffix_len..][0..nn.len], nn);
        suffix_len += nn.len;
    }
    if (col.flags.unique) {
        const uq = " UQ";
        @memcpy(suffix_buf[suffix_len..][0..uq.len], uq);
        suffix_len += uq.len;
    }

    if (type_str.len > 0) {
        return std.fmt.allocPrint(allocator, "  {s} {s}{s}", .{ col.name, type_str, suffix_buf[0..suffix_len] });
    } else {
        return std.fmt.allocPrint(allocator, "  {s}{s}", .{ col.name, suffix_buf[0..suffix_len] });
    }
}

// ── Value Conversion ─────────────────────────────────────────────────

fn valueToString(allocator: std.mem.Allocator, val: Value) ![]const u8 {
    return switch (val) {
        .integer => |v| try std.fmt.allocPrint(allocator, "{d}", .{v}),
        .real => |v| try std.fmt.allocPrint(allocator, "{d}", .{v}),
        .text => |v| try allocator.dupe(u8, v),
        .blob => |v| blk: {
            const hex_len = 2 + v.len * 2 + 1;
            const hex_buf = try allocator.alloc(u8, hex_len);
            hex_buf[0] = 'X';
            hex_buf[1] = '\'';
            const digits = "0123456789abcdef";
            for (v, 0..) |byte, i| {
                hex_buf[2 + i * 2] = digits[byte >> 4];
                hex_buf[2 + i * 2 + 1] = digits[byte & 0x0f];
            }
            hex_buf[hex_len - 1] = '\'';
            break :blk hex_buf;
        },
        .boolean => |v| try allocator.dupe(u8, if (v) "TRUE" else "FALSE"),
        .date => |v| try executor.formatDate(allocator, v),
        .time => |v| try executor.formatTime(allocator, v),
        .timestamp => |v| try executor.formatTimestamp(allocator, v),
        .interval => |v| try executor.formatInterval(allocator, v),
        .numeric => |v| try executor.formatNumeric(allocator, v),
        .uuid => |v| try executor.formatUuid(allocator, v),
        .array => |v| try executor.formatArray(allocator, v),
        .tsvector => |v| try allocator.dupe(u8, v),
        .tsquery => |v| try allocator.dupe(u8, v),
        .null_value => try allocator.dupe(u8, "NULL"),
    };
}

// ── Rendering ────────────────────────────────────────────────────────

fn renderUI(app: *App, buf: *tui.Buffer, area: tui.Rect) !void {
    const allocator = app.allocator;

    // Main layout: vertical split — content area + status bar (1 row)
    const main_chunks = try tui.layout.split(allocator, .vertical, area, &.{
        .{ .min = 3 },
        .{ .length = 1 },
    });
    defer allocator.free(main_chunks);

    const content_area = main_chunks[0];
    const status_area = main_chunks[1];

    // Content: horizontal split — schema sidebar (25%) + right panel (75%)
    const h_chunks = try tui.layout.split(allocator, .horizontal, content_area, &.{
        .{ .percentage = 25 },
        .{ .percentage = 75 },
    });
    defer allocator.free(h_chunks);

    const schema_area = h_chunks[0];
    const right_area = h_chunks[1];

    // Right panel: vertical split — results + input (3 rows)
    const right_chunks = try tui.layout.split(allocator, .vertical, right_area, &.{
        .{ .min = 3 },
        .{ .length = 3 },
    });
    defer allocator.free(right_chunks);

    const results_area = right_chunks[0];
    const input_area = right_chunks[1];

    // Render schema tree
    renderSchemaTree(app, buf, schema_area);

    // Render results table
    renderResultsTable(app, buf, results_area);

    // Render SQL input
    renderSQLInput(app, buf, input_area);

    // Render status bar
    renderStatusBar(app, buf, status_area);
}

fn isTableIndex(table_indices: []const usize, idx: usize) bool {
    for (table_indices) |ti| {
        if (ti == idx) return true;
        if (ti > idx) break;
    }
    return false;
}

fn renderSchemaTree(app: *App, buf: *tui.Buffer, area: tui.Rect) void {
    const is_focused = app.focus == .schema;
    const border_style: tui.Style = if (is_focused) .{ .fg = .cyan, .bold = true } else .{};

    const block = tui.widgets.Block.init()
        .withBorderStyle(border_style)
        .withTitle("Schema", .top_left)
        .withTitleStyle(if (is_focused) tui.Style{ .fg = .cyan, .bold = true } else .{});

    block.render(buf, area);
    const inner = block.inner(area);
    if (inner.width == 0 or inner.height == 0) return;

    if (app.schema_items.items.len == 0) {
        buf.setString(inner.x, inner.y, "(no tables)", .{ .fg = .bright_black });
        return;
    }

    // Adjust offset for scrolling
    if (app.schema_selected < app.schema_offset) {
        app.schema_offset = app.schema_selected;
    } else if (app.schema_selected >= app.schema_offset + inner.height) {
        app.schema_offset = app.schema_selected - inner.height + 1;
    }

    // Render visible items manually with different styles for tables vs columns
    var row: u16 = 0;
    var idx = app.schema_offset;
    while (row < inner.height and idx < app.schema_items.items.len) : ({ row += 1; idx += 1; }) {
        const item = app.schema_items.items[idx];
        const is_table = isTableIndex(app.schema_table_indices.items, idx);
        const is_selected = is_focused and idx == app.schema_selected;

        const item_style: tui.Style = if (is_selected)
            .{ .fg = .cyan, .bold = true, .reverse = true }
        else if (is_table)
            .{ .fg = .yellow, .bold = true }
        else
            .{ .fg = .bright_black };

        // Clear the row first
        var cx: u16 = 0;
        while (cx < inner.width) : (cx += 1) {
            buf.setChar(inner.x + cx, inner.y + row, ' ', if (is_selected) item_style else .{});
        }

        // Render the item text
        var x: u16 = 0;
        for (item) |c| {
            if (x >= inner.width) break;
            buf.setChar(inner.x + x, inner.y + row, c, item_style);
            x += 1;
        }
    }
}

fn renderResultsTable(app: *App, buf: *tui.Buffer, area: tui.Rect) void {
    const is_focused = app.focus == .results;
    const border_style: tui.Style = if (is_focused) .{ .fg = .cyan, .bold = true } else .{};

    const block = tui.widgets.Block.init()
        .withBorderStyle(border_style)
        .withTitle("Results", .top_left)
        .withTitleStyle(if (is_focused) tui.Style{ .fg = .cyan, .bold = true } else .{});

    if (app.result_columns.items.len == 0) {
        block.render(buf, area);
        const inner = block.inner(area);
        if (inner.width > 0 and inner.height > 0) {
            const msg = if (app.result_message.len > 0) app.result_message else "Execute a query to see results";
            buf.setString(inner.x, inner.y, msg, .{ .fg = .bright_black });
        }
        return;
    }

    // Build columns with auto-width
    var columns_buf: [16]tui.widgets.Column = undefined;
    const num_cols = @min(app.result_columns.items.len, 16);
    for (app.result_columns.items[0..num_cols], 0..) |col_name, i| {
        columns_buf[i] = .{
            .title = col_name,
            .width = tui.widgets.ColumnWidth.ofMin(@intCast(@min(col_name.len + 2, 30))),
            .alignment = .left,
        };
    }

    const table = tui.widgets.Table.init(
        columns_buf[0..num_cols],
        @ptrCast(app.result_rows.items),
    )
        .withBlock(block)
        .withSelected(if (is_focused) @as(?usize, app.result_selected) else null)
        .withOffset(app.result_offset)
        .withHeaderStyle(.{ .bold = true, .underline = true })
        .withSelectedStyle(.{ .bg = .{ .indexed = 236 } })
        .withColumnSpacing(2);

    table.render(buf, area);
}

fn renderSQLInput(app: *App, buf: *tui.Buffer, area: tui.Rect) void {
    const is_focused = app.focus == .input;
    const border_style: tui.Style = if (is_focused) .{ .fg = .cyan, .bold = true } else .{};

    const block = tui.widgets.Block.init()
        .withBorderStyle(border_style)
        .withTitle("SQL", .top_left)
        .withTitleStyle(if (is_focused) tui.Style{ .fg = .cyan, .bold = true } else .{});

    block.render(buf, area);
    const inner = block.inner(area);
    if (inner.width == 0 or inner.height == 0) return;

    // Determine display text and style
    const has_text = app.input_text.items.len > 0;
    const display_text = if (has_text) app.input_text.items else "Type SQL here, press Enter to execute...";
    const display_style: tui.Style = if (has_text) .{} else .{ .fg = .bright_black };

    // Render text
    var x = inner.x;
    for (display_text) |c| {
        if (x >= inner.x + inner.width) break;
        buf.setChar(x, inner.y, c, display_style);
        x += 1;
    }

    // Render cursor (reverse style at cursor position)
    const cursor_x: u16 = blk: {
        if (is_focused and has_text) {
            const cx = inner.x + @as(u16, @intCast(@min(app.input_cursor, inner.width - 1)));
            const cursor_char: u8 = if (app.input_cursor < app.input_text.items.len)
                app.input_text.items[app.input_cursor]
            else
                ' ';
            buf.setChar(cx, inner.y, cursor_char, .{ .reverse = true });
            break :blk cx;
        } else if (is_focused) {
            // Empty input — cursor on first cell
            buf.setChar(inner.x, inner.y, ' ', .{ .reverse = true });
            break :blk inner.x;
        }
        break :blk 0;
    };

    // Render completion popup if active
    if (app.completion_visible and app.completion_items.items.len > 0) {
        renderCompletionPopup(app, buf, cursor_x, inner.y);
    }
}

fn renderCompletionPopup(app: *App, buf: *tui.Buffer, cursor_x: u16, cursor_y: u16) void {
    const items = app.completion_items.items;
    if (items.len == 0) return;

    const max_visible: usize = 10;
    const visible_count = @min(items.len, max_visible);

    // Calculate popup dimensions
    var max_width: usize = 12; // "Completions" title
    for (items) |item| {
        var item_width = item.text.len;
        if (item.description) |desc| {
            item_width += 3 + desc.len; // " - description"
        }
        if (item_width > max_width) max_width = item_width;
    }

    const width: u16 = @min(@as(u16, @intCast(max_width + 4)), 80);
    const height: u16 = @intCast(visible_count + 2); // +2 for borders

    // Calculate popup position (below cursor, with bounds checking)
    const popup_x: u16 = @intCast(@max(0, @min(
        @as(i32, cursor_x),
        @as(i32, buf.width) - @as(i32, width),
    )));

    const popup_y: u16 = @intCast(@max(0, @min(
        @as(i32, cursor_y) + 1,
        @as(i32, buf.height) - @as(i32, height),
    )));

    const popup_area = tui.Rect{
        .x = popup_x,
        .y = popup_y,
        .width = width,
        .height = height,
    };

    // Draw border
    const block = tui.widgets.Block.init()
        .withTitle("Completions", .top_left)
        .withBorderStyle(.{ .fg = .cyan });
    block.render(buf, popup_area);
    const inner = block.inner(popup_area);

    // Render visible items
    const scroll_offset: usize = if (app.completion_selected >= max_visible)
        app.completion_selected - max_visible + 1
    else
        0;

    const visible_end = @min(scroll_offset + max_visible, items.len);
    var row: u16 = 0;

    // Track selected item for tooltip
    var selected_item_area: ?tui.Rect = null;
    var selected_keyword_help: ?[]const u8 = null;

    for (items[scroll_offset..visible_end], scroll_offset..) |item, i| {
        if (row >= inner.height) break;

        const is_selected = (i == app.completion_selected);
        const item_style: tui.Style = if (is_selected)
            .{ .fg = .cyan, .reverse = true }
        else
            .{};

        // Clear the row
        var cx: u16 = 0;
        while (cx < inner.width) : (cx += 1) {
            buf.setChar(inner.x + cx, inner.y + row, ' ', if (is_selected) item_style else .{});
        }

        // Render item text
        var x: u16 = 0;
        for (item.text) |c| {
            if (x >= inner.width) break;
            buf.setChar(inner.x + x, inner.y + row, c, item_style);
            x += 1;
        }

        // Render description if present
        if (item.description) |desc| {
            if (x + 3 < inner.width) {
                buf.setString(inner.x + x, inner.y + row, " - ", .{ .fg = .bright_black });
                x += 3;
                for (desc) |c| {
                    if (x >= inner.width) break;
                    buf.setChar(inner.x + x, inner.y + row, c, .{ .fg = .bright_black });
                    x += 1;
                }
            }
        }

        // If this is the selected item, store its area and check for help text
        if (is_selected) {
            selected_item_area = tui.Rect{
                .x = inner.x,
                .y = inner.y + row,
                .width = inner.width,
                .height = 1,
            };
            // Check for SQL keyword help first
            if (getSqlKeywordHelp(item.text)) |kw_help| {
                selected_keyword_help = kw_help;
            } else if (item.description) |desc| {
                // Check if this is a table name — show table stats
                if (std.mem.eql(u8, desc, "table")) {
                    selected_keyword_help = getTableHelp(app.db, item.text);
                }
            }
        }

        row += 1;
    }

    // Render tooltip for selected item
    if (selected_keyword_help) |help_text| {
        if (selected_item_area) |item_area| {
            var tooltip = tui.widgets.Tooltip.init(help_text)
                .withPosition(.right)
                .withStyle(.{ .fg = .black, .bg = .bright_yellow })
                .withArrow(true);
            tooltip.show(item_area);
            tooltip.render(buf.*, tui.Rect{
                .x = 0,
                .y = 0,
                .width = buf.width,
                .height = buf.height,
            });
        }
    }
}

fn renderStatusBar(app: *App, buf: *tui.Buffer, area: tui.Rect) void {
    if (area.width == 0 or area.height == 0) return;

    const bar_style = tui.Style{ .bg = .bright_black };

    // Fill the entire bar with background
    var fill_x = area.x;
    while (fill_x < area.x + area.width) : (fill_x += 1) {
        buf.setChar(fill_x, area.y, ' ', bar_style);
    }

    // Left: [PANE] db_path | N table(s)
    const pane_name = switch (app.focus) {
        .schema => "SCHEMA",
        .results => "RESULTS",
        .input => "SQL",
    };

    var x = area.x;
    const pane_style = tui.Style{ .fg = .black, .bg = .cyan, .bold = true };
    for (pane_name) |c| {
        if (x >= area.x + area.width) break;
        buf.setChar(x, area.y, c, pane_style);
        x += 1;
    }

    for (app.status_left) |c| {
        if (x >= area.x + area.width) break;
        buf.setChar(x, area.y, c, bar_style);
        x += 1;
    }

    // Right: status message
    if (app.status_right.len > 0) {
        const right_start = if (area.width > app.status_right.len)
            area.x + area.width - @as(u16, @intCast(app.status_right.len))
        else
            area.x;
        var rx = right_start;
        for (app.status_right) |c| {
            if (rx >= area.x + area.width) break;
            buf.setChar(rx, area.y, c, bar_style);
            rx += 1;
        }
    }

    // Center: Tab:switch Enter:exec Ctrl+C:quit
    const center_text = "Tab:switch  Enter:exec  Ctrl+C:quit";
    const center_start = if (area.width > center_text.len)
        area.x + (area.width - @as(u16, @intCast(center_text.len))) / 2
    else
        area.x;
    var cx = center_start;
    for (center_text) |c| {
        if (cx >= area.x + area.width) break;
        buf.setChar(cx, area.y, c, bar_style);
        cx += 1;
    }
}

// ── Local renderDiff (workaround for sailor#5: adaptToNewApi bug) ────

fn writeU16(writer: anytype, val: u16) !void {
    var buf: [5]u8 = undefined; // max 5 digits for u16
    var n = val;
    var len: usize = 0;
    if (n == 0) {
        buf[0] = '0';
        len = 1;
    } else {
        while (n > 0) : (len += 1) {
            buf[len] = @intCast('0' + (n % 10));
            n /= 10;
        }
        // Reverse
        var i: usize = 0;
        var j: usize = len - 1;
        while (i < j) {
            const tmp = buf[i];
            buf[i] = buf[j];
            buf[j] = tmp;
            i += 1;
            j -= 1;
        }
    }
    try writer.writeAll(buf[0..len]);
}

fn writeColorFg(writer: anytype, color: tui.Color) !void {
    switch (color) {
        .reset => try writer.writeAll("\x1b[39m"),
        .black => try writer.writeAll("\x1b[30m"),
        .red => try writer.writeAll("\x1b[31m"),
        .green => try writer.writeAll("\x1b[32m"),
        .yellow => try writer.writeAll("\x1b[33m"),
        .blue => try writer.writeAll("\x1b[34m"),
        .magenta => try writer.writeAll("\x1b[35m"),
        .cyan => try writer.writeAll("\x1b[36m"),
        .white => try writer.writeAll("\x1b[37m"),
        .bright_black => try writer.writeAll("\x1b[90m"),
        .bright_red => try writer.writeAll("\x1b[91m"),
        .bright_green => try writer.writeAll("\x1b[92m"),
        .bright_yellow => try writer.writeAll("\x1b[93m"),
        .bright_blue => try writer.writeAll("\x1b[94m"),
        .bright_magenta => try writer.writeAll("\x1b[95m"),
        .bright_cyan => try writer.writeAll("\x1b[96m"),
        .bright_white => try writer.writeAll("\x1b[97m"),
        .indexed => |idx| {
            try writer.writeAll("\x1b[38;5;");
            try writeU16(writer, @intCast(idx));
            try writer.writeAll("m");
        },
        .rgb => |c| {
            try writer.writeAll("\x1b[38;2;");
            try writeU16(writer, @intCast(c.r));
            try writer.writeByte(';');
            try writeU16(writer, @intCast(c.g));
            try writer.writeByte(';');
            try writeU16(writer, @intCast(c.b));
            try writer.writeAll("m");
        },
    }
}

fn writeColorBg(writer: anytype, color: tui.Color) !void {
    switch (color) {
        .reset => try writer.writeAll("\x1b[49m"),
        .black => try writer.writeAll("\x1b[40m"),
        .red => try writer.writeAll("\x1b[41m"),
        .green => try writer.writeAll("\x1b[42m"),
        .yellow => try writer.writeAll("\x1b[43m"),
        .blue => try writer.writeAll("\x1b[44m"),
        .magenta => try writer.writeAll("\x1b[45m"),
        .cyan => try writer.writeAll("\x1b[46m"),
        .white => try writer.writeAll("\x1b[47m"),
        .bright_black => try writer.writeAll("\x1b[100m"),
        .bright_red => try writer.writeAll("\x1b[101m"),
        .bright_green => try writer.writeAll("\x1b[102m"),
        .bright_yellow => try writer.writeAll("\x1b[103m"),
        .bright_blue => try writer.writeAll("\x1b[104m"),
        .bright_magenta => try writer.writeAll("\x1b[105m"),
        .bright_cyan => try writer.writeAll("\x1b[106m"),
        .bright_white => try writer.writeAll("\x1b[107m"),
        .indexed => |idx| {
            try writer.writeAll("\x1b[48;5;");
            try writeU16(writer, @intCast(idx));
            try writer.writeAll("m");
        },
        .rgb => |c| {
            try writer.writeAll("\x1b[48;2;");
            try writeU16(writer, @intCast(c.r));
            try writer.writeByte(';');
            try writeU16(writer, @intCast(c.g));
            try writer.writeByte(';');
            try writeU16(writer, @intCast(c.b));
            try writer.writeAll("m");
        },
    }
}

fn applyStyle(writer: anytype, style: tui.Style) !void {
    if (style.fg) |fg| try writeColorFg(writer, fg);
    if (style.bg) |bg| try writeColorBg(writer, bg);
    if (style.bold) try writer.writeAll("\x1b[1m");
    if (style.dim) try writer.writeAll("\x1b[2m");
    if (style.italic) try writer.writeAll("\x1b[3m");
    if (style.underline) try writer.writeAll("\x1b[4m");
    if (style.blink) try writer.writeAll("\x1b[5m");
    if (style.reverse) try writer.writeAll("\x1b[7m");
    if (style.strikethrough) try writer.writeAll("\x1b[9m");
}

/// Local renderDiff that avoids std.fmt.format (sailor#5 workaround)
fn localRenderDiff(diff_ops: []const tui.buffer.DiffOp, writer: anytype) !void {
    var current_style: ?tui.Style = null;
    var current_x: ?u16 = null;
    var current_y: ?u16 = null;

    for (diff_ops) |op| {
        // Move cursor if needed
        if (current_x == null or current_y == null or
            current_x.? != op.x or current_y.? != op.y)
        {
            // ANSI cursor position (1-indexed): ESC[row;colH
            try writer.writeAll("\x1b[");
            try writeU16(writer, op.y + 1);
            try writer.writeByte(';');
            try writeU16(writer, op.x + 1);
            try writer.writeByte('H');
            current_x = op.x;
            current_y = op.y;
        }

        // Apply style if changed
        const has_style = op.cell.style.fg != null or
            op.cell.style.bg != null or
            op.cell.style.bold or
            op.cell.style.dim or
            op.cell.style.italic or
            op.cell.style.underline or
            op.cell.style.blink or
            op.cell.style.reverse or
            op.cell.style.strikethrough;

        if (has_style) {
            if (current_style == null or !std.meta.eql(current_style.?, op.cell.style)) {
                try writer.writeAll("\x1b[0m"); // reset
                try applyStyle(writer, op.cell.style);
                current_style = op.cell.style;
            }
        } else {
            if (current_style != null) {
                try writer.writeAll("\x1b[0m"); // reset
                current_style = null;
            }
        }

        // Write character
        var char_buf: [4]u8 = undefined;
        const len = std.unicode.utf8Encode(op.cell.char, &char_buf) catch 1;
        if (len == 1 and op.cell.char < 128) {
            try writer.writeByte(@intCast(op.cell.char));
        } else {
            try writer.writeAll(char_buf[0..len]);
        }

        current_x = op.x + 1; // Advance cursor position
    }

    // Reset style at end
    if (current_style != null) {
        try writer.writeAll("\x1b[0m");
    }
}

// ── Public Entry Point ───────────────────────────────────────────────

pub fn run(allocator: std.mem.Allocator, db: *Database, db_path: []const u8) !void {
    const builtin = @import("builtin");
    if (comptime builtin.os.tag == .windows) {
        return error.NotATty;
    }

    // Initialize terminal
    var term = try tui.Terminal.init(allocator);
    defer term.deinit();

    // Enter raw mode
    var raw = try sailor.term.RawMode.enter(std.posix.STDIN_FILENO);
    defer raw.deinit();

    // Hide cursor & enter alternate screen
    const stdout = std.fs.File.stdout();
    var stdout_buf: [4096]u8 = undefined;
    var writer = stdout.writer(&stdout_buf);

    writer.interface.writeAll("\x1b[?1049h") catch {}; // enter alt screen
    writer.interface.writeAll("\x1b[?25l") catch {}; // hide cursor
    writer.interface.flush() catch {};

    defer {
        var dw = stdout.writer(&stdout_buf);
        dw.interface.writeAll("\x1b[?25h") catch {}; // show cursor
        dw.interface.writeAll("\x1b[?1049l") catch {}; // leave alt screen
        dw.interface.flush() catch {};
    }

    // Initialize app state
    var app = App.init(allocator, db, db_path);
    defer app.deinit();
    app.refreshSchema();
    app.updateStatus();

    // Main event loop
    while (!app.should_quit) {
        // Clear buffer
        term.clear();

        // Render UI
        try renderUI(&app, &term.current, term.size());

        // Diff and render
        const diff_ops = try tui.buffer.diff(allocator, term.previous, term.current);
        defer allocator.free(diff_ops);

        var render_writer = stdout.writer(&stdout_buf);
        try localRenderDiff(diff_ops, &render_writer.interface);
        render_writer.interface.flush() catch {};

        // Swap buffers
        const tmp_cells = term.current.cells;
        term.current.cells = term.previous.cells;
        term.previous.cells = tmp_cells;

        // Read input
        const byte = sailor.term.readByte(50) catch null;
        if (byte) |b| {
            if (b == 27) {
                // Escape sequence
                const b2 = sailor.term.readByte(20) catch null;
                const b3 = sailor.term.readByte(20) catch null;
                app.handleEscapeSequence(b2, b3);
            } else {
                app.handleKey(b);
            }
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────

test "App init and deinit" {
    // App.init doesn't require a real database - test basic struct creation
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    try std.testing.expectEqual(Pane.input, app.focus);
    try std.testing.expect(!app.should_quit);
}

test "App handleKey Tab cycles focus" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    try std.testing.expectEqual(Pane.input, app.focus);

    app.handleKey(9); // Tab
    try std.testing.expectEqual(Pane.schema, app.focus);

    app.handleKey(9); // Tab
    try std.testing.expectEqual(Pane.results, app.focus);

    app.handleKey(9); // Tab
    try std.testing.expectEqual(Pane.input, app.focus);
}

test "App handleKey Ctrl+C quits" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    try std.testing.expect(!app.should_quit);
    app.handleKey(3); // Ctrl+C
    try std.testing.expect(app.should_quit);
}

test "App handleKey Ctrl+Q quits" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    app.handleKey(17); // Ctrl+Q
    try std.testing.expect(app.should_quit);
}

test "App input text editing" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .input,
    };
    defer app.deinit();

    // Type "SELECT"
    for ("SELECT") |c| {
        app.handleKey(c);
    }
    try std.testing.expectEqualStrings("SELECT", app.input_text.items);
    try std.testing.expectEqual(@as(usize, 6), app.input_cursor);

    // Backspace
    app.handleKey(127);
    try std.testing.expectEqualStrings("SELEC", app.input_text.items);
    try std.testing.expectEqual(@as(usize, 5), app.input_cursor);

    // Ctrl+A move to start
    app.handleKey(1);
    try std.testing.expectEqual(@as(usize, 0), app.input_cursor);

    // Ctrl+E move to end
    app.handleKey(5);
    try std.testing.expectEqual(@as(usize, 5), app.input_cursor);

    // Ctrl+U clear
    app.handleKey(21);
    try std.testing.expectEqual(@as(usize, 0), app.input_text.items.len);
    try std.testing.expectEqual(@as(usize, 0), app.input_cursor);
}

test "App schema navigation" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .schema,
    };
    defer app.deinit();

    // Add mock schema items
    const t1 = try allocator.dupe(u8, "users");
    try app.schema_items.append(allocator, t1);
    const t2 = try allocator.dupe(u8, "orders");
    try app.schema_items.append(allocator, t2);

    try std.testing.expectEqual(@as(usize, 0), app.schema_selected);

    // Down arrow via escape sequence
    app.handleEscapeSequence('[', 'B');
    try std.testing.expectEqual(@as(usize, 1), app.schema_selected);

    // Up arrow
    app.handleEscapeSequence('[', 'A');
    try std.testing.expectEqual(@as(usize, 0), app.schema_selected);

    // Up at 0 stays at 0
    app.handleEscapeSequence('[', 'A');
    try std.testing.expectEqual(@as(usize, 0), app.schema_selected);
}

test "App results navigation" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .results,
    };
    defer app.deinit();

    // Add mock result rows
    const row1 = try allocator.alloc([]const u8, 1);
    row1[0] = try allocator.dupe(u8, "Alice");
    try app.result_rows.append(allocator, row1);

    const row2 = try allocator.alloc([]const u8, 1);
    row2[0] = try allocator.dupe(u8, "Bob");
    try app.result_rows.append(allocator, row2);

    // Down
    app.handleEscapeSequence('[', 'B');
    try std.testing.expectEqual(@as(usize, 1), app.result_selected);

    // Down at last row stays
    app.handleEscapeSequence('[', 'B');
    try std.testing.expectEqual(@as(usize, 1), app.result_selected);
}

test "App input cursor movement with arrows" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .input,
    };
    defer app.deinit();

    // Type "abc"
    for ("abc") |c| app.handleKey(c);

    // Move left
    app.handleEscapeSequence('[', 'D');
    try std.testing.expectEqual(@as(usize, 2), app.input_cursor);

    // Move right
    app.handleEscapeSequence('[', 'C');
    try std.testing.expectEqual(@as(usize, 3), app.input_cursor);

    // Right at end stays
    app.handleEscapeSequence('[', 'C');
    try std.testing.expectEqual(@as(usize, 3), app.input_cursor);
}

test "valueToString all types" {
    const allocator = std.testing.allocator;

    {
        const s = try valueToString(allocator, .{ .integer = 42 });
        defer allocator.free(s);
        try std.testing.expectEqualStrings("42", s);
    }
    {
        const s = try valueToString(allocator, .{ .text = "hello" });
        defer allocator.free(s);
        try std.testing.expectEqualStrings("hello", s);
    }
    {
        const s = try valueToString(allocator, .{ .boolean = true });
        defer allocator.free(s);
        try std.testing.expectEqualStrings("TRUE", s);
    }
    {
        const s = try valueToString(allocator, .null_value);
        defer allocator.free(s);
        try std.testing.expectEqualStrings("NULL", s);
    }
}

test "renderUI does not crash with empty state" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    var buf = try tui.Buffer.init(allocator, 80, 24);
    defer buf.deinit();

    const area = tui.Rect.new(0, 0, 80, 24);
    try renderUI(&app, &buf, area);

    // Check that something was rendered (borders at least)
    const corner = buf.getConst(0, 0);
    try std.testing.expect(corner != null);
}

test "renderUI with populated data" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "mydb.db",
    };
    defer app.deinit();

    // Add schema items
    const t1 = try allocator.dupe(u8, "users");
    try app.schema_items.append(allocator, t1);

    // Add result columns
    const c1 = try allocator.dupe(u8, "name");
    try app.result_columns.append(allocator, c1);

    // Add result row
    const cells = try allocator.alloc([]const u8, 1);
    cells[0] = try allocator.dupe(u8, "Alice");
    try app.result_rows.append(allocator, cells);

    // Add input text
    try app.input_text.appendSlice(allocator, "SELECT * FROM users;");
    app.input_cursor = 20;

    app.status_left = try allocator.dupe(u8, " mydb.db | 1 table(s)");
    app.status_right = try allocator.dupe(u8, "Ready ");

    var buf = try tui.Buffer.init(allocator, 80, 24);
    defer buf.deinit();

    const area = tui.Rect.new(0, 0, 80, 24);
    try renderUI(&app, &buf, area);

    // Verify blocks have borders
    const top_left = buf.getConst(0, 0);
    try std.testing.expect(top_left != null);
    try std.testing.expectEqual(@as(u21, '┌'), top_left.?.char);
}

test "Pane enum values" {
    try std.testing.expectEqual(Pane.schema, Pane.schema);
    try std.testing.expectEqual(Pane.results, Pane.results);
    try std.testing.expectEqual(Pane.input, Pane.input);
}

test "App clearResults cleans up memory" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    // Add some result data
    const col = try allocator.dupe(u8, "name");
    try app.result_columns.append(allocator, col);

    const cells = try allocator.alloc([]const u8, 1);
    cells[0] = try allocator.dupe(u8, "test");
    try app.result_rows.append(allocator, cells);

    app.result_message = try allocator.dupe(u8, "1 row(s)");
    app.result_selected = 5;
    app.result_offset = 3;

    app.clearResults();

    try std.testing.expectEqual(@as(usize, 0), app.result_columns.items.len);
    try std.testing.expectEqual(@as(usize, 0), app.result_rows.items.len);
    try std.testing.expectEqual(@as(usize, 0), app.result_selected);
    try std.testing.expectEqual(@as(usize, 0), app.result_offset);
    try std.testing.expectEqual(@as(usize, 0), app.result_message.len);
}

test "handleEscapeSequence ignores invalid sequences" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .schema,
    };
    defer app.deinit();

    // null bytes
    app.handleEscapeSequence(null, null);
    // Invalid b2
    app.handleEscapeSequence('X', 'A');
    // null b3
    app.handleEscapeSequence('[', null);

    // Should not crash and state should be unchanged
    try std.testing.expectEqual(@as(usize, 0), app.schema_selected);
}

test "input insert in middle" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .input,
    };
    defer app.deinit();

    // Type "AC"
    app.handleKey('A');
    app.handleKey('C');

    // Move cursor left to between A and C
    app.handleEscapeSequence('[', 'D');
    try std.testing.expectEqual(@as(usize, 1), app.input_cursor);

    // Insert 'B'
    app.handleKey('B');
    try std.testing.expectEqualStrings("ABC", app.input_text.items);
    try std.testing.expectEqual(@as(usize, 2), app.input_cursor);
}

test "backspace at position 0 does nothing" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .input,
    };
    defer app.deinit();

    app.handleKey(127); // Backspace on empty
    try std.testing.expectEqual(@as(usize, 0), app.input_text.items.len);
    try std.testing.expectEqual(@as(usize, 0), app.input_cursor);
}

test "schemaTableForIndex finds correct table" {
    // Flat list: [0]="users", [1]="  id INTEGER PK", [2]="  name TEXT", [3]="orders", [4]="  total REAL"
    const items = [_][]const u8{ "users", "  id INTEGER PK", "  name TEXT", "orders", "  total REAL" };
    const table_indices = [_]usize{ 0, 3 };

    // Index 0 → "users" (table header itself)
    try std.testing.expectEqualStrings("users", schemaTableForIndex(&table_indices, &items, 0).?);
    // Index 1 → "users" (column of users)
    try std.testing.expectEqualStrings("users", schemaTableForIndex(&table_indices, &items, 1).?);
    // Index 2 → "users" (column of users)
    try std.testing.expectEqualStrings("users", schemaTableForIndex(&table_indices, &items, 2).?);
    // Index 3 → "orders" (table header)
    try std.testing.expectEqualStrings("orders", schemaTableForIndex(&table_indices, &items, 3).?);
    // Index 4 → "orders" (column of orders)
    try std.testing.expectEqualStrings("orders", schemaTableForIndex(&table_indices, &items, 4).?);
    // Out of bounds
    try std.testing.expect(schemaTableForIndex(&table_indices, &items, 5) == null);
    // Empty
    try std.testing.expect(schemaTableForIndex(&[_]usize{}, &items, 0) == null);
}

test "formatColumnLabel basic types and constraints" {
    const allocator = std.testing.allocator;

    {
        const col = ColumnInfo{ .name = "id", .column_type = .integer, .flags = .{ .primary_key = true } };
        const label = try formatColumnLabel(allocator, col);
        defer allocator.free(label);
        try std.testing.expectEqualStrings("  id INTEGER PK", label);
    }
    {
        const col = ColumnInfo{ .name = "name", .column_type = .text, .flags = .{ .not_null = true } };
        const label = try formatColumnLabel(allocator, col);
        defer allocator.free(label);
        try std.testing.expectEqualStrings("  name TEXT NN", label);
    }
    {
        const col = ColumnInfo{ .name = "email", .column_type = .text, .flags = .{ .not_null = true, .unique = true } };
        const label = try formatColumnLabel(allocator, col);
        defer allocator.free(label);
        try std.testing.expectEqualStrings("  email TEXT NN UQ", label);
    }
    {
        const col = ColumnInfo{ .name = "data", .column_type = .untyped, .flags = .{} };
        const label = try formatColumnLabel(allocator, col);
        defer allocator.free(label);
        try std.testing.expectEqualStrings("  data", label);
    }
    {
        const col = ColumnInfo{ .name = "config", .column_type = .json, .flags = .{} };
        const label = try formatColumnLabel(allocator, col);
        defer allocator.free(label);
        try std.testing.expectEqualStrings("  config JSON", label);
    }
    {
        const col = ColumnInfo{ .name = "metadata", .column_type = .jsonb, .flags = .{ .not_null = true } };
        const label = try formatColumnLabel(allocator, col);
        defer allocator.free(label);
        try std.testing.expectEqualStrings("  metadata JSONB NN", label);
    }
}

test "getSqlKeywordHelp returns help for known keywords" {
    // Test case-insensitive matching
    try std.testing.expectEqualStrings("Retrieve rows from tables", getSqlKeywordHelp("SELECT").?);
    try std.testing.expectEqualStrings("Retrieve rows from tables", getSqlKeywordHelp("select").?);
    try std.testing.expectEqualStrings("Retrieve rows from tables", getSqlKeywordHelp("SeLeCt").?);

    // Test various SQL keywords
    try std.testing.expectEqualStrings("Add new rows to a table", getSqlKeywordHelp("INSERT").?);
    try std.testing.expectEqualStrings("Modify existing rows", getSqlKeywordHelp("UPDATE").?);
    try std.testing.expectEqualStrings("Remove rows from a table", getSqlKeywordHelp("DELETE").?);
    try std.testing.expectEqualStrings("Specify source table(s)", getSqlKeywordHelp("FROM").?);
    try std.testing.expectEqualStrings("Filter rows with conditions", getSqlKeywordHelp("WHERE").?);
    try std.testing.expectEqualStrings("Combine rows from multiple tables", getSqlKeywordHelp("JOIN").?);
    try std.testing.expectEqualStrings("Return matching rows from both tables", getSqlKeywordHelp("INNER").?);
    try std.testing.expectEqualStrings("Return all left table rows + matches", getSqlKeywordHelp("LEFT").?);
    try std.testing.expectEqualStrings("Sort result rows", getSqlKeywordHelp("ORDER").?);
    try std.testing.expectEqualStrings("Specify sort/group column(s)", getSqlKeywordHelp("BY").?);
    try std.testing.expectEqualStrings("Restrict number of rows returned", getSqlKeywordHelp("LIMIT").?);
    try std.testing.expectEqualStrings("Remove duplicate rows", getSqlKeywordHelp("DISTINCT").?);
}

test "getSqlKeywordHelp returns null for unknown keywords" {
    try std.testing.expect(getSqlKeywordHelp("FOOBAR") == null);
    try std.testing.expect(getSqlKeywordHelp("NOT_A_KEYWORD") == null);
    try std.testing.expect(getSqlKeywordHelp("") == null);
}

test "getSqlKeywordHelp covers transaction keywords" {
    try std.testing.expectEqualStrings("Start transaction", getSqlKeywordHelp("BEGIN").?);
    try std.testing.expectEqualStrings("Save transaction changes", getSqlKeywordHelp("COMMIT").?);
    try std.testing.expectEqualStrings("Discard transaction changes", getSqlKeywordHelp("ROLLBACK").?);
    try std.testing.expectEqualStrings("Create rollback point within transaction", getSqlKeywordHelp("SAVEPOINT").?);
}

test "getSqlKeywordHelp covers advanced SQL features" {
    try std.testing.expectEqualStrings("Define CTE (Common Table Expression)", getSqlKeywordHelp("WITH").?);
    try std.testing.expectEqualStrings("Enable recursive CTE", getSqlKeywordHelp("RECURSIVE").?);
    try std.testing.expectEqualStrings("Define window for window function", getSqlKeywordHelp("OVER").?);
    try std.testing.expectEqualStrings("Collect table statistics for optimizer", getSqlKeywordHelp("ANALYZE").?);
    try std.testing.expectEqualStrings("Show query execution plan", getSqlKeywordHelp("EXPLAIN").?);
    try std.testing.expectEqualStrings("Reclaim storage and optimize database", getSqlKeywordHelp("VACUUM").?);
}

test "getTableHelp shows table metadata tooltip" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Create unique database file to avoid state persistence across tests
    const path = "test_table_help_users.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var db = try Database.open(allocator, path, .{});
    defer db.close();

    // Create test table
    _ = try db.execSQL("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)");

    // Get tooltip for the table
    const help = getTableHelp(&db, "users");
    try testing.expect(help != null);
    const help_text = help.?;

    // Verify tooltip contains expected information
    try testing.expect(std.mem.indexOf(u8, help_text, "Table: users") != null);
    try testing.expect(std.mem.indexOf(u8, help_text, "3 columns") != null);
    try testing.expect(std.mem.indexOf(u8, help_text, "id: integer") != null);
    try testing.expect(std.mem.indexOf(u8, help_text, "name: text") != null);
    try testing.expect(std.mem.indexOf(u8, help_text, "age: integer") != null);
}

test "getTableHelp handles nonexistent tables" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var db = try Database.open(allocator, ":memory:", .{});
    defer db.close();

    // Query nonexistent table should return null
    const help = getTableHelp(&db, "nonexistent_table");
    try testing.expect(help == null);
}

test "getTableHelp truncates long column lists" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Create unique database file to avoid state persistence across tests
    const path = "test_table_help_products.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var db = try Database.open(allocator, path, .{});
    defer db.close();

    // Create table with more than 3 columns
    _ = try db.execSQL("CREATE TABLE products (id INTEGER, name TEXT, price REAL, stock INTEGER, category TEXT)");

    // Get tooltip
    const help = getTableHelp(&db, "products");
    try testing.expect(help != null);
    const help_text = help.?;

    // Should show "5 columns" and first 3 columns + "..."
    try testing.expect(std.mem.indexOf(u8, help_text, "5 columns") != null);
    try testing.expect(std.mem.indexOf(u8, help_text, "...") != null);
    try testing.expect(std.mem.indexOf(u8, help_text, "id: integer") != null);
    try testing.expect(std.mem.indexOf(u8, help_text, "name: text") != null);
    try testing.expect(std.mem.indexOf(u8, help_text, "price: real") != null);
}
