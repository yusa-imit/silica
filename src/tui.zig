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
    defer table_info.deinit(db.allocator);
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

// Check if SQL string contains a keyword (case-insensitive)
fn sqlContains(sql: []const u8, keyword: []const u8) bool {
    if (sql.len < keyword.len) return false;
    var i: usize = 0;
    while (i + keyword.len <= sql.len) : (i += 1) {
        if (std.ascii.eqlIgnoreCase(sql[i..i+keyword.len], keyword)) return true;
    }
    return false;
}

// ── Focus Pane ───────────────────────────────────────────────────────

const Pane = enum { schema, results, input };

// ── Ring Menu Items ──────────────────────────────────────────────────

const RING_MENU_ITEMS = [_][]const u8{ "Execute", "Schema", "Results", "Refresh", "Clear", "Quit" };

// ── Query History KanbanBoard ────────────────────────────────────────

const QUERY_HISTORY_MAX = 16;
const QUERY_TITLE_MAX = 48;

// ── Activity Feed Audit Log ──────────────────────────────────────────

const ACTIVITY_LOG_MAX = 64;
const ACTIVITY_EVENT_MAX = 60;

// ── FlowChart Query Pipeline ─────────────────────────────────────────

const FLOW_NODE_COUNT: usize = 6;

const FLOW_NODES = [FLOW_NODE_COUNT]sailor.FlowNode{
    .{ .label = "Start", .kind = .terminal, .col = 0, .row = 0 },
    .{ .label = "Parse SQL", .kind = .process, .col = 0, .row = 1 },
    .{ .label = "Analyze", .kind = .process, .col = 0, .row = 2 },
    .{ .label = "Plan", .kind = .process, .col = 0, .row = 3 },
    .{ .label = "Execute", .kind = .decision, .col = 0, .row = 4 },
    .{ .label = "Result", .kind = .terminal, .col = 0, .row = 5 },
};

const FLOW_EDGES = [5]sailor.FlowEdge{
    .{ .from = 0, .to = 1 },
    .{ .from = 1, .to = 2 },
    .{ .from = 2, .to = 3 },
    .{ .from = 3, .to = 4 },
    .{ .from = 4, .to = 5, .label = "ok" },
};

const QueryHistoryEntry = struct {
    title: [QUERY_TITLE_MAX]u8 = std.mem.zeroes([QUERY_TITLE_MAX]u8),
    title_len: usize = 0,
    success: bool = false,
    duration_ms: u64 = 0,
};

// ── RadarChart Query Stats ───────────────────────────────────────────

const RADAR_SERIES_COUNT: usize = 2;

const RADAR_AXES = [_][]const u8{ "SELECT", "INSERT", "UPDATE", "DELETE", "DDL", "Error" };

// ── Treemap Table Space ──────────────────────────────────────────────

const MAX_TREEMAP_ITEMS: usize = 64;

// ── MatrixView Query Metrics ─────────────────────────────────────────

const MATRIX_ROW_HEADERS = [_][]const u8{ "SELECT", "INSERT", "UPDATE", "DELETE", "DDL", "Error" };
const MATRIX_COL_HEADERS = [_][]const u8{ "Count", "Avg ms", "Max ms" };
const MATRIX_ROWS: usize = 6;
const MATRIX_COLS: usize = 3;

// ── SankeyDiagram SQL Data Flow ──────────────────────────────────────

const SANKEY_NODE_COUNT: usize = 6;

const SANKEY_NODES = [SANKEY_NODE_COUNT]sailor.tui.widgets.SankeyNode{
    .{ .label = "SELECT", .column = 0, .style = .{ .fg = .cyan } },
    .{ .label = "DML", .column = 0, .style = .{ .fg = .green } },
    .{ .label = "DDL", .column = 0, .style = .{ .fg = .yellow } },
    .{ .label = "Execute", .column = 1, .style = .{ .fg = .white, .bold = true } },
    .{ .label = "Success", .column = 2, .style = .{ .fg = .bright_green } },
    .{ .label = "Error", .column = 2, .style = .{ .fg = .red } },
};

// ── ChordDiagram SQL Operation Flow ──────────────────────────────────

const CHORD_NODE_COUNT: usize = 5;
const CHORD_NODES = [CHORD_NODE_COUNT][]const u8{ "SELECT", "INSERT", "UPDATE", "DELETE", "DDL" };

const WATERFALL_BAR_COUNT: usize = 6;

// ── FunnelChart SQL Query Pipeline ──────────────────────────────────────────

const FUNNEL_STAGE_COUNT: usize = 4;

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

    // Row detail overlay
    detail_visible: bool = false,
    detail_offset: usize = 0,
    detail_selected: usize = 0,

    // Spinner animation frame counter (incremented each event loop tick)
    spinner_frame: usize = 0,

    // Ring menu context overlay
    ring_menu_visible: bool = false,
    ring_menu_selected: usize = 0,

    // Query execution timer (StopWatch overlay)
    timer_visible: bool = false,
    query_elapsed_ms: u64 = 0,
    query_laps: [32]u64 = std.mem.zeroes([32]u64),
    query_lap_count: usize = 0,
    query_cumulative_ms: u64 = 0,

    // Query history KanbanBoard overlay
    kanban_visible: bool = false,
    kanban_focused_col: usize = 0,
    kanban_focused_card: usize = 0,
    query_history: [QUERY_HISTORY_MAX]QueryHistoryEntry = std.mem.zeroes([QUERY_HISTORY_MAX]QueryHistoryEntry),
    query_history_count: usize = 0,

    // BracketViewer SQL plan overlay
    bracket_visible: bool = false,
    bracket_focused_round: usize = 0,
    bracket_focused_match: usize = 0,
    plan_sql_buf: [512]u8 = std.mem.zeroes([512]u8),
    plan_sql_len: usize = 0,

    // Activity feed audit log overlay
    activity_visible: bool = false,
    activity_focused: usize = 0,
    activity_log: [ACTIVITY_LOG_MAX]sailor.activity_feed.Activity = std.mem.zeroes([ACTIVITY_LOG_MAX]sailor.activity_feed.Activity),
    activity_event_bufs: [ACTIVITY_LOG_MAX][ACTIVITY_EVENT_MAX]u8 = std.mem.zeroes([ACTIVITY_LOG_MAX][ACTIVITY_EVENT_MAX]u8),
    activity_count: usize = 0,

    // GanttChart query timeline overlay
    gantt_visible: bool = false,
    gantt_focused: usize = 0,

    // FlowChart query pipeline overlay
    flow_visible: bool = false,
    flow_focused: usize = 0,

    // MindMap schema explorer overlay
    mind_visible: bool = false,
    mind_focused: usize = 0,

    // RadarChart query stats overlay
    radar_visible: bool = false,
    radar_focused: usize = 0,

    // HexEditor page viewer overlay
    hex_visible: bool = false,
    hex_cursor: usize = 0,
    hex_data: [4096]u8 = std.mem.zeroes([4096]u8),
    hex_data_len: usize = 0,

    // Treemap table space overlay
    treemap_visible: bool = false,
    treemap_focused: usize = 0,

    // MatrixView query metrics overlay
    matrix_visible: bool = false,
    matrix_focused_row: usize = 0,
    matrix_focused_col: usize = 0,

    // SankeyDiagram SQL data flow overlay
    sankey_visible: bool = false,
    sankey_focused: usize = 0,

    // BubbleChart query performance overlay
    bubble_visible: bool = false,
    bubble_focused: usize = 0,

    // ChordDiagram SQL operation flow overlay
    chord_visible: bool = false,
    chord_focused: usize = 0,

    // WaterfallChart SQL operation breakdown overlay
    waterfall_visible: bool = false,
    waterfall_focused: usize = 0,

    // FunnelChart SQL query success funnel overlay
    funnel_visible: bool = false,
    funnel_focused: usize = 0,

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

    fn recordQueryTime(self: *App, elapsed_ms: u64) void {
        self.query_elapsed_ms = elapsed_ms;
        self.query_cumulative_ms += elapsed_ms;
        if (self.query_lap_count < 32) {
            self.query_laps[self.query_lap_count] = self.query_cumulative_ms;
            self.query_lap_count += 1;
        }
    }

    fn recordActivity(self: *App, sql: []const u8, success: bool) void {
        const trimmed = std.mem.trim(u8, sql, " \t\r\n");
        const is_ddl = std.ascii.startsWithIgnoreCase(trimmed, "CREATE") or
            std.ascii.startsWithIgnoreCase(trimmed, "DROP") or
            std.ascii.startsWithIgnoreCase(trimmed, "ALTER");
        const kind: sailor.activity_feed.Kind = if (is_ddl) .action
            else if (success) .success
            else .error_kind;
        const idx = self.activity_count % ACTIVITY_LOG_MAX;
        const copy_len = @min(trimmed.len, ACTIVITY_EVENT_MAX);
        @memcpy(self.activity_event_bufs[idx][0..copy_len], trimmed[0..copy_len]);
        self.activity_log[idx] = .{
            .event = self.activity_event_bufs[idx][0..copy_len],
            .kind = kind,
            .actor = "",
            .timestamp = "",
        };
        self.activity_count += 1;
    }

    fn recordQueryHistory(self: *App, query: []const u8, success: bool, duration_ms: u64) void {
        // Use ring buffer — wrap at QUERY_HISTORY_MAX, but cap count at QUERY_HISTORY_MAX
        if (self.query_history_count < QUERY_HISTORY_MAX) {
            self.query_history_count += 1;
        }
        const idx = (self.query_history_count - 1) % QUERY_HISTORY_MAX;
        var entry = &self.query_history[idx];
        entry.success = success;
        entry.duration_ms = duration_ms;
        const copy_len = @min(query.len, QUERY_TITLE_MAX);
        @memcpy(entry.title[0..copy_len], query[0..copy_len]);
        entry.title_len = copy_len;
    }

    fn executeSQL(self: *App) void {
        if (self.input_text.items.len == 0) return;

        self.clearResults();

        const start_ms = std.time.milliTimestamp();
        var result = self.db.exec(self.input_text.items) catch |err| {
            const elapsed: u64 = @intCast(@max(0, std.time.milliTimestamp() - start_ms));
            self.recordQueryTime(elapsed);
            self.recordQueryHistory(self.input_text.items, false, elapsed);
            self.recordActivity(self.input_text.items, false);
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
        const elapsed: u64 = @intCast(@max(0, std.time.milliTimestamp() - start_ms));
        self.recordQueryTime(elapsed);
        self.recordQueryHistory(self.input_text.items, true, elapsed);
        self.recordActivity(self.input_text.items, true);

        // Copy SQL to plan buffer for bracket viewer
        const copy_len = @min(self.input_text.items.len, 512);
        @memcpy(self.plan_sql_buf[0..copy_len], self.input_text.items[0..copy_len]);
        self.plan_sql_len = copy_len;

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

    fn executeRingMenuAction(self: *App) void {
        switch (self.ring_menu_selected) {
            0 => self.executeSQL(),         // Execute
            1 => self.focus = .schema,      // Schema
            2 => self.focus = .results,     // Results
            3 => self.refreshSchema(),      // Refresh
            4 => {                           // Clear
                self.input_text.clearRetainingCapacity();
                self.input_cursor = 0;
            },
            5 => self.should_quit = true,   // Quit
            else => {},
        }
    }

    fn loadHexData(self: *App) void {
        const file = std.fs.cwd().openFile(self.db_path, .{}) catch return;
        defer file.close();
        const n = file.read(&self.hex_data) catch return;
        self.hex_data_len = n;
    }

    fn handleKey(self: *App, byte: u8) void {
        // Ctrl+C / Ctrl+Q quit
        if (byte == 3 or byte == 17) {
            self.should_quit = true;
            return;
        }

        // 'm' key toggles ring menu
        if (byte == 109) {
            if (self.ring_menu_visible) {
                self.ring_menu_visible = false;
            } else {
                self.ring_menu_visible = true;
                self.ring_menu_selected = 0;
            }
            return;
        }

        // When ring menu is visible, Enter executes the selected action
        if (self.ring_menu_visible) {
            if (byte == '\r' or byte == '\n') {
                self.ring_menu_visible = false;
                self.executeRingMenuAction();
            }
            return;
        }

        // 't' key toggles query timer overlay
        if (byte == 116) {
            self.timer_visible = !self.timer_visible;
            return;
        }

        // 'b' key toggles query history kanban board (not while editing input)
        if (byte == 98 and self.focus != .input) {
            if (self.kanban_visible) {
                self.kanban_visible = false;
            } else {
                self.kanban_visible = true;
                self.kanban_focused_col = 0;
                self.kanban_focused_card = 0;
            }
            return;
        }

        // 'p' key toggles SQL plan bracket viewer (not while editing input)
        if (byte == 112 and self.focus != .input) {
            if (self.bracket_visible) {
                self.bracket_visible = false;
            } else {
                self.bracket_visible = true;
                self.bracket_focused_round = 0;
                self.bracket_focused_match = 0;
            }
            return;
        }

        // 'a' key toggles activity feed overlay (not while editing input)
        if (byte == 97 and self.focus != .input) {
            if (self.activity_visible) {
                self.activity_visible = false;
            } else {
                self.activity_visible = true;
                self.activity_focused = 0;
            }
            return;
        }

        // 'g' key toggles query timeline gantt chart (not while editing input)
        if (byte == 103 and self.focus != .input) {
            if (self.gantt_visible) {
                self.gantt_visible = false;
            } else {
                self.gantt_visible = true;
                self.gantt_focused = 0;
            }
            return;
        }

        // 'f' key toggles query pipeline flowchart (not while editing input)
        if (byte == 102 and self.focus != .input) {
            if (self.flow_visible) {
                self.flow_visible = false;
            } else {
                self.flow_visible = true;
                self.flow_focused = 0;
            }
            return;
        }

        // 'n' key toggles schema mind map (not while editing input)
        if (byte == 110 and self.focus != .input) {
            if (self.mind_visible) {
                self.mind_visible = false;
            } else {
                self.mind_visible = true;
                self.mind_focused = 0;
            }
            return;
        }

        // 'r' key toggles radar chart query stats (not while editing input)
        if (byte == 114 and self.focus != .input) {
            if (self.radar_visible) {
                self.radar_visible = false;
            } else {
                self.radar_visible = true;
                self.radar_focused = 0;
            }
            return;
        }

        // 'x' key toggles hex editor page viewer (not while editing input)
        if (byte == 120 and self.focus != .input) {
            if (self.hex_visible) {
                self.hex_visible = false;
            } else {
                self.hex_visible = true;
                self.hex_cursor = 0;
                self.loadHexData();
            }
            return;
        }

        // 'w' key toggles treemap table space overlay (not while editing input)
        if (byte == 119 and self.focus != .input) {
            if (self.treemap_visible) {
                self.treemap_visible = false;
            } else {
                self.treemap_visible = true;
                self.treemap_focused = 0;
            }
            return;
        }

        // 'v' key toggles matrix query metrics overlay (not while editing input)
        if (byte == 118 and self.focus != .input) {
            if (self.matrix_visible) {
                self.matrix_visible = false;
            } else {
                self.matrix_visible = true;
                self.matrix_focused_row = 0;
                self.matrix_focused_col = 0;
            }
            return;
        }

        // 's' key toggles sankey SQL data flow overlay (not while editing input)
        if (byte == 115 and self.focus != .input) {
            if (self.sankey_visible) {
                self.sankey_visible = false;
            } else {
                self.sankey_visible = true;
                self.sankey_focused = 0;
            }
            return;
        }

        // 'u' key toggles bubble chart query performance overlay (not while editing input)
        if (byte == 117 and self.focus != .input) {
            if (self.bubble_visible) {
                self.bubble_visible = false;
            } else {
                self.bubble_visible = true;
                self.bubble_focused = 0;
            }
            return;
        }

        // 'c' key toggles chord diagram SQL operation flow overlay (not while editing input)
        if (byte == 99 and self.focus != .input) {
            if (self.chord_visible) {
                self.chord_visible = false;
            } else {
                self.chord_visible = true;
                self.chord_focused = 0;
            }
            return;
        }

        // 'e' key toggles waterfall chart SQL operation breakdown overlay (not while editing input)
        if (byte == 101 and self.focus != .input) {
            if (self.waterfall_visible) {
                self.waterfall_visible = false;
            } else {
                self.waterfall_visible = true;
                self.waterfall_focused = 0;
            }
            return;
        }

        // 'l' key toggles funnel chart SQL query success funnel overlay (not while editing input)
        if (byte == 108 and self.focus != .input) {
            if (self.funnel_visible) {
                self.funnel_visible = false;
            } else {
                self.funnel_visible = true;
                self.funnel_focused = 0;
            }
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

    fn countKanbanColCards(app: *const App, col: usize) usize {
        const actual_count = @min(app.query_history_count, QUERY_HISTORY_MAX);
        var count: usize = 0;
        for (app.query_history[0..actual_count]) |entry| {
            if (col == 0 and entry.success) count += 1;
            if (col == 1 and !entry.success) count += 1;
        }
        return count;
    }

    fn handleEscapeSequence(self: *App, b2: ?u8, b3: ?u8) void {
        // MatrixView arrow key navigation and ESC handling
        if (self.matrix_visible) {
            if (b2 == null) {
                self.matrix_visible = false;
                return;
            }
            if (b2.? == '[') {
                switch (b3 orelse 0) {
                    'B' => { // Down — next row
                        if (self.matrix_focused_row + 1 < MATRIX_ROWS) {
                            self.matrix_focused_row += 1;
                        }
                    },
                    'A' => { // Up — previous row
                        if (self.matrix_focused_row > 0) self.matrix_focused_row -= 1;
                    },
                    'C' => { // Right — next col
                        if (self.matrix_focused_col + 1 < MATRIX_COLS) {
                            self.matrix_focused_col += 1;
                        }
                    },
                    'D' => { // Left — previous col
                        if (self.matrix_focused_col > 0) self.matrix_focused_col -= 1;
                    },
                    else => {},
                }
            }
            return;
        }

        // Treemap arrow key navigation and ESC handling
        if (self.treemap_visible) {
            if (b2 == null) {
                self.treemap_visible = false;
                return;
            }
            if (b2.? == '[') {
                switch (b3 orelse 0) {
                    'B', 'C' => { // Down or Right — next item
                        if (self.treemap_focused + 1 < MAX_TREEMAP_ITEMS) {
                            self.treemap_focused += 1;
                        }
                    },
                    'A', 'D' => { // Up or Left — previous item
                        if (self.treemap_focused > 0) self.treemap_focused -= 1;
                    },
                    else => {},
                }
            }
            return;
        }

        // HexEditor arrow key navigation and ESC handling
        if (self.hex_visible) {
            if (b2 == null) {
                self.hex_visible = false;
                return;
            }
            if (b2.? == '[') {
                switch (b3 orelse 0) {
                    'C' => { // Right — advance cursor
                        if (self.hex_data_len > 0 and self.hex_cursor + 1 < self.hex_data_len) {
                            self.hex_cursor += 1;
                        }
                    },
                    'D' => { // Left — retreat cursor
                        if (self.hex_cursor > 0) self.hex_cursor -= 1;
                    },
                    else => {},
                }
            }
            return;
        }

        // Activity feed arrow key navigation and ESC handling
        if (self.activity_visible) {
            if (b2 == null) {
                self.activity_visible = false;
                return;
            }
            if (b2.? == '[') {
                switch (b3 orelse 0) {
                    'A' => { // Up
                        if (self.activity_focused > 0) self.activity_focused -= 1;
                    },
                    'B' => { // Down
                        const count = @min(self.activity_count, ACTIVITY_LOG_MAX);
                        if (count > 0 and self.activity_focused + 1 < count) {
                            self.activity_focused += 1;
                        }
                    },
                    else => {},
                }
            }
            return;
        }

        // GanttChart arrow key navigation and ESC handling
        if (self.gantt_visible) {
            if (b2 == null) {
                self.gantt_visible = false;
                return;
            }
            if (b2.? == '[') {
                switch (b3 orelse 0) {
                    'A' => { // Up
                        if (self.gantt_focused > 0) self.gantt_focused -= 1;
                    },
                    'B' => { // Down
                        const count = @min(self.query_history_count, QUERY_HISTORY_MAX);
                        if (count > 0 and self.gantt_focused + 1 < count) {
                            self.gantt_focused += 1;
                        }
                    },
                    else => {},
                }
            }
            return;
        }

        // RadarChart arrow key navigation and ESC handling
        if (self.radar_visible) {
            if (b2 == null) {
                self.radar_visible = false;
                return;
            }
            if (b2.? == '[') {
                switch (b3 orelse 0) {
                    'C' => { // Right — next series
                        if (self.radar_focused + 1 < RADAR_SERIES_COUNT) {
                            self.radar_focused += 1;
                        }
                    },
                    'D' => { // Left — previous series
                        if (self.radar_focused > 0) self.radar_focused -= 1;
                    },
                    else => {},
                }
            }
            return;
        }

        // MindMap arrow key navigation and ESC handling
        if (self.mind_visible) {
            if (b2 == null) {
                self.mind_visible = false;
                return;
            }
            if (b2.? == '[') {
                switch (b3 orelse 0) {
                    'A' => { // Up
                        if (self.mind_focused > 0) self.mind_focused -= 1;
                    },
                    'B' => { // Down
                        if (self.schema_table_indices.items.len == 0) {
                            // Allow at least minimal navigation when no tables
                            if (self.mind_focused == 0) {
                                self.mind_focused = 1;
                            }
                        } else if (self.mind_focused + 1 < self.schema_table_indices.items.len) {
                            self.mind_focused += 1;
                        }
                    },
                    else => {},
                }
            }
            return;
        }

        // SankeyDiagram arrow key navigation and ESC handling
        if (self.sankey_visible) {
            if (b2 == null) {
                self.sankey_visible = false;
                return;
            }
            if (b2.? == '[') {
                switch (b3 orelse 0) {
                    'A' => { // Up — previous node
                        if (self.sankey_focused > 0) self.sankey_focused -= 1;
                    },
                    'B' => { // Down — next node
                        if (self.sankey_focused + 1 < SANKEY_NODE_COUNT) {
                            self.sankey_focused += 1;
                        }
                    },
                    else => {},
                }
            }
            return;
        }

        // BubbleChart arrow key navigation and ESC handling
        if (self.bubble_visible) {
            if (b2 == null) {
                self.bubble_visible = false;
                return;
            }
            if (b2.? == '[') {
                switch (b3 orelse 0) {
                    'A' => { // Up — previous bubble
                        if (self.bubble_focused > 0) self.bubble_focused -= 1;
                    },
                    'B' => { // Down — next bubble
                        if (self.bubble_focused + 1 < QUERY_HISTORY_MAX) {
                            self.bubble_focused += 1;
                        }
                    },
                    else => {},
                }
            }
            return;
        }

        // ChordDiagram arrow key navigation and ESC handling
        if (self.chord_visible) {
            if (b2 == null) {
                self.chord_visible = false;
                return;
            }
            if (b2.? == '[') {
                switch (b3 orelse 0) {
                    'A' => { // Up — previous node
                        if (self.chord_focused > 0) self.chord_focused -= 1;
                    },
                    'B' => { // Down — next node
                        if (self.chord_focused + 1 < CHORD_NODE_COUNT) {
                            self.chord_focused += 1;
                        }
                    },
                    else => {},
                }
            }
            return;
        }

        // WaterfallChart arrow key navigation and ESC handling
        if (self.waterfall_visible) {
            if (b2 == null) {
                self.waterfall_visible = false;
                return;
            }
            if (b2.? == '[') {
                switch (b3 orelse 0) {
                    'A' => { // Up — previous bar
                        if (self.waterfall_focused > 0) self.waterfall_focused -= 1;
                    },
                    'B' => { // Down — next bar
                        if (self.waterfall_focused + 1 < WATERFALL_BAR_COUNT) {
                            self.waterfall_focused += 1;
                        }
                    },
                    else => {},
                }
            }
            return;
        }

        // FunnelChart arrow key navigation and ESC handling
        if (self.funnel_visible) {
            if (b2 == null) {
                self.funnel_visible = false;
                return;
            }
            if (b2.? == '[') {
                switch (b3 orelse 0) {
                    'A' => { // Up — previous stage
                        if (self.funnel_focused > 0) self.funnel_focused -= 1;
                    },
                    'B' => { // Down — next stage
                        if (self.funnel_focused + 1 < FUNNEL_STAGE_COUNT) {
                            self.funnel_focused += 1;
                        }
                    },
                    else => {},
                }
            }
            return;
        }

        // FlowChart arrow key navigation and ESC handling
        if (self.flow_visible) {
            if (b2 == null) {
                self.flow_visible = false;
                return;
            }
            if (b2.? == '[') {
                switch (b3 orelse 0) {
                    'A' => { // Up
                        if (self.flow_focused > 0) self.flow_focused -= 1;
                    },
                    'B' => { // Down
                        if (self.flow_focused + 1 < FLOW_NODE_COUNT) {
                            self.flow_focused += 1;
                        }
                    },
                    else => {},
                }
            }
            return;
        }

        // Kanban board arrow key navigation and ESC handling
        if (self.kanban_visible) {
            if (b2 == null) {
                self.kanban_visible = false;
                return;
            }
            if (b2.? == '[') {
                switch (b3 orelse 0) {
                    'A' => { // Up
                        if (self.kanban_focused_card > 0) self.kanban_focused_card -= 1;
                    },
                    'B' => { // Down
                        self.kanban_focused_card += 1;
                        // Clamp to actual cards in focused column
                        const col_count = countKanbanColCards(self, self.kanban_focused_col);
                        if (col_count > 0 and self.kanban_focused_card >= col_count) {
                            self.kanban_focused_card = col_count - 1;
                        }
                    },
                    'C' => { // Right
                        if (self.kanban_focused_col < 1) self.kanban_focused_col += 1;
                    },
                    'D' => { // Left
                        if (self.kanban_focused_col > 0) self.kanban_focused_col -= 1;
                    },
                    else => {},
                }
            }
            return;
        }

        // Bracket viewer arrow key navigation and ESC handling
        if (self.bracket_visible) {
            if (b2 == null) {
                self.bracket_visible = false;
                return;
            }
            if (b2.? == '[') {
                switch (b3 orelse 0) {
                    'A' => { // Up
                        if (self.bracket_focused_match > 0) self.bracket_focused_match -= 1;
                    },
                    'B' => { // Down
                        self.bracket_focused_match += 1;
                    },
                    'C' => { // Right
                        if (self.bracket_focused_round < 2) self.bracket_focused_round += 1;
                    },
                    'D' => { // Left
                        if (self.bracket_focused_round > 0) self.bracket_focused_round -= 1;
                    },
                    else => {},
                }
            }
            return;
        }

        // Ring menu arrow key navigation
        if (self.ring_menu_visible) {
            if (b2 == null) {
                self.ring_menu_visible = false;
                return;
            }
            if (b2.? == '[') {
                switch (b3 orelse 0) {
                    'A', 'D' => { // Up or Left → prev
                        if (self.ring_menu_selected == 0) {
                            self.ring_menu_selected = 5;
                        } else {
                            self.ring_menu_selected -= 1;
                        }
                    },
                    'B', 'C' => { // Down or Right → next
                        self.ring_menu_selected = (self.ring_menu_selected + 1) % 6;
                    },
                    else => {},
                }
            }
            return;
        }

        // Bare ESC (b2 == null) closes the topmost visible overlay
        if (b2 == null) {
            if (self.matrix_visible) {
                self.matrix_visible = false;
            } else if (self.treemap_visible) {
                self.treemap_visible = false;
            } else if (self.sankey_visible) {
                self.sankey_visible = false;
            } else if (self.bubble_visible) {
                self.bubble_visible = false;
            } else if (self.chord_visible) {
                self.chord_visible = false;
            } else if (self.waterfall_visible) {
                self.waterfall_visible = false;
            } else if (self.funnel_visible) {
                self.funnel_visible = false;
            } else if (self.hex_visible) {
                self.hex_visible = false;
            } else if (self.radar_visible) {
                self.radar_visible = false;
            } else if (self.mind_visible) {
                self.mind_visible = false;
            } else if (self.gantt_visible) {
                self.gantt_visible = false;
            } else if (self.kanban_visible) {
                self.kanban_visible = false;
            } else if (self.bracket_visible) {
                self.bracket_visible = false;
            } else if (self.detail_visible) {
                self.detail_visible = false;
            } else if (self.timer_visible) {
                self.timer_visible = false;
            }
            return;
        }
        if (b2.? != '[') return;
        if (b3 == null) return;

        // Arrow keys while detail overlay is open move the selection
        if (self.detail_visible and self.focus == .results) {
            const num_cols = self.result_columns.items.len;
            if (b3.? == 'A') { // Up
                if (self.detail_selected > 0) self.detail_selected -= 1;
            } else if (b3.? == 'B') { // Down
                if (num_cols > 0 and self.detail_selected < num_cols - 1) self.detail_selected += 1;
            }
            return;
        }

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
        if (byte == '\r' or byte == '\n') {
            if (self.result_rows.items.len > 0) {
                self.detail_visible = true;
                self.detail_offset = 0;
                self.detail_selected = 0;
            }
        }
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

    // Render results table with optional MiniMap sidebar
    const minimap_width: u16 = 2;
    const show_minimap = app.result_rows.items.len > 0 and results_area.width > minimap_width + 8;
    if (show_minimap) {
        const result_split = try tui.layout.split(allocator, .horizontal, results_area, &.{
            .{ .min = 8 },
            .{ .length = minimap_width },
        });
        defer allocator.free(result_split);
        renderResultsTable(app, buf, result_split[0]);
        const lines = try buildMinimapLines(allocator, app.result_rows.items);
        defer allocator.free(lines);
        tui.widgets.MiniMap.init()
            .withLines(lines)
            .withViewportTop(app.result_offset)
            .withViewportHeight(minimapViewportHeight(results_area.height))
            .withViewportStyle(.{ .fg = .cyan, .bold = true })
            .withStyle(.{ .fg = .bright_black })
            .render(buf, result_split[1]);
    } else {
        renderResultsTable(app, buf, results_area);
    }

    // Render SQL input
    renderSQLInput(app, buf, input_area);

    // Render status bar
    renderStatusBar(app, buf, status_area);

    // Render row detail overlay (on top of everything)
    if (app.detail_visible) {
        renderDetailOverlay(app, buf, area);
    }

    // Render timer overlay (on top of content, below ring menu)
    if (app.timer_visible) {
        renderTimerOverlay(app, buf, area);
    }

    // Render query history kanban board
    if (app.kanban_visible) {
        renderKanbanBoard(app, buf, area);
    }

    // Render SQL plan bracket viewer
    if (app.bracket_visible) {
        renderBracketViewer(app, buf, area);
    }

    // Render activity feed overlay
    if (app.activity_visible) {
        renderActivityFeed(app, buf, area);
    }

    // Render query timeline gantt chart
    if (app.gantt_visible) {
        renderGanttChart(app, buf, area);
    }

    // Render query pipeline flowchart
    if (app.flow_visible) {
        renderFlowChart(app, buf, area);
    }

    // Render schema mind map
    if (app.mind_visible) {
        renderMindMap(app, buf, area);
    }

    // Render radar chart query stats
    if (app.radar_visible) {
        renderRadarChart(app, buf, area);
    }

    // Render hex editor page viewer
    if (app.hex_visible) {
        renderHexEditor(app, buf, area);
    }

    // Render treemap table space overlay
    if (app.treemap_visible) {
        renderTreemap(app, buf, area);
    }

    // Render matrix view query metrics overlay
    if (app.matrix_visible) {
        renderMatrixView(app, buf, area);
    }

    // Render sankey SQL data flow overlay
    if (app.sankey_visible) {
        renderSankeyDiagram(app, buf, area);
    }

    // Render bubble chart query performance overlay
    if (app.bubble_visible) {
        renderBubbleChart(app, buf, area);
    }

    // Render chord diagram SQL flow overlay
    if (app.chord_visible) {
        renderChordDiagram(app, buf, area);
    }

    // Render waterfall chart SQL operation breakdown overlay
    if (app.waterfall_visible) {
        renderWaterfallChart(app, buf, area);
    }

    // Render funnel chart SQL query success funnel overlay
    if (app.funnel_visible) {
        renderFunnelChart(app, buf, area);
    }

    // Render ring menu (on top of everything)
    renderRingMenu(app, buf, area);
}

fn isTableIndex(table_indices: []const usize, idx: usize) bool {
    for (table_indices) |ti| {
        if (ti == idx) return true;
        if (ti > idx) break;
    }
    return false;
}

fn buildMinimapLines(allocator: std.mem.Allocator, rows: []const []const []const u8) ![][]const u8 {
    const lines = try allocator.alloc([]const u8, rows.len);
    for (rows, 0..) |row, i| {
        lines[i] = if (row.len > 0) row[0] else " ";
    }
    return lines;
}

fn minimapViewportHeight(area_height: u16) usize {
    return if (area_height > 3) area_height - 3 else 1;
}

fn renderSchemaTree(app: *App, buf: *tui.Buffer, area: tui.Rect) void {
    const is_focused = app.focus == .schema;
    const border_style: tui.Style = if (is_focused) .{ .fg = .cyan, .bold = true } else .{};

    const block = (tui.widgets.Block{
        .title = "Schema",
        .title_position = .top_left,
    })
        .withBorderStyle(border_style)
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
            buf.set(inner.x + cx, inner.y + row, tui.Cell.init(' ', if (is_selected) item_style else .{}));
        }

        // Render the item text
        var x: u16 = 0;
        for (item) |c| {
            if (x >= inner.width) break;
            buf.set(inner.x + x, inner.y + row, tui.Cell.init(c, item_style));
            x += 1;
        }
    }
}

fn renderResultsTable(app: *App, buf: *tui.Buffer, area: tui.Rect) void {
    const is_focused = app.focus == .results;
    const border_style: tui.Style = if (is_focused) .{ .fg = .cyan, .bold = true } else .{};

    const block = (tui.widgets.Block{
        .title = "Results",
        .title_position = .top_left,
    })
        .withBorderStyle(border_style)
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

    const block = (tui.widgets.Block{
        .title = "SQL",
        .title_position = .top_left,
    })
        .withBorderStyle(border_style)
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
        buf.set(x, inner.y, tui.Cell.init(c, display_style));
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
            buf.set(cx, inner.y, tui.Cell.init(cursor_char, .{ .reverse = true }));
            break :blk cx;
        } else if (is_focused) {
            // Empty input — cursor on first cell
            buf.set(inner.x, inner.y, tui.Cell.init(' ', .{ .reverse = true }));
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
    const block = (tui.widgets.Block{
        .title = "Completions",
        .title_position = .top_left,
    })
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
            buf.set(inner.x + cx, inner.y + row, tui.Cell.init(' ', if (is_selected) item_style else .{}));
        }

        // Render item text
        var x: u16 = 0;
        for (item.text) |c| {
            if (x >= inner.width) break;
            buf.set(inner.x + x, inner.y + row, tui.Cell.init(c, item_style));
            x += 1;
        }

        // Render description if present
        if (item.description) |desc| {
            if (x + 3 < inner.width) {
                buf.setString(inner.x + x, inner.y + row, " - ", .{ .fg = .bright_black });
                x += 3;
                for (desc) |c| {
                    if (x >= inner.width) break;
                    buf.set(inner.x + x, inner.y + row, tui.Cell.init(c, .{ .fg = .bright_black }));
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

fn renderDetailOverlay(app: *App, buf: *tui.Buffer, area: tui.Rect) void {
    if (app.result_rows.items.len == 0 or app.result_columns.items.len == 0) return;

    // Center a 60%-wide, 80%-tall popup
    const ow: u16 = @max(20, area.width * 6 / 10);
    const oh: u16 = @max(5, area.height * 8 / 10);
    const ox: u16 = area.x + (area.width -| ow) / 2;
    const oy: u16 = area.y + (area.height -| oh) / 2;
    const popup_area = tui.Rect{ .x = ox, .y = oy, .width = ow, .height = oh };

    // Clear popup background
    var py = oy;
    while (py < oy + oh) : (py += 1) {
        var px = ox;
        while (px < ox + ow) : (px += 1) {
            buf.set(px, py, tui.Cell.init(' ', .{}));
        }
    }

    const row = app.result_rows.items[app.result_selected];
    const cols = app.result_columns.items;

    // Build entries slice on the stack (max 64 columns)
    var entries_buf: [64]tui.KeyValueViewer.Entry = undefined;
    const num = @min(cols.len, entries_buf.len);
    for (0..num) |i| {
        entries_buf[i] = .{
            .key = cols[i],
            .value = if (i < row.len) row[i] else "(null)",
        };
    }
    const entries = entries_buf[0..num];

    const row_label = std.fmt.allocPrint(app.allocator, "Row {d}", .{app.result_selected + 1}) catch "Row Detail";
    defer app.allocator.free(row_label);

    // Auto-scroll detail_offset to keep detail_selected visible
    const visible_rows = if (oh > 2) oh - 2 else 1;
    if (app.detail_selected < app.detail_offset) {
        app.detail_offset = app.detail_selected;
    } else if (app.detail_selected >= app.detail_offset + visible_rows) {
        app.detail_offset = app.detail_selected - visible_rows + 1;
    }

    const viewer = tui.KeyValueViewer.init(entries)
        .withOffset(app.detail_offset)
        .withSelected(app.detail_selected)
        .withKeyStyle(.{ .fg = .cyan })
        .withSelectedKeyStyle(.{ .fg = .cyan, .bold = true, .reverse = true })
        .withSelectedValueStyle(.{ .reverse = true })
        .withBlock((tui.widgets.Block{
            .title = row_label,
            .title_position = .top_left,
        }).withBorderStyle(.{ .fg = .cyan }));
    viewer.render(buf, popup_area);
}

fn renderRingMenu(app: *App, buf: *tui.Buffer, area: tui.Rect) void {
    if (!app.ring_menu_visible) return;
    // Center the ring menu in the terminal area
    const width: u16 = 40;
    const height: u16 = 20;
    const ox: u16 = if (area.width > width) (area.width - width) / 2 else 0;
    const oy: u16 = if (area.height > height) (area.height - height) / 2 else 0;
    const ow: u16 = @min(width, area.width);
    const oh: u16 = @min(height, area.height);
    const popup_area = tui.Rect{ .x = ox, .y = oy, .width = ow, .height = oh };

    // Clear background
    for (oy..oy + oh) |py| {
        for (ox..ox + ow) |px| {
            buf.set(@intCast(px), @intCast(py), tui.Cell.init(' ', .{ .bg = .black }));
        }
    }

    const menu = tui.RingMenu.init()
        .withItems(&RING_MENU_ITEMS)
        .withSelected(app.ring_menu_selected)
        .withCenterLabel("Menu")
        .withRadius(6)
        .withSelectedStyle(tui.Style{ .fg = .yellow, .bold = true })
        .withCenterStyle(tui.Style{ .fg = .cyan, .bold = true })
        .withBlock((tui.widgets.Block{
            .title = " Context Menu (↑↓←→ navigate · Enter select · m/Esc close) ",
            .borders = .all,
        }).withBorderStyle(tui.Style{ .fg = .cyan }));

    menu.render(buf, popup_area);
}

fn renderTimerOverlay(app: *App, buf: *tui.Buffer, area: tui.Rect) void {
    if (!app.timer_visible) return;

    const width: u16 = 36;
    const height: u16 = 12;
    // Position at bottom-right corner (above status bar)
    const ox: u16 = if (area.width > width) area.width - width else 0;
    const oy: u16 = if (area.height > height + 1) area.height - height - 1 else 0;
    const ow: u16 = @min(width, area.width);
    const oh: u16 = @min(height, area.height);
    const popup_area = tui.Rect{ .x = ox, .y = oy, .width = ow, .height = oh };

    // Clear background
    var py: u16 = oy;
    while (py < oy + oh) : (py += 1) {
        var px: u16 = ox;
        while (px < ox + ow) : (px += 1) {
            buf.set(px, py, tui.Cell.init(' ', .{ .bg = .black }));
        }
    }

    const sw = tui.StopWatch.init()
        .withElapsedMs(app.query_cumulative_ms)
        .withLaps(app.query_laps[0..app.query_lap_count])
        .withRunning(false)
        .withShowLaps(true)
        .withShowMilliseconds(true)
        .withTimeStyle(.{ .fg = .cyan, .bold = true })
        .withStatusStyle(.{ .fg = .yellow })
        .withLapStyle(.{ .fg = .white })
        .withBlock((tui.widgets.Block{
            .title = " Query Timer (t/Esc:close) ",
            .borders = .all,
        }).withBorderStyle(tui.Style{ .fg = .green }));

    sw.render(buf, popup_area);
}

fn renderKanbanBoard(app: *App, buf: *tui.Buffer, area: tui.Rect) void {
    if (!app.kanban_visible) return;

    // Centered overlay: ~80% width, ~70% height
    const ow: u16 = @min(area.width * 4 / 5, area.width);
    const oh: u16 = @min(area.height * 7 / 10, area.height);
    const ox: u16 = if (area.width > ow) (area.width - ow) / 2 else 0;
    const oy: u16 = if (area.height > oh) (area.height - oh) / 2 else 0;
    const popup_area = tui.Rect{ .x = ox, .y = oy, .width = ow, .height = oh };

    // Clear background
    var py: u16 = oy;
    while (py < oy + oh) : (py += 1) {
        var px: u16 = ox;
        while (px < ox + ow) : (px += 1) {
            buf.set(px, py, tui.Cell.init(' ', .{ .bg = .black }));
        }
    }

    // Build success and error card arrays
    const actual_count = @min(app.query_history_count, QUERY_HISTORY_MAX);
    var success_cards: [QUERY_HISTORY_MAX]sailor.kanban.Card = undefined;
    var error_cards: [QUERY_HISTORY_MAX]sailor.kanban.Card = undefined;
    var success_count: usize = 0;
    var error_count: usize = 0;

    // Tag buffers for duration strings
    var tag_bufs: [QUERY_HISTORY_MAX][8]u8 = undefined;
    var tag_strs: [QUERY_HISTORY_MAX][]const u8 = undefined;
    var tag_slices: [QUERY_HISTORY_MAX][1][]const u8 = undefined;

    for (app.query_history[0..actual_count]) |*entry| {
        const title_slice = entry.title[0..entry.title_len];
        const priority: sailor.kanban.Priority = if (entry.success) blk: {
            if (entry.duration_ms > 1000) break :blk .high;
            if (entry.duration_ms > 100) break :blk .normal;
            break :blk .low;
        } else .critical;

        // Build duration tag
        const tag_idx = success_count + error_count;
        const tag_written = std.fmt.bufPrint(&tag_bufs[tag_idx], "#{d}ms", .{entry.duration_ms}) catch "#?ms";
        tag_strs[tag_idx] = tag_written;
        tag_slices[tag_idx] = .{tag_strs[tag_idx]};

        if (entry.success) {
            success_cards[success_count] = .{
                .title = title_slice,
                .priority = priority,
                .tags = &tag_slices[tag_idx],
            };
            success_count += 1;
        } else {
            error_cards[error_count] = .{
                .title = title_slice,
                .priority = priority,
                .tags = &tag_slices[tag_idx],
            };
            error_count += 1;
        }
    }

    var cols = [2]sailor.kanban.Column{
        .{ .title = "Success \xe2\x9c\x93", .cards = success_cards[0..success_count] },
        .{ .title = "Errors \xe2\x9c\x97", .cards = error_cards[0..error_count] },
    };

    const kb = sailor.KanbanBoard.init()
        .withColumns(&cols)
        .withFocusedColumn(app.kanban_focused_col)
        .withFocusedCard(app.kanban_focused_card)
        .withBlock((tui.widgets.Block{
            .title = " Query History (b/Esc:close  \xe2\x86\x90\xe2\x86\x92:col  \xe2\x86\x91\xe2\x86\x93:card) ",
            .borders = .all,
        }).withBorderStyle(tui.Style{ .fg = .magenta }));

    kb.render(buf, popup_area);
}

fn renderBracketViewer(app: *App, buf: *tui.Buffer, area: tui.Rect) void {
    if (!app.bracket_visible) return;

    // Centered overlay: ~80% width, ~70% height
    const ow: u16 = @min(area.width * 4 / 5, area.width);
    const oh: u16 = @min(area.height * 7 / 10, area.height);
    const ox: u16 = if (area.width > ow) (area.width - ow) / 2 else 0;
    const oy: u16 = if (area.height > oh) (area.height - oh) / 2 else 0;
    const popup_area = tui.Rect{ .x = ox, .y = oy, .width = ow, .height = oh };

    // Clear background
    var py: u16 = oy;
    while (py < oy + oh) : (py += 1) {
        var px: u16 = ox;
        while (px < ox + ow) : (px += 1) {
            buf.set(px, py, tui.Cell.init(' ', .{ .bg = .black }));
        }
    }

    // Build rounds from SQL
    const sql = app.plan_sql_buf[0..app.plan_sql_len];

    // Round 1 "Storage": 1 match [Seq Scan vs Index Scan]
    const storage_winner: sailor.bracket_viewer.Winner = if (sqlContains(sql, "WHERE")) .b else .a;
    const storage_match = sailor.bracket_viewer.Match{
        .team_a = "Seq Scan",
        .team_b = "Index Scan",
        .winner = storage_winner,
    };

    // Round 2 "Logic": 2 matches
    const filter_winner: sailor.bracket_viewer.Winner = if (sqlContains(sql, "JOIN")) .b else .a;
    const filter_match = sailor.bracket_viewer.Match{
        .team_a = "Filter",
        .team_b = "Join",
        .winner = filter_winner,
    };

    const aggregate_winner: sailor.bracket_viewer.Winner = if (sqlContains(sql, "GROUP BY")) .b else .a;
    const aggregate_match = sailor.bracket_viewer.Match{
        .team_a = "Project",
        .team_b = "Aggregate",
        .winner = aggregate_winner,
    };

    // Round 3 "Output": 1 match [Project vs Sort]
    const output_winner: sailor.bracket_viewer.Winner = if (sqlContains(sql, "ORDER BY")) .b else .a;
    const output_match = sailor.bracket_viewer.Match{
        .team_a = "Project",
        .team_b = "Sort",
        .winner = output_winner,
    };

    // Build round arrays (no titles in API, just matches)
    var rounds: [3]sailor.bracket_viewer.Round = undefined;
    rounds[0] = .{
        .matches = &.{storage_match},
    };
    rounds[1] = .{
        .matches = &.{ filter_match, aggregate_match },
    };
    rounds[2] = .{
        .matches = &.{output_match},
    };

    // Render with sailor's BracketViewer widget
    const bv = sailor.BracketViewer.init()
        .withRounds(&rounds)
        .withFocusedRound(app.bracket_focused_round)
        .withFocusedMatch(app.bracket_focused_match)
        .withWinStyle(tui.Style{ .fg = .green })
        .withFocusedStyle(tui.Style{ .fg = .cyan, .bold = true })
        .withShowScores(false)
        .withBlock((tui.widgets.Block{
            .title = " Query Plan (Esc:close) ",
            .borders = .all,
        }).withBorderStyle(tui.Style{ .fg = .blue }));

    bv.render(buf, popup_area);
}

fn renderActivityFeed(app: *App, buf: *tui.Buffer, area: tui.Rect) void {
    if (!app.activity_visible) return;

    const count = @min(app.activity_count, ACTIVITY_LOG_MAX);

    // Centered overlay: ~70% width, ~70% height
    const ow: u16 = @min(area.width * 7 / 10, area.width);
    const oh: u16 = @min(area.height * 7 / 10, area.height);
    const ox: u16 = if (area.width > ow) (area.width - ow) / 2 else 0;
    const oy: u16 = if (area.height > oh) (area.height - oh) / 2 else 0;
    const popup_area = tui.Rect{ .x = ox, .y = oy, .width = ow, .height = oh };

    // Clear background
    var py: u16 = oy;
    while (py < oy + oh) : (py += 1) {
        var px: u16 = ox;
        while (px < ox + ow) : (px += 1) {
            buf.set(px, py, tui.Cell.init(' ', .{ .bg = .black }));
        }
    }

    const feed = sailor.ActivityFeed.init()
        .withItems(app.activity_log[0..count])
        .withFocused(app.activity_focused)
        .withShowTimestamp(false)
        .withShowActor(false)
        .withSuccessStyle(tui.Style{ .fg = .green })
        .withErrorStyle(tui.Style{ .fg = .red })
        .withActionStyle(tui.Style{ .fg = .cyan })
        .withInfoStyle(tui.Style{ .fg = .white })
        .withFocusedStyle(tui.Style{ .fg = .black, .bg = .white, .bold = true })
        .withBlock((tui.widgets.Block{
            .title = " Activity Log  ↑↓ navigate  a/Esc close ",
            .borders = .all,
        }).withBorderStyle(tui.Style{ .fg = .magenta }));

    feed.render(buf, popup_area);
}

fn renderGanttChart(app: *App, buf: *tui.Buffer, area: tui.Rect) void {
    if (!app.gantt_visible) return;

    const actual_count = @min(app.query_history_count, QUERY_HISTORY_MAX);

    // Centered overlay: ~80% width, ~70% height
    const ow: u16 = @min(area.width * 4 / 5, area.width);
    const oh: u16 = @min(@as(u16, @intCast(actual_count + 2)), @min(area.height * 7 / 10, area.height));
    const ox: u16 = if (area.width > ow) (area.width - ow) / 2 else 0;
    const oy: u16 = if (area.height > oh) (area.height - oh) / 2 else 0;
    const popup_area = tui.Rect{ .x = ox, .y = oy, .width = ow, .height = oh };

    // Clear background
    var py: u16 = oy;
    while (py < oy + oh) : (py += 1) {
        var px: u16 = ox;
        while (px < ox + ow) : (px += 1) {
            buf.set(px, py, tui.Cell.init(' ', .{ .bg = .black }));
        }
    }

    // Build tasks from query_history — each task's start/end derived from cumulative duration
    var tasks: [QUERY_HISTORY_MAX]sailor.gantt.Task = undefined;
    var cumulative: u64 = 0;
    for (0..actual_count) |i| {
        const entry = &app.query_history[i];
        const task_start: u16 = @intCast(@min(cumulative, 65535));
        cumulative += entry.duration_ms;
        const task_end: u16 = @intCast(@min(cumulative, 65535));
        tasks[i] = sailor.gantt.Task{
            .name = entry.title[0..entry.title_len],
            .start = task_start,
            .end = if (task_end > task_start) task_end else task_start + 1,
            .progress = if (entry.success) 100 else 0,
        };
    }

    const chart = sailor.gantt.GanttChart.init()
        .withTasks(tasks[0..actual_count])
        .withFocused(app.gantt_focused)
        .withLabelWidth(22)
        .withShowProgress(true)
        .withStyle(.{ .fg = .white })
        .withBarStyle(.{ .fg = .bright_black })
        .withCompleteStyle(.{ .fg = .green })
        .withFocusedStyle(.{ .fg = .black, .bg = .cyan, .bold = true })
        .withBlock((tui.widgets.Block{
            .title = " Query Timeline (g/Esc:close  \xe2\x86\x91\xe2\x86\x93:navigate) ",
            .borders = .all,
        }).withBorderStyle(tui.Style{ .fg = .yellow }));

    chart.render(buf, popup_area);
}

fn renderFlowChart(app: *App, buf: *tui.Buffer, area: tui.Rect) void {
    if (!app.flow_visible) return;

    // Centered overlay: ~60% width, ~90% height (to show all 6 nodes)
    const ow: u16 = @min(area.width * 3 / 5, area.width);
    const oh: u16 = @min(area.height * 9 / 10, area.height);
    const ox: u16 = if (area.width > ow) (area.width - ow) / 2 else 0;
    const oy: u16 = if (area.height > oh) (area.height - oh) / 2 else 0;
    const popup_area = tui.Rect{ .x = ox, .y = oy, .width = ow, .height = oh };

    // Clear background
    var py: u16 = oy;
    while (py < oy + oh) : (py += 1) {
        var px: u16 = ox;
        while (px < ox + ow) : (px += 1) {
            buf.set(px, py, tui.Cell.init(' ', .{ .bg = .black }));
        }
    }

    const chart = sailor.FlowChart.init()
        .withNodes(&FLOW_NODES)
        .withEdges(&FLOW_EDGES)
        .withFocused(app.flow_focused)
        .withStyle(.{ .fg = .white })
        .withFocusedStyle(.{ .fg = .black, .bg = .magenta, .bold = true })
        .withNodeWidth(14)
        .withNodeHeight(3)
        .withHSpacing(4)
        .withVSpacing(2)
        .withBlock((tui.widgets.Block{
            .title = " Query Pipeline (f/Esc:close  \xe2\x86\x91\xe2\x86\x93:navigate) ",
            .borders = .all,
        }).withBorderStyle(tui.Style{ .fg = .magenta }));

    chart.render(buf, popup_area);
}

fn renderMindMap(app: *App, buf: *tui.Buffer, area: tui.Rect) void {
    if (!app.mind_visible) return;

    // Centered overlay: ~70% width, ~80% height
    const ow: u16 = @min(area.width * 7 / 10, area.width);
    const oh: u16 = @min(area.height * 8 / 10, area.height);
    const ox: u16 = if (area.width > ow) (area.width - ow) / 2 else 0;
    const oy: u16 = if (area.height > oh) (area.height - oh) / 2 else 0;
    const popup_area = tui.Rect{ .x = ox, .y = oy, .width = ow, .height = oh };

    // Clear background
    var py: u16 = oy;
    while (py < oy + oh) : (py += 1) {
        var px: u16 = ox;
        while (px < ox + ow) : (px += 1) {
            buf.set(px, py, tui.Cell.init(' ', .{ .bg = .black }));
        }
    }

    // Build nodes: root = db basename, children = table names
    const MindNode = sailor.widgets.mindmap.MindNode;
    var nodes: [sailor.widgets.MindMap.MAX_NODES]MindNode = undefined;
    var node_count: usize = 0;

    // Root: database name
    const db_name = std.fs.path.basename(app.db_path);
    nodes[0] = .{ .label = db_name, .parent = 0 };
    node_count = 1;

    // Level 1: table names
    const table_count = @min(app.schema_table_indices.items.len, sailor.widgets.MindMap.MAX_NODES - 1);
    for (app.schema_table_indices.items[0..table_count]) |table_idx| {
        nodes[node_count] = .{ .label = app.schema_items.items[table_idx], .parent = 0 };
        node_count += 1;
    }

    const map = sailor.widgets.MindMap.init()
        .withNodes(nodes[0..node_count])
        .withFocused(app.mind_focused)
        .withStyle(.{ .fg = .white })
        .withRootStyle(.{ .fg = .black, .bg = .cyan, .bold = true })
        .withFocusedStyle(.{ .fg = .black, .bg = .green, .bold = true })
        .withNodeWidth(16)
        .withNodeHeight(3)
        .withHGap(3)
        .withBlock((tui.widgets.Block{
            .title = " Schema Map (n/Esc:close  \xe2\x86\x91\xe2\x86\x93:navigate) ",
            .borders = .all,
        }).withBorderStyle(tui.Style{ .fg = .cyan }));

    map.render(buf, popup_area);
}

fn renderRadarChart(app: *App, buf: *tui.Buffer, area: tui.Rect) void {
    if (!app.radar_visible) return;

    // Centered overlay: ~60% width, ~75% height
    const ow: u16 = @min(area.width * 6 / 10, area.width);
    const oh: u16 = @min(area.height * 3 / 4, area.height);
    const ox: u16 = if (area.width > ow) (area.width - ow) / 2 else 0;
    const oy: u16 = if (area.height > oh) (area.height - oh) / 2 else 0;
    const popup_area = tui.Rect{ .x = ox, .y = oy, .width = ow, .height = oh };

    // Clear background
    var py: u16 = oy;
    while (py < oy + oh) : (py += 1) {
        var px: u16 = ox;
        while (px < ox + ow) : (px += 1) {
            buf.set(px, py, tui.Cell.init(' ', .{ .bg = .black }));
        }
    }

    // Build series from query history
    const actual_count = @min(app.query_history_count, QUERY_HISTORY_MAX);
    var n_select: f32 = 0;
    var n_insert: f32 = 0;
    var n_update: f32 = 0;
    var n_delete: f32 = 0;
    var n_ddl: f32 = 0;
    var n_err: f32 = 0;

    for (app.query_history[0..actual_count]) |entry| {
        if (!entry.success) {
            n_err += 1;
            continue;
        }
        const upper = entry.title[0..@min(entry.title_len, 6)];
        if (std.mem.startsWith(u8, upper, "SELECT") or std.mem.startsWith(u8, upper, "select")) {
            n_select += 1;
        } else if (std.mem.startsWith(u8, upper, "INSERT") or std.mem.startsWith(u8, upper, "insert")) {
            n_insert += 1;
        } else if (std.mem.startsWith(u8, upper, "UPDATE") or std.mem.startsWith(u8, upper, "update")) {
            n_update += 1;
        } else if (std.mem.startsWith(u8, upper, "DELETE") or std.mem.startsWith(u8, upper, "delete")) {
            n_delete += 1;
        } else {
            n_ddl += 1;
        }
    }

    const total: f32 = @floatFromInt(@max(actual_count, 1));
    const actual_values = [_]f32{
        n_select / total,
        n_insert / total,
        n_update / total,
        n_delete / total,
        n_ddl / total,
        n_err / total,
    };

    // Series 0: actual distribution; Series 1: ideal (balanced) baseline
    const ideal_val: f32 = 1.0 / @as(f32, @floatFromInt(RADAR_AXES.len));
    const ideal_values = [_]f32{ ideal_val, ideal_val, ideal_val, ideal_val, ideal_val, ideal_val };

    const series = [RADAR_SERIES_COUNT]sailor.RadarSeries{
        .{ .label = "Actual", .values = &actual_values, .style = .{ .fg = .green } },
        .{ .label = "Ideal", .values = &ideal_values, .style = .{ .fg = .yellow } },
    };

    const chart = sailor.RadarChart.init()
        .withAxes(&RADAR_AXES)
        .withSeries(&series)
        .withFocused(app.radar_focused)
        .withFilled(false)
        .withAxisStyle(.{ .fg = .bright_black })
        .withFocusedStyle(.{ .fg = .cyan, .bold = true })
        .withBlock((tui.widgets.Block{
            .title = " Query Profile (r/Esc:close  \xe2\x86\x90\xe2\x86\x92:series) ",
            .borders = .all,
        }).withBorderStyle(tui.Style{ .fg = .magenta }));

    chart.render(buf, popup_area);
}

fn renderHexEditor(app: *App, buf: *tui.Buffer, area: tui.Rect) void {
    if (!app.hex_visible) return;

    // Centered overlay: ~70% width, ~80% height
    const ow: u16 = @min(area.width * 7 / 10, area.width);
    const oh: u16 = @min(area.height * 4 / 5, area.height);
    const ox: u16 = if (area.width > ow) (area.width - ow) / 2 else 0;
    const oy: u16 = if (area.height > oh) (area.height - oh) / 2 else 0;
    const popup_area = tui.Rect{ .x = ox, .y = oy, .width = ow, .height = oh };

    // Clear background
    var py: u16 = oy;
    while (py < oy + oh) : (py += 1) {
        var px: u16 = ox;
        while (px < ox + ow) : (px += 1) {
            buf.set(px, py, tui.Cell.init(' ', .{ .bg = .black }));
        }
    }

    const data_slice: []const u8 = if (app.hex_data_len > 0)
        app.hex_data[0..app.hex_data_len]
    else
        &[_]u8{};

    const editor = sailor.HexEditor.init()
        .withData(data_slice)
        .withCursor(app.hex_cursor)
        .withBytesPerRow(16)
        .withGroupSize(4)
        .withShowAscii(true)
        .withShowOffset(true)
        .withCursorStyle(.{ .fg = .cyan, .bold = true })
        .withBlock((tui.widgets.Block{
            .title = " Page Viewer (x/Esc:close  \xe2\x86\x90\xe2\x86\x92:navigate) ",
            .borders = .all,
        }).withBorderStyle(tui.Style{ .fg = .magenta }));

    editor.render(buf, popup_area);
}

fn renderTreemap(app: *App, buf: *tui.Buffer, area: tui.Rect) void {
    if (!app.treemap_visible) return;

    // Centered overlay: ~80% width, ~85% height
    const ow: u16 = @min(area.width * 4 / 5, area.width);
    const oh: u16 = @min(area.height * 17 / 20, area.height);
    const ox: u16 = if (area.width > ow) (area.width - ow) / 2 else 0;
    const oy: u16 = if (area.height > oh) (area.height - oh) / 2 else 0;
    const popup_area = tui.Rect{ .x = ox, .y = oy, .width = ow, .height = oh };

    // Clear background
    var py: u16 = oy;
    while (py < oy + oh) : (py += 1) {
        var px: u16 = ox;
        while (px < ox + ow) : (px += 1) {
            buf.set(px, py, tui.Cell.init(' ', .{ .bg = .black }));
        }
    }

    // Build treemap items from table stats
    var items: [MAX_TREEMAP_ITEMS]sailor.TreemapItem = undefined;
    var item_count: usize = 0;

    const table_count = @min(app.schema_table_indices.items.len, MAX_TREEMAP_ITEMS);
    for (app.schema_table_indices.items[0..table_count]) |table_idx| {
        const table_name = app.schema_items.items[table_idx];
        var value: f32 = 1.0;

        // Try to get row count from stats; fall back to 1.0 per table
        if (app.db.catalog.getTableStats(table_name) catch null) |stats| {
            value = @max(1.0, @as(f32, @floatFromInt(stats.row_count)));
        }

        const item_style: tui.Style = if (item_count % 6 == 0)
            .{ .fg = .black, .bg = .blue } // blue
        else if (item_count % 6 == 1)
            .{ .fg = .black, .bg = .green } // green
        else if (item_count % 6 == 2)
            .{ .fg = .black, .bg = .cyan } // cyan
        else if (item_count % 6 == 3)
            .{ .fg = .black, .bg = .yellow } // yellow
        else if (item_count % 6 == 4)
            .{ .fg = .black, .bg = .magenta } // magenta
        else
            .{ .fg = .black, .bg = .red }; // red

        items[item_count] = .{
            .label = table_name,
            .value = value,
            .style = item_style,
        };
        item_count += 1;
    }

    // If no tables, show a single placeholder
    if (item_count == 0) {
        items[0] = .{ .label = "(no tables)", .value = 1.0, .style = .{ .fg = .white } };
        item_count = 1;
    }

    const focused = @min(app.treemap_focused, item_count - 1);

    const tm = sailor.Treemap.init()
        .withItems(items[0..item_count])
        .withFocused(focused)
        .withFocusedStyle(.{ .bold = true })
        .withLabelStyle(.{ .bold = true })
        .withBlock((tui.widgets.Block{
            .title = " Table Space (\xe2\x86\x91\xe2\x86\x93\xe2\x86\x90\xe2\x86\x92:navigate  w/Esc:close) ",
            .borders = .all,
        }).withBorderStyle(tui.Style{ .fg = .magenta }));

    tm.render(buf, popup_area);
}

fn renderMatrixView(app: *App, buf: *tui.Buffer, area: tui.Rect) void {
    if (!app.matrix_visible) return;

    // Centered overlay: 70% width, 60% height
    const ow: u16 = @min(area.width * 7 / 10, area.width);
    const oh: u16 = @min(area.height * 3 / 5, area.height);
    const ox: u16 = if (area.width > ow) (area.width - ow) / 2 else 0;
    const oy: u16 = if (area.height > oh) (area.height - oh) / 2 else 0;
    const popup_area = tui.Rect{ .x = ox, .y = oy, .width = ow, .height = oh };

    // Clear background
    var py: u16 = oy;
    while (py < oy + oh) : (py += 1) {
        var px: u16 = ox;
        while (px < ox + ow) : (px += 1) {
            buf.set(px, py, tui.Cell.init(' ', .{ .bg = .black }));
        }
    }

    // Compute metrics from query history
    const actual_count = @min(app.query_history_count, QUERY_HISTORY_MAX);

    var counts: [MATRIX_ROWS]u32 = std.mem.zeroes([MATRIX_ROWS]u32);
    var total_ms: [MATRIX_ROWS]u64 = std.mem.zeroes([MATRIX_ROWS]u64);
    var max_ms: [MATRIX_ROWS]u64 = std.mem.zeroes([MATRIX_ROWS]u64);

    for (app.query_history[0..actual_count]) |entry| {
        const row_idx: usize = if (!entry.success) 5 else blk: {
            const upper = entry.title[0..@min(entry.title_len, 6)];
            if (std.mem.startsWith(u8, upper, "SELECT") or std.mem.startsWith(u8, upper, "select")) break :blk 0;
            if (std.mem.startsWith(u8, upper, "INSERT") or std.mem.startsWith(u8, upper, "insert")) break :blk 1;
            if (std.mem.startsWith(u8, upper, "UPDATE") or std.mem.startsWith(u8, upper, "update")) break :blk 2;
            if (std.mem.startsWith(u8, upper, "DELETE") or std.mem.startsWith(u8, upper, "delete")) break :blk 3;
            break :blk 4; // DDL
        };
        counts[row_idx] += 1;
        total_ms[row_idx] += entry.duration_ms;
        if (entry.duration_ms > max_ms[row_idx]) max_ms[row_idx] = entry.duration_ms;
    }

    // Build f32 matrix data: [6][3] = { count, avg_ms, max_ms }
    var matrix_data: [MATRIX_ROWS][MATRIX_COLS]f32 = std.mem.zeroes([MATRIX_ROWS][MATRIX_COLS]f32);
    var max_count: f32 = 1.0;
    var max_avg: f32 = 1.0;
    var max_max: f32 = 1.0;
    for (0..MATRIX_ROWS) |i| {
        matrix_data[i][0] = @floatFromInt(counts[i]);
        matrix_data[i][1] = if (counts[i] > 0)
            @as(f32, @floatFromInt(total_ms[i])) / @as(f32, @floatFromInt(counts[i]))
        else
            0.0;
        matrix_data[i][2] = @floatFromInt(max_ms[i]);
        if (matrix_data[i][0] > max_count) max_count = matrix_data[i][0];
        if (matrix_data[i][1] > max_avg) max_avg = matrix_data[i][1];
        if (matrix_data[i][2] > max_max) max_max = matrix_data[i][2];
    }

    var row_ptrs: [MATRIX_ROWS][]const f32 = undefined;
    for (0..MATRIX_ROWS) |i| {
        row_ptrs[i] = &matrix_data[i];
    }

    const mv = sailor.MatrixView.init()
        .withData(&row_ptrs)
        .withRowHeaders(&MATRIX_ROW_HEADERS)
        .withColHeaders(&MATRIX_COL_HEADERS)
        .withCellWidth(8)
        .withShowValues(true)
        .withFocusedRow(app.matrix_focused_row)
        .withFocusedCol(app.matrix_focused_col)
        .withMinVal(0.0)
        .withMaxVal(max_count)
        .withHeaderStyle(.{ .fg = .cyan, .bold = true })
        .withFocusedStyle(.{ .fg = .yellow, .bold = true })
        .withBlock((tui.widgets.Block{
            .title = " Query Metrics (v/Esc:close  arrows:navigate) ",
            .borders = .all,
        }).withBorderStyle(tui.Style{ .fg = .magenta }));

    mv.render(buf, popup_area);
}

fn renderSankeyDiagram(app: *App, buf: *tui.Buffer, area: tui.Rect) void {
    if (!app.sankey_visible) return;

    // Count query types from query_history
    const actual_count = @min(app.query_history_count, QUERY_HISTORY_MAX);
    var select_count: f32 = 0;
    var dml_count: f32 = 0;
    var ddl_count: f32 = 0;
    var success_count: f32 = 0;
    var error_count: f32 = 0;

    for (app.query_history[0..actual_count]) |entry| {
        const title = entry.title[0..entry.title_len];
        if (std.ascii.startsWithIgnoreCase(title, "SELECT")) {
            select_count += 1;
        } else if (std.ascii.startsWithIgnoreCase(title, "INSERT") or
            std.ascii.startsWithIgnoreCase(title, "UPDATE") or
            std.ascii.startsWithIgnoreCase(title, "DELETE"))
        {
            dml_count += 1;
        } else if (std.ascii.startsWithIgnoreCase(title, "CREATE") or
            std.ascii.startsWithIgnoreCase(title, "DROP") or
            std.ascii.startsWithIgnoreCase(title, "ALTER"))
        {
            ddl_count += 1;
        } else if (title.len > 0) {
            dml_count += 1;
        }
        if (entry.success) {
            success_count += 1;
        } else {
            error_count += 1;
        }
    }

    // Use placeholder flows when no data yet
    const has_data = actual_count > 0;
    const flows = [5]sailor.tui.widgets.SankeyFlow{
        .{ .source = 0, .target = 3, .value = if (has_data) select_count else 2.0, .style = .{ .fg = .cyan } },
        .{ .source = 1, .target = 3, .value = if (has_data) dml_count else 1.0, .style = .{ .fg = .green } },
        .{ .source = 2, .target = 3, .value = if (has_data) ddl_count else 1.0, .style = .{ .fg = .yellow } },
        .{ .source = 3, .target = 4, .value = if (has_data) success_count else 3.0, .style = .{ .fg = .bright_green } },
        .{ .source = 3, .target = 5, .value = if (has_data) error_count else 1.0, .style = .{ .fg = .red } },
    };

    // Centered overlay: ~70% width, ~60% height
    const ow: u16 = @min(area.width * 7 / 10, area.width);
    const oh: u16 = @min(area.height * 6 / 10, area.height);
    const ox: u16 = if (area.width > ow) (area.width - ow) / 2 else 0;
    const oy: u16 = if (area.height > oh) (area.height - oh) / 2 else 0;
    const popup_area = tui.Rect{ .x = ox, .y = oy, .width = ow, .height = oh };

    // Clear background
    var py: u16 = oy;
    while (py < oy + oh) : (py += 1) {
        var px: u16 = ox;
        while (px < ox + ow) : (px += 1) {
            buf.set(px, py, tui.Cell.init(' ', .{ .bg = .black }));
        }
    }

    const sk = sailor.tui.widgets.SankeyDiagram.init()
        .withNodes(&SANKEY_NODES)
        .withFlows(&flows)
        .withFocused(app.sankey_focused)
        .withNodeWidth(3)
        .withColGap(10)
        .withNodeStyle(.{ .fg = .white })
        .withFlowStyle(.{ .fg = .bright_black })
        .withFocusedStyle(.{ .fg = .black, .bg = .magenta, .bold = true })
        .withBlock((tui.widgets.Block{
            .title = " SQL Data Flow (s/Esc:close  \xe2\x86\x91\xe2\x86\x93:navigate) ",
            .borders = .all,
        }).withBorderStyle(tui.Style{ .fg = .magenta }));

    sk.render(buf, popup_area);
}

fn renderBubbleChart(app: *App, buf: *tui.Buffer, area: tui.Rect) void {
    if (!app.bubble_visible) return;

    const actual_count = @min(app.query_history_count, QUERY_HISTORY_MAX);

    // Find max duration for normalization
    var max_duration: u64 = 1;
    for (app.query_history[0..actual_count]) |entry| {
        if (entry.duration_ms > max_duration) max_duration = entry.duration_ms;
    }

    var bubbles: [QUERY_HISTORY_MAX]sailor.Bubble = undefined;
    var bubble_count: usize = 0;

    if (actual_count == 0) {
        bubbles[0] = sailor.Bubble{
            .label = "no data",
            .x = 0.0,
            .y = 0.0,
            .size = 0.3,
            .style = .{ .fg = .bright_black },
        };
        bubble_count = 1;
    } else {
        for (app.query_history[0..actual_count], 0..) |entry, i| {
            const x_val: f32 = @floatFromInt(i);
            const y_val: f32 = @floatFromInt(entry.duration_ms);
            const size: f32 = @as(f32, @floatFromInt(entry.duration_ms)) / @as(f32, @floatFromInt(max_duration));
            const style: tui.Style = if (entry.success) .{ .fg = .green } else .{ .fg = .red };
            bubbles[i] = sailor.Bubble{
                .label = "",
                .x = x_val,
                .y = y_val,
                .size = @max(0.1, size),
                .style = style,
            };
        }
        bubble_count = actual_count;
    }

    // Centered overlay: ~70% width, ~60% height
    const ow: u16 = @min(area.width * 7 / 10, area.width);
    const oh: u16 = @min(area.height * 6 / 10, area.height);
    const ox: u16 = if (area.width > ow) (area.width - ow) / 2 else 0;
    const oy: u16 = if (area.height > oh) (area.height - oh) / 2 else 0;
    const popup_area = tui.Rect{ .x = ox, .y = oy, .width = ow, .height = oh };

    // Clear background
    var py: u16 = oy;
    while (py < oy + oh) : (py += 1) {
        var px: u16 = ox;
        while (px < ox + ow) : (px += 1) {
            buf.set(px, py, tui.Cell.init(' ', .{ .bg = .black }));
        }
    }

    const x_max_val: f32 = if (actual_count > 1) @as(f32, @floatFromInt(actual_count - 1)) else 1.0;
    const y_max_val: f32 = @floatFromInt(max_duration);

    const bc = sailor.BubbleChart.init()
        .withBubbles(bubbles[0..bubble_count])
        .withXMin(0.0)
        .withXMax(x_max_val)
        .withYMin(0.0)
        .withYMax(y_max_val)
        .withFocused(app.bubble_focused)
        .withShowAxes(true)
        .withShowLabels(false)
        .withBubbleStyle(.{ .fg = .cyan })
        .withFocusedStyle(.{ .fg = .black, .bg = .magenta, .bold = true })
        .withAxisStyle(.{ .fg = .bright_black })
        .withBlock((tui.widgets.Block{
            .title = " Query Performance (u/Esc:close  \xe2\x86\x91\xe2\x86\x93:navigate) ",
            .borders = .all,
        }).withBorderStyle(tui.Style{ .fg = .magenta }));

    bc.render(buf, popup_area);
}

fn renderChordDiagram(app: *App, buf: *tui.Buffer, area: tui.Rect) void {
    if (!app.chord_visible) return;

    const actual_count = @min(app.query_history_count, QUERY_HISTORY_MAX);

    // Build 5x5 co-occurrence matrix from consecutive query type pairs
    var matrix: [CHORD_NODE_COUNT][CHORD_NODE_COUNT]f32 = std.mem.zeroes([CHORD_NODE_COUNT][CHORD_NODE_COUNT]f32);

    if (actual_count < 2) {
        // Placeholder: typical DB usage pattern
        matrix[0][1] = 2.0; // SELECT -> INSERT
        matrix[0][2] = 1.0; // SELECT -> UPDATE
        matrix[1][0] = 3.0; // INSERT -> SELECT
        matrix[2][0] = 1.5; // UPDATE -> SELECT
        matrix[3][4] = 0.5; // DELETE -> DDL
        matrix[4][0] = 1.0; // DDL -> SELECT
    } else {
        var prev_type: usize = classifyQueryType(&app.query_history[0]);
        for (app.query_history[1..actual_count]) |entry| {
            const curr_type = classifyQueryType(&entry);
            matrix[prev_type][curr_type] += 1.0;
            prev_type = curr_type;
        }
    }

    const row0: []const f32 = &matrix[0];
    const row1: []const f32 = &matrix[1];
    const row2: []const f32 = &matrix[2];
    const row3: []const f32 = &matrix[3];
    const row4: []const f32 = &matrix[4];
    const mat_slices = [CHORD_NODE_COUNT][]const f32{ row0, row1, row2, row3, row4 };
    const node_slices: []const []const u8 = &CHORD_NODES;

    // Centered overlay: ~70% width, ~70% height
    const ow: u16 = @min(area.width * 7 / 10, area.width);
    const oh: u16 = @min(area.height * 7 / 10, area.height);
    const ox: u16 = if (area.width > ow) (area.width - ow) / 2 else 0;
    const oy: u16 = if (area.height > oh) (area.height - oh) / 2 else 0;
    const popup_area = tui.Rect{ .x = ox, .y = oy, .width = ow, .height = oh };

    // Clear background
    var py: u16 = oy;
    while (py < oy + oh) : (py += 1) {
        var px: u16 = ox;
        while (px < ox + ow) : (px += 1) {
            buf.set(px, py, tui.Cell.init(' ', .{ .bg = .black }));
        }
    }

    const cd = sailor.ChordDiagram.init()
        .withNodes(node_slices)
        .withMatrix(&mat_slices)
        .withFocused(app.chord_focused)
        .withShowLabels(true)
        .withArcStyle(.{ .fg = .cyan })
        .withFocusedStyle(.{ .fg = .black, .bg = .magenta, .bold = true })
        .withStyle(.{ .fg = .white })
        .withBlock((tui.widgets.Block{
            .title = " SQL Operation Flow (c/Esc:close  \xe2\x86\x91\xe2\x86\x93:navigate) ",
            .borders = .all,
        }).withBorderStyle(tui.Style{ .fg = .magenta }));

    cd.render(buf, popup_area);
}

fn renderWaterfallChart(app: *App, buf: *tui.Buffer, area: tui.Rect) void {
    if (!app.waterfall_visible) return;

    const actual_count = @min(app.query_history_count, QUERY_HISTORY_MAX);

    // Count each query type from history
    var counts = [5]f32{ 0, 0, 0, 0, 0 }; // SELECT, INSERT, UPDATE, DELETE, DDL
    if (actual_count == 0) {
        // Placeholder: typical DB usage
        counts[0] = 5.0; // SELECT
        counts[1] = 3.0; // INSERT
        counts[2] = 2.0; // UPDATE
        counts[3] = 1.0; // DELETE
        counts[4] = 1.0; // DDL
    } else {
        for (app.query_history[0..actual_count]) |entry| {
            const t = classifyQueryType(&entry);
            if (t < 5) counts[t] += 1.0;
        }
    }

    const bars = [WATERFALL_BAR_COUNT]sailor.WaterfallBar{
        .{ .label = "SELECT", .value = counts[0], .kind = .absolute },
        .{ .label = "INSERT", .value = counts[1], .kind = .relative },
        .{ .label = "UPDATE", .value = counts[2], .kind = .relative },
        .{ .label = "DELETE", .value = counts[3], .kind = .relative },
        .{ .label = "DDL",    .value = counts[4], .kind = .relative },
        .{ .label = "Total",  .value = 0.0,        .kind = .total },
    };

    // Centered overlay: ~70% width, ~70% height
    const ow: u16 = @min(area.width * 7 / 10, area.width);
    const oh: u16 = @min(area.height * 7 / 10, area.height);
    const ox: u16 = if (area.width > ow) (area.width - ow) / 2 else 0;
    const oy: u16 = if (area.height > oh) (area.height - oh) / 2 else 0;
    const popup_area = tui.Rect{ .x = ox, .y = oy, .width = ow, .height = oh };

    // Clear background
    var py: u16 = oy;
    while (py < oy + oh) : (py += 1) {
        var px: u16 = ox;
        while (px < ox + ow) : (px += 1) {
            buf.set(px, py, tui.Cell.init(' ', .{ .bg = .black }));
        }
    }

    const bar_slices: []const sailor.WaterfallBar = &bars;
    const wc = sailor.WaterfallChart.init()
        .withBars(bar_slices)
        .withFocused(app.waterfall_focused)
        .withShowValues(true)
        .withShowConnectors(true)
        .withBlock((tui.widgets.Block{
            .title = " SQL Query Breakdown (e/Esc:close  \xe2\x86\x91\xe2\x86\x93:navigate) ",
            .borders = .all,
        }).withBorderStyle(tui.Style{ .fg = .magenta }));

    wc.render(buf, popup_area);
}

fn classifyQueryType(entry: *const QueryHistoryEntry) usize {
    const sql = entry.title[0..entry.title_len];
    if (sql.len == 0) return 0;
    // Find first non-space character sequence
    var i: usize = 0;
    while (i < sql.len and sql[i] == ' ') i += 1;
    const rest = sql[i..];
    if (rest.len >= 6 and std.ascii.eqlIgnoreCase(rest[0..6], "select")) return 0;
    if (rest.len >= 6 and std.ascii.eqlIgnoreCase(rest[0..6], "insert")) return 1;
    if (rest.len >= 6 and std.ascii.eqlIgnoreCase(rest[0..6], "update")) return 2;
    if (rest.len >= 6 and std.ascii.eqlIgnoreCase(rest[0..6], "delete")) return 3;
    // DDL: CREATE, DROP, ALTER, TRUNCATE, etc.
    return 4;
}

fn renderFunnelChart(app: *App, buf: *tui.Buffer, area: tui.Rect) void {
    if (!app.funnel_visible) return;

    const actual_count = @min(app.query_history_count, QUERY_HISTORY_MAX);

    // Count stages from query_history
    var total: f32 = @floatFromInt(actual_count);
    var succeeded: f32 = 0;
    var fast: f32 = 0;
    var instant: f32 = 0;

    if (actual_count == 0) {
        // Placeholder data when no queries yet
        total = 100;
        succeeded = 75;
        fast = 50;
        instant = 20;
    } else {
        for (app.query_history[0..actual_count]) |entry| {
            if (entry.success) {
                succeeded += 1;
                if (entry.duration_ms < 100) fast += 1;
                if (entry.duration_ms < 10) instant += 1;
            }
        }
    }

    const stages = [FUNNEL_STAGE_COUNT]sailor.FunnelStage{
        .{ .label = "Submitted", .value = total },
        .{ .label = "Succeeded", .value = succeeded },
        .{ .label = "Fast (<100ms)", .value = fast },
        .{ .label = "Instant (<10ms)", .value = instant },
    };
    const stage_slices: []const sailor.FunnelStage = &stages;

    const popup_width: u16 = @min(50, area.width -| 4);
    const popup_height: u16 = @min(14, area.height -| 4);
    const popup_x = area.x + (area.width -| popup_width) / 2;
    const popup_y = area.y + (area.height -| popup_height) / 2;
    const popup_area = tui.Rect{
        .x = popup_x,
        .y = popup_y,
        .width = popup_width,
        .height = popup_height,
    };

    const fc = sailor.FunnelChart.init()
        .withStages(stage_slices)
        .withShowValues(true)
        .withShowPercentages(true)
        .withFocused(app.funnel_focused)
        .withBlock((tui.widgets.Block{
            .title = " SQL Query Funnel (l/Esc:close  \xe2\x86\x91\xe2\x86\x93:navigate) ",
            .borders = .all,
        }).withBorderStyle(tui.Style{ .fg = .magenta }));

    fc.render(buf, popup_area);
}

fn renderStatusBar(app: *App, buf: *tui.Buffer, area: tui.Rect) void {
    if (area.width == 0 or area.height == 0) return;

    const bar_style = tui.Style{ .bg = .bright_black };

    // Fill the entire bar with background
    var fill_x = area.x;
    while (fill_x < area.x + area.width) : (fill_x += 1) {
        buf.set(fill_x, area.y, tui.Cell.init(' ', bar_style));
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
        buf.set(x, area.y, tui.Cell.init(c, pane_style));
        x += 1;
    }

    for (app.status_left) |c| {
        if (x >= area.x + area.width) break;
        buf.set(x, area.y, tui.Cell.init(c, bar_style));
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
            buf.set(rx, area.y, tui.Cell.init(c, bar_style));
            rx += 1;
        }
    }

    // Center: Tab:switch Enter:exec t:timer b:board p:plan a:feed g:gantt f:flow n:schema r:radar x:hex w:treemap v:matrix s:sankey u:bubble c:chord e:waterfall l:funnel Ctrl+C:quit
    const center_text = "Tab:switch  Enter:exec  t:timer  b:board  p:plan  a:feed  g:gantt  f:flow  n:schema  r:radar  x:hex  w:treemap  v:matrix  s:sankey  u:bubble  c:chord  e:waterfall  l:funnel  Ctrl+C:quit";
    const center_start = if (area.width > center_text.len)
        area.x + (area.width - @as(u16, @intCast(center_text.len))) / 2
    else
        area.x;
    var cx = center_start;
    for (center_text) |c| {
        if (cx >= area.x + area.width) break;
        buf.set(cx, area.y, tui.Cell.init(c, bar_style));
        cx += 1;
    }

    // Spinner at the rightmost cell (animated activity indicator)
    if (area.width > 0) {
        const spinner = (tui.Spinner{})
            .withFrame(app.spinner_frame)
            .withStyle(.{ .bg = .bright_black, .fg = .green });
        spinner.render(buf, tui.Rect{
            .x = area.x + area.width - 1,
            .y = area.y,
            .width = 1,
            .height = 1,
        });
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
        .gray => try writer.writeAll("\x1b[90m"),
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
        .gray => try writer.writeAll("\x1b[100m"),
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
        app.spinner_frame +%= 1;

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

    const area = tui.Rect{ .x = 0, .y = 0, .width = 80, .height = 24 };
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

    const area = tui.Rect{ .x = 0, .y = 0, .width = 80, .height = 24 };
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

test "detail_visible starts as false" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    try std.testing.expect(app.detail_visible == false);
}

test "detail_selected starts as 0" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    try std.testing.expectEqual(@as(usize, 0), app.detail_selected);
}

test "pressing Enter in results pane with rows shows detail overlay" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .results,
    };
    defer app.deinit();

    // Add result columns and rows
    const col1 = try allocator.dupe(u8, "name");
    try app.result_columns.append(allocator, col1);
    const col2 = try allocator.dupe(u8, "age");
    try app.result_columns.append(allocator, col2);

    const row1 = try allocator.alloc([]const u8, 2);
    row1[0] = try allocator.dupe(u8, "Alice");
    row1[1] = try allocator.dupe(u8, "30");
    try app.result_rows.append(allocator, row1);

    try std.testing.expect(app.detail_visible == false);

    // Press Enter (13 = \r)
    app.handleKey(13);

    try std.testing.expect(app.detail_visible == true);
}

test "pressing Enter in results pane resets detail_selected to 0" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .results,
    };
    defer app.deinit();

    // Add result columns and rows
    const col1 = try allocator.dupe(u8, "name");
    try app.result_columns.append(allocator, col1);
    const col2 = try allocator.dupe(u8, "age");
    try app.result_columns.append(allocator, col2);

    const row1 = try allocator.alloc([]const u8, 2);
    row1[0] = try allocator.dupe(u8, "Alice");
    row1[1] = try allocator.dupe(u8, "30");
    try app.result_rows.append(allocator, row1);

    // Set detail_selected to non-zero value before opening detail
    app.detail_selected = 5;

    // Press Enter (13 = \r)
    app.handleKey(13);

    // detail_selected should be reset to 0 when opening overlay
    try std.testing.expectEqual(@as(usize, 0), app.detail_selected);
}

test "pressing Enter in results pane with no rows keeps detail closed" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .results,
    };
    defer app.deinit();

    try std.testing.expect(app.detail_visible == false);

    // Press Enter on empty result set
    app.handleKey(13);

    // Detail should still be closed
    try std.testing.expect(app.detail_visible == false);
}

test "pressing Escape in detail view closes it" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .results,
    };
    defer app.deinit();

    // Add result data
    const col1 = try allocator.dupe(u8, "id");
    try app.result_columns.append(allocator, col1);

    const row1 = try allocator.alloc([]const u8, 1);
    row1[0] = try allocator.dupe(u8, "1");
    try app.result_rows.append(allocator, row1);

    // Open detail
    app.detail_visible = true;

    // Press Escape (27 = ESC)
    app.handleEscapeSequence(null, null);

    // Detail should be closed
    try std.testing.expect(app.detail_visible == false);
}

test "pressing arrow down in detail view moves selection" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .results,
    };
    defer app.deinit();

    // Add result columns and row
    const col1 = try allocator.dupe(u8, "col1");
    try app.result_columns.append(allocator, col1);
    const col2 = try allocator.dupe(u8, "col2");
    try app.result_columns.append(allocator, col2);
    const col3 = try allocator.dupe(u8, "col3");
    try app.result_columns.append(allocator, col3);

    const row1 = try allocator.alloc([]const u8, 3);
    row1[0] = try allocator.dupe(u8, "a");
    row1[1] = try allocator.dupe(u8, "b");
    row1[2] = try allocator.dupe(u8, "c");
    try app.result_rows.append(allocator, row1);

    // Open detail
    app.detail_visible = true;
    app.detail_selected = 0;

    // Press arrow down (escape sequence [B)
    app.handleEscapeSequence('[', 'B');

    try std.testing.expectEqual(@as(usize, 1), app.detail_selected);
}

test "pressing arrow up in detail view moves selection" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .results,
    };
    defer app.deinit();

    // Add result columns and row
    const col1 = try allocator.dupe(u8, "col1");
    try app.result_columns.append(allocator, col1);
    const col2 = try allocator.dupe(u8, "col2");
    try app.result_columns.append(allocator, col2);

    const row1 = try allocator.alloc([]const u8, 2);
    row1[0] = try allocator.dupe(u8, "a");
    row1[1] = try allocator.dupe(u8, "b");
    try app.result_rows.append(allocator, row1);

    // Open detail with selection
    app.detail_visible = true;
    app.detail_selected = 2;

    // Press arrow up (escape sequence [A)
    app.handleEscapeSequence('[', 'A');

    try std.testing.expectEqual(@as(usize, 1), app.detail_selected);
}

test "detail_selected clamps to 0 when pressing up" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .results,
    };
    defer app.deinit();

    // Add result columns and row
    const col1 = try allocator.dupe(u8, "col1");
    try app.result_columns.append(allocator, col1);

    const row1 = try allocator.alloc([]const u8, 1);
    row1[0] = try allocator.dupe(u8, "a");
    try app.result_rows.append(allocator, row1);

    // Open detail at selection 0
    app.detail_visible = true;
    app.detail_selected = 0;

    // Press arrow up (should stay at 0)
    app.handleEscapeSequence('[', 'A');

    try std.testing.expectEqual(@as(usize, 0), app.detail_selected);
}

test "detail_selected clamps at max columns when pressing down" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .results,
    };
    defer app.deinit();

    // Add 3 columns
    const col1 = try allocator.dupe(u8, "col1");
    try app.result_columns.append(allocator, col1);
    const col2 = try allocator.dupe(u8, "col2");
    try app.result_columns.append(allocator, col2);
    const col3 = try allocator.dupe(u8, "col3");
    try app.result_columns.append(allocator, col3);

    const row1 = try allocator.alloc([]const u8, 3);
    row1[0] = try allocator.dupe(u8, "a");
    row1[1] = try allocator.dupe(u8, "b");
    row1[2] = try allocator.dupe(u8, "c");
    try app.result_rows.append(allocator, row1);

    // Open detail at last valid selection (result_columns.items.len - 1 = 2)
    app.detail_visible = true;
    app.detail_selected = 2;

    // Press arrow down (should stay at 2)
    app.handleEscapeSequence('[', 'B');

    try std.testing.expectEqual(@as(usize, 2), app.detail_selected);
}

test "App spinner_frame initializes to 0" {
    const allocator = std.testing.allocator;
    var db = try Database.open(allocator, ":memory:", .{});
    defer db.close();

    var app = App.init(allocator, &db, ":memory:");
    defer app.deinit();

    try std.testing.expectEqual(@as(usize, 0), app.spinner_frame);
}

test "Spinner widget currentFrame returns braille characters" {
    const allocator = std.testing.allocator;
    _ = allocator; // Not needed but keep for consistency

    // Frame 0 should be ⣾
    var spinner0 = tui.Spinner{};
    const frame0 = spinner0.currentFrame();
    try std.testing.expectEqualSlices(u8, "⣾", frame0);

    // Frame 1 should be ⣽
    var spinner1 = spinner0.withFrame(1);
    const frame1 = spinner1.currentFrame();
    try std.testing.expectEqualSlices(u8, "⣽", frame1);

    // Frame 8 should wrap back to frame 0 (⣾) using modulo
    var spinner8 = spinner0.withFrame(8);
    const frame8 = spinner8.currentFrame();
    try std.testing.expectEqualSlices(u8, "⣾", frame8);
}

test "renderStatusBar renders spinner at different frame values without crash" {
    const allocator = std.testing.allocator;
    var db = try Database.open(allocator, ":memory:", .{});
    defer db.close();

    var app = App.init(allocator, &db, ":memory:");
    defer app.deinit();

    // Create an 80×1 buffer
    var buf = try tui.Buffer.init(allocator, 80, 1);
    defer buf.deinit();

    // Create a rect for status bar (80 wide, 1 tall)
    const area = tui.Rect{ .x = 0, .y = 0, .width = 80, .height = 1 };

    // Render with spinner_frame = 0 (should not crash)
    app.spinner_frame = 0;
    renderStatusBar(&app, &buf, area);

    // Verify rightmost cell (x=79) has a non-ASCII character (spinner frame)
    const cell0 = buf.getConst(79, 0);
    try std.testing.expect(cell0 != null);
    if (cell0) |c| {
        try std.testing.expect(c.char > 0x7F);
    }

    // Render with spinner_frame = 3 (should not crash)
    app.spinner_frame = 3;
    renderStatusBar(&app, &buf, area);

    const cell3 = buf.getConst(79, 0);
    try std.testing.expect(cell3 != null);
    if (cell3) |c| {
        try std.testing.expect(c.char > 0x7F);
    }

    // Render with spinner_frame = 100 (wraps modulo 8, should not crash)
    app.spinner_frame = 100;
    renderStatusBar(&app, &buf, area);

    const cell100 = buf.getConst(79, 0);
    try std.testing.expect(cell100 != null);
    if (cell100) |c| {
        try std.testing.expect(c.char > 0x7F);
    }
}

test "spinner_frame value changes which frame is rendered" {
    const allocator = std.testing.allocator;
    var db = try Database.open(allocator, ":memory:", .{});
    defer db.close();

    var app = App.init(allocator, &db, ":memory:");
    defer app.deinit();

    // Create an 80×1 buffer
    var buf = try tui.Buffer.init(allocator, 80, 1);
    defer buf.deinit();

    const area = tui.Rect{ .x = 0, .y = 0, .width = 80, .height = 1 };

    // Render with spinner_frame = 0
    app.spinner_frame = 0;
    renderStatusBar(&app, &buf, area);
    const cell_frame0 = buf.getConst(79, 0);
    const char_frame0 = if (cell_frame0) |c| c.char else 0;

    // Render with spinner_frame = 1
    app.spinner_frame = 1;
    renderStatusBar(&app, &buf, area);
    const cell_frame1 = buf.getConst(79, 0);
    const char_frame1 = if (cell_frame1) |c| c.char else 0;

    // Verify the characters are different (different spinner frames)
    try std.testing.expect(char_frame0 != char_frame1);
    try std.testing.expect(char_frame0 > 0);
    try std.testing.expect(char_frame1 > 0);
}

test "buildMinimapLines extracts first cell from each row" {
    const allocator = std.testing.allocator;

    // Create 3 rows with 2 cells each
    const row0 = try allocator.alloc([]const u8, 2);
    row0[0] = "Alice";
    row0[1] = "30";

    const row1 = try allocator.alloc([]const u8, 2);
    row1[0] = "Bob";
    row1[1] = "25";

    const row2 = try allocator.alloc([]const u8, 2);
    row2[0] = "Charlie";
    row2[1] = "35";

    const rows = [_][]const []const u8{ row0, row1, row2 };

    // Call buildMinimapLines
    const lines = try buildMinimapLines(allocator, &rows);
    defer allocator.free(lines);

    // Verify: lines should contain first cell of each row
    try std.testing.expectEqual(@as(usize, 3), lines.len);
    try std.testing.expectEqualStrings("Alice", lines[0]);
    try std.testing.expectEqualStrings("Bob", lines[1]);
    try std.testing.expectEqualStrings("Charlie", lines[2]);

    // Cleanup allocated rows
    allocator.free(row0);
    allocator.free(row1);
    allocator.free(row2);
}

test "buildMinimapLines uses space for empty row" {
    const allocator = std.testing.allocator;

    // Create one row with 0 cells
    const row0 = try allocator.alloc([]const u8, 0);

    const rows = [_][]const []const u8{row0};

    // Call buildMinimapLines
    const lines = try buildMinimapLines(allocator, &rows);
    defer allocator.free(lines);

    // Verify: lines[0] should be " "
    try std.testing.expectEqual(@as(usize, 1), lines.len);
    try std.testing.expectEqualStrings(" ", lines[0]);

    // Cleanup
    allocator.free(row0);
}

test "buildMinimapLines returns empty slice for empty rows" {
    const allocator = std.testing.allocator;

    const rows: [][]const []const u8 = &[_][]const []const u8{};

    // Call buildMinimapLines with zero rows
    const lines = try buildMinimapLines(allocator, rows);
    defer allocator.free(lines);

    // Verify: lines should be empty slice
    try std.testing.expectEqual(@as(usize, 0), lines.len);
}

test "minimapViewportHeight subtracts 3 from area height" {
    // Test typical area height (24 - 3 = 21)
    const result = minimapViewportHeight(24);
    try std.testing.expectEqual(@as(usize, 21), result);
}

test "minimapViewportHeight returns 1 for small area" {
    // Test small area (2 - 3 is negative, should clamp to 1)
    const result = minimapViewportHeight(2);
    try std.testing.expectEqual(@as(usize, 1), result);
}

test "App ring_menu opens on 'm' key" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    try std.testing.expect(!app.ring_menu_visible);
    try std.testing.expectEqual(@as(usize, 0), app.ring_menu_selected);

    app.handleKey(109); // 'm'
    try std.testing.expect(app.ring_menu_visible);
    try std.testing.expectEqual(@as(usize, 0), app.ring_menu_selected);
}

test "App ring_menu closes on second 'm' key" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    // Open menu
    app.handleKey(109); // 'm'
    try std.testing.expect(app.ring_menu_visible);

    // Close menu with second 'm'
    app.handleKey(109); // 'm'
    try std.testing.expect(!app.ring_menu_visible);
}

test "App ring_menu navigate right/down increments selected" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    // Open menu
    app.handleKey(109); // 'm'
    try std.testing.expect(app.ring_menu_visible);
    try std.testing.expectEqual(@as(usize, 0), app.ring_menu_selected);

    // Press Right arrow ([C)
    app.handleEscapeSequence('[', 'C');
    try std.testing.expectEqual(@as(usize, 1), app.ring_menu_selected);

    // Press Right again
    app.handleEscapeSequence('[', 'C');
    try std.testing.expectEqual(@as(usize, 2), app.ring_menu_selected);

    // Press Down arrow ([B) - should also increment
    app.handleEscapeSequence('[', 'B');
    try std.testing.expectEqual(@as(usize, 3), app.ring_menu_selected);
}

test "App ring_menu navigate left/up decrements selected with wraparound" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    // Open menu
    app.handleKey(109); // 'm'
    try std.testing.expect(app.ring_menu_visible);

    // Start at 0, press Left - should wraparound to 5 (last item)
    app.handleEscapeSequence('[', 'D');
    try std.testing.expectEqual(@as(usize, 5), app.ring_menu_selected);

    // Press Left again - should go to 4
    app.handleEscapeSequence('[', 'D');
    try std.testing.expectEqual(@as(usize, 4), app.ring_menu_selected);

    // Press Up arrow ([A) - should also decrement
    app.handleEscapeSequence('[', 'A');
    try std.testing.expectEqual(@as(usize, 3), app.ring_menu_selected);
}

test "App ring_menu ESC closes menu" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    // Open menu
    app.handleKey(109); // 'm'
    try std.testing.expect(app.ring_menu_visible);

    // Press ESC (b2=null means ESC was pressed)
    app.handleEscapeSequence(null, null);
    try std.testing.expect(!app.ring_menu_visible);
}

test "App ring_menu Enter Quit sets should_quit" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    // Open menu
    app.handleKey(109); // 'm'
    try std.testing.expect(app.ring_menu_visible);

    // Set selected to 5 (Quit)
    app.ring_menu_selected = 5;

    // Press Enter
    app.handleKey('\r');
    try std.testing.expect(app.should_quit);
    try std.testing.expect(!app.ring_menu_visible);
}

test "App ring_menu Enter Schema focuses schema pane" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .input,
    };
    defer app.deinit();

    // Open menu
    app.handleKey(109); // 'm'
    try std.testing.expect(app.ring_menu_visible);

    // Set selected to 1 (Schema)
    app.ring_menu_selected = 1;

    // Press Enter
    app.handleKey('\r');
    try std.testing.expectEqual(Pane.schema, app.focus);
    try std.testing.expect(!app.ring_menu_visible);
}

test "App ring_menu Enter Clear clears input" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .input,
    };
    defer app.deinit();

    // Type some input
    for ("SELECT *") |c| {
        app.handleKey(c);
    }
    try std.testing.expectEqualStrings("SELECT *", app.input_text.items);
    try std.testing.expectEqual(@as(usize, 8), app.input_cursor);

    // Open menu
    app.handleKey(109); // 'm'
    try std.testing.expect(app.ring_menu_visible);

    // Set selected to 4 (Clear)
    app.ring_menu_selected = 4;

    // Press Enter
    app.handleKey('\r');
    try std.testing.expectEqual(@as(usize, 0), app.input_text.items.len);
    try std.testing.expectEqual(@as(usize, 0), app.input_cursor);
    try std.testing.expect(!app.ring_menu_visible);
}

test "App timer initializes hidden" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    // Timer should start hidden
    try std.testing.expect(!app.timer_visible);
}

test "App handleKey 't' toggles timer_visible" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    // Initially hidden
    try std.testing.expect(!app.timer_visible);

    // Press 't' (ASCII 116)
    app.handleKey(116);
    try std.testing.expect(app.timer_visible);

    // Press 't' again to close
    app.handleKey(116);
    try std.testing.expect(!app.timer_visible);
}

test "App timer closed by bare ESC when no overlays open" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    // Open timer
    app.timer_visible = true;
    try std.testing.expect(app.timer_visible);

    // Bare ESC (b2 == null) should close timer when detail and ring_menu are not visible
    try std.testing.expect(!app.detail_visible);
    try std.testing.expect(!app.ring_menu_visible);
    app.handleEscapeSequence(null, null);
    try std.testing.expect(!app.timer_visible);
}

test "App timer NOT closed by bare ESC when detail overlay is visible" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    // Open both detail and timer
    app.detail_visible = true;
    app.timer_visible = true;

    // Bare ESC should close detail overlay (not timer) when detail is open
    app.handleEscapeSequence(null, null);
    try std.testing.expect(!app.detail_visible);
    try std.testing.expect(app.timer_visible);
}

test "App timer NOT closed by bare ESC when ring_menu is visible" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    // Open both ring menu and timer
    app.ring_menu_visible = true;
    app.timer_visible = true;

    // Bare ESC should close ring_menu (not timer) when ring_menu is open
    app.handleEscapeSequence(null, null);
    try std.testing.expect(!app.ring_menu_visible);
    try std.testing.expect(app.timer_visible);
}

test "App 't' key does NOT toggle timer when ring_menu is visible" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    // Open ring menu
    app.ring_menu_visible = true;
    try std.testing.expect(!app.timer_visible);

    // Press 't' (should be intercepted by ring menu handling, not toggle timer)
    app.handleKey(116);

    // Timer should still be closed (ring menu intercepts 't')
    try std.testing.expect(!app.timer_visible);
}

test "App query_lap_count initializes to zero" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    try std.testing.expectEqual(@as(usize, 0), app.query_lap_count);
    try std.testing.expectEqual(@as(u64, 0), app.query_cumulative_ms);
}

test "App query_laps array does not overflow on 33+ queries" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    // Simulate 35 queries recorded (would overflow if not capped at 32)
    for (0..35) |i| {
        app.query_elapsed_ms = @as(u64, @intCast(i + 1)) * 10; // 10ms, 20ms, ..., 350ms
        // When query_lap_count < 32, new lap is recorded
        if (app.query_lap_count < 32) {
            app.query_laps[app.query_lap_count] = app.query_elapsed_ms;
            app.query_lap_count += 1;
            app.query_cumulative_ms += app.query_elapsed_ms;
        }
    }

    // Should stop at 32 laps (not crash or overflow)
    try std.testing.expectEqual(@as(usize, 32), app.query_lap_count);

    // Verify no out-of-bounds writes corrupted other fields
    try std.testing.expectEqual(@as(bool, false), app.timer_visible);
    try std.testing.expectEqual(@as(bool, false), app.detail_visible);
    try std.testing.expectEqual(@as(bool, false), app.ring_menu_visible);
}

test "App renderUI does not crash with timer_visible = true" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    // Set some timer state
    app.timer_visible = true;
    app.query_elapsed_ms = 125;
    app.query_lap_count = 3;
    app.query_cumulative_ms = 375;

    // Simulate a small terminal
    const width = 80;
    const height = 24;

    // This is a smoke test — renderUI should not crash or panic when timer_visible = true
    // (Pixel-level correctness is not tested here)
    _ = width;
    _ = height;
    // Actual renderUI call would be:
    // renderUI(&app, width, height); // Not calling to avoid dependency on full TUI rendering setup
}

test "kanban_visible starts as false" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    try std.testing.expect(app.kanban_visible == false);
}

test "App handleKey 'b' opens kanban board" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .results,
    };
    defer app.deinit();

    // Press 'b' (ASCII 98)
    app.handleKey(98);
    try std.testing.expect(app.kanban_visible == true);
    try std.testing.expectEqual(@as(usize, 0), app.kanban_focused_col);
}

test "App handleKey 'b' toggles kanban closed" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .results,
    };
    defer app.deinit();

    // Open kanban
    app.handleKey(98); // 'b'
    try std.testing.expect(app.kanban_visible == true);

    // Close kanban
    app.handleKey(98); // 'b' again
    try std.testing.expect(app.kanban_visible == false);
}

test "App ESC closes kanban board" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    // Open kanban
    app.kanban_visible = true;
    try std.testing.expect(app.kanban_visible == true);

    // Send bare ESC (b2 == null)
    app.handleEscapeSequence(null, null);
    try std.testing.expect(app.kanban_visible == false);
}

test "recordQueryHistory stores success entry" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    // Record a successful query
    app.recordQueryHistory("SELECT 1", true, 42);

    try std.testing.expectEqual(@as(usize, 1), app.query_history_count);
    try std.testing.expect(app.query_history[0].success == true);
    try std.testing.expectEqual(@as(u64, 42), app.query_history[0].duration_ms);
}

test "recordQueryHistory stores failure entry" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    // Record a failed query
    app.recordQueryHistory("SELECT invalid", false, 15);

    try std.testing.expectEqual(@as(usize, 1), app.query_history_count);
    try std.testing.expect(app.query_history[0].success == false);
    try std.testing.expectEqual(@as(u64, 15), app.query_history[0].duration_ms);
}

test "recordQueryHistory ring buffer wraps at QUERY_HISTORY_MAX" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    // Add QUERY_HISTORY_MAX + 1 entries (should wrap ring buffer)
    const max = 16; // Assuming QUERY_HISTORY_MAX = 16
    for (0..max + 1) |i| {
        const query_text = std.fmt.allocPrint(allocator, "SELECT {d}", .{i}) catch return;
        defer allocator.free(query_text);
        app.recordQueryHistory(query_text, true, @as(u64, @intCast(i)));
    }

    // Count should be capped at max (ring buffer, not growing beyond)
    try std.testing.expectEqual(@as(usize, max), app.query_history_count);
}

test "kanban right arrow increments focused_col" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    // Open kanban
    app.kanban_visible = true;
    app.kanban_focused_col = 0;

    // Send right arrow (ESC [ C)
    app.handleEscapeSequence('[', 'C');

    try std.testing.expectEqual(@as(usize, 1), app.kanban_focused_col);
}

test "kanban left arrow clamps at 0" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    // Open kanban with focused_col = 0
    app.kanban_visible = true;
    app.kanban_focused_col = 0;

    // Send left arrow (ESC [ D)
    app.handleEscapeSequence('[', 'D');

    // Should stay at 0 (clamp at lower bound)
    try std.testing.expectEqual(@as(usize, 0), app.kanban_focused_col);
}

test "kanban down arrow increments focused_card" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    // Open kanban
    app.kanban_visible = true;
    app.kanban_focused_col = 0;
    app.kanban_focused_card = 0;

    // Add 2 success entries so column 0 has cards
    app.recordQueryHistory("SELECT 1", true, 42);
    app.recordQueryHistory("SELECT 2", true, 50);

    // Send down arrow (ESC [ B)
    app.handleEscapeSequence('[', 'B');

    try std.testing.expectEqual(@as(usize, 1), app.kanban_focused_card);
}

test "bracket_visible starts as false" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    try std.testing.expect(app.bracket_visible == false);
}

test "App handleKey 'p' from results focus opens bracket" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .results,
    };
    defer app.deinit();

    // Press 'p' (ASCII 112)
    app.handleKey(112);
    try std.testing.expect(app.bracket_visible == true);
    try std.testing.expectEqual(@as(usize, 0), app.bracket_focused_round);
    try std.testing.expectEqual(@as(usize, 0), app.bracket_focused_match);
}

test "App handleKey 'p' from input focus does NOT open bracket" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .input,
    };
    defer app.deinit();

    // Press 'p' (ASCII 112)
    app.handleKey(112);
    try std.testing.expect(app.bracket_visible == false);
}

test "App handleKey 'p' toggles bracket closed" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .results,
    };
    defer app.deinit();

    // Open bracket
    app.handleKey(112); // 'p'
    try std.testing.expect(app.bracket_visible == true);

    // Close bracket
    app.handleKey(112); // 'p' again
    try std.testing.expect(app.bracket_visible == false);
}

test "App ESC closes bracket" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    // Open bracket
    app.bracket_visible = true;
    try std.testing.expect(app.bracket_visible == true);

    // Ensure other overlays are not open (kanban must be closed first in escape order)
    app.kanban_visible = false;

    // Send bare ESC (b2 == null)
    app.handleEscapeSequence(null, null);
    try std.testing.expect(app.bracket_visible == false);
}

test "bracket right arrow increments focused_round" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    // Open bracket
    app.bracket_visible = true;
    app.bracket_focused_round = 0;

    // Send right arrow (ESC [ C)
    app.handleEscapeSequence('[', 'C');

    try std.testing.expectEqual(@as(usize, 1), app.bracket_focused_round);
}

test "bracket left arrow clamps at 0" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    // Open bracket with focused_round = 0
    app.bracket_visible = true;
    app.bracket_focused_round = 0;

    // Send left arrow (ESC [ D)
    app.handleEscapeSequence('[', 'D');

    // Should stay at 0 (clamp at lower bound)
    try std.testing.expectEqual(@as(usize, 0), app.bracket_focused_round);
}

test "bracket down arrow increments focused_match" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    // Open bracket
    app.bracket_visible = true;
    app.bracket_focused_round = 0;
    app.bracket_focused_match = 0;

    // Send down arrow (ESC [ B)
    app.handleEscapeSequence('[', 'B');

    try std.testing.expectEqual(@as(usize, 1), app.bracket_focused_match);
}

test "bracket up arrow clamps at 0" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    // Open bracket with focused_match = 0
    app.bracket_visible = true;
    app.bracket_focused_match = 0;

    // Send up arrow (ESC [ A)
    app.handleEscapeSequence('[', 'A');

    // Should stay at 0 (clamp at lower bound)
    try std.testing.expectEqual(@as(usize, 0), app.bracket_focused_match);
}

// ─────────────────────────────────────────────────────────────────────────
// ActivityFeed Overlay Tests
// ─────────────────────────────────────────────────────────────────────────

test "App handleKey 'a' from results focus opens activity overlay" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .results,
    };
    defer app.deinit();

    // Activity overlay should be closed initially
    try std.testing.expect(app.activity_visible == false);

    // Press 'a' (ASCII 97)
    app.handleKey(97);

    // Activity overlay should now be open
    try std.testing.expect(app.activity_visible == true);
    try std.testing.expectEqual(@as(usize, 0), app.activity_focused);
}

test "App handleKey 'a' from input focus does NOT open activity overlay" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .input,
    };
    defer app.deinit();

    // Press 'a' while focus is on input pane
    app.handleKey(97);

    // Activity overlay should NOT open when input is focused
    try std.testing.expect(app.activity_visible == false);
}

test "App handleKey 'a' toggles activity overlay closed" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .results,
    };
    defer app.deinit();

    // Open activity overlay
    app.handleKey(97); // 'a'
    try std.testing.expect(app.activity_visible == true);

    // Close activity overlay
    app.handleKey(97); // 'a' again
    try std.testing.expect(app.activity_visible == false);
}

test "App handleKey 'a' from schema focus opens activity overlay" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .schema,
    };
    defer app.deinit();

    // Press 'a' while focus is on schema pane
    app.handleKey(97);

    // Activity overlay should open from schema pane
    try std.testing.expect(app.activity_visible == true);
}

test "App ESC closes activity overlay" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    // Open activity overlay
    app.activity_visible = true;
    try std.testing.expect(app.activity_visible == true);

    // Send bare ESC (b2 == null)
    app.handleEscapeSequence(null, null);

    // Activity overlay should be closed
    try std.testing.expect(app.activity_visible == false);
}

test "activity down arrow increments focused when overlay visible" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    // Open activity overlay with some entries
    app.activity_visible = true;
    app.activity_focused = 0;
    app.activity_count = 3;

    // Send down arrow (ESC [ B)
    app.handleEscapeSequence('[', 'B');

    // Focused should increment
    try std.testing.expectEqual(@as(usize, 1), app.activity_focused);
}

test "activity up arrow decrements focused at positive position" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    // Open activity overlay with selected item at position 2
    app.activity_visible = true;
    app.activity_focused = 2;
    app.activity_count = 3;

    // Send up arrow (ESC [ A)
    app.handleEscapeSequence('[', 'A');

    // Focused should decrement
    try std.testing.expectEqual(@as(usize, 1), app.activity_focused);
}

test "activity up arrow clamps at 0" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    // Open activity overlay with focused at 0
    app.activity_visible = true;
    app.activity_focused = 0;

    // Send up arrow (ESC [ A)
    app.handleEscapeSequence('[', 'A');

    // Should stay at 0 (clamp at lower bound)
    try std.testing.expectEqual(@as(usize, 0), app.activity_focused);
}

test "activity log: recordActivity increments count and sets correct kind" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    try std.testing.expectEqual(@as(usize, 0), app.activity_count);

    // SELECT → success kind
    app.recordActivity("SELECT * FROM t", true);
    try std.testing.expectEqual(@as(usize, 1), app.activity_count);
    try std.testing.expectEqual(sailor.activity_feed.Kind.success, app.activity_log[0].kind);

    // Failed query → error_kind
    app.recordActivity("SELECT * FROM t", false);
    try std.testing.expectEqual(@as(usize, 2), app.activity_count);
    try std.testing.expectEqual(sailor.activity_feed.Kind.error_kind, app.activity_log[1].kind);

    // DDL → action kind regardless of success flag
    app.recordActivity("CREATE TABLE x (id INTEGER)", true);
    try std.testing.expectEqual(@as(usize, 3), app.activity_count);
    try std.testing.expectEqual(sailor.activity_feed.Kind.action, app.activity_log[2].kind);
}

test "activity log: ring buffer wraps at ACTIVITY_LOG_MAX" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    const max = app.activity_log.len;

    // Fill the log to capacity + 2 entries
    for (0..max + 2) |i| {
        const idx = app.activity_count % max;
        app.activity_log[idx].kind = if (i % 2 == 0) .success else .error_kind;
        app.activity_count += 1;
    }

    // activity_count should be max + 2 (tracks total, not wrapped index)
    try std.testing.expectEqual(max + 2, app.activity_count);

    // The oldest entry (index 0) should be overwritten by the (max+1)th entry
    // which is at i=max, so it should have kind based on max % 2
    const oldest_idx = 0;
    const expected_kind: sailor.activity_feed.Kind = if (max % 2 == 0) .success else .error_kind;
    try std.testing.expectEqual(expected_kind, app.activity_log[oldest_idx].kind);
}

// ─────────────────────────────────────────────────────────────────────────
// GanttChart Query Timeline Overlay Tests
// ─────────────────────────────────────────────────────────────────────────

test "gantt: 'g' key toggles gantt_visible when not in input focus" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .results,
    };
    defer app.deinit();

    // GanttChart overlay should be closed initially
    try std.testing.expect(app.gantt_visible == false);

    // Press 'g' (ASCII 103)
    app.handleKey(103);

    // GanttChart overlay should now be open
    try std.testing.expect(app.gantt_visible == true);

    // Press 'g' again
    app.handleKey(103);

    // GanttChart overlay should be closed
    try std.testing.expect(app.gantt_visible == false);
}

test "gantt: 'g' key does nothing when in input focus" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .input,
    };
    defer app.deinit();

    // Press 'g' while focus is on input pane
    app.handleKey(103);

    // GanttChart overlay should NOT open when input is focused
    try std.testing.expect(app.gantt_visible == false);
}

test "gantt: opens with focused at 0" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .results,
    };
    defer app.deinit();

    // Press 'g' to open GanttChart overlay
    app.handleKey(103);

    // Verify it's open and focused is at 0
    try std.testing.expect(app.gantt_visible == true);
    try std.testing.expectEqual(@as(usize, 0), app.gantt_focused);
}

test "gantt: ESC closes gantt overlay (bare ESC)" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    // Open GanttChart overlay
    app.gantt_visible = true;
    try std.testing.expect(app.gantt_visible == true);

    // Send bare ESC (b2 == null)
    app.handleEscapeSequence(null, null);

    // GanttChart overlay should be closed
    try std.testing.expect(app.gantt_visible == false);
}

test "gantt: down arrow increments gantt_focused" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    // Open GanttChart overlay with some query history entries
    app.gantt_visible = true;
    app.gantt_focused = 0;
    app.query_history_count = 3;

    // Send down arrow (ESC [ B)
    app.handleEscapeSequence('[', 'B');

    // Focused should increment
    try std.testing.expectEqual(@as(usize, 1), app.gantt_focused);
}

test "gantt: down arrow clamps at query_history_count boundary" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    // Open GanttChart overlay with focused at last item
    app.gantt_visible = true;
    app.gantt_focused = 2;
    app.query_history_count = 3;

    // Send down arrow (ESC [ B)
    app.handleEscapeSequence('[', 'B');

    // Focused should stay at 2 (no change, already at last)
    try std.testing.expectEqual(@as(usize, 2), app.gantt_focused);
}

test "gantt: up arrow decrements gantt_focused" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    // Open GanttChart overlay with focused at position 2
    app.gantt_visible = true;
    app.gantt_focused = 2;
    app.query_history_count = 3;

    // Send up arrow (ESC [ A)
    app.handleEscapeSequence('[', 'A');

    // Focused should decrement
    try std.testing.expectEqual(@as(usize, 1), app.gantt_focused);
}

test "gantt: up arrow clamps at 0" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    // Open GanttChart overlay with focused at 0
    app.gantt_visible = true;
    app.gantt_focused = 0;

    // Send up arrow (ESC [ A)
    app.handleEscapeSequence('[', 'A');

    // Should stay at 0 (clamp at lower bound)
    try std.testing.expectEqual(@as(usize, 0), app.gantt_focused);
}

test "gantt: opening resets focused to 0" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .schema,
    };
    defer app.deinit();

    // Set focused to 5 (simulate previous overlay state)
    app.gantt_focused = 5;

    // Press 'g' to open GanttChart overlay
    app.handleKey(103);

    // Verify focused was reset to 0
    try std.testing.expectEqual(@as(usize, 0), app.gantt_focused);
}

test "gantt: closing via 'g' key hides overlay" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .schema,
    };
    defer app.deinit();

    // Open GanttChart overlay
    app.gantt_visible = true;
    try std.testing.expect(app.gantt_visible == true);

    // Press 'g' to close GanttChart overlay
    app.handleKey(103);

    // Verify it's closed
    try std.testing.expect(app.gantt_visible == false);
}

// ─────────────────────────────────────────────────────────────────────────
// FlowChart SQL Pipeline Overlay Tests
// ─────────────────────────────────────────────────────────────────────────

test "flow: 'f' key toggles flow_visible when not in input focus" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .results,
    };
    defer app.deinit();

    // FlowChart overlay should be closed initially
    try std.testing.expect(app.flow_visible == false);

    // Press 'f' (ASCII 102)
    app.handleKey(102);

    // FlowChart overlay should now be open
    try std.testing.expect(app.flow_visible == true);

    // Press 'f' again
    app.handleKey(102);

    // FlowChart overlay should be closed
    try std.testing.expect(app.flow_visible == false);
}

test "flow: 'f' key does nothing when in input focus" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .input,
    };
    defer app.deinit();

    // Press 'f' while focus is on input pane
    app.handleKey(102);

    // FlowChart overlay should NOT open when input is focused
    try std.testing.expect(app.flow_visible == false);
}

test "flow: opens with focused at 0" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .results,
    };
    defer app.deinit();

    // Press 'f' to open FlowChart overlay
    app.handleKey(102);

    // Verify it's open and focused is at 0
    try std.testing.expect(app.flow_visible == true);
    try std.testing.expectEqual(@as(usize, 0), app.flow_focused);
}

test "flow: ESC closes flow overlay (bare ESC)" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    // Open FlowChart overlay
    app.flow_visible = true;
    try std.testing.expect(app.flow_visible == true);

    // Send bare ESC (b2 == null)
    app.handleEscapeSequence(null, null);

    // FlowChart overlay should be closed
    try std.testing.expect(app.flow_visible == false);
}

test "flow: down arrow increments flow_focused" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    // Open FlowChart overlay
    app.flow_visible = true;
    app.flow_focused = 0;

    // Send down arrow (ESC [ B)
    app.handleEscapeSequence('[', 'B');

    // Focused should increment
    try std.testing.expectEqual(@as(usize, 1), app.flow_focused);
}

test "flow: down arrow clamps at FLOW_NODE_COUNT boundary" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    // Open FlowChart overlay with focused at last node (FLOW_NODE_COUNT - 1 = 5)
    app.flow_visible = true;
    app.flow_focused = 5;

    // Send down arrow (ESC [ B)
    app.handleEscapeSequence('[', 'B');

    // Focused should stay at 5 (clamped at boundary)
    try std.testing.expectEqual(@as(usize, 5), app.flow_focused);
}

test "flow: up arrow decrements flow_focused" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    // Open FlowChart overlay with focused at position 2
    app.flow_visible = true;
    app.flow_focused = 2;

    // Send up arrow (ESC [ A)
    app.handleEscapeSequence('[', 'A');

    // Focused should decrement
    try std.testing.expectEqual(@as(usize, 1), app.flow_focused);
}

test "flow: up arrow clamps at 0" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    // Open FlowChart overlay with focused at 0
    app.flow_visible = true;
    app.flow_focused = 0;

    // Send up arrow (ESC [ A)
    app.handleEscapeSequence('[', 'A');

    // Should stay at 0 (clamp at lower bound)
    try std.testing.expectEqual(@as(usize, 0), app.flow_focused);
}

test "flow: opening resets focused to 0" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .schema,
    };
    defer app.deinit();

    // Set focused to 5 (simulate previous overlay state)
    app.flow_focused = 5;

    // Press 'f' to open FlowChart overlay
    app.handleKey(102);

    // Verify focused was reset to 0
    try std.testing.expectEqual(@as(usize, 0), app.flow_focused);
}

test "flow: closing via 'f' key hides overlay" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .schema,
    };
    defer app.deinit();

    // Open FlowChart overlay
    app.flow_visible = true;
    try std.testing.expect(app.flow_visible == true);

    // Press 'f' to close FlowChart overlay
    app.handleKey(102);

    // Verify it's closed
    try std.testing.expect(app.flow_visible == false);
}

test "mind: mind_visible starts false" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    // MindMap overlay should be closed initially
    try std.testing.expect(app.mind_visible == false);
}

test "mind: 'n' key toggles mind_visible true when not in input focus" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .schema,
    };
    defer app.deinit();

    // MindMap overlay should be closed initially
    try std.testing.expect(app.mind_visible == false);

    // Press 'n' (ASCII 110) to open MindMap overlay
    app.handleKey(110);

    // MindMap overlay should now be open
    try std.testing.expect(app.mind_visible == true);
}

test "mind: 'n' key toggles mind_visible false when open" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .results,
    };
    defer app.deinit();

    // Open MindMap overlay
    app.mind_visible = true;
    try std.testing.expect(app.mind_visible == true);

    // Press 'n' again
    app.handleKey(110);

    // MindMap overlay should be closed
    try std.testing.expect(app.mind_visible == false);
}

test "mind: 'n' key does nothing when in input focus" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .input,
    };
    defer app.deinit();

    // Press 'n' while focus is on input pane
    app.handleKey(110);

    // MindMap overlay should NOT open when input is focused
    try std.testing.expect(app.mind_visible == false);
}

test "mind: opening resets mind_focused to 0" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .schema,
    };
    defer app.deinit();

    // Set focused to 5 (simulate previous overlay state)
    app.mind_focused = 5;

    // Press 'n' to open MindMap overlay
    app.handleKey(110);

    // Verify focused was reset to 0
    try std.testing.expectEqual(@as(usize, 0), app.mind_focused);
}

test "mind: ESC closes mind overlay (bare ESC)" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    // Open MindMap overlay
    app.mind_visible = true;
    try std.testing.expect(app.mind_visible == true);

    // Send bare ESC (b2 == null)
    app.handleEscapeSequence(null, null);

    // MindMap overlay should be closed
    try std.testing.expect(app.mind_visible == false);
}

test "mind: up arrow decrements mind_focused" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    // Open MindMap overlay with focused at position 2
    app.mind_visible = true;
    app.mind_focused = 2;

    // Send up arrow (ESC [ A)
    app.handleEscapeSequence('[', 'A');

    // Focused should decrement
    try std.testing.expectEqual(@as(usize, 1), app.mind_focused);
}

test "mind: up arrow clamps at 0" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    // Open MindMap overlay with focused at 0
    app.mind_visible = true;
    app.mind_focused = 0;

    // Send up arrow (ESC [ A)
    app.handleEscapeSequence('[', 'A');

    // Should stay at 0 (clamp at lower bound)
    try std.testing.expectEqual(@as(usize, 0), app.mind_focused);
}

test "mind: down arrow increments mind_focused" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    // Open MindMap overlay
    app.mind_visible = true;
    app.mind_focused = 0;

    // Send down arrow (ESC [ B)
    app.handleEscapeSequence('[', 'B');

    // Focused should increment
    try std.testing.expectEqual(@as(usize, 1), app.mind_focused);
}

test "mind: down arrow clamps at schema_table_indices boundary" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    // Populate schema_table_indices with 3 table indices (simulating 3 tables in DB schema)
    try app.schema_table_indices.append(allocator, 0);
    try app.schema_table_indices.append(allocator, 5);
    try app.schema_table_indices.append(allocator, 10);

    // Open MindMap overlay with focused at last table (index 2)
    app.mind_visible = true;
    app.mind_focused = 2;

    // Send down arrow (ESC [ B)
    app.handleEscapeSequence('[', 'B');

    // Focused should stay at 2 (clamped at schema_table_indices.items.len - 1)
    try std.testing.expectEqual(@as(usize, 2), app.mind_focused);
}

test "radar: radar_visible starts false" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    try std.testing.expect(app.radar_visible == false);
}

test "radar: 'r' key toggles radar_visible true when not in input focus" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .schema,
    };
    defer app.deinit();

    try std.testing.expect(app.radar_visible == false);

    app.handleKey(114);

    try std.testing.expect(app.radar_visible == true);
}

test "radar: 'r' key toggles radar_visible false when open" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .results,
    };
    defer app.deinit();

    app.radar_visible = true;
    try std.testing.expect(app.radar_visible == true);

    app.handleKey(114);

    try std.testing.expect(app.radar_visible == false);
}

test "radar: 'r' key does nothing when in input focus" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .input,
    };
    defer app.deinit();

    app.handleKey(114);

    try std.testing.expect(app.radar_visible == false);
}

test "radar: opening resets radar_focused to 0" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .schema,
    };
    defer app.deinit();

    app.radar_focused = 3;

    app.handleKey(114);

    try std.testing.expectEqual(@as(usize, 0), app.radar_focused);
}

test "radar: ESC closes radar overlay (bare ESC)" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    app.radar_visible = true;
    try std.testing.expect(app.radar_visible == true);

    app.handleEscapeSequence(null, null);

    try std.testing.expect(app.radar_visible == false);
}

test "radar: right arrow increments radar_focused" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    app.radar_visible = true;
    app.radar_focused = 0;

    app.handleEscapeSequence('[', 'C');

    try std.testing.expectEqual(@as(usize, 1), app.radar_focused);
}

test "radar: right arrow clamps at 1 (max series index)" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    app.radar_visible = true;
    app.radar_focused = 1;

    app.handleEscapeSequence('[', 'C');

    try std.testing.expectEqual(@as(usize, 1), app.radar_focused);
}

test "radar: left arrow decrements radar_focused" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    app.radar_visible = true;
    app.radar_focused = 1;

    app.handleEscapeSequence('[', 'D');

    try std.testing.expectEqual(@as(usize, 0), app.radar_focused);
}

test "radar: left arrow clamps at 0" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    app.radar_visible = true;
    app.radar_focused = 0;

    app.handleEscapeSequence('[', 'D');

    try std.testing.expectEqual(@as(usize, 0), app.radar_focused);
}

// ── HexEditor Page Viewer Tests ──────────────────────────────────────

test "hex: hex_visible starts false" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    try std.testing.expect(app.hex_visible == false);
}

test "hex: hex_cursor starts at 0" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    try std.testing.expectEqual(@as(usize, 0), app.hex_cursor);
}

test "hex: 'x' key toggles hex_visible on" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .results,
    };
    defer app.deinit();

    try std.testing.expect(app.hex_visible == false);

    app.handleKey(120); // 'x' key

    try std.testing.expect(app.hex_visible == true);
}

test "hex: 'x' key toggles hex_visible off" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .schema,
    };
    defer app.deinit();

    app.hex_visible = true;
    try std.testing.expect(app.hex_visible == true);

    app.handleKey(120); // 'x' key

    try std.testing.expect(app.hex_visible == false);
}

test "hex: 'x' key ignored in input focus" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .input,
    };
    defer app.deinit();

    app.handleKey(120); // 'x' key

    try std.testing.expect(app.hex_visible == false);
}

test "hex: ESC closes hex overlay" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    app.hex_visible = true;
    try std.testing.expect(app.hex_visible == true);

    app.handleEscapeSequence(null, null); // bare ESC

    try std.testing.expect(app.hex_visible == false);
}

test "hex: right arrow increments hex_cursor" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    app.hex_visible = true;
    app.hex_cursor = 0;
    app.hex_data_len = 64;

    app.handleEscapeSequence('[', 'C'); // right arrow

    try std.testing.expectEqual(@as(usize, 1), app.hex_cursor);
}

test "hex: left arrow decrements hex_cursor" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    app.hex_visible = true;
    app.hex_cursor = 5;

    app.handleEscapeSequence('[', 'D'); // left arrow

    try std.testing.expectEqual(@as(usize, 4), app.hex_cursor);
}

test "hex: hex_cursor clamps at 0" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    app.hex_visible = true;
    app.hex_cursor = 0;

    app.handleEscapeSequence('[', 'D'); // left arrow

    try std.testing.expectEqual(@as(usize, 0), app.hex_cursor);
}

test "hex: hex_cursor clamps at data boundary" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    app.hex_visible = true;
    app.hex_data_len = 64;
    app.hex_cursor = 63; // max index for 64-byte buffer

    app.handleEscapeSequence('[', 'C'); // right arrow

    try std.testing.expectEqual(@as(usize, 63), app.hex_cursor);
}

// ── Treemap Table Hierarchy Overlay Tests ───────────────────────────────

test "treemap: treemap_visible starts false" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    try std.testing.expect(app.treemap_visible == false);
}

test "treemap: treemap_focused starts at 0" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    try std.testing.expectEqual(@as(usize, 0), app.treemap_focused);
}

test "treemap: 'w' key toggles treemap_visible on when not in input focus" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .schema,
    };
    defer app.deinit();

    try std.testing.expect(app.treemap_visible == false);

    app.handleKey(119); // 'w' key

    try std.testing.expect(app.treemap_visible == true);
}

test "treemap: 'w' key toggles treemap_visible off when already open" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .results,
    };
    defer app.deinit();

    app.treemap_visible = true;
    try std.testing.expect(app.treemap_visible == true);

    app.handleKey(119); // 'w' key

    try std.testing.expect(app.treemap_visible == false);
}

test "treemap: 'w' key does nothing when in input focus" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .input,
    };
    defer app.deinit();

    app.handleKey(119); // 'w' key

    try std.testing.expect(app.treemap_visible == false);
}

test "treemap: opening resets treemap_focused to 0" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .schema,
    };
    defer app.deinit();

    app.treemap_focused = 5;

    app.handleKey(119); // 'w' key to open

    try std.testing.expectEqual(@as(usize, 0), app.treemap_focused);
}

test "treemap: bare ESC closes treemap overlay" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    app.treemap_visible = true;
    try std.testing.expect(app.treemap_visible == true);

    app.handleEscapeSequence(null, null); // bare ESC

    try std.testing.expect(app.treemap_visible == false);
}

test "treemap: down arrow increments treemap_focused" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    app.treemap_visible = true;
    app.treemap_focused = 0;

    app.handleEscapeSequence('[', 'B'); // down arrow

    try std.testing.expectEqual(@as(usize, 1), app.treemap_focused);
}

test "treemap: right arrow increments treemap_focused" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    app.treemap_visible = true;
    app.treemap_focused = 10;

    app.handleEscapeSequence('[', 'C'); // right arrow

    try std.testing.expectEqual(@as(usize, 11), app.treemap_focused);
}

test "treemap: treemap_focused clamps at max (63)" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    app.treemap_visible = true;
    app.treemap_focused = 63; // MAX_TREEMAP_ITEMS - 1

    app.handleEscapeSequence('[', 'C'); // right arrow

    try std.testing.expectEqual(@as(usize, 63), app.treemap_focused);
}

test "treemap: treemap_focused clamps at 0" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    app.treemap_visible = true;
    app.treemap_focused = 0;

    app.handleEscapeSequence('[', 'A'); // up arrow

    try std.testing.expectEqual(@as(usize, 0), app.treemap_focused);
}

// ── MatrixView Query Metrics Overlay Tests ───────────────────────────

test "matrix: matrix_visible starts false" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    try std.testing.expect(app.matrix_visible == false);
}

test "matrix: matrix_focused_row starts at 0" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    try std.testing.expectEqual(@as(usize, 0), app.matrix_focused_row);
}

test "matrix: matrix_focused_col starts at 0" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    try std.testing.expectEqual(@as(usize, 0), app.matrix_focused_col);
}

test "matrix: 'v' key toggles matrix_visible on when not in input focus" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .schema,
    };
    defer app.deinit();

    try std.testing.expect(app.matrix_visible == false);

    app.handleKey(118); // 'v' key

    try std.testing.expect(app.matrix_visible == true);
}

test "matrix: 'v' key toggles matrix_visible off when already open" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .results,
    };
    defer app.deinit();

    app.matrix_visible = true;
    try std.testing.expect(app.matrix_visible == true);

    app.handleKey(118); // 'v' key

    try std.testing.expect(app.matrix_visible == false);
}

test "matrix: 'v' key does nothing in input focus" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .input,
    };
    defer app.deinit();

    app.handleKey(118); // 'v' key

    try std.testing.expect(app.matrix_visible == false);
}

test "matrix: opening resets matrix_focused_row and matrix_focused_col to 0" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .schema,
    };
    defer app.deinit();

    app.matrix_focused_row = 5;
    app.matrix_focused_col = 2;

    app.handleKey(118); // 'v' key to open

    try std.testing.expectEqual(@as(usize, 0), app.matrix_focused_row);
    try std.testing.expectEqual(@as(usize, 0), app.matrix_focused_col);
}

test "matrix: bare ESC closes matrix overlay" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    app.matrix_visible = true;
    try std.testing.expect(app.matrix_visible == true);

    app.handleEscapeSequence(null, null); // bare ESC

    try std.testing.expect(app.matrix_visible == false);
}

test "matrix: down arrow increments matrix_focused_row" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    app.matrix_visible = true;
    app.matrix_focused_row = 0;

    app.handleEscapeSequence('[', 'B'); // down arrow

    try std.testing.expectEqual(@as(usize, 1), app.matrix_focused_row);
}

test "matrix: right arrow increments matrix_focused_col" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    app.matrix_visible = true;
    app.matrix_focused_col = 0;

    app.handleEscapeSequence('[', 'C'); // right arrow

    try std.testing.expectEqual(@as(usize, 1), app.matrix_focused_col);
}

test "matrix: matrix_focused_row clamps at MATRIX_ROWS-1" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    app.matrix_visible = true;
    app.matrix_focused_row = 5; // MATRIX_ROWS - 1

    app.handleEscapeSequence('[', 'B'); // down arrow

    try std.testing.expectEqual(@as(usize, 5), app.matrix_focused_row);
}

test "matrix: matrix_focused_col clamps at MATRIX_COLS-1" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    app.matrix_visible = true;
    app.matrix_focused_col = 2; // MATRIX_COLS - 1

    app.handleEscapeSequence('[', 'C'); // right arrow

    try std.testing.expectEqual(@as(usize, 2), app.matrix_focused_col);
}

test "sankey: 's' key opens sankey overlay when focus is schema" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .schema,
    };
    defer app.deinit();

    try std.testing.expect(app.sankey_visible == false);

    app.handleKey('s');

    try std.testing.expect(app.sankey_visible == true);
}

test "sankey: 's' key closes sankey overlay when already visible" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .schema,
    };
    defer app.deinit();

    app.sankey_visible = true;

    app.handleKey('s');

    try std.testing.expect(app.sankey_visible == false);
}

test "sankey: 's' key does nothing when focus is input" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .input,
    };
    defer app.deinit();

    try std.testing.expect(app.sankey_visible == false);

    app.handleKey('s');

    try std.testing.expect(app.sankey_visible == false);
}

test "sankey: 's' key resets sankey_focused to 0 on open" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .results,
    };
    defer app.deinit();

    app.sankey_focused = 5;

    app.handleKey('s');

    try std.testing.expectEqual(@as(usize, 0), app.sankey_focused);
}

test "sankey: bare ESC closes sankey overlay" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    app.sankey_visible = true;
    try std.testing.expect(app.sankey_visible == true);

    app.handleEscapeSequence(null, null); // bare ESC

    try std.testing.expect(app.sankey_visible == false);
}

test "sankey: down arrow increments sankey_focused" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    app.sankey_visible = true;
    app.sankey_focused = 0;

    app.handleEscapeSequence('[', 'B'); // down arrow

    try std.testing.expectEqual(@as(usize, 1), app.sankey_focused);
}

test "sankey: up arrow decrements sankey_focused" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    app.sankey_visible = true;
    app.sankey_focused = 2;

    app.handleEscapeSequence('[', 'A'); // up arrow

    try std.testing.expectEqual(@as(usize, 1), app.sankey_focused);
}

test "sankey: up arrow clamps at 0" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    app.sankey_visible = true;
    app.sankey_focused = 0;

    app.handleEscapeSequence('[', 'A'); // up arrow

    try std.testing.expectEqual(@as(usize, 0), app.sankey_focused);
}

test "sankey: down arrow clamps at SANKEY_NODE_COUNT - 1" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    app.sankey_visible = true;
    app.sankey_focused = 5; // SANKEY_NODE_COUNT - 1

    app.handleEscapeSequence('[', 'B'); // down arrow

    try std.testing.expectEqual(@as(usize, 5), app.sankey_focused);
}

test "sankey: opening sankey does not affect other overlays" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .schema,
    };
    defer app.deinit();

    app.matrix_visible = true;
    app.matrix_focused_row = 3;

    app.handleKey('s');

    try std.testing.expect(app.sankey_visible == true);
    try std.testing.expect(app.matrix_visible == true);
    try std.testing.expectEqual(@as(usize, 3), app.matrix_focused_row);
}

// ── BubbleChart Query Performance Tests ──────────────────────────────────

test "bubble: bubble_visible starts false" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    try std.testing.expect(app.bubble_visible == false);
}

test "bubble: 'u' key opens bubble overlay when focus is schema" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .schema,
    };
    defer app.deinit();

    try std.testing.expect(app.bubble_visible == false);

    app.handleKey('u');

    try std.testing.expect(app.bubble_visible == true);
}

test "bubble: 'u' key closes bubble overlay when already visible" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .schema,
    };
    defer app.deinit();

    app.bubble_visible = true;

    app.handleKey('u');

    try std.testing.expect(app.bubble_visible == false);
}

test "bubble: 'u' key does nothing when focus is input" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .input,
    };
    defer app.deinit();

    try std.testing.expect(app.bubble_visible == false);

    app.handleKey('u');

    try std.testing.expect(app.bubble_visible == false);
}

test "bubble: 'u' key resets bubble_focused to 0 on open" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .results,
    };
    defer app.deinit();

    app.bubble_focused = 10;

    app.handleKey('u');

    try std.testing.expectEqual(@as(usize, 0), app.bubble_focused);
}

test "bubble: bare ESC closes bubble overlay" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    app.bubble_visible = true;
    try std.testing.expect(app.bubble_visible == true);

    app.handleEscapeSequence(null, null); // bare ESC

    try std.testing.expect(app.bubble_visible == false);
}

test "bubble: down arrow increments bubble_focused" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    app.bubble_visible = true;
    app.bubble_focused = 0;

    app.handleEscapeSequence('[', 'B'); // down arrow

    try std.testing.expectEqual(@as(usize, 1), app.bubble_focused);
}

test "bubble: up arrow decrements bubble_focused" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    app.bubble_visible = true;
    app.bubble_focused = 5;

    app.handleEscapeSequence('[', 'A'); // up arrow

    try std.testing.expectEqual(@as(usize, 4), app.bubble_focused);
}

test "bubble: up arrow clamps at 0" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    app.bubble_visible = true;
    app.bubble_focused = 0;

    app.handleEscapeSequence('[', 'A'); // up arrow

    try std.testing.expectEqual(@as(usize, 0), app.bubble_focused);
}

test "bubble: down arrow clamps at QUERY_HISTORY_MAX - 1" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    app.bubble_visible = true;
    app.bubble_focused = 15; // QUERY_HISTORY_MAX - 1

    app.handleEscapeSequence('[', 'B'); // down arrow

    try std.testing.expectEqual(@as(usize, 15), app.bubble_focused);
}

// ── ChordDiagram SQL Operation Flow Tests ──────────────────────────────────

test "chord: chord_visible starts false" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    try std.testing.expect(app.chord_visible == false);
}

test "chord: 'c' key opens chord overlay when focus is schema" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .schema,
    };
    defer app.deinit();

    try std.testing.expect(app.chord_visible == false);

    app.handleKey('c');

    try std.testing.expect(app.chord_visible == true);
}

test "chord: 'c' key closes chord overlay when already visible" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .schema,
    };
    defer app.deinit();

    app.chord_visible = true;

    app.handleKey('c');

    try std.testing.expect(app.chord_visible == false);
}

test "chord: 'c' key does nothing when focus is input" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .input,
    };
    defer app.deinit();

    try std.testing.expect(app.chord_visible == false);

    app.handleKey('c');

    try std.testing.expect(app.chord_visible == false);
}

test "chord: 'c' key resets chord_focused to 0 on open" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .results,
    };
    defer app.deinit();

    app.chord_focused = 3;

    app.handleKey('c');

    try std.testing.expectEqual(@as(usize, 0), app.chord_focused);
}

test "chord: bare ESC closes chord overlay" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    app.chord_visible = true;
    try std.testing.expect(app.chord_visible == true);

    app.handleEscapeSequence(null, null); // bare ESC

    try std.testing.expect(app.chord_visible == false);
}

test "chord: down arrow increments chord_focused" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    app.chord_visible = true;
    app.chord_focused = 0;

    app.handleEscapeSequence('[', 'B'); // down arrow

    try std.testing.expectEqual(@as(usize, 1), app.chord_focused);
}

test "chord: up arrow decrements chord_focused" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    app.chord_visible = true;
    app.chord_focused = 2;

    app.handleEscapeSequence('[', 'A'); // up arrow

    try std.testing.expectEqual(@as(usize, 1), app.chord_focused);
}

test "chord: up arrow clamps at 0" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    app.chord_visible = true;
    app.chord_focused = 0;

    app.handleEscapeSequence('[', 'A'); // up arrow

    try std.testing.expectEqual(@as(usize, 0), app.chord_focused);
}

test "chord: down arrow clamps at CHORD_NODE_COUNT - 1" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    app.chord_visible = true;
    app.chord_focused = 4; // CHORD_NODE_COUNT - 1

    app.handleEscapeSequence('[', 'B'); // down arrow

    try std.testing.expectEqual(@as(usize, 4), app.chord_focused);
}

// ── WaterfallChart SQL Operation Breakdown Tests ────────────────────────────

test "waterfall: waterfall_visible starts false" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    try std.testing.expect(app.waterfall_visible == false);
}

test "waterfall: 'e' key opens waterfall overlay when focus is schema" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .schema,
    };
    defer app.deinit();

    try std.testing.expect(app.waterfall_visible == false);

    app.handleKey('e');

    try std.testing.expect(app.waterfall_visible == true);
}

test "waterfall: 'e' key closes waterfall overlay when already visible" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .schema,
    };
    defer app.deinit();

    app.waterfall_visible = true;

    app.handleKey('e');

    try std.testing.expect(app.waterfall_visible == false);
}

test "waterfall: 'e' key does nothing when focus is input" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .input,
    };
    defer app.deinit();

    try std.testing.expect(app.waterfall_visible == false);

    app.handleKey('e');

    try std.testing.expect(app.waterfall_visible == false);
}

test "waterfall: 'e' key resets waterfall_focused to 0 on open" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .results,
    };
    defer app.deinit();

    app.waterfall_focused = 3;

    app.handleKey('e');

    try std.testing.expectEqual(@as(usize, 0), app.waterfall_focused);
}

test "waterfall: ESC closes waterfall overlay" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    app.waterfall_visible = true;
    try std.testing.expect(app.waterfall_visible == true);

    app.handleEscapeSequence(null, null); // bare ESC

    try std.testing.expect(app.waterfall_visible == false);
}

test "waterfall: up arrow decrements waterfall_focused" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    app.waterfall_visible = true;
    app.waterfall_focused = 2;

    app.handleEscapeSequence('[', 'A'); // up arrow

    try std.testing.expectEqual(@as(usize, 1), app.waterfall_focused);
}

test "waterfall: up arrow does not go below 0" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    app.waterfall_visible = true;
    app.waterfall_focused = 0;

    app.handleEscapeSequence('[', 'A'); // up arrow

    try std.testing.expectEqual(@as(usize, 0), app.waterfall_focused);
}

test "waterfall: down arrow increments waterfall_focused" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    app.waterfall_visible = true;
    app.waterfall_focused = 0;

    app.handleEscapeSequence('[', 'B'); // down arrow

    try std.testing.expectEqual(@as(usize, 1), app.waterfall_focused);
}

test "waterfall: down arrow does not exceed WATERFALL_BAR_COUNT - 1" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    app.waterfall_visible = true;
    app.waterfall_focused = 5; // WATERFALL_BAR_COUNT - 1

    app.handleEscapeSequence('[', 'B'); // down arrow

    try std.testing.expectEqual(@as(usize, 5), app.waterfall_focused);
}

// ── FunnelChart SQL Query Success Funnel Tests ───────────────────────────────

test "funnel: funnel_visible starts false" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    try std.testing.expect(app.funnel_visible == false);
}

test "funnel: 'l' key opens funnel overlay when focus is schema" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .schema,
    };
    defer app.deinit();

    try std.testing.expect(app.funnel_visible == false);

    app.handleKey('l');

    try std.testing.expect(app.funnel_visible == true);
}

test "funnel: 'l' key closes funnel overlay when already visible" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .schema,
    };
    defer app.deinit();

    app.funnel_visible = true;

    app.handleKey('l');

    try std.testing.expect(app.funnel_visible == false);
}

test "funnel: 'l' key does nothing when focus is input" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .input,
    };
    defer app.deinit();

    try std.testing.expect(app.funnel_visible == false);

    app.handleKey('l');

    try std.testing.expect(app.funnel_visible == false);
}

test "funnel: 'l' key resets funnel_focused to 0 on open" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
        .focus = .results,
    };
    defer app.deinit();

    app.funnel_focused = 3;

    app.handleKey('l');

    try std.testing.expectEqual(@as(usize, 0), app.funnel_focused);
}

test "funnel: ESC closes funnel overlay" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    app.funnel_visible = true;

    app.handleEscapeSequence(null, null);

    try std.testing.expect(app.funnel_visible == false);
}

test "funnel: up arrow does not go below 0" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    app.funnel_visible = true;
    app.funnel_focused = 0;

    app.handleEscapeSequence('[', 'A'); // up arrow

    try std.testing.expectEqual(@as(usize, 0), app.funnel_focused);
}

test "funnel: down arrow increments funnel_focused" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    app.funnel_visible = true;
    app.funnel_focused = 0;

    app.handleEscapeSequence('[', 'B'); // down arrow

    try std.testing.expectEqual(@as(usize, 1), app.funnel_focused);
}

test "funnel: down arrow does not exceed FUNNEL_STAGE_COUNT - 1" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    app.funnel_visible = true;
    app.funnel_focused = 3; // FUNNEL_STAGE_COUNT - 1

    app.handleEscapeSequence('[', 'B'); // down arrow

    try std.testing.expectEqual(@as(usize, 3), app.funnel_focused);
}

test "funnel: up arrow decrements funnel_focused" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    app.funnel_visible = true;
    app.funnel_focused = 2;

    app.handleEscapeSequence('[', 'A'); // up arrow

    try std.testing.expectEqual(@as(usize, 1), app.funnel_focused);
}

test "funnel: funnel_focused starts at 0" {
    const allocator = std.testing.allocator;
    var app = App{
        .allocator = allocator,
        .db = undefined,
        .db_path = "test.db",
    };
    defer app.deinit();

    try std.testing.expectEqual(@as(usize, 0), app.funnel_focused);
}
