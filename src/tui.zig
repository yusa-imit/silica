const std = @import("std");
const sailor = @import("sailor");
const silica = @import("silica");
const tui = sailor.tui;

const Database = silica.engine.Database;
const QueryResult = silica.engine.QueryResult;
const Value = silica.executor.Value;
const Row = silica.executor.Row;

// ── Focus Pane ───────────────────────────────────────────────────────

const Pane = enum { schema, results, input };

// ── Application State ────────────────────────────────────────────────

const App = struct {
    allocator: std.mem.Allocator,
    db: *Database,
    db_path: []const u8,
    should_quit: bool = false,
    focus: Pane = .input,

    // Schema tree
    schema_items: std.ArrayListUnmanaged([]const u8) = .{},
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
        self.input_text.deinit(self.allocator);
        if (self.status_left.len > 0) self.allocator.free(self.status_left);
        if (self.status_right.len > 0) self.allocator.free(self.status_right);
    }

    fn clearSchema(self: *App) void {
        for (self.schema_items.items) |item| self.allocator.free(item);
        self.schema_items.deinit(self.allocator);
        self.schema_items = .{};
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

    fn refreshSchema(self: *App) void {
        self.clearSchema();

        const tables = self.db.catalog.listTables(self.allocator) catch return;
        defer self.allocator.free(tables);

        for (tables) |table_name| {
            const name = self.allocator.dupe(u8, table_name) catch continue;
            self.schema_items.append(self.allocator, name) catch {
                self.allocator.free(name);
                continue;
            };
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
            self.schema_items.items.len,
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
            // Enter: load SELECT * FROM <table> LIMIT 100 into input
            if (self.schema_selected < self.schema_items.items.len) {
                const table = self.schema_items.items[self.schema_selected];
                self.input_text.clearRetainingCapacity();
                self.input_cursor = 0;
                const sql = std.fmt.allocPrint(self.allocator, "SELECT * FROM {s} LIMIT 100;", .{table}) catch return;
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
        if (byte == '\r' or byte == '\n') {
            // Enter: execute SQL
            self.executeSQL();
            return;
        }

        if (byte == 127 or byte == 8) {
            // Backspace
            if (self.input_cursor > 0) {
                _ = self.input_text.orderedRemove(self.input_cursor - 1);
                self.input_cursor -= 1;
            }
            return;
        }

        // Ctrl+U: clear input
        if (byte == 21) {
            self.input_text.clearRetainingCapacity();
            self.input_cursor = 0;
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
        }
    }
};

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

fn renderSchemaTree(app: *App, buf: *tui.Buffer, area: tui.Rect) void {
    const is_focused = app.focus == .schema;
    const border_style: tui.Style = if (is_focused) .{ .fg = .cyan, .bold = true } else .{};

    const block = tui.widgets.Block.init()
        .withBorderStyle(border_style)
        .withTitle("Schema", .top_left)
        .withTitleStyle(if (is_focused) tui.Style{ .fg = .cyan, .bold = true } else .{});

    if (app.schema_items.items.len == 0) {
        block.render(buf, area);
        const inner = block.inner(area);
        if (inner.width > 0 and inner.height > 0) {
            buf.setString(inner.x, inner.y, "(no tables)", .{ .fg = .bright_black });
        }
        return;
    }

    const list = tui.widgets.List.init(app.schema_items.items)
        .withSelected(if (is_focused) @as(?usize, app.schema_selected) else null)
        .withOffset(app.schema_offset)
        .withBlock(block)
        .withSelectedStyle(.{ .fg = .cyan, .bold = true })
        .withHighlightSymbol("> ");

    list.render(buf, area);
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
    if (is_focused and has_text) {
        const cursor_x = inner.x + @as(u16, @intCast(@min(app.input_cursor, inner.width - 1)));
        const cursor_char: u8 = if (app.input_cursor < app.input_text.items.len)
            app.input_text.items[app.input_cursor]
        else
            ' ';
        buf.setChar(cursor_x, inner.y, cursor_char, .{ .reverse = true });
    } else if (is_focused) {
        // Empty input — cursor on first cell
        buf.setChar(inner.x, inner.y, ' ', .{ .reverse = true });
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
