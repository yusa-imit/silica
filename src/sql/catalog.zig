//! Schema Catalog — in-memory schema table backed by B+Tree on Page 1.
//!
//! The schema catalog stores table metadata (table name, column definitions,
//! constraints) in a B+Tree rooted at SCHEMA_ROOT_PAGE_ID (page 1).
//!
//! Serialization format for table entries:
//!   [column_count: u16]
//!   for each column:
//!     [name_len: u16][name_bytes...][type_tag: u8][constraint_flags: u8]
//!   [table_constraint_count: u16]
//!   for each table constraint:
//!     [constraint_type: u8][payload...]
//!
//! Column constraint flags (bitfield):
//!   bit 0: PRIMARY KEY
//!   bit 1: NOT NULL
//!   bit 2: UNIQUE
//!   bit 3: AUTOINCREMENT
//!   bit 4: HAS DEFAULT (default value stored separately)
//!
//! Table constraint types:
//!   0x01: PRIMARY KEY (composite)
//!   0x02: UNIQUE (composite)

const std = @import("std");
const Allocator = std.mem.Allocator;
const btree_mod = @import("../storage/btree.zig");
const buffer_pool_mod = @import("../storage/buffer_pool.zig");
const page_mod = @import("../storage/page.zig");
const ast = @import("ast.zig");

const BTree = btree_mod.BTree;
const Cursor = btree_mod.Cursor;
const BufferPool = buffer_pool_mod.BufferPool;
const Pager = page_mod.Pager;
const SCHEMA_ROOT_PAGE_ID = page_mod.SCHEMA_ROOT_PAGE_ID;

// ── Data Types ──────────────────────────────────────────────────────────

/// Column data type stored in the catalog (1-byte tag).
pub const ColumnType = enum(u8) {
    integer = 0x01,
    real = 0x02,
    text = 0x03,
    blob = 0x04,
    boolean = 0x05,
    date = 0x06,
    time = 0x07,
    timestamp = 0x08,
    interval = 0x09,
    numeric = 0x0A,
    uuid = 0x0B,
    array = 0x0C,
    json = 0x0D,
    jsonb = 0x0E,
    tsvector = 0x0F,
    tsquery = 0x10,
    /// Column has no explicit type declaration.
    untyped = 0x00,
};

/// Convert from AST DataType to catalog ColumnType.
pub fn columnTypeFromAst(dt: ?ast.DataType) ColumnType {
    const t = dt orelse return .untyped;
    return switch (t) {
        .type_integer, .type_int => .integer,
        .type_real => .real,
        .type_text, .type_varchar => .text,
        .type_blob => .blob,
        .type_boolean => .boolean,
        .type_date => .date,
        .type_time => .time,
        .type_timestamp => .timestamp,
        .type_interval => .interval,
        .type_numeric, .type_decimal => .numeric,
        .type_uuid => .uuid,
        .type_serial, .type_bigserial => .integer,
        .type_array => .array,
        .type_json => .json,
        .type_jsonb => .jsonb,
        .type_tsvector => .tsvector,
        .type_tsquery => .tsquery,
    };
}

// ── Constraint Flags ────────────────────────────────────────────────────

/// Bitfield for column-level constraints.
pub const ConstraintFlags = packed struct(u8) {
    primary_key: bool = false,
    not_null: bool = false,
    unique: bool = false,
    autoincrement: bool = false,
    has_default: bool = false,
    _padding: u3 = 0,
};

/// Extract constraint flags from AST column constraints.
pub fn constraintFlagsFromAst(constraints: []const ast.ColumnConstraint) ConstraintFlags {
    var flags = ConstraintFlags{};
    for (constraints) |c| {
        switch (c) {
            .primary_key => |pk| {
                flags.primary_key = true;
                flags.autoincrement = pk.autoincrement;
                // PRIMARY KEY implies NOT NULL
                flags.not_null = true;
            },
            .not_null => flags.not_null = true,
            .unique => flags.unique = true,
            .default => flags.has_default = true,
            .check, .foreign_key => {},
        }
    }
    return flags;
}

// ── Schema Info Structures ──────────────────────────────────────────────

/// Column metadata stored in the catalog.
pub const ColumnInfo = struct {
    name: []const u8,
    column_type: ColumnType,
    flags: ConstraintFlags,
};

/// Composite table-level constraint stored in the catalog.
pub const TableConstraintInfo = union(enum) {
    primary_key: []const []const u8,
    unique: []const []const u8,
};

/// Metadata for a secondary index on a table column.
pub const IndexInfo = struct {
    /// Name of the indexed column.
    column_name: []const u8,
    /// Column index in the table's column list.
    column_index: u16,
    /// B+Tree root page ID for this index (maps column_value → row_key).
    root_page_id: u32,
};

/// Complete table metadata.
pub const TableInfo = struct {
    name: []const u8,
    columns: []const ColumnInfo,
    table_constraints: []const TableConstraintInfo,
    /// B+Tree root page ID for this table's row data (0 = not yet allocated).
    data_root_page_id: u32 = 0,
    /// Secondary indexes on this table.
    indexes: []const IndexInfo = &.{},

    /// Free all heap-allocated memory owned by this TableInfo.
    pub fn deinit(self: *const TableInfo, allocator: Allocator) void {
        for (self.indexes) |idx| allocator.free(idx.column_name);
        allocator.free(self.indexes);
        for (self.table_constraints) |tc| {
            switch (tc) {
                .primary_key => |cols| {
                    for (cols) |col_name| allocator.free(col_name);
                    allocator.free(cols);
                },
                .unique => |cols| {
                    for (cols) |col_name| allocator.free(col_name);
                    allocator.free(cols);
                },
            }
        }
        allocator.free(self.table_constraints);
        for (self.columns) |col| allocator.free(col.name);
        allocator.free(self.columns);
        allocator.free(self.name);
    }

    /// Find an index for the given column name. Returns null if no index exists.
    pub fn findIndex(self: *const TableInfo, column_name: []const u8) ?IndexInfo {
        for (self.indexes) |idx| {
            if (std.ascii.eqlIgnoreCase(idx.column_name, column_name)) return idx;
        }
        return null;
    }
};

// ── Serialization ───────────────────────────────────────────────────────

/// Serialize a table definition into bytes for B+Tree storage.
/// Format: [data_root_page_id: u32][column_count: u16][columns...]
///         [table_constraint_count: u16][constraints...]
///         [index_count: u16][indexes...]
pub fn serializeTable(allocator: Allocator, columns: []const ColumnInfo, table_constraints: []const TableConstraintInfo, data_root_page_id: u32) ![]u8 {
    return serializeTableFull(allocator, columns, table_constraints, &.{}, data_root_page_id);
}

/// Serialize a table definition with index information.
pub fn serializeTableFull(allocator: Allocator, columns: []const ColumnInfo, table_constraints: []const TableConstraintInfo, indexes: []const IndexInfo, data_root_page_id: u32) ![]u8 {
    // Calculate total size
    var size: usize = 4 + 2; // data_root_page_id: u32 + column_count: u16
    for (columns) |col| {
        size += 2 + col.name.len + 1 + 1; // name_len + name + type + flags
    }
    size += 2; // table_constraint_count: u16
    for (table_constraints) |tc| {
        size += 1; // constraint_type tag
        switch (tc) {
            .primary_key, .unique => |cols| {
                size += 2; // column count
                for (cols) |col_name| {
                    size += 2 + col_name.len; // name_len + name
                }
            },
        }
    }
    // Index info
    size += 2; // index_count: u16
    for (indexes) |idx| {
        size += 2 + idx.column_name.len + 2 + 4; // name_len + name + col_index + root_page_id
    }

    const buf = try allocator.alloc(u8, size);
    errdefer allocator.free(buf);

    var pos: usize = 0;

    // Data root page ID
    std.mem.writeInt(u32, buf[pos..][0..4], data_root_page_id, .little);
    pos += 4;

    // Column count
    std.mem.writeInt(u16, buf[pos..][0..2], @intCast(columns.len), .little);
    pos += 2;

    // Columns
    for (columns) |col| {
        std.mem.writeInt(u16, buf[pos..][0..2], @intCast(col.name.len), .little);
        pos += 2;
        @memcpy(buf[pos..][0..col.name.len], col.name);
        pos += col.name.len;
        buf[pos] = @intFromEnum(col.column_type);
        pos += 1;
        buf[pos] = @bitCast(col.flags);
        pos += 1;
    }

    // Table constraints
    std.mem.writeInt(u16, buf[pos..][0..2], @intCast(table_constraints.len), .little);
    pos += 2;

    for (table_constraints) |tc| {
        switch (tc) {
            .primary_key => |cols| {
                buf[pos] = 0x01;
                pos += 1;
                std.mem.writeInt(u16, buf[pos..][0..2], @intCast(cols.len), .little);
                pos += 2;
                for (cols) |col_name| {
                    std.mem.writeInt(u16, buf[pos..][0..2], @intCast(col_name.len), .little);
                    pos += 2;
                    @memcpy(buf[pos..][0..col_name.len], col_name);
                    pos += col_name.len;
                }
            },
            .unique => |cols| {
                buf[pos] = 0x02;
                pos += 1;
                std.mem.writeInt(u16, buf[pos..][0..2], @intCast(cols.len), .little);
                pos += 2;
                for (cols) |col_name| {
                    std.mem.writeInt(u16, buf[pos..][0..2], @intCast(col_name.len), .little);
                    pos += 2;
                    @memcpy(buf[pos..][0..col_name.len], col_name);
                    pos += col_name.len;
                }
            },
        }
    }

    // Indexes
    std.mem.writeInt(u16, buf[pos..][0..2], @intCast(indexes.len), .little);
    pos += 2;
    for (indexes) |idx| {
        std.mem.writeInt(u16, buf[pos..][0..2], @intCast(idx.column_name.len), .little);
        pos += 2;
        @memcpy(buf[pos..][0..idx.column_name.len], idx.column_name);
        pos += idx.column_name.len;
        std.mem.writeInt(u16, buf[pos..][0..2], idx.column_index, .little);
        pos += 2;
        std.mem.writeInt(u32, buf[pos..][0..4], idx.root_page_id, .little);
        pos += 4;
    }

    std.debug.assert(pos == size);
    return buf;
}

/// Deserialize a table definition from bytes stored in B+Tree.
pub fn deserializeTable(allocator: Allocator, name: []const u8, data: []const u8) !TableInfo {
    if (data.len < 6) return error.InvalidSchemaData; // 4 (page_id) + 2 (col_count)

    var pos: usize = 0;

    // Data root page ID
    const data_root_page_id = std.mem.readInt(u32, data[pos..][0..4], .little);
    pos += 4;

    // Column count
    const col_count = std.mem.readInt(u16, data[pos..][0..2], .little);
    pos += 2;

    const columns = try allocator.alloc(ColumnInfo, col_count);
    errdefer {
        for (columns[0..col_count]) |col| allocator.free(col.name);
        allocator.free(columns);
    }

    var cols_initialized: usize = 0;
    errdefer {
        for (columns[0..cols_initialized]) |col| allocator.free(col.name);
    }

    for (columns) |*col| {
        if (pos + 4 > data.len) return error.InvalidSchemaData;
        const name_len = std.mem.readInt(u16, data[pos..][0..2], .little);
        pos += 2;
        if (pos + name_len + 2 > data.len) return error.InvalidSchemaData;
        col.name = try allocator.dupe(u8, data[pos..][0..name_len]);
        pos += name_len;
        col.column_type = @enumFromInt(data[pos]);
        pos += 1;
        col.flags = @bitCast(data[pos]);
        pos += 1;
        cols_initialized += 1;
    }

    // Table constraints
    if (pos + 2 > data.len) return error.InvalidSchemaData;
    const tc_count = std.mem.readInt(u16, data[pos..][0..2], .little);
    pos += 2;

    const table_constraints = try allocator.alloc(TableConstraintInfo, tc_count);
    errdefer {
        // Note: partial cleanup handled by caller via TableInfo.deinit
        allocator.free(table_constraints);
    }

    var tcs_initialized: usize = 0;
    _ = &tcs_initialized;

    for (table_constraints) |*tc| {
        if (pos + 1 > data.len) return error.InvalidSchemaData;
        const tag = data[pos];
        pos += 1;

        switch (tag) {
            0x01, 0x02 => {
                if (pos + 2 > data.len) return error.InvalidSchemaData;
                const num_cols = std.mem.readInt(u16, data[pos..][0..2], .little);
                pos += 2;

                const col_names = try allocator.alloc([]const u8, num_cols);
                var names_init: usize = 0;
                errdefer {
                    for (col_names[0..names_init]) |cn| allocator.free(cn);
                    allocator.free(col_names);
                }

                for (col_names) |*cn| {
                    if (pos + 2 > data.len) return error.InvalidSchemaData;
                    const cn_len = std.mem.readInt(u16, data[pos..][0..2], .little);
                    pos += 2;
                    if (pos + cn_len > data.len) return error.InvalidSchemaData;
                    cn.* = try allocator.dupe(u8, data[pos..][0..cn_len]);
                    pos += cn_len;
                    names_init += 1;
                }

                if (tag == 0x01) {
                    tc.* = .{ .primary_key = col_names };
                } else {
                    tc.* = .{ .unique = col_names };
                }
                tcs_initialized += 1;
            },
            else => return error.InvalidSchemaData,
        }
    }

    // Indexes (optional — backward compatible with older format)
    var indexes: []IndexInfo = &.{};
    if (pos + 2 <= data.len) {
        const idx_count = std.mem.readInt(u16, data[pos..][0..2], .little);
        pos += 2;

        if (idx_count > 0) {
            const idx_list = try allocator.alloc(IndexInfo, idx_count);
            var idxs_initialized: usize = 0;
            errdefer {
                for (idx_list[0..idxs_initialized]) |idx| allocator.free(idx.column_name);
                allocator.free(idx_list);
            }

            for (idx_list) |*idx| {
                if (pos + 2 > data.len) return error.InvalidSchemaData;
                const cn_len = std.mem.readInt(u16, data[pos..][0..2], .little);
                pos += 2;
                if (pos + cn_len + 6 > data.len) return error.InvalidSchemaData;
                idx.column_name = try allocator.dupe(u8, data[pos..][0..cn_len]);
                pos += cn_len;
                idx.column_index = std.mem.readInt(u16, data[pos..][0..2], .little);
                pos += 2;
                idx.root_page_id = std.mem.readInt(u32, data[pos..][0..4], .little);
                pos += 4;
                idxs_initialized += 1;
            }
            indexes = idx_list;
        }
    }

    const owned_name = try allocator.dupe(u8, name);

    return .{
        .name = owned_name,
        .columns = columns,
        .table_constraints = table_constraints,
        .data_root_page_id = data_root_page_id,
        .indexes = indexes,
    };
}

// ── Schema Catalog ──────────────────────────────────────────────────────

pub const CatalogError = error{
    TableAlreadyExists,
    TableNotFound,
    InvalidSchemaData,
    OutOfMemory,
    ViewAlreadyExists,
    ViewNotFound,
    TypeAlreadyExists,
    TypeNotFound,
};

/// Schema catalog backed by a B+Tree on the schema root page (page 1).
///
/// The catalog uses table names as B+Tree keys and serialized TableInfo
/// as values. It initializes the schema root page as an empty B+Tree leaf
/// if the database is newly created.
pub const Catalog = struct {
    allocator: Allocator,
    tree: BTree,
    pool: *BufferPool,

    /// Initialize the schema catalog. If `is_new_db` is true, the schema
    /// root page will be initialized as an empty B+Tree leaf.
    pub fn init(allocator: Allocator, pool: *BufferPool, is_new_db: bool) !Catalog {
        if (is_new_db) {
            // Allocate page 1 if it doesn't exist yet, and initialize as leaf
            const pager = pool.pager;
            // Ensure we have at least 2 pages (page 0 = header, page 1 = schema root)
            while (pager.page_count < 2) {
                _ = try pager.allocPage();
            }
            const frame = try pool.fetchPage(SCHEMA_ROOT_PAGE_ID);
            btree_mod.initLeafPage(frame.data, pager.page_size, SCHEMA_ROOT_PAGE_ID);
            pool.unpinPage(SCHEMA_ROOT_PAGE_ID, true);
        }

        return .{
            .allocator = allocator,
            .tree = BTree.init(pool, SCHEMA_ROOT_PAGE_ID),
            .pool = pool,
        };
    }

    /// Create a new table in the catalog.
    /// Returns error.TableAlreadyExists if a table with the same name exists.
    /// `data_root_page_id` is the B+Tree root page for this table's row data.
    pub fn createTable(self: *Catalog, name: []const u8, columns: []const ColumnInfo, table_constraints: []const TableConstraintInfo, data_root_page_id: u32) !void {
        return self.createTableWithIndexes(name, columns, table_constraints, &.{}, data_root_page_id);
    }

    /// Create a new table in the catalog with secondary indexes.
    pub fn createTableWithIndexes(self: *Catalog, name: []const u8, columns: []const ColumnInfo, table_constraints: []const TableConstraintInfo, indexes: []const IndexInfo, data_root_page_id: u32) !void {
        // Check if table already exists
        const existing = try self.tree.get(self.allocator, name);
        if (existing) |v| {
            self.allocator.free(v);
            return CatalogError.TableAlreadyExists;
        }

        // Serialize and store
        const value = try serializeTableFull(self.allocator, columns, table_constraints, indexes, data_root_page_id);
        defer self.allocator.free(value);

        self.tree.insert(name, value) catch |err| {
            return switch (err) {
                error.DuplicateKey => CatalogError.TableAlreadyExists,
                error.OutOfMemory => CatalogError.OutOfMemory,
                else => err,
            };
        };
    }

    /// Create a table from an AST CreateTableStmt.
    pub fn createTableFromAst(self: *Catalog, stmt: *const ast.CreateTableStmt) !void {
        // Check for IF NOT EXISTS
        if (stmt.if_not_exists) {
            const existing = try self.tree.get(self.allocator, stmt.name);
            if (existing) |v| {
                self.allocator.free(v);
                return; // silently succeed
            }
        }

        // Convert AST columns to ColumnInfo
        const columns = try self.allocator.alloc(ColumnInfo, stmt.columns.len);
        defer self.allocator.free(columns);

        for (stmt.columns, 0..) |col_def, i| {
            var flags = constraintFlagsFromAst(col_def.constraints);
            // SERIAL/BIGSERIAL implies NOT NULL + AUTOINCREMENT
            if (col_def.data_type) |dt| {
                if (dt == .type_serial or dt == .type_bigserial) {
                    flags.autoincrement = true;
                    flags.not_null = true;
                }
            }
            columns[i] = .{
                .name = col_def.name,
                .column_type = columnTypeFromAst(col_def.data_type),
                .flags = flags,
            };
        }

        // Convert AST table constraints
        var tc_list = std.ArrayListUnmanaged(TableConstraintInfo){};
        defer {
            for (tc_list.items) |tc| {
                switch (tc) {
                    .primary_key => |cols| self.allocator.free(cols),
                    .unique => |cols| self.allocator.free(cols),
                }
            }
            tc_list.deinit(self.allocator);
        }

        for (stmt.table_constraints) |tc| {
            switch (tc) {
                .primary_key => |pk| {
                    const cols = try self.allocator.dupe([]const u8, pk.columns);
                    try tc_list.append(self.allocator, .{ .primary_key = cols });
                },
                .unique => |uq| {
                    const cols = try self.allocator.dupe([]const u8, uq.columns);
                    try tc_list.append(self.allocator, .{ .unique = cols });
                },
                .check, .foreign_key => {}, // Skip for now
            }
        }

        // Allocate a new B+Tree root page for the table's row data.
        // Write directly via pager (not buffer pool) since newly allocated pages
        // have zero content which would fail header deserialization.
        const pager = self.pool.pager;
        const data_root_id = try pager.allocPage();
        {
            const raw = try pager.allocPageBuf();
            defer pager.freePageBuf(raw);
            btree_mod.initLeafPage(raw, pager.page_size, data_root_id);
            try pager.writePage(data_root_id, raw);
        }

        // Create secondary index B+Trees for PRIMARY KEY columns
        var idx_list = std.ArrayListUnmanaged(IndexInfo){};
        defer idx_list.deinit(self.allocator);

        for (columns, 0..) |col, ci| {
            if (col.flags.primary_key) {
                const idx_root_id = try pager.allocPage();
                {
                    const raw = try pager.allocPageBuf();
                    defer pager.freePageBuf(raw);
                    btree_mod.initLeafPage(raw, pager.page_size, idx_root_id);
                    try pager.writePage(idx_root_id, raw);
                }
                try idx_list.append(self.allocator, .{
                    .column_name = col.name,
                    .column_index = @intCast(ci),
                    .root_page_id = idx_root_id,
                });
            }
        }

        try self.createTableWithIndexes(stmt.name, columns, tc_list.items, idx_list.items, data_root_id);
    }

    /// Drop a table from the catalog.
    /// Returns error.TableNotFound if the table doesn't exist (unless if_exists is true).
    pub fn dropTable(self: *Catalog, name: []const u8, if_exists: bool) !void {
        const existing = try self.tree.get(self.allocator, name);
        if (existing) |v| {
            self.allocator.free(v);
        } else {
            if (if_exists) return;
            return CatalogError.TableNotFound;
        }

        self.tree.delete(name) catch |err| {
            return switch (err) {
                error.OutOfMemory => CatalogError.OutOfMemory,
                else => err,
            };
        };
    }

    /// Look up a table by name. Caller must call TableInfo.deinit() on the result.
    pub fn getTable(self: *Catalog, name: []const u8) !TableInfo {
        const value = try self.tree.get(self.allocator, name) orelse
            return CatalogError.TableNotFound;
        defer self.allocator.free(value);

        return deserializeTable(self.allocator, name, value);
    }

    /// Check if a table exists.
    pub fn tableExists(self: *Catalog, name: []const u8) !bool {
        const value = try self.tree.get(self.allocator, name);
        if (value) |v| {
            self.allocator.free(v);
            return true;
        }
        return false;
    }

    /// List all table names. Caller must free each name and the returned slice.
    pub fn listTables(self: *Catalog, allocator: Allocator) ![][]const u8 {
        var cursor = Cursor.init(allocator, &self.tree);
        defer cursor.deinit();

        try cursor.seekFirst();

        var names = std.ArrayListUnmanaged([]const u8){};
        errdefer {
            for (names.items) |n| allocator.free(n);
            names.deinit(allocator);
        }

        while (try cursor.next()) |entry| {
            defer allocator.free(entry.value);
            defer allocator.free(entry.key);

            // Skip view entries (they have the VIEW_KEY_PREFIX)
            if (entry.key.len > VIEW_KEY_PREFIX.len and
                std.mem.eql(u8, entry.key[0..VIEW_KEY_PREFIX.len], VIEW_KEY_PREFIX))
            {
                continue;
            }

            const name_copy = try allocator.dupe(u8, entry.key);
            errdefer allocator.free(name_copy);
            try names.append(allocator, name_copy);
        }

        return names.toOwnedSlice(allocator);
    }

    /// Find a column by name in a table. Returns the column index and info.
    pub fn findColumn(self: *Catalog, table_name: []const u8, column_name: []const u8) !struct { index: usize, info: ColumnInfo } {
        const table = try self.getTable(table_name);
        defer table.deinit(self.allocator);

        for (table.columns, 0..) |col, i| {
            if (std.mem.eql(u8, col.name, column_name)) {
                return .{
                    .index = i,
                    .info = .{
                        .name = try self.allocator.dupe(u8, col.name),
                        .column_type = col.column_type,
                        .flags = col.flags,
                    },
                };
            }
        }
        return error.ColumnNotFound;
    }

    // ── View Catalog ────────────────────────────────────────────────────

    const VIEW_KEY_PREFIX = "\x00view\x00";

    fn makeViewKey(self: *Catalog, name: []const u8) ![]u8 {
        const key = try self.allocator.alloc(u8, VIEW_KEY_PREFIX.len + name.len);
        @memcpy(key[0..VIEW_KEY_PREFIX.len], VIEW_KEY_PREFIX);
        @memcpy(key[VIEW_KEY_PREFIX.len..], name);
        return key;
    }

    /// Store a view definition. The SQL text is stored as the value.
    pub fn createView(
        self: *Catalog,
        name: []const u8,
        sql: []const u8,
        or_replace: bool,
        if_not_exists: bool,
        column_names: []const []const u8,
        check_option: u8,
    ) !void {
        const key = try self.makeViewKey(name);
        defer self.allocator.free(key);

        // Check if view already exists
        const existing = try self.tree.get(self.allocator, key);
        if (existing) |v| {
            self.allocator.free(v);
            if (or_replace) {
                // Drop existing and re-create
                self.tree.delete(key) catch {};
            } else if (if_not_exists) {
                return;
            } else {
                return CatalogError.ViewAlreadyExists;
            }
        }

        // Also check if a table with the same name exists
        if (try self.tableExists(name)) {
            return CatalogError.TableAlreadyExists;
        }

        // Serialize: [col_count: u16][col_names...][check_option: u8][sql_text]
        var buf_size: usize = 2; // col_count
        for (column_names) |cn| {
            buf_size += 2 + cn.len; // len_prefix + name
        }
        buf_size += 1; // check_option
        buf_size += sql.len;

        const value = try self.allocator.alloc(u8, buf_size);
        defer self.allocator.free(value);

        var offset: usize = 0;
        std.mem.writeInt(u16, value[offset..][0..2], @intCast(column_names.len), .little);
        offset += 2;
        for (column_names) |cn| {
            std.mem.writeInt(u16, value[offset..][0..2], @intCast(cn.len), .little);
            offset += 2;
            @memcpy(value[offset..][0..cn.len], cn);
            offset += cn.len;
        }
        value[offset] = check_option;
        offset += 1;
        @memcpy(value[offset..], sql);

        self.tree.insert(key, value) catch |err| {
            return switch (err) {
                error.OutOfMemory => CatalogError.OutOfMemory,
                else => CatalogError.InvalidSchemaData,
            };
        };
    }

    /// Drop a view from the catalog.
    pub fn dropView(self: *Catalog, name: []const u8, if_exists: bool) !void {
        const key = try self.makeViewKey(name);
        defer self.allocator.free(key);

        const existing = try self.tree.get(self.allocator, key);
        if (existing) |v| {
            self.allocator.free(v);
        } else {
            if (if_exists) return;
            return CatalogError.ViewNotFound;
        }

        self.tree.delete(key) catch |err| {
            return switch (err) {
                error.OutOfMemory => CatalogError.OutOfMemory,
                else => err,
            };
        };
    }

    /// View definition returned from catalog lookup.
    pub const ViewInfo = struct {
        name: []const u8,
        sql: []const u8,
        column_names: []const []const u8,
        /// 0 = none, 1 = local, 2 = cascaded
        check_option: u8 = 0,
        allocator: Allocator,

        pub fn deinit(self: ViewInfo) void {
            self.allocator.free(self.name);
            self.allocator.free(self.sql);
            for (self.column_names) |cn| self.allocator.free(cn);
            self.allocator.free(self.column_names);
        }
    };

    pub const EnumTypeInfo = struct {
        name: []const u8,
        values: []const []const u8,
        allocator: Allocator,

        pub fn deinit(self: EnumTypeInfo) void {
            self.allocator.free(self.name);
            for (self.values) |v| self.allocator.free(v);
            self.allocator.free(self.values);
        }
    };

    pub const DomainTypeInfo = struct {
        name: []const u8,
        base_type: ast.DataType,
        constraint: ?[]const u8 = null,
        allocator: Allocator,

        pub fn deinit(self: DomainTypeInfo) void {
            self.allocator.free(self.name);
            if (self.constraint) |c| self.allocator.free(c);
        }
    };

    /// Look up a view by name. Caller must call ViewInfo.deinit().
    pub fn getView(self: *Catalog, name: []const u8) !ViewInfo {
        const key = try self.makeViewKey(name);
        defer self.allocator.free(key);

        const value = try self.tree.get(self.allocator, key) orelse
            return CatalogError.ViewNotFound;
        defer self.allocator.free(value);

        // Deserialize: [col_count: u16][col_names...][check_option: u8][sql_text]
        if (value.len < 2) return CatalogError.InvalidSchemaData;

        var offset: usize = 0;
        const col_count = std.mem.readInt(u16, value[offset..][0..2], .little);
        offset += 2;

        const col_names = try self.allocator.alloc([]const u8, col_count);
        errdefer {
            for (col_names[0..col_count]) |cn| self.allocator.free(cn);
            self.allocator.free(col_names);
        }

        for (0..col_count) |i| {
            if (offset + 2 > value.len) return CatalogError.InvalidSchemaData;
            const cn_len = std.mem.readInt(u16, value[offset..][0..2], .little);
            offset += 2;
            if (offset + cn_len > value.len) return CatalogError.InvalidSchemaData;
            col_names[i] = try self.allocator.dupe(u8, value[offset..][0..cn_len]);
            offset += cn_len;
        }

        // Read check_option byte (0=none, 1=local, 2=cascaded)
        if (offset >= value.len) return CatalogError.InvalidSchemaData;
        const check_opt = value[offset];
        offset += 1;

        const sql_text = try self.allocator.dupe(u8, value[offset..]);

        return .{
            .name = try self.allocator.dupe(u8, name),
            .sql = sql_text,
            .column_names = col_names,
            .check_option = check_opt,
            .allocator = self.allocator,
        };
    }

    /// Check if a view exists.
    pub fn viewExists(self: *Catalog, name: []const u8) !bool {
        const key = try self.makeViewKey(name);
        defer self.allocator.free(key);

        const value = try self.tree.get(self.allocator, key);
        if (value) |v| {
            self.allocator.free(v);
            return true;
        }
        return false;
    }

    /// List all view names. Caller must free each name and the returned slice.
    pub fn listViews(self: *Catalog, allocator: Allocator) ![][]const u8 {
        var cursor = Cursor.init(allocator, &self.tree);
        defer cursor.deinit();

        try cursor.seekFirst();

        var names = std.ArrayListUnmanaged([]const u8){};
        errdefer {
            for (names.items) |n| allocator.free(n);
            names.deinit(allocator);
        }

        while (try cursor.next()) |entry| {
            defer allocator.free(entry.value);
            defer allocator.free(entry.key);

            // Only include view entries (key starts with VIEW_KEY_PREFIX)
            if (entry.key.len > VIEW_KEY_PREFIX.len and
                std.mem.eql(u8, entry.key[0..VIEW_KEY_PREFIX.len], VIEW_KEY_PREFIX))
            {
                const view_name = entry.key[VIEW_KEY_PREFIX.len..];
                const name_copy = try allocator.dupe(u8, view_name);
                errdefer allocator.free(name_copy);
                try names.append(allocator, name_copy);
            }
        }

        return names.toOwnedSlice(allocator);
    }

    // ── ENUM Types ──────────────────────────────────────────────────────

    fn makeTypeKey(self: *Catalog, name: []const u8) ![]u8 {
        const prefix = "type:";
        const key = try self.allocator.alloc(u8, prefix.len + name.len);
        @memcpy(key[0..prefix.len], prefix);
        @memcpy(key[prefix.len..], name);
        return key;
    }

    pub fn createEnumType(
        self: *Catalog,
        name: []const u8,
        values: []const []const u8,
    ) !void {
        const key = try self.makeTypeKey(name);
        defer self.allocator.free(key);

        // Check if type already exists
        const existing = try self.tree.get(self.allocator, key);
        if (existing) |v| {
            self.allocator.free(v);
            return CatalogError.TypeAlreadyExists;
        }

        // Check if name collides with a table
        if (try self.tableExists(name)) {
            return CatalogError.TableAlreadyExists;
        }

        // Serialize: [value_count: u16][values...]
        // Each value: [length: u16][text]
        var total_size: usize = 2; // value_count
        for (values) |v| {
            total_size += 2 + v.len; // length prefix + text
        }

        const data = try self.allocator.alloc(u8, total_size);
        defer self.allocator.free(data);

        var offset: usize = 0;
        std.mem.writeInt(u16, data[offset..][0..2], @intCast(values.len), .little);
        offset += 2;

        for (values) |v| {
            std.mem.writeInt(u16, data[offset..][0..2], @intCast(v.len), .little);
            offset += 2;
            @memcpy(data[offset..][0..v.len], v);
            offset += v.len;
        }

        try self.tree.insert(key, data);
    }

    pub fn getEnumType(self: *Catalog, name: []const u8) !EnumTypeInfo {
        const key = try self.makeTypeKey(name);
        defer self.allocator.free(key);

        const value = try self.tree.get(self.allocator, key) orelse
            return CatalogError.TypeNotFound;
        defer self.allocator.free(value);

        // Deserialize: [value_count: u16][values...]
        if (value.len < 2) return CatalogError.InvalidSchemaData;

        var offset: usize = 0;
        const value_count = std.mem.readInt(u16, value[offset..][0..2], .little);
        offset += 2;

        const values = try self.allocator.alloc([]const u8, value_count);
        var allocated_count: usize = 0;
        errdefer {
            for (values[0..allocated_count]) |v| self.allocator.free(v);
            self.allocator.free(values);
        }

        for (0..value_count) |i| {
            if (offset + 2 > value.len) return CatalogError.InvalidSchemaData;
            const len = std.mem.readInt(u16, value[offset..][0..2], .little);
            offset += 2;
            if (offset + len > value.len) return CatalogError.InvalidSchemaData;
            values[i] = try self.allocator.dupe(u8, value[offset..][0..len]);
            allocated_count += 1;
            offset += len;
        }

        return .{
            .name = try self.allocator.dupe(u8, name),
            .values = values,
            .allocator = self.allocator,
        };
    }

    pub fn dropEnumType(self: *Catalog, name: []const u8, if_exists: bool) !void {
        const key = try self.makeTypeKey(name);
        defer self.allocator.free(key);

        const exists = try self.tree.get(self.allocator, key);
        if (exists) |v| {
            self.allocator.free(v);
            try self.tree.delete(key);
        } else {
            if (!if_exists) return CatalogError.TypeNotFound;
        }
    }

    pub fn enumTypeExists(self: *Catalog, name: []const u8) !bool {
        const key = try self.makeTypeKey(name);
        defer self.allocator.free(key);
        const value = try self.tree.get(self.allocator, key);
        if (value) |v| {
            self.allocator.free(v);
            return true;
        }
        return false;
    }

    pub fn listEnumTypes(self: *Catalog, allocator: Allocator) ![][]const u8 {
        var cursor = Cursor.init(allocator, &self.tree);
        defer cursor.deinit();

        try cursor.seekFirst();

        var names = std.ArrayListUnmanaged([]const u8){};
        errdefer {
            for (names.items) |n| allocator.free(n);
            names.deinit(allocator);
        }

        const TYPE_KEY_PREFIX = "type:";
        while (try cursor.next()) |entry| {
            defer allocator.free(entry.value);
            defer allocator.free(entry.key);

            // Only include type entries (key starts with TYPE_KEY_PREFIX)
            if (entry.key.len > TYPE_KEY_PREFIX.len and
                std.mem.eql(u8, entry.key[0..TYPE_KEY_PREFIX.len], TYPE_KEY_PREFIX))
            {
                const type_name = entry.key[TYPE_KEY_PREFIX.len..];
                const name_copy = try allocator.dupe(u8, type_name);
                errdefer allocator.free(name_copy);
                try names.append(allocator, name_copy);
            }
        }

        return names.toOwnedSlice(allocator);
    }

    fn makeDomainKey(self: *Catalog, name: []const u8) ![]u8 {
        const prefix = "domain:";
        const key = try self.allocator.alloc(u8, prefix.len + name.len);
        @memcpy(key[0..prefix.len], prefix);
        @memcpy(key[prefix.len..], name);
        return key;
    }

    pub fn createDomain(
        self: *Catalog,
        name: []const u8,
        base_type: ast.DataType,
        constraint: ?[]const u8,
    ) !void {
        const key = try self.makeDomainKey(name);
        defer self.allocator.free(key);

        // Check if domain already exists
        const existing = try self.tree.get(self.allocator, key);
        if (existing) |v| {
            self.allocator.free(v);
            return CatalogError.TypeAlreadyExists;
        }

        // Check if name collides with a table or enum type
        if (try self.tableExists(name)) {
            return CatalogError.TableAlreadyExists;
        }
        if (try self.enumTypeExists(name)) {
            return CatalogError.TypeAlreadyExists;
        }

        // Serialize: [base_type: u8][has_constraint: u8][constraint_len: u16][constraint_text]
        const has_constraint: u8 = if (constraint != null) 1 else 0;
        const constraint_len: u16 = if (constraint) |c| @intCast(c.len) else 0;
        const total_size: usize = 1 + 1 + 2 + constraint_len;

        const data = try self.allocator.alloc(u8, total_size);
        defer self.allocator.free(data);

        var offset: usize = 0;
        data[offset] = @intFromEnum(base_type);
        offset += 1;
        data[offset] = has_constraint;
        offset += 1;
        std.mem.writeInt(u16, data[offset..][0..2], constraint_len, .little);
        offset += 2;
        if (constraint) |c| {
            @memcpy(data[offset..][0..c.len], c);
        }

        try self.tree.insert(key, data);
    }

    pub fn getDomain(self: *Catalog, name: []const u8) !DomainTypeInfo {
        const key = try self.makeDomainKey(name);
        defer self.allocator.free(key);

        const value = try self.tree.get(self.allocator, key) orelse
            return CatalogError.TypeNotFound;
        defer self.allocator.free(value);

        // Deserialize: [base_type: u8][has_constraint: u8][constraint_len: u16][constraint_text]
        if (value.len < 4) return CatalogError.InvalidSchemaData;

        var offset: usize = 0;
        const base_type_byte = value[offset];
        offset += 1;
        const has_constraint = value[offset];
        offset += 1;
        const constraint_len = std.mem.readInt(u16, value[offset..][0..2], .little);
        offset += 2;

        if (offset + constraint_len > value.len) return CatalogError.InvalidSchemaData;

        const constraint_opt = if (has_constraint == 1)
            try self.allocator.dupe(u8, value[offset..][0..constraint_len])
        else
            null;

        return .{
            .name = try self.allocator.dupe(u8, name),
            .base_type = @enumFromInt(base_type_byte),
            .constraint = constraint_opt,
            .allocator = self.allocator,
        };
    }

    pub fn dropDomain(self: *Catalog, name: []const u8, if_exists: bool) !void {
        const key = try self.makeDomainKey(name);
        defer self.allocator.free(key);

        const exists = try self.tree.get(self.allocator, key);
        if (exists) |v| {
            self.allocator.free(v);
            try self.tree.delete(key);
        } else {
            if (!if_exists) return CatalogError.TypeNotFound;
        }
    }

    pub fn domainExists(self: *Catalog, name: []const u8) !bool {
        const key = try self.makeDomainKey(name);
        defer self.allocator.free(key);
        const value = try self.tree.get(self.allocator, key);
        if (value) |v| {
            self.allocator.free(v);
            return true;
        }
        return false;
    }

    // ── Stored Functions ────────────────────────────────────────────────

    /// Function metadata stored in catalog.
    pub const FunctionInfo = struct {
        name: []const u8,
        parameters: []const FunctionParamInfo,
        return_type: FunctionReturnInfo,
        language: []const u8,
        body: []const u8,
        volatility: ast.FunctionVolatility,
        allocator: Allocator,

        pub fn deinit(self: FunctionInfo) void {
            self.allocator.free(self.name);
            for (self.parameters) |param| {
                self.allocator.free(param.name);
            }
            self.allocator.free(self.parameters);
            switch (self.return_type) {
                .scalar => {},
                .setof => {},
                .table => |cols| {
                    for (cols) |col| {
                        self.allocator.free(col.name);
                    }
                    self.allocator.free(cols);
                },
            }
            self.allocator.free(self.language);
            self.allocator.free(self.body);
        }
    };

    /// Function parameter metadata (simplified from AST).
    pub const FunctionParamInfo = struct {
        name: []const u8,
        data_type: ast.DataType,
    };

    /// Function return type metadata (simplified from AST).
    pub const FunctionReturnInfo = union(enum) {
        scalar: ast.DataType,
        setof: ast.DataType,
        table: []const FunctionTableColumn,
    };

    /// Table function column metadata.
    pub const FunctionTableColumn = struct {
        name: []const u8,
        data_type: ast.DataType,
    };

    fn makeFunctionKey(self: *Catalog, name: []const u8) ![]u8 {
        const prefix = "func:";
        const key = try self.allocator.alloc(u8, prefix.len + name.len);
        @memcpy(key[0..prefix.len], prefix);
        @memcpy(key[prefix.len..], name);
        return key;
    }

    /// Create or replace a stored function.
    pub fn createFunction(
        self: *Catalog,
        stmt: ast.CreateFunctionStmt,
    ) !void {
        const key = try self.makeFunctionKey(stmt.name);
        defer self.allocator.free(key);

        // If OR REPLACE, delete existing entry first
        if (stmt.or_replace) {
            // Check if exists
            const existing = try self.tree.get(self.allocator, key);
            if (existing) |v| {
                self.allocator.free(v);
                // Delete old entry before inserting new one
                try self.tree.delete(key);
            }
        } else {
            // Without OR REPLACE, fail if already exists
            const existing = try self.tree.get(self.allocator, key);
            if (existing) |v| {
                self.allocator.free(v);
                return CatalogError.TableAlreadyExists; // Reuse error for "already exists"
            }
        }

        // Serialize function metadata:
        // [param_count: u16]
        // for each param:
        //   [name_len: u16][name_bytes...][data_type: u8]
        // [return_type_tag: u8]
        // if return_type is scalar/setof:
        //   [data_type: u8]
        // if return_type is table:
        //   [col_count: u16]
        //   for each col:
        //     [name_len: u16][name_bytes...][data_type: u8]
        // [language_len: u16][language_bytes...]
        // [body_len: u32][body_bytes...]
        // [volatility: u8]

        var total_size: usize = 2; // param_count

        // Parameters
        for (stmt.parameters) |param| {
            total_size += 2 + param.name.len + 1; // name_len + name + data_type
        }

        // Return type
        total_size += 1; // return_type_tag
        switch (stmt.return_type) {
            .scalar, .setof => total_size += 1, // data_type
            .table => |cols| {
                total_size += 2; // col_count
                for (cols) |col| {
                    total_size += 2 + col.name.len + 1 + 1; // name_len + name + has_type + data_type
                }
            },
        }

        // Language
        total_size += 2 + stmt.language.len;

        // Body
        total_size += 4 + stmt.body.len;

        // Volatility
        total_size += 1;

        const data = try self.allocator.alloc(u8, total_size);
        defer self.allocator.free(data);

        var offset: usize = 0;

        // Serialize parameters
        std.mem.writeInt(u16, data[offset..][0..2], @intCast(stmt.parameters.len), .little);
        offset += 2;
        for (stmt.parameters) |param| {
            std.mem.writeInt(u16, data[offset..][0..2], @intCast(param.name.len), .little);
            offset += 2;
            @memcpy(data[offset..][0..param.name.len], param.name);
            offset += param.name.len;
            data[offset] = @intFromEnum(param.data_type);
            offset += 1;
        }

        // Serialize return type
        data[offset] = @intFromEnum(stmt.return_type);
        offset += 1;
        switch (stmt.return_type) {
            .scalar => |dt| {
                data[offset] = @intFromEnum(dt);
                offset += 1;
            },
            .setof => |dt| {
                data[offset] = @intFromEnum(dt);
                offset += 1;
            },
            .table => |cols| {
                std.mem.writeInt(u16, data[offset..][0..2], @intCast(cols.len), .little);
                offset += 2;
                for (cols) |col| {
                    std.mem.writeInt(u16, data[offset..][0..2], @intCast(col.name.len), .little);
                    offset += 2;
                    @memcpy(data[offset..][0..col.name.len], col.name);
                    offset += col.name.len;
                    // Store whether type is present
                    data[offset] = if (col.data_type != null) 1 else 0;
                    offset += 1;
                    // Store type value (default to untyped if null)
                    const dt = col.data_type orelse .type_integer;
                    data[offset] = @intFromEnum(dt);
                    offset += 1;
                }
            },
        }

        // Serialize language
        std.mem.writeInt(u16, data[offset..][0..2], @intCast(stmt.language.len), .little);
        offset += 2;
        @memcpy(data[offset..][0..stmt.language.len], stmt.language);
        offset += stmt.language.len;

        // Serialize body
        std.mem.writeInt(u32, data[offset..][0..4], @intCast(stmt.body.len), .little);
        offset += 4;
        @memcpy(data[offset..][0..stmt.body.len], stmt.body);
        offset += stmt.body.len;

        // Serialize volatility
        data[offset] = @intFromEnum(stmt.volatility);

        try self.tree.insert(key, data);
    }

    /// Retrieve function metadata by name. Caller must call FunctionInfo.deinit().
    pub fn getFunction(self: *Catalog, name: []const u8) !FunctionInfo {
        const key = try self.makeFunctionKey(name);
        defer self.allocator.free(key);

        const value = try self.tree.get(self.allocator, key) orelse
            return CatalogError.TypeNotFound; // Reuse error for "not found"
        defer self.allocator.free(value);

        if (value.len < 2) return CatalogError.InvalidSchemaData;

        var offset: usize = 0;

        // Deserialize parameters
        const param_count = std.mem.readInt(u16, value[offset..][0..2], .little);
        offset += 2;

        const params = try self.allocator.alloc(FunctionParamInfo, param_count);
        var allocated_params: usize = 0;
        errdefer {
            for (params[0..allocated_params]) |p| self.allocator.free(p.name);
            self.allocator.free(params);
        }

        for (0..param_count) |i| {
            if (offset + 2 > value.len) return CatalogError.InvalidSchemaData;
            const name_len = std.mem.readInt(u16, value[offset..][0..2], .little);
            offset += 2;
            if (offset + name_len > value.len) return CatalogError.InvalidSchemaData;
            const param_name = try self.allocator.dupe(u8, value[offset..][0..name_len]);
            offset += name_len;
            if (offset >= value.len) return CatalogError.InvalidSchemaData;
            const param_type: ast.DataType = @enumFromInt(value[offset]);
            offset += 1;
            params[i] = .{ .name = param_name, .data_type = param_type };
            allocated_params += 1;
        }

        // Deserialize return type
        if (offset >= value.len) return CatalogError.InvalidSchemaData;
        const return_tag_byte = value[offset];
        offset += 1;

        // We need to know the enum tag values for FunctionReturn
        // Looking at ast.zig, FunctionReturn is: scalar=0, table=1, setof=2 (auto-numbered)
        const return_type: FunctionReturnInfo = blk: {
            if (return_tag_byte == 0) { // scalar
                if (offset >= value.len) return CatalogError.InvalidSchemaData;
                const dt: ast.DataType = @enumFromInt(value[offset]);
                offset += 1;
                break :blk .{ .scalar = dt };
            } else if (return_tag_byte == 1) { // table
                if (offset + 2 > value.len) return CatalogError.InvalidSchemaData;
                const col_count = std.mem.readInt(u16, value[offset..][0..2], .little);
                offset += 2;

                const cols = try self.allocator.alloc(FunctionTableColumn, col_count);
                var allocated_cols: usize = 0;
                errdefer {
                    for (cols[0..allocated_cols]) |c| self.allocator.free(c.name);
                    self.allocator.free(cols);
                }

                for (0..col_count) |i| {
                    if (offset + 2 > value.len) return CatalogError.InvalidSchemaData;
                    const col_name_len = std.mem.readInt(u16, value[offset..][0..2], .little);
                    offset += 2;
                    if (offset + col_name_len > value.len) return CatalogError.InvalidSchemaData;
                    const col_name = try self.allocator.dupe(u8, value[offset..][0..col_name_len]);
                    offset += col_name_len;
                    // Read has_type flag
                    if (offset >= value.len) return CatalogError.InvalidSchemaData;
                    _ = value[offset]; // has_type flag (currently unused)
                    offset += 1;
                    // Read data_type value
                    if (offset >= value.len) return CatalogError.InvalidSchemaData;
                    const col_type: ast.DataType = @enumFromInt(value[offset]);
                    offset += 1;
                    cols[i] = .{ .name = col_name, .data_type = col_type };
                    allocated_cols += 1;
                }

                break :blk .{ .table = cols };
            } else if (return_tag_byte == 2) { // setof
                if (offset >= value.len) return CatalogError.InvalidSchemaData;
                const dt: ast.DataType = @enumFromInt(value[offset]);
                offset += 1;
                break :blk .{ .setof = dt };
            } else {
                return CatalogError.InvalidSchemaData;
            }
        };

        // Deserialize language
        if (offset + 2 > value.len) return CatalogError.InvalidSchemaData;
        const language_len = std.mem.readInt(u16, value[offset..][0..2], .little);
        offset += 2;
        if (offset + language_len > value.len) return CatalogError.InvalidSchemaData;
        const language = try self.allocator.dupe(u8, value[offset..][0..language_len]);
        errdefer self.allocator.free(language);
        offset += language_len;

        // Deserialize body
        if (offset + 4 > value.len) return CatalogError.InvalidSchemaData;
        const body_len = std.mem.readInt(u32, value[offset..][0..4], .little);
        offset += 4;
        if (offset + body_len > value.len) return CatalogError.InvalidSchemaData;
        const body = try self.allocator.dupe(u8, value[offset..][0..body_len]);
        errdefer self.allocator.free(body);
        offset += body_len;

        // Deserialize volatility
        if (offset >= value.len) return CatalogError.InvalidSchemaData;
        const volatility: ast.FunctionVolatility = @enumFromInt(value[offset]);

        return .{
            .name = try self.allocator.dupe(u8, name),
            .parameters = params,
            .return_type = return_type,
            .language = language,
            .body = body,
            .volatility = volatility,
            .allocator = self.allocator,
        };
    }

    /// Drop a stored function.
    pub fn dropFunction(self: *Catalog, name: []const u8, if_exists: bool) !void {
        const key = try self.makeFunctionKey(name);
        defer self.allocator.free(key);

        const exists = try self.tree.get(self.allocator, key);
        if (exists) |v| {
            self.allocator.free(v);
            try self.tree.delete(key);
        } else {
            if (!if_exists) return CatalogError.TypeNotFound;
        }
    }

    /// Check if a function exists.
    pub fn functionExists(self: *Catalog, name: []const u8) !bool {
        const key = try self.makeFunctionKey(name);
        defer self.allocator.free(key);
        const value = try self.tree.get(self.allocator, key);
        if (value) |v| {
            self.allocator.free(v);
            return true;
        }
        return false;
    }

    /// List all stored function names. Caller must free each name and the returned slice.
    pub fn listFunctions(self: *Catalog, allocator: Allocator) ![][]const u8 {
        var cursor = Cursor.init(allocator, &self.tree);
        defer cursor.deinit();

        try cursor.seekFirst();

        var names = std.ArrayListUnmanaged([]const u8){};
        errdefer {
            for (names.items) |n| allocator.free(n);
            names.deinit(allocator);
        }

        const func_prefix = "func:";
        while (try cursor.next()) |entry| {
            defer allocator.free(entry.value);
            defer allocator.free(entry.key);

            if (entry.key.len > func_prefix.len and
                std.mem.eql(u8, entry.key[0..func_prefix.len], func_prefix))
            {
                const func_name = entry.key[func_prefix.len..];
                const name_copy = try allocator.dupe(u8, func_name);
                errdefer allocator.free(name_copy);
                try names.append(allocator, name_copy);
            }
        }

        return names.toOwnedSlice(allocator);
    }

    // ── Triggers ──────────────────────────────────────────────────────────

    /// Trigger metadata stored in catalog.
    pub const TriggerInfo = struct {
        name: []const u8,
        table_name: []const u8,
        timing: ast.TriggerTiming,
        event: ast.TriggerEvent,
        update_columns: []const []const u8, // For UPDATE OF col1, col2, ...
        level: ast.TriggerLevel,
        when_condition: ?[]const u8, // Serialized WHEN condition (optional)
        body: []const u8,
        enabled: bool, // Trigger activation state
        allocator: Allocator,

        pub fn deinit(self: TriggerInfo) void {
            self.allocator.free(self.name);
            self.allocator.free(self.table_name);
            for (self.update_columns) |col| {
                self.allocator.free(col);
            }
            self.allocator.free(self.update_columns);
            if (self.when_condition) |cond| {
                self.allocator.free(cond);
            }
            self.allocator.free(self.body);
        }
    };

    fn makeTriggerKey(self: *Catalog, name: []const u8) ![]u8 {
        const prefix = "trig:";
        const key = try self.allocator.alloc(u8, prefix.len + name.len);
        @memcpy(key[0..prefix.len], prefix);
        @memcpy(key[prefix.len..], name);
        return key;
    }

    /// Create or replace a trigger.
    pub fn createTrigger(
        self: *Catalog,
        stmt: ast.CreateTriggerStmt,
    ) !void {
        const key = try self.makeTriggerKey(stmt.name);
        defer self.allocator.free(key);

        // If OR REPLACE, delete existing entry first
        if (stmt.or_replace) {
            // Check if exists
            const existing = try self.tree.get(self.allocator, key);
            if (existing) |v| {
                self.allocator.free(v);
                // Delete old entry before inserting new one
                try self.tree.delete(key);
            }
        } else {
            // Without OR REPLACE, fail if already exists
            const existing = try self.tree.get(self.allocator, key);
            if (existing) |v| {
                self.allocator.free(v);
                return CatalogError.TableAlreadyExists; // Reuse error for "already exists"
            }
        }

        // Serialize trigger metadata:
        // [table_name_len: u16][table_name_bytes...]
        // [timing: u8]
        // [event: u8]
        // [update_col_count: u16]
        // for each update_col:
        //   [col_name_len: u16][col_name_bytes...]
        // [level: u8]
        // [has_when: u8]
        // if has_when:
        //   [when_len: u32][when_bytes...]
        // [body_len: u32][body_bytes...]
        // [enabled: u8]

        var total_size: usize = 0;

        // Table name
        total_size += 2 + stmt.table_name.len;

        // Timing
        total_size += 1;

        // Event
        total_size += 1;

        // Update columns
        total_size += 2; // col_count
        for (stmt.update_columns) |col| {
            total_size += 2 + col.len;
        }

        // Level
        total_size += 1;

        // When condition
        total_size += 1; // has_when flag
        if (stmt.when_condition) |_| {
            // For now, we'll serialize the condition as empty (executor not implemented yet)
            total_size += 4; // when_len (0 for now)
        }

        // Body
        total_size += 4 + stmt.body.len;

        // Enabled (default true for new triggers)
        total_size += 1;

        const data = try self.allocator.alloc(u8, total_size);
        defer self.allocator.free(data);

        var offset: usize = 0;

        // Write table name
        std.mem.writeInt(u16, data[offset..][0..2], @intCast(stmt.table_name.len), .little);
        offset += 2;
        @memcpy(data[offset..][0..stmt.table_name.len], stmt.table_name);
        offset += stmt.table_name.len;

        // Write timing
        data[offset] = @intFromEnum(stmt.timing);
        offset += 1;

        // Write event
        data[offset] = @intFromEnum(stmt.event);
        offset += 1;

        // Write update columns
        std.mem.writeInt(u16, data[offset..][0..2], @intCast(stmt.update_columns.len), .little);
        offset += 2;
        for (stmt.update_columns) |col| {
            std.mem.writeInt(u16, data[offset..][0..2], @intCast(col.len), .little);
            offset += 2;
            @memcpy(data[offset..][0..col.len], col);
            offset += col.len;
        }

        // Write level
        data[offset] = @intFromEnum(stmt.level);
        offset += 1;

        // Write when condition
        if (stmt.when_condition) |_| {
            data[offset] = 1; // has_when = true
            offset += 1;
            // Serialize as empty string for now (executor TBD)
            std.mem.writeInt(u32, data[offset..][0..4], 0, .little);
            offset += 4;
        } else {
            data[offset] = 0; // has_when = false
            offset += 1;
        }

        // Write body
        std.mem.writeInt(u32, data[offset..][0..4], @intCast(stmt.body.len), .little);
        offset += 4;
        @memcpy(data[offset..][0..stmt.body.len], stmt.body);
        offset += stmt.body.len;

        // Write enabled (true by default)
        data[offset] = 1;

        try self.tree.insert(key, data);
    }

    /// Retrieve a trigger by name.
    pub fn getTrigger(self: *Catalog, name: []const u8) !TriggerInfo {
        const key = try self.makeTriggerKey(name);
        defer self.allocator.free(key);

        const value = try self.tree.get(self.allocator, key);
        if (value == null) return CatalogError.TypeNotFound; // Reuse error

        defer self.allocator.free(value.?);

        var offset: usize = 0;

        // Deserialize table name
        if (offset + 2 > value.?.len) return CatalogError.InvalidSchemaData;
        const table_name_len = std.mem.readInt(u16, value.?[offset..][0..2], .little);
        offset += 2;
        if (offset + table_name_len > value.?.len) return CatalogError.InvalidSchemaData;
        const table_name = try self.allocator.dupe(u8, value.?[offset..][0..table_name_len]);
        errdefer self.allocator.free(table_name);
        offset += table_name_len;

        // Deserialize timing
        if (offset >= value.?.len) return CatalogError.InvalidSchemaData;
        const timing: ast.TriggerTiming = @enumFromInt(value.?[offset]);
        offset += 1;

        // Deserialize event
        if (offset >= value.?.len) return CatalogError.InvalidSchemaData;
        const event: ast.TriggerEvent = @enumFromInt(value.?[offset]);
        offset += 1;

        // Deserialize update columns
        if (offset + 2 > value.?.len) return CatalogError.InvalidSchemaData;
        const col_count = std.mem.readInt(u16, value.?[offset..][0..2], .little);
        offset += 2;
        const update_cols = try self.allocator.alloc([]const u8, col_count);
        errdefer {
            for (update_cols) |col| self.allocator.free(col);
            self.allocator.free(update_cols);
        }
        for (update_cols) |*col| {
            if (offset + 2 > value.?.len) return CatalogError.InvalidSchemaData;
            const col_len = std.mem.readInt(u16, value.?[offset..][0..2], .little);
            offset += 2;
            if (offset + col_len > value.?.len) return CatalogError.InvalidSchemaData;
            col.* = try self.allocator.dupe(u8, value.?[offset..][0..col_len]);
            offset += col_len;
        }

        // Deserialize level
        if (offset >= value.?.len) return CatalogError.InvalidSchemaData;
        const level: ast.TriggerLevel = @enumFromInt(value.?[offset]);
        offset += 1;

        // Deserialize when condition
        if (offset >= value.?.len) return CatalogError.InvalidSchemaData;
        const has_when = value.?[offset];
        offset += 1;
        var when_condition: ?[]const u8 = null;
        if (has_when == 1) {
            if (offset + 4 > value.?.len) return CatalogError.InvalidSchemaData;
            const when_len = std.mem.readInt(u32, value.?[offset..][0..4], .little);
            offset += 4;
            if (when_len > 0) {
                if (offset + when_len > value.?.len) return CatalogError.InvalidSchemaData;
                when_condition = try self.allocator.dupe(u8, value.?[offset..][0..when_len]);
                offset += when_len;
            }
        }
        errdefer if (when_condition) |cond| self.allocator.free(cond);

        // Deserialize body
        if (offset + 4 > value.?.len) return CatalogError.InvalidSchemaData;
        const body_len = std.mem.readInt(u32, value.?[offset..][0..4], .little);
        offset += 4;
        if (offset + body_len > value.?.len) return CatalogError.InvalidSchemaData;
        const body = try self.allocator.dupe(u8, value.?[offset..][0..body_len]);
        errdefer self.allocator.free(body);
        offset += body_len;

        // Deserialize enabled
        if (offset >= value.?.len) return CatalogError.InvalidSchemaData;
        const enabled = value.?[offset] == 1;

        return .{
            .name = try self.allocator.dupe(u8, name),
            .table_name = table_name,
            .timing = timing,
            .event = event,
            .update_columns = update_cols,
            .level = level,
            .when_condition = when_condition,
            .body = body,
            .enabled = enabled,
            .allocator = self.allocator,
        };
    }

    /// Drop a trigger.
    pub fn dropTrigger(self: *Catalog, name: []const u8, if_exists: bool) !void {
        const key = try self.makeTriggerKey(name);
        defer self.allocator.free(key);

        const exists = try self.tree.get(self.allocator, key);
        if (exists) |v| {
            self.allocator.free(v);
            try self.tree.delete(key);
        } else {
            if (!if_exists) return CatalogError.TypeNotFound;
        }
    }

    /// Check if a trigger exists.
    pub fn triggerExists(self: *Catalog, name: []const u8) !bool {
        const key = try self.makeTriggerKey(name);
        defer self.allocator.free(key);
        const value = try self.tree.get(self.allocator, key);
        if (value) |v| {
            self.allocator.free(v);
            return true;
        }
        return false;
    }

    /// Alter trigger enabled/disabled state.
    pub fn alterTrigger(
        self: *Catalog,
        name: []const u8,
        enable: bool,
    ) !void {
        // Retrieve existing trigger
        var info = try self.getTrigger(name);
        defer info.deinit();

        // Update enabled state
        info.enabled = enable;

        // Re-serialize with updated state
        const key = try self.makeTriggerKey(name);
        defer self.allocator.free(key);

        var total_size: usize = 0;
        total_size += 2 + info.table_name.len;
        total_size += 1; // timing
        total_size += 1; // event
        total_size += 2; // col_count
        for (info.update_columns) |col| {
            total_size += 2 + col.len;
        }
        total_size += 1; // level
        total_size += 1; // has_when
        if (info.when_condition) |cond| {
            total_size += 4 + cond.len;
        }
        total_size += 4 + info.body.len;
        total_size += 1; // enabled

        const data = try self.allocator.alloc(u8, total_size);
        defer self.allocator.free(data);

        var offset: usize = 0;

        // Write table name
        std.mem.writeInt(u16, data[offset..][0..2], @intCast(info.table_name.len), .little);
        offset += 2;
        @memcpy(data[offset..][0..info.table_name.len], info.table_name);
        offset += info.table_name.len;

        // Write timing
        data[offset] = @intFromEnum(info.timing);
        offset += 1;

        // Write event
        data[offset] = @intFromEnum(info.event);
        offset += 1;

        // Write update columns
        std.mem.writeInt(u16, data[offset..][0..2], @intCast(info.update_columns.len), .little);
        offset += 2;
        for (info.update_columns) |col| {
            std.mem.writeInt(u16, data[offset..][0..2], @intCast(col.len), .little);
            offset += 2;
            @memcpy(data[offset..][0..col.len], col);
            offset += col.len;
        }

        // Write level
        data[offset] = @intFromEnum(info.level);
        offset += 1;

        // Write when condition
        if (info.when_condition) |cond| {
            data[offset] = 1; // has_when = true
            offset += 1;
            std.mem.writeInt(u32, data[offset..][0..4], @intCast(cond.len), .little);
            offset += 4;
            @memcpy(data[offset..][0..cond.len], cond);
            offset += cond.len;
        } else {
            data[offset] = 0; // has_when = false
            offset += 1;
        }

        // Write body
        std.mem.writeInt(u32, data[offset..][0..4], @intCast(info.body.len), .little);
        offset += 4;
        @memcpy(data[offset..][0..info.body.len], info.body);
        offset += info.body.len;

        // Write enabled (updated value)
        data[offset] = if (info.enabled) 1 else 0;

        // Delete old entry before inserting updated one
        try self.tree.delete(key);
        try self.tree.insert(key, data);
    }

    /// List all trigger names. Caller must free each name and the returned slice.
    pub fn listTriggers(self: *Catalog, allocator: Allocator) ![][]const u8 {
        var cursor = Cursor.init(allocator, &self.tree);
        defer cursor.deinit();

        try cursor.seekFirst();

        var names = std.ArrayListUnmanaged([]const u8){};
        errdefer {
            for (names.items) |n| allocator.free(n);
            names.deinit(allocator);
        }

        const trig_prefix = "trig:";
        while (try cursor.next()) |entry| {
            defer allocator.free(entry.value);
            defer allocator.free(entry.key);

            if (entry.key.len > trig_prefix.len and
                std.mem.eql(u8, entry.key[0..trig_prefix.len], trig_prefix))
            {
                const trig_name = entry.key[trig_prefix.len..];
                const name_copy = try allocator.dupe(u8, trig_name);
                errdefer allocator.free(name_copy);
                try names.append(allocator, name_copy);
            }
        }

        return names.toOwnedSlice(allocator);
    }

    // ── Role Management ─────────────────────────────────────────────────

    /// Role metadata stored in catalog.
    pub const RoleInfo = struct {
        name: []const u8,
        login: bool,
        superuser: bool,
        createdb: bool,
        createrole: bool,
        inherit: bool,
        password: ?[]const u8,
        valid_until: ?[]const u8,
        allocator: Allocator,

        pub fn deinit(self: RoleInfo) void {
            self.allocator.free(self.name);
            if (self.password) |pwd| {
                self.allocator.free(pwd);
            }
            if (self.valid_until) |ts| {
                self.allocator.free(ts);
            }
        }
    };

    fn makeRoleKey(self: *Catalog, name: []const u8) ![]u8 {
        const prefix = "role:";
        const key = try self.allocator.alloc(u8, prefix.len + name.len);
        @memcpy(key[0..prefix.len], prefix);
        @memcpy(key[prefix.len..], name);
        return key;
    }

    /// Create or replace a role.
    pub fn createRole(
        self: *Catalog,
        stmt: ast.CreateRoleStmt,
    ) !void {
        const key = try self.makeRoleKey(stmt.name);
        defer self.allocator.free(key);

        // If OR REPLACE, delete existing entry first
        if (stmt.or_replace) {
            const existing = try self.tree.get(self.allocator, key);
            if (existing) |v| {
                self.allocator.free(v);
                try self.tree.delete(key);
            }
        } else {
            // Without OR REPLACE, fail if already exists
            const existing = try self.tree.get(self.allocator, key);
            if (existing) |v| {
                self.allocator.free(v);
                return CatalogError.TableAlreadyExists; // Reuse error for "already exists"
            }
        }

        // Serialize role metadata:
        // [login: u8]
        // [superuser: u8]
        // [createdb: u8]
        // [createrole: u8]
        // [inherit: u8]
        // [has_password: u8]
        // if has_password:
        //   [password_len: u32][password_bytes...]
        // [has_valid_until: u8]
        // if has_valid_until:
        //   [valid_until_len: u32][valid_until_bytes...]

        var total_size: usize = 6; // 5 bool flags + has_password

        if (stmt.options.password) |pwd| {
            total_size += 4 + pwd.len; // password_len + password
        }

        total_size += 1; // has_valid_until
        if (stmt.options.valid_until) |ts| {
            total_size += 4 + ts.len; // valid_until_len + timestamp
        }

        const data = try self.allocator.alloc(u8, total_size);
        defer self.allocator.free(data);

        var offset: usize = 0;

        // Write boolean flags (default to true for login, inherit; false for others)
        data[offset] = if (stmt.options.login) |v| @intFromBool(v) else 1; // LOGIN by default
        offset += 1;
        data[offset] = if (stmt.options.superuser) |v| @intFromBool(v) else 0;
        offset += 1;
        data[offset] = if (stmt.options.createdb) |v| @intFromBool(v) else 0;
        offset += 1;
        data[offset] = if (stmt.options.createrole) |v| @intFromBool(v) else 0;
        offset += 1;
        data[offset] = if (stmt.options.inherit) |v| @intFromBool(v) else 1; // INHERIT by default
        offset += 1;

        // Write password
        if (stmt.options.password) |pwd| {
            data[offset] = 1; // has_password
            offset += 1;
            std.mem.writeInt(u32, data[offset..][0..4], @intCast(pwd.len), .little);
            offset += 4;
            @memcpy(data[offset..][0..pwd.len], pwd);
            offset += pwd.len;
        } else {
            data[offset] = 0; // has_password = false
            offset += 1;
        }

        // Write valid_until
        if (stmt.options.valid_until) |ts| {
            data[offset] = 1; // has_valid_until
            offset += 1;
            std.mem.writeInt(u32, data[offset..][0..4], @intCast(ts.len), .little);
            offset += 4;
            @memcpy(data[offset..][0..ts.len], ts);
        } else {
            data[offset] = 0; // has_valid_until = false
        }

        try self.tree.insert(key, data);
    }

    /// Retrieve role metadata by name. Caller must call RoleInfo.deinit().
    pub fn getRole(self: *Catalog, name: []const u8) !RoleInfo {
        const key = try self.makeRoleKey(name);
        defer self.allocator.free(key);

        const value = try self.tree.get(self.allocator, key);
        if (value == null) return CatalogError.TypeNotFound;
        defer self.allocator.free(value.?);

        var offset: usize = 0;

        // Read boolean flags
        if (offset >= value.?.len) return CatalogError.InvalidSchemaData;
        const login = value.?[offset] == 1;
        offset += 1;

        if (offset >= value.?.len) return CatalogError.InvalidSchemaData;
        const superuser = value.?[offset] == 1;
        offset += 1;

        if (offset >= value.?.len) return CatalogError.InvalidSchemaData;
        const createdb = value.?[offset] == 1;
        offset += 1;

        if (offset >= value.?.len) return CatalogError.InvalidSchemaData;
        const createrole = value.?[offset] == 1;
        offset += 1;

        if (offset >= value.?.len) return CatalogError.InvalidSchemaData;
        const inherit = value.?[offset] == 1;
        offset += 1;

        // Read password
        if (offset >= value.?.len) return CatalogError.InvalidSchemaData;
        const has_password = value.?[offset] == 1;
        offset += 1;

        var password: ?[]const u8 = null;
        if (has_password) {
            if (offset + 4 > value.?.len) return CatalogError.InvalidSchemaData;
            const pwd_len = std.mem.readInt(u32, value.?[offset..][0..4], .little);
            offset += 4;

            if (offset + pwd_len > value.?.len) return CatalogError.InvalidSchemaData;
            password = try self.allocator.dupe(u8, value.?[offset..][0..pwd_len]);
            offset += pwd_len;
        }
        errdefer if (password) |p| self.allocator.free(p);

        // Read valid_until
        if (offset >= value.?.len) return CatalogError.InvalidSchemaData;
        const has_valid_until = value.?[offset] == 1;
        offset += 1;

        var valid_until: ?[]const u8 = null;
        if (has_valid_until) {
            if (offset + 4 > value.?.len) return CatalogError.InvalidSchemaData;
            const ts_len = std.mem.readInt(u32, value.?[offset..][0..4], .little);
            offset += 4;

            if (offset + ts_len > value.?.len) return CatalogError.InvalidSchemaData;
            valid_until = try self.allocator.dupe(u8, value.?[offset..][0..ts_len]);
        }

        return .{
            .name = try self.allocator.dupe(u8, name),
            .login = login,
            .superuser = superuser,
            .createdb = createdb,
            .createrole = createrole,
            .inherit = inherit,
            .password = password,
            .valid_until = valid_until,
            .allocator = self.allocator,
        };
    }

    /// Drop a role.
    pub fn dropRole(self: *Catalog, name: []const u8, if_exists: bool) !void {
        const key = try self.makeRoleKey(name);
        defer self.allocator.free(key);

        const exists = try self.tree.get(self.allocator, key);
        if (exists) |v| {
            self.allocator.free(v);
            try self.tree.delete(key);
        } else {
            if (!if_exists) return CatalogError.TypeNotFound;
        }
    }

    /// Check if a role exists.
    pub fn roleExists(self: *Catalog, name: []const u8) !bool {
        const key = try self.makeRoleKey(name);
        defer self.allocator.free(key);
        const value = try self.tree.get(self.allocator, key);
        if (value) |v| {
            self.allocator.free(v);
            return true;
        }
        return false;
    }

    /// Alter role options.
    pub fn alterRole(
        self: *Catalog,
        name: []const u8,
        options: ast.RoleOptions,
    ) !void {
        // Retrieve existing role
        var info = try self.getRole(name);
        defer info.deinit();

        // Update fields based on options
        if (options.login) |v| info.login = v;
        if (options.superuser) |v| info.superuser = v;
        if (options.createdb) |v| info.createdb = v;
        if (options.createrole) |v| info.createrole = v;
        if (options.inherit) |v| info.inherit = v;

        if (options.password) |pwd| {
            if (info.password) |old_pwd| {
                self.allocator.free(old_pwd);
            }
            info.password = try self.allocator.dupe(u8, pwd);
        }

        if (options.valid_until) |ts| {
            if (info.valid_until) |old_ts| {
                self.allocator.free(old_ts);
            }
            info.valid_until = try self.allocator.dupe(u8, ts);
        }

        // Serialize updated role (same format as createRole)
        var total_size: usize = 6; // 5 bool flags + has_password

        if (info.password) |pwd| {
            total_size += 4 + pwd.len;
        }

        total_size += 1; // has_valid_until
        if (info.valid_until) |ts| {
            total_size += 4 + ts.len;
        }

        const data = try self.allocator.alloc(u8, total_size);
        defer self.allocator.free(data);

        var offset: usize = 0;

        // Write boolean flags
        data[offset] = @intFromBool(info.login);
        offset += 1;
        data[offset] = @intFromBool(info.superuser);
        offset += 1;
        data[offset] = @intFromBool(info.createdb);
        offset += 1;
        data[offset] = @intFromBool(info.createrole);
        offset += 1;
        data[offset] = @intFromBool(info.inherit);
        offset += 1;

        // Write password
        if (info.password) |pwd| {
            data[offset] = 1;
            offset += 1;
            std.mem.writeInt(u32, data[offset..][0..4], @intCast(pwd.len), .little);
            offset += 4;
            @memcpy(data[offset..][0..pwd.len], pwd);
            offset += pwd.len;
        } else {
            data[offset] = 0;
            offset += 1;
        }

        // Write valid_until
        if (info.valid_until) |ts| {
            data[offset] = 1;
            offset += 1;
            std.mem.writeInt(u32, data[offset..][0..4], @intCast(ts.len), .little);
            offset += 4;
            @memcpy(data[offset..][0..ts.len], ts);
        } else {
            data[offset] = 0;
        }

        // Delete old entry before inserting updated one
        const key = try self.makeRoleKey(name);
        defer self.allocator.free(key);
        try self.tree.delete(key);
        try self.tree.insert(key, data);
    }

    /// List all role names. Caller must free each name and the returned slice.
    pub fn listRoles(self: *Catalog, allocator: Allocator) ![][]const u8 {
        var cursor = Cursor.init(allocator, &self.tree);
        defer cursor.deinit();

        try cursor.seekFirst();

        var names = std.ArrayListUnmanaged([]const u8){};
        errdefer {
            for (names.items) |n| allocator.free(n);
            names.deinit(allocator);
        }

        const role_prefix = "role:";
        while (try cursor.next()) |entry| {
            defer allocator.free(entry.value);
            defer allocator.free(entry.key);

            if (entry.key.len > role_prefix.len and
                std.mem.eql(u8, entry.key[0..role_prefix.len], role_prefix))
            {
                const role_name = entry.key[role_prefix.len..];
                const name_copy = try allocator.dupe(u8, role_name);
                errdefer allocator.free(name_copy);
                try names.append(allocator, name_copy);
            }
        }

        return names.toOwnedSlice(allocator);
    }
};

// ── Tests ───────────────────────────────────────────────────────────────

test "ColumnType from AST DataType" {
    try std.testing.expectEqual(ColumnType.integer, columnTypeFromAst(.type_integer));
    try std.testing.expectEqual(ColumnType.integer, columnTypeFromAst(.type_int));
    try std.testing.expectEqual(ColumnType.real, columnTypeFromAst(.type_real));
    try std.testing.expectEqual(ColumnType.text, columnTypeFromAst(.type_text));
    try std.testing.expectEqual(ColumnType.text, columnTypeFromAst(.type_varchar));
    try std.testing.expectEqual(ColumnType.blob, columnTypeFromAst(.type_blob));
    try std.testing.expectEqual(ColumnType.boolean, columnTypeFromAst(.type_boolean));
    try std.testing.expectEqual(ColumnType.untyped, columnTypeFromAst(null));
    // SERIAL/BIGSERIAL map to integer
    try std.testing.expectEqual(ColumnType.integer, columnTypeFromAst(.type_serial));
    try std.testing.expectEqual(ColumnType.integer, columnTypeFromAst(.type_bigserial));
}

test "ConstraintFlags from AST constraints" {
    const flags1 = constraintFlagsFromAst(&.{
        .{ .primary_key = .{ .autoincrement = true } },
    });
    try std.testing.expect(flags1.primary_key);
    try std.testing.expect(flags1.autoincrement);
    try std.testing.expect(flags1.not_null); // implied

    const flags2 = constraintFlagsFromAst(&.{
        .not_null,
        .unique,
    });
    try std.testing.expect(!flags2.primary_key);
    try std.testing.expect(flags2.not_null);
    try std.testing.expect(flags2.unique);
}

test "ConstraintFlags bitfield roundtrip" {
    const flags = ConstraintFlags{
        .primary_key = true,
        .not_null = true,
        .unique = false,
        .autoincrement = true,
    };
    const byte: u8 = @bitCast(flags);
    const decoded: ConstraintFlags = @bitCast(byte);
    try std.testing.expect(decoded.primary_key);
    try std.testing.expect(decoded.not_null);
    try std.testing.expect(!decoded.unique);
    try std.testing.expect(decoded.autoincrement);
}

test "serialize and deserialize table" {
    const allocator = std.testing.allocator;

    const columns = [_]ColumnInfo{
        .{ .name = "id", .column_type = .integer, .flags = .{ .primary_key = true, .not_null = true } },
        .{ .name = "name", .column_type = .text, .flags = .{ .not_null = true } },
        .{ .name = "email", .column_type = .text, .flags = .{} },
        .{ .name = "age", .column_type = .integer, .flags = .{} },
    };

    const tc = [_]TableConstraintInfo{
        .{ .unique = &.{ "email", "name" } },
    };

    const data = try serializeTable(allocator, &columns, &tc, 42);
    defer allocator.free(data);

    const table = try deserializeTable(allocator, "users", data);
    defer table.deinit(allocator);

    try std.testing.expectEqualStrings("users", table.name);
    try std.testing.expectEqual(@as(u32, 42), table.data_root_page_id);
    try std.testing.expectEqual(@as(usize, 4), table.columns.len);
    try std.testing.expectEqualStrings("id", table.columns[0].name);
    try std.testing.expectEqual(ColumnType.integer, table.columns[0].column_type);
    try std.testing.expect(table.columns[0].flags.primary_key);
    try std.testing.expect(table.columns[0].flags.not_null);
    try std.testing.expectEqualStrings("name", table.columns[1].name);
    try std.testing.expectEqual(ColumnType.text, table.columns[1].column_type);
    try std.testing.expectEqualStrings("email", table.columns[2].name);
    try std.testing.expectEqual(@as(usize, 1), table.table_constraints.len);
    switch (table.table_constraints[0]) {
        .unique => |cols| {
            try std.testing.expectEqual(@as(usize, 2), cols.len);
            try std.testing.expectEqualStrings("email", cols[0]);
            try std.testing.expectEqualStrings("name", cols[1]);
        },
        else => return error.TestUnexpectedResult,
    }
}

test "serialize empty table" {
    const allocator = std.testing.allocator;

    const data = try serializeTable(allocator, &.{}, &.{}, 0);
    defer allocator.free(data);

    const table = try deserializeTable(allocator, "empty", data);
    defer table.deinit(allocator);

    try std.testing.expectEqualStrings("empty", table.name);
    try std.testing.expectEqual(@as(u32, 0), table.data_root_page_id);
    try std.testing.expectEqual(@as(usize, 0), table.columns.len);
    try std.testing.expectEqual(@as(usize, 0), table.table_constraints.len);
}

test "deserialize invalid data" {
    const allocator = std.testing.allocator;

    // Too short (needs at least 6 bytes: 4 for page_id + 2 for col_count)
    try std.testing.expectError(error.InvalidSchemaData, deserializeTable(allocator, "bad", &.{}));
    try std.testing.expectError(error.InvalidSchemaData, deserializeTable(allocator, "bad", &.{ 0, 0, 0, 0, 0 }));
}

test "serialize and deserialize JSON/JSONB columns" {
    const allocator = std.testing.allocator;

    const columns = [_]ColumnInfo{
        .{ .name = "id", .column_type = .integer, .flags = .{ .primary_key = true, .not_null = true } },
        .{ .name = "data", .column_type = .json, .flags = .{} },
        .{ .name = "metadata", .column_type = .jsonb, .flags = .{ .not_null = true } },
        .{ .name = "config", .column_type = .json, .flags = .{} },
    };

    const data = try serializeTable(allocator, &columns, &.{}, 123);
    defer allocator.free(data);

    const table = try deserializeTable(allocator, "documents", data);
    defer table.deinit(allocator);

    try std.testing.expectEqualStrings("documents", table.name);
    try std.testing.expectEqual(@as(u32, 123), table.data_root_page_id);
    try std.testing.expectEqual(@as(usize, 4), table.columns.len);

    // Verify column types
    try std.testing.expectEqualStrings("id", table.columns[0].name);
    try std.testing.expectEqual(ColumnType.integer, table.columns[0].column_type);

    try std.testing.expectEqualStrings("data", table.columns[1].name);
    try std.testing.expectEqual(ColumnType.json, table.columns[1].column_type);
    try std.testing.expect(!table.columns[1].flags.not_null);

    try std.testing.expectEqualStrings("metadata", table.columns[2].name);
    try std.testing.expectEqual(ColumnType.jsonb, table.columns[2].column_type);
    try std.testing.expect(table.columns[2].flags.not_null);

    try std.testing.expectEqualStrings("config", table.columns[3].name);
    try std.testing.expectEqual(ColumnType.json, table.columns[3].column_type);
}

// Helper: create a test Catalog backed by a temp file.
const TestCatalog = struct {
    pager: *Pager,
    pool: *BufferPool,
    catalog: Catalog,
    path: []const u8,

    fn setup(allocator: Allocator, path: []const u8) !TestCatalog {
        const pager = try allocator.create(Pager);
        pager.* = try Pager.init(allocator, path, .{});

        const pool = try allocator.create(BufferPool);
        pool.* = try BufferPool.init(allocator, pager, 64);

        const catalog = try Catalog.init(allocator, pool, true);

        return .{
            .pager = pager,
            .pool = pool,
            .catalog = catalog,
            .path = path,
        };
    }

    fn teardown(self: *TestCatalog, allocator: Allocator) void {
        self.pool.deinit();
        self.pager.deinit();
        allocator.destroy(self.pool);
        allocator.destroy(self.pager);
        std.fs.cwd().deleteFile(self.path) catch {};
    }
};

test "Catalog create and get table" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_create.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    const columns = [_]ColumnInfo{
        .{ .name = "id", .column_type = .integer, .flags = .{ .primary_key = true, .not_null = true } },
        .{ .name = "name", .column_type = .text, .flags = .{ .not_null = true } },
    };

    try tc.catalog.createTable("users", &columns, &.{}, 0);

    // Retrieve it
    const table = try tc.catalog.getTable("users");
    defer table.deinit(allocator);

    try std.testing.expectEqualStrings("users", table.name);
    try std.testing.expectEqual(@as(usize, 2), table.columns.len);
    try std.testing.expectEqualStrings("id", table.columns[0].name);
    try std.testing.expectEqual(ColumnType.integer, table.columns[0].column_type);
}

test "Catalog tableExists" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_exists.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    try std.testing.expect(!try tc.catalog.tableExists("users"));

    try tc.catalog.createTable("users", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{} },
    }, &.{}, 0);

    try std.testing.expect(try tc.catalog.tableExists("users"));
    try std.testing.expect(!try tc.catalog.tableExists("posts"));
}

test "Catalog duplicate table error" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_dup.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    try tc.catalog.createTable("t1", &.{
        .{ .name = "a", .column_type = .integer, .flags = .{} },
    }, &.{}, 0);

    try std.testing.expectError(CatalogError.TableAlreadyExists, tc.catalog.createTable("t1", &.{
        .{ .name = "b", .column_type = .text, .flags = .{} },
    }, &.{}, 0));
}

test "Catalog drop table" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_drop.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    try tc.catalog.createTable("t1", &.{
        .{ .name = "x", .column_type = .integer, .flags = .{} },
    }, &.{}, 0);

    try std.testing.expect(try tc.catalog.tableExists("t1"));

    try tc.catalog.dropTable("t1", false);

    try std.testing.expect(!try tc.catalog.tableExists("t1"));
}

test "Catalog drop nonexistent table" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_drop_ne.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    // Without IF EXISTS — should error
    try std.testing.expectError(CatalogError.TableNotFound, tc.catalog.dropTable("ghost", false));

    // With IF EXISTS — should succeed silently
    try tc.catalog.dropTable("ghost", true);
}

test "Catalog listTables" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_list.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    // Empty catalog
    const empty = try tc.catalog.listTables(allocator);
    defer allocator.free(empty);
    try std.testing.expectEqual(@as(usize, 0), empty.len);

    // Create some tables (B+Tree stores keys sorted lexicographically)
    try tc.catalog.createTable("alpha", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{} },
    }, &.{}, 0);
    try tc.catalog.createTable("beta", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{} },
    }, &.{}, 0);
    try tc.catalog.createTable("gamma", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{} },
    }, &.{}, 0);

    const names = try tc.catalog.listTables(allocator);
    defer {
        for (names) |n| allocator.free(n);
        allocator.free(names);
    }

    try std.testing.expectEqual(@as(usize, 3), names.len);
    // B+Tree keys are sorted
    try std.testing.expectEqualStrings("alpha", names[0]);
    try std.testing.expectEqualStrings("beta", names[1]);
    try std.testing.expectEqualStrings("gamma", names[2]);
}

test "Catalog createTableFromAst" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_ast.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    const create_stmt = ast.CreateTableStmt{
        .name = "users",
        .columns = &.{
            .{
                .name = "id",
                .data_type = .type_integer,
                .constraints = &.{
                    .{ .primary_key = .{ .autoincrement = true } },
                },
            },
            .{
                .name = "name",
                .data_type = .type_text,
                .constraints = &.{
                    .not_null,
                },
            },
            .{
                .name = "email",
                .data_type = .type_varchar,
                .constraints = &.{
                    .unique,
                },
            },
        },
        .table_constraints = &.{},
    };

    try tc.catalog.createTableFromAst(&create_stmt);

    const table = try tc.catalog.getTable("users");
    defer table.deinit(allocator);

    try std.testing.expectEqualStrings("users", table.name);
    try std.testing.expectEqual(@as(usize, 3), table.columns.len);
    try std.testing.expectEqualStrings("id", table.columns[0].name);
    try std.testing.expectEqual(ColumnType.integer, table.columns[0].column_type);
    try std.testing.expect(table.columns[0].flags.primary_key);
    try std.testing.expect(table.columns[0].flags.autoincrement);
    try std.testing.expectEqualStrings("name", table.columns[1].name);
    try std.testing.expect(table.columns[1].flags.not_null);
    try std.testing.expectEqualStrings("email", table.columns[2].name);
    try std.testing.expectEqual(ColumnType.text, table.columns[2].column_type);
    try std.testing.expect(table.columns[2].flags.unique);
}

test "Catalog createTableFromAst with IF NOT EXISTS" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_ast_ine.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    const create_stmt = ast.CreateTableStmt{
        .name = "t1",
        .if_not_exists = true,
        .columns = &.{
            .{ .name = "a", .data_type = .type_integer },
        },
    };

    try tc.catalog.createTableFromAst(&create_stmt);
    // Second time should succeed silently
    try tc.catalog.createTableFromAst(&create_stmt);

    try std.testing.expect(try tc.catalog.tableExists("t1"));
}

test "Catalog findColumn" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_findcol.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    try tc.catalog.createTable("users", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{ .primary_key = true, .not_null = true } },
        .{ .name = "name", .column_type = .text, .flags = .{ .not_null = true } },
        .{ .name = "email", .column_type = .text, .flags = .{} },
    }, &.{}, 0);

    const result = try tc.catalog.findColumn("users", "name");
    defer allocator.free(result.info.name);

    try std.testing.expectEqual(@as(usize, 1), result.index);
    try std.testing.expectEqualStrings("name", result.info.name);
    try std.testing.expectEqual(ColumnType.text, result.info.column_type);
    try std.testing.expect(result.info.flags.not_null);

    // Column not found
    try std.testing.expectError(error.ColumnNotFound, tc.catalog.findColumn("users", "nonexistent"));

    // Table not found
    try std.testing.expectError(CatalogError.TableNotFound, tc.catalog.findColumn("ghost", "id"));
}

test "Catalog multiple tables with constraints" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_multi.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    // Table with composite primary key
    try tc.catalog.createTable("orders", &.{
        .{ .name = "user_id", .column_type = .integer, .flags = .{ .not_null = true } },
        .{ .name = "product_id", .column_type = .integer, .flags = .{ .not_null = true } },
        .{ .name = "quantity", .column_type = .integer, .flags = .{} },
    }, &.{
        .{ .primary_key = &.{ "user_id", "product_id" } },
    }, 0);

    // Table with unique constraint
    try tc.catalog.createTable("products", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{ .primary_key = true, .not_null = true } },
        .{ .name = "sku", .column_type = .text, .flags = .{ .not_null = true } },
        .{ .name = "name", .column_type = .text, .flags = .{} },
    }, &.{
        .{ .unique = &.{"sku"} },
    }, 0);

    // Verify orders
    const orders = try tc.catalog.getTable("orders");
    defer orders.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 3), orders.columns.len);
    try std.testing.expectEqual(@as(usize, 1), orders.table_constraints.len);
    switch (orders.table_constraints[0]) {
        .primary_key => |cols| {
            try std.testing.expectEqual(@as(usize, 2), cols.len);
            try std.testing.expectEqualStrings("user_id", cols[0]);
            try std.testing.expectEqualStrings("product_id", cols[1]);
        },
        else => return error.TestUnexpectedResult,
    }

    // Verify products
    const products = try tc.catalog.getTable("products");
    defer products.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 3), products.columns.len);
    switch (products.table_constraints[0]) {
        .unique => |cols| {
            try std.testing.expectEqual(@as(usize, 1), cols.len);
            try std.testing.expectEqualStrings("sku", cols[0]);
        },
        else => return error.TestUnexpectedResult,
    }
}

test "Catalog drop and recreate table" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_recreate.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    // Create, drop, recreate with different schema
    try tc.catalog.createTable("t1", &.{
        .{ .name = "a", .column_type = .integer, .flags = .{} },
    }, &.{}, 0);

    try tc.catalog.dropTable("t1", false);

    try tc.catalog.createTable("t1", &.{
        .{ .name = "x", .column_type = .text, .flags = .{ .not_null = true } },
        .{ .name = "y", .column_type = .real, .flags = .{} },
    }, &.{}, 0);

    const table = try tc.catalog.getTable("t1");
    defer table.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 2), table.columns.len);
    try std.testing.expectEqualStrings("x", table.columns[0].name);
    try std.testing.expectEqual(ColumnType.text, table.columns[0].column_type);
    try std.testing.expectEqualStrings("y", table.columns[1].name);
    try std.testing.expectEqual(ColumnType.real, table.columns[1].column_type);
}

// ── View Catalog Tests ──────────────────────────────────────────────────

test "Catalog createView and getView" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_view_create.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    try tc.catalog.createView("v1", "SELECT * FROM t1", false, false, &.{ "a", "b" }, 0);

    const info = try tc.catalog.getView("v1");
    defer info.deinit();

    try std.testing.expectEqualStrings("v1", info.name);
    try std.testing.expectEqualStrings("SELECT * FROM t1", info.sql);
    try std.testing.expectEqual(@as(usize, 2), info.column_names.len);
    try std.testing.expectEqualStrings("a", info.column_names[0]);
    try std.testing.expectEqualStrings("b", info.column_names[1]);
}

test "Catalog createView with no column names" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_view_nocols.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    try tc.catalog.createView("v1", "SELECT 1", false, false, &.{}, 0);

    const info = try tc.catalog.getView("v1");
    defer info.deinit();

    try std.testing.expectEqual(@as(usize, 0), info.column_names.len);
    try std.testing.expectEqualStrings("SELECT 1", info.sql);
}

test "Catalog createView duplicate error" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_view_dup.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    try tc.catalog.createView("v1", "SELECT 1", false, false, &.{}, 0);
    try std.testing.expectError(CatalogError.ViewAlreadyExists, tc.catalog.createView("v1", "SELECT 2", false, false, &.{}, 0));
}

test "Catalog createView OR REPLACE overwrites" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_view_replace.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    try tc.catalog.createView("v1", "SELECT 1", false, false, &.{}, 0);
    try tc.catalog.createView("v1", "SELECT 2", true, false, &.{}, 0);

    const info = try tc.catalog.getView("v1");
    defer info.deinit();
    try std.testing.expectEqualStrings("SELECT 2", info.sql);
}

test "Catalog createView IF NOT EXISTS skips existing" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_view_ifne.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    try tc.catalog.createView("v1", "SELECT 1", false, false, &.{}, 0);
    try tc.catalog.createView("v1", "SELECT 2", false, true, &.{}, 0);

    const info = try tc.catalog.getView("v1");
    defer info.deinit();
    try std.testing.expectEqualStrings("SELECT 1", info.sql); // original preserved
}

test "Catalog createView name collision with table" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_view_collision.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    try tc.catalog.createTable("t1", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{} },
    }, &.{}, 0);

    try std.testing.expectError(CatalogError.TableAlreadyExists, tc.catalog.createView("t1", "SELECT 1", false, false, &.{}, 0));
}

test "Catalog dropView" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_view_drop.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    try tc.catalog.createView("v1", "SELECT 1", false, false, &.{}, 0);
    try std.testing.expect(try tc.catalog.viewExists("v1"));

    try tc.catalog.dropView("v1", false);
    try std.testing.expect(!try tc.catalog.viewExists("v1"));
}

test "Catalog dropView nonexistent error" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_view_drop_ne.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    try std.testing.expectError(CatalogError.ViewNotFound, tc.catalog.dropView("ghost", false));
    try tc.catalog.dropView("ghost", true); // IF EXISTS — no error
}

test "Catalog viewExists" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_view_exists.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    try std.testing.expect(!try tc.catalog.viewExists("v1"));
    try tc.catalog.createView("v1", "SELECT 1", false, false, &.{}, 0);
    try std.testing.expect(try tc.catalog.viewExists("v1"));
}

test "Catalog getView not found" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_view_get_nf.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    try std.testing.expectError(CatalogError.ViewNotFound, tc.catalog.getView("ghost"));
}

test "Catalog listViews" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_view_list.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    // No views initially
    const empty = try tc.catalog.listViews(allocator);
    defer allocator.free(empty);
    try std.testing.expectEqual(@as(usize, 0), empty.len);

    // Create views
    try tc.catalog.createView("alpha_view", "SELECT 1", false, false, &.{}, 0);
    try tc.catalog.createView("beta_view", "SELECT 2", false, false, &.{}, 0);

    const views = try tc.catalog.listViews(allocator);
    defer {
        for (views) |v| allocator.free(v);
        allocator.free(views);
    }

    try std.testing.expectEqual(@as(usize, 2), views.len);
}

test "Catalog listViews does not include tables" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_view_list_notbl.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    try tc.catalog.createTable("t1", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{} },
    }, &.{}, 0);
    try tc.catalog.createView("v1", "SELECT 1", false, false, &.{}, 0);

    const views = try tc.catalog.listViews(allocator);
    defer {
        for (views) |v| allocator.free(v);
        allocator.free(views);
    }

    try std.testing.expectEqual(@as(usize, 1), views.len);
    try std.testing.expectEqualStrings("v1", views[0]);

    // Tables list should not include views
    const tables = try tc.catalog.listTables(allocator);
    defer {
        for (tables) |t| allocator.free(t);
        allocator.free(tables);
    }

    try std.testing.expectEqual(@as(usize, 1), tables.len);
    try std.testing.expectEqualStrings("t1", tables[0]);
}

test "Catalog view serialization roundtrip with many columns" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_view_serial.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    const col_names = [_][]const u8{ "col_a", "col_b", "col_c", "col_d", "col_e" };
    const sql = "SELECT col_a, col_b, col_c, col_d, col_e FROM wide_table";
    try tc.catalog.createView("wide_view", sql, false, false, &col_names, 0);

    const info = try tc.catalog.getView("wide_view");
    defer info.deinit();

    try std.testing.expectEqual(@as(usize, 5), info.column_names.len);
    for (col_names, 0..) |expected, i| {
        try std.testing.expectEqualStrings(expected, info.column_names[i]);
    }
    try std.testing.expectEqualStrings(sql, info.sql);
}

// ── ENUM Type Tests ─────────────────────────────────────────────────────

test "Catalog createEnumType and getEnumType" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_enum_create.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    const values = [_][]const u8{ "'happy'", "'sad'", "'neutral'" };
    try tc.catalog.createEnumType("mood", &values);

    const info = try tc.catalog.getEnumType("mood");
    defer info.deinit();

    try std.testing.expectEqualStrings("mood", info.name);
    try std.testing.expectEqual(@as(usize, 3), info.values.len);
    try std.testing.expectEqualStrings("'happy'", info.values[0]);
    try std.testing.expectEqualStrings("'sad'", info.values[1]);
    try std.testing.expectEqualStrings("'neutral'", info.values[2]);
}

test "Catalog createEnumType duplicate error" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_enum_dup.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    const values = [_][]const u8{"'active'"};
    try tc.catalog.createEnumType("status", &values);
    try std.testing.expectError(CatalogError.TypeAlreadyExists, tc.catalog.createEnumType("status", &values));
}

test "Catalog dropEnumType" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_enum_drop.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    const values = [_][]const u8{"'active'"};
    try tc.catalog.createEnumType("status", &values);
    try std.testing.expect(try tc.catalog.enumTypeExists("status"));

    try tc.catalog.dropEnumType("status", false);
    try std.testing.expect(!try tc.catalog.enumTypeExists("status"));
}

test "Catalog dropEnumType nonexistent error" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_enum_drop_ne.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    try std.testing.expectError(CatalogError.TypeNotFound, tc.catalog.dropEnumType("ghost", false));
    try tc.catalog.dropEnumType("ghost", true); // IF EXISTS — no error
}

test "Catalog enumTypeExists" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_enum_exists.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    try std.testing.expect(!try tc.catalog.enumTypeExists("mood"));

    const values = [_][]const u8{"'happy'"};
    try tc.catalog.createEnumType("mood", &values);
    try std.testing.expect(try tc.catalog.enumTypeExists("mood"));
}

test "Catalog listEnumTypes" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_enum_list.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    // No types initially
    const empty = try tc.catalog.listEnumTypes(allocator);
    defer allocator.free(empty);
    try std.testing.expectEqual(@as(usize, 0), empty.len);

    // Create types
    const vals1 = [_][]const u8{"'happy'"};
    const vals2 = [_][]const u8{"'active'"};
    try tc.catalog.createEnumType("mood", &vals1);
    try tc.catalog.createEnumType("status", &vals2);

    const types = try tc.catalog.listEnumTypes(allocator);
    defer {
        for (types) |t| allocator.free(t);
        allocator.free(types);
    }

    try std.testing.expectEqual(@as(usize, 2), types.len);
}

test "Catalog createEnumType name collision with table" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_enum_collision.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    try tc.catalog.createTable("t1", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{} },
    }, &.{}, 0);

    const values = [_][]const u8{"'active'"};
    try std.testing.expectError(CatalogError.TableAlreadyExists, tc.catalog.createEnumType("t1", &values));
}

test "Catalog createEnumType with empty values array" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_enum_empty.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    const empty_values: []const []const u8 = &.{};
    try tc.catalog.createEnumType("empty_enum", empty_values);

    const info = try tc.catalog.getEnumType("empty_enum");
    defer info.deinit();

    try std.testing.expectEqual(@as(usize, 0), info.values.len);
}

test "Catalog createEnumType with single value" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_enum_single.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    const values = [_][]const u8{"only"};
    try tc.catalog.createEnumType("single_enum", &values);

    const info = try tc.catalog.getEnumType("single_enum");
    defer info.deinit();

    try std.testing.expectEqual(@as(usize, 1), info.values.len);
    try std.testing.expectEqualStrings("only", info.values[0]);
}

test "Catalog createEnumType with many values" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_enum_many.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    // Create 100 enum values
    var values_buf: [100][]const u8 = undefined;
    var owned_values = std.ArrayListUnmanaged([]const u8){};
    defer {
        for (owned_values.items) |v| allocator.free(v);
        owned_values.deinit(allocator);
    }

    for (0..100) |i| {
        const s = try std.fmt.allocPrint(allocator, "{d}", .{i});
        try owned_values.append(allocator, s);
        values_buf[i] = s;
    }

    try tc.catalog.createEnumType("many_enum", &values_buf);

    const info = try tc.catalog.getEnumType("many_enum");
    defer info.deinit();

    try std.testing.expectEqual(@as(usize, 100), info.values.len);
    try std.testing.expectEqualStrings("0", info.values[0]);
    try std.testing.expectEqualStrings("99", info.values[99]);
}

test "Catalog createEnumType with long value names" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_enum_long.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    // Create a value that's 1000 characters long
    const long_value = try allocator.alloc(u8, 1000);
    defer allocator.free(long_value);
    @memset(long_value, 'a');

    const values = [_][]const u8{long_value};
    try tc.catalog.createEnumType("long_enum", &values);

    const info = try tc.catalog.getEnumType("long_enum");
    defer info.deinit();

    try std.testing.expectEqual(@as(usize, 1), info.values.len);
    try std.testing.expectEqual(@as(usize, 1000), info.values[0].len);
    try std.testing.expect(std.mem.allEqual(u8, info.values[0], 'a'));
}

test "Catalog createEnumType with duplicate values" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_enum_dup_vals.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    // Duplicate values should be allowed (validation is semantic, not at storage level)
    const values = [_][]const u8{ "active", "active", "inactive" };
    try tc.catalog.createEnumType("dup_enum", &values);

    const info = try tc.catalog.getEnumType("dup_enum");
    defer info.deinit();

    try std.testing.expectEqual(@as(usize, 3), info.values.len);
    try std.testing.expectEqualStrings("active", info.values[0]);
    try std.testing.expectEqualStrings("active", info.values[1]);
    try std.testing.expectEqualStrings("inactive", info.values[2]);
}

test "Catalog createEnumType with empty string value" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_enum_empty_str.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    const values = [_][]const u8{ "", "active", "" };
    try tc.catalog.createEnumType("empty_str_enum", &values);

    const info = try tc.catalog.getEnumType("empty_str_enum");
    defer info.deinit();

    try std.testing.expectEqual(@as(usize, 3), info.values.len);
    try std.testing.expectEqualStrings("", info.values[0]);
    try std.testing.expectEqualStrings("active", info.values[1]);
    try std.testing.expectEqualStrings("", info.values[2]);
}

test "Catalog createEnumType with special characters" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_enum_special.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    const values = [_][]const u8{ "hello\nworld", "tab\there", "emoji😀", "quote\"here", "null\x00byte" };
    try tc.catalog.createEnumType("special_enum", &values);

    const info = try tc.catalog.getEnumType("special_enum");
    defer info.deinit();

    try std.testing.expectEqual(@as(usize, 5), info.values.len);
    try std.testing.expectEqualStrings("hello\nworld", info.values[0]);
    try std.testing.expectEqualStrings("tab\there", info.values[1]);
    try std.testing.expectEqualStrings("emoji😀", info.values[2]);
    try std.testing.expectEqualStrings("quote\"here", info.values[3]);
    try std.testing.expectEqualStrings("null\x00byte", info.values[4]);
}

test "Catalog getEnumType with corrupted data - truncated count" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_enum_corrupt1.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    // Manually insert corrupted data with only 1 byte (should be at least 2)
    const key = try tc.catalog.makeTypeKey("corrupt1");
    defer allocator.free(key);

    const bad_data = [_]u8{0x01}; // Only 1 byte
    try tc.catalog.tree.insert(key, &bad_data);

    try std.testing.expectError(CatalogError.InvalidSchemaData, tc.catalog.getEnumType("corrupt1"));
}

test "Catalog getEnumType with corrupted data - truncated value length" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_enum_corrupt2.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    // Create data with value_count=1 but no value length field
    const key = try tc.catalog.makeTypeKey("corrupt2");
    defer allocator.free(key);

    const bad_data = [_]u8{ 0x01, 0x00 }; // count=1, but missing value length+data
    try tc.catalog.tree.insert(key, &bad_data);

    try std.testing.expectError(CatalogError.InvalidSchemaData, tc.catalog.getEnumType("corrupt2"));
}

test "Catalog getEnumType with corrupted data - truncated value data" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_enum_corrupt3.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    // Create data with value_count=1, length=10, but only 5 bytes of data
    const key = try tc.catalog.makeTypeKey("corrupt3");
    defer allocator.free(key);

    const bad_data = [_]u8{ 0x01, 0x00, 0x0A, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05 }; // count=1, len=10, but only 5 bytes
    try tc.catalog.tree.insert(key, &bad_data);

    try std.testing.expectError(CatalogError.InvalidSchemaData, tc.catalog.getEnumType("corrupt3"));
}

test "Catalog listEnumTypes excludes tables and views" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_enum_list_filter.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    // Create table and view
    try tc.catalog.createTable("my_table", &.{
        .{ .name = "id", .column_type = .integer, .flags = .{} },
    }, &.{}, 0);

    const empty_cols: []const []const u8 = &.{};
    try tc.catalog.createView("my_view", "SELECT 1", false, false, empty_cols, 0);

    // Create enum types
    const values1 = [_][]const u8{ "a", "b" };
    try tc.catalog.createEnumType("enum1", &values1);

    const values2 = [_][]const u8{ "x", "y" };
    try tc.catalog.createEnumType("enum2", &values2);

    const types = try tc.catalog.listEnumTypes(allocator);
    defer {
        for (types) |t| allocator.free(t);
        allocator.free(types);
    }

    // Should only list the 2 enum types, not table or view
    try std.testing.expectEqual(@as(usize, 2), types.len);

    // Check that returned names are enum types
    const has_enum1 = std.mem.eql(u8, types[0], "enum1") or std.mem.eql(u8, types[1], "enum1");
    const has_enum2 = std.mem.eql(u8, types[0], "enum2") or std.mem.eql(u8, types[1], "enum2");
    try std.testing.expect(has_enum1);
    try std.testing.expect(has_enum2);
}

test "Catalog createDomain basic" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_domain_basic.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    try tc.catalog.createDomain("pos_int", .type_integer, "VALUE > 0");

    const info = try tc.catalog.getDomain("pos_int");
    defer info.deinit();

    try std.testing.expectEqualStrings("pos_int", info.name);
    try std.testing.expectEqual(ast.DataType.type_integer, info.base_type);
    try std.testing.expect(info.constraint != null);
    try std.testing.expectEqualStrings("VALUE > 0", info.constraint.?);
}

test "Catalog createDomain without constraint" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_domain_no_constraint.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    try tc.catalog.createDomain("email", .type_text, null);

    const info = try tc.catalog.getDomain("email");
    defer info.deinit();

    try std.testing.expectEqualStrings("email", info.name);
    try std.testing.expectEqual(ast.DataType.type_text, info.base_type);
    try std.testing.expect(info.constraint == null);
}

test "Catalog createDomain duplicate error" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_domain_duplicate.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    try tc.catalog.createDomain("my_domain", .type_integer, null);
    try std.testing.expectError(CatalogError.TypeAlreadyExists, tc.catalog.createDomain("my_domain", .type_text, null));
}

test "Catalog createDomain conflicts with table" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_domain_table_conflict.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    const cols = [_]ColumnInfo{.{ .name = "id", .column_type = .integer, .flags = .{} }};
    try tc.catalog.createTable("users", &cols, &[_]TableConstraintInfo{}, 0);

    try std.testing.expectError(CatalogError.TableAlreadyExists, tc.catalog.createDomain("users", .type_integer, null));
}

test "Catalog createDomain conflicts with enum" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_domain_enum_conflict.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    const values = [_][]const u8{ "a", "b" };
    try tc.catalog.createEnumType("status", &values);

    try std.testing.expectError(CatalogError.TypeAlreadyExists, tc.catalog.createDomain("status", .type_text, null));
}

test "Catalog dropDomain basic" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_domain_drop.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    try tc.catalog.createDomain("temp_domain", .type_integer, null);
    try std.testing.expect(try tc.catalog.domainExists("temp_domain"));

    try tc.catalog.dropDomain("temp_domain", false);
    try std.testing.expect(!try tc.catalog.domainExists("temp_domain"));
}

test "Catalog dropDomain if_exists" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_domain_drop_if_exists.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    // Should not error when if_exists=true and domain doesn't exist
    try tc.catalog.dropDomain("nonexistent", true);

    // Should error when if_exists=false and domain doesn't exist
    try std.testing.expectError(CatalogError.TypeNotFound, tc.catalog.dropDomain("nonexistent", false));
}

test "Catalog domainExists" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_domain_exists.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    try std.testing.expect(!try tc.catalog.domainExists("my_domain"));

    try tc.catalog.createDomain("my_domain", .type_integer, null);
    try std.testing.expect(try tc.catalog.domainExists("my_domain"));

    try tc.catalog.dropDomain("my_domain", false);
    try std.testing.expect(!try tc.catalog.domainExists("my_domain"));
}

test "Catalog getDomain not found" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_domain_not_found.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    try std.testing.expectError(CatalogError.TypeNotFound, tc.catalog.getDomain("nonexistent"));
}

test "Catalog getDomain with corrupted data" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_domain_corrupt.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    // Manually insert corrupted data with only 3 bytes (should be at least 4)
    const key = try tc.catalog.makeDomainKey("corrupt");
    defer allocator.free(key);

    const bad_data = [_]u8{ 0x01, 0x00, 0x01 }; // base_type, has_constraint, missing constraint_len
    try tc.catalog.tree.insert(key, &bad_data);

    try std.testing.expectError(CatalogError.InvalidSchemaData, tc.catalog.getDomain("corrupt"));
}

test "Catalog getDomain with constraint length overflow" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_domain_corrupt_len.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    const key = try tc.catalog.makeDomainKey("corrupt_len");
    defer allocator.free(key);

    // [base_type: 1][has_constraint: 1][constraint_len: 100 (but no data)]
    const bad_data = [_]u8{ 0x01, 0x01, 0x64, 0x00 }; // constraint_len=100 but no data
    try tc.catalog.tree.insert(key, &bad_data);

    try std.testing.expectError(CatalogError.InvalidSchemaData, tc.catalog.getDomain("corrupt_len"));
}

// ── Function Catalog Tests ──────────────────────────────────────────────

test "Catalog createFunction and getFunction — scalar return" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_func_scalar.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    const stmt = ast.CreateFunctionStmt{
        .name = "add_one",
        .parameters = &.{
            .{ .name = "x", .data_type = .type_integer },
        },
        .return_type = .{ .scalar = .type_integer },
        .language = "sfl",
        .body = "RETURN x + 1;",
        .volatility = .immutable,
        .or_replace = false,
    };

    try tc.catalog.createFunction(stmt);

    const info = try tc.catalog.getFunction("add_one");
    defer info.deinit();

    try std.testing.expectEqualStrings("add_one", info.name);
    try std.testing.expectEqual(@as(usize, 1), info.parameters.len);
    try std.testing.expectEqualStrings("x", info.parameters[0].name);
    try std.testing.expectEqual(ast.DataType.type_integer, info.parameters[0].data_type);
    try std.testing.expectEqual(ast.FunctionVolatility.immutable, info.volatility);
    try std.testing.expectEqualStrings("sfl", info.language);
    try std.testing.expectEqualStrings("RETURN x + 1;", info.body);

    switch (info.return_type) {
        .scalar => |dt| try std.testing.expectEqual(ast.DataType.type_integer, dt),
        else => try std.testing.expect(false),
    }
}

test "Catalog createFunction — RETURNS TABLE" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_func_table.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    const cols = [_]ast.ColumnDef{
        .{ .name = "id", .data_type = .type_integer, .constraints = &.{} },
        .{ .name = "name", .data_type = .type_text, .constraints = &.{} },
    };

    const stmt = ast.CreateFunctionStmt{
        .name = "get_users",
        .parameters = &.{},
        .return_type = .{ .table = &cols },
        .language = "sfl",
        .body = "SELECT id, name FROM users;",
        .volatility = .stable,
        .or_replace = false,
    };

    try tc.catalog.createFunction(stmt);

    const info = try tc.catalog.getFunction("get_users");
    defer info.deinit();

    try std.testing.expectEqualStrings("get_users", info.name);
    try std.testing.expectEqual(@as(usize, 0), info.parameters.len);
    try std.testing.expectEqual(ast.FunctionVolatility.stable, info.volatility);

    switch (info.return_type) {
        .table => |tcols| {
            try std.testing.expectEqual(@as(usize, 2), tcols.len);
            try std.testing.expectEqualStrings("id", tcols[0].name);
            try std.testing.expectEqual(ast.DataType.type_integer, tcols[0].data_type);
            try std.testing.expectEqualStrings("name", tcols[1].name);
            try std.testing.expectEqual(ast.DataType.type_text, tcols[1].data_type);
        },
        else => try std.testing.expect(false),
    }
}

test "Catalog createFunction — RETURNS SETOF" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_func_setof.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    const stmt = ast.CreateFunctionStmt{
        .name = "generate_series",
        .parameters = &.{
            .{ .name = "start", .data_type = .type_integer },
            .{ .name = "stop", .data_type = .type_integer },
        },
        .return_type = .{ .setof = .type_integer },
        .language = "sfl",
        .body = "RETURN QUERY SELECT x FROM generate(start, stop);",
        .volatility = .immutable,
        .or_replace = false,
    };

    try tc.catalog.createFunction(stmt);

    const info = try tc.catalog.getFunction("generate_series");
    defer info.deinit();

    try std.testing.expectEqualStrings("generate_series", info.name);
    try std.testing.expectEqual(@as(usize, 2), info.parameters.len);

    switch (info.return_type) {
        .setof => |dt| try std.testing.expectEqual(ast.DataType.type_integer, dt),
        else => try std.testing.expect(false),
    }
}

test "Catalog createFunction — OR REPLACE" {

    const allocator = std.testing.allocator;
    const path = "test_catalog_func_replace.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    const stmt1 = ast.CreateFunctionStmt{
        .name = "my_func",
        .parameters = &.{},
        .return_type = .{ .scalar = .type_integer },
        .language = "sfl",
        .body = "RETURN 1;",
        .volatility = .immutable,
        .or_replace = false,
    };

    try tc.catalog.createFunction(stmt1);

    // Try to create again without OR REPLACE — should fail
    try std.testing.expectError(CatalogError.TableAlreadyExists, tc.catalog.createFunction(stmt1));

    // Create with OR REPLACE — should succeed
    const stmt2 = ast.CreateFunctionStmt{
        .name = "my_func",
        .parameters = &.{},
        .return_type = .{ .scalar = .type_integer },
        .language = "sfl",
        .body = "RETURN 2;",
        .volatility = .stable,
        .or_replace = true,
    };

    try tc.catalog.createFunction(stmt2);

    const info = try tc.catalog.getFunction("my_func");
    defer info.deinit();

    try std.testing.expectEqualStrings("RETURN 2;", info.body);
    try std.testing.expectEqual(ast.FunctionVolatility.stable, info.volatility);
}

test "Catalog dropFunction basic" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_func_drop.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    const stmt = ast.CreateFunctionStmt{
        .name = "temp_func",
        .parameters = &.{},
        .return_type = .{ .scalar = .type_integer },
        .language = "sfl",
        .body = "RETURN 42;",
        .volatility = .immutable,
        .or_replace = false,
    };

    try tc.catalog.createFunction(stmt);
    try std.testing.expect(try tc.catalog.functionExists("temp_func"));

    try tc.catalog.dropFunction("temp_func", false);
    try std.testing.expect(!try tc.catalog.functionExists("temp_func"));
}

test "Catalog dropFunction — not exists error" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_func_drop_notfound.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    try std.testing.expectError(CatalogError.TypeNotFound, tc.catalog.dropFunction("nonexistent", false));
}

test "Catalog dropFunction IF EXISTS" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_func_drop_ifexists.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    // Should not error when IF EXISTS is true
    try tc.catalog.dropFunction("nonexistent", true);
}

test "Catalog functionExists" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_func_exists.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    try std.testing.expect(!try tc.catalog.functionExists("my_func"));

    const stmt = ast.CreateFunctionStmt{
        .name = "my_func",
        .parameters = &.{},
        .return_type = .{ .scalar = .type_integer },
        .language = "sfl",
        .body = "RETURN 1;",
        .volatility = .immutable,
        .or_replace = false,
    };

    try tc.catalog.createFunction(stmt);
    try std.testing.expect(try tc.catalog.functionExists("my_func"));
}

test "Catalog listFunctions" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_func_list.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    const stmt1 = ast.CreateFunctionStmt{
        .name = "func1",
        .parameters = &.{},
        .return_type = .{ .scalar = .type_integer },
        .language = "sfl",
        .body = "RETURN 1;",
        .volatility = .immutable,
        .or_replace = false,
    };

    const stmt2 = ast.CreateFunctionStmt{
        .name = "func2",
        .parameters = &.{},
        .return_type = .{ .scalar = .type_text },
        .language = "sfl",
        .body = "RETURN 'hello';",
        .volatility = .stable,
        .or_replace = false,
    };

    try tc.catalog.createFunction(stmt1);
    try tc.catalog.createFunction(stmt2);

    const names = try tc.catalog.listFunctions(allocator);
    defer {
        for (names) |n| allocator.free(n);
        allocator.free(names);
    }

    try std.testing.expectEqual(@as(usize, 2), names.len);
    // Names should be in some order — just check both exist
    var found_func1 = false;
    var found_func2 = false;
    for (names) |n| {
        if (std.mem.eql(u8, n, "func1")) found_func1 = true;
        if (std.mem.eql(u8, n, "func2")) found_func2 = true;
    }
    try std.testing.expect(found_func1);
    try std.testing.expect(found_func2);
}

test "Catalog createFunction — multiple parameters" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_func_multi_param.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    const params = [_]ast.FunctionParam{
        .{ .name = "a", .data_type = .type_integer },
        .{ .name = "b", .data_type = .type_real },
        .{ .name = "c", .data_type = .type_text },
    };

    const stmt = ast.CreateFunctionStmt{
        .name = "multi_param",
        .parameters = &params,
        .return_type = .{ .scalar = .type_text },
        .language = "sfl",
        .body = "RETURN c || ' ' || CAST(a + b AS TEXT);",
        .volatility = .immutable,
        .or_replace = false,
    };

    try tc.catalog.createFunction(stmt);

    const info = try tc.catalog.getFunction("multi_param");
    defer info.deinit();

    try std.testing.expectEqual(@as(usize, 3), info.parameters.len);
    try std.testing.expectEqualStrings("a", info.parameters[0].name);
    try std.testing.expectEqual(ast.DataType.type_integer, info.parameters[0].data_type);
    try std.testing.expectEqualStrings("b", info.parameters[1].name);
    try std.testing.expectEqual(ast.DataType.type_real, info.parameters[1].data_type);
    try std.testing.expectEqualStrings("c", info.parameters[2].name);
    try std.testing.expectEqual(ast.DataType.type_text, info.parameters[2].data_type);
}

test "Catalog createFunction — empty parameter list" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_func_no_param.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    const stmt = ast.CreateFunctionStmt{
        .name = "no_params",
        .parameters = &.{},
        .return_type = .{ .scalar = .type_timestamp },
        .language = "sfl",
        .body = "RETURN CURRENT_TIMESTAMP();",
        .volatility = .vol,
        .or_replace = false,
    };

    try tc.catalog.createFunction(stmt);

    const info = try tc.catalog.getFunction("no_params");
    defer info.deinit();

    try std.testing.expectEqual(@as(usize, 0), info.parameters.len);
    try std.testing.expectEqual(ast.FunctionVolatility.vol, info.volatility);
}

test "Catalog getFunction — not found" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_func_notfound.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    try std.testing.expectError(CatalogError.TypeNotFound, tc.catalog.getFunction("nonexistent"));
}

test "Catalog createFunction — large body text" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_func_large_body.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    var body_buf: [1000]u8 = undefined;
    for (&body_buf) |*b| b.* = 'x';
    const large_body = body_buf[0..];

    const stmt = ast.CreateFunctionStmt{
        .name = "large_func",
        .parameters = &.{},
        .return_type = .{ .scalar = .type_text },
        .language = "sfl",
        .body = large_body,
        .volatility = .immutable,
        .or_replace = false,
    };

    try tc.catalog.createFunction(stmt);

    const info = try tc.catalog.getFunction("large_func");
    defer info.deinit();

    try std.testing.expectEqual(@as(usize, 1000), info.body.len);
    for (info.body) |b| try std.testing.expectEqual(@as(u8, 'x'), b);
}

test "Catalog createFunction — all volatility types" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_func_volatility.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    const stmt_immutable = ast.CreateFunctionStmt{
        .name = "immut_func",
        .parameters = &.{},
        .return_type = .{ .scalar = .type_integer },
        .language = "sfl",
        .body = "RETURN 1;",
        .volatility = .immutable,
        .or_replace = false,
    };

    const stmt_stable = ast.CreateFunctionStmt{
        .name = "stable_func",
        .parameters = &.{},
        .return_type = .{ .scalar = .type_timestamp },
        .language = "sfl",
        .body = "RETURN CURRENT_TIMESTAMP();",
        .volatility = .stable,
        .or_replace = false,
    };

    const stmt_volatile = ast.CreateFunctionStmt{
        .name = "vol_func",
        .parameters = &.{},
        .return_type = .{ .scalar = .type_integer },
        .language = "sfl",
        .body = "INSERT INTO log VALUES (1); RETURN 1;",
        .volatility = .vol,
        .or_replace = false,
    };

    try tc.catalog.createFunction(stmt_immutable);
    try tc.catalog.createFunction(stmt_stable);
    try tc.catalog.createFunction(stmt_volatile);

    const info1 = try tc.catalog.getFunction("immut_func");
    defer info1.deinit();
    try std.testing.expectEqual(ast.FunctionVolatility.immutable, info1.volatility);

    const info2 = try tc.catalog.getFunction("stable_func");
    defer info2.deinit();
    try std.testing.expectEqual(ast.FunctionVolatility.stable, info2.volatility);

    const info3 = try tc.catalog.getFunction("vol_func");
    defer info3.deinit();
    try std.testing.expectEqual(ast.FunctionVolatility.vol, info3.volatility);
}

test "Catalog createTrigger — basic AFTER INSERT trigger" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_trig_basic.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    const stmt = ast.CreateTriggerStmt{
        .name = "audit_log",
        .table_name = "users",
        .timing = .after,
        .event = .insert,
        .update_columns = &.{},
        .level = .row,
        .when_condition = null,
        .body = "INSERT INTO audit VALUES (NEW.id)",
        .or_replace = false,
    };

    try tc.catalog.createTrigger(stmt);

    const info = try tc.catalog.getTrigger("audit_log");
    defer info.deinit();

    try std.testing.expectEqualStrings("audit_log", info.name);
    try std.testing.expectEqualStrings("users", info.table_name);
    try std.testing.expectEqual(ast.TriggerTiming.after, info.timing);
    try std.testing.expectEqual(ast.TriggerEvent.insert, info.event);
    try std.testing.expectEqual(ast.TriggerLevel.row, info.level);
    try std.testing.expectEqualStrings("INSERT INTO audit VALUES (NEW.id)", info.body);
    try std.testing.expect(info.enabled);
    try std.testing.expectEqual(@as(usize, 0), info.update_columns.len);
    try std.testing.expect(info.when_condition == null);
}

test "Catalog createTrigger — with UPDATE OF columns" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_trig_update_of.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    const stmt = ast.CreateTriggerStmt{
        .name = "validate_email",
        .table_name = "users",
        .timing = .before,
        .event = .update,
        .update_columns = &.{ "email", "name" },
        .level = .row,
        .when_condition = null,
        .body = "SELECT check_email(NEW.email)",
        .or_replace = false,
    };

    try tc.catalog.createTrigger(stmt);

    const info = try tc.catalog.getTrigger("validate_email");
    defer info.deinit();

    try std.testing.expectEqualStrings("validate_email", info.name);
    try std.testing.expectEqual(ast.TriggerTiming.before, info.timing);
    try std.testing.expectEqual(ast.TriggerEvent.update, info.event);
    try std.testing.expectEqual(@as(usize, 2), info.update_columns.len);
    try std.testing.expectEqualStrings("email", info.update_columns[0]);
    try std.testing.expectEqualStrings("name", info.update_columns[1]);
}

test "Catalog triggerExists" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_trig_exists.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    try std.testing.expect(!try tc.catalog.triggerExists("nonexistent"));

    const stmt = ast.CreateTriggerStmt{
        .name = "test_trig",
        .table_name = "test_table",
        .timing = .after,
        .event = .delete,
        .update_columns = &.{},
        .level = .statement,
        .when_condition = null,
        .body = "NOTIFY admin",
        .or_replace = false,
    };

    try tc.catalog.createTrigger(stmt);
    try std.testing.expect(try tc.catalog.triggerExists("test_trig"));
}

test "Catalog dropTrigger" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_trig_drop.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    const stmt = ast.CreateTriggerStmt{
        .name = "temp_trig",
        .table_name = "temp_table",
        .timing = .instead_of,
        .event = .insert,
        .update_columns = &.{},
        .level = .row,
        .when_condition = null,
        .body = "INSERT INTO real_table VALUES (NEW.*)",
        .or_replace = false,
    };

    try tc.catalog.createTrigger(stmt);
    try std.testing.expect(try tc.catalog.triggerExists("temp_trig"));

    try tc.catalog.dropTrigger("temp_trig", false);
    try std.testing.expect(!try tc.catalog.triggerExists("temp_trig"));
}

test "Catalog dropTrigger IF EXISTS" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_trig_drop_if_exists.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    // Should not error when trigger doesn't exist
    try tc.catalog.dropTrigger("nonexistent", true);

    // Should error when trigger doesn't exist without IF EXISTS
    try std.testing.expectError(CatalogError.TypeNotFound, tc.catalog.dropTrigger("nonexistent", false));
}

test "Catalog alterTrigger — ENABLE" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_trig_alter_enable.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    const stmt = ast.CreateTriggerStmt{
        .name = "check_trig",
        .table_name = "orders",
        .timing = .before,
        .event = .update,
        .update_columns = &.{},
        .level = .row,
        .when_condition = null,
        .body = "SELECT validate_order(NEW.id)",
        .or_replace = false,
    };

    try tc.catalog.createTrigger(stmt);

    // Initially enabled
    {
        const info = try tc.catalog.getTrigger("check_trig");
        defer info.deinit();
        try std.testing.expect(info.enabled);
    }

    // Disable it
    try tc.catalog.alterTrigger("check_trig", false);
    {
        const info = try tc.catalog.getTrigger("check_trig");
        defer info.deinit();
        try std.testing.expect(!info.enabled);
    }

    // Re-enable it
    try tc.catalog.alterTrigger("check_trig", true);
    {
        const info = try tc.catalog.getTrigger("check_trig");
        defer info.deinit();
        try std.testing.expect(info.enabled);
    }
}

test "Catalog listTriggers" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_trig_list.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    const stmt1 = ast.CreateTriggerStmt{
        .name = "trig_a",
        .table_name = "table_a",
        .timing = .after,
        .event = .insert,
        .update_columns = &.{},
        .level = .row,
        .when_condition = null,
        .body = "-- trigger a body",
        .or_replace = false,
    };

    const stmt2 = ast.CreateTriggerStmt{
        .name = "trig_b",
        .table_name = "table_b",
        .timing = .before,
        .event = .delete,
        .update_columns = &.{},
        .level = .statement,
        .when_condition = null,
        .body = "-- trigger b body",
        .or_replace = false,
    };

    try tc.catalog.createTrigger(stmt1);
    try tc.catalog.createTrigger(stmt2);

    const names = try tc.catalog.listTriggers(allocator);
    defer {
        for (names) |n| allocator.free(n);
        allocator.free(names);
    }

    try std.testing.expectEqual(@as(usize, 2), names.len);
    // Names might not be in insertion order due to B+Tree ordering
}

test "Catalog createTrigger — with WHEN condition" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_trig_when.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    // Create a simple literal expression for WHEN condition
    const when_expr = ast.Expr{
        .boolean_literal = true,
    };

    const stmt = ast.CreateTriggerStmt{
        .name = "check_status",
        .table_name = "users",
        .timing = .before,
        .event = .update,
        .update_columns = &.{},
        .level = .row,
        .when_condition = &when_expr,
        .body = "SELECT validate_status(NEW.status)",
        .or_replace = false,
    };

    try tc.catalog.createTrigger(stmt);

    const info = try tc.catalog.getTrigger("check_status");
    defer info.deinit();

    try std.testing.expectEqualStrings("check_status", info.name);
    try std.testing.expectEqualStrings("users", info.table_name);
    // NOTE: When condition is currently serialized as empty (length 0),
    // so it deserializes to null even though we passed a non-null when_condition.
    // This is expected behavior until Milestone 14E-14H implement trigger execution.
    try std.testing.expect(info.when_condition == null);
}

test "Catalog createTrigger — empty body edge case" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_trig_empty_body.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    const stmt = ast.CreateTriggerStmt{
        .name = "empty_trig",
        .table_name = "test_table",
        .timing = .after,
        .event = .insert,
        .update_columns = &.{},
        .level = .row,
        .when_condition = null,
        .body = "",
        .or_replace = false,
    };

    try tc.catalog.createTrigger(stmt);

    const info = try tc.catalog.getTrigger("empty_trig");
    defer info.deinit();

    try std.testing.expectEqualStrings("", info.body);
}

test "Catalog createTrigger — large body stress test" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_trig_large_body.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    // Create a large SQL body (4KB of text)
    const large_body = try allocator.alloc(u8, 4096);
    defer allocator.free(large_body);
    @memset(large_body, 'X');

    const stmt = ast.CreateTriggerStmt{
        .name = "large_trig",
        .table_name = "test_table",
        .timing = .after,
        .event = .insert,
        .update_columns = &.{},
        .level = .row,
        .when_condition = null,
        .body = large_body,
        .or_replace = false,
    };

    try tc.catalog.createTrigger(stmt);

    const info = try tc.catalog.getTrigger("large_trig");
    defer info.deinit();

    try std.testing.expectEqual(@as(usize, 4096), info.body.len);
    try std.testing.expect(std.mem.eql(u8, large_body, info.body));
}

test "Catalog createTrigger — multiple triggers on same table" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_trig_multiple_same_table.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    const stmt1 = ast.CreateTriggerStmt{
        .name = "before_insert_users",
        .table_name = "users",
        .timing = .before,
        .event = .insert,
        .update_columns = &.{},
        .level = .row,
        .when_condition = null,
        .body = "SELECT validate_user(NEW.*)",
        .or_replace = false,
    };

    const stmt2 = ast.CreateTriggerStmt{
        .name = "after_insert_users",
        .table_name = "users",
        .timing = .after,
        .event = .insert,
        .update_columns = &.{},
        .level = .row,
        .when_condition = null,
        .body = "INSERT INTO audit_log VALUES (NEW.id)",
        .or_replace = false,
    };

    const stmt3 = ast.CreateTriggerStmt{
        .name = "before_delete_users",
        .table_name = "users",
        .timing = .before,
        .event = .delete,
        .update_columns = &.{},
        .level = .row,
        .when_condition = null,
        .body = "SELECT archive_user(OLD.*)",
        .or_replace = false,
    };

    try tc.catalog.createTrigger(stmt1);
    try tc.catalog.createTrigger(stmt2);
    try tc.catalog.createTrigger(stmt3);

    // All triggers should exist independently
    try std.testing.expect(try tc.catalog.triggerExists("before_insert_users"));
    try std.testing.expect(try tc.catalog.triggerExists("after_insert_users"));
    try std.testing.expect(try tc.catalog.triggerExists("before_delete_users"));

    const names = try tc.catalog.listTriggers(allocator);
    defer {
        for (names) |n| allocator.free(n);
        allocator.free(names);
    }

    try std.testing.expectEqual(@as(usize, 3), names.len);
}

test "Catalog getTrigger — nonexistent trigger" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_trig_get_nonexistent.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    try std.testing.expectError(CatalogError.TypeNotFound, tc.catalog.getTrigger("does_not_exist"));
}
// ── Role Catalog Tests ──────────────────────────────────────────────────

test "Catalog createRole — basic role with defaults" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_role_basic.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    const stmt = ast.CreateRoleStmt{
        .name = "test_user",
        .options = .{},
        .or_replace = false,
    };

    try tc.catalog.createRole(stmt);

    const info = try tc.catalog.getRole("test_user");
    defer info.deinit();

    try std.testing.expectEqualStrings("test_user", info.name);
    try std.testing.expect(info.login); // default LOGIN
    try std.testing.expect(!info.superuser); // default NOSUPERUSER
    try std.testing.expect(!info.createdb); // default NOCREATEDB
    try std.testing.expect(!info.createrole); // default NOCREATEROLE
    try std.testing.expect(info.inherit); // default INHERIT
    try std.testing.expect(info.password == null);
    try std.testing.expect(info.valid_until == null);
}

test "Catalog createRole — all options" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_role_all_options.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    const stmt = ast.CreateRoleStmt{
        .name = "admin",
        .options = .{
            .login = true,
            .superuser = true,
            .createdb = true,
            .createrole = true,
            .inherit = false,
            .password = "secret123",
            .valid_until = "2025-12-31 23:59:59",
        },
        .or_replace = false,
    };

    try tc.catalog.createRole(stmt);

    const info = try tc.catalog.getRole("admin");
    defer info.deinit();

    try std.testing.expectEqualStrings("admin", info.name);
    try std.testing.expect(info.login);
    try std.testing.expect(info.superuser);
    try std.testing.expect(info.createdb);
    try std.testing.expect(info.createrole);
    try std.testing.expect(!info.inherit);
    try std.testing.expect(info.password != null);
    try std.testing.expectEqualStrings("secret123", info.password.?);
    try std.testing.expect(info.valid_until != null);
    try std.testing.expectEqualStrings("2025-12-31 23:59:59", info.valid_until.?);
}

test "Catalog createRole — NOLOGIN role" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_role_nologin.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    const stmt = ast.CreateRoleStmt{
        .name = "readonly",
        .options = .{
            .login = false,
        },
        .or_replace = false,
    };

    try tc.catalog.createRole(stmt);

    const info = try tc.catalog.getRole("readonly");
    defer info.deinit();

    try std.testing.expectEqualStrings("readonly", info.name);
    try std.testing.expect(!info.login);
}

test "Catalog createRole — OR REPLACE existing role" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_role_or_replace.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    // Create initial role
    const stmt1 = ast.CreateRoleStmt{
        .name = "test_user",
        .options = .{ .superuser = false },
        .or_replace = false,
    };
    try tc.catalog.createRole(stmt1);

    // Replace with OR REPLACE
    const stmt2 = ast.CreateRoleStmt{
        .name = "test_user",
        .options = .{ .superuser = true },
        .or_replace = true,
    };
    try tc.catalog.createRole(stmt2);

    const info = try tc.catalog.getRole("test_user");
    defer info.deinit();

    try std.testing.expect(info.superuser); // Updated value
}

test "Catalog createRole — duplicate role without OR REPLACE" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_role_duplicate.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    const stmt = ast.CreateRoleStmt{
        .name = "test_user",
        .options = .{},
        .or_replace = false,
    };
    try tc.catalog.createRole(stmt);

    // Attempt to create again without OR REPLACE should fail
    try std.testing.expectError(CatalogError.TableAlreadyExists, tc.catalog.createRole(stmt));
}

test "Catalog dropRole — existing role" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_role_drop.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    const stmt = ast.CreateRoleStmt{
        .name = "temp_user",
        .options = .{},
        .or_replace = false,
    };
    try tc.catalog.createRole(stmt);

    try tc.catalog.dropRole("temp_user", false);

    // Should not exist after drop
    try std.testing.expect(!(try tc.catalog.roleExists("temp_user")));
}

test "Catalog dropRole — IF EXISTS on nonexistent role" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_role_drop_if_exists.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    // Should succeed with IF EXISTS
    try tc.catalog.dropRole("nonexistent", true);

    // Should fail without IF EXISTS
    try std.testing.expectError(CatalogError.TypeNotFound, tc.catalog.dropRole("nonexistent", false));
}

test "Catalog roleExists" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_role_exists.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    const stmt = ast.CreateRoleStmt{
        .name = "existing_user",
        .options = .{},
        .or_replace = false,
    };
    try tc.catalog.createRole(stmt);

    try std.testing.expect(try tc.catalog.roleExists("existing_user"));
    try std.testing.expect(!(try tc.catalog.roleExists("nonexistent_user")));
}

test "Catalog alterRole — change options" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_role_alter.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    // Create initial role
    const stmt = ast.CreateRoleStmt{
        .name = "test_user",
        .options = .{
            .superuser = false,
            .createdb = false,
            .password = "old_password",
        },
        .or_replace = false,
    };
    try tc.catalog.createRole(stmt);

    // Alter role
    const alter_opts = ast.RoleOptions{
        .superuser = true,
        .createdb = true,
        .password = "new_password",
    };
    try tc.catalog.alterRole("test_user", alter_opts);

    // Verify changes
    const info = try tc.catalog.getRole("test_user");
    defer info.deinit();

    try std.testing.expect(info.superuser);
    try std.testing.expect(info.createdb);
    try std.testing.expectEqualStrings("new_password", info.password.?);
}

test "Catalog alterRole — partial update" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_role_alter_partial.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    // Create initial role with all options
    const stmt = ast.CreateRoleStmt{
        .name = "test_user",
        .options = .{
            .login = true,
            .superuser = false,
            .createdb = true,
            .password = "original",
        },
        .or_replace = false,
    };
    try tc.catalog.createRole(stmt);

    // Alter only superuser
    const alter_opts = ast.RoleOptions{
        .superuser = true,
    };
    try tc.catalog.alterRole("test_user", alter_opts);

    // Verify superuser changed, other fields unchanged
    const info = try tc.catalog.getRole("test_user");
    defer info.deinit();

    try std.testing.expect(info.login); // unchanged
    try std.testing.expect(info.superuser); // changed
    try std.testing.expect(info.createdb); // unchanged
    try std.testing.expectEqualStrings("original", info.password.?); // unchanged
}

test "Catalog alterRole — nonexistent role" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_role_alter_nonexistent.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    const alter_opts = ast.RoleOptions{ .superuser = true };
    try std.testing.expectError(CatalogError.TypeNotFound, tc.catalog.alterRole("nonexistent", alter_opts));
}

test "Catalog listRoles" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_role_list.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    // Create multiple roles
    const stmt1 = ast.CreateRoleStmt{ .name = "user1", .options = .{}, .or_replace = false };
    const stmt2 = ast.CreateRoleStmt{ .name = "user2", .options = .{}, .or_replace = false };
    const stmt3 = ast.CreateRoleStmt{ .name = "admin", .options = .{}, .or_replace = false };

    try tc.catalog.createRole(stmt1);
    try tc.catalog.createRole(stmt2);
    try tc.catalog.createRole(stmt3);

    // List roles
    const names = try tc.catalog.listRoles(allocator);
    defer {
        for (names) |n| allocator.free(n);
        allocator.free(names);
    }

    try std.testing.expectEqual(@as(usize, 3), names.len);

    // Names should be present (order not guaranteed)
    var found_user1 = false;
    var found_user2 = false;
    var found_admin = false;
    for (names) |name| {
        if (std.mem.eql(u8, name, "user1")) found_user1 = true;
        if (std.mem.eql(u8, name, "user2")) found_user2 = true;
        if (std.mem.eql(u8, name, "admin")) found_admin = true;
    }
    try std.testing.expect(found_user1);
    try std.testing.expect(found_user2);
    try std.testing.expect(found_admin);
}

test "Catalog getRole — nonexistent role" {
    const allocator = std.testing.allocator;
    const path = "test_catalog_role_get_nonexistent.db";

    var tc = try TestCatalog.setup(allocator, path);
    defer tc.teardown(allocator);

    try std.testing.expectError(CatalogError.TypeNotFound, tc.catalog.getRole("does_not_exist"));
}
