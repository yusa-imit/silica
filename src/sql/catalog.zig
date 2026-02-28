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
            columns[i] = .{
                .name = col_def.name,
                .column_type = columnTypeFromAst(col_def.data_type),
                .flags = constraintFlagsFromAst(col_def.constraints),
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
            const name_copy = try allocator.dupe(u8, entry.key);
            errdefer allocator.free(name_copy);
            defer allocator.free(entry.key);
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
