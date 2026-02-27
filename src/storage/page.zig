//! Page Manager (Pager) — manages fixed-size pages on disk.
//!
//! The database file is a sequence of fixed-size pages. Page 0 contains the
//! database header. Each page has a type tag and a CRC32C checksum for
//! integrity verification. Free pages are managed as a linked list.
//!
//! File layout:
//!   Page 0:    Database header (magic "SLCA", version, page size, etc.)
//!   Page 1:    Schema table root (reserved for future B+Tree)
//!   Page 2..N: Data, index, overflow, and freelist pages

const std = @import("std");
const checksum = @import("../util/checksum.zig");

// ── Constants ──────────────────────────────────────────────────────────

pub const MAGIC = [4]u8{ 'S', 'L', 'C', 'A' };
pub const FORMAT_VERSION: u32 = 1;
pub const DEFAULT_PAGE_SIZE: u32 = 4096;
pub const MIN_PAGE_SIZE: u32 = 512;
pub const MAX_PAGE_SIZE: u32 = 65536;
pub const HEADER_PAGE_ID: u32 = 0;
pub const SCHEMA_ROOT_PAGE_ID: u32 = 1;

// ── Page Types ─────────────────────────────────────────────────────────

pub const PageType = enum(u8) {
    /// Database header page (page 0 only)
    header = 0x01,
    /// Internal B+Tree node
    internal = 0x02,
    /// Leaf B+Tree node
    leaf = 0x03,
    /// Overflow page for large values
    overflow = 0x04,
    /// Free page (part of freelist)
    free = 0x05,
};

// ── Page Header ────────────────────────────────────────────────────────
// Every page starts with this 16-byte header.

pub const PAGE_HEADER_SIZE: u32 = 16;

/// Per-page header at the start of every page.
pub const PageHeader = struct {
    /// Type of this page
    page_type: PageType,
    /// Reserved byte for flags
    flags: u8 = 0,
    /// Number of cells/entries in this page (meaning varies by page type)
    cell_count: u16 = 0,
    /// Page number
    page_id: u32,
    /// Offset to first free byte within the page (for space management)
    free_offset: u32 = 0,
    /// CRC32C checksum of the page content (header excluded from checksum)
    checksum_value: u32 = 0,

    pub fn serialize(self: PageHeader, buf: []u8) void {
        std.debug.assert(buf.len >= PAGE_HEADER_SIZE);
        buf[0] = @intFromEnum(self.page_type);
        buf[1] = self.flags;
        std.mem.writeInt(u16, buf[2..4], self.cell_count, .little);
        std.mem.writeInt(u32, buf[4..8], self.page_id, .little);
        std.mem.writeInt(u32, buf[8..12], self.free_offset, .little);
        std.mem.writeInt(u32, buf[12..16], self.checksum_value, .little);
    }

    pub fn deserialize(buf: []const u8) PageHeader {
        std.debug.assert(buf.len >= PAGE_HEADER_SIZE);
        return .{
            .page_type = @enumFromInt(buf[0]),
            .flags = buf[1],
            .cell_count = std.mem.readInt(u16, buf[2..4], .little),
            .page_id = std.mem.readInt(u32, buf[4..8], .little),
            .free_offset = std.mem.readInt(u32, buf[8..12], .little),
            .checksum_value = std.mem.readInt(u32, buf[12..16], .little),
        };
    }
};

// ── Database Header (Page 0) ───────────────────────────────────────────
// Stored in the first page. Contains metadata about the entire database.

pub const DB_HEADER_SIZE: u32 = 64;

pub const DatabaseHeader = struct {
    magic: [4]u8 = MAGIC,
    format_version: u32 = FORMAT_VERSION,
    page_size: u32 = DEFAULT_PAGE_SIZE,
    page_count: u32 = 0,
    freelist_head: u32 = 0, // 0 = no free pages
    schema_version: u32 = 0,
    wal_mode: u8 = 0,
    reserved: [31]u8 = [_]u8{0} ** 31,

    pub fn serialize(self: DatabaseHeader, buf: []u8) void {
        std.debug.assert(buf.len >= DB_HEADER_SIZE);
        @memcpy(buf[0..4], &self.magic);
        std.mem.writeInt(u32, buf[4..8], self.format_version, .little);
        std.mem.writeInt(u32, buf[8..12], self.page_size, .little);
        std.mem.writeInt(u32, buf[12..16], self.page_count, .little);
        std.mem.writeInt(u32, buf[16..20], self.freelist_head, .little);
        std.mem.writeInt(u32, buf[20..24], self.schema_version, .little);
        buf[24] = self.wal_mode;
        @memcpy(buf[25..56], &self.reserved);
        // Bytes 56-63 reserved for future use, zero-filled
        @memset(buf[56..64], 0);
    }

    pub fn deserialize(buf: []const u8) !DatabaseHeader {
        std.debug.assert(buf.len >= DB_HEADER_SIZE);
        if (!std.mem.eql(u8, buf[0..4], &MAGIC)) return error.InvalidMagic;
        const version = std.mem.readInt(u32, buf[4..8], .little);
        if (version != FORMAT_VERSION) return error.UnsupportedVersion;
        const page_size = std.mem.readInt(u32, buf[8..12], .little);
        if (!isValidPageSize(page_size)) return error.InvalidPageSize;

        var reserved: [31]u8 = undefined;
        @memcpy(&reserved, buf[25..56]);

        return .{
            .magic = MAGIC,
            .format_version = version,
            .page_size = page_size,
            .page_count = std.mem.readInt(u32, buf[12..16], .little),
            .freelist_head = std.mem.readInt(u32, buf[16..20], .little),
            .schema_version = std.mem.readInt(u32, buf[20..24], .little),
            .wal_mode = buf[24],
            .reserved = reserved,
        };
    }
};

// ── Pager ──────────────────────────────────────────────────────────────

pub const Pager = struct {
    file: std.fs.File,
    page_size: u32,
    page_count: u32,
    freelist_head: u32,
    schema_version: u32,
    allocator: std.mem.Allocator,

    pub const InitOptions = struct {
        page_size: u32 = DEFAULT_PAGE_SIZE,
    };

    /// Open or create a database file. If the file is empty, initialize it.
    pub fn init(allocator: std.mem.Allocator, path: []const u8, opts: InitOptions) !Pager {
        if (!isValidPageSize(opts.page_size)) return error.InvalidPageSize;

        const file = try std.fs.cwd().createFile(path, .{
            .read = true,
            .truncate = false,
        });
        errdefer file.close();

        const stat = try file.stat();

        if (stat.size == 0) {
            // New database — write header page
            var self = Pager{
                .file = file,
                .page_size = opts.page_size,
                .page_count = 2, // page 0 (header) + page 1 (schema root)
                .freelist_head = 0,
                .schema_version = 0,
                .allocator = allocator,
            };
            try self.writeHeaderPage();
            try self.writeEmptyPage(SCHEMA_ROOT_PAGE_ID, .leaf);
            return self;
        }

        // Existing database — read and validate header
        if (stat.size < DB_HEADER_SIZE) return error.CorruptDatabase;

        var header_buf: [DB_HEADER_SIZE]u8 = undefined;
        const bytes_read = try file.preadAll(&header_buf, 0);
        if (bytes_read < DB_HEADER_SIZE) return error.CorruptDatabase;

        const db_header = try DatabaseHeader.deserialize(&header_buf);

        return Pager{
            .file = file,
            .page_size = db_header.page_size,
            .page_count = db_header.page_count,
            .freelist_head = db_header.freelist_head,
            .schema_version = db_header.schema_version,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Pager) void {
        self.file.close();
    }

    /// Allocate a page buffer. Caller must free with `freePage`.
    pub fn allocPageBuf(self: *Pager) ![]u8 {
        return try self.allocator.alloc(u8, self.page_size);
    }

    pub fn freePageBuf(self: *Pager, buf: []u8) void {
        self.allocator.free(buf);
    }

    /// Read a page from disk into the provided buffer.
    /// Verifies the checksum before returning.
    pub fn readPage(self: *Pager, page_id: u32, buf: []u8) !void {
        if (page_id >= self.page_count) return error.PageOutOfBounds;
        if (buf.len < self.page_size) return error.BufferTooSmall;

        const offset: u64 = @as(u64, page_id) * @as(u64, self.page_size);
        const bytes_read = try self.file.preadAll(buf[0..self.page_size], offset);
        if (bytes_read < self.page_size) return error.IncompleteRead;

        // Skip checksum verification for header page (it uses DB header format)
        if (page_id == HEADER_PAGE_ID) return;

        // Verify checksum: computed over content after the page header
        const header = PageHeader.deserialize(buf[0..PAGE_HEADER_SIZE]);
        const content = buf[PAGE_HEADER_SIZE..self.page_size];
        if (!checksum.verify(content, header.checksum_value)) {
            return error.ChecksumMismatch;
        }
    }

    /// Write a page to disk. Computes and stores the checksum.
    pub fn writePage(self: *Pager, page_id: u32, buf: []u8) !void {
        if (page_id >= self.page_count) return error.PageOutOfBounds;
        if (buf.len < self.page_size) return error.BufferTooSmall;

        // Skip checksum for header page
        if (page_id != HEADER_PAGE_ID) {
            // Compute checksum over content after page header
            const content = buf[PAGE_HEADER_SIZE..self.page_size];
            const crc = checksum.crc32c(content);
            // Write checksum into header's checksum field (bytes 12-16)
            std.mem.writeInt(u32, buf[12..16], crc, .little);
        }

        const offset: u64 = @as(u64, page_id) * @as(u64, self.page_size);
        try self.file.pwriteAll(buf[0..self.page_size], offset);
    }

    /// Allocate a new page, reusing a freelist page if available.
    /// Returns the page ID of the newly allocated page.
    pub fn allocPage(self: *Pager) !u32 {
        if (self.freelist_head != 0) {
            // Reuse a page from the freelist
            const page_id = self.freelist_head;
            const buf = try self.allocPageBuf();
            defer self.freePageBuf(buf);

            try self.readPage(page_id, buf);
            // The first 4 bytes of the content area hold the next free page pointer
            const next_free = std.mem.readInt(u32, buf[PAGE_HEADER_SIZE..][0..4], .little);
            self.freelist_head = next_free;
            try self.updateHeaderPage();
            return page_id;
        }

        // Extend the file with a new page
        const page_id = self.page_count;
        self.page_count += 1;
        try self.updateHeaderPage();

        // Initialize the new page on disk
        const buf = try self.allocPageBuf();
        defer self.freePageBuf(buf);
        @memset(buf, 0);
        const offset: u64 = @as(u64, page_id) * @as(u64, self.page_size);
        try self.file.pwriteAll(buf[0..self.page_size], offset);

        return page_id;
    }

    /// Free a page and add it to the freelist.
    pub fn freePage(self: *Pager, page_id: u32) !void {
        if (page_id == HEADER_PAGE_ID or page_id == SCHEMA_ROOT_PAGE_ID) {
            return error.CannotFreeReservedPage;
        }
        if (page_id >= self.page_count) return error.PageOutOfBounds;

        const buf = try self.allocPageBuf();
        defer self.freePageBuf(buf);
        @memset(buf, 0);

        // Write free page: header + pointer to previous freelist head
        const header = PageHeader{
            .page_type = .free,
            .page_id = page_id,
        };
        header.serialize(buf[0..PAGE_HEADER_SIZE]);

        // Store the old freelist head in the content area
        std.mem.writeInt(u32, buf[PAGE_HEADER_SIZE..][0..4], self.freelist_head, .little);

        try self.writePage(page_id, buf);

        self.freelist_head = page_id;
        try self.updateHeaderPage();
    }

    /// Get the usable content size per page (total - header).
    pub fn contentSize(self: *Pager) u32 {
        return self.page_size - PAGE_HEADER_SIZE;
    }

    // ── Internal helpers ────────────────────────────────────────────────

    fn writeHeaderPage(self: *Pager) !void {
        const buf = try self.allocPageBuf();
        defer self.freePageBuf(buf);
        @memset(buf, 0);

        const db_header = DatabaseHeader{
            .page_size = self.page_size,
            .page_count = self.page_count,
            .freelist_head = self.freelist_head,
            .schema_version = self.schema_version,
        };
        db_header.serialize(buf[0..DB_HEADER_SIZE]);

        try self.file.pwriteAll(buf[0..self.page_size], 0);
    }

    fn updateHeaderPage(self: *Pager) !void {
        try self.writeHeaderPage();
    }

    fn writeEmptyPage(self: *Pager, page_id: u32, page_type: PageType) !void {
        const buf = try self.allocPageBuf();
        defer self.freePageBuf(buf);
        @memset(buf, 0);

        const header = PageHeader{
            .page_type = page_type,
            .page_id = page_id,
            .free_offset = PAGE_HEADER_SIZE,
        };
        header.serialize(buf[0..PAGE_HEADER_SIZE]);

        try self.writePage(page_id, buf);
    }
};

// ── Utility ────────────────────────────────────────────────────────────

fn isValidPageSize(size: u32) bool {
    if (size < MIN_PAGE_SIZE or size > MAX_PAGE_SIZE) return false;
    // Must be a power of 2
    return (size & (size - 1)) == 0;
}

// ── Tests ──────────────────────────────────────────────────────────────

test "PageHeader serialize/deserialize roundtrip" {
    const header = PageHeader{
        .page_type = .leaf,
        .flags = 0x42,
        .cell_count = 100,
        .page_id = 7,
        .free_offset = 256,
        .checksum_value = 0xDEADBEEF,
    };
    var buf: [PAGE_HEADER_SIZE]u8 = undefined;
    header.serialize(&buf);

    const restored = PageHeader.deserialize(&buf);
    try std.testing.expectEqual(header.page_type, restored.page_type);
    try std.testing.expectEqual(header.flags, restored.flags);
    try std.testing.expectEqual(header.cell_count, restored.cell_count);
    try std.testing.expectEqual(header.page_id, restored.page_id);
    try std.testing.expectEqual(header.free_offset, restored.free_offset);
    try std.testing.expectEqual(header.checksum_value, restored.checksum_value);
}

test "DatabaseHeader serialize/deserialize roundtrip" {
    const header = DatabaseHeader{
        .page_size = 4096,
        .page_count = 10,
        .freelist_head = 5,
        .schema_version = 3,
        .wal_mode = 1,
    };
    var buf: [DB_HEADER_SIZE]u8 = undefined;
    header.serialize(&buf);

    const restored = try DatabaseHeader.deserialize(&buf);
    try std.testing.expectEqual(header.page_size, restored.page_size);
    try std.testing.expectEqual(header.page_count, restored.page_count);
    try std.testing.expectEqual(header.freelist_head, restored.freelist_head);
    try std.testing.expectEqual(header.schema_version, restored.schema_version);
    try std.testing.expectEqual(header.wal_mode, restored.wal_mode);
}

test "DatabaseHeader rejects invalid magic" {
    var buf: [DB_HEADER_SIZE]u8 = undefined;
    const header = DatabaseHeader{};
    header.serialize(&buf);
    buf[0] = 'X'; // corrupt magic
    try std.testing.expectError(error.InvalidMagic, DatabaseHeader.deserialize(&buf));
}

test "DatabaseHeader rejects invalid page size" {
    var buf: [DB_HEADER_SIZE]u8 = undefined;
    const header = DatabaseHeader{ .page_size = 1000 }; // not power of 2
    header.serialize(&buf);
    try std.testing.expectError(error.InvalidPageSize, DatabaseHeader.deserialize(&buf));
}

test "isValidPageSize" {
    try std.testing.expect(isValidPageSize(512));
    try std.testing.expect(isValidPageSize(1024));
    try std.testing.expect(isValidPageSize(4096));
    try std.testing.expect(isValidPageSize(8192));
    try std.testing.expect(isValidPageSize(16384));
    try std.testing.expect(isValidPageSize(32768));
    try std.testing.expect(isValidPageSize(65536));

    try std.testing.expect(!isValidPageSize(0));
    try std.testing.expect(!isValidPageSize(256));  // too small
    try std.testing.expect(!isValidPageSize(1000)); // not power of 2
    try std.testing.expect(!isValidPageSize(5000)); // not power of 2
    try std.testing.expect(!isValidPageSize(131072)); // too large
}

test "Pager create new database" {
    const allocator = std.testing.allocator;
    const path = "test_pager_create.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    {
        var pager = try Pager.init(allocator, path, .{});
        defer pager.deinit();

        try std.testing.expectEqual(@as(u32, 4096), pager.page_size);
        try std.testing.expectEqual(@as(u32, 2), pager.page_count); // header + schema root
        try std.testing.expectEqual(@as(u32, 0), pager.freelist_head);
    }

    // Reopen and verify
    {
        var pager = try Pager.init(allocator, path, .{});
        defer pager.deinit();

        try std.testing.expectEqual(@as(u32, 4096), pager.page_size);
        try std.testing.expectEqual(@as(u32, 2), pager.page_count);
    }
}

test "Pager create with custom page size" {
    const allocator = std.testing.allocator;
    const path = "test_pager_custom_size.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    {
        var pager = try Pager.init(allocator, path, .{ .page_size = 8192 });
        defer pager.deinit();
        try std.testing.expectEqual(@as(u32, 8192), pager.page_size);
    }

    // Reopen — page size read from header, opts.page_size ignored
    {
        var pager = try Pager.init(allocator, path, .{});
        defer pager.deinit();
        try std.testing.expectEqual(@as(u32, 8192), pager.page_size);
    }
}

test "Pager reject invalid page size" {
    const allocator = std.testing.allocator;
    const result = Pager.init(allocator, "test_invalid.db", .{ .page_size = 1000 });
    try std.testing.expectError(error.InvalidPageSize, result);
}

test "Pager allocPage extends file" {
    const allocator = std.testing.allocator;
    const path = "test_pager_alloc.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    try std.testing.expectEqual(@as(u32, 2), pager.page_count);

    const id1 = try pager.allocPage();
    try std.testing.expectEqual(@as(u32, 2), id1);
    try std.testing.expectEqual(@as(u32, 3), pager.page_count);

    const id2 = try pager.allocPage();
    try std.testing.expectEqual(@as(u32, 3), id2);
    try std.testing.expectEqual(@as(u32, 4), pager.page_count);
}

test "Pager write and read page with checksum" {
    const allocator = std.testing.allocator;
    const path = "test_pager_rw.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    const page_id = try pager.allocPage();
    const buf = try pager.allocPageBuf();
    defer pager.freePageBuf(buf);

    // Prepare a leaf page with some data
    @memset(buf, 0);
    const header = PageHeader{
        .page_type = .leaf,
        .page_id = page_id,
        .cell_count = 1,
    };
    header.serialize(buf[0..PAGE_HEADER_SIZE]);

    // Write some payload
    const payload = "Hello, Silica!";
    @memcpy(buf[PAGE_HEADER_SIZE..][0..payload.len], payload);

    // Write page (checksum will be computed automatically)
    try pager.writePage(page_id, buf);

    // Read it back — checksum verified automatically
    var read_buf = try pager.allocPageBuf();
    defer pager.freePageBuf(read_buf);
    try pager.readPage(page_id, read_buf);

    // Verify payload matches
    try std.testing.expectEqualStrings(payload, read_buf[PAGE_HEADER_SIZE..][0..payload.len]);
}

test "Pager detects checksum corruption" {
    const allocator = std.testing.allocator;
    const path = "test_pager_corrupt.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    const page_id = try pager.allocPage();
    const buf = try pager.allocPageBuf();
    defer pager.freePageBuf(buf);

    @memset(buf, 0);
    const header = PageHeader{
        .page_type = .leaf,
        .page_id = page_id,
    };
    header.serialize(buf[0..PAGE_HEADER_SIZE]);
    try pager.writePage(page_id, buf);

    // Corrupt a byte in the content area directly on disk
    const offset: u64 = @as(u64, page_id) * @as(u64, pager.page_size) + PAGE_HEADER_SIZE + 10;
    try pager.file.pwriteAll(&[_]u8{0xFF}, offset);

    // Read should detect corruption
    const read_buf = try pager.allocPageBuf();
    defer pager.freePageBuf(read_buf);
    try std.testing.expectError(error.ChecksumMismatch, pager.readPage(page_id, read_buf));
}

test "Pager freelist: free and reuse pages" {
    const allocator = std.testing.allocator;
    const path = "test_pager_freelist.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    // Allocate 3 pages
    const id1 = try pager.allocPage();
    const id2 = try pager.allocPage();
    const id3 = try pager.allocPage();
    try std.testing.expectEqual(@as(u32, 5), pager.page_count);

    // Free pages in order: id2, id1 (id3 stays allocated)
    try pager.freePage(id2);
    try std.testing.expectEqual(id2, pager.freelist_head);

    try pager.freePage(id1);
    try std.testing.expectEqual(id1, pager.freelist_head);

    // Allocating should reuse freed pages (LIFO: id1 first, then id2)
    const reused1 = try pager.allocPage();
    try std.testing.expectEqual(id1, reused1);

    const reused2 = try pager.allocPage();
    try std.testing.expectEqual(id2, reused2);

    // Next alloc should extend the file
    const new_id = try pager.allocPage();
    try std.testing.expectEqual(@as(u32, 5), new_id);
    try std.testing.expectEqual(@as(u32, 6), pager.page_count);

    _ = id3;
}

test "Pager cannot free reserved pages" {
    const allocator = std.testing.allocator;
    const path = "test_pager_reserved.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    try std.testing.expectError(error.CannotFreeReservedPage, pager.freePage(HEADER_PAGE_ID));
    try std.testing.expectError(error.CannotFreeReservedPage, pager.freePage(SCHEMA_ROOT_PAGE_ID));
}

test "Pager readPage out of bounds" {
    const allocator = std.testing.allocator;
    const path = "test_pager_oob.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    const buf = try pager.allocPageBuf();
    defer pager.freePageBuf(buf);
    try std.testing.expectError(error.PageOutOfBounds, pager.readPage(99, buf));
}

test "Pager persistence across reopen" {
    const allocator = std.testing.allocator;
    const path = "test_pager_persist.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    const payload = "persistent data test";

    // Create, write, close
    var page_id: u32 = undefined;
    {
        var pager = try Pager.init(allocator, path, .{});
        defer pager.deinit();

        page_id = try pager.allocPage();
        const buf = try pager.allocPageBuf();
        defer pager.freePageBuf(buf);

        @memset(buf, 0);
        const header = PageHeader{
            .page_type = .leaf,
            .page_id = page_id,
            .cell_count = 1,
        };
        header.serialize(buf[0..PAGE_HEADER_SIZE]);
        @memcpy(buf[PAGE_HEADER_SIZE..][0..payload.len], payload);
        try pager.writePage(page_id, buf);
    }

    // Reopen and verify
    {
        var pager = try Pager.init(allocator, path, .{});
        defer pager.deinit();

        try std.testing.expectEqual(@as(u32, 3), pager.page_count);

        var buf = try pager.allocPageBuf();
        defer pager.freePageBuf(buf);
        try pager.readPage(page_id, buf);

        try std.testing.expectEqualStrings(payload, buf[PAGE_HEADER_SIZE..][0..payload.len]);
    }
}

test "Pager freelist persists across reopen" {
    const allocator = std.testing.allocator;
    const path = "test_pager_freelist_persist.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    // Create pages, free one, close
    {
        var pager = try Pager.init(allocator, path, .{});
        defer pager.deinit();

        _ = try pager.allocPage(); // id 2
        _ = try pager.allocPage(); // id 3
        try pager.freePage(2);
        try std.testing.expectEqual(@as(u32, 2), pager.freelist_head);
    }

    // Reopen — freelist head should be restored
    {
        var pager = try Pager.init(allocator, path, .{});
        defer pager.deinit();

        try std.testing.expectEqual(@as(u32, 2), pager.freelist_head);

        // Allocating should reuse the freed page
        const reused = try pager.allocPage();
        try std.testing.expectEqual(@as(u32, 2), reused);
    }
}

test "Pager with 512 byte page size" {
    const allocator = std.testing.allocator;
    const path = "test_pager_512.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{ .page_size = 512 });
    defer pager.deinit();

    try std.testing.expectEqual(@as(u32, 512), pager.page_size);
    try std.testing.expectEqual(@as(u32, 512 - PAGE_HEADER_SIZE), pager.contentSize());

    const page_id = try pager.allocPage();
    const buf = try pager.allocPageBuf();
    defer pager.freePageBuf(buf);

    @memset(buf, 0);
    const header = PageHeader{
        .page_type = .leaf,
        .page_id = page_id,
    };
    header.serialize(buf[0..PAGE_HEADER_SIZE]);
    try pager.writePage(page_id, buf);

    const read_buf = try pager.allocPageBuf();
    defer pager.freePageBuf(read_buf);
    try pager.readPage(page_id, read_buf);
}
