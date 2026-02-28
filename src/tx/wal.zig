//! Write-Ahead Log (WAL) — Ensures atomic, durable transactions.
//!
//! All page modifications are written to a WAL file before being applied to
//! the main database. The WAL format is a sequence of frames, each containing
//! a page number, page data, and a CRC32C checksum. Frames between commit
//! marks form a transaction.
//!
//! On crash recovery, only committed frames are replayed. Uncommitted
//! trailing frames are discarded.

const std = @import("std");
const Allocator = std.mem.Allocator;
const checksum_mod = @import("../util/checksum.zig");
const page_mod = @import("../storage/page.zig");
const Pager = page_mod.Pager;

// ── Constants ──────────────────────────────────────────────────────────

pub const WAL_MAGIC = [4]u8{ 'S', 'L', 'C', 'W' };
pub const WAL_VERSION: u32 = 1;
pub const WAL_HEADER_SIZE: u32 = 32;
pub const WAL_FRAME_HEADER_SIZE: u32 = 24;

// ── WAL Header ─────────────────────────────────────────────────────────

pub const WalHeader = struct {
    magic: [4]u8 = WAL_MAGIC,
    format_version: u32 = WAL_VERSION,
    page_size: u32,
    checkpoint_seq: u32 = 0,
    salt_1: u32,
    salt_2: u32,
    frame_count: u32 = 0,
    checksum: u32 = 0,

    pub fn serialize(self: WalHeader, buf: *[WAL_HEADER_SIZE]u8) void {
        @memcpy(buf[0..4], &self.magic);
        std.mem.writeInt(u32, buf[4..8], self.format_version, .little);
        std.mem.writeInt(u32, buf[8..12], self.page_size, .little);
        std.mem.writeInt(u32, buf[12..16], self.checkpoint_seq, .little);
        std.mem.writeInt(u32, buf[16..20], self.salt_1, .little);
        std.mem.writeInt(u32, buf[20..24], self.salt_2, .little);
        std.mem.writeInt(u32, buf[24..28], self.frame_count, .little);
        // Compute checksum over first 28 bytes
        const cksum = checksum_mod.crc32c(buf[0..28]);
        std.mem.writeInt(u32, buf[28..32], cksum, .little);
    }

    pub fn deserialize(buf: *const [WAL_HEADER_SIZE]u8) !WalHeader {
        if (!std.mem.eql(u8, buf[0..4], &WAL_MAGIC)) return error.InvalidWalMagic;
        const version = std.mem.readInt(u32, buf[4..8], .little);
        if (version != WAL_VERSION) return error.UnsupportedWalVersion;
        const expected_cksum = std.mem.readInt(u32, buf[28..32], .little);
        const actual_cksum = checksum_mod.crc32c(buf[0..28]);
        if (expected_cksum != actual_cksum) return error.WalHeaderCorrupt;

        return WalHeader{
            .magic = WAL_MAGIC,
            .format_version = version,
            .page_size = std.mem.readInt(u32, buf[8..12], .little),
            .checkpoint_seq = std.mem.readInt(u32, buf[12..16], .little),
            .salt_1 = std.mem.readInt(u32, buf[16..20], .little),
            .salt_2 = std.mem.readInt(u32, buf[20..24], .little),
            .frame_count = std.mem.readInt(u32, buf[24..28], .little),
            .checksum = expected_cksum,
        };
    }
};

// ── WAL Frame Header ───────────────────────────────────────────────────

pub const WalFrameHeader = struct {
    page_id: u32,
    db_page_count: u32, // 0 = non-commit, >0 = commit frame
    salt_1: u32,
    salt_2: u32,
    frame_checksum: u32,
    reserved: u32 = 0,

    pub fn serialize(self: WalFrameHeader, buf: *[WAL_FRAME_HEADER_SIZE]u8) void {
        std.mem.writeInt(u32, buf[0..4], self.page_id, .little);
        std.mem.writeInt(u32, buf[4..8], self.db_page_count, .little);
        std.mem.writeInt(u32, buf[8..12], self.salt_1, .little);
        std.mem.writeInt(u32, buf[12..16], self.salt_2, .little);
        std.mem.writeInt(u32, buf[16..20], self.frame_checksum, .little);
        std.mem.writeInt(u32, buf[20..24], self.reserved, .little);
    }

    pub fn deserialize(buf: *const [WAL_FRAME_HEADER_SIZE]u8) WalFrameHeader {
        return WalFrameHeader{
            .page_id = std.mem.readInt(u32, buf[0..4], .little),
            .db_page_count = std.mem.readInt(u32, buf[4..8], .little),
            .salt_1 = std.mem.readInt(u32, buf[8..12], .little),
            .salt_2 = std.mem.readInt(u32, buf[12..16], .little),
            .frame_checksum = std.mem.readInt(u32, buf[16..20], .little),
            .reserved = std.mem.readInt(u32, buf[20..24], .little),
        };
    }

    pub fn isCommit(self: WalFrameHeader) bool {
        return self.db_page_count > 0;
    }
};

// ── WAL Manager ────────────────────────────────────────────────────────

pub const Wal = struct {
    allocator: Allocator,
    file: ?std.fs.File,
    wal_path: []const u8,
    page_size: u32,
    header: WalHeader,

    /// Committed page index: page_id → frame_index (most recent committed version).
    page_index: std.AutoHashMap(u32, u32),

    /// Pending (uncommitted) page index: page_id → frame_index.
    pending_index: std.AutoHashMap(u32, u32),

    /// Total frames currently in the WAL file (committed + pending).
    total_frame_count: u32,

    /// Number of committed frames.
    committed_frame_count: u32,

    // ── Lifecycle ──────────────────────────────────────────────

    pub fn init(allocator: Allocator, db_path: []const u8, page_size: u32) !Wal {
        // Construct WAL path: db_path + "-wal"
        const wal_path = try std.fmt.allocPrint(allocator, "{s}-wal", .{db_path});
        errdefer allocator.free(wal_path);

        var wal = Wal{
            .allocator = allocator,
            .file = null,
            .wal_path = wal_path,
            .page_size = page_size,
            .header = WalHeader{
                .page_size = page_size,
                .salt_1 = 0,
                .salt_2 = 0,
            },
            .page_index = std.AutoHashMap(u32, u32).init(allocator),
            .pending_index = std.AutoHashMap(u32, u32).init(allocator),
            .total_frame_count = 0,
            .committed_frame_count = 0,
        };
        errdefer {
            wal.page_index.deinit();
            wal.pending_index.deinit();
        }

        // Try to open existing WAL file for recovery
        const file = std.fs.cwd().openFile(wal_path, .{ .mode = .read_write }) catch |err| switch (err) {
            error.FileNotFound => {
                // No WAL file — will be created on first write
                return wal;
            },
            else => return err,
        };
        wal.file = file;

        // Attempt recovery
        wal.recover() catch {
            // Recovery failed — delete corrupt WAL and start fresh
            file.close();
            wal.file = null;
            std.fs.cwd().deleteFile(wal_path) catch {};
        };

        return wal;
    }

    pub fn deinit(self: *Wal) void {
        if (self.file) |f| f.close();
        self.allocator.free(self.wal_path);
        self.page_index.deinit();
        self.pending_index.deinit();
    }

    // ── Write Path ─────────────────────────────────────────────

    /// Write a page image as a new WAL frame. Not yet committed.
    pub fn writeFrame(self: *Wal, page_id: u32, page_data: []const u8) !void {
        std.debug.assert(page_data.len == self.page_size);

        // Ensure WAL file is open
        if (self.file == null) {
            try self.createWalFile();
        }
        const file = self.file.?;

        // Build frame header
        var fh_buf: [WAL_FRAME_HEADER_SIZE]u8 = undefined;
        const frame_cksum = computeFrameChecksum(page_id, self.header.salt_1, self.header.salt_2, page_data);

        const fh = WalFrameHeader{
            .page_id = page_id,
            .db_page_count = 0, // non-commit
            .salt_1 = self.header.salt_1,
            .salt_2 = self.header.salt_2,
            .frame_checksum = frame_cksum,
        };
        fh.serialize(&fh_buf);

        // Compute file offset
        const offset = self.frameOffset(self.total_frame_count);

        // Write frame header + page data
        try file.pwriteAll(&fh_buf, offset);
        try file.pwriteAll(page_data, offset + WAL_FRAME_HEADER_SIZE);

        // Track in pending index
        try self.pending_index.put(page_id, self.total_frame_count);
        self.total_frame_count += 1;
    }

    /// Commit the current transaction.
    /// Rewrites the last pending frame as a commit frame (with db_page_count set),
    /// fsyncs the WAL file, then promotes all pending frames to committed.
    pub fn commit(self: *Wal, db_page_count: u32) !void {
        if (self.pending_index.count() == 0) return; // nothing to commit

        const file = self.file orelse return error.WalNotOpen;

        // The last written frame needs to become the commit frame.
        // We rewrite its header with db_page_count set.
        const last_frame_idx = self.total_frame_count - 1;
        const last_offset = self.frameOffset(last_frame_idx);

        // Read back the last frame header to get its page_id
        var fh_buf: [WAL_FRAME_HEADER_SIZE]u8 = undefined;
        const bytes_read = try file.preadAll(&fh_buf, last_offset);
        if (bytes_read < WAL_FRAME_HEADER_SIZE) return error.WalCorrupt;

        var fh = WalFrameHeader.deserialize(&fh_buf);

        // Read page data for checksum recomputation
        const page_buf = try self.allocator.alloc(u8, self.page_size);
        defer self.allocator.free(page_buf);
        const data_read = try file.preadAll(page_buf, last_offset + WAL_FRAME_HEADER_SIZE);
        if (data_read < self.page_size) return error.WalCorrupt;

        // Rewrite as commit frame
        fh.db_page_count = db_page_count;
        fh.frame_checksum = computeFrameChecksum(fh.page_id, fh.salt_1, fh.salt_2, page_buf);
        fh.serialize(&fh_buf);
        try file.pwriteAll(&fh_buf, last_offset);

        // Update WAL header frame count
        self.header.frame_count = self.total_frame_count;
        var hdr_buf: [WAL_HEADER_SIZE]u8 = undefined;
        self.header.serialize(&hdr_buf);
        try file.pwriteAll(&hdr_buf, 0);

        // fsync
        try file.sync();

        // Promote pending → committed
        var it = self.pending_index.iterator();
        while (it.next()) |entry| {
            try self.page_index.put(entry.key_ptr.*, entry.value_ptr.*);
        }
        self.pending_index.clearRetainingCapacity();
        self.committed_frame_count = self.total_frame_count;
    }

    /// Rollback the current transaction — discard all pending frames.
    pub fn rollback(self: *Wal) !void {
        if (self.pending_index.count() == 0) return;

        // Truncate WAL back to committed length
        if (self.file) |file| {
            const committed_end = self.frameOffset(self.committed_frame_count);
            try file.setEndPos(committed_end);
        }

        self.pending_index.clearRetainingCapacity();
        self.total_frame_count = self.committed_frame_count;
    }

    // ── Read Path ──────────────────────────────────────────────

    /// Check if the WAL contains a version of the given page.
    /// Checks pending (uncommitted) first, then committed.
    /// Returns true if found and read into buf.
    pub fn readPage(self: *Wal, page_id: u32, buf: []u8) !bool {
        const file = self.file orelse return false;

        // Check pending (same-transaction visibility)
        if (self.pending_index.get(page_id)) |frame_idx| {
            try self.readFrameData(file, frame_idx, buf);
            return true;
        }

        // Check committed
        if (self.page_index.get(page_id)) |frame_idx| {
            try self.readFrameData(file, frame_idx, buf);
            return true;
        }

        return false;
    }

    // ── Checkpoint ─────────────────────────────────────────────

    /// Copy all committed WAL pages to the main DB file, then reset the WAL.
    pub fn checkpoint(self: *Wal, pager: *Pager) !void {
        if (self.page_index.count() == 0) return; // nothing to checkpoint

        const file = self.file orelse return;
        const page_buf = try self.allocator.alloc(u8, self.page_size);
        defer self.allocator.free(page_buf);

        // Track max db_page_count from commit frames
        var max_db_page_count: u32 = 0;

        // Write each committed page to the main DB
        var it = self.page_index.iterator();
        while (it.next()) |entry| {
            const page_id = entry.key_ptr.*;
            const frame_idx = entry.value_ptr.*;

            // Read the frame's page data
            try self.readFrameData(file, frame_idx, page_buf);

            // Read frame header to get db_page_count for commit frames
            const fh_offset = self.frameOffset(frame_idx);
            var fh_buf: [WAL_FRAME_HEADER_SIZE]u8 = undefined;
            _ = try file.preadAll(&fh_buf, fh_offset);
            const fh = WalFrameHeader.deserialize(&fh_buf);
            if (fh.db_page_count > max_db_page_count) {
                max_db_page_count = fh.db_page_count;
            }

            // Write to main DB
            try pager.writePage(page_id, page_buf);
        }

        // Update pager's page_count if needed
        if (max_db_page_count > pager.page_count) {
            pager.page_count = max_db_page_count;
        }

        // Flush the pager's header page
        try pager.flushHeader();

        // fsync main DB
        pager.file.sync() catch {};

        // Reset WAL — truncate and write fresh header
        try file.setEndPos(0);
        self.header.checkpoint_seq += 1;
        self.header.frame_count = 0;
        // Generate new salts
        var rng = std.Random.DefaultPrng.init(@as(u64, @bitCast(std.time.milliTimestamp())));
        const random = rng.random();
        self.header.salt_1 = random.int(u32);
        self.header.salt_2 = random.int(u32);

        var hdr_buf: [WAL_HEADER_SIZE]u8 = undefined;
        self.header.serialize(&hdr_buf);
        try file.pwriteAll(&hdr_buf, 0);
        try file.sync();

        // Clear indexes
        self.page_index.clearRetainingCapacity();
        self.pending_index.clearRetainingCapacity();
        self.total_frame_count = 0;
        self.committed_frame_count = 0;
    }

    // ── Recovery ───────────────────────────────────────────────

    /// Rebuild page_index from committed transactions in the WAL file.
    fn recover(self: *Wal) !void {
        const file = self.file orelse return;

        // Read WAL header
        var hdr_buf: [WAL_HEADER_SIZE]u8 = undefined;
        const hdr_read = try file.preadAll(&hdr_buf, 0);
        if (hdr_read < WAL_HEADER_SIZE) return error.WalCorrupt;

        self.header = try WalHeader.deserialize(&hdr_buf);
        if (self.header.page_size != self.page_size) return error.WalPageSizeMismatch;

        // Scan frames
        var temp_index = std.AutoHashMap(u32, u32).init(self.allocator);
        defer temp_index.deinit();

        const frame_size = WAL_FRAME_HEADER_SIZE + self.page_size;
        var frame_idx: u32 = 0;
        const page_buf = try self.allocator.alloc(u8, self.page_size);
        defer self.allocator.free(page_buf);
        var fh_buf: [WAL_FRAME_HEADER_SIZE]u8 = undefined;

        while (true) {
            const offset = WAL_HEADER_SIZE + @as(u64, frame_idx) * frame_size;

            // Read frame header
            const fh_read = try file.preadAll(&fh_buf, offset);
            if (fh_read < WAL_FRAME_HEADER_SIZE) break; // end of file

            const fh = WalFrameHeader.deserialize(&fh_buf);

            // Validate salts
            if (fh.salt_1 != self.header.salt_1 or fh.salt_2 != self.header.salt_2) break;

            // Read page data
            const data_read = try file.preadAll(page_buf, offset + WAL_FRAME_HEADER_SIZE);
            if (data_read < self.page_size) break; // incomplete frame

            // Verify checksum
            const expected = computeFrameChecksum(fh.page_id, fh.salt_1, fh.salt_2, page_buf);
            if (fh.frame_checksum != expected) break; // corrupt frame

            // Track in temp index
            try temp_index.put(fh.page_id, frame_idx);

            // If commit frame, promote temp to committed
            if (fh.isCommit()) {
                var temp_it = temp_index.iterator();
                while (temp_it.next()) |entry| {
                    try self.page_index.put(entry.key_ptr.*, entry.value_ptr.*);
                }
                temp_index.clearRetainingCapacity();
                self.committed_frame_count = frame_idx + 1;
            }

            frame_idx += 1;
        }

        self.total_frame_count = self.committed_frame_count;

        // Truncate any uncommitted trailing frames
        if (self.committed_frame_count < frame_idx) {
            const committed_end = WAL_HEADER_SIZE + @as(u64, self.committed_frame_count) * frame_size;
            try file.setEndPos(committed_end);
        }

        // Update header frame_count
        self.header.frame_count = self.committed_frame_count;
    }

    // ── Internal Helpers ───────────────────────────────────────

    fn frameOffset(self: *const Wal, frame_index: u32) u64 {
        const frame_size = WAL_FRAME_HEADER_SIZE + self.page_size;
        return WAL_HEADER_SIZE + @as(u64, frame_index) * frame_size;
    }

    fn readFrameData(self: *const Wal, file: std.fs.File, frame_index: u32, buf: []u8) !void {
        const offset = self.frameOffset(frame_index) + WAL_FRAME_HEADER_SIZE;
        const bytes_read = try file.preadAll(buf, offset);
        if (bytes_read < self.page_size) return error.WalCorrupt;
    }

    fn createWalFile(self: *Wal) !void {
        const file = try std.fs.cwd().createFile(self.wal_path, .{ .read = true });
        self.file = file;

        // Generate random salts
        var rng = std.Random.DefaultPrng.init(@as(u64, @bitCast(std.time.milliTimestamp())));
        const random = rng.random();
        self.header.salt_1 = random.int(u32);
        self.header.salt_2 = random.int(u32);
        self.header.frame_count = 0;
        self.header.checkpoint_seq = 0;

        var hdr_buf: [WAL_HEADER_SIZE]u8 = undefined;
        self.header.serialize(&hdr_buf);
        try file.pwriteAll(&hdr_buf, 0);
    }
};

/// Compute the frame checksum: CRC32C over the first 16 bytes of frame header
/// fields (page_id, db_page_count=0, salt_1, salt_2) plus the page data.
fn computeFrameChecksum(page_id: u32, salt_1: u32, salt_2: u32, page_data: []const u8) u32 {
    var hdr_bytes: [16]u8 = undefined;
    std.mem.writeInt(u32, hdr_bytes[0..4], page_id, .little);
    std.mem.writeInt(u32, hdr_bytes[4..8], 0, .little); // db_page_count not included in checksum
    std.mem.writeInt(u32, hdr_bytes[8..12], salt_1, .little);
    std.mem.writeInt(u32, hdr_bytes[12..16], salt_2, .little);
    const partial = checksum_mod.crc32c(&hdr_bytes);
    return checksum_mod.crc32cUpdate(partial, page_data);
}

// ── Tests ──────────────────────────────────────────────────────────────

const testing = std.testing;

test "WalHeader serialize/deserialize roundtrip" {
    const header = WalHeader{
        .page_size = 4096,
        .checkpoint_seq = 5,
        .salt_1 = 0xDEADBEEF,
        .salt_2 = 0xCAFEBABE,
        .frame_count = 42,
    };
    var buf: [WAL_HEADER_SIZE]u8 = undefined;
    header.serialize(&buf);

    const restored = try WalHeader.deserialize(&buf);
    try testing.expectEqual(header.page_size, restored.page_size);
    try testing.expectEqual(header.checkpoint_seq, restored.checkpoint_seq);
    try testing.expectEqual(header.salt_1, restored.salt_1);
    try testing.expectEqual(header.salt_2, restored.salt_2);
    try testing.expectEqual(header.frame_count, restored.frame_count);
}

test "WalHeader rejects invalid magic" {
    var buf: [WAL_HEADER_SIZE]u8 = undefined;
    const header = WalHeader{ .page_size = 4096, .salt_1 = 1, .salt_2 = 2 };
    header.serialize(&buf);
    buf[0] = 'X'; // corrupt magic
    try testing.expectError(error.InvalidWalMagic, WalHeader.deserialize(&buf));
}

test "WalHeader rejects corrupt checksum" {
    var buf: [WAL_HEADER_SIZE]u8 = undefined;
    const header = WalHeader{ .page_size = 4096, .salt_1 = 1, .salt_2 = 2 };
    header.serialize(&buf);
    buf[24] ^= 0xFF; // flip a byte in frame_count area
    try testing.expectError(error.WalHeaderCorrupt, WalHeader.deserialize(&buf));
}

test "WalFrameHeader serialize/deserialize roundtrip" {
    const fh = WalFrameHeader{
        .page_id = 7,
        .db_page_count = 100,
        .salt_1 = 0x11111111,
        .salt_2 = 0x22222222,
        .frame_checksum = 0xAABBCCDD,
    };
    var buf: [WAL_FRAME_HEADER_SIZE]u8 = undefined;
    fh.serialize(&buf);

    const restored = WalFrameHeader.deserialize(&buf);
    try testing.expectEqual(fh.page_id, restored.page_id);
    try testing.expectEqual(fh.db_page_count, restored.db_page_count);
    try testing.expectEqual(fh.salt_1, restored.salt_1);
    try testing.expectEqual(fh.salt_2, restored.salt_2);
    try testing.expectEqual(fh.frame_checksum, restored.frame_checksum);
}

test "WalFrameHeader isCommit" {
    const commit_frame = WalFrameHeader{ .page_id = 1, .db_page_count = 10, .salt_1 = 0, .salt_2 = 0, .frame_checksum = 0 };
    const non_commit = WalFrameHeader{ .page_id = 1, .db_page_count = 0, .salt_1 = 0, .salt_2 = 0, .frame_checksum = 0 };
    try testing.expect(commit_frame.isCommit());
    try testing.expect(!non_commit.isCommit());
}

test "Wal init with no existing WAL file" {
    const path = "test_wal_init.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    defer std.fs.cwd().deleteFile("test_wal_init.db-wal") catch {};

    var wal = try Wal.init(testing.allocator, path, 4096);
    defer wal.deinit();

    try testing.expect(wal.file == null);
    try testing.expectEqual(@as(u32, 0), wal.total_frame_count);
    try testing.expectEqual(@as(u32, 0), wal.committed_frame_count);
}

test "Wal write frame and commit" {
    const path = "test_wal_write.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    defer std.fs.cwd().deleteFile("test_wal_write.db-wal") catch {};

    var wal = try Wal.init(testing.allocator, path, 512);
    defer wal.deinit();

    // Write a frame
    var page_data: [512]u8 = undefined;
    @memset(&page_data, 0);
    page_data[0] = 0xAB;
    page_data[1] = 0xCD;
    try wal.writeFrame(5, &page_data);

    try testing.expectEqual(@as(u32, 1), wal.total_frame_count);
    try testing.expectEqual(@as(u32, 0), wal.committed_frame_count);
    try testing.expect(wal.pending_index.get(5) != null);

    // Commit
    try wal.commit(10);
    try testing.expectEqual(@as(u32, 1), wal.committed_frame_count);
    try testing.expect(wal.page_index.get(5) != null);
    try testing.expectEqual(@as(u32, 0), wal.pending_index.count());
}

test "Wal read page from committed index" {
    const path = "test_wal_read.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    defer std.fs.cwd().deleteFile("test_wal_read.db-wal") catch {};

    var wal = try Wal.init(testing.allocator, path, 512);
    defer wal.deinit();

    // Write and commit page 3
    var page_data: [512]u8 = undefined;
    @memset(&page_data, 0);
    const marker = "WAL_PAGE_3";
    @memcpy(page_data[0..marker.len], marker);
    try wal.writeFrame(3, &page_data);
    try wal.commit(5);

    // Read back
    var read_buf: [512]u8 = undefined;
    const found = try wal.readPage(3, &read_buf);
    try testing.expect(found);
    try testing.expectEqualStrings(marker, read_buf[0..marker.len]);

    // Non-existent page
    const found2 = try wal.readPage(99, &read_buf);
    try testing.expect(!found2);
}

test "Wal same-transaction visibility via pending index" {
    const path = "test_wal_pending.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    defer std.fs.cwd().deleteFile("test_wal_pending.db-wal") catch {};

    var wal = try Wal.init(testing.allocator, path, 512);
    defer wal.deinit();

    // Write frame but don't commit
    var page_data: [512]u8 = undefined;
    @memset(&page_data, 0);
    page_data[0] = 0xFF;
    try wal.writeFrame(7, &page_data);

    // Should be readable from pending index
    var read_buf: [512]u8 = undefined;
    const found = try wal.readPage(7, &read_buf);
    try testing.expect(found);
    try testing.expectEqual(@as(u8, 0xFF), read_buf[0]);
}

test "Wal rollback discards pending frames" {
    const path = "test_wal_rollback.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    defer std.fs.cwd().deleteFile("test_wal_rollback.db-wal") catch {};

    var wal = try Wal.init(testing.allocator, path, 512);
    defer wal.deinit();

    // Write and commit a frame
    var page1: [512]u8 = undefined;
    @memset(&page1, 0);
    page1[0] = 0x11;
    try wal.writeFrame(1, &page1);
    try wal.commit(5);

    // Write another frame but rollback
    var page2: [512]u8 = undefined;
    @memset(&page2, 0);
    page2[0] = 0x22;
    try wal.writeFrame(2, &page2);
    try testing.expectEqual(@as(u32, 2), wal.total_frame_count);

    try wal.rollback();
    try testing.expectEqual(@as(u32, 1), wal.total_frame_count);
    try testing.expectEqual(@as(u32, 0), wal.pending_index.count());

    // Page 1 still readable (committed), page 2 gone
    var buf: [512]u8 = undefined;
    try testing.expect(try wal.readPage(1, &buf));
    try testing.expect(!try wal.readPage(2, &buf));
}

test "Wal multiple transactions" {
    const path = "test_wal_multitx.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    defer std.fs.cwd().deleteFile("test_wal_multitx.db-wal") catch {};

    var wal = try Wal.init(testing.allocator, path, 512);
    defer wal.deinit();

    // Transaction 1: write pages 1, 2
    var p1: [512]u8 = undefined;
    @memset(&p1, 0);
    p1[0] = 0xAA;
    try wal.writeFrame(1, &p1);
    var p2: [512]u8 = undefined;
    @memset(&p2, 0);
    p2[0] = 0xBB;
    try wal.writeFrame(2, &p2);
    try wal.commit(5);

    // Transaction 2: overwrite page 1
    var p1v2: [512]u8 = undefined;
    @memset(&p1v2, 0);
    p1v2[0] = 0xCC;
    try wal.writeFrame(1, &p1v2);
    try wal.commit(5);

    // Page 1 should have the latest value
    var buf: [512]u8 = undefined;
    try testing.expect(try wal.readPage(1, &buf));
    try testing.expectEqual(@as(u8, 0xCC), buf[0]);

    // Page 2 still has its original value
    try testing.expect(try wal.readPage(2, &buf));
    try testing.expectEqual(@as(u8, 0xBB), buf[0]);
}

test "Wal checkpoint writes to main DB" {
    const path = "test_wal_ckpt.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    defer std.fs.cwd().deleteFile("test_wal_ckpt.db-wal") catch {};

    const PageHeader = page_mod.PageHeader;
    const PAGE_HEADER_SIZE = page_mod.PAGE_HEADER_SIZE;

    // Create a real database file with a pager
    var pager = try Pager.init(testing.allocator, path, .{ .page_size = 512 });

    // Allocate a page so pager has it
    const pid = try pager.allocPage();

    var wal = try Wal.init(testing.allocator, path, 512);
    defer wal.deinit();

    // Write via WAL — use properly formatted page with PageHeader
    var page_data: [512]u8 = undefined;
    @memset(&page_data, 0);
    const hdr = PageHeader{ .page_type = .leaf, .page_id = pid, .cell_count = 7 };
    hdr.serialize(page_data[0..PAGE_HEADER_SIZE]);
    const marker = "CHECKPOINT_DATA";
    @memcpy(page_data[PAGE_HEADER_SIZE..][0..marker.len], marker);
    try wal.writeFrame(pid, &page_data);
    try wal.commit(pager.page_count);

    // Checkpoint — writes to main DB (pager.writePage recomputes checksum)
    try wal.checkpoint(&pager);

    // WAL should be reset
    try testing.expectEqual(@as(u32, 0), wal.page_index.count());
    try testing.expectEqual(@as(u32, 0), wal.total_frame_count);

    // Read directly from pager (validates checksum)
    var read_buf: [512]u8 = undefined;
    try pager.readPage(pid, &read_buf);
    try testing.expectEqualStrings(marker, read_buf[PAGE_HEADER_SIZE..][0..marker.len]);

    // Verify cell_count survived
    const restored_hdr = PageHeader.deserialize(read_buf[0..PAGE_HEADER_SIZE]);
    try testing.expectEqual(@as(u16, 7), restored_hdr.cell_count);

    pager.deinit();
}

test "Wal recovery replays committed frames" {
    const path = "test_wal_recover.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    defer std.fs.cwd().deleteFile("test_wal_recover.db-wal") catch {};

    // First session: write and commit
    {
        var wal = try Wal.init(testing.allocator, path, 512);
        var page_data: [512]u8 = undefined;
        @memset(&page_data, 0);
        page_data[0] = 0x42;
        try wal.writeFrame(3, &page_data);
        try wal.commit(5);
        // Close WITHOUT checkpoint — simulates crash
        wal.deinit();
    }

    // Second session: should recover committed frames
    {
        var wal = try Wal.init(testing.allocator, path, 512);
        defer wal.deinit();

        try testing.expectEqual(@as(u32, 1), wal.committed_frame_count);
        try testing.expect(wal.page_index.get(3) != null);

        var buf: [512]u8 = undefined;
        const found = try wal.readPage(3, &buf);
        try testing.expect(found);
        try testing.expectEqual(@as(u8, 0x42), buf[0]);
    }
}

test "Wal recovery discards uncommitted frames" {
    const path = "test_wal_recover_uncommit.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    defer std.fs.cwd().deleteFile("test_wal_recover_uncommit.db-wal") catch {};

    // First session: commit tx1, then write tx2 without commit
    {
        var wal = try Wal.init(testing.allocator, path, 512);
        var p1: [512]u8 = undefined;
        @memset(&p1, 0);
        p1[0] = 0x11;
        try wal.writeFrame(1, &p1);
        try wal.commit(5);

        // Uncommitted frame
        var p2: [512]u8 = undefined;
        @memset(&p2, 0);
        p2[0] = 0x22;
        try wal.writeFrame(2, &p2);
        // Close without commit — simulates crash
        wal.deinit();
    }

    // Second session
    {
        var wal = try Wal.init(testing.allocator, path, 512);
        defer wal.deinit();

        // Only tx1 should be recovered
        try testing.expectEqual(@as(u32, 1), wal.committed_frame_count);
        try testing.expect(wal.page_index.get(1) != null);
        try testing.expect(wal.page_index.get(2) == null);
    }
}

test "Wal computeFrameChecksum consistency" {
    var data: [512]u8 = undefined;
    @memset(&data, 0xAB);
    const ck1 = computeFrameChecksum(5, 0x111, 0x222, &data);
    const ck2 = computeFrameChecksum(5, 0x111, 0x222, &data);
    try testing.expectEqual(ck1, ck2);

    // Different page_id → different checksum
    const ck3 = computeFrameChecksum(6, 0x111, 0x222, &data);
    try testing.expect(ck1 != ck3);

    // Different data → different checksum
    data[0] = 0;
    const ck4 = computeFrameChecksum(5, 0x111, 0x222, &data);
    try testing.expect(ck1 != ck4);
}
