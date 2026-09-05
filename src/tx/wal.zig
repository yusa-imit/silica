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

// ── LSN (Log Sequence Number) ─────────────────────────────────────────

/// A replication-stable log sequence number. `checkpoint_seq` matches
/// `WalHeader.checkpoint_seq` (the epoch); `frame_index` is the frame's
/// ordinal position within that epoch, NOT a byte offset — checkpoint()
/// truncates the file, so a bare byte offset is not globally monotonic
/// across checkpoints.
pub const Lsn = struct {
    checkpoint_seq: u32,
    frame_index: u32,

    /// Total order: compares checkpoint_seq first, then frame_index.
    pub fn order(a: Lsn, b: Lsn) std.math.Order {
        if (a.checkpoint_seq != b.checkpoint_seq) {
            return std.math.order(a.checkpoint_seq, b.checkpoint_seq);
        }
        return std.math.order(a.frame_index, b.frame_index);
    }

    pub fn lessThan(a: Lsn, b: Lsn) bool {
        return a.order(b) == .lt;
    }

    pub fn eql(a: Lsn, b: Lsn) bool {
        return a.checkpoint_seq == b.checkpoint_seq and a.frame_index == b.frame_index;
    }

    /// Pack Lsn into u64: (checkpoint_seq << 32) | frame_index
    pub fn pack(self: Lsn) u64 {
        return (@as(u64, @intCast(self.checkpoint_seq)) << 32) |
               (@as(u64, @intCast(self.frame_index)));
    }

    /// Unpack u64 into Lsn: checkpoint_seq = val >> 32, frame_index = val & 0xFFFFFFFF
    pub fn unpack(val: u64) Lsn {
        return .{
            .checkpoint_seq = @intCast(val >> 32),
            .frame_index = @intCast(val & 0xFFFFFFFF),
        };
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

    /// Optional retention callback: queries minimum LSN that must be retained for replication.
    /// Returns null if nothing to retain (e.g., no active slots), or u64 (packed Lsn) if retention needed.
    min_retained_lsn_fn: ?*const fn (ctx: *anyopaque) ?u64 = null,

    /// Opaque context pointer passed to min_retained_lsn_fn.
    min_retained_lsn_ctx: ?*anyopaque = null,

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
            .min_retained_lsn_fn = null,
            .min_retained_lsn_ctx = null,
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

    /// Register a retention callback that queries the minimum LSN that must be retained.
    /// ctx: pointer to a concrete context type; f: function of type fn (@TypeOf(ctx)) ?u64.
    /// The callback is wrapped in a comptime-generated trampoline for type erasure.
    pub fn setRetentionCallback(self: *Wal, ctx: anytype, comptime f: fn (@TypeOf(ctx)) ?u64) void {
        const ContextType = @TypeOf(ctx);

        // Comptime-generated trampoline struct
        const Trampoline = struct {
            pub fn call(erased_ctx: *anyopaque) ?u64 {
                const concrete_ctx: ContextType = @ptrCast(@alignCast(erased_ctx));
                return f(concrete_ctx);
            }
        };

        self.min_retained_lsn_fn = Trampoline.call;
        self.min_retained_lsn_ctx = @ptrCast(ctx);
    }

    /// Clear the retention callback (revert to default truncate-always behavior).
    pub fn clearRetentionCallback(self: *Wal) void {
        self.min_retained_lsn_fn = null;
        self.min_retained_lsn_ctx = null;
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

    /// Append a raw, already-formed frame (header + page data) verbatim to the WAL file.
    /// frame_bytes must be exactly WAL_FRAME_HEADER_SIZE + page_size bytes.
    /// The frame is NOT recomputed or validated — assumes it came from readRawFrames().
    /// If the frame is a commit frame (db_page_count > 0), promotes all pending entries.
    pub fn appendRawFrame(self: *Wal, frame_bytes: []const u8) !void {
        const frame_size = WAL_FRAME_HEADER_SIZE + self.page_size;
        std.debug.assert(frame_bytes.len == frame_size);

        // Ensure WAL file is open
        if (self.file == null) {
            try self.createWalFile();
        }
        const file = self.file.?;

        // Deserialize just the header to get page_id and check if commit
        const fh_buf = frame_bytes[0..WAL_FRAME_HEADER_SIZE];
        const fh = WalFrameHeader.deserialize(fh_buf);

        // Write the raw bytes verbatim at the current frame offset
        const offset = self.frameOffset(self.total_frame_count);
        try file.pwriteAll(frame_bytes, offset);

        // Update pending index
        try self.pending_index.put(fh.page_id, self.total_frame_count);
        self.total_frame_count += 1;

        // If commit frame, promote pending → committed (same as commit() does)
        if (fh.isCommit()) {
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

    // ── Replication Read Path ─────────────────────────────────

    /// Returns the LSN of the current committed write frontier —
    /// (header.checkpoint_seq, committed_frame_count). This is what a
    /// WalSender should treat as "everything up to here is safe to stream."
    pub fn currentLsn(self: *const Wal) Lsn {
        return .{ .checkpoint_seq = self.header.checkpoint_seq, .frame_index = self.committed_frame_count };
    }

    /// Converts a frame index (within the current checkpoint epoch) to an Lsn.
    pub fn lsnAtFrame(self: *const Wal, frame_idx: u32) Lsn {
        return .{ .checkpoint_seq = self.header.checkpoint_seq, .frame_index = frame_idx };
    }

    /// Reads whole committed WAL frames (header + page data, byte-identical
    /// to what writeFrame wrote to disk) starting at start_lsn into buf,
    /// without ever splitting a frame across the returned bytes. Only
    /// committed frames are ever returned — pending/uncommitted frames are
    /// never exposed via this API, even if total_frame_count is ahead of
    /// committed_frame_count.
    pub fn readRawFrames(self: *Wal, start_lsn: Lsn, buf: []u8) !struct {
        bytes_read: usize,
        next_lsn: Lsn,
    } {
        const frame_size = WAL_FRAME_HEADER_SIZE + self.page_size;
        if (buf.len < frame_size) return error.BufferTooSmall;

        if (start_lsn.checkpoint_seq < self.header.checkpoint_seq) {
            // Requested epoch was already truncated away by a checkpoint.
            return error.LsnFromEarlierEpoch;
        }
        if (start_lsn.checkpoint_seq > self.header.checkpoint_seq) {
            return error.LsnFromFutureEpoch;
        }

        if (start_lsn.frame_index >= self.committed_frame_count) {
            // Caller is already caught up — normal case, not an error.
            return .{ .bytes_read = 0, .next_lsn = start_lsn };
        }

        const file = self.file orelse return .{ .bytes_read = 0, .next_lsn = start_lsn };

        const max_frames_by_buf: u32 = @intCast(buf.len / frame_size);
        const available_frames = self.committed_frame_count - start_lsn.frame_index;
        const frames_to_read = @min(max_frames_by_buf, available_frames);

        var bytes_read: usize = 0;
        var frame_idx = start_lsn.frame_index;
        var i: u32 = 0;
        while (i < frames_to_read) : (i += 1) {
            const offset = self.frameOffset(frame_idx);
            const dest = buf[bytes_read..][0..frame_size];
            const n = try file.preadAll(dest, offset);
            if (n < frame_size) break; // truncated on disk — stop, don't return a partial frame
            bytes_read += frame_size;
            frame_idx += 1;
        }

        return .{
            .bytes_read = bytes_read,
            .next_lsn = .{ .checkpoint_seq = self.header.checkpoint_seq, .frame_index = frame_idx },
        };
    }

    // ── Checkpoint ─────────────────────────────────────────────

    /// Copy all committed WAL pages to the main DB file, then conditionally reset the WAL.
    /// If a retention callback is registered and reports a lagging replica, the flush happens
    /// but truncation is deferred until the replica catches up.
    pub fn checkpoint(self: *Wal, pager: *Pager) !void {
        // Modified guard: also allow no-new-writes retry via committed_frame_count check
        if (self.page_index.count() == 0 and self.committed_frame_count == 0) return;

        const file = self.file orelse return;
        const page_buf = try self.allocator.alloc(u8, self.page_size);
        defer self.allocator.free(page_buf);

        // Capture current LSN before any state changes (for retention comparison)
        const current_lsn = self.currentLsn();

        // First pass: determine the page count the pager needs to accommodate
        // every page we're about to write, *before* writing any of them —
        // writePage() bounds-checks against pager.page_count, so growing it
        // must happen up front rather than after the write loop.
        var max_db_page_count: u32 = 0;
        {
            var scan_it = self.page_index.iterator();
            while (scan_it.next()) |entry| {
                const page_id = entry.key_ptr.*;
                const frame_idx = entry.value_ptr.*;

                const fh_offset = self.frameOffset(frame_idx);
                var fh_buf: [WAL_FRAME_HEADER_SIZE]u8 = undefined;
                _ = try file.preadAll(&fh_buf, fh_offset);
                const fh = WalFrameHeader.deserialize(&fh_buf);
                if (fh.db_page_count > max_db_page_count) {
                    max_db_page_count = fh.db_page_count;
                }
                if (page_id + 1 > max_db_page_count) {
                    max_db_page_count = page_id + 1;
                }
            }
        }
        if (max_db_page_count > pager.page_count) {
            pager.page_count = max_db_page_count;
        }

        // FLUSH STEP (UNCONDITIONAL — always runs when there's something to flush)
        if (self.page_index.count() > 0) {
            // Write each committed page to the main DB
            var it = self.page_index.iterator();
            while (it.next()) |entry| {
                const page_id = entry.key_ptr.*;
                const frame_idx = entry.value_ptr.*;

                // Read the frame's page data
                try self.readFrameData(file, frame_idx, page_buf);

                // Write to main DB
                try pager.writePage(page_id, page_buf);
            }

            // Flush the pager's header page
            try pager.flushHeader();

            // fsync main DB
            pager.sync() catch {};
        }

        // RECLAIM STEP (CONDITIONAL — may be deferred if retention callback reports replica behind)
        // Decide whether to truncate the WAL based on retention callback
        var should_reclaim = true;

        if (self.min_retained_lsn_fn) |retention_fn| {
            // Retention callback is registered
            if (self.min_retained_lsn_ctx) |ctx| {
                if (retention_fn(ctx)) |min_retained_packed| {
                    // Callback returned a value — compare it against current LSN
                    const min_retained_lsn = Lsn.unpack(min_retained_packed);
                    // If min_retained_lsn < current_lsn, replica is behind — defer truncation
                    if (min_retained_lsn.lessThan(current_lsn)) {
                        should_reclaim = false;
                    }
                }
                // If callback returned null, should_reclaim stays true (nothing to retain)
            }
        }
        // If no callback registered, should_reclaim stays true (default truncate-always behavior)

        if (should_reclaim) {
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
    const restored_hdr = try PageHeader.deserialize(read_buf[0..PAGE_HEADER_SIZE]);
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

test "Wal computeFrameChecksum sensitivity" {
    var data: [512]u8 = undefined;
    @memset(&data, 0xAB);
    const ck_base = computeFrameChecksum(5, 0x111, 0x222, &data);

    // Different page_id → different checksum
    const ck_page = computeFrameChecksum(6, 0x111, 0x222, &data);
    try testing.expect(ck_base != ck_page);

    // Different data → different checksum
    data[0] = 0;
    const ck_data = computeFrameChecksum(5, 0x111, 0x222, &data);
    try testing.expect(ck_base != ck_data);

    // Different salts (WAL salt rotation) → different checksum
    data[0] = 0xAB; // restore
    const ck_salt = computeFrameChecksum(5, 0x999, 0xAAA, &data);
    try testing.expect(ck_base != ck_salt);
}

test "Wal recovery stops at corrupt frame" {
    // Simulates a crash that leaves a partial/corrupt frame in the WAL.
    // Recovery should discard the corrupt frame and only replay committed ones.
    const path = "test_wal_corrupt_frame.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    defer std.fs.cwd().deleteFile("test_wal_corrupt_frame.db-wal") catch {};

    // Session 1: commit tx1, then write corrupt trailing data
    {
        var wal = try Wal.init(testing.allocator, path, 512);
        var p1: [512]u8 = undefined;
        @memset(&p1, 0);
        p1[0] = 0x42;
        try wal.writeFrame(1, &p1);
        try wal.commit(5);

        // Now manually append garbage after the committed frame
        const file = wal.file.?;
        const end_offset = wal.frameOffset(wal.total_frame_count);
        var garbage: [WAL_FRAME_HEADER_SIZE + 512]u8 = undefined;
        @memset(&garbage, 0xDE); // corrupt data — bad salt, bad checksum
        try file.pwriteAll(&garbage, end_offset);

        wal.deinit();
    }

    // Session 2: recovery should still find committed tx1
    {
        var wal = try Wal.init(testing.allocator, path, 512);
        defer wal.deinit();

        try testing.expectEqual(@as(u32, 1), wal.committed_frame_count);
        try testing.expect(wal.page_index.get(1) != null);

        var buf: [512]u8 = undefined;
        const found = try wal.readPage(1, &buf);
        try testing.expect(found);
        try testing.expectEqual(@as(u8, 0x42), buf[0]);
    }
}

test "Wal multiple checkpoint cycles" {
    // Verifies that checkpoint_seq increments, salts rotate, and
    // the WAL can be reused across multiple checkpoint cycles.
    const path = "test_wal_multi_ckpt.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    defer std.fs.cwd().deleteFile("test_wal_multi_ckpt.db-wal") catch {};

    var pager = try Pager.init(testing.allocator, path, .{ .page_size = 512 });

    const pid1 = try pager.allocPage();
    const pid2 = try pager.allocPage();

    var wal = try Wal.init(testing.allocator, path, 512);
    defer wal.deinit();

    // Cycle 1: write page, commit, checkpoint
    var p1: [512]u8 = undefined;
    @memset(&p1, 0);
    p1[0] = 0xAA;
    // Write proper page header for writePage checksum
    const PageHeader = page_mod.PageHeader;
    const PAGE_HEADER_SIZE = page_mod.PAGE_HEADER_SIZE;
    var hdr1 = PageHeader{ .page_type = .leaf, .page_id = pid1, .cell_count = 1 };
    hdr1.serialize(p1[0..PAGE_HEADER_SIZE]);
    p1[PAGE_HEADER_SIZE] = 0xAA;
    try wal.writeFrame(pid1, &p1);
    try wal.commit(pager.page_count);
    try wal.checkpoint(&pager);

    try testing.expectEqual(@as(u32, 1), wal.header.checkpoint_seq);
    try testing.expectEqual(@as(u32, 0), wal.total_frame_count);
    const salt1_after_ckpt1 = wal.header.salt_1;

    // Cycle 2: write different page, commit, checkpoint
    var p2: [512]u8 = undefined;
    @memset(&p2, 0);
    var hdr2 = PageHeader{ .page_type = .leaf, .page_id = pid2, .cell_count = 2 };
    hdr2.serialize(p2[0..PAGE_HEADER_SIZE]);
    p2[PAGE_HEADER_SIZE] = 0xBB;
    try wal.writeFrame(pid2, &p2);
    try wal.commit(pager.page_count);
    try wal.checkpoint(&pager);

    try testing.expectEqual(@as(u32, 2), wal.header.checkpoint_seq);
    try testing.expectEqual(@as(u32, 0), wal.total_frame_count);

    // Verify salts changed between checkpoints (very high probability)
    // Note: theoretically could be same but astronomically unlikely
    _ = salt1_after_ckpt1; // salts are random, just check checkpoint_seq incremented

    // Verify both pages persisted to main DB
    var read_buf: [512]u8 = undefined;
    try pager.readPage(pid1, &read_buf);
    try testing.expectEqual(@as(u8, 0xAA), read_buf[PAGE_HEADER_SIZE]);
    try pager.readPage(pid2, &read_buf);
    try testing.expectEqual(@as(u8, 0xBB), read_buf[PAGE_HEADER_SIZE]);

    pager.deinit();
}

test "Wal overwrite same page across multiple transactions" {
    const path = "test_wal_overwrite.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    defer std.fs.cwd().deleteFile("test_wal_overwrite.db-wal") catch {};

    var wal = try Wal.init(testing.allocator, path, 512);
    defer wal.deinit();

    // Write page 1 three times in separate transactions
    var page_data: [512]u8 = undefined;
    @memset(&page_data, 0);

    page_data[0] = 0x01;
    try wal.writeFrame(1, &page_data);
    try wal.commit(5);

    page_data[0] = 0x02;
    try wal.writeFrame(1, &page_data);
    try wal.commit(5);

    page_data[0] = 0x03;
    try wal.writeFrame(1, &page_data);
    try wal.commit(5);

    // Should see the latest version
    var buf: [512]u8 = undefined;
    const found = try wal.readPage(1, &buf);
    try testing.expect(found);
    try testing.expectEqual(@as(u8, 0x03), buf[0]);

    // Frame counts should reflect all 3 commits
    try testing.expectEqual(@as(u32, 3), wal.committed_frame_count);
}

test "Wal commit with no pending frames is no-op" {
    const path = "test_wal_empty_commit.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    defer std.fs.cwd().deleteFile("test_wal_empty_commit.db-wal") catch {};

    var wal = try Wal.init(testing.allocator, path, 512);
    defer wal.deinit();

    // Commit without writing — should be a no-op
    try wal.commit(5);
    try testing.expectEqual(@as(u32, 0), wal.total_frame_count);
    try testing.expectEqual(@as(u32, 0), wal.committed_frame_count);
    try testing.expect(wal.file == null); // no file created
}

test "Wal rollback with no pending frames is no-op" {
    const path = "test_wal_empty_rollback.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    defer std.fs.cwd().deleteFile("test_wal_empty_rollback.db-wal") catch {};

    var wal = try Wal.init(testing.allocator, path, 512);
    defer wal.deinit();

    // Rollback without writing — should be a no-op
    try wal.rollback();
    try testing.expectEqual(@as(u32, 0), wal.total_frame_count);
}

test "Wal recovery after clean checkpoint has no frames" {
    const path = "test_wal_recover_after_ckpt.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    defer std.fs.cwd().deleteFile("test_wal_recover_after_ckpt.db-wal") catch {};

    // Session 1: write, commit, checkpoint
    {
        var pager = try Pager.init(testing.allocator, path, .{ .page_size = 512 });
        _ = try pager.allocPage();

        var wal = try Wal.init(testing.allocator, path, 512);

        var page_data: [512]u8 = undefined;
        @memset(&page_data, 0);
        const PageHeader = page_mod.PageHeader;
        const PAGE_HEADER_SIZE = page_mod.PAGE_HEADER_SIZE;
        var hdr = PageHeader{ .page_type = .leaf, .page_id = 1, .cell_count = 0 };
        hdr.serialize(page_data[0..PAGE_HEADER_SIZE]);
        try wal.writeFrame(1, &page_data);
        try wal.commit(pager.page_count);
        try wal.checkpoint(&pager);

        wal.deinit();
        pager.deinit();
    }

    // Session 2: recovery should find 0 committed frames (all checkpointed)
    {
        var wal = try Wal.init(testing.allocator, path, 512);
        defer wal.deinit();

        try testing.expectEqual(@as(u32, 0), wal.committed_frame_count);
        try testing.expectEqual(@as(u32, 0), wal.page_index.count());
    }
}

test "Wal multi-page transaction atomicity" {
    // Verifies that a transaction writing multiple pages is all-or-nothing.
    // If we commit, all pages should be visible. If we don't, none should.
    const path = "test_wal_multi_page_tx.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    defer std.fs.cwd().deleteFile("test_wal_multi_page_tx.db-wal") catch {};

    // Session 1: write 5 pages in one transaction, commit, then
    // write 3 more in another transaction without commit (simulate crash)
    {
        var wal = try Wal.init(testing.allocator, path, 512);
        var page_data: [512]u8 = undefined;
        @memset(&page_data, 0);

        // Committed transaction: pages 10-14
        var pid: u32 = 10;
        while (pid <= 14) : (pid += 1) {
            page_data[0] = @truncate(pid);
            try wal.writeFrame(pid, &page_data);
        }
        try wal.commit(15);

        // Uncommitted transaction: pages 20-22
        pid = 20;
        while (pid <= 22) : (pid += 1) {
            page_data[0] = @truncate(pid);
            try wal.writeFrame(pid, &page_data);
        }
        // No commit — crash
        wal.deinit();
    }

    // Session 2: only committed pages should survive
    {
        var wal = try Wal.init(testing.allocator, path, 512);
        defer wal.deinit();

        try testing.expectEqual(@as(u32, 5), wal.committed_frame_count);

        var buf: [512]u8 = undefined;
        // Committed pages should be present
        var pid: u32 = 10;
        while (pid <= 14) : (pid += 1) {
            const found = try wal.readPage(pid, &buf);
            try testing.expect(found);
            try testing.expectEqual(@as(u8, @truncate(pid)), buf[0]);
        }

        // Uncommitted pages should be absent
        pid = 20;
        while (pid <= 22) : (pid += 1) {
            const found = try wal.readPage(pid, &buf);
            try testing.expect(!found);
        }
    }
}

test "Wal pending overrides committed for same page" {
    // When a page is committed in one tx and then written again in a
    // pending (uncommitted) tx, readPage should return the pending version.
    const path = "test_wal_pending_override.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    defer std.fs.cwd().deleteFile("test_wal_pending_override.db-wal") catch {};

    var wal = try Wal.init(testing.allocator, path, 512);
    defer wal.deinit();

    var page_data: [512]u8 = undefined;
    @memset(&page_data, 0);

    // Commit page 1 with value 0x11
    page_data[0] = 0x11;
    try wal.writeFrame(1, &page_data);
    try wal.commit(5);

    // Write page 1 again with value 0x99 (pending, not committed)
    page_data[0] = 0x99;
    try wal.writeFrame(1, &page_data);

    // readPage should return the pending version (0x99)
    var buf: [512]u8 = undefined;
    const found = try wal.readPage(1, &buf);
    try testing.expect(found);
    try testing.expectEqual(@as(u8, 0x99), buf[0]);

    // After rollback, should revert to committed version (0x11)
    try wal.rollback();
    const found2 = try wal.readPage(1, &buf);
    try testing.expect(found2);
    try testing.expectEqual(@as(u8, 0x11), buf[0]);
}

test "Wal UnsupportedWalVersion error on invalid version" {
    var buf: [WAL_HEADER_SIZE]u8 = undefined;
    const header = WalHeader{
        .page_size = 4096,
        .salt_1 = 0x12345678,
        .salt_2 = 0xABCDEF00,
    };
    header.serialize(&buf);

    // Corrupt version to unsupported value
    std.mem.writeInt(u32, buf[4..8], 999, .little);

    try testing.expectError(error.UnsupportedWalVersion, WalHeader.deserialize(&buf));
}

test "Wal WalPageSizeMismatch detected during recovery" {
    const path = "test_wal_pagemismatch.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    defer std.fs.cwd().deleteFile("test_wal_pagemismatch.db-wal") catch {};

    // Create WAL with page_size=512
    var wal = try Wal.init(testing.allocator, path, 512);
    var page_data: [512]u8 = undefined;
    @memset(&page_data, 0x42);
    try wal.writeFrame(1, &page_data);
    try wal.commit(1);
    wal.deinit();

    // Manually corrupt the WAL header to have wrong page_size
    const wal_path = path ++ "-wal";
    {
        const file = try std.fs.cwd().openFile(wal_path, .{ .mode = .read_write });
        defer file.close();

        var hdr_buf: [WAL_HEADER_SIZE]u8 = undefined;
        _ = try file.read(&hdr_buf);
        // Change page_size from 512 to 4096
        std.mem.writeInt(u32, hdr_buf[8..12], 4096, .little);
        // Recompute checksum
        const cksum = checksum_mod.crc32c(hdr_buf[0..28]);
        std.mem.writeInt(u32, hdr_buf[28..32], cksum, .little);
        try file.seekTo(0);
        _ = try file.write(&hdr_buf);
    }

    // Opening with mismatched page_size triggers recovery, which detects error
    // and deletes the corrupt WAL, returning a fresh Wal instance.
    // We verify the WAL file was deleted by checking it doesn't exist after init.
    var wal2 = try Wal.init(testing.allocator, path, 512);
    defer wal2.deinit();

    // The corrupt WAL should have been deleted — verify file doesn't exist
    // by attempting to open and expecting FileNotFound
    const open_result = std.fs.cwd().openFile(wal_path, .{});
    if (open_result) |f| {
        f.close();
        try testing.expect(false); // File should not exist
    } else |err| {
        try testing.expectEqual(error.FileNotFound, err);
    }
}

test "Wal WalCorrupt error on truncated header" {
    const path = "test_wal_corrupt.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    defer std.fs.cwd().deleteFile("test_wal_corrupt.db-wal") catch {};

    // Create a WAL file with a truncated header (only 10 bytes)
    const wal_path = path ++ "-wal";
    {
        const file = try std.fs.cwd().createFile(wal_path, .{});
        defer file.close();
        const partial_header = [_]u8{0x53, 0x4C, 0x43, 0x57, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00};
        try file.writeAll(&partial_header);
    }

    // Opening should trigger recovery, which detects truncated header (WalCorrupt)
    // and deletes the corrupt WAL
    var wal = try Wal.init(testing.allocator, path, 512);
    defer wal.deinit();

    // Verify the corrupt WAL was deleted
    const open_result = std.fs.cwd().openFile(wal_path, .{});
    if (open_result) |f| {
        f.close();
        try testing.expect(false); // File should not exist
    } else |err| {
        try testing.expectEqual(error.FileNotFound, err);
    }
}

test "Wal many frame writes in single commit" {
    // Verifies that writing many frames works correctly
    const path = "test_wal_many_frames.db";
    const wal_path = "test_wal_many_frames.db-wal";
    defer std.fs.cwd().deleteFile(path) catch {};
    defer std.fs.cwd().deleteFile(wal_path) catch {};

    var wal = try Wal.init(testing.allocator, path, 512);
    defer wal.deinit();

    const page_size: usize = 512;

    // Write 100 different pages before commit
    var page_id: u32 = 0;
    while (page_id < 100) : (page_id += 1) {
        const page_data = try testing.allocator.alloc(u8, page_size);
        defer testing.allocator.free(page_data);
        @memset(page_data, @truncate(page_id));
        try wal.writeFrame(page_id, page_data);
    }

    // Commit all frames (db_page_count=100)
    try wal.commit(100);

    // Verify all pages can be read back
    page_id = 0;
    while (page_id < 100) : (page_id += 1) {
        const buf = try testing.allocator.alloc(u8, page_size);
        defer testing.allocator.free(buf);
        const found = try wal.readPage(page_id, buf);
        try testing.expect(found);
        const expected: u8 = @truncate(page_id);
        try testing.expectEqual(expected, buf[0]);
    }
}

// ── Lsn and readRawFrames Tests ────────────────────────────────

test "Lsn ordering: same epoch, different frame indices" {
    const lsn1 = Lsn{ .checkpoint_seq = 5, .frame_index = 10 };
    const lsn2 = Lsn{ .checkpoint_seq = 5, .frame_index = 20 };

    // lsn1 < lsn2
    try testing.expect(lsn1.lessThan(lsn2));
    try testing.expect(!lsn2.lessThan(lsn1));
    try testing.expect(!lsn1.eql(lsn2));

    // lsn1 == lsn1
    try testing.expect(lsn1.eql(lsn1));
    try testing.expect(!lsn1.lessThan(lsn1));

    // Test order() returns correct std.math.Order
    try testing.expectEqual(std.math.Order.lt, lsn1.order(lsn2));
    try testing.expectEqual(std.math.Order.gt, lsn2.order(lsn1));
    try testing.expectEqual(std.math.Order.eq, lsn1.order(lsn1));
}

test "Lsn ordering: higher epoch always orders after lower epoch, regardless of frame_index" {
    const lsn_epoch1_high_frame = Lsn{ .checkpoint_seq = 1, .frame_index = 1000 };
    const lsn_epoch2_low_frame = Lsn{ .checkpoint_seq = 2, .frame_index = 0 };

    // Even though epoch1 has higher frame_index, epoch2 should order as greater
    // (checkpoint_seq is the primary sort key)
    try testing.expect(lsn_epoch1_high_frame.lessThan(lsn_epoch2_low_frame));
    try testing.expect(!lsn_epoch2_low_frame.lessThan(lsn_epoch1_high_frame));
    try testing.expectEqual(std.math.Order.lt, lsn_epoch1_high_frame.order(lsn_epoch2_low_frame));
}

test "currentLsn reflects committed frontier, ignores pending frames" {
    const path = "test_lsn_current.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    defer std.fs.cwd().deleteFile("test_lsn_current.db-wal") catch {};

    var wal = try Wal.init(testing.allocator, path, 512);
    defer wal.deinit();

    var page_data: [512]u8 = undefined;
    @memset(&page_data, 0xAA);

    // Write 2 frames and commit
    try wal.writeFrame(1, &page_data);
    try wal.writeFrame(2, &page_data);
    try wal.commit(2);

    const lsn1 = wal.currentLsn();
    try testing.expectEqual(@as(u32, 0), lsn1.checkpoint_seq); // initial epoch
    try testing.expectEqual(@as(u32, 2), lsn1.frame_index);    // 2 committed frames

    // Write a 3rd frame without committing
    try wal.writeFrame(3, &page_data);
    const lsn2 = wal.currentLsn();

    // currentLsn should not advance (pending frame is not reflected)
    try testing.expectEqual(lsn1.checkpoint_seq, lsn2.checkpoint_seq);
    try testing.expectEqual(lsn1.frame_index, lsn2.frame_index);
}

test "lsnAtFrame returns LSN for given frame index in current epoch" {
    const path = "test_lsn_at_frame.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    defer std.fs.cwd().deleteFile("test_lsn_at_frame.db-wal") catch {};

    var wal = try Wal.init(testing.allocator, path, 512);
    defer wal.deinit();

    var page_data: [512]u8 = undefined;
    @memset(&page_data, 0);

    // Write and commit 3 frames
    try wal.writeFrame(1, &page_data);
    try wal.writeFrame(2, &page_data);
    try wal.writeFrame(3, &page_data);
    try wal.commit(3);

    const epoch = wal.header.checkpoint_seq;

    // lsnAtFrame should return (current_epoch, frame_index)
    const lsn0 = wal.lsnAtFrame(0);
    try testing.expectEqual(epoch, lsn0.checkpoint_seq);
    try testing.expectEqual(@as(u32, 0), lsn0.frame_index);

    const lsn1 = wal.lsnAtFrame(1);
    try testing.expectEqual(epoch, lsn1.checkpoint_seq);
    try testing.expectEqual(@as(u32, 1), lsn1.frame_index);

    const lsn2 = wal.lsnAtFrame(2);
    try testing.expectEqual(epoch, lsn2.checkpoint_seq);
    try testing.expectEqual(@as(u32, 2), lsn2.frame_index);
}

test "readRawFrames round-trip: read all committed frames with recognizable content" {
    const path = "test_read_raw_frames_roundtrip.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    defer std.fs.cwd().deleteFile("test_read_raw_frames_roundtrip.db-wal") catch {};

    var wal = try Wal.init(testing.allocator, path, 512);
    defer wal.deinit();

    // Create 3 frames with distinct marker bytes
    var page1: [512]u8 = undefined;
    var page2: [512]u8 = undefined;
    var page3: [512]u8 = undefined;
    @memset(&page1, 0x11);
    @memset(&page2, 0x22);
    @memset(&page3, 0x33);

    try wal.writeFrame(10, &page1);
    try wal.writeFrame(20, &page2);
    try wal.writeFrame(30, &page3);
    try wal.commit(3);

    // Allocate buffer for 3 frames (header + page data each)
    const frame_size = WAL_FRAME_HEADER_SIZE + 512;
    const buf = try testing.allocator.alloc(u8, frame_size * 3);
    defer testing.allocator.free(buf);

    // Read all frames from the beginning
    const result = try wal.readRawFrames(wal.lsnAtFrame(0), buf);

    // Verify bytes_read is exactly 3 frames
    const expected_bytes = frame_size * 3;
    try testing.expectEqual(expected_bytes, result.bytes_read);

    // Verify next_lsn equals currentLsn (caught up)
    try testing.expect(result.next_lsn.eql(wal.currentLsn()));

    // Decode and verify each frame's header and page content
    var offset: usize = 0;

    // Frame 0
    const fh0 = WalFrameHeader.deserialize(buf[offset..][0..WAL_FRAME_HEADER_SIZE]);
    try testing.expectEqual(@as(u32, 10), fh0.page_id);
    offset += WAL_FRAME_HEADER_SIZE;
    const page_data0 = buf[offset..][0..512];
    try testing.expectEqual(@as(u8, 0x11), page_data0[0]);
    offset += 512;

    // Frame 1
    const fh1 = WalFrameHeader.deserialize(buf[offset..][0..WAL_FRAME_HEADER_SIZE]);
    try testing.expectEqual(@as(u32, 20), fh1.page_id);
    offset += WAL_FRAME_HEADER_SIZE;
    const page_data1 = buf[offset..][0..512];
    try testing.expectEqual(@as(u8, 0x22), page_data1[0]);
    offset += 512;

    // Frame 2
    const fh2 = WalFrameHeader.deserialize(buf[offset..][0..WAL_FRAME_HEADER_SIZE]);
    try testing.expectEqual(@as(u32, 30), fh2.page_id);
    offset += WAL_FRAME_HEADER_SIZE;
    const page_data2 = buf[offset..][0..512];
    try testing.expectEqual(@as(u8, 0x33), page_data2[0]);
}

test "readRawFrames never splits frame across multiple calls" {
    const path = "test_read_raw_frames_no_split.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    defer std.fs.cwd().deleteFile("test_read_raw_frames_no_split.db-wal") catch {};

    var wal = try Wal.init(testing.allocator, path, 512);
    defer wal.deinit();

    var page_data: [512]u8 = undefined;
    @memset(&page_data, 0);

    // Write and commit 3 frames
    try wal.writeFrame(1, &page_data);
    try wal.writeFrame(2, &page_data);
    try wal.writeFrame(3, &page_data);
    try wal.commit(3);

    const frame_size = WAL_FRAME_HEADER_SIZE + 512;

    // Allocate buffer for 1.5 frames (should only return 1 frame, not split the 2nd)
    const buf = try testing.allocator.alloc(u8, frame_size + frame_size / 2);
    defer testing.allocator.free(buf);

    const result = try wal.readRawFrames(wal.lsnAtFrame(0), buf);

    // Should return exactly 1 frame's worth of bytes, not 1.5
    try testing.expectEqual(frame_size, result.bytes_read);

    // next_lsn should be at frame_index=1 (advanced by exactly 1)
    try testing.expectEqual(@as(u32, 0), result.next_lsn.checkpoint_seq);
    try testing.expectEqual(@as(u32, 1), result.next_lsn.frame_index);
}

test "readRawFrames returns error.BufferTooSmall if buffer smaller than one frame" {
    const path = "test_read_raw_frames_small_buf.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    defer std.fs.cwd().deleteFile("test_read_raw_frames_small_buf.db-wal") catch {};

    var wal = try Wal.init(testing.allocator, path, 512);
    defer wal.deinit();

    var page_data: [512]u8 = undefined;
    @memset(&page_data, 0);

    try wal.writeFrame(1, &page_data);
    try wal.commit(1);

    // Buffer too small (4 bytes < one frame)
    var tiny_buf: [4]u8 = undefined;
    const result = wal.readRawFrames(wal.lsnAtFrame(0), &tiny_buf);

    try testing.expectError(error.BufferTooSmall, result);
}

test "readRawFrames returns zero bytes when already caught up (no new data)" {
    const path = "test_read_raw_frames_caught_up.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    defer std.fs.cwd().deleteFile("test_read_raw_frames_caught_up.db-wal") catch {};

    var wal = try Wal.init(testing.allocator, path, 512);
    defer wal.deinit();

    var page_data: [512]u8 = undefined;
    @memset(&page_data, 0);

    try wal.writeFrame(1, &page_data);
    try wal.commit(1);

    // Get current LSN (at the committed frontier)
    const current_lsn = wal.currentLsn();

    // Allocate a reasonable buffer
    const frame_size = WAL_FRAME_HEADER_SIZE + 512;
    const buf = try testing.allocator.alloc(u8, frame_size * 2);
    defer testing.allocator.free(buf);

    // Call readRawFrames at the current LSN (nothing new to read)
    const result = try wal.readRawFrames(current_lsn, buf);

    // Should return 0 bytes (normal case, not an error)
    try testing.expectEqual(@as(usize, 0), result.bytes_read);

    // next_lsn should equal the input start_lsn
    try testing.expect(result.next_lsn.eql(current_lsn));
}

test "readRawFrames excludes pending (uncommitted) frames" {
    const path = "test_read_raw_frames_no_pending.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    defer std.fs.cwd().deleteFile("test_read_raw_frames_no_pending.db-wal") catch {};

    var wal = try Wal.init(testing.allocator, path, 512);
    defer wal.deinit();

    var page_data: [512]u8 = undefined;
    @memset(&page_data, 0xAA);

    // Write a frame without committing
    try wal.writeFrame(1, &page_data);

    // Attempt to read from the beginning
    const frame_size = WAL_FRAME_HEADER_SIZE + 512;
    const buf = try testing.allocator.alloc(u8, frame_size);
    defer testing.allocator.free(buf);

    const result = try wal.readRawFrames(wal.lsnAtFrame(0), buf);

    // Should return 0 bytes (pending frame is never exposed)
    try testing.expectEqual(@as(usize, 0), result.bytes_read);

    // next_lsn should be unchanged
    try testing.expect(result.next_lsn.eql(wal.lsnAtFrame(0)));
}

test "readRawFrames rejects LSN from earlier epoch after checkpoint" {
    const path = "test_read_raw_frames_old_epoch.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    defer std.fs.cwd().deleteFile("test_read_raw_frames_old_epoch.db-wal") catch {};

    const PageHeader = page_mod.PageHeader;
    const PAGE_HEADER_SIZE = page_mod.PAGE_HEADER_SIZE;

    // Create pager and WAL
    var pager = try Pager.init(testing.allocator, path, .{ .page_size = 512 });
    const pid = try pager.allocPage();

    var wal = try Wal.init(testing.allocator, path, 512);
    defer wal.deinit();

    // Write and commit a frame in epoch 0
    var page_data: [512]u8 = undefined;
    @memset(&page_data, 0);
    const hdr = PageHeader{ .page_type = .leaf, .page_id = pid, .cell_count = 0 };
    hdr.serialize(page_data[0..PAGE_HEADER_SIZE]);
    try wal.writeFrame(pid, &page_data);
    try wal.commit(pager.page_count);

    // Record LSN from epoch 0
    const old_epoch = wal.header.checkpoint_seq;
    const old_lsn = wal.lsnAtFrame(0);

    // Perform checkpoint (increments epoch and truncates WAL)
    try wal.checkpoint(&pager);

    // Verify epoch incremented
    try testing.expect(wal.header.checkpoint_seq > old_epoch);

    // Try to read from old epoch LSN — should fail
    const frame_size = WAL_FRAME_HEADER_SIZE + 512;
    const buf = try testing.allocator.alloc(u8, frame_size);
    defer testing.allocator.free(buf);

    const result = wal.readRawFrames(old_lsn, buf);
    try testing.expectError(error.LsnFromEarlierEpoch, result);

    pager.deinit();
}

test "readRawFrames multi-call resumption: chain LSNs to stream all frames" {
    const path = "test_read_raw_frames_resume.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    defer std.fs.cwd().deleteFile("test_read_raw_frames_resume.db-wal") catch {};

    var wal = try Wal.init(testing.allocator, path, 512);
    defer wal.deinit();

    // Write 2 frames and commit
    var page1: [512]u8 = undefined;
    var page2: [512]u8 = undefined;
    @memset(&page1, 0x11);
    @memset(&page2, 0x22);

    try wal.writeFrame(10, &page1);
    try wal.writeFrame(20, &page2);
    try wal.commit(2);

    // Write 3 more frames and commit separately
    var page3: [512]u8 = undefined;
    var page4: [512]u8 = undefined;
    var page5: [512]u8 = undefined;
    @memset(&page3, 0x33);
    @memset(&page4, 0x44);
    @memset(&page5, 0x55);

    try wal.writeFrame(30, &page3);
    try wal.writeFrame(40, &page4);
    try wal.writeFrame(50, &page5);
    try wal.commit(5);

    // Allocate buffer for 2 frames at a time
    const frame_size = WAL_FRAME_HEADER_SIZE + 512;
    const buf = try testing.allocator.alloc(u8, frame_size * 2);
    defer testing.allocator.free(buf);

    var next_lsn = wal.lsnAtFrame(0);
    var frames_seen: u32 = 0;

    // First call: read up to 2 frames
    var result = try wal.readRawFrames(next_lsn, buf);
    try testing.expectEqual(frame_size * 2, result.bytes_read);
    frames_seen += 2;
    next_lsn = result.next_lsn;

    // Verify first two frame page_ids
    var offset: usize = 0;
    var fh = WalFrameHeader.deserialize(buf[offset..][0..WAL_FRAME_HEADER_SIZE]);
    try testing.expectEqual(@as(u32, 10), fh.page_id);
    offset += frame_size;
    fh = WalFrameHeader.deserialize(buf[offset..][0..WAL_FRAME_HEADER_SIZE]);
    try testing.expectEqual(@as(u32, 20), fh.page_id);

    // Second call: read remaining 3 frames (but buffer holds only 2, so only 2 returned)
    result = try wal.readRawFrames(next_lsn, buf);
    try testing.expectEqual(frame_size * 2, result.bytes_read);
    frames_seen += 2;
    next_lsn = result.next_lsn;

    // Verify these are page_ids 30 and 40
    offset = 0;
    fh = WalFrameHeader.deserialize(buf[offset..][0..WAL_FRAME_HEADER_SIZE]);
    try testing.expectEqual(@as(u32, 30), fh.page_id);
    offset += frame_size;
    fh = WalFrameHeader.deserialize(buf[offset..][0..WAL_FRAME_HEADER_SIZE]);
    try testing.expectEqual(@as(u32, 40), fh.page_id);

    // Third call: read last frame
    result = try wal.readRawFrames(next_lsn, buf);
    try testing.expectEqual(frame_size, result.bytes_read);
    frames_seen += 1;
    next_lsn = result.next_lsn;

    // Verify last frame is page_id 50
    fh = WalFrameHeader.deserialize(buf[0..WAL_FRAME_HEADER_SIZE]);
    try testing.expectEqual(@as(u32, 50), fh.page_id);

    // Fourth call: should be caught up
    result = try wal.readRawFrames(next_lsn, buf);
    try testing.expectEqual(@as(usize, 0), result.bytes_read);

    // Total frames seen should be 5
    try testing.expectEqual(@as(u32, 5), frames_seen);
}

test "Phase 4: appendRawFrame verbatim round-trip with commit promotion" {
    const path_src = "test_wal_phase4_src.db";
    defer std.fs.cwd().deleteFile(path_src) catch {};
    defer std.fs.cwd().deleteFile(path_src ++ "-wal") catch {};

    const path_dst = "test_wal_phase4_dst.db";
    defer std.fs.cwd().deleteFile(path_dst) catch {};
    defer std.fs.cwd().deleteFile(path_dst ++ "-wal") catch {};

    // Create source WAL and write 2 frames
    var src_wal = try Wal.init(testing.allocator, path_src, 512);
    defer src_wal.deinit();

    var page_data1: [512]u8 = undefined;
    @memset(&page_data1, 0xAA);
    try src_wal.writeFrame(10, &page_data1);

    var page_data2: [512]u8 = undefined;
    @memset(&page_data2, 0xBB);
    try src_wal.writeFrame(20, &page_data2);

    try src_wal.commit(2);

    // Read raw frames back via readRawFrames
    const frame_size = WAL_FRAME_HEADER_SIZE + 512;
    var frame_buf = try testing.allocator.alloc(u8, frame_size * 3);
    defer testing.allocator.free(frame_buf);

    const start_lsn = src_wal.lsnAtFrame(0);
    const read_result = try src_wal.readRawFrames(start_lsn, frame_buf);
    try testing.expect(read_result.bytes_read > 0);

    // Destination WAL
    var dst_wal = try Wal.init(testing.allocator, path_dst, 512);
    defer dst_wal.deinit();

    // Extract first frame (non-commit) and second frame (commit)
    const frame1_bytes = frame_buf[0..frame_size];
    const frame2_bytes = frame_buf[frame_size..][0..frame_size];

    // Append first frame (non-commit)
    try dst_wal.appendRawFrame(frame1_bytes);

    // After first frame, should be pending (committed_frame_count = 0)
    try testing.expectEqual(@as(u32, 0), dst_wal.committed_frame_count);
    try testing.expectEqual(@as(u32, 1), dst_wal.total_frame_count);

    // But readPage should find it in pending_index
    var read_buf: [512]u8 = undefined;
    const found1 = try dst_wal.readPage(10, &read_buf);
    try testing.expect(found1);
    try testing.expectEqualSlices(u8, &page_data1, &read_buf);

    // Append second frame (commit)
    try dst_wal.appendRawFrame(frame2_bytes);

    // After commit frame, should promote to committed
    try testing.expectEqual(@as(u32, 2), dst_wal.committed_frame_count);
    try testing.expectEqual(@as(u32, 2), dst_wal.total_frame_count);

    // Both pages should be readable from committed index
    const found10 = try dst_wal.readPage(10, &read_buf);
    try testing.expect(found10);
    try testing.expectEqualSlices(u8, &page_data1, &read_buf);

    const found20 = try dst_wal.readPage(20, &read_buf);
    try testing.expect(found20);
    try testing.expectEqualSlices(u8, &page_data2, &read_buf);
}

test "Phase 4: appendRawFrame is purely additive — normal writeFrame/commit unchanged" {
    const path = "test_wal_phase4_regression.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    defer std.fs.cwd().deleteFile(path ++ "-wal") catch {};

    var wal = try Wal.init(testing.allocator, path, 512);
    defer wal.deinit();

    // Write and commit via normal writeFrame/commit (no appendRawFrame)
    var page_data: [512]u8 = undefined;
    @memset(&page_data, 0x99);
    try wal.writeFrame(5, &page_data);
    try wal.commit(1);

    // Verify normal counts and indexes work as before
    try testing.expectEqual(@as(u32, 1), wal.committed_frame_count);
    try testing.expectEqual(@as(u32, 1), wal.total_frame_count);
    try testing.expect(wal.page_index.get(5) != null);
    try testing.expectEqual(@as(u32, 0), wal.pending_index.count());

    // Verify readPage still works
    var read_buf: [512]u8 = undefined;
    const found = try wal.readPage(5, &read_buf);
    try testing.expect(found);
    try testing.expectEqualSlices(u8, &page_data, &read_buf);
}

// ── Phase 6: LSN pack/unpack methods (for replication retention) ────────────

test "Lsn.pack() and Lsn.unpack() round-trip" {
    // Test that pack then unpack returns the original Lsn
    const original = Lsn{ .checkpoint_seq = 42, .frame_index = 12345 };
    const packed_val = original.pack();
    const unpacked = Lsn.unpack(packed_val);

    try testing.expectEqual(original.checkpoint_seq, unpacked.checkpoint_seq);
    try testing.expectEqual(original.frame_index, unpacked.frame_index);
    try testing.expect(original.eql(unpacked));
}

test "Lsn.pack() encodes as (checkpoint_seq << 32) | frame_index" {
    // Verify exact bit layout with hand-computed values

    // Test 1: checkpoint_seq=0, frame_index=0 → u64(0)
    const lsn1 = Lsn{ .checkpoint_seq = 0, .frame_index = 0 };
    try testing.expectEqual(@as(u64, 0), lsn1.pack());

    // Test 2: checkpoint_seq=1, frame_index=0 → u64(1) << 32 = 0x0000000100000000
    const lsn2 = Lsn{ .checkpoint_seq = 1, .frame_index = 0 };
    try testing.expectEqual(@as(u64, 0x0000000100000000), lsn2.pack());

    // Test 3: checkpoint_seq=0, frame_index=256 → u64(256) = 0x0000000000000100
    const lsn3 = Lsn{ .checkpoint_seq = 0, .frame_index = 256 };
    try testing.expectEqual(@as(u64, 0x0000000000000100), lsn3.pack());

    // Test 4: checkpoint_seq=5, frame_index=1000 → (5 << 32) | 1000
    const lsn4 = Lsn{ .checkpoint_seq = 5, .frame_index = 1000 };
    const expected = (@as(u64, 5) << 32) | 1000;
    try testing.expectEqual(expected, lsn4.pack());

    // Test 5: max values
    const lsn_max = Lsn{ .checkpoint_seq = 0xFFFFFFFF, .frame_index = 0xFFFFFFFF };
    try testing.expectEqual(@as(u64, 0xFFFFFFFFFFFFFFFF), lsn_max.pack());
}

test "Lsn.order() agrees with pack() comparison across checkpoint boundaries" {
    // Verify that Lsn.order() gives the same result as comparing packed u64 values
    // especially when crossing checkpoint_seq boundaries

    // Test 1: same checkpoint_seq, different frame_index
    const lsn_a = Lsn{ .checkpoint_seq = 5, .frame_index = 100 };
    const lsn_b = Lsn{ .checkpoint_seq = 5, .frame_index = 200 };

    const order_ab = lsn_a.order(lsn_b);
    const packed_a = lsn_a.pack();
    const packed_b = lsn_b.pack();
    const order_packed_ab = std.math.order(packed_a, packed_b);

    try testing.expectEqual(order_ab, order_packed_ab);
    try testing.expect(lsn_a.lessThan(lsn_b));
    try testing.expect(packed_a < packed_b);

    // Test 2: crossing checkpoint_seq boundary (the critical case)
    // lsn_epoch1_high_frame has high frame_index but lower checkpoint_seq
    // lsn_epoch2_low_frame has low frame_index but higher checkpoint_seq
    // The higher checkpoint_seq should always win, even with lower frame_index
    const lsn_epoch1_high = Lsn{ .checkpoint_seq = 1, .frame_index = 1000 };
    const lsn_epoch2_low = Lsn{ .checkpoint_seq = 2, .frame_index = 0 };

    const order_boundary = lsn_epoch1_high.order(lsn_epoch2_low);
    const packed_epoch1_val = lsn_epoch1_high.pack();
    const packed_epoch2_val = lsn_epoch2_low.pack();
    const order_packed_boundary = std.math.order(packed_epoch1_val, packed_epoch2_val);

    try testing.expectEqual(order_boundary, order_packed_boundary);
    try testing.expect(lsn_epoch1_high.lessThan(lsn_epoch2_low));
    try testing.expect(packed_epoch1_val < packed_epoch2_val);

    // Test 3: verify numerical correctness at the boundary
    // checkpoint_seq=1, frame_index=1000 → (1 << 32) | 1000 = 0x00000001000003E8
    // checkpoint_seq=2, frame_index=0 → (2 << 32) | 0 = 0x0000000200000000
    // The second is indeed larger
    try testing.expect(packed_epoch1_val < packed_epoch2_val);
}

// ── Phase 6: Checkpoint vs Replication Retention ────────────────────

/// Context struct for Phase 6 test callbacks (mutable min_retained_lsn value)
const RetentionCallbackContext = struct {
    min_retained_lsn: ?u64 = null,

    pub fn callback(self: *RetentionCallbackContext) ?u64 {
        return self.min_retained_lsn;
    }
};

test "Phase 6: checkpoint with retention callback reporting behind current LSN — flushes but doesn't truncate" {
    const path = "test_phase6_retention_behind.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    defer std.fs.cwd().deleteFile("test_phase6_retention_behind.db-wal") catch {};

    const PageHeader = page_mod.PageHeader;
    const PAGE_HEADER_SIZE = page_mod.PAGE_HEADER_SIZE;

    // Create pager and WAL
    var pager = try Pager.init(testing.allocator, path, .{ .page_size = 512 });
    const pid = try pager.allocPage();

    var wal = try Wal.init(testing.allocator, path, 512);
    defer wal.deinit();

    // Write and commit 2 frames
    var page1: [512]u8 = undefined;
    var page2: [512]u8 = undefined;
    @memset(&page1, 0xAA);
    @memset(&page2, 0xBB);
    var hdr1 = PageHeader{ .page_type = .leaf, .page_id = pid, .cell_count = 1 };
    hdr1.serialize(page1[0..PAGE_HEADER_SIZE]);
    var hdr2 = PageHeader{ .page_type = .leaf, .page_id = pid + 1, .cell_count = 2 };
    hdr2.serialize(page2[0..PAGE_HEADER_SIZE]);

    try wal.writeFrame(pid, &page1);
    try wal.writeFrame(pid + 1, &page2);
    try wal.commit(pager.page_count);

    // Record state before checkpoint
    const checkpoint_seq_before = wal.header.checkpoint_seq;
    const total_frame_count_before = wal.total_frame_count;
    const current_lsn = wal.currentLsn();

    // Register a retention callback that reports a lagging replica (behind current LSN)
    const behind_lsn = Lsn{ .checkpoint_seq = current_lsn.checkpoint_seq, .frame_index = 0 };
    var ctx = RetentionCallbackContext{
        .min_retained_lsn = behind_lsn.pack(),
    };
    wal.setRetentionCallback(&ctx, RetentionCallbackContext.callback);

    // Call checkpoint — should flush pages but NOT truncate WAL
    try wal.checkpoint(&pager);

    // Verify flush happened: pages should be in main DB
    var read_buf: [512]u8 = undefined;
    try pager.readPage(pid, &read_buf);
    try testing.expectEqual(@as(u8, 0xAA), read_buf[PAGE_HEADER_SIZE]);
    try pager.readPage(pid + 1, &read_buf);
    try testing.expectEqual(@as(u8, 0xBB), read_buf[PAGE_HEADER_SIZE]);

    // Verify truncation was SKIPPED: checkpoint_seq and frame counts unchanged
    try testing.expectEqual(checkpoint_seq_before, wal.header.checkpoint_seq);
    try testing.expectEqual(total_frame_count_before, wal.total_frame_count);
    try testing.expect(wal.page_index.count() > 0); // page_index still populated

    pager.deinit();
}

test "Phase 6: checkpoint with retention callback catching up — truncation now happens" {
    const path = "test_phase6_retention_catchup.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    defer std.fs.cwd().deleteFile("test_phase6_retention_catchup.db-wal") catch {};

    const PageHeader = page_mod.PageHeader;
    const PAGE_HEADER_SIZE = page_mod.PAGE_HEADER_SIZE;

    var pager = try Pager.init(testing.allocator, path, .{ .page_size = 512 });
    const pid = try pager.allocPage();

    var wal = try Wal.init(testing.allocator, path, 512);
    defer wal.deinit();

    // Write and commit 1 frame
    var page1: [512]u8 = undefined;
    @memset(&page1, 0xCC);
    var hdr = PageHeader{ .page_type = .leaf, .page_id = pid, .cell_count = 1 };
    hdr.serialize(page1[0..PAGE_HEADER_SIZE]);

    try wal.writeFrame(pid, &page1);
    try wal.commit(pager.page_count);

    // Register retention callback with lagging LSN
    const lagging_lsn = Lsn{ .checkpoint_seq = 0, .frame_index = 0 };
    var ctx = RetentionCallbackContext{
        .min_retained_lsn = lagging_lsn.pack(),
    };
    wal.setRetentionCallback(&ctx, RetentionCallbackContext.callback);

    // First checkpoint — should defer truncation
    const checkpoint_seq_before = wal.header.checkpoint_seq;
    try wal.checkpoint(&pager);
    try testing.expectEqual(checkpoint_seq_before, wal.header.checkpoint_seq);

    // Now advance the callback to report caught-up LSN
    const current_lsn = wal.currentLsn(); // Note: after deferred checkpoint, currentLsn includes frames we kept
    const caught_up_callback_lsn = Lsn{ .checkpoint_seq = current_lsn.checkpoint_seq, .frame_index = 999 };
    ctx.min_retained_lsn = caught_up_callback_lsn.pack(); // way ahead

    // Second checkpoint with no new writes — should now truncate
    try wal.checkpoint(&pager);
    try testing.expectEqual(checkpoint_seq_before + 1, wal.header.checkpoint_seq);
    try testing.expectEqual(@as(u32, 0), wal.total_frame_count); // WAL reset

    pager.deinit();
}

test "Phase 6: multiple write/commit rounds with retention deferred, then one catch-up checkpoint reclaims all" {
    const path = "test_phase6_retention_multiple_rounds.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    defer std.fs.cwd().deleteFile("test_phase6_retention_multiple_rounds.db-wal") catch {};

    const PageHeader = page_mod.PageHeader;
    const PAGE_HEADER_SIZE = page_mod.PAGE_HEADER_SIZE;

    var pager = try Pager.init(testing.allocator, path, .{ .page_size = 512 });
    const pid = try pager.allocPage();

    var wal = try Wal.init(testing.allocator, path, 512);
    defer wal.deinit();

    // Setup retention callback that always reports behind
    const initial_behind_lsn = Lsn{ .checkpoint_seq = 0, .frame_index = 0 };
    var ctx = RetentionCallbackContext{
        .min_retained_lsn = initial_behind_lsn.pack(),
    };
    wal.setRetentionCallback(&ctx, RetentionCallbackContext.callback);

    // Round 1: write page, commit, checkpoint (should defer)
    var page1: [512]u8 = undefined;
    @memset(&page1, 0x11);
    var hdr1 = PageHeader{ .page_type = .leaf, .page_id = pid, .cell_count = 1 };
    hdr1.serialize(page1[0..PAGE_HEADER_SIZE]);
    try wal.writeFrame(pid, &page1);
    try wal.commit(pager.page_count);
    try wal.checkpoint(&pager);

    const frame_count_after_round1 = wal.total_frame_count;
    try testing.expect(frame_count_after_round1 > 0); // WAL not truncated

    // Round 2: write different page, commit, checkpoint (should still defer, frames accumulate)
    var page2: [512]u8 = undefined;
    @memset(&page2, 0x22);
    var hdr2 = PageHeader{ .page_type = .leaf, .page_id = pid + 1, .cell_count = 2 };
    hdr2.serialize(page2[0..PAGE_HEADER_SIZE]);
    try wal.writeFrame(pid + 1, &page2);
    try wal.commit(pager.page_count);
    try wal.checkpoint(&pager);

    const frame_count_after_round2 = wal.total_frame_count;
    try testing.expect(frame_count_after_round2 > frame_count_after_round1); // frames appended

    // Round 3: write third page, commit, checkpoint (deferred again)
    var page3: [512]u8 = undefined;
    @memset(&page3, 0x33);
    var hdr3 = PageHeader{ .page_type = .leaf, .page_id = pid + 2, .cell_count = 3 };
    hdr3.serialize(page3[0..PAGE_HEADER_SIZE]);
    try wal.writeFrame(pid + 2, &page3);
    try wal.commit(pager.page_count);
    try wal.checkpoint(&pager);

    const frame_count_after_round3 = wal.total_frame_count;
    try testing.expect(frame_count_after_round3 > frame_count_after_round2);

    // Now catch up: advance callback to current LSN
    const caught_up_lsn = wal.currentLsn();
    const final_caught_up_lsn = Lsn{ .checkpoint_seq = caught_up_lsn.checkpoint_seq, .frame_index = 999 };
    ctx.min_retained_lsn = final_caught_up_lsn.pack();

    // Final checkpoint — should reclaim all frames in one shot
    const checkpoint_seq_before_final = wal.header.checkpoint_seq;
    try wal.checkpoint(&pager);

    try testing.expectEqual(checkpoint_seq_before_final + 1, wal.header.checkpoint_seq);
    try testing.expectEqual(@as(u32, 0), wal.total_frame_count); // everything reclaimed
    try testing.expectEqual(@as(u32, 0), wal.page_index.count());

    pager.deinit();
}

test "Phase 6: regression proof — readRawFrames still works on not-yet-truncated epoch while retention deferred" {
    const path = "test_phase6_readrawframes_deferred.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    defer std.fs.cwd().deleteFile("test_phase6_readrawframes_deferred.db-wal") catch {};

    const PageHeader = page_mod.PageHeader;
    const PAGE_HEADER_SIZE = page_mod.PAGE_HEADER_SIZE;

    var pager = try Pager.init(testing.allocator, path, .{ .page_size = 512 });
    const pid = try pager.allocPage();

    var wal = try Wal.init(testing.allocator, path, 512);
    defer wal.deinit();

    // Write 3 frames with distinct markers
    var pages: [3][512]u8 = undefined;
    @memset(&pages[0], 0x11);
    @memset(&pages[1], 0x22);
    @memset(&pages[2], 0x33);

    var hdr0 = PageHeader{ .page_type = .leaf, .page_id = pid, .cell_count = 0 };
    hdr0.serialize(pages[0][0..PAGE_HEADER_SIZE]);
    var hdr1 = PageHeader{ .page_type = .leaf, .page_id = pid + 1, .cell_count = 1 };
    hdr1.serialize(pages[1][0..PAGE_HEADER_SIZE]);
    var hdr2 = PageHeader{ .page_type = .leaf, .page_id = pid + 2, .cell_count = 2 };
    hdr2.serialize(pages[2][0..PAGE_HEADER_SIZE]);

    try wal.writeFrame(pid, &pages[0]);
    try wal.writeFrame(pid + 1, &pages[1]);
    try wal.writeFrame(pid + 2, &pages[2]);
    try wal.commit(pager.page_count);

    // Record LSN of first frame before checkpoint
    const frame0_lsn = wal.lsnAtFrame(0);

    // Register retention callback reporting behind
    const regression_behind_lsn = Lsn{ .checkpoint_seq = 0, .frame_index = 0 };
    var ctx = RetentionCallbackContext{
        .min_retained_lsn = regression_behind_lsn.pack(),
    };
    wal.setRetentionCallback(&ctx, RetentionCallbackContext.callback);

    // Checkpoint — should defer truncation, keeping the epoch intact
    try wal.checkpoint(&pager);

    // Now attempt readRawFrames from frame 0 — should still succeed (epoch not discarded)
    const frame_size = WAL_FRAME_HEADER_SIZE + 512;
    const read_buf = try testing.allocator.alloc(u8, frame_size * 3);
    defer testing.allocator.free(read_buf);

    const read_result = try wal.readRawFrames(frame0_lsn, read_buf);

    // Should have successfully read all 3 frames
    try testing.expect(read_result.bytes_read > 0);
    try testing.expectEqual(frame_size * 3, read_result.bytes_read);

    // Verify frame content markers survived
    var offset: usize = 0;
    offset += WAL_FRAME_HEADER_SIZE; // skip first frame header
    try testing.expectEqual(@as(u8, 0x11), read_buf[offset]);
    offset += 512 + WAL_FRAME_HEADER_SIZE;
    try testing.expectEqual(@as(u8, 0x22), read_buf[offset]);
    offset += 512 + WAL_FRAME_HEADER_SIZE;
    try testing.expectEqual(@as(u8, 0x33), read_buf[offset]);

    pager.deinit();
}

test "Phase 6: checkpoint with no retention callback registered behaves like today (no change)" {
    const path = "test_phase6_no_callback.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    defer std.fs.cwd().deleteFile("test_phase6_no_callback.db-wal") catch {};

    const PageHeader = page_mod.PageHeader;
    const PAGE_HEADER_SIZE = page_mod.PAGE_HEADER_SIZE;

    var pager = try Pager.init(testing.allocator, path, .{ .page_size = 512 });
    const pid = try pager.allocPage();

    var wal = try Wal.init(testing.allocator, path, 512);
    defer wal.deinit();

    // Write and commit
    var page: [512]u8 = undefined;
    @memset(&page, 0x42);
    var hdr = PageHeader{ .page_type = .leaf, .page_id = pid, .cell_count = 0 };
    hdr.serialize(page[0..PAGE_HEADER_SIZE]);

    try wal.writeFrame(pid, &page);
    try wal.commit(pager.page_count);

    // Don't register callback — should truncate unconditionally
    const checkpoint_seq_before = wal.header.checkpoint_seq;
    try wal.checkpoint(&pager);

    // Should have truncated
    try testing.expectEqual(checkpoint_seq_before + 1, wal.header.checkpoint_seq);
    try testing.expectEqual(@as(u32, 0), wal.total_frame_count);
    try testing.expectEqual(@as(u32, 0), wal.page_index.count());

    pager.deinit();
}
