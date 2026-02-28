//! Buffer Pool — LRU page cache with pin/unpin semantics and dirty tracking.
//!
//! The buffer pool sits between the B+Tree (and other consumers) and the Pager,
//! caching frequently accessed pages in memory. Pages are loaded on demand and
//! evicted using an LRU policy when the pool is full.
//!
//! Key invariants:
//!   - A pinned page (pin_count > 0) is never evicted.
//!   - A dirty page is flushed to disk before eviction.
//!   - Each page appears at most once in the pool.

const std = @import("std");
const page_mod = @import("page.zig");
const Pager = page_mod.Pager;
const PageHeader = page_mod.PageHeader;
const PAGE_HEADER_SIZE = page_mod.PAGE_HEADER_SIZE;
const wal_mod = @import("../tx/wal.zig");
const Wal = wal_mod.Wal;

// ── Configuration ────────────────────────────────────────────────────

pub const DEFAULT_POOL_SIZE: u32 = 2000;

// ── Buffer Frame ─────────────────────────────────────────────────────

/// A single frame in the buffer pool, holding one cached page.
pub const BufferFrame = struct {
    /// The page data buffer (page_size bytes, owned by the pool).
    data: []u8,
    /// Which page this frame holds (0 = unused sentinel).
    page_id: u32,
    /// Number of active pins. Must reach 0 before eviction.
    pin_count: u32 = 0,
    /// Whether the page has been modified since last flush.
    is_dirty: bool = false,
    /// Intrusive doubly-linked list pointers for LRU chain.
    prev: ?*BufferFrame = null,
    next: ?*BufferFrame = null,

    /// Mark the frame as dirty (will be written back on eviction/flush).
    pub fn markDirty(self: *BufferFrame) void {
        self.is_dirty = true;
    }

    /// Get the page header from this frame's data.
    pub fn getHeader(self: *const BufferFrame) PageHeader {
        return PageHeader.deserialize(self.data[0..PAGE_HEADER_SIZE]);
    }
};

// ── Buffer Pool ──────────────────────────────────────────────────────

pub const BufferPool = struct {
    /// The underlying pager for disk I/O.
    pager: *Pager,
    /// Allocator used for frame data buffers and internal structures.
    allocator: std.mem.Allocator,
    /// Maximum number of frames in the pool.
    capacity: u32,
    /// All frames (pre-allocated array).
    frames: []BufferFrame,
    /// Map from page_id → frame index for O(1) lookup.
    page_map: std.AutoHashMap(u32, u32),
    /// Number of frames currently in use.
    frame_count: u32 = 0,

    // LRU doubly-linked list: head = most recently used, tail = least recently used.
    // Only unpinned frames are in the LRU list.
    lru_head: ?*BufferFrame = null,
    lru_tail: ?*BufferFrame = null,
    lru_size: u32 = 0,

    // Stats
    hits: u64 = 0,
    misses: u64 = 0,

    // Optional WAL for write-ahead logging. When set, dirty page flushes
    // go through the WAL instead of directly to the Pager.
    wal: ?*Wal = null,

    /// Set the WAL for write-ahead logging. When set, dirty page writes
    /// go through the WAL instead of directly to the Pager.
    pub fn setWal(self: *BufferPool, w: *Wal) void {
        self.wal = w;
    }

    pub fn init(allocator: std.mem.Allocator, pager: *Pager, capacity: u32) !BufferPool {
        const cap = if (capacity == 0) DEFAULT_POOL_SIZE else capacity;

        const frames = try allocator.alloc(BufferFrame, cap);
        for (frames) |*frame| {
            frame.* = BufferFrame{
                .data = try allocator.alloc(u8, pager.page_size),
                .page_id = 0,
            };
        }

        return BufferPool{
            .pager = pager,
            .allocator = allocator,
            .capacity = cap,
            .frames = frames,
            .page_map = std.AutoHashMap(u32, u32).init(allocator),
        };
    }

    pub fn deinit(self: *BufferPool) void {
        // Flush all dirty pages before teardown
        self.flushAll() catch {};

        for (self.frames) |*frame| {
            self.allocator.free(frame.data);
        }
        self.allocator.free(self.frames);
        self.page_map.deinit();
    }

    /// Fetch a page into the buffer pool and return a pinned frame.
    /// The caller MUST call `unpinPage` when done.
    pub fn fetchPage(self: *BufferPool, page_id: u32) !*BufferFrame {
        // Fast path: page already in pool
        if (self.page_map.get(page_id)) |frame_idx| {
            const frame = &self.frames[frame_idx];
            if (frame.pin_count == 0) {
                // Remove from LRU list (it was unpinned)
                self.lruRemove(frame);
            }
            frame.pin_count += 1;
            self.hits += 1;
            return frame;
        }

        // Miss — need to load from disk (or WAL)
        self.misses += 1;
        const frame = try self.getEvictableFrame(page_id);

        // Check WAL first for the latest version of this page
        const from_wal = if (self.wal) |w| try w.readPage(page_id, frame.data) else false;
        if (!from_wal) {
            try self.pager.readPage(page_id, frame.data);
        }
        frame.page_id = page_id;
        frame.pin_count = 1;
        frame.is_dirty = false;
        frame.prev = null;
        frame.next = null;
        try self.page_map.put(page_id, self.frameIndex(frame));
        return frame;
    }

    /// Fetch a new page (just allocated by pager) into the pool.
    /// The page buffer is zeroed. Caller MUST call `unpinPage` when done.
    pub fn fetchNewPage(self: *BufferPool, page_id: u32) !*BufferFrame {
        const frame = try self.getEvictableFrame(page_id);
        @memset(frame.data, 0);
        frame.page_id = page_id;
        frame.pin_count = 1;
        frame.is_dirty = true; // new page is dirty by definition
        frame.prev = null;
        frame.next = null;
        try self.page_map.put(page_id, self.frameIndex(frame));
        return frame;
    }

    /// Unpin a page. If `dirty` is true, the page is marked dirty.
    /// When pin_count reaches 0, the frame enters the LRU eviction list.
    pub fn unpinPage(self: *BufferPool, page_id: u32, dirty: bool) void {
        const frame_idx = self.page_map.get(page_id) orelse return;
        const frame = &self.frames[frame_idx];
        if (frame.pin_count == 0) return;
        if (dirty) frame.is_dirty = true;
        frame.pin_count -= 1;
        if (frame.pin_count == 0) {
            self.lruPushFront(frame);
        }
    }

    /// Flush a specific page to disk if dirty. Does not unpin.
    pub fn flushPage(self: *BufferPool, page_id: u32) !void {
        const frame_idx = self.page_map.get(page_id) orelse return;
        const frame = &self.frames[frame_idx];
        if (frame.is_dirty) {
            try self.writePageOut(page_id, frame.data);
            frame.is_dirty = false;
        }
    }

    /// Flush all dirty pages to disk.
    pub fn flushAll(self: *BufferPool) !void {
        for (self.frames[0..self.frame_count]) |*frame| {
            if (frame.is_dirty) {
                try self.writePageOut(frame.page_id, frame.data);
                frame.is_dirty = false;
            }
        }
    }

    /// Return the number of pages currently cached.
    pub fn pageCount(self: *const BufferPool) u32 {
        return self.frame_count;
    }

    /// Return true if the given page is currently in the pool.
    pub fn containsPage(self: *const BufferPool, page_id: u32) bool {
        return self.page_map.contains(page_id);
    }

    // ── Internal: LRU management ─────────────────────────────────────

    /// Find or allocate a frame for a new page. Evicts LRU if pool is full.
    fn getEvictableFrame(self: *BufferPool, new_page_id: u32) !*BufferFrame {
        _ = new_page_id;
        if (self.frame_count < self.capacity) {
            // Free frame available — use next slot
            const idx = self.frame_count;
            self.frame_count += 1;
            return &self.frames[idx];
        }

        // Pool is full — evict the LRU (tail) frame
        const victim = self.lru_tail orelse return error.BufferPoolFull;
        self.lruRemove(victim);

        // Flush if dirty
        if (victim.is_dirty) {
            try self.writePageOut(victim.page_id, victim.data);
            victim.is_dirty = false;
        }

        // Remove old mapping
        _ = self.page_map.remove(victim.page_id);

        return victim;
    }

    /// Write a page out — through WAL if available, otherwise directly to pager.
    fn writePageOut(self: *BufferPool, page_id: u32, data: []u8) !void {
        if (self.wal) |w| {
            try w.writeFrame(page_id, data);
        } else {
            try self.pager.writePage(page_id, data);
        }
    }

    fn frameIndex(self: *BufferPool, frame: *BufferFrame) u32 {
        const base = @intFromPtr(self.frames.ptr);
        const ptr = @intFromPtr(frame);
        return @intCast((ptr - base) / @sizeOf(BufferFrame));
    }

    /// Push a frame to the front (MRU end) of the LRU list.
    fn lruPushFront(self: *BufferPool, frame: *BufferFrame) void {
        frame.prev = null;
        frame.next = self.lru_head;
        if (self.lru_head) |head| {
            head.prev = frame;
        }
        self.lru_head = frame;
        if (self.lru_tail == null) {
            self.lru_tail = frame;
        }
        self.lru_size += 1;
    }

    /// Remove a frame from the LRU list.
    fn lruRemove(self: *BufferPool, frame: *BufferFrame) void {
        if (frame.prev) |prev| {
            prev.next = frame.next;
        } else {
            self.lru_head = frame.next;
        }
        if (frame.next) |next| {
            next.prev = frame.prev;
        } else {
            self.lru_tail = frame.prev;
        }
        frame.prev = null;
        frame.next = null;
        self.lru_size -= 1;
    }
};

// ── Tests ────────────────────────────────────────────────────────────

test "BufferPool basic fetch and unpin" {
    const allocator = std.testing.allocator;
    const path = "test_bp_basic.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    // Allocate a page and write some data through pager
    const pid = try pager.allocPage();
    const raw = try pager.allocPageBuf();
    defer pager.freePageBuf(raw);
    @memset(raw, 0);
    const hdr = PageHeader{ .page_type = .leaf, .page_id = pid, .cell_count = 42 };
    hdr.serialize(raw[0..PAGE_HEADER_SIZE]);
    try pager.writePage(pid, raw);

    // Now use buffer pool
    var pool = try BufferPool.init(allocator, &pager, 10);
    defer pool.deinit();

    const frame = try pool.fetchPage(pid);
    try std.testing.expectEqual(pid, frame.page_id);
    try std.testing.expectEqual(@as(u32, 1), frame.pin_count);
    try std.testing.expect(!frame.is_dirty);

    // Verify data came through
    const fhdr = frame.getHeader();
    try std.testing.expectEqual(@as(u16, 42), fhdr.cell_count);

    // Unpin
    pool.unpinPage(pid, false);
    try std.testing.expectEqual(@as(u32, 0), frame.pin_count);
}

test "BufferPool cache hit increments stats" {
    const allocator = std.testing.allocator;
    const path = "test_bp_hits.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    const pid = try pager.allocPage();
    const raw = try pager.allocPageBuf();
    defer pager.freePageBuf(raw);
    @memset(raw, 0);
    const hdr = PageHeader{ .page_type = .leaf, .page_id = pid };
    hdr.serialize(raw[0..PAGE_HEADER_SIZE]);
    try pager.writePage(pid, raw);

    var pool = try BufferPool.init(allocator, &pager, 10);
    defer pool.deinit();

    // First fetch = miss
    const f1 = try pool.fetchPage(pid);
    try std.testing.expectEqual(@as(u64, 0), pool.hits);
    try std.testing.expectEqual(@as(u64, 1), pool.misses);

    pool.unpinPage(pid, false);

    // Second fetch = hit
    const f2 = try pool.fetchPage(pid);
    try std.testing.expectEqual(@as(u64, 1), pool.hits);
    try std.testing.expectEqual(@as(u64, 1), pool.misses);
    try std.testing.expectEqual(f1, f2);

    pool.unpinPage(pid, false);
}

test "BufferPool dirty page flush" {
    const allocator = std.testing.allocator;
    const path = "test_bp_dirty.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    const pid = try pager.allocPage();
    const raw = try pager.allocPageBuf();
    defer pager.freePageBuf(raw);
    @memset(raw, 0);
    const hdr = PageHeader{ .page_type = .leaf, .page_id = pid };
    hdr.serialize(raw[0..PAGE_HEADER_SIZE]);
    try pager.writePage(pid, raw);

    var pool = try BufferPool.init(allocator, &pager, 10);
    defer pool.deinit();

    // Fetch, modify, unpin as dirty
    const frame = try pool.fetchPage(pid);
    const payload = "modified via buffer pool";
    @memcpy(frame.data[PAGE_HEADER_SIZE..][0..payload.len], payload);
    pool.unpinPage(pid, true);
    try std.testing.expect(frame.is_dirty);

    // Flush
    try pool.flushPage(pid);
    try std.testing.expect(!frame.is_dirty);

    // Verify on disk by reading through pager directly
    try pager.readPage(pid, raw);
    try std.testing.expectEqualStrings(payload, raw[PAGE_HEADER_SIZE..][0..payload.len]);
}

test "BufferPool LRU eviction" {
    const allocator = std.testing.allocator;
    const path = "test_bp_eviction.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    // Create 5 pages
    var pids: [5]u32 = undefined;
    const raw = try pager.allocPageBuf();
    defer pager.freePageBuf(raw);
    for (&pids) |*pid| {
        pid.* = try pager.allocPage();
        @memset(raw, 0);
        const hdr = PageHeader{ .page_type = .leaf, .page_id = pid.* };
        hdr.serialize(raw[0..PAGE_HEADER_SIZE]);
        try pager.writePage(pid.*, raw);
    }

    // Pool with capacity 3 — forces eviction
    var pool = try BufferPool.init(allocator, &pager, 3);
    defer pool.deinit();

    // Load pages 0,1,2 — fills pool
    for (pids[0..3]) |pid| {
        _ = try pool.fetchPage(pid);
        pool.unpinPage(pid, false);
    }
    try std.testing.expectEqual(@as(u32, 3), pool.pageCount());
    try std.testing.expect(pool.containsPage(pids[0]));
    try std.testing.expect(pool.containsPage(pids[1]));
    try std.testing.expect(pool.containsPage(pids[2]));

    // Access page 0 to make it MRU (it's currently LRU tail)
    _ = try pool.fetchPage(pids[0]);
    pool.unpinPage(pids[0], false);

    // Load page 3 — should evict page 1 (LRU)
    _ = try pool.fetchPage(pids[3]);
    pool.unpinPage(pids[3], false);
    try std.testing.expect(!pool.containsPage(pids[1]));
    try std.testing.expect(pool.containsPage(pids[0]));
    try std.testing.expect(pool.containsPage(pids[2]));
    try std.testing.expect(pool.containsPage(pids[3]));

    // Load page 4 — should evict page 2
    _ = try pool.fetchPage(pids[4]);
    pool.unpinPage(pids[4], false);
    try std.testing.expect(!pool.containsPage(pids[2]));
}

test "BufferPool pinned pages not evicted" {
    const allocator = std.testing.allocator;
    const path = "test_bp_pin.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    // Create 4 pages
    var pids: [4]u32 = undefined;
    const raw = try pager.allocPageBuf();
    defer pager.freePageBuf(raw);
    for (&pids) |*pid| {
        pid.* = try pager.allocPage();
        @memset(raw, 0);
        const hdr = PageHeader{ .page_type = .leaf, .page_id = pid.* };
        hdr.serialize(raw[0..PAGE_HEADER_SIZE]);
        try pager.writePage(pid.*, raw);
    }

    // Pool with capacity 3
    var pool = try BufferPool.init(allocator, &pager, 3);
    defer pool.deinit();

    // Pin all 3 frames (don't unpin)
    _ = try pool.fetchPage(pids[0]);
    _ = try pool.fetchPage(pids[1]);
    _ = try pool.fetchPage(pids[2]);

    // All pinned — fetching page 3 should fail
    try std.testing.expectError(error.BufferPoolFull, pool.fetchPage(pids[3]));

    // Unpin one page — now eviction can proceed
    pool.unpinPage(pids[0], false);
    const frame = try pool.fetchPage(pids[3]);
    try std.testing.expectEqual(pids[3], frame.page_id);
    pool.unpinPage(pids[3], false);
}

test "BufferPool multiple pin/unpin" {
    const allocator = std.testing.allocator;
    const path = "test_bp_multipin.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    const pid = try pager.allocPage();
    const raw = try pager.allocPageBuf();
    defer pager.freePageBuf(raw);
    @memset(raw, 0);
    const hdr = PageHeader{ .page_type = .leaf, .page_id = pid };
    hdr.serialize(raw[0..PAGE_HEADER_SIZE]);
    try pager.writePage(pid, raw);

    var pool = try BufferPool.init(allocator, &pager, 10);
    defer pool.deinit();

    // Pin the same page multiple times
    const f1 = try pool.fetchPage(pid);
    try std.testing.expectEqual(@as(u32, 1), f1.pin_count);
    const f2 = try pool.fetchPage(pid);
    try std.testing.expectEqual(@as(u32, 2), f2.pin_count);
    try std.testing.expectEqual(f1, f2); // same frame

    // Unpin once — still pinned
    pool.unpinPage(pid, false);
    try std.testing.expectEqual(@as(u32, 1), f1.pin_count);

    // Unpin again — now unpinned
    pool.unpinPage(pid, false);
    try std.testing.expectEqual(@as(u32, 0), f1.pin_count);
}

test "BufferPool fetchNewPage" {
    const allocator = std.testing.allocator;
    const path = "test_bp_newpage.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    var pool = try BufferPool.init(allocator, &pager, 10);
    defer pool.deinit();

    // Allocate via pager, then fetch as new into pool
    const pid = try pager.allocPage();
    const frame = try pool.fetchNewPage(pid);

    try std.testing.expectEqual(pid, frame.page_id);
    try std.testing.expect(frame.is_dirty);
    try std.testing.expectEqual(@as(u32, 1), frame.pin_count);

    // Write some content, unpin, flush
    const hdr = PageHeader{ .page_type = .leaf, .page_id = pid };
    hdr.serialize(frame.data[0..PAGE_HEADER_SIZE]);

    pool.unpinPage(pid, true);
    try pool.flushPage(pid);

    // Verify on disk
    const raw = try pager.allocPageBuf();
    defer pager.freePageBuf(raw);
    try pager.readPage(pid, raw);
    const rhdr = PageHeader.deserialize(raw[0..PAGE_HEADER_SIZE]);
    try std.testing.expectEqual(page_mod.PageType.leaf, rhdr.page_type);
}

test "BufferPool flushAll writes all dirty pages" {
    const allocator = std.testing.allocator;
    const path = "test_bp_flushall.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    var pool = try BufferPool.init(allocator, &pager, 10);
    defer pool.deinit();

    // Create and dirty 3 pages
    var pids: [3]u32 = undefined;
    for (&pids) |*pid| {
        pid.* = try pager.allocPage();
        const frame = try pool.fetchNewPage(pid.*);
        const hdr = PageHeader{ .page_type = .leaf, .page_id = pid.* };
        hdr.serialize(frame.data[0..PAGE_HEADER_SIZE]);
        const tag = [_]u8{ @as(u8, @truncate(pid.*)) + 'A' };
        @memcpy(frame.data[PAGE_HEADER_SIZE..][0..1], &tag);
        pool.unpinPage(pid.*, true);
    }

    try pool.flushAll();

    // Verify all flushed
    const raw = try pager.allocPageBuf();
    defer pager.freePageBuf(raw);
    for (pids) |pid| {
        try pager.readPage(pid, raw);
        const expected = [_]u8{ @as(u8, @truncate(pid)) + 'A' };
        try std.testing.expectEqual(expected[0], raw[PAGE_HEADER_SIZE]);
    }
}

test "BufferPool eviction flushes dirty pages" {
    const allocator = std.testing.allocator;
    const path = "test_bp_evict_dirty.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    // Pool of size 2
    var pool = try BufferPool.init(allocator, &pager, 2);
    defer pool.deinit();

    // Create 3 pages
    var pids: [3]u32 = undefined;
    for (&pids) |*pid| {
        pid.* = try pager.allocPage();
    }

    // Fetch page 0, modify, unpin as dirty
    const f0 = try pool.fetchNewPage(pids[0]);
    const hdr0 = PageHeader{ .page_type = .leaf, .page_id = pids[0] };
    hdr0.serialize(f0.data[0..PAGE_HEADER_SIZE]);
    f0.data[PAGE_HEADER_SIZE] = 0xAA;
    pool.unpinPage(pids[0], true);

    // Fetch page 1, unpin
    const f1 = try pool.fetchNewPage(pids[1]);
    const hdr1 = PageHeader{ .page_type = .leaf, .page_id = pids[1] };
    hdr1.serialize(f1.data[0..PAGE_HEADER_SIZE]);
    pool.unpinPage(pids[1], false);

    // Fetch page 2 — evicts page 0 (LRU, dirty — should flush first)
    const f2 = try pool.fetchNewPage(pids[2]);
    const hdr2 = PageHeader{ .page_type = .leaf, .page_id = pids[2] };
    hdr2.serialize(f2.data[0..PAGE_HEADER_SIZE]);
    pool.unpinPage(pids[2], false);

    // Verify page 0 was flushed to disk
    const raw = try pager.allocPageBuf();
    defer pager.freePageBuf(raw);
    try pager.readPage(pids[0], raw);
    try std.testing.expectEqual(@as(u8, 0xAA), raw[PAGE_HEADER_SIZE]);
}

test "BufferPool pool size 1" {
    const allocator = std.testing.allocator;
    const path = "test_bp_size1.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    // Create 3 pages on disk
    var pids: [3]u32 = undefined;
    const raw = try pager.allocPageBuf();
    defer pager.freePageBuf(raw);
    for (&pids, 0..) |*pid, i| {
        pid.* = try pager.allocPage();
        @memset(raw, 0);
        const hdr = PageHeader{ .page_type = .leaf, .page_id = pid.* };
        hdr.serialize(raw[0..PAGE_HEADER_SIZE]);
        // Write unique data to each page
        raw[PAGE_HEADER_SIZE] = @as(u8, @truncate(i)) + 'A';
        try pager.writePage(pid.*, raw);
    }

    // Pool with capacity 1
    var pool = try BufferPool.init(allocator, &pager, 1);
    defer pool.deinit();

    // Fetch page 0, unpin
    _ = try pool.fetchPage(pids[0]);
    pool.unpinPage(pids[0], false);
    try std.testing.expect(pool.containsPage(pids[0]));

    // Fetch page 1 — evicts page 0
    _ = try pool.fetchPage(pids[1]);
    pool.unpinPage(pids[1], false);
    try std.testing.expect(!pool.containsPage(pids[0]));
    try std.testing.expect(pool.containsPage(pids[1]));

    // Fetch page 2 — evicts page 1
    _ = try pool.fetchPage(pids[2]);
    pool.unpinPage(pids[2], false);
    try std.testing.expect(!pool.containsPage(pids[1]));
    try std.testing.expect(pool.containsPage(pids[2]));

    // Fetch page 0 again — evicts page 2, verify data integrity
    const frame = try pool.fetchPage(pids[0]);
    try std.testing.expectEqual(@as(u8, 'A'), frame.data[PAGE_HEADER_SIZE]);
    pool.unpinPage(pids[0], false);
    try std.testing.expect(!pool.containsPage(pids[2]));
    try std.testing.expect(pool.containsPage(pids[0]));
}

test "BufferPool LRU order with 5 pages" {
    const allocator = std.testing.allocator;
    const path = "test_bp_lru5.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    // Create 5 pages on disk
    var pids: [5]u32 = undefined;
    const raw = try pager.allocPageBuf();
    defer pager.freePageBuf(raw);
    for (&pids) |*pid| {
        pid.* = try pager.allocPage();
        @memset(raw, 0);
        const hdr = PageHeader{ .page_type = .leaf, .page_id = pid.* };
        hdr.serialize(raw[0..PAGE_HEADER_SIZE]);
        try pager.writePage(pid.*, raw);
    }

    // Pool with capacity 3
    var pool = try BufferPool.init(allocator, &pager, 3);
    defer pool.deinit();

    // Load pages 0, 1, 2 and unpin all
    for (pids[0..3]) |pid| {
        _ = try pool.fetchPage(pid);
        pool.unpinPage(pid, false);
    }

    // Access page 1 to make it MRU
    _ = try pool.fetchPage(pids[1]);
    pool.unpinPage(pids[1], false);

    // Fetch page 3 — should evict page 0 (LRU tail)
    _ = try pool.fetchPage(pids[3]);
    pool.unpinPage(pids[3], false);
    try std.testing.expect(!pool.containsPage(pids[0]));
    try std.testing.expect(pool.containsPage(pids[1]));
    try std.testing.expect(pool.containsPage(pids[2]));
    try std.testing.expect(pool.containsPage(pids[3]));

    // Fetch page 4 — should evict page 2
    _ = try pool.fetchPage(pids[4]);
    pool.unpinPage(pids[4], false);
    try std.testing.expect(!pool.containsPage(pids[0]));
    try std.testing.expect(pool.containsPage(pids[1]));
    try std.testing.expect(!pool.containsPage(pids[2]));
    try std.testing.expect(pool.containsPage(pids[3]));
    try std.testing.expect(pool.containsPage(pids[4]));
}

test "BufferPool dirty flag cycles" {
    const allocator = std.testing.allocator;
    const path = "test_bp_dirty_cycles.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    const pid = try pager.allocPage();
    const raw = try pager.allocPageBuf();
    defer pager.freePageBuf(raw);
    @memset(raw, 0);
    const hdr = PageHeader{ .page_type = .leaf, .page_id = pid };
    hdr.serialize(raw[0..PAGE_HEADER_SIZE]);
    try pager.writePage(pid, raw);

    var pool = try BufferPool.init(allocator, &pager, 10);
    defer pool.deinit();

    // First cycle: fetch, unpin as dirty, flush
    const frame = try pool.fetchPage(pid);
    pool.unpinPage(pid, true);
    try std.testing.expect(frame.is_dirty);
    try pool.flushPage(pid);
    try std.testing.expect(!frame.is_dirty);

    // Second cycle: fetch again, unpin as dirty again, flush again
    _ = try pool.fetchPage(pid);
    pool.unpinPage(pid, true);
    try std.testing.expect(frame.is_dirty);
    try pool.flushPage(pid);
    try std.testing.expect(!frame.is_dirty);

    // Third cycle to be sure
    _ = try pool.fetchPage(pid);
    pool.unpinPage(pid, true);
    try std.testing.expect(frame.is_dirty);
    try pool.flushPage(pid);
    try std.testing.expect(!frame.is_dirty);
}

test "BufferPool re-fetch after eviction preserves data" {
    const allocator = std.testing.allocator;
    const path = "test_bp_refetch.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    // Create 3 pages
    var pids: [3]u32 = undefined;
    const raw = try pager.allocPageBuf();
    defer pager.freePageBuf(raw);
    for (&pids) |*pid| {
        pid.* = try pager.allocPage();
        @memset(raw, 0);
        const hdr = PageHeader{ .page_type = .leaf, .page_id = pid.* };
        hdr.serialize(raw[0..PAGE_HEADER_SIZE]);
        try pager.writePage(pid.*, raw);
    }

    // Pool with capacity 2
    var pool = try BufferPool.init(allocator, &pager, 2);
    defer pool.deinit();

    // Write unique data to page 0 via pool
    const f0 = try pool.fetchPage(pids[0]);
    const unique_data = "PAGE_ZERO_DATA";
    @memcpy(f0.data[PAGE_HEADER_SIZE..][0..unique_data.len], unique_data);
    pool.unpinPage(pids[0], true); // Mark dirty

    // Fetch page 1, unpin
    _ = try pool.fetchPage(pids[1]);
    pool.unpinPage(pids[1], false);

    // Fetch page 2 — evicts page 0 (LRU), should flush dirty page 0
    _ = try pool.fetchPage(pids[2]);
    pool.unpinPage(pids[2], false);
    try std.testing.expect(!pool.containsPage(pids[0]));

    // Re-fetch page 0 — evicts page 1, load from disk
    const f0_refetch = try pool.fetchPage(pids[0]);
    const data_slice = f0_refetch.data[PAGE_HEADER_SIZE..][0..unique_data.len];
    try std.testing.expectEqualStrings(unique_data, data_slice);
    pool.unpinPage(pids[0], false);
}

test "BufferPool stress rapid pin unpin" {
    const allocator = std.testing.allocator;
    const path = "test_bp_stress_pin.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    // Create one page
    const pid = try pager.allocPage();
    const raw = try pager.allocPageBuf();
    defer pager.freePageBuf(raw);
    @memset(raw, 0);
    const hdr = PageHeader{ .page_type = .leaf, .page_id = pid };
    hdr.serialize(raw[0..PAGE_HEADER_SIZE]);
    try pager.writePage(pid, raw);

    var pool = try BufferPool.init(allocator, &pager, 10);
    defer pool.deinit();

    // Pin 100 times
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        _ = try pool.fetchPage(pid);
    }

    const frame_idx = pool.page_map.get(pid).?;
    const frame = &pool.frames[frame_idx];
    try std.testing.expectEqual(@as(u32, 100), frame.pin_count);

    // Unpin 100 times
    i = 0;
    while (i < 100) : (i += 1) {
        pool.unpinPage(pid, false);
    }

    try std.testing.expectEqual(@as(u32, 0), frame.pin_count);
}

test "BufferPool large page ids" {
    const allocator = std.testing.allocator;
    const path = "test_bp_large_ids.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    // Allocate 10 pages (will have page IDs 2-11)
    var pids: [10]u32 = undefined;
    const raw = try pager.allocPageBuf();
    defer pager.freePageBuf(raw);
    for (&pids) |*pid| {
        pid.* = try pager.allocPage();
        @memset(raw, 0);
        const hdr = PageHeader{ .page_type = .leaf, .page_id = pid.* };
        hdr.serialize(raw[0..PAGE_HEADER_SIZE]);
        try pager.writePage(pid.*, raw);
    }

    // Pool with capacity 10
    var pool = try BufferPool.init(allocator, &pager, 10);
    defer pool.deinit();

    // Fetch all pages into pool
    for (pids) |pid| {
        const frame = try pool.fetchPage(pid);
        try std.testing.expectEqual(pid, frame.page_id);
    }

    // Verify all are in pool
    for (pids) |pid| {
        try std.testing.expect(pool.containsPage(pid));
    }

    // Unpin all
    for (pids) |pid| {
        pool.unpinPage(pid, false);
    }

    // Verify still in pool (just unpinned)
    for (pids) |pid| {
        try std.testing.expect(pool.containsPage(pid));
    }
}
