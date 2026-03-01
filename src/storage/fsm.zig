//! Free Space Map (FSM) — tracks available space per page for efficient inserts.
//!
//! Each page's free space is encoded as a single byte (category 0-255),
//! where 0 means the page is full and 255 means it's completely empty.
//! The mapping is linear: category = (free_bytes * 255) / max_usable_space.
//!
//! The FSM is maintained in-memory as a HashMap and persisted to dedicated
//! FSM pages on disk during checkpoint/shutdown. On startup, FSM pages are
//! read to restore the map; if absent or corrupt, a full leaf-page scan
//! rebuilds it.
//!
//! FSM page layout:
//!   [PageHeader 16B][entry_count: u32][next_fsm_page: u32]
//!   [entries: (page_id: u32, category: u8, _pad: [3]u8) × N]
//!
//! Each entry is 8 bytes. A 4096-byte page holds (4096-16-8)/8 = 509 entries.

const std = @import("std");
const page_mod = @import("page.zig");
const PageHeader = page_mod.PageHeader;
const PAGE_HEADER_SIZE = page_mod.PAGE_HEADER_SIZE;

// ── Constants ──────────────────────────────────────────────────────────

/// Size of the FSM page metadata after the page header.
/// [entry_count: u32][next_fsm_page: u32]
const FSM_META_SIZE: u32 = 8;

/// Size of each FSM entry on disk: [page_id: u32][category: u8][pad: 3B]
const FSM_ENTRY_SIZE: u32 = 8;

/// Maximum category value (fully empty page).
pub const MAX_CATEGORY: u8 = 255;

/// Leaf page overhead: page header (16) + prev/next pointers (8).
const LEAF_OVERHEAD: u16 = PAGE_HEADER_SIZE + 8;

// ── Free Space Map ─────────────────────────────────────────────────────

pub const FreeSpaceMap = struct {
    map: std.AutoHashMap(u32, u8),
    page_size: u32,
    max_usable: u16,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, page_size: u32) FreeSpaceMap {
        const usable: u16 = @intCast(page_size - @as(u32, LEAF_OVERHEAD));
        return .{
            .map = std.AutoHashMap(u32, u8).init(allocator),
            .page_size = page_size,
            .max_usable = usable,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *FreeSpaceMap) void {
        self.map.deinit();
    }

    /// Convert free bytes to a category (0-255).
    pub fn bytesToCategory(self: *const FreeSpaceMap, free_bytes: u16) u8 {
        if (free_bytes == 0) return 0;
        if (free_bytes >= self.max_usable) return MAX_CATEGORY;
        // Linear mapping: category = (free_bytes * 255) / max_usable
        const result = (@as(u32, free_bytes) * 255) / @as(u32, self.max_usable);
        return @intCast(@min(result, 255));
    }

    /// Convert a category back to minimum free bytes (ceiling division).
    pub fn categoryToMinBytes(self: *const FreeSpaceMap, category: u8) u16 {
        if (category == 0) return 0;
        if (category == MAX_CATEGORY) return self.max_usable;
        // Ceiling division: (category * max_usable + 254) / 255
        const result = (@as(u32, category) * @as(u32, self.max_usable) + 254) / 255;
        return @intCast(@min(result, self.max_usable));
    }

    /// Update the free space record for a page.
    /// If the page is full (category 0), the entry is removed to save memory.
    pub fn update(self: *FreeSpaceMap, page_id: u32, free_bytes: u16) !void {
        const cat = self.bytesToCategory(free_bytes);
        if (cat == 0) {
            _ = self.map.remove(page_id);
        } else {
            try self.map.put(page_id, cat);
        }
    }

    /// Remove a page from the FSM (e.g., when freed to the freelist).
    pub fn remove(self: *FreeSpaceMap, page_id: u32) void {
        _ = self.map.remove(page_id);
    }

    /// Get the free space category for a page, or 0 if not tracked.
    pub fn getCategory(self: *const FreeSpaceMap, page_id: u32) u8 {
        return self.map.get(page_id) orelse 0;
    }

    /// Find any page with at least `needed_bytes` of free space.
    /// Returns null if no such page exists.
    pub fn findPage(self: *const FreeSpaceMap, needed_bytes: u16) ?u32 {
        const needed_cat = self.bytesToCategory(needed_bytes);
        if (needed_cat == 0) return null;

        var best_page: ?u32 = null;
        var best_cat: u8 = 0;

        var it = self.map.iterator();
        while (it.next()) |entry| {
            if (entry.value_ptr.* >= needed_cat and entry.value_ptr.* > best_cat) {
                best_cat = entry.value_ptr.*;
                best_page = entry.key_ptr.*;
            }
        }

        return best_page;
    }

    /// Find a page with at least `needed_bytes` free, preferring the page
    /// closest to `hint_page_id` for locality.
    pub fn findPageNear(self: *const FreeSpaceMap, needed_bytes: u16, hint_page_id: u32) ?u32 {
        const needed_cat = self.bytesToCategory(needed_bytes);
        if (needed_cat == 0) return null;

        var best_page: ?u32 = null;
        var best_distance: u32 = std.math.maxInt(u32);

        var it = self.map.iterator();
        while (it.next()) |entry| {
            if (entry.value_ptr.* >= needed_cat) {
                const page_id = entry.key_ptr.*;
                const distance = if (page_id >= hint_page_id)
                    page_id - hint_page_id
                else
                    hint_page_id - page_id;

                if (distance < best_distance) {
                    best_distance = distance;
                    best_page = page_id;
                }
            }
        }

        return best_page;
    }

    /// Return the total estimated free space across all tracked pages (in bytes).
    pub fn totalFreeSpace(self: *const FreeSpaceMap) u64 {
        var total: u64 = 0;
        var it = self.map.iterator();
        while (it.next()) |entry| {
            total += @as(u64, self.categoryToMinBytes(entry.value_ptr.*));
        }
        return total;
    }

    /// Return the number of pages tracked by the FSM.
    pub fn trackedPages(self: *const FreeSpaceMap) u32 {
        return @intCast(self.map.count());
    }

    /// Return summary statistics.
    pub const Stats = struct {
        tracked_pages: u32,
        total_free_bytes: u64,
        empty_pages: u32,
        nearly_full_pages: u32,
        avg_category: f64,
    };

    pub fn getStats(self: *const FreeSpaceMap) Stats {
        var stats = Stats{
            .tracked_pages = 0,
            .total_free_bytes = 0,
            .empty_pages = 0,
            .nearly_full_pages = 0,
            .avg_category = 0,
        };

        var cat_sum: u64 = 0;
        var it = self.map.iterator();
        while (it.next()) |entry| {
            stats.tracked_pages += 1;
            const cat = entry.value_ptr.*;
            cat_sum += @as(u64, cat);
            stats.total_free_bytes += @as(u64, self.categoryToMinBytes(cat));

            if (cat >= 250) stats.empty_pages += 1;
            if (cat <= 10) stats.nearly_full_pages += 1;
        }

        if (stats.tracked_pages > 0) {
            stats.avg_category = @as(f64, @floatFromInt(cat_sum)) /
                @as(f64, @floatFromInt(stats.tracked_pages));
        }

        return stats;
    }

    // ── Disk Persistence ───────────────────────────────────────────────

    /// Number of FSM entries that fit in a single page.
    fn entriesPerPage(self: *const FreeSpaceMap) u32 {
        return (self.page_size - PAGE_HEADER_SIZE - FSM_META_SIZE) / FSM_ENTRY_SIZE;
    }

    /// Serialize all FSM entries to disk pages via the pager.
    /// Returns the page ID of the first FSM page (to store in DB header).
    /// Returns 0 if the FSM is empty.
    pub fn saveToDisk(self: *FreeSpaceMap, pager: *page_mod.Pager) !u32 {
        if (self.map.count() == 0) return 0;

        const per_page = self.entriesPerPage();
        const total_entries: u32 = @intCast(self.map.count());
        const num_pages = (total_entries + per_page - 1) / per_page;

        // Allocate FSM pages
        var fsm_pages = try self.allocator.alloc(u32, num_pages);
        defer self.allocator.free(fsm_pages);

        for (0..num_pages) |i| {
            fsm_pages[i] = try pager.allocPage();
        }

        // Write entries to pages
        var it = self.map.iterator();
        var page_idx: u32 = 0;
        var entry_idx: u32 = 0;

        var buf = try pager.allocPageBuf();
        defer pager.freePageBuf(buf);

        while (page_idx < num_pages) {
            @memset(buf, 0);

            const entries_this_page = @min(per_page, total_entries - page_idx * per_page);
            const next_page = if (page_idx + 1 < num_pages) fsm_pages[page_idx + 1] else @as(u32, 0);

            // Write page header
            const hdr = PageHeader{
                .page_type = .fsm,
                .page_id = fsm_pages[page_idx],
                .cell_count = @intCast(entries_this_page),
            };
            hdr.serialize(buf[0..PAGE_HEADER_SIZE]);

            // Write FSM metadata
            std.mem.writeInt(u32, buf[PAGE_HEADER_SIZE..][0..4], entries_this_page, .little);
            std.mem.writeInt(u32, buf[PAGE_HEADER_SIZE + 4 ..][0..4], next_page, .little);

            // Write entries
            const data_start = PAGE_HEADER_SIZE + FSM_META_SIZE;
            var local_idx: u32 = 0;
            while (local_idx < entries_this_page) {
                if (it.next()) |entry| {
                    const off = data_start + local_idx * FSM_ENTRY_SIZE;
                    std.mem.writeInt(u32, buf[off..][0..4], entry.key_ptr.*, .little);
                    buf[off + 4] = entry.value_ptr.*;
                    // bytes 5-7 are padding (already zeroed)
                    local_idx += 1;
                    entry_idx += 1;
                } else break;
            }

            try pager.writePage(fsm_pages[page_idx], buf);
            page_idx += 1;
        }

        return fsm_pages[0];
    }

    /// Load FSM entries from disk pages starting at `fsm_head`.
    pub fn loadFromDisk(self: *FreeSpaceMap, pager: *page_mod.Pager, fsm_head: u32) !void {
        if (fsm_head == 0) return;

        var buf = try pager.allocPageBuf();
        defer pager.freePageBuf(buf);

        var current_page = fsm_head;
        while (current_page != 0) {
            if (current_page >= pager.page_count) break;

            try pager.readPage(current_page, buf);

            const hdr = PageHeader.deserialize(buf[0..PAGE_HEADER_SIZE]);
            if (hdr.page_type != .fsm) break;

            const entry_count = std.mem.readInt(u32, buf[PAGE_HEADER_SIZE..][0..4], .little);
            const next_page = std.mem.readInt(u32, buf[PAGE_HEADER_SIZE + 4 ..][0..4], .little);

            const data_start = PAGE_HEADER_SIZE + FSM_META_SIZE;
            for (0..entry_count) |i| {
                const off = data_start + @as(u32, @intCast(i)) * FSM_ENTRY_SIZE;
                if (off + FSM_ENTRY_SIZE > self.page_size) break;

                const page_id = std.mem.readInt(u32, buf[off..][0..4], .little);
                const category = buf[off + 4];

                if (category > 0) {
                    try self.map.put(page_id, category);
                }
            }

            current_page = next_page;
        }
    }

    /// Clear all FSM entries.
    pub fn clear(self: *FreeSpaceMap) void {
        self.map.clearRetainingCapacity();
    }
};

// ── Tests ──────────────────────────────────────────────────────────────

test "bytesToCategory and categoryToMinBytes basic" {
    var fsm = FreeSpaceMap.init(std.testing.allocator, 4096);
    defer fsm.deinit();

    // max_usable = 4096 - 24 = 4072
    try std.testing.expectEqual(@as(u16, 4072), fsm.max_usable);

    // 0 bytes free → category 0
    try std.testing.expectEqual(@as(u8, 0), fsm.bytesToCategory(0));

    // Full usable space → category 255
    try std.testing.expectEqual(@as(u8, 255), fsm.bytesToCategory(4072));

    // Half space → approximately 127
    const half_cat = fsm.bytesToCategory(2036);
    try std.testing.expect(half_cat >= 126 and half_cat <= 128);

    // Category 0 → 0 bytes
    try std.testing.expectEqual(@as(u16, 0), fsm.categoryToMinBytes(0));

    // Category 255 → max usable
    try std.testing.expectEqual(@as(u16, 4072), fsm.categoryToMinBytes(255));
}

test "bytesToCategory edge cases" {
    var fsm = FreeSpaceMap.init(std.testing.allocator, 4096);
    defer fsm.deinit();

    // 1 byte free → should be category 0 (too small to matter)
    const cat_1 = fsm.bytesToCategory(1);
    try std.testing.expect(cat_1 <= 1);

    // Exceeding max → should cap at 255
    try std.testing.expectEqual(@as(u8, 255), fsm.bytesToCategory(5000));

    // Very small page size
    var fsm_small = FreeSpaceMap.init(std.testing.allocator, 512);
    defer fsm_small.deinit();
    // max_usable = 512 - 24 = 488
    try std.testing.expectEqual(@as(u16, 488), fsm_small.max_usable);
    try std.testing.expectEqual(@as(u8, 255), fsm_small.bytesToCategory(488));
}

test "roundtrip category conversion" {
    var fsm = FreeSpaceMap.init(std.testing.allocator, 4096);
    defer fsm.deinit();

    // For any free_bytes, converting to category and back should give a lower bound
    const test_values = [_]u16{ 0, 1, 16, 100, 500, 1000, 2000, 3000, 4000, 4072 };
    for (test_values) |free_bytes| {
        const cat = fsm.bytesToCategory(free_bytes);
        const min_bytes = fsm.categoryToMinBytes(cat);
        // min_bytes should be <= free_bytes (category is a lower bound)
        try std.testing.expect(min_bytes <= free_bytes);
    }
}

test "update and getCategory" {
    var fsm = FreeSpaceMap.init(std.testing.allocator, 4096);
    defer fsm.deinit();

    // Initially, unknown pages return 0
    try std.testing.expectEqual(@as(u8, 0), fsm.getCategory(42));

    // Update with some free space
    try fsm.update(42, 2000);
    const cat = fsm.getCategory(42);
    try std.testing.expect(cat > 0);
    try std.testing.expect(cat > 100); // 2000/4072 * 255 ≈ 125

    // Update to full → entry removed
    try fsm.update(42, 0);
    try std.testing.expectEqual(@as(u8, 0), fsm.getCategory(42));
}

test "update overwrites previous value" {
    var fsm = FreeSpaceMap.init(std.testing.allocator, 4096);
    defer fsm.deinit();

    try fsm.update(10, 1000);
    const cat1 = fsm.getCategory(10);

    try fsm.update(10, 3000);
    const cat2 = fsm.getCategory(10);

    try std.testing.expect(cat2 > cat1);
}

test "remove" {
    var fsm = FreeSpaceMap.init(std.testing.allocator, 4096);
    defer fsm.deinit();

    try fsm.update(5, 2000);
    try std.testing.expect(fsm.getCategory(5) > 0);

    fsm.remove(5);
    try std.testing.expectEqual(@as(u8, 0), fsm.getCategory(5));

    // Removing non-existent page is fine
    fsm.remove(999);
}

test "findPage returns page with enough space" {
    var fsm = FreeSpaceMap.init(std.testing.allocator, 4096);
    defer fsm.deinit();

    // No pages tracked → null
    try std.testing.expectEqual(@as(?u32, null), fsm.findPage(100));

    // Add pages with different free space amounts
    try fsm.update(10, 500);  // ~31 category
    try fsm.update(20, 2000); // ~125 category
    try fsm.update(30, 4000); // ~250 category

    // Looking for 100 bytes → should find one of the pages
    const page = fsm.findPage(100);
    try std.testing.expect(page != null);

    // Looking for 3000 bytes → should find page 30
    const big_page = fsm.findPage(3000);
    try std.testing.expect(big_page != null);
    try std.testing.expectEqual(@as(u32, 30), big_page.?);

    // Looking for more than max → null
    try std.testing.expectEqual(@as(?u32, null), fsm.findPage(5000));
}

test "findPage prefers best fit" {
    var fsm = FreeSpaceMap.init(std.testing.allocator, 4096);
    defer fsm.deinit();

    try fsm.update(1, 1000);
    try fsm.update(2, 2000);
    try fsm.update(3, 3000);

    // findPage returns the page with the most free space
    const page = fsm.findPage(500);
    try std.testing.expect(page != null);
    try std.testing.expectEqual(@as(u32, 3), page.?);
}

test "findPageNear prefers locality" {
    var fsm = FreeSpaceMap.init(std.testing.allocator, 4096);
    defer fsm.deinit();

    // All pages have plenty of space
    try fsm.update(10, 3000);
    try fsm.update(50, 3000);
    try fsm.update(100, 3000);

    // Hint page 45 → should prefer page 50 (closest)
    const page = fsm.findPageNear(100, 45);
    try std.testing.expect(page != null);
    try std.testing.expectEqual(@as(u32, 50), page.?);

    // Hint page 95 → should prefer page 100
    const page2 = fsm.findPageNear(100, 95);
    try std.testing.expect(page2 != null);
    try std.testing.expectEqual(@as(u32, 100), page2.?);
}

test "totalFreeSpace and trackedPages" {
    var fsm = FreeSpaceMap.init(std.testing.allocator, 4096);
    defer fsm.deinit();

    try std.testing.expectEqual(@as(u64, 0), fsm.totalFreeSpace());
    try std.testing.expectEqual(@as(u32, 0), fsm.trackedPages());

    try fsm.update(1, 1000);
    try fsm.update(2, 2000);

    try std.testing.expectEqual(@as(u32, 2), fsm.trackedPages());
    const total = fsm.totalFreeSpace();
    // Should be roughly 1000 + 2000 = 3000, but category quantization loses precision
    try std.testing.expect(total >= 2500 and total <= 3500);
}

test "getStats" {
    var fsm = FreeSpaceMap.init(std.testing.allocator, 4096);
    defer fsm.deinit();

    // Empty FSM stats
    var stats = fsm.getStats();
    try std.testing.expectEqual(@as(u32, 0), stats.tracked_pages);
    try std.testing.expectEqual(@as(f64, 0), stats.avg_category);

    // Add various pages
    try fsm.update(1, 100);   // nearly full
    try fsm.update(2, 4072);  // empty (category 255)
    try fsm.update(3, 2000);  // half full

    stats = fsm.getStats();
    try std.testing.expectEqual(@as(u32, 3), stats.tracked_pages);
    try std.testing.expectEqual(@as(u32, 1), stats.empty_pages);
    try std.testing.expect(stats.avg_category > 0);
    try std.testing.expect(stats.total_free_bytes > 0);
}

test "clear" {
    var fsm = FreeSpaceMap.init(std.testing.allocator, 4096);
    defer fsm.deinit();

    try fsm.update(1, 1000);
    try fsm.update(2, 2000);
    try std.testing.expectEqual(@as(u32, 2), fsm.trackedPages());

    fsm.clear();
    try std.testing.expectEqual(@as(u32, 0), fsm.trackedPages());
    try std.testing.expectEqual(@as(u8, 0), fsm.getCategory(1));
}

test "entriesPerPage calculation" {
    var fsm = FreeSpaceMap.init(std.testing.allocator, 4096);
    defer fsm.deinit();

    // (4096 - 16 - 8) / 8 = 4072 / 8 = 509
    try std.testing.expectEqual(@as(u32, 509), fsm.entriesPerPage());

    var fsm_small = FreeSpaceMap.init(std.testing.allocator, 512);
    defer fsm_small.deinit();

    // (512 - 16 - 8) / 8 = 488 / 8 = 61
    try std.testing.expectEqual(@as(u32, 61), fsm_small.entriesPerPage());
}

test "disk persistence roundtrip" {
    // Create a temporary database file
    const test_path = "test_fsm_persist.db";
    defer std.fs.cwd().deleteFile(test_path) catch {};

    var pager = try page_mod.Pager.init(std.testing.allocator, test_path, .{});
    defer pager.deinit();

    // Create and populate an FSM
    var fsm = FreeSpaceMap.init(std.testing.allocator, pager.page_size);
    defer fsm.deinit();

    try fsm.update(5, 1000);
    try fsm.update(10, 2000);
    try fsm.update(15, 3000);
    try fsm.update(20, 500);
    try fsm.update(25, 4072);

    // Save to disk
    const fsm_head = try fsm.saveToDisk(&pager);
    try std.testing.expect(fsm_head > 0);

    // Create a new FSM and load from disk
    var fsm2 = FreeSpaceMap.init(std.testing.allocator, pager.page_size);
    defer fsm2.deinit();

    try fsm2.loadFromDisk(&pager, fsm_head);

    // Verify all entries match
    try std.testing.expectEqual(fsm.trackedPages(), fsm2.trackedPages());
    try std.testing.expectEqual(fsm.getCategory(5), fsm2.getCategory(5));
    try std.testing.expectEqual(fsm.getCategory(10), fsm2.getCategory(10));
    try std.testing.expectEqual(fsm.getCategory(15), fsm2.getCategory(15));
    try std.testing.expectEqual(fsm.getCategory(20), fsm2.getCategory(20));
    try std.testing.expectEqual(fsm.getCategory(25), fsm2.getCategory(25));
}

test "disk persistence with many entries spanning multiple pages" {
    const test_path = "test_fsm_multi_page.db";
    defer std.fs.cwd().deleteFile(test_path) catch {};

    // Use small page size to force multiple FSM pages
    var pager = try page_mod.Pager.init(std.testing.allocator, test_path, .{ .page_size = 512 });
    defer pager.deinit();

    var fsm = FreeSpaceMap.init(std.testing.allocator, pager.page_size);
    defer fsm.deinit();

    // 61 entries per 512-byte page → add 150 entries to span 3 pages
    for (2..152) |i| {
        try fsm.update(@intCast(i), @intCast(100 + i));
    }

    try std.testing.expectEqual(@as(u32, 150), fsm.trackedPages());

    const fsm_head = try fsm.saveToDisk(&pager);
    try std.testing.expect(fsm_head > 0);

    // Load into new FSM
    var fsm2 = FreeSpaceMap.init(std.testing.allocator, pager.page_size);
    defer fsm2.deinit();

    try fsm2.loadFromDisk(&pager, fsm_head);
    try std.testing.expectEqual(@as(u32, 150), fsm2.trackedPages());

    // Spot-check some entries
    try std.testing.expectEqual(fsm.getCategory(50), fsm2.getCategory(50));
    try std.testing.expectEqual(fsm.getCategory(100), fsm2.getCategory(100));
    try std.testing.expectEqual(fsm.getCategory(151), fsm2.getCategory(151));
}

test "loadFromDisk with invalid head returns empty" {
    const test_path = "test_fsm_invalid.db";
    defer std.fs.cwd().deleteFile(test_path) catch {};

    var pager = try page_mod.Pager.init(std.testing.allocator, test_path, .{});
    defer pager.deinit();

    var fsm = FreeSpaceMap.init(std.testing.allocator, pager.page_size);
    defer fsm.deinit();

    // Head page 0 → no-op
    try fsm.loadFromDisk(&pager, 0);
    try std.testing.expectEqual(@as(u32, 0), fsm.trackedPages());

    // Head page beyond page_count → no-op
    try fsm.loadFromDisk(&pager, 999);
    try std.testing.expectEqual(@as(u32, 0), fsm.trackedPages());
}

test "empty FSM saveToDisk returns 0" {
    const test_path = "test_fsm_empty.db";
    defer std.fs.cwd().deleteFile(test_path) catch {};

    var pager = try page_mod.Pager.init(std.testing.allocator, test_path, .{});
    defer pager.deinit();

    var fsm = FreeSpaceMap.init(std.testing.allocator, pager.page_size);
    defer fsm.deinit();

    const head = try fsm.saveToDisk(&pager);
    try std.testing.expectEqual(@as(u32, 0), head);
}

test "different page sizes" {
    const sizes = [_]u32{ 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536 };
    for (sizes) |ps| {
        var fsm = FreeSpaceMap.init(std.testing.allocator, ps);
        defer fsm.deinit();

        const expected_usable: u16 = @intCast(ps - @as(u32, LEAF_OVERHEAD));
        try std.testing.expectEqual(expected_usable, fsm.max_usable);

        // Full → 255, empty → 0
        try std.testing.expectEqual(@as(u8, 255), fsm.bytesToCategory(expected_usable));
        try std.testing.expectEqual(@as(u8, 0), fsm.bytesToCategory(0));

        // Roundtrip
        try std.testing.expectEqual(expected_usable, fsm.categoryToMinBytes(255));
        try std.testing.expectEqual(@as(u16, 0), fsm.categoryToMinBytes(0));
    }
}

test "findPage with zero needed bytes" {
    var fsm = FreeSpaceMap.init(std.testing.allocator, 4096);
    defer fsm.deinit();

    try fsm.update(1, 1000);

    // 0 needed bytes → null (category 0, meaningless request)
    try std.testing.expectEqual(@as(?u32, null), fsm.findPage(0));
}

test "update multiple pages then remove some" {
    var fsm = FreeSpaceMap.init(std.testing.allocator, 4096);
    defer fsm.deinit();

    for (0..10) |i| {
        try fsm.update(@intCast(i + 2), @intCast(500 + i * 200));
    }
    try std.testing.expectEqual(@as(u32, 10), fsm.trackedPages());

    // Remove half
    for (0..5) |i| {
        fsm.remove(@intCast(i + 2));
    }
    try std.testing.expectEqual(@as(u32, 5), fsm.trackedPages());

    // Remaining pages should still be findable
    const page = fsm.findPage(500);
    try std.testing.expect(page != null);
    try std.testing.expect(page.? >= 7);
}

test "memory leak detection" {
    var fsm = FreeSpaceMap.init(std.testing.allocator, 4096);
    defer fsm.deinit();

    for (0..100) |i| {
        try fsm.update(@intCast(i), @intCast(100 + i * 10));
    }

    for (0..50) |i| {
        fsm.remove(@intCast(i));
    }

    fsm.clear();
    // std.testing.allocator will catch any leaks
}
