//! Overflow Pages — chain-based storage for large values that exceed inline capacity.
//!
//! When a B+Tree leaf cell's value is too large to fit inline within a single page,
//! the value is split: a prefix is stored inline and the remainder spills into a chain
//! of overflow pages. Each overflow page stores a next-page pointer followed by payload.
//!
//! Overflow page layout:
//!   [PageHeader 16B][next_page u32][payload...]
//!
//! The `next_page` field is 0 for the last page in the chain.
//!
//! Threshold: A cell is considered oversized when its total size exceeds
//! `maxInlineSize(page_size)`, which is roughly 1/4 of the usable leaf space.
//! When overflow is needed, we keep an inline prefix of the value and store
//! the overflow page ID (u32 LE) right after the inline prefix.

const std = @import("std");
const page_mod = @import("page.zig");
const buffer_pool_mod = @import("buffer_pool.zig");
const varint = @import("../util/varint.zig");

const Pager = page_mod.Pager;
const BufferPool = buffer_pool_mod.BufferPool;
const PageHeader = page_mod.PageHeader;
const PageType = page_mod.PageType;
const PAGE_HEADER_SIZE = page_mod.PAGE_HEADER_SIZE;

// ── Constants ──────────────────────────────────────────────────────────

/// Size of the next-page pointer at the start of overflow page content.
const OVERFLOW_NEXT_PTR_SIZE: u32 = 4;

/// Overflow page content starts after page header + next pointer.
const OVERFLOW_HEADER_SIZE: u32 = PAGE_HEADER_SIZE + OVERFLOW_NEXT_PTR_SIZE;

/// Size of the overflow page pointer stored at the end of an inline cell.
pub const OVERFLOW_PTR_SIZE: u32 = 4;

// ── Error Types ────────────────────────────────────────────────────────

pub const OverflowError = error{
    CorruptOverflowChain,
    PayloadTooLarge,
};

// ── Public API ─────────────────────────────────────────────────────────

/// Calculate the maximum cell size that can be stored inline (without overflow).
/// This is 1/4 of the usable leaf page space (page_size - page_header - leaf_header).
/// The leaf header is 8 bytes (prev_leaf + next_leaf).
pub fn maxInlineSize(page_size: u32) u32 {
    const usable = page_size - PAGE_HEADER_SIZE - 8; // 8 = leaf header (prev+next)
    return usable / 4;
}

/// For a value that overflows, calculate how many bytes of the value are stored inline.
/// We keep as much of the value inline as possible while leaving room for the overflow
/// page pointer (4 bytes). The inline prefix size is: maxInlineSize - key_overhead - OVERFLOW_PTR_SIZE.
///
/// Returns the number of value bytes to store inline.
pub fn inlineValueSize(page_size: u32, key: []const u8) u32 {
    const max_inline = maxInlineSize(page_size);
    const key_len_size: u32 = @intCast(varint.encodedLen(key.len));
    const key_overhead = key_len_size + @as(u32, @intCast(key.len));
    // value_len varint: we need to account for the full value length varint,
    // which encodes the TOTAL value length (not just inline portion).
    // In the worst case, a large value length varint takes up to 10 bytes.
    // We'll use a conservative estimate of 5 bytes for the value length varint
    // (covers up to 4 GB values).
    const val_len_varint_max: u32 = 5;
    const overhead = key_overhead + val_len_varint_max + OVERFLOW_PTR_SIZE;
    if (overhead >= max_inline) return 0;
    return max_inline - overhead;
}

/// Check whether a key-value pair requires overflow pages.
pub fn needsOverflow(page_size: u32, key: []const u8, value: []const u8) bool {
    const key_len_size: u32 = @intCast(varint.encodedLen(key.len));
    const val_len_size: u32 = @intCast(varint.encodedLen(value.len));
    const cell_size = key_len_size + @as(u32, @intCast(key.len)) + val_len_size + @as(u32, @intCast(value.len));
    return cell_size > maxInlineSize(page_size);
}

/// Payload capacity per overflow page.
pub fn overflowPageCapacity(page_size: u32) u32 {
    return page_size - OVERFLOW_HEADER_SIZE;
}

/// Write a value's overflow portion to a chain of overflow pages.
/// `overflow_data` is the portion of the value that doesn't fit inline.
/// Returns the page ID of the first overflow page in the chain.
pub fn writeOverflowChain(
    pool: *BufferPool,
    pager: *Pager,
    overflow_data: []const u8,
) !u32 {
    if (overflow_data.len == 0) return 0;

    const page_size = pager.page_size;
    const capacity = overflowPageCapacity(page_size);

    // Calculate how many pages we need
    const num_pages = (overflow_data.len + capacity - 1) / capacity;

    // Allocate all pages first, then link them
    var page_ids = try pager.allocator.alloc(u32, num_pages);
    defer pager.allocator.free(page_ids);

    for (0..num_pages) |i| {
        page_ids[i] = try pager.allocPage();
    }

    // Write data into pages (forward order, linking each to the next)
    var data_offset: usize = 0;
    for (0..num_pages) |i| {
        const frame = try pool.fetchNewPage(page_ids[i]);

        // Initialize overflow page header
        const hdr = PageHeader{
            .page_type = .overflow,
            .flags = 0,
            .cell_count = 0,
            .page_id = page_ids[i],
            .free_offset = 0,
            .checksum_value = 0,
        };
        hdr.serialize(frame.data[0..PAGE_HEADER_SIZE]);

        // Write next-page pointer
        const next_page: u32 = if (i + 1 < num_pages) page_ids[i + 1] else 0;
        std.mem.writeInt(u32, frame.data[PAGE_HEADER_SIZE..][0..4], next_page, .little);

        // Write payload
        const chunk_start = data_offset;
        const chunk_end = @min(data_offset + capacity, overflow_data.len);
        const chunk_len = chunk_end - chunk_start;
        @memcpy(frame.data[OVERFLOW_HEADER_SIZE..][0..chunk_len], overflow_data[chunk_start..chunk_end]);

        // Store actual payload size in cell_count field (reused for overflow pages)
        // This helps us know exactly how much data is on this page.
        std.mem.writeInt(u16, frame.data[2..4], @intCast(chunk_len), .little);

        pool.unpinPage(page_ids[i], true);
        data_offset = chunk_end;
    }

    return page_ids[0];
}

/// Read the full overflow value by following the chain starting at `first_page_id`.
/// `inline_prefix` is the portion of the value stored inline in the leaf cell.
/// `total_value_len` is the total length of the complete value.
/// Caller owns the returned slice and must free it with `allocator`.
pub fn readOverflowValue(
    allocator: std.mem.Allocator,
    pool: *BufferPool,
    page_size: u32,
    first_overflow_page: u32,
    inline_prefix: []const u8,
    total_value_len: usize,
) ![]u8 {
    const result = try allocator.alloc(u8, total_value_len);
    errdefer allocator.free(result);

    // Copy inline prefix
    @memcpy(result[0..inline_prefix.len], inline_prefix);

    var offset: usize = inline_prefix.len;
    var current_page = first_overflow_page;
    const capacity = overflowPageCapacity(page_size);

    while (current_page != 0 and offset < total_value_len) {
        const frame = try pool.fetchPage(current_page);
        defer pool.unpinPage(current_page, false);

        // Verify page type
        const hdr = PageHeader.deserialize(frame.data[0..PAGE_HEADER_SIZE]);
        if (hdr.page_type != .overflow) {
            return OverflowError.CorruptOverflowChain;
        }

        // Read payload size from cell_count field
        const chunk_len: usize = hdr.cell_count;
        const actual_len = @min(chunk_len, total_value_len - offset);

        // Sanity check
        if (actual_len > capacity) {
            return OverflowError.CorruptOverflowChain;
        }

        @memcpy(result[offset..][0..actual_len], frame.data[OVERFLOW_HEADER_SIZE..][0..actual_len]);
        offset += actual_len;

        // Read next page pointer
        current_page = std.mem.readInt(u32, frame.data[PAGE_HEADER_SIZE..][0..4], .little);
    }

    if (offset != total_value_len) {
        return OverflowError.CorruptOverflowChain;
    }

    return result;
}

/// Free all pages in an overflow chain starting at `first_page_id`.
/// Follows the next-page pointers and frees each page back to the freelist.
pub fn freeOverflowChain(
    pool: *BufferPool,
    pager: *Pager,
    first_page_id: u32,
) !void {
    var current_page = first_page_id;

    while (current_page != 0) {
        const frame = try pool.fetchPage(current_page);
        const hdr = PageHeader.deserialize(frame.data[0..PAGE_HEADER_SIZE]);

        if (hdr.page_type != .overflow) {
            pool.unpinPage(current_page, false);
            return OverflowError.CorruptOverflowChain;
        }

        // Read next pointer before freeing
        const next_page = std.mem.readInt(u32, frame.data[PAGE_HEADER_SIZE..][0..4], .little);
        pool.unpinPage(current_page, false);

        try pager.freePage(current_page);
        current_page = next_page;
    }
}

/// Calculate the total number of overflow pages needed for a given overflow data size.
pub fn overflowPageCount(page_size: u32, overflow_data_len: usize) u32 {
    if (overflow_data_len == 0) return 0;
    const capacity = overflowPageCapacity(page_size);
    return @intCast((overflow_data_len + capacity - 1) / capacity);
}

// ── Tests ──────────────────────────────────────────────────────────────

test "maxInlineSize - default page size" {
    // page_size=4096, usable=4096-16-8=4072, max_inline=1018
    const max = maxInlineSize(4096);
    try std.testing.expectEqual(@as(u32, 1018), max);
}

test "maxInlineSize - minimum page size" {
    // page_size=512, usable=512-16-8=488, max_inline=122
    const max = maxInlineSize(512);
    try std.testing.expectEqual(@as(u32, 122), max);
}

test "maxInlineSize - maximum page size" {
    // page_size=65536, usable=65536-16-8=65512, max_inline=16378
    const max = maxInlineSize(65536);
    try std.testing.expectEqual(@as(u32, 16378), max);
}

test "needsOverflow - small value fits inline" {
    // key=10 bytes, value=100 bytes. cell_size = 1+10+1+100 = 112
    // max_inline(4096) = 1018, so 112 < 1018 → no overflow
    const key = "0123456789";
    const value = "a" ** 100;
    try std.testing.expect(!needsOverflow(4096, key, value));
}

test "needsOverflow - large value needs overflow" {
    // key=10 bytes, value=2000 bytes. cell_size = 1+10+2+2000 = 2013
    // max_inline(4096) = 1018, so 2013 > 1018 → overflow needed
    const key = "0123456789";
    const value = "a" ** 2000;
    try std.testing.expect(needsOverflow(4096, key, value));
}

test "needsOverflow - borderline case" {
    // max_inline(4096) = 1018
    // key=4 bytes → key_len_varint=1, cell = 1+4+val_len_varint+val
    // For val of 1011 bytes: val_len_varint=2, cell = 1+4+2+1011 = 1018 → exactly fits
    const key = "abcd";
    const value_fits = "x" ** 1011;
    try std.testing.expect(!needsOverflow(4096, key, value_fits));

    // For val of 1012 bytes: val_len_varint=2, cell = 1+4+2+1012 = 1019 → overflow
    const value_overflow = "x" ** 1012;
    try std.testing.expect(needsOverflow(4096, key, value_overflow));
}

test "overflowPageCapacity - default page size" {
    // page_size=4096, capacity = 4096 - 16 - 4 = 4076
    try std.testing.expectEqual(@as(u32, 4076), overflowPageCapacity(4096));
}

test "overflowPageCount - various sizes" {
    const cap = overflowPageCapacity(4096); // 4076
    try std.testing.expectEqual(@as(u32, 0), overflowPageCount(4096, 0));
    try std.testing.expectEqual(@as(u32, 1), overflowPageCount(4096, 1));
    try std.testing.expectEqual(@as(u32, 1), overflowPageCount(4096, cap));
    try std.testing.expectEqual(@as(u32, 2), overflowPageCount(4096, cap + 1));
    try std.testing.expectEqual(@as(u32, 3), overflowPageCount(4096, cap * 2 + 1));
}

test "inlineValueSize - calculation" {
    const key = "test_key"; // 8 bytes, varint=1
    const inline_size = inlineValueSize(4096, key);
    // max_inline=1018, key_overhead=1+8=9, val_len_varint_max=5, overflow_ptr=4
    // inline = 1018 - 9 - 5 - 4 = 1000
    try std.testing.expectEqual(@as(u32, 1000), inline_size);
}

test "write and read overflow chain - single page" {
    const allocator = std.testing.allocator;
    const page_size: u32 = 4096;
    const test_path = "test_overflow_single.db";

    var pager = try Pager.init(allocator, test_path, .{ .page_size = page_size });
    defer {
        pager.deinit();
        std.fs.cwd().deleteFile(test_path) catch {};
    }

    var pool = try BufferPool.init(allocator, &pager, 100);
    defer pool.deinit();

    // Write 100 bytes of overflow data
    const overflow_data = "X" ** 100;
    const first_page = try writeOverflowChain(&pool, &pager, overflow_data);
    try std.testing.expect(first_page != 0);

    // Read it back with empty inline prefix
    const result = try readOverflowValue(allocator, &pool, page_size, first_page, "", overflow_data.len);
    defer allocator.free(result);

    try std.testing.expectEqualSlices(u8, overflow_data, result);
}

test "write and read overflow chain - multiple pages" {
    const allocator = std.testing.allocator;
    const page_size: u32 = 4096;
    const test_path = "test_overflow_multi.db";

    var pager = try Pager.init(allocator, test_path, .{ .page_size = page_size });
    defer {
        pager.deinit();
        std.fs.cwd().deleteFile(test_path) catch {};
    }

    var pool = try BufferPool.init(allocator, &pager, 100);
    defer pool.deinit();

    // Write data that spans 3 overflow pages
    const capacity = overflowPageCapacity(page_size); // 4076
    const total_len = capacity * 2 + 500; // 8652 bytes → 3 pages
    const overflow_data = try allocator.alloc(u8, total_len);
    defer allocator.free(overflow_data);

    // Fill with a recognizable pattern
    for (0..total_len) |i| {
        overflow_data[i] = @intCast(i % 251); // prime modulus for variety
    }

    const first_page = try writeOverflowChain(&pool, &pager, overflow_data);
    try std.testing.expect(first_page != 0);

    // Verify page count
    try std.testing.expectEqual(@as(u32, 3), overflowPageCount(page_size, total_len));

    // Read it back
    const result = try readOverflowValue(allocator, &pool, page_size, first_page, "", total_len);
    defer allocator.free(result);

    try std.testing.expectEqualSlices(u8, overflow_data, result);
}

test "write and read overflow chain - with inline prefix" {
    const allocator = std.testing.allocator;
    const page_size: u32 = 4096;
    const test_path = "test_overflow_prefix.db";

    var pager = try Pager.init(allocator, test_path, .{ .page_size = page_size });
    defer {
        pager.deinit();
        std.fs.cwd().deleteFile(test_path) catch {};
    }

    var pool = try BufferPool.init(allocator, &pager, 100);
    defer pool.deinit();

    // Simulate a value where 200 bytes are inline and 500 bytes overflow
    const inline_prefix = "I" ** 200;
    const overflow_data = "O" ** 500;
    const total_len = 700;

    const first_page = try writeOverflowChain(&pool, &pager, overflow_data);

    // Read complete value
    const result = try readOverflowValue(allocator, &pool, page_size, first_page, inline_prefix, total_len);
    defer allocator.free(result);

    // Verify inline prefix
    try std.testing.expectEqualSlices(u8, inline_prefix, result[0..200]);
    // Verify overflow portion
    try std.testing.expectEqualSlices(u8, overflow_data, result[200..700]);
}

test "free overflow chain" {
    const allocator = std.testing.allocator;
    const page_size: u32 = 4096;
    const test_path = "test_overflow_free.db";

    var pager = try Pager.init(allocator, test_path, .{ .page_size = page_size });
    defer {
        pager.deinit();
        std.fs.cwd().deleteFile(test_path) catch {};
    }

    var pool = try BufferPool.init(allocator, &pager, 100);
    defer pool.deinit();

    // Write a chain
    const overflow_data = "Z" ** 5000;
    const first_page = try writeOverflowChain(&pool, &pager, overflow_data);

    // Record page count before freeing
    const pages_before = pager.page_count;

    // Free the chain
    try freeOverflowChain(&pool, &pager, first_page);

    // Allocating new pages should reuse freed ones
    const new_page1 = try pager.allocPage();
    const new_page2 = try pager.allocPage();

    // The freed pages should be reused (page count shouldn't grow much)
    _ = new_page1;
    _ = new_page2;
    // The total page count should not have grown since we freed and re-allocated
    try std.testing.expect(pager.page_count <= pages_before);
}

test "overflow chain - empty data returns zero" {
    const allocator = std.testing.allocator;
    const page_size: u32 = 4096;
    const test_path = "test_overflow_empty.db";

    var pager = try Pager.init(allocator, test_path, .{ .page_size = page_size });
    defer {
        pager.deinit();
        std.fs.cwd().deleteFile(test_path) catch {};
    }

    var pool = try BufferPool.init(allocator, &pager, 10);
    defer pool.deinit();

    // Empty overflow data should return 0 (no pages allocated)
    const first_page = try writeOverflowChain(&pool, &pager, "");
    try std.testing.expectEqual(@as(u32, 0), first_page);
}

test "overflow chain - exact page boundary" {
    const allocator = std.testing.allocator;
    const page_size: u32 = 4096;
    const test_path = "test_overflow_exact.db";

    var pager = try Pager.init(allocator, test_path, .{ .page_size = page_size });
    defer {
        pager.deinit();
        std.fs.cwd().deleteFile(test_path) catch {};
    }

    var pool = try BufferPool.init(allocator, &pager, 100);
    defer pool.deinit();

    // Data exactly fills one overflow page
    const capacity = overflowPageCapacity(page_size);
    const overflow_data = try allocator.alloc(u8, capacity);
    defer allocator.free(overflow_data);
    @memset(overflow_data, 0xAB);

    const first_page = try writeOverflowChain(&pool, &pager, overflow_data);
    try std.testing.expectEqual(@as(u32, 1), overflowPageCount(page_size, capacity));

    const result = try readOverflowValue(allocator, &pool, page_size, first_page, "", capacity);
    defer allocator.free(result);

    try std.testing.expectEqualSlices(u8, overflow_data, result);
}

test "overflow chain - large multi-page chain" {
    const allocator = std.testing.allocator;
    const page_size: u32 = 4096;
    const test_path = "test_overflow_large.db";

    var pager = try Pager.init(allocator, test_path, .{ .page_size = page_size });
    defer {
        pager.deinit();
        std.fs.cwd().deleteFile(test_path) catch {};
    }

    var pool = try BufferPool.init(allocator, &pager, 200);
    defer pool.deinit();

    // Write 50KB of data → ~13 overflow pages
    const total_len: usize = 50000;
    const overflow_data = try allocator.alloc(u8, total_len);
    defer allocator.free(overflow_data);

    for (0..total_len) |i| {
        overflow_data[i] = @intCast((i * 7 + 13) % 256);
    }

    const first_page = try writeOverflowChain(&pool, &pager, overflow_data);

    const result = try readOverflowValue(allocator, &pool, page_size, first_page, "", total_len);
    defer allocator.free(result);

    try std.testing.expectEqualSlices(u8, overflow_data, result);
}

test "overflow chain - minimum page size stress" {
    const allocator = std.testing.allocator;
    const page_size: u32 = 512;
    const test_path = "test_overflow_minpage.db";

    var pager = try Pager.init(allocator, test_path, .{ .page_size = page_size });
    defer {
        pager.deinit();
        std.fs.cwd().deleteFile(test_path) catch {};
    }

    var pool = try BufferPool.init(allocator, &pager, 200);
    defer pool.deinit();

    // With 512-byte pages, overflow capacity = 512-16-4 = 492 bytes per page
    // 2000 bytes → 5 pages
    const overflow_data = "M" ** 2000;
    const first_page = try writeOverflowChain(&pool, &pager, overflow_data);

    try std.testing.expectEqual(@as(u32, 5), overflowPageCount(page_size, 2000));

    const result = try readOverflowValue(allocator, &pool, page_size, first_page, "", 2000);
    defer allocator.free(result);

    try std.testing.expectEqualSlices(u8, overflow_data, result);
}

test "free and rewrite overflow chain" {
    const allocator = std.testing.allocator;
    const page_size: u32 = 4096;
    const test_path = "test_overflow_rewrite.db";

    var pager = try Pager.init(allocator, test_path, .{ .page_size = page_size });
    defer {
        pager.deinit();
        std.fs.cwd().deleteFile(test_path) catch {};
    }

    var pool = try BufferPool.init(allocator, &pager, 100);
    defer pool.deinit();

    // Write first chain
    const data1 = "A" ** 5000;
    const first1 = try writeOverflowChain(&pool, &pager, data1);

    // Free it
    try freeOverflowChain(&pool, &pager, first1);

    // Write a different chain — should reuse pages
    const data2 = "B" ** 3000;
    const first2 = try writeOverflowChain(&pool, &pager, data2);

    // Read back and verify
    const result = try readOverflowValue(allocator, &pool, page_size, first2, "", 3000);
    defer allocator.free(result);

    try std.testing.expectEqualSlices(u8, data2, result);
}
