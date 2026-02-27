//! B+Tree — Primary index structure for Silica.
//!
//! Uses a slotted-page layout where cell pointers grow from the front of the
//! content area and cell data grows from the back. Keys are variable-length
//! byte strings compared lexicographically. Values are opaque byte strings.
//!
//! Page layouts:
//!   Leaf:     [PageHeader 16B][prev_leaf 4B][next_leaf 4B][cell_ptrs...] ... [cells←]
//!   Internal: [PageHeader 16B][right_child 4B][cell_ptrs...] ... [cells←]
//!
//! Leaf cell:     [key_len varint][key_data][value_len varint][value_data]
//! Internal cell: [left_child u32 LE][key_len varint][key_data]

const std = @import("std");
const page_mod = @import("page.zig");
const buffer_pool_mod = @import("buffer_pool.zig");
const varint = @import("../util/varint.zig");

const Pager = page_mod.Pager;
const BufferPool = buffer_pool_mod.BufferPool;
const BufferFrame = buffer_pool_mod.BufferFrame;
const PageHeader = page_mod.PageHeader;
const PAGE_HEADER_SIZE = page_mod.PAGE_HEADER_SIZE;
const PageType = page_mod.PageType;

// ── Constants ──────────────────────────────────────────────────────────

/// Size of the cell pointer (offset within page, u16).
const CELL_PTR_SIZE: u16 = 2;

/// Leaf page: bytes after page header reserved for sibling pointers.
/// [prev_leaf: u32][next_leaf: u32] = 8 bytes.
const LEAF_HEADER_SIZE: u16 = 8;

/// Internal page: bytes after page header reserved for right-most child pointer.
/// [right_child: u32] = 4 bytes.
const INTERNAL_HEADER_SIZE: u16 = 4;

// ── Error Types ────────────────────────────────────────────────────────

pub const BTreeError = error{
    KeyNotFound,
    DuplicateKey,
    PageCorrupt,
    NodeFull,
    InvalidNodeType,
    KeyTooLarge,
    ValueTooLarge,
};

/// Error type for insert operations — uses anyerror to support recursive calls
/// with the complex error set from Pager + BufferPool + BTree.
const InsertError = anyerror;

// ── B+Tree ─────────────────────────────────────────────────────────────

pub const BTree = struct {
    pool: *BufferPool,
    pager: *Pager,
    root_page_id: u32,

    pub fn init(pool: *BufferPool, root_page_id: u32) BTree {
        return .{
            .pool = pool,
            .pager = pool.pager,
            .root_page_id = root_page_id,
        };
    }

    // ── Point Lookup ───────────────────────────────────────────────────

    /// Look up a key and return its value. Caller must free the returned slice.
    pub fn get(self: *BTree, allocator: std.mem.Allocator, key: []const u8) !?[]u8 {
        var page_id = self.root_page_id;

        while (true) {
            const frame = try self.pool.fetchPage(page_id);
            defer self.pool.unpinPage(page_id, false);

            const header = PageHeader.deserialize(frame.data[0..PAGE_HEADER_SIZE]);

            switch (header.page_type) {
                .internal => {
                    page_id = findChildInInternal(frame.data, self.pager.page_size, header.cell_count, key);
                },
                .leaf => {
                    return findInLeaf(allocator, frame.data, self.pager.page_size, header.cell_count, key);
                },
                else => return BTreeError.InvalidNodeType,
            }
        }
    }

    // ── Insert ─────────────────────────────────────────────────────────

    /// Insert a key-value pair. Returns error.DuplicateKey if key already exists.
    pub fn insert(self: *BTree, key: []const u8, value: []const u8) InsertError!void {
        const result = try self.insertIntoNode(self.root_page_id, key, value);
        if (result.split) |split| {
            // Root was split — create a new root
            const new_root_id = try self.pager.allocPage();
            const frame = try self.pool.fetchNewPage(new_root_id);

            initInternalPage(frame.data, self.pager.page_size, new_root_id);
            // Set right_child to the new (right) page
            setRightChild(frame.data, split.new_page_id);
            // Insert the promoted key with left_child = old root
            insertInternalCell(frame.data, self.pager.page_size, 0, 0, self.root_page_id, split.promoted_key) catch unreachable;

            self.pool.unpinPage(new_root_id, true);

            self.root_page_id = new_root_id;
        }
    }

    /// Recursive insert. Returns split info if the node was split.
    fn insertIntoNode(self: *BTree, page_id: u32, key: []const u8, value: []const u8) InsertError!InsertResult {
        const frame = try self.pool.fetchPage(page_id);
        const header = PageHeader.deserialize(frame.data[0..PAGE_HEADER_SIZE]);

        switch (header.page_type) {
            .leaf => {
                return self.insertIntoLeaf(frame, page_id, key, value);
            },
            .internal => {
                return self.insertIntoInternal(frame, page_id, key, value);
            },
            else => {
                self.pool.unpinPage(page_id, false);
                return BTreeError.InvalidNodeType;
            },
        }
    }

    fn insertIntoLeaf(self: *BTree, frame: *BufferFrame, page_id: u32, key: []const u8, value: []const u8) InsertError!InsertResult {
        const page_size = self.pager.page_size;
        const header = PageHeader.deserialize(frame.data[0..PAGE_HEADER_SIZE]);
        const cell_count = header.cell_count;

        // Find insertion position via binary search
        const pos = leafSearchPosition(frame.data, page_size, cell_count, key);

        // Check for duplicate key
        if (pos.found) {
            self.pool.unpinPage(page_id, false);
            return BTreeError.DuplicateKey;
        }

        // Try to insert into this leaf
        const cell_size = leafCellSize(key, value);
        const free_space = leafFreeSpace(frame.data, page_size, cell_count);

        if (free_space >= cell_size + CELL_PTR_SIZE) {
            // Fits — insert directly
            insertLeafCell(frame.data, page_size, cell_count, pos.index, key, value) catch {
                self.pool.unpinPage(page_id, false);
                return BTreeError.PageCorrupt;
            };
            self.pool.unpinPage(page_id, true);
            return .{ .split = null };
        }

        // Need to split
        const split = try self.splitLeaf(frame, page_id, pos.index, key, value);
        return .{ .split = split };
    }

    fn insertIntoInternal(self: *BTree, frame: *BufferFrame, page_id: u32, key: []const u8, value: []const u8) InsertError!InsertResult {
        const page_size = self.pager.page_size;
        const header = PageHeader.deserialize(frame.data[0..PAGE_HEADER_SIZE]);
        const cell_count = header.cell_count;

        // Find which child to descend into
        const child_page_id = findChildInInternal(frame.data, page_size, cell_count, key);
        self.pool.unpinPage(page_id, false);

        // Recurse
        const child_result = try self.insertIntoNode(child_page_id, key, value);

        if (child_result.split) |split| {
            // Child was split — insert promoted key into this internal node
            const re_frame = try self.pool.fetchPage(page_id);
            const re_header = PageHeader.deserialize(re_frame.data[0..PAGE_HEADER_SIZE]);
            const re_count = re_header.cell_count;

            const insert_pos = internalSearchPosition(re_frame.data, page_size, re_count, split.promoted_key);

            const cell_size = internalCellSize(split.promoted_key);
            const free = internalFreeSpace(re_frame.data, page_size, re_count);

            if (free >= cell_size + CELL_PTR_SIZE) {
                // Update the child pointer: the cell at insert_pos currently points to
                // child_page_id. After insert, the new cell at insert_pos will point to
                // child_page_id (left side of split), and the cell at insert_pos+1 or
                // right_child will point to split.new_page_id.

                // First, we need to update the pointer that currently leads to child_page_id
                // to instead point to split.new_page_id for keys >= promoted key.
                // The promoted key's left_child is child_page_id (the original, left half).
                // The cell to the right (or right_child) should become split.new_page_id.
                updateChildPointer(re_frame.data, page_size, re_count, insert_pos, split.new_page_id);
                insertInternalCell(re_frame.data, page_size, re_count, insert_pos, child_page_id, split.promoted_key) catch {
                    self.pool.unpinPage(page_id, false);
                    return BTreeError.PageCorrupt;
                };
                self.pool.unpinPage(page_id, true);
                return .{ .split = null };
            }

            // Need to split internal node
            const internal_split = try self.splitInternal(re_frame, page_id, insert_pos, child_page_id, split.new_page_id, split.promoted_key);
            return .{ .split = internal_split };
        }

        return .{ .split = null };
    }

    // ── Delete ─────────────────────────────────────────────────────────

    /// Delete a key. Returns error.KeyNotFound if key does not exist.
    /// Note: This is a simple delete that does not perform merges/rebalancing.
    /// Underflow handling (merges) will be implemented in Milestone 2C.
    pub fn delete(self: *BTree, key: []const u8) !void {
        try self.deleteFromNode(self.root_page_id, key);
    }

    fn deleteFromNode(self: *BTree, page_id: u32, key: []const u8) !void {
        const frame = try self.pool.fetchPage(page_id);
        const header = PageHeader.deserialize(frame.data[0..PAGE_HEADER_SIZE]);

        switch (header.page_type) {
            .leaf => {
                const pos = leafSearchPosition(frame.data, self.pager.page_size, header.cell_count, key);
                if (!pos.found) {
                    self.pool.unpinPage(page_id, false);
                    return BTreeError.KeyNotFound;
                }
                deleteLeafCell(frame.data, self.pager.page_size, header.cell_count, pos.index);
                self.pool.unpinPage(page_id, true);
            },
            .internal => {
                const child_page_id = findChildInInternal(frame.data, self.pager.page_size, header.cell_count, key);
                self.pool.unpinPage(page_id, false);
                try self.deleteFromNode(child_page_id, key);
            },
            else => {
                self.pool.unpinPage(page_id, false);
                return BTreeError.InvalidNodeType;
            },
        }
    }

    // ── Split Operations ───────────────────────────────────────────────

    fn splitLeaf(self: *BTree, frame: *BufferFrame, page_id: u32, insert_pos: u16, key: []const u8, value: []const u8) InsertError!SplitInfo {
        const page_size = self.pager.page_size;
        const header = PageHeader.deserialize(frame.data[0..PAGE_HEADER_SIZE]);
        const old_count = header.cell_count;

        // Save old page data — cell refs point into frame.data which we'll reinit
        const saved = try self.pager.allocator.alloc(u8, page_size);
        defer self.pager.allocator.free(saved);
        @memcpy(saved, frame.data[0..page_size]);

        // Collect all existing cells (from saved copy) + the new one
        const total = @as(u32, old_count) + 1;
        var cells = try self.pager.allocator.alloc(CellRef, total);
        defer self.pager.allocator.free(cells);

        var idx: u32 = 0;
        for (0..old_count) |i| {
            if (i == insert_pos) {
                cells[idx] = .{ .key = key, .value = value, .from_new = true };
                idx += 1;
            }
            const cell = readLeafCell(saved, page_size, @intCast(i));
            cells[idx] = .{ .key = cell.key, .value = cell.value, .from_new = false };
            idx += 1;
        }
        if (insert_pos == old_count) {
            cells[idx] = .{ .key = key, .value = value, .from_new = true };
        }

        // Split point: keep first half in old page, second half in new page
        const split_point: u32 = total / 2;

        // Create new right leaf page
        const new_page_id = try self.pager.allocPage();
        const new_frame = try self.pool.fetchNewPage(new_page_id);

        initLeafPage(new_frame.data, page_size, new_page_id);

        // Read old leaf's sibling pointers from saved copy
        const old_next = getNextLeaf(saved);

        // Reinitialize old page (safe now — cells reference saved copy)
        initLeafPage(frame.data, page_size, page_id);

        // Write left half into old page
        for (0..split_point) |i| {
            insertLeafCell(frame.data, page_size, @intCast(i), @intCast(i), cells[i].key, cells[i].value) catch unreachable;
        }

        // Write right half into new page
        for (split_point..total) |i| {
            const j: u16 = @intCast(i - split_point);
            insertLeafCell(new_frame.data, page_size, j, j, cells[i].key, cells[i].value) catch unreachable;
        }

        // Update sibling pointers: old -> new -> old_next
        setNextLeaf(frame.data, new_page_id);
        setPrevLeaf(new_frame.data, page_id);
        setNextLeaf(new_frame.data, old_next);

        // If there was a next leaf, update its prev pointer
        if (old_next != 0) {
            const next_frame = try self.pool.fetchPage(old_next);
            setPrevLeaf(next_frame.data, new_page_id);
            self.pool.unpinPage(old_next, true);
        }

        // The promoted key is the first key of the new (right) page
        const promoted = readLeafCell(new_frame.data, page_size, 0);

        self.pool.unpinPage(page_id, true);
        self.pool.unpinPage(new_page_id, true);

        return SplitInfo{
            .promoted_key = promoted.key,
            .new_page_id = new_page_id,
        };
    }

    fn splitInternal(self: *BTree, frame: *BufferFrame, page_id: u32, insert_pos: u16, left_child: u32, right_child: u32, key: []const u8) InsertError!SplitInfo {
        _ = right_child; // already applied via updateChildPointer before this call
        const page_size = self.pager.page_size;
        const header = PageHeader.deserialize(frame.data[0..PAGE_HEADER_SIZE]);
        const old_count = header.cell_count;

        // Save page data — cell key refs point into frame.data
        const saved = try self.pager.allocator.alloc(u8, page_size);
        defer self.pager.allocator.free(saved);
        @memcpy(saved, frame.data[0..page_size]);

        // Collect all existing cells (from saved copy) + the new one
        const total = @as(u32, old_count) + 1;
        var cells = try self.pager.allocator.alloc(InternalCellRef, total);
        defer self.pager.allocator.free(cells);

        var idx: u32 = 0;
        for (0..old_count) |i| {
            if (i == insert_pos) {
                cells[idx] = .{ .left_child = left_child, .key = key };
                idx += 1;
            }
            const cell = readInternalCell(saved, page_size, @intCast(i));
            cells[idx] = .{ .left_child = cell.left_child, .key = cell.key };
            idx += 1;
        }
        if (insert_pos == old_count) {
            cells[idx] = .{ .left_child = left_child, .key = key };
        }

        const split_point: u32 = total / 2;
        const promoted_key = cells[split_point].key;
        const old_right_child = cells[split_point].left_child;
        const original_right_child = getRightChild(saved);

        // Create new right internal page
        const new_page_id = try self.pager.allocPage();
        const new_frame = try self.pool.fetchNewPage(new_page_id);
        initInternalPage(new_frame.data, page_size, new_page_id);

        // Reinitialize old page (safe — cells reference saved copy)
        initInternalPage(frame.data, page_size, page_id);

        // Write left half
        for (0..split_point) |i| {
            insertInternalCell(frame.data, page_size, @intCast(i), @intCast(i), cells[i].left_child, cells[i].key) catch unreachable;
        }
        setRightChild(frame.data, old_right_child);

        // Write right half (skip split_point — it's promoted)
        for ((split_point + 1)..total) |i| {
            const j: u16 = @intCast(i - split_point - 1);
            insertInternalCell(new_frame.data, page_size, j, j, cells[i].left_child, cells[i].key) catch unreachable;
        }
        setRightChild(new_frame.data, original_right_child);

        self.pool.unpinPage(page_id, true);
        self.pool.unpinPage(new_page_id, true);

        return SplitInfo{
            .promoted_key = promoted_key,
            .new_page_id = new_page_id,
        };
    }
};

// ── Result Types ───────────────────────────────────────────────────────

const SplitInfo = struct {
    promoted_key: []const u8,
    new_page_id: u32,
};

const InsertResult = struct {
    split: ?SplitInfo,
};

const CellRef = struct {
    key: []const u8,
    value: []const u8,
    from_new: bool,
};

const InternalCellRef = struct {
    left_child: u32,
    key: []const u8,
};

const SearchResult = struct {
    index: u16,
    found: bool,
};

// ── Page Initialization ────────────────────────────────────────────────

/// Initialize a page buffer as an empty leaf node.
pub fn initLeafPage(data: []u8, page_size: u32, page_id: u32) void {
    @memset(data[0..page_size], 0);
    const header = PageHeader{
        .page_type = .leaf,
        .page_id = page_id,
        .cell_count = 0,
        .free_offset = 0,
    };
    header.serialize(data[0..PAGE_HEADER_SIZE]);
    // prev_leaf = 0, next_leaf = 0 (already zeroed)
}

/// Initialize a page buffer as an empty internal node.
pub fn initInternalPage(data: []u8, page_size: u32, page_id: u32) void {
    @memset(data[0..page_size], 0);
    const header = PageHeader{
        .page_type = .internal,
        .page_id = page_id,
        .cell_count = 0,
        .free_offset = 0,
    };
    header.serialize(data[0..PAGE_HEADER_SIZE]);
    // right_child = 0 (already zeroed)
}

// ── Leaf Page Operations ───────────────────────────────────────────────

/// Content area start offset for leaf pages.
fn leafContentStart() u16 {
    return PAGE_HEADER_SIZE + LEAF_HEADER_SIZE;
}

/// Get the offset where cell pointers begin for a leaf page.
fn leafCellPtrOffset(index: u16) u16 {
    return leafContentStart() + index * CELL_PTR_SIZE;
}

/// Get the cell data offset from a cell pointer.
fn readCellPtr(data: []const u8, ptr_offset: u16) u16 {
    return std.mem.readInt(u16, data[ptr_offset..][0..2], .little);
}

/// Write a cell pointer.
fn writeCellPtr(data: []u8, ptr_offset: u16, cell_offset: u16) void {
    std.mem.writeInt(u16, data[ptr_offset..][0..2], cell_offset, .little);
}

/// Calculate the size of a leaf cell (without cell pointer).
fn leafCellSize(key: []const u8, value: []const u8) u16 {
    const key_len_size: u16 = @intCast(varint.encodedLen(key.len));
    const val_len_size: u16 = @intCast(varint.encodedLen(value.len));
    return key_len_size + @as(u16, @intCast(key.len)) + val_len_size + @as(u16, @intCast(value.len));
}

/// Calculate free space in a leaf page.
fn leafFreeSpace(data: []const u8, page_size: u32, cell_count: u16) u16 {
    const ptrs_end = leafCellPtrOffset(cell_count);
    const cells_start = if (cell_count == 0) @as(u16, @intCast(page_size)) else lowestCellOffset(data, cell_count, true);
    if (cells_start <= ptrs_end) return 0;
    return cells_start - ptrs_end;
}

/// Find the lowest cell offset (first cell data byte) among all cells.
fn lowestCellOffset(data: []const u8, cell_count: u16, is_leaf: bool) u16 {
    var lowest: u16 = std.math.maxInt(u16);
    const base = if (is_leaf) leafContentStart() else internalContentStart();
    for (0..cell_count) |i| {
        const ptr_off = base + @as(u16, @intCast(i)) * CELL_PTR_SIZE;
        const cell_off = readCellPtr(data, ptr_off);
        if (cell_off < lowest) lowest = cell_off;
    }
    return lowest;
}

/// Read a leaf cell at the given index.
const LeafCell = struct {
    key: []const u8,
    value: []const u8,
};

fn readLeafCell(data: []const u8, page_size: u32, index: u16) LeafCell {
    _ = page_size;
    const ptr_off = leafCellPtrOffset(index);
    const cell_off = readCellPtr(data, ptr_off);

    var offset: usize = cell_off;

    // Read key
    const key_dec = varint.decode(data[offset..]) catch return .{ .key = &.{}, .value = &.{} };
    offset += key_dec.bytes_read;
    const key_len: usize = @intCast(key_dec.value);
    const key_data = data[offset..][0..key_len];
    offset += key_len;

    // Read value
    const val_dec = varint.decode(data[offset..]) catch return .{ .key = key_data, .value = &.{} };
    offset += val_dec.bytes_read;
    const val_len: usize = @intCast(val_dec.value);
    const val_data = data[offset..][0..val_len];

    return .{ .key = key_data, .value = val_data };
}

/// Binary search for a key in a leaf page. Returns position and whether found.
fn leafSearchPosition(data: []const u8, page_size: u32, cell_count: u16, key: []const u8) SearchResult {
    if (cell_count == 0) return .{ .index = 0, .found = false };

    var low: u16 = 0;
    var high: u16 = cell_count;

    while (low < high) {
        const mid = low + (high - low) / 2;
        const cell = readLeafCell(data, page_size, mid);

        switch (std.mem.order(u8, cell.key, key)) {
            .lt => low = mid + 1,
            .gt => high = mid,
            .eq => return .{ .index = mid, .found = true },
        }
    }

    return .{ .index = low, .found = false };
}

/// Find a key's value in a leaf page. Returns owned copy or null.
fn findInLeaf(allocator: std.mem.Allocator, data: []const u8, page_size: u32, cell_count: u16, key: []const u8) !?[]u8 {
    const pos = leafSearchPosition(data, page_size, cell_count, key);
    if (!pos.found) return null;

    const cell = readLeafCell(data, page_size, pos.index);
    const result = try allocator.alloc(u8, cell.value.len);
    @memcpy(result, cell.value);
    return result;
}

/// Insert a cell into a leaf page at the given position.
/// Shifts existing cell pointers right. Does NOT check free space.
fn insertLeafCell(data: []u8, page_size: u32, cell_count: u16, pos: u16, key: []const u8, value: []const u8) !void {
    // Calculate where to write cell data (grow from the end)
    const cell_size = leafCellSize(key, value);
    const cells_bottom = if (cell_count == 0)
        @as(u16, @intCast(page_size))
    else
        lowestCellOffset(data, cell_count, true);
    const cell_off = cells_bottom - cell_size;

    // Write cell data
    var offset: usize = cell_off;
    var vbuf: [varint.max_encoded_len]u8 = undefined;

    var n = varint.encode(key.len, &vbuf) catch return error.PageCorrupt;
    @memcpy(data[offset..][0..n], vbuf[0..n]);
    offset += n;
    @memcpy(data[offset..][0..key.len], key);
    offset += key.len;

    n = varint.encode(value.len, &vbuf) catch return error.PageCorrupt;
    @memcpy(data[offset..][0..n], vbuf[0..n]);
    offset += n;
    @memcpy(data[offset..][0..value.len], value);

    // Shift cell pointers right to make room at pos
    if (pos < cell_count) {
        const src_start = leafCellPtrOffset(pos);
        const src_end = leafCellPtrOffset(cell_count);
        const dst_start = leafCellPtrOffset(pos + 1);
        std.mem.copyBackwards(u8, data[dst_start..][0..(src_end - src_start)], data[src_start..][0..(src_end - src_start)]);
    }

    // Write the new cell pointer
    writeCellPtr(data, leafCellPtrOffset(pos), cell_off);

    // Update cell_count in page header
    std.mem.writeInt(u16, data[2..4], cell_count + 1, .little);
}

/// Delete a cell from a leaf page at the given position.
/// Shifts cell pointers left. Does NOT reclaim cell data space.
fn deleteLeafCell(data: []u8, page_size: u32, cell_count: u16, pos: u16) void {
    _ = page_size;
    // Shift cell pointers left to close the gap (dst < src, may overlap)
    if (pos + 1 < cell_count) {
        const src_start = leafCellPtrOffset(pos + 1);
        const src_end = leafCellPtrOffset(cell_count);
        const dst_start = leafCellPtrOffset(pos);
        const len = src_end - src_start;
        std.mem.copyForwards(u8, data[dst_start..][0..len], data[src_start..][0..len]);
    }

    // Update cell_count in page header
    std.mem.writeInt(u16, data[2..4], cell_count - 1, .little);
}

// ── Leaf Sibling Pointers ──────────────────────────────────────────────

fn getPrevLeaf(data: []const u8) u32 {
    return std.mem.readInt(u32, data[PAGE_HEADER_SIZE..][0..4], .little);
}

fn getNextLeaf(data: []const u8) u32 {
    return std.mem.readInt(u32, data[PAGE_HEADER_SIZE + 4 ..][0..4], .little);
}

fn setPrevLeaf(data: []u8, page_id: u32) void {
    std.mem.writeInt(u32, data[PAGE_HEADER_SIZE..][0..4], page_id, .little);
}

fn setNextLeaf(data: []u8, page_id: u32) void {
    std.mem.writeInt(u32, data[PAGE_HEADER_SIZE + 4 ..][0..4], page_id, .little);
}

// ── Internal Page Operations ───────────────────────────────────────────

fn internalContentStart() u16 {
    return PAGE_HEADER_SIZE + INTERNAL_HEADER_SIZE;
}

fn internalCellPtrOffset(index: u16) u16 {
    return internalContentStart() + index * CELL_PTR_SIZE;
}

fn getRightChild(data: []const u8) u32 {
    return std.mem.readInt(u32, data[PAGE_HEADER_SIZE..][0..4], .little);
}

fn setRightChild(data: []u8, child_id: u32) void {
    std.mem.writeInt(u32, data[PAGE_HEADER_SIZE..][0..4], child_id, .little);
}

const InternalCell = struct {
    left_child: u32,
    key: []const u8,
};

fn readInternalCell(data: []const u8, page_size: u32, index: u16) InternalCell {
    _ = page_size;
    const ptr_off = internalCellPtrOffset(index);
    const cell_off = readCellPtr(data, ptr_off);

    var offset: usize = cell_off;

    // Read left_child (4 bytes LE)
    const left_child = std.mem.readInt(u32, data[offset..][0..4], .little);
    offset += 4;

    // Read key
    const key_dec = varint.decode(data[offset..]) catch return .{ .left_child = left_child, .key = &.{} };
    offset += key_dec.bytes_read;
    const key_len: usize = @intCast(key_dec.value);
    const key_data = data[offset..][0..key_len];

    return .{ .left_child = left_child, .key = key_data };
}

fn internalCellSize(key: []const u8) u16 {
    const key_len_size: u16 = @intCast(varint.encodedLen(key.len));
    return 4 + key_len_size + @as(u16, @intCast(key.len)); // child_id + key_len + key_data
}

fn internalFreeSpace(data: []const u8, page_size: u32, cell_count: u16) u16 {
    const ptrs_end = internalCellPtrOffset(cell_count);
    const cells_start = if (cell_count == 0) @as(u16, @intCast(page_size)) else lowestCellOffset(data, cell_count, false);
    if (cells_start <= ptrs_end) return 0;
    return cells_start - ptrs_end;
}

/// Binary search in internal node. Returns position where key would be inserted.
fn internalSearchPosition(data: []const u8, page_size: u32, cell_count: u16, key: []const u8) u16 {
    if (cell_count == 0) return 0;

    var low: u16 = 0;
    var high: u16 = cell_count;

    while (low < high) {
        const mid = low + (high - low) / 2;
        const cell = readInternalCell(data, page_size, mid);

        switch (std.mem.order(u8, cell.key, key)) {
            .lt => low = mid + 1,
            .gt => high = mid,
            .eq => return mid,
        }
    }

    return low;
}

/// Find which child page to descend into for a given key.
fn findChildInInternal(data: []const u8, page_size: u32, cell_count: u16, key: []const u8) u32 {
    // Binary search: find the first key > search_key
    var low: u16 = 0;
    var high: u16 = cell_count;

    while (low < high) {
        const mid = low + (high - low) / 2;
        const cell = readInternalCell(data, page_size, mid);

        switch (std.mem.order(u8, cell.key, key)) {
            .lt, .eq => low = mid + 1,
            .gt => high = mid,
        }
    }

    // low is now the index of the first key > search_key
    if (low == cell_count) {
        // Key >= all keys → go to right_child
        return getRightChild(data);
    }

    // Go to cell[low].left_child
    const cell = readInternalCell(data, page_size, low);
    return cell.left_child;
}

/// Insert a cell into an internal page at the given position.
fn insertInternalCell(data: []u8, page_size: u32, cell_count: u16, pos: u16, left_child: u32, key: []const u8) !void {
    const cell_size = internalCellSize(key);
    const cells_bottom = if (cell_count == 0)
        @as(u16, @intCast(page_size))
    else
        lowestCellOffset(data, cell_count, false);
    const cell_off = cells_bottom - cell_size;

    // Write cell data: [left_child u32 LE][key_len varint][key_data]
    var offset: usize = cell_off;
    std.mem.writeInt(u32, data[offset..][0..4], left_child, .little);
    offset += 4;

    var vbuf: [varint.max_encoded_len]u8 = undefined;
    const n = varint.encode(key.len, &vbuf) catch return error.PageCorrupt;
    @memcpy(data[offset..][0..n], vbuf[0..n]);
    offset += n;
    @memcpy(data[offset..][0..key.len], key);

    // Shift cell pointers right
    if (pos < cell_count) {
        const src_start = internalCellPtrOffset(pos);
        const src_end = internalCellPtrOffset(cell_count);
        const dst_start = internalCellPtrOffset(pos + 1);
        std.mem.copyBackwards(u8, data[dst_start..][0..(src_end - src_start)], data[src_start..][0..(src_end - src_start)]);
    }

    // Write cell pointer
    writeCellPtr(data, internalCellPtrOffset(pos), cell_off);

    // Update cell_count
    std.mem.writeInt(u16, data[2..4], cell_count + 1, .little);
}

/// Update the child pointer at a given position in an internal node.
/// If pos < cell_count, updates cell[pos].left_child.
/// If pos == cell_count, updates right_child.
fn updateChildPointer(data: []u8, page_size: u32, cell_count: u16, pos: u16, new_child: u32) void {
    if (pos == cell_count) {
        setRightChild(data, new_child);
        return;
    }
    _ = page_size;
    const ptr_off = internalCellPtrOffset(pos);
    const cell_off = readCellPtr(data, ptr_off);
    std.mem.writeInt(u32, data[cell_off..][0..4], new_child, .little);
}

// ── Tests ──────────────────────────────────────────────────────────────

test "leaf page init and basic cell operations" {
    var buf: [4096]u8 = undefined;
    initLeafPage(&buf, 4096, 5);

    const header = PageHeader.deserialize(buf[0..PAGE_HEADER_SIZE]);
    try std.testing.expectEqual(PageType.leaf, header.page_type);
    try std.testing.expectEqual(@as(u32, 5), header.page_id);
    try std.testing.expectEqual(@as(u16, 0), header.cell_count);

    // Insert a key-value pair
    try insertLeafCell(&buf, 4096, 0, 0, "hello", "world");
    const h2 = PageHeader.deserialize(buf[0..PAGE_HEADER_SIZE]);
    try std.testing.expectEqual(@as(u16, 1), h2.cell_count);

    const cell = readLeafCell(&buf, 4096, 0);
    try std.testing.expectEqualStrings("hello", cell.key);
    try std.testing.expectEqualStrings("world", cell.value);
}

test "leaf page multiple inserts in sorted order" {
    var buf: [4096]u8 = undefined;
    initLeafPage(&buf, 4096, 1);

    // Insert keys in sorted order
    try insertLeafCell(&buf, 4096, 0, 0, "apple", "1");
    try insertLeafCell(&buf, 4096, 1, 1, "banana", "2");
    try insertLeafCell(&buf, 4096, 2, 2, "cherry", "3");

    const c0 = readLeafCell(&buf, 4096, 0);
    const c1 = readLeafCell(&buf, 4096, 1);
    const c2 = readLeafCell(&buf, 4096, 2);

    try std.testing.expectEqualStrings("apple", c0.key);
    try std.testing.expectEqualStrings("banana", c1.key);
    try std.testing.expectEqualStrings("cherry", c2.key);
    try std.testing.expectEqualStrings("1", c0.value);
    try std.testing.expectEqualStrings("2", c1.value);
    try std.testing.expectEqualStrings("3", c2.value);
}

test "leaf page insert in middle" {
    var buf: [4096]u8 = undefined;
    initLeafPage(&buf, 4096, 1);

    try insertLeafCell(&buf, 4096, 0, 0, "apple", "1");
    try insertLeafCell(&buf, 4096, 1, 1, "cherry", "3");
    // Insert "banana" at position 1 (between apple and cherry)
    try insertLeafCell(&buf, 4096, 2, 1, "banana", "2");

    const c0 = readLeafCell(&buf, 4096, 0);
    const c1 = readLeafCell(&buf, 4096, 1);
    const c2 = readLeafCell(&buf, 4096, 2);

    try std.testing.expectEqualStrings("apple", c0.key);
    try std.testing.expectEqualStrings("banana", c1.key);
    try std.testing.expectEqualStrings("cherry", c2.key);
}

test "leaf page binary search" {
    var buf: [4096]u8 = undefined;
    initLeafPage(&buf, 4096, 1);

    try insertLeafCell(&buf, 4096, 0, 0, "apple", "1");
    try insertLeafCell(&buf, 4096, 1, 1, "banana", "2");
    try insertLeafCell(&buf, 4096, 2, 2, "cherry", "3");
    try insertLeafCell(&buf, 4096, 3, 3, "date", "4");

    // Exact match
    const r1 = leafSearchPosition(&buf, 4096, 4, "banana");
    try std.testing.expect(r1.found);
    try std.testing.expectEqual(@as(u16, 1), r1.index);

    // Not found — would insert before "banana"
    const r2 = leafSearchPosition(&buf, 4096, 4, "avocado");
    try std.testing.expect(!r2.found);
    try std.testing.expectEqual(@as(u16, 1), r2.index);

    // Not found — would insert at end
    const r3 = leafSearchPosition(&buf, 4096, 4, "elderberry");
    try std.testing.expect(!r3.found);
    try std.testing.expectEqual(@as(u16, 4), r3.index);

    // Not found — would insert at start
    const r4 = leafSearchPosition(&buf, 4096, 4, "aardvark");
    try std.testing.expect(!r4.found);
    try std.testing.expectEqual(@as(u16, 0), r4.index);
}

test "leaf page delete cell" {
    var buf: [4096]u8 = undefined;
    initLeafPage(&buf, 4096, 1);

    try insertLeafCell(&buf, 4096, 0, 0, "apple", "1");
    try insertLeafCell(&buf, 4096, 1, 1, "banana", "2");
    try insertLeafCell(&buf, 4096, 2, 2, "cherry", "3");

    // Delete "banana" at position 1
    deleteLeafCell(&buf, 4096, 3, 1);

    const h = PageHeader.deserialize(buf[0..PAGE_HEADER_SIZE]);
    try std.testing.expectEqual(@as(u16, 2), h.cell_count);

    const c0 = readLeafCell(&buf, 4096, 0);
    const c1 = readLeafCell(&buf, 4096, 1);
    try std.testing.expectEqualStrings("apple", c0.key);
    try std.testing.expectEqualStrings("cherry", c1.key);
}

test "leaf page sibling pointers" {
    var buf: [4096]u8 = undefined;
    initLeafPage(&buf, 4096, 5);

    try std.testing.expectEqual(@as(u32, 0), getPrevLeaf(&buf));
    try std.testing.expectEqual(@as(u32, 0), getNextLeaf(&buf));

    setPrevLeaf(&buf, 4);
    setNextLeaf(&buf, 6);
    try std.testing.expectEqual(@as(u32, 4), getPrevLeaf(&buf));
    try std.testing.expectEqual(@as(u32, 6), getNextLeaf(&buf));
}

test "internal page init and cell operations" {
    var buf: [4096]u8 = undefined;
    initInternalPage(&buf, 4096, 10);

    const header = PageHeader.deserialize(buf[0..PAGE_HEADER_SIZE]);
    try std.testing.expectEqual(PageType.internal, header.page_type);
    try std.testing.expectEqual(@as(u32, 10), header.page_id);
    try std.testing.expectEqual(@as(u16, 0), header.cell_count);

    setRightChild(&buf, 99);
    try std.testing.expectEqual(@as(u32, 99), getRightChild(&buf));

    // Insert a cell: left_child=5, key="middle"
    try insertInternalCell(&buf, 4096, 0, 0, 5, "middle");
    const h2 = PageHeader.deserialize(buf[0..PAGE_HEADER_SIZE]);
    try std.testing.expectEqual(@as(u16, 1), h2.cell_count);

    const cell = readInternalCell(&buf, 4096, 0);
    try std.testing.expectEqual(@as(u32, 5), cell.left_child);
    try std.testing.expectEqualStrings("middle", cell.key);
}

test "internal page child lookup" {
    var buf: [4096]u8 = undefined;
    initInternalPage(&buf, 4096, 10);
    setRightChild(&buf, 30);

    // Internal node: [child=10, key="dog"] [child=20, key="mouse"] right_child=30
    try insertInternalCell(&buf, 4096, 0, 0, 10, "dog");
    try insertInternalCell(&buf, 4096, 1, 1, 20, "mouse");

    // "cat" < "dog" → go to cell[0].left_child = 10
    try std.testing.expectEqual(@as(u32, 10), findChildInInternal(&buf, 4096, 2, "cat"));
    // "dog" <= "dog" → go past cell[0], cell[1].left_child = 20
    try std.testing.expectEqual(@as(u32, 20), findChildInInternal(&buf, 4096, 2, "dog"));
    // "fish" between "dog" and "mouse" → cell[1].left_child = 20
    try std.testing.expectEqual(@as(u32, 20), findChildInInternal(&buf, 4096, 2, "fish"));
    // "mouse" <= "mouse" → go past cell[1], right_child = 30
    try std.testing.expectEqual(@as(u32, 30), findChildInInternal(&buf, 4096, 2, "mouse"));
    // "zebra" > "mouse" → right_child = 30
    try std.testing.expectEqual(@as(u32, 30), findChildInInternal(&buf, 4096, 2, "zebra"));
}

test "leaf free space calculation" {
    var buf: [4096]u8 = undefined;
    initLeafPage(&buf, 4096, 1);

    // Empty leaf: all content area is free
    // Content starts at PAGE_HEADER_SIZE + LEAF_HEADER_SIZE = 16 + 8 = 24
    // Page size = 4096, so free = 4096 - 24 = 4072
    const initial_free = leafFreeSpace(&buf, 4096, 0);
    try std.testing.expectEqual(@as(u16, 4096 - 24), initial_free);

    // Insert a cell and verify free space decreases
    try insertLeafCell(&buf, 4096, 0, 0, "test", "value");
    const after_insert = leafFreeSpace(&buf, 4096, 1);
    // Cell: 1 (key_len varint) + 4 (key) + 1 (val_len varint) + 5 (value) = 11
    // Cell pointer: 2
    // So free should decrease by 11 + 2 = 13
    try std.testing.expectEqual(initial_free - 11 - 2, after_insert);
}

test "BTree insert and get single key" {
    const allocator = std.testing.allocator;
    const path = "test_btree_single.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    // Allocate a root leaf page
    const root_id = try pager.allocPage();
    {
        const raw = try pager.allocPageBuf();
        defer pager.freePageBuf(raw);
        initLeafPage(raw, pager.page_size, root_id);
        try pager.writePage(root_id, raw);
    }

    var pool = try BufferPool.init(allocator, &pager, 100);
    defer pool.deinit();

    var tree = BTree.init(&pool, root_id);

    try tree.insert("hello", "world");
    const val = try tree.get(allocator, "hello");
    try std.testing.expect(val != null);
    try std.testing.expectEqualStrings("world", val.?);
    allocator.free(val.?);

    // Non-existent key
    const missing = try tree.get(allocator, "missing");
    try std.testing.expect(missing == null);
}

test "BTree insert multiple keys and get" {
    const allocator = std.testing.allocator;
    const path = "test_btree_multi.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    const root_id = try pager.allocPage();
    {
        const raw = try pager.allocPageBuf();
        defer pager.freePageBuf(raw);
        initLeafPage(raw, pager.page_size, root_id);
        try pager.writePage(root_id, raw);
    }

    var pool = try BufferPool.init(allocator, &pager, 100);
    defer pool.deinit();

    var tree = BTree.init(&pool, root_id);

    // Insert several keys
    try tree.insert("banana", "yellow");
    try tree.insert("apple", "red");
    try tree.insert("cherry", "red");
    try tree.insert("date", "brown");
    try tree.insert("elderberry", "purple");

    // Verify all keys
    const pairs = [_]struct { k: []const u8, v: []const u8 }{
        .{ .k = "apple", .v = "red" },
        .{ .k = "banana", .v = "yellow" },
        .{ .k = "cherry", .v = "red" },
        .{ .k = "date", .v = "brown" },
        .{ .k = "elderberry", .v = "purple" },
    };

    for (pairs) |pair| {
        const val = try tree.get(allocator, pair.k);
        try std.testing.expect(val != null);
        try std.testing.expectEqualStrings(pair.v, val.?);
        allocator.free(val.?);
    }
}

test "BTree duplicate key rejected" {
    const allocator = std.testing.allocator;
    const path = "test_btree_dup.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    const root_id = try pager.allocPage();
    {
        const raw = try pager.allocPageBuf();
        defer pager.freePageBuf(raw);
        initLeafPage(raw, pager.page_size, root_id);
        try pager.writePage(root_id, raw);
    }

    var pool = try BufferPool.init(allocator, &pager, 100);
    defer pool.deinit();

    var tree = BTree.init(&pool, root_id);

    try tree.insert("key1", "val1");
    const result = tree.insert("key1", "val2");
    try std.testing.expectError(BTreeError.DuplicateKey, result);
}

test "BTree delete key" {
    const allocator = std.testing.allocator;
    const path = "test_btree_delete.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    const root_id = try pager.allocPage();
    {
        const raw = try pager.allocPageBuf();
        defer pager.freePageBuf(raw);
        initLeafPage(raw, pager.page_size, root_id);
        try pager.writePage(root_id, raw);
    }

    var pool = try BufferPool.init(allocator, &pager, 100);
    defer pool.deinit();

    var tree = BTree.init(&pool, root_id);

    try tree.insert("a", "1");
    try tree.insert("b", "2");
    try tree.insert("c", "3");

    // Delete "b"
    try tree.delete("b");

    // "b" should be gone
    const val = try tree.get(allocator, "b");
    try std.testing.expect(val == null);

    // "a" and "c" should still exist
    const a = try tree.get(allocator, "a");
    try std.testing.expect(a != null);
    try std.testing.expectEqualStrings("1", a.?);
    allocator.free(a.?);

    const c = try tree.get(allocator, "c");
    try std.testing.expect(c != null);
    try std.testing.expectEqualStrings("3", c.?);
    allocator.free(c.?);
}

test "BTree delete non-existent key" {
    const allocator = std.testing.allocator;
    const path = "test_btree_del_missing.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    const root_id = try pager.allocPage();
    {
        const raw = try pager.allocPageBuf();
        defer pager.freePageBuf(raw);
        initLeafPage(raw, pager.page_size, root_id);
        try pager.writePage(root_id, raw);
    }

    var pool = try BufferPool.init(allocator, &pager, 100);
    defer pool.deinit();

    var tree = BTree.init(&pool, root_id);

    try tree.insert("exists", "yes");
    const result = tree.delete("missing");
    try std.testing.expectError(BTreeError.KeyNotFound, result);
}

test "BTree leaf split creates valid tree" {
    const allocator = std.testing.allocator;
    const path = "test_btree_split.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    // Use small page size to force splits quickly
    var pager = try Pager.init(allocator, path, .{ .page_size = 512 });
    defer pager.deinit();

    const root_id = try pager.allocPage();
    {
        const raw = try pager.allocPageBuf();
        defer pager.freePageBuf(raw);
        initLeafPage(raw, pager.page_size, root_id);
        try pager.writePage(root_id, raw);
    }

    var pool = try BufferPool.init(allocator, &pager, 100);
    defer pool.deinit();

    var tree = BTree.init(&pool, root_id);

    // Insert enough keys to force a leaf split
    // With 512 byte pages: content = 512 - 16 - 8 = 488 bytes
    // Each cell ≈ 1 + key_len + 1 + val_len + 2 (ptr) bytes
    // With 8-byte keys and 8-byte values: ~20 bytes each → ~24 keys per page
    var key_buf: [8]u8 = undefined;
    var val_buf: [8]u8 = undefined;

    const num_keys: u32 = 40; // should definitely cause at least one split
    for (0..num_keys) |i| {
        const k = std.fmt.bufPrint(&key_buf, "key{d:0>4}", .{i}) catch unreachable;
        const v = std.fmt.bufPrint(&val_buf, "val{d:0>4}", .{i}) catch unreachable;
        try tree.insert(k, v);
    }

    // Verify all keys are still retrievable
    for (0..num_keys) |i| {
        const k = std.fmt.bufPrint(&key_buf, "key{d:0>4}", .{i}) catch unreachable;
        const expected = std.fmt.bufPrint(&val_buf, "val{d:0>4}", .{i}) catch unreachable;
        const val = try tree.get(allocator, k);
        try std.testing.expect(val != null);
        try std.testing.expectEqualStrings(expected, val.?);
        allocator.free(val.?);
    }
}

test "BTree many inserts with splits then deletes" {
    const allocator = std.testing.allocator;
    const path = "test_btree_many.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{ .page_size = 512 });
    defer pager.deinit();

    const root_id = try pager.allocPage();
    {
        const raw = try pager.allocPageBuf();
        defer pager.freePageBuf(raw);
        initLeafPage(raw, pager.page_size, root_id);
        try pager.writePage(root_id, raw);
    }

    var pool = try BufferPool.init(allocator, &pager, 200);
    defer pool.deinit();

    var tree = BTree.init(&pool, root_id);

    var key_buf: [10]u8 = undefined;
    var val_buf: [10]u8 = undefined;

    const N: u32 = 100;

    // Insert N keys
    for (0..N) |i| {
        const k = std.fmt.bufPrint(&key_buf, "k{d:0>6}", .{i}) catch unreachable;
        const v = std.fmt.bufPrint(&val_buf, "v{d:0>6}", .{i}) catch unreachable;
        try tree.insert(k, v);
    }

    // Verify all present
    for (0..N) |i| {
        const k = std.fmt.bufPrint(&key_buf, "k{d:0>6}", .{i}) catch unreachable;
        const val = try tree.get(allocator, k);
        try std.testing.expect(val != null);
        allocator.free(val.?);
    }

    // Delete even-numbered keys
    for (0..N) |i| {
        if (i % 2 == 0) {
            const k = std.fmt.bufPrint(&key_buf, "k{d:0>6}", .{i}) catch unreachable;
            try tree.delete(k);
        }
    }

    // Verify: even keys gone, odd keys present
    for (0..N) |i| {
        const k = std.fmt.bufPrint(&key_buf, "k{d:0>6}", .{i}) catch unreachable;
        const val = try tree.get(allocator, k);
        if (i % 2 == 0) {
            try std.testing.expect(val == null);
        } else {
            try std.testing.expect(val != null);
            allocator.free(val.?);
        }
    }
}

test "BTree internal node split with many keys" {
    const allocator = std.testing.allocator;
    const path = "test_btree_deep.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{ .page_size = 512 });
    defer pager.deinit();

    const root_id = try pager.allocPage();
    {
        const raw = try pager.allocPageBuf();
        defer pager.freePageBuf(raw);
        initLeafPage(raw, pager.page_size, root_id);
        try pager.writePage(root_id, raw);
    }

    var pool = try BufferPool.init(allocator, &pager, 300);
    defer pool.deinit();

    var tree = BTree.init(&pool, root_id);

    var key_buf: [16]u8 = undefined;
    var val_buf: [16]u8 = undefined;

    const N: u32 = 200;

    // Insert 500 keys with 512-byte pages to force multiple levels
    for (0..N) |i| {
        const k = std.fmt.bufPrint(&key_buf, "key{d:0>8}", .{i}) catch unreachable;
        const v = std.fmt.bufPrint(&val_buf, "val{d:0>8}", .{i}) catch unreachable;
        try tree.insert(k, v);
    }

    // Verify all keys retrievable
    for (0..N) |i| {
        const k = std.fmt.bufPrint(&key_buf, "key{d:0>8}", .{i}) catch unreachable;
        const val = try tree.get(allocator, k);
        try std.testing.expect(val != null);
        const expected = std.fmt.bufPrint(&val_buf, "val{d:0>8}", .{i}) catch unreachable;
        try std.testing.expectEqualStrings(expected, val.?);
        allocator.free(val.?);
    }
}

test "BTree leaf sibling chain after splits" {
    const allocator = std.testing.allocator;
    const path = "test_btree_chain.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{ .page_size = 512 });
    defer pager.deinit();

    const root_id = try pager.allocPage();
    {
        const raw = try pager.allocPageBuf();
        defer pager.freePageBuf(raw);
        initLeafPage(raw, pager.page_size, root_id);
        try pager.writePage(root_id, raw);
    }

    var pool = try BufferPool.init(allocator, &pager, 200);
    defer pool.deinit();

    var tree = BTree.init(&pool, root_id);

    var key_buf: [10]u8 = undefined;
    var val_buf: [10]u8 = undefined;

    const N: u32 = 60;

    // Insert 60 keys to force leaf splits
    for (0..N) |i| {
        const k = std.fmt.bufPrint(&key_buf, "k{d:0>6}", .{i}) catch unreachable;
        const v = std.fmt.bufPrint(&val_buf, "v{d:0>6}", .{i}) catch unreachable;
        try tree.insert(k, v);
    }

    // Walk the leaf chain from leftmost to rightmost
    // First, find the leftmost leaf by descending from root
    var page_id = root_id;
    while (true) {
        const frame = try pool.fetchPage(page_id);
        defer pool.unpinPage(page_id, false);

        const header = PageHeader.deserialize(frame.data[0..PAGE_HEADER_SIZE]);
        if (header.page_type == .leaf) {
            break;
        } else if (header.page_type == .internal) {
            // Go to leftmost child
            if (header.cell_count == 0) {
                // Empty internal node shouldn't happen, but handle gracefully
                break;
            }
            const ptr_off = internalCellPtrOffset(0);
            const cell_off = readCellPtr(frame.data, ptr_off);
            page_id = std.mem.readInt(u32, frame.data[cell_off..][0..4], .little);
        } else {
            return error.InvalidNodeType;
        }
    }

    // Now walk the chain and collect all keys
    var collected_keys = std.ArrayList([]const u8){};
    defer {
        for (collected_keys.items) |k| {
            allocator.free(k);
        }
        collected_keys.deinit(allocator);
    }

    var current_page_id = page_id;
    while (current_page_id != 0) {
        const frame = try pool.fetchPage(current_page_id);
        defer pool.unpinPage(current_page_id, false);

        const header = PageHeader.deserialize(frame.data[0..PAGE_HEADER_SIZE]);
        try std.testing.expect(header.page_type == .leaf);

        // Collect keys from this leaf
        for (0..header.cell_count) |i| {
            const cell = readLeafCell(frame.data, pager.page_size, @intCast(i));
            const key_copy = try allocator.alloc(u8, cell.key.len);
            @memcpy(key_copy, cell.key);
            try collected_keys.append(allocator, key_copy);
        }

        // Move to next leaf
        current_page_id = getNextLeaf(frame.data);
    }

    // Verify all keys present and sorted
    try std.testing.expectEqual(N, collected_keys.items.len);
    for (0..N) |i| {
        const expected = std.fmt.bufPrint(&key_buf, "k{d:0>6}", .{i}) catch unreachable;
        try std.testing.expectEqualStrings(expected, collected_keys.items[i]);
    }
}

test "BTree insert and delete all keys" {
    const allocator = std.testing.allocator;
    const path = "test_btree_delete_all.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{ .page_size = 512 });
    defer pager.deinit();

    const root_id = try pager.allocPage();
    {
        const raw = try pager.allocPageBuf();
        defer pager.freePageBuf(raw);
        initLeafPage(raw, pager.page_size, root_id);
        try pager.writePage(root_id, raw);
    }

    var pool = try BufferPool.init(allocator, &pager, 100);
    defer pool.deinit();

    var tree = BTree.init(&pool, root_id);

    var key_buf: [10]u8 = undefined;
    var val_buf: [10]u8 = undefined;

    const N: u32 = 50;

    // Insert 50 keys
    for (0..N) |i| {
        const k = std.fmt.bufPrint(&key_buf, "k{d:0>5}", .{i}) catch unreachable;
        const v = std.fmt.bufPrint(&val_buf, "v{d:0>5}", .{i}) catch unreachable;
        try tree.insert(k, v);
    }

    // Delete all keys
    for (0..N) |i| {
        const k = std.fmt.bufPrint(&key_buf, "k{d:0>5}", .{i}) catch unreachable;
        try tree.delete(k);
    }

    // Verify all keys are gone
    for (0..N) |i| {
        const k = std.fmt.bufPrint(&key_buf, "k{d:0>5}", .{i}) catch unreachable;
        const val = try tree.get(allocator, k);
        try std.testing.expect(val == null);
    }
}

test "BTree reverse order inserts" {
    const allocator = std.testing.allocator;
    const path = "test_btree_reverse.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{ .page_size = 512 });
    defer pager.deinit();

    const root_id = try pager.allocPage();
    {
        const raw = try pager.allocPageBuf();
        defer pager.freePageBuf(raw);
        initLeafPage(raw, pager.page_size, root_id);
        try pager.writePage(root_id, raw);
    }

    var pool = try BufferPool.init(allocator, &pager, 200);
    defer pool.deinit();

    var tree = BTree.init(&pool, root_id);

    var key_buf: [10]u8 = undefined;
    var val_buf: [10]u8 = undefined;

    const N: u32 = 80;

    // Insert in reverse order
    var i: u32 = N;
    while (i > 0) {
        i -= 1;
        const k = std.fmt.bufPrint(&key_buf, "key{d:0>4}", .{i}) catch unreachable;
        const v = std.fmt.bufPrint(&val_buf, "val{d:0>4}", .{i}) catch unreachable;
        try tree.insert(k, v);
    }

    // Verify all keys retrievable
    for (0..N) |idx| {
        const k = std.fmt.bufPrint(&key_buf, "key{d:0>4}", .{idx}) catch unreachable;
        const val = try tree.get(allocator, k);
        try std.testing.expect(val != null);
        const expected = std.fmt.bufPrint(&val_buf, "val{d:0>4}", .{idx}) catch unreachable;
        try std.testing.expectEqualStrings(expected, val.?);
        allocator.free(val.?);
    }
}

test "BTree empty key and value" {
    const allocator = std.testing.allocator;
    const path = "test_btree_empty.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    const root_id = try pager.allocPage();
    {
        const raw = try pager.allocPageBuf();
        defer pager.freePageBuf(raw);
        initLeafPage(raw, pager.page_size, root_id);
        try pager.writePage(root_id, raw);
    }

    var pool = try BufferPool.init(allocator, &pager, 100);
    defer pool.deinit();

    var tree = BTree.init(&pool, root_id);

    // Insert empty key and empty value
    try tree.insert("", "");

    // Insert key "a" with empty value
    try tree.insert("a", "");

    // Verify both retrievable
    const val1 = try tree.get(allocator, "");
    try std.testing.expect(val1 != null);
    try std.testing.expectEqualStrings("", val1.?);
    allocator.free(val1.?);

    const val2 = try tree.get(allocator, "a");
    try std.testing.expect(val2 != null);
    try std.testing.expectEqualStrings("", val2.?);
    allocator.free(val2.?);
}

test "BTree single byte keys" {
    const allocator = std.testing.allocator;
    const path = "test_btree_single_byte.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{ .page_size = 512 });
    defer pager.deinit();

    const root_id = try pager.allocPage();
    {
        const raw = try pager.allocPageBuf();
        defer pager.freePageBuf(raw);
        initLeafPage(raw, pager.page_size, root_id);
        try pager.writePage(root_id, raw);
    }

    var pool = try BufferPool.init(allocator, &pager, 100);
    defer pool.deinit();

    var tree = BTree.init(&pool, root_id);

    // Insert all 26 lowercase letters
    var i: u8 = 0;
    while (i < 26) : (i += 1) {
        const key = [_]u8{'a' + i};
        const value = [_]u8{'A' + i};
        try tree.insert(&key, &value);
    }

    // Verify all retrievable
    i = 0;
    while (i < 26) : (i += 1) {
        const key = [_]u8{'a' + i};
        const val = try tree.get(allocator, &key);
        try std.testing.expect(val != null);
        try std.testing.expectEqual(@as(usize, 1), val.?.len);
        try std.testing.expectEqual('A' + i, val.?[0]);
        allocator.free(val.?);
    }
}

test "BTree delete first and last keys" {
    const allocator = std.testing.allocator;
    const path = "test_btree_delete_edges.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    const root_id = try pager.allocPage();
    {
        const raw = try pager.allocPageBuf();
        defer pager.freePageBuf(raw);
        initLeafPage(raw, pager.page_size, root_id);
        try pager.writePage(root_id, raw);
    }

    var pool = try BufferPool.init(allocator, &pager, 100);
    defer pool.deinit();

    var tree = BTree.init(&pool, root_id);

    // Insert keys a through z
    var i: u8 = 0;
    while (i < 26) : (i += 1) {
        const key = [_]u8{'a' + i};
        const value = [_]u8{'A' + i};
        try tree.insert(&key, &value);
    }

    // Delete first (a) and last (z)
    try tree.delete("a");
    try tree.delete("z");

    // Verify a and z are gone
    const val_a = try tree.get(allocator, "a");
    try std.testing.expect(val_a == null);

    const val_z = try tree.get(allocator, "z");
    try std.testing.expect(val_z == null);

    // Verify rest are present
    i = 1;
    while (i < 25) : (i += 1) {
        const key = [_]u8{'a' + i};
        const val = try tree.get(allocator, &key);
        try std.testing.expect(val != null);
        try std.testing.expectEqual('A' + i, val.?[0]);
        allocator.free(val.?);
    }
}

test "BTree insert after deletes reuses correct positions" {
    const allocator = std.testing.allocator;
    const path = "test_btree_reuse_positions.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    const root_id = try pager.allocPage();
    {
        const raw = try pager.allocPageBuf();
        defer pager.freePageBuf(raw);
        initLeafPage(raw, pager.page_size, root_id);
        try pager.writePage(root_id, raw);
    }

    var pool = try BufferPool.init(allocator, &pager, 100);
    defer pool.deinit();

    var tree = BTree.init(&pool, root_id);

    // Insert a, b, c, d
    try tree.insert("a", "value_a");
    try tree.insert("b", "value_b");
    try tree.insert("c", "value_c");
    try tree.insert("d", "value_d");

    // Delete b and c
    try tree.delete("b");
    try tree.delete("c");

    // Insert bb and cc (which should go in their sorted positions)
    try tree.insert("bb", "value_bb");
    try tree.insert("cc", "value_cc");

    // Verify all 4 keys present with correct values
    const val_a = try tree.get(allocator, "a");
    try std.testing.expect(val_a != null);
    try std.testing.expectEqualStrings("value_a", val_a.?);
    allocator.free(val_a.?);

    const val_bb = try tree.get(allocator, "bb");
    try std.testing.expect(val_bb != null);
    try std.testing.expectEqualStrings("value_bb", val_bb.?);
    allocator.free(val_bb.?);

    const val_cc = try tree.get(allocator, "cc");
    try std.testing.expect(val_cc != null);
    try std.testing.expectEqualStrings("value_cc", val_cc.?);
    allocator.free(val_cc.?);

    const val_d = try tree.get(allocator, "d");
    try std.testing.expect(val_d != null);
    try std.testing.expectEqualStrings("value_d", val_d.?);
    allocator.free(val_d.?);

    // Verify deleted keys are gone
    const val_b = try tree.get(allocator, "b");
    try std.testing.expect(val_b == null);

    const val_c = try tree.get(allocator, "c");
    try std.testing.expect(val_c == null);
}
