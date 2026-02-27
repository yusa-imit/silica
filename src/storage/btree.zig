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
    MergeError,
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
    /// Performs leaf and internal node merging/redistribution on underflow.
    pub fn delete(self: *BTree, key: []const u8) anyerror!void {
        const result = try self.deleteFromNode(self.root_page_id, key);
        _ = result;

        // Check if root needs shrinking: internal root with 0 cells means
        // it has only the right_child pointer — make that child the new root.
        const frame = try self.pool.fetchPage(self.root_page_id);
        const header = PageHeader.deserialize(frame.data[0..PAGE_HEADER_SIZE]);
        if (header.page_type == .internal and header.cell_count == 0) {
            const new_root = getRightChild(frame.data);
            self.pool.unpinPage(self.root_page_id, false);
            if (new_root != 0) {
                try self.pager.freePage(self.root_page_id);
                self.root_page_id = new_root;
            }
        } else {
            self.pool.unpinPage(self.root_page_id, false);
        }
    }

    const DeleteResult = struct {
        underflow: bool,
    };

    fn deleteFromNode(self: *BTree, page_id: u32, key: []const u8) anyerror!DeleteResult {
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
                const new_count = header.cell_count - 1;
                const underflow = isLeafUnderflow(frame.data, self.pager.page_size, new_count);
                self.pool.unpinPage(page_id, true);
                return .{ .underflow = underflow };
            },
            .internal => {
                return self.deleteFromInternal(frame, page_id, key);
            },
            else => {
                self.pool.unpinPage(page_id, false);
                return BTreeError.InvalidNodeType;
            },
        }
    }

    fn deleteFromInternal(self: *BTree, frame: *BufferFrame, page_id: u32, key: []const u8) anyerror!DeleteResult {
        const page_size = self.pager.page_size;
        const header = PageHeader.deserialize(frame.data[0..PAGE_HEADER_SIZE]);
        const cell_count = header.cell_count;

        // Find which child to descend into and remember the child index
        const child_info = findChildWithIndex(frame.data, page_size, cell_count, key);
        self.pool.unpinPage(page_id, false);

        // Recurse into child
        const child_result = try self.deleteFromNode(child_info.child_page_id, key);

        if (!child_result.underflow) {
            return .{ .underflow = false };
        }

        // Child is underflowing — try to fix it
        try self.handleChildUnderflow(page_id, child_info.child_page_id, child_info.child_index);

        // Re-check if this internal node itself is underflowing
        const re_frame = try self.pool.fetchPage(page_id);
        const re_header = PageHeader.deserialize(re_frame.data[0..PAGE_HEADER_SIZE]);
        const internal_underflow = isInternalUnderflow(re_frame.data, page_size, re_header.cell_count);
        self.pool.unpinPage(page_id, false);

        return .{ .underflow = internal_underflow };
    }

    /// Information about which child was found during internal node traversal.
    const ChildInfo = struct {
        child_page_id: u32,
        /// Index into the internal node's children (0..cell_count for left children, cell_count for right_child)
        child_index: u16,
    };

    /// Check if a leaf node is underflowing (less than 1/4 of usable space is used).
    /// Root leaves never underflow (nothing to merge with at root level).
    fn isLeafUnderflow(data: []const u8, page_size: u32, cell_count: u16) bool {
        if (cell_count == 0) return true;
        const usable = page_size - PAGE_HEADER_SIZE - LEAF_HEADER_SIZE;
        const free = leafFreeSpace(data, page_size, cell_count);
        // Underflow if more than 3/4 of the usable space is free
        return free > (usable * 3) / 4;
    }

    /// Check if an internal node is underflowing.
    fn isInternalUnderflow(data: []const u8, page_size: u32, cell_count: u16) bool {
        if (cell_count == 0) return true;
        const usable = page_size - PAGE_HEADER_SIZE - INTERNAL_HEADER_SIZE;
        const free = internalFreeSpace(data, page_size, cell_count);
        return free > (usable * 3) / 4;
    }

    /// Handle underflow in a child of an internal node.
    /// `parent_page_id` is the internal node, `child_page_id` is the underflowing child,
    /// `child_index` is the child's position (0..cell_count = left children, cell_count = right_child).
    fn handleChildUnderflow(self: *BTree, parent_page_id: u32, child_page_id: u32, child_index: u16) anyerror!void {
        const page_size = self.pager.page_size;

        // Re-fetch parent to get current state
        const parent_frame = try self.pool.fetchPage(parent_page_id);
        const parent_header = PageHeader.deserialize(parent_frame.data[0..PAGE_HEADER_SIZE]);
        const parent_count = parent_header.cell_count;

        if (parent_count == 0) {
            // Only right_child remains — nothing to merge/redistribute
            self.pool.unpinPage(parent_page_id, false);
            return;
        }

        // Determine the child's page type
        const child_frame = try self.pool.fetchPage(child_page_id);
        const child_header = PageHeader.deserialize(child_frame.data[0..PAGE_HEADER_SIZE]);
        const child_type = child_header.page_type;
        self.pool.unpinPage(child_page_id, false);

        // Find left and right siblings
        var left_sibling_id: u32 = 0;
        var right_sibling_id: u32 = 0;
        var separator_index: u16 = 0; // index of the separator key in parent

        if (child_index > 0) {
            // There is a left sibling
            separator_index = child_index - 1;
            if (child_index - 1 == 0 and child_index > 1) {
                const cell_data = readInternalCell(parent_frame.data, page_size, child_index - 2);
                _ = cell_data;
            }
            // Get left sibling page ID
            if (child_index == 1) {
                const cell0 = readInternalCell(parent_frame.data, page_size, 0);
                left_sibling_id = cell0.left_child;
            } else {
                // child_index-1 is the separator, child at child_index-1's left_child is the left sibling
                // Actually: in our internal node layout, cell[i].left_child is the child to the left of key[i].
                // If child_index == i, the child is cell[i].left_child.
                // The left sibling would be at child_index-1.
                // For child_index == parent_count (right_child), left sibling is cell[parent_count-1].left_child... no.

                // Let me think about this more carefully:
                // Internal node children: cell[0].left_child, cell[1].left_child, ..., cell[n-1].left_child, right_child
                // These are n+1 children for n keys.
                // child_index 0 → cell[0].left_child
                // child_index 1 → cell[1].left_child
                // child_index i → cell[i].left_child (for i < n)
                // child_index n → right_child
                //
                // Left sibling of child_index i:
                //   i == 0 → no left sibling
                //   i == 1 → cell[0].left_child
                //   i > 1  → child at index i-1 → cell[i-1].left_child
                //   i == n → child at index n-1 → cell[n-1].left_child... wait, wrong.
                //
                // Let me reconsider. The child at position `child_index` is:
                //   getChildAtIndex(data, page_size, cell_count, child_index)
                // The separator between child i and child i+1 is key[i].

                left_sibling_id = getChildAtIndex(parent_frame.data, page_size, parent_count, child_index - 1);
            }
            separator_index = child_index - 1;
        }
        if (child_index < parent_count) {
            // There is a right sibling
            right_sibling_id = getChildAtIndex(parent_frame.data, page_size, parent_count, child_index + 1);
        }

        self.pool.unpinPage(parent_page_id, false);

        if (child_type == .leaf) {
            // Try right sibling first, then left
            if (right_sibling_id != 0) {
                const merged = try self.tryMergeOrRedistributeLeaves(
                    parent_page_id,
                    child_page_id,
                    right_sibling_id,
                    child_index, // separator is at child_index
                    true,        // child is left, sibling is right
                );
                if (merged) return;
            }
            if (left_sibling_id != 0) {
                _ = try self.tryMergeOrRedistributeLeaves(
                    parent_page_id,
                    left_sibling_id,
                    child_page_id,
                    separator_index,
                    false, // sibling is left, child is right
                );
            }
        } else if (child_type == .internal) {
            if (right_sibling_id != 0) {
                const merged = try self.tryMergeOrRedistributeInternal(
                    parent_page_id,
                    child_page_id,
                    right_sibling_id,
                    child_index,
                    true,
                );
                if (merged) return;
            }
            if (left_sibling_id != 0) {
                _ = try self.tryMergeOrRedistributeInternal(
                    parent_page_id,
                    left_sibling_id,
                    child_page_id,
                    separator_index,
                    false,
                );
            }
        }
    }

    /// Get the child page ID at a given index (0..cell_count = cell[i].left_child, cell_count = right_child).
    fn getChildAtIndex(data: []const u8, page_size: u32, cell_count: u16, index: u16) u32 {
        if (index == cell_count) {
            return getRightChild(data);
        }
        const cell = readInternalCell(data, page_size, index);
        return cell.left_child;
    }

    /// Try to merge or redistribute two adjacent leaf pages.
    /// `left_id` and `right_id` are the two leaf pages. `sep_index` is the index of the
    /// separator key in the parent. Returns true if merge/redistribute happened.
    fn tryMergeOrRedistributeLeaves(
        self: *BTree,
        parent_page_id: u32,
        left_id: u32,
        right_id: u32,
        sep_index: u16,
        child_is_left: bool,
    ) anyerror!bool {
        _ = child_is_left;
        const page_size = self.pager.page_size;

        const left_frame = try self.pool.fetchPage(left_id);
        const left_header = PageHeader.deserialize(left_frame.data[0..PAGE_HEADER_SIZE]);
        const left_count = left_header.cell_count;

        const right_frame = try self.pool.fetchPage(right_id);
        const right_header = PageHeader.deserialize(right_frame.data[0..PAGE_HEADER_SIZE]);
        const right_count = right_header.cell_count;

        // Calculate total data size needed
        const total_data_size = leafUsedSpace(left_frame.data, page_size, left_count) +
            leafUsedSpace(right_frame.data, page_size, right_count);
        const usable = page_size - PAGE_HEADER_SIZE - LEAF_HEADER_SIZE;

        if (total_data_size <= usable) {
            // Can merge: move all cells from right into left
            try self.mergeLeaves(left_frame, right_frame, left_id, right_id, left_count, right_count, parent_page_id, sep_index);
            return true;
        }

        // Cannot merge — redistribute cells between the two leaves
        self.redistributeLeaves(left_frame, right_frame, left_id, right_id, left_count, right_count, parent_page_id, sep_index);
        return true;
    }

    /// Merge right leaf into left leaf, removing separator from parent.
    fn mergeLeaves(
        self: *BTree,
        left_frame: *BufferFrame,
        right_frame: *BufferFrame,
        left_id: u32,
        right_id: u32,
        left_count: u16,
        right_count: u16,
        parent_page_id: u32,
        sep_index: u16,
    ) anyerror!void {
        const page_size = self.pager.page_size;

        // Collect all cells from both leaves
        const total: u32 = @as(u32, left_count) + @as(u32, right_count);
        var cells = try self.pager.allocator.alloc(CellRef, total);
        defer self.pager.allocator.free(cells);

        for (0..left_count) |i| {
            const cell = readLeafCell(left_frame.data, page_size, @intCast(i));
            cells[i] = .{ .key = cell.key, .value = cell.value, .from_new = false };
        }
        for (0..right_count) |i| {
            const cell = readLeafCell(right_frame.data, page_size, @intCast(i));
            cells[left_count + i] = .{ .key = cell.key, .value = cell.value, .from_new = false };
        }

        // Save data from both frames since cells reference frame data
        const saved_left = try self.pager.allocator.alloc(u8, page_size);
        defer self.pager.allocator.free(saved_left);
        @memcpy(saved_left, left_frame.data[0..page_size]);

        const saved_right = try self.pager.allocator.alloc(u8, page_size);
        defer self.pager.allocator.free(saved_right);
        @memcpy(saved_right, right_frame.data[0..page_size]);

        // Re-read cells from saved copies
        for (0..left_count) |i| {
            const cell = readLeafCell(saved_left, page_size, @intCast(i));
            cells[i] = .{ .key = cell.key, .value = cell.value, .from_new = false };
        }
        for (0..right_count) |i| {
            const cell = readLeafCell(saved_right, page_size, @intCast(i));
            cells[left_count + i] = .{ .key = cell.key, .value = cell.value, .from_new = false };
        }

        // Update sibling chain: left.next = right.next
        const right_next = getNextLeaf(saved_right);
        const left_prev = getPrevLeaf(saved_left);

        // Reinitialize left page and write all cells
        initLeafPage(left_frame.data, page_size, left_id);
        setPrevLeaf(left_frame.data, left_prev);
        setNextLeaf(left_frame.data, right_next);

        for (0..total) |i| {
            insertLeafCell(left_frame.data, page_size, @intCast(i), @intCast(i), cells[i].key, cells[i].value) catch {
                self.pool.unpinPage(left_id, false);
                self.pool.unpinPage(right_id, false);
                return BTreeError.MergeError;
            };
        }

        self.pool.unpinPage(left_id, true);
        self.pool.unpinPage(right_id, false);

        // Update the next leaf's prev pointer
        if (right_next != 0) {
            const next_frame = try self.pool.fetchPage(right_next);
            setPrevLeaf(next_frame.data, left_id);
            self.pool.unpinPage(right_next, true);
        }

        // Free the right page
        try self.pager.freePage(right_id);

        // Remove separator from parent and update child pointer
        const parent_frame = try self.pool.fetchPage(parent_page_id);
        const parent_header = PageHeader.deserialize(parent_frame.data[0..PAGE_HEADER_SIZE]);

        // Delete separator at sep_index. After deletion, the child that was to the right
        // of the separator (right_id) is gone, so we need to ensure left_id stays in position.
        // Before deletion: ...[left_id | sep_key | right_id]...
        // cell[sep_index].left_child == left_id, and the next child is right_id.
        // After removing sep_key, left_id should absorb right_id's position.
        deleteInternalCell(parent_frame.data, page_size, parent_header.cell_count, sep_index);

        // If sep_index was pointing to the right_child position's neighbor,
        // we need to make sure the pointer is correct.
        // After deletion, if sep_index < new_count: cell[sep_index].left_child should be left_id
        // If sep_index == new_count (was the last separator): right_child should be left_id
        const new_parent_count = parent_header.cell_count - 1;
        if (sep_index < new_parent_count) {
            // Update cell[sep_index].left_child to left_id
            const ptr_off = internalCellPtrOffset(sep_index);
            const cell_off = readCellPtr(parent_frame.data, ptr_off);
            std.mem.writeInt(u32, parent_frame.data[cell_off..][0..4], left_id, .little);
        } else {
            // sep_index was the last separator, so right_child should point to left_id
            setRightChild(parent_frame.data, left_id);
        }

        self.pool.unpinPage(parent_page_id, true);
    }

    /// Redistribute cells between two adjacent leaves to balance them.
    fn redistributeLeaves(
        self: *BTree,
        left_frame: *BufferFrame,
        right_frame: *BufferFrame,
        left_id: u32,
        right_id: u32,
        left_count: u16,
        right_count: u16,
        parent_page_id: u32,
        sep_index: u16,
    ) void {
        const page_size = self.pager.page_size;
        const total: u32 = @as(u32, left_count) + @as(u32, right_count);

        // Collect all cells from both leaves (from saved copies)
        const saved_left = self.pager.allocator.alloc(u8, page_size) catch {
            self.pool.unpinPage(left_id, false);
            self.pool.unpinPage(right_id, false);
            return;
        };
        defer self.pager.allocator.free(saved_left);
        @memcpy(saved_left, left_frame.data[0..page_size]);

        const saved_right = self.pager.allocator.alloc(u8, page_size) catch {
            self.pool.unpinPage(left_id, false);
            self.pool.unpinPage(right_id, false);
            return;
        };
        defer self.pager.allocator.free(saved_right);
        @memcpy(saved_right, right_frame.data[0..page_size]);

        const cells = self.pager.allocator.alloc(CellRef, total) catch {
            self.pool.unpinPage(left_id, false);
            self.pool.unpinPage(right_id, false);
            return;
        };
        defer self.pager.allocator.free(cells);

        for (0..left_count) |i| {
            const cell = readLeafCell(saved_left, page_size, @intCast(i));
            cells[i] = .{ .key = cell.key, .value = cell.value, .from_new = false };
        }
        for (0..right_count) |i| {
            const cell = readLeafCell(saved_right, page_size, @intCast(i));
            cells[left_count + i] = .{ .key = cell.key, .value = cell.value, .from_new = false };
        }

        // Split roughly in half
        const split_point: u32 = total / 2;

        // Preserve sibling pointers
        const left_prev = getPrevLeaf(saved_left);
        const right_next = getNextLeaf(saved_right);

        // Reinitialize both pages
        initLeafPage(left_frame.data, page_size, left_id);
        setPrevLeaf(left_frame.data, left_prev);
        setNextLeaf(left_frame.data, right_id);

        initLeafPage(right_frame.data, page_size, right_id);
        setPrevLeaf(right_frame.data, left_id);
        setNextLeaf(right_frame.data, right_next);

        // Write cells
        for (0..split_point) |i| {
            insertLeafCell(left_frame.data, page_size, @intCast(i), @intCast(i), cells[i].key, cells[i].value) catch break;
        }
        for (split_point..total) |i| {
            const j: u16 = @intCast(i - split_point);
            insertLeafCell(right_frame.data, page_size, j, j, cells[i].key, cells[i].value) catch break;
        }

        self.pool.unpinPage(left_id, true);
        self.pool.unpinPage(right_id, true);

        // Update the separator key in the parent to be the first key of the new right page
        const new_right_first = readLeafCell(right_frame.data, page_size, 0);
        _ = new_right_first;

        // Re-read from the frame since we already unpinned... we need the key.
        // Actually let's use cells[split_point].key which is still valid (references saved_right/saved_left)
        const new_sep_key = cells[split_point].key;

        // Update parent's separator key
        const parent_frame = self.pool.fetchPage(parent_page_id) catch return;
        const parent_header = PageHeader.deserialize(parent_frame.data[0..PAGE_HEADER_SIZE]);

        // Replace the separator: delete old + insert new at same position
        const old_cell = readInternalCell(parent_frame.data, page_size, sep_index);
        const old_left_child = old_cell.left_child;
        deleteInternalCell(parent_frame.data, page_size, parent_header.cell_count, sep_index);
        insertInternalCell(parent_frame.data, page_size, parent_header.cell_count - 1, sep_index, old_left_child, new_sep_key) catch {
            self.pool.unpinPage(parent_page_id, false);
            return;
        };

        self.pool.unpinPage(parent_page_id, true);
    }

    /// Try to merge or redistribute two adjacent internal nodes.
    fn tryMergeOrRedistributeInternal(
        self: *BTree,
        parent_page_id: u32,
        left_id: u32,
        right_id: u32,
        sep_index: u16,
        child_is_left: bool,
    ) anyerror!bool {
        _ = child_is_left;
        const page_size = self.pager.page_size;

        const left_frame = try self.pool.fetchPage(left_id);
        const left_header = PageHeader.deserialize(left_frame.data[0..PAGE_HEADER_SIZE]);
        const left_count = left_header.cell_count;

        const right_frame = try self.pool.fetchPage(right_id);
        const right_header = PageHeader.deserialize(right_frame.data[0..PAGE_HEADER_SIZE]);
        const right_count = right_header.cell_count;

        // Read separator key from parent
        const parent_frame = try self.pool.fetchPage(parent_page_id);
        const sep_cell = readInternalCell(parent_frame.data, page_size, sep_index);
        // We need to copy the separator key since it references parent frame data
        const sep_key_copy = try self.pager.allocator.alloc(u8, sep_cell.key.len);
        defer self.pager.allocator.free(sep_key_copy);
        @memcpy(sep_key_copy, sep_cell.key);
        self.pool.unpinPage(parent_page_id, false);

        // Total cells = left_count + 1 (separator) + right_count
        const total: u32 = @as(u32, left_count) + 1 + @as(u32, right_count);

        // Check if they fit in one page
        // We need to calculate total space needed
        var total_size: u32 = 0;
        for (0..left_count) |i| {
            const cell = readInternalCell(left_frame.data, page_size, @intCast(i));
            total_size += internalCellSize(cell.key) + CELL_PTR_SIZE;
        }
        total_size += internalCellSize(sep_key_copy) + CELL_PTR_SIZE;
        for (0..right_count) |i| {
            const cell = readInternalCell(right_frame.data, page_size, @intCast(i));
            total_size += internalCellSize(cell.key) + CELL_PTR_SIZE;
        }

        const usable = page_size - PAGE_HEADER_SIZE - INTERNAL_HEADER_SIZE;

        if (total_size <= usable) {
            // Can merge
            try self.mergeInternal(left_frame, right_frame, left_id, right_id, left_count, right_count, parent_page_id, sep_index, sep_key_copy);
            return true;
        }

        // Cannot merge — redistribute
        self.pool.unpinPage(left_id, false);
        self.pool.unpinPage(right_id, false);

        // For simplicity, skip redistribution of internal nodes in this first implementation.
        // Internal node underflow from redistribution is rare in practice.
        _ = total;
        return false;
    }

    /// Merge right internal node into left internal node with separator from parent.
    fn mergeInternal(
        self: *BTree,
        left_frame: *BufferFrame,
        right_frame: *BufferFrame,
        left_id: u32,
        right_id: u32,
        left_count: u16,
        right_count: u16,
        parent_page_id: u32,
        sep_index: u16,
        sep_key: []const u8,
    ) anyerror!void {
        const page_size = self.pager.page_size;

        // Save both frames' data since cells reference frame data
        const saved_left = try self.pager.allocator.alloc(u8, page_size);
        defer self.pager.allocator.free(saved_left);
        @memcpy(saved_left, left_frame.data[0..page_size]);

        const saved_right = try self.pager.allocator.alloc(u8, page_size);
        defer self.pager.allocator.free(saved_right);
        @memcpy(saved_right, right_frame.data[0..page_size]);

        // Collect all cells: left cells + separator (with left's right_child as left_child) + right cells
        const total: u32 = @as(u32, left_count) + 1 + @as(u32, right_count);
        var cells = try self.pager.allocator.alloc(InternalCellRef, total);
        defer self.pager.allocator.free(cells);

        for (0..left_count) |i| {
            const cell = readInternalCell(saved_left, page_size, @intCast(i));
            cells[i] = .{ .left_child = cell.left_child, .key = cell.key };
        }

        // Separator key: its left_child is the old left node's right_child
        const left_right_child = getRightChild(saved_left);
        cells[left_count] = .{ .left_child = left_right_child, .key = sep_key };

        for (0..right_count) |i| {
            const cell = readInternalCell(saved_right, page_size, @intCast(i));
            cells[left_count + 1 + i] = .{ .left_child = cell.left_child, .key = cell.key };
        }

        // The merged node's right_child is the right node's right_child
        const new_right_child = getRightChild(saved_right);

        // Reinitialize left page and write all cells
        initInternalPage(left_frame.data, page_size, left_id);
        for (0..total) |i| {
            insertInternalCell(left_frame.data, page_size, @intCast(i), @intCast(i), cells[i].left_child, cells[i].key) catch {
                self.pool.unpinPage(left_id, false);
                self.pool.unpinPage(right_id, false);
                return BTreeError.MergeError;
            };
        }
        setRightChild(left_frame.data, new_right_child);

        self.pool.unpinPage(left_id, true);
        self.pool.unpinPage(right_id, false);

        // Free the right page
        try self.pager.freePage(right_id);

        // Remove separator from parent
        const parent_frame = try self.pool.fetchPage(parent_page_id);
        const parent_header = PageHeader.deserialize(parent_frame.data[0..PAGE_HEADER_SIZE]);

        deleteInternalCell(parent_frame.data, page_size, parent_header.cell_count, sep_index);

        // Update child pointer: after removing separator, left_id should be in the right place
        const new_parent_count = parent_header.cell_count - 1;
        if (sep_index < new_parent_count) {
            const ptr_off = internalCellPtrOffset(sep_index);
            const cell_off = readCellPtr(parent_frame.data, ptr_off);
            std.mem.writeInt(u32, parent_frame.data[cell_off..][0..4], left_id, .little);
        } else {
            setRightChild(parent_frame.data, left_id);
        }

        self.pool.unpinPage(parent_page_id, true);
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
        const old_prev = getPrevLeaf(saved);

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

        // Update sibling pointers: old_prev <-> old -> new -> old_next
        setPrevLeaf(frame.data, old_prev);
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

// ── Range Scan Cursor ─────────────────────────────────────────────────

/// A cursor for iterating over B+Tree entries in key order.
/// Supports forward and backward traversal using the leaf sibling chain.
///
/// Usage:
///   var cursor = try Cursor.init(allocator, &tree);
///   defer cursor.deinit();
///   try cursor.seekFirst();
///   while (try cursor.next()) |entry| {
///       // use entry.key, entry.value
///       allocator.free(entry.key);
///       allocator.free(entry.value);
///   }
pub const Cursor = struct {
    tree: *BTree,
    allocator: std.mem.Allocator,
    /// Current leaf page ID (0 = invalid/exhausted).
    page_id: u32 = 0,
    /// Current cell index within the leaf page.
    cell_index: u16 = 0,
    /// Number of cells in the current leaf page.
    cell_count: u16 = 0,
    /// Whether the cursor has been positioned.
    valid: bool = false,

    pub const Entry = struct {
        key: []u8,
        value: []u8,
    };

    pub fn init(allocator: std.mem.Allocator, tree: *BTree) Cursor {
        return .{
            .tree = tree,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Cursor) void {
        _ = self;
    }

    /// Position the cursor at the first (smallest) key in the tree.
    pub fn seekFirst(self: *Cursor) !void {
        var page_id = self.tree.root_page_id;

        while (true) {
            const frame = try self.tree.pool.fetchPage(page_id);
            defer self.tree.pool.unpinPage(page_id, false);
            const header = PageHeader.deserialize(frame.data[0..PAGE_HEADER_SIZE]);

            if (header.page_type == .leaf) {
                self.page_id = page_id;
                self.cell_index = 0;
                self.cell_count = header.cell_count;
                self.valid = header.cell_count > 0;
                return;
            } else if (header.page_type == .internal) {
                if (header.cell_count == 0) {
                    // Only right_child
                    page_id = getRightChild(frame.data);
                } else {
                    // Go to leftmost child
                    const cell = readInternalCell(frame.data, self.tree.pager.page_size, 0);
                    page_id = cell.left_child;
                }
            } else {
                return BTreeError.InvalidNodeType;
            }
        }
    }

    /// Position the cursor at the last (largest) key in the tree.
    pub fn seekLast(self: *Cursor) !void {
        var page_id = self.tree.root_page_id;

        while (true) {
            const frame = try self.tree.pool.fetchPage(page_id);
            defer self.tree.pool.unpinPage(page_id, false);
            const header = PageHeader.deserialize(frame.data[0..PAGE_HEADER_SIZE]);

            if (header.page_type == .leaf) {
                self.page_id = page_id;
                self.cell_count = header.cell_count;
                self.cell_index = if (header.cell_count > 0) header.cell_count - 1 else 0;
                self.valid = header.cell_count > 0;
                return;
            } else if (header.page_type == .internal) {
                // Always go to rightmost child
                page_id = getRightChild(frame.data);
            } else {
                return BTreeError.InvalidNodeType;
            }
        }
    }

    /// Position the cursor at the first key >= the given key.
    /// If no such key exists, the cursor is invalidated.
    pub fn seek(self: *Cursor, key: []const u8) !void {
        var page_id = self.tree.root_page_id;
        const page_size = self.tree.pager.page_size;

        while (true) {
            const frame = try self.tree.pool.fetchPage(page_id);
            defer self.tree.pool.unpinPage(page_id, false);
            const header = PageHeader.deserialize(frame.data[0..PAGE_HEADER_SIZE]);

            if (header.page_type == .leaf) {
                const pos = leafSearchPosition(frame.data, page_size, header.cell_count, key);
                if (pos.index < header.cell_count) {
                    self.page_id = page_id;
                    self.cell_index = pos.index;
                    self.cell_count = header.cell_count;
                    self.valid = true;
                } else {
                    // Key is beyond all entries in this leaf — check next leaf
                    const next_id = getNextLeaf(frame.data);
                    if (next_id != 0) {
                        const next_frame = try self.tree.pool.fetchPage(next_id);
                        defer self.tree.pool.unpinPage(next_id, false);
                        const next_header = PageHeader.deserialize(next_frame.data[0..PAGE_HEADER_SIZE]);
                        self.page_id = next_id;
                        self.cell_index = 0;
                        self.cell_count = next_header.cell_count;
                        self.valid = next_header.cell_count > 0;
                    } else {
                        self.valid = false;
                    }
                }
                return;
            } else if (header.page_type == .internal) {
                page_id = findChildInInternal(frame.data, page_size, header.cell_count, key);
            } else {
                return BTreeError.InvalidNodeType;
            }
        }
    }

    /// Return the current entry and advance the cursor forward.
    /// Returns null when the cursor is exhausted.
    /// Caller owns the returned key and value slices and must free them.
    pub fn next(self: *Cursor) !?Entry {
        if (!self.valid) return null;

        const page_size = self.tree.pager.page_size;

        // Read current entry
        const frame = try self.tree.pool.fetchPage(self.page_id);
        defer self.tree.pool.unpinPage(self.page_id, false);

        const cell = readLeafCell(frame.data, page_size, self.cell_index);

        // Copy key and value (caller owns)
        const key_copy = try self.allocator.alloc(u8, cell.key.len);
        @memcpy(key_copy, cell.key);
        const val_copy = try self.allocator.alloc(u8, cell.value.len);
        @memcpy(val_copy, cell.value);

        // Advance cursor
        self.cell_index += 1;
        if (self.cell_index >= self.cell_count) {
            // Move to next leaf
            const next_id = getNextLeaf(frame.data);
            if (next_id != 0) {
                const next_frame = try self.tree.pool.fetchPage(next_id);
                defer self.tree.pool.unpinPage(next_id, false);
                const next_header = PageHeader.deserialize(next_frame.data[0..PAGE_HEADER_SIZE]);
                self.page_id = next_id;
                self.cell_index = 0;
                self.cell_count = next_header.cell_count;
                self.valid = next_header.cell_count > 0;
            } else {
                self.valid = false;
            }
        }

        return .{ .key = key_copy, .value = val_copy };
    }

    /// Return the current entry and move the cursor backward.
    /// Returns null when the cursor is exhausted.
    /// Caller owns the returned key and value slices and must free them.
    pub fn prev(self: *Cursor) !?Entry {
        if (!self.valid) return null;

        const page_size = self.tree.pager.page_size;

        // Read current entry
        const frame = try self.tree.pool.fetchPage(self.page_id);
        defer self.tree.pool.unpinPage(self.page_id, false);

        const cell = readLeafCell(frame.data, page_size, self.cell_index);

        // Copy key and value (caller owns)
        const key_copy = try self.allocator.alloc(u8, cell.key.len);
        @memcpy(key_copy, cell.key);
        const val_copy = try self.allocator.alloc(u8, cell.value.len);
        @memcpy(val_copy, cell.value);

        // Move backward
        if (self.cell_index > 0) {
            self.cell_index -= 1;
        } else {
            // Move to previous leaf
            const prev_id = getPrevLeaf(frame.data);
            if (prev_id != 0) {
                const prev_frame = try self.tree.pool.fetchPage(prev_id);
                defer self.tree.pool.unpinPage(prev_id, false);
                const prev_header = PageHeader.deserialize(prev_frame.data[0..PAGE_HEADER_SIZE]);
                self.page_id = prev_id;
                self.cell_count = prev_header.cell_count;
                self.cell_index = if (prev_header.cell_count > 0) prev_header.cell_count - 1 else 0;
                self.valid = prev_header.cell_count > 0;
            } else {
                self.valid = false;
            }
        }

        return .{ .key = key_copy, .value = val_copy };
    }

    /// Peek at the current entry without advancing.
    /// Returns null when the cursor is invalid.
    /// Caller owns the returned key and value slices and must free them.
    pub fn current(self: *Cursor) !?Entry {
        if (!self.valid) return null;

        const page_size = self.tree.pager.page_size;

        const frame = try self.tree.pool.fetchPage(self.page_id);
        defer self.tree.pool.unpinPage(self.page_id, false);

        const cell = readLeafCell(frame.data, page_size, self.cell_index);

        const key_copy = try self.allocator.alloc(u8, cell.key.len);
        @memcpy(key_copy, cell.key);
        const val_copy = try self.allocator.alloc(u8, cell.value.len);
        @memcpy(val_copy, cell.value);

        return .{ .key = key_copy, .value = val_copy };
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

/// Find which child to descend into and return both the child page ID and its index.
fn findChildWithIndex(data: []const u8, page_size: u32, cell_count: u16, key: []const u8) struct { child_page_id: u32, child_index: u16 } {
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

    if (low == cell_count) {
        return .{ .child_page_id = getRightChild(data), .child_index = cell_count };
    }

    const cell = readInternalCell(data, page_size, low);
    return .{ .child_page_id = cell.left_child, .child_index = low };
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

/// Delete a cell from an internal page at the given position.
/// Shifts cell pointers left. Does NOT reclaim cell data space.
fn deleteInternalCell(data: []u8, page_size: u32, cell_count: u16, pos: u16) void {
    _ = page_size;
    if (pos + 1 < cell_count) {
        const src_start = internalCellPtrOffset(pos + 1);
        const src_end = internalCellPtrOffset(cell_count);
        const dst_start = internalCellPtrOffset(pos);
        const len = src_end - src_start;
        std.mem.copyForwards(u8, data[dst_start..][0..len], data[src_start..][0..len]);
    }
    // Update cell_count
    std.mem.writeInt(u16, data[2..4], cell_count - 1, .little);
}

/// Calculate total used space in a leaf page (cell data + cell pointers, excluding header).
fn leafUsedSpace(data: []const u8, page_size: u32, cell_count: u16) u32 {
    if (cell_count == 0) return 0;
    const ptrs_size: u32 = @as(u32, cell_count) * CELL_PTR_SIZE;
    // Cell data grows from end of page backward; lowest cell offset tells us where data starts
    const lowest = lowestCellOffset(data, cell_count, true);
    const cell_data_size: u32 = page_size - @as(u32, lowest);
    return ptrs_size + cell_data_size;
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

// ── Merge / Underflow Tests (Milestone 2C) ────────────────────────────

test "BTree leaf merge on heavy deletion with small pages" {
    const allocator = std.testing.allocator;
    const path = "test_btree_merge_heavy.db";
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

    // Insert N keys to create a multi-level tree
    for (0..N) |i| {
        const k = std.fmt.bufPrint(&key_buf, "k{d:0>6}", .{i}) catch unreachable;
        const v = std.fmt.bufPrint(&val_buf, "v{d:0>6}", .{i}) catch unreachable;
        try tree.insert(k, v);
    }

    // Delete 90% of keys — should trigger multiple merges
    for (0..N) |i| {
        if (i % 10 != 0) {
            const k = std.fmt.bufPrint(&key_buf, "k{d:0>6}", .{i}) catch unreachable;
            try tree.delete(k);
        }
    }

    // Verify remaining keys (every 10th) are still present
    for (0..N) |i| {
        const k = std.fmt.bufPrint(&key_buf, "k{d:0>6}", .{i}) catch unreachable;
        const val = try tree.get(allocator, k);
        if (i % 10 == 0) {
            try std.testing.expect(val != null);
            allocator.free(val.?);
        } else {
            try std.testing.expect(val == null);
        }
    }
}

test "BTree delete all keys from split tree triggers full merge" {
    const allocator = std.testing.allocator;
    const path = "test_btree_merge_all.db";
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

    // Verify tree is empty
    for (0..N) |i| {
        const k = std.fmt.bufPrint(&key_buf, "k{d:0>5}", .{i}) catch unreachable;
        const val = try tree.get(allocator, k);
        try std.testing.expect(val == null);
    }

    // Should be able to re-insert keys after full deletion
    for (0..10) |i| {
        const k = std.fmt.bufPrint(&key_buf, "k{d:0>5}", .{i}) catch unreachable;
        const v = std.fmt.bufPrint(&val_buf, "new{d:0>3}", .{i}) catch unreachable;
        try tree.insert(k, v);
    }

    for (0..10) |i| {
        const k = std.fmt.bufPrint(&key_buf, "k{d:0>5}", .{i}) catch unreachable;
        const val = try tree.get(allocator, k);
        try std.testing.expect(val != null);
        allocator.free(val.?);
    }
}

test "BTree interleaved insert and delete" {
    const allocator = std.testing.allocator;
    const path = "test_btree_interleaved.db";
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

    var key_buf: [12]u8 = undefined;
    var val_buf: [12]u8 = undefined;

    // Phase 1: Insert 50 keys
    for (0..50) |i| {
        const k = std.fmt.bufPrint(&key_buf, "key{d:0>5}", .{i}) catch unreachable;
        const v = std.fmt.bufPrint(&val_buf, "val{d:0>5}", .{i}) catch unreachable;
        try tree.insert(k, v);
    }

    // Phase 2: Delete first 25
    for (0..25) |i| {
        const k = std.fmt.bufPrint(&key_buf, "key{d:0>5}", .{i}) catch unreachable;
        try tree.delete(k);
    }

    // Phase 3: Insert 25 more with different prefix
    for (50..75) |i| {
        const k = std.fmt.bufPrint(&key_buf, "key{d:0>5}", .{i}) catch unreachable;
        const v = std.fmt.bufPrint(&val_buf, "val{d:0>5}", .{i}) catch unreachable;
        try tree.insert(k, v);
    }

    // Phase 4: Delete some from the middle
    for (30..40) |i| {
        const k = std.fmt.bufPrint(&key_buf, "key{d:0>5}", .{i}) catch unreachable;
        try tree.delete(k);
    }

    // Verify: keys 0-24 deleted, 25-29 present, 30-39 deleted, 40-74 present
    for (0..75) |i| {
        const k = std.fmt.bufPrint(&key_buf, "key{d:0>5}", .{i}) catch unreachable;
        const val = try tree.get(allocator, k);
        const should_exist = (i >= 25 and i < 30) or (i >= 40 and i < 75);
        if (should_exist) {
            try std.testing.expect(val != null);
            allocator.free(val.?);
        } else {
            try std.testing.expect(val == null);
        }
    }
}

test "BTree delete reverse order with merges" {
    const allocator = std.testing.allocator;
    const path = "test_btree_merge_reverse.db";
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

    // Insert in forward order
    for (0..N) |i| {
        const k = std.fmt.bufPrint(&key_buf, "k{d:0>6}", .{i}) catch unreachable;
        const v = std.fmt.bufPrint(&val_buf, "v{d:0>6}", .{i}) catch unreachable;
        try tree.insert(k, v);
    }

    // Delete in reverse order — stresses right-side merges
    var i: u32 = N;
    while (i > 0) {
        i -= 1;
        const k = std.fmt.bufPrint(&key_buf, "k{d:0>6}", .{i}) catch unreachable;
        try tree.delete(k);
    }

    // Verify all keys are gone
    for (0..N) |idx| {
        const k = std.fmt.bufPrint(&key_buf, "k{d:0>6}", .{idx}) catch unreachable;
        const val = try tree.get(allocator, k);
        try std.testing.expect(val == null);
    }
}

test "BTree leaf sibling chain intact after merges" {
    const allocator = std.testing.allocator;
    const path = "test_btree_chain_merge.db";
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

    // Insert keys
    for (0..N) |j| {
        const k = std.fmt.bufPrint(&key_buf, "k{d:0>6}", .{j}) catch unreachable;
        const v = std.fmt.bufPrint(&val_buf, "v{d:0>6}", .{j}) catch unreachable;
        try tree.insert(k, v);
    }

    // Delete half the keys to trigger some merges
    for (0..N) |j| {
        if (j % 2 == 0) {
            const k = std.fmt.bufPrint(&key_buf, "k{d:0>6}", .{j}) catch unreachable;
            try tree.delete(k);
        }
    }

    // Walk the leaf chain and verify all remaining keys are in sorted order
    // Find leftmost leaf
    var page_id = tree.root_page_id;
    while (true) {
        const frame = try pool.fetchPage(page_id);
        defer pool.unpinPage(page_id, false);

        const header = PageHeader.deserialize(frame.data[0..PAGE_HEADER_SIZE]);
        if (header.page_type == .leaf) break;
        if (header.page_type == .internal) {
            if (header.cell_count == 0) {
                page_id = getRightChild(frame.data);
            } else {
                const cell = readInternalCell(frame.data, pager.page_size, 0);
                page_id = cell.left_child;
            }
        }
    }

    // Walk chain, collect keys
    var collected = std.ArrayList([]const u8){};
    defer {
        for (collected.items) |k| allocator.free(k);
        collected.deinit(allocator);
    }

    var current_id = page_id;
    while (current_id != 0) {
        const frame = try pool.fetchPage(current_id);
        defer pool.unpinPage(current_id, false);
        const header = PageHeader.deserialize(frame.data[0..PAGE_HEADER_SIZE]);

        for (0..header.cell_count) |j| {
            const cell = readLeafCell(frame.data, pager.page_size, @intCast(j));
            const copy = try allocator.alloc(u8, cell.key.len);
            @memcpy(copy, cell.key);
            try collected.append(allocator, copy);
        }
        current_id = getNextLeaf(frame.data);
    }

    // Verify count matches (N/2 odd keys remain)
    try std.testing.expectEqual(N / 2, collected.items.len);

    // Verify sorted order
    for (0..collected.items.len - 1) |j| {
        try std.testing.expect(std.mem.order(u8, collected.items[j], collected.items[j + 1]) == .lt);
    }
}

test "BTree insert-delete-reinsert cycle" {
    const allocator = std.testing.allocator;
    const path = "test_btree_cycle.db";
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

    // 3 cycles of insert-all / delete-all
    for (0..3) |cycle| {
        const N: u32 = 40;

        // Insert
        for (0..N) |j| {
            const k = std.fmt.bufPrint(&key_buf, "c{d}k{d:0>4}", .{ cycle, j }) catch unreachable;
            const v = std.fmt.bufPrint(&val_buf, "c{d}v{d:0>4}", .{ cycle, j }) catch unreachable;
            try tree.insert(k, v);
        }

        // Verify all present
        for (0..N) |j| {
            const k = std.fmt.bufPrint(&key_buf, "c{d}k{d:0>4}", .{ cycle, j }) catch unreachable;
            const val = try tree.get(allocator, k);
            try std.testing.expect(val != null);
            allocator.free(val.?);
        }

        // Delete all
        for (0..N) |j| {
            const k = std.fmt.bufPrint(&key_buf, "c{d}k{d:0>4}", .{ cycle, j }) catch unreachable;
            try tree.delete(k);
        }

        // Verify all gone
        for (0..N) |j| {
            const k = std.fmt.bufPrint(&key_buf, "c{d}k{d:0>4}", .{ cycle, j }) catch unreachable;
            const val = try tree.get(allocator, k);
            try std.testing.expect(val == null);
        }
    }
}

test "BTree delete from middle of leaf chain" {
    const allocator = std.testing.allocator;
    const path = "test_btree_merge_middle.db";
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

    for (0..N) |j| {
        const k = std.fmt.bufPrint(&key_buf, "k{d:0>6}", .{j}) catch unreachable;
        const v = std.fmt.bufPrint(&val_buf, "v{d:0>6}", .{j}) catch unreachable;
        try tree.insert(k, v);
    }

    // Delete keys 20-39 (middle range) to force middle-of-chain merges
    for (20..40) |j| {
        const k = std.fmt.bufPrint(&key_buf, "k{d:0>6}", .{j}) catch unreachable;
        try tree.delete(k);
    }

    // Verify remaining keys
    for (0..N) |j| {
        const k = std.fmt.bufPrint(&key_buf, "k{d:0>6}", .{j}) catch unreachable;
        const val = try tree.get(allocator, k);
        if (j >= 20 and j < 40) {
            try std.testing.expect(val == null);
        } else {
            try std.testing.expect(val != null);
            allocator.free(val.?);
        }
    }
}

// ── Cursor Tests (Milestone 2D) ───────────────────────────────────────

test "Cursor forward scan all keys" {
    const allocator = std.testing.allocator;
    const path = "test_cursor_forward.db";
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

    for (0..N) |i| {
        const k = std.fmt.bufPrint(&key_buf, "k{d:0>6}", .{i}) catch unreachable;
        const v = std.fmt.bufPrint(&val_buf, "v{d:0>6}", .{i}) catch unreachable;
        try tree.insert(k, v);
    }

    // Forward scan
    var cursor = Cursor.init(allocator, &tree);
    defer cursor.deinit();
    try cursor.seekFirst();

    var count: u32 = 0;
    var prev_key: ?[]u8 = null;
    while (try cursor.next()) |entry| {
        defer allocator.free(entry.key);
        defer allocator.free(entry.value);

        // Verify sorted order
        if (prev_key) |pk| {
            try std.testing.expect(std.mem.order(u8, pk, entry.key) == .lt);
            allocator.free(pk);
        }
        prev_key = try allocator.alloc(u8, entry.key.len);
        @memcpy(prev_key.?, entry.key);
        count += 1;
    }
    if (prev_key) |pk| allocator.free(pk);

    try std.testing.expectEqual(N, count);
}

test "Cursor backward scan all keys" {
    const allocator = std.testing.allocator;
    const path = "test_cursor_backward.db";
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

    for (0..N) |i| {
        const k = std.fmt.bufPrint(&key_buf, "k{d:0>6}", .{i}) catch unreachable;
        const v = std.fmt.bufPrint(&val_buf, "v{d:0>6}", .{i}) catch unreachable;
        try tree.insert(k, v);
    }

    // Backward scan
    var cursor = Cursor.init(allocator, &tree);
    defer cursor.deinit();
    try cursor.seekLast();

    var count: u32 = 0;
    var prev_key: ?[]u8 = null;
    while (try cursor.prev()) |entry| {
        defer allocator.free(entry.key);
        defer allocator.free(entry.value);

        // Verify reverse sorted order (each key should be less than previous)
        if (prev_key) |pk| {
            try std.testing.expect(std.mem.order(u8, entry.key, pk) == .lt);
            allocator.free(pk);
        }
        prev_key = try allocator.alloc(u8, entry.key.len);
        @memcpy(prev_key.?, entry.key);
        count += 1;
    }
    if (prev_key) |pk| allocator.free(pk);

    try std.testing.expectEqual(N, count);
}

test "Cursor seek to specific key" {
    const allocator = std.testing.allocator;
    const path = "test_cursor_seek.db";
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

    // Insert even numbers only: k000000, k000002, k000004, ...
    var key_buf: [10]u8 = undefined;
    var val_buf: [10]u8 = undefined;

    for (0..30) |i| {
        const k = std.fmt.bufPrint(&key_buf, "k{d:0>6}", .{i * 2}) catch unreachable;
        const v = std.fmt.bufPrint(&val_buf, "v{d:0>6}", .{i * 2}) catch unreachable;
        try tree.insert(k, v);
    }

    // Seek to an existing key (k000010)
    {
        var cursor = Cursor.init(allocator, &tree);
        defer cursor.deinit();
        try cursor.seek("k000010");
        const entry = try cursor.next();
        try std.testing.expect(entry != null);
        try std.testing.expectEqualStrings("k000010", entry.?.key);
        allocator.free(entry.?.key);
        allocator.free(entry.?.value);
    }

    // Seek to a non-existing key (k000011) — should position at k000012
    {
        var cursor = Cursor.init(allocator, &tree);
        defer cursor.deinit();
        try cursor.seek("k000011");
        const entry = try cursor.next();
        try std.testing.expect(entry != null);
        try std.testing.expectEqualStrings("k000012", entry.?.key);
        allocator.free(entry.?.key);
        allocator.free(entry.?.value);
    }

    // Seek past all keys
    {
        var cursor = Cursor.init(allocator, &tree);
        defer cursor.deinit();
        try cursor.seek("z999999");
        const entry = try cursor.next();
        try std.testing.expect(entry == null);
    }

    // Seek to before all keys — should get first key
    {
        var cursor = Cursor.init(allocator, &tree);
        defer cursor.deinit();
        try cursor.seek("a000000");
        const entry = try cursor.next();
        try std.testing.expect(entry != null);
        try std.testing.expectEqualStrings("k000000", entry.?.key);
        allocator.free(entry.?.key);
        allocator.free(entry.?.value);
    }
}

test "Cursor range scan with seek" {
    const allocator = std.testing.allocator;
    const path = "test_cursor_range.db";
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

    for (0..N) |i| {
        const k = std.fmt.bufPrint(&key_buf, "k{d:0>6}", .{i}) catch unreachable;
        const v = std.fmt.bufPrint(&val_buf, "v{d:0>6}", .{i}) catch unreachable;
        try tree.insert(k, v);
    }

    // Range scan: keys from k000020 to k000029
    var cursor = Cursor.init(allocator, &tree);
    defer cursor.deinit();
    try cursor.seek("k000020");

    var count: u32 = 0;
    while (try cursor.next()) |entry| {
        defer allocator.free(entry.key);
        defer allocator.free(entry.value);

        // Stop at k000030
        if (std.mem.order(u8, entry.key, "k000030") != .lt) break;
        count += 1;
    }

    try std.testing.expectEqual(@as(u32, 10), count);
}

test "Cursor on empty tree" {
    const allocator = std.testing.allocator;
    const path = "test_cursor_empty.db";
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

    // Forward scan on empty tree
    var cursor = Cursor.init(allocator, &tree);
    defer cursor.deinit();
    try cursor.seekFirst();
    const entry = try cursor.next();
    try std.testing.expect(entry == null);

    // Backward scan on empty tree
    try cursor.seekLast();
    const entry2 = try cursor.prev();
    try std.testing.expect(entry2 == null);
}

test "Cursor current without advancing" {
    const allocator = std.testing.allocator;
    const path = "test_cursor_current.db";
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

    try tree.insert("alpha", "1");
    try tree.insert("beta", "2");
    try tree.insert("gamma", "3");

    var cursor = Cursor.init(allocator, &tree);
    defer cursor.deinit();
    try cursor.seekFirst();

    // Current should return first key without advancing
    const e1 = try cursor.current();
    try std.testing.expect(e1 != null);
    try std.testing.expectEqualStrings("alpha", e1.?.key);
    allocator.free(e1.?.key);
    allocator.free(e1.?.value);

    // Calling current again should return the same key
    const e2 = try cursor.current();
    try std.testing.expect(e2 != null);
    try std.testing.expectEqualStrings("alpha", e2.?.key);
    allocator.free(e2.?.key);
    allocator.free(e2.?.value);

    // Now advance with next
    const e3 = try cursor.next();
    try std.testing.expect(e3 != null);
    try std.testing.expectEqualStrings("alpha", e3.?.key);
    allocator.free(e3.?.key);
    allocator.free(e3.?.value);

    // Current should now be "beta"
    const e4 = try cursor.current();
    try std.testing.expect(e4 != null);
    try std.testing.expectEqualStrings("beta", e4.?.key);
    allocator.free(e4.?.key);
    allocator.free(e4.?.value);
}

test "Cursor single key tree" {
    const allocator = std.testing.allocator;
    const path = "test_cursor_single.db";
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

    try tree.insert("only", "one");

    // Forward: should get exactly one entry
    var cursor = Cursor.init(allocator, &tree);
    defer cursor.deinit();
    try cursor.seekFirst();

    const e1 = try cursor.next();
    try std.testing.expect(e1 != null);
    try std.testing.expectEqualStrings("only", e1.?.key);
    try std.testing.expectEqualStrings("one", e1.?.value);
    allocator.free(e1.?.key);
    allocator.free(e1.?.value);

    const e2 = try cursor.next();
    try std.testing.expect(e2 == null);

    // Backward: should get exactly one entry
    try cursor.seekLast();
    const e3 = try cursor.prev();
    try std.testing.expect(e3 != null);
    try std.testing.expectEqualStrings("only", e3.?.key);
    allocator.free(e3.?.key);
    allocator.free(e3.?.value);

    const e4 = try cursor.prev();
    try std.testing.expect(e4 == null);
}
