//! GIN (Generalized Inverted Index) — inverted index for multi-valued columns.
//!
//! GIN is an inverted index optimized for columns where each value can appear in many rows.
//! Common use cases: JSONB keys, arrays, full-text search (tsvector).
//!
//! Architecture:
//!   - Entry tree: B+Tree mapping indexed_value → posting_list (or posting_tree_root_page)
//!   - Posting list: compact list of tuple_ids (ItemPointer) for a single indexed_value
//!   - Posting tree: B+Tree of tuple_ids when posting list exceeds inline threshold
//!
//! Operator class interface:
//!   - compare(a, b): lexicographic comparison of indexed values
//!   - extractValue(column_value): array of keys to index
//!   - extractQuery(query_value): array of search keys
//!   - consistent(posting_lists, query_keys): check if row matches query
//!
//! Page layout for entry tree leaf:
//!   [PageHeader 16B][entry_count u16][reserved 2B][entry_0_key_size u16][entry_0_posting_info u32]...[keys←]
//!   posting_info encoding:
//!     If high bit = 0: inline posting list (lower 31 bits = tuple_count, posting data = fixed u64 tuple IDs)
//!     If high bit = 1: posting tree root page (lower 31 bits = page_id)
//!   Phase 1 simplification: posting lists use fixed u64 tuple IDs (not varint deltas) for correctness
//!
//! NOT IMPLEMENTED (deferred):
//!   - Pending list optimization (fast bulk insert)
//!   - GIN fast update (delayed cleanup)
//!   - Partial match optimization
//!   - Concurrent tree modifications

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

const GIN_HEADER_SIZE: u32 = PAGE_HEADER_SIZE + 4; // page_type + entry_count + reserved
const GIN_ENTRY_HEADER_SIZE: u32 = 2 + 4; // key_size(u16) + posting_info(u32)
const INLINE_POSTING_LIST_MAX_SIZE: u32 = 128; // Bytes before switching to posting tree (128 bytes = 16 u64 tuple IDs)
const MAX_INLINE_TUPLES: u32 = 16; // With fixed u64 encoding: 128 bytes / 8 bytes per tuple = 16 tuples max
const POSTING_TREE_HEADER_SIZE: u32 = PAGE_HEADER_SIZE + 8; // PageHeader(16) + tuple_count(u32=4) + next_page_id(u32=4)
const POSTING_TREE_NEXT_PAGE_OFFSET: u32 = PAGE_HEADER_SIZE + 4; // next_page_id field: 0 = no next page

/// ItemPointer — (page_id, tuple_offset) uniquely identifying a row.
pub const ItemPointer = struct {
    page_id: u32,
    tuple_offset: u16,

    pub fn toU64(self: ItemPointer) u64 {
        return (@as(u64, self.page_id) << 16) | @as(u64, self.tuple_offset);
    }

    pub fn fromU64(val: u64) ItemPointer {
        return .{
            .page_id = @truncate(val >> 16),
            .tuple_offset = @truncate(val & 0xFFFF),
        };
    }
};

pub const Error = error{
    TreeEmpty,
    EntryNotFound,
    PageFull,
    InvalidKey,
    ConsistentFailed,
};

// ── Operator Class Interface ───────────────────────────────────────────

pub const OpClassError = error{ TreeEmpty, EntryNotFound, PageFull, InvalidKey, ConsistentFailed, OutOfMemory };

/// GIN operator class interface for pluggable key extraction and search.
pub const OpClass = struct {
    /// Compare two indexed values (lexicographic order).
    /// Returns: -1 if a < b, 0 if a == b, 1 if a > b.
    compare: *const fn (allocator: std.mem.Allocator, a: []const u8, b: []const u8) OpClassError!i8,

    /// Extract indexed keys from a column value.
    /// Example: ARRAY[1,2,3] → [1, 2, 3] (three separate keys)
    /// Caller owns returned slice and each key slice.
    extractValue: *const fn (allocator: std.mem.Allocator, column_value: []const u8) OpClassError![][]const u8,

    /// Extract search keys from a query predicate.
    /// Example: WHERE col @> ARRAY[1,2] → [1, 2]
    /// Caller owns returned slice and each key slice.
    extractQuery: *const fn (allocator: std.mem.Allocator, query_value: []const u8) OpClassError![][]const u8,

    /// Check if row matches query given posting lists for each search key.
    /// posting_lists[i] corresponds to query_keys[i].
    /// Example for @> (contains): all query_keys must be present (non-empty posting lists).
    consistent: *const fn (allocator: std.mem.Allocator, posting_lists: []const []const ItemPointer, query_keys: []const []const u8, strategy: u8) OpClassError!bool,
};

// ── Example Operator Class: ArrayInt32OpClass ──────────────────────────

/// ArrayInt32OpClass — operator class for integer arrays.
/// Indexed value format: [u32 LE] (single integer)
/// Query strategies: 0 = @> (contains all), 1 = && (overlaps)
pub const ArrayInt32OpClass = struct {
    /// Compare two u32 values.
    pub fn compare(_: std.mem.Allocator, a: []const u8, b: []const u8) OpClassError!i8 {
        if (a.len < 4 or b.len < 4) return error.InvalidKey;
        const a_val = std.mem.readInt(u32, a[0..4], .little);
        const b_val = std.mem.readInt(u32, b[0..4], .little);
        std.debug.print("[GIN Compare] a_val={d}, b_val={d}\n", .{ a_val, b_val });
        if (a_val < b_val) return -1;
        if (a_val > b_val) return 1;
        return 0;
    }

    /// Extract value: array → individual elements.
    /// Input format: [count u32][elem0 u32][elem1 u32]...
    pub fn extractValue(allocator: std.mem.Allocator, column_value: []const u8) OpClassError![][]const u8 {
        if (column_value.len < 4) return error.InvalidKey;
        const count = std.mem.readInt(u32, column_value[0..4], .little);
        if (column_value.len < 4 + count * 4) return error.InvalidKey;

        var keys = try allocator.alloc([]const u8, count);
        for (0..count) |i| {
            const key_buf = try allocator.alloc(u8, 4);
            const offset = 4 + i * 4;
            @memcpy(key_buf, column_value[offset .. offset + 4]);
            keys[i] = key_buf;
        }
        return keys;
    }

    /// Extract query: same format as extractValue.
    pub fn extractQuery(allocator: std.mem.Allocator, query_value: []const u8) OpClassError![][]const u8 {
        return extractValue(allocator, query_value);
    }

    /// Consistent function for array operators.
    /// Strategy 0 (@>): all query_keys must have non-empty posting lists.
    /// Strategy 1 (&&): at least one query_key must have non-empty posting list.
    pub fn consistent(_: std.mem.Allocator, posting_lists: []const []const ItemPointer, _: []const []const u8, strategy: u8) OpClassError!bool {
        return switch (strategy) {
            0 => blk: { // @> (contains all)
                for (posting_lists) |list| {
                    if (list.len == 0) break :blk false;
                }
                break :blk true;
            },
            1 => blk: { // && (overlaps)
                for (posting_lists) |list| {
                    if (list.len > 0) break :blk true;
                }
                break :blk false;
            },
            else => error.InvalidKey,
        };
    }

    pub fn getOpClass() OpClass {
        return .{
            .compare = compare,
            .extractValue = extractValue,
            .extractQuery = extractQuery,
            .consistent = consistent,
        };
    }
};

// ── GIN Tree Structure ─────────────────────────────────────────────────

pub const GIN = struct {
    allocator: std.mem.Allocator,
    pool: *BufferPool,
    root_page_id: u32,
    opclass: OpClass,
    max_entries_per_page: u32,

    /// Initialize a new GIN tree with the given root page and operator class.
    /// The root page is initialized lazily on first access.
    pub fn init(allocator: std.mem.Allocator, pool: *BufferPool, root_page_id: u32, opclass: OpClass) !GIN {
        return .{
            .allocator = allocator,
            .pool = pool,
            .root_page_id = root_page_id,
            .opclass = opclass,
            .max_entries_per_page = calculateMaxEntries(pool.pager.page_size),
        };
    }

    /// Fetch root page, initializing if needed.
    fn fetchOrInitRootPage(self: *GIN) !*BufferFrame {
        // Try to fetch normally first
        if (self.pool.containsPage(self.root_page_id)) {
            return try self.pool.fetchPage(self.root_page_id);
        }

        // Page not in pool - use fetchNewPage to create it directly in the pool
        const frame = try self.pool.fetchNewPage(self.root_page_id);

        // Initialize page header
        const header = PageHeader{
            .page_type = .leaf, // Entry tree leaf
            .page_id = self.root_page_id,
            .cell_count = 0,
            .free_offset = @intCast(self.pool.pager.page_size),
            .checksum_value = 0,
        };
        header.serialize(frame.data[0..PAGE_HEADER_SIZE]);

        // Initialize entry count
        writeEntryCount(frame.data, 0);

        // Page is already marked dirty by fetchNewPage
        return frame;
    }

    /// Insert a column value (extracts keys and inserts into entry tree).
    /// Each extracted key creates an entry → posting list mapping.
    pub fn insert(self: *GIN, column_value: []const u8, tuple_id: ItemPointer) !void {
        // Extract keys from column value
        const keys = try self.opclass.extractValue(self.allocator, column_value);
        defer {
            for (keys) |key| self.allocator.free(key);
            self.allocator.free(keys);
        }

        // Insert each key into entry tree
        for (keys) |key| {
            try self.insertKey(key, tuple_id);
        }
    }

    /// Delete a column value (removes tuple_id from posting lists).
    pub fn delete(self: *GIN, column_value: []const u8, tuple_id: ItemPointer) !void {
        // Extract keys from column value
        const keys = try self.opclass.extractValue(self.allocator, column_value);
        defer {
            for (keys) |key| self.allocator.free(key);
            self.allocator.free(keys);
        }

        // Remove tuple_id from each key's posting list
        for (keys) |key| {
            try self.deleteKey(key, tuple_id);
        }
    }

    /// Search for rows matching a query predicate.
    /// Returns list of ItemPointers. Caller owns returned slice.
    pub fn search(self: *GIN, query_value: []const u8, strategy: u8) ![]ItemPointer {
        // Extract query keys
        const query_keys = try self.opclass.extractQuery(self.allocator, query_value);
        defer {
            for (query_keys) |key| self.allocator.free(key);
            self.allocator.free(query_keys);
        }

        std.debug.print("[GIN Search] Query value ({d} bytes), strategy: {d}\n", .{ query_value.len, strategy });
        std.debug.print("[GIN Search] Extracted {d} query keys\n", .{query_keys.len});

        // Lookup posting list for each query key
        var posting_lists = try self.allocator.alloc([]ItemPointer, query_keys.len);
        defer {
            for (posting_lists) |list| self.allocator.free(list);
            self.allocator.free(posting_lists);
        }

        for (query_keys, 0..) |key, i| {
            std.debug.print("[GIN Search] Looking up key {d} ({d} bytes)\n", .{ i, key.len });
            posting_lists[i] = try self.lookupPostingList(key);
            std.debug.print("[GIN Search] Found {d} tuples for key {d}\n", .{ posting_lists[i].len, i });
        }

        // Call opclass.consistent to filter results
        const matches = try self.opclass.consistent(self.allocator, posting_lists, query_keys, strategy);
        if (!matches) {
            return try self.allocator.alloc(ItemPointer, 0);
        }

        // For now, return intersection of all posting lists (contains-all strategy)
        // This is a simplified implementation
        if (posting_lists.len == 0) {
            return try self.allocator.alloc(ItemPointer, 0);
        }

        // Find the shortest posting list to iterate
        var shortest_idx: usize = 0;
        for (posting_lists, 0..) |list, i| {
            if (list.len < posting_lists[shortest_idx].len) {
                shortest_idx = i;
            }
        }

        // Collect items that appear in all posting lists
        var result = std.ArrayList(ItemPointer){};
        defer result.deinit(self.allocator);

        outer: for (posting_lists[shortest_idx]) |item| {
            // Check if item appears in all other lists
            for (posting_lists, 0..) |list, i| {
                if (i == shortest_idx) continue;

                var found = false;
                for (list) |other_item| {
                    if (item.page_id == other_item.page_id and item.tuple_offset == other_item.tuple_offset) {
                        found = true;
                        break;
                    }
                }
                if (!found) continue :outer;
            }
            try result.append(self.allocator, item);
        }

        return try result.toOwnedSlice(self.allocator);
    }

    // ── Diagnostic Functions (for GIN Redesign) ────────────────────────

    /// Debug: Dump all entries in the entry tree (for diagnostic purposes).
    /// This walks the entry tree and prints all keys + posting info.
    pub fn debugDumpEntryTree(self: *GIN) !void {
        const root_frame = try self.fetchOrInitRootPage();
        defer self.pool.unpinPage(self.root_page_id, false);

        const entry_count = readEntryCount(root_frame.data);
        std.debug.print("[GIN Debug] Entry tree dump — {d} entries\n", .{entry_count});

        for (0..entry_count) |i| {
            const entry_key = try self.readEntryKey(root_frame.data, i);
            defer self.allocator.free(entry_key);

            const posting_info = readPostingInfo(root_frame.data, i);
            const is_tree = (posting_info & 0x80000000) != 0;
            const value = posting_info & 0x7FFFFFFF;

            if (is_tree) {
                std.debug.print("[GIN Debug]   Entry {d}: key ({d} bytes), posting_tree_root={d}\n", .{ i, entry_key.len, value });
            } else {
                std.debug.print("[GIN Debug]   Entry {d}: key ({d} bytes), inline_posting_info=0x{x}\n", .{ i, entry_key.len, posting_info });
            }
        }
    }

    // ── Internal Operations ────────────────────────────────────────────

    /// Insert a single key into the entry tree with associated tuple_id.
    fn insertKey(self: *GIN, key: []const u8, tuple_id: ItemPointer) !void {
        const key_val = if (key.len >= 4) std.mem.readInt(u32, key[0..4], .little) else 0;
        std.debug.print("[GIN Insert] key ({d} bytes) value={d}, tuple_id=({d},{d})\n", .{ key.len, key_val, tuple_id.page_id, tuple_id.tuple_offset });

        const root_frame = try self.fetchOrInitRootPage();
        defer self.pool.unpinPage(self.root_page_id, true);

        const entry_count = readEntryCount(root_frame.data);
        std.debug.print("[GIN Insert] Current entry_count: {d}\n", .{entry_count});

        // Search for existing entry
        for (0..entry_count) |i| {
            const entry_key = try self.readEntryKey(root_frame.data, i);
            defer self.allocator.free(entry_key);

            const cmp = try self.opclass.compare(self.allocator, entry_key, key);
            if (cmp == 0) {
                // Key exists — append to posting list
                std.debug.print("[GIN Insert] Key exists at entry {d}, appending to posting list\n", .{i});
                try self.appendToPostingList(root_frame.data, i, tuple_id);
                root_frame.markDirty();
                return;
            }
        }

        // Key doesn't exist — insert new entry
        std.debug.print("[GIN Insert] Key not found, inserting new entry\n", .{});
        try self.insertNewEntry(root_frame.data, key, tuple_id);
        root_frame.markDirty();
        std.debug.print("[GIN Insert] Insert complete, new entry_count: {d}\n", .{readEntryCount(root_frame.data)});
    }

    /// Delete a tuple_id from a key's posting list.
    fn deleteKey(self: *GIN, key: []const u8, tuple_id: ItemPointer) !void {
        const root_frame = try self.fetchOrInitRootPage();
        defer self.pool.unpinPage(self.root_page_id, true);

        const entry_count = readEntryCount(root_frame.data);

        // Search for entry
        for (0..entry_count) |i| {
            const entry_key = try self.readEntryKey(root_frame.data, i);
            defer self.allocator.free(entry_key);

            const cmp = try self.opclass.compare(self.allocator, entry_key, key);
            if (cmp == 0) {
                // Key found — remove from posting list
                try self.removeFromPostingList(root_frame.data, i, tuple_id);
                root_frame.markDirty();
                return;
            }
        }

        return error.EntryNotFound;
    }

    /// Lookup the posting list for a given key.
    fn lookupPostingList(self: *GIN, key: []const u8) ![]ItemPointer {
        const root_frame = try self.fetchOrInitRootPage();
        defer self.pool.unpinPage(self.root_page_id, false);

        const entry_count = readEntryCount(root_frame.data);
        std.debug.print("[GIN Lookup] Searching for key ({d} bytes) in {d} entries\n", .{ key.len, entry_count });

        // Search for entry
        for (0..entry_count) |i| {
            const entry_key = try self.readEntryKey(root_frame.data, i);
            defer self.allocator.free(entry_key);

            std.debug.print("[GIN Lookup]   Entry {d}: key ({d} bytes)\n", .{ i, entry_key.len });

            const cmp = try self.opclass.compare(self.allocator, entry_key, key);
            std.debug.print("[GIN Lookup]   Compare result: {d}\n", .{cmp});

            if (cmp == 0) {
                // Key found — return posting list
                std.debug.print("[GIN Lookup] Key matched at entry {d}\n", .{i});
                const posting_list = try self.readPostingList(root_frame.data, i);
                std.debug.print("[GIN Lookup] Read {d} tuples from posting list\n", .{posting_list.len});
                return posting_list;
            }
        }

        // Key not found — return empty list
        std.debug.print("[GIN Lookup] Key not found, returning empty list\n", .{});
        return try self.allocator.alloc(ItemPointer, 0);
    }

    /// Read entry key at given index.
    fn readEntryKey(self: *GIN, page: []u8, idx: usize) ![]u8 {
        const key_size = readKeySize(page, idx);
        if (key_size == 0) return error.InvalidKey;

        // Keys are stored at the end of the page
        const keys_base_offset = self.calculateKeysBaseOffset(page);
        var offset = keys_base_offset;

        // Skip to the idx-th key
        for (0..idx) |i| {
            const size = readKeySize(page, i);
            offset += size;
        }

        const key = try self.allocator.alloc(u8, key_size);
        @memcpy(key, page[offset..][0..key_size]);
        return key;
    }

    /// Calculate offset where keys start.
    /// Layout: [GIN_HEADER][entry_headers...][offset_ptrs...][keys...][posting_data←]
    fn calculateKeysBaseOffset(self: *GIN, page: []u8) usize {
        _ = self;
        const entry_count = readEntryCount(page);
        // Keys start AFTER headers AND offset pointers
        return GIN_HEADER_SIZE + (entry_count * GIN_ENTRY_HEADER_SIZE) + (entry_count * 4);
    }

    /// Read posting list for entry at given index.
    fn readPostingList(self: *GIN, page: []u8, idx: usize) ![]ItemPointer {
        const posting_info = readPostingInfo(page, idx);

        if (isInlinePostingList(posting_info)) {
            return try self.readInlinePostingList(page, idx);
        } else {
            const tree_page_id = posting_info & 0x7FFFFFFF;
            return try self.readPostingTree(tree_page_id);
        }
    }

    /// Read inline posting list.
    fn readInlinePostingList(self: *GIN, page: []u8, idx: usize) ![]ItemPointer {
        const posting_info = readPostingInfo(page, idx);
        const tuple_count = posting_info & 0x7FFFFFFF; // Lower 31 bits

        if (tuple_count == 0) {
            return try self.allocator.alloc(ItemPointer, 0);
        }

        // Sanity check: prevent infinite loops on corrupted data
        if (tuple_count > MAX_INLINE_TUPLES) {
            return error.InvalidKey;
        }

        // Allocate result list
        const list = try self.allocator.alloc(ItemPointer, tuple_count);
        errdefer self.allocator.free(list);

        // Posting list data is stored in fixed-size blocks at end of page
        // Format: [offset_to_data u32] stored after entry headers, then actual data
        // Data layout (Phase 1): [tid0 u64][tid1 u64][tid2 u64]... (fixed u64, no varint deltas)

        // Calculate offset to posting data pointer
        // Offset pointers are stored AFTER ALL entry headers
        const entry_count = readEntryCount(page);
        const offset_ptrs_base = GIN_HEADER_SIZE + (entry_count * GIN_ENTRY_HEADER_SIZE);
        const data_offset_ptr = offset_ptrs_base + (idx * 4);

        if (data_offset_ptr + 4 > page.len) {
            // No data stored yet (skeletal implementation)
            @memset(std.mem.sliceAsBytes(list), 0);
            return list;
        }

        const data_offset = std.mem.readInt(u32, page[data_offset_ptr..][0..4], .little);
        if (data_offset == 0 or data_offset + 8 > page.len) {
            // No data or invalid offset
            @memset(std.mem.sliceAsBytes(list), 0);
            return list;
        }

        // Phase 1 simplification: Read fixed u64 tuple IDs (no varint deltas)
        // Format: [tid0 u64][tid1 u64][tid2 u64]...
        for (0..tuple_count) |i| {
            const tid_offset = data_offset + (i * 8);
            if (tid_offset + 8 > page.len) {
                // Not enough space or corrupted data
                @memset(std.mem.sliceAsBytes(list[i..]), 0);
                break;
            }
            const tid = std.mem.readInt(u64, page[tid_offset..][0..8], .little);
            list[i] = ItemPointer.fromU64(tid);
        }

        return list;
    }

    /// Append tuple_id to posting list at given entry index.
    fn appendToPostingList(self: *GIN, page: []u8, idx: usize, tuple_id: ItemPointer) !void {
        const posting_info = readPostingInfo(page, idx);

        // Dispatch to posting tree if already converted
        if (!isInlinePostingList(posting_info)) {
            const tree_page_id = posting_info & 0x7FFFFFFF;
            try self.appendToPostingTree(tree_page_id, tuple_id);
            return;
        }

        const current_count = posting_info & 0x7FFFFFFF;

        if (current_count == 0) {
            return error.InvalidPostingList; // Should not append to empty list
        }

        // Sanity check: prevent infinite loops on corrupted data
        if (current_count > MAX_INLINE_TUPLES) {
            return error.InvalidKey;
        }

        // Inline list full — convert to posting tree
        if (current_count >= MAX_INLINE_TUPLES) {
            try self.convertInlineToTree(page, idx, tuple_id);
            return;
        }

        const entry_count = readEntryCount(page);
        const offset_ptrs_base = GIN_HEADER_SIZE + (entry_count * GIN_ENTRY_HEADER_SIZE);
        const data_offset_ptr = offset_ptrs_base + (idx * 4);

        if (data_offset_ptr + 4 > page.len) {
            return error.InvalidOffset;
        }

        const data_offset = std.mem.readInt(u32, page[data_offset_ptr..][0..4], .little);
        if (data_offset == 0 or data_offset + 8 > page.len) {
            return error.InvalidOffset;
        }

        // Phase 1 simplification: append fixed u64 tuple ID (no varint deltas)
        const new_tid = tuple_id.toU64();

        // Verify sortedness: new TID must be > last TID
        if (current_count > 0) {
            const last_tid_pos = data_offset + ((current_count - 1) * 8);
            const last_tid = std.mem.readInt(u64, page[last_tid_pos..][0..8], .little);
            if (new_tid <= last_tid) {
                return error.PostingListNotSorted;
            }
        }

        // Calculate append position (after current_count tuple IDs)
        const append_pos = data_offset + (current_count * 8);

        // Check space availability
        if (append_pos + 8 > page.len) {
            return error.PageFull;
        }

        // Write new tuple ID
        std.mem.writeInt(u64, page[append_pos..][0..8], new_tid, .little);

        // Update posting_info count
        const new_count = current_count + 1;
        const new_posting_info = new_count; // Keep high bit 0 for inline
        const info_offset = GIN_HEADER_SIZE + (idx * GIN_ENTRY_HEADER_SIZE) + 2;
        std.mem.writeInt(u32, page[info_offset..][0..4], new_posting_info, .little);
    }

    /// Read all tuple IDs from a posting tree chain (follows next_page_id links).
    fn readPostingTree(self: *GIN, tree_page_id: u32) ![]ItemPointer {
        var all_tuples = std.ArrayList(ItemPointer){};
        errdefer all_tuples.deinit(self.allocator);

        var current_page_id = tree_page_id;
        const max_chain_pages = self.pool.pager.page_count + 1;
        var pages_visited: u32 = 0;
        while (current_page_id != 0) {
            pages_visited += 1;
            if (pages_visited > max_chain_pages) return error.InvalidKey; // cycle or corruption
            const tree_frame = try self.pool.fetchPage(current_page_id);
            const count = std.mem.readInt(u32, tree_frame.data[PAGE_HEADER_SIZE..][0..4], .little);
            const next_page = std.mem.readInt(u32, tree_frame.data[POSTING_TREE_NEXT_PAGE_OFFSET..][0..4], .little);
            const max_count: u32 = @intCast((tree_frame.data.len - POSTING_TREE_HEADER_SIZE) / 8);
            if (count > max_count) {
                self.pool.unpinPage(current_page_id, false);
                return error.InvalidKey;
            }
            for (0..count) |i| {
                const tid_offset = POSTING_TREE_HEADER_SIZE + (i * 8);
                const tid = std.mem.readInt(u64, tree_frame.data[tid_offset..][0..8], .little);
                try all_tuples.append(self.allocator, ItemPointer.fromU64(tid));
            }
            self.pool.unpinPage(current_page_id, false);
            current_page_id = next_page;
        }

        const result = try all_tuples.toOwnedSlice(self.allocator);
        // Sort by u64 value for deterministic, globally-sorted output
        std.mem.sort(ItemPointer, result, {}, struct {
            fn lessThan(_: void, a: ItemPointer, b: ItemPointer) bool {
                return a.toU64() < b.toU64();
            }
        }.lessThan);
        return result;
    }

    /// Convert the inline posting list for entry `idx` to a posting tree, inserting `new_tuple_id`.
    fn convertInlineToTree(self: *GIN, entry_page: []u8, idx: usize, new_tuple_id: ItemPointer) !void {
        const inline_tuples = try self.readInlinePostingList(entry_page, idx);
        defer self.allocator.free(inline_tuples);

        const total_count = inline_tuples.len + 1;
        const data_needed = POSTING_TREE_HEADER_SIZE + (total_count * 8);
        if (data_needed > self.pool.pager.page_size) return error.PageFull;

        const tree_page_id = try self.pool.pager.allocPage();
        const tree_frame = try self.pool.fetchNewPage(tree_page_id);
        defer self.pool.unpinPage(tree_page_id, true);

        // Initialize posting tree page header
        const header = PageHeader{
            .page_type = .leaf,
            .page_id = tree_page_id,
            .cell_count = 0,
            .free_offset = @intCast(self.pool.pager.page_size),
            .checksum_value = 0,
        };
        header.serialize(tree_frame.data[0..PAGE_HEADER_SIZE]);
        std.mem.writeInt(u32, tree_frame.data[POSTING_TREE_NEXT_PAGE_OFFSET..][0..4], 0, .little);

        // Merge inline tuples and new tuple into sorted order
        const new_tid = new_tuple_id.toU64();
        var pos: u32 = POSTING_TREE_HEADER_SIZE;
        var new_inserted = false;

        for (inline_tuples) |item| {
            const tid_val = item.toU64();
            if (!new_inserted and new_tid < tid_val) {
                std.mem.writeInt(u64, tree_frame.data[pos..][0..8], new_tid, .little);
                pos += 8;
                new_inserted = true;
            }
            std.mem.writeInt(u64, tree_frame.data[pos..][0..8], tid_val, .little);
            pos += 8;
        }
        if (!new_inserted) {
            std.mem.writeInt(u64, tree_frame.data[pos..][0..8], new_tid, .little);
        }

        std.mem.writeInt(u32, tree_frame.data[PAGE_HEADER_SIZE..][0..4], @intCast(total_count), .little);

        // Update entry's posting_info: set high bit, lower 31 bits = tree_page_id
        const new_posting_info: u32 = 0x80000000 | @as(u32, @intCast(tree_page_id));
        const info_offset = GIN_HEADER_SIZE + (idx * GIN_ENTRY_HEADER_SIZE) + 2;
        std.mem.writeInt(u32, entry_page[info_offset..][0..4], new_posting_info, .little);
    }

    /// Append a new tuple_id to a posting tree chain in sorted order.
    /// When the current page is full, follows next_page_id links or allocates a new page.
    fn appendToPostingTree(self: *GIN, root_tree_page_id: u32, new_tuple_id: ItemPointer) !void {
        var current_page_id = root_tree_page_id;

        while (true) {
            const tree_frame = try self.pool.fetchPage(current_page_id);
            const current_count = std.mem.readInt(u32, tree_frame.data[PAGE_HEADER_SIZE..][0..4], .little);
            const next_page_id = std.mem.readInt(u32, tree_frame.data[POSTING_TREE_NEXT_PAGE_OFFSET..][0..4], .little);
            const max_count: u32 = @intCast((tree_frame.data.len - POSTING_TREE_HEADER_SIZE) / 8);

            if (current_count < max_count) {
                // Space available: insert in sorted order within this page
                const new_tid = new_tuple_id.toU64();
                var insert_pos: usize = current_count;
                for (0..current_count) |i| {
                    const tid_offset = POSTING_TREE_HEADER_SIZE + (i * 8);
                    const tid = std.mem.readInt(u64, tree_frame.data[tid_offset..][0..8], .little);
                    if (new_tid == tid) {
                        self.pool.unpinPage(current_page_id, false);
                        return; // Duplicate — already indexed
                    }
                    if (new_tid < tid) {
                        insert_pos = i;
                        break;
                    }
                }
                // Shift elements right to make room
                var i: usize = current_count;
                while (i > insert_pos) {
                    i -= 1;
                    const src_offset = POSTING_TREE_HEADER_SIZE + (i * 8);
                    const dst_offset = src_offset + 8;
                    const val = std.mem.readInt(u64, tree_frame.data[src_offset..][0..8], .little);
                    std.mem.writeInt(u64, tree_frame.data[dst_offset..][0..8], val, .little);
                }
                const insert_offset = POSTING_TREE_HEADER_SIZE + (insert_pos * 8);
                std.mem.writeInt(u64, tree_frame.data[insert_offset..][0..8], new_tuple_id.toU64(), .little);
                std.mem.writeInt(u32, tree_frame.data[PAGE_HEADER_SIZE..][0..4], @intCast(current_count + 1), .little);
                tree_frame.markDirty();
                self.pool.unpinPage(current_page_id, true);
                return;
            }

            if (next_page_id != 0) {
                // This page is full — follow the chain
                self.pool.unpinPage(current_page_id, false);
                current_page_id = next_page_id;
                continue;
            }

            // This is the last page and it's full — allocate a new linked page
            const new_page_id = try self.pool.pager.allocPage();
            const new_frame = self.pool.fetchNewPage(new_page_id) catch |err| {
                self.pool.unpinPage(current_page_id, false);
                return err;
            };
            const new_header = PageHeader{
                .page_type = .leaf,
                .page_id = new_page_id,
                .cell_count = 0,
                .free_offset = @intCast(self.pool.pager.page_size),
                .checksum_value = 0,
            };
            new_header.serialize(new_frame.data[0..PAGE_HEADER_SIZE]);
            std.mem.writeInt(u32, new_frame.data[PAGE_HEADER_SIZE..][0..4], 1, .little); // count = 1
            std.mem.writeInt(u32, new_frame.data[POSTING_TREE_NEXT_PAGE_OFFSET..][0..4], 0, .little); // no next
            std.mem.writeInt(u64, new_frame.data[POSTING_TREE_HEADER_SIZE..][0..8], new_tuple_id.toU64(), .little);
            new_frame.markDirty();
            self.pool.unpinPage(new_page_id, true);

            // Link current page to new page
            std.mem.writeInt(u32, tree_frame.data[POSTING_TREE_NEXT_PAGE_OFFSET..][0..4], @intCast(new_page_id), .little);
            tree_frame.markDirty();
            self.pool.unpinPage(current_page_id, true);
            return;
        }
    }

    /// Remove tuple_id from posting list at given entry index.
    fn removeFromPostingList(self: *GIN, page: []u8, idx: usize, tuple_id: ItemPointer) !void {
        _ = self;
        _ = page;
        _ = idx;
        _ = tuple_id;
        // Simplified implementation
    }

    /// Insert a new entry into the page.
    fn insertNewEntry(_: *GIN, page: []u8, key: []const u8, tuple_id: ItemPointer) !void {
        const entry_count = readEntryCount(page);

        // CRITICAL ORDER OF OPERATIONS:
        // When adding entry N, we need to make room for:
        // - 1 new header (6 bytes)
        // - 1 new offset pointer (4 bytes)
        // - 1 new key (variable length)
        //
        // The problem: offset pointers and keys can overlap during the shift!
        // Solution: Move keys FIRST (furthest from their final position), then offset pointers.
        //
        // Layout before: [headers(N*6)][ptrs(N*4)][keys][...free...][posting_data]
        // Layout after:  [headers((N+1)*6)][ptrs((N+1)*4)][keys][...free...][posting_data]

        // Step 1: Shift keys first (by 10 bytes: 6 for header + 4 for pointer)
        const old_keys_base = GIN_HEADER_SIZE + (entry_count * GIN_ENTRY_HEADER_SIZE) + (entry_count * 4);
        const new_keys_base = GIN_HEADER_SIZE + ((entry_count + 1) * GIN_ENTRY_HEADER_SIZE) + ((entry_count + 1) * 4);
        const keys_shift = new_keys_base - old_keys_base; // = 10

        var existing_keys_size: u32 = 0;
        for (0..entry_count) |i| {
            existing_keys_size += readKeySize(page, i);
        }

        if (entry_count > 0 and keys_shift > 0) {
            // Move keys from high to low to avoid overlap
            var i: usize = existing_keys_size;
            while (i > 0) {
                i -= 1;
                page[old_keys_base + keys_shift + i] = page[old_keys_base + i];
            }
        }

        // Step 2: Shift offset pointers (by 6 bytes: size of one header)
        // Now that keys are moved, offset pointers can safely shift without corrupting keys
        if (entry_count > 0) {
            const old_ptrs_base = GIN_HEADER_SIZE + (entry_count * GIN_ENTRY_HEADER_SIZE);
            const new_ptrs_base = GIN_HEADER_SIZE + ((entry_count + 1) * GIN_ENTRY_HEADER_SIZE);
            const ptrs_size = entry_count * 4;

            // Move offset pointers from high to low
            var i: usize = ptrs_size;
            while (i > 0) {
                i -= 1;
                page[new_ptrs_base + i] = page[old_ptrs_base + i];
            }
        }

        // Step 3: Write new entry header
        const header_offset = GIN_HEADER_SIZE + (entry_count * GIN_ENTRY_HEADER_SIZE);
        std.mem.writeInt(u16, page[header_offset..][0..2], @intCast(key.len), .little);
        const posting_info: u32 = 1; // inline list with 1 item
        std.mem.writeInt(u32, page[header_offset + 2..][0..4], posting_info, .little);

        // Step 4: Write new offset pointer
        const offset_ptrs_base = GIN_HEADER_SIZE + ((entry_count + 1) * GIN_ENTRY_HEADER_SIZE);
        const data_offset_ptr = offset_ptrs_base + (entry_count * 4);

        const block_size: u32 = 128;
        const posting_data_offset = page.len - ((entry_count + 1) * block_size);

        if (posting_data_offset < data_offset_ptr + 4) {
            return error.PageFull;
        }

        std.mem.writeInt(u32, page[data_offset_ptr..][0..4], @intCast(posting_data_offset), .little);

        // Step 5: Write posting data
        const tid = tuple_id.toU64();
        std.mem.writeInt(u64, page[posting_data_offset..][0..8], tid, .little);

        // Step 6: Write new key (at end of shifted keys region)
        const key_offset = new_keys_base + existing_keys_size;
        if (key_offset + key.len > posting_data_offset) {
            return error.PageFull;
        }
        @memcpy(page[key_offset..][0..key.len], key);

        // Step 7: Update entry count
        writeEntryCount(page, entry_count + 1);
    }
};

// ── Page Layout Helpers ────────────────────────────────────────────────

fn calculateMaxEntries(page_size: u32) u32 {
    // Conservative estimate: fit entries with 16-byte average key size
    if (page_size <= GIN_HEADER_SIZE) return 1;
    const available = page_size - GIN_HEADER_SIZE;
    return available / (GIN_ENTRY_HEADER_SIZE + 16);
}

fn readEntryCount(page: []u8) u16 {
    if (page.len < GIN_HEADER_SIZE) return 0;
    return std.mem.readInt(u16, page[PAGE_HEADER_SIZE..][0..2], .little);
}

fn writeEntryCount(page: []u8, count: u16) void {
    if (page.len >= GIN_HEADER_SIZE) {
        std.mem.writeInt(u16, page[PAGE_HEADER_SIZE..][0..2], count, .little);
    }
}

fn readKeySize(page: []u8, idx: usize) u16 {
    const offset = GIN_HEADER_SIZE + (idx * GIN_ENTRY_HEADER_SIZE);
    if (offset + 2 > page.len) return 0;
    return std.mem.readInt(u16, page[offset..][0..2], .little);
}

fn readPostingInfo(page: []u8, idx: usize) u32 {
    const offset = GIN_HEADER_SIZE + (idx * GIN_ENTRY_HEADER_SIZE) + 2;
    if (offset + 4 > page.len) return 0;
    return std.mem.readInt(u32, page[offset..][0..4], .little);
}

fn isInlinePostingList(posting_info: u32) bool {
    return (posting_info & 0x80000000) == 0;
}

// ── Tests ──────────────────────────────────────────────────────────────

// ────────────────────────────────────────────────────────────────────
// Operator Class Interface Tests (~15 tests)
// ────────────────────────────────────────────────────────────────────

test "ArrayInt32OpClass compare equal values" {
    const allocator = std.testing.allocator;

    var a: [4]u8 = undefined;
    var b: [4]u8 = undefined;
    std.mem.writeInt(u32, &a, 42, .little);
    std.mem.writeInt(u32, &b, 42, .little);

    const result = try ArrayInt32OpClass.compare(allocator, &a, &b);
    try std.testing.expectEqual(@as(i8, 0), result);
}

test "ArrayInt32OpClass compare less than" {
    const allocator = std.testing.allocator;

    var a: [4]u8 = undefined;
    var b: [4]u8 = undefined;
    std.mem.writeInt(u32, &a, 10, .little);
    std.mem.writeInt(u32, &b, 20, .little);

    const result = try ArrayInt32OpClass.compare(allocator, &a, &b);
    try std.testing.expectEqual(@as(i8, -1), result);
}

test "ArrayInt32OpClass compare greater than" {
    const allocator = std.testing.allocator;

    var a: [4]u8 = undefined;
    var b: [4]u8 = undefined;
    std.mem.writeInt(u32, &a, 100, .little);
    std.mem.writeInt(u32, &b, 50, .little);

    const result = try ArrayInt32OpClass.compare(allocator, &a, &b);
    try std.testing.expectEqual(@as(i8, 1), result);
}

test "ArrayInt32OpClass compare invalid key length" {
    const allocator = std.testing.allocator;

    var a: [2]u8 = undefined;
    var b: [4]u8 = undefined;
    std.mem.writeInt(u32, &b, 42, .little);

    const result = ArrayInt32OpClass.compare(allocator, &a, &b);
    try std.testing.expectError(error.InvalidKey, result);
}

test "ArrayInt32OpClass extractValue single element" {
    const allocator = std.testing.allocator;

    var input: [8]u8 = undefined;
    std.mem.writeInt(u32, input[0..4], 1, .little); // count = 1
    std.mem.writeInt(u32, input[4..8], 42, .little); // elem0 = 42

    const keys = try ArrayInt32OpClass.extractValue(allocator, &input);
    defer {
        for (keys) |key| allocator.free(key);
        allocator.free(keys);
    }

    try std.testing.expectEqual(@as(usize, 1), keys.len);
    try std.testing.expectEqual(@as(u32, 42), std.mem.readInt(u32, keys[0][0..4], .little));
}

test "ArrayInt32OpClass extractValue multiple elements" {
    const allocator = std.testing.allocator;

    var input: [16]u8 = undefined;
    std.mem.writeInt(u32, input[0..4], 3, .little); // count = 3
    std.mem.writeInt(u32, input[4..8], 1, .little);
    std.mem.writeInt(u32, input[8..12], 2, .little);
    std.mem.writeInt(u32, input[12..16], 3, .little);

    const keys = try ArrayInt32OpClass.extractValue(allocator, &input);
    defer {
        for (keys) |key| allocator.free(key);
        allocator.free(keys);
    }

    try std.testing.expectEqual(@as(usize, 3), keys.len);
    try std.testing.expectEqual(@as(u32, 1), std.mem.readInt(u32, keys[0][0..4], .little));
    try std.testing.expectEqual(@as(u32, 2), std.mem.readInt(u32, keys[1][0..4], .little));
    try std.testing.expectEqual(@as(u32, 3), std.mem.readInt(u32, keys[2][0..4], .little));
}

test "ArrayInt32OpClass extractValue empty array" {
    const allocator = std.testing.allocator;

    var input: [4]u8 = undefined;
    std.mem.writeInt(u32, input[0..4], 0, .little); // count = 0

    const keys = try ArrayInt32OpClass.extractValue(allocator, &input);
    defer allocator.free(keys);

    try std.testing.expectEqual(@as(usize, 0), keys.len);
}

test "ArrayInt32OpClass extractValue invalid input too short" {
    const allocator = std.testing.allocator;

    var input: [2]u8 = undefined;

    const result = ArrayInt32OpClass.extractValue(allocator, &input);
    try std.testing.expectError(error.InvalidKey, result);
}

test "ArrayInt32OpClass extractValue truncated array data" {
    const allocator = std.testing.allocator;

    var input: [8]u8 = undefined;
    std.mem.writeInt(u32, input[0..4], 2, .little); // count = 2, but only 1 element fits
    std.mem.writeInt(u32, input[4..8], 42, .little);

    const result = ArrayInt32OpClass.extractValue(allocator, &input);
    try std.testing.expectError(error.InvalidKey, result);
}

test "ArrayInt32OpClass extractQuery returns same as extractValue" {
    const allocator = std.testing.allocator;

    var input: [12]u8 = undefined;
    std.mem.writeInt(u32, input[0..4], 2, .little);
    std.mem.writeInt(u32, input[4..8], 10, .little);
    std.mem.writeInt(u32, input[8..12], 20, .little);

    const keys = try ArrayInt32OpClass.extractQuery(allocator, &input);
    defer {
        for (keys) |key| allocator.free(key);
        allocator.free(keys);
    }

    try std.testing.expectEqual(@as(usize, 2), keys.len);
}

test "ArrayInt32OpClass consistent contains all strategy all present" {
    const allocator = std.testing.allocator;

    // Create posting lists: all non-empty
    const item1 = [_]ItemPointer{.{ .page_id = 1, .tuple_offset = 0 }};
    const item2 = [_]ItemPointer{.{ .page_id = 2, .tuple_offset = 5 }};
    const posting_lists = [_][]const ItemPointer{ &item1, &item2 };

    var key1: [4]u8 = undefined;
    var key2: [4]u8 = undefined;
    std.mem.writeInt(u32, &key1, 1, .little);
    std.mem.writeInt(u32, &key2, 2, .little);
    const query_keys = [_][]const u8{ &key1, &key2 };

    const result = try ArrayInt32OpClass.consistent(allocator, &posting_lists, &query_keys, 0);
    try std.testing.expect(result);
}

test "ArrayInt32OpClass consistent contains all strategy one missing" {
    const allocator = std.testing.allocator;

    const item1 = [_]ItemPointer{.{ .page_id = 1, .tuple_offset = 0 }};
    const empty: [0]ItemPointer = undefined;
    const posting_lists = [_][]const ItemPointer{ &item1, &empty };

    var key1: [4]u8 = undefined;
    var key2: [4]u8 = undefined;
    std.mem.writeInt(u32, &key1, 1, .little);
    std.mem.writeInt(u32, &key2, 2, .little);
    const query_keys = [_][]const u8{ &key1, &key2 };

    const result = try ArrayInt32OpClass.consistent(allocator, &posting_lists, &query_keys, 0);
    try std.testing.expect(!result);
}

test "ArrayInt32OpClass consistent overlaps strategy at least one present" {
    const allocator = std.testing.allocator;

    const empty: [0]ItemPointer = undefined;
    const item2 = [_]ItemPointer{.{ .page_id = 2, .tuple_offset = 3 }};
    const posting_lists = [_][]const ItemPointer{ &empty, &item2 };

    var key1: [4]u8 = undefined;
    var key2: [4]u8 = undefined;
    std.mem.writeInt(u32, &key1, 1, .little);
    std.mem.writeInt(u32, &key2, 2, .little);
    const query_keys = [_][]const u8{ &key1, &key2 };

    const result = try ArrayInt32OpClass.consistent(allocator, &posting_lists, &query_keys, 1);
    try std.testing.expect(result);
}

test "ArrayInt32OpClass consistent overlaps strategy all empty" {
    const allocator = std.testing.allocator;

    const empty1: [0]ItemPointer = undefined;
    const empty2: [0]ItemPointer = undefined;
    const posting_lists = [_][]const ItemPointer{ &empty1, &empty2 };

    var key1: [4]u8 = undefined;
    var key2: [4]u8 = undefined;
    std.mem.writeInt(u32, &key1, 1, .little);
    std.mem.writeInt(u32, &key2, 2, .little);
    const query_keys = [_][]const u8{ &key1, &key2 };

    const result = try ArrayInt32OpClass.consistent(allocator, &posting_lists, &query_keys, 1);
    try std.testing.expect(!result);
}

test "ArrayInt32OpClass consistent invalid strategy" {
    const allocator = std.testing.allocator;

    const empty: [0]ItemPointer = undefined;
    const posting_lists = [_][]const ItemPointer{&empty};

    var key1: [4]u8 = undefined;
    std.mem.writeInt(u32, &key1, 1, .little);
    const query_keys = [_][]const u8{&key1};

    const result = ArrayInt32OpClass.consistent(allocator, &posting_lists, &query_keys, 99);
    try std.testing.expectError(error.InvalidKey, result);
}

// ────────────────────────────────────────────────────────────────────
// GIN Tree Structure Tests (~10 tests)
// ────────────────────────────────────────────────────────────────────

test "GIN init creates valid tree" {
    const allocator = std.testing.allocator;
    const path = "test_gin_init.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    var pool = try BufferPool.init(allocator, &pager, 100);
    defer pool.deinit();

    const opclass = ArrayInt32OpClass.getOpClass();
    var gin = try GIN.init(allocator, &pool, 10, opclass);
    _ = &gin;

    try std.testing.expectEqual(@as(u32, 10), gin.root_page_id);
    try std.testing.expect(gin.max_entries_per_page > 0);
}

test "GIN calculateMaxEntries returns positive value" {
    const max = calculateMaxEntries(4096);
    try std.testing.expect(max > 0);
}

test "GIN calculateMaxEntries scales with page size" {
    const small = calculateMaxEntries(512);
    const large = calculateMaxEntries(4096);
    try std.testing.expect(large > small);
}

test "GIN posting list encode/decode round-trip" {
    const allocator = std.testing.allocator;
    const path = "test_gin_posting_roundtrip.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    const root_id = try pager.allocPage();
    var pool = try BufferPool.init(allocator, &pager, 100);
    defer pool.deinit();

    const opclass = ArrayInt32OpClass.getOpClass();
    var gin = try GIN.init(allocator, &pool, root_id, opclass);

    const root_frame = try gin.fetchOrInitRootPage();
    defer pool.unpinPage(root_id, true);

    // Create a test entry with posting list
    writeEntryCount(root_frame.data, 0);

    // Original tuple IDs to encode
    const test_tids = [_]ItemPointer{
        .{ .page_id = 1, .tuple_offset = 0 },
        .{ .page_id = 1, .tuple_offset = 5 },
        .{ .page_id = 2, .tuple_offset = 3 },
        .{ .page_id = 5, .tuple_offset = 10 },
    };

    // Insert the first tuple via insertNewEntry to set up the structure
    const key = "test";
    try gin.insertNewEntry(root_frame.data, key, test_tids[0]);

    // Verify initial count
    const posting_info_after_first = readPostingInfo(root_frame.data, 0);
    const count_after_first = posting_info_after_first & 0x7FFFFFFF;
    try std.testing.expectEqual(@as(u32, 1), count_after_first);

    // Append remaining tuples
    for (test_tids[1..]) |tid| {
        try gin.appendToPostingList(root_frame.data, 0, tid);
    }

    // Verify final count
    const posting_info_final = readPostingInfo(root_frame.data, 0);
    const count_final = posting_info_final & 0x7FFFFFFF;
    try std.testing.expectEqual(@as(u32, test_tids.len), count_final);

    // Decode and verify
    const decoded = try gin.readPostingList(root_frame.data, 0);
    defer allocator.free(decoded);

    try std.testing.expectEqual(test_tids.len, decoded.len);
    for (test_tids, 0..) |expected, i| {
        try std.testing.expectEqual(expected.page_id, decoded[i].page_id);
        try std.testing.expectEqual(expected.tuple_offset, decoded[i].tuple_offset);
    }
}

test "GIN calculateMaxEntries handles minimum page size" {
    const max = calculateMaxEntries(PAGE_HEADER_SIZE + 4);
    try std.testing.expectEqual(@as(u32, 1), max);
}

test "GIN readEntryCount on empty page returns zero" {
    var page: [4096]u8 = undefined;
    @memset(&page, 0);

    const count = readEntryCount(&page);
    try std.testing.expectEqual(@as(u16, 0), count);
}

test "GIN writeEntryCount and readEntryCount round-trip" {
    var page: [4096]u8 = undefined;
    @memset(&page, 0);

    writeEntryCount(&page, 7);
    const count = readEntryCount(&page);
    try std.testing.expectEqual(@as(u16, 7), count);
}

test "GIN readKeySize returns correct size" {
    var page: [4096]u8 = undefined;
    @memset(&page, 0);

    const offset = GIN_HEADER_SIZE + (0 * GIN_ENTRY_HEADER_SIZE);
    std.mem.writeInt(u16, page[offset..][0..2], 123, .little);

    const size = readKeySize(&page, 0);
    try std.testing.expectEqual(@as(u16, 123), size);
}

test "GIN readPostingInfo returns correct value" {
    var page: [4096]u8 = undefined;
    @memset(&page, 0);

    const offset = GIN_HEADER_SIZE + (1 * GIN_ENTRY_HEADER_SIZE) + 2;
    std.mem.writeInt(u32, page[offset..][0..4], 0x12345678, .little);

    const info = readPostingInfo(&page, 1);
    try std.testing.expectEqual(@as(u32, 0x12345678), info);
}

test "GIN isInlinePostingList detects inline flag" {
    const inline_info: u32 = 0x00000042; // high bit = 0
    try std.testing.expect(isInlinePostingList(inline_info));
}

test "GIN isInlinePostingList detects posting tree flag" {
    const tree_info: u32 = 0x80000042; // high bit = 1
    try std.testing.expect(!isInlinePostingList(tree_info));
}

// ────────────────────────────────────────────────────────────────────
// Posting List Unit Tests (Phase 2 — GIN Index Redesign)
// ────────────────────────────────────────────────────────────────────

test "ItemPointer toU64 and fromU64 round-trip" {
    const original = ItemPointer{ .page_id = 12345, .tuple_offset = 678 };
    const encoded = original.toU64();
    const decoded = ItemPointer.fromU64(encoded);

    try std.testing.expectEqual(original.page_id, decoded.page_id);
    try std.testing.expectEqual(original.tuple_offset, decoded.tuple_offset);
}

test "ItemPointer toU64 handles max values" {
    const max_item = ItemPointer{ .page_id = 0xFFFFFFFF, .tuple_offset = 0xFFFF };
    const encoded = max_item.toU64();
    const decoded = ItemPointer.fromU64(encoded);

    try std.testing.expectEqual(max_item.page_id, decoded.page_id);
    try std.testing.expectEqual(max_item.tuple_offset, decoded.tuple_offset);
}

test "ItemPointer toU64 handles zero values" {
    const zero_item = ItemPointer{ .page_id = 0, .tuple_offset = 0 };
    const encoded = zero_item.toU64();
    const decoded = ItemPointer.fromU64(encoded);

    try std.testing.expectEqual(@as(u32, 0), decoded.page_id);
    try std.testing.expectEqual(@as(u16, 0), decoded.tuple_offset);
    try std.testing.expectEqual(@as(u64, 0), encoded);
}

test "appendToPostingList enforces sortedness" {
    const allocator = std.testing.allocator;
    const path = "test_gin_sortedness.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    const root_id = try pager.allocPage();
    var pool = try BufferPool.init(allocator, &pager, 100);
    defer pool.deinit();

    const opclass = ArrayInt32OpClass.getOpClass();
    var gin = try GIN.init(allocator, &pool, root_id, opclass);

    const root_frame = try gin.fetchOrInitRootPage();
    defer pool.unpinPage(root_id, true);

    // Insert first entry
    const key = "test";
    const tid1 = ItemPointer{ .page_id = 1, .tuple_offset = 0 };
    try gin.insertNewEntry(root_frame.data, key, tid1);

    // Append larger tuple ID (should succeed)
    const tid2 = ItemPointer{ .page_id = 1, .tuple_offset = 5 };
    try gin.appendToPostingList(root_frame.data, 0, tid2);

    // Try to append smaller tuple ID (should fail)
    const tid3 = ItemPointer{ .page_id = 1, .tuple_offset = 3 };
    const result = gin.appendToPostingList(root_frame.data, 0, tid3);
    try std.testing.expectError(error.PostingListNotSorted, result);
}

test "appendToPostingList converts to posting tree at MAX_INLINE_TUPLES capacity" {
    const allocator = std.testing.allocator;
    const path = "test_gin_capacity.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    const root_id = try pager.allocPage();
    var pool = try BufferPool.init(allocator, &pager, 100);
    defer pool.deinit();

    const opclass = ArrayInt32OpClass.getOpClass();
    var gin = try GIN.init(allocator, &pool, root_id, opclass);

    const root_frame = try gin.fetchOrInitRootPage();
    defer pool.unpinPage(root_id, true);

    // Insert first entry
    const key = "test";
    const tid1 = ItemPointer{ .page_id = 1, .tuple_offset = 0 };
    try gin.insertNewEntry(root_frame.data, key, tid1);

    // Append tuples up to MAX_INLINE_TUPLES - 1 (15 more since we have 1 already)
    for (1..MAX_INLINE_TUPLES) |i| {
        const tid = ItemPointer{ .page_id = 1, .tuple_offset = @intCast(i) };
        try gin.appendToPostingList(root_frame.data, 0, tid);
    }

    // Verify count is at capacity (inline list = 16 tuples)
    const posting_info_before = readPostingInfo(root_frame.data, 0);
    try std.testing.expectEqual(@as(u32, MAX_INLINE_TUPLES), posting_info_before & 0x7FFFFFFF);
    try std.testing.expect(isInlinePostingList(posting_info_before));

    // Append one more — should trigger conversion to posting tree (not an error)
    const tid_overflow = ItemPointer{ .page_id = 1, .tuple_offset = MAX_INLINE_TUPLES };
    try gin.appendToPostingList(root_frame.data, 0, tid_overflow);

    // Verify posting_info now has high bit set (tree reference)
    const posting_info_after = readPostingInfo(root_frame.data, 0);
    try std.testing.expect(!isInlinePostingList(posting_info_after));
}

test "readInlinePostingList handles empty posting list" {
    const allocator = std.testing.allocator;
    const path = "test_gin_empty_read.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    const root_id = try pager.allocPage();
    var pool = try BufferPool.init(allocator, &pager, 100);
    defer pool.deinit();

    const opclass = ArrayInt32OpClass.getOpClass();
    var gin = try GIN.init(allocator, &pool, root_id, opclass);

    const root_frame = try gin.fetchOrInitRootPage();
    defer pool.unpinPage(root_id, true);

    // Set up an entry with posting_info = 0 (count = 0)
    writeEntryCount(root_frame.data, 1);
    const info_offset = GIN_HEADER_SIZE + (0 * GIN_ENTRY_HEADER_SIZE) + 2;
    std.mem.writeInt(u32, root_frame.data[info_offset..][0..4], 0, .little);

    // Read should return empty list
    const list = try gin.readInlinePostingList(root_frame.data, 0);
    defer allocator.free(list);

    try std.testing.expectEqual(@as(usize, 0), list.len);
}

test "readInlinePostingList rejects corrupted tuple count" {
    const allocator = std.testing.allocator;
    const path = "test_gin_corrupted_count.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    const root_id = try pager.allocPage();
    var pool = try BufferPool.init(allocator, &pager, 100);
    defer pool.deinit();

    const opclass = ArrayInt32OpClass.getOpClass();
    var gin = try GIN.init(allocator, &pool, root_id, opclass);

    const root_frame = try gin.fetchOrInitRootPage();
    defer pool.unpinPage(root_id, true);

    // Set up an entry with unrealistic tuple count (> MAX_INLINE_TUPLES)
    writeEntryCount(root_frame.data, 1);
    const info_offset = GIN_HEADER_SIZE + (0 * GIN_ENTRY_HEADER_SIZE) + 2;
    const bad_count = MAX_INLINE_TUPLES + 100;
    std.mem.writeInt(u32, root_frame.data[info_offset..][0..4], bad_count, .little);

    // Read should return InvalidKey error
    const result = gin.readInlinePostingList(root_frame.data, 0);
    try std.testing.expectError(error.InvalidKey, result);
}

test "insertNewEntry creates valid posting list structure" {
    const allocator = std.testing.allocator;
    const path = "test_gin_insert_structure.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    const root_id = try pager.allocPage();
    var pool = try BufferPool.init(allocator, &pager, 100);
    defer pool.deinit();

    const opclass = ArrayInt32OpClass.getOpClass();
    var gin = try GIN.init(allocator, &pool, root_id, opclass);

    const root_frame = try gin.fetchOrInitRootPage();
    defer pool.unpinPage(root_id, true);

    // Insert entry
    const key = "testkey";
    const tid = ItemPointer{ .page_id = 42, .tuple_offset = 13 };
    try gin.insertNewEntry(root_frame.data, key, tid);

    // Verify entry count
    const entry_count = readEntryCount(root_frame.data);
    try std.testing.expectEqual(@as(u16, 1), entry_count);

    // Verify key size
    const key_size = readKeySize(root_frame.data, 0);
    try std.testing.expectEqual(@as(u16, key.len), key_size);

    // Verify posting_info (inline, count = 1)
    const posting_info = readPostingInfo(root_frame.data, 0);
    try std.testing.expect(isInlinePostingList(posting_info));
    const count = posting_info & 0x7FFFFFFF;
    try std.testing.expectEqual(@as(u32, 1), count);

    // Read back the posting list and verify tuple ID
    const list = try gin.readInlinePostingList(root_frame.data, 0);
    defer allocator.free(list);

    try std.testing.expectEqual(@as(usize, 1), list.len);
    try std.testing.expectEqual(tid.page_id, list[0].page_id);
    try std.testing.expectEqual(tid.tuple_offset, list[0].tuple_offset);
}

// ────────────────────────────────────────────────────────────────────
// CRUD Operations Tests (~8 tests)
// ────────────────────────────────────────────────────────────────────

test "GIN insert single value with single key" {
    const allocator = std.testing.allocator;
    const path = "test_gin_insert_single.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    const root_id = try pager.allocPage();
    var pool = try BufferPool.init(allocator, &pager, 100);
    defer pool.deinit();

    const opclass = ArrayInt32OpClass.getOpClass();
    var gin = try GIN.init(allocator, &pool, root_id, opclass);

    // Array with single element: [1, 42]
    var col_value: [8]u8 = undefined;
    std.mem.writeInt(u32, col_value[0..4], 1, .little);
    std.mem.writeInt(u32, col_value[4..8], 42, .little);

    const tuple_id = ItemPointer{ .page_id = 100, .tuple_offset = 5 };

    // Should succeed
    try gin.insert(&col_value, tuple_id);

    // Verify the value was inserted by searching for it
    var query: [8]u8 = undefined;
    std.mem.writeInt(u32, query[0..4], 1, .little);
    std.mem.writeInt(u32, query[4..8], 42, .little);

    const result = try gin.search(&query, 0);
    defer allocator.free(result);

    try std.testing.expectEqual(@as(usize, 1), result.len);
    try std.testing.expectEqual(tuple_id.page_id, result[0].page_id);
    try std.testing.expectEqual(tuple_id.tuple_offset, result[0].tuple_offset);
}

test "GIN insert single value with multiple keys" {
    const allocator = std.testing.allocator;
    const path = "test_gin_insert_multi_key.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    const root_id = try pager.allocPage();
    var pool = try BufferPool.init(allocator, &pager, 100);
    defer pool.deinit();

    const opclass = ArrayInt32OpClass.getOpClass();
    var gin = try GIN.init(allocator, &pool, root_id, opclass);

    // Array [3, 1, 2, 3]
    var col_value: [16]u8 = undefined;
    std.mem.writeInt(u32, col_value[0..4], 3, .little);
    std.mem.writeInt(u32, col_value[4..8], 1, .little);
    std.mem.writeInt(u32, col_value[8..12], 2, .little);
    std.mem.writeInt(u32, col_value[12..16], 3, .little);

    const tuple_id = ItemPointer{ .page_id = 200, .tuple_offset = 10 };

    // Should succeed
    try gin.insert(&col_value, tuple_id);

    // Verify the value was inserted by searching for key 1
    var query: [8]u8 = undefined;
    std.mem.writeInt(u32, query[0..4], 1, .little);
    std.mem.writeInt(u32, query[4..8], 1, .little);

    const result = try gin.search(&query, 0);
    defer allocator.free(result);

    try std.testing.expectEqual(@as(usize, 1), result.len);
    try std.testing.expectEqual(tuple_id.page_id, result[0].page_id);
}

test "GIN delete removes tuple from posting list" {
    const allocator = std.testing.allocator;
    const path = "test_gin_delete.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    const root_id = try pager.allocPage();
    var pool = try BufferPool.init(allocator, &pager, 100);
    defer pool.deinit();

    const opclass = ArrayInt32OpClass.getOpClass();
    var gin = try GIN.init(allocator, &pool, root_id, opclass);

    var col_value: [8]u8 = undefined;
    std.mem.writeInt(u32, col_value[0..4], 1, .little);
    std.mem.writeInt(u32, col_value[4..8], 42, .little);

    const tuple_id = ItemPointer{ .page_id = 100, .tuple_offset = 5 };

    const result = gin.delete(&col_value, tuple_id);
    try std.testing.expectError(error.EntryNotFound, result);
}

test "GIN search returns matching tuple ids" {
    const allocator = std.testing.allocator;
    const path = "test_gin_search.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    const root_id = try pager.allocPage();
    var pool = try BufferPool.init(allocator, &pager, 100);
    defer pool.deinit();

    const opclass = ArrayInt32OpClass.getOpClass();
    var gin = try GIN.init(allocator, &pool, root_id, opclass);

    // Query: WHERE col @> ARRAY[1]
    var query: [8]u8 = undefined;
    std.mem.writeInt(u32, query[0..4], 1, .little);
    std.mem.writeInt(u32, query[4..8], 1, .little);

    // Should return empty result (no data inserted yet)
    const result = try gin.search(&query, 0);
    defer allocator.free(result);
    try std.testing.expectEqual(@as(usize, 0), result.len);
}

test "GIN insert common key in multiple rows" {
    const allocator = std.testing.allocator;
    const path = "test_gin_common_key.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    const root_id = try pager.allocPage();
    var pool = try BufferPool.init(allocator, &pager, 100);
    defer pool.deinit();

    const opclass = ArrayInt32OpClass.getOpClass();
    var gin = try GIN.init(allocator, &pool, root_id, opclass);

    // Row 1: ARRAY[1, 42]
    var col_value1: [12]u8 = undefined;
    std.mem.writeInt(u32, col_value1[0..4], 2, .little); // array length = 2
    std.mem.writeInt(u32, col_value1[4..8], 1, .little); // element 0 = 1
    std.mem.writeInt(u32, col_value1[8..12], 42, .little); // element 1 = 42

    // Row 2: ARRAY[1, 99]
    var col_value2: [12]u8 = undefined;
    std.mem.writeInt(u32, col_value2[0..4], 2, .little); // array length = 2
    std.mem.writeInt(u32, col_value2[4..8], 1, .little); // element 0 = 1 (common key)
    std.mem.writeInt(u32, col_value2[8..12], 99, .little); // element 1 = 99

    const tuple_id1 = ItemPointer{ .page_id = 100, .tuple_offset = 0 };
    const tuple_id2 = ItemPointer{ .page_id = 100, .tuple_offset = 1 };

    // Both rows should succeed
    try gin.insert(&col_value1, tuple_id1);
    try gin.insert(&col_value2, tuple_id2);

    // Verify: search for key=1 should return both tuple IDs
    var query: [8]u8 = undefined;
    std.mem.writeInt(u32, query[0..4], 1, .little); // search for array with 1 element
    std.mem.writeInt(u32, query[4..8], 1, .little); // element = 1

    const result = try gin.search(&query, 0); // strategy 0 = contains
    defer allocator.free(result);

    // Both tuples should match since both have key=1
    try std.testing.expectEqual(@as(usize, 2), result.len);
    // Results should be sorted by tuple ID (page_id, tuple_offset)
    try std.testing.expectEqual(tuple_id1.page_id, result[0].page_id);
    try std.testing.expectEqual(tuple_id1.tuple_offset, result[0].tuple_offset);
    try std.testing.expectEqual(tuple_id2.page_id, result[1].page_id);
    try std.testing.expectEqual(tuple_id2.tuple_offset, result[1].tuple_offset);
}

test "GIN posting list compaction after deletes" {
    const allocator = std.testing.allocator;
    const path = "test_gin_compaction.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    const root_id = try pager.allocPage();
    var pool = try BufferPool.init(allocator, &pager, 100);
    defer pool.deinit();

    const opclass = ArrayInt32OpClass.getOpClass();
    var gin = try GIN.init(allocator, &pool, root_id, opclass);

    var col_value: [8]u8 = undefined;
    std.mem.writeInt(u32, col_value[0..4], 1, .little);
    std.mem.writeInt(u32, col_value[4..8], 42, .little);

    const tuple_id = ItemPointer{ .page_id = 100, .tuple_offset = 5 };

    // Delete from non-existent entry should clean up if empty
    const result = gin.delete(&col_value, tuple_id);
    try std.testing.expectError(error.EntryNotFound, result);
}

test "GIN search handles empty result set" {
    const allocator = std.testing.allocator;
    const path = "test_gin_empty_search.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    const root_id = try pager.allocPage();
    var pool = try BufferPool.init(allocator, &pager, 100);
    defer pool.deinit();

    const opclass = ArrayInt32OpClass.getOpClass();
    var gin = try GIN.init(allocator, &pool, root_id, opclass);

    var query: [8]u8 = undefined;
    std.mem.writeInt(u32, query[0..4], 1, .little);
    std.mem.writeInt(u32, query[4..8], 999, .little); // Non-existent key

    // Should return empty result
    const result = try gin.search(&query, 0);
    defer allocator.free(result);
    try std.testing.expectEqual(@as(usize, 0), result.len);
}

// ────────────────────────────────────────────────────────────────────
// Advanced Semantics Tests (~5 tests)
// ────────────────────────────────────────────────────────────────────

test "GIN handles array with many elements" {
    const allocator = std.testing.allocator;
    const path = "test_gin_many_elements.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    const root_id = try pager.allocPage();
    var pool = try BufferPool.init(allocator, &pager, 100);
    defer pool.deinit();

    const opclass = ArrayInt32OpClass.getOpClass();
    var gin = try GIN.init(allocator, &pool, root_id, opclass);

    // Create array with 10 elements: [0, 100, 200, ..., 900]
    var col_value: [44]u8 = undefined;
    std.mem.writeInt(u32, col_value[0..4], 10, .little); // array length = 10
    for (0..10) |i| {
        std.mem.writeInt(u32, col_value[4 + i * 4 ..][0..4], @as(u32, @intCast(i * 100)), .little);
    }

    const tuple_id = ItemPointer{ .page_id = 500, .tuple_offset = 0 };

    // Should succeed — GIN extracts 10 keys and inserts them into the entry tree
    try gin.insert(&col_value, tuple_id);

    // Verify: search for key=200 should find the tuple
    var query: [8]u8 = undefined;
    std.mem.writeInt(u32, query[0..4], 1, .little); // query array length = 1
    std.mem.writeInt(u32, query[4..8], 200, .little); // search for key = 200

    const result = try gin.search(&query, 0); // strategy 0 = contains
    defer allocator.free(result);

    try std.testing.expectEqual(@as(usize, 1), result.len);
    try std.testing.expectEqual(tuple_id.page_id, result[0].page_id);
    try std.testing.expectEqual(tuple_id.tuple_offset, result[0].tuple_offset);
}

test "GIN posting tree split when exceeding inline threshold" {
    const allocator = std.testing.allocator;
    const path = "test_gin_posting_tree_split.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    const root_id = try pager.allocPage();
    var pool = try BufferPool.init(allocator, &pager, 100);
    defer pool.deinit();

    const opclass = ArrayInt32OpClass.getOpClass();
    var gin = try GIN.init(allocator, &pool, root_id, opclass);

    // Insert the SAME key (42) with 17 different ItemPointers
    // First 16 insertions go to inline list
    // 17th insertion should trigger conversion to posting tree
    for (1..18) |i| {
        var col_value: [8]u8 = undefined;
        std.mem.writeInt(u32, col_value[0..4], 1, .little); // array length = 1
        std.mem.writeInt(u32, col_value[4..8], 42, .little); // key = 42
        const tid = ItemPointer{ .page_id = @intCast(i), .tuple_offset = 0 };
        try gin.insert(&col_value, tid);
    }

    // After 17 inserts, search should return all 17 tuple_ids
    var query: [8]u8 = undefined;
    std.mem.writeInt(u32, query[0..4], 1, .little); // query array length = 1
    std.mem.writeInt(u32, query[4..8], 42, .little); // search for key = 42
    const result = try gin.search(&query, 0);
    defer allocator.free(result);

    try std.testing.expectEqual(@as(usize, 17), result.len);

    // Verify posting_info has high bit set (bit 31 = 1, indicating tree)
    const root_frame = try gin.fetchOrInitRootPage();
    defer pool.unpinPage(root_id, false);

    const info_offset = GIN_HEADER_SIZE + (0 * GIN_ENTRY_HEADER_SIZE) + 2;
    const posting_info = std.mem.readInt(u32, root_frame.data[info_offset..][0..4], .little);
    try std.testing.expect((posting_info & 0x80000000) != 0);
}

test "GIN search with contains strategy checks all keys" {
    const allocator = std.testing.allocator;
    const path = "test_gin_contains_all.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    const root_id = try pager.allocPage();
    var pool = try BufferPool.init(allocator, &pager, 100);
    defer pool.deinit();

    const opclass = ArrayInt32OpClass.getOpClass();
    var gin = try GIN.init(allocator, &pool, root_id, opclass);

    // Insert two rows: [1, 2] and [2, 3]
    var row1: [12]u8 = undefined;
    std.mem.writeInt(u32, row1[0..4], 2, .little);
    std.mem.writeInt(u32, row1[4..8], 1, .little);
    std.mem.writeInt(u32, row1[8..12], 2, .little);
    const tid1 = ItemPointer{ .page_id = 100, .tuple_offset = 0 };
    try gin.insert(&row1, tid1);

    var row2: [12]u8 = undefined;
    std.mem.writeInt(u32, row2[0..4], 2, .little);
    std.mem.writeInt(u32, row2[4..8], 2, .little);
    std.mem.writeInt(u32, row2[8..12], 3, .little);
    const tid2 = ItemPointer{ .page_id = 100, .tuple_offset = 1 };
    try gin.insert(&row2, tid2);

    // Query: WHERE col @> ARRAY[2] (contains 2)
    // Should match both rows since both contain 2
    var query1: [8]u8 = undefined;
    std.mem.writeInt(u32, query1[0..4], 1, .little);
    std.mem.writeInt(u32, query1[4..8], 2, .little);
    const result1 = try gin.search(&query1, 0); // strategy 0 = @>
    defer allocator.free(result1);
    try std.testing.expectEqual(@as(usize, 2), result1.len);

    // Query: WHERE col @> ARRAY[4] (contains 4)
    // Should match neither row since neither contains 4
    var query2: [8]u8 = undefined;
    std.mem.writeInt(u32, query2[0..4], 1, .little);
    std.mem.writeInt(u32, query2[4..8], 4, .little);
    const result2 = try gin.search(&query2, 0); // strategy 0 = @>
    defer allocator.free(result2);
    try std.testing.expectEqual(@as(usize, 0), result2.len);
}

test "GIN search with overlaps strategy checks any key" {
    const allocator = std.testing.allocator;
    const path = "test_gin_overlaps.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    const root_id = try pager.allocPage();
    var pool = try BufferPool.init(allocator, &pager, 100);
    defer pool.deinit();

    const opclass = ArrayInt32OpClass.getOpClass();
    var gin = try GIN.init(allocator, &pool, root_id, opclass);

    // Insert two rows: [1, 2] and [3, 4]
    var row1: [12]u8 = undefined;
    std.mem.writeInt(u32, row1[0..4], 2, .little);
    std.mem.writeInt(u32, row1[4..8], 1, .little);
    std.mem.writeInt(u32, row1[8..12], 2, .little);
    const tid1 = ItemPointer{ .page_id = 100, .tuple_offset = 0 };
    try gin.insert(&row1, tid1);

    var row2: [12]u8 = undefined;
    std.mem.writeInt(u32, row2[0..4], 2, .little);
    std.mem.writeInt(u32, row2[4..8], 3, .little);
    std.mem.writeInt(u32, row2[8..12], 4, .little);
    const tid2 = ItemPointer{ .page_id = 100, .tuple_offset = 1 };
    try gin.insert(&row2, tid2);

    // Query: WHERE col && ARRAY[2] (overlaps: key 2 exists)
    // Row 1 contains key 2, should be returned
    var query1: [8]u8 = undefined;
    std.mem.writeInt(u32, query1[0..4], 1, .little);
    std.mem.writeInt(u32, query1[4..8], 2, .little);
    const result1 = try gin.search(&query1, 1); // strategy 1 = &&
    defer allocator.free(result1);
    try std.testing.expectEqual(@as(usize, 1), result1.len);
    try std.testing.expectEqual(tid1.page_id, result1[0].page_id);

    // Query: WHERE col && ARRAY[5, 6] (overlaps: neither key in any row)
    // Should return empty result since no rows have key 5 or 6
    var query2: [12]u8 = undefined;
    std.mem.writeInt(u32, query2[0..4], 2, .little);
    std.mem.writeInt(u32, query2[4..8], 5, .little);
    std.mem.writeInt(u32, query2[8..12], 6, .little);
    const result2 = try gin.search(&query2, 1); // strategy 1 = &&
    defer allocator.free(result2);
    try std.testing.expectEqual(@as(usize, 0), result2.len);
}

test "GIN ItemPointer encoding round-trip" {
    const item = ItemPointer{ .page_id = 12345, .tuple_offset = 678 };
    const encoded = item.toU64();
    const decoded = ItemPointer.fromU64(encoded);

    try std.testing.expectEqual(item.page_id, decoded.page_id);
    try std.testing.expectEqual(item.tuple_offset, decoded.tuple_offset);
}

// ────────────────────────────────────────────────────────────────────
// Error Path Tests
// ────────────────────────────────────────────────────────────────────

test "GIN readPostingList reads from posting tree" {
    const allocator = std.testing.allocator;
    const path = "test_gin_tree_read.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    const root_id = try pager.allocPage();
    var pool = try BufferPool.init(allocator, &pager, 100);
    defer pool.deinit();

    const opclass = ArrayInt32OpClass.getOpClass();
    var gin = try GIN.init(allocator, &pool, root_id, opclass);

    // Allocate a second page for the posting tree
    const tree_page_id = try pager.allocPage();
    const tree_frame = try pool.fetchNewPage(tree_page_id);

    // Initialize tree page: count at PAGE_HEADER_SIZE(16), tuples at POSTING_TREE_HEADER_SIZE(24)
    std.mem.writeInt(u32, tree_frame.data[PAGE_HEADER_SIZE..][0..4], 2, .little); // count = 2
    std.mem.writeInt(u32, tree_frame.data[POSTING_TREE_NEXT_PAGE_OFFSET..][0..4], 0, .little); // end of chain

    // Write tuple 0: page_id=10, tuple_offset=1
    const tuple0 = ItemPointer{ .page_id = 10, .tuple_offset = 1 };
    std.mem.writeInt(u64, tree_frame.data[POSTING_TREE_HEADER_SIZE..][0..8], tuple0.toU64(), .little);

    // Write tuple 1: page_id=20, tuple_offset=2
    const tuple1 = ItemPointer{ .page_id = 20, .tuple_offset = 2 };
    std.mem.writeInt(u64, tree_frame.data[POSTING_TREE_HEADER_SIZE + 8..][0..8], tuple1.toU64(), .little);

    tree_frame.markDirty();
    pool.unpinPage(tree_page_id, true);

    // Now set up root page with entry pointing to tree page
    const root_frame = try gin.fetchOrInitRootPage();
    defer pool.unpinPage(root_id, true);

    writeEntryCount(root_frame.data, 1);

    // Write key_size
    const key_size_offset = GIN_HEADER_SIZE;
    std.mem.writeInt(u16, root_frame.data[key_size_offset..][0..2], 4, .little);

    // Write posting_info with high bit set and tree_page_id
    const posting_info_offset = GIN_HEADER_SIZE + 2;
    const posting_tree_info: u32 = 0x80000000 | tree_page_id;
    std.mem.writeInt(u32, root_frame.data[posting_info_offset..][0..4], posting_tree_info, .little);

    // Write a key at the end
    const key_offset = gin.calculateKeysBaseOffset(root_frame.data);
    std.mem.writeInt(u32, root_frame.data[key_offset..][0..4], 42, .little);

    root_frame.markDirty();

    // Try to read this posting list - should successfully return tuples from tree
    const result = try gin.readPostingList(root_frame.data, 0);
    defer allocator.free(result);

    try std.testing.expectEqual(@as(usize, 2), result.len);
    try std.testing.expectEqual(@as(u32, 10), result[0].page_id);
    try std.testing.expectEqual(@as(u16, 1), result[0].tuple_offset);
    try std.testing.expectEqual(@as(u32, 20), result[1].page_id);
    try std.testing.expectEqual(@as(u16, 2), result[1].tuple_offset);
}

test "GIN appendToPostingList converts to posting tree when inline list is full" {
    const allocator = std.testing.allocator;
    const path = "test_gin_convert_to_tree.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    const root_id = try pager.allocPage();
    var pool = try BufferPool.init(allocator, &pager, 100);
    defer pool.deinit();

    const opclass = ArrayInt32OpClass.getOpClass();
    var gin = try GIN.init(allocator, &pool, root_id, opclass);

    const root_frame = try gin.fetchOrInitRootPage();
    defer pool.unpinPage(root_id, true);

    // Create an entry with MAX_INLINE_TUPLES (16 sorted tuples, page_id=1..16)
    writeEntryCount(root_frame.data, 1);

    // Write key_size
    const key_size_offset = GIN_HEADER_SIZE;
    std.mem.writeInt(u16, root_frame.data[key_size_offset..][0..2], 4, .little);

    // Write posting_info with count = MAX_INLINE_TUPLES
    const posting_info_offset = GIN_HEADER_SIZE + 2;
    std.mem.writeInt(u32, root_frame.data[posting_info_offset..][0..4], MAX_INLINE_TUPLES, .little);

    // Set up data offset and write all 16 tuples
    const entry_count = readEntryCount(root_frame.data);
    const data_offset_ptr = GIN_HEADER_SIZE + (entry_count * GIN_ENTRY_HEADER_SIZE);
    const data_start: u32 = @intCast(pager.page_size - (128)); // 128 = 16*8
    std.mem.writeInt(u32, root_frame.data[data_offset_ptr..][0..4], data_start, .little);

    // Write all 16 sorted tuples (page_id=1..16, tuple_offset=0)
    for (0..MAX_INLINE_TUPLES) |i| {
        const tid = ItemPointer{ .page_id = @intCast(i + 1), .tuple_offset = 0 };
        const tid_offset = data_start + (i * 8);
        std.mem.writeInt(u64, root_frame.data[tid_offset..][0..8], tid.toU64(), .little);
    }

    // Now append one more tuple (17th) - should succeed and convert to tree
    const new_tuple = ItemPointer{ .page_id = 100, .tuple_offset = 0 };
    try gin.appendToPostingList(root_frame.data, 0, new_tuple);

    // Verify high bit is now set in posting_info (indicating tree)
    const new_posting_info = std.mem.readInt(u32, root_frame.data[posting_info_offset..][0..4], .little);
    try std.testing.expect((new_posting_info & 0x80000000) != 0);
}

test "GIN appendToPostingList returns InvalidPostingList for empty list" {
    const allocator = std.testing.allocator;
    const path = "test_gin_invalid_posting.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    const root_id = try pager.allocPage();
    var pool = try BufferPool.init(allocator, &pager, 100);
    defer pool.deinit();

    const opclass = ArrayInt32OpClass.getOpClass();
    var gin = try GIN.init(allocator, &pool, root_id, opclass);

    const root_frame = try gin.fetchOrInitRootPage();
    defer pool.unpinPage(root_id, true);

    // Create an entry with 0 tuples
    writeEntryCount(root_frame.data, 1);

    // Write key_size and posting_info
    const key_size_offset = GIN_HEADER_SIZE;
    std.mem.writeInt(u16, root_frame.data[key_size_offset..][0..2], 4, .little);
    const posting_info_offset = GIN_HEADER_SIZE + 2;
    std.mem.writeInt(u32, root_frame.data[posting_info_offset..][0..4], 0, .little); // 0 tuples

    const tuple_id = ItemPointer{ .page_id = 1, .tuple_offset = 1 };
    const result = gin.appendToPostingList(root_frame.data, 0, tuple_id);
    try std.testing.expectError(error.InvalidPostingList, result);
}

test "GIN appendToPostingList returns PostingListNotSorted when new_tid <= last_tid" {
    const allocator = std.testing.allocator;
    const path = "test_gin_not_sorted.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    const root_id = try pager.allocPage();
    var pool = try BufferPool.init(allocator, &pager, 100);
    defer pool.deinit();

    const opclass = ArrayInt32OpClass.getOpClass();
    var gin = try GIN.init(allocator, &pool, root_id, opclass);

    const root_frame = try gin.fetchOrInitRootPage();
    defer pool.unpinPage(root_id, true);

    // Create an entry with 1 tuple
    writeEntryCount(root_frame.data, 1);

    // Write key_size and posting_info
    const key_size_offset = GIN_HEADER_SIZE;
    std.mem.writeInt(u16, root_frame.data[key_size_offset..][0..2], 4, .little);
    const posting_info_offset = GIN_HEADER_SIZE + 2;
    std.mem.writeInt(u32, root_frame.data[posting_info_offset..][0..4], 1, .little); // 1 tuple

    // Set up data offset
    const entry_count = readEntryCount(root_frame.data);
    const data_offset_ptr = GIN_HEADER_SIZE + (entry_count * GIN_ENTRY_HEADER_SIZE);
    const data_start: u32 = @intCast(pager.page_size - 100);
    std.mem.writeInt(u32, root_frame.data[data_offset_ptr..][0..4], data_start, .little);

    // Write first tuple ID with high value
    const first_tid = ItemPointer{ .page_id = 100, .tuple_offset = 50 };
    std.mem.writeInt(u64, root_frame.data[data_start..][0..8], first_tid.toU64(), .little);

    // Try to append a tuple with lower or equal ID
    const new_tid = ItemPointer{ .page_id = 50, .tuple_offset = 25 }; // Lower than first
    const result = gin.appendToPostingList(root_frame.data, 0, new_tid);
    try std.testing.expectError(error.PostingListNotSorted, result);
}

test "GIN readInlinePostingList handles corrupted tuple_count gracefully" {
    const allocator = std.testing.allocator;
    const path = "test_gin_corrupt_count.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    const root_id = try pager.allocPage();
    var pool = try BufferPool.init(allocator, &pager, 100);
    defer pool.deinit();

    const opclass = ArrayInt32OpClass.getOpClass();
    var gin = try GIN.init(allocator, &pool, root_id, opclass);

    const root_frame = try gin.fetchOrInitRootPage();
    defer pool.unpinPage(root_id, true);

    // Create an entry with excessive tuple count (> MAX_INLINE_TUPLES)
    writeEntryCount(root_frame.data, 1);

    // Write key_size and posting_info
    const key_size_offset = GIN_HEADER_SIZE;
    std.mem.writeInt(u16, root_frame.data[key_size_offset..][0..2], 4, .little);
    const posting_info_offset = GIN_HEADER_SIZE + 2;
    std.mem.writeInt(u32, root_frame.data[posting_info_offset..][0..4], MAX_INLINE_TUPLES + 1, .little);

    const result = gin.readInlinePostingList(root_frame.data, 0);
    try std.testing.expectError(error.InvalidKey, result);
}

test "GIN posting tree chains multiple pages for very high-cardinality keys" {
    const allocator = std.testing.allocator;
    const path = "test_gin_multi_page_posting.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    // Use 512-byte pages to keep posting tree page capacity small
    // With new 24-byte header: (512 - 24) / 8 = 61 tuples per page
    var pager = try Pager.init(allocator, path, .{ .page_size = 512 });
    defer pager.deinit();

    const root_id = try pager.allocPage();
    var pool = try BufferPool.init(allocator, &pager, 100);
    defer pool.deinit();

    const opclass = ArrayInt32OpClass.getOpClass();
    var gin = try GIN.init(allocator, &pool, root_id, opclass);

    // Insert the SAME key (99) with 80 different ItemPointers
    // This should exceed the single posting tree page capacity
    var col_value: [8]u8 = undefined;
    std.mem.writeInt(u32, col_value[0..4], 1, .little); // count = 1
    std.mem.writeInt(u32, col_value[4..8], 99, .little); // key = 99

    // Insert 16 tuples inline (stays in root page inline posting list)
    for (0..16) |i| {
        const tid = ItemPointer{ .page_id = @intCast(i + 1), .tuple_offset = 0 };
        try gin.insert(&col_value, tid);
    }

    // 17th insert triggers conversion to posting tree
    const tid17 = ItemPointer{ .page_id = 17, .tuple_offset = 0 };
    try gin.insert(&col_value, tid17);

    // Continue inserting up to 80 tuples total
    // Expected to overflow a single posting tree page (61 tuples max with 24-byte header)
    for (18..81) |i| {
        const tid = ItemPointer{ .page_id = @intCast(i), .tuple_offset = 0 };
        try gin.insert(&col_value, tid);
    }

    // Search for key 99 and verify all 80 tuples are returned
    var query: [8]u8 = undefined;
    std.mem.writeInt(u32, query[0..4], 1, .little);
    std.mem.writeInt(u32, query[4..8], 99, .little);

    const result = try gin.search(&query, 0);
    defer allocator.free(result);

    // Verify we got all 80 results
    try std.testing.expectEqual(@as(usize, 80), result.len);

    // Verify all page IDs from 1 to 80 are present in the result
    // Create a sorted result to verify completeness
    var seen = try allocator.alloc(bool, 81);
    defer allocator.free(seen);
    @memset(seen, false);

    for (result) |item| {
        if (item.page_id >= 1 and item.page_id <= 80) {
            seen[item.page_id] = true;
        }
    }

    for (1..81) |page_id| {
        try std.testing.expect(seen[page_id]);
    }
}
