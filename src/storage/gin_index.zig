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

/// Lexicographic byte comparison shared by opclasses whose GIN entry-tree
/// ordering only needs a consistent total order (not a semantically
/// meaningful one) — plain memcmp-with-length-tiebreak, type-agnostic across
/// polymorphic key encodings.
fn lexCompareKeys(a: []const u8, b: []const u8) i8 {
    const shared_len = @min(a.len, b.len);
    const cmp = std.mem.order(u8, a[0..shared_len], b[0..shared_len]);
    return switch (cmp) {
        .lt => -1,
        .gt => 1,
        .eq => if (a.len < b.len) -1 else if (a.len > b.len) @as(i8, 1) else 0,
    };
}

// ── Real-world Operator Class: ArrayOpsOpClass ─────────────────────────

/// ArrayOpsOpClass — operator class for `array_ops`: GIN support for SQL
/// ARRAY columns via `@>` (contains) and `&&` (overlaps).
///
/// Indexed value wire format independently reimplements (the storage layer
/// must not import the sql layer) the tag+payload `Value` serialization
/// used by src/sql/executor.zig's `serializeValue`: a leading tag byte
/// selects the payload shape (0x00 null, 0x01 integer i64 LE, 0x02 real f64
/// bits LE, 0x03 text u32-len+bytes, 0x04 blob u32-len+bytes, 0x05 boolean 1
/// byte, 0x06 date i32 LE, 0x07 time i64 LE, 0x08 timestamp i64 LE, 0x09
/// interval i32+i32+i64, 0x0A numeric u8-scale+i128 LE, 0x0B uuid 16 bytes,
/// 0x0C array u32-count + N nested tag+payload values, 0x0F tsvector
/// u32-len+bytes, 0x10 tsquery u32-len+bytes). `column_value`/`query_value`
/// must be a 0x0C array value; each element becomes one GIN key (its own
/// tag+payload span, verbatim).
pub const ArrayOpsOpClass = struct {
    /// Lexicographic byte comparison. GIN's entry-tree ordering only needs a
    /// consistent total order for tree placement/lookup, not a semantically
    /// meaningful (e.g. numeric) order, so plain memcmp-with-length-tiebreak
    /// suffices and stays type-agnostic across the polymorphic key encoding.
    pub fn compare(_: std.mem.Allocator, a: []const u8, b: []const u8) OpClassError!i8 {
        return lexCompareKeys(a, b);
    }

    /// Total bytes (including the leading tag byte) consumed by one
    /// tag+payload value at the start of `data`. Used to skip over array
    /// elements without allocating/decoding their contents.
    fn valueSpanLen(data: []const u8) OpClassError!usize {
        if (data.len < 1) return error.InvalidKey;
        const tag = data[0];
        return switch (tag) {
            0x00 => 1, // null
            0x01, 0x02, 0x07, 0x08 => blk: { // integer/real/time/timestamp: 8-byte payload
                if (data.len < 9) return error.InvalidKey;
                break :blk 9;
            },
            0x03, 0x04 => blk: { // text/blob: u32 len prefix + bytes
                if (data.len < 5) return error.InvalidKey;
                const len = std.mem.readInt(u32, data[1..5], .little);
                if (data.len < 5 + @as(usize, len)) return error.InvalidKey;
                break :blk 5 + @as(usize, len);
            },
            0x05 => blk: { // boolean: 1 byte
                if (data.len < 2) return error.InvalidKey;
                break :blk 2;
            },
            0x06 => blk: { // date: 4-byte payload
                if (data.len < 5) return error.InvalidKey;
                break :blk 5;
            },
            0x09 => blk: { // interval: i32 + i32 + i64
                if (data.len < 17) return error.InvalidKey;
                break :blk 17;
            },
            0x0A => blk: { // numeric: u8 scale + i128
                if (data.len < 18) return error.InvalidKey;
                break :blk 18;
            },
            0x0B => blk: { // uuid: 16 bytes
                if (data.len < 17) return error.InvalidKey;
                break :blk 17;
            },
            0x0C => blk: { // nested array: u32 count + N nested values
                if (data.len < 5) return error.InvalidKey;
                const count = std.mem.readInt(u32, data[1..5], .little);
                var pos: usize = 5;
                var i: u32 = 0;
                while (i < count) : (i += 1) {
                    pos += try valueSpanLen(data[pos..]);
                }
                break :blk pos;
            },
            0x0F, 0x10 => blk: { // tsvector/tsquery: u32 len prefix + bytes
                if (data.len < 5) return error.InvalidKey;
                const len = std.mem.readInt(u32, data[1..5], .little);
                if (data.len < 5 + @as(usize, len)) return error.InvalidKey;
                break :blk 5 + @as(usize, len);
            },
            else => error.InvalidKey,
        };
    }

    /// Extract one key per array element (the element's own tag+payload
    /// bytes, verbatim). `column_value` must start with the array tag 0x0C.
    pub fn extractValue(allocator: std.mem.Allocator, column_value: []const u8) OpClassError![][]const u8 {
        if (column_value.len < 5 or column_value[0] != 0x0C) return error.InvalidKey;
        const count = std.mem.readInt(u32, column_value[1..5], .little);
        // Reject counts that can't possibly fit before allocating `count` key
        // slots — each element needs at least 1 byte (the null tag), so a
        // corrupted/malformed count claiming more elements than remaining
        // bytes would otherwise attempt a huge allocation up front.
        if (@as(usize, count) > column_value.len - 5) return error.InvalidKey;

        const keys = try allocator.alloc([]const u8, count);
        var inited: usize = 0;
        errdefer {
            for (keys[0..inited]) |k| allocator.free(k);
            allocator.free(keys);
        }

        var pos: usize = 5;
        var i: u32 = 0;
        while (i < count) : (i += 1) {
            const span = try valueSpanLen(column_value[pos..]);
            keys[i] = try allocator.dupe(u8, column_value[pos..][0..span]);
            pos += span;
            inited += 1;
        }
        return keys;
    }

    /// Query-side extraction: identical format/semantics to extractValue —
    /// the right-hand side of `@>`/`&&` is serialized the same way.
    pub fn extractQuery(allocator: std.mem.Allocator, query_value: []const u8) OpClassError![][]const u8 {
        return extractValue(allocator, query_value);
    }

    /// Strategy 0 (@>): every query key's posting list must be non-empty.
    /// Strategy 1 (&&): at least one query key's posting list must be non-empty.
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

// ── Real-world Operator Class: JsonbOpsOpClass ─────────────────────────

/// JsonbOpsOpClass — operator class for `jsonb_ops`: GIN support for the
/// `@>` (containment) operator on JSON/JSONB columns.
///
/// Silica stores JSON/JSONB column values as raw JSON text (`Value.text`,
/// see src/sql/executor.zig's `type_json`/`type_jsonb` handling) — there is
/// no dedicated json/jsonb tag in the tag+payload wire scheme, so
/// `column_value`/`query_value` here must be tag 0x03 (text): `[0x03][u32
/// len LE][json text bytes]`.
///
/// Indexed entries are produced by a recursive walk shared by extractValue
/// AND extractQuery — this is required for soundness: `@>` GIN-accelerated
/// scans must never produce false negatives, which holds as long as every
/// key/kv-pair/element the query's own walk would emit is also emitted by
/// the row's walk whenever the row genuinely contains the query (recursive
/// containment implies the same entries appear somewhere in the row's own
/// flattened entry set, just not necessarily at the same nesting depth —
/// that's why this opclass is lossy and a downstream recheck against the
/// real `jsonContains` is mandatory, never optional).
///
/// Entry shapes (each is its own GIN key):
///   - object key existence, any depth:      0x01 ++ u32 keylen ++ key
///   - object key + scalar value, any depth: 0x02 ++ u32 keylen ++ key ++ scalarEncode(value)
///   - array scalar element, any depth:      0x03 ++ scalarEncode(value)
///   - bare scalar document root:            0x03 ++ scalarEncode(value) (same shape as array element — the "this scalar exists" query is identical either way)
///
/// `scalarEncode` is a local tag+payload scheme (independent of executor.zig's):
///   0x00 null, 0x01 bool (1B), 0x02 integer i64 LE (8B), 0x03 float f64-bits LE (8B),
///   0x04 string (u32 len + bytes), 0x05 number_string (u32 len + bytes, overflow literals).
pub const JsonbOpsOpClass = struct {
    pub fn compare(_: std.mem.Allocator, a: []const u8, b: []const u8) OpClassError!i8 {
        return lexCompareKeys(a, b);
    }

    fn isScalarJson(node: std.json.Value) bool {
        return switch (node) {
            .object, .array => false,
            else => true,
        };
    }

    /// Encode a scalar std.json.Value as tag+payload bytes. Caller owns the
    /// returned slice.
    fn scalarEncode(allocator: std.mem.Allocator, node: std.json.Value) OpClassError![]u8 {
        return switch (node) {
            .null => blk: {
                const buf = try allocator.alloc(u8, 1);
                buf[0] = 0x00;
                break :blk buf;
            },
            .bool => |b| blk: {
                const buf = try allocator.alloc(u8, 2);
                buf[0] = 0x01;
                buf[1] = if (b) 1 else 0;
                break :blk buf;
            },
            .integer => |i| blk: {
                const buf = try allocator.alloc(u8, 9);
                buf[0] = 0x02;
                std.mem.writeInt(i64, buf[1..9], i, .little);
                break :blk buf;
            },
            .float => |f| blk: {
                const buf = try allocator.alloc(u8, 9);
                buf[0] = 0x03;
                std.mem.writeInt(u64, buf[1..9], @bitCast(f), .little);
                break :blk buf;
            },
            .string => |s| blk: {
                const buf = try allocator.alloc(u8, 5 + s.len);
                buf[0] = 0x04;
                std.mem.writeInt(u32, buf[1..5], @intCast(s.len), .little);
                @memcpy(buf[5..], s);
                break :blk buf;
            },
            .number_string => |s| blk: {
                const buf = try allocator.alloc(u8, 5 + s.len);
                buf[0] = 0x05;
                std.mem.writeInt(u32, buf[1..5], @intCast(s.len), .little);
                @memcpy(buf[5..], s);
                break :blk buf;
            },
            .object, .array => error.InvalidKey, // scalarEncode is for leaf values only
        };
    }

    /// Recursively walk a parsed JSON node, appending GIN entries for every
    /// object key/kv-pair and array scalar element at every nesting depth.
    /// Scalar leaves are a no-op here (the parent object/array branch, or
    /// the top-level extractValue/extractQuery caller for a bare-scalar
    /// root, is responsible for emitting the leaf's own entry — this keeps
    /// each entry emitted exactly once).
    fn walk(allocator: std.mem.Allocator, node: std.json.Value, entries: *std.ArrayList([]const u8)) OpClassError!void {
        switch (node) {
            .object => |obj| {
                var it = obj.iterator();
                while (it.next()) |entry| {
                    const key = entry.key_ptr.*;
                    const kbuf = try allocator.alloc(u8, 5 + key.len);
                    kbuf[0] = 0x01;
                    std.mem.writeInt(u32, kbuf[1..5], @intCast(key.len), .little);
                    @memcpy(kbuf[5..], key);
                    entries.append(allocator, kbuf) catch |err| {
                        allocator.free(kbuf);
                        return err;
                    };

                    if (isScalarJson(entry.value_ptr.*)) {
                        const sval = try scalarEncode(allocator, entry.value_ptr.*);
                        defer allocator.free(sval);
                        const kvbuf = try allocator.alloc(u8, 5 + key.len + sval.len);
                        kvbuf[0] = 0x02;
                        std.mem.writeInt(u32, kvbuf[1..5], @intCast(key.len), .little);
                        @memcpy(kvbuf[5 .. 5 + key.len], key);
                        @memcpy(kvbuf[5 + key.len ..], sval);
                        entries.append(allocator, kvbuf) catch |err| {
                            allocator.free(kvbuf);
                            return err;
                        };
                    }

                    try walk(allocator, entry.value_ptr.*, entries);
                }
            },
            .array => |arr| {
                for (arr.items) |elem| {
                    if (isScalarJson(elem)) {
                        const sval = try scalarEncode(allocator, elem);
                        defer allocator.free(sval);
                        const ebuf = try allocator.alloc(u8, 1 + sval.len);
                        ebuf[0] = 0x03;
                        @memcpy(ebuf[1..], sval);
                        entries.append(allocator, ebuf) catch |err| {
                            allocator.free(ebuf);
                            return err;
                        };
                    }
                    try walk(allocator, elem, entries);
                }
            },
            else => {},
        }
    }

    /// Parse the tag-0x03 wire format and walk the JSON document, producing
    /// one GIN key per entry (see the type doc comment for entry shapes).
    pub fn extractValue(allocator: std.mem.Allocator, column_value: []const u8) OpClassError![][]const u8 {
        if (column_value.len < 5 or column_value[0] != 0x03) return error.InvalidKey;
        const len = std.mem.readInt(u32, column_value[1..5], .little);
        if (column_value.len < 5 + @as(usize, len)) return error.InvalidKey;
        const json_text = column_value[5..][0..len];

        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const parsed = std.json.parseFromSlice(std.json.Value, arena.allocator(), json_text, .{}) catch return error.InvalidKey;

        var entries = std.ArrayList([]const u8){};
        errdefer {
            for (entries.items) |e| allocator.free(e);
            entries.deinit(allocator);
        }

        switch (parsed.value) {
            .object, .array => try walk(allocator, parsed.value, &entries),
            else => {
                const sval = try scalarEncode(allocator, parsed.value);
                defer allocator.free(sval);
                const ebuf = try allocator.alloc(u8, 1 + sval.len);
                ebuf[0] = 0x03;
                @memcpy(ebuf[1..], sval);
                entries.append(allocator, ebuf) catch |err| {
                    allocator.free(ebuf);
                    return err;
                };
            },
        }

        return try entries.toOwnedSlice(allocator);
    }

    /// Query-side extraction: identical format/semantics to extractValue —
    /// the right-hand side of `@>` is serialized the same way and must
    /// share the exact same recursive walker for soundness (see type doc).
    pub fn extractQuery(allocator: std.mem.Allocator, query_value: []const u8) OpClassError![][]const u8 {
        return extractValue(allocator, query_value);
    }

    /// Strategy 0 (`@>`, contains): every query key's posting list must be
    /// non-empty. No other strategy is supported yet — `?`/`?|`/`?&` need a
    /// plain-text/text-array query wire format the current single-shape
    /// `extractQuery` signature can't cleanly express; deferred.
    pub fn consistent(_: std.mem.Allocator, posting_lists: []const []const ItemPointer, _: []const []const u8, strategy: u8) OpClassError!bool {
        return switch (strategy) {
            0 => blk: {
                for (posting_lists) |list| {
                    if (list.len == 0) break :blk false;
                }
                break :blk true;
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

// ── Real-world Operator Class: TsvectorOpsOpClass ──────────────────────

/// TsvectorOpsOpClass — operator class for `tsvector_ops`: GIN support for
/// full-text search with `@@` (match) operator on TSVECTOR columns.
///
/// Wire format:
///   - tsvector column value: tag 0x0F + u32 LE len + text bytes
///     Text format: space-separated sorted-unique lexemes from textToTsvector
///     (lowercased, stemmed, stop-word-filtered)
///   - tsquery query value: tag 0x10 + u32 LE len + text bytes
///     Text format: space-ampersand-space joined lexemes from textToTsquery
///     (lowercased, stemmed, stop-word-filtered; AND-only semantics currently)
///
/// Each lexeme in the text becomes one GIN key (indexed separately).
/// `@@` is strategy 0 (AND semantics): all query keys must match.
pub const TsvectorOpsOpClass = struct {
    pub fn compare(_: std.mem.Allocator, a: []const u8, b: []const u8) OpClassError!i8 {
        return lexCompareKeys(a, b);
    }

    /// Extract lexemes from a tsvector column value.
    /// Input format: [0x0F][u32 len LE][text bytes]
    /// Text format: space-separated lexemes (e.g., "cat dog run")
    /// Returns: slice of duped lexemes, one per GIN key.
    /// Empty text → empty slice (not an error).
    pub fn extractValue(allocator: std.mem.Allocator, column_value: []const u8) OpClassError![][]const u8 {
        // Validate tag and length prefix
        if (column_value.len < 5 or column_value[0] != 0x0F) return error.InvalidKey;
        const len = std.mem.readInt(u32, column_value[1..5], .little);
        if (column_value.len < 5 + @as(usize, len)) return error.InvalidKey;

        const text = column_value[5..][0..len];

        // If empty text, return empty slice
        if (text.len == 0) {
            const keys = try allocator.alloc([]const u8, 0);
            return keys;
        }

        // Split on single space " "
        var lexemes = std.ArrayList([]const u8){};
        errdefer {
            for (lexemes.items) |lex| allocator.free(lex);
            lexemes.deinit(allocator);
        }

        var pos: usize = 0;
        while (pos < text.len) {
            // Find next space starting from pos
            var sep_pos: ?usize = null;
            var i = pos;
            while (i < text.len) {
                if (text[i] == ' ') {
                    sep_pos = i;
                    break;
                }
                i += 1;
            }

            if (sep_pos) |sep| {
                // Found separator
                const lexeme = text[pos..sep];
                if (lexeme.len > 0) {
                    const lex_dup = try allocator.dupe(u8, lexeme);
                    lexemes.append(allocator, lex_dup) catch |err| {
                        allocator.free(lex_dup);
                        return err;
                    };
                }
                pos = sep + 1;
            } else {
                // No more separators
                if (pos < text.len) {
                    const lexeme = text[pos..];
                    if (lexeme.len > 0) {
                        const lex_dup = try allocator.dupe(u8, lexeme);
                        lexemes.append(allocator, lex_dup) catch |err| {
                            allocator.free(lex_dup);
                            return err;
                        };
                    }
                }
                break;
            }
        }

        return try lexemes.toOwnedSlice(allocator);
    }

    /// Extract lexemes from a tsquery query value.
    /// Input format: [0x10][u32 len LE][text bytes]
    /// Text format: space-ampersand-space joined lexemes (e.g., "cat & dog & run")
    /// Returns: slice of duped lexemes, one per GIN key.
    /// Empty text → empty slice (not an error).
    pub fn extractQuery(allocator: std.mem.Allocator, query_value: []const u8) OpClassError![][]const u8 {
        // Validate tag and length prefix
        if (query_value.len < 5 or query_value[0] != 0x10) return error.InvalidKey;
        const len = std.mem.readInt(u32, query_value[1..5], .little);
        if (query_value.len < 5 + @as(usize, len)) return error.InvalidKey;

        const text = query_value[5..][0..len];

        // If empty text, return empty slice
        if (text.len == 0) {
            const keys = try allocator.alloc([]const u8, 0);
            return keys;
        }

        // Split on " & " (space-ampersand-space)
        var lexemes = std.ArrayList([]const u8){};
        errdefer {
            for (lexemes.items) |lex| allocator.free(lex);
            lexemes.deinit(allocator);
        }

        var pos: usize = 0;
        while (pos < text.len) {
            // Find next " & " starting from pos
            var sep_pos: ?usize = null;
            var i = pos;
            while (i + 3 <= text.len) {
                if (text[i] == ' ' and text[i + 1] == '&' and text[i + 2] == ' ') {
                    sep_pos = i;
                    break;
                }
                i += 1;
            }

            if (sep_pos) |sep| {
                // Found separator
                const lexeme = text[pos..sep];
                if (lexeme.len > 0) {
                    const lex_dup = try allocator.dupe(u8, lexeme);
                    lexemes.append(allocator, lex_dup) catch |err| {
                        allocator.free(lex_dup);
                        return err;
                    };
                }
                pos = sep + 3;
            } else {
                // No more separators
                if (pos < text.len) {
                    const lexeme = text[pos..];
                    if (lexeme.len > 0) {
                        const lex_dup = try allocator.dupe(u8, lexeme);
                        lexemes.append(allocator, lex_dup) catch |err| {
                            allocator.free(lex_dup);
                            return err;
                        };
                    }
                }
                break;
            }
        }

        return try lexemes.toOwnedSlice(allocator);
    }

    /// Strategy 0 (@@, match with AND semantics): all query keys must have
    /// non-empty posting lists. Invalid strategies return error.
    pub fn consistent(_: std.mem.Allocator, posting_lists: []const []const ItemPointer, _: []const []const u8, strategy: u8) OpClassError!bool {
        return switch (strategy) {
            0 => blk: {
                for (posting_lists) |list| {
                    if (list.len == 0) break :blk false;
                }
                break :blk true;
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

        // Lookup posting list for each query key
        var posting_lists = try self.allocator.alloc([]ItemPointer, query_keys.len);
        defer {
            for (posting_lists) |list| self.allocator.free(list);
            self.allocator.free(posting_lists);
        }

        for (query_keys, 0..) |key, i| {
            posting_lists[i] = try self.lookupPostingList(key);
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
        _ = entry_count;
    }

    // ── Internal Operations ────────────────────────────────────────────

    /// Insert a single key into the entry tree with associated tuple_id.
    fn insertKey(self: *GIN, key: []const u8, tuple_id: ItemPointer) !void {
        const root_frame = try self.fetchOrInitRootPage();
        defer self.pool.unpinPage(self.root_page_id, true);

        const entry_count = readEntryCount(root_frame.data);

        // Search for existing entry
        for (0..entry_count) |i| {
            const entry_key = try self.readEntryKey(root_frame.data, i);
            defer self.allocator.free(entry_key);

            const cmp = try self.opclass.compare(self.allocator, entry_key, key);
            if (cmp == 0) {
                // Key exists — append to posting list
                try self.appendToPostingList(root_frame.data, i, tuple_id);
                root_frame.markDirty();
                return;
            }
        }

        // Key doesn't exist — insert new entry
        try self.insertNewEntry(root_frame.data, key, tuple_id);
        root_frame.markDirty();
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

        // Search for entry
        for (0..entry_count) |i| {
            const entry_key = try self.readEntryKey(root_frame.data, i);
            defer self.allocator.free(entry_key);

            const cmp = try self.opclass.compare(self.allocator, entry_key, key);

            if (cmp == 0) {
                // Key found — return posting list
                const posting_list = try self.readPostingList(root_frame.data, i);
                return posting_list;
            }
        }

        // Key not found — return empty list
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
    /// For inline lists, uses insertion-sort to maintain sorted order.
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

        // Find insertion position to maintain sorted order (insertion-sort)
        var insert_pos: usize = current_count;
        for (0..current_count) |i| {
            const tid_offset = data_offset + (i * 8);
            if (tid_offset + 8 > page.len) {
                return error.InvalidOffset;
            }
            const tid = std.mem.readInt(u64, page[tid_offset..][0..8], .little);
            if (new_tid == tid) {
                return; // Duplicate — already indexed
            }
            if (new_tid < tid) {
                insert_pos = i;
                break;
            }
        }

        // Calculate insertion position offset
        const insert_offset = data_offset + (insert_pos * 8);

        // Check space availability
        if (insert_offset + 8 > page.len) {
            return error.PageFull;
        }

        // Shift elements right to make room at insert_pos
        var i: usize = current_count;
        while (i > insert_pos) {
            i -= 1;
            const src_offset = data_offset + (i * 8);
            const dst_offset = src_offset + 8;
            if (src_offset + 8 > page.len or dst_offset + 8 > page.len) {
                return error.InvalidOffset;
            }
            const val = std.mem.readInt(u64, page[src_offset..][0..8], .little);
            std.mem.writeInt(u64, page[dst_offset..][0..8], val, .little);
        }

        // Write new tuple ID at insertion position
        std.mem.writeInt(u64, page[insert_offset..][0..8], new_tid, .little);

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
        const posting_info = readPostingInfo(page, idx);

        if (isInlinePostingList(posting_info)) {
            // Inline case: find tuple_id and shift remaining entries left
            try self.removeFromInlinePostingList(page, idx, tuple_id);
        } else {
            // Tree case: walk page chain and remove from posting tree
            const tree_page_id = posting_info & 0x7FFFFFFF;
            try self.removeFromPostingTree(tree_page_id, tuple_id);
        }
    }

    /// Remove tuple_id from an inline posting list at entry index.
    /// Shifts remaining entries left, decrements count.
    fn removeFromInlinePostingList(self: *GIN, page: []u8, idx: usize, tuple_id: ItemPointer) !void {
        _ = self; // Not needed for inline case
        const posting_info = readPostingInfo(page, idx);
        const tuple_count = posting_info & 0x7FFFFFFF;

        if (tuple_count == 0) {
            return error.EntryNotFound;
        }

        // Read the offset pointer to locate the posting data
        const entry_count = readEntryCount(page);
        const offset_ptrs_base = GIN_HEADER_SIZE + (entry_count * GIN_ENTRY_HEADER_SIZE);
        const data_offset_ptr = offset_ptrs_base + (idx * 4);

        if (data_offset_ptr + 4 > page.len) {
            return error.EntryNotFound;
        }

        const data_offset = std.mem.readInt(u32, page[data_offset_ptr..][0..4], .little);
        if (data_offset == 0 or data_offset + 8 > page.len) {
            return error.EntryNotFound;
        }

        const target_u64 = tuple_id.toU64();
        var found_idx: ?usize = null;

        // Search for the tuple_id in the inline list
        for (0..tuple_count) |i| {
            const tid_offset = data_offset + (i * 8);
            if (tid_offset + 8 > page.len) break;
            const tid = std.mem.readInt(u64, page[tid_offset..][0..8], .little);
            if (tid == target_u64) {
                found_idx = i;
                break;
            }
        }

        if (found_idx == null) {
            return error.EntryNotFound;
        }

        const remove_idx = found_idx.?;

        // Shift remaining entries left to close the gap
        for (remove_idx..tuple_count - 1) |i| {
            const src_offset = data_offset + ((i + 1) * 8);
            const dst_offset = data_offset + (i * 8);
            if (src_offset + 8 > page.len) break;
            const val = std.mem.readInt(u64, page[src_offset..][0..8], .little);
            std.mem.writeInt(u64, page[dst_offset..][0..8], val, .little);
        }

        // Decrement count and write back posting_info
        const new_count = tuple_count - 1;
        const new_posting_info = new_count; // Keep high bit 0 for inline
        const info_offset = GIN_HEADER_SIZE + (idx * GIN_ENTRY_HEADER_SIZE) + 2;
        std.mem.writeInt(u32, page[info_offset..][0..4], new_posting_info, .little);
    }

    /// Remove tuple_id from a posting tree chain.
    /// Walks the page chain, finds and removes the tuple on its page, decrements page count.
    fn removeFromPostingTree(self: *GIN, root_tree_page_id: u32, tuple_id: ItemPointer) !void {
        const target_u64 = tuple_id.toU64();
        var current_page_id = root_tree_page_id;

        const max_chain_pages = self.pool.pager.page_count + 1;
        var pages_visited: u32 = 0;

        while (current_page_id != 0) {
            pages_visited += 1;
            if (pages_visited > max_chain_pages) return error.InvalidKey; // cycle or corruption

            const tree_frame = try self.pool.fetchPage(current_page_id);
            var found_idx: ?usize = null;

            const current_count = std.mem.readInt(u32, tree_frame.data[PAGE_HEADER_SIZE..][0..4], .little);
            const next_page_id = std.mem.readInt(u32, tree_frame.data[POSTING_TREE_NEXT_PAGE_OFFSET..][0..4], .little);

            // Search for tuple_id on this page
            for (0..current_count) |i| {
                const tid_offset = POSTING_TREE_HEADER_SIZE + (i * 8);
                if (tid_offset + 8 > tree_frame.data.len) {
                    self.pool.unpinPage(current_page_id, false);
                    return error.InvalidKey;
                }
                const tid = std.mem.readInt(u64, tree_frame.data[tid_offset..][0..8], .little);
                if (tid == target_u64) {
                    found_idx = i;
                    break;
                }
            }

            if (found_idx != null) {
                // Found it on this page — remove it
                const remove_idx = found_idx.?;

                // Shift remaining entries left to close the gap
                for (remove_idx..current_count - 1) |i| {
                    const src_offset = POSTING_TREE_HEADER_SIZE + ((i + 1) * 8);
                    const dst_offset = POSTING_TREE_HEADER_SIZE + (i * 8);
                    if (src_offset + 8 > tree_frame.data.len) {
                        self.pool.unpinPage(current_page_id, false);
                        return error.InvalidKey;
                    }
                    const val = std.mem.readInt(u64, tree_frame.data[src_offset..][0..8], .little);
                    std.mem.writeInt(u64, tree_frame.data[dst_offset..][0..8], val, .little);
                }

                // Decrement count
                const new_count = current_count - 1;
                std.mem.writeInt(u32, tree_frame.data[PAGE_HEADER_SIZE..][0..4], @intCast(new_count), .little);
                tree_frame.markDirty();
                self.pool.unpinPage(current_page_id, true);
                return;
            }

            // Not found on this page — continue to next
            self.pool.unpinPage(current_page_id, false);
            current_page_id = next_page_id;
        }

        // Tuple not found in the entire posting tree
        return error.EntryNotFound;
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
// Operator Class: ArrayOpsOpClass Tests (~18 tests)
// ────────────────────────────────────────────────────────────────────
// ArrayOpsOpClass implements array_ops: serialized SQL ARRAY with polymorphic
// element types (tag+payload wire format). Each array element becomes one key.
// Comparison is lexicographic (byte-wise). Strategies: 0=@> (contains all),
// 1=&& (overlaps).

test "ArrayOpsOpClass extractValue single integer element" {
    const allocator = std.testing.allocator;

    // Array[1 element]: tag=0x0C, count=1 (u32 LE), then 1 i64 element (tag 0x01 + 8 bytes LE)
    // Wire format: [0x0C, 0x01,0x00,0x00,0x00, 0x01,0x00,0x00,0x00,0x00,0x00,0x00,0x00]
    //             (tag)  (count LE)         (tag) (value LE = 1)
    var input: [14]u8 = undefined;
    input[0] = 0x0C; // array tag
    std.mem.writeInt(u32, input[1..5], 1, .little); // count = 1
    input[5] = 0x01; // integer tag
    std.mem.writeInt(i64, input[6..14], 42, .little); // value = 42

    const keys = try ArrayOpsOpClass.extractValue(allocator, &input);
    defer {
        for (keys) |key| allocator.free(key);
        allocator.free(keys);
    }

    try std.testing.expectEqual(@as(usize, 1), keys.len);
    try std.testing.expectEqual(@as(usize, 9), keys[0].len); // tag(1) + i64(8)
    try std.testing.expectEqual(@as(u8, 0x01), keys[0][0]); // tag preserved
    try std.testing.expectEqual(@as(i64, 42), std.mem.readInt(i64, keys[0][1..9], .little));
}

test "ArrayOpsOpClass extractValue multiple element array" {
    const allocator = std.testing.allocator;

    // Array[2 elements]: [i64(10), i64(20)]
    // Total: tag(1) + count(4) + elem0_tag(1) + elem0_value(8) + elem1_tag(1) + elem1_value(8)
    var input: [23]u8 = undefined;
    input[0] = 0x0C;
    std.mem.writeInt(u32, input[1..5], 2, .little); // count = 2
    input[5] = 0x01;
    std.mem.writeInt(i64, input[6..14], 10, .little);
    input[14] = 0x01;
    std.mem.writeInt(i64, input[15..23], 20, .little);

    const keys = try ArrayOpsOpClass.extractValue(allocator, &input);
    defer {
        for (keys) |key| allocator.free(key);
        allocator.free(keys);
    }

    try std.testing.expectEqual(@as(usize, 2), keys.len);
    try std.testing.expectEqual(@as(usize, 9), keys[0].len);
    try std.testing.expectEqual(@as(usize, 9), keys[1].len);
    try std.testing.expectEqual(@as(i64, 10), std.mem.readInt(i64, keys[0][1..9], .little));
    try std.testing.expectEqual(@as(i64, 20), std.mem.readInt(i64, keys[1][1..9], .little));
}

test "ArrayOpsOpClass extractValue empty array" {
    const allocator = std.testing.allocator;

    var input: [5]u8 = undefined;
    input[0] = 0x0C;
    std.mem.writeInt(u32, input[1..5], 0, .little); // count = 0

    const keys = try ArrayOpsOpClass.extractValue(allocator, &input);
    defer allocator.free(keys);

    try std.testing.expectEqual(@as(usize, 0), keys.len);
}

test "ArrayOpsOpClass extractValue count exceeds remaining buffer rejected before allocating" {
    const allocator = std.testing.allocator;

    // count claims ~4 billion elements but the buffer has zero bytes left —
    // must be rejected before attempting `allocator.alloc([]const u8, count)`.
    var input: [5]u8 = undefined;
    input[0] = 0x0C;
    std.mem.writeInt(u32, input[1..5], 0xFFFFFFFF, .little);

    const result = ArrayOpsOpClass.extractValue(allocator, &input);
    try std.testing.expectError(error.InvalidKey, result);
}

test "ArrayOpsOpClass extractValue input too short for count field" {
    const allocator = std.testing.allocator;

    var input: [1]u8 = undefined;
    input[0] = 0x0C; // only tag, no count bytes

    const result = ArrayOpsOpClass.extractValue(allocator, &input);
    try std.testing.expectError(error.InvalidKey, result);
}

test "ArrayOpsOpClass extractValue non-array leading tag" {
    const allocator = std.testing.allocator;

    var input: [9]u8 = undefined;
    input[0] = 0x01; // integer tag, not array
    std.mem.writeInt(i64, input[1..9], 42, .little);

    const result = ArrayOpsOpClass.extractValue(allocator, &input);
    try std.testing.expectError(error.InvalidKey, result);
}

test "ArrayOpsOpClass extractValue truncated element data" {
    const allocator = std.testing.allocator;

    // Claim 2 elements but only provide 1 complete element
    var input: [14]u8 = undefined;
    input[0] = 0x0C;
    std.mem.writeInt(u32, input[1..5], 2, .little); // count = 2
    input[5] = 0x01;
    std.mem.writeInt(i64, input[6..14], 10, .little);
    // Missing second element

    const result = ArrayOpsOpClass.extractValue(allocator, input[0..14]); // truncated: only 14 bytes
    try std.testing.expectError(error.InvalidKey, result);
}

test "ArrayOpsOpClass extractValue array with text element" {
    const allocator = std.testing.allocator;

    // Array[1]: text value "hello"
    // Wire: [0x0C, count=1, 0x03, length=5, "hello"]
    var input: [5 + 1 + 4 + 4 + 5]u8 = undefined;
    var pos: usize = 0;
    input[pos] = 0x0C;
    pos += 1;
    std.mem.writeInt(u32, input[pos..][0..4], 1, .little); // count = 1
    pos += 4;
    input[pos] = 0x03; // text tag
    pos += 1;
    std.mem.writeInt(u32, input[pos..][0..4], 5, .little); // length = 5
    pos += 4;
    @memcpy(input[pos..][0..5], "hello");

    const keys = try ArrayOpsOpClass.extractValue(allocator, &input);
    defer {
        for (keys) |key| allocator.free(key);
        allocator.free(keys);
    }

    try std.testing.expectEqual(@as(usize, 1), keys.len);
    try std.testing.expectEqual(@as(usize, 1 + 4 + 5), keys[0].len); // tag + length_prefix + text
    try std.testing.expectEqual(@as(u8, 0x03), keys[0][0]); // tag preserved
    try std.testing.expectEqual(@as(u32, 5), std.mem.readInt(u32, keys[0][1..5], .little));
    try std.testing.expectEqualSlices(u8, "hello", keys[0][5..10]);
}

test "ArrayOpsOpClass extractQuery returns same extraction as extractValue" {
    const allocator = std.testing.allocator;

    var input: [23]u8 = undefined;
    input[0] = 0x0C;
    std.mem.writeInt(u32, input[1..5], 2, .little); // count = 2
    input[5] = 0x01;
    std.mem.writeInt(i64, input[6..14], 100, .little);
    input[14] = 0x01;
    std.mem.writeInt(i64, input[15..23], 200, .little);

    const keys = try ArrayOpsOpClass.extractQuery(allocator, &input);
    defer {
        for (keys) |key| allocator.free(key);
        allocator.free(keys);
    }

    try std.testing.expectEqual(@as(usize, 2), keys.len);
    try std.testing.expectEqual(@as(i64, 100), std.mem.readInt(i64, keys[0][1..9], .little));
    try std.testing.expectEqual(@as(i64, 200), std.mem.readInt(i64, keys[1][1..9], .little));
}

test "ArrayOpsOpClass compare equal keys" {
    const allocator = std.testing.allocator;

    // Two integer keys with value 42
    var key_a: [9]u8 = undefined;
    var key_b: [9]u8 = undefined;
    key_a[0] = 0x01;
    std.mem.writeInt(i64, key_a[1..9], 42, .little);
    key_b[0] = 0x01;
    std.mem.writeInt(i64, key_b[1..9], 42, .little);

    const result = try ArrayOpsOpClass.compare(allocator, &key_a, &key_b);
    try std.testing.expectEqual(@as(i8, 0), result);
}

test "ArrayOpsOpClass compare less than" {
    const allocator = std.testing.allocator;

    // Two integer keys: 10 vs 20
    var key_a: [9]u8 = undefined;
    var key_b: [9]u8 = undefined;
    key_a[0] = 0x01;
    std.mem.writeInt(i64, key_a[1..9], 10, .little);
    key_b[0] = 0x01;
    std.mem.writeInt(i64, key_b[1..9], 20, .little);

    const result = try ArrayOpsOpClass.compare(allocator, &key_a, &key_b);
    try std.testing.expectEqual(@as(i8, -1), result);
}

test "ArrayOpsOpClass compare greater than" {
    const allocator = std.testing.allocator;

    // Two integer keys: 100 vs 50
    var key_a: [9]u8 = undefined;
    var key_b: [9]u8 = undefined;
    key_a[0] = 0x01;
    std.mem.writeInt(i64, key_a[1..9], 100, .little);
    key_b[0] = 0x01;
    std.mem.writeInt(i64, key_b[1..9], 50, .little);

    const result = try ArrayOpsOpClass.compare(allocator, &key_a, &key_b);
    try std.testing.expectEqual(@as(i8, 1), result);
}

test "ArrayOpsOpClass compare antisymmetry" {
    const allocator = std.testing.allocator;

    var key_a: [9]u8 = undefined;
    var key_b: [9]u8 = undefined;
    key_a[0] = 0x01;
    std.mem.writeInt(i64, key_a[1..9], 5, .little);
    key_b[0] = 0x01;
    std.mem.writeInt(i64, key_b[1..9], 15, .little);

    const cmp_ab = try ArrayOpsOpClass.compare(allocator, &key_a, &key_b);
    const cmp_ba = try ArrayOpsOpClass.compare(allocator, &key_b, &key_a);

    // cmp_ab should be -1, cmp_ba should be 1 (opposite signs)
    try std.testing.expectEqual(@as(i8, -cmp_ab), cmp_ba);
}

test "ArrayOpsOpClass consistent contains all strategy all present" {
    const allocator = std.testing.allocator;

    const item1 = [_]ItemPointer{.{ .page_id = 1, .tuple_offset = 0 }};
    const item2 = [_]ItemPointer{.{ .page_id = 2, .tuple_offset = 5 }};
    const posting_lists = [_][]const ItemPointer{ &item1, &item2 };

    var key1: [9]u8 = undefined;
    var key2: [9]u8 = undefined;
    key1[0] = 0x01;
    std.mem.writeInt(i64, key1[1..9], 1, .little);
    key2[0] = 0x01;
    std.mem.writeInt(i64, key2[1..9], 2, .little);
    const query_keys = [_][]const u8{ &key1, &key2 };

    const result = try ArrayOpsOpClass.consistent(allocator, &posting_lists, &query_keys, 0);
    try std.testing.expect(result);
}

test "ArrayOpsOpClass consistent contains all strategy one missing" {
    const allocator = std.testing.allocator;

    const item1 = [_]ItemPointer{.{ .page_id = 1, .tuple_offset = 0 }};
    const empty: [0]ItemPointer = undefined;
    const posting_lists = [_][]const ItemPointer{ &item1, &empty };

    var key1: [9]u8 = undefined;
    var key2: [9]u8 = undefined;
    key1[0] = 0x01;
    std.mem.writeInt(i64, key1[1..9], 1, .little);
    key2[0] = 0x01;
    std.mem.writeInt(i64, key2[1..9], 2, .little);
    const query_keys = [_][]const u8{ &key1, &key2 };

    const result = try ArrayOpsOpClass.consistent(allocator, &posting_lists, &query_keys, 0);
    try std.testing.expect(!result);
}

test "ArrayOpsOpClass consistent contains all strategy zero query keys" {
    const allocator = std.testing.allocator;

    const empty_lists: [0][]const ItemPointer = undefined;
    const empty_keys: [0][]const u8 = undefined;

    const result = try ArrayOpsOpClass.consistent(allocator, &empty_lists, &empty_keys, 0);
    try std.testing.expect(result); // vacuously true
}

test "ArrayOpsOpClass consistent overlaps strategy at least one present" {
    const allocator = std.testing.allocator;

    const empty: [0]ItemPointer = undefined;
    const item2 = [_]ItemPointer{.{ .page_id = 2, .tuple_offset = 3 }};
    const posting_lists = [_][]const ItemPointer{ &empty, &item2 };

    var key1: [9]u8 = undefined;
    var key2: [9]u8 = undefined;
    key1[0] = 0x01;
    std.mem.writeInt(i64, key1[1..9], 1, .little);
    key2[0] = 0x01;
    std.mem.writeInt(i64, key2[1..9], 2, .little);
    const query_keys = [_][]const u8{ &key1, &key2 };

    const result = try ArrayOpsOpClass.consistent(allocator, &posting_lists, &query_keys, 1);
    try std.testing.expect(result);
}

test "ArrayOpsOpClass consistent overlaps strategy all empty" {
    const allocator = std.testing.allocator;

    const empty1: [0]ItemPointer = undefined;
    const empty2: [0]ItemPointer = undefined;
    const posting_lists = [_][]const ItemPointer{ &empty1, &empty2 };

    var key1: [9]u8 = undefined;
    var key2: [9]u8 = undefined;
    key1[0] = 0x01;
    std.mem.writeInt(i64, key1[1..9], 1, .little);
    key2[0] = 0x01;
    std.mem.writeInt(i64, key2[1..9], 2, .little);
    const query_keys = [_][]const u8{ &key1, &key2 };

    const result = try ArrayOpsOpClass.consistent(allocator, &posting_lists, &query_keys, 1);
    try std.testing.expect(!result);
}

test "ArrayOpsOpClass consistent invalid strategy" {
    const allocator = std.testing.allocator;

    const empty: [0]ItemPointer = undefined;
    const posting_lists = [_][]const ItemPointer{&empty};

    var key1: [9]u8 = undefined;
    key1[0] = 0x01;
    std.mem.writeInt(i64, key1[1..9], 1, .little);
    const query_keys = [_][]const u8{&key1};

    const result = ArrayOpsOpClass.consistent(allocator, &posting_lists, &query_keys, 99);
    try std.testing.expectError(error.InvalidKey, result);
}

test "ArrayOpsOpClass getOpClass returns valid opclass" {
    const opclass = ArrayOpsOpClass.getOpClass();

    // Smoke test: verify all function pointers are non-null by calling them once
    const allocator = std.testing.allocator;

    // Test compare
    var key_a: [9]u8 = undefined;
    key_a[0] = 0x01;
    std.mem.writeInt(i64, key_a[1..9], 1, .little);
    const cmp_result = try opclass.compare(allocator, &key_a, &key_a);
    try std.testing.expectEqual(@as(i8, 0), cmp_result);

    // Test extractValue
    var arr: [14]u8 = undefined;
    arr[0] = 0x0C;
    std.mem.writeInt(u32, arr[1..5], 1, .little);
    arr[5] = 0x01;
    std.mem.writeInt(i64, arr[6..14], 42, .little);
    const keys = try opclass.extractValue(allocator, &arr);
    defer {
        for (keys) |key| allocator.free(key);
        allocator.free(keys);
    }
    try std.testing.expectEqual(@as(usize, 1), keys.len);

    // Test extractQuery (delegate to extractValue)
    const query_keys = try opclass.extractQuery(allocator, &arr);
    defer {
        for (query_keys) |key| allocator.free(key);
        allocator.free(query_keys);
    }
    try std.testing.expectEqual(@as(usize, 1), query_keys.len);

    // Test consistent
    const item = [_]ItemPointer{.{ .page_id = 1, .tuple_offset = 0 }};
    const posting_lists = [_][]const ItemPointer{&item};
    const consistent_result = try opclass.consistent(allocator, &posting_lists, query_keys, 0);
    try std.testing.expect(consistent_result);
}

// ── Operator Class: JsonbOpsOpClass Tests ──────────────────────────────
// JsonbOpsOpClass implements jsonb_ops: serialized JSON/JSONB with recursive
// key extraction (object keys + array elements). Supports only strategy 0 (@>).
// Wire format: tag 0x03 (text) LE u32 length + JSON bytes. Recursive walker
// emits key-exists entries (0x01+keylen+key) and key-value entries (0x02+...,
// if scalar), plus array-element scalar entries (0x03+scalarEncode).

test "JsonbOpsOpClass compare equal keys" {
    const allocator = std.testing.allocator;

    // Two text keys with same content: "hello"
    var key_a: [10]u8 = undefined;
    var key_b: [10]u8 = undefined;
    @memcpy(key_a[0..5], "hello");
    @memcpy(key_b[0..5], "hello");

    const result = try JsonbOpsOpClass.compare(allocator, key_a[0..5], key_b[0..5]);
    try std.testing.expectEqual(@as(i8, 0), result);
}

test "JsonbOpsOpClass compare less than" {
    const allocator = std.testing.allocator;

    var key_a: [5]u8 = undefined;
    var key_b: [5]u8 = undefined;
    @memcpy(key_a[0..5], "apple");
    @memcpy(key_b[0..5], "zebra");

    const result = try JsonbOpsOpClass.compare(allocator, &key_a, &key_b);
    try std.testing.expectEqual(@as(i8, -1), result);
}

test "JsonbOpsOpClass compare greater than" {
    const allocator = std.testing.allocator;

    var key_a: [4]u8 = undefined;
    var key_b: [4]u8 = undefined;
    @memcpy(key_a[0..4], "zulu");
    @memcpy(key_b[0..4], "alfa");

    const result = try JsonbOpsOpClass.compare(allocator, &key_a, &key_b);
    try std.testing.expectEqual(@as(i8, 1), result);
}

test "JsonbOpsOpClass compare antisymmetry" {
    const allocator = std.testing.allocator;

    var key_a: [3]u8 = undefined;
    var key_b: [3]u8 = undefined;
    @memcpy(key_a[0..3], "cat");
    @memcpy(key_b[0..3], "dog");

    const cmp_ab = try JsonbOpsOpClass.compare(allocator, &key_a, &key_b);
    const cmp_ba = try JsonbOpsOpClass.compare(allocator, &key_b, &key_a);

    try std.testing.expectEqual(@as(i8, -cmp_ab), cmp_ba);
}

test "JsonbOpsOpClass extractValue simple object with integer and string keys" {
    const allocator = std.testing.allocator;

    // JSON: {"a":1,"b":"hi"}
    // Wire format: [0x03, len(u32 LE), json text]
    const json_text = "{\"a\":1,\"b\":\"hi\"}";
    var input: [5 + json_text.len]u8 = undefined;
    input[0] = 0x03; // tag
    std.mem.writeInt(u32, input[1..5], @intCast(json_text.len), .little);
    @memcpy(input[5..], json_text);

    const keys = try JsonbOpsOpClass.extractValue(allocator, &input);
    defer {
        for (keys) |key| allocator.free(key);
        allocator.free(keys);
    }

    // Expected: key-exists for "a", key-value for "a"(1), key-exists for "b", key-value for "b"("hi")
    // That's 4 entries total
    try std.testing.expectEqual(@as(usize, 4), keys.len);

    // Verify first entry is key-exists for "a": 0x01 + u32(1) + "a"
    try std.testing.expectEqual(@as(u8, 0x01), keys[0][0]);
    const a_keylen = std.mem.readInt(u32, keys[0][1..5], .little);
    try std.testing.expectEqual(@as(u32, 1), a_keylen);
    try std.testing.expectEqualSlices(u8, "a", keys[0][5..6]);
}

test "JsonbOpsOpClass extractValue nested object" {
    const allocator = std.testing.allocator;

    // JSON: {"a":{"b":1}}
    // Outer key "a" should have key-exists only (value is object, not scalar)
    // Inner key "b" should have key-exists and key-value (value is scalar 1)
    const json_text = "{\"a\":{\"b\":1}}";
    var input: [5 + json_text.len]u8 = undefined;
    input[0] = 0x03;
    std.mem.writeInt(u32, input[1..5], @intCast(json_text.len), .little);
    @memcpy(input[5..], json_text);

    const keys = try JsonbOpsOpClass.extractValue(allocator, &input);
    defer {
        for (keys) |key| allocator.free(key);
        allocator.free(keys);
    }

    // Expected: key-exists for "a", key-exists for "b", key-value for "b"(1)
    try std.testing.expect(keys.len >= 3);
}

test "JsonbOpsOpClass extractValue array of scalars" {
    const allocator = std.testing.allocator;

    // JSON: [1,2,3]
    // Should produce: element-scalar for 1, element-scalar for 2, element-scalar for 3
    const json_text = "[1,2,3]";
    var input: [5 + json_text.len]u8 = undefined;
    input[0] = 0x03;
    std.mem.writeInt(u32, input[1..5], @intCast(json_text.len), .little);
    @memcpy(input[5..], json_text);

    const keys = try JsonbOpsOpClass.extractValue(allocator, &input);
    defer {
        for (keys) |key| allocator.free(key);
        allocator.free(keys);
    }

    // Array elements should produce entries
    try std.testing.expect(keys.len >= 3);
}

test "JsonbOpsOpClass extractValue array containing object" {
    const allocator = std.testing.allocator;

    // JSON: [{"x":1}]
    // Should produce both object key entries and array element entries
    const json_text = "[{\"x\":1}]";
    var input: [5 + json_text.len]u8 = undefined;
    input[0] = 0x03;
    std.mem.writeInt(u32, input[1..5], @intCast(json_text.len), .little);
    @memcpy(input[5..], json_text);

    const keys = try JsonbOpsOpClass.extractValue(allocator, &input);
    defer {
        for (keys) |key| allocator.free(key);
        allocator.free(keys);
    }

    // Should have entries for the nested key "x"
    try std.testing.expect(keys.len >= 2);
}

test "JsonbOpsOpClass extractValue bare scalar root" {
    const allocator = std.testing.allocator;

    // JSON: 5 (bare scalar)
    // Should produce exactly one entry: 0x03 + scalarEncode(5)
    const json_text = "5";
    var input: [5 + json_text.len]u8 = undefined;
    input[0] = 0x03;
    std.mem.writeInt(u32, input[1..5], @intCast(json_text.len), .little);
    @memcpy(input[5..], json_text);

    const keys = try JsonbOpsOpClass.extractValue(allocator, &input);
    defer {
        for (keys) |key| allocator.free(key);
        allocator.free(keys);
    }

    try std.testing.expectEqual(@as(usize, 1), keys.len);
    try std.testing.expectEqual(@as(u8, 0x03), keys[0][0]); // array-element-scalar tag
}

test "JsonbOpsOpClass extractValue malformed tag" {
    const allocator = std.testing.allocator;

    // Wrong leading tag (not 0x03)
    var input: [10]u8 = undefined;
    input[0] = 0x01; // integer tag, not text
    std.mem.writeInt(u32, input[1..5], 5, .little);

    const result = JsonbOpsOpClass.extractValue(allocator, &input);
    try std.testing.expectError(error.InvalidKey, result);
}

test "JsonbOpsOpClass extractValue truncated length prefix" {
    const allocator = std.testing.allocator;

    // Tag 0x03 claims more bytes than actually present
    var input: [5]u8 = undefined;
    input[0] = 0x03;
    std.mem.writeInt(u32, input[1..5], 1000, .little); // claims 1000 bytes

    const result = JsonbOpsOpClass.extractValue(allocator, &input);
    try std.testing.expectError(error.InvalidKey, result);
}

test "JsonbOpsOpClass extractValue non-JSON text" {
    const allocator = std.testing.allocator;

    // Valid tag 0x03 but invalid JSON text
    const invalid_json = "not { valid json";
    var input: [5 + invalid_json.len]u8 = undefined;
    input[0] = 0x03;
    std.mem.writeInt(u32, input[1..5], @intCast(invalid_json.len), .little);
    @memcpy(input[5..], invalid_json);

    const result = JsonbOpsOpClass.extractValue(allocator, &input);
    try std.testing.expectError(error.InvalidKey, result);
}

test "JsonbOpsOpClass extractQuery equivalence with extractValue" {
    const allocator = std.testing.allocator;

    // JSON: {"key":"value"}
    const json_text = "{\"key\":\"value\"}";
    var input: [5 + json_text.len]u8 = undefined;
    input[0] = 0x03;
    std.mem.writeInt(u32, input[1..5], @intCast(json_text.len), .little);
    @memcpy(input[5..], json_text);

    const value_keys = try JsonbOpsOpClass.extractValue(allocator, &input);
    defer {
        for (value_keys) |key| allocator.free(key);
        allocator.free(value_keys);
    }

    const query_keys = try JsonbOpsOpClass.extractQuery(allocator, &input);
    defer {
        for (query_keys) |key| allocator.free(key);
        allocator.free(query_keys);
    }

    try std.testing.expectEqual(value_keys.len, query_keys.len);
}

test "JsonbOpsOpClass consistent contains all strategy all present" {
    const allocator = std.testing.allocator;

    const item1 = [_]ItemPointer{.{ .page_id = 1, .tuple_offset = 0 }};
    const item2 = [_]ItemPointer{.{ .page_id = 2, .tuple_offset = 5 }};
    const posting_lists = [_][]const ItemPointer{ &item1, &item2 };

    var key1: [1]u8 = undefined;
    var key2: [1]u8 = undefined;
    key1[0] = 'a';
    key2[0] = 'b';
    const query_keys = [_][]const u8{ &key1, &key2 };

    const result = try JsonbOpsOpClass.consistent(allocator, &posting_lists, &query_keys, 0);
    try std.testing.expect(result);
}

test "JsonbOpsOpClass consistent contains all strategy one missing" {
    const allocator = std.testing.allocator;

    const item1 = [_]ItemPointer{.{ .page_id = 1, .tuple_offset = 0 }};
    const empty: [0]ItemPointer = undefined;
    const posting_lists = [_][]const ItemPointer{ &item1, &empty };

    var key1: [1]u8 = undefined;
    var key2: [1]u8 = undefined;
    key1[0] = 'a';
    key2[0] = 'b';
    const query_keys = [_][]const u8{ &key1, &key2 };

    const result = try JsonbOpsOpClass.consistent(allocator, &posting_lists, &query_keys, 0);
    try std.testing.expect(!result);
}

test "JsonbOpsOpClass consistent contains all strategy zero query keys" {
    const allocator = std.testing.allocator;

    const empty_lists: [0][]const ItemPointer = undefined;
    const empty_keys: [0][]const u8 = undefined;

    const result = try JsonbOpsOpClass.consistent(allocator, &empty_lists, &empty_keys, 0);
    try std.testing.expect(result); // vacuously true
}

test "JsonbOpsOpClass consistent unsupported strategy" {
    const allocator = std.testing.allocator;

    const empty: [0]ItemPointer = undefined;
    const posting_lists = [_][]const ItemPointer{&empty};

    var key1: [1]u8 = undefined;
    key1[0] = 'a';
    const query_keys = [_][]const u8{&key1};

    const result = JsonbOpsOpClass.consistent(allocator, &posting_lists, &query_keys, 99);
    try std.testing.expectError(error.InvalidKey, result);
}

test "JsonbOpsOpClass getOpClass returns valid opclass" {
    const opclass = JsonbOpsOpClass.getOpClass();

    // Smoke test: verify all function pointers are non-null by calling them once
    const allocator = std.testing.allocator;

    // Test compare
    var key_a: [5]u8 = undefined;
    @memcpy(key_a[0..5], "hello");
    const cmp_result = try opclass.compare(allocator, &key_a, &key_a);
    try std.testing.expectEqual(@as(i8, 0), cmp_result);

    // Test extractValue
    const json_text = "{\"x\":1}";
    var input: [5 + json_text.len]u8 = undefined;
    input[0] = 0x03;
    std.mem.writeInt(u32, input[1..5], @intCast(json_text.len), .little);
    @memcpy(input[5..], json_text);

    const keys = try opclass.extractValue(allocator, &input);
    defer {
        for (keys) |key| allocator.free(key);
        allocator.free(keys);
    }
    try std.testing.expect(keys.len > 0);

    // Test extractQuery
    const query_keys = try opclass.extractQuery(allocator, &input);
    defer {
        for (query_keys) |key| allocator.free(key);
        allocator.free(query_keys);
    }
    try std.testing.expectEqual(keys.len, query_keys.len);

    // Test consistent
    const item = [_]ItemPointer{.{ .page_id = 1, .tuple_offset = 0 }};
    const posting_lists = [_][]const ItemPointer{&item};
    if (query_keys.len > 0) {
        const consistent_result = try opclass.consistent(allocator, &posting_lists, query_keys[0..1], 0);
        try std.testing.expect(consistent_result);
    }
}

// ── Operator Class: TsvectorOpsOpClass Tests ───────────────────────────
// TsvectorOpsOpClass implements tsvector_ops: full-text search with tsvector
// column values and tsquery predicates. Supports only strategy 0 (@@).
// Wire format: tag 0x0F (tsvector) or 0x10 (tsquery), u32 LE len + raw text.
// tsvector: space-separated sorted-unique lexemes (e.g. "cat dog run")
// tsquery: space-ampersand-space joined lexemes (e.g. "cat & dog & run")

test "TsvectorOpsOpClass extractValue single lexeme" {
    const allocator = std.testing.allocator;

    // tsvector with single lexeme: "hello"
    const text = "hello";
    var input: [5 + text.len]u8 = undefined;
    input[0] = 0x0F; // tsvector tag
    std.mem.writeInt(u32, input[1..5], @intCast(text.len), .little);
    @memcpy(input[5..], text);

    const keys = try TsvectorOpsOpClass.extractValue(allocator, &input);
    defer {
        for (keys) |key| allocator.free(key);
        allocator.free(keys);
    }

    try std.testing.expectEqual(@as(usize, 1), keys.len);
    try std.testing.expectEqualSlices(u8, "hello", keys[0]);
}

test "TsvectorOpsOpClass extractValue multiple lexemes" {
    const allocator = std.testing.allocator;

    // tsvector with multiple space-separated lexemes: "cat dog run"
    const text = "cat dog run";
    var input: [5 + text.len]u8 = undefined;
    input[0] = 0x0F;
    std.mem.writeInt(u32, input[1..5], @intCast(text.len), .little);
    @memcpy(input[5..], text);

    const keys = try TsvectorOpsOpClass.extractValue(allocator, &input);
    defer {
        for (keys) |key| allocator.free(key);
        allocator.free(keys);
    }

    try std.testing.expectEqual(@as(usize, 3), keys.len);
    try std.testing.expectEqualSlices(u8, "cat", keys[0]);
    try std.testing.expectEqualSlices(u8, "dog", keys[1]);
    try std.testing.expectEqualSlices(u8, "run", keys[2]);
}

test "TsvectorOpsOpClass extractValue empty tsvector" {
    const allocator = std.testing.allocator;

    // Empty tsvector text (valid, 0 keys)
    const text = "";
    var input: [5 + text.len]u8 = undefined;
    input[0] = 0x0F;
    std.mem.writeInt(u32, input[1..5], @intCast(text.len), .little);

    const keys = try TsvectorOpsOpClass.extractValue(allocator, &input);
    defer allocator.free(keys);

    try std.testing.expectEqual(@as(usize, 0), keys.len);
}

test "TsvectorOpsOpClass extractValue wrong tag rejected" {
    const allocator = std.testing.allocator;

    // Wrong tag (0x03 instead of 0x0F)
    const text = "hello";
    var input: [5 + text.len]u8 = undefined;
    input[0] = 0x03; // text tag, not tsvector
    std.mem.writeInt(u32, input[1..5], @intCast(text.len), .little);
    @memcpy(input[5..], text);

    const result = TsvectorOpsOpClass.extractValue(allocator, &input);
    try std.testing.expectError(error.InvalidKey, result);
}

test "TsvectorOpsOpClass extractValue truncated length prefix" {
    const allocator = std.testing.allocator;

    // Only 1 byte (tag) but no length prefix
    var input: [1]u8 = undefined;
    input[0] = 0x0F;

    const result = TsvectorOpsOpClass.extractValue(allocator, &input);
    try std.testing.expectError(error.InvalidKey, result);
}

test "TsvectorOpsOpClass extractValue length exceeds buffer" {
    const allocator = std.testing.allocator;

    // Length prefix claims more bytes than actually present
    var input: [5]u8 = undefined;
    input[0] = 0x0F;
    std.mem.writeInt(u32, input[1..5], 1000, .little); // claims 1000 bytes

    const result = TsvectorOpsOpClass.extractValue(allocator, &input);
    try std.testing.expectError(error.InvalidKey, result);
}

test "TsvectorOpsOpClass extractQuery single lexeme" {
    const allocator = std.testing.allocator;

    // tsquery with single lexeme: "hello"
    const text = "hello";
    var input: [5 + text.len]u8 = undefined;
    input[0] = 0x10; // tsquery tag
    std.mem.writeInt(u32, input[1..5], @intCast(text.len), .little);
    @memcpy(input[5..], text);

    const keys = try TsvectorOpsOpClass.extractQuery(allocator, &input);
    defer {
        for (keys) |key| allocator.free(key);
        allocator.free(keys);
    }

    try std.testing.expectEqual(@as(usize, 1), keys.len);
    try std.testing.expectEqualSlices(u8, "hello", keys[0]);
}

test "TsvectorOpsOpClass extractQuery multiple lexemes with AND" {
    const allocator = std.testing.allocator;

    // tsquery with multiple lexemes joined by " & ": "cat & dog & run"
    const text = "cat & dog & run";
    var input: [5 + text.len]u8 = undefined;
    input[0] = 0x10;
    std.mem.writeInt(u32, input[1..5], @intCast(text.len), .little);
    @memcpy(input[5..], text);

    const keys = try TsvectorOpsOpClass.extractQuery(allocator, &input);
    defer {
        for (keys) |key| allocator.free(key);
        allocator.free(keys);
    }

    try std.testing.expectEqual(@as(usize, 3), keys.len);
    try std.testing.expectEqualSlices(u8, "cat", keys[0]);
    try std.testing.expectEqualSlices(u8, "dog", keys[1]);
    try std.testing.expectEqualSlices(u8, "run", keys[2]);
}

test "TsvectorOpsOpClass extractQuery empty tsquery" {
    const allocator = std.testing.allocator;

    // Empty tsquery text (valid, 0 keys)
    const text = "";
    var input: [5 + text.len]u8 = undefined;
    input[0] = 0x10;
    std.mem.writeInt(u32, input[1..5], @intCast(text.len), .little);

    const keys = try TsvectorOpsOpClass.extractQuery(allocator, &input);
    defer allocator.free(keys);

    try std.testing.expectEqual(@as(usize, 0), keys.len);
}

test "TsvectorOpsOpClass extractQuery wrong tag rejected" {
    const allocator = std.testing.allocator;

    // Wrong tag (0x03 instead of 0x10)
    const text = "hello";
    var input: [5 + text.len]u8 = undefined;
    input[0] = 0x03; // text tag, not tsquery
    std.mem.writeInt(u32, input[1..5], @intCast(text.len), .little);
    @memcpy(input[5..], text);

    const result = TsvectorOpsOpClass.extractQuery(allocator, &input);
    try std.testing.expectError(error.InvalidKey, result);
}

test "TsvectorOpsOpClass extractQuery truncated length prefix" {
    const allocator = std.testing.allocator;

    // Only 1 byte (tag) but no length prefix
    var input: [1]u8 = undefined;
    input[0] = 0x10;

    const result = TsvectorOpsOpClass.extractQuery(allocator, &input);
    try std.testing.expectError(error.InvalidKey, result);
}

test "TsvectorOpsOpClass extractQuery length exceeds buffer" {
    const allocator = std.testing.allocator;

    // Length prefix claims more bytes than actually present (for tsquery tag 0x10)
    var input: [5]u8 = undefined;
    input[0] = 0x10; // tsquery tag
    std.mem.writeInt(u32, input[1..5], 1000, .little); // claims 1000 bytes

    const result = TsvectorOpsOpClass.extractQuery(allocator, &input);
    try std.testing.expectError(error.InvalidKey, result);
}

test "TsvectorOpsOpClass extractQuery leading separator produces one key" {
    const allocator = std.testing.allocator;

    // tsquery with leading separator: " & cat"
    // Bug: current code produces 2 keys (empty string and "cat")
    // Expected: exactly 1 key "cat"
    const text = " & cat";
    var input: [5 + text.len]u8 = undefined;
    input[0] = 0x10;
    std.mem.writeInt(u32, input[1..5], @intCast(text.len), .little);
    @memcpy(input[5..], text);

    const keys = try TsvectorOpsOpClass.extractQuery(allocator, &input);
    defer {
        for (keys) |key| allocator.free(key);
        allocator.free(keys);
    }

    // Should be exactly 1 key "cat", not 2 keys with empty string first
    try std.testing.expectEqual(@as(usize, 1), keys.len);
    try std.testing.expectEqualSlices(u8, "cat", keys[0]);
}

test "TsvectorOpsOpClass extractQuery trailing separator produces one key" {
    const allocator = std.testing.allocator;

    // tsquery with trailing separator: "cat & "
    // Bug: current code produces 2 keys ("cat" and empty string)
    // Expected: exactly 1 key "cat"
    const text = "cat & ";
    var input: [5 + text.len]u8 = undefined;
    input[0] = 0x10;
    std.mem.writeInt(u32, input[1..5], @intCast(text.len), .little);
    @memcpy(input[5..], text);

    const keys = try TsvectorOpsOpClass.extractQuery(allocator, &input);
    defer {
        for (keys) |key| allocator.free(key);
        allocator.free(keys);
    }

    // Should be exactly 1 key "cat", not 2 keys with empty string at end
    try std.testing.expectEqual(@as(usize, 1), keys.len);
    try std.testing.expectEqualSlices(u8, "cat", keys[0]);
}

test "TsvectorOpsOpClass extractQuery consecutive separators produces correct count" {
    const allocator = std.testing.allocator;

    // tsquery with consecutive separators: "cat &  & dog"
    // (double space before the & and after)
    // Bug: current code produces 3 keys ("cat", empty string, "dog")
    // Expected: exactly 2 keys "cat" and "dog"
    const text = "cat &  & dog";
    var input: [5 + text.len]u8 = undefined;
    input[0] = 0x10;
    std.mem.writeInt(u32, input[1..5], @intCast(text.len), .little);
    @memcpy(input[5..], text);

    const keys = try TsvectorOpsOpClass.extractQuery(allocator, &input);
    defer {
        for (keys) |key| allocator.free(key);
        allocator.free(keys);
    }

    // Should be exactly 2 keys "cat" and "dog", no empty key in between
    try std.testing.expectEqual(@as(usize, 2), keys.len);
    try std.testing.expectEqualSlices(u8, "cat", keys[0]);
    try std.testing.expectEqualSlices(u8, "dog", keys[1]);
}

test "TsvectorOpsOpClass compare equal keys" {
    const allocator = std.testing.allocator;

    var key_a: [5]u8 = undefined;
    var key_b: [5]u8 = undefined;
    @memcpy(key_a[0..5], "hello");
    @memcpy(key_b[0..5], "hello");

    const result = try TsvectorOpsOpClass.compare(allocator, &key_a, &key_b);
    try std.testing.expectEqual(@as(i8, 0), result);
}

test "TsvectorOpsOpClass compare less than" {
    const allocator = std.testing.allocator;

    var key_a: [3]u8 = undefined;
    var key_b: [3]u8 = undefined;
    @memcpy(key_a[0..3], "abc");
    @memcpy(key_b[0..3], "xyz");

    const result = try TsvectorOpsOpClass.compare(allocator, &key_a, &key_b);
    try std.testing.expectEqual(@as(i8, -1), result);
}

test "TsvectorOpsOpClass compare greater than" {
    const allocator = std.testing.allocator;

    var key_a: [3]u8 = undefined;
    var key_b: [3]u8 = undefined;
    @memcpy(key_a[0..3], "xyz");
    @memcpy(key_b[0..3], "abc");

    const result = try TsvectorOpsOpClass.compare(allocator, &key_a, &key_b);
    try std.testing.expectEqual(@as(i8, 1), result);
}

test "TsvectorOpsOpClass consistent strategy 0 all query keys present" {
    const allocator = std.testing.allocator;

    const item1 = [_]ItemPointer{.{ .page_id = 1, .tuple_offset = 0 }};
    const item2 = [_]ItemPointer{.{ .page_id = 2, .tuple_offset = 5 }};
    const posting_lists = [_][]const ItemPointer{ &item1, &item2 };

    var key1: [3]u8 = undefined;
    var key2: [3]u8 = undefined;
    @memcpy(key1[0..3], "cat");
    @memcpy(key2[0..3], "dog");
    const query_keys = [_][]const u8{ &key1, &key2 };

    const result = try TsvectorOpsOpClass.consistent(allocator, &posting_lists, &query_keys, 0);
    try std.testing.expect(result);
}

test "TsvectorOpsOpClass consistent strategy 0 one query key missing" {
    const allocator = std.testing.allocator;

    const item1 = [_]ItemPointer{.{ .page_id = 1, .tuple_offset = 0 }};
    const empty: [0]ItemPointer = undefined;
    const posting_lists = [_][]const ItemPointer{ &item1, &empty };

    var key1: [3]u8 = undefined;
    var key2: [3]u8 = undefined;
    @memcpy(key1[0..3], "cat");
    @memcpy(key2[0..3], "dog");
    const query_keys = [_][]const u8{ &key1, &key2 };

    const result = try TsvectorOpsOpClass.consistent(allocator, &posting_lists, &query_keys, 0);
    try std.testing.expect(!result);
}

test "TsvectorOpsOpClass consistent invalid strategy" {
    const allocator = std.testing.allocator;

    const empty: [0]ItemPointer = undefined;
    const posting_lists = [_][]const ItemPointer{&empty};

    var key1: [3]u8 = undefined;
    @memcpy(key1[0..3], "cat");
    const query_keys = [_][]const u8{&key1};

    const result = TsvectorOpsOpClass.consistent(allocator, &posting_lists, &query_keys, 99);
    try std.testing.expectError(error.InvalidKey, result);
}

test "TsvectorOpsOpClass getOpClass returns valid opclass" {
    const opclass = TsvectorOpsOpClass.getOpClass();

    // Smoke test: verify all function pointers are wired and produce expected results
    const allocator = std.testing.allocator;

    // Test compare
    var key_a: [5]u8 = undefined;
    @memcpy(key_a[0..5], "hello");
    const cmp_result = try opclass.compare(allocator, &key_a, &key_a);
    try std.testing.expectEqual(@as(i8, 0), cmp_result);

    // Test extractValue
    const text = "foo bar";
    var input: [5 + text.len]u8 = undefined;
    input[0] = 0x0F;
    std.mem.writeInt(u32, input[1..5], @intCast(text.len), .little);
    @memcpy(input[5..], text);

    const keys = try opclass.extractValue(allocator, &input);
    defer {
        for (keys) |key| allocator.free(key);
        allocator.free(keys);
    }
    try std.testing.expectEqual(@as(usize, 2), keys.len);

    // Test extractQuery
    const query_text = "foo & bar";
    var query_input: [5 + query_text.len]u8 = undefined;
    query_input[0] = 0x10;
    std.mem.writeInt(u32, query_input[1..5], @intCast(query_text.len), .little);
    @memcpy(query_input[5..], query_text);

    const query_keys = try opclass.extractQuery(allocator, &query_input);
    defer {
        for (query_keys) |key| allocator.free(key);
        allocator.free(query_keys);
    }
    try std.testing.expectEqual(@as(usize, 2), query_keys.len);

    // Test consistent
    const item = [_]ItemPointer{.{ .page_id = 1, .tuple_offset = 0 }};
    const posting_lists = [_][]const ItemPointer{&item};
    if (query_keys.len > 0) {
        const consistent_result = try opclass.consistent(allocator, &posting_lists, query_keys[0..1], 0);
        try std.testing.expect(consistent_result);
    }
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

    // Append smaller tuple ID (should succeed via insertion-sort, inserting in middle)
    const tid3 = ItemPointer{ .page_id = 1, .tuple_offset = 3 };
    try gin.appendToPostingList(root_frame.data, 0, tid3);

    // Verify the list is sorted after insertion-sort
    const inline_list = try gin.readInlinePostingList(root_frame.data, 0);
    defer allocator.free(inline_list);
    try std.testing.expectEqual(@as(usize, 3), inline_list.len);
    for (inline_list[0 .. inline_list.len - 1], 1..) |curr, i| {
        const next = inline_list[i];
        try std.testing.expect(curr.toU64() < next.toU64());
    }
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

    // Append a tuple with lower ID (should succeed via insertion-sort, inserting at beginning)
    const new_tid = ItemPointer{ .page_id = 50, .tuple_offset = 25 }; // Lower than first
    try gin.appendToPostingList(root_frame.data, 0, new_tid);

    // Verify the list is sorted with the new tuple at the beginning
    const inline_list = try gin.readInlinePostingList(root_frame.data, 0);
    defer allocator.free(inline_list);
    try std.testing.expectEqual(@as(usize, 2), inline_list.len);
    try std.testing.expectEqual(new_tid.toU64(), inline_list[0].toU64());
    try std.testing.expectEqual(first_tid.toU64(), inline_list[1].toU64());
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

// ────────────────────────────────────────────────────────────────────
// Real Deletion Tests for removeFromPostingList
// ────────────────────────────────────────────────────────────────────

test "GIN delete one tuple from inline posting list with multiple tuples" {
    const allocator = std.testing.allocator;
    const path = "test_gin_delete_inline_one.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    const root_id = try pager.allocPage();
    var pool = try BufferPool.init(allocator, &pager, 100);
    defer pool.deinit();

    const opclass = ArrayInt32OpClass.getOpClass();
    var gin = try GIN.init(allocator, &pool, root_id, opclass);

    // Insert 4 different tuples under the same key (42)
    const key_value: u32 = 42;
    const tuples = [_]ItemPointer{
        .{ .page_id = 100, .tuple_offset = 0 },
        .{ .page_id = 100, .tuple_offset = 5 },
        .{ .page_id = 101, .tuple_offset = 2 },
        .{ .page_id = 102, .tuple_offset = 10 },
    };

    var col_value: [8]u8 = undefined;
    std.mem.writeInt(u32, col_value[0..4], 1, .little);
    std.mem.writeInt(u32, col_value[4..8], key_value, .little);

    for (tuples) |tid| {
        try gin.insert(&col_value, tid);
    }

    // Verify all 4 tuples are present before deletion
    var query: [8]u8 = undefined;
    std.mem.writeInt(u32, query[0..4], 1, .little);
    std.mem.writeInt(u32, query[4..8], key_value, .little);

    const before_result = try gin.search(&query, 0);
    defer allocator.free(before_result);
    try std.testing.expectEqual(@as(usize, 4), before_result.len);

    // Delete the second tuple (100, 5)
    const to_delete = tuples[1];
    try gin.delete(&col_value, to_delete);

    // Verify only 3 tuples remain and the deleted one is gone
    var after_result = try gin.search(&query, 0);
    defer allocator.free(after_result);
    try std.testing.expectEqual(@as(usize, 3), after_result.len);

    // Verify the deleted tuple is not in the result
    for (after_result) |tid| {
        try std.testing.expect(!(tid.page_id == to_delete.page_id and tid.tuple_offset == to_delete.tuple_offset));
    }

    // Verify remaining tuples are still sorted
    for (after_result[0 .. after_result.len - 1]) |curr| {
        const next = after_result[after_result.len - 1];
        try std.testing.expect(curr.toU64() < next.toU64());
    }
}

test "GIN delete then re-insert tuple in inline posting list maintains consistency" {
    const allocator = std.testing.allocator;
    const path = "test_gin_delete_reinsert_inline.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    const root_id = try pager.allocPage();
    var pool = try BufferPool.init(allocator, &pager, 100);
    defer pool.deinit();

    const opclass = ArrayInt32OpClass.getOpClass();
    var gin = try GIN.init(allocator, &pool, root_id, opclass);

    // Insert 3 tuples under the same key
    const key_value: u32 = 999;
    const tuple1 = ItemPointer{ .page_id = 50, .tuple_offset = 0 };
    const tuple2 = ItemPointer{ .page_id = 60, .tuple_offset = 10 };
    const tuple3 = ItemPointer{ .page_id = 70, .tuple_offset = 20 };

    var col_value: [8]u8 = undefined;
    std.mem.writeInt(u32, col_value[0..4], 1, .little);
    std.mem.writeInt(u32, col_value[4..8], key_value, .little);

    try gin.insert(&col_value, tuple1);
    try gin.insert(&col_value, tuple2);
    try gin.insert(&col_value, tuple3);

    // Verify 3 tuples present
    var query: [8]u8 = undefined;
    std.mem.writeInt(u32, query[0..4], 1, .little);
    std.mem.writeInt(u32, query[4..8], key_value, .little);

    const result1 = try gin.search(&query, 0);
    defer allocator.free(result1);
    try std.testing.expectEqual(@as(usize, 3), result1.len);

    // Delete tuple2
    try gin.delete(&col_value, tuple2);

    // Verify 2 tuples remain
    const result2 = try gin.search(&query, 0);
    defer allocator.free(result2);
    try std.testing.expectEqual(@as(usize, 2), result2.len);

    // Re-insert a different tuple (tuple2 replacement)
    const tuple2_new = ItemPointer{ .page_id = 65, .tuple_offset = 15 };
    try gin.insert(&col_value, tuple2_new);

    // Verify 3 tuples now, including the new one
    var result3 = try gin.search(&query, 0);
    defer allocator.free(result3);
    try std.testing.expectEqual(@as(usize, 3), result3.len);

    // Verify the new tuple is present
    var found_new = false;
    for (result3) |tid| {
        if (tid.page_id == tuple2_new.page_id and tid.tuple_offset == tuple2_new.tuple_offset) {
            found_new = true;
            break;
        }
    }
    try std.testing.expect(found_new);

    // Verify result is still sorted
    for (result3[0 .. result3.len - 1], 1..) |curr, i| {
        const next = result3[i];
        try std.testing.expect(curr.toU64() < next.toU64());
    }
}

test "GIN delete one tuple from posting tree maintains sortedness" {
    const allocator = std.testing.allocator;
    const path = "test_gin_delete_tree_one.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    const root_id = try pager.allocPage();
    var pool = try BufferPool.init(allocator, &pager, 100);
    defer pool.deinit();

    const opclass = ArrayInt32OpClass.getOpClass();
    var gin = try GIN.init(allocator, &pool, root_id, opclass);

    // Insert 18 tuples under the same key to force conversion to posting tree
    const key_value: u32 = 777;
    const num_tuples = 18;

    var col_value: [8]u8 = undefined;
    std.mem.writeInt(u32, col_value[0..4], 1, .little);
    std.mem.writeInt(u32, col_value[4..8], key_value, .little);

    var inserted_tuples = try allocator.alloc(ItemPointer, num_tuples);
    defer allocator.free(inserted_tuples);

    for (0..num_tuples) |i| {
        const tid = ItemPointer{ .page_id = @intCast(200 + i), .tuple_offset = 0 };
        inserted_tuples[i] = tid;
        try gin.insert(&col_value, tid);
    }

    // Verify all 18 tuples are present
    var query: [8]u8 = undefined;
    std.mem.writeInt(u32, query[0..4], 1, .little);
    std.mem.writeInt(u32, query[4..8], key_value, .little);

    const before_result = try gin.search(&query, 0);
    defer allocator.free(before_result);
    try std.testing.expectEqual(@as(usize, 18), before_result.len);

    // Verify it's in a posting tree (high bit set in posting_info)
    const root_frame = try gin.fetchOrInitRootPage();
    defer pool.unpinPage(root_id, false);
    const posting_info = readPostingInfo(root_frame.data, 0);
    try std.testing.expect((posting_info & 0x80000000) != 0);

    // Delete a tuple from the middle (e.g., 9th tuple)
    const to_delete = inserted_tuples[9];
    try gin.delete(&col_value, to_delete);

    // Verify 17 tuples remain
    var after_result = try gin.search(&query, 0);
    defer allocator.free(after_result);
    try std.testing.expectEqual(@as(usize, 17), after_result.len);

    // Verify the deleted tuple is gone
    for (after_result) |tid| {
        try std.testing.expect(!(tid.page_id == to_delete.page_id and tid.tuple_offset == to_delete.tuple_offset));
    }

    // Verify remaining tuples are sorted
    for (after_result[0 .. after_result.len - 1], 1..) |curr, i| {
        const next = after_result[i];
        try std.testing.expect(curr.toU64() < next.toU64());
    }
}

test "GIN delete non-existent tuple_id within existing key's posting list should error" {
    const allocator = std.testing.allocator;
    const path = "test_gin_delete_nonexistent_tuple.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var pager = try Pager.init(allocator, path, .{});
    defer pager.deinit();

    const root_id = try pager.allocPage();
    var pool = try BufferPool.init(allocator, &pager, 100);
    defer pool.deinit();

    const opclass = ArrayInt32OpClass.getOpClass();
    var gin = try GIN.init(allocator, &pool, root_id, opclass);

    // Insert 3 tuples under a key
    const key_value: u32 = 555;
    const existing_tuples = [_]ItemPointer{
        .{ .page_id = 30, .tuple_offset = 0 },
        .{ .page_id = 40, .tuple_offset = 0 },
        .{ .page_id = 50, .tuple_offset = 0 },
    };

    var col_value: [8]u8 = undefined;
    std.mem.writeInt(u32, col_value[0..4], 1, .little);
    std.mem.writeInt(u32, col_value[4..8], key_value, .little);

    for (existing_tuples) |tid| {
        try gin.insert(&col_value, tid);
    }

    // Try to delete a tuple_id that was never inserted under this key
    const nonexistent_tuple = ItemPointer{ .page_id = 999, .tuple_offset = 999 };

    // Expected behavior: Deleting a tuple_id that doesn't exist within the key's posting list
    // should return an error (entry not found within the posting list).
    // This is different from deleting a key that was never inserted at all.
    // The key EXISTS but this specific tuple_id was never added to it.
    // Current status: removeFromPostingList is a no-op stub, so delete will silently succeed.
    // This test will FAIL until removeFromPostingList correctly searches for the tuple_id
    // and returns error.EntryNotFound if not found.
    const result = gin.delete(&col_value, nonexistent_tuple);
    try std.testing.expectError(error.EntryNotFound, result);
}
