//! Volcano-model query executor — iterator-based, pull execution engine.
//!
//! Each operator implements `open()`, `next()`, `close()` and produces
//! Row tuples that flow up the operator tree. The executor translates
//! optimized LogicalPlan nodes into a tree of iterators.
//!
//! Supported operators:
//!   Scan     — full table scan via B+Tree cursor
//!   Filter   — predicate evaluation (WHERE)
//!   Project  — column selection / expression computation
//!   Sort     — in-memory ORDER BY
//!   Limit    — row count restriction with OFFSET
//!   Aggregate — GROUP BY with aggregate functions
//!   NestedLoopJoin — nested loop join (all join types)
//!   Values   — literal row set (INSERT VALUES)

const std = @import("std");
const Allocator = std.mem.Allocator;
const ast = @import("ast.zig");
const catalog_mod = @import("catalog.zig");
const planner_mod = @import("planner.zig");
const btree_mod = @import("../storage/btree.zig");
const buffer_pool_mod = @import("../storage/buffer_pool.zig");
const page_mod = @import("../storage/page.zig");

const mvcc_mod = @import("../tx/mvcc.zig");

const BTree = btree_mod.BTree;
const Cursor = btree_mod.Cursor;
const BufferPool = buffer_pool_mod.BufferPool;
const Pager = page_mod.Pager;
const Catalog = catalog_mod.Catalog;
const ColumnInfo = catalog_mod.ColumnInfo;
const ColumnType = catalog_mod.ColumnType;
const TableInfo = catalog_mod.TableInfo;
const PlanNode = planner_mod.PlanNode;
const LogicalPlan = planner_mod.LogicalPlan;
const PlanType = planner_mod.PlanType;
const AggFunc = planner_mod.AggFunc;
const TupleHeader = mvcc_mod.TupleHeader;
const Snapshot = mvcc_mod.Snapshot;

// ── Date/Time Constants & Utilities ───────────────────────────────────────

const MICROS_PER_SECOND: i64 = 1_000_000;
const MICROS_PER_MINUTE: i64 = 60 * MICROS_PER_SECOND;
const MICROS_PER_HOUR: i64 = 60 * MICROS_PER_MINUTE;
const MICROS_PER_DAY: i64 = 24 * MICROS_PER_HOUR;

fn isLeapYear(year: i32) bool {
    return (@mod(year, 4) == 0 and @mod(year, 100) != 0) or (@mod(year, 400) == 0);
}

fn daysInMonth(month: u8, year: i32) u8 {
    const days = [_]u8{ 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };
    if (month == 2 and isLeapYear(year)) return 29;
    return days[month - 1];
}

/// Convert year/month/day to days since Unix epoch (1970-01-01).
/// Uses the civil_from_days algorithm.
fn dateToDays(year: i32, month: u8, day: u8) i32 {
    const year_adj: i32 = if (month <= 2) 1 else 0;
    const y = year - year_adj;
    const era = @divFloor(y, 400);
    const yoe: i32 = y - era * 400;
    const m_adj: i32 = if (month > 2) -3 else 9;
    const m = @as(i32, month) + m_adj;
    const doy = @divFloor((153 * m + 2), 5) + @as(i32, day) - 1;
    const doe = yoe * 365 + @divFloor(yoe, 4) - @divFloor(yoe, 100) + doy;
    return era * 146097 + doe - 719468;
}

/// Convert days since epoch to year/month/day.
fn daysToDate(days: i32) struct { year: i32, month: u8, day: u8 } {
    const z = days + 719468;
    const era = @divFloor(z, 146097);
    const doe: i32 = z - era * 146097;
    const yoe = @divFloor(doe - @divFloor(doe, 1460) + @divFloor(doe, 36524) - @divFloor(doe, 146096), 365);
    const y = yoe + era * 400;
    const doy = doe - (365 * yoe + @divFloor(yoe, 4) - @divFloor(yoe, 100));
    const mp = @divFloor(5 * doy + 2, 153);
    const d = @as(u8, @intCast(doy - @divFloor((153 * mp + 2), 5) + 1));
    const m_adj: i32 = if (mp < 10) 3 else -9;
    const m = @as(u8, @intCast(mp + m_adj));
    const year_adj: i32 = if (m <= 2) 1 else 0;
    return .{ .year = y + year_adj, .month = m, .day = d };
}

/// Parse 'YYYY-MM-DD' format into days since epoch.
fn parseDateString(s: []const u8) ?i32 {
    if (s.len < 10) return null;
    const year = std.fmt.parseInt(i32, s[0..4], 10) catch return null;
    if (s[4] != '-') return null;
    const month = std.fmt.parseInt(u8, s[5..7], 10) catch return null;
    if (s[7] != '-') return null;
    const day = std.fmt.parseInt(u8, s[8..10], 10) catch return null;
    if (month < 1 or month > 12) return null;
    if (day < 1 or day > daysInMonth(month, year)) return null;
    return dateToDays(year, month, day);
}

/// Parse 'HH:MM:SS' or 'HH:MM:SS.ffffff' format into microseconds since midnight.
fn parseTimeString(s: []const u8) ?i64 {
    if (s.len < 8) return null;
    const hour = std.fmt.parseInt(u8, s[0..2], 10) catch return null;
    if (s[2] != ':') return null;
    const minute = std.fmt.parseInt(u8, s[3..5], 10) catch return null;
    if (s[5] != ':') return null;

    var second: u8 = 0;
    var micros: i64 = 0;

    if (s.len >= 8) {
        second = std.fmt.parseInt(u8, s[6..8], 10) catch return null;
    }

    if (s.len > 8 and s[8] == '.') {
        // Parse fractional seconds
        var frac_str: [6]u8 = [_]u8{'0'} ** 6;
        const frac_len = @min(s.len - 9, 6);
        @memcpy(frac_str[0..frac_len], s[9..][0..frac_len]);
        const frac = std.fmt.parseInt(u32, &frac_str, 10) catch return null;
        micros = @as(i64, frac);
    }

    if (hour > 23 or minute > 59 or second > 59) return null;

    return @as(i64, hour) * MICROS_PER_HOUR +
           @as(i64, minute) * MICROS_PER_MINUTE +
           @as(i64, second) * MICROS_PER_SECOND + micros;
}

/// Parse 'YYYY-MM-DD HH:MM:SS' format into microseconds since epoch.
fn parseTimestampString(s: []const u8) ?i64 {
    if (s.len < 19) return null;
    const days = parseDateString(s[0..10]) orelse return null;
    if (s[10] != ' ') return null;
    const time_micros = parseTimeString(s[11..]) orelse return null;
    return @as(i64, days) * MICROS_PER_DAY + time_micros;
}

/// Format days since epoch as 'YYYY-MM-DD'.
pub fn formatDate(allocator: Allocator, days: i32) ![]u8 {
    const date = daysToDate(days);
    // Handle year carefully - format as signed for negative years, but don't show + for positive
    if (date.year < 0) {
        return std.fmt.allocPrint(allocator, "{d:0>5}-{d:0>2}-{d:0>2}", .{ date.year, @as(u32, date.month), @as(u32, date.day) });
    } else {
        return std.fmt.allocPrint(allocator, "{d:0>4}-{d:0>2}-{d:0>2}", .{ @as(u32, @intCast(date.year)), @as(u32, date.month), @as(u32, date.day) });
    }
}

/// Format microseconds since midnight as 'HH:MM:SS'.
pub fn formatTime(allocator: Allocator, micros: i64) ![]u8 {
    const total_secs = @divTrunc(micros, MICROS_PER_SECOND);
    const hour = @as(u32, @intCast(@divTrunc(total_secs, 3600)));
    const minute = @as(u32, @intCast(@divTrunc(@mod(total_secs, 3600), 60)));
    const second = @as(u32, @intCast(@mod(total_secs, 60)));
    return std.fmt.allocPrint(allocator, "{d:0>2}:{d:0>2}:{d:0>2}", .{ hour, minute, second });
}

/// Format microseconds since epoch as 'YYYY-MM-DD HH:MM:SS'.
pub fn formatTimestamp(allocator: Allocator, micros: i64) ![]u8 {
    const days: i32 = @intCast(@divTrunc(micros, MICROS_PER_DAY));
    const time_micros = @mod(micros, MICROS_PER_DAY);
    const date = daysToDate(days);
    const total_secs = @divTrunc(time_micros, MICROS_PER_SECOND);
    const hour = @as(u32, @intCast(@divTrunc(total_secs, 3600)));
    const minute = @as(u32, @intCast(@divTrunc(@mod(total_secs, 3600), 60)));
    const second = @as(u32, @intCast(@mod(total_secs, 60)));
    // Handle year carefully - format as unsigned for positive years
    if (date.year < 0) {
        return std.fmt.allocPrint(allocator, "{d:0>5}-{d:0>2}-{d:0>2} {d:0>2}:{d:0>2}:{d:0>2}",
            .{ date.year, @as(u32, date.month), @as(u32, date.day), hour, minute, second });
    } else {
        return std.fmt.allocPrint(allocator, "{d:0>4}-{d:0>2}-{d:0>2} {d:0>2}:{d:0>2}:{d:0>2}",
            .{ @as(u32, @intCast(date.year)), @as(u32, date.month), @as(u32, date.day), hour, minute, second });
    }
}

/// Format an interval as PostgreSQL-compatible string (e.g., "1 year 2 mons 3 days 04:05:06").
pub fn formatInterval(allocator: Allocator, iv: Value.Interval) ![]u8 {
    var parts: [4][]const u8 = undefined;
    var bufs: [4][]u8 = undefined;
    var count: usize = 0;

    const years = @divTrunc(iv.months, 12);
    const mons = @mod(iv.months, 12);

    if (years != 0) {
        bufs[count] = try std.fmt.allocPrint(allocator, "{d} year{s}", .{ years, if (years == 1 or years == -1) "" else "s" });
        parts[count] = bufs[count];
        count += 1;
    }
    if (mons != 0) {
        bufs[count] = try std.fmt.allocPrint(allocator, "{d} mon{s}", .{ mons, if (mons == 1 or mons == -1) "" else "s" });
        parts[count] = bufs[count];
        count += 1;
    }
    if (iv.days != 0) {
        bufs[count] = try std.fmt.allocPrint(allocator, "{d} day{s}", .{ iv.days, if (iv.days == 1 or iv.days == -1) "" else "s" });
        parts[count] = bufs[count];
        count += 1;
    }

    const abs_micros = if (iv.micros < 0) -iv.micros else iv.micros;
    const total_secs = @divTrunc(abs_micros, MICROS_PER_SECOND);
    const h = @as(u32, @intCast(@divTrunc(total_secs, 3600)));
    const m = @as(u32, @intCast(@divTrunc(@mod(total_secs, 3600), 60)));
    const s = @as(u32, @intCast(@mod(total_secs, 60)));
    if (iv.micros != 0) {
        if (iv.micros < 0) {
            bufs[count] = try std.fmt.allocPrint(allocator, "-{d:0>2}:{d:0>2}:{d:0>2}", .{ h, m, s });
        } else {
            bufs[count] = try std.fmt.allocPrint(allocator, "{d:0>2}:{d:0>2}:{d:0>2}", .{ h, m, s });
        }
        parts[count] = bufs[count];
        count += 1;
    }

    if (count == 0) {
        return try std.fmt.allocPrint(allocator, "00:00:00", .{});
    }

    // Join parts with spaces
    var total_len: usize = 0;
    for (parts[0..count]) |p| total_len += p.len;
    total_len += count - 1; // spaces

    const result = try allocator.alloc(u8, total_len);
    var pos: usize = 0;
    for (parts[0..count], 0..) |p, i| {
        @memcpy(result[pos..][0..p.len], p);
        pos += p.len;
        if (i < count - 1) {
            result[pos] = ' ';
            pos += 1;
        }
    }
    for (bufs[0..count]) |b| allocator.free(b);

    return result;
}

/// Parse PostgreSQL-style interval string: "1 year 2 months 3 days 04:05:06"
/// Also supports short forms: "1 day", "2 hours", "30 minutes", "1 year 6 months"
fn parseIntervalString(s: []const u8) ?Value.Interval {
    var months: i32 = 0;
    var days: i32 = 0;
    var micros: i64 = 0;

    var i: usize = 0;
    while (i < s.len) {
        // Skip whitespace
        while (i < s.len and s[i] == ' ') i += 1;
        if (i >= s.len) break;

        // Check for time component (HH:MM:SS)
        if (isTimeComponent(s[i..])) {
            micros += parseTimeComponent(s[i..]) orelse return null;
            break;
        }

        // Check for negative time component (-HH:MM:SS)
        if (s[i] == '-' and i + 1 < s.len and isTimeComponent(s[i + 1 ..])) {
            micros -= parseTimeComponent(s[i + 1 ..]) orelse return null;
            break;
        }

        // Parse number
        const neg = s[i] == '-';
        if (neg) i += 1;
        const num_start = i;
        while (i < s.len and s[i] >= '0' and s[i] <= '9') i += 1;
        if (i == num_start) return null;
        const num = std.fmt.parseInt(i32, s[num_start..i], 10) catch return null;
        const value = if (neg) -num else num;

        // Skip whitespace
        while (i < s.len and s[i] == ' ') i += 1;

        // Parse unit
        if (i >= s.len) {
            // Bare number — treat as seconds
            micros += @as(i64, value) * MICROS_PER_SECOND;
            break;
        }

        const unit_start = i;
        while (i < s.len and s[i] != ' ' and s[i] >= 'a' and s[i] <= 'z') i += 1;
        // Also handle uppercase
        if (i == unit_start) {
            while (i < s.len and s[i] != ' ' and ((s[i] >= 'a' and s[i] <= 'z') or (s[i] >= 'A' and s[i] <= 'Z'))) i += 1;
        }
        const unit = s[unit_start..i];

        if (startsWithI(unit, "year")) {
            months += value * 12;
        } else if (startsWithI(unit, "mon")) {
            months += value;
        } else if (startsWithI(unit, "week")) {
            days += value * 7;
        } else if (startsWithI(unit, "day")) {
            days += value;
        } else if (startsWithI(unit, "hour")) {
            micros += @as(i64, value) * MICROS_PER_HOUR;
        } else if (startsWithI(unit, "min")) {
            micros += @as(i64, value) * MICROS_PER_MINUTE;
        } else if (startsWithI(unit, "sec")) {
            micros += @as(i64, value) * MICROS_PER_SECOND;
        } else {
            return null;
        }
    }

    return .{ .months = months, .days = days, .micros = micros };
}

fn isTimeComponent(s: []const u8) bool {
    // Check for HH:MM pattern
    if (s.len < 5) return false;
    return s[0] >= '0' and s[0] <= '9' and
        s[1] >= '0' and s[1] <= '9' and s[2] == ':';
}

fn parseTimeComponent(s: []const u8) ?i64 {
    if (s.len < 5) return null;
    const h = std.fmt.parseInt(i64, s[0..2], 10) catch return null;
    if (s[2] != ':') return null;
    const m = std.fmt.parseInt(i64, s[3..5], 10) catch return null;
    var sec: i64 = 0;
    if (s.len >= 8 and s[5] == ':') {
        sec = std.fmt.parseInt(i64, s[6..8], 10) catch return null;
    }
    return h * MICROS_PER_HOUR + m * MICROS_PER_MINUTE + sec * MICROS_PER_SECOND;
}

fn startsWithI(s: []const u8, prefix: []const u8) bool {
    if (s.len < prefix.len) return false;
    for (s[0..prefix.len], prefix) |a, b| {
        if (std.ascii.toLower(a) != std.ascii.toLower(b)) return false;
    }
    return true;
}

/// Add months to a date, clamping day to the last valid day of the resulting month.
fn addMonthsToDate(date_days: i32, month_delta: i32) i32 {
    const date = daysToDate(date_days);
    var new_month = @as(i32, date.month) + month_delta;
    var new_year = date.year;
    // Normalize month to 1..12
    new_year += @divFloor(new_month - 1, 12);
    new_month = @mod(new_month - 1, 12) + 1;
    const max_day = daysInMonth(@intCast(new_month), new_year);
    const new_day = @min(date.day, max_day);
    return dateToDays(new_year, @intCast(new_month), new_day);
}

/// Add interval to timestamp: months first (calendar), then days, then microseconds.
fn addIntervalToTimestamp(ts: i64, iv: Value.Interval) i64 {
    // Extract date and time parts
    const day_part: i32 = @intCast(@divTrunc(ts, MICROS_PER_DAY));
    const time_part = @mod(ts, MICROS_PER_DAY);
    // Add months (calendar arithmetic)
    const after_months = addMonthsToDate(day_part, iv.months);
    // Add days
    const after_days = after_months + iv.days;
    // Reconstruct timestamp with time component
    return @as(i64, after_days) * MICROS_PER_DAY + time_part + iv.micros;
}

fn negateInterval(iv: Value.Interval) Value.Interval {
    return .{ .months = -iv.months, .days = -iv.days, .micros = -iv.micros };
}

// ── Numeric Helpers ──────────────────────────────────────────────────────

/// Compute 10^exp as i128.
fn powI128(base: i128, exp: u8) i128 {
    var result: i128 = 1;
    var i: u8 = 0;
    while (i < exp) : (i += 1) {
        result *%= base;
    }
    return result;
}

/// Convert integer to Numeric with given scale.
fn intToNumeric(v: i64, scale: u8) Value.Numeric {
    return .{ .value = @as(i128, v) *% powI128(10, scale), .scale = scale };
}

/// Convert f64 to Numeric with given scale.
fn realToNumeric(v: f64, scale: u8) Value.Numeric {
    const factor: f64 = @floatFromInt(powI128(10, scale));
    const scaled: i128 = @intFromFloat(v * factor);
    return .{ .value = scaled, .scale = scale };
}

/// Convert Numeric to f64.
fn numericToReal(n: Value.Numeric) f64 {
    const factor: f64 = @floatFromInt(powI128(10, n.scale));
    return @as(f64, @floatFromInt(n.value)) / factor;
}

/// Align two numerics to the same scale (the larger one).
fn alignScale(a: Value.Numeric, b: Value.Numeric) struct { av: i128, bv: i128, scale: u8 } {
    if (a.scale == b.scale) return .{ .av = a.value, .bv = b.value, .scale = a.scale };
    if (a.scale > b.scale) {
        const diff = a.scale - b.scale;
        return .{ .av = a.value, .bv = b.value *% powI128(10, diff), .scale = a.scale };
    } else {
        const diff = b.scale - a.scale;
        return .{ .av = a.value *% powI128(10, diff), .bv = b.value, .scale = b.scale };
    }
}

/// Compare two Numeric values.
fn compareNumeric(a: Value.Numeric, b: Value.Numeric) std.math.Order {
    const aligned = alignScale(a, b);
    return std.math.order(aligned.av, aligned.bv);
}

/// Parse a decimal string like "123.45" into Numeric.
fn parseNumericString(s: []const u8) ?Value.Numeric {
    if (s.len == 0) return null;

    var start: usize = 0;
    var negative = false;
    if (s[0] == '-') {
        negative = true;
        start = 1;
    } else if (s[0] == '+') {
        start = 1;
    }
    if (start >= s.len) return null;

    var int_part: i128 = 0;
    var frac_part: i128 = 0;
    var scale: u8 = 0;
    var in_frac = false;

    for (s[start..]) |c| {
        if (c == '.') {
            if (in_frac) return null; // double dot
            in_frac = true;
            continue;
        }
        if (c < '0' or c > '9') return null;
        const digit: i128 = c - '0';
        if (in_frac) {
            if (scale >= 38) return null; // max precision
            frac_part = frac_part * 10 + digit;
            scale += 1;
        } else {
            int_part = int_part * 10 + digit;
        }
    }

    var value = int_part * powI128(10, scale) + frac_part;
    if (negative) value = -value;
    return .{ .value = value, .scale = scale };
}

/// Format a Numeric value as a string (e.g., 12345 with scale=2 → "123.45").
pub fn formatNumeric(allocator: Allocator, n: Value.Numeric) ![]u8 {
    if (n.scale == 0) {
        return std.fmt.allocPrint(allocator, "{d}", .{n.value});
    }

    const is_negative = n.value < 0;
    const abs_val = if (is_negative) -n.value else n.value;
    const divisor = powI128(10, n.scale);
    const int_part = @divTrunc(abs_val, divisor);
    const frac_part = @mod(abs_val, divisor);

    // Format fractional part with leading zeros
    const frac_str = try std.fmt.allocPrint(allocator, "{d}", .{frac_part});
    defer allocator.free(frac_str);

    // Pad with leading zeros if needed
    const scale_usize: usize = @intCast(n.scale);
    if (is_negative) {
        if (frac_str.len >= scale_usize) {
            return std.fmt.allocPrint(allocator, "-{d}.{s}", .{ int_part, frac_str });
        } else {
            const pad = scale_usize - frac_str.len;
            const padded = try allocator.alloc(u8, scale_usize);
            defer allocator.free(padded);
            @memset(padded[0..pad], '0');
            @memcpy(padded[pad..], frac_str);
            return std.fmt.allocPrint(allocator, "-{d}.{s}", .{ int_part, padded });
        }
    } else {
        if (frac_str.len >= scale_usize) {
            return std.fmt.allocPrint(allocator, "{d}.{s}", .{ int_part, frac_str });
        } else {
            const pad = scale_usize - frac_str.len;
            const padded = try allocator.alloc(u8, scale_usize);
            defer allocator.free(padded);
            @memset(padded[0..pad], '0');
            @memcpy(padded[pad..], frac_str);
            return std.fmt.allocPrint(allocator, "{d}.{s}", .{ int_part, padded });
        }
    }
}

// ── UUID Helpers ─────────────────────────────────────────────────────────

/// Format a UUID as "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx".
pub fn formatUuid(allocator: Allocator, bytes: [16]u8) ![]u8 {
    const hex = "0123456789abcdef";
    const result = try allocator.alloc(u8, 36);
    var pos: usize = 0;
    for (bytes, 0..) |byte, i| {
        result[pos] = hex[byte >> 4];
        result[pos + 1] = hex[byte & 0x0f];
        pos += 2;
        if (i == 3 or i == 5 or i == 7 or i == 9) {
            result[pos] = '-';
            pos += 1;
        }
    }
    return result;
}

/// Parse a UUID string "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx" into 16 bytes.
fn parseUuidString(s: []const u8) ?[16]u8 {
    // Accept with or without dashes
    var bytes: [16]u8 = undefined;
    var bi: usize = 0;
    var i: usize = 0;
    while (i < s.len and bi < 16) {
        if (s[i] == '-') {
            i += 1;
            continue;
        }
        if (i + 1 >= s.len) return null;
        const hi: u8 = hexDigit(s[i]) orelse return null;
        const lo: u8 = hexDigit(s[i + 1]) orelse return null;
        bytes[bi] = (hi << 4) | lo;
        bi += 1;
        i += 2;
    }
    if (bi != 16) return null;
    return bytes;
}

fn hexDigit(c: u8) ?u4 {
    if (c >= '0' and c <= '9') return @intCast(c - '0');
    if (c >= 'a' and c <= 'f') return @intCast(c - 'a' + 10);
    if (c >= 'A' and c <= 'F') return @intCast(c - 'A' + 10);
    return null;
}

/// Generate a random v4 UUID.
fn generateUuidV4() [16]u8 {
    var bytes: [16]u8 = undefined;
    std.crypto.random.bytes(&bytes);
    // Set version 4 (bits 48-51)
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    // Set variant 2 (bits 64-65)
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    return bytes;
}

// ── MVCC Context ──────────────────────────────────────────────────────────

/// MVCC context passed to scan operators for visibility filtering.
/// When null/disabled, all rows are visible (legacy/auto-commit mode).
pub const MvccContext = struct {
    snapshot: Snapshot,
    current_xid: u32,
    current_cid: u16,
    /// When true, rows carry MVCC headers and need visibility checks.
    enabled: bool = true,
    /// Optional reference to TransactionManager for commit/abort status lookup.
    /// Enables correct visibility for tuples without hint flags (e.g., aborted txns).
    tm: ?*mvcc_mod.TransactionManager = null,
};

// ── Value Type ──────────────────────────────────────────────────────────

/// A runtime value in the executor.
pub const Value = union(enum) {
    integer: i64,
    real: f64,
    text: []const u8,
    blob: []const u8,
    boolean: bool,
    date: i32, // days since epoch (1970-01-01)
    time: i64, // microseconds since midnight
    timestamp: i64, // microseconds since epoch
    interval: Interval, // composite: months + days + microseconds
    numeric: Numeric, // fixed-point decimal
    uuid: [16]u8, // 128-bit UUID
    null_value,

    pub const Interval = struct {
        months: i32,
        days: i32,
        micros: i64,
    };

    pub const Numeric = struct {
        value: i128, // unscaled value (e.g., 12345 for 123.45 with scale=2)
        scale: u8, // number of decimal digits (0-38)
    };

    /// Compare two values. Returns .lt, .eq, or .gt.
    /// NULLs sort last (greater than any non-null value).
    pub fn compare(a: Value, b: Value) std.math.Order {
        // NULL handling: NULL > everything
        if (a == .null_value and b == .null_value) return .eq;
        if (a == .null_value) return .gt;
        if (b == .null_value) return .lt;

        // Same type comparison
        return switch (a) {
            .integer => |av| switch (b) {
                .integer => |bv| std.math.order(av, bv),
                .real => |bv| std.math.order(@as(f64, @floatFromInt(av)), bv),
                else => .lt, // integers < text/blob/bool
            },
            .real => |av| switch (b) {
                .integer => |bv| std.math.order(av, @as(f64, @floatFromInt(bv))),
                .real => |bv| std.math.order(av, bv),
                else => .lt,
            },
            .text => |av| switch (b) {
                .text => |bv| std.mem.order(u8, av, bv),
                else => .gt, // text > numbers
            },
            .blob => |av| switch (b) {
                .blob => |bv| std.mem.order(u8, av, bv),
                else => .gt,
            },
            .boolean => |av| switch (b) {
                .boolean => |bv| {
                    if (av == bv) return .eq;
                    if (!av and bv) return .lt;
                    return .gt;
                },
                else => .gt,
            },
            .date => |av| switch (b) {
                .date => |bv| std.math.order(av, bv),
                .timestamp => |bv| {
                    // Convert date to timestamp (midnight) and compare
                    const a_ts = @as(i64, av) * MICROS_PER_DAY;
                    return std.math.order(a_ts, bv);
                },
                else => .lt, // dates < time/text/blob/bool
            },
            .time => |av| switch (b) {
                .time => |bv| std.math.order(av, bv),
                else => .lt, // time < text/blob/bool
            },
            .timestamp => |av| switch (b) {
                .timestamp => |bv| std.math.order(av, bv),
                .date => |bv| {
                    // Convert date to timestamp (midnight) and compare
                    const b_ts = @as(i64, bv) * MICROS_PER_DAY;
                    return std.math.order(av, b_ts);
                },
                else => .lt, // timestamps < text/blob/bool
            },
            .interval => |av| switch (b) {
                .interval => |bv| {
                    // Compare months first, then days, then microseconds
                    const m = std.math.order(av.months, bv.months);
                    if (m != .eq) return m;
                    const d = std.math.order(av.days, bv.days);
                    if (d != .eq) return d;
                    return std.math.order(av.micros, bv.micros);
                },
                else => .lt, // intervals < text/blob/bool
            },
            .numeric => |av| switch (b) {
                .numeric => |bv| compareNumeric(av, bv),
                .integer => |bv| compareNumeric(av, intToNumeric(bv, av.scale)),
                .real => |bv| {
                    const bn = realToNumeric(bv, av.scale);
                    return compareNumeric(av, bn);
                },
                else => .lt,
            },
            .uuid => |av| switch (b) {
                .uuid => |bv| std.mem.order(u8, &av, &bv),
                else => .lt,
            },
            .null_value => unreachable,
        };
    }

    /// Check if two values are equal.
    pub fn eql(a: Value, b: Value) bool {
        return a.compare(b) == .eq;
    }

    /// Check if value is truthy (for boolean evaluation).
    pub fn isTruthy(self: Value) bool {
        return switch (self) {
            .integer => |v| v != 0,
            .real => |v| v != 0.0,
            .text => |v| v.len > 0,
            .blob => |v| v.len > 0,
            .boolean => |v| v,
            .date, .time, .timestamp, .interval => true, // temporal values are always truthy
            .numeric => |v| v.value != 0,
            .uuid => true, // UUID values are always truthy
            .null_value => false,
        };
    }

    /// Convert to integer if possible.
    pub fn toInteger(self: Value) ?i64 {
        return switch (self) {
            .integer => |v| v,
            .real => |v| @intFromFloat(v),
            .boolean => |v| if (v) @as(i64, 1) else 0,
            .text => |v| std.fmt.parseInt(i64, v, 10) catch null,
            .date => |v| @as(i64, v), // days as integer
            .time, .timestamp => |v| v, // microseconds as integer
            .interval => |v| v.micros, // total microseconds component
            .numeric => |v| @as(i64, @intCast(@divTrunc(v.value, powI128(10, v.scale)))),
            else => null,
        };
    }

    /// Convert to float if possible.
    pub fn toReal(self: Value) ?f64 {
        return switch (self) {
            .integer => |v| @floatFromInt(v),
            .real => |v| v,
            .boolean => |v| if (v) @as(f64, 1.0) else 0.0,
            .text => |v| std.fmt.parseFloat(f64, v) catch null,
            .date => |v| @floatFromInt(v),
            .time, .timestamp => |v| @floatFromInt(v),
            .interval => |v| @floatFromInt(v.micros),
            .numeric => |v| numericToReal(v),
            else => null,
        };
    }

    /// Duplicate a value, allocating copies of text/blob data.
    pub fn dupe(self: Value, allocator: Allocator) !Value {
        return switch (self) {
            .text => |v| .{ .text = try allocator.dupe(u8, v) },
            .blob => |v| .{ .blob = try allocator.dupe(u8, v) },
            else => self,
        };
    }

    /// Free heap-allocated value data.
    pub fn free(self: Value, allocator: Allocator) void {
        switch (self) {
            .text => |v| allocator.free(v),
            .blob => |v| allocator.free(v),
            else => {},
        }
    }
};

// ── Row ─────────────────────────────────────────────────────────────────

/// A row is a sequence of named column values.
pub const Row = struct {
    /// Column names (not owned — references schema or plan).
    columns: []const []const u8,
    /// Column values (owned by this row).
    values: []Value,
    allocator: Allocator,

    pub fn deinit(self: *Row) void {
        for (self.values) |v| v.free(self.allocator);
        self.allocator.free(self.values);
        self.allocator.free(self.columns);
    }

    /// Look up a column value by name (case-insensitive).
    pub fn getColumn(self: *const Row, name: []const u8) ?Value {
        for (self.columns, 0..) |col, i| {
            if (std.ascii.eqlIgnoreCase(col, name)) return self.values[i];
        }
        return null;
    }

    /// Look up a column value by qualified name (table.column).
    pub fn getQualifiedColumn(self: *const Row, table: []const u8, column: []const u8) ?Value {
        // Try "table.column" format first
        for (self.columns, 0..) |col, i| {
            // Check if column name is "table.column"
            if (std.mem.indexOf(u8, col, ".")) |dot_pos| {
                const col_table = col[0..dot_pos];
                const col_name = col[dot_pos + 1 ..];
                if (std.ascii.eqlIgnoreCase(col_table, table) and std.ascii.eqlIgnoreCase(col_name, column)) {
                    return self.values[i];
                }
            }
        }
        // Fall back to unqualified lookup
        return self.getColumn(column);
    }

    /// Create a deep copy of this row.
    pub fn clone(self: *const Row, allocator: Allocator) !Row {
        const cols = try allocator.alloc([]const u8, self.columns.len);
        errdefer allocator.free(cols);
        for (self.columns, 0..) |c, i| {
            cols[i] = c;
        }

        const vals = try allocator.alloc(Value, self.values.len);
        errdefer allocator.free(vals);
        var inited: usize = 0;
        errdefer for (vals[0..inited]) |v| v.free(allocator);
        for (self.values, 0..) |v, i| {
            vals[i] = try v.dupe(allocator);
            inited += 1;
        }

        return .{
            .columns = cols,
            .values = vals,
            .allocator = allocator,
        };
    }
};

// ── Row Serialization ───────────────────────────────────────────────────

/// Serialize a row's values into bytes for B+Tree storage.
/// Format: [col_count: u16] for each col: [type_tag: u8][value_data...]
///   integer: 8 bytes (i64 little-endian)
///   real: 8 bytes (f64 little-endian)
///   text: [len: u32][bytes...]
///   blob: [len: u32][bytes...]
///   boolean: 1 byte (0 or 1)
///   null: 0 bytes (tag only)
pub fn serializeRow(allocator: Allocator, values: []const Value) ![]u8 {
    var size: usize = 2; // col_count
    for (values) |v| {
        size += 1; // type tag
        switch (v) {
            .integer => size += 8,
            .real => size += 8,
            .text => |t| size += 4 + t.len,
            .blob => |b| size += 4 + b.len,
            .boolean => size += 1,
            .date => size += 4,
            .time => size += 8,
            .timestamp => size += 8,
            .interval => size += 16, // months(4) + days(4) + micros(8)
            .numeric => size += 17, // scale(1) + value(16)
            .uuid => size += 16, // 128-bit UUID
            .null_value => {},
        }
    }

    const buf = try allocator.alloc(u8, size);
    errdefer allocator.free(buf);
    var pos: usize = 0;

    std.mem.writeInt(u16, buf[pos..][0..2], @intCast(values.len), .little);
    pos += 2;

    for (values) |v| {
        switch (v) {
            .integer => |i| {
                buf[pos] = 0x01;
                pos += 1;
                std.mem.writeInt(i64, buf[pos..][0..8], i, .little);
                pos += 8;
            },
            .real => |r| {
                buf[pos] = 0x02;
                pos += 1;
                std.mem.writeInt(u64, buf[pos..][0..8], @bitCast(r), .little);
                pos += 8;
            },
            .text => |t| {
                buf[pos] = 0x03;
                pos += 1;
                std.mem.writeInt(u32, buf[pos..][0..4], @intCast(t.len), .little);
                pos += 4;
                @memcpy(buf[pos..][0..t.len], t);
                pos += t.len;
            },
            .blob => |b| {
                buf[pos] = 0x04;
                pos += 1;
                std.mem.writeInt(u32, buf[pos..][0..4], @intCast(b.len), .little);
                pos += 4;
                @memcpy(buf[pos..][0..b.len], b);
                pos += b.len;
            },
            .boolean => |b| {
                buf[pos] = 0x05;
                pos += 1;
                buf[pos] = if (b) 1 else 0;
                pos += 1;
            },
            .date => |d| {
                buf[pos] = 0x06;
                pos += 1;
                std.mem.writeInt(i32, buf[pos..][0..4], d, .little);
                pos += 4;
            },
            .time => |t| {
                buf[pos] = 0x07;
                pos += 1;
                std.mem.writeInt(i64, buf[pos..][0..8], t, .little);
                pos += 8;
            },
            .timestamp => |ts| {
                buf[pos] = 0x08;
                pos += 1;
                std.mem.writeInt(i64, buf[pos..][0..8], ts, .little);
                pos += 8;
            },
            .interval => |iv| {
                buf[pos] = 0x09;
                pos += 1;
                std.mem.writeInt(i32, buf[pos..][0..4], iv.months, .little);
                pos += 4;
                std.mem.writeInt(i32, buf[pos..][0..4], iv.days, .little);
                pos += 4;
                std.mem.writeInt(i64, buf[pos..][0..8], iv.micros, .little);
                pos += 8;
            },
            .numeric => |n| {
                buf[pos] = 0x0A;
                pos += 1;
                buf[pos] = n.scale;
                pos += 1;
                std.mem.writeInt(i128, buf[pos..][0..16], n.value, .little);
                pos += 16;
            },
            .uuid => |u| {
                buf[pos] = 0x0B;
                pos += 1;
                @memcpy(buf[pos..][0..16], &u);
                pos += 16;
            },
            .null_value => {
                buf[pos] = 0x00;
                pos += 1;
            },
        }
    }

    std.debug.assert(pos == size);
    return buf;
}

/// Deserialize a row's values from B+Tree storage bytes.
pub fn deserializeRow(allocator: Allocator, data: []const u8) ![]Value {
    if (data.len < 2) return error.InvalidRowData;
    var pos: usize = 0;

    const col_count = std.mem.readInt(u16, data[pos..][0..2], .little);
    pos += 2;

    const values = try allocator.alloc(Value, col_count);
    var inited: usize = 0;
    errdefer {
        for (values[0..inited]) |v| v.free(allocator);
        allocator.free(values);
    }

    for (values) |*v| {
        if (pos >= data.len) return error.InvalidRowData;
        const tag = data[pos];
        pos += 1;

        switch (tag) {
            0x01 => { // integer
                if (pos + 8 > data.len) return error.InvalidRowData;
                v.* = .{ .integer = std.mem.readInt(i64, data[pos..][0..8], .little) };
                pos += 8;
            },
            0x02 => { // real
                if (pos + 8 > data.len) return error.InvalidRowData;
                v.* = .{ .real = @bitCast(std.mem.readInt(u64, data[pos..][0..8], .little)) };
                pos += 8;
            },
            0x03 => { // text
                if (pos + 4 > data.len) return error.InvalidRowData;
                const len = std.mem.readInt(u32, data[pos..][0..4], .little);
                pos += 4;
                if (pos + len > data.len) return error.InvalidRowData;
                v.* = .{ .text = try allocator.dupe(u8, data[pos..][0..len]) };
                pos += len;
            },
            0x04 => { // blob
                if (pos + 4 > data.len) return error.InvalidRowData;
                const len = std.mem.readInt(u32, data[pos..][0..4], .little);
                pos += 4;
                if (pos + len > data.len) return error.InvalidRowData;
                v.* = .{ .blob = try allocator.dupe(u8, data[pos..][0..len]) };
                pos += len;
            },
            0x05 => { // boolean
                if (pos >= data.len) return error.InvalidRowData;
                v.* = .{ .boolean = data[pos] != 0 };
                pos += 1;
            },
            0x06 => { // date
                if (pos + 4 > data.len) return error.InvalidRowData;
                v.* = .{ .date = std.mem.readInt(i32, data[pos..][0..4], .little) };
                pos += 4;
            },
            0x07 => { // time
                if (pos + 8 > data.len) return error.InvalidRowData;
                v.* = .{ .time = std.mem.readInt(i64, data[pos..][0..8], .little) };
                pos += 8;
            },
            0x08 => { // timestamp
                if (pos + 8 > data.len) return error.InvalidRowData;
                v.* = .{ .timestamp = std.mem.readInt(i64, data[pos..][0..8], .little) };
                pos += 8;
            },
            0x09 => { // interval
                if (pos + 16 > data.len) return error.InvalidRowData;
                const months = std.mem.readInt(i32, data[pos..][0..4], .little);
                pos += 4;
                const days = std.mem.readInt(i32, data[pos..][0..4], .little);
                pos += 4;
                const micros = std.mem.readInt(i64, data[pos..][0..8], .little);
                pos += 8;
                v.* = .{ .interval = .{ .months = months, .days = days, .micros = micros } };
            },
            0x0A => { // numeric
                if (pos + 17 > data.len) return error.InvalidRowData;
                const scale = data[pos];
                pos += 1;
                const value = std.mem.readInt(i128, data[pos..][0..16], .little);
                pos += 16;
                v.* = .{ .numeric = .{ .value = value, .scale = scale } };
            },
            0x0B => { // uuid
                if (pos + 16 > data.len) return error.InvalidRowData;
                var uuid_bytes: [16]u8 = undefined;
                @memcpy(&uuid_bytes, data[pos..][0..16]);
                pos += 16;
                v.* = .{ .uuid = uuid_bytes };
            },
            0x00 => { // null
                v.* = .null_value;
            },
            else => return error.InvalidRowData,
        }
        inited += 1;
    }

    return values;
}

// ── Expression Evaluator ────────────────────────────────────────────────

pub const EvalError = error{
    OutOfMemory,
    TypeError,
    DivisionByZero,
    ColumnNotFound,
    UnsupportedExpression,
};

/// Evaluate an AST expression against a row, producing a Value.
pub fn evalExpr(allocator: Allocator, expr: *const ast.Expr, row: *const Row) EvalError!Value {
    switch (expr.*) {
        .integer_literal => |v| return .{ .integer = v },
        .float_literal => |v| return .{ .real = v },
        .string_literal => |v| return .{ .text = try allocator.dupe(u8, v) },
        .boolean_literal => |v| return .{ .boolean = v },
        .null_literal => return .null_value,

        .column_ref => |ref| {
            if (ref.prefix) |table| {
                return (row.getQualifiedColumn(table, ref.name) orelse
                    return EvalError.ColumnNotFound).dupe(allocator) catch return EvalError.OutOfMemory;
            }
            return (row.getColumn(ref.name) orelse
                return EvalError.ColumnNotFound).dupe(allocator) catch return EvalError.OutOfMemory;
        },

        .paren => |inner| return evalExpr(allocator, inner, row),

        .unary_op => |op| {
            const operand = try evalExpr(allocator, op.operand, row);
            defer operand.free(allocator);
            return evalUnaryOp(op.op, operand);
        },

        .binary_op => |op| {
            const left = try evalExpr(allocator, op.left, row);
            defer left.free(allocator);
            const right = try evalExpr(allocator, op.right, row);
            defer right.free(allocator);
            return evalBinaryOp(allocator, op.op, left, right);
        },

        .is_null => |is| {
            const val = try evalExpr(allocator, is.expr, row);
            defer val.free(allocator);
            const result = val == .null_value;
            return .{ .boolean = if (is.negated) !result else result };
        },

        .between => |bt| {
            const val = try evalExpr(allocator, bt.expr, row);
            defer val.free(allocator);
            const low = try evalExpr(allocator, bt.low, row);
            defer low.free(allocator);
            const high = try evalExpr(allocator, bt.high, row);
            defer high.free(allocator);
            const in_range = val.compare(low) != .lt and val.compare(high) != .gt;
            return .{ .boolean = if (bt.negated) !in_range else in_range };
        },

        .in_list => |il| {
            const val = try evalExpr(allocator, il.expr, row);
            defer val.free(allocator);
            var found = false;
            for (il.list) |item| {
                const item_val = try evalExpr(allocator, item, row);
                defer item_val.free(allocator);
                if (val.eql(item_val)) {
                    found = true;
                    break;
                }
            }
            return .{ .boolean = if (il.negated) !found else found };
        },

        .like => |lk| {
            const val = try evalExpr(allocator, lk.expr, row);
            defer val.free(allocator);
            const pat = try evalExpr(allocator, lk.pattern, row);
            defer pat.free(allocator);
            const text_val = switch (val) {
                .text => |t| t,
                else => return .{ .boolean = false },
            };
            const pattern = switch (pat) {
                .text => |t| t,
                else => return .{ .boolean = false },
            };
            const matches = likeMatch(text_val, pattern);
            return .{ .boolean = if (lk.negated) !matches else matches };
        },

        .case_expr => |ce| {
            if (ce.operand) |operand| {
                const op_val = try evalExpr(allocator, operand, row);
                defer op_val.free(allocator);
                for (ce.when_clauses) |wc| {
                    const when_val = try evalExpr(allocator, wc.condition, row);
                    defer when_val.free(allocator);
                    if (op_val.eql(when_val)) {
                        return evalExpr(allocator, wc.result, row);
                    }
                }
            } else {
                for (ce.when_clauses) |wc| {
                    const cond = try evalExpr(allocator, wc.condition, row);
                    defer cond.free(allocator);
                    if (cond.isTruthy()) {
                        return evalExpr(allocator, wc.result, row);
                    }
                }
            }
            if (ce.else_expr) |else_e| {
                return evalExpr(allocator, else_e, row);
            }
            return .null_value;
        },

        .cast => |c| {
            const val = try evalExpr(allocator, c.expr, row);
            defer val.free(allocator);
            return evalCast(allocator, val, c.target_type);
        },

        .function_call => |fc| {
            return evalFunctionCall(allocator, fc, row);
        },

        // Window function values are pre-computed by WindowOp and stored in the row.
        // When evalExpr encounters a window_function, the value should already be
        // in the row under the window function's alias/name.
        .window_function => return EvalError.UnsupportedExpression,

        // Unsupported in row-level evaluation (aggregates handled in AggregateExecutor)
        .blob_literal,
        .subquery,
        .bind_parameter,
        => return EvalError.UnsupportedExpression,
    }
}

fn evalUnaryOp(op: ast.UnaryOp, operand: Value) Value {
    return switch (op) {
        .negate => switch (operand) {
            .integer => |v| .{ .integer = -v },
            .real => |v| .{ .real = -v },
            else => .null_value,
        },
        .not => .{ .boolean = !operand.isTruthy() },
        .bitwise_not => switch (operand) {
            .integer => |v| .{ .integer = ~v },
            else => .null_value,
        },
    };
}

fn evalBinaryOp(allocator: Allocator, op: ast.BinaryOp, left: Value, right: Value) EvalError!Value {
    // NULL propagation for most ops
    if (left == .null_value or right == .null_value) {
        return switch (op) {
            .@"and" => blk: {
                // FALSE AND NULL = FALSE
                if (left == .boolean and !left.boolean) break :blk Value{ .boolean = false };
                if (right == .boolean and !right.boolean) break :blk Value{ .boolean = false };
                break :blk Value.null_value;
            },
            .@"or" => blk: {
                // TRUE OR NULL = TRUE
                if (left == .boolean and left.boolean) break :blk Value{ .boolean = true };
                if (right == .boolean and right.boolean) break :blk Value{ .boolean = true };
                break :blk Value.null_value;
            },
            else => .null_value,
        };
    }

    return switch (op) {
        // Arithmetic
        .add => evalArithmetic(left, right, .add),
        .subtract => evalArithmetic(left, right, .sub),
        .multiply => evalArithmetic(left, right, .mul),
        .divide => {
            if (right == .integer and right.integer == 0) return EvalError.DivisionByZero;
            if (right == .real and right.real == 0.0) return EvalError.DivisionByZero;
            return evalArithmetic(left, right, .div);
        },
        .modulo => {
            if (right == .integer and right.integer == 0) return EvalError.DivisionByZero;
            return switch (left) {
                .integer => |a| switch (right) {
                    .integer => |b| Value{ .integer = @mod(a, b) },
                    else => .null_value,
                },
                else => .null_value,
            };
        },

        // Comparison
        .equal => .{ .boolean = left.eql(right) },
        .not_equal => .{ .boolean = !left.eql(right) },
        .less_than => .{ .boolean = left.compare(right) == .lt },
        .greater_than => .{ .boolean = left.compare(right) == .gt },
        .less_than_or_equal => .{ .boolean = left.compare(right) != .gt },
        .greater_than_or_equal => .{ .boolean = left.compare(right) != .lt },

        // Logical
        .@"and" => .{ .boolean = left.isTruthy() and right.isTruthy() },
        .@"or" => .{ .boolean = left.isTruthy() or right.isTruthy() },

        // String concatenation
        .concat => blk: {
            const l = switch (left) {
                .text => |t| t,
                else => break :blk Value.null_value,
            };
            const r = switch (right) {
                .text => |t| t,
                else => break :blk Value.null_value,
            };
            const result = allocator.alloc(u8, l.len + r.len) catch return EvalError.OutOfMemory;
            @memcpy(result[0..l.len], l);
            @memcpy(result[l.len..], r);
            break :blk Value{ .text = result };
        },

        // Bitwise
        .bitwise_and => switch (left) {
            .integer => |a| switch (right) {
                .integer => |b| Value{ .integer = a & b },
                else => .null_value,
            },
            else => .null_value,
        },
        .bitwise_or => switch (left) {
            .integer => |a| switch (right) {
                .integer => |b| Value{ .integer = a | b },
                else => .null_value,
            },
            else => .null_value,
        },
        .left_shift => switch (left) {
            .integer => |a| switch (right) {
                .integer => |b| Value{ .integer = if (b >= 0 and b < 64) a << @intCast(b) else 0 },
                else => .null_value,
            },
            else => .null_value,
        },
        .right_shift => switch (left) {
            .integer => |a| switch (right) {
                .integer => |b| Value{ .integer = if (b >= 0 and b < 64) a >> @intCast(b) else 0 },
                else => .null_value,
            },
            else => .null_value,
        },
    };
}

const ArithOp = enum { add, sub, mul, div };

fn evalArithmetic(left: Value, right: Value, op: ArithOp) Value {
    // Date/timestamp arithmetic with integers
    if (left == .date and right == .integer) {
        if (op == .add) return .{ .date = left.date +% @as(i32, @intCast(right.integer)) };
        if (op == .sub) return .{ .date = left.date -% @as(i32, @intCast(right.integer)) };
    }
    if (left == .date and right == .date and op == .sub) {
        return .{ .integer = @as(i64, left.date) - @as(i64, right.date) };
    }
    if (left == .timestamp and right == .timestamp and op == .sub) {
        return .{ .integer = left.timestamp - right.timestamp };
    }

    // Interval arithmetic
    if (left == .interval and right == .interval) {
        const a = left.interval;
        const b = right.interval;
        if (op == .add) return .{ .interval = .{ .months = a.months +% b.months, .days = a.days +% b.days, .micros = a.micros +% b.micros } };
        if (op == .sub) return .{ .interval = .{ .months = a.months -% b.months, .days = a.days -% b.days, .micros = a.micros -% b.micros } };
    }

    // date +/- interval → date (if no time component) or timestamp (if time component)
    if (left == .date and right == .interval) {
        const iv = right.interval;
        if (iv.micros != 0) {
            // Has time component → upcast to TIMESTAMP
            const ts = @as(i64, left.date) * MICROS_PER_DAY;
            if (op == .add) return .{ .timestamp = addIntervalToTimestamp(ts, iv) };
            if (op == .sub) return .{ .timestamp = addIntervalToTimestamp(ts, negateInterval(iv)) };
        }
        const adjusted = addMonthsToDate(left.date, if (op == .sub) -iv.months else iv.months);
        if (op == .add) return .{ .date = adjusted +% iv.days };
        if (op == .sub) return .{ .date = adjusted -% iv.days };
    }

    // timestamp +/- interval → timestamp
    if (left == .timestamp and right == .interval) {
        const iv = right.interval;
        if (op == .add) return .{ .timestamp = addIntervalToTimestamp(left.timestamp, iv) };
        if (op == .sub) return .{ .timestamp = addIntervalToTimestamp(left.timestamp, negateInterval(iv)) };
    }

    // interval + date/timestamp (commutative for add)
    if (left == .interval and right == .date and op == .add) {
        const iv = left.interval;
        if (iv.micros != 0) {
            // Has time component → upcast to TIMESTAMP
            const ts = @as(i64, right.date) * MICROS_PER_DAY;
            return .{ .timestamp = addIntervalToTimestamp(ts, iv) };
        }
        return .{ .date = addMonthsToDate(right.date, iv.months) +% iv.days };
    }
    if (left == .interval and right == .timestamp and op == .add) {
        return .{ .timestamp = addIntervalToTimestamp(right.timestamp, left.interval) };
    }

    // interval * integer / integer * interval
    if (left == .interval and right == .integer) {
        const iv = left.interval;
        const n = @as(i32, @intCast(right.integer));
        if (op == .mul) return .{ .interval = .{ .months = iv.months *% n, .days = iv.days *% n, .micros = iv.micros *% @as(i64, right.integer) } };
        if (op == .div) return .{ .interval = .{ .months = @divTrunc(iv.months, n), .days = @divTrunc(iv.days, n), .micros = @divTrunc(iv.micros, right.integer) } };
    }
    if (left == .integer and right == .interval and op == .mul) {
        const iv = right.interval;
        const n = @as(i32, @intCast(left.integer));
        return .{ .interval = .{ .months = iv.months *% n, .days = iv.days *% n, .micros = iv.micros *% left.integer } };
    }

    // Numeric arithmetic
    if (left == .numeric and right == .numeric) {
        const aligned = alignScale(left.numeric, right.numeric);
        return switch (op) {
            .add => .{ .numeric = .{ .value = aligned.av +% aligned.bv, .scale = aligned.scale } },
            .sub => .{ .numeric = .{ .value = aligned.av -% aligned.bv, .scale = aligned.scale } },
            .mul => .{ .numeric = .{ .value = @divTrunc(aligned.av *% aligned.bv, powI128(10, aligned.scale)), .scale = aligned.scale } },
            .div => .{ .numeric = .{ .value = @divTrunc(aligned.av *% powI128(10, aligned.scale), aligned.bv), .scale = aligned.scale } },
        };
    }
    // numeric op integer → numeric
    if (left == .numeric and right == .integer) {
        const bn = intToNumeric(right.integer, left.numeric.scale);
        const aligned = alignScale(left.numeric, bn);
        return switch (op) {
            .add => .{ .numeric = .{ .value = aligned.av +% aligned.bv, .scale = aligned.scale } },
            .sub => .{ .numeric = .{ .value = aligned.av -% aligned.bv, .scale = aligned.scale } },
            .mul => .{ .numeric = .{ .value = @divTrunc(aligned.av *% aligned.bv, powI128(10, aligned.scale)), .scale = aligned.scale } },
            .div => .{ .numeric = .{ .value = @divTrunc(aligned.av *% powI128(10, aligned.scale), aligned.bv), .scale = aligned.scale } },
        };
    }
    // integer op numeric → numeric
    if (left == .integer and right == .numeric) {
        const an = intToNumeric(left.integer, right.numeric.scale);
        const aligned = alignScale(an, right.numeric);
        return switch (op) {
            .add => .{ .numeric = .{ .value = aligned.av +% aligned.bv, .scale = aligned.scale } },
            .sub => .{ .numeric = .{ .value = aligned.av -% aligned.bv, .scale = aligned.scale } },
            .mul => .{ .numeric = .{ .value = @divTrunc(aligned.av *% aligned.bv, powI128(10, aligned.scale)), .scale = aligned.scale } },
            .div => .{ .numeric = .{ .value = @divTrunc(aligned.av *% powI128(10, aligned.scale), aligned.bv), .scale = aligned.scale } },
        };
    }

    // Try integer arithmetic
    if (left == .integer and right == .integer) {
        const a = left.integer;
        const b = right.integer;
        return .{ .integer = switch (op) {
            .add => a +% b,
            .sub => a -% b,
            .mul => a *% b,
            .div => @divTrunc(a, b),
        } };
    }

    // Fall back to float
    const a = left.toReal() orelse return .null_value;
    const b = right.toReal() orelse return .null_value;
    return .{ .real = switch (op) {
        .add => a + b,
        .sub => a - b,
        .mul => a * b,
        .div => a / b,
    } };
}

fn evalCast(allocator: Allocator, val: Value, target: ast.DataType) EvalError!Value {
    return switch (target) {
        .type_integer, .type_int, .type_serial, .type_bigserial => .{ .integer = val.toInteger() orelse return .null_value },
        .type_real => .{ .real = val.toReal() orelse return .null_value },
        .type_text, .type_varchar => blk: {
            const s = switch (val) {
                .integer => |v| std.fmt.allocPrint(allocator, "{d}", .{v}) catch return EvalError.OutOfMemory,
                .real => |v| std.fmt.allocPrint(allocator, "{d}", .{v}) catch return EvalError.OutOfMemory,
                .boolean => |v| (allocator.dupe(u8, if (v) "true" else "false") catch return EvalError.OutOfMemory),
                .text => |v| (allocator.dupe(u8, v) catch return EvalError.OutOfMemory),
                .date => |v| formatDate(allocator, v) catch return EvalError.OutOfMemory,
                .time => |v| formatTime(allocator, v) catch return EvalError.OutOfMemory,
                .timestamp => |v| formatTimestamp(allocator, v) catch return EvalError.OutOfMemory,
                .interval => |v| formatInterval(allocator, v) catch return EvalError.OutOfMemory,
                .numeric => |v| formatNumeric(allocator, v) catch return EvalError.OutOfMemory,
                .uuid => |v| formatUuid(allocator, v) catch return EvalError.OutOfMemory,
                .null_value => return .null_value,
                .blob => return .null_value,
            };
            break :blk Value{ .text = s };
        },
        .type_boolean => .{ .boolean = val.isTruthy() },
        .type_blob => .null_value,
        .type_date => blk: {
            const days = switch (val) {
                .text => |v| parseDateString(v) orelse return .null_value,
                .timestamp => |v| @as(i32, @intCast(@divTrunc(v, MICROS_PER_DAY))),
                .date => |v| v,
                else => return .null_value,
            };
            break :blk Value{ .date = days };
        },
        .type_time => blk: {
            const micros = switch (val) {
                .text => |v| parseTimeString(v) orelse return .null_value,
                .time => |v| v,
                else => return .null_value,
            };
            break :blk Value{ .time = micros };
        },
        .type_timestamp => blk: {
            const ts = switch (val) {
                .text => |v| parseTimestampString(v) orelse return .null_value,
                .date => |v| @as(i64, v) * MICROS_PER_DAY, // date at midnight
                .timestamp => |v| v,
                else => return .null_value,
            };
            break :blk Value{ .timestamp = ts };
        },
        .type_interval => blk: {
            const iv = switch (val) {
                .text => |v| parseIntervalString(v) orelse return .null_value,
                .interval => |v| v,
                .integer => |v| Value.Interval{ .months = 0, .days = 0, .micros = v * MICROS_PER_SECOND },
                else => return .null_value,
            };
            break :blk Value{ .interval = iv };
        },
        .type_numeric, .type_decimal => blk: {
            const n = switch (val) {
                .text => |v| parseNumericString(v) orelse return .null_value,
                .integer => |v| Value.Numeric{ .value = @as(i128, v), .scale = 0 },
                .real => |v| realToNumeric(v, 6), // default 6 decimal places for float→numeric
                .numeric => |v| v,
                else => return .null_value,
            };
            break :blk Value{ .numeric = n };
        },
        .type_uuid => blk: {
            const bytes = switch (val) {
                .text => |v| parseUuidString(v) orelse return .null_value,
                .uuid => |v| v,
                else => return .null_value,
            };
            break :blk Value{ .uuid = bytes };
        },
    };
}

fn evalFunctionCall(allocator: Allocator, fc: anytype, row: *const Row) EvalError!Value {
    // Aggregate functions: look up result by column name from the aggregate output row.
    // When an Aggregate operator has already computed COUNT/SUM/etc., the result is
    // stored as a named column. The Project operator just needs to retrieve it.
    if (isAggregateFuncName(fc.name)) {
        // Build the expected column name for this aggregate
        const col_name = aggResultColName(fc);
        for (row.columns, 0..) |c, i| {
            if (std.ascii.eqlIgnoreCase(c, col_name)) {
                return row.values[i].dupe(allocator) catch return EvalError.OutOfMemory;
            }
        }
        // If not found by exact name, the aggregate wasn't computed — fall through
        return EvalError.UnsupportedExpression;
    }

    // Built-in scalar functions
    if (std.ascii.eqlIgnoreCase(fc.name, "abs")) {
        if (fc.args.len != 1) return EvalError.TypeError;
        const arg = try evalExpr(allocator, fc.args[0], row);
        defer arg.free(allocator);
        return switch (arg) {
            .integer => |v| Value{ .integer = if (v < 0) -v else v },
            .real => |v| Value{ .real = @abs(v) },
            else => .null_value,
        };
    }
    if (std.ascii.eqlIgnoreCase(fc.name, "length")) {
        if (fc.args.len != 1) return EvalError.TypeError;
        const arg = try evalExpr(allocator, fc.args[0], row);
        defer arg.free(allocator);
        return switch (arg) {
            .text => |v| Value{ .integer = @intCast(v.len) },
            .blob => |v| Value{ .integer = @intCast(v.len) },
            else => .null_value,
        };
    }
    if (std.ascii.eqlIgnoreCase(fc.name, "upper")) {
        if (fc.args.len != 1) return EvalError.TypeError;
        const arg = try evalExpr(allocator, fc.args[0], row);
        defer arg.free(allocator);
        return switch (arg) {
            .text => |v| blk: {
                const upper = allocator.alloc(u8, v.len) catch return EvalError.OutOfMemory;
                for (v, 0..) |c, i| upper[i] = std.ascii.toUpper(c);
                break :blk Value{ .text = upper };
            },
            else => .null_value,
        };
    }
    if (std.ascii.eqlIgnoreCase(fc.name, "lower")) {
        if (fc.args.len != 1) return EvalError.TypeError;
        const arg = try evalExpr(allocator, fc.args[0], row);
        defer arg.free(allocator);
        return switch (arg) {
            .text => |v| blk: {
                const lower = allocator.alloc(u8, v.len) catch return EvalError.OutOfMemory;
                for (v, 0..) |c, i| lower[i] = std.ascii.toLower(c);
                break :blk Value{ .text = lower };
            },
            else => .null_value,
        };
    }
    if (std.ascii.eqlIgnoreCase(fc.name, "coalesce")) {
        for (fc.args) |arg| {
            const val = try evalExpr(allocator, arg, row);
            if (val != .null_value) return val;
            val.free(allocator);
        }
        return .null_value;
    }
    if (std.ascii.eqlIgnoreCase(fc.name, "typeof")) {
        if (fc.args.len != 1) return EvalError.TypeError;
        const arg = try evalExpr(allocator, fc.args[0], row);
        defer arg.free(allocator);
        const type_name: []const u8 = switch (arg) {
            .integer => "integer",
            .real => "real",
            .text => "text",
            .blob => "blob",
            .boolean => "boolean",
            .date => "date",
            .time => "time",
            .timestamp => "timestamp",
            .interval => "interval",
            .numeric => "numeric",
            .uuid => "uuid",
            .null_value => "null",
        };
        return Value{ .text = allocator.dupe(u8, type_name) catch return EvalError.OutOfMemory };
    }

    // Temporal functions
    if (std.ascii.eqlIgnoreCase(fc.name, "current_date")) {
        const now = std.time.timestamp();
        const days: i32 = @intCast(@divTrunc(now, 86400)); // seconds per day
        return Value{ .date = days };
    }
    if (std.ascii.eqlIgnoreCase(fc.name, "current_time")) {
        const now = std.time.timestamp();
        const micros_today = @mod(now, 86400) * MICROS_PER_SECOND;
        return Value{ .time = micros_today };
    }
    if (std.ascii.eqlIgnoreCase(fc.name, "current_timestamp") or std.ascii.eqlIgnoreCase(fc.name, "now")) {
        const now = std.time.timestamp();
        return Value{ .timestamp = now * MICROS_PER_SECOND };
    }

    // UUID generation
    if (std.ascii.eqlIgnoreCase(fc.name, "gen_random_uuid")) {
        return Value{ .uuid = generateUuidV4() };
    }

    return EvalError.UnsupportedExpression;
}

/// Check if a function name is an aggregate function.
fn isAggregateFuncName(name: []const u8) bool {
    const agg_names = [_][]const u8{ "count", "sum", "avg", "min", "max" };
    for (agg_names) |n| {
        if (std.ascii.eqlIgnoreCase(name, n)) return true;
    }
    return false;
}

/// Build the column name that the AggregateOp would use for this function call.
fn aggResultColName(fc: anytype) []const u8 {
    // COUNT(*) → "count(*)", COUNT(x) → "count", SUM(x) → "sum", etc.
    if (std.ascii.eqlIgnoreCase(fc.name, "count")) {
        if (fc.args.len > 0 and fc.args[0].* == .column_ref and
            std.mem.eql(u8, fc.args[0].column_ref.name, "*"))
        {
            return "count(*)";
        }
        return "count";
    }
    if (std.ascii.eqlIgnoreCase(fc.name, "sum")) return "sum";
    if (std.ascii.eqlIgnoreCase(fc.name, "avg")) return "avg";
    if (std.ascii.eqlIgnoreCase(fc.name, "min")) return "min";
    if (std.ascii.eqlIgnoreCase(fc.name, "max")) return "max";
    return fc.name;
}

/// SQL LIKE pattern matching (% = any string, _ = any char).
fn likeMatch(text: []const u8, pattern: []const u8) bool {
    var ti: usize = 0;
    var pi: usize = 0;
    var star_pi: ?usize = null;
    var star_ti: usize = 0;

    while (ti < text.len or pi < pattern.len) {
        if (pi < pattern.len) {
            if (pattern[pi] == '%') {
                star_pi = pi;
                star_ti = ti;
                pi += 1;
                continue;
            }
            if (ti < text.len) {
                if (pattern[pi] == '_' or std.ascii.toLower(pattern[pi]) == std.ascii.toLower(text[ti])) {
                    ti += 1;
                    pi += 1;
                    continue;
                }
            }
        }
        if (star_pi) |sp| {
            pi = sp + 1;
            star_ti += 1;
            ti = star_ti;
            if (ti > text.len) return false;
        } else {
            return false;
        }
    }
    return true;
}

// ── Executor Interface ──────────────────────────────────────────────────

/// Error type for executor operations.
pub const ExecError = error{
    OutOfMemory,
    TypeError,
    DivisionByZero,
    ColumnNotFound,
    UnsupportedExpression,
    TableNotFound,
    InvalidRowData,
    StorageError,
    ExecutionError,
};

/// The Volcano-model iterator interface.
/// Each operator produces rows one at a time via next().
pub const RowIterator = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        next: *const fn (ptr: *anyopaque) ExecError!?Row,
        close: *const fn (ptr: *anyopaque) void,
    };

    pub fn next(self: RowIterator) ExecError!?Row {
        return self.vtable.next(self.ptr);
    }

    pub fn close(self: RowIterator) void {
        self.vtable.close(self.ptr);
    }
};

// ── Scan Operator ───────────────────────────────────────────────────────

/// Full table scan via B+Tree cursor.
pub const ScanOp = struct {
    allocator: Allocator,
    tree: BTree,
    cursor: ?Cursor = null,
    col_names: []const []const u8,
    opened: bool = false,
    /// MVCC context for visibility filtering (null = all rows visible).
    mvcc_ctx: ?MvccContext = null,

    /// Create a ScanOp. After placement on the heap, call initCursor() to
    /// set up the cursor with a stable pointer to the tree.
    pub fn init(allocator: Allocator, pool: *BufferPool, data_root_page_id: u32, col_names: []const []const u8) ScanOp {
        return .{
            .allocator = allocator,
            .tree = BTree.init(pool, data_root_page_id),
            .col_names = col_names,
        };
    }

    /// Must be called after the ScanOp is at its final heap location.
    pub fn initCursor(self: *ScanOp) void {
        self.cursor = Cursor.init(self.allocator, &self.tree);
    }

    pub fn open(self: *ScanOp) ExecError!void {
        if (self.cursor == null) self.initCursor();
        self.cursor.?.seekFirst() catch return ExecError.StorageError;
        self.opened = true;
    }

    pub fn next(self: *ScanOp) ExecError!?Row {
        if (!self.opened) try self.open();

        while (true) {
            const entry = self.cursor.?.next() catch return ExecError.StorageError;
            if (entry == null) return null;

            defer self.allocator.free(entry.?.key);

            // MVCC visibility check: deserialize header and filter invisible tuples
            if (self.mvcc_ctx) |ctx| {
                if (ctx.enabled and mvcc_mod.isVersionedRow(entry.?.value)) {
                    const header = TupleHeader.deserialize(entry.?.value[1..][0..mvcc_mod.TUPLE_HEADER_SIZE]);
                    if (!mvcc_mod.isTupleVisibleWithTm(header, ctx.snapshot, ctx.current_xid, ctx.current_cid, ctx.tm)) {
                        // Tuple not visible — skip it
                        self.allocator.free(entry.?.value);
                        continue;
                    }
                    // Visible: deserialize column data (skip MVCC header)
                    const values = deserializeRow(self.allocator, entry.?.value[mvcc_mod.MVCC_ROW_OVERHEAD..]) catch {
                        self.allocator.free(entry.?.value);
                        return ExecError.InvalidRowData;
                    };
                    self.allocator.free(entry.?.value);
                    errdefer {
                        for (values) |v| v.free(self.allocator);
                        self.allocator.free(values);
                    }

                    const cols = self.allocator.alloc([]const u8, self.col_names.len) catch return ExecError.OutOfMemory;
                    for (self.col_names, 0..) |c, i| cols[i] = c;
                    return Row{ .columns = cols, .values = values, .allocator = self.allocator };
                }
            }

            defer self.allocator.free(entry.?.value);

            // No MVCC context or legacy row: return all rows.
            // Still need to detect MVCC format and skip the overhead for committed rows.
            const row_bytes = if (mvcc_mod.isVersionedRow(entry.?.value))
                entry.?.value[mvcc_mod.MVCC_ROW_OVERHEAD..]
            else
                entry.?.value;
            const values = deserializeRow(self.allocator, row_bytes) catch return ExecError.InvalidRowData;
            errdefer {
                for (values) |v| v.free(self.allocator);
                self.allocator.free(values);
            }

            // Build column names
            const cols = self.allocator.alloc([]const u8, self.col_names.len) catch return ExecError.OutOfMemory;
            for (self.col_names, 0..) |c, i| cols[i] = c;

            return Row{
                .columns = cols,
                .values = values,
                .allocator = self.allocator,
            };
        }
    }

    pub fn close(self: *ScanOp) void {
        if (self.cursor) |*c| c.deinit();
    }

    pub fn iterator(self: *ScanOp) RowIterator {
        return .{
            .ptr = self,
            .vtable = &.{
                .next = @ptrCast(&ScanOp.next),
                .close = @ptrCast(&ScanOp.close),
            },
        };
    }
};

// ── Index Scan Operator ─────────────────────────────────────────────────

/// Index-based point lookup: uses a secondary index B+Tree to find
/// matching row keys, then fetches full rows from the data B+Tree.
/// Returns at most one row for an equality lookup.
pub const IndexScanOp = struct {
    allocator: Allocator,
    pool: *BufferPool,
    data_root_page_id: u32,
    index_root_page_id: u32,
    lookup_key: []const u8,
    col_names: []const []const u8,
    exhausted: bool = false,
    /// MVCC context for visibility filtering (null = all rows visible).
    mvcc_ctx: ?MvccContext = null,

    pub fn init(
        allocator: Allocator,
        pool: *BufferPool,
        data_root_page_id: u32,
        index_root_page_id: u32,
        lookup_key: []const u8,
        col_names: []const []const u8,
    ) IndexScanOp {
        return .{
            .allocator = allocator,
            .pool = pool,
            .data_root_page_id = data_root_page_id,
            .index_root_page_id = index_root_page_id,
            .lookup_key = lookup_key,
            .col_names = col_names,
        };
    }

    pub fn next(self: *IndexScanOp) ExecError!?Row {
        if (self.exhausted) return null;
        self.exhausted = true;

        // Look up the index to find the row key
        var idx_tree = BTree.init(self.pool, self.index_root_page_id);
        const row_key = idx_tree.get(self.allocator, self.lookup_key) catch return ExecError.StorageError;
        if (row_key == null) return null; // No matching index entry
        defer self.allocator.free(row_key.?);

        // Fetch the actual row from the data B+Tree
        var data_tree = BTree.init(self.pool, self.data_root_page_id);
        const row_data = data_tree.get(self.allocator, row_key.?) catch return ExecError.StorageError;
        if (row_data == null) return null; // Orphaned index entry
        defer self.allocator.free(row_data.?);

        // MVCC visibility check
        if (self.mvcc_ctx) |ctx| {
            if (ctx.enabled and mvcc_mod.isVersionedRow(row_data.?)) {
                const header = TupleHeader.deserialize(row_data.?[1..][0..mvcc_mod.TUPLE_HEADER_SIZE]);
                if (!mvcc_mod.isTupleVisible(header, ctx.snapshot, ctx.current_xid, ctx.current_cid)) {
                    return null; // Tuple not visible
                }
                // Visible: deserialize column data (skip MVCC header)
                const values = deserializeRow(self.allocator, row_data.?[mvcc_mod.MVCC_ROW_OVERHEAD..]) catch return ExecError.InvalidRowData;
                errdefer {
                    for (values) |v| v.free(self.allocator);
                    self.allocator.free(values);
                }
                const cols = self.allocator.alloc([]const u8, self.col_names.len) catch return ExecError.OutOfMemory;
                for (self.col_names, 0..) |c, i| cols[i] = c;
                return Row{ .columns = cols, .values = values, .allocator = self.allocator };
            }
        }

        // No MVCC context or legacy row — still detect MVCC format and skip overhead
        const row_bytes = if (mvcc_mod.isVersionedRow(row_data.?))
            row_data.?[mvcc_mod.MVCC_ROW_OVERHEAD..]
        else
            row_data.?;
        const values = deserializeRow(self.allocator, row_bytes) catch return ExecError.InvalidRowData;
        errdefer {
            for (values) |v| v.free(self.allocator);
            self.allocator.free(values);
        }

        const cols = self.allocator.alloc([]const u8, self.col_names.len) catch return ExecError.OutOfMemory;
        for (self.col_names, 0..) |c, i| cols[i] = c;

        return Row{
            .columns = cols,
            .values = values,
            .allocator = self.allocator,
        };
    }

    pub fn close(_: *IndexScanOp) void {}

    pub fn iterator(self: *IndexScanOp) RowIterator {
        return .{
            .ptr = self,
            .vtable = &.{
                .next = @ptrCast(&IndexScanOp.next),
                .close = @ptrCast(&IndexScanOp.close),
            },
        };
    }
};

// ── Filter Operator ─────────────────────────────────────────────────────

/// Applies a predicate to filter rows from its input.
pub const FilterOp = struct {
    allocator: Allocator,
    input: RowIterator,
    predicate: *const ast.Expr,

    pub fn init(allocator: Allocator, input: RowIterator, predicate: *const ast.Expr) FilterOp {
        return .{
            .allocator = allocator,
            .input = input,
            .predicate = predicate,
        };
    }

    pub fn next(self: *FilterOp) ExecError!?Row {
        while (true) {
            var row = try self.input.next() orelse return null;
            const val = evalExpr(self.allocator, self.predicate, &row) catch |err| {
                row.deinit();
                return switch (err) {
                    error.OutOfMemory => ExecError.OutOfMemory,
                    error.TypeError => ExecError.TypeError,
                    error.DivisionByZero => ExecError.DivisionByZero,
                    error.ColumnNotFound => ExecError.ColumnNotFound,
                    error.UnsupportedExpression => ExecError.UnsupportedExpression,
                };
            };
            defer val.free(self.allocator);

            if (val.isTruthy()) return row;
            row.deinit();
        }
    }

    pub fn close(self: *FilterOp) void {
        self.input.close();
    }

    pub fn iterator(self: *FilterOp) RowIterator {
        return .{
            .ptr = self,
            .vtable = &.{
                .next = @ptrCast(&FilterOp.next),
                .close = @ptrCast(&FilterOp.close),
            },
        };
    }
};

// ── Project Operator ────────────────────────────────────────────────────

/// Selects/computes output columns from each input row.
pub const ProjectOp = struct {
    allocator: Allocator,
    input: RowIterator,
    columns: []const PlanNode.ProjectColumn,

    pub fn init(allocator: Allocator, input: RowIterator, columns: []const PlanNode.ProjectColumn) ProjectOp {
        return .{
            .allocator = allocator,
            .input = input,
            .columns = columns,
        };
    }

    pub fn next(self: *ProjectOp) ExecError!?Row {
        var row = try self.input.next() orelse return null;
        defer row.deinit();

        const vals = self.allocator.alloc(Value, self.columns.len) catch return ExecError.OutOfMemory;
        var inited: usize = 0;
        errdefer {
            for (vals[0..inited]) |v| v.free(self.allocator);
            self.allocator.free(vals);
        }

        const col_names = self.allocator.alloc([]const u8, self.columns.len) catch return ExecError.OutOfMemory;
        errdefer self.allocator.free(col_names);

        for (self.columns, 0..) |col, i| {
            vals[i] = evalExpr(self.allocator, col.expr, &row) catch |err| {
                // When an aggregate function has an alias (e.g., SUM(x) AS total),
                // the AggregateOp stores the result under the alias name. Try looking
                // up by alias before reporting an error.
                if (err == error.UnsupportedExpression) {
                    // Try alias first (aggregates, window functions with AS)
                    if (col.alias) |alias| {
                        if (row.getColumn(alias)) |v| {
                            vals[i] = v.dupe(self.allocator) catch return ExecError.OutOfMemory;
                            inited += 1;
                            col_names[i] = col.alias orelse exprColumnName(col.expr);
                            continue;
                        }
                    }
                    // For window functions, look up by function name
                    if (col.expr.* == .window_function) {
                        if (row.getColumn(col.expr.window_function.name)) |v| {
                            vals[i] = v.dupe(self.allocator) catch return ExecError.OutOfMemory;
                            inited += 1;
                            col_names[i] = col.alias orelse col.expr.window_function.name;
                            continue;
                        }
                    }
                }
                return switch (err) {
                    error.OutOfMemory => ExecError.OutOfMemory,
                    error.TypeError => ExecError.TypeError,
                    error.DivisionByZero => ExecError.DivisionByZero,
                    error.ColumnNotFound => ExecError.ColumnNotFound,
                    error.UnsupportedExpression => ExecError.UnsupportedExpression,
                };
            };
            inited += 1;
            col_names[i] = col.alias orelse exprColumnName(col.expr);
        }

        return Row{
            .columns = col_names,
            .values = vals,
            .allocator = self.allocator,
        };
    }

    pub fn close(self: *ProjectOp) void {
        self.input.close();
    }

    pub fn iterator(self: *ProjectOp) RowIterator {
        return .{
            .ptr = self,
            .vtable = &.{
                .next = @ptrCast(&ProjectOp.next),
                .close = @ptrCast(&ProjectOp.close),
            },
        };
    }
};

/// Extract a display name from an expression (for unnamed columns).
fn exprColumnName(expr: *const ast.Expr) []const u8 {
    return switch (expr.*) {
        .column_ref => |ref| ref.name,
        .function_call => |fc| fc.name,
        .window_function => |wf| wf.name,
        .integer_literal => "?column?",
        .float_literal => "?column?",
        .string_literal => "?column?",
        else => "?column?",
    };
}

// ── Limit Operator ──────────────────────────────────────────────────────

/// Restricts output to a maximum number of rows, with optional offset.
pub const LimitOp = struct {
    allocator: Allocator,
    input: RowIterator,
    limit_count: ?u64,
    offset_count: u64,
    returned: u64 = 0,
    skipped: u64 = 0,

    pub fn init(allocator: Allocator, input: RowIterator, limit_count: ?u64, offset_count: u64) LimitOp {
        return .{
            .allocator = allocator,
            .input = input,
            .limit_count = limit_count,
            .offset_count = offset_count,
        };
    }

    pub fn next(self: *LimitOp) ExecError!?Row {
        // Skip offset rows
        while (self.skipped < self.offset_count) {
            var row = try self.input.next() orelse return null;
            row.deinit();
            self.skipped += 1;
        }

        // Check limit
        if (self.limit_count) |lim| {
            if (self.returned >= lim) return null;
        }

        const row = try self.input.next() orelse return null;
        self.returned += 1;
        return row;
    }

    pub fn close(self: *LimitOp) void {
        self.input.close();
    }

    pub fn iterator(self: *LimitOp) RowIterator {
        return .{
            .ptr = self,
            .vtable = &.{
                .next = @ptrCast(&LimitOp.next),
                .close = @ptrCast(&LimitOp.close),
            },
        };
    }
};

// ── Sort Operator ───────────────────────────────────────────────────────

/// In-memory sort. Materializes all input rows, sorts, then emits.
pub const SortOp = struct {
    allocator: Allocator,
    input: RowIterator,
    order_by: []const ast.OrderByItem,
    rows: std.ArrayListUnmanaged(Row) = .{},
    index: usize = 0,
    materialized: bool = false,

    pub fn init(allocator: Allocator, input: RowIterator, order_by: []const ast.OrderByItem) SortOp {
        return .{
            .allocator = allocator,
            .input = input,
            .order_by = order_by,
        };
    }

    fn materialize(self: *SortOp) ExecError!void {
        // Collect all rows
        while (true) {
            const row = try self.input.next() orelse break;
            self.rows.append(self.allocator, row) catch return ExecError.OutOfMemory;
        }

        // Sort using order_by expressions
        const ctx = SortContext{ .order_by = self.order_by, .allocator = self.allocator };
        std.sort.block(Row, self.rows.items, ctx, SortContext.lessThan);

        self.materialized = true;
    }

    pub fn next(self: *SortOp) ExecError!?Row {
        if (!self.materialized) try self.materialize();

        if (self.index >= self.rows.items.len) return null;
        const row = self.rows.items[self.index];
        self.index += 1;
        // Transfer ownership — don't deinit here
        return row;
    }

    pub fn close(self: *SortOp) void {
        // Free any remaining rows that haven't been consumed
        for (self.rows.items[self.index..]) |*row| row.deinit();
        self.rows.deinit(self.allocator);
        self.input.close();
    }

    pub fn iterator(self: *SortOp) RowIterator {
        return .{
            .ptr = self,
            .vtable = &.{
                .next = @ptrCast(&SortOp.next),
                .close = @ptrCast(&SortOp.close),
            },
        };
    }

    const SortContext = struct {
        order_by: []const ast.OrderByItem,
        allocator: Allocator,

        fn lessThan(ctx: SortContext, a: Row, b: Row) bool {
            for (ctx.order_by) |ob| {
                const av = evalExpr(ctx.allocator, ob.expr, &a) catch Value.null_value;
                defer av.free(ctx.allocator);
                const bv = evalExpr(ctx.allocator, ob.expr, &b) catch Value.null_value;
                defer bv.free(ctx.allocator);

                const order = av.compare(bv);
                if (order == .eq) continue;

                return switch (ob.direction) {
                    .asc => order == .lt,
                    .desc => order == .gt,
                };
            }
            return false;
        }
    };
};

// ── Window Operator ─────────────────────────────────────────────────────

/// Window function operator. Buffers all input rows, sorts by partition+order keys,
/// then emits each row with computed window function values appended as extra columns.
pub const WindowOp = struct {
    allocator: Allocator,
    input: RowIterator,
    /// Window function AST expressions (each is .window_function variant).
    funcs: []const *const ast.Expr,
    /// Aliases for each window function result column.
    aliases: []const ?[]const u8,
    /// Buffered and augmented rows with window function results.
    result_rows: std.ArrayListUnmanaged(Row) = .{},
    index: usize = 0,
    materialized: bool = false,

    pub fn init(
        allocator: Allocator,
        input: RowIterator,
        funcs: []const *const ast.Expr,
        aliases: []const ?[]const u8,
    ) WindowOp {
        return .{
            .allocator = allocator,
            .input = input,
            .funcs = funcs,
            .aliases = aliases,
        };
    }

    fn materialize(self: *WindowOp) ExecError!void {
        const alloc = self.allocator;

        // 1. Buffer all input rows
        var input_rows = std.ArrayListUnmanaged(Row){};
        while (true) {
            const row = try self.input.next() orelse break;
            input_rows.append(alloc, row) catch return ExecError.OutOfMemory;
        }

        // Process each window function specification
        // For simplicity, we process all window functions together since they
        // share the same input rows. For each function, we sort by its
        // partition+order keys, compute values, then merge results.

        // Build per-row result values for each window function
        const num_funcs = self.funcs.len;
        const num_rows = input_rows.items.len;

        // results[func_idx][row_idx] — computed window value for each function per row
        var func_results = alloc.alloc([]Value, num_funcs) catch return ExecError.OutOfMemory;
        defer {
            for (func_results) |fr| alloc.free(fr);
            alloc.free(func_results);
        }

        for (self.funcs, 0..) |func_expr, fi| {
            func_results[fi] = alloc.alloc(Value, num_rows) catch return ExecError.OutOfMemory;

            const wf = func_expr.window_function;

            // Build sort indices for this window's partition+order
            var indices = alloc.alloc(usize, num_rows) catch return ExecError.OutOfMemory;
            defer alloc.free(indices);
            for (0..num_rows) |i| indices[i] = i;

            // Sort indices by partition_by + order_by keys
            const sort_ctx = WindowSortContext{
                .partition_by = wf.partition_by,
                .order_by = wf.order_by,
                .rows = input_rows.items,
                .allocator = alloc,
            };
            std.sort.block(usize, indices, sort_ctx, WindowSortContext.lessThan);

            // Walk sorted indices, detect partition boundaries, compute window values
            var partition_start: usize = 0;
            var ri: usize = 0;
            while (ri < num_rows) {
                // Find end of current partition
                partition_start = ri;
                var partition_end = ri + 1;
                while (partition_end < num_rows) {
                    if (!samePartition(alloc, wf.partition_by, &input_rows.items[indices[partition_start]], &input_rows.items[indices[partition_end]])) break;
                    partition_end += 1;
                }

                // Compute values for each row in this partition
                computePartitionValues(
                    alloc,
                    wf,
                    input_rows.items,
                    indices[partition_start..partition_end],
                    func_results[fi],
                );

                ri = partition_end;
            }
        }

        // 2. Build output rows: original columns + window function columns
        for (input_rows.items, 0..) |*row, row_idx| {
            const orig_cols = row.columns.len;
            const new_cols = alloc.alloc([]const u8, orig_cols + num_funcs) catch return ExecError.OutOfMemory;
            const new_vals = alloc.alloc(Value, orig_cols + num_funcs) catch return ExecError.OutOfMemory;

            // Copy original columns
            for (0..orig_cols) |i| {
                new_cols[i] = row.columns[i];
                new_vals[i] = row.values[i];
            }

            // Append window function results
            for (0..num_funcs) |fi| {
                const alias = self.aliases[fi] orelse self.funcs[fi].window_function.name;
                new_cols[orig_cols + fi] = alias;
                new_vals[orig_cols + fi] = func_results[fi][row_idx];
            }

            // Free old column/value arrays (values ownership transferred)
            alloc.free(row.columns);
            alloc.free(row.values);
            row.columns = new_cols;
            row.values = new_vals;

            self.result_rows.append(alloc, row.*) catch return ExecError.OutOfMemory;
        }
        input_rows.deinit(alloc);

        self.materialized = true;
    }

    pub fn next(self: *WindowOp) ExecError!?Row {
        if (!self.materialized) try self.materialize();
        if (self.index >= self.result_rows.items.len) return null;
        const row = self.result_rows.items[self.index];
        self.index += 1;
        return row;
    }

    pub fn close(self: *WindowOp) void {
        for (self.result_rows.items[self.index..]) |*row| row.deinit();
        self.result_rows.deinit(self.allocator);
        self.input.close();
    }

    pub fn iterator(self: *WindowOp) RowIterator {
        return .{
            .ptr = self,
            .vtable = &.{
                .next = @ptrCast(&WindowOp.next),
                .close = @ptrCast(&WindowOp.close),
            },
        };
    }

    // ── Partition helpers ────────────────────────────────────────────

    const WindowSortContext = struct {
        partition_by: []const *const ast.Expr,
        order_by: []const ast.OrderByItem,
        rows: []const Row,
        allocator: Allocator,

        fn lessThan(ctx: WindowSortContext, a_idx: usize, b_idx: usize) bool {
            const a = &ctx.rows[a_idx];
            const b = &ctx.rows[b_idx];
            // Compare partition keys first
            for (ctx.partition_by) |pb| {
                const av = evalExpr(ctx.allocator, pb, a) catch Value.null_value;
                defer av.free(ctx.allocator);
                const bv = evalExpr(ctx.allocator, pb, b) catch Value.null_value;
                defer bv.free(ctx.allocator);
                const order = av.compare(bv);
                if (order != .eq) return order == .lt;
            }
            // Then order keys
            for (ctx.order_by) |ob| {
                const av = evalExpr(ctx.allocator, ob.expr, a) catch Value.null_value;
                defer av.free(ctx.allocator);
                const bv = evalExpr(ctx.allocator, ob.expr, b) catch Value.null_value;
                defer bv.free(ctx.allocator);
                const order = av.compare(bv);
                if (order == .eq) continue;
                return switch (ob.direction) {
                    .asc => order == .lt,
                    .desc => order == .gt,
                };
            }
            return false;
        }
    };

    fn samePartition(alloc: Allocator, partition_by: []const *const ast.Expr, a: *const Row, b: *const Row) bool {
        for (partition_by) |pb| {
            const av = evalExpr(alloc, pb, a) catch Value.null_value;
            defer av.free(alloc);
            const bv = evalExpr(alloc, pb, b) catch Value.null_value;
            defer bv.free(alloc);
            if (!av.eql(bv)) return false;
        }
        return true;
    }

    fn toLower(name: []const u8, buf: []u8) []const u8 {
        const len = @min(name.len, buf.len);
        for (0..len) |i| {
            buf[i] = if (name[i] >= 'A' and name[i] <= 'Z') name[i] + 32 else name[i];
        }
        return buf[0..len];
    }

    /// Compute window function values for a partition slice of sorted indices.
    fn computePartitionValues(
        alloc: Allocator,
        wf: ast.WindowFunctionExpr,
        all_rows: []const Row,
        partition_indices: []const usize,
        results: []Value,
    ) void {
        var name_buf: [32]u8 = undefined;
        const func_name_lower = toLower(wf.name, &name_buf);

        const part_len = partition_indices.len;

        if (std.mem.eql(u8, func_name_lower, "row_number")) {
            for (partition_indices, 0..) |orig_idx, pos| {
                results[orig_idx] = .{ .integer = @intCast(pos + 1) };
            }
        } else if (std.mem.eql(u8, func_name_lower, "rank")) {
            var rank: i64 = 1;
            for (partition_indices, 0..) |orig_idx, pos| {
                if (pos > 0 and !orderKeysEqual(alloc, wf.order_by, &all_rows[partition_indices[pos - 1]], &all_rows[orig_idx])) {
                    rank = @intCast(pos + 1);
                }
                results[orig_idx] = .{ .integer = rank };
            }
        } else if (std.mem.eql(u8, func_name_lower, "dense_rank")) {
            var rank: i64 = 1;
            for (partition_indices, 0..) |orig_idx, pos| {
                if (pos > 0 and !orderKeysEqual(alloc, wf.order_by, &all_rows[partition_indices[pos - 1]], &all_rows[orig_idx])) {
                    rank += 1;
                }
                results[orig_idx] = .{ .integer = rank };
            }
        } else if (std.mem.eql(u8, func_name_lower, "ntile")) {
            const n: i64 = if (wf.args.len > 0) blk: {
                const val: Value = evalExpr(alloc, wf.args[0], &all_rows[partition_indices[0]]) catch .null_value;
                break :blk val.toInteger() orelse 1;
            } else 1;
            const bucket_size = if (n > 0) @max(1, @divTrunc(@as(i64, @intCast(part_len)), n)) else 1;
            for (partition_indices, 0..) |orig_idx, pos| {
                const tile = @min(n, @divTrunc(@as(i64, @intCast(pos)), bucket_size) + 1);
                results[orig_idx] = .{ .integer = tile };
            }
        } else if (std.mem.eql(u8, func_name_lower, "lag")) {
            const offset: usize = if (wf.args.len > 1) blk: {
                const val: Value = evalExpr(alloc, wf.args[1], &all_rows[partition_indices[0]]) catch .null_value;
                break :blk @intCast(val.toInteger() orelse 1);
            } else 1;
            const default_val: Value = if (wf.args.len > 2)
                evalExpr(alloc, wf.args[2], &all_rows[partition_indices[0]]) catch Value.null_value
            else
                Value.null_value;
            for (partition_indices, 0..) |orig_idx, pos| {
                if (pos >= offset) {
                    const prev_idx = partition_indices[pos - offset];
                    if (wf.args.len > 0) {
                        results[orig_idx] = evalExpr(alloc, wf.args[0], &all_rows[prev_idx]) catch Value.null_value;
                    } else {
                        results[orig_idx] = Value.null_value;
                    }
                } else {
                    results[orig_idx] = default_val.dupe(alloc) catch Value.null_value;
                }
            }
            if (wf.args.len > 2) default_val.free(alloc);
        } else if (std.mem.eql(u8, func_name_lower, "lead")) {
            const offset: usize = if (wf.args.len > 1) blk: {
                const val: Value = evalExpr(alloc, wf.args[1], &all_rows[partition_indices[0]]) catch .null_value;
                break :blk @intCast(val.toInteger() orelse 1);
            } else 1;
            const default_val: Value = if (wf.args.len > 2)
                evalExpr(alloc, wf.args[2], &all_rows[partition_indices[0]]) catch Value.null_value
            else
                Value.null_value;
            for (partition_indices, 0..) |orig_idx, pos| {
                if (pos + offset < part_len) {
                    const next_idx = partition_indices[pos + offset];
                    if (wf.args.len > 0) {
                        results[orig_idx] = evalExpr(alloc, wf.args[0], &all_rows[next_idx]) catch Value.null_value;
                    } else {
                        results[orig_idx] = Value.null_value;
                    }
                } else {
                    results[orig_idx] = default_val.dupe(alloc) catch Value.null_value;
                }
            }
            if (wf.args.len > 2) default_val.free(alloc);
        } else if (std.mem.eql(u8, func_name_lower, "first_value")) {
            if (wf.args.len > 0) {
                for (partition_indices) |orig_idx| {
                    results[orig_idx] = evalExpr(alloc, wf.args[0], &all_rows[partition_indices[0]]) catch Value.null_value;
                }
            } else {
                for (partition_indices) |orig_idx| results[orig_idx] = Value.null_value;
            }
        } else if (std.mem.eql(u8, func_name_lower, "last_value")) {
            // Default frame is RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            // so last_value with default frame returns current row's value.
            // With ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING, it returns actual last.
            const use_full_partition = if (wf.frame) |frame|
                @as(std.meta.Tag(ast.WindowFrameBound), frame.end) == .unbounded_following
            else
                false;
            if (wf.args.len > 0) {
                if (use_full_partition) {
                    for (partition_indices) |orig_idx| {
                        results[orig_idx] = evalExpr(alloc, wf.args[0], &all_rows[partition_indices[part_len - 1]]) catch Value.null_value;
                    }
                } else {
                    // Default frame: last_value = current row's value
                    for (partition_indices) |orig_idx| {
                        results[orig_idx] = evalExpr(alloc, wf.args[0], &all_rows[orig_idx]) catch Value.null_value;
                    }
                }
            } else {
                for (partition_indices) |orig_idx| results[orig_idx] = Value.null_value;
            }
        } else if (std.mem.eql(u8, func_name_lower, "nth_value")) {
            const n: usize = if (wf.args.len > 1) blk: {
                const val: Value = evalExpr(alloc, wf.args[1], &all_rows[partition_indices[0]]) catch .null_value;
                break :blk @intCast(val.toInteger() orelse 1);
            } else 1;
            if (wf.args.len > 0 and n >= 1 and n <= part_len) {
                for (partition_indices) |orig_idx| {
                    results[orig_idx] = evalExpr(alloc, wf.args[0], &all_rows[partition_indices[n - 1]]) catch Value.null_value;
                }
            } else {
                for (partition_indices) |orig_idx| results[orig_idx] = Value.null_value;
            }
        } else if (std.mem.eql(u8, func_name_lower, "percent_rank")) {
            if (part_len <= 1) {
                for (partition_indices) |orig_idx| results[orig_idx] = .{ .real = 0.0 };
            } else {
                var rank: i64 = 1;
                for (partition_indices, 0..) |orig_idx, pos| {
                    if (pos > 0 and !orderKeysEqual(alloc, wf.order_by, &all_rows[partition_indices[pos - 1]], &all_rows[orig_idx])) {
                        rank = @intCast(pos + 1);
                    }
                    const pr = @as(f64, @floatFromInt(rank - 1)) / @as(f64, @floatFromInt(part_len - 1));
                    results[orig_idx] = .{ .real = pr };
                }
            }
        } else if (std.mem.eql(u8, func_name_lower, "cume_dist")) {
            for (partition_indices, 0..) |orig_idx, pos| {
                // Count rows with order key <= current row
                var count: usize = pos + 1;
                // Include subsequent rows with equal order key
                var k = pos + 1;
                while (k < part_len and orderKeysEqual(alloc, wf.order_by, &all_rows[orig_idx], &all_rows[partition_indices[k]])) : (k += 1) {
                    count += 1;
                }
                const cd = @as(f64, @floatFromInt(count)) / @as(f64, @floatFromInt(part_len));
                results[orig_idx] = .{ .real = cd };
            }
        } else {
            // Aggregate as window function (SUM, COUNT, AVG, MIN, MAX)
            computeAggregateWindow(alloc, wf, all_rows, partition_indices, results);
        }
    }

    fn computeAggregateWindow(
        alloc: Allocator,
        wf: ast.WindowFunctionExpr,
        all_rows: []const Row,
        partition_indices: []const usize,
        results: []Value,
    ) void {
        var name_buf: [32]u8 = undefined;
        const func_name_lower = toLower(wf.name, &name_buf);
        const is_count_star = std.mem.eql(u8, func_name_lower, "count") and
            (wf.args.len == 0 or (wf.args.len > 0 and wf.args[0].* == .column_ref and std.mem.eql(u8, wf.args[0].column_ref.name, "*")));

        // Determine frame bounds for each row
        for (partition_indices, 0..) |orig_idx, pos| {
            const frame_range = resolveFrameRange(wf.frame, pos, partition_indices.len, wf.order_by.len > 0);
            const frame_start = frame_range[0];
            const frame_end = frame_range[1];

            if (is_count_star) {
                results[orig_idx] = .{ .integer = @intCast(frame_end - frame_start) };
            } else if (std.mem.eql(u8, func_name_lower, "count")) {
                var count: i64 = 0;
                for (frame_start..frame_end) |fi| {
                    if (wf.args.len > 0) {
                        const v = evalExpr(alloc, wf.args[0], &all_rows[partition_indices[fi]]) catch Value.null_value;
                        defer v.free(alloc);
                        if (v != .null_value) count += 1;
                    }
                }
                results[orig_idx] = .{ .integer = count };
            } else if (std.mem.eql(u8, func_name_lower, "sum")) {
                var sum_int: i64 = 0;
                var sum_real: f64 = 0;
                var has_real = false;
                var has_any = false;
                for (frame_start..frame_end) |fi| {
                    if (wf.args.len > 0) {
                        const v = evalExpr(alloc, wf.args[0], &all_rows[partition_indices[fi]]) catch Value.null_value;
                        defer v.free(alloc);
                        switch (v) {
                            .integer => |iv| {
                                sum_int += iv;
                                has_any = true;
                            },
                            .real => |rv| {
                                sum_real += rv;
                                has_real = true;
                                has_any = true;
                            },
                            else => {},
                        }
                    }
                }
                if (!has_any) {
                    results[orig_idx] = Value.null_value;
                } else if (has_real) {
                    results[orig_idx] = .{ .real = sum_real + @as(f64, @floatFromInt(sum_int)) };
                } else {
                    results[orig_idx] = .{ .integer = sum_int };
                }
            } else if (std.mem.eql(u8, func_name_lower, "avg")) {
                var sum: f64 = 0;
                var count: usize = 0;
                for (frame_start..frame_end) |fi| {
                    if (wf.args.len > 0) {
                        const v = evalExpr(alloc, wf.args[0], &all_rows[partition_indices[fi]]) catch Value.null_value;
                        defer v.free(alloc);
                        switch (v) {
                            .integer => |iv| {
                                sum += @floatFromInt(iv);
                                count += 1;
                            },
                            .real => |rv| {
                                sum += rv;
                                count += 1;
                            },
                            else => {},
                        }
                    }
                }
                if (count == 0) {
                    results[orig_idx] = Value.null_value;
                } else {
                    results[orig_idx] = .{ .real = sum / @as(f64, @floatFromInt(count)) };
                }
            } else if (std.mem.eql(u8, func_name_lower, "min")) {
                var min_val: Value = Value.null_value;
                for (frame_start..frame_end) |fi| {
                    if (wf.args.len > 0) {
                        const v = evalExpr(alloc, wf.args[0], &all_rows[partition_indices[fi]]) catch Value.null_value;
                        if (v == .null_value) {
                            v.free(alloc);
                            continue;
                        }
                        if (min_val == .null_value or v.compare(min_val) == .lt) {
                            min_val.free(alloc);
                            min_val = v;
                        } else {
                            v.free(alloc);
                        }
                    }
                }
                results[orig_idx] = min_val;
            } else if (std.mem.eql(u8, func_name_lower, "max")) {
                var max_val: Value = Value.null_value;
                for (frame_start..frame_end) |fi| {
                    if (wf.args.len > 0) {
                        const v = evalExpr(alloc, wf.args[0], &all_rows[partition_indices[fi]]) catch Value.null_value;
                        if (v == .null_value) {
                            v.free(alloc);
                            continue;
                        }
                        if (max_val == .null_value or v.compare(max_val) == .gt) {
                            max_val.free(alloc);
                            max_val = v;
                        } else {
                            v.free(alloc);
                        }
                    }
                }
                results[orig_idx] = max_val;
            } else {
                results[orig_idx] = Value.null_value;
            }
        }
    }

    /// Resolve frame bounds to a [start, end) range within the partition.
    /// SQL standard: with ORDER BY → default RANGE UNBOUNDED PRECEDING TO CURRENT ROW;
    /// without ORDER BY → default entire partition.
    fn resolveFrameRange(frame: ?*const ast.WindowFrameSpec, pos: usize, part_len: usize, has_order_by: bool) [2]usize {
        if (frame) |f| {
            const start = resolveOneBound(f.start, pos, part_len, true);
            const end = resolveOneBound(f.end, pos, part_len, false);
            return .{ start, @min(end, part_len) };
        }
        if (has_order_by) {
            // Default with ORDER BY: RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            return .{ 0, pos + 1 };
        }
        // Default without ORDER BY: entire partition
        return .{ 0, part_len };
    }

    fn resolveOneBound(bound: ast.WindowFrameBound, pos: usize, part_len: usize, is_start: bool) usize {
        _ = is_start;
        return switch (bound) {
            .unbounded_preceding => 0,
            .unbounded_following => part_len,
            .current_row => pos + 1, // exclusive end for current row
            .expr_preceding => |_| if (pos > 0) pos else 0, // simplified: treat as 1 preceding
            .expr_following => |_| @min(pos + 2, part_len), // simplified: treat as 1 following
        };
    }

    fn orderKeysEqual(alloc: Allocator, order_by: []const ast.OrderByItem, a: *const Row, b: *const Row) bool {
        for (order_by) |ob| {
            const av = evalExpr(alloc, ob.expr, a) catch Value.null_value;
            defer av.free(alloc);
            const bv = evalExpr(alloc, ob.expr, b) catch Value.null_value;
            defer bv.free(alloc);
            if (!av.eql(bv)) return false;
        }
        return true;
    }
};

// ── Aggregate Operator ──────────────────────────────────────────────────

/// GROUP BY with aggregate functions. Materializes all input, groups, then emits.
pub const AggregateOp = struct {
    allocator: Allocator,
    input: RowIterator,
    group_by: []const *const ast.Expr,
    aggregates: []const planner_mod.PlanNode.AggregateExpr,
    result_rows: std.ArrayListUnmanaged(Row) = .{},
    index: usize = 0,
    materialized: bool = false,

    pub fn init(
        allocator: Allocator,
        input: RowIterator,
        group_by: []const *const ast.Expr,
        aggregates: []const planner_mod.PlanNode.AggregateExpr,
    ) AggregateOp {
        return .{
            .allocator = allocator,
            .input = input,
            .group_by = group_by,
            .aggregates = aggregates,
        };
    }

    fn materialize(self: *AggregateOp) ExecError!void {
        // Collect all input rows
        var input_rows = std.ArrayListUnmanaged(Row){};
        defer {
            for (input_rows.items) |*r| r.deinit();
            input_rows.deinit(self.allocator);
        }

        while (true) {
            const row = try self.input.next() orelse break;
            input_rows.append(self.allocator, row) catch return ExecError.OutOfMemory;
        }

        if (input_rows.items.len == 0 and self.group_by.len == 0) {
            // No rows + no GROUP BY = produce one row with aggregate defaults
            try self.emitAggregateRow(&.{});
        } else if (self.group_by.len == 0) {
            // No GROUP BY = all rows in one group
            try self.emitAggregateRow(input_rows.items);
        } else {
            // Group rows by GROUP BY key
            // Simple approach: sort by group key, then scan for group boundaries
            const ctx = GroupContext{ .group_by = self.group_by, .allocator = self.allocator };
            std.sort.block(Row, input_rows.items, ctx, GroupContext.lessThan);

            var group_start: usize = 0;
            for (input_rows.items[1..], 1..) |_, i| {
                if (!groupKeysEqual(self.allocator, self.group_by, &input_rows.items[group_start], &input_rows.items[i])) {
                    try self.emitAggregateRow(input_rows.items[group_start..i]);
                    group_start = i;
                }
            }
            try self.emitAggregateRow(input_rows.items[group_start..]);
        }

        self.materialized = true;
    }

    fn emitAggregateRow(self: *AggregateOp, group: []const Row) ExecError!void {
        const total_cols = self.group_by.len + self.aggregates.len;
        const vals = self.allocator.alloc(Value, total_cols) catch return ExecError.OutOfMemory;
        var inited: usize = 0;
        errdefer {
            for (vals[0..inited]) |v| v.free(self.allocator);
            self.allocator.free(vals);
        }

        const col_names = self.allocator.alloc([]const u8, total_cols) catch return ExecError.OutOfMemory;
        errdefer self.allocator.free(col_names);

        // Group by columns from first row in group
        for (self.group_by, 0..) |gb_expr, i| {
            if (group.len > 0) {
                vals[i] = evalExpr(self.allocator, gb_expr, &group[0]) catch .null_value;
            } else {
                vals[i] = .null_value;
            }
            inited += 1;
            col_names[i] = exprColumnName(gb_expr);
        }

        // Aggregate columns
        for (self.aggregates, 0..) |agg, i| {
            const idx = self.group_by.len + i;
            vals[idx] = self.computeAggregate(agg, group);
            inited += 1;
            col_names[idx] = agg.alias orelse aggFuncName(agg.func);
        }

        self.result_rows.append(self.allocator, Row{
            .columns = col_names,
            .values = vals,
            .allocator = self.allocator,
        }) catch return ExecError.OutOfMemory;
    }

    fn computeAggregate(self: *AggregateOp, agg: planner_mod.PlanNode.AggregateExpr, group: []const Row) Value {
        switch (agg.func) {
            .count_star => return .{ .integer = @intCast(group.len) },
            .count => {
                var count: i64 = 0;
                for (group) |*row| {
                    if (agg.arg) |arg_expr| {
                        const val = evalExpr(self.allocator, arg_expr, row) catch continue;
                        defer val.free(self.allocator);
                        if (val != .null_value) count += 1;
                    }
                }
                return .{ .integer = count };
            },
            .sum => {
                var int_sum: i64 = 0;
                var float_sum: f64 = 0;
                var has_float = false;
                var has_value = false;
                for (group) |*row| {
                    if (agg.arg) |arg_expr| {
                        const val = evalExpr(self.allocator, arg_expr, row) catch continue;
                        defer val.free(self.allocator);
                        switch (val) {
                            .integer => |v| {
                                int_sum += v;
                                has_value = true;
                            },
                            .real => |v| {
                                float_sum += v;
                                has_float = true;
                                has_value = true;
                            },
                            else => {},
                        }
                    }
                }
                if (!has_value) return .null_value;
                if (has_float) return .{ .real = float_sum + @as(f64, @floatFromInt(int_sum)) };
                return .{ .integer = int_sum };
            },
            .avg => {
                var sum: f64 = 0;
                var count: f64 = 0;
                for (group) |*row| {
                    if (agg.arg) |arg_expr| {
                        const val = evalExpr(self.allocator, arg_expr, row) catch continue;
                        defer val.free(self.allocator);
                        if (val.toReal()) |v| {
                            sum += v;
                            count += 1;
                        }
                    }
                }
                if (count == 0) return .null_value;
                return .{ .real = sum / count };
            },
            .min => {
                var min_val: ?Value = null;
                for (group) |*row| {
                    if (agg.arg) |arg_expr| {
                        const val = evalExpr(self.allocator, arg_expr, row) catch continue;
                        if (val == .null_value) {
                            val.free(self.allocator);
                            continue;
                        }
                        if (min_val) |current| {
                            if (val.compare(current) == .lt) {
                                current.free(self.allocator);
                                min_val = val;
                            } else {
                                val.free(self.allocator);
                            }
                        } else {
                            min_val = val;
                        }
                    }
                }
                if (min_val) |v| {
                    defer {
                        var mv = v;
                        mv.free(self.allocator);
                    }
                    return v.dupe(self.allocator) catch .null_value;
                }
                return .null_value;
            },
            .max => {
                var max_val: ?Value = null;
                for (group) |*row| {
                    if (agg.arg) |arg_expr| {
                        const val = evalExpr(self.allocator, arg_expr, row) catch continue;
                        if (val == .null_value) {
                            val.free(self.allocator);
                            continue;
                        }
                        if (max_val) |current| {
                            if (val.compare(current) == .gt) {
                                current.free(self.allocator);
                                max_val = val;
                            } else {
                                val.free(self.allocator);
                            }
                        } else {
                            max_val = val;
                        }
                    }
                }
                if (max_val) |v| {
                    defer {
                        var mv = v;
                        mv.free(self.allocator);
                    }
                    return v.dupe(self.allocator) catch .null_value;
                }
                return .null_value;
            },
        }
    }

    pub fn next(self: *AggregateOp) ExecError!?Row {
        if (!self.materialized) try self.materialize();
        if (self.index >= self.result_rows.items.len) return null;
        const row = self.result_rows.items[self.index];
        self.index += 1;
        return row;
    }

    pub fn close(self: *AggregateOp) void {
        for (self.result_rows.items[self.index..]) |*r| r.deinit();
        self.result_rows.deinit(self.allocator);
        self.input.close();
    }

    pub fn iterator(self: *AggregateOp) RowIterator {
        return .{
            .ptr = self,
            .vtable = &.{
                .next = @ptrCast(&AggregateOp.next),
                .close = @ptrCast(&AggregateOp.close),
            },
        };
    }

    const GroupContext = struct {
        group_by: []const *const ast.Expr,
        allocator: Allocator,

        fn lessThan(ctx: GroupContext, a: Row, b: Row) bool {
            for (ctx.group_by) |expr| {
                const av = evalExpr(ctx.allocator, expr, &a) catch Value.null_value;
                defer av.free(ctx.allocator);
                const bv = evalExpr(ctx.allocator, expr, &b) catch Value.null_value;
                defer bv.free(ctx.allocator);
                const order = av.compare(bv);
                if (order == .eq) continue;
                return order == .lt;
            }
            return false;
        }
    };
};

fn groupKeysEqual(allocator: Allocator, group_by: []const *const ast.Expr, a: *const Row, b: *const Row) bool {
    for (group_by) |expr| {
        const av = evalExpr(allocator, expr, a) catch Value.null_value;
        defer av.free(allocator);
        const bv = evalExpr(allocator, expr, b) catch Value.null_value;
        defer bv.free(allocator);
        if (!av.eql(bv)) return false;
    }
    return true;
}

fn aggFuncName(func: AggFunc) []const u8 {
    return switch (func) {
        .count => "count",
        .sum => "sum",
        .avg => "avg",
        .min => "min",
        .max => "max",
        .count_star => "count(*)",
    };
}

// ── Nested Loop Join ────────────────────────────────────────────────────

/// Nested loop join — iterates all combinations of left and right rows.
pub const NestedLoopJoinOp = struct {
    allocator: Allocator,
    left: RowIterator,
    right: RowIterator,
    join_type: ast.JoinType,
    on_condition: ?*const ast.Expr,
    left_row: ?Row = null,
    right_rows: std.ArrayListUnmanaged(Row) = .{},
    right_index: usize = 0,
    right_materialized: bool = false,
    left_matched: bool = false,

    pub fn init(
        allocator: Allocator,
        left: RowIterator,
        right: RowIterator,
        join_type: ast.JoinType,
        on_condition: ?*const ast.Expr,
    ) NestedLoopJoinOp {
        return .{
            .allocator = allocator,
            .left = left,
            .right = right,
            .join_type = join_type,
            .on_condition = on_condition,
        };
    }

    fn materializeRight(self: *NestedLoopJoinOp) ExecError!void {
        while (true) {
            const row = try self.right.next() orelse break;
            self.right_rows.append(self.allocator, row) catch return ExecError.OutOfMemory;
        }
        self.right_materialized = true;
    }

    pub fn next(self: *NestedLoopJoinOp) ExecError!?Row {
        if (!self.right_materialized) try self.materializeRight();

        while (true) {
            // Get current left row
            if (self.left_row == null) {
                self.left_row = try self.left.next();
                if (self.left_row == null) return null;
                self.right_index = 0;
                self.left_matched = false;
            }

            // Try each right row
            while (self.right_index < self.right_rows.items.len) {
                const right_row = &self.right_rows.items[self.right_index];
                self.right_index += 1;

                // Build combined row
                var combined = try combineRows(self.allocator, &self.left_row.?, right_row);

                // Check join condition
                if (self.on_condition) |cond| {
                    const val = evalExpr(self.allocator, cond, &combined) catch {
                        combined.deinit();
                        continue;
                    };
                    defer val.free(self.allocator);
                    if (!val.isTruthy()) {
                        combined.deinit();
                        continue;
                    }
                }

                self.left_matched = true;
                return combined;
            }

            // Exhausted right side for current left row
            if (!self.left_matched and (self.join_type == .left or self.join_type == .full)) {
                // Emit left row with NULLs for right columns
                const result = try leftOuterRow(self.allocator, &self.left_row.?, self.right_rows.items);
                self.left_row.?.deinit();
                self.left_row = null;
                return result;
            }

            self.left_row.?.deinit();
            self.left_row = null;
        }
    }

    pub fn close(self: *NestedLoopJoinOp) void {
        if (self.left_row) |*lr| lr.deinit();
        for (self.right_rows.items) |*r| r.deinit();
        self.right_rows.deinit(self.allocator);
        self.left.close();
        self.right.close();
    }

    pub fn iterator(self: *NestedLoopJoinOp) RowIterator {
        return .{
            .ptr = self,
            .vtable = &.{
                .next = @ptrCast(&NestedLoopJoinOp.next),
                .close = @ptrCast(&NestedLoopJoinOp.close),
            },
        };
    }
};

fn combineRows(allocator: Allocator, left: *const Row, right: *const Row) ExecError!Row {
    const total = left.columns.len + right.columns.len;
    const cols = allocator.alloc([]const u8, total) catch return ExecError.OutOfMemory;
    errdefer allocator.free(cols);
    const vals = allocator.alloc(Value, total) catch return ExecError.OutOfMemory;
    var inited: usize = 0;
    errdefer {
        for (vals[0..inited]) |v| v.free(allocator);
        allocator.free(vals);
    }

    for (left.columns, 0..) |c, i| {
        cols[i] = c;
        vals[i] = left.values[i].dupe(allocator) catch return ExecError.OutOfMemory;
        inited += 1;
    }
    for (right.columns, 0..) |c, i| {
        cols[left.columns.len + i] = c;
        vals[left.columns.len + i] = right.values[i].dupe(allocator) catch return ExecError.OutOfMemory;
        inited += 1;
    }

    return Row{
        .columns = cols,
        .values = vals,
        .allocator = allocator,
    };
}

fn leftOuterRow(allocator: Allocator, left: *const Row, right_sample: []const Row) ExecError!Row {
    const right_cols = if (right_sample.len > 0) right_sample[0].columns.len else 0;
    const total = left.columns.len + right_cols;
    const cols = allocator.alloc([]const u8, total) catch return ExecError.OutOfMemory;
    errdefer allocator.free(cols);
    const vals = allocator.alloc(Value, total) catch return ExecError.OutOfMemory;
    var inited: usize = 0;
    errdefer {
        for (vals[0..inited]) |v| v.free(allocator);
        allocator.free(vals);
    }

    for (left.columns, 0..) |c, i| {
        cols[i] = c;
        vals[i] = left.values[i].dupe(allocator) catch return ExecError.OutOfMemory;
        inited += 1;
    }
    if (right_sample.len > 0) {
        for (right_sample[0].columns, 0..) |c, i| {
            cols[left.columns.len + i] = c;
            vals[left.columns.len + i] = .null_value;
            inited += 1;
        }
    }

    return Row{
        .columns = cols,
        .values = vals,
        .allocator = allocator,
    };
}

// ── Values Operator ─────────────────────────────────────────────────────

/// Produces literal rows from INSERT VALUES.
pub const ValuesOp = struct {
    allocator: Allocator,
    col_names: []const []const u8,
    rows: []const []const *const ast.Expr,
    index: usize = 0,

    pub fn init(allocator: Allocator, col_names: []const []const u8, rows: []const []const *const ast.Expr) ValuesOp {
        return .{
            .allocator = allocator,
            .col_names = col_names,
            .rows = rows,
        };
    }

    pub fn next(self: *ValuesOp) ExecError!?Row {
        if (self.index >= self.rows.len) return null;

        const exprs = self.rows[self.index];
        self.index += 1;

        // Create an empty row for evaluation context
        const empty_row = Row{
            .columns = &.{},
            .values = &.{},
            .allocator = self.allocator,
        };

        const vals = self.allocator.alloc(Value, exprs.len) catch return ExecError.OutOfMemory;
        var inited: usize = 0;
        errdefer {
            for (vals[0..inited]) |v| v.free(self.allocator);
            self.allocator.free(vals);
        }

        for (exprs, 0..) |expr, i| {
            vals[i] = evalExpr(self.allocator, expr, &empty_row) catch |err| {
                return switch (err) {
                    error.OutOfMemory => ExecError.OutOfMemory,
                    error.TypeError => ExecError.TypeError,
                    error.DivisionByZero => ExecError.DivisionByZero,
                    error.ColumnNotFound => ExecError.ColumnNotFound,
                    error.UnsupportedExpression => ExecError.UnsupportedExpression,
                };
            };
            inited += 1;
        }

        const cols = self.allocator.alloc([]const u8, self.col_names.len) catch return ExecError.OutOfMemory;
        for (self.col_names, 0..) |c, i| cols[i] = c;

        return Row{
            .columns = cols,
            .values = vals,
            .allocator = self.allocator,
        };
    }

    pub fn close(_: *ValuesOp) void {}

    pub fn iterator(self: *ValuesOp) RowIterator {
        return .{
            .ptr = self,
            .vtable = &.{
                .next = @ptrCast(&ValuesOp.next),
                .close = @ptrCast(&ValuesOp.close),
            },
        };
    }
};

// ── Empty Operator ──────────────────────────────────────────────────────

/// Produces no rows (used for DDL results).
/// Produces exactly one row with no columns (like Oracle's DUAL table).
/// This allows `SELECT <expr>` without FROM to produce one result row.
pub const EmptyOp = struct {
    allocator: Allocator,
    done: bool = false,

    pub fn next(self: *EmptyOp) ExecError!?Row {
        if (self.done) return null;
        self.done = true;
        // Return a single row with no columns — ProjectOp will evaluate
        // literal expressions against this empty row.
        const cols = self.allocator.alloc([]const u8, 0) catch return ExecError.OutOfMemory;
        const vals = self.allocator.alloc(Value, 0) catch {
            self.allocator.free(cols);
            return ExecError.OutOfMemory;
        };
        return Row{
            .columns = cols,
            .values = vals,
            .allocator = self.allocator,
        };
    }

    pub fn close(_: *EmptyOp) void {}

    pub fn iterator(self: *EmptyOp) RowIterator {
        return .{
            .ptr = self,
            .vtable = &.{
                .next = @ptrCast(&EmptyOp.next),
                .close = @ptrCast(&EmptyOp.close),
            },
        };
    }
};

// ── Materialized Operator ────────────────────────────────────────────────

/// Serves pre-materialized rows (used for view expansion).
pub const MaterializedOp = struct {
    allocator: Allocator,
    col_names: []const []const u8,
    rows: []const []Value,
    index: usize = 0,

    pub fn init(allocator: Allocator, col_names: []const []const u8, rows: []const []Value) MaterializedOp {
        return .{
            .allocator = allocator,
            .col_names = col_names,
            .rows = rows,
        };
    }

    pub fn next(self: *MaterializedOp) ExecError!?Row {
        if (self.index >= self.rows.len) return null;

        const source_vals = self.rows[self.index];
        self.index += 1;

        // Duplicate values for the returned row (Row.deinit will free them)
        const vals = self.allocator.alloc(Value, source_vals.len) catch return ExecError.OutOfMemory;
        var inited: usize = 0;
        errdefer {
            for (vals[0..inited]) |v| v.free(self.allocator);
            self.allocator.free(vals);
        }
        for (source_vals, 0..) |v, i| {
            vals[i] = v.dupe(self.allocator) catch return ExecError.OutOfMemory;
            inited += 1;
        }

        const cols = self.allocator.alloc([]const u8, self.col_names.len) catch return ExecError.OutOfMemory;
        for (self.col_names, 0..) |c, i| cols[i] = c;

        return Row{
            .columns = cols,
            .values = vals,
            .allocator = self.allocator,
        };
    }

    pub fn close(self: *MaterializedOp) void {
        for (self.rows) |vals| {
            for (vals) |v| v.free(self.allocator);
            self.allocator.free(vals);
        }
        self.allocator.free(self.rows);
        for (self.col_names) |name| self.allocator.free(@constCast(name));
        self.allocator.free(self.col_names);
    }

    pub fn iterator(self: *MaterializedOp) RowIterator {
        return .{
            .ptr = self,
            .vtable = &.{
                .next = @ptrCast(&MaterializedOp.next),
                .close = @ptrCast(&MaterializedOp.close),
            },
        };
    }
};

// ── Distinct Operator ────────────────────────────────────────────────────

/// Eliminates duplicate rows from input.
/// - Plain DISTINCT: hashes all column values, emits each unique row once.
/// - DISTINCT ON: hashes only the specified expression values, emits only
///   the first row per unique key (relies on input being sorted by ON exprs).
pub const DistinctOp = struct {
    allocator: Allocator,
    input: RowIterator,
    /// For DISTINCT ON: expressions to evaluate for the dedup key.
    /// Empty = plain DISTINCT (use all columns).
    on_exprs: []const *const ast.Expr,
    /// Hash set of seen keys for deduplication.
    seen: std.StringHashMapUnmanaged(void) = .{},
    /// Track all allocated keys for cleanup.
    allocated_keys: std.ArrayListUnmanaged([]u8) = .{},

    pub fn init(allocator: Allocator, input: RowIterator, on_exprs: []const *const ast.Expr) DistinctOp {
        return .{
            .allocator = allocator,
            .input = input,
            .on_exprs = on_exprs,
        };
    }

    /// Build the dedup key for a row.
    fn buildKey(self: *DistinctOp, row: *const Row) ExecError![]u8 {
        if (self.on_exprs.len == 0) {
            // Plain DISTINCT: serialize all columns
            return serializeRow(self.allocator, row.values) catch return ExecError.OutOfMemory;
        }

        // DISTINCT ON: evaluate each expression and serialize
        var vals = std.ArrayListUnmanaged(Value){};
        defer {
            for (vals.items) |v| v.free(self.allocator);
            vals.deinit(self.allocator);
        }
        for (self.on_exprs) |expr| {
            const v = evalExpr(self.allocator, expr, row) catch return ExecError.OutOfMemory;
            vals.append(self.allocator, v) catch {
                v.free(self.allocator);
                return ExecError.OutOfMemory;
            };
        }
        return serializeRow(self.allocator, vals.items) catch return ExecError.OutOfMemory;
    }

    pub fn next(self: *DistinctOp) ExecError!?Row {
        while (true) {
            var row = try self.input.next() orelse return null;
            const key = self.buildKey(&row) catch {
                row.deinit();
                return ExecError.OutOfMemory;
            };
            if (self.seen.contains(key)) {
                self.allocator.free(key);
                row.deinit();
                continue;
            }
            self.allocated_keys.append(self.allocator, key) catch {
                self.allocator.free(key);
                row.deinit();
                return ExecError.OutOfMemory;
            };
            self.seen.put(self.allocator, key, {}) catch {
                row.deinit();
                return ExecError.OutOfMemory;
            };
            return row;
        }
    }

    pub fn close(self: *DistinctOp) void {
        self.input.close();
        self.seen.deinit(self.allocator);
        for (self.allocated_keys.items) |key| self.allocator.free(key);
        self.allocated_keys.deinit(self.allocator);
    }

    pub fn iterator(self: *DistinctOp) RowIterator {
        return .{
            .ptr = self,
            .vtable = &.{
                .next = @ptrCast(&DistinctOp.next),
                .close = @ptrCast(&DistinctOp.close),
            },
        };
    }
};

// ── Set Operation Operator ───────────────────────────────────────────────

/// Executes UNION, UNION ALL, INTERSECT, EXCEPT between two row iterators.
/// For UNION/INTERSECT/EXCEPT, uses a hash set of serialized rows for dedup.
pub const SetOpOp = struct {
    allocator: Allocator,
    left: RowIterator,
    right: RowIterator,
    op: ast.SetOpType,
    /// Phase: true = reading from left, false = reading from right
    reading_left: bool = true,
    /// For deduplication (UNION, INTERSECT, EXCEPT):
    /// Stores serialized row keys that have been seen/materialized.
    seen: std.StringHashMapUnmanaged(void) = .{},
    /// For INTERSECT/EXCEPT: materialized right side rows (serialized keys).
    right_set: ?std.StringHashMapUnmanaged(void) = null,
    /// Track all allocated keys for cleanup.
    allocated_keys: std.ArrayListUnmanaged([]u8) = .{},
    initialized: bool = false,

    pub fn init(allocator: Allocator, left: RowIterator, right: RowIterator, op: ast.SetOpType) SetOpOp {
        return .{
            .allocator = allocator,
            .left = left,
            .right = right,
            .op = op,
        };
    }

    /// Serialize a row's values into a comparable key for dedup.
    fn rowKey(self: *SetOpOp, row: *const Row) ![]u8 {
        return serializeRow(self.allocator, row.values);
    }

    /// Materialize all rows from the right side into a hash set.
    fn materializeRight(self: *SetOpOp) !void {
        var right_set = std.StringHashMapUnmanaged(void){};
        while (try self.right.next()) |*r| {
            var row = r.*;
            defer row.deinit();
            const key = try self.rowKey(&row);
            self.allocated_keys.append(self.allocator, key) catch return ExecError.OutOfMemory;
            right_set.put(self.allocator, key, {}) catch return ExecError.OutOfMemory;
        }
        self.right_set = right_set;
    }

    pub fn next(self: *SetOpOp) ExecError!?Row {
        // Initialize right side for INTERSECT/EXCEPT
        if (!self.initialized) {
            self.initialized = true;
            if (self.op == .intersect or self.op == .except) {
                try self.materializeRight();
            }
        }

        switch (self.op) {
            .union_all => return self.nextUnionAll(),
            .@"union" => return self.nextUnion(),
            .intersect => return self.nextIntersect(),
            .except => return self.nextExcept(),
        }
    }

    fn nextUnionAll(self: *SetOpOp) ExecError!?Row {
        if (self.reading_left) {
            if (try self.left.next()) |row| return row;
            self.reading_left = false;
        }
        return self.right.next();
    }

    fn nextUnion(self: *SetOpOp) ExecError!?Row {
        // Read from left, then right, skipping duplicates
        while (self.reading_left) {
            var row = try self.left.next() orelse {
                self.reading_left = false;
                break;
            };
            const key = self.rowKey(&row) catch {
                row.deinit();
                return ExecError.OutOfMemory;
            };
            if (self.seen.contains(key)) {
                self.allocator.free(key);
                row.deinit();
                continue;
            }
            self.allocated_keys.append(self.allocator, key) catch {
                self.allocator.free(key);
                row.deinit();
                return ExecError.OutOfMemory;
            };
            self.seen.put(self.allocator, key, {}) catch {
                row.deinit();
                return ExecError.OutOfMemory;
            };
            return row;
        }

        // Right side
        while (true) {
            var row = try self.right.next() orelse return null;
            const key = self.rowKey(&row) catch {
                row.deinit();
                return ExecError.OutOfMemory;
            };
            if (self.seen.contains(key)) {
                self.allocator.free(key);
                row.deinit();
                continue;
            }
            self.allocated_keys.append(self.allocator, key) catch {
                self.allocator.free(key);
                row.deinit();
                return ExecError.OutOfMemory;
            };
            self.seen.put(self.allocator, key, {}) catch {
                row.deinit();
                return ExecError.OutOfMemory;
            };
            return row;
        }
    }

    fn nextIntersect(self: *SetOpOp) ExecError!?Row {
        const rs = self.right_set orelse return null;
        while (true) {
            var row = try self.left.next() orelse return null;
            const key = self.rowKey(&row) catch {
                row.deinit();
                return ExecError.OutOfMemory;
            };
            defer self.allocator.free(key);
            if (rs.contains(key)) {
                // Also deduplicate: don't emit same row twice
                if (!self.seen.contains(key)) {
                    const key_dup = self.allocator.dupe(u8, key) catch {
                        row.deinit();
                        return ExecError.OutOfMemory;
                    };
                    self.allocated_keys.append(self.allocator, key_dup) catch {
                        self.allocator.free(key_dup);
                        row.deinit();
                        return ExecError.OutOfMemory;
                    };
                    self.seen.put(self.allocator, key_dup, {}) catch {
                        row.deinit();
                        return ExecError.OutOfMemory;
                    };
                    return row;
                }
                row.deinit();
                continue;
            }
            row.deinit();
        }
    }

    fn nextExcept(self: *SetOpOp) ExecError!?Row {
        const rs = self.right_set orelse return null;
        while (true) {
            var row = try self.left.next() orelse return null;
            const key = self.rowKey(&row) catch {
                row.deinit();
                return ExecError.OutOfMemory;
            };
            defer self.allocator.free(key);
            if (!rs.contains(key)) {
                // Also deduplicate within left side
                if (!self.seen.contains(key)) {
                    const key_dup = self.allocator.dupe(u8, key) catch {
                        row.deinit();
                        return ExecError.OutOfMemory;
                    };
                    self.allocated_keys.append(self.allocator, key_dup) catch {
                        self.allocator.free(key_dup);
                        row.deinit();
                        return ExecError.OutOfMemory;
                    };
                    self.seen.put(self.allocator, key_dup, {}) catch {
                        row.deinit();
                        return ExecError.OutOfMemory;
                    };
                    return row;
                }
                row.deinit();
                continue;
            }
            row.deinit();
        }
    }

    pub fn close(self: *SetOpOp) void {
        self.left.close();
        self.right.close();
        self.seen.deinit(self.allocator);
        if (self.right_set) |*rs| rs.deinit(self.allocator);
        for (self.allocated_keys.items) |key| self.allocator.free(key);
        self.allocated_keys.deinit(self.allocator);
    }

    pub fn iterator(self: *SetOpOp) RowIterator {
        return .{
            .ptr = self,
            .vtable = &.{
                .next = @ptrCast(&SetOpOp.next),
                .close = @ptrCast(&SetOpOp.close),
            },
        };
    }
};

// ── Execution Result ────────────────────────────────────────────────────

/// Result of executing a SQL statement.
pub const ExecResult = struct {
    /// For SELECT: iterator that produces rows.
    iterator: ?RowIterator = null,
    /// For DML: number of rows affected.
    rows_affected: u64 = 0,
    /// Human-readable message (for DDL, etc.).
    message: []const u8 = "",
};

// ── Tests ───────────────────────────────────────────────────────────────

test "Value compare" {
    const a = Value{ .integer = 42 };
    const b = Value{ .integer = 100 };
    const c = Value{ .integer = 42 };
    const n: Value = .null_value;

    try std.testing.expectEqual(std.math.Order.lt, a.compare(b));
    try std.testing.expectEqual(std.math.Order.gt, b.compare(a));
    try std.testing.expectEqual(std.math.Order.eq, a.compare(c));
    try std.testing.expect(a.eql(c));
    try std.testing.expect(!a.eql(b));

    // NULL comparisons
    try std.testing.expectEqual(std.math.Order.gt, n.compare(a));
    try std.testing.expectEqual(std.math.Order.lt, a.compare(n));
    try std.testing.expectEqual(std.math.Order.eq, n.compare(n));
}

test "Value isTruthy" {
    try std.testing.expect(Value.isTruthy(.{ .integer = 1 }));
    try std.testing.expect(!Value.isTruthy(.{ .integer = 0 }));
    try std.testing.expect(Value.isTruthy(.{ .boolean = true }));
    try std.testing.expect(!Value.isTruthy(.{ .boolean = false }));
    try std.testing.expect(!Value.isTruthy(.null_value));
    try std.testing.expect(Value.isTruthy(.{ .text = "hello" }));
    try std.testing.expect(!Value.isTruthy(.{ .text = "" }));
}

test "Value toInteger and toReal" {
    try std.testing.expectEqual(@as(?i64, 42), (Value{ .integer = 42 }).toInteger());
    try std.testing.expectEqual(@as(?i64, 3), (Value{ .real = 3.7 }).toInteger());
    try std.testing.expectEqual(@as(?i64, 1), (Value{ .boolean = true }).toInteger());
    const null_val: Value = .null_value;
    try std.testing.expectEqual(@as(?i64, null), null_val.toInteger());

    try std.testing.expectEqual(@as(?f64, 42.0), (Value{ .integer = 42 }).toReal());
    try std.testing.expectEqual(@as(?f64, 3.14), (Value{ .real = 3.14 }).toReal());
}

test "Value dupe and free" {
    const allocator = std.testing.allocator;
    const original = Value{ .text = "hello" };
    const duped = try original.dupe(allocator);
    defer duped.free(allocator);

    try std.testing.expectEqualStrings("hello", duped.text);
    // Verify it's a different allocation
    try std.testing.expect(original.text.ptr != duped.text.ptr);
}

test "Row getColumn" {
    const allocator = std.testing.allocator;
    const cols = try allocator.alloc([]const u8, 2);
    defer allocator.free(cols);
    cols[0] = "id";
    cols[1] = "name";

    const vals = try allocator.alloc(Value, 2);
    defer allocator.free(vals);
    vals[0] = .{ .integer = 1 };
    vals[1] = .{ .text = "Alice" };

    const row = Row{ .columns = cols, .values = vals, .allocator = allocator };

    try std.testing.expectEqual(@as(i64, 1), row.getColumn("id").?.integer);
    try std.testing.expectEqualStrings("Alice", row.getColumn("name").?.text);
    try std.testing.expectEqual(@as(?Value, null), row.getColumn("age"));
}

test "serialize and deserialize row" {
    const allocator = std.testing.allocator;
    const values = [_]Value{
        .{ .integer = 42 },
        .{ .text = "hello" },
        .{ .boolean = true },
        .null_value,
        .{ .real = 3.14 },
    };

    const data = try serializeRow(allocator, &values);
    defer allocator.free(data);

    const result = try deserializeRow(allocator, data);
    defer {
        for (result) |v| v.free(allocator);
        allocator.free(result);
    }

    try std.testing.expectEqual(@as(usize, 5), result.len);
    try std.testing.expectEqual(@as(i64, 42), result[0].integer);
    try std.testing.expectEqualStrings("hello", result[1].text);
    try std.testing.expect(result[2].boolean);
    try std.testing.expectEqual(Value.null_value, result[3]);
    try std.testing.expectEqual(@as(f64, 3.14), result[4].real);
}

test "evalExpr literals" {
    const allocator = std.testing.allocator;
    const empty_row = Row{ .columns = &.{}, .values = &.{}, .allocator = allocator };

    const int_expr = ast.Expr{ .integer_literal = 42 };
    const v1 = try evalExpr(allocator, &int_expr, &empty_row);
    defer v1.free(allocator);
    try std.testing.expectEqual(@as(i64, 42), v1.integer);

    const str_expr = ast.Expr{ .string_literal = "hello" };
    const v2 = try evalExpr(allocator, &str_expr, &empty_row);
    defer v2.free(allocator);
    try std.testing.expectEqualStrings("hello", v2.text);

    const bool_expr = ast.Expr{ .boolean_literal = true };
    const v3 = try evalExpr(allocator, &bool_expr, &empty_row);
    try std.testing.expect(v3.boolean);

    const null_expr = ast.Expr{ .null_literal = {} };
    const v4 = try evalExpr(allocator, &null_expr, &empty_row);
    try std.testing.expectEqual(Value.null_value, v4);
}

test "evalExpr column reference" {
    const allocator = std.testing.allocator;
    const cols = try allocator.alloc([]const u8, 2);
    defer allocator.free(cols);
    cols[0] = "id";
    cols[1] = "name";

    const vals = try allocator.alloc(Value, 2);
    defer allocator.free(vals);
    vals[0] = .{ .integer = 1 };
    vals[1] = .{ .text = "Alice" };

    const row = Row{ .columns = cols, .values = vals, .allocator = allocator };

    const ref_expr = ast.Expr{ .column_ref = .{ .name = "id" } };
    const v = try evalExpr(allocator, &ref_expr, &row);
    defer v.free(allocator);
    try std.testing.expectEqual(@as(i64, 1), v.integer);

    const bad_ref = ast.Expr{ .column_ref = .{ .name = "nonexistent" } };
    try std.testing.expectError(EvalError.ColumnNotFound, evalExpr(allocator, &bad_ref, &row));
}

test "evalExpr binary arithmetic" {
    const allocator = std.testing.allocator;
    const empty_row = Row{ .columns = &.{}, .values = &.{}, .allocator = allocator };

    const left = ast.Expr{ .integer_literal = 10 };
    const right = ast.Expr{ .integer_literal = 3 };
    const add_expr = ast.Expr{ .binary_op = .{ .op = .add, .left = &left, .right = &right } };
    const v = try evalExpr(allocator, &add_expr, &empty_row);
    try std.testing.expectEqual(@as(i64, 13), v.integer);

    const sub_expr = ast.Expr{ .binary_op = .{ .op = .subtract, .left = &left, .right = &right } };
    const v2 = try evalExpr(allocator, &sub_expr, &empty_row);
    try std.testing.expectEqual(@as(i64, 7), v2.integer);

    const mul_expr = ast.Expr{ .binary_op = .{ .op = .multiply, .left = &left, .right = &right } };
    const v3 = try evalExpr(allocator, &mul_expr, &empty_row);
    try std.testing.expectEqual(@as(i64, 30), v3.integer);

    const div_expr = ast.Expr{ .binary_op = .{ .op = .divide, .left = &left, .right = &right } };
    const v4 = try evalExpr(allocator, &div_expr, &empty_row);
    try std.testing.expectEqual(@as(i64, 3), v4.integer);

    const zero = ast.Expr{ .integer_literal = 0 };
    const div_zero = ast.Expr{ .binary_op = .{ .op = .divide, .left = &left, .right = &zero } };
    try std.testing.expectError(EvalError.DivisionByZero, evalExpr(allocator, &div_zero, &empty_row));
}

test "evalExpr comparison" {
    const allocator = std.testing.allocator;
    const empty_row = Row{ .columns = &.{}, .values = &.{}, .allocator = allocator };

    const left = ast.Expr{ .integer_literal = 10 };
    const right = ast.Expr{ .integer_literal = 20 };

    const lt = ast.Expr{ .binary_op = .{ .op = .less_than, .left = &left, .right = &right } };
    const v = try evalExpr(allocator, &lt, &empty_row);
    try std.testing.expect(v.boolean);

    const eq = ast.Expr{ .binary_op = .{ .op = .equal, .left = &left, .right = &left } };
    const v2 = try evalExpr(allocator, &eq, &empty_row);
    try std.testing.expect(v2.boolean);

    const neq = ast.Expr{ .binary_op = .{ .op = .not_equal, .left = &left, .right = &right } };
    const v3 = try evalExpr(allocator, &neq, &empty_row);
    try std.testing.expect(v3.boolean);
}

test "evalExpr logical AND/OR" {
    const allocator = std.testing.allocator;
    const empty_row = Row{ .columns = &.{}, .values = &.{}, .allocator = allocator };

    const t = ast.Expr{ .boolean_literal = true };
    const f = ast.Expr{ .boolean_literal = false };

    const and_expr = ast.Expr{ .binary_op = .{ .op = .@"and", .left = &t, .right = &f } };
    const v = try evalExpr(allocator, &and_expr, &empty_row);
    try std.testing.expect(!v.boolean);

    const or_expr = ast.Expr{ .binary_op = .{ .op = .@"or", .left = &t, .right = &f } };
    const v2 = try evalExpr(allocator, &or_expr, &empty_row);
    try std.testing.expect(v2.boolean);
}

test "evalExpr AND/OR with NULL (three-valued logic)" {
    const allocator = std.testing.allocator;
    const empty_row = Row{ .columns = &.{}, .values = &.{}, .allocator = allocator };

    const t = ast.Expr{ .boolean_literal = true };
    const f = ast.Expr{ .boolean_literal = false };
    const n = ast.Expr{ .null_literal = {} };

    // FALSE AND NULL = FALSE
    const and_fn = ast.Expr{ .binary_op = .{ .op = .@"and", .left = &f, .right = &n } };
    const v1 = try evalExpr(allocator, &and_fn, &empty_row);
    try std.testing.expect(v1 == .boolean and !v1.boolean);

    // NULL AND FALSE = FALSE (commutative)
    const and_nf = ast.Expr{ .binary_op = .{ .op = .@"and", .left = &n, .right = &f } };
    const v2 = try evalExpr(allocator, &and_nf, &empty_row);
    try std.testing.expect(v2 == .boolean and !v2.boolean);

    // TRUE AND NULL = NULL
    const and_tn = ast.Expr{ .binary_op = .{ .op = .@"and", .left = &t, .right = &n } };
    const v3 = try evalExpr(allocator, &and_tn, &empty_row);
    try std.testing.expect(v3 == .null_value);

    // NULL AND TRUE = NULL
    const and_nt = ast.Expr{ .binary_op = .{ .op = .@"and", .left = &n, .right = &t } };
    const v4 = try evalExpr(allocator, &and_nt, &empty_row);
    try std.testing.expect(v4 == .null_value);

    // TRUE OR NULL = TRUE
    const or_tn = ast.Expr{ .binary_op = .{ .op = .@"or", .left = &t, .right = &n } };
    const v5 = try evalExpr(allocator, &or_tn, &empty_row);
    try std.testing.expect(v5 == .boolean and v5.boolean);

    // NULL OR TRUE = TRUE (commutative)
    const or_nt = ast.Expr{ .binary_op = .{ .op = .@"or", .left = &n, .right = &t } };
    const v6 = try evalExpr(allocator, &or_nt, &empty_row);
    try std.testing.expect(v6 == .boolean and v6.boolean);

    // FALSE OR NULL = NULL
    const or_fn = ast.Expr{ .binary_op = .{ .op = .@"or", .left = &f, .right = &n } };
    const v7 = try evalExpr(allocator, &or_fn, &empty_row);
    try std.testing.expect(v7 == .null_value);

    // NULL OR FALSE = NULL
    const or_nf = ast.Expr{ .binary_op = .{ .op = .@"or", .left = &n, .right = &f } };
    const v8 = try evalExpr(allocator, &or_nf, &empty_row);
    try std.testing.expect(v8 == .null_value);

    // NULL AND NULL = NULL
    const and_nn = ast.Expr{ .binary_op = .{ .op = .@"and", .left = &n, .right = &n } };
    const v9 = try evalExpr(allocator, &and_nn, &empty_row);
    try std.testing.expect(v9 == .null_value);

    // NULL OR NULL = NULL
    const or_nn = ast.Expr{ .binary_op = .{ .op = .@"or", .left = &n, .right = &n } };
    const v10 = try evalExpr(allocator, &or_nn, &empty_row);
    try std.testing.expect(v10 == .null_value);
}

test "evalExpr IS NULL / IS NOT NULL" {
    const allocator = std.testing.allocator;
    const empty_row = Row{ .columns = &.{}, .values = &.{}, .allocator = allocator };

    const null_expr = ast.Expr{ .null_literal = {} };
    const is_null = ast.Expr{ .is_null = .{ .expr = &null_expr } };
    const v = try evalExpr(allocator, &is_null, &empty_row);
    try std.testing.expect(v.boolean);

    const is_not_null = ast.Expr{ .is_null = .{ .expr = &null_expr, .negated = true } };
    const v2 = try evalExpr(allocator, &is_not_null, &empty_row);
    try std.testing.expect(!v2.boolean);

    const int_expr = ast.Expr{ .integer_literal = 42 };
    const not_null = ast.Expr{ .is_null = .{ .expr = &int_expr } };
    const v3 = try evalExpr(allocator, &not_null, &empty_row);
    try std.testing.expect(!v3.boolean);
}

test "evalExpr BETWEEN" {
    const allocator = std.testing.allocator;
    const empty_row = Row{ .columns = &.{}, .values = &.{}, .allocator = allocator };

    const val = ast.Expr{ .integer_literal = 5 };
    const low = ast.Expr{ .integer_literal = 1 };
    const high = ast.Expr{ .integer_literal = 10 };

    const between = ast.Expr{ .between = .{ .expr = &val, .low = &low, .high = &high } };
    const v = try evalExpr(allocator, &between, &empty_row);
    try std.testing.expect(v.boolean);

    const out = ast.Expr{ .integer_literal = 15 };
    const not_between = ast.Expr{ .between = .{ .expr = &out, .low = &low, .high = &high } };
    const v2 = try evalExpr(allocator, &not_between, &empty_row);
    try std.testing.expect(!v2.boolean);
}

test "evalExpr IN list" {
    const allocator = std.testing.allocator;
    const empty_row = Row{ .columns = &.{}, .values = &.{}, .allocator = allocator };

    const val = ast.Expr{ .integer_literal = 3 };
    const item1 = ast.Expr{ .integer_literal = 1 };
    const item2 = ast.Expr{ .integer_literal = 3 };
    const item3 = ast.Expr{ .integer_literal = 5 };
    const list = [_]*const ast.Expr{ &item1, &item2, &item3 };

    const in_expr = ast.Expr{ .in_list = .{ .expr = &val, .list = &list } };
    const v = try evalExpr(allocator, &in_expr, &empty_row);
    try std.testing.expect(v.boolean);

    const val2 = ast.Expr{ .integer_literal = 4 };
    const not_in = ast.Expr{ .in_list = .{ .expr = &val2, .list = &list } };
    const v2 = try evalExpr(allocator, &not_in, &empty_row);
    try std.testing.expect(!v2.boolean);
}

test "evalExpr LIKE" {
    const allocator = std.testing.allocator;
    const empty_row = Row{ .columns = &.{}, .values = &.{}, .allocator = allocator };

    const text = ast.Expr{ .string_literal = "hello world" };
    const pat1 = ast.Expr{ .string_literal = "hello%" };
    const like1 = ast.Expr{ .like = .{ .expr = &text, .pattern = &pat1 } };
    const v1 = try evalExpr(allocator, &like1, &empty_row);
    defer v1.free(allocator);
    try std.testing.expect(v1.boolean);

    const pat2 = ast.Expr{ .string_literal = "h_llo%" };
    const like2 = ast.Expr{ .like = .{ .expr = &text, .pattern = &pat2 } };
    const v2 = try evalExpr(allocator, &like2, &empty_row);
    defer v2.free(allocator);
    try std.testing.expect(v2.boolean);

    const pat3 = ast.Expr{ .string_literal = "goodbye%" };
    const like3 = ast.Expr{ .like = .{ .expr = &text, .pattern = &pat3 } };
    const v3 = try evalExpr(allocator, &like3, &empty_row);
    defer v3.free(allocator);
    try std.testing.expect(!v3.boolean);
}

test "evalExpr unary negation" {
    const allocator = std.testing.allocator;
    const empty_row = Row{ .columns = &.{}, .values = &.{}, .allocator = allocator };

    const inner = ast.Expr{ .integer_literal = 42 };
    const neg = ast.Expr{ .unary_op = .{ .op = .negate, .operand = &inner } };
    const v = try evalExpr(allocator, &neg, &empty_row);
    try std.testing.expectEqual(@as(i64, -42), v.integer);
}

test "evalExpr NOT" {
    const allocator = std.testing.allocator;
    const empty_row = Row{ .columns = &.{}, .values = &.{}, .allocator = allocator };

    const inner = ast.Expr{ .boolean_literal = true };
    const not_expr = ast.Expr{ .unary_op = .{ .op = .not, .operand = &inner } };
    const v = try evalExpr(allocator, &not_expr, &empty_row);
    try std.testing.expect(!v.boolean);
}

test "LIKE pattern matching" {
    try std.testing.expect(likeMatch("hello", "hello"));
    try std.testing.expect(likeMatch("hello", "%"));
    try std.testing.expect(likeMatch("hello", "h%"));
    try std.testing.expect(likeMatch("hello", "%o"));
    try std.testing.expect(likeMatch("hello", "h_llo"));
    try std.testing.expect(likeMatch("hello", "%ll%"));
    try std.testing.expect(!likeMatch("hello", "world"));
    try std.testing.expect(!likeMatch("hello", "h_lo"));
    try std.testing.expect(likeMatch("", ""));
    try std.testing.expect(likeMatch("", "%"));
    try std.testing.expect(!likeMatch("", "_"));
}

test "LIKE pattern matching: extended cases" {
    // Multiple wildcards
    try std.testing.expect(likeMatch("abcdef", "a%c%f"));
    try std.testing.expect(likeMatch("abcdef", "%b%e%"));
    try std.testing.expect(!likeMatch("abcdef", "a%c%z"));

    // Multiple underscores
    try std.testing.expect(likeMatch("abc", "___"));
    try std.testing.expect(!likeMatch("abcd", "___"));
    try std.testing.expect(!likeMatch("ab", "___"));

    // Mixed % and _
    try std.testing.expect(likeMatch("abc", "_%c"));
    try std.testing.expect(likeMatch("abc", "a_%"));
    try std.testing.expect(likeMatch("a", "%_%")); // at least 1 char
    try std.testing.expect(!likeMatch("", "%_%")); // empty doesn't match "at least 1"

    // Case insensitivity
    try std.testing.expect(likeMatch("Hello", "hello"));
    try std.testing.expect(likeMatch("HELLO", "h%o"));

    // Consecutive percent signs (should behave like single %)
    try std.testing.expect(likeMatch("abc", "%%"));
    try std.testing.expect(likeMatch("abc", "a%%c"));

    // Pattern longer than text
    try std.testing.expect(!likeMatch("ab", "abc"));
    try std.testing.expect(!likeMatch("a", "ab"));

    // Single char text
    try std.testing.expect(likeMatch("x", "_"));
    try std.testing.expect(likeMatch("x", "%"));
    try std.testing.expect(likeMatch("x", "x"));
    try std.testing.expect(!likeMatch("x", "y"));
}

test "FilterOp filters rows" {
    const allocator = std.testing.allocator;

    // Create a simple in-memory data source
    var data = InMemorySource.init(allocator, &.{ "id", "name" });
    try data.addRow(&.{ Value{ .integer = 1 }, Value{ .text = "Alice" } });
    try data.addRow(&.{ Value{ .integer = 2 }, Value{ .text = "Bob" } });
    try data.addRow(&.{ Value{ .integer = 3 }, Value{ .text = "Charlie" } });
    defer data.deinit();

    // Filter: id > 1
    const id_ref = ast.Expr{ .column_ref = .{ .name = "id" } };
    const one = ast.Expr{ .integer_literal = 1 };
    const predicate = ast.Expr{ .binary_op = .{ .op = .greater_than, .left = &id_ref, .right = &one } };

    var filter = FilterOp.init(allocator, data.iterator(), &predicate);
    defer filter.close();

    var row1 = (try filter.next()).?;
    defer row1.deinit();
    try std.testing.expectEqual(@as(i64, 2), row1.getColumn("id").?.integer);

    var row2 = (try filter.next()).?;
    defer row2.deinit();
    try std.testing.expectEqual(@as(i64, 3), row2.getColumn("id").?.integer);

    try std.testing.expectEqual(@as(?Row, null), try filter.next());
}

test "ProjectOp selects columns" {
    const allocator = std.testing.allocator;

    var data = InMemorySource.init(allocator, &.{ "id", "name", "age" });
    try data.addRow(&.{ Value{ .integer = 1 }, Value{ .text = "Alice" }, Value{ .integer = 30 } });
    defer data.deinit();

    const name_ref = ast.Expr{ .column_ref = .{ .name = "name" } };
    const cols = [_]PlanNode.ProjectColumn{
        .{ .expr = &name_ref, .alias = "user_name" },
    };

    var proj = ProjectOp.init(allocator, data.iterator(), &cols);
    defer proj.close();

    var row = (try proj.next()).?;
    defer row.deinit();
    try std.testing.expectEqual(@as(usize, 1), row.columns.len);
    try std.testing.expectEqualStrings("user_name", row.columns[0]);
    try std.testing.expectEqualStrings("Alice", row.values[0].text);

    try std.testing.expectEqual(@as(?Row, null), try proj.next());
}

test "LimitOp with limit and offset" {
    const allocator = std.testing.allocator;

    var data = InMemorySource.init(allocator, &.{"id"});
    try data.addRow(&.{Value{ .integer = 1 }});
    try data.addRow(&.{Value{ .integer = 2 }});
    try data.addRow(&.{Value{ .integer = 3 }});
    try data.addRow(&.{Value{ .integer = 4 }});
    try data.addRow(&.{Value{ .integer = 5 }});
    defer data.deinit();

    var limit = LimitOp.init(allocator, data.iterator(), 2, 1);
    defer limit.close();

    var r1 = (try limit.next()).?;
    defer r1.deinit();
    try std.testing.expectEqual(@as(i64, 2), r1.getColumn("id").?.integer);

    var r2 = (try limit.next()).?;
    defer r2.deinit();
    try std.testing.expectEqual(@as(i64, 3), r2.getColumn("id").?.integer);

    try std.testing.expectEqual(@as(?Row, null), try limit.next());
}

test "SortOp sorts rows" {
    const allocator = std.testing.allocator;

    var data = InMemorySource.init(allocator, &.{"val"});
    try data.addRow(&.{Value{ .integer = 3 }});
    try data.addRow(&.{Value{ .integer = 1 }});
    try data.addRow(&.{Value{ .integer = 2 }});
    defer data.deinit();

    const val_ref = ast.Expr{ .column_ref = .{ .name = "val" } };
    const order = [_]ast.OrderByItem{
        .{ .expr = &val_ref, .direction = .asc },
    };

    var sort = SortOp.init(allocator, data.iterator(), &order);
    defer sort.close();

    var r1 = (try sort.next()).?;
    defer r1.deinit();
    try std.testing.expectEqual(@as(i64, 1), r1.getColumn("val").?.integer);

    var r2 = (try sort.next()).?;
    defer r2.deinit();
    try std.testing.expectEqual(@as(i64, 2), r2.getColumn("val").?.integer);

    var r3 = (try sort.next()).?;
    defer r3.deinit();
    try std.testing.expectEqual(@as(i64, 3), r3.getColumn("val").?.integer);

    try std.testing.expectEqual(@as(?Row, null), try sort.next());
}

test "SortOp descending" {
    const allocator = std.testing.allocator;

    var data = InMemorySource.init(allocator, &.{"val"});
    try data.addRow(&.{Value{ .integer = 1 }});
    try data.addRow(&.{Value{ .integer = 3 }});
    try data.addRow(&.{Value{ .integer = 2 }});
    defer data.deinit();

    const val_ref = ast.Expr{ .column_ref = .{ .name = "val" } };
    const order = [_]ast.OrderByItem{
        .{ .expr = &val_ref, .direction = .desc },
    };

    var sort = SortOp.init(allocator, data.iterator(), &order);
    defer sort.close();

    var r1 = (try sort.next()).?;
    defer r1.deinit();
    try std.testing.expectEqual(@as(i64, 3), r1.getColumn("val").?.integer);

    var r2 = (try sort.next()).?;
    defer r2.deinit();
    try std.testing.expectEqual(@as(i64, 2), r2.getColumn("val").?.integer);

    var r3 = (try sort.next()).?;
    defer r3.deinit();
    try std.testing.expectEqual(@as(i64, 1), r3.getColumn("val").?.integer);
}

test "AggregateOp count_star" {
    const allocator = std.testing.allocator;

    var data = InMemorySource.init(allocator, &.{"id"});
    try data.addRow(&.{Value{ .integer = 1 }});
    try data.addRow(&.{Value{ .integer = 2 }});
    try data.addRow(&.{Value{ .integer = 3 }});
    defer data.deinit();

    const aggs = [_]planner_mod.PlanNode.AggregateExpr{
        .{ .func = .count_star, .alias = "cnt" },
    };

    var agg = AggregateOp.init(allocator, data.iterator(), &.{}, &aggs);
    defer agg.close();

    var row = (try agg.next()).?;
    defer row.deinit();
    try std.testing.expectEqual(@as(i64, 3), row.getColumn("cnt").?.integer);

    try std.testing.expectEqual(@as(?Row, null), try agg.next());
}

test "AggregateOp sum and avg" {
    const allocator = std.testing.allocator;

    var data = InMemorySource.init(allocator, &.{"val"});
    try data.addRow(&.{Value{ .integer = 10 }});
    try data.addRow(&.{Value{ .integer = 20 }});
    try data.addRow(&.{Value{ .integer = 30 }});
    defer data.deinit();

    const val_ref = ast.Expr{ .column_ref = .{ .name = "val" } };
    const aggs = [_]planner_mod.PlanNode.AggregateExpr{
        .{ .func = .sum, .arg = &val_ref, .alias = "total" },
        .{ .func = .avg, .arg = &val_ref, .alias = "average" },
    };

    var agg = AggregateOp.init(allocator, data.iterator(), &.{}, &aggs);
    defer agg.close();

    var row = (try agg.next()).?;
    defer row.deinit();
    try std.testing.expectEqual(@as(i64, 60), row.getColumn("total").?.integer);
    try std.testing.expectEqual(@as(f64, 20.0), row.getColumn("average").?.real);
}

test "AggregateOp min and max" {
    const allocator = std.testing.allocator;

    var data = InMemorySource.init(allocator, &.{"val"});
    try data.addRow(&.{Value{ .integer = 30 }});
    try data.addRow(&.{Value{ .integer = 10 }});
    try data.addRow(&.{Value{ .integer = 20 }});
    defer data.deinit();

    const val_ref = ast.Expr{ .column_ref = .{ .name = "val" } };
    const aggs = [_]planner_mod.PlanNode.AggregateExpr{
        .{ .func = .min, .arg = &val_ref, .alias = "min_val" },
        .{ .func = .max, .arg = &val_ref, .alias = "max_val" },
    };

    var agg = AggregateOp.init(allocator, data.iterator(), &.{}, &aggs);
    defer agg.close();

    var row = (try agg.next()).?;
    defer row.deinit();
    try std.testing.expectEqual(@as(i64, 10), row.getColumn("min_val").?.integer);
    try std.testing.expectEqual(@as(i64, 30), row.getColumn("max_val").?.integer);
}

test "AggregateOp with GROUP BY" {
    const allocator = std.testing.allocator;

    var data = InMemorySource.init(allocator, &.{ "dept", "salary" });
    try data.addRow(&.{ Value{ .text = "eng" }, Value{ .integer = 100 } });
    try data.addRow(&.{ Value{ .text = "eng" }, Value{ .integer = 200 } });
    try data.addRow(&.{ Value{ .text = "hr" }, Value{ .integer = 150 } });
    defer data.deinit();

    const dept_ref = ast.Expr{ .column_ref = .{ .name = "dept" } };
    const salary_ref = ast.Expr{ .column_ref = .{ .name = "salary" } };
    const group_by = [_]*const ast.Expr{&dept_ref};
    const aggs = [_]planner_mod.PlanNode.AggregateExpr{
        .{ .func = .sum, .arg = &salary_ref, .alias = "total" },
    };

    var agg = AggregateOp.init(allocator, data.iterator(), &group_by, &aggs);
    defer agg.close();

    var r1 = (try agg.next()).?;
    defer r1.deinit();
    // Sorted by group key — "eng" first
    try std.testing.expectEqualStrings("eng", r1.getColumn("dept").?.text);
    try std.testing.expectEqual(@as(i64, 300), r1.getColumn("total").?.integer);

    var r2 = (try agg.next()).?;
    defer r2.deinit();
    try std.testing.expectEqualStrings("hr", r2.getColumn("dept").?.text);
    try std.testing.expectEqual(@as(i64, 150), r2.getColumn("total").?.integer);

    try std.testing.expectEqual(@as(?Row, null), try agg.next());
}

test "ValuesOp produces literal rows" {
    const allocator = std.testing.allocator;

    const e1 = ast.Expr{ .integer_literal = 1 };
    const e2 = ast.Expr{ .string_literal = "Alice" };
    const e3 = ast.Expr{ .integer_literal = 2 };
    const e4 = ast.Expr{ .string_literal = "Bob" };
    const row1 = [_]*const ast.Expr{ &e1, &e2 };
    const row2 = [_]*const ast.Expr{ &e3, &e4 };
    const rows = [_][]const *const ast.Expr{ &row1, &row2 };

    var values = ValuesOp.init(allocator, &.{ "id", "name" }, &rows);
    defer values.close();

    var r1 = (try values.next()).?;
    defer r1.deinit();
    try std.testing.expectEqual(@as(i64, 1), r1.getColumn("id").?.integer);
    try std.testing.expectEqualStrings("Alice", r1.getColumn("name").?.text);

    var r2 = (try values.next()).?;
    defer r2.deinit();
    try std.testing.expectEqual(@as(i64, 2), r2.getColumn("id").?.integer);
    try std.testing.expectEqualStrings("Bob", r2.getColumn("name").?.text);

    try std.testing.expectEqual(@as(?Row, null), try values.next());
}

test "NestedLoopJoinOp inner join" {
    const allocator = std.testing.allocator;

    var left = InMemorySource.init(allocator, &.{ "users.id", "users.name" });
    try left.addRow(&.{ Value{ .integer = 1 }, Value{ .text = "Alice" } });
    try left.addRow(&.{ Value{ .integer = 2 }, Value{ .text = "Bob" } });
    defer left.deinit();

    var right = InMemorySource.init(allocator, &.{ "orders.user_id", "orders.amount" });
    try right.addRow(&.{ Value{ .integer = 1 }, Value{ .integer = 100 } });
    try right.addRow(&.{ Value{ .integer = 1 }, Value{ .integer = 200 } });
    try right.addRow(&.{ Value{ .integer = 3 }, Value{ .integer = 50 } });
    defer right.deinit();

    // ON users.id = orders.user_id
    const left_ref = ast.Expr{ .column_ref = .{ .name = "id", .prefix = "users" } };
    const right_ref = ast.Expr{ .column_ref = .{ .name = "user_id", .prefix = "orders" } };
    const cond = ast.Expr{ .binary_op = .{ .op = .equal, .left = &left_ref, .right = &right_ref } };

    var join = NestedLoopJoinOp.init(allocator, left.iterator(), right.iterator(), .inner, &cond);
    defer join.close();

    var r1 = (try join.next()).?;
    defer r1.deinit();
    try std.testing.expectEqual(@as(i64, 1), r1.getQualifiedColumn("users", "id").?.integer);
    try std.testing.expectEqual(@as(i64, 100), r1.getQualifiedColumn("orders", "amount").?.integer);

    var r2 = (try join.next()).?;
    defer r2.deinit();
    try std.testing.expectEqual(@as(i64, 1), r2.getQualifiedColumn("users", "id").?.integer);
    try std.testing.expectEqual(@as(i64, 200), r2.getQualifiedColumn("orders", "amount").?.integer);

    // Bob (id=2) has no matching orders, so no row emitted
    try std.testing.expectEqual(@as(?Row, null), try join.next());
}

test "EmptyOp produces one dual row" {
    var empty = EmptyOp{ .allocator = std.testing.allocator };
    // EmptyOp produces exactly one row with no columns (DUAL table behavior)
    var row = (try empty.next()) orelse return error.ExpectedRow;
    defer row.deinit();
    try std.testing.expectEqual(@as(usize, 0), row.values.len);
    try std.testing.expectEqual(@as(usize, 0), row.columns.len);
    // Second call returns null
    try std.testing.expectEqual(@as(?Row, null), try empty.next());
}

test "evalExpr CASE expression" {
    const allocator = std.testing.allocator;
    const empty_row = Row{ .columns = &.{}, .values = &.{}, .allocator = allocator };

    // CASE WHEN true THEN 1 ELSE 0 END
    const when_cond = ast.Expr{ .boolean_literal = true };
    const then_val = ast.Expr{ .integer_literal = 1 };
    const else_val = ast.Expr{ .integer_literal = 0 };
    const when_clauses = [_]ast.WhenClause{
        .{ .condition = &when_cond, .result = &then_val },
    };
    const case_expr = ast.Expr{ .case_expr = .{ .when_clauses = &when_clauses, .else_expr = &else_val } };
    const v = try evalExpr(allocator, &case_expr, &empty_row);
    defer v.free(allocator);
    try std.testing.expectEqual(@as(i64, 1), v.integer);
}

test "evalExpr NULL arithmetic propagation" {
    const allocator = std.testing.allocator;
    const empty_row = Row{ .columns = &.{}, .values = &.{}, .allocator = allocator };

    const int_val = ast.Expr{ .integer_literal = 5 };
    const null_val = ast.Expr{ .null_literal = {} };
    const add_null = ast.Expr{ .binary_op = .{ .op = .add, .left = &int_val, .right = &null_val } };
    const v = try evalExpr(allocator, &add_null, &empty_row);
    try std.testing.expectEqual(Value.null_value, v);
}

test "AggregateOp empty input no GROUP BY" {
    const allocator = std.testing.allocator;

    var data = InMemorySource.init(allocator, &.{"val"});
    defer data.deinit();

    const val_ref = ast.Expr{ .column_ref = .{ .name = "val" } };
    const aggs = [_]planner_mod.PlanNode.AggregateExpr{
        .{ .func = .count_star, .alias = "cnt" },
        .{ .func = .sum, .arg = &val_ref, .alias = "total" },
    };

    var agg = AggregateOp.init(allocator, data.iterator(), &.{}, &aggs);
    defer agg.close();

    // With no GROUP BY and no rows, should emit one row with defaults
    var row = (try agg.next()).?;
    defer row.deinit();
    try std.testing.expectEqual(@as(i64, 0), row.getColumn("cnt").?.integer);
    try std.testing.expectEqual(Value.null_value, row.getColumn("total").?);
}

test "evalExpr string concatenation" {
    const allocator = std.testing.allocator;
    const empty_row = Row{ .columns = &.{}, .values = &.{}, .allocator = allocator };

    const left = ast.Expr{ .string_literal = "hello" };
    const right = ast.Expr{ .string_literal = " world" };
    const concat = ast.Expr{ .binary_op = .{ .op = .concat, .left = &left, .right = &right } };
    const v = try evalExpr(allocator, &concat, &empty_row);
    defer v.free(allocator);
    try std.testing.expectEqualStrings("hello world", v.text);
}

test "evalExpr CAST" {
    const allocator = std.testing.allocator;
    const empty_row = Row{ .columns = &.{}, .values = &.{}, .allocator = allocator };

    const int_val = ast.Expr{ .integer_literal = 42 };
    const cast_expr = ast.Expr{ .cast = .{ .expr = &int_val, .target_type = .type_text } };
    const v = try evalExpr(allocator, &cast_expr, &empty_row);
    defer v.free(allocator);
    try std.testing.expectEqualStrings("42", v.text);
}

// ── Test Helper: In-Memory Row Source ───────────────────────────────────

/// A simple in-memory row source for testing operators without storage.
const InMemorySource = struct {
    allocator: Allocator,
    col_names: []const []const u8,
    rows: std.ArrayListUnmanaged([]Value),
    index: usize = 0,

    fn init(allocator: Allocator, col_names: []const []const u8) InMemorySource {
        return .{
            .allocator = allocator,
            .col_names = col_names,
            .rows = .{},
        };
    }

    fn addRow(self: *InMemorySource, values: []const Value) !void {
        const row_vals = try self.allocator.alloc(Value, values.len);
        for (values, 0..) |v, i| {
            row_vals[i] = try v.dupe(self.allocator);
        }
        try self.rows.append(self.allocator, row_vals);
    }

    fn deinit(self: *InMemorySource) void {
        for (self.rows.items) |row_vals| {
            for (row_vals) |v| v.free(self.allocator);
            self.allocator.free(row_vals);
        }
        self.rows.deinit(self.allocator);
    }

    fn nextFn(ptr: *anyopaque) ExecError!?Row {
        const self: *InMemorySource = @ptrCast(@alignCast(ptr));
        if (self.index >= self.rows.items.len) return null;

        const row_vals = self.rows.items[self.index];
        self.index += 1;

        const vals = self.allocator.alloc(Value, row_vals.len) catch return ExecError.OutOfMemory;
        errdefer self.allocator.free(vals);
        var inited: usize = 0;
        errdefer for (vals[0..inited]) |v| v.free(self.allocator);

        for (row_vals, 0..) |v, i| {
            vals[i] = v.dupe(self.allocator) catch return ExecError.OutOfMemory;
            inited += 1;
        }

        const cols = self.allocator.alloc([]const u8, self.col_names.len) catch return ExecError.OutOfMemory;
        for (self.col_names, 0..) |c, i| cols[i] = c;

        return Row{
            .columns = cols,
            .values = vals,
            .allocator = self.allocator,
        };
    }

    fn closeFn(_: *anyopaque) void {}

    fn iterator(self: *InMemorySource) RowIterator {
        return .{
            .ptr = self,
            .vtable = &.{
                .next = &InMemorySource.nextFn,
                .close = &InMemorySource.closeFn,
            },
        };
    }
};

// ── MaterializedOp Tests ────────────────────────────────────────────────

test "MaterializedOp with multiple rows" {
    const allocator = std.testing.allocator;

    // Build col_names (owned by MaterializedOp — freed in close)
    const col_names = try allocator.alloc([]const u8, 2);
    col_names[0] = try allocator.dupe(u8, "id");
    col_names[1] = try allocator.dupe(u8, "name");

    // Build rows (owned by MaterializedOp — freed in close)
    const rows = try allocator.alloc([]Value, 2);
    rows[0] = try allocator.alloc(Value, 2);
    rows[0][0] = Value{ .integer = 1 };
    rows[0][1] = Value{ .text = try allocator.dupe(u8, "alice") };
    rows[1] = try allocator.alloc(Value, 2);
    rows[1][0] = Value{ .integer = 2 };
    rows[1][1] = Value{ .text = try allocator.dupe(u8, "bob") };

    var op = MaterializedOp.init(allocator, col_names, rows);

    // Row 1
    var r1 = (try op.next()).?;
    defer r1.deinit();
    try std.testing.expectEqual(@as(i64, 1), r1.values[0].integer);
    try std.testing.expectEqualStrings("alice", r1.values[1].text);
    try std.testing.expectEqualStrings("id", r1.columns[0]);
    try std.testing.expectEqualStrings("name", r1.columns[1]);

    // Row 2
    var r2 = (try op.next()).?;
    defer r2.deinit();
    try std.testing.expectEqual(@as(i64, 2), r2.values[0].integer);
    try std.testing.expectEqualStrings("bob", r2.values[1].text);

    // No more rows
    try std.testing.expect(try op.next() == null);

    op.close();
}

test "MaterializedOp with empty rows" {
    const allocator = std.testing.allocator;

    const col_names = try allocator.alloc([]const u8, 1);
    col_names[0] = try allocator.dupe(u8, "x");

    const rows = try allocator.alloc([]Value, 0);

    var op = MaterializedOp.init(allocator, col_names, rows);

    try std.testing.expect(try op.next() == null);

    op.close();
}

test "MaterializedOp single row with null value" {
    const allocator = std.testing.allocator;

    const col_names = try allocator.alloc([]const u8, 2);
    col_names[0] = try allocator.dupe(u8, "a");
    col_names[1] = try allocator.dupe(u8, "b");

    const rows = try allocator.alloc([]Value, 1);
    rows[0] = try allocator.alloc(Value, 2);
    rows[0][0] = Value{ .integer = 42 };
    rows[0][1] = .null_value;

    var op = MaterializedOp.init(allocator, col_names, rows);

    var r = (try op.next()).?;
    defer r.deinit();
    try std.testing.expectEqual(@as(i64, 42), r.values[0].integer);
    try std.testing.expect(r.values[1] == .null_value);

    try std.testing.expect(try op.next() == null);

    op.close();
}

test "MaterializedOp iterator interface" {
    const allocator = std.testing.allocator;

    const col_names = try allocator.alloc([]const u8, 1);
    col_names[0] = try allocator.dupe(u8, "val");

    const rows = try allocator.alloc([]Value, 2);
    rows[0] = try allocator.alloc(Value, 1);
    rows[0][0] = Value{ .integer = 10 };
    rows[1] = try allocator.alloc(Value, 1);
    rows[1][0] = Value{ .integer = 20 };

    var op = MaterializedOp.init(allocator, col_names, rows);
    var iter = op.iterator();

    var r1 = (try iter.next()).?;
    defer r1.deinit();
    try std.testing.expectEqual(@as(i64, 10), r1.values[0].integer);

    var r2 = (try iter.next()).?;
    defer r2.deinit();
    try std.testing.expectEqual(@as(i64, 20), r2.values[0].integer);

    try std.testing.expect(try iter.next() == null);

    iter.close();
}

test "MaterializedOp with text values duplicates correctly" {
    const allocator = std.testing.allocator;

    const col_names = try allocator.alloc([]const u8, 1);
    col_names[0] = try allocator.dupe(u8, "name");

    const rows = try allocator.alloc([]Value, 1);
    rows[0] = try allocator.alloc(Value, 1);
    rows[0][0] = Value{ .text = try allocator.dupe(u8, "hello") };

    var op = MaterializedOp.init(allocator, col_names, rows);

    var r = (try op.next()).?;
    // The returned row has a duplicated text value (independent memory)
    try std.testing.expectEqualStrings("hello", r.values[0].text);
    r.deinit();

    // After consuming all rows, close frees the source data
    try std.testing.expect(try op.next() == null);
    op.close();
}

// ── SetOpOp Unit Tests ───────────────────────────────────────────────

test "SetOpOp UNION ALL concatenates both sides" {
    const allocator = std.testing.allocator;

    var left = InMemorySource.init(allocator, &.{"id"});
    try left.addRow(&.{Value{ .integer = 1 }});
    try left.addRow(&.{Value{ .integer = 2 }});
    defer left.deinit();

    var right = InMemorySource.init(allocator, &.{"id"});
    try right.addRow(&.{Value{ .integer = 2 }});
    try right.addRow(&.{Value{ .integer = 3 }});
    defer right.deinit();

    var op = SetOpOp.init(allocator, left.iterator(), right.iterator(), .union_all);
    defer op.close();

    var count: usize = 0;
    while (try op.next()) |*rp| {
        var row = rp.*;
        defer row.deinit();
        count += 1;
    }
    try std.testing.expectEqual(@as(usize, 4), count);
}

test "SetOpOp UNION deduplicates rows" {
    const allocator = std.testing.allocator;

    var left = InMemorySource.init(allocator, &.{"id"});
    try left.addRow(&.{Value{ .integer = 1 }});
    try left.addRow(&.{Value{ .integer = 2 }});
    defer left.deinit();

    var right = InMemorySource.init(allocator, &.{"id"});
    try right.addRow(&.{Value{ .integer = 2 }});
    try right.addRow(&.{Value{ .integer = 3 }});
    defer right.deinit();

    var op = SetOpOp.init(allocator, left.iterator(), right.iterator(), .@"union");
    defer op.close();

    var count: usize = 0;
    while (try op.next()) |*rp| {
        var row = rp.*;
        defer row.deinit();
        count += 1;
    }
    // 1, 2, 3 — duplicate 2 removed
    try std.testing.expectEqual(@as(usize, 3), count);
}

test "SetOpOp INTERSECT returns common rows only" {
    const allocator = std.testing.allocator;

    var left = InMemorySource.init(allocator, &.{ "id", "name" });
    try left.addRow(&.{ Value{ .integer = 1 }, Value{ .text = "a" } });
    try left.addRow(&.{ Value{ .integer = 2 }, Value{ .text = "b" } });
    try left.addRow(&.{ Value{ .integer = 3 }, Value{ .text = "c" } });
    defer left.deinit();

    var right = InMemorySource.init(allocator, &.{ "id", "name" });
    try right.addRow(&.{ Value{ .integer = 2 }, Value{ .text = "b" } });
    try right.addRow(&.{ Value{ .integer = 3 }, Value{ .text = "c" } });
    try right.addRow(&.{ Value{ .integer = 4 }, Value{ .text = "d" } });
    defer right.deinit();

    var op = SetOpOp.init(allocator, left.iterator(), right.iterator(), .intersect);
    defer op.close();

    var count: usize = 0;
    while (try op.next()) |*rp| {
        var row = rp.*;
        defer row.deinit();
        count += 1;
    }
    // Common: (2,'b'), (3,'c')
    try std.testing.expectEqual(@as(usize, 2), count);
}

test "SetOpOp EXCEPT returns left-only rows" {
    const allocator = std.testing.allocator;

    var left = InMemorySource.init(allocator, &.{"id"});
    try left.addRow(&.{Value{ .integer = 1 }});
    try left.addRow(&.{Value{ .integer = 2 }});
    try left.addRow(&.{Value{ .integer = 3 }});
    defer left.deinit();

    var right = InMemorySource.init(allocator, &.{"id"});
    try right.addRow(&.{Value{ .integer = 2 }});
    try right.addRow(&.{Value{ .integer = 3 }});
    defer right.deinit();

    var op = SetOpOp.init(allocator, left.iterator(), right.iterator(), .except);
    defer op.close();

    var row = (try op.next()).?;
    defer row.deinit();
    try std.testing.expectEqual(@as(i64, 1), row.values[0].integer);

    // No more rows
    try std.testing.expect(try op.next() == null);
}

test "SetOpOp UNION with empty left side" {
    const allocator = std.testing.allocator;

    var left = InMemorySource.init(allocator, &.{"id"});
    defer left.deinit();

    var right = InMemorySource.init(allocator, &.{"id"});
    try right.addRow(&.{Value{ .integer = 1 }});
    defer right.deinit();

    var op = SetOpOp.init(allocator, left.iterator(), right.iterator(), .@"union");
    defer op.close();

    var row = (try op.next()).?;
    defer row.deinit();
    try std.testing.expectEqual(@as(i64, 1), row.values[0].integer);
    try std.testing.expect(try op.next() == null);
}

test "SetOpOp INTERSECT with empty right returns empty" {
    const allocator = std.testing.allocator;

    var left = InMemorySource.init(allocator, &.{"id"});
    try left.addRow(&.{Value{ .integer = 1 }});
    defer left.deinit();

    var right = InMemorySource.init(allocator, &.{"id"});
    defer right.deinit();

    var op = SetOpOp.init(allocator, left.iterator(), right.iterator(), .intersect);
    defer op.close();

    try std.testing.expect(try op.next() == null);
}

test "SetOpOp EXCEPT with empty right returns all left rows" {
    const allocator = std.testing.allocator;

    var left = InMemorySource.init(allocator, &.{"id"});
    try left.addRow(&.{Value{ .integer = 1 }});
    try left.addRow(&.{Value{ .integer = 2 }});
    defer left.deinit();

    var right = InMemorySource.init(allocator, &.{"id"});
    defer right.deinit();

    var op = SetOpOp.init(allocator, left.iterator(), right.iterator(), .except);
    defer op.close();

    var count: usize = 0;
    while (try op.next()) |*rp| {
        var row = rp.*;
        defer row.deinit();
        count += 1;
    }
    try std.testing.expectEqual(@as(usize, 2), count);
}

test "SetOpOp UNION deduplicates within same side" {
    const allocator = std.testing.allocator;

    var left = InMemorySource.init(allocator, &.{"id"});
    try left.addRow(&.{Value{ .integer = 1 }});
    try left.addRow(&.{Value{ .integer = 1 }});
    try left.addRow(&.{Value{ .integer = 2 }});
    defer left.deinit();

    var right = InMemorySource.init(allocator, &.{"id"});
    try right.addRow(&.{Value{ .integer = 2 }});
    try right.addRow(&.{Value{ .integer = 2 }});
    defer right.deinit();

    var op = SetOpOp.init(allocator, left.iterator(), right.iterator(), .@"union");
    defer op.close();

    var count: usize = 0;
    while (try op.next()) |*rp| {
        var row = rp.*;
        defer row.deinit();
        count += 1;
    }
    // Deduplicated: [1, 2]
    try std.testing.expectEqual(@as(usize, 2), count);
}

test "DistinctOp eliminates duplicate rows" {
    const allocator = std.testing.allocator;

    var data = InMemorySource.init(allocator, &.{ "id", "name" });
    try data.addRow(&.{ Value{ .integer = 1 }, Value{ .text = "Alice" } });
    try data.addRow(&.{ Value{ .integer = 2 }, Value{ .text = "Bob" } });
    try data.addRow(&.{ Value{ .integer = 1 }, Value{ .text = "Alice" } }); // duplicate
    try data.addRow(&.{ Value{ .integer = 3 }, Value{ .text = "Carol" } });
    try data.addRow(&.{ Value{ .integer = 2 }, Value{ .text = "Bob" } }); // duplicate
    defer data.deinit();

    var op = DistinctOp.init(allocator, data.iterator(), &.{});
    defer op.close();

    var count: usize = 0;
    while (try op.next()) |*rp| {
        var row = rp.*;
        defer row.deinit();
        count += 1;
    }
    try std.testing.expectEqual(@as(usize, 3), count);
}

test "DistinctOp all unique rows pass through" {
    const allocator = std.testing.allocator;

    var data = InMemorySource.init(allocator, &.{"val"});
    try data.addRow(&.{Value{ .integer = 1 }});
    try data.addRow(&.{Value{ .integer = 2 }});
    try data.addRow(&.{Value{ .integer = 3 }});
    defer data.deinit();

    var op = DistinctOp.init(allocator, data.iterator(), &.{});
    defer op.close();

    var count: usize = 0;
    while (try op.next()) |*rp| {
        var row = rp.*;
        defer row.deinit();
        count += 1;
    }
    try std.testing.expectEqual(@as(usize, 3), count);
}

test "DistinctOp empty input" {
    const allocator = std.testing.allocator;

    var data = InMemorySource.init(allocator, &.{"val"});
    defer data.deinit();

    var op = DistinctOp.init(allocator, data.iterator(), &.{});
    defer op.close();

    try std.testing.expectEqual(@as(?Row, null), try op.next());
}

test "DistinctOp handles NULL values" {
    const allocator = std.testing.allocator;

    var data = InMemorySource.init(allocator, &.{"val"});
    try data.addRow(&.{Value.null_value});
    try data.addRow(&.{Value{ .integer = 1 }});
    try data.addRow(&.{Value.null_value}); // duplicate null
    defer data.deinit();

    var op = DistinctOp.init(allocator, data.iterator(), &.{});
    defer op.close();

    var count: usize = 0;
    while (try op.next()) |*rp| {
        var row = rp.*;
        defer row.deinit();
        count += 1;
    }
    try std.testing.expectEqual(@as(usize, 2), count);
}

test "DistinctOp with mixed types" {
    const allocator = std.testing.allocator;

    var data = InMemorySource.init(allocator, &.{ "a", "b" });
    try data.addRow(&.{ Value{ .integer = 1 }, Value{ .text = "x" } });
    try data.addRow(&.{ Value{ .integer = 1 }, Value{ .text = "y" } });
    try data.addRow(&.{ Value{ .integer = 1 }, Value{ .text = "x" } }); // dup
    try data.addRow(&.{ Value{ .integer = 2 }, Value{ .text = "x" } });
    defer data.deinit();

    var op = DistinctOp.init(allocator, data.iterator(), &.{});
    defer op.close();

    var count: usize = 0;
    while (try op.next()) |*rp| {
        var row = rp.*;
        defer row.deinit();
        count += 1;
    }
    try std.testing.expectEqual(@as(usize, 3), count);
}

// ── DISTINCT ON Unit Tests ──────────────────────────────────────────────

test "DistinctOp with DISTINCT ON single column keeps first row per group" {
    const allocator = std.testing.allocator;

    // Rows: (dept, name, salary) — sorted by dept
    var data = InMemorySource.init(allocator, &.{ "dept", "name", "salary" });
    try data.addRow(&.{ Value{ .text = "eng" }, Value{ .text = "Alice" }, Value{ .integer = 100 } });
    try data.addRow(&.{ Value{ .text = "eng" }, Value{ .text = "Bob" }, Value{ .integer = 90 } });
    try data.addRow(&.{ Value{ .text = "sales" }, Value{ .text = "Carol" }, Value{ .integer = 80 } });
    try data.addRow(&.{ Value{ .text = "sales" }, Value{ .text = "Dave" }, Value{ .integer = 70 } });
    defer data.deinit();

    // DISTINCT ON (dept) — column_ref for "dept"
    const on_expr = ast.Expr{ .column_ref = .{ .name = "dept" } };

    var op = DistinctOp.init(allocator, data.iterator(), &.{&on_expr});
    defer op.close();

    // Should get first row per dept: Alice (eng) and Carol (sales)
    var r1 = (try op.next()).?;
    defer r1.deinit();
    try std.testing.expectEqualStrings("Alice", r1.values[1].text);

    var r2 = (try op.next()).?;
    defer r2.deinit();
    try std.testing.expectEqualStrings("Carol", r2.values[1].text);

    // No more rows
    try std.testing.expect(try op.next() == null);
}

test "DistinctOp with DISTINCT ON multiple columns" {
    const allocator = std.testing.allocator;

    // Rows: (a, b, c) — sorted by a, b
    var data = InMemorySource.init(allocator, &.{ "a", "b", "c" });
    try data.addRow(&.{ Value{ .integer = 1 }, Value{ .integer = 10 }, Value{ .text = "x" } });
    try data.addRow(&.{ Value{ .integer = 1 }, Value{ .integer = 10 }, Value{ .text = "y" } }); // same (a,b)
    try data.addRow(&.{ Value{ .integer = 1 }, Value{ .integer = 20 }, Value{ .text = "z" } });
    try data.addRow(&.{ Value{ .integer = 2 }, Value{ .integer = 10 }, Value{ .text = "w" } });
    defer data.deinit();

    // DISTINCT ON (a, b)
    const expr_a = ast.Expr{ .column_ref = .{ .name = "a" } };
    const expr_b = ast.Expr{ .column_ref = .{ .name = "b" } };

    var op = DistinctOp.init(allocator, data.iterator(), &.{ &expr_a, &expr_b });
    defer op.close();

    // Should get 3 rows: (1,10,x), (1,20,z), (2,10,w) — second row (1,10,y) is dup
    var r1 = (try op.next()).?;
    defer r1.deinit();
    try std.testing.expectEqualStrings("x", r1.values[2].text);

    var r2 = (try op.next()).?;
    defer r2.deinit();
    try std.testing.expectEqualStrings("z", r2.values[2].text);

    var r3 = (try op.next()).?;
    defer r3.deinit();
    try std.testing.expectEqualStrings("w", r3.values[2].text);

    try std.testing.expect(try op.next() == null);
}

test "DistinctOp with DISTINCT ON and NULL in dedup key" {
    const allocator = std.testing.allocator;

    var data = InMemorySource.init(allocator, &.{ "key", "val" });
    try data.addRow(&.{ Value.null_value, Value{ .integer = 1 } });
    try data.addRow(&.{ Value.null_value, Value{ .integer = 2 } }); // same NULL key
    try data.addRow(&.{ Value{ .integer = 10 }, Value{ .integer = 3 } });
    defer data.deinit();

    const on_expr = ast.Expr{ .column_ref = .{ .name = "key" } };

    var op = DistinctOp.init(allocator, data.iterator(), &.{&on_expr});
    defer op.close();

    // NULL key group: first row only
    var r1 = (try op.next()).?;
    defer r1.deinit();
    try std.testing.expectEqual(Value.null_value, r1.values[0]);
    try std.testing.expectEqual(@as(i64, 1), r1.values[1].integer);

    // Non-null key group
    var r2 = (try op.next()).?;
    defer r2.deinit();
    try std.testing.expectEqual(@as(i64, 10), r2.values[0].integer);

    try std.testing.expect(try op.next() == null);
}

test "DistinctOp with DISTINCT ON empty input" {
    const allocator = std.testing.allocator;

    var data = InMemorySource.init(allocator, &.{ "key", "val" });
    defer data.deinit();

    const on_expr = ast.Expr{ .column_ref = .{ .name = "key" } };

    var op = DistinctOp.init(allocator, data.iterator(), &.{&on_expr});
    defer op.close();

    try std.testing.expect(try op.next() == null);
}

// ── SetOpOp NULL Handling Tests ─────────────────────────────────────────

test "SetOpOp UNION ALL with NULL values in multi-column rows" {
    const allocator = std.testing.allocator;

    var left = InMemorySource.init(allocator, &.{ "id", "name" });
    try left.addRow(&.{ Value{ .integer = 1 }, Value.null_value });
    try left.addRow(&.{ Value.null_value, Value{ .text = "a" } });
    defer left.deinit();

    var right = InMemorySource.init(allocator, &.{ "id", "name" });
    try right.addRow(&.{ Value.null_value, Value{ .text = "a" } });
    try right.addRow(&.{ Value{ .integer = 2 }, Value.null_value });
    defer right.deinit();

    var op = SetOpOp.init(allocator, left.iterator(), right.iterator(), .union_all);
    defer op.close();

    var count: usize = 0;
    while (try op.next()) |*rp| {
        var row = rp.*;
        defer row.deinit();
        count += 1;
    }
    // UNION ALL returns all 4 rows (no dedup)
    try std.testing.expectEqual(@as(usize, 4), count);
}

test "SetOpOp UNION deduplicates NULL rows" {
    const allocator = std.testing.allocator;

    var left = InMemorySource.init(allocator, &.{"val"});
    try left.addRow(&.{Value.null_value});
    try left.addRow(&.{Value{ .integer = 1 }});
    defer left.deinit();

    var right = InMemorySource.init(allocator, &.{"val"});
    try right.addRow(&.{Value.null_value}); // duplicate NULL
    try right.addRow(&.{Value{ .integer = 2 }});
    defer right.deinit();

    var op = SetOpOp.init(allocator, left.iterator(), right.iterator(), .@"union");
    defer op.close();

    var count: usize = 0;
    while (try op.next()) |*rp| {
        var row = rp.*;
        defer row.deinit();
        count += 1;
    }
    // NULL, 1, 2 — duplicate NULL removed
    try std.testing.expectEqual(@as(usize, 3), count);
}

test "SetOpOp INTERSECT with NULL rows" {
    const allocator = std.testing.allocator;

    var left = InMemorySource.init(allocator, &.{ "id", "name" });
    try left.addRow(&.{ Value.null_value, Value{ .text = "a" } });
    try left.addRow(&.{ Value{ .integer = 1 }, Value.null_value });
    try left.addRow(&.{ Value{ .integer = 2 }, Value{ .text = "b" } });
    defer left.deinit();

    var right = InMemorySource.init(allocator, &.{ "id", "name" });
    try right.addRow(&.{ Value.null_value, Value{ .text = "a" } }); // matches left row 1
    try right.addRow(&.{ Value{ .integer = 1 }, Value{ .text = "c" } }); // no match
    try right.addRow(&.{ Value{ .integer = 3 }, Value{ .text = "d" } });
    defer right.deinit();

    var op = SetOpOp.init(allocator, left.iterator(), right.iterator(), .intersect);
    defer op.close();

    var count: usize = 0;
    while (try op.next()) |*rp| {
        var row = rp.*;
        defer row.deinit();
        count += 1;
        // The only match is (NULL, "a")
        try std.testing.expectEqual(Value.null_value, row.values[0]);
        try std.testing.expectEqualStrings("a", row.values[1].text);
    }
    try std.testing.expectEqual(@as(usize, 1), count);
}

test "SetOpOp EXCEPT with NULL rows" {
    const allocator = std.testing.allocator;

    var left = InMemorySource.init(allocator, &.{"val"});
    try left.addRow(&.{Value.null_value});
    try left.addRow(&.{Value{ .integer = 1 }});
    try left.addRow(&.{Value{ .integer = 2 }});
    defer left.deinit();

    var right = InMemorySource.init(allocator, &.{"val"});
    try right.addRow(&.{Value.null_value}); // remove NULL from result
    try right.addRow(&.{Value{ .integer = 1 }}); // remove 1 from result
    defer right.deinit();

    var op = SetOpOp.init(allocator, left.iterator(), right.iterator(), .except);
    defer op.close();

    // Only 2 should remain
    var row = (try op.next()).?;
    defer row.deinit();
    try std.testing.expectEqual(@as(i64, 2), row.values[0].integer);

    try std.testing.expect(try op.next() == null);
}

test "SetOpOp UNION with multi-column NULL combinations" {
    const allocator = std.testing.allocator;

    var left = InMemorySource.init(allocator, &.{ "a", "b" });
    try left.addRow(&.{ Value.null_value, Value.null_value });
    try left.addRow(&.{ Value{ .integer = 1 }, Value.null_value });
    defer left.deinit();

    var right = InMemorySource.init(allocator, &.{ "a", "b" });
    try right.addRow(&.{ Value.null_value, Value.null_value }); // dup of left row 1
    try right.addRow(&.{ Value.null_value, Value{ .integer = 2 } }); // different — b differs
    defer right.deinit();

    var op = SetOpOp.init(allocator, left.iterator(), right.iterator(), .@"union");
    defer op.close();

    var count: usize = 0;
    while (try op.next()) |*rp| {
        var row = rp.*;
        defer row.deinit();
        count += 1;
    }
    // (NULL,NULL), (1,NULL), (NULL,2) — 3 unique rows
    try std.testing.expectEqual(@as(usize, 3), count);
}

// ── Row Utility Tests ───────────────────────────────────────────────────

test "Row.clone creates independent deep copy" {
    const allocator = std.testing.allocator;

    // Create original row with text values (heap-allocated)
    const cols = try allocator.alloc([]const u8, 3);
    cols[0] = "id";
    cols[1] = "name";
    cols[2] = "tag";

    const vals = try allocator.alloc(Value, 3);
    vals[0] = Value{ .integer = 42 };
    vals[1] = Value{ .text = try allocator.dupe(u8, "hello") };
    vals[2] = Value.null_value;

    var original = Row{
        .columns = cols,
        .values = vals,
        .allocator = allocator,
    };

    // Clone
    var cloned = try original.clone(allocator);

    // Verify values match
    try std.testing.expectEqual(@as(i64, 42), cloned.values[0].integer);
    try std.testing.expectEqualStrings("hello", cloned.values[1].text);
    try std.testing.expectEqual(Value.null_value, cloned.values[2]);

    // Verify independence: free original, cloned should still be valid
    original.deinit();

    try std.testing.expectEqualStrings("hello", cloned.values[1].text);
    try std.testing.expectEqual(@as(i64, 42), cloned.values[0].integer);

    cloned.deinit();
}

test "Row.clone with empty row" {
    const allocator = std.testing.allocator;

    const cols = try allocator.alloc([]const u8, 0);
    const vals = try allocator.alloc(Value, 0);

    var original = Row{
        .columns = cols,
        .values = vals,
        .allocator = allocator,
    };

    var cloned = try original.clone(allocator);

    try std.testing.expectEqual(@as(usize, 0), cloned.columns.len);
    try std.testing.expectEqual(@as(usize, 0), cloned.values.len);

    original.deinit();
    cloned.deinit();
}

test "Row.getQualifiedColumn with table prefix" {
    const allocator = std.testing.allocator;

    const cols = try allocator.alloc([]const u8, 3);
    cols[0] = "users.id";
    cols[1] = "users.name";
    cols[2] = "orders.id";

    const vals = try allocator.alloc(Value, 3);
    vals[0] = Value{ .integer = 1 };
    vals[1] = Value{ .text = try allocator.dupe(u8, "Alice") };
    vals[2] = Value{ .integer = 100 };

    var row = Row{
        .columns = cols,
        .values = vals,
        .allocator = allocator,
    };
    defer row.deinit();

    // Qualified lookup: users.id → 1
    const uid = row.getQualifiedColumn("users", "id");
    try std.testing.expect(uid != null);
    try std.testing.expectEqual(@as(i64, 1), uid.?.integer);

    // Qualified lookup: orders.id → 100
    const oid = row.getQualifiedColumn("orders", "id");
    try std.testing.expect(oid != null);
    try std.testing.expectEqual(@as(i64, 100), oid.?.integer);

    // Qualified lookup: users.name → "Alice"
    const name = row.getQualifiedColumn("users", "name");
    try std.testing.expect(name != null);
    try std.testing.expectEqualStrings("Alice", name.?.text);

    // Non-existent table — falls back to unqualified lookup for "id"
    // but no plain "id" column exists (all are qualified), so returns null
    const none = row.getQualifiedColumn("products", "id");
    try std.testing.expect(none == null);
}

test "Row.getQualifiedColumn falls back to unqualified" {
    const allocator = std.testing.allocator;

    const cols = try allocator.alloc([]const u8, 2);
    cols[0] = "id";
    cols[1] = "name";

    const vals = try allocator.alloc(Value, 2);
    vals[0] = Value{ .integer = 5 };
    vals[1] = Value{ .text = try allocator.dupe(u8, "Bob") };

    var row = Row{
        .columns = cols,
        .values = vals,
        .allocator = allocator,
    };
    defer row.deinit();

    // No "t.id" found, falls back to plain "id"
    const val = row.getQualifiedColumn("t", "id");
    try std.testing.expect(val != null);
    try std.testing.expectEqual(@as(i64, 5), val.?.integer);
}

test "Date parsing and formatting" {
    const allocator = std.testing.allocator;

    // Parse valid date
    const days1 = parseDateString("2024-03-15");
    try std.testing.expect(days1 != null);

    // Format it back
    const s1 = try formatDate(allocator, days1.?);
    defer allocator.free(s1);
    try std.testing.expectEqualStrings("2024-03-15", s1);

    // Test epoch
    const epoch = parseDateString("1970-01-01");
    try std.testing.expect(epoch != null);
    try std.testing.expectEqual(@as(i32, 0), epoch.?);

    // Test invalid dates
    try std.testing.expect(parseDateString("2024-13-01") == null); // invalid month
    try std.testing.expect(parseDateString("2024-02-30") == null); // invalid day
    try std.testing.expect(parseDateString("not-a-date") == null);
}

test "Time parsing and formatting" {
    const allocator = std.testing.allocator;

    // Parse valid time
    const micros1 = parseTimeString("14:30:45");
    try std.testing.expect(micros1 != null);

    // Format it back
    const s1 = try formatTime(allocator, micros1.?);
    defer allocator.free(s1);
    try std.testing.expectEqualStrings("14:30:45", s1);

    // Test midnight
    const midnight = parseTimeString("00:00:00");
    try std.testing.expect(midnight != null);
    try std.testing.expectEqual(@as(i64, 0), midnight.?);

    // Test with fractional seconds
    const with_frac = parseTimeString("12:34:56.123456");
    try std.testing.expect(with_frac != null);
    const expected = 12 * MICROS_PER_HOUR + 34 * MICROS_PER_MINUTE + 56 * MICROS_PER_SECOND + 123456;
    try std.testing.expectEqual(expected, with_frac.?);
}

test "Timestamp parsing and formatting" {
    const allocator = std.testing.allocator;

    // Parse valid timestamp
    const micros1 = parseTimestampString("2024-03-15 14:30:45");
    try std.testing.expect(micros1 != null);

    // Format it back
    const s1 = try formatTimestamp(allocator, micros1.?);
    defer allocator.free(s1);
    try std.testing.expectEqualStrings("2024-03-15 14:30:45", s1);
}

test "Date arithmetic" {
    // date + integer
    const base = Value{ .date = 100 };
    const add_result = evalArithmetic(base, Value{ .integer = 5 }, .add);
    try std.testing.expectEqual(Value{ .date = 105 }, add_result);

    // date - integer
    const sub_result = evalArithmetic(base, Value{ .integer = 10 }, .sub);
    try std.testing.expectEqual(Value{ .date = 90 }, sub_result);

    // date - date
    const date1 = Value{ .date = 100 };
    const date2 = Value{ .date = 95 };
    const diff = evalArithmetic(date1, date2, .sub);
    try std.testing.expectEqual(Value{ .integer = 5 }, diff);
}

test "CAST with temporal types" {
    const allocator = std.testing.allocator;

    // CAST text to date
    const text_date = Value{ .text = "2024-03-15" };
    const date_val = try evalCast(allocator, text_date, .type_date);
    try std.testing.expect(date_val == .date);

    // CAST date to text
    const date_val2 = Value{ .date = parseDateString("2024-03-15").? };
    const text_val = try evalCast(allocator, date_val2, .type_text);
    defer text_val.free(allocator);
    try std.testing.expectEqualStrings("2024-03-15", text_val.text);

    // CAST timestamp to date
    const ts = parseTimestampString("2024-03-15 14:30:45").?;
    const ts_val = Value{ .timestamp = ts };
    const date_from_ts = try evalCast(allocator, ts_val, .type_date);
    try std.testing.expect(date_from_ts == .date);

    // CAST date to timestamp
    const date_to_ts = try evalCast(allocator, date_val2, .type_timestamp);
    try std.testing.expect(date_to_ts == .timestamp);
}

test "Serialize and deserialize temporal types" {
    const allocator = std.testing.allocator;

    const values = [_]Value{
        Value{ .date = 19800 },
        Value{ .time = 14 * MICROS_PER_HOUR + 30 * MICROS_PER_MINUTE },
        Value{ .timestamp = 1710508245000000 },
    };

    const serialized = try serializeRow(allocator, &values);
    defer allocator.free(serialized);

    const deserialized = try deserializeRow(allocator, serialized);
    defer {
        for (deserialized) |v| v.free(allocator);
        allocator.free(deserialized);
    }

    try std.testing.expectEqual(@as(usize, 3), deserialized.len);
    try std.testing.expectEqual(Value{ .date = 19800 }, deserialized[0]);
    try std.testing.expectEqual(Value{ .time = 14 * MICROS_PER_HOUR + 30 * MICROS_PER_MINUTE }, deserialized[1]);
    try std.testing.expectEqual(Value{ .timestamp = 1710508245000000 }, deserialized[2]);
}

test "Value comparison with temporal types" {
    // date comparison
    const d1 = Value{ .date = 100 };
    const d2 = Value{ .date = 200 };
    try std.testing.expectEqual(std.math.Order.lt, d1.compare(d2));
    try std.testing.expectEqual(std.math.Order.eq, d1.compare(d1));

    // time comparison
    const t1 = Value{ .time = 1000 };
    const t2 = Value{ .time = 2000 };
    try std.testing.expectEqual(std.math.Order.lt, t1.compare(t2));

    // timestamp comparison
    const ts1 = Value{ .timestamp = 1000000 };
    const ts2 = Value{ .timestamp = 2000000 };
    try std.testing.expectEqual(std.math.Order.lt, ts1.compare(ts2));

    // date vs timestamp comparison
    const date = Value{ .date = 1 }; // 1 day = 86400000000 microseconds
    const ts = Value{ .timestamp = MICROS_PER_DAY }; // same as 1 day
    try std.testing.expectEqual(std.math.Order.eq, date.compare(ts));
}

// ── NUMERIC Unit Tests ──────────────────────────────────────────────────

test "parseNumericString basic" {
    // Integer-like
    const n1 = parseNumericString("123").?;
    try std.testing.expectEqual(@as(i128, 123), n1.value);
    try std.testing.expectEqual(@as(u8, 0), n1.scale);

    // Decimal
    const n2 = parseNumericString("123.45").?;
    try std.testing.expectEqual(@as(i128, 12345), n2.value);
    try std.testing.expectEqual(@as(u8, 2), n2.scale);

    // Negative
    const n3 = parseNumericString("-99.9").?;
    try std.testing.expectEqual(@as(i128, -999), n3.value);
    try std.testing.expectEqual(@as(u8, 1), n3.scale);

    // Zero
    const n4 = parseNumericString("0.00").?;
    try std.testing.expectEqual(@as(i128, 0), n4.value);
    try std.testing.expectEqual(@as(u8, 2), n4.scale);

    // Leading zeros in fractional part
    const n5 = parseNumericString("1.05").?;
    try std.testing.expectEqual(@as(i128, 105), n5.value);
    try std.testing.expectEqual(@as(u8, 2), n5.scale);

    // Invalid
    try std.testing.expect(parseNumericString("") == null);
    try std.testing.expect(parseNumericString("abc") == null);
    try std.testing.expect(parseNumericString("1.2.3") == null);
    try std.testing.expect(parseNumericString("-") == null);
}

test "formatNumeric basic" {
    const allocator = std.testing.allocator;

    // Integer (scale 0)
    const s1 = try formatNumeric(allocator, .{ .value = 42, .scale = 0 });
    defer allocator.free(s1);
    try std.testing.expectEqualStrings("42", s1);

    // Decimal
    const s2 = try formatNumeric(allocator, .{ .value = 12345, .scale = 2 });
    defer allocator.free(s2);
    try std.testing.expectEqualStrings("123.45", s2);

    // Negative decimal
    const s3 = try formatNumeric(allocator, .{ .value = -999, .scale = 1 });
    defer allocator.free(s3);
    try std.testing.expectEqualStrings("-99.9", s3);

    // Leading zeros in fraction
    const s4 = try formatNumeric(allocator, .{ .value = 105, .scale = 2 });
    defer allocator.free(s4);
    try std.testing.expectEqualStrings("1.05", s4);

    // Zero with scale
    const s5 = try formatNumeric(allocator, .{ .value = 0, .scale = 3 });
    defer allocator.free(s5);
    try std.testing.expectEqualStrings("0.000", s5);

    // Small value with leading zeros
    const s6 = try formatNumeric(allocator, .{ .value = 1, .scale = 3 });
    defer allocator.free(s6);
    try std.testing.expectEqualStrings("0.001", s6);
}

test "Numeric comparison" {
    const a = Value{ .numeric = .{ .value = 100, .scale = 2 } }; // 1.00
    const b = Value{ .numeric = .{ .value = 200, .scale = 2 } }; // 2.00
    try std.testing.expectEqual(std.math.Order.lt, a.compare(b));
    try std.testing.expectEqual(std.math.Order.gt, b.compare(a));
    try std.testing.expectEqual(std.math.Order.eq, a.compare(a));

    // Cross-scale comparison: 1.0 vs 1.00
    const c = Value{ .numeric = .{ .value = 10, .scale = 1 } }; // 1.0
    const d = Value{ .numeric = .{ .value = 100, .scale = 2 } }; // 1.00
    try std.testing.expectEqual(std.math.Order.eq, c.compare(d));

    // Numeric vs integer: 1.50 vs 2
    const e = Value{ .numeric = .{ .value = 150, .scale = 2 } }; // 1.50
    const f = Value{ .integer = 2 };
    try std.testing.expectEqual(std.math.Order.lt, e.compare(f));
}

test "Numeric arithmetic" {
    // add
    const a = Value{ .numeric = .{ .value = 100, .scale = 2 } }; // 1.00
    const b = Value{ .numeric = .{ .value = 250, .scale = 2 } }; // 2.50
    const sum = evalArithmetic(a, b, .add);
    try std.testing.expectEqual(@as(i128, 350), sum.numeric.value);
    try std.testing.expectEqual(@as(u8, 2), sum.numeric.scale);

    // subtract
    const diff = evalArithmetic(b, a, .sub);
    try std.testing.expectEqual(@as(i128, 150), diff.numeric.value);

    // multiply: 1.50 * 2.00 = 3.00
    const c = Value{ .numeric = .{ .value = 150, .scale = 2 } };
    const d = Value{ .numeric = .{ .value = 200, .scale = 2 } };
    const prod = evalArithmetic(c, d, .mul);
    try std.testing.expectEqual(@as(i128, 300), prod.numeric.value);
    try std.testing.expectEqual(@as(u8, 2), prod.numeric.scale);

    // divide: 3.00 / 1.50 = 2.00
    const three = Value{ .numeric = .{ .value = 300, .scale = 2 } };
    const onePointFive = Value{ .numeric = .{ .value = 150, .scale = 2 } };
    const quot = evalArithmetic(three, onePointFive, .div);
    try std.testing.expectEqual(@as(i128, 200), quot.numeric.value);
    try std.testing.expectEqual(@as(u8, 2), quot.numeric.scale);
}

test "Numeric + integer arithmetic" {
    const a = Value{ .numeric = .{ .value = 150, .scale = 2 } }; // 1.50
    const b = Value{ .integer = 3 };

    // numeric + integer
    const sum = evalArithmetic(a, b, .add);
    try std.testing.expectEqual(@as(i128, 450), sum.numeric.value);
    try std.testing.expectEqual(@as(u8, 2), sum.numeric.scale);

    // integer + numeric
    const sum2 = evalArithmetic(b, a, .add);
    try std.testing.expectEqual(@as(i128, 450), sum2.numeric.value);

    // numeric * integer: 1.50 * 3 = 4.50
    const prod = evalArithmetic(a, b, .mul);
    try std.testing.expectEqual(@as(i128, 450), prod.numeric.value);
    try std.testing.expectEqual(@as(u8, 2), prod.numeric.scale);
}

test "Numeric toInteger and toReal" {
    const n = Value{ .numeric = .{ .value = 12345, .scale = 2 } }; // 123.45
    try std.testing.expectEqual(@as(i64, 123), n.toInteger().?);

    const r = n.toReal().?;
    try std.testing.expect(@abs(r - 123.45) < 0.001);

    // Zero
    const z = Value{ .numeric = .{ .value = 0, .scale = 2 } };
    try std.testing.expectEqual(@as(i64, 0), z.toInteger().?);
    try std.testing.expect(z.toReal().? == 0.0);
}

test "Numeric serialization roundtrip" {
    const allocator = std.testing.allocator;

    const values = [_]Value{
        .{ .numeric = .{ .value = 12345, .scale = 2 } },
        .{ .numeric = .{ .value = -99900, .scale = 3 } },
        .{ .numeric = .{ .value = 0, .scale = 0 } },
    };

    const data = try serializeRow(allocator, &values);
    defer allocator.free(data);

    const deserialized = try deserializeRow(allocator, data);
    defer {
        for (deserialized) |v| v.free(allocator);
        allocator.free(deserialized);
    }

    try std.testing.expectEqual(@as(usize, 3), deserialized.len);
    try std.testing.expectEqual(@as(i128, 12345), deserialized[0].numeric.value);
    try std.testing.expectEqual(@as(u8, 2), deserialized[0].numeric.scale);
    try std.testing.expectEqual(@as(i128, -99900), deserialized[1].numeric.value);
    try std.testing.expectEqual(@as(u8, 3), deserialized[1].numeric.scale);
    try std.testing.expectEqual(@as(i128, 0), deserialized[2].numeric.value);
    try std.testing.expectEqual(@as(u8, 0), deserialized[2].numeric.scale);
}

test "Numeric isTruthy" {
    const nonzero = Value{ .numeric = .{ .value = 1, .scale = 0 } };
    try std.testing.expect(nonzero.isTruthy());

    const zero = Value{ .numeric = .{ .value = 0, .scale = 2 } };
    try std.testing.expect(!zero.isTruthy());
}

test "Numeric CAST from text" {
    const allocator = std.testing.allocator;
    const result = try evalCast(allocator, Value{ .text = "123.45" }, .type_numeric);
    try std.testing.expectEqual(@as(i128, 12345), result.numeric.value);
    try std.testing.expectEqual(@as(u8, 2), result.numeric.scale);
}

test "Numeric CAST from integer" {
    const allocator = std.testing.allocator;
    const result = try evalCast(allocator, Value{ .integer = 42 }, .type_numeric);
    try std.testing.expectEqual(@as(i128, 42), result.numeric.value);
    try std.testing.expectEqual(@as(u8, 0), result.numeric.scale);
}

test "Numeric CAST to text" {
    const allocator = std.testing.allocator;
    const result = try evalCast(allocator, Value{ .numeric = .{ .value = 12345, .scale = 2 } }, .type_text);
    defer result.free(allocator);
    try std.testing.expectEqualStrings("123.45", result.text);
}

test "Numeric CAST to integer (truncates)" {
    const allocator = std.testing.allocator;
    const result = try evalCast(allocator, Value{ .numeric = .{ .value = 12345, .scale = 2 } }, .type_integer);
    try std.testing.expectEqual(@as(i64, 123), result.integer);
}

test "powI128" {
    try std.testing.expectEqual(@as(i128, 1), powI128(10, 0));
    try std.testing.expectEqual(@as(i128, 10), powI128(10, 1));
    try std.testing.expectEqual(@as(i128, 100), powI128(10, 2));
    try std.testing.expectEqual(@as(i128, 1000), powI128(10, 3));
    try std.testing.expectEqual(@as(i128, 1000000), powI128(10, 6));
}

// ── UUID Unit Tests ──────────────────────────────────────────────────────

test "UUID parseUuidString with dashes" {
    const result = parseUuidString("550e8400-e29b-41d4-a716-446655440000");
    try std.testing.expect(result != null);
    const bytes = result.?;
    try std.testing.expectEqual(@as(u8, 0x55), bytes[0]);
    try std.testing.expectEqual(@as(u8, 0x0e), bytes[1]);
    try std.testing.expectEqual(@as(u8, 0x84), bytes[2]);
    try std.testing.expectEqual(@as(u8, 0x00), bytes[3]);
    try std.testing.expectEqual(@as(u8, 0xe2), bytes[4]);
    try std.testing.expectEqual(@as(u8, 0x9b), bytes[5]);
}

test "UUID parseUuidString without dashes" {
    const result = parseUuidString("550e8400e29b41d4a716446655440000");
    try std.testing.expect(result != null);
    const bytes = result.?;
    try std.testing.expectEqual(@as(u8, 0x55), bytes[0]);
    try std.testing.expectEqual(@as(u8, 0x00), bytes[15]);
}

test "UUID parseUuidString uppercase" {
    const result = parseUuidString("550E8400-E29B-41D4-A716-446655440000");
    try std.testing.expect(result != null);
}

test "UUID parseUuidString invalid" {
    try std.testing.expect(parseUuidString("not-a-uuid") == null);
    try std.testing.expect(parseUuidString("550e8400-e29b-41d4-a716") == null); // too short
    try std.testing.expect(parseUuidString("zzzzzzzz-zzzz-zzzz-zzzz-zzzzzzzzzzzz") == null); // invalid hex
}

test "UUID formatUuid roundtrip" {
    const allocator = std.testing.allocator;
    const input = "550e8400-e29b-41d4-a716-446655440000";
    const bytes = parseUuidString(input).?;
    const formatted = try formatUuid(allocator, bytes);
    defer allocator.free(formatted);
    try std.testing.expectEqualStrings(input, formatted);
}

test "UUID generateUuidV4 version and variant" {
    const uuid = generateUuidV4();
    // Version 4: byte 6 high nibble must be 0x4
    try std.testing.expectEqual(@as(u8, 0x40), uuid[6] & 0xf0);
    // Variant 2: byte 8 top 2 bits must be 10
    try std.testing.expectEqual(@as(u8, 0x80), uuid[8] & 0xc0);
}

test "UUID comparison" {
    const a = Value{ .uuid = parseUuidString("00000000-0000-0000-0000-000000000001").? };
    const b = Value{ .uuid = parseUuidString("00000000-0000-0000-0000-000000000002").? };
    const c = Value{ .uuid = parseUuidString("00000000-0000-0000-0000-000000000001").? };
    try std.testing.expectEqual(std.math.Order.lt, a.compare(b));
    try std.testing.expectEqual(std.math.Order.gt, b.compare(a));
    try std.testing.expectEqual(std.math.Order.eq, a.compare(c));
}

test "UUID serialization roundtrip" {
    const allocator = std.testing.allocator;
    const original = parseUuidString("550e8400-e29b-41d4-a716-446655440000").?;
    const row = [_]Value{Value{ .uuid = original }};
    const data = try serializeRow(allocator, &row);
    defer allocator.free(data);
    const deserialized = try deserializeRow(allocator, data);
    defer allocator.free(deserialized);
    try std.testing.expectEqualSlices(u8, &original, &deserialized[0].uuid);
}

test "UUID isTruthy" {
    const v = Value{ .uuid = parseUuidString("550e8400-e29b-41d4-a716-446655440000").? };
    try std.testing.expect(v.isTruthy());
}
