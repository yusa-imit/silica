//! MVCC (Multi-Version Concurrency Control) — Snapshot isolation and visibility.
//!
//! Every row version carries a tuple header with (xmin, xmax) transaction IDs.
//! xmin = the transaction that created this version.
//! xmax = the transaction that deleted/updated this version (0 = still live).
//!
//! Visibility rules determine which row versions a transaction can see,
//! based on its snapshot of active transactions at the time the snapshot
//! was taken.
//!
//! Isolation levels:
//!   READ COMMITTED  — fresh snapshot per statement
//!   REPEATABLE READ — single snapshot for entire transaction
//!   SERIALIZABLE    — snapshot + SSI conflict detection (future)

const std = @import("std");
const Allocator = std.mem.Allocator;
const executor_mod = @import("../sql/executor.zig");
const Value = executor_mod.Value;

// ── Constants ──────────────────────────────────────────────────────────

/// Invalid/unassigned transaction ID.
pub const INVALID_XID: u32 = 0;

/// Bootstrap transaction (used for initial schema creation).
pub const BOOTSTRAP_XID: u32 = 1;

/// First user transaction ID.
pub const FIRST_NORMAL_XID: u32 = 2;

/// Maximum transaction ID before wraparound.
pub const MAX_XID: u32 = std.math.maxInt(u32) - 1;

/// Frozen XID — marks a tuple as visible to all transactions (vacuumed).
pub const FROZEN_XID: u32 = 1;

/// Size of the MVCC tuple header in bytes.
/// Layout: [xmin:4][xmax:4][cid:2][flags:1][padding:1] = 12 bytes
pub const TUPLE_HEADER_SIZE: usize = 12;

// ── Tuple Header ───────────────────────────────────────────────────────

/// Flags for tuple header state.
pub const TupleFlags = packed struct(u8) {
    /// xmin is known committed.
    xmin_committed: bool = false,
    /// xmin is known aborted.
    xmin_aborted: bool = false,
    /// xmax is known committed.
    xmax_committed: bool = false,
    /// xmax is known aborted.
    xmax_aborted: bool = false,
    /// Tuple has been updated (xmax points to new version).
    updated: bool = false,
    /// Reserved bits.
    _reserved: u3 = 0,
};

/// MVCC tuple header prepended to every row version.
pub const TupleHeader = struct {
    /// Transaction ID that created this row version.
    xmin: u32,
    /// Transaction ID that deleted/replaced this version (0 = live).
    xmax: u32,
    /// Command ID within the transaction (for intra-transaction visibility).
    cid: u16,
    /// Hint flags to avoid repeated transaction status lookups.
    flags: TupleFlags,

    /// Serialize the tuple header to bytes.
    pub fn serialize(self: TupleHeader, buf: *[TUPLE_HEADER_SIZE]u8) void {
        std.mem.writeInt(u32, buf[0..4], self.xmin, .little);
        std.mem.writeInt(u32, buf[4..8], self.xmax, .little);
        std.mem.writeInt(u16, buf[8..10], self.cid, .little);
        buf[10] = @bitCast(self.flags);
        buf[11] = 0; // padding
    }

    /// Deserialize a tuple header from bytes.
    pub fn deserialize(buf: *const [TUPLE_HEADER_SIZE]u8) TupleHeader {
        return .{
            .xmin = std.mem.readInt(u32, buf[0..4], .little),
            .xmax = std.mem.readInt(u32, buf[4..8], .little),
            .cid = std.mem.readInt(u16, buf[8..10], .little),
            .flags = @bitCast(buf[10]),
        };
    }

    /// Create a header for a newly inserted row.
    pub fn forInsert(xid: u32, cid: u16) TupleHeader {
        return .{
            .xmin = xid,
            .xmax = INVALID_XID,
            .cid = cid,
            .flags = .{},
        };
    }

    /// Mark this tuple as deleted by the given transaction.
    pub fn markDeleted(self: *TupleHeader, xid: u32, cid: u16) void {
        self.xmax = xid;
        self.cid = cid;
        self.flags.updated = false;
    }

    /// Mark this tuple as updated (deleted + replaced by new version).
    pub fn markUpdated(self: *TupleHeader, xid: u32, cid: u16) void {
        self.xmax = xid;
        self.cid = cid;
        self.flags.updated = true;
    }

    /// Check if the tuple is "frozen" — visible to all transactions.
    pub fn isFrozen(self: TupleHeader) bool {
        return self.xmin == FROZEN_XID;
    }
};

// ── Isolation Level ────────────────────────────────────────────────────

pub const IsolationLevel = enum {
    read_committed,
    repeatable_read,
    serializable,
};

// ── Transaction State ──────────────────────────────────────────────────

pub const TransactionState = enum {
    active,
    committed,
    aborted,
};

// ── Snapshot ───────────────────────────────────────────────────────────

/// A point-in-time view of which transactions are active.
/// Used to determine tuple visibility.
pub const Snapshot = struct {
    /// The lowest XID that was active when the snapshot was taken.
    /// All XIDs < xmin are either committed or aborted.
    xmin: u32,
    /// The first XID that had not yet been assigned when the snapshot was taken.
    /// All XIDs >= xmax are invisible.
    xmax: u32,
    /// Set of XIDs that were active (in-progress) when the snapshot was taken.
    /// These are in the range [xmin, xmax) and their tuples are invisible.
    active_xids: []const u32,

    /// Allocator used for active_xids (null for static/empty snapshots).
    allocator: ?Allocator,

    /// An empty snapshot that sees everything (used for bootstrap/DDL).
    pub const EMPTY = Snapshot{
        .xmin = FIRST_NORMAL_XID,
        .xmax = FIRST_NORMAL_XID,
        .active_xids = &.{},
        .allocator = null,
    };

    /// Check if a transaction ID is considered "in progress" in this snapshot.
    pub fn isActive(self: Snapshot, xid: u32) bool {
        if (xid < self.xmin) return false;
        if (xid >= self.xmax) return true;
        for (self.active_xids) |active| {
            if (active == xid) return true;
        }
        return false;
    }

    /// Check if a transaction ID is visible (committed and not active).
    pub fn isVisible(self: Snapshot, xid: u32) bool {
        if (xid == FROZEN_XID) return true;
        if (xid == INVALID_XID) return false;
        return !self.isActive(xid);
    }

    pub fn deinit(self: *Snapshot) void {
        if (self.allocator) |alloc| {
            if (self.active_xids.len > 0) {
                alloc.free(self.active_xids);
            }
        }
        self.* = undefined;
    }
};

// ── Visibility Check ───────────────────────────────────────────────────

/// Determine if a tuple is visible to the given snapshot.
///
/// Visibility rules (PostgreSQL-compatible):
///   1. If xmin is not committed → invisible (inserting txn not done)
///      UNLESS xmin == my own transaction → visible if cid < my current cid
///   2. If xmin is committed (or frozen):
///      a. If xmax is invalid (0) → visible (not deleted)
///      b. If xmax is not committed → visible (deleting txn not done)
///         UNLESS xmax == my own transaction → invisible if cid < my current cid
///      c. If xmax is committed → invisible (deleted)
///
/// For simplicity we use the snapshot approach:
///   - xmin visible = snapshot.isVisible(xmin) OR (xmin == current_xid AND cid check)
///   - xmax visible = snapshot.isVisible(xmax) AND xmax != 0
pub fn isTupleVisible(
    header: TupleHeader,
    snapshot: Snapshot,
    current_xid: u32,
    current_cid: u16,
) bool {
    return isTupleVisibleWithTm(header, snapshot, current_xid, current_cid, null);
}

/// Determine tuple visibility, optionally consulting TransactionManager for
/// commit/abort status when hint flags are not set. Without a TM reference,
/// non-hinted transactions that are not in the active snapshot are assumed
/// committed (the original behavior). With a TM reference, aborted
/// transactions are correctly identified as invisible.
pub fn isTupleVisibleWithTm(
    header: TupleHeader,
    snapshot: Snapshot,
    current_xid: u32,
    current_cid: u16,
    tm: ?*TransactionManager,
) bool {
    // Rule 1: Is the creating transaction visible?
    const xmin_visible = blk: {
        if (header.xmin == FROZEN_XID) break :blk true;
        if (header.xmin == current_xid) {
            // Our own transaction: visible if created before current command
            if (header.cid >= current_cid) break :blk false;
            break :blk true;
        }
        // Check hint flags first
        if (header.flags.xmin_aborted) break :blk false;
        if (header.flags.xmin_committed) break :blk true;
        // Consult TransactionManager if available (correct aborted detection)
        if (tm) |t| {
            if (t.isAborted(header.xmin)) break :blk false;
            if (t.isCommitted(header.xmin)) break :blk true;
        }
        // Fall back to snapshot
        break :blk snapshot.isVisible(header.xmin);
    };

    if (!xmin_visible) return false;

    // Rule 2: Is the deleting transaction visible? (if any)
    if (header.xmax == INVALID_XID) return true; // Not deleted

    const xmax_visible = blk: {
        if (header.xmax == current_xid) {
            // Our own transaction deleted this: invisible if deleted before current command
            if (header.cid < current_cid) break :blk true;
            break :blk false;
        }
        // Check hint flags first
        if (header.flags.xmax_aborted) break :blk false;
        if (header.flags.xmax_committed) break :blk true;
        // Consult TransactionManager if available
        if (tm) |t| {
            if (t.isAborted(header.xmax)) break :blk false;
            if (t.isCommitted(header.xmax)) break :blk true;
        }
        // Fall back to snapshot
        break :blk snapshot.isVisible(header.xmax);
    };

    // If deleting transaction is visible, tuple is invisible (it's been deleted)
    return !xmax_visible;
}

// ── Transaction Manager ────────────────────────────────────────────────

/// Manages transaction IDs, snapshots, and transaction state.
pub const TransactionManager = struct {
    allocator: Allocator,
    /// Next transaction ID to assign.
    next_xid: u32,
    /// Map of active transactions: XID → state.
    active_txns: std.AutoHashMapUnmanaged(u32, TransactionInfo),
    /// Oldest active XID (for VACUUM horizon).
    oldest_active_xid: u32,

    pub const TransactionInfo = struct {
        state: TransactionState,
        isolation: IsolationLevel,
        /// Snapshot taken at transaction start (for REPEATABLE READ / SERIALIZABLE).
        snapshot: ?Snapshot,
        /// Current command ID within this transaction.
        current_cid: u16,
    };

    pub fn init(allocator: Allocator) TransactionManager {
        return .{
            .allocator = allocator,
            .next_xid = FIRST_NORMAL_XID,
            .active_txns = .{},
            .oldest_active_xid = FIRST_NORMAL_XID,
        };
    }

    pub fn deinit(self: *TransactionManager) void {
        // Clean up any remaining snapshots
        var it = self.active_txns.iterator();
        while (it.next()) |entry| {
            if (entry.value_ptr.snapshot) |*snap| {
                snap.deinit();
            }
        }
        self.active_txns.deinit(self.allocator);
    }

    /// Begin a new transaction. Returns the assigned XID.
    pub fn begin(self: *TransactionManager, isolation: IsolationLevel) !u32 {
        if (self.next_xid > MAX_XID) return error.TransactionIdWraparound;

        const xid = self.next_xid;
        self.next_xid += 1;

        const info = TransactionInfo{
            .state = .active,
            .isolation = isolation,
            .snapshot = null,
            .current_cid = 0,
        };

        // Add to active transactions BEFORE taking snapshot so the
        // snapshot includes this transaction as active (PostgreSQL behavior).
        try self.active_txns.put(self.allocator, xid, info);
        errdefer _ = self.active_txns.remove(xid);

        // For REPEATABLE READ and SERIALIZABLE, take snapshot at transaction start
        if (isolation != .read_committed) {
            const snapshot = try self.takeSnapshot();
            const entry = self.active_txns.getPtr(xid).?;
            entry.snapshot = snapshot;
        }

        self.updateOldestActive();

        return xid;
    }

    /// Take a snapshot of currently active transactions.
    pub fn takeSnapshot(self: *TransactionManager) !Snapshot {
        // Count active transactions
        var count: usize = 0;
        var it = self.active_txns.iterator();
        while (it.next()) |entry| {
            if (entry.value_ptr.state == .active) count += 1;
        }

        const active_xids = try self.allocator.alloc(u32, count);
        errdefer self.allocator.free(active_xids);

        var idx: usize = 0;
        var min_active: u32 = self.next_xid;
        it = self.active_txns.iterator();
        while (it.next()) |entry| {
            if (entry.value_ptr.state == .active) {
                active_xids[idx] = entry.key_ptr.*;
                if (entry.key_ptr.* < min_active) min_active = entry.key_ptr.*;
                idx += 1;
            }
        }

        return .{
            .xmin = min_active,
            .xmax = self.next_xid,
            .active_xids = active_xids,
            .allocator = self.allocator,
        };
    }

    /// Get a snapshot for the current statement in a transaction.
    /// READ COMMITTED: fresh snapshot every time.
    /// REPEATABLE READ / SERIALIZABLE: reuse transaction snapshot.
    pub fn getSnapshot(self: *TransactionManager, xid: u32) !Snapshot {
        const info = self.active_txns.getPtr(xid) orelse return error.TransactionNotFound;
        if (info.state != .active) return error.TransactionNotActive;

        return switch (info.isolation) {
            .read_committed => try self.takeSnapshot(),
            .repeatable_read, .serializable => info.snapshot orelse return error.SnapshotMissing,
        };
    }

    /// Advance the command ID for a transaction (called per statement).
    pub fn advanceCid(self: *TransactionManager, xid: u32) !u16 {
        const info = self.active_txns.getPtr(xid) orelse return error.TransactionNotFound;
        if (info.state != .active) return error.TransactionNotActive;
        const cid = info.current_cid;
        if (cid == std.math.maxInt(u16)) return error.CommandIdOverflow;
        info.current_cid += 1;
        return cid;
    }

    /// Get the current command ID for a transaction.
    pub fn getCurrentCid(self: *TransactionManager, xid: u32) !u16 {
        const info = self.active_txns.getPtr(xid) orelse return error.TransactionNotFound;
        return info.current_cid;
    }

    /// Reset the command ID for a transaction (used by ROLLBACK TO SAVEPOINT).
    pub fn resetCid(self: *TransactionManager, xid: u32, cid: u16) !void {
        const info = self.active_txns.getPtr(xid) orelse return error.TransactionNotFound;
        if (info.state != .active) return error.TransactionNotActive;
        info.current_cid = cid;
    }

    /// Commit a transaction.
    pub fn commit(self: *TransactionManager, xid: u32) !void {
        const info = self.active_txns.getPtr(xid) orelse return error.TransactionNotFound;
        if (info.state != .active) return error.TransactionNotActive;

        info.state = .committed;
        // Free snapshot if it was transaction-scoped
        if (info.isolation != .read_committed) {
            if (info.snapshot) |*snap| {
                snap.deinit();
                info.snapshot = null;
            }
        }
        self.updateOldestActive();
    }

    /// Abort (rollback) a transaction.
    pub fn abort(self: *TransactionManager, xid: u32) !void {
        const info = self.active_txns.getPtr(xid) orelse return error.TransactionNotFound;
        if (info.state != .active) return error.TransactionNotActive;

        info.state = .aborted;
        if (info.isolation != .read_committed) {
            if (info.snapshot) |*snap| {
                snap.deinit();
                info.snapshot = null;
            }
        }
        self.updateOldestActive();
    }

    /// Get the state of a transaction.
    pub fn getState(self: *TransactionManager, xid: u32) ?TransactionState {
        const info = self.active_txns.get(xid) orelse return null;
        return info.state;
    }

    /// Check if a transaction is committed.
    pub fn isCommitted(self: *TransactionManager, xid: u32) bool {
        if (xid == FROZEN_XID or xid == BOOTSTRAP_XID) return true;
        const info = self.active_txns.get(xid) orelse return true; // Unknown = committed (pruned)
        return info.state == .committed;
    }

    /// Check if a transaction is aborted.
    pub fn isAborted(self: *TransactionManager, xid: u32) bool {
        const info = self.active_txns.get(xid) orelse return false;
        return info.state == .aborted;
    }

    /// Get the oldest active XID — VACUUM horizon.
    pub fn getVacuumHorizon(self: *TransactionManager) u32 {
        return self.oldest_active_xid;
    }

    /// Clean up completed transactions that are below the VACUUM horizon.
    /// This prevents the active_txns map from growing indefinitely.
    pub fn pruneCompleted(self: *TransactionManager) void {
        var to_remove = std.ArrayListUnmanaged(u32){};
        defer to_remove.deinit(self.allocator);

        var it = self.active_txns.iterator();
        while (it.next()) |entry| {
            if (entry.value_ptr.state != .active and
                entry.key_ptr.* < self.oldest_active_xid)
            {
                to_remove.append(self.allocator, entry.key_ptr.*) catch continue;
            }
        }

        for (to_remove.items) |xid| {
            _ = self.active_txns.remove(xid);
        }
    }

    fn updateOldestActive(self: *TransactionManager) void {
        var oldest: u32 = self.next_xid;
        var it = self.active_txns.iterator();
        while (it.next()) |entry| {
            if (entry.value_ptr.state == .active and entry.key_ptr.* < oldest) {
                oldest = entry.key_ptr.*;
            }
        }
        self.oldest_active_xid = oldest;
    }
};

// ── Versioned Row Serialization ────────────────────────────────────────

/// Serialize a versioned row: version byte + MVCC header + column data.
/// Format: [0xAA][TupleHeader 12B][col_count:2][columns...]
pub fn serializeVersionedRow(
    allocator: Allocator,
    header: TupleHeader,
    values: []const Value,
) ![]u8 {
    // Serialize column data
    const col_data = try executor_mod.serializeRow(allocator, values);
    defer allocator.free(col_data);

    // Prepend version byte + MVCC header
    const buf = try allocator.alloc(u8, MVCC_ROW_OVERHEAD + col_data.len);
    errdefer allocator.free(buf);

    buf[0] = ROW_VERSION_MVCC;
    header.serialize(buf[1..][0..TUPLE_HEADER_SIZE]);
    @memcpy(buf[MVCC_ROW_OVERHEAD..], col_data);

    return buf;
}

/// Deserialize a versioned row into MVCC header + column values.
/// Expects format: [0xAA][TupleHeader 12B][col_count:2][columns...]
pub fn deserializeVersionedRow(
    allocator: Allocator,
    data: []const u8,
) !struct { header: TupleHeader, values: []Value } {
    if (data.len < MVCC_ROW_OVERHEAD + 2) return error.InvalidRowData;
    if (data[0] != ROW_VERSION_MVCC) return error.InvalidRowData;

    const header = TupleHeader.deserialize(data[1..][0..TUPLE_HEADER_SIZE]);

    const values = try executor_mod.deserializeRow(allocator, data[MVCC_ROW_OVERHEAD..]);

    return .{ .header = header, .values = values };
}

/// Row format version prefix byte.
/// Legacy rows have no prefix (start directly with col_count u16).
/// MVCC rows start with this magic byte, followed by the TupleHeader.
pub const ROW_VERSION_MVCC: u8 = 0xAA;

/// Total overhead for MVCC row format: 1 (version byte) + 12 (TupleHeader).
pub const MVCC_ROW_OVERHEAD: usize = 1 + TUPLE_HEADER_SIZE;

/// Detect whether raw row data has an MVCC tuple header.
/// MVCC rows: [0xAA][TupleHeader 12B][col_count:2][columns...]
/// Legacy rows: [col_count:2][type_tag:1][...]
pub fn isVersionedRow(data: []const u8) bool {
    if (data.len < MVCC_ROW_OVERHEAD + 2) return false;
    return data[0] == ROW_VERSION_MVCC;
}

// ── Tests ──────────────────────────────────────────────────────────────

test "TupleHeader serialize/deserialize roundtrip" {
    const header = TupleHeader{
        .xmin = 42,
        .xmax = 100,
        .cid = 3,
        .flags = .{ .xmin_committed = true },
    };

    var buf: [TUPLE_HEADER_SIZE]u8 = undefined;
    header.serialize(&buf);

    const restored = TupleHeader.deserialize(&buf);
    try std.testing.expectEqual(@as(u32, 42), restored.xmin);
    try std.testing.expectEqual(@as(u32, 100), restored.xmax);
    try std.testing.expectEqual(@as(u16, 3), restored.cid);
    try std.testing.expect(restored.flags.xmin_committed);
    try std.testing.expect(!restored.flags.xmin_aborted);
    try std.testing.expect(!restored.flags.xmax_committed);
}

test "TupleHeader forInsert" {
    const header = TupleHeader.forInsert(5, 0);
    try std.testing.expectEqual(@as(u32, 5), header.xmin);
    try std.testing.expectEqual(INVALID_XID, header.xmax);
    try std.testing.expectEqual(@as(u16, 0), header.cid);
}

test "TupleHeader markDeleted and markUpdated" {
    var header = TupleHeader.forInsert(5, 0);

    header.markDeleted(10, 1);
    try std.testing.expectEqual(@as(u32, 10), header.xmax);
    try std.testing.expectEqual(@as(u16, 1), header.cid);
    try std.testing.expect(!header.flags.updated);

    header.markUpdated(11, 2);
    try std.testing.expectEqual(@as(u32, 11), header.xmax);
    try std.testing.expectEqual(@as(u16, 2), header.cid);
    try std.testing.expect(header.flags.updated);
}

test "TupleHeader isFrozen" {
    const frozen = TupleHeader{
        .xmin = FROZEN_XID,
        .xmax = INVALID_XID,
        .cid = 0,
        .flags = .{},
    };
    try std.testing.expect(frozen.isFrozen());

    const normal = TupleHeader.forInsert(5, 0);
    try std.testing.expect(!normal.isFrozen());
}

test "TupleFlags packed struct layout" {
    const flags = TupleFlags{
        .xmin_committed = true,
        .xmax_committed = true,
    };
    const byte: u8 = @bitCast(flags);
    try std.testing.expect(byte != 0);

    const back: TupleFlags = @bitCast(byte);
    try std.testing.expect(back.xmin_committed);
    try std.testing.expect(back.xmax_committed);
    try std.testing.expect(!back.xmin_aborted);
    try std.testing.expect(!back.xmax_aborted);
}

test "Snapshot empty" {
    const snap = Snapshot.EMPTY;
    // Empty snapshot: xmin=2, xmax=2, no active XIDs
    try std.testing.expect(!snap.isActive(1));
    try std.testing.expect(snap.isActive(2)); // >= xmax → active (not yet assigned)
    try std.testing.expect(snap.isVisible(FROZEN_XID));
    try std.testing.expect(!snap.isVisible(INVALID_XID));
}

test "Snapshot isActive and isVisible" {
    const active_xids = [_]u32{ 5, 7 };
    const snap = Snapshot{
        .xmin = 5,
        .xmax = 10,
        .active_xids = &active_xids,
        .allocator = null,
    };

    // XIDs < xmin: not active, visible
    try std.testing.expect(!snap.isActive(3));
    try std.testing.expect(snap.isVisible(3));

    // Active XIDs: active, not visible
    try std.testing.expect(snap.isActive(5));
    try std.testing.expect(!snap.isVisible(5));
    try std.testing.expect(snap.isActive(7));
    try std.testing.expect(!snap.isVisible(7));

    // XIDs in [xmin, xmax) but not active: visible
    try std.testing.expect(!snap.isActive(6));
    try std.testing.expect(snap.isVisible(6));
    try std.testing.expect(!snap.isActive(8));
    try std.testing.expect(snap.isVisible(8));

    // XIDs >= xmax: active (future), not visible
    try std.testing.expect(snap.isActive(10));
    try std.testing.expect(!snap.isVisible(10));
    try std.testing.expect(snap.isActive(100));
    try std.testing.expect(!snap.isVisible(100));
}

test "isTupleVisible — basic cases" {
    const active_xids = [_]u32{5};
    const snap = Snapshot{
        .xmin = 5,
        .xmax = 10,
        .active_xids = &active_xids,
        .allocator = null,
    };

    // Case 1: committed insert, no delete → visible
    {
        const h = TupleHeader{
            .xmin = 3, // committed (< snap.xmin)
            .xmax = INVALID_XID,
            .cid = 0,
            .flags = .{},
        };
        try std.testing.expect(isTupleVisible(h, snap, 8, 0));
    }

    // Case 2: committed insert, committed delete → invisible
    {
        const h = TupleHeader{
            .xmin = 3,
            .xmax = 4, // committed (< snap.xmin)
            .cid = 0,
            .flags = .{},
        };
        try std.testing.expect(!isTupleVisible(h, snap, 8, 0));
    }

    // Case 3: active insert → invisible
    {
        const h = TupleHeader{
            .xmin = 5, // active
            .xmax = INVALID_XID,
            .cid = 0,
            .flags = .{},
        };
        try std.testing.expect(!isTupleVisible(h, snap, 8, 0));
    }

    // Case 4: committed insert, active delete → visible (delete not yet committed)
    {
        const h = TupleHeader{
            .xmin = 3,
            .xmax = 5, // active → not committed
            .cid = 0,
            .flags = .{},
        };
        try std.testing.expect(isTupleVisible(h, snap, 8, 0));
    }

    // Case 5: future insert → invisible
    {
        const h = TupleHeader{
            .xmin = 15, // >= snap.xmax → future
            .xmax = INVALID_XID,
            .cid = 0,
            .flags = .{},
        };
        try std.testing.expect(!isTupleVisible(h, snap, 8, 0));
    }

    // Case 6: frozen tuple → always visible
    {
        const h = TupleHeader{
            .xmin = FROZEN_XID,
            .xmax = INVALID_XID,
            .cid = 0,
            .flags = .{},
        };
        try std.testing.expect(isTupleVisible(h, snap, 8, 0));
    }
}

test "isTupleVisible — own transaction" {
    const snap = Snapshot.EMPTY;

    // Own transaction insert, cid < current_cid → visible
    {
        const h = TupleHeader{
            .xmin = 5,
            .xmax = INVALID_XID,
            .cid = 0,
            .flags = .{},
        };
        try std.testing.expect(isTupleVisible(h, snap, 5, 1));
    }

    // Own transaction insert, cid >= current_cid → invisible
    {
        const h = TupleHeader{
            .xmin = 5,
            .xmax = INVALID_XID,
            .cid = 1,
            .flags = .{},
        };
        try std.testing.expect(!isTupleVisible(h, snap, 5, 1));
    }

    // Own transaction deleted, cid < current_cid → invisible
    {
        const h = TupleHeader{
            .xmin = 3,
            .xmax = 5,
            .cid = 0,
            .flags = .{ .xmin_committed = true },
        };
        try std.testing.expect(!isTupleVisible(h, snap, 5, 1));
    }

    // Own transaction deleted, cid >= current_cid → visible (delete hasn't happened yet for this cmd)
    {
        const h = TupleHeader{
            .xmin = 3,
            .xmax = 5,
            .cid = 2,
            .flags = .{ .xmin_committed = true },
        };
        try std.testing.expect(isTupleVisible(h, snap, 5, 1));
    }
}

test "isTupleVisible — hint flags" {
    const snap = Snapshot{
        .xmin = 5,
        .xmax = 10,
        .active_xids = &.{},
        .allocator = null,
    };

    // xmin_aborted hint → invisible
    {
        const h = TupleHeader{
            .xmin = 6,
            .xmax = INVALID_XID,
            .cid = 0,
            .flags = .{ .xmin_aborted = true },
        };
        try std.testing.expect(!isTupleVisible(h, snap, 8, 0));
    }

    // xmin_committed + xmax_aborted → visible (delete was rolled back)
    {
        const h = TupleHeader{
            .xmin = 6,
            .xmax = 7,
            .cid = 0,
            .flags = .{ .xmin_committed = true, .xmax_aborted = true },
        };
        try std.testing.expect(isTupleVisible(h, snap, 8, 0));
    }

    // xmin_committed + xmax_committed → invisible
    {
        const h = TupleHeader{
            .xmin = 6,
            .xmax = 7,
            .cid = 0,
            .flags = .{ .xmin_committed = true, .xmax_committed = true },
        };
        try std.testing.expect(!isTupleVisible(h, snap, 8, 0));
    }
}

test "TransactionManager — begin and commit" {
    const allocator = std.testing.allocator;
    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    const xid1 = try tm.begin(.read_committed);
    try std.testing.expectEqual(FIRST_NORMAL_XID, xid1);

    const xid2 = try tm.begin(.read_committed);
    try std.testing.expectEqual(FIRST_NORMAL_XID + 1, xid2);

    try std.testing.expectEqual(TransactionState.active, tm.getState(xid1).?);
    try std.testing.expectEqual(TransactionState.active, tm.getState(xid2).?);

    try tm.commit(xid1);
    try std.testing.expectEqual(TransactionState.committed, tm.getState(xid1).?);
    try std.testing.expect(tm.isCommitted(xid1));

    try tm.abort(xid2);
    try std.testing.expectEqual(TransactionState.aborted, tm.getState(xid2).?);
    try std.testing.expect(tm.isAborted(xid2));
}

test "TransactionManager — snapshot for READ COMMITTED" {
    const allocator = std.testing.allocator;
    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    const xid1 = try tm.begin(.read_committed);
    const xid2 = try tm.begin(.read_committed);

    // Each getSnapshot for READ COMMITTED returns a fresh snapshot
    var snap1 = try tm.getSnapshot(xid1);
    defer snap1.deinit();

    // Both xid1 and xid2 should be active in the snapshot
    try std.testing.expect(snap1.isActive(xid1));
    try std.testing.expect(snap1.isActive(xid2));

    // Commit xid2 and take another snapshot
    try tm.commit(xid2);
    var snap2 = try tm.getSnapshot(xid1);
    defer snap2.deinit();

    // Now xid2 should not be active
    try std.testing.expect(snap2.isActive(xid1));
    try std.testing.expect(!snap2.isActive(xid2));

    try tm.commit(xid1);
}

test "TransactionManager — snapshot for REPEATABLE READ" {
    const allocator = std.testing.allocator;
    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    const xid1 = try tm.begin(.read_committed);
    const xid2 = try tm.begin(.repeatable_read);

    // xid2 snapshot was taken at begin time, should see xid1 as active
    const snap = try tm.getSnapshot(xid2);
    try std.testing.expect(snap.isActive(xid1));
    try std.testing.expect(snap.isActive(xid2));

    // Commit xid1 — but xid2's snapshot shouldn't change
    try tm.commit(xid1);
    const snap_again = try tm.getSnapshot(xid2);
    // Same snapshot object — still sees xid1 as active
    try std.testing.expect(snap_again.isActive(xid1));

    try tm.commit(xid2);
}

test "TransactionManager — advanceCid" {
    const allocator = std.testing.allocator;
    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    const xid = try tm.begin(.read_committed);

    const cid0 = try tm.advanceCid(xid);
    try std.testing.expectEqual(@as(u16, 0), cid0);

    const cid1 = try tm.advanceCid(xid);
    try std.testing.expectEqual(@as(u16, 1), cid1);

    const current = try tm.getCurrentCid(xid);
    try std.testing.expectEqual(@as(u16, 2), current);

    try tm.commit(xid);
}

test "TransactionManager — VACUUM horizon" {
    const allocator = std.testing.allocator;
    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    const xid1 = try tm.begin(.read_committed);
    const xid2 = try tm.begin(.read_committed);
    _ = xid2;

    try std.testing.expectEqual(xid1, tm.getVacuumHorizon());

    try tm.commit(xid1);
    // After committing xid1, oldest active should be xid2
    try std.testing.expectEqual(FIRST_NORMAL_XID + 1, tm.getVacuumHorizon());
}

test "TransactionManager — pruneCompleted" {
    const allocator = std.testing.allocator;
    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    const xid1 = try tm.begin(.read_committed);
    const xid2 = try tm.begin(.read_committed);
    const xid3 = try tm.begin(.read_committed);

    try tm.commit(xid1);
    try tm.abort(xid2);
    // xid3 is still active

    tm.pruneCompleted();

    // xid1 and xid2 should be pruned (< oldest_active = xid3)
    try std.testing.expectEqual(@as(?TransactionState, null), tm.getState(xid1));
    try std.testing.expectEqual(@as(?TransactionState, null), tm.getState(xid2));
    // xid3 should still be there
    try std.testing.expectEqual(TransactionState.active, tm.getState(xid3).?);

    try tm.commit(xid3);
}

test "TransactionManager — error on committed/aborted begin operations" {
    const allocator = std.testing.allocator;
    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    const xid = try tm.begin(.read_committed);
    try tm.commit(xid);

    // Committing again should fail
    try std.testing.expectError(error.TransactionNotActive, tm.commit(xid));
    try std.testing.expectError(error.TransactionNotActive, tm.abort(xid));
}

test "TransactionManager — unknown transaction" {
    const allocator = std.testing.allocator;
    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    try std.testing.expectError(error.TransactionNotFound, tm.commit(99));
    try std.testing.expectError(error.TransactionNotFound, tm.abort(99));
    try std.testing.expectEqual(@as(?TransactionState, null), tm.getState(99));
}

test "versioned row serialization roundtrip" {
    const allocator = std.testing.allocator;

    const header = TupleHeader.forInsert(42, 0);
    const values = [_]Value{
        .{ .integer = 1 },
        .{ .text = "hello" },
        .{ .boolean = true },
    };

    const data = try serializeVersionedRow(allocator, header, &values);
    defer allocator.free(data);

    const result = try deserializeVersionedRow(allocator, data);
    defer {
        for (result.values) |v| v.free(allocator);
        allocator.free(result.values);
    }

    try std.testing.expectEqual(@as(u32, 42), result.header.xmin);
    try std.testing.expectEqual(INVALID_XID, result.header.xmax);
    try std.testing.expectEqual(@as(usize, 3), result.values.len);
    try std.testing.expectEqual(@as(i64, 1), result.values[0].integer);
    try std.testing.expect(std.mem.eql(u8, "hello", result.values[1].text));
    try std.testing.expect(result.values[2].boolean);
}

test "versioned row with deleted header" {
    const allocator = std.testing.allocator;

    var header = TupleHeader.forInsert(10, 0);
    header.markDeleted(15, 1);

    const values = [_]Value{
        .{ .integer = 42 },
    };

    const data = try serializeVersionedRow(allocator, header, &values);
    defer allocator.free(data);

    const result = try deserializeVersionedRow(allocator, data);
    defer {
        for (result.values) |v| v.free(allocator);
        allocator.free(result.values);
    }

    try std.testing.expectEqual(@as(u32, 10), result.header.xmin);
    try std.testing.expectEqual(@as(u32, 15), result.header.xmax);
    try std.testing.expectEqual(@as(u16, 1), result.header.cid);
    try std.testing.expect(!result.header.flags.updated);
}

test "Snapshot deinit frees active_xids" {
    const allocator = std.testing.allocator;

    const active = try allocator.alloc(u32, 3);
    active[0] = 10;
    active[1] = 20;
    active[2] = 30;

    var snap = Snapshot{
        .xmin = 10,
        .xmax = 40,
        .active_xids = active,
        .allocator = allocator,
    };

    snap.deinit();
    // If deinit didn't free, test allocator would detect leak
}

test "multiple concurrent transactions with visibility" {
    const allocator = std.testing.allocator;
    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    // Three concurrent transactions
    const xid_a = try tm.begin(.read_committed);
    const xid_b = try tm.begin(.read_committed);
    const xid_c = try tm.begin(.read_committed);

    // Transaction A inserts a row
    const header_a = TupleHeader.forInsert(xid_a, 0);

    // Before A commits: B should not see A's row
    {
        var snap_b = try tm.getSnapshot(xid_b);
        defer snap_b.deinit();
        try std.testing.expect(!isTupleVisible(header_a, snap_b, xid_b, 0));
    }

    // A commits
    try tm.commit(xid_a);

    // After A commits: B should see A's row (READ COMMITTED = fresh snapshot)
    {
        var snap_b = try tm.getSnapshot(xid_b);
        defer snap_b.deinit();
        try std.testing.expect(isTupleVisible(header_a, snap_b, xid_b, 0));
    }

    // B deletes A's row
    var header_b = header_a;
    header_b.markDeleted(xid_b, 0);

    // Before B commits: C should still see the row (delete not committed)
    {
        var snap_c = try tm.getSnapshot(xid_c);
        defer snap_c.deinit();
        try std.testing.expect(isTupleVisible(header_b, snap_c, xid_c, 0));
    }

    // B commits
    try tm.commit(xid_b);

    // After B commits: C should NOT see the row
    {
        var snap_c = try tm.getSnapshot(xid_c);
        defer snap_c.deinit();
        try std.testing.expect(!isTupleVisible(header_b, snap_c, xid_c, 0));
    }

    try tm.commit(xid_c);
}

// ── Stabilization: Edge Case Tests ──────────────────────────────────

test "isTupleVisible — same-txn insert then delete" {
    const snap = Snapshot.EMPTY;

    // Same transaction inserts (cid=0) then deletes (cid=1). Current command is cid=2.
    // xmin visible (cid 0 < 2), xmax visible (cid 1 < 2) → invisible (deleted)
    {
        const h = TupleHeader{
            .xmin = 5,
            .xmax = 5,
            .cid = 1, // delete cid
            .flags = .{},
        };
        // Simulate: insert at cid=0, delete at cid=1, now at cid=2
        // xmin=5, current_xid=5, cid_header=? — but header only stores one cid.
        // In practice, the insert header has cid=0 and the delete header has cid=1.
        // Since both xmin and xmax are current_xid, we check cid for both.
        // xmin check: header.cid(1) >= current_cid(2)? No → xmin visible
        // xmax check: header.cid(1) < current_cid(2)? Yes → xmax visible → tuple invisible
        try std.testing.expect(!isTupleVisible(h, snap, 5, 2));
    }

    // Same txn: delete hasn't happened yet from current command's perspective
    // Insert cid=0, delete cid=2, current at cid=1
    {
        const h = TupleHeader{
            .xmin = 5,
            .xmax = 5,
            .cid = 2, // delete cid (future relative to current)
            .flags = .{},
        };
        // xmin: own txn, cid(2) >= current_cid(1)? Yes → xmin NOT visible? No...
        // Wait — the header stores ONE cid, which is the latest operation's cid.
        // For insert, cid is insert cid. For delete, cid is overwritten to delete cid.
        // So when xmax is set, cid = delete cid. xmin check uses same cid field.
        // This means for same-txn insert+delete, the original insert cid is lost.
        // This is a known PostgreSQL behavior: the tuple header's cid is the last operation's cid.
        // PostgreSQL handles this with combo CIDs for same-transaction insert+delete.
        // For now, our simplified model: cid=2 for both checks.
        // xmin check: cid(2) >= current_cid(1)? Yes → xmin NOT visible → tuple invisible
        try std.testing.expect(!isTupleVisible(h, snap, 5, 1));
    }
}

test "isTupleVisible — cid boundary exactly equal" {
    const snap = Snapshot.EMPTY;

    // xmin = current_xid, cid == current_cid → invisible (not yet created from this cmd's view)
    {
        const h = TupleHeader{
            .xmin = 5,
            .xmax = INVALID_XID,
            .cid = 3,
            .flags = .{},
        };
        try std.testing.expect(!isTupleVisible(h, snap, 5, 3));
    }

    // xmin = current_xid, cid = current_cid - 1 → visible
    {
        const h = TupleHeader{
            .xmin = 5,
            .xmax = INVALID_XID,
            .cid = 2,
            .flags = .{},
        };
        try std.testing.expect(isTupleVisible(h, snap, 5, 3));
    }
}

test "isTupleVisible — frozen tuple with committed xmax" {
    const snap = Snapshot{
        .xmin = 5,
        .xmax = 10,
        .active_xids = &.{},
        .allocator = null,
    };

    // Frozen xmin, committed xmax → invisible (deleted)
    {
        const h = TupleHeader{
            .xmin = FROZEN_XID,
            .xmax = 3, // committed (< snap.xmin)
            .cid = 0,
            .flags = .{},
        };
        try std.testing.expect(!isTupleVisible(h, snap, 8, 0));
    }

    // Frozen xmin, active xmax → visible (delete in progress)
    {
        const active_xids = [_]u32{6};
        const snap2 = Snapshot{
            .xmin = 5,
            .xmax = 10,
            .active_xids = &active_xids,
            .allocator = null,
        };
        const h = TupleHeader{
            .xmin = FROZEN_XID,
            .xmax = 6, // active
            .cid = 0,
            .flags = .{},
        };
        try std.testing.expect(isTupleVisible(h, snap2, 8, 0));
    }
}

test "isTupleVisible — contradictory hint flags" {
    const snap = Snapshot{
        .xmin = 5,
        .xmax = 10,
        .active_xids = &.{},
        .allocator = null,
    };

    // Both xmin_committed and xmin_aborted set (data corruption) → aborted takes priority
    {
        const h = TupleHeader{
            .xmin = 6,
            .xmax = INVALID_XID,
            .cid = 0,
            .flags = .{ .xmin_committed = true, .xmin_aborted = true },
        };
        // aborted is checked first in the code → invisible
        try std.testing.expect(!isTupleVisible(h, snap, 8, 0));
    }

    // Both xmax_committed and xmax_aborted set → aborted takes priority → visible
    {
        const h = TupleHeader{
            .xmin = 6,
            .xmax = 7,
            .cid = 0,
            .flags = .{ .xmin_committed = true, .xmax_committed = true, .xmax_aborted = true },
        };
        // xmax aborted checked first → xmax not visible → tuple IS visible
        try std.testing.expect(isTupleVisible(h, snap, 8, 0));
    }
}

test "Snapshot.isActive — boundary values" {
    const active_xids = [_]u32{ 5, 9 };
    const snap = Snapshot{
        .xmin = 5,
        .xmax = 10,
        .active_xids = &active_xids,
        .allocator = null,
    };

    // xid == xmin AND in active list → active
    try std.testing.expect(snap.isActive(5));

    // xid == xmax - 1 AND in active list → active
    try std.testing.expect(snap.isActive(9));

    // xid == xmax → active (future)
    try std.testing.expect(snap.isActive(10));

    // xid == xmin - 1 → not active
    try std.testing.expect(!snap.isActive(4));

    // xid in [xmin, xmax) but NOT in active list → not active (committed)
    try std.testing.expect(!snap.isActive(6));
    try std.testing.expect(!snap.isActive(7));
    try std.testing.expect(!snap.isActive(8));
}

test "TransactionManager — XID boundary at MAX_XID" {
    const allocator = std.testing.allocator;
    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    // Set next_xid near the maximum
    tm.next_xid = MAX_XID;

    // Should succeed: MAX_XID is assignable
    const xid = try tm.begin(.read_committed);
    try std.testing.expectEqual(MAX_XID, xid);
    try tm.commit(xid);

    // Now next_xid = MAX_XID + 1, should fail
    try std.testing.expectError(error.TransactionIdWraparound, tm.begin(.read_committed));
}

test "TransactionManager — CID overflow" {
    const allocator = std.testing.allocator;
    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    const xid = try tm.begin(.read_committed);

    // Set CID near max via direct manipulation (skip 65534 calls)
    const info = tm.active_txns.getPtr(xid).?;
    info.current_cid = std.math.maxInt(u16);

    // Should return the max CID value but then error on next call
    // Actually, advanceCid returns current and increments, so at maxInt it should error
    try std.testing.expectError(error.CommandIdOverflow, tm.advanceCid(xid));

    // Verify it didn't wrap
    try std.testing.expectEqual(std.math.maxInt(u16), info.current_cid);

    try tm.commit(xid);
}

test "TransactionManager — takeSnapshot with no active transactions" {
    const allocator = std.testing.allocator;
    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    // Begin and immediately commit a transaction
    const xid = try tm.begin(.read_committed);
    try tm.commit(xid);

    // Begin another just to take a snapshot from
    const xid2 = try tm.begin(.read_committed);
    try tm.commit(xid2);

    // All committed, take raw snapshot (only xid1 and xid2 are in map, both committed)
    // Only getSnapshot checks active state, takeSnapshot just scans
    var snap = try tm.takeSnapshot();
    defer snap.deinit();

    // No active txns: active_xids should be empty
    try std.testing.expectEqual(@as(usize, 0), snap.active_xids.len);
    // xmin should be next_xid (no active txns to lower it)
    try std.testing.expectEqual(tm.next_xid, snap.xmin);
    try std.testing.expectEqual(tm.next_xid, snap.xmax);
}

test "TransactionManager — isCommitted special XIDs" {
    const allocator = std.testing.allocator;
    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    // FROZEN_XID and BOOTSTRAP_XID are always committed
    try std.testing.expect(tm.isCommitted(FROZEN_XID));
    try std.testing.expect(tm.isCommitted(BOOTSTRAP_XID));

    // Unknown XID (pruned) is considered committed
    try std.testing.expect(tm.isCommitted(999));

    // Active transaction is NOT committed
    const xid = try tm.begin(.read_committed);
    try std.testing.expect(!tm.isCommitted(xid));

    // After commit, it IS committed
    try tm.commit(xid);
    try std.testing.expect(tm.isCommitted(xid));
}

test "TransactionManager — isAborted edge cases" {
    const allocator = std.testing.allocator;
    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    // Unknown XID → not aborted
    try std.testing.expect(!tm.isAborted(999));

    // Active → not aborted
    const xid = try tm.begin(.read_committed);
    try std.testing.expect(!tm.isAborted(xid));

    // Committed → not aborted
    try tm.commit(xid);
    try std.testing.expect(!tm.isAborted(xid));

    // Aborted → aborted
    const xid2 = try tm.begin(.read_committed);
    try tm.abort(xid2);
    try std.testing.expect(tm.isAborted(xid2));
}

test "TransactionManager — pruneCompleted with nothing to prune" {
    const allocator = std.testing.allocator;
    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    const xid1 = try tm.begin(.read_committed);
    const xid2 = try tm.begin(.read_committed);

    // All active — prune should do nothing
    tm.pruneCompleted();

    try std.testing.expectEqual(TransactionState.active, tm.getState(xid1).?);
    try std.testing.expectEqual(TransactionState.active, tm.getState(xid2).?);

    try tm.commit(xid1);
    try tm.commit(xid2);
}

test "TransactionManager — all transactions aborted vacuum horizon" {
    const allocator = std.testing.allocator;
    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    const xid1 = try tm.begin(.read_committed);
    const xid2 = try tm.begin(.read_committed);

    try tm.abort(xid1);
    try tm.abort(xid2);

    // No active transactions — vacuum horizon should be next_xid
    try std.testing.expectEqual(tm.next_xid, tm.getVacuumHorizon());
}

test "TransactionManager — SERIALIZABLE isolation snapshot lifecycle" {
    const allocator = std.testing.allocator;
    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    const xid1 = try tm.begin(.read_committed);
    const xid2 = try tm.begin(.serializable);

    // Serializable gets snapshot at begin (same as repeatable_read)
    const snap = try tm.getSnapshot(xid2);
    try std.testing.expect(snap.isActive(xid1));
    try std.testing.expect(snap.isActive(xid2));

    // Commit xid1 — serializable snapshot should NOT change
    try tm.commit(xid1);
    const snap_after = try tm.getSnapshot(xid2);
    try std.testing.expect(snap_after.isActive(xid1)); // Still sees xid1 as active

    try tm.commit(xid2);
}

test "serializeVersionedRow — empty values" {
    const allocator = std.testing.allocator;

    const header = TupleHeader.forInsert(10, 0);
    const empty_values = [_]Value{};

    const data = try serializeVersionedRow(allocator, header, &empty_values);
    defer allocator.free(data);

    // Should be MVCC_ROW_OVERHEAD + 2 bytes (magic byte + header + col_count=0)
    try std.testing.expectEqual(MVCC_ROW_OVERHEAD + 2, data.len);

    const result = try deserializeVersionedRow(allocator, data);
    defer allocator.free(result.values);

    try std.testing.expectEqual(@as(u32, 10), result.header.xmin);
    try std.testing.expectEqual(@as(usize, 0), result.values.len);
}

test "serializeVersionedRow — null values" {
    const allocator = std.testing.allocator;

    const header = TupleHeader.forInsert(7, 0);
    const values = [_]Value{
        .null_value,
        .{ .integer = 42 },
        .null_value,
    };

    const data = try serializeVersionedRow(allocator, header, &values);
    defer allocator.free(data);

    const result = try deserializeVersionedRow(allocator, data);
    defer {
        for (result.values) |v| v.free(allocator);
        allocator.free(result.values);
    }

    try std.testing.expectEqual(@as(usize, 3), result.values.len);
    try std.testing.expectEqual(Value.null_value, result.values[0]);
    try std.testing.expectEqual(@as(i64, 42), result.values[1].integer);
    try std.testing.expectEqual(Value.null_value, result.values[2]);
}

test "deserializeVersionedRow — data too short" {
    const allocator = std.testing.allocator;

    // Less than MVCC_ROW_OVERHEAD + 2 bytes
    const short_data = [_]u8{ROW_VERSION_MVCC} ++ [_]u8{0} ** TUPLE_HEADER_SIZE;
    try std.testing.expectError(error.InvalidRowData, deserializeVersionedRow(allocator, &short_data));

    // Too short even for magic byte check
    const very_short = [_]u8{ROW_VERSION_MVCC} ++ [_]u8{0} ** 5;
    try std.testing.expectError(error.InvalidRowData, deserializeVersionedRow(allocator, &very_short));

    // Empty data
    try std.testing.expectError(error.InvalidRowData, deserializeVersionedRow(allocator, &[_]u8{}));

    // Wrong magic byte
    var wrong_magic: [MVCC_ROW_OVERHEAD + 2]u8 = undefined;
    wrong_magic[0] = 0x00; // not ROW_VERSION_MVCC
    try std.testing.expectError(error.InvalidRowData, deserializeVersionedRow(allocator, &wrong_magic));
}

test "isVersionedRow — detection" {
    // Too short — always false
    try std.testing.expect(!isVersionedRow(&[_]u8{ROW_VERSION_MVCC} ** 5));
    try std.testing.expect(!isVersionedRow(&[_]u8{}));
    try std.testing.expect(!isVersionedRow(&[_]u8{ROW_VERSION_MVCC}));
    // Exactly MVCC_ROW_OVERHEAD + 1 (need +2 for col_count) — false
    var almost: [MVCC_ROW_OVERHEAD + 1]u8 = undefined;
    almost[0] = ROW_VERSION_MVCC;
    try std.testing.expect(!isVersionedRow(&almost));

    // All zeros — no magic byte → false (legacy row)
    try std.testing.expect(!isVersionedRow(&[_]u8{0} ** 20));

    // Valid MVCC row with magic byte
    var valid_mvcc: [MVCC_ROW_OVERHEAD + 4]u8 = undefined;
    valid_mvcc[0] = ROW_VERSION_MVCC;
    const hdr = TupleHeader.forInsert(FIRST_NORMAL_XID, 0);
    hdr.serialize(valid_mvcc[1..][0..TUPLE_HEADER_SIZE]);
    std.mem.writeInt(u16, valid_mvcc[MVCC_ROW_OVERHEAD..][0..2], 1, .little);
    valid_mvcc[MVCC_ROW_OVERHEAD + 2] = 0;
    valid_mvcc[MVCC_ROW_OVERHEAD + 3] = 0;
    try std.testing.expect(isVersionedRow(&valid_mvcc));

    // Legacy row without magic byte — false
    var legacy: [20]u8 = undefined;
    std.mem.writeInt(u16, legacy[0..2], 2, .little);
    legacy[2] = 0x01; // type tag (integer)
    try std.testing.expect(!isVersionedRow(&legacy));

    // Wrong first byte (not 0xAA) — false
    var wrong_byte = valid_mvcc;
    wrong_byte[0] = 0xBB;
    try std.testing.expect(!isVersionedRow(&wrong_byte));

    // Verify roundtrip: serialize a real versioned row and detect it
    const values = &[_]Value{Value{ .integer = 42 }};
    const versioned = try serializeVersionedRow(std.testing.allocator, hdr, values);
    defer std.testing.allocator.free(versioned);
    try std.testing.expect(isVersionedRow(versioned));
    try std.testing.expectEqual(ROW_VERSION_MVCC, versioned[0]);
}

test "TupleFlags — all combinations" {
    // All flags set
    const all_set = TupleFlags{
        .xmin_committed = true,
        .xmin_aborted = true,
        .xmax_committed = true,
        .xmax_aborted = true,
        .updated = true,
    };
    const byte: u8 = @bitCast(all_set);
    try std.testing.expect(byte != 0);
    const back: TupleFlags = @bitCast(byte);
    try std.testing.expect(back.xmin_committed);
    try std.testing.expect(back.xmin_aborted);
    try std.testing.expect(back.xmax_committed);
    try std.testing.expect(back.xmax_aborted);
    try std.testing.expect(back.updated);

    // No flags set
    const none = TupleFlags{};
    const zero: u8 = @bitCast(none);
    try std.testing.expectEqual(@as(u8, 0), zero);
}

test "TupleHeader serialize/deserialize — all fields max values" {
    const header = TupleHeader{
        .xmin = MAX_XID,
        .xmax = MAX_XID,
        .cid = std.math.maxInt(u16),
        .flags = .{
            .xmin_committed = true,
            .xmin_aborted = true,
            .xmax_committed = true,
            .xmax_aborted = true,
            .updated = true,
        },
    };

    var buf: [TUPLE_HEADER_SIZE]u8 = undefined;
    header.serialize(&buf);

    const restored = TupleHeader.deserialize(&buf);
    try std.testing.expectEqual(MAX_XID, restored.xmin);
    try std.testing.expectEqual(MAX_XID, restored.xmax);
    try std.testing.expectEqual(std.math.maxInt(u16), restored.cid);
    try std.testing.expect(restored.flags.xmin_committed);
    try std.testing.expect(restored.flags.updated);
}

test "TransactionManager — getSnapshot errors for non-active transaction" {
    const allocator = std.testing.allocator;
    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    const xid = try tm.begin(.read_committed);
    try tm.commit(xid);

    // getSnapshot on committed transaction should fail
    try std.testing.expectError(error.TransactionNotActive, tm.getSnapshot(xid));

    // getSnapshot on unknown transaction should fail
    try std.testing.expectError(error.TransactionNotFound, tm.getSnapshot(999));
}

test "TransactionManager — advanceCid errors" {
    const allocator = std.testing.allocator;
    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    // Unknown transaction
    try std.testing.expectError(error.TransactionNotFound, tm.advanceCid(999));

    // Committed transaction
    const xid = try tm.begin(.read_committed);
    try tm.commit(xid);
    try std.testing.expectError(error.TransactionNotActive, tm.advanceCid(xid));
}

test "TransactionManager — getCurrentCid errors" {
    const allocator = std.testing.allocator;
    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    try std.testing.expectError(error.TransactionNotFound, tm.getCurrentCid(999));
}

test "TransactionManager — resetCid" {
    const allocator = std.testing.allocator;
    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    const xid = try tm.begin(.read_committed);

    // Advance CID a few times
    _ = try tm.advanceCid(xid);
    _ = try tm.advanceCid(xid);
    _ = try tm.advanceCid(xid);
    try std.testing.expectEqual(@as(u16, 3), try tm.getCurrentCid(xid));

    // Reset CID to 1 (as if rolling back to a savepoint)
    try tm.resetCid(xid, 1);
    try std.testing.expectEqual(@as(u16, 1), try tm.getCurrentCid(xid));

    // Can advance again from the reset position
    const cid = try tm.advanceCid(xid);
    try std.testing.expectEqual(@as(u16, 1), cid);
    try std.testing.expectEqual(@as(u16, 2), try tm.getCurrentCid(xid));
}

test "TransactionManager — resetCid errors" {
    const allocator = std.testing.allocator;
    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    // Error for non-existent transaction
    try std.testing.expectError(error.TransactionNotFound, tm.resetCid(999, 0));

    // Error for non-active transaction
    const xid = try tm.begin(.read_committed);
    try tm.commit(xid);
    try std.testing.expectError(error.TransactionNotActive, tm.resetCid(xid, 0));
}

test "isTupleVisibleWithTm — aborted xmin without hint flags consults TM" {
    const allocator = std.testing.allocator;
    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    // Start and abort a transaction
    const xid1 = try tm.begin(.read_committed);
    try tm.abort(xid1);

    // Start an observer transaction
    const xid2 = try tm.begin(.read_committed);

    // Create tuple with no hint flags — TM must be consulted
    const h = TupleHeader{
        .xmin = xid1,
        .xmax = INVALID_XID,
        .cid = 0,
        .flags = .{}, // No hint flags
    };

    var snap = try tm.getSnapshot(xid2);
    defer snap.deinit();

    // Without TM: snapshot would treat xid1 as committed (pruned/not-active) → visible
    // With TM: correctly identifies xid1 as aborted → invisible
    try std.testing.expect(!isTupleVisibleWithTm(h, snap, xid2, 0, &tm));

    // Without TM reference: falls back to snapshot → incorrectly visible
    try std.testing.expect(isTupleVisible(h, snap, xid2, 0));

    try tm.commit(xid2);
}

test "isTupleVisibleWithTm — aborted xmax without hint flags consults TM" {
    const allocator = std.testing.allocator;
    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    const xid_inserter = try tm.begin(.read_committed);
    const xid_deleter = try tm.begin(.read_committed);
    const xid_observer = try tm.begin(.read_committed);

    // Commit inserter, abort deleter
    try tm.commit(xid_inserter);
    try tm.abort(xid_deleter);

    // Tuple: committed insert, aborted delete — should be visible
    const h = TupleHeader{
        .xmin = xid_inserter,
        .xmax = xid_deleter,
        .cid = 0,
        .flags = .{ .xmin_committed = true }, // xmax has no hint flags
    };

    var snap = try tm.getSnapshot(xid_observer);
    defer snap.deinit();

    // With TM: xmax aborted → delete rolled back → tuple IS visible
    try std.testing.expect(isTupleVisibleWithTm(h, snap, xid_observer, 0, &tm));

    try tm.commit(xid_observer);
}

test "isTupleVisible — own txn future delete not yet visible" {
    const snap = Snapshot{
        .xmin = 5,
        .xmax = 10,
        .active_xids = &.{},
        .allocator = null,
    };

    // Own transaction deleted this row at cid=3, current cid=2
    // Since cid(3) >= current_cid(2), xmax is not visible → tuple IS visible
    const h = TupleHeader{
        .xmin = FROZEN_XID,
        .xmax = 8, // our own xid
        .cid = 3,
        .flags = .{},
    };
    try std.testing.expect(isTupleVisible(h, snap, 8, 2));

    // At cid=4, the delete at cid=3 is visible → tuple is NOT visible
    try std.testing.expect(!isTupleVisible(h, snap, 8, 4));

    // At cid=3, the delete at cid=3 is NOT visible (cid < current_cid required)
    // cid(3) < current_cid(3) is false → xmax not visible → tuple IS visible
    try std.testing.expect(isTupleVisible(h, snap, 8, 3));
}

test "isTupleVisible — own txn insert at same cid invisible" {
    const snap = Snapshot{
        .xmin = 5,
        .xmax = 10,
        .active_xids = &.{},
        .allocator = null,
    };

    // Tuple inserted by own txn at cid=2, observing at cid=2
    // cid(2) >= current_cid(2) → xmin NOT visible → invisible
    const h = TupleHeader{
        .xmin = 8, // our own xid
        .xmax = INVALID_XID,
        .cid = 2,
        .flags = .{},
    };
    try std.testing.expect(!isTupleVisible(h, snap, 8, 2));

    // At cid=3, the insert at cid=2 IS visible
    try std.testing.expect(isTupleVisible(h, snap, 8, 3));
}

test "TransactionManager — REPEATABLE READ snapshot stability across commits" {
    const allocator = std.testing.allocator;
    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    // xid1 starts first
    const xid1 = try tm.begin(.read_committed);

    // xid2 starts with REPEATABLE READ
    const xid2 = try tm.begin(.repeatable_read);

    // xid3 starts after xid2's snapshot
    const xid3 = try tm.begin(.read_committed);

    // xid1 commits
    try tm.commit(xid1);

    // xid2's snapshot should still see xid1 as active (snapshot taken at begin)
    const snap = try tm.getSnapshot(xid2);
    try std.testing.expect(snap.isActive(xid1)); // Still in original snapshot

    // xid3 is also active in xid2's snapshot since it started before snapshot
    try std.testing.expect(snap.isActive(xid3));

    // Calling getSnapshot again returns the SAME snapshot (no re-allocation)
    const snap2 = try tm.getSnapshot(xid2);
    try std.testing.expectEqual(snap.xmin, snap2.xmin);
    try std.testing.expectEqual(snap.xmax, snap2.xmax);

    try tm.commit(xid3);
    try tm.commit(xid2);
}

test "TransactionManager — READ COMMITTED fresh snapshot per statement" {
    const allocator = std.testing.allocator;
    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    const xid1 = try tm.begin(.read_committed);
    const xid2 = try tm.begin(.read_committed);

    // First snapshot: xid1 is active
    var snap1 = try tm.getSnapshot(xid2);
    defer snap1.deinit();
    try std.testing.expect(snap1.isActive(xid1));

    // Commit xid1
    try tm.commit(xid1);

    // Second snapshot: xid1 is NO LONGER active (fresh snapshot)
    var snap2 = try tm.getSnapshot(xid2);
    defer snap2.deinit();
    try std.testing.expect(!snap2.isActive(xid1));

    try tm.commit(xid2);
}

test "TransactionManager — multiple concurrent transactions snapshot" {
    const allocator = std.testing.allocator;
    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    // Start 5 concurrent transactions
    var xids: [5]u32 = undefined;
    for (&xids) |*xid| {
        xid.* = try tm.begin(.read_committed);
    }

    // Take snapshot from xid[0]
    var snap = try tm.getSnapshot(xids[0]);
    defer snap.deinit();

    // All 5 should be active
    for (xids) |xid| {
        try std.testing.expect(snap.isActive(xid));
    }
    try std.testing.expectEqual(@as(usize, 5), snap.active_xids.len);

    // Commit all
    for (xids) |xid| {
        try tm.commit(xid);
    }
}

test "TransactionManager — commit then abort sequence" {
    const allocator = std.testing.allocator;
    var tm = TransactionManager.init(allocator);
    defer tm.deinit();

    const xid = try tm.begin(.read_committed);
    try tm.commit(xid);

    // Double-commit should fail
    try std.testing.expectError(error.TransactionNotActive, tm.commit(xid));

    // Abort after commit should also fail
    try std.testing.expectError(error.TransactionNotActive, tm.abort(xid));

    // Abort then commit
    const xid2 = try tm.begin(.read_committed);
    try tm.abort(xid2);
    try std.testing.expectError(error.TransactionNotActive, tm.commit(xid2));
}

test "Snapshot.isVisible — special XIDs" {
    const snap = Snapshot{
        .xmin = 5,
        .xmax = 10,
        .active_xids = &.{},
        .allocator = null,
    };

    // FROZEN_XID always visible
    try std.testing.expect(snap.isVisible(FROZEN_XID));

    // INVALID_XID never visible
    try std.testing.expect(!snap.isVisible(INVALID_XID));

    // Committed XID below xmin → visible
    try std.testing.expect(snap.isVisible(3));

    // XID at or above xmax → not visible (future)
    try std.testing.expect(!snap.isVisible(10));
    try std.testing.expect(!snap.isVisible(11));

    // XID in [xmin, xmax) and NOT in active list → visible (committed)
    try std.testing.expect(snap.isVisible(7));
}

test "Snapshot.EMPTY — sees nothing as active" {
    const snap = Snapshot.EMPTY;
    // No transactions are active in empty snapshot
    try std.testing.expect(!snap.isActive(0));
    try std.testing.expect(!snap.isActive(1));
    try std.testing.expect(snap.isActive(FIRST_NORMAL_XID)); // >= xmax
    try std.testing.expect(snap.isActive(100));
}
