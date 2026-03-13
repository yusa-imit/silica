// Replication Slot Management for Silica
//
// Manages replication slots that track WAL consumption by replicas.
// Slots prevent WAL recycling until all replicas have received the data.

const std = @import("std");
const Allocator = std.mem.Allocator;
const protocol = @import("protocol.zig");
const LSN = protocol.LSN;
const ReplicationSlot = protocol.ReplicationSlot;
const SlotState = protocol.SlotState;

/// Errors for slot operations
pub const SlotError = error{
    SlotAlreadyExists,
    SlotNotFound,
    SlotInUse,
    InvalidSlotName,
    OutOfMemory,
    SerializationError,
};

/// Slot Manager — manages replication slot lifecycle and persistence
pub const SlotManager = struct {
    allocator: Allocator,
    /// In-memory slot cache (slot_name -> ReplicationSlot)
    slots: std.StringHashMap(ReplicationSlot),
    /// Mutex for concurrent access
    mutex: std.Thread.Mutex,

    pub fn init(allocator: Allocator) SlotManager {
        return .{
            .allocator = allocator,
            .slots = std.StringHashMap(ReplicationSlot).init(allocator),
            .mutex = .{},
        };
    }

    pub fn deinit(self: *SlotManager) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        var it = self.slots.iterator();
        while (it.next()) |entry| {
            var slot = entry.value_ptr.*;
            slot.deinit();
        }
        self.slots.deinit();
    }

    /// Create a new replication slot
    pub fn createSlot(self: *SlotManager, name: []const u8, temporary: bool) SlotError!void {
        if (name.len == 0 or name.len > 255) {
            return SlotError.InvalidSlotName;
        }

        self.mutex.lock();
        defer self.mutex.unlock();

        // Check for duplicate
        if (self.slots.contains(name)) {
            return SlotError.SlotAlreadyExists;
        }

        // Create slot
        const slot = ReplicationSlot.init(self.allocator, name, temporary) catch |err| {
            return if (err == error.OutOfMemory) SlotError.OutOfMemory else SlotError.SerializationError;
        };

        // Store in cache
        self.slots.put(slot.name, slot) catch |err| {
            var mut_slot = slot;
            mut_slot.deinit();
            return if (err == error.OutOfMemory) SlotError.OutOfMemory else SlotError.SerializationError;
        };
    }

    /// Get a replication slot by name
    pub fn getSlot(self: *SlotManager, name: []const u8) SlotError!ReplicationSlot {
        self.mutex.lock();
        defer self.mutex.unlock();

        const slot = self.slots.get(name) orelse return SlotError.SlotNotFound;
        return slot;
    }

    /// Drop a replication slot
    pub fn dropSlot(self: *SlotManager, name: []const u8) SlotError!void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const kv = self.slots.fetchRemove(name) orelse return SlotError.SlotNotFound;
        var slot = kv.value;

        // Cannot drop active slots
        if (slot.state == .active) {
            // Put back
            self.slots.put(slot.name, slot) catch unreachable; // already had capacity
            return SlotError.SlotInUse;
        }

        slot.deinit();
    }

    /// Check if slot exists
    pub fn slotExists(self: *SlotManager, name: []const u8) bool {
        self.mutex.lock();
        defer self.mutex.unlock();

        return self.slots.contains(name);
    }

    /// List all slot names
    pub fn listSlots(self: *SlotManager, allocator: Allocator) ![][]const u8 {
        self.mutex.lock();
        defer self.mutex.unlock();

        var names = std.ArrayList([]const u8){};
        errdefer names.deinit(allocator);

        var it = self.slots.keyIterator();
        while (it.next()) |name_ptr| {
            const name_copy = try allocator.dupe(u8, name_ptr.*);
            errdefer allocator.free(name_copy);
            try names.append(allocator, name_copy);
        }

        return names.toOwnedSlice(allocator);
    }

    /// Activate a slot (mark as in use)
    pub fn activateSlot(self: *SlotManager, name: []const u8) SlotError!void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const slot_ptr = self.slots.getPtr(name) orelse return SlotError.SlotNotFound;

        if (slot_ptr.state == .active) {
            return SlotError.SlotInUse;
        }

        slot_ptr.state = .active;
        slot_ptr.active_since = std.time.microTimestamp();
    }

    /// Deactivate a slot (mark as inactive)
    pub fn deactivateSlot(self: *SlotManager, name: []const u8) SlotError!void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const slot_ptr = self.slots.getPtr(name) orelse return SlotError.SlotNotFound;
        slot_ptr.state = .inactive;
        slot_ptr.active_since = null;
    }

    /// Update slot LSN positions
    pub fn updateSlotLSN(
        self: *SlotManager,
        name: []const u8,
        restart_lsn: ?LSN,
        confirmed_flush_lsn: ?LSN,
    ) SlotError!void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const slot_ptr = self.slots.getPtr(name) orelse return SlotError.SlotNotFound;

        if (restart_lsn) |lsn| {
            slot_ptr.restart_lsn = lsn;
        }
        if (confirmed_flush_lsn) |lsn| {
            slot_ptr.confirmed_flush_lsn = lsn;
        }
    }

    /// Get the minimum restart LSN across all slots
    /// Returns null if no slots exist
    pub fn getMinRestartLSN(self: *SlotManager) ?LSN {
        self.mutex.lock();
        defer self.mutex.unlock();

        var min_lsn: ?LSN = null;
        var it = self.slots.valueIterator();
        while (it.next()) |slot| {
            if (min_lsn) |current_min| {
                if (slot.restart_lsn < current_min) {
                    min_lsn = slot.restart_lsn;
                }
            } else {
                min_lsn = slot.restart_lsn;
            }
        }
        return min_lsn;
    }

    /// Drop all temporary slots (called on connection close)
    pub fn dropTemporarySlots(self: *SlotManager) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        var to_remove = std.ArrayList([]const u8){};
        defer to_remove.deinit(self.allocator);

        var it = self.slots.iterator();
        while (it.next()) |entry| {
            if (entry.value_ptr.temporary) {
                to_remove.append(self.allocator, entry.key_ptr.*) catch continue; // best effort
            }
        }

        for (to_remove.items) |name| {
            const kv = self.slots.fetchRemove(name) orelse continue;
            var slot = kv.value;
            slot.deinit();
        }
    }
};

// ============================================================================
// Tests
// ============================================================================

test "SlotManager: create and get slot" {
    const allocator = std.testing.allocator;
    var manager = SlotManager.init(allocator);
    defer manager.deinit();

    // Create slot
    try manager.createSlot("slot1", false);

    // Get slot
    const slot = try manager.getSlot("slot1");
    try std.testing.expectEqualStrings("slot1", slot.name);
    try std.testing.expectEqual(SlotState.inactive, slot.state);
    try std.testing.expectEqual(@as(LSN, 0), slot.restart_lsn);
    try std.testing.expectEqual(false, slot.temporary);
}

test "SlotManager: duplicate slot error" {
    const allocator = std.testing.allocator;
    var manager = SlotManager.init(allocator);
    defer manager.deinit();

    try manager.createSlot("slot1", false);

    // Duplicate creation should fail
    try std.testing.expectError(SlotError.SlotAlreadyExists, manager.createSlot("slot1", false));
}

test "SlotManager: drop slot" {
    const allocator = std.testing.allocator;
    var manager = SlotManager.init(allocator);
    defer manager.deinit();

    try manager.createSlot("slot1", false);
    try manager.dropSlot("slot1");

    // Should not exist
    try std.testing.expectError(SlotError.SlotNotFound, manager.getSlot("slot1"));
}

test "SlotManager: cannot drop active slot" {
    const allocator = std.testing.allocator;
    var manager = SlotManager.init(allocator);
    defer manager.deinit();

    try manager.createSlot("slot1", false);
    try manager.activateSlot("slot1");

    // Cannot drop active slot
    try std.testing.expectError(SlotError.SlotInUse, manager.dropSlot("slot1"));

    // Can drop after deactivation
    try manager.deactivateSlot("slot1");
    try manager.dropSlot("slot1");
}

test "SlotManager: slot activation" {
    const allocator = std.testing.allocator;
    var manager = SlotManager.init(allocator);
    defer manager.deinit();

    try manager.createSlot("slot1", false);

    // Activate
    try manager.activateSlot("slot1");
    var slot = try manager.getSlot("slot1");
    try std.testing.expectEqual(SlotState.active, slot.state);
    try std.testing.expect(slot.active_since != null);

    // Cannot activate again
    try std.testing.expectError(SlotError.SlotInUse, manager.activateSlot("slot1"));

    // Deactivate
    try manager.deactivateSlot("slot1");
    slot = try manager.getSlot("slot1");
    try std.testing.expectEqual(SlotState.inactive, slot.state);
    try std.testing.expect(slot.active_since == null);
}

test "SlotManager: update LSN" {
    const allocator = std.testing.allocator;
    var manager = SlotManager.init(allocator);
    defer manager.deinit();

    try manager.createSlot("slot1", false);

    // Update restart LSN
    try manager.updateSlotLSN("slot1", 1000, null);
    var slot = try manager.getSlot("slot1");
    try std.testing.expectEqual(@as(LSN, 1000), slot.restart_lsn);
    try std.testing.expectEqual(@as(LSN, 0), slot.confirmed_flush_lsn);

    // Update confirmed flush LSN
    try manager.updateSlotLSN("slot1", null, 2000);
    slot = try manager.getSlot("slot1");
    try std.testing.expectEqual(@as(LSN, 1000), slot.restart_lsn);
    try std.testing.expectEqual(@as(LSN, 2000), slot.confirmed_flush_lsn);

    // Update both
    try manager.updateSlotLSN("slot1", 3000, 4000);
    slot = try manager.getSlot("slot1");
    try std.testing.expectEqual(@as(LSN, 3000), slot.restart_lsn);
    try std.testing.expectEqual(@as(LSN, 4000), slot.confirmed_flush_lsn);
}

test "SlotManager: get min restart LSN" {
    const allocator = std.testing.allocator;
    var manager = SlotManager.init(allocator);
    defer manager.deinit();

    // No slots = null
    try std.testing.expectEqual(@as(?LSN, null), manager.getMinRestartLSN());

    // Create slots with different LSNs
    try manager.createSlot("slot1", false);
    try manager.updateSlotLSN("slot1", 1000, null);

    try manager.createSlot("slot2", false);
    try manager.updateSlotLSN("slot2", 500, null);

    try manager.createSlot("slot3", false);
    try manager.updateSlotLSN("slot3", 2000, null);

    // Min should be 500
    try std.testing.expectEqual(@as(?LSN, 500), manager.getMinRestartLSN());
}

test "SlotManager: list slots" {
    const allocator = std.testing.allocator;
    var manager = SlotManager.init(allocator);
    defer manager.deinit();

    try manager.createSlot("slot1", false);
    try manager.createSlot("slot2", false);
    try manager.createSlot("slot3", false);

    const names = try manager.listSlots(allocator);
    defer {
        for (names) |name| {
            allocator.free(name);
        }
        allocator.free(names);
    }

    try std.testing.expectEqual(@as(usize, 3), names.len);

    // Check all names are present (order undefined)
    var found_slot1 = false;
    var found_slot2 = false;
    var found_slot3 = false;
    for (names) |name| {
        if (std.mem.eql(u8, name, "slot1")) found_slot1 = true;
        if (std.mem.eql(u8, name, "slot2")) found_slot2 = true;
        if (std.mem.eql(u8, name, "slot3")) found_slot3 = true;
    }
    try std.testing.expect(found_slot1);
    try std.testing.expect(found_slot2);
    try std.testing.expect(found_slot3);
}

test "SlotManager: drop temporary slots" {
    const allocator = std.testing.allocator;
    var manager = SlotManager.init(allocator);
    defer manager.deinit();

    try manager.createSlot("permanent1", false);
    try manager.createSlot("temp1", true);
    try manager.createSlot("permanent2", false);
    try manager.createSlot("temp2", true);

    manager.dropTemporarySlots();

    // Permanent slots should still exist
    _ = try manager.getSlot("permanent1");
    _ = try manager.getSlot("permanent2");

    // Temporary slots should be gone
    try std.testing.expectError(SlotError.SlotNotFound, manager.getSlot("temp1"));
    try std.testing.expectError(SlotError.SlotNotFound, manager.getSlot("temp2"));
}

test "SlotManager: slot exists" {
    const allocator = std.testing.allocator;
    var manager = SlotManager.init(allocator);
    defer manager.deinit();

    try std.testing.expect(!manager.slotExists("slot1"));
    try manager.createSlot("slot1", false);
    try std.testing.expect(manager.slotExists("slot1"));
    try manager.dropSlot("slot1");
    try std.testing.expect(!manager.slotExists("slot1"));
}

test "SlotManager: invalid slot name" {
    const allocator = std.testing.allocator;
    var manager = SlotManager.init(allocator);
    defer manager.deinit();

    // Empty name
    try std.testing.expectError(SlotError.InvalidSlotName, manager.createSlot("", false));

    // Name too long (> 255 chars)
    const long_name = "a" ** 256;
    try std.testing.expectError(SlotError.InvalidSlotName, manager.createSlot(long_name, false));
}

test "SlotManager: slot not found errors" {
    const allocator = std.testing.allocator;
    var manager = SlotManager.init(allocator);
    defer manager.deinit();

    try std.testing.expectError(SlotError.SlotNotFound, manager.getSlot("nonexistent"));
    try std.testing.expectError(SlotError.SlotNotFound, manager.dropSlot("nonexistent"));
    try std.testing.expectError(SlotError.SlotNotFound, manager.activateSlot("nonexistent"));
    try std.testing.expectError(SlotError.SlotNotFound, manager.deactivateSlot("nonexistent"));
    try std.testing.expectError(SlotError.SlotNotFound, manager.updateSlotLSN("nonexistent", 100, 200));
}

// ============================================================================
// Edge Case Tests
// ============================================================================

test "SlotManager: concurrent slot operations" {
    const allocator = std.testing.allocator;
    var manager = SlotManager.init(allocator);
    defer manager.deinit();

    // Create multiple slots
    try manager.createSlot("slot1", false);
    try manager.createSlot("slot2", false);
    try manager.createSlot("slot3", false);

    // Activate all
    try manager.activateSlot("slot1");
    try manager.activateSlot("slot2");
    try manager.activateSlot("slot3");

    // Update LSNs
    try manager.updateSlotLSN("slot1", 100, 200);
    try manager.updateSlotLSN("slot2", 150, 250);
    try manager.updateSlotLSN("slot3", 50, 100);

    // Verify min LSN
    try std.testing.expectEqual(@as(?LSN, 50), manager.getMinRestartLSN());

    // Deactivate and drop
    try manager.deactivateSlot("slot3");
    try manager.dropSlot("slot3");

    // Min LSN should now be 100
    try std.testing.expectEqual(@as(?LSN, 100), manager.getMinRestartLSN());
}

test "SlotManager: LSN overflow" {
    const allocator = std.testing.allocator;
    var manager = SlotManager.init(allocator);
    defer manager.deinit();

    try manager.createSlot("slot1", false);

    // Max LSN value
    const max_lsn: LSN = std.math.maxInt(LSN);
    try manager.updateSlotLSN("slot1", max_lsn, max_lsn);

    const slot = try manager.getSlot("slot1");
    try std.testing.expectEqual(max_lsn, slot.restart_lsn);
    try std.testing.expectEqual(max_lsn, slot.confirmed_flush_lsn);
}

test "SlotManager: empty slot list" {
    const allocator = std.testing.allocator;
    var manager = SlotManager.init(allocator);
    defer manager.deinit();

    const names = try manager.listSlots(allocator);
    defer allocator.free(names);

    try std.testing.expectEqual(@as(usize, 0), names.len);
}

test "SlotManager: drop all slots including temporary" {
    const allocator = std.testing.allocator;
    var manager = SlotManager.init(allocator);
    defer manager.deinit();

    try manager.createSlot("perm1", false);
    try manager.createSlot("temp1", true);
    try manager.createSlot("perm2", false);
    try manager.createSlot("temp2", true);
    try manager.createSlot("temp3", true);

    // Drop temporary slots
    manager.dropTemporarySlots();

    // Should have 2 permanent slots remaining
    const names = try manager.listSlots(allocator);
    defer {
        for (names) |name| allocator.free(name);
        allocator.free(names);
    }

    try std.testing.expectEqual(@as(usize, 2), names.len);
}

test "SlotManager: special characters in slot name" {
    const allocator = std.testing.allocator;
    var manager = SlotManager.init(allocator);
    defer manager.deinit();

    // Valid special chars
    try manager.createSlot("slot-with-dash", false);
    try manager.createSlot("slot_with_underscore", false);
    try manager.createSlot("slot.with.dot", false);

    try std.testing.expect(manager.slotExists("slot-with-dash"));
    try std.testing.expect(manager.slotExists("slot_with_underscore"));
    try std.testing.expect(manager.slotExists("slot.with.dot"));
}

test "SlotManager: update only restart LSN" {
    const allocator = std.testing.allocator;
    var manager = SlotManager.init(allocator);
    defer manager.deinit();

    try manager.createSlot("slot1", false);
    try manager.updateSlotLSN("slot1", 1000, 2000);

    // Update only restart LSN
    try manager.updateSlotLSN("slot1", 3000, null);

    const slot = try manager.getSlot("slot1");
    try std.testing.expectEqual(@as(LSN, 3000), slot.restart_lsn);
    try std.testing.expectEqual(@as(LSN, 2000), slot.confirmed_flush_lsn);
}

test "SlotManager: update only confirmed flush LSN" {
    const allocator = std.testing.allocator;
    var manager = SlotManager.init(allocator);
    defer manager.deinit();

    try manager.createSlot("slot1", false);
    try manager.updateSlotLSN("slot1", 1000, 2000);

    // Update only confirmed flush LSN
    try manager.updateSlotLSN("slot1", null, 4000);

    const slot = try manager.getSlot("slot1");
    try std.testing.expectEqual(@as(LSN, 1000), slot.restart_lsn);
    try std.testing.expectEqual(@as(LSN, 4000), slot.confirmed_flush_lsn);
}

test "SlotManager: many slots" {
    const allocator = std.testing.allocator;
    var manager = SlotManager.init(allocator);
    defer manager.deinit();

    const num_slots = 100;
    var i: usize = 0;
    while (i < num_slots) : (i += 1) {
        var buf: [32]u8 = undefined;
        const name = try std.fmt.bufPrint(&buf, "slot{d}", .{i});
        try manager.createSlot(name, false);
        try manager.updateSlotLSN(name, @intCast(i * 100), null);
    }

    // Min LSN should be 0 (slot0)
    try std.testing.expectEqual(@as(?LSN, 0), manager.getMinRestartLSN());

    // List should have all slots
    const names = try manager.listSlots(allocator);
    defer {
        for (names) |name| allocator.free(name);
        allocator.free(names);
    }
    try std.testing.expectEqual(num_slots, names.len);
}

test "SlotManager: activate already active slot" {
    const allocator = std.testing.allocator;
    var manager = SlotManager.init(allocator);
    defer manager.deinit();

    try manager.createSlot("slot1", false);
    try manager.activateSlot("slot1");

    // Activating again should fail
    try std.testing.expectError(SlotError.SlotInUse, manager.activateSlot("slot1"));
}

test "SlotManager: deactivate inactive slot" {
    const allocator = std.testing.allocator;
    var manager = SlotManager.init(allocator);
    defer manager.deinit();

    try manager.createSlot("slot1", false);

    // Deactivating inactive slot should succeed (idempotent)
    try manager.deactivateSlot("slot1");

    const slot = try manager.getSlot("slot1");
    try std.testing.expectEqual(SlotState.inactive, slot.state);
}
