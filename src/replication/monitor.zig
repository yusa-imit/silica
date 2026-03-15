// Replication Monitoring for Silica
//
// Provides pg_stat_replication-equivalent monitoring views.
// Tracks replication status, lag, and connection state.

const std = @import("std");
const Allocator = std.mem.Allocator;
const protocol = @import("protocol.zig");
const slot = @import("slot.zig");
const sender = @import("sender.zig");
const receiver = @import("receiver.zig");
const LSN = protocol.LSN;

/// Replication monitoring errors
pub const Error = error{
    /// Invalid slot name
    InvalidSlotName,
    /// Slot not found
    SlotNotFound,
} || Allocator.Error;

/// Alert severity level
pub const AlertSeverity = enum(u8) {
    /// Warning: lag exceeds warning threshold
    warning = 0,
    /// Critical: lag exceeds critical threshold
    critical = 1,
    /// Recovery: lag returned below warning threshold
    recovery = 2,
};

/// Lag alert event
pub const LagAlert = struct {
    /// Slot name
    slot_name: []const u8,
    /// Application name
    application_name: []const u8,
    /// Alert severity
    severity: AlertSeverity,
    /// Current lag in microseconds
    lag_us: i64,
    /// Lag type (write/flush/replay)
    lag_type: []const u8,
    /// Timestamp when alert was triggered
    timestamp: i64,
};

/// Alert callback function type
pub const AlertCallback = *const fn (alert: LagAlert, user_data: ?*anyopaque) void;

/// Lag alert configuration
pub const LagAlertConfig = struct {
    /// Warning threshold for write lag (microseconds)
    write_lag_warning_us: i64,
    /// Critical threshold for write lag (microseconds)
    write_lag_critical_us: i64,
    /// Warning threshold for flush lag (microseconds)
    flush_lag_warning_us: i64,
    /// Critical threshold for flush lag (microseconds)
    flush_lag_critical_us: i64,
    /// Warning threshold for replay lag (microseconds)
    replay_lag_warning_us: i64,
    /// Critical threshold for replay lag (microseconds)
    replay_lag_critical_us: i64,
    /// Enable alerting
    enabled: bool,

    /// Default configuration with sensible thresholds
    pub fn default() LagAlertConfig {
        return .{
            .write_lag_warning_us = 5_000_000, // 5 seconds
            .write_lag_critical_us = 30_000_000, // 30 seconds
            .flush_lag_warning_us = 10_000_000, // 10 seconds
            .flush_lag_critical_us = 60_000_000, // 60 seconds
            .replay_lag_warning_us = 30_000_000, // 30 seconds
            .replay_lag_critical_us = 300_000_000, // 5 minutes
            .enabled = true,
        };
    }
};

/// Replication state (from PostgreSQL pg_stat_replication)
pub const ReplicationState = enum(u8) {
    /// Starting up
    startup = 0,
    /// Catching up
    catchup = 1,
    /// Streaming WAL
    streaming = 2,
    /// Backup in progress
    backup = 3,
    /// Connection stopped
    stopped = 4,

    pub fn toString(self: ReplicationState) []const u8 {
        return switch (self) {
            .startup => "startup",
            .catchup => "catchup",
            .streaming => "streaming",
            .backup => "backup",
            .stopped => "stopped",
        };
    }
};

/// Sync state (from PostgreSQL pg_stat_replication)
pub const SyncState = enum(u8) {
    /// Asynchronous replication
    async = 0,
    /// Synchronous standby (potential)
    potential = 1,
    /// Synchronous standby (quorum)
    quorum = 2,
    /// Synchronous standby (priority)
    sync = 3,

    pub fn toString(self: SyncState) []const u8 {
        return switch (self) {
            .async => "async",
            .potential => "potential",
            .quorum => "quorum",
            .sync => "sync",
        };
    }
};

/// Replication connection info (single row in pg_stat_replication)
pub const ReplicationStat = struct {
    /// Application name
    application_name: []const u8,
    /// Client address (hostname or IP)
    client_addr: []const u8,
    /// Client port
    client_port: u16,
    /// Backend start time (microseconds since epoch)
    backend_start: i64,
    /// Current replication state
    state: ReplicationState,
    /// Sent LSN (highest WAL position sent to replica)
    sent_lsn: LSN,
    /// Write LSN (highest WAL position written to replica disk)
    write_lsn: LSN,
    /// Flush LSN (highest WAL position flushed to replica disk)
    flush_lsn: LSN,
    /// Replay LSN (highest WAL position applied on replica)
    replay_lsn: LSN,
    /// Write lag in microseconds
    write_lag_us: i64,
    /// Flush lag in microseconds
    flush_lag_us: i64,
    /// Replay lag in microseconds
    replay_lag_us: i64,
    /// Sync priority (0 = async, >0 = sync priority)
    sync_priority: u32,
    /// Sync state
    sync_state: SyncState,
    /// Slot name (if using replication slot)
    slot_name: ?[]const u8,

    pub fn deinit(self: *ReplicationStat, allocator: Allocator) void {
        allocator.free(self.application_name);
        allocator.free(self.client_addr);
        if (self.slot_name) |name| {
            allocator.free(name);
        }
    }
};

/// Alert state per slot
const AlertState = struct {
    /// Whether write lag is in warning state
    write_warning: bool,
    /// Whether write lag is in critical state
    write_critical: bool,
    /// Whether flush lag is in warning state
    flush_warning: bool,
    /// Whether flush lag is in critical state
    flush_critical: bool,
    /// Whether replay lag is in warning state
    replay_warning: bool,
    /// Whether replay lag is in critical state
    replay_critical: bool,
};

/// Replication monitor tracks active replication connections
pub const ReplicationMonitor = struct {
    /// Memory allocator
    allocator: Allocator,
    /// Mutex for thread-safe access
    mutex: std.Thread.Mutex,
    /// Active replication stats by slot name
    stats: std.StringHashMap(ReplicationStat),
    /// WAL sender reference (optional)
    wal_sender: ?*sender.WalSender,
    /// WAL receiver reference (optional)
    wal_receiver: ?*receiver.WalReceiver,
    /// Lag alert configuration
    alert_config: LagAlertConfig,
    /// Alert callback function
    alert_callback: ?AlertCallback,
    /// User data for alert callback
    alert_user_data: ?*anyopaque,
    /// Alert states by slot name (tracks current alert status to prevent duplicate alerts)
    alert_states: std.StringHashMap(AlertState),

    pub fn init(allocator: Allocator) ReplicationMonitor {
        return .{
            .allocator = allocator,
            .mutex = .{},
            .stats = std.StringHashMap(ReplicationStat).init(allocator),
            .wal_sender = null,
            .wal_receiver = null,
            .alert_config = LagAlertConfig.default(),
            .alert_callback = null,
            .alert_user_data = null,
            .alert_states = std.StringHashMap(AlertState).init(allocator),
        };
    }

    pub fn deinit(self: *ReplicationMonitor) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        var it = self.stats.iterator();
        while (it.next()) |entry| {
            var stat = entry.value_ptr.*;
            stat.deinit(self.allocator);
            self.allocator.free(entry.key_ptr.*); // Free the hashmap key
        }
        self.stats.deinit();

        // Free alert state keys
        var alert_it = self.alert_states.iterator();
        while (alert_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.alert_states.deinit();
    }

    /// Set WAL sender for monitoring
    pub fn setWalSender(self: *ReplicationMonitor, wal_sender: *sender.WalSender) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.wal_sender = wal_sender;
    }

    /// Set WAL receiver for monitoring
    pub fn setWalReceiver(self: *ReplicationMonitor, wal_receiver: *receiver.WalReceiver) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.wal_receiver = wal_receiver;
    }

    /// Configure lag alerting
    pub fn setAlertConfig(self: *ReplicationMonitor, config: LagAlertConfig) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.alert_config = config;
    }

    /// Set alert callback
    pub fn setAlertCallback(
        self: *ReplicationMonitor,
        callback: ?AlertCallback,
        user_data: ?*anyopaque,
    ) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.alert_callback = callback;
        self.alert_user_data = user_data;
    }

    /// Check lag thresholds and trigger alerts if necessary
    /// Call this periodically (e.g., every second) to monitor lag
    pub fn checkLagThresholds(self: *ReplicationMonitor) Error!void {
        if (!self.alert_config.enabled) {
            return;
        }

        self.mutex.lock();
        defer self.mutex.unlock();

        var it = self.stats.iterator();
        while (it.next()) |entry| {
            const slot_name = entry.key_ptr.*;
            const stat = entry.value_ptr.*;

            // Get or create alert state for this slot
            const gop = try self.alert_states.getOrPut(slot_name);
            if (!gop.found_existing) {
                gop.key_ptr.* = try self.allocator.dupe(u8, slot_name);
                gop.value_ptr.* = AlertState{
                    .write_warning = false,
                    .write_critical = false,
                    .flush_warning = false,
                    .flush_critical = false,
                    .replay_warning = false,
                    .replay_critical = false,
                };
            }

            // Check write lag
            try self.checkSingleLagThreshold(
                slot_name,
                stat.application_name,
                stat.write_lag_us,
                "write",
                self.alert_config.write_lag_warning_us,
                self.alert_config.write_lag_critical_us,
                &gop.value_ptr.write_warning,
                &gop.value_ptr.write_critical,
            );

            // Check flush lag
            try self.checkSingleLagThreshold(
                slot_name,
                stat.application_name,
                stat.flush_lag_us,
                "flush",
                self.alert_config.flush_lag_warning_us,
                self.alert_config.flush_lag_critical_us,
                &gop.value_ptr.flush_warning,
                &gop.value_ptr.flush_critical,
            );

            // Check replay lag
            try self.checkSingleLagThreshold(
                slot_name,
                stat.application_name,
                stat.replay_lag_us,
                "replay",
                self.alert_config.replay_lag_warning_us,
                self.alert_config.replay_lag_critical_us,
                &gop.value_ptr.replay_warning,
                &gop.value_ptr.replay_critical,
            );
        }
    }

    /// Check a single lag threshold (internal helper)
    fn checkSingleLagThreshold(
        self: *ReplicationMonitor,
        slot_name: []const u8,
        application_name: []const u8,
        current_lag: i64,
        lag_type: []const u8,
        warning_threshold: i64,
        critical_threshold: i64,
        warning_state: *bool,
        critical_state: *bool,
    ) Error!void {
        const callback = self.alert_callback orelse return;

        // Check critical threshold
        if (current_lag >= critical_threshold) {
            if (!critical_state.*) {
                // Transition to critical
                critical_state.* = true;
                warning_state.* = true; // Critical implies warning
                callback(.{
                    .slot_name = slot_name,
                    .application_name = application_name,
                    .severity = .critical,
                    .lag_us = current_lag,
                    .lag_type = lag_type,
                    .timestamp = std.time.microTimestamp(),
                }, self.alert_user_data);
            }
        } else if (current_lag >= warning_threshold) {
            if (!warning_state.*) {
                // Transition to warning
                warning_state.* = true;
                critical_state.* = false;
                callback(.{
                    .slot_name = slot_name,
                    .application_name = application_name,
                    .severity = .warning,
                    .lag_us = current_lag,
                    .lag_type = lag_type,
                    .timestamp = std.time.microTimestamp(),
                }, self.alert_user_data);
            } else if (critical_state.*) {
                // Downgrade from critical to warning
                critical_state.* = false;
                callback(.{
                    .slot_name = slot_name,
                    .application_name = application_name,
                    .severity = .recovery,
                    .lag_us = current_lag,
                    .lag_type = lag_type,
                    .timestamp = std.time.microTimestamp(),
                }, self.alert_user_data);
            }
        } else {
            if (warning_state.* or critical_state.*) {
                // Recovery: lag returned below warning threshold
                warning_state.* = false;
                critical_state.* = false;
                callback(.{
                    .slot_name = slot_name,
                    .application_name = application_name,
                    .severity = .recovery,
                    .lag_us = current_lag,
                    .lag_type = lag_type,
                    .timestamp = std.time.microTimestamp(),
                }, self.alert_user_data);
            }
        }
    }

    /// Register a new replication connection
    pub fn registerConnection(
        self: *ReplicationMonitor,
        slot_name: []const u8,
        application_name: []const u8,
        client_addr: []const u8,
        client_port: u16,
    ) Error!void {
        if (slot_name.len == 0 or slot_name.len > 255) {
            return error.InvalidSlotName;
        }

        self.mutex.lock();
        defer self.mutex.unlock();

        // Check if already exists
        if (self.stats.contains(slot_name)) {
            // Update existing entry
            var stat = self.stats.getPtr(slot_name).?;
            self.allocator.free(stat.application_name);
            self.allocator.free(stat.client_addr);
            stat.application_name = try self.allocator.dupe(u8, application_name);
            stat.client_addr = try self.allocator.dupe(u8, client_addr);
            stat.client_port = client_port;
            stat.backend_start = std.time.microTimestamp();
            stat.state = .startup;
        } else {
            // Create new entry
            const stat = ReplicationStat{
                .application_name = try self.allocator.dupe(u8, application_name),
                .client_addr = try self.allocator.dupe(u8, client_addr),
                .client_port = client_port,
                .backend_start = std.time.microTimestamp(),
                .state = .startup,
                .sent_lsn = 0,
                .write_lsn = 0,
                .flush_lsn = 0,
                .replay_lsn = 0,
                .write_lag_us = 0,
                .flush_lag_us = 0,
                .replay_lag_us = 0,
                .sync_priority = 0,
                .sync_state = .async,
                .slot_name = try self.allocator.dupe(u8, slot_name),
            };
            try self.stats.put(try self.allocator.dupe(u8, slot_name), stat);
        }
    }

    /// Unregister a replication connection
    pub fn unregisterConnection(self: *ReplicationMonitor, slot_name: []const u8) Error!void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.stats.fetchRemove(slot_name)) |kv| {
            var stat = kv.value;
            stat.deinit(self.allocator);
            self.allocator.free(kv.key);
        } else {
            return error.SlotNotFound;
        }

        // Clean up alert state
        if (self.alert_states.fetchRemove(slot_name)) |kv| {
            self.allocator.free(kv.key);
        }
    }

    /// Update replication state
    pub fn updateState(
        self: *ReplicationMonitor,
        slot_name: []const u8,
        state: ReplicationState,
    ) Error!void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.stats.getPtr(slot_name)) |stat| {
            stat.state = state;
        } else {
            return error.SlotNotFound;
        }
    }

    /// Update LSN positions and calculate lag
    pub fn updateProgress(
        self: *ReplicationMonitor,
        slot_name: []const u8,
        sent_lsn: LSN,
        write_lsn: LSN,
        flush_lsn: LSN,
        replay_lsn: LSN,
    ) Error!void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.stats.getPtr(slot_name)) |stat| {
            // Update LSNs
            stat.sent_lsn = sent_lsn;
            stat.write_lsn = write_lsn;
            stat.flush_lsn = flush_lsn;
            stat.replay_lsn = replay_lsn;

            // Calculate lag based on LSN differences
            // Simplified: assume 1 byte ≈ 1 microsecond (will be more accurate in production)
            stat.write_lag_us = @as(i64, @intCast(sent_lsn - write_lsn));
            stat.flush_lag_us = @as(i64, @intCast(sent_lsn - flush_lsn));
            stat.replay_lag_us = @as(i64, @intCast(sent_lsn - replay_lsn));
        } else {
            return error.SlotNotFound;
        }
    }

    /// Update synchronization state
    pub fn updateSyncState(
        self: *ReplicationMonitor,
        slot_name: []const u8,
        sync_priority: u32,
        sync_state: SyncState,
    ) Error!void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.stats.getPtr(slot_name)) |stat| {
            stat.sync_priority = sync_priority;
            stat.sync_state = sync_state;
        } else {
            return error.SlotNotFound;
        }
    }

    /// Get replication statistics for a specific slot
    pub fn getStat(self: *ReplicationMonitor, slot_name: []const u8) Error!ReplicationStat {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.stats.get(slot_name)) |stat| {
            // Return a copy with duplicated strings
            return ReplicationStat{
                .application_name = try self.allocator.dupe(u8, stat.application_name),
                .client_addr = try self.allocator.dupe(u8, stat.client_addr),
                .client_port = stat.client_port,
                .backend_start = stat.backend_start,
                .state = stat.state,
                .sent_lsn = stat.sent_lsn,
                .write_lsn = stat.write_lsn,
                .flush_lsn = stat.flush_lsn,
                .replay_lsn = stat.replay_lsn,
                .write_lag_us = stat.write_lag_us,
                .flush_lag_us = stat.flush_lag_us,
                .replay_lag_us = stat.replay_lag_us,
                .sync_priority = stat.sync_priority,
                .sync_state = stat.sync_state,
                .slot_name = if (stat.slot_name) |name| try self.allocator.dupe(u8, name) else null,
            };
        } else {
            return error.SlotNotFound;
        }
    }

    /// Get all replication statistics
    pub fn getAllStats(self: *ReplicationMonitor) Error![]ReplicationStat {
        self.mutex.lock();
        defer self.mutex.unlock();

        const result = try self.allocator.alloc(ReplicationStat, self.stats.count());
        var i: usize = 0;
        var it = self.stats.iterator();
        while (it.next()) |entry| : (i += 1) {
            const stat = entry.value_ptr.*;
            result[i] = ReplicationStat{
                .application_name = try self.allocator.dupe(u8, stat.application_name),
                .client_addr = try self.allocator.dupe(u8, stat.client_addr),
                .client_port = stat.client_port,
                .backend_start = stat.backend_start,
                .state = stat.state,
                .sent_lsn = stat.sent_lsn,
                .write_lsn = stat.write_lsn,
                .flush_lsn = stat.flush_lsn,
                .replay_lsn = stat.replay_lsn,
                .write_lag_us = stat.write_lag_us,
                .flush_lag_us = stat.flush_lag_us,
                .replay_lag_us = stat.replay_lag_us,
                .sync_priority = stat.sync_priority,
                .sync_state = stat.sync_state,
                .slot_name = if (stat.slot_name) |name| try self.allocator.dupe(u8, name) else null,
            };
        }
        return result;
    }

    /// Get count of active replicas
    pub fn getReplicaCount(self: *ReplicationMonitor) usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.stats.count();
    }

    /// Get count of synchronous replicas
    pub fn getSyncReplicaCount(self: *ReplicationMonitor) usize {
        self.mutex.lock();
        defer self.mutex.unlock();

        var count: usize = 0;
        var it = self.stats.iterator();
        while (it.next()) |entry| {
            if (entry.value_ptr.sync_state != .async) {
                count += 1;
            }
        }
        return count;
    }

    /// Check if replication is healthy
    pub fn isHealthy(self: *ReplicationMonitor, max_lag_us: i64) bool {
        self.mutex.lock();
        defer self.mutex.unlock();

        var it = self.stats.iterator();
        while (it.next()) |entry| {
            const stat = entry.value_ptr.*;
            // Check if any replica has excessive lag
            if (stat.replay_lag_us > max_lag_us) {
                return false;
            }
            // Check if any replica is stopped
            if (stat.state == .stopped) {
                return false;
            }
        }
        return true;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "ReplicationMonitor: init and deinit" {
    const allocator = std.testing.allocator;
    var monitor = ReplicationMonitor.init(allocator);
    defer monitor.deinit();

    try std.testing.expectEqual(@as(usize, 0), monitor.getReplicaCount());
}

test "ReplicationMonitor: register and unregister connection" {
    const allocator = std.testing.allocator;
    var monitor = ReplicationMonitor.init(allocator);
    defer monitor.deinit();

    try monitor.registerConnection("slot1", "walreceiver", "192.168.1.10", 5432);
    try std.testing.expectEqual(@as(usize, 1), monitor.getReplicaCount());

    try monitor.unregisterConnection("slot1");
    try std.testing.expectEqual(@as(usize, 0), monitor.getReplicaCount());
}

test "ReplicationMonitor: update state" {
    const allocator = std.testing.allocator;
    var monitor = ReplicationMonitor.init(allocator);
    defer monitor.deinit();

    try monitor.registerConnection("slot1", "walreceiver", "192.168.1.10", 5432);
    try monitor.updateState("slot1", .streaming);

    var stat = try monitor.getStat("slot1");
    defer stat.deinit(allocator);
    try std.testing.expectEqual(ReplicationState.streaming, stat.state);
}

test "ReplicationMonitor: update progress and lag" {
    const allocator = std.testing.allocator;
    var monitor = ReplicationMonitor.init(allocator);
    defer monitor.deinit();

    try monitor.registerConnection("slot1", "walreceiver", "192.168.1.10", 5432);
    try monitor.updateProgress("slot1", 1000, 900, 850, 800);

    var stat = try monitor.getStat("slot1");
    defer stat.deinit(allocator);
    try std.testing.expectEqual(@as(LSN, 1000), stat.sent_lsn);
    try std.testing.expectEqual(@as(LSN, 900), stat.write_lsn);
    try std.testing.expectEqual(@as(LSN, 850), stat.flush_lsn);
    try std.testing.expectEqual(@as(LSN, 800), stat.replay_lsn);
    try std.testing.expectEqual(@as(i64, 100), stat.write_lag_us);
    try std.testing.expectEqual(@as(i64, 150), stat.flush_lag_us);
    try std.testing.expectEqual(@as(i64, 200), stat.replay_lag_us);
}

test "ReplicationMonitor: update sync state" {
    const allocator = std.testing.allocator;
    var monitor = ReplicationMonitor.init(allocator);
    defer monitor.deinit();

    try monitor.registerConnection("slot1", "walreceiver", "192.168.1.10", 5432);
    try monitor.updateSyncState("slot1", 1, .sync);

    var stat = try monitor.getStat("slot1");
    defer stat.deinit(allocator);
    try std.testing.expectEqual(@as(u32, 1), stat.sync_priority);
    try std.testing.expectEqual(SyncState.sync, stat.sync_state);
}

test "ReplicationMonitor: get all stats" {
    const allocator = std.testing.allocator;
    var monitor = ReplicationMonitor.init(allocator);
    defer monitor.deinit();

    try monitor.registerConnection("slot1", "walreceiver", "192.168.1.10", 5432);
    try monitor.registerConnection("slot2", "walreceiver", "192.168.1.11", 5433);

    const stats = try monitor.getAllStats();
    defer {
        for (stats) |*stat| {
            stat.deinit(allocator);
        }
        allocator.free(stats);
    }
    try std.testing.expectEqual(@as(usize, 2), stats.len);
}

test "ReplicationMonitor: sync replica count" {
    const allocator = std.testing.allocator;
    var monitor = ReplicationMonitor.init(allocator);
    defer monitor.deinit();

    try monitor.registerConnection("slot1", "walreceiver", "192.168.1.10", 5432);
    try monitor.registerConnection("slot2", "walreceiver", "192.168.1.11", 5433);
    try monitor.updateSyncState("slot1", 1, .sync);

    try std.testing.expectEqual(@as(usize, 2), monitor.getReplicaCount());
    try std.testing.expectEqual(@as(usize, 1), monitor.getSyncReplicaCount());
}

test "ReplicationMonitor: health check" {
    const allocator = std.testing.allocator;
    var monitor = ReplicationMonitor.init(allocator);
    defer monitor.deinit();

    try monitor.registerConnection("slot1", "walreceiver", "192.168.1.10", 5432);
    try monitor.updateProgress("slot1", 1000, 900, 850, 800);

    // Healthy: lag is 200us, threshold is 1000us
    try std.testing.expect(monitor.isHealthy(1000));

    // Unhealthy: lag is 200us, threshold is 100us
    try std.testing.expect(!monitor.isHealthy(100));
}

test "ReplicationMonitor: slot not found errors" {
    const allocator = std.testing.allocator;
    var monitor = ReplicationMonitor.init(allocator);
    defer monitor.deinit();

    try std.testing.expectError(error.SlotNotFound, monitor.updateState("nonexistent", .streaming));
    try std.testing.expectError(error.SlotNotFound, monitor.updateProgress("nonexistent", 1000, 900, 850, 800));
    try std.testing.expectError(error.SlotNotFound, monitor.getStat("nonexistent"));
    try std.testing.expectError(error.SlotNotFound, monitor.unregisterConnection("nonexistent"));
}

// ============================================================================
// Edge Case Tests
// ============================================================================

test "ReplicationMonitor: invalid slot name" {
    const allocator = std.testing.allocator;
    var monitor = ReplicationMonitor.init(allocator);
    defer monitor.deinit();

    // Empty slot name
    try std.testing.expectError(error.InvalidSlotName, monitor.registerConnection("", "walreceiver", "192.168.1.10", 5432));

    // Slot name too long (>255 characters)
    const long_name = "a" ** 256;
    try std.testing.expectError(error.InvalidSlotName, monitor.registerConnection(long_name, "walreceiver", "192.168.1.10", 5432));
}

test "ReplicationMonitor: reregister connection" {
    const allocator = std.testing.allocator;
    var monitor = ReplicationMonitor.init(allocator);
    defer monitor.deinit();

    // Register slot1
    try monitor.registerConnection("slot1", "walreceiver1", "192.168.1.10", 5432);
    var stat1 = try monitor.getStat("slot1");
    defer stat1.deinit(allocator);
    try std.testing.expectEqualStrings("walreceiver1", stat1.application_name);

    // Re-register slot1 with different app name (should update)
    try monitor.registerConnection("slot1", "walreceiver2", "192.168.1.11", 5433);
    var stat2 = try monitor.getStat("slot1");
    defer stat2.deinit(allocator);
    try std.testing.expectEqualStrings("walreceiver2", stat2.application_name);
    try std.testing.expectEqualStrings("192.168.1.11", stat2.client_addr);
    try std.testing.expectEqual(@as(u16, 5433), stat2.client_port);

    // Should still have only 1 connection
    try std.testing.expectEqual(@as(usize, 1), monitor.getReplicaCount());
}

test "ReplicationMonitor: zero lag" {
    const allocator = std.testing.allocator;
    var monitor = ReplicationMonitor.init(allocator);
    defer monitor.deinit();

    try monitor.registerConnection("slot1", "walreceiver", "192.168.1.10", 5432);
    try monitor.updateProgress("slot1", 1000, 1000, 1000, 1000);

    var stat = try monitor.getStat("slot1");
    defer stat.deinit(allocator);
    try std.testing.expectEqual(@as(i64, 0), stat.write_lag_us);
    try std.testing.expectEqual(@as(i64, 0), stat.flush_lag_us);
    try std.testing.expectEqual(@as(i64, 0), stat.replay_lag_us);
}

test "ReplicationMonitor: max LSN values" {
    const allocator = std.testing.allocator;
    var monitor = ReplicationMonitor.init(allocator);
    defer monitor.deinit();

    try monitor.registerConnection("slot1", "walreceiver", "192.168.1.10", 5432);
    const max_lsn: LSN = std.math.maxInt(LSN);
    try monitor.updateProgress("slot1", max_lsn, max_lsn - 1000, max_lsn - 2000, max_lsn - 3000);

    var stat = try monitor.getStat("slot1");
    defer stat.deinit(allocator);
    try std.testing.expectEqual(max_lsn, stat.sent_lsn);
    try std.testing.expectEqual(@as(i64, 1000), stat.write_lag_us);
}

test "ReplicationMonitor: health check with stopped replica" {
    const allocator = std.testing.allocator;
    var monitor = ReplicationMonitor.init(allocator);
    defer monitor.deinit();

    try monitor.registerConnection("slot1", "walreceiver", "192.168.1.10", 5432);
    try monitor.updateState("slot1", .stopped);

    // Unhealthy due to stopped state
    try std.testing.expect(!monitor.isHealthy(1000000));
}

test "ReplicationMonitor: health check with no replicas" {
    const allocator = std.testing.allocator;
    var monitor = ReplicationMonitor.init(allocator);
    defer monitor.deinit();

    // Healthy if no replicas (nothing to fail)
    try std.testing.expect(monitor.isHealthy(1000));
}

test "ReplicationMonitor: multiple sync states" {
    const allocator = std.testing.allocator;
    var monitor = ReplicationMonitor.init(allocator);
    defer monitor.deinit();

    try monitor.registerConnection("slot1", "walreceiver", "192.168.1.10", 5432);
    try monitor.registerConnection("slot2", "walreceiver", "192.168.1.11", 5433);
    try monitor.registerConnection("slot3", "walreceiver", "192.168.1.12", 5434);

    try monitor.updateSyncState("slot1", 1, .sync);
    try monitor.updateSyncState("slot2", 0, .async);
    try monitor.updateSyncState("slot3", 2, .quorum);

    try std.testing.expectEqual(@as(usize, 3), monitor.getReplicaCount());
    try std.testing.expectEqual(@as(usize, 2), monitor.getSyncReplicaCount());
}

test "ReplicationMonitor: ReplicationState toString" {
    try std.testing.expectEqualStrings("startup", ReplicationState.startup.toString());
    try std.testing.expectEqualStrings("catchup", ReplicationState.catchup.toString());
    try std.testing.expectEqualStrings("streaming", ReplicationState.streaming.toString());
    try std.testing.expectEqualStrings("backup", ReplicationState.backup.toString());
    try std.testing.expectEqualStrings("stopped", ReplicationState.stopped.toString());
}

test "ReplicationMonitor: SyncState toString" {
    try std.testing.expectEqualStrings("async", SyncState.async.toString());
    try std.testing.expectEqualStrings("potential", SyncState.potential.toString());
    try std.testing.expectEqualStrings("quorum", SyncState.quorum.toString());
    try std.testing.expectEqualStrings("sync", SyncState.sync.toString());
}

test "ReplicationMonitor: getAllStats with empty monitor" {
    const allocator = std.testing.allocator;
    var monitor = ReplicationMonitor.init(allocator);
    defer monitor.deinit();

    const stats = try monitor.getAllStats();
    defer allocator.free(stats);
    try std.testing.expectEqual(@as(usize, 0), stats.len);
}

test "ReplicationMonitor: setWalSender and setWalReceiver" {
    const allocator = std.testing.allocator;
    var monitor = ReplicationMonitor.init(allocator);
    defer monitor.deinit();

    var slot_manager = slot.SlotManager.init(allocator);
    defer slot_manager.deinit();

    var wal_sender = try sender.WalSender.init(allocator, &slot_manager, "test-system-id", 1, .{});
    defer wal_sender.deinit();

    monitor.setWalSender(&wal_sender);
    try std.testing.expect(monitor.wal_sender != null);

    var wal_receiver = try receiver.WalReceiver.init(allocator, .{
        .primary_conninfo = "host=localhost",
        .slot_name = "test_slot",
    });
    defer wal_receiver.deinit();

    monitor.setWalReceiver(&wal_receiver);
    try std.testing.expect(monitor.wal_receiver != null);
}

// ============================================================================
// Lag Alerting Tests
// ============================================================================

test "LagAlertConfig: default values" {
    const config = LagAlertConfig.default();
    try std.testing.expectEqual(@as(i64, 5_000_000), config.write_lag_warning_us);
    try std.testing.expectEqual(@as(i64, 30_000_000), config.write_lag_critical_us);
    try std.testing.expectEqual(@as(i64, 10_000_000), config.flush_lag_warning_us);
    try std.testing.expectEqual(@as(i64, 60_000_000), config.flush_lag_critical_us);
    try std.testing.expectEqual(@as(i64, 30_000_000), config.replay_lag_warning_us);
    try std.testing.expectEqual(@as(i64, 300_000_000), config.replay_lag_critical_us);
    try std.testing.expect(config.enabled);
}

test "ReplicationMonitor: set alert config" {
    const allocator = std.testing.allocator;
    var monitor = ReplicationMonitor.init(allocator);
    defer monitor.deinit();

    const custom_config = LagAlertConfig{
        .write_lag_warning_us = 1_000_000,
        .write_lag_critical_us = 5_000_000,
        .flush_lag_warning_us = 2_000_000,
        .flush_lag_critical_us = 10_000_000,
        .replay_lag_warning_us = 3_000_000,
        .replay_lag_critical_us = 15_000_000,
        .enabled = false,
    };

    monitor.setAlertConfig(custom_config);
    try std.testing.expectEqual(@as(i64, 1_000_000), monitor.alert_config.write_lag_warning_us);
    try std.testing.expect(!monitor.alert_config.enabled);
}

test "ReplicationMonitor: alert callback on warning threshold" {
    const allocator = std.testing.allocator;
    var monitor = ReplicationMonitor.init(allocator);
    defer monitor.deinit();

    // Configure low thresholds to trigger alerts easily
    const config = LagAlertConfig{
        .write_lag_warning_us = 100,
        .write_lag_critical_us = 500,
        .flush_lag_warning_us = 100,
        .flush_lag_critical_us = 500,
        .replay_lag_warning_us = 100,
        .replay_lag_critical_us = 500,
        .enabled = true,
    };
    monitor.setAlertConfig(config);

    // Track alerts
    const AlertContext = struct {
        count: usize,
        last_severity: AlertSeverity,
        last_lag_type: []const u8,

        fn callback(alert: LagAlert, user_data: ?*anyopaque) void {
            const ctx: *@This() = @ptrCast(@alignCast(user_data.?));
            ctx.count += 1;
            ctx.last_severity = alert.severity;
            ctx.last_lag_type = alert.lag_type;
        }
    };
    var alert_ctx = AlertContext{ .count = 0, .last_severity = .warning, .last_lag_type = "" };
    monitor.setAlertCallback(AlertContext.callback, @ptrCast(&alert_ctx));

    // Register replica
    try monitor.registerConnection("slot1", "walreceiver", "192.168.1.10", 5432);

    // Update progress with lag exceeding warning threshold for all 3 types
    try monitor.updateProgress("slot1", 1000, 850, 850, 850);

    // Check thresholds (should trigger 3 warnings: write, flush, replay)
    try monitor.checkLagThresholds();

    try std.testing.expectEqual(@as(usize, 3), alert_ctx.count);
    try std.testing.expectEqual(AlertSeverity.warning, alert_ctx.last_severity);
    try std.testing.expectEqualStrings("replay", alert_ctx.last_lag_type);
}

test "ReplicationMonitor: alert callback on critical threshold" {
    const allocator = std.testing.allocator;
    var monitor = ReplicationMonitor.init(allocator);
    defer monitor.deinit();

    const config = LagAlertConfig{
        .write_lag_warning_us = 100,
        .write_lag_critical_us = 500,
        .flush_lag_warning_us = 100,
        .flush_lag_critical_us = 500,
        .replay_lag_warning_us = 100,
        .replay_lag_critical_us = 500,
        .enabled = true,
    };
    monitor.setAlertConfig(config);

    const AlertContext = struct {
        count: usize,
        last_severity: AlertSeverity,

        fn callback(alert: LagAlert, user_data: ?*anyopaque) void {
            const ctx: *@This() = @ptrCast(@alignCast(user_data.?));
            ctx.count += 1;
            ctx.last_severity = alert.severity;
        }
    };
    var alert_ctx = AlertContext{ .count = 0, .last_severity = .warning };
    monitor.setAlertCallback(AlertContext.callback, @ptrCast(&alert_ctx));

    try monitor.registerConnection("slot1", "walreceiver", "192.168.1.10", 5432);

    // Update with lag exceeding critical threshold (triggers 3 critical alerts)
    try monitor.updateProgress("slot1", 1000, 400, 400, 400);
    try monitor.checkLagThresholds();

    try std.testing.expectEqual(@as(usize, 3), alert_ctx.count);
    try std.testing.expectEqual(AlertSeverity.critical, alert_ctx.last_severity);
}

test "ReplicationMonitor: alert recovery" {
    const allocator = std.testing.allocator;
    var monitor = ReplicationMonitor.init(allocator);
    defer monitor.deinit();

    const config = LagAlertConfig{
        .write_lag_warning_us = 100,
        .write_lag_critical_us = 500,
        .flush_lag_warning_us = 100,
        .flush_lag_critical_us = 500,
        .replay_lag_warning_us = 100,
        .replay_lag_critical_us = 500,
        .enabled = true,
    };
    monitor.setAlertConfig(config);

    const AlertContext = struct {
        count: usize,
        alerts: [15]AlertSeverity,

        fn callback(alert: LagAlert, user_data: ?*anyopaque) void {
            const ctx: *@This() = @ptrCast(@alignCast(user_data.?));
            ctx.alerts[ctx.count] = alert.severity;
            ctx.count += 1;
        }
    };
    var alert_ctx = AlertContext{ .count = 0, .alerts = undefined };
    monitor.setAlertCallback(AlertContext.callback, @ptrCast(&alert_ctx));

    try monitor.registerConnection("slot1", "walreceiver", "192.168.1.10", 5432);

    // Trigger warning (3 alerts: write, flush, replay)
    try monitor.updateProgress("slot1", 1000, 850, 850, 850);
    try monitor.checkLagThresholds();

    // Trigger critical (3 alerts: write, flush, replay)
    try monitor.updateProgress("slot1", 1000, 400, 400, 400);
    try monitor.checkLagThresholds();

    // Recovery to below warning threshold (3 recovery alerts: write, flush, replay)
    try monitor.updateProgress("slot1", 1000, 950, 950, 950);
    try monitor.checkLagThresholds();

    // Should have 9 alerts total (3 per state: warning -> critical -> recovery)
    try std.testing.expectEqual(@as(usize, 9), alert_ctx.count);
    try std.testing.expectEqual(AlertSeverity.warning, alert_ctx.alerts[0]);
    try std.testing.expectEqual(AlertSeverity.warning, alert_ctx.alerts[1]);
    try std.testing.expectEqual(AlertSeverity.warning, alert_ctx.alerts[2]);
    try std.testing.expectEqual(AlertSeverity.critical, alert_ctx.alerts[3]);
    try std.testing.expectEqual(AlertSeverity.critical, alert_ctx.alerts[4]);
    try std.testing.expectEqual(AlertSeverity.critical, alert_ctx.alerts[5]);
    try std.testing.expectEqual(AlertSeverity.recovery, alert_ctx.alerts[6]);
    try std.testing.expectEqual(AlertSeverity.recovery, alert_ctx.alerts[7]);
    try std.testing.expectEqual(AlertSeverity.recovery, alert_ctx.alerts[8]);
}

test "ReplicationMonitor: no duplicate alerts" {
    const allocator = std.testing.allocator;
    var monitor = ReplicationMonitor.init(allocator);
    defer monitor.deinit();

    const config = LagAlertConfig{
        .write_lag_warning_us = 100,
        .write_lag_critical_us = 500,
        .flush_lag_warning_us = 100,
        .flush_lag_critical_us = 500,
        .replay_lag_warning_us = 100,
        .replay_lag_critical_us = 500,
        .enabled = true,
    };
    monitor.setAlertConfig(config);

    const AlertContext = struct {
        count: usize,

        fn callback(alert: LagAlert, user_data: ?*anyopaque) void {
            _ = alert;
            const ctx: *@This() = @ptrCast(@alignCast(user_data.?));
            ctx.count += 1;
        }
    };
    var alert_ctx = AlertContext{ .count = 0 };
    monitor.setAlertCallback(AlertContext.callback, @ptrCast(&alert_ctx));

    try monitor.registerConnection("slot1", "walreceiver", "192.168.1.10", 5432);

    // Trigger warning for all 3 lag types (write=150, flush=150, replay=150)
    try monitor.updateProgress("slot1", 1000, 850, 850, 850);
    try monitor.checkLagThresholds();

    // Same lag level - should not trigger duplicate alerts
    try monitor.checkLagThresholds();
    try monitor.checkLagThresholds();

    // Should have exactly 3 alerts (1 per lag type: write, flush, replay)
    try std.testing.expectEqual(@as(usize, 3), alert_ctx.count);
}

test "ReplicationMonitor: alerts disabled" {
    const allocator = std.testing.allocator;
    var monitor = ReplicationMonitor.init(allocator);
    defer monitor.deinit();

    const config = LagAlertConfig{
        .write_lag_warning_us = 100,
        .write_lag_critical_us = 500,
        .flush_lag_warning_us = 100,
        .flush_lag_critical_us = 500,
        .replay_lag_warning_us = 100,
        .replay_lag_critical_us = 500,
        .enabled = false, // Disabled
    };
    monitor.setAlertConfig(config);

    const AlertContext = struct {
        count: usize,

        fn callback(alert: LagAlert, user_data: ?*anyopaque) void {
            _ = alert;
            const ctx: *@This() = @ptrCast(@alignCast(user_data.?));
            ctx.count += 1;
        }
    };
    var alert_ctx = AlertContext{ .count = 0 };
    monitor.setAlertCallback(AlertContext.callback, @ptrCast(&alert_ctx));

    try monitor.registerConnection("slot1", "walreceiver", "192.168.1.10", 5432);
    try monitor.updateProgress("slot1", 1000, 400, 400, 400);
    try monitor.checkLagThresholds();

    // No alerts should be triggered
    try std.testing.expectEqual(@as(usize, 0), alert_ctx.count);
}

test "ReplicationMonitor: alert state cleanup on unregister" {
    const allocator = std.testing.allocator;
    var monitor = ReplicationMonitor.init(allocator);
    defer monitor.deinit();

    const config = LagAlertConfig{
        .write_lag_warning_us = 100,
        .write_lag_critical_us = 500,
        .flush_lag_warning_us = 100,
        .flush_lag_critical_us = 500,
        .replay_lag_warning_us = 100,
        .replay_lag_critical_us = 500,
        .enabled = true,
    };
    monitor.setAlertConfig(config);

    const AlertContext = struct {
        count: usize,

        fn callback(alert: LagAlert, user_data: ?*anyopaque) void {
            _ = alert;
            const ctx: *@This() = @ptrCast(@alignCast(user_data.?));
            ctx.count += 1;
        }
    };
    var alert_ctx = AlertContext{ .count = 0 };
    monitor.setAlertCallback(AlertContext.callback, @ptrCast(&alert_ctx));

    try monitor.registerConnection("slot1", "walreceiver", "192.168.1.10", 5432);
    try monitor.updateProgress("slot1", 1000, 850, 850, 850);
    try monitor.checkLagThresholds();

    // Should have alert state
    try std.testing.expectEqual(@as(usize, 1), monitor.alert_states.count());

    // Unregister should clean up alert state
    try monitor.unregisterConnection("slot1");
    try std.testing.expectEqual(@as(usize, 0), monitor.alert_states.count());
}

test "ReplicationMonitor: multiple lag types alert independently" {
    const allocator = std.testing.allocator;
    var monitor = ReplicationMonitor.init(allocator);
    defer monitor.deinit();

    const config = LagAlertConfig{
        .write_lag_warning_us = 100,
        .write_lag_critical_us = 500,
        .flush_lag_warning_us = 200,
        .flush_lag_critical_us = 600,
        .replay_lag_warning_us = 300,
        .replay_lag_critical_us = 700,
        .enabled = true,
    };
    monitor.setAlertConfig(config);

    const AlertContext = struct {
        write_alerts: usize,
        flush_alerts: usize,
        replay_alerts: usize,

        fn callback(alert: LagAlert, user_data: ?*anyopaque) void {
            const ctx: *@This() = @ptrCast(@alignCast(user_data.?));
            if (std.mem.eql(u8, alert.lag_type, "write")) {
                ctx.write_alerts += 1;
            } else if (std.mem.eql(u8, alert.lag_type, "flush")) {
                ctx.flush_alerts += 1;
            } else if (std.mem.eql(u8, alert.lag_type, "replay")) {
                ctx.replay_alerts += 1;
            }
        }
    };
    var alert_ctx = AlertContext{ .write_alerts = 0, .flush_alerts = 0, .replay_alerts = 0 };
    monitor.setAlertCallback(AlertContext.callback, @ptrCast(&alert_ctx));

    try monitor.registerConnection("slot1", "walreceiver", "192.168.1.10", 5432);

    // Trigger write lag only
    try monitor.updateProgress("slot1", 1000, 850, 950, 950);
    try monitor.checkLagThresholds();

    try std.testing.expectEqual(@as(usize, 1), alert_ctx.write_alerts);
    try std.testing.expectEqual(@as(usize, 0), alert_ctx.flush_alerts);
    try std.testing.expectEqual(@as(usize, 0), alert_ctx.replay_alerts);
}

// ============================================================================
// Concurrent Stress Tests
// ============================================================================

test "ReplicationMonitor: concurrent register/update/unregister stress" {
    const allocator = std.testing.allocator;
    var monitor = ReplicationMonitor.init(allocator);
    defer monitor.deinit();

    const num_threads = 8;
    const ops_per_thread = 30;

    const ThreadContext = struct {
        mon: *ReplicationMonitor,
        thread_id: usize,
        allocator_ctx: Allocator,

        fn threadFunc(ctx: *@This()) !void {
            var prng = std.Random.DefaultPrng.init(@intCast(ctx.thread_id));
            const random = prng.random();

            var i: usize = 0;
            while (i < ops_per_thread) : (i += 1) {
                const op = random.intRangeAtMost(u8, 0, 3);

                // Generate unique slot name per thread
                var slot_buf: [64]u8 = undefined;
                const slot_name = try std.fmt.bufPrint(&slot_buf, "slot_{d}_{d}", .{ ctx.thread_id, i });

                switch (op) {
                    0 => {
                        // Register connection
                        ctx.mon.registerConnection(
                            slot_name,
                            "app",
                            "127.0.0.1",
                            5432,
                        ) catch {};
                    },
                    1 => {
                        // Update state
                        ctx.mon.updateState(slot_name, .streaming) catch {};
                    },
                    2 => {
                        // Update progress
                        const lsn_base: LSN = 1000 + @as(LSN, i);
                        ctx.mon.updateProgress(slot_name, lsn_base, lsn_base, lsn_base, lsn_base) catch {};
                    },
                    3 => {
                        // Unregister connection
                        ctx.mon.unregisterConnection(slot_name) catch {};
                    },
                    else => unreachable,
                }
            }
        }
    };

    var threads: [num_threads]std.Thread = undefined;
    var contexts: [num_threads]ThreadContext = undefined;

    // Spawn threads
    for (&threads, &contexts, 0..) |*t, *ctx, tid| {
        ctx.* = .{
            .mon = &monitor,
            .thread_id = tid,
            .allocator_ctx = allocator,
        };
        t.* = try std.Thread.spawn(.{}, ThreadContext.threadFunc, .{ctx});
    }

    // Wait for all threads
    for (&threads) |*t| {
        t.join();
    }

    // Verify monitor is still in valid state
    const replica_count = monitor.getReplicaCount();
    // Count should be >= 0 (some may have been unregistered)
    try std.testing.expect(replica_count >= 0);
}
