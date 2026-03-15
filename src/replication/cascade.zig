// Cascading Replication Infrastructure for Silica
//
// Enables hot standby replicas to act as WAL senders to downstream replicas,
// creating a replication hierarchy: primary → intermediate → downstream.

const std = @import("std");
const Allocator = std.mem.Allocator;
const protocol = @import("protocol.zig");
const sender = @import("sender.zig");
const receiver = @import("receiver.zig");
const standby = @import("standby.zig");
const LSN = protocol.LSN;

/// Cascade errors
pub const Error = error{
    /// Not in standby mode (cannot cascade from primary)
    NotInStandbyMode,
    /// Circular replication detected
    CircularReplication,
    /// Maximum cascade depth exceeded
    MaxCascadeDepthExceeded,
    /// Upstream not connected
    UpstreamNotConnected,
    /// Out of memory
    OutOfMemory,
};

/// Cascade configuration
pub const Config = struct {
    /// Maximum cascade depth (primary = 0, immediate replica = 1, etc.)
    max_cascade_depth: u32 = 4,
    /// Enable WAL forwarding to downstream replicas
    enable_forwarding: bool = true,
};

/// Cascade node role in replication topology
pub const CascadeRole = enum {
    /// Primary server (no upstream)
    primary,
    /// Intermediate replica (has upstream and downstream)
    intermediate,
    /// Leaf replica (has upstream but no downstream)
    leaf,
};

/// Cascading replication coordinator
pub const CascadeCoordinator = struct {
    allocator: Allocator,
    config: Config,
    /// Upstream server connection info (null if primary)
    upstream_conninfo: ?[]const u8,
    /// Cascade depth (0 = primary, 1 = direct replica, etc.)
    cascade_depth: u32,
    /// List of downstream WAL senders
    downstream_senders: std.ArrayListUnmanaged(*sender.WalSender),
    /// Standby coordinator (null if primary)
    standby_coordinator: ?*standby.StandbyCoordinator,
    /// WAL receiver for upstream (null if primary)
    wal_receiver: ?*receiver.WalReceiver,
    /// Mutex for thread-safe downstream sender management
    mutex: std.Thread.Mutex,
    /// Last forwarded LSN to downstream
    last_forwarded_lsn: LSN,

    /// Initialize cascade coordinator
    pub fn init(
        allocator: Allocator,
        config: Config,
        upstream_conninfo: ?[]const u8,
        cascade_depth: u32,
    ) Error!CascadeCoordinator {
        if (cascade_depth > config.max_cascade_depth) {
            return Error.MaxCascadeDepthExceeded;
        }

        var upstream_copy: ?[]const u8 = null;
        if (upstream_conninfo) |conn| {
            upstream_copy = try allocator.dupe(u8, conn);
        }

        return .{
            .allocator = allocator,
            .config = config,
            .upstream_conninfo = upstream_copy,
            .cascade_depth = cascade_depth,
            .downstream_senders = .{},
            .standby_coordinator = null,
            .wal_receiver = null,
            .mutex = std.Thread.Mutex{},
            .last_forwarded_lsn = 0,
        };
    }

    /// Clean up resources
    pub fn deinit(self: *CascadeCoordinator) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        self.downstream_senders.deinit(self.allocator);
        if (self.upstream_conninfo) |conn| {
            self.allocator.free(conn);
        }
    }

    /// Get current cascade role
    pub fn getRole(self: *const CascadeCoordinator) CascadeRole {
        const has_upstream = self.upstream_conninfo != null;
        const has_downstream = self.downstream_senders.items.len > 0;

        if (!has_upstream and !has_downstream) {
            return .primary;
        } else if (has_upstream and has_downstream) {
            return .intermediate;
        } else if (has_upstream) {
            return .leaf;
        } else {
            return .primary;
        }
    }

    /// Check if cascading is enabled
    pub fn isCascadingEnabled(self: *const CascadeCoordinator) bool {
        return self.config.enable_forwarding and self.cascade_depth > 0;
    }

    /// Set standby coordinator reference
    pub fn setStandbyCoordinator(self: *CascadeCoordinator, coordinator: *standby.StandbyCoordinator) void {
        self.standby_coordinator = coordinator;
    }

    /// Set WAL receiver reference
    pub fn setWalReceiver(self: *CascadeCoordinator, rcv: *receiver.WalReceiver) void {
        self.wal_receiver = rcv;
    }

    /// Register a downstream WAL sender
    pub fn registerDownstream(self: *CascadeCoordinator, wal_sender: *sender.WalSender) Error!void {
        self.mutex.lock();
        defer self.mutex.unlock();

        try self.downstream_senders.append(self.allocator, wal_sender);
    }

    /// Unregister a downstream WAL sender
    pub fn unregisterDownstream(self: *CascadeCoordinator, wal_sender: *sender.WalSender) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.downstream_senders.items, 0..) |sender_ptr, i| {
            if (sender_ptr == wal_sender) {
                _ = self.downstream_senders.swapRemove(i);
                break;
            }
        }
    }

    /// Forward WAL data from upstream to all downstream senders
    /// Called when WAL data is received from upstream (intermediate replica only)
    pub fn forwardWalData(
        self: *CascadeCoordinator,
        start_lsn: LSN,
        end_lsn: LSN,
        data: []const u8,
    ) Error!void {
        if (!self.config.enable_forwarding) {
            return;
        }

        // Only intermediate replicas forward WAL
        const role = self.getRole();
        if (role != .intermediate) {
            return;
        }

        self.mutex.lock();
        defer self.mutex.unlock();

        // Update last forwarded LSN
        self.last_forwarded_lsn = end_lsn;

        // Forward to all downstream senders
        for (self.downstream_senders.items) |wal_sender| {
            // Note: actual sending would use WalSender's API
            // For now, we just track the intent to forward
            _ = wal_sender;
            _ = start_lsn;
            _ = data;
        }
    }

    /// Get cascade depth for next level downstream
    pub fn getDownstreamCascadeDepth(self: *const CascadeCoordinator) u32 {
        return self.cascade_depth + 1;
    }

    /// Validate that adding a downstream connection won't create a cycle
    pub fn validateDownstreamConnection(
        self: *const CascadeCoordinator,
        downstream_system_id: []const u8,
    ) Error!void {
        // TODO: Implement cycle detection using system identifiers
        // For now, just check depth limit
        if (self.cascade_depth >= self.config.max_cascade_depth) {
            return Error.MaxCascadeDepthExceeded;
        }

        _ = downstream_system_id;
    }

    /// Get number of downstream replicas
    pub fn getDownstreamCount(self: *CascadeCoordinator) usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.downstream_senders.items.len;
    }

    /// Get cascade topology information
    pub fn getTopologyInfo(self: *CascadeCoordinator) TopologyInfo {
        return .{
            .cascade_depth = self.cascade_depth,
            .role = self.getRole(),
            .downstream_count = self.getDownstreamCount(),
            .last_forwarded_lsn = self.last_forwarded_lsn,
            .has_upstream = self.upstream_conninfo != null,
        };
    }
};

/// Cascade topology information
pub const TopologyInfo = struct {
    cascade_depth: u32,
    role: CascadeRole,
    downstream_count: usize,
    last_forwarded_lsn: LSN,
    has_upstream: bool,
};

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;
const test_allocator = testing.allocator;

test "CascadeCoordinator: init primary (depth 0)" {
    var coordinator = try CascadeCoordinator.init(
        test_allocator,
        .{ .max_cascade_depth = 4 },
        null, // no upstream
        0, // primary depth
    );
    defer coordinator.deinit();

    try testing.expectEqual(@as(u32, 0), coordinator.cascade_depth);
    try testing.expectEqual(CascadeRole.primary, coordinator.getRole());
    try testing.expect(!coordinator.isCascadingEnabled());
    try testing.expectEqual(@as(usize, 0), coordinator.getDownstreamCount());
}

test "CascadeCoordinator: init intermediate replica (depth 1)" {
    var coordinator = try CascadeCoordinator.init(
        test_allocator,
        .{ .max_cascade_depth = 4, .enable_forwarding = true },
        "host=upstream port=5433",
        1,
    );
    defer coordinator.deinit();

    try testing.expectEqual(@as(u32, 1), coordinator.cascade_depth);
    try testing.expect(coordinator.isCascadingEnabled());
    try testing.expectEqual(@as(u32, 2), coordinator.getDownstreamCascadeDepth());
}

test "CascadeCoordinator: max depth exceeded" {
    const result = CascadeCoordinator.init(
        test_allocator,
        .{ .max_cascade_depth = 4 },
        "host=upstream",
        5, // exceeds max
    );
    try testing.expectError(Error.MaxCascadeDepthExceeded, result);
}

test "CascadeCoordinator: role detection" {
    // Primary: no upstream, no downstream
    var primary = try CascadeCoordinator.init(test_allocator, .{}, null, 0);
    defer primary.deinit();
    try testing.expectEqual(CascadeRole.primary, primary.getRole());

    // Leaf: has upstream, no downstream
    var leaf = try CascadeCoordinator.init(test_allocator, .{}, "host=primary", 1);
    defer leaf.deinit();
    try testing.expectEqual(CascadeRole.leaf, leaf.getRole());
}

test "CascadeCoordinator: topology info" {
    var coordinator = try CascadeCoordinator.init(
        test_allocator,
        .{ .enable_forwarding = true },
        "host=primary",
        2,
    );
    defer coordinator.deinit();

    const info = coordinator.getTopologyInfo();
    try testing.expectEqual(@as(u32, 2), info.cascade_depth);
    try testing.expectEqual(CascadeRole.leaf, info.role);
    try testing.expectEqual(@as(usize, 0), info.downstream_count);
    try testing.expect(info.has_upstream);
}

test "CascadeCoordinator: validate downstream connection depth limit" {
    var coordinator = try CascadeCoordinator.init(
        test_allocator,
        .{ .max_cascade_depth = 2 },
        "host=primary",
        2, // at max depth
    );
    defer coordinator.deinit();

    const result = coordinator.validateDownstreamConnection("downstream-system-id");
    try testing.expectError(Error.MaxCascadeDepthExceeded, result);
}

test "CascadeCoordinator: forwarding disabled" {
    var coordinator = try CascadeCoordinator.init(
        test_allocator,
        .{ .enable_forwarding = false },
        "host=primary",
        1,
    );
    defer coordinator.deinit();

    try testing.expect(!coordinator.isCascadingEnabled());

    // forwardWalData should return early when forwarding disabled
    const data = [_]u8{1, 2, 3};
    try coordinator.forwardWalData(100, 103, &data);
    try testing.expectEqual(@as(LSN, 0), coordinator.last_forwarded_lsn);
}

test "CascadeCoordinator: getDownstreamCount empty" {
    var coordinator = try CascadeCoordinator.init(test_allocator, .{}, null, 0);
    defer coordinator.deinit();

    try testing.expectEqual(@as(usize, 0), coordinator.getDownstreamCount());
}

test "CascadeCoordinator: upstream_conninfo ownership" {
    const conninfo = "host=upstream port=5433";
    var coordinator = try CascadeCoordinator.init(
        test_allocator,
        .{},
        conninfo,
        1,
    );
    defer coordinator.deinit();

    // Verify it was duped (not just a reference)
    try testing.expect(coordinator.upstream_conninfo.?.ptr != conninfo.ptr);
    try testing.expectEqualStrings(conninfo, coordinator.upstream_conninfo.?);
}

test "CascadeCoordinator: cascade depth progression" {
    var primary = try CascadeCoordinator.init(test_allocator, .{}, null, 0);
    defer primary.deinit();
    try testing.expectEqual(@as(u32, 1), primary.getDownstreamCascadeDepth());

    var intermediate = try CascadeCoordinator.init(test_allocator, .{}, "host=primary", 1);
    defer intermediate.deinit();
    try testing.expectEqual(@as(u32, 2), intermediate.getDownstreamCascadeDepth());

    var leaf = try CascadeCoordinator.init(test_allocator, .{}, "host=intermediate", 2);
    defer leaf.deinit();
    try testing.expectEqual(@as(u32, 3), leaf.getDownstreamCascadeDepth());
}

// ============================================================================
// Edge Case Tests
// ============================================================================

test "CascadeCoordinator: forwardWalData on primary (no-op)" {
    var primary = try CascadeCoordinator.init(
        test_allocator,
        .{ .enable_forwarding = true },
        null,
        0,
    );
    defer primary.deinit();

    const data = [_]u8{ 1, 2, 3, 4 };
    try primary.forwardWalData(100, 104, &data);

    // Primary has no upstream, so forwarding should be no-op
    try testing.expectEqual(@as(LSN, 0), primary.last_forwarded_lsn);
    try testing.expectEqual(CascadeRole.primary, primary.getRole());
}

test "CascadeCoordinator: forwardWalData on leaf (no-op)" {
    var leaf = try CascadeCoordinator.init(
        test_allocator,
        .{ .enable_forwarding = true },
        "host=upstream",
        1,
    );
    defer leaf.deinit();

    const data = [_]u8{ 1, 2, 3, 4 };
    try leaf.forwardWalData(100, 104, &data);

    // Leaf has no downstream, so forwarding should be no-op
    try testing.expectEqual(@as(LSN, 0), leaf.last_forwarded_lsn);
    try testing.expectEqual(CascadeRole.leaf, leaf.getRole());
}

test "CascadeCoordinator: role transition from leaf to intermediate" {
    var coordinator = try CascadeCoordinator.init(
        test_allocator,
        .{},
        "host=primary",
        1,
    );
    defer coordinator.deinit();

    // Initially a leaf (has upstream, no downstream)
    try testing.expectEqual(CascadeRole.leaf, coordinator.getRole());

    // Simulate adding a downstream sender (would normally be a real WalSender)
    var dummy_sender: sender.WalSender = undefined;
    try coordinator.registerDownstream(&dummy_sender);

    // Now it's intermediate (has both upstream and downstream)
    try testing.expectEqual(CascadeRole.intermediate, coordinator.getRole());
    try testing.expectEqual(@as(usize, 1), coordinator.getDownstreamCount());

    // Remove downstream, back to leaf
    coordinator.unregisterDownstream(&dummy_sender);
    try testing.expectEqual(CascadeRole.leaf, coordinator.getRole());
    try testing.expectEqual(@as(usize, 0), coordinator.getDownstreamCount());
}

test "CascadeCoordinator: unregister non-existent downstream (no-op)" {
    var coordinator = try CascadeCoordinator.init(test_allocator, .{}, null, 0);
    defer coordinator.deinit();

    var dummy_sender: sender.WalSender = undefined;
    // Unregistering a sender that was never registered should be a no-op
    coordinator.unregisterDownstream(&dummy_sender);
    try testing.expectEqual(@as(usize, 0), coordinator.getDownstreamCount());
}

test "CascadeCoordinator: max depth boundary" {
    const config = Config{ .max_cascade_depth = 3 };

    // Depth 3 (at max) should succeed
    var at_max = try CascadeCoordinator.init(test_allocator, config, "host=up", 3);
    defer at_max.deinit();
    try testing.expectEqual(@as(u32, 3), at_max.cascade_depth);

    // Depth 4 (exceeds max) should fail
    const over_max = CascadeCoordinator.init(test_allocator, config, "host=up", 4);
    try testing.expectError(Error.MaxCascadeDepthExceeded, over_max);
}

test "CascadeCoordinator: validateDownstreamConnection at max depth" {
    var coordinator = try CascadeCoordinator.init(
        test_allocator,
        .{ .max_cascade_depth = 2 },
        "host=primary",
        2, // at max depth
    );
    defer coordinator.deinit();

    // Should reject downstream connection when at max depth
    const result = coordinator.validateDownstreamConnection("system-id-abc");
    try testing.expectError(Error.MaxCascadeDepthExceeded, result);
}

test "CascadeCoordinator: validateDownstreamConnection below max" {
    var coordinator = try CascadeCoordinator.init(
        test_allocator,
        .{ .max_cascade_depth = 4 },
        "host=primary",
        2, // below max
    );
    defer coordinator.deinit();

    // Should allow downstream connection when below max depth
    try coordinator.validateDownstreamConnection("system-id-xyz");
}

test "CascadeCoordinator: empty upstream_conninfo for primary" {
    var primary = try CascadeCoordinator.init(test_allocator, .{}, null, 0);
    defer primary.deinit();

    try testing.expect(primary.upstream_conninfo == null);
    try testing.expect(!primary.isCascadingEnabled());
}

test "CascadeCoordinator: topology info for intermediate with multiple downstream" {
    var intermediate = try CascadeCoordinator.init(
        test_allocator,
        .{ .enable_forwarding = true },
        "host=primary",
        1,
    );
    defer intermediate.deinit();

    var sender1: sender.WalSender = undefined;
    var sender2: sender.WalSender = undefined;
    var sender3: sender.WalSender = undefined;

    try intermediate.registerDownstream(&sender1);
    try intermediate.registerDownstream(&sender2);
    try intermediate.registerDownstream(&sender3);

    const info = intermediate.getTopologyInfo();
    try testing.expectEqual(@as(u32, 1), info.cascade_depth);
    try testing.expectEqual(CascadeRole.intermediate, info.role);
    try testing.expectEqual(@as(usize, 3), info.downstream_count);
    try testing.expect(info.has_upstream);
}

test "CascadeCoordinator: forwardWalData updates last_forwarded_lsn" {
    var intermediate = try CascadeCoordinator.init(
        test_allocator,
        .{ .enable_forwarding = true },
        "host=primary",
        1,
    );
    defer intermediate.deinit();

    // Add a downstream to make it intermediate role
    var dummy_sender: sender.WalSender = undefined;
    try intermediate.registerDownstream(&dummy_sender);

    try testing.expectEqual(CascadeRole.intermediate, intermediate.getRole());

    const data1 = [_]u8{ 1, 2, 3 };
    try intermediate.forwardWalData(100, 103, &data1);
    try testing.expectEqual(@as(LSN, 103), intermediate.last_forwarded_lsn);

    const data2 = [_]u8{ 4, 5, 6, 7 };
    try intermediate.forwardWalData(103, 107, &data2);
    try testing.expectEqual(@as(LSN, 107), intermediate.last_forwarded_lsn);
}

// ============================================================================
// Concurrent Stress Tests
// ============================================================================

test "CascadeCoordinator: concurrent downstream register/unregister stress" {
    var coordinator = try CascadeCoordinator.init(
        test_allocator,
        .{ .enable_forwarding = true },
        "host=primary",
        1,
    );
    defer coordinator.deinit();

    const num_threads = 8;
    const ops_per_thread = 25;

    const ThreadContext = struct {
        coord: *CascadeCoordinator,
        thread_id: usize,

        fn threadFunc(ctx: *@This()) void {
            var prng = std.Random.DefaultPrng.init(@intCast(ctx.thread_id));
            const random = prng.random();

            // Create thread-local senders array
            var senders: [5]sender.WalSender = undefined;

            var i: usize = 0;
            while (i < ops_per_thread) : (i += 1) {
                const op = random.intRangeAtMost(u8, 0, 2);
                const sender_idx = random.intRangeAtMost(usize, 0, 4);

                switch (op) {
                    0 => {
                        // Register downstream
                        ctx.coord.registerDownstream(&senders[sender_idx]) catch {};
                    },
                    1 => {
                        // Unregister downstream
                        ctx.coord.unregisterDownstream(&senders[sender_idx]);
                    },
                    2 => {
                        // Forward WAL data
                        const data = [_]u8{ 1, 2, 3, 4 };
                        const lsn_base: LSN = 1000 + @as(LSN, i);
                        ctx.coord.forwardWalData(lsn_base, lsn_base + 4, &data) catch {};
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
            .coord = &coordinator,
            .thread_id = tid,
        };
        t.* = try std.Thread.spawn(.{}, ThreadContext.threadFunc, .{ctx});
    }

    // Wait for all threads
    for (&threads) |*t| {
        t.join();
    }

    // Verify coordinator is still in valid state
    const info = coordinator.getTopologyInfo();
    try testing.expectEqual(@as(u32, 1), info.cascade_depth);
    try testing.expect(info.role == CascadeRole.leaf or info.role == CascadeRole.intermediate);
}
