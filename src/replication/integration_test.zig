//! Phase 5 End-to-End WAL Replication Integration Test
//!
//! This test exercises real TCP-based WAL replication across two independent
//! Database instances (primary and replica) over a loopback socket pair.
//! It verifies that:
//! 1. Sender thread reads real WAL frames from primary
//! 2. Receiver thread applies real frames to replica's own WAL + Pager
//! 3. Replica's data file converges byte-identically to primary's
//! 4. Standby status feedback loop works (replica's LSN reaches sender)

const std = @import("std");
const Allocator = std.mem.Allocator;

const sender_mod = @import("sender.zig");
const receiver_mod = @import("receiver.zig");
const transport = @import("transport.zig");
const protocol = @import("protocol.zig");
const slot = @import("slot.zig");
const wal_mod = @import("../tx/wal.zig");
const page_mod = @import("../storage/page.zig");
const checksum_mod = @import("../util/checksum.zig");

const WalSender = sender_mod.WalSender;
const WalReceiver = receiver_mod.WalReceiver;
const Pager = page_mod.Pager;
const Wal = wal_mod.Wal;
const Lsn = wal_mod.Lsn;

/// Sender loop: reads WAL frames from sender.wal, packs into messages, sends over stream.
pub fn runSenderLoop(
    sender: *WalSender,
    stream: std.net.Stream,
    allocator: Allocator,
    stop: *std.atomic.Value(bool),
) !void {
    var buf: [8192]u8 = undefined;

    while (!stop.load(.acquire)) {
        // Capture the LSN before reading: readWalChunk advances current_lsn by
        // frame count (packed checkpoint_seq/frame_index), not by byte count,
        // so wal_start/wal_end must bracket the call rather than be derived
        // from chunk_size arithmetic.
        const wal_start = sender.current_lsn;
        const chunk_result = sender.readWalChunk(&buf) catch |err| {
            // If stream is closed and stop flag is set, exit cleanly
            if (stop.load(.acquire)) {
                return;
            }
            return err;
        };

        if (chunk_result) |chunk_size| {
            const wal_end = sender.current_lsn;

            const data_copy = try allocator.dupe(u8, buf[0..chunk_size]);
            defer allocator.free(data_copy);

            const msg = protocol.BackendMessage{
                .wal_data = .{
                    .wal_start = wal_start,
                    .wal_end = wal_end,
                    .server_timestamp = std.time.microTimestamp(),
                    .data = data_copy,
                },
            };

            try transport.sendBackendMessage(stream, allocator, msg);
        } else {
            // No data available: check if we should send keepalive
            if (sender.shouldSendKeepalive()) {
                const keepalive_msg = sender.createKeepaliveMessage(true);
                try transport.sendBackendMessage(stream, allocator, keepalive_msg);
            }

            // Sleep briefly to avoid busy-waiting
            std.Thread.sleep(10_000_000); // 10ms
        }
    }
}

/// Receiver loop: reads messages from stream, dispatches wal_data/keepalive,
/// periodically sends standby_status updates.
pub fn runReceiverLoop(
    receiver: *WalReceiver,
    stream: std.net.Stream,
    allocator: Allocator,
    stop: *std.atomic.Value(bool),
) !void {
    while (!stop.load(.acquire)) {
        // Try to receive a message from sender
        const msg_result = transport.receiveBackendMessage(stream, allocator) catch |err| {
            // Read timed out (SO_RCVTIMEO): no message yet, recheck stop flag.
            if (err == error.WouldBlock) {
                continue;
            }
            // Stream closed or error
            if (err == error.EndOfStream or err == error.ConnectionClosed or err == error.NotOpenForReading) {
                if (stop.load(.acquire)) {
                    return;
                }
                return err;
            }

            return err;
        };

        // Dispatch based on message type
        switch (msg_result) {
            .wal_data => |wd| {
                try receiver.processWalData(wd.wal_start, wd.wal_end, wd.data);
            },
            .keepalive => |ka| {
                const reply_needed = try receiver.processKeepalive(ka.wal_end, ka.reply_requested);
                if (reply_needed) {
                    const status = receiver.createStatusUpdate(false);
                    try transport.sendFrontendMessage(stream, allocator, status);
                }
            },
            else => {
                // Ignore other message types in receiver loop
            },
        }

        transport.deinitBackendMessage(allocator, @constCast(&msg_result));

        // Periodically send status updates
        if (receiver.shouldSendStatus()) {
            const status = receiver.createStatusUpdate(false);
            try transport.sendFrontendMessage(stream, allocator, status);
        }
    }
}

/// Sender status reader loop: reads FrontendMessages from replica (standby status updates),
/// invokes sender.processStandbyStatus to update slot LSN positions.
pub fn runSenderStatusReaderLoop(
    sender: *WalSender,
    stream: std.net.Stream,
    allocator: Allocator,
    stop: *std.atomic.Value(bool),
) !void {
    while (!stop.load(.acquire)) {
        // Try to receive a FrontendMessage from replica
        const msg_result = transport.receiveFrontendMessage(stream, allocator) catch |err| {
            // Read timed out (SO_RCVTIMEO): no message yet, recheck stop flag.
            if (err == error.WouldBlock) {
                continue;
            }
            // Stream closed or error
            if (err == error.EndOfStream or err == error.ConnectionClosed or err == error.NotOpenForReading) {
                if (stop.load(.acquire)) {
                    return;
                }
                return err;
            }

            return err;
        };

        // Dispatch based on message type
        switch (msg_result) {
            .standby_status => |ss| {
                try sender.processStandbyStatus(ss.write_lsn, ss.flush_lsn, ss.apply_lsn);
            },
            else => {
                // Ignore other message types in status reader loop
            },
        }

        transport.deinitFrontendMessage(allocator, @constCast(&msg_result));
    }
}

test "Phase 5: end-to-end WAL replication over real loopback socket" {
    const allocator = std.testing.allocator;
    const primary_path = "test_phase5_primary.db";
    const replica_path = "test_phase5_replica.db";

    defer std.fs.cwd().deleteFile(primary_path) catch {};
    defer std.fs.cwd().deleteFile(primary_path ++ "-wal") catch {};
    defer std.fs.cwd().deleteFile(replica_path) catch {};
    defer std.fs.cwd().deleteFile(replica_path ++ "-wal") catch {};

    // ── Setup: Create primary Wal + Pager ──

    var primary_pager = try Pager.init(allocator, primary_path, .{ .page_size = 4096 });
    defer primary_pager.deinit();

    var primary_wal = try Wal.init(allocator, primary_path, 4096);
    defer primary_wal.deinit();

    // ── Setup: Create replica Wal + Pager ──

    var replica_pager = try Pager.init(allocator, replica_path, .{ .page_size = 4096 });
    defer replica_pager.deinit();

    var replica_wal = try Wal.init(allocator, replica_path, 4096);
    defer replica_wal.deinit();

    // ── Setup: Create loopback TCP socket pair ──

    // Bind server on 127.0.0.1:0 (OS-assigned ephemeral port)
    const server_addr = try std.net.Address.parseIp("127.0.0.1", 0);
    var server_socket = try server_addr.listen(.{
        .reuse_address = true,
    });
    defer server_socket.deinit();

    const server_port = server_socket.listen_address.in.getPort();

    // Spawn a thread that accepts one incoming connection
    var accepted_stream: ?std.net.Stream = null;
    var accept_error: ?anyerror = null;

    const accept_thread = try std.Thread.spawn(.{}, acceptConnection, .{
        &server_socket,
        &accepted_stream,
        &accept_error,
    });

    // Connect to the server from the "client" side using tcpConnectToAddress
    const server_addr_for_connect = try std.net.Address.parseIp("127.0.0.1", server_port);
    const client_stream = try std.net.tcpConnectToAddress(server_addr_for_connect);
    // Closed explicitly below (before thread join) to unblock in-flight reads;
    // no defer here to avoid a double-close panic.

    // Wait for accept to complete and get the server-side stream
    var wait_count: u32 = 0;
    while (accepted_stream == null and accept_error == null and wait_count < 1000) : (wait_count += 1) {
        std.Thread.sleep(1_000_000); // 1 ms
    }
    try std.testing.expect(accepted_stream != null);
    if (accept_error) |err| return err;

    accept_thread.join();

    var server_stream = accepted_stream.?;
    // Closed explicitly below (before thread join) to unblock in-flight reads;
    // no defer here to avoid a double-close panic.

    // Bound blocking reads on both ends so loop threads periodically recheck
    // their stop flag even if shutdown() doesn't reliably unblock a
    // concurrent blocking recv() on the CI runner's kernel.
    try transport.setReceiveTimeout(server_stream, 200);
    try transport.setReceiveTimeout(client_stream, 200);

    // ── Setup: Write 2+ transactions to primary WAL ──

    // Transaction 1: Write frames to pages 100, 101
    var page_data1: [4096]u8 = undefined;
    @memset(&page_data1, 0x11);
    page_data1[0] = 0x03; // PageType.leaf
    try primary_wal.writeFrame(100, &page_data1);

    var page_data2: [4096]u8 = undefined;
    @memset(&page_data2, 0x22);
    page_data2[0] = 0x03;
    try primary_wal.writeFrame(101, &page_data2);

    try primary_wal.commit(2); // commit frame with db_page_count=2

    // ── Gap: Simulate keepalive interval (no writes for a moment) ──
    std.Thread.sleep(10_000_000); // 10 ms

    // Transaction 2: Write frames to pages 102, 103
    var page_data3: [4096]u8 = undefined;
    @memset(&page_data3, 0x33);
    page_data3[0] = 0x03;
    try primary_wal.writeFrame(102, &page_data3);

    var page_data4: [4096]u8 = undefined;
    @memset(&page_data4, 0x44);
    page_data4[0] = 0x03;
    try primary_wal.writeFrame(103, &page_data4);

    try primary_wal.commit(2); // Second commit

    // ── Setup: Create sender and receiver ──

    const sender_config = sender_mod.Config{ .keepalive_interval_ms = 200 };
    const sender_slot_mgr = try allocator.create(slot.SlotManager);
    defer allocator.destroy(sender_slot_mgr);
    sender_slot_mgr.* = slot.SlotManager.init(allocator);
    defer sender_slot_mgr.deinit();

    // Create and activate the replication slot on the sender
    try sender_slot_mgr.createSlot("test-replica-slot", false);

    var sender = try WalSender.init(
        allocator,
        sender_slot_mgr,
        "test-system-id",
        1,
        sender_config,
    );
    defer {
        allocator.free(sender.system_id);
        if (sender.slot_name) |name| {
            allocator.free(name);
        }
    }

    sender.wal = &primary_wal;
    // Note: phase 3's existing tests create wal_mutex; for simplicity in this test,
    // we reuse the primary_wal directly and don't need wal_mutex since threads coordinate via stop flag
    var wal_mutex = std.Thread.Mutex{};
    sender.wal_mutex = &wal_mutex;
    sender.stream = server_stream;

    // Start replication on the created slot
    try sender.startReplication("test-replica-slot", 0);

    const receiver_config = receiver_mod.Config{
        .primary_conninfo = "host=127.0.0.1",
        .slot_name = "test-replica-slot",
        .status_interval_ms = 200,
    };
    var receiver = try receiver_mod.WalReceiver.init(allocator, receiver_config);
    defer receiver.deinit();

    receiver.local_wal = &replica_wal;
    receiver.local_pager = &replica_pager;
    receiver.stream = client_stream;

    // ── Spawn sender loop thread ──

    var sender_stop = std.atomic.Value(bool).init(false);
    const sender_thread = try std.Thread.spawn(.{}, runSenderLoop, .{
        &sender,
        server_stream,
        allocator,
        &sender_stop,
    });

    // ── Spawn receiver loop thread ──

    var receiver_stop = std.atomic.Value(bool).init(false);
    const receiver_thread = try std.Thread.spawn(.{}, runReceiverLoop, .{
        &receiver,
        client_stream,
        allocator,
        &receiver_stop,
    });

    // ── Spawn sender status reader thread ──
    // This thread reads FrontendMessages (standby_status updates) from the replica
    // and invokes sender.processStandbyStatus to update the slot's confirmed_flush_lsn.

    const status_reader_thread = try std.Thread.spawn(.{}, runSenderStatusReaderLoop, .{
        &sender,
        server_stream,
        allocator,
        &sender_stop,
    });

    // ── Wait for replication to converge (with timeout) ──

    const timeout_ms = 5000;
    const deadline = std.time.milliTimestamp() + timeout_ms;
    var converged = false;

    while (std.time.milliTimestamp() < deadline) {
        // commit() re-tags the last written frame as the commit frame rather
        // than appending a new one, so each 2-write transaction contributes
        // 2 committed frames: 4 total across both transactions.
        if (receiver.apply_lsn.frame_index >= 4) {
            converged = true;
            break;
        }
        std.Thread.sleep(100_000_000); // 100 ms
    }

    // ── Wait a bit more to allow status messages to flow ──
    // The receiver sends status updates every status_interval_ms (200ms).
    // Give time for at least 2-3 round-trips and for sender to receive & process.
    std.Thread.sleep(1_500_000_000); // 1.5s

    // ── Initiate clean shutdown ──

    sender_stop.store(true, .release);
    receiver_stop.store(true, .release);

    // shutdown() (not just close()) is required to reliably unblock threads
    // parked in a blocking read/write on these fds: server_stream is shared
    // by sender_thread (writer) and status_reader_thread (reader), and on
    // Linux, close() from a different thread does not wake a concurrent
    // blocking syscall on the same fd — only shutdown() is guaranteed to.
    std.posix.shutdown(server_stream.handle, .both) catch {};
    std.posix.shutdown(client_stream.handle, .both) catch {};
    server_stream.close();
    client_stream.close();

    sender_thread.join();
    receiver_thread.join();
    status_reader_thread.join();

    // ── Verify convergence ──

    try std.testing.expect(converged);

    // ── Checkpoint primary's own WAL into its pager ──
    // The primary_pager was never written to directly (only primary_wal), so
    // it must be checkpointed before its pages can be read back for comparison.
    try primary_wal.checkpoint(&primary_pager);

    // ── Verify byte-identical pages ──
    // Compare pages 100-103 between primary and replica data files
    // Ignore checksum bytes 12-15 (recomputed on write)

    var primary_buf: [4096]u8 = undefined;
    var replica_buf: [4096]u8 = undefined;

    try primary_pager.readPage(100, &primary_buf);
    try replica_pager.readPage(100, &replica_buf);
    try std.testing.expectEqualSlices(u8, primary_buf[0..12], replica_buf[0..12]);
    try std.testing.expectEqualSlices(u8, primary_buf[16..], replica_buf[16..]);

    try primary_pager.readPage(101, &primary_buf);
    try replica_pager.readPage(101, &replica_buf);
    try std.testing.expectEqualSlices(u8, primary_buf[0..12], replica_buf[0..12]);
    try std.testing.expectEqualSlices(u8, primary_buf[16..], replica_buf[16..]);

    try primary_pager.readPage(102, &primary_buf);
    try replica_pager.readPage(102, &replica_buf);
    try std.testing.expectEqualSlices(u8, primary_buf[0..12], replica_buf[0..12]);
    try std.testing.expectEqualSlices(u8, primary_buf[16..], replica_buf[16..]);

    try primary_pager.readPage(103, &primary_buf);
    try replica_pager.readPage(103, &replica_buf);
    try std.testing.expectEqualSlices(u8, primary_buf[0..12], replica_buf[0..12]);
    try std.testing.expectEqualSlices(u8, primary_buf[16..], replica_buf[16..]);

    // ── Verify standby status feedback reached sender ──
    // The receiver periodically sends standby_status messages (at status_interval_ms = 200ms)
    // which are read by the status reader thread and processed by sender.processStandbyStatus().
    // This updates the slot's confirmed_flush_lsn. We verify the feedback loop worked by
    // checking that the slot's confirmed_flush_lsn was updated from its initial value of 0.

    try std.testing.expect(receiver.apply_lsn.frame_index > 0);
    const final_slot = try sender_slot_mgr.getSlot("test-replica-slot");
    try std.testing.expect(final_slot.confirmed_flush_lsn > 0);
}

/// Helper thread function to accept one incoming connection.
fn acceptConnection(
    server: *std.net.Server,
    accepted: *?std.net.Stream,
    err: *?anyerror,
) void {
    const conn = server.accept() catch |e| {
        err.* = e;
        return;
    };
    accepted.* = conn.stream;
}
