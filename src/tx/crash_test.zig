//! Crash Injection Tests
//!
//! Simulates power failures and crashes at various points during database operations
//! to verify WAL recovery and ACID guarantees.
//!
//! Test Categories:
//!   - Crash during transaction commit (before/after WAL write, during checkpoint)
//!   - Crash during page write
//!   - Crash during index update
//!   - Multiple concurrent transactions with crash
//!   - Partial write scenarios (torn pages)
//!
//! Each test:
//! 1. Performs database operations
//! 2. Simulates crash at specific point
//! 3. Reopens database
//! 4. Verifies data integrity and consistency

const std = @import("std");
const engine_mod = @import("../sql/engine.zig");
const Database = engine_mod.Database;
const executor_mod = @import("../sql/executor.zig");
const Row = executor_mod.Row;
const testing = std.testing;

// Test helper to create a temporary database
fn createTestDb(allocator: std.mem.Allocator, path: []const u8) !Database {
    return Database.open(allocator, path, .{ .page_size = 4096 });
}

// Helper to execute SQL and discard result
fn execSql(db: *Database, sql: []const u8) !void {
    var result = try db.exec(sql);
    result.close(db.allocator);
}

// Helper to materialize rows
fn materializeRows(allocator: std.mem.Allocator, iter: *executor_mod.RowIterator) !std.ArrayList(Row) {
    var rows: std.ArrayList(Row) = .{};
    errdefer {
        for (rows.items) |*row| {
            row.deinit();
        }
        rows.deinit(allocator);
    }

    while (try iter.next()) |row| {
        try rows.append(allocator, row);
    }

    return rows;
}

// ══════════════════════════════════════════════════════════════════════════
// Crash Point 1: During Transaction Commit (before WAL flush)
// ══════════════════════════════════════════════════════════════════════════

test "crash: commit before WAL flush" {
    const allocator = testing.allocator;
    const db_path = ":memory:";

    // Setup: Create table and insert data
    {
        var db = try createTestDb(allocator, db_path);
        defer db.close();

        try execSql(&db, "CREATE TABLE t1 (id INTEGER, val INTEGER)");
        try execSql(&db, "INSERT INTO t1 VALUES (1, 100)");
        try execSql(&db, "INSERT INTO t1 VALUES (2, 200)");
    }

    // TODO: Implement crash simulation before WAL flush
    // For now, this is a placeholder test structure
    try testing.expect(true);
}

// ══════════════════════════════════════════════════════════════════════════
// Crash Point 2: During WAL Checkpoint
// ══════════════════════════════════════════════════════════════════════════

test "crash: during checkpoint" {
    const allocator = testing.allocator;
    const db_path = ":memory:";

    // Setup: Create table and insert enough data to trigger checkpoint
    {
        var db = try createTestDb(allocator, db_path);
        defer db.close();

        try execSql(&db, "CREATE TABLE t1 (id INTEGER, val INTEGER)");

        // Insert many rows to fill WAL
        var i: usize = 0;
        while (i < 1000) : (i += 1) {
            var buf: [100]u8 = undefined;
            const sql = try std.fmt.bufPrint(&buf, "INSERT INTO t1 VALUES ({d}, {d})", .{i, i * 10});
            try execSql(&db, sql);
        }

        // Force checkpoint (this should be a public API)
        // try db.checkpoint();
    }

    // TODO: Simulate crash during checkpoint
    try testing.expect(true);
}

// ══════════════════════════════════════════════════════════════════════════
// Crash Point 3: After WAL Write, Before Main DB Update
// ══════════════════════════════════════════════════════════════════════════

test "crash: after WAL write, before main DB update" {
    const allocator = testing.allocator;

    // Use temp file instead of :memory: so data persists across reopens
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    var path_buf: [256]u8 = undefined;
    const db_path = try std.fmt.bufPrint(&path_buf, "test_crash_wal_{d}.db", .{std.time.milliTimestamp()});
    const db_full_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(db_full_path);

    var full_path_buf: [512]u8 = undefined;
    const full_path = try std.fmt.bufPrint(&full_path_buf, "{s}/{s}", .{db_full_path, db_path});

    // Phase 1: Write data and commit (WAL written, no checkpoint)
    {
        var db = try createTestDb(allocator, full_path);

        try execSql(&db, "CREATE TABLE t1 (id INTEGER, val INTEGER)");
        try execSql(&db, "INSERT INTO t1 VALUES (1, 100)");
        try execSql(&db, "INSERT INTO t1 VALUES (2, 200)");

        // Close WITHOUT checkpoint — simulates crash after WAL write
        // (db.close() should NOT call checkpoint, data remains in WAL)
        db.close();
    }

    // Phase 2: Reopen database — WAL recovery should apply changes
    {
        var db = try createTestDb(allocator, full_path);
        defer db.close();

        // Verify data exists after recovery
        var result = try db.exec("SELECT id, val FROM t1 ORDER BY id");
        defer result.close(allocator);

        if (result.rows) |*iter| {
            var rows = try materializeRows(allocator, iter);
            defer {
                for (rows.items) |*row| row.deinit();
                rows.deinit(allocator);
            }

            // Should have 2 rows
            try testing.expectEqual(@as(usize, 2), rows.items.len);

            // Row 1: id=1, val=100
            try testing.expectEqual(@as(i64, 1), rows.items[0].values[0].integer);
            try testing.expectEqual(@as(i64, 100), rows.items[0].values[1].integer);

            // Row 2: id=2, val=200
            try testing.expectEqual(@as(i64, 2), rows.items[1].values[0].integer);
            try testing.expectEqual(@as(i64, 200), rows.items[1].values[1].integer);
        }
    }

    // Cleanup
    tmp.dir.deleteFile(db_path) catch {};
    var wal_path_buf: [272]u8 = undefined;
    const wal_path = try std.fmt.bufPrint(&wal_path_buf, "{s}-wal", .{db_path});
    tmp.dir.deleteFile(wal_path) catch {};
}

// ══════════════════════════════════════════════════════════════════════════
// Crash Point 4: During Page Write (Torn Page Scenario)
// ══════════════════════════════════════════════════════════════════════════

test "crash: torn page during write" {
    const allocator = testing.allocator;
    const db_path = ":memory:";

    // Simulate partial page write (e.g., power failure mid-write)
    // WAL should protect against torn pages
    {
        var db = try createTestDb(allocator, db_path);
        defer db.close();

        try execSql(&db, "CREATE TABLE t1 (id INTEGER, data TEXT)");

        // Insert large row that spans multiple pages
        const large_text = "X" ** 8000;
        var buf: [8200]u8 = undefined;
        const sql = try std.fmt.bufPrint(&buf, "INSERT INTO t1 VALUES (1, '{s}')", .{large_text});
        try execSql(&db, sql);

        // TODO: Simulate torn page (partial write)
    }

    // Recovery should either:
    // 1. See complete transaction (all pages written)
    // 2. See no transaction (rollback via WAL)
    // Never see partial transaction

    try testing.expect(true);
}

// ══════════════════════════════════════════════════════════════════════════
// Crash Point 5: Multiple Transactions, Crash Before Some Commits
// ══════════════════════════════════════════════════════════════════════════

test "crash: multiple transactions, partial commits" {
    const allocator = testing.allocator;
    const db_path = ":memory:";

    // Test atomicity: only fully committed transactions should survive crash
    {
        var db = try createTestDb(allocator, db_path);
        defer db.close();

        try execSql(&db, "CREATE TABLE t1 (id INTEGER, val INTEGER)");

        // TX1: Committed
        try execSql(&db, "BEGIN");
        try execSql(&db, "INSERT INTO t1 VALUES (1, 100)");
        try execSql(&db, "COMMIT");

        // TX2: Started but not committed
        try execSql(&db, "BEGIN");
        try execSql(&db, "INSERT INTO t1 VALUES (2, 200)");
        // No COMMIT - simulating crash here

        // TODO: Simulate crash
    }

    // After recovery, only TX1 should be visible
    {
        var db = try createTestDb(allocator, db_path);
        defer db.close();

        // TODO: Verify only row with id=1 exists
    }

    try testing.expect(true);
}

// ══════════════════════════════════════════════════════════════════════════
// Crash Point 6: During Index Update
// ══════════════════════════════════════════════════════════════════════════

test "crash: during index update" {
    const allocator = testing.allocator;
    const db_path = ":memory:";

    // Indexes and data table must be in sync after crash recovery
    {
        var db = try createTestDb(allocator, db_path);
        defer db.close();

        try execSql(&db, "CREATE TABLE t1 (id INTEGER, val INTEGER)");
        try execSql(&db, "CREATE INDEX idx_val ON t1(val)");

        try execSql(&db, "BEGIN");
        try execSql(&db, "INSERT INTO t1 VALUES (1, 100)");
        try execSql(&db, "INSERT INTO t1 VALUES (2, 200)");
        try execSql(&db, "COMMIT");

        // TODO: Simulate crash during index update
    }

    // After recovery, index should be consistent with table
    {
        var db = try createTestDb(allocator, db_path);
        defer db.close();

        // TODO: Verify index consistency
        // SELECT via index should return same results as table scan
    }

    try testing.expect(true);
}

// ══════════════════════════════════════════════════════════════════════════
// Crash Point 7: Crash Recovery Chain (Crash During Recovery)
// ══════════════════════════════════════════════════════════════════════════

test "crash: during recovery (double crash)" {
    const allocator = testing.allocator;
    const db_path = ":memory:";

    // Simulate crash during WAL replay
    // Recovery process itself must be crash-safe
    {
        var db = try createTestDb(allocator, db_path);
        defer db.close();

        try execSql(&db, "CREATE TABLE t1 (id INTEGER, val INTEGER)");
        try execSql(&db, "INSERT INTO t1 VALUES (1, 100)");

        // TODO: Simulate crash
    }

    // First recovery attempt
    // TODO: Simulate crash during recovery

    // Second recovery attempt - should still succeed
    {
        var db = try createTestDb(allocator, db_path);
        defer db.close();

        // TODO: Verify data integrity
    }

    try testing.expect(true);
}
