const std = @import("std");
const silica = @import("silica");
const harness = @import("harness.zig");

/// Microbenchmarks for core database operations
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var suite = harness.BenchmarkSuite.init(allocator, "Microbenchmarks");
    defer suite.deinit();

    const stdout = std.fs.File.stdout().writer();

    // Benchmark 1: Point lookup by primary key
    {
        var db = try setupTestDatabase(allocator);
        defer cleanupTestDatabase(allocator, &db);

        const BenchContext = struct {
            db_ptr: *silica.Database,
            fn bench(self: @This()) !void {
                var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
                defer arena.deinit();
                const alloc = arena.allocator();

                const query = "SELECT * FROM bench_table WHERE id = 500";
                var result_set = try self.db_ptr.execute(alloc, query);
                defer result_set.deinit();
                // Consume result
                _ = try result_set.next();
            }
        };
        const ctx = BenchContext{ .db_ptr = &db };
        const result = try harness.runBenchmark(allocator, "Point lookup (PK, cached)", 10000, 100, ctx.bench);

        try suite.add(result);

        // Check against PRD target: < 5 µs (cached)
        const passed = harness.BenchmarkSuite.checkTarget(result, 5.0);
        try stdout.print("✓ Point lookup target (< 5 µs): {s}\n", .{if (passed) "PASS" else "FAIL"});
    }

    // Benchmark 2: Sequential insert
    {
        var db = try setupTestDatabase(allocator);
        defer cleanupTestDatabase(allocator, &db);

        const start = std.time.nanoTimestamp();
        var i: usize = 0;
        while (i < 100000) : (i += 1) {
            var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
            defer arena.deinit();
            const alloc = arena.allocator();

            const query = try std.fmt.allocPrint(alloc, "INSERT INTO bench_table VALUES ({d}, 'row{d}')", .{ i + 1000, i });
            var result_set = try db.execute(alloc, query);
            result_set.deinit();
        }
        const end = std.time.nanoTimestamp();
        const elapsed_sec = @as(f64, @floatFromInt(end - start)) / 1_000_000_000.0;
        const rows_per_sec = 100000.0 / elapsed_sec;

        try stdout.print("Sequential insert: {d:.0} rows/sec\n", .{rows_per_sec});
        try stdout.print("✓ Insert target (> 100K rows/sec): {s}\n\n", .{if (rows_per_sec > 100000.0) "PASS" else "FAIL"});
    }

    // Benchmark 3: Range scan
    {
        var db = try setupTestDatabase(allocator);
        defer cleanupTestDatabase(allocator, &db);

        const RangeBenchContext = struct {
            db_ptr: *silica.Database,
            fn bench(self: @This()) !void {
                var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
                defer arena.deinit();
                const alloc = arena.allocator();

                const query = "SELECT * FROM bench_table WHERE id BETWEEN 100 AND 1100";
                var result_set = try self.db_ptr.execute(alloc, query);
                defer result_set.deinit();

                // Consume all rows
                var count: usize = 0;
                while (try result_set.next()) |_| {
                    count += 1;
                }
            }
        };
        const range_ctx = RangeBenchContext{ .db_ptr = &db };
        const result = try harness.runBenchmark(allocator, "Range scan (1000 rows)", 1000, 10, range_ctx.bench);

        try suite.add(result);

        // Estimate throughput: 1000 rows per operation
        const rows_per_sec = 1000.0 * result.ops_per_sec;
        try stdout.print("Range scan throughput: {d:.0} rows/sec\n", .{rows_per_sec});
        try stdout.print("✓ Range scan target (> 500K rows/sec): {s}\n\n", .{if (rows_per_sec > 500000.0) "PASS" else "FAIL"});
    }

    // Print summary
    try suite.print(stdout);
}

fn setupTestDatabase(allocator: std.mem.Allocator) !silica.Database {
    // Create temporary test database
    const db_path = "bench_test.db";
    std.fs.cwd().deleteFile(db_path) catch {};

    var db = try silica.Database.init(allocator, db_path, .{});

    // Create test table
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const create_table = "CREATE TABLE bench_table (id INTEGER PRIMARY KEY, name TEXT)";
    var result = try db.execute(alloc, create_table);
    result.deinit();

    // Populate with 1000 rows for warmup
    var i: usize = 0;
    while (i < 1000) : (i += 1) {
        var stmt_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer stmt_arena.deinit();
        const stmt_alloc = stmt_arena.allocator();

        const insert = try std.fmt.allocPrint(stmt_alloc, "INSERT INTO bench_table VALUES ({d}, 'row{d}')", .{ i, i });
        var insert_result = try db.execute(stmt_alloc, insert);
        insert_result.deinit();
    }

    return db;
}

fn cleanupTestDatabase(_: std.mem.Allocator, db: *silica.Database) void {
    db.deinit();
    std.fs.cwd().deleteFile("bench_test.db") catch {};
    std.fs.cwd().deleteFile("bench_test.db-wal") catch {};
}
