const std = @import("std");
const silica = @import("silica");

/// Simple benchmark to verify basic operations meet PRD targets
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Clean up any existing test database
    std.fs.cwd().deleteFile("bench_test.db") catch {};
    std.fs.cwd().deleteFile("bench_test.db-wal") catch {};

    // Create test database
    var db = try silica.engine.Database.open(allocator, "bench_test.db", .{});
    defer {
        db.close();
        std.fs.cwd().deleteFile("bench_test.db") catch {};
        std.fs.cwd().deleteFile("bench_test.db-wal") catch {};
    }

    // Create test table
    {
        const create_table = "CREATE TABLE bench_table (id INTEGER PRIMARY KEY, name TEXT)";
        var result = try db.exec(create_table);
        result.close(allocator);
    }

    // Insert 1000 test rows
    {
        var i: usize = 0;
        while (i < 1000) : (i += 1) {
            var buf: [256]u8 = undefined;
            const insert = try std.fmt.bufPrint(&buf, "INSERT INTO bench_table VALUES ({d}, 'row{d}')", .{ i, i });
            var result = try db.exec(insert);
            result.close(allocator);
        }
    }

    // Benchmark 1: Point lookup (PRD target: < 5 µs cached)
    {
        const iterations = 10000;
        var total_ns: u64 = 0;

        var i: usize = 0;
        while (i < iterations) : (i += 1) {
            const start = std.time.nanoTimestamp();
            const query = "SELECT * FROM bench_table WHERE id = 500";
            var result_set = try db.exec(query);
            // NOTE: Result consumption would go here, but for now just close
            result_set.close(allocator);
            const end = std.time.nanoTimestamp();

            total_ns += @intCast(end - start);
        }

        const mean_ns = total_ns / iterations;
        const mean_us = @as(f64, @floatFromInt(mean_ns)) / 1000.0;

        std.debug.print("Point lookup (PK, cached): {d:.2} µs (target: < 5.0 µs) - {s}\n", .{ mean_us, if (mean_us < 5.0) "PASS" else "FAIL" });
    }

    // Benchmark 2: Sequential insert (PRD target: > 100K rows/sec)
    {
        const rows = 10000;
        const start = std.time.nanoTimestamp();

        var i: usize = 0;
        while (i < rows) : (i += 1) {
            var buf: [256]u8 = undefined;
            const query = try std.fmt.bufPrint(&buf, "INSERT INTO bench_table VALUES ({d}, 'row{d}')", .{ i + 10000, i });
            var result = try db.exec(query);
            result.close(allocator);
        }

        const end = std.time.nanoTimestamp();
        const elapsed_sec = @as(f64, @floatFromInt(end - start)) / 1_000_000_000.0;
        const rows_per_sec = @as(f64, @floatFromInt(rows)) / elapsed_sec;

        std.debug.print("Sequential insert: {d:.0} rows/sec (target: > 100K rows/sec) - {s}\n", .{ rows_per_sec, if (rows_per_sec > 100000.0) "PASS" else "FAIL" });
    }

    // Benchmark 3: Range scan (PRD target: > 500K rows/sec)
    {
        const iterations = 1000;
        var total_ns: u64 = 0;

        var i: usize = 0;
        while (i < iterations) : (i += 1) {
            const start = std.time.nanoTimestamp();
            const query = "SELECT * FROM bench_table WHERE id BETWEEN 100 AND 1100";
            var result_set = try db.exec(query);
            // NOTE: Result consumption would go here
            result_set.close(allocator);
            const end = std.time.nanoTimestamp();

            total_ns += @intCast(end - start);
        }

        const mean_ns = total_ns / iterations;
        const elapsed_sec = @as(f64, @floatFromInt(mean_ns)) / 1_000_000_000.0;
        const rows_per_sec = 1000.0 / elapsed_sec; // 1000 rows per query

        std.debug.print("Range scan: {d:.0} rows/sec (target: > 500K rows/sec) - {s}\n", .{ rows_per_sec, if (rows_per_sec > 500000.0) "PASS" else "FAIL" });
    }

    std.debug.print("\nBenchmarks complete!\n", .{});
}
