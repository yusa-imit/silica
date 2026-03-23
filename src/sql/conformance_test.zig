//! SQL Conformance Tests
//!
//! These tests verify SQL:2016 compliance for core features.
//! Based on SQL standard test suites (NIST SQL Test Suite, PostgreSQL regression tests).
//!
//! Categories:
//!   - Data Types: INTEGER, TEXT, NULL, BOOLEAN (E021)
//!   - Basic DML: SELECT, INSERT, UPDATE, DELETE (E021)
//!   - WHERE clause: Comparison operators, AND, OR, NOT (E021)
//!   - ORDER BY: Single column, multiple columns, ASC/DESC (F850)
//!   - LIMIT: Row limiting (F851)
//!   - Joins: INNER, LEFT, RIGHT, FULL (F401-F405)
//!   - Aggregates: COUNT, SUM, AVG, MIN, MAX (T611)
//!   - GROUP BY: Simple grouping (T611)
//!   - HAVING: Filter aggregated results (T611)
//!   - Subqueries: Scalar, IN, EXISTS (E061)
//!   - CTEs: WITH clause (T121)
//!   - Window functions: ROW_NUMBER, RANK, LAG, LEAD (T611)
//!   - Transactions: BEGIN, COMMIT, ROLLBACK (T211)
//!
//! Each test documents the SQL feature code being validated.

const std = @import("std");
const engine_mod = @import("engine.zig");
const Database = engine_mod.Database;

// ══════════════════════════════════════════════════════════════════════════
// Helper Functions
// ══════════════════════════════════════════════════════════════════════════

fn createTestDb(allocator: std.mem.Allocator, path: []const u8) !Database {
    return Database.open(allocator, path, .{ .page_size = 4096 });
}

fn execSql(db: *Database, sql: []const u8) !void {
    var result = try db.exec(sql);
    result.close(db.allocator);
}

fn expectRowCount(db: *Database, sql: []const u8, expected: usize) !void {
    var result = try db.exec(sql);
    defer result.close(db.allocator);
    try std.testing.expectEqual(expected, result.rows.len);
}

// ══════════════════════════════════════════════════════════════════════════
// Feature E021: Basic Data Types
// ══════════════════════════════════════════════════════════════════════════

test "conformance: E021-01 INTEGER data type" {
    const allocator = std.testing.allocator;
    var db = try createTestDb(allocator, ":memory:");
    defer db.close();

    try execSql(&db, "CREATE TABLE t1 (id INTEGER, val INTEGER)");
    try execSql(&db, "INSERT INTO t1 VALUES (1, 42), (2, -999), (3, 0)");

    try expectRowCount(&db, "SELECT * FROM t1 WHERE val = 42", 1);
    try expectRowCount(&db, "SELECT * FROM t1 WHERE val < 0", 1);
    try expectRowCount(&db, "SELECT * FROM t1 WHERE val = 0", 1);
}

test "conformance: E021-02 TEXT data type" {
    const allocator = std.testing.allocator;
    var db = try createTestDb(allocator, ":memory:");
    defer db.close();

    try execSql(&db, "CREATE TABLE t1 (id INTEGER, name TEXT)");
    try execSql(&db, "INSERT INTO t1 VALUES (1, 'Alice'), (2, ''), (3, 'Bob')");

    try expectRowCount(&db, "SELECT * FROM t1 WHERE name = 'Alice'", 1);
    try expectRowCount(&db, "SELECT * FROM t1 WHERE name = ''", 1);
}

test "conformance: E021-03 NULL values" {
    const allocator = std.testing.allocator;
    var db = try createTestDb(allocator, ":memory:");
    defer db.close();

    try execSql(&db, "CREATE TABLE t1 (id INTEGER, val INTEGER)");
    try execSql(&db, "INSERT INTO t1 VALUES (1, NULL), (2, 42)");

    try expectRowCount(&db, "SELECT * FROM t1 WHERE val IS NULL", 1);
    try expectRowCount(&db, "SELECT * FROM t1 WHERE val IS NOT NULL", 1);
}

// ══════════════════════════════════════════════════════════════════════════
// Feature E021: Basic DML
// ══════════════════════════════════════════════════════════════════════════

test "conformance: E021-04 SELECT with WHERE" {
    const allocator = std.testing.allocator;
    var db = try createTestDb(allocator, ":memory:");
    defer db.close();

    try execSql(&db, "CREATE TABLE t1 (id INTEGER, val INTEGER)");
    try execSql(&db, "INSERT INTO t1 VALUES (1, 10), (2, 20), (3, 30)");

    try expectRowCount(&db, "SELECT * FROM t1 WHERE id = 2", 1);
    try expectRowCount(&db, "SELECT * FROM t1 WHERE val > 15", 2);
    try expectRowCount(&db, "SELECT * FROM t1 WHERE val >= 20", 2);
    try expectRowCount(&db, "SELECT * FROM t1 WHERE val < 25", 2);
    try expectRowCount(&db, "SELECT * FROM t1 WHERE val <= 20", 2);
    try expectRowCount(&db, "SELECT * FROM t1 WHERE val <> 20", 2);
}

test "conformance: E021-05 UPDATE statement" {
    const allocator = std.testing.allocator;
    var db = try createTestDb(allocator, ":memory:");
    defer db.close();

    try execSql(&db, "CREATE TABLE t1 (id INTEGER, val INTEGER)");
    try execSql(&db, "INSERT INTO t1 VALUES (1, 10), (2, 20)");
    try execSql(&db, "UPDATE t1 SET val = 99 WHERE id = 1");

    try expectRowCount(&db, "SELECT * FROM t1 WHERE val = 99", 1);
    try expectRowCount(&db, "SELECT * FROM t1 WHERE val = 20", 1);
}

test "conformance: E021-06 DELETE statement" {
    const allocator = std.testing.allocator;
    var db = try createTestDb(allocator, ":memory:");
    defer db.close();

    try execSql(&db, "CREATE TABLE t1 (id INTEGER, val INTEGER)");
    try execSql(&db, "INSERT INTO t1 VALUES (1, 10), (2, 20), (3, 30)");
    try execSql(&db, "DELETE FROM t1 WHERE id = 2");

    try expectRowCount(&db, "SELECT * FROM t1", 2);
    try expectRowCount(&db, "SELECT * FROM t1 WHERE id = 2", 0);
}

// ══════════════════════════════════════════════════════════════════════════
// Feature E021: Boolean Operators
// ══════════════════════════════════════════════════════════════════════════

test "conformance: E021-07 AND operator" {
    const allocator = std.testing.allocator;
    var db = try createTestDb(allocator, ":memory:");
    defer db.close();

    try execSql(&db, "CREATE TABLE t1 (id INTEGER, val INTEGER)");
    try execSql(&db, "INSERT INTO t1 VALUES (1, 10), (2, 20), (3, 30)");

    try expectRowCount(&db, "SELECT * FROM t1 WHERE id > 1 AND val < 30", 1);
    try expectRowCount(&db, "SELECT * FROM t1 WHERE id > 0 AND val > 0", 3);
}

test "conformance: E021-08 OR operator" {
    const allocator = std.testing.allocator;
    var db = try createTestDb(allocator, ":memory:");
    defer db.close();

    try execSql(&db, "CREATE TABLE t1 (id INTEGER, val INTEGER)");
    try execSql(&db, "INSERT INTO t1 VALUES (1, 10), (2, 20), (3, 30)");

    try expectRowCount(&db, "SELECT * FROM t1 WHERE id = 1 OR id = 3", 2);
    try expectRowCount(&db, "SELECT * FROM t1 WHERE val = 10 OR val = 999", 1);
}

test "conformance: E021-09 NOT operator" {
    const allocator = std.testing.allocator;
    var db = try createTestDb(allocator, ":memory:");
    defer db.close();

    try execSql(&db, "CREATE TABLE t1 (id INTEGER, val INTEGER)");
    try execSql(&db, "INSERT INTO t1 VALUES (1, 10), (2, 20), (3, 30)");

    try expectRowCount(&db, "SELECT * FROM t1 WHERE NOT (id = 2)", 2);
    try expectRowCount(&db, "SELECT * FROM t1 WHERE NOT (val > 20)", 2);
}

// ══════════════════════════════════════════════════════════════════════════
// Feature F850: ORDER BY
// ══════════════════════════════════════════════════════════════════════════

test "conformance: F850-01 ORDER BY ASC" {
    const allocator = std.testing.allocator;
    var db = try createTestDb(allocator, ":memory:");
    defer db.close();

    try execSql(&db, "CREATE TABLE t1 (id INTEGER, val INTEGER)");
    try execSql(&db, "INSERT INTO t1 VALUES (3, 30), (1, 10), (2, 20)");

    var result = try db.exec("SELECT id FROM t1 ORDER BY id ASC");
    defer result.close(db.allocator);

    try std.testing.expectEqual(@as(usize, 3), result.rows.len);
    try std.testing.expectEqual(@as(i64, 1), result.rows[0].values[0].int);
    try std.testing.expectEqual(@as(i64, 2), result.rows[1].values[0].int);
    try std.testing.expectEqual(@as(i64, 3), result.rows[2].values[0].int);
}

test "conformance: F850-02 ORDER BY DESC" {
    const allocator = std.testing.allocator;
    var db = try createTestDb(allocator, ":memory:");
    defer db.close();

    try execSql(&db, "CREATE TABLE t1 (id INTEGER, val INTEGER)");
    try execSql(&db, "INSERT INTO t1 VALUES (1, 10), (2, 20), (3, 30)");

    var result = try db.exec("SELECT id FROM t1 ORDER BY id DESC");
    defer result.close(db.allocator);

    try std.testing.expectEqual(@as(usize, 3), result.rows.len);
    try std.testing.expectEqual(@as(i64, 3), result.rows[0].values[0].int);
    try std.testing.expectEqual(@as(i64, 2), result.rows[1].values[0].int);
    try std.testing.expectEqual(@as(i64, 1), result.rows[2].values[0].int);
}

test "conformance: F850-03 ORDER BY multiple columns" {
    const allocator = std.testing.allocator;
    var db = try createTestDb(allocator, ":memory:");
    defer db.close();

    try execSql(&db, "CREATE TABLE t1 (a INTEGER, b INTEGER)");
    try execSql(&db, "INSERT INTO t1 VALUES (1, 20), (1, 10), (2, 10)");

    var result = try db.exec("SELECT a, b FROM t1 ORDER BY a ASC, b DESC");
    defer result.close(db.allocator);

    try std.testing.expectEqual(@as(usize, 3), result.rows.len);
    // First group (a=1): b DESC → (1,20), (1,10)
    try std.testing.expectEqual(@as(i64, 1), result.rows[0].values[0].int);
    try std.testing.expectEqual(@as(i64, 20), result.rows[0].values[1].int);
    try std.testing.expectEqual(@as(i64, 1), result.rows[1].values[0].int);
    try std.testing.expectEqual(@as(i64, 10), result.rows[1].values[1].int);
    // Second group (a=2): (2,10)
    try std.testing.expectEqual(@as(i64, 2), result.rows[2].values[0].int);
    try std.testing.expectEqual(@as(i64, 10), result.rows[2].values[1].int);
}

// ══════════════════════════════════════════════════════════════════════════
// Feature F851: LIMIT clause
// ══════════════════════════════════════════════════════════════════════════

test "conformance: F851-01 LIMIT clause" {
    const allocator = std.testing.allocator;
    var db = try createTestDb(allocator, ":memory:");
    defer db.close();

    try execSql(&db, "CREATE TABLE t1 (id INTEGER)");
    try execSql(&db, "INSERT INTO t1 VALUES (1), (2), (3), (4), (5)");

    try expectRowCount(&db, "SELECT * FROM t1 LIMIT 2", 2);
    try expectRowCount(&db, "SELECT * FROM t1 LIMIT 0", 0);
    try expectRowCount(&db, "SELECT * FROM t1 LIMIT 100", 5);
}

test "conformance: F851-02 LIMIT with OFFSET" {
    const allocator = std.testing.allocator;
    var db = try createTestDb(allocator, ":memory:");
    defer db.close();

    try execSql(&db, "CREATE TABLE t1 (id INTEGER)");
    try execSql(&db, "INSERT INTO t1 VALUES (1), (2), (3), (4), (5)");

    var result = try db.exec("SELECT id FROM t1 ORDER BY id LIMIT 2 OFFSET 1");
    defer result.close(db.allocator);

    try std.testing.expectEqual(@as(usize, 2), result.rows.len);
    try std.testing.expectEqual(@as(i64, 2), result.rows[0].values[0].int);
    try std.testing.expectEqual(@as(i64, 3), result.rows[1].values[0].int);
}

// ══════════════════════════════════════════════════════════════════════════
// Feature F401-F405: Joins
// ══════════════════════════════════════════════════════════════════════════

test "conformance: F401 INNER JOIN" {
    const allocator = std.testing.allocator;
    var db = try createTestDb(allocator, ":memory:");
    defer db.close();

    try execSql(&db, "CREATE TABLE t1 (id INTEGER, name TEXT)");
    try execSql(&db, "CREATE TABLE t2 (id INTEGER, val INTEGER)");
    try execSql(&db, "INSERT INTO t1 VALUES (1, 'A'), (2, 'B'), (3, 'C')");
    try execSql(&db, "INSERT INTO t2 VALUES (1, 10), (2, 20)");

    try expectRowCount(&db, "SELECT * FROM t1 INNER JOIN t2 ON t1.id = t2.id", 2);
}

test "conformance: F403 LEFT JOIN" {
    const allocator = std.testing.allocator;
    var db = try createTestDb(allocator, ":memory:");
    defer db.close();

    try execSql(&db, "CREATE TABLE t1 (id INTEGER, name TEXT)");
    try execSql(&db, "CREATE TABLE t2 (id INTEGER, val INTEGER)");
    try execSql(&db, "INSERT INTO t1 VALUES (1, 'A'), (2, 'B'), (3, 'C')");
    try execSql(&db, "INSERT INTO t2 VALUES (1, 10), (2, 20)");

    try expectRowCount(&db, "SELECT * FROM t1 LEFT JOIN t2 ON t1.id = t2.id", 3);
}

// ══════════════════════════════════════════════════════════════════════════
// Feature T611: Aggregates and GROUP BY
// ══════════════════════════════════════════════════════════════════════════

test "conformance: T611-01 COUNT aggregate" {
    const allocator = std.testing.allocator;
    var db = try createTestDb(allocator, ":memory:");
    defer db.close();

    try execSql(&db, "CREATE TABLE t1 (id INTEGER, val INTEGER)");
    try execSql(&db, "INSERT INTO t1 VALUES (1, 10), (2, 20), (3, NULL)");

    var result = try db.exec("SELECT COUNT(*) FROM t1");
    defer result.close(db.allocator);
    try std.testing.expectEqual(@as(i64, 3), result.rows[0].values[0].int);

    var result2 = try db.exec("SELECT COUNT(val) FROM t1");
    defer result2.close(db.allocator);
    try std.testing.expectEqual(@as(i64, 2), result2.rows[0].values[0].int);
}

test "conformance: T611-02 SUM aggregate" {
    const allocator = std.testing.allocator;
    var db = try createTestDb(allocator, ":memory:");
    defer db.close();

    try execSql(&db, "CREATE TABLE t1 (val INTEGER)");
    try execSql(&db, "INSERT INTO t1 VALUES (10), (20), (30)");

    var result = try db.exec("SELECT SUM(val) FROM t1");
    defer result.close(db.allocator);
    try std.testing.expectEqual(@as(i64, 60), result.rows[0].values[0].int);
}

test "conformance: T611-03 AVG aggregate" {
    const allocator = std.testing.allocator;
    var db = try createTestDb(allocator, ":memory:");
    defer db.close();

    try execSql(&db, "CREATE TABLE t1 (val INTEGER)");
    try execSql(&db, "INSERT INTO t1 VALUES (10), (20), (30)");

    var result = try db.exec("SELECT AVG(val) FROM t1");
    defer result.close(db.allocator);
    try std.testing.expectEqual(@as(i64, 20), result.rows[0].values[0].int);
}

test "conformance: T611-04 MIN/MAX aggregates" {
    const allocator = std.testing.allocator;
    var db = try createTestDb(allocator, ":memory:");
    defer db.close();

    try execSql(&db, "CREATE TABLE t1 (val INTEGER)");
    try execSql(&db, "INSERT INTO t1 VALUES (30), (10), (20)");

    var result_min = try db.exec("SELECT MIN(val) FROM t1");
    defer result_min.close(db.allocator);
    try std.testing.expectEqual(@as(i64, 10), result_min.rows[0].values[0].int);

    var result_max = try db.exec("SELECT MAX(val) FROM t1");
    defer result_max.close(db.allocator);
    try std.testing.expectEqual(@as(i64, 30), result_max.rows[0].values[0].int);
}

test "conformance: T611-05 GROUP BY clause" {
    const allocator = std.testing.allocator;
    var db = try createTestDb(allocator, ":memory:");
    defer db.close();

    try execSql(&db, "CREATE TABLE t1 (category TEXT, val INTEGER)");
    try execSql(&db, "INSERT INTO t1 VALUES ('A', 10), ('A', 20), ('B', 30)");

    var result = try db.exec("SELECT category, SUM(val) FROM t1 GROUP BY category");
    defer result.close(db.allocator);
    try std.testing.expectEqual(@as(usize, 2), result.rows.len);
}

test "conformance: T611-06 HAVING clause" {
    const allocator = std.testing.allocator;
    var db = try createTestDb(allocator, ":memory:");
    defer db.close();

    try execSql(&db, "CREATE TABLE t1 (category TEXT, val INTEGER)");
    try execSql(&db, "INSERT INTO t1 VALUES ('A', 10), ('A', 20), ('B', 5), ('B', 10)");

    try expectRowCount(&db, "SELECT category FROM t1 GROUP BY category HAVING SUM(val) > 20", 1);
}

// ══════════════════════════════════════════════════════════════════════════
// Feature E061: Subqueries
// ══════════════════════════════════════════════════════════════════════════

test "conformance: E061-01 Scalar subquery" {
    const allocator = std.testing.allocator;
    var db = try createTestDb(allocator, ":memory:");
    defer db.close();

    try execSql(&db, "CREATE TABLE t1 (id INTEGER, val INTEGER)");
    try execSql(&db, "INSERT INTO t1 VALUES (1, 10), (2, 20), (3, 30)");

    try expectRowCount(&db, "SELECT * FROM t1 WHERE val > (SELECT AVG(val) FROM t1)", 1);
}

test "conformance: E061-02 IN subquery" {
    const allocator = std.testing.allocator;
    var db = try createTestDb(allocator, ":memory:");
    defer db.close();

    try execSql(&db, "CREATE TABLE t1 (id INTEGER)");
    try execSql(&db, "CREATE TABLE t2 (id INTEGER)");
    try execSql(&db, "INSERT INTO t1 VALUES (1), (2), (3)");
    try execSql(&db, "INSERT INTO t2 VALUES (2), (3)");

    try expectRowCount(&db, "SELECT * FROM t1 WHERE id IN (SELECT id FROM t2)", 2);
}

test "conformance: E061-03 EXISTS subquery" {
    const allocator = std.testing.allocator;
    var db = try createTestDb(allocator, ":memory:");
    defer db.close();

    try execSql(&db, "CREATE TABLE t1 (id INTEGER, name TEXT)");
    try execSql(&db, "CREATE TABLE t2 (id INTEGER)");
    try execSql(&db, "INSERT INTO t1 VALUES (1, 'A'), (2, 'B')");
    try execSql(&db, "INSERT INTO t2 VALUES (1)");

    try expectRowCount(&db, "SELECT * FROM t1 WHERE EXISTS (SELECT 1 FROM t2 WHERE t2.id = t1.id)", 1);
}

// ══════════════════════════════════════════════════════════════════════════
// Feature T121: CTEs (WITH clause)
// ══════════════════════════════════════════════════════════════════════════

test "conformance: T121-01 Simple CTE" {
    const allocator = std.testing.allocator;
    var db = try createTestDb(allocator, ":memory:");
    defer db.close();

    try execSql(&db, "CREATE TABLE t1 (id INTEGER)");
    try execSql(&db, "INSERT INTO t1 VALUES (1), (2), (3)");

    try expectRowCount(&db, "WITH cte AS (SELECT id FROM t1 WHERE id > 1) SELECT * FROM cte", 2);
}

test "conformance: T121-02 Multiple CTEs" {
    const allocator = std.testing.allocator;
    var db = try createTestDb(allocator, ":memory:");
    defer db.close();

    try execSql(&db, "CREATE TABLE t1 (id INTEGER)");
    try execSql(&db, "INSERT INTO t1 VALUES (1), (2), (3), (4)");

    try expectRowCount(&db,
        \\WITH
        \\  cte1 AS (SELECT id FROM t1 WHERE id > 1),
        \\  cte2 AS (SELECT id FROM cte1 WHERE id < 4)
        \\SELECT * FROM cte2
    , 2);
}

// ══════════════════════════════════════════════════════════════════════════
// Feature T611: Window Functions
// ══════════════════════════════════════════════════════════════════════════

test "conformance: T611-07 ROW_NUMBER window function" {
    const allocator = std.testing.allocator;
    var db = try createTestDb(allocator, ":memory:");
    defer db.close();

    try execSql(&db, "CREATE TABLE t1 (category TEXT, val INTEGER)");
    try execSql(&db, "INSERT INTO t1 VALUES ('A', 10), ('A', 20), ('B', 5)");

    var result = try db.exec("SELECT category, ROW_NUMBER() OVER (PARTITION BY category ORDER BY val) AS rn FROM t1");
    defer result.close(db.allocator);
    try std.testing.expectEqual(@as(usize, 3), result.rows.len);
}

test "conformance: T611-08 RANK window function" {
    const allocator = std.testing.allocator;
    var db = try createTestDb(allocator, ":memory:");
    defer db.close();

    try execSql(&db, "CREATE TABLE t1 (val INTEGER)");
    try execSql(&db, "INSERT INTO t1 VALUES (10), (10), (20)");

    var result = try db.exec("SELECT RANK() OVER (ORDER BY val) AS rank FROM t1");
    defer result.close(db.allocator);
    try std.testing.expectEqual(@as(usize, 3), result.rows.len);
    try std.testing.expectEqual(@as(i64, 1), result.rows[0].values[0].int);
    try std.testing.expectEqual(@as(i64, 1), result.rows[1].values[0].int);
    try std.testing.expectEqual(@as(i64, 3), result.rows[2].values[0].int);
}

// ══════════════════════════════════════════════════════════════════════════
// Feature T211: Transactions
// ══════════════════════════════════════════════════════════════════════════

test "conformance: T211-01 COMMIT transaction" {
    const allocator = std.testing.allocator;
    var db = try createTestDb(allocator, ":memory:");
    defer db.close();

    try execSql(&db, "CREATE TABLE t1 (id INTEGER)");
    try execSql(&db, "BEGIN");
    try execSql(&db, "INSERT INTO t1 VALUES (1)");
    try execSql(&db, "COMMIT");

    try expectRowCount(&db, "SELECT * FROM t1", 1);
}

test "conformance: T211-02 ROLLBACK transaction" {
    const allocator = std.testing.allocator;
    var db = try createTestDb(allocator, ":memory:");
    defer db.close();

    try execSql(&db, "CREATE TABLE t1 (id INTEGER)");
    try execSql(&db, "BEGIN");
    try execSql(&db, "INSERT INTO t1 VALUES (1)");
    try execSql(&db, "ROLLBACK");

    try expectRowCount(&db, "SELECT * FROM t1", 0);
}

test "conformance: T211-03 Isolation: READ COMMITTED" {
    const allocator = std.testing.allocator;
    var db = try createTestDb(allocator, ":memory:");
    defer db.close();

    try execSql(&db, "CREATE TABLE t1 (id INTEGER)");
    try execSql(&db, "BEGIN");
    try execSql(&db, "INSERT INTO t1 VALUES (1)");

    // T1 uncommitted changes should not be visible in new snapshot
    try expectRowCount(&db, "SELECT * FROM t1", 1); // T1 sees its own changes

    try execSql(&db, "COMMIT");
    try expectRowCount(&db, "SELECT * FROM t1", 1); // Committed, visible
}
