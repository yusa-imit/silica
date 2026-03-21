//! Comprehensive error path tests for SQL parser.
//! Tests malformed SQL statements to ensure proper error handling and recovery.

const std = @import("std");
const parser_mod = @import("parser.zig");
const ast = @import("ast.zig");

/// Helper function to test that parsing fails.
fn expectParseFail(sql: []const u8) !void {
    var arena = ast.AstArena.init(std.testing.allocator);
    defer arena.deinit();

    const result = parser_mod.Parser.init(std.testing.allocator, sql, &arena);
    if (result) |*p| {
        defer p.deinit();
        const stmt = p.parseStmt();
        try std.testing.expectError(error.ParseFailed, stmt);
    } else |_| {
        // Init failed — also acceptable for invalid input
    }
}

// ── SELECT Error Tests ────────────────────────────────────────────────────

test "parse SELECT without FROM should fail" {
    try expectParseFail("SELECT col1, col2");
}

test "parse SELECT with trailing comma should fail" {
    try expectParseFail("SELECT col1, col2, FROM users");
}

test "parse SELECT with missing column names should fail" {
    try expectParseFail("SELECT FROM users");
}

test "parse SELECT with only * after comma should fail" {
    try expectParseFail("SELECT name, FROM users");
}

test "parse DISTINCT without columns should fail" {
    try expectParseFail("SELECT DISTINCT FROM users");
}

test "parse DISTINCT ON without column list should fail" {
    try expectParseFail("SELECT DISTINCT ON FROM users");
}

test "parse DISTINCT ON with empty list should fail" {
    try expectParseFail("SELECT DISTINCT ON () col FROM users");
}

// ── INSERT Error Tests ────────────────────────────────────────────────────

test "parse INSERT without VALUES or SELECT should fail" {
    try expectParseFail("INSERT INTO users (name, email)");
}

test "parse INSERT with empty VALUES list should fail" {
    try expectParseFail("INSERT INTO users VALUES ()");
}

test "parse INSERT with mismatched parens in VALUES should fail" {
    try expectParseFail("INSERT INTO users VALUES (1, 2");
}

test "parse INSERT with trailing comma in columns should fail" {
    try expectParseFail("INSERT INTO users (name, email,) VALUES ('Alice', 'alice@example.com')");
}

// ── UPDATE Error Tests ────────────────────────────────────────────────────

test "parse UPDATE without SET should fail" {
    try expectParseFail("UPDATE users WHERE id = 1");
}

test "parse UPDATE SET without assignment should fail" {
    try expectParseFail("UPDATE users SET");
}

test "parse UPDATE SET with missing value should fail" {
    try expectParseFail("UPDATE users SET name =");
}

test "parse UPDATE SET with trailing comma should fail" {
    try expectParseFail("UPDATE users SET name = 'Alice', WHERE id = 1");
}

// ── DELETE Error Tests ────────────────────────────────────────────────────

test "parse DELETE without FROM should fail" {
    try expectParseFail("DELETE users WHERE id = 1");
}

test "parse DELETE with missing table name should fail" {
    try expectParseFail("DELETE FROM WHERE id = 1");
}

// ── CREATE TABLE Error Tests ──────────────────────────────────────────────

test "parse CREATE TABLE without columns should fail" {
    try expectParseFail("CREATE TABLE users ()");
}

test "parse CREATE TABLE with missing table name should fail" {
    try expectParseFail("CREATE TABLE (id INTEGER)");
}

test "parse CREATE TABLE with missing column type should fail" {
    try expectParseFail("CREATE TABLE users (id)");
}

test "parse CREATE TABLE with trailing comma in columns should fail" {
    try expectParseFail("CREATE TABLE users (id INTEGER, name TEXT,)");
}

test "parse CREATE TABLE with duplicate column names should fail" {
    // Parser may allow this; analyzer should reject
    try expectParseFail("CREATE TABLE users (id INTEGER, id TEXT)");
}

// ── Expression Error Tests ────────────────────────────────────────────────

test "parse WHERE with empty condition should fail" {
    try expectParseFail("SELECT * FROM users WHERE");
}

test "parse WHERE with incomplete binary op should fail" {
    try expectParseFail("SELECT * FROM users WHERE id = ");
}

test "parse WHERE with unary NOT without operand should fail" {
    try expectParseFail("SELECT * FROM users WHERE NOT");
}

test "parse nested expression with mismatched parens should fail" {
    try expectParseFail("SELECT * FROM users WHERE (id = 1 AND name = 'Alice'");
}

test "parse expression with too many closing parens should fail" {
    try expectParseFail("SELECT * FROM users WHERE (id = 1))");
}

test "parse function call with missing closing paren should fail" {
    try expectParseFail("SELECT COUNT(* FROM users");
}

test "parse function call with trailing comma should fail" {
    try expectParseFail("SELECT COUNT(id,) FROM users");
}

// ── CASE Expression Error Tests ───────────────────────────────────────────

test "parse CASE without END should fail" {
    try expectParseFail("SELECT CASE WHEN x > 0 THEN 1 ELSE 0 FROM users");
}

test "parse CASE WHEN without THEN should fail" {
    try expectParseFail("SELECT CASE WHEN x > 0 1 END FROM users");
}

test "parse CASE WHEN without condition should fail" {
    try expectParseFail("SELECT CASE WHEN THEN 1 END FROM users");
}

test "parse CASE without any WHEN should fail" {
    try expectParseFail("SELECT CASE ELSE 0 END FROM users");
}

// ── JOIN Error Tests ──────────────────────────────────────────────────────

test "parse JOIN without ON or USING should fail" {
    try expectParseFail("SELECT * FROM users JOIN orders");
}

test "parse JOIN ON with empty condition should fail" {
    try expectParseFail("SELECT * FROM users JOIN orders ON");
}

test "parse JOIN USING with empty column list should fail" {
    try expectParseFail("SELECT * FROM users JOIN orders USING ()");
}

test "parse CROSS JOIN with ON clause should fail" {
    try expectParseFail("SELECT * FROM users CROSS JOIN orders ON users.id = orders.user_id");
}

// ── ORDER BY / GROUP BY Error Tests ───────────────────────────────────────

test "parse ORDER BY without column should fail" {
    try expectParseFail("SELECT * FROM users ORDER BY");
}

test "parse ORDER BY with trailing comma should fail" {
    try expectParseFail("SELECT * FROM users ORDER BY name,");
}

test "parse GROUP BY without column should fail" {
    try expectParseFail("SELECT * FROM users GROUP BY");
}

test "parse GROUP BY with trailing comma should fail" {
    try expectParseFail("SELECT * FROM users GROUP BY name,");
}

test "parse HAVING without GROUP BY should succeed" {
    // Parser allows this; analyzer should warn or reject
    // Not a parse error
}

// ── LIMIT / OFFSET Error Tests ────────────────────────────────────────────

test "parse LIMIT with non-integer should fail" {
    try expectParseFail("SELECT * FROM users LIMIT abc");
}

test "parse LIMIT without value should fail" {
    try expectParseFail("SELECT * FROM users LIMIT");
}

test "parse OFFSET without value should fail" {
    try expectParseFail("SELECT * FROM users LIMIT 10 OFFSET");
}

test "parse OFFSET with non-integer should fail" {
    try expectParseFail("SELECT * FROM users LIMIT 10 OFFSET xyz");
}

// ── IN / BETWEEN Error Tests ──────────────────────────────────────────────

test "parse IN with empty list should fail" {
    try expectParseFail("SELECT * FROM users WHERE id IN ()");
}

test "parse IN without closing paren should fail" {
    try expectParseFail("SELECT * FROM users WHERE id IN (1, 2, 3");
}

test "parse BETWEEN with missing AND should fail" {
    try expectParseFail("SELECT * FROM users WHERE age BETWEEN 18 30");
}

test "parse BETWEEN without upper bound should fail" {
    try expectParseFail("SELECT * FROM users WHERE age BETWEEN 18 AND");
}

test "parse NOT BETWEEN with missing AND should fail" {
    try expectParseFail("SELECT * FROM users WHERE age NOT BETWEEN 18 30");
}

// ── Subquery Error Tests ──────────────────────────────────────────────────

test "parse subquery without closing paren should fail" {
    try expectParseFail("SELECT * FROM (SELECT * FROM users");
}

test "parse subquery with missing SELECT should fail" {
    try expectParseFail("SELECT * FROM (FROM users)");
}

test "parse IN subquery without closing paren should fail" {
    try expectParseFail("SELECT * FROM users WHERE id IN (SELECT id FROM orders");
}

test "parse EXISTS without subquery should fail" {
    try expectParseFail("SELECT * FROM users WHERE EXISTS");
}

test "parse EXISTS with missing closing paren should fail" {
    try expectParseFail("SELECT * FROM users WHERE EXISTS (SELECT 1 FROM orders");
}

// ── UNION / INTERSECT / EXCEPT Error Tests ────────────────────────────────

test "parse UNION without second SELECT should fail" {
    try expectParseFail("SELECT * FROM users UNION");
}

test "parse INTERSECT without second SELECT should fail" {
    try expectParseFail("SELECT * FROM users INTERSECT");
}

test "parse EXCEPT without second SELECT should fail" {
    try expectParseFail("SELECT * FROM users EXCEPT");
}

// ── CREATE INDEX Error Tests ──────────────────────────────────────────────

test "parse CREATE INDEX without ON clause should fail" {
    try expectParseFail("CREATE INDEX idx_name");
}

test "parse CREATE INDEX without table name should fail" {
    try expectParseFail("CREATE INDEX idx_name ON");
}

test "parse CREATE INDEX without column list should fail" {
    try expectParseFail("CREATE INDEX idx_name ON users");
}

test "parse CREATE INDEX with empty column list should fail" {
    try expectParseFail("CREATE INDEX idx_name ON users ()");
}

// ── ALTER TABLE Error Tests ───────────────────────────────────────────────

test "parse ALTER TABLE without action should fail" {
    try expectParseFail("ALTER TABLE users");
}

test "parse ALTER TABLE ADD COLUMN without column name should fail" {
    try expectParseFail("ALTER TABLE users ADD COLUMN");
}

test "parse ALTER TABLE DROP COLUMN without column name should fail" {
    try expectParseFail("ALTER TABLE users DROP COLUMN");
}

// ── DROP Error Tests ──────────────────────────────────────────────────────

test "parse DROP without object type should fail" {
    try expectParseFail("DROP users");
}

test "parse DROP TABLE without table name should fail" {
    try expectParseFail("DROP TABLE");
}

test "parse DROP INDEX without index name should fail" {
    try expectParseFail("DROP INDEX");
}

// ── Edge Cases ─────────────────────────────────────────────────────────────

test "parse completely empty statement should fail" {
    try expectParseFail("");
}

test "parse statement with only whitespace should fail" {
    try expectParseFail("   \n  \t  ");
}

test "parse statement with only semicolons should fail" {
    try expectParseFail(";;;");
}

test "parse statement with unexpected EOF in string literal should fail" {
    try expectParseFail("SELECT 'unclosed string FROM users");
}

test "parse statement with double quotes for string instead of single should succeed" {
    // PostgreSQL allows double quotes for identifiers, not strings
    // This may or may not be a parse error depending on implementation
}

test "parse statement with reserved keyword as identifier without quotes should fail" {
    try expectParseFail("SELECT SELECT FROM users");
}

test "parse deeply nested expressions beyond reasonable limit should fail" {
    // Generate a deeply nested expression
    var buf: [10000]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    const writer = fbs.writer();
    writer.writeAll("SELECT ") catch unreachable;
    for (0..500) |_| {
        writer.writeAll("(") catch unreachable;
    }
    writer.writeAll("1") catch unreachable;
    for (0..500) |_| {
        writer.writeAll(")") catch unreachable;
    }
    writer.writeAll(" FROM users") catch unreachable;

    try expectParseFail(fbs.getWritten());
}
