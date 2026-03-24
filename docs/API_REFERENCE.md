# Silica API Reference

Comprehensive reference for the Silica Zig embedded API.

---

## Table of Contents

- [Database](#database)
  - [Database.open()](#databaseopen)
  - [Database.close()](#databaseclose)
  - [Database.exec()](#databaseexec)
  - [Database.prepare()](#databaseprepare)
  - [Database.begin()](#databasebegin)
  - [Database.commit()](#databasecommit)
  - [Database.rollback()](#databaserollback)
  - [Database.savepoint()](#databasesavepoint)
  - [Database.releaseSavepoint()](#databasereleasesavepoint)
  - [Database.rollbackToSavepoint()](#databaserollbacktosavepoint)
- [PreparedStatement](#preparedstatement)
  - [PreparedStatement.bind()](#preparedstatementbind)
  - [PreparedStatement.execute()](#preparedstatementexecute)
  - [PreparedStatement.close()](#preparedstatementclose)
- [QueryResult](#queryresult)
  - [QueryResult.deinit()](#queryresultdeinit)
- [RowIterator](#rowiterator)
  - [RowIterator.next()](#rowiteratornext)
- [Row](#row)
  - [Row.deinit()](#rowdeinit)
- [Value](#value)
- [Isolation Levels](#isolation-levels)
- [Error Types](#error-types)
- [Configuration](#configuration)
- [C FFI API](#c-ffi-api)

---

## Database

The `Database` struct is the main entry point for embedded database access. It manages connections, transactions, and query execution.

### Database.open()

Opens or creates a database file.

```zig
pub fn open(
    allocator: std.mem.Allocator,
    path: []const u8,
    options: OpenOptions,
) !Database
```

**Parameters:**
- `allocator`: Memory allocator for database operations
- `path`: Database file path (use `":memory:"` for in-memory database)
- `options`: Configuration options

**OpenOptions:**
```zig
pub const OpenOptions = struct {
    page_size: u32 = 4096,           // Page size in bytes (512-65536)
    cache_size: u32 = 2000,          // Buffer pool size (pages)
    wal_enabled: bool = true,        // Enable write-ahead log
    isolation_level: IsolationLevel = .read_committed,
};
```

**Returns:** `Database` instance

**Errors:**
- `error.FileNotFound` — Database file doesn't exist and creation failed
- `error.InvalidPageSize` — Page size not power of 2 or out of range
- `error.CorruptDatabase` — Database file header invalid
- `error.OutOfMemory` — Allocation failed

**Example:**
```zig
var db = try Database.open(allocator, "myapp.db", .{
    .page_size = 8192,
    .cache_size = 5000,
    .isolation_level = .repeatable_read,
});
defer db.close();
```

---

### Database.close()

Closes the database, flushing pending writes and releasing resources.

```zig
pub fn close(self: *Database) void
```

**Notes:**
- Automatically commits active transaction (if any)
- Flushes buffer pool dirty pages to disk
- Closes WAL file
- Safe to call multiple times (idempotent)

**Example:**
```zig
db.close();
```

---

### Database.exec()

Executes a SQL statement and returns results.

```zig
pub fn exec(
    self: *Database,
    sql: []const u8,
    params: anytype,
) !QueryResult
```

**Parameters:**
- `sql`: SQL statement (supports `?` placeholders)
- `params`: Tuple of parameter values (must match placeholder count)

**Returns:** `QueryResult` containing rows (for SELECT) or affected row count (for DML)

**Errors:**
- `error.SyntaxError` — SQL parsing failed
- `error.UnknownTable` — Table doesn't exist
- `error.UnknownColumn` — Column doesn't exist
- `error.TypeMismatch` — Parameter type doesn't match expected type
- `error.UniqueViolation` — Unique constraint violated
- `error.ForeignKeyViolation` — Foreign key constraint violated
- `error.LockConflict` — Row locked by another transaction
- `error.SerializationFailure` — Transaction conflict (retry required)

**Example:**
```zig
// DDL
_ = try db.exec("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)", .{});

// INSERT
_ = try db.exec("INSERT INTO users VALUES (?, ?)", .{ 1, "Alice" });

// SELECT
var result = try db.exec("SELECT id, name FROM users WHERE id = ?", .{1});
defer result.deinit();

if (result.rows) |*rows| {
    while (try rows.next()) |row| {
        defer row.deinit();
        const id = row.values[0].integer;
        const name = row.values[1].text;
        std.debug.print("User: {}, {s}\n", .{ id, name });
    }
}

// UPDATE
const update_result = try db.exec("UPDATE users SET name = ? WHERE id = ?", .{ "Bob", 1 });
defer update_result.deinit();
std.debug.print("Updated {} rows\n", .{update_result.rows_affected});
```

---

### Database.prepare()

Prepares a SQL statement for repeated execution with different parameters (46x-52x faster than `exec()`).

```zig
pub fn prepare(
    self: *Database,
    sql: []const u8,
) !PreparedStatement
```

**Parameters:**
- `sql`: SQL statement with `?` placeholders

**Returns:** `PreparedStatement` handle

**Errors:** Same as `exec()` (syntax errors detected at prepare time)

**Example:**
```zig
var stmt = try db.prepare("INSERT INTO users VALUES (?, ?)");
defer stmt.close();

try stmt.bind(0, .{ .integer = 1 });
try stmt.bind(1, .{ .text = "Alice" });
_ = try stmt.execute();

try stmt.bind(0, .{ .integer = 2 });
try stmt.bind(1, .{ .text = "Bob" });
_ = try stmt.execute();
```

---

### Database.begin()

Starts a new transaction.

```zig
pub fn begin(self: *Database, isolation_level: ?IsolationLevel) !void
```

**Parameters:**
- `isolation_level`: Override default isolation level for this transaction (optional)

**Errors:**
- `error.TransactionAlreadyActive` — Cannot nest transactions (use savepoints instead)

**Example:**
```zig
try db.begin(.serializable);
defer db.rollback() catch {};  // Ensure cleanup on error

try db.exec("INSERT INTO accounts VALUES (1, 100)", .{});
try db.exec("INSERT INTO accounts VALUES (2, 200)", .{});

try db.commit();
```

---

### Database.commit()

Commits the active transaction, making changes durable.

```zig
pub fn commit(self: *Database) !void
```

**Errors:**
- `error.NoActiveTransaction` — No transaction to commit
- `error.SerializationFailure` — Conflict detected (SERIALIZABLE only)

**Example:**
```zig
try db.begin(null);
_ = try db.exec("UPDATE accounts SET balance = balance - 100 WHERE id = 1", .{});
try db.commit();
```

---

### Database.rollback()

Aborts the active transaction, discarding all changes.

```zig
pub fn rollback(self: *Database) !void
```

**Errors:**
- `error.NoActiveTransaction` — No transaction to rollback

**Example:**
```zig
try db.begin(null);
_ = try db.exec("DELETE FROM users", .{});
try db.rollback();  // No changes persisted
```

---

### Database.savepoint()

Creates a savepoint within a transaction (allows partial rollback).

```zig
pub fn savepoint(self: *Database, name: []const u8) !void
```

**Parameters:**
- `name`: Savepoint identifier

**Errors:**
- `error.NoActiveTransaction` — Must be called within a transaction

**Example:**
```zig
try db.begin(null);
_ = try db.exec("INSERT INTO users VALUES (1, 'Alice')", .{});

try db.savepoint("sp1");
_ = try db.exec("INSERT INTO users VALUES (2, 'Bob')", .{});

try db.rollbackToSavepoint("sp1");  // Bob not inserted
try db.commit();  // Only Alice committed
```

---

### Database.releaseSavepoint()

Releases a savepoint, keeping its changes but freeing resources.

```zig
pub fn releaseSavepoint(self: *Database, name: []const u8) !void
```

**Parameters:**
- `name`: Savepoint identifier

**Errors:**
- `error.UnknownSavepoint` — Savepoint doesn't exist

---

### Database.rollbackToSavepoint()

Rolls back to a savepoint, discarding changes made after it.

```zig
pub fn rollbackToSavepoint(self: *Database, name: []const u8) !void
```

**Parameters:**
- `name`: Savepoint identifier

**Errors:**
- `error.UnknownSavepoint` — Savepoint doesn't exist

---

## PreparedStatement

Represents a pre-compiled SQL statement with parameter placeholders.

### PreparedStatement.bind()

Binds a parameter value to a placeholder position.

```zig
pub fn bind(self: *PreparedStatement, index: u32, value: Value) !void
```

**Parameters:**
- `index`: Zero-based placeholder index
- `value`: Parameter value (tagged union)

**Errors:**
- `error.IndexOutOfBounds` — Invalid parameter index
- `error.TypeMismatch` — Value type doesn't match expected type

**Example:**
```zig
var stmt = try db.prepare("SELECT * FROM users WHERE id = ? AND name = ?");
defer stmt.close();

try stmt.bind(0, .{ .integer = 42 });
try stmt.bind(1, .{ .text = "Alice" });

var result = try stmt.execute();
defer result.deinit();
```

---

### PreparedStatement.execute()

Executes the prepared statement with bound parameters.

```zig
pub fn execute(self: *PreparedStatement) !QueryResult
```

**Returns:** `QueryResult` (same as `Database.exec()`)

**Errors:** Same as `Database.exec()` (runtime errors)

**Example:**
```zig
var stmt = try db.prepare("INSERT INTO logs (timestamp, message) VALUES (?, ?)");
defer stmt.close();

for (events) |event| {
    try stmt.bind(0, .{ .integer = event.timestamp });
    try stmt.bind(1, .{ .text = event.message });
    _ = try stmt.execute();
}
```

---

### PreparedStatement.close()

Frees resources associated with the prepared statement.

```zig
pub fn close(self: *PreparedStatement) void
```

**Notes:**
- Must be called to avoid memory leaks
- Safe to call multiple times (idempotent)

---

## QueryResult

Represents the result of a SQL query (rows or affected row count).

```zig
pub const QueryResult = struct {
    rows: ?RowIterator,      // SELECT results (null for DML)
    rows_affected: u64,       // DML affected row count (0 for SELECT)
    columns: []const []const u8,  // Column names (SELECT only)
};
```

### QueryResult.deinit()

Frees memory associated with the query result.

```zig
pub fn deinit(self: *QueryResult) void
```

**Example:**
```zig
var result = try db.exec("SELECT * FROM users", .{});
defer result.deinit();
```

---

## RowIterator

Iterator for result rows (cursor).

### RowIterator.next()

Fetches the next row from the result set.

```zig
pub fn next(self: *RowIterator) !?Row
```

**Returns:**
- `Row` if more rows available
- `null` if end of result set

**Errors:**
- `error.IoError` — Disk read failed

**Example:**
```zig
var result = try db.exec("SELECT id, name FROM users", .{});
defer result.deinit();

if (result.rows) |*rows| {
    while (try rows.next()) |row| {
        defer row.deinit();
        std.debug.print("id={}, name={s}\n", .{ row.values[0].integer, row.values[1].text });
    }
}
```

---

## Row

Represents a single result row.

```zig
pub const Row = struct {
    values: []Value,  // Column values (array length = column count)
};
```

### Row.deinit()

Frees memory for the row (required for TEXT/BLOB values).

```zig
pub fn deinit(self: *Row) void
```

**Example:**
```zig
while (try rows.next()) |row| {
    defer row.deinit();  // MUST call to avoid leaks
    // ... use row
}
```

---

## Value

Tagged union representing a SQL value.

```zig
pub const Value = union(enum) {
    null,
    integer: i64,
    real: f64,
    text: []const u8,
    blob: []const u8,
    boolean: bool,
};
```

**Access Pattern:**
```zig
switch (value) {
    .null => std.debug.print("NULL\n", .{}),
    .integer => |n| std.debug.print("{}\n", .{n}),
    .real => |f| std.debug.print("{d}\n", .{f}),
    .text => |s| std.debug.print("{s}\n", .{s}),
    .blob => |b| std.debug.print("BLOB({} bytes)\n", .{b.len}),
    .boolean => |b| std.debug.print("{}\n", .{b}),
}
```

---

## Isolation Levels

```zig
pub const IsolationLevel = enum {
    read_uncommitted,  // No isolation (dirty reads allowed)
    read_committed,    // Per-statement snapshot (prevents dirty reads)
    repeatable_read,   // Per-transaction snapshot (prevents non-repeatable reads)
    serializable,      // SSI conflict detection (prevents all anomalies)
};
```

**Guarantees:**

| Level | Dirty Read | Non-Repeatable Read | Phantom Read | Write Skew | Lost Update |
|-------|-----------|---------------------|--------------|------------|-------------|
| READ UNCOMMITTED | ❌ Allowed | ❌ Allowed | ❌ Allowed | ❌ Allowed | ❌ Allowed |
| READ COMMITTED | ✅ Prevented | ❌ Allowed | ❌ Allowed | ❌ Allowed | ⚠️ Use SELECT FOR UPDATE |
| REPEATABLE READ | ✅ Prevented | ✅ Prevented | ❌ Allowed | ❌ Allowed | ⚠️ Use SELECT FOR UPDATE |
| SERIALIZABLE | ✅ Prevented | ✅ Prevented | ✅ Prevented | ✅ Prevented | ✅ Prevented |

**Known Issues:**
- **Issue #16**: Lost Update in READ COMMITTED — `UPDATE SET col = col + 1` may lose concurrent updates (workaround: use REPEATABLE READ or SELECT FOR UPDATE)
- **Issue #15**: SSI not fully implemented — SERIALIZABLE may miss some write skew anomalies

---

## Error Types

```zig
pub const DatabaseError = error{
    // File I/O
    FileNotFound,
    PermissionDenied,
    IoError,

    // Database format
    CorruptDatabase,
    InvalidPageSize,
    IncompatibleVersion,

    // SQL parsing
    SyntaxError,
    UnknownTable,
    UnknownColumn,
    UnknownFunction,

    // Type system
    TypeMismatch,
    InvalidCast,

    // Constraints
    UniqueViolation,
    ForeignKeyViolation,
    NotNullViolation,
    CheckViolation,

    // Transactions
    NoActiveTransaction,
    TransactionAlreadyActive,
    LockConflict,
    DeadlockDetected,
    SerializationFailure,
    UnknownSavepoint,

    // Prepared statements
    IndexOutOfBounds,
    ParameterNotBound,

    // System
    OutOfMemory,
    Overflow,
};
```

**Retry Logic (Serialization Failures):**
```zig
const max_retries = 3;
var attempt: u32 = 0;

while (attempt < max_retries) : (attempt += 1) {
    try db.begin(.serializable);

    db.exec("UPDATE accounts SET balance = balance - 100 WHERE id = 1", .{}) catch |err| {
        try db.rollback();
        if (err == error.SerializationFailure and attempt < max_retries - 1) {
            std.time.sleep(100 * std.time.ns_per_ms);  // Backoff
            continue;
        }
        return err;
    };

    try db.commit();
    break;
}
```

---

## Configuration

Runtime configuration via SQL:

```zig
// Set parameter
_ = try db.exec("SET work_mem = '64MB'", .{});

// Show current value
var result = try db.exec("SHOW work_mem", .{});
defer result.deinit();

// Reset to default
_ = try db.exec("RESET work_mem", .{});

// Show all parameters
var all = try db.exec("SHOW ALL", .{});
defer all.deinit();
```

**Available Parameters:**
- `work_mem` — Memory for sort/hash operations (default: 4MB)
- `max_connections` — Max concurrent connections (default: 100, server mode only)
- `statement_timeout` — Query timeout in milliseconds (default: 0 = disabled)
- `search_path` — Schema search order (default: "public")
- `application_name` — Client application identifier

See [CONFIGURATION.md](CONFIGURATION.md) for details.

---

## C FFI API

Silica provides a C-compatible API for interoperability.

### Example (C)

```c
#include "silica.h"

int main(void) {
    // Open database
    silica_db_t *db = silica_open("mydb.db", NULL);
    if (!db) {
        fprintf(stderr, "Failed to open database\n");
        return 1;
    }

    // Execute SQL
    char *errmsg = NULL;
    int rc = silica_exec(db, "CREATE TABLE users (id INTEGER, name TEXT)", &errmsg);
    if (rc != SILICA_OK) {
        fprintf(stderr, "Error: %s\n", errmsg);
        silica_free(errmsg);
        silica_close(db);
        return 1;
    }

    // Prepared statement
    silica_stmt_t *stmt = silica_prepare(db, "INSERT INTO users VALUES (?, ?)", &errmsg);
    if (!stmt) {
        fprintf(stderr, "Prepare error: %s\n", errmsg);
        silica_free(errmsg);
        silica_close(db);
        return 1;
    }

    silica_bind_int(stmt, 0, 1);
    silica_bind_text(stmt, 1, "Alice", -1);
    silica_execute(stmt);

    silica_finalize(stmt);
    silica_close(db);
    return 0;
}
```

### C API Functions

```c
// Database management
silica_db_t* silica_open(const char *path, const char *options);
void silica_close(silica_db_t *db);

// Direct execution
int silica_exec(silica_db_t *db, const char *sql, char **errmsg);

// Prepared statements
silica_stmt_t* silica_prepare(silica_db_t *db, const char *sql, char **errmsg);
int silica_bind_int(silica_stmt_t *stmt, int index, int64_t value);
int silica_bind_double(silica_stmt_t *stmt, int index, double value);
int silica_bind_text(silica_stmt_t *stmt, int index, const char *value, int len);
int silica_bind_blob(silica_stmt_t *stmt, int index, const void *value, int len);
int silica_bind_null(silica_stmt_t *stmt, int index);

int silica_execute(silica_stmt_t *stmt);
int silica_step(silica_stmt_t *stmt);  // SILICA_ROW, SILICA_DONE, or error
int silica_finalize(silica_stmt_t *stmt);

// Column access
int silica_column_count(silica_stmt_t *stmt);
const char* silica_column_name(silica_stmt_t *stmt, int index);
int silica_column_type(silica_stmt_t *stmt, int index);
int64_t silica_column_int(silica_stmt_t *stmt, int index);
double silica_column_double(silica_stmt_t *stmt, int index);
const char* silica_column_text(silica_stmt_t *stmt, int index);
const void* silica_column_blob(silica_stmt_t *stmt, int index);
int silica_column_bytes(silica_stmt_t *stmt, int index);

// Transactions
int silica_begin(silica_db_t *db);
int silica_commit(silica_db_t *db);
int silica_rollback(silica_db_t *db);

// Memory management
void silica_free(void *ptr);

// Return codes
#define SILICA_OK 0
#define SILICA_ERROR 1
#define SILICA_ROW 100
#define SILICA_DONE 101

// Type codes
#define SILICA_INTEGER 1
#define SILICA_REAL 2
#define SILICA_TEXT 3
#define SILICA_BLOB 4
#define SILICA_NULL 5
```

**Build C FFI:**
```bash
zig build-lib src/ffi.zig -dynamic -OReleaseSafe -femit-h=silica.h
```

---

## Performance Tips

1. **Use Prepared Statements**: 46x-52x faster than `exec()` for repeated queries
2. **Batch Inserts in Transactions**: Wrap multiple INSERTs in a single transaction
3. **Index Hot Columns**: Create indexes on columns used in WHERE/JOIN clauses
4. **Tune work_mem**: Increase for complex sorts/aggregates (default 4MB)
5. **Use COPY for Bulk Loads**: Faster than individual INSERTs (server mode)
6. **VACUUM Regularly**: Reclaim dead tuple space after large UPDATEs/DELETEs

**Batch Insert Example:**
```zig
try db.begin(null);
defer db.rollback() catch {};

var stmt = try db.prepare("INSERT INTO logs VALUES (?, ?)");
defer stmt.close();

for (0..10000) |i| {
    try stmt.bind(0, .{ .integer = @intCast(i) });
    try stmt.bind(1, .{ .text = "log entry" });
    _ = try stmt.execute();
}

try db.commit();  // 10K inserts in one transaction
```

---

## Next Steps

- **[Getting Started Guide](GETTING_STARTED.md)** — Comprehensive tutorial
- **[SQL Reference](SQL_REFERENCE.md)** — Supported SQL syntax
- **[Operations Guide](OPERATIONS_GUIDE.md)** — Backup, restore, monitoring
