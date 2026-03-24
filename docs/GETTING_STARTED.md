# Getting Started with Silica

Complete tutorial for using Silica in embedded and server modes.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Part 1: Embedded Mode](#part-1-embedded-mode)
  - [Hello World](#hello-world)
  - [Creating Tables](#creating-tables)
  - [Inserting Data](#inserting-data)
  - [Querying Data](#querying-data)
  - [Updating and Deleting](#updating-and-deleting)
  - [Transactions](#transactions)
  - [Prepared Statements](#prepared-statements)
- [Part 2: Advanced Features](#part-2-advanced-features)
  - [Indexes](#indexes)
  - [JSON/JSONB](#jsonjsonb)
  - [Full-Text Search](#full-text-search)
  - [Views and CTEs](#views-and-ctes)
  - [Window Functions](#window-functions)
  - [Triggers](#triggers)
- [Part 3: Server Mode](#part-3-server-mode)
  - [Starting the Server](#starting-the-server)
  - [Connecting with psql](#connecting-with-psql)
  - [Client Libraries](#client-libraries)
  - [Replication](#replication)
- [Part 4: Operations](#part-4-operations)
  - [Backup and Restore](#backup-and-restore)
  - [Performance Tuning](#performance-tuning)
  - [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

- **Zig 0.15.x** or later ([download here](https://ziglang.org/download/))
- Basic knowledge of SQL
- Terminal/command line familiarity

**Verify Zig installation:**
```bash
zig version
# Expected: 0.15.0 or later
```

---

## Installation

### Option 1: From Source

```bash
git clone https://github.com/yusa-imit/silica.git
cd silica
zig build -Doptimize=ReleaseSafe
sudo cp zig-out/bin/silica /usr/local/bin/
```

**Verify:**
```bash
silica --version
# Expected: silica 0.3.0
```

### Option 2: Zig Package Manager

Add to your project's `build.zig.zon`:
```zig
.dependencies = .{
    .silica = .{
        .url = "https://github.com/yusa-imit/silica/archive/v0.3.0.tar.gz",
        .hash = "1220...",  // Run `zig fetch` to get hash
    },
},
```

Then in `build.zig`:
```zig
const silica = b.dependency("silica", .{});
exe.root_module.addImport("silica", silica.module("silica"));
```

---

## Part 1: Embedded Mode

Embedded mode runs Silica in-process — no separate server needed. Perfect for desktop apps, CLI tools, and embedded systems.

### Hello World

Create `hello.zig`:

```zig
const std = @import("std");
const silica = @import("silica");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Open in-memory database
    var db = try silica.Database.open(allocator, ":memory:", .{});
    defer db.close();

    // Create table
    _ = try db.exec("CREATE TABLE greetings (message TEXT)", .{});

    // Insert data
    _ = try db.exec("INSERT INTO greetings VALUES (?)", .{"Hello, Silica!"});

    // Query
    var result = try db.exec("SELECT message FROM greetings", .{});
    defer result.deinit();

    if (result.rows) |*rows| {
        while (try rows.next()) |row| {
            defer row.deinit();
            std.debug.print("{s}\n", .{row.values[0].text});
        }
    }
}
```

**Build and run:**
```bash
zig build-exe hello.zig --dep silica -Mroot=/path/to/silica/src/main.zig
./hello
# Output: Hello, Silica!
```

---

### Creating Tables

Tables are defined with `CREATE TABLE`:

```zig
const schema =
    \\CREATE TABLE users (
    \\    id INTEGER PRIMARY KEY,
    \\    username TEXT NOT NULL UNIQUE,
    \\    email TEXT,
    \\    created_at INTEGER DEFAULT (unixepoch())
    \\);

_ = try db.exec(schema, .{});
```

**Supported types:**
- `INTEGER` — 64-bit signed integer
- `REAL` — 64-bit float
- `TEXT` — UTF-8 string
- `BLOB` — Binary data
- `DATE`, `TIME`, `TIMESTAMP` — Date/time types
- `NUMERIC(p, s)` — Fixed-point decimal
- `UUID` — UUID v4
- `JSON`, `JSONB` — JSON data
- `ARRAY` — Arrays of any type

**Constraints:**
- `PRIMARY KEY` — Unique identifier, creates index
- `NOT NULL` — Disallow null values
- `UNIQUE` — Unique constraint, creates index
- `CHECK` — Custom validation expression
- `DEFAULT` — Default value
- `FOREIGN KEY` — Referential integrity

**Example with constraints:**
```zig
const posts_schema =
    \\CREATE TABLE posts (
    \\    id INTEGER PRIMARY KEY,
    \\    user_id INTEGER NOT NULL,
    \\    title TEXT NOT NULL CHECK (length(title) > 0),
    \\    content TEXT,
    \\    status TEXT DEFAULT 'draft' CHECK (status IN ('draft', 'published')),
    \\    views INTEGER DEFAULT 0 CHECK (views >= 0),
    \\    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    \\);

_ = try db.exec(posts_schema, .{});
```

---

### Inserting Data

**Single insert:**
```zig
_ = try db.exec("INSERT INTO users (id, username, email) VALUES (?, ?, ?)", .{
    1,
    "alice",
    "alice@example.com",
});
```

**Batch insert (with transaction):**
```zig
try db.begin(null);
defer db.rollback() catch {};  // Ensure cleanup on error

for (0..1000) |i| {
    _ = try db.exec("INSERT INTO users (id, username) VALUES (?, ?)", .{
        i,
        try std.fmt.allocPrint(allocator, "user{}", .{i}),
    });
}

try db.commit();
```

**Insert with RETURNING:**
```zig
var result = try db.exec(
    "INSERT INTO users (username, email) VALUES (?, ?) RETURNING id",
    .{ "bob", "bob@example.com" }
);
defer result.deinit();

if (result.rows) |*rows| {
    if (try rows.next()) |row| {
        defer row.deinit();
        const new_id = row.values[0].integer;
        std.debug.print("Inserted user with id={}\n", .{new_id});
    }
}
```

---

### Querying Data

**Simple SELECT:**
```zig
var result = try db.exec("SELECT id, username FROM users", .{});
defer result.deinit();

if (result.rows) |*rows| {
    while (try rows.next()) |row| {
        defer row.deinit();
        const id = row.values[0].integer;
        const username = row.values[1].text;
        std.debug.print("User {}: {s}\n", .{ id, username });
    }
}
```

**With WHERE clause:**
```zig
var result = try db.exec(
    "SELECT * FROM users WHERE username LIKE ?",
    .{"%alice%"}
);
defer result.deinit();
```

**JOIN query:**
```zig
const join_query =
    \\SELECT users.username, posts.title
    \\FROM users
    \\INNER JOIN posts ON users.id = posts.user_id
    \\WHERE posts.status = 'published'
    \\ORDER BY posts.created_at DESC
    \\LIMIT 10;

var result = try db.exec(join_query, .{});
defer result.deinit();
```

**Aggregates:**
```zig
var result = try db.exec(
    \\SELECT user_id, COUNT(*) as post_count, AVG(views) as avg_views
    \\FROM posts
    \\GROUP BY user_id
    \\HAVING COUNT(*) > 5
    \\ORDER BY post_count DESC
, .{});
defer result.deinit();
```

---

### Updating and Deleting

**UPDATE:**
```zig
var result = try db.exec(
    "UPDATE users SET email = ? WHERE id = ?",
    .{ "newemail@example.com", 1 }
);
defer result.deinit();

std.debug.print("Updated {} rows\n", .{result.rows_affected});
```

**DELETE:**
```zig
var result = try db.exec("DELETE FROM posts WHERE views < ?", .{10});
defer result.deinit();

std.debug.print("Deleted {} posts\n", .{result.rows_affected});
```

---

### Transactions

Transactions ensure ACID properties (Atomicity, Consistency, Isolation, Durability).

**Basic transaction:**
```zig
try db.begin(null);
defer db.rollback() catch {};  // Rollback on error

_ = try db.exec("UPDATE accounts SET balance = balance - 100 WHERE id = 1", .{});
_ = try db.exec("UPDATE accounts SET balance = balance + 100 WHERE id = 2", .{});

try db.commit();  // Both updates or neither
```

**Isolation levels:**
```zig
// READ COMMITTED (default)
try db.begin(.read_committed);

// REPEATABLE READ (prevents non-repeatable reads)
try db.begin(.repeatable_read);

// SERIALIZABLE (prevents all anomalies)
try db.begin(.serializable);
```

**Retry logic for serialization failures:**
```zig
const max_retries = 3;
var attempt: u32 = 0;

while (attempt < max_retries) : (attempt += 1) {
    try db.begin(.serializable);

    db.exec("UPDATE accounts SET balance = balance - 100 WHERE id = 1", .{}) catch |err| {
        try db.rollback();
        if (err == error.SerializationFailure and attempt < max_retries - 1) {
            std.time.sleep(100 * std.time.ns_per_ms);
            continue;
        }
        return err;
    };

    try db.commit();
    break;
}
```

**Savepoints:**
```zig
try db.begin(null);

_ = try db.exec("INSERT INTO users VALUES (1, 'alice')", .{});

try db.savepoint("sp1");
_ = try db.exec("INSERT INTO users VALUES (2, 'bob')", .{});

// Decide to rollback bob only
try db.rollbackToSavepoint("sp1");

try db.commit();  // Only alice committed
```

---

### Prepared Statements

Prepared statements are **46x-52x faster** than `exec()` for repeated queries.

**Example:**
```zig
var stmt = try db.prepare("INSERT INTO logs (timestamp, level, message) VALUES (?, ?, ?)");
defer stmt.close();

const log_entries = [_]struct { i64, []const u8, []const u8 }{
    .{ 1700000000, "INFO", "Server started" },
    .{ 1700000001, "WARN", "High memory usage" },
    .{ 1700000002, "ERROR", "Connection failed" },
};

for (log_entries) |entry| {
    try stmt.bind(0, .{ .integer = entry[0] });
    try stmt.bind(1, .{ .text = entry[1] });
    try stmt.bind(2, .{ .text = entry[2] });
    _ = try stmt.execute();
}
```

**SELECT with prepared statement:**
```zig
var stmt = try db.prepare("SELECT * FROM users WHERE age > ? AND city = ?");
defer stmt.close();

try stmt.bind(0, .{ .integer = 25 });
try stmt.bind(1, .{ .text = "New York" });

var result = try stmt.execute();
defer result.deinit();

if (result.rows) |*rows| {
    while (try rows.next()) |row| {
        defer row.deinit();
        // ... process row
    }
}
```

---

## Part 2: Advanced Features

### Indexes

Indexes speed up queries on specific columns.

**Create B+Tree index:**
```zig
_ = try db.exec("CREATE INDEX idx_users_email ON users(email)", .{});
```

**Unique index:**
```zig
_ = try db.exec("CREATE UNIQUE INDEX idx_users_username ON users(username)", .{});
```

**Multi-column index:**
```zig
_ = try db.exec("CREATE INDEX idx_posts_user_status ON posts(user_id, status)", .{});
```

**Hash index (exact match only):**
```zig
_ = try db.exec("CREATE INDEX idx_sessions_token USING HASH ON sessions(token)", .{});
```

**Concurrent index creation (no table locks):**
```zig
_ = try db.exec("CREATE INDEX CONCURRENTLY idx_large_table ON large_table(column)", .{});
```

**Drop index:**
```zig
_ = try db.exec("DROP INDEX idx_users_email", .{});
```

---

### JSON/JSONB

Silica supports JSON with binary storage (JSONB) and rich operators.

**Store JSON:**
```zig
const json_data =
    \\{"name": "Alice", "age": 30, "tags": ["developer", "zig"]}
;

_ = try db.exec(
    "INSERT INTO users (id, profile) VALUES (?, ?::jsonb)",
    .{ 1, json_data }
);
```

**Query JSON fields:**
```zig
// Extract text value
var result = try db.exec("SELECT profile->>'name' FROM users WHERE id = ?", .{1});
defer result.deinit();

// Extract nested value
var result2 = try db.exec("SELECT profile->'tags'->0 FROM users WHERE id = ?", .{1});
defer result2.deinit();
```

**JSON operators:**
- `->` — Extract JSON object/array (returns JSON)
- `->>` — Extract JSON object/array as text
- `@>` — Contains (JSONB only)
- `?` — Key exists (JSONB only)
- `?|` — Any key exists
- `?&` — All keys exist

**Example queries:**
```zig
// Find users with tag "developer"
var result = try db.exec(
    "SELECT * FROM users WHERE profile->'tags' @> '[\"developer\"]'::jsonb",
    .{}
);
defer result.deinit();

// Check if key exists
var result2 = try db.exec("SELECT * FROM users WHERE profile ? 'email'", .{});
defer result2.deinit();
```

**GIN index for fast JSON queries:**
```zig
_ = try db.exec("CREATE INDEX idx_users_profile ON users USING GIN (profile)", .{});
```

---

### Full-Text Search

Silica supports PostgreSQL-style full-text search with TSVECTOR/TSQUERY.

**Create FTS index:**
```zig
_ = try db.exec(
    \\ALTER TABLE posts ADD COLUMN search_vector TSVECTOR
    \\  GENERATED ALWAYS AS (to_tsvector('english', title || ' ' || content)) STORED
, .{});

_ = try db.exec("CREATE INDEX idx_posts_search ON posts USING GIN (search_vector)", .{});
```

**Search:**
```zig
var result = try db.exec(
    "SELECT title FROM posts WHERE search_vector @@ to_tsquery('english', ?)",
    .{"zig & database"}
);
defer result.deinit();
```

**Ranking:**
```zig
var result = try db.exec(
    \\SELECT title, ts_rank(search_vector, query) AS rank
    \\FROM posts, to_tsquery('english', 'zig & performance') query
    \\WHERE search_vector @@ query
    \\ORDER BY rank DESC
    \\LIMIT 10
, .{});
defer result.deinit();
```

---

### Views and CTEs

**Regular view:**
```zig
_ = try db.exec(
    \\CREATE VIEW active_users AS
    \\SELECT id, username, email
    \\FROM users
    \\WHERE last_login > unixepoch() - 86400
, .{});

// Query view
var result = try db.exec("SELECT * FROM active_users", .{});
defer result.deinit();
```

**Materialized view (cached results):**
```zig
_ = try db.exec(
    \\CREATE MATERIALIZED VIEW user_stats AS
    \\SELECT user_id, COUNT(*) as post_count, SUM(views) as total_views
    \\FROM posts
    \\GROUP BY user_id
, .{});

// Refresh when data changes
_ = try db.exec("REFRESH MATERIALIZED VIEW user_stats", .{});
```

**CTEs (WITH clause):**
```zig
var result = try db.exec(
    \\WITH top_authors AS (
    \\    SELECT user_id, COUNT(*) as count
    \\    FROM posts
    \\    GROUP BY user_id
    \\    ORDER BY count DESC
    \\    LIMIT 10
    \\)
    \\SELECT users.username, top_authors.count
    \\FROM top_authors
    \\JOIN users ON users.id = top_authors.user_id
, .{});
defer result.deinit();
```

**Recursive CTE:**
```zig
var result = try db.exec(
    \\WITH RECURSIVE subordinates AS (
    \\    SELECT id, name, manager_id FROM employees WHERE id = ?
    \\    UNION ALL
    \\    SELECT e.id, e.name, e.manager_id
    \\    FROM employees e
    \\    INNER JOIN subordinates s ON e.manager_id = s.id
    \\)
    \\SELECT * FROM subordinates
, .{1});  // Find all subordinates of employee 1
defer result.deinit();
```

---

### Window Functions

Window functions perform calculations across rows related to the current row.

**ROW_NUMBER:**
```zig
var result = try db.exec(
    \\SELECT username, score,
    \\       ROW_NUMBER() OVER (ORDER BY score DESC) as rank
    \\FROM leaderboard
, .{});
defer result.deinit();
```

**RANK and DENSE_RANK:**
```zig
var result = try db.exec(
    \\SELECT category, product, sales,
    \\       RANK() OVER (PARTITION BY category ORDER BY sales DESC) as rank
    \\FROM products
, .{});
defer result.deinit();
```

**LAG and LEAD (access previous/next row):**
```zig
var result = try db.exec(
    \\SELECT date, price,
    \\       price - LAG(price) OVER (ORDER BY date) as change
    \\FROM stock_prices
, .{});
defer result.deinit();
```

**Frame specs:**
```zig
var result = try db.exec(
    \\SELECT date, value,
    \\       AVG(value) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) as moving_avg
    \\FROM metrics
, .{});
defer result.deinit();
```

---

### Triggers

Triggers automatically execute code in response to DML events.

**Create trigger:**
```zig
_ = try db.exec(
    \\CREATE TRIGGER update_timestamp
    \\BEFORE UPDATE ON posts
    \\FOR EACH ROW
    \\BEGIN
    \\    NEW.updated_at = unixepoch();
    \\END
, .{});
```

**AFTER trigger:**
```zig
_ = try db.exec(
    \\CREATE TRIGGER audit_log
    \\AFTER DELETE ON users
    \\FOR EACH ROW
    \\BEGIN
    \\    INSERT INTO audit_log (action, user_id, timestamp)
    \\    VALUES ('DELETE', OLD.id, unixepoch());
    \\END
, .{});
```

---

## Part 3: Server Mode

Server mode runs Silica as a standalone PostgreSQL-compatible database server.

### Starting the Server

**Create data directory:**
```bash
mkdir -p /var/lib/silica
```

**Initialize database:**
```bash
silica init --data-dir /var/lib/silica
```

**Start server:**
```bash
silica server --data-dir /var/lib/silica --port 5433 --max-connections 100
```

**With configuration file:**
```bash
# Create /etc/silica/silica.conf
cat > /etc/silica/silica.conf <<EOF
port = 5433
max_connections = 200
work_mem = 64MB
shared_buffers = 256MB
wal_level = replica
EOF

silica server --config /etc/silica/silica.conf
```

**Run as daemon:**
```bash
silica server --data-dir /var/lib/silica --daemon --log-file /var/log/silica.log
```

---

### Connecting with psql

```bash
psql -h localhost -p 5433 -U postgres -d mydb
```

**Example session:**
```sql
postgres=# CREATE DATABASE myapp;
CREATE DATABASE

postgres=# \c myapp
You are now connected to database "myapp".

myapp=# CREATE TABLE users (id SERIAL PRIMARY KEY, name TEXT);
CREATE TABLE

myapp=# INSERT INTO users (name) VALUES ('Alice'), ('Bob');
INSERT 0 2

myapp=# SELECT * FROM users;
 id │ name
────┼───────
  1 │ Alice
  2 │ Bob
(2 rows)
```

---

### Client Libraries

Silica is compatible with PostgreSQL client libraries.

**Python (psycopg2):**
```python
import psycopg2

conn = psycopg2.connect(
    host="localhost",
    port=5433,
    user="postgres",
    password="secret",
    database="mydb"
)

cur = conn.cursor()
cur.execute("SELECT * FROM users")
for row in cur.fetchall():
    print(row)

conn.close()
```

**Node.js (pg):**
```javascript
const { Client } = require('pg');

const client = new Client({
  host: 'localhost',
  port: 5433,
  user: 'postgres',
  database: 'mydb',
});

await client.connect();
const res = await client.query('SELECT * FROM users');
console.log(res.rows);
await client.end();
```

**Go (pgx):**
```go
package main

import (
    "context"
    "fmt"
    "github.com/jackc/pgx/v5"
)

func main() {
    conn, _ := pgx.Connect(context.Background(), "postgres://localhost:5433/mydb")
    defer conn.Close(context.Background())

    rows, _ := conn.Query(context.Background(), "SELECT id, name FROM users")
    defer rows.Close()

    for rows.Next() {
        var id int
        var name string
        rows.Scan(&id, &name)
        fmt.Printf("User: %d, %s\n", id, name)
    }
}
```

---

### Replication

Set up streaming replication for high availability.

**Primary server:**
```bash
# /etc/silica/primary.conf
wal_level = replica
max_wal_senders = 5
wal_keep_size = 1GB

silica server --config /etc/silica/primary.conf
```

**Replica server:**
```bash
# Take base backup
silica-basebackup -h primary-host -p 5433 -D /var/lib/silica-replica

# /var/lib/silica-replica/recovery.conf
primary_conninfo = 'host=primary-host port=5433 user=replicator'
standby_mode = on

silica server --data-dir /var/lib/silica-replica --port 5434
```

**Monitor replication:**
```sql
-- On primary
SELECT * FROM pg_stat_replication;

-- On replica
SELECT pg_last_wal_receive_lsn(), pg_last_wal_replay_lsn();
```

---

## Part 4: Operations

### Backup and Restore

**Logical backup (SQL dump):**
```bash
# Backup
silica-dump -h localhost -p 5433 mydb > backup.sql

# Restore
psql -h localhost -p 5433 mydb < backup.sql
```

**Physical backup (copy files):**
```bash
# Stop server first
systemctl stop silica

# Copy data directory
cp -r /var/lib/silica /backup/silica-$(date +%Y%m%d)

# Restart server
systemctl start silica
```

**Point-in-time recovery:**
```bash
# Restore base backup
cp -r /backup/silica-base /var/lib/silica

# Replay WAL to specific time
silica-recovery --target-time "2024-03-25 14:30:00"
```

---

### Performance Tuning

**Key parameters:**
```ini
# /etc/silica/silica.conf

# Memory
shared_buffers = 256MB      # 25% of RAM
work_mem = 64MB             # Per-sort operation
maintenance_work_mem = 128MB

# WAL
wal_buffers = 16MB
checkpoint_timeout = 5min
max_wal_size = 1GB

# Query planning
random_page_cost = 1.1      # SSD (4.0 for HDD)
effective_cache_size = 2GB  # OS cache estimate
```

**EXPLAIN ANALYZE:**
```sql
EXPLAIN ANALYZE SELECT * FROM users WHERE age > 25;
```

**Create appropriate indexes:**
```sql
-- Identify slow queries
SELECT query, mean_exec_time FROM pg_stat_statements
ORDER BY mean_exec_time DESC LIMIT 10;

-- Add indexes
CREATE INDEX idx_users_age ON users(age);
```

**VACUUM regularly:**
```sql
VACUUM ANALYZE users;  -- Reclaim space and update stats
```

---

### Monitoring

**pg_stat_activity:**
```sql
SELECT pid, usename, state, query
FROM pg_stat_activity
WHERE state = 'active';
```

**pg_locks:**
```sql
SELECT locktype, relation::regclass, mode, granted
FROM pg_locks
WHERE NOT granted;  -- Find blocked queries
```

**Configuration:**
```sql
SHOW ALL;  -- View all parameters
SHOW work_mem;
SET work_mem = '128MB';
```

---

## Troubleshooting

### Database file locked

**Symptom:** `error.FileLocked` when opening database

**Cause:** Another process has the database open

**Solution:**
```bash
# Find process
lsof mydb.db

# Kill if needed
kill <pid>
```

---

### Out of memory

**Symptom:** `error.OutOfMemory` during large queries

**Cause:** Insufficient `work_mem`

**Solution:**
```sql
SET work_mem = '256MB';  -- Increase for this session
```

---

### Slow queries

**Symptom:** Queries taking >1 second

**Solution:**
1. Run `EXPLAIN ANALYZE` to see query plan
2. Check if indexes are used (`Index Scan` vs `Seq Scan`)
3. Create indexes on WHERE/JOIN columns
4. Update statistics: `ANALYZE table_name`

---

### Serialization failures

**Symptom:** `error.SerializationFailure` in SERIALIZABLE mode

**Cause:** Transaction conflict detected by SSI

**Solution:** Retry transaction with exponential backoff (see [Transactions](#transactions))

---

### Known Issues

See [docs/KNOWN_ISSUES.md](KNOWN_ISSUES.md) for current limitations:
- Issue #16: MVCC visibility bugs (NoRows errors, lost updates)
- Issue #15: SSI not fully implemented

---

## Next Steps

- **[SQL Reference](SQL_REFERENCE.md)** — Complete SQL syntax guide
- **[Operations Guide](OPERATIONS_GUIDE.md)** — Production deployment, monitoring, tuning
- **[Architecture Guide](ARCHITECTURE.md)** — Internal design, storage format
- **[API Reference](API_REFERENCE.md)** — Detailed API documentation

---

**Questions? Issues?**
- [GitHub Issues](https://github.com/yusa-imit/silica/issues)
- [Discussions](https://github.com/yusa-imit/silica/discussions)
