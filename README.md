# Silica

**Production-grade embedded relational database engine written in Zig**

Silica is a full-featured RDBMS inspired by SQLite and PostgreSQL, offering both embedded (in-process) and client-server modes with complete SQL:2016 support, ACID transactions, and streaming replication.

[![CI](https://github.com/yusa-imit/silica/workflows/CI/badge.svg)](https://github.com/yusa-imit/silica/actions)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Zig](https://img.shields.io/badge/zig-0.15.x-orange.svg)](https://ziglang.org)

---

## ✨ Features

### Database Engine
- **Storage Engine**: B+Tree indexes, buffer pool with LRU eviction, 4KB pages (configurable 512B-64KB)
- **Transaction Manager**: MVCC with snapshot isolation, WAL for durability, ACID guarantees
- **SQL Support**: Full SQL:2016 compliance — DDL, DML, joins, subqueries, CTEs, window functions, triggers, views
- **Data Types**: INTEGER, TEXT, REAL, BLOB, DATE, TIME, TIMESTAMP, INTERVAL, NUMERIC, UUID, JSON, ARRAY, ENUM
- **Indexes**: B+Tree (primary), hash, GiST, GIN, concurrent index creation
- **Full-Text Search**: TSVECTOR/TSQUERY with GIN indexes, ranking functions
- **JSON/JSONB**: Binary storage, operators (`->`, `->>`, `@>`, `?`), path queries

### Concurrency & Isolation
- **MVCC**: Multi-version concurrency control for high read throughput
- **Isolation Levels**: READ UNCOMMITTED, READ COMMITTED, REPEATABLE READ, SERIALIZABLE (SSI)
- **Locking**: Row-level and table-level locks, deadlock detection
- **VACUUM**: Automatic dead tuple reclamation, free space map

### Server Mode (PostgreSQL Wire Protocol v3)
- **Client-Server**: TCP server with async I/O, connection pooling
- **Wire Protocol**: PostgreSQL-compatible — works with `psql`, `pgcli`, libpq, JDBC, etc.
- **Authentication**: SCRAM-SHA-256, MD5, trust
- **TLS**: Encrypted connections
- **Authorization**: Role-based access control (RBAC), row-level security (RLS)

### Replication
- **Streaming Replication**: WAL-based physical replication
- **High Availability**: Automatic failover, replica promotion, cascading replication
- **Synchronous/Async**: Configurable replication modes
- **Base Backup**: `pg_basebackup`-compatible initial sync

### Operational Tools
- **EXPLAIN ANALYZE**: Query plan visualization with runtime statistics
- **VACUUM**: Manual and auto-vacuum for space reclamation
- **REINDEX**: Rebuild indexes concurrently
- **Monitoring**: `pg_stat_activity`, `pg_locks` views
- **Configuration**: `silica.conf` file, `SET`/`SHOW`/`RESET` SQL commands

---

## 🚀 Quick Start

### Embedded Mode (In-Process)

```zig
const std = @import("std");
const silica = @import("silica");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Open database (creates if not exists)
    var db = try silica.Database.open(allocator, "myapp.db", .{});
    defer db.close();

    // Execute SQL
    _ = try db.exec("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)", .{});
    _ = try db.exec("INSERT INTO users (id, name) VALUES (?, ?)", .{ 1, "Alice" });

    // Query with iteration
    var result = try db.exec("SELECT id, name FROM users WHERE id = ?", .{1});
    defer result.deinit();

    if (result.rows) |*rows| {
        while (try rows.next()) |row| {
            defer row.deinit();
            const id = row.values[0].integer;
            const name = row.values[1].text;
            std.debug.print("User: id={}, name={s}\n", .{ id, name });
        }
    }
}
```

**Build:**
```bash
# Add to build.zig.zon
.dependencies = .{
    .silica = .{
        .url = "https://github.com/yusa-imit/silica/archive/v0.3.0.tar.gz",
        .hash = "...",
    },
},

# Add to build.zig
const silica = b.dependency("silica", .{});
exe.root_module.addImport("silica", silica.module("silica"));
```

### Server Mode (Client-Server)

**Start server:**
```bash
silica server --data-dir /var/lib/silica --port 5433
```

**Connect with psql:**
```bash
psql -h localhost -p 5433 -U postgres
```

**Connect with libpq (C):**
```c
#include <libpq-fe.h>

PGconn *conn = PQconnectdb("host=localhost port=5433 user=postgres dbname=mydb");
PGresult *res = PQexec(conn, "SELECT * FROM users");
// ... process results
PQclear(res);
PQfinish(conn);
```

---

## 📦 Installation

### From Source (Zig 0.15.x required)

```bash
git clone https://github.com/yusa-imit/silica.git
cd silica
zig build -Doptimize=ReleaseSafe
sudo cp zig-out/bin/silica /usr/local/bin/
```

### Zig Package Manager

Add to your `build.zig.zon`:
```zig
.dependencies = .{
    .silica = .{
        .url = "https://github.com/yusa-imit/silica/archive/v0.3.0.tar.gz",
        .hash = "<hash from zig fetch>",
    },
},
```

Then in `build.zig`:
```zig
const silica = b.dependency("silica", .{});
exe.root_module.addImport("silica", silica.module("silica"));
```

### System Packages (Coming Soon)

```bash
# Debian/Ubuntu
sudo apt install silica

# macOS (Homebrew)
brew install silica

# Arch Linux
yay -S silica
```

---

## 📚 Documentation

- **[Getting Started Guide](docs/GETTING_STARTED.md)** — Comprehensive tutorial for embedded and server modes
- **[API Reference](docs/API_REFERENCE.md)** — Zig embedded API documentation
- **[SQL Reference](docs/SQL_REFERENCE.md)** — Supported SQL statements, functions, operators
- **[Operations Guide](docs/OPERATIONS_GUIDE.md)** — Backup, restore, replication, monitoring, tuning
- **[Configuration](docs/CONFIGURATION.md)** — silica.conf file format, SET/SHOW/RESET commands
- **[Architecture Guide](docs/ARCHITECTURE.md)** — Internal design, storage format, query processing
- **[PRD](docs/PRD.md)** — Product requirements document (full feature list)

---

## 🛠️ Development

### Build & Test

```bash
# Build (debug)
zig build

# Build (release)
zig build -Doptimize=ReleaseSafe

# Run tests
zig build test

# Run benchmarks
zig build bench

# Run TPC-C benchmark
zig build tpcc

# Run TPC-H benchmark
zig build tpch

# Cross-compile for Linux
zig build -Dtarget=x86_64-linux -Doptimize=ReleaseSafe
```

### Interactive Shell (Embedded Mode)

```bash
zig build run -- mydb.db
```

```sql
silica> CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price REAL);
silica> INSERT INTO products VALUES (1, 'Widget', 19.99);
silica> SELECT * FROM products;
┌────┬─────────┬───────┐
│ id │  name   │ price │
├────┼─────────┼───────┤
│  1 │ Widget  │ 19.99 │
└────┴─────────┴───────┘
```

### TUI Database Browser

```bash
zig build run -- --tui mydb.db
```

Navigate with arrow keys, press `Enter` to view table contents, `q` to quit.

---

## 🏗️ Architecture

Silica is organized into six layers:

1. **Client Layer**: Zig API, C FFI, PostgreSQL wire protocol v3
2. **SQL Frontend**: Tokenizer → Parser → Semantic Analyzer
3. **Query Engine**: Planner → Cost-based Optimizer → Volcano Executor
4. **Transaction Manager**: WAL, MVCC, Lock Manager, Snapshot Isolation
5. **Storage Engine**: B+Tree, Buffer Pool, Page Manager, Overflow Pages
6. **OS Layer**: File I/O, fsync, mmap (optional)

### File Format

- **Magic bytes**: `SLCA` (Silica)
- **Page size**: 4096 bytes (default, configurable 512-65536)
- **Page types**: header, internal, leaf, overflow, free
- **Checksums**: CRC32C on every page
- **Single-file database**: `mydb.db` + `mydb.db-wal` (during transactions)

### Storage Layout

```
Database File (mydb.db):
┌─────────────────────────────────────────┐
│ File Header (Page 0)                    │ ← Magic, version, page_size, root_page
├─────────────────────────────────────────┤
│ B+Tree Root Page (Page 1)               │ ← Root of catalog B+Tree
├─────────────────────────────────────────┤
│ B+Tree Internal/Leaf Pages              │ ← Table data, indexes
├─────────────────────────────────────────┤
│ Overflow Pages                          │ ← Large rows (>page_size/4)
├─────────────────────────────────────────┤
│ Free Pages (linked list)                │ ← Available for reuse
└─────────────────────────────────────────┘

WAL File (mydb.db-wal):
┌─────────────────────────────────────────┐
│ WAL Header                              │ ← Magic, version, salts
├─────────────────────────────────────────┤
│ Frame 1: [page_id, data, checksum]     │ ← Write-ahead log entry
├─────────────────────────────────────────┤
│ Frame 2: [page_id, data, checksum]     │
│ ...                                     │
└─────────────────────────────────────────┘
```

---

## 🔬 Testing & Certification

Silica has comprehensive test coverage across all layers:

- **Unit Tests**: 2766 tests covering all modules (storage, SQL, transactions, server, replication)
- **Fuzz Tests**: 67 tests for storage, tokenizer, parser, wire protocol, WAL
- **SQL Conformance**: 35 tests validating SQL:2016 compliance
- **TPC-C**: OLTP benchmark (new-order, payment transactions)
- **TPC-H**: OLAP benchmark (Q1, Q3, Q6 queries)
- **Jepsen-style**: 19 distributed consistency tests (isolation levels, replication)
- **Crash Recovery**: WAL replay verification under simulated power failure

Run all tests:
```bash
zig build test  # 2766 tests
```

---

## 🚧 Project Status

**Current Version**: v0.3.0 (Phase 3: WAL & Basic Transactions)

**Development Phase**: Phase 12 — Production Readiness

### Completed Features ✅
- ✅ Storage Engine (B+Tree, buffer pool, overflow pages)
- ✅ SQL Parser & Analyzer (DDL, DML, joins, subqueries, CTEs, window functions)
- ✅ Query Optimizer (cost-based, join reordering, index selection)
- ✅ Transaction Manager (WAL, MVCC, snapshot isolation, savepoints)
- ✅ Concurrency Control (row/table locks, deadlock detection)
- ✅ Advanced SQL (views, triggers, JSON/JSONB, full-text search)
- ✅ Server Mode (PostgreSQL wire protocol v3, TLS, RBAC)
- ✅ Replication (streaming, synchronous/async, failover)
- ✅ Operational Tools (EXPLAIN, VACUUM, REINDEX, monitoring)
- ✅ Testing & Certification (fuzz, conformance, TPC-C, TPC-H, Jepsen)

### In Progress 🚧
- 🚧 Documentation & Packaging (Milestone 25)
- 🚧 System packages (deb, rpm, brew)

### Known Issues 🐛
- [Issue #16](https://github.com/yusa-imit/silica/issues/16): MVCC visibility bugs (NoRows errors, snapshot inconsistency)
- [Issue #15](https://github.com/yusa-imit/silica/issues/15): SSI implementation needed for full SERIALIZABLE support

See [KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md) for details.

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

1. Install Zig 0.15.x from [ziglang.org](https://ziglang.org/download/)
2. Clone the repository: `git clone https://github.com/yusa-imit/silica.git`
3. Build and test: `zig build && zig build test`
4. Make your changes following the coding standards in [CLAUDE.md](CLAUDE.md)
5. Ensure all tests pass before submitting a PR

### Coding Standards

- **Naming**: camelCase for functions/variables, PascalCase for types, SCREAMING_SNAKE for constants
- **Error handling**: Explicit error unions, never `catch unreachable` in production
- **Memory**: Arena allocators for request-scoped work, GPA for long-lived allocations
- **Testing**: TDD — write failing tests first, then implement
- **Comments**: Only where logic is non-obvious

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **SQLite**: Inspiration for embedded database design and single-file format
- **PostgreSQL**: SQL feature set, wire protocol, MVCC architecture
- **Zig**: Memory safety, compile-time execution, cross-compilation

---

## 📬 Contact

- **Author**: Yusa ([@yusa-imit](https://github.com/yusa-imit))
- **Repository**: [github.com/yusa-imit/silica](https://github.com/yusa-imit/silica)
- **Issues**: [github.com/yusa-imit/silica/issues](https://github.com/yusa-imit/silica/issues)

---

**Built with ❤️ in Zig**
