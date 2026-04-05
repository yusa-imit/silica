# Silica Examples

This directory contains example SQL scripts and code samples demonstrating Silica's features.

## SQL Examples

### `quickstart.sql` — Minimal Quick Start

A minimal working example to get started quickly:
- Basic table creation
- Simple INSERT queries
- SELECT with WHERE and ORDER BY
- Perfect for first-time users

**Run:**
```bash
silica quickstart.db < quickstart.sql
```

### `tutorial_simple.sql` — Core SQL Operations

Covers essential SQL operations without complex joins:
- Table creation with constraints
- INSERT, UPDATE, DELETE
- SELECT queries with filtering and sorting
- Aggregate functions
- Transactions
- Basic indexes

**Run:**
```bash
silica tutorial_simple.db < tutorial_simple.sql
```

### `tutorial.sql` — Comprehensive SQL Tutorial

A complete SQL tutorial covering:
- Table creation with constraints (PRIMARY KEY, FOREIGN KEY, CHECK, UNIQUE)
- Data types (INTEGER, TEXT, BOOLEAN, TIMESTAMP)
- INSERT operations (single and multi-row)
- Basic queries (SELECT, WHERE, ORDER BY)
- Joins (INNER JOIN, LEFT JOIN)
- Aggregate functions (COUNT, SUM, AVG)
- Subqueries (correlated and uncorrelated)
- Common Table Expressions (CTEs, recursive CTEs)
- Window functions (ROW_NUMBER, RANK, DENSE_RANK)
- Transactions (BEGIN, COMMIT, ROLLBACK)
- Indexes for performance
- Advanced features (CASE, string/date functions, COALESCE)
- Views

**Run:**
```bash
silica tutorial.db < tutorial.sql
```

### `advanced_features.sql` — Production-Grade Features ✨ **NEW**

Showcases Silica's unique production capabilities:
- **MVCC Transactions**: Snapshot isolation, REPEATABLE READ, SERIALIZABLE
- **Window Functions**: ROW_NUMBER, RANK, LAG, LEAD, running totals with frames
- **Recursive CTEs**: Organizational hierarchies, graph traversal
- **JSON/JSONB**: Operators (->>, @>, ?), indexing with GIN
- **Full-Text Search**: TSVECTOR, TSQUERY, ts_rank ranking
- **Advanced Indexes**: Hash, GiST, GIN, CREATE INDEX CONCURRENTLY
- **Materialized Views**: Pre-computed aggregations
- **Set Operations**: UNION, INTERSECT, EXCEPT
- **Comprehensive Constraints**: CHECK, UNIQUE, FOREIGN KEY with CASCADE
- **Performance Analysis**: EXPLAIN, EXPLAIN ANALYZE with runtime stats

**Run:**
```bash
silica advanced_demo.db < advanced_features.sql

# Or interactively to see each feature:
silica advanced_demo.db
silica> .read advanced_features.sql
```

**Clean up:**
```bash
rm -f quickstart.db tutorial_simple.db tutorial.db advanced_demo.db *.db-wal *.db-shm
```

## Code Examples (Coming Soon)

Future additions will include:
- Embedded API examples (Zig)
- C FFI examples
- Server mode examples
- Replication setup examples
- Performance benchmarking scripts

## Contributing Examples

Have a useful example? Contributions welcome! Please ensure:
1. Examples are well-commented and self-contained
2. SQL scripts include cleanup steps
3. Code examples include error handling
4. README documents usage and expected output

## See Also

- [SQL Reference](../docs/SQL_REFERENCE.md) — Complete SQL syntax guide
- [API Reference](../docs/API_REFERENCE.md) — Embedded API documentation
- [Getting Started Guide](../docs/GETTING_STARTED.md) — Installation and setup
- [Operations Guide](../docs/OPERATIONS_GUIDE.md) — Backup, monitoring, tuning
