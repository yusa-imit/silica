# Silica Examples

This directory contains example SQL scripts and code samples demonstrating Silica's features.

## SQL Examples

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

**Run the tutorial:**

```bash
# Method 1: Pipe the script
silica tutorial.db < tutorial.sql

# Method 2: Interactive mode
silica tutorial.db
silica> .read tutorial.sql

# Method 3: Step through interactively
silica tutorial.db
silica> -- Copy-paste queries one at a time
```

**Clean up:**

```bash
rm -f tutorial.db tutorial.db-wal tutorial.db-shm
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
