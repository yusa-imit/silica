# Silica Architecture Guide

**Internal design, module structure, and implementation details**

> **Version**: v0.12 (Phase 12: Production Readiness)
> **Last Updated**: 2026-03-25

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Storage Layer](#storage-layer)
4. [SQL Frontend](#sql-frontend)
5. [Query Engine](#query-engine)
6. [Transaction Manager](#transaction-manager)
7. [Concurrency Control](#concurrency-control)
8. [Replication](#replication)
9. [Module Dependency Graph](#module-dependency-graph)
10. [Key Algorithms](#key-algorithms)

---

## Overview

### Design Philosophy

Silica is a **production-grade relational database** written in Zig, inspired by SQLite's simplicity and PostgreSQL's feature completeness. The design follows these principles:

1. **Correctness first**: ACID guarantees, data integrity, crash recovery
2. **Simplicity**: Single-file format, minimal configuration, zero dependencies
3. **Performance**: B+Tree indexing, cost-based optimization, MVCC
4. **Embeddability**: In-process library with clean Zig API and C FFI
5. **Wire compatibility**: PostgreSQL protocol v3 for client library interop

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        Client Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐    │
│  │  Zig API     │  │   C FFI      │  │ Wire Protocol   │    │
│  │ (Database)   │  │ (silica_*()) │  │ (PostgreSQL v3) │    │
│  └──────────────┘  └──────────────┘  └─────────────────┘    │
└──────────────────────────────────────────────────────────────┘
                            │
┌──────────────────────────────────────────────────────────────┐
│                      SQL Frontend                            │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐    │
│  │  Tokenizer   │→ │   Parser     │→ │   Analyzer      │    │
│  │  (Lexer)     │  │   (AST)      │  │   (Semantic)    │    │
│  └──────────────┘  └──────────────┘  └─────────────────┘    │
└──────────────────────────────────────────────────────────────┘
                            │
┌──────────────────────────────────────────────────────────────┐
│                      Query Engine                            │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐    │
│  │   Planner    │→ │  Optimizer   │→ │   Executor      │    │
│  │ (Logical)    │  │ (Physical)   │  │ (Volcano Model) │    │
│  └──────────────┘  └──────────────┘  └─────────────────┘    │
└──────────────────────────────────────────────────────────────┘
                            │
┌──────────────────────────────────────────────────────────────┐
│                   Transaction Manager                        │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐    │
│  │     WAL      │  │  Lock Mgr    │  │      MVCC       │    │
│  │ (Durability) │  │ (Isolation)  │  │ (Visibility)    │    │
│  └──────────────┘  └──────────────┘  └─────────────────┘    │
└──────────────────────────────────────────────────────────────┘
                            │
┌──────────────────────────────────────────────────────────────┐
│                      Storage Layer                           │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐    │
│  │  B+Tree      │  │ Buffer Pool  │  │  Page Manager   │    │
│  │ (Index)      │  │ (LRU Cache)  │  │ (File I/O)      │    │
│  └──────────────┘  └──────────────┘  └─────────────────┘    │
└──────────────────────────────────────────────────────────────┘
                            │
┌──────────────────────────────────────────────────────────────┐
│                         OS Layer                             │
│         File I/O, fsync, mmap (optional), threads            │
└──────────────────────────────────────────────────────────────┘
```

---

## System Architecture

### File Format

Silica uses a **single-file database format** with the following structure:

```
┌─────────────────────────────────────────────────────────────┐
│  File Header (Page 0)                                       │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Magic: "SLCA" (4 bytes)                               │  │
│  │ Version: 1 (4 bytes)                                  │  │
│  │ Page Size: 4096 (4 bytes)                             │  │
│  │ Page Count: N (8 bytes)                               │  │
│  │ Freelist Head: 0 (8 bytes)                            │  │
│  │ Schema B+Tree Root: 1 (8 bytes)                       │  │
│  │ Reserved: (padding to 4096 bytes)                     │  │
│  └───────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  Page 1: Schema B+Tree Root (Internal Node)                 │
├─────────────────────────────────────────────────────────────┤
│  Page 2: User Table B+Tree Root (Internal Node)             │
├─────────────────────────────────────────────────────────────┤
│  Page 3: Leaf Node (Tuple Data)                             │
├─────────────────────────────────────────────────────────────┤
│  Page 4: Overflow Page (Large Tuples)                       │
├─────────────────────────────────────────────────────────────┤
│  Page 5: Free Page (in Freelist)                            │
├─────────────────────────────────────────────────────────────┤
│  ...                                                         │
└─────────────────────────────────────────────────────────────┘

Companion files:
- mydb.db-wal     : Write-Ahead Log (for crash recovery)
- mydb.db-shm     : Shared memory index (WAL index)
```

### Page Types

Each page has a **8-byte header**:

```zig
pub const PageHeader = packed struct {
    page_type: PageType,      // 1 byte
    flags: u8,                 // 1 byte (reserved)
    cell_count: u16,           // 2 bytes (for B+Tree nodes)
    checksum: u32,             // 4 bytes (CRC32C)
};

pub const PageType = enum(u8) {
    header = 0x01,       // File header (page 0)
    internal = 0x02,     // B+Tree internal node
    leaf = 0x03,         // B+Tree leaf node
    overflow = 0x04,     // Overflow page (large values)
    free = 0x05,         // Free page (in freelist)
};
```

### Isolation & Durability

Silica provides **ACID guarantees** through:

- **Atomicity**: WAL (Write-Ahead Log) ensures all-or-nothing commits
- **Consistency**: Foreign keys, constraints, triggers enforce invariants
- **Isolation**: MVCC (Multi-Version Concurrency Control) with snapshot isolation
- **Durability**: fsync after WAL write, checkpoint to main database file

---

## Storage Layer

### Page Manager

**Purpose**: Low-level file I/O, page allocation/deallocation, checksum verification.

**Location**: `src/storage/page.zig` (530 lines)

**Key operations:**

```zig
pub const Pager = struct {
    file: std.fs.File,
    page_size: u32,
    page_count: u64,
    freelist_head: u64,  // Linked list of free pages

    pub fn init(file: std.fs.File, page_size: u32) !Pager;
    pub fn readPage(self: *Pager, page_num: u64, buf: []u8) !void;
    pub fn writePage(self: *Pager, page_num: u64, data: []const u8) !void;
    pub fn allocPage(self: *Pager) !u64;  // Returns new page number
    pub fn freePage(self: *Pager, page_num: u64) !void;
    pub fn checkpoint(self: *Pager) !void;  // fsync to disk
};
```

**Freelist design:**

Free pages are organized as a **linked list** embedded in the pages themselves:

```
Free Page Layout:
┌────────────────────────────────────────┐
│ PageHeader (8 bytes)                   │
│   page_type: .free                     │
├────────────────────────────────────────┤
│ Next Free Page Number (8 bytes)        │
│   0 = end of list                      │
├────────────────────────────────────────┤
│ Unused space (page_size - 16 bytes)    │
└────────────────────────────────────────┘
```

**Allocation:**
1. If `freelist_head != 0`, pop from freelist
2. Otherwise, extend file with new page (increment `page_count`)

**Deallocation:**
1. Set `page_type = .free`
2. Set `next = freelist_head`
3. Update `freelist_head = page_num`

### Buffer Pool

**Purpose**: In-memory LRU cache of disk pages, reduces file I/O.

**Location**: `src/storage/buffer_pool.zig` (480 lines)

**Key operations:**

```zig
pub const BufferPool = struct {
    capacity: usize,       // Max cached pages (default: 2000)
    frames: []Frame,       // Page cache
    lru: LRUCache,         // Eviction policy
    dirty: HashSet,        // Pages modified in memory

    pub fn init(allocator: Allocator, capacity: usize) !BufferPool;
    pub fn pin(self: *BufferPool, page_num: u64) !*Frame;
    pub fn unpin(self: *BufferPool, page_num: u64) void;
    pub fn markDirty(self: *BufferPool, page_num: u64) void;
    pub fn flush(self: *BufferPool) !void;  // Write dirty pages to disk
    pub fn evict(self: *BufferPool) !void;  // Free a frame (LRU)
};

pub const Frame = struct {
    page_num: u64,
    data: [4096]u8,
    pin_count: u32,     // Reference count (cannot evict if > 0)
    dirty: bool,
};
```

**LRU eviction policy:**

When the buffer pool is full and a new page is needed:
1. Find a frame with `pin_count == 0` (unpinned)
2. If dirty, flush to disk (WAL first if in transaction)
3. Reuse frame for new page

**Concurrency:**

The buffer pool is **NOT thread-safe** by default. In server mode, each connection has its own buffer pool (no shared state). Future versions may add a shared buffer pool with fine-grained locking.

### B+Tree

**Purpose**: Index structure for fast key-value lookups, range scans, and sorted access.

**Location**: `src/storage/btree.zig` (4300 lines)

**Key operations:**

```zig
pub const BTree = struct {
    pager: *Pager,
    buffer_pool: *BufferPool,
    root_page: u64,

    pub fn insert(self: *BTree, key: []const u8, value: []const u8) !void;
    pub fn delete(self: *BTree, key: []const u8) !void;
    pub fn lookup(self: *BTree, key: []const u8) !?[]const u8;
    pub fn scan(self: *BTree, start_key: ?[]const u8) !Cursor;
    pub fn split(self: *BTree, node_page: u64) !void;
    pub fn merge(self: *BTree, node_page: u64) !void;
};

pub const Cursor = struct {
    btree: *BTree,
    page_num: u64,
    cell_idx: u16,

    pub fn next(self: *Cursor) !?KVPair;
    pub fn prev(self: *Cursor) !?KVPair;
};
```

**Internal node layout:**

```
Internal Node (page_type = .internal):
┌────────────────────────────────────────┐
│ PageHeader (8 bytes)                   │
│   page_type: .internal                 │
│   cell_count: N                        │
├────────────────────────────────────────┤
│ Cell Offset Array [N * 2 bytes]        │
│   [offset_0, offset_1, ..., offset_N-1]│
├────────────────────────────────────────┤
│ Right Child Pointer (8 bytes)          │
│   Page number of rightmost child       │
├────────────────────────────────────────┤
│ Free Space                             │
├────────────────────────────────────────┤
│ Cell Area (grows upward from bottom)   │
│ ┌──────────────────────────────────┐   │
│ │ Cell 0:                          │   │
│ │   Key Length (varint)            │   │
│ │   Key Data                       │   │
│ │   Child Page Number (8 bytes)    │   │
│ └──────────────────────────────────┘   │
│ ┌──────────────────────────────────┐   │
│ │ Cell 1: ...                      │   │
│ └──────────────────────────────────┘   │
└────────────────────────────────────────┘
```

**Leaf node layout:**

```
Leaf Node (page_type = .leaf):
┌────────────────────────────────────────┐
│ PageHeader (8 bytes)                   │
│   page_type: .leaf                     │
│   cell_count: N                        │
├────────────────────────────────────────┤
│ Cell Offset Array [N * 2 bytes]        │
├────────────────────────────────────────┤
│ Next Leaf Pointer (8 bytes)            │
│   Page number of next leaf (for scans) │
├────────────────────────────────────────┤
│ Free Space                             │
├────────────────────────────────────────┤
│ Cell Area                              │
│ ┌──────────────────────────────────┐   │
│ │ Cell 0:                          │   │
│ │   Key Length (varint)            │   │
│ │   Key Data                       │   │
│ │   Value Length (varint)          │   │
│ │   Value Data (or overflow ptr)   │   │
│ └──────────────────────────────────┘   │
└────────────────────────────────────────┘
```

**Split algorithm:**

When a leaf node exceeds `page_size` during insert:
1. Allocate a new sibling page
2. Move top half of cells to sibling
3. Update parent's cell array to include new separator key
4. If parent overflows, recursively split parent
5. If root splits, create new root (tree height increases)

**Merge algorithm:**

When a leaf node falls below fill threshold (< 50%) during delete:
1. Check left/right siblings for merge candidate
2. If combined size fits in one page, merge:
   - Move all cells from sibling to current node
   - Free sibling page
   - Remove separator key from parent
3. If parent underflows, recursively merge parent

**Range scan:**

To scan `WHERE key >= 'start' AND key < 'end'`:
1. Search for `start` key (descend to leaf)
2. Iterate cells in leaf, returning matching keys
3. Follow `next_leaf` pointer to continue scan
4. Stop when `key >= end` or end of table

---

## SQL Frontend

### Tokenizer

**Purpose**: Lexical analysis — convert SQL text into token stream.

**Location**: `src/sql/tokenizer.zig` (1100 lines)

**Token types:**

```zig
pub const TokenType = enum {
    // Keywords
    select, insert, update, delete, create, alter, drop,
    table, index, view, trigger, function,
    where, from, join, on, group_by, having, order_by, limit,
    and_, or_, not_,

    // Literals
    integer_literal,   // 42
    float_literal,     // 3.14
    string_literal,    // 'hello'
    identifier,        // column_name

    // Operators
    equals,            // =
    not_equals,        // !=, <>
    less_than,         // <
    greater_than,      // >
    plus, minus, star, slash,

    // Punctuation
    lparen, rparen, comma, semicolon, dot,

    // Special
    eof,
    invalid,
};

pub const Token = struct {
    type: TokenType,
    lexeme: []const u8,  // Source text slice
    line: u32,
    column: u32,
};
```

**Key operations:**

```zig
pub const Tokenizer = struct {
    input: []const u8,
    position: usize,
    line: u32,
    column: u32,

    pub fn init(input: []const u8) Tokenizer;
    pub fn next(self: *Tokenizer) Token;
    pub fn peek(self: *Tokenizer) Token;
    pub fn expect(self: *Tokenizer, expected: TokenType) !Token;
};
```

**String literal handling:**

Single-quoted strings with escape sequences:
- `''` → `'` (doubled quote = literal quote)
- `\n` → newline
- `\t` → tab

**Identifier case sensitivity:**

Unquoted identifiers are **case-insensitive** (lowercased):
- `SELECT UserId` → tokens: `select`, `userid`

Double-quoted identifiers are **case-sensitive**:
- `SELECT "UserId"` → tokens: `select`, `UserId`

### Parser

**Purpose**: Syntax analysis — convert token stream into Abstract Syntax Tree (AST).

**Location**: `src/sql/parser.zig` (3500 lines), `src/sql/ast.zig` (800 lines)

**Parsing technique**: Recursive Descent (top-down, LL(1))

**AST node hierarchy:**

```zig
pub const Stmt = union(enum) {
    select: SelectStmt,
    insert: InsertStmt,
    update: UpdateStmt,
    delete: DeleteStmt,
    create_table: CreateTableStmt,
    create_index: CreateIndexStmt,
    drop_table: DropTableStmt,
    // ... 30+ statement types
};

pub const SelectStmt = struct {
    columns: []SelectColumn,
    from: ?TableRef,
    where_clause: ?Expr,
    group_by: ?[]Expr,
    having: ?Expr,
    order_by: ?[]OrderByExpr,
    limit: ?LimitExpr,
    // ... window functions, CTEs, etc.
};

pub const Expr = union(enum) {
    column_ref: ColumnRef,
    literal: Literal,
    binary_op: BinaryOp,
    function_call: FunctionCall,
    subquery: SelectStmt,
    case_expr: CaseExpr,
    bind_parameter: u32,  // Prepared statement placeholder
    // ... 20+ expression types
};
```

**Error recovery:**

On syntax error, parser attempts to recover by:
1. Skipping tokens until synchronization point (`;`, keywords)
2. Reporting error with line/column position
3. Continuing parse to find additional errors

**Example parse:**

```sql
SELECT name, age FROM users WHERE age > 25 ORDER BY name LIMIT 10;
```

Parsed AST:
```
SelectStmt {
  columns: [
    { expr: ColumnRef("name"), alias: null },
    { expr: ColumnRef("age"), alias: null },
  ],
  from: TableRef("users", alias: null),
  where_clause: BinaryOp {
    op: GreaterThan,
    left: ColumnRef("age"),
    right: IntegerLiteral(25),
  },
  order_by: [
    { expr: ColumnRef("name"), direction: Ascending },
  ],
  limit: LimitExpr { count: 10, offset: null },
}
```

### Semantic Analyzer

**Purpose**: Type checking, name resolution, constraint validation.

**Location**: `src/sql/analyzer.zig` (4100 lines)

**Key operations:**

```zig
pub const Analyzer = struct {
    schema: *SchemaProvider,
    allocator: Allocator,

    pub fn analyzeStmt(self: *Analyzer, stmt: ast.Stmt) !AnalyzedStmt;
    pub fn resolveColumn(self: *Analyzer, col_ref: ast.ColumnRef) !ResolvedColumn;
    pub fn inferType(self: *Analyzer, expr: ast.Expr) !ColumnType;
    pub fn checkConstraints(self: *Analyzer, stmt: ast.Stmt) !void;
};

pub const ResolvedColumn = struct {
    table: []const u8,
    column: []const u8,
    col_type: ColumnType,
    nullable: bool,
};
```

**Analysis phases:**

1. **Name resolution**: Map column references to schema definitions
   - Handle table aliases (`SELECT u.name FROM users u`)
   - Resolve ambiguous columns (error if not qualified)
   - Expand `SELECT *` to column list

2. **Type inference**: Compute result types for expressions
   - Literal types: `42` → INTEGER, `'hello'` → TEXT
   - Binary ops: `age + 5` → INTEGER (if `age` is INTEGER)
   - Function calls: `LENGTH('hello')` → INTEGER
   - Coercion: `age = '25'` → implicit cast to INTEGER

3. **Constraint validation**:
   - Primary key uniqueness (during INSERT/UPDATE)
   - Foreign key references (table + column existence)
   - NOT NULL enforcement
   - CHECK constraints

4. **Permission checks** (future):
   - Table access (SELECT/INSERT/UPDATE/DELETE)
   - Column access (for row-level security)

---

## Query Engine

### Planner

**Purpose**: Convert analyzed AST into logical query plan.

**Location**: `src/sql/planner.zig` (1800 lines)

**Logical plan nodes:**

```zig
pub const LogicalPlan = struct {
    root: *Node,
    plan_type: PlanType,  // .select, .insert, .update, .delete
};

pub const Node = union(enum) {
    scan: ScanNode,
    filter: FilterNode,
    project: ProjectNode,
    join: JoinNode,
    sort: SortNode,
    limit: LimitNode,
    aggregate: AggregateNode,
    // ... 15+ node types
};

pub const ScanNode = struct {
    table: []const u8,
    columns: []ResolvedColumn,
    index: ?[]const u8,  // NULL = seq scan, else index scan
};

pub const JoinNode = struct {
    left: *Node,
    right: *Node,
    join_type: JoinType,  // .inner, .left, .right, .full
    condition: Expr,
};
```

**Planning process:**

1. **FROM clause**: Create scan nodes for each table
2. **JOIN clause**: Create join nodes (Cartesian product if no condition)
3. **WHERE clause**: Create filter nodes
4. **GROUP BY**: Create aggregate nodes
5. **SELECT clause**: Create project nodes (column selection)
6. **ORDER BY**: Create sort nodes
7. **LIMIT/OFFSET**: Create limit nodes

**Example plan:**

```sql
SELECT name, COUNT(*) FROM users WHERE age > 25 GROUP BY name ORDER BY COUNT(*) DESC LIMIT 10;
```

Logical plan:
```
Limit (10)
  └─ Sort (COUNT(*) DESC)
      └─ Aggregate (GROUP BY name, COUNT(*))
          └─ Filter (age > 25)
              └─ Scan (users)
```

### Optimizer

**Purpose**: Transform logical plan into efficient physical plan.

**Location**: `src/sql/optimizer.zig` (1500 lines)

**Optimization rules:**

1. **Predicate pushdown**: Move filters closer to scans
   ```
   Before: Join → Filter (users.age > 25)
   After:  Join → Scan(users, filter: age > 25)
   ```

2. **Index selection**: Replace sequential scan with index scan
   ```
   Before: Scan(users, filter: id = 42)
   After:  IndexScan(users, index: PRIMARY KEY, key: 42)
   ```

3. **Join reordering**: Optimize multi-table joins (cost-based)
   ```
   Before: users JOIN orders JOIN items
   After:  items JOIN orders JOIN users  (if items.rows < users.rows)
   ```

4. **Projection elimination**: Remove unused columns early
   ```
   Before: Scan(users, columns: [id, name, email, age]) → Project(name)
   After:  Scan(users, columns: [name])
   ```

5. **Constant folding**: Evaluate compile-time expressions
   ```
   Before: WHERE age > (20 + 5)
   After:  WHERE age > 25
   ```

**Cost model:**

```zig
pub const CostEstimator = struct {
    pub fn estimateSeqScan(rows: u64) f64 {
        return @as(f64, @floatFromInt(rows)) * SEQ_SCAN_COST_PER_ROW;
    }

    pub fn estimateIndexScan(rows: u64, selectivity: f64) f64 {
        const matched_rows = @as(f64, @floatFromInt(rows)) * selectivity;
        return matched_rows * INDEX_SCAN_COST_PER_ROW + INDEX_LOOKUP_COST;
    }

    pub fn estimateJoin(left_rows: u64, right_rows: u64, join_type: JoinType) f64 {
        return switch (join_type) {
            .nested_loop => @as(f64, @floatFromInt(left_rows * right_rows)),
            .hash => @as(f64, @floatFromInt(left_rows + right_rows)) * HASH_BUILD_COST,
        };
    }
};
```

**Statistics-based estimation:**

After `ANALYZE` command, optimizer uses:
- Table row counts
- Column histograms (equi-depth, ~10 buckets)
- Index statistics (unique values, NULL count)

### Executor

**Purpose**: Execute physical plan and return result rows.

**Location**: `src/sql/executor.zig` (7000 lines)

**Execution model**: Volcano (iterator-based, pull model)

Each operator implements the iterator interface:

```zig
pub const Operator = struct {
    vtable: *const VTable,
    state: *anyopaque,  // Operator-specific state

    pub const VTable = struct {
        open: *const fn (*anyopaque) anyerror!void,
        next: *const fn (*anyopaque) anyerror!?Row,
        close: *const fn (*anyopaque) void,
    };

    pub fn open(self: *Operator) !void {
        return self.vtable.open(self.state);
    }

    pub fn next(self: *Operator) !?Row {
        return self.vtable.next(self.state);
    }

    pub fn close(self: *Operator) void {
        self.vtable.close(self.state);
    }
};
```

**Example operators:**

**SeqScanOp:**

```zig
pub const SeqScanOp = struct {
    table_name: []const u8,
    cursor: ?BTree.Cursor,
    filter: ?Expr,

    pub fn open(self: *SeqScanOp) !void {
        self.cursor = try btree.scan(null);  // Start from first key
    }

    pub fn next(self: *SeqScanOp) !?Row {
        while (try self.cursor.?.next()) |kv| {
            const row = try deserializeRow(kv.value);
            if (self.filter == null or try evaluateExpr(self.filter.?, row)) {
                return row;
            }
        }
        return null;  // End of scan
    }

    pub fn close(self: *SeqScanOp) void {
        self.cursor = null;
    }
};
```

**NestedLoopJoinOp:**

```zig
pub const NestedLoopJoinOp = struct {
    left: *Operator,
    right: *Operator,
    condition: Expr,
    current_left_row: ?Row,

    pub fn open(self: *NestedLoopJoinOp) !void {
        try self.left.open();
        try self.right.open();
        self.current_left_row = try self.left.next();
    }

    pub fn next(self: *NestedLoopJoinOp) !?Row {
        while (self.current_left_row != null) {
            while (try self.right.next()) |right_row| {
                const joined_row = try combineRows(self.current_left_row.?, right_row);
                if (try evaluateExpr(self.condition, joined_row)) {
                    return joined_row;
                }
            }
            // Right input exhausted, fetch next left row and reset right
            self.current_left_row = try self.left.next();
            self.right.close();
            try self.right.open();
        }
        return null;  // Both inputs exhausted
    }

    pub fn close(self: *NestedLoopJoinOp) void {
        self.left.close();
        self.right.close();
    }
};
```

---

## Transaction Manager

### Write-Ahead Log (WAL)

**Purpose**: Crash recovery, durability guarantee.

**Location**: `src/tx/wal.zig` (900 lines)

**WAL file format:**

```
WAL File (mydb.db-wal):
┌─────────────────────────────────────────┐
│ WAL Header (32 bytes)                   │
│   Magic: "WLHD" (4 bytes)               │
│   Version: 1 (4 bytes)                  │
│   Page Size: 4096 (4 bytes)             │
│   Checksum: (4 bytes)                   │
│   Salt-1: (4 bytes, random)             │
│   Salt-2: (4 bytes, random)             │
│   Reserved: (8 bytes)                   │
├─────────────────────────────────────────┤
│ Frame 0:                                │
│   Page Number: 5 (8 bytes)              │
│   DB Size After Commit: 100 (8 bytes)   │
│   Salt-1: (4 bytes, matches header)     │
│   Salt-2: (4 bytes, matches header)     │
│   Frame Checksum: (8 bytes)             │
│   Page Data: (4096 bytes)               │
├─────────────────────────────────────────┤
│ Frame 1: ...                            │
├─────────────────────────────────────────┤
│ Commit Frame:                           │
│   DB Size After Commit: 105 (non-zero)  │
│   (Marks end of transaction)            │
└─────────────────────────────────────────┘
```

**WAL write protocol:**

1. **Begin transaction**: Allocate transaction ID
2. **Modify pages**: Write modified pages to WAL (not main DB yet)
3. **Commit**:
   - Append commit frame (db_size != 0)
   - fsync WAL file
   - Return success
4. **Checkpoint** (async):
   - Copy WAL frames to main database file
   - Truncate WAL file

**Crash recovery:**

On database open:
1. Check if WAL file exists
2. Read WAL header, validate checksum and salts
3. Scan frames, validate checksums
4. Find last complete transaction (last commit frame)
5. Replay frames up to last commit
6. Discard frames after last commit (incomplete transaction)
7. Checkpoint (copy frames to main DB)

**WAL index (shared memory):**

The `.db-shm` file is a shared memory index for fast WAL frame lookup:

```zig
pub const WalIndex = struct {
    frames: []FrameInfo,  // Page number → WAL frame offset

    pub fn lookup(self: *WalIndex, page_num: u64) ?u64 {
        // Returns WAL frame offset, or null if page not in WAL
    }
};
```

This avoids scanning the entire WAL for every page read.

### Lock Manager

**Purpose**: Isolation, deadlock detection, lock scheduling.

**Location**: `src/tx/lock.zig` (800 lines)

**Lock types:**

```zig
pub const LockMode = enum {
    // Table locks (coarse-grained)
    access_share,        // SELECT
    row_share,           // SELECT FOR UPDATE
    row_exclusive,       // INSERT/UPDATE/DELETE
    share_update_exclusive,  // VACUUM, ANALYZE
    share,               // CREATE INDEX
    share_row_exclusive, // (rare)
    exclusive,           // (rare)
    access_exclusive,    // DROP TABLE, VACUUM FULL

    // Row locks (fine-grained)
    row_share_lock,      // SELECT FOR SHARE
    row_exclusive_lock,  // SELECT FOR UPDATE, UPDATE, DELETE
};
```

**Lock compatibility matrix:**

|                      | AccessShare | RowShare | RowExclusive | ShareUpdateExclusive | Share | ShareRowExclusive | Exclusive | AccessExclusive |
|----------------------|-------------|----------|--------------|----------------------|-------|-------------------|-----------|-----------------|
| **AccessShare**      | ✓           | ✓        | ✓            | ✓                    | ✓     | ✓                 | ✗         | ✗               |
| **RowShare**         | ✓           | ✓        | ✓            | ✓                    | ✓     | ✗                 | ✗         | ✗               |
| **RowExclusive**     | ✓           | ✓        | ✓            | ✗                    | ✗     | ✗                 | ✗         | ✗               |
| **ShareUpdateExclusive** | ✓       | ✓        | ✗            | ✗                    | ✗     | ✗                 | ✗         | ✗               |
| **Share**            | ✓           | ✓        | ✗            | ✗                    | ✓     | ✗                 | ✗         | ✗               |
| **ShareRowExclusive**| ✓           | ✗        | ✗            | ✗                    | ✗     | ✗                 | ✗         | ✗               |
| **Exclusive**        | ✗           | ✗        | ✗            | ✗                    | ✗     | ✗                 | ✗         | ✗               |
| **AccessExclusive**  | ✗           | ✗        | ✗            | ✗                    | ✗     | ✗                 | ✗         | ✗               |

**Lock data structures:**

```zig
pub const LockManager = struct {
    table_locks: HashMap(TableID, TableLockList),
    row_locks: HashMap(RowID, RowLockList),
    wait_graph: WaitGraph,  // For deadlock detection

    pub fn acquireTableLock(self: *LockManager, txn_id: u64, table_id: u64, mode: LockMode) !void;
    pub fn acquireRowLock(self: *LockManager, txn_id: u64, row_id: u64, mode: LockMode) !void;
    pub fn releaseAllLocks(self: *LockManager, txn_id: u64) void;
    pub fn detectDeadlock(self: *LockManager) !?[]u64;  // Returns cycle of txn IDs
};

pub const TableLockList = struct {
    holders: ArrayList(LockHolder),  // Transactions currently holding lock
    waiters: ArrayList(LockWaiter),  // Transactions waiting for lock
};

pub const LockHolder = struct {
    txn_id: u64,
    mode: LockMode,
    count: u32,  // Lock count (for lock upgrades)
};
```

**Deadlock detection:**

Silica uses a **wait-for graph** to detect deadlocks:

1. Build graph: `Txn A → Txn B` if A waits for lock held by B
2. Run DFS to detect cycles
3. If cycle found, abort youngest transaction (smallest txn_id)

**Lock upgrade:**

If a transaction holds a lock and requests a stronger lock:
- `RowShareLock` → `RowExclusiveLock`: Allowed (no waiting txns)
- Upgrade fails if other transactions hold conflicting locks

---

## Concurrency Control

### MVCC (Multi-Version Concurrency Control)

**Purpose**: Allow concurrent reads without blocking writes, and vice versa.

**Location**: `src/tx/mvcc.zig` (600 lines)

**Tuple versioning:**

Every row (tuple) has hidden system columns:

```zig
pub const TupleHeader = struct {
    xmin: u64,  // Transaction ID that created this version
    xmax: u64,  // Transaction ID that deleted/updated this version (0 = active)
    cid: u32,   // Command ID within transaction (for statement-level visibility)
};
```

**Visibility rules:**

A tuple is visible to transaction `T` with snapshot `S` if:

```zig
pub fn isTupleVisible(tuple: TupleHeader, snapshot: Snapshot) bool {
    // Rule 1: Tuple created after snapshot → not visible
    if (tuple.xmin > snapshot.xmax) {
        return false;
    }

    // Rule 2: Tuple created by in-progress txn (not in snapshot) → not visible
    if (snapshot.isInProgress(tuple.xmin)) {
        return false;
    }

    // Rule 3: Tuple deleted before snapshot → not visible
    if (tuple.xmax != 0 and tuple.xmax < snapshot.xmin) {
        return false;
    }

    // Rule 4: Tuple deleted by committed txn in snapshot → not visible
    if (tuple.xmax != 0 and snapshot.isCommitted(tuple.xmax)) {
        return false;
    }

    // All checks passed → visible
    return true;
}
```

**Snapshot types:**

```zig
pub const Snapshot = struct {
    xmin: u64,         // Oldest active txn at snapshot time
    xmax: u64,         // Next txn ID at snapshot time
    in_progress: []u64,  // List of active txn IDs
};

pub fn takeSnapshot(txn_manager: *TransactionManager) Snapshot {
    return .{
        .xmin = txn_manager.oldest_active_txn,
        .xmax = txn_manager.next_txn_id,
        .in_progress = txn_manager.active_txns.items,  // Copy
    };
}
```

**Isolation levels:**

| Level | Snapshot Timing | Anomalies Prevented |
|-------|-----------------|---------------------|
| READ UNCOMMITTED | (not implemented) | None |
| **READ COMMITTED** | Per-statement snapshot | Dirty reads |
| **REPEATABLE READ** | Per-transaction snapshot | Dirty reads, non-repeatable reads, phantom reads |
| **SERIALIZABLE** | Per-transaction snapshot + SSI | All anomalies (equivalent to serial execution) |

**Example:**

```
Timeline:
  T1: BEGIN;
  T1: SELECT * FROM accounts;  -- Snapshot S1 taken (xmin=1, xmax=5, in_progress=[2,3])
  T2: BEGIN;
  T2: UPDATE accounts SET balance = 200 WHERE id = 1;  -- xmin=2, xmax=0
  T2: COMMIT;  -- xmin=2 now committed
  T1: SELECT * FROM accounts;  -- Still sees old value (balance=100) if REPEATABLE READ

Tuple versions:
  Row 1 (old): xmin=1, xmax=2, balance=100  → Visible to T1 (xmax=2 is committed but < xmin in S1)
  Row 1 (new): xmin=2, xmax=0, balance=200  → Not visible to T1 (xmin=2 is in S1.in_progress)
```

### Serializable Snapshot Isolation (SSI)

**Purpose**: Prevent write skew and other anomalies not prevented by snapshot isolation.

**Location**: `src/tx/mvcc.zig` (SSI implementation)

**SSI algorithm:**

Silica uses **read-write conflict detection**:

1. Track **rw-antidependencies**: T1 reads X, T2 writes X, T1 commits first
2. Build **conflict graph**: T1 → T2 if rw-antidependency exists
3. Detect **dangerous structures**: Two consecutive rw-edges forming a cycle
4. Abort one transaction in the cycle

**Data structures:**

```zig
pub const SSIManager = struct {
    read_sets: HashMap(TxnID, HashSet(RowID)),
    write_sets: HashMap(TxnID, HashSet(RowID)),
    conflict_graph: WaitGraph,

    pub fn recordRead(self: *SSIManager, txn_id: u64, row_id: u64) !void;
    pub fn recordWrite(self: *SSIManager, txn_id: u64, row_id: u64) !void;
    pub fn checkConflicts(self: *SSIManager, txn_id: u64) !void;
};
```

**Example (write skew prevention):**

```sql
-- On-call doctor scenario
-- Constraint: At least one doctor must be on call

-- T1:
BEGIN ISOLATION LEVEL SERIALIZABLE;
SELECT COUNT(*) FROM doctors WHERE on_call = true;  -- Returns 2
-- (Sees 2 doctors on call, safe to go off-call)
UPDATE doctors SET on_call = false WHERE id = 1;

-- T2 (concurrent):
BEGIN ISOLATION LEVEL SERIALIZABLE;
SELECT COUNT(*) FROM doctors WHERE on_call = true;  -- Returns 2
UPDATE doctors SET on_call = false WHERE id = 2;

-- Without SSI: Both commit → constraint violated (0 doctors on call)
-- With SSI: Second commit aborted with SerializationFailure error
```

---

## Replication

### Architecture

```
┌─────────────────┐         ┌─────────────────┐
│    Primary      │         │    Replica      │
│                 │         │                 │
│  ┌───────────┐  │         │  ┌───────────┐  │
│  │ WAL Sender│──┼────────>│  │WAL Receiver│ │
│  └───────────┘  │  TCP    │  └───────────┘  │
│        │        │         │        │        │
│  ┌───────────┐  │         │  ┌───────────┐  │
│  │  WAL File │  │         │  │  WAL File │  │
│  └───────────┘  │         │  └───────────┘  │
│        │        │         │        │        │
│  ┌───────────┐  │         │  ┌───────────┐  │
│  │  Main DB  │  │         │  │  Main DB  │  │
│  └───────────┘  │         │  └───────────┘  │
└─────────────────┘         └─────────────────┘
   Read-Write                   Read-Only
```

### WAL Sender

**Purpose**: Stream WAL records from primary to replica.

**Location**: `src/replication/sender.zig` (700 lines)

**Protocol:**

1. **Handshake**: Replica connects, sends `START_REPLICATION` command with LSN
2. **Streaming**: Primary sends WAL frames as they are written
3. **Keepalive**: Heartbeat messages every `wal_sender_timeout` seconds
4. **Feedback**: Replica sends acknowledgments (write_lsn, flush_lsn, apply_lsn)

**WAL sender state machine:**

```zig
pub const WalSender = struct {
    state: State,
    replica_addr: std.net.Address,
    start_lsn: u64,  // LSN to start streaming from
    last_sent_lsn: u64,

    pub const State = enum {
        idle,
        starting,
        catchup,      // Sending historical WAL
        streaming,    // Sending real-time WAL
        stopping,
    };

    pub fn run(self: *WalSender) !void {
        while (self.state != .stopping) {
            switch (self.state) {
                .starting => try self.sendHandshake(),
                .catchup => try self.sendCatchupWal(),
                .streaming => try self.streamRealtimeWal(),
                else => {},
            }
        }
    }
};
```

### WAL Receiver

**Purpose**: Receive WAL records from primary and apply them to replica.

**Location**: `src/replication/receiver.zig` (650 lines)

**Protocol:**

1. **Connect**: Establish TCP connection to primary
2. **Identify**: Send `IDENTIFY_SYSTEM` command
3. **Start**: Send `START_REPLICATION` command with last applied LSN
4. **Receive**: Read WAL frames from socket
5. **Apply**: Write frames to local WAL file
6. **Acknowledge**: Send feedback (write_lsn, flush_lsn, apply_lsn)

**Hot standby:**

With `hot_standby = on`, replica allows read-only queries:
- Read-only transactions use snapshot from last applied LSN
- Write transactions are rejected
- Conflicts resolved by aborting replica queries (future: cancellation protocol)

---

## Module Dependency Graph

```
┌──────────────────────────────────────────────────────────────┐
│                          main.zig                            │
│                       (Entry Point)                          │
└─────────────────────┬────────────────────────────────────────┘
                      │
         ┌────────────┴─────────────┬────────────────┐
         │                          │                │
         ▼                          ▼                ▼
    ┌─────────┐              ┌──────────┐      ┌─────────┐
    │ cli.zig │              │ tui.zig  │      │ server/ │
    │ (CLI)   │              │  (TUI)   │      │(Server) │
    └────┬────┘              └─────┬────┘      └────┬────┘
         │                         │                │
         └─────────────┬───────────┴────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │ sql/engine.zig │
              │  (Database)    │
              └────────┬───────┘
                       │
         ┌─────────────┼─────────────┐
         │             │             │
         ▼             ▼             ▼
    ┌────────┐   ┌──────────┐  ┌─────────┐
    │ sql/   │   │   tx/    │  │catalog/ │
    │executor│   │ (WAL,    │  │(Schema) │
    │        │   │  MVCC,   │  │         │
    │        │   │  Lock)   │  │         │
    └────┬───┘   └─────┬────┘  └────┬────┘
         │             │            │
         └──────┬──────┴────────────┘
                │
                ▼
         ┌──────────────┐
         │  storage/    │
         │  (B+Tree,    │
         │   Buffer,    │
         │   Page)      │
         └──────┬───────┘
                │
                ▼
         ┌──────────────┐
         │    util/     │
         │  (CRC32C,    │
         │   Varint)    │
         └──────────────┘
```

**Build order (bottom-up):**

1. `util/` — Pure functions, no dependencies
2. `storage/` — Depends on `util/`
3. `catalog/` — Depends on `storage/`
4. `tx/` — Depends on `storage/`, `catalog/`
5. `sql/` (tokenizer, parser, analyzer) — Depends on `catalog/`
6. `sql/` (planner, optimizer, executor) — Depends on `tx/`, `storage/`, `catalog/`
7. `sql/engine.zig` — Depends on all `sql/` and `tx/`
8. `server/` — Depends on `sql/engine.zig`
9. `cli.zig`, `tui.zig` — Depends on `sql/engine.zig`
10. `main.zig` — Entry point

---

## Key Algorithms

### B+Tree Bulk Loading

**Purpose**: Fast initial load of sorted data (10-100x faster than sequential inserts).

**Algorithm:**

1. Sort input data by key
2. Build leaf nodes bottom-up (fill to ~90% capacity)
3. Build internal nodes from leaf separator keys
4. Set `next_leaf` pointers for range scans

**Pseudocode:**

```
bulkLoad(sorted_kvs):
  leaves = []
  current_leaf = allocPage()

  for kv in sorted_kvs:
    if current_leaf.isFull():
      leaves.append(current_leaf)
      current_leaf = allocPage()
    current_leaf.append(kv)

  leaves.append(current_leaf)

  # Link leaves
  for i in 0..leaves.len-1:
    leaves[i].next_leaf = leaves[i+1].page_num

  # Build internal nodes
  return buildInternal(leaves)

buildInternal(nodes):
  if nodes.len == 1:
    return nodes[0]  # Root

  parents = []
  current_parent = allocPage()

  for node in nodes:
    separator_key = node.firstKey()
    if current_parent.isFull():
      parents.append(current_parent)
      current_parent = allocPage()
    current_parent.append(separator_key, node.page_num)

  parents.append(current_parent)
  return buildInternal(parents)  # Recurse
```

### Histogram-Based Selectivity Estimation

**Purpose**: Estimate how many rows match a predicate (for cost-based optimization).

**Data structure:**

```zig
pub const Histogram = struct {
    buckets: []Bucket,  // Equi-depth buckets (~10 buckets)

    pub const Bucket = struct {
        min_value: Value,
        max_value: Value,
        distinct_count: u64,
        row_count: u64,
    };
};
```

**Estimation:**

```zig
pub fn estimateSelectivity(hist: Histogram, predicate: Predicate) f64 {
    switch (predicate) {
        .equals => |val| {
            const bucket = hist.findBucket(val);
            return 1.0 / @as(f64, @floatFromInt(bucket.distinct_count));
        },
        .range => |range| {
            var total_rows: u64 = 0;
            for (hist.buckets) |bucket| {
                if (bucket.overlaps(range)) {
                    total_rows += bucket.row_count * bucket.overlapRatio(range);
                }
            }
            return @as(f64, @floatFromInt(total_rows)) / @as(f64, @floatFromInt(hist.totalRows()));
        },
    }
}
```

**Example:**

```sql
SELECT * FROM users WHERE age BETWEEN 25 AND 35;
```

Histogram:
```
Bucket 0: [18, 24], 1000 rows, 7 distinct
Bucket 1: [25, 34], 1500 rows, 10 distinct
Bucket 2: [35, 44], 1200 rows, 10 distinct
```

Estimated selectivity:
- Bucket 1 fully overlaps → 1500 rows
- Bucket 2 partially overlaps (only 35) → 1200 * (1/10) = 120 rows
- **Total**: 1620 / 3700 = 0.437 (43.7% of rows match)

### Deadlock Detection (DFS Cycle Detection)

**Purpose**: Detect cycles in wait-for graph to identify deadlocks.

**Algorithm:**

```zig
pub fn detectDeadlock(wait_graph: WaitGraph) ?[]TxnID {
    var visited = HashSet(TxnID).init(allocator);
    var stack = ArrayList(TxnID).init(allocator);

    for (wait_graph.nodes) |txn_id| {
        if (dfs(wait_graph, txn_id, &visited, &stack)) {
            return stack.items;  // Cycle found
        }
    }

    return null;  // No cycle
}

fn dfs(graph: WaitGraph, node: TxnID, visited: *HashSet, stack: *ArrayList) bool {
    if (stack.contains(node)) {
        // Cycle detected: node is already in current path
        return true;
    }

    if (visited.contains(node)) {
        // Already visited in previous path, no cycle from here
        return false;
    }

    visited.put(node);
    stack.append(node);

    for (graph.outgoing(node)) |neighbor| {
        if (dfs(graph, neighbor, visited, stack)) {
            return true;
        }
    }

    _ = stack.pop();
    return false;
}
```

**Abort strategy:**

When a cycle is detected, abort the **youngest transaction** (highest txn_id) to minimize wasted work.

---

## Performance Characteristics

### Time Complexity

| Operation | Average | Worst Case | Notes |
|-----------|---------|------------|-------|
| B+Tree insert | O(log n) | O(log n) | With splits |
| B+Tree lookup | O(log n) | O(log n) | Binary search + tree traversal |
| B+Tree delete | O(log n) | O(log n) | With merges |
| B+Tree range scan | O(log n + k) | O(log n + k) | k = result size |
| Hash index lookup | O(1) | O(n) | Worst case: all keys hash to same bucket |
| Sequential scan | O(n) | O(n) | Full table scan |
| Nested loop join | O(n × m) | O(n × m) | n, m = table sizes |
| Hash join | O(n + m) | O(n × m) | Worst case: hash collisions |
| Sort | O(n log n) | O(n log n) | External merge sort |

### Space Complexity

| Component | Memory Usage | Notes |
|-----------|--------------|-------|
| Buffer pool | `capacity × page_size` | Default: 2000 × 4096 = 8 MB |
| Transaction table | `O(active_txns)` | Typically < 1000 entries |
| Lock table | `O(locks)` | Grows with concurrent writes |
| Query arena | `O(plan_size)` | Freed after query execution |
| WAL buffer | `wal_buffers` | Default: 16 MB |

---

## Future Enhancements

### Planned Features

1. **Parallel query execution**: Intra-query parallelism for large scans/aggregates
2. **Adaptive indexing**: Automatic index creation based on query patterns
3. **Compression**: Page-level compression (LZ4, Zstd)
4. **Partitioning**: Range, hash, list partitioning
5. **Materialized views**: Incremental maintenance
6. **Query compilation**: JIT compilation for hot paths (LLVM backend)
7. **Distributed transactions**: 2PC for multi-database consistency

### Optimization Opportunities

- **Columnar storage**: For OLAP workloads (analytical queries)
- **Vectorized execution**: Process rows in batches (SIMD)
- **Index-only scans**: Avoid table access when index covers query
- **Join pruning**: Skip outer join sides when not needed
- **Predicate reordering**: Evaluate cheap filters first

---

## Appendix

### File Reference

| Module | Lines | Tests | Description |
|--------|-------|-------|-------------|
| `src/main.zig` | 250 | 0 | Entry point, CLI/TUI routing |
| `src/cli.zig` | 380 | 0 | CLI argument parsing |
| `src/tui.zig` | 1376 | 18 | TUI database browser |
| `src/storage/page.zig` | 530 | 45 | Page manager, file I/O |
| `src/storage/btree.zig` | 4300 | 53 | B+Tree index |
| `src/storage/buffer_pool.zig` | 480 | 32 | LRU buffer pool |
| `src/storage/overflow.zig` | 280 | 18 | Overflow page handling |
| `src/sql/tokenizer.zig` | 1100 | 142 | SQL lexer |
| `src/sql/parser.zig` | 3500 | 785 | SQL parser (AST) |
| `src/sql/analyzer.zig` | 4100 | 198 | Semantic analysis |
| `src/sql/planner.zig` | 1800 | 87 | Logical plan generation |
| `src/sql/optimizer.zig` | 1500 | 64 | Query optimization |
| `src/sql/executor.zig` | 7000 | 360 | Query execution |
| `src/sql/engine.zig` | 9000 | 515 | Database integration |
| `src/tx/wal.zig` | 900 | 78 | Write-ahead log |
| `src/tx/mvcc.zig` | 600 | 95 | MVCC visibility |
| `src/tx/lock.zig` | 800 | 60 | Lock manager |
| `src/server/wire.zig` | 1900 | 58 | PostgreSQL wire protocol |
| `src/server/connection.zig` | 850 | 34 | Connection handling |
| `src/util/checksum.zig` | 120 | 12 | CRC32C checksums |
| `src/util/varint.zig` | 150 | 19 | Variable-length integers |
| **Total** | **~40,000** | **2766** | 43 modules |

### Glossary

- **ACID**: Atomicity, Consistency, Isolation, Durability
- **AST**: Abstract Syntax Tree
- **B+Tree**: Self-balancing tree data structure for sorted data
- **CRC32C**: Cyclic Redundancy Check (Castagnoli polynomial)
- **CTE**: Common Table Expression (WITH clause)
- **DML**: Data Manipulation Language (SELECT, INSERT, UPDATE, DELETE)
- **DDL**: Data Definition Language (CREATE, ALTER, DROP)
- **LSN**: Log Sequence Number (WAL position)
- **LRU**: Least Recently Used (eviction policy)
- **MVCC**: Multi-Version Concurrency Control
- **PITR**: Point-In-Time Recovery
- **SSI**: Serializable Snapshot Isolation
- **TID**: Tuple Identifier (row ID)
- **WAL**: Write-Ahead Log

---

**Questions or Contributions?**

- GitHub: [github.com/yusa-imit/silica/issues](https://github.com/yusa-imit/silica/issues)
- Email: yusa@example.com

**Last Updated**: 2026-03-25
**Version**: v0.12 (Phase 12: Production Readiness)
