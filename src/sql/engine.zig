//! Database Engine — top-level integration layer for Silica.
//!
//! Connects the full SQL pipeline:
//!   SQL text → Tokenizer → Parser → AST
//!   AST → Analyzer (validate against catalog)
//!   AST → Planner → Logical Plan
//!   Logical Plan → Optimizer → Optimized Plan
//!   Optimized Plan → Executor → Results
//!
//! Provides the primary `Database` API:
//!   var db = try Database.open(allocator, "app.db", .{});
//!   defer db.close();
//!   var result = try db.exec("SELECT * FROM users");

const std = @import("std");
const Allocator = std.mem.Allocator;

const tokenizer_mod = @import("tokenizer.zig");
const parser_mod = @import("parser.zig");
const ast_mod = @import("ast.zig");
const analyzer_mod = @import("analyzer.zig");
const catalog_mod = @import("catalog.zig");
const planner_mod = @import("planner.zig");
const optimizer_mod = @import("optimizer.zig");
const executor_mod = @import("executor.zig");
const btree_mod = @import("../storage/btree.zig");
const buffer_pool_mod = @import("../storage/buffer_pool.zig");
const page_mod = @import("../storage/page.zig");

const Tokenizer = tokenizer_mod.Tokenizer;
const Parser = parser_mod.Parser;
const AstArena = ast_mod.AstArena;
const Analyzer = analyzer_mod.Analyzer;
const SchemaProvider = analyzer_mod.SchemaProvider;
const Catalog = catalog_mod.Catalog;
const ColumnInfo = catalog_mod.ColumnInfo;
const TableInfo = catalog_mod.TableInfo;
const Planner = planner_mod.Planner;
const Optimizer = optimizer_mod.Optimizer;
const PlanNode = planner_mod.PlanNode;
const LogicalPlan = planner_mod.LogicalPlan;
const PlanType = planner_mod.PlanType;
const BTree = btree_mod.BTree;
const BufferPool = buffer_pool_mod.BufferPool;
const Pager = page_mod.Pager;
const wal_mod = @import("../tx/wal.zig");
const Wal = wal_mod.Wal;
const mvcc_mod = @import("../tx/mvcc.zig");
const TransactionManager = mvcc_mod.TransactionManager;
const TupleHeader = mvcc_mod.TupleHeader;
const Snapshot = mvcc_mod.Snapshot;
const IsolationLevel = mvcc_mod.IsolationLevel;

const Value = executor_mod.Value;
const Row = executor_mod.Row;
const RowIterator = executor_mod.RowIterator;
const ExecError = executor_mod.ExecError;
const ScanOp = executor_mod.ScanOp;
const FilterOp = executor_mod.FilterOp;
const ProjectOp = executor_mod.ProjectOp;
const LimitOp = executor_mod.LimitOp;
const SortOp = executor_mod.SortOp;
const AggregateOp = executor_mod.AggregateOp;
const NestedLoopJoinOp = executor_mod.NestedLoopJoinOp;
const ValuesOp = executor_mod.ValuesOp;
const EmptyOp = executor_mod.EmptyOp;
const IndexScanOp = executor_mod.IndexScanOp;
const MvccContext = executor_mod.MvccContext;
const serializeRow = executor_mod.serializeRow;
const evalExpr = executor_mod.evalExpr;

// ── MVCC Row Detection ──────────────────────────────────────────────

/// Detect whether raw row data has an MVCC tuple header.
/// Delegates to mvcc_mod.isVersionedRow for consistent detection.
fn isVersionedRowData(data: []const u8) bool {
    return mvcc_mod.isVersionedRow(data);
}

// ── Index Key Encoding ───────────────────────────────────────────────

/// Encode a Value as an index key for B+Tree storage.
/// Integer values use 8-byte big-endian encoding with sign flip for correct
/// lexicographic ordering. Text values use raw bytes. NULL encodes as empty.
fn valueToIndexKey(allocator: Allocator, val: Value) ![]u8 {
    return switch (val) {
        .integer => |i| {
            const buf = try allocator.alloc(u8, 8);
            // Flip the sign bit for correct lexicographic ordering of signed integers:
            // -3, -2, -1, 0, 1, 2, 3 should sort correctly as byte sequences.
            const unsigned: u64 = @bitCast(i);
            const flipped = unsigned ^ (@as(u64, 1) << 63);
            std.mem.writeInt(u64, buf[0..8], flipped, .big);
            return buf;
        },
        .text => |t| try allocator.dupe(u8, t),
        .real => |r| {
            const buf = try allocator.alloc(u8, 8);
            const bits: u64 = @bitCast(r);
            // For floats: if positive, flip sign bit; if negative, flip all bits
            const flipped = if (r >= 0) bits ^ (@as(u64, 1) << 63) else ~bits;
            std.mem.writeInt(u64, buf[0..8], flipped, .big);
            return buf;
        },
        .boolean => |b| {
            const buf = try allocator.alloc(u8, 1);
            buf[0] = if (b) 1 else 0;
            return buf;
        },
        .null_value => try allocator.alloc(u8, 0),
        .blob => |b| try allocator.dupe(u8, b),
    };
}

/// Encode a literal integer as an index key (same encoding as valueToIndexKey).
fn integerToIndexKey(allocator: Allocator, i: i64) ![]u8 {
    const buf = try allocator.alloc(u8, 8);
    const unsigned: u64 = @bitCast(i);
    const flipped = unsigned ^ (@as(u64, 1) << 63);
    std.mem.writeInt(u64, buf[0..8], flipped, .big);
    return buf;
}

/// Encode a literal string as an index key.
fn stringToIndexKey(allocator: Allocator, s: []const u8) ![]u8 {
    return allocator.dupe(u8, s);
}

// ── Index Selection ──────────────────────────────────────────────────

/// Result of extracting an equality predicate `column = literal`.
const EqualityPredicate = struct {
    column_name: []const u8,
    value_type: union(enum) {
        integer: i64,
        text: []const u8,
    },
};

/// Extract a simple `column_ref = literal` or `literal = column_ref` pattern
/// from an expression. Returns null for non-equality or complex predicates.
fn extractEqualityPredicate(expr: *const ast_mod.Expr) ?EqualityPredicate {
    if (expr.* != .binary_op) return null;
    const op = expr.binary_op;
    if (op.op != .equal) return null;

    // Try: column_ref = literal
    if (op.left.* == .column_ref and op.right.* != .column_ref) {
        const col = op.left.column_ref;
        // Only handle unqualified column refs (no table prefix) for now
        if (col.prefix != null) return null;
        if (op.right.* == .integer_literal) {
            return .{ .column_name = col.name, .value_type = .{ .integer = op.right.integer_literal } };
        }
        if (op.right.* == .string_literal) {
            return .{ .column_name = col.name, .value_type = .{ .text = op.right.string_literal } };
        }
    }

    // Try: literal = column_ref
    if (op.right.* == .column_ref and op.left.* != .column_ref) {
        const col = op.right.column_ref;
        if (col.prefix != null) return null;
        if (op.left.* == .integer_literal) {
            return .{ .column_name = col.name, .value_type = .{ .integer = op.left.integer_literal } };
        }
        if (op.left.* == .string_literal) {
            return .{ .column_name = col.name, .value_type = .{ .text = op.left.string_literal } };
        }
    }

    return null;
}

// ── Database ─────────────────────────────────────────────────────────

pub const OpenOptions = struct {
    page_size: u32 = 4096,
    cache_size: u32 = 2000,
    wal_mode: bool = false,
};

pub const EngineError = error{
    OutOfMemory,
    ParseError,
    AnalysisError,
    PlanError,
    ExecutionError,
    StorageError,
    TableNotFound,
    TableAlreadyExists,
    InvalidData,
    TransactionError,
    NoActiveTransaction,
};

/// Active transaction context for the current session.
pub const TransactionContext = struct {
    xid: u32,
    isolation: IsolationLevel,
    /// For REPEATABLE READ / SERIALIZABLE: snapshot taken at BEGIN.
    /// For READ COMMITTED: null (fresh snapshot per statement).
    snapshot: ?Snapshot = null,

    pub fn deinit(self: *TransactionContext) void {
        if (self.snapshot) |*snap| snap.deinit();
    }
};

/// Result of executing a SQL statement.
pub const QueryResult = struct {
    /// For SELECT: row iterator (caller must call close()).
    rows: ?RowIterator = null,
    /// For DML: number of rows affected.
    rows_affected: u64 = 0,
    /// Human-readable message.
    message: []const u8 = "",
    /// The AST arena that owns all plan/expr memory. Must be kept alive
    /// while iterating rows (expressions reference arena memory).
    /// Caller must deinit after closing the row iterator.
    _arena: ?*AstArena = null,
    /// Heap-allocated operator chain. Caller must free these.
    _ops: ?*OperatorChain = null,

    pub fn close(self: *QueryResult, allocator: Allocator) void {
        if (self.rows) |r| r.close();
        if (self._ops) |ops| ops.deinit(allocator);
        if (self._arena) |arena| {
            arena.deinit();
            allocator.destroy(arena);
        }
    }
};

/// Holds heap-allocated operator structs to keep them alive during iteration.
const OperatorChain = struct {
    scan: ?*ScanOp = null,
    index_scan: ?*IndexScanOp = null,
    filter: ?*FilterOp = null,
    project: ?*ProjectOp = null,
    limit: ?*LimitOp = null,
    sort: ?*SortOp = null,
    aggregate: ?*AggregateOp = null,
    join: ?*NestedLoopJoinOp = null,
    values: ?*ValuesOp = null,
    empty: ?*EmptyOp = null,
    // For joins, we may need a second scan
    scan2: ?*ScanOp = null,
    /// Per-statement RC snapshot that needs cleanup (owned by this chain).
    rc_snapshot: ?Snapshot = null,

    fn freeScanColNames(allocator: Allocator, s: *ScanOp) void {
        for (s.col_names) |name| allocator.free(@constCast(name));
        allocator.free(s.col_names);
    }

    fn deinit(self: *OperatorChain, allocator: Allocator) void {
        // Free per-statement RC snapshot if allocated
        if (self.rc_snapshot) |*snap| snap.deinit();
        // Operators are closed via RowIterator.close() already.
        // We just free the heap-allocated structs.
        if (self.scan) |s| {
            freeScanColNames(allocator, s);
            allocator.destroy(s);
        }
        if (self.scan2) |s| {
            freeScanColNames(allocator, s);
            allocator.destroy(s);
        }
        if (self.index_scan) |is| {
            for (is.col_names) |name| allocator.free(@constCast(name));
            allocator.free(is.col_names);
            allocator.free(is.lookup_key);
            allocator.destroy(is);
        }
        if (self.filter) |f| allocator.destroy(f);
        if (self.project) |p| allocator.destroy(p);
        if (self.limit) |l| allocator.destroy(l);
        if (self.sort) |s| allocator.destroy(s);
        if (self.aggregate) |a| allocator.destroy(a);
        if (self.join) |j| allocator.destroy(j);
        if (self.values) |v| allocator.destroy(v);
        if (self.empty) |e| allocator.destroy(e);
        allocator.destroy(self);
    }
};

/// The main database handle. Connects all engine layers.
pub const Database = struct {
    allocator: Allocator,
    pager: *Pager,
    pool: *BufferPool,
    catalog: Catalog,
    /// Arena for schema lookups — TableInfo allocated here is freed between
    /// exec calls. Avoids the analyzer needing to manage TableInfo lifetimes.
    schema_arena: std.heap.ArenaAllocator,
    /// Optional WAL for transaction durability.
    wal: ?*Wal = null,
    /// Transaction manager for MVCC.
    tm: TransactionManager,
    /// Currently active transaction (null = auto-commit mode).
    current_txn: ?TransactionContext = null,

    /// Open or create a database file.
    pub fn open(allocator: Allocator, path: []const u8, opts: OpenOptions) !Database {
        const pager = try allocator.create(Pager);
        errdefer allocator.destroy(pager);
        pager.* = try Pager.init(allocator, path, .{ .page_size = opts.page_size });
        errdefer pager.deinit();

        const is_new = pager.page_count <= 1;

        const pool = try allocator.create(BufferPool);
        errdefer allocator.destroy(pool);
        pool.* = try BufferPool.init(allocator, pager, opts.cache_size);
        errdefer pool.deinit();

        // Initialize WAL if requested
        var wal_ptr: ?*Wal = null;
        if (opts.wal_mode) {
            const w = try allocator.create(Wal);
            errdefer allocator.destroy(w);
            w.* = try Wal.init(allocator, path, pager.page_size);
            pool.setWal(w);
            wal_ptr = w;
        }

        var cat = try Catalog.init(allocator, pool, is_new);
        _ = &cat;

        return .{
            .allocator = allocator,
            .pager = pager,
            .pool = pool,
            .catalog = cat,
            .schema_arena = std.heap.ArenaAllocator.init(allocator),
            .wal = wal_ptr,
            .tm = TransactionManager.init(allocator),
        };
    }

    /// Close the database, flushing all dirty pages.
    pub fn close(self: *Database) void {
        // Abort any active transaction
        if (self.current_txn) |*txn| {
            self.tm.abort(txn.xid) catch {};
            txn.deinit();
            self.current_txn = null;
        }

        self.tm.deinit();
        self.schema_arena.deinit();

        // Checkpoint WAL before closing — writes committed pages to main DB
        if (self.wal) |w| {
            w.checkpoint(self.pager) catch {};
            w.deinit();
            self.allocator.destroy(w);
            // Clear WAL pointer so BufferPool.deinit doesn't try to use it
            self.pool.wal = null;
        }

        self.pool.deinit();
        self.pager.deinit();
        self.allocator.destroy(self.pool);
        self.allocator.destroy(self.pager);
    }

    // ── Transaction Management ────────────────────────────────────────

    /// Begin an explicit transaction with the given isolation level.
    pub fn beginTransaction(self: *Database, isolation: IsolationLevel) EngineError!void {
        if (self.current_txn != null) return EngineError.TransactionError; // Already in a transaction

        const xid = self.tm.begin(isolation) catch return EngineError.TransactionError;
        self.current_txn = .{
            .xid = xid,
            .isolation = isolation,
        };

        // For REPEATABLE READ / SERIALIZABLE, take snapshot at BEGIN time
        if (isolation != .read_committed) {
            const snap = self.tm.getSnapshot(xid) catch return EngineError.TransactionError;
            self.current_txn.?.snapshot = snap;
        }
    }

    /// Commit the current transaction.
    pub fn commitTransaction(self: *Database) EngineError!void {
        const txn = self.current_txn orelse return EngineError.NoActiveTransaction;
        self.tm.commit(txn.xid) catch return EngineError.TransactionError;
        self.commitWal() catch {};
        var ctx = self.current_txn.?;
        ctx.deinit();
        self.current_txn = null;
    }

    /// Rollback the current transaction.
    pub fn rollbackTransaction(self: *Database) EngineError!void {
        const txn = self.current_txn orelse return EngineError.NoActiveTransaction;
        self.tm.abort(txn.xid) catch return EngineError.TransactionError;
        if (self.wal) |w| {
            w.rollback() catch {};
        }
        var ctx = self.current_txn.?;
        ctx.deinit();
        self.current_txn = null;
    }

    /// Get MVCC context for the current statement.
    /// If `ops` is provided, RC snapshots are stored in `ops.rc_snapshot` for cleanup.
    /// If `ops` is null (UPDATE/DELETE), the caller must free the snapshot.
    fn getMvccContext(self: *Database) EngineError!?MvccContext {
        return self.getMvccContextWithOps(null);
    }

    /// Get MVCC context, optionally storing RC snapshot in OperatorChain.
    fn getMvccContextWithOps(self: *Database, ops: ?*OperatorChain) EngineError!?MvccContext {
        if (self.current_txn) |txn| {
            const cid = self.tm.getCurrentCid(txn.xid) catch return EngineError.TransactionError;

            // For READ COMMITTED: fresh snapshot per statement
            if (txn.isolation == .read_committed) {
                const snap = self.tm.getSnapshot(txn.xid) catch return EngineError.TransactionError;
                // If we have an OperatorChain, store the snapshot there for cleanup
                if (ops) |o| {
                    o.rc_snapshot = snap;
                }
                return MvccContext{
                    .snapshot = snap,
                    .current_xid = txn.xid,
                    .current_cid = cid,
                };
            }
            // REPEATABLE READ / SERIALIZABLE: use stored snapshot (not owned by us)
            if (txn.snapshot) |snap| {
                return MvccContext{
                    .snapshot = snap,
                    .current_xid = txn.xid,
                    .current_cid = cid,
                };
            }
        }
        // No explicit transaction — return null (auto-commit, no MVCC filtering needed)
        return null;
    }

    /// Advance the command ID for the current transaction.
    /// Returns the CID to use for this statement's writes.
    fn advanceStatementCid(self: *Database) EngineError!u16 {
        if (self.current_txn) |txn| {
            return self.tm.advanceCid(txn.xid) catch return EngineError.TransactionError;
        }
        return 0; // Auto-commit: CID doesn't matter
    }

    /// Execute a SQL statement and return results.
    pub fn exec(self: *Database, sql: []const u8) EngineError!QueryResult {
        // Delegate to execSQL which handles both DDL and DML correctly
        return self.execSQL(sql);
    }

    fn executePlan(self: *Database, arena: *AstArena, plan: LogicalPlan) EngineError!QueryResult {
        // Advance CID in explicit transactions (one CID per statement)
        if (self.current_txn != null) {
            _ = self.advanceStatementCid() catch {};
        }

        const result = switch (plan.plan_type) {
            .select_query => self.executeSelect(arena, plan),
            .insert => self.executeInsert(arena, plan),
            .update => self.executeUpdate(arena, plan),
            .delete => self.executeDelete(arena, plan),
            .create_table => self.executeCreateTable(arena, plan),
            .drop_table => self.executeDropTable(arena, plan),
            else => {
                arena.deinit();
                self.allocator.destroy(arena);
                return .{ .message = "OK" };
            },
        };

        // Commit WAL after successful DML/DDL in auto-commit mode only.
        // In explicit transactions, WAL is committed at COMMIT time.
        if (self.current_txn == null) {
            if (result) |r| {
                if (r.rows == null) {
                    self.commitWal() catch {};
                }
            } else |_| {}
        }

        return result;
    }

    /// Flush dirty buffer pool pages and commit the WAL transaction.
    fn commitWal(self: *Database) !void {
        if (self.wal) |w| {
            try self.pool.flushAll();
            try w.commit(self.pager.page_count);
        }
    }

    // ── SELECT execution ──────────────────────────────────────────────

    fn executeSelect(self: *Database, arena: *AstArena, plan: LogicalPlan) EngineError!QueryResult {
        const ops = self.allocator.create(OperatorChain) catch return EngineError.OutOfMemory;
        ops.* = .{};
        errdefer ops.deinit(self.allocator);

        const iter = self.buildIterator(plan.root, ops) catch return EngineError.ExecutionError;
        return .{
            .rows = iter,
            ._arena = arena,
            ._ops = ops,
        };
    }

    /// Recursively translate a PlanNode tree into an executor RowIterator chain.
    fn buildIterator(self: *Database, node: *const PlanNode, ops: *OperatorChain) EngineError!RowIterator {
        return switch (node.*) {
            .scan => |s| self.buildScan(s, ops),
            .filter => |f| self.buildFilter(f, ops),
            .project => |p| self.buildProject(p, ops),
            .limit => |l| self.buildLimit(l, ops),
            .sort => |s| self.buildSort(s, ops),
            .aggregate => |a| self.buildAggregate(a, ops),
            .join => |j| self.buildJoin(j, ops),
            .values => |v| self.buildValues(v, ops),
            .empty => self.buildEmpty(ops),
        };
    }

    fn buildScan(self: *Database, scan: PlanNode.Scan, ops: *OperatorChain) EngineError!RowIterator {
        // Look up table in catalog to get data root page ID
        var table_info = self.catalog.getTable(scan.table) catch return EngineError.TableNotFound;
        defer table_info.deinit(self.allocator);

        // Build column names for the scan
        const col_names = self.allocator.alloc([]const u8, table_info.columns.len) catch return EngineError.OutOfMemory;
        for (table_info.columns, 0..) |col, i| {
            if (scan.alias) |alias| {
                col_names[i] = std.fmt.allocPrint(self.allocator, "{s}.{s}", .{ alias, col.name }) catch return EngineError.OutOfMemory;
            } else {
                col_names[i] = self.allocator.dupe(u8, col.name) catch return EngineError.OutOfMemory;
            }
        }

        const scan_op = self.allocator.create(ScanOp) catch return EngineError.OutOfMemory;
        scan_op.* = ScanOp.init(self.allocator, self.pool, table_info.data_root_page_id, col_names);
        // Set MVCC context for visibility filtering (RC snapshot stored in ops for cleanup)
        scan_op.mvcc_ctx = try self.getMvccContextWithOps(ops);
        scan_op.initCursor(); // Must be called after heap placement

        if (ops.scan == null) {
            ops.scan = scan_op;
        } else {
            ops.scan2 = scan_op;
        }
        return scan_op.iterator();
    }

    fn buildFilter(self: *Database, filter: PlanNode.Filter, ops: *OperatorChain) EngineError!RowIterator {
        // Try index selection: if the predicate is `col = literal` and the
        // input is a table scan on a table with an index on that column,
        // use IndexScanOp for a direct B+Tree lookup instead of full scan.
        if (filter.input.* == .scan) {
            if (self.tryBuildIndexScan(filter.input.scan, filter.predicate, ops)) |iter| {
                return iter;
            }
        }

        const input = try self.buildIterator(filter.input, ops);
        const filter_op = self.allocator.create(FilterOp) catch return EngineError.OutOfMemory;
        filter_op.* = FilterOp.init(self.allocator, input, filter.predicate);
        ops.filter = filter_op;
        return filter_op.iterator();
    }

    /// Try to use a secondary index for a predicate on a scan.
    /// Returns a RowIterator if successful, null if index scan is not applicable.
    fn tryBuildIndexScan(self: *Database, scan: PlanNode.Scan, predicate: *const ast_mod.Expr, ops: *OperatorChain) ?RowIterator {
        // Only handle simple equality predicates: column_ref = literal
        const eq = extractEqualityPredicate(predicate) orelse return null;

        // Look up the table to find index info
        var table_info = self.catalog.getTable(scan.table) catch return null;
        defer table_info.deinit(self.allocator);

        // Check if there's an index on the referenced column
        const col_name = eq.column_name;
        const idx_info = table_info.findIndex(col_name) orelse return null;

        // Encode the literal value as an index key
        const idx_key = switch (eq.value_type) {
            .integer => |v| integerToIndexKey(self.allocator, v) catch return null,
            .text => |v| stringToIndexKey(self.allocator, v) catch return null,
        };

        // Build column names for the scan
        const col_names = self.allocator.alloc([]const u8, table_info.columns.len) catch return null;
        for (table_info.columns, 0..) |col, i| {
            if (scan.alias) |alias| {
                col_names[i] = std.fmt.allocPrint(self.allocator, "{s}.{s}", .{ alias, col.name }) catch return null;
            } else {
                col_names[i] = self.allocator.dupe(u8, col.name) catch return null;
            }
        }

        const idx_op = self.allocator.create(IndexScanOp) catch return null;
        idx_op.* = IndexScanOp.init(
            self.allocator,
            self.pool,
            table_info.data_root_page_id,
            idx_info.root_page_id,
            idx_key,
            col_names,
        );
        // Set MVCC context for visibility filtering (RC snapshot stored in ops for cleanup)
        idx_op.mvcc_ctx = self.getMvccContextWithOps(ops) catch return null;
        ops.index_scan = idx_op;
        return idx_op.iterator();
    }

    fn buildProject(self: *Database, project: PlanNode.Project, ops: *OperatorChain) EngineError!RowIterator {
        const input = try self.buildIterator(project.input, ops);

        // Check if this is a SELECT * (single column_ref with name "*")
        if (project.columns.len == 1) {
            if (project.columns[0].expr.* == .column_ref) {
                const ref = project.columns[0].expr.column_ref;
                if (std.mem.eql(u8, ref.name, "*") and ref.prefix == null) {
                    // SELECT * — pass through all columns without projection
                    return input;
                }
            }
        }

        const project_op = self.allocator.create(ProjectOp) catch return EngineError.OutOfMemory;
        project_op.* = ProjectOp.init(self.allocator, input, project.columns);
        ops.project = project_op;
        return project_op.iterator();
    }

    fn buildLimit(self: *Database, limit: PlanNode.Limit, ops: *OperatorChain) EngineError!RowIterator {
        const input = try self.buildIterator(limit.input, ops);

        // Evaluate limit/offset expressions (they should be integer literals)
        const empty_row = Row{ .columns = &.{}, .values = &.{}, .allocator = self.allocator };
        const limit_count: ?u64 = if (limit.limit_expr) |expr| blk: {
            const val = evalExpr(self.allocator, expr, &empty_row) catch break :blk null;
            defer val.free(self.allocator);
            break :blk if (val.toInteger()) |i| @intCast(i) else null;
        } else null;
        const offset_count: u64 = if (limit.offset_expr) |expr| blk: {
            const val = evalExpr(self.allocator, expr, &empty_row) catch break :blk 0;
            defer val.free(self.allocator);
            break :blk if (val.toInteger()) |i| @intCast(i) else 0;
        } else 0;

        const limit_op = self.allocator.create(LimitOp) catch return EngineError.OutOfMemory;
        limit_op.* = LimitOp.init(self.allocator, input, limit_count, offset_count);
        ops.limit = limit_op;
        return limit_op.iterator();
    }

    fn buildSort(self: *Database, sort: PlanNode.Sort, ops: *OperatorChain) EngineError!RowIterator {
        const input = try self.buildIterator(sort.input, ops);
        const sort_op = self.allocator.create(SortOp) catch return EngineError.OutOfMemory;
        sort_op.* = SortOp.init(self.allocator, input, sort.order_by);
        ops.sort = sort_op;
        return sort_op.iterator();
    }

    fn buildAggregate(self: *Database, agg: PlanNode.Aggregate, ops: *OperatorChain) EngineError!RowIterator {
        const input = try self.buildIterator(agg.input, ops);
        const agg_op = self.allocator.create(AggregateOp) catch return EngineError.OutOfMemory;
        agg_op.* = AggregateOp.init(self.allocator, input, agg.group_by, agg.aggregates);
        ops.aggregate = agg_op;
        return agg_op.iterator();
    }

    fn buildJoin(self: *Database, join: PlanNode.Join, ops: *OperatorChain) EngineError!RowIterator {
        const left = try self.buildIterator(join.left, ops);
        const right = try self.buildIterator(join.right, ops);
        const join_op = self.allocator.create(NestedLoopJoinOp) catch return EngineError.OutOfMemory;
        join_op.* = NestedLoopJoinOp.init(self.allocator, left, right, join.join_type, join.on_condition);
        ops.join = join_op;
        return join_op.iterator();
    }

    fn buildValues(self: *Database, values: PlanNode.Values, ops: *OperatorChain) EngineError!RowIterator {
        const col_names: []const []const u8 = if (values.columns) |cols| cols else &.{};
        const values_op = self.allocator.create(ValuesOp) catch return EngineError.OutOfMemory;
        values_op.* = ValuesOp.init(self.allocator, col_names, values.rows);
        ops.values = values_op;
        return values_op.iterator();
    }

    fn buildEmpty(self: *Database, ops: *OperatorChain) EngineError!RowIterator {
        const empty_op = self.allocator.create(EmptyOp) catch return EngineError.OutOfMemory;
        empty_op.* = .{};
        ops.empty = empty_op;
        return empty_op.iterator();
    }

    // ── INSERT execution ──────────────────────────────────────────────

    fn executeInsert(self: *Database, arena: *AstArena, plan: LogicalPlan) EngineError!QueryResult {
        defer {
            arena.deinit();
            self.allocator.destroy(arena);
        }

        const values_node = switch (plan.root.*) {
            .values => |v| v,
            else => return EngineError.ExecutionError,
        };

        // Look up table to get data root page ID and column info
        var table_info = self.catalog.getTable(values_node.table) catch return EngineError.TableNotFound;
        defer table_info.deinit(self.allocator);

        var tree = BTree.init(self.pool, table_info.data_root_page_id);
        const empty_row = Row{ .columns = &.{}, .values = &.{}, .allocator = self.allocator };

        // Find the next available row key by scanning to the end
        var next_key: u64 = self.findNextRowKey(&tree) catch return EngineError.StorageError;

        var rows_inserted: u64 = 0;
        for (values_node.rows) |row_exprs| {
            // Evaluate value expressions
            const vals = self.allocator.alloc(Value, row_exprs.len) catch return EngineError.OutOfMemory;
            var inited: usize = 0;
            defer {
                for (vals[0..inited]) |v| v.free(self.allocator);
                self.allocator.free(vals);
            }

            for (row_exprs, 0..) |expr, i| {
                vals[i] = evalExpr(self.allocator, expr, &empty_row) catch return EngineError.ExecutionError;
                inited += 1;
            }

            // Serialize the row (with MVCC header if in a transaction)
            const row_data = if (self.current_txn) |txn| blk: {
                const cid = self.tm.getCurrentCid(txn.xid) catch return EngineError.TransactionError;
                const header = TupleHeader.forInsert(txn.xid, cid);
                break :blk mvcc_mod.serializeVersionedRow(self.allocator, header, vals) catch return EngineError.OutOfMemory;
            } else serializeRow(self.allocator, vals) catch return EngineError.OutOfMemory;
            defer self.allocator.free(row_data);

            // Generate key: 8-byte big-endian u64 for lexicographic ordering
            var key_buf: [8]u8 = undefined;
            std.mem.writeInt(u64, &key_buf, next_key, .big);
            const key = self.allocator.dupe(u8, &key_buf) catch return EngineError.OutOfMemory;
            defer self.allocator.free(key);
            next_key += 1;

            // Insert into B+Tree
            tree.insert(key, row_data) catch return EngineError.StorageError;

            // Update root page ID in case of B+Tree split
            if (tree.root_page_id != table_info.data_root_page_id) {
                self.updateTableRootPage(values_node.table, tree.root_page_id) catch return EngineError.StorageError;
                table_info.data_root_page_id = tree.root_page_id;
            }

            // Maintain secondary indexes
            self.insertIndexEntries(values_node.table, &table_info, vals, &key_buf) catch return EngineError.StorageError;

            rows_inserted += 1;
        }

        return .{
            .rows_affected = rows_inserted,
            .message = "INSERT",
        };
    }

    /// Find the next available row key by scanning the B+Tree for the max existing key.
    /// Keys are 8-byte big-endian u64 for lexicographic ordering.
    fn findNextRowKey(self: *Database, tree: *BTree) !u64 {
        var cursor = btree_mod.Cursor.init(self.allocator, tree);
        defer cursor.deinit();

        try cursor.seekFirst();

        var max_key: u64 = 0;
        var found_any = false;
        while (try cursor.next()) |entry| {
            defer self.allocator.free(entry.key);
            defer self.allocator.free(entry.value);
            if (entry.key.len == 8) {
                const k = std.mem.readInt(u64, entry.key[0..8], .big);
                if (!found_any or k >= max_key) {
                    max_key = k;
                    found_any = true;
                }
            }
        }

        return if (found_any) max_key + 1 else 0;
    }

    /// Update a table's root page ID in the catalog after a B+Tree split.
    fn updateTableRootPage(self: *Database, table_name: []const u8, new_root: u32) !void {
        // Re-read the table info, update root, re-serialize and update
        var info = try self.catalog.getTable(table_name);
        defer info.deinit(self.allocator);

        const value = try catalog_mod.serializeTableFull(
            self.allocator,
            info.columns,
            info.table_constraints,
            info.indexes,
            new_root,
        );
        defer self.allocator.free(value);

        // Delete and re-insert in the schema B+Tree
        self.catalog.tree.delete(table_name) catch return;
        self.catalog.tree.insert(table_name, value) catch return;
    }

    /// Update an index's root page ID in the catalog after a B+Tree split.
    fn updateIndexRootPage(self: *Database, table_name: []const u8, col_name: []const u8, new_idx_root: u32) !void {
        var info = try self.catalog.getTable(table_name);
        defer info.deinit(self.allocator);

        // Build a mutable copy of indexes with updated root page
        const new_indexes = try self.allocator.alloc(catalog_mod.IndexInfo, info.indexes.len);
        defer self.allocator.free(new_indexes);
        for (info.indexes, 0..) |idx, i| {
            new_indexes[i] = idx;
            if (std.ascii.eqlIgnoreCase(idx.column_name, col_name)) {
                new_indexes[i].root_page_id = new_idx_root;
            }
        }

        const value = try catalog_mod.serializeTableFull(
            self.allocator,
            info.columns,
            info.table_constraints,
            new_indexes,
            info.data_root_page_id,
        );
        defer self.allocator.free(value);

        self.catalog.tree.delete(table_name) catch return;
        self.catalog.tree.insert(table_name, value) catch return;
    }

    // ── Index Maintenance ─────────────────────────────────────────────

    /// Insert index entries for all indexed columns of a newly inserted row.
    fn insertIndexEntries(self: *Database, table_name: []const u8, table_info: *const catalog_mod.TableInfo, vals: []const Value, row_key: []const u8) !void {
        for (table_info.indexes) |idx| {
            if (idx.column_index >= vals.len) continue;
            const idx_key = valueToIndexKey(self.allocator, vals[idx.column_index]) catch continue;
            defer self.allocator.free(idx_key);
            const idx_val = try self.allocator.dupe(u8, row_key);
            defer self.allocator.free(idx_val);

            var idx_tree = BTree.init(self.pool, idx.root_page_id);
            idx_tree.insert(idx_key, idx_val) catch {};

            if (idx_tree.root_page_id != idx.root_page_id) {
                self.updateIndexRootPage(table_name, idx.column_name, idx_tree.root_page_id) catch {};
            }
        }
    }

    /// Remove index entries for all indexed columns of a row being deleted.
    fn deleteIndexEntries(self: *Database, table_info: *const catalog_mod.TableInfo, vals: []const Value) void {
        for (table_info.indexes) |idx| {
            if (idx.column_index >= vals.len) continue;
            const idx_key = valueToIndexKey(self.allocator, vals[idx.column_index]) catch continue;
            defer self.allocator.free(idx_key);

            var idx_tree = BTree.init(self.pool, idx.root_page_id);
            idx_tree.delete(idx_key) catch {};
        }
    }

    // ── UPDATE execution ──────────────────────────────────────────────

    fn executeUpdate(self: *Database, arena: *AstArena, plan: LogicalPlan) EngineError!QueryResult {
        defer {
            arena.deinit();
            self.allocator.destroy(arena);
        }

        // Plan structure for UPDATE: Project(Filter(Scan)) or Project(Scan)
        // The Project node has columns with alias = column name to update
        // We need to: scan rows, evaluate WHERE, apply assignments, write back

        // Extract the update info from the plan tree
        var scan_table: []const u8 = "";
        var predicate: ?*const ast_mod.Expr = null;
        var assignments: []const PlanNode.ProjectColumn = &.{};

        // Walk the plan tree to extract components
        var current = plan.root;
        while (true) {
            switch (current.*) {
                .project => |p| {
                    assignments = p.columns;
                    current = p.input;
                },
                .filter => |f| {
                    predicate = f.predicate;
                    current = f.input;
                },
                .scan => |s| {
                    scan_table = s.table;
                    break;
                },
                else => return EngineError.ExecutionError,
            }
        }

        // Look up table
        var table_info = self.catalog.getTable(scan_table) catch return EngineError.TableNotFound;
        defer table_info.deinit(self.allocator);

        var tree = BTree.init(self.pool, table_info.data_root_page_id);

        // Build column names
        const col_names = self.allocator.alloc([]const u8, table_info.columns.len) catch return EngineError.OutOfMemory;
        defer self.allocator.free(col_names);
        for (table_info.columns, 0..) |col, i| {
            col_names[i] = col.name;
        }

        // Scan all rows, collect those matching predicate, update them
        var cursor = btree_mod.Cursor.init(self.allocator, &tree);
        defer cursor.deinit();
        cursor.seekFirst() catch return EngineError.StorageError;

        const UpdateEntry = struct { key: []u8, value: []u8, old_values: []Value };

        // Collect key-value pairs to update (can't modify B+Tree while iterating)
        var updates = std.ArrayListUnmanaged(UpdateEntry){};
        defer {
            for (updates.items) |item| {
                self.allocator.free(item.key);
                self.allocator.free(item.value);
                for (item.old_values) |v| v.free(self.allocator);
                self.allocator.free(item.old_values);
            }
            updates.deinit(self.allocator);
        }

        // Get MVCC context for visibility filtering (owned snapshot for RC)
        var mvcc_ctx = try self.getMvccContext();
        defer if (mvcc_ctx) |*ctx| {
            if (self.current_txn != null and self.current_txn.?.isolation == .read_committed) {
                ctx.snapshot.deinit();
            }
        };

        while (cursor.next() catch null) |entry| {
            defer self.allocator.free(entry.key);

            // Deserialize row (handle MVCC header if present)
            var values: []Value = undefined;
            if (isVersionedRowData(entry.value)) {
                // MVCC row: check visibility first
                if (mvcc_ctx) |ctx| {
                    const hdr = TupleHeader.deserialize(entry.value[1..][0..mvcc_mod.TUPLE_HEADER_SIZE]);
                    if (!mvcc_mod.isTupleVisible(hdr, ctx.snapshot, ctx.current_xid, ctx.current_cid)) {
                        self.allocator.free(entry.value);
                        continue;
                    }
                }
                values = executor_mod.deserializeRow(self.allocator, entry.value[mvcc_mod.MVCC_ROW_OVERHEAD..]) catch {
                    self.allocator.free(entry.value);
                    continue;
                };
            } else {
                values = executor_mod.deserializeRow(self.allocator, entry.value) catch {
                    self.allocator.free(entry.value);
                    continue;
                };
            }
            self.allocator.free(entry.value);

            var row = Row{
                .columns = col_names,
                .values = values,
                .allocator = self.allocator,
            };

            // Check predicate
            if (predicate) |pred| {
                const val = evalExpr(self.allocator, pred, &row) catch {
                    for (row.values) |v| v.free(self.allocator);
                    self.allocator.free(row.values);
                    continue;
                };
                defer val.free(self.allocator);
                if (!val.isTruthy()) {
                    for (row.values) |v| v.free(self.allocator);
                    self.allocator.free(row.values);
                    continue;
                }
            }

            // Save old values for index maintenance
            const old_values = self.allocator.alloc(Value, row.values.len) catch {
                for (row.values) |v| v.free(self.allocator);
                self.allocator.free(row.values);
                return EngineError.OutOfMemory;
            };
            for (row.values, 0..) |v, vi| {
                old_values[vi] = v.dupe(self.allocator) catch {
                    for (old_values[0..vi]) |ov| ov.free(self.allocator);
                    self.allocator.free(old_values);
                    for (row.values) |rv| rv.free(self.allocator);
                    self.allocator.free(row.values);
                    return EngineError.OutOfMemory;
                };
            }

            // Apply assignments
            for (assignments) |assign| {
                const new_val = evalExpr(self.allocator, assign.expr, &row) catch continue;
                // Find column index by alias (= column name)
                const col_name = assign.alias orelse {
                    new_val.free(self.allocator);
                    continue;
                };
                var found = false;
                for (col_names, 0..) |cn, ci| {
                    if (std.ascii.eqlIgnoreCase(cn, col_name)) {
                        row.values[ci].free(self.allocator);
                        row.values[ci] = new_val;
                        found = true;
                        break;
                    }
                }
                if (!found) new_val.free(self.allocator);
            }

            // Re-serialize (with MVCC header if in a transaction)
            const new_data = if (self.current_txn) |txn| blk: {
                const cid = self.tm.getCurrentCid(txn.xid) catch {
                    for (old_values) |ov| ov.free(self.allocator);
                    self.allocator.free(old_values);
                    for (row.values) |v| v.free(self.allocator);
                    self.allocator.free(row.values);
                    return EngineError.TransactionError;
                };
                const header = TupleHeader.forInsert(txn.xid, cid);
                break :blk mvcc_mod.serializeVersionedRow(self.allocator, header, row.values) catch {
                    for (old_values) |ov| ov.free(self.allocator);
                    self.allocator.free(old_values);
                    for (row.values) |v| v.free(self.allocator);
                    self.allocator.free(row.values);
                    return EngineError.OutOfMemory;
                };
            } else serializeRow(self.allocator, row.values) catch {
                for (old_values) |ov| ov.free(self.allocator);
                self.allocator.free(old_values);
                for (row.values) |v| v.free(self.allocator);
                self.allocator.free(row.values);
                return EngineError.OutOfMemory;
            };

            const key_copy = self.allocator.dupe(u8, entry.key) catch {
                self.allocator.free(new_data);
                for (old_values) |ov| ov.free(self.allocator);
                self.allocator.free(old_values);
                for (row.values) |v| v.free(self.allocator);
                self.allocator.free(row.values);
                return EngineError.OutOfMemory;
            };

            updates.append(self.allocator, .{ .key = key_copy, .value = new_data, .old_values = old_values }) catch {
                self.allocator.free(key_copy);
                self.allocator.free(new_data);
                for (old_values) |ov| ov.free(self.allocator);
                self.allocator.free(old_values);
                for (row.values) |v| v.free(self.allocator);
                self.allocator.free(row.values);
                return EngineError.OutOfMemory;
            };

            for (row.values) |v| v.free(self.allocator);
            self.allocator.free(row.values);
        }

        // Apply updates (delete + re-insert) and maintain indexes
        const rows_updated = updates.items.len;
        for (updates.items) |item| {
            // Remove old index entries
            self.deleteIndexEntries(&table_info, item.old_values);

            tree.delete(item.key) catch {};
            tree.insert(item.key, item.value) catch {};

            // Add new index entries: deserialize new row values for index
            const idx_data = if (isVersionedRowData(item.value))
                item.value[mvcc_mod.MVCC_ROW_OVERHEAD..]
            else
                item.value;
            const new_values = executor_mod.deserializeRow(self.allocator, idx_data) catch continue;
            defer {
                for (new_values) |v| v.free(self.allocator);
                self.allocator.free(new_values);
            }
            self.insertIndexEntries(scan_table, &table_info, new_values, item.key) catch {};
        }

        // Update root page if needed
        if (tree.root_page_id != table_info.data_root_page_id) {
            self.updateTableRootPage(scan_table, tree.root_page_id) catch {};
        }

        return .{
            .rows_affected = @intCast(rows_updated),
            .message = "UPDATE",
        };
    }

    // ── DELETE execution ──────────────────────────────────────────────

    fn executeDelete(self: *Database, arena: *AstArena, plan: LogicalPlan) EngineError!QueryResult {
        defer {
            arena.deinit();
            self.allocator.destroy(arena);
        }

        // Plan structure for DELETE: Filter(Scan) or Scan
        var scan_table: []const u8 = "";
        var predicate: ?*const ast_mod.Expr = null;

        var current = plan.root;
        while (true) {
            switch (current.*) {
                .filter => |f| {
                    predicate = f.predicate;
                    current = f.input;
                },
                .scan => |s| {
                    scan_table = s.table;
                    break;
                },
                else => return EngineError.ExecutionError,
            }
        }

        var table_info = self.catalog.getTable(scan_table) catch return EngineError.TableNotFound;
        defer table_info.deinit(self.allocator);

        var tree = BTree.init(self.pool, table_info.data_root_page_id);

        const col_names = self.allocator.alloc([]const u8, table_info.columns.len) catch return EngineError.OutOfMemory;
        defer self.allocator.free(col_names);
        for (table_info.columns, 0..) |col, i| {
            col_names[i] = col.name;
        }

        var cursor = btree_mod.Cursor.init(self.allocator, &tree);
        defer cursor.deinit();
        cursor.seekFirst() catch return EngineError.StorageError;

        const DeleteEntry = struct { key: []u8, values: []Value };

        // Collect keys to delete (with row values for index maintenance)
        var deletes = std.ArrayListUnmanaged(DeleteEntry){};
        defer {
            for (deletes.items) |d| {
                self.allocator.free(d.key);
                for (d.values) |v| v.free(self.allocator);
                self.allocator.free(d.values);
            }
            deletes.deinit(self.allocator);
        }

        // Get MVCC context for visibility filtering (owned snapshot for RC)
        var mvcc_ctx_del = try self.getMvccContext();
        defer if (mvcc_ctx_del) |*ctx| {
            if (self.current_txn != null and self.current_txn.?.isolation == .read_committed) {
                ctx.snapshot.deinit();
            }
        };

        while (cursor.next() catch null) |entry| {
            // Deserialize row for predicate check and index maintenance
            var values: []Value = undefined;
            if (isVersionedRowData(entry.value)) {
                // MVCC row: check visibility first
                if (mvcc_ctx_del) |ctx| {
                    const hdr = TupleHeader.deserialize(entry.value[1..][0..mvcc_mod.TUPLE_HEADER_SIZE]);
                    if (!mvcc_mod.isTupleVisible(hdr, ctx.snapshot, ctx.current_xid, ctx.current_cid)) {
                        self.allocator.free(entry.value);
                        self.allocator.free(entry.key);
                        continue;
                    }
                }
                values = executor_mod.deserializeRow(self.allocator, entry.value[mvcc_mod.MVCC_ROW_OVERHEAD..]) catch {
                    self.allocator.free(entry.value);
                    self.allocator.free(entry.key);
                    continue;
                };
            } else {
                values = executor_mod.deserializeRow(self.allocator, entry.value) catch {
                    self.allocator.free(entry.value);
                    self.allocator.free(entry.key);
                    continue;
                };
            }
            self.allocator.free(entry.value);

            if (predicate) |pred| {
                var row = Row{
                    .columns = col_names,
                    .values = values,
                    .allocator = self.allocator,
                };

                const val = evalExpr(self.allocator, pred, &row) catch {
                    for (values) |v| v.free(self.allocator);
                    self.allocator.free(values);
                    self.allocator.free(entry.key);
                    continue;
                };
                defer val.free(self.allocator);

                if (!val.isTruthy()) {
                    for (values) |v| v.free(self.allocator);
                    self.allocator.free(values);
                    self.allocator.free(entry.key);
                    continue;
                }
            }

            // Mark for deletion (keep values for index maintenance)
            deletes.append(self.allocator, .{ .key = entry.key, .values = values }) catch {
                for (values) |v| v.free(self.allocator);
                self.allocator.free(values);
                self.allocator.free(entry.key);
                continue;
            };
        }

        const rows_deleted = deletes.items.len;
        for (deletes.items) |d| {
            // Remove index entries before deleting the row
            self.deleteIndexEntries(&table_info, d.values);
            tree.delete(d.key) catch {};
        }

        if (tree.root_page_id != table_info.data_root_page_id) {
            self.updateTableRootPage(scan_table, tree.root_page_id) catch {};
        }

        return .{
            .rows_affected = @intCast(rows_deleted),
            .message = "DELETE",
        };
    }

    // ── DDL execution ────────────────────────────────────────────────

    fn executeCreateTable(self: *Database, arena: *AstArena, plan: LogicalPlan) EngineError!QueryResult {
        defer {
            arena.deinit();
            self.allocator.destroy(arena);
        }

        // The original AST is in the plan as an Empty node with description.
        // We need to re-parse to get the CreateTableStmt, or we can store it differently.
        // For now, we'll use the description to identify it and use the analyzer's validated info.
        _ = plan;

        // This path should not normally be reached — DDL is handled before planning.
        return .{ .message = "OK" };
    }

    fn executeDropTable(self: *Database, arena: *AstArena, plan: LogicalPlan) EngineError!QueryResult {
        defer {
            arena.deinit();
            self.allocator.destroy(arena);
        }
        _ = plan;
        return .{ .message = "OK" };
    }

    /// Execute a full SQL statement including DDL. This is the primary entry point.
    pub fn execSQL(self: *Database, sql: []const u8) EngineError!QueryResult {
        // Reset schema arena from previous exec call
        _ = self.schema_arena.reset(.retain_capacity);

        // Parse first to determine statement type
        var arena = self.allocator.create(AstArena) catch return EngineError.OutOfMemory;
        arena.* = AstArena.init(self.allocator);
        errdefer {
            arena.deinit();
            self.allocator.destroy(arena);
        }

        var infra_alloc = std.heap.ArenaAllocator.init(self.allocator);
        defer infra_alloc.deinit();

        var p = Parser.init(infra_alloc.allocator(), sql, arena) catch return EngineError.ParseError;
        defer p.deinit();

        const maybe_stmt = p.parseStatement() catch return EngineError.ParseError;
        const stmt = maybe_stmt orelse return EngineError.ParseError;

        // Handle DDL directly (before analysis/planning)
        switch (stmt) {
            .create_table => |ct| {
                self.catalog.createTableFromAst(&ct) catch |err| {
                    return switch (err) {
                        error.TableAlreadyExists => EngineError.TableAlreadyExists,
                        error.OutOfMemory => EngineError.OutOfMemory,
                        else => EngineError.StorageError,
                    };
                };
                arena.deinit();
                self.allocator.destroy(arena);
                self.commitWal() catch {};
                return .{ .message = "CREATE TABLE" };
            },
            .drop_table => |dt| {
                self.catalog.dropTable(dt.name, dt.if_exists) catch |err| {
                    return switch (err) {
                        error.TableNotFound => EngineError.TableNotFound,
                        error.OutOfMemory => EngineError.OutOfMemory,
                        else => EngineError.StorageError,
                    };
                };
                arena.deinit();
                self.allocator.destroy(arena);
                self.commitWal() catch {};
                return .{ .message = "DROP TABLE" };
            },
            .transaction => |txn| {
                arena.deinit();
                self.allocator.destroy(arena);
                switch (txn) {
                    .begin => {
                        self.beginTransaction(.read_committed) catch {
                            return .{ .message = "ERROR: already in a transaction" };
                        };
                        return .{ .message = "BEGIN" };
                    },
                    .commit => {
                        self.commitTransaction() catch {
                            return .{ .message = "WARNING: there is no transaction in progress" };
                        };
                        return .{ .message = "COMMIT" };
                    },
                    .rollback => {
                        self.rollbackTransaction() catch {
                            return .{ .message = "WARNING: there is no transaction in progress" };
                        };
                        return .{ .message = "ROLLBACK" };
                    },
                    .savepoint, .release => return .{ .message = "OK" },
                }
            },
            else => {},
        }

        // DML: analyze → plan → optimize → execute
        const provider = self.schemaProvider();

        var an = Analyzer.init(self.allocator, provider);
        defer an.deinit();
        an.analyze(stmt);
        if (an.hasErrors()) return EngineError.AnalysisError;

        var plnr = Planner.init(arena, provider);
        const plan = plnr.plan(stmt) catch return EngineError.PlanError;

        var opt = Optimizer.init(arena);
        const optimized = opt.optimize(plan) catch return EngineError.PlanError;

        return self.executePlan(arena, optimized);
    }

    // ── SchemaProvider adapter ────────────────────────────────────────

    fn schemaProvider(self: *Database) SchemaProvider {
        return .{
            .ptr = @ptrCast(self),
            .vtable = &.{
                .getTable = @ptrCast(&catalogGetTable),
                .tableExists = @ptrCast(&catalogTableExists),
            },
        };
    }

    fn catalogGetTable(self: *Database, _: Allocator, name: []const u8) ?TableInfo {
        // Deserialize using the schema arena so memory is managed by the
        // Database and freed between exec calls. The analyzer doesn't need
        // to manage TableInfo lifetimes.
        const value = self.catalog.tree.get(self.allocator, name) catch return null;
        if (value == null) return null;
        defer self.allocator.free(value.?);
        return catalog_mod.deserializeTable(self.schema_arena.allocator(), name, value.?) catch null;
    }

    fn catalogTableExists(self: *Database, name: []const u8) bool {
        return self.catalog.tableExists(name) catch false;
    }
};

// ── Tests ───────────────────────────────────────────────────────────────

const testing = std.testing;

fn createTestDb(allocator: Allocator, path: []const u8) !Database {
    std.fs.cwd().deleteFile(path) catch {};
    return Database.open(allocator, path, .{});
}

fn cleanupTestDb(db: *Database, path: []const u8) void {
    db.close();
    std.fs.cwd().deleteFile(path) catch {};
}

test "Database open and close" {
    const path = "test_engine_open.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var db = try Database.open(testing.allocator, path, .{});
    db.close();

    // Re-open should work
    var db2 = try Database.open(testing.allocator, path, .{});
    db2.close();
}

test "CREATE TABLE via execSQL" {
    const path = "test_eng_create.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var result = try db.execSQL("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT)");
    defer result.close(testing.allocator);

    try testing.expectEqualStrings("CREATE TABLE", result.message);

    // Verify table exists
    try testing.expect(try db.catalog.tableExists("users"));
}

test "CREATE TABLE IF NOT EXISTS" {
    const path = "test_eng_create_ine.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r1 = try db.execSQL("CREATE TABLE t1 (id INTEGER)");
    defer r1.close(testing.allocator);

    // Should not error
    var r2 = try db.execSQL("CREATE TABLE IF NOT EXISTS t1 (id INTEGER)");
    defer r2.close(testing.allocator);
}

test "DROP TABLE via execSQL" {
    const path = "test_eng_drop.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r1 = try db.execSQL("CREATE TABLE t1 (id INTEGER)");
    defer r1.close(testing.allocator);

    var r2 = try db.execSQL("DROP TABLE t1");
    defer r2.close(testing.allocator);
    try testing.expectEqualStrings("DROP TABLE", r2.message);

    try testing.expect(!try db.catalog.tableExists("t1"));
}

test "INSERT and SELECT round-trip" {
    const path = "test_eng_insert_sel.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    // Create table
    var r1 = try db.execSQL("CREATE TABLE users (id INTEGER, name TEXT)");
    defer r1.close(testing.allocator);

    // Insert rows
    var r2 = try db.execSQL("INSERT INTO users (id, name) VALUES (1, 'Alice')");
    defer r2.close(testing.allocator);
    try testing.expectEqual(@as(u64, 1), r2.rows_affected);

    var r3 = try db.execSQL("INSERT INTO users (id, name) VALUES (2, 'Bob')");
    defer r3.close(testing.allocator);
    try testing.expectEqual(@as(u64, 1), r3.rows_affected);

    // Select all rows
    var r4 = try db.execSQL("SELECT id, name FROM users");
    defer r4.close(testing.allocator);

    var count: usize = 0;
    while (try r4.rows.?.next()) |*row_ptr| {
        var row = row_ptr.*;
        defer row.deinit();
        count += 1;
    }
    try testing.expectEqual(@as(usize, 2), count);
}

test "INSERT multiple rows" {
    const path = "test_eng_insert_multi.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r1 = try db.execSQL("CREATE TABLE items (id INTEGER, val TEXT)");
    defer r1.close(testing.allocator);

    var r2 = try db.execSQL("INSERT INTO items (id, val) VALUES (1, 'a'), (2, 'b'), (3, 'c')");
    defer r2.close(testing.allocator);
    try testing.expectEqual(@as(u64, 3), r2.rows_affected);

    // Count rows
    var r3 = try db.execSQL("SELECT id FROM items");
    defer r3.close(testing.allocator);

    var count: usize = 0;
    while (try r3.rows.?.next()) |*row_ptr| {
        var row = row_ptr.*;
        defer row.deinit();
        count += 1;
    }
    try testing.expectEqual(@as(usize, 3), count);
}

test "SELECT with WHERE clause" {
    const path = "test_eng_sel_where.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r1 = try db.execSQL("CREATE TABLE nums (id INTEGER, val INTEGER)");
    defer r1.close(testing.allocator);

    var r2 = try db.execSQL("INSERT INTO nums (id, val) VALUES (1, 10), (2, 20), (3, 30)");
    defer r2.close(testing.allocator);

    var r3 = try db.execSQL("SELECT id, val FROM nums WHERE val > 15");
    defer r3.close(testing.allocator);

    var count: usize = 0;
    while (try r3.rows.?.next()) |*row_ptr| {
        var row = row_ptr.*;
        defer row.deinit();
        count += 1;
        // Values should be > 15
        const val = row.getColumn("val").?;
        try testing.expect(val.integer > 15);
    }
    try testing.expectEqual(@as(usize, 2), count);
}

test "SELECT with ORDER BY" {
    const path = "test_eng_sel_order.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r1 = try db.execSQL("CREATE TABLE nums (id INTEGER, val INTEGER)");
    defer r1.close(testing.allocator);

    var r2 = try db.execSQL("INSERT INTO nums (id, val) VALUES (1, 30), (2, 10), (3, 20)");
    defer r2.close(testing.allocator);

    var r3 = try db.execSQL("SELECT id, val FROM nums ORDER BY val ASC");
    defer r3.close(testing.allocator);

    var prev_val: i64 = -1;
    while (try r3.rows.?.next()) |*row_ptr| {
        var row = row_ptr.*;
        defer row.deinit();
        const val = row.getColumn("val").?.integer;
        try testing.expect(val >= prev_val);
        prev_val = val;
    }
}

test "SELECT with LIMIT" {
    const path = "test_eng_sel_limit.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r1 = try db.execSQL("CREATE TABLE nums (id INTEGER)");
    defer r1.close(testing.allocator);

    var r2 = try db.execSQL("INSERT INTO nums (id) VALUES (1), (2), (3), (4), (5)");
    defer r2.close(testing.allocator);

    var r3 = try db.execSQL("SELECT id FROM nums LIMIT 3");
    defer r3.close(testing.allocator);

    var count: usize = 0;
    while (try r3.rows.?.next()) |*row_ptr| {
        var row = row_ptr.*;
        defer row.deinit();
        count += 1;
    }
    try testing.expectEqual(@as(usize, 3), count);
}

test "SELECT COUNT(*) aggregate" {
    const path = "test_eng_count.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r1 = try db.execSQL("CREATE TABLE items (id INTEGER, name TEXT)");
    defer r1.close(testing.allocator);

    var r2 = try db.execSQL("INSERT INTO items (id, name) VALUES (1, 'a')");
    defer r2.close(testing.allocator);

    var r2b = try db.execSQL("INSERT INTO items (id, name) VALUES (2, 'b')");
    defer r2b.close(testing.allocator);

    var r2c = try db.execSQL("INSERT INTO items (id, name) VALUES (3, 'c')");
    defer r2c.close(testing.allocator);

    var r3 = try db.execSQL("SELECT COUNT(*) FROM items");
    defer r3.close(testing.allocator);

    try testing.expect(r3.rows != null);
    const row = (try r3.rows.?.next()) orelse return error.ExpectedRow;
    var row_mut = row;
    defer row_mut.deinit();

    try testing.expectEqual(@as(usize, 1), row.values.len);
    try testing.expect(row.values[0] == .integer);
    try testing.expectEqual(@as(i64, 3), row.values[0].integer);
}

test "DELETE with WHERE" {
    const path = "test_eng_delete.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r1 = try db.execSQL("CREATE TABLE items (id INTEGER, val INTEGER)");
    defer r1.close(testing.allocator);

    var r2 = try db.execSQL("INSERT INTO items (id, val) VALUES (1, 10), (2, 20), (3, 30)");
    defer r2.close(testing.allocator);

    var r3 = try db.execSQL("DELETE FROM items WHERE val = 20");
    defer r3.close(testing.allocator);
    try testing.expectEqual(@as(u64, 1), r3.rows_affected);

    // Verify only 2 rows remain
    var r4 = try db.execSQL("SELECT id FROM items");
    defer r4.close(testing.allocator);

    var count: usize = 0;
    while (try r4.rows.?.next()) |*row_ptr| {
        var row = row_ptr.*;
        defer row.deinit();
        count += 1;
    }
    try testing.expectEqual(@as(usize, 2), count);
}

test "UPDATE with WHERE" {
    const path = "test_eng_update.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r1 = try db.execSQL("CREATE TABLE items (id INTEGER, val INTEGER)");
    defer r1.close(testing.allocator);

    var r2 = try db.execSQL("INSERT INTO items (id, val) VALUES (1, 10), (2, 20), (3, 30)");
    defer r2.close(testing.allocator);

    var r3 = try db.execSQL("UPDATE items SET val = 99 WHERE id = 2");
    defer r3.close(testing.allocator);
    try testing.expectEqual(@as(u64, 1), r3.rows_affected);

    // Verify the update
    var r4 = try db.execSQL("SELECT id, val FROM items WHERE id = 2");
    defer r4.close(testing.allocator);

    if (try r4.rows.?.next()) |*row_ptr| {
        var row = row_ptr.*;
        defer row.deinit();
        try testing.expectEqual(@as(i64, 99), row.getColumn("val").?.integer);
    } else {
        return error.TestExpectedRow;
    }
}

test "SELECT * (all columns)" {
    const path = "test_eng_sel_star.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r1 = try db.execSQL("CREATE TABLE t (a INTEGER, b TEXT, c REAL)");
    defer r1.close(testing.allocator);

    var r2 = try db.execSQL("INSERT INTO t (a, b, c) VALUES (1, 'hello', 3.14)");
    defer r2.close(testing.allocator);

    var r3 = try db.execSQL("SELECT * FROM t");
    defer r3.close(testing.allocator);

    if (try r3.rows.?.next()) |*row_ptr| {
        var row = row_ptr.*;
        defer row.deinit();
        try testing.expectEqual(@as(usize, 3), row.columns.len);
    } else {
        return error.TestExpectedRow;
    }
}

test "table not found error" {
    const path = "test_eng_notfound.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    const result = db.execSQL("SELECT * FROM nonexistent");
    try testing.expectError(EngineError.AnalysisError, result);
}

test "duplicate CREATE TABLE error" {
    const path = "test_eng_dup_create.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r1 = try db.execSQL("CREATE TABLE t1 (id INTEGER)");
    defer r1.close(testing.allocator);

    const result = db.execSQL("CREATE TABLE t1 (id INTEGER)");
    try testing.expectError(EngineError.TableAlreadyExists, result);
}

test "DROP TABLE IF EXISTS on nonexistent" {
    const path = "test_eng_drop_ine.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r1 = try db.execSQL("DROP TABLE IF EXISTS nonexistent");
    defer r1.close(testing.allocator);
    try testing.expectEqualStrings("DROP TABLE", r1.message);
}

test "empty SELECT result" {
    const path = "test_eng_empty_sel.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r1 = try db.execSQL("CREATE TABLE t (id INTEGER)");
    defer r1.close(testing.allocator);

    // No rows inserted — SELECT should return 0 rows
    var r2 = try db.execSQL("SELECT id FROM t");
    defer r2.close(testing.allocator);

    const row = try r2.rows.?.next();
    try testing.expect(row == null);
}

test "SELECT with expression" {
    const path = "test_eng_sel_expr.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r1 = try db.execSQL("CREATE TABLE nums (a INTEGER, b INTEGER)");
    defer r1.close(testing.allocator);

    var r2 = try db.execSQL("INSERT INTO nums (a, b) VALUES (10, 20)");
    defer r2.close(testing.allocator);

    var r3 = try db.execSQL("SELECT a + b FROM nums");
    defer r3.close(testing.allocator);

    if (try r3.rows.?.next()) |*row_ptr| {
        var row = row_ptr.*;
        defer row.deinit();
        try testing.expectEqual(@as(i64, 30), row.values[0].integer);
    } else {
        return error.TestExpectedRow;
    }
}

test "SELECT with LIMIT and OFFSET" {
    const path = "test_eng_limit_off.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r1 = try db.execSQL("CREATE TABLE t (id INTEGER)");
    defer r1.close(testing.allocator);

    // Insert 5 rows individually
    var ins1 = try db.execSQL("INSERT INTO t (id) VALUES (1)");
    defer ins1.close(testing.allocator);
    var ins2 = try db.execSQL("INSERT INTO t (id) VALUES (2)");
    defer ins2.close(testing.allocator);
    var ins3 = try db.execSQL("INSERT INTO t (id) VALUES (3)");
    defer ins3.close(testing.allocator);
    var ins4 = try db.execSQL("INSERT INTO t (id) VALUES (4)");
    defer ins4.close(testing.allocator);
    var ins5 = try db.execSQL("INSERT INTO t (id) VALUES (5)");
    defer ins5.close(testing.allocator);

    // LIMIT 2 OFFSET 2 should skip first 2, return next 2
    var r3 = try db.execSQL("SELECT id FROM t LIMIT 2 OFFSET 2");
    defer r3.close(testing.allocator);

    var count: usize = 0;
    while (try r3.rows.?.next()) |*row_ptr| {
        var row = row_ptr.*;
        defer row.deinit();
        count += 1;
    }
    try testing.expectEqual(@as(usize, 2), count);
}

test "DELETE all rows (no WHERE)" {
    const path = "test_eng_del_all.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r1 = try db.execSQL("CREATE TABLE t (id INTEGER)");
    defer r1.close(testing.allocator);

    var r2 = try db.execSQL("INSERT INTO t (id) VALUES (1)");
    defer r2.close(testing.allocator);
    var r2b = try db.execSQL("INSERT INTO t (id) VALUES (2)");
    defer r2b.close(testing.allocator);
    var r2c = try db.execSQL("INSERT INTO t (id) VALUES (3)");
    defer r2c.close(testing.allocator);

    // DELETE without WHERE should delete all
    var r3 = try db.execSQL("DELETE FROM t");
    defer r3.close(testing.allocator);
    try testing.expectEqual(@as(u64, 3), r3.rows_affected);

    // Verify empty
    var r4 = try db.execSQL("SELECT id FROM t");
    defer r4.close(testing.allocator);
    const row = try r4.rows.?.next();
    try testing.expect(row == null);
}

test "UPDATE all rows (no WHERE)" {
    const path = "test_eng_upd_all.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r1 = try db.execSQL("CREATE TABLE t (id INTEGER, val INTEGER)");
    defer r1.close(testing.allocator);

    var ins1 = try db.execSQL("INSERT INTO t (id, val) VALUES (1, 10)");
    defer ins1.close(testing.allocator);
    var ins2 = try db.execSQL("INSERT INTO t (id, val) VALUES (2, 20)");
    defer ins2.close(testing.allocator);

    // UPDATE without WHERE should update all rows
    var r3 = try db.execSQL("UPDATE t SET val = 0");
    defer r3.close(testing.allocator);
    try testing.expectEqual(@as(u64, 2), r3.rows_affected);

    // Verify all values are 0
    var r4 = try db.execSQL("SELECT id, val FROM t");
    defer r4.close(testing.allocator);

    var count: usize = 0;
    while (try r4.rows.?.next()) |*row_ptr| {
        var row = row_ptr.*;
        defer row.deinit();
        try testing.expectEqual(@as(i64, 0), row.getColumn("val").?.integer);
        count += 1;
    }
    try testing.expectEqual(@as(usize, 2), count);
}

test "SUM aggregate" {
    const path = "test_eng_sum.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r1 = try db.execSQL("CREATE TABLE t (val INTEGER)");
    defer r1.close(testing.allocator);

    var ins1 = try db.execSQL("INSERT INTO t (val) VALUES (10)");
    defer ins1.close(testing.allocator);
    var ins2 = try db.execSQL("INSERT INTO t (val) VALUES (20)");
    defer ins2.close(testing.allocator);
    var ins3 = try db.execSQL("INSERT INTO t (val) VALUES (30)");
    defer ins3.close(testing.allocator);

    var r3 = try db.execSQL("SELECT SUM(val) FROM t");
    defer r3.close(testing.allocator);

    try testing.expect(r3.rows != null);
    const row = (try r3.rows.?.next()) orelse return error.ExpectedRow;
    var row_mut = row;
    defer row_mut.deinit();
    try testing.expect(row.values[0] == .integer);
    try testing.expectEqual(@as(i64, 60), row.values[0].integer);
}

test "MIN and MAX aggregates" {
    const path = "test_eng_minmax.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r1 = try db.execSQL("CREATE TABLE t (val INTEGER)");
    defer r1.close(testing.allocator);

    var ins1 = try db.execSQL("INSERT INTO t (val) VALUES (5)");
    defer ins1.close(testing.allocator);
    var ins2 = try db.execSQL("INSERT INTO t (val) VALUES (15)");
    defer ins2.close(testing.allocator);
    var ins3 = try db.execSQL("INSERT INTO t (val) VALUES (10)");
    defer ins3.close(testing.allocator);

    var r_min = try db.execSQL("SELECT MIN(val) FROM t");
    defer r_min.close(testing.allocator);
    const min_row = (try r_min.rows.?.next()) orelse return error.ExpectedRow;
    var min_mut = min_row;
    defer min_mut.deinit();
    try testing.expectEqual(@as(i64, 5), min_row.values[0].integer);

    var r_max = try db.execSQL("SELECT MAX(val) FROM t");
    defer r_max.close(testing.allocator);
    const max_row = (try r_max.rows.?.next()) orelse return error.ExpectedRow;
    var max_mut = max_row;
    defer max_mut.deinit();
    try testing.expectEqual(@as(i64, 15), max_row.values[0].integer);
}

test "SELECT with compound WHERE (AND)" {
    const path = "test_eng_and.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r1 = try db.execSQL("CREATE TABLE t (id INTEGER, status TEXT, val INTEGER)");
    defer r1.close(testing.allocator);

    var ins1 = try db.execSQL("INSERT INTO t (id, status, val) VALUES (1, 'active', 10)");
    defer ins1.close(testing.allocator);
    var ins2 = try db.execSQL("INSERT INTO t (id, status, val) VALUES (2, 'active', 20)");
    defer ins2.close(testing.allocator);
    var ins3 = try db.execSQL("INSERT INTO t (id, status, val) VALUES (3, 'inactive', 30)");
    defer ins3.close(testing.allocator);

    var r3 = try db.execSQL("SELECT id FROM t WHERE status = 'active' AND val > 15");
    defer r3.close(testing.allocator);

    var count: usize = 0;
    while (try r3.rows.?.next()) |*row_ptr| {
        var row = row_ptr.*;
        defer row.deinit();
        count += 1;
        try testing.expectEqual(@as(i64, 2), row.values[0].integer);
    }
    try testing.expectEqual(@as(usize, 1), count);
}

test "SELECT with OR in WHERE" {
    const path = "test_eng_or.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r1 = try db.execSQL("CREATE TABLE t (id INTEGER, val INTEGER)");
    defer r1.close(testing.allocator);

    var ins1 = try db.execSQL("INSERT INTO t (id, val) VALUES (1, 10)");
    defer ins1.close(testing.allocator);
    var ins2 = try db.execSQL("INSERT INTO t (id, val) VALUES (2, 20)");
    defer ins2.close(testing.allocator);
    var ins3 = try db.execSQL("INSERT INTO t (id, val) VALUES (3, 30)");
    defer ins3.close(testing.allocator);

    var r3 = try db.execSQL("SELECT id FROM t WHERE val = 10 OR val = 30");
    defer r3.close(testing.allocator);

    var count: usize = 0;
    while (try r3.rows.?.next()) |*row_ptr| {
        var row = row_ptr.*;
        defer row.deinit();
        count += 1;
    }
    try testing.expectEqual(@as(usize, 2), count);
}

test "NULL value insertion and retrieval" {
    const path = "test_eng_null.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r1 = try db.execSQL("CREATE TABLE t (id INTEGER, name TEXT)");
    defer r1.close(testing.allocator);

    var r2 = try db.execSQL("INSERT INTO t (id, name) VALUES (1, NULL)");
    defer r2.close(testing.allocator);

    var r3 = try db.execSQL("SELECT id, name FROM t");
    defer r3.close(testing.allocator);

    if (try r3.rows.?.next()) |*row_ptr| {
        var row = row_ptr.*;
        defer row.deinit();
        try testing.expectEqual(@as(i64, 1), row.values[0].integer);
        try testing.expect(row.values[1] == .null_value);
    } else {
        return error.TestExpectedRow;
    }
}

test "data persistence across close and reopen" {
    const path = "test_eng_persist.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    // Write data and close
    {
        var db = try createTestDb(testing.allocator, path);
        var r1 = try db.execSQL("CREATE TABLE t (id INTEGER, name TEXT)");
        r1.close(testing.allocator);
        var r2 = try db.execSQL("INSERT INTO t (id, name) VALUES (1, 'hello')");
        r2.close(testing.allocator);
        var r3 = try db.execSQL("INSERT INTO t (id, name) VALUES (2, 'world')");
        r3.close(testing.allocator);
        db.close();
    }

    // Reopen and verify
    {
        var db = try Database.open(testing.allocator, path, .{});
        defer cleanupTestDb(&db, path);

        // Table should still exist
        try testing.expect(try db.catalog.tableExists("t"));

        // Rows should still be there
        var r4 = try db.execSQL("SELECT id, name FROM t");
        defer r4.close(testing.allocator);

        var count: usize = 0;
        while (try r4.rows.?.next()) |*row_ptr| {
            var row = row_ptr.*;
            defer row.deinit();
            count += 1;
        }
        try testing.expectEqual(@as(usize, 2), count);
    }
}

test "multiple tables in same database" {
    const path = "test_eng_multi_tbl.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r1 = try db.execSQL("CREATE TABLE users (id INTEGER, name TEXT)");
    defer r1.close(testing.allocator);
    var r2 = try db.execSQL("CREATE TABLE orders (id INTEGER, user_id INTEGER, amount INTEGER)");
    defer r2.close(testing.allocator);

    var ins1 = try db.execSQL("INSERT INTO users (id, name) VALUES (1, 'Alice')");
    defer ins1.close(testing.allocator);
    var ins2 = try db.execSQL("INSERT INTO orders (id, user_id, amount) VALUES (1, 1, 100)");
    defer ins2.close(testing.allocator);
    var ins3 = try db.execSQL("INSERT INTO orders (id, user_id, amount) VALUES (2, 1, 200)");
    defer ins3.close(testing.allocator);

    // Query each table independently
    var r_users = try db.execSQL("SELECT name FROM users");
    defer r_users.close(testing.allocator);
    var u_count: usize = 0;
    while (try r_users.rows.?.next()) |*row_ptr| {
        var row = row_ptr.*;
        defer row.deinit();
        u_count += 1;
    }
    try testing.expectEqual(@as(usize, 1), u_count);

    var r_orders = try db.execSQL("SELECT amount FROM orders");
    defer r_orders.close(testing.allocator);
    var o_count: usize = 0;
    while (try r_orders.rows.?.next()) |*row_ptr| {
        var row = row_ptr.*;
        defer row.deinit();
        o_count += 1;
    }
    try testing.expectEqual(@as(usize, 2), o_count);
}

test "parse error returns ParseError" {
    const path = "test_eng_parse_err.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    const result = db.execSQL("SELEKT * FORM users");
    try testing.expectError(EngineError.ParseError, result);
}

test "DROP nonexistent table without IF EXISTS" {
    const path = "test_eng_drop_noex.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    const result = db.execSQL("DROP TABLE nonexistent");
    try testing.expectError(EngineError.TableNotFound, result);
}

test "INNER JOIN two tables" {
    const path = "test_eng_join_inner.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    // Create two tables
    var r1 = try db.execSQL("CREATE TABLE users (id INTEGER, name TEXT)");
    r1.close(testing.allocator);
    var r2 = try db.execSQL("CREATE TABLE orders (user_id INTEGER, product TEXT)");
    r2.close(testing.allocator);

    // Insert data
    var ins1 = try db.execSQL("INSERT INTO users VALUES (1, 'Alice')");
    ins1.close(testing.allocator);
    var ins2 = try db.execSQL("INSERT INTO users VALUES (2, 'Bob')");
    ins2.close(testing.allocator);
    var ins3 = try db.execSQL("INSERT INTO orders VALUES (1, 'Widget')");
    ins3.close(testing.allocator);
    var ins4 = try db.execSQL("INSERT INTO orders VALUES (1, 'Gadget')");
    ins4.close(testing.allocator);
    var ins5 = try db.execSQL("INSERT INTO orders VALUES (2, 'Gizmo')");
    ins5.close(testing.allocator);

    // Join query
    var result = try db.execSQL("SELECT u.name, o.product FROM users u JOIN orders o ON u.id = o.user_id");
    defer result.close(testing.allocator);

    var count: usize = 0;
    while (try result.rows.?.next()) |*rp| {
        var row = rp.*;
        defer row.deinit();
        count += 1;
    }
    try testing.expectEqual(@as(usize, 3), count);
}

test "LEFT JOIN with unmatched rows" {
    const path = "test_eng_join_left.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r1 = try db.execSQL("CREATE TABLE departments (id INTEGER, name TEXT)");
    r1.close(testing.allocator);
    var r2 = try db.execSQL("CREATE TABLE employees (dept_id INTEGER, name TEXT)");
    r2.close(testing.allocator);

    var ins1 = try db.execSQL("INSERT INTO departments VALUES (1, 'Engineering')");
    ins1.close(testing.allocator);
    var ins2 = try db.execSQL("INSERT INTO departments VALUES (2, 'Marketing')");
    ins2.close(testing.allocator);
    var ins3 = try db.execSQL("INSERT INTO employees VALUES (1, 'Alice')");
    ins3.close(testing.allocator);

    // LEFT JOIN — Marketing has no employees, should still appear
    var result = try db.execSQL("SELECT d.name, e.name FROM departments d LEFT JOIN employees e ON d.id = e.dept_id");
    defer result.close(testing.allocator);

    var count: usize = 0;
    var null_found = false;
    while (try result.rows.?.next()) |*row_ptr| {
        var row = row_ptr.*;
        defer row.deinit();
        if (row.values.len >= 2 and row.values[1] == .null_value) {
            null_found = true;
        }
        count += 1;
    }
    try testing.expectEqual(@as(usize, 2), count);
    try testing.expect(null_found);
}

test "GROUP BY with COUNT" {
    const path = "test_eng_group_count.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r1 = try db.execSQL("CREATE TABLE sales (category TEXT, amount INTEGER)");
    r1.close(testing.allocator);

    var ins1 = try db.execSQL("INSERT INTO sales VALUES ('A', 10)");
    ins1.close(testing.allocator);
    var ins2 = try db.execSQL("INSERT INTO sales VALUES ('A', 20)");
    ins2.close(testing.allocator);
    var ins3 = try db.execSQL("INSERT INTO sales VALUES ('B', 30)");
    ins3.close(testing.allocator);
    var ins4 = try db.execSQL("INSERT INTO sales VALUES ('A', 40)");
    ins4.close(testing.allocator);

    var result = try db.execSQL("SELECT category, COUNT(*) FROM sales GROUP BY category");
    defer result.close(testing.allocator);

    var count: usize = 0;
    while (try result.rows.?.next()) |*row_ptr| {
        var row = row_ptr.*;
        defer row.deinit();
        count += 1;
    }
    // Two groups: A and B
    try testing.expectEqual(@as(usize, 2), count);
}

test "GROUP BY with SUM" {
    const path = "test_eng_group_sum.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r1 = try db.execSQL("CREATE TABLE expenses (dept TEXT, cost INTEGER)");
    r1.close(testing.allocator);

    var ins1 = try db.execSQL("INSERT INTO expenses VALUES ('eng', 100)");
    ins1.close(testing.allocator);
    var ins2 = try db.execSQL("INSERT INTO expenses VALUES ('eng', 200)");
    ins2.close(testing.allocator);
    var ins3 = try db.execSQL("INSERT INTO expenses VALUES ('mkt', 50)");
    ins3.close(testing.allocator);

    var result = try db.execSQL("SELECT dept, SUM(cost) FROM expenses GROUP BY dept");
    defer result.close(testing.allocator);

    var count: usize = 0;
    while (try result.rows.?.next()) |*row_ptr| {
        var row = row_ptr.*;
        defer row.deinit();
        count += 1;
    }
    try testing.expectEqual(@as(usize, 2), count);
}

test "AVG aggregate" {
    const path = "test_eng_avg.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r1 = try db.execSQL("CREATE TABLE scores (val INTEGER)");
    r1.close(testing.allocator);

    var ins1 = try db.execSQL("INSERT INTO scores VALUES (10)");
    ins1.close(testing.allocator);
    var ins2 = try db.execSQL("INSERT INTO scores VALUES (20)");
    ins2.close(testing.allocator);
    var ins3 = try db.execSQL("INSERT INTO scores VALUES (30)");
    ins3.close(testing.allocator);

    var result = try db.execSQL("SELECT AVG(val) FROM scores");
    defer result.close(testing.allocator);

    if (try result.rows.?.next()) |*row_ptr| {
        var row = row_ptr.*;
        defer row.deinit();
        // AVG(10,20,30) = 20.0
        const avg_val = row.values[0];
        switch (avg_val) {
            .real => |r| try testing.expectApproxEqAbs(@as(f64, 20.0), r, 0.001),
            .integer => |i| try testing.expectEqual(@as(i64, 20), i),
            else => return error.TestUnexpectedResult,
        }
    } else {
        return error.TestUnexpectedResult;
    }
}

test "multiple aggregates in one query" {
    const path = "test_eng_multi_agg.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r1 = try db.execSQL("CREATE TABLE items (price INTEGER)");
    r1.close(testing.allocator);

    var ins1 = try db.execSQL("INSERT INTO items VALUES (5)");
    ins1.close(testing.allocator);
    var ins2 = try db.execSQL("INSERT INTO items VALUES (15)");
    ins2.close(testing.allocator);
    var ins3 = try db.execSQL("INSERT INTO items VALUES (25)");
    ins3.close(testing.allocator);

    var result = try db.execSQL("SELECT COUNT(*), MIN(price), MAX(price), SUM(price) FROM items");
    defer result.close(testing.allocator);

    if (try result.rows.?.next()) |*row_ptr| {
        var row = row_ptr.*;
        defer row.deinit();
        // Should have 4 values
        try testing.expectEqual(@as(usize, 4), row.values.len);
    } else {
        return error.TestUnexpectedResult;
    }
}

test "ORDER BY DESC" {
    const path = "test_eng_order_desc.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r1 = try db.execSQL("CREATE TABLE nums (val INTEGER)");
    r1.close(testing.allocator);

    var ins1 = try db.execSQL("INSERT INTO nums VALUES (3)");
    ins1.close(testing.allocator);
    var ins2 = try db.execSQL("INSERT INTO nums VALUES (1)");
    ins2.close(testing.allocator);
    var ins3 = try db.execSQL("INSERT INTO nums VALUES (2)");
    ins3.close(testing.allocator);

    var result = try db.execSQL("SELECT val FROM nums ORDER BY val DESC");
    defer result.close(testing.allocator);

    var prev_val: ?i64 = null;
    while (try result.rows.?.next()) |*row_ptr| {
        var row = row_ptr.*;
        defer row.deinit();
        const val = row.values[0].toInteger() orelse continue;
        if (prev_val) |pv| {
            try testing.expect(val <= pv);
        }
        prev_val = val;
    }
    try testing.expect(prev_val != null);
}

test "SELECT with LIKE in WHERE" {
    const path = "test_eng_like.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r1 = try db.execSQL("CREATE TABLE products (name TEXT)");
    r1.close(testing.allocator);

    var ins1 = try db.execSQL("INSERT INTO products VALUES ('Apple Pie')");
    ins1.close(testing.allocator);
    var ins2 = try db.execSQL("INSERT INTO products VALUES ('Banana Split')");
    ins2.close(testing.allocator);
    var ins3 = try db.execSQL("INSERT INTO products VALUES ('Apple Sauce')");
    ins3.close(testing.allocator);

    var result = try db.execSQL("SELECT name FROM products WHERE name LIKE 'Apple%'");
    defer result.close(testing.allocator);

    var count: usize = 0;
    while (try result.rows.?.next()) |*row_ptr| {
        var row = row_ptr.*;
        defer row.deinit();
        count += 1;
    }
    try testing.expectEqual(@as(usize, 2), count);
}

test "SELECT with BETWEEN in WHERE" {
    const path = "test_eng_between.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r1 = try db.execSQL("CREATE TABLE temps (day INTEGER, temp INTEGER)");
    r1.close(testing.allocator);

    var ins1 = try db.execSQL("INSERT INTO temps VALUES (1, 15)");
    ins1.close(testing.allocator);
    var ins2 = try db.execSQL("INSERT INTO temps VALUES (2, 25)");
    ins2.close(testing.allocator);
    var ins3 = try db.execSQL("INSERT INTO temps VALUES (3, 35)");
    ins3.close(testing.allocator);
    var ins4 = try db.execSQL("INSERT INTO temps VALUES (4, 20)");
    ins4.close(testing.allocator);

    var result = try db.execSQL("SELECT day FROM temps WHERE temp BETWEEN 15 AND 25");
    defer result.close(testing.allocator);

    var count: usize = 0;
    while (try result.rows.?.next()) |*row_ptr| {
        var row = row_ptr.*;
        defer row.deinit();
        count += 1;
    }
    // temp 15, 25, 20 match
    try testing.expectEqual(@as(usize, 3), count);
}

test "SELECT with IN list" {
    const path = "test_eng_in_list.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r1 = try db.execSQL("CREATE TABLE colors (id INTEGER, name TEXT)");
    r1.close(testing.allocator);

    var ins1 = try db.execSQL("INSERT INTO colors VALUES (1, 'red')");
    ins1.close(testing.allocator);
    var ins2 = try db.execSQL("INSERT INTO colors VALUES (2, 'blue')");
    ins2.close(testing.allocator);
    var ins3 = try db.execSQL("INSERT INTO colors VALUES (3, 'green')");
    ins3.close(testing.allocator);

    var result = try db.execSQL("SELECT name FROM colors WHERE name IN ('red', 'green')");
    defer result.close(testing.allocator);

    var count: usize = 0;
    while (try result.rows.?.next()) |*row_ptr| {
        var row = row_ptr.*;
        defer row.deinit();
        count += 1;
    }
    try testing.expectEqual(@as(usize, 2), count);
}

test "UPDATE with expression" {
    const path = "test_eng_upd_expr.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r1 = try db.execSQL("CREATE TABLE inventory (item TEXT, qty INTEGER)");
    r1.close(testing.allocator);

    var ins1 = try db.execSQL("INSERT INTO inventory VALUES ('bolts', 10)");
    ins1.close(testing.allocator);
    var ins2 = try db.execSQL("INSERT INTO inventory VALUES ('nuts', 20)");
    ins2.close(testing.allocator);

    // Update with arithmetic expression
    var upd = try db.execSQL("UPDATE inventory SET qty = qty + 5 WHERE item = 'bolts'");
    upd.close(testing.allocator);
    try testing.expectEqual(@as(u64, 1), upd.rows_affected);

    // Verify
    var result = try db.execSQL("SELECT qty FROM inventory WHERE item = 'bolts'");
    defer result.close(testing.allocator);

    if (try result.rows.?.next()) |*row_ptr| {
        var row = row_ptr.*;
        defer row.deinit();
        try testing.expectEqual(@as(i64, 15), row.values[0].toInteger().?);
    } else {
        return error.TestUnexpectedResult;
    }
}

test "INSERT and DELETE then re-insert" {
    const path = "test_eng_reinsert.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r1 = try db.execSQL("CREATE TABLE log (msg TEXT)");
    r1.close(testing.allocator);

    var ins1 = try db.execSQL("INSERT INTO log VALUES ('first')");
    ins1.close(testing.allocator);
    var ins2 = try db.execSQL("INSERT INTO log VALUES ('second')");
    ins2.close(testing.allocator);

    var del = try db.execSQL("DELETE FROM log");
    del.close(testing.allocator);
    try testing.expectEqual(@as(u64, 2), del.rows_affected);

    // Empty now
    var empty = try db.execSQL("SELECT COUNT(*) FROM log");
    defer empty.close(testing.allocator);
    if (try empty.rows.?.next()) |*row_ptr| {
        var row = row_ptr.*;
        defer row.deinit();
        try testing.expectEqual(@as(i64, 0), row.values[0].toInteger().?);
    }

    // Re-insert should work
    var ins3 = try db.execSQL("INSERT INTO log VALUES ('third')");
    ins3.close(testing.allocator);

    var check = try db.execSQL("SELECT COUNT(*) FROM log");
    defer check.close(testing.allocator);
    if (try check.rows.?.next()) |*row_ptr| {
        var row = row_ptr.*;
        defer row.deinit();
        try testing.expectEqual(@as(i64, 1), row.values[0].toInteger().?);
    }
}

test "large INSERT batch" {
    const path = "test_eng_large_insert.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r1 = try db.execSQL("CREATE TABLE big (id INTEGER, val TEXT)");
    r1.close(testing.allocator);

    // Insert 100 rows
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        var buf: [128]u8 = undefined;
        const sql = std.fmt.bufPrint(&buf, "INSERT INTO big VALUES ({d}, 'row_{d}')", .{ i, i }) catch unreachable;
        var ins = try db.execSQL(sql);
        ins.close(testing.allocator);
    }

    // Verify count
    var result = try db.execSQL("SELECT COUNT(*) FROM big");
    defer result.close(testing.allocator);

    if (try result.rows.?.next()) |*row_ptr| {
        var row = row_ptr.*;
        defer row.deinit();
        try testing.expectEqual(@as(i64, 100), row.values[0].toInteger().?);
    }
}

test "SELECT with negative value in WHERE" {
    const path = "test_eng_neg_where.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r1 = try db.execSQL("CREATE TABLE balances (acct TEXT, amount INTEGER)");
    r1.close(testing.allocator);

    var ins1 = try db.execSQL("INSERT INTO balances VALUES ('a', -50)");
    ins1.close(testing.allocator);
    var ins2 = try db.execSQL("INSERT INTO balances VALUES ('b', 100)");
    ins2.close(testing.allocator);
    var ins3 = try db.execSQL("INSERT INTO balances VALUES ('c', -10)");
    ins3.close(testing.allocator);

    var result = try db.execSQL("SELECT acct FROM balances WHERE amount < 0");
    defer result.close(testing.allocator);

    var count: usize = 0;
    while (try result.rows.?.next()) |*row_ptr| {
        var row = row_ptr.*;
        defer row.deinit();
        count += 1;
    }
    try testing.expectEqual(@as(usize, 2), count);
}

test "empty table aggregate returns zero" {
    const path = "test_eng_empty_agg.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r1 = try db.execSQL("CREATE TABLE empty_t (val INTEGER)");
    r1.close(testing.allocator);

    var result = try db.execSQL("SELECT COUNT(*) FROM empty_t");
    defer result.close(testing.allocator);

    if (try result.rows.?.next()) |*row_ptr| {
        var row = row_ptr.*;
        defer row.deinit();
        try testing.expectEqual(@as(i64, 0), row.values[0].toInteger().?);
    } else {
        return error.TestUnexpectedResult;
    }
}

test "INSERT with NULL explicit and column types" {
    const path = "test_eng_null_insert.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r1 = try db.execSQL("CREATE TABLE nullable (a INTEGER, b TEXT, c REAL)");
    r1.close(testing.allocator);

    var ins1 = try db.execSQL("INSERT INTO nullable VALUES (1, NULL, 3.14)");
    ins1.close(testing.allocator);
    var ins2 = try db.execSQL("INSERT INTO nullable VALUES (NULL, 'hello', NULL)");
    ins2.close(testing.allocator);

    var result = try db.execSQL("SELECT * FROM nullable");
    defer result.close(testing.allocator);

    var count: usize = 0;
    while (try result.rows.?.next()) |*row_ptr| {
        var row = row_ptr.*;
        defer row.deinit();
        count += 1;
    }
    try testing.expectEqual(@as(usize, 2), count);
}

test "SELECT with IS NULL filter" {
    const path = "test_eng_is_null.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r1 = try db.execSQL("CREATE TABLE maybe (id INTEGER, note TEXT)");
    r1.close(testing.allocator);

    var ins1 = try db.execSQL("INSERT INTO maybe VALUES (1, 'present')");
    ins1.close(testing.allocator);
    var ins2 = try db.execSQL("INSERT INTO maybe VALUES (2, NULL)");
    ins2.close(testing.allocator);
    var ins3 = try db.execSQL("INSERT INTO maybe VALUES (3, NULL)");
    ins3.close(testing.allocator);

    var result = try db.execSQL("SELECT id FROM maybe WHERE note IS NULL");
    defer result.close(testing.allocator);

    var count: usize = 0;
    while (try result.rows.?.next()) |*row_ptr| {
        var row = row_ptr.*;
        defer row.deinit();
        count += 1;
    }
    try testing.expectEqual(@as(usize, 2), count);
}

test "transaction statements return OK" {
    const path = "test_eng_txn.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r1 = try db.execSQL("BEGIN");
    r1.close(testing.allocator);
    try testing.expectEqualStrings("BEGIN", r1.message);

    var r2 = try db.execSQL("COMMIT");
    r2.close(testing.allocator);
    try testing.expectEqualStrings("COMMIT", r2.message);
}

// ── Index Selection Tests ────────────────────────────────────────────

test "index scan: WHERE on PRIMARY KEY integer column" {
    const path = "test_eng_idxscan1.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r0 = try db.execSQL("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);");
    r0.close(testing.allocator);

    var r1 = try db.execSQL("INSERT INTO users (id, name) VALUES (1, 'Alice');");
    r1.close(testing.allocator);
    var r2 = try db.execSQL("INSERT INTO users (id, name) VALUES (2, 'Bob');");
    r2.close(testing.allocator);
    var r3 = try db.execSQL("INSERT INTO users (id, name) VALUES (3, 'Charlie');");
    r3.close(testing.allocator);

    // This WHERE should use index scan on the 'id' PK column
    var result = try db.execSQL("SELECT * FROM users WHERE id = 2;");
    defer result.close(testing.allocator);

    try testing.expect(result.rows != null);
    var count: usize = 0;
    var found_bob = false;
    while (try result.rows.?.next()) |*rp| {
        var row = rp.*;
        defer row.deinit();
        count += 1;
        // Check 'name' column is 'Bob'
        for (row.columns, row.values) |col, val| {
            if (std.mem.eql(u8, col, "name")) {
                if (val == .text) {
                    if (std.mem.eql(u8, val.text, "Bob")) found_bob = true;
                }
            }
        }
    }
    try testing.expectEqual(@as(usize, 1), count);
    try testing.expect(found_bob);
}

test "index scan: WHERE PK not found returns empty" {
    const path = "test_eng_idxscan2.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r0 = try db.execSQL("CREATE TABLE items (id INTEGER PRIMARY KEY, val TEXT);");
    r0.close(testing.allocator);
    var r1 = try db.execSQL("INSERT INTO items (id, val) VALUES (10, 'ten');");
    r1.close(testing.allocator);

    // Lookup a non-existent key
    var result = try db.execSQL("SELECT * FROM items WHERE id = 999;");
    defer result.close(testing.allocator);

    try testing.expect(result.rows != null);
    var count: usize = 0;
    while (try result.rows.?.next()) |*rp| {
        var row = rp.*;
        defer row.deinit();
        count += 1;
    }
    try testing.expectEqual(@as(usize, 0), count);
}

test "index scan: multiple inserts then PK lookup" {
    const path = "test_eng_idxscan3.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r0 = try db.execSQL("CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price INTEGER);");
    r0.close(testing.allocator);

    // Insert 20 rows
    var ins_idx: usize = 0;
    while (ins_idx < 20) : (ins_idx += 1) {
        const sql = try std.fmt.allocPrint(testing.allocator, "INSERT INTO products (id, name, price) VALUES ({d}, 'product_{d}', {d});", .{ ins_idx + 1, ins_idx + 1, (ins_idx + 1) * 10 });
        defer testing.allocator.free(sql);
        var r = try db.execSQL(sql);
        r.close(testing.allocator);
    }

    // Look up the 15th product
    var result = try db.execSQL("SELECT * FROM products WHERE id = 15;");
    defer result.close(testing.allocator);

    try testing.expect(result.rows != null);
    const row_ptr = try result.rows.?.next();
    try testing.expect(row_ptr != null);
    var row = row_ptr.?;
    defer row.deinit();

    // Verify price = 150
    for (row.columns, row.values) |col, val| {
        if (std.mem.eql(u8, col, "price")) {
            try testing.expectEqual(@as(i64, 150), val.integer);
        }
    }

    // No more rows
    const next = try result.rows.?.next();
    try testing.expect(next == null);
}

test "index scan: DELETE then index lookup returns empty" {
    const path = "test_eng_idxscan4.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r0 = try db.execSQL("CREATE TABLE items (id INTEGER PRIMARY KEY, val TEXT);");
    r0.close(testing.allocator);
    var r1 = try db.execSQL("INSERT INTO items (id, val) VALUES (1, 'one');");
    r1.close(testing.allocator);
    var r2 = try db.execSQL("INSERT INTO items (id, val) VALUES (2, 'two');");
    r2.close(testing.allocator);

    // Delete id=1
    var rdel = try db.execSQL("DELETE FROM items WHERE id = 1;");
    rdel.close(testing.allocator);
    try testing.expectEqual(@as(u64, 1), rdel.rows_affected);

    // Index lookup for deleted row should return empty
    var result = try db.execSQL("SELECT * FROM items WHERE id = 1;");
    defer result.close(testing.allocator);
    try testing.expect(result.rows != null);
    const row_ptr = try result.rows.?.next();
    try testing.expect(row_ptr == null);

    // But id=2 should still be found
    var r3 = try db.execSQL("SELECT * FROM items WHERE id = 2;");
    defer r3.close(testing.allocator);
    try testing.expect(r3.rows != null);
    const row2 = try r3.rows.?.next();
    try testing.expect(row2 != null);
    var r2row = row2.?;
    defer r2row.deinit();
}

test "index scan: UPDATE then index lookup returns updated value" {
    const path = "test_eng_idxscan5.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r0 = try db.execSQL("CREATE TABLE items (id INTEGER PRIMARY KEY, val TEXT);");
    r0.close(testing.allocator);
    var r1 = try db.execSQL("INSERT INTO items (id, val) VALUES (1, 'old');");
    r1.close(testing.allocator);

    // Update the value
    var rup = try db.execSQL("UPDATE items SET val = 'new' WHERE id = 1;");
    rup.close(testing.allocator);
    try testing.expectEqual(@as(u64, 1), rup.rows_affected);

    // Index lookup should return the updated value
    var result = try db.execSQL("SELECT * FROM items WHERE id = 1;");
    defer result.close(testing.allocator);
    try testing.expect(result.rows != null);
    const row_ptr = try result.rows.?.next();
    try testing.expect(row_ptr != null);
    var row = row_ptr.?;
    defer row.deinit();

    for (row.columns, row.values) |col, val| {
        if (std.mem.eql(u8, col, "val")) {
            try testing.expect(val == .text);
            try testing.expectEqualStrings("new", val.text);
        }
    }
}

test "index scan: text PRIMARY KEY lookup" {
    const path = "test_eng_idxscan6.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r0 = try db.execSQL("CREATE TABLE config (key TEXT PRIMARY KEY, value TEXT);");
    r0.close(testing.allocator);
    var r1 = try db.execSQL("INSERT INTO config (key, value) VALUES ('host', 'localhost');");
    r1.close(testing.allocator);
    var r2 = try db.execSQL("INSERT INTO config (key, value) VALUES ('port', '8080');");
    r2.close(testing.allocator);

    // Text PK lookup
    var result = try db.execSQL("SELECT * FROM config WHERE key = 'port';");
    defer result.close(testing.allocator);
    try testing.expect(result.rows != null);
    const row_ptr = try result.rows.?.next();
    try testing.expect(row_ptr != null);
    var row = row_ptr.?;
    defer row.deinit();

    for (row.columns, row.values) |col, val| {
        if (std.mem.eql(u8, col, "value")) {
            try testing.expect(val == .text);
            try testing.expectEqualStrings("8080", val.text);
        }
    }
}

test "index scan: non-PK column falls back to full scan" {
    const path = "test_eng_idxscan7.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r0 = try db.execSQL("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT);");
    r0.close(testing.allocator);
    var r1 = try db.execSQL("INSERT INTO items (id, name) VALUES (1, 'Alice');");
    r1.close(testing.allocator);
    var r2 = try db.execSQL("INSERT INTO items (id, name) VALUES (2, 'Alice');");
    r2.close(testing.allocator);

    // WHERE on non-indexed column — should fall back to full scan
    var result = try db.execSQL("SELECT * FROM items WHERE name = 'Alice';");
    defer result.close(testing.allocator);
    try testing.expect(result.rows != null);
    var count: usize = 0;
    while (try result.rows.?.next()) |*rp| {
        var row = rp.*;
        defer row.deinit();
        count += 1;
    }
    try testing.expectEqual(@as(usize, 2), count);
}

test "index scan: persistence across close/reopen" {
    const path = "test_eng_idxscan8.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    // Create and populate
    {
        var db = try createTestDb(testing.allocator, path);
        var r0 = try db.execSQL("CREATE TABLE items (id INTEGER PRIMARY KEY, val TEXT);");
        r0.close(testing.allocator);
        var r1 = try db.execSQL("INSERT INTO items (id, val) VALUES (42, 'answer');");
        r1.close(testing.allocator);
        db.close();
    }

    // Reopen and query via index
    {
        var db = try Database.open(testing.allocator, path, .{});
        defer cleanupTestDb(&db, path);

        var result = try db.execSQL("SELECT * FROM items WHERE id = 42;");
        defer result.close(testing.allocator);
        try testing.expect(result.rows != null);
        const row_ptr = try result.rows.?.next();
        try testing.expect(row_ptr != null);
        var row = row_ptr.?;
        defer row.deinit();

        for (row.columns, row.values) |col, val| {
            if (std.mem.eql(u8, col, "val")) {
                try testing.expect(val == .text);
                try testing.expectEqualStrings("answer", val.text);
            }
        }
    }
}

test "extractEqualityPredicate: column = integer" {
    const left = ast_mod.Expr{ .column_ref = .{ .name = "id" } };
    const right = ast_mod.Expr{ .integer_literal = 42 };
    const expr = ast_mod.Expr{ .binary_op = .{ .op = .equal, .left = &left, .right = &right } };
    const result = extractEqualityPredicate(&expr);
    try testing.expect(result != null);
    try testing.expectEqualStrings("id", result.?.column_name);
    try testing.expectEqual(@as(i64, 42), result.?.value_type.integer);
}

test "extractEqualityPredicate: integer = column (reversed)" {
    const left = ast_mod.Expr{ .integer_literal = 7 };
    const right = ast_mod.Expr{ .column_ref = .{ .name = "age" } };
    const expr = ast_mod.Expr{ .binary_op = .{ .op = .equal, .left = &left, .right = &right } };
    const result = extractEqualityPredicate(&expr);
    try testing.expect(result != null);
    try testing.expectEqualStrings("age", result.?.column_name);
    try testing.expectEqual(@as(i64, 7), result.?.value_type.integer);
}

test "extractEqualityPredicate: column = string" {
    const left = ast_mod.Expr{ .column_ref = .{ .name = "name" } };
    const right = ast_mod.Expr{ .string_literal = "Alice" };
    const expr = ast_mod.Expr{ .binary_op = .{ .op = .equal, .left = &left, .right = &right } };
    const result = extractEqualityPredicate(&expr);
    try testing.expect(result != null);
    try testing.expectEqualStrings("name", result.?.column_name);
    try testing.expectEqualStrings("Alice", result.?.value_type.text);
}

test "extractEqualityPredicate: returns null for non-equality" {
    const left = ast_mod.Expr{ .column_ref = .{ .name = "id" } };
    const right = ast_mod.Expr{ .integer_literal = 5 };
    const expr = ast_mod.Expr{ .binary_op = .{ .op = .greater_than, .left = &left, .right = &right } };
    const result = extractEqualityPredicate(&expr);
    try testing.expect(result == null);
}

test "extractEqualityPredicate: returns null for qualified column" {
    const left = ast_mod.Expr{ .column_ref = .{ .name = "id", .prefix = "t" } };
    const right = ast_mod.Expr{ .integer_literal = 5 };
    const expr = ast_mod.Expr{ .binary_op = .{ .op = .equal, .left = &left, .right = &right } };
    const result = extractEqualityPredicate(&expr);
    try testing.expect(result == null);
}

test "valueToIndexKey: integer ordering" {
    // Verify that encoding preserves lexicographic order for integers
    const k1 = try integerToIndexKey(testing.allocator, -10);
    defer testing.allocator.free(k1);
    const k2 = try integerToIndexKey(testing.allocator, 0);
    defer testing.allocator.free(k2);
    const k3 = try integerToIndexKey(testing.allocator, 10);
    defer testing.allocator.free(k3);

    try testing.expect(std.mem.order(u8, k1, k2) == .lt);
    try testing.expect(std.mem.order(u8, k2, k3) == .lt);
    try testing.expect(std.mem.order(u8, k1, k3) == .lt);
}

test "catalog index serialization roundtrip" {
    const columns = [_]catalog_mod.ColumnInfo{
        .{ .name = "id", .column_type = .integer, .flags = .{ .primary_key = true, .not_null = true } },
        .{ .name = "name", .column_type = .text, .flags = .{ .not_null = true } },
    };
    const indexes = [_]catalog_mod.IndexInfo{
        .{ .column_name = "id", .column_index = 0, .root_page_id = 42 },
    };

    const data = try catalog_mod.serializeTableFull(testing.allocator, &columns, &.{}, &indexes, 10);
    defer testing.allocator.free(data);

    var info = try catalog_mod.deserializeTable(testing.allocator, "test", data);
    defer info.deinit(testing.allocator);

    try testing.expectEqual(@as(u32, 10), info.data_root_page_id);
    try testing.expectEqual(@as(usize, 2), info.columns.len);
    try testing.expectEqual(@as(usize, 1), info.indexes.len);
    try testing.expectEqualStrings("id", info.indexes[0].column_name);
    try testing.expectEqual(@as(u16, 0), info.indexes[0].column_index);
    try testing.expectEqual(@as(u32, 42), info.indexes[0].root_page_id);
}

test "catalog index backward compatibility (no indexes)" {
    // Serialize without indexes (old format)
    const columns = [_]catalog_mod.ColumnInfo{
        .{ .name = "id", .column_type = .integer, .flags = .{} },
    };
    const data = try catalog_mod.serializeTable(testing.allocator, &columns, &.{}, 5);
    defer testing.allocator.free(data);

    var info = try catalog_mod.deserializeTable(testing.allocator, "test", data);
    defer info.deinit(testing.allocator);

    try testing.expectEqual(@as(u32, 5), info.data_root_page_id);
    try testing.expectEqual(@as(usize, 1), info.columns.len);
    // serializeTable now includes index_count=0, which deserialize reads correctly
    try testing.expectEqual(@as(usize, 0), info.indexes.len);
}

// ── WAL Mode Integration Tests ─────────────────────────────────────────

test "WAL mode: open and close" {
    const path = "test_eng_wal_open.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    defer std.fs.cwd().deleteFile("test_eng_wal_open.db-wal") catch {};

    var db = try Database.open(testing.allocator, path, .{ .wal_mode = true });
    try testing.expect(db.wal != null);
    db.close();
}

test "WAL mode: CREATE TABLE and INSERT" {
    const path = "test_eng_wal_insert.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    defer std.fs.cwd().deleteFile("test_eng_wal_insert.db-wal") catch {};

    var db = try Database.open(testing.allocator, path, .{ .wal_mode = true });
    defer db.close();

    var r1 = try db.execSQL("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)");
    r1.close(testing.allocator);

    var r2 = try db.execSQL("INSERT INTO items (id, name) VALUES (1, 'apple')");
    r2.close(testing.allocator);
    try testing.expectEqual(@as(u64, 1), r2.rows_affected);

    var r3 = try db.execSQL("INSERT INTO items (id, name) VALUES (2, 'banana')");
    r3.close(testing.allocator);
    try testing.expectEqual(@as(u64, 1), r3.rows_affected);

    // Verify data is readable
    var r4 = try db.execSQL("SELECT id, name FROM items ORDER BY id");
    defer r4.close(testing.allocator);
    var count: usize = 0;
    while (try r4.rows.?.next()) |*row_ptr| {
        var row = row_ptr.*;
        defer row.deinit();
        count += 1;
    }
    try testing.expectEqual(@as(usize, 2), count);
}

test "WAL mode: data persistence across close and reopen" {
    const path = "test_eng_wal_persist.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    defer std.fs.cwd().deleteFile("test_eng_wal_persist.db-wal") catch {};

    // Session 1: create table and insert data
    {
        var db = try Database.open(testing.allocator, path, .{ .wal_mode = true });
        var r1 = try db.execSQL("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)");
        r1.close(testing.allocator);
        var r2 = try db.execSQL("INSERT INTO users (id, name) VALUES (1, 'Alice')");
        r2.close(testing.allocator);
        var r3 = try db.execSQL("INSERT INTO users (id, name) VALUES (2, 'Bob')");
        r3.close(testing.allocator);
        db.close(); // checkpoints WAL to main DB
    }

    // Session 2: reopen and verify data persisted
    {
        var db = try Database.open(testing.allocator, path, .{ .wal_mode = true });
        defer db.close();

        var r = try db.execSQL("SELECT id, name FROM users ORDER BY id");
        defer r.close(testing.allocator);
        var count: usize = 0;
        while (try r.rows.?.next()) |*row_ptr| {
            var row = row_ptr.*;
            defer row.deinit();
            count += 1;
        }
        try testing.expectEqual(@as(usize, 2), count);
    }
}

test "WAL mode: UPDATE and DELETE" {
    const path = "test_eng_wal_upd_del.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    defer std.fs.cwd().deleteFile("test_eng_wal_upd_del.db-wal") catch {};

    var db = try Database.open(testing.allocator, path, .{ .wal_mode = true });
    defer db.close();

    var r1 = try db.execSQL("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)");
    r1.close(testing.allocator);
    var r2 = try db.execSQL("INSERT INTO t (id, val) VALUES (1, 10)");
    r2.close(testing.allocator);
    var r3 = try db.execSQL("INSERT INTO t (id, val) VALUES (2, 20)");
    r3.close(testing.allocator);
    var r4 = try db.execSQL("INSERT INTO t (id, val) VALUES (3, 30)");
    r4.close(testing.allocator);

    // UPDATE
    var r5 = try db.execSQL("UPDATE t SET val = 99 WHERE id = 2");
    r5.close(testing.allocator);
    try testing.expectEqual(@as(u64, 1), r5.rows_affected);

    // DELETE
    var r6 = try db.execSQL("DELETE FROM t WHERE id = 3");
    r6.close(testing.allocator);
    try testing.expectEqual(@as(u64, 1), r6.rows_affected);

    // Verify remaining rows
    var r7 = try db.execSQL("SELECT id, val FROM t ORDER BY id");
    defer r7.close(testing.allocator);
    var count: usize = 0;
    while (try r7.rows.?.next()) |*row_ptr| {
        var row = row_ptr.*;
        defer row.deinit();
        count += 1;
    }
    try testing.expectEqual(@as(usize, 2), count);
}

test "WAL mode: aggregates" {
    const path = "test_eng_wal_agg.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    defer std.fs.cwd().deleteFile("test_eng_wal_agg.db-wal") catch {};

    var db = try Database.open(testing.allocator, path, .{ .wal_mode = true });
    defer db.close();

    var r1 = try db.execSQL("CREATE TABLE scores (id INTEGER PRIMARY KEY, score INTEGER)");
    r1.close(testing.allocator);
    var r2 = try db.execSQL("INSERT INTO scores (id, score) VALUES (1, 80)");
    r2.close(testing.allocator);
    var r3 = try db.execSQL("INSERT INTO scores (id, score) VALUES (2, 90)");
    r3.close(testing.allocator);
    var r4 = try db.execSQL("INSERT INTO scores (id, score) VALUES (3, 100)");
    r4.close(testing.allocator);

    var r5 = try db.execSQL("SELECT COUNT(*), SUM(score) FROM scores");
    defer r5.close(testing.allocator);
    var row_count: usize = 0;
    while (try r5.rows.?.next()) |*row_ptr| {
        var row = row_ptr.*;
        defer row.deinit();
        // COUNT(*) = 3
        try testing.expectEqual(Value{ .integer = 3 }, row.values[0]);
        // SUM = 270
        try testing.expectEqual(Value{ .integer = 270 }, row.values[1]);
        row_count += 1;
    }
    try testing.expectEqual(@as(usize, 1), row_count);
}

test "WAL mode: multiple tables" {
    const path = "test_eng_wal_multi.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    defer std.fs.cwd().deleteFile("test_eng_wal_multi.db-wal") catch {};

    var db = try Database.open(testing.allocator, path, .{ .wal_mode = true });
    defer db.close();

    var r1 = try db.execSQL("CREATE TABLE t1 (id INTEGER PRIMARY KEY, name TEXT)");
    r1.close(testing.allocator);
    var r2 = try db.execSQL("CREATE TABLE t2 (id INTEGER PRIMARY KEY, val INTEGER)");
    r2.close(testing.allocator);

    var r3 = try db.execSQL("INSERT INTO t1 (id, name) VALUES (1, 'hello')");
    r3.close(testing.allocator);
    var r4 = try db.execSQL("INSERT INTO t2 (id, val) VALUES (1, 42)");
    r4.close(testing.allocator);

    // Verify t1
    var r5 = try db.execSQL("SELECT name FROM t1");
    defer r5.close(testing.allocator);
    var count1: usize = 0;
    while (try r5.rows.?.next()) |*row_ptr| {
        var row = row_ptr.*;
        defer row.deinit();
        count1 += 1;
    }
    try testing.expectEqual(@as(usize, 1), count1);

    // Verify t2
    var r6 = try db.execSQL("SELECT val FROM t2");
    defer r6.close(testing.allocator);
    var count2: usize = 0;
    while (try r6.rows.?.next()) |*row_ptr| {
        var row = row_ptr.*;
        defer row.deinit();
        try testing.expectEqual(Value{ .integer = 42 }, row.values[0]);
        count2 += 1;
    }
    try testing.expectEqual(@as(usize, 1), count2);
}

test "WAL mode: DROP TABLE" {
    const path = "test_eng_wal_drop.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    defer std.fs.cwd().deleteFile("test_eng_wal_drop.db-wal") catch {};

    var db = try Database.open(testing.allocator, path, .{ .wal_mode = true });
    defer db.close();

    var r1 = try db.execSQL("CREATE TABLE temp_wal (id INTEGER PRIMARY KEY)");
    r1.close(testing.allocator);
    var r2 = try db.execSQL("INSERT INTO temp_wal (id) VALUES (1)");
    r2.close(testing.allocator);
    var r3 = try db.execSQL("DROP TABLE temp_wal");
    r3.close(testing.allocator);

    // Attempting to query dropped table should fail
    const r4 = db.execSQL("SELECT * FROM temp_wal");
    try testing.expectError(EngineError.AnalysisError, r4);
}

test "WAL mode: large batch insert" {
    const path = "test_eng_wal_batch.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    defer std.fs.cwd().deleteFile("test_eng_wal_batch.db-wal") catch {};

    var db = try Database.open(testing.allocator, path, .{ .wal_mode = true });
    defer db.close();

    var r1 = try db.execSQL("CREATE TABLE nums (id INTEGER PRIMARY KEY, val INTEGER)");
    r1.close(testing.allocator);

    // Insert 50 rows
    var idx: usize = 0;
    while (idx < 50) : (idx += 1) {
        var buf: [128]u8 = undefined;
        const sql = std.fmt.bufPrint(&buf, "INSERT INTO nums (id, val) VALUES ({d}, {d})", .{ idx + 1, (idx + 1) * 10 }) catch unreachable;
        var r = try db.execSQL(sql);
        r.close(testing.allocator);
    }

    var r2 = try db.execSQL("SELECT COUNT(*) FROM nums");
    defer r2.close(testing.allocator);
    while (try r2.rows.?.next()) |*row_ptr| {
        var row = row_ptr.*;
        defer row.deinit();
        try testing.expectEqual(Value{ .integer = 50 }, row.values[0]);
    }
}

// ── Error Handling Tests ──────────────────────────────────────────────

test "error: parse invalid SQL" {
    const path = "test_eng_err_parse.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var db = try Database.open(testing.allocator, path, .{});
    defer db.close();

    try testing.expectError(EngineError.ParseError, db.execSQL("SELEKT * FROM foo"));
    try testing.expectError(EngineError.ParseError, db.execSQL("CREATE TABLE"));
    try testing.expectError(EngineError.ParseError, db.execSQL("INSERT INTO"));
}

test "error: empty and whitespace SQL" {
    const path = "test_eng_err_empty.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var db = try Database.open(testing.allocator, path, .{});
    defer db.close();

    try testing.expectError(EngineError.ParseError, db.execSQL(""));
    try testing.expectError(EngineError.ParseError, db.execSQL("   "));
    try testing.expectError(EngineError.ParseError, db.execSQL(";"));
}

test "error: SELECT from non-existent table" {
    const path = "test_eng_err_no_table.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var db = try Database.open(testing.allocator, path, .{});
    defer db.close();

    try testing.expectError(EngineError.AnalysisError, db.execSQL("SELECT * FROM nonexistent"));
}

test "error: INSERT into non-existent table" {
    const path = "test_eng_err_ins_notbl.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var db = try Database.open(testing.allocator, path, .{});
    defer db.close();

    try testing.expectError(EngineError.AnalysisError, db.execSQL("INSERT INTO nonexistent (id) VALUES (1)"));
}

test "error: UPDATE non-existent table" {
    const path = "test_eng_err_upd_notbl.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var db = try Database.open(testing.allocator, path, .{});
    defer db.close();

    try testing.expectError(EngineError.AnalysisError, db.execSQL("UPDATE nonexistent SET x = 1"));
}

test "error: DELETE from non-existent table" {
    const path = "test_eng_err_del_notbl.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var db = try Database.open(testing.allocator, path, .{});
    defer db.close();

    try testing.expectError(EngineError.AnalysisError, db.execSQL("DELETE FROM nonexistent"));
}

test "error: CREATE TABLE that already exists" {
    const path = "test_eng_err_dup_tbl.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var db = try Database.open(testing.allocator, path, .{});
    defer db.close();

    var r1 = try db.execSQL("CREATE TABLE users (id INTEGER PRIMARY KEY)");
    r1.close(testing.allocator);

    try testing.expectError(EngineError.TableAlreadyExists, db.execSQL("CREATE TABLE users (id INTEGER)"));
}

test "error: DROP TABLE that does not exist" {
    const path = "test_eng_err_drop_notbl.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var db = try Database.open(testing.allocator, path, .{});
    defer db.close();

    try testing.expectError(EngineError.TableNotFound, db.execSQL("DROP TABLE nonexistent"));
}

test "DROP TABLE IF EXISTS on non-existent table succeeds" {
    const path = "test_eng_drop_ifex.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var db = try Database.open(testing.allocator, path, .{});
    defer db.close();

    var r = try db.execSQL("DROP TABLE IF EXISTS nonexistent");
    r.close(testing.allocator);
    try testing.expect(std.mem.eql(u8, "DROP TABLE", r.message));
}

test "error: SELECT with non-existent column" {
    const path = "test_eng_err_nocol.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var db = try Database.open(testing.allocator, path, .{});
    defer db.close();

    var r1 = try db.execSQL("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)");
    r1.close(testing.allocator);

    try testing.expectError(EngineError.AnalysisError, db.execSQL("SELECT nonexistent FROM items"));
}

test "error: INSERT column count mismatch" {
    const path = "test_eng_err_colmis.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var db = try Database.open(testing.allocator, path, .{});
    defer db.close();

    var r1 = try db.execSQL("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)");
    r1.close(testing.allocator);

    // Too many values
    try testing.expectError(EngineError.AnalysisError, db.execSQL("INSERT INTO items (id) VALUES (1, 'extra')"));
}

test "successive exec calls on same database" {
    const path = "test_eng_multi_exec.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var db = try Database.open(testing.allocator, path, .{});
    defer db.close();

    // Multiple DDL+DML calls in sequence
    var r1 = try db.execSQL("CREATE TABLE a (id INTEGER PRIMARY KEY)");
    r1.close(testing.allocator);

    var r2 = try db.execSQL("CREATE TABLE b (id INTEGER PRIMARY KEY, ref INTEGER)");
    r2.close(testing.allocator);

    var r3 = try db.execSQL("INSERT INTO a (id) VALUES (1)");
    r3.close(testing.allocator);

    var r4 = try db.execSQL("INSERT INTO a (id) VALUES (2)");
    r4.close(testing.allocator);

    var r5 = try db.execSQL("INSERT INTO b (id, ref) VALUES (10, 1)");
    r5.close(testing.allocator);

    // SELECT from each table
    var ra = try db.execSQL("SELECT COUNT(*) FROM a");
    defer ra.close(testing.allocator);
    while (try ra.rows.?.next()) |*row_ptr| {
        var row = row_ptr.*;
        defer row.deinit();
        try testing.expectEqual(Value{ .integer = 2 }, row.values[0]);
    }

    var rb = try db.execSQL("SELECT COUNT(*) FROM b");
    defer rb.close(testing.allocator);
    while (try rb.rows.?.next()) |*row_ptr| {
        var row = row_ptr.*;
        defer row.deinit();
        try testing.expectEqual(Value{ .integer = 1 }, row.values[0]);
    }
}

test "transaction statement returns OK" {
    const path = "test_eng_txn_ok.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var db = try Database.open(testing.allocator, path, .{});
    defer db.close();

    var r1 = try db.execSQL("BEGIN");
    r1.close(testing.allocator);
    try testing.expect(std.mem.eql(u8, "BEGIN", r1.message));

    var r2 = try db.execSQL("COMMIT");
    r2.close(testing.allocator);
    try testing.expect(std.mem.eql(u8, "COMMIT", r2.message));
}

test "error: exec after error recovers" {
    const path = "test_eng_err_recover.db";
    defer std.fs.cwd().deleteFile(path) catch {};

    var db = try Database.open(testing.allocator, path, .{});
    defer db.close();

    // Trigger an error
    try testing.expectError(EngineError.ParseError, db.execSQL("INVALID SQL"));

    // Should be able to continue using the database
    var r1 = try db.execSQL("CREATE TABLE t (id INTEGER PRIMARY KEY)");
    r1.close(testing.allocator);

    var r2 = try db.execSQL("INSERT INTO t (id) VALUES (42)");
    r2.close(testing.allocator);
    try testing.expectEqual(@as(u64, 1), r2.rows_affected);

    var r3 = try db.execSQL("SELECT id FROM t");
    defer r3.close(testing.allocator);
    while (try r3.rows.?.next()) |*row_ptr| {
        var row = row_ptr.*;
        defer row.deinit();
        try testing.expectEqual(Value{ .integer = 42 }, row.values[0]);
    }
}

// ── MVCC Integration Tests ──────────────────────────────────────────

test "MVCC: BEGIN starts a transaction" {
    const path = "test_mvcc_begin.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    try testing.expect(db.current_txn == null);

    var r1 = try db.execSQL("BEGIN");
    r1.close(testing.allocator);
    try testing.expectEqualStrings("BEGIN", r1.message);
    try testing.expect(db.current_txn != null);

    var r2 = try db.execSQL("COMMIT");
    r2.close(testing.allocator);
    try testing.expectEqualStrings("COMMIT", r2.message);
    try testing.expect(db.current_txn == null);
}

test "MVCC: ROLLBACK aborts a transaction" {
    const path = "test_mvcc_rollback.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r1 = try db.execSQL("BEGIN");
    r1.close(testing.allocator);
    try testing.expect(db.current_txn != null);

    var r2 = try db.execSQL("ROLLBACK");
    r2.close(testing.allocator);
    try testing.expectEqualStrings("ROLLBACK", r2.message);
    try testing.expect(db.current_txn == null);
}

test "MVCC: double BEGIN returns error message" {
    const path = "test_mvcc_dbl_begin.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r1 = try db.execSQL("BEGIN");
    r1.close(testing.allocator);

    // Second BEGIN should return error message, not crash
    var r2 = try db.execSQL("BEGIN");
    r2.close(testing.allocator);
    try testing.expect(std.mem.startsWith(u8, r2.message, "ERROR"));

    // Transaction still active — clean up
    var r3 = try db.execSQL("COMMIT");
    r3.close(testing.allocator);
}

test "MVCC: COMMIT without BEGIN returns warning" {
    const path = "test_mvcc_no_begin_commit.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r1 = try db.execSQL("COMMIT");
    r1.close(testing.allocator);
    try testing.expect(std.mem.startsWith(u8, r1.message, "WARNING"));
}

test "MVCC: INSERT in transaction writes versioned rows" {
    const path = "test_mvcc_insert_ver.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r0 = try db.execSQL("CREATE TABLE items (id INTEGER, name TEXT)");
    r0.close(testing.allocator);

    // Start transaction and insert
    var r1 = try db.execSQL("BEGIN");
    r1.close(testing.allocator);

    var r2 = try db.execSQL("INSERT INTO items (id, name) VALUES (1, 'Alpha')");
    r2.close(testing.allocator);
    try testing.expectEqual(@as(u64, 1), r2.rows_affected);

    // SELECT within same transaction should see the row
    var r3 = try db.execSQL("SELECT id, name FROM items");
    var count: usize = 0;
    while (try r3.rows.?.next()) |*row_ptr| {
        var row = row_ptr.*;
        defer row.deinit();
        try testing.expectEqual(Value{ .integer = 1 }, row.values[0]);
        try testing.expectEqualStrings("Alpha", row.values[1].text);
        count += 1;
    }
    r3.close(testing.allocator);
    try testing.expectEqual(@as(usize, 1), count);

    var r4 = try db.execSQL("COMMIT");
    r4.close(testing.allocator);
}

test "MVCC: multiple inserts in transaction" {
    const path = "test_mvcc_multi_ins.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r0 = try db.execSQL("CREATE TABLE nums (id INTEGER, val INTEGER)");
    r0.close(testing.allocator);

    var r1 = try db.execSQL("BEGIN");
    r1.close(testing.allocator);

    var r2 = try db.execSQL("INSERT INTO nums (id, val) VALUES (1, 100)");
    r2.close(testing.allocator);
    var r3 = try db.execSQL("INSERT INTO nums (id, val) VALUES (2, 200)");
    r3.close(testing.allocator);
    var r4 = try db.execSQL("INSERT INTO nums (id, val) VALUES (3, 300)");
    r4.close(testing.allocator);

    // All three rows visible within transaction
    var r5 = try db.execSQL("SELECT id, val FROM nums");
    var count: usize = 0;
    while (try r5.rows.?.next()) |*row_ptr| {
        var row = row_ptr.*;
        defer row.deinit();
        count += 1;
    }
    r5.close(testing.allocator);
    try testing.expectEqual(@as(usize, 3), count);

    var r6 = try db.execSQL("COMMIT");
    r6.close(testing.allocator);

    // Should still be visible after commit
    var r7 = try db.execSQL("SELECT id FROM nums");
    count = 0;
    while (try r7.rows.?.next()) |*row_ptr| {
        var row = row_ptr.*;
        defer row.deinit();
        count += 1;
    }
    r7.close(testing.allocator);
    try testing.expectEqual(@as(usize, 3), count);
}

test "MVCC: UPDATE in transaction writes versioned rows" {
    const path = "test_mvcc_update_ver.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r0 = try db.execSQL("CREATE TABLE items (id INTEGER, name TEXT)");
    r0.close(testing.allocator);

    // Insert outside transaction (auto-commit)
    var r1 = try db.execSQL("INSERT INTO items (id, name) VALUES (1, 'before')");
    r1.close(testing.allocator);

    // Start transaction and update
    var r2 = try db.execSQL("BEGIN");
    r2.close(testing.allocator);

    var r3 = try db.execSQL("UPDATE items SET name = 'after' WHERE id = 1");
    r3.close(testing.allocator);
    try testing.expectEqual(@as(u64, 1), r3.rows_affected);

    // SELECT within transaction should see the updated value
    var r4 = try db.execSQL("SELECT id, name FROM items");
    var found = false;
    while (try r4.rows.?.next()) |*row_ptr| {
        var row = row_ptr.*;
        defer row.deinit();
        try testing.expectEqualStrings("after", row.values[1].text);
        found = true;
    }
    r4.close(testing.allocator);
    try testing.expect(found);

    var r5 = try db.execSQL("COMMIT");
    r5.close(testing.allocator);
}

test "MVCC: DELETE in transaction writes versioned rows" {
    const path = "test_mvcc_delete_ver.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r0 = try db.execSQL("CREATE TABLE items (id INTEGER, name TEXT)");
    r0.close(testing.allocator);

    // Insert outside transaction (auto-commit)
    var r1 = try db.execSQL("INSERT INTO items (id, name) VALUES (1, 'a'), (2, 'b')");
    r1.close(testing.allocator);

    // Start transaction and delete
    var r2 = try db.execSQL("BEGIN");
    r2.close(testing.allocator);

    var r3 = try db.execSQL("DELETE FROM items WHERE id = 1");
    r3.close(testing.allocator);
    try testing.expectEqual(@as(u64, 1), r3.rows_affected);

    var r4 = try db.execSQL("COMMIT");
    r4.close(testing.allocator);
}

test "MVCC: auto-commit mode still works" {
    const path = "test_mvcc_autocommit.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r0 = try db.execSQL("CREATE TABLE t (id INTEGER, val TEXT)");
    r0.close(testing.allocator);

    // Without BEGIN — auto-commit (no MVCC headers, plain row format)
    var r1 = try db.execSQL("INSERT INTO t (id, val) VALUES (1, 'hello')");
    r1.close(testing.allocator);

    var r2 = try db.execSQL("SELECT id, val FROM t");
    var count: usize = 0;
    while (try r2.rows.?.next()) |*row_ptr| {
        var row = row_ptr.*;
        defer row.deinit();
        try testing.expectEqual(Value{ .integer = 1 }, row.values[0]);
        try testing.expectEqualStrings("hello", row.values[1].text);
        count += 1;
    }
    r2.close(testing.allocator);
    try testing.expectEqual(@as(usize, 1), count);
}

test "MVCC: transaction manager XID assignment" {
    const path = "test_mvcc_xid.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    // First transaction should get XID 2 (FIRST_NORMAL_XID)
    try db.beginTransaction(.read_committed);
    try testing.expectEqual(@as(u32, mvcc_mod.FIRST_NORMAL_XID), db.current_txn.?.xid);
    try db.commitTransaction();

    // Second transaction should get XID 3
    try db.beginTransaction(.read_committed);
    try testing.expectEqual(@as(u32, mvcc_mod.FIRST_NORMAL_XID + 1), db.current_txn.?.xid);
    try db.commitTransaction();
}

test "MVCC: transaction context cleanup on close" {
    const path = "test_mvcc_cleanup.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);

    // Start transaction but don't commit — close should abort it
    var r1 = try db.execSQL("BEGIN");
    r1.close(testing.allocator);
    try testing.expect(db.current_txn != null);

    // close() should abort the transaction cleanly (no leaks)
    cleanupTestDb(&db, path);
}

test "MVCC: mixed auto-commit and explicit transaction" {
    const path = "test_mvcc_mixed.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    // Auto-commit insert
    var r0 = try db.execSQL("CREATE TABLE t (id INTEGER, val TEXT)");
    r0.close(testing.allocator);
    var r1 = try db.execSQL("INSERT INTO t (id, val) VALUES (1, 'auto')");
    r1.close(testing.allocator);

    // Explicit transaction insert
    var r2 = try db.execSQL("BEGIN");
    r2.close(testing.allocator);
    var r3 = try db.execSQL("INSERT INTO t (id, val) VALUES (2, 'explicit')");
    r3.close(testing.allocator);
    var r4 = try db.execSQL("COMMIT");
    r4.close(testing.allocator);

    // Both rows should be visible
    var r5 = try db.execSQL("SELECT id FROM t");
    var count: usize = 0;
    while (try r5.rows.?.next()) |*row_ptr| {
        var row = row_ptr.*;
        defer row.deinit();
        count += 1;
    }
    r5.close(testing.allocator);
    // The auto-commit row is in legacy format and the txn row is in MVCC format.
    // Both should be readable (backward-compatible).
    try testing.expectEqual(@as(usize, 2), count);
}

test "MVCC: ROLLBACK with no BEGIN returns warning" {
    const path = "test_mvcc_rollback_warn.db";
    defer std.fs.cwd().deleteFile(path) catch {};
    var db = try createTestDb(testing.allocator, path);
    defer cleanupTestDb(&db, path);

    var r1 = try db.execSQL("ROLLBACK");
    r1.close(testing.allocator);
    try testing.expect(std.mem.startsWith(u8, r1.message, "WARNING"));
}
