const std = @import("std");
const silica = @import("silica");
const harness = @import("harness");

/// TPC-H benchmark implementation
/// Reference: http://www.tpc.org/tpch/spec/tpch2.18.0.pdf
///
/// TPC-H models a wholesale supplier decision support system with 8 tables
/// and 22 complex analytical queries (joins, aggregates, sorting).

const Allocator = std.mem.Allocator;
const Database = silica.engine.Database;
const Value = silica.engine.Value;

/// TPC-H configuration (scale factor = GB of raw data)
pub const Config = struct {
    scale_factor: u32 = 1, // 1 = 1GB, 10 = 10GB, 100 = 100GB (TPC-H spec)
    // Derived row counts (for SF=1):
    // PART: 200K, SUPPLIER: 10K, PARTSUPP: 800K, CUSTOMER: 150K, ORDERS: 1.5M, LINEITEM: 6M
    // Reduced for quick testing:
    parts: u32 = 1000, // ~200K for SF=1
    suppliers: u32 = 100, // ~10K for SF=1
    customers: u32 = 1500, // ~150K for SF=1
    orders: u32 = 1500, // ~1.5M for SF=1
    lineitems_per_order: u32 = 4, // Average ~4 line items per order
};

/// Create TPC-H schema (8 tables)
pub fn createSchema(db: *Database, allocator: Allocator) !void {
    // PART table
    {
        const sql =
            \\CREATE TABLE part (
            \\  p_partkey INTEGER PRIMARY KEY,
            \\  p_name TEXT,
            \\  p_mfgr TEXT,
            \\  p_brand TEXT,
            \\  p_type TEXT,
            \\  p_size INTEGER,
            \\  p_container TEXT,
            \\  p_retailprice REAL,
            \\  p_comment TEXT
            \\)
        ;
        var result = try db.exec(sql);
        result.close(allocator);
    }

    // SUPPLIER table
    {
        const sql =
            \\CREATE TABLE supplier (
            \\  s_suppkey INTEGER PRIMARY KEY,
            \\  s_name TEXT,
            \\  s_address TEXT,
            \\  s_nationkey INTEGER,
            \\  s_phone TEXT,
            \\  s_acctbal REAL,
            \\  s_comment TEXT
            \\)
        ;
        var result = try db.exec(sql);
        result.close(allocator);
    }

    // PARTSUPP table (part-supplier relationship)
    {
        const sql =
            \\CREATE TABLE partsupp (
            \\  ps_partkey INTEGER,
            \\  ps_suppkey INTEGER,
            \\  ps_availqty INTEGER,
            \\  ps_supplycost REAL,
            \\  ps_comment TEXT,
            \\  PRIMARY KEY (ps_partkey, ps_suppkey)
            \\)
        ;
        var result = try db.exec(sql);
        result.close(allocator);
    }

    // CUSTOMER table
    {
        const sql =
            \\CREATE TABLE customer (
            \\  c_custkey INTEGER PRIMARY KEY,
            \\  c_name TEXT,
            \\  c_address TEXT,
            \\  c_nationkey INTEGER,
            \\  c_phone TEXT,
            \\  c_acctbal REAL,
            \\  c_mktsegment TEXT,
            \\  c_comment TEXT
            \\)
        ;
        var result = try db.exec(sql);
        result.close(allocator);
    }

    // ORDERS table
    {
        const sql =
            \\CREATE TABLE orders (
            \\  o_orderkey INTEGER PRIMARY KEY,
            \\  o_custkey INTEGER,
            \\  o_orderstatus TEXT,
            \\  o_totalprice REAL,
            \\  o_orderdate TEXT,
            \\  o_orderpriority TEXT,
            \\  o_clerk TEXT,
            \\  o_shippriority INTEGER,
            \\  o_comment TEXT
            \\)
        ;
        var result = try db.exec(sql);
        result.close(allocator);
    }

    // LINEITEM table (order line items)
    {
        const sql =
            \\CREATE TABLE lineitem (
            \\  l_orderkey INTEGER,
            \\  l_partkey INTEGER,
            \\  l_suppkey INTEGER,
            \\  l_linenumber INTEGER,
            \\  l_quantity REAL,
            \\  l_extendedprice REAL,
            \\  l_discount REAL,
            \\  l_tax REAL,
            \\  l_returnflag TEXT,
            \\  l_linestatus TEXT,
            \\  l_shipdate TEXT,
            \\  l_commitdate TEXT,
            \\  l_receiptdate TEXT,
            \\  l_shipinstruct TEXT,
            \\  l_shipmode TEXT,
            \\  l_comment TEXT,
            \\  PRIMARY KEY (l_orderkey, l_linenumber)
            \\)
        ;
        var result = try db.exec(sql);
        result.close(allocator);
    }

    // NATION table (25 rows, fixed)
    {
        const sql =
            \\CREATE TABLE nation (
            \\  n_nationkey INTEGER PRIMARY KEY,
            \\  n_name TEXT,
            \\  n_regionkey INTEGER,
            \\  n_comment TEXT
            \\)
        ;
        var result = try db.exec(sql);
        result.close(allocator);
    }

    // REGION table (5 rows, fixed)
    {
        const sql =
            \\CREATE TABLE region (
            \\  r_regionkey INTEGER PRIMARY KEY,
            \\  r_name TEXT,
            \\  r_comment TEXT
            \\)
        ;
        var result = try db.exec(sql);
        result.close(allocator);
    }

    // Create indices for query performance
    {
        var r1 = try db.exec("CREATE INDEX idx_lineitem_shipdate ON lineitem(l_shipdate)");
        r1.close(allocator);
        var r2 = try db.exec("CREATE INDEX idx_orders_orderdate ON orders(o_orderdate)");
        r2.close(allocator);
        var r3 = try db.exec("CREATE INDEX idx_customer_mktsegment ON customer(c_mktsegment)");
        r3.close(allocator);
    }
}

/// Pseudo-random number generator for TPC-H data generation
const Random = struct {
    state: u64,

    pub fn init(seed: u64) Random {
        return .{ .state = seed };
    }

    pub fn next(self: *Random) u32 {
        self.state = (self.state *% 1103515245 +% 12345) & 0x7fffffff;
        return @intCast(self.state);
    }

    pub fn range(self: *Random, min: u32, max: u32) u32 {
        const n = self.next();
        return min + (n % (max - min + 1));
    }

    pub fn text(self: *Random, buf: []u8, len: usize) []const u8 {
        const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 ";
        var i: usize = 0;
        while (i < len and i < buf.len) : (i += 1) {
            buf[i] = chars[self.range(0, chars.len - 1)];
        }
        return buf[0..@min(len, buf.len)];
    }

    pub fn date(self: *Random, buf: []u8) []const u8 {
        const year = self.range(1992, 1998);
        const month = self.range(1, 12);
        const day = self.range(1, 28);
        return std.fmt.bufPrint(buf, "{d:0>4}-{d:0>2}-{d:0>2}", .{ year, month, day }) catch "1995-01-01";
    }
};

/// Load TPC-H data
pub fn loadData(db: *Database, allocator: Allocator, config: Config) !void {
    var rng = Random.init(54321);
    var buf: [256]u8 = undefined;

    // Load REGION (5 rows, fixed)
    std.debug.print("Loading regions...\n", .{});
    const regions = [_][]const u8{ "AFRICA", "AMERICA", "ASIA", "EUROPE", "MIDDLE EAST" };
    for (regions, 0..) |name, i| {
        const sql = try std.fmt.bufPrint(&buf,
            "INSERT INTO region VALUES ({d}, '{s}', 'Comment')",
            .{ i, name }
        );
        var result = try db.exec(sql);
        result.close(allocator);
    }

    // Load NATION (25 rows)
    std.debug.print("Loading nations...\n", .{});
    var n_key: u32 = 0;
    while (n_key < 25) : (n_key += 1) {
        const r_key = n_key % 5;
        const sql = try std.fmt.bufPrint(&buf,
            "INSERT INTO nation VALUES ({d}, 'NATION_{d}', {d}, 'Comment')",
            .{ n_key, n_key, r_key }
        );
        var result = try db.exec(sql);
        result.close(allocator);
    }

    // Load SUPPLIER
    std.debug.print("Loading {d} suppliers...\n", .{config.suppliers});
    var s_key: u32 = 1;
    while (s_key <= config.suppliers) : (s_key += 1) {
        const s_name = try std.fmt.bufPrint(&buf, "Supplier#{d:0>9}", .{s_key});
        const s_nationkey = rng.range(0, 24);
        const s_acctbal = @as(f64, @floatFromInt(rng.range(0, 1000000))) / 100.0;

        const sql = try std.fmt.bufPrint(&buf,
            "INSERT INTO supplier VALUES ({d}, '{s}', 'Address', {d}, '123-456-7890', {d:.2}, 'Comment')",
            .{ s_key, s_name, s_nationkey, s_acctbal }
        );
        var result = try db.exec(sql);
        result.close(allocator);
    }

    // Load PART
    std.debug.print("Loading {d} parts...\n", .{config.parts});
    var p_key: u32 = 1;
    while (p_key <= config.parts) : (p_key += 1) {
        const p_name = try std.fmt.bufPrint(&buf, "Part#{d:0>9}", .{p_key});
        const p_size = rng.range(1, 50);
        const p_retailprice = @as(f64, @floatFromInt(rng.range(100, 200000))) / 100.0;

        const sql = try std.fmt.bufPrint(&buf,
            "INSERT INTO part VALUES ({d}, '{s}', 'Manufacturer', 'Brand', 'Type', {d}, 'Container', {d:.2}, 'Comment')",
            .{ p_key, p_name, p_size, p_retailprice }
        );
        var result = try db.exec(sql);
        result.close(allocator);
    }

    // Load PARTSUPP (4 suppliers per part)
    std.debug.print("Loading partsupp relationships...\n", .{});
    p_key = 1;
    while (p_key <= config.parts) : (p_key += 1) {
        var supp_idx: u32 = 0;
        while (supp_idx < 4 and supp_idx < config.suppliers) : (supp_idx += 1) {
            const ps_suppkey = (s_key + supp_idx) % config.suppliers + 1;
            const ps_availqty = rng.range(1, 9999);
            const ps_supplycost = @as(f64, @floatFromInt(rng.range(100, 100000))) / 100.0;

            const sql = try std.fmt.bufPrint(&buf,
                "INSERT INTO partsupp VALUES ({d}, {d}, {d}, {d:.2}, 'Comment')",
                .{ p_key, ps_suppkey, ps_availqty, ps_supplycost }
            );
            var result = try db.exec(sql);
            result.close(allocator);
        }
    }

    // Load CUSTOMER
    std.debug.print("Loading {d} customers...\n", .{config.customers});
    var c_key: u32 = 1;
    while (c_key <= config.customers) : (c_key += 1) {
        const c_name = try std.fmt.bufPrint(&buf, "Customer#{d:0>9}", .{c_key});
        const c_nationkey = rng.range(0, 24);
        // Account balance can be negative (subtract 1000 to shift range)
        const c_acctbal = (@as(f64, @floatFromInt(rng.range(0, 1100000))) / 100.0) - 1000.0;
        const segments = [_][]const u8{ "AUTOMOBILE", "BUILDING", "FURNITURE", "MACHINERY", "HOUSEHOLD" };
        const c_mktsegment = segments[rng.range(0, 4)];

        const sql = try std.fmt.bufPrint(&buf,
            "INSERT INTO customer VALUES ({d}, '{s}', 'Address', {d}, '123-456-7890', {d:.2}, '{s}', 'Comment')",
            .{ c_key, c_name, c_nationkey, c_acctbal, c_mktsegment }
        );
        var result = try db.exec(sql);
        result.close(allocator);
    }

    // Load ORDERS
    std.debug.print("Loading {d} orders...\n", .{config.orders});
    var o_key: u32 = 1;
    while (o_key <= config.orders) : (o_key += 1) {
        const o_custkey = rng.range(1, config.customers);
        const statuses = [_][]const u8{ "O", "F", "P" };
        const o_orderstatus = statuses[rng.range(0, 2)];
        const o_totalprice = @as(f64, @floatFromInt(rng.range(100000, 50000000))) / 100.0;
        const o_orderdate = rng.date(&buf);
        const priorities = [_][]const u8{ "1-URGENT", "2-HIGH", "3-MEDIUM", "4-NOT SPECIFIED", "5-LOW" };
        const o_orderpriority = priorities[rng.range(0, 4)];

        const sql = try std.fmt.bufPrint(&buf,
            "INSERT INTO orders VALUES ({d}, {d}, '{s}', {d:.2}, '{s}', '{s}', 'Clerk#000000001', 0, 'Comment')",
            .{ o_key, o_custkey, o_orderstatus, o_totalprice, o_orderdate, o_orderpriority }
        );
        var result = try db.exec(sql);
        result.close(allocator);
    }

    // Load LINEITEM
    std.debug.print("Loading lineitems...\n", .{});
    o_key = 1;
    while (o_key <= config.orders) : (o_key += 1) {
        const num_lines = rng.range(1, config.lineitems_per_order);
        var l_num: u32 = 1;
        while (l_num <= num_lines) : (l_num += 1) {
            const l_partkey = rng.range(1, config.parts);
            const l_suppkey = rng.range(1, config.suppliers);
            const l_quantity = @as(f64, @floatFromInt(rng.range(1, 50)));
            const l_extendedprice = @as(f64, @floatFromInt(rng.range(100, 20000))) / 100.0;
            const l_discount = @as(f64, @floatFromInt(rng.range(0, 10))) / 100.0;
            const l_tax = @as(f64, @floatFromInt(rng.range(0, 8))) / 100.0;
            const flags = [_][]const u8{ "A", "N", "R" };
            const l_returnflag = flags[rng.range(0, 2)];
            const l_linestatus = flags[rng.range(0, 1)];
            const l_shipdate = rng.date(&buf);

            const sql = try std.fmt.bufPrint(&buf,
                "INSERT INTO lineitem VALUES ({d}, {d}, {d}, {d}, {d:.0}, {d:.2}, {d:.2}, {d:.2}, '{s}', '{s}', '{s}', '{s}', '{s}', 'DELIVER IN PERSON', 'TRUCK', 'Comment')",
                .{ o_key, l_partkey, l_suppkey, l_num, l_quantity, l_extendedprice, l_discount, l_tax, l_returnflag, l_linestatus, l_shipdate, l_shipdate, l_shipdate }
            );
            var result = try db.exec(sql);
            result.close(allocator);
        }
    }

    std.debug.print("Data loading complete.\n", .{});
}

/// TPC-H Query 1 — Pricing Summary Report
pub fn query1(db: *Database, allocator: Allocator) !harness.BenchmarkResult {
    const sql =
        \\SELECT
        \\  l_returnflag,
        \\  l_linestatus,
        \\  COUNT(*) as count_order,
        \\  SUM(l_quantity) as sum_qty,
        \\  SUM(l_extendedprice) as sum_base_price,
        \\  AVG(l_quantity) as avg_qty,
        \\  AVG(l_extendedprice) as avg_price
        \\FROM lineitem
        \\WHERE l_shipdate <= '1998-12-01'
        \\GROUP BY l_returnflag, l_linestatus
        \\ORDER BY l_returnflag, l_linestatus
    ;

    const start = std.time.nanoTimestamp();
    var result = try db.exec(sql);
    defer result.close(allocator);

    // Consume result
    var count: usize = 0;
    if (result.rows) |*rows| {
        while (try rows.next()) |*row_ptr| {
            var row = row_ptr.*;
            defer row.deinit();
            count += 1;
        }
    }
    const end = std.time.nanoTimestamp();

    const elapsed_ns: u64 = @intCast(end - start);
    return harness.BenchmarkResult{
        .name = "TPC-H Q1: Pricing Summary Report",
        .iterations = 1,
        .total_ns = elapsed_ns,
        .mean_ns = elapsed_ns,
        .min_ns = elapsed_ns,
        .max_ns = elapsed_ns,
        .stddev_ns = 0,
        .ops_per_sec = 1_000_000_000.0 / @as(f64, @floatFromInt(elapsed_ns)),
    };
}

/// TPC-H Query 3 — Shipping Priority Query
pub fn query3(db: *Database, allocator: Allocator) !harness.BenchmarkResult {
    const sql =
        \\SELECT
        \\  l_orderkey,
        \\  SUM(l_extendedprice) as revenue,
        \\  o_orderdate,
        \\  o_shippriority
        \\FROM customer, orders, lineitem
        \\WHERE c_mktsegment = 'BUILDING'
        \\  AND c_custkey = o_custkey
        \\  AND l_orderkey = o_orderkey
        \\  AND o_orderdate < '1995-03-15'
        \\  AND l_shipdate > '1995-03-15'
        \\GROUP BY l_orderkey, o_orderdate, o_shippriority
        \\ORDER BY revenue DESC, o_orderdate
        \\LIMIT 10
    ;

    const start = std.time.nanoTimestamp();
    var result = try db.exec(sql);
    defer result.close(allocator);

    var count: usize = 0;
    if (result.rows) |*rows| {
        while (try rows.next()) |*row_ptr| {
            var row = row_ptr.*;
            defer row.deinit();
            count += 1;
        }
    }
    const end = std.time.nanoTimestamp();

    const elapsed_ns: u64 = @intCast(end - start);
    return harness.BenchmarkResult{
        .name = "TPC-H Q3: Shipping Priority",
        .iterations = 1,
        .total_ns = elapsed_ns,
        .mean_ns = elapsed_ns,
        .min_ns = elapsed_ns,
        .max_ns = elapsed_ns,
        .stddev_ns = 0,
        .ops_per_sec = 1_000_000_000.0 / @as(f64, @floatFromInt(elapsed_ns)),
    };
}

/// TPC-H Query 6 — Forecasting Revenue Change Query
pub fn query6(db: *Database, allocator: Allocator) !harness.BenchmarkResult {
    const sql =
        \\SELECT SUM(l_extendedprice * l_discount) as revenue
        \\FROM lineitem
        \\WHERE l_shipdate >= '1994-01-01'
        \\  AND l_shipdate < '1995-01-01'
        \\  AND l_discount >= 0.05
        \\  AND l_discount <= 0.07
        \\  AND l_quantity < 24
    ;

    const start = std.time.nanoTimestamp();
    var result = try db.exec(sql);
    defer result.close(allocator);

    var count: usize = 0;
    if (result.rows) |*rows| {
        while (try rows.next()) |*row_ptr| {
            var row = row_ptr.*;
            defer row.deinit();
            count += 1;
        }
    }
    const end = std.time.nanoTimestamp();

    const elapsed_ns: u64 = @intCast(end - start);
    return harness.BenchmarkResult{
        .name = "TPC-H Q6: Forecasting Revenue Change",
        .iterations = 1,
        .total_ns = elapsed_ns,
        .mean_ns = elapsed_ns,
        .min_ns = elapsed_ns,
        .max_ns = elapsed_ns,
        .stddev_ns = 0,
        .ops_per_sec = 1_000_000_000.0 / @as(f64, @floatFromInt(elapsed_ns)),
    };
}

/// Run TPC-H benchmark
pub fn runBenchmark(allocator: Allocator, config: Config) !void {
    std.debug.print("=== TPC-H Benchmark ===\n", .{});
    std.debug.print("Configuration: SF={d} (lightweight mode)\n", .{config.scale_factor});

    // Clean up old test database
    std.fs.cwd().deleteFile("tpch_bench.db") catch {};
    std.fs.cwd().deleteFile("tpch_bench.db-wal") catch {};

    // Create database and schema
    var db = try Database.open(allocator, "tpch_bench.db", .{ .wal_mode = true });
    defer {
        db.close();
        std.fs.cwd().deleteFile("tpch_bench.db") catch {};
        std.fs.cwd().deleteFile("tpch_bench.db-wal") catch {};
    }

    std.debug.print("\nCreating TPC-H schema...\n", .{});
    try createSchema(&db, allocator);

    std.debug.print("Loading initial data...\n", .{});
    try loadData(&db, allocator, config);

    // Run queries
    std.debug.print("\n=== Running Queries ===\n", .{});

    const q1_result = try query1(&db, allocator);
    std.debug.print("Q1: {d:.2} ms ({d:.0} ops/sec)\n", .{
        @as(f64, @floatFromInt(q1_result.mean_ns)) / 1_000_000.0,
        q1_result.ops_per_sec,
    });

    const q3_result = try query3(&db, allocator);
    std.debug.print("Q3: {d:.2} ms ({d:.0} ops/sec)\n", .{
        @as(f64, @floatFromInt(q3_result.mean_ns)) / 1_000_000.0,
        q3_result.ops_per_sec,
    });

    const q6_result = try query6(&db, allocator);
    std.debug.print("Q6: {d:.2} ms ({d:.0} ops/sec)\n", .{
        @as(f64, @floatFromInt(q6_result.mean_ns)) / 1_000_000.0,
        q6_result.ops_per_sec,
    });

    std.debug.print("\nTPC-H benchmark complete!\n", .{});
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Lightweight configuration for quick testing
    const config = Config{
        .scale_factor = 1,
        .parts = 1000,
        .suppliers = 100,
        .customers = 1500,
        .orders = 1500,
        .lineitems_per_order = 4,
    };

    try runBenchmark(allocator, config);
}
