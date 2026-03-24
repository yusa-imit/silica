const std = @import("std");
const silica = @import("silica");
const harness = @import("harness");

/// TPC-C benchmark implementation
/// Reference: http://www.tpc.org/tpcc/spec/tpcc_current.pdf
///
/// TPC-C models a wholesale supplier with warehouses, districts, and customers.
/// It measures transaction throughput in new-order transactions per minute (tpmC).

const Allocator = std.mem.Allocator;
const Database = silica.engine.Database;
const Value = silica.engine.Value;

/// TPC-C configuration (scale factor = number of warehouses)
pub const Config = struct {
    warehouses: u32 = 1, // Scale factor (1 warehouse = ~100MB data)
    districts_per_warehouse: u32 = 10, // Fixed by spec
    customers_per_district: u32 = 3000, // Fixed by spec
    items: u32 = 100000, // Fixed by spec (100K items)
    orders_per_district: u32 = 3000, // Initial orders
};

/// Create TPC-C schema (9 tables)
pub fn createSchema(db: *Database, allocator: Allocator) !void {
    // WAREHOUSE table
    {
        const sql =
            \\CREATE TABLE warehouse (
            \\  w_id INTEGER PRIMARY KEY,
            \\  w_name TEXT,
            \\  w_street_1 TEXT,
            \\  w_street_2 TEXT,
            \\  w_city TEXT,
            \\  w_state TEXT,
            \\  w_zip TEXT,
            \\  w_tax REAL,
            \\  w_ytd REAL
            \\)
        ;
        var result = try db.exec(sql);
        result.close(allocator);
    }

    // DISTRICT table
    {
        const sql =
            \\CREATE TABLE district (
            \\  d_id INTEGER,
            \\  d_w_id INTEGER,
            \\  d_name TEXT,
            \\  d_street_1 TEXT,
            \\  d_street_2 TEXT,
            \\  d_city TEXT,
            \\  d_state TEXT,
            \\  d_zip TEXT,
            \\  d_tax REAL,
            \\  d_ytd REAL,
            \\  d_next_o_id INTEGER,
            \\  PRIMARY KEY (d_w_id, d_id)
            \\)
        ;
        var result = try db.exec(sql);
        result.close(allocator);
    }

    // CUSTOMER table
    {
        const sql =
            \\CREATE TABLE customer (
            \\  c_id INTEGER,
            \\  c_d_id INTEGER,
            \\  c_w_id INTEGER,
            \\  c_first TEXT,
            \\  c_middle TEXT,
            \\  c_last TEXT,
            \\  c_street_1 TEXT,
            \\  c_street_2 TEXT,
            \\  c_city TEXT,
            \\  c_state TEXT,
            \\  c_zip TEXT,
            \\  c_phone TEXT,
            \\  c_since TEXT,
            \\  c_credit TEXT,
            \\  c_credit_lim REAL,
            \\  c_discount REAL,
            \\  c_balance REAL,
            \\  c_ytd_payment REAL,
            \\  c_payment_cnt INTEGER,
            \\  c_delivery_cnt INTEGER,
            \\  c_data TEXT,
            \\  PRIMARY KEY (c_w_id, c_d_id, c_id)
            \\)
        ;
        var result = try db.exec(sql);
        result.close(allocator);
    }

    // NEW-ORDER table
    {
        const sql =
            \\CREATE TABLE new_order (
            \\  no_o_id INTEGER,
            \\  no_d_id INTEGER,
            \\  no_w_id INTEGER,
            \\  PRIMARY KEY (no_w_id, no_d_id, no_o_id)
            \\)
        ;
        var result = try db.exec(sql);
        result.close(allocator);
    }

    // ORDER table
    {
        const sql =
            \\CREATE TABLE orders (
            \\  o_id INTEGER,
            \\  o_d_id INTEGER,
            \\  o_w_id INTEGER,
            \\  o_c_id INTEGER,
            \\  o_entry_d TEXT,
            \\  o_carrier_id INTEGER,
            \\  o_ol_cnt INTEGER,
            \\  o_all_local INTEGER,
            \\  PRIMARY KEY (o_w_id, o_d_id, o_id)
            \\)
        ;
        var result = try db.exec(sql);
        result.close(allocator);
    }

    // ORDER-LINE table
    {
        const sql =
            \\CREATE TABLE order_line (
            \\  ol_o_id INTEGER,
            \\  ol_d_id INTEGER,
            \\  ol_w_id INTEGER,
            \\  ol_number INTEGER,
            \\  ol_i_id INTEGER,
            \\  ol_supply_w_id INTEGER,
            \\  ol_delivery_d TEXT,
            \\  ol_quantity INTEGER,
            \\  ol_amount REAL,
            \\  ol_dist_info TEXT,
            \\  PRIMARY KEY (ol_w_id, ol_d_id, ol_o_id, ol_number)
            \\)
        ;
        var result = try db.exec(sql);
        result.close(allocator);
    }

    // ITEM table
    {
        const sql =
            \\CREATE TABLE item (
            \\  i_id INTEGER PRIMARY KEY,
            \\  i_im_id INTEGER,
            \\  i_name TEXT,
            \\  i_price REAL,
            \\  i_data TEXT
            \\)
        ;
        var result = try db.exec(sql);
        result.close(allocator);
    }

    // STOCK table
    {
        const sql =
            \\CREATE TABLE stock (
            \\  s_i_id INTEGER,
            \\  s_w_id INTEGER,
            \\  s_quantity INTEGER,
            \\  s_dist_01 TEXT,
            \\  s_dist_02 TEXT,
            \\  s_dist_03 TEXT,
            \\  s_dist_04 TEXT,
            \\  s_dist_05 TEXT,
            \\  s_dist_06 TEXT,
            \\  s_dist_07 TEXT,
            \\  s_dist_08 TEXT,
            \\  s_dist_09 TEXT,
            \\  s_dist_10 TEXT,
            \\  s_ytd INTEGER,
            \\  s_order_cnt INTEGER,
            \\  s_remote_cnt INTEGER,
            \\  s_data TEXT,
            \\  PRIMARY KEY (s_w_id, s_i_id)
            \\)
        ;
        var result = try db.exec(sql);
        result.close(allocator);
    }

    // HISTORY table
    {
        const sql =
            \\CREATE TABLE history (
            \\  h_c_id INTEGER,
            \\  h_c_d_id INTEGER,
            \\  h_c_w_id INTEGER,
            \\  h_d_id INTEGER,
            \\  h_w_id INTEGER,
            \\  h_date TEXT,
            \\  h_amount REAL,
            \\  h_data TEXT
            \\)
        ;
        var result = try db.exec(sql);
        result.close(allocator);
    }

    // Create indices for performance
    {
        var result1 = try db.exec("CREATE INDEX idx_customer_last ON customer(c_w_id, c_d_id, c_last)");
        result1.close(allocator);
        var result2 = try db.exec("CREATE INDEX idx_orders_customer ON orders(o_w_id, o_d_id, o_c_id)");
        result2.close(allocator);
    }
}

/// Pseudo-random number generator for TPC-C data generation
/// Uses simple LCG for reproducibility
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
        const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
        var i: usize = 0;
        while (i < len and i < buf.len) : (i += 1) {
            buf[i] = chars[self.range(0, chars.len - 1)];
        }
        return buf[0..@min(len, buf.len)];
    }

    pub fn lastName(self: *Random, buf: []u8) []const u8 {
        const syllables = [_][]const u8{ "BAR", "OUGHT", "ABLE", "PRI", "PRES", "ESE", "ANTI", "CALLY", "ATION", "EING" };
        const idx1 = self.range(0, syllables.len - 1);
        const idx2 = self.range(0, syllables.len - 1);
        const idx3 = self.range(0, syllables.len - 1);
        return std.fmt.bufPrint(buf, "{s}{s}{s}", .{ syllables[idx1], syllables[idx2], syllables[idx3] }) catch "UNKNOWN";
    }
};

/// Load initial data into TPC-C tables
pub fn loadData(db: *Database, allocator: Allocator, config: Config) !void {
    var rng = Random.init(12345); // Fixed seed for reproducibility
    var buf: [256]u8 = undefined;

    // Load ITEMs (100K items, independent of warehouse count)
    std.debug.print("Loading {d} items...\n", .{config.items});
    var i_id: u32 = 1;
    while (i_id <= config.items) : (i_id += 1) {
        const i_name = rng.text(&buf, 24);
        const i_price = @as(f64, @floatFromInt(rng.range(100, 10000))) / 100.0;
        const i_data = rng.text(&buf, 50);

        const sql = std.fmt.bufPrint(&buf,
            "INSERT INTO item VALUES ({d}, {d}, '{s}', {d:.2}, '{s}')",
            .{ i_id, rng.range(1, 10000), i_name, i_price, i_data }
        ) catch continue;
        var result = try db.exec(sql);
        result.close(allocator);
    }

    // Load WAREHOUSEs, DISTRICTs, CUSTOMERs, etc.
    std.debug.print("Loading {d} warehouses...\n", .{config.warehouses});
    var w_id: u32 = 1;
    while (w_id <= config.warehouses) : (w_id += 1) {
        // WAREHOUSE
        const w_name = rng.text(&buf, 10);
        const w_tax = @as(f64, @floatFromInt(rng.range(0, 2000))) / 10000.0;
        const w_ytd = 300000.00;

        const w_sql = std.fmt.bufPrint(&buf,
            "INSERT INTO warehouse VALUES ({d}, '{s}', 'St1', 'St2', 'City', 'ST', '12345', {d:.4}, {d:.2})",
            .{ w_id, w_name, w_tax, w_ytd }
        ) catch continue;
        var w_result = try db.exec(w_sql);
        w_result.close(allocator);

        // Load STOCK for this warehouse (100K items)
        var s_i_id: u32 = 1;
        while (s_i_id <= config.items) : (s_i_id += 1) {
            const s_quantity = rng.range(10, 100);
            const s_data = rng.text(&buf, 50);

            const s_sql = std.fmt.bufPrint(&buf,
                "INSERT INTO stock VALUES ({d}, {d}, {d}, 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 0, 0, 0, '{s}')",
                .{ s_i_id, w_id, s_quantity, s_data }
            ) catch continue;
            var s_result = try db.exec(s_sql);
            s_result.close(allocator);
        }

        // Load DISTRICTs
        var d_id: u32 = 1;
        while (d_id <= config.districts_per_warehouse) : (d_id += 1) {
            const d_name = rng.text(&buf, 10);
            const d_tax = @as(f64, @floatFromInt(rng.range(0, 2000))) / 10000.0;
            const d_ytd = 30000.00;

            const d_sql = std.fmt.bufPrint(&buf,
                "INSERT INTO district VALUES ({d}, {d}, '{s}', 'St1', 'St2', 'City', 'ST', '12345', {d:.4}, {d:.2}, 3001)",
                .{ d_id, w_id, d_name, d_tax, d_ytd }
            ) catch continue;
            var d_result = try db.exec(d_sql);
            d_result.close(allocator);

            // Load CUSTOMERs for this district
            var c_id: u32 = 1;
            while (c_id <= config.customers_per_district) : (c_id += 1) {
                const c_first = rng.text(&buf, 16);
                const c_last = rng.lastName(&buf);
                const c_credit = if (rng.range(0, 9) == 0) "BC" else "GC";
                const c_discount = @as(f64, @floatFromInt(rng.range(0, 5000))) / 10000.0;

                const c_sql = std.fmt.bufPrint(&buf,
                    "INSERT INTO customer VALUES ({d}, {d}, {d}, '{s}', 'OE', '{s}', 'St1', 'St2', 'City', 'ST', '12345', '1234567890', '2024-01-01', '{s}', 50000.00, {d:.4}, -10.00, 10.00, 1, 0, 'Data')",
                    .{ c_id, d_id, w_id, c_first, c_last, c_credit, c_discount }
                ) catch continue;
                var c_result = try db.exec(c_sql);
                c_result.close(allocator);
            }
        }
    }

    std.debug.print("Data loading complete.\n", .{});
}

/// TPC-C New-Order transaction (most common, ~45% of mix)
pub fn newOrderTransaction(db: *Database, allocator: Allocator, w_id: u32, d_id: u32, c_id: u32) !void {
    // 1. Get next order ID from district
    var buf: [512]u8 = undefined;
    const get_next_o_id_sql = try std.fmt.bufPrint(&buf,
        "SELECT d_next_o_id FROM district WHERE d_w_id = {d} AND d_id = {d}",
        .{ w_id, d_id }
    );
    var result1 = try db.exec(get_next_o_id_sql);
    defer result1.close(allocator);

    var o_id: i64 = 3001; // Default if not found
    if (result1.rows) |*rows| {
        if (try rows.next()) |*row_ptr| {
            var row = row_ptr.*;
            defer row.deinit();
            o_id = row.values[0].integer;
        }
    }

    // 2. Increment next_o_id
    const update_district_sql = try std.fmt.bufPrint(&buf,
        "UPDATE district SET d_next_o_id = {d} WHERE d_w_id = {d} AND d_id = {d}",
        .{ o_id + 1, w_id, d_id }
    );
    var result2 = try db.exec(update_district_sql);
    result2.close(allocator);

    // 3. Insert new order
    const insert_order_sql = try std.fmt.bufPrint(&buf,
        "INSERT INTO orders VALUES ({d}, {d}, {d}, {d}, '2024-01-01', NULL, 5, 1)",
        .{ o_id, d_id, w_id, c_id }
    );
    var result3 = try db.exec(insert_order_sql);
    result3.close(allocator);

    // 4. Insert new_order entry
    const insert_new_order_sql = try std.fmt.bufPrint(&buf,
        "INSERT INTO new_order VALUES ({d}, {d}, {d})",
        .{ o_id, d_id, w_id }
    );
    var result4 = try db.exec(insert_new_order_sql);
    result4.close(allocator);

    // 5. Insert 5 order lines (simplified)
    var ol_number: u32 = 1;
    while (ol_number <= 5) : (ol_number += 1) {
        const i_id = (ol_number * 1000) % 100000 + 1; // Random item
        const insert_ol_sql = try std.fmt.bufPrint(&buf,
            "INSERT INTO order_line VALUES ({d}, {d}, {d}, {d}, {d}, {d}, NULL, 5, 100.0, 'Info')",
            .{ o_id, d_id, w_id, ol_number, i_id, w_id }
        );
        var result5 = try db.exec(insert_ol_sql);
        result5.close(allocator);
    }
}

/// TPC-C Payment transaction (~43% of mix)
pub fn paymentTransaction(db: *Database, allocator: Allocator, w_id: u32, d_id: u32, c_id: u32, h_amount: f64) !void {
    var buf: [512]u8 = undefined;

    // 1. Update warehouse YTD
    const update_warehouse_sql = try std.fmt.bufPrint(&buf,
        "UPDATE warehouse SET w_ytd = w_ytd + {d:.2} WHERE w_id = {d}",
        .{ h_amount, w_id }
    );
    var result1 = try db.exec(update_warehouse_sql);
    result1.close(allocator);

    // 2. Update district YTD
    const update_district_sql = try std.fmt.bufPrint(&buf,
        "UPDATE district SET d_ytd = d_ytd + {d:.2} WHERE d_w_id = {d} AND d_id = {d}",
        .{ h_amount, w_id, d_id }
    );
    var result2 = try db.exec(update_district_sql);
    result2.close(allocator);

    // 3. Update customer balance
    const update_customer_sql = try std.fmt.bufPrint(&buf,
        "UPDATE customer SET c_balance = c_balance - {d:.2}, c_ytd_payment = c_ytd_payment + {d:.2}, c_payment_cnt = c_payment_cnt + 1 WHERE c_w_id = {d} AND c_d_id = {d} AND c_id = {d}",
        .{ h_amount, h_amount, w_id, d_id, c_id }
    );
    var result3 = try db.exec(update_customer_sql);
    result3.close(allocator);

    // 4. Insert history record
    const insert_history_sql = try std.fmt.bufPrint(&buf,
        "INSERT INTO history VALUES ({d}, {d}, {d}, {d}, {d}, '2024-01-01', {d:.2}, 'Payment')",
        .{ c_id, d_id, w_id, d_id, w_id, h_amount }
    );
    var result4 = try db.exec(insert_history_sql);
    result4.close(allocator);
}

/// Run TPC-C benchmark
pub fn runBenchmark(allocator: Allocator, config: Config, duration_sec: u64) !void {
    std.debug.print("=== TPC-C Benchmark ===\n", .{});
    std.debug.print("Configuration: {d} warehouses\n", .{config.warehouses});

    // Clean up old test database
    std.fs.cwd().deleteFile("tpcc_bench.db") catch {};
    std.fs.cwd().deleteFile("tpcc_bench.db-wal") catch {};

    // Create database and schema
    var db = try Database.open(allocator, "tpcc_bench.db", .{ .wal_mode = true });
    defer {
        db.close();
        std.fs.cwd().deleteFile("tpcc_bench.db") catch {};
        std.fs.cwd().deleteFile("tpcc_bench.db-wal") catch {};
    }

    std.debug.print("\nCreating TPC-C schema...\n", .{});
    try createSchema(&db, allocator);

    std.debug.print("Loading initial data...\n", .{});
    try loadData(&db, allocator, config);

    // Run benchmark for specified duration
    std.debug.print("\nRunning transactions for {d} seconds...\n", .{duration_sec});
    const start = std.time.milliTimestamp();
    var tx_count: u64 = 0;
    var rng = Random.init(@intCast(std.time.milliTimestamp()));

    while (true) {
        const now = std.time.milliTimestamp();
        if (now - start >= duration_sec * 1000) break;

        const w_id: u32 = rng.range(1, config.warehouses);
        const d_id: u32 = rng.range(1, config.districts_per_warehouse);
        const c_id: u32 = rng.range(1, config.customers_per_district);

        // Transaction mix: 45% new-order, 43% payment, 12% other (not implemented)
        const tx_type = rng.range(0, 99);
        if (tx_type < 45) {
            try newOrderTransaction(&db, allocator, w_id, d_id, c_id);
        } else if (tx_type < 88) {
            const amount = @as(f64, @floatFromInt(rng.range(100, 500000))) / 100.0;
            try paymentTransaction(&db, allocator, w_id, d_id, c_id, amount);
        }

        tx_count += 1;
    }

    const end = std.time.milliTimestamp();
    const elapsed_sec = @as(f64, @floatFromInt(end - start)) / 1000.0;
    const tpmC = (@as(f64, @floatFromInt(tx_count)) / elapsed_sec) * 60.0;

    std.debug.print("\n=== Results ===\n", .{});
    std.debug.print("Total transactions: {d}\n", .{tx_count});
    std.debug.print("Duration: {d:.2} seconds\n", .{elapsed_sec});
    std.debug.print("Throughput: {d:.0} tpmC (transactions per minute)\n", .{tpmC});
    std.debug.print("Average latency: {d:.2} ms\n", .{(elapsed_sec * 1000.0) / @as(f64, @floatFromInt(tx_count))});
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Lightweight configuration for quick testing
    const config = Config{
        .warehouses = 1,
        .districts_per_warehouse = 2, // Reduced from 10
        .customers_per_district = 100, // Reduced from 3000
        .items = 1000, // Reduced from 100K
        .orders_per_district = 0,
    };

    try runBenchmark(allocator, config, 10); // 10 second run
}
