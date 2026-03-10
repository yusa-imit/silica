const std = @import("std");
const net = std.net;
const os = std.os;
const Allocator = std.mem.Allocator;
const silica = @import("silica");
const Database = silica.engine.Database;
const Connection = @import("connection.zig").Connection;
const wire = @import("wire.zig");

/// Server represents a TCP server that accepts PostgreSQL wire protocol connections
pub const Server = struct {
    allocator: Allocator,
    address: net.Address,
    listener: net.Server,
    database: *Database,
    running: bool,
    max_connections: usize,
    active_connections: std.atomic.Value(usize),

    const Self = @This();

    /// ServerConfig holds server configuration
    pub const Config = struct {
        host: []const u8 = "127.0.0.1",
        port: u16 = 5433,
        max_connections: usize = 100,
        database_path: []const u8,
    };

    /// Initialize a new server with the given configuration
    pub fn init(allocator: Allocator, config: Config) !Self {
        // Parse the address
        const address = try net.Address.parseIp(config.host, config.port);

        // Open the database
        const database = try allocator.create(Database);
        errdefer allocator.destroy(database);
        database.* = try Database.open(allocator, config.database_path, .{});

        // Create the TCP listener
        const listener = try address.listen(.{
            .reuse_address = true,
            .reuse_port = false,
        });

        return Self{
            .allocator = allocator,
            .address = address,
            .listener = listener,
            .database = database,
            .running = false,
            .max_connections = config.max_connections,
            .active_connections = std.atomic.Value(usize).init(0),
        };
    }

    /// Deinitialize the server and clean up resources
    pub fn deinit(self: *Self) void {
        self.listener.deinit();
        self.database.close();
        self.allocator.destroy(self.database);
    }

    /// Start the server and begin accepting connections
    pub fn start(self: *Self) !void {
        self.running = true;
        std.debug.print("Silica server listening on {}\n", .{self.address});

        while (self.running) {
            // Check connection limit
            const current_connections = self.active_connections.load(.acquire);
            if (current_connections >= self.max_connections) {
                // Wait a bit before checking again
                std.time.sleep(10 * std.time.ns_per_ms);
                continue;
            }

            // Accept a new connection
            const client = self.listener.accept() catch |err| {
                std.debug.print("Error accepting connection: {}\n", .{err});
                continue;
            };

            // Increment connection counter
            _ = self.active_connections.fetchAdd(1, .acquire);

            // Handle the connection in a separate thread
            const thread = try std.Thread.spawn(.{}, handleConnection, .{ self, client });
            thread.detach();
        }
    }

    /// Stop the server gracefully
    pub fn stop(self: *Self) void {
        self.running = false;
    }

    /// Handle a single client connection
    fn handleConnection(self: *Self, client: net.Server.Connection) void {
        defer {
            _ = self.active_connections.fetchSub(1, .release);
            client.stream.close();
        }

        // Create a connection handler
        var conn = Connection.init(self.allocator, self.database);
        defer conn.deinit();

        // Process messages from the client
        self.processMessages(&conn, client.stream) catch |err| {
            std.debug.print("Connection error: {}\n", .{err});
        };
    }

    /// Process wire protocol messages from a client stream
    fn processMessages(self: *Self, conn: *Connection, stream: net.Stream) !void {
        const reader = stream.reader();
        const writer = stream.writer();

        // Send initial ready for query (startup handshake simplified for now)
        const ready = wire.ReadyForQuery{ .status = .idle };
        try ready.write(writer);

        // Main message loop
        while (true) {
            // Read next message from client
            const msg = wire.readMessage(self.allocator, reader) catch |err| {
                if (err == error.EndOfStream) {
                    // Client disconnected gracefully
                    return;
                }
                return err;
            };
            defer msg.deinit(self.allocator);

            // Handle message based on type
            switch (msg) {
                .query => |query_msg| {
                    try conn.handleSimpleQuery(query_msg, writer);
                },
                .parse => |parse_msg| {
                    try conn.handleParse(parse_msg, writer);
                },
                .bind => |bind_msg| {
                    try conn.handleBind(bind_msg, writer);
                },
                .execute => |execute_msg| {
                    try conn.handleExecute(execute_msg, writer);
                },
                .close => |close_msg| {
                    try conn.handleClose(close_msg, writer);
                },
                .sync => {
                    try conn.handleSync(writer);
                },
                .terminate => {
                    // Client requested termination
                    return;
                },
                else => {
                    // Unsupported message type - send error
                    const err_msg = wire.ErrorResponse{
                        .severity = "ERROR",
                        .code = "08P01", // protocol_violation
                        .message = "Unsupported message type",
                    };
                    try err_msg.write(writer);
                },
            }
        }
    }
};

// Tests
test "Server.init and deinit" {
    const allocator = std.testing.allocator;

    // Create a temporary database for testing
    const db_path = "test_server.db";
    defer std.fs.cwd().deleteFile(db_path) catch {};

    // Initialize server
    var server = try Server.init(allocator, .{
        .host = "127.0.0.1",
        .port = 15433, // Use a non-standard port for testing
        .database_path = db_path,
    });
    defer server.deinit();

    try std.testing.expect(!server.running);
    try std.testing.expectEqual(@as(usize, 100), server.max_connections);
    try std.testing.expectEqual(@as(usize, 0), server.active_connections.load(.acquire));
}

test "Server.init with custom config" {
    const allocator = std.testing.allocator;

    const db_path = "test_server_custom.db";
    defer std.fs.cwd().deleteFile(db_path) catch {};

    var server = try Server.init(allocator, .{
        .host = "127.0.0.1",
        .port = 15434,
        .max_connections = 50,
        .database_path = db_path,
    });
    defer server.deinit();

    try std.testing.expectEqual(@as(usize, 50), server.max_connections);
    try std.testing.expectEqual(@as(u16, 15434), server.address.getPort());
}

test "Server.stop sets running flag" {
    const allocator = std.testing.allocator;

    const db_path = "test_server_stop.db";
    defer std.fs.cwd().deleteFile(db_path) catch {};

    var server = try Server.init(allocator, .{
        .database_path = db_path,
    });
    defer server.deinit();

    server.running = true;
    server.stop();
    try std.testing.expect(!server.running);
}
