const std = @import("std");
const net = std.net;
const os = std.os;
const Allocator = std.mem.Allocator;
const silica = @import("silica");
const Database = silica.engine.Database;
const Connection = @import("connection.zig").Connection;
const wire = @import("wire.zig");
const auth = @import("auth.zig");

/// Server represents a TCP server that accepts PostgreSQL wire protocol connections
pub const Server = struct {
    allocator: Allocator,
    address: net.Address,
    listener: net.Server,
    database: *Database,
    running: bool,
    max_connections: usize,
    active_connections: std.atomic.Value(usize),
    auth_method: auth.AuthMethod,
    credentials: auth.CredentialStore,

    const Self = @This();

    /// ServerConfig holds server configuration
    pub const Config = struct {
        host: []const u8 = "127.0.0.1",
        port: u16 = 5433,
        max_connections: usize = 100,
        database_path: []const u8,
        auth_method: auth.AuthMethod = .trust, // Default to trust for development
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
        });

        // Initialize credential store
        var credentials = auth.CredentialStore.init(allocator);

        // Add default user for development (only if using MD5 or SCRAM)
        if (config.auth_method == .md5) {
            const hash = try auth.storePasswordMd5(allocator, "postgres", "postgres");
            defer allocator.free(hash);
            try credentials.addUser("postgres", hash);
        } else if (config.auth_method == .scram_sha_256) {
            const hash = try auth.storePasswordScram(allocator, "postgres", .{});
            defer allocator.free(hash);
            try credentials.addUser("postgres", hash);
        }

        return Self{
            .allocator = allocator,
            .address = address,
            .listener = listener,
            .database = database,
            .running = false,
            .max_connections = config.max_connections,
            .active_connections = std.atomic.Value(usize).init(0),
            .auth_method = config.auth_method,
            .credentials = credentials,
        };
    }

    /// Deinitialize the server and clean up resources
    pub fn deinit(self: *Self) void {
        self.credentials.deinit();
        self.listener.deinit();
        self.database.close();
        self.allocator.destroy(self.database);
    }

    /// Start the server and begin accepting connections
    pub fn start(self: *Self) !void {
        self.running = true;
        std.debug.print("Silica server listening on {any}\n", .{self.address});

        while (self.running) {
            // Check connection limit
            const current_connections = self.active_connections.load(.acquire);
            if (current_connections >= self.max_connections) {
                // Wait a bit before checking again
                std.Thread.sleep(10 * std.time.ns_per_ms);
                continue;
            }

            // Accept a new connection
            const client = self.listener.accept() catch |err| {
                std.debug.print("Error accepting connection: {any}\n", .{err});
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
    /// Sets running flag to false and waits for all active connections to finish
    pub fn stop(self: *Self) void {
        self.running = false;
    }

    /// Wait for all active connections to finish
    /// timeout_ms: maximum time to wait in milliseconds (0 = wait indefinitely)
    /// Returns true if all connections finished, false if timeout expired
    pub fn waitForConnections(self: *Self, timeout_ms: u64) bool {
        if (timeout_ms == 0) {
            // Wait indefinitely
            while (self.active_connections.load(.acquire) > 0) {
                std.Thread.sleep(100 * std.time.ns_per_ms);
            }
            return true;
        } else {
            // Wait with timeout
            const start_time = std.time.milliTimestamp();
            while (self.active_connections.load(.acquire) > 0) {
                const elapsed = std.time.milliTimestamp() - start_time;
                if (elapsed >= timeout_ms) {
                    return false; // Timeout
                }
                std.Thread.sleep(100 * std.time.ns_per_ms);
            }
            return true;
        }
    }

    /// Shutdown the server gracefully with a timeout
    /// timeout_ms: maximum time to wait for connections to finish (0 = wait indefinitely)
    /// Returns true if shutdown completed cleanly, false if forceful shutdown required
    pub fn shutdown(self: *Self, timeout_ms: u64) bool {
        std.debug.print("Initiating graceful shutdown...\n", .{});

        // Stop accepting new connections
        self.stop();

        // Wait for active connections to finish
        const all_finished = self.waitForConnections(timeout_ms);

        if (all_finished) {
            std.debug.print("All connections closed cleanly\n", .{});
        } else {
            const remaining = self.active_connections.load(.acquire);
            std.debug.print("Shutdown timeout: {d} connections still active\n", .{remaining});
        }

        return all_finished;
    }

    /// Handle a single client connection
    fn handleConnection(self: *Self, client: net.Server.Connection) void {
        defer {
            _ = self.active_connections.fetchSub(1, .release);
            client.stream.close();
        }

        // Perform startup handshake and authentication
        const startup_info = self.performStartup(client.stream) catch |err| {
            std.debug.print("Startup handshake failed: {any}\n", .{err});
            return;
        };
        defer self.allocator.free(startup_info.user);
        defer self.allocator.free(startup_info.database);

        // Create a connection handler with authenticated user
        var conn = Connection.init(self.allocator, self.database, startup_info.user, startup_info.database) catch |err| {
            std.debug.print("Failed to initialize connection: {any}\n", .{err});
            return;
        };
        defer conn.deinit();

        // Process messages from the client
        self.processMessages(&conn, client.stream) catch |err| {
            std.debug.print("Connection error: {any}\n", .{err});
        };
    }

    /// Perform startup handshake and authentication
    fn performStartup(self: *Self, stream: net.Stream) !struct { user: []const u8, database: []const u8 } {
        const StreamReader = std.io.GenericReader(net.Stream, net.Stream.ReadError, struct {
            fn read(s: net.Stream, buffer: []u8) net.Stream.ReadError!usize {
                return s.read(buffer);
            }
        }.read);
        const reader = StreamReader{ .context = stream };

        // Read startup message (untagged)
        const startup_payload = try wire.readStartupMessage(reader, self.allocator);
        defer self.allocator.free(startup_payload);

        var startup = try wire.Startup.parse(startup_payload, self.allocator);
        defer startup.deinit(self.allocator);

        // Authenticate based on configured method
        switch (self.auth_method) {
            .trust => {
                // TRUST: no password verification
                try auth.authenticateTrust(startup.user);

                // Send AuthenticationOk
                var write_buf = std.ArrayListUnmanaged(u8){};
                defer write_buf.deinit(self.allocator);
                const writer = write_buf.writer(self.allocator);

                const auth_ok = wire.Authentication{ .auth_type = .ok, .salt = null };
                try auth_ok.write(writer);
                try stream.writeAll(write_buf.items);
            },
            .md5 => {
                // MD5: send salt, wait for password response
                const salt = auth.generateMd5Salt(std.crypto.random);

                // Send AuthenticationMD5Password
                var write_buf = std.ArrayListUnmanaged(u8){};
                defer write_buf.deinit(self.allocator);
                var writer = write_buf.writer(self.allocator);

                const auth_md5 = wire.Authentication{ .auth_type = .md5_password, .salt = salt };
                try auth_md5.write(writer);
                try stream.writeAll(write_buf.items);

                // Read password response
                const msg = try wire.readMessage(reader, self.allocator);
                defer self.allocator.free(msg.payload);

                if (msg.msg_type != 'p') return error.ProtocolError;
                const pwd_msg = try wire.PasswordMessage.parse(msg.payload);

                // Verify password
                const cred = self.credentials.getUser(startup.user) orelse return error.AuthenticationFailed;
                const verified = try auth.verifyPasswordMd5(self.allocator, cred.password_hash, pwd_msg.password, startup.user, salt);
                if (!verified) return error.AuthenticationFailed;

                // Send AuthenticationOk
                write_buf.clearRetainingCapacity();
                writer = write_buf.writer(self.allocator);
                const auth_ok = wire.Authentication{ .auth_type = .ok, .salt = null };
                try auth_ok.write(writer);
                try stream.writeAll(write_buf.items);
            },
            .scram_sha_256 => {
                // SCRAM-SHA-256: send challenge (simplified - full SASL exchange TBD)
                // For now, fall back to AuthenticationCleartextPassword
                var write_buf = std.ArrayListUnmanaged(u8){};
                defer write_buf.deinit(self.allocator);
                var writer = write_buf.writer(self.allocator);

                const auth_cleartext = wire.Authentication{ .auth_type = .cleartext_password, .salt = null };
                try auth_cleartext.write(writer);
                try stream.writeAll(write_buf.items);

                // Read password response
                const msg = try wire.readMessage(reader, self.allocator);
                defer self.allocator.free(msg.payload);

                if (msg.msg_type != 'p') return error.ProtocolError;
                const pwd_msg = try wire.PasswordMessage.parse(msg.payload);

                // Verify password
                const cred = self.credentials.getUser(startup.user) orelse return error.AuthenticationFailed;
                const verified = try auth.verifyPasswordScram(cred.password_hash, pwd_msg.password);
                if (!verified) return error.AuthenticationFailed;

                // Send AuthenticationOk
                write_buf.clearRetainingCapacity();
                writer = write_buf.writer(self.allocator);
                const auth_ok = wire.Authentication{ .auth_type = .ok, .salt = null };
                try auth_ok.write(writer);
                try stream.writeAll(write_buf.items);
            },
        }

        // Send ParameterStatus messages
        {
            var write_buf = std.ArrayListUnmanaged(u8){};
            defer write_buf.deinit(self.allocator);
            const writer = write_buf.writer(self.allocator);

            const params = wire.ParameterStatus{ .name = "server_version", .value = "14.0" };
            try params.write(writer);
            try stream.writeAll(write_buf.items);

            write_buf.clearRetainingCapacity();
            const encoding_params = wire.ParameterStatus{ .name = "client_encoding", .value = "UTF8" };
            try encoding_params.write(writer);
            try stream.writeAll(write_buf.items);
        }

        // Send ReadyForQuery
        {
            var write_buf = std.ArrayListUnmanaged(u8){};
            defer write_buf.deinit(self.allocator);
            const writer = write_buf.writer(self.allocator);

            const ready = wire.ReadyForQuery{ .status = .idle };
            try ready.write(writer);
            try stream.writeAll(write_buf.items);
        }

        // Return user and database (must dupe for caller)
        return .{
            .user = try self.allocator.dupe(u8, startup.user),
            .database = try self.allocator.dupe(u8, startup.database),
        };
    }

    /// Process wire protocol messages from a client stream
    fn processMessages(self: *Self, conn: *Connection, stream: net.Stream) !void {
        // Create a GenericReader that wraps the stream
        const StreamReader = std.io.GenericReader(net.Stream, net.Stream.ReadError, struct {
            fn read(s: net.Stream, buffer: []u8) net.Stream.ReadError!usize {
                return s.read(buffer);
            }
        }.read);
        const reader = StreamReader{ .context = stream };

        var write_buf = std.ArrayListUnmanaged(u8){};
        defer write_buf.deinit(self.allocator);

        // Main message loop (startup handshake already completed)
        while (true) {
            // Read next message from client
            const msg = wire.readMessage(reader, self.allocator) catch |err| {
                if (err == error.EndOfStream) {
                    // Client disconnected gracefully
                    return;
                }
                return err;
            };
            defer self.allocator.free(msg.payload);

            // Clear write buffer for next response
            write_buf.clearRetainingCapacity();
            const writer = write_buf.writer(self.allocator);

            // Handle message based on type
            switch (msg.msg_type) {
                'Q' => { // Query
                    const query_msg = try wire.Query.parse(msg.payload);
                    try conn.handleSimpleQuery(query_msg, writer);
                },
                'P' => { // Parse
                    const parse_msg = try wire.Parse.parse(msg.payload, self.allocator);
                    defer parse_msg.deinit(self.allocator);
                    try conn.handleParse(parse_msg, writer);
                },
                'B' => { // Bind
                    const bind_msg = try wire.Bind.parse(msg.payload, self.allocator);
                    defer bind_msg.deinit(self.allocator);
                    try conn.handleBind(bind_msg, writer);
                },
                'E' => { // Execute
                    const execute_msg = try wire.Execute.parse(msg.payload);
                    try conn.handleExecute(execute_msg.portal_name, execute_msg.max_rows, writer);
                },
                'C' => { // Close
                    const close_msg = try wire.Close.parse(msg.payload);
                    try conn.handleClose(close_msg.close_type, close_msg.name, writer);
                },
                'S' => { // Sync
                    try conn.handleSync(writer);
                },
                'X' => { // Terminate
                    // Client requested termination
                    return;
                },
                else => {
                    // Unsupported message type - send error
                    const fields = try self.allocator.alloc(wire.ErrorResponse.Field, 3);
                    defer self.allocator.free(fields);
                    fields[0] = .{ .code = 'S', .value = "ERROR" };
                    fields[1] = .{ .code = 'C', .value = "08P01" }; // protocol_violation
                    fields[2] = .{ .code = 'M', .value = "Unsupported message type" };
                    const err_msg = wire.ErrorResponse{ .fields = fields };
                    try err_msg.write(writer);
                },
            }

            // Send buffered response to client
            if (write_buf.items.len > 0) {
                try stream.writeAll(write_buf.items);
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

test "Server.waitForConnections returns immediately when no active connections" {
    const allocator = std.testing.allocator;

    const db_path = "test_server_wait_none.db";
    defer std.fs.cwd().deleteFile(db_path) catch {};

    var server = try Server.init(allocator, .{
        .database_path = db_path,
    });
    defer server.deinit();

    // No active connections, should return immediately
    const finished = server.waitForConnections(5000);
    try std.testing.expect(finished);
}

test "Server.waitForConnections waits for active connections" {
    const allocator = std.testing.allocator;

    const db_path = "test_server_wait_active.db";
    defer std.fs.cwd().deleteFile(db_path) catch {};

    var server = try Server.init(allocator, .{
        .database_path = db_path,
    });
    defer server.deinit();

    // Simulate active connection
    _ = server.active_connections.fetchAdd(1, .acquire);

    // Spawn thread to decrement after 200ms
    const thread = try std.Thread.spawn(.{}, struct {
        fn decrementAfterDelay(s: *Server) void {
            std.Thread.sleep(200 * std.time.ns_per_ms);
            _ = s.active_connections.fetchSub(1, .release);
        }
    }.decrementAfterDelay, .{&server});
    thread.detach();

    // Wait with 1 second timeout - should succeed
    const finished = server.waitForConnections(1000);
    try std.testing.expect(finished);
    try std.testing.expectEqual(@as(usize, 0), server.active_connections.load(.acquire));
}

test "Server.waitForConnections times out when connections don't finish" {
    const allocator = std.testing.allocator;

    const db_path = "test_server_wait_timeout.db";
    defer std.fs.cwd().deleteFile(db_path) catch {};

    var server = try Server.init(allocator, .{
        .database_path = db_path,
    });
    defer server.deinit();

    // Simulate active connection that won't finish
    _ = server.active_connections.fetchAdd(1, .acquire);
    defer _ = server.active_connections.fetchSub(1, .release);

    // Wait with short timeout - should timeout
    const finished = server.waitForConnections(100);
    try std.testing.expect(!finished);
    try std.testing.expectEqual(@as(usize, 1), server.active_connections.load(.acquire));
}

test "Server.shutdown with no active connections" {
    const allocator = std.testing.allocator;

    const db_path = "test_server_shutdown_clean.db";
    defer std.fs.cwd().deleteFile(db_path) catch {};

    var server = try Server.init(allocator, .{
        .database_path = db_path,
    });
    defer server.deinit();

    server.running = true;

    // Shutdown should complete immediately
    const clean = server.shutdown(1000);
    try std.testing.expect(clean);
    try std.testing.expect(!server.running);
}

test "Server.shutdown with timeout on active connections" {
    const allocator = std.testing.allocator;

    const db_path = "test_server_shutdown_timeout.db";
    defer std.fs.cwd().deleteFile(db_path) catch {};

    var server = try Server.init(allocator, .{
        .database_path = db_path,
    });
    defer server.deinit();

    server.running = true;

    // Simulate active connection
    _ = server.active_connections.fetchAdd(1, .acquire);
    defer _ = server.active_connections.fetchSub(1, .release);

    // Shutdown with short timeout - should timeout
    const clean = server.shutdown(100);
    try std.testing.expect(!clean);
    try std.testing.expect(!server.running);
}

test "Server.active_connections - atomicity of increment/decrement" {
    const allocator = std.testing.allocator;

    const db_path = "test_server_atomicity.db";
    defer std.fs.cwd().deleteFile(db_path) catch {};

    var server = try Server.init(allocator, .{
        .database_path = db_path,
    });
    defer server.deinit();

    // Test that concurrent increment/decrement operations are atomic
    const thread_count = 10;
    const operations_per_thread = 100;

    var threads: [thread_count]std.Thread = undefined;
    for (&threads) |*thread| {
        thread.* = try std.Thread.spawn(.{}, struct {
            fn worker(s: *Server) void {
                var i: usize = 0;
                while (i < operations_per_thread) : (i += 1) {
                    _ = s.active_connections.fetchAdd(1, .acquire);
                    _ = s.active_connections.fetchSub(1, .release);
                }
            }
        }.worker, .{&server});
    }

    for (threads) |thread| {
        thread.join();
    }

    // After all threads complete, counter should be back to 0
    try std.testing.expectEqual(@as(usize, 0), server.active_connections.load(.acquire));
}

test "Server.max_connections - enforce connection limit" {
    const allocator = std.testing.allocator;

    const db_path = "test_server_max_conn.db";
    defer std.fs.cwd().deleteFile(db_path) catch {};

    var server = try Server.init(allocator, .{
        .database_path = db_path,
        .max_connections = 5,
    });
    defer server.deinit();

    // Simulate reaching max connections
    var i: usize = 0;
    while (i < 5) : (i += 1) {
        _ = server.active_connections.fetchAdd(1, .acquire);
    }
    defer {
        i = 0;
        while (i < 5) : (i += 1) {
            _ = server.active_connections.fetchSub(1, .release);
        }
    }

    try std.testing.expectEqual(@as(usize, 5), server.active_connections.load(.acquire));

    // Verify we're at the limit
    const current = server.active_connections.load(.acquire);
    try std.testing.expect(current >= server.max_connections);
}

test "Server.waitForConnections - zero timeout means wait indefinitely" {
    const allocator = std.testing.allocator;

    const db_path = "test_server_wait_zero.db";
    defer std.fs.cwd().deleteFile(db_path) catch {};

    var server = try Server.init(allocator, .{
        .database_path = db_path,
    });
    defer server.deinit();

    // Simulate active connection
    _ = server.active_connections.fetchAdd(1, .acquire);

    // Spawn thread to decrement after 100ms
    const thread = try std.Thread.spawn(.{}, struct {
        fn decrementAfterDelay(s: *Server) void {
            std.Thread.sleep(100 * std.time.ns_per_ms);
            _ = s.active_connections.fetchSub(1, .release);
        }
    }.decrementAfterDelay, .{&server});
    thread.detach();

    // Wait with 0 timeout (indefinite) - should succeed
    const finished = server.waitForConnections(0);
    try std.testing.expect(finished);
    try std.testing.expectEqual(@as(usize, 0), server.active_connections.load(.acquire));
}
