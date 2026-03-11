//! Authentication Methods for Silica Database Server
//!
//! Implements PostgreSQL-compatible authentication:
//! - TRUST: no authentication (development/local only)
//! - MD5: MD5-hashed password (legacy, but widely supported)
//! - SCRAM-SHA-256: modern SASL-based authentication (PostgreSQL 10+)
//!
//! Reference: https://www.postgresql.org/docs/current/auth-methods.html

const std = @import("std");
const crypto = std.crypto;
const Allocator = std.mem.Allocator;

/// Authentication method selection
pub const AuthMethod = enum {
    trust, // No authentication (DANGEROUS - dev only)
    md5, // MD5-hashed password
    scram_sha_256, // SCRAM-SHA-256 (recommended)
};

/// User credential store entry
pub const UserCredential = struct {
    username: []const u8,
    password_hash: []const u8, // Format depends on auth method
    allocator: Allocator,

    pub fn init(allocator: Allocator, username: []const u8, password_hash: []const u8) !UserCredential {
        const username_copy = try allocator.dupe(u8, username);
        errdefer allocator.free(username_copy);
        const hash_copy = try allocator.dupe(u8, password_hash);
        return UserCredential{
            .username = username_copy,
            .password_hash = hash_copy,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *UserCredential) void {
        self.allocator.free(self.username);
        self.allocator.free(self.password_hash);
    }
};

/// Simple in-memory credential store
/// Production should use catalog-backed storage
pub const CredentialStore = struct {
    users: std.StringHashMap(UserCredential),
    allocator: Allocator,

    pub fn init(allocator: Allocator) CredentialStore {
        return CredentialStore{
            .users = std.StringHashMap(UserCredential).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *CredentialStore) void {
        var it = self.users.valueIterator();
        while (it.next()) |cred| {
            var mut_cred = cred.*;
            mut_cred.deinit();
        }
        self.users.deinit();
    }

    pub fn addUser(self: *CredentialStore, username: []const u8, password_hash: []const u8) !void {
        const cred = try UserCredential.init(self.allocator, username, password_hash);
        try self.users.put(cred.username, cred);
    }

    pub fn getUser(self: *CredentialStore, username: []const u8) ?*const UserCredential {
        return self.users.getPtr(username);
    }
};

// ── TRUST Authentication ──────────────────────────────────────────────────

/// TRUST authentication - no password verification
/// WARNING: Only use in development or trusted networks
pub fn authenticateTrust(username: []const u8) !void {
    // Always succeeds - just log the username
    _ = username;
}

// ── MD5 Authentication ──────────────────────────────────────────────────

/// Generate MD5 salt (4 random bytes)
pub fn generateMd5Salt(random: std.Random) [4]u8 {
    var salt: [4]u8 = undefined;
    random.bytes(&salt);
    return salt;
}

/// Hash password for MD5 authentication
/// Format: md5(md5(password + username) + salt)
pub fn hashPasswordMd5(
    allocator: Allocator,
    password: []const u8,
    username: []const u8,
    salt: [4]u8,
) ![]const u8 {
    var inner_hash: [32]u8 = undefined; // hex encoding of MD5
    var outer_hash: [32]u8 = undefined;

    // Inner hash: md5(password + username)
    {
        var hasher = crypto.hash.Md5.init(.{});
        hasher.update(password);
        hasher.update(username);
        var digest: [16]u8 = undefined;
        hasher.final(&digest);
        _ = std.fmt.bufPrint(&inner_hash, "{s}", .{std.fmt.fmtSliceHexLower(&digest)}) catch unreachable;
    }

    // Outer hash: md5(inner_hash + salt)
    {
        var hasher = crypto.hash.Md5.init(.{});
        hasher.update(&inner_hash);
        hasher.update(&salt);
        var digest: [16]u8 = undefined;
        hasher.final(&digest);
        _ = std.fmt.bufPrint(&outer_hash, "{s}", .{std.fmt.fmtSliceHexLower(&digest)}) catch unreachable;
    }

    // Return with "md5" prefix
    const result = try allocator.alloc(u8, 3 + 32);
    @memcpy(result[0..3], "md5");
    @memcpy(result[3..], &outer_hash);
    return result;
}

/// Verify MD5 password
pub fn verifyPasswordMd5(
    allocator: Allocator,
    stored_hash: []const u8,
    provided_password: []const u8,
    username: []const u8,
    salt: [4]u8,
) !bool {
    const computed_hash = try hashPasswordMd5(allocator, provided_password, username, salt);
    defer allocator.free(computed_hash);
    return std.mem.eql(u8, stored_hash, computed_hash);
}

/// Store MD5 password hash in database
/// Format: md5(password + username) (without salt)
pub fn storePasswordMd5(
    allocator: Allocator,
    password: []const u8,
    username: []const u8,
) ![]const u8 {
    var hash: [32]u8 = undefined;
    var hasher = crypto.hash.Md5.init(.{});
    hasher.update(password);
    hasher.update(username);
    var digest: [16]u8 = undefined;
    hasher.final(&digest);
    _ = std.fmt.bufPrint(&hash, "{s}", .{std.fmt.fmtSliceHexLower(&digest)}) catch unreachable;

    const result = try allocator.alloc(u8, 3 + 32);
    @memcpy(result[0..3], "md5");
    @memcpy(result[3..], &hash);
    return result;
}

// ── SCRAM-SHA-256 Authentication ──────────────────────────────────────────

/// SCRAM-SHA-256 configuration
pub const ScramConfig = struct {
    iterations: u32 = 4096, // PBKDF2 iterations (PostgreSQL default)
};

/// Generate SCRAM-SHA-256 salt (16 random bytes)
pub fn generateScramSalt(random: std.Random) [16]u8 {
    var salt: [16]u8 = undefined;
    random.bytes(&salt);
    return salt;
}

/// Store SCRAM-SHA-256 password hash
/// Format: SCRAM-SHA-256$<iterations>:<salt_base64>$<stored_key_base64>:<server_key_base64>
pub fn storePasswordScram(
    allocator: Allocator,
    password: []const u8,
    config: ScramConfig,
) ![]const u8 {
    const salt = generateScramSalt(std.crypto.random);

    // Compute salted password: Hi(password, salt, iterations)
    var salted_password: [32]u8 = undefined;
    try crypto.pwhash.pbkdf2(&salted_password, password, &salt, config.iterations, crypto.auth.hmac.sha2.HmacSha256);

    // Client key = HMAC(salted_password, "Client Key")
    var client_key: [32]u8 = undefined;
    crypto.auth.hmac.sha2.HmacSha256.create(&client_key, "Client Key", &salted_password);

    // Stored key = H(client_key)
    var stored_key: [32]u8 = undefined;
    crypto.hash.sha2.Sha256.hash(&client_key, &stored_key, .{});

    // Server key = HMAC(salted_password, "Server Key")
    var server_key: [32]u8 = undefined;
    crypto.auth.hmac.sha2.HmacSha256.create(&server_key, "Server Key", &salted_password);

    // Encode to base64
    const encoder = std.base64.standard.Encoder;
    var salt_b64: [24]u8 = undefined;
    var stored_key_b64: [44]u8 = undefined;
    var server_key_b64: [44]u8 = undefined;
    _ = encoder.encode(&salt_b64, &salt);
    _ = encoder.encode(&stored_key_b64, &stored_key);
    _ = encoder.encode(&server_key_b64, &server_key);

    // Format: SCRAM-SHA-256$iterations:salt$stored_key:server_key
    var buf = std.ArrayList(u8).init(allocator);
    defer buf.deinit();
    try buf.appendSlice("SCRAM-SHA-256$");
    try std.fmt.format(buf.writer(), "{d}:", .{config.iterations});
    try buf.appendSlice(&salt_b64);
    try buf.append('$');
    try buf.appendSlice(&stored_key_b64);
    try buf.append(':');
    try buf.appendSlice(&server_key_b64);

    return buf.toOwnedSlice();
}

/// Verify SCRAM-SHA-256 password (simplified - full SASL exchange in wire protocol handler)
pub fn verifyPasswordScram(
    stored_hash: []const u8,
    provided_password: []const u8,
) !bool {
    // Parse stored hash: SCRAM-SHA-256$iterations:salt$stored_key:server_key
    if (!std.mem.startsWith(u8, stored_hash, "SCRAM-SHA-256$")) return error.InvalidHashFormat;

    const parts = std.mem.tokenizeScalar(u8, stored_hash[14..], '$');
    var iter_salt = parts.next() orelse return error.InvalidHashFormat;
    const keys = parts.next() orelse return error.InvalidHashFormat;

    // Split iterations:salt
    const colon1 = std.mem.indexOfScalar(u8, iter_salt, ':') orelse return error.InvalidHashFormat;
    const iterations_str = iter_salt[0..colon1];
    const salt_b64 = iter_salt[colon1 + 1 ..];

    const iterations = try std.fmt.parseInt(u32, iterations_str, 10);

    // Decode salt
    const decoder = std.base64.standard.Decoder;
    var salt: [16]u8 = undefined;
    try decoder.decode(&salt, salt_b64);

    // Split stored_key:server_key
    const colon2 = std.mem.indexOfScalar(u8, keys, ':') orelse return error.InvalidHashFormat;
    const stored_key_b64 = keys[0..colon2];

    // Decode stored key
    var expected_stored_key: [32]u8 = undefined;
    try decoder.decode(&expected_stored_key, stored_key_b64);

    // Compute salted password from provided password
    var salted_password: [32]u8 = undefined;
    try crypto.pwhash.pbkdf2(&salted_password, provided_password, &salt, iterations, crypto.auth.hmac.sha2.HmacSha256);

    // Client key = HMAC(salted_password, "Client Key")
    var client_key: [32]u8 = undefined;
    crypto.auth.hmac.sha2.HmacSha256.create(&client_key, "Client Key", &salted_password);

    // Stored key = H(client_key)
    var computed_stored_key: [32]u8 = undefined;
    crypto.hash.sha2.Sha256.hash(&client_key, &computed_stored_key, .{});

    // Compare
    return crypto.utils.timingSafeEql([32]u8, expected_stored_key, computed_stored_key);
}

// ── Tests ──────────────────────────────────────────────────────────────

test "CredentialStore - add and get user" {
    const allocator = std.testing.allocator;
    var store = CredentialStore.init(allocator);
    defer store.deinit();

    try store.addUser("alice", "md5abc123");
    const cred = store.getUser("alice");
    try std.testing.expect(cred != null);
    try std.testing.expectEqualStrings("alice", cred.?.username);
    try std.testing.expectEqualStrings("md5abc123", cred.?.password_hash);

    const missing = store.getUser("bob");
    try std.testing.expect(missing == null);
}

test "TRUST authentication - always succeeds" {
    try authenticateTrust("anyone");
}

test "MD5 - generate salt" {
    const random = std.Random.DefaultPrng.init(42);
    const salt1 = generateMd5Salt(random.random());
    const salt2 = generateMd5Salt(random.random());
    // Different salts should be generated
    try std.testing.expect(!std.mem.eql(u8, &salt1, &salt2));
}

test "MD5 - store and verify password" {
    const allocator = std.testing.allocator;

    const stored = try storePasswordMd5(allocator, "secret", "alice");
    defer allocator.free(stored);

    // Stored hash should start with "md5"
    try std.testing.expect(std.mem.startsWith(u8, stored, "md5"));
    try std.testing.expectEqual(35, stored.len); // "md5" + 32 hex chars

    // Verify with correct password
    const salt: [4]u8 = .{ 0xDE, 0xAD, 0xBE, 0xEF };
    const verified = try verifyPasswordMd5(allocator, stored, "secret", "alice", salt);
    try std.testing.expect(!verified); // Different salt, so won't match

    // Verify with matching salt
    const correct_hash = try hashPasswordMd5(allocator, "secret", "alice", salt);
    defer allocator.free(correct_hash);
    const match = try verifyPasswordMd5(allocator, correct_hash, "secret", "alice", salt);
    try std.testing.expect(match);

    // Wrong password
    const wrong = try verifyPasswordMd5(allocator, correct_hash, "wrong", "alice", salt);
    try std.testing.expect(!wrong);
}

test "SCRAM-SHA-256 - store password" {
    const allocator = std.testing.allocator;

    const stored = try storePasswordScram(allocator, "secret", .{});
    defer allocator.free(stored);

    // Should start with SCRAM-SHA-256$
    try std.testing.expect(std.mem.startsWith(u8, stored, "SCRAM-SHA-256$"));

    // Should have format: SCRAM-SHA-256$iterations:salt$stored_key:server_key
    const parts = std.mem.tokenizeScalar(u8, stored[14..], '$');
    const iter_salt = parts.next();
    const keys = parts.next();
    try std.testing.expect(iter_salt != null);
    try std.testing.expect(keys != null);
    try std.testing.expect(std.mem.indexOfScalar(u8, iter_salt.?, ':') != null);
    try std.testing.expect(std.mem.indexOfScalar(u8, keys.?, ':') != null);
}

test "SCRAM-SHA-256 - verify password" {
    const allocator = std.testing.allocator;

    const stored = try storePasswordScram(allocator, "secret", .{});
    defer allocator.free(stored);

    // Correct password
    const verified = try verifyPasswordScram(stored, "secret");
    try std.testing.expect(verified);

    // Wrong password
    const wrong = try verifyPasswordScram(stored, "wrong");
    try std.testing.expect(!wrong);
}

test "SCRAM-SHA-256 - invalid hash format" {
    const result = verifyPasswordScram("invalid", "password");
    try std.testing.expectError(error.InvalidHashFormat, result);
}

test "SCRAM-SHA-256 - different iterations" {
    const allocator = std.testing.allocator;

    const config = ScramConfig{ .iterations = 8192 };
    const stored = try storePasswordScram(allocator, "secret", config);
    defer allocator.free(stored);

    // Should contain "8192:"
    try std.testing.expect(std.mem.indexOf(u8, stored, "8192:") != null);

    const verified = try verifyPasswordScram(stored, "secret");
    try std.testing.expect(verified);
}
