//! Configuration Manager — runtime parameter storage and validation
//!
//! Manages session-level and global configuration parameters with type checking,
//! range validation, and default value handling. Parameters persist for the
//! lifetime of a session (connection) unless explicitly reset.
//!
//! Supported parameter types:
//!   - INTEGER: numeric values with optional min/max range
//!   - TEXT: string values
//!   - BOOLEAN: true/false/on/off
//!   - SIZE: memory sizes with unit parsing (KB/MB/GB)

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const ParameterType = enum {
    integer,
    text,
    boolean,
    size, // memory size in bytes (supports KB, MB, GB suffixes)
};

pub const ParameterDef = struct {
    name: []const u8,
    param_type: ParameterType,
    default_value: []const u8,
    min_value: ?i64 = null,
    max_value: ?i64 = null,
    description: []const u8,
};

pub const ConfigError = error{
    UnknownParameter,
    InvalidType,
    OutOfRange,
    InvalidSizeFormat,
    OutOfMemory,
};

/// Parameter name/value pair for getAll()
pub const ParamEntry = struct {
    name: []const u8,
    value: []const u8,
};

/// Runtime configuration manager
pub const ConfigManager = struct {
    allocator: Allocator,
    /// Parameter definitions (immutable registry)
    definitions: std.StringHashMap(ParameterDef),
    /// Current values (session state)
    values: std.StringHashMap([]const u8),

    pub fn init(allocator: Allocator) ConfigManager {
        return .{
            .allocator = allocator,
            .definitions = std.StringHashMap(ParameterDef).init(allocator),
            .values = std.StringHashMap([]const u8).init(allocator),
        };
    }

    pub fn deinit(self: *ConfigManager) void {
        // Free all stored values
        var value_iter = self.values.iterator();
        while (value_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.values.deinit();
        self.definitions.deinit();
    }

    /// Register a parameter with its definition
    pub fn registerParameter(self: *ConfigManager, def: ParameterDef) !void {
        try self.definitions.put(def.name, def);
        // Initialize with default value
        const name_copy = try self.allocator.dupe(u8, def.name);
        const value_copy = try self.allocator.dupe(u8, def.default_value);
        try self.values.put(name_copy, value_copy);
    }

    /// Set a parameter value
    pub fn set(self: *ConfigManager, name: []const u8, value: []const u8) ConfigError!void {
        const def = self.definitions.get(name) orelse return error.UnknownParameter;

        // Validate value according to type
        switch (def.param_type) {
            .integer => {
                const int_val = std.fmt.parseInt(i64, value, 10) catch return error.InvalidType;
                if (def.min_value) |min| {
                    if (int_val < min) return error.OutOfRange;
                }
                if (def.max_value) |max| {
                    if (int_val > max) return error.OutOfRange;
                }
            },
            .boolean => {
                // Accept: true, false, on, off, 1, 0
                const lower = std.ascii.toLower(value[0]);
                if (value.len == 1) {
                    if (lower != '1' and lower != '0') return error.InvalidType;
                } else {
                    const is_valid = std.mem.eql(u8, value, "true") or
                        std.mem.eql(u8, value, "false") or
                        std.mem.eql(u8, value, "on") or
                        std.mem.eql(u8, value, "off") or
                        std.mem.eql(u8, value, "TRUE") or
                        std.mem.eql(u8, value, "FALSE") or
                        std.mem.eql(u8, value, "ON") or
                        std.mem.eql(u8, value, "OFF");
                    if (!is_valid) return error.InvalidType;
                }
            },
            .size => {
                _ = parseSize(value) catch return error.InvalidSizeFormat;
            },
            .text => {
                // TEXT accepts any value
            },
        }

        // Store the validated value
        if (self.values.getPtr(name)) |existing_value| {
            self.allocator.free(existing_value.*);
            existing_value.* = try self.allocator.dupe(u8, value);
        } else {
            const name_copy = try self.allocator.dupe(u8, name);
            const value_copy = try self.allocator.dupe(u8, value);
            try self.values.put(name_copy, value_copy);
        }
    }

    /// Get a parameter value
    pub fn get(self: *const ConfigManager, name: []const u8) ?[]const u8 {
        return self.values.get(name);
    }

    /// Reset a parameter to its default value
    pub fn reset(self: *ConfigManager, name: []const u8) ConfigError!void {
        const def = self.definitions.get(name) orelse return error.UnknownParameter;
        try self.set(name, def.default_value);
    }

    /// Reset all parameters to defaults
    pub fn resetAll(self: *ConfigManager) ConfigError!void {
        var iter = self.definitions.iterator();
        while (iter.next()) |entry| {
            try self.set(entry.key_ptr.*, entry.value_ptr.default_value);
        }
    }

    /// Get all parameters as name/value pairs
    pub fn getAll(self: *const ConfigManager, allocator: Allocator) ![]ParamEntry {
        var list = std.ArrayListUnmanaged(ParamEntry){};
        defer list.deinit(allocator);

        var iter = self.values.iterator();
        while (iter.next()) |entry| {
            try list.append(allocator, .{
                .name = entry.key_ptr.*,
                .value = entry.value_ptr.*,
            });
        }

        return list.toOwnedSlice(allocator);
    }
};

/// Parse size string with unit suffix (e.g., "4MB", "1024KB")
fn parseSize(s: []const u8) !i64 {
    if (s.len == 0) return error.InvalidSizeFormat;

    // Find where the unit starts
    var num_end: usize = s.len;
    for (s, 0..) |c, i| {
        if (!std.ascii.isDigit(c)) {
            num_end = i;
            break;
        }
    }

    if (num_end == 0) return error.InvalidSizeFormat;

    const num_str = s[0..num_end];
    const base = try std.fmt.parseInt(i64, num_str, 10);

    if (num_end == s.len) {
        // No unit, return as-is (bytes)
        return base;
    }

    const unit = s[num_end..];
    const multiplier: i64 = if (std.mem.eql(u8, unit, "KB") or std.mem.eql(u8, unit, "kb"))
        1024
    else if (std.mem.eql(u8, unit, "MB") or std.mem.eql(u8, unit, "mb"))
        1024 * 1024
    else if (std.mem.eql(u8, unit, "GB") or std.mem.eql(u8, unit, "gb"))
        1024 * 1024 * 1024
    else
        return error.InvalidSizeFormat;

    return base * multiplier;
}

// ── Tests ─────────────────────────────────────────────────────────────

test "ConfigManager init and deinit" {
    var config = ConfigManager.init(std.testing.allocator);
    defer config.deinit();
}

test "registerParameter sets default value" {
    var config = ConfigManager.init(std.testing.allocator);
    defer config.deinit();

    try config.registerParameter(.{
        .name = "work_mem",
        .param_type = .integer,
        .default_value = "4194304",
        .description = "Memory for sorts and hashes",
    });

    const value = config.get("work_mem").?;
    try std.testing.expectEqualStrings("4194304", value);
}

test "set updates parameter value" {
    var config = ConfigManager.init(std.testing.allocator);
    defer config.deinit();

    try config.registerParameter(.{
        .name = "work_mem",
        .param_type = .integer,
        .default_value = "4194304",
        .description = "Memory for sorts and hashes",
    });

    try config.set("work_mem", "8388608");

    const value = config.get("work_mem").?;
    try std.testing.expectEqualStrings("8388608", value);
}

test "set unknown parameter returns error" {
    var config = ConfigManager.init(std.testing.allocator);
    defer config.deinit();

    const result = config.set("unknown_param", "123");
    try std.testing.expectError(error.UnknownParameter, result);
}

test "set integer with invalid type returns error" {
    var config = ConfigManager.init(std.testing.allocator);
    defer config.deinit();

    try config.registerParameter(.{
        .name = "max_connections",
        .param_type = .integer,
        .default_value = "100",
        .description = "Maximum connections",
    });

    const result = config.set("max_connections", "not_a_number");
    try std.testing.expectError(error.InvalidType, result);
}

test "set integer below min returns error" {
    var config = ConfigManager.init(std.testing.allocator);
    defer config.deinit();

    try config.registerParameter(.{
        .name = "max_connections",
        .param_type = .integer,
        .default_value = "100",
        .min_value = 10,
        .max_value = 1000,
        .description = "Maximum connections",
    });

    const result = config.set("max_connections", "5");
    try std.testing.expectError(error.OutOfRange, result);
}

test "set integer above max returns error" {
    var config = ConfigManager.init(std.testing.allocator);
    defer config.deinit();

    try config.registerParameter(.{
        .name = "max_connections",
        .param_type = .integer,
        .default_value = "100",
        .min_value = 10,
        .max_value = 1000,
        .description = "Maximum connections",
    });

    const result = config.set("max_connections", "2000");
    try std.testing.expectError(error.OutOfRange, result);
}

test "set boolean accepts true/false" {
    var config = ConfigManager.init(std.testing.allocator);
    defer config.deinit();

    try config.registerParameter(.{
        .name = "enable_feature",
        .param_type = .boolean,
        .default_value = "false",
        .description = "Enable feature flag",
    });

    try config.set("enable_feature", "true");
    try std.testing.expectEqualStrings("true", config.get("enable_feature").?);

    try config.set("enable_feature", "false");
    try std.testing.expectEqualStrings("false", config.get("enable_feature").?);
}

test "set boolean accepts on/off" {
    var config = ConfigManager.init(std.testing.allocator);
    defer config.deinit();

    try config.registerParameter(.{
        .name = "enable_feature",
        .param_type = .boolean,
        .default_value = "false",
        .description = "Enable feature flag",
    });

    try config.set("enable_feature", "on");
    try std.testing.expectEqualStrings("on", config.get("enable_feature").?);

    try config.set("enable_feature", "off");
    try std.testing.expectEqualStrings("off", config.get("enable_feature").?);
}

test "set boolean accepts 1/0" {
    var config = ConfigManager.init(std.testing.allocator);
    defer config.deinit();

    try config.registerParameter(.{
        .name = "enable_feature",
        .param_type = .boolean,
        .default_value = "0",
        .description = "Enable feature flag",
    });

    try config.set("enable_feature", "1");
    try std.testing.expectEqualStrings("1", config.get("enable_feature").?);

    try config.set("enable_feature", "0");
    try std.testing.expectEqualStrings("0", config.get("enable_feature").?);
}

test "set boolean rejects invalid values" {
    var config = ConfigManager.init(std.testing.allocator);
    defer config.deinit();

    try config.registerParameter(.{
        .name = "enable_feature",
        .param_type = .boolean,
        .default_value = "false",
        .description = "Enable feature flag",
    });

    try std.testing.expectError(error.InvalidType, config.set("enable_feature", "yes"));
    try std.testing.expectError(error.InvalidType, config.set("enable_feature", "no"));
    try std.testing.expectError(error.InvalidType, config.set("enable_feature", "maybe"));
}

test "set size accepts MB suffix" {
    var config = ConfigManager.init(std.testing.allocator);
    defer config.deinit();

    try config.registerParameter(.{
        .name = "work_mem",
        .param_type = .size,
        .default_value = "4MB",
        .description = "Memory for sorts",
    });

    try config.set("work_mem", "8MB");
    try std.testing.expectEqualStrings("8MB", config.get("work_mem").?);
}

test "set size accepts KB suffix" {
    var config = ConfigManager.init(std.testing.allocator);
    defer config.deinit();

    try config.registerParameter(.{
        .name = "work_mem",
        .param_type = .size,
        .default_value = "4096KB",
        .description = "Memory for sorts",
    });

    try config.set("work_mem", "8192KB");
    try std.testing.expectEqualStrings("8192KB", config.get("work_mem").?);
}

test "set size accepts plain bytes" {
    var config = ConfigManager.init(std.testing.allocator);
    defer config.deinit();

    try config.registerParameter(.{
        .name = "work_mem",
        .param_type = .size,
        .default_value = "4194304",
        .description = "Memory for sorts",
    });

    try config.set("work_mem", "8388608");
    try std.testing.expectEqualStrings("8388608", config.get("work_mem").?);
}

test "set size rejects invalid format" {
    var config = ConfigManager.init(std.testing.allocator);
    defer config.deinit();

    try config.registerParameter(.{
        .name = "work_mem",
        .param_type = .size,
        .default_value = "4MB",
        .description = "Memory for sorts",
    });

    try std.testing.expectError(error.InvalidSizeFormat, config.set("work_mem", "4XB"));
    try std.testing.expectError(error.InvalidSizeFormat, config.set("work_mem", "not_a_size"));
}

test "set text accepts any value" {
    var config = ConfigManager.init(std.testing.allocator);
    defer config.deinit();

    try config.registerParameter(.{
        .name = "search_path",
        .param_type = .text,
        .default_value = "public",
        .description = "Schema search path",
    });

    try config.set("search_path", "public, private, test");
    try std.testing.expectEqualStrings("public, private, test", config.get("search_path").?);
}

test "reset restores default value" {
    var config = ConfigManager.init(std.testing.allocator);
    defer config.deinit();

    try config.registerParameter(.{
        .name = "work_mem",
        .param_type = .integer,
        .default_value = "4194304",
        .description = "Memory for sorts",
    });

    try config.set("work_mem", "8388608");
    try std.testing.expectEqualStrings("8388608", config.get("work_mem").?);

    try config.reset("work_mem");
    try std.testing.expectEqualStrings("4194304", config.get("work_mem").?);
}

test "reset unknown parameter returns error" {
    var config = ConfigManager.init(std.testing.allocator);
    defer config.deinit();

    try std.testing.expectError(error.UnknownParameter, config.reset("unknown_param"));
}

test "resetAll restores all defaults" {
    var config = ConfigManager.init(std.testing.allocator);
    defer config.deinit();

    try config.registerParameter(.{
        .name = "work_mem",
        .param_type = .integer,
        .default_value = "4194304",
        .description = "Memory for sorts",
    });

    try config.registerParameter(.{
        .name = "max_connections",
        .param_type = .integer,
        .default_value = "100",
        .description = "Max connections",
    });

    try config.set("work_mem", "8388608");
    try config.set("max_connections", "200");

    try config.resetAll();

    try std.testing.expectEqualStrings("4194304", config.get("work_mem").?);
    try std.testing.expectEqualStrings("100", config.get("max_connections").?);
}

test "getAll returns all parameters" {
    var config = ConfigManager.init(std.testing.allocator);
    defer config.deinit();

    try config.registerParameter(.{
        .name = "work_mem",
        .param_type = .integer,
        .default_value = "4194304",
        .description = "Memory for sorts",
    });

    try config.registerParameter(.{
        .name = "max_connections",
        .param_type = .integer,
        .default_value = "100",
        .description = "Max connections",
    });

    const all_params = try config.getAll(std.testing.allocator);
    defer std.testing.allocator.free(all_params);

    try std.testing.expectEqual(@as(usize, 2), all_params.len);
}

test "get non-existent parameter returns null" {
    var config = ConfigManager.init(std.testing.allocator);
    defer config.deinit();

    try std.testing.expect(config.get("does_not_exist") == null);
}

test "parseSize handles MB correctly" {
    const result = try parseSize("4MB");
    try std.testing.expectEqual(@as(i64, 4 * 1024 * 1024), result);
}

test "parseSize handles KB correctly" {
    const result = try parseSize("1024KB");
    try std.testing.expectEqual(@as(i64, 1024 * 1024), result);
}

test "parseSize handles GB correctly" {
    const result = try parseSize("2GB");
    try std.testing.expectEqual(@as(i64, 2 * 1024 * 1024 * 1024), result);
}

test "parseSize handles plain bytes" {
    const result = try parseSize("12345");
    try std.testing.expectEqual(@as(i64, 12345), result);
}

test "parseSize case insensitive units" {
    const mb_result = try parseSize("4mb");
    try std.testing.expectEqual(@as(i64, 4 * 1024 * 1024), mb_result);

    const kb_result = try parseSize("512kb");
    try std.testing.expectEqual(@as(i64, 512 * 1024), kb_result);
}

test "parseSize rejects empty string" {
    try std.testing.expectError(error.InvalidSizeFormat, parseSize(""));
}

test "parseSize rejects invalid unit" {
    try std.testing.expectError(error.InvalidSizeFormat, parseSize("4TB"));
    try std.testing.expectError(error.InvalidSizeFormat, parseSize("4XB"));
}

// ── Additional comprehensive tests ─────────────────────────────────────

test "registerParameter with integer bounds" {
    var config = ConfigManager.init(std.testing.allocator);
    defer config.deinit();

    try config.registerParameter(.{
        .name = "max_workers",
        .param_type = .integer,
        .default_value = "10",
        .min_value = 1,
        .max_value = 100,
        .description = "Maximum worker threads",
    });

    // Default should be within bounds
    const value = config.get("max_workers").?;
    try std.testing.expectEqualStrings("10", value);
}

test "set integer at boundary values" {
    var config = ConfigManager.init(std.testing.allocator);
    defer config.deinit();

    try config.registerParameter(.{
        .name = "port",
        .param_type = .integer,
        .default_value = "5432",
        .min_value = 1024,
        .max_value = 65535,
        .description = "Port number",
    });

    // Set to minimum boundary
    try config.set("port", "1024");
    try std.testing.expectEqualStrings("1024", config.get("port").?);

    // Set to maximum boundary
    try config.set("port", "65535");
    try std.testing.expectEqualStrings("65535", config.get("port").?);
}

test "set integer just below min boundary fails" {
    var config = ConfigManager.init(std.testing.allocator);
    defer config.deinit();

    try config.registerParameter(.{
        .name = "port",
        .param_type = .integer,
        .default_value = "5432",
        .min_value = 1024,
        .max_value = 65535,
        .description = "Port number",
    });

    try std.testing.expectError(error.OutOfRange, config.set("port", "1023"));
}

test "set integer just above max boundary fails" {
    var config = ConfigManager.init(std.testing.allocator);
    defer config.deinit();

    try config.registerParameter(.{
        .name = "port",
        .param_type = .integer,
        .default_value = "5432",
        .min_value = 1024,
        .max_value = 65535,
        .description = "Port number",
    });

    try std.testing.expectError(error.OutOfRange, config.set("port", "65536"));
}

test "set negative integer value" {
    var config = ConfigManager.init(std.testing.allocator);
    defer config.deinit();

    try config.registerParameter(.{
        .name = "temp_buffers",
        .param_type = .integer,
        .default_value = "1024",
        .description = "Temporary buffer size",
    });

    // Should accept negative if no min bound
    try config.set("temp_buffers", "-100");
    try std.testing.expectEqualStrings("-100", config.get("temp_buffers").?);
}

test "set negative integer below min boundary fails" {
    var config = ConfigManager.init(std.testing.allocator);
    defer config.deinit();

    try config.registerParameter(.{
        .name = "offset",
        .param_type = .integer,
        .default_value = "0",
        .min_value = -10,
        .max_value = 10,
        .description = "Offset value",
    });

    try std.testing.expectError(error.OutOfRange, config.set("offset", "-11"));
}

test "set integer with leading zeros" {
    var config = ConfigManager.init(std.testing.allocator);
    defer config.deinit();

    try config.registerParameter(.{
        .name = "connections",
        .param_type = .integer,
        .default_value = "100",
        .description = "Connection count",
    });

    try config.set("connections", "0050");
    try std.testing.expectEqualStrings("0050", config.get("connections").?);
}

test "set boolean uppercase values" {
    var config = ConfigManager.init(std.testing.allocator);
    defer config.deinit();

    try config.registerParameter(.{
        .name = "feature_flag",
        .param_type = .boolean,
        .default_value = "false",
        .description = "Feature toggle",
    });

    try config.set("feature_flag", "TRUE");
    try std.testing.expectEqualStrings("TRUE", config.get("feature_flag").?);

    try config.set("feature_flag", "FALSE");
    try std.testing.expectEqualStrings("FALSE", config.get("feature_flag").?);

    try config.set("feature_flag", "ON");
    try std.testing.expectEqualStrings("ON", config.get("feature_flag").?);

    try config.set("feature_flag", "OFF");
    try std.testing.expectEqualStrings("OFF", config.get("feature_flag").?);
}

test "set boolean rejects arbitrary mixed case" {
    var config = ConfigManager.init(std.testing.allocator);
    defer config.deinit();

    try config.registerParameter(.{
        .name = "feature_flag",
        .param_type = .boolean,
        .default_value = "false",
        .description = "Feature toggle",
    });

    // These should fail - only lowercase, uppercase, and exact "1"/"0" are allowed
    try std.testing.expectError(error.InvalidType, config.set("feature_flag", "oN"));
    try std.testing.expectError(error.InvalidType, config.set("feature_flag", "oFf"));
    try std.testing.expectError(error.InvalidType, config.set("feature_flag", "TrUe"));
}

test "set boolean rejects numeric 2" {
    var config = ConfigManager.init(std.testing.allocator);
    defer config.deinit();

    try config.registerParameter(.{
        .name = "feature_flag",
        .param_type = .boolean,
        .default_value = "false",
        .description = "Feature toggle",
    });

    try std.testing.expectError(error.InvalidType, config.set("feature_flag", "2"));
}

test "set size with lowercase units" {
    var config = ConfigManager.init(std.testing.allocator);
    defer config.deinit();

    try config.registerParameter(.{
        .name = "cache_size",
        .param_type = .size,
        .default_value = "256MB",
        .description = "Cache size",
    });

    try config.set("cache_size", "512mb");
    try std.testing.expectEqualStrings("512mb", config.get("cache_size").?);

    try config.set("cache_size", "1gb");
    try std.testing.expectEqualStrings("1gb", config.get("cache_size").?);
}

test "set size with mixed case units fails" {
    var config = ConfigManager.init(std.testing.allocator);
    defer config.deinit();

    try config.registerParameter(.{
        .name = "cache_size",
        .param_type = .size,
        .default_value = "256MB",
        .description = "Cache size",
    });

    try std.testing.expectError(error.InvalidSizeFormat, config.set("cache_size", "256Mb"));
    try std.testing.expectError(error.InvalidSizeFormat, config.set("cache_size", "256mB"));
}

test "set size with zero value" {
    var config = ConfigManager.init(std.testing.allocator);
    defer config.deinit();

    try config.registerParameter(.{
        .name = "cache_size",
        .param_type = .size,
        .default_value = "256MB",
        .description = "Cache size",
    });

    try config.set("cache_size", "0");
    try std.testing.expectEqualStrings("0", config.get("cache_size").?);
}

test "set size with zero and unit" {
    var config = ConfigManager.init(std.testing.allocator);
    defer config.deinit();

    try config.registerParameter(.{
        .name = "cache_size",
        .param_type = .size,
        .default_value = "256MB",
        .description = "Cache size",
    });

    try config.set("cache_size", "0MB");
    try std.testing.expectEqualStrings("0MB", config.get("cache_size").?);
}

test "set size with large value" {
    var config = ConfigManager.init(std.testing.allocator);
    defer config.deinit();

    try config.registerParameter(.{
        .name = "cache_size",
        .param_type = .size,
        .default_value = "1MB",
        .description = "Cache size",
    });

    try config.set("cache_size", "100GB");
    try std.testing.expectEqualStrings("100GB", config.get("cache_size").?);
}

test "set size rejects unit-only string" {
    var config = ConfigManager.init(std.testing.allocator);
    defer config.deinit();

    try config.registerParameter(.{
        .name = "cache_size",
        .param_type = .size,
        .default_value = "1MB",
        .description = "Cache size",
    });

    try std.testing.expectError(error.InvalidSizeFormat, config.set("cache_size", "MB"));
}

test "set size rejects space in value" {
    var config = ConfigManager.init(std.testing.allocator);
    defer config.deinit();

    try config.registerParameter(.{
        .name = "cache_size",
        .param_type = .size,
        .default_value = "1MB",
        .description = "Cache size",
    });

    try std.testing.expectError(error.InvalidSizeFormat, config.set("cache_size", "512 MB"));
}

test "text parameter accepts empty string" {
    var config = ConfigManager.init(std.testing.allocator);
    defer config.deinit();

    try config.registerParameter(.{
        .name = "schema_search_path",
        .param_type = .text,
        .default_value = "public",
        .description = "Schema search path",
    });

    try config.set("schema_search_path", "");
    try std.testing.expectEqualStrings("", config.get("schema_search_path").?);
}

test "text parameter accepts special characters" {
    var config = ConfigManager.init(std.testing.allocator);
    defer config.deinit();

    try config.registerParameter(.{
        .name = "connection_string",
        .param_type = .text,
        .default_value = "localhost",
        .description = "Connection string",
    });

    try config.set("connection_string", "host=db.example.com:5432;user=admin;pass=p@ss!word");
    try std.testing.expectEqualStrings("host=db.example.com:5432;user=admin;pass=p@ss!word", config.get("connection_string").?);
}

test "text parameter accepts numeric strings" {
    var config = ConfigManager.init(std.testing.allocator);
    defer config.deinit();

    try config.registerParameter(.{
        .name = "version_string",
        .param_type = .text,
        .default_value = "1.0",
        .description = "Version string",
    });

    try config.set("version_string", "1.2.3-alpha");
    try std.testing.expectEqualStrings("1.2.3-alpha", config.get("version_string").?);
}

test "multiple parameters independent values" {
    var config = ConfigManager.init(std.testing.allocator);
    defer config.deinit();

    try config.registerParameter(.{
        .name = "work_mem",
        .param_type = .integer,
        .default_value = "4194304",
        .description = "Work memory",
    });

    try config.registerParameter(.{
        .name = "enable_debug",
        .param_type = .boolean,
        .default_value = "false",
        .description = "Debug mode",
    });

    try config.registerParameter(.{
        .name = "log_level",
        .param_type = .text,
        .default_value = "info",
        .description = "Log level",
    });

    try config.set("work_mem", "8388608");
    try config.set("enable_debug", "true");
    try config.set("log_level", "debug");

    try std.testing.expectEqualStrings("8388608", config.get("work_mem").?);
    try std.testing.expectEqualStrings("true", config.get("enable_debug").?);
    try std.testing.expectEqualStrings("debug", config.get("log_level").?);
}

test "reset one parameter leaves others unchanged" {
    var config = ConfigManager.init(std.testing.allocator);
    defer config.deinit();

    try config.registerParameter(.{
        .name = "param_a",
        .param_type = .integer,
        .default_value = "10",
        .description = "Parameter A",
    });

    try config.registerParameter(.{
        .name = "param_b",
        .param_type = .integer,
        .default_value = "20",
        .description = "Parameter B",
    });

    try config.set("param_a", "100");
    try config.set("param_b", "200");

    try config.reset("param_a");

    try std.testing.expectEqualStrings("10", config.get("param_a").?);
    try std.testing.expectEqualStrings("200", config.get("param_b").?);
}

test "resetAll with mixed parameter types" {
    var config = ConfigManager.init(std.testing.allocator);
    defer config.deinit();

    try config.registerParameter(.{
        .name = "int_param",
        .param_type = .integer,
        .default_value = "42",
        .description = "Integer parameter",
    });

    try config.registerParameter(.{
        .name = "bool_param",
        .param_type = .boolean,
        .default_value = "true",
        .description = "Boolean parameter",
    });

    try config.registerParameter(.{
        .name = "text_param",
        .param_type = .text,
        .default_value = "default_value",
        .description = "Text parameter",
    });

    try config.registerParameter(.{
        .name = "size_param",
        .param_type = .size,
        .default_value = "256MB",
        .description = "Size parameter",
    });

    try config.set("int_param", "999");
    try config.set("bool_param", "false");
    try config.set("text_param", "custom_value");
    try config.set("size_param", "512MB");

    try config.resetAll();

    try std.testing.expectEqualStrings("42", config.get("int_param").?);
    try std.testing.expectEqualStrings("true", config.get("bool_param").?);
    try std.testing.expectEqualStrings("default_value", config.get("text_param").?);
    try std.testing.expectEqualStrings("256MB", config.get("size_param").?);
}

test "getAll returns entries for each parameter" {
    var config = ConfigManager.init(std.testing.allocator);
    defer config.deinit();

    try config.registerParameter(.{
        .name = "alpha",
        .param_type = .integer,
        .default_value = "1",
        .description = "Alpha",
    });

    try config.registerParameter(.{
        .name = "beta",
        .param_type = .text,
        .default_value = "test",
        .description = "Beta",
    });

    try config.registerParameter(.{
        .name = "gamma",
        .param_type = .boolean,
        .default_value = "false",
        .description = "Gamma",
    });

    const all_params = try config.getAll(std.testing.allocator);
    defer std.testing.allocator.free(all_params);

    try std.testing.expectEqual(@as(usize, 3), all_params.len);

    // Verify all entries are present
    var found_alpha = false;
    var found_beta = false;
    var found_gamma = false;

    for (all_params) |entry| {
        if (std.mem.eql(u8, entry.name, "alpha")) {
            found_alpha = true;
            try std.testing.expectEqualStrings("1", entry.value);
        } else if (std.mem.eql(u8, entry.name, "beta")) {
            found_beta = true;
            try std.testing.expectEqualStrings("test", entry.value);
        } else if (std.mem.eql(u8, entry.name, "gamma")) {
            found_gamma = true;
            try std.testing.expectEqualStrings("false", entry.value);
        }
    }

    try std.testing.expect(found_alpha);
    try std.testing.expect(found_beta);
    try std.testing.expect(found_gamma);
}

test "getAll reflects current values after set" {
    var config = ConfigManager.init(std.testing.allocator);
    defer config.deinit();

    try config.registerParameter(.{
        .name = "param1",
        .param_type = .integer,
        .default_value = "100",
        .description = "Param 1",
    });

    try config.set("param1", "999");

    const all_params = try config.getAll(std.testing.allocator);
    defer std.testing.allocator.free(all_params);

    try std.testing.expectEqual(@as(usize, 1), all_params.len);
    try std.testing.expectEqualStrings("param1", all_params[0].name);
    try std.testing.expectEqualStrings("999", all_params[0].value);
}

test "get returns null for parameter not registered" {
    var config = ConfigManager.init(std.testing.allocator);
    defer config.deinit();

    try config.registerParameter(.{
        .name = "exists",
        .param_type = .integer,
        .default_value = "1",
        .description = "Exists",
    });

    try std.testing.expect(config.get("does_not_exist") == null);
    try std.testing.expect(config.get("exists") != null);
}

test "overwrite parameter value multiple times" {
    var config = ConfigManager.init(std.testing.allocator);
    defer config.deinit();

    try config.registerParameter(.{
        .name = "counter",
        .param_type = .integer,
        .default_value = "0",
        .description = "Counter",
    });

    try config.set("counter", "1");
    try std.testing.expectEqualStrings("1", config.get("counter").?);

    try config.set("counter", "2");
    try std.testing.expectEqualStrings("2", config.get("counter").?);

    try config.set("counter", "3");
    try std.testing.expectEqualStrings("3", config.get("counter").?);

    try config.reset("counter");
    try std.testing.expectEqualStrings("0", config.get("counter").?);
}

test "integer parameter without min/max bounds accepts any valid integer" {
    var config = ConfigManager.init(std.testing.allocator);
    defer config.deinit();

    try config.registerParameter(.{
        .name = "unbounded",
        .param_type = .integer,
        .default_value = "0",
        .description = "Unbounded",
    });

    try config.set("unbounded", "999999999");
    try std.testing.expectEqualStrings("999999999", config.get("unbounded").?);

    try config.set("unbounded", "-999999999");
    try std.testing.expectEqualStrings("-999999999", config.get("unbounded").?);
}

test "integer parameter with only min bound" {
    var config = ConfigManager.init(std.testing.allocator);
    defer config.deinit();

    try config.registerParameter(.{
        .name = "min_only",
        .param_type = .integer,
        .default_value = "10",
        .min_value = 5,
        .description = "Min only",
    });

    try config.set("min_only", "1000000");
    try std.testing.expectEqualStrings("1000000", config.get("min_only").?);

    try std.testing.expectError(error.OutOfRange, config.set("min_only", "4"));
}

test "integer parameter with only max bound" {
    var config = ConfigManager.init(std.testing.allocator);
    defer config.deinit();

    try config.registerParameter(.{
        .name = "max_only",
        .param_type = .integer,
        .default_value = "100",
        .max_value = 200,
        .description = "Max only",
    });

    try config.set("max_only", "-1000000");
    try std.testing.expectEqualStrings("-1000000", config.get("max_only").?);

    try std.testing.expectError(error.OutOfRange, config.set("max_only", "201"));
}

test "parseSize corner case: single digit plus unit" {
    const result = try parseSize("1KB");
    try std.testing.expectEqual(@as(i64, 1024), result);
}

test "parseSize large multiplier" {
    const result = try parseSize("1000GB");
    try std.testing.expectEqual(@as(i64, 1000 * 1024 * 1024 * 1024), result);
}
