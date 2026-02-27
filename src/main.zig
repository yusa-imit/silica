//! Silica — Production-grade embedded relational database engine written in Zig.
//!
//! A lightweight, high-performance storage engine inspired by SQLite's simplicity
//! and embeddability. Single-file database format with ACID guarantees.

const std = @import("std");

// Utility modules
pub const checksum = @import("util/checksum.zig");
pub const varint = @import("util/varint.zig");

test {
    // Pull in tests from all imported modules
    std.testing.refAllDecls(@This());
}
