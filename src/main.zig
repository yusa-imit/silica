//! Silica — Production-grade embedded relational database engine written in Zig.
//!
//! A lightweight, high-performance storage engine inspired by SQLite's simplicity
//! and embeddability. Single-file database format with ACID guarantees.

const std = @import("std");

// Utility modules
pub const checksum = @import("util/checksum.zig");
pub const varint = @import("util/varint.zig");

// Storage modules
pub const page = @import("storage/page.zig");
pub const buffer_pool = @import("storage/buffer_pool.zig");
pub const btree = @import("storage/btree.zig");
pub const overflow = @import("storage/overflow.zig");
pub const fuzz = @import("storage/fuzz.zig");

// SQL modules
pub const tokenizer = @import("sql/tokenizer.zig");

test {
    // Pull in tests from all imported modules
    std.testing.refAllDecls(@This());
}
