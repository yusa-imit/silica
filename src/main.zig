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
pub const hash_index = @import("storage/hash_index.zig");
pub const gist_index = @import("storage/gist_index.zig");
pub const gin_index = @import("storage/gin_index.zig");
pub const overflow = @import("storage/overflow.zig");
pub const fsm = @import("storage/fsm.zig");
pub const fuzz = @import("storage/fuzz.zig");

// SQL modules
pub const tokenizer = @import("sql/tokenizer.zig");
pub const tokenizer_fuzz = @import("sql/tokenizer_fuzz.zig");
pub const ast = @import("sql/ast.zig");
pub const parser = @import("sql/parser.zig");
pub const parser_fuzz = @import("sql/parser_fuzz.zig");
pub const catalog = @import("sql/catalog.zig");
pub const analyzer = @import("sql/analyzer.zig");
pub const planner = @import("sql/planner.zig");
pub const optimizer = @import("sql/optimizer.zig");
pub const executor = @import("sql/executor.zig");
pub const engine = @import("sql/engine.zig");
pub const stats = @import("sql/stats.zig");
pub const selectivity = @import("sql/selectivity.zig");
pub const cost = @import("sql/cost.zig");
pub const conformance_test = @import("sql/conformance_test.zig");

// Transaction modules
pub const wal = @import("tx/wal.zig");
pub const wal_fuzz = @import("tx/wal_fuzz.zig");
pub const crash_test = @import("tx/crash_test.zig");
pub const mvcc = @import("tx/mvcc.zig");
pub const lock = @import("tx/lock.zig");
pub const vacuum = @import("tx/vacuum.zig");

// Configuration modules
pub const config_manager = @import("config/manager.zig");
pub const config_file = @import("config/file.zig");

// Replication modules
pub const replication_protocol = @import("replication/protocol.zig");
pub const replication_slot = @import("replication/slot.zig");
pub const replication_sender = @import("replication/sender.zig");
pub const replication_receiver = @import("replication/receiver.zig");
pub const replication_standby = @import("replication/standby.zig");
pub const replication_sync = @import("replication/sync.zig");
pub const replication_promotion = @import("replication/promotion.zig");
pub const replication_cascade = @import("replication/cascade.zig");
pub const replication_backup = @import("replication/backup.zig");
pub const replication_monitor = @import("replication/monitor.zig");
pub const replication_switchover = @import("replication/switchover.zig");

// Note: Server modules (wire, connection, server) are not imported here
// to avoid circular dependencies (they import "silica"). Their tests
// are run via the CLI test module which has proper module setup.

test {
    // Pull in tests from all imported modules
    std.testing.refAllDecls(@This());
}
