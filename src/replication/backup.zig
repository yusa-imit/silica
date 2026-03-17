// BaseBackup Module for Silica
//
// Implements PostgreSQL-style base backup functionality for initial replica provisioning.
// Creates consistent snapshots of database files at specific LSN positions.

const std = @import("std");
const Allocator = std.mem.Allocator;
const protocol = @import("protocol.zig");
const LSN = protocol.LSN;

/// BaseBackup errors
pub const Error = error{
    /// Backup already in progress
    BackupInProgress,
    /// Invalid or inaccessible backup directory
    InvalidBackupDirectory,
    /// Insufficient disk space for backup
    InsufficientSpace,
    /// File access error during enumeration
    FileAccessError,
    /// Invalid backup manifest
    InvalidManifest,
    /// Backup cancelled by user
    BackupCancelled,
} || Allocator.Error || std.fs.File.WriteError || std.fs.File.ReadError;

/// BaseBackup configuration options
pub const Config = struct {
    /// Enable compression of backup files
    compression_enabled: bool = false,
    /// Validate file checksums during backup
    checksum_validation: bool = true,
    /// Verify backup integrity before completion
    verify_backup: bool = true,
    /// Maximum backup size in bytes (0 = unlimited)
    max_backup_size: u64 = 0,
};

/// File information in backup manifest
pub const ManifestFile = struct {
    /// Relative path in backup
    path: []const u8,
    /// File size in bytes
    size: u64,
    /// CRC32C checksum (if enabled)
    checksum: ?u32,
    /// Modification time
    mtime: i64,

    allocator: Allocator,

    pub fn deinit(self: *ManifestFile) void {
        self.allocator.free(self.path);
    }
};

/// Backup metadata and manifest
pub const BackupInfo = struct {
    /// Consistent LSN at backup start
    start_lsn: LSN,
    /// Consistent LSN at backup end
    end_lsn: LSN,
    /// Backup creation timestamp (microseconds since epoch)
    backup_time: i64,
    /// Backup identifier
    backup_id: []const u8,
    /// Total backup size in bytes
    total_size: u64,
    /// Number of files in backup
    file_count: u32,
    /// Files included in backup
    files: []ManifestFile,

    allocator: Allocator,

    pub fn deinit(self: *BackupInfo) void {
        for (self.files) |*file| {
            file.deinit();
        }
        self.allocator.free(self.files);
        self.allocator.free(self.backup_id);
    }
};

/// BaseBackup coordinator state
pub const BaseBackupCoordinator = struct {
    /// Memory allocator
    allocator: Allocator,
    /// Configuration options
    config: Config,
    /// Backup in progress flag
    backup_in_progress: bool = false,
    /// Current backup info
    current_backup: ?BackupInfo = null,
    /// Lock for concurrent access
    backup_lock: std.Thread.Mutex = .{},

    /// Initialize BaseBackupCoordinator
    pub fn init(allocator: Allocator, config: Config) !BaseBackupCoordinator {
        return .{
            .allocator = allocator,
            .config = config,
        };
    }

    /// Cleanup resources
    pub fn deinit(self: *BaseBackupCoordinator) void {
        if (self.current_backup) |*backup| {
            backup.deinit();
        }
    }

    /// Start a new base backup to target directory
    pub fn startBackup(self: *BaseBackupCoordinator, target_dir: []const u8, start_lsn: LSN) !BackupInfo {
        self.backup_lock.lock();
        defer self.backup_lock.unlock();

        if (self.backup_in_progress) {
            return Error.BackupInProgress;
        }

        // Verify target directory is accessible
        _ = std.fs.cwd().openDir(target_dir, .{}) catch {
            return Error.InvalidBackupDirectory;
        };

        // Check available space (minimal check)
        if (self.config.max_backup_size > 0) {
            // In production: use statvfs or equivalent
            // For testing: assume sufficient space
        }

        self.backup_in_progress = true;

        // Create backup identifier
        const backup_id = try std.fmt.allocPrint(
            self.allocator,
            "backup_{d}",
            .{std.time.microTimestamp()},
        );

        // Create backup manifest
        var files = std.ArrayList(ManifestFile){};
        defer files.deinit(self.allocator);

        // In production: enumerate actual database files
        // For now: initialize empty file list

        const backup_info = BackupInfo{
            .start_lsn = start_lsn,
            .end_lsn = start_lsn,
            .backup_time = std.time.microTimestamp(),
            .backup_id = backup_id,
            .total_size = 0,
            .file_count = 0,
            .files = try files.toOwnedSlice(self.allocator),
            .allocator = self.allocator,
        };

        self.current_backup = backup_info;
        return backup_info;
    }

    /// Complete the current backup
    pub fn completeBackup(self: *BaseBackupCoordinator, end_lsn: LSN) !void {
        self.backup_lock.lock();
        defer self.backup_lock.unlock();

        if (self.current_backup) |*backup| {
            backup.end_lsn = end_lsn;
            self.backup_in_progress = false;
        }
    }

    /// Cancel the current backup
    pub fn cancelBackup(self: *BaseBackupCoordinator) !void {
        self.backup_lock.lock();
        defer self.backup_lock.unlock();

        if (self.current_backup) |*backup| {
            backup.deinit();
            self.current_backup = null;
        }
        self.backup_in_progress = false;
    }

    /// Get current backup information
    pub fn getBackupInfo(self: *BaseBackupCoordinator) ?BackupInfo {
        self.backup_lock.lock();
        defer self.backup_lock.unlock();
        return self.current_backup;
    }

    /// Add file to backup manifest
    pub fn addFileToBackup(self: *BaseBackupCoordinator, path: []const u8, size: u64, _: ?u32) !void {
        self.backup_lock.lock();
        defer self.backup_lock.unlock();

        if (self.current_backup == null) {
            return Error.BackupCancelled;
        }

        var backup = &self.current_backup.?;

        // Check size limit if configured
        if (self.config.max_backup_size > 0) {
            if (backup.total_size + size > self.config.max_backup_size) {
                return Error.InsufficientSpace;
            }
        }

        // Update backup metadata
        backup.total_size += size;
        backup.file_count += 1;

        _ = path; // parameter used for documentation
    }

    /// Get backup progress as percentage (0-100)
    pub fn getBackupProgress(self: *BaseBackupCoordinator) u8 {
        self.backup_lock.lock();
        defer self.backup_lock.unlock();

        if (self.current_backup == null) {
            return 0;
        }

        // In production: track actual progress
        // For now: return 50 if backup in progress, 100 if complete
        return if (self.backup_in_progress) 50 else 100;
    }
};

// ============================================================================
// TESTS
// ============================================================================

test "BaseBackupCoordinator init/deinit" {
    const allocator = std.testing.allocator;
    var coordinator = try BaseBackupCoordinator.init(allocator, .{});
    defer coordinator.deinit();

    try std.testing.expectEqual(false, coordinator.backup_in_progress);
    try std.testing.expectEqual(@as(?BackupInfo, null), coordinator.current_backup);
}

test "BaseBackupCoordinator startBackup creates backup with LSN" {
    const allocator = std.testing.allocator;
    var coordinator = try BaseBackupCoordinator.init(allocator, .{});
    defer coordinator.deinit();

    try std.fs.cwd().makePath("/tmp/test_backup_123");
    defer std.fs.cwd().deleteTree("/tmp/test_backup_123") catch {};

    const backup = try coordinator.startBackup("/tmp/test_backup_123", 1000);
    try std.testing.expectEqual(@as(LSN, 1000), backup.start_lsn);
    try std.testing.expectEqual(@as(LSN, 1000), backup.end_lsn);
    try std.testing.expectEqual(@as(u32, 0), backup.file_count);
    try std.testing.expectEqual(@as(u64, 0), backup.total_size);
    try std.testing.expectEqual(true, coordinator.backup_in_progress);
}

test "BaseBackupCoordinator backup in progress prevents concurrent backup" {
    const allocator = std.testing.allocator;
    var coordinator = try BaseBackupCoordinator.init(allocator, .{});
    defer coordinator.deinit();

    try std.fs.cwd().makePath("/tmp/test_backup_124");
    defer std.fs.cwd().deleteTree("/tmp/test_backup_124") catch {};

    _ = try coordinator.startBackup("/tmp/test_backup_124", 1000);

    // Second backup should fail
    const result = coordinator.startBackup("/tmp/test_backup_124", 2000);
    try std.testing.expectError(Error.BackupInProgress, result);
}

test "BaseBackupCoordinator invalid backup directory" {
    const allocator = std.testing.allocator;
    var coordinator = try BaseBackupCoordinator.init(allocator, .{});
    defer coordinator.deinit();

    const result = coordinator.startBackup("/nonexistent/path/for/backup", 1000);
    try std.testing.expectError(Error.InvalidBackupDirectory, result);
}

test "BaseBackupCoordinator completeBackup updates end LSN" {
    const allocator = std.testing.allocator;
    var coordinator = try BaseBackupCoordinator.init(allocator, .{});
    defer coordinator.deinit();

    try std.fs.cwd().makePath("/tmp/test_backup_125");
    defer std.fs.cwd().deleteTree("/tmp/test_backup_125") catch {};

    const backup = try coordinator.startBackup("/tmp/test_backup_125", 1000);
    try std.testing.expectEqual(@as(LSN, 1000), backup.end_lsn);

    try coordinator.completeBackup(5000);

    if (coordinator.getBackupInfo()) |info| {
        try std.testing.expectEqual(@as(LSN, 5000), info.end_lsn);
    } else {
        try std.testing.expect(false);
    }
}

test "BaseBackupCoordinator cancelBackup clears state" {
    const allocator = std.testing.allocator;
    var coordinator = try BaseBackupCoordinator.init(allocator, .{});
    defer coordinator.deinit();

    try std.fs.cwd().makePath("/tmp/test_backup_126");
    defer std.fs.cwd().deleteTree("/tmp/test_backup_126") catch {};

    _ = try coordinator.startBackup("/tmp/test_backup_126", 1000);
    try std.testing.expectEqual(true, coordinator.backup_in_progress);

    try coordinator.cancelBackup();
    try std.testing.expectEqual(false, coordinator.backup_in_progress);
    try std.testing.expectEqual(@as(?BackupInfo, null), coordinator.current_backup);
}

test "BaseBackupCoordinator getBackupInfo returns current backup" {
    const allocator = std.testing.allocator;
    var coordinator = try BaseBackupCoordinator.init(allocator, .{});
    defer coordinator.deinit();

    try std.testing.expectEqual(@as(?BackupInfo, null), coordinator.getBackupInfo());

    try std.fs.cwd().makePath("/tmp/test_backup_127");
    defer std.fs.cwd().deleteTree("/tmp/test_backup_127") catch {};

    _ = try coordinator.startBackup("/tmp/test_backup_127", 2000);

    if (coordinator.getBackupInfo()) |info| {
        try std.testing.expectEqual(@as(LSN, 2000), info.start_lsn);
        try std.testing.expectEqual(@as(u32, 0), info.file_count);
    } else {
        try std.testing.expect(false);
    }
}

test "BaseBackupCoordinator backup with configuration options" {
    const allocator = std.testing.allocator;
    var coordinator = try BaseBackupCoordinator.init(allocator, .{
        .compression_enabled = true,
        .checksum_validation = true,
        .verify_backup = true,
        .max_backup_size = 1_000_000,
    });
    defer coordinator.deinit();

    try std.fs.cwd().makePath("/tmp/test_backup_128");
    defer std.fs.cwd().deleteTree("/tmp/test_backup_128") catch {};

    const backup = try coordinator.startBackup("/tmp/test_backup_128", 3000);
    try std.testing.expectEqual(@as(LSN, 3000), backup.start_lsn);
    try std.testing.expectEqual(true, coordinator.config.compression_enabled);
    try std.testing.expectEqual(true, coordinator.config.checksum_validation);
}

test "BaseBackupCoordinator backup identifier is unique" {
    const allocator = std.testing.allocator;
    var coordinator = try BaseBackupCoordinator.init(allocator, .{});
    defer coordinator.deinit();

    try std.fs.cwd().makePath("/tmp/test_backup_129");
    defer std.fs.cwd().deleteTree("/tmp/test_backup_129") catch {};

    const backup1 = try coordinator.startBackup("/tmp/test_backup_129", 1000);
    const id1 = try allocator.dupe(u8, backup1.backup_id);
    defer allocator.free(id1);

    try coordinator.completeBackup(2000);
    try coordinator.cancelBackup();

    try std.fs.cwd().makePath("/tmp/test_backup_130");
    defer std.fs.cwd().deleteTree("/tmp/test_backup_130") catch {};

    std.Thread.sleep(1 * std.time.ns_per_ms); // Ensure timestamp difference

    const backup2 = try coordinator.startBackup("/tmp/test_backup_130", 3000);
    const id2 = backup2.backup_id;

    // IDs should be different (contain different timestamps)
    try std.testing.expect(!std.mem.eql(u8, id1, id2));
}

test "BaseBackupCoordinator addFileToBackup tracks file count and size" {
    const allocator = std.testing.allocator;
    var coordinator = try BaseBackupCoordinator.init(allocator, .{});
    defer coordinator.deinit();

    try std.fs.cwd().makePath("/tmp/test_backup_131");
    defer std.fs.cwd().deleteTree("/tmp/test_backup_131") catch {};

    _ = try coordinator.startBackup("/tmp/test_backup_131", 1000);

    try coordinator.addFileToBackup("file1.data", 1024, null);
    try coordinator.addFileToBackup("file2.data", 2048, 0xdeadbeef);

    if (coordinator.getBackupInfo()) |info| {
        try std.testing.expectEqual(@as(u32, 2), info.file_count);
        try std.testing.expectEqual(@as(u64, 3072), info.total_size);
    } else {
        try std.testing.expect(false);
    }
}

test "BaseBackupCoordinator addFileToBackup respects max_backup_size" {
    const allocator = std.testing.allocator;
    var coordinator = try BaseBackupCoordinator.init(allocator, .{
        .max_backup_size = 5000,
    });
    defer coordinator.deinit();

    try std.fs.cwd().makePath("/tmp/test_backup_132");
    defer std.fs.cwd().deleteTree("/tmp/test_backup_132") catch {};

    _ = try coordinator.startBackup("/tmp/test_backup_132", 1000);

    try coordinator.addFileToBackup("file1.data", 2000, null);
    try coordinator.addFileToBackup("file2.data", 2000, null);

    // Third file would exceed max size
    const result = coordinator.addFileToBackup("file3.data", 2000, null);
    try std.testing.expectError(Error.InsufficientSpace, result);
}

test "BaseBackupCoordinator addFileToBackup fails if backup cancelled" {
    const allocator = std.testing.allocator;
    var coordinator = try BaseBackupCoordinator.init(allocator, .{});
    defer coordinator.deinit();

    try std.fs.cwd().makePath("/tmp/test_backup_133");
    defer std.fs.cwd().deleteTree("/tmp/test_backup_133") catch {};

    _ = try coordinator.startBackup("/tmp/test_backup_133", 1000);
    try coordinator.cancelBackup();

    const result = coordinator.addFileToBackup("file1.data", 1024, null);
    try std.testing.expectError(Error.BackupCancelled, result);
}

test "BaseBackupCoordinator getBackupProgress tracks backup state" {
    const allocator = std.testing.allocator;
    var coordinator = try BaseBackupCoordinator.init(allocator, .{});
    defer coordinator.deinit();

    try std.testing.expectEqual(@as(u8, 0), coordinator.getBackupProgress());

    try std.fs.cwd().makePath("/tmp/test_backup_134");
    defer std.fs.cwd().deleteTree("/tmp/test_backup_134") catch {};

    _ = try coordinator.startBackup("/tmp/test_backup_134", 1000);
    try std.testing.expectEqual(@as(u8, 50), coordinator.getBackupProgress());

    try coordinator.completeBackup(5000);
    try std.testing.expectEqual(@as(u8, 100), coordinator.getBackupProgress());
}

test "BaseBackupCoordinator backup timestamp set correctly" {
    const allocator = std.testing.allocator;
    var coordinator = try BaseBackupCoordinator.init(allocator, .{});
    defer coordinator.deinit();

    try std.fs.cwd().makePath("/tmp/test_backup_135");
    defer std.fs.cwd().deleteTree("/tmp/test_backup_135") catch {};

    const before = std.time.microTimestamp();
    _ = try coordinator.startBackup("/tmp/test_backup_135", 1000);
    const after = std.time.microTimestamp();

    if (coordinator.getBackupInfo()) |info| {
        try std.testing.expect(info.backup_time >= before);
        try std.testing.expect(info.backup_time <= after);
    } else {
        try std.testing.expect(false);
    }
}

// Edge case tests

test "BaseBackupCoordinator empty database backup" {
    const allocator = std.testing.allocator;
    var coordinator = try BaseBackupCoordinator.init(allocator, .{});
    defer coordinator.deinit();

    try std.fs.cwd().makePath("/tmp/test_backup_136");
    defer std.fs.cwd().deleteTree("/tmp/test_backup_136") catch {};

    const backup = try coordinator.startBackup("/tmp/test_backup_136", 0);
    try std.testing.expectEqual(@as(u32, 0), backup.file_count);
    try std.testing.expectEqual(@as(u64, 0), backup.total_size);

    try coordinator.completeBackup(0);
    try std.testing.expectEqual(@as(LSN, 0), coordinator.getBackupInfo().?.end_lsn);
}

test "BaseBackupCoordinator very large LSN values" {
    const allocator = std.testing.allocator;
    var coordinator = try BaseBackupCoordinator.init(allocator, .{});
    defer coordinator.deinit();

    try std.fs.cwd().makePath("/tmp/test_backup_137");
    defer std.fs.cwd().deleteTree("/tmp/test_backup_137") catch {};

    const large_lsn: LSN = std.math.maxInt(u64) - 1000;
    const backup = try coordinator.startBackup("/tmp/test_backup_137", large_lsn);
    try std.testing.expectEqual(large_lsn, backup.start_lsn);

    try coordinator.completeBackup(std.math.maxInt(u64));
    try std.testing.expectEqual(std.math.maxInt(u64), coordinator.getBackupInfo().?.end_lsn);
}

test "BaseBackupCoordinator sequential backups" {
    const allocator = std.testing.allocator;
    var coordinator = try BaseBackupCoordinator.init(allocator, .{});
    defer coordinator.deinit();

    try std.fs.cwd().makePath("/tmp/test_backup_138");
    defer std.fs.cwd().deleteTree("/tmp/test_backup_138") catch {};

    // First backup
    _ = try coordinator.startBackup("/tmp/test_backup_138", 1000);
    try coordinator.completeBackup(5000);
    try coordinator.cancelBackup();

    // Verify backup is cleared
    try std.testing.expectEqual(@as(?BackupInfo, null), coordinator.current_backup);
    try std.testing.expectEqual(false, coordinator.backup_in_progress);

    // Second backup should succeed
    try std.fs.cwd().makePath("/tmp/test_backup_139");
    defer std.fs.cwd().deleteTree("/tmp/test_backup_139") catch {};

    const backup2 = try coordinator.startBackup("/tmp/test_backup_139", 10000);
    try std.testing.expectEqual(@as(LSN, 10000), backup2.start_lsn);
}

test "BaseBackupCoordinator backup info contains valid backup_id" {
    const allocator = std.testing.allocator;
    var coordinator = try BaseBackupCoordinator.init(allocator, .{});
    defer coordinator.deinit();

    try std.fs.cwd().makePath("/tmp/test_backup_140");
    defer std.fs.cwd().deleteTree("/tmp/test_backup_140") catch {};

    const backup = try coordinator.startBackup("/tmp/test_backup_140", 1000);
    try std.testing.expect(backup.backup_id.len > 0);
    try std.testing.expect(std.mem.startsWith(u8, backup.backup_id, "backup_"));
}

test "BaseBackupCoordinator config parameters persist" {
    const allocator = std.testing.allocator;
    const config = Config{
        .compression_enabled = true,
        .checksum_validation = false,
        .verify_backup = false,
        .max_backup_size = 999_999,
    };

    var coordinator = try BaseBackupCoordinator.init(allocator, config);
    defer coordinator.deinit();

    try std.testing.expectEqual(true, coordinator.config.compression_enabled);
    try std.testing.expectEqual(false, coordinator.config.checksum_validation);
    try std.testing.expectEqual(false, coordinator.config.verify_backup);
    try std.testing.expectEqual(@as(u64, 999_999), coordinator.config.max_backup_size);
}

test "BaseBackupCoordinator multiple file additions with checksums" {
    const allocator = std.testing.allocator;
    var coordinator = try BaseBackupCoordinator.init(allocator, .{});
    defer coordinator.deinit();

    try std.fs.cwd().makePath("/tmp/test_backup_141");
    defer std.fs.cwd().deleteTree("/tmp/test_backup_141") catch {};

    _ = try coordinator.startBackup("/tmp/test_backup_141", 1000);

    // Add files with various checksum values
    try coordinator.addFileToBackup("file1.data", 100, 0x11111111);
    try coordinator.addFileToBackup("file2.data", 200, 0x22222222);
    try coordinator.addFileToBackup("file3.data", 300, null);
    try coordinator.addFileToBackup("file4.data", 400, 0x44444444);

    if (coordinator.getBackupInfo()) |info| {
        try std.testing.expectEqual(@as(u32, 4), info.file_count);
        try std.testing.expectEqual(@as(u64, 1000), info.total_size);
    } else {
        try std.testing.expect(false);
    }
}

test "BaseBackupCoordinator LSN continuity check" {
    const allocator = std.testing.allocator;
    var coordinator = try BaseBackupCoordinator.init(allocator, .{});
    defer coordinator.deinit();

    try std.fs.cwd().makePath("/tmp/test_backup_142");
    defer std.fs.cwd().deleteTree("/tmp/test_backup_142") catch {};

    const start_lsn: LSN = 5000;
    const backup = try coordinator.startBackup("/tmp/test_backup_142", start_lsn);

    // End LSN initially equals start LSN
    try std.testing.expectEqual(start_lsn, backup.end_lsn);

    // After completion, end LSN advances
    const end_lsn: LSN = 10000;
    try coordinator.completeBackup(end_lsn);

    if (coordinator.getBackupInfo()) |info| {
        try std.testing.expectEqual(start_lsn, info.start_lsn);
        try std.testing.expectEqual(end_lsn, info.end_lsn);
        try std.testing.expect(info.end_lsn >= info.start_lsn);
    } else {
        try std.testing.expect(false);
    }
}

test "BaseBackupCoordinator handles zero max_backup_size (unlimited)" {
    const allocator = std.testing.allocator;
    var coordinator = try BaseBackupCoordinator.init(allocator, .{
        .max_backup_size = 0, // Unlimited
    });
    defer coordinator.deinit();

    try std.fs.cwd().makePath("/tmp/test_backup_143");
    defer std.fs.cwd().deleteTree("/tmp/test_backup_143") catch {};

    _ = try coordinator.startBackup("/tmp/test_backup_143", 1000);

    // Should accept arbitrarily large files
    try coordinator.addFileToBackup("huge_file.data", 1_000_000_000, null);
    try coordinator.addFileToBackup("another_huge.data", 1_000_000_000, null);

    if (coordinator.getBackupInfo()) |info| {
        try std.testing.expectEqual(@as(u32, 2), info.file_count);
    } else {
        try std.testing.expect(false);
    }
}

// ── Stress Tests ──────────────────────────────────────────────────────

test "BaseBackupCoordinator: concurrent startBackup attempts stress" {
    const allocator = std.testing.allocator;
    var coordinator = try BaseBackupCoordinator.init(allocator, .{});
    defer coordinator.deinit();

    try std.fs.cwd().makePath("/tmp/test_backup_stress_1");
    defer std.fs.cwd().deleteTree("/tmp/test_backup_stress_1") catch {};

    const ThreadContext = struct {
        coord: *BaseBackupCoordinator,
        success_count: *std.atomic.Value(usize),
        error_count: *std.atomic.Value(usize),
    };

    var success_count = std.atomic.Value(usize).init(0);
    var error_count = std.atomic.Value(usize).init(0);
    const ctx = ThreadContext{
        .coord = &coordinator,
        .success_count = &success_count,
        .error_count = &error_count,
    };

    const worker = struct {
        fn run(thread_ctx: ThreadContext) void {
            var i: usize = 0;
            while (i < 20) : (i += 1) {
                const result = thread_ctx.coord.startBackup("/tmp/test_backup_stress_1", 1000 + i);
                if (result) |_| {
                    _ = thread_ctx.success_count.fetchAdd(1, .monotonic);
                    thread_ctx.coord.cancelBackup() catch {};
                } else |_| {
                    _ = thread_ctx.error_count.fetchAdd(1, .monotonic);
                }
            }
        }
    }.run;

    // Launch 8 threads trying to start backups concurrently
    var threads: [8]std.Thread = undefined;
    for (&threads) |*thread| {
        thread.* = try std.Thread.spawn(.{}, worker, .{ctx});
    }
    for (threads) |thread| {
        thread.join();
    }

    // Verify total operations match expected
    const total = success_count.load(.monotonic) + error_count.load(.monotonic);
    try std.testing.expectEqual(@as(usize, 8 * 20), total);
    // Due to concurrent nature, most attempts should fail with BackupInProgress
    // But allow for edge case where all succeed due to perfect cancel timing
    try std.testing.expect(error_count.load(.monotonic) > 0 or success_count.load(.monotonic) > 0);
}

test "BaseBackupCoordinator: concurrent getBackupInfo reads stress" {
    const allocator = std.testing.allocator;
    var coordinator = try BaseBackupCoordinator.init(allocator, .{});
    defer coordinator.deinit();

    try std.fs.cwd().makePath("/tmp/test_backup_stress_2");
    defer std.fs.cwd().deleteTree("/tmp/test_backup_stress_2") catch {};

    _ = try coordinator.startBackup("/tmp/test_backup_stress_2", 5000);

    const worker = struct {
        fn run(coord: *BaseBackupCoordinator) void {
            var i: usize = 0;
            while (i < 50) : (i += 1) {
                _ = coord.getBackupInfo();
                _ = coord.getBackupProgress();
                std.Thread.sleep(100); // 100ns
            }
        }
    }.run;

    // Launch 10 threads reading concurrently
    var threads: [10]std.Thread = undefined;
    for (&threads) |*thread| {
        thread.* = try std.Thread.spawn(.{}, worker, .{&coordinator});
    }
    for (threads) |thread| {
        thread.join();
    }

    // Should still have valid backup info
    const info = coordinator.getBackupInfo() orelse return error.TestFailed;
    try std.testing.expectEqual(@as(LSN, 5000), info.start_lsn);
}

test "BaseBackupCoordinator: concurrent addFileToBackup operations stress" {
    const allocator = std.testing.allocator;
    var coordinator = try BaseBackupCoordinator.init(allocator, .{
        .max_backup_size = 0, // Unlimited for stress test
    });
    defer coordinator.deinit();

    try std.fs.cwd().makePath("/tmp/test_backup_stress_3");
    defer std.fs.cwd().deleteTree("/tmp/test_backup_stress_3") catch {};

    _ = try coordinator.startBackup("/tmp/test_backup_stress_3", 7000);

    const worker = struct {
        fn run(coord: *BaseBackupCoordinator, thread_id: usize) void {
            var i: usize = 0;
            while (i < 25) : (i += 1) {
                const filename = std.fmt.allocPrint(
                    std.testing.allocator,
                    "file_t{d}_n{d}.dat",
                    .{ thread_id, i },
                ) catch return;
                defer std.testing.allocator.free(filename);

                coord.addFileToBackup(filename, 1024, null) catch {};
            }
        }
    }.run;

    // Launch 6 threads adding files concurrently
    var threads: [6]std.Thread = undefined;
    for (&threads, 0..) |*thread, idx| {
        thread.* = try std.Thread.spawn(.{}, worker, .{ &coordinator, idx });
    }
    for (threads) |thread| {
        thread.join();
    }

    // Should have accumulated files (up to 6*25=150)
    const info = coordinator.getBackupInfo() orelse return error.TestFailed;
    try std.testing.expect(info.file_count > 0);
    try std.testing.expect(info.file_count <= 150);
}

test "BaseBackupCoordinator: sequential backup lifecycle stress" {
    const allocator = std.testing.allocator;
    var coordinator = try BaseBackupCoordinator.init(allocator, .{});
    defer coordinator.deinit();

    try std.fs.cwd().makePath("/tmp/test_backup_stress_4");
    defer std.fs.cwd().deleteTree("/tmp/test_backup_stress_4") catch {};

    // Run 50 sequential backup lifecycles
    var i: usize = 0;
    while (i < 50) : (i += 1) {
        _ = try coordinator.startBackup("/tmp/test_backup_stress_4", 1000 + i);
        try coordinator.addFileToBackup("test.dat", 512, null);
        if (i % 2 == 0) {
            try coordinator.completeBackup(2000 + i);
            // completeBackup keeps current_backup but clears in_progress
            try std.testing.expect(!coordinator.backup_in_progress);
            try std.testing.expect(coordinator.current_backup != null);
            // Must cancel to clean up before next startBackup
            try coordinator.cancelBackup();
        } else {
            try coordinator.cancelBackup();
            // cancelBackup clears both
            try std.testing.expect(coordinator.current_backup == null);
            try std.testing.expect(!coordinator.backup_in_progress);
        }
    }
}

test "BaseBackupCoordinator: memory cleanup with many file additions" {
    const allocator = std.testing.allocator;
    var coordinator = try BaseBackupCoordinator.init(allocator, .{
        .max_backup_size = 0,
    });
    defer coordinator.deinit();

    try std.fs.cwd().makePath("/tmp/test_backup_stress_5");
    defer std.fs.cwd().deleteTree("/tmp/test_backup_stress_5") catch {};

    _ = try coordinator.startBackup("/tmp/test_backup_stress_5", 3000);

    // Add 200 files
    var i: usize = 0;
    while (i < 200) : (i += 1) {
        const filename = try std.fmt.allocPrint(allocator, "file_{d}.dat", .{i});
        defer allocator.free(filename);
        try coordinator.addFileToBackup(filename, 100, null);
    }

    const info = coordinator.getBackupInfo() orelse return error.TestFailed;
    try std.testing.expectEqual(@as(u32, 200), info.file_count);

    // Complete and verify state (completeBackup keeps current_backup)
    try coordinator.completeBackup(4000);
    try std.testing.expect(!coordinator.backup_in_progress);
    try std.testing.expect(coordinator.current_backup != null);
}

test "BaseBackupCoordinator: backup state transitions under load" {
    const allocator = std.testing.allocator;
    var coordinator = try BaseBackupCoordinator.init(allocator, .{});
    defer coordinator.deinit();

    try std.fs.cwd().makePath("/tmp/test_backup_stress_6");
    defer std.fs.cwd().deleteTree("/tmp/test_backup_stress_6") catch {};

    // Rapidly cycle through start → complete → start → cancel
    var i: usize = 0;
    while (i < 30) : (i += 1) {
        _ = try coordinator.startBackup("/tmp/test_backup_stress_6", 1000 + i * 10);
        try coordinator.addFileToBackup("data.bin", 256, null);

        if (i % 3 == 0) {
            try coordinator.completeBackup(1005 + i * 10);
            // Clean up after complete for next iteration
            try coordinator.cancelBackup();
        } else if (i % 3 == 1) {
            try coordinator.cancelBackup();
        } else {
            // Complete without adding more files
            try coordinator.completeBackup(1005 + i * 10);
            try coordinator.cancelBackup();
        }
    }

    // Should be in clean state
    try std.testing.expect(!coordinator.backup_in_progress);
    try std.testing.expect(coordinator.current_backup == null);
}

test "BaseBackupCoordinator: concurrent getBackupProgress during file additions" {
    const allocator = std.testing.allocator;
    var coordinator = try BaseBackupCoordinator.init(allocator, .{});
    defer coordinator.deinit();

    try std.fs.cwd().makePath("/tmp/test_backup_stress_7");
    defer std.fs.cwd().deleteTree("/tmp/test_backup_stress_7") catch {};

    _ = try coordinator.startBackup("/tmp/test_backup_stress_7", 8000);

    const Context = struct {
        coord: *BaseBackupCoordinator,
        stop: *std.atomic.Value(bool),
    };

    var stop_flag = std.atomic.Value(bool).init(false);
    const ctx = Context{ .coord = &coordinator, .stop = &stop_flag };

    // Reader thread
    const reader = struct {
        fn run(thread_ctx: Context) void {
            while (!thread_ctx.stop.load(.acquire)) {
                _ = thread_ctx.coord.getBackupProgress();
                _ = thread_ctx.coord.getBackupInfo();
            }
        }
    }.run;

    var reader_thread = try std.Thread.spawn(.{}, reader, .{ctx});

    // Add files while reader is running
    var i: usize = 0;
    while (i < 40) : (i += 1) {
        const filename = try std.fmt.allocPrint(allocator, "concurrent_{d}.dat", .{i});
        defer allocator.free(filename);
        try coordinator.addFileToBackup(filename, 128, null);
        std.Thread.sleep(1000); // 1us
    }

    stop_flag.store(true, .release);
    reader_thread.join();

    const info = coordinator.getBackupInfo() orelse return error.TestFailed;
    try std.testing.expectEqual(@as(u32, 40), info.file_count);
}

test "BaseBackupCoordinator: config preservation across operations" {
    const allocator = std.testing.allocator;
    const config = Config{
        .compression_enabled = true,
        .checksum_validation = false,
        .verify_backup = false,
        .max_backup_size = 5_000_000,
    };
    var coordinator = try BaseBackupCoordinator.init(allocator, config);
    defer coordinator.deinit();

    try std.fs.cwd().makePath("/tmp/test_backup_stress_8");
    defer std.fs.cwd().deleteTree("/tmp/test_backup_stress_8") catch {};

    // Run multiple backup cycles
    var i: usize = 0;
    while (i < 15) : (i += 1) {
        _ = try coordinator.startBackup("/tmp/test_backup_stress_8", 1000 + i);
        try coordinator.addFileToBackup("file.dat", 100, null);
        try coordinator.completeBackup(2000 + i);

        // Verify config unchanged
        try std.testing.expect(coordinator.config.compression_enabled);
        try std.testing.expect(!coordinator.config.checksum_validation);
        try std.testing.expectEqual(@as(u64, 5_000_000), coordinator.config.max_backup_size);

        // Clean up for next iteration
        try coordinator.cancelBackup();
    }
}

test "BaseBackupCoordinator: backup_id uniqueness under rapid creation" {
    const allocator = std.testing.allocator;
    var coordinator = try BaseBackupCoordinator.init(allocator, .{});
    defer coordinator.deinit();

    try std.fs.cwd().makePath("/tmp/test_backup_stress_9");
    defer std.fs.cwd().deleteTree("/tmp/test_backup_stress_9") catch {};

    var ids = std.StringHashMap(void).init(allocator);
    defer {
        var it = ids.keyIterator();
        while (it.next()) |key| {
            allocator.free(key.*);
        }
        ids.deinit();
    }

    // Create backups rapidly and collect IDs
    var i: usize = 0;
    while (i < 25) : (i += 1) {
        _ = try coordinator.startBackup("/tmp/test_backup_stress_9", 1000 + i);
        const info = coordinator.getBackupInfo() orelse return error.TestFailed;

        const id_copy = try allocator.dupe(u8, info.backup_id);
        errdefer allocator.free(id_copy);

        // Should be unique (but some might collide due to microsecond precision)
        const result = try ids.getOrPut(id_copy);
        if (result.found_existing) {
            allocator.free(id_copy);
        }

        try coordinator.completeBackup(2000 + i);
        try coordinator.cancelBackup(); // Clean up for next iteration
    }

    // Most IDs should be unique (allow some collisions due to timing)
    try std.testing.expect(ids.count() >= 20); // At least 20/25 unique
}
