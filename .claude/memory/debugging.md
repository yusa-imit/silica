# Silica — Debugging Notes

> Record bugs encountered, root causes, and solutions here.

## Known Issues

### GitHub Issue #1: DuplicateKey Bug (RESOLVED)
- **Symptom**: `BTreeError.DuplicateKey` in OR REPLACE and ALTER operations for functions/triggers
- **Root Cause**: Catalog code checked if key existed but didn't delete before inserting, causing B+Tree to reject duplicate
- **Impact**: All OR REPLACE operations and ALTER TRIGGER operations failed
- **Fix (c25f4a0)**: Delete existing entry before insert in createFunction/createTrigger (when or_replace=true) and alterTrigger
- **Tests**: Re-enabled 4 previously disabled tests, all 1618 tests now passing
- **Note**: Original issue #1 had multiple symptoms. Buffer pool cache staleness may still affect multi-table INSERT (different root cause)

## Active Issues

(None)

## Recently Fixed Bugs

### GiST Enum Value Missing from Catalog (dd1600d, March 20, 2026)
- **Symptom**: CI build failure: `enum 'sql.catalog.IndexType' has no member named 'gist'`
- **Root Cause**: Commit 5ca06fc added GiST index implementation with references to `IndexType.gist` in engine.zig, but the enum value `gist = 2` was added to catalog.zig locally and never committed
- **Impact**: Build passed locally due to uncommitted catalog.zig change, but failed on CI with clean checkout
- **Fix (2 commits)**:
  1. 360e003: Added .gist cases to switch statements in engine.zig and executor.zig (but CI still failed because enum didn't exist)
  2. dd1600d: Committed the missing `gist = 2` enum value to catalog.zig
- **Detection**: Build succeeded locally, failed on CI — `zig build test` doesn't catch uncommitted dependencies
- **Prevention**: Always run `git status` before committing to verify no uncommitted changes that the commit depends on
- **Lesson**: When adding a new enum value + its usages, commit them TOGETHER in same commit to prevent broken CI

### Wire Protocol Integer Overflow Security Vulnerabilities (8b5f54d, March 11, 2026)
- **Symptom**: Fuzz tests caused panics/crashes (signal 6) in Parse.parse and Bind.parse
- **Root Cause**: Unchecked `@intCast()` operations on i16/i32 counts read from untrusted wire protocol data
  - Parse.parse: `@intCast(param_count)` panics if param_count is negative
  - Bind.parse: `@intCast(format_count)`, `@intCast(param_count)`, `@intCast(result_format_count)` all vulnerable
  - Bind.parse: `@intCast(val_len)` panics on negative values (except -1 which means NULL)
- **Security Impact**: Malicious clients could crash server with malformed messages
- **Fix**: Added validation checks before all @intCast operations:
  ```zig
  const count = std.mem.readInt(i16, ...);
  if (count < 0) return error.InvalidMessage;
  const safe_count = @intCast(count); // Now guaranteed non-negative
  ```
- **Fuzz Tests**: Added wire_fuzz.zig with 11 tests covering all message types
- **Lesson**: ALWAYS validate integer values from untrusted input before @intCast
- **Rule**: Fuzz testing wire protocol parsers is MANDATORY for security

### CI Failure: net.Stream.Writer incompatibility (763c70e + 6 follow-ups, March 11, 2026)
- **Symptom**: Build fails on CI (Linux) but works locally (macOS) with `no field or member function named 'writeAll'`
- **Root Cause**: `net.Stream.writer(&buffer)` returns platform-specific types without standard writer methods (writeByte, writeInt, writeAll)
- **Main Fix** (763c70e):
  - Writing: Use `ArrayListUnmanaged` buffer → `.writer(allocator)` → `stream.writeAll(buf.items)`
  - Reading: Wrap stream with `GenericReader(net.Stream, ReadError, read_fn)` to get standard reader interface
  - wire.zig: Use `writer.writeByte()` directly instead of custom helper (14 call sites)
  - server.zig: Manual message parsing based on `msg_type` byte, call individual .parse() methods
- **Follow-up Fixes** (6 commits to fix API mismatches revealed by CI):
  - 3783a9b: Added Value.array variant in connection.valueToText (PostgreSQL {val1,val2} format)
  - e91042a: Fixed rows_affected optional binding (u64 not ?u64 in getCommandTag)
  - 708905d: Added Execute/Close message types to wire.zig (were referenced but not implemented)
  - 2a27015: Fixed ErrorResponse construction (uses .fields array, not direct .severity/.code/.message)
  - connection.zig: Updated QueryResult API (use .rows iterator not .columns, Value.null_value variant)
- **Closed**: Issue #2
- **Lesson**: Cross-platform issues may hide API mismatches - comprehensive CI coverage essential

### AST Exhaustive Switch Statements (8b0ae6d)
- **Symptom**: CI build failure after adding `create_function` and `drop_function` AST nodes
- **Cause**: Exhaustive switch statements in `analyzer.zig`, `planner.zig`, and `cli.zig` didn't handle new statement types
- **Fix**: Added empty cases in analyzer (no analysis needed yet), planner (returning `planTransaction()` as these are handled early in engine), and CLI (`printStmtInfo` now formats function statements)
- **Lesson**: When adding new AST node types, always grep for all switch statements on `ast.Stmt` to find exhaustive switches

### MVCC Aborted Transaction Visibility (c42d358)
- **Symptom**: Rows from aborted transactions visible in subsequent READ COMMITTED transactions
- **Cause**: `isTupleVisible` fell back to `snapshot.isVisible()` when hint flags absent. Snapshot cannot distinguish committed from aborted — both look "not active"
- **Fix**: Added `isTupleVisibleWithTm()` that consults `TransactionManager.isAborted()` when hint flags are not set. MvccContext now carries `tm: ?*TransactionManager`

### MVCC REPEATABLE READ Double-Free Segfault (c42d358)
- **Symptom**: Segfault in `Snapshot.deinit()` when committing a REPEATABLE READ transaction
- **Cause**: `TransactionManager.commit()` frees the snapshot's `active_xids`. Then `TransactionContext.deinit()` tries to free the same allocation via its copy of the Snapshot struct
- **Fix**: `commitTransaction`/`rollbackTransaction` sets `self.current_txn.?.snapshot = null` BEFORE calling `tm.commit()`/`tm.abort()` to prevent double-free

## Solved Problems

### Zig 0.15 Build API Change
- **Symptom**: `addStaticLibrary` not found on `Build`
- **Cause**: Zig 0.15 replaced `addStaticLibrary` with `addLibrary(.{.linkage = .static})`
- **Fix**: Use `b.addModule()` + `b.addLibrary(.{.root_module = mod, .linkage = .static})`
- **Also**: `addTest` now uses `.root_module` instead of `.root_source_file`

### Zig 0.15 CRC32 API Change
- **Symptom**: `Crc32WithPoly(.Castagnoli)` not found
- **Cause**: Zig 0.15 uses named types instead of polymorphic constructors
- **Fix**: Use `std.hash.crc.Crc32Iscsi` (CRC32C = Castagnoli = iSCSI)

### build.zig.zon Fingerprint Required
- **Symptom**: `missing top-level 'fingerprint' field`
- **Cause**: Zig 0.14+ requires a fingerprint in build.zig.zon
- **Fix**: Add the suggested fingerprint value from the error message

## Common Zig Pitfalls for Database Development
- `std.testing.allocator` detects memory leaks — always use in tests
- File operations need explicit error handling for ENOSPC, EACCES
- Alignment matters for mmap — pages must be aligned to page_size
- `@memcpy` does not handle overlapping regions — use `@memmove` for in-place operations
- Integer overflow in page number arithmetic — use `std.math.add` for checked arithmetic
- Zig 0.15 is stricter about `var` vs `const` — use `const` when variable is never mutated
- `.name = .@"silica"` is NOT needed for simple identifiers — use `.name = .silica`

<!-- Add new debugging notes above this line -->
