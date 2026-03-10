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

### CI Failure: net.Stream.Writer incompatibility (March 11, 2026 - IN PROGRESS)

**Symptom**: Build fails on CI (Linux) but works locally (macOS) with error:
```
src/server/wire.zig:40:15: error: no field or member function named 'writeAll' in 'net.Stream.Writer__struct_38627'
```

**Root Cause**:
- Zig 0.15's `net.Stream.writer(&buffer)` returns `Io.Writer` type on Linux
- `Io.Writer` doesn't have standard `std.io.Writer` methods (`writeByte`, `writeInt`, `writeAll`)
- `wire.zig` expects writers with these standard methods (uses `anytype` parameter)
- Tests work because they use `std.ArrayList(u8).writer()` which DOES have standard methods
- macOS and Linux have different `net.Stream.Writer` implementations

**Solution**:
Don't use `net.Stream.writer()`. Follow test pattern:
1. Use `std.ArrayList(u8)` for buffering
2. Call `.writer()` on ArrayList → standard writer
3. Pass to wire functions
4. Use `stream.writeAll(buf.items)` to send
5. For reading: `stream.read()` + `std.io.fixedBufferStream()`

**Example**:
```zig
var buf = std.ArrayList(u8).init(allocator);
defer buf.deinit();
const writer = buf.writer();  // Standard writer

try message.write(writer);
try stream.writeAll(buf.items);
```

**Files to Fix**:
- `src/server/server.zig`: `processMessages()` needs ArrayList buffer refactor
- Last commit (137b22c) tried `writeByte` helper but wrong approach

**Status**: Root cause identified, fix in progress

## Recently Fixed Bugs

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
