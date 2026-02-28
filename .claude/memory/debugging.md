# Silica — Debugging Notes

> Record bugs encountered, root causes, and solutions here.

## Known Issues
- (none currently)

## Recently Fixed Bugs

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
