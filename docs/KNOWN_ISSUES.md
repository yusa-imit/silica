# Known Issues

## Current Known Issues

**Status**: ✅ No critical known issues as of v1.0.1

All major bugs have been resolved. The following issues were previously documented and have been fixed:

---

## Resolved Issues

### Multi-Row INSERT DuplicateKey Bug (GitHub #1) — ✅ RESOLVED

**Status**: ✅ Fixed in v0.4.0 (Closed: 2026-03-02)
**Severity**: High
**Affects**: Multi-row INSERT operations

#### Description
Creating 2+ tables and inserting many rows caused `BTreeError.DuplicateKey` errors. The issue was related to:
- `findNextRowKey()` implementation
- `updateTableRootPage()` buffer pool integration
- Buffer pool cache staleness

#### Resolution
Fixed by improving B+Tree root page tracking and buffer pool consistency during catalog updates.

---

### MVCC Concurrent UPDATE Limitation (GitHub #20) — ✅ RESOLVED

**Status**: ✅ Fixed in v0.7.0 (Closed: 2026-03-27)
**Severity**: Medium
**Affects**: SERIALIZABLE isolation with concurrent UPDATEs

#### Description
Silica's original UPDATE implementation used a delete+insert pattern which caused visibility issues during concurrent operations. When a transaction:
1. UPDATEs a row (delete old + insert new)
2. ABORTs or is in-flight

The old row was gone (physically deleted from B+Tree) and the new row was invisible (xmin=uncommitted XID). Result: NoRows errors!

#### Resolution
Implemented proper MVCC with:
- **Version Chains**: Store multiple tuple versions per key
- **Delayed Deletion**: Mark tuples deleted (`xmax` set) but keep in B+Tree until VACUUM
- **MVCC-Aware UPDATE**: In-place modification preserving old versions for concurrent readers
- **Shared Transaction Manager**: Coordination across connections

All SERIALIZABLE isolation tests now pass, including:
- Bank transfer atomicity tests
- Lost update prevention
- Write skew detection
- Dirty read prevention

---

## Deferred Enhancements (Non-Blocking)

### GIN Index Posting Tree — Single-Page Limit (GitHub #54 — Partially Resolved)

**Status**: Partially resolved (basic posting tree implemented)
**Severity**: Low
**Affects**: GIN indexes with very high-cardinality keys (>509 rows per key on 4096-byte pages)

#### Description
GIN indexes now support posting trees for keys appearing in >16 rows (up to ~509 rows per 4096-byte page). The posting tree is a flat sorted array of u64 tuple IDs stored on a single page. When the posting tree page fills up, `error.PageFull` is returned.

#### Remaining Limitation
- Single-page posting tree only — does not support multi-page (B+Tree) posting trees
- Max ~509 tuples per posting tree on 4096-byte pages (`(page_size - 20) / 8`)
- Workaround for >509 occurrences: Use regular B+Tree index

#### Current Behavior (as of Session 316)
- Keys appearing in ≤16 rows: inline posting list (fast, in entry page)
- Keys appearing in 17–509 rows: posting tree page (single sorted page)
- Keys appearing in >509 rows: `error.PageFull` on insert

#### Next Steps
Multi-page posting trees will be implemented in a future release for very high-cardinality scenarios.

---

### Crash Test Infrastructure

**Status**: Test infrastructure enhancement
**Severity**: Low
**Affects**: Crash recovery testing

#### Description
Some crash recovery tests require crash injection infrastructure (torn page simulation, mid-checkpoint crashes, etc.) that is not yet implemented.

#### Impact
- 7 tests are skipped pending crash simulation infrastructure
- Manual crash recovery testing has been performed
- WAL recovery is functional and verified through other tests

#### Next Steps
Crash injection infrastructure will be added in future milestones to enable comprehensive crash recovery testing.

---

## Reporting New Issues

If you encounter any bugs or unexpected behavior, please report them at:
https://github.com/yusa-imit/silica/issues

Include:
- Silica version (`silica --version`)
- Zig version (`zig version`)
- Operating system and architecture
- Minimal reproduction case (SQL statements or code)
- Expected vs actual behavior
- Error messages or stack traces
