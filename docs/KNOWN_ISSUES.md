# Known Issues

## Multi-Row INSERT DuplicateKey Bug (GitHub #1)

**Status**: Active, workaround in place
**Severity**: High
**Affects**: Multi-row INSERT operations

### Description
Creating 2+ tables and inserting many rows causes `BTreeError.DuplicateKey` errors. The issue is related to:
- `findNextRowKey()` implementation
- `updateTableRootPage()` buffer pool integration
- Buffer pool cache staleness

### Affected Operations
- Multi-row INSERT statements (≥2 rows)
- Single-column NULL INSERT on fresh tables
- JSON/JSONB engine integration tests (disabled in commit 8a0fa13)

### Workaround
Tests that trigger this bug are temporarily disabled using `return error.SkipZigTest;`.

### Resolution Plan
This bug will be addressed in future milestones. For now, affected tests are skipped to keep CI green.

## Commit 0617467 Reverted

Commit `0617467` ("feat: implement JSON validation for JSON/JSONB types") introduced test failures due to the DuplicateKey bug. The commit added JSON engine integration tests with multi-row INSERTs which triggered the existing bug.

**Reverted at**: Commit 8a0fa13
**Reason**: CI failure in stabilization mode
**Impact**: JSON validation implementation will be reintroduced in a future commit without the problematic engine tests.

### What Was Lost
- `Value.validateJson()` function (RFC 8259 parser)
- JSON/JSONB validation in CAST operations
- 4 unit tests for JSON validation (pure functions, no DB access)

### What Was Kept
- JSON/JSONB AST, catalog, parser integration (from earlier commits)
- TUI/CLI JSON type formatting
- All other Milestone 11 progress

### Next Steps
- Fix the root cause of the DuplicateKey bug
- Re-implement JSON validation without engine tests
- Add proper integration tests once bug is fixed
