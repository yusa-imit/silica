# Session 40 — STABILIZATION

## Mode
STABILIZATION (counter % 5 == 0)

## Completed
- Investigated Jepsen bank transfer test failure (expected 1000, found 995)
- Identified ROOT CAUSE: Architectural limitation in concurrent connection support
- Documented critical finding: Silica does NOT support concurrent connections (v0.7.0)
- Updated memory: architecture.md, debugging.md with concurrency limitations

## Root Cause Analysis

### Per-Connection Resource Isolation
Each `Database.open()` creates:
1. Separate **Buffer Pool** → isolated page cache
2. Separate **WAL instance** → CRITICAL BUG: multiple Wal objects write to same `-wal` file without sync!
3. Shared TransactionManager (correct)

### Concurrency Bugs Discovered
1. **WAL Corruption**: Interleaved writes from multiple Wal instances → corrupt frames, lost data
2. **Stale Cache**: Connection A modifies page → flushes to WAL → Connection B serves old cached copy
3. **Rollback Hazard**: Connection A's rollback truncates WAL, may discard Connection B's commits

### Failed "Quick Fix" Attempts
- Flush buffer pool after every write → Made it WORSE (995 → 956 data loss)
- Re-read from WAL on cache hit → Still failing, more corruption
- Reason: Fundamental architecture issue, not a simple bug

## Impact
- Jepsen concurrent tests are INVALID for current architecture
- Multi-connection workloads are UNSAFE
- Issue #20 symptoms (NoRows, data loss) are architectural, not bugs

## Fix Required (Milestone 26+)
1. Shared buffer pool with proper locking
2. Single WAL manager or serialized WAL writes  
3. Multi-version storage for true MVCC concurrency

## Current Status
- **Documented limitation**: Single-connection mode only
- **Issue #20**: Updated with detailed analysis
- **Tests**: Still failing (concurrent tests invalid for v0.7.0)
- **No code changes**: Reverted attempted fixes (made it worse)

## Files Changed
- `.claude/memory/architecture.md` — Added concurrency limitations section
- `.claude/memory/debugging.md` — Added Session 40 finding as Active Issue

## Commits
- `17912c9` — docs: document concurrency architecture limitation (Session 40)

## Next Priority
- Milestone 26: Implement shared buffer pool + WAL coordination
- OR: Document single-connection requirement in README/PRD
- OR: Disable/skip concurrent Jepsen tests until Milestone 26
