# Session 34 Summary - FEATURE MODE

## Mode
FEATURE (counter: 34, not divisible by 5)

## What Was Done

### CI Failure Investigation
- **Task**: Fix failing SERIALIZABLE SSI tests (NoRows errors, data loss)
- **Approach**: Deep dive into MVCC implementation, visibility rules, transaction abort handling
- **Discovery**: Root cause identified - UPDATE's delete+insert pattern breaks MVCC
  
### Root Cause Analysis
1. **UPDATE Implementation** (engine.zig:2915-2919):
   ```zig
   tree.delete(item.key) catch {};  // Deletes old version
   tree.insert(item.key, item.value) catch {};  // Inserts new version
   ```

2. **The Problem**:
   - Transaction T1 UPDATEs row: delete(K) + insert(K, V_new)
   - T1 ABORTs
   - Old version: GONE (physically deleted from BTree)
   - New version: INVISIBLE (xmin=aborted_xid)
   - Result: NoRows error!

3. **Why It Happens**:
   - BTree doesn't support version chains (no multiple versions per key)
   - WAL rollback truncates pending writes, but in-memory BTree state unchanged
   - BufferPool is per-connection → other connections don't see rollback
   - Attempted fix (discardDirtyPages) failed - BTree structure already modified

### Attempted Fixes
1. ❌ `BufferPool.discardDirtyPages()` - reloads pages from disk after rollback
   - Problem: BTree structure already modified in-memory, reloading raw pages doesn't help
2. ❌ Shared BufferPool registry - too complex for quick fix
3. ❌ BTree.update/upsert method - BTree architecture doesn't support overwrites

### Outcome
- **Issue #20 created**: Documents root cause and solution approaches
- **Status**: CI still RED - requires architectural changes
- **Blocker**: SERIALIZABLE isolation unreliable until MVCC properly implemented

## Files Changed
- None committed (reverted incomplete fixes)

## Tests
- Lost update test: STILL FAILING (expected 100, got 50)
- Bank transfer test: STILL FAILING (NoRows errors)
- Write skew test: STILL FAILING

## Next Priority
**HIGH PRIORITY**: Fix Issue #20 - requires one of:
1. Implement version chains in BTree
2. Add undo logging for transaction rollback
3. Implement shared BufferPool (like SharedTmRegistry)
4. MVCC-aware UPDATE that preserves old versions

## Lessons Learned
- DELETE+INSERT pattern incompatible with MVCC without version chains
- Per-connection BufferPools cause isolation violations
- Quick fixes impossible - need proper MVCC infrastructure
- BTree-based storage requires rethinking for true MVCC

## Time Spent
~2 hours of investigation and attempted fixes
