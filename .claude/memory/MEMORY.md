# Silica Project Memory

> **Note**: Detailed session-by-session history (sessions 299–453+) has moved to the
> Claude Code auto-memory system outside this repo (`~/.claude/projects/.../memory/`,
> not git-tracked). This file is kept minimal and updated at end-of-cycle per CLAUDE.md
> protocol, but treat the auto-memory system as the primary source of truth for recent
> session detail — this file lags behind by design to avoid duplicate maintenance.

## Current State (Session 507, 2026-08-27)
- **Version**: v1.0.1 (production ready, all 12 phases complete); index-only-scan phased plan completed 7/7 (session 494). #125 (SAVEPOINT/ROLLBACK undo-log) and #126 (column UNIQUE enforcement) — both CLOSED, fully fixed (commits `ca74a32`, `d24bb5e`), superseding the session-502 "in progress" note below which was stale by session 507.
- **Mode**: Feature. CI green + 0 open issues at session start. Architect-reviewed bitmap index scans (full phased plan in architecture.md) and found a real prerequisite bug: non-unique `.btree` indexes reject INSERT of duplicate column values (filed **issue #128**). Implemented phase 0a (TDD: `IndexInfo.composite_key` field, backward-compatible serialize/deserialize, commit `6400232`) — groundwork only, bug not yet fixed end-to-end (phases 0b/0c remain: composite-key building in insertIndexEntries/deleteIndexEntries, multi-row IndexScanOp).
- **Dependencies**: sailor v2.95.0, zuda v2.3.0 (bumped session 506)
- **CI**: ✅ GREEN at session start; push for commit 6400232 pending verification
- **Tests**: 4557/4580 passed, 23 skipped, 0 failed, `zig build` clean
- **Open issues**: 1 — #128 (non-unique index duplicate-value INSERT bug / bitmap-scan prerequisite), phase 0a done, 0b/0c remain
- **Next priority**: continue #128 phase 0b (engine.zig composite-key wiring in insertIndexEntries/deleteIndexEntries) since bugs take priority over new feature work; full bitmap-scan plan (5 phases after #128) documented in architecture.md for once #128 is closed

## Pattern: Maintenance Cycle
Since v1.0.0 release, sessions follow a predictable pattern:
- **STABILIZATION** (every 5th session, `.claude/session-counter % 5 == 0`): full health audit — CI, dependencies, build, tests
- **FEATURE** (other sessions): incremental improvements — sailor/zuda dependency migrations + TUI overlay wiring, SQL feature gaps, test quality
- Every cycle: CI check first, TDD (test-writer before implementation), commit+push per unit of work

## Known TODOs in Codebase (as of session 453)

### High-value but non-trivial (need architect review before attempting)
- **Index-only scan optimization** — secondary index B+Tree leaves only store the heap `row_key`, not column data, so covered queries still need a heap fetch; also needs an MVCC visibility-map equivalent. Not a quick win.
- **GiST/GIN native storage** — currently B+Tree fallback for DML; native range-query/inverted-index semantics not wired in.
- **Replication WAL sender/receiver** (`src/replication/receiver.zig`, `sender.zig`) — `connect()`, `processWalData()`, `flushWal()`, `applyWal()` are literal no-op TODO stubs that fake success; no real TCP or WAL file I/O yet. Large, networking-heavy gap.
- **MATCH_RECOGNIZE** (SQL:2016 row pattern matching) — not yet started.

### Minor
- `txid_current()` hardcoded to return 1 — needs wiring to real TM XID when session context is available.

## Project Conventions (Reinforced)
- **catch unreachable**: only with justified SAFETY comments, never in production error paths.
- **Zero warnings**: strict compilation standards enforced.
- **Test quality**: focus on meaningful validation (failure-path coverage), not coverage numbers.
- **Commit discipline**: small, focused commits with descriptive messages; `git add <specific files>`, never `-A`.
- **Memory compression**: keep this file minimal; detailed history lives in auto-memory.
