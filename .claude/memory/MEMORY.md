# Silica Project Memory

> **Note**: Detailed session-by-session history (sessions 299–453+) has moved to the
> Claude Code auto-memory system outside this repo (`~/.claude/projects/.../memory/`,
> not git-tracked). This file is kept minimal and updated at end-of-cycle per CLAUDE.md
> protocol, but treat the auto-memory system as the primary source of truth for recent
> session detail — this file lags behind by design to avoid duplicate maintenance.

## Current State (Session 502, 2026-08-25)
- **Version**: v1.0.1 (production ready, all 12 phases complete); index-only-scan phased plan completed 7/7 (session 494)
- **Mode**: Feature — continued in-progress bug fix (issue #125), per protocol bugs take priority over new feature work
- **Dependencies**: sailor v2.94.4, zuda v2.2.0 (last reverified session 495)
- **CI**: ✅ GREEN at session start
- **Tests**: 4536/4558 passed, 22 skipped, 0 failed, `zig build` clean
- **Open issues**: 2 — #125 SAVEPOINT rollback-after-commit bug (physical undo log fix, 7/8 steps done as of this session, see debugging.md), #126 column-level UNIQUE constraint never enforced (not yet fixed, see debugging.md)

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
