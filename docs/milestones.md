# silica — Milestones

## Current Status

- **Latest tagged release**: v0.3.0 (Phase 3: WAL & Basic Transactions)
- **Current development**: Phase 10 — Advanced Optimization (Milestone 20 complete, Milestone 21 in progress)
- **Tests**: 2301/2304 passing, 3 skipped (EXPLAIN/EXPLAIN ANALYZE complete with 4 integration tests; cost-based join selection infrastructure complete)
- **Branch**: `main`
- **Blockers**: zuda migrations blocked until zuda releases target modules
- **Known bugs**: #3 (Flaky AutoVacuumDaemon test, currently passing)

> **Note**: Phases 4-8 were completed iteratively without tagged releases. All work is on `main` branch. Git tags for v0.4.0+ will be created when appropriate release points are determined.

---

## Active Milestones

### Milestone 19: Replication Operations (Phase 9)

- [x] Synchronous replication mode (19A complete)
- [x] Replica promotion (pg_promote equivalent) (19B complete)
- [x] Cascading replication (19C complete)
- [x] Base backup (pg_basebackup equivalent) (19D complete)
- [x] Replication monitoring (pg_stat_replication equivalent) (19E complete)
- [x] Replication lag metrics and alerting hooks (19F complete)
- [x] Switchover procedure: controlled primary/replica swap (19G complete — 25 tests)

### Milestone 20: Statistics & Cost Model (Phase 10) ✅ COMPLETE

- [x] ANALYZE command (20A complete — executor implemented with table/column stats collection)
- [x] Histograms for column statistics (20B complete — equi-depth histograms with ~10 buckets, 9 tests)
- [x] Selectivity estimation (20C complete — SelectivityEstimator with equality/range/NULL/IN/LIKE/logical predicates, 25 tests)
- [x] I/O + CPU cost model (20D complete — CostEstimator with seq/index scan, join, sort, aggregate costs; 23 tests)

### Milestone 21: Advanced Optimization (Phase 10)

- [x] DP join ordering (21A partial — simplified two-table join reordering; multi-way joins deferred)
- [x] Hash/merge join selection (21B partial — HashJoinOp/MergeJoinOp executors + cost-based selection infrastructure complete; TEMPORARILY DISABLED until HashJoinOp join key extraction fixed [hardcoded to first column, needs ON condition parsing]; 5 optimizer tests)
- [ ] Subquery decorrelation
- [ ] Index-only scans
- [x] EXPLAIN ANALYZE (21C complete — EXPLAIN and EXPLAIN ANALYZE syntax, plan text formatting via formatPlan(), arena-based memory management; runtime statistics collection deferred to future; 4 integration tests)

---

## Upcoming Milestones

### Phase 11: Additional Index Types

**Milestone 22**: Hash, GiST, GIN Indexes
- `CREATE INDEX CONCURRENTLY`, bitmap index scans

### Phase 12: Production Readiness

**Milestone 23**: Operational Tools
- `EXPLAIN ANALYZE`, `VACUUM`, monitoring views, config system

**Milestone 24**: Testing & Certification
- TPC-C/TPC-H benchmarks, jepsen-style testing, fuzz campaign, SQL conformance

**Milestone 25**: Documentation & Packaging
- API reference, ops guide, SQL reference, system packages

---

## Completed Milestones

| Phase | Milestone | Name | Release | Summary |
|-------|-----------|------|---------|---------|
| Phase 1 | 1-2 | Storage Foundation | v0.1.0 | Page Manager, B+Tree, Buffer Pool, Overflow, CRC32C, Varint |
| Phase 2 | 3-4 | SQL Core | v0.2.0 | Tokenizer, Parser, AST, Analyzer, Catalog, Planner, Optimizer, Executor, Engine, secondary indexes |
| Phase 3 | 5 | WAL & Basic Transactions | v0.3.0 | WAL frame writer, commit, rollback, checkpoint, recovery, buffer pool WAL routing |
| Phase 4 | 6 | MVCC Core | — | Tuple versioning (xmin, xmax), transaction ID management, snapshot isolation, visibility rules, READ COMMITTED / REPEATABLE READ, row-level locking, concurrent writer conflict detection |
| Phase 4 | 7 | VACUUM & SSI | — | Dead tuple reclamation, auto-vacuum, free space map, SERIALIZABLE via SSI (rw-antidependency tracking), deadlock detection, savepoints |
| Phase 5 | 8 | Views & CTEs | — | Views (regular, materialized, updatable), CTEs (WITH, WITH RECURSIVE), set operations (UNION/INTERSECT/EXCEPT), DISTINCT ON |
| Phase 5 | 9 | Window Functions | — | ROW_NUMBER, RANK, DENSE_RANK, LAG, LEAD, FIRST_VALUE, LAST_VALUE, frame specs (ROWS/RANGE/GROUPS), WindowAgg operator |
| Phase 5 | 10 | Advanced Data Types | — | DATE/TIME/TIMESTAMP/INTERVAL, NUMERIC/DECIMAL, UUID, SERIAL, ARRAY, ENUM, domain types, type coercion matrix |
| Phase 6 | 11 | JSON/JSONB | — | Binary storage, operators (->/->>/@>/?), functions, GIN index, SQL/JSON path |
| Phase 6 | 12 | Full-Text Search | — | TSVECTOR/TSQUERY, @@ operator, ranking, GIN index, text search configs |
| Phase 7 | 13 | Stored Functions | — | SFL (Silica Function Language), scalar & set-returning, volatility categories |
| Phase 7 | 14 | Triggers | — | Row/statement-level, BEFORE/AFTER/INSTEAD OF, OLD/NEW references, WHEN conditions |
| Phase 8 | 15 | Wire Protocol v3 | — | Simple & extended query protocol, prepared statements, COPY, TLS |
| Phase 8 | 16 | Server & Connection Mgmt | — | Async I/O event loop, session state, authentication (SCRAM-SHA-256), `silica server` CLI |
| Phase 8 | 17 | Authorization (RBAC) | — | Roles, GRANT/REVOKE, row-level security (RLS), information_schema |
| Phase 9 | 18 | WAL Sender/Receiver | — | Replication protocol (18A), replication slot management (18B), WAL sender process (18C), WAL receiver process (18D), hot standby coordinator (18E) |

### Closed Issues

| # | Title | Closed |
|---|-------|--------|
| #2 | CI: net.Stream.Writer incompatibility on Linux | 2026-03-11 |
| #1 | DuplicateKey error when inserting rows across multiple tables | 2026-03-02 |

### Open Issues

| # | Title | Labels | Status |
|---|-------|--------|--------|
| #3 | Flaky test: AutoVacuumDaemon — inserts only never trigger vacuum | bug | Open |
| #4 | feat: migrate to zuda v1.0 for B+Tree and LRU cache | enhancement, from:zuda | Blocked on zuda |
| #5 | feat: migrate data structures to zuda v1.0 | enhancement, from:zuda | Blocked on zuda |

---

## Milestone Establishment Process

12 phases, 25 milestones total. Detailed checklists in `docs/PRD.md` Section 11.

Dependency order: Storage -> SQL -> Transaction(MVCC) -> Catalog(Views/Triggers) -> Server -> Replication

---

## Dependency Migration Tracking

### Sailor Library

- **Current in silica**: v1.16.0
- **Latest available**: v1.16.0
- **Repo**: https://github.com/yusa-imit/sailor

| Version | Features | Status | Notes |
|---------|----------|--------|-------|
| v0.1.0 | arg, color | DONE | CLI entry point |
| v0.2.0 | REPL + fmt | DONE | Interactive SQL shell |
| v0.3.0 | fmt (output modes) | DONE | table/csv/json/jsonl/plain |
| v0.4.0 | tui | DONE | TUI database browser (Input/StatusBar workaround for sailor#4) |
| v0.5.0 | advanced widgets | PARTIAL | Schema sidebar done; Tree, TextArea, Dialog, Notification pending |
| v1.0.0 | production ready | PARTIAL | Dependency updated to v1.1.0; theme/animation/refactoring items pending |
| v1.0.3 | bug fix (Tree widget) | DONE | Included in v1.1.0 upgrade |
| v1.1.0 | accessibility & i18n | DONE | Unicode width, RTL, keyboard nav; CJK testing pending |
| v1.2.0 | layout & composition | READY | Grid, ScrollView, overlay, split panes, responsive breakpoints |
| v1.3.0 | performance & DX | READY | RenderBudget, LazyBuffer, EventBatcher, DebugOverlay, ThemeWatcher |
| v1.4.0 | advanced input & forms | READY | Form, Select, Checkbox, RadioGroup, Validators, Input masks |
| v1.5.0 | state management & testing | READY | MockTerminal, snapshot tests, event bus, command pattern |
| v1.6.0 | data visualization | READY | ScatterPlot, Histogram, TimeSeriesChart, Heatmap, PieChart |
| v1.6.1 | bug fix (PieChart overflow) | READY | Integer overflow fix for data viz widgets |
| v1.7.0 | advanced layout & rendering | READY | FlexBox, viewport clipping, shadow effects, layout caching |
| v1.8.0 | network & async | READY | HttpClient, WebSocket, AsyncEventLoop, TaskRunner, LogViewer |
| v1.9.0 | developer tools | READY | WidgetDebugger, PerformanceProfiler, CompletionPopup, ThemeEditor |
| v1.10.0 | mouse & gamepad | READY | Mouse events, widget interaction, gamepad, touch, input mapping |
| v1.11.0 | terminal graphics | DONE | Particle effects, blur/transparency, Sixel/Kitty, transitions |
| v1.12.0 | enterprise & accessibility | READY | Session recording, audit logging, WCAG AAA themes, screen reader |
| v1.13.0 | text editing & rich input | READY | Syntax highlighting, code editor, autocomplete, multi-cursor |
| v1.13.1 | bug fix (data viz overflow) | READY | Integer overflow fix; no direct impact on silica currently |
| v1.14.0 | performance & memory | DONE | Memory pooling, render profiling, virtual rendering, layout caching, buffer compression |
| v1.15.0 | stability & thread safety | DONE | Thread safety fixes, memory leak fixes, platform testing, XTGETTCAP capability detection |
| v1.16.0 | advanced terminal features | DONE | Capability database, bracketed paste, synchronized output, hyperlink support, focus tracking |

**High-priority sailor upgrades for silica**:
- v1.9.0: CompletionPopup for SQL keyword/table/column completion
- v1.13.0: Code editor widget for SQL editing, autocomplete for SQL completion
- v1.12.0: Audit logging for SQL query compliance tracking
- v1.5.0: MockTerminal/snapshot testing for TUI test coverage

### zuda Library

- **Current**: Not yet integrated
- **Repo**: https://github.com/yusa-imit/zuda
- **Status**: All targets PENDING (waiting for zuda releases)

| Custom Implementation | File | zuda Replacement | Status |
|----------------------|------|------------------|--------|
| B+Tree | `src/storage/btree.zig` | `zuda.containers.trees.BTree` | PENDING |
| Buffer Pool (LRU) | `src/storage/buffer_pool.zig` | `zuda.containers.hashing.LRUCache` | PENDING |
| Deadlock Detection (DFS) | `src/tx/lock.zig` | `zuda.algorithms.graph.cycle_detection` | PENDING |

**Migration exclusions** (domain-specific, will NOT be migrated):
- `src/storage/fsm.zig` — Free Space Map
- `src/storage/page.zig` — Pager
- `src/storage/overflow.zig` — Overflow page handling
- `src/tx/mvcc.zig` — MVCC logic
- `src/tx/wal.zig` — WAL
- `src/tx/vacuum.zig` — Vacuum
- `src/util/varint.zig` — LEB128 encoding
- `src/util/checksum.zig` — CRC32C

> silica's B+Tree is a 4300 LOC complex implementation. Migration to zuda BTree requires full verification of Cursor (seekFirst/seekLast/seek/next/prev) and overflow page support.
