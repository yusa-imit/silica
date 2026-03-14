# silica — Milestones

## Current Status

- **Latest release**: v0.3.0 (Phase 3: WAL & Basic Transactions)
- **Current phase**: Phase 5 — Advanced SQL (per CLAUDE.md header)
- **Note**: Phase 4 (MVCC) status may need verification — check git tags and test status
- **Blockers**: None
- **Branch**: `main`

---

## Active Milestones

### Phase 4: MVCC & Full Transactions

**Milestone 6**: MVCC Core
- Tuple versioning with `(xmin, xmax)` transaction IDs
- Transaction ID management
- Snapshot isolation
- Visibility rules
- READ COMMITTED / REPEATABLE READ isolation levels
- Row-level locking
- Concurrent writer conflict detection

**Milestone 7**: VACUUM & SSI
- Dead tuple reclamation
- Auto-vacuum
- Free space map
- SERIALIZABLE via SSI (rw-antidependency tracking)
- Deadlock detection
- Savepoints

### Phase 5: Advanced SQL (CURRENT per CLAUDE.md)

**Milestone 8**: Views & CTEs
- Views (regular, materialized, updatable)
- CTEs (`WITH`, `WITH RECURSIVE`)
- Set operations (UNION/INTERSECT/EXCEPT)
- `DISTINCT ON`

**Milestone 9**: Window Functions
- `ROW_NUMBER`, `RANK`, `DENSE_RANK`, `LAG`, `LEAD`, `FIRST_VALUE`, `LAST_VALUE`
- Frame specs (ROWS/RANGE/GROUPS)
- WindowAgg operator

**Milestone 10**: Advanced Data Types
- DATE/TIME/TIMESTAMP/INTERVAL
- NUMERIC/DECIMAL
- UUID, SERIAL
- ARRAY, ENUM
- Domain types, type coercion matrix

### Phase 6: JSON & Full-Text Search

**Milestone 11**: JSON/JSONB
- Binary storage, operators (`->`, `->>`, `@>`, `?`), functions
- GIN index, SQL/JSON path

**Milestone 12**: Full-Text Search
- TSVECTOR/TSQUERY, `@@` operator, ranking
- GIN index, text search configs

### Phase 7: Stored Functions & Triggers

**Milestone 13**: Stored Functions
- SFL (Silica Function Language), scalar & set-returning, volatility categories

**Milestone 14**: Triggers
- Row/statement-level, BEFORE/AFTER/INSTEAD OF, OLD/NEW references, WHEN conditions

### Phase 8: Client-Server & Wire Protocol

**Milestone 15**: PostgreSQL Wire Protocol v3
- Simple & extended query protocol, prepared statements, COPY, TLS

**Milestone 16**: Server & Connection Management
- Async I/O event loop, session state, authentication (SCRAM-SHA-256), `silica server` CLI

**Milestone 17**: Authorization (RBAC)
- Roles, GRANT/REVOKE, row-level security, `information_schema`

### Phase 9: Streaming Replication

**Milestone 18**: WAL Sender/Receiver
- Replication slots, hot standby, replication protocol

**Milestone 19**: Replication Operations
- Synchronous mode, replica promotion, cascading replication, base backup, monitoring

### Phase 10: Cost-Based Optimizer

**Milestone 20**: Statistics & Cost Model
- `ANALYZE`, histograms, selectivity estimation, I/O + CPU cost model

**Milestone 21**: Advanced Optimization
- DP join ordering, hash/merge join selection, subquery decorrelation, index-only scans, `EXPLAIN ANALYZE`

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

| Phase | Name | Release | Summary |
|-------|------|---------|---------|
| Phase 1 | Storage Foundation | v0.1.0 | Page Manager, B+Tree, Buffer Pool, Overflow, CRC32C, Varint |
| Phase 2 | SQL Core | v0.2.0 | Tokenizer, Parser, AST, Analyzer, Catalog, Planner, Optimizer, Executor, Engine, secondary indexes |
| Phase 3 | WAL & Basic Transactions | v0.3.0 | WAL frame writer, commit, rollback, checkpoint, recovery, buffer pool WAL routing |

---

## Milestone Establishment Process

12 phases, 25 milestones total. Detailed checklists in `docs/PRD.md` Section 11.

Dependency order: Storage -> SQL -> Transaction(MVCC) -> Catalog(Views/Triggers) -> Server -> Replication

---

## Dependency Migration Tracking

### Sailor Library

- **Current in silica**: v1.11.0
- **Latest available**: v1.13.1
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
| v1.13.1 | bug fix (data viz overflow) | N/A | Integer overflow fix; no impact on silica currently |

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
