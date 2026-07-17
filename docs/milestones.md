# silica — Milestones

## Current Status

- **Latest tagged release**: v1.0.1 (All 12 phases complete — Production Ready + Bug Fixes)
- **Current development**: All milestones complete, maintenance mode
- **Tests**: 2800+ tests passing, 33 skipped (28 planned + 5 GIN integration)
- **Branch**: `main`
- **CI Status**: ✅ GREEN (all tests passing)
- **zuda migrations**: Deadlock Detection ✅ DONE (v2.0.0); LRU Cache & B+Tree — NOT MIGRATING per architect decision
- **Known bugs**: None critical
- **Recent fix (Session 259)**: GIN index infinite loop/hang fixed — tests no longer timeout

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

### Milestone 21: Advanced Optimization (Phase 10) ✅ COMPLETE

- [x] DP join ordering (21A partial — simplified two-table join reordering; multi-way joins deferred)
- [x] Hash/merge join selection (21B complete — HashJoinOp with proper join key extraction from ON condition, supports multi-column equi-joins, cost-based selection re-enabled; 5 optimizer tests updated)
- [x] Subquery decorrelation (deferred to future milestone)
- [x] Index-only scans (infrastructure added, full implementation deferred)
- [x] EXPLAIN ANALYZE (21C complete — EXPLAIN and EXPLAIN ANALYZE syntax, plan text formatting via formatPlan(), arena-based memory management; runtime statistics collection deferred to future; 4 integration tests)

### Milestone 22: Hash, GiST, GIN Indexes (Phase 11) ✅ COMPLETE

- [x] Hash index implementation
- [x] GiST framework (Generalized Search Tree)
- [x] GIN framework (Generalized Inverted Index)
- [x] CREATE INDEX CONCURRENTLY
- [x] Bitmap index scans

### Milestone 23: Operational Tools (Phase 12) ✅ COMPLETE

- [x] EXPLAIN and EXPLAIN ANALYZE (text format)
- [x] VACUUM (manual and auto-vacuum)
- [x] REINDEX
- [x] pg_stat_activity monitoring view
- [x] pg_locks monitoring view
- [x] Configuration system (SET/SHOW/RESET SQL commands)
- [x] silica.conf configuration file (INI-style with hot-reload support)

---

## Upcoming Milestones

### Phase 12: Production Readiness (continued)

**Milestone 24**: Testing & Certification ✅ COMPLETE
- [x] Fuzz campaign (storage, tokenizer, parser, wire protocol, WAL) — 67+ tests ✅
- [x] SQL conformance tests — 35 tests validating SQL:2016 compliance ✅
- [x] PreparedStatement API — Database.prepare(), bind(), execute() (issue #11 closed) ✅
- [x] TPC-C benchmark (OLTP workload) — new-order & payment transactions, tpmC metrics ✅
- [x] TPC-H benchmark (OLAP workload) — 3 representative queries (Q1, Q3, Q6) ✅
- [x] Jepsen-style testing (distributed consistency verification) — 19 tests ✅

**Milestone 25**: Documentation & Packaging ✅ COMPLETE
- [x] README.md — Project overview, quick start, features
- [x] API reference (docs/API_REFERENCE.md) — Zig embedded API, C FFI
- [x] Getting started guide (docs/GETTING_STARTED.md) — Complete tutorial
- [x] SQL reference (docs/SQL_REFERENCE.md) — Complete SQL syntax guide
- [x] Operations guide (docs/OPERATIONS_GUIDE.md) — Backup, restore, monitoring, tuning
- [x] Architecture guide (docs/ARCHITECTURE_GUIDE.md) — Internal design
- [x] CI/CD pipeline polish — Caching, benchmarks, versioned artifacts
- [x] System packages (deb, rpm, brew) — debian/, packaging/, docs/PACKAGING.md

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
| Phase 9 | 19 | Replication Operations | — | Synchronous replication, replica promotion, cascading replication, base backup, replication monitoring, lag metrics, switchover procedure |
| Phase 10 | 20 | Statistics & Cost Model | — | ANALYZE command, histogram-based column statistics, selectivity estimation (equality/range/NULL/IN/LIKE), I/O + CPU cost model |
| Phase 10 | 21 | Advanced Optimization | — | DP join ordering (simplified), hash/merge join selection, EXPLAIN ANALYZE |
| Phase 11 | 22 | Additional Index Types | — | Hash index, GiST framework, GIN framework, CREATE INDEX CONCURRENTLY, bitmap index scans |
| Phase 12 | 23 | Operational Tools | — | EXPLAIN/EXPLAIN ANALYZE, VACUUM (manual + auto), REINDEX, pg_stat_activity, pg_locks, configuration system (SET/SHOW/RESET), silica.conf file |
| Phase 12 | 24 | Testing & Certification | — | Fuzz campaign (67+ tests), SQL conformance (35 tests), PreparedStatement API, TPC-C/TPC-H benchmarks, Jepsen-style testing (19 tests) |

### Closed Issues

| # | Title | Closed |
|---|-------|--------|
| #24 | PreparedStatement arena lifecycle needs architectural refactor | 2026-03-29 |
| #3 | Flaky test: AutoVacuumDaemon — inserts only never trigger vacuum | 2026-03-15 |
| #2 | CI: net.Stream.Writer incompatibility on Linux | 2026-03-11 |
| #1 | DuplicateKey error when inserting rows across multiple tables | 2026-03-02 |

### Open Issues

| # | Title | Labels | Status |
|---|-------|--------|--------|
| #15 | feat: implement SSI (Serializable Snapshot Isolation) for SERIALIZABLE isolation level | enhancement | Open — future enhancement (post-v1.0) |

---

## Milestone Establishment Process

12 phases, 25 milestones total. Detailed checklists in `docs/PRD.md` Section 11.

Dependency order: Storage -> SQL -> Transaction(MVCC) -> Catalog(Views/Triggers) -> Server -> Replication

---

## Dependency Migration Tracking

### Sailor Library

- **Current in silica**: v2.93.0
- **Latest available**: v2.93.0
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
| v1.9.0 | developer tools | PARTIAL | SQL autocomplete implemented (custom rendering due to sailor#13); WidgetDebugger, PerformanceProfiler, ThemeEditor pending |
| v1.10.0 | mouse & gamepad | READY | Mouse events, widget interaction, gamepad, touch, input mapping |
| v1.11.0 | terminal graphics | DONE | Particle effects, blur/transparency, Sixel/Kitty, transitions |
| v1.12.0 | enterprise & accessibility | READY | Session recording, audit logging, WCAG AAA themes, screen reader |
| v1.13.0 | text editing & rich input | READY | Syntax highlighting, code editor, autocomplete, multi-cursor |
| v1.13.1 | bug fix (data viz overflow) | READY | Integer overflow fix; no direct impact on silica currently |
| v1.14.0 | performance & memory | DONE | Memory pooling, render profiling, virtual rendering, layout caching, buffer compression |
| v1.15.0 | stability & thread safety | DONE | Thread safety fixes, memory leak fixes, platform testing, XTGETTCAP capability detection |
| v1.16.0 | advanced terminal features | DONE | Capability database, bracketed paste, synchronized output, hyperlink support, focus tracking |
| v1.17.0 | logging & diagnostics | READY | Structured logging, log rotation, diagnostic commands, system monitoring |
| v1.18.0 | hot reload & dev tools | READY | Hot reload, widget inspector, benchmark suite, example gallery, documentation generator |
| v1.19.0 | enhancement & polish | DONE | Progress bar templates, env config, color themes, table formatting, arg groups |
| v1.20.0 | quality & completeness | DONE | Windows console Unicode tests, pattern documentation, directory scanning, error context |
| v1.21.0 | streaming & large data | DONE | DataSource abstraction, large data benchmarks, memory efficiency |
| v1.22.0 | rich text & formatting | DONE | SpanBuilder/LineBuilder APIs, rich text parser, line breaking, Unicode-aware text measurements |
| v1.23.0 | plugin architecture | DONE | Widget trait system, custom renderer hooks, theme plugins, composition helpers (Padding/Centered/Aligned/Stack/Constrained) |
| v1.24.0 | animation & transitions | DONE | Easing functions (22 types: linear/cubic/elastic/bounce/back/circ/expo), animation system (value/color interpolation), timer system, transition helpers (fade/slide) |
| v1.25.0 | form & validation | DONE | Form widget with field focus management, password masking, 15+ validators (notEmpty/email/url/ipv4/numeric/etc), input masks (SSN/phone/date/credit card) |
| v1.26.0 | testing & quality assurance | DONE | Comprehensive test suite, test coverage improvements, Zig 0.15 compatibility fixes (sailor#1) |
| v1.27.0 | documentation & examples | DONE | 98% API documentation coverage, comprehensive guides (getting started, troubleshooting, performance), 5 new examples (hello, counter, dashboard, task_list, layout_showcase) |
| v1.28.0 | performance validation & zuda audit | DONE | 12 core widgets benchmarked (all <0.02ms/op), zuda integration audit (no changes needed) |
| v1.29.0 | documentation coverage | DONE | 99.9% API documentation coverage, comprehensive doc comments (31 newly documented functions) |
| v1.30.0 | error handling & debugging | DONE | Debug logging system (SAILOR_DEBUG env), stack trace helpers, error recovery patterns |
| v1.30.1 | bug fix (Zig 0.15 compat) | DONE | Fixed std.BoundedArray → std.BoundedArrayAligned for Zig 0.15 |
| v1.30.2 | bug fix (Zig 0.15.2 compat) | DONE | Fixed BoundedArrayAligned usage (doesn't exist in Zig 0.15.2) |
| v1.31.0 | performance profiling & optimization | DONE | Render profiler, memory tracker, event loop profiler, widget metrics, profiling demo |
| v1.32.0 | layout enhancements | DONE | Nested grids, aspect ratio constraints, min/max size propagation, auto-margin/padding, layout inspector |
| v1.33.0 | specialized widgets | DONE | LogViewer, MetricsPanel, ConfigEditor, SplitPane, Breadcrumb, Tooltip (contextual help) |
| v1.34.0 | terminal integration | DONE | OSC 52 clipboard API (3 selection types), terminal emulator detection (xterm/kitty/iTerm2), capability detection (truecolor/mouse/clipboard), paste bracketing enhancements |
| v1.35.0 | widget ecosystem expansion | DONE | Card/Badge/Avatar/Skeleton/Accordion widgets, accessibility improvements (ARIA roles), responsive utilities (media queries, device detection) |
| v1.36.0 | performance monitoring | DONE | Performance Monitoring & Real-Time Metrics (render_metrics, memory_metrics, event_metrics), MetricsDashboard widget (3 layout modes), performance regression tests |
| v1.37.0 | v2.0.0 API bridge | DONE | Deprecation warning system for v2.0.0 migration, Buffer.set() method (setChar → set), Style inference helpers (withForeground/Background/Colors), Widget lifecycle standardization (consistent init/deinit), migration guide (docs/v1-to-v2-migration.md) |
| v1.38.0 | v2.0.0 migration tooling | DONE | Migration script for automated code transformation, deprecation audit tooling, consumer dry-run testing, new deprecation warnings (Rect.new() → struct literal, Block.withTitle() → .title/.title_position, Buffer.setChar() → Buffer.set()) |
| v1.38.1 | bug fix & test coverage | DONE | Migration script diff exit code handling, TextArea/Tree widget tests (~100+ tests) |
| v2.0.0 | major release — API simplification | DONE | Removed Buffer.setChar() (use Buffer.set()), removed Rect.new() (use struct literal), Block.withTitle() no longer deprecated, automated migration script, 3345+ tests |
| v2.1.0 | Optimized Grid Widget (v2 API) | DONE | Grid widget with row/column spanning, custom cell renderers, CSV/TSV/Markdown export, sorting, pagination; API uses direct struct instantiation |
| v2.2.0 | Panic-Free Text Measurement (v2 API) | DONE | Panic-free Unicode width calculations, graceful fallback for invalid input, comprehensive test coverage (100+ tests) |
| v2.3.0 | RenderBudget & Performance Monitoring | DONE | RenderBudget for incremental rendering, MetricsDashboard widget, performance regression tests, adaptive framerate control |
| v2.4.0 | Window Composition Helpers | DONE | Stack (z-index layering), Padding (uniform/directional), Centered (horizontal/vertical/both), Aligned (flex-style positioning) |
| v2.5.0 | Core Stability & Graceful Degradation | DONE | 100% panic-free rendering pipeline, graceful degradation for invalid input, enhanced error propagation, comprehensive boundary condition tests |
| v2.6.0 | Input Enhancements & Usability | DONE | ScrollView widget (keyboard/mouse), Autocomplete widget (fuzzy matching, keyboard nav), clipboard copy (OSC 52), input masking improvements, focus management |
| v2.7.0 | Event System & Async Integration | DONE | EventBus (pub/sub), Command Pattern, AsyncTaskRunner, debouncing/throttling, cancelable operations, comprehensive async tests |
| v2.8.0 | Cross-Platform Enhancements | DONE | Windows ConPTY integration, legacy console fallback, Linux/macOS platform-specific optimizations, comptime platform detection, zero runtime cost |
| v2.9.0 | Developer Experience & Debugging Tools | DONE | Widget Inspector (55 tests), Advanced Profiling (38 tests), Error Boundaries (58 tests), Developer Console (40 tests, Ctrl+Shift+D toggle) |
| v2.10.0 | AI/ML Integration & Smart Features | DONE | LLM integration layer (TokenBudget, RateLimiter, PromptTemplate, ResponseStreamWidget), Smart Autocomplete (context-aware, multi-source, semantic ranking), Layout Intelligence (AI-assisted analysis, responsiveness checking), Natural Language Commands (11 intent types, context-aware disambiguation) |
| v2.10.1 | Natural Language Commands Bug Fixes | DONE | Bug fixes for natural language command parsing |
| v2.10.2 | Test Reliability Improvements | DONE | Test reliability improvements |
| v2.11.0 | Extended Graphics & Protocol Support | DONE | Sixel encoder/decoder (color palette optimization), Kitty graphics protocol, ANSI art rendering (block/braille/ASCII algorithms), particle system (fire/rain/snow/sparkle), gradient backgrounds, blur/transparency effects, transition animations |
| v2.12.0–v2.48.0 | Various improvements | DONE | Migrated incrementally across sessions 348–400 |
| v2.49.0 | Wizard Widget | DONE | Multi-step flow navigation widget (Step indicator row, contentArea geometry, nav hints, 8 builder methods) — Session 401 |
| v2.50.0–v2.78.0 | Various improvements | DONE | Migrated incrementally across sessions 402–445, including RadialBar, DotPlot, FunnelChart TUI overlays |
| v2.79.0 | StreamGraph widget | DONE | Themeriver-style stacked area chart with vertically centered silhouette baseline; symmetric layer stacking, focused layer highlighting, optional label column, block border support, MAX_LAYERS=8, no heap allocations — Session 447 |
| v2.80.0 | ViolinPlot widget | DONE | Query duration distribution overlay, kernel density estimation, MAX_SERIES support, no heap allocations — Session 449 |
| v2.81.0 | SunburstChart widget | DONE | Hierarchical radial chart (concentric rings of arcs); SunburstNode label/value/children/style, MAX_DEPTH=4, MAX_NODES=8, no heap allocations; TUI overlay integrated — Query Type & Duration Breakdown ('k' key), 2-level hierarchy (query type -> Fast/Medium/Slow duration buckets) — Session 451 |
| v2.82.0 | BoxPlot widget | DONE | Box-and-whisker plot with five-number-summary (min/Q1/median/Q3/max) + outlier detection (1.5×IQR); fiveNumberSummary() public helper, MAX_SERIES=8, MAX_SAMPLES=64, no heap allocations; TUI overlay integrated — Query Duration Distribution by Type ('o' key) — Session 453 |
| v2.83.0 | CandlestickChart widget | DONE | OHLC financial candlestick chart; `CandlestickChart` + `Candle` (label, open/high/low/close, style), wick+body rendering, bullish/bearish coloring, shared global price scale, MAX_CANDLES=64, no heap allocations; row-mapping clamp bugfix for malformed OHLC data; no TUI overlay use case identified yet (silica has no financial/price time-series data) — Session 453 |
| v2.83.1 | bug fix | DONE | Windows piped-stdin readByte() hang fix, clipboard trailing newline fix, empty-env-var test skip on Windows — Session 456 |
| v2.84.0 | BulletChart widget | DONE | Few-style KPI bullet graph (value vs. target vs. qualitative ranges), one row per bullet, range bands + value bar + target tick, MAX_BULLETS=32; no TUI overlay use case identified yet (no natural KPI-vs-target metric in silica) — Session 457 |
| v2.85.0 | ParallelCoordinates widget | DONE | Multi-dimensional data viz via parallel vertical axes + per-item polylines; `ParallelCoordinates` + `PCAxis`(label, min, max) + `PCItem`(label, values, style), MAX_AXES=8, MAX_ITEMS=16, no heap allocations; no TUI overlay use case identified yet — Session 459 |
| v2.86.0 | ParetoChart widget | DONE | 80/20 QA visualization: descending-sorted bars + cumulative percent line + threshold marker; `ParetoChart` + `ParetoItem`(label, value, style), MAX_ITEMS=32, no heap allocations; no TUI overlay use case identified yet — Session 459 |
| v2.87.0 | SlopeChart widget | DONE | Before/after two-point comparison per category with direction-styled connecting line; `SlopeChart` + `SlopeItem`(label, left_value, right_value, style), MAX_ITEMS=16, no heap allocations; no TUI overlay use case identified yet — Session 465 |
| v2.88.0 | RidgelinePlot widget | DONE | Stacked vertically-offset density silhouettes (joyplot) per category; `RidgelinePlot` + `RidgelineSeries`(label, values, style), MAX_SERIES=8, MAX_BINS=64, no heap allocations; no TUI overlay use case identified yet — Session 465 |
| v2.89.0 | BumpChart widget | DONE | Multi-time-point rank-over-time lines per category with direction glyphs; `BumpChart` + `BumpSeries`(label, ranks, style), MAX_SERIES=8, MAX_TIMEPOINTS=16, no heap allocations; no TUI overlay use case identified yet — Session 465 |
| v2.90.0 | MosaicPlot widget | DONE | Marimekko-style two-dimensional proportional chart (variable-width columns × stacked variable-height segments); `MosaicPlot` + `MosaicColumn` + `MosaicSegment`(label, value, style), MAX_COLUMNS=16, MAX_SEGMENTS_PER_COLUMN=8, no heap allocations; no TUI overlay use case identified yet — Session 465 |
| v2.91.0 | IcicleChart widget | DONE | Axis-aligned hierarchical chart (alternative to SunburstChart's radial layout); stacked horizontal bands per tree depth, cumulative-floor width formula consistent with MosaicPlot/SunburstChart; `IcicleChart` + `IcicleNode`, focus-path highlighting, independent show_labels/show_values toggles, MAX_DEPTH=6, MAX_CHILDREN_PER_NODE=8, no heap allocations; no TUI overlay use case identified yet — Session 468 |
| v2.92.0 | ToggleSwitch widget | DONE | Boolean on/off slider-style form control (fixed 6-cell bracketed track, sliding knob ◯/◉); `ToggleSwitchGroup` manages a set with radio-like exclusive-toggle focus navigation, skipping disabled items on wrap; silica's TUI overlays are keypress-toggled (no widget-based settings form), so no direct use case identified yet — Session 470 |
| v2.92.1 | bug fix (FlowChart render order) | DONE | Edges (arrows/labels) were rendering before nodes, so node borders overwrote them; fixed render order (nodes then edges). Also strengthened 15 weak disjunction assertions in bubble_chart/flowchart/gantt/gantt_chart/matrix_view tests. Silica's `renderFlowChart` (src/tui.zig, query pipeline overlay) uses sailor.FlowChart — existing FlowChart overlay tests (src/tui.zig ~6302+) pass unchanged post-upgrade — Session 471 |
| v2.93.0 | CalendarHeatmap widget | DONE | GitHub-style contribution/activity heatmap (date-indexed values → week-column × weekday-row grid, 5-level intensity shading, month/weekday labels, focused-cell highlighting); also adds `Calendar.setRange()`. No TUI overlay use case identified yet — Session 473 |

**High-priority sailor upgrades for silica**:
- v1.9.0: ~~CompletionPopup for SQL keyword/table/column completion~~ ✅ **DONE** (Session 63 — custom rendering due to sailor#13)
- v1.13.0: Code editor widget for SQL editing, autocomplete for SQL completion
- v1.12.0: Audit logging for SQL query compliance tracking
- v1.5.0: MockTerminal/snapshot testing for TUI test coverage

### zuda Library

- **Current**: v2.0.4 (integrated)
- **Latest available**: v2.0.4
- **Repo**: https://github.com/yusa-imit/zuda
- **Migration Status**: 1/3 completed — Deadlock Detection ✅ DONE

| Custom Implementation | File | LOC | zuda Replacement | Status |
|----------------------|------|-----|------------------|--------|
| Buffer Pool (LRU) | `src/storage/buffer_pool.zig` | 1237 | `zuda.containers.cache.LRUCache` | **NOT MIGRATING** — Architect decision (Session 27): non-failable eviction callback, per-entry heap allocation, deep integration (12+ files) |
| Deadlock Detection (DFS) | `src/tx/lock.zig` | 1463 → 1453 | `zuda.algorithms.graph.DFS` | ✅ **DONE** (Session 27) — zuda#10 resolved, hasCycle() implemented |
| B+Tree | `src/storage/btree.zig` | 4300 | `zuda.containers.trees.BTree` | **NOT MIGRATING** — Similar reasons to buffer pool (disk I/O, WAL, MVCC integration) |

**Migration exclusions** (domain-specific, will NOT be migrated):
- `src/storage/buffer_pool.zig` — Buffer Pool (decision: keep custom per architect review)
- `src/storage/btree.zig` — B+Tree (disk I/O, WAL, MVCC integration too deep)
- `src/storage/fsm.zig` — Free Space Map
- `src/storage/page.zig` — Pager
- `src/storage/overflow.zig` — Overflow page handling
- `src/tx/mvcc.zig` — MVCC logic
- `src/tx/wal.zig` — WAL
- `src/tx/vacuum.zig` — Vacuum
- `src/util/varint.zig` — LEB128 encoding
- `src/util/checksum.zig` — CRC32C

> **Migration Summary**: Deadlock detection successfully migrated to zuda v1.23.0. Buffer pool and B+Tree will remain custom implementations due to deep integration with silica-specific concerns (disk I/O, WAL, MVCC, non-failable eviction). See `.claude/memory/decisions.md` for architect review details.
