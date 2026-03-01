# Silica — Claude Code Orchestrator

> **Silica**: Zig로 작성된 프로덕션 등급 풀 RDBMS — 듀얼 모드 (임베디드 + 클라이언트-서버), MVCC, Full SQL:2016, 스트리밍 복제
> Current Phase: **Phase 4 — MVCC & Full Transactions**

---

## Project Overview

- **Language**: Zig 0.15.x (stable)
- **Type**: Full-featured relational database — dual-mode (embedded + client-server)
- **Build**: `zig build` / `zig build test`
- **PRD**: `docs/PRD.md` (전체 요구사항 참조)
- **Branch Strategy**: `main` (development)

## Repository Structure

```
silica/
├── CLAUDE.md                    # THIS FILE — orchestrator
├── docs/PRD.md                  # Product Requirements Document
├── .gitignore                   # Git ignore rules
├── .claude/
│   ├── agents/                  # Custom subagent definitions (6 agents)
│   │   ├── zig-developer.md     #   model: sonnet — Zig 구현
│   │   ├── code-reviewer.md     #   model: sonnet — 코드 리뷰
│   │   ├── test-writer.md       #   model: sonnet — 테스트 작성
│   │   ├── architect.md         #   model: opus   — 아키텍처 설계
│   │   ├── git-manager.md       #   model: haiku  — Git 운영
│   │   └── ci-cd.md             #   model: haiku  — CI/CD 관리
│   ├── commands/                # Slash commands (skills)
│   ├── memory/                  # Persistent agent memory
│   └── settings.json            # Claude Code permissions
├── .github/workflows/           # CI/CD pipelines
│   ├── ci.yml                   #   Build, test, cross-compile
│   └── release.yml              #   Release pipeline
└── src/                         # Source code
    ├── main.zig                 #   엔트리포인트
    ├── cli.zig                  #   CLI (sailor arg/color/fmt integration)
    ├── tui.zig                  #   TUI database browser (sailor.tui)
    ├── storage/                 #   Storage Engine
    │   ├── page.zig             #     Page Manager (Pager)
    │   ├── btree.zig            #     B+Tree implementation
    │   ├── buffer_pool.zig      #     Buffer Pool (LRU cache)
    │   ├── overflow.zig         #     Overflow page handling
    │   └── fuzz.zig             #     B+Tree fuzz tests
    ├── sql/                     #   SQL Frontend & Engine
    │   ├── tokenizer.zig        #     Tokenizer (Lexer)
    │   ├── parser.zig           #     Recursive descent parser → AST
    │   ├── ast.zig              #     AST node definitions
    │   ├── analyzer.zig         #     Semantic analysis
    │   ├── catalog.zig          #     Schema catalog (B+Tree backed)
    │   ├── planner.zig          #     Query planner (logical plan)
    │   ├── optimizer.zig        #     Rule-based optimizer
    │   ├── executor.zig         #     Volcano-model executor
    │   └── engine.zig           #     Database integration layer
    ├── tx/                      #   Transaction Manager
    │   ├── wal.zig              #     Write-Ahead Log
    │   ├── mvcc.zig             #     MVCC visibility & snapshots (planned)
    │   ├── lock.zig             #     Lock Manager (planned)
    │   └── vacuum.zig           #     Dead tuple reclamation (planned)
    ├── catalog/                 #   Extended Catalog (planned)
    │   ├── views.zig            #     View definitions & expansion
    │   ├── functions.zig        #     Stored function registry
    │   ├── triggers.zig         #     Trigger definitions & execution
    │   └── sequences.zig        #     Sequence generators
    ├── types/                   #   Extended Type System (planned)
    │   ├── datetime.zig         #     DATE, TIME, TIMESTAMP, INTERVAL
    │   ├── numeric.zig          #     NUMERIC/DECIMAL fixed-point
    │   ├── json.zig             #     JSON/JSONB storage & operators
    │   ├── array.zig            #     ARRAY type
    │   ├── uuid.zig             #     UUID type
    │   └── fts.zig              #     TSVECTOR/TSQUERY full-text search
    ├── server/                  #   Client-Server Mode (planned)
    │   ├── wire.zig             #     PostgreSQL wire protocol v3
    │   ├── connection.zig       #     Connection handling & session state
    │   ├── auth.zig             #     Authentication (SCRAM-SHA-256)
    │   └── server.zig           #     TCP server & event loop
    ├── replication/             #   Streaming Replication (planned)
    │   ├── sender.zig           #     WAL sender (primary)
    │   ├── receiver.zig         #     WAL receiver (replica)
    │   └── slot.zig             #     Replication slot management
    └── util/                    #   Utilities
        ├── checksum.zig         #     CRC32C checksums
        └── varint.zig           #     Variable-length integer encoding
```

> **Note**: 파일 구조는 참고안. `(planned)`로 표기된 파일은 아직 존재하지 않으며 해당 Phase 구현 시 생성됨. 실제 소스 코드가 기준.

---

## Development Workflow

### Autonomous Development Protocol

Claude Code는 이 프로젝트에서 **완전 자율 개발**을 수행한다. 다음 프로토콜을 따른다:

1. **작업 수신** → PRD 또는 사용자 지시를 분석
2. **계획 수립** → 대화형 세션: `EnterPlanMode`로 사용자 승인; 자율 세션(`claude -p`): 내부적으로 계획 후 즉시 구현 진행 (plan mode 도구 사용 금지)
3. **팀 구성** → 작업 복잡도에 따라 동적으로 팀/서브에이전트 생성
4. **구현** → 코딩, 테스트, 리뷰를 병렬 수행
5. **검증** → `zig build test`로 전체 테스트 통과 확인
6. **커밋** → 변경사항 커밋 (사용자 요청 시)
7. **메모리 갱신** → 학습된 내용을 `.claude/memory/`에 기록

### Team Orchestration

복잡한 작업 시 다음 패턴으로 팀을 구성한다:

```
Leader (orchestrator)
├── zig-developer   — 구현 담당
├── code-reviewer   — 코드 리뷰 & 품질 보증
├── test-writer     — 테스트 작성
└── architect       — 설계 검토 (필요 시)
```

**팀 생성 기준**:
- 3개 이상 파일 수정이 필요한 작업 → 팀 구성
- 단일 파일 수정 → 직접 수행
- 아키텍처 변경 → architect 포함

### Automated Session Execution

자동화 세션(cron job 등)에서는 다음 프로토콜을 순서대로 실행한다.

**컨텍스트 복원** — 세션 시작 시 다음 파일을 읽어 프로젝트 상태 파악:
1. `.claude/memory/project-context.md` — 현재 phase, 체크리스트, 진행 상황
2. `.claude/memory/architecture.md` — 아키텍처 결정사항
3. `.claude/memory/decisions.md` — 기술 결정 로그
4. `.claude/memory/debugging.md` — 알려진 이슈와 해결법
5. `.claude/memory/patterns.md` — 검증된 코드 패턴

**실행 사이클**:

| Phase | 내용 | 비고 |
|-------|------|------|
| 1. 상태 파악 | `/status` 실행, git log·빌드·테스트 상태 점검 | 체크리스트에서 다음 미완료 항목 식별 |
| 1.5. 이슈 확인 | `gh issue list --state open --limit 10` | 아래 이슈 우선순위 프로토콜 참조 |
| 2. 계획 | 구현 전략을 내부적으로 수립 (텍스트 출력) | `EnterPlanMode`/`ExitPlanMode` 사용 금지 — 비대화형 세션에서 블로킹됨 |
| 3. 구현 → 검증 → 커밋 (반복) | 아래 **구현 루프** 참조 | 단위별로 즉시 커밋+푸시 |
| 4. 코드 리뷰 | `/review` — PRD 준수·메모리 안전성·테스트 커버리지 확인 | 이슈 발견 시 수정 후 재커밋 |
| 5. 릴리즈 판단 | 릴리즈 조건 충족 시 자동 릴리즈 | 아래 Release & Patch Policy 참조 |
| 6. 메모리 갱신 | `.claude/memory/` 파일 업데이트 | 별도 커밋: `chore: update session memory` → push |
| 7. 세션 요약 | 구조화된 요약 출력 | 아래 템플릿 참조 |

**구현 루프** (Phase 3 상세):

작업을 작은 단위로 분할하고, 각 단위마다 다음을 반복한다:
1. 코드 작성 (하나의 모듈/파일 단위)
2. 테스트 작성 및 `zig build test` 통과 확인
3. 즉시 커밋 + `git push` — 다음 단위로 넘어가기 전에 반드시 수행
- 미커밋 변경사항을 여러 파일에 걸쳐 누적하지 않는다
- 한 사이클 내에 완료할 수 없는 작업은 동작하는 중간 상태로 커밋+푸시한다
- `git add -A` 금지 — 변경된 파일을 명시적으로 지정

**이슈 우선순위 프로토콜** (Phase 1.5):

세션 시작 시 GitHub Issues를 확인하고 우선순위를 결정한다:

```bash
gh issue list --state open --limit 10 --json number,title,labels,createdAt
```

| 우선순위 | 조건 | 행동 |
|---------|------|------|
| 1 (최우선) | `bug` 라벨 | 다른 작업보다 항상 우선 처리 |
| 2 (높음) | `from:*` 라벨 (소비자 프로젝트 요청) | 현재 작업보다 우선 |
| 3 (보통) | `feature-request` + 현재 phase 범위 내 | 현재 작업과 병행 |
| 4 (낮음) | `feature-request` + 미래 phase | 적어두고 넘어감 |

- 이슈 처리 후: `gh issue close <number> --comment "Fixed in <commit-hash>"`
- 진행 상황 공유: `gh issue comment <number> --body "Working on this"`

**작업 선택 규칙**:
- `build.zig`가 없으면 프로젝트 부트스트랩부터 시작
- 이전 세션의 미커밋 변경사항이 있으면: 테스트 통과 시 커밋+푸시, 실패 시 폐기
- 테스트 실패 중이면 새 기능 추가 전에 수정
- 의존성 순서 준수: Storage → SQL → Transaction(MVCC) → Catalog(Views/Triggers) → Server → Replication
- 사이클당 하나의 집중 작업만 수행
- 이전 세션의 미완료 작업이 있으면 먼저 완료

**세션 요약 템플릿**:

    ## Session Summary
    ### Completed
    - [이번 사이클에서 완료한 내용]
    ### Files Changed
    - [생성/수정된 파일 목록]
    ### Tests
    - [테스트 수, 통과/실패 상태]
    ### Next Priority
    - [다음 사이클에서 작업할 내용]
    ### Issues / Blockers
    - [발생한 문제 또는 미해결 이슈]

### Available Custom Agents

| Agent | Model | File | Purpose |
|-------|-------|------|---------|
| zig-developer | sonnet | `.claude/agents/zig-developer.md` | Zig 코드 구현, 빌드 오류 해결 |
| code-reviewer | sonnet | `.claude/agents/code-reviewer.md` | 코드 리뷰, 품질/보안 검사 |
| test-writer | sonnet | `.claude/agents/test-writer.md` | 유닛/통합 테스트 작성 |
| architect | opus | `.claude/agents/architect.md` | 아키텍처 설계, 모듈 구조 결정 |
| git-manager | haiku | `.claude/agents/git-manager.md` | Git 운영, 브랜치/커밋 관리 |
| ci-cd | haiku | `.claude/agents/ci-cd.md` | GitHub Actions, CI/CD 파이프라인 |

### Available Slash Commands

| Command | File | Purpose |
|---------|------|---------|
| /build | `.claude/commands/build.md` | 프로젝트 빌드 |
| /test | `.claude/commands/test.md` | 테스트 실행 |
| /review | `.claude/commands/review.md` | 현재 변경사항 코드 리뷰 |
| /implement | `.claude/commands/implement.md` | 기능 구현 워크플로우 |
| /fix | `.claude/commands/fix.md` | 버그 수정 워크플로우 |
| /release | `.claude/commands/release.md` | 릴리스 워크플로우 |
| /status | `.claude/commands/status.md` | 프로젝트 상태 확인 |
| /bench | `.claude/commands/bench.md` | 성능 벤치마크 실행 |

---

## Coding Standards

### Zig Conventions

- **Naming**: camelCase for functions/variables, PascalCase for types, SCREAMING_SNAKE for constants
- **Error handling**: Always use explicit error unions, never `catch unreachable` in production code
- **Memory**: Prefer arena allocators for request-scoped work, GPA for long-lived allocations
- **Testing**: Every public function must have corresponding tests in the same file
- **Comments**: Only where logic is non-obvious. No doc comments on self-explanatory functions
- **Imports**: Group stdlib, then project imports, then test imports

### File Organization

- One module per file. PRD의 구조는 초기 참고안이며, 실제 구현에 따라 변경 가능. 소스 코드가 기준
- Keep files under 500 lines; split into submodules if exceeded
- Public API at top of file, private helpers at bottom
- Tests at the bottom of each file within `test` block

### Error Messages

User-facing errors must follow this pattern:
```
✗ [Context]: [What happened]

  [Details with syntax highlighting]

  Hint: [Actionable suggestion]
```

### Database-Specific Conventions

- **Page operations**: Always check page bounds before read/write. Use CRC32C for integrity.
- **B+Tree invariants**: After every operation, verify: sorted keys, valid child pointers, balanced depth.
- **Buffer Pool**: Pin/unpin must be balanced. Use `defer pool.unpin(page)` pattern.
- **WAL**: Never modify main DB file directly — always through WAL first.
- **MVCC**: Every row version must carry `(xmin, xmax)` transaction IDs. Visibility checks are mandatory before returning any tuple.
- **Isolation correctness**: Never weaken an isolation level's guarantees. READ COMMITTED = per-statement snapshot. REPEATABLE READ = per-transaction snapshot. SERIALIZABLE = SSI with rw-antidependency tracking.
- **Locking discipline**: Acquire locks in a consistent order to prevent deadlocks. Row locks before index locks. Always release locks on transaction end (commit or rollback).
- **Concurrency safety**: All shared data structures (buffer pool, lock table, transaction table) must be protected by appropriate synchronization primitives.
- **Wire protocol**: Follow PostgreSQL wire protocol v3 exactly — byte-level compatibility is required for client library interop.
- **Replication**: WAL records must be self-describing (contain enough info to replay without the original query). LSN ordering must be strictly monotonic.
- **Allocator discipline**: Storage engine uses arena per-transaction; buffer pool uses page-aligned allocator. Server mode: arena per-connection for session state.

---

## Git Workflow

### Branch Strategy

- `main` — primary development branch
- Feature branches: `feat/<name>`, `fix/<name>`, `refactor/<name>`

### Commit Convention

```
<type>: <subject>

<body>

Co-Authored-By: Claude <noreply@anthropic.com>
```

Types: `feat`, `fix`, `refactor`, `test`, `chore`, `docs`, `perf`, `ci`

### PR Convention

- Title: `<type>: <concise description>` (under 70 chars)
- Body: Summary bullets + Test plan
- Always target `main` unless specified otherwise

---

## Memory System

### Long-Term Memory Preservation

에이전트와 오케스트레이터는 `.claude/memory/` 디렉토리에 장기 기억을 보존한다.

**메모리 파일 구조**:
```
.claude/memory/
├── project-context.md    # 프로젝트 개요 (PRD 요약, 진행 상황)
├── architecture.md       # 아키텍처 결정사항
├── decisions.md          # 주요 기술 결정 로그
├── debugging.md          # 디버깅 인사이트, 해결된 문제
├── patterns.md           # 검증된 코드 패턴
└── session-summaries/    # 세션별 요약 (압축된 기억)
```

**메모리 프로토콜**:
1. 세션 시작 시 `.claude/memory/` 파일들을 읽어 컨텍스트 복원
2. 중요한 결정/발견 시 즉시 해당 메모리 파일에 기록
3. 세션 종료 전 `session-summaries/`에 해당 세션의 핵심 내용 요약
4. 메모리 파일이 200줄을 초과하면 핵심만 남기고 압축

**메모리 압축 규칙**:
- 해결된 문제는 1-2줄 요약으로 압축
- 반복 확인된 패턴만 유지, 일회성 발견은 제거
- 최신 정보가 과거 정보보다 우선

---

## Implementation Roadmap

전체 12 Phase, 25 Milestone. 자세한 체크리스트는 `docs/PRD.md` Section 11 참조.

### Phase 1: Storage Foundation ✅
- Page Manager, B+Tree, Buffer Pool, Overflow, CRC32C, Varint
- 릴리즈: v0.1.0

### Phase 2: SQL Core ✅
- Tokenizer, Parser (Pratt precedence), AST, Semantic Analyzer
- Schema Catalog (B+Tree backed), Query Planner, Rule-based Optimizer
- Volcano-model Executor, Database Engine integration
- Secondary indexes, WHERE with index selection

### Phase 3: WAL & Basic Transactions ✅
- WAL frame writer, commit, rollback, checkpoint, recovery
- Buffer pool WAL routing, Engine WAL mode integration

### Phase 4: MVCC & Full Transactions ← **CURRENT**
- **Milestone 6**: MVCC Core — tuple versioning `(xmin, xmax)`, transaction ID management, snapshot isolation, visibility rules, READ COMMITTED / REPEATABLE READ, row-level locking, concurrent writer conflict detection
- **Milestone 7**: VACUUM & SSI — dead tuple reclamation, auto-vacuum, free space map, SERIALIZABLE via SSI (rw-antidependency tracking), deadlock detection, savepoints

### Phase 5: Advanced SQL
- **Milestone 8**: Views (regular, materialized, updatable), CTEs (`WITH`, `WITH RECURSIVE`), set operations (UNION/INTERSECT/EXCEPT), `DISTINCT ON`
- **Milestone 9**: Window functions — `ROW_NUMBER`, `RANK`, `DENSE_RANK`, `LAG`, `LEAD`, `FIRST_VALUE`, `LAST_VALUE`, frame specs (ROWS/RANGE/GROUPS), WindowAgg operator
- **Milestone 10**: Advanced data types — DATE/TIME/TIMESTAMP/INTERVAL, NUMERIC/DECIMAL, UUID, SERIAL, ARRAY, ENUM, domain types, type coercion matrix

### Phase 6: JSON & Full-Text Search
- **Milestone 11**: JSON/JSONB — binary storage, operators (`->`, `->>`, `@>`, `?`), functions, GIN index, SQL/JSON path
- **Milestone 12**: Full-text search — TSVECTOR/TSQUERY, `@@` operator, ranking, GIN index, text search configs

### Phase 7: Stored Functions & Triggers
- **Milestone 13**: Stored functions — SFL (Silica Function Language), scalar & set-returning, volatility categories
- **Milestone 14**: Triggers — row/statement-level, BEFORE/AFTER/INSTEAD OF, OLD/NEW references, WHEN conditions

### Phase 8: Client-Server & Wire Protocol
- **Milestone 15**: PostgreSQL wire protocol v3 — simple & extended query protocol, prepared statements, COPY, TLS
- **Milestone 16**: Server & connection management — async I/O event loop, session state, authentication (SCRAM-SHA-256), `silica server` CLI
- **Milestone 17**: Authorization (RBAC) — roles, GRANT/REVOKE, row-level security, `information_schema`

### Phase 9: Streaming Replication
- **Milestone 18**: WAL sender/receiver — replication slots, hot standby, replication protocol
- **Milestone 19**: Replication operations — synchronous mode, replica promotion, cascading replication, base backup, monitoring

### Phase 10: Cost-Based Optimizer
- **Milestone 20**: Statistics & cost model — `ANALYZE`, histograms, selectivity estimation, I/O + CPU cost model
- **Milestone 21**: Advanced optimization — DP join ordering, hash/merge join selection, subquery decorrelation, index-only scans, `EXPLAIN ANALYZE`

### Phase 11: Additional Index Types
- **Milestone 22**: Hash, GiST, GIN indexes — `CREATE INDEX CONCURRENTLY`, bitmap index scans

### Phase 12: Production Readiness
- **Milestone 23**: Operational tools — `EXPLAIN ANALYZE`, `VACUUM`, monitoring views, config system
- **Milestone 24**: Testing & certification — TPC-C/TPC-H benchmarks, jepsen-style testing, fuzz campaign, SQL conformance
- **Milestone 25**: Documentation & packaging — API reference, ops guide, SQL reference, system packages

---

## Quick Reference

```bash
# Build
zig build

# Test
zig build test

# Run embedded shell
zig build run -- mydb.db

# Run TUI browser
zig build run -- --tui mydb.db

# Run server (Phase 8+)
zig build run -- server --data-dir /var/lib/silica --port 5433

# Cross-compile
zig build -Dtarget=x86_64-linux -Doptimize=ReleaseSafe

# Clean
rm -rf zig-out .zig-cache

# Benchmark
zig build bench
```

---

## Rules for Claude Code

1. **Always read before writing** — 파일 수정 전 반드시 Read로 현재 내용 확인
2. **Test after every change** — 코드 변경 후 `zig build test` 실행
3. **Incremental commits** — 기능 단위로 작은 커밋
4. **Memory updates** — 중요한 발견/결정은 즉시 메모리에 기록
5. **No over-engineering** — 현재 phase에 필요한 것만 구현
6. **PRD is source of truth for requirements** — 기능 요구사항은 `docs/PRD.md` 참조. 단, 파일/폴더 구조는 참고안이며 실제 소스 코드가 기준
7. **Team cleanup** — 팀 작업 완료 후 반드시 해산
8. **Error messages matter** — 사용자 경험은 에러 메시지 품질로 결정됨
9. **Stop if stuck** — 동일 에러가 3회 시도 후에도 지속되면 `.claude/memory/debugging.md`에 기록하고 다음 작업으로 이동
10. **No scope creep** — 현재 Phase 체크리스트 범위를 벗어나는 작업 금지
11. **Respect CI** — CI 파이프라인이 존재하면 `ci.yml` 호환성 유지
12. **Never force push** — 파괴적 git 명령어 금지, `main` 브랜치 직접 수정 금지
13. **Database correctness first** — 성능보다 정확성 우선. 데이터 무결성 검증 테스트 필수
14. **Page-level atomicity** — 모든 페이지 쓰기는 원자적이어야 함. 부분 쓰기 = 데이터 손상
15. **MVCC visibility is sacred** — 가시성 규칙 위반은 데이터 무결성 위반과 동일. 모든 튜플 반환 전 visibility check 필수
16. **Wire protocol byte-compatibility** — PostgreSQL 클라이언트 라이브러리와의 호환성은 바이트 레벨에서 보장. 프로토콜 편의를 위한 deviations 금지
17. **Isolation level guarantees are non-negotiable** — 각 격리 수준이 보장하는 속성을 절대로 약화시키지 않음. 의심스러우면 더 강한 격리 적용

---

## Release & Patch Policy

세션 사이클의 **Step 5 (릴리즈 판단)** 에서 아래 조건을 확인하고, 충족 시 자율적으로 릴리즈를 수행한다.

### 마이너 릴리즈 (v0.X.0)

phase의 모든 모듈이 완성되었을 때 자율적으로 릴리즈를 수행한다.

**릴리즈 조건 (ALL must be true)**:
1. 현재 phase의 체크리스트 항목이 **모두 완료** (`[x]`)
2. `zig build test` — 전체 통과, 0 failures
3. 크로스 컴파일 타겟 빌드 성공
4. `bug` 라벨 이슈가 **0개** (open)

**릴리즈 조건 확인 방법**:
```bash
LAST_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "")
git log ${LAST_TAG}..HEAD --oneline
gh issue list --state open --label bug --limit 5
```

**릴리즈 절차**:
1. `build.zig.zon`의 version 업데이트
2. CLAUDE.md phase 체크리스트에 완료 표시
3. 커밋: `chore: bump version to v0.X.0`
4. 태그: `git tag -a v0.X.0 -m "Release v0.X.0: <phase 요약>"`
5. 푸시: `git push && git push origin v0.X.0`
6. GitHub Release: `gh release create v0.X.0 --title "v0.X.0: <phase 요약>" --notes "<릴리즈 노트>"`
7. 관련 이슈 닫기: `gh issue close <number> --comment "Resolved in v0.X.0"`
8. Discord 알림: `openclaw message send --channel discord --target user:264745080709971968 --message "[silica] Released v0.X.0 — <요약>"`

### 패치 릴리즈 (v0.X.Y)

버그 수정 시 패치 릴리즈를 즉시 발행한다.

**트리거 조건**:
- 사용자 보고 버그가 수정된 커밋이 존재하지만 릴리즈 태그가 없을 때
- 빌드/테스트 실패를 수정한 커밋
- 크로스 컴파일 깨짐을 수정한 커밋

**패치 vs 마이너 판단**:
- 버그 수정만 포함 → PATCH (v0.X.Y)
- 새 기능 포함 → MINOR (v0.X+1.0)

**버전 규칙**:
- PATCH 번호만 증가 (예: v0.1.0 → v0.1.1)
- `build.zig.zon` version 수정 불필요 — 태그만으로 충분
- 기능 커밋을 패치에 포함하지 않음

**패치 릴리즈 절차**:
1. 버그 수정 커밋 식별
2. `zig build test` 통과 확인
3. 태그: `git tag -a v0.X.Y <commit-hash> -m "Release v0.X.Y: <수정 요약>"`
4. 푸시: `git push origin v0.X.Y`
5. GitHub Release: `gh release create v0.X.Y --title "v0.X.Y: <요약>" --notes "<릴리즈 노트>"`
6. 관련 이슈에 릴리즈 코멘트 추가
7. Discord 알림

---

## Sailor Migration

silica는 `sailor` 라이브러리(https://github.com/yusa-imit/sailor)를 점진적으로 도입한다.
각 버전이 READY 상태가 되면, 해당 세션에서 마이그레이션을 수행한다.

### 마이그레이션 프로토콜

1. 세션 시작 시 이 섹션을 확인한다
2. `status: READY`인 미완료 마이그레이션이 있으면, 현재 작업보다 **우선** 수행한다
3. 마이그레이션 완료 후 `status: DONE`으로 변경하고 커밋한다
4. `zig build test` 통과 확인 필수

### sailor 이슈 발행 프로토콜

sailor 라이브러리를 사용하는 중 버그를 발견하거나, 필요한 기능이 없을 때 GitHub Issue를 발행한다.

**버그 발행**:
```bash
gh issue create --repo yusa-imit/sailor \
  --title "bug: <간단한 설명>" \
  --label "bug,from:silica" \
  --body "## 증상
<어떤 문제가 발생했는지>

## 재현 방법
<코드 또는 단계>

## 기대 동작
<어떻게 동작해야 하는지>

## 환경
- sailor 버전: <version>
- Zig 버전: 0.15.x
- OS: <os>"
```

**기능 요청 발행**:
```bash
gh issue create --repo yusa-imit/sailor \
  --title "feat: <필요한 기능>" \
  --label "feature-request,from:silica" \
  --body "## 필요한 이유
<silica에서 왜 이 기능이 필요한지>

## 제안하는 API
<원하는 함수 시그니처나 사용 예시>

## 현재 워크어라운드
<없으면 '없음'>"
```

**발행 조건**:
- sailor의 기존 API로 해결할 수 없는 문제일 때만 발행
- 동일한 이슈가 이미 열려있는지 먼저 확인: `gh issue list --repo yusa-imit/sailor --state open --search "<keyword>"`
- 이슈 발행 후 현재 작업으로 복귀 (sailor 수정을 직접 하지 않음)

**로컬 워크어라운드 금지 (CRITICAL)**:
- sailor에 버그가 있으면 **절대로 로컬에서 자체 구현으로 우회하지 않는다**
- 반드시 sailor repo에 이슈를 발행하고, sailor 에이전트가 수정할 때까지 기다린다
- sailor 에이전트(cron job)가 `from:*` 라벨 이슈를 최우선으로 처리한다
- 수정이 릴리스되면 `zig fetch --save`로 sailor 의존성을 업데이트한다
- 해당 기능이 아직 안 되면 그 기능을 사용하는 코드를 작성하지 않고 다른 작업으로 넘어간다

### v0.1.0 — arg, color (status: DONE)

sailor가 v0.1.0을 릴리즈하면 status가 READY로 변경된다.

**작업 내용**:
- [x] `build.zig.zon`에 sailor 의존성 추가
- [x] `build.zig`에 CLI executable 빌드 타겟 추가 (기존 library + 새 CLI)
- [x] `src/cli.zig` 생성 — `sailor.arg`로 CLI 진입점 구현
  - `silica <db_path>` — interactive SQL 셸 진입
  - `--help`, `--version`, `--header`, `--csv`, `--json` 플래그
- [x] 에러 출력에 `sailor.color` 적용
- [x] 기존 테스트 전체 통과 확인
- [x] 커밋: `feat: add CLI entry point with sailor v0.1.0`

### v0.2.0 — REPL + fmt (status: DONE)

**작업 내용**:
- [x] `sailor.repl`로 interactive SQL 셸 구현
  - 프롬프트: `silica> ` (기본), `   ...> ` (multi-line)
  - 히스토리: `~/.silica_history`
  - 자동완성: SQL 키워드, 테이블명, 컬럼명
  - 하이라이팅: SQL 키워드 색상
  - 멀티라인: `;`로 끝나지 않으면 계속 입력
- [x] SQL tokenizer/parser 연결 (silica 자체 모듈)
- [x] 커밋: `feat: add interactive SQL shell with sailor.repl`

### v0.3.0 — fmt (status: DONE)

**작업 내용**:
- [x] 쿼리 결과 포매팅에 `sailor.fmt` 적용
- [x] `.mode` 명령어 구현: table, csv, json, jsonl, plain
- [x] `SELECT` 결과를 정렬된 테이블로 표시
- [x] NULL 값 처리
- [x] 커밋: `feat: add output modes with sailor.fmt`

### v0.4.0 — tui (status: DONE)

**작업 내용**:
- [x] `silica --tui <db_path>` 모드 추가
- [x] `sailor.tui` 위젯으로 구현:
  - 좌측: 스키마 트리 (List 위젯 — 테이블 목록)
  - 우측 상단: 쿼리 결과 (Table 위젯)
  - 우측 하단: SQL 입력 (직접 구현 — sailor Input 위젯 버그로 인해)
  - 하단: StatusBar (직접 구현 — sailor StatusBar 위젯 버그로 인해)
- [x] 커밋: `feat: add TUI database browser with sailor.tui`
- **Note**: sailor v0.4.0의 Input/StatusBar 위젯에 API 불일치 버그 발견 → https://github.com/yusa-imit/sailor/issues/4

### v0.5.0 — advanced widgets (status: READY)

**작업 내용**:
- [x] `build.zig.zon`에 sailor v0.5.0 의존성 업데이트
- [x] `build.zig.zon`에 sailor v0.5.1 패치 업데이트 (sailor#3, #4, #5, #6 수정)
- [x] 스키마 사이드바 개선: 테이블+컬럼 계층 표시 (List 기반, 타입/제약조건 표시)
- [ ] 쿼리 플랜 시각화: `Tree` 위젯으로 EXPLAIN 결과 계층 표시
- [ ] SQL 편집기: `TextArea` 위젯으로 멀티라인 쿼리 에디터 교체
- [ ] 성능 차트: `LineChart`로 쿼리 실행 시간 추이 그래프
- [ ] 위험 쿼리 확인: `Dialog` 위젯으로 `DROP TABLE` 등 확인 프롬프트
- [ ] 결과 알림: `Notification`으로 쿼리 성공/실패 메시지
- [x] 커밋: `feat: enhance TUI schema sidebar with hierarchical table+column view`
- **Note**: sailor#7 filed — `renderDiff` still uses `std.fmt.format` (cross-compile fails). `localRenderDiff` workaround kept until fixed.
- **Note**: Phase 5 위젯 (Tree, TextArea, Dialog, Notification) now compile with v0.5.1 — ready for FEATURE mode integration

### v1.0.0 — production ready (status: READY)

**첫 안정 릴리즈**: 모든 기능 완성, 종합 문서화 포함

**v1.0.1 패치 릴리즈**: 크로스 컴파일 수정 (sailor#7 해결)
- `renderDiff` std.fmt.format → writer.print 수정
- x86_64-linux-gnu, x86_64-windows-msvc 크로스 컴파일 정상 동작 확인
- API 변경 없음 — drop-in replacement

**작업 내용**:
- [ ] `build.zig.zon`에 sailor v1.0.0 의존성 업데이트
- [ ] [Getting Started Guide](https://github.com/yusa-imit/sailor/blob/v1.0.0/docs/GUIDE.md) 참조하여 모범 사례 적용
- [ ] [API Reference](https://github.com/yusa-imit/sailor/blob/v1.0.0/docs/API.md) 기반으로 기존 코드 리팩토링
- [ ] 테마 시스템 활용: SQL TUI에 다크/라이트 모드 또는 SQL 신택스 하이라이팅 테마
- [ ] 애니메이션 효과 추가 (선택사항): 쿼리 실행 프로그레스, 결과 로딩
- [ ] 성능 최적화: sailor 벤치마크 기반으로 렌더링 성능 개선
- [ ] 기존 테스트 전체 통과 확인
- [ ] 커밋: `feat: upgrade to sailor v1.0.0 with theming and polish`

### v1.0.3 — bug fix release (status: READY)

**sailor v1.0.3 released** (2026-03-02) — Zig 0.15.2 compatibility patch

- **Bug fix**: Tree widget ArrayList API updated for Zig 0.15.2
- **Impact on silica**: None (silica doesn't use Tree widget)
- [ ] `build.zig.zon`에 sailor v1.0.3 의존성 업데이트 (optional, no breaking changes)
- [ ] 기존 테스트 전체 통과 확인

**Note**: This is an optional upgrade. Tree widget fix doesn't affect silica's current functionality.

### v1.1.0 — Accessibility & Internationalization (status: READY)

**sailor v1.1.0 released** (2026-03-02) — Accessibility and i18n features

- **New features**:
  - Accessibility module (screen reader hints, semantic labels)
  - Focus management system (tab order, focus ring)
  - Keyboard navigation protocol (custom key bindings)
  - Unicode width calculation (CJK, emoji proper sizing)
  - Bidirectional text support (RTL rendering for Arabic/Hebrew)
- **Impact on silica**: High priority — critical for SQL shell internationalization
  - Unicode width fixes essential for proper CJK/emoji data display
  - RTL support enables Arabic/Hebrew text in query results
  - Keyboard navigation improves SQL shell interactivity
  - Accessibility features enhance screen reader support for database tools
- [ ] `build.zig.zon`에 sailor v1.1.0 의존성 업데이트
- [ ] 기존 테스트 전체 통과 확인
- [ ] Consider keyboard bindings for SQL history navigation (Ctrl+R)
- [ ] Test Unicode width with CJK column data

**Note**: Non-breaking upgrade. Unicode/RTL improvements automatically benefit international database content display.
