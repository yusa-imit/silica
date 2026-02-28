# Silica — Claude Code Orchestrator

> **Silica**: Zig로 작성된 프로덕션 등급 임베디드 관계형 데이터베이스 엔진
> Current Phase: **Phase 1 — Storage Foundation**

---

## Project Overview

- **Language**: Zig 0.14.x (stable)
- **Type**: Embedded relational database (SQLite-like)
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
    ├── storage/                 #   Storage Engine
    │   ├── page.zig             #     Page Manager (Pager)
    │   ├── btree.zig            #     B+Tree implementation
    │   ├── buffer_pool.zig      #     Buffer Pool (LRU cache)
    │   └── overflow.zig         #     Overflow page handling
    ├── sql/                     #   SQL Frontend
    │   ├── tokenizer.zig        #     Tokenizer (Lexer)
    │   ├── parser.zig           #     Recursive descent parser → AST
    │   ├── ast.zig              #     AST node definitions
    │   └── analyzer.zig         #     Semantic analysis
    ├── query/                   #   Query Engine
    │   ├── planner.zig          #     Query planner (logical plan)
    │   ├── optimizer.zig        #     Rule-based optimizer
    │   └── executor.zig         #     Volcano-model executor
    ├── tx/                      #   Transaction Manager
    │   ├── wal.zig              #     Write-Ahead Log
    │   ├── lock.zig             #     Lock Manager
    │   └── checkpoint.zig       #     WAL checkpoint
    ├── server/                  #   Client-Server Mode
    │   ├── wire.zig             #     Wire protocol
    │   └── connection.zig       #     Connection handling
    └── util/                    #   Utilities
        ├── checksum.zig         #     CRC32C checksums
        └── varint.zig           #     Variable-length integer encoding
```

> **Note**: `src/`, `build.zig`는 Phase 1 구현 시 생성됨. 현재는 문서·설정·CI만 존재.

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

**8단계 실행 사이클**:

| Phase | 내용 | 비고 |
|-------|------|------|
| 1. 상태 파악 | `/status` 실행, git log·빌드·테스트 상태 점검 | 체크리스트에서 다음 미완료 항목 식별 |
| 2. 계획 | 구현 전략을 내부적으로 수립 (텍스트 출력) | `EnterPlanMode`/`ExitPlanMode` 사용 금지 — 비대화형 세션에서 블로킹됨 |
| 3. 구현 → 검증 → 커밋 (반복) | 아래 **구현 루프** 참조 | 단위별로 즉시 커밋+푸시 |
| 4. 코드 리뷰 | `/review` — PRD 준수·메모리 안전성·테스트 커버리지 확인 | 이슈 발견 시 수정 후 재커밋 |
| 5. 메모리 갱신 | `.claude/memory/` 파일 업데이트 | 별도 커밋: `chore: update session memory` → push |
| 6. 세션 요약 | 구조화된 요약 출력 | 아래 템플릿 참조 |

**구현 루프** (Phase 3 상세):

작업을 작은 단위로 분할하고, 각 단위마다 다음을 반복한다:
1. 코드 작성 (하나의 모듈/파일 단위)
2. 테스트 작성 및 `zig build test` 통과 확인
3. 즉시 커밋 + `git push` — 다음 단위로 넘어가기 전에 반드시 수행
- 미커밋 변경사항을 여러 파일에 걸쳐 누적하지 않는다
- 한 사이클 내에 완료할 수 없는 작업은 동작하는 중간 상태로 커밋+푸시한다
- `git add -A` 금지 — 변경된 파일을 명시적으로 지정

**작업 선택 규칙**:
- `build.zig`가 없으면 프로젝트 부트스트랩부터 시작
- 이전 세션의 미커밋 변경사항이 있으면: 테스트 통과 시 커밋+푸시, 실패 시 폐기
- 테스트 실패 중이면 새 기능 추가 전에 수정
- 의존성 순서 준수: Storage → SQL → Query → Transaction → Server
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
- **Allocator discipline**: Storage engine uses arena per-transaction; buffer pool uses page-aligned allocator.

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

## Phase 1 Implementation Roadmap

현재 Phase 1 (Storage Foundation) 구현 중. 우선순위:

1. **Page Manager & File Format** — `src/storage/page.zig`
   - Database file header (Page 0) — Magic: "SLCA"
   - Page read/write with CRC32C checksums
   - Freelist management
   - 기본 테스트: DB 생성, 페이지 쓰기, 재오픈 후 검증

2. **B+Tree** — `src/storage/btree.zig`
   - Insert, delete, point lookup
   - Leaf page splits/merges
   - Range scan cursors (forward/backward)
   - Overflow pages for large values

3. **Buffer Pool** — `src/storage/buffer_pool.zig`
   - LRU page cache (default: 2000 pages ≈ 8 MB)
   - Dirty page tracking
   - Pin/unpin semantics

4. **Utilities** — `src/util/`
   - CRC32C checksum
   - Varint encoding/decoding

---

## Quick Reference

```bash
# Build (src/ 생성 후)
zig build

# Test
zig build test

# Run (embedded library — test harness)
zig build run

# Cross-compile (example)
zig build -Dtarget=x86_64-linux -Doptimize=ReleaseSafe

# Clean
rm -rf zig-out .zig-cache

# Benchmark (Phase 2+)
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
- Zig 버전: 0.14.x
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

### v0.4.0 — tui (status: READY)

**작업 내용**:
- [ ] `silica --tui <db_path>` 모드 추가
- [ ] `sailor.tui` 위젯으로 구현:
  - 좌측: 스키마 트리 (Tree 위젯 — 테이블/컬럼/인덱스)
  - 우측 상단: 쿼리 결과 (Table 위젯)
  - 우측 하단: SQL 입력 (Input 위젯)
  - 하단: StatusBar (행 수, 쿼리 시간, DB 크기)
- [ ] 커밋: `feat: add TUI database browser with sailor.tui`

### v0.5.0 — advanced widgets (status: READY)

**작업 내용**:
- [ ] `build.zig.zon`에 sailor v0.5.0 의존성 업데이트
- [ ] 쿼리 플랜 시각화: `Tree` 위젯으로 EXPLAIN 결과 계층 표시
- [ ] SQL 편집기: `TextArea` 위젯으로 멀티라인 쿼리 에디터 교체
- [ ] 성능 차트: `LineChart`로 쿼리 실행 시간 추이 그래프
- [ ] 위험 쿼리 확인: `Dialog` 위젯으로 `DROP TABLE` 등 확인 프롬프트
- [ ] 결과 알림: `Notification`으로 쿼리 성공/실패 메시지
- [ ] 커밋: `feat: enhance TUI with advanced widgets from sailor v0.5.0`
