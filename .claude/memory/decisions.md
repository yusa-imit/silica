# Silica — Technical Decisions

> Format: Date | Context | Decision | Rationale | Consequences

## Decision: Project Language — Zig
- **Date**: 2026-02-27
- **Context**: Need an embedded DB with no hidden allocations, C ABI interop, cross-compilation
- **Decision**: Use Zig 0.14.x stable
- **Rationale**: Explicit memory control, comptime metaprogramming, zero-overhead C FFI, no GC
- **Consequences**: Smaller ecosystem than C/Rust; must track Zig stable releases

## Decision: File Format — Single-file SQLite-style
- **Date**: 2026-02-27
- **Context**: Need simple deployment and backup model for embedded use
- **Decision**: Single database file with fixed-size pages, magic "SLCA"
- **Rationale**: SQLite's single-file model is proven for embeddability
- **Consequences**: WAL file is separate; need careful page-level locking

## Decision: B+Tree as Primary Index Structure
- **Date**: 2026-02-27
- **Context**: Need efficient point lookups and range scans
- **Decision**: B+Tree with doubly-linked leaf pages
- **Rationale**: Standard for OLTP databases; leaf links enable fast range scans
- **Consequences**: Must handle splits, merges, overflow pages correctly

## Decision: Buffer Pool LRU — Keep Custom, Do Not Migrate to zuda
- **Date**: 2026-03-26
- **Context**: zuda v1.23.0 provides LRUCache with pin/unpin (zuda#9 resolved). Evaluated feasibility of replacing silica's buffer pool LRU eviction with zuda's generic implementation.
- **Decision**: DO NOT MIGRATE. Keep the custom buffer pool implementation.
- **Rationale**: Four blocking incompatibilities: (1) zuda's eviction callback is non-failable (void), but dirty page flush can fail — silent data loss risk; (2) per-entry heap allocation vs. silica's pre-allocated frame array — unacceptable allocation churn in hot path; (3) BufferFrame.data is directly accessed as raw []u8 by 12+ files — adapter layer would add overhead to every B+Tree operation; (4) the replaceable LRU logic is only ~30 lines, not worth the coupling.
- **Consequences**: Buffer pool remains self-contained with custom LRU implementation. Future zuda improvements (failable callbacks, pre-allocated pools) could reopen this decision. Deadlock detection in lock.zig identified as better zuda migration target.
- **Review**: Architect agent review (Session 27, agent ID a3298b8)

<!-- Add new decisions above this line -->
