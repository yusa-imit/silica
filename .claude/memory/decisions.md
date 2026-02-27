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

<!-- Add new decisions above this line -->
