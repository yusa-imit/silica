# Silica Project Memory

## Current State (Session 327)
- **Version**: v1.0.1 (production ready, all 12 phases complete)
- **Mode**: Maintenance — incremental improvements, test coverage, quality
- **Dependencies**: sailor v2.10.0 ✅, zuda v2.0.4 ✅ (both latest)
- **CI**: ✅ GREEN (100% pass rate)
- **Tests**: 2979+ passed, 31 skipped (+9 new json_agg/array_agg tests this session)
- **Source Files**: 55 (stable)

## Recent Session History

### Session 327 (2026-05-26) — FEATURE MODE
- **Focus**: json_agg and array_agg aggregate functions
- **Result**: ✅ Both aggregate functions implemented; 9 tests added (7 executor + 2 planner)
- **Commits**: deb6f42

### Session 326 (2026-05-26) — FEATURE MODE
- **Focus**: JSON scalar SQL functions (extract path, to_json)
- **Result**: ✅ json_extract_path, jsonb_extract_path, json_extract_path_text, jsonb_extract_path_text, to_json, to_jsonb implemented
- **Commits**: 47f69d2, aaecf9e

### Session 323 (2026-05-25) — FEATURE MODE
- **Focus**: JSON built-in SQL functions
- **Result**: ✅ json_key_exists, json_typeof, json_object_keys + jsonb aliases; json_set, json_insert; json_array_length, json_build_object, json_build_array, json_strip_nulls
- **Commits**: 7a16a15, 419439d, b3932ed

### Session 322 (2026-05-25) — FEATURE MODE
- **Focus**: Crash tests + $N parameter binding
- **Result**: ✅ 4/7 crash tests now real (was all skipped); $N params fully implemented
- **Commits**: 47c5a2f, 4ee795f, 9913a26

### Sessions 299-321 (compressed)
- GIN posting tree chains for high-cardinality keys
- Test quality improvements, stabilization sessions
- All sessions: ✅ CI green, no regressions

## Pattern: Maintenance Cycle
Since v1.0.0 release, sessions follow predictable pattern:
- **STABILIZATION** (every 5th): Full health audit — CI, dependencies, build, tests
- **FEATURE**: Incremental improvements — crash tests, SQL features, test quality
- **Stability**: 100% CI pass rate, zero regressions

## Next Session Priorities
1. **Watch for**: New sailor/zuda releases (`gh release list --repo yusa-imit/{sailor,zuda}`)
2. **Monitor**: GitHub issues for bug reports
3. **Next stabilization**: Session 330
4. **Potential work**: json_each (requires SRF support), row_to_json, index-only scan, remaining crash test skips

## Known TODOs in Codebase

### High-Value but Non-Trivial
- **Crash injection tests** (src/tx/crash_test.zig) — 4/7 active, 3 skipped (index update, double crash, after-WAL-before-DB)
- **Index-only scan optimization** (src/sql/optimizer.zig:2031) — requires catalog integration
- **json_each / json_object_keys SRF** — set-returning functions require SRF infrastructure

### Minor Enhancements
- JSON validation for CAST (reverted in 8a0fa13)
- Advanced SQL types (NUMERIC, UUID, ARRAY) (src/sql/engine.zig:1237)
- json_each, row_to_json (require SRF support)

## Project Conventions (Reinforced)
- **catch unreachable**: 177 instances, all justified with SAFETY comments
- **Zero warnings**: Strict compilation standards enforced
- **Test quality**: Focus on meaningful validation, not coverage numbers
- **Commit discipline**: Small, focused commits with descriptive messages
- **Memory compression**: Keep MEMORY.md under 200 lines, compress repetitive sessions
