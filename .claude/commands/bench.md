Run performance benchmarks for the Silica project.

Steps:
1. Verify `zig build bench` target exists; if not, report that benchmarks are not yet configured
2. Run benchmarks and capture results
3. Compare against PRD performance targets (Section 7.1):
   - Point lookup (by PK): < 5 us (cached)
   - Sequential insert: > 100K rows/sec
   - Range scan throughput: > 500K rows/sec
   - Database open time: < 10 ms (1 GB DB)
   - Binary size (stripped): < 2 MB
   - Memory overhead (idle): < 1 MB + cache
4. Report results with pass/fail for each target

Optional: $ARGUMENTS
- If user specifies a component (e.g., "btree", "buffer_pool"), focus on that
- If user says "compare", compare with previous benchmark results
- If user says "profile", add timing breakdowns
