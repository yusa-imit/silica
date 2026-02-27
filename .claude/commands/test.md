Run tests for the Silica project.

Steps:
1. Run `zig build test` in the project root
2. If any tests fail:
   a. Parse the failure output to identify which test(s) failed
   b. Read the relevant test file(s)
   c. Analyze the failure cause
   d. Report findings with file:line references
3. If all tests pass, report success with test count

Optional: $ARGUMENTS
- If user specifies a module (e.g., "storage", "btree", "sql"), only test that module
- If user says "verbose", add `--summary all` flag
- If user says "coverage", report which modules have tests and which don't
- If user says "fuzz", run fuzz tests (if available)
