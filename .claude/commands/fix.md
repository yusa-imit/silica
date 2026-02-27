Fix a bug in the Silica project.

Bug description: $ARGUMENTS

Workflow:
1. **Reproduce**: Understand the bug. If a command/test reproduces it, run it.
2. **Locate**: Use Grep/Glob to find relevant code. Read the source files.
3. **Analyze**: Identify root cause. Check `.claude/memory/debugging.md` for similar past issues. For database bugs, check B+Tree invariants, page integrity, and WAL consistency.
4. **Fix**: Apply the minimal fix needed. Don't refactor unrelated code.
5. **Test**: Ensure existing tests still pass. Add a regression test for this bug.
6. **Verify**: Run `zig build test` to confirm everything passes.
7. **Memory**: Record the bug and fix in `.claude/memory/debugging.md` for future reference.
8. **Report**: Summarize the root cause, the fix, and the regression test added.
