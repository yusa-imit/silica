Implement a feature for the Silica project.

Feature description: $ARGUMENTS

Workflow:
1. **Understand**: Read `docs/PRD.md` and `CLAUDE.md` for context. Read `.claude/memory/` files for past decisions.
2. **Plan**: Enter plan mode. Identify which files need to be created/modified. Design the approach. Get user approval.
3. **Implement**: Write the code following Zig conventions from CLAUDE.md. Pay special attention to database-specific conventions (page safety, B+Tree invariants, allocator discipline).
4. **Test**: Write tests for all new functionality. Run `zig build test`.
5. **Review**: Self-review the changes against the code review checklist (correctness, integrity, safety, performance).
6. **Memory**: Update `.claude/memory/` with any architectural decisions or patterns discovered.
7. **Report**: Summarize what was implemented, files changed, tests added.

For complex features (3+ files), consider spawning a team:
- Use `zig-developer` agent for implementation
- Use `test-writer` agent for tests
- Use `code-reviewer` agent for review
- Use `architect` agent for design review (if architectural changes needed)
