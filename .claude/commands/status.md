Show the current status of the Silica project.

Gather and display:

1. **Git Status**:
   - Current branch
   - Uncommitted changes count
   - Last commit message and date
   - Ahead/behind remote

2. **Build Status**:
   - Run `zig build` and report success/failure
   - Library/binary size if exists

3. **Test Status**:
   - Run `zig build test` and report pass/fail count

4. **Project Progress**:
   - Read `docs/PRD.md` milestone checklist
   - Check which source files exist vs expected structure in CLAUDE.md
   - Report completion percentage per phase

5. **Memory Summary**:
   - Read `.claude/memory/` files
   - Report key recent decisions and known issues

Format output as a clear dashboard.
