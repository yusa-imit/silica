Prepare and execute a release for the Silica project.

Version: $ARGUMENTS (e.g., "v0.1.0")

Workflow:
1. **Pre-flight checks**:
   - Run `zig build test` — all tests must pass
   - Run `git status` — working tree must be clean
   - Check current branch (should be on main or release branch)
2. **Version bump**: Update version in `build.zig.zon` if applicable
3. **Build verification**: Cross-compile for all 6 targets to verify builds
4. **Changelog**: Generate summary of changes since last tag
5. **Commit**: Commit version bump with `chore: bump version to <version>`
6. **PR**: Create PR to main with release notes
7. **Tag**: After merge, tag the release
8. **Report**: Summary of release steps completed

Note: Actual tagging and pushing should be confirmed with user before execution.
