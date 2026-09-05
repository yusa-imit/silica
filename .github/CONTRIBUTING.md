# Contributing to Silica

Thank you for considering contributing to Silica! We welcome contributions from the community.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)
- [License](#license)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please be respectful and constructive in all interactions.

## Getting Started

### Prerequisites

- Zig 0.15.x (stable)
- Git
- Basic knowledge of database internals (helpful but not required)

### Building from Source

```bash
# Clone the repository
git clone https://github.com/yusa-imit/silica.git
cd silica

# Build the project
zig build

# Run tests
zig build test

# Run the CLI
zig build run -- mydb.db
```

### Project Structure

```
silica/
├── src/           # Source code
│   ├── storage/   # Storage engine (B+Tree, Buffer Pool, WAL)
│   ├── sql/       # SQL frontend (Tokenizer, Parser, Executor)
│   ├── tx/        # Transaction management (MVCC, Locks)
│   ├── server/    # PostgreSQL wire protocol server
│   └── ...
├── docs/          # Documentation
├── examples/      # SQL examples
├── bench/         # Benchmark suites (TPC-C, TPC-H)
└── tests/         # Integration tests
```

## Development Workflow

### 1. Fork and Clone

Fork the repository on GitHub and clone your fork:

```bash
git clone https://github.com/YOUR-USERNAME/silica.git
cd silica
git remote add upstream https://github.com/yusa-imit/silica.git
```

### 2. Create a Branch

Create a feature branch for your work:

```bash
git checkout -b feat/my-feature
# or
git checkout -b fix/my-bugfix
```

Branch naming convention:
- `feat/` — New features
- `fix/` — Bug fixes
- `refactor/` — Code refactoring
- `test/` — Test additions/improvements
- `docs/` — Documentation changes
- `chore/` — Build, CI, dependencies

### 3. Make Changes

Write your code following the [Coding Standards](#coding-standards).

### 4. Test Your Changes

Always test your changes before submitting:

```bash
# Run full test suite
zig build test

# Run specific tests
zig test src/storage/btree.zig

# Run benchmarks (if applicable)
zig build bench
```

### 5. Commit Your Changes

Write clear, descriptive commit messages:

```
<type>: <subject>

<body>

Co-Authored-By: Your Name <your.email@example.com>
```

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`, `ci`

Examples:
```
feat: add UPSERT support for INSERT statements

Implements INSERT ... ON CONFLICT DO UPDATE/NOTHING syntax
for handling unique constraint violations gracefully.

Co-Authored-By: Alice Developer <alice@example.com>
```

```
fix: resolve deadlock in concurrent index builds

Acquire locks in consistent order: table lock before index lock.
Prevents circular wait condition in concurrent CREATE INDEX.

Closes #123

Co-Authored-By: Bob Developer <bob@example.com>
```

### 6. Push and Create Pull Request

```bash
git push origin feat/my-feature
```

Then open a pull request on GitHub.

## Coding Standards

### Zig Conventions

- **Naming**:
  - Functions/variables: `camelCase`
  - Types: `PascalCase`
  - Constants: `SCREAMING_SNAKE_CASE`
- **Error Handling**: Always use explicit error unions. Never `catch unreachable` in production code unless documented with SAFETY comments explaining the invariant.
- **Memory Management**:
  - Prefer arena allocators for request-scoped work
  - Use GPA for long-lived allocations
  - Always defer `deinit()` for owned resources
- **Testing**: All public functions must have tests. Use TDD: write failing tests first, then implement.
- **Comments**: Only add comments where logic is non-obvious. No doc comments on self-explanatory functions.

### Database-Specific Conventions

- **Page Operations**: Always check bounds before read/write. Use CRC32C for integrity.
- **B+Tree Invariants**: Verify sorted keys, valid pointers, balanced depth after every operation.
- **Buffer Pool**: Pin/unpin must be balanced. Use `defer pool.unpin(page)` pattern.
- **WAL**: Never modify main DB file directly — always through WAL first.
- **MVCC**: Every row version must carry `(xmin, xmax)` transaction IDs. Visibility checks are mandatory.
- **Isolation Correctness**: Never weaken an isolation level's guarantees.
- **Wire Protocol**: Follow PostgreSQL wire protocol v3 exactly — byte-level compatibility required.

### Code Organization

- One module per file
- Keep files under 500 lines (split into submodules if exceeded)
- Public API at top, private helpers at bottom
- Tests at bottom of each file within `test` blocks

### Error Messages

User-facing errors must be clear and actionable:

```
✗ [Context]: [What happened]

  [Details with syntax highlighting]

  Hint: [Actionable suggestion]
```

Example:
```
✗ Parser: Unexpected token 'FROM' at line 3, column 8

  SELECT id, name, FROM users;
                    ^^^^

  Hint: Expected column name or expression before 'FROM'
```

## Testing

### Test Categories

1. **Unit Tests**: In-file tests at bottom of each module
2. **Integration Tests**: Cross-module tests in `src/sql/engine.zig`, etc.
3. **Fuzz Tests**: In `src/storage/fuzz.zig`, `src/sql/parser_fuzz.zig`
4. **Conformance Tests**: SQL:2016 compliance in `src/sql/conformance_test.zig`
5. **Benchmarks**: Performance tests in `bench/`

### Writing Good Tests

- Use descriptive test names: `test "B+Tree handles leaf splits correctly"`
- Test edge cases: empty input, max values, boundary conditions
- Test error paths: invalid input, conflicts, resource exhaustion
- Use `std.testing.allocator` to detect memory leaks
- Keep tests focused: one assertion per logical check

Example:

```zig
test "Buffer Pool pins and unpins correctly" {
    const allocator = std.testing.allocator;
    var pool = try BufferPool.init(allocator, 100);
    defer pool.deinit();

    // Pin page
    const page = try pool.getPage(1);
    defer pool.unpin(page); // Always defer unpin

    // Verify page is pinned
    try std.testing.expectEqual(1, page.pin_count);

    // Modify and mark dirty
    page.data[0] = 42;
    page.markDirty();

    // Unpin happens via defer
}
```

### Test Coverage

Aim for high coverage of critical code paths:
- Storage layer: 100% (data integrity critical)
- SQL parser: 90%+ (correctness essential)
- Transaction management: 100% (ACID properties)
- Server protocol: 80%+ (compatibility important)

Run tests frequently:

```bash
# During development
zig build test

# Before submitting PR
zig build test --summary all
```

## Pull Request Process

### Before Submitting

- [ ] All tests pass (`zig build test`)
- [ ] No compiler warnings
- [ ] Code follows style guide
- [ ] New tests added for new functionality
- [ ] Documentation updated (if applicable)
- [ ] Commit messages follow convention

### PR Guidelines

1. **Title**: Use conventional commit format
   - Example: `feat: add LATERAL JOIN support`

2. **Description**: Include:
   - Summary of changes (bullet points)
   - Test plan
   - Related issues (Closes #123)

3. **Size**: Keep PRs focused and reasonably sized
   - Prefer small, incremental PRs over large refactors
   - If adding a large feature, consider breaking into multiple PRs

4. **Review**: Address review comments promptly
   - Be open to feedback
   - Ask questions if unclear
   - Update PR based on feedback

### Example PR Description

```markdown
## Summary
- Implements LATERAL JOIN syntax for correlated subqueries
- Adds executor support for lateral correlation
- Updates optimizer to handle lateral references

## Test Plan
- Added 12 new tests covering:
  - Basic LATERAL JOIN with scalar subquery
  - LATERAL with table-valued function
  - Multiple LATERAL joins in same query
  - Error cases: invalid lateral references

## Related Issues
Closes #456

## Performance Impact
No measurable impact on non-lateral queries. LATERAL queries execute correctly with expected cost model.
```

## Issue Guidelines

### Reporting Bugs

Use the bug report template and include:

- **Description**: Clear explanation of the problem
- **Steps to Reproduce**: Minimal reproducer
- **Expected Behavior**: What should happen
- **Actual Behavior**: What actually happens
- **Environment**: OS, Zig version, Silica version
- **Logs/Errors**: Relevant error messages or stack traces

Example:

```markdown
**Description**
SELECT with ORDER BY on indexed column returns wrong results

**Steps to Reproduce**
CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);
INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob'), (3, 'Charlie');
CREATE INDEX idx_name ON users(name);
SELECT * FROM users ORDER BY name;

**Expected Behavior**
Returns rows sorted alphabetically: Alice, Bob, Charlie

**Actual Behavior**
Returns rows in insertion order: Alice, Bob, Charlie

**Environment**
- OS: macOS 14.2
- Zig: 0.15.0
- Silica: v1.0.0
```

### Feature Requests

Use the feature request template and include:

- **Description**: What feature you'd like
- **Use Case**: Why it's needed
- **Examples**: How it would be used
- **Alternatives**: Other solutions you've considered

Example:

```markdown
**Description**
Add support for partial indexes (indexes with WHERE clause)

**Use Case**
For large tables, indexing only a subset of rows (e.g., active users)
reduces index size and improves performance.

**Examples**
CREATE INDEX idx_active_users ON users(name) WHERE active = true;

**Alternatives**
- Filtered queries (less efficient)
- Materialized views (requires storage)
- Application-level filtering
```

### Good First Issues

Look for issues labeled `good-first-issue` for beginner-friendly contributions.

## Development Tips

### Debugging

Use Zig's built-in debugging tools:

```bash
# Debug build
zig build

# Run with debug info
zig build run -- --debug

# Use std.log for debugging
std.log.debug("Page {d} pin_count: {d}", .{page_num, page.pin_count});
```

### Performance Profiling

Use the benchmark suite:

```bash
# Run simple benchmark
zig build bench

# Run TPC-C (OLTP)
zig build tpcc

# Run TPC-H (OLAP)
zig build tpch
```

### CI/CD

GitHub Actions runs automatically on PRs:
- Build verification
- Full test suite
- Cross-compilation (Linux, macOS, Windows, RISC-V)
- Benchmark comparison

Check CI results before merging.

## Communication

- **Issues**: For bugs and feature requests
- **Pull Requests**: For code contributions
- **Discussions**: For questions and ideas (use GitHub Discussions)

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (see LICENSE file).

---

Thank you for contributing to Silica! Your efforts help make this project better for everyone.
