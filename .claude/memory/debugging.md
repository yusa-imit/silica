# Silica — Debugging Notes

> Record bugs encountered, root causes, and solutions here.

## Known Issues
- (none yet — project in initial setup phase)

## Solved Problems
- (none yet)

## Common Zig Pitfalls for Database Development
- `std.testing.allocator` detects memory leaks — always use in tests
- File operations need explicit error handling for ENOSPC, EACCES
- Alignment matters for mmap — pages must be aligned to page_size
- `@memcpy` does not handle overlapping regions — use `@memmove` for in-place operations
- Integer overflow in page number arithmetic — use `std.math.add` for checked arithmetic

<!-- Add new debugging notes above this line -->
