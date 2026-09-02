//! MATCH_RECOGNIZE (SQL:2016) — Pure row pattern matching engine.
//!
//! A standalone backtracking regex-style matcher over PatternNode, with ZERO dependency
//! on Row/Value/executor.zig/Database. Caller supplies a predicate callback to evaluate
//! DEFINE conditions; pattern_match.zig handles the matching logic, backtracking, and
//! memory management of the result.

const std = @import("std");
const ast = @import("ast.zig");

/// A single successful match of a pattern over a contiguous row range.
pub const Match = struct {
    /// First row index (inclusive) consumed by this match, relative to the caller's row numbering.
    start: usize,
    /// One-past-the-last row index consumed by this match.
    end_exclusive: usize,
    /// variable_per_row[i] is the pattern variable name bound to row (start + i).
    /// Length is always (end_exclusive - start). Allocated with the `allocator` passed to findMatch;
    /// caller owns it and must free both the slice and each contained []const u8 is NOT owned
    /// separately — the strings themselves borrow from the PatternNode's variable names (no need to
    /// free the individual strings, only the outer slice via `allocator.free(match.variable_per_row)`).
    variable_per_row: []const []const u8,
};

/// Caller-supplied predicate: "does row `row_idx` satisfy pattern variable `variable`'s DEFINE
/// condition, given the tentative bindings assigned so far in this in-progress match attempt?"
/// `bindings_so_far` is indexed by absolute row_idx and covers exactly the rows already tentatively
/// consumed before `row_idx` in the CURRENT match attempt (i.e. bindings_so_far.len == row_idx - match_start,
/// and bindings_so_far[k] is the variable bound to row (match_start + k)). This lets a real DEFINE
/// condition implement PREV()/FIRST() by looking at prior tentative bindings, without pattern_match.zig
/// knowing anything about Row/Value/expressions — it just threads this array through.
/// Returns true if `variable` can match at `row_idx`.
pub const MatchContext = struct {
    ptr: *anyopaque,
    tryVariableFn: *const fn (ptr: *anyopaque, variable: []const u8, row_idx: usize, match_start: usize, bindings_so_far: []const []const u8) bool,

    pub fn tryVariable(self: MatchContext, variable: []const u8, row_idx: usize, match_start: usize, bindings_so_far: []const []const u8) bool {
        return self.tryVariableFn(self.ptr, variable, row_idx, match_start, bindings_so_far);
    }
};

/// Attempts to match `pattern` starting exactly at row index `start` (NOT a search over all start
/// positions — the caller tries successive start positions itself). Rows exist in range [0, row_count).
/// Semantics: alternation tries branches in listed order, first successful branch wins (no backtracking
/// into an earlier successful alternative once a later required part of the pattern fails — matches
/// SQL:2016 "first match" semantics, NOT POSIX leftmost-longest). Quantifiers are greedy: `+`/`*` try
/// to consume as many repetitions as possible first, then backtrack (give back rows) one at a time if
/// a later part of the pattern (in a concat) cannot otherwise match. `?` tries one repetition before
/// zero. Returns the resulting Match on success (allocated with `allocator`), or null if no match is
/// possible starting at `start` after exhausting all backtracking options. Returns `error.OutOfMemory`
/// only for allocation failures, never as a "no match" signal.
pub fn findMatch(allocator: std.mem.Allocator, pattern: *const ast.PatternNode, row_count: usize, start: usize, ctx: MatchContext) std.mem.Allocator.Error!?Match {
    // Use an arena for temporary state during matching to simplify cleanup on backtracking
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_allocator = arena.allocator();

    var bindings = std.ArrayListUnmanaged([]const u8){};

    if (try tryMatchNode(pattern, start, start, &bindings, row_count, ctx, arena_allocator)) |end_pos| {
        // Match succeeded! Copy the bindings to the caller's allocator
        const variable_per_row = try allocator.dupe([]const u8, bindings.items);
        return Match{
            .start = start,
            .end_exclusive = end_pos,
            .variable_per_row = variable_per_row,
        };
    }

    return null;
}

/// Internal: attempt to match a single node starting at `pos`, accumulating variable bindings.
/// Returns the position after a successful match, or null if the match fails.
fn tryMatchNode(
    node: *const ast.PatternNode,
    pos: usize,
    match_start: usize,
    bindings: *std.ArrayListUnmanaged([]const u8),
    row_count: usize,
    ctx: MatchContext,
    allocator: std.mem.Allocator,
) std.mem.Allocator.Error!?usize {
    switch (node.*) {
        .variable => |name| {
            if (pos >= row_count) return null;
            if (ctx.tryVariable(name, pos, match_start, bindings.items)) {
                try bindings.append(allocator, name);
                return pos + 1;
            }
            return null;
        },

        .group => |child| {
            return tryMatchNode(child, pos, match_start, bindings, row_count, ctx, allocator);
        },

        .concat => |children| {
            return tryMatchConcat(children, 0, pos, match_start, bindings, row_count, ctx, allocator);
        },

        .alternation => |branches| {
            for (branches) |branch| {
                // Try this branch. If it succeeds, we're done (no backtracking to other branches).
                // If it fails, try the next branch.
                var bindings_copy = try bindings.clone(allocator);
                if (try tryMatchNode(branch, pos, match_start, &bindings_copy, row_count, ctx, allocator)) |next_pos| {
                    // Success! Update the original bindings and return.
                    bindings.clearRetainingCapacity();
                    for (bindings_copy.items) |b| {
                        try bindings.append(allocator, b);
                    }
                    return next_pos;
                }
                // Branch failed, try next
            }
            return null;
        },

        .quantified => |q| {
            return tryMatchQuantified(q.node, q.quantifier, pos, match_start, bindings, row_count, ctx, allocator);
        },
    }
}

/// Internal: match a sequence of concatenated children, starting at child index `child_idx`.
/// For quantified children, we handle backtracking explicitly: try all possible repetition counts
/// (greedy first) and for each, try to match the rest of the concat.
fn tryMatchConcat(
    children: []const *const ast.PatternNode,
    child_idx: usize,
    pos: usize,
    match_start: usize,
    bindings: *std.ArrayListUnmanaged([]const u8),
    row_count: usize,
    ctx: MatchContext,
    allocator: std.mem.Allocator,
) std.mem.Allocator.Error!?usize {
    if (child_idx >= children.len) {
        // All children matched successfully
        return pos;
    }

    const first_child = children[child_idx];
    const remaining_children = children[child_idx + 1 ..];

    // Special handling for quantified children: try all possible repetition counts
    if (first_child.* == .quantified) {
        const q = first_child.quantified;
        const endpoints = try getAllQuantifierEndpoints(q.node, q.quantifier, pos, match_start, bindings, row_count, ctx, allocator);
        defer {
            for (endpoints) |ep| {
                allocator.free(ep.bindings_snapshot);
            }
            allocator.free(endpoints);
        }

        // Try endpoints in order (greedy first), and for each, try to match the rest of the concat
        for (endpoints) |endpoint| {
            var bindings_temp = std.ArrayListUnmanaged([]const u8){};
            for (endpoint.bindings_snapshot) |b| {
                try bindings_temp.append(allocator, b);
            }
            if (try tryMatchConcat(remaining_children, 0, endpoint.pos, match_start, &bindings_temp, row_count, ctx, allocator)) |final_pos| {
                // Success! Update the original bindings and return
                bindings.clearRetainingCapacity();
                for (bindings_temp.items) |b| {
                    try bindings.append(allocator, b);
                }
                bindings_temp.deinit(allocator);
                return final_pos;
            }
            bindings_temp.deinit(allocator);
            // This endpoint didn't work, try the next one
        }
        return null;
    }

    // Non-quantified child: match it normally
    if (try tryMatchNode(first_child, pos, match_start, bindings, row_count, ctx, allocator)) |next_pos| {
        // First child matched, continue with remaining children
        return tryMatchConcat(remaining_children, 0, next_pos, match_start, bindings, row_count, ctx, allocator);
    }

    return null;
}

/// Internal: match a quantified pattern (zero_or_more, one_or_more, zero_or_one).
/// When called directly (not from concat), returns the greedy match.
fn tryMatchQuantified(
    node: *const ast.PatternNode,
    quantifier: ast.PatternQuantifier,
    pos: usize,
    match_start: usize,
    bindings: *std.ArrayListUnmanaged([]const u8),
    row_count: usize,
    ctx: MatchContext,
    allocator: std.mem.Allocator,
) std.mem.Allocator.Error!?usize {
    const endpoints = try getAllQuantifierEndpoints(node, quantifier, pos, match_start, bindings, row_count, ctx, allocator);
    defer allocator.free(endpoints);

    if (endpoints.len > 0) {
        // Return the greedy endpoint (first one) and restore its bindings
        bindings.clearRetainingCapacity();
        for (endpoints[0].bindings_snapshot) |b| {
            try bindings.append(allocator, b);
        }
        return endpoints[0].pos;
    }

    return null;
}

const QuantifierEndpoint = struct {
    pos: usize,
    bindings_snapshot: []const []const u8, // Borrowed from allocator, caller must free
};

/// Internal: compute all possible endpoints for a quantified pattern.
/// Returns endpoints sorted from greedy to less-greedy, each with the corresponding bindings.
fn getAllQuantifierEndpoints(
    node: *const ast.PatternNode,
    quantifier: ast.PatternQuantifier,
    pos: usize,
    match_start: usize,
    bindings: *std.ArrayListUnmanaged([]const u8),
    row_count: usize,
    ctx: MatchContext,
    allocator: std.mem.Allocator,
) std.mem.Allocator.Error![]QuantifierEndpoint {
    const min_count: usize = switch (quantifier) {
        .one_or_more => 1,
        .zero_or_more => 0,
        .zero_or_one => 0,
    };

    const max_one_repetition = quantifier == .zero_or_one;

    // Collect all endpoints by greedily matching and recording state at each step
    var endpoints = std.ArrayListUnmanaged(QuantifierEndpoint){};
    defer endpoints.deinit(allocator);

    // Record the initial state (zero repetitions)
    var initial_bindings = try bindings.clone(allocator);
    try endpoints.append(allocator, .{
        .pos = pos,
        .bindings_snapshot = try allocator.dupe([]const u8, initial_bindings.items),
    });
    initial_bindings.deinit(allocator);

    // Greedily match as many repetitions as possible, recording each step
    var current_pos = pos;
    var current_bindings = try bindings.clone(allocator);
    var count: usize = 0;

    while (count < 1000) { // Safety limit
        if (max_one_repetition and count > 0) break;

        var bindings_for_attempt = try current_bindings.clone(allocator);
        if (try tryMatchNode(node, current_pos, match_start, &bindings_for_attempt, row_count, ctx, allocator)) |next_pos| {
            count += 1;
            current_pos = next_pos;
            current_bindings.deinit(allocator);
            current_bindings = bindings_for_attempt;

            // Record this endpoint
            try endpoints.append(allocator, .{
                .pos = next_pos,
                .bindings_snapshot = try allocator.dupe([]const u8, current_bindings.items),
            });
        } else {
            bindings_for_attempt.deinit(allocator);
            break;
        }
    }

    current_bindings.deinit(allocator);

    // Filter: keep only endpoints that satisfy the minimum count
    var result = std.ArrayListUnmanaged(QuantifierEndpoint){};
    var i = endpoints.items.len;
    while (i > 0) {
        i -= 1;
        if (i >= min_count) {
            try result.append(allocator, endpoints.items[i]);
        } else {
            // Free bindings snapshot that we're not keeping
            allocator.free(endpoints.items[i].bindings_snapshot);
        }
    }

    return result.items;
}


// ==============================================================================
// TEST HELPERS — used below to construct PatternNode trees without the parser
// ==============================================================================

/// Test helper: construct a PatternNode for a simple variable reference.
/// Allocates from gpa; caller must free the returned pointer.
fn testMakeVariable(allocator: std.mem.Allocator, name: []const u8) std.mem.Allocator.Error!*const ast.PatternNode {
    const node = try allocator.create(ast.PatternNode);
    node.* = .{ .variable = name };
    return node;
}

/// Test helper: construct a PatternNode for concatenation of children.
/// Allocates from gpa; caller must free the result and all children.
fn testMakeConcat(allocator: std.mem.Allocator, children: []const *const ast.PatternNode) std.mem.Allocator.Error!*const ast.PatternNode {
    const node = try allocator.create(ast.PatternNode);
    const children_copy = try allocator.dupe(*const ast.PatternNode, children);
    node.* = .{ .concat = children_copy };
    return node;
}

/// Test helper: construct a PatternNode for alternation of branches.
/// Allocates from gpa; caller must free the result and all branches.
fn testMakeAlternation(allocator: std.mem.Allocator, branches: []const *const ast.PatternNode) std.mem.Allocator.Error!*const ast.PatternNode {
    const node = try allocator.create(ast.PatternNode);
    const branches_copy = try allocator.dupe(*const ast.PatternNode, branches);
    node.* = .{ .alternation = branches_copy };
    return node;
}

/// Test helper: construct a PatternNode for a quantified sub-pattern.
/// Allocates from gpa; caller must free the result.
fn testMakeQuantified(allocator: std.mem.Allocator, node: *const ast.PatternNode, quantifier: ast.PatternQuantifier) std.mem.Allocator.Error!*const ast.PatternNode {
    const result = try allocator.create(ast.PatternNode);
    result.* = .{ .quantified = .{ .node = node, .quantifier = quantifier } };
    return result;
}

/// Test helper: construct a PatternNode for a grouped sub-pattern.
fn testMakeGroup(allocator: std.mem.Allocator, node: *const ast.PatternNode) std.mem.Allocator.Error!*const ast.PatternNode {
    const result = try allocator.create(ast.PatternNode);
    result.* = .{ .group = node };
    return result;
}

/// Test context helper: a simple table-driven context where each row has a fixed set of
/// allowed variable names. Used to define "row i allows variable X" declaratively.
const TestContext = struct {
    /// allowed_per_row[i] is a slice of allowed variable names at row i.
    /// If a variable is not in this slice, tryVariable returns false.
    allowed_per_row: []const []const []const u8,

    fn tryVariableImpl(ptr: *anyopaque, variable: []const u8, row_idx: usize, _: usize, _: []const []const u8) bool {
        const self: *TestContext = @ptrCast(@alignCast(ptr));
        if (row_idx >= self.allowed_per_row.len) return false;
        const allowed = self.allowed_per_row[row_idx];
        for (allowed) |v| {
            if (std.mem.eql(u8, v, variable)) return true;
        }
        return false;
    }

    fn toMatchContext(self: *TestContext) MatchContext {
        return .{
            .ptr = self,
            .tryVariableFn = TestContext.tryVariableImpl,
        };
    }
};

// ==============================================================================
// TESTS
// ==============================================================================

test "pattern_match: simple concatenation A B C" {
    const allocator = std.testing.allocator;
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const gpa_allocator = gpa.allocator();

    // Build pattern nodes
    const a_node = try testMakeVariable(gpa_allocator, "A");
    defer gpa_allocator.destroy(a_node);
    const b_node = try testMakeVariable(gpa_allocator, "B");
    defer gpa_allocator.destroy(b_node);
    const c_node = try testMakeVariable(gpa_allocator, "C");
    defer gpa_allocator.destroy(c_node);

    const pattern = try testMakeConcat(gpa_allocator, &.{ a_node, b_node, c_node });
    defer {
        gpa_allocator.free(pattern.concat);
        gpa_allocator.destroy(pattern);
    }

    // Set up test context: row 0 allows "A", row 1 allows "B", row 2 allows "C"
    var ctx_impl = TestContext{
        .allowed_per_row = &.{
            &.{"A"},
            &.{"B"},
            &.{"C"},
        },
    };

    // Attempt match starting at row 0
    const result = try findMatch(allocator, pattern, 3, 0, ctx_impl.toMatchContext());

    // This test FAILS currently (via panic), but when implemented should:
    // - Return non-null match
    // - start == 0, end_exclusive == 3
    // - variable_per_row == ["A", "B", "C"]
    if (result) |match| {
        defer allocator.free(match.variable_per_row);
        try std.testing.expectEqual(@as(usize, 0), match.start);
        try std.testing.expectEqual(@as(usize, 3), match.end_exclusive);
        try std.testing.expectEqual(@as(usize, 3), match.variable_per_row.len);
        try std.testing.expectEqualStrings("A", match.variable_per_row[0]);
        try std.testing.expectEqualStrings("B", match.variable_per_row[1]);
        try std.testing.expectEqualStrings("C", match.variable_per_row[2]);
    } else {
        try std.testing.expect(false); // should not be null
    }
}

test "pattern_match: concatenation failure when middle row does not allow its variable" {
    const allocator = std.testing.allocator;
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const gpa_allocator = gpa.allocator();

    const a_node = try testMakeVariable(gpa_allocator, "A");
    defer gpa_allocator.destroy(a_node);
    const b_node = try testMakeVariable(gpa_allocator, "B");
    defer gpa_allocator.destroy(b_node);
    const c_node = try testMakeVariable(gpa_allocator, "C");
    defer gpa_allocator.destroy(c_node);

    const pattern = try testMakeConcat(gpa_allocator, &.{ a_node, b_node, c_node });
    defer {
        gpa_allocator.free(pattern.concat);
        gpa_allocator.destroy(pattern);
    }

    // Row 1 does NOT allow "B" (only allows "X")
    var ctx_impl = TestContext{
        .allowed_per_row = &.{
            &.{"A"},
            &.{"X"}, // does not allow "B"
            &.{"C"},
        },
    };

    const result = try findMatch(allocator, pattern, 3, 0, ctx_impl.toMatchContext());

    // Should fail to match (return null) because row 1 cannot match "B"
    try std.testing.expectEqual(@as(?Match, null), result);
}

test "pattern_match: alternation picks first successful branch" {
    const allocator = std.testing.allocator;
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const gpa_allocator = gpa.allocator();

    // Pattern: A (B | C)
    const a_node = try testMakeVariable(gpa_allocator, "A");
    defer gpa_allocator.destroy(a_node);
    const b_node = try testMakeVariable(gpa_allocator, "B");
    defer gpa_allocator.destroy(b_node);
    const c_node = try testMakeVariable(gpa_allocator, "C");
    defer gpa_allocator.destroy(c_node);

    const alt = try testMakeAlternation(gpa_allocator, &.{ b_node, c_node });
    defer {
        gpa_allocator.free(alt.alternation);
        gpa_allocator.destroy(alt);
    }

    const pattern = try testMakeConcat(gpa_allocator, &.{ a_node, alt });
    defer {
        gpa_allocator.free(pattern.concat);
        gpa_allocator.destroy(pattern);
    }

    // Row 0 allows "A", row 1 allows BOTH "B" and "C"
    var ctx_impl = TestContext{
        .allowed_per_row = &.{
            &.{"A"},
            &.{ "B", "C" }, // both allowed
        },
    };

    const result = try findMatch(allocator, pattern, 2, 0, ctx_impl.toMatchContext());

    // Should match and pick "B" (first branch listed in alternation)
    if (result) |match| {
        defer allocator.free(match.variable_per_row);
        try std.testing.expectEqualStrings("B", match.variable_per_row[1]);
    } else {
        try std.testing.expect(false); // should not be null
    }
}

test "pattern_match: alternation tries second branch when first fails" {
    const allocator = std.testing.allocator;
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const gpa_allocator = gpa.allocator();

    // Pattern: A (B | C)
    const a_node = try testMakeVariable(gpa_allocator, "A");
    defer gpa_allocator.destroy(a_node);
    const b_node = try testMakeVariable(gpa_allocator, "B");
    defer gpa_allocator.destroy(b_node);
    const c_node = try testMakeVariable(gpa_allocator, "C");
    defer gpa_allocator.destroy(c_node);

    const alt = try testMakeAlternation(gpa_allocator, &.{ b_node, c_node });
    defer {
        gpa_allocator.free(alt.alternation);
        gpa_allocator.destroy(alt);
    }

    const pattern = try testMakeConcat(gpa_allocator, &.{ a_node, alt });
    defer {
        gpa_allocator.free(pattern.concat);
        gpa_allocator.destroy(pattern);
    }

    // Row 0 allows "A", row 1 allows ONLY "C" (not "B")
    var ctx_impl = TestContext{
        .allowed_per_row = &.{
            &.{"A"},
            &.{"C"}, // only C, not B
        },
    };

    const result = try findMatch(allocator, pattern, 2, 0, ctx_impl.toMatchContext());

    // Should match with "C" (second branch, after first branch "B" failed)
    if (result) |match| {
        defer allocator.free(match.variable_per_row);
        try std.testing.expectEqualStrings("C", match.variable_per_row[1]);
    } else {
        try std.testing.expect(false); // should not be null
    }
}

test "pattern_match: greedy quantifier B+ backtracks to allow C to match" {
    const allocator = std.testing.allocator;
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const gpa_allocator = gpa.allocator();

    // Pattern: A B+ C  (3 rows total)
    // Setup: row0="A" only, row1="B" only, row2 allows "B" or "C"
    // Greedy B+ will first try to consume both row1 AND row2 as B,
    // leaving nothing for mandatory C → fails → backtracks B+ to consume only row1,
    // leaving row2 for C (which matches) → succeeds.

    const a_node = try testMakeVariable(gpa_allocator, "A");
    defer gpa_allocator.destroy(a_node);
    const b_node = try testMakeVariable(gpa_allocator, "B");
    defer gpa_allocator.destroy(b_node);
    const c_node = try testMakeVariable(gpa_allocator, "C");
    defer gpa_allocator.destroy(c_node);

    const b_plus = try testMakeQuantified(gpa_allocator, b_node, .one_or_more);
    defer gpa_allocator.destroy(b_plus);

    const pattern = try testMakeConcat(gpa_allocator, &.{ a_node, b_plus, c_node });
    defer {
        gpa_allocator.free(pattern.concat);
        gpa_allocator.destroy(pattern);
    }

    var ctx_impl = TestContext{
        .allowed_per_row = &.{
            &.{"A"},
            &.{"B"},
            &.{ "B", "C" }, // allows both
        },
    };

    const result = try findMatch(allocator, pattern, 3, 0, ctx_impl.toMatchContext());

    // Should match all 3 rows: A, B, C
    // (B+ backtracks from consuming both row1 and row2 to consuming only row1)
    if (result) |match| {
        defer allocator.free(match.variable_per_row);
        try std.testing.expectEqual(@as(usize, 0), match.start);
        try std.testing.expectEqual(@as(usize, 3), match.end_exclusive);
        try std.testing.expectEqual(@as(usize, 3), match.variable_per_row.len);
        try std.testing.expectEqualStrings("A", match.variable_per_row[0]);
        try std.testing.expectEqualStrings("B", match.variable_per_row[1]);
        try std.testing.expectEqualStrings("C", match.variable_per_row[2]);
    } else {
        try std.testing.expect(false); // should not be null
    }
}

test "pattern_match: quantifier ? matches one occurrence when available" {
    const allocator = std.testing.allocator;
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const gpa_allocator = gpa.allocator();

    // Pattern: A B? C (3 rows)
    // row0="A", row1="B", row2="C"
    // Should match all 3 with B present

    const a_node = try testMakeVariable(gpa_allocator, "A");
    defer gpa_allocator.destroy(a_node);
    const b_node = try testMakeVariable(gpa_allocator, "B");
    defer gpa_allocator.destroy(b_node);
    const c_node = try testMakeVariable(gpa_allocator, "C");
    defer gpa_allocator.destroy(c_node);

    const b_optional = try testMakeQuantified(gpa_allocator, b_node, .zero_or_one);
    defer gpa_allocator.destroy(b_optional);

    const pattern = try testMakeConcat(gpa_allocator, &.{ a_node, b_optional, c_node });
    defer {
        gpa_allocator.free(pattern.concat);
        gpa_allocator.destroy(pattern);
    }

    var ctx_impl = TestContext{
        .allowed_per_row = &.{
            &.{"A"},
            &.{"B"},
            &.{"C"},
        },
    };

    const result = try findMatch(allocator, pattern, 3, 0, ctx_impl.toMatchContext());

    if (result) |match| {
        defer allocator.free(match.variable_per_row);
        try std.testing.expectEqual(@as(usize, 3), match.variable_per_row.len);
        try std.testing.expectEqualStrings("A", match.variable_per_row[0]);
        try std.testing.expectEqualStrings("B", match.variable_per_row[1]);
        try std.testing.expectEqualStrings("C", match.variable_per_row[2]);
    } else {
        try std.testing.expect(false); // should not be null
    }
}

test "pattern_match: quantifier ? matches zero occurrences when variable unavailable" {
    const allocator = std.testing.allocator;
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const gpa_allocator = gpa.allocator();

    // Pattern: A B? C (2 rows)
    // row0="A", row1="C" (no row allows "B")
    // Should match both rows with B absent

    const a_node = try testMakeVariable(gpa_allocator, "A");
    defer gpa_allocator.destroy(a_node);
    const b_node = try testMakeVariable(gpa_allocator, "B");
    defer gpa_allocator.destroy(b_node);
    const c_node = try testMakeVariable(gpa_allocator, "C");
    defer gpa_allocator.destroy(c_node);

    const b_optional = try testMakeQuantified(gpa_allocator, b_node, .zero_or_one);
    defer gpa_allocator.destroy(b_optional);

    const pattern = try testMakeConcat(gpa_allocator, &.{ a_node, b_optional, c_node });
    defer {
        gpa_allocator.free(pattern.concat);
        gpa_allocator.destroy(pattern);
    }

    var ctx_impl = TestContext{
        .allowed_per_row = &.{
            &.{"A"},
            &.{"C"}, // no B available
        },
    };

    const result = try findMatch(allocator, pattern, 2, 0, ctx_impl.toMatchContext());

    // Should match 2 rows with B absent (variable_per_row = ["A", "C"])
    if (result) |match| {
        defer allocator.free(match.variable_per_row);
        try std.testing.expectEqual(@as(usize, 2), match.variable_per_row.len);
        try std.testing.expectEqualStrings("A", match.variable_per_row[0]);
        try std.testing.expectEqualStrings("C", match.variable_per_row[1]);
    } else {
        try std.testing.expect(false); // should not be null
    }
}

test "pattern_match: nested group with quantifier (B C)+" {
    const allocator = std.testing.allocator;
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const gpa_allocator = gpa.allocator();

    // Pattern: A (B C)+ D?
    // Rows: row0="A", row1="B", row2="C", row3="B", row4="C", row5="D"
    // Should match all 6 rows with (B C) repeated twice and D present

    const a_node = try testMakeVariable(gpa_allocator, "A");
    defer gpa_allocator.destroy(a_node);
    const b_node = try testMakeVariable(gpa_allocator, "B");
    defer gpa_allocator.destroy(b_node);
    const c_node = try testMakeVariable(gpa_allocator, "C");
    defer gpa_allocator.destroy(c_node);
    const d_node = try testMakeVariable(gpa_allocator, "D");
    defer gpa_allocator.destroy(d_node);

    const bc_group = try testMakeConcat(gpa_allocator, &.{ b_node, c_node });
    defer {
        gpa_allocator.free(bc_group.concat);
        gpa_allocator.destroy(bc_group);
    }

    const bc_plus = try testMakeQuantified(gpa_allocator, bc_group, .one_or_more);
    defer gpa_allocator.destroy(bc_plus);

    const d_optional = try testMakeQuantified(gpa_allocator, d_node, .zero_or_one);
    defer gpa_allocator.destroy(d_optional);

    const pattern = try testMakeConcat(gpa_allocator, &.{ a_node, bc_plus, d_optional });
    defer {
        gpa_allocator.free(pattern.concat);
        gpa_allocator.destroy(pattern);
    }

    var ctx_impl = TestContext{
        .allowed_per_row = &.{
            &.{"A"},
            &.{"B"},
            &.{"C"},
            &.{"B"},
            &.{"C"},
            &.{"D"},
        },
    };

    const result = try findMatch(allocator, pattern, 6, 0, ctx_impl.toMatchContext());

    if (result) |match| {
        defer allocator.free(match.variable_per_row);
        try std.testing.expectEqual(@as(usize, 6), match.variable_per_row.len);
        try std.testing.expectEqualStrings("A", match.variable_per_row[0]);
        try std.testing.expectEqualStrings("B", match.variable_per_row[1]);
        try std.testing.expectEqualStrings("C", match.variable_per_row[2]);
        try std.testing.expectEqualStrings("B", match.variable_per_row[3]);
        try std.testing.expectEqualStrings("C", match.variable_per_row[4]);
        try std.testing.expectEqualStrings("D", match.variable_per_row[5]);
    } else {
        try std.testing.expect(false); // should not be null
    }
}

test "pattern_match: no match when first variable cannot be satisfied" {
    const allocator = std.testing.allocator;
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const gpa_allocator = gpa.allocator();

    // Pattern: A B
    const a_node = try testMakeVariable(gpa_allocator, "A");
    defer gpa_allocator.destroy(a_node);
    const b_node = try testMakeVariable(gpa_allocator, "B");
    defer gpa_allocator.destroy(b_node);

    const pattern = try testMakeConcat(gpa_allocator, &.{ a_node, b_node });
    defer {
        gpa_allocator.free(pattern.concat);
        gpa_allocator.destroy(pattern);
    }

    // Row 0 does not allow "A" at all
    var ctx_impl = TestContext{
        .allowed_per_row = &.{
            &.{"X"}, // not "A"
            &.{"B"},
        },
    };

    const result = try findMatch(allocator, pattern, 2, 0, ctx_impl.toMatchContext());

    // Should fail to match
    try std.testing.expectEqual(@as(?Match, null), result);
}

test "pattern_match: start offset is respected (does not scan backwards)" {
    const allocator = std.testing.allocator;
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const gpa_allocator = gpa.allocator();

    // Pattern: A B C (same as test 1)
    const a_node = try testMakeVariable(gpa_allocator, "A");
    defer gpa_allocator.destroy(a_node);
    const b_node = try testMakeVariable(gpa_allocator, "B");
    defer gpa_allocator.destroy(b_node);
    const c_node = try testMakeVariable(gpa_allocator, "C");
    defer gpa_allocator.destroy(c_node);

    const pattern = try testMakeConcat(gpa_allocator, &.{ a_node, b_node, c_node });
    defer {
        gpa_allocator.free(pattern.concat);
        gpa_allocator.destroy(pattern);
    }

    // Row 0 allows "A", row 1 allows "B", row 2 allows "C"
    var ctx_impl = TestContext{
        .allowed_per_row = &.{
            &.{"A"},
            &.{"B"},
            &.{"C"},
        },
    };

    // Try to match starting at row 1 (not row 0)
    // Since row 1 only allows "B" (not "A"), pattern should fail
    const result = try findMatch(allocator, pattern, 3, 1, ctx_impl.toMatchContext());

    // Should return null because row 1 cannot match "A"
    try std.testing.expectEqual(@as(?Match, null), result);
}

test "pattern_match: memory is properly allocated and can be freed" {
    const allocator = std.testing.allocator;
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const gpa_allocator = gpa.allocator();

    // Build a simple pattern: A B
    const a_node = try testMakeVariable(gpa_allocator, "A");
    defer gpa_allocator.destroy(a_node);
    const b_node = try testMakeVariable(gpa_allocator, "B");
    defer gpa_allocator.destroy(b_node);

    const pattern = try testMakeConcat(gpa_allocator, &.{ a_node, b_node });
    defer {
        gpa_allocator.free(pattern.concat);
        gpa_allocator.destroy(pattern);
    }

    var ctx_impl = TestContext{
        .allowed_per_row = &.{
            &.{"A"},
            &.{"B"},
        },
    };

    const result = try findMatch(allocator, pattern, 2, 0, ctx_impl.toMatchContext());

    // Allocator should detect any leaks when match is freed
    if (result) |match| {
        defer allocator.free(match.variable_per_row);
        try std.testing.expectEqual(@as(usize, 2), match.variable_per_row.len);
    } else {
        try std.testing.expect(false);
    }
}
