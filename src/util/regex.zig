//! Thompson NFA-based regular expression engine.
//!
//! Supports:
//!   - Literals (a, b, ...)
//!   - Wildcard (.)
//!   - Character classes ([abc], [a-z], [^abc])
//!   - Quantifiers (*, +, ?)
//!   - Alternation (a|b)
//!   - Groups ((expr))
//!   - Anchors (^, $)
//!   - Escape sequences (\d, \w, \s, \D, \W, \S)

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Error = error{
    InvalidPattern,
    OutOfMemory,
};

pub const Flags = struct {
    ignore_case: bool = false,
    dot_all: bool = false,
};

pub const Match = struct {
    start: usize,
    end: usize,
    n_groups: u32,
    saves: [64]?usize, // saves[i*2] = group i start, saves[i*2+1] = group i end

    /// Get the text of capture group n (1-indexed). n=0 returns whole match.
    pub fn groupText(self: Match, n: usize, src: []const u8) ?[]const u8 {
        const idx = n * 2;
        if (idx >= 64) return null;

        const start = self.saves[idx] orelse return null;
        const end = self.saves[idx + 1] orelse return null;

        if (start > end or end > src.len) return null;
        return src[start..end];
    }
};

// ─ AST Nodes ─

const ClsItem = union(enum) {
    single: u8,
    range: struct { lo: u8, hi: u8 },
    word,    // \w: [a-zA-Z0-9_]
    digit,   // \d: [0-9]
    space,   // \s: [ \t\n\r\f\v]
    nword,   // \W
    ndigit,  // \D
    nspace,  // \S
};

const CharClass = struct {
    neg: bool,
    items: []const ClsItem,
};

const Node = union(enum) {
    lit: u8,
    any: void,
    cls: struct { neg: bool, items: []const ClsItem },
    seq: []const *const Node,
    alt: struct { a: *const Node, b: *const Node },
    star: struct { child: *const Node, greedy: bool },
    plus: struct { child: *const Node, greedy: bool },
    opt: struct { child: *const Node, greedy: bool },
    group: struct { id: u32, child: *const Node },
    anchor_start: void,
    anchor_end: void,
};

// ─ Bytecode instructions ─

const InstTag = enum {
    char,       // match specific char
    any,        // match any char
    cls,        // match character class
    split,      // fork execution (a: next, b: jmp_to)
    jmp,        // unconditional jump
    save,       // save current position to slot n
    match,      // success
    anchor_end, // match only at end of input
};

const Inst = struct {
    tag: InstTag,
    data: u32,
};

pub const Regex = struct {
    insts: []const Inst,
    char_classes: []const CharClass,
    allocator: Allocator,
    n_groups: u32,
    pattern: []const u8,

    pub fn compile(alloc: Allocator, pattern: []const u8) Error!Regex {
        var arena_state = std.heap.ArenaAllocator.init(alloc);
        defer arena_state.deinit();
        const arena = arena_state.allocator();

        var parser = Parser{
            .input = pattern,
            .pos = 0,
            .arena = arena,
        };

        const root = try parser.parseExpr();
        const group_count: u32 = parser.next_group_id - 1;

        var compiler: Compiler = undefined;
        compiler.arena = arena;
        compiler.alloc = alloc;
        compiler.group_count = group_count;
        compiler.insts = std.ArrayListUnmanaged(Inst){};
        compiler.char_classes = std.ArrayListUnmanaged(CharClass){};
        defer compiler.insts.deinit(alloc);
        defer compiler.char_classes.deinit(alloc);

        try compiler.compile(root);

        const insts = try alloc.dupe(Inst, compiler.insts.items);

        var char_classes_out = std.ArrayListUnmanaged(CharClass){};

        for (compiler.char_classes.items) |cc| {
            const items = try alloc.dupe(ClsItem, cc.items);
            try char_classes_out.append(alloc, CharClass{ .neg = cc.neg, .items = items });
        }

        const pattern_copy = try alloc.dupe(u8, pattern);

        return Regex{
            .insts = insts,
            .char_classes = try char_classes_out.toOwnedSlice(alloc),
            .allocator = alloc,
            .n_groups = group_count,
            .pattern = pattern_copy,
        };
    }

    pub fn deinit(self: *Regex, alloc: Allocator) void {
        for (self.char_classes) |cls| {
            alloc.free(cls.items);
        }
        alloc.free(self.char_classes);
        alloc.free(self.insts);
        alloc.free(self.pattern);
    }

    pub fn find(self: *const Regex, alloc: Allocator, text: []const u8, flags: Flags) Error!?Match {
        const max_start = if (self.startsWithAnchor()) 1 else text.len + 1;

        var start_pos: usize = 0;
        while (start_pos < max_start) : (start_pos += 1) {
            var saves: [64]?usize = undefined;
            for (&saves) |*s| s.* = null;

            saves[0] = start_pos;

            if (try self.matchAt(alloc, text, start_pos, &saves, flags)) |end_pos| {
                saves[1] = end_pos;
                return Match{
                    .start = start_pos,
                    .end = end_pos,
                    .n_groups = self.n_groups,
                    .saves = saves,
                };
            }
        }

        return null;
    }

    fn startsWithAnchor(self: *const Regex) bool {
        return self.insts.len > 0 and self.insts[0].tag == .save and self.insts[0].data == 0;
    }

    fn matchAt(self: *const Regex, alloc: Allocator, text: []const u8, start_pos: usize, saves: *[64]?usize, flags: Flags) Error!?usize {
        var stack = std.ArrayListUnmanaged(Thread){};
        defer stack.deinit(alloc);

        try stack.append(alloc, Thread{ .pc = 0, .pos = start_pos, .saves = saves.* });

        var seen = std.AutoHashMap(ThreadKey, void).init(alloc);
        defer seen.deinit();

        while (stack.items.len > 0) {
            const thread = stack.pop() orelse continue;

            const key = ThreadKey{ .pc = thread.pc, .pos = thread.pos };
            if (seen.contains(key)) continue;
            try seen.put(key, {});

            if (thread.pc >= self.insts.len) continue;

            const inst = self.insts[thread.pc];

            switch (inst.tag) {
                .char => {
                    if (thread.pos < text.len) {
                        const matches = if (flags.ignore_case)
                            std.ascii.toLower(text[thread.pos]) == std.ascii.toLower(@as(u8, @intCast(inst.data)))
                        else
                            text[thread.pos] == @as(u8, @intCast(inst.data));

                        if (matches) {
                            try stack.append(alloc, Thread{
                                .pc = thread.pc + 1,
                                .pos = thread.pos + 1,
                                .saves = thread.saves,
                            });
                        }
                    }
                },
                .any => {
                    if (thread.pos < text.len and (flags.dot_all or text[thread.pos] != '\n')) {
                        try stack.append(alloc, Thread{
                            .pc = thread.pc + 1,
                            .pos = thread.pos + 1,
                            .saves = thread.saves,
                        });
                    }
                },
                .cls => {
                    if (thread.pos < text.len) {
                        const cls_idx = inst.data;
                        if (cls_idx < self.char_classes.len) {
                            const matches = self.matchesCharClass(
                                text[thread.pos],
                                self.char_classes[cls_idx],
                                flags.ignore_case,
                            );
                            if (matches) {
                                try stack.append(alloc, Thread{
                                    .pc = thread.pc + 1,
                                    .pos = thread.pos + 1,
                                    .saves = thread.saves,
                                });
                            }
                        }
                    }
                },
                .split => {
                    // Push both branches (exit first, body second for greedy matching)
                    try stack.append(alloc, Thread{
                        .pc = @as(u32, @intCast(inst.data)),
                        .pos = thread.pos,
                        .saves = thread.saves,
                    });
                    try stack.append(alloc, Thread{
                        .pc = thread.pc + 1,
                        .pos = thread.pos,
                        .saves = thread.saves,
                    });
                },
                .jmp => {
                    try stack.append(alloc, Thread{
                        .pc = @as(u32, @intCast(inst.data)),
                        .pos = thread.pos,
                        .saves = thread.saves,
                    });
                },
                .save => {
                    var new_saves = thread.saves;
                    new_saves[inst.data] = thread.pos;
                    try stack.append(alloc, Thread{
                        .pc = thread.pc + 1,
                        .pos = thread.pos,
                        .saves = new_saves,
                    });
                },
                .match => {
                    saves.* = thread.saves;
                    return thread.pos;
                },
                .anchor_end => {
                    if (thread.pos == text.len) {
                        try stack.append(alloc, Thread{
                            .pc = thread.pc + 1,
                            .pos = thread.pos,
                            .saves = thread.saves,
                        });
                    }
                },
            }
        }

        return null;
    }

    fn matchesCharClass(self: *const Regex, ch: u8, cc: CharClass, ignore_case: bool) bool {
        _ = self;

        var positive_match = false;
        for (cc.items) |item| {
            const matches = switch (item) {
                .single => |c| if (ignore_case) std.ascii.toLower(c) == std.ascii.toLower(ch) else c == ch,
                .range => |r| ch >= r.lo and ch <= r.hi,
                .word => isWordChar(ch),
                .digit => ch >= '0' and ch <= '9',
                .space => ch == ' ' or ch == '\t' or ch == '\n' or ch == '\r' or ch == '\x0c' or ch == '\x0b',
                .nword => !isWordChar(ch),
                .ndigit => !(ch >= '0' and ch <= '9'),
                .nspace => !(ch == ' ' or ch == '\t' or ch == '\n' or ch == '\r' or ch == '\x0c' or ch == '\x0b'),
            };
            if (matches) {
                positive_match = true;
                break;
            }
        }

        return if (cc.neg) !positive_match else positive_match;
    }

    pub fn replace(
        self: *const Regex,
        alloc: Allocator,
        text: []const u8,
        replacement: []const u8,
        global: bool,
        flags: Flags,
    ) Error![]u8 {
        var result = std.ArrayListUnmanaged(u8){};
        errdefer result.deinit(alloc);

        var search_pos: usize = 0;

        while (search_pos <= text.len) {
            var saves: [64]?usize = undefined;
            for (&saves) |*s| s.* = null;
            saves[0] = search_pos;

            if (try self.matchAt(alloc, text, search_pos, &saves, flags)) |end_pos| {
                const match_start = saves[0] orelse search_pos;

                try result.appendSlice(alloc, text[search_pos..match_start]);
                try result.appendSlice(alloc, replacement);

                saves[1] = end_pos;
                search_pos = end_pos;

                if (end_pos == match_start) {
                    // Zero-length match: advance by 1 to avoid infinite loop
                    if (search_pos < text.len) {
                        try result.append(alloc, text[search_pos]);
                        search_pos += 1;
                    } else {
                        break;
                    }
                }

                if (!global) {
                    try result.appendSlice(alloc, text[search_pos..]);
                    break;
                }
            } else {
                if (search_pos < text.len) {
                    try result.append(alloc, text[search_pos]);
                    search_pos += 1;
                } else break;
            }
        }

        return result.toOwnedSlice(alloc);
    }
};

// ─ Thread for NFA simulation ─

const Thread = struct {
    pc: u32,
    pos: usize,
    saves: [64]?usize,
};

const ThreadKey = struct {
    pc: u32,
    pos: usize,
};

// ─ Parser ─

const Parser = struct {
    input: []const u8,
    pos: usize,
    arena: Allocator,
    next_group_id: u32 = 1,

    fn peek(self: *Parser) ?u8 {
        if (self.pos >= self.input.len) return null;
        return self.input[self.pos];
    }

    fn consume(self: *Parser) ?u8 {
        if (self.pos >= self.input.len) return null;
        const ch = self.input[self.pos];
        self.pos += 1;
        return ch;
    }

    fn parseExpr(self: *Parser) Error!*const Node {
        var terms = std.ArrayListUnmanaged(*const Node){};

        try terms.append(self.arena, try self.parseTerm());

        while (self.peek() == '|') {
            _ = self.consume();
            try terms.append(self.arena, try self.parseTerm());
        }

        if (terms.items.len == 1) {
            return terms.items[0];
        }

        var result = terms.items[0];
        for (terms.items[1..]) |term| {
            const alt = try self.arena.create(Node);
            alt.* = Node{ .alt = .{ .a = result, .b = term } };
            result = alt;
        }
        return result;
    }

    fn parseTerm(self: *Parser) Error!*const Node {
        var factors = std.ArrayListUnmanaged(*const Node){};

        while (self.peek() != null and self.peek() != '|' and self.peek() != ')') {
            try factors.append(self.arena, try self.parseFactor());
        }

        if (factors.items.len == 0) {
            const empty = try self.arena.create(Node);
            empty.* = Node{ .seq = &.{} };
            return empty;
        }

        if (factors.items.len == 1) {
            return factors.items[0];
        }

        const seq = try self.arena.dupe(*const Node, factors.items);
        const node = try self.arena.create(Node);
        node.* = Node{ .seq = seq };
        return node;
    }

    fn parseFactor(self: *Parser) Error!*const Node {
        var atom = try self.parseAtom();

        while (self.peek()) |ch| {
            switch (ch) {
                '*' => {
                    _ = self.consume();
                    const node = try self.arena.create(Node);
                    node.* = Node{ .star = .{ .child = atom, .greedy = true } };
                    atom = node;
                },
                '+' => {
                    _ = self.consume();
                    const node = try self.arena.create(Node);
                    node.* = Node{ .plus = .{ .child = atom, .greedy = true } };
                    atom = node;
                },
                '?' => {
                    _ = self.consume();
                    const node = try self.arena.create(Node);
                    node.* = Node{ .opt = .{ .child = atom, .greedy = true } };
                    atom = node;
                },
                else => break,
            }
        }

        return atom;
    }

    fn parseAtom(self: *Parser) Error!*const Node {
        const ch_opt = self.peek();
        if (ch_opt == null) return Error.InvalidPattern;
        const ch = ch_opt.?;

        switch (ch) {
            '^' => {
                _ = self.consume();
                const node = try self.arena.create(Node);
                node.* = Node.anchor_start;
                return node;
            },
            '$' => {
                _ = self.consume();
                const node = try self.arena.create(Node);
                node.* = Node.anchor_end;
                return node;
            },
            '.' => {
                _ = self.consume();
                const node = try self.arena.create(Node);
                node.* = Node.any;
                return node;
            },
            '[' => {
                _ = self.consume();
                return try self.parseCharClass();
            },
            '(' => {
                _ = self.consume();
                const gid = self.next_group_id;
                self.next_group_id += 1;
                const expr = try self.parseExpr();
                if (self.consume() != ')') return Error.InvalidPattern;
                const node = try self.arena.create(Node);
                node.* = Node{ .group = .{ .id = gid, .child = expr } };
                return node;
            },
            '\\' => {
                _ = self.consume();
                const escaped = self.consume() orelse return Error.InvalidPattern;
                switch (escaped) {
                    'd' => {
                        const cls = try self.arena.dupe(ClsItem, &.{ClsItem.digit});
                        const node = try self.arena.create(Node);
                        node.* = Node{ .cls = .{ .neg = false, .items = cls } };
                        return node;
                    },
                    'D' => {
                        const cls = try self.arena.dupe(ClsItem, &.{ClsItem.ndigit});
                        const node = try self.arena.create(Node);
                        node.* = Node{ .cls = .{ .neg = false, .items = cls } };
                        return node;
                    },
                    'w' => {
                        const cls = try self.arena.dupe(ClsItem, &.{ClsItem.word});
                        const node = try self.arena.create(Node);
                        node.* = Node{ .cls = .{ .neg = false, .items = cls } };
                        return node;
                    },
                    'W' => {
                        const cls = try self.arena.dupe(ClsItem, &.{ClsItem.nword});
                        const node = try self.arena.create(Node);
                        node.* = Node{ .cls = .{ .neg = false, .items = cls } };
                        return node;
                    },
                    's' => {
                        const cls = try self.arena.dupe(ClsItem, &.{ClsItem.space});
                        const node = try self.arena.create(Node);
                        node.* = Node{ .cls = .{ .neg = false, .items = cls } };
                        return node;
                    },
                    'S' => {
                        const cls = try self.arena.dupe(ClsItem, &.{ClsItem.nspace});
                        const node = try self.arena.create(Node);
                        node.* = Node{ .cls = .{ .neg = false, .items = cls } };
                        return node;
                    },
                    else => {
                        const node = try self.arena.create(Node);
                        node.* = Node{ .lit = escaped };
                        return node;
                    },
                }
            },
            ')', '*', '+', '?' => return Error.InvalidPattern,
            else => {
                const lit_char = self.consume().?;
                const node = try self.arena.create(Node);
                node.* = Node{ .lit = lit_char };
                return node;
            },
        }
    }

    fn parseCharClass(self: *Parser) Error!*const Node {
        var items = std.ArrayListUnmanaged(ClsItem){};

        const negated = if (self.peek() == '^') neg: {
            _ = self.consume();
            break :neg true;
        } else false;

        while (self.peek() != null and self.peek() != ']') {
            const ch = self.consume().?;

            if (ch == '-' and self.peek() != null and self.peek() != ']' and items.items.len > 0) {
                const next = self.consume().?;
                if (items.pop()) |last| {
                    if (last == .single) {
                        const lo = last.single;
                        try items.append(self.arena, ClsItem{ .range = .{ .lo = lo, .hi = next } });
                    } else {
                        try items.append(self.arena, last);
                        try items.append(self.arena, ClsItem{ .single = ch });
                    }
                } else {
                    try items.append(self.arena, ClsItem{ .single = ch });
                }
            } else {
                try items.append(self.arena, ClsItem{ .single = ch });
            }
        }

        if (self.consume() != ']') return Error.InvalidPattern;

        const cls = try self.arena.dupe(ClsItem, items.items);
        const node = try self.arena.create(Node);
        node.* = Node{ .cls = .{ .neg = negated, .items = cls } };
        return node;
    }
};

// ─ Compiler ─

const Compiler = struct {
    insts: std.ArrayListUnmanaged(Inst),
    char_classes: std.ArrayListUnmanaged(CharClass),
    arena: Allocator,
    alloc: Allocator,
    group_count: u32,

    fn compile(self: *Compiler, node: *const Node) Error!void {
        _ = try self.compileNode(node);
        try self.insts.append(self.alloc, Inst{ .tag = .match, .data = 0 });
    }

    fn compileNode(self: *Compiler, node: *const Node) Error!u32 {
        return switch (node.*) {
            .lit => |ch| try self.compileLit(ch),
            .any => try self.compileAny(),
            .cls => try self.compileClassData(node),
            .seq => |s| try self.compileSeq(s),
            .alt => try self.compileAlt(node),
            .star => try self.compileStar(node),
            .plus => try self.compilePlus(node),
            .opt => try self.compileOpt(node),
            .group => try self.compileGroup(node),
            .anchor_start => try self.compileAnchorStart(),
            .anchor_end => try self.compileAnchorEnd(),
        };
    }

    fn compileLit(self: *Compiler, ch: u8) Error!u32 {
        const pc = @as(u32, @intCast(self.insts.items.len));
        try self.insts.append(self.alloc, Inst{ .tag = .char, .data = ch });
        return pc;
    }

    fn compileAny(self: *Compiler) Error!u32 {
        const pc = @as(u32, @intCast(self.insts.items.len));
        try self.insts.append(self.alloc, Inst{ .tag = .any, .data = 0 });
        return pc;
    }

    fn compileClassData(self: *Compiler, cls_node: *const Node) Error!u32 {
        const cls_data = cls_node.cls;
        const cls_idx = @as(u32, @intCast(self.char_classes.items.len));
        try self.char_classes.append(self.alloc, CharClass{ .neg = cls_data.neg, .items = cls_data.items });
        const pc = @as(u32, @intCast(self.insts.items.len));
        try self.insts.append(self.alloc, Inst{ .tag = .cls, .data = cls_idx });
        return pc;
    }

    fn compileSeq(self: *Compiler, nodes: []const *const Node) Error!u32 {
        if (nodes.len == 0) {
            const pc = @as(u32, @intCast(self.insts.items.len));
            return pc;
        }

        var start_pc: u32 = undefined;
        for (nodes, 0..) |n, i| {
            const pc = try self.compileNode(n);
            if (i == 0) start_pc = pc;
        }
        return start_pc;
    }

    fn compileAlt(self: *Compiler, alt_node: *const Node) Error!u32 {
        const alt_data = alt_node.alt;
        const split_pc = @as(u32, @intCast(self.insts.items.len));
        try self.insts.append(self.alloc, Inst{ .tag = .split, .data = 0 }); // Patch later

        _ = try self.compileNode(alt_data.a);
        const jmp_pc = @as(u32, @intCast(self.insts.items.len));
        try self.insts.append(self.alloc, Inst{ .tag = .jmp, .data = 0 }); // Patch later

        self.insts.items[split_pc].data = @as(u32, @intCast(self.insts.items.len));

        _ = try self.compileNode(alt_data.b);
        const after_pc = @as(u32, @intCast(self.insts.items.len));

        self.insts.items[jmp_pc].data = after_pc;

        return split_pc;
    }

    fn compileStar(self: *Compiler, star_node: *const Node) Error!u32 {
        const star_data = star_node.star;
        const split_pc = @as(u32, @intCast(self.insts.items.len));
        try self.insts.append(self.alloc, Inst{ .tag = .split, .data = 0 }); // Patch later

        _ = try self.compileNode(star_data.child);
        try self.insts.append(self.alloc, Inst{ .tag = .jmp, .data = split_pc });

        const after_pc = @as(u32, @intCast(self.insts.items.len));
        self.insts.items[split_pc].data = after_pc;

        return split_pc;
    }

    fn compilePlus(self: *Compiler, plus_node: *const Node) Error!u32 {
        const plus_data = plus_node.plus;
        const child_pc = try self.compileNode(plus_data.child);

        const split_pc = @as(u32, @intCast(self.insts.items.len));
        try self.insts.append(self.alloc, Inst{ .tag = .split, .data = 0 }); // Patch later

        try self.insts.append(self.alloc, Inst{ .tag = .jmp, .data = child_pc });

        const after_pc = @as(u32, @intCast(self.insts.items.len));
        self.insts.items[split_pc].data = after_pc;

        return child_pc;
    }

    fn compileOpt(self: *Compiler, opt_node: *const Node) Error!u32 {
        const opt_data = opt_node.opt;
        const split_pc = @as(u32, @intCast(self.insts.items.len));
        try self.insts.append(self.alloc, Inst{ .tag = .split, .data = 0 }); // Patch later

        _ = try self.compileNode(opt_data.child);

        const after_pc = @as(u32, @intCast(self.insts.items.len));
        self.insts.items[split_pc].data = after_pc;

        return split_pc;
    }

    fn compileGroup(self: *Compiler, grp_node: *const Node) Error!u32 {
        const grp_data = grp_node.group;
        const start_slot = grp_data.id * 2;
        const start_save_pc = @as(u32, @intCast(self.insts.items.len));
        try self.insts.append(self.alloc, Inst{ .tag = .save, .data = start_slot });

        _ = try self.compileNode(grp_data.child);

        const end_slot = grp_data.id * 2 + 1;
        try self.insts.append(self.alloc, Inst{ .tag = .save, .data = end_slot });

        return start_save_pc;
    }

    fn compileAnchorStart(self: *Compiler) Error!u32 {
        const pc = @as(u32, @intCast(self.insts.items.len));
        try self.insts.append(self.alloc, Inst{ .tag = .save, .data = 0 });
        return pc;
    }

    fn compileAnchorEnd(self: *Compiler) Error!u32 {
        const pc = @as(u32, @intCast(self.insts.items.len));
        try self.insts.append(self.alloc, Inst{ .tag = .anchor_end, .data = 0 });
        return pc;
    }
};

fn isWordChar(ch: u8) bool {
    return (ch >= 'a' and ch <= 'z') or
        (ch >= 'A' and ch <= 'Z') or
        (ch >= '0' and ch <= '9') or
        ch == '_';
}

// ─ Tests ─

test "regex: literal match" {
    const alloc = std.testing.allocator;
    var rx = try Regex.compile(alloc, "hello");
    defer rx.deinit(alloc);

    const m = try rx.find(alloc, "hello world", Flags{});
    try std.testing.expect(m != null);
    try std.testing.expectEqual(@as(usize, 0), m.?.start);
    try std.testing.expectEqual(@as(usize, 5), m.?.end);
}

test "regex: dot wildcard" {
    const alloc = std.testing.allocator;
    var rx = try Regex.compile(alloc, "c.t");
    defer rx.deinit(alloc);

    const m = try rx.find(alloc, "cat", Flags{});
    try std.testing.expect(m != null);
}

test "regex: character class" {
    const alloc = std.testing.allocator;
    var rx = try Regex.compile(alloc, "[a-z]+");
    defer rx.deinit(alloc);

    const m = try rx.find(alloc, "hello", Flags{});
    try std.testing.expect(m != null);
}

test "regex: capture group" {
    const alloc = std.testing.allocator;
    var rx = try Regex.compile(alloc, "(hello) (world)");
    defer rx.deinit(alloc);

    const m = try rx.find(alloc, "hello world", Flags{});
    try std.testing.expect(m != null);
    try std.testing.expectEqual(@as(u32, 2), rx.n_groups);
}

test "regex: star quantifier" {
    const alloc = std.testing.allocator;
    var rx = try Regex.compile(alloc, "a*b");
    defer rx.deinit(alloc);

    const m1 = try rx.find(alloc, "aaab", Flags{});
    try std.testing.expect(m1 != null);

    const m2 = try rx.find(alloc, "b", Flags{});
    try std.testing.expect(m2 != null);
}

test "regex: case insensitive" {
    const alloc = std.testing.allocator;
    var rx = try Regex.compile(alloc, "hello");
    defer rx.deinit(alloc);

    const m = try rx.find(alloc, "HELLO", Flags{ .ignore_case = true });
    try std.testing.expect(m != null);
}

test "regex: replace" {
    const alloc = std.testing.allocator;
    var rx = try Regex.compile(alloc, "world");
    defer rx.deinit(alloc);

    const result = try rx.replace(alloc, "hello world", "there", false, Flags{});
    defer alloc.free(result);

    try std.testing.expectEqualStrings("hello there", result);
}

test "regex: replace all" {
    const alloc = std.testing.allocator;
    var rx = try Regex.compile(alloc, "cat");
    defer rx.deinit(alloc);

    const result = try rx.replace(alloc, "cat bat cat", "dog", true, Flags{});
    defer alloc.free(result);

    try std.testing.expectEqualStrings("dog bat dog", result);
}

// ─ Alternation tests ─

test "regex: alternation simple a|b matches a" {
    const alloc = std.testing.allocator;
    var rx = try Regex.compile(alloc, "a|b");
    defer rx.deinit(alloc);

    const m = try rx.find(alloc, "a", Flags{});
    try std.testing.expect(m != null);
    try std.testing.expectEqual(@as(usize, 0), m.?.start);
    try std.testing.expectEqual(@as(usize, 1), m.?.end);
}

test "regex: alternation simple a|b matches b" {
    const alloc = std.testing.allocator;
    var rx = try Regex.compile(alloc, "a|b");
    defer rx.deinit(alloc);

    const m = try rx.find(alloc, "b", Flags{});
    try std.testing.expect(m != null);
}

test "regex: alternation longer patterns cat|dog" {
    const alloc = std.testing.allocator;
    var rx = try Regex.compile(alloc, "cat|dog");
    defer rx.deinit(alloc);

    const m1 = try rx.find(alloc, "my cat is here", Flags{});
    try std.testing.expect(m1 != null);
    try std.testing.expectEqual(@as(usize, 3), m1.?.start);
    try std.testing.expectEqual(@as(usize, 6), m1.?.end);

    const m2 = try rx.find(alloc, "my dog is here", Flags{});
    try std.testing.expect(m2 != null);
    try std.testing.expectEqual(@as(usize, 3), m2.?.start);
    try std.testing.expectEqual(@as(usize, 6), m2.?.end);
}

test "regex: alternation no match" {
    const alloc = std.testing.allocator;
    var rx = try Regex.compile(alloc, "cat|dog");
    defer rx.deinit(alloc);

    const m = try rx.find(alloc, "bird", Flags{});
    try std.testing.expect(m == null);
}

// ─ Anchor tests ─

test "regex: anchor start ^hello matches at beginning" {
    const alloc = std.testing.allocator;
    var rx = try Regex.compile(alloc, "^hello");
    defer rx.deinit(alloc);

    const m = try rx.find(alloc, "hello world", Flags{});
    try std.testing.expect(m != null);
    try std.testing.expectEqual(@as(usize, 0), m.?.start);
}

test "regex: anchor start ^hello does not match in middle" {
    const alloc = std.testing.allocator;
    var rx = try Regex.compile(alloc, "^hello");
    defer rx.deinit(alloc);

    const m = try rx.find(alloc, "say hello", Flags{});
    try std.testing.expect(m == null);
}

test "regex: anchor end world$ matches at end" {
    const alloc = std.testing.allocator;
    var rx = try Regex.compile(alloc, "world$");
    defer rx.deinit(alloc);

    const m = try rx.find(alloc, "hello world", Flags{});
    try std.testing.expect(m != null);
    try std.testing.expectEqual(@as(usize, 6), m.?.start);
    try std.testing.expectEqual(@as(usize, 11), m.?.end);
}

// ─ Plus quantifier tests ─

test "regex: plus quantifier a+ matches aaa" {
    const alloc = std.testing.allocator;
    var rx = try Regex.compile(alloc, "a+");
    defer rx.deinit(alloc);

    const m = try rx.find(alloc, "aaa", Flags{});
    try std.testing.expect(m != null);
    try std.testing.expectEqual(@as(usize, 0), m.?.start);
    try std.testing.expectEqual(@as(usize, 3), m.?.end);
}

test "regex: plus quantifier a+ no match on empty" {
    const alloc = std.testing.allocator;
    var rx = try Regex.compile(alloc, "a+");
    defer rx.deinit(alloc);

    const m = try rx.find(alloc, "", Flags{});
    try std.testing.expect(m == null);
}

test "regex: plus quantifier a+ no match starting with b" {
    const alloc = std.testing.allocator;
    var rx = try Regex.compile(alloc, "a+");
    defer rx.deinit(alloc);

    const m = try rx.find(alloc, "bbb", Flags{});
    try std.testing.expect(m == null);
}

test "regex: plus quantifier in context" {
    const alloc = std.testing.allocator;
    var rx = try Regex.compile(alloc, "a+b");
    defer rx.deinit(alloc);

    const m = try rx.find(alloc, "aaab", Flags{});
    try std.testing.expect(m != null);
    try std.testing.expectEqual(@as(usize, 0), m.?.start);
    try std.testing.expectEqual(@as(usize, 4), m.?.end);
}

// ─ Optional quantifier tests ─

test "regex: optional quantifier colou?r matches color" {
    const alloc = std.testing.allocator;
    var rx = try Regex.compile(alloc, "colou?r");
    defer rx.deinit(alloc);

    const m = try rx.find(alloc, "color", Flags{});
    try std.testing.expect(m != null);
}

test "regex: optional quantifier colou?r matches colour" {
    const alloc = std.testing.allocator;
    var rx = try Regex.compile(alloc, "colou?r");
    defer rx.deinit(alloc);

    const m = try rx.find(alloc, "colour", Flags{});
    try std.testing.expect(m != null);
}

test "regex: optional quantifier in pattern" {
    const alloc = std.testing.allocator;
    var rx = try Regex.compile(alloc, "a?b");
    defer rx.deinit(alloc);

    const m1 = try rx.find(alloc, "ab", Flags{});
    try std.testing.expect(m1 != null);

    const m2 = try rx.find(alloc, "b", Flags{});
    try std.testing.expect(m2 != null);
}

// ─ Escape sequence tests: \d ─

test "regex: escape \\d+ matches 123" {
    const alloc = std.testing.allocator;
    var rx = try Regex.compile(alloc, "\\d+");
    defer rx.deinit(alloc);

    const m = try rx.find(alloc, "123", Flags{});
    try std.testing.expect(m != null);
    try std.testing.expectEqual(@as(usize, 0), m.?.start);
    try std.testing.expectEqual(@as(usize, 3), m.?.end);
}

test "regex: escape \\d+ no match on abc" {
    const alloc = std.testing.allocator;
    var rx = try Regex.compile(alloc, "\\d+");
    defer rx.deinit(alloc);

    const m = try rx.find(alloc, "abc", Flags{});
    try std.testing.expect(m == null);
}

test "regex: escape \\d in context" {
    const alloc = std.testing.allocator;
    var rx = try Regex.compile(alloc, "call \\d+");
    defer rx.deinit(alloc);

    const m = try rx.find(alloc, "call 555", Flags{});
    try std.testing.expect(m != null);
    try std.testing.expectEqual(@as(usize, 0), m.?.start);
    try std.testing.expectEqual(@as(usize, 8), m.?.end);
}

// ─ Escape sequence tests: \w ─

test "regex: escape \\w+ matches hello_world" {
    const alloc = std.testing.allocator;
    var rx = try Regex.compile(alloc, "\\w+");
    defer rx.deinit(alloc);

    const m = try rx.find(alloc, "hello_world", Flags{});
    try std.testing.expect(m != null);
    try std.testing.expectEqual(@as(usize, 0), m.?.start);
    try std.testing.expectEqual(@as(usize, 11), m.?.end);
}

test "regex: escape \\w+ no match on !!!" {
    const alloc = std.testing.allocator;
    var rx = try Regex.compile(alloc, "\\w+");
    defer rx.deinit(alloc);

    const m = try rx.find(alloc, "!!!", Flags{});
    try std.testing.expect(m == null);
}

// ─ Escape sequence tests: \s ─

test "regex: escape \\s+ matches whitespace" {
    const alloc = std.testing.allocator;
    var rx = try Regex.compile(alloc, "\\s+");
    defer rx.deinit(alloc);

    const m = try rx.find(alloc, "hello   world", Flags{});
    try std.testing.expect(m != null);
    try std.testing.expectEqual(@as(usize, 5), m.?.start);
    try std.testing.expectEqual(@as(usize, 8), m.?.end);
}

test "regex: escape \\s+ matches tab" {
    const alloc = std.testing.allocator;
    var rx = try Regex.compile(alloc, "\\s+");
    defer rx.deinit(alloc);

    const m = try rx.find(alloc, "hello\tworld", Flags{});
    try std.testing.expect(m != null);
}

// ─ Escape sequence tests: \D ─

test "regex: escape \\D+ matches abc" {
    const alloc = std.testing.allocator;
    var rx = try Regex.compile(alloc, "\\D+");
    defer rx.deinit(alloc);

    const m = try rx.find(alloc, "abc", Flags{});
    try std.testing.expect(m != null);
    try std.testing.expectEqual(@as(usize, 0), m.?.start);
    try std.testing.expectEqual(@as(usize, 3), m.?.end);
}

test "regex: escape \\D+ no match on 123" {
    const alloc = std.testing.allocator;
    var rx = try Regex.compile(alloc, "\\D+");
    defer rx.deinit(alloc);

    const m = try rx.find(alloc, "123", Flags{});
    try std.testing.expect(m == null);
}

// ─ groupText() tests ─

test "regex: groupText group 0 is whole match" {
    const alloc = std.testing.allocator;
    var rx = try Regex.compile(alloc, "(\\d+)-(\\d+)");
    defer rx.deinit(alloc);

    const text = "call 555-1234";
    const m = try rx.find(alloc, text, Flags{});
    try std.testing.expect(m != null);

    const whole = m.?.groupText(0, text);
    try std.testing.expect(whole != null);
    try std.testing.expectEqualStrings("555-1234", whole.?);
}

test "regex: groupText group 1 first capture" {
    const alloc = std.testing.allocator;
    var rx = try Regex.compile(alloc, "(\\d+)-(\\d+)");
    defer rx.deinit(alloc);

    const text = "call 555-1234";
    const m = try rx.find(alloc, text, Flags{});
    try std.testing.expect(m != null);

    const group1 = m.?.groupText(1, text);
    try std.testing.expect(group1 != null);
    try std.testing.expectEqualStrings("555", group1.?);
}

test "regex: groupText group 2 second capture" {
    const alloc = std.testing.allocator;
    var rx = try Regex.compile(alloc, "(\\d+)-(\\d+)");
    defer rx.deinit(alloc);

    const text = "call 555-1234";
    const m = try rx.find(alloc, text, Flags{});
    try std.testing.expect(m != null);

    const group2 = m.?.groupText(2, text);
    try std.testing.expect(group2 != null);
    try std.testing.expectEqualStrings("1234", group2.?);
}

test "regex: groupText nonexistent group returns null" {
    const alloc = std.testing.allocator;
    var rx = try Regex.compile(alloc, "(\\d+)");
    defer rx.deinit(alloc);

    const text = "123";
    const m = try rx.find(alloc, text, Flags{});
    try std.testing.expect(m != null);

    const group2 = m.?.groupText(2, text);
    try std.testing.expect(group2 == null);
}

// ─ Non-match cases ─

test "regex: non-match xyz in hello world" {
    const alloc = std.testing.allocator;
    var rx = try Regex.compile(alloc, "xyz");
    defer rx.deinit(alloc);

    const m = try rx.find(alloc, "hello world", Flags{});
    try std.testing.expect(m == null);
}

test "regex: non-match pattern not found" {
    const alloc = std.testing.allocator;
    var rx = try Regex.compile(alloc, "elephant");
    defer rx.deinit(alloc);

    const m = try rx.find(alloc, "cat dog bird", Flags{});
    try std.testing.expect(m == null);
}

// ─ Invalid pattern tests ─

test "regex: invalid pattern unclosed bracket" {
    const alloc = std.testing.allocator;

    const result = Regex.compile(alloc, "[unclosed");
    try std.testing.expect(result == error.InvalidPattern);
}

test "regex: invalid pattern unclosed group" {
    const alloc = std.testing.allocator;

    const result = Regex.compile(alloc, "(unclosed");
    try std.testing.expect(result == error.InvalidPattern);
}

test "regex: invalid pattern bad quantifier" {
    const alloc = std.testing.allocator;

    const result = Regex.compile(alloc, "*invalid");
    try std.testing.expect(result == error.InvalidPattern);
}

// ─ Empty pattern tests ─

test "regex: empty pattern matches at position 0" {
    const alloc = std.testing.allocator;
    var rx = try Regex.compile(alloc, "");
    defer rx.deinit(alloc);

    const m = try rx.find(alloc, "hello", Flags{});
    try std.testing.expect(m != null);
    try std.testing.expectEqual(@as(usize, 0), m.?.start);
    try std.testing.expectEqual(@as(usize, 0), m.?.end);
}

test "regex: empty pattern on empty string" {
    const alloc = std.testing.allocator;
    var rx = try Regex.compile(alloc, "");
    defer rx.deinit(alloc);

    const m = try rx.find(alloc, "", Flags{});
    try std.testing.expect(m != null);
}

// ─ anchor_end ($) fix tests ─

test "regex: anchor end world$ does not match when not at end" {
    const alloc = std.testing.allocator;
    var rx = try Regex.compile(alloc, "world$");
    defer rx.deinit(alloc);

    const m = try rx.find(alloc, "world is big", Flags{});
    try std.testing.expect(m == null);
}

test "regex: anchor end ^abc$ exact match only" {
    const alloc = std.testing.allocator;
    var rx = try Regex.compile(alloc, "^abc$");
    defer rx.deinit(alloc);

    const yes = try rx.find(alloc, "abc", Flags{});
    try std.testing.expect(yes != null);

    const no = try rx.find(alloc, "abcd", Flags{});
    try std.testing.expect(no == null);

    const no2 = try rx.find(alloc, "xabc", Flags{});
    try std.testing.expect(no2 == null);
}

// ─ negated char class fix tests ─

test "regex: negated class [^aeiou]+ matches consonants" {
    const alloc = std.testing.allocator;
    var rx = try Regex.compile(alloc, "[^aeiou]+");
    defer rx.deinit(alloc);

    const m = try rx.find(alloc, "bcdfg", Flags{});
    try std.testing.expect(m != null);
    try std.testing.expectEqual(@as(usize, 0), m.?.start);
    try std.testing.expectEqual(@as(usize, 5), m.?.end);
}

test "regex: negated class [^0-9]+ matches non-digits" {
    const alloc = std.testing.allocator;
    var rx = try Regex.compile(alloc, "[^0-9]+");
    defer rx.deinit(alloc);

    const m = try rx.find(alloc, "abc", Flags{});
    try std.testing.expect(m != null);

    const no = try rx.find(alloc, "123", Flags{});
    try std.testing.expect(no == null);
}

// ─ dot_all flag fix tests ─

test "regex: dot does not match newline without dot_all" {
    const alloc = std.testing.allocator;
    var rx = try Regex.compile(alloc, "a.b");
    defer rx.deinit(alloc);

    const m = try rx.find(alloc, "a\nb", Flags{});
    try std.testing.expect(m == null);
}

test "regex: dot matches newline with dot_all" {
    const alloc = std.testing.allocator;
    var rx = try Regex.compile(alloc, "a.b");
    defer rx.deinit(alloc);

    const m = try rx.find(alloc, "a\nb", Flags{ .dot_all = true });
    try std.testing.expect(m != null);
}
