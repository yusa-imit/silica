const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Sailor dependency
    const sailor_dep = b.dependency("sailor", .{
        .target = target,
        .optimize = optimize,
    });

    // zuda dependency
    const zuda_dep = b.dependency("zuda", .{
        .target = target,
        .optimize = optimize,
    });

    // Library module — Silica as an embeddable library
    const mod = b.addModule("silica", .{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    mod.addImport("zuda", zuda_dep.module("zuda"));

    // Static library artifact for consumers
    const lib = b.addLibrary(.{
        .name = "silica",
        .root_module = mod,
        .linkage = .static,
    });
    b.installArtifact(lib);

    // CLI executable — `silica` command-line tool
    const cli_mod = b.addModule("silica-cli", .{
        .root_source_file = b.path("src/cli.zig"),
        .target = target,
        .optimize = optimize,
    });
    cli_mod.addImport("sailor", sailor_dep.module("sailor"));
    cli_mod.addImport("silica", mod);

    const cli_exe = b.addExecutable(.{
        .name = "silica",
        .root_module = cli_mod,
    });
    b.installArtifact(cli_exe);

    // Run CLI
    const run_cmd = b.addRunArtifact(cli_exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run the Silica CLI");
    run_step.dependOn(&run_cmd.step);

    // Unit tests for library
    const lib_unit_tests = b.addTest(.{
        .root_module = mod,
    });
    const run_lib_unit_tests = b.addRunArtifact(lib_unit_tests);

    // Unit tests for CLI
    const cli_test_mod = b.addModule("silica-cli-test", .{
        .root_source_file = b.path("src/cli.zig"),
        .target = target,
        .optimize = optimize,
    });
    cli_test_mod.addImport("sailor", sailor_dep.module("sailor"));
    cli_test_mod.addImport("silica", mod);
    const cli_unit_tests = b.addTest(.{
        .root_module = cli_test_mod,
    });
    const run_cli_unit_tests = b.addRunArtifact(cli_unit_tests);

    // CLI tests must run AFTER library tests (not in parallel) because both
    // include library test blocks that create test DB files in the working directory.
    run_cli_unit_tests.step.dependOn(&run_lib_unit_tests.step);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_cli_unit_tests.step);

    // Add lib-only test step for debugging
    const lib_test_step = b.step("test-lib", "Run library tests only");
    lib_test_step.dependOn(&run_lib_unit_tests.step);

    // Simple benchmark executable
    const bench_mod = b.addModule("bench", .{
        .root_source_file = b.path("bench/simple.zig"),
        .target = target,
        .optimize = optimize,
    });
    bench_mod.addImport("silica", mod);

    const bench_exe = b.addExecutable(.{
        .name = "silica-bench",
        .root_module = bench_mod,
    });
    b.installArtifact(bench_exe);

    const run_bench = b.addRunArtifact(bench_exe);
    run_bench.step.dependOn(b.getInstallStep());

    const bench_step = b.step("bench", "Run performance benchmarks");
    bench_step.dependOn(&run_bench.step);

    // TPC-C benchmark executable
    const tpcc_mod = b.addModule("tpcc", .{
        .root_source_file = b.path("bench/tpcc.zig"),
        .target = target,
        .optimize = optimize,
    });
    tpcc_mod.addImport("silica", mod);
    tpcc_mod.addImport("harness", b.addModule("harness", .{
        .root_source_file = b.path("bench/harness.zig"),
        .target = target,
        .optimize = optimize,
    }));

    const tpcc_exe = b.addExecutable(.{
        .name = "tpcc-bench",
        .root_module = tpcc_mod,
    });
    b.installArtifact(tpcc_exe);

    const run_tpcc = b.addRunArtifact(tpcc_exe);
    run_tpcc.step.dependOn(b.getInstallStep());

    const tpcc_step = b.step("tpcc", "Run TPC-C benchmark");
    tpcc_step.dependOn(&run_tpcc.step);
}
