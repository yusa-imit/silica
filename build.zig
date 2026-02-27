const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Sailor dependency
    const sailor_dep = b.dependency("sailor", .{
        .target = target,
        .optimize = optimize,
    });

    // Library module — Silica as an embeddable library
    const mod = b.addModule("silica", .{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

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

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_unit_tests.step);
    test_step.dependOn(&run_cli_unit_tests.step);
}
