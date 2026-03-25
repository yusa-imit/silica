class Silica < Formula
  desc "Production-grade embedded relational database written in Zig"
  homepage "https://github.com/yusa-imit/silica"
  url "https://github.com/yusa-imit/silica/archive/refs/tags/v0.3.0.tar.gz"
  sha256 "REPLACE_WITH_ACTUAL_SHA256_AFTER_RELEASE"
  license "MIT"
  head "https://github.com/yusa-imit/silica.git", branch: "main"

  depends_on "zig" => :build

  def install
    # Build release binary
    system "zig", "build", "-Doptimize=ReleaseSafe"

    # Install binary
    bin.install "zig-out/bin/silica"

    # Install configuration example
    (etc/"silica").install "silica.conf.example" => "silica.conf"

    # Install documentation
    doc.install "README.md"
    doc.install "docs/API_REFERENCE.md"
    doc.install "docs/GETTING_STARTED.md"
    doc.install "docs/SQL_REFERENCE.md"
    doc.install "docs/OPERATIONS_GUIDE.md"
    doc.install "docs/ARCHITECTURE_GUIDE.md"

    # Create data directory
    (var/"lib/silica").mkpath
  end

  service do
    run [opt_bin/"silica", "server", "--data-dir", var/"lib/silica", "--port", "5433"]
    keep_alive true
    log_path var/"log/silica.log"
    error_log_path var/"log/silica_error.log"
    working_dir var/"lib/silica"
  end

  test do
    # Create test database
    testdb = testpath/"test.db"
    system bin/"silica", testdb, "-c", "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);"
    system bin/"silica", testdb, "-c", "INSERT INTO users VALUES (1, 'Alice');"
    output = shell_output("#{bin}/silica #{testdb} -c 'SELECT name FROM users WHERE id = 1;'")
    assert_match "Alice", output

    # Test version output
    assert_match "silica", shell_output("#{bin}/silica --version")
  end
end
