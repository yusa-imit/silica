# Security Policy

## Supported Versions

We take security seriously and actively maintain the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0.0 | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in Silica, please report it responsibly.

### How to Report

**DO NOT** create a public GitHub issue for security vulnerabilities.

Instead, please report vulnerabilities via one of the following methods:

1. **GitHub Security Advisories** (preferred):
   - Go to https://github.com/yusa-imit/silica/security/advisories
   - Click "New draft security advisory"
   - Provide detailed information about the vulnerability

2. **Email**:
   - Send details to: [security@silica-db.org]
   - Use PGP encryption if possible (key available on request)
   - Include "SECURITY" in the subject line

### What to Include

Please provide the following information in your report:

- **Description**: Clear explanation of the vulnerability
- **Impact**: Potential security impact (data leak, DoS, privilege escalation, etc.)
- **Affected Versions**: Which versions are vulnerable
- **Steps to Reproduce**: Detailed steps or proof-of-concept
- **Suggested Fix**: If you have ideas for mitigation
- **Disclosure Timeline**: When you plan to publicly disclose (if applicable)

### Example Report

```
Subject: SECURITY - SQL Injection in Prepared Statement Binding

Description:
Parameter binding in prepared statements does not properly escape
certain control characters, allowing SQL injection.

Impact:
- Arbitrary SQL execution
- Data exfiltration
- Potential database corruption

Affected Versions:
1.0.0 - 1.2.0

Steps to Reproduce:
1. Prepare statement: SELECT * FROM users WHERE name = $1
2. Bind parameter: "'; DROP TABLE users; --"
3. Execute query
4. Observe that DROP TABLE executes

Suggested Fix:
Use parameterized queries at the wire protocol level,
not string concatenation.

Disclosure Timeline:
Plan to disclose 90 days after initial report.
```

## Response Timeline

We aim to respond to security reports within:

- **Initial Response**: 48 hours
- **Triage and Severity Assessment**: 5 business days
- **Fix Development**: Depends on severity
  - Critical: 7 days
  - High: 14 days
  - Medium: 30 days
  - Low: 60 days
- **Public Disclosure**: After fix is released

## Severity Levels

We use the following severity classification:

### Critical
- Remote code execution
- Authentication bypass
- Complete database compromise
- Data exfiltration at scale

### High
- Privilege escalation
- SQL injection
- Denial of service (network-level)
- Data corruption

### Medium
- Denial of service (resource exhaustion)
- Information disclosure (limited)
- CSRF or XSS in server mode

### Low
- Minor information leaks
- Edge case crashes
- Configuration issues

## Security Best Practices

When using Silica in production:

### 1. Network Security (Server Mode)

- **Bind to localhost** by default:
  ```bash
  silica server --host 127.0.0.1 --port 5433 mydb.db
  ```

- **Use authentication**:
  - Configure SCRAM-SHA-256 authentication
  - Never use trust authentication in production
  - Use strong passwords (16+ characters, mixed case, symbols)

- **Enable TLS**:
  ```bash
  silica server --ssl-cert cert.pem --ssl-key key.pem mydb.db
  ```

- **Firewall rules**:
  - Allow only trusted IPs
  - Use VPN or SSH tunneling for remote access

### 2. File Permissions (Embedded Mode)

- Database file: `chmod 600 mydb.db` (owner read/write only)
- WAL file: `chmod 600 mydb.db-wal`
- Config file: `chmod 600 silica.conf`
- Avoid running as root

### 3. SQL Injection Prevention

- **Always use prepared statements**:
  ```zig
  const stmt = try db.prepare("SELECT * FROM users WHERE id = $1");
  defer stmt.deinit();
  try stmt.bind(&[_]Value{Value.initInteger(user_id)});
  const result = try stmt.execute();
  ```

- **Never concatenate user input**:
  ```zig
  // BAD - vulnerable to SQL injection
  const query = try std.fmt.allocPrint(allocator, "SELECT * FROM users WHERE name = '{s}'", .{user_input});
  const result = try db.execSQL(query);

  // GOOD - safe parameterized query
  const stmt = try db.prepare("SELECT * FROM users WHERE name = $1");
  try stmt.bind(&[_]Value{Value.initText(user_input)});
  const result = try stmt.execute();
  ```

### 4. Resource Limits

Configure limits to prevent DoS:

```conf
# silica.conf
max_connections = 100
max_memory_mb = 2048
query_timeout_ms = 30000
statement_cache_size = 1000
```

### 5. Monitoring and Auditing

- Monitor `pg_stat_activity` for suspicious queries
- Enable query logging in development
- Set up alerting for:
  - High connection count
  - Long-running queries
  - Authentication failures
  - Disk space exhaustion

### 6. Update Regularly

- Subscribe to security advisories
- Update to latest stable version
- Test updates in staging before production
- Monitor GitHub releases: https://github.com/yusa-imit/silica/releases

## Known Security Considerations

### 1. Single-Writer Mode (Embedded)

Silica uses single-writer concurrency in embedded mode. Multiple processes
writing to the same database file will cause corruption.

**Mitigation**: Use file locking or server mode for multi-process access.

### 2. WAL File Access

WAL files contain unencrypted database changes. Anyone with read access
to the WAL file can see recent transactions.

**Mitigation**: Protect file permissions and encrypt at the filesystem level.

### 3. Memory Limits

Large queries or transactions can exhaust memory, causing OOM.

**Mitigation**: Configure `max_memory_mb` and `query_timeout_ms`.

### 4. PostgreSQL Wire Protocol Compatibility

Server mode implements PostgreSQL wire protocol v3 for client compatibility.
Authentication mechanisms (SCRAM-SHA-256, MD5) follow PostgreSQL standards.

**Mitigation**: Use strong authentication and TLS in production.

## Security Features

Silica includes the following security features:

- **Prepared Statements**: Prevent SQL injection via parameterized queries
- **SCRAM-SHA-256 Authentication**: Strong password hashing (server mode)
- **TLS Support**: Encrypted client-server communication (server mode)
- **MVCC Isolation**: Transaction isolation prevents dirty reads
- **CRC32C Checksums**: Detect page corruption and tampering
- **WAL Integrity**: Crash recovery ensures ACID guarantees
- **Resource Limits**: Configurable memory and connection limits

## Disclosure Policy

We follow responsible disclosure:

1. Security researchers report vulnerabilities privately
2. We develop and test a fix
3. We release a patched version
4. We publish a security advisory 7 days after release
5. Researcher receives credit in advisory (if desired)

We appreciate the security research community and will credit
researchers who report vulnerabilities responsibly.

## Contact

For non-security issues, use GitHub Issues:
https://github.com/yusa-imit/silica/issues

For security issues, use GitHub Security Advisories or email.

---

Thank you for helping keep Silica secure!
