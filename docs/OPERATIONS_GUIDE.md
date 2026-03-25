# Silica Operations Guide

**Production deployment, monitoring, tuning, and maintenance for Silica database**

> **Version**: v0.12 (Phase 12: Production Readiness)
> **Last Updated**: 2026-03-25

---

## Table of Contents

1. [Installation & Deployment](#installation--deployment)
2. [Backup & Restore](#backup--restore)
3. [Monitoring](#monitoring)
4. [Performance Tuning](#performance-tuning)
5. [Maintenance Operations](#maintenance-operations)
6. [Replication Setup](#replication-setup)
7. [Troubleshooting](#troubleshooting)
8. [Security Best Practices](#security-best-practices)

---

## Installation & Deployment

### System Requirements

**Minimum:**
- OS: Linux, macOS, or Windows
- CPU: 1 core
- RAM: 512 MB
- Disk: 100 MB for binary + database size

**Recommended (Production):**
- OS: Linux (Ubuntu 20.04+, Debian 11+, RHEL 8+)
- CPU: 4+ cores
- RAM: 4+ GB
- Disk: SSD with sufficient IOPS (1000+ IOPS recommended)
- Network: Low latency for replication (< 10ms RTT)

### Installation Methods

#### From Source

```bash
# Clone repository
git clone https://github.com/yusa-imit/silica
cd silica

# Build release binary
zig build -Doptimize=ReleaseFast

# Install to system path
sudo cp zig-out/bin/silica /usr/local/bin/

# Verify installation
silica --version
```

#### From Package Managers

```bash
# Homebrew (macOS/Linux)
brew install silica

# APT (Debian/Ubuntu)
sudo apt install silica

# DNF (Fedora/RHEL)
sudo dnf install silica
```

### Deployment Modes

#### 1. Embedded Mode

For single-process applications, use Silica as an embedded library:

```zig
const silica = @import("silica");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var db = try silica.Database.open(gpa.allocator(), "myapp.db");
    defer db.close();

    // Your application logic
}
```

#### 2. Server Mode (Client-Server)

For multi-client applications, run Silica as a standalone server:

**Configuration file** (`/etc/silica/silica.conf`):

```ini
# Network
listen_addresses = '*'
port = 5433
max_connections = 100

# Memory
shared_buffers = 256MB
work_mem = 4MB
maintenance_work_mem = 64MB

# WAL
wal_level = replica
fsync = on
synchronous_commit = on
wal_buffers = 16MB

# Replication
max_wal_senders = 10
wal_keep_size = 1GB
```

**Systemd service** (`/etc/systemd/system/silica.service`):

```ini
[Unit]
Description=Silica Database Server
After=network.target

[Service]
Type=simple
User=silica
Group=silica
ExecStart=/usr/local/bin/silica server --config /etc/silica/silica.conf --data-dir /var/lib/silica
Restart=on-failure
RestartSec=5s

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/silica

[Install]
WantedBy=multi-user.target
```

**Start server:**

```bash
# Enable and start service
sudo systemctl enable silica
sudo systemctl start silica

# Check status
sudo systemctl status silica
```

### Directory Structure

```
/var/lib/silica/           # Data directory
├── silica.db              # Main database file
├── silica.db-wal          # Write-Ahead Log
├── silica.db-shm          # Shared memory file (WAL index)
├── base/                  # Tablespaces (future)
└── pg_wal/                # WAL archive (for PITR)

/etc/silica/
└── silica.conf            # Configuration file

/var/log/silica/
└── silica.log             # Server logs
```

---

## Backup & Restore

### Backup Strategies

#### 1. Logical Backup (SQL Dump)

**Advantages:**
- Human-readable SQL format
- Cross-version compatible
- Selective table backup
- Portable across platforms

**Disadvantages:**
- Slower for large databases
- Requires more disk space (text format)
- No point-in-time recovery

**Perform logical backup:**

```bash
# Full database dump
silica-dump -h localhost -p 5433 mydb > backup.sql

# Specific tables
silica-dump -h localhost -p 5433 mydb -t users -t orders > partial.sql

# Compressed backup
silica-dump -h localhost -p 5433 mydb | gzip > backup.sql.gz
```

**Restore from dump:**

```bash
# Restore to existing database
psql -h localhost -p 5433 mydb < backup.sql

# Create new database and restore
createdb -h localhost -p 5433 mydb_restored
psql -h localhost -p 5433 mydb_restored < backup.sql

# From compressed backup
gunzip < backup.sql.gz | psql -h localhost -p 5433 mydb
```

#### 2. Physical Backup (File Copy)

**Advantages:**
- Fastest backup method
- Binary-level consistency
- Minimal overhead
- Supports point-in-time recovery

**Disadvantages:**
- Platform-specific
- Requires downtime or careful coordination
- Larger backup size (includes indexes)

**Cold backup (database offline):**

```bash
# Stop database server
sudo systemctl stop silica

# Copy data directory
sudo cp -r /var/lib/silica /backup/silica-$(date +%Y%m%d-%H%M%S)

# Restart database server
sudo systemctl start silica
```

**Hot backup (database online):**

```bash
# Start backup mode (future: pg_start_backup equivalent)
psql -h localhost -p 5433 -c "SELECT pg_start_backup('daily-backup');"

# Copy database files (rsync for incremental)
rsync -av --exclude silica.db-wal /var/lib/silica/ /backup/silica-hot/

# Stop backup mode
psql -h localhost -p 5433 -c "SELECT pg_stop_backup();"
```

#### 3. Base Backup (for Replication)

**Use `silica-basebackup` to create replica-ready backup:**

```bash
# Take base backup from primary server
silica-basebackup -h primary-host -p 5433 -D /var/lib/silica-replica -X stream

# Options:
#   -D: target directory for backup
#   -X stream: stream WAL during backup (recommended)
#   -z: compress backup with gzip
#   -c fast: use fast checkpoint (less impact on primary)
```

### Continuous Archiving & Point-in-Time Recovery (PITR)

**Configuration** (`silica.conf`):

```ini
# Enable WAL archiving
wal_level = replica
archive_mode = on
archive_command = 'cp %p /backup/wal_archive/%f'
```

**Recovery procedure:**

1. **Restore base backup:**

```bash
# Stop database
sudo systemctl stop silica

# Remove current data directory
sudo rm -rf /var/lib/silica/*

# Restore base backup
sudo cp -r /backup/silica-base/* /var/lib/silica/

# Create recovery.conf
cat > /var/lib/silica/recovery.conf << EOF
restore_command = 'cp /backup/wal_archive/%f %p'
recovery_target_time = '2026-03-25 10:30:00'
EOF
```

2. **Start database in recovery mode:**

```bash
# Silica will apply WAL files until target time
sudo systemctl start silica

# Monitor recovery
tail -f /var/log/silica/silica.log
```

3. **Promote to production:**

```bash
# Once recovery completes, database auto-promotes
# Verify with:
psql -h localhost -p 5433 -c "SELECT pg_is_in_recovery();"
# Should return: f (false)
```

### Backup Schedule Recommendations

| Database Size | Backup Type | Frequency | Retention |
|---------------|-------------|-----------|-----------|
| < 10 GB | Logical | Daily | 7 days |
| 10-100 GB | Physical | Daily | 14 days |
| 100+ GB | Physical + PITR | Hourly WAL archiving, daily base backup | 30 days |

**Automated backup script:**

```bash
#!/bin/bash
# /usr/local/bin/silica-backup.sh

DATE=$(date +%Y%m%d-%H%M%S)
BACKUP_DIR="/backup/silica"
DB_HOST="localhost"
DB_PORT="5433"
DB_NAME="mydb"

# Rotate old backups (keep last 7 days)
find "$BACKUP_DIR" -type f -name "*.sql.gz" -mtime +7 -delete

# Perform backup
silica-dump -h "$DB_HOST" -p "$DB_PORT" "$DB_NAME" | gzip > "$BACKUP_DIR/backup-$DATE.sql.gz"

# Verify backup integrity
if [ $? -eq 0 ]; then
    echo "Backup completed: $BACKUP_DIR/backup-$DATE.sql.gz"
else
    echo "ERROR: Backup failed" >&2
    exit 1
fi
```

**Cron schedule:**

```cron
# Daily backup at 2 AM
0 2 * * * /usr/local/bin/silica-backup.sh >> /var/log/silica/backup.log 2>&1
```

---

## Monitoring

### System Monitoring Views

#### 1. Connection Monitoring (`pg_stat_activity`)

**View active connections and queries:**

```sql
SELECT pid, usename, application_name, client_addr, state, query
FROM pg_stat_activity;
```

**Output columns:**

| Column | Type | Description |
|--------|------|-------------|
| `pid` | INTEGER | Process ID of backend |
| `usename` | TEXT | Database user name |
| `application_name` | TEXT | Client application name |
| `client_addr` | TEXT | Client IP address |
| `query` | TEXT | Current query text |
| `state` | TEXT | Backend state (active, idle, idle in transaction) |
| `query_start` | TIMESTAMP | Query start time |
| `state_change` | TIMESTAMP | Last state change time |

**Common queries:**

```sql
-- Count connections by state
SELECT state, COUNT(*)
FROM pg_stat_activity
GROUP BY state;

-- Find long-running queries (> 1 minute)
SELECT pid, query, state,
       EXTRACT(EPOCH FROM (NOW() - query_start)) AS duration_seconds
FROM pg_stat_activity
WHERE state = 'active'
  AND query_start < NOW() - INTERVAL '1 minute';

-- Kill stuck connection
SELECT pg_terminate_backend(12345);  -- Replace with actual PID
```

#### 2. Lock Monitoring (`pg_locks`)

**View current locks:**

```sql
SELECT locktype, mode, pid, relation, tuple, granted
FROM pg_locks;
```

**Output columns:**

| Column | Type | Description |
|--------|------|-------------|
| `locktype` | TEXT | Type of lock (relation, tuple) |
| `mode` | TEXT | Lock mode (ShareLock, ExclusiveLock, etc.) |
| `pid` | INTEGER | Process holding lock |
| `relation` | INTEGER | Table OID (for table locks) |
| `tuple` | INTEGER | Row ID (for row locks) |
| `granted` | BOOLEAN | Whether lock is granted (true) or waiting (false) |

**Common queries:**

```sql
-- Find blocking locks
SELECT
    blocking.pid AS blocking_pid,
    blocking.mode AS blocking_mode,
    blocked.pid AS blocked_pid,
    blocked.mode AS blocked_mode
FROM pg_locks blocking
JOIN pg_locks blocked
  ON blocking.relation = blocked.relation
WHERE blocking.granted = true
  AND blocked.granted = false;

-- Count locks by type
SELECT locktype, mode, COUNT(*)
FROM pg_locks
GROUP BY locktype, mode;
```

#### 3. Replication Monitoring (`pg_stat_replication`)

**View replication status (on primary):**

```sql
SELECT pid, application_name, client_addr, state,
       sent_lsn, write_lsn, flush_lsn, replay_lsn,
       sync_priority, sync_state
FROM pg_stat_replication;
```

**Replication lag calculation:**

```sql
-- Lag in bytes
SELECT
    application_name,
    sent_lsn - replay_lsn AS lag_bytes,
    EXTRACT(EPOCH FROM (NOW() - write_time)) AS lag_seconds
FROM pg_stat_replication;
```

### System Catalog Queries

**List all tables:**

```sql
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public';
```

**Table size:**

```sql
SELECT
    table_name,
    pg_size_pretty(pg_total_relation_size(table_name::regclass)) AS total_size,
    pg_size_pretty(pg_relation_size(table_name::regclass)) AS table_size,
    pg_size_pretty(pg_indexes_size(table_name::regclass)) AS indexes_size
FROM information_schema.tables
WHERE table_schema = 'public';
```

**Index usage:**

```sql
SELECT
    indexname,
    idx_scan AS scans,
    idx_tup_read AS tuples_read,
    idx_tup_fetch AS tuples_fetched
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;
```

### Query Performance Analysis

#### EXPLAIN

**Analyze query execution plan:**

```sql
-- Show logical plan
EXPLAIN SELECT * FROM users WHERE age > 25;

-- Show logical + physical plan with cost estimates
EXPLAIN ANALYZE SELECT * FROM users WHERE age > 25;
```

**Output format:**

```
QueryPlan
├─ SeqScan (table=users, cost=0.00..10.50, rows=100)
│  └─ Filter: (age > 25)
└─ Limit: 1000
```

**Key metrics:**

- **cost**: Estimated execution cost (startup..total)
- **rows**: Estimated row count
- **actual time**: Real execution time (ANALYZE only)
- **actual rows**: Actual row count (ANALYZE only)

#### Slow Query Log

**Enable slow query logging** (`silica.conf`):

```ini
log_min_duration_statement = 1000  # Log queries > 1 second
log_statement = 'all'              # Log all statements
```

**Analyze slow queries:**

```bash
# Parse slow query log
grep "duration:" /var/log/silica/silica.log | \
  awk '{print $3, $NF}' | \
  sort -rn | \
  head -20
```

### Metrics & Alerting

**Key metrics to monitor:**

| Metric | Warning Threshold | Critical Threshold | Action |
|--------|------------------|-------------------|--------|
| Connection count | > 80% of max | > 95% of max | Investigate connection leaks |
| Active query duration | > 1 minute | > 5 minutes | Analyze slow queries |
| Replication lag | > 10 MB | > 100 MB | Check network, disk I/O |
| Disk usage | > 80% | > 90% | Run VACUUM, expand disk |
| Lock wait time | > 10 seconds | > 30 seconds | Investigate blocking locks |

**Prometheus exporter (future):**

```bash
# Start metrics exporter
silica-exporter --port 9187 --db-host localhost --db-port 5433
```

**Grafana dashboard:**

Import pre-built Silica dashboard from [grafana.com/dashboards](https://grafana.com/grafana/dashboards).

---

## Performance Tuning

### Memory Configuration

**Buffer pool** (`silica.conf`):

```ini
# Amount of memory for caching database pages
# Recommendation: 25% of total RAM for dedicated server
shared_buffers = 2GB

# Work memory per query operation (sort, hash)
# Recommendation: total_ram / max_connections / 4
work_mem = 16MB

# Memory for maintenance operations (VACUUM, CREATE INDEX)
# Recommendation: 5-10% of total RAM
maintenance_work_mem = 512MB
```

**Guidelines:**

- **Embedded mode**: Use smaller buffers (64-256 MB) to avoid memory pressure
- **Server mode**: Dedicate 50-75% of system RAM to Silica
- **High concurrency**: Reduce `work_mem` to avoid OOM (Out of Memory)
- **Analytical workloads**: Increase `work_mem` for large sorts/aggregates

### Query Optimization

#### 1. Indexing Strategy

**Create indexes on:**

- Primary keys (automatic)
- Foreign keys (manual)
- Columns in WHERE clauses
- Columns in JOIN conditions
- Columns in ORDER BY clauses

**Index types:**

```sql
-- B+Tree index (default, sorted data)
CREATE INDEX idx_users_age ON users(age);

-- Hash index (equality lookups only)
CREATE INDEX idx_users_email ON users USING HASH (email);

-- GIN index (full-text search, JSONB)
CREATE INDEX idx_articles_content ON articles USING GIN (to_tsvector('english', content));

-- Composite index (multi-column queries)
CREATE INDEX idx_orders_user_date ON orders(user_id, order_date);
```

**Unused index detection:**

```sql
-- Find indexes with zero scans
SELECT schemaname, tablename, indexname, idx_scan
FROM pg_stat_user_indexes
WHERE idx_scan = 0
ORDER BY pg_relation_size(indexrelid) DESC;
```

#### 2. Query Rewriting

**Avoid SELECT *:**

```sql
-- Bad: Fetches all columns (wasted I/O)
SELECT * FROM users WHERE age > 25;

-- Good: Fetch only needed columns
SELECT id, name, email FROM users WHERE age > 25;
```

**Use LIMIT:**

```sql
-- Bad: Fetches all matching rows
SELECT * FROM orders WHERE status = 'pending';

-- Good: Limit result set
SELECT * FROM orders WHERE status = 'pending' LIMIT 100;
```

**Use prepared statements:**

```zig
// Prepare once, execute multiple times (46x faster)
var stmt = try db.prepare("SELECT * FROM users WHERE id = ?");
defer stmt.close();

for (user_ids) |id| {
    try stmt.bind(0, .{ .integer = id });
    var result = try stmt.execute();
    defer result.deinit();
    // Process result
}
```

#### 3. Table Design

**Normalize data:**

- Eliminate redundant data (reduce storage, update anomalies)
- Use foreign keys for referential integrity
- Split large tables into related smaller tables

**Denormalize for read-heavy workloads:**

- Pre-join frequently accessed data
- Use materialized views for complex aggregates
- Trade storage space for query speed

**Partitioning (future):**

```sql
-- Partition large table by date range
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    order_date DATE NOT NULL,
    -- ...
) PARTITION BY RANGE (order_date);

CREATE TABLE orders_2026_q1 PARTITION OF orders
    FOR VALUES FROM ('2026-01-01') TO ('2026-04-01');
```

### I/O Optimization

**WAL configuration:**

```ini
# WAL mode (minimal, replica, logical)
wal_level = replica

# Force fsync after each commit (safety vs speed)
fsync = on
synchronous_commit = on  # Safe, slower
# synchronous_commit = off  # Faster, risk of data loss on crash

# WAL buffer size (reduce disk writes)
wal_buffers = 16MB
```

**Checkpoint tuning:**

```ini
# Checkpoint frequency (balance between recovery time and I/O)
checkpoint_timeout = 5min
max_wal_size = 1GB
min_wal_size = 80MB

# Spread checkpoint I/O over time (reduce spikes)
checkpoint_completion_target = 0.9
```

**SSD optimization:**

```ini
# Increase random_page_cost for SSDs (default: 4.0)
random_page_cost = 1.1

# Enable parallel queries (multi-core CPUs)
max_parallel_workers = 8
max_parallel_workers_per_gather = 4
```

### Benchmarking

**TPC-C (OLTP):**

```bash
# Run TPC-C benchmark
zig build tpcc

# Sample output:
# TPC-C Results:
#   Transactions: 5000
#   Duration: 60s
#   tpmC: 5000
#   Avg Latency: 12ms
```

**TPC-H (OLAP):**

```bash
# Run TPC-H benchmark
zig build tpch

# Sample output:
# TPC-H Results:
#   Q1 (Pricing Summary): 250ms
#   Q3 (Shipping Priority): 180ms
#   Q6 (Forecasting): 45ms
```

**Custom benchmarks:**

```bash
# Simple point lookup benchmark
for i in {1..10000}; do
    psql -h localhost -p 5433 -c "SELECT * FROM users WHERE id = $i" > /dev/null
done
```

---

## Maintenance Operations

### VACUUM

**Purpose:**
- Reclaim dead tuple space (from UPDATEs, DELETEs)
- Update table statistics for query planner
- Prevent transaction ID wraparound

**Usage:**

```sql
-- VACUUM all tables (cannot run inside transaction)
VACUUM;

-- VACUUM specific table
VACUUM users;

-- VACUUM with ANALYZE (update statistics)
VACUUM ANALYZE users;

-- Auto-vacuum (background process)
-- Configured in silica.conf:
autovacuum = on
autovacuum_naptime = 1min
autovacuum_vacuum_threshold = 50
autovacuum_vacuum_scale_factor = 0.2
```

**When to run VACUUM:**

- After bulk DELETE or UPDATE operations
- When query performance degrades
- Before taking backups (reduce file size)
- Automatically via auto-vacuum (recommended)

**Monitoring:**

```sql
-- Check dead tuple ratio
SELECT
    schemaname,
    relname,
    n_live_tup AS live_tuples,
    n_dead_tup AS dead_tuples,
    ROUND(100.0 * n_dead_tup / NULLIF(n_live_tup + n_dead_tup, 0), 2) AS dead_ratio
FROM pg_stat_user_tables
WHERE n_dead_tup > 0
ORDER BY dead_ratio DESC;
```

### ANALYZE

**Purpose:**
- Collect table statistics (row count, data distribution)
- Improve query planner decisions (index selection, join order)

**Usage:**

```sql
-- ANALYZE all tables
ANALYZE;

-- ANALYZE specific table
ANALYZE orders;

-- ANALYZE specific column
ANALYZE orders(user_id);
```

**When to run ANALYZE:**

- After bulk data loads (INSERT, COPY)
- After creating new indexes
- When query plans look incorrect (EXPLAIN)
- Automatically via auto-analyze (recommended)

**Configuration:**

```ini
autovacuum_analyze_threshold = 50
autovacuum_analyze_scale_factor = 0.1
```

### REINDEX

**Purpose:**
- Rebuild corrupted indexes
- Compact bloated indexes
- Improve index performance

**Usage:**

```sql
-- REINDEX specific index
REINDEX INDEX idx_users_age;

-- REINDEX all indexes on a table
REINDEX TABLE users;

-- REINDEX all indexes in database
REINDEX DATABASE mydb;
```

**When to run REINDEX:**

- After heavy UPDATE/DELETE workload (index bloat)
- After index corruption detected
- During low-traffic maintenance windows

**Monitoring index bloat:**

```sql
SELECT
    indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size,
    idx_scan AS scans,
    idx_tup_read AS tuples_read
FROM pg_stat_user_indexes
ORDER BY pg_relation_size(indexrelid) DESC;
```

---

## Replication Setup

### Architecture

```
┌─────────────┐       WAL Stream        ┌─────────────┐
│   Primary   │ ────────────────────────>│   Replica   │
│   (Master)  │                          │  (Standby)  │
└─────────────┘                          └─────────────┘
      │                                         │
      │                                         │
   Clients                                  Read-only
  (Read/Write)                               Queries
```

### Primary Server Setup

**1. Configure WAL archiving** (`/etc/silica/silica.conf`):

```ini
# Replication settings
wal_level = replica
max_wal_senders = 10
wal_keep_size = 1GB

# Synchronous replication (optional, slower but safer)
synchronous_standby_names = 'replica1,replica2'
synchronous_commit = on
```

**2. Create replication user:**

```sql
CREATE USER replicator WITH REPLICATION PASSWORD 'secure_password';
```

**3. Allow replica connections** (`pg_hba.conf` equivalent):

```ini
# Allow replication from replica servers
host replication replicator 192.168.1.0/24 scram-sha-256
```

**4. Restart primary server:**

```bash
sudo systemctl restart silica
```

### Replica Server Setup

**1. Take base backup from primary:**

```bash
# Create data directory
sudo mkdir -p /var/lib/silica-replica
sudo chown silica:silica /var/lib/silica-replica

# Take base backup
silica-basebackup -h primary-host -p 5433 \
    -D /var/lib/silica-replica \
    -U replicator -X stream
```

**2. Create `standby.signal` file:**

```bash
touch /var/lib/silica-replica/standby.signal
```

**3. Configure replica** (`/var/lib/silica-replica/silica.conf`):

```ini
# Connection to primary
primary_conninfo = 'host=primary-host port=5433 user=replicator password=secure_password'

# Hot standby (allow read-only queries)
hot_standby = on

# Restore command for WAL archiving
restore_command = 'cp /backup/wal_archive/%f %p'
```

**4. Start replica:**

```bash
sudo systemctl start silica-replica
```

**5. Verify replication:**

```sql
-- On primary:
SELECT * FROM pg_stat_replication;

-- On replica:
SELECT pg_is_in_recovery();  -- Should return 't' (true)
```

### Failover & Switchover

#### Automatic Failover (Replica Promotion)

**Trigger promotion on replica:**

```bash
# Promote replica to primary
sudo silica-ctl promote -D /var/lib/silica-replica
```

**Result:**
- Replica stops accepting WAL from old primary
- Replica becomes new primary (read-write)
- Old primary must be reconfigured as replica

#### Controlled Switchover

**1. Stop writes on primary:**

```sql
-- Block new connections
ALTER SYSTEM SET max_connections TO 0;
SELECT pg_reload_conf();

-- Wait for active queries to finish
SELECT * FROM pg_stat_activity WHERE state = 'active';
```

**2. Ensure replica is caught up:**

```sql
-- On primary:
SELECT sent_lsn FROM pg_stat_replication WHERE application_name = 'replica1';

-- On replica:
SELECT replay_lsn FROM pg_stat_wal_receiver;

-- Wait until sent_lsn == replay_lsn
```

**3. Promote replica:**

```bash
sudo silica-ctl promote -D /var/lib/silica-replica
```

**4. Reconfigure old primary as replica:**

```bash
# Stop old primary
sudo systemctl stop silica

# Create standby.signal
touch /var/lib/silica/standby.signal

# Update primary_conninfo to point to new primary
vi /var/lib/silica/silica.conf

# Start as replica
sudo systemctl start silica
```

---

## Troubleshooting

### Common Issues

#### 1. Database Won't Start

**Symptoms:**
- `silica server` exits immediately
- Error: "could not open database file"

**Diagnosis:**

```bash
# Check permissions
ls -la /var/lib/silica/

# Check disk space
df -h

# Check error logs
tail -f /var/log/silica/silica.log
```

**Solutions:**

```bash
# Fix permissions
sudo chown -R silica:silica /var/lib/silica

# Free disk space
sudo silica VACUUM;  # If database is accessible

# Repair corrupted database
silica-recovery --data-dir /var/lib/silica
```

#### 2. High CPU Usage

**Symptoms:**
- Server process consuming 100% CPU
- Slow query responses

**Diagnosis:**

```sql
-- Find expensive queries
SELECT pid, query, state,
       EXTRACT(EPOCH FROM (NOW() - query_start)) AS duration
FROM pg_stat_activity
WHERE state = 'active'
ORDER BY duration DESC;
```

**Solutions:**

```sql
-- Kill expensive query
SELECT pg_terminate_backend(12345);

-- Add missing index
CREATE INDEX idx_orders_user ON orders(user_id);

-- Increase work_mem for sorts
SET work_mem = '64MB';
```

#### 3. Out of Memory (OOM)

**Symptoms:**
- Database killed by OOM killer
- Error: "out of memory"

**Diagnosis:**

```bash
# Check kernel logs
dmesg | grep -i "out of memory"

# Check memory usage
free -h
ps aux | grep silica
```

**Solutions:**

```ini
# Reduce memory settings in silica.conf
shared_buffers = 512MB  # Down from 2GB
work_mem = 4MB          # Down from 16MB
max_connections = 50    # Down from 100
```

#### 4. Replication Lag

**Symptoms:**
- Replica data is stale
- High replication lag (> 100 MB)

**Diagnosis:**

```sql
-- On primary:
SELECT
    application_name,
    sent_lsn - replay_lsn AS lag_bytes,
    state
FROM pg_stat_replication;
```

**Solutions:**

- **Network bottleneck**: Check bandwidth, latency
- **Disk I/O bottleneck**: Upgrade to SSD, increase IOPS
- **Replica overload**: Scale out (add more replicas), offload reads
- **WAL accumulation**: Increase `wal_keep_size` on primary

#### 5. Deadlocks

**Symptoms:**
- Error: "deadlock detected"
- Transactions timing out

**Diagnosis:**

```sql
-- Monitor locks
SELECT * FROM pg_locks WHERE NOT granted;

-- Enable deadlock logging
SET log_lock_waits = on;
SET deadlock_timeout = 1s;
```

**Solutions:**

- **Acquire locks in consistent order**: Always lock tables/rows in same order
- **Use shorter transactions**: Reduce lock hold time
- **Retry logic**: Handle deadlock errors gracefully in application

```zig
// Deadlock retry pattern
const max_retries = 3;
var retries: usize = 0;

while (retries < max_retries) : (retries += 1) {
    db.begin(.repeatable_read) catch |err| {
        if (err == error.DeadlockDetected) continue;
        return err;
    };

    // Your transaction logic

    db.commit() catch |err| {
        if (err == error.DeadlockDetected) {
            try db.rollback();
            continue;
        }
        return err;
    };

    break;  // Success
}
```

---

## Security Best Practices

### 1. Authentication

**Use strong authentication:**

```ini
# Require encrypted passwords (SCRAM-SHA-256)
password_encryption = scram-sha-256

# Disable trust authentication in production
# pg_hba.conf:
host all all 0.0.0.0/0 scram-sha-256  # NOT trust
```

**Create role-based users:**

```sql
-- Create read-only user
CREATE USER readonly WITH PASSWORD 'strong_password';
GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly;

-- Create application user
CREATE USER appuser WITH PASSWORD 'strong_password';
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO appuser;

-- Revoke public access
REVOKE ALL ON DATABASE mydb FROM PUBLIC;
```

### 2. Network Security

**Bind to specific interfaces:**

```ini
# Listen only on private network (NOT 0.0.0.0)
listen_addresses = '192.168.1.10'
```

**Enable TLS encryption:**

```ini
ssl = on
ssl_cert_file = '/etc/silica/server.crt'
ssl_key_file = '/etc/silica/server.key'
ssl_ca_file = '/etc/silica/root.crt'
```

**Firewall rules:**

```bash
# Allow Silica port only from application servers
sudo ufw allow from 192.168.1.0/24 to any port 5433
sudo ufw deny 5433
```

### 3. Data Encryption

**Encrypted connections:**

```bash
# Client must use SSL
psql "sslmode=require host=db.example.com port=5433 dbname=mydb"
```

**Transparent data encryption (future):**

```ini
# Encrypt data at rest
data_encryption = on
encryption_key_file = '/etc/silica/encryption.key'
```

### 4. Audit Logging

**Enable comprehensive logging:**

```ini
# Log all connections and disconnections
log_connections = on
log_disconnections = on

# Log all DDL statements (CREATE, ALTER, DROP)
log_statement = 'ddl'

# Log slow queries (> 1 second)
log_min_duration_statement = 1000

# Log authentication failures
log_failed_logins = on
```

**Review logs regularly:**

```bash
# Monitor authentication failures
grep "FAILED" /var/log/silica/silica.log

# Monitor privilege escalation attempts
grep "permission denied" /var/log/silica/silica.log
```

### 5. Principle of Least Privilege

**Grant minimal permissions:**

```sql
-- Bad: Grant all privileges
GRANT ALL PRIVILEGES ON DATABASE mydb TO appuser;

-- Good: Grant only needed privileges
GRANT SELECT, INSERT ON orders TO appuser;
GRANT UPDATE (status) ON orders TO appuser;  -- Column-level
```

**Use row-level security (RLS):**

```sql
-- Enable RLS
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;

-- Policy: Users can only see their own orders
CREATE POLICY user_orders ON orders
    FOR SELECT
    USING (user_id = current_user_id());
```

### 6. Backup Security

**Encrypt backups:**

```bash
# Encrypt backup with GPG
silica-dump mydb | gzip | gpg -e -r admin@example.com > backup.sql.gz.gpg

# Decrypt and restore
gpg -d backup.sql.gz.gpg | gunzip | psql mydb
```

**Store backups securely:**

- Use encrypted storage (AWS S3 with SSE, Azure Blob with encryption)
- Restrict access (IAM policies, signed URLs)
- Test restore procedures regularly

---

## Appendix

### Configuration Parameter Reference

See [CONFIGURATION.md](CONFIGURATION.md) for full parameter reference.

### SQL Command Reference

See [SQL_REFERENCE.md](SQL_REFERENCE.md) for complete SQL syntax.

### API Reference

See [API_REFERENCE.md](API_REFERENCE.md) for Zig embedded API and C FFI.

### Further Reading

- **PostgreSQL Documentation**: [postgresql.org/docs](https://www.postgresql.org/docs/)
- **SQLite Internals**: [sqlite.org/arch.html](https://www.sqlite.org/arch.html)
- **Database Reliability Engineering**: O'Reilly Media

---

**Questions or Issues?**

- GitHub: [github.com/yusa-imit/silica/issues](https://github.com/yusa-imit/silica/issues)
- Email: yusa@example.com

**Last Updated**: 2026-03-25
**Version**: v0.12 (Phase 12: Production Readiness)
