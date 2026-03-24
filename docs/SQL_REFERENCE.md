# Silica SQL Reference

Complete reference for SQL statements, data types, functions, and operators supported by Silica.

---

## Table of Contents

- [Data Definition Language (DDL)](#data-definition-language-ddl)
- [Data Manipulation Language (DML)](#data-manipulation-language-dml)
- [Data Query Language (DQL)](#data-query-language-dql)
- [Transaction Control](#transaction-control)
- [Data Types](#data-types)
- [Operators](#operators)
- [Built-in Functions](#built-in-functions)
- [Indexes](#indexes)
- [Views](#views)
- [Triggers](#triggers)
- [System Catalog](#system-catalog)

---

## Data Definition Language (DDL)

### CREATE TABLE

```sql
CREATE TABLE table_name (
    column_name data_type [constraints],
    ...
    [table_constraints]
);
```

**Column constraints:**
- `PRIMARY KEY` — Unique identifier, creates B+Tree index
- `NOT NULL` — Disallow NULL values
- `UNIQUE` — Unique constraint, creates B+Tree index
- `CHECK (expression)` — Custom validation
- `DEFAULT value` — Default value for column
- `GENERATED ALWAYS AS (expression) STORED` — Computed column
- `REFERENCES table(column)` — Foreign key

**Table constraints:**
- `PRIMARY KEY (col1, col2, ...)` — Composite primary key
- `UNIQUE (col1, col2, ...)` — Composite unique constraint
- `CHECK (expression)` — Table-level check
- `FOREIGN KEY (col) REFERENCES table(col) [ON DELETE CASCADE|SET NULL|RESTRICT] [ON UPDATE CASCADE|SET NULL|RESTRICT]`

**Examples:**
```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    email TEXT CHECK (email LIKE '%@%'),
    age INTEGER CHECK (age >= 0 AND age < 150),
    created_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE posts (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    title TEXT NOT NULL CHECK (length(title) > 0),
    content TEXT,
    status TEXT DEFAULT 'draft' CHECK (status IN ('draft', 'published', 'archived')),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
```

### ALTER TABLE

```sql
ALTER TABLE table_name
    ADD COLUMN column_name data_type [constraints];

ALTER TABLE table_name
    DROP COLUMN column_name;

ALTER TABLE table_name
    RENAME TO new_table_name;

ALTER TABLE table_name
    RENAME COLUMN old_name TO new_name;
```

**Examples:**
```sql
ALTER TABLE users ADD COLUMN phone TEXT;
ALTER TABLE users DROP COLUMN phone;
ALTER TABLE users RENAME TO accounts;
ALTER TABLE users RENAME COLUMN username TO login;
```

### DROP TABLE

```sql
DROP TABLE [IF EXISTS] table_name [CASCADE];
```

**Examples:**
```sql
DROP TABLE temp_data;
DROP TABLE IF EXISTS old_table;
DROP TABLE users CASCADE;  -- Also drop dependent views/triggers
```

### CREATE INDEX

See [Indexes](#indexes) section.

---

## Data Manipulation Language (DML)

### INSERT

```sql
INSERT INTO table_name [(column1, column2, ...)]
VALUES (value1, value2, ...),
       (value1, value2, ...);

INSERT INTO table_name [(column1, column2, ...)]
SELECT ...;

INSERT INTO table_name [(column1, column2, ...)]
VALUES (...)
RETURNING column1, column2, ...;
```

**Examples:**
```sql
-- Single row
INSERT INTO users (id, username, email)
VALUES (1, 'alice', 'alice@example.com');

-- Multiple rows
INSERT INTO users (id, username, email) VALUES
    (2, 'bob', 'bob@example.com'),
    (3, 'carol', 'carol@example.com');

-- From SELECT
INSERT INTO archived_users
SELECT * FROM users WHERE last_login < current_timestamp - interval '1 year';

-- With RETURNING
INSERT INTO users (username, email)
VALUES ('dave', 'dave@example.com')
RETURNING id;  -- Returns the auto-generated id
```

### UPDATE

```sql
UPDATE table_name
SET column1 = value1, column2 = value2, ...
[WHERE condition]
[RETURNING columns];
```

**Examples:**
```sql
UPDATE users
SET email = 'newemail@example.com'
WHERE id = 1;

-- Expressions
UPDATE accounts
SET balance = balance - 100
WHERE id = 1;

-- Multiple columns
UPDATE posts
SET status = 'published', published_at = current_timestamp
WHERE id = 42;

-- With RETURNING
UPDATE users SET email = 'new@example.com' WHERE id = 1 RETURNING username;
```

**⚠️ Known Issue (#16):** Lost update race condition in `UPDATE SET col = col + value` under READ COMMITTED. Use REPEATABLE READ or SELECT FOR UPDATE as workaround.

### DELETE

```sql
DELETE FROM table_name
[WHERE condition]
[RETURNING columns];
```

**Examples:**
```sql
DELETE FROM users WHERE id = 1;
DELETE FROM posts WHERE created_at < current_timestamp - interval '1 year';
DELETE FROM users WHERE id IN (SELECT user_id FROM banned_users);

-- With RETURNING
DELETE FROM users WHERE id = 1 RETURNING username, email;
```

---

## Data Query Language (DQL)

### SELECT

```sql
SELECT [DISTINCT] [DISTINCT ON (expression)] select_list
FROM table_name
[JOIN ...]
[WHERE condition]
[GROUP BY columns]
[HAVING condition]
[WINDOW window_name AS (window_spec)]
[ORDER BY columns [ASC|DESC]]
[LIMIT count]
[OFFSET count];
```

**Select list:**
- `*` — All columns
- `column_name` — Specific column
- `expression AS alias` — Computed column with alias
- `table_name.*` — All columns from table

**Examples:**
```sql
-- Basic
SELECT * FROM users;
SELECT id, username FROM users;

-- WHERE
SELECT * FROM users WHERE age > 25;
SELECT * FROM users WHERE username LIKE 'a%' AND email IS NOT NULL;

-- ORDER BY
SELECT * FROM users ORDER BY created_at DESC;
SELECT * FROM products ORDER BY category, price DESC;

-- LIMIT and OFFSET
SELECT * FROM posts ORDER BY created_at DESC LIMIT 10;
SELECT * FROM posts ORDER BY id LIMIT 10 OFFSET 20;  -- Page 3

-- DISTINCT
SELECT DISTINCT category FROM products;
SELECT DISTINCT ON (user_id) user_id, created_at FROM posts ORDER BY user_id, created_at DESC;
```

### JOINs

```sql
-- INNER JOIN (only matching rows)
SELECT users.username, posts.title
FROM users
INNER JOIN posts ON users.id = posts.user_id;

-- LEFT JOIN (all left rows, null-filled right if no match)
SELECT users.username, posts.title
FROM users
LEFT JOIN posts ON users.id = posts.user_id;

-- CROSS JOIN (Cartesian product)
SELECT * FROM colors CROSS JOIN sizes;

-- Multiple joins
SELECT users.username, posts.title, comments.content
FROM users
INNER JOIN posts ON users.id = posts.user_id
LEFT JOIN comments ON posts.id = comments.post_id;

-- Self join
SELECT e.name AS employee, m.name AS manager
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.id;
```

### GROUP BY and Aggregates

```sql
SELECT column1, aggregate_function(column2)
FROM table_name
GROUP BY column1
[HAVING condition];
```

**Examples:**
```sql
-- COUNT
SELECT category, COUNT(*) FROM products GROUP BY category;

-- SUM, AVG
SELECT user_id, SUM(amount) AS total, AVG(amount) AS average
FROM orders
GROUP BY user_id;

-- HAVING
SELECT category, COUNT(*) as count
FROM products
GROUP BY category
HAVING COUNT(*) > 5;

-- Multiple aggregates
SELECT
    DATE(created_at) as date,
    COUNT(*) as total,
    COUNT(DISTINCT user_id) as unique_users,
    AVG(views) as avg_views
FROM posts
GROUP BY DATE(created_at);
```

### Subqueries

```sql
-- Scalar subquery
SELECT (SELECT COUNT(*) FROM users) AS user_count;

-- IN subquery
SELECT * FROM users WHERE id IN (SELECT user_id FROM premium_members);

-- EXISTS subquery
SELECT * FROM users u WHERE EXISTS (SELECT 1 FROM posts WHERE user_id = u.id);

-- NOT EXISTS
SELECT * FROM users u WHERE NOT EXISTS (SELECT 1 FROM posts WHERE user_id = u.id);

-- Correlated subquery
SELECT username, (SELECT COUNT(*) FROM posts WHERE user_id = users.id) AS post_count
FROM users;
```

### Common Table Expressions (CTEs)

```sql
WITH cte_name AS (
    SELECT ...
)
SELECT * FROM cte_name;
```

**Examples:**
```sql
-- Basic CTE
WITH active_users AS (
    SELECT * FROM users WHERE last_login > current_timestamp - interval '1 day'
)
SELECT * FROM active_users;

-- Multiple CTEs
WITH
    top_authors AS (
        SELECT user_id, COUNT(*) as count FROM posts GROUP BY user_id ORDER BY count DESC LIMIT 10
    ),
    recent_posts AS (
        SELECT * FROM posts WHERE created_at > current_timestamp - interval '1 week'
    )
SELECT a.user_id, u.username, a.count
FROM top_authors a
JOIN users u ON a.user_id = u.id;

-- Recursive CTE
WITH RECURSIVE subordinates AS (
    SELECT id, name, manager_id FROM employees WHERE id = 1
    UNION ALL
    SELECT e.id, e.name, e.manager_id
    FROM employees e
    INNER JOIN subordinates s ON e.manager_id = s.id
)
SELECT * FROM subordinates;
```

### Set Operations

```sql
SELECT ... UNION [ALL] SELECT ...;
SELECT ... INTERSECT SELECT ...;
SELECT ... EXCEPT SELECT ...;
```

**Examples:**
```sql
-- UNION (distinct)
SELECT username FROM users
UNION
SELECT username FROM archived_users;

-- UNION ALL (with duplicates)
SELECT email FROM users
UNION ALL
SELECT email FROM subscribers;

-- INTERSECT
SELECT username FROM premium_users
INTERSECT
SELECT username FROM active_users;

-- EXCEPT (difference)
SELECT username FROM all_users
EXCEPT
SELECT username FROM banned_users;
```

### Window Functions

```sql
SELECT
    column,
    window_function() OVER (
        [PARTITION BY partition_expr]
        [ORDER BY order_expr]
        [frame_clause]
    )
FROM table;
```

**Window functions:**
- `ROW_NUMBER()` — Sequential row number within partition
- `RANK()` — Rank with gaps for ties
- `DENSE_RANK()` — Rank without gaps
- `LAG(expr, offset, default)` — Value from previous row
- `LEAD(expr, offset, default)` — Value from next row
- `FIRST_VALUE(expr)` — First value in window frame
- `LAST_VALUE(expr)` — Last value in window frame
- `NTH_VALUE(expr, n)` — Nth value in window frame

**Examples:**
```sql
-- ROW_NUMBER
SELECT username, score, ROW_NUMBER() OVER (ORDER BY score DESC) as rank
FROM leaderboard;

-- PARTITION BY
SELECT category, product, sales,
       RANK() OVER (PARTITION BY category ORDER BY sales DESC) as rank
FROM products;

-- LAG/LEAD
SELECT date, price,
       price - LAG(price) OVER (ORDER BY date) as change
FROM stock_prices;

-- Moving average (frame spec)
SELECT date, value,
       AVG(value) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) as moving_avg
FROM metrics;
```

---

## Transaction Control

### BEGIN

```sql
BEGIN [TRANSACTION] [ISOLATION LEVEL {READ UNCOMMITTED | READ COMMITTED | REPEATABLE READ | SERIALIZABLE}];
```

**Examples:**
```sql
BEGIN;  -- Default isolation level
BEGIN ISOLATION LEVEL SERIALIZABLE;
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ;
```

### COMMIT

```sql
COMMIT [TRANSACTION];
```

### ROLLBACK

```sql
ROLLBACK [TRANSACTION];
```

### SAVEPOINT

```sql
SAVEPOINT savepoint_name;
RELEASE SAVEPOINT savepoint_name;
ROLLBACK TO SAVEPOINT savepoint_name;
```

**Example:**
```sql
BEGIN;
INSERT INTO users VALUES (1, 'alice');
SAVEPOINT sp1;
INSERT INTO users VALUES (2, 'bob');
ROLLBACK TO SAVEPOINT sp1;  -- Bob not inserted
COMMIT;  -- Only alice committed
```

---

## Data Types

### Numeric Types

| Type | Storage | Range | Description |
|------|---------|-------|-------------|
| `INTEGER` | 8 bytes | -2^63 to 2^63-1 | Signed 64-bit integer |
| `BIGINT` | 8 bytes | -2^63 to 2^63-1 | Alias for INTEGER |
| `SMALLINT` | 8 bytes | -2^63 to 2^63-1 | Alias for INTEGER (no size difference) |
| `REAL` | 8 bytes | IEEE 754 | 64-bit floating point |
| `DOUBLE PRECISION` | 8 bytes | IEEE 754 | Alias for REAL |
| `NUMERIC(p, s)` | Variable | Exact decimal | Fixed-point (precision, scale) |
| `DECIMAL(p, s)` | Variable | Exact decimal | Alias for NUMERIC |

**Examples:**
```sql
CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    price NUMERIC(10, 2),  -- Up to 99999999.99
    discount REAL          -- Approximate
);
```

### String Types

| Type | Description |
|------|-------------|
| `TEXT` | Variable-length UTF-8 string |
| `VARCHAR(n)` | Variable-length string (length limit n) |
| `CHAR(n)` | Fixed-length string (space-padded) |

**Examples:**
```sql
CREATE TABLE users (
    username VARCHAR(50),
    bio TEXT,
    country_code CHAR(2)  -- 'US', 'UK', etc.
);
```

### Binary Types

| Type | Description |
|------|-------------|
| `BLOB` | Binary large object |
| `BYTEA` | Alias for BLOB |

### Boolean Type

| Type | Values |
|------|--------|
| `BOOLEAN` | `TRUE`, `FALSE`, `NULL` |

### Date/Time Types

| Type | Storage | Range | Description |
|------|---------|-------|-------------|
| `DATE` | 4 bytes | 4713 BC to 294276 AD | Calendar date |
| `TIME` | 8 bytes | 00:00:00 to 23:59:59.999999 | Time of day |
| `TIMESTAMP` | 8 bytes | 4713 BC to 294276 AD | Date and time |
| `INTERVAL` | 16 bytes | -178000000 to 178000000 years | Time span |

**Examples:**
```sql
CREATE TABLE events (
    event_date DATE,
    event_time TIME,
    created_at TIMESTAMP DEFAULT current_timestamp,
    duration INTERVAL
);

INSERT INTO events VALUES (
    '2024-03-25',
    '14:30:00',
    current_timestamp,
    interval '2 hours 30 minutes'
);
```

### UUID Type

| Type | Storage | Description |
|------|---------|-------------|
| `UUID` | 16 bytes | RFC 4122 UUID |

**Example:**
```sql
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id INTEGER,
    created_at TIMESTAMP DEFAULT current_timestamp
);
```

### JSON Types

| Type | Storage | Description |
|------|---------|-------------|
| `JSON` | Text | JSON text storage |
| `JSONB` | Binary | Binary JSON (faster, indexed) |

**Example:**
```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    profile JSONB
);

INSERT INTO users VALUES (1, '{"name": "Alice", "age": 30, "tags": ["developer"]}');
SELECT profile->>'name' FROM users WHERE id = 1;  -- 'Alice'
```

### Array Types

```sql
column_name data_type[]
column_name data_type ARRAY
```

**Example:**
```sql
CREATE TABLE articles (
    id INTEGER PRIMARY KEY,
    tags TEXT[],
    scores INTEGER ARRAY
);

INSERT INTO articles VALUES (1, ARRAY['zig', 'database'], ARRAY[10, 20, 30]);
SELECT tags[1] FROM articles WHERE id = 1;  -- 'zig'
```

### Enum Types

```sql
CREATE TYPE enum_name AS ENUM ('value1', 'value2', ...);
```

**Example:**
```sql
CREATE TYPE status AS ENUM ('pending', 'active', 'inactive');

CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    status status DEFAULT 'pending'
);
```

### Serial Types (Auto-increment)

| Type | Description |
|------|-------------|
| `SERIAL` | Auto-incrementing INTEGER |
| `BIGSERIAL` | Auto-incrementing BIGINT |

**Example:**
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,  -- Auto-generates 1, 2, 3, ...
    username TEXT
);

INSERT INTO users (username) VALUES ('alice');  -- id=1
```

---

## Operators

### Comparison Operators

| Operator | Description |
|----------|-------------|
| `=` | Equal |
| `<>`, `!=` | Not equal |
| `<` | Less than |
| `<=` | Less than or equal |
| `>` | Greater than |
| `>=` | Greater than or equal |
| `BETWEEN a AND b` | Range (inclusive) |
| `IN (values)` | Value in list |
| `NOT IN (values)` | Value not in list |
| `IS NULL` | Is NULL |
| `IS NOT NULL` | Is not NULL |
| `LIKE pattern` | Pattern match (% = any, _ = single char) |
| `NOT LIKE pattern` | Pattern not match |
| `ILIKE pattern` | Case-insensitive LIKE |
| `~` | Regex match |
| `~*` | Case-insensitive regex |

### Logical Operators

| Operator | Description |
|----------|-------------|
| `AND` | Logical AND |
| `OR` | Logical OR |
| `NOT` | Logical NOT |

### Arithmetic Operators

| Operator | Description |
|----------|-------------|
| `+` | Addition |
| `-` | Subtraction |
| `*` | Multiplication |
| `/` | Division |
| `%` | Modulo |
| `^` | Exponentiation |

### String Operators

| Operator | Description |
|----------|-------------|
| `||` | Concatenation |

**Example:**
```sql
SELECT 'Hello' || ' ' || 'World';  -- 'Hello World'
```

### JSON Operators

| Operator | Description |
|----------|-------------|
| `->` | Extract JSON field (returns JSON) |
| `->>` | Extract JSON field as text |
| `@>` | Contains (JSONB only) |
| `?` | Key exists (JSONB only) |
| `?|` | Any key exists |
| `?&` | All keys exist |

**Examples:**
```sql
SELECT '{"a": {"b": "c"}}'::jsonb -> 'a';           -- {"b": "c"}
SELECT '{"a": {"b": "c"}}'::jsonb -> 'a' ->> 'b';   -- 'c'
SELECT '{"a": [1,2,3]}'::jsonb @> '{"a": [1]}';     -- true
SELECT '{"a": 1, "b": 2}'::jsonb ? 'a';             -- true
```

### Array Operators

| Operator | Description |
|----------|-------------|
| `||` | Array concatenation |
| `@>` | Contains array |
| `<@` | Contained by array |
| `&&` | Overlaps |

---

## Built-in Functions

### String Functions

| Function | Description | Example |
|----------|-------------|---------|
| `length(str)` | String length | `length('hello')` → 5 |
| `lower(str)` | Lowercase | `lower('HELLO')` → 'hello' |
| `upper(str)` | Uppercase | `upper('hello')` → 'HELLO' |
| `trim(str)` | Remove leading/trailing spaces | `trim(' hello ')` → 'hello' |
| `ltrim(str)` | Remove leading spaces | `ltrim(' hello')` → 'hello' |
| `rtrim(str)` | Remove trailing spaces | `rtrim('hello ')` → 'hello' |
| `substring(str, start, len)` | Extract substring | `substring('hello', 2, 3)` → 'ell' |
| `position(substr IN str)` | Find position | `position('ll' IN 'hello')` → 3 |
| `replace(str, from, to)` | Replace substring | `replace('hello', 'l', 'r')` → 'herro' |
| `concat(str1, str2, ...)` | Concatenate | `concat('a', 'b', 'c')` → 'abc' |
| `concat_ws(sep, str1, ...)` | Concat with separator | `concat_ws(',', 'a', 'b')` → 'a,b' |
| `split_part(str, delim, n)` | Split and get part | `split_part('a-b-c', '-', 2)` → 'b' |
| `starts_with(str, prefix)` | Check prefix | `starts_with('hello', 'he')` → true |
| `regexp_match(str, pattern)` | Regex match (first) | `regexp_match('hello123', '[0-9]+')` → '123' |
| `regexp_replace(str, pat, repl)` | Regex replace | `regexp_replace('hello', 'l+', 'r')` → 'hero' |

### Math Functions

| Function | Description | Example |
|----------|-------------|---------|
| `abs(x)` | Absolute value | `abs(-5)` → 5 |
| `ceil(x)` | Ceiling | `ceil(4.2)` → 5 |
| `floor(x)` | Floor | `floor(4.8)` → 4 |
| `round(x, digits)` | Round | `round(3.14159, 2)` → 3.14 |
| `trunc(x)` | Truncate | `trunc(4.8)` → 4 |
| `mod(x, y)` | Modulo | `mod(10, 3)` → 1 |
| `power(x, y)` | Exponentiation | `power(2, 3)` → 8 |
| `sqrt(x)` | Square root | `sqrt(16)` → 4 |
| `log(x)` | Base-10 logarithm | `log(100)` → 2 |
| `ln(x)` | Natural logarithm | `ln(2.71828)` → 1 |
| `exp(x)` | e^x | `exp(1)` → 2.71828 |
| `pi()` | Pi constant | `pi()` → 3.14159... |
| `random()` | Random [0, 1) | `random()` → 0.73829... |
| `greatest(x, y, ...)` | Maximum | `greatest(1, 5, 3)` → 5 |
| `least(x, y, ...)` | Minimum | `least(1, 5, 3)` → 1 |

### Date/Time Functions

| Function | Description | Example |
|----------|-------------|---------|
| `now()` | Current timestamp | `now()` → '2024-03-25 14:30:00' |
| `current_timestamp` | Current timestamp | Same as `now()` |
| `current_date` | Current date | `current_date` → '2024-03-25' |
| `current_time` | Current time | `current_time` → '14:30:00' |
| `date_trunc(field, ts)` | Truncate | `date_trunc('hour', now())` |
| `date_part(field, ts)` | Extract field | `date_part('year', now())` → 2024 |
| `extract(field FROM ts)` | Extract field | `extract(YEAR FROM now())` → 2024 |
| `age(ts1, ts2)` | Interval between | `age(ts1, ts2)` |
| `make_date(y, m, d)` | Construct date | `make_date(2024, 3, 25)` |
| `make_timestamp(y,m,d,h,min,s)` | Construct timestamp | `make_timestamp(2024,3,25,14,30,0)` |
| `to_char(ts, format)` | Format as string | `to_char(now(), 'YYYY-MM-DD')` |
| `to_timestamp(str, format)` | Parse timestamp | `to_timestamp('2024-03-25', 'YYYY-MM-DD')` |

### Aggregate Functions

| Function | Description | Example |
|----------|-------------|---------|
| `count(*)` | Row count | `SELECT count(*) FROM users` |
| `count(expr)` | Non-null count | `SELECT count(email) FROM users` |
| `count(DISTINCT expr)` | Distinct count | `SELECT count(DISTINCT city) FROM users` |
| `sum(expr)` | Sum | `SELECT sum(amount) FROM orders` |
| `avg(expr)` | Average | `SELECT avg(price) FROM products` |
| `min(expr)` | Minimum | `SELECT min(created_at) FROM posts` |
| `max(expr)` | Maximum | `SELECT max(score) FROM games` |
| `string_agg(expr, delim)` | Concat strings | `SELECT string_agg(name, ', ') FROM users` |
| `array_agg(expr)` | Collect array | `SELECT array_agg(tag) FROM tags` |
| `bool_and(expr)` | Logical AND | `SELECT bool_and(active) FROM users` |
| `bool_or(expr)` | Logical OR | `SELECT bool_or(premium) FROM users` |

### System Functions

| Function | Description | Example |
|----------|-------------|---------|
| `version()` | Database version | `version()` → 'Silica 0.3.0' |
| `current_user` | Current username | `current_user` → 'postgres' |
| `current_database()` | Current database | `current_database()` → 'mydb' |
| `gen_random_uuid()` | Generate UUID | `gen_random_uuid()` → 'a0eebc99-...' |
| `txid_current()` | Current transaction ID | `txid_current()` → 42 |
| `pg_table_size(table)` | Table size (bytes) | `pg_table_size('users')` → 8192 |

---

## Indexes

### CREATE INDEX

```sql
CREATE [UNIQUE] INDEX [CONCURRENTLY] [IF NOT EXISTS] index_name
ON table_name [USING method] (column1 [ASC|DESC], ...);
```

**Index methods:**
- `BTREE` (default) — B+Tree index (range + equality)
- `HASH` — Hash index (equality only, faster)
- `GIN` — Generalized Inverted Index (arrays, JSON, full-text)
- `GIST` — Generalized Search Tree (spatial, custom types)

**Examples:**
```sql
-- B+Tree index (default)
CREATE INDEX idx_users_email ON users(email);

-- Unique index
CREATE UNIQUE INDEX idx_users_username ON users(username);

-- Multi-column index
CREATE INDEX idx_posts_user_status ON posts(user_id, status);

-- Hash index
CREATE INDEX idx_sessions_token USING HASH ON sessions(token);

-- GIN index (JSON)
CREATE INDEX idx_users_profile ON users USING GIN (profile);

-- Concurrent creation (no table locks)
CREATE INDEX CONCURRENTLY idx_large_table ON large_table(column);

-- Partial index
CREATE INDEX idx_active_users ON users(username) WHERE active = true;
```

### DROP INDEX

```sql
DROP INDEX [IF EXISTS] index_name;
```

### REINDEX

```sql
REINDEX INDEX index_name;
REINDEX TABLE table_name;
```

---

## Views

### CREATE VIEW

```sql
CREATE [OR REPLACE] VIEW view_name AS
SELECT ...;
```

**Example:**
```sql
CREATE VIEW active_users AS
SELECT id, username, email
FROM users
WHERE last_login > current_timestamp - interval '1 day';
```

### CREATE MATERIALIZED VIEW

```sql
CREATE MATERIALIZED VIEW view_name AS
SELECT ...;
```

**Example:**
```sql
CREATE MATERIALIZED VIEW user_stats AS
SELECT user_id, COUNT(*) as post_count, SUM(views) as total_views
FROM posts
GROUP BY user_id;

-- Refresh when data changes
REFRESH MATERIALIZED VIEW user_stats;
```

### DROP VIEW

```sql
DROP VIEW [IF EXISTS] view_name;
DROP MATERIALIZED VIEW [IF EXISTS] view_name;
```

---

## Triggers

### CREATE TRIGGER

```sql
CREATE TRIGGER trigger_name
{BEFORE | AFTER} {INSERT | UPDATE | DELETE}
ON table_name
[FOR EACH ROW]
[WHEN (condition)]
BEGIN
    -- Trigger body (Silica Function Language)
END;
```

**Example:**
```sql
CREATE TRIGGER update_timestamp
BEFORE UPDATE ON posts
FOR EACH ROW
BEGIN
    NEW.updated_at = current_timestamp;
END;
```

### DROP TRIGGER

```sql
DROP TRIGGER [IF EXISTS] trigger_name ON table_name;
```

---

## System Catalog

### Information Schema

```sql
-- List tables
SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';

-- List columns
SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'users';
```

### System Views

```sql
-- Active connections
SELECT * FROM pg_stat_activity;

-- Locks
SELECT * FROM pg_locks WHERE NOT granted;

-- Configuration
SHOW ALL;
SET work_mem = '64MB';
RESET work_mem;
```

---

## SQL Conformance

Silica implements SQL:2016 core features with PostgreSQL extensions:

- ✅ E021: Basic data types and DML
- ✅ E031: Identifiers
- ✅ E061: Subqueries
- ✅ E101: Basic data manipulation
- ✅ F031: Basic schema manipulation
- ✅ F051: Datetime types
- ✅ F201: CAST function
- ✅ F261: CASE expression
- ✅ F401: JOIN
- ✅ F850: ORDER BY
- ✅ F851: LIMIT/OFFSET
- ✅ T121: CTEs
- ✅ T141: Window functions
- ✅ T211: Triggers

See [docs/conformance_test.zig](../src/sql/conformance_test.zig) for conformance tests.

---

## Performance Tips

1. **Use indexes wisely** — Index columns used in WHERE, JOIN, ORDER BY
2. **Prepared statements** — 46x-52x faster than exec() for repeated queries
3. **Batch DML in transactions** — Wrap multiple INSERTs/UPDATEs in single transaction
4. **Analyze tables** — Run `ANALYZE table_name` after bulk loads
5. **Partial indexes** — Index only relevant rows (e.g., `WHERE active = true`)
6. **EXPLAIN queries** — Use `EXPLAIN ANALYZE` to understand query plans

---

## See Also

- [API Reference](API_REFERENCE.md) — Zig/C API documentation
- [Getting Started](GETTING_STARTED.md) — Comprehensive tutorial
- [Operations Guide](OPERATIONS_GUIDE.md) — Deployment, monitoring, tuning
