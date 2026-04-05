-- Advanced Silica Features Examples
--
-- This file demonstrates production-grade features that distinguish Silica
-- from basic SQL databases: MVCC transactions, window functions, CTEs,
-- JSON operators, full-text search, and advanced indexing.
--
-- Run with: silica advanced_demo.db < examples/advanced_features.sql

-- ══════════════════════════════════════════════════════════════════════════
-- 1. MVCC Transactions & Isolation Levels
-- ══════════════════════════════════════════════════════════════════════════

-- Create accounts table for MVCC demonstration
CREATE TABLE accounts (
    id INTEGER PRIMARY KEY,
    holder TEXT NOT NULL,
    balance NUMERIC(10, 2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert initial data
INSERT INTO accounts (id, holder, balance) VALUES (1, 'Alice', 1000.00);
INSERT INTO accounts (id, holder, balance) VALUES (2, 'Bob', 500.00);
INSERT INTO accounts (id, holder, balance) VALUES (3, 'Charlie', 1500.00);

-- Transaction with snapshot isolation (default: READ COMMITTED)
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;

-- Verify balances after transfer
SELECT holder, balance FROM accounts ORDER BY id;

-- REPEATABLE READ isolation (per-transaction snapshot)
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ;
-- Query sees consistent snapshot throughout transaction
SELECT COUNT(*), SUM(balance) FROM accounts;
-- Other concurrent transactions won't affect this snapshot
COMMIT;

-- ══════════════════════════════════════════════════════════════════════════
-- 2. Window Functions & Analytics
-- ══════════════════════════════════════════════════════════════════════════

-- Create sales data for analytics
CREATE TABLE sales (
    id INTEGER PRIMARY KEY,
    region TEXT NOT NULL,
    product TEXT NOT NULL,
    amount REAL NOT NULL,
    sale_date TEXT NOT NULL
);

-- Note: Using explicit IDs for compatibility
INSERT INTO sales (id, region, product, amount, sale_date) VALUES (1, 'North', 'Laptop', 1200.00, '2024-01-15');
INSERT INTO sales (id, region, product, amount, sale_date) VALUES (2, 'North', 'Mouse', 25.00, '2024-01-16');
INSERT INTO sales (id, region, product, amount, sale_date) VALUES (3, 'South', 'Laptop', 1150.00, '2024-01-17');
INSERT INTO sales (id, region, product, amount, sale_date) VALUES (4, 'South', 'Keyboard', 75.00, '2024-01-18');
INSERT INTO sales (id, region, product, amount, sale_date) VALUES (5, 'North', 'Monitor', 300.00, '2024-01-19');
INSERT INTO sales (id, region, product, amount, sale_date) VALUES (6, 'East', 'Laptop', 1180.00, '2024-01-20');
INSERT INTO sales (id, region, product, amount, sale_date) VALUES (7, 'East', 'Mouse', 30.00, '2024-01-21');
INSERT INTO sales (id, region, product, amount, sale_date) VALUES (8, 'West', 'Keyboard', 80.00, '2024-01-22');

-- ROW_NUMBER: Assign unique numbers by ordering
SELECT
    region,
    product,
    amount,
    ROW_NUMBER() OVER (ORDER BY amount DESC) AS row_num
FROM sales
ORDER BY amount DESC;

-- RANK: Assign ranks with gaps for ties
SELECT
    product,
    amount,
    RANK() OVER (ORDER BY amount DESC) AS rank
FROM sales;

-- DENSE_RANK: Assign ranks without gaps for ties
SELECT
    product,
    amount,
    DENSE_RANK() OVER (ORDER BY amount DESC) AS dense_rank
FROM sales;

-- ══════════════════════════════════════════════════════════════════════════
-- 3. Common Table Expressions (CTEs) & Recursive Queries
-- ══════════════════════════════════════════════════════════════════════════

-- Simple CTE for readability
WITH regional_totals AS (
    SELECT
        region,
        SUM(amount) AS total,
        COUNT(*) AS count
    FROM sales
    GROUP BY region
)
SELECT
    region,
    total,
    count,
    total / count AS avg_sale
FROM regional_totals
WHERE total > 100
ORDER BY total DESC;

-- Multiple CTEs
WITH
    high_value AS (
        SELECT * FROM sales WHERE amount > 100
    ),
    low_value AS (
        SELECT * FROM sales WHERE amount <= 100
    )
SELECT
    'High Value' AS category,
    COUNT(*) AS count,
    SUM(amount) AS total
FROM high_value
UNION ALL
SELECT
    'Low Value' AS category,
    COUNT(*) AS count,
    SUM(amount) AS total
FROM low_value;

-- Recursive CTE: Generate number sequence
WITH RECURSIVE numbers(n) AS (
    SELECT 1
    UNION ALL
    SELECT n + 1 FROM numbers WHERE n < 10
)
SELECT n FROM numbers;

-- Recursive CTE: Organizational hierarchy
CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    manager_id INTEGER REFERENCES employees(id),
    title TEXT NOT NULL
);

INSERT INTO employees (id, name, manager_id, title) VALUES (1, 'Alice', NULL, 'CEO');
INSERT INTO employees (id, name, manager_id, title) VALUES (2, 'Bob', 1, 'VP Engineering');
INSERT INTO employees (id, name, manager_id, title) VALUES (3, 'Charlie', 1, 'VP Sales');
INSERT INTO employees (id, name, manager_id, title) VALUES (4, 'David', 2, 'Senior Engineer');
INSERT INTO employees (id, name, manager_id, title) VALUES (5, 'Eve', 2, 'Junior Engineer');
INSERT INTO employees (id, name, manager_id, title) VALUES (6, 'Frank', 3, 'Sales Manager');

WITH RECURSIVE org_chart(id, name, title, level, path) AS (
    -- Base case: top-level employees (CEO)
    SELECT id, name, title, 0, name
    FROM employees
    WHERE manager_id IS NULL
    UNION ALL
    -- Recursive case: employees reporting to current level
    SELECT
        e.id,
        e.name,
        e.title,
        oc.level + 1,
        oc.path || ' > ' || e.name
    FROM employees e
    JOIN org_chart oc ON e.manager_id = oc.id
)
SELECT
    level,
    name,
    title,
    path AS reporting_chain
FROM org_chart
ORDER BY path;

-- ══════════════════════════════════════════════════════════════════════════
-- 4. JSON & JSONB Support
-- ══════════════════════════════════════════════════════════════════════════

-- Create table with JSON columns
CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    metadata JSON,
    specs JSONB  -- Binary format for efficient indexing
);

INSERT INTO products (id, name, metadata, specs) VALUES (1, 'Laptop Pro', '{"brand": "TechCo", "year": 2024}', '{"cpu": "Intel i7", "ram": "16GB", "storage": "512GB SSD"}');
INSERT INTO products (id, name, metadata, specs) VALUES (2, 'Phone X', '{"brand": "PhoneCo", "year": 2024}', '{"screen": "6.1 inch", "camera": "48MP", "battery": "4000mAh"}');
INSERT INTO products (id, name, metadata, specs) VALUES (3, 'Tablet Air', '{"brand": "TechCo", "year": 2023}', '{"screen": "10.5 inch", "storage": "256GB", "weight": "450g"}');

-- JSON operators: Extract values
SELECT
    name,
    metadata->>'brand' AS brand,
    metadata->>'year' AS year
FROM products;

-- JSONB containment (@>)
SELECT name
FROM products
WHERE specs @> '{"cpu": "Intel i7"}';

-- JSON path queries
SELECT
    name,
    specs->'ram' AS ram_spec
FROM products
WHERE specs ? 'ram';  -- Key exists check

-- ══════════════════════════════════════════════════════════════════════════
-- 5. Full-Text Search with TSVECTOR
-- ══════════════════════════════════════════════════════════════════════════

-- Create documents table for FTS
CREATE TABLE documents (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    search_vector TSVECTOR
);

-- Insert documents with pre-computed search vectors
INSERT INTO documents (id, title, content, search_vector) VALUES (1, 'Database Fundamentals', 'Learn about ACID properties, transactions, and query optimization.', to_tsvector('Database Fundamentals Learn about ACID properties, transactions, and query optimization.'));
INSERT INTO documents (id, title, content, search_vector) VALUES (2, 'Web Development Guide', 'Modern web development with React, Node.js, and PostgreSQL.', to_tsvector('Web Development Guide Modern web development with React, Node.js, and PostgreSQL.'));
INSERT INTO documents (id, title, content, search_vector) VALUES (3, 'Machine Learning Basics', 'Introduction to neural networks, gradient descent, and optimization.', to_tsvector('Machine Learning Basics Introduction to neural networks, gradient descent, and optimization.'));

-- Full-text search with @@ operator
SELECT title
FROM documents
WHERE search_vector @@ to_tsquery('optimization');

-- Ranked search results
SELECT
    title,
    ts_rank(search_vector, to_tsquery('database | web')) AS relevance
FROM documents
WHERE search_vector @@ to_tsquery('database | web')
ORDER BY relevance DESC;

-- ══════════════════════════════════════════════════════════════════════════
-- 6. Advanced Indexing (Hash, GiST, GIN)
-- ══════════════════════════════════════════════════════════════════════════

-- Hash index for exact-match queries
CREATE INDEX idx_accounts_holder_hash ON accounts USING HASH (holder);

-- GIN index for JSONB containment queries
CREATE INDEX idx_products_specs_gin ON products USING GIN (specs);

-- GIN index for full-text search
CREATE INDEX idx_documents_search_gin ON documents USING GIN (search_vector);

-- GiST index for range queries (example with geometric types if supported)
-- CREATE INDEX idx_spatial_gist ON locations USING GIST (coordinates);

-- CREATE INDEX CONCURRENTLY (non-blocking index builds)
CREATE INDEX CONCURRENTLY idx_sales_date ON sales (sale_date);

-- ══════════════════════════════════════════════════════════════════════════
-- 7. Materialized Views
-- ══════════════════════════════════════════════════════════════════════════

-- Create materialized view for expensive aggregation
CREATE MATERIALIZED VIEW regional_sales_summary AS
SELECT
    region,
    COUNT(*) AS total_sales,
    SUM(amount) AS revenue,
    AVG(amount) AS avg_sale,
    MAX(amount) AS max_sale
FROM sales
GROUP BY region;

-- Query materialized view (fast, uses pre-computed results)
SELECT * FROM regional_sales_summary ORDER BY revenue DESC;

-- Refresh materialized view after data changes
-- REFRESH MATERIALIZED VIEW regional_sales_summary;

-- ══════════════════════════════════════════════════════════════════════════
-- 8. Set Operations (UNION, INTERSECT, EXCEPT)
-- ══════════════════════════════════════════════════════════════════════════

-- Create additional tables for set operations
CREATE TABLE customers_2023 (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE TABLE customers_2024 (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);

INSERT INTO customers_2023 (id, name) VALUES (1, 'Alice');
INSERT INTO customers_2023 (id, name) VALUES (2, 'Bob');
INSERT INTO customers_2023 (id, name) VALUES (3, 'Charlie');

INSERT INTO customers_2024 (id, name) VALUES (2, 'Bob');
INSERT INTO customers_2024 (id, name) VALUES (3, 'Charlie');
INSERT INTO customers_2024 (id, name) VALUES (4, 'David');

-- UNION: All customers (deduplicated)
SELECT name FROM customers_2023
UNION
SELECT name FROM customers_2024;

-- INTERSECT: Returning customers (in both years)
SELECT name FROM customers_2023
INTERSECT
SELECT name FROM customers_2024;

-- EXCEPT: Lost customers (2023 only)
SELECT name FROM customers_2023
EXCEPT
SELECT name FROM customers_2024;

-- ══════════════════════════════════════════════════════════════════════════
-- 9. Constraints & Data Integrity
-- ══════════════════════════════════════════════════════════════════════════

-- Create table with comprehensive constraints
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES employees(id) ON DELETE CASCADE,
    order_date TEXT NOT NULL,
    total REAL NOT NULL CHECK (total >= 0),
    status TEXT NOT NULL CHECK (status IN ('pending', 'shipped', 'delivered', 'cancelled')),
    UNIQUE (customer_id, order_date)  -- One order per customer per day
);

-- Foreign key with CASCADE
INSERT INTO orders (id, customer_id, order_date, total, status) VALUES (1, 1, '2024-04-06', 199.99, 'shipped');
INSERT INTO orders (id, customer_id, order_date, total, status) VALUES (2, 2, '2024-04-06', 49.99, 'pending');

-- CHECK constraint violation (uncommitted, will fail)
-- INSERT INTO orders (customer_id, total, status) VALUES (1, -10.00, 'pending');

-- ══════════════════════════════════════════════════════════════════════════
-- 10. Performance Analysis with EXPLAIN ANALYZE
-- ══════════════════════════════════════════════════════════════════════════

-- Analyze query execution plan (text format)
EXPLAIN SELECT * FROM sales WHERE region = 'North';

-- EXPLAIN ANALYZE: Show runtime statistics
EXPLAIN ANALYZE SELECT
    region,
    COUNT(*) AS count,
    SUM(amount) AS total
FROM sales
WHERE amount > 50
GROUP BY region;

-- Compare index vs sequential scan
EXPLAIN SELECT * FROM products WHERE name = 'Laptop Pro';

-- Check join strategy
EXPLAIN SELECT
    e.name AS employee,
    o.total AS order_total
FROM employees e
JOIN orders o ON e.id = o.customer_id
WHERE o.status = 'shipped';

-- ══════════════════════════════════════════════════════════════════════════
-- Summary
-- ══════════════════════════════════════════════════════════════════════════

-- This file demonstrates:
-- ✓ MVCC transactions with isolation levels
-- ✓ Window functions (ROW_NUMBER, RANK, LAG, LEAD, running totals)
-- ✓ CTEs (simple, multiple, recursive)
-- ✓ JSON/JSONB operators and indexing
-- ✓ Full-text search with TSVECTOR/TSQUERY
-- ✓ Advanced index types (Hash, GiST, GIN)
-- ✓ Materialized views
-- ✓ Set operations (UNION, INTERSECT, EXCEPT)
-- ✓ Comprehensive constraints (CHECK, UNIQUE, FOREIGN KEY)
-- ✓ Query performance analysis (EXPLAIN, EXPLAIN ANALYZE)

-- For more examples, see:
--   - quickstart.sql       (basic SQL syntax)
--   - tutorial.sql         (comprehensive features tour)
--   - tutorial_simple.sql  (core operations without complex joins)
