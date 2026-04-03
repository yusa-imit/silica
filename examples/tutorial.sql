-- Silica Tutorial: Getting Started with SQL
-- This file demonstrates core SQL features in Silica.
-- Run with: silica tutorial.db < tutorial.sql
-- Or interactively: silica tutorial.db
--   silica> .read tutorial.sql

-- ============================================================================
-- Part 1: Tables and Basic Data Types
-- ============================================================================

CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    age INTEGER CHECK (age >= 0),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    content TEXT,
    published BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE post_tags (
    post_id INTEGER NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    tag_id INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    PRIMARY KEY (post_id, tag_id)
);

-- ============================================================================
-- Part 2: Inserting Data
-- ============================================================================

-- Note: Using single-row INSERTs to avoid issue #1 (multi-row INSERT bug)
INSERT INTO users (email, name, age) VALUES ('alice@example.com', 'Alice Smith', 28);
INSERT INTO users (email, name, age) VALUES ('bob@example.com', 'Bob Johnson', 35);
INSERT INTO users (email, name, age) VALUES ('carol@example.com', 'Carol Williams', 42);

INSERT INTO tags (name) VALUES ('tutorial');
INSERT INTO tags (name) VALUES ('sql');
INSERT INTO tags (name) VALUES ('database');
INSERT INTO tags (name) VALUES ('zig');
INSERT INTO tags (name) VALUES ('embedded');

INSERT INTO posts (user_id, title, content, published) VALUES (1, 'Getting Started with Silica', 'Silica is a lightweight embedded database...', TRUE);
INSERT INTO posts (user_id, title, content, published) VALUES (1, 'Advanced SQL Features', 'Learn about CTEs, window functions, and more...', FALSE);
INSERT INTO posts (user_id, title, content, published) VALUES (2, 'Why Zig for Databases', 'Zig provides low-level control with safety...', TRUE);
INSERT INTO posts (user_id, title, content, published) VALUES (3, 'Database Design Patterns', 'Best practices for schema design...', TRUE);

INSERT INTO post_tags (post_id, tag_id) VALUES (1, 1);  -- post 1: tutorial
INSERT INTO post_tags (post_id, tag_id) VALUES (1, 2);  -- post 1: sql
INSERT INTO post_tags (post_id, tag_id) VALUES (1, 3);  -- post 1: database
INSERT INTO post_tags (post_id, tag_id) VALUES (2, 2);  -- post 2: sql
INSERT INTO post_tags (post_id, tag_id) VALUES (3, 4);  -- post 3: zig
INSERT INTO post_tags (post_id, tag_id) VALUES (3, 5);  -- post 3: embedded
INSERT INTO post_tags (post_id, tag_id) VALUES (4, 3);  -- post 4: database

-- ============================================================================
-- Part 3: Basic Queries
-- ============================================================================

-- Select all users
SELECT * FROM users;

-- Filter with WHERE clause
SELECT name, age FROM users WHERE age > 30;

-- Select from posts
SELECT id, user_id, title, published FROM posts WHERE published = TRUE;

-- ============================================================================
-- Part 4: Aggregate Functions
-- ============================================================================

-- Count total posts
SELECT COUNT(*) AS total_posts FROM posts;

-- Count published posts
SELECT COUNT(*) AS published_count FROM posts WHERE published = TRUE;

-- Average content length
SELECT AVG(LENGTH(content)) AS avg_length FROM posts WHERE content IS NOT NULL;

-- ============================================================================
-- Part 5: Subqueries
-- ============================================================================

-- Find users with published posts (using IN subquery)
SELECT name FROM users
WHERE id IN (SELECT user_id FROM posts WHERE published = TRUE);

-- ============================================================================
-- Part 6: Transactions
-- ============================================================================

-- Start a transaction
BEGIN TRANSACTION;

-- Update multiple rows atomically
UPDATE posts SET published = TRUE WHERE user_id = 1;

-- Verify changes
SELECT title, published FROM posts WHERE user_id = 1;

-- Commit the transaction
COMMIT;

-- ============================================================================
-- Part 7: Indexes for Performance
-- ============================================================================

-- Create index on frequently queried column
CREATE INDEX idx_posts_user_id ON posts(user_id);

-- Create composite index
CREATE INDEX idx_posts_published_created ON posts(published, created_at);

-- List all indexes (in interactive mode, use: .indexes)

-- ============================================================================
-- Part 8: Advanced Features
-- ============================================================================

-- CASE expression
SELECT
    title,
    CASE
        WHEN published THEN 'Published'
        ELSE 'Draft'
    END AS status
FROM posts;

-- String functions
SELECT
    UPPER(name) AS name_upper,
    LOWER(email) AS email_lower,
    LENGTH(name) AS name_length
FROM users;

-- COALESCE for NULL handling
SELECT
    title,
    COALESCE(content, '[No content]') AS content_display
FROM posts;

-- ============================================================================
-- Part 9: Cleanup (Optional)
-- ============================================================================

-- Drop tables (in correct order due to foreign keys)
DROP TABLE IF EXISTS post_tags;
DROP TABLE IF EXISTS tags;
DROP TABLE IF EXISTS posts;
DROP TABLE IF EXISTS users;

-- ============================================================================
-- End of Tutorial
-- ============================================================================
-- For more information, see:
-- - SQL Reference: docs/SQL_REFERENCE.md
-- - API Reference: docs/API_REFERENCE.md
-- - Operations Guide: docs/OPERATIONS_GUIDE.md
