-- Silica Simple Tutorial: Core SQL Features
-- This tutorial demonstrates fundamental SQL operations in Silica.
-- Run with: silica tutorial.db < tutorial_simple.sql

-- ============================================================================
-- Part 1: Create Tables
-- ============================================================================

-- Create a simple users table
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE,
    age INTEGER,
    active BOOLEAN DEFAULT TRUE
);

-- Create a posts table
CREATE TABLE posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    author_id INTEGER,
    title TEXT NOT NULL,
    body TEXT,
    published BOOLEAN DEFAULT FALSE
);

-- ============================================================================
-- Part 2: Insert Data
-- ============================================================================

INSERT INTO users (name, email, age) VALUES ('Alice', 'alice@example.com', 28);
INSERT INTO users (name, email, age) VALUES ('Bob', 'bob@example.com', 35);
INSERT INTO users (name, email, age) VALUES ('Carol', 'carol@example.com', 42);

INSERT INTO posts (author_id, title, body, published) VALUES (1, 'Hello World', 'This is my first post', TRUE);
INSERT INTO posts (author_id, title, body, published) VALUES (1, 'Second Post', 'Another great post', FALSE);
INSERT INTO posts (author_id, title, body, published) VALUES (2, 'Bob Blog', 'Blogging about databases', TRUE);

-- ============================================================================
-- Part 3: Query Data
-- ============================================================================

-- Select all users
SELECT * FROM users;

-- Select specific columns
SELECT name, email FROM users;

-- Filter with WHERE
SELECT name, age FROM users WHERE age > 30;

-- Order results
SELECT name, age FROM users ORDER BY age DESC;

-- Limit results
SELECT name FROM users LIMIT 2;

-- ============================================================================
-- Part 4: Aggregate Functions
-- ============================================================================

-- Count rows
SELECT COUNT(*) AS total_users FROM users;

-- Average
SELECT AVG(age) AS average_age FROM users;

-- Min and Max
SELECT MIN(age) AS youngest, MAX(age) AS oldest FROM users;

-- Sum
SELECT SUM(age) AS total_age FROM users;

-- ============================================================================
-- Part 5: Update Data
-- ============================================================================

-- Update a single row
UPDATE users SET active = FALSE WHERE name = 'Carol';

-- Update multiple rows
UPDATE posts SET published = TRUE WHERE author_id = 1;

-- Verify updates
SELECT name, active FROM users;
SELECT title, published FROM posts;

-- ============================================================================
-- Part 6: Delete Data
-- ============================================================================

-- Delete specific row
DELETE FROM posts WHERE id = 2;

-- Verify deletion
SELECT id, title FROM posts;

-- ============================================================================
-- Part 7: Transactions
-- ============================================================================

-- Begin transaction
BEGIN TRANSACTION;

-- Make changes
INSERT INTO users (name, email, age) VALUES ('Dave', 'dave@example.com', 30);
UPDATE users SET age = 29 WHERE name = 'Alice';

-- Commit transaction
COMMIT;

-- Verify changes
SELECT name, age FROM users;

-- ============================================================================
-- Part 8: String Functions
-- ============================================================================

-- UPPER and LOWER
SELECT UPPER(name) AS upper_name, LOWER(email) AS lower_email FROM users;

-- LENGTH
SELECT name, LENGTH(name) AS name_length FROM users;

-- SUBSTR
SELECT name, SUBSTR(email, 1, 5) AS email_prefix FROM users;

-- ============================================================================
-- Part 9: CASE Expressions
-- ============================================================================

SELECT
    name,
    age,
    CASE
        WHEN age < 30 THEN 'Young'
        WHEN age < 40 THEN 'Middle'
        ELSE 'Senior'
    END AS age_group
FROM users;

-- ============================================================================
-- Part 10: NULL Handling
-- ============================================================================

-- Insert row with NULL
INSERT INTO posts (author_id, title, body) VALUES (3, 'Draft Post', NULL);

-- Check for NULL
SELECT title FROM posts WHERE body IS NULL;

-- COALESCE for default values
SELECT title, COALESCE(body, '[No content]') AS content FROM posts;

-- ============================================================================
-- Part 11: Indexes
-- ============================================================================

-- Create index
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_posts_author ON posts(author_id);

-- ============================================================================
-- Part 12: Cleanup
-- ============================================================================

-- Drop tables
DROP TABLE IF EXISTS posts;
DROP TABLE IF EXISTS users;

-- ============================================================================
-- End of Simple Tutorial
-- ============================================================================
