-- Silica Quickstart Example
-- Demonstrates basic SQL operations
-- Run with: silica quickstart.db < quickstart.sql

-- Create a table
CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    price REAL,
    in_stock BOOLEAN DEFAULT TRUE
);

-- Insert data
INSERT INTO products (id, name, price, in_stock) VALUES (1, 'Laptop', 999.99, TRUE);
INSERT INTO products (id, name, price, in_stock) VALUES (2, 'Mouse', 29.99, TRUE);
INSERT INTO products (id, name, price, in_stock) VALUES (3, 'Keyboard', 79.99, FALSE);

-- Query all products
SELECT * FROM products;

-- Filter results
SELECT name, price FROM products WHERE in_stock = TRUE;

-- Update a row
UPDATE products SET in_stock = TRUE WHERE id = 3;

-- Aggregate function
SELECT COUNT(*) AS total_products, AVG(price) AS avg_price FROM products;

-- Create an index
CREATE INDEX idx_products_price ON products(price);

-- Cleanup
DROP TABLE products;
