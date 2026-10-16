"""Forty-two harder questions, used only for the model-written ("natural mistakes") study.

The first 40 questions were answered correctly about 94% of the time by the
five models, which left too few real mistakes to compare juries. These
questions target the kinds of slips the paper itself highlights ("cases that
are easy to get subtly wrong, for example duplicates versus COUNT(DISTINCT)"):

  * NULLs inside NOT IN, and NULL-safe comparisons
  * join fan-out (joining two one-to-many tables and counting both)
  * integer division in SQLite (1/4 = 0)
  * ties (top-N, "most recent", "second highest")
  * empty groups that must still appear with 0
  * "for all" conditions that are vacuously true for empty sets
  * filters placed in WHERE that silently turn a LEFT JOIN into an INNER JOIN

Every question states exactly which columns to return and how to treat
missing values, so a "wrong" label means a real mistake, not a different
reading of the question. Each question also carries one typical wrong
answer ("trap"); the test suite checks that the ten databases really tell
the trap apart from the reference, so the trap is detectable.

These questions are NOT used for planted bugs, so data/candidates.jsonl and
all earlier votes are unchanged.
"""

from __future__ import annotations

from .task_def import Task

HARD = "hard_traps"

HARD_TASKS: list[Task] = [
    Task("X01", HARD, "List the names of categories that contain no products. Return only the category name.",
         "SELECT c.name FROM Categories c WHERE NOT EXISTS "
         "(SELECT 1 FROM Products p WHERE p.category_id = c.category_id)",
         planted=(("SELECT name FROM Categories WHERE category_id NOT IN (SELECT category_id FROM Products)",
                   "null_in_not_in"),)),
    Task("X02", HARD, "For each category, show the category name and the total revenue from items in orders "
                      "with status 'completed', where revenue is quantity times the product's price. Include "
                      "every category, showing 0 for categories with no such revenue.",
         "SELECT c.name, COALESCE(SUM(CASE WHEN o.status = 'completed' THEN oi.quantity * p.price END), 0) "
         "FROM Categories c LEFT JOIN Products p ON p.category_id = c.category_id "
         "LEFT JOIN Order_Items oi ON oi.product_id = p.product_id "
         "LEFT JOIN Orders o ON o.order_id = oi.order_id GROUP BY c.category_id, c.name",
         planted=(("SELECT c.name, COALESCE(SUM(oi.quantity * p.price), 0) FROM Categories c "
                   "LEFT JOIN Products p ON p.category_id = c.category_id "
                   "LEFT JOIN Order_Items oi ON oi.product_id = p.product_id "
                   "LEFT JOIN Orders o ON o.order_id = oi.order_id WHERE o.status = 'completed' "
                   "GROUP BY c.category_id, c.name", "where_kills_left_join"),)),
    Task("X03", HARD, "What percentage of all orders have status 'cancelled'? Return one number between 0 and "
                      "100 (for example 23.5), not rounded.",
         "SELECT 100.0 * SUM(status = 'cancelled') / COUNT(*) FROM Orders",
         planted=(("SELECT SUM(status = 'cancelled') * 100 / COUNT(*) FROM Orders", "integer_division"),)),
    Task("X04", HARD, "For each location, show the location and the percentage of users in that location who "
                      "are active (a number from 0 to 100, not rounded). Ignore users whose location is unknown.",
         "SELECT location, 100.0 * SUM(is_active) / COUNT(*) FROM Users WHERE location IS NOT NULL "
         "GROUP BY location",
         planted=(("SELECT location, SUM(is_active) * 100 / COUNT(*) FROM Users WHERE location IS NOT NULL "
                   "GROUP BY location", "integer_division"),)),
    Task("X05", HARD, "What is the second highest distinct product price? If several products share the highest "
                      "price, the answer is the next lower price.",
         "SELECT MAX(price) FROM Products WHERE price < (SELECT MAX(price) FROM Products)",
         planted=(("SELECT price FROM Products ORDER BY price DESC LIMIT 1 OFFSET 1", "ties_ignored"),)),
    Task("X06", HARD, "List the usernames of users who have bought the same product in at least two different "
                      "orders (orders of any status). Show each username once.",
         "SELECT u.username FROM Users u WHERE EXISTS (SELECT 1 FROM Orders o JOIN Order_Items oi "
         "ON oi.order_id = o.order_id WHERE o.user_id = u.user_id GROUP BY oi.product_id "
         "HAVING COUNT(DISTINCT o.order_id) >= 2)",
         planted=(("SELECT DISTINCT u.username FROM Users u JOIN Orders o ON o.user_id = u.user_id "
                   "JOIN Order_Items oi ON oi.order_id = o.order_id GROUP BY u.user_id, oi.product_id "
                   "HAVING COUNT(*) >= 2", "duplicates_counted"),)),
    Task("X07", HARD, "What is the average number of orders per user, counting users who never ordered as "
                      "having 0 orders? Return one number, not rounded.",
         "SELECT 1.0 * (SELECT COUNT(*) FROM Orders) / (SELECT COUNT(*) FROM Users)",
         planted=(("SELECT AVG(n) FROM (SELECT COUNT(*) AS n FROM Orders GROUP BY user_id)",
                   "zero_rows_dropped"),)),
    Task("X08", HARD, "List the usernames of users who have placed at least one order and none of whose orders "
                      "has status 'cancelled'.",
         "SELECT u.username FROM Users u JOIN Orders o ON o.user_id = u.user_id "
         "GROUP BY u.user_id, u.username HAVING SUM(o.status = 'cancelled') = 0",
         planted=(("SELECT username FROM Users u WHERE NOT EXISTS (SELECT 1 FROM Orders o "
                   "WHERE o.user_id = u.user_id AND o.status = 'cancelled')", "vacuous_truth"),)),
    Task("X09", HARD, "For each category, show the category name and the name of the product with the highest "
                      "total quantity ordered (orders of any status). If several products tie, show all of them. "
                      "Skip categories whose products were never ordered.",
         "WITH q AS (SELECT p.category_id, p.name, SUM(oi.quantity) AS qty FROM Products p "
         "JOIN Order_Items oi ON oi.product_id = p.product_id WHERE p.category_id IS NOT NULL "
         "GROUP BY p.product_id) SELECT c.name, q.name FROM q JOIN Categories c "
         "ON c.category_id = q.category_id WHERE q.qty = (SELECT MAX(q2.qty) FROM q q2 "
         "WHERE q2.category_id = q.category_id)",
         planted=(("WITH q AS (SELECT p.category_id, p.name, COUNT(*) AS qty FROM Products p "
                   "JOIN Order_Items oi ON oi.product_id = p.product_id WHERE p.category_id IS NOT NULL "
                   "GROUP BY p.product_id) SELECT c.name, q.name FROM q JOIN Categories c "
                   "ON c.category_id = q.category_id WHERE q.qty = (SELECT MAX(q2.qty) FROM q q2 "
                   "WHERE q2.category_id = q.category_id)", "count_not_sum"),)),
    Task("X10", HARD, "Among users who have at least one completed order, list the usernames of those whose "
                      "total spending on completed orders is higher than the average of that total over these "
                      "users.",
         "WITH s AS (SELECT user_id, SUM(total_amount) AS t FROM Orders WHERE status = 'completed' "
         "GROUP BY user_id) SELECT u.username FROM Users u JOIN s ON s.user_id = u.user_id "
         "WHERE s.t > (SELECT AVG(t) FROM s)",
         planted=(("SELECT u.username FROM Users u JOIN Orders o ON o.user_id = u.user_id "
                   "WHERE o.status = 'completed' GROUP BY u.user_id, u.username HAVING SUM(o.total_amount) > "
                   "(SELECT AVG(total_amount) FROM Orders WHERE status = 'completed')",
                   "wrong_average"),)),
    Task("X11", HARD, "List the usernames of users who have at least one order with status 'completed' but have "
                      "never written a review.",
         "SELECT u.username FROM Users u WHERE EXISTS (SELECT 1 FROM Orders o WHERE o.user_id = u.user_id "
         "AND o.status = 'completed') AND NOT EXISTS (SELECT 1 FROM Reviews r WHERE r.user_id = u.user_id)",
         planted=(("SELECT DISTINCT u.username FROM Users u JOIN Orders o ON o.user_id = u.user_id "
                   "LEFT JOIN Reviews r ON r.user_id = u.user_id WHERE o.status = 'completed' "
                   "OR r.review_id IS NULL", "wrong_logic"),)),
    Task("X12", HARD, "List the names of products that have been ordered at least once (any status) but that "
                      "were never reviewed by any of the users who ordered them.",
         "SELECT p.name FROM Products p WHERE EXISTS (SELECT 1 FROM Order_Items oi "
         "WHERE oi.product_id = p.product_id) AND NOT EXISTS (SELECT 1 FROM Order_Items oi "
         "JOIN Orders o ON o.order_id = oi.order_id JOIN Reviews r ON r.user_id = o.user_id "
         "AND r.product_id = p.product_id WHERE oi.product_id = p.product_id)",
         planted=(("SELECT p.name FROM Products p WHERE EXISTS (SELECT 1 FROM Order_Items oi "
                   "WHERE oi.product_id = p.product_id) AND NOT EXISTS (SELECT 1 FROM Reviews r "
                   "WHERE r.product_id = p.product_id)", "wrong_condition"),)),
    Task("X13", HARD, "For each user, show the username, the number of orders they placed, and the number of "
                      "reviews they wrote. Include every user, showing 0 where they have none.",
         "SELECT u.username, (SELECT COUNT(*) FROM Orders o WHERE o.user_id = u.user_id), "
         "(SELECT COUNT(*) FROM Reviews r WHERE r.user_id = u.user_id) FROM Users u",
         planted=(("SELECT u.username, COUNT(o.order_id), COUNT(r.review_id) FROM Users u "
                   "LEFT JOIN Orders o ON o.user_id = u.user_id LEFT JOIN Reviews r ON r.user_id = u.user_id "
                   "GROUP BY u.user_id, u.username", "join_fan_out"),)),
    Task("X14", HARD, "For each product, show the product name, the total quantity ordered (any status; 0 if "
                      "never ordered) and its number of reviews (0 if none).",
         "SELECT p.name, (SELECT COALESCE(SUM(oi.quantity), 0) FROM Order_Items oi "
         "WHERE oi.product_id = p.product_id), (SELECT COUNT(*) FROM Reviews r "
         "WHERE r.product_id = p.product_id) FROM Products p",
         planted=(("SELECT p.name, COALESCE(SUM(oi.quantity), 0), COUNT(r.review_id) FROM Products p "
                   "LEFT JOIN Order_Items oi ON oi.product_id = p.product_id "
                   "LEFT JOIN Reviews r ON r.product_id = p.product_id GROUP BY p.product_id, p.name",
                   "join_fan_out"),)),
    Task("X15", HARD, "For each user who has placed orders, show the username and the number of days between "
                      "their first and their most recent order date (0 if all their orders are on one day).",
         "SELECT u.username, CAST(julianday(MAX(o.order_date)) - julianday(MIN(o.order_date)) AS INTEGER) "
         "FROM Users u JOIN Orders o ON o.user_id = u.user_id GROUP BY u.user_id, u.username",
         planted=(("SELECT u.username, COUNT(DISTINCT o.order_date) - 1 FROM Users u "
                   "JOIN Orders o ON o.user_id = u.user_id GROUP BY u.user_id, u.username",
                   "wrong_measure"),)),
    Task("X16", HARD, "For each month of 2024, show the month as 'YYYY-MM' and the number of different users who "
                      "placed at least one order (any status) in that month. Skip months with no orders.",
         "SELECT strftime('%Y-%m', order_date), COUNT(DISTINCT user_id) FROM Orders "
         "WHERE order_date >= '2024-01-01' AND order_date < '2025-01-01' GROUP BY strftime('%Y-%m', order_date)",
         planted=(("SELECT strftime('%Y-%m', order_date), COUNT(user_id) FROM Orders "
                   "WHERE order_date >= '2024-01-01' AND order_date < '2025-01-01' "
                   "GROUP BY strftime('%Y-%m', order_date)", "distinct_counting"),)),
    Task("X17", HARD, "Among users who have placed at least one order, what percentage have placed two or more "
                      "orders? Return one number from 0 to 100, not rounded.",
         "SELECT 100.0 * SUM(n >= 2) / COUNT(*) FROM (SELECT COUNT(*) AS n FROM Orders GROUP BY user_id)",
         planted=(("SELECT 100.0 * (SELECT COUNT(*) FROM (SELECT user_id FROM Orders GROUP BY user_id "
                   "HAVING COUNT(*) >= 2)) / (SELECT COUNT(*) FROM Users)", "wrong_denominator"),)),
    Task("X18", HARD, "List the names of products whose average review rating is higher than the average rating "
                      "over all reviews. Only consider products with at least one review.",
         "SELECT p.name FROM Products p JOIN Reviews r ON r.product_id = p.product_id "
         "GROUP BY p.product_id, p.name HAVING AVG(r.rating) > (SELECT AVG(rating) FROM Reviews)",
         planted=(("SELECT p.name FROM Products p JOIN Reviews r ON r.product_id = p.product_id "
                   "GROUP BY p.product_id, p.name HAVING AVG(r.rating) > (SELECT AVG(a) FROM "
                   "(SELECT AVG(rating) AS a FROM Reviews GROUP BY product_id))", "average_of_averages"),)),
    Task("X19", HARD, "List the names of categories that have at least one product and in which every product "
                      "has a stock quantity greater than zero.",
         "SELECT c.name FROM Categories c JOIN Products p ON p.category_id = c.category_id "
         "GROUP BY c.category_id, c.name HAVING MIN(p.stock_quantity) > 0",
         planted=(("SELECT c.name FROM Categories c WHERE NOT EXISTS (SELECT 1 FROM Products p "
                   "WHERE p.category_id = c.category_id AND p.stock_quantity = 0)", "vacuous_truth"),)),
    Task("X20", HARD, "List the usernames of users who reviewed at least one product that they never ordered "
                      "(in any order of any status). Show each username once.",
         "SELECT u.username FROM Users u WHERE EXISTS (SELECT 1 FROM Reviews r WHERE r.user_id = u.user_id "
         "AND NOT EXISTS (SELECT 1 FROM Orders o JOIN Order_Items oi ON oi.order_id = o.order_id "
         "WHERE o.user_id = u.user_id AND oi.product_id = r.product_id))",
         planted=(("SELECT DISTINCT u.username FROM Users u JOIN Reviews r ON r.user_id = u.user_id "
                   "WHERE r.product_id NOT IN (SELECT oi.product_id FROM Order_Items oi)",
                   "wrong_scope"),)),
    Task("X21", HARD, "For each order, show the order id and the number of days since the same user's previous "
                      "order (NULL for a user's first order). Order each user's orders by order date, breaking "
                      "ties by order id.",
         "SELECT order_id, CAST(julianday(order_date) - julianday(LAG(order_date) OVER "
         "(PARTITION BY user_id ORDER BY order_date, order_id)) AS INTEGER) FROM Orders",
         planted=(("SELECT order_id, CAST(julianday(order_date) - julianday(LAG(order_date) OVER "
                   "(ORDER BY order_date, order_id)) AS INTEGER) FROM Orders", "window_partition"),)),
    Task("X22", HARD, "List the usernames of users whose most recent order has status 'cancelled'. The most "
                      "recent order is the one with the latest order date; if several share that date, it is "
                      "the one with the highest order id.",
         "SELECT u.username FROM Users u JOIN Orders o ON o.user_id = u.user_id WHERE o.status = 'cancelled' "
         "AND NOT EXISTS (SELECT 1 FROM Orders o2 WHERE o2.user_id = o.user_id AND (o2.order_date > "
         "o.order_date OR (o2.order_date = o.order_date AND o2.order_id > o.order_id)))",
         planted=(("SELECT DISTINCT u.username FROM Users u JOIN Orders o ON o.user_id = u.user_id "
                   "WHERE o.status = 'cancelled' AND o.order_date = (SELECT MAX(o2.order_date) FROM Orders o2 "
                   "WHERE o2.user_id = u.user_id)", "ties_ignored"),)),
    Task("X23", HARD, "Show the id of the order (or orders, if several tie) that contains the largest number of "
                      "different products.",
         "WITH n AS (SELECT order_id, COUNT(DISTINCT product_id) AS k FROM Order_Items GROUP BY order_id) "
         "SELECT order_id FROM n WHERE k = (SELECT MAX(k) FROM n)",
         planted=(("SELECT order_id FROM Order_Items GROUP BY order_id ORDER BY COUNT(DISTINCT product_id) DESC "
                   "LIMIT 1", "ties_ignored"),)),
    Task("X24", HARD, "List every pair of product names that appear together in at least two different orders. "
                      "Show each pair once, as two columns, with the alphabetically smaller name first.",
         "SELECT p1.name, p2.name FROM Order_Items a JOIN Order_Items b ON a.order_id = b.order_id "
         "JOIN Products p1 ON p1.product_id = a.product_id JOIN Products p2 ON p2.product_id = b.product_id "
         "WHERE p1.name < p2.name GROUP BY p1.name, p2.name HAVING COUNT(DISTINCT a.order_id) >= 2",
         planted=(("SELECT p1.name, p2.name FROM Order_Items a JOIN Order_Items b ON a.order_id = b.order_id "
                   "JOIN Products p1 ON p1.product_id = a.product_id JOIN Products p2 "
                   "ON p2.product_id = b.product_id WHERE p1.name < p2.name GROUP BY p1.name, p2.name "
                   "HAVING COUNT(*) >= 2", "duplicates_counted"),)),
    Task("X25", HARD, "List the usernames of users whose age is greater than the average age of users in the "
                      "same location. Ignore users whose age or location is unknown.",
         "SELECT u.username FROM Users u WHERE u.age IS NOT NULL AND u.location IS NOT NULL AND u.age > "
         "(SELECT AVG(u2.age) FROM Users u2 WHERE u2.location = u.location)",
         planted=(("SELECT u.username FROM Users u WHERE u.location IS NOT NULL AND u.age > "
                   "(SELECT AVG(COALESCE(u2.age, 0)) FROM Users u2 WHERE u2.location = u.location)",
                   "null_handling"),)),
    Task("X26", HARD, "For each category that has at least one completed sale, show the category name and its "
                      "share of the total completed revenue as a percentage (0 to 100, not rounded). Revenue is "
                      "quantity times the product's price, counted over items in orders with status "
                      "'completed'. Ignore products without a category in both the category totals and the "
                      "overall total.",
         "WITH r AS (SELECT p.category_id, SUM(oi.quantity * p.price) AS rev FROM Order_Items oi "
         "JOIN Orders o ON o.order_id = oi.order_id JOIN Products p ON p.product_id = oi.product_id "
         "WHERE o.status = 'completed' AND p.category_id IS NOT NULL GROUP BY p.category_id) "
         "SELECT c.name, 100.0 * r.rev / (SELECT SUM(rev) FROM r) FROM r JOIN Categories c "
         "ON c.category_id = r.category_id",
         planted=(("WITH r AS (SELECT p.category_id, SUM(oi.quantity * p.price) AS rev FROM Order_Items oi "
                   "JOIN Orders o ON o.order_id = oi.order_id JOIN Products p ON p.product_id = oi.product_id "
                   "WHERE o.status = 'completed' AND p.category_id IS NOT NULL GROUP BY p.category_id) "
                   "SELECT c.name, 100.0 * r.rev / (SELECT SUM(total_amount) FROM Orders "
                   "WHERE status = 'completed') FROM r JOIN Categories c ON c.category_id = r.category_id",
                   "wrong_denominator"),)),
    Task("X27", HARD, "For each number of orders a user can have (0, 1, 2, and so on), show that number and how "
                      "many users have placed exactly that many orders. Include users who never ordered "
                      "(as 0 orders).",
         "SELECT n, COUNT(*) FROM (SELECT u.user_id, COUNT(o.order_id) AS n FROM Users u "
         "LEFT JOIN Orders o ON o.user_id = u.user_id GROUP BY u.user_id) GROUP BY n",
         planted=(("SELECT n, COUNT(*) FROM (SELECT user_id, COUNT(*) AS n FROM Orders GROUP BY user_id) "
                   "GROUP BY n", "zero_rows_dropped"),)),
    Task("X28", HARD, "How many orders have status 'shipped' or 'completed' and no shipping address (the "
                      "address is missing)?",
         "SELECT COUNT(*) FROM Orders WHERE status IN ('shipped', 'completed') AND shipping_address IS NULL",
         planted=(("SELECT COUNT(*) FROM Orders WHERE status = 'shipped' OR status = 'completed' "
                   "AND shipping_address IS NULL", "operator_precedence"),)),
    Task("X29", HARD, "For each age group, show the group name and the number of users in it. The groups are "
                      "'under 18' (age below 18), '18-34', '35-54', '55 and over', and 'unknown' for users whose "
                      "age is missing. Skip groups with no users.",
         "SELECT CASE WHEN age IS NULL THEN 'unknown' WHEN age < 18 THEN 'under 18' WHEN age <= 34 THEN '18-34' "
         "WHEN age <= 54 THEN '35-54' ELSE '55 and over' END AS g, COUNT(*) FROM Users GROUP BY g",
         planted=(("SELECT CASE WHEN age < 18 THEN 'under 18' WHEN age <= 34 THEN '18-34' WHEN age <= 54 "
                   "THEN '35-54' ELSE '55 and over' END AS g, COUNT(*) FROM Users GROUP BY g",
                   "null_handling"),)),
    Task("X30", HARD, "List the usernames of users whose total spending on completed orders placed in 2024 is "
                      "greater than their total spending on completed orders placed in 2023. A year with no "
                      "completed orders counts as 0.",
         "SELECT u.username FROM Users u JOIN Orders o ON o.user_id = u.user_id WHERE o.status = 'completed' "
         "GROUP BY u.user_id, u.username HAVING SUM(CASE WHEN o.order_date LIKE '2024%' THEN o.total_amount "
         "ELSE 0 END) > SUM(CASE WHEN o.order_date LIKE '2023%' THEN o.total_amount ELSE 0 END)",
         planted=(("SELECT u.username FROM Users u JOIN Orders a ON a.user_id = u.user_id AND a.status = "
                   "'completed' AND a.order_date LIKE '2024%' JOIN Orders b ON b.user_id = u.user_id AND "
                   "b.status = 'completed' AND b.order_date LIKE '2023%' GROUP BY u.user_id, u.username "
                   "HAVING SUM(a.total_amount) > SUM(b.total_amount)", "zero_rows_dropped"),)),
    Task("X31", HARD, "What is the average number of items per order, where the items of an order are the sum "
                      "of the quantities of its order lines? Return one number, not rounded.",
         "SELECT 1.0 * SUM(quantity) / COUNT(DISTINCT order_id) FROM Order_Items",
         planted=(("SELECT SUM(quantity) / COUNT(DISTINCT order_id) FROM Order_Items", "integer_division"),)),
    Task("X32", HARD, "For each user who has placed orders, show the username and the status of their most "
                      "recent order. The most recent order is the one with the latest order date; if several "
                      "share that date, it is the one with the highest order id.",
         "SELECT u.username, o.status FROM Users u JOIN Orders o ON o.user_id = u.user_id WHERE NOT EXISTS "
         "(SELECT 1 FROM Orders o2 WHERE o2.user_id = o.user_id AND (o2.order_date > o.order_date OR "
         "(o2.order_date = o.order_date AND o2.order_id > o.order_id)))",
         planted=(("SELECT u.username, o.status FROM Users u JOIN Orders o ON o.user_id = u.user_id "
                   "WHERE o.order_date = (SELECT MAX(o2.order_date) FROM Orders o2 WHERE o2.user_id = u.user_id)",
                   "ties_ignored"),)),
    Task("X33", HARD, "List the names of products that were ordered in 2023 but not in 2024 (orders of any "
                      "status).",
         "SELECT p.name FROM Products p WHERE EXISTS (SELECT 1 FROM Order_Items oi JOIN Orders o "
         "ON o.order_id = oi.order_id WHERE oi.product_id = p.product_id AND o.order_date LIKE '2023%') "
         "AND NOT EXISTS (SELECT 1 FROM Order_Items oi JOIN Orders o ON o.order_id = oi.order_id "
         "WHERE oi.product_id = p.product_id AND o.order_date LIKE '2024%')",
         planted=(("SELECT DISTINCT p.name FROM Products p JOIN Order_Items oi ON oi.product_id = p.product_id "
                   "JOIN Orders o ON o.order_id = oi.order_id WHERE o.order_date LIKE '2023%' "
                   "AND o.order_date NOT LIKE '2024%'", "wrong_scope"),)),
    Task("X34", HARD, "For each category, show the category name and the number of different users who reviewed "
                      "at least one of its products. Include every category, showing 0 if none.",
         "SELECT c.name, COUNT(DISTINCT r.user_id) FROM Categories c LEFT JOIN Products p "
         "ON p.category_id = c.category_id LEFT JOIN Reviews r ON r.product_id = p.product_id "
         "GROUP BY c.category_id, c.name",
         planted=(("SELECT c.name, COUNT(r.user_id) FROM Categories c LEFT JOIN Products p "
                   "ON p.category_id = c.category_id LEFT JOIN Reviews r ON r.product_id = p.product_id "
                   "GROUP BY c.category_id, c.name", "distinct_counting"),)),
    Task("X35", HARD, "Show the username of the user who wrote the most reviews. If several users tie, show all "
                      "of them.",
         "WITH n AS (SELECT user_id, COUNT(*) AS k FROM Reviews GROUP BY user_id) SELECT u.username "
         "FROM Users u JOIN n ON n.user_id = u.user_id WHERE n.k = (SELECT MAX(k) FROM n)",
         planted=(("SELECT u.username FROM Users u JOIN Reviews r ON r.user_id = u.user_id "
                   "GROUP BY u.user_id, u.username ORDER BY COUNT(*) DESC LIMIT 1", "ties_ignored"),)),
    Task("X36", HARD, "List the usernames of users who placed two or more orders on the same day. Show each "
                      "username once.",
         "SELECT u.username FROM Users u WHERE EXISTS (SELECT 1 FROM Orders o WHERE o.user_id = u.user_id "
         "GROUP BY o.order_date HAVING COUNT(*) >= 2)",
         planted=(("SELECT u.username FROM Users u JOIN Orders o ON o.user_id = u.user_id "
                   "GROUP BY u.user_id, u.username HAVING COUNT(*) >= 2", "missing_grouping"),)),
    Task("X37", HARD, "For each location, show the location and the number of orders placed by users from that "
                      "location. Group users whose location is unknown under the label 'Unknown'. Include "
                      "locations whose users placed no orders, showing 0.",
         "SELECT COALESCE(u.location, 'Unknown'), COUNT(o.order_id) FROM Users u LEFT JOIN Orders o "
         "ON o.user_id = u.user_id GROUP BY COALESCE(u.location, 'Unknown')",
         planted=(("SELECT COALESCE(u.location, 'Unknown'), COUNT(*) FROM Users u LEFT JOIN Orders o "
                   "ON o.user_id = u.user_id GROUP BY COALESCE(u.location, 'Unknown')", "count_star"),)),
    Task("X38", HARD, "List the names of products whose price equals the highest price in their own category. "
                      "Ignore products without a category. If several products in a category share the "
                      "highest price, list all of them.",
         "SELECT p.name FROM Products p WHERE p.category_id IS NOT NULL AND p.price = "
         "(SELECT MAX(p2.price) FROM Products p2 WHERE p2.category_id = p.category_id)",
         planted=(("SELECT p.name FROM Products p WHERE p.product_id IN (SELECT (SELECT p2.product_id "
                   "FROM Products p2 WHERE p2.category_id = c.category_id ORDER BY p2.price DESC LIMIT 1) "
                   "FROM Categories c)", "ties_ignored"),)),
    Task("X39", HARD, "For each user who has written reviews, show the username and the number of different "
                      "products they reviewed that they had also ordered (orders of any status), showing 0 if "
                      "none.",
         "SELECT u.username, (SELECT COUNT(DISTINCT r.product_id) FROM Reviews r WHERE r.user_id = u.user_id "
         "AND EXISTS (SELECT 1 FROM Orders o JOIN Order_Items oi ON oi.order_id = o.order_id "
         "WHERE o.user_id = u.user_id AND oi.product_id = r.product_id)) FROM Users u "
         "WHERE EXISTS (SELECT 1 FROM Reviews r WHERE r.user_id = u.user_id)",
         planted=(("SELECT u.username, COUNT(DISTINCT r.product_id) FROM Users u JOIN Reviews r "
                   "ON r.user_id = u.user_id JOIN Orders o ON o.user_id = u.user_id JOIN Order_Items oi "
                   "ON oi.order_id = o.order_id AND oi.product_id = r.product_id GROUP BY u.user_id, u.username",
                   "zero_rows_dropped"),)),
    Task("X40", HARD, "List the names of products that have never been ordered by any user located in "
                      "'Lubbock' (orders of any status). Products that were never ordered at all also count.",
         "SELECT p.name FROM Products p WHERE NOT EXISTS (SELECT 1 FROM Order_Items oi JOIN Orders o "
         "ON o.order_id = oi.order_id JOIN Users u ON u.user_id = o.user_id WHERE oi.product_id = p.product_id "
         "AND u.location = 'Lubbock')",
         planted=(("SELECT DISTINCT p.name FROM Products p JOIN Order_Items oi ON oi.product_id = p.product_id "
                   "JOIN Orders o ON o.order_id = oi.order_id JOIN Users u ON u.user_id = o.user_id "
                   "WHERE u.location <> 'Lubbock'", "wrong_scope"),)),
    Task("X41", HARD, "For each user who has placed orders, show the username and the average total amount of "
                      "their orders with status 'completed'. Show NULL for users who have orders but none of "
                      "them completed.",
         "SELECT u.username, AVG(CASE WHEN o.status = 'completed' THEN o.total_amount END) FROM Users u "
         "JOIN Orders o ON o.user_id = u.user_id GROUP BY u.user_id, u.username",
         planted=(("SELECT u.username, AVG(o.total_amount) FROM Users u JOIN Orders o ON o.user_id = u.user_id "
                   "WHERE o.status = 'completed' GROUP BY u.user_id, u.username", "zero_rows_dropped"),)),
    Task("X42", HARD, "How many different products have been ordered by at least one inactive user (is_active "
                      "= 0), in orders of any status? Return one number.",
         "SELECT COUNT(DISTINCT oi.product_id) FROM Order_Items oi JOIN Orders o ON o.order_id = oi.order_id "
         "JOIN Users u ON u.user_id = o.user_id WHERE u.is_active = 0",
         planted=(("SELECT COUNT(*) FROM Order_Items oi JOIN Orders o ON o.order_id = oi.order_id "
                   "JOIN Users u ON u.user_id = o.user_id WHERE u.is_active = 0", "distinct_counting"),)),
]
