"""Forty text-to-SQL tasks over the e-commerce schema.

Eight tasks in each of the five tiers described in "Vibe Coding on Trial":
basic retrieval, aggregate analysis, multi-table joins and grouping,
filtered aggregation with HAVING, and subqueries / advanced constructs.

Each task has
  * question  : the natural-language request a judge sees
  * gold      : the human-written reference query
  * rewrites  : different but equivalent correct queries
  * planted   : hand-written subtle bugs that automatic mutation
                cannot produce, each tagged with a bug type

Every label is re-derived by execution (see build_candidates.py); nothing
here is trusted without running it on all ten instances.
"""

from __future__ import annotations

from .task_def import Task


T1, T2, T3, T4, T5 = (
    "basic_retrieval",
    "aggregate_analysis",
    "joins_grouping",
    "having_filters",
    "subqueries_advanced",
)

TASKS: list[Task] = [
    # ------------------------------------------------------------------ T1
    Task("B1", T1, "List the username and email of every active user.",
         "SELECT username, email FROM Users WHERE is_active = 1",
         ("SELECT u.username, u.email FROM Users AS u WHERE u.is_active <> 0",),
         (("SELECT username, email FROM Users", "missing_filter"),)),
    Task("B2", T1, "List the name and price of every product that costs more than 49.99.",
         "SELECT name, price FROM Products WHERE price > 49.99",
         ("SELECT name, price FROM Products WHERE NOT price <= 49.99",)),
    Task("B3", T1, "List the usernames of users whose location is unknown.",
         "SELECT username FROM Users WHERE location IS NULL",
         ("SELECT u.username FROM Users u WHERE u.location IS NULL ORDER BY u.username",)),
    Task("B4", T1, "List the ids of all orders placed in the year 2024.",
         "SELECT order_id FROM Orders WHERE order_date >= '2024-01-01' AND order_date < '2025-01-01'",
         ("SELECT order_id FROM Orders WHERE strftime('%Y', order_date) = '2024'",),
         (("SELECT order_id FROM Orders WHERE order_date > '2024-01-01' AND order_date < '2024-12-31'",
           "wrong_boundary"),)),
    Task("B5", T1, "List the names of products that are out of stock (stock quantity of zero).",
         "SELECT name FROM Products WHERE stock_quantity = 0",
         ("SELECT name FROM Products WHERE stock_quantity < 1",),
         (("SELECT name FROM Products WHERE stock_quantity = 0 AND description IS NOT NULL",
           "extra_filter"),)),
    Task("B6", T1, "List the usernames of users who are at least 18 years old.",
         "SELECT username FROM Users WHERE age >= 18",
         ("SELECT username FROM Users WHERE age > 17",),
         (("SELECT username FROM Users WHERE age >= 18 OR age IS NULL", "null_handling"),)),
    Task("B7", T1, "List the names of products that have no description.",
         "SELECT name FROM Products WHERE description IS NULL"),
    Task("B8", T1, "List the ids of reviews that gave a rating of 4 or 5.",
         "SELECT review_id FROM Reviews WHERE rating >= 4",
         ("SELECT review_id FROM Reviews WHERE rating IN (4, 5)",)),

    # ------------------------------------------------------------------ T2
    Task("A1", T2, "How many active users are there?",
         "SELECT COUNT(*) FROM Users WHERE is_active = 1",
         ("SELECT SUM(is_active) FROM Users",)),
    Task("A2", T2, "How many different users have placed at least one order?",
         "SELECT COUNT(DISTINCT user_id) FROM Orders",
         ("SELECT COUNT(*) FROM Users u WHERE EXISTS (SELECT 1 FROM Orders o WHERE o.user_id = u.user_id)",)),
    Task("A3", T2, "What is the average age of users? Ignore users whose age is unknown.",
         "SELECT AVG(age) FROM Users",
         ("SELECT SUM(age) * 1.0 / COUNT(age) FROM Users",),
         (("SELECT SUM(age) * 1.0 / COUNT(*) FROM Users", "null_handling"),
          ("SELECT AVG(COALESCE(age, 0)) FROM Users", "null_handling"))),
    Task("A4", T2, "For each category, show the category name and the number of products in it. "
                   "Include categories that have no products (show 0).",
         "SELECT c.name, COUNT(p.product_id) FROM Categories c "
         "LEFT JOIN Products p ON p.category_id = c.category_id GROUP BY c.category_id, c.name",
         ("SELECT c.name, (SELECT COUNT(*) FROM Products p WHERE p.category_id = c.category_id) FROM Categories c",)),
    Task("A5", T2, "What is the total revenue (sum of total_amount) of completed orders?",
         "SELECT SUM(total_amount) FROM Orders WHERE status = 'completed'",
         ("SELECT TOTAL(total_amount) FROM Orders WHERE status IN ('completed')",)),
    Task("A6", T2, "For each order status, show the status and the number of orders with that status.",
         "SELECT status, COUNT(*) FROM Orders GROUP BY status",
         ("SELECT status, COUNT(order_id) FROM Orders GROUP BY status ORDER BY status",)),
    Task("A7", T2, "What are the highest and the lowest product prices? Return the highest first.",
         "SELECT MAX(price), MIN(price) FROM Products",
         ("SELECT (SELECT price FROM Products ORDER BY price DESC LIMIT 1), "
          "(SELECT price FROM Products ORDER BY price ASC LIMIT 1)",)),
    Task("A8", T2, "For each product, show the product name and its number of reviews. "
                   "Include products with no reviews (show 0).",
         "SELECT p.name, COUNT(r.review_id) FROM Products p "
         "LEFT JOIN Reviews r ON r.product_id = p.product_id GROUP BY p.product_id, p.name",
         ("SELECT p.name, (SELECT COUNT(*) FROM Reviews r WHERE r.product_id = p.product_id) FROM Products p",)),

    # ------------------------------------------------------------------ T3
    Task("J1", T3, "For each user who has placed at least one order, show the username "
                   "and the number of orders they placed.",
         "SELECT u.username, COUNT(*) FROM Users u JOIN Orders o ON o.user_id = u.user_id "
         "GROUP BY u.user_id, u.username",
         ("SELECT u.username, x.n FROM Users u JOIN (SELECT user_id, COUNT(*) AS n FROM Orders "
          "GROUP BY user_id) x ON x.user_id = u.user_id",)),
    Task("J2", T3, "Show every user's username and the total amount they have spent across all "
                   "their orders. Include users who never ordered, showing 0 for them.",
         "SELECT u.username, COALESCE(SUM(o.total_amount), 0) FROM Users u "
         "LEFT JOIN Orders o ON o.user_id = u.user_id GROUP BY u.user_id, u.username",
         ("SELECT u.username, (SELECT TOTAL(o.total_amount) FROM Orders o WHERE o.user_id = u.user_id) "
          "FROM Users u",)),
    Task("J3", T3, "For each category, show the category name and the total quantity of items sold "
                   "in that category. Only include categories with at least one item sold.",
         "SELECT c.name, SUM(oi.quantity) FROM Categories c "
         "JOIN Products p ON p.category_id = c.category_id "
         "JOIN Order_Items oi ON oi.product_id = p.product_id GROUP BY c.category_id, c.name",
         (),
         (("SELECT c.name, COUNT(oi.order_item_id) FROM Categories c "
           "JOIN Products p ON p.category_id = c.category_id "
           "JOIN Order_Items oi ON oi.product_id = p.product_id GROUP BY c.category_id, c.name",
           "wrong_aggregate"),)),
    Task("J4", T3, "For each year, show the year and the number of orders placed in that year.",
         "SELECT strftime('%Y', order_date), COUNT(*) FROM Orders GROUP BY strftime('%Y', order_date)",
         ("SELECT substr(order_date, 1, 4) AS y, COUNT(order_id) FROM Orders GROUP BY y",)),
    Task("J5", T3, "List the names of products that have been ordered by at least one user located "
                   "in Lubbock. Show each product only once.",
         "SELECT DISTINCT p.name FROM Products p JOIN Order_Items oi ON oi.product_id = p.product_id "
         "JOIN Orders o ON o.order_id = oi.order_id JOIN Users u ON u.user_id = o.user_id "
         "WHERE u.location = 'Lubbock'",
         ("SELECT p.name FROM Products p WHERE p.product_id IN (SELECT oi.product_id FROM Order_Items oi "
          "JOIN Orders o ON o.order_id = oi.order_id JOIN Users u ON u.user_id = o.user_id "
          "WHERE u.location = 'Lubbock')",)),
    Task("J6", T3, "For each user who has placed orders, show the username and the number of "
                   "different products they have ordered.",
         "SELECT u.username, COUNT(DISTINCT oi.product_id) FROM Users u "
         "JOIN Orders o ON o.user_id = u.user_id JOIN Order_Items oi ON oi.order_id = o.order_id "
         "GROUP BY u.user_id, u.username"),
    Task("J7", T3, "Show the name of each product together with the name of its category. "
                   "Include products that have no category.",
         "SELECT p.name, c.name FROM Products p LEFT JOIN Categories c ON c.category_id = p.category_id",
         ("SELECT p.name, (SELECT c.name FROM Categories c WHERE c.category_id = p.category_id) FROM Products p",)),
    Task("J8", T3, "For each month of 2024, show the month number (01 to 12) and the total revenue "
                   "of completed orders placed in that month. Skip months with no such orders.",
         "SELECT strftime('%m', order_date), SUM(total_amount) FROM Orders "
         "WHERE status = 'completed' AND strftime('%Y', order_date) = '2024' "
         "GROUP BY strftime('%m', order_date)",
         ("SELECT substr(order_date, 6, 2) AS m, SUM(total_amount) FROM Orders "
          "WHERE status = 'completed' AND order_date BETWEEN '2024-01-01' AND '2024-12-31' GROUP BY m",)),

    # ------------------------------------------------------------------ T4
    Task("H1", T4, "List the usernames of users who have placed more than 3 orders.",
         "SELECT u.username FROM Users u JOIN Orders o ON o.user_id = u.user_id "
         "GROUP BY u.user_id, u.username HAVING COUNT(*) > 3",
         ("SELECT username FROM Users WHERE (SELECT COUNT(*) FROM Orders o "
          "WHERE o.user_id = Users.user_id) >= 4",)),
    Task("H2", T4, "List the names of categories whose average product price is above 50.",
         "SELECT c.name FROM Categories c JOIN Products p ON p.category_id = c.category_id "
         "GROUP BY c.category_id, c.name HAVING AVG(p.price) > 50"),
    Task("H3", T4, "List the names of products that have at least 2 reviews and an average "
                   "rating of at least 4.",
         "SELECT p.name FROM Products p JOIN Reviews r ON r.product_id = p.product_id "
         "GROUP BY p.product_id, p.name HAVING COUNT(*) >= 2 AND AVG(r.rating) >= 4"),
    Task("H4", T4, "List the usernames of users whose total spending on completed orders exceeds 500.",
         "SELECT u.username FROM Users u JOIN Orders o ON o.user_id = u.user_id "
         "WHERE o.status = 'completed' GROUP BY u.user_id, u.username HAVING SUM(o.total_amount) > 500"),
    Task("H5", T4, "List the names of products that appear in more than 3 different orders.",
         "SELECT p.name FROM Products p JOIN Order_Items oi ON oi.product_id = p.product_id "
         "GROUP BY p.product_id, p.name HAVING COUNT(DISTINCT oi.order_id) > 3"),
    Task("H6", T4, "For each location, show the location and its number of active users, "
                   "but only for locations with at least 3 active users. Ignore users with an "
                   "unknown location.",
         "SELECT location, COUNT(*) FROM Users WHERE is_active = 1 AND location IS NOT NULL "
         "GROUP BY location HAVING COUNT(*) >= 3",
         (),
         (("SELECT location, COUNT(*) FROM Users WHERE location IS NOT NULL "
           "GROUP BY location HAVING SUM(is_active) >= 3", "wrong_aggregate"),)),
    Task("H7", T4, "List the ids of users who have reviewed at least 2 different products.",
         "SELECT user_id FROM Reviews GROUP BY user_id HAVING COUNT(DISTINCT product_id) >= 2"),
    Task("H8", T4, "List the ids of orders that contain more than 2 line items.",
         "SELECT order_id FROM Order_Items GROUP BY order_id HAVING COUNT(*) > 2",
         (),
         (("SELECT order_id FROM Order_Items GROUP BY order_id HAVING SUM(quantity) > 2",
           "wrong_aggregate"),)),

    # ------------------------------------------------------------------ T5
    Task("S1", T5, "List the names of products priced above the average price of all products.",
         "SELECT name FROM Products WHERE price > (SELECT AVG(price) FROM Products)"),
    Task("S2", T5, "List the usernames of users who have never placed an order.",
         "SELECT username FROM Users u WHERE NOT EXISTS "
         "(SELECT 1 FROM Orders o WHERE o.user_id = u.user_id)",
         ("SELECT u.username FROM Users u LEFT JOIN Orders o ON o.user_id = u.user_id "
          "WHERE o.order_id IS NULL",)),
    Task("S3", T5, "List the names of products that have never been reviewed.",
         "SELECT name FROM Products WHERE product_id NOT IN (SELECT product_id FROM Reviews)",
         ("SELECT p.name FROM Products p WHERE NOT EXISTS "
          "(SELECT 1 FROM Reviews r WHERE r.product_id = p.product_id)",),
         (("SELECT p.name FROM Products p LEFT JOIN Reviews r ON r.product_id = p.product_id "
           "WHERE r.rating IS NOT NULL", "inverted_logic"),)),
    Task("S4", T5, "List the names of products that have never been ordered.",
         "SELECT p.name FROM Products p WHERE NOT EXISTS "
         "(SELECT 1 FROM Order_Items oi WHERE oi.product_id = p.product_id)",
         ("SELECT p.name FROM Products p LEFT JOIN Order_Items oi ON oi.product_id = p.product_id "
          "WHERE oi.order_item_id IS NULL",)),
    Task("S5", T5, "For each user who has placed orders, show the username and the date of their "
                   "most recent order.",
         "SELECT u.username, MAX(o.order_date) FROM Users u JOIN Orders o ON o.user_id = u.user_id "
         "GROUP BY u.user_id, u.username",
         ("SELECT u.username, (SELECT o.order_date FROM Orders o WHERE o.user_id = u.user_id "
          "ORDER BY o.order_date DESC LIMIT 1) AS d FROM Users u WHERE d IS NOT NULL",)),
    Task("S6", T5, "List the usernames of users who have bought at least one product from every "
                   "category that contains products.",
         "SELECT u.username FROM Users u WHERE NOT EXISTS ("
         "SELECT 1 FROM Categories c WHERE EXISTS (SELECT 1 FROM Products p0 WHERE p0.category_id = c.category_id) "
         "AND NOT EXISTS (SELECT 1 FROM Orders o JOIN Order_Items oi ON oi.order_id = o.order_id "
         "JOIN Products p ON p.product_id = oi.product_id "
         "WHERE o.user_id = u.user_id AND p.category_id = c.category_id))",
         ("SELECT u.username FROM Users u JOIN Orders o ON o.user_id = u.user_id "
          "JOIN Order_Items oi ON oi.order_id = o.order_id JOIN Products p ON p.product_id = oi.product_id "
          "WHERE p.category_id IS NOT NULL GROUP BY u.user_id, u.username "
          "HAVING COUNT(DISTINCT p.category_id) = (SELECT COUNT(DISTINCT category_id) FROM Products "
          "WHERE category_id IS NOT NULL)",),
         (("SELECT u.username FROM Users u JOIN Orders o ON o.user_id = u.user_id "
           "JOIN Order_Items oi ON oi.order_id = o.order_id JOIN Products p ON p.product_id = oi.product_id "
           "WHERE p.category_id IS NOT NULL GROUP BY u.user_id, u.username "
           "HAVING COUNT(DISTINCT p.category_id) = (SELECT COUNT(*) FROM Categories)",
           "wrong_subquery"),)),
    Task("S7", T5, "For each order, show the order id, the user id, and that user's running total "
                   "of spending up to and including this order, ordering each user's orders by order "
                   "date and breaking ties by order id.",
         "SELECT order_id, user_id, SUM(total_amount) OVER (PARTITION BY user_id "
         "ORDER BY order_date, order_id) FROM Orders",
         ("SELECT o.order_id, o.user_id, (SELECT SUM(o2.total_amount) FROM Orders o2 "
          "WHERE o2.user_id = o.user_id AND (o2.order_date < o.order_date OR "
          "(o2.order_date = o.order_date AND o2.order_id <= o.order_id))) FROM Orders o",),
         (("SELECT order_id, user_id, SUM(total_amount) OVER (PARTITION BY user_id "
           "ORDER BY order_date) FROM Orders", "window_frame"),
          ("SELECT order_id, user_id, SUM(total_amount) OVER (ORDER BY order_date, order_id) "
           "FROM Orders", "window_partition"))),
    Task("S8", T5, "For each category, show the category name and the name of its most expensive "
                   "product. If several products tie for the highest price, show all of them.",
         "SELECT c.name, p.name FROM Categories c JOIN Products p ON p.category_id = c.category_id "
         "WHERE p.price = (SELECT MAX(p2.price) FROM Products p2 WHERE p2.category_id = c.category_id)",
         ("SELECT c.name, p.name FROM Categories c JOIN Products p ON p.category_id = c.category_id "
          "WHERE NOT EXISTS (SELECT 1 FROM Products p2 WHERE p2.category_id = c.category_id "
          "AND p2.price > p.price)",),
         (("SELECT c.name, p.name FROM Categories c JOIN Products p ON p.category_id = c.category_id "
           "WHERE p.price = (SELECT MAX(p2.price) FROM Products p2)", "wrong_subquery"),
          ("SELECT c.name, p.name FROM Categories c JOIN Products p ON p.category_id = c.category_id "
           "WHERE p.product_id = (SELECT p2.product_id FROM Products p2 WHERE p2.category_id = c.category_id "
           "ORDER BY p2.price DESC LIMIT 1)", "ties_dropped"))),
]

from .hard_tasks import HARD_TASKS  # noqa: E402

# TASKS: the original 40 questions (used for planted bugs).
# ALL_TASKS: those plus 40 harder questions (used for model-written queries).
ALL_TASKS = TASKS + HARD_TASKS
TASK_BY_ID = {t.id: t for t in ALL_TASKS}
