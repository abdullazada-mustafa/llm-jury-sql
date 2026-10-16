"""E-commerce schema and seeded database instances.

The schema mirrors the six-table online store described in
"Vibe Coding on Trial" (Ullah & Serwadda, 2026): Users, Categories,
Products, Orders, Order_Items, Reviews.

Each instance is generated from a seed so the whole study is reproducible.
The generator deliberately plants the edge cases that subtle SQL bugs trip
over: users with no orders, products with no reviews, NULL ages and
locations, repeated purchases of the same product, ties on prices, and
orders in every status.
"""

from __future__ import annotations

import random
import sqlite3
from datetime import date, timedelta

SCHEMA_SQL = """
CREATE TABLE Users (
    user_id     INTEGER PRIMARY KEY,
    username    TEXT NOT NULL UNIQUE,
    email       TEXT NOT NULL,
    first_name  TEXT NOT NULL,
    last_name   TEXT NOT NULL,
    age         INTEGER,            -- nullable on purpose
    location    TEXT,               -- nullable on purpose
    created_at  TEXT NOT NULL,      -- ISO date
    is_active   INTEGER NOT NULL    -- 0 or 1
);

CREATE TABLE Categories (
    category_id INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE Products (
    product_id     INTEGER PRIMARY KEY,
    name           TEXT NOT NULL,
    description    TEXT,
    price          REAL NOT NULL,
    stock_quantity INTEGER NOT NULL,
    category_id    INTEGER REFERENCES Categories(category_id)
);

CREATE TABLE Orders (
    order_id         INTEGER PRIMARY KEY,
    user_id          INTEGER NOT NULL REFERENCES Users(user_id),
    order_date       TEXT NOT NULL,   -- ISO date
    total_amount     REAL NOT NULL,
    status           TEXT NOT NULL,   -- pending, shipped, completed, cancelled
    shipping_address TEXT
);

CREATE TABLE Order_Items (
    order_item_id INTEGER PRIMARY KEY,
    order_id      INTEGER NOT NULL REFERENCES Orders(order_id),
    product_id    INTEGER NOT NULL REFERENCES Products(product_id),
    quantity      INTEGER NOT NULL
);

CREATE TABLE Reviews (
    review_id   INTEGER PRIMARY KEY,
    user_id     INTEGER NOT NULL REFERENCES Users(user_id),
    product_id  INTEGER NOT NULL REFERENCES Products(product_id),
    rating      INTEGER NOT NULL,     -- 1..5
    review_text TEXT,
    review_date TEXT NOT NULL
);
""".strip()

# Shown to judges. Same tables and columns, no generator details.
SCHEMA_FOR_PROMPT = SCHEMA_SQL

TABLES = ["Users", "Categories", "Products", "Orders", "Order_Items", "Reviews"]
STATUSES = ["pending", "shipped", "completed", "cancelled"]
CITIES = ["Austin", "Lubbock", "Dallas", "Houston", "Denver", "Boston", "Seattle"]
CATEGORY_NAMES = ["Electronics", "Books", "Home", "Sports", "Toys", "Garden", "Music"]
FIRST = ["Ava", "Ben", "Cara", "Dan", "Eli", "Fay", "Gus", "Hana", "Ivan", "Jade",
         "Kai", "Lena", "Max", "Nora", "Omar", "Pia", "Quin", "Rosa", "Sam", "Tara"]
LAST = ["Smith", "Lopez", "Kim", "Patel", "Nguyen", "Brown", "Garcia", "Chen"]

START = date(2022, 1, 1)


def _d(rng: random.Random, start: date, span_days: int) -> str:
    return (start + timedelta(days=rng.randrange(span_days))).isoformat()


def build_instance(seed: int) -> sqlite3.Connection:
    """Create one in-memory database instance filled from `seed`."""
    rng = random.Random(seed)
    con = sqlite3.connect(":memory:")
    con.executescript(SCHEMA_SQL)

    n_users = rng.randint(28, 40)
    n_cats = rng.randint(5, len(CATEGORY_NAMES))
    n_products = rng.randint(30, 45)

    users = []
    for uid in range(1, n_users + 1):
        fn, ln = rng.choice(FIRST), rng.choice(LAST)
        users.append((
            uid,
            f"{fn.lower()}{uid}",
            f"{fn.lower()}.{ln.lower()}{uid}@example.com",
            fn,
            ln,
            None if rng.random() < 0.15 else rng.randint(16, 75),
            None if rng.random() < 0.12 else rng.choice(CITIES),
            _d(rng, START, 1000),
            0 if rng.random() < 0.2 else 1,
        ))
    con.executemany("INSERT INTO Users VALUES (?,?,?,?,?,?,?,?,?)", users)

    cats = [(cid, CATEGORY_NAMES[cid - 1], f"{CATEGORY_NAMES[cid - 1]} items",
             _d(rng, START, 300)) for cid in range(1, n_cats + 1)]
    con.executemany("INSERT INTO Categories VALUES (?,?,?,?)", cats)

    price_pool = [4.99, 9.99, 14.5, 19.99, 25.0, 49.99, 79.0, 99.99, 149.0, 299.0]
    products = []
    for pid in range(1, n_products + 1):
        # ~5% of products have no category (exercises NULL join keys).
        # The last category is never used, so one category is always empty
        # (separates LEFT JOIN from INNER JOIN, COUNT(col) from COUNT(*)).
        cat = None if rng.random() < 0.05 else rng.randint(1, n_cats - 1)
        products.append((
            pid,
            f"Product {pid}",
            None if rng.random() < 0.3 else f"Description of product {pid}",
            rng.choice(price_pool),          # repeated prices -> ties
            0 if rng.random() < 0.1 else rng.randint(1, 200),
            cat,
        ))
    con.executemany("INSERT INTO Products VALUES (?,?,?,?,?,?)", products)
    price_of = {p[0]: p[3] for p in products}

    # About a quarter of users never order, and the last ~15% of products
    # are never ordered (separates "never ordered" queries from their bugs).
    buyers = [u[0] for u in users if rng.random() > 0.25]
    orderable = max(1, int(n_products * 0.85))
    orders, items = [], []
    oid = iid = 0
    for uid in buyers:
        last_date = None
        for _ in range(rng.randint(1, 5)):
            oid += 1
            n_lines = rng.randint(1, 4)
            chosen = [rng.randint(1, orderable) for _ in range(n_lines)]
            if n_lines > 1 and rng.random() < 0.15:
                chosen[-1] = chosen[0]   # same product twice in one order
            total = 0.0
            for pid in chosen:
                iid += 1
                qty = rng.randint(1, 4)
                items.append((iid, oid, pid, qty))
                total += qty * price_of[pid]
            # Sometimes a user places two orders on the same day
            # (separates window functions with and without a tie-breaker).
            if last_date is not None and rng.random() < 0.15:
                odate = last_date
            else:
                odate = _d(rng, date(2023, 1, 1), 900)
            last_date = odate
            orders.append((
                oid, uid, odate, round(total, 2),
                rng.choice(STATUSES),
                None if rng.random() < 0.1 else f"{rng.randint(1, 999)} Main St",
            ))
    con.executemany("INSERT INTO Orders VALUES (?,?,?,?,?,?)", orders)
    con.executemany("INSERT INTO Order_Items VALUES (?,?,?,?)", items)

    reviews = []
    rid = 0
    for _ in range(rng.randint(40, 70)):
        rid += 1
        reviews.append((
            rid,
            rng.randint(1, n_users),
            rng.randint(1, n_products),       # some products get none
            rng.randint(1, 5),
            None if rng.random() < 0.2 else "Review text",
            _d(rng, date(2023, 1, 1), 900),
        ))
    con.executemany("INSERT INTO Reviews VALUES (?,?,?,?,?,?)", reviews)
    con.commit()
    return con


# Ten independent instances, as in the paper's labeling protocol.
DEFAULT_SEEDS = list(range(101, 111))
