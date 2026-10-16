"""Execution-grounded labeling, following the paper's protocol.

A candidate is CORRECT only if its canonicalized result equals the
ground truth's on every one of the ten seeded instances.

Canonicalization: column names are ignored, floats are rounded, NULLs get
a fixed sentinel, and rows are compared as a sorted multiset, so neither
column aliases nor row order can change a label.
"""

from __future__ import annotations

import sqlite3
import time
from functools import lru_cache

from .db import DEFAULT_SEEDS, build_instance

ROW_CAP = 10_000
TIMEOUT_S = 5.0
_NULL = ("<NULL>",)


class ExecError(Exception):
    pass


@lru_cache(maxsize=None)
def instance(seed: int) -> sqlite3.Connection:
    return build_instance(seed)


def _canon_value(v):
    if v is None:
        return _NULL
    if isinstance(v, float):
        return ("n", round(v, 2))
    if isinstance(v, int):
        return ("n", float(v))
    return ("s", str(v))


def run(sql: str, seed: int):
    """Execute `sql` on instance `seed` and return a canonical result."""
    con = instance(seed)
    deadline = time.monotonic() + TIMEOUT_S
    con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        cur = con.execute(sql)
        rows = cur.fetchmany(ROW_CAP + 1)
    except sqlite3.Error as e:  # syntax errors, timeouts, etc.
        raise ExecError(str(e)) from e
    finally:
        con.set_progress_handler(None, 0)
        con.rollback()  # never let a candidate change an instance
    if len(rows) > ROW_CAP:
        raise ExecError("row cap exceeded")
    return tuple(sorted(tuple(_canon_value(v) for v in r) for r in rows))


def same_on_all(sql_a: str, sql_b: str, seeds=DEFAULT_SEEDS) -> bool:
    """True if both queries give identical canonical results on every seed."""
    for s in seeds:
        try:
            if run(sql_a, s) != run(sql_b, s):
                return False
        except ExecError:
            return False
    return True


def label(candidate: str, gold: str, seeds=DEFAULT_SEEDS) -> bool:
    """Paper's protocol: correct iff results match gold on all instances."""
    return same_on_all(candidate, gold, seeds)


def differing_seeds(candidate: str, gold: str, seeds=DEFAULT_SEEDS) -> list[int]:
    out = []
    for s in seeds:
        try:
            if run(candidate, s) != run(gold, s):
                out.append(s)
        except ExecError:
            out.append(s)
    return out


def lenient_same(candidate: str, gold: str, seeds=DEFAULT_SEEDS) -> bool:
    """Like label(), but extra columns and a different column order are allowed.

    The candidate is accepted if some choice of its columns, in some order, reproduces
    the reference result on every instance. Used only to flag model-written queries
    whose 'mistake' is harmless (for example also returning the price).
    """
    from itertools import permutations
    for s in seeds:
        try:
            c, g = run(candidate, s), run(gold, s)
        except ExecError:
            return False
        if len(c) != len(g):
            return False
        if not g:
            continue
        nc, ng = len(c[0]), len(g[0])
        if nc < ng:
            return False
        if not any(tuple(sorted(tuple(r[k] for k in cols) for r in c)) == g
                   for cols in permutations(range(nc), ng)):
            return False
    return True
