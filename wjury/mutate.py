"""Bug planting by mutating the reference query's syntax tree.

Each operator makes one small, realistic change at one location in the
query, the kind of slip a model (or a person) makes: `>` instead of `>=`,
a dropped DISTINCT, an INNER JOIN where a LEFT JOIN was needed, a missing
filter. A mutant is only kept as a WRONG candidate if execution proves it
gives a different result on at least one of the ten databases; mutants
that happen to give identical results are discarded, because their true
label would be ambiguous.

A few semantics-preserving rewrites (flipping a comparison, reordering
AND, COUNT(*) vs COUNT(1)) produce extra CORRECT candidates, so that both
classes are well represented.

All candidates, right or wrong, go through the same pretty-printer, so
formatting style never gives away the label.
"""

from __future__ import annotations

import re
from typing import Callable, Iterator

import sqlglot
from sqlglot import exp

DIALECT = "sqlite"


def parse(sql: str) -> exp.Expression:
    return sqlglot.parse_one(sql, read=DIALECT)


def render(tree: exp.Expression) -> str:
    sql = tree.sql(dialect=DIALECT, pretty=True)
    # sqlglot prints `x IS NOT NULL` as `NOT x IS NULL`; restore the usual form.
    return re.sub(r"NOT ([\w\.]+) IS NULL", r"\1 IS NOT NULL", sql)


def normalize(sql: str) -> str:
    return render(parse(sql))


# ---------------------------------------------------------------- helpers

def _sites(tree: exp.Expression, pred: Callable[[exp.Expression], bool]) -> list[exp.Expression]:
    return [n for n in tree.walk() if pred(n)]


def _each_site(tree, pred, transform) -> Iterator[exp.Expression]:
    """Yield one mutated copy of `tree` per node matching `pred`."""
    n_sites = len(_sites(tree, pred))
    for i in range(n_sites):
        t2 = tree.copy()
        node = _sites(t2, pred)[i]
        transform(node)
        yield t2


def _replace(node: exp.Expression, new: exp.Expression) -> None:
    node.replace(new)


# ---------------------------------------------------------------- bug operators
# Each returns an iterator of mutated trees. Names map to bug types below.

def op_boundary(tree):
    swap = {exp.GT: exp.GTE, exp.GTE: exp.GT, exp.LT: exp.LTE, exp.LTE: exp.LT}
    yield from _each_site(
        tree, lambda n: type(n) in swap,
        lambda n: _replace(n, swap[type(n)](this=n.this.copy(), expression=n.expression.copy())))


def op_null_eq(tree):
    # x IS NULL  ->  x = NULL   (always false in SQL)
    yield from _each_site(
        tree, lambda n: isinstance(n, exp.Is) and isinstance(n.expression, exp.Null),
        lambda n: _replace(n, exp.EQ(this=n.this.copy(), expression=exp.Null())))


def op_count_distinct_drop(tree):
    yield from _each_site(
        tree, lambda n: isinstance(n, exp.Count) and isinstance(n.this, exp.Distinct),
        lambda n: _replace(n, exp.Count(this=n.this.expressions[0].copy())))


def op_count_distinct_add(tree):
    yield from _each_site(
        tree, lambda n: isinstance(n, exp.Count) and isinstance(n.this, exp.Column),
        lambda n: _replace(n, exp.Count(this=exp.Distinct(expressions=[n.this.copy()]))))


def op_count_star(tree):
    # COUNT(col) -> COUNT(*): counts NULL rows from outer joins
    yield from _each_site(
        tree, lambda n: isinstance(n, exp.Count) and isinstance(n.this, exp.Column),
        lambda n: _replace(n, exp.Count(this=exp.Star())))


def op_join_inner(tree):
    def to_inner(n):
        n.set("side", None)
        n.set("kind", None)
    yield from _each_site(tree, lambda n: isinstance(n, exp.Join) and n.args.get("side") == "LEFT", to_inner)


def op_join_left(tree):
    yield from _each_site(
        tree,
        lambda n: isinstance(n, exp.Join) and not n.args.get("side") and n.args.get("on") is not None,
        lambda n: n.set("side", "LEFT"))


def op_distinct_drop(tree):
    yield from _each_site(
        tree, lambda n: isinstance(n, exp.Select) and n.args.get("distinct") is not None,
        lambda n: n.set("distinct", None))


def op_drop_condition(tree):
    """Remove one AND operand of a WHERE/HAVING, or the whole clause if it has one condition."""
    def is_clause(n):
        return isinstance(n, (exp.Where, exp.Having))

    # (a) whole single-condition clause
    def single(n):
        return is_clause(n) and not isinstance(n.this, exp.And)
    yield from _each_site(tree, single, lambda n: n.pop())

    # (b) one side of a top-level AND inside a clause
    def and_in_clause(n):
        return isinstance(n, exp.And) and is_clause(n.parent)
    for side in ("this", "expression"):
        keep = "expression" if side == "this" else "this"
        yield from _each_site(
            tree, and_in_clause, lambda n, k=keep: _replace(n, n.args[k].copy()))


def op_minmax_swap(tree):
    swap = {exp.Max: exp.Min, exp.Min: exp.Max}
    yield from _each_site(
        tree, lambda n: type(n) in swap,
        lambda n: _replace(n, swap[type(n)](this=n.this.copy())))


def op_coalesce_drop(tree):
    yield from _each_site(
        tree, lambda n: isinstance(n, exp.Coalesce),
        lambda n: _replace(n, n.this.copy()))


def op_partition_drop(tree):
    yield from _each_site(
        tree, lambda n: isinstance(n, exp.Window) and n.args.get("partition_by"),
        lambda n: n.set("partition_by", None))


def op_not_exists_flip(tree):
    yield from _each_site(
        tree, lambda n: isinstance(n, exp.Not) and isinstance(n.this, exp.Exists),
        lambda n: _replace(n, n.this.copy()))


def op_not_in_flip(tree):
    # NOT x IN (...)  ->  x IN (...)
    yield from _each_site(
        tree, lambda n: isinstance(n, exp.Not) and isinstance(n.this, exp.In),
        lambda n: _replace(n, n.this.copy()))


PRIMARY_KEY = {
    "users": "user_id", "categories": "category_id", "products": "product_id",
    "orders": "order_id", "order_items": "order_item_id", "reviews": "review_id",
}


def _alias_map(tree) -> dict[str, str]:
    out = {}
    for t in tree.find_all(exp.Table):
        name = t.name.lower()
        out[(t.alias or t.name).lower()] = name
    return out


def op_wrong_join_key(tree):
    """ON a.fk = b.pk  ->  ON a.pk = b.pk  (joins on the wrong id column)."""
    aliases = _alias_map(tree)

    def site(n):
        if not (isinstance(n, exp.EQ) and isinstance(n.this, exp.Column)
                and isinstance(n.expression, exp.Column)):
            return False
        j = n.find_ancestor(exp.Join)
        if j is None or j.args.get("on") is None:
            return False
        tbl = aliases.get((n.this.table or "").lower())
        return tbl in PRIMARY_KEY and PRIMARY_KEY[tbl] != n.this.name.lower()

    def mutate(n):
        tbl = aliases[(n.this.table or "").lower()]
        n.this.set("this", exp.to_identifier(PRIMARY_KEY[tbl]))

    yield from _each_site(tree, site, mutate)


_LITERAL_SLIPS = {
    "completed": "complete",      # plausible but not a valid status
    "Lubbock": "lubbock",         # '=' is case-sensitive in SQLite
}


def op_wrong_literal(tree):
    yield from _each_site(
        tree, lambda n: isinstance(n, exp.Literal) and n.is_string and n.this in _LITERAL_SLIPS,
        lambda n: _replace(n, exp.Literal.string(_LITERAL_SLIPS[n.this])))


def op_const_off_by_one(tree):
    """`> 3` -> `> 2`, `>= 2` -> `>= 3`: integer threshold off by one."""
    def site(n):
        return (isinstance(n, exp.Literal) and not n.is_string and n.this.isdigit()
                and isinstance(n.parent, (exp.GT, exp.GTE, exp.LT, exp.LTE)))

    def mutate(n):
        v = int(n.this)
        _replace(n, exp.Literal.number(v - 1 if v > 1 else v + 1))
    yield from _each_site(tree, site, mutate)


def op_sum_to_count(tree):
    yield from _each_site(
        tree, lambda n: isinstance(n, exp.Sum) and isinstance(n.this, exp.Column),
        lambda n: _replace(n, exp.Count(this=n.this.copy())))


def op_and_to_or(tree):
    def site(n):
        return isinstance(n, exp.And) and n.find_ancestor(exp.Where, exp.Having) is not None
    yield from _each_site(
        tree, site,
        lambda n: _replace(n, exp.Or(this=n.this.copy(), expression=n.expression.copy())))


def op_order_flip(tree):
    """ORDER BY x DESC LIMIT 1 -> ORDER BY x LIMIT 1 (picks the wrong end)."""
    def site(n):
        return isinstance(n, exp.Ordered) and n.find_ancestor(exp.Select) is not None \
            and n.find_ancestor(exp.Select).args.get("limit") is not None
    def flip(n):
        desc = not n.args.get("desc")
        n.set("desc", desc)
        # keep SQLite's default NULL placement so no unusual NULLS clause is printed
        n.set("nulls_first", not desc)
    yield from _each_site(tree, site, flip)


BUG_OPERATORS: dict[str, tuple[Callable, str]] = {
    "wrong_join_key": (op_wrong_join_key, "wrong_join_key"),
    "wrong_literal": (op_wrong_literal, "wrong_literal"),
    "const_off_by_one": (op_const_off_by_one, "wrong_boundary"),
    "sum_to_count": (op_sum_to_count, "wrong_aggregate"),
    "and_to_or": (op_and_to_or, "wrong_logic"),
    "order_flip": (op_order_flip, "wrong_order"),
    "boundary": (op_boundary, "wrong_boundary"),
    "null_eq": (op_null_eq, "null_handling"),
    "coalesce_drop": (op_coalesce_drop, "null_handling"),
    "count_distinct_drop": (op_count_distinct_drop, "distinct_counting"),
    "count_distinct_add": (op_count_distinct_add, "distinct_counting"),
    "count_star": (op_count_star, "count_star"),
    "join_inner": (op_join_inner, "join_type"),
    "join_left": (op_join_left, "join_type"),
    "distinct_drop": (op_distinct_drop, "duplicates"),
    "drop_condition": (op_drop_condition, "missing_filter"),
    "minmax_swap": (op_minmax_swap, "wrong_aggregate"),
    "partition_drop": (op_partition_drop, "window_partition"),
    "not_exists_flip": (op_not_exists_flip, "inverted_logic"),
    "not_in_flip": (op_not_in_flip, "inverted_logic"),
}


def bug_mutants(sql: str) -> Iterator[tuple[str, str, str]]:
    """Yield (rendered_sql, operator_name, bug_type) for every single-site mutation."""
    tree = parse(sql)
    for name, (op, bug_type) in BUG_OPERATORS.items():
        for t2 in op(tree):
            try:
                yield render(t2), name, bug_type
            except Exception:  # an operator produced an unprintable tree; skip it
                continue


# ---------------------------------------------------------------- correct rewrites

def eq_flip_comparison(tree):
    flip = {exp.GT: exp.LT, exp.GTE: exp.LTE, exp.LT: exp.GT, exp.LTE: exp.GTE}
    yield from _each_site(
        tree, lambda n: type(n) in flip,
        lambda n: _replace(n, flip[type(n)](this=n.expression.copy(), expression=n.this.copy())))


def eq_swap_and(tree):
    yield from _each_site(
        tree, lambda n: isinstance(n, exp.And),
        lambda n: _replace(n, exp.And(this=n.expression.copy(), expression=n.this.copy())))


def eq_count_one(tree):
    yield from _each_site(
        tree, lambda n: isinstance(n, exp.Count) and isinstance(n.this, exp.Star),
        lambda n: _replace(n, exp.Count(this=exp.Literal.number(1))))


EQUIV_OPERATORS = {
    "flip_comparison": eq_flip_comparison,
    "swap_and": eq_swap_and,
    "count_one": eq_count_one,
}


def equivalent_variants(sql: str) -> Iterator[tuple[str, str]]:
    tree = parse(sql)
    for name, op in EQUIV_OPERATORS.items():
        for t2 in op(tree):
            yield render(t2), name
