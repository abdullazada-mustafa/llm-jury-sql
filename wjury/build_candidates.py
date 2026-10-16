"""Build the candidate set the judges will review.

    python -m wjury.build_candidates            # writes data/candidates.jsonl

For every task this collects
  CORRECT candidates: the reference query, hand-written rewrites, and
                      automatic equivalent rewrites
  WRONG candidates:   hand-planted bugs plus automatic single-site mutants

and labels every one of them by execution on all ten databases, exactly as
in the paper's protocol. Mutants that give the same results as the
reference everywhere are thrown away (their true label is ambiguous), and
so are mutants that fail to run (judges would reject those trivially).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .execution import ExecError, differing_seeds, run
from .db import DEFAULT_SEEDS
from .mutate import bug_mutants, equivalent_variants, normalize
from .tasks import TASKS

ROOT = Path(__file__).resolve().parent.parent


def _runs_everywhere(sql: str) -> bool:
    try:
        for s in DEFAULT_SEEDS:
            run(sql, s)
        return True
    except ExecError:
        return False


def build(max_correct: int = 3, max_wrong: int = 8, seed: int = 7):
    rng = random.Random(seed)
    candidates, stats = [], Counter()

    for task in TASKS:
        gold = normalize(task.gold)
        assert not differing_seeds(gold, task.gold), f"{task.id}: normalizing changed gold"

        # ---- correct pool (order = priority)
        correct, seen = [], set()

        def add_correct(sql, source, operator=None):
            if sql in seen:
                return
            if differing_seeds(sql, task.gold):
                stats["correct_rejected_by_execution"] += 1
                return
            seen.add(sql)
            correct.append({"sql": sql, "source": source, "operator": operator})

        add_correct(gold, "gold")
        for r in task.rewrites:
            add_correct(normalize(r), "rewrite")
        variants = list(equivalent_variants(task.gold))
        rng.shuffle(variants)
        for sql, op in variants:
            add_correct(sql, "auto_equivalent", op)
        correct = correct[:max_correct]

        # ---- wrong pool
        planted, auto, = [], []
        for raw, bug_type in task.planted:
            sql = normalize(raw)
            if sql in seen:
                continue
            if not _runs_everywhere(sql):
                stats["discarded_error"] += 1
                continue
            d = differing_seeds(sql, task.gold)
            if not d:
                stats["discarded_equivalent"] += 1
                continue
            seen.add(sql)
            planted.append({"sql": sql, "source": "planted", "operator": "hand",
                            "bug_type": bug_type, "differs_on": len(d)})
        sources = [task.gold, *task.rewrites]   # mutate every correct form
        mutants = [m for src in sources for m in bug_mutants(src)]
        for sql, op, bug_type in mutants:
            if sql in seen:
                continue
            seen.add(sql)
            if not _runs_everywhere(sql):
                stats["discarded_error"] += 1
                continue
            d = differing_seeds(sql, task.gold)
            if not d:
                stats["discarded_equivalent"] += 1
                continue
            auto.append({"sql": sql, "source": "auto_mutant", "operator": op,
                         "bug_type": bug_type, "differs_on": len(d)})

        # planted bugs first, then round-robin over bug types for variety
        by_type = defaultdict(list)
        for m in auto:
            by_type[m["bug_type"]].append(m)
        for lst in by_type.values():
            rng.shuffle(lst)
        types = sorted(by_type)
        rng.shuffle(types)
        wrong = list(planted)
        while len(wrong) < max_wrong and any(by_type[t] for t in types):
            for t in types:
                if by_type[t] and len(wrong) < max_wrong:
                    wrong.append(by_type[t].pop())
        wrong = wrong[:max_wrong]

        for i, c in enumerate(correct):
            candidates.append({
                "cid": f"{task.id}-c{i}", "task_id": task.id, "tier": task.tier,
                "question": task.question, "sql": c["sql"], "label": 1,
                "source": c["source"], "operator": c["operator"], "bug_type": None,
                "differs_on": 0,
            })
        for i, w in enumerate(wrong):
            candidates.append({
                "cid": f"{task.id}-w{i}", "task_id": task.id, "tier": task.tier,
                "question": task.question, "sql": w["sql"], "label": 0,
                "source": w["source"], "operator": w["operator"], "bug_type": w["bug_type"],
                "differs_on": w["differs_on"],
            })
    return candidates, stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "data" / "candidates.jsonl"))
    ap.add_argument("--max-correct", type=int, default=3)
    ap.add_argument("--max-wrong", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    cands, stats = build(args.max_correct, args.max_wrong, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for c in cands:
            f.write(json.dumps(c) + "\n")

    n_pos = sum(c["label"] for c in cands)
    print(f"wrote {len(cands)} candidates to {out}  ({n_pos} correct, {len(cands) - n_pos} wrong)")
    print("bug types:", dict(Counter(c["bug_type"] for c in cands if not c["label"]).most_common()))
    print("by tier  :", dict(Counter((c["tier"], c["label"]) for c in cands)))
    print("discarded:", dict(stats))


if __name__ == "__main__":
    main()
