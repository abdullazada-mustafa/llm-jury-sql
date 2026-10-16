"""Run with:  python -m pytest -q"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from wjury import aggregate as ag
from wjury.build_candidates import build
from wjury.db import DEFAULT_SEEDS
from wjury.execution import differing_seeds, run
from wjury.generate import extract_sql
from wjury.judges import parse_reply
from wjury.mutate import normalize
from wjury.tasks import TASKS

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- data and labels

def test_every_reference_query_runs_and_rewrites_are_correct():
    for t in TASKS:
        for s in DEFAULT_SEEDS:
            run(t.gold, s)
        for r in t.rewrites:
            assert not differing_seeds(r, t.gold), f"{t.id}: rewrite disagrees with reference"


@pytest.fixture(scope="module")
def candidates():
    cands, _ = build()
    return cands


def test_labels_match_execution(candidates):
    gold = {t.id: t.gold for t in TASKS}
    for c in candidates:
        diff = differing_seeds(c["sql"], gold[c["task_id"]])
        assert bool(diff) == (c["label"] == 0), c["cid"]


def test_no_formatting_gives_away_the_label(candidates):
    for c in candidates:
        assert normalize(c["sql"]) == c["sql"], c["cid"]


def test_committed_candidate_file_is_current(candidates):
    saved = [json.loads(l) for l in (ROOT / "data" / "candidates.jsonl").open(encoding="utf-8")]
    assert saved == candidates, "run: python -m wjury.build_candidates"


def test_queries_cannot_modify_the_databases():
    with pytest.raises(Exception):
        run("DELETE FROM Users", DEFAULT_SEEDS[0])
        assert run("SELECT COUNT(*) FROM Users", DEFAULT_SEEDS[0]) == ()


# ---------------------------------------------------------------- parsing

@pytest.mark.parametrize("text,expected", [
    ('{"verdict": "CORRECT", "confidence": 85}', (1, 0.85)),
    ('{"verdict":"incorrect","confidence":"90"}', (0, 0.90)),
    ('```json\n{"verdict": "CORRECT", "confidence": 100}\n```', (1, 1.0)),
    ('<think>maybe CORRECT</think>{"verdict": "INCORRECT", "confidence": 60}', (0, 0.60)),
    ("The query is INCORRECT.", (0, None)),
    ("This query is not correct", (0, None)),
    ("", (None, None)),
])
def test_parse_reply(text, expected):
    assert parse_reply(text) == expected


def test_extract_sql():
    assert extract_sql("```sql\nSELECT 1;\n```") == "SELECT 1"
    assert extract_sql("Here you go: SELECT a FROM b; -- done") == "SELECT a FROM b"


# ---------------------------------------------------------------- aggregation

def _synthetic(n=4000, seed=0, tprs=(0.95, 0.9, 0.7), fprs=(0.3, 0.4, 0.68)):   # last ~ random
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    tprs, fprs = np.array(tprs), np.array(fprs)
    p = np.where(y[:, None] == 1, tprs, fprs)
    V = (rng.random(p.shape) < p).astype(float)
    return V, y, tprs, fprs


def test_fixed_rules():
    V = np.array([[1, 1], [1, 0], [0, 0], [1, np.nan]], dtype=float)
    assert ag.unanimous(V).tolist() == [1, 0, 0, 0]
    V3 = np.array([[1, 1, 0], [1, 0, 0]], dtype=float)
    assert ag.majority(V3).tolist() == [1, 0]


def test_weighted_jury_learns_rates_and_mutes_random_judge():
    V, y, tprs, fprs = _synthetic()
    jury = ag.WeightedJury().fit(V, np.full(V.shape, np.nan), y)
    assert np.allclose(jury.tpr, tprs, atol=0.03) and np.allclose(jury.fpr, fprs, atol=0.03)
    strength = jury.w_yes - jury.w_no
    assert strength[-1] < 0.25 * strength[0]


def test_dawid_skene_recovers_rates_without_labels():
    # needs at least three informative judges with independent errors to be identifiable
    V, y, tprs, fprs = _synthetic(tprs=(0.95, 0.9, 0.85, 0.7), fprs=(0.3, 0.4, 0.35, 0.68))
    _, tpr, fpr, pi = ag.dawid_skene(V)
    assert np.allclose(tpr, tprs, atol=0.05) and np.allclose(fpr, fprs, atol=0.05)
    assert abs(pi - y.mean()) < 0.05


def test_threshold_for_fpr_respects_target():
    rng = np.random.default_rng(1)
    s, y = rng.normal(size=500), rng.integers(0, 2, 500)
    t = ag.threshold_for_fpr(s, y, 0.2)
    assert ag.rates(s >= t, y)[1] <= 0.2


def test_platt_scaling_fixes_overconfidence():
    rng = np.random.default_rng(2)
    true_p = rng.uniform(0.2, 0.8, 5000)
    y = (rng.random(5000) < true_p).astype(float)
    stated = 1 / (1 + np.exp(-3 * np.log(true_p / (1 - true_p))))     # overconfident
    a, b = ag.platt_fit(np.log(stated / (1 - stated)), y)
    assert 0.25 < a < 0.45 and abs(b) < 0.1


# ---------------------------------------------------------------- end to end

def test_end_to_end_with_simulated_judges(tmp_path):
    votes = tmp_path / "votes.jsonl"
    models = "sim-strong-a,sim-strong-b,sim-random-veto"
    subprocess.run([sys.executable, "-m", "wjury.judges", "--backend", "sim", "--models", models,
                    "--out", str(votes)], check=True, cwd=ROOT, capture_output=True)
    lines = votes.read_text().splitlines()
    assert len(lines) == 3 * len([*(ROOT / "data" / "candidates.jsonl").open(encoding="utf-8")])
    # resuming does no new work
    subprocess.run([sys.executable, "-m", "wjury.judges", "--backend", "sim", "--models", models,
                    "--out", str(votes)], check=True, cwd=ROOT, capture_output=True)
    assert len(votes.read_text().splitlines()) == len(lines)
    subprocess.run([sys.executable, "-m", "wjury.analyze", "--votes", str(votes), "--out", str(tmp_path),
                    "--boot", "100"], check=True, cwd=ROOT, capture_output=True)
    summary = (tmp_path / "summary.md").read_text()
    assert summary.startswith("# SIMULATED")
    assert (tmp_path / "figures" / "3_weighted_vs_unanimous.png").exists()
    assert (tmp_path / "tables" / "headline.csv").exists()


# ---------------------------------------------------------------- harder questions

def test_hard_questions_run_and_traps_are_detectable():
    from wjury.hard_tasks import HARD_TASKS
    assert len(HARD_TASKS) == 42
    for t in HARD_TASKS:
        sizes = [len(run(t.gold, s)) for s in DEFAULT_SEEDS]
        assert sum(n > 0 for n in sizes) >= 8, f"{t.id}: reference result empty too often"
        for sql, _ in t.planted:
            assert differing_seeds(sql, t.gold), f"{t.id}: trap is not detectable on the data"


def test_planted_set_uses_only_original_questions(candidates):
    assert {c["task_id"][0] for c in candidates} <= set("BAJHS")


def test_lenient_label_allows_extra_columns_only():
    from wjury.execution import lenient_same
    gold = "SELECT c.name, p.name FROM Categories c JOIN Products p ON p.category_id = c.category_id"
    assert lenient_same("SELECT p.name, c.name, p.price FROM Categories c JOIN Products p "
                        "ON p.category_id = c.category_id", gold)
    assert not lenient_same("SELECT p.name FROM Products p", gold)


def test_generate_refuses_to_replace_judged_answers(tmp_path):
    from wjury import generate
    out = tmp_path / "generated.jsonl"
    out.write_text(json.dumps({"cid": "G-m-B1", "task_id": "B1", "operator": "m"}) + "\n")
    with pytest.raises(SystemExit):
        generate.main(["--models", "m", "--out", str(out), "--raw-out", str(tmp_path / "raw.jsonl")])
    assert "G-m-B1" in out.read_text()


def test_committed_results_match_committed_votes():
    """The numbers in the README come from the committed votes; check they still load."""
    from wjury.analyze import load
    root = Path(__file__).resolve().parent.parent
    if not (root / "data" / "generated.jsonl").exists():
        pytest.skip("no model-written queries committed")
    d = load([root / "data" / "generated.jsonl"], root / "results" / "votes.jsonl")
    assert len(d["y"]) == 397 and int((d["y"] == 0).sum()) == 93
    assert len(d["models"]) == 5


def test_fixed_effects_recover_a_planted_self_effect():
    from wjury.followups import _fe_coefs
    rng = np.random.default_rng(0)
    n, m = 60, 5
    own = np.zeros((n, m))
    own[np.arange(n), rng.integers(0, m, n)] = 1
    also = (rng.random((n, m)) < 0.3) * (1 - own)
    Y = rng.normal(size=(n, 1)) + rng.normal(size=(1, m)) + 0.2 * own + 0.1 * also
    c = _fe_coefs(Y, [own, also])
    assert abs(c[0] - 0.2) < 1e-9 and abs(c[1] - 0.1) < 1e-9


def test_independence_baseline():
    from wjury.followups import _at_least
    assert abs(_at_least([0.5, 0.5, 0.5], 2) - 0.5) < 1e-12
    assert abs(_at_least([0.2, 0.3], 2) - 0.06) < 1e-12


def test_unanimous_jury_stops_at_first_no():
    from wjury.followups import _stop_time
    VS = np.array([[1, 1], [0, 1]])
    med = np.array([2.0, 3.0])
    wrong = np.array([False, True])
    # the correct query asks both judges (5 s); the wrong one stops after the first (2 s)
    assert abs(_stop_time(VS, med, (0, 1), wrong, 0.5) - 3.5) < 1e-12


def test_audit_covers_every_wrong_query():
    import csv
    root = Path(__file__).resolve().parent.parent
    with (root / "data" / "audit_wrong_queries.csv").open(encoding="utf-8") as f:
        audit = {r["cid"]: r for r in csv.DictReader(f)}
    gen = [json.loads(l) for l in (root / "data" / "generated.jsonl").open(encoding="utf-8")]
    assert set(audit) == {c["cid"] for c in gen if c["label"] == 0}
    assert {r["category"] for r in audit.values()} <= {"real", "extra_columns", "ambiguous",
                                                        "format_rule", "duplicates"}
    for c in gen:
        if c["label"] == 0 and c["label_lenient"] == 1:
            assert audit[c["cid"]]["category"] == "extra_columns"
