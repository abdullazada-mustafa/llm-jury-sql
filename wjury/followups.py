"""Follow-up analyses on the saved votes, each answering a sentence in the paper.

    python -m wjury.followups                      # model-written queries, strict labels

Ullah and Serwadda (ICSC 2026) raise these points without testing them; the quotes are
from the paper:

  shared     "errors across judges can be correlated (and simple majorities can amplify
             a shared bias)"                                              (Section I)
  self       "self-judging may introduce optimism or style bias"          (Section III-C)
  pergen     "This argues for per-generator validation" ... "deployments should test both
             inclusive (with self-judging) and exclusive configurations"  (Section V)
  hard       future work: "adding harder compositional queries"           (Section V)
  speed      "we do not optimize for latency or cost"                     (Section V)
  adaptive   future work: "adaptive or weighted committees learned from per-judge
             calibration"                                                 (Section V)

No new model runs are needed: everything uses results/votes.jsonl. Intervals are 95%
bootstrap over whole questions, as in wjury.analyze.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

from . import aggregate as ag
from .analyze import (ROOT, TaskBootstrap, all_committees, apply_policy, choose_by_cost, committee_name,
                      headline, load, pct, pp, short, task_folds, write_csv, _cost)

STRONG = ("gemma3:12b", "gpt-oss:20b", "phi4:14b")   # the three judges that block most wrong queries


class Boot:
    """Bootstrap over whole questions (tasks), optionally stratified by a group of tasks."""

    def __init__(self, tasks, n_boot=2000, seed=1, strata=None):
        uniq = sorted(set(tasks))
        members = {t: np.where(tasks == t)[0] for t in uniq}
        groups = ([uniq] if strata is None else
                  [[t for t in uniq if strata[t] == g] for g in sorted({strata[t] for t in uniq})])
        rng = np.random.default_rng(seed)
        self.samples = []
        for _ in range(n_boot):
            pick = [t for g in groups for t in rng.choice(g, size=len(g), replace=True)]
            self.samples.append(np.concatenate([members[t] for t in pick]))

    def ci(self, stat):
        with np.errstate(divide="ignore", invalid="ignore"):
            vals = np.array([stat(idx) for idx in self.samples], float)
        return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def _at_least(p, k):
    """P(at least k of the independent events with probabilities p happen)."""
    dist = np.array([1.0])
    for q in p:
        dist = np.convolve(dist, [1 - q, q])
    return float(dist[k:].sum())


def _idx(d, names):
    return tuple(d["models"].index(n) for n in names if n in d["models"])


# ---------------------------------------------------------------- 1. shared mistakes


def shared_mistakes(d, boot):
    """Do judges get fooled by the same wrong queries, and what does that do to juries?

    For every jury the observed share of wrong queries it passes is compared with what it
    would pass if the judges' mistakes were independent (same per-judge rates). A ratio
    above 1 means shared blind spots: adding judges protects less than independence implies.
    """
    V, y, m = d["V"], d["y"], d["models"]
    W = y == 0
    pairs = []
    for i, j in itertools.combinations(range(len(m)), 2):
        a, b = V[W, i], V[W, j]
        both, indep = float(np.mean(a * b)), float(a.mean() * b.mean())
        pairs.append({"judge_a": m[i], "judge_b": m[j], "a_fooled": float(a.mean()),
                      "b_fooled": float(b.mean()), "both_fooled": both,
                      "both_if_independent": indep, "ratio": both / indep if indep else np.nan,
                      "b_fooled_when_a_fooled": float((a * b).sum() / a.sum()) if a.sum() else np.nan,
                      "a_fooled_when_b_fooled": float((a * b).sum() / b.sum()) if b.sum() else np.nan})

    def jury_stats(S, rule, idx, lab):
        sub = idx[y[idx] == lab]
        VS = V[sub][:, S]
        p = VS.mean(0)
        if rule == "unanimous":
            obs, ind = ag.unanimous(VS).mean(), float(np.prod(p))
        else:
            k = len(S) // 2 + 1
            obs, ind = ag.majority(VS).mean(), _at_least(p, k)
        return float(obs), ind

    allrows = np.arange(len(y))
    juries = [("all five", tuple(range(len(m)))), ("gemma3 + gpt-oss + phi4", _idx(d, STRONG)),
              ("gpt-oss + phi4", _idx(d, ("gpt-oss:20b", "phi4:14b"))),
              ("gemma3 + gpt-oss", _idx(d, ("gemma3:12b", "gpt-oss:20b"))),
              ("gemma3 + phi4", _idx(d, ("gemma3:12b", "phi4:14b")))]
    rows = []
    for name, S in juries:
        for rule in ("unanimous", "majority"):
            if rule == "majority" and len(S) < 3:
                continue
            for lab, what in ((0, "wrong passed"), (1, "correct passed")):
                obs, ind = jury_stats(S, rule, allrows, lab)
                lo, hi = boot.ci(lambda idx: np.divide(*jury_stats(S, rule, idx, lab)))
                rows.append({"jury": name, "rule": rule, "measure": what, "observed": obs,
                             "if_independent": ind, "ratio": obs / ind if ind else np.nan,
                             "ratio_lo": lo, "ratio_hi": hi})
    return pairs, rows


def majority_vs_recipe(d, preds, tboot):
    """The paper names majority voting as untested; compare it with the unanimous recipe."""
    V, y = d["V"], d["y"]
    out = []
    systems = {"Paper recipe (best unanimous jury)": preds["paper_recipe"],
               "Unanimous, all five": ag.unanimous(V),
               "Majority, all five (3 of 5)": ag.majority(V),
               "Unanimous, gemma3 + gpt-oss + phi4": ag.unanimous(V[:, _idx(d, STRONG)]),
               "Majority, gemma3 + gpt-oss + phi4 (2 of 3)": ag.majority(V[:, _idx(d, STRONG)])}
    for name, pred in systems.items():
        t, f, J = ag.rates(pred, y)
        dci = tboot.ci_diff(pred, preds["paper_recipe"])
        out.append({"system": name, "tpr": t, "fpr": f, "youden_j": J,
                    "d_j_vs_recipe": J - ag.rates(preds["paper_recipe"], y)[2],
                    "d_j_lo": dci["j"][0], "d_j_hi": dci["j"][1],
                    "d_fpr_vs_recipe": f - ag.rates(preds["paper_recipe"], y)[1],
                    "d_fpr_lo": dci["fpr"][0], "d_fpr_hi": dci["fpr"][1]})
    return out


# ---------------------------------------------------------------- 2. self-judging


def _fe_coefs(Y, Xs):
    """Two-way fixed-effects linear probability model on a balanced query x judge panel:
    pass = query effect + judge effect + sum_k c_k * X_k. Returns the c_k."""
    def within(A):
        return A - A.mean(1, keepdims=True) - A.mean(0, keepdims=True) + A.mean()
    A = np.column_stack([within(X).ravel() for X in Xs])
    return np.linalg.lstsq(A, within(Y).ravel(), rcond=None)[0]


def self_judging(d, boot, all_task_ids):
    """Is a judge easier on its own SQL, and why?

    For each query and judge, two indicators:
      own            the judge wrote this query
      also_failed    the judge did not write it, but its own answer to the same question
                     was wrong (or did not run)
    Fixed effects for each query (how hard it is) and each judge (how lenient it is) are
    removed, so the coefficients are extra chances of passing, in percentage points.
    own minus also_failed is the part that is about the judge's own text rather than
    about misunderstanding the question the same way.
    """
    V, y, gen, m, task = d["V"], d["y"], d["gen"], d["models"], d["task"]
    answered = {(g, t): lab for g, t, lab in zip(gen, task, y)}
    failed = {(g, t): answered.get((g, t), 0) == 0 for g in m for t in all_task_ids}
    OWN = (gen[:, None] == np.array(m)[None, :]).astype(float)
    ALSO = np.array([[float(m[j] != gen[i] and failed[(m[j], task[i])]) for j in range(len(m))]
                     for i in range(len(y))])

    def coefs(idx):
        c = _fe_coefs(V[idx], [OWN[idx], ALSO[idx]])
        return {"own": c[0], "also_failed": c[1], "own_minus_also_failed": c[0] - c[1]}

    rows = []
    for lab, what in ((0, "wrong queries passed"), (1, "correct queries passed")):
        idx = np.where(y == lab)[0]
        est = coefs(idx)
        for k, v in est.items():
            lo, hi = boot.ci(lambda b, k=k: coefs(b[y[b] == lab])[k])
            rows.append({"measure": what, "effect": k, "points": float(v), "lo": lo, "hi": hi})
    # descriptive: each judge on its own wrong queries vs the other judges on the same queries
    desc = []
    for j, mod in enumerate(m):
        own = (y == 0) & (gen == mod)
        peers = [k for k in range(len(m)) if k != j]
        desc.append({"judge": mod, "own_wrong": int(own.sum()), "passes_own_wrong": float(V[own, j].mean()),
                     "others_pass_same": float(V[own][:, peers].mean())})
    return rows, desc


# ---------------------------------------------------------------- 3. per-generator juries


def per_generator(d, folds, tboot):
    """The paper picks committees in-sample per generator and recommends per-generator
    validation and testing inclusive vs exclusive (self-judging) configurations. Here the
    choice is made on training questions only and scored on held-out questions.

    pooled_inclusive   best unanimous jury on all training queries (the paper recipe)
    pooled_exclusive   same, but the writer model may not sit on its own jury
    pergen_inclusive   best jury on training queries from the same writer only
    pergen_exclusive   same, without the writer on its own jury
    """
    V, y, gen, m = d["V"], d["y"], d["gen"], d["models"]
    M = len(m)
    keys = ["pooled_inclusive", "pooled_exclusive", "pergen_inclusive", "pergen_exclusive"]
    preds = {k: np.zeros(len(y), int) for k in keys}
    picks = {k: {} for k in keys}

    def score(S, rows):
        j = ag.rates(ag.unanimous(V[rows][:, S]), y[rows])[2]
        return -np.inf if np.isnan(j) else j

    for tr, te in folds:
        for g in sorted(set(gen)):
            gi = m.index(g)
            te_g, tr_g = te & (gen == g), tr & (gen == g)
            for scope, rows in (("pooled", tr), ("pergen", tr_g)):
                for incl in ("inclusive", "exclusive"):
                    cands = [S for S in all_committees(M, 2) if incl == "inclusive" or gi not in S]
                    S = max(cands, key=lambda S: score(S, rows))
                    k = f"{scope}_{incl}"
                    preds[k][te_g] = ag.unanimous(V[te_g][:, S])
                    name = committee_name(m, S)
                    picks[k].setdefault(g, []).append(name)
    rows = []
    for g in sorted(set(gen)) + ["(all writers)"]:
        sel = np.ones(len(y), bool) if g == "(all writers)" else gen == g
        r = {"writer": g, "n_wrong": int((y[sel] == 0).sum())}
        for k in keys:
            t, f, J = ag.rates(preds[k][sel], y[sel])
            r[f"{k}_tpr"], r[f"{k}_fpr"], r[f"{k}_j"] = t, f, J
        rows.append(r)
    comps = []
    for a, b, what in (("pergen_inclusive", "pooled_inclusive", "per-writer choice vs one shared choice"),
                       ("pooled_exclusive", "pooled_inclusive", "writer kept off its own jury vs allowed"),
                       ("pergen_exclusive", "pergen_inclusive", "same, with per-writer choice")):
        ci = tboot.ci_diff(preds[a], preds[b])
        ta, fa, ja = ag.rates(preds[a], y)
        tb, fb, jb = ag.rates(preds[b], y)
        comps.append({"comparison": what, "a": a, "b": b, "d_tpr": ta - tb, "d_fpr": fa - fb,
                      "d_j": ja - jb, "d_j_lo": ci["j"][0], "d_j_hi": ci["j"][1],
                      "d_fpr_lo": ci["fpr"][0], "d_fpr_hi": ci["fpr"][1]})
    return rows, comps, picks


# ---------------------------------------------------------------- 4. harder questions


def harder_questions(d, preds, all_task_ids, n_writers):
    """Original 40 questions vs the 42 harder ones: writers' accuracy and jury safety."""
    y, V, m, task, gen = d["y"], d["V"], d["models"], d["task"], d["gen"]
    hard = d["tier"] == "hard_traps"
    hard_tasks = set(task[hard])
    strata = {t: (t in hard_tasks) for t in set(task)}
    boot = Boot(task, strata=strata)
    present = set(zip(gen, task))
    rows = []
    for name, sel_q in (("original 40", ~hard), ("harder 42", hard)):
        tset = {t for t in all_task_ids if (t in hard_tasks) == (name == "harder 42")}
        n_ans = len(tset) * n_writers
        no_run = sum((g, t) not in present for g in m for t in tset)
        r = {"questions": name, "answers": n_ans, "did_not_run": no_run,
             "correct": int((y[sel_q] == 1).sum()), "wrong": int((y[sel_q] == 0).sum())}
        for j, mod in enumerate(m):
            r[f"{short(mod)}_passes_wrong"] = float(V[sel_q & (y == 0), j].mean()) if (sel_q & (y == 0)).any() else np.nan
        for k in ("paper_recipe", "confidence_all"):
            if k in preds:
                t, f, J = ag.rates(preds[k][sel_q], y[sel_q])
                r[f"{k}_tpr"], r[f"{k}_fpr"] = t, f
        rows.append(r)

    def fpr_gap(idx, pred):
        h = idx[hard[idx] & (y[idx] == 0)]
        o = idx[~hard[idx] & (y[idx] == 0)]
        return pred[h].mean() - pred[o].mean() if len(h) and len(o) else np.nan

    def wrong_share_gap(idx):
        return 1 - y[idx][hard[idx]].mean() - (1 - y[idx][~hard[idx]].mean())

    gaps = {"wrong_share": (wrong_share_gap(np.arange(len(y))), boot.ci(wrong_share_gap))}
    for k in ("paper_recipe", "confidence_all"):
        if k in preds:
            gaps[f"fpr_{k}"] = (fpr_gap(np.arange(len(y)), preds[k]),
                                boot.ci(lambda idx, p=preds[k]: fpr_gap(idx, p)))
    gj = _idx(d, ("gpt-oss:20b",))[0]
    gaps["fpr_gpt-oss"] = (fpr_gap(np.arange(len(y)), V[:, gj]),
                           boot.ci(lambda idx: fpr_gap(idx, V[:, gj])))
    return rows, gaps


# ---------------------------------------------------------------- 5. speed vs safety


def _stop_time(VS, med, order, wmask, pw):
    """Expected judge-seconds per query for a unanimous jury that stops at the first NO."""
    reach = np.ones(len(VS))
    t = np.zeros(len(VS))
    for j in order:
        t += reach * med[j]
        reach = reach * (VS[:, j] == 1)
    return (1 - pw) * t[~wmask].mean() + pw * t[wmask].mean()


def speed_vs_safety(d, folds, pw=0.12):
    """Judge time per query against Youden's J for every unanimous jury.

    A unanimous jury can stop asking judges as soon as one says INCORRECT; the decision is
    identical, only faster. The order of judges is chosen on training questions to keep the
    expected time lowest, then timed on held-out questions. Times use each judge's median
    seconds per query on the laptop (the mean is inflated by model loading).
    """
    V, y, lat, m = d["V"], d["y"], d["lat"], d["models"]
    med = np.nanmedian(lat, axis=0)
    rows = []
    for S in all_committees(len(m)):
        VS = V[:, S]
        t, f, J = ag.rates(ag.unanimous(VS), y)
        full = float(med[list(S)].sum())
        stop = 0.0
        for tr, te in folds:
            best = min(itertools.permutations(range(len(S))),
                       key=lambda o: _stop_time(VS[tr], med[list(S)], o, y[tr] == 0, pw))
            stop += _stop_time(VS[te], med[list(S)], best, y[te] == 0, pw) * te.sum()
        stop /= len(y)
        rows.append({"committee": committee_name(m, S), "size": len(S), "tpr": t, "fpr": f,
                     "youden_j": J, "seconds_all_judges": full, "seconds_stop_at_first_no": stop,
                     "time_saved": 1 - stop / full, "S": S})
    # Pareto front: no other jury is both faster and at least as good
    for r in rows:
        r["pareto"] = not any(o["seconds_stop_at_first_no"] < r["seconds_stop_at_first_no"] - 1e-9
                              and o["youden_j"] >= r["youden_j"] for o in rows)
    return rows


# ---------------------------------------------------------------- 6. adaptive jury


def learning_curve(d, sizes=(5, 10, 15, 20, 30, 40, 50, 60), n_orders=40, R=50, pw=0.12, lam=0.001):
    """How many checked questions does a jury need before learning pays off?

    Questions arrive in a random order. After the first k questions are checked (their
    queries' true labels known), each method is fitted on them and scored on all queries
    of the questions not yet seen. Repeated over many random orders.
    """
    V, P, y, task, lat = d["V"], d["P"], d["y"], d["task"], d["lat"]
    m = V.shape[1]
    sec = np.nan_to_num(np.nanmean(lat, axis=0), nan=0.0)
    uniq = np.array(sorted(set(task)))
    methods = ["unanimous_all", "paper_recipe", "weighted", "confidence", "cost_auto"]
    res = {k: {meth: {"j": [], "cost": []} for meth in methods} for k in sizes}
    n_checked = {k: [] for k in sizes}
    for o in range(n_orders):
        order = np.random.default_rng(1000 + o).permutation(uniq)
        for k in sizes:
            tr = np.isin(task, order[:k])
            te = ~tr
            if (y[tr] == 0).sum() < 1 or (y[tr] == 1).sum() < 1:
                continue
            n_checked[k].append(int(tr.sum()))
            out = {"unanimous_all": (ag.unanimous(V[te]), lam * sec.sum())}
            S = max(all_committees(m, 2), key=lambda S: ag.rates(ag.unanimous(V[tr][:, S]), y[tr])[2])
            out["paper_recipe"] = (ag.unanimous(V[te][:, S]), lam * sec[list(S)].sum())
            for meth, cls in (("weighted", ag.WeightedJury), ("confidence", ag.ConfidenceJury)):
                jury = cls().fit(V[tr], P[tr], y[tr])
                thr = ag.threshold_for_youden(jury.score(V[tr], P[tr]), y[tr])
                out[meth] = ((jury.score(V[te], P[te]) >= thr).astype(int), lam * sec.sum())
            pol, _ = choose_by_cost(V, P, y, tr, R, lam, pw, sec, True)
            out["cost_auto"] = (apply_policy(pol, V, P, te), lam * sec[list(pol[0])].sum())
            for meth, (pred, comp) in out.items():
                res[k][meth]["j"].append(ag.rates(pred, y[te])[2])
                res[k][meth]["cost"].append(_cost(pred, y[te], R, comp, pw))
    rows = []
    for k in sizes:
        for meth in methods:
            jj, cc = np.array(res[k][meth]["j"]), np.array(res[k][meth]["cost"])
            rows.append({"questions_checked": k, "queries_checked": float(np.mean(n_checked[k])),
                         "method": meth, "j_mean": float(np.nanmean(jj)),
                         "j_p10": float(np.nanpercentile(jj, 10)), "j_p90": float(np.nanpercentile(jj, 90)),
                         f"cost_R{R}_mean": float(np.nanmean(cc)),
                         f"cost_R{R}_p10": float(np.nanpercentile(cc, 10)),
                         f"cost_R{R}_p90": float(np.nanpercentile(cc, 90)), "runs": len(jj)})
    return rows


# ---------------------------------------------------------------- figures


def figures(speed, curve, outdir, R=50):
    from .figures import AQUA, BLUE, GRAY, INK, ORANGE, _finish, plt
    outdir.mkdir(parents=True, exist_ok=True)

    # speed vs safety
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    xs = [r["seconds_stop_at_first_no"] for r in speed]
    ax.scatter(xs, [r["youden_j"] for r in speed], s=26, color=GRAY, zorder=3,
               label="Unanimous jury, stops at the first NO")
    ax.scatter([r["seconds_all_judges"] for r in speed], [r["youden_j"] for r in speed], s=26,
               facecolor="none", edgecolor=GRAY, lw=1, zorder=2, label="Same jury, asks every judge")
    front = sorted([r for r in speed if r["pareto"]], key=lambda r: r["seconds_stop_at_first_no"])
    ax.plot([r["seconds_stop_at_first_no"] for r in front], [r["youden_j"] for r in front], color=BLUE,
            lw=1.6, zorder=4, label="Fastest jury for each level of quality")
    ax.set_xlabel("Judge time per query on the laptop (seconds)")
    ax.set_ylabel("Youden's J (higher is better)")
    ax.set_title("Speed versus safety")
    ax.legend(fontsize=8, loc="lower right")
    _finish(fig, outdir / "7_speed_vs_safety.png", False)

    # learning curve: quality (left) and cost at stakes R (right)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    left = {"unanimous_all": ("Unanimous, all five (nothing to learn)", GRAY),
            "paper_recipe": ("Paper recipe (best unanimous jury so far)", ORANGE),
            "weighted": ("Reliability-weighted, all five", AQUA)}
    right = {"unanimous_all": ("Unanimous, all five (nothing to learn)", GRAY),
             "paper_recipe": ("Paper recipe (best unanimous jury so far)", ORANGE),
             "cost_auto": ("Jury, rule and threshold chosen for the cost", BLUE)}
    for meth, (lab, col) in left.items():
        rr = [r for r in curve if r["method"] == meth]
        axes[0].plot([r["queries_checked"] for r in rr], [r["j_mean"] for r in rr], color=col, marker="o",
                     ms=3.5, label=lab)
    for meth, (lab, col) in right.items():
        rr = [r for r in curve if r["method"] == meth]
        x = [r["queries_checked"] for r in rr]
        axes[1].plot(x, [r[f"cost_R{R}_mean"] for r in rr], color=col, marker="o", ms=3.5, label=lab)
        if meth == "cost_auto":
            axes[1].fill_between(x, [r[f"cost_R{R}_p10"] for r in rr], [r[f"cost_R{R}_p90"] for r in rr],
                                 color=col, alpha=0.12, lw=0, label="middle 80% of random orders")
    axes[1].axhline(1.0, color=INK, lw=1, ls="--", label="Human reviews every query")
    axes[0].set_ylabel("Youden's J on questions not yet seen")
    axes[0].set_ylim(0.5, 0.9)
    axes[1].set_ylabel(f"Cost per query at R = {R} (human reviews)")
    for ax in axes:
        ax.set_xlabel("Queries already checked (true label known)")
    axes[0].set_title("Quality as checked queries accumulate")
    axes[1].set_title(f"Cost when a missed bug costs {R} reviews")
    axes[0].legend(fontsize=7.5, loc="lower right")
    axes[1].legend(fontsize=7.5, loc="upper right")
    _finish(fig, outdir / "8_learning_curve.png", False)


# ---------------------------------------------------------------- main


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--votes", default=str(ROOT / "results" / "votes.jsonl"))
    ap.add_argument("--candidates", default=str(ROOT / "data" / "generated.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "results" / "natural" / "followups"))
    ap.add_argument("--label", choices=["strict", "lenient", "audited"], default="strict")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--orders", type=int, default=40)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    from .tasks import ALL_TASKS
    out = Path(args.out)
    tables = out / "tables"
    d = load([args.candidates], args.votes, label=args.label)
    folds = task_folds(d["task"], k=5)
    tboot = TaskBootstrap(d["task"], d["y"], n_boot=args.boot)
    boot = Boot(d["task"], n_boot=args.boot)
    _, preds, _ = headline(d, folds, tboot)

    pairs, juries = shared_mistakes(d, boot)
    maj = majority_vs_recipe(d, preds, tboot)
    selfj, selfdesc = self_judging(d, boot, [t.id for t in ALL_TASKS])
    pg_rows, pg_comps, _ = per_generator(d, folds, tboot)
    hard_rows, hard_gaps = harder_questions(d, preds, [t.id for t in ALL_TASKS], len(d["models"]))
    speed = speed_vs_safety(d, folds)
    curve = learning_curve(d, n_orders=args.orders)

    write_csv(tables / "shared_mistakes_pairs.csv", pairs)
    write_csv(tables / "shared_mistakes_juries.csv", juries)
    write_csv(tables / "majority_vs_recipe.csv", maj)
    write_csv(tables / "self_judging.csv", selfj)
    write_csv(tables / "self_judging_by_judge.csv", selfdesc)
    write_csv(tables / "per_generator.csv", pg_rows)
    write_csv(tables / "per_generator_comparisons.csv", pg_comps)
    write_csv(tables / "harder_questions.csv", hard_rows)
    write_csv(tables / "speed_vs_safety.csv", speed)
    write_csv(tables / "learning_curve.csv", curve)
    if not args.no_figures:
        figures(speed, curve, out / "figures")

    def iv(lo, hi, f=pp):
        return f"[{f(lo)}, {f(hi)}]"

    L = [f"# Follow-up analyses ({args.label} labels)", "",
         f"{len(d['y'])} model-written queries ({int((d['y'] == 0).sum())} wrong). "
         "Each section answers a sentence from Ullah and Serwadda (2026). Intervals: 95% bootstrap over questions.", ""]

    L += ["## 1. Shared mistakes", "", "> \"errors across judges can be correlated (and simple majorities can amplify a "
          "shared bias)\" (Section I)", "",
          "| Jury | Rule | Wrong queries passed | If mistakes were independent | Ratio [95% interval] |",
          "|---|---|---|---|---|"]
    for r in juries:
        if r["measure"] == "wrong passed":
            L.append(f"| {r['jury']} | {r['rule']} | {pct(r['observed'])} | {100 * r['if_independent']:.1f}% | "
                     f"{r['ratio']:.1f}x [{r['ratio_lo']:.1f}, {r['ratio_hi']:.1f}] |")
    go = [p for p in pairs if "gpt-oss" in p["judge_a"] + p["judge_b"] and "codegemma" not in p["judge_a"]]
    L += ["", "When gpt-oss is fooled, how often the others are fooled too (vs their usual rate):", ""]
    for p in go:
        other, when, usual = ((p["judge_b"], p["b_fooled_when_a_fooled"], p["b_fooled"])
                              if "gpt-oss" in p["judge_a"] else (p["judge_a"], p["a_fooled_when_b_fooled"], p["a_fooled"]))
        L.append(f"- {short(other)}: {pct(when)} (usually {pct(usual)})")
    L += ["", "| System | Passes correct | Passes wrong | J | J vs recipe [95%] | Wrong passed vs recipe [95%] |",
          "|---|---|---|---|---|---|"]
    for r in maj:
        L.append(f"| {r['system']} | {pct(r['tpr'])} | {pct(r['fpr'])} | {r['youden_j']:.2f} | "
                 f"{pp(r['d_j_vs_recipe'])} {iv(r['d_j_lo'], r['d_j_hi'])} | "
                 f"{pp(r['d_fpr_vs_recipe'])} {iv(r['d_fpr_lo'], r['d_fpr_hi'])} |")

    L += ["", "## 2. Is a model easier on its own SQL?", "",
          "> \"self-judging may introduce optimism or style bias\" (Section III-C)", "",
          "Extra chance, in percentage points, that a judge passes a query, after removing how hard the query "
          "is and how lenient the judge is in general:", "",
          "| | Wrong queries [95%] | Correct queries [95%] |", "|---|---|---|"]
    labels = {"own": "Judge wrote this query", "also_failed": "Judge did not write it, but got the same question wrong",
              "own_minus_also_failed": "Difference: own text vs same misunderstanding"}
    for k, lab in labels.items():
        w = next(r for r in selfj if r["effect"] == k and r["measure"].startswith("wrong"))
        c = next(r for r in selfj if r["effect"] == k and r["measure"].startswith("correct"))
        L.append(f"| {lab} | {pp(w['points'])} {iv(w['lo'], w['hi'])} | {pp(c['points'])} {iv(c['lo'], c['hi'])} |")
    L += ["", "| Judge | Its own wrong queries | It passed | Other judges passed the same queries |",
          "|---|---|---|---|"]
    for r in selfdesc:
        L.append(f"| {short(r['judge'])} | {r['own_wrong']} | {pct(r['passes_own_wrong'])} | {pct(r['others_pass_same'])} |")

    L += ["", "## 3. One jury per writer model, and keeping the writer off its own jury", "",
          "> \"This argues for per-generator validation\" ... \"deployments should test both inclusive (with "
          "self-judging) and exclusive configurations\" (Section V)", "",
          "Juries are chosen on training questions and scored on held-out questions (the paper chose them on "
          "the same queries it reported).", "",
          "| Comparison | J difference [95%] | Wrong passed difference [95%] |", "|---|---|---|"]
    for c in pg_comps:
        L.append(f"| {c['comparison']} | {pp(c['d_j'])} {iv(c['d_j_lo'], c['d_j_hi'])} | "
                 f"{pp(c['d_fpr'])} {iv(c['d_fpr_lo'], c['d_fpr_hi'])} |")
    L += ["", "| Writer | Wrong | J: shared choice | J: per-writer choice | J: shared, writer excluded | "
          "J: per-writer, writer excluded |", "|---|---|---|---|---|---|"]
    for r in pg_rows:
        L.append(f"| {short(r['writer'])} | {r['n_wrong']} | {r['pooled_inclusive_j']:.2f} | "
                 f"{r['pergen_inclusive_j']:.2f} | {r['pooled_exclusive_j']:.2f} | {r['pergen_exclusive_j']:.2f} |")

    L += ["", "## 4. Harder questions", "", "> future work: \"adding harder compositional queries\" (Section V)", "",
          "| Questions | Answers | Did not run | Correct | Wrong | Wrong passed by paper recipe | "
          "Wrong passed by gpt-oss |", "|---|---|---|---|---|---|---|"]
    for r in hard_rows:
        L.append(f"| {r['questions']} | {r['answers']} | {r['did_not_run']} | {r['correct']} | {r['wrong']} | "
                 f"{pct(r.get('paper_recipe_fpr'))} | {pct(r.get('gpt-oss:20b_passes_wrong'))} |")
    for k, (v, (lo, hi)) in hard_gaps.items():
        L.append(f"- harder minus original, {k.replace('_', ' ')}: {pp(v)} points {iv(lo, hi)}")

    best = max(speed, key=lambda r: r["youden_j"])
    near = min((r for r in speed if r["youden_j"] >= best["youden_j"] - 0.02),
               key=lambda r: r["seconds_stop_at_first_no"])
    allj = next(r for r in speed if r["size"] == len(d["models"]))
    saved = np.array([r["time_saved"] for r in speed if r["size"] >= 2])
    L += ["", "## 5. Speed versus safety", "", "> \"we do not optimize for latency or cost\" (Section V)", "",
          f"- Stopping a unanimous jury at the first INCORRECT gives the same decisions and saves "
          f"{100 * saved.mean():.0f}% of judge time on average over all juries of 2+ judges "
          f"(range {100 * saved.min():.0f}% to {100 * saved.max():.0f}%).",
          f"- All five judges: {allj['seconds_all_judges']:.1f} s per query asking everyone, "
          f"{allj['seconds_stop_at_first_no']:.1f} s with stopping.",
          f"- Best unanimous jury (in-sample): {best['committee']}, J {best['youden_j']:.2f}, "
          f"{best['seconds_stop_at_first_no']:.1f} s per query.",
          f"- Fastest jury within 0.02 of that J: {near['committee']}, J {near['youden_j']:.2f}, "
          f"{near['seconds_stop_at_first_no']:.1f} s per query.", "",
          "Fastest unanimous jury that keeps wrong queries passed under a target (in-sample, as in the paper):", "",
          "| Target: wrong passed at most | Fastest jury | Passes correct | Passes wrong | Seconds per query |",
          "|---|---|---|---|---|"]
    for target in (0.20, 0.15, 0.10):
        ok = [r for r in speed if r["fpr"] <= target + 1e-9]
        if ok:
            r = min(ok, key=lambda r: (r["seconds_stop_at_first_no"], -r["tpr"]))
            L.append(f"| {pct(target)} | {r['committee']} | {pct(r['tpr'])} | {pct(r['fpr'])} | "
                     f"{r['seconds_stop_at_first_no']:.1f} |")
        else:
            L.append(f"| {pct(target)} | no unanimous jury gets there | | | |")
    L += ["", "Fastest juries for each level of quality (Pareto front):", "",
          "| Jury | J | Seconds per query (stop at first NO) | Asking every judge |", "|---|---|---|---|"]
    for r in sorted([r for r in speed if r["pareto"]], key=lambda r: r["seconds_stop_at_first_no"]):
        L.append(f"| {r['committee']} | {r['youden_j']:.2f} | {r['seconds_stop_at_first_no']:.1f} | "
                 f"{r['seconds_all_judges']:.1f} |")

    L += ["", "## 6. A jury that learns as checked queries arrive", "",
          "> future work: \"adaptive or weighted committees learned from per-judge calibration\" (Section V)", "",
          f"Questions arrive in random order ({args.orders} orders). After k questions are checked, each method "
          "is fitted on them and scored on the questions not yet seen. Mean over orders.", "",
          "| Questions checked | Queries checked | J: unanimous all | J: paper recipe | J: weighted | "
          "Cost R=50: paper recipe | Cost R=50: chosen for cost |", "|---|---|---|---|---|---|---|"]
    for k in sorted({r["questions_checked"] for r in curve}):
        rr = {r["method"]: r for r in curve if r["questions_checked"] == k}
        L.append(f"| {k} | {rr['paper_recipe']['queries_checked']:.0f} | {rr['unanimous_all']['j_mean']:.2f} | "
                 f"{rr['paper_recipe']['j_mean']:.2f} | {rr['weighted']['j_mean']:.2f} | "
                 f"{rr['paper_recipe']['cost_R50_mean']:.2f} | {rr['cost_auto']['cost_R50_mean']:.2f} |")
    out.mkdir(parents=True, exist_ok=True)
    (out / "followups.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
