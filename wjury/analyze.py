"""Turn saved judge votes into tables, figures and a summary.

    python -m wjury.analyze --votes results/votes.jsonl --out results

Everything that learns something (judge weights, calibration, thresholds,
which jury to use) is fitted on some tasks and evaluated on *different*
tasks (5-fold cross-validation grouped by task), so no rule is graded on
the questions it was tuned on. Confidence intervals come from resampling
whole tasks (bootstrap), because queries from the same task are related.

Outputs
  tables/judges.csv              each judge's true/false accept rates (paper's metrics)
  tables/unanimous.csv           every unanimous committee, as in the paper
  tables/veto_impact.csv         how much each judge changes unanimous committees
  tables/matched.csv             unanimous vs weighted juries at the same safety level
  tables/cost.csv                automatic jury choice vs fixed recipes, per cost ratio
  tables/bug_types.csv           false accepts per bug type
  tables/tiers.csv               true/false accepts per difficulty tier
  figures/*.png                  the charts used in the README and report
  summary.md                     the headline numbers in plain words
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import aggregate as ag

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- loading


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).open(encoding="utf-8") if l.strip()]


def load(candidate_files, votes_file, models=None, label="strict"):
    cands = [c for f in candidate_files for c in load_jsonl(f)]
    if label == "lenient":
        cands = [{**c, "label": c.get("label_lenient", c["label"])} for c in cands]
    elif label == "audited":
        # strict label, except wrong queries that a reading by hand found to be no real
        # mistake (extra columns, an ambiguous question, a formatting rule, a repeated row)
        audit_path = ROOT / "data" / "audit_wrong_queries.csv"
        with audit_path.open(encoding="utf-8") as f:
            ok = {r["cid"] for r in csv.DictReader(f) if r["category"] != "real"}
        cands = [{**c, "label": 1 if c["cid"] in ok else c["label"]} for c in cands]
    cand_by_id = {c["cid"]: c for c in cands}
    latest = {}
    backends = set()
    for r in load_jsonl(votes_file):          # later lines win (retries, reruns)
        if r["cid"] not in cand_by_id or r.get("error"):
            continue
        latest[(r["cid"], r["model"])] = r
        backends.add(r.get("backend", "?"))
    all_models = sorted({m for _, m in latest})
    models = models or all_models
    missing = {m: sum((c["cid"], m) not in latest for c in cands) for m in models}
    keep = [c for c in cands if all((c["cid"], m) in latest for m in models)]
    if len(keep) < len(cands):
        print(f"note: using {len(keep)}/{len(cands)} candidates answered by every judge; "
              f"missing per judge: { {m: n for m, n in missing.items() if n} }")
    n, m = len(keep), len(models)
    V = np.full((n, m), np.nan)
    P = np.full((n, m), np.nan)
    lat = np.full((n, m), np.nan)
    for i, c in enumerate(keep):
        for j, mod in enumerate(models):
            r = latest[(c["cid"], mod)]
            if r["verdict"] is not None:
                V[i, j] = r["verdict"]
                if r.get("confidence") is not None:
                    conf = float(r["confidence"])
                    P[i, j] = conf if r["verdict"] == 1 else 1 - conf
            if r.get("latency_s") is not None:
                lat[i, j] = r["latency_s"]
    # A verdict without a readable confidence keeps its vote: use that judge's typical certainty.
    for j in range(m):
        certainty = np.where(V[:, j] == 1, P[:, j], 1 - P[:, j])
        med = np.nanmedian(certainty) if np.isfinite(certainty).any() else 0.75
        fill = ~np.isnan(V[:, j]) & np.isnan(P[:, j])
        P[fill, j] = np.where(V[fill, j] == 1, med, 1 - med)
    if all(r.get("confidence") is None for r in latest.values()):
        P[:] = np.nan        # no judge reported confidence at all: confidence rules are skipped
    data = {
        "cands": keep, "models": models, "V": V, "P": P, "lat": lat,
        "y": np.array([c["label"] for c in keep]),
        "task": np.array([c["task_id"] for c in keep]),
        "bug": np.array([c["bug_type"] or "" for c in keep]),
        "tier": np.array([c["tier"] for c in keep]),
        "gen": np.array([c["operator"] if c.get("source") == "generated" else "" for c in keep]),
        "simulated": backends == {"sim"} or "sim" in backends,
    }
    return data


# ---------------------------------------------------------------- cross-validation


def task_folds(tasks, k=5, seed=0):
    uniq = sorted(set(tasks))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    groups = [set(uniq[i::k]) for i in range(k)]
    return [(np.array([t not in g for t in tasks]), np.array([t in g for t in tasks])) for g in groups]


def oof_predict(maker, V, P, y, folds, threshold_rule):
    """Out-of-fold accept decisions and scores for a scored jury.

    threshold_rule(train_scores, train_y, train_mask, fold_index) -> threshold
    """
    pred = np.zeros(len(y), dtype=int)
    scores = np.zeros(len(y))
    for k, (tr, te) in enumerate(folds):
        jury = maker().fit(V[tr], P[tr], y[tr])
        s_tr = jury.score(V[tr], P[tr])
        t = threshold_rule(s_tr, y[tr], tr, k)
        s_te = jury.score(V[te], P[te])
        scores[te] = s_te
        pred[te] = (s_te >= t).astype(int)
    return pred, scores


# ---------------------------------------------------------------- bootstrap over tasks


class TaskBootstrap:
    def __init__(self, tasks, y, n_boot=2000, seed=1):
        self.uniq = np.array(sorted(set(tasks)))
        idx = {t: i for i, t in enumerate(self.uniq)}
        self.tix = np.array([idx[t] for t in tasks])
        self.y = y
        rng = np.random.default_rng(seed)
        self.samples = rng.integers(0, len(self.uniq), size=(n_boot, len(self.uniq)))
        self.pos = np.bincount(self.tix, weights=(y == 1), minlength=len(self.uniq))
        self.neg = np.bincount(self.tix, weights=(y == 0), minlength=len(self.uniq))

    def _per_task(self, pred):
        tp = np.bincount(self.tix, weights=(pred == 1) & (self.y == 1), minlength=len(self.uniq))
        fp = np.bincount(self.tix, weights=(pred == 1) & (self.y == 0), minlength=len(self.uniq))
        return tp, fp

    def rates(self, pred):
        tp, fp = self._per_task(pred)
        P = self.pos[self.samples].sum(1)
        N = self.neg[self.samples].sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            # resamples with no correct (or no wrong) queries give NaN and are skipped below
            return tp[self.samples].sum(1) / P, fp[self.samples].sum(1) / N

    def ci_diff(self, pred_a, pred_b):
        """95% intervals for (TPR_a - TPR_b), (FPR_a - FPR_b), (J_a - J_b)."""
        ta, fa = self.rates(pred_a)
        tb, fb = self.rates(pred_b)
        out = {}
        for name, d in (("tpr", ta - tb), ("fpr", fa - fb), ("j", (ta - fa) - (tb - fb))):
            out[name] = (float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5)))
        return out

    def cost(self, pred, R, pw, compute=0.0):
        """Bootstrap distribution of the expected cost per query (see _cost)."""
        t, f = self.rates(pred)
        reject = (1 - pw) * (1 - t) + pw * (1 - f)
        return R * pw * f + reject + compute

    def ci(self, pred):
        t, f = self.rates(pred)
        return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
                for k, v in (("tpr", t), ("fpr", f), ("j", t - f))}


# ---------------------------------------------------------------- experiments


def short(m):
    return m.replace("sim-", "")


def committee_name(models, S):
    return " + ".join(short(models[j]) for j in S)


def per_judge(d):
    V, P, y, lat, models = d["V"], d["P"], d["y"], d["lat"], d["models"]
    rows = []
    for j, m in enumerate(models):
        v = V[:, j]
        ok = ~np.isnan(v)
        tpr, fpr, J = ag.rates(v[ok], y[ok])
        right = ok & (v == y)
        wrong = ok & (v != y)
        conf = np.where(V[:, j] == 1, P[:, j], 1 - P[:, j])     # stated certainty in own verdict
        rows.append({
            "judge": m, "tpr": tpr, "fpr": fpr, "youden_j": J,
            "unreadable": float((~ok).mean()),
            "conf_when_right": float(np.nanmean(conf[right])) if right.any() else np.nan,
            "conf_when_wrong": float(np.nanmean(conf[wrong])) if wrong.any() else np.nan,
            "mean_latency_s": float(np.nanmean(lat[:, j])),
            "pattern": ("rubber stamp" if fpr >= 0.85 and tpr >= 0.95
                        else "near-random" if abs(J) < 0.1 else ""),
        })
    return rows


def all_committees(m, min_size=1):
    for k in range(min_size, m + 1):
        yield from itertools.combinations(range(m), k)


def unanimous_table(d):
    rows = []
    for S in all_committees(len(d["models"])):
        pred = ag.unanimous(d["V"][:, S])
        tpr, fpr, J = ag.rates(pred, d["y"])
        rows.append({"committee": committee_name(d["models"], S), "size": len(S),
                     "tpr": tpr, "fpr": fpr, "youden_j": J, "S": S})
    return rows


def veto_impact(d):
    """What adding one judge does to a unanimous jury, averaged over every jury it could join.

    For each committee S without judge j: compare unanimous(S) with unanimous(S + j).
    Reported as the share of correct / wrong queries that S passed but S + j blocks.
    """
    V, y, m = d["V"], d["y"], len(d["models"])
    rows = []
    for j, mod in enumerate(d["models"]):
        lost_tp, lost_fp = [], []
        for S in all_committees(m):
            if j in S:
                continue
            t0, f0, _ = ag.rates(ag.unanimous(V[:, S]), y)
            t1, f1, _ = ag.rates(ag.unanimous(V[:, S + (j,)]), y)
            if t0 > 0:
                lost_tp.append((t0 - t1) / t0)
            if f0 > 0:
                lost_fp.append((f0 - f1) / f0)
        rows.append({"judge": mod,
                     "correct_blocked_when_added": float(np.mean(lost_tp)),
                     "wrong_blocked_when_added": float(np.mean(lost_fp))})
    return rows


def matched_comparison(d, folds, boot, rules=("weighted", "confidence", "dawid_skene")):
    """For every committee of 2+ judges: the paper's unanimous vote vs scored juries.

    Two operating points for each scored jury, both tuned on training tasks only:
      best_j       threshold with the highest Youden's J (the paper's summary metric)
      same_safety  most permissive threshold whose false accept rate is no higher than
                   the unanimous jury's (on training tasks). With yes/no votes only a few
                   thresholds exist, so this often reproduces the unanimous decision exactly;
                   with confidence scores it can be tuned finely.
    """
    V, P, y = d["V"], d["P"], d["y"]
    rows = []
    for S in all_committees(len(d["models"]), min_size=2):
        VS, PS = V[:, S], P[:, S]
        u_pred = ag.unanimous(VS)
        u_tpr, u_fpr, u_j = ag.rates(u_pred, y)
        for rule in rules:
            if rule == "confidence" and np.isnan(PS).all():
                continue
            for point in ("best_j", "same_safety"):
                if point == "best_j":
                    def thr(s_tr, y_tr, tr, k):
                        return ag.threshold_for_youden(s_tr, y_tr)
                else:
                    def thr(s_tr, y_tr, tr, k, VS=VS):
                        target = ag.rates(ag.unanimous(VS[tr]), y_tr)[1]
                        return ag.threshold_for_fpr(s_tr, y_tr, target)
                pred, _ = oof_predict(ag.SCORED_JURIES[rule], VS, PS, y, folds, thr)
                tpr, fpr, J = ag.rates(pred, y)
                ci = boot.ci_diff(pred, u_pred)
                rows.append({
                    "committee": committee_name(d["models"], S), "size": len(S), "rule": rule,
                    "operating_point": point,
                    "unanimous_tpr": u_tpr, "unanimous_fpr": u_fpr, "unanimous_j": u_j,
                    "tpr": tpr, "fpr": fpr, "youden_j": J,
                    "d_tpr": tpr - u_tpr, "d_tpr_lo": ci["tpr"][0], "d_tpr_hi": ci["tpr"][1],
                    "d_fpr": fpr - u_fpr, "d_fpr_lo": ci["fpr"][0], "d_fpr_hi": ci["fpr"][1],
                    "d_j": J - u_j, "d_j_lo": ci["j"][0], "d_j_hi": ci["j"][1],
                })
    return rows


def headline(d, folds, boot):
    """The main comparison, everything cross-validated by task.

    paper_recipe        unanimous committee (2+ judges) with the best J on training tasks
    unanimous_all       every judge must agree
    weighted_all        reliability-weighted vote of all judges, best-J threshold
    confidence_all      confidence-weighted vote of all judges, best-J threshold
    confidence_safe     confidence-weighted vote of all judges, tuned to the paper
                        recipe's false accept rate on training tasks
    label_free          Dawid-Skene on all votes without labels, 50% posterior threshold
    """
    V, P, y = d["V"], d["P"], d["y"]
    m = V.shape[1]
    preds, picks = {}, Counter()

    recipe = np.zeros(len(y), int)
    recipe_fpr_train = {}
    for k, (tr, te) in enumerate(folds):
        bestS = max(all_committees(m, 2), key=lambda S: ag.rates(ag.unanimous(V[tr][:, S]), y[tr])[2])
        picks[committee_name(d["models"], bestS)] += 1
        recipe[te] = ag.unanimous(V[te][:, bestS])
        recipe_fpr_train[k] = ag.rates(ag.unanimous(V[tr][:, bestS]), y[tr])[1]
    preds["paper_recipe"] = recipe
    preds["unanimous_all"] = ag.unanimous(V)

    def best_j(s, yy, tr, k):
        return ag.threshold_for_youden(s, yy)

    def safe(s, yy, tr, k):
        return ag.threshold_for_fpr(s, yy, recipe_fpr_train[k])

    preds["weighted_all"], _ = oof_predict(ag.WeightedJury, V, P, y, folds, best_j)
    if not np.isnan(P).all():
        preds["confidence_all"], _ = oof_predict(ag.ConfidenceJury, V, P, y, folds, best_j)
        preds["confidence_safe"], _ = oof_predict(ag.ConfidenceJury, V, P, y, folds, safe)
    q, *_ = ag.dawid_skene(V)
    preds["label_free"] = (q >= 0.5).astype(int)

    rows = []
    for name, pred in preds.items():
        t, f, J = ag.rates(pred, y)
        ci = boot.ci(pred)
        dci = boot.ci_diff(pred, preds["paper_recipe"])
        rows.append({"system": name, "tpr": t, "tpr_lo": ci["tpr"][0], "tpr_hi": ci["tpr"][1],
                     "fpr": f, "fpr_lo": ci["fpr"][0], "fpr_hi": ci["fpr"][1],
                     "youden_j": J, "j_lo": ci["j"][0], "j_hi": ci["j"][1],
                     "d_j_vs_recipe": J - ag.rates(preds["paper_recipe"], y)[2],
                     "d_j_lo": dci["j"][0], "d_j_hi": dci["j"][1]})
    return rows, preds, picks


def full_panel_roc(d, folds):
    V, P, y = d["V"], d["P"], d["y"]
    out = {}
    for rule in ("weighted", "confidence", "dawid_skene"):
        if rule == "confidence" and np.isnan(P).all():
            continue
        _, scores = oof_predict(ag.SCORED_JURIES[rule], V, P, y, folds, lambda s, yy, tr, k: 0.0)
        out[rule] = {"roc": ag.roc_points(scores, y), "auc": ag.auc(scores, y)}
    return out


def label_free(d):
    """Dawid-Skene on all votes with no labels and its natural 50% threshold."""
    q, tpr, fpr, pi = ag.dawid_skene(d["V"])
    pred = (q >= 0.5).astype(int)
    t, f, J = ag.rates(pred, d["y"])
    est = [{"judge": m, "est_tpr": float(a), "est_fpr": float(b)} for m, a, b in zip(d["models"], tpr, fpr)]
    return {"tpr": t, "fpr": f, "youden_j": J, "estimated": est}


# ---------------------------------------------------------------- cost-aware selection


def policy_space(m, conf_available):
    yield ((), "human_reviews_all")
    for S in all_committees(m):
        yield (S, "unanimous")
        if len(S) >= 3:
            yield (S, "majority")
        if len(S) >= 2:
            yield (S, "weighted")
            if conf_available:
                yield (S, "confidence")


def _cost(pred, y, R, compute, pw):
    """Expected cost per query when a share `pw` of incoming queries is wrong.

    Uses the jury's true and false accept rates, so the answer does not depend on how
    many correct and wrong queries happen to be in the test set.
    """
    tpr, fpr, _ = ag.rates(pred, y)
    reject = (1 - pw) * (1 - tpr) + pw * (1 - fpr)
    return R * pw * fpr + reject + compute


def choose_by_cost(V, P, y, tr, R, lam, pw, sec, conf_ok):
    """Lowest expected cost on rows `tr` over every jury, rule and threshold.

    Returns two policies (committee, rule, threshold, fitted jury): the best overall, and
    the best that keeps the paper's unanimous rule (or sends every query to a human).
    """
    best = (np.inf, None)
    best_u = (np.inf, None)
    m = V.shape[1]
    for S, rule in policy_space(m, conf_ok):
        compute = lam * sec[list(S)].sum()
        if rule == "human_reviews_all":
            c = _cost(np.zeros(tr.sum(), int), y[tr], R, 0.0, pw)
            pol = (S, rule, None, None)
            if c < best[0]:
                best = (c, pol)
            if c < best_u[0]:
                best_u = (c, pol)
        elif rule in ("unanimous", "majority"):
            f = ag.unanimous if rule == "unanimous" else ag.majority
            c = _cost(f(V[tr][:, S]), y[tr], R, compute, pw)
            pol = (S, rule, None, None)
            if c < best[0]:
                best = (c, pol)
            if rule == "unanimous" and c < best_u[0]:
                best_u = (c, pol)
        else:
            jury = ag.SCORED_JURIES[rule]().fit(V[tr][:, S], P[tr][:, S], y[tr])
            s = jury.score(V[tr][:, S], P[tr][:, S])
            grid = np.unique(s) if len(np.unique(s)) <= 64 else np.quantile(s, np.linspace(0, 1, 65))
            for t in np.concatenate([grid, [np.inf]]):
                c = _cost((s >= t).astype(int), y[tr], R, compute, pw)
                if c < best[0]:
                    best = (c, (S, rule, t, jury))
    return best[1], best_u[1]


def apply_policy(pol, V, P, rows):
    S, rule, t, jury = pol
    if rule == "human_reviews_all":
        return np.zeros(int(np.sum(rows)) if rows.dtype == bool else len(rows), int)
    if rule == "unanimous":
        return ag.unanimous(V[rows][:, S])
    if rule == "majority":
        return ag.majority(V[rows][:, S])
    return (jury.score(V[rows][:, S], P[rows][:, S]) >= t).astype(int)


def cost_aware(d, folds, ratios, lam, pw, boot=None):
    """Choose jury, rule and threshold on training tasks to minimize expected cost.

    Cost per query, in units of "one human review of a rejected query":
      R * P(query wrong and accepted) + P(query rejected) + lam * (judge seconds per query)
    with a share `pw` of incoming queries assumed wrong.
    """
    V, P, y, lat = d["V"], d["P"], d["y"], d["lat"]
    m = V.shape[1]
    sec = np.nan_to_num(np.nanmean(lat, axis=0), nan=0.0)
    conf_ok = not np.isnan(P).all()
    rows = []
    for R in ratios:
        # unanimous_by_cost: the paper's rule only (any unanimous jury, or a human for
        # everything), but chosen by cost. It isolates what the strictness dial adds.
        test_pred = {"auto": np.zeros(len(y), int), "paper_recipe": np.zeros(len(y), int),
                     "unanimous_by_cost": np.zeros(len(y), int), "best_single": np.zeros(len(y), int)}
        test_compute = {k: np.zeros(len(y)) for k in test_pred}
        choices = Counter()
        for tr, te in folds:
            pol, pol_u = choose_by_cost(V, P, y, tr, R, lam, pw, sec, conf_ok)
            S, rule = pol[0], pol[1]
            choices[(committee_name(d["models"], S) or "no judges", rule)] += 1
            p = apply_policy(pol, V, P, te)
            test_pred["auto"][te] = p
            test_compute["auto"][te] = lam * sec[list(S)].sum()
            test_pred["unanimous_by_cost"][te] = apply_policy(pol_u, V, P, te)
            test_compute["unanimous_by_cost"][te] = lam * sec[list(pol_u[0])].sum()

            # the paper's recipe: the unanimous committee (size >= 2) with the best J on training tasks
            bestS = max(all_committees(m, 2), key=lambda S: ag.rates(ag.unanimous(V[tr][:, S]), y[tr])[2])
            test_pred["paper_recipe"][te] = ag.unanimous(V[te][:, bestS])
            test_compute["paper_recipe"][te] = lam * sec[list(bestS)].sum()
            bj = max(range(m), key=lambda j: ag.rates(np.nan_to_num(V[tr][:, j]), y[tr])[2])
            test_pred["best_single"][te] = np.nan_to_num(V[te][:, bj]).astype(int)
            test_compute["best_single"][te] = lam * sec[bj]

        row = {"ratio": R}
        for k in test_pred:
            row[f"cost_{k}"] = _cost(test_pred[k], y, R, test_compute[k].mean(), pw)
        if boot is not None:
            ca = boot.cost(test_pred["auto"], R, pw, test_compute["auto"].mean())
            cr = boot.cost(test_pred["paper_recipe"], R, pw, test_compute["paper_recipe"].mean())
            cu = boot.cost(test_pred["unanimous_by_cost"], R, pw, test_compute["unanimous_by_cost"].mean())
            for name, dlt in (("vs_recipe", ca - cr), ("vs_unanimous_by_cost", ca - cu),
                              ("vs_human_all", ca - 1.0)):
                row[f"auto_{name}_lo"] = float(np.nanpercentile(dlt, 2.5))
                row[f"auto_{name}_hi"] = float(np.nanpercentile(dlt, 97.5))
        row["cost_human_reviews_all"] = 1.0
        row["cost_accept_all"] = R * pw
        top, n = choices.most_common(1)[0]
        row["typical_choice"] = f"{top[1]}: {top[0]} ({n}/{len(folds)} folds)"
        rows.append(row)
    return rows


# ---------------------------------------------------------------- breakdowns


def breakdown(d, preds: dict, key):
    groups = defaultdict(list)
    for i, g in enumerate(d[key]):
        groups[g].append(i)
    rows = []
    for g, idx in sorted(groups.items()):
        idx = np.array(idx)
        row = {key: g or "(correct queries)", "n_correct": int((d["y"][idx] == 1).sum()),
               "n_wrong": int((d["y"][idx] == 0).sum())}
        for name, pred in preds.items():
            t, f, _ = ag.rates(pred[idx], d["y"][idx])
            row[f"tpr_{name}"], row[f"fpr_{name}"] = t, f
        rows.append(row)
    return rows


def adaptive_by_generator(d, folds, boot, min_wrong=5):
    """Pooled weights vs weights learned separately for each SQL-writing model.

    Both use the reliability-weighted jury with a best-J threshold, fitted on training
    tasks only. The per-generator version uses only the training queries written by the
    same model; if that model has fewer than `min_wrong` wrong training queries, it falls
    back to the pooled weights.
    """
    V, P, y, gen = d["V"], d["P"], d["y"], d["gen"]
    if not (gen != "").any():
        return [], None
    pooled = np.zeros(len(y), int)
    adaptive = np.zeros(len(y), int)
    fallbacks = 0
    for tr, te in folds:
        jp = ag.WeightedJury().fit(V[tr], P[tr], y[tr])
        tp = ag.threshold_for_youden(jp.score(V[tr], P[tr]), y[tr])
        pooled[te] = (jp.score(V[te], P[te]) >= tp).astype(int)
        for g in set(gen[te]):
            gte = te & (gen == g)
            gtr = tr & (gen == g)
            if g == "" or ((y[gtr] == 0).sum() < min_wrong) or ((y[gtr] == 1).sum() < min_wrong):
                adaptive[gte] = pooled[gte]
                fallbacks += 1
                continue
            jg = ag.WeightedJury().fit(V[gtr], P[gtr], y[gtr])
            tg = ag.threshold_for_youden(jg.score(V[gtr], P[gtr]), y[gtr])
            adaptive[gte] = (jg.score(V[gte], P[gte]) >= tg).astype(int)
    rows = []
    for g in sorted(set(gen)) + ["(all)"]:
        idx = np.ones(len(y), bool) if g == "(all)" else gen == g
        tpp, fpp, jpp = ag.rates(pooled[idx], y[idx])
        tpa, fpa, jpa = ag.rates(adaptive[idx], y[idx])
        rows.append({"generator": g or "(planted)", "n_correct": int((y[idx] == 1).sum()),
                     "n_wrong": int((y[idx] == 0).sum()), "pooled_tpr": tpp, "pooled_fpr": fpp,
                     "pooled_j": jpp, "adaptive_tpr": tpa, "adaptive_fpr": fpa, "adaptive_j": jpa})
    ci = boot.ci_diff(adaptive, pooled)
    rows[-1].update({"d_j_lo": ci["j"][0], "d_j_hi": ci["j"][1], "fallback_folds": fallbacks})
    return rows, ci


def writer_by_judge(d):
    """Share of each writer model's WRONG queries that each judge passed.

    Reads along a column for self-judging (diagonal) vs judging other models.
    Descriptive only: a writer's mistakes differ in difficulty, so compare the
    diagonal with the other judges on the same row, not with the judge's own column.
    """
    gen, y, V, m = d["gen"], d["y"], d["V"], d["models"]
    rows = []
    for g in sorted(set(gen) - {""}):
        idx = (gen == g) & (y == 0)
        if not idx.any():
            continue
        row = {"writer": g, "n_wrong": int(idx.sum())}
        for j, name in enumerate(m):
            row[name] = float(V[idx, j].mean())
        rows.append(row)
    return rows


def per_judge_bugs(d):
    """Share of wrong queries of each bug type that each judge alone let through."""
    rows = []
    for bug in sorted(set(d["bug"][d["y"] == 0])):
        idx = (d["y"] == 0) & (d["bug"] == bug)
        row = {"bug": bug, "n_wrong": int(idx.sum())}
        for j, m in enumerate(d["models"]):
            v = d["V"][idx, j]
            row[m] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
        rows.append(row)
    return sorted(rows, key=lambda r: -r["n_wrong"])


def write_review(d, path):
    """Every wrong query with each judge's vote, for reading by hand."""
    rows = []
    for i, c in enumerate(d["cands"]):
        if d["y"][i] != 0:
            continue
        row = {"cid": c["cid"], "task": c["task_id"], "bug_type": c.get("bug_type"),
               "lenient_label": c.get("label_lenient", ""), "question": c["question"],
               "sql": c["sql"].replace("\n", " ")}
        for j, m in enumerate(d["models"]):
            v = d["V"][i, j]
            row[m] = "" if np.isnan(v) else ("CORRECT" if v == 1 else "INCORRECT")
        rows.append(row)
    write_csv(path, rows)


# ---------------------------------------------------------------- output helpers


def write_csv(path, rows, drop=("S",)):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [k for k in rows[0] if k not in drop]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k in keys})


def pct(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.0f}%"


def pp(x):
    return f"{100 * x:+.1f}"


# ---------------------------------------------------------------- main


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--votes", default=str(ROOT / "results" / "votes.jsonl"))
    ap.add_argument("--candidates", default=str(ROOT / "data" / "candidates.jsonl"),
                    help="comma-separated candidate files")
    ap.add_argument("--models", default="", help="judges to include (default: all in the votes file)")
    ap.add_argument("--out", default=str(ROOT / "results" / "planted"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--ratios", default="1,2,5,10,20,50,100",
                    help="cost of one false accept, in units of one human review")
    ap.add_argument("--lam", type=float, default=0.001,
                    help="cost of one judge-second, in units of one human review")
    ap.add_argument("--wrong-share", type=float, default=0.12,
                    help="share of incoming queries that are wrong, for the cost analysis "
                         "(about 12%% among the paper's six best generators)")
    ap.add_argument("--label", choices=["strict", "lenient", "audited"], default="strict",
                    help="strict: results must match exactly (the paper's rule); lenient: extra or "
                         "reordered columns are allowed (only affects model-written queries)")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    d = load(args.candidates.split(","), args.votes,
             [m.strip() for m in args.models.split(",") if m.strip()] or None, args.label)
    out = Path(args.out)
    tables = out / "tables"
    y = d["y"]
    print(f"{len(y)} candidates ({int(y.sum())} correct, {int((y == 0).sum())} wrong), "
          f"{len(set(d['task']))} tasks, judges: {d['models']}"
          + ("   [SIMULATED JUDGES]" if d["simulated"] else ""))

    n_wrong = int((y == 0).sum())
    if n_wrong < 30:
        print(f"WARNING: only {n_wrong} wrong queries. Rates on wrong queries and all comparisons are "
              f"very uncertain; treat them as anecdotes, not results.")
    write_review(d, out / "tables" / "wrong_queries_review.csv")
    folds = task_folds(d["task"], args.folds)
    boot = TaskBootstrap(d["task"], y, args.boot)

    judges = per_judge(d)
    unan = unanimous_table(d)
    veto = veto_impact(d)
    head, hpreds, picks = headline(d, folds, boot)
    matched = matched_comparison(d, folds, boot)
    roc = full_panel_roc(d, folds)
    lf = label_free(d)
    ratios = [float(r) for r in args.ratios.split(",")]
    cost = cost_aware(d, folds, ratios, args.lam, args.wrong_share, boot)
    bd_preds = {"paper_recipe": hpreds["paper_recipe"], "weighted_all": hpreds["weighted_all"]}
    if "confidence_all" in hpreds:
        bd_preds["confidence_all"] = hpreds["confidence_all"]
    bugs = [r for r in breakdown(d, bd_preds, "bug") if r["n_wrong"]]
    tiers = breakdown(d, bd_preds, "tier")
    judge_bugs = per_judge_bugs(d)
    adaptive, _ = adaptive_by_generator(d, folds, boot)
    writers = writer_by_judge(d)

    write_csv(tables / "judges.csv", judges)
    write_csv(tables / "unanimous.csv", unan)
    write_csv(tables / "veto_impact.csv", veto)
    write_csv(tables / "headline.csv", head)
    write_csv(tables / "per_committee.csv", matched)
    write_csv(tables / "cost.csv", cost)
    write_csv(tables / "bug_types.csv", bugs)
    write_csv(tables / "tiers.csv", tiers)
    write_csv(tables / "judge_bug_types.csv", judge_bugs)
    write_csv(tables / "adaptive_by_generator.csv", adaptive)
    write_csv(tables / "writer_by_judge.csv", writers)
    (tables / "label_free.json").write_text(json.dumps(lf, indent=2, default=float))

    # ------------------------------------------------ summary in plain words
    H = {r["system"]: r for r in head}
    names = {
        "paper_recipe": "Paper recipe: best unanimous jury, picked on training tasks",
        "unanimous_all": "Unanimous, all judges",
        "weighted_all": "Reliability-weighted vote, all judges",
        "confidence_all": "Confidence-weighted vote, all judges",
        "confidence_safe": "Confidence-weighted, tuned to the recipe's false accept rate",
        "label_free": "Dawid-Skene, all judges, no labels at all",
    }
    L = []
    if d["simulated"]:
        L += ["# SIMULATED RESULTS (pipeline test only, not findings)", "",
              "These numbers come from fake judges with hand-set error rates. They only show "
              "that the code runs end to end. Do not cite them.", ""]
    else:
        L += ["# Results summary", ""]
    if n_wrong < 30:
        L += [f"**Warning: only {n_wrong} wrong queries.** Everything measured on wrong queries is very "
              "uncertain. Read tables/wrong_queries_review.csv query by query instead.", ""]
    L += [f"{len(y)} candidate queries ({int(y.sum())} correct, {int((y == 0).sum())} wrong) from "
          f"{len(set(d['task']))} tasks, reviewed by {len(d['models'])} judges. "
          f"Everything that is tuned is tuned on other tasks (5-fold, grouped by task); "
          f"intervals are 95% bootstrap over tasks. Labels: {args.label}.", "",
          "## Each judge alone", "",
          "| Judge | Passes correct | Passes wrong | Youden's J | Pattern |", "|---|---|---|---|---|"]
    for r in judges:
        L.append(f"| {r['judge']} | {pct(r['tpr'])} | {pct(r['fpr'])} | {r['youden_j']:.2f} | {r['pattern']} |")
    best_judge = max(judges, key=lambda r: r["youden_j"])["judge"]
    jb = [r for r in judge_bugs if r["n_wrong"] >= 3]
    jb = sorted(jb, key=lambda r: -r[best_judge])[:5]
    L += ["", f"Bug types that most often fool the best single judge (**{best_judge}**), "
              "share of those wrong queries it passed:", ""]
    L += [f"- {r['bug']} (n={r['n_wrong']}): {pct(r[best_judge])}" for r in jb]
    L += ["", "## Main comparison", "",
          "| System | Passes correct | Passes wrong | Youden's J | J vs paper recipe |", "|---|---|---|---|---|"]
    for k in names:
        if k in H:
            r = H[k]
            rel = "" if k == "paper_recipe" else f"{pp(r['d_j_vs_recipe'])} [{pp(r['d_j_lo'])}, {pp(r['d_j_hi'])}]"
            L.append(f"| {names[k]} | {pct(r['tpr'])} | {pct(r['fpr'])} | {r['youden_j']:.2f} "
                     f"[{r['j_lo']:.2f}, {r['j_hi']:.2f}] | {rel} |")
    L += ["", "Committee the paper recipe picked: " + ", ".join(f"{c} ({n}/{len(folds)} folds)"
                                                            for c, n in picks.most_common()) + "."]
    if roc:
        L.append("Area under the ROC curve for the full panel: "
                 + ", ".join(f"{k} {v['auc']:.3f}" for k, v in roc.items()) + ".")
    worst = max(veto, key=lambda r: r["correct_blocked_when_added"] - r["wrong_blocked_when_added"])
    L += ["", "## The veto effect", "",
          "| Judge | Correct queries it blocks when added | Wrong queries it blocks when added |",
          "|---|---|---|"]
    for r in veto:
        L.append(f"| {r['judge']} | {pct(r['correct_blocked_when_added'])} | {pct(r['wrong_blocked_when_added'])} |")
    if worst["correct_blocked_when_added"] > worst["wrong_blocked_when_added"]:
        L += ["", f"Adding **{worst['judge']}** to a unanimous jury blocks on average "
                  f"{pct(worst['correct_blocked_when_added'])} of the correct queries the jury used to pass, "
                  f"but only {pct(worst['wrong_blocked_when_added'])} of the wrong ones: it hurts the jury.", ""]
    else:
        least = min(veto, key=lambda r: r["wrong_blocked_when_added"])
        L += ["", "No judge blocks more correct queries than wrong ones when added. The least useful is "
                  f"**{least['judge']}**, which blocks only {pct(least['wrong_blocked_when_added'])} of the wrong "
                  f"queries a jury used to pass.", ""]
    for rule in ("weighted", "confidence"):
        rs = [r for r in matched if r["rule"] == rule and r["operating_point"] == "best_j"]
        if rs:
            better = sum(r["d_j_lo"] > 0 for r in rs)
            worse = sum(r["d_j_hi"] < 0 for r in rs)
            L.append(f"- Per committee, **{rule}** vs unanimous (best-J threshold): Youden's J higher in "
                     f"{sum(r['d_j'] > 0 for r in rs)}/{len(rs)} committees, clearly higher in {better}, "
                     f"clearly lower in {worse}.")
    if adaptive:
        a = adaptive[-1]
        L += ["", "## Weights learned per SQL-writing model (\"adaptive\")", "",
              "| Writer model | Wrong queries | J, pooled weights | J, per-writer weights |", "|---|---|---|---|"]
        for r in adaptive:
            L.append(f"| {r['generator']} | {r['n_wrong']} | {r['pooled_j']:.2f} | {r['adaptive_j']:.2f} |")
        L += ["", f"Overall difference (per-writer minus pooled): {pp(a['adaptive_j'] - a['pooled_j'])} points, "
                  f"95% interval {pp(a['d_j_lo'])} to {pp(a['d_j_hi'])}."]
    if writers:
        L += ["", "## Who wrote the wrong query vs who judged it", "",
              "Share of each writer's wrong queries that each judge passed (lower is better). "
              "The diagonal is a model judging its own SQL.", "",
              "| Writer (wrong queries) | " + " | ".join(short(x) for x in d["models"]) + " |",
              "|---|" + "---|" * len(d["models"])]
        for r in writers:
            cells = [(f"**{pct(r[x])}**" if x == r["writer"] else pct(r[x])) for x in d["models"]]
            L.append(f"| {short(r['writer'])} ({r['n_wrong']}) | " + " | ".join(cells) + " |")
    L += ["", "## Choosing the jury automatically by cost", "",
          f"Cost per query, in units of one human review. A wrong query that slips through costs R reviews. "
          f"Assumes {pct(args.wrong_share)} of incoming queries are wrong.",
          "",
          "| R | Automatic choice | Paper recipe | Automatic minus recipe (95% interval) "
          "| Unanimous jury chosen by cost | Automatic minus that (95% interval) | Best single judge "
          "| Human reviews all | Accept all | Usual choice |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in cost:
        L.append(f"| {r['ratio']:g} | {r['cost_auto']:.3f} | {r['cost_paper_recipe']:.3f} | "
                 f"{r['cost_auto'] - r['cost_paper_recipe']:+.3f} [{r['auto_vs_recipe_lo']:+.3f}, "
                 f"{r['auto_vs_recipe_hi']:+.3f}] | {r['cost_unanimous_by_cost']:.3f} | "
                 f"{r['cost_auto'] - r['cost_unanimous_by_cost']:+.3f} [{r['auto_vs_unanimous_by_cost_lo']:+.3f}, "
                 f"{r['auto_vs_unanimous_by_cost_hi']:+.3f}] | {r['cost_best_single']:.3f} | "
                 f"{r['cost_human_reviews_all']:.3f} | {r['cost_accept_all']:.3f} | {r['typical_choice']} |")
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))

    if not args.no_figures:
        from .figures import make_all
        make_all(d, judges, unan, matched, head, roc, cost, bugs, veto, out / "figures")
        print(f"\nfigures written to {out / 'figures'}")


if __name__ == "__main__":
    main()
