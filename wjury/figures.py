"""Static figures for the README and the report (matplotlib, light theme).

Colors follow a validated colorblind-safe palette: blue = weighted jury,
orange = confidence-weighted jury, gray = the paper's unanimous juries.
Every chart has a CSV table next to it in results/tables.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, GRAY_LIGHT = "#8a8984", "#c9c8c3"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e7e6e2", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
    "lines.linewidth": 2, "axes.axisbelow": True,
})


def _finish(fig, path, simulated):
    if simulated:
        fig.text(0.5, 0.995, "SIMULATED JUDGES: pipeline test only, not results",
                 ha="center", va="top", fontsize=10, color=ORANGE, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96 if simulated else 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def _short(m):
    return m.replace("sim-", "")


def fig_judges(judges, path, simulated):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=GRAY_LIGHT)
    ax.text(0.62, 0.55, "random judge", color=INK2, fontsize=8, rotation=38)
    for r in judges:
        ax.scatter(r["fpr"], r["tpr"], s=60, color=BLUE, zorder=3, edgecolor=SURFACE, linewidth=1.5)
        ax.annotate(_short(r["judge"]), (r["fpr"], r["tpr"]), xytext=(6, -3),
                    textcoords="offset points", fontsize=8.5, color=INK)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.04)
    ax.set_xlabel("Wrong queries passed (false accept rate)")
    ax.set_ylabel("Correct queries passed (true accept rate)")
    ax.set_title("Each judge alone")
    _finish(fig, path, simulated)


def fig_roc(unan, roc, head, path, simulated):
    H = {r["system"]: r for r in head}
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=GRAY_LIGHT)
    xs = [r["fpr"] for r in unan if r["size"] >= 2]
    ys = [r["tpr"] for r in unan if r["size"] >= 2]
    ax.scatter(xs, ys, s=34, color=GRAY, zorder=3, label="Unanimous juries (paper's rule)",
               edgecolor=SURFACE, linewidth=1)
    colors = {"weighted": BLUE, "confidence": ORANGE}
    names = {"weighted": "Reliability-weighted, all judges", "confidence": "Confidence-weighted, all judges"}
    for rule in ("weighted", "confidence"):
        if rule in roc:
            pts = roc[rule]["roc"]
            ax.plot(pts[:, 0], pts[:, 1], color=colors[rule], drawstyle="steps-post",
                    label=f"{names[rule]} (AUC {roc[rule]['auc']:.2f})", zorder=2)
    # operating points actually used (thresholds chosen on training tasks); shown in the
    # legend rather than as text labels, because the three points sit close together
    marks = [("paper_recipe", "Paper recipe, as used", INK, "s"),
             ("weighted_all", "Reliability-weighted, as used", BLUE, "o"),
             ("confidence_all", "Confidence-weighted, as used", ORANGE, "D")]
    for key, lab, col, mk in marks:
        if key in H:
            r = H[key]
            ax.scatter([r["fpr"]], [r["tpr"]], s=95, marker=mk, facecolor="none", edgecolor=col,
                       lw=1.8, zorder=4, label=lab)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.04)
    ax.set_xlabel("Wrong queries passed (false accept rate)")
    ax.set_ylabel("Correct queries passed (true accept rate)")
    ax.set_title("Weighted juries vs every unanimous jury")
    ax.legend(loc="lower right", fontsize=8)
    _finish(fig, path, simulated)


def fig_committees(matched, path, simulated):
    rows = [r for r in matched if r["operating_point"] == "best_j"]
    by = {}
    for r in rows:
        by.setdefault(r["committee"], {})[r["rule"]] = r
    comms = sorted(by, key=lambda c: next(iter(by[c].values()))["unanimous_j"])
    if not comms:
        return
    fig, ax = plt.subplots(figsize=(8.4, max(3.0, 0.3 * len(comms) + 1.3)))
    yy = np.arange(len(comms))
    u = [next(iter(by[c].values()))["unanimous_j"] for c in comms]
    ax.scatter(u, yy, s=38, color=GRAY, zorder=3, label="Unanimous (paper's rule)")
    for rule, col, lab in (("weighted", BLUE, "Reliability-weighted"), ("confidence", ORANGE, "Confidence-weighted")):
        vals = [by[c][rule]["youden_j"] if rule in by[c] else np.nan for c in comms]
        if not np.all(np.isnan(vals)):
            ax.scatter(vals, yy, s=38, color=col, zorder=4, label=lab, edgecolor=SURFACE, linewidth=1)
    ax.set_yticks(yy)
    ax.set_yticklabels(comms, fontsize=8)
    ax.axvline(0, color=GRAY_LIGHT, lw=1)
    ax.set_xlabel("Youden's J (correct passed minus wrong passed), higher is better")
    ax.set_title("Same judges, three ways to combine their votes", pad=26)
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
              handletextpad=0.3, columnspacing=1.2)
    _finish(fig, path, simulated)


def fig_veto(veto, path, simulated):
    rows = sorted(veto, key=lambda r: r["correct_blocked_when_added"] - r["wrong_blocked_when_added"])
    fig, ax = plt.subplots(figsize=(6.4, 0.55 * len(rows) + 1.6))
    yy = np.arange(len(rows))
    hgt = 0.36
    ax.barh(yy + hgt / 2, [r["wrong_blocked_when_added"] for r in rows], height=hgt, color=BLUE,
            label="Wrong queries it blocks (good)")
    ax.barh(yy - hgt / 2, [r["correct_blocked_when_added"] for r in rows], height=hgt, color=ORANGE,
            label="Correct queries it blocks (cost)")
    ax.set_yticks(yy)
    ax.set_yticklabels([_short(r["judge"]) for r in rows], fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of the jury's previous passes that this judge blocks when added")
    ax.set_title("What each judge does when added to a unanimous jury")
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=8.5, loc="upper right")
    _finish(fig, path, simulated)


def fig_cost(cost, path, simulated):
    R = np.array([r["ratio"] for r in cost])
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.plot(R, [r["cost_human_reviews_all"] for r in cost], color=GRAY, ls="--", lw=1.5,
            label="Human reviews everything")
    ax.plot(R, [r["cost_accept_all"] for r in cost], color=GRAY, ls=":", lw=1.5,
            label="Accept everything")
    ax.plot(R, [r["cost_paper_recipe"] for r in cost], color=ORANGE, marker="o", ms=5,
            label="Best unanimous jury (paper's recipe)")
    if "cost_unanimous_by_cost" in cost[0]:
        ax.plot(R, [r["cost_unanimous_by_cost"] for r in cost], color=AQUA, marker="s", ms=4.5,
                label="Best unanimous jury for the cost")
    ax.plot(R, [r["cost_auto"] for r in cost], color=BLUE, marker="o", ms=5,
            label="Any jury, rule and threshold for the cost")
    ax.set_xscale("log")
    ax.set_xticks(R)
    ax.set_xticklabels([f"{r:g}" for r in R])
    top = max(1.4, min(4.0, max(r["cost_paper_recipe"] for r in cost) * 1.05))
    ax.set_ylim(0, top)
    ax.set_xlabel("Cost of one wrong query slipping through (in human reviews)")
    ax.set_ylabel("Expected cost per query (in human reviews)")
    ax.set_title("Cost of each review strategy (lower is better)")
    ax.legend(fontsize=8.5, loc="upper left")
    _finish(fig, path, simulated)


def _fold_rare(bugs, keys, min_n=5):
    """Merge bug types with fewer than `min_n` wrong queries into one 'other' row."""
    keep = [r for r in bugs if r["n_wrong"] >= min_n]
    rare = [r for r in bugs if r["n_wrong"] < min_n]
    if rare:
        n = sum(r["n_wrong"] for r in rare)
        other = {"bug": "other (rare types)", "n_wrong": n}
        for k in keys:
            other[f"fpr_{k}"] = sum(r[f"fpr_{k}"] * r["n_wrong"] for r in rare) / n
        keep.append(other)
    return keep


def fig_bugs(bugs, path, simulated):
    series = [("paper_recipe", GRAY, "Paper recipe (best unanimous jury)"),
              ("weighted_all", BLUE, "Reliability-weighted, all judges"),
              ("confidence_all", ORANGE, "Confidence-weighted, all judges")]
    series = [s for s in series if f"fpr_{s[0]}" in bugs[0]]
    rows = _fold_rare(bugs, [s[0] for s in series])
    rows = sorted(rows, key=lambda r: (r["bug"] != "other (rare types)", r["fpr_paper_recipe"]))
    k = len(series)
    hgt = 0.8 / k
    fig, ax = plt.subplots(figsize=(6.8, 0.5 * len(rows) + 1.6))
    yy = np.arange(len(rows))
    for i, (key, col, lab) in enumerate(series):
        ax.barh(yy + (k - 1 - 2 * i) * hgt / 2, [r[f"fpr_{key}"] for r in rows], height=hgt * 0.92,
                color=col, label=lab)
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{r['bug']}  (n={r['n_wrong']})" for r in rows], fontsize=8.5)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Share of wrong queries that got through")
    ax.set_title("Which kinds of bugs slip past the jury")
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=8, loc="lower right")
    _finish(fig, path, simulated)


def make_all(d, judges, unan, matched, head, roc, cost, bugs, veto, outdir: Path):
    sim = d["simulated"]
    for old in outdir.glob("*.png"):
        old.unlink()
    fig_judges(judges, outdir / "1_judges.png", sim)
    fig_veto(veto, outdir / "2_veto_effect.png", sim)
    fig_roc(unan, roc, head, outdir / "3_weighted_vs_unanimous.png", sim)
    fig_committees(matched, outdir / "4_per_committee.png", sim)
    fig_cost(cost, outdir / "5_cost.png", sim)
    if len(bugs) >= 2:  # with model-written queries only, every bug is "natural": nothing to compare
        fig_bugs(bugs, outdir / "6_bug_types.png", sim)
