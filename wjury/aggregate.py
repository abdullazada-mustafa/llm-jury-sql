"""Ways of turning several judges' votes into one accept/reject decision.

Notation: for each candidate query c and judge j,
  V[c, j]  = 1 if the judge said CORRECT, 0 if INCORRECT, NaN if no usable answer
  P[c, j]  = judge's probability that the query is correct, built from its verdict and
             stated confidence (conf if it said CORRECT, 1 - conf if INCORRECT); NaN if missing
  y[c]     = 1 if the query is truly correct (by execution), else 0

Rules
  unanimous      accept only if every judge says CORRECT  (the paper's rule)
  majority       accept if more than half say CORRECT
  weighted       each judge's vote is worth log(TPR/FPR) when it says CORRECT and
                 log((1-TPR)/(1-FPR)) when it says INCORRECT, with that judge's true and
                 false accept rates measured on training queries. A judge that accepts
                 good and bad queries at the same rate gets a weight near zero; a
                 rubber stamp's CORRECT is worth almost nothing. This is the classic
                 "naive Bayes" vote combination.
  confidence     like weighted, but uses each judge's stated confidence, first
                 recalibrated on training queries (Platt scaling), so an over-confident
                 judge's 95% is turned into what it is really worth.
  dawid_skene    the same idea learned WITHOUT any labels, by the EM algorithm of
                 Dawid and Skene (1979); useful when no labeled queries exist.

The last three give a score; a candidate is accepted when the score is at least
a threshold. The threshold sets the safety level and is always chosen on
training queries only.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-6


# ---------------------------------------------------------------- fixed rules

def unanimous(V: np.ndarray) -> np.ndarray:
    """1 where every judge voted CORRECT. A missing answer counts as not CORRECT."""
    return np.all(np.nan_to_num(V, nan=0.0) == 1, axis=1).astype(int)


def majority(V: np.ndarray) -> np.ndarray:
    yes = np.nansum(V, axis=1)
    return (yes > V.shape[1] / 2).astype(int)


# ---------------------------------------------------------------- per-judge rates

def judge_rates(V: np.ndarray, y: np.ndarray, alpha: float = 1.0):
    """Smoothed true accept rate and false accept rate of each judge."""
    pos, neg = y == 1, y == 0
    yes = np.nan_to_num(V, nan=0.0)
    seen = ~np.isnan(V)
    tpr = (yes[pos].sum(0) + alpha) / (seen[pos].sum(0) + 2 * alpha)
    fpr = (yes[neg].sum(0) + alpha) / (seen[neg].sum(0) + 2 * alpha)
    return tpr, fpr


def weights_from_rates(tpr, fpr):
    w_yes = np.log(np.clip(tpr, EPS, 1) / np.clip(fpr, EPS, 1))
    w_no = np.log(np.clip(1 - tpr, EPS, 1) / np.clip(1 - fpr, EPS, 1))
    return w_yes, w_no


def weighted_scores(V, w_yes, w_no) -> np.ndarray:
    """Sum of each judge's log-likelihood ratio; missing answers add nothing."""
    s = np.where(V == 1, w_yes, 0.0) + np.where(V == 0, w_no, 0.0)
    return np.nan_to_num(s, nan=0.0).sum(axis=1)


class WeightedJury:
    name = "weighted"

    def fit(self, V, P, y):
        self.tpr, self.fpr = judge_rates(V, y)
        self.w_yes, self.w_no = weights_from_rates(self.tpr, self.fpr)
        return self

    def score(self, V, P):
        return weighted_scores(V, self.w_yes, self.w_no)


# ---------------------------------------------------------------- confidence-based

def _logit(p):
    p = np.clip(p, 0.01, 0.99)
    return np.log(p / (1 - p))


def platt_fit(x, y, iters=50, l2=1e-2):
    """Fit P(y=1) = sigmoid(a*x + b) by Newton's method with a small ridge penalty."""
    X = np.column_stack([x, np.ones_like(x)])
    w = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w)))
        g = X.T @ (p - y) + l2 * w
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(2)
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


class ConfidenceJury:
    """Recalibrate each judge's stated confidence, then add calibrated log-odds."""
    name = "confidence"

    def fit(self, V, P, y):
        self.prior = _logit(np.array([y.mean()]))[0]
        self.params = []
        for j in range(V.shape[1]):
            ok = ~np.isnan(P[:, j])
            if ok.sum() < 5 or len(np.unique(y[ok])) < 2:
                self.params.append(None)
                continue
            self.params.append(platt_fit(_logit(P[ok, j]), y[ok].astype(float)))
        return self

    def score(self, V, P):
        s = np.zeros(len(P))
        for j, w in enumerate(self.params):
            if w is None:
                continue
            ok = ~np.isnan(P[:, j])
            s[ok] += w[0] * _logit(P[ok, j]) + w[1] - self.prior
        return s


# ---------------------------------------------------------------- unsupervised

def dawid_skene(V, iters=100, tol=1e-8):
    """Binary Dawid-Skene EM. Returns (posterior P(correct) per item, tpr, fpr, prior)."""
    n, m = V.shape
    seen = ~np.isnan(V)
    yes = np.nan_to_num(V, nan=0.0)
    # start from majority vote
    q = np.clip(np.nanmean(V, axis=1), 0.05, 0.95)
    q = np.nan_to_num(q, nan=0.5)
    for _ in range(iters):
        pi = np.clip(q.mean(), EPS, 1 - EPS)
        tpr = (q[:, None] * yes).sum(0) / np.maximum((q[:, None] * seen).sum(0), EPS)
        fpr = ((1 - q)[:, None] * yes).sum(0) / np.maximum(((1 - q)[:, None] * seen).sum(0), EPS)
        tpr, fpr = np.clip(tpr, 0.01, 0.99), np.clip(fpr, 0.01, 0.99)
        w_yes, w_no = weights_from_rates(tpr, fpr)
        logit = np.log(pi / (1 - pi)) + weighted_scores(V, w_yes, w_no)
        q_new = 1 / (1 + np.exp(-logit))
        if np.abs(q_new - q).max() < tol:
            q = q_new
            break
        q = q_new
    return q, tpr, fpr, pi


class DawidSkeneJury:
    """Learns judge reliability from the unlabeled votes alone; ignores y in fit."""
    name = "dawid_skene"

    def fit(self, V, P, y):
        _, self.tpr, self.fpr, self.pi = dawid_skene(V)
        self.w_yes, self.w_no = weights_from_rates(self.tpr, self.fpr)
        return self

    def score(self, V, P):
        return np.log(self.pi / (1 - self.pi)) + weighted_scores(V, self.w_yes, self.w_no)


SCORED_JURIES = {"weighted": WeightedJury, "confidence": ConfidenceJury, "dawid_skene": DawidSkeneJury}


# ---------------------------------------------------------------- metrics and thresholds

def rates(pred, y):
    pred, y = np.asarray(pred), np.asarray(y)
    tpr = pred[y == 1].mean() if (y == 1).any() else np.nan
    fpr = pred[y == 0].mean() if (y == 0).any() else np.nan
    return tpr, fpr, tpr - fpr


def threshold_for_fpr(scores, y, max_fpr):
    """Lowest threshold whose false accept rate on (scores, y) is <= max_fpr.

    Lower threshold = more accepts = higher true accept rate, so this is the
    most permissive setting that still meets the safety target.
    """
    cands = np.unique(np.concatenate([scores, [np.inf]]))
    neg = scores[y == 0]
    for t in cands:                       # ascending
        if neg.size == 0 or (neg >= t).mean() <= max_fpr + 1e-12:
            return t
    return np.inf


def threshold_for_youden(scores, y):
    best_t, best_j = np.inf, -np.inf
    for t in np.unique(np.concatenate([scores, [np.inf]])):
        _, _, j = rates(scores >= t, y)
        if j > best_j:
            best_t, best_j = t, j
    return best_t


def roc_points(scores, y):
    out = []
    for t in np.unique(np.concatenate([scores, [np.inf, -np.inf]]))[::-1]:
        tpr, fpr, _ = rates(scores >= t, y)
        out.append((fpr, tpr))
    return np.array(sorted(set(out)))


def auc(scores, y):
    pts = roc_points(scores, y)
    return float(np.trapezoid(pts[:, 1], pts[:, 0])) if hasattr(np, "trapezoid") \
        else float(np.trapz(pts[:, 1], pts[:, 0]))
