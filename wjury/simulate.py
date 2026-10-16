"""SIMULATED judges, for testing the pipeline end to end. NOT RESULTS.

These fake judges look at the true label (which real judges never see) and
answer with chosen error rates, so the analysis code can be exercised and
checked before any real model is run. Their profiles imitate the kinds of
judges visible in the paper's figures:

  sim-strong-a / sim-strong-b  good judges that still let many bugs through
  sim-rubber-stamp             says CORRECT to almost everything
  sim-random-veto              vetoes good and bad queries at a similar rate
  sim-medium                   in between

Errors are correlated across judges (some bugs fool everyone), as they are
for real models. Anything produced with these judges is written to a
separate folder and labeled SIMULATED.
"""

from __future__ import annotations

import hashlib
import json
import random
from statistics import NormalDist

_N = NormalDist()

# P(says CORRECT | query correct), P(says CORRECT | query wrong), confidence skill, latency (s)
PROFILES = {
    "sim-strong-a":     (0.96, 0.45, 0.8, 9.0),
    "sim-strong-b":     (0.94, 0.50, 0.7, 4.0),
    "sim-medium":       (0.90, 0.65, 0.5, 3.0),
    "sim-rubber-stamp": (0.99, 0.95, 0.1, 2.0),
    "sim-random-veto":  (0.72, 0.68, 0.0, 6.0),
}
SIM_MODELS = list(PROFILES)

# Some bug types fool judges more often (multiplies the false-accept odds).
HARDNESS = {"null_handling": 1.8, "join_type": 1.6, "distinct_counting": 1.5, "count_star": 1.6,
            "window_frame": 2.0, "ties_dropped": 2.0, "wrong_boundary": 1.3,
            "missing_filter": 0.5, "wrong_literal": 0.7}


def _u(*parts) -> float:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return (int.from_bytes(h[:8], "big") + 0.5) / 2**64


def _odds_scale(p: float, k: float) -> float:
    o = p / (1 - p) * k
    return o / (1 + o)


class SimulatedJudge:
    def __init__(self, model: str):
        if model not in PROFILES:
            raise SystemExit(f"unknown simulated judge {model!r}; choose from {SIM_MODELS}")
        self.model = model
        self.tpr, self.fpr, self.skill, self.latency = PROFILES[model]

    def __call__(self, cand: dict) -> dict:
        label = cand["label"]
        p_yes = self.tpr if label else _odds_scale(self.fpr, HARDNESS.get(cand.get("bug_type"), 1.0))
        # correlated latent: a shared "this one looks fine" factor plus judge-specific noise
        shared = _N.inv_cdf(_u("shared", cand["cid"]))
        own = _N.inv_cdf(_u(self.model, cand["cid"]))
        z = 0.6 * shared + 0.8 * own
        says_correct = _N.cdf(z) < p_yes
        right = says_correct == bool(label)
        rng = random.Random(_u("conf", self.model, cand["cid"]))
        base = 0.85 + rng.uniform(-0.1, 0.12)
        conf = base + (0.08 if right else -0.12) * self.skill + rng.gauss(0, 0.04)
        conf = int(round(100 * min(max(conf, 0.5), 1.0)))
        text = json.dumps({"verdict": "CORRECT" if says_correct else "INCORRECT", "confidence": conf})
        return {"text": text, "prompt_tokens": 700, "completion_tokens": 20,
                "latency_s": self.latency * (0.8 + 0.4 * rng.random())}
