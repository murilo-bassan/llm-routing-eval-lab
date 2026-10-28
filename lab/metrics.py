"""Metrics that survive a skeptical reviewer.

Accuracy alone hides the failure that matters (out-of-scope accepted as an
action), says nothing about confidence quality, and ignores noise. So:
macro-F1 and per-class scores, out-of-scope recall and false-accept rate,
expected calibration error, bootstrap confidence intervals, an exact McNemar
test between two arms on the same items, and the coverage/accuracy curve that
picks a cascade threshold.
"""
from __future__ import annotations

import math
import random
from collections import Counter

OOS = "out_of_scope"


def accuracy(y: list[str], p: list[str]) -> float:
    return sum(a == b for a, b in zip(y, p)) / len(y)


def per_class(y: list[str], p: list[str]) -> dict[str, dict[str, float]]:
    out = {}
    for c in sorted(set(y) | set(p)):
        tp = sum(a == c and b == c for a, b in zip(y, p))
        fp = sum(a != c and b == c for a, b in zip(y, p))
        fn = sum(a == c and b != c for a, b in zip(y, p))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[c] = {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3), "support": tp + fn}
    return out


def macro_f1(y: list[str], p: list[str]) -> float:
    pc = per_class(y, p)
    classes = [c for c in pc if pc[c]["support"] > 0]
    return sum(pc[c]["f1"] for c in classes) / len(classes)


def oos_metrics(y: list[str], p: list[str]) -> dict[str, float]:
    oos = [b for a, b in zip(y, p) if a == OOS]
    ins = [b for a, b in zip(y, p) if a != OOS]
    return {"oos_recall": round(sum(b == OOS for b in oos) / len(oos), 3) if oos else float("nan"),
            # an out-of-scope request routed to a real action: the expensive mistake
            "oos_false_accept": round(sum(b != OOS for b in oos) / len(oos), 3) if oos else float("nan"),
            "in_scope_rejected": round(sum(b == OOS for b in ins) / len(ins), 3) if ins else float("nan")}


def confusion(y: list[str], p: list[str]) -> dict[str, dict[str, int]]:
    labels = sorted(set(y) | set(p))
    m = {a: {b: 0 for b in labels} for a in labels}
    for a, b in zip(y, p):
        m[a][b] += 1
    return m


def ece(correct: list[bool], conf: list[float], bins: int = 10) -> float:
    """Expected calibration error: |accuracy − confidence| averaged over confidence bins, weighted by bin size."""
    total, n = 0.0, len(correct)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf) if lo <= c < hi or (b == bins - 1 and c == 1.0)]
        if idx:
            acc = sum(correct[i] for i in idx) / len(idx)
            avg = sum(conf[i] for i in idx) / len(idx)
            total += len(idx) / n * abs(acc - avg)
    return round(total, 4)


def bootstrap_ci(correct: list[bool], iters: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(correct)
    stats = sorted(sum(correct[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return round(stats[int(iters * alpha / 2)], 3), round(stats[int(iters * (1 - alpha / 2)) - 1], 3)


def mcnemar(a_correct: list[bool], b_correct: list[bool]) -> dict[str, float]:
    """Exact two-sided McNemar test on the discordant pairs. p < 0.05 → the arms really differ on this set."""
    b = sum(x and not y for x, y in zip(a_correct, b_correct))   # A right, B wrong
    c = sum(y and not x for x, y in zip(a_correct, b_correct))   # B right, A wrong
    n = b + c
    if n == 0:
        return {"a_only": 0, "b_only": 0, "p_value": 1.0}
    k = min(b, c)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"a_only": b, "b_only": c, "p_value": round(p, 5)}


def coverage_curve(correct: list[bool], conf: list[float], steps: int = 20) -> list[dict[str, float]]:
    """For each threshold τ: share of requests answered with confidence ≥ τ, and accuracy on those."""
    out = []
    for i in range(steps + 1):
        tau = i / steps
        kept = [c for c, s in zip(correct, conf) if s >= tau]
        out.append({"threshold": tau, "coverage": round(len(kept) / len(correct), 3),
                    "accuracy": round(sum(kept) / len(kept), 3) if kept else 1.0})
    return out


def percentile(values: list[float], q: float) -> float:
    s = sorted(values)
    return round(s[min(len(s) - 1, int(q * len(s)))], 3) if s else 0.0


def label_balance(y: list[str]) -> dict[str, int]:
    return dict(Counter(y))
