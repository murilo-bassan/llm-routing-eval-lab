"""Runs every arm on the same held-out items and computes the full scorecard."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from lab import metrics
from lab.arms import Prediction, Router
from lab.arms.classic import FuzzyRouter, KeywordRouter, KNNRouter, NaiveBayesRouter
from lab.arms.llm import CascadeRouter, LLMRouter
from lab.dataset import Example, split


@dataclass
class ArmResult:
    name: str
    predictions: list[Prediction]
    scorecard: dict[str, Any] = field(default_factory=dict)


def default_arms() -> list[Router]:
    arms: list[Router] = [KeywordRouter(), FuzzyRouter(), NaiveBayesRouter(), KNNRouter(),
                          CascadeRouter(KeywordRouter(), KNNRouter(), threshold=0.75)]
    if key := os.environ.get("LLM_API_KEY"):
        model = os.environ.get("LAB_LLM_MODEL", "")
        llm = LLMRouter(os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1"), key, model,
                        float(os.environ.get("LAB_PRICE_IN", "0")), float(os.environ.get("LAB_PRICE_OUT", "0")))
        arms += [llm, CascadeRouter(KeywordRouter(), llm, threshold=float(os.environ.get("LAB_CASCADE_TAU", "0.75")))]
    return arms


def score(name: str, test: list[Example], preds: list[Prediction]) -> dict[str, Any]:
    y = [e.intent for e in test]
    p = [x.intent for x in preds]
    correct = [a == b for a, b in zip(y, p)]
    conf = [min(1.0, max(0.0, x.confidence)) for x in preds]
    lat = [x.latency_ms for x in preds]
    return {
        "arm": name, "n": len(test),
        "accuracy": round(metrics.accuracy(y, p), 3), "accuracy_ci95": metrics.bootstrap_ci(correct),
        "macro_f1": round(metrics.macro_f1(y, p), 3), **metrics.oos_metrics(y, p),
        "ece": metrics.ece(correct, conf),
        "latency_p50_ms": metrics.percentile(lat, 0.5), "latency_p95_ms": metrics.percentile(lat, 0.95),
        "cost_per_1k_usd": round(sum(x.cost_usd for x in preds) / len(preds) * 1000, 4),
        "per_class": metrics.per_class(y, p), "confusion": metrics.confusion(y, p),
        "coverage_curve": metrics.coverage_curve(correct, conf),
    }


def run(arms: list[Router] | None = None) -> dict[str, Any]:
    train, test = split()
    results: list[ArmResult] = []
    for arm in arms or default_arms():
        arm.fit(train)
        preds = [arm.predict(e.text) for e in test]
        results.append(ArmResult(arm.name, preds, score(arm.name, test, preds)))
        if isinstance(arm, CascadeRouter):
            results[-1].scorecard["escalation_rate"] = round(arm.escalations / len(test), 3)

    best = max(results, key=lambda r: r.scorecard["macro_f1"])
    truth = [e.intent for e in test]
    best_correct = [p.intent == t for p, t in zip(best.predictions, truth)]
    significance = {r.name: metrics.mcnemar(best_correct, [p.intent == t for p, t in zip(r.predictions, truth)])
                    for r in results if r is not best}
    errors = [{"text": e.text, "truth": e.intent, "predicted": p.intent, "confidence": round(p.confidence, 3)}
              for e, p in zip(test, best.predictions) if p.intent != e.intent]
    return {"train_size": len(train), "test_size": len(test), "split": "by template (no paraphrase leakage)",
            "leaderboard": [r.scorecard for r in sorted(results, key=lambda r: -r.scorecard["macro_f1"])],
            "best": best.name, "significance_vs_best": significance, "best_errors": errors,
            "llm_included": any(r.name.startswith("llm") for r in results)}
