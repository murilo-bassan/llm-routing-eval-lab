"""Routing arms. Every arm returns an intent *and a confidence*, because a router
that cannot say how sure it is cannot be put in a cascade or given a threshold."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from lab.dataset import Example


@dataclass(frozen=True)
class Prediction:
    intent: str
    confidence: float
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    tokens: int = 0
    arm: str = ""


class Router(Protocol):
    name: str

    def fit(self, train: list[Example]) -> None: ...

    def predict(self, text: str) -> Prediction: ...


def timed(name: str, fn, text: str) -> Prediction:
    started = time.perf_counter()
    intent, conf = fn(text)
    return Prediction(intent, float(conf), (time.perf_counter() - started) * 1000, arm=name)
