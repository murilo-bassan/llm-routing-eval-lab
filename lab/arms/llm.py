"""LLM arm (JSON mode) and the cascade that decides when to pay for it.

The LLM arm talks to any OpenAI-compatible endpoint with `LLM_API_KEY` from the
environment. Responses are cached on disk keyed by sha256(model + prompt), so a
benchmark can be re-run and re-analyzed without paying twice — and a cache is
the only form in which LLM results enter this repository's reports: they are
measured, never typed in.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import httpx

from lab.arms import Prediction
from lab.dataset import INTENTS, Example

PROMPT_VERSION = "route-v2"
SYSTEM = """You route requests for an internal operations assistant.
Pick exactly one intent from the list. If the request is not something the
assistant does, pick "out_of_scope" — do not stretch an intent to fit.
Reply with JSON only: {{"intent": "<id>", "confidence": <0..1>}}

Intents:
{intents}

Examples:
{shots}"""


class LLMRouter:
    def __init__(self, base_url: str, api_key: str, model: str, price_in: float = 0.0, price_out: float = 0.0,
                 shots_per_intent: int = 2, cache_dir: Path = Path(".cache/llm"),
                 transport: httpx.BaseTransport | None = None) -> None:
        self.name = f"llm:{model}"
        self.model, self.price_in, self.price_out = model, price_in, price_out   # USD per 1M tokens
        self.shots_per_intent = shots_per_intent
        self.cache_dir = cache_dir
        self._http = httpx.Client(base_url=base_url, timeout=60, transport=transport,
                                  headers={"Authorization": f"Bearer {api_key}"})

    def fit(self, train: list[Example]) -> None:
        shots, per = [], {}
        for e in train:
            if per.get(e.intent, 0) < self.shots_per_intent:
                per[e.intent] = per.get(e.intent, 0) + 1
                shots.append(f'"{e.text}" -> {e.intent}')
        self.system = SYSTEM.format(intents="\n".join(f"- {k}: {v}" for k, v in INTENTS.items()), shots="\n".join(shots))

    def _cache_path(self, text: str) -> Path:
        key = hashlib.sha256(f"{PROMPT_VERSION}|{self.model}|{self.system}|{text}".encode()).hexdigest()
        return self.cache_dir / f"{key}.json"

    def predict(self, text: str) -> Prediction:
        path = self._cache_path(text)
        if path.exists():
            c = json.loads(path.read_text(encoding="utf-8"))
        else:
            started = time.perf_counter()
            r = self._http.post("/chat/completions", json={
                "model": self.model, "temperature": 0, "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": self.system}, {"role": "user", "content": text}]})
            r.raise_for_status()
            data = r.json()
            usage = data.get("usage", {})
            c = {"content": data["choices"][0]["message"]["content"], "latency_ms": (time.perf_counter() - started) * 1000,
                 "tokens_in": usage.get("prompt_tokens", 0), "tokens_out": usage.get("completion_tokens", 0)}
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(c), encoding="utf-8")
        try:
            parsed = json.loads(c["content"])
            intent = parsed["intent"] if parsed.get("intent") in INTENTS else "out_of_scope"
            conf = float(parsed.get("confidence", 0.5))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            intent, conf = "out_of_scope", 0.0     # unparseable output is a routing failure, counted as such
        cost = (c["tokens_in"] * self.price_in + c["tokens_out"] * self.price_out) / 1e6
        return Prediction(intent, conf, c["latency_ms"], cost, c["tokens_in"] + c["tokens_out"], self.name)


class CascadeRouter:
    """Cheap arm first; escalate to the expensive arm only when the cheap one is unsure."""

    def __init__(self, cheap, expensive, threshold: float) -> None:
        self.cheap, self.expensive, self.threshold = cheap, expensive, threshold
        self.name = f"cascade({cheap.name}→{expensive.name}@{threshold:.2f})"
        self.escalations = 0

    def fit(self, train: list[Example]) -> None:
        self.cheap.fit(train)
        self.expensive.fit(train)

    def predict(self, text: str) -> Prediction:
        first = self.cheap.predict(text)
        if first.confidence >= self.threshold:
            return Prediction(first.intent, first.confidence, first.latency_ms, first.cost_usd, first.tokens, self.name)
        self.escalations += 1
        second = self.expensive.predict(text)
        return Prediction(second.intent, second.confidence, first.latency_ms + second.latency_ms,
                          first.cost_usd + second.cost_usd, first.tokens + second.tokens, self.name)
