"""Regression gate: thresholds per arm in gates.json; any breach fails CI with a readable diff."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT = Path(__file__).resolve().parents[1] / "gates.json"


def check(results: dict[str, Any], gates_path: Path = DEFAULT) -> list[str]:
    gates = json.loads(gates_path.read_text(encoding="utf-8"))
    by_arm = {row["arm"]: row for row in results["leaderboard"]}
    failures = []
    for arm, rules in gates.items():
        row = by_arm.get(arm)
        if row is None:
            failures.append(f"{arm}: arm missing from results")
            continue
        for rule, limit in rules.items():
            kind, metric = rule.split("_", 1)
            value = row[metric]
            if (kind == "min" and value < limit) or (kind == "max" and value > limit):
                failures.append(f"{arm}: {metric} = {value} breaks {kind} {limit}")
    return failures
