"""python -m lab.cli [--html reports/scorecard.html] [--json reports/results.json] [--gate]

Exit code 1 when --gate is set and any threshold in gates.json is breached.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lab import gate, report, runner


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", type=Path, default=Path("reports/scorecard.html"))
    ap.add_argument("--json", type=Path, default=Path("reports/results.json"))
    ap.add_argument("--gate", action="store_true")
    args = ap.parse_args(argv)

    results = runner.run()
    args.html.parent.mkdir(parents=True, exist_ok=True)
    args.html.write_text(report.to_html(results), encoding="utf-8")
    args.html.with_name(args.html.stem + ".pt.html").write_text(report.to_html(results, "pt"), encoding="utf-8")
    args.json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(report.to_markdown(results))
    print(f"\nscorecard → {args.html}")
    if args.gate:
        failures = gate.check(results)
        for f in failures:
            print(f"GATE FAILED: {f}", file=sys.stderr)
        if failures:
            return 1
        print("gates passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
