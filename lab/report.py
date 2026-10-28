"""Standalone HTML scorecard (VOID theme, inline SVG, no JS) and a Markdown summary."""
from __future__ import annotations

import html
from typing import Any

PALETTE = ["#c6ff00", "#a58bff", "#ffb020", "#4fd1c5", "#ff4d5e", "#f4f4f0", "#7a7a76"]
CSS = """
:root{--bg:#000;--s:#060606;--l:#1c1c1c;--ink:#f4f4f0;--m:#7a7a76;--a:#c6ff00;--v:#a58bff;--d:#ff4d5e}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 Inter,system-ui,sans-serif;padding:40px 16px}
main{max-width:1120px;margin:0 auto}h1{font:600 34px/1.1 'Space Grotesk',system-ui;letter-spacing:-.02em;margin:6px 0 8px}
.label{font:500 10.5px ui-monospace,monospace;letter-spacing:.14em;text-transform:uppercase;color:var(--m)}
.card{border:1px solid var(--l);border-radius:3px;background:var(--s);margin:18px 0;padding:18px;overflow-x:auto}
table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:7px 10px;border-bottom:1px solid var(--l);white-space:nowrap}
th{font:500 10.5px ui-monospace,monospace;letter-spacing:.12em;text-transform:uppercase;color:var(--m)}
.mono{font-family:ui-monospace,monospace;font-size:12.5px}.muted{color:var(--m)}.a{color:var(--a)}.d{color:var(--d)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}@media(max-width:900px){.grid{grid-template-columns:1fr}}
"""


def _curve_svg(rows: list[dict[str, Any]]) -> str:
    W, H, P = 520, 300, 40
    x = lambda c: P + c * (W - 2 * P)                       # noqa: E731
    y = lambda a: H - P - (a - 0.5) / 0.5 * (H - 2 * P)    # accuracy axis from 0.5 to 1.0
    parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="coverage vs accuracy">']
    for t in (0.5, 0.75, 1.0):
        parts.append(f'<line x1="{P}" x2="{W - P}" y1="{y(t)}" y2="{y(t)}" stroke="#1c1c1c"/>'
                     f'<text x="{P - 6}" y="{y(t) + 4}" fill="#7a7a76" font-size="10" text-anchor="end" font-family="monospace">{t:.2f}</text>')
    for c in (0, 0.5, 1):
        parts.append(f'<text x="{x(c)}" y="{H - P + 16}" fill="#7a7a76" font-size="10" text-anchor="middle" font-family="monospace">{c:.0%}</text>')
    parts.append(f'<text x="{W - P}" y="{H - 6}" fill="#7a7a76" font-size="10" text-anchor="end" font-family="monospace">coverage →</text>')
    for i, row in enumerate(rows):
        pts = " ".join(f"{x(pt['coverage']):.1f},{y(max(0.5, pt['accuracy'])):.1f}" for pt in row["coverage_curve"])
        color = PALETTE[i % len(PALETTE)]
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.8"/>'
                     f'<text x="{P + 60}" y="{H - P - 14 - (len(rows) - 1 - i) * 13}" fill="{color}" font-size="10.5" font-family="monospace">{html.escape(row["arm"])}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _confusion_svg(m: dict[str, dict[str, int]]) -> str:
    labels = list(m)
    n, cell, left, top = len(labels), 34, 150, 120
    W, H = left + n * cell + 10, top + n * cell + 10
    peak = max(max(r.values()) for r in m.values()) or 1
    parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="confusion matrix">']
    for j, lab in enumerate(labels):
        parts.append(f'<text transform="translate({left + j * cell + cell / 2},{top - 6}) rotate(-50)" fill="#7a7a76" font-size="10" font-family="monospace">{lab}</text>')
    for i, a in enumerate(labels):
        parts.append(f'<text x="{left - 8}" y="{top + i * cell + cell / 2 + 4}" fill="#7a7a76" font-size="10" text-anchor="end" font-family="monospace">{a}</text>')
        for j, b in enumerate(labels):
            v = m[a][b]
            color = "198,255,0" if a == b else "255,77,94"
            alpha = 0.06 + 0.9 * v / peak if v else 0.03
            parts.append(f'<rect x="{left + j * cell}" y="{top + i * cell}" width="{cell - 2}" height="{cell - 2}" fill="rgba({color},{alpha:.2f})"/>')
            if v:
                parts.append(f'<text x="{left + j * cell + cell / 2 - 1}" y="{top + i * cell + cell / 2 + 4}" fill="{"#000" if alpha > .5 else "#f4f4f0"}" font-size="11" text-anchor="middle" font-family="monospace">{v}</text>')
    parts.append("</svg>")
    return "".join(parts)


def to_html(r: dict[str, Any]) -> str:
    e = html.escape
    lb = r["leaderboard"]
    best = next(x for x in lb if x["arm"] == r["best"])
    rows = "".join(
        f"<tr><td class=mono>{e(x['arm'])}</td><td class=mono>{x['accuracy']:.3f} <span class=muted>[{x['accuracy_ci95'][0]:.2f}–{x['accuracy_ci95'][1]:.2f}]</span></td>"
        f"<td class='mono {'a' if x['arm'] == r['best'] else ''}'>{x['macro_f1']:.3f}</td><td class=mono>{x['oos_recall']:.2f}</td>"
        f"<td class='mono {'d' if x['oos_false_accept'] > 0.3 else ''}'>{x['oos_false_accept']:.2f}</td><td class=mono>{x['ece']:.3f}</td>"
        f"<td class=mono>{x['latency_p50_ms']:.2f} / {x['latency_p95_ms']:.2f}</td><td class=mono>${x['cost_per_1k_usd']:.4f}</td>"
        f"<td class=mono>{x.get('escalation_rate', '—')}</td></tr>" for x in lb)
    sig = "".join(f"<tr><td class=mono>{e(arm)}</td><td class=mono>{s['a_only']}</td><td class=mono>{s['b_only']}</td>"
                  f"<td class='mono {'a' if s['p_value'] < 0.05 else 'muted'}'>{s['p_value']:.4f}</td></tr>"
                  for arm, s in r["significance_vs_best"].items())
    errs = "".join(f"<tr><td>{e(x['text'])}</td><td class=mono>{x['truth']}</td><td class='mono d'>{x['predicted']}</td>"
                   f"<td class=mono>{x['confidence']}</td></tr>" for x in r["best_errors"][:15])
    llm_note = ("" if r["llm_included"] else
                "<p class=muted>LLM arms not run: set <span class=mono>LLM_API_KEY</span> and <span class=mono>LAB_LLM_MODEL</span> "
                "to include them. No LLM numbers are ever written by hand into this report.</p>")
    return f"""<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Routing benchmark</title><style>{CSS}</style></head><body><main>
<div class=label>llm-routing-eval-lab · scorecard</div><h1>Who routes best — and how sure is it?</h1>
<p class=muted>{r['test_size']} held-out requests ({r['train_size']} train), split {e(r['split'])}. Best by macro-F1:
<span class='mono a'>{e(r['best'])}</span>.</p>{llm_note}
<section class=card><div class=label>leaderboard</div><table><tr><th>arm</th><th>accuracy [95% CI]</th><th>macro-F1</th>
<th>OOS recall</th><th>OOS false-accept</th><th>ECE</th><th>latency p50/p95 ms</th><th>cost / 1k</th><th>escalated</th></tr>{rows}</table></section>
<div class=grid>
<section class=card><div class=label>coverage vs accuracy — pick the cascade threshold here</div>{_curve_svg(lb)}</section>
<section class=card><div class=label>confusion — {e(best['arm'])}</div>{_confusion_svg(best['confusion'])}</section>
</div>
<section class=card><div class=label>is the difference real? exact McNemar vs {e(r['best'])}</div><table>
<tr><th>arm</th><th>only best right</th><th>only this right</th><th>p-value</th></tr>{sig}</table></section>
<section class=card><div class=label>where the best arm fails</div><table><tr><th>request</th><th>truth</th><th>predicted</th><th>conf</th></tr>{errs}</table></section>
</main></body></html>"""


def to_markdown(r: dict[str, Any]) -> str:
    lines = [f"Held-out: {r['test_size']} requests · split {r['split']}", "",
             "| arm | accuracy [95% CI] | macro-F1 | OOS recall | OOS false-accept | ECE | p50 ms | escalated |",
             "|---|---|---|---|---|---|---|---|"]
    for x in r["leaderboard"]:
        lines.append(f"| {x['arm']} | {x['accuracy']:.3f} [{x['accuracy_ci95'][0]:.2f}–{x['accuracy_ci95'][1]:.2f}] | "
                     f"{x['macro_f1']:.3f} | {x['oos_recall']:.2f} | {x['oos_false_accept']:.2f} | {x['ece']:.3f} | "
                     f"{x['latency_p50_ms']:.2f} | {x.get('escalation_rate', '—')} |")
    return "\n".join(lines)
