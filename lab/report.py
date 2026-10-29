"""Standalone HTML scorecard (shared portfolio look, inline SVG, no JS) and a Markdown summary."""
from __future__ import annotations

import html
from typing import Any

PALETTE = ["#c6ff00", "#a58bff", "#ffb020", "#4fd1c5", "#ff4d5e", "#f4f4f0", "#7a7a76"]
CSS = """
:root{--bg:#000;--s:#111113;--s2:#19191c;--l:#222226;--ink:#f5f5f2;--soft:#cfcfca;--m:#92928c;--a:#c6ff00;--v:#a58bff;--d:#ff5c6c}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 Geist,'Segoe UI',system-ui,sans-serif;padding:40px 16px;-webkit-font-smoothing:antialiased}
main{max-width:1120px;margin:0 auto}h1{font:600 32px/1.1 Geist,'Segoe UI',system-ui,sans-serif;letter-spacing:-.03em;margin:10px 0 10px}
.label{font:600 14px Geist,'Segoe UI',system-ui,sans-serif;color:var(--ink);margin-bottom:10px}.top{display:flex;justify-content:space-between;color:var(--m);font-size:13.5px}
.card{border:1px solid var(--l);border-radius:14px;background:var(--s);margin:18px 0;padding:20px;overflow-x:auto}
table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:10px 12px;border-bottom:1px solid var(--l);white-space:nowrap}
th{font:500 13px Geist,'Segoe UI',system-ui,sans-serif;color:var(--m);white-space:normal;vertical-align:bottom}
.mono{font-family:'Geist Mono',ui-monospace,monospace;font-size:13px}.muted{color:var(--m)}.a{color:var(--a)}.d{color:var(--d)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}@media(max-width:900px){.grid{grid-template-columns:1fr}}
.plain{border-color:rgba(198,255,0,.3);background:rgba(198,255,0,.06)}.plain ul{margin:0;padding-left:20px;display:grid;gap:8px;color:var(--soft)}
a{color:var(--a)}.hint{color:var(--m);font-size:13.5px;margin:-4px 0 12px}
"""


def _curve_svg(rows: list[dict[str, Any]], tr=lambda s: s) -> str:
    W, H, P = 520, 300, 40
    x = lambda c: P + c * (W - 2 * P)                       # noqa: E731
    y = lambda a: H - P - (a - 0.5) / 0.5 * (H - 2 * P)    # accuracy axis from 0.5 to 1.0
    parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="coverage vs accuracy">']
    for t in (0.5, 0.75, 1.0):
        parts.append(f'<line x1="{P}" x2="{W - P}" y1="{y(t)}" y2="{y(t)}" stroke="#1c1c1c"/>'
                     f'<text x="{P - 6}" y="{y(t) + 4}" fill="#7a7a76" font-size="10" text-anchor="end" font-family="Geist, sans-serif">{t:.2f}</text>')
    for c in (0, 0.5, 1):
        parts.append(f'<text x="{x(c)}" y="{H - P + 16}" fill="#7a7a76" font-size="10" text-anchor="middle" font-family="Geist, sans-serif">{c:.0%}</text>')
    parts.append(f'<text x="{W - P}" y="{H - 6}" fill="#7a7a76" font-size="10" text-anchor="end" font-family="Geist, sans-serif">{tr("coverage →")}</text>')
    for i, row in enumerate(rows):
        pts = " ".join(f"{x(pt['coverage']):.1f},{y(max(0.5, pt['accuracy'])):.1f}" for pt in row["coverage_curve"])
        color = PALETTE[i % len(PALETTE)]
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.8"/>'
                     f'<text x="{P + 60}" y="{H - P - 14 - (len(rows) - 1 - i) * 13}" fill="{color}" font-size="10.5" font-family="Geist, sans-serif">{html.escape(row["arm"])}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _confusion_svg(m: dict[str, dict[str, int]]) -> str:
    labels = list(m)
    n, cell, left, top = len(labels), 34, 150, 120
    W, H = left + n * cell + 10, top + n * cell + 10
    peak = max(max(r.values()) for r in m.values()) or 1
    parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="confusion matrix">']
    for j, lab in enumerate(labels):
        parts.append(f'<text transform="translate({left + j * cell + cell / 2},{top - 6}) rotate(-50)" fill="#7a7a76" font-size="10" font-family="Geist, sans-serif">{lab}</text>')
    for i, a in enumerate(labels):
        parts.append(f'<text x="{left - 8}" y="{top + i * cell + cell / 2 + 4}" fill="#7a7a76" font-size="10" text-anchor="end" font-family="Geist, sans-serif">{a}</text>')
        for j, b in enumerate(labels):
            v = m[a][b]
            color = "198,255,0" if a == b else "255,77,94"
            alpha = 0.06 + 0.9 * v / peak if v else 0.03
            parts.append(f'<rect x="{left + j * cell}" y="{top + i * cell}" width="{cell - 2}" height="{cell - 2}" fill="rgba({color},{alpha:.2f})"/>')
            if v:
                parts.append(f'<text x="{left + j * cell + cell / 2 - 1}" y="{top + i * cell + cell / 2 + 4}" fill="{"#000" if alpha > .5 else "#f4f4f0"}" font-size="11" text-anchor="middle" font-family="Geist, sans-serif">{v}</text>')
    parts.append("</svg>")
    return "".join(parts)


PT = {
 'Which method picks the right tool?': 'Qual método escolhe a ferramenta certa?',
 'test requests the methods never saw while training': 'pedidos de teste que os métodos nunca viram no treino',
 'for training': 'para treino',
 'In plain words': 'Em poucas palavras',
 'is the best method: right': 'é o melhor método: acerta',
 'of the time.': 'das vezes.',
 'Its lead over': 'A vantagem sobre',
 'could still be luck (chance': 'ainda pode ser sorte (chance de',
 'is unlikely to be luck (chance': 'dificilmente é sorte (chance de',
 'more labeled data would settle it.': 'mais dados rotulados resolveriam.',
 'act on requests they should refuse': 'agem em pedidos que deveriam recusar',
 'of the time, even though their accuracy looks close.': 'das vezes, mesmo com acurácia parecida.',
 'method': 'método', 'right answers (likely range)': 'acertos (faixa provável)', 'balanced score': 'nota equilibrada',
 'refuses what it should': 'recusa o que deve', 'acts when it shouldn\'t': 'age quando não devia', 'overconfidence': 'excesso de confiança',
 'time (ms)': 'tempo (ms)', 'cost per 1,000': 'custo por mil', 'sent to 2nd method': 'enviado ao 2º método',
 'All methods': 'Todos os métodos',
 'Answer fewer, get more right': 'Responder menos, acertar mais',
 'If a method only answers when it is confident, it gets more right. This is where to set the cascade threshold.':
   'Se o método só responde quando está confiante, acerta mais. É aqui que se escolhe o limite da cascata.',
 'What it mixed up': 'O que ele confundiu',
 'Is the difference real, or luck?': 'A diferença é real ou sorte?',
 'compared with': 'comparado com',
 'only the best got it right': 'só o melhor acertou', 'only this one got it right': 'só este acertou', 'chance it is luck': 'chance de ser sorte',
 'Where the best method gets it wrong': 'Onde o melhor método erra',
 'right tool': 'ferramenta certa', 'it chose': 'escolheu', 'confidence': 'confiança',
 'coverage →': 'respondidos →', 'Routing report': 'Relatório de roteamento',
 'The LLM was not tested in this run. Set': 'O LLM não foi testado nesta execução. Defina', 'and': 'e',
 'to include it. No LLM number is ever typed into this report by hand.': 'para incluí-lo. Nenhum número de LLM é digitado à mão neste relatório.',
 'Exact McNemar test: it only looks at the requests that one method got right and the other got wrong.':
   'Teste exato de McNemar: olha só os pedidos que um método acertou e o outro errou.',
 'request': 'pedido',
}


def to_html(r: dict[str, Any], lang: str = "en") -> str:
    e = html.escape
    tr = (lambda s: PT.get(s, s)) if lang == "pt" else (lambda s: s)
    other = '<a href="scorecard.html">EN</a> · <b class=a>PT</b>' if lang == "pt" else '<b class=a>EN</b> · <a href="scorecard.pt.html">PT</a>'
    pct = (lambda v: f"{v * 100:.0f}%".replace(".", ",")) if lang == "pt" else (lambda v: f"{v * 100:.0f}%")
    lb = r["leaderboard"]
    best = next(x for x in lb if x["arm"] == r["best"])
    # the three findings, written for someone who has never heard of McNemar
    runner = next((x for x in lb if x["arm"] != r["best"]), None)
    sig_runner = r["significance_vs_best"].get(runner["arm"]) if runner else None
    risky = [x for x in lb if x["oos_false_accept"] > 0.3]
    plain = [f"<b class=a>{e(best['arm'])}</b> {tr('is the best method: right')} {pct(best['accuracy'])} {tr('of the time.')}"]
    if sig_runner:
        luck = sig_runner["p_value"] >= 0.05
        plain.append(f"{tr('Its lead over')} <b>{e(runner['arm'])}</b> "
                     + (f"{tr('could still be luck (chance')} {sig_runner['p_value']:.2f}); {tr('more labeled data would settle it.')}" if luck
                        else f"{tr('is unlikely to be luck (chance')} {sig_runner['p_value']:.3f}).")
                     )
    if risky:
        plain.append(f"<b class=d>{', '.join(e(x['arm']) for x in risky)}</b> {tr('act on requests they should refuse')} "
                     f"{pct(min(x['oos_false_accept'] for x in risky))}–{pct(max(x['oos_false_accept'] for x in risky))} "
                     f"{tr('of the time, even though their accuracy looks close.')}")
    plain_html = "".join(f"<li>{p}</li>" for p in plain)
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
                f"<p class=muted>{tr('The LLM was not tested in this run. Set')} <span class=mono>LLM_API_KEY</span> {tr('and')} <span class=mono>LAB_LLM_MODEL</span> "
                f"{tr('to include it. No LLM number is ever typed into this report by hand.')}</p>")
    return f"""<!doctype html><html lang={"pt-BR" if lang == "pt" else "en"}><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<link rel=stylesheet href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500&display=swap">
<title>Routing report</title><style>{CSS}</style></head><body><main>
<div class=top><span>routebench · {tr('Routing report')}</span><span>{other}</span></div><h1>{tr('Which method picks the right tool?')}</h1>
<p class=muted>{r['test_size']} {tr('test requests the methods never saw while training')} ({r['train_size']} {tr('for training')}).</p>{llm_note}
<section class="card plain"><div class=label>{tr('In plain words')}</div><ul>{plain_html}</ul></section>
<section class=card><div class=label>{tr('All methods')}</div><table><tr><th>{tr('method')}</th><th>{tr('right answers (likely range)')}</th><th>{tr('balanced score')}</th>
<th>{tr('refuses what it should')}</th><th>{tr("acts when it shouldn't")}</th><th>{tr('overconfidence')}</th><th>{tr('time (ms)')}</th><th>{tr('cost per 1,000')}</th><th>{tr('sent to 2nd method')}</th></tr>{rows}</table></section>
<div class=grid>
<section class=card><div class=label>{tr('Answer fewer, get more right')}</div><p class=hint>{tr('If a method only answers when it is confident, it gets more right. This is where to set the cascade threshold.')}</p>{_curve_svg(lb, tr)}</section>
<section class=card><div class=label>{tr('What it mixed up')} · {e(best['arm'])}</div>{_confusion_svg(best['confusion'])}</section>
</div>
<section class=card><div class=label>{tr('Is the difference real, or luck?')} <span class=muted>({tr('compared with')} {e(r['best'])})</span></div><table>
<p class=hint>{tr('Exact McNemar test: it only looks at the requests that one method got right and the other got wrong.')}</p>
<tr><th>{tr('method')}</th><th>{tr('only the best got it right')}</th><th>{tr('only this one got it right')}</th><th>{tr('chance it is luck')}</th></tr>{sig}</table></section>
<section class=card><div class=label>{tr('Where the best method gets it wrong')}</div><table><tr><th>{tr('request')}</th><th>{tr('right tool')}</th><th>{tr('it chose')}</th><th>{tr('confidence')}</th></tr>{errs}</table></section>
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
