<div align="center">

# routebench · llm-routing-eval-lab

**Which router should send a request to which tool — and when is an LLM worth paying for?**
A benchmark harness that reports what accuracy hides: out-of-scope false-accepts, calibration,
confidence intervals, statistical significance and the cascade threshold.

![python](https://img.shields.io/badge/python-3.11%2B-000?style=flat-square&labelColor=000&color=c6ff00)
![stats](https://img.shields.io/badge/McNemar%20%C2%B7%20bootstrap%20%C2%B7%20ECE-000?style=flat-square&labelColor=000&color=a58bff)
![tests](https://img.shields.io/badge/tests-14%20%2B%20CI%20gate-000?style=flat-square&labelColor=000&color=c6ff00)

**English** · [Português](README.pt-BR.md)

<img src="docs/screenshots/scorecard-top.png" alt="scorecard" width="100%">

</div>

---

## The question

Every tool-using assistant routes first: *which tool does this request need — or none?*
Teams usually pick between hand-written keywords, a small classifier and an LLM call by
gut feeling. This lab makes it a measured decision, with the methodology a reviewer
would ask for ([ADR 0001](docs/adr/0001-how-to-compare-routers.md)).

## Arms

| arm | how | cost |
|---|---|---|
| `keywords` | hand-written keyword lists — what most systems start with | free |
| `fuzzy-nn` | difflib nearest training request — the `mcp-tool-gateway` baseline | free |
| `naive-bayes` | multinomial NB over words + char 4-grams (typo-tolerant) | free |
| `embed-knn` | cosine kNN over hashed char-n-gram embeddings, similarity-gated out-of-scope | free |
| `cascade(a→b@τ)` | cheap arm answers when confident ≥ τ, otherwise escalates | mixed |
| `llm:<model>` | any OpenAI-compatible model, JSON mode, few-shot, **disk-cached** | per token |

## Results (offline arms, held-out templates — `python -m lab.cli`)

| arm | accuracy [95% CI] | macro-F1 | OOS false-accept | ECE | escalated |
|---|---|---|---|---|---|
| **cascade(keywords→embed-knn@0.75)** | **0.815** [0.73–0.89] | **0.802** | 0.00 | 0.133 | 32% |
| keywords | 0.759 [0.68–0.84] | 0.730 | 0.00 | 0.079 | — |
| fuzzy-nn | 0.694 [0.61–0.78] | 0.654 | **0.85** | 0.128 | — |
| naive-bayes | 0.620 [0.53–0.71] | 0.585 | **1.00** | 0.212 | — |
| embed-knn | 0.593 [0.50–0.69] | 0.598 | 0.00 | 0.220 | — |

What the scorecard says that a single accuracy number would not:
- The cascade beats each of its parts, sending only a third of traffic to the second arm.
- Its lead over plain keywords is **not statistically significant** on 108 items
  (McNemar p = 0.18) — more labeled data is needed before claiming it.
- Two learned arms route **85–100% of out-of-scope requests to real actions**; on accuracy
  alone they look merely "a bit worse".
- With a key configured, the same run adds `llm:<model>` and `cascade(keywords→llm)`, with
  cost per 1k requests — the numbers that decide whether the LLM earns its price.

> **No LLM result in this repository was typed by hand.** LLM arms run only when
> `LLM_API_KEY` is in your environment (or a CI secret); their responses are cached by
> prompt hash so analyses are reproducible.

## Methodology highlights
- **Template-level split** — no paraphrase leakage into the test set.
- **Out-of-scope as a class**, with false-accept rate reported separately.
- **Exact McNemar test** vs the best arm; **bootstrap 95% CIs**; **ECE** calibration.
- **Coverage/accuracy curves** — where to set τ for a cascade.
- **Regression gate**: `gates.json` floors per arm; `python -m lab.cli --gate` exits 1 on a breach (CI).
- Self-contained **HTML scorecard** (inline SVG, no JS), in English and Portuguese, uploaded as a CI artifact.

## Quick start

```bash
git clone https://github.com/murilo-bassan/llm-routing-eval-lab.git && cd llm-routing-eval-lab
pip install -r requirements-dev.txt
python -m pytest
python -m lab.cli --gate          # → reports/scorecard.html and reports/scorecard.pt.html
# with an LLM:  LLM_API_KEY=... LAB_LLM_MODEL=... LAB_PRICE_IN=... LAB_PRICE_OUT=... python -m lab.cli
```

<img src="docs/screenshots/scorecard.png" alt="full scorecard" width="100%">

---

<sub>Part of an AI engineering portfolio — see [ai-engineering-portfolio](https://github.com/murilo-bassan/ai-engineering-portfolio). Dataset is synthetic. MIT licensed.</sub>
