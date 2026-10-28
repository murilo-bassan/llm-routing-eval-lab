# ADR 0001 — How routers are compared

**Status:** accepted

## Context
Routing benchmarks are easy to make flattering: split by row so paraphrases leak
into the test set, report accuracy only, ignore the out-of-scope class, compare two
numbers that differ by noise, and quote an LLM score someone ran once by hand.

## Decision
1. **Split by template.** A held-out request never shares a paraphrase template with training.
2. **Out-of-scope is a class**, and its *false-accept rate* is reported separately: routing
   "me conta uma piada" to `create_purchase_order` is the costly mistake.
3. **Macro-F1** ranks arms (accuracy hides ignored classes); **95% bootstrap CIs** on accuracy.
4. **Exact McNemar** between the best arm and each other arm, on the same items:
   a gap that is not significant is reported as such.
5. **Calibration (ECE) and the coverage/accuracy curve** — confidence is what a cascade
   threshold is chosen from, so it is measured, not assumed.
6. **LLM numbers are measured or absent.** The LLM arm runs only with a key; responses are
   cached by `sha256(prompt version, model, prompt, text)` so re-analysis is free and
   reproducible. Nothing is pasted into reports by hand.
7. **A regression gate** (`gates.json`) fails CI when a chosen arm drops below its floor.

## Consequences
- On the offline arms the honest finding is that a cheap **keywords → kNN cascade** beats
  every single arm (macro-F1 0.80 vs 0.73), escalating a third of the traffic — and that
  two learned arms that look "fine" on accuracy accept **85–100%** of out-of-scope requests.
- The same harness answers the expensive question — "is the LLM worth it, and for which
  third of the traffic?" — the day a key is configured, without changing code.
