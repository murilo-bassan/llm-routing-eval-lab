import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab import cli, dataset, gate, metrics, report, runner  # noqa: E402
from lab.arms import Prediction  # noqa: E402
from lab.arms.classic import KeywordRouter, KNNRouter, NaiveBayesRouter  # noqa: E402
from lab.arms.llm import CascadeRouter, LLMRouter  # noqa: E402


# ------------------------------------------------------------------ dataset

def test_split_is_by_template_without_leakage():
    train, test = dataset.split()
    assert not {e.template for e in train} & {e.template for e in test}
    assert {e.intent for e in test} == set(dataset.INTENTS)          # every class is evaluated
    assert len(test) / (len(train) + len(test)) == pytest.approx(0.3, abs=0.08)


def test_dataset_is_deterministic():
    assert dataset.examples() == dataset.examples.__wrapped__()


# ------------------------------------------------------------------ metrics

def test_mcnemar_exact():
    a = [True] * 10 + [False] * 0
    b = [False] * 10
    assert metrics.mcnemar(a, b) == {"a_only": 10, "b_only": 0, "p_value": round(2 / 2 ** 10, 5)}
    assert metrics.mcnemar([True, False], [False, True])["p_value"] == 1.0


def test_ece_perfect_and_overconfident():
    assert metrics.ece([True, True, False, False], [1.0, 1.0, 0.0, 0.0]) == 0
    assert metrics.ece([False] * 10, [0.95] * 10) == pytest.approx(0.95)


def test_bootstrap_ci_brackets_the_mean():
    correct = [True] * 80 + [False] * 20
    lo, hi = metrics.bootstrap_ci(correct)
    assert lo < 0.8 < hi and hi - lo < 0.2


def test_oos_metrics():
    y = ["out_of_scope", "out_of_scope", "a", "a"]
    p = ["out_of_scope", "a", "out_of_scope", "a"]
    assert metrics.oos_metrics(y, p) == {"oos_recall": 0.5, "oos_false_accept": 0.5, "in_scope_rejected": 0.5}


def test_macro_f1_punishes_ignoring_a_class():
    y = ["a"] * 9 + ["b"]
    p = ["a"] * 10
    assert metrics.accuracy(y, p) == 0.9 and metrics.macro_f1(y, p) < 0.5


def test_coverage_curve_is_monotone_in_coverage():
    curve = metrics.coverage_curve([True, False, True, True], [0.9, 0.2, 0.6, 0.4])
    cov = [c["coverage"] for c in curve]
    assert cov == sorted(cov, reverse=True) and curve[0]["coverage"] == 1.0


# ------------------------------------------------------------------ arms

@pytest.fixture(scope="module")
def train():
    return dataset.split()[0]


def test_keyword_router():
    k = KeywordRouter()
    assert k.predict("valida o cnpj 11.222.333/0001-81").intent == "validate_tax_id"
    assert k.predict("me conta uma piada").intent == "out_of_scope"


def test_learned_arms_tolerate_typos(train):
    for arm in (KNNRouter(), NaiveBayesRouter()):
        arm.fit(train)
        assert arm.predict("quais convenios estao atasados em GO").intent in {"flag_risks", "search_agreements"}


def test_cascade_escalates_only_when_unsure(train):
    calls = []

    class Spy:
        name = "spy"

        def fit(self, t):
            pass

        def predict(self, text):
            calls.append(text)
            return Prediction("run_status", 0.9)

    c = CascadeRouter(KeywordRouter(), Spy(), threshold=0.75)
    c.fit(train)
    assert c.predict("valida o cpf 529.982.247-25").intent == "validate_tax_id" and calls == []
    assert c.predict("cadê o resultado").intent == "run_status" and len(calls) == 1


def test_llm_arm_parses_caches_and_costs(tmp_path, train):
    replies = iter([{"intent": "price_research", "confidence": 0.92}, "not json"])
    hits = []

    def handler(request):
        hits.append(json.loads(request.content))
        content = next(replies)
        return httpx.Response(200, json={"choices": [{"message": {"content": content if isinstance(content, str) else json.dumps(content)}}],
                                         "usage": {"prompt_tokens": 400, "completion_tokens": 12}})

    arm = LLMRouter("https://llm.example", "k", "m", price_in=1.0, price_out=4.0, cache_dir=tmp_path,
                    transport=httpx.MockTransport(handler))
    arm.fit(train)
    p = arm.predict("quanto custa café?")
    assert p.intent == "price_research" and p.cost_usd == pytest.approx((400 + 48) / 1e6)
    assert arm.predict("quanto custa café?").intent == "price_research" and len(hits) == 1   # cached
    bad = arm.predict("outra coisa")
    assert bad.intent == "out_of_scope" and bad.confidence == 0.0       # unparseable counts as a failure
    assert "out_of_scope" in hits[0]["messages"][0]["content"]


# ------------------------------------------------------------------ end to end

def test_benchmark_and_gate(tmp_path):
    results = runner.run()
    names = [r["arm"] for r in results["leaderboard"]]
    assert names[0] == results["best"] and not results["llm_included"]
    assert gate.check(results) == []
    strict = tmp_path / "g.json"
    strict.write_text(json.dumps({"keywords": {"min_macro_f1": 0.99}, "ghost": {"min_accuracy": 0.1}}))
    failures = gate.check(results, strict)
    assert any("keywords: macro_f1" in f for f in failures) and any("ghost" in f for f in failures)


def test_cli_writes_report_and_gates(tmp_path):
    code = cli.main(["--html", str(tmp_path / "r.html"), "--json", str(tmp_path / "r.json"), "--gate"])
    assert code == 0
    html = (tmp_path / "r.html").read_text(encoding="utf-8")
    assert "<svg" in html and "McNemar" in html and "<script" not in html
    assert "LLM arms not run" in html          # no LLM numbers without a real run
    pt = (tmp_path / "r.pt.html").read_text(encoding="utf-8")
    assert "Quem roteia melhor" in pt and "Braços com LLM não rodaram" in pt and 'href="scorecard.html"' in pt
    assert report.to_markdown(json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))).startswith("Held-out")
