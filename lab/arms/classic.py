"""Cheap arms: hand-written keywords, fuzzy nearest neighbour, Naive Bayes, embedding centroids."""
from __future__ import annotations

import difflib
import math
from collections import Counter, defaultdict

import numpy as np

from lab.arms import Prediction, timed
from lab.dataset import Example
from lab.text import fold, hashed_ngrams, tokens

OOS = "out_of_scope"


class KeywordRouter:
    """What most systems start with: keyword lists someone wrote by hand. No training."""
    name = "keywords"
    KEYWORDS = {
        "validate_tax_id": ["cnpj", "cpf", "valid", "digito", "documento"],
        "search_invoices": ["nota", "notas", "invoice", "invoices", "nf"],
        "compute_withholding": ["retencao", "retem", "withholding", "imposto", "tributo", "iss", "liquido"],
        "search_agreements": ["convenio", "convenios", "agreement", "agreements", "repasse", "programa"],
        "flag_risks": ["atrasado", "risco", "overdue", "irregular", "alerta", "vencido"],
        "price_research": ["preco", "custa", "cost", "valor", "price", "paga"],
        "create_purchase_order": ["compra", "comprar", "pedido", "order", "buy", "oc"],
        "run_status": ["status", "run", "execucao", "rodando", "job", "andamento"],
    }

    def fit(self, train: list[Example]) -> None:
        pass

    def _route(self, text: str) -> tuple[str, float]:
        words = set(fold(text).replace("?", " ").split())
        hits = {i: sum(1 for k in kws if k in words) for i, kws in self.KEYWORDS.items()}
        best = max(hits, key=hits.get)
        total = sum(hits.values())
        return (OOS, 0.5) if total == 0 else (best, hits[best] / total)

    def predict(self, text: str) -> Prediction:
        return timed(self.name, self._route, text)


class FuzzyRouter:
    """Nearest training utterance by difflib ratio — the 'deterministic baseline' of mcp-tool-gateway."""
    name = "fuzzy-nn"

    def __init__(self, oos_below: float = 0.45) -> None:
        self.oos_below = oos_below
        self.train: list[tuple[str, str]] = []

    def fit(self, train: list[Example]) -> None:
        self.train = [(fold(e.text), e.intent) for e in train]

    def _route(self, text: str) -> tuple[str, float]:
        q = fold(text)
        score, intent = max((difflib.SequenceMatcher(None, q, t).ratio(), i) for t, i in self.train)
        return (OOS, 1 - score) if score < self.oos_below else (intent, score)

    def predict(self, text: str) -> Prediction:
        return timed(self.name, self._route, text)


def _features(text: str) -> list[str]:
    """Words plus char 4-grams inside each word: 'atasados' still shares grams with 'atrasados'."""
    words = tokens(text)
    return words + [f"#{w[i:i + 4]}" for w in words for i in range(max(1, len(w) - 3))]


class NaiveBayesRouter:
    """Multinomial Naive Bayes over words + char 4-grams, Laplace smoothing; posterior = softmax of log-probs."""
    name = "naive-bayes"

    def fit(self, train: list[Example]) -> None:
        self.counts: dict[str, Counter] = defaultdict(Counter)
        prior = Counter(e.intent for e in train)
        for e in train:
            self.counts[e.intent].update(_features(e.text))
        self.vocab = {w for c in self.counts.values() for w in c}
        self.log_prior = {i: math.log(n / len(train)) for i, n in prior.items()}
        self.totals = {i: sum(c.values()) for i, c in self.counts.items()}

    def _route(self, text: str) -> tuple[str, float]:
        toks = [t for t in _features(text) if t in self.vocab]
        v = len(self.vocab)
        logp = {i: self.log_prior[i] + sum(math.log((self.counts[i][t] + 1) / (self.totals[i] + v)) for t in toks)
                for i in self.log_prior}
        m = max(logp.values())
        z = sum(math.exp(x - m) for x in logp.values())
        best = max(logp, key=logp.get)
        return best, math.exp(logp[best] - m) / z

    def predict(self, text: str) -> Prediction:
        return timed(self.name, self._route, text)


class KNNRouter:
    """k nearest training requests by cosine over hashed char-n-gram embeddings, similarity-weighted vote.
    Nothing similar enough → out_of_scope."""
    name = "embed-knn"

    def __init__(self, k: int = 5, oos_below: float = 0.30) -> None:
        self.k, self.oos_below = k, oos_below

    def fit(self, train: list[Example]) -> None:
        self.vecs = hashed_ngrams([e.text for e in train])
        self.labels = [e.intent for e in train]

    def _route(self, text: str) -> tuple[str, float]:
        sims = self.vecs @ hashed_ngrams([text])[0]
        top = np.argsort(-sims)[: self.k]
        if float(sims[top[0]]) < self.oos_below:
            return OOS, 1 - float(sims[top[0]])
        votes: dict[str, float] = defaultdict(float)
        for i in top:
            votes[self.labels[i]] += max(0.0, float(sims[i]))
        best = max(votes, key=votes.get)
        return best, votes[best] / sum(votes.values())

    def predict(self, text: str) -> Prediction:
        return timed(self.name, self._route, text)
