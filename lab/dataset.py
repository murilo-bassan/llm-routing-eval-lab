"""A labeled routing dataset: utterance → intent, PT and EN, synthetic.

Utterances come from paraphrase *templates* filled with slot values, then
roughened (typos, casing, missing accents). The train/test split is done **by
template**, never by row: a test sentence never shares a template with any
training sentence. Splitting by row would let a classifier memorize templates
and report accuracy it will not have in production — the most common leak in
intent benchmarks.

`out_of_scope` is a real class. A router that never says "I don't know" looks
great on in-scope accuracy and terrible in production.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache

INTENTS: dict[str, str] = {
    "validate_tax_id": "Check whether a CPF/CNPJ is valid",
    "search_invoices": "Find invoices by supplier, item or date",
    "compute_withholding": "Compute taxes withheld on an invoice",
    "search_agreements": "Find public-transfer agreements",
    "flag_risks": "List agreements with risk signals (late, low execution)",
    "price_research": "Estimate what the public sector pays for a product",
    "create_purchase_order": "Create a purchase order (a write action)",
    "run_status": "Check the status of an agent run or job",
    "out_of_scope": "Anything the system does not do",
}

TEMPLATES: dict[str, list[str]] = {
    "validate_tax_id": [
        "valida o cnpj {cnpj}", "esse CPF {cpf} é válido?", "checar se o documento {cnpj} existe",
        "is tax id {cnpj} valid", "confere o dígito verificador do {cpf}", "verify CPF {cpf} please",
        "o cnpj {cnpj} tá certo?", "quero saber se {cpf} é um cpf válido", "validate this cnpj: {cnpj}",
        "pode conferir o documento {cpf} do fornecedor?",
    ],
    "search_invoices": [
        "notas fiscais do fornecedor {supplier}", "buscar notas de {item} em {month}", "find invoices from {supplier}",
        "quais notas a {supplier} emitiu?", "lista as NF de {item}", "show me invoices for {item} issued in {month}",
        "procura nota fiscal de {item} do fornecedor {supplier}", "tem nota da {supplier} em {month}?",
        "invoices for {item} please", "me mostra as notas de serviço de {month}",
    ],
    "compute_withholding": [
        "calcula a retenção da nota {nf}", "quanto de imposto retém na {nf}?", "compute withholding for invoice {nf}",
        "qual o ISS retido na nota {nf}", "taxes withheld on {nf}?", "valor líquido a pagar da {nf}",
        "faz o cálculo dos tributos da nota {nf}", "quanto sobra pra pagar na {nf} depois dos impostos",
        "withholding breakdown for {nf}", "retenções da nf {nf}",
    ],
    "search_agreements": [
        "convênios de {town}", "quais convênios existem em {state}?", "find agreements in {town}",
        "lista os convênios do programa {program}", "acordos de repasse para {town}", "agreements for program {program}",
        "me mostra os convênios ativos em {state}", "tem convênio com {town}?", "transferências do programa {program} em {state}",
        "search grants in {state}",
    ],
    "flag_risks": [
        "quais convênios estão atrasados em {state}?", "convênios com risco em {town}", "which agreements are overdue in {state}",
        "lista os convênios com baixa execução", "tem algum convênio irregular em {state}?", "risk flags for agreements in {state}",
        "prestação de contas atrasada em {town}", "quais repasses estão em situação de alerta?", "late reports in {state}",
        "convênios vencidos que ainda não executaram",
    ],
    "price_research": [
        "quanto custa {product}?", "preço médio de {product}", "how much does {product} cost",
        "qual o valor de referência para {product}", "pesquisa de preço de {product}", "what do agencies pay for {product}",
        "quanto o governo paga por {product}?", "preço praticado de {product} em {state}", "price benchmark for {product}",
        "tô precisando do preço de mercado de {product}",
    ],
    "create_purchase_order": [
        "cria um pedido de compra de {qty} {product}", "compra {qty} {product} do fornecedor {supplier}",
        "place an order for {qty} {product}", "emite ordem de compra para {product}", "quero comprar {qty} {product}",
        "order {qty} units of {product} from {supplier}", "abre um pedido de {product} com a {supplier}",
        "faz a compra de {qty} {product}", "buy {qty} {product}", "gera a OC de {qty} {product}",
    ],
    "run_status": [
        "qual o status do run {run}?", "a execução {run} terminou?", "status of job {run}", "o agente {run} ainda tá rodando?",
        "is run {run} done", "cadê o resultado da execução {run}", "check progress of {run}", "o pedido {run} foi aprovado?",
        "andamento da tarefa {run}", "what happened to run {run}",
    ],
    "out_of_scope": [
        "qual a previsão do tempo amanhã?", "me conta uma piada", "quem ganhou o jogo ontem?", "write a poem about the sea",
        "como faço bolo de cenoura?", "what's the capital of Australia", "traduz 'hello' pra japonês", "oi, tudo bem?",
        "recomenda um filme", "how do I reset my phone", "qual o horário do almoço?", "marca uma reunião com a diretoria",
    ],
}

SLOTS = {
    "cnpj": ["11.222.333/0001-81", "45.723.174/0001-10", "12345678000195"], "cpf": ["529.982.247-25", "111.444.777-35"],
    "supplier": ["Atlas", "Boreal", "Cobalto", "Delta", "Faro"], "item": ["notebooks", "papel", "toner", "consultoria"],
    "month": ["março", "janeiro", "May", "outubro"], "nf": ["NF-100003", "NF-100457", "NF-100120"],
    "town": ["Vila Aurora", "Serra Clara", "Porto Lume", "Lagoa Prata"], "state": ["MS", "GO", "PA", "BA"],
    "program": ["P-1001", "P-1004", "água rural"], "product": ["notebook", "papel A4", "café", "cadeira giratória", "monitor"],
    "qty": ["10", "25", "3", "100"], "run": ["8f3a2c", "run-42", "b71e09"],
}


@dataclass(frozen=True)
class Example:
    text: str
    intent: str
    template: str     # "intent:index" — the split key


def _rough(text: str, rng: random.Random) -> str:
    r = rng.random()
    if r < 0.2:
        text = text.lower()
    elif r < 0.27:
        text = text.upper()
    if rng.random() < 0.3:
        text = text.translate(str.maketrans("áàãâéêíóôõúç", "aaaaeeiooouc"))
    if rng.random() < 0.12 and len(text) > 8:
        i = rng.randrange(2, len(text) - 2)
        text = text[:i] + text[i + 1:]
    return text


@lru_cache(maxsize=1)
def examples(per_template: int = 4, seed: int = 7) -> tuple[Example, ...]:
    rng = random.Random(seed)
    out = []
    for intent, templates in TEMPLATES.items():
        for t_index, template in enumerate(templates):
            seen = set()
            for _ in range(per_template * 3):
                filled = template.format(**{k: rng.choice(v) for k, v in SLOTS.items()})
                text = _rough(filled, rng)
                if text not in seen:
                    seen.add(text)
                    out.append(Example(text, intent, f"{intent}:{t_index}"))
                if len(seen) == per_template:
                    break
    return tuple(out)


def split(test_share: float = 0.3, seed: int = 13) -> tuple[list[Example], list[Example]]:
    """Template-level split: whole templates go to test, per intent, so no paraphrase leaks."""
    rng = random.Random(seed)
    train, test = [], []
    for intent, templates in TEMPLATES.items():
        ids = list(range(len(templates)))
        rng.shuffle(ids)
        held = set(ids[: max(2, round(len(ids) * test_share))])
        for ex in examples():
            if ex.intent == intent:
                (test if int(ex.template.split(":")[1]) in held else train).append(ex)
    return train, test
