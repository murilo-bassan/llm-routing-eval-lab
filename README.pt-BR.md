<div align="center">

# routebench · llm-routing-eval-lab

**Qual roteador deve mandar cada pedido para cada tool, e quando vale a pena pagar por um LLM?**
Um benchmark que mostra o que a acurácia esconde: falsos aceites fora de escopo, calibração,
intervalos de confiança, significância estatística e o limiar da cascata.

![python](https://img.shields.io/badge/python-3.11%2B-000?style=flat-square&labelColor=000&color=c6ff00)
![stats](https://img.shields.io/badge/McNemar%20%C2%B7%20bootstrap%20%C2%B7%20ECE-000?style=flat-square&labelColor=000&color=a58bff)
![testes](https://img.shields.io/badge/testes-14%20%2B%20port%C3%A3o%20no%20CI-000?style=flat-square&labelColor=000&color=c6ff00)

[English](README.md) · **Português**

<img src="docs/screenshots/scorecard-top.png" alt="scorecard" width="100%">

</div>

---

## A pergunta

Todo assistente que usa tools começa roteando: *qual tool este pedido precisa, ou nenhuma?*
As equipes costumam escolher no feeling entre palavras-chave escritas à mão, um classificador
leve e uma chamada de LLM. Este laboratório transforma isso numa decisão medida, com a
metodologia que um revisor pediria ([ADR 0001](docs/adr/0001-how-to-compare-routers.md), em inglês).

## Braços

| braço | como | custo |
|---|---|---|
| `keywords` | listas de palavras-chave escritas à mão: por onde a maioria dos sistemas começa | grátis |
| `fuzzy-nn` | pedido de treino mais próximo via difflib: o baseline do `mcp-tool-gateway` | grátis |
| `naive-bayes` | Naive Bayes multinomial sobre palavras + 4-gramas de caracteres (tolera erro de digitação) | grátis |
| `embed-knn` | kNN por cosseno sobre embeddings de n-gramas com hash, com limiar de similaridade para fora de escopo | grátis |
| `cascade(a→b@τ)` | o braço barato responde quando a confiança é ≥ τ; senão, escala | misto |
| `llm:<modelo>` | qualquer modelo compatível com OpenAI, modo JSON, few-shot, **cache em disco** | por token |

## Resultados (braços offline, templates separados para teste — `python -m lab.cli`)

| braço | acurácia [IC 95%] | macro-F1 | falso aceite fora de escopo | ECE | escalado |
|---|---|---|---|---|---|
| **cascade(keywords→embed-knn@0.75)** | **0,815** [0,73–0,89] | **0,802** | 0,00 | 0,133 | 32% |
| keywords | 0,759 [0,68–0,84] | 0,730 | 0,00 | 0,079 | — |
| fuzzy-nn | 0,694 [0,61–0,78] | 0,654 | **0,85** | 0,128 | — |
| naive-bayes | 0,620 [0,53–0,71] | 0,585 | **1,00** | 0,212 | — |
| embed-knn | 0,593 [0,50–0,69] | 0,598 | 0,00 | 0,220 | — |

O que o scorecard mostra e um único número de acurácia não mostraria:
- A cascata vence cada uma das suas partes, mandando só um terço do tráfego para o segundo braço.
- A vantagem dela sobre as palavras-chave puras **não é estatisticamente significativa** com 108
  itens (McNemar p = 0,18): é preciso mais dados rotulados antes de afirmar isso.
- Dois braços aprendidos mandam **85–100% dos pedidos fora de escopo para ações reais**; olhando só a
  acurácia, parecem apenas "um pouco piores".
- Com uma chave configurada, a mesma execução acrescenta `llm:<modelo>` e `cascade(keywords→llm)`,
  com custo por mil pedidos: os números que decidem se o LLM vale o preço.

> **Nenhum resultado de LLM neste repositório foi digitado à mão.** Os braços com LLM só rodam
> quando `LLM_API_KEY` está no seu ambiente (ou num secret do CI); as respostas ficam em cache
> pelo hash do prompt, então as análises são reproduzíveis.

## Destaques da metodologia
- **Split por template**: nenhuma paráfrase vaza para o conjunto de teste.
- **Fora de escopo como classe**, com a taxa de falso aceite reportada à parte.
- **Teste exato de McNemar** contra o melhor braço; **IC 95% por bootstrap**; calibração por **ECE**.
- **Curvas de cobertura × acurácia**: onde colocar o τ de uma cascata.
- **Portão de regressão**: `gates.json` define pisos por braço; `python -m lab.cli --gate` sai com 1 se algum for violado (CI).
- **Scorecard HTML** independente (SVG inline, sem JS), em inglês e em português, enviado como artefato do CI.

## Como rodar

```bash
git clone https://github.com/murilo-bassan/llm-routing-eval-lab.git && cd llm-routing-eval-lab
pip install -r requirements-dev.txt
python -m pytest
python -m lab.cli --gate          # → reports/scorecard.html e reports/scorecard.pt.html
# com um LLM:  LLM_API_KEY=... LAB_LLM_MODEL=... LAB_PRICE_IN=... LAB_PRICE_OUT=... python -m lab.cli
```

<img src="docs/screenshots/scorecard.png" alt="scorecard completo" width="100%">

---

<sub>Parte de um portfólio de engenharia de IA: veja [ai-engineering-portfolio](https://github.com/murilo-bassan/ai-engineering-portfolio). O dataset é sintético. Licença MIT.</sub>
