# routebench

[English](README.md) · **Português**

Compara formas de decidir qual ferramenta deve atender um pedido, de uma lista de palavras-chave
até um LLM, e mostra qual vale o que custa.

![routebench](docs/screenshots/scorecard-top.png)

## A pergunta

Todo assistente que usa ferramentas começa decidindo: *este pedido precisa de qual ferramenta, ou de nenhuma?*
Normalmente isso é escolhido no feeling. Aqui a escolha é medida.

## Os métodos comparados

| método | como funciona | custo |
|---|---|---|
| `keywords` | listas de palavras-chave escritas à mão | grátis |
| `fuzzy-nn` | procura o exemplo mais parecido | grátis |
| `naive-bayes` | classificador simples treinado com exemplos | grátis |
| `embed-knn` | compara por similaridade de vetores | grátis |
| `cascade` | usa o barato e só chama o caro quando o barato não tem certeza | misto |
| `llm` | qualquer modelo compatível com a API da OpenAI | por token |

## Resultados

| método | acurácia | pedidos fora de escopo aceitos por engano | casos enviados ao 2º método |
|---|---|---|---|
| **cascade (keywords → embed-knn)** | **81,5%** | 0% | 32% |
| keywords | 75,9% | 0% | — |
| fuzzy-nn | 69,4% | **85%** | — |
| naive-bayes | 62,0% | **100%** | — |
| embed-knn | 59,3% | 0% | — |

O que a acurácia sozinha não mostra:

- A cascata ganha de cada método isolado e só repassa um terço dos pedidos.
- Mas a vantagem dela sobre as palavras-chave **pode ser sorte**: com 108 exemplos o teste estatístico não confirma (p = 0,18). Precisaria de mais dados.
- Dois métodos parecem só "um pouco piores", mas executam uma ação em **85% a 100%** dos pedidos que deveriam recusar.

Com uma chave de API configurada, o LLM entra na mesma tabela, com o custo por mil pedidos.
Nenhum número de LLM é digitado à mão: as respostas ficam em cache, então o resultado pode ser refeito.

## Como a comparação é feita

- Frases de teste nunca são variações das frases de treino.
- Pedidos fora de escopo são uma categoria própria, com taxa de erro separada.
- Intervalos de confiança e teste de McNemar, para saber quando a diferença é real.
- Calibração: quando o método diz "tenho 90% de certeza", acerta 90% das vezes?
- O CI falha se algum método cair abaixo do mínimo definido em `gates.json`.

Detalhes em [`docs/adr/0001-how-to-compare-routers.md`](docs/adr/0001-how-to-compare-routers.md) (em inglês).

## Rodando

```bash
git clone https://github.com/murilo-bassan/llm-routing-eval-lab.git
cd llm-routing-eval-lab
pip install -r requirements-dev.txt
python -m pytest
python -m lab.cli --gate      # gera reports/scorecard.html e reports/scorecard.pt.html
```

Com um LLM: `LLM_API_KEY=... LAB_LLM_MODEL=... LAB_PRICE_IN=... LAB_PRICE_OUT=... python -m lab.cli`

![relatório completo](docs/screenshots/scorecard.png)

**Stack:** Python, NumPy

O conjunto de dados é fictício. Licença MIT.
