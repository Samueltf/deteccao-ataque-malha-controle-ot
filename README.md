# Detecção de manipulação da malha de controle em sistema industrial

Projeto Integrado em Ciência de Dados (ECD-108) — PPGMCS/UNIMONTES
Prof.ª Luciana Balieiro Cosme · Aluno: **Samuel Rodrigues Gonçalves**
Execução individual autorizada pela professora.

## Sobre o projeto

Classificação binária supervisionada sobre janelas de série temporal multivariada para
detectar manipulação da malha de controle em um sistema industrial, usando as 86 tags de
sensores e atuadores do **HAI Security Dataset 23.05** — com a detecção avaliada sob um
**orçamento de alarme** definido pela operação, e não por acurácia.

O ataque tratado aqui é o que não dispara alarme: altera setpoint ou comando de atuador
mantendo a variável dentro da faixa operacional, e só se torna visível quando o efeito já
ocorreu. Em julho de 2026 a CISA documentou campanha ativa com esse perfil contra PLCs
Rockwell, Schneider e Siemens (alerta AA26-097A).

**Estado atual:** Módulos 1 a 4 concluídos. Falta a entrega final (M5): slides e banca.

## Como executar

```bash
git clone https://github.com/Samueltf/deteccao-ataque-malha-controle-ot.git
cd deteccao-ataque-malha-controle-ot

python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# o dashboard ja roda: data/processed (4,4 MB) esta versionado
streamlit run app/app.py
```

Para rodar os cadernos ou regenerar os dados preparados a partir da fonte original:

```bash
# baixa os CSV do HAI (~170 MB dos arquivos com ataque) e refaz data/processed
python scripts/preparar_dados.py

jupyter notebook notebooks/final/01_eda.ipynb               # relatorio de EDA (M2)
jupyter notebook notebooks/final/02_modelagem.ipynb         # relatorio de modelagem (M3)
jupyter notebook notebooks/final/03_interpretabilidade.ipynb  # interpretabilidade (M4)
```

No Google Colab, clone o repositório e rode a célula de download antes das demais; o
notebook resolve os caminhos a partir da raiz do repositório.

Fonte dos dados: <https://github.com/icsdataset/hai> — licença CC BY-SA 4.0,
acesso público e direto, sem cadastro. Data de acesso: 19/09/2026.

## Estrutura

```
data/raw/           CSV originais, baixados sob demanda (nao versionados)
data/processed/     dados tratados gerados pelo pipeline
notebooks/
  exploracao/       rascunhos
  final/            01_eda.ipynb (M2), 02_modelagem.ipynb (M3), 03_interpretabilidade.ipynb (M4)
data/raw/           CSV originais, baixados sob demanda (nao versionados)
data/processed/     recorte leve que o dashboard consome (versionado, 4,4 MB)
data/externo/       tabela dos 18 ataques do teste, extraida do manual tecnico do HAI
scripts/
  preparar_dados.py baixa os CSV e gera data/processed
src/
  data_loading.py   download e carregamento com tipos economicos
  preprocess.py     todas as decisoes do M2 como funcoes reutilizaveis
  avaliacao.py      metricas por evento, orcamento de alarme e calibracao de limiar
  shap_util.py      SHAP em forma fechada para o modelo linear
  rotulos.py        as 64 tags em linguagem de operador, do manual tecnico
models/             modelo final serializado
reports/figures/    as figuras dos tres relatorios
app/app.py          dashboard Streamlit do M4
experiments_log.md  registro dos 14 runs do M3 (substitui o MLflow)
prompts_log.md      diario de uso de IA, exigido pela disciplina
```

## Dados

| | |
|---|---|
| Registros | 1.180.800 (328 h a 1 Hz) |
| Operação normal | 4 arquivos, 249 h |
| Com ataque | 2 arquivos, 79 h, **52 ataques rotulados** |
| Variáveis | 86 tags de sensores e atuadores |
| Período | 04 a 25/08/2022 |
| Classe positiva | **4,00%** dos segundos (1 para 24) |

## Resultados

### Achados da análise exploratória (M2)

| Achado | Valor | Consequência |
|---|---|---|
| Ausentes, duplicatas, lacunas de timestamp | zero nos 6 arquivos | qualidade não é o gargalo |
| Tags sem variância | 18 nos 6 arquivos (27 só no `hai-test1`) | removidas; as 10 constantes em parte dos arquivos foram mantidas |
| Tags fora da faixa documentada | 15 | apenas 2 exigiram intervenção |
| Maior correlação com o alvo | **0,199** | nenhuma variável isolada resolve — sinal é multivariado |
| Pares com \|r\| > 0,95 | 34 | tratados em três categorias distintas |
| `P1_PCV02D` fora do repouso | **0,07% normal × 6,89% sob ataque (94×)** | feature derivada que Pearson não enxerga |
| Duração dos ataques | mediana 132 s; 17 de 52 abaixo de 2 min | limita o tamanho da janela |

### Baselines

Teste intocado: 18 ataques em 25,6 h, orçamento de 1 alarme falso/hora e 2% do tempo normal
sob alarme. Detalhe em [`experiments_log.md`](experiments_log.md).

| Modelo | Cobertura | Alarmes falsos/h | Latência | F1 (ponto) |
|---|---|---|---|---|
| Baseline trivial (`DummyClassifier`) | 0 de 18 | 0,00 | — | 0,000 |
| Resíduo comando/realimentação, limiar p99 (medido no M1) | 1 de 14 em `hai-test1` | — | — | — |
| Alarme por limite fixo | 7 de 18 — 38,9% | 1,09 | 25 s | 0,537 |
| Isolation Forest (só operação normal) | 8 de 18 — 44,4% | 1,29 | 13 s | 0,501 |
| Gradient boosting sobre janelas | 10 de 18 — 55,6% | 0,94 | 0 s | 0,543 |
| **Regressão logística sobre janelas (final)** | **8 de 18 — 44,4%** | **0,43** | **0 s** | **0,564** |

### O achado do M3

O erro não está espalhado pelos ataques: está concentrado em uma classe inteira.

| Classe de ataque | O que o atacante manipula | Cobertura (modelo final) |
|---|---|---|
| **Ponto de E/S** (11 ataques) | setpoint, variável de processo ou comando — pontos que existem entre as 86 tags | **8 de 11 — 73%** |
| **Ponto interno** (7 ataques) | parâmetro do algoritmo do controlador — ponto que **não** existe nas tags | **0 de 7 — 0%** |

A meta de 70% declarada no M1 é atingida na classe que os dados enxergam, e é inalcançável na
outra: detectar manipulação de ponto interno exige visibilidade do programa do controlador, não
mais modelos sobre o historiador. Essa é a recomendação operacional que sai do projeto.

### Métricas do projeto

- **Primária:** cobertura de eventos sob orçamento de no máximo 1 alarme falso por hora **e**
  no máximo 2% do tempo normal sob alarme. Meta declarada no M1: ≥ 70% dos ataques.
  A segunda restrição impede que um detector que acusa o tempo todo registre "zero alarmes falsos".
- **Apoio:** latência mediana de detecção (meta ≤ 60 s) e PR-AUC por ponto.
- **Negócio:** alarmes falsos por turno de 8 horas.

Acurácia não é usada: responder sempre "normal" rende 96,00% sem detectar nada.

## Decisões de pré-processamento

Implementadas em `src/preprocess.py`, justificadas no notebook de EDA.

1. **Tags constantes** — remover as 18 sem variância nos seis arquivos; manter as 10 que
   variam em parte deles.
2. **Valores impossíveis** — clipar em zero e sinalizar com flag apenas `P3_LCV01D` e
   `P3_LCP01D` (comando de válvula negativo). Outliers que são sinal foram mantidos.
3. **Redundância (\|r\| > 0,95)** — três tratamentos: remoção da menos interpretável na
   duplicação pura; colapso do bloco de 6 tags de P2 (r = 1,000) em uma; e, nos pares
   comando/realimentação, **substituição do par pela diferença**. O terceiro é um desvio
   consciente da regra "remova a menos interpretável", documentado no notebook: descartar
   um lado apagaria o resíduo que carrega o sinal de manipulação.
4. **Normalização em dois ramos** — `RobustScaler` no ramo logístico (18 variáveis com
   |assimetria| > 3), nenhuma normalização no ramo de árvore/boosting. `fit` exclusivamente
   no trecho cronologicamente anterior ao corte.

**Validação temporal.** Split estritamente cronológico em três trechos: treino (0–144.180 s,
40,0 h, 33 eventos), validação (144.180–192.240 s, 13,4 h, só para calibrar o limiar de alarme) e
teste (192.240–284.400 s, 25,6 h, 18 eventos, usado uma única vez). Janelas que cruzam uma
fronteira são descartadas. `TimeSeriesSplit(5)` no lugar de `StratifiedKFold` — divergência
declarada do checklist da disciplina, justificada no notebook do M3.

**Modelo final.** Regressão logística com `class_weight='balanced'` e `StandardScaler` dentro do
`Pipeline`, sobre janelas de 60 s com passo de 30 s (média, desvio-padrão móvel e delta de cada uma
das 64 variáveis = 192 features). Escolhido pela cobertura de eventos medida **fora-da-amostra
dentro do treino**, nas dobras do `TimeSeriesSplit` — nunca pelo teste. Serializado em
`models/modelo_final_logistica.pkl`.

**Desbalanceamento.** SMOTE não é usado: interpola entre janelas vizinhas, que compartilham
segundos, o que é vazamento disfarçado de balanceamento. A opção adotada é
`class_weight='balanced'`.

## Dashboard (M4)

```bash
streamlit run app/app.py
```

Três visões, cada uma respondendo a uma pergunta:

1. **Panorama da operação** — KPIs, linha do tempo das 25,6 h com os ataques e os alarmes, e a
   tabela dos 18 ataques. O **limiar de alarme é um slider**: mover recalcula tudo em tempo real,
   o que torna a troca cobertura × alarme falso uma decisão do usuário, não um número fixo.
2. **Por que esta janela** — escolha de uma janela de 60 s (ataque, alarme falso ou operação
   normal) e o **waterfall SHAP** explicando a decisão, com um simulador "e se" para qualquer
   variável.
3. **O que o modelo valoriza** — importância global, agrupada por equipamento ou por coluna do
   modelo, e a comparação de evidência entre as duas classes de ataque.

As limitações aparecem em `st.warning()` logo abaixo do título, visíveis sem rolar a página.

### Interpretabilidade

O modelo final é linear, então o valor de Shapley tem **forma fechada** — `src/shap_util.py`
calcula `φᵢ = coefᵢ·(zᵢ − E[zᵢ])` direto, com erro de aditividade da ordem de 10⁻⁶ contra o
`decision_function`. Não há amostragem: a explicação é exata e instantânea, o que permite
recalculá-la a cada clique no dashboard.

O achado do M4 está em `03_interpretabilidade.ipynb`: sob ataque de ponto de E/S o modelo reúne
**quatro vezes** mais evidência do que em operação normal (Σ|φ| de 43,5 contra 10,7); sob ataque de
**ponto interno** a evidência é de 11,0 — estatisticamente igual à da operação normal. O ataque de
ponto interno não fica *quase* detectado: fica **indistinguível**. Baixar o limiar não resolve.

## Limitações

- Dados de **testbed**, não de planta real: o modelo não é transferível para outra
  instalação sem recalibração.
- **52 ataques** no total — a avaliação por evento tem amostra pequena; os resultados são
  reportados como distribuição, não como número único.
- Os rótulos por subprocesso foram removidos a partir da versão 22.04 do dataset, o que
  limita o escopo à detecção, sem localização do alvo.

## Uso de Inteligência Artificial

Declarado conforme a política da disciplina. O diário de prompts está em
[`prompts_log.md`](prompts_log.md).

## Referências

- SHIN, H.-K. et al. *HAI 1.0: HIL-based Augmented ICS Security Dataset*. USENIX CSET 2020.
- CISA. *Iranian-Affiliated Cyber Actors Exploit PLCs Across US Critical Infrastructure*,
  alerta AA26-097A, jul. 2026.
- DRAGOS. *OT Cybersecurity Year in Review 2026* (dados de 2025).
