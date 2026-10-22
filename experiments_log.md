# Registro de experimentos — Módulo 3

Projeto Integrado em Ciência de Dados (ECD-108) · Samuel Rodrigues Gonçalves

Substitui o MLflow previsto no plano de disciplina, conforme orientação da Semana 3.
Reprodução: `notebooks/final/02_modelagem.ipynb`, que gera todos os números desta tabela.

**Protocolo comum a todos os runs**

- Dados: HAI 23.05, arquivos com ataque (`hai-test1` + `hai-test2`), 284.400 s, 52 ataques.
- Pré-processamento: `src/preprocess.py` (decisões do M2) → 64 variáveis.
- Features: janela deslizante com média, desvio-padrão móvel e delta por variável → 192 features.
- Corte **cronológico**: treino 0–144.180 s (40,0 h, 33 eventos) · validação 144.180–192.240 s
  (13,4 h, calibração do limiar) · teste 192.240–284.400 s (25,6 h, 18 eventos).
- CV: `TimeSeriesSplit(5)` dentro do treino. **Não** `StratifiedKFold` — divergência declarada
  na seção 2.1 do notebook (embaralhar série temporal é vazamento).
- Busca de hiperparâmetros: `RandomizedSearchCV`, `scoring='average_precision'`.
- Limiar de alarme: calibrado **na validação**, pelo orçamento de ≤ 1 alarme falso/hora e
  ≤ 2% do tempo normal sob alarme.
- `random_state=42` / `np.random.seed(42)` em tudo.
- **O conjunto de teste foi usado uma única vez**, na passagem final (runs 10 a 13).

**Métricas das colunas**

- `F1 CV` — F1 da classe ataque, média ± desvio das 5 dobras do `TimeSeriesSplit`.
- `PR-AUC CV` — média das 5 dobras; é a métrica otimizada na busca.
- `Cobertura OOF` — métrica **primária** do projeto medida fora-da-amostra dentro do treino:
  em cada dobra, ajusta no passado, calibra o limiar na 1ª metade do trecho de validação da dobra
  e mede na 2ª. 14 eventos ao todo. Foi por esta coluna que o modelo final foi escolhido.
- `Teste` — preenchida **somente** nos runs da passagem final.

## Runs

| # | Modelo | Janela | Params principais | F1 CV | PR-AUC CV | Cobertura OOF | Teste | Decisão |
|---|---|---|---|---|---|---|---|---|
| 1 | `DummyClassifier(strategy='prior')` | 60 s | — | 0,000 ± 0,000 | 0,057 ± 0,008 | 0/14 — 0% | ver run 14 | Piso. Acurácia 93,98% sem detectar nada: confirma que acurácia não serve aqui. |
| 2 | Alarme por limite fixo (faixa p0,5–p99,5 do normal) | 60 s | sem aprendizado | n/a | n/a | 6/14 — 43% · 1,56 alarme falso/h | ver run 13 | Representa a regra que a planta já tem. Cobertura razoável, custo de alarme acima do orçamento. |
| 3 | `IsolationForest` (ajustado só em janelas normais) | 60 s | `n_estimators=400` | n/a | n/a | 5/14 — 36% · 1,20 alarme falso/h | ver run 12 | Não supervisionado, serve para ataque nunca visto. Perde para o linear e estoura o orçamento. |
| 4 | `LogisticRegression` + `StandardScaler` | 30 s | `C=0,0178` · `class_weight='balanced'` | — | 0,317 | 8/14 — 57% · 1,38 alarme falso/h | — | Empata em cobertura com a janela de 60 s e custa 6× mais alarme falso. |
| 5 | **`LogisticRegression` + `StandardScaler`** | **60 s** | **`C=0,00562` · `class_weight='balanced'`** | **0,303 ± 0,151** | **0,328 ± 0,167** | **8/14 — 57% · 0,24 alarme falso/h** | **ver run 14** | **Escolhido.** Melhor cobertura OOF ao menor custo de alarme, melhor F1 e PR-AUC em CV, o mais simples entre os que chegam lá, coeficientes auditáveis. |
| 6 | `LogisticRegression` + `StandardScaler` | 120 s | `C=0,1` | — | 0,382 | 4/14 — 29% · 0,06 alarme falso/h | — | **Melhor PR-AUC de todos e pior cobertura de eventos.** A janela longa dilui o ataque curto. Caso explícito de métrica por ponto discordando da métrica de evento — e a razão de a seleção ser feita pela segunda. |
| 7 | `HistGradientBoostingClassifier` | 30 s | `lr=0,1` · `max_iter=100` · `leaf=15` · `min_leaf=10` · `l2=0,0` | — | 0,274 | 6/14 — 43% · 0,54 alarme falso/h | — | Não supera o linear. |
| 8 | `HistGradientBoostingClassifier` | 60 s | `lr=0,03` · `max_iter=400` · `leaf=15` · `min_leaf=40` · `l2=0,1` | 0,211 ± 0,174 | 0,236 ± 0,137 | 4/14 — 29% · 0,18 alarme falso/h | ver run 11 | Não supera o linear em nenhum critério de treino. Mantido na comparação final. |
| 9 | `HistGradientBoostingClassifier` | 120 s | `lr=0,03` · `max_iter=200` · `leaf=15` · `min_leaf=20` · `l2=0,0` | — | 0,260 | 5/14 — 36% · 0,24 alarme falso/h | — | Idem. |

### Passagem final no teste — uma única vez

18 ataques em 25,6 h · limiar calibrado na validação pelo orçamento de 1 alarme falso/h e 2% do tempo normal.

| # | Modelo | Cobertura | Alarmes falsos/h | % do tempo normal em alarme | Latência mediana | Recall | F1 | AUC-ROC | PR-AUC |
|---|---|---|---|---|---|---|---|---|---|
| 10 | Baseline trivial | 0/18 — 0,0% | 0,00 | 0,00% | — | 0,000 | 0,000 | 0,500 | 0,060 |
| 11 | `HistGradientBoosting` | 10/18 — 55,6% | 0,94 | 3,28% | 0 s | 0,492 | 0,543 | 0,823 | 0,490 |
| 12 | `IsolationForest` | 8/18 — 44,4% | 1,29 | 3,85% | 12,5 s | 0,454 | 0,501 | 0,728 | 0,434 |
| 13 | Alarme por limite fixo | 7/18 — 38,9% | 1,09 | 3,20% | 25 s | 0,476 | 0,537 | 0,822 | 0,515 |
| 14 | **`LogisticRegression` (modelo final)** | **8/18 — 44,4%** | **0,43** | **1,87%** | **0 s** | **0,465** | **0,564** | **0,762** | **0,521** |

Por classe de ataque, no modelo final: **ponto de E/S 8 de 11 (73%)** · **ponto interno 0 de 7 (0%)**.

## Nota de reprodutibilidade

Todos os runs usam `random_state=42`, e os números desta tabela vêm da execução dos cadernos na
minha própria máquina, em ambiente limpo (venv nova a partir de `requirements.txt`).

Executando o mesmo caderno em outra máquina, com número diferente de núcleos, **a regressão
logística reproduz exatamente**: mesmo `C = 0,00562`, mesmo PR-AUC em CV (0,328), mesmos 8 de 18
ataques no teste a 0,43 alarme falso/hora. O `HistGradientBoostingClassifier` **não**: a busca
aleatória converge para combinações de hiperparâmetros diferentes conforme o paralelismo
disponível, e as métricas dele oscilam na segunda casa (PR-AUC em CV entre 0,236 e 0,252; alarme
falso no teste entre 0,94 e 0,98 por hora). A oscilação não muda a conclusão — o boosting não
supera o linear em nenhum critério medido dentro do treino — mas é mais uma razão prática para
preferir o modelo simples: ele é auditável e dá o mesmo resultado em qualquer lugar.

## Conclusão e justificativa da escolha

A regressão logística foi escolhida **antes** de qualquer medição no teste, pelo único critério
disponível que mede a métrica primária do projeto sem tocar nele: cobertura de eventos
fora-da-amostra nas dobras do `TimeSeriesSplit`. Ela lidera nesse critério (57%), ao menor custo de
alarme (0,24/h), e também lidera em F1 e PR-AUC na validação cruzada. É, além disso, o modelo mais
simples entre os avaliados — a regra da disciplina é só avançar para o complexo se o simples não
atingir a métrica, e aqui o complexo não supera o simples em nenhum critério de treino.

**Divergência registrada, não corrigida.** Na passagem final, o gradient boosting cobriu 10 eventos
contra 8 da logística. Trocar o modelo por causa disso seria ajustar no teste — a armadilha nº 4 da
disciplina. Com 18 eventos, dois de diferença cabem na variação amostral; e o boosting só chega lá
gastando 3,3% do tempo normal em alarme contra 1,9% da logística. A escolha permanece.

**Meta do M1.** A meta de 70% de cobertura não foi atingida no total (44%), e só seria alcançada a
~5 alarmes falsos por hora, o que inviabiliza o detector na operação. Ela **é** atingida na classe
de ataque que os dados conseguem enxergar (ponto de E/S, 73%). Os sete ataques de ponto interno
manipulam parâmetros do algoritmo do controlador, que não existem entre as 86 tags — limitação dos
dados, não do modelo.
