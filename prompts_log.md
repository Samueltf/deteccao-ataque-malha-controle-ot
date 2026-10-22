# Declaração de uso de Inteligência Artificial

**Projeto Integrado em Ciência de Dados (ECD-108)** — PPGMCS/UNIMONTES
Aluno: Samuel Rodrigues Gonçalves · Prof.ª Luciana Balieiro Cosme

Conforme a política "Uso ético de Inteligência Artificial nas atividades da disciplina".

**Ferramenta:** Claude (Anthropic), modelo Opus 5.

## Como usei

- **Discussão de alternativas** nos pontos de decisão: bases candidatas, tratamento de redundância,
  estratégia de validação, escolha de modelo, formato do dashboard. 
- **Sugestão inicial de estrutura e revisão de código** em Python, a partir das decisões e do desenho que defini.
- **Revisão de redação** dos relatórios.

## O que é meu

A escolha do domínio e do problema, a seleção da base de dados, o recorte do projeto — detecção com
orçamento de alarme em vez de acurácia —, as metas das métricas, a estratégia de validação temporal,
o desenho da solução e as decisões técnicas abaixo. Sou integralmente responsável pelo conteúdo
entregue.

## Decisões por módulo

**M1 — definição do problema.** Avaliei três bases e escolhi o HAI 23.05 depois de abrir as duas
finalistas: a da CISA tinha 113 registros positivos em 4.034, desbalanceamento que inviabilizava o
projeto. Defini o recorte como detecção binária em janelas, com cobertura de eventos sob orçamento
de alarme — acurácia não serve numa base com 4% de positivos. Medi o baseline físico antes de
modelar, para saber o piso.

**M2 — exploração.** Tomei quatro decisões de pré-processamento depois de ver o custo de cada
alternativa. A principal: tratar os pares comando/realimentação pela **diferença** em vez de
descartar um lado. É desvio consciente da regra geral, porque descartar apagaria o resíduo que
carrega o sinal de manipulação. Registrei como desvio no relatório.

**M3 — modelagem.** Mantive o corte cronológico e usei `TimeSeriesSplit` no lugar do
`StratifiedKFold` do checklist — embaralhar série temporal é vazamento —, declarando a divergência.
Escolhi o modelo final pela métrica primária medida fora-da-amostra dentro do treino, sem tocar no
teste. Na passagem final o gradient boosting cobriu mais eventos que a regressão logística
escolhida; **mantive a escolha** e registrei a divergência, porque trocar depois de ver o teste é a
armadilha que a disciplina nomeia.

**M4 — interpretabilidade e dashboard.** Optei pelo SHAP em forma fechada, exato para modelo linear,
com teste de aditividade no caderno. Troquei a entrada manual de valores por um simulador, porque
com 192 variáveis a entrada manual não faz sentido. Montei o dicionário de rótulos das 64 tags a
partir do manual técnico do dataset. Reorganizei a narrativa do aplicativo para as limitações
ficarem visíveis sem rolar a página.

## Verificação

Reexecutei os cadernos do zero, em ambiente limpo e em outra máquina, com as versões fixadas em
`requirements.txt`. O modelo final reproduziu exatamente. O gradient boosting oscila na segunda casa
conforme o paralelismo da máquina — registrado em `experiments_log.md`.
