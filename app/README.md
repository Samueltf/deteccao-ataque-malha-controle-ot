# Dashboard — Módulo 4

Dashboard Streamlit que transforma o modelo do M3 em um produto de dados: explica as
decisões do detector e comunica o projeto para um público misto.

## Executar

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

O app abre em <http://localhost:8501>. Não é preciso baixar os CSV brutos: ele lê o recorte
preparado em `data/processed/` (4,4 MB), que está versionado. Para regenerá-lo a partir da
fonte original, rode antes `python scripts/preparar_dados.py` (baixa ~170 MB).

## As três visões

| Aba | Pergunta que responde | Interação |
|---|---|---|
| **Panorama da operação** | qual é a situação geral nas 25,6 h de teste? | o **limiar de alarme** é um slider: mover recalcula cobertura, alarmes falsos por hora, latência e a tabela dos 18 ataques |
| **Por que esta janela** | o que o modelo diz sobre este caso e por quê? | escolha do grupo, da ocorrência e da janela; **waterfall SHAP**; simulador "e se" sobre qualquer variável |
| **O que o modelo valoriza** | quais fatores importam no geral? | alternância entre agrupar por equipamento (64) ou por coluna do modelo (192); tabela completa filtrável |

As limitações ficam em `st.warning()` logo abaixo do título — visíveis sem rolar a página —
e na íntegra no rodapé.

## Estrutura

O `app.py` não faz pré-processamento próprio: carrega o modelo serializado em `models/` e
chama as mesmas funções de `src/` usadas nos notebooks do M2, M3 e M4. O carregamento do
modelo e o cálculo do SHAP ficam sob `@st.cache_resource`, e as métricas por limiar sob
`@st.cache_data` — sem isso o Streamlit recalcularia tudo a cada clique.

Rótulos: nenhuma tela mostra nome de coluna do modelo. `src/rotulos.py` traduz as 64 tags
para o nome do equipamento e da grandeza, a partir da tabela de pontos do manual técnico
oficial do HAI.
