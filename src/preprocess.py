"""Pre-processamento do HAI 23.05.

Todas as decisoes do Modulo 2 estao implementadas aqui como funcoes reutilizaveis,
para serem chamadas tanto pelos notebooks quanto pelo app Streamlit do M4.
A justificativa de cada decisao esta no relatorio de EDA (notebooks/final/01_eda.ipynb).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

COL_TEMPO = "timestamp"

# --- D1: tags sem variancia em NENHUM dos seis arquivos do release -------------
# Apuradas sobre os 6 arquivos (1.180.800 registros). Atencao: a lista obtida
# olhando apenas o hai-test1 teria 27 tags; 10 delas variam em outros arquivos
# e NAO podem ser removidas.
TAGS_CONSTANTES = [
    "P1_PIT01_HH", "P1_PP01AD", "P1_PP01AR", "P1_PP01BD", "P1_PP01BR",
    "P1_PP02D", "P1_PP02R", "P1_SOL01D", "P1_SOL03D", "P1_STSP",
    "P2_RTR", "P2_TripEx", "P2_VTR01", "P2_VTR02", "P2_VTR03", "P2_VTR04",
    "P3_LH01", "P3_LL01",
]

# --- D2: redundancia com |r| > 0,95 -------------------------------------------
# (a) duplicacao pura: mantem a mais interpretavel, descarta a outra
DUPLICATAS = {"P4_ST_TT01": "P1_TIT01", "x1003_10_SETPOINT_OUT": "P1_FT02Z"}

# (b) bloco de 6 tags de P2 com r = 1,000 entre si -> colapsa em uma
BLOCO_P2 = ["P2_ATSW_Lamp", "P2_AutoGO", "P2_MASW", "P2_MASW_Lamp",
            "P2_ManualGO", "P2_ManualSD"]
BLOCO_P2_REPRESENTANTE = "P2_MASW"

# (c) pares comando/realimentacao: substitui o par pela diferenca.
# Desvio consciente da regra "remova a menos interpretavel": descartar um lado
# apagaria justamente o residuo que carrega o sinal de manipulacao.
PARES_CMD_FB = [
    ("P1_FCV01D", "P1_FCV01Z"), ("P1_FCV02D", "P1_FCV02Z"),
    ("P1_FCV03D", "P1_FCV03Z"), ("P1_LCV01D", "P1_LCV01Z"),
    ("P1_PCV01D", "P1_PCV01Z"), ("P1_PCV02D", "P1_PCV02Z"),
]

# --- D3: valores fora da faixa documentada no manual tecnico ------------------
# Apenas estes dois exigem intervencao: comando de valvula negativo e
# fisicamente impossivel (faixa documentada 0-27648).
CLIPAR_EM_ZERO = ["P3_LCV01D", "P3_LCP01D"]

# Mantidas de proposito (outlier que e sinal, nao erro):
#   P1_FT02  excede o maximo documentado em 0,6% -> ruido de sensor
#   P1_PIT01 minimo -0,006 bar                   -> ruido em torno de zero
#   P2_VT01  abaixo da faixa nominal             -> maquina parada
#   P3_LIT01 em escala crua, nao convertida      -> documentada, nao alterada


def remover_constantes(X: pd.DataFrame, tags: list[str] | None = None) -> pd.DataFrame:
    """D1 - descarta as tags sem variancia em nenhum arquivo do release."""
    alvo = [c for c in (tags or TAGS_CONSTANTES) if c in X.columns]
    return X.drop(columns=alvo)


def tratar_valores_impossiveis(X: pd.DataFrame, colunas: list[str] | None = None
                               ) -> pd.DataFrame:
    """D3 - clipa em zero e cria flag binaria marcando onde o valor era impossivel."""
    X = X.copy()
    for c in (colunas or CLIPAR_EM_ZERO):
        if c in X.columns:
            X[f"{c}_impossivel"] = (X[c] < 0).astype("int8")
            X[c] = X[c].clip(lower=0)
    return X


def tratar_redundancia(X: pd.DataFrame) -> pd.DataFrame:
    """D2 - aplica os tres tratamentos de redundancia."""
    X = X.copy()
    X = X.drop(columns=[c for c in DUPLICATAS if c in X.columns])
    sobra = [c for c in BLOCO_P2 if c in X.columns and c != BLOCO_P2_REPRESENTANTE]
    X = X.drop(columns=sobra)
    for cmd, fb in PARES_CMD_FB:
        if cmd in X.columns and fb in X.columns:
            X[f"{cmd[:-1]}_residuo"] = (X[cmd] - X[fb]).astype("float32")
            X = X.drop(columns=[fb])
    return X


def taxa_desvio_repouso(s: pd.Series, repouso: float | None = None) -> pd.Series:
    """Feature derivada: 1 quando a tag sai do seu valor de repouso.

    Motivada pelo achado do M2: P1_PCV02D fica no valor de repouso em 99,65%
    do tempo e sai dele 94x mais frequentemente sob ataque (6,89% x 0,07%).
    """
    base = repouso if repouso is not None else s.mode().iloc[0]
    return (s != base).astype("int8")


def preparar(X: pd.DataFrame, com_flags_repouso: tuple[str, ...] = ("P1_PCV02D",)
             ) -> pd.DataFrame:
    """Pipeline completo do M2, na ordem em que as decisoes se aplicam."""
    X = X.drop(columns=[COL_TEMPO], errors="ignore")
    X = remover_constantes(X)
    X = tratar_valores_impossiveis(X)
    for tag in com_flags_repouso:
        if tag in X.columns:
            X[f"{tag}_fora_repouso"] = taxa_desvio_repouso(X[tag])
    X = tratar_redundancia(X)
    return X


# --- D4: normalizacao em dois ramos ------------------------------------------
def ajustar_scaler(X_treino: pd.DataFrame) -> RobustScaler:
    """Ajusta o RobustScaler APENAS no trecho de treino.

    RobustScaler (mediana e IQR) porque ha outliers relevantes que sao sinal.
    O `fit` jamais ve o trecho de teste: em serie temporal isso vazaria
    informacao do futuro para o passado.
    Para modelos de arvore/boosting a normalizacao e dispensavel - nao chame.
    """
    return RobustScaler().fit(X_treino)


def aplicar_scaler(scaler: RobustScaler, X: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)


# --- janelas (ponte para o M3) ------------------------------------------------
def janelas(X: pd.DataFrame, y: pd.Series | None = None, tamanho: int = 60,
            passo: int = 30, corte: int | None = None):
    """Estatisticas moveis por janela, com corte cronologico respeitado.

    `corte` e o indice da fronteira treino/teste: janelas que a cruzam sao
    descartadas, para que nenhuma amostra contenha dados dos dois lados.
    Uma janela e rotulada como ataque se qualquer segundo dela for ataque.
    """
    feats, rot, inicios = [], [], []
    vals = X.to_numpy(dtype="float32")
    for i in range(0, len(X) - tamanho + 1, passo):
        j = i + tamanho
        if corte is not None and i < corte < j:
            continue
        bloco = vals[i:j]
        feats.append(np.concatenate([bloco.mean(0), bloco.std(0),
                                     bloco[-1] - bloco[0]]))
        inicios.append(i)
        if y is not None:
            rot.append(int(y.iloc[i:j].max()))
    cols = ([f"{c}_media" for c in X.columns] + [f"{c}_desvio" for c in X.columns]
            + [f"{c}_delta" for c in X.columns])
    F = pd.DataFrame(np.vstack(feats), columns=cols)
    F.insert(0, "inicio_seg", inicios)
    return (F, pd.Series(rot, name="label")) if y is not None else F
