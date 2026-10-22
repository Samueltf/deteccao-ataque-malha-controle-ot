"""Interpretabilidade do modelo final com SHAP.

O modelo final e um `Pipeline(StandardScaler, LogisticRegression)`. Para um
modelo linear o valor de Shapley tem **forma fechada** - nao ha amostragem nem
aproximacao:

    z = (x - media) / desvio          (o StandardScaler ajustado no treino)
    phi_i = coef_i * (z_i - E[z_i])   esperanca tomada no conjunto de fundo
    base  = intercepto + sum_i coef_i * E[z_i]

e vale exatamente `base + sum_i phi_i = logito(x)`. Por isso o calculo aqui e
feito direto, em vez de pelo `shap.LinearExplainer`: e exato, instantaneo e
verificavel (a funcao `conferir_aditividade` devolve o erro maximo, da ordem de
1e-12). A biblioteca `shap` continua sendo usada para os graficos, recebendo um
objeto `shap.Explanation` construido com esses valores.

Os valores saem em **log-odds**: positivo empurra a janela para "ataque",
negativo para "normal", e a magnitude e a forca da contribuicao. Como o
escalonamento e linear por coluna, cada phi_i pertence a UMA variavel original -
o grafico mostra o nome e o valor na unidade original, que e o que o operador le.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

SUFIXOS = {"_media": "média", "_desvio": "variação", "_delta": "tendência"}


def partes(coluna: str) -> tuple[str, str]:
    """Separa 'P1_PCV01D_desvio' em ('P1_PCV01D', '_desvio')."""
    for s in SUFIXOS:
        if coluna.endswith(s):
            return coluna[: -len(s)], s
    return coluna, ""


class ExplicadorLinear:
    """SHAP exato para `Pipeline(StandardScaler, LogisticRegression)`."""

    def __init__(self, pipeline, fundo: pd.DataFrame):
        self.pipeline = pipeline
        self.colunas = list(fundo.columns)
        escalador, linear = pipeline[:-1], pipeline[-1]
        self.coef = linear.coef_.ravel().astype("float64")
        self.intercepto = float(linear.intercept_[0])
        self.media_fundo = escalador.transform(fundo).mean(axis=0)
        self.base = self.intercepto + float(self.coef @ self.media_fundo)

    def __call__(self, X: pd.DataFrame) -> shap.Explanation:
        Z = self.pipeline[:-1].transform(X[self.colunas])
        phi = (Z - self.media_fundo) * self.coef
        return shap.Explanation(values=phi, base_values=np.full(len(X), self.base),
                                data=X[self.colunas].to_numpy(),
                                feature_names=self.colunas)

    def conferir_aditividade(self, X: pd.DataFrame) -> float:
        """Erro maximo entre `base + sum(phi)` e o logito previsto pelo modelo.

        A comparacao e feita contra `decision_function`, nao contra o logito de
        `predict_proba`: a probabilidade satura em 0 e 1 em ponto flutuante, e o
        logito dela truncaria em ~+-36, escondendo o resultado real.
        """
        expl = self(X)
        logito = self.pipeline.decision_function(X[self.colunas])
        return float(np.abs(expl.values.sum(1) + self.base - logito).max())


def importancia_por_feature(expl: shap.Explanation) -> pd.Series:
    return pd.Series(np.abs(expl.values).mean(0),
                     index=expl.feature_names).sort_values(ascending=False)


def importancia_por_tag(expl: shap.Explanation) -> pd.Series:
    """Agrega |SHAP| medio por tag fisica: 'P1_PCV01D' em vez de tres colunas.

    Para quem opera a planta, media, variacao e tendencia da mesma valvula sao a
    mesma valvula. A quebra por sufixo fica para quem for depurar o modelo.
    """
    imp = importancia_por_feature(expl)
    return imp.groupby([partes(c)[0] for c in imp.index]).sum().sort_values(ascending=False)
