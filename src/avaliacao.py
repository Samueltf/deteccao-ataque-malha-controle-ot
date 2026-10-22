"""Metricas de avaliacao do projeto.

Duas familias convivem aqui:

* as metricas por ponto pedidas na disciplina (recall e F1 da classe minoritaria,
  AUC-ROC, PR-AUC, matriz de confusao), calculadas por `metricas_ponto`;
* as metricas por evento, que sao as metricas primarias deste projeto: um ataque
  de 132 s detectado no segundo 30 e um ataque detectado no segundo 130 valem o
  mesmo para o F1 por ponto, mas nao valem o mesmo para a operacao.

A metrica primaria e a **cobertura de eventos sob orcamento de alarme**: quantos
dos ataques rotulados o modelo acusa, limitado a no maximo um alarme falso por
hora. O limiar que respeita esse orcamento e calibrado em um trecho de validacao
cronologicamente anterior ao teste - nunca no proprio teste.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


# --- eventos ------------------------------------------------------------------
def eventos(y: np.ndarray | pd.Series) -> list[tuple[int, int]]:
    """Blocos contiguos de rotulo 1, como pares (primeiro_seg, ultimo_seg)."""
    v = np.asarray(y).astype(np.int8)
    d = np.diff(np.concatenate([[0], v, [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def janelas_para_segundos(inicios: np.ndarray, positivas: np.ndarray,
                          tamanho: int, n_seg: int) -> np.ndarray:
    """Converte predicao por janela em alarme por segundo.

    Uma janela acusada marca todos os seus `tamanho` segundos. E a forma mais
    conservadora: nao presume em que ponto da janela o modelo viu o ataque.
    """
    flag = np.zeros(n_seg, dtype=bool)
    for i in inicios[positivas.astype(bool)]:
        flag[i:i + tamanho] = True
    return flag


def metricas_evento(flag: np.ndarray, y: np.ndarray, ini: int, fim: int) -> dict:
    """Avalia os alarmes no trecho [ini, fim) por evento, nao por segundo."""
    y = np.asarray(y).astype(np.int8)
    f = flag.copy()
    f[:ini] = False
    f[fim:] = False
    evs = [(a, b) for a, b in eventos(y) if ini <= a < fim]

    detectados, latencias = 0, []
    for a, b in evs:
        dentro = np.flatnonzero(f[a:b + 1])
        if dentro.size:
            detectados += 1
            latencias.append(int(dentro[0]))

    # alarme falso = bloco contiguo de alarme que nao toca nenhum ataque
    falsos = 0
    for a, b in eventos(f[ini:fim].astype(np.int8)):
        if y[ini + a:ini + b + 1].max() == 0:
            falsos += 1
    horas = (fim - ini) / 3600

    # guarda contra alarme permanente: fracao do tempo NORMAL sob alarme.
    # sem ela, um modelo que acusa sempre marcaria "zero alarmes falsos"
    # (um unico bloco continuo que toca os ataques) e passaria no orcamento.
    normal = y[ini:fim] == 0
    frac = float(f[ini:fim][normal].mean()) if normal.any() else float("nan")

    return {
        "eventos": len(evs),
        "detectados": detectados,
        "cobertura": detectados / len(evs) if evs else float("nan"),
        "latencia_mediana": float(np.median(latencias)) if latencias else float("nan"),
        "latencia_p90": float(np.percentile(latencias, 90)) if latencias else float("nan"),
        "falsos_alarmes": falsos,
        "falsos_por_hora": falsos / horas,
        "falsos_por_turno": falsos / horas * 8,
        "fracao_normal_em_alarme": frac,
    }


# --- limiar sob orcamento de alarme -------------------------------------------
def calibrar_limiar(score: np.ndarray, inicios: np.ndarray, y_seg: np.ndarray,
                    ini: int, fim: int, tamanho: int,
                    orcamento_hora: float = 1.0, orcamento_tempo: float = 0.02,
                    n_grade: int = 60) -> float:
    """Menor limiar cujo custo de alarme falso cabe no orcamento, no trecho dado.

    O orcamento tem duas restricoes simultaneas: no maximo `orcamento_hora`
    blocos de alarme falso por hora (interrupcoes do operador) e no maximo
    `orcamento_tempo` do tempo normal sob alarme (guarda contra alarme
    permanente). Varre a grade de cima para baixo e devolve o limiar mais
    permissivo que respeita as duas. Chamada SEMPRE em trecho de validacao
    anterior ao teste: calibrar no teste seria escolher o limiar olhando a
    resposta.
    """
    grade = np.quantile(score, np.linspace(0.50, 0.9995, n_grade))
    # ponto de partida: acima do maior score, ou seja, nenhum alarme. So baixa
    # se algum limiar couber no orcamento - um modelo que nao consegue caber
    # fica sem acusar nada, que e o resultado correto para ele.
    escolhido = float(np.nextafter(score.max(), np.inf))
    for lim in sorted(set(grade.tolist()), reverse=True):
        flag = janelas_para_segundos(inicios, score >= lim, tamanho, len(y_seg))
        m = metricas_evento(flag, y_seg, ini, fim)
        if (m["falsos_por_hora"] <= orcamento_hora
                and m["fracao_normal_em_alarme"] <= orcamento_tempo):
            escolhido = float(lim)
        else:
            break
    return escolhido


# --- metricas por ponto (as pedidas na disciplina) ----------------------------
def metricas_ponto(y_true: np.ndarray, y_pred: np.ndarray,
                   score: np.ndarray | None = None) -> dict:
    m = {
        "recall_1": recall_score(y_true, y_pred, zero_division=0),
        "precisao_1": precision_score(y_true, y_pred, zero_division=0),
        "f1_1": f1_score(y_true, y_pred, zero_division=0),
    }
    if score is not None and len(np.unique(y_true)) > 1:
        m["auc_roc"] = roc_auc_score(y_true, score)
        m["pr_auc"] = average_precision_score(y_true, score)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    m |= {"vn": int(tn), "fp": int(fp), "fn": int(fn), "vp": int(tp)}
    return m
