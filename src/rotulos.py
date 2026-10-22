# -*- coding: utf-8 -*-
"""Nomes das variaveis em linguagem de quem opera a planta.

O usuario do dashboard nao e quem treinou o modelo: `P1_PCV01D_desvio` nao diz
nada a um operador de sala de controle. Este modulo traduz cada tag para o nome
do equipamento e da grandeza, com a unidade.

As descricoes vem da tabela de pontos do manual tecnico oficial do HAI
(`hai_dataset_technical_details.pdf`, secao DATA POINTS, pontos 1 a 87),
traduzidas. As variaveis derivadas criadas no M2 - residuos comando/
realimentacao, flags de valor impossivel e de desvio do repouso - nao estao no
manual e estao marcadas como derivadas.
"""
from __future__ import annotations

# tag -> (rotulo em portugues, unidade, origem)
TAGS: dict[str, tuple[str, str, str]] = {
    # --- P1: caldeira / trocador de calor --------------------------------------
    "P1_FCV01D": ("Comando de abertura da válvula de vazão FCV01", "%", "manual"),
    "P1_FCV02D": ("Comando de abertura da válvula de vazão FCV02", "%", "manual"),
    "P1_FCV03D": ("Comando de abertura da válvula de vazão FCV03", "%", "manual"),
    "P1_FT01": ("Vazão medida no tanque de retorno (FT01)", "mmH2O", "manual"),
    "P1_FT01Z": ("Vazão de entrada convertida (FT01)", "l/h", "manual"),
    "P1_FT02": ("Vazão medida no tanque de água de aquecimento", "mmH2O", "manual"),
    "P1_FT02Z": ("Vazão de saída convertida (FT02)", "l/h", "manual"),
    "P1_FT03": ("Vazão medida no tanque de retorno (FT03)", "mmH2O", "manual"),
    "P1_FT03Z": ("Vazão de saída convertida (FT03)", "l/h", "manual"),
    "P1_LCV01D": ("Comando de abertura da válvula de nível LCV01", "%", "manual"),
    "P1_LIT01": ("Nível do tanque de retorno", "mm", "manual"),
    "P1_PCV01D": ("Comando de abertura da válvula de pressão PCV01", "%", "manual"),
    "P1_PCV02D": ("Comando de abertura da válvula de pressão PCV02", "%", "manual"),
    "P1_PIT01": ("Pressão na saída do trocador de calor", "bar", "manual"),
    "P1_PIT02": ("Pressão de recalque da bomba de água de aquecimento", "bar", "manual"),
    "P1_PP04": ("Saída de controle da bomba de resfriamento", "%", "manual"),
    "P1_PP04D": ("Comando de acionamento da bomba de resfriamento", "liga/desliga",
                 "coluna presente no CSV, ausente da tabela de pontos do manual"),
    "P1_PP04SP": ("Setpoint de temperatura do resfriamento", "°C", "manual"),
    "P1_TIT01": ("Temperatura na saída do trocador de calor", "°C", "manual"),
    "P1_TIT02": ("Temperatura do tanque de água de aquecimento", "°C", "manual"),
    "P1_TIT03": ("Temperatura do tanque principal", "°C", "manual"),
    # --- P2: turbina a vapor ----------------------------------------------------
    "P2_24Vdc": ("Tensão de alimentação 24 Vcc do DCS", "V", "manual"),
    "P2_AutoSD": ("Demanda de rotação em modo automático", "RPM", "manual"),
    "P2_Emerg": ("Botão de emergência", "liga/desliga", "manual"),
    "P2_MASW": ("Chave manual (1) / automático (0)", "liga/desliga", "manual"),
    "P2_OnOff": ("Chave liga/desliga do DCS da turbina", "liga/desliga", "manual"),
    "P2_SCO": ("Saída do controlador de velocidade", "—", "manual"),
    "P2_SCST": ("Variação de rotação proporcional à frequência da turbina", "RPM", "manual"),
    "P2_SIT01": ("Rotação medida da turbina", "RPM", "manual"),
    "P2_VIBTR01": ("Vibração do eixo — eixo Y, 1º disco", "µm", "manual"),
    "P2_VIBTR02": ("Vibração do eixo — eixo X, 1º disco", "µm", "manual"),
    "P2_VIBTR03": ("Vibração do eixo — eixo Y, 2º disco", "µm", "manual"),
    "P2_VIBTR04": ("Vibração do eixo — eixo X, 2º disco", "µm", "manual"),
    "P2_VT01": ("Defasagem da sonda de fase do eixo", "rad/s", "manual"),
    # --- P3: reservatório superior ----------------------------------------------
    "P3_FIT01": ("Vazão de entrada no reservatório superior", "—", "manual"),
    "P3_LCP01D": ("Comando de velocidade da bomba LCP01", "—", "manual"),
    "P3_LCV01D": ("Comando de abertura da válvula LCV01 (reservatório superior)", "—", "manual"),
    "P3_LIT01": ("Nível do reservatório superior", "%", "manual"),
    "P3_PIT01": ("Pressão de entrada no reservatório superior", "—", "manual"),
    # --- P4: simulação HIL de geração -------------------------------------------
    "P4_HT_FD": ("Desvio de frequência da hidrelétrica", "mHz", "manual"),
    "P4_HT_PO": ("Potência gerada pela hidrelétrica", "MW", "manual"),
    "P4_HT_PS": ("Potência programada da hidrelétrica", "MW", "manual"),
    "P4_LD": ("Demanda elétrica total", "MW", "manual"),
    "P4_ST_FD": ("Desvio de frequência da turbina a vapor", "Hz", "manual"),
    "P4_ST_GOV": ("Abertura do governador da turbina a vapor", "—", "manual"),
    "P4_ST_LD": ("Demanda elétrica da turbina a vapor", "MW", "manual"),
    "P4_ST_PO": ("Potência gerada pela turbina a vapor", "MW", "manual"),
    "P4_ST_PS": ("Potência programada da turbina a vapor", "MW", "manual"),
    "P4_ST_PT01": ("Pressão de vapor da turbina (valor digital)", "—", "manual"),
    # --- pontos internos do algoritmo do controlador -----------------------------
    "x1001_05_SETPOINT_OUT": ("Setpoint interno do bloco 1001.05 — malha de pressão", "—", "manual"),
    "x1001_15_ASSIGN_OUT": ("Saída do bloco 1001.15 — malha de pressão", "—", "manual"),
    "x1002_07_SETPOINT_OUT": ("Setpoint interno do bloco 1002.07 — malha de nível", "—", "manual"),
    "x1002_08_SETPOINT_OUT": ("Setpoint interno do bloco 1002.08 — malha de nível", "—", "manual"),
    "x1003_18_SETPOINT_OUT": ("Setpoint interno do bloco 1003.18 — malha de temperatura", "—", "manual"),
    "x1003_24_SUM_OUT": ("Saída do somador 1003.24 — malha de temperatura", "—", "manual"),
    # --- variáveis derivadas, criadas no Módulo 2 --------------------------------
    "P1_FCV01_residuo": ("Desvio entre comando e posição real da válvula FCV01", "p.p.", "derivada (M2)"),
    "P1_FCV02_residuo": ("Desvio entre comando e posição real da válvula FCV02", "p.p.", "derivada (M2)"),
    "P1_FCV03_residuo": ("Desvio entre comando e posição real da válvula FCV03", "p.p.", "derivada (M2)"),
    "P1_LCV01_residuo": ("Desvio entre comando e posição real da válvula LCV01", "p.p.", "derivada (M2)"),
    "P1_PCV01_residuo": ("Desvio entre comando e posição real da válvula PCV01", "p.p.", "derivada (M2)"),
    "P1_PCV02_residuo": ("Desvio entre comando e posição real da válvula PCV02", "p.p.", "derivada (M2)"),
    "P1_PCV02D_fora_repouso": ("Válvula PCV02 fora do valor de repouso", "indicador", "derivada (M2)"),
    "P3_LCP01D_impossivel": ("Comando da bomba LCP01 em valor impossível (negativo)", "indicador", "derivada (M2)"),
    "P3_LCV01D_impossivel": ("Comando da válvula LCV01 em valor impossível (negativo)", "indicador", "derivada (M2)"),
}

# sufixo da janela -> como descrever a estatistica para o usuario
SUFIXOS: dict[str, str] = {
    "_media": "média no minuto",
    "_desvio": "variação no minuto",
    "_delta": "tendência no minuto",
}


def separar(coluna: str) -> tuple[str, str]:
    """'P1_PCV01D_desvio' -> ('P1_PCV01D', '_desvio')."""
    for s in SUFIXOS:
        if coluna.endswith(s):
            return coluna[: -len(s)], s
    return coluna, ""


def rotular(coluna: str, com_estatistica: bool = True, com_unidade: bool = False) -> str:
    """Nome legivel de uma coluna de feature.

    >>> rotular('P1_PCV01D_desvio')
    'Comando de abertura da válvula de pressão PCV01 — variação no minuto'
    """
    tag, suf = separar(coluna)
    nome, unidade, _ = TAGS.get(tag, (tag, "", "não catalogada"))
    if com_unidade and unidade not in ("", "—"):
        nome = f"{nome} ({unidade})"
    if com_estatistica and suf:
        nome = f"{nome} — {SUFIXOS[suf]}"
    return nome


def unidade(coluna: str) -> str:
    return TAGS.get(separar(coluna)[0], ("", "", ""))[1]


def origem(coluna: str) -> str:
    """De onde veio a descricao: 'manual' ou 'derivada (M2)'."""
    return TAGS.get(separar(coluna)[0], ("", "", "não catalogada"))[2]


def mapa(colunas) -> dict[str, str]:
    """Dicionario pronto para `DataFrame.rename(columns=...)` antes de exibir."""
    return {c: rotular(c) for c in colunas}
