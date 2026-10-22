# -*- coding: utf-8 -*-
"""Detecção de manipulação da malha de controle — dashboard.

Projeto Integrado em Ciência de Dados (ECD-108) — Módulo 4
PPGMCS/UNIMONTES · Samuel Rodrigues Gonçalves

Execução:
    pip install -r requirements.txt
    streamlit run app/app.py

O app não faz pré-processamento próprio: ele carrega o modelo serializado no M3
e as janelas já preparadas em data/processed, e chama as mesmas funções de src/
usadas nos notebooks. É isso que faz o dashboard bater com o relatório.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from matplotlib.patches import Patch

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src import avaliacao as av          # noqa: E402
from src import rotulos as rt            # noqa: E402
from src import shap_util as su          # noqa: E402

# --- identidade visual --------------------------------------------------------
S1, S2, S8, NEU = "#2a78d6", "#eb6834", "#e34948", "#c8ccd1"
SURF, TXT, TXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2df"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
    "axes.labelcolor": TXT2, "text.color": TXT, "xtick.color": TXT2, "ytick.color": TXT2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 120})


# =============================================================================
# Carregamento — protegido por cache: roda uma vez, não a cada interação
# =============================================================================
@st.cache_resource(show_spinner="Carregando modelo e calculando explicações…")
def carregar():
    modelo = pickle.loads((RAIZ / "models" / "modelo_final_logistica.pkl").read_bytes())
    proc = RAIZ / "data" / "processed"
    teste = pd.read_parquet(proc / "janelas_teste.parquet")
    fundo = pd.read_parquet(proc / "fundo_shap.parquet")[modelo["colunas"]]
    seg = pd.read_parquet(proc / "rotulo_segundo_teste.parquet")
    ataques = pd.read_csv(RAIZ / "data" / "externo" / "ataques_teste.csv")
    meta = json.loads((proc / "meta.json").read_text())

    cols = modelo["colunas"]
    X = teste[cols]
    teste = teste.copy()
    teste["prob"] = modelo["modelo"].predict_proba(X)[:, 1]

    explicador = su.ExplicadorLinear(modelo["modelo"], fundo)
    shap_valores = explicador(X).values

    # qual ataque cada janela cobre
    eventos = av.eventos(seg["rotulo"].to_numpy())
    ataques = ataques.copy()
    ataques["inicio_rel"] = [a for a, _ in eventos]
    ataques["fim_rel"] = [b for _, b in eventos]
    tam = meta["tamanho_janela"]
    ini_rel = teste["inicio_seg"].to_numpy() - meta["corte_teste"]
    teste["inicio_rel"] = ini_rel
    teste["ataque"] = ""
    for r in ataques.itertuples():
        teste.loc[(ini_rel + tam > r.inicio_rel) & (ini_rel <= r.fim_rel), "ataque"] = r.id
    teste["classe"] = teste["ataque"].map(dict(zip(ataques.id, ataques.classe))).fillna("normal")

    return modelo, teste, X, fundo, seg, ataques, meta, explicador, shap_valores, eventos


MODELO, TESTE, X, FUNDO, SEG, ATAQUES, META, EXPLICADOR, SHAP_VAL, EVENTOS = carregar()
COLS = MODELO["colunas"]
TAM = META["tamanho_janela"]
N_SEG = META["n_segundos_teste"]
Y_SEG = SEG["rotulo"].to_numpy()
INSTANTES = pd.to_datetime(SEG["instante"])
LIMIAR_M3 = float(MODELO["limiar"])


@st.cache_data(show_spinner=False)
def desempenho(limiar: float) -> dict:
    """Métricas por evento para um limiar qualquer — é o que o slider recalcula."""
    flag = av.janelas_para_segundos(TESTE["inicio_rel"].to_numpy(),
                                    (TESTE["prob"].to_numpy() >= limiar), TAM, N_SEG)
    m = av.metricas_evento(flag, Y_SEG, 0, N_SEG)
    det = {r.id: bool(flag[r.inicio_rel:r.fim_rel + 1].any()) for r in ATAQUES.itertuples()}
    m["por_classe"] = {c: (sum(det[i] for i in g.id), len(g))
                       for c, g in ATAQUES.groupby("classe")}
    m["detectados_por_id"] = det
    m["flag"] = flag
    return m


def cabecalho(fig, titulo, sub):
    fig.text(.005, 1.14, titulo, ha="left", fontsize=11.5, fontweight="bold", color=TXT)
    fig.text(.005, 1.03, sub, ha="left", fontsize=8.8, color=TXT2)


# =============================================================================
# Página
# =============================================================================
st.set_page_config(page_title="Detecção de manipulação da malha de controle",
                   page_icon="⚙", layout="wide")

# ---- Ato 1: o problema -------------------------------------------------------
st.title("Detecção de manipulação da malha de controle industrial")
st.markdown(
    "**O ataque que este sistema procura é o que não dispara alarme:** alguém com acesso ao "
    "controlador altera um setpoint ou o comando de uma válvula mantendo as variáveis dentro da "
    "faixa operacional. Abaixo, o que um detector treinado sobre as 86 tags do **HAI Security "
    "Dataset 23.05** consegue — e o que não consegue — em 25,6 h de operação com 18 ataques reais. "
    "**É apoio à decisão do operador, não substituto dela.**")

with st.expander("O problema, em detalhe"):
    st.markdown(
        "Um alarme convencional de limite só reage quando a variável sai da faixa. O ataque tratado "
        "aqui nunca a tira: ele desloca o ponto de operação e espera. Quando o efeito aparece, a "
        "manipulação já está em curso há minutos.\n\n"
        "Em julho de 2026 a CISA documentou campanha ativa com esse perfil contra PLCs Rockwell, "
        "Schneider e Siemens em água, energia e municípios (alerta AA26-097A): a lógica maliciosa "
        "preservava o funcionamento aparente, desabilitava desligamento e alarme e manipulava as "
        "telas de HMI/SCADA. O relatório anual da Dragos de 2026 registra tempo médio de permanência "
        "de 42 dias em redes OT.\n\n"
        "O HAI é uma bancada com simulação em tempo real (HIL) de geração térmica e hidrelétrica, "
        "operada por DCS e PLCs reais. Os 52 ataques do conjunto foram executados nela, com rótulo "
        "por segundo. Este painel usa o trecho final do dataset, que o modelo nunca viu no treino.")

st.warning(
    "**Este detector não enxerga ataque de ponto interno** — 7 dos 18 ataques alteram parâmetros do "
    "algoritmo do controlador, que não existem entre as 86 tags, e nenhum foi acusado. "
    "Dados de bancada, não de planta real · treinado em 40 h e 33 eventos · "
    "18 ataques no teste, então um evento a mais ou a menos move a cobertura em 5,6 pontos. "
    "Detalhe completo no fim da página.", icon="⚠")

aba1, aba2, aba3 = st.tabs(["Panorama da operação",
                            "Por que esta janela",
                            "O que o modelo valoriza"])

# =============================================================================
# Visão 1 — panorama
# =============================================================================
with aba1:
    topo_e, topo_d = st.columns([3, 1])
    topo_e.subheader("Qual é a situação geral?")
    with topo_d:
        limiar = st.slider(
            "Limiar de alarme", min_value=0.50, max_value=0.999, value=LIMIAR_M3, step=0.001,
            format="%.3f",
            help="Mais baixo acusa mais ataques e também mais alarmes falsos. O valor inicial é o "
                 "calibrado no Módulo 3, que cabe no orçamento de 1 alarme falso por hora.")

    d = desempenho(limiar)
    io_det, io_tot = d["por_classe"].get("io", (0, 0))
    int_det, int_tot = d["por_classe"].get("interno", (0, 0))

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Ataques acusados", f"{d['detectados']} de {d['eventos']}",
              help="Quantos dos ataques executados o sistema acusou pelo menos uma vez.")
    k2.metric("Alarmes falsos por hora", f"{d['falsos_por_hora']:.2f}",
              help="Blocos de alarme que não correspondem a ataque nenhum. O orçamento declarado "
                   "no projeto é no máximo 1 por hora.")
    k3.metric("Tempo até acusar",
              "—" if np.isnan(d["latencia_mediana"]) else f"{d['latencia_mediana']:.0f} s",
              help="Mediana do tempo entre o início do ataque e o primeiro alarme.")
    k4.metric("Ataques de ponto de E/S", f"{io_det} de {io_tot}",
              delta=f"ponto interno: {int_det} de {int_tot}", delta_color="off",
              help="Ponto de E/S: o atacante mexe em setpoint, medição ou comando — tags que "
                   "existem no historiador. Ponto interno: mexe em parâmetro do algoritmo do "
                   "controlador, que não é registrado.")
    if abs(limiar - LIMIAR_M3) > 1e-9:
        st.caption(f"Limiar alterado para {limiar:.3f}. O valor do relatório é {LIMIAR_M3:.3f}.")

    fig, ax = plt.subplots(figsize=(13, 3.2))
    t = INSTANTES.to_numpy()[TESTE["inicio_rel"].to_numpy()]
    for r in ATAQUES.itertuples():
        ax.axvspan(INSTANTES.iloc[r.inicio_rel], INSTANTES.iloc[max(r.fim_rel, r.inicio_rel + 240)],
                   color=S1 if r.classe == "io" else S8, alpha=.30, lw=0)
    ax.plot(t, TESTE["prob"], lw=.7, color=TXT2, alpha=.8)
    ax.axhline(limiar, color="#111", ls="--", lw=1.1)
    ac = TESTE[TESTE["prob"] >= limiar]
    ax.plot(INSTANTES.to_numpy()[ac["inicio_rel"].to_numpy()], np.full(len(ac), 1.06),
            "|", color=S2, ms=9, mew=1.6)
    ax.set_ylim(0, 1.13); ax.set_ylabel("probabilidade de ataque")
    ax.set_yticks([0, .5, 1.0])
    ax.legend(handles=[Patch(color=S1, alpha=.45, label="ataque de ponto de E/S"),
                       Patch(color=S8, alpha=.45, label="ataque de ponto interno"),
                       plt.Line2D([], [], color=S2, marker="|", ls="", ms=9, mew=1.6, label="alarme"),
                       plt.Line2D([], [], color="#111", ls="--", lw=1.1, label="limiar")],
              fontsize=8, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0))
    cabecalho(fig, "As 25,6 horas do conjunto de teste",
              "Faixas sombreadas são os ataques rotulados (alargadas para ficarem visíveis); "
              "os traços no topo são os alarmes do modelo.")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("##### Os 18 ataques, um a um")
    tabela = ATAQUES.assign(
        acusado=[("sim" if d["detectados_por_id"][i] else "não") for i in ATAQUES.id],
        classe=ATAQUES.classe.map({"io": "ponto de E/S", "interno": "ponto interno"}),
        duracao=ATAQUES.duracao_s.map(lambda s: f"{s // 60}min {s % 60:02d}s"),
    )[["id", "cenario", "classe", "inicio_rotulo", "duracao", "pontos_alvo", "acusado"]]
    tabela.columns = ["ataque", "cenário", "classe", "início", "duração",
                      "o que foi manipulado", "acusado"]
    st.dataframe(tabela, hide_index=True, use_container_width=True)
    st.caption("Fonte da tabela de ataques: manual técnico oficial do HAI 23.05, "
               "seção ATTACK OPERATION (ataques nº 35 a 52).")

# =============================================================================
# Visão 2 — predição individual
# =============================================================================
with aba2:
    st.subheader("O que o modelo diz sobre este caso — e por quê")
    d = desempenho(limiar)

    grupos = {"Ataques rotulados": "ataque", "Alarmes falsos": "falso", "Operação normal": "normal"}
    c1, c2 = st.columns([1, 2])
    with c1:
        grupo = st.selectbox("Que tipo de janela examinar", list(grupos))
    if grupos[grupo] == "ataque":
        with c2:
            escolhas = {
                f"{r.id} — {r.cenario} — "
                f"{'ponto de E/S' if r.classe == 'io' else 'ponto interno'} — "
                f"{'acusado' if d['detectados_por_id'][r.id] else 'não acusado'}": r.id
                for r in ATAQUES.itertuples()}
            ocorrencia = st.selectbox("Ocorrência", list(escolhas))
        sub = TESTE[TESTE["ataque"] == escolhas[ocorrencia]]
    elif grupos[grupo] == "falso":
        sub = TESTE[(TESTE["rotulo"] == 0) & (TESTE["prob"] >= limiar)]
        with c2:
            st.caption(f"{len(sub)} janelas normais em que o modelo disparou alarme.")
    else:
        sub = TESTE[(TESTE["rotulo"] == 0) & (TESTE["prob"] < limiar)].iloc[::17]
        with c2:
            st.caption("Amostra de janelas sem ataque e sem alarme, para comparação.")

    if len(sub) == 0:
        st.info("Nenhuma janela neste grupo com o limiar atual.")
    else:
        opcoes = {f"{pd.Timestamp(r.instante):%d/%m %H:%M:%S}  ·  probabilidade {r.prob:.3f}": r.Index
                  for r in sub.itertuples()}
        rotulo_sel = st.selectbox("Janela de 60 segundos", list(opcoes),
                                  index=int(np.argmax(sub["prob"].to_numpy())))
        i = opcoes[rotulo_sel]
        linha = TESTE.loc[i]

        m1, m2, m3 = st.columns(3)
        m1.metric("Probabilidade de ataque", f"{linha.prob:.3f}")
        m2.metric("Veredito", "ALARME" if linha.prob >= limiar else "sem alarme",
                  help=f"Limiar em uso: {limiar:.3f}")
        m3.metric("Rótulo real", "ataque" if linha.rotulo else "operação normal")

        pos = int(TESTE.index.get_loc(i))
        e = shap.Explanation(values=SHAP_VAL[pos],
                             base_values=EXPLICADOR.base,
                             data=X.iloc[pos].to_numpy(),
                             feature_names=[rt.rotular(c) for c in COLS])
        plt.figure()
        shap.plots.waterfall(e, max_display=11, show=False)
        fig = plt.gcf(); fig.set_size_inches(10, 4.2)
        plt.gca().tick_params(axis="y", labelsize=8)
        for ax in fig.axes:
            for eixo in (ax.yaxis, ax.xaxis):
                antigos = [tl.get_text() for tl in eixo.get_ticklabels()]
                novos = [a.replace("$", "").replace("E[f(X)]", "base").replace("f(x)", "saída")
                         .replace(" other features", " outras variáveis").replace("Sum of ", "soma de ")
                         if ("f(" in a or "other features" in a) else a for a in antigos]
                if novos != antigos:
                    eixo.set_ticks(eixo.get_ticklocs()); eixo.set_ticklabels(novos)
        cabecalho(fig, "Quanto cada variável empurrou esta janela",
                  "Barras vermelhas empurram para ataque, azuis para normal. "
                  "De 'base' (operação típica) até 'saída' (esta janela).")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        with st.expander("E se a planta estivesse diferente? — simulador"):
            st.caption(
                "Altere o valor de uma variável nesta janela e veja a probabilidade mudar. "
                "Como o modelo é linear, o efeito é exato: a contribuição muda na proporção do "
                "deslocamento. Serve para responder 'o quanto teria de mudar para disparar?'.")
            imp = pd.Series(np.abs(SHAP_VAL).mean(0), index=COLS).sort_values(ascending=False)
            top = list(imp.head(20).index)
            cs1, cs2 = st.columns([2, 1])
            with cs1:
                alvo = st.selectbox("Variável", top, format_func=lambda c: rt.rotular(c, com_unidade=True))
            faixa_lo, faixa_hi = float(X[alvo].quantile(.01)), float(X[alvo].quantile(.99))
            atual = float(X[alvo].iloc[pos])
            lo, hi = min(faixa_lo, atual), max(faixa_hi, atual)
            if hi - lo < 1e-9:
                hi = lo + 1.0
            with cs2:
                st.metric("Valor nesta janela", f"{atual:,.3f}".replace(",", "."))
            novo = st.slider("Valor simulado", min_value=float(lo), max_value=float(hi),
                             value=float(atual), step=float((hi - lo) / 200))
            simulada = X.iloc[[pos]].copy()
            simulada[alvo] = novo
            p_nova = float(MODELO["modelo"].predict_proba(simulada)[:, 1][0])
            r1, r2 = st.columns(2)
            r1.metric("Probabilidade simulada", f"{p_nova:.3f}", delta=f"{p_nova - linha.prob:+.3f}")
            r2.metric("Veredito simulado", "ALARME" if p_nova >= limiar else "sem alarme")
            st.caption("Simulação hipotética sobre o modelo; não é uma previsão do comportamento "
                       "físico da planta, que tem acoplamento entre as variáveis.")

# =============================================================================
# Visão 3 — interpretabilidade global
# =============================================================================
with aba3:
    st.subheader("Quais fatores o modelo considera relevantes no geral?")
    expl = shap.Explanation(values=SHAP_VAL, base_values=EXPLICADOR.base,
                            data=X.to_numpy(), feature_names=COLS)

    modo = st.radio("Agrupar por", ["Equipamento (64 variáveis físicas)",
                                    "Coluna do modelo (192 features)"], horizontal=True,
                    help="Cada variável física entra no modelo três vezes: média, variação e "
                         "tendência dentro da janela de 60 s. Agrupar soma as três.")
    por_equipamento = modo.startswith("Equipamento")

    g1, g2 = st.columns([1, 1])
    with g1:
        serie = (su.importancia_por_tag(expl) if por_equipamento
                 else su.importancia_por_feature(expl)).head(14)[::-1]
        rotulos_y = [rt.rotular(t, com_estatistica=not por_equipamento) for t in serie.index]
        cor = [S2 if rt.origem(t).startswith("derivada") else S1 for t in serie.index]
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        ax.barh(range(len(serie)), serie.values, color=cor, height=.7)
        ax.set_yticks(range(len(serie)))
        ax.set_yticklabels([r if len(r) < 58 else r[:55] + "…" for r in rotulos_y], fontsize=7.8)
        ax.set_xlabel("influência média sobre a decisão (|SHAP|, log-odds)", fontsize=8.5)
        ax.grid(axis="y", visible=False); ax.set_axisbelow(True)
        ax.legend(handles=[Patch(color=S1, label="tag do historiador"),
                           Patch(color=S2, label="variável derivada (M2)")],
                  fontsize=8, loc="lower right")
        cabecalho(fig, "O que mais pesa na decisão",
                  "Quanto maior a barra, mais aquela variável move o resultado do modelo.")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with g2:
        nomes = [rt.rotular(c) for c in COLS]
        e_leg = shap.Explanation(values=SHAP_VAL, base_values=EXPLICADOR.base,
                                 data=X.to_numpy(), feature_names=nomes)
        plt.figure()
        shap.plots.beeswarm(e_leg, max_display=12, show=False, color_bar=False)
        fig = plt.gcf(); fig.set_size_inches(6.6, 4.6)
        ax = plt.gca()
        ax.tick_params(axis="y", labelsize=7.6)
        ax.set_xlabel("contribuição para a decisão (log-odds)", fontsize=8.5)
        antigos = [tl.get_text() for tl in ax.yaxis.get_ticklabels()]
        novos = [a.replace("Sum of ", "soma de ").replace(" other features", " outras variáveis")
                 for a in antigos]
        if novos != antigos:
            ax.yaxis.set_ticks(ax.yaxis.get_ticklocs()); ax.yaxis.set_ticklabels(novos)
        cabecalho(fig, "Cada ponto é uma janela de 60 s",
                  "À direita do zero, a variável empurrou aquela janela para ataque.")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("##### Por que ataque de ponto interno escapa")
    grupos_cl = {"operação normal": TESTE.classe == "normal",
                 "ataque de ponto de E/S": TESTE.classe == "io",
                 "ataque de ponto interno": TESTE.classe == "interno"}
    massa = np.abs(SHAP_VAL).sum(1)
    comp = pd.DataFrame([{
        "situação": g,
        "janelas": int(m.sum()),
        "probabilidade média": round(float(TESTE.prob[m].mean()), 3),
        "evidência que o modelo reúne": round(float(massa[m.to_numpy()].mean()), 1),
    } for g, m in grupos_cl.items()])
    st.dataframe(comp, hide_index=True, use_container_width=True)
    st.info(
        "Sob ataque de **ponto de E/S** o modelo reúne cerca de **quatro vezes** mais evidência do "
        "que em operação normal. Sob ataque de **ponto interno**, a evidência é praticamente a "
        "mesma da operação normal — por isso baixar o limiar não resolve: o alarme passaria a "
        "disparar também na operação normal. Detectar essa classe exige monitorar alteração de "
        "lógica e de parâmetros na estação de engenharia, não mais modelos sobre o historiador.",
        icon="ℹ")

    with st.expander("Tabela completa de influência"):
        completa = pd.DataFrame({
            "variável": [rt.rotular(c) for c in COLS],
            "coluna do modelo": COLS,
            "origem": [rt.origem(c) for c in COLS],
            "|SHAP| médio": np.abs(SHAP_VAL).mean(0).round(4),
        }).sort_values("|SHAP| médio", ascending=False)
        st.dataframe(completa, hide_index=True, use_container_width=True, height=320)

# =============================================================================
# Ato 4 — limitações, na íntegra
# =============================================================================
st.divider()
st.markdown("### Limitações, na íntegra")
lc1, lc2 = st.columns(2)
with lc1:
    st.markdown(
        "**Dos dados**\n\n"
        "- Bancada HIL do HAI 23.05, não planta real; os números não transferem sem recalibração.\n"
        "- 7 dos 18 ataques do teste manipulam parâmetros internos do algoritmo do controlador, "
        "que não estão entre as 86 tags. Nenhum modelo sobre esse conjunto resolve essa classe.\n"
        "- 18 eventos no teste: um a mais ou a menos move a cobertura em 5,6 pontos.\n"
        "- Rótulos por subprocesso foram removidos do dataset a partir da versão 22.04 — o escopo "
        "é detecção, sem localizar qual subprocesso foi atacado.")
with lc2:
    st.markdown(
        "**Do modelo e da operação**\n\n"
        "- Treinado em 40 h e 33 eventos; regimes não representados geram alarme falso.\n"
        "- O limiar é fixo e precisaria ser recalibrado periodicamente: a definição de \"normal\" "
        "muda com manutenção, estação do ano e mudança de produto.\n"
        "- A latência é medida offline, em dados de 1 Hz; não inclui coleta, transporte e "
        "apresentação do alarme.\n"
        "- **Quem decide é o operador.** O sistema prioriza o que investigar; não bloqueia, não "
        "atua sobre o processo e não substitui julgamento humano.")

st.caption(
    f"Modelo: regressão logística sobre janelas de {TAM} s · limiar do relatório {LIMIAR_M3:.3f} · "
    f"dados: HAI Security Dataset 23.05 (CC BY-SA 4.0), github.com/icsdataset/hai, acesso em "
    f"19/09/2026 · explicações SHAP em forma fechada, exatas · "
    f"Samuel Rodrigues Gonçalves — ECD-108, PPGMCS/UNIMONTES")
