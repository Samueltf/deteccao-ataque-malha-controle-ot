"""M4 - prepara os dados leves que o dashboard consome.

Reproduz exatamente o pipeline do M2/M3 e grava em data/processed:
  janelas_teste.parquet  - as janelas do trecho de teste, com rotulo e timestamp
  fundo_shap.parquet     - amostra do treino usada como background do SHAP
Nenhum CSV bruto e versionado (ver .gitignore); este script os baixa sob demanda.
"""
import sys, pickle, json
from pathlib import Path
import numpy as np, pandas as pd
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src import data_loading as dl, preprocess as pp, avaliacao as av

SEMENTE, TAM, PASSO = 42, 60, 30
CORTE_TREINO, CORTE_TESTE = 144180, 192240

X_bruto, y = dl.carregar_rotulado(RAIZ / 'data' / 'raw')
ts = pd.to_datetime(X_bruto['timestamp'])
X = pp.preparar(X_bruto)
print('segundos', len(X), '| variaveis', X.shape[1], '| ataque', round(y.mean()*100, 2), '%')

F, rot = pp.janelas(X, y, tamanho=TAM, passo=PASSO, corte=CORTE_TESTE)
ini = F['inicio_seg'].to_numpy()
Xw = F.drop(columns=['inicio_seg'])
m_tr = (ini + TAM) <= CORTE_TREINO
m_va = (ini >= CORTE_TREINO) & (ini + TAM <= CORTE_TESTE)
m_te = ini >= CORTE_TESTE
print('janelas treino/valid/teste:', m_tr.sum(), m_va.sum(), m_te.sum())

# --- teste: o que o app mostra -------------------------------------------------
teste = Xw[m_te].reset_index(drop=True).astype('float32')
teste.insert(0, 'inicio_seg', ini[m_te])
teste.insert(1, 'instante', ts.to_numpy()[ini[m_te]])
teste.insert(2, 'rotulo', rot.to_numpy()[m_te].astype('int8'))
teste.to_parquet(RAIZ / 'data' / 'processed' / 'janelas_teste.parquet', index=False)

# --- validacao: necessaria para recalibrar o limiar quando o usuario mexe nele --
valid = Xw[m_va].reset_index(drop=True).astype('float32')
valid.insert(0, 'inicio_seg', ini[m_va])
valid.insert(1, 'rotulo', rot.to_numpy()[m_va].astype('int8'))
valid.to_parquet(RAIZ / 'data' / 'processed' / 'janelas_validacao.parquet', index=False)

# --- fundo do SHAP: amostra do treino -----------------------------------------
rng = np.random.default_rng(SEMENTE)
idx = rng.choice(np.flatnonzero(m_tr), size=500, replace=False)
Xw.iloc[idx].reset_index(drop=True).astype('float32').to_parquet(
    RAIZ / 'data' / 'processed' / 'fundo_shap.parquet', index=False)

# --- rotulo por segundo do trecho de teste, para a linha do tempo --------------
seg = pd.DataFrame({'instante': ts.to_numpy()[CORTE_TESTE:], 'rotulo': y.to_numpy()[CORTE_TESTE:]})
seg.to_parquet(RAIZ / 'data' / 'processed' / 'rotulo_segundo_teste.parquet', index=False)

meta = {'tamanho_janela': TAM, 'passo': PASSO, 'corte_treino': CORTE_TREINO,
        'corte_teste': CORTE_TESTE, 'semente': SEMENTE,
        'n_segundos_teste': int(len(X) - CORTE_TESTE),
        'n_eventos_teste': len([e for e in av.eventos(y) if e[0] >= CORTE_TESTE]),
        'primeiro_instante_teste': str(ts.iloc[CORTE_TESTE]),
        'ultimo_instante_teste': str(ts.iloc[-1]),
        'features': list(Xw.columns)}
(RAIZ / 'data' / 'processed' / 'meta.json').write_text(json.dumps(meta, indent=1))
print('pronto')
