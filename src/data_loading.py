"""Download e carregamento do HAI Security Dataset 23.05.

Os CSV brutos NAO sao versionados (ver .gitignore). Este modulo baixa os
arquivos sob demanda a partir do repositorio oficial e os carrega com tipos
economicos de memoria.

Fonte: https://github.com/icsdataset/hai  (licenca CC BY-SA 4.0)
Os arquivos estao sob Git LFS; o endpoint `media.githubusercontent.com`
entrega o conteudo real (o `raw.githubusercontent.com` devolve so o ponteiro).
"""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import pandas as pd

BASE_URL = "https://media.githubusercontent.com/media/icsdataset/hai/master/hai-23.05"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

ARQUIVOS_NORMAIS = ["hai-train1", "hai-train2", "hai-train3", "hai-train4"]
ARQUIVOS_ATAQUE = ["hai-test1", "hai-test2"]
ARQUIVOS_ROTULO = ["label-test1", "label-test2"]

COL_TEMPO = "timestamp"
COL_ALVO = "label"


def baixar(nome: str, destino: Path | None = None, forcar: bool = False) -> Path:
    """Baixa um CSV do dataset se ele ainda nao existir localmente."""
    destino = Path(destino or RAW_DIR)
    destino.mkdir(parents=True, exist_ok=True)
    alvo = destino / f"{nome}.csv"
    if alvo.exists() and not forcar:
        return alvo
    urllib.request.urlretrieve(f"{BASE_URL}/{nome}.csv", alvo)
    return alvo


def baixar_tudo(destino: Path | None = None) -> list[Path]:
    todos = ARQUIVOS_NORMAIS + ARQUIVOS_ATAQUE + ARQUIVOS_ROTULO
    return [baixar(n, destino) for n in todos]


def sha256(caminho: Path, blocos: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as fh:
        for bloco in iter(lambda: fh.read(blocos), b""):
            h.update(bloco)
    return h.hexdigest()


def carregar(nome: str, destino: Path | None = None) -> pd.DataFrame:
    """Carrega um CSV do dataset, com timestamp em datetime e sensores em float32."""
    caminho = baixar(nome, destino)
    df = pd.read_csv(caminho)
    if COL_TEMPO in df.columns:
        df[COL_TEMPO] = pd.to_datetime(df[COL_TEMPO])
    for c in df.columns:
        if c not in (COL_TEMPO, COL_ALVO):
            df[c] = df[c].astype("float32")
    return df


def carregar_rotulado(destino: Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Junta os dois arquivos com ataque e seus rotulos, em ordem cronologica.

    Retorna (X com timestamp, y). test1 (12-13/08) vem antes de test2 (17-19/08).
    """
    partes, rotulos = [], []
    for dados, rotulo in zip(ARQUIVOS_ATAQUE, ARQUIVOS_ROTULO):
        partes.append(carregar(dados, destino))
        rotulos.append(carregar(rotulo, destino)[COL_ALVO].astype("int8"))
    X = pd.concat(partes, ignore_index=True)
    y = pd.concat(rotulos, ignore_index=True)
    assert len(X) == len(y), "dados e rotulos com tamanhos diferentes"
    return X, y


def carregar_normal(destino: Path | None = None) -> pd.DataFrame:
    """Junta os quatro arquivos de operacao normal (sem ataque)."""
    return pd.concat([carregar(n, destino) for n in ARQUIVOS_NORMAIS], ignore_index=True)
