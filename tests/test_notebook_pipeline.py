"""Testes de pipeline do notebook (não dependem de executar o ``.ipynb``).

Estes testes exercitam os mesmos estágios do `gzcmd_record_linkage` que o notebook
demonstra, porém de forma rápida e isolada, usando o CSV sintético versionado em
``data/synthetic/comparador_sintetico.csv``. Crescem junto com a Wave 2/3 do plano.

- **TST2.2.a**: a carga via ``load_comparador_csv`` produz as colunas engenheiradas
  esperadas, com *ranges* válidos (escores agregados em [0, 1]).
"""

from __future__ import annotations

import pandas as pd
import pytest
from _nb_paths import DATA_SYNTHETIC_DIR

from gzcmd_record_linkage.loader import LoadConfig, load_comparador_csv

CSV_PATH = DATA_SYNTHETIC_DIR / "comparador_sintetico.csv"


@pytest.fixture(scope="module")
def df_loaded() -> pd.DataFrame:
    assert CSV_PATH.exists(), (
        f"CSV sintético não encontrado em {CSV_PATH}. "
        "Gere-o executando o notebook ou synthetic_data.to_comparador_csv(...)."
    )
    return load_comparador_csv(CSV_PATH, cfg=LoadConfig(macd_enabled=True))


def test_tst_2_2_a_colunas_engenheiradas_presentes(df_loaded: pd.DataFrame) -> None:
    """As colunas agregadas, flags e MACD esperadas existem após a carga."""
    agregadas = [
        "nota_final",
        "TARGET",
        "nome_score_total",
        "mae_score_total",
        "dtnasc_score_total",
        "endereco_score_total",
        "municipio_score",
    ]
    flags = ["mae_missing", "dtnasc_all_zero", "endereco_zero", "diff_ano"]
    faltando = [c for c in (*agregadas, *flags) if c not in df_loaded.columns]
    assert not faltando, f"Colunas engenheiradas ausentes: {faltando}"

    macd_cols = [c for c in df_loaded.columns if c.startswith("macd_")]
    assert macd_cols, "Nenhuma coluna MACD foi produzida com macd_enabled=True."


def test_tst_2_2_a_ranges_validos(df_loaded: pd.DataFrame) -> None:
    """Escores agregados em [0, 1]; nota_final em faixa plausível; TARGET binário."""
    for col in [
        "nome_score_total",
        "mae_score_total",
        "dtnasc_score_total",
        "endereco_score_total",
        "municipio_score",
    ]:
        serie = df_loaded[col].astype(float)
        assert serie.min() >= 0.0 - 1e-9, f"{col} abaixo de 0"
        assert serie.max() <= 1.0 + 1e-9, f"{col} acima de 1"

    nota = df_loaded["nota_final"].astype(float)
    assert nota.min() >= 0.0 - 1e-9
    assert nota.max() <= 11.0  # escala ~0–10 + folga para âncora high

    assert set(df_loaded["TARGET"].astype(int).unique()) <= {0, 1}


def test_tst_2_2_a_ambas_as_classes_presentes(df_loaded: pd.DataFrame) -> None:
    """O dataset tem ambas as classes (pré-requisito para calibração/avaliação)."""
    contagem = df_loaded["TARGET"].astype(int).value_counts()
    assert contagem.get(0, 0) > 0, "Sem exemplos não-match (TARGET=0)."
    assert contagem.get(1, 0) > 0, "Sem exemplos match (TARGET=1)."
