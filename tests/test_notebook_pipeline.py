"""Testes de pipeline do notebook (não dependem de executar o ``.ipynb``).

Estes testes exercitam os mesmos estágios do `gzcmd_record_linkage` que o notebook
demonstra, porém de forma rápida e isolada, usando o CSV sintético versionado em
``data/synthetic/comparador_sintetico.csv``. Crescem junto com a Wave 2/3 do plano.

- **TST2.2.a**: a carga via ``load_comparador_csv`` produz as colunas engenheiradas
  esperadas, com *ranges* válidos (escores agregados em [0, 1]).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import synthetic_data
from _nb_paths import DATA_SYNTHETIC_DIR

from gzcmd_record_linkage.bands import BandAssigner
from gzcmd_record_linkage.calibration import compute_p_cal, fit_platt_from_df
from gzcmd_record_linkage.config import load_config
from gzcmd_record_linkage.guardrails import apply_guardrails
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


@pytest.fixture(scope="module")
def band_assigner() -> BandAssigner:
    from importlib.resources import files

    config_path = str(files("gzcmd_record_linkage") / "gzcmd_v3_config.yaml")
    return BandAssigner.from_config(load_config(config_path))


def test_tst_2_3_a_fronteiras_de_banda(band_assigner: BandAssigner) -> None:
    """As bandas batem com as fronteiras da config, incluindo os edges de fronteira.

    Semântica: ``min <= nota < max`` (max exclusivo), exceto ``high`` cujo
    ``inclusive_max=True`` o torna ``min <= nota <= max``.
    """
    notas = pd.Series([0.0, 4.999, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 999.0])
    bandas = band_assigner.assign(notas).tolist()
    assert bandas == [
        "low",  # 0.0
        "low",  # 4.999 (< 5)
        "grey_low",  # 5.0 (fronteira inferior inclusiva)
        "grey_mid",  # 6.0
        "grey_high",  # 7.0
        "near_high",  # 8.0
        "high",  # 9.0
        "high",  # 10.0 (high inclui o max)
        "high",  # 999.0 (max=999 inclusivo)
    ]


def test_tst_2_3_a_assign_nao_muta_e_retorna_string(
    df_loaded: pd.DataFrame, band_assigner: BandAssigner
) -> None:
    """``assign`` não muta a entrada e devolve Series de strings de banda."""
    antes = df_loaded["nota_final"].copy()
    bandas = band_assigner.assign(df_loaded["nota_final"])
    pd.testing.assert_series_equal(df_loaded["nota_final"], antes)
    validas = {"low", "grey_low", "grey_mid", "grey_high", "near_high", "high"}
    assert set(bandas.dropna().unique()) <= validas


# ---------------------------------------------------------------------------
# Fase 2.4 — Calibração (Platt): rotas A/B, ausência de vazamento, recuperação p*
# ---------------------------------------------------------------------------
CLIP_MIN = 1e-6
CLIP_MAX = 0.999999


@pytest.fixture(scope="module")
def calib_data(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Dataset determinístico (seed 42) + posterior verdadeira ``p*`` alinhada.

    Regenera via ``synthetic_data`` para obter ``p_true`` (não presente no CSV),
    grava em CSV temporário e recarrega pelo ``loader`` — espelho fiel do notebook.
    A ordem das linhas é preservada (CSV sem índice), de modo que ``p_true`` se
    alinha por posição às linhas carregadas.
    """
    ds = synthetic_data.generate_comparador(seed=42, n_pairs=600, scenarios="all")
    path = tmp_path_factory.mktemp("calib") / "comparador.csv"
    synthetic_data.to_comparador_csv(ds.frame, path)
    df = load_comparador_csv(path, cfg=LoadConfig(macd_enabled=True))
    return df, ds.p_true.to_numpy(dtype=float)


def test_tst_2_4_a_p_cal_em_intervalo_e_monotono(
    calib_data: tuple[pd.DataFrame, np.ndarray],
) -> None:
    """Rota A: ``p_cal`` fica no clip [1e-6, 0.999999] e é monótono na nota."""
    df, _ = calib_data
    model = fit_platt_from_df(df)
    p_cal = compute_p_cal(df, method="platt", model=model)

    assert p_cal.min() >= CLIP_MIN - 1e-12
    assert p_cal.max() <= CLIP_MAX + 1e-12
    # Platt com slope > 0 é estritamente monótono na nota_final.
    assert model.slope > 0.0
    ordem = df["nota_final"].astype(float).argsort()
    p_ordenado = p_cal.to_numpy()[ordem.to_numpy()]
    assert np.all(np.diff(p_ordenado) >= -1e-9), (
        "p_cal deveria ser não-decrescente na nota"
    )


def test_tst_2_4_b_degrada_com_elegancia_classe_unica() -> None:
    """Degradação elegante: classe única falha explicitamente; poucos positivos OK.

    Com uma única classe, a Hessiana é singular e o ajuste é matematicamente
    indefinido. A biblioteca não produz lixo silencioso: levanta um ``RuntimeError``
    claro. Isso é o comportamento *gracioso* desejado (falha alta e nomeada, não NaN).
    """
    base = synthetic_data.generate_comparador(seed=7, n_pairs=40).frame

    # Classe única (todos match): ajuste indefinido -> erro explícito.
    df_uni = base.copy()
    df_uni["TARGET"] = 1
    df_uni["PAR"] = 1
    with pytest.raises(RuntimeError, match="(?i)platt"):
        fit_platt_from_df(df_uni)

    # Poucos positivos (2 em 40): ainda ajusta, com parâmetros finitos e p_cal válido.
    df_few = base.copy()
    df_few["TARGET"] = 0
    df_few.iloc[:2, df_few.columns.get_loc("TARGET")] = 1
    model_few = fit_platt_from_df(df_few)
    assert np.isfinite(model_few.slope) and np.isfinite(model_few.intercept)
    p_few = compute_p_cal(df_few, method="platt", model=model_few)
    assert p_few.between(CLIP_MIN - 1e-12, CLIP_MAX + 1e-12).all()


def test_tst_2_4_d_split_group_aware_sem_vazamento(
    calib_data: tuple[pd.DataFrame, np.ndarray],
) -> None:
    """Rota B: split por COMPREC/REFREC não compartilha grupos entre treino/teste."""
    df, _ = calib_data
    for split_by, col in [("comprec", "COMPREC"), ("refrec", "REFREC")]:
        train_idx, test_idx = synthetic_data.group_aware_split_indices(
            df, split_by=split_by, test_size=0.3, seed=42, group_stratify=True
        )
        grupos_train = set(df.iloc[train_idx][col].astype(str))
        grupos_test = set(df.iloc[test_idx][col].astype(str))
        assert grupos_train.isdisjoint(grupos_test), (
            f"Vazamento: grupos {col} compartilhados entre treino e teste."
        )
        # Cobre todas as linhas, sem sobreposição de índices.
        assert len(set(train_idx) & set(test_idx)) == 0
        assert len(train_idx) + len(test_idx) == len(df)


def test_tst_2_4_e_recupera_posterior_verdadeira(
    calib_data: tuple[pd.DataFrame, np.ndarray],
) -> None:
    """Held-out: Platt ajustado no treino recupera ``p*`` no teste (erro pequeno)."""
    df, p_true = calib_data
    train_idx, test_idx = synthetic_data.group_aware_split_indices(
        df, split_by="comprec", test_size=0.3, seed=42, group_stratify=True
    )
    model = fit_platt_from_df(df.iloc[train_idx])
    p_cal_test = compute_p_cal(df.iloc[test_idx], method="platt", model=model)
    mae = float(np.mean(np.abs(p_cal_test.to_numpy() - p_true[test_idx])))
    assert mae < 0.05, f"Erro médio |p_cal - p*| no teste = {mae:.4f} (esperado < 0.05)"


# ---------------------------------------------------------------------------
# Fase 2.5 — Guardrails determinísticos antes da política de custo
# ---------------------------------------------------------------------------
def test_tst_2_5_a_guardrails_disparam_com_motivos_corretos(
    df_loaded: pd.DataFrame,
) -> None:
    """Todos os tipos de guardrail disparam e carregam seus motivos esperados."""
    gout = apply_guardrails(df_loaded)
    fired = pd.DataFrame(
        {
            "guardrail": gout.guardrail,
            "reason": gout.reason,
        }
    ).dropna()

    assert {"ALWAYS_MATCH", "ALWAYS_NONMATCH", "FORCE_REVIEW"} <= set(
        fired["guardrail"]
    )

    expected_reason_to_guardrail = {
        "nota_final_high": "ALWAYS_MATCH",
        "nota_final_low": "ALWAYS_NONMATCH",
        "temporal_filter": "ALWAYS_NONMATCH",
        "homonimia_risk": "FORCE_REVIEW",
    }
    observed_reason_to_guardrail = dict(
        fired.groupby("reason", observed=True)["guardrail"].first()
    )

    for reason, guardrail in expected_reason_to_guardrail.items():
        assert observed_reason_to_guardrail[reason] == guardrail
