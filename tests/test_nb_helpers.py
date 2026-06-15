"""Testes das métricas de calibração de ``nb_helpers`` (DEC-08 / TST2.4.c).

Casos fechados (valores conhecidos analiticamente) garantem que ECE e Brier
estão corretos, além de robustez a bins vazios e validação de entrada.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from nb_helpers import brier_score, expected_calibration_error, llm_review_stub


def test_brier_previsoes_perfeitas_eh_zero() -> None:
    y = np.array([0, 1, 0, 1, 1])
    p = y.astype(float)
    assert brier_score(y, p) == pytest.approx(0.0)


def test_brier_previsor_constante_base_rate() -> None:
    # Prevalência f=0.4; previsor constante c=f => Brier = f*(1-f) = 0.24.
    y = np.array([1, 1, 0, 0, 0])
    f = y.mean()
    p = np.full_like(y, fill_value=f, dtype=float)
    assert brier_score(y, p) == pytest.approx(f * (1.0 - f))
    assert brier_score(y, p) == pytest.approx(0.24)


def test_brier_caso_manual() -> None:
    # ((0.2-0)^2 + (0.8-1)^2 + (0.5-1)^2) / 3 = (0.04 + 0.04 + 0.25)/3
    y = np.array([0, 1, 1])
    p = np.array([0.2, 0.8, 0.5])
    assert brier_score(y, p) == pytest.approx((0.04 + 0.04 + 0.25) / 3.0)


def test_ece_previsoes_perfeitas_eh_zero() -> None:
    y = np.array([0, 1, 0, 1, 1, 0])
    p = y.astype(float)
    assert expected_calibration_error(y, p, n_bins=10) == pytest.approx(0.0)


def test_ece_caso_manual_dois_bins() -> None:
    # n_bins=2: bin0=[0,0.5), bin1=[0.5,1].
    # bin0: p={0.1,0.2,0.4}, y={0,0,1} -> conf=0.2333..., acc=1/3 -> |.|=0.1
    # bin1: p={0.6,0.9}, y={1,1} -> conf=0.75, acc=1.0 -> |.|=0.25
    # ECE = (3/5)*0.1 + (2/5)*0.25 = 0.06 + 0.10 = 0.16
    y = np.array([0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.4, 0.6, 0.9])
    assert expected_calibration_error(y, p, n_bins=2) == pytest.approx(0.16)


def test_ece_robusto_a_bins_vazios() -> None:
    # Todas as previsões na faixa [0.50, 0.55) (bin 10 de 20); demais vazios.
    y = np.array([1, 1, 0, 1])
    p = np.array([0.51, 0.52, 0.53, 0.54])
    val = expected_calibration_error(y, p, n_bins=20)
    # conf = 0.525, acc = 0.75 -> ECE = |0.75 - 0.525| = 0.225
    assert val == pytest.approx(abs(0.75 - p.mean()))


def test_ece_extremos_p_zero_e_um() -> None:
    # p exatamente 0 e 1 devem cair no primeiro/último bin sem erro de índice.
    y = np.array([0, 1])
    p = np.array([0.0, 1.0])
    assert expected_calibration_error(y, p, n_bins=10) == pytest.approx(0.0)


def test_aceita_pandas_series() -> None:
    y = pd.Series([0, 1, 1, 0])
    p = pd.Series([0.1, 0.9, 0.8, 0.2])
    assert brier_score(y, p) >= 0.0
    assert 0.0 <= expected_calibration_error(y, p, n_bins=5) <= 1.0


def test_validacao_tamanhos_diferentes() -> None:
    with pytest.raises(ValueError):
        brier_score([0, 1], [0.5])
    with pytest.raises(ValueError):
        expected_calibration_error([0, 1], [0.5])


def test_validacao_vazio_e_nan() -> None:
    with pytest.raises(ValueError):
        brier_score([], [])
    with pytest.raises(ValueError):
        expected_calibration_error([0, 1], [np.nan, 0.5])


def test_validacao_n_bins_invalido() -> None:
    with pytest.raises(ValueError):
        expected_calibration_error([0, 1], [0.2, 0.8], n_bins=0)


# ---------------------------------------------------------------------------
# TST3.3.a — stub determinístico de revisão LLM
# ---------------------------------------------------------------------------

_RATES = {
    "grey_low": {"e_fp": 0.08, "e_fn": 0.12},
    "grey_mid": {"e_fp": 0.06, "e_fn": 0.10},
    "grey_high": {"e_fp": 0.04, "e_fn": 0.08},
    "near_high": {"e_fp": 0.02, "e_fn": 0.06},
    "low": {"e_fp": 0.10, "e_fn": 0.15},
    "high": {"e_fp": 0.01, "e_fn": 0.03},
}


def _review_frame(n: int, *, band: str, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    target = rng.integers(0, 2, size=n)
    return pd.DataFrame({"TARGET": target, "band": [band] * n})


def test_stub_deterministico_mesma_seed() -> None:
    df = _review_frame(200, band="grey_mid", seed=1)
    out_a = llm_review_stub(df, seed=42, error_rates_by_band=_RATES)
    out_b = llm_review_stub(df, seed=42, error_rates_by_band=_RATES)
    assert out_a.equals(out_b)
    assert set(out_a.unique()) <= {"MATCH", "NONMATCH"}
    assert list(out_a.index) == list(df.index)


def test_stub_seeds_diferentes_podem_diferir() -> None:
    df = _review_frame(200, band="grey_mid", seed=2)
    out_a = llm_review_stub(df, seed=42, error_rates_by_band=_RATES)
    out_b = llm_review_stub(df, seed=7, error_rates_by_band=_RATES)
    assert not out_a.equals(out_b)


def test_stub_taxas_de_erro_respeitadas() -> None:
    # Conjunto grande, banda única, classes balanceadas → taxa empírica ~ config.
    n = 8000
    rng = np.random.default_rng(123)
    target = rng.integers(0, 2, size=n)
    df = pd.DataFrame({"TARGET": target, "band": ["grey_low"] * n})
    decisions = llm_review_stub(df, seed=42, error_rates_by_band=_RATES)
    pos = df["TARGET"] == 1
    neg = ~pos
    # FN: positivos marcados NONMATCH; FP: negativos marcados MATCH.
    emp_fn = (decisions[pos] == "NONMATCH").mean()
    emp_fp = (decisions[neg] == "MATCH").mean()
    assert emp_fn == pytest.approx(_RATES["grey_low"]["e_fn"], abs=0.03)
    assert emp_fp == pytest.approx(_RATES["grey_low"]["e_fp"], abs=0.03)


def test_stub_fallback_banda_desconhecida() -> None:
    df = _review_frame(50, band="banda_inexistente", seed=3)
    # Sem mapa de taxas → usa fallback; não deve levantar e cobre {MATCH,NONMATCH}.
    out = llm_review_stub(df, seed=42, error_rates_by_band=None)
    assert set(out.unique()) <= {"MATCH", "NONMATCH"}
    assert len(out) == 50


def test_stub_vazio_retorna_serie_vazia() -> None:
    df = pd.DataFrame({"TARGET": [], "band": []})
    out = llm_review_stub(df, seed=42, error_rates_by_band=_RATES)
    assert len(out) == 0


def test_stub_exige_colunas() -> None:
    with pytest.raises(KeyError):
        llm_review_stub(pd.DataFrame({"band": ["low"]}), error_rates_by_band=_RATES)
    with pytest.raises(KeyError):
        llm_review_stub(pd.DataFrame({"TARGET": [1]}), error_rates_by_band=_RATES)
