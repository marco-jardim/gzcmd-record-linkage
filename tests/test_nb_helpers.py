"""Testes das métricas de calibração de ``nb_helpers`` (DEC-08 / TST2.4.c).

Casos fechados (valores conhecidos analiticamente) garantem que ECE e Brier
estão corretos, além de robustez a bins vazios e validação de entrada.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from nb_helpers import brier_score, expected_calibration_error


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
