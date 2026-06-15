"""Helpers do notebook didático: métricas de calibração, plots e stub de LLM.

Vive **fora** da biblioteca publicada (D7 / DEC-01). Importável nos testes via
``pythonpath`` configurado no ``pyproject.toml``.

Conteúdo planejado
-------------------
- ``expected_calibration_error`` / ``brier_score`` (DEC-08): métricas quantitativas
  de calibração, com testes próprios (Fase 2.4).
- ``llm_review_stub`` (R-05 / Fase 3.3): simulação determinística da revisão LLM,
  usando as taxas de erro por banda da config. SEM chamadas de rede.
- Funções de plotagem (matplotlib) com títulos/eixos/legendas em PT-BR.

Status: ESQUELETO (Fase 0.3). Implementação nas Fases 2.4 e 3.3.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "expected_calibration_error",
    "brier_score",
    "llm_review_stub",
]


def expected_calibration_error(
    y_true: Sequence[float] | np.ndarray | pd.Series,
    p_pred: Sequence[float] | np.ndarray | pd.Series,
    *,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (binning explícito). Ver DEC-08 / Fase 2.4.

    ECE = soma_b (|B_b| / N) * |acc(B_b) - conf(B_b)|, robusto a bins vazios.
    """
    raise NotImplementedError("Implementado na Fase 2.4.")


def brier_score(
    y_true: Sequence[float] | np.ndarray | pd.Series,
    p_pred: Sequence[float] | np.ndarray | pd.Series,
) -> float:
    """Brier score = média((p - y)^2). Ver DEC-08 / Fase 2.4."""
    raise NotImplementedError("Implementado na Fase 2.4.")


def llm_review_stub(
    df_review: pd.DataFrame,
    *,
    seed: int = 42,
    error_rates_by_band: Mapping[str, Mapping[str, float]] | None = None,
) -> pd.Series:
    """Stub determinístico da revisão clerical/LLM (R-05 / Fase 3.3).

    Decide MATCH/NONMATCH para os pares roteados a ``LLM_REVIEW`` usando as taxas
    de erro por banda da config (e_fp/e_fn). Determinístico por ``seed``; SEM rede.
    """
    raise NotImplementedError("Implementado na Fase 3.3.")
