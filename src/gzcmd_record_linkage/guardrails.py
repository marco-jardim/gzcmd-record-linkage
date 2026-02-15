from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from typing import Iterable

import numpy as np
import pandas as pd


class GuardrailDecision(str, Enum):
    ALWAYS_MATCH = "ALWAYS_MATCH"
    ALWAYS_NONMATCH = "ALWAYS_NONMATCH"
    FORCE_REVIEW = "FORCE_REVIEW"


@dataclass(frozen=True)
class GuardrailOutput:
    guardrail: pd.Series
    reason: pd.Series


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    raw = df[col]
    if not isinstance(raw, pd.Series):
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.Series(
        pd.to_numeric(raw, errors="coerce"), index=df.index, dtype="float64"
    )


def _all_zero(df: pd.DataFrame, cols: Iterable[str]) -> pd.Series:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.Series(False, index=df.index, dtype="bool")
    block = df.loc[:, cols].apply(pd.to_numeric, errors="coerce")
    return block.fillna(0.0).eq(0.0).all(axis=1)


def _bool_indicator(df: pd.DataFrame, col: str) -> pd.Series:
    """Coerce an indicator-like column into a boolean Series."""

    if col not in df.columns:
        return pd.Series(False, index=df.index, dtype="bool")
    raw = df[col]
    if not isinstance(raw, pd.Series):
        return pd.Series(False, index=df.index, dtype="bool")
    s = pd.Series(pd.to_numeric(raw, errors="coerce"), index=df.index, dtype="float64")
    return s.fillna(0.0).gt(0.0)


def apply_guardrails(
    df: pd.DataFrame,
    *,
    temporal_days: int = 180,
    nota_always_match: float = 10.0,
    nota_always_nonmatch: float = 3.0,
    homonimia_min_nota: float = 7.0,
    homonimia_year_gap: float = 5.0,
) -> GuardrailOutput:
    """Compute guardrails used by the policy engine.

    Returns two aligned Series:
    - guardrail: {ALWAYS_MATCH, ALWAYS_NONMATCH, FORCE_REVIEW, <NA>}
    - reason: pipe-joined short reasons

    Expected columns (best effort; rules degrade gracefully):
    - R_DTOBITO_dt, C_DTDIAG_dt (datetime64)
    - nota final (or nota_final alias)
    - DTNASC comparison subscores: DTNASC ano igual, DTNASC mes igual, DTNASC dia igual, DTNASC dt iguais
    - diff_ano (from raw DOB years)
    - ENDERECO subscores
    - NOME subscores
    """

    out_guardrail = pd.Series(pd.NA, index=df.index, dtype="string")
    out_reason = pd.Series("", index=df.index, dtype="string")

    def _set(mask: pd.Series, action: str, reason: str) -> None:
        nonlocal out_guardrail, out_reason
        mask2 = mask.fillna(False) & out_guardrail.isna()
        out_guardrail = out_guardrail.mask(mask2, action)
        out_reason = out_reason.mask(mask2, reason)

    nota = _num(df, "nota final").fillna(_num(df, "nota_final"))

    # 1) temporal_filter: death before diagnosis minus buffer
    dt_obito = df.get("R_DTOBITO_dt")
    dt_diag = df.get("C_DTDIAG_dt")
    if (dt_obito is not None) and (dt_diag is not None):
        ob = pd.to_datetime(dt_obito, errors="coerce")
        dg = pd.to_datetime(dt_diag, errors="coerce")
        bad = ob.notna() & dg.notna() & (ob < (dg - timedelta(days=int(temporal_days))))
        _set(bad, GuardrailDecision.ALWAYS_NONMATCH.value, "temporal_filter")

    # 2) always_nonmatch
    _set(
        nota.notna() & (nota < nota_always_nonmatch),
        GuardrailDecision.ALWAYS_NONMATCH.value,
        "nota_final_low",
    )

    # 3) homonimia_risk: DTNASC subscores all zero + big raw year gap + address empty
    if "dtnasc_all_zero" in df.columns:
        dtnasc_all_zero = _bool_indicator(df, "dtnasc_all_zero")
    else:
        dtnasc_all_zero = _all_zero(
            df,
            [
                "DTNASC ano igual",
                "DTNASC mes igual",
                "DTNASC dia igual",
                "DTNASC dt iguais",
            ],
        )
    diff_ano = _num(df, "diff_ano")
    if "endereco_zero" in df.columns:
        endereco_zero = _bool_indicator(df, "endereco_zero")
    else:
        endereco_zero = _all_zero(
            df,
            [
                "ENDERECO via igual",
                "ENDERECO via prox",
                "ENDERECO numero igual",
                "ENDERECO compl prox",
                "ENDERECO texto prox",
                "ENDERECO tokens jacc",
            ],
        )
    homonimia = (
        nota.notna()
        & (nota >= homonimia_min_nota)
        & dtnasc_all_zero
        & diff_ano.notna()
        & (diff_ano > homonimia_year_gap)
        & endereco_zero
    )
    _set(homonimia, GuardrailDecision.FORCE_REVIEW.value, "homonimia_risk")

    # 4) always_match: only if strong criteria are present
    nome_qtd = _num(df, "NOME qtd frag iguais")
    nome_prim = _num(df, "NOME prim frag igual")
    nome_ult = _num(df, "NOME ult frag igual")
    dtnasc_dt = _num(df, "DTNASC dt iguais")
    mun_ok = _num(df, "CODMUNRES local igual")

    strong_criteria = (
        nome_qtd.notna()
        & (nome_qtd >= 0.95)
        & (nome_prim >= 1.0)
        & (nome_ult >= 1.0)
        & (dtnasc_dt >= 1.0)
        & (mun_ok >= 1.0)
    )
    _set(
        nota.notna() & (nota >= nota_always_match) & strong_criteria,
        GuardrailDecision.ALWAYS_MATCH.value,
        "nota_final_high",
    )

    # Empty reason for rows without guardrail
    out_reason = out_reason.mask(out_guardrail.isna(), pd.NA)
    return GuardrailOutput(guardrail=out_guardrail, reason=out_reason)
