from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


COL_COMPREC = "COMPREC"
COL_REFREC = "REFREC"
COL_PASSO = "PASSO"
COL_PAR = "PAR"
COL_TARGET = "TARGET"


def resolve_prefixed_column(df: pd.DataFrame, canonical: str) -> str | None:
    """Resolve OpenRecLink-style columns like `COMPREC,C,12,0` to `COMPREC`.

    Strategy:
    - If `canonical` exists, return it.
    - Else, find a single column that starts with `canonical + ','`.
    - If ambiguous or missing, return None.
    """

    if canonical in df.columns:
        return canonical

    prefix = f"{canonical},"
    matches = [c for c in df.columns if isinstance(c, str) and c.startswith(prefix)]
    if len(matches) == 1:
        return matches[0]
    return None


def ensure_canonical_column(df: pd.DataFrame, canonical: str) -> pd.DataFrame:
    """Ensure `df[canonical]` exists by aliasing a prefixed source if needed."""

    if canonical in df.columns:
        return df
    src = resolve_prefixed_column(df, canonical)
    if src is None:
        return df
    df[canonical] = df[src]
    return df


def clean_col_name(name: str) -> str:
    s = str(name).strip().replace(" ", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s


def add_cleaned_column_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Adds cleaned aliases for every column (never overwrites)."""

    for col in list(df.columns):
        cleaned = clean_col_name(col)
        if cleaned != col and cleaned not in df.columns:
            df[cleaned] = df[col]
    return df


def _num(df: pd.DataFrame, col: str, *, fillna: float | None = None) -> pd.Series:
    if col not in df.columns:
        s = pd.Series(np.nan, index=df.index, dtype="float64")
    else:
        s_raw = pd.to_numeric(df[col], errors="coerce")
        s = pd.Series(s_raw, index=df.index, dtype="float64")
    if fillna is not None:
        s = s.fillna(float(fillna))
    return s


def _digits_only(x: object) -> str | None:
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    d = "".join(ch for ch in s if ch.isdigit())
    return d or None


def parse_date_yyyymmdd(series: pd.Series) -> pd.Series:
    norm = series.map(_digits_only)
    norm = norm.map(lambda s: s.zfill(8)[:8] if s is not None else None)
    norm = norm.where(norm != "00000000", other=None)
    return pd.to_datetime(norm, format="%Y%m%d", errors="coerce")


def parse_date_ddmmyyyy(series: pd.Series) -> pd.Series:
    norm = series.map(_digits_only)
    norm = norm.map(lambda s: s.zfill(8)[:8] if s is not None else None)
    norm = norm.where(norm != "00000000", other=None)
    return pd.to_datetime(norm, format="%d%m%Y", errors="coerce")


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    if COL_PAR in df.columns:
        par_num = pd.to_numeric(df[COL_PAR], errors="coerce")
        par = pd.Series(par_num, index=df.index, dtype="float64")
    else:
        par = pd.Series(np.nan, index=df.index, dtype="float64")

    df[COL_TARGET] = par.isin([1, 2]).astype("int8")
    return df


def _all_zero(df: pd.DataFrame, cols: Iterable[str]) -> pd.Series:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.Series(False, index=df.index, dtype="bool")
    block = df.loc[:, cols].apply(pd.to_numeric, errors="coerce")
    return block.fillna(0.0).eq(0.0).all(axis=1)


@dataclass(frozen=True)
class LoadConfig:
    macd_enabled: bool = True


def feature_engineer(df: pd.DataFrame, *, cfg: LoadConfig) -> pd.DataFrame:
    out = df.copy()
    out = add_cleaned_column_aliases(out)

    # Normalize OpenRecLink-prefixed metadata/date columns into canonical names
    for canonical in [
        COL_COMPREC,
        COL_REFREC,
        "R_DTNASC",
        "C_DTNASC",
        "R_DTOBITO",
        "C_DTDIAG",
    ]:
        out = ensure_canonical_column(out, canonical)

    # Ensure numeric-ish key columns exist as numeric
    for col in [
        "nota final",
        "NOME qtd frag iguais",
        "NOME prim frag igual",
        "NOME ult frag igual",
        "NOME prim ult frag igual",
        "NOMEMAE qtd frag iguais",
        "NOMEMAE prim frag igual",
        "NOMEMAE ult frag igual",
        "NOMEMAE prim ult frag igual",
        "DTNASC ano igual",
        "DTNASC mes igual",
        "DTNASC dia igual",
        "DTNASC dt iguais",
        "ENDERECO via igual",
        "ENDERECO via prox",
        "ENDERECO numero igual",
        "ENDERECO compl prox",
        "ENDERECO texto prox",
        "ENDERECO tokens jacc",
        "CODMUNRES local igual",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = add_target(out)

    # Parse event dates (used by guardrails)
    if "R_DTOBITO" in out.columns:
        out["R_DTOBITO_dt"] = parse_date_ddmmyyyy(
            pd.Series(out["R_DTOBITO"], index=out.index)
        )
    if "C_DTDIAG" in out.columns:
        out["C_DTDIAG_dt"] = parse_date_yyyymmdd(
            pd.Series(out["C_DTDIAG"], index=out.index)
        )

    # Parse DOB raw dates
    r_dtnasc_dt = (
        parse_date_yyyymmdd(pd.Series(out["R_DTNASC"], index=out.index))
        if "R_DTNASC" in out.columns
        else pd.Series(pd.NaT, index=out.index)
    )
    c_dtnasc_dt = (
        parse_date_yyyymmdd(pd.Series(out["C_DTNASC"], index=out.index))
        if "C_DTNASC" in out.columns
        else pd.Series(pd.NaT, index=out.index)
    )
    out["R_DTNASC_dt"] = r_dtnasc_dt
    out["C_DTNASC_dt"] = c_dtnasc_dt
    out["diff_ano"] = (
        (r_dtnasc_dt.dt.year - c_dtnasc_dt.dt.year).abs().astype("float64")
    )

    # DTNASC subscores all zero (captures missingness / disagreement in OpenRecLink comparisons)
    dtnasc_cols = [
        "DTNASC dt iguais",
        "DTNASC dt ap 1digi",
        "DTNASC dt inv dia",
        "DTNASC dt inv mes",
        "DTNASC dt inv ano",
    ]
    out["dtnasc_all_zero"] = _all_zero(out, dtnasc_cols)

    # Aggregate NB03-style totals (keeps within [0,1] for typical inputs)
    nome_qtd = _num(out, "NOME qtd frag iguais", fillna=0.0)
    nome_prim = _num(out, "NOME prim frag igual", fillna=0.0)
    nome_ult = _num(out, "NOME ult frag igual", fillna=0.0)
    out["nome_score_total"] = (
        0.5 * nome_qtd + 0.25 * nome_prim + 0.25 * nome_ult
    ).clip(0.0, 1.0)

    mae_qtd = _num(out, "NOMEMAE qtd frag iguais", fillna=0.0)
    mae_prim = _num(out, "NOMEMAE prim frag igual", fillna=0.0)
    mae_ult = _num(out, "NOMEMAE ult frag igual", fillna=0.0)
    out["mae_score_total"] = (0.5 * mae_qtd + 0.25 * mae_prim + 0.25 * mae_ult).clip(
        0.0, 1.0
    )
    out["mae_missing"] = _all_zero(
        out,
        [
            "NOMEMAE qtd frag iguais",
            "NOMEMAE prim frag igual",
            "NOMEMAE ult frag igual",
            "NOMEMAE prim ult frag igual",
        ],
    ).astype("int8")

    dtnasc_block = pd.concat([_num(out, c, fillna=0.0) for c in dtnasc_cols], axis=1)
    dtnasc_mean = dtnasc_block.mean(axis=1)
    out["dtnasc_score_total"] = pd.Series(
        dtnasc_mean, index=out.index, dtype="float64"
    ).clip(0.0, 1.0)

    end_cols = [
        "ENDERECO via igual",
        "ENDERECO via prox",
        "ENDERECO numero igual",
        "ENDERECO compl prox",
        "ENDERECO texto prox",
        "ENDERECO tokens jacc",
    ]
    endereco_block = pd.concat([_num(out, c, fillna=0.0) for c in end_cols], axis=1)
    endereco_mean = endereco_block.mean(axis=1)
    out["endereco_score_total"] = pd.Series(
        endereco_mean, index=out.index, dtype="float64"
    ).clip(0.0, 1.0)
    out["endereco_zero"] = _all_zero(out, end_cols).astype("int8")

    out["municipio_score"] = _num(out, "CODMUNRES local igual", fillna=0.0).clip(
        0.0, 1.0
    )

    # MACD feature set (continuous DOB signal from raw dates)
    if cfg.macd_enabled:
        days_diff = (c_dtnasc_dt - r_dtnasc_dt).dt.days.astype("float64")
        days_abs = days_diff.abs()
        years_abs = (days_abs / 365.25).astype("float64")
        years_abs_capped = years_abs.clip(upper=20.0)

        out["macd_nasc_diff_capped"] = years_abs_capped
        out["macd_nasc_year_match"] = (
            (r_dtnasc_dt.dt.year == c_dtnasc_dt.dt.year).fillna(False).astype("int8")
        )
        out["macd_nasc_month_match"] = (
            (r_dtnasc_dt.dt.month == c_dtnasc_dt.dt.month).fillna(False).astype("int8")
        )
        out["macd_nasc_day_match"] = (
            (r_dtnasc_dt.dt.day == c_dtnasc_dt.dt.day).fillna(False).astype("int8")
        )
        out["macd_nasc_partial_overlap"] = (
            out["macd_nasc_year_match"].astype("float64")
            + out["macd_nasc_month_match"].astype("float64")
            + out["macd_nasc_day_match"].astype("float64")
        ) / 3.0
        out["macd_nasc_close"] = (days_abs <= 30).fillna(False).astype("int8")
        out["macd_nasc_very_close"] = (days_abs <= 7).fillna(False).astype("int8")

        nome_perf = (nome_qtd >= 0.95) & (nome_prim >= 1.0) & (nome_ult >= 1.0)
        far = years_abs.notna() & (years_abs >= 5.0)
        out["macd_nome_perf_x_date_far"] = (nome_perf & far).astype("int8")
        out["macd_nome_perf_x_year_diff"] = (
            nome_perf.astype("float64") * years_abs_capped.fillna(0.0)
        ).astype("float64")

    # Basic weighted "rule score" (useful for anchor rules / explainability)
    rule_cols = [
        "NOME prim frag igual",
        "DTNASC dt iguais",
        "CODMUNRES local igual",
    ]
    present = [c for c in rule_cols if c in out.columns]
    if present:
        fired = pd.concat(
            [_num(out, c, fillna=0.0).ge(1.0).astype("float64") for c in present],
            axis=1,
        )
        out["score_regras"] = fired.sum(axis=1)
    else:
        out["score_regras"] = 0.0

    # Fail fast on required metadata
    for req in [COL_COMPREC, COL_REFREC, COL_PASSO, COL_PAR]:
        if req not in out.columns:
            raise KeyError(f"Missing required column: {req}")

    # Preferred order for downstream
    first = [
        COL_COMPREC,
        COL_REFREC,
        COL_PASSO,
        COL_PAR,
        COL_TARGET,
        "nota final" if "nota final" in out.columns else "nota_final",
        "diff_ano",
        "dtnasc_all_zero",
        "nome_score_total",
        "mae_score_total",
        "dtnasc_score_total",
        "endereco_score_total",
        "municipio_score",
    ]
    first = [c for c in first if c in out.columns]
    rest = [c for c in out.columns if c not in first]
    return out.loc[:, first + rest]


def load_comparador_csv(
    path: str | Path = Path("data") / "COMPARADORSEMIDENT.csv",
    *,
    cfg: LoadConfig = LoadConfig(),
) -> pd.DataFrame:
    """Load and feature-engineer the thesis CSV (sep=';', decimal=',')."""

    p = Path(path)
    try:
        df = pd.read_csv(
            p,
            sep=";",
            decimal=",",
            encoding="utf-8",
            keep_default_na=True,
            na_values=["", " ", "NA", "N/A", "null", "None"],
            low_memory=False,
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            p,
            sep=";",
            decimal=",",
            encoding="latin-1",
            keep_default_na=True,
            na_values=["", " ", "NA", "N/A", "null", "None"],
            low_memory=False,
        )

    return feature_engineer(df, cfg=cfg)
