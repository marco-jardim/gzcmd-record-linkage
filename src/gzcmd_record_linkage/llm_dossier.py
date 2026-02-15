"""Build a JSON dossier per candidate pair for LLM clerical review.

The dossier contains ONLY linkage sub-scores and model outputs — never PII.
It is the sole input the LLM sees, enforcing a strict information boundary.

Usage::

    from gzcmd_record_linkage.llm_dossier import build_dossier, build_dossiers

    dossier = build_dossier(row, pair_id="abc123")
    dossiers = build_dossiers(df)  # DataFrame with LLM_REVIEW rows
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

__all__ = [
    "Dossier",
    "DossierFeatures",
    "DossierModelOutputs",
    "build_dossier",
    "build_dossiers",
    "make_pair_id",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "gzcmd_v3.dossier.1.0"

# Raw sub-score columns from the comparator (canonical names after loader)
_NAME_COLS = [
    "NOME qtd frag iguais",
    "NOME prim frag igual",
    "NOME ult frag igual",
    "NOME prim ult frag igual",
]

_MOTHER_COLS = [
    "NOMEMAE qtd frag iguais",
    "NOMEMAE prim frag igual",
    "NOMEMAE ult frag igual",
    "NOMEMAE prim ult frag igual",
]

_DOB_COLS = [
    "DTNASC ano igual",
    "DTNASC mes igual",
    "DTNASC dia igual",
    "DTNASC dt iguais",
]

_ADDRESS_COLS = [
    "ENDERECO via igual",
    "ENDERECO via prox",
    "ENDERECO numero igual",
    "ENDERECO compl prox",
    "ENDERECO texto prox",
    "ENDERECO tokens jacc",
]

_MUNICIPALITY_COL = "CODMUNRES local igual"

# Engineered composite scores
_COMPOSITE_COLS = [
    "nome_score_total",
    "mae_score_total",
    "dtnasc_score_total",
    "endereco_score_total",
    "municipio_score",
]

# Boolean flags
_FLAG_COLS = [
    "mae_missing",
    "dtnasc_all_zero",
    "endereco_zero",
]

# MACD engineered features (optional)
_MACD_COLS = [
    "macd_nasc_diff_capped",
    "macd_nasc_year_match",
    "macd_nasc_month_match",
    "macd_nasc_day_match",
    "macd_nasc_partial_overlap",
    "macd_nasc_close",
    "macd_nasc_very_close",
    "macd_nome_perf_x_date_far",
    "macd_nome_perf_x_year_diff",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DossierFeatures:
    """Linkage sub-scores grouped by domain."""

    nota_final: float
    name: dict[str, float]
    mother: dict[str, float]
    dob: dict[str, float]
    address: dict[str, float]
    municipality: float
    composites: dict[str, float]
    flags: dict[str, bool | int]
    macd: dict[str, float] | None = None


@dataclass(frozen=True)
class DossierModelOutputs:
    """Pipeline model outputs for this pair."""

    p_cal: float
    band: str
    base_choice: str
    evr: float
    guardrail: str | None = None
    guardrail_reason: str | None = None


@dataclass(frozen=True)
class Dossier:
    """Complete dossier for one candidate pair."""

    schema_version: str
    pair_id: str
    features: DossierFeatures
    model_outputs: DossierModelOutputs

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (JSON-safe)."""
        d = asdict(self)
        # Clean NaN/None values for JSON safety
        return _sanitize(d)

    def to_json(self, **kwargs: Any) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, **kwargs)


# ---------------------------------------------------------------------------
# Pair ID generation
# ---------------------------------------------------------------------------


def make_pair_id(
    comprec: Any,
    refrec: Any,
    passo: Any,
) -> str:
    """Deterministic, non-reversible pair identifier.

    Uses SHA-256 of ``COMPREC|REFREC|PASSO`` — no PII leaks.
    """
    raw = f"{comprec}|{refrec}|{passo}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Dossier builder (single row)
# ---------------------------------------------------------------------------


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Coerce to float, replacing NaN/None with *default*."""
    if value is None:
        return default
    try:
        f = float(value)
        return default if math.isnan(f) else round(f, 4)
    except (TypeError, ValueError):
        return default


def _safe_bool(value: Any) -> bool:
    """Coerce to bool."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return bool(value)


def _resolve_col(row: pd.Series, canonical: str) -> Any:
    """Try canonical name, then common variants with type suffixes stripped."""
    if canonical in row.index:
        return row[canonical]
    # loader may have left the original name with type suffix: "col,Int32"
    for col in row.index:
        if col.split(",")[0].strip() == canonical:
            return row[col]
    return None


def _extract_group(row: pd.Series, cols: list[str]) -> dict[str, float]:
    """Extract a dict of {column: float} for a group of sub-scores."""
    result: dict[str, float] = {}
    for col in cols:
        val = _resolve_col(row, col)
        if val is not None:
            result[col] = _safe_float(val)
    return result


def build_dossier(
    row: pd.Series,
    *,
    pair_id: str | None = None,
    include_macd: bool = True,
) -> Dossier:
    """Build a :class:`Dossier` from a single DataFrame row.

    Parameters
    ----------
    row:
        A Series representing one candidate pair (one row of the enriched
        DataFrame after feature engineering + triage).
    pair_id:
        Pre-computed pair ID.  If *None*, computed from COMPREC/REFREC/PASSO.
    include_macd:
        Include MACD engineered features when present.
    """
    if pair_id is None:
        pair_id = make_pair_id(
            _resolve_col(row, "COMPREC"),
            _resolve_col(row, "REFREC"),
            _resolve_col(row, "PASSO"),
        )

    nota = _safe_float(
        _resolve_col(row, "nota final") or _resolve_col(row, "nota_final")
    )

    # Sub-scores by domain
    name_scores = _extract_group(row, _NAME_COLS)
    mother_scores = _extract_group(row, _MOTHER_COLS)
    dob_scores = _extract_group(row, _DOB_COLS)
    address_scores = _extract_group(row, _ADDRESS_COLS)
    municipality = _safe_float(_resolve_col(row, _MUNICIPALITY_COL))

    # Composites
    composites = _extract_group(row, _COMPOSITE_COLS)

    # Flags
    flags: dict[str, bool | int] = {}
    for col in _FLAG_COLS:
        val = _resolve_col(row, col)
        if val is not None:
            flags[col] = _safe_bool(val)

    # MACD (optional)
    macd: dict[str, float] | None = None
    if include_macd:
        macd_data = _extract_group(row, _MACD_COLS)
        if macd_data:
            macd = macd_data

    features = DossierFeatures(
        nota_final=nota,
        name=name_scores,
        mother=mother_scores,
        dob=dob_scores,
        address=address_scores,
        municipality=municipality,
        composites=composites,
        flags=flags,
        macd=macd,
    )

    # Model outputs
    p_cal = _safe_float(_resolve_col(row, "p_cal"), default=0.5)
    band = str(_resolve_col(row, "band") or "unknown")
    base_choice = str(_resolve_col(row, "base_choice") or "NONMATCH")
    evr = _safe_float(_resolve_col(row, "evr"))

    guardrail_val = _resolve_col(row, "guardrail")
    guardrail = (
        str(guardrail_val)
        if guardrail_val is not None
        and not (isinstance(guardrail_val, float) and math.isnan(guardrail_val))
        else None
    )
    # pandas NA check
    if guardrail and guardrail.lower() in ("nan", "<na>", "none"):
        guardrail = None

    guardrail_reason_val = _resolve_col(row, "guardrail_reason")
    guardrail_reason = (
        str(guardrail_reason_val)
        if guardrail_reason_val is not None
        and not (
            isinstance(guardrail_reason_val, float) and math.isnan(guardrail_reason_val)
        )
        else None
    )
    if guardrail_reason and guardrail_reason.lower() in ("nan", "<na>", "none"):
        guardrail_reason = None

    model_outputs = DossierModelOutputs(
        p_cal=p_cal,
        band=band,
        base_choice=base_choice,
        evr=evr,
        guardrail=guardrail,
        guardrail_reason=guardrail_reason,
    )

    return Dossier(
        schema_version=SCHEMA_VERSION,
        pair_id=pair_id,
        features=features,
        model_outputs=model_outputs,
    )


# ---------------------------------------------------------------------------
# Batch builder
# ---------------------------------------------------------------------------


def build_dossiers(
    df: pd.DataFrame,
    *,
    include_macd: bool = True,
    only_llm_review: bool = True,
) -> list[Dossier]:
    """Build dossiers for multiple rows.

    Parameters
    ----------
    df:
        DataFrame after triage (must have ``action`` column).
    only_llm_review:
        If *True* (default), only builds dossiers for rows where
        ``action == "LLM_REVIEW"``.
    include_macd:
        Include MACD features when present.

    Returns
    -------
    list[Dossier]
        One dossier per qualifying row, in original index order.
    """
    if only_llm_review and "action" in df.columns:
        mask = df["action"] == "LLM_REVIEW"
        subset = df.loc[mask]
    else:
        subset = df

    dossiers: list[Dossier] = []
    for idx, row in subset.iterrows():
        dossiers.append(build_dossier(row, include_macd=include_macd))
    return dossiers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sanitize(obj: Any) -> Any:
    """Recursively replace NaN/None in nested dict/list for JSON safety."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return 0.0
    return obj
