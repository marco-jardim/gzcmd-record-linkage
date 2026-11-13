"""Thesis context queues C and CM: per-death context features and their queues.

Port of ``scripts/contexto_obito.py`` of the thesis research repository. Both
queues are *exploratory* in the thesis (``estudo4_gzcmd.tex``, line 231; glossary
entry in ``glossario.tex``, line 20) and are **not** a recommended default.

* ``O3_contexto`` (queue C) is the nested cross-fitted XGBoost + isotonic model
  (``oof.nested_oof_probabilities``) over the 30 comparator attributes plus the
  10 context features;
* ``O3_contexto_MACD`` (queue CM) adds the 9 MACD indicators (missing -> -1);
* ``F4_contexto`` fuses O1 and ``O3_contexto`` by reciprocal rank fusion, k=60
  (``THESIS_CONTEXT_FUSIONS``).

Context features. For each key (REFREC, then COMPREC) the candidates sharing
the key form a group; using only the final score (``nota final``, zero-imputed)
over the *full* candidate universe (before any cutoff, no labels):
``n_candidatos`` (rows in the group, the row itself included), ``posicao``
(descending rank, method "min", base 1), ``delta_melhor`` (score minus the group
maximum), ``delta_segundo`` (score minus the second-highest score counted with
multiplicity; singleton -> 0) and ``eh_melhor`` (1 for every row tied at the
maximum). Column names: ``ctx_refrec_*`` and ``ctx_comprec_*``.

Leakage caveat, stated in the thesis: the folds are grouped by REFREC, but a
COMPREC can appear under several REFREC partitions, so COMPREC context (and the
labels of the same COMPREC in other folds) can cross the partition. Treat the
results as an exploratory ceiling, not as validated performance.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from gzcmd_record_linkage.fusion import FusionSpec, fuse
from gzcmd_record_linkage.loader import resolve_prefixed_column
from gzcmd_record_linkage.oof import N_SPLITS, SEED, nested_oof_probabilities
from gzcmd_record_linkage.ranking import Order, order_by_score

CONTEXT_SUFFIXES: tuple[str, ...] = (
    "n_candidatos",
    "posicao",
    "delta_melhor",
    "delta_segundo",
    "eh_melhor",
)

CONTEXT_FEATURES: tuple[str, ...] = tuple(
    f"ctx_{group}_{suffix}"
    for group in ("refrec", "comprec")
    for suffix in CONTEXT_SUFFIXES
)
"""The 10 thesis context columns, in thesis order (keys REFREC then COMPREC)."""

FEATURE_PREFIXES: tuple[str, ...] = (
    "NOME ",
    "NOMEMAE ",
    "DTNASC ",
    "CODMUNRES ",
    "ENDERECO ",
)
"""Column prefixes of the thesis comparator allowlist (30 attributes with
``nota final``)."""

MACD_COLUMNS: tuple[str, ...] = (
    "macd_nasc_close",
    "macd_nasc_day_match",
    "macd_nasc_diff_capped",
    "macd_nasc_month_match",
    "macd_nasc_partial_overlap",
    "macd_nasc_very_close",
    "macd_nasc_year_match",
    "macd_nome_perf_x_date_far",
    "macd_nome_perf_x_year_diff",
)
"""The 9 MACD indicators produced by ``loader.feature_engineer``, in the thesis
column order: alphabetical, as in ``features.macd.names`` of the thesis ablation
metadata (``sensibilidade_o3_macd.prepare`` keeps that order for ``O3_MACD``)."""

CONTEXT_VARIANTS: tuple[str, ...] = ("O3_contexto", "O3_contexto_MACD")

THESIS_CONTEXT_FUSIONS: Mapping[str, FusionSpec] = MappingProxyType(
    {"F4_contexto": FusionSpec("F4_contexto", "rrf", ("O1", "O3_contexto"), k=60)}
)
"""The thesis context fusion, read-only: reciprocal rank fusion (k=60) of O1 and
O3_contexto, in that arm order (it is the tie order). Exploratory, post hoc."""

_SCORE_ALIASES = ("nota final", "nota_final")


def _numeric(series: pd.Series) -> pd.Series:
    """Thesis conversion: decimal comma, coerce errors to missing, then zero."""
    values = pd.to_numeric(
        series.astype(str).str.replace(",", ".", regex=False), errors="coerce"
    )
    return values.astype("float64").fillna(0.0)


def _resolve_score_column(df: pd.DataFrame, score_col: str) -> str:
    if score_col in df.columns:
        return score_col
    if score_col == _SCORE_ALIASES[0] and _SCORE_ALIASES[1] in df.columns:
        return _SCORE_ALIASES[1]
    raise ValueError(f"score column {score_col!r} not found")


def _group_keys(df: pd.DataFrame, canonical: str) -> pd.Series:
    column = resolve_prefixed_column(df, canonical)
    if column is None:
        raise ValueError(f"grouping key {canonical} not found")
    key = df[column].astype("string").str.strip()
    if key.isna().any() or key.eq("").any():
        raise ValueError(f"grouping key {canonical} has missing or blank values")
    return key.astype(object)


def context_features(
    df: pd.DataFrame,
    *,
    score_col: str = "nota final",
    keys: Sequence[str] = ("REFREC", "COMPREC"),
) -> pd.DataFrame:
    """Compute the thesis context features of every candidate row.

    ``df`` is the full candidate universe (one row per candidate pair). Only
    ``score_col`` (default ``nota final``, falling back to ``nota_final``) and
    the grouping ``keys`` are read, never labels or other scores. The score is
    converted as in the thesis (decimal comma accepted, missing -> 0) and must
    end up finite; keys are stripped and must not be missing or blank
    (``ValueError`` otherwise).

    Returns a float64 DataFrame aligned with ``df.index`` whose columns, for
    each key in ``keys``, are ``ctx_<key>_n_candidatos``, ``..._posicao``,
    ``..._delta_melhor``, ``..._delta_segundo`` and ``..._eh_melhor`` (with the
    default keys: ``CONTEXT_FEATURES``). Definitions in the module docstring.
    """
    if not keys:
        raise ValueError("at least one grouping key is required")
    score = _numeric(df[_resolve_score_column(df, score_col)]).reset_index(drop=True)
    if not np.isfinite(score.to_numpy()).all():
        raise ValueError("score is not finite")
    columns: dict[str, pd.Series] = {}
    for name in keys:
        key = _group_keys(df, name).reset_index(drop=True)
        grouped = score.groupby(key, sort=False)
        best = grouped.transform("max")
        # nlargest keeps multiplicity: two tied best rows share the second note.
        second = grouped.agg(lambda s: s.nlargest(2).iloc[-1])
        prefix = f"ctx_{name.lower()}_"
        columns[prefix + "n_candidatos"] = grouped.transform("size")
        columns[prefix + "posicao"] = grouped.rank(ascending=False, method="min")
        columns[prefix + "delta_melhor"] = score - best
        columns[prefix + "delta_segundo"] = score - key.map(second)
        columns[prefix + "eh_melhor"] = score.eq(best).astype(int)
    result = pd.DataFrame(columns).astype("float64")
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("context is not finite")
    result.index = df.index
    return result


def base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Thesis comparator allowlist: columns with a ``FEATURE_PREFIXES`` prefix
    plus ``nota final``/``nota_final``, in ``df`` order, converted as in the
    thesis (decimal comma, missing -> 0), float64. Never identifiers or labels."""
    if not df.columns.is_unique:
        raise ValueError("ambiguous (duplicated) column names")
    names = [
        c
        for c in df.columns
        if str(c).startswith(FEATURE_PREFIXES) or c in _SCORE_ALIASES
    ]
    if not names:
        raise ValueError("no comparator attribute columns found")
    return df[names].apply(_numeric).astype("float64")


def macd_features(
    df: pd.DataFrame, *, columns: Sequence[str] | None = None
) -> pd.DataFrame:
    """The 9 MACD indicators, numeric, missing -> -1 (thesis imputation).

    ``df`` must already hold them (``loader.feature_engineer`` with MACD
    enabled). With ``columns=None`` the output follows the thesis order,
    ``MACD_COLUMNS`` (alphabetical), whatever the order in ``df``; the ``macd_*``
    columns of ``df`` must be exactly that set. XGBoost's column subsampling
    depends on column order, so change ``columns`` only on purpose.
    """
    if columns is None:
        found = {c for c in df.columns if str(c).startswith("macd_")}
        if found != set(MACD_COLUMNS):
            raise ValueError("expected exactly the 9 MACD indicator columns")
        columns = MACD_COLUMNS
    out = df[list(columns)].apply(pd.to_numeric, errors="raise").astype("float64")
    return out.fillna(-1.0)


def context_design_matrices(
    df: pd.DataFrame,
    *,
    variants: Sequence[str] = CONTEXT_VARIANTS,
    score_col: str = "nota final",
    macd_columns: Sequence[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Model inputs of the thesis variants, one DataFrame per name in ``variants``.

    ``O3_contexto``: ``base_features`` (30 attributes, ``df`` order) then the 10
    context features. ``O3_contexto_MACD``: ``base_features``, the 9
    ``macd_features`` (thesis order) and the 10 context features. Exactly the
    matrices ``context_queues`` feeds to the OOF routine.
    """
    unknown = [v for v in variants if v not in CONTEXT_VARIANTS]
    if unknown or not variants:
        raise ValueError(f"variants must be a non-empty subset of {CONTEXT_VARIANTS}")
    base = base_features(df)
    context = context_features(df, score_col=score_col)
    matrices: dict[str, pd.DataFrame] = {}
    for variant in variants:
        if variant == "O3_contexto_MACD":
            parts = [base, macd_features(df, columns=macd_columns), context]
        else:
            parts = [base, context]
        matrices[variant] = pd.concat(parts, axis=1)
    return matrices


def thesis_context_features(
    x: pd.DataFrame, groups: Mapping[str, pd.Series]
) -> pd.DataFrame:
    """Thesis signature ``context_features(x, groups)`` of ``contexto_obito.py``.

    ``x`` is the numeric attribute matrix with exactly one ``nota final`` /
    ``nota_final`` column (finite, already zero-imputed); ``groups`` maps
    ``"refrec"`` and ``"comprec"`` to the (stripped) keys, indexed like ``x``.
    Same output as ``context_features`` on the equivalent frame; it raises
    ``ValueError`` on a missing or ambiguous score column, a non-finite score and
    missing, blank or misaligned keys, as the thesis does.
    """
    score_columns = [name for name in _SCORE_ALIASES if name in x]
    if len(score_columns) != 1:
        raise ValueError("exactly one final-score column is required")
    if not np.isfinite(x[score_columns[0]].astype("float64").to_numpy()).all():
        raise ValueError("score is not finite")
    frame = pd.DataFrame(
        {score_columns[0]: x[score_columns[0]].to_numpy()}, index=x.index
    )
    for name in ("refrec", "comprec"):
        key = groups[name]
        if not key.index.equals(x.index):
            raise ValueError("context keys are missing or misaligned")
        frame[name.upper()] = key.to_numpy()
    return context_features(frame, score_col=score_columns[0])


@dataclass(frozen=True)
class ContextQueues:
    """Result of ``context_queues``.

    ``probabilities`` (full length, aligned with the input rows) and ``folds``
    come from the nested OOF; ``orders`` are complete best-first permutations
    of positions ``0..m-1`` *within the queue rows* ``rows`` (the positions of
    the input rows that form the queue, all rows when no ``queue_mask`` was
    given): ``rows[orders[v][i]]`` is the input row at position i of the queue.
    """

    probabilities: Mapping[str, NDArray[np.float64]]
    folds: NDArray[np.int64]
    orders: Mapping[str, Order]
    rows: NDArray[np.int64]


def context_queues(
    df: pd.DataFrame,
    y: ArrayLike,
    *,
    variants: Sequence[str] = CONTEXT_VARIANTS,
    groups_col: str = "REFREC",
    score_col: str = "nota final",
    macd_columns: Sequence[str] | None = None,
    queue_mask: ArrayLike | None = None,
    tie_ids: ArrayLike | None = None,
    n_splits: int = N_SPLITS,
    seed: int = SEED,
    xgb_params: Mapping[str, Any] | None = None,
) -> ContextQueues:
    """Build the context queues of the thesis: C (``O3_contexto``) and/or CM
    (``O3_contexto_MACD``). Exploratory; see the module docstring.

    ``df`` is the full candidate universe with the comparator columns (and the
    ``macd_*`` columns when ``O3_contexto_MACD`` is requested), the ``REFREC``
    and ``COMPREC`` keys and ``nota final``. ``y`` holds binary labels (the
    thesis: ``PAR`` in {1, 2}). The model inputs are ``base_features`` (30
    attributes), then for CM ``macd_features``, then the 10 context features, in
    that column order. The OOF folds group by ``groups_col`` (``REFREC``, as in
    the thesis, or ``COMPREC``) with ``seed`` and ``n_splits``; ``xgb_params``
    overrides the thesis XGBoost parameters. The OOF always runs on every row of
    ``df`` (the context is a property of the full universe); ``queue_mask`` (a
    boolean per row, e.g. rows below the cutoff) only restricts which rows are
    ordered.

    Each order sorts by probability, descending; ties (isotonic output has many)
    follow ``tie_ids`` ascending (thesis rule: the pair-id hash, see ``fusion``)
    or the row position without ``tie_ids`` (``tie_ids`` has one id per row of
    ``df``).
    """
    unknown = [v for v in variants if v not in CONTEXT_VARIANTS]
    if unknown or not variants:
        raise ValueError(f"variants must be a non-empty subset of {CONTEXT_VARIANTS}")
    key = groups_col.upper()
    if key not in ("REFREC", "COMPREC"):
        raise ValueError("groups_col must be REFREC or COMPREC")
    labels = np.asarray(y)
    if len(labels) != len(df):
        raise ValueError("y must have one label per row of df")
    n = len(df)
    if queue_mask is None:
        rows = np.arange(n, dtype=np.int64)
    else:
        mask = np.asarray(queue_mask)
        if mask.dtype != bool or mask.shape != (n,):
            raise ValueError("queue_mask must be a boolean array, one per row")
        rows = np.flatnonzero(mask).astype(np.int64)
    ties = None
    if tie_ids is not None:
        ties = np.asarray(tie_ids)
        if ties.shape != (n,):
            raise ValueError("tie_ids must have one id per row of df")

    groups = _group_keys(df, key).to_numpy(dtype=str)
    matrices = context_design_matrices(
        df, variants=variants, score_col=score_col, macd_columns=macd_columns
    )
    probabilities: dict[str, NDArray[np.float64]] = {}
    orders: dict[str, Order] = {}
    folds: NDArray[np.int64] | None = None
    for variant in variants:
        p, variant_folds = nested_oof_probabilities(
            matrices[variant],
            labels,
            n_splits=n_splits,
            seed=seed,
            split_by=key.lower(),
            groups=groups,
            xgb_params=xgb_params,
        )
        if folds is not None and not np.array_equal(folds, variant_folds):
            raise ValueError("fold partitions of the variants diverge")
        folds = variant_folds
        probabilities[variant] = p
        orders[variant] = order_by_score(
            p[rows], tie_ids=None if ties is None else ties[rows]
        )
    if folds is None:
        raise ValueError("no variant was computed")
    return ContextQueues(
        probabilities=MappingProxyType(probabilities),
        folds=folds,
        orders=MappingProxyType(orders),
        rows=rows,
    )


def context_fusions(
    orders: Mapping[str, Order],
    *,
    fusions: Mapping[str, FusionSpec] = THESIS_CONTEXT_FUSIONS,
    tie_ids: ArrayLike | None = None,
) -> dict[str, Order]:
    """Fuse base orders by name: ``orders`` must hold every arm of the specs,
    e.g. O1 (the score queue of the thesis) and ``context_queues(...).orders``
    merged. ``tie_ids`` (one per queue position) decides RRF ties as in
    ``fusion.fuse``. All orders must cover the same queue rows. Exploratory."""
    return {name: fuse(orders, spec, tie_ids=tie_ids) for name, spec in fusions.items()}
