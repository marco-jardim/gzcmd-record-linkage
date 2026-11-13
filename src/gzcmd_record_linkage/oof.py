"""Nested out-of-fold XGBoost probabilities with isotonic calibration.

Faithful port of ``nested_oof_probabilities`` (and its XGBoost factory) from the
thesis research repository (``gzcmd/ml_oof.py``). Every outer fold gets its own
inner out-of-fold training probabilities, so outer-fold labels cannot reach the
calibrator through another fold's classifier. Splits are deterministic.

This is an evaluation artifact of the thesis (supervised, cross-fitted, uses the
labels), not a deployable model and not a label-free calibration. Missing values
are passed to XGBoost untouched: the imputation (zero for the 30 comparator
attributes, -1 for the MACD indicators) happens before this routine, see
``context.base_features`` and ``context.macd_features``.

Parity details kept from the thesis: ``StratifiedKFold`` (``split_by="row"``) or
``StratifiedGroupKFold`` (``split_by="refrec"``/``"comprec"``), both with
``shuffle=True``; outer ``random_state=seed``; inner ``random_state=seed + fold
+ 1``; the model seed is always ``seed``; ``IsotonicRegression(out_of_bounds=
"clip")`` fitted on the inner probabilities against the outer-training labels.
XGBoost runs with the thesis defaults (CPU, ``n_jobs=8``); this module does not
probe for CUDA.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

SEED = 42
N_SPLITS = 5

THESIS_XGB_PARAMS: Mapping[str, Any] = MappingProxyType(
    {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "n_jobs": 8,
        "verbosity": 0,
    }
)
"""XGBoost parameters of the thesis (read-only). ``scale_pos_weight`` (negatives
over positives of each fit) and ``random_state`` (the seed) are set per fit."""

_SPLIT_BY = ("row", "refrec", "comprec")

ModelFactory = Callable[[np.ndarray, int], Any]
CalibratorFactory = Callable[[], Any]
Audit = Callable[[int, str, np.ndarray, np.ndarray, np.ndarray], None]


def thesis_classifier(
    y: ArrayLike, seed: int = SEED, params: Mapping[str, Any] | None = None
) -> Any:
    """Build the thesis XGBoost classifier for the labels ``y`` of one fit.

    ``scale_pos_weight`` is ``(y == 0).sum() / (y == 1).sum()`` and
    ``random_state`` is ``seed``. ``params`` overrides entries of
    ``THESIS_XGB_PARAMS`` (it may also override ``scale_pos_weight`` or
    ``random_state``).
    """
    from xgboost import XGBClassifier

    labels = np.asarray(y)
    settings: dict[str, Any] = {
        **THESIS_XGB_PARAMS,
        "scale_pos_weight": float((labels == 0).sum() / (labels == 1).sum()),
        "random_state": seed,
    }
    if params:
        settings.update(params)
    return XGBClassifier(**settings)


def nested_oof_probabilities(
    x: pd.DataFrame,
    y: ArrayLike,
    *,
    n_splits: int = N_SPLITS,
    seed: int = SEED,
    split_by: str = "row",
    groups: ArrayLike | None = None,
    xgb_params: Mapping[str, Any] | None = None,
    model_factory: ModelFactory | None = None,
    calibrator_factory: CalibratorFactory | None = None,
    audit: Audit | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Return calibrated out-of-fold probabilities and the outer fold of each row.

    No fit, including the inner ones, sees the outer test rows. ``y`` must be
    binary (0/1) and aligned with ``x`` (a DataFrame, indexed positionally).
    With ``split_by`` ``"refrec"`` or ``"comprec"`` the splits are grouped and
    ``groups`` (aligned, no missing values, at least ``n_splits`` distinct
    values) is required; ``"row"`` splits rows and ignores ``groups``. Each
    class needs at least ``n_splits + 1`` examples.

    ``xgb_params`` overrides the thesis XGBoost parameters (see
    ``THESIS_XGB_PARAMS``) for the default classifier; it cannot be combined with
    ``model_factory`` (a callable ``(y_fit, seed) -> model``).
    ``calibrator_factory`` defaults to ``IsotonicRegression(out_of_bounds="clip")``.
    ``audit(fold, stage, fit_idx, predict_idx, outer_test_idx)`` is called after
    each inner fit (stage ``"inner"``) and after the outer fit
    (``"outer_and_calibrator"``), which makes the fit/predict boundary testable.
    """
    if model_factory is not None and xgb_params is not None:
        raise ValueError("xgb_params applies to the default model_factory only")

    def default_model(fit_labels: np.ndarray, fit_seed: int) -> Any:
        return thesis_classifier(fit_labels, fit_seed, xgb_params)

    def default_calibrator() -> Any:
        return IsotonicRegression(out_of_bounds="clip")

    make_model = model_factory or default_model
    make_calibrator = calibrator_factory or default_calibrator

    labels = np.asarray(y, dtype=int)
    if len(x) != len(labels) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Aligned binary labels required")
    if np.bincount(labels).min() < n_splits + 1:
        raise ValueError("Too few examples per class for nested stratified OOF")
    if split_by not in _SPLIT_BY:
        raise ValueError("split_by must be row, refrec or comprec")
    grouped = split_by != "row"
    group_values: np.ndarray | None = None
    if grouped:
        if groups is None:
            raise ValueError("Aligned nonmissing groups required for group OOF")
        group_values = np.asarray(groups)
        if group_values.shape != labels.shape or pd.isna(group_values).any():
            raise ValueError("Aligned nonmissing groups required for group OOF")
        if len(np.unique(group_values)) < n_splits:
            raise ValueError("Too few distinct groups for group OOF")
    probabilities = np.full(len(labels), np.nan)
    folds = np.full(len(labels), -1, dtype=int)
    splitter = StratifiedGroupKFold if grouped else StratifiedKFold
    outer = splitter(n_splits, shuffle=True, random_state=seed)
    outer_splits = (
        outer.split(x, labels, group_values) if grouped else outer.split(x, labels)
    )
    for fold, (train, test) in enumerate(outer_splits):
        if group_values is not None and not set(group_values[train]).isdisjoint(
            group_values[test]
        ):
            raise ValueError("Outer OOF group leakage")
        if len(np.unique(labels[train])) != 2:
            raise ValueError("Outer training fold must contain both classes")
        inner_probs = np.full(len(train), np.nan)
        inner = splitter(n_splits, shuffle=True, random_state=seed + fold + 1)
        inner_splits = (
            inner.split(x.iloc[train], labels[train], group_values[train])
            if group_values is not None
            else inner.split(x.iloc[train], labels[train])
        )
        for inner_train, inner_test in inner_splits:
            fit_idx, pred_idx = train[inner_train], train[inner_test]
            if group_values is not None and not set(group_values[fit_idx]).isdisjoint(
                group_values[pred_idx]
            ):
                raise ValueError("Inner OOF group leakage")
            if len(np.unique(labels[fit_idx])) != 2:
                raise ValueError("Inner training fold must contain both classes")
            model = make_model(labels[fit_idx], seed)
            model.fit(x.iloc[fit_idx], labels[fit_idx])
            inner_probs[inner_test] = model.predict_proba(x.iloc[pred_idx])[:, 1]
            if audit is not None:
                audit(fold, "inner", fit_idx.copy(), pred_idx.copy(), test.copy())
        if not np.isfinite(inner_probs).all():
            raise ValueError("Incomplete inner OOF predictions")
        calibrator = make_calibrator()
        calibrator.fit(inner_probs, labels[train])
        model = make_model(labels[train], seed)
        model.fit(x.iloc[train], labels[train])
        probabilities[test] = calibrator.predict(
            model.predict_proba(x.iloc[test])[:, 1]
        )
        folds[test] = fold
        if audit is not None:
            audit(fold, "outer_and_calibrator", train.copy(), test.copy(), test.copy())
    if not np.isfinite(probabilities).all() or (folds < 0).any():
        raise ValueError("Incomplete outer OOF predictions")
    return probabilities, folds


def numeric_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Thesis comparator allowlist as a float64 matrix (``ml_oof.numeric_features``).

    Keeps the columns whose name starts with one of ``context.FEATURE_PREFIXES``
    plus ``nota final``/``nota_final``, in ``raw`` order, never identifiers or
    labels. Decimal commas become points, unparsable values become missing and
    missing becomes 0.0 (the thesis imputation for the comparator attributes).
    Unlike ``context.base_features`` this port does not raise when no column
    matches (it returns an empty frame), exactly as the thesis routine.
    """
    from gzcmd_record_linkage.context import FEATURE_PREFIXES

    cols = [
        c
        for c in raw
        if c.startswith(FEATURE_PREFIXES) or c in {"nota final", "nota_final"}
    ]
    x = raw[cols].apply(
        lambda s: pd.to_numeric(
            s.astype(str).str.replace(",", ".", regex=False), errors="coerce"
        )
    )
    return x.astype("float64").fillna(0.0)


def pair_ids(df: pd.DataFrame) -> pd.Series:
    """Unique ``make_pair_id(COMPREC, REFREC, PASSO)`` of each row (thesis port).

    The three columns are resolved with ``loader.resolve_prefixed_column``
    (OpenRecLink headers such as ``COMPREC,C,12,0`` are accepted) and stripped.
    Raises ``ValueError`` when a column is missing or the identifiers repeat.
    """
    from gzcmd_record_linkage.llm_dossier import make_pair_id
    from gzcmd_record_linkage.loader import resolve_prefixed_column

    cols = [
        resolve_prefixed_column(df, name) for name in ("COMPREC", "REFREC", "PASSO")
    ]
    if any(c is None for c in cols):
        raise ValueError("Missing pair metadata")
    values = [df[c].astype(str).str.strip() for c in cols]
    ids = pd.Series(
        [make_pair_id(*v) for v in zip(*values, strict=True)], index=df.index
    )
    if not ids.is_unique:
        raise ValueError("pair_id must be unique")
    return ids


_OOF_METHOD = "nested-xgb-isotonic-5x5-v1"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_xgb_oof(
    input_csv: str | Path,
    *,
    output_path: str | Path | None = None,
    reuse: bool = False,
    split_by: str = "row",
    xgb_params: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Nested out-of-fold XGBoost + isotonic probabilities of every pair of a CSV.

    Generic port of the thesis ``ml_oof.fit_xgb_oof``. The comparator CSV
    (``;`` separated, header metadata such as ``COMPREC,C,12,0`` accepted) is
    read as strings; features come from ``numeric_features`` and the label is
    ``PAR in {1, 2}``. Returns a frame with ``pair_id`` (``pair_ids``), ``nota``,
    ``p_cal_platt`` (a supervised Platt fit on ``nota final``, for comparison),
    ``p_cal_ml`` (the nested OOF probability) and ``fold``. No raw record or
    identifier other than the hashed ``pair_id`` is kept.

    ``output_path`` is where the table is persisted (plus a ``.json`` sidecar with
    the fingerprint, feature list and the CSV hash); ``None`` persists nothing
    and ``reuse`` then raises. With ``reuse=True`` a persisted table is returned
    without refitting only if the sidecar fingerprint matches the current input
    (SHA-256 of the CSV bytes, method, seed, folds, ``split_by``, the effective
    XGBoost parameters and the sklearn/xgboost versions) and the table hash and
    row count match. Otherwise it is recomputed and overwritten. The fingerprint
    does not depend on any source file of the thesis or of this package.
    """
    import sklearn
    import xgboost

    from gzcmd_record_linkage.calibration import compute_p_cal, fit_platt_from_df
    from gzcmd_record_linkage.loader import resolve_prefixed_column

    if split_by not in _SPLIT_BY:
        raise ValueError("split_by must be row, refrec or comprec")
    if reuse and output_path is None:
        raise ValueError("reuse requires output_path")
    source = Path(input_csv)
    output = Path(output_path) if output_path is not None else None
    meta_path = output.with_suffix(".json") if output is not None else None
    effective_params = {**THESIS_XGB_PARAMS, **(xgb_params or {})}
    fingerprint = json.loads(
        json.dumps(
            {
                "method": _OOF_METHOD,
                "seed": SEED,
                "folds": N_SPLITS,
                "split_by": split_by,
                "xgb_params": effective_params,
                "source_sha256": _file_sha256(source),
                "sklearn": sklearn.__version__,
                "xgboost": xgboost.__version__,
            },
            sort_keys=True,
        )
    )
    if (
        reuse
        and output is not None
        and meta_path is not None
        and output.exists()
        and meta_path.exists()
    ):
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata.get("fingerprint") == fingerprint and metadata.get(
            "csv_sha256"
        ) == _file_sha256(output):
            cached = pd.read_csv(output, dtype={"pair_id": str})
            if len(cached) == metadata.get("n_rows"):
                return cached

    raw = pd.read_csv(source, sep=";", dtype=str, low_memory=False)
    raw.columns = pd.Index([c.split(",")[0] for c in raw.columns])
    if not raw.columns.is_unique:
        raise ValueError("Ambiguous comparator headers after metadata normalization")
    x = numeric_features(raw)
    y = (
        pd.to_numeric(raw["PAR"], errors="coerce")
        .fillna(0)
        .isin([1, 2])
        .astype(int)
        .to_numpy()
    )
    groups: np.ndarray | None = None
    if split_by != "row":
        column = resolve_prefixed_column(raw, split_by.upper())
        if column is None:
            raise ValueError(f"Missing {split_by.upper()} group column")
        series = raw[column].astype("string").str.strip()
        if series.isna().any() or series.eq("").any():
            raise ValueError("Missing group identity")
        groups = series.to_numpy(dtype=str)
    probabilities, folds = nested_oof_probabilities(
        x, y, split_by=split_by, groups=groups, xgb_params=xgb_params
    )
    score = x["nota final"] if "nota final" in x else x["nota_final"]
    platt_df = pd.DataFrame({"nota final": score, "TARGET": y})
    platt = compute_p_cal(platt_df, method="platt", model=fit_platt_from_df(platt_df))
    result = pd.DataFrame(
        {
            "pair_id": pair_ids(raw),
            "nota": score,
            "p_cal_platt": platt,
            "p_cal_ml": probabilities,
            "fold": folds,
        }
    )
    if output is not None and meta_path is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False)
        meta_path.write_text(
            json.dumps(
                {
                    "fingerprint": fingerprint,
                    "features": list(x.columns),
                    "n_rows": len(result),
                    "n_groups": (
                        int(len(np.unique(groups))) if groups is not None else len(raw)
                    ),
                    "csv_sha256": _file_sha256(output),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return result
