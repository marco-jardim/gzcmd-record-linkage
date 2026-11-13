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

from collections.abc import Callable, Mapping
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
