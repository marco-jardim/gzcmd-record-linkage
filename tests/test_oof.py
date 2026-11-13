"""Nested OOF: determinism, leakage boundary and parity with the thesis routine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from _thesis_repo import load_thesis, requires_thesis
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from xgboost import XGBClassifier

from gzcmd_record_linkage.oof import (
    N_SPLITS,
    SEED,
    THESIS_XGB_PARAMS,
    nested_oof_probabilities,
    thesis_classifier,
)

FAST = {"n_estimators": 8, "max_depth": 3, "n_jobs": 1}


def synthetic(n_groups: int = 90, per_group: int = 3, seed: int = 7):
    rng = np.random.default_rng(seed)
    n = n_groups * per_group
    groups = np.repeat([f"G{i:03d}" for i in range(n_groups)], per_group)
    y = np.zeros(n, dtype=int)
    y[::per_group][rng.permutation(n_groups)[:36]] = 1
    signal = rng.normal(size=(n, 3)) + y[:, None] * np.array([1.5, 0.8, 0.0])
    x = pd.DataFrame(signal, columns=["a", "b", "c"])
    x.iloc[rng.integers(0, n, 15), 1] = np.nan  # XGBoost handles missing natively
    return x, y, groups


def reference_nested(x, y, *, split_by, groups, params, n_splits=N_SPLITS, seed=SEED):
    """Inline re-implementation of the thesis routine (gzcmd/ml_oof.py)."""

    def model(labels):
        settings = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": float((labels == 0).sum() / (labels == 1).sum()),
            "eval_metric": "logloss",
            "n_jobs": 8,
            "random_state": seed,
            "verbosity": 0,
        }
        settings.update(params)
        return XGBClassifier(**settings)

    grouped = split_by != "row"
    cls = StratifiedGroupKFold if grouped else StratifiedKFold
    probs = np.full(len(y), np.nan)
    folds = np.full(len(y), -1)
    outer = cls(n_splits, shuffle=True, random_state=seed)
    pairs = outer.split(x, y, groups) if grouped else outer.split(x, y)
    for fold, (train, test) in enumerate(pairs):
        inner_probs = np.full(len(train), np.nan)
        inner = cls(n_splits, shuffle=True, random_state=seed + fold + 1)
        if grouped:
            inner_pairs = inner.split(x.iloc[train], y[train], groups[train])
        else:
            inner_pairs = inner.split(x.iloc[train], y[train])
        for it, ie in inner_pairs:
            m = model(y[train[it]])
            m.fit(x.iloc[train[it]], y[train[it]])
            inner_probs[ie] = m.predict_proba(x.iloc[train[ie]])[:, 1]
        cal = IsotonicRegression(out_of_bounds="clip").fit(inner_probs, y[train])
        m = model(y[train])
        m.fit(x.iloc[train], y[train])
        probs[test] = cal.predict(m.predict_proba(x.iloc[test])[:, 1])
        folds[test] = fold
    return probs, folds


@pytest.mark.parametrize("split_by", ["refrec", "row"])
def test_parity_with_thesis_routine(split_by):
    x, y, groups = synthetic()
    got_p, got_f = nested_oof_probabilities(
        x, y, split_by=split_by, groups=groups, xgb_params=FAST
    )
    want_p, want_f = reference_nested(
        x, y, split_by=split_by, groups=groups, params=FAST
    )
    np.testing.assert_array_equal(got_f, want_f)
    np.testing.assert_array_equal(got_p, want_p)


def test_deterministic_and_seed_sensitive():
    x, y, groups = synthetic()
    kwargs = {"split_by": "refrec", "groups": groups, "xgb_params": FAST}
    p1, f1 = nested_oof_probabilities(x, y, **kwargs)
    p2, f2 = nested_oof_probabilities(x, y, **kwargs)
    np.testing.assert_array_equal(p1, p2)
    np.testing.assert_array_equal(f1, f2)
    _, f3 = nested_oof_probabilities(x, y, seed=SEED + 1, **kwargs)
    assert not np.array_equal(f1, f3)
    assert np.all((p1 >= 0) & (p1 <= 1))
    assert set(f1) == set(range(N_SPLITS))


def test_fit_predict_boundary_and_group_disjointness():
    x, y, groups = synthetic()
    calls = []

    def audit(fold, stage, fit_idx, pred_idx, test_idx):
        calls.append(stage)
        assert set(fit_idx).isdisjoint(test_idx)
        assert set(groups[fit_idx]).isdisjoint(groups[pred_idx])
        if stage == "inner":
            assert set(pred_idx).isdisjoint(test_idx)

    nested_oof_probabilities(
        x, y, split_by="refrec", groups=groups, xgb_params=FAST, audit=audit
    )
    assert calls.count("inner") == N_SPLITS * N_SPLITS
    assert calls.count("outer_and_calibrator") == N_SPLITS


def test_thesis_classifier_parameters():
    y = np.array([0] * 30 + [1] * 10)
    params = thesis_classifier(y, 42).get_params()
    for name, value in THESIS_XGB_PARAMS.items():
        assert params[name] == value
    assert params["scale_pos_weight"] == 3.0
    assert params["random_state"] == 42
    assert params["subsample"] == 0.8 and params["n_estimators"] == 300
    assert (
        thesis_classifier(y, 1, {"n_estimators": 2}).get_params()["n_estimators"] == 2
    )
    assert not hasattr(THESIS_XGB_PARAMS, "__setitem__")


def test_validation_errors():
    x, y, groups = synthetic()
    with pytest.raises(ValueError, match="binary labels"):
        nested_oof_probabilities(x, y + 1, xgb_params=FAST)
    with pytest.raises(ValueError, match="binary labels"):
        nested_oof_probabilities(x.iloc[:-1], y, xgb_params=FAST)
    few = np.zeros_like(y)
    few[:3] = 1
    with pytest.raises(ValueError, match="Too few examples"):
        nested_oof_probabilities(x, few, xgb_params=FAST)
    with pytest.raises(ValueError, match="split_by"):
        nested_oof_probabilities(x, y, split_by="other", xgb_params=FAST)
    with pytest.raises(ValueError, match="groups"):
        nested_oof_probabilities(x, y, split_by="refrec", xgb_params=FAST)
    bad = groups.astype(object)
    bad[0] = None
    with pytest.raises(ValueError, match="groups"):
        nested_oof_probabilities(x, y, split_by="refrec", groups=bad, xgb_params=FAST)
    with pytest.raises(ValueError, match="Too few distinct groups"):
        nested_oof_probabilities(
            x, y, split_by="refrec", groups=np.arange(len(y)) % 3, xgb_params=FAST
        )
    with pytest.raises(ValueError, match="default model_factory"):
        nested_oof_probabilities(
            x, y, xgb_params=FAST, model_factory=lambda labels, seed: None
        )


def _real_subsample(n_pos_groups: int, n_neg_groups: int):
    """Deterministic subsample of the thesis data keeping whole REFREC groups."""
    data = load_thesis()["data"]
    y, groups = data["y"], data["groups"]
    rng = np.random.default_rng(0)
    positive = np.unique(groups[y == 1])
    negative = np.setdiff1d(np.unique(groups), positive)
    chosen = np.concatenate(
        [
            rng.choice(positive, n_pos_groups, replace=False),
            rng.choice(negative, n_neg_groups, replace=False),
        ]
    )
    rows = np.flatnonzero(np.isin(groups, chosen))
    x = data["x"]["O3_MACD"].iloc[rows].reset_index(drop=True)
    return x, y[rows], groups[rows]


@requires_thesis
def test_parity_with_thesis_ml_oof_reduced_params_on_real_data():
    thesis = load_thesis()["ml_oof"]
    x, y, groups = _real_subsample(40, 60)
    assert y.sum() > N_SPLITS * 2
    reduced = {"n_estimators": 25, "max_depth": 4, "n_jobs": 2}

    def thesis_factory(labels, seed):
        return thesis._classifier(labels, seed).set_params(**reduced)

    want_p, want_f = thesis.nested_oof_probabilities(
        x, y, split_by="refrec", groups=groups, model_factory=thesis_factory
    )
    got_p, got_f = nested_oof_probabilities(
        x, y, split_by="refrec", groups=groups, xgb_params=reduced
    )
    np.testing.assert_array_equal(got_f, want_f)
    np.testing.assert_array_equal(got_p, want_p)


@requires_thesis
def test_parity_with_thesis_ml_oof_default_params_on_real_data():
    """Unchanged thesis routine (300 trees, depth 6) on ~2,000 real rows."""
    thesis = load_thesis()["ml_oof"]
    x, y, groups = _real_subsample(25, 35)
    want_p, want_f = thesis.nested_oof_probabilities(
        x, y, split_by="refrec", groups=groups
    )
    got_p, got_f = nested_oof_probabilities(x, y, split_by="refrec", groups=groups)
    np.testing.assert_array_equal(got_f, want_f)
    np.testing.assert_array_equal(got_p, want_p)
