"""Tests of the training-only undersampling port and its parity with the thesis."""

from __future__ import annotations

import importlib
import sys
from functools import lru_cache
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest
from _thesis_repo import THESIS_ROOT

from gzcmd_record_linkage import subamostragem as sub
from gzcmd_record_linkage.oof import nested_oof_probabilities, thesis_classifier

THESIS_MODULES_PRESENT = all(
    (THESIS_ROOT / name).exists()
    for name in ("gzcmd/subamostragem.py", "gzcmd/ml_oof.py")
)
requires_thesis_module = pytest.mark.skipif(
    not THESIS_MODULES_PRESENT, reason="thesis subamostragem module not found"
)

FAST_XGB = {"n_estimators": 8, "max_depth": 3, "n_jobs": 1}


def fast_model(y: np.ndarray, seed: int) -> Any:
    return thesis_classifier(y, seed, FAST_XGB)


@lru_cache(maxsize=1)
def thesis_module() -> ModuleType:
    """Import ``gzcmd.subamostragem`` of the thesis by path, leaving no trace."""
    owned = ("gzcmd", "scripts")

    def is_owned(name: str) -> bool:
        return any(name == ns or name.startswith(ns + ".") for ns in owned)

    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if is_owned(k)}
    bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(THESIS_ROOT))
    try:
        module = importlib.import_module("gzcmd.subamostragem")
    finally:
        sys.path.remove(str(THESIS_ROOT))
        sys.dont_write_bytecode = bytecode
        for name in [k for k in sys.modules if is_owned(k)]:
            del sys.modules[name]
        sys.modules.update(saved)
    return module


def synthetic() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    x = pd.DataFrame({"signal": np.arange(1200) / 1200})
    y = (np.arange(1200) % 12 == 0).astype(int)
    groups = np.arange(1200) // 3
    return x, y, groups


class SpyModel:
    def fit(self, x: pd.DataFrame, y: np.ndarray) -> SpyModel:
        self.fit_indices = set(x.index)
        self.offset = float(y.mean()) * 0.01
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        assert self.fit_indices.isdisjoint(x.index)
        p = 0.01 + 0.9 * x.signal.to_numpy() + self.offset
        return np.column_stack([1 - p, p])


def noisy() -> dict[str, Any]:
    """Small deterministic data with signal, groups, steps, sexes and ids."""
    rng = np.random.default_rng(7)
    n = 600
    y = (rng.random(n) < 0.12).astype(int)
    x = pd.DataFrame(
        {
            "a": rng.normal(size=n) + 1.5 * y,
            "b": rng.normal(size=n),
            "c": rng.normal(size=n) - y,
        }
    )
    sexes = rng.choice(["F", "M", "MISSING"], size=n, p=[0.45, 0.45, 0.1])
    return {
        "x": x,
        "y": y,
        "groups": np.arange(n) // 3,
        "ids": np.array([f"id-{i:04d}" for i in range(n)]),
        "below": rng.random(n) < 0.8,
        "steps": np.repeat(["6", "9", "11"], n // 3),
        "sexes": sexes,
    }


def test_constants_are_the_thesis_defaults() -> None:
    assert sub.RATIOS == (1, 5, 20, 50)
    assert tuple(sub.REMOVAL_SEEDS) == tuple(range(1000, 1010))


@pytest.mark.parametrize("ratio", [1, 5, 20, 50, None])
def test_only_negatives_removed_and_effective_ratio(ratio: int | None) -> None:
    y = np.array([1] * 13 + [0] * 1000)
    chosen, beta = sub.sample_training_indices(y, ratio, np.random.default_rng(1000))
    assert set(np.flatnonzero(y)) <= set(chosen)
    assert (y[np.setdiff1d(np.arange(len(y)), chosen)] == 0).all()
    assert np.unique(chosen).size == len(chosen)
    negatives = 1000 if ratio is None else min(1000, 13 * ratio)
    assert (y[chosen] == 0).sum() == negatives
    assert beta == negatives / 1000


def test_sampling_matches_inline_reimplementation() -> None:
    y = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    chosen, beta = sub.sample_training_indices(y, 5, np.random.default_rng(3))
    rng = np.random.default_rng(3)
    negatives = np.flatnonzero(y == 0)
    picked = rng.choice(negatives, size=min(len(negatives), 5 * 3), replace=False)
    expected = np.sort(np.concatenate([np.flatnonzero(y == 1), picked]))
    np.testing.assert_array_equal(chosen, expected)
    assert beta == len(picked) / len(negatives)
    with pytest.raises(ValueError, match="Unsupported"):
        sub.sample_training_indices(y, 2, np.random.default_rng(3))
    with pytest.raises(ValueError, match="Binary"):
        sub.sample_training_indices(np.zeros(5), 1, np.random.default_rng(3))


def test_original_factory_reproduces_unchanged_nested_oof() -> None:
    x, y, groups = synthetic()
    expected, folds = nested_oof_probabilities(
        x, y, groups=groups, split_by="refrec", model_factory=lambda *_: SpyModel()
    )
    actual, actual_folds, _ = sub.undersampled_oof(
        x, y, groups, model_factory=lambda *_: SpyModel()
    )
    np.testing.assert_array_equal(actual["isotonic"], expected)
    np.testing.assert_array_equal(actual["isotonic_prior"], expected)
    np.testing.assert_array_equal(actual_folds, folds)


def test_fit_predict_spies_prove_training_only_sampling() -> None:
    x, y, groups = synthetic()
    full_predictions: list[tuple[int, ...]] = []
    sampled_predictions: list[tuple[int, ...]] = []
    fit_events: list[tuple[np.ndarray, np.ndarray]] = []

    class RecordingModel(SpyModel):
        def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
            sampled_predictions.append(tuple(x.index))
            return super().predict_proba(x)

    class FullModel(SpyModel):
        def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
            full_predictions.append(tuple(x.index))
            return super().predict_proba(x)

    nested_oof_probabilities(
        x, y, groups=groups, split_by="refrec", model_factory=lambda *_: FullModel()
    )

    def audit(stage: str, original: np.ndarray, selected: np.ndarray) -> None:
        if stage == "fit":
            assert set(original[y[original] == 1]) <= set(selected)
            assert (y[np.setdiff1d(original, selected)] == 0).all()
            fit_events.append((original, selected))

    _, fold, effective = sub.undersampled_oof(
        x, y, groups, ratio=1, audit=audit, model_factory=lambda *_: RecordingModel()
    )
    assert sampled_predictions == full_predictions
    assert len(fit_events) == len(effective) == 30
    assert all(e["n_positive"] == e["n_negative"] for e in effective)
    assert [e["stage"] for e in effective[:6]] == ["inner"] * 5 + ["outer"]
    for outer in range(5):
        test = np.flatnonzero(fold == outer)
        for original, selected in fit_events[outer * 6 : (outer + 1) * 6]:
            assert set(groups[selected]).isdisjoint(groups[test])
            assert set(selected) <= set(original)
    outer_predictions = [
        i for j, pred in enumerate(sampled_predictions) if j % 6 == 5 for i in pred
    ]
    assert sorted(outer_predictions) == list(range(len(x)))


def test_undersampled_oof_requires_unique_index() -> None:
    x, y, groups = synthetic()
    x.index = np.zeros(len(x), dtype=int)
    with pytest.raises(ValueError, match="Unique feature index"):
        sub.undersampled_oof(x, y, groups, model_factory=lambda *_: SpyModel())


@pytest.mark.parametrize("beta", [0.001, 0.01, 0.2, 1.0])
def test_prior_equation_9_inverts_equation_2(beta: float) -> None:
    p = np.array([0, 0.0001, 0.01, 0.1, 0.5, 0.9, 1.0])
    sampled = p / (p + beta * (1 - p))
    np.testing.assert_allclose(
        sub.correct_prior(sampled, beta), p, rtol=1e-12, atol=1e-13
    )
    np.testing.assert_allclose(sub.correct_prior(0.5, 0.1), 1 / 11)


@pytest.mark.parametrize(
    ("p", "beta"), [(-0.1, 0.5), (1.1, 0.5), (0.2, 0), (0.2, 1.1), (np.nan, 0.5)]
)
def test_prior_rejects_invalid_inputs(p: float, beta: float) -> None:
    with pytest.raises(ValueError, match="beta"):
        sub.correct_prior(p, beta)


def test_ece_quantile_handles_constant_probabilities() -> None:
    assert sub.ece_quantile([0, 0, 1, 1], [0.5] * 4) == 0
    assert sub.ece_quantile([0, 0, 1, 1], [0.2] * 4) == pytest.approx(0.3)


def test_ece_quantile_matches_inline_reimplementation() -> None:
    rng = np.random.default_rng(11)
    p = rng.random(500)
    y = (rng.random(500) < p).astype(int)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, 11)))
    expected = 0.0
    for b in range(len(edges) - 1):
        inside = (p >= edges[b]) & (
            p < edges[b + 1] if b < len(edges) - 2 else p <= p.max()
        )
        expected += inside.mean() * abs(y[inside].mean() - p[inside].mean())
    assert sub.ece_quantile(y, p) == pytest.approx(expected, abs=1e-12)


def aggregate_synthetic(seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    x, y, groups = synthetic()
    probabilities, _, _ = sub.undersampled_oof(
        x, y, groups, ratio=1, removal_seed=seed, model_factory=lambda *_: SpyModel()
    )
    return sub.evaluate(
        y,
        probabilities,
        np.array([f"private-{i:04}" for i in range(len(y))]),
        np.ones(len(y), dtype=bool),
        np.repeat(["6", "9", "11"], 400),
        np.tile(["F", "M"], 600),
        ratio="1:1",
        removal_seed=seed,
    )


def test_outputs_are_aggregate_and_regeneration_identical() -> None:
    metrics, curve = aggregate_synthetic(1000)
    metrics2, curve2 = aggregate_synthetic(1000)
    pd.testing.assert_frame_equal(metrics, metrics2, check_exact=True)
    pd.testing.assert_frame_equal(curve, curve2, check_exact=True)
    reference = metrics.copy()
    reference["ratio"] = "original"
    summary, bands = sub.summarize(pd.concat([reference, metrics]), curve)
    assert summary.delta_median.dropna().eq(0).all()
    missing = summary.metric.eq("sens96_sexo_MISSING")
    assert summary.loc[missing, "delta_median"].isna().all()
    for table in (metrics, curve, summary, bands):
        assert "pair_id" not in table.columns
        assert "private-" not in table.to_csv(index=False)
    assert {"mean", "minimum", "maximum"} <= set(bands)
    assert "seed" not in bands


def test_evaluate_parameters_replace_every_study_constant() -> None:
    d = noisy()
    scores = {"p": np.linspace(0, 1, len(d["y"]))[::-1] * 0.5 + d["y"] * 0.1}
    metrics, curve = sub.evaluate(
        d["y"],
        scores,
        d["ids"],
        d["below"],
        np.repeat(["x", "y"], len(d["y"]) // 2),
        d["sexes"],
        ratio="r",
        removal_seed=5,
        top_k=(3, 10_000),
        targets=(2,),
        sensitivity_k=7,
        step_labels=("x", "y"),
        sex_labels=("F", "OTHER"),
        missing_label="OTHER",
        min_cell=1,
        curve_max_k=4,
        count_name="hits",
        step_name="stage",
        sex_name="gender",
    )
    names = set(metrics.metric)
    assert {"hits_top_3", "k_min_2", "sens7_stage_x", "sens7_gender_F"} <= names
    assert "hits_top_10000" not in names
    assert not any(n.startswith(("par2", "sens96", "k_min_9")) for n in names)
    assert list(curve.columns) == ["ratio", "seed", "variant", "k", "hits"]
    assert curve.k.max() == 4
    other = metrics.loc[metrics.metric.eq("sens7_gender_OTHER")]
    assert other.value.isna().all()
    other_ref = metrics.assign(ratio="ref")
    summary, bands = sub.summarize(
        pd.concat([other_ref, metrics]), curve, reference_ratio="ref", count_name="hits"
    )
    assert set(summary.ratio) == {"ref", "r"}
    assert {"mean", "minimum", "maximum"} <= set(bands)


def test_evaluate_matches_inline_top_k_and_sensitivity() -> None:
    y = np.array([1, 0, 1, 0, 0, 1, 0, 0, 1, 0])
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    metrics, curve = sub.evaluate(
        y,
        {"v": p},
        np.arange(10),
        np.ones(10, dtype=bool),
        np.repeat(["6", "9"], 5),
        np.tile(["F", "M"], 5),
        ratio=1,
        removal_seed=0,
        top_k=(2, 5),
        targets=(2, 9),
        sensitivity_k=3,
        step_labels=("6", "9"),
        sex_labels=("F", "M"),
        min_cell=2,
    )
    row = metrics.loc[metrics.population.eq("below")].set_index("metric").value
    assert row["par2_top_2"] == 1 and row["par2_top_5"] == 2
    assert row["k_min_2"] == 3 and np.isnan(row["k_min_9"])
    # top 3 = indices 0, 1, 2; positives there: 0 and 2 of the 4 positives.
    assert row["sens3_passo_6"] == 1.0 and row["sens3_passo_9"] == 0.0
    assert row["sens3_sexo_F_num"] == 2 and row["sens3_sexo_F_den"] == 3
    assert list(curve.par2) == [1, 1, 2, 2, 2, 3, 3, 3, 4, 4]


@requires_thesis_module
def test_thesis_constants_match() -> None:
    thesis = thesis_module()
    assert thesis.RATIOS == sub.RATIOS
    assert tuple(thesis.REMOVAL_SEEDS) == tuple(sub.REMOVAL_SEEDS)
    assert thesis.TOP_K == sub.THESIS_TOP_K


@requires_thesis_module
@pytest.mark.parametrize("ratio", [1, 5, 20, 50, None])
def test_thesis_sampling_and_prior_parity(ratio: int | None) -> None:
    thesis = thesis_module()
    y = noisy()["y"]
    a = thesis.sample_training_indices(y, ratio, np.random.default_rng(1003))
    b = sub.sample_training_indices(y, ratio, np.random.default_rng(1003))
    np.testing.assert_array_equal(a[0], b[0])
    assert a[1] == b[1]
    p = np.random.default_rng(2).random(100)
    for beta in (0.05, 0.5, 1.0):
        np.testing.assert_array_equal(
            thesis.correct_prior(p, beta), sub.correct_prior(p, beta)
        )
    q, labels = np.random.default_rng(4).random(300), (np.arange(300) % 7 == 0)
    assert thesis.ece_quantile(labels, q) == sub.ece_quantile(labels, q)


@requires_thesis_module
@pytest.mark.parametrize("ratio", [None, 1, 20])
def test_thesis_undersampled_oof_and_evaluate_parity(ratio: int | None) -> None:
    thesis = thesis_module()
    d = noisy()
    kwargs = {"ratio": ratio, "removal_seed": 1004, "model_factory": fast_model}
    expected, e_folds, e_eff = thesis.undersampled_oof(
        d["x"], d["y"], d["groups"], **kwargs
    )
    actual, a_folds, a_eff = sub.undersampled_oof(d["x"], d["y"], d["groups"], **kwargs)
    assert list(expected) == list(actual)
    for variant in expected:
        np.testing.assert_array_equal(actual[variant], expected[variant])
    np.testing.assert_array_equal(a_folds, e_folds)
    assert a_eff == e_eff
    evaluation = {key: d[key] for key in ("ids", "below", "steps", "sexes")} | {
        "ratio": str(ratio),
        "removal_seed": 1004,
    }
    t_metrics, t_curve = thesis.evaluate(d["y"], expected, **evaluation)
    l_metrics, l_curve = sub.evaluate(d["y"], actual, **evaluation)
    pd.testing.assert_frame_equal(l_metrics, t_metrics, check_exact=True)
    pd.testing.assert_frame_equal(l_curve, t_curve, check_exact=True)
    reference = t_metrics.assign(ratio="original")
    t_summary, t_bands = thesis.summarize(pd.concat([reference, t_metrics]), t_curve)
    l_summary, l_bands = sub.summarize(pd.concat([reference, l_metrics]), l_curve)
    pd.testing.assert_frame_equal(l_summary, t_summary, check_exact=True)
    pd.testing.assert_frame_equal(l_bands, t_bands, check_exact=True)
