from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from gzcmd_record_linkage.fellegi_sunter import (
    agreement_mask,
    design_matrix,
    discretize_fields,
    fit_fellegi_sunter,
)
from gzcmd_record_linkage.field_recalibrator import (
    FieldRecalibration,
    RecalibrationPrior,
    fit_field_recalibrator,
    stratum_weight_parts,
    stratum_weights,
)

NAMES = ("a:L1", "a:L2", "b:L1", "b:missing")
MASK = (True, True, True, False)


def _prior(
    mean: tuple[float, ...] = (0.5, 1.0, 2.0, 1.0),
    intercept: float = -2.0,
    sd_coef: float = 1.0,
    sd_intercept: float = 2.0,
    trust_intercept: float | None = None,
    trust_coef: float | None = None,
) -> RecalibrationPrior:
    return RecalibrationPrior(
        mean_intercept=intercept,
        mean_coef=mean,
        sd_intercept=sd_intercept,
        sd_coef=(sd_coef,) * len(mean),
        agreement_mask=MASK,
        trust_intercept=trust_intercept,
        trust_coef=trust_coef,
        feature_names=NAMES,
    )


def _data(
    n: int,
    intercept: float,
    coef: tuple[float, ...],
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = (rng.random((n, len(coef))) < 0.4).astype(float)
    p = expit(intercept + X @ np.asarray(coef))
    y = (rng.random(n) < p).astype(float)
    return X, y


def test_recovers_known_logistic_with_weak_prior() -> None:
    true_intercept, true_coef = -1.5, (1.2, 2.5, 0.8, -0.7)
    X, y = _data(40000, true_intercept, true_coef, seed=1)
    prior = _prior(
        mean=(0.0, 0.0, 0.0, 0.0), intercept=0.0, sd_coef=100, sd_intercept=100
    )
    fit = fit_field_recalibrator(X, y, prior)
    assert fit.status == "fitted"
    assert fit.intercept == pytest.approx(true_intercept, abs=0.15)
    assert np.allclose(fit.coef[:3], true_coef[:3], atol=0.15)
    # the free (missing) coefficient may be negative
    assert fit.coef[3] == pytest.approx(true_coef[3], abs=0.15)
    assert fit.n_labeled == 40000
    assert fit.n_pos + fit.n_neg == 40000
    assert fit.feature_names == NAMES


def test_few_data_stays_near_prior() -> None:
    X, y = _data(25, 3.0, (-1.0, 0.0, 0.0, 3.0), seed=2)
    prior = _prior(sd_coef=0.05, sd_intercept=0.05)
    fit = fit_field_recalibrator(X, y, prior, min_per_class=1, min_labeled=10)
    assert fit.status == "fitted"
    assert fit.intercept == pytest.approx(prior.mean_intercept, abs=0.15)
    assert np.allclose(fit.coef, prior.mean_coef, atol=0.15)


def test_agreement_coefficients_are_non_negative() -> None:
    # data say agreement is anti-correlated with the match label
    X, y = _data(5000, 0.0, (-2.0, -2.0, -2.0, -2.0), seed=3)
    prior = _prior(sd_coef=10, sd_intercept=10)
    fit = fit_field_recalibrator(X, y, prior)
    assert fit.status == "fitted"
    assert all(c >= 0.0 for c in fit.coef[:3])
    assert min(fit.coef[:3]) == pytest.approx(0.0, abs=1e-9)
    assert fit.coef[3] < -1.0  # unconstrained level goes negative


def test_trust_box_is_respected_and_intersected_with_non_negativity() -> None:
    X, y = _data(20000, -4.0, (4.0, 4.0, 4.0, 4.0), seed=4)
    prior = _prior(
        mean=(0.5, 1.0, 0.1, 0.0),
        intercept=-1.0,
        sd_coef=50,
        sd_intercept=50,
        trust_coef=0.5,
        trust_intercept=0.5,
    )
    fit = fit_field_recalibrator(X, y, prior)
    mean = np.asarray(prior.mean_coef)
    coef = np.asarray(fit.coef)
    assert np.all(coef <= mean + 0.5 + 1e-9)
    assert np.all(coef[:3] >= 0.0)
    assert np.all(coef[[0, 1, 3]] >= mean[[0, 1, 3]] - 0.5 - 1e-9)
    assert abs(fit.intercept - prior.mean_intercept) <= 0.5 + 1e-9
    # data pull everything up, so the upper edges are active
    assert coef[0] == pytest.approx(1.0, abs=1e-6)
    assert fit.intercept == pytest.approx(-0.5, abs=1e-6)


def test_trust_box_with_negative_fs_mean_pins_at_zero() -> None:
    X, y = _data(3000, 0.0, (1.0, 1.0, 1.0, 0.0), seed=5)
    prior = _prior(mean=(-2.0, 1.0, 1.0, 0.0), trust_coef=0.5, trust_intercept=1.0)
    fit = fit_field_recalibrator(X, y, prior)
    assert fit.status == "fitted"
    assert fit.coef[0] == pytest.approx(0.0, abs=1e-9)


def test_weights_change_the_fit_as_expected() -> None:
    rng = np.random.default_rng(6)
    n = 2000
    X = np.zeros((n, 4))
    X[:, 0] = rng.random(n) < 0.5
    y = np.where(X[:, 0] == 1.0, rng.random(n) < 0.8, rng.random(n) < 0.2).astype(float)
    prior = _prior(
        mean=(0.0, 0.0, 0.0, 0.0), intercept=0.0, sd_coef=100, sd_intercept=100
    )
    plain = fit_field_recalibrator(X, y, prior)
    flat = fit_field_recalibrator(X, y, prior, weights=np.ones(n))
    assert plain.coef == pytest.approx(flat.coef)
    # downweight the x=1 rows that are positive: the slope shrinks
    w = np.where((X[:, 0] == 1.0) & (y == 1.0), 0.2, 1.0)
    shrunk = fit_field_recalibrator(X, y, prior, weights=w)
    assert shrunk.coef[0] < plain.coef[0] - 0.5
    # weights doubled uniformly leave a weak-prior fit essentially unchanged
    doubled = fit_field_recalibrator(X, y, prior, weights=np.full(n, 2.0))
    assert doubled.coef[0] == pytest.approx(plain.coef[0], abs=0.02)


def test_zero_weights_ignore_rows() -> None:
    X, y = _data(3000, -1.0, (1.0, 1.0, 1.0, 0.0), seed=7)
    prior = _prior(sd_coef=100, sd_intercept=100)
    w = np.ones(len(y))
    w[1500:] = 0.0
    a = fit_field_recalibrator(X, y, prior, weights=w)
    b = fit_field_recalibrator(X[:1500], y[:1500], prior)
    # Platt targets use unweighted counts, so allow a small difference.
    assert np.allclose(a.coef, b.coef, atol=0.1)


def test_fractional_labels_are_kept_as_targets() -> None:
    n = 400
    X = np.zeros((n, 4))
    X[: n // 2, 0] = 1.0
    y = np.where(X[:, 0] == 1.0, 0.16, 0.0)
    y[:5] = 1.0  # a few hard positives so both classes pass the checks
    y[n // 2 : n // 2 + 3] = 0.0
    prior = _prior(
        mean=(0.0, 0.0, 0.0, 0.0), intercept=0.0, sd_coef=100, sd_intercept=100
    )
    fit = fit_field_recalibrator(X, y, prior, min_per_class=3)
    assert fit.status == "fitted"
    assert fit.n_pos == 5 and fit.n_neg == n - 5
    # rows with x0 = 1 are mostly 0.16 fractional targets plus five hard 1s.
    p1 = float(fit.predict_proba(np.array([[1.0, 0, 0, 0]]))[0])
    assert 0.16 < p1 < 0.22


def test_hard_labels_use_platt_targets() -> None:
    # perfectly separated data would diverge without Platt smoothing; with a
    # weak prior the fit stays finite and bounded by the target ratio.
    n = 100
    X = np.zeros((n, 4))
    X[:50, 0] = 1.0
    y = np.zeros(n)
    y[:50] = 1.0
    prior = _prior(
        mean=(0.0, 0.0, 0.0, 0.0), intercept=0.0, sd_coef=1e3, sd_intercept=1e3
    )
    fit = fit_field_recalibrator(X, y, prior)
    assert fit.status == "fitted"
    p_pos = float(fit.predict_proba(np.array([[1.0, 0, 0, 0]]))[0])
    p_neg = float(fit.predict_proba(np.array([[0.0, 0, 0, 0]]))[0])
    assert p_pos == pytest.approx(51 / 52, abs=1e-3)
    assert p_neg == pytest.approx(1 / 52, abs=1e-3)


def test_insufficient_returns_current() -> None:
    X, y = _data(15, 0.0, (1.0, 1.0, 1.0, 0.0), seed=8)
    prior = _prior()
    current = FieldRecalibration(
        intercept=-0.3,
        coef=(0.1, 0.2, 0.3, 0.4),
        feature_names=NAMES,
        status="fitted",
        n_labeled=40,
        n_pos=10,
        n_neg=30,
    )
    out = fit_field_recalibrator(X, y, prior, current=current)
    assert out.status == "skipped_insufficient"
    assert out.intercept == current.intercept and out.coef == current.coef
    assert out.n_labeled == 15
    assert out.message


def test_insufficient_without_current_returns_prior_model() -> None:
    X = np.zeros((50, 4))
    y = np.zeros(50)
    y[:2] = 1.0  # fewer than min_per_class positives
    prior = _prior()
    out = fit_field_recalibrator(X, y, prior)
    assert out.status == "skipped_insufficient"
    assert out.intercept == prior.mean_intercept
    assert out.coef == prior.mean_coef
    assert out.n_pos == 2 and out.n_neg == 48


def test_optimizer_failure_falls_back_to_current(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gzcmd_record_linkage.field_recalibrator as module

    class Failed:
        x = np.array([np.nan, 0.0, 0.0, 0.0, 0.0])
        fun = float("nan")
        success = False
        message = "boom"

    monkeypatch.setattr(module, "minimize", lambda *a, **k: Failed())
    X, y = _data(100, 0.0, (1.0, 1.0, 1.0, 0.0), seed=9)
    prior = _prior()
    current = FieldRecalibration(
        intercept=-0.3,
        coef=(0.1, 0.2, 0.3, 0.4),
        feature_names=NAMES,
        status="fitted",
        n_labeled=40,
        n_pos=10,
        n_neg=30,
    )
    out = fit_field_recalibrator(X, y, prior, current=current)
    assert out.status == "fallback_prior"
    assert (out.intercept, out.coef) == (current.intercept, current.coef)
    assert out.coef != prior.mean_coef  # the current model, not the prior
    assert out.n_labeled == 100
    assert "non-finite" in out.message
    # without `current` the (projected) prior model is the fallback
    bare = fit_field_recalibrator(X, y, prior)
    assert bare.status == "fallback_prior"
    assert bare.coef == prior.as_recalibration().coef


def test_input_validation() -> None:
    prior = _prior()
    X, y = _data(50, 0.0, (1.0, 1.0, 1.0, 0.0))
    with pytest.raises(ValueError):
        fit_field_recalibrator(X[:, :2], y, prior)
    with pytest.raises(ValueError):
        fit_field_recalibrator(X, y[:-1], prior)
    with pytest.raises(ValueError):
        fit_field_recalibrator(X, y * 2.0 + 0.5, prior)
    with pytest.raises(ValueError):
        fit_field_recalibrator(X, y, prior, weights=-np.ones(len(y)))
    with pytest.raises(ValueError):
        fit_field_recalibrator(X, y, prior, weights=np.ones(len(y) - 1))
    bad = X.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        fit_field_recalibrator(bad, y, prior)
    with pytest.raises(ValueError):
        fit_field_recalibrator(X[0], y, prior)


def test_prior_validation() -> None:
    with pytest.raises(ValueError):
        RecalibrationPrior(0.0, (0.0,), 1.0, (0.0,), (True,))
    with pytest.raises(ValueError):
        RecalibrationPrior(0.0, (0.0, 0.0), 1.0, (1.0,), (True, True))


def test_prior_from_fellegi_sunter_and_equivalence() -> None:
    rng = np.random.default_rng(3)
    n = 4000
    match = rng.random(n) < 0.15
    frame = pd.DataFrame(
        {
            col: np.where(
                match,
                rng.choice([0.0, 0.3, 0.8, 1.0], n, p=[0.1, 0.1, 0.2, 0.6]),
                rng.choice([0.0, 0.3, 0.8, 1.0], n, p=[0.6, 0.2, 0.15, 0.05]),
            )
            for col in (
                "nome_score_total",
                "mae_score_total",
                "dtnasc_score_total",
                "endereco_score_total",
            )
        }
    )
    frame["municipio_score"] = np.where(match, 1.0, rng.choice([0.0, 0.4, 1.0], n))
    frame["mae_missing"] = (rng.random(n) < 0.1).astype(float)
    levels = discretize_fields(frame)
    model = fit_fellegi_sunter(levels)
    matrix, names = design_matrix(levels)
    prior = RecalibrationPrior.from_fellegi_sunter(
        model,
        names,
        sd_intercept=1.0,
        sd_coef=0.5,
        trust_intercept=2.0,
        trust_coef=1.0,
    )
    intercept, coef = model.logistic_coefficients()
    assert prior.mean_intercept == intercept
    assert prior.mean_coef == tuple(coef)
    assert prior.agreement_mask == tuple(agreement_mask(names))
    assert prior.sd_coef == (0.5,) * len(names)
    assert (prior.trust_intercept, prior.trust_coef) == (2.0, 1.0)
    # the unprojected prior means are the exact logistic form of the FS model
    np.testing.assert_allclose(
        expit(prior.mean_intercept + matrix @ np.asarray(prior.mean_coef)),
        model.predict_proba(levels),
        atol=1e-10,
    )
    # the prior model is those means projected onto the bounds
    prior_model = prior.as_recalibration()
    assert prior_model.status == "prior"
    mean = np.asarray(prior.mean_coef)
    lower = np.where(np.asarray(prior.agreement_mask), 0.0, -np.inf)
    lower = np.maximum(lower, mean - 1.0)
    expected = np.minimum(np.maximum(mean, lower), np.maximum(mean + 1.0, lower))
    np.testing.assert_allclose(prior_model.coef, expected)
    assert prior_model.intercept == pytest.approx(
        np.clip(prior.mean_intercept, intercept - 2.0, intercept + 2.0)
    )
    with pytest.raises(ValueError):
        RecalibrationPrior.from_fellegi_sunter(
            model, names[:-1], sd_intercept=1.0, sd_coef=1.0
        )


def test_prior_model_is_projected_onto_the_bounds() -> None:
    prior = RecalibrationPrior(
        mean_intercept=5.0,
        mean_coef=(-0.7, 0.4, 3.0, -2.0),
        sd_intercept=1.0,
        sd_coef=(1.0,) * 4,
        agreement_mask=(True, True, True, False),
        trust_intercept=1.0,
        trust_coef=1.0,
        feature_names=NAMES,
    )
    model = prior.as_recalibration()
    # the mean is inside its box; the >= 0 cut moves the negative agreement coef
    assert model.intercept == 5.0
    assert model.coef == (0.0, 0.4, 3.0, -2.0)
    free = RecalibrationPrior(
        mean_intercept=0.0,
        mean_coef=(-0.7, 0.4),
        sd_intercept=1.0,
        sd_coef=(1.0, 1.0),
        agreement_mask=(True, True),
    )
    assert free.as_recalibration().coef == (0.0, 0.4)
    # the fallback of a skipped fit is the same projected model
    X = np.zeros((50, 2))
    y = np.zeros(50)
    out = fit_field_recalibrator(X, y, free)
    assert out.status == "skipped_insufficient" and out.coef == (0.0, 0.4)


def test_recalibration_round_trip() -> None:
    rec = FieldRecalibration(
        intercept=-1.25,
        coef=(0.5, 1.5, 0.0, -0.25),
        feature_names=NAMES,
        status="fitted",
        n_labeled=30,
        n_pos=10,
        n_neg=20,
        message="ok",
    )
    payload = json.loads(json.dumps(rec.to_dict()))
    assert FieldRecalibration.from_dict(payload) == rec
    X = np.eye(4)
    np.testing.assert_allclose(
        rec.predict_proba(X),
        expit(rec.intercept + X @ np.asarray(rec.coef)),
        atol=1e-12,
    )
    with pytest.raises(ValueError):
        rec.predict_proba(np.zeros((2, 3)))


def test_stratum_weights_basic_mean_one() -> None:
    strata = ["a", "a", "b", "b", "b", "c"]
    sizes = {"a": 100, "b": 30, "c": 6}
    w = stratum_weights(strata, sizes, max_ratio=100.0)
    assert w.mean() == pytest.approx(1.0)
    raw = np.array([50, 50, 10, 10, 10, 6], dtype=float)
    np.testing.assert_allclose(w, raw / raw.mean())


def test_stratum_weights_cap_is_ratio_to_minimum() -> None:
    strata = ["a"] * 4 + ["b"] * 2
    sizes = {"a": 4.0, "b": 2000.0}  # w_a = 1, w_b = 1000
    raw, capped = stratum_weight_parts(strata, sizes, max_ratio=10.0)
    assert raw.max() == pytest.approx(1000.0)
    assert capped.max() == pytest.approx(10.0)
    assert capped.min() == pytest.approx(1.0)
    w = stratum_weights(strata, sizes, max_ratio=10.0)
    assert w.mean() == pytest.approx(1.0)
    assert w.max() / w.min() == pytest.approx(10.0)


def test_stratum_weights_extreme_strata_keep_positives_relevant() -> None:
    # 200 positives in a small stratum, 58 negatives in an enormous one: the old
    # quantile cut left the positives with a vanishing share of the total weight.
    strata = ["0:0"] * 200 + ["1:3"] * 58
    sizes = {"0:0": 200.0, "1:3": 5_000_000.0}
    is_pos = np.array([True] * 200 + [False] * 58)
    w = stratum_weights(strata, sizes, max_ratio=10.0)
    assert w.max() / w.min() <= 10.0 + 1e-9
    assert w[is_pos].sum() / w.sum() > 0.2


def test_stratum_weights_counts_all_reviewed_and_restricts_to_mask() -> None:
    strata = ["a", "a", "a", "b"]
    sizes = {"a": 30.0, "b": 10.0}  # n counts all four reviewed: w_a = 10, w_b = 10
    mask = np.array([True, False, True, True])
    w = stratum_weights(strata, sizes, max_ratio=10.0, mask=mask)
    assert w.shape == (3,)
    assert w.mean() == pytest.approx(1.0)
    np.testing.assert_allclose(w, 1.0)
    with pytest.raises(ValueError):
        stratum_weights(strata, sizes, mask=[1, 0, 1, 1])


def test_stratum_weights_keys_must_be_str() -> None:
    with pytest.raises(ValueError, match="str"):
        stratum_weights([0, 0, 1], {0: 10.0, 1: 5.0})
    with pytest.raises(ValueError, match="str"):
        stratum_weights([(0, 1), (0, 1)], {(0, 1): 10.0})
    with pytest.raises(ValueError, match="str"):
        stratum_weights(np.array([["a", "b"], ["a", "b"]]), {"a": 1.0, "b": 1.0})


def test_stratum_weights_missing_size_raises() -> None:
    with pytest.raises(KeyError):
        stratum_weights(["a", "b"], {"a": 10})
    with pytest.raises(ValueError):
        stratum_weights(["a"], {"a": 1}, max_ratio=0.5)
    with pytest.raises(ValueError):
        stratum_weights(["a"], {"a": 0})
    assert stratum_weights([], {}).size == 0


def test_objective_gradient_matches_finite_differences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gzcmd_record_linkage.field_recalibrator as module

    captured: dict[str, object] = {}

    class Stop:
        x = np.array([np.nan, 0.0, 0.0, 0.0, 0.0])
        fun = float("nan")
        success = False
        message = "captured"

    def fake_minimize(fun, x0, **kwargs):  # noqa: ANN001, ANN003, ANN202
        captured["fun"] = fun
        return Stop()

    monkeypatch.setattr(module, "minimize", fake_minimize)
    X, y = _data(120, -0.5, (1.0, 0.5, 1.5, 0.0), seed=4)
    weights = np.random.default_rng(1).random(len(y)) + 0.5
    fit_field_recalibrator(X, y, _prior(), weights=weights)
    fun = captured["fun"]
    assert callable(fun)
    theta = np.array([-1.0, 0.4, 0.9, 1.3, 0.7])
    _, grad = fun(theta)
    eps = 1e-6
    numeric = np.empty_like(theta)
    for j in range(theta.size):
        step = np.zeros_like(theta)
        step[j] = eps
        numeric[j] = (fun(theta + step)[0] - fun(theta - step)[0]) / (2 * eps)
    np.testing.assert_allclose(grad, numeric, rtol=1e-5, atol=1e-6)
