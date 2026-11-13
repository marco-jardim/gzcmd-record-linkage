"""ASR (sequential arrangement): cut mask, nested calibration, Platt, rates, cache.

Inline re-implementations on synthetic data plus skip-if-absent parity against
the thesis modules (loaded read-only by path) on a small real subsample.
"""

from __future__ import annotations

import operator
from dataclasses import astuple

import numpy as np
import pandas as pd
import pytest
from _thesis_arranjo import load_asr, load_raw, real_subsample, requires_thesis_asr
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedGroupKFold

from gzcmd_record_linkage import arranjo_corte as ac
from gzcmd_record_linkage.arranjo_corte import (
    THESIS_CUTOFFS,
    THESIS_MODEL,
    THESIS_PARTITION,
    THESIS_PROMPT_VERSION,
    THESIS_REMOVED_FIELDS,
    LogitPlatt,
    below_cutoff,
    cache_matches,
    crossfit_rates,
    estimate_rates,
    nested_cutoff_probabilities,
    sha256,
    summarize_actions,
)
from gzcmd_record_linkage.oof import (
    nested_oof_probabilities,
    numeric_features,
    thesis_classifier,
)
from gzcmd_record_linkage.policy_engine import LLMError

FAST = {"n_estimators": 8, "max_depth": 3, "n_jobs": 1}


class Model:
    """Deterministic stand-in that records its fit rows and forbids in-sample use."""

    def fit(self, frame, labels):
        self.fitted = set(frame.index)
        self.offset = 0.01 * float(np.mean(labels)) + 0.01 * float(
            np.mean(frame.signal.to_numpy() * labels)
        )

    def predict_proba(self, frame):
        assert self.fitted.isdisjoint(frame.index)
        p = 0.02 + 0.9 * frame.signal.to_numpy() + self.offset
        return np.column_stack([1 - p, p])


def synthetic():
    groups = np.repeat(np.arange(100), 4)
    x = pd.DataFrame({"signal": np.arange(400) / 400})
    y = np.tile([0, 1, 0, 1], 100)
    return x, y, groups


def fast_factory(y, seed):
    return thesis_classifier(y, seed, FAST)


# --- below_cutoff ---------------------------------------------------------------


def raw_frame():
    return pd.DataFrame(
        {
            "PASSO": ["6", "6", "9", "9", "11", "11"],
            "SCORE": ["7,5", "7,0", "7,9", "7,8", "7,91", "7,90"],
            "PAR": ["1", "0", "1", "2", "1", "0"],
        }
    )


def test_below_cutoff_inline_reference_and_decimal_comma():
    raw = raw_frame()
    cut = {6: 7.02408276169, 9: 7.84444860082, 11: 7.90191406713}
    score = raw.SCORE.str.replace(",", ".").astype(float)
    expected = np.array(
        [s < cut[int(p)] for s, p in zip(score, raw.PASSO, strict=True)]
    )
    np.testing.assert_array_equal(below_cutoff(raw), expected)
    assert below_cutoff(raw).tolist() == [False, True, False, True, False, True]


def test_below_cutoff_custom_cutoffs_and_missing_step():
    raw = raw_frame()
    mask = below_cutoff(raw, {6: 7.4, 9: 8.0, 11: 0.0})
    assert mask.tolist() == [False, True, True, True, False, False]
    with pytest.raises(ValueError, match="Missing score or preregistered ORL step"):
        below_cutoff(raw, {6: 7.0})
    bad = raw.copy()
    bad.loc[0, "SCORE"] = "x"
    with pytest.raises(ValueError):
        below_cutoff(bad)


def test_below_cutoff_expected_partition():
    raw = raw_frame()
    # accepted at/above the cut: PAR must be exactly 1 and only there.
    assert below_cutoff(raw, expected=(3, 3, 1)).tolist() == [
        False,
        True,
        False,
        True,
        False,
        True,
    ]
    with pytest.raises(ValueError, match="differs from 2/4/1"):
        below_cutoff(raw, expected=(2, 4, 1))
    with pytest.raises(ValueError, match="differs from 3/3/0"):
        below_cutoff(raw, expected=(3, 3, 0))
    broken = raw.copy()
    broken.loc[0, "PAR"] = "0"  # an accepted pair that is not PAR=1
    with pytest.raises(ValueError, match="differs from"):
        below_cutoff(broken, expected=(3, 3, 1))
    assert below_cutoff(broken).shape == (6,)  # default: no partition check


def test_thesis_constants():
    assert dict(THESIS_CUTOFFS) == {
        6: 7.02408276169,
        9: 7.84444860082,
        11: 7.90191406713,
    }
    assert THESIS_PARTITION == (211, 61485, 36)
    assert THESIS_MODEL == "claude-haiku-5-5"
    assert THESIS_PROMPT_VERSION == "v4"
    assert len(THESIS_REMOVED_FIELDS) == 6
    assert all(f.startswith("model_outputs.") for f in THESIS_REMOVED_FIELDS)
    with pytest.raises(TypeError):
        operator.setitem(THESIS_CUTOFFS, 6, 0.0)


def test_sha256(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc")
    assert sha256(path) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


# --- LogitPlatt -----------------------------------------------------------------


def platt_data(seed=3, n=600):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.15).astype(int)
    logit = rng.normal(-2.0, 1.2, n) + 2.5 * y
    return expit(logit), y


def test_logit_platt_matches_independent_minimization():
    p, y = platt_data()
    platt = LogitPlatt().fit(p, y)
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    x = np.log(clipped / (1 - clipped))
    pos, neg = y.sum(), (y == 0).sum()
    target = np.where(y == 1, (pos + 1) / (pos + 2), 1 / (neg + 2))

    def loss(w):
        z = w[0] + w[1] * x
        return np.sum(np.logaddexp(0, z) - target * z) + 0.5 * 0.001 * w[1] ** 2

    reference = minimize(
        loss,
        [0.0, 1.0],
        method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 5000},
    )
    assert platt.optimizer == "newton_backtracking"
    assert platt.model.intercept == pytest.approx(reference.x[0], abs=1e-4)
    assert platt.model.slope == pytest.approx(reference.x[1], abs=1e-4)
    expected = expit(reference.x[0] + reference.x[1] * x)
    np.testing.assert_allclose(platt.predict(p), expected, atol=1e-4)


def test_logit_platt_validation_and_unfitted():
    with pytest.raises(ValueError, match="not been fitted"):
        LogitPlatt().predict([0.5])
    with pytest.raises(ValueError, match="binary labels"):
        LogitPlatt().fit([0.1, 0.2, 0.3], [1, 1, 1])
    with pytest.raises(ValueError, match="binary labels"):
        LogitPlatt().fit([0.1, 0.2], [0, 1, 1])
    with pytest.raises(ValueError, match="Finite logits"):
        LogitPlatt().fit([0.1, np.nan, 0.3, 0.4], [0, 1, 0, 1])


def test_logit_platt_clips_extreme_probabilities():
    p = np.array([0.0, 1e-9, 0.2, 0.7, 1.0 - 1e-9, 1.0])
    y = np.array([0, 0, 0, 1, 1, 1])
    platt = LogitPlatt().fit(p, y)
    out = platt.predict(p)
    assert np.isfinite(out).all() and ((out > 0) & (out < 1)).all()
    assert (np.diff(out) >= 0).all()


# --- nested_cutoff_probabilities ------------------------------------------------


def test_all_mask_exactly_reproduces_nested_oof(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Isotonic reproduction must not fit Platt")

    monkeypatch.setattr(LogitPlatt, "fit", forbidden)
    x, y, groups = synthetic()
    expected, expected_fold = nested_oof_probabilities(
        x, y, groups=groups, split_by="refrec", model_factory=lambda *_: Model()
    )
    observed, fold, metadata = nested_cutoff_probabilities(
        x,
        y,
        np.ones(len(y), dtype=bool),
        groups,
        model_factory=lambda *_: Model(),
        include_a3=False,
        include_platt=False,
    )
    assert set(observed) == {"P", "B1"}
    np.testing.assert_array_equal(observed["P"], expected)
    np.testing.assert_array_equal(observed["B1"], expected)
    np.testing.assert_array_equal(fold, expected_fold)
    assert all(m["principal"] == "isotonic" for m in metadata)


def reference_p_b1(x, y, below, groups, factory, n_splits=5, seed=42):
    """Inline re-implementation of arms P and B1 (isotonic, no fallback)."""
    p, b1 = np.full(len(y), np.nan), np.full(len(y), np.nan)
    outer = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
    for fold, (train, test) in enumerate(outer.split(x, y, groups)):
        inner_probs = np.full(len(train), np.nan)
        inner = StratifiedGroupKFold(
            n_splits, shuffle=True, random_state=seed + fold + 1
        )
        for a, b in inner.split(x.iloc[train], y[train], groups[train]):
            m = factory(y[train[a]], seed)
            m.fit(x.iloc[train[a]], y[train[a]])
            inner_probs[b] = m.predict_proba(x.iloc[train[b]])[:, 1]
        m = factory(y[train], seed)
        m.fit(x.iloc[train], y[train])
        raw = m.predict_proba(x.iloc[test])[:, 1]
        keep = below[train]
        p[test] = (
            IsotonicRegression(out_of_bounds="clip")
            .fit(inner_probs[keep], y[train][keep])
            .predict(raw)
        )
        b1[test] = (
            IsotonicRegression(out_of_bounds="clip")
            .fit(inner_probs, y[train])
            .predict(raw)
        )
    return p, b1


def test_p_and_b1_match_inline_reference_with_cut_mask():
    x, y, groups = synthetic()
    below = np.arange(len(y)) % 4 != 0
    expected_p, expected_b1 = reference_p_b1(x, y, below, groups, lambda *_: Model())
    observed, _, metadata = nested_cutoff_probabilities(
        x,
        y,
        below,
        groups,
        model_factory=lambda *_: Model(),
        include_a3=False,
        include_platt=False,
    )
    np.testing.assert_array_equal(observed["P"], expected_p)
    np.testing.assert_array_equal(observed["B1"], expected_b1)
    assert not np.array_equal(observed["P"], observed["B1"])
    assert [m["fold"] for m in metadata] == [0, 1, 2, 3, 4]
    assert all(m["calibration_par2"] > 10 for m in metadata)


def test_arms_audit_and_leakage_boundary():
    x, y, groups = synthetic()
    below = np.arange(len(y)) % 4 != 0
    events = []

    def audit(fold, stage, fit, predict, outer_test):
        assert set(groups[fit]).isdisjoint(groups[outer_test])
        assert set(groups[fit]).isdisjoint(groups[predict])
        if stage in {"calibrator", "A3_calibrator", "A3_inner"}:
            assert below[fit].all()
        if stage in {"inner", "A3_inner"}:
            assert set(predict).isdisjoint(outer_test)
        events.append(stage)

    probabilities, folds, _ = nested_cutoff_probabilities(
        x, y, below, groups, model_factory=lambda *_: Model(), audit=audit
    )
    assert set(probabilities) == {"P", "B1", "P-Platt", "A3"}
    assert set(events) == {
        "inner",
        "A3_inner",
        "calibrator",
        "outer_and_calibrator",
        "A3_calibrator",
    }
    assert sorted(set(folds)) == [0, 1, 2, 3, 4]
    for values in probabilities.values():
        assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()


def test_fallback_to_platt_when_few_par2_below_cut():
    x, y, groups = synthetic()
    below = np.ones(len(y), dtype=bool)
    par2 = np.zeros(len(y), dtype=bool)
    probabilities, _, metadata = nested_cutoff_probabilities(
        x,
        y,
        below,
        groups,
        par2=par2,
        model_factory=lambda *_: Model(),
        include_a3=True,
    )
    assert all(
        m["principal"] == "platt" and m["calibration_par2"] == 0 for m in metadata
    )
    np.testing.assert_array_equal(probabilities["P"], probabilities["P-Platt"])
    # Fallback is Platt even without the P-Platt arm.
    only_p, _, meta = nested_cutoff_probabilities(
        x,
        y,
        below,
        groups,
        par2=par2,
        model_factory=lambda *_: Model(),
        include_a3=False,
        include_platt=False,
    )
    np.testing.assert_array_equal(only_p["P"], probabilities["P"])
    assert set(only_p) == {"P", "B1"}
    assert all(m["principal"] == "platt" for m in meta)


def test_nested_cutoff_validation():
    x, y, groups = synthetic()
    ones = np.ones(len(y), dtype=bool)
    with pytest.raises(ValueError, match="Aligned arrays"):
        nested_cutoff_probabilities(
            x, y, ones[:-1], groups, model_factory=lambda *_: Model()
        )
    with pytest.raises(ValueError, match="Binary labels"):
        nested_cutoff_probabilities(
            x,
            np.ones(len(y), dtype=int),
            ones,
            groups,
            model_factory=lambda *_: Model(),
        )
    with pytest.raises(ValueError, match="both classes"):
        nested_cutoff_probabilities(
            x,
            y,
            y == 0,
            groups,
            model_factory=lambda *_: Model(),
            include_a3=False,
            include_platt=False,
        )


def test_default_factory_is_thesis_classifier():
    x, y, groups = synthetic()
    x = pd.DataFrame({"a": np.arange(len(y)) % 7 + y * 3.0, "b": np.arange(len(y)) % 5})
    seen = []

    def audit(fold, stage, fit, predict, outer_test):
        seen.append(stage)

    # With the default factory the fit is XGBoost (8 threads); keep it tiny.
    small = x.iloc[:200].reset_index(drop=True)
    labels, grp = y[:200], groups[:200]
    probabilities, _, _ = nested_cutoff_probabilities(
        small,
        labels,
        np.ones(200, dtype=bool),
        grp,
        include_a3=False,
        include_platt=False,
        audit=audit,
    )
    assert np.isfinite(probabilities["P"]).all() and "outer_and_calibrator" in seen


# --- reviewer rates, cache, summary --------------------------------------------


def rates_frame():
    rng = np.random.default_rng(11)
    n = 240
    band = rng.choice(["grey_low", "grey_mid", "grey_high", "near_high", "high"], n)
    band[:4] = "high"
    frame = pd.DataFrame(
        {
            "band": band,
            "y": rng.integers(0, 2, n),
            "decision": rng.choice(
                ["MATCH", "NONMATCH", "UNSURE"], n, p=[0.4, 0.4, 0.2]
            ),
            "fold": rng.integers(0, 5, n),
            "par": rng.choice([1, 2, 0], n, p=[0.3, 0.2, 0.5]),
        }
    )
    frame.loc[frame.band.eq("high").to_numpy().nonzero()[0][4:], "band"] = "near_high"
    return frame


FIXED = {
    "grey_low": {"e_fp": 0.1, "e_fn": 0.2, "e_unsure": 0.05},
    "grey_mid": {"e_fp": 0.1, "e_fn": 0.2},
    "grey_high": {"e_fp": 0.1, "e_fn": 0.2},
    "near_high": {"e_fp": 0.1, "e_fn": 0.2},
    "high": {"e_fp": 0.01, "e_fn": 0.02, "e_unsure": 0.03},
}


def test_estimate_rates_m3_and_fixed_fallback():
    frame = rates_frame()
    rates = estimate_rates(frame, FIXED)
    cell = frame[frame.band.eq("grey_low")]
    neg, pos = cell[cell.y.eq(0)], cell[cell.y.eq(1)]
    assert rates["grey_low"].e_fp == (neg.decision.eq("MATCH")).sum() / len(neg)
    assert rates["grey_low"].e_fn == (pos.decision.eq("NONMATCH")).sum() / len(pos)
    assert rates["grey_low"].e_unsure == cell.decision.eq("UNSURE").mean()
    # 'high' has 4 rows (< 10): fixed rates, e_unsure defaulting to 0 when absent.
    assert rates["high"] == LLMError(0.01, 0.02, 0.03)
    assert estimate_rates(frame.iloc[:0], {"grey_mid": FIXED["grey_mid"]})[
        "grey_mid"
    ] == (LLMError(0.1, 0.2, 0.0))
    for band_rates in rates.values():
        assert isinstance(band_rates, LLMError)


def test_estimate_rates_unsure_never_fp_or_fn():
    frame = pd.DataFrame(
        {"band": ["b"] * 12, "y": [0] * 6 + [1] * 6, "decision": ["UNSURE"] * 12}
    )
    rate = estimate_rates(frame, {"b": {"e_fp": 0.5, "e_fn": 0.5}})["b"]
    assert (rate.e_fp, rate.e_fn, rate.e_unsure) == (0.0, 0.0, 1.0)


def test_crossfit_excludes_evaluated_fold_and_singleton_is_pooled():
    frame = rates_frame()
    result = crossfit_rates(frame, FIXED, [0, 1, 2, 3, 4])
    assert list(result) == [0, 1, 2, 3, 4]
    for fold, rates in result.items():
        assert rates == estimate_rates(frame[frame.fold.ne(fold)], FIXED)
    pooled = crossfit_rates(frame, FIXED, [0])
    assert pooled[0] == estimate_rates(frame, FIXED)


def test_crossfit_par2_grey_reestimates_only_grey_fn():
    frame = rates_frame()
    base = crossfit_rates(frame, FIXED, [0, 1, 2, 3, 4])
    grey = crossfit_rates(frame, FIXED, [0, 1, 2, 3, 4], par2_grey=True)
    greys = {"grey_low", "grey_mid", "grey_high", "near_high"}
    for fold in base:
        train = frame[frame.fold.ne(fold)]
        pooled = train[train.band.isin(greys)]
        pos = pooled.par.eq(2)
        fn = (pooled.decision.eq("NONMATCH") & pos).sum() / pos.sum()
        for band, rate in grey[fold].items():
            if band in greys:
                assert rate.e_fn == fn
                assert (rate.e_fp, rate.e_unsure) == (
                    base[fold][band].e_fp,
                    base[fold][band].e_unsure,
                )
            else:
                assert rate == base[fold][band]


def run_identity(**changes):
    run = {
        "mode": "vigilancia",
        "model": THESIS_MODEL,
        "prompt_sha256": "p" * 8,
        "dossier_sha256": {"id1": "d1"},
        "removed_fields": list(THESIS_REMOVED_FIELDS),
        "complete": True,
        "prompt_version": "v4",
    }
    run.update(changes)
    return run


def test_cache_matches_requires_the_complete_identity():
    def ok(identity=None, run=None, **kw):
        args = dict(
            mode="vigilancia", pair_id="id1", prompt_sha256="p" * 8, dossier_sha256="d1"
        )
        args.update(kw)
        return cache_matches(identity or run_identity(), run or run_identity(), **args)

    assert ok() is True
    assert ok(mode="confirmacao") is False
    assert ok(pair_id="other") is False
    assert ok(dossier_sha256="x") is False
    assert ok(prompt_sha256="q") is False
    assert ok(run=run_identity(model="other")) is False
    assert ok(run=run_identity(complete=False)) is False
    assert ok(run=run_identity(prompt_version="v3")) is False
    assert (
        ok(run=run_identity(removed_fields=list(THESIS_REMOVED_FIELDS)[:-1])) is False
    )
    assert ok(identity=run_identity(dossier_sha256={})) is False
    assert ok(run=run_identity(dossier_sha256={})) is False
    # Parameters override the thesis defaults.
    assert (
        ok(model="m2", identity=run_identity(model="m2"), run=run_identity(model="m2"))
        is True
    )
    assert (
        ok(
            removed_fields=["a"],
            identity=run_identity(),
            run=run_identity(removed_fields=["a"]),
        )
        is True
    )
    assert ok(prompt_version="v9", run=run_identity(prompt_version="v9")) is True


def test_summarize_actions_allowlist_columns():
    out = pd.DataFrame(
        {
            "band": ["grey_low", "grey_low", "high"],
            "action": ["LLM_REVIEW", "LLM_REVIEW", "AUTO_MATCH"],
            "y": [1, 0, 1],
        }
    )
    summary = summarize_actions(out, "vigilancia", "P")
    assert list(summary.columns) == ac.SUMMARY_COLUMNS
    assert "y" not in summary.columns
    got = {(r.banda, r.acao): r.n for r in summary.itertuples()}
    assert got == {("grey_low", "LLM_REVIEW"): 2, ("high", "AUTO_MATCH"): 1}
    assert set(summary.modo) == {"vigilancia"} and set(summary.braco) == {"P"}


# --- parity with the thesis modules (skipped when absent) ------------------------


@requires_thesis_asr
def test_parity_constants_cache_and_summaries():
    thesis = load_asr()["corte"]
    assert dict(THESIS_CUTOFFS) == thesis.CUTOFFS
    assert THESIS_MODEL == thesis.MODEL
    assert THESIS_REMOVED_FIELDS == thesis.REMOVED_FIELDS
    assert ac.SUMMARY_COLUMNS == thesis.SUMMARY_COLUMNS
    assert ac.RATE_COLUMNS == thesis.RATE_COLUMNS

    frame = rates_frame()
    for fixed_rates in (FIXED,):
        ours = estimate_rates(frame, fixed_rates)
        theirs = thesis.estimate_rates(frame, fixed_rates)
        assert {b: astuple(r) for b, r in ours.items()} == {
            b: astuple(r) for b, r in theirs.items()
        }
        for par2_grey in (False, True):
            a = crossfit_rates(frame, fixed_rates, [0, 1, 2, 3, 4], par2_grey=par2_grey)
            b = thesis.crossfit_rates(
                frame, fixed_rates, [0, 1, 2, 3, 4], par2_grey=par2_grey
            )
            assert {f: {k: astuple(v) for k, v in r.items()} for f, r in a.items()} == {
                f: {k: astuple(v) for k, v in r.items()} for f, r in b.items()
            }
    identity = run_identity()
    for run in (
        run_identity(),
        run_identity(complete=False),
        run_identity(model="x"),
        run_identity(removed_fields=[]),
    ):
        kw = dict(
            mode="vigilancia", pair_id="id1", prompt_sha256="p" * 8, dossier_sha256="d1"
        )
        assert cache_matches(identity, run, **kw) == thesis.cache_matches(
            identity, run, **kw
        )
    out = pd.DataFrame({"band": ["a", "a", "b"], "action": ["x", "y", "x"]})
    pd.testing.assert_frame_equal(
        summarize_actions(out, "m", "P"), thesis.summarize_actions(out, "m", "P")
    )


@requires_thesis_asr
def test_parity_logit_platt_and_default_classifier():
    modules = load_asr()
    p, y = platt_data(seed=5)
    ours, theirs = LogitPlatt().fit(p, y), modules["corte"].LogitPlatt().fit(p, y)
    assert (ours.model.intercept, ours.model.slope) == (
        theirs.model.intercept,
        theirs.model.slope,
    )
    assert ours.optimizer == theirs.optimizer
    np.testing.assert_array_equal(ours.predict(p), theirs.predict(p))
    labels = np.array([0, 0, 0, 1, 1])
    assert (
        thesis_classifier(labels, 7).get_params()
        == modules["ml_oof"]._classifier(labels, 7).get_params()
    )


@requires_thesis_asr
def test_parity_below_cutoff_on_the_real_base():
    thesis = load_asr()["corte"]
    raw = load_raw()
    expected = thesis.below_cutoff(raw)  # thesis default: asserts 211/61485/36
    np.testing.assert_array_equal(
        below_cutoff(raw, expected=THESIS_PARTITION), expected
    )
    np.testing.assert_array_equal(below_cutoff(raw), expected)
    assert (int((~expected).sum()), int(expected.sum())) == THESIS_PARTITION[:2]


@requires_thesis_asr
@pytest.mark.parametrize("force_fallback", [False, True])
def test_parity_nested_cutoff_on_a_real_subsample(force_fallback):
    thesis = load_asr()["corte"]
    raw = load_raw()
    below_all = below_cutoff(raw)
    rows = real_subsample(raw, below_all)
    part = raw.iloc[rows].reset_index(drop=True)
    x = numeric_features(part)
    par = pd.to_numeric(part["PAR"], errors="coerce")
    y = par.isin([1, 2]).astype(int).to_numpy()
    par2 = np.zeros(len(part), dtype=bool) if force_fallback else par.eq(2).to_numpy()
    kwargs = dict(
        par2=par2, model_factory=fast_factory, include_a3=True, include_platt=True
    )
    below = below_all[rows]
    groups = part["REFREC"].str.strip().to_numpy()
    ours = nested_cutoff_probabilities(x, y, below, groups, **kwargs)
    theirs = thesis.nested_cutoff_probabilities(x, y, below, groups, **kwargs)
    assert set(ours[0]) == set(theirs[0]) == {"P", "B1", "P-Platt", "A3"}
    for arm in ours[0]:
        np.testing.assert_array_equal(ours[0][arm], theirs[0][arm])
    np.testing.assert_array_equal(ours[1], theirs[1])
    assert ours[2] == theirs[2]
    expected = "platt" if force_fallback else "isotonic"
    assert {m["principal"] for m in ours[2]} == {expected}
