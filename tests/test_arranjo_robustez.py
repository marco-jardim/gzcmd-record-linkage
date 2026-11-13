"""ASR robustness analysis: grid, dominance, verdict, intervals, checkpoints.

Inline re-implementations on synthetic data plus skip-if-absent parity against
the thesis module (loaded read-only by path) on a small real subsample.
"""

from __future__ import annotations

import inspect
import json
import os
from dataclasses import astuple

import numpy as np
import pandas as pd
import pytest
from _thesis_arranjo import load_asr, real_inputs, requires_thesis_asr
from scipy.stats import beta

from gzcmd_record_linkage import arranjo_robustez as ar
from gzcmd_record_linkage.arranjo_corte import (
    THESIS_REMOVED_FIELDS,
    nested_cutoff_probabilities,
)
from gzcmd_record_linkage.arranjo_robustez import (
    Replica,
    action_stability,
    atomic_json,
    checkpoint_matches,
    classify,
    clopper_pearson,
    dominates,
    grid,
    replace_backoff,
    replica_factory,
    run_replica,
    safety_stop,
    validate_dossiers,
)
from gzcmd_record_linkage.oof import thesis_classifier

R0 = Replica("N00", "N", 42, 42, 8)


# --- grid and factory -------------------------------------------------------------


def test_thesis_grid():
    replicas = grid()
    assert len(replicas) == 40
    block_n, block_p = replicas[:20], replicas[20:]
    assert block_n[0] == R0
    assert [r.replica for r in block_n] == [f"N{i:02d}" for i in range(20)]
    assert all(r.bloco == "N" and r.fold_seed == 42 for r in block_n)
    assert [(r.xgb_seed, r.n_jobs) for r in block_n[:4]] == [
        (42, 8),
        (42, 2),
        (101, 8),
        (101, 2),
    ]
    assert {r.xgb_seed for r in block_n} == {42, *range(101, 110)}
    assert all(r.bloco == "P" and r.xgb_seed == 42 and r.n_jobs == 8 for r in block_p)
    assert [r.fold_seed for r in block_p] == list(range(1000, 1200, 10))
    assert [r.replica for r in block_p][:2] == ["P00", "P01"]


def test_grid_is_parameterised():
    replicas = grid(
        xgb_seeds=(1, 2),
        n_jobs=(4,),
        fold_seeds=(7, 8, 9),
        base_fold_seed=3,
        base_xgb_seed=5,
        base_n_jobs=6,
    )
    assert replicas == [
        Replica("N00", "N", 3, 1, 4),
        Replica("N01", "N", 3, 2, 4),
        Replica("P00", "P", 7, 5, 6),
        Replica("P01", "P", 8, 5, 6),
        Replica("P02", "P", 9, 5, 6),
    ]


def test_replica_factory_only_changes_seed_and_n_jobs():
    y = np.array([0, 0, 0, 1, 1])
    made = replica_factory(101, 2)(y, 999)  # the fold seed is ignored
    reference = thesis_classifier(y, 999, {"random_state": 101, "n_jobs": 2})
    assert made.get_params() == reference.get_params()
    base = thesis_classifier(y, 101).get_params()
    differences = {
        k for k, v in made.get_params().items() if repr(base[k]) != repr(v)
    }  # NaN != NaN
    assert differences == {"n_jobs"}


# --- run_replica: seeds, splits and bitwise R0 -------------------------------------


@pytest.fixture(scope="module")
def synthetic_runs():
    rng = np.random.default_rng(5)
    groups = np.repeat([f"G{i:02d}" for i in range(80)], 3)
    y = np.zeros(len(groups), dtype=int)
    y[::3][:30] = 1
    x = pd.DataFrame(
        rng.normal(size=(len(y), 3)) + y[:, None] * np.array([1.5, 0.8, 0.0]),
        columns=list("abc"),
    )
    below = np.arange(len(y)) % 5 != 4
    kwargs = dict(x=x, y=y, below=below, groups=groups, par2=y.astype(bool))
    return {
        "default": nested_cutoff_probabilities(
            x, y, below, groups, par2=y.astype(bool), include_a3=False
        ),
        "r0": run_replica(**kwargs, replica=R0),
        "xgb": run_replica(**kwargs, replica=Replica("N02", "N", 42, 101, 8)),
        "fold": run_replica(**kwargs, replica=Replica("P00", "P", 1000, 42, 8)),
    }


def test_r0_is_bitwise_the_default_result(synthetic_runs):
    default, r0 = synthetic_runs["default"], synthetic_runs["r0"]
    assert set(r0[0]) == {"P", "B1", "P-Platt"}
    for arm in default[0]:
        np.testing.assert_array_equal(r0[0][arm], default[0][arm])
    np.testing.assert_array_equal(r0[1], default[1])
    assert r0[2] == default[2]


def test_fold_seed_changes_splits_and_xgb_seed_does_not(synthetic_runs):
    r0_folds = synthetic_runs["r0"][1]
    np.testing.assert_array_equal(synthetic_runs["xgb"][1], r0_folds)
    assert not np.array_equal(synthetic_runs["fold"][1], r0_folds)
    assert not np.array_equal(
        synthetic_runs["xgb"][0]["P"], synthetic_runs["r0"][0]["P"]
    )


# --- dominance, intervals, verdict ----------------------------------------------


BASE = {"aceitos": 10, "fp_abaixo": 2, "perdidos": 3, "pares_humano": 4, "pares_llm": 5}


def test_dominates_truth_table():
    assert dominates(BASE, BASE) is False  # no strict inequality
    better = {
        "aceitos": {**BASE, "aceitos": 11},
        "fp_abaixo": {**BASE, "fp_abaixo": 1},
        "perdidos": {**BASE, "perdidos": 2},
        "pares_humano": {**BASE, "pares_humano": 3},
        "pares_llm": {**BASE, "pares_llm": 4},
    }
    for key, a in better.items():
        assert dominates(a, BASE) is True, key
        assert dominates(BASE, a) is False, key
    # One strict improvement but one tolerance-zero regression is not dominance.
    assert dominates({**BASE, "aceitos": 11, "fp_abaixo": 3}, BASE) is False
    assert dominates({**BASE, "aceitos": 9, "perdidos": 0}, BASE) is False
    # Extra keys are ignored; custom vectors are honoured.
    assert dominates({**BASE, "x": 0}, {**BASE, "x": 9}) is False
    assert dominates(
        {"u": 2, "v": 1}, {"u": 1, "v": 1}, maximize=("u",), minimize=("v",)
    )
    assert not dominates(
        {"u": 1, "v": 1}, {"u": 1, "v": 1}, maximize=("u",), minimize=("v",)
    )


def test_clopper_pearson_matches_beta_quantiles():
    assert clopper_pearson(0) == (0.0, pytest.approx(1 - 0.025 ** (1 / 20)))
    assert clopper_pearson(20) == (pytest.approx(0.025 ** (1 / 20)), 1.0)
    lo, hi = clopper_pearson(18)
    assert lo == pytest.approx(beta.ppf(0.025, 18, 3))
    assert hi == pytest.approx(beta.ppf(0.975, 19, 2))
    lo, hi = clopper_pearson(3, 10, alpha=0.1)
    assert lo == pytest.approx(beta.ppf(0.05, 3, 8)) and hi == pytest.approx(
        beta.ppf(0.95, 4, 7)
    )
    for k, n in ((-1, 20), (21, 20), (0, 0), (1, 0)):
        with pytest.raises(ValueError, match="Invalid binomial counts"):
            clopper_pearson(k, n)


def test_classify_thesis_thresholds_and_parameters():
    assert classify((18, 20), (0, 1)) == "estável"
    assert classify((18, 17), (0, 1)) == "mista"
    assert classify((18, 20), (0, 2)) == "não sustentada"
    assert classify((9, 5), (0, 1)) == "não sustentada"
    assert classify((9, 12), (0, 1)) == "mista"
    assert classify((20, 20), (1, 1)) == "estável"
    assert classify((10, 10), (0, 0)) == "mista"
    for bad in ((1,), (1, 2, 3)):
        with pytest.raises(ValueError, match="Both fixed blocks"):
            classify(bad, (0, 0))
        with pytest.raises(ValueError, match="Both fixed blocks"):
            classify((20, 20), bad)
    assert classify((9, 9), (0, 0), stable_min=9) == "estável"
    assert classify((9, 12), (0, 0), unsustained_below=10) == "mista"
    assert classify((5, 5), (0, 0), unsustained_below=6) == "não sustentada"
    assert (
        classify((20, 20), (2, 2), unsustained_inverse_min=3, stable_inverse_max=2)
        == "estável"
    )


# --- safety stop, files, checkpoints ---------------------------------------------


LIMITS = dict(max_pairs=100, max_cost=2.0, max_seconds=60.0)


def test_safety_stop_boundaries():
    safety_stop(100, cost=2.0, projected_seconds=60.0, **LIMITS)  # limits are inclusive
    with pytest.raises(RuntimeError, match="STOP: pares=101"):
        safety_stop(101, cost=0.0, projected_seconds=0.0, **LIMITS)
    with pytest.raises(RuntimeError, match="custo_USD=2.00000100"):
        safety_stop(1, cost=2.000001, projected_seconds=0.0, **LIMITS)
    with pytest.raises(RuntimeError, match="projeção_s=60.001"):
        safety_stop(1, cost=0.0, projected_seconds=60.001, **LIMITS)
    # No thesis money/time defaults: every ceiling is a required keyword.
    params = inspect.signature(safety_stop).parameters
    assert all(
        p.default is inspect.Parameter.empty for n, p in params.items() if n != "n_new"
    )


def test_atomic_json_roundtrip_and_replace(tmp_path):
    path = tmp_path / "sub" / "out.json"
    atomic_json(path, {"nome": "estável", "n": [1, 2]})
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "nome": "estável",
        "n": [1, 2],
    }
    assert "estável" in path.read_text(encoding="utf-8")  # ensure_ascii=False
    atomic_json(path, {"v": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"v": 2}
    assert not path.with_suffix(".json.tmp").exists()


def test_replace_backoff_retries_permission_errors(tmp_path, monkeypatch):
    calls, sleeps = [], []

    def flaky(source, destination):
        calls.append((source, destination))
        if len(calls) < 3:
            raise PermissionError("locked")

    monkeypatch.setattr(os, "replace", flaky)
    monkeypatch.setattr(ar.time, "sleep", sleeps.append)
    replace_backoff("a", "b")
    assert len(calls) == 3 and sleeps == [0.1, 0.2]

    def always(source, destination):
        raise PermissionError("locked")

    monkeypatch.setattr(os, "replace", always)
    with pytest.raises(PermissionError):
        replace_backoff("a", "b")
    assert len(sleeps) == 2 + 7


def test_checkpoint_matches_requires_both_files_and_equal_identity(tmp_path):
    base = tmp_path / "rep"
    identity = {"fold_seed": 42, "xgb_seed": 42, "n_jobs": 8, "versions": ["1.8.0"]}
    assert checkpoint_matches(base, identity) is False
    base.with_suffix(".json").write_text(
        json.dumps({"identity": identity}), encoding="utf-8"
    )
    assert checkpoint_matches(base, identity) is False  # npz missing
    base.with_suffix(".npz").write_bytes(b"")
    assert checkpoint_matches(base, identity) is True
    assert checkpoint_matches(base, {**identity, "n_jobs": 2}) is False
    base.with_suffix(".json").write_text(json.dumps({"other": 1}), encoding="utf-8")
    assert checkpoint_matches(base, identity) is False


# --- action stability ---------------------------------------------------------------


def stability_frames():
    ids = [f"p{i}" for i in range(6)]
    reference = pd.DataFrame(
        {
            "pair_id": ids,
            "p_cal": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "action": [
                "LLM_REVIEW",
                "LLM_REVIEW",
                "AUTO",
                "AUTO",
                "LLM_REVIEW",
                "HUMAN",
            ],
        }
    )
    current = reference.copy()
    current.loc[1, ["p_cal", "action"]] = [0.25, "AUTO"]
    current.loc[3, ["p_cal", "action"]] = [0.4 + 1e-15, "LLM_REVIEW"]
    return current.sample(frac=1, random_state=0), reference, ids[:5]


def test_action_stability_inline_reference():
    current, reference, below_ids = stability_frames()
    result = action_stability(current, reference, below_ids)
    assert result["p_alterados"] == 1  # 1e-15 is within the 1e-14 tolerance
    assert result["delta_p_max"] == pytest.approx(0.05)
    assert result["proporcao_acao_alterada"] == pytest.approx(2 / 5)
    # LLM_REVIEW: current {p0, p3, p4}, reference {p0, p1, p4} -> |inter|=2, |union|=4.
    assert result["jaccard_llm"] == pytest.approx(2 / 4)
    assert action_stability(reference, reference, below_ids) == {
        "p_alterados": 0,
        "delta_p_max": 0.0,
        "proporcao_acao_alterada": 0.0,
        "jaccard_llm": 1.0,
    }
    empty = reference.assign(action="AUTO")
    assert action_stability(empty, empty, below_ids)["jaccard_llm"] == 1.0
    assert (
        action_stability(current, reference, below_ids, tolerance=0.0)["p_alterados"]
        == 2
    )


# --- dossier validation -------------------------------------------------------------


class View:
    """Minimal dossier view: ``to_json(sort_keys=True)`` is what gets hashed."""

    def __init__(self, data):
        self.data = data

    def to_json(self, sort_keys=False):
        return json.dumps(self.data, sort_keys=sort_keys)


def test_validate_dossiers_uses_journal_identity(tmp_path):
    import hashlib

    views = {"id1": View({"b": 2, "a": 1}), "id2": View({"c": 3})}
    digest = {
        k: hashlib.sha256(v.to_json(sort_keys=True).encode()).hexdigest()
        for k, v in views.items()
    }
    run = {
        "mode": "vigilancia",
        "model": "claude-haiku-5-5",
        "prompt_sha256": "ph",
        "removed_fields": list(THESIS_REMOVED_FIELDS),
        "complete": True,
        "prompt_version": "v4",
        "dossier_sha256": digest,
        "results": [{"pair_id": "id1"}, {"pair_id": "id2"}],
    }
    path = tmp_path / "run.json"
    validate_dossiers(
        path, run, "vigilancia", views, prompt_sha256="ph"
    )  # run is identity
    journal = tmp_path / "run_journal"
    journal.mkdir()
    (journal / "identity.json").write_text(
        json.dumps({**run, "dossier_sha256": {}}), encoding="utf-8"
    )
    with pytest.raises(
        ValueError, match="Dossier cache identity mismatch: vigilancia/id1"
    ):
        validate_dossiers(path, run, "vigilancia", views, prompt_sha256="ph")
    (journal / "identity.json").write_text(json.dumps(run), encoding="utf-8")
    validate_dossiers(path, run, "vigilancia", views, prompt_sha256="ph")
    with pytest.raises(ValueError, match="mismatch"):
        validate_dossiers(path, run, "confirmacao", views, prompt_sha256="ph")
    other = {**run, "model": "m2"}
    with pytest.raises(ValueError, match="mismatch"):  # journal identity says otherwise
        validate_dossiers(
            path, other, "vigilancia", views, prompt_sha256="ph", model="m2"
        )
    (journal / "identity.json").write_text(json.dumps(other), encoding="utf-8")
    validate_dossiers(path, other, "vigilancia", views, prompt_sha256="ph", model="m2")


# --- parity with the thesis module (skipped when absent) -----------------------


@requires_thesis_asr
def test_parity_grid_dominance_intervals_and_verdict():
    thesis = load_asr()["robustez"]
    assert [astuple(r) for r in grid()] == [astuple(r) for r in thesis.grid()]
    rng = np.random.default_rng(1)
    keys = ("aceitos", "fp_abaixo", "perdidos", "pares_humano", "pares_llm")
    for _ in range(300):
        a = dict(zip(keys, rng.integers(0, 3, 5).tolist(), strict=True))
        b = dict(zip(keys, rng.integers(0, 3, 5).tolist(), strict=True))
        assert dominates(a, b) == thesis.dominates(a, b)
    for k in range(21):
        assert clopper_pearson(k) == thesis.clopper_pearson(k)
    for k in range(18):
        assert clopper_pearson(k, 17, 0.1) == thesis.clopper_pearson(k, 17, 0.1)
    for counts in (
        (a, b) for a in (0, 9, 10, 17, 18, 20) for b in (0, 9, 10, 17, 18, 20)
    ):
        for inverse in ((a, b) for a in (0, 1, 2, 5) for b in (0, 1, 2, 5)):
            assert classify(counts, inverse) == thesis.classify(counts, inverse)


@requires_thesis_asr
def test_parity_files_and_stability(tmp_path):
    thesis = load_asr()["robustez"]
    current, reference, below_ids = stability_frames()
    assert action_stability(current, reference, below_ids) == thesis.action_stability(
        current, reference, below_ids
    )
    value = {"nome": "estável", "n": [1, 2.5, None]}
    atomic_json(tmp_path / "ours.json", value)
    thesis.atomic_json(tmp_path / "theirs.json", value)
    assert (tmp_path / "ours.json").read_bytes() == (
        tmp_path / "theirs.json"
    ).read_bytes()
    identity = {"seed": 1}
    for name, module_check in (
        ("ours", checkpoint_matches),
        ("theirs", thesis.checkpoint_matches),
    ):
        base = tmp_path / name
        base.with_suffix(".json").write_text(
            json.dumps({"identity": identity}), encoding="utf-8"
        )
        base.with_suffix(".npz").write_bytes(b"")
        assert module_check(base, identity) is True
        assert module_check(base, {"seed": 2}) is False


@requires_thesis_asr
def test_parity_replica_factory_and_run_replica_on_a_real_subsample():
    modules = load_asr()
    thesis = modules["robustez"]
    y = np.array([0, 0, 0, 1, 1])
    assert (
        replica_factory(101, 2)(y, 0).get_params()
        == thesis.replica_factory(101, 2)(y, 0).get_params()
    )
    x, labels, below, groups, par2 = real_inputs()
    replica = Replica("N03", "N", 42, 101, 2)
    ours = run_replica(x, labels, below, groups, par2=par2, replica=replica)
    theirs = thesis.run_replica(
        x, labels, below, groups, par2=par2, replica=thesis.Replica(*astuple(replica))
    )
    assert set(ours[0]) == set(theirs[0]) == {"P", "B1", "P-Platt"}
    for arm in ours[0]:
        np.testing.assert_array_equal(ours[0][arm], theirs[0][arm])
    np.testing.assert_array_equal(ours[1], theirs[1])
    assert ours[2] == theirs[2]
