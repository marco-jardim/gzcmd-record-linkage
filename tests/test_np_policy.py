"""Neyman-Pearson (NP) policy: thesis parity for ``policy_kind="np"``."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from gzcmd_record_linkage.config import load_config
from gzcmd_record_linkage.np_policy import (
    NPConstraints,
    NPInfeasibleError,
    select_match_prefix,
)
from gzcmd_record_linkage.policy_engine import (
    Budget,
    Costs,
    HumanBudget,
    LLMError,
    PolicyEngineV3,
)
from gzcmd_record_linkage.runner import build_engine_from_config, run_v3

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data/synthetic/comparador_sintetico.csv"
FIXTURE = Path(__file__).parent / "fixtures" / "gzcmd_reference_config_v31.yaml"


def _np_engine(frame_budget=2, human=None, constraints=None, e_fn=0.05):
    return PolicyEngineV3(
        Costs(100, 20, 0.005),
        {"a": LLMError(0, e_fn)},
        Budget(frame_budget),
        policy_version="v3.1",
        policy_kind="np",
        human_budget=human or HumanBudget(0),
        np_constraints=constraints,
    )


def _reference_prefix(p, fixed, constraints):
    """Independent O(n^2) re-implementation of the descending-p prefix rule."""
    ordered = p.sort_values(ascending=False, kind="stable")
    guard_fp = float((1 - fixed).sum())
    best = None
    for k in range(len(ordered) + 1):
        count = len(fixed) + k
        fp = guard_fp + float((1 - ordered.iloc[:k]).sum())
        limit = (
            constraints.max_fp_rate_auto * count
            if constraints.max_fp_expected_total is None
            else constraints.max_fp_expected_total
        )
        if fp <= limit + 1e-12:
            best = k
    return None if best is None else list(ordered.index[:best])


def test_np_greedy_prefix_is_optimal_against_all_subsets() -> None:
    p = pd.Series([0.9, 1.0, 0.5, 0.995, 0.97], index=["a", "b", "c", "d", "e"])
    fixed = pd.Series([0.99, 1.0])
    constraint = NPConstraints(max_fp_rate_auto=0.02)
    chosen, info = select_match_prefix(p, fixed, constraint)
    assert chosen.tolist() == ["b", "d", "e"]
    assert info["np_fp_esperado_auto"] <= info["np_fp_limite"] + 1e-12
    feasible_rewards = []
    for size in range(len(p) + 1):
        for subset in combinations(p.index, size):
            values = p.loc[list(subset)]
            fp = float((1 - fixed).sum() + (1 - values).sum())
            if fp <= 0.02 * (len(fixed) + len(values)) + 1e-12:
                feasible_rewards.append(float(values.sum()))
    assert p.loc[chosen].sum() == pytest.approx(max(feasible_rewards))


def test_np_absolute_fp_budget_overrides_rate_and_includes_guards() -> None:
    p = pd.Series([0.99, 0.98, 0.96, 0.6])
    selected, info = select_match_prefix(
        p, pd.Series([0.99]), NPConstraints(0.00001, 0.045)
    )
    assert selected.tolist() == [0, 1]
    assert info["np_fp_esperado_auto"] == pytest.approx(0.04)
    assert info["np_fp_limite"] == 0.045


def test_np_initial_guard_rate_can_be_repaired_by_zero_risk_prefix() -> None:
    selected, info = select_match_prefix(
        pd.Series([1.0, 1.0, 0.97]), pd.Series([0.99]), NPConstraints(0.005)
    )
    assert selected.tolist() == [0, 1]
    assert info["np_taxa_fp_auto"] <= 0.005


def test_np_impossible_guards_raise_without_consuming_budgets() -> None:
    frame = pd.DataFrame(
        {
            "nota final": [9.0, 6.0],
            "p_cal": [0.9, 1.0],
            "band": ["a"] * 2,
            "guardrail": ["ALWAYS_MATCH", pd.NA],
        }
    )
    budget, human = Budget(10), HumanBudget()
    engine = PolicyEngineV3(
        Costs(100, 20, 0.005),
        {"a": LLMError(0, 0)},
        budget,
        policy_version="v3.1",
        policy_kind="np",
        human_budget=human,
        np_constraints=NPConstraints(0.001),
    )
    with pytest.raises(NPInfeasibleError) as error:
        engine.triage(frame)
    assert error.value.diagnostics["motivo"] == "guardas_fp"
    assert error.value.diagnostics["taxa_minima_atingivel"] == pytest.approx(0.05)
    assert budget.llm_used == human.used == 0


def test_np_review_gain_and_nonmatch_threshold() -> None:
    frame = pd.DataFrame(
        {"nota final": [6.0] * 4, "p_cal": [1.0, 0.99, 0.1, 0.001], "band": ["a"] * 4}
    )
    engine = _np_engine(2)
    out = engine.triage(frame)
    assert out.action.tolist() == ["MATCH", "LLM_REVIEW", "LLM_REVIEW", "NONMATCH"]
    assert out.decision_reason.tolist() == [
        "np_match_prefix",
        "np_llm_positive_gain",
        "np_llm_positive_gain",
        "np_nonmatch_threshold",
    ]
    assert out.np_fp_esperado_auto.iloc[0] <= out.np_fp_limite.iloc[0] + 1e-12
    assert engine.budget.llm_used == 2
    assert engine.np_diagnostics["np_n_match_auto"] == 1


def test_np_accepts_nota_final_underscore_column() -> None:
    frame = pd.DataFrame(
        {"nota_final": [6.0, 6.0], "p_cal": [1.0, 0.001], "band": ["a"] * 2}
    )
    out = _np_engine(0).triage(frame)
    assert out.action.tolist() == ["MATCH", "NONMATCH"]


def test_np_human_capacity_does_not_enable_unsafe_auto_fallback() -> None:
    frame = pd.DataFrame({"nota final": [6.0], "p_cal": [0.5], "band": ["a"]})
    engine = PolicyEngineV3(
        Costs(100, 20, 0.005),
        {"a": LLMError(1, 1)},
        Budget(2),
        policy_version="v3.1",
        policy_kind="np",
        human_budget=HumanBudget(0),
    )
    with pytest.raises(NPInfeasibleError) as error:
        engine.triage(frame)
    assert error.value.diagnostics["motivo"] == "orcamento_humano"
    assert error.value.diagnostics["humanos_necessarios"] == 1
    assert error.value.diagnostics["humanos_disponiveis"] == 0
    assert engine.budget.llm_used == engine.human_budget.used == 0


def test_np_human_referral_when_llm_has_no_gain_and_columns() -> None:
    frame = pd.DataFrame(
        {
            "nota final": [4.0, 9.0, 6.0, 6.0],
            "p_cal": [0.3, 0.999, 0.5, 0.001],
            "band": ["a"] * 4,
            "guardrail": ["ALWAYS_NONMATCH", "ALWAYS_MATCH", pd.NA, pd.NA],
        }
    )
    human = HumanBudget(None)
    engine = PolicyEngineV3(
        Costs(100, 20, 0.005),
        {"a": LLMError(1, 1)},
        Budget(5),
        policy_version="v3.1",
        policy_kind="np",
        human_budget=human,
        np_constraints=NPConstraints(0.01, None, 0.02),
    )
    out = engine.triage(frame)
    assert out.action.tolist() == ["NONMATCH", "MATCH", "DEFERRED", "NONMATCH"]
    assert out.decision_reason.tolist() == [
        "guard_low",
        "guard_high_probability",
        "np_human",
        "np_nonmatch_threshold",
    ]
    assert out.human_requested.tolist() == [False, False, True, False]
    assert out.review_area.tolist() == [False, False, True, True]
    assert human.used == 1 and engine.budget.llm_used == 0
    for column in (
        "selected_loss",
        "grey_zone",
        "high_consistency_failed",
        "loss_human",
    ):
        assert column in out.columns
    assert out.selected_loss.iloc[2] == pytest.approx(out.loss_human.iloc[2])


@pytest.mark.parametrize("problem", ["duplicate", "nota", "p", "guard"])
def test_np_rejects_invalid_rows(problem) -> None:
    frame = pd.DataFrame(
        {"nota final": [6.0, 6.0], "p_cal": [0.5, 0.5], "band": ["a", "a"]}
    )
    if problem == "duplicate":
        frame.index = [0, 0]
        match = "unique rows"
    elif problem == "nota":
        frame.loc[1, "nota final"] = np.nan
        match = "finite scores"
    elif problem == "p":
        frame.loc[1, "p_cal"] = 1.5
        match = r"p_cal must be finite|finite probabilities"
    else:
        frame["guardrail"] = ["ALWAYS_NONMATCH", pd.NA]
        match = "ALWAYS_NONMATCH"
    with pytest.raises(ValueError, match=match):
        _np_engine(1, HumanBudget(None)).triage(frame)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("absolute", [None, 0.3])
def test_select_match_prefix_parity_with_inline_reference(seed, absolute) -> None:
    rng = np.random.default_rng(seed)
    p = pd.Series(1 - rng.beta(0.4, 8, 60), index=[f"c{i}" for i in range(60)])
    fixed = pd.Series(1 - rng.beta(0.3, 20, 5))
    constraints = NPConstraints(0.01, absolute)
    expected = _reference_prefix(p, fixed, constraints)
    if expected is None:
        with pytest.raises(NPInfeasibleError):
            select_match_prefix(p, fixed, constraints)
        return
    chosen, info = select_match_prefix(p, fixed, constraints)
    assert list(chosen) == expected
    assert info["np_n_match_auto"] == len(fixed) + len(expected)
    assert info["np_fp_esperado_auto"] == pytest.approx(
        (1 - fixed).sum() + (1 - p.loc[expected]).sum()
    )


def test_select_match_prefix_rejects_invalid_probabilities() -> None:
    with pytest.raises(ValueError, match="finite probabilities"):
        select_match_prefix(
            pd.Series([0.5, 1.2]), pd.Series([], dtype=float), NPConstraints()
        )
    with pytest.raises(ValueError, match="finite probabilities"):
        select_match_prefix(pd.Series([0.5]), pd.Series([np.nan]), NPConstraints())


@pytest.mark.parametrize(
    "constraints", [(-0.1, None, 0.02), (0.1, -1, 0.02), (0.1, None, 2)]
)
def test_invalid_np_constraints(constraints) -> None:
    with pytest.raises(ValueError):
        NPConstraints(*constraints)


def test_np_defaults() -> None:
    constraints = NPConstraints()
    assert constraints.max_fp_rate_auto == 0.002
    assert constraints.max_fp_expected_total is None
    assert constraints.max_fn_rate_auto == 0.02


# --------------------------------------------------------------------------- #
# Engine construction and the default (cost) path
# --------------------------------------------------------------------------- #


def test_engine_policy_kind_validation_and_default() -> None:
    args = (Costs(100, 20, 0.005), {"a": LLMError(0, 0)}, Budget(1))
    assert PolicyEngineV3(*args).policy_kind == "cost"
    assert PolicyEngineV3(*args, policy_version="v3.1").policy_kind == "cost"
    with pytest.raises(ValueError, match="NP requires policy_version v3.1"):
        PolicyEngineV3(*args, policy_kind="np")
    with pytest.raises(ValueError, match="policy_kind must be cost or np"):
        PolicyEngineV3(*args, policy_version="v3.1", policy_kind="other")


def test_cost_kind_is_unchanged_by_np_constraints() -> None:
    frame = pd.DataFrame(
        {
            "nota final": [6.0, 6.0, 8.0],
            "p_cal": [0.9, 0.1, 0.5],
            "band": ["a"] * 3,
        }
    )

    def args():  # fresh budget per engine: triage consumes it
        return (Costs(100, 20, 0.005), {"a": LLMError(0.02, 0.1)}, Budget(3))

    plain = PolicyEngineV3(*args(), policy_version="v3.1").triage(frame)
    other = PolicyEngineV3(
        *args(),
        policy_version="v3.1",
        policy_kind="cost",
        np_constraints=NPConstraints(0.9, 100.0, 0.9),
    ).triage(frame)
    pd.testing.assert_frame_equal(plain, other)


# --------------------------------------------------------------------------- #
# Config parsing and runner wiring
# --------------------------------------------------------------------------- #


def _write_config(tmp_path, mutate=None, name="config.yaml"):
    data = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    # The fixture's measured-rates file lives elsewhere; use the inline rates.
    data["llm_review"]["reliability"].pop("estimated_rates_path", None)
    if mutate is not None:
        mutate(data)
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _conf(data):
    return data["decision_policy"]["modes"]["confirmacao"]


def test_fixture_config_defaults_to_cost_and_parses_np_block(tmp_path) -> None:
    cfg = load_config(FIXTURE)
    for mode_cfg in cfg.decision_policy.modes.values():
        assert mode_cfg.policy_kind == "cost"
    assert cfg.decision_policy.modes["confirmacao"].np_policy == NPConstraints(
        0.002, None, 0.02
    )
    # A mode with no ``np:`` block gets the thesis defaults.
    assert cfg.decision_policy.modes["vigilancia"].np_policy == NPConstraints()


def test_config_np_selectable_for_confirmacao_with_custom_constraints(tmp_path) -> None:
    def mutate(data):
        _conf(data)["policy_kind"] = "np"
        _conf(data)["np"] = {
            "max_fp_rate_auto": 0.01,
            "max_fp_expected_total": 3,
            "max_fn_rate_auto": 0.05,
        }

    cfg = load_config(_write_config(tmp_path, mutate))
    mode_cfg = cfg.decision_policy.modes["confirmacao"]
    assert mode_cfg.policy_kind == "np"
    assert mode_cfg.np_policy == NPConstraints(0.01, 3.0, 0.05)
    engine = build_engine_from_config(cfg, mode="confirmacao")
    assert engine.policy_kind == "np"
    assert engine.np_constraints == NPConstraints(0.01, 3.0, 0.05)


def test_config_rejects_np_outside_confirmacao_and_unknown_kind(tmp_path) -> None:
    def vigilancia_np(data):
        data["decision_policy"]["modes"]["vigilancia"]["policy_kind"] = "np"

    with pytest.raises(ValueError, match="np for confirmacao only"):
        load_config(_write_config(tmp_path, vigilancia_np))

    def unknown(data):
        _conf(data)["policy_kind"] = "minimax"

    with pytest.raises(ValueError, match="policy_kind must be cost"):
        load_config(_write_config(tmp_path, unknown, "unknown.yaml"))

    def bad_rate(data):
        _conf(data)["np"] = {"max_fp_rate_auto": 2}

    with pytest.raises(ValueError, match="NP rates"):
        load_config(_write_config(tmp_path, bad_rate, "rate.yaml"))


def test_build_engine_np_requires_v31_and_overrides(tmp_path) -> None:
    def to_np(data):
        _conf(data)["policy_kind"] = "np"

    cfg = load_config(_write_config(tmp_path, to_np))
    assert build_engine_from_config(cfg, mode="confirmacao").policy_kind == "np"
    # Thesis semantics: a v3 run is always the cost policy.
    legacy = build_engine_from_config(cfg, mode="confirmacao", policy_version="v3")
    assert legacy.policy_kind == "cost"
    # Argument overrides the YAML; NP is never selectable outside confirmacao.
    assert (
        build_engine_from_config(
            cfg, mode="confirmacao", policy_kind="cost"
        ).policy_kind
        == "cost"
    )
    custom = NPConstraints(0.5)
    engine = build_engine_from_config(cfg, mode="confirmacao", np_constraints=custom)
    assert engine.np_constraints is custom
    with pytest.raises(ValueError, match="confirmacao only"):
        build_engine_from_config(cfg, mode="vigilancia", policy_kind="np")


def test_runner_end_to_end_np_yaml(tmp_path) -> None:
    def to_np(data):
        _conf(data)["policy_kind"] = "np"
        _conf(data)["np"] = {"max_fp_rate_auto": 0.5, "max_fn_rate_auto": 0.02}

    np_path = _write_config(tmp_path, to_np)
    cost_path = _write_config(tmp_path, None, "cost.yaml")
    out, summary = run_v3(
        input_csv=CSV, config_path=np_path, mode="confirmacao", p_cal="platt"
    )
    assert summary.policy_version == "v3.1"
    assert len(out) == 607
    assert set(out.action) <= {"MATCH", "NONMATCH", "LLM_REVIEW", "DEFERRED"}
    assert set(out.decision_reason) <= {
        "guard_low",
        "guard_high_probability",
        "np_match_prefix",
        "np_nonmatch_threshold",
        "np_llm_positive_gain",
        "np_human",
    }
    assert (out.np_fp_esperado_auto <= out.np_fp_limite + 1e-12).all()
    auto = out.action.eq("MATCH")
    assert int(auto.sum()) == int(out.np_n_match_auto.iloc[0])
    expected_fp = float((1 - out.loc[auto, "p_cal"]).sum())
    assert expected_fp == pytest.approx(float(out.np_fp_esperado_auto.iloc[0]))
    assert summary.llm_used == int(out.action.eq("LLM_REVIEW").sum())
    assert summary.human_used == int(out.action.eq("DEFERRED").sum())
    # The cost policy on the same data takes the other code path.
    cost, _ = run_v3(
        input_csv=CSV, config_path=cost_path, mode="confirmacao", p_cal="platt"
    )
    assert "np_fp_limite" not in cost.columns
    assert "np_human" not in set(cost.decision_reason)
    # Run-level override gives the same result as the YAML switch.
    override, _ = run_v3(
        input_csv=CSV,
        config_path=cost_path,
        mode="confirmacao",
        p_cal="platt",
        policy_kind="np",
        np_constraints=NPConstraints(0.5, None, 0.02),
    )
    pd.testing.assert_frame_equal(out, override)


def test_runner_np_infeasible_raises_before_consuming_budgets(tmp_path) -> None:
    def impossible(data):
        _conf(data)["policy_kind"] = "np"
        _conf(data)["np"] = {"max_fp_expected_total": 0.0}

    path = _write_config(tmp_path, impossible)
    with pytest.raises(NPInfeasibleError) as error:
        run_v3(input_csv=CSV, config_path=path, mode="confirmacao", p_cal="platt")
    assert error.value.diagnostics["motivo"] == "guardas_fp"
