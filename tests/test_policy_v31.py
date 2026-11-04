from __future__ import annotations

import hashlib
import importlib
import os
import sys
from dataclasses import replace
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from gzcmd_record_linkage.bands import BandAssigner
from gzcmd_record_linkage.cli import main
from gzcmd_record_linkage.config import load_config
from gzcmd_record_linkage.guardrails import apply_guardrails
from gzcmd_record_linkage.policy_engine import (
    Budget,
    Costs,
    HumanBudget,
    HumanError,
    LLMError,
    PolicyEngineV3,
)
from gzcmd_record_linkage.runner import build_engine_from_config, run_v3

CONFIG = str(files("gzcmd_record_linkage") / "gzcmd_v3_config.yaml")
CSV = Path(__file__).resolve().parents[1] / "data/synthetic/comparador_sintetico.csv"


@pytest.fixture(scope="module")
def reference():
    root = Path(os.environ.get("GZCMD_REFERENCE_ROOT", "D:/git/phd-research"))
    if not (root / "gzcmd/gzcmd_v3_policy_engine.py").is_file():
        pytest.skip("Motor de referência ausente; defina GZCMD_REFERENCE_ROOT")
    previous_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root))
    try:
        return (
            importlib.import_module("gzcmd.gzcmd_v3_policy_engine"),
            importlib.import_module("gzcmd.guardrails"),
            importlib.import_module("gzcmd.bands"),
            importlib.import_module("gzcmd.config"),
        )
    finally:
        sys.path.remove(str(root))
        sys.dont_write_bytecode = previous_bytecode


@pytest.mark.parametrize("problem", ["duplicate", "nota", "guard"])
def test_v31_rejects_invalid_rows(problem):
    frame = pd.DataFrame(
        {"nota_final": [6, 6], "p_cal": [0.5, 0.5], "band": ["a", "a"]}
    )
    if problem == "duplicate":
        frame.index = [0, 0]
    elif problem == "nota":
        frame.loc[0, "nota_final"] = np.nan
    else:
        frame["guardrail"] = "ALWAYS_NONMATCH"
    engine = PolicyEngineV3(
        Costs(10, 50, 0.25), {"a": LLMError(0, 0)}, Budget(1), policy_version="v3.1"
    )
    with pytest.raises(ValueError):
        engine.triage(frame)
    assert engine.budget.llm_used == engine.human_budget.used == 0


def cases():
    scores = [
        0,
        2.999,
        3,
        4.999,
        5,
        5.999,
        6,
        6.999,
        7,
        7.999,
        8,
        8.999,
        9,
        9.001,
        10,
        999,
    ]
    probs = [0, 0.001, 0.1, 0.5, 0.89, 0.9, 0.95, 1]
    frame = pd.DataFrame(
        [(s, p) for s in scores for p in probs], columns=["nota_final", "p_cal"]
    )
    frame.index = pd.Index([f"pair-{i}" for i in range(len(frame))])
    frame["diff_ano"] = 10
    frame["dtnasc_all_zero"] = 1
    frame["endereco_zero"] = 1
    frame["NOME qtd frag iguais"] = 1
    frame["NOME prim frag igual"] = 1
    frame["NOME ult frag igual"] = 1
    frame["DTNASC dt iguais"] = 1
    frame["CODMUNRES local igual"] = 1
    frame["R_DTOBITO_dt"] = pd.Timestamp("2020-01-01")
    frame["C_DTDIAG_dt"] = pd.Timestamp("2021-01-01")
    return frame


@pytest.mark.parametrize("version", ["v3", "v3.1"])
@pytest.mark.parametrize("mode", ["vigilancia", "confirmacao"])
@pytest.mark.parametrize("llm_max,human_max", [(0, None), (3, 0), (5, 2), (1000, None)])
@pytest.mark.parametrize("unsure", [0, 0.2])
def test_reference_parity(reference, version, mode, llm_max, human_max, unsure):
    # Abstenção não existia no perfil histórico v3.
    unsure = unsure if version == "v3.1" else 0
    ref, guards, bands, config = reference
    cfg = load_config(CONFIG)
    frame = cases()
    frame["band"] = BandAssigner.from_config(cfg).assign(frame.nota_final)
    ref_cfg = config.load_config(CONFIG)
    pd.testing.assert_series_equal(
        frame.band,
        bands.BandAssigner.from_config(ref_cfg).assign(frame.nota_final),
        check_names=False,
    )
    actual_g = apply_guardrails(frame, policy_version=version)
    expected_g = guards.apply_guardrails(frame, policy_version=version)
    pd.testing.assert_series_equal(actual_g.guardrail, expected_g.guardrail)
    pd.testing.assert_series_equal(actual_g.reason, expected_g.reason)
    frame["guardrail"] = actual_g.guardrail
    m = cfg.decision_policy.modes[mode]
    engines = []
    for module in [sys.modules[PolicyEngineV3.__module__], ref]:
        engines.append(
            module.PolicyEngineV3(
                module.Costs(m.false_positive_cost, m.false_negative_cost, 0.005, 1),
                {
                    b: module.LLMError(r["e_fp"], r["e_fn"], unsure)
                    for b, r in cfg.llm_review.error_rates_by_band.items()
                },
                module.Budget(llm_max),
                policy_version=version,
                human_error=module.HumanError(0.05, 0.05),
                human_budget=module.HumanBudget(human_max),
                min_auto_match=m.min_auto_match_threshold,
                max_auto_nonmatch=m.max_auto_nonmatch_threshold,
            )
        )
    actual, expected = [e.triage(frame) for e in engines]
    # The historical engine did not expose the two automatic loss columns.
    pd.testing.assert_frame_equal(actual, expected.loc[:, actual.columns])
    assert engines[0].budget.llm_used == engines[1].budget.llm_used
    assert engines[0].human_budget.used == engines[1].human_budget.used
    if version == "v3.1" and llm_max == 1000:
        assert set(actual.action) == {"MATCH", "NONMATCH", "LLM_REVIEW", "DEFERRED"}


def test_four_actions_and_tie_order():
    frame = pd.DataFrame(
        {
            "nota_final": [4.99, 6, 7, 8, 9, 9.5],
            "p_cal": [0.99, 0, 0.5, 1, 0.9, 0.5],
            "band": ["a", "a", "a", "a", "a", "b"],
        }
    )
    frame["guardrail"] = apply_guardrails(frame, policy_version="v3.1").guardrail
    engine = PolicyEngineV3(
        Costs(10, 50, 0.25),
        {"a": LLMError(0, 0), "b": LLMError(1, 1)},
        Budget(1),
        policy_version="v3.1",
        human_error=HumanError(0, 0),
    )
    out = engine.triage(frame)
    assert out.action.tolist() == [
        "NONMATCH",
        "NONMATCH",
        "LLM_REVIEW",
        "MATCH",
        "MATCH",
        "DEFERRED",
    ]
    assert (engine.budget.llm_used, engine.human_budget.used) == (1, 1)
    tie = PolicyEngineV3(
        Costs(2, 2, 1),
        {"a": LLMError(0, 0)},
        Budget(2),
        policy_version="v3.1",
        human_error=HumanError(0, 0),
    )
    out = tie.triage(pd.DataFrame({"nota_final": [6], "p_cal": [0.5], "band": ["a"]}))
    assert out.action.tolist() == ["MATCH"]


def test_default_and_explicit_v3_are_identical():
    default, s_default = run_v3(
        input_csv=CSV, config_path=CONFIG, mode="vigilancia", p_cal="platt"
    )
    explicit, s_explicit = run_v3(
        input_csv=CSV,
        config_path=CONFIG,
        mode="vigilancia",
        p_cal="platt",
        policy_version="v3",
    )
    pd.testing.assert_frame_equal(default, explicit)
    assert s_default == s_explicit
    assert s_default.policy_version == "v3"
    assert load_config(CONFIG).guardrails.policy_version == "v3"
    assert (
        build_engine_from_config(load_config(CONFIG), mode="vigilancia").policy_version
        == "v3"
    )


@pytest.mark.parametrize(
    "mode,version,digest,human_used",
    [
        (
            "vigilancia",
            "v3",
            "377ef3670ec3ba9344f8d13f5eda73adbdeecebad9833d99b843e8db842ac0fa",
            0,
        ),
        (
            "confirmacao",
            "v3",
            "b798f4419a52437ac6e0f08a398c23f0006ea7690950a6a7cb3b3acbeb56c42e",
            0,
        ),
        (
            "vigilancia",
            "v3.1",
            "78be8229570fcb379450b5dad23135b5f8ab4627e1b4b9b373257172c94dd9b2",
            111,
        ),
        (
            "confirmacao",
            "v3.1",
            "58c8c9df5ea98b9bd5f2a4f0039297340806e4e1bf591b2350898364175b859f",
            109,
        ),
    ],
)
def test_versioned_synthetic_regression(mode, version, digest, human_used):
    out, summary = run_v3(
        input_csv=CSV,
        config_path=CONFIG,
        mode=mode,
        p_cal="platt",
        policy_version=version,
    )
    assert len(out) == 607
    assert hashlib.sha256("\n".join(out.action).encode()).hexdigest() == digest
    assert summary.human_used == human_used


def test_config_selection_and_override(tmp_path):
    data = yaml.safe_load(Path(CONFIG).read_text(encoding="utf-8"))
    data["guardrails"]["policy_version"] = "v3.1"
    data["decision_policy"]["modes"]["vigilancia"]["human_budget"] = {
        "max_pairs_per_window": 2
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    cfg = load_config(path)
    engine = build_engine_from_config(cfg, mode="vigilancia", human_used=1)
    assert engine.policy_version == "v3.1"
    assert engine.human_budget == HumanBudget(2, 1)
    assert (
        build_engine_from_config(
            cfg, mode="vigilancia", policy_version="v3"
        ).policy_version
        == "v3"
    )
    out, summary = run_v3(input_csv=CSV, config_path=path, mode="vigilancia")
    assert summary.policy_version == "v3.1"
    assert summary.human_used <= 2
    assert len(out) == 607
    data["guardrails"]["policy_version"] = "v4"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="policy_version"):
        load_config(path)


def test_cli_v31(tmp_path, capsys):
    output = tmp_path / "decisions.csv"
    assert (
        main(
            [
                "run",
                "--input",
                str(CSV),
                "--mode",
                "vigilancia",
                "--policy-version",
                "v3.1",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert "policy_version=v3.1" in capsys.readouterr().out
    out = pd.read_csv(output, sep=";")
    assert set(out.action) <= {"MATCH", "NONMATCH", "LLM_REVIEW", "DEFERRED"}
    assert "loss_human" in out


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_probabilities(bad):
    engine = PolicyEngineV3(
        Costs(10, 50, 0.25), {"a": LLMError(0, 0)}, Budget(0), policy_version="v3.1"
    )
    with pytest.raises(ValueError, match="p_cal"):
        engine.triage(pd.DataFrame({"nota_final": [6], "p_cal": [bad], "band": ["a"]}))


def test_high_guard_and_legacy_risks():
    frame = cases()
    guard = apply_guardrails(frame, policy_version="v3.1")
    assert not guard.guardrail[frame.nota_final.between(5, 8.999)].notna().any()
    assert (
        guard.guardrail[frame.nota_final.ge(9) & frame.p_cal.ge(0.9)]
        .eq("ALWAYS_MATCH")
        .all()
    )
    assert (
        guard.reason[frame.nota_final.ge(9) & frame.p_cal.lt(0.9)]
        .eq("high_probability_inconsistent")
        .all()
    )
    assert apply_guardrails(frame).reason.eq("temporal_filter").all()
    no_dates = frame.drop(columns=["R_DTOBITO_dt", "C_DTDIAG_dt"])
    assert (
        apply_guardrails(no_dates)
        .reason[frame.nota_final.ge(7)]
        .eq("homonimia_risk")
        .all()
    )


def test_budget_validation_and_bad_version():
    for bad in [-1, True, 1.5]:
        with pytest.raises(ValueError, match="Human"):
            HumanBudget(bad)
    with pytest.raises(ValueError, match="Human"):
        HumanError(np.nan, 0.05)
    with pytest.raises(ValueError, match="policy_version"):
        build_engine_from_config(
            load_config(CONFIG), mode="vigilancia", policy_version=""
        )
    cfg = load_config(CONFIG)
    cfg = replace(cfg, guardrails=replace(cfg.guardrails, policy_version="v3.1"))
    assert build_engine_from_config(cfg, mode="vigilancia").policy_version == "v3.1"
