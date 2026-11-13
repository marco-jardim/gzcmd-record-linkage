"""NP evaluation layer (``np_evaluation``): behaviour and thesis parity."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from _thesis_eval import load_thesis_eval, requires_thesis_eval, thesis_config

import gzcmd_record_linkage.config as config_module
from gzcmd_record_linkage.config import GZCMDConfig, load_config
from gzcmd_record_linkage.cost_evaluation import expected_funnel_metrics
from gzcmd_record_linkage.guardrails import apply_guardrails
from gzcmd_record_linkage.np_evaluation import (
    NP_RATES,
    compare_confirmation,
    evaluate_np_point,
)
from gzcmd_record_linkage.policy_engine import (
    Budget,
    Costs,
    HumanBudget,
    HumanError,
    LLMError,
    PolicyEngineV3,
)
from gzcmd_record_linkage.reliability import load_band_rates

PACKAGED = Path(config_module.__file__).parent / "gzcmd_v31_config.yaml"


@pytest.fixture(scope="module")
def cfg() -> GZCMDConfig:
    return load_config(PACKAGED)


@pytest.fixture(scope="module")
def rates(cfg: GZCMDConfig) -> dict[str, LLMError]:
    return load_band_rates(
        cfg.llm_review.error_rates_by_band,
        mode="confirmacao",
        source=cfg.llm_review.estimated_rates_path,
    )


def _with_guardrail(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["guardrail"] = apply_guardrails(frame, policy_version="v3.1").guardrail
    return frame


def _feasible_frame() -> pd.DataFrame:
    """50 very confident guard MATCHs, 10 grey pairs, 10 clear NONMATCHs."""
    return _with_guardrail(
        pd.DataFrame(
            {
                "nota final": [9.5] * 50 + [6.0] * 10 + [3.0] * 10,
                "p_cal": [0.9995] * 50 + [0.5] * 10 + [0.001] * 10,
                "band": ["high"] * 50 + ["grey_mid"] * 10 + ["low"] * 10,
                "TARGET": [1] * 50 + [1] * 5 + [0] * 5 + [0] * 10,
            }
        )
    )


def _mixed_frame() -> pd.DataFrame:
    """Guard FP mass .175 over 50 guard MATCHs: only a loose alpha is feasible."""
    frame = pd.DataFrame(
        {
            "nota final": [9.5] * 50 + [6.0] * 10 + [3.0] * 10,
            "p_cal": [0.9965] * 50 + [0.5] * 10 + [0.001] * 10,
            "band": ["high"] * 50 + ["grey_mid"] * 10 + ["low"] * 10,
            "TARGET": [1] * 49 + [0] + [1] * 5 + [0] * 5 + [0] * 10,
        }
    )
    return _with_guardrail(frame)


def test_np_rates_grid() -> None:
    assert NP_RATES == (0.0005, 0.001, 0.002, 0.005, 0.01)


def test_feasible_point_reports_settings_metrics_and_diagnostics(cfg, rates) -> None:
    frame = _feasible_frame()
    row, out = evaluate_np_point(frame, cfg, rates, alpha=0.001)
    assert out is not None
    mode = cfg.decision_policy.modes["confirmacao"]
    assert row["modo"] == "confirmacao"
    assert (row["policy_kind"], row["max_fp_rate_auto"]) == ("np", 0.001)
    assert row["max_fn_rate_auto"] == mode.np_policy.max_fn_rate_auto
    assert row["C_llm"] == mode.llm_review_cost
    assert (row["e_fp_h"], row["e_fn_h"]) == (mode.human_e_fp, mode.human_e_fn)
    assert row["orcamento_humano"] == np.inf  # config budget is None
    assert row["viavel"] is True and row["status"] == "viavel"
    # Metrics are the cost-side metrics of the triaged frame, plus NP diagnostics.
    human = HumanError(mode.human_e_fp, mode.human_e_fn)
    for key, value in expected_funnel_metrics(out, rates, human).items():
        assert row[key] == pytest.approx(value, nan_ok=True), key
    # 50 compulsory guard MATCHs, expected FP sum = 50 * .0005 = .025 <= .001 * 50.
    assert row["ALWAYS_MATCH"] == row["np_n_match_auto"] == row["n_guardas_match"] == 50
    assert row["np_fp_esperado_auto"] == pytest.approx(0.025)
    assert row["np_taxa_fp_auto"] <= 0.001
    assert row["np_fp_limite"] == pytest.approx(0.05)
    assert out.action.value_counts().to_dict() == {
        "MATCH": 50,
        "DEFERRED": 10,
        "NONMATCH": 10,
    }


def test_infeasible_point_has_diagnostics_and_no_fabricated_metrics(cfg, rates) -> None:
    frame = _with_guardrail(
        pd.DataFrame(
            {
                "nota final": [9.5] * 100,
                "p_cal": [0.95] * 100,
                "band": ["high"] * 100,
                "TARGET": [1] * 100,
            }
        )
    )
    row, out = evaluate_np_point(frame, cfg, rates, alpha=0.001)
    assert out is None
    assert row["viavel"] is False and row["status"] == "inviavel"
    assert row["motivo"] == "guardas_fp"
    assert row["fp_esperado_guardas"] == pytest.approx(5.0)
    assert row["n_guardas_match"] == 100
    assert row["taxa_minima_atingivel"] == pytest.approx(0.05)
    for metric in ("AUTO_MATCH", "fp_esperado_total", "perda_esperada_total"):
        assert metric not in row


def test_cost_kind_is_unconstrained_and_has_no_alpha(cfg, rates) -> None:
    row, out = evaluate_np_point(
        _feasible_frame(), cfg, rates, alpha=0.001, kind="cost"
    )
    assert out is not None
    assert row["viavel"] is True and row["status"] == "custo_sem_restricao"
    assert row["policy_kind"] == "cost" and np.isnan(row["max_fp_rate_auto"])
    assert "np_n_match_auto" not in row


def test_cost_kind_is_never_infeasible(cfg, rates) -> None:
    frame = _with_guardrail(
        pd.DataFrame(
            {
                "nota final": [9.5] * 100,
                "p_cal": [0.95] * 100,
                "band": ["high"] * 100,
                "TARGET": [1] * 100,
            }
        )
    )
    row, out = evaluate_np_point(frame, cfg, rates, alpha=0.001, kind="cost")
    assert out is not None and row["viavel"] is True
    assert row["fp_esperado_auto_match_incl_guardas"] == pytest.approx(5.0)


def test_human_error_and_budget_overrides(cfg, rates) -> None:
    frame = _feasible_frame()
    row, _ = evaluate_np_point(frame, cfg, rates, alpha=0.001, error=0.2, cap=3)
    assert (row["e_fp_h"], row["e_fn_h"]) == (0.2, 0.2)
    assert row["orcamento_humano"] == 3
    assert row["viavel"] is True
    default, _ = evaluate_np_point(frame, cfg, rates, alpha=0.001)
    assert (default["e_fp_h"], default["e_fn_h"]) == (0.05, 0.05)
    # Human error changes the expected loss of the deferred pairs.
    assert row["perda_esperada_total"] != pytest.approx(default["perda_esperada_total"])
    row, _ = evaluate_np_point(frame, cfg, rates, alpha=0.001, cap=None)
    assert row["orcamento_humano"] == np.inf and row["viavel"] is True


def test_invalid_alpha_is_rejected(cfg, rates) -> None:
    with pytest.raises(ValueError, match="NP rates must be finite probabilities"):
        evaluate_np_point(_feasible_frame(), cfg, rates, alpha=1.5)


def test_engine_triage_matches_directly_built_engine(cfg, rates) -> None:
    """The returned frame is exactly what an equivalent engine produces."""
    frame = _mixed_frame()
    mode = cfg.decision_policy.modes["confirmacao"]
    engine = PolicyEngineV3(
        Costs(
            mode.false_positive_cost,
            mode.false_negative_cost,
            mode.llm_review_cost,
            mode.human_review_cost,
        ),
        rates,
        Budget(mode.llm_max_calls_per_window),
        policy_version="v3.1",
        policy_kind="np",
        human_error=HumanError(mode.human_e_fp, mode.human_e_fn),
        human_budget=HumanBudget(None),
        np_constraints=replace(mode.np_policy, max_fp_rate_auto=0.01),
    )
    expected = engine.triage(frame)
    row, out = evaluate_np_point(frame, cfg, rates, alpha=0.01)
    pd.testing.assert_frame_equal(out, expected)
    for key, value in engine.np_diagnostics.items():
        assert row[key] == pytest.approx(value)


def test_compare_confirmation_runs_cost_then_np_at_three_alphas(cfg) -> None:
    table = compare_confirmation(
        _mixed_frame(), cfg, cfg.llm_review.estimated_rates_path
    )
    assert table.policy_kind.tolist() == ["cost", "np", "np", "np"]
    alphas = table.max_fp_rate_auto.tolist()
    assert np.isnan(alphas[0]) and alphas[1:] == [0.001, 0.002, 0.01]
    assert table.status.tolist() == [
        "custo_sem_restricao",
        "inviavel",
        "inviavel",
        "viavel",
    ]
    infeasible = table[table.status.eq("inviavel")]
    assert infeasible.motivo.eq("guardas_fp").all()
    assert infeasible.AUTO_MATCH.isna().all()
    assert infeasible.fp_esperado_total.isna().all()
    assert infeasible.taxa_minima_atingivel.tolist() == pytest.approx([0.0035] * 2)
    feasible = table[table.status.eq("viavel")].iloc[0]
    assert feasible.np_taxa_fp_auto == pytest.approx(0.0035)


def test_compare_confirmation_uses_the_given_rates_file(cfg, tmp_path) -> None:
    """Measured rates from ``rates_path`` (>=10 decisions) override the fallback."""
    import yaml

    block = {
        "estimates_by_band": {"grey_mid": {"n": 10, "u": 0.3}},
        "error_rates_by_band": {"grey_mid": {"e_fp": 0.4, "e_fn": 0.5}},
    }
    source = tmp_path / "rates.yaml"
    source.write_text(
        yaml.safe_dump({"by_mode": {"confirmacao": {"pooled": block}}}),
        encoding="utf-8",
    )
    frame = _feasible_frame()
    default = compare_confirmation(frame, cfg, cfg.llm_review.estimated_rates_path)
    custom = compare_confirmation(frame, cfg, source)
    # Grey pairs are sent to LLM review in the cost policy: u=.3 vs the packaged .0.
    cost_default, cost_custom = default.iloc[0], custom.iloc[0]
    assert cost_custom.lote_llm == cost_default.lote_llm
    assert cost_custom.humano_via_llm_esperado == pytest.approx(
        0.3 * cost_custom.lote_llm
    )
    assert cost_default.humano_via_llm_esperado == pytest.approx(0.0)


@requires_thesis_eval
def test_thesis_parity_point_and_table(rates, tmp_path) -> None:
    thesis = load_thesis_eval()
    t_cfg = thesis_config()
    lib_cfg = load_config(PACKAGED)
    t_rates = thesis.np_evaluation.load_band_rates(
        t_cfg.llm_review.error_rates_by_band,
        mode="confirmacao",
        source=lib_cfg.llm_review.estimated_rates_path,
    )
    assert {b: tuple(vars(r).values()) for b, r in t_rates.items()} == {
        b: tuple(vars(r).values()) for b, r in rates.items()
    }
    for frame in (_feasible_frame(), _mixed_frame()):
        for kwargs in (
            {"alpha": 0.001},
            {"alpha": 0.01},
            {"alpha": 0.002, "kind": "cost"},
            {"alpha": 0.01, "error": 0.1, "cap": 4},
            {"alpha": 0.01, "cap": None},
        ):
            t_row, t_out = thesis.np_evaluation.evaluate_np_point(
                frame, t_cfg, t_rates, **kwargs
            )
            row, out = evaluate_np_point(frame, lib_cfg, rates, **kwargs)
            assert list(row) == list(t_row), kwargs
            for key, value in t_row.items():
                assert row[key] == pytest.approx(value, nan_ok=True, abs=0), key
            assert (out is None) == (t_out is None)
            if out is not None:
                pd.testing.assert_frame_equal(out, t_out, check_like=True)
        pd.testing.assert_frame_equal(
            compare_confirmation(
                frame, lib_cfg, lib_cfg.llm_review.estimated_rates_path
            ),
            thesis.np_evaluation.compare_confirmation(
                frame, t_cfg, lib_cfg.llm_review.estimated_rates_path
            ),
        )
