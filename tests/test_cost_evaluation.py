"""Cost-side expected funnel metrics: hand-computed values and thesis parity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from _thesis_eval import load_thesis_eval, requires_thesis_eval

from gzcmd_record_linkage.cost_evaluation import expected_funnel_metrics
from gzcmd_record_linkage.policy_engine import HumanError, LLMError

RATES = {"a": LLMError(0.1, 0.2, 0.1)}
HUMAN = HumanError(0.05, 0.1)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "nota final": [9.5, 9.1, 6.0, 2.0, 6.0, 7.0, 6.0, 5.5],
            "p_cal": [0.99, 0.9, 0.2, 0.01, 0.5, 0.4, 0.5, 0.3],
            "band": ["a"] * 8,
            "TARGET": [1, 0, 1, 0, 1, 0, 1, 0],
            "guardrail": [
                "ALWAYS_MATCH",
                pd.NA,
                pd.NA,
                "ALWAYS_NONMATCH",
                pd.NA,
                pd.NA,
                pd.NA,
                pd.NA,
            ],
            "action": [
                "MATCH",
                "MATCH",
                "NONMATCH",
                "NONMATCH",
                "LLM_REVIEW",
                "LLM_REVIEW",
                "DEFERRED",
                "DEFERRED",
            ],
            "selected_loss": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
    )


def test_hand_computed_expectations() -> None:
    m = expected_funnel_metrics(_frame(), RATES, HUMAN)
    # Counts by action / guard (compulsory guard rows are not "automatic").
    assert (m["lote_llm"], m["humano_direto"]) == (2, 2)
    assert (m["AUTO_MATCH"], m["AUTO_NONMATCH"]) == (1, 1)
    assert (m["ALWAYS_MATCH"], m["ALWAYS_NONMATCH"]) == (1, 1)
    assert (m["AUTO_MATCH_vinculos"], m["AUTO_MATCH_FP"]) == (0, 1)
    assert m["AUTO_NONMATCH_vinculos"] == 1
    assert m["vinculos_perdidos_auto_nonmatch"] == 1
    assert m["vinculos_perdidos_cinza"] == 1
    assert m["vinculos_perdidos_total_nonmatch"] == 1
    assert (m["FP_automaticos"], m["FP_auto_match"]) == (1, 1)
    # Expected quantities (u = .1 per LLM row, human errors .05/.1).
    assert m["humano_via_llm_esperado"] == pytest.approx(0.2)
    assert m["humano_total_esperado"] == pytest.approx(2.2)
    assert m["fp_esperado_auto_match_incl_guardas"] == pytest.approx(0.11)
    assert m["taxa_fp_esperada_auto_match"] == pytest.approx(0.055)
    assert m["vinculos_esperados_auto_match_incl_guardas"] == pytest.approx(1.89)
    assert m["fp_esperado_total"] == pytest.approx(0.11 + 0.11 + 0.06 + 0.0055)
    assert m["sens_esperada_automatica"] == pytest.approx(1.7 / 4)
    assert m["sens_esperada_com_humano"] == pytest.approx((1.7 + 0.9 + 0.09) / 4)
    assert m["perda_esperada_total"] == pytest.approx(36.0)


def test_alternate_column_names_and_empty_denominators() -> None:
    frame = _frame().rename(columns={"TARGET": "y", "nota final": "nota"})
    assert expected_funnel_metrics(frame, RATES, HUMAN) == pytest.approx(
        expected_funnel_metrics(_frame(), RATES, HUMAN)
    )
    no_match = _frame().assign(action="NONMATCH")
    assert expected_funnel_metrics(no_match, RATES, HUMAN)[
        "taxa_fp_esperada_auto_match"
    ] == pytest.approx(0.0)
    no_positive = _frame().assign(TARGET=0)
    m = expected_funnel_metrics(no_positive, RATES, HUMAN)
    assert np.isnan(m["sens_esperada_automatica"])
    assert np.isnan(m["sens_esperada_com_humano"])


def test_missing_reliability_band_is_an_error() -> None:
    with pytest.raises(ValueError, match="Missing reliability band"):
        expected_funnel_metrics(_frame().assign(band="zzz"), RATES, HUMAN)


def test_matches_thesis_sweep_case() -> None:
    frame = pd.DataFrame(
        {
            "nota final": [6.0] * 2,
            "p_cal": [0.5, 0.5],
            "band": ["a", "a"],
            "TARGET": [1, 1],
            "guardrail": [pd.NA, pd.NA],
            "action": ["LLM_REVIEW", "DEFERRED"],
            "selected_loss": [2.0, 3.0],
        }
    )
    m = expected_funnel_metrics(frame, RATES, HumanError(0.05, 0.05))
    assert m["humano_direto"] == 1
    assert m["humano_total_esperado"] == pytest.approx(1.1)
    assert m["sens_esperada_automatica"] == pytest.approx(0.35)
    assert m["sens_esperada_com_humano"] == pytest.approx((0.7 + 0.1 * 0.95 + 0.95) / 2)
    assert m["perda_esperada_total"] == 5


@requires_thesis_eval
def test_thesis_parity_on_synthetic_frame() -> None:
    thesis = load_thesis_eval()
    t_rates = {"a": thesis.policy_engine.LLMError(0.1, 0.2, 0.1)}
    t_human = thesis.policy_engine.HumanError(0.05, 0.1)
    expected = thesis.cost_evaluation.expected_funnel_metrics(
        _frame(), t_rates, t_human
    )
    actual = expected_funnel_metrics(_frame(), RATES, HUMAN)
    assert list(actual) == list(expected)
    assert actual == pytest.approx(expected, rel=0, abs=0)
