from __future__ import annotations

import numpy as np
import pandas as pd
from _thesis_eval import load_thesis_eval, requires_thesis_eval

from gzcmd_record_linkage.guardrails import (
    apply_guardrails,
    high_guard_birth_variant,
    strong_birth_conflict,
)


def test_guardrails_expected_labels_and_reasons() -> None:
    df = pd.DataFrame(
        {
            "nota final": [9.0, 2.0, 7.0, 10.0],
            "R_DTOBITO_dt": [
                pd.Timestamp("2020-01-01"),
                pd.NaT,
                pd.NaT,
                pd.NaT,
            ],
            "C_DTDIAG_dt": [
                pd.Timestamp("2021-01-01"),
                pd.Timestamp("2020-01-01"),
                pd.NaT,
                pd.NaT,
            ],
            "dtnasc_all_zero": [False, False, True, False],
            "diff_ano": [0.0, 0.0, 10.0, 0.0],
            "endereco_zero": [False, False, True, False],
            "NOME qtd frag iguais": [0.95, 0.0, 0.0, 0.95],
            "NOME prim frag igual": [1, 0, 0, 1],
            "NOME ult frag igual": [1, 0, 0, 1],
            "DTNASC dt iguais": [1, 0, 0, 1],
            "CODMUNRES local igual": [1, 0, 0, 1],
        }
    )

    out = apply_guardrails(df)
    assert out.guardrail.tolist() == [
        "ALWAYS_NONMATCH",
        "ALWAYS_NONMATCH",
        "FORCE_REVIEW",
        "ALWAYS_MATCH",
    ]
    assert out.reason.tolist() == [
        "temporal_filter",
        "nota_final_low",
        "homonimia_risk",
        "nota_final_high",
    ]


def test_strong_birth_conflict_requires_known_differing_years() -> None:
    frame = pd.DataFrame({"diff_ano": [0.0, 2.0, np.nan, -1.0, "3", "x"]})
    assert strong_birth_conflict(frame).tolist() == [
        False,
        True,
        False,
        False,
        True,
        False,
    ]
    # Missing column: no evidence of conflict anywhere.
    assert strong_birth_conflict(pd.DataFrame(index=range(3))).tolist() == [False] * 3


def test_birth_variant_is_stricter_subset_and_never_recovers_rejected_p() -> None:
    frame = pd.DataFrame(
        {
            "nota final": [9.0] * 5,
            "p_cal": [0.99, 0.96, 0.98, 0.94, 0.89],
            "diff_ano": [0.0, 2.0, np.nan, 0.0, 0.0],
        }
    )
    variant = high_guard_birth_variant(frame)
    assert variant.tolist() == [True, False, True, False, False]
    current = apply_guardrails(frame, policy_version="v3.1").guardrail.eq(
        "ALWAYS_MATCH"
    )
    assert (variant & ~current).sum() == 0
    assert not (variant & frame.p_cal.lt(0.9)).any()


def test_birth_variant_thresholds_and_nota_alias() -> None:
    frame = pd.DataFrame(
        {
            "nota_final": [9.0, 8.99, 10.0, np.nan],
            "p_cal": [0.95, 0.99, 0.9499, 0.99],
        }
    )
    assert high_guard_birth_variant(frame).tolist() == [True, False, False, False]
    # "nota final" wins where present; the alias only fills its gaps.
    both = frame.assign(**{"nota final": [np.nan, 9.5, 9.5, 9.5]})
    assert high_guard_birth_variant(both).tolist() == [True, True, False, True]


@requires_thesis_eval
def test_birth_guards_match_thesis() -> None:
    guards = load_thesis_eval().guardrails
    rng = np.random.default_rng(7)
    n = 300
    frame = pd.DataFrame(
        {
            "nota final": rng.choice([np.nan, 5.0, 8.99, 9.0, 9.5, 10.0], n),
            "nota_final": rng.choice([np.nan, 9.0, 10.0], n),
            "p_cal": rng.choice([0.5, 0.9, 0.9499, 0.95, 0.99], n),
            "diff_ano": rng.choice([np.nan, -1.0, 0.0, 1.0, 5.0], n),
        }
    )
    pd.testing.assert_series_equal(
        strong_birth_conflict(frame), guards.strong_birth_conflict(frame)
    )
    pd.testing.assert_series_equal(
        high_guard_birth_variant(frame), guards.high_guard_birth_variant(frame)
    )
