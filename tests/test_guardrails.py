from __future__ import annotations

import pandas as pd

from gzcmd_record_linkage.guardrails import apply_guardrails


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
