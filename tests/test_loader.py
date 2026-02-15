from __future__ import annotations

import pandas as pd

from gzcmd_record_linkage.loader import LoadConfig, feature_engineer


def _base_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "COMPREC,C,12,0": ["A", "B"],
            "REFREC,C,12,0": ["R1", "R2"],
            "PASSO": [1, 1],
            "PAR": [1, 0],
            "nota final": [9.0, 2.5],
            "R_DTNASC,C,8,0": ["19800101", "19700101"],
            "C_DTNASC,C,8,0": ["19800102", "19700101"],
            "R_DTOBITO,C,10,0": ["01012020", "01012020"],
            "C_DTDIAG,C,10,0": ["20200101", "20190101"],
            "NOME qtd frag iguais": [0.95, 0.50],
            "NOME prim frag igual": [1, 0],
            "NOME ult frag igual": [1, 0],
            "NOMEMAE qtd frag iguais": [0.0, 0.0],
            "NOMEMAE prim frag igual": [0.0, 0.0],
            "NOMEMAE ult frag igual": [0.0, 0.0],
            "DTNASC dt iguais": [0.0, 0.0],
            "DTNASC dt ap 1digi": [0.0, 0.0],
            "DTNASC dt inv dia": [0.0, 0.0],
            "DTNASC dt inv mes": [0.0, 0.0],
            "DTNASC dt inv ano": [0.0, 0.0],
            "ENDERECO qtd frag iguais": [0.0, 0.0],
            "ENDERECO prim frag igual": [0.0, 0.0],
            "ENDERECO seg frag igual": [0.0, 0.0],
            "ENDERECO ter frag igual": [0.0, 0.0],
            "ENDERECO quart frag igual": [0.0, 0.0],
            "ENDERECO ult frag igual": [0.0, 0.0],
            "CODMUNRES local igual": [1.0, 0.0],
        }
    )


def test_feature_engineer_resolves_prefixed_columns_and_aliases() -> None:
    out = feature_engineer(_base_df(), cfg=LoadConfig(macd_enabled=False))

    assert "COMPREC" in out.columns
    assert "REFREC" in out.columns
    assert out["COMPREC"].tolist() == ["A", "B"]
    assert out["REFREC"].tolist() == ["R1", "R2"]

    # Cleaned alias is added for 'nota final'
    assert "nota_final" in out.columns
    assert out["nota_final"].tolist() == [9.0, 2.5]


def test_feature_engineer_macd_toggle() -> None:
    out_off = feature_engineer(_base_df(), cfg=LoadConfig(macd_enabled=False))
    assert "macd_nasc_diff_capped" not in out_off.columns

    out_on = feature_engineer(_base_df(), cfg=LoadConfig(macd_enabled=True))
    assert "macd_nasc_diff_capped" in out_on.columns
    assert "macd_nasc_year_match" in out_on.columns
