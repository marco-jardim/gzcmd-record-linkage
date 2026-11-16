from __future__ import annotations

import pandas as pd
import pytest
from test_loader import _base_df

from gzcmd_record_linkage.loader import LoadConfig, feature_engineer

COMPOSITES = [
    "nome_score_total",
    "mae_score_total",
    "mae_missing",
    "dtnasc_score_total",
    "endereco_score_total",
    "municipio_score",
    "dtnasc_all_zero",
    "endereco_zero",
]


def test_require_par_false_without_par_creates_no_label_columns() -> None:
    df = _base_df().drop(columns=["PAR"])
    out = feature_engineer(df, cfg=LoadConfig(require_par=False))
    assert "PAR" not in out.columns
    assert "TARGET" not in out.columns
    for col in COMPOSITES:
        assert col in out.columns


def test_default_without_par_still_raises_key_error() -> None:
    df = _base_df().drop(columns=["PAR"])
    with pytest.raises(KeyError, match="PAR"):
        feature_engineer(df, cfg=LoadConfig())


def test_require_par_false_with_par_matches_default() -> None:
    expected = feature_engineer(_base_df(), cfg=LoadConfig())
    actual = feature_engineer(_base_df(), cfg=LoadConfig(require_par=False))
    assert "TARGET" in actual.columns
    pd.testing.assert_frame_equal(actual, expected)
