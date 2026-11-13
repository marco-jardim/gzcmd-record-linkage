"""``numeric_features`` and ``pair_ids``: thesis comparator allowlist and pair ids."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest
from _thesis_arranjo import load_asr, load_raw, requires_thesis_asr

from gzcmd_record_linkage.oof import numeric_features, pair_ids


def raw_frame():
    return pd.DataFrame(
        {
            "COMPREC": [" 1", "2 ", "3"],
            "REFREC": ["10", "10", "11"],
            "PASSO": ["6", "9", "6"],
            "NOME A": ["0,5", "1", "x"],
            "NOMEMAE B": ["2,25", None, "3"],
            "DTNASC C": ["1", "2", "3"],
            "CODMUNRES D": ["0", "1", "1,5"],
            "ENDERECO E": ["1", "1", "1"],
            "nota final": ["8,5", "7", "9"],
            "PAR": ["1", "0", "2"],
            "NOMEOUTRO": ["1", "1", "1"],
        }
    )


def test_numeric_features_allowlist_decimal_comma_and_zero_fill():
    x = numeric_features(raw_frame())
    assert list(x.columns) == [
        "NOME A",
        "NOMEMAE B",
        "DTNASC C",
        "CODMUNRES D",
        "ENDERECO E",
        "nota final",
    ]
    assert (x.dtypes == "float64").all()
    assert x["NOME A"].tolist() == [0.5, 1.0, 0.0]  # unparsable -> 0.0
    assert x["NOMEMAE B"].tolist() == [2.25, 0.0, 3.0]  # missing -> 0.0
    assert x["nota final"].tolist() == [8.5, 7.0, 9.0]
    assert not x.isna().any().any()
    alias = numeric_features(pd.DataFrame({"nota_final": ["1,5"], "PAR": ["1"]}))
    assert list(alias.columns) == ["nota_final"] and alias.iloc[0, 0] == 1.5


def test_pair_ids_hash_strip_and_prefixed_headers():
    df = raw_frame()
    ids = pair_ids(df)
    expected = [
        hashlib.sha256(f"{c}|{r}|{p}".encode()).hexdigest()[:16]
        for c, r, p in (("1", "10", "6"), ("2", "10", "9"), ("3", "11", "6"))
    ]
    assert ids.tolist() == expected
    prefixed = df.rename(columns={"COMPREC": "COMPREC,C,12,0", "PASSO": "PASSO,N,2,0"})
    assert pair_ids(prefixed).tolist() == expected
    assert (ids.index == df.index).all()


def test_pair_ids_errors():
    df = raw_frame()
    with pytest.raises(ValueError, match="Missing pair metadata"):
        pair_ids(df.drop(columns="PASSO"))
    with pytest.raises(ValueError, match="pair_id must be unique"):
        pair_ids(pd.concat([df, df.iloc[:1]], ignore_index=True))


@requires_thesis_asr
def test_parity_with_the_thesis_helpers_on_the_real_base():
    thesis = load_asr()["ml_oof"]
    raw = load_raw().iloc[:3000].reset_index(drop=True)
    ours = numeric_features(raw)
    theirs = thesis.numeric_features(raw)
    pd.testing.assert_frame_equal(ours, theirs)
    assert ours.shape[1] == 30
    np.testing.assert_array_equal(
        pair_ids(raw).to_numpy(), thesis.pair_ids(raw).to_numpy()
    )
