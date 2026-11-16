from __future__ import annotations

import json

import pandas as pd
import pytest

from gzcmd_record_linkage.llm_dossier import DossierView, build_dossier
from gzcmd_record_linkage.llm_review import build_review_messages


def _dossier():
    row = pd.Series(
        {
            "COMPREC": 1,
            "REFREC": 2,
            "PASSO": 3,
            "nota final": 7.5,
            "NOME qtd frag iguais": 0.5,
        }
    )
    return build_dossier(row)


def test_remove_nota_final_keeps_other_features() -> None:
    dossier = _dossier()
    view = DossierView(dossier, remove_fields=["features.nota_final"])
    payload = json.loads(view.to_json())
    expected = dossier.to_dict()
    del expected["features"]["nota_final"]
    assert "nota_final" not in payload["features"]
    assert payload == expected
    assert view.pair_id == dossier.pair_id
    assert "nota_final" in dossier.to_dict()["features"]


def test_combines_with_remove_model_outputs() -> None:
    dossier = _dossier()
    view = DossierView(dossier, True, remove_fields=("features.nota_final",))
    payload = view.to_dict()
    assert "model_outputs" not in payload
    assert "nota_final" not in payload["features"]


def test_remove_model_output_subfield() -> None:
    payload = DossierView(_dossier(), remove_fields=["model_outputs.p_cal"]).to_dict()
    assert "p_cal" not in payload["model_outputs"]
    assert "band" in payload["model_outputs"]


def test_default_unchanged() -> None:
    dossier = _dossier()
    assert DossierView(dossier).to_dict() == dossier.to_dict()
    assert DossierView(dossier, remove_fields=()).to_dict() == dossier.to_dict()


@pytest.mark.parametrize("path", ["", ".", "features.", ".features", "a..b"])
def test_malformed_path_raises_value_error(path: str) -> None:
    with pytest.raises(ValueError):
        DossierView(_dossier(), remove_fields=[path])


@pytest.mark.parametrize(
    "path", ["features.nota_finall", "nope", "features.name.nope.x", "features.nope.x"]
)
def test_missing_path_raises_key_error_with_path(path: str) -> None:
    with pytest.raises(KeyError, match="nope|nota_finall"):
        DossierView(_dossier(), remove_fields=[path])


def test_build_review_messages_works_with_view() -> None:
    view = DossierView(_dossier(), remove_fields=["features.nota_final"])
    messages = build_review_messages(view, prompt_version="v4")
    assert messages
    body = "\n".join(m["content"] for m in messages)
    assert view.pair_id in body
    assert view.to_json(indent=2) in body
    assert '"nota_final"' not in body


def test_dossier_source_columns_are_exactly_what_build_dossier_reads() -> None:
    from gzcmd_record_linkage.llm_dossier import DOSSIER_SOURCE_COLUMNS

    assert len(DOSSIER_SOURCE_COLUMNS) == len(set(DOSSIER_SOURCE_COLUMNS))
    keys = {"COMPREC": 1, "REFREC": 2, "PASSO": 3}
    for column in DOSSIER_SOURCE_COLUMNS:
        low = pd.Series({**keys, column: 0.0})
        high = pd.Series({**keys, column: 1.0})
        a = build_dossier(low, pair_id="p").to_json()
        b = build_dossier(high, pair_id="p").to_json()
        assert a != b, f"{column} does not change the dossier"
    # the suffix variant is resolved like the canonical name
    named = pd.Series({**keys, "NOME prim frag igual,Float": 1.0})
    assert '"NOME prim frag igual": 1.0' in build_dossier(named, pair_id="p").to_json()
    for expected in ("nota final", "nome_score_total", "endereco_zero", "p_cal"):
        assert expected in DOSSIER_SOURCE_COLUMNS
