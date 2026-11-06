"""Historical extract boundary and descriptive-only configuration regression."""
from importlib.resources import files

import pandas as pd
import yaml

from gzcmd_record_linkage.llm_dossier import build_dossier


def test_historical_extract_sends_exactly_fourteen_real_subscores():
    groups = {
        "name": ["NOME qtd frag iguais", "NOME prim frag igual", "NOME ult frag igual"],
        "mother": [
            "NOMEMAE qtd frag iguais", "NOMEMAE prim frag igual",
            "NOMEMAE ult frag igual",
        ],
        "dob": ["DTNASC dt iguais"],
        "address": [
            "ENDERECO via igual", "ENDERECO via prox", "ENDERECO numero igual",
            "ENDERECO compl prox", "ENDERECO texto prox", "ENDERECO tokens jacc",
        ],
    }
    sent = [column for columns in groups.values() for column in columns]
    row = pd.Series({column: (i + 1) / 20 for i, column in enumerate(sent)})
    row["CODMUNRES local igual"] = 0.75
    omitted = [
        "NOME qtd frag raros", "NOME qtd frag comuns", "NOME qtd frag muito parec",
        "NOME qtd frag abrev", "NOMEMAE qtd frag raros", "NOMEMAE qtd frag comuns",
        "NOMEMAE qtd frag muito parec", "NOMEMAE qtd frag abrev", "DTNASC dt ap 1digi",
        "DTNASC dt inv dia", "DTNASC dt inv mes", "DTNASC dt inv ano",
        "CODMUNRES uf igual", "CODMUNRES uf prox", "CODMUNRES local prox",
    ]
    for column in omitted:
        row[column] = 0.9
    features = build_dossier(row).to_dict()["features"]
    assert len(row) == 29
    assert sum(len(features[group]) for group in groups) + 1 == 14
    for group, columns in groups.items():
        assert features[group] == {column: row[column] for column in columns}
    assert features["municipality"] == 0.75


def test_nonexistent_extract_columns_do_not_extend_the_dossier():
    row = pd.Series({"nota final": 7.5})
    baseline = build_dossier(row).to_dict()
    for column in (
        "NOME prim ult frag igual", "NOMEMAE prim ult frag igual",
        "DTNASC ano igual", "DTNASC mes igual", "DTNASC dia igual",
    ):
        row[column] = 1.0
    assert build_dossier(row).to_dict() == baseline


def test_package_generation_defaults_are_preserved_and_conflict_is_reserved():
    resource = files("gzcmd_record_linkage").joinpath("gzcmd_v3_config.yaml")
    text = resource.read_text(encoding="utf-8")
    config = yaml.safe_load(text)
    for reviewer in ("agent_a", "agent_b", "arbiter"):
        model = config["llm_review"]["model"][reviewer]
        assert model["name"] == "gpt-5.2-pro"
        assert model["temperature"] == 0.0
        assert model["max_tokens"] == 500
    assert config["consistency"]["conflict_margin_delta"] == 0.03
    assert "RESERVED, NOT IMPLEMENTED" in text
