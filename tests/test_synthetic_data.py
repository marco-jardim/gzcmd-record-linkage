from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import synthetic_data as sd

from gzcmd_record_linkage.loader import LoadConfig, load_comparador_csv

REQUIRED_COLUMNS = [
    "COMPREC,C,12,0",
    "REFREC,C,12,0",
    "PASSO",
    "PAR",
    "nota final",
    "R_DTNASC,C,8,0",
    "C_DTNASC,C,8,0",
    "R_DTOBITO,C,10,0",
    "C_DTDIAG,C,10,0",
    "NOME qtd frag iguais",
    "NOME prim frag igual",
    "NOME ult frag igual",
    "NOME prim ult frag igual",
    "NOMEMAE qtd frag iguais",
    "NOMEMAE prim frag igual",
    "NOMEMAE ult frag igual",
    "NOMEMAE prim ult frag igual",
    "DTNASC dt iguais",
    "DTNASC dt ap 1digi",
    "DTNASC dt inv dia",
    "DTNASC dt inv mes",
    "DTNASC dt inv ano",
    "ENDERECO via igual",
    "ENDERECO via prox",
    "ENDERECO numero igual",
    "ENDERECO compl prox",
    "ENDERECO texto prox",
    "ENDERECO tokens jacc",
    "CODMUNRES local igual",
]


def _target(frame: pd.DataFrame) -> pd.Series:
    return frame["PAR"].isin([1, 2]).astype(int)


def _load(path, *, macd_enabled: bool):
    config = LoadConfig(macd_enabled=macd_enabled)
    parameters = inspect.signature(load_comparador_csv).parameters
    for name in ("config", "load_config", "cfg"):
        if name in parameters:
            return load_comparador_csv(path, **{name: config})
    return load_comparador_csv(path, macd_enabled=macd_enabled)


def test_schema_raw_columns_dates_and_p_true_alignment() -> None:
    dataset = sd.generate_comparador(seed=123, n_pairs=600, match_ratio=0.5)
    frame = dataset.frame

    assert set(REQUIRED_COLUMNS).issubset(frame.columns)
    assert pd.api.types.is_numeric_dtype(frame["nota final"])
    for column in [
        "R_DTNASC,C,8,0",
        "C_DTNASC,C,8,0",
        "R_DTOBITO,C,10,0",
        "C_DTDIAG,C,10,0",
    ]:
        assert frame[column].map(lambda value: isinstance(value, str)).all()
        assert frame[column].str.len().eq(8).all()
    assert "p_true" not in frame.columns
    assert dataset.p_true.index.equals(frame.index)
    assert dataset.p_true.between(0.0, 1.0, inclusive="neither").all()


def test_generate_comparador_is_deterministic_by_seed() -> None:
    first = sd.generate_comparador(seed=10, n_pairs=600)
    second = sd.generate_comparador(seed=10, n_pairs=600)
    different = sd.generate_comparador(seed=11, n_pairs=600)

    assert first.frame.equals(second.frame)
    assert first.p_true.equals(second.p_true)
    assert not first.frame.equals(different.frame)


def test_distribution_has_target_ratio_classes_and_grey_zone() -> None:
    dataset = sd.generate_comparador(seed=42, n_pairs=600, match_ratio=0.5)
    target = _target(dataset.frame)

    assert abs(target.mean() - 0.5) <= 0.10
    assert set(target.unique()) == {0, 1}
    assert dataset.frame["nota final"].between(5.0, 8.0, inclusive="left").any()


def test_csv_round_trip_loads_with_and_without_macd(tmp_path) -> None:
    dataset = sd.generate_comparador(seed=42, n_pairs=600)
    path = sd.to_comparador_csv(dataset.frame, tmp_path / "comparador.csv")

    loaded_with_macd = _load(path, macd_enabled=True)
    loaded_without_macd = _load(path, macd_enabled=False)

    for loaded in [loaded_with_macd, loaded_without_macd]:
        assert {
            "nome_score_total",
            "dtnasc_score_total",
            "endereco_score_total",
            "TARGET",
        }.issubset(loaded.columns)
        expected_target = _target(dataset.frame).to_numpy()
        np.testing.assert_array_equal(loaded["TARGET"].to_numpy(), expected_target)

    macd_columns_enabled = {
        column for column in loaded_with_macd.columns if "macd" in column.lower()
    }
    macd_columns_disabled = {
        column for column in loaded_without_macd.columns if "macd" in column.lower()
    }
    assert macd_columns_enabled
    assert not macd_columns_disabled


def test_edge_boundaries_and_score_range_are_present() -> None:
    dataset = sd.generate_comparador(seed=42, n_pairs=600)
    notas = set(dataset.frame["nota final"])

    assert {5.0, 6.0, 7.0, 8.0, 9.0}.issubset(notas)
    assert dataset.frame["nota final"].between(0.0, 10.0).all()


def test_nota_final_is_not_perfectly_separable_and_classes_overlap() -> None:
    dataset = sd.generate_comparador(seed=42, n_pairs=600)
    frame = dataset.frame
    target = _target(frame)
    scores = frame["nota final"]
    ranks = scores.rank(method="average")
    n_pos = int(target.sum())
    n_neg = int(len(target) - n_pos)
    sum_ranks_pos = float(ranks[target == 1].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    assert 0.5 < auc < 0.99
    positive_range = scores[target == 1].min(), scores[target == 1].max()
    negative_range = scores[target == 0].min(), scores[target == 0].max()
    overlap_low = max(positive_range[0], negative_range[0])
    overlap_high = min(positive_range[1], negative_range[1])
    assert overlap_low <= overlap_high


def test_true_posterior_bins_are_empirically_monotonic() -> None:
    dataset = sd.generate_comparador(seed=42, n_pairs=600)
    target = _target(dataset.frame)
    binned = pd.DataFrame({"p_true": dataset.p_true, "target": target})
    binned["bin"] = pd.qcut(binned["p_true"], q=10, duplicates="drop")
    empirical = binned.groupby("bin", observed=True)["target"].mean().to_numpy()

    assert np.all(empirical[1:] >= empirical[:-1] - 0.05)


def test_group_aware_split_has_no_group_leakage_and_is_deterministic() -> None:
    dataset = sd.generate_comparador(seed=42, n_pairs=600)
    frame = dataset.frame

    for split_by, column in [
        ("comprec", "COMPREC,C,12,0"),
        ("refrec", "REFREC,C,12,0"),
    ]:
        train_idx, test_idx = sd.group_aware_split_indices(
            frame, split_by=split_by, seed=7
        )
        train_groups = set(frame.loc[train_idx, column])
        test_groups = set(frame.loc[test_idx, column])
        assert train_groups.isdisjoint(test_groups)

        train_idx_again, test_idx_again = sd.group_aware_split_indices(
            frame, split_by=split_by, seed=7
        )
        np.testing.assert_array_equal(train_idx, train_idx_again)
        np.testing.assert_array_equal(test_idx, test_idx_again)


def _scenario_csv(tmp_path):
    dataset = sd.generate_comparador(seed=123, n_pairs=40, scenarios="all")
    csv_path = tmp_path / "comparador_scenarios.csv"
    sd.to_comparador_csv(dataset.frame, csv_path)
    return dataset, csv_path


def _config_path() -> str:
    from importlib.resources import files

    return str(files("gzcmd_record_linkage") / "gzcmd_v3_config.yaml")


def test_tst_1_2_a_scenarios_have_expected_bands(tmp_path):
    from gzcmd_record_linkage.bands import BandAssigner
    from gzcmd_record_linkage.config import load_config
    from gzcmd_record_linkage.loader import LoadConfig, load_comparador_csv

    _, csv_path = _scenario_csv(tmp_path)
    df = load_comparador_csv(csv_path, cfg=LoadConfig())
    bands = BandAssigner.from_config(load_config(_config_path())).assign(
        df["nota final"]
    )

    expected = {
        "match_obvio": "high",
        "nonmatch_obvio": "low",
        "homonimo": "grey_high",
        "obito_antes_diag": "grey_mid",
        "mae_ausente": "grey_mid",
        "datas_invertidas": "grey_mid",
        "zona_cinzenta": "grey_mid",
    }
    for name, band in expected.items():
        row_index = df.index[df["COMPREC"] == f"SCEN-{name}"][0]
        assert bands.loc[row_index] == band


def test_tst_1_2_b_scenarios_trigger_expected_guardrails(tmp_path):
    from gzcmd_record_linkage.guardrails import apply_guardrails
    from gzcmd_record_linkage.loader import LoadConfig, load_comparador_csv

    _, csv_path = _scenario_csv(tmp_path)
    df = load_comparador_csv(csv_path, cfg=LoadConfig())
    guardrails = apply_guardrails(df)
    row_by_comprec = {value: index for index, value in df["COMPREC"].items()}

    expected = {
        "match_obvio": ("ALWAYS_MATCH", "nota_final_high"),
        "nonmatch_obvio": ("ALWAYS_NONMATCH", "nota_final_low"),
        "homonimo": ("FORCE_REVIEW", "homonimia_risk"),
        "obito_antes_diag": ("ALWAYS_NONMATCH", "temporal_filter"),
    }
    for name, (guardrail, reason) in expected.items():
        row_index = row_by_comprec[f"SCEN-{name}"]
        assert guardrails.guardrail.loc[row_index] == guardrail
        assert guardrails.reason.loc[row_index] == reason


def test_tst_1_2_c_zona_cinzenta_routes_to_llm_review(tmp_path):
    from gzcmd_record_linkage.runner import run_v3

    _, csv_path = _scenario_csv(tmp_path)
    out_df, _ = run_v3(
        input_csv=csv_path,
        config_path=_config_path(),
        mode="confirmacao",
    )
    row = out_df.loc[out_df["COMPREC"] == "SCEN-zona_cinzenta"].iloc[0]

    assert row["action"] == "LLM_REVIEW"


def test_tst_1_2_d_scenarios_are_deterministic_and_indexed():
    first = sd.generate_comparador(seed=123, n_pairs=40, scenarios="all")
    second = sd.generate_comparador(seed=123, n_pairs=40, scenarios="all")

    assert first.frame.equals(second.frame)
    assert first.meta["scenarios"] == {
        name: 40 + index for index, name in enumerate(sd.SCENARIO_NAMES)
    }
    assert set(first.meta["scenarios"]) == set(sd.SCENARIO_NAMES)
