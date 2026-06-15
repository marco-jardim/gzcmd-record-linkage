from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import pandas as pd
import pytest
from pandas.api.types import is_datetime64_any_dtype, is_string_dtype

from gzcmd_record_linkage.bands import BandAssigner
from gzcmd_record_linkage.calibration import (
    PlattModel,
    compute_p_cal,
    fit_platt_from_df,
)
from gzcmd_record_linkage.config import GZCMDConfig, load_config
from gzcmd_record_linkage.guardrails import apply_guardrails
from gzcmd_record_linkage.loader import LoadConfig, load_comparador_csv
from gzcmd_record_linkage.runner import build_engine_from_config

RAW_COLUMNS = [
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

ENGINEERED_COLUMNS = {
    "TARGET",
    "nota_final",
    "diff_ano",
    "dtnasc_all_zero",
    "nome_score_total",
    "mae_score_total",
    "mae_missing",
    "dtnasc_score_total",
    "endereco_score_total",
    "endereco_zero",
    "municipio_score",
    "score_regras",
}

EXPECTED_MACD_COLUMNS = {
    "macd_nasc_close",
    "macd_nasc_day_match",
    "macd_nasc_diff_capped",
    "macd_nasc_month_match",
    "macd_nasc_partial_overlap",
    "macd_nasc_very_close",
    "macd_nasc_year_match",
    "macd_nome_perf_x_date_far",
    "macd_nome_perf_x_year_diff",
}

DATETIME_COLUMNS = {
    "R_DTNASC_dt",
    "C_DTNASC_dt",
    "R_DTOBITO_dt",
    "C_DTDIAG_dt",
}

SCORE_COLUMNS = {
    "nome_score_total",
    "mae_score_total",
    "dtnasc_score_total",
    "endereco_score_total",
    "municipio_score",
}


def _config_path() -> Path:
    return Path(str(files("gzcmd_record_linkage") / "gzcmd_v3_config.yaml"))


@pytest.fixture
def cfg() -> GZCMDConfig:
    return load_config(_config_path())


def _raw_input_df() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    specs = [
        # ALWAYS_MATCH guardrail row.
        (
            "000000000001",
            "100000000001",
            1,
            1,
            10.0,
            "19800115",
            "19800115",
            "00000000",
            "20200101",
            1.0,
            1,
            1,
            1,
            1.0,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            1,
            1.0,
            1,
            1.0,
            1.0,
            1.0,
            1,
        ),
        (
            "000000000002",
            "100000000002",
            1,
            2,
            9.2,
            "19750620",
            "19750620",
            "00000000",
            "20200315",
            0.9,
            1,
            1,
            0,
            0.9,
            1,
            1,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            0.8,
            1,
            0.7,
            0.8,
            0.8,
            1,
        ),
        (
            "000000000003",
            "100000000003",
            2,
            1,
            8.4,
            "19991231",
            "19991230",
            "00000000",
            "20200630",
            0.8,
            1,
            0,
            0,
            0.8,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            0.7,
            1,
            0.6,
            0.7,
            0.7,
            1,
        ),
        (
            "000000000004",
            "100000000004",
            2,
            2,
            7.6,
            "19640101",
            "19640110",
            "00000000",
            "20200520",
            0.8,
            1,
            1,
            0,
            0.7,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            0.7,
            0,
            0.5,
            0.7,
            0.6,
            1,
        ),
        (
            "000000000005",
            "100000000005",
            3,
            1,
            6.8,
            "20000229",
            "20000229",
            "00000000",
            "20201212",
            0.7,
            1,
            0,
            0,
            0.7,
            1,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0.6,
            1,
            0.5,
            0.6,
            0.6,
            1,
        ),
        (
            "000000000006",
            "100000000006",
            3,
            2,
            5.9,
            "19880704",
            "19880704",
            "00000000",
            "20201111",
            0.7,
            0,
            1,
            0,
            0.6,
            0,
            1,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            0.6,
            1,
            0.4,
            0.5,
            0.6,
            1,
        ),
        (
            "000000000007",
            "100000000007",
            4,
            0,
            5.1,
            "19700101",
            "19710101",
            "00000000",
            "20200404",
            0.6,
            0,
            1,
            0,
            0.5,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            1,
            0.4,
            0,
            0.3,
            0.4,
            0.4,
            0,
        ),
        # Temporal guardrail case: R_DTOBITO DDMMYYYY, C_DTDIAG YYYYMMDD.
        (
            "000000000008",
            "100000000008",
            4,
            0,
            4.3,
            "19550505",
            "19570505",
            "01012020",
            "20210101",
            0.4,
            0,
            0,
            0,
            0.4,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0.3,
            0,
            0.2,
            0.3,
            0.3,
            0,
        ),
        (
            "000000000009",
            "100000000009",
            5,
            0,
            3.4,
            "19900115",
            "19920115",
            "00000000",
            "20200808",
            0.3,
            0,
            0,
            0,
            0.3,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0.2,
            0,
            0.1,
            0.2,
            0.2,
            0,
        ),
        # nota_final_low guardrail row.
        (
            "000000000010",
            "100000000010",
            5,
            0,
            2.7,
            "19851224",
            "19861224",
            "00000000",
            "20200909",
            0.2,
            0,
            0,
            0,
            0.2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0.1,
            0,
            0.0,
            0.1,
            0.1,
            0,
        ),
        (
            "000000000011",
            "100000000011",
            6,
            0,
            1.8,
            "19771130",
            "19791130",
            "00000000",
            "20201010",
            0.1,
            0,
            0,
            0,
            0.1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0.0,
            0,
            0.0,
            0.0,
            0.0,
            0,
        ),
        (
            "000000000012",
            "100000000012",
            6,
            0,
            0.9,
            "19660606",
            "19680606",
            "00000000",
            "20200202",
            0.0,
            0,
            0,
            0,
            0.0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0.0,
            0,
            0.0,
            0.0,
            0.0,
            0,
        ),
    ]
    for spec in specs:
        rows.append(dict(zip(RAW_COLUMNS, spec, strict=True)))
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def _write_raw_csv(tmp_path: Path) -> Path:
    path = tmp_path / "comparador.csv"
    _raw_input_df().to_csv(path, sep=";", decimal=",", encoding="utf-8", index=False)
    return path


def _load_fixture(tmp_path: Path, *, macd_enabled: bool = True) -> pd.DataFrame:
    return load_comparador_csv(
        _write_raw_csv(tmp_path), cfg=LoadConfig(macd_enabled=macd_enabled)
    )


def test_loader_produces_engineered_columns(tmp_path: Path) -> None:
    df = _load_fixture(tmp_path, macd_enabled=True)

    expected_columns = ENGINEERED_COLUMNS | EXPECTED_MACD_COLUMNS | DATETIME_COLUMNS
    assert expected_columns <= set(df.columns)
    actual_macd_columns = set(df.filter(regex=r"^macd_").columns)
    assert actual_macd_columns == EXPECTED_MACD_COLUMNS, "\n" + "\n".join(
        sorted(actual_macd_columns)
    )
    assert set(df["TARGET"].dropna().unique()) == {0, 1}

    for column in SCORE_COLUMNS:
        assert df[column].between(0, 1).all(), column

    for column in DATETIME_COLUMNS:
        assert is_datetime64_any_dtype(df[column]), column


def test_macd_toggle(tmp_path: Path) -> None:
    without_macd = _load_fixture(tmp_path, macd_enabled=False)
    with_macd = _load_fixture(tmp_path, macd_enabled=True)

    assert not any(column.startswith("macd_") for column in without_macd.columns)
    actual_macd_columns = set(with_macd.filter(regex=r"^macd_").columns)
    assert actual_macd_columns == EXPECTED_MACD_COLUMNS, "\n" + "\n".join(
        sorted(actual_macd_columns)
    )


def test_band_assigner_contract(tmp_path: Path, cfg: GZCMDConfig) -> None:
    df = _load_fixture(tmp_path)
    before_columns = list(df.columns)

    bands = BandAssigner.from_config(cfg).assign(df["nota_final"])

    assert is_string_dtype(bands.dtype)
    assert list(df.columns) == before_columns
    expected_bands = {definition.name for definition in cfg.bands.definitions}
    actual_bands = set(bands.dropna().unique())
    assert actual_bands <= expected_bands


def test_calibration_contract(tmp_path: Path, cfg: GZCMDConfig) -> None:
    df = _load_fixture(tmp_path)
    before_columns = list(df.columns)
    clip_min = float(cfg.calibration.clip_min)
    clip_max = float(cfg.calibration.clip_max)

    stub = compute_p_cal(df, method="stub", clip_min=clip_min, clip_max=clip_max)
    assert isinstance(stub, pd.Series)
    assert stub.between(clip_min, clip_max).all()
    assert stub.min() >= 0 and stub.max() <= 1
    assert list(df.columns) == before_columns

    model = fit_platt_from_df(
        df,
        l2=cfg.calibration.platt.l2,
        max_iter=cfg.calibration.platt.max_iter,
        tol=cfg.calibration.platt.tol,
    )
    assert isinstance(model, PlattModel)
    assert isinstance(model.intercept, float)
    assert isinstance(model.slope, float)

    platt = compute_p_cal(
        df, method="platt", model=model, clip_min=clip_min, clip_max=clip_max
    )
    assert isinstance(platt, pd.Series)
    assert platt.between(0, 1).all()
    assert list(df.columns) == before_columns


def test_guardrails_contract(tmp_path: Path) -> None:
    df = _load_fixture(tmp_path)

    output = apply_guardrails(df)

    assert output.guardrail.index.equals(df.index)
    assert output.reason.index.equals(df.index)
    expected_guardrails = {"ALWAYS_MATCH", "ALWAYS_NONMATCH", "FORCE_REVIEW"}
    assert set(output.guardrail.dropna().unique()) <= expected_guardrails


def test_triage_contract(tmp_path: Path, cfg: GZCMDConfig) -> None:
    df = _load_fixture(tmp_path)
    model = fit_platt_from_df(
        df,
        l2=cfg.calibration.platt.l2,
        max_iter=cfg.calibration.platt.max_iter,
        tol=cfg.calibration.platt.tol,
    )
    df["p_cal"] = compute_p_cal(
        df,
        method="platt",
        model=model,
        clip_min=cfg.calibration.clip_min,
        clip_max=cfg.calibration.clip_max,
    )
    df["band"] = BandAssigner.from_config(cfg).assign(df["nota_final"])
    guardrails = apply_guardrails(df)
    df["guardrail"] = guardrails.guardrail

    engine = build_engine_from_config(cfg, mode="vigilancia")
    triaged = engine.triage(df)

    expected_columns = {
        "base_choice",
        "base_loss",
        "loss_llm",
        "evr",
        "action",
        "review_requested",
    }
    assert expected_columns <= set(triaged.columns)
    assert set(triaged["action"].dropna().unique()) <= {
        "MATCH",
        "NONMATCH",
        "LLM_REVIEW",
    }
    assert "action" not in df.columns


def test_pcal_methods_in_unit_interval(tmp_path: Path, cfg: GZCMDConfig) -> None:
    df = _load_fixture(tmp_path)
    clip_min = float(cfg.calibration.clip_min)
    clip_max = float(cfg.calibration.clip_max)
    model = fit_platt_from_df(
        df,
        l2=cfg.calibration.platt.l2,
        max_iter=cfg.calibration.platt.max_iter,
        tol=cfg.calibration.platt.tol,
    )

    stub = compute_p_cal(df, method="stub", clip_min=clip_min, clip_max=clip_max)
    platt = compute_p_cal(
        df, method="platt", model=model, clip_min=clip_min, clip_max=clip_max
    )

    assert stub.min() >= 0
    assert stub.max() <= 1
    assert platt.min() >= 0
    assert platt.max() <= 1
