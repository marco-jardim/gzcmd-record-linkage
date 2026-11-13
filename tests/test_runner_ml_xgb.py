"""``run_v3(p_cal="ml_xgb")``: nested OOF XGBoost + isotonic route, config and CLI."""

from __future__ import annotations

import importlib
import sys
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest
import yaml

from gzcmd_record_linkage import cli, oof
from gzcmd_record_linkage.config import load_config
from gzcmd_record_linkage.policy_engine import LLMError
from gzcmd_record_linkage.runner import RunSummary, build_engine_from_config, run_v3

CSV = Path(__file__).resolve().parents[1] / "data/synthetic/comparador_sintetico.csv"
THESIS_REPO = Path("D:/git/phd-research")
V3 = Path(str(files("gzcmd_record_linkage") / "gzcmd_v3_config.yaml"))
V31 = Path(str(files("gzcmd_record_linkage") / "gzcmd_v31_config.yaml"))
TINY = {
    "n_estimators": 3,
    "max_depth": 2,
    "learning_rate": 0.3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "n_jobs": 1,
    "verbosity": 0,
}


@pytest.fixture
def tiny_xgb(monkeypatch):
    monkeypatch.setattr(oof, "THESIS_XGB_PARAMS", MappingProxyType(TINY))


def _raw() -> pd.DataFrame:
    raw = pd.read_csv(CSV, sep=";", dtype=str, low_memory=False)
    raw.columns = pd.Index([c.split(",")[0] for c in raw.columns])
    return raw


def _config_with(tmp_path: Path, **calibration) -> Path:
    raw = yaml.safe_load(V3.read_text(encoding="utf-8"))
    raw["calibration"].update(calibration)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_ml_xgb_route_end_to_end(tiny_xgb, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out, summary = run_v3(
        input_csv=CSV,
        config_path=V3,
        mode="vigilancia",
        p_cal="ml_xgb",
        split_by="refrec",
    )
    assert summary.p_cal_method == "ml_xgb"
    assert summary.p_cal_params is None
    assert summary.rows == len(out) == len(_raw())
    assert out["p_cal"].between(0, 1).all()
    assert out["p_cal"].nunique() > 2
    assert list(tmp_path.iterdir()) == []  # nothing persisted by default


def test_ml_xgb_matches_nested_oof_with_thesis_split(tiny_xgb):
    out, _ = run_v3(
        input_csv=CSV,
        config_path=V3,
        mode="vigilancia",
        p_cal="ml_xgb",
        split_by="refrec",
    )
    raw = _raw()
    x = oof.numeric_features(raw)
    y = raw["PAR"].astype(float).isin([1, 2]).astype(int).to_numpy()
    expected, folds = oof.nested_oof_probabilities(
        x, y, split_by="refrec", groups=raw["REFREC"].str.strip().to_numpy(dtype=str)
    )
    by_pair = pd.Series(expected, index=oof.pair_ids(raw))
    np.testing.assert_allclose(
        out["p_cal"].to_numpy(), by_pair.loc[oof.pair_ids(out)].to_numpy()
    )
    table = oof.fit_xgb_oof(CSV, split_by="refrec")
    np.testing.assert_array_equal(table["fold"].to_numpy(), folds)


def test_split_by_defaults_to_yaml_evaluation_split(tiny_xgb, monkeypatch):
    seen = []
    real = oof.fit_xgb_oof
    import gzcmd_record_linkage.runner as runner

    def spy(*args, **kwargs):
        seen.append(kwargs["split_by"])
        return real(*args, **kwargs)

    monkeypatch.setattr(runner, "fit_xgb_oof", spy)
    run_v3(input_csv=CSV, config_path=V3, mode="vigilancia", p_cal="ml_xgb")
    run_v3(
        input_csv=CSV,
        config_path=V3,
        mode="vigilancia",
        p_cal="ml_xgb",
        split_by="comprec",
    )
    assert load_config(V3).evaluation.split_by == "row"
    assert seen == ["row", "comprec"]


def test_cache_reuse_and_mismatch(tiny_xgb, tmp_path, monkeypatch):
    path = tmp_path / "cache" / "oof.csv"
    first = oof.fit_xgb_oof(CSV, output_path=path, split_by="refrec")
    assert path.exists() and path.with_suffix(".json").exists()

    calls = []
    real = oof.nested_oof_probabilities

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(oof, "nested_oof_probabilities", counting)
    again = oof.fit_xgb_oof(CSV, output_path=path, reuse=True, split_by="refrec")
    assert calls == []
    pd.testing.assert_frame_equal(again, pd.read_csv(path, dtype={"pair_id": str}))
    np.testing.assert_allclose(again["p_cal_ml"], first["p_cal_ml"])

    # Mismatching split_by, parameters, input bytes or tampered table: recompute.
    oof.fit_xgb_oof(CSV, output_path=path, reuse=True, split_by="row")
    assert len(calls) == 1
    oof.fit_xgb_oof(
        CSV,
        output_path=path,
        reuse=True,
        split_by="row",
        xgb_params={"n_estimators": 4},
    )
    assert len(calls) == 2
    other = tmp_path / "other.csv"
    other.write_bytes(CSV.read_bytes() + b"\n")
    oof.fit_xgb_oof(
        other,
        output_path=path,
        reuse=True,
        split_by="row",
        xgb_params={"n_estimators": 4},
    )
    assert len(calls) == 3  # one extra byte in the input invalidates the cache
    fresh = oof.fit_xgb_oof(CSV, output_path=path, reuse=True, split_by="refrec")
    assert len(calls) == 4  # the table now describes another input/params
    fitted = oof.fit_xgb_oof(CSV, output_path=path, reuse=True, split_by="refrec")
    assert len(calls) == 4  # and is valid again after the rewrite
    path.write_text(path.read_text(encoding="utf-8") + "x,0,0,0,0\n", encoding="utf-8")
    oof.fit_xgb_oof(CSV, output_path=path, reuse=True, split_by="refrec")
    assert len(calls) == 5  # tampered table: hash mismatch
    np.testing.assert_allclose(fresh["p_cal_ml"], first["p_cal_ml"])
    np.testing.assert_allclose(fitted["p_cal_ml"], first["p_cal_ml"])


def test_reuse_without_path_is_an_error():
    with pytest.raises(ValueError, match="reuse requires output_path"):
        oof.fit_xgb_oof(CSV, reuse=True)


def test_runner_reuses_persisted_table(tiny_xgb, tmp_path, monkeypatch):
    path = tmp_path / "oof.csv"
    kwargs = dict(
        input_csv=CSV,
        config_path=V3,
        mode="vigilancia",
        p_cal="ml_xgb",
        split_by="refrec",
        ml_oof_path=path,
    )
    first, _ = run_v3(**kwargs)

    def boom(*args, **kwargs):
        raise AssertionError("refit despite a valid cache")

    monkeypatch.setattr(oof, "nested_oof_probabilities", boom)
    second, _ = run_v3(**kwargs, reuse_ml_oof=True)
    np.testing.assert_allclose(first["p_cal"], second["p_cal"])


def test_runner_rejects_incomplete_oof_table(tiny_xgb, monkeypatch):
    import gzcmd_record_linkage.runner as runner

    def short(*args, **kwargs):
        table = oof.fit_xgb_oof(*args, **kwargs)
        return table.iloc[:-5]

    monkeypatch.setattr(runner, "fit_xgb_oof", short)
    with pytest.raises(ValueError, match="does not cover every pair_id"):
        run_v3(input_csv=CSV, config_path=V3, mode="vigilancia", p_cal="ml_xgb")


def test_config_p_cal_takes_precedence_over_method(tmp_path):
    assert load_config(V3).calibration.p_cal is None
    assert load_config(V31).calibration.p_cal == "ml_xgb"
    assert load_config(V31).calibration.method == "anchor_platt"

    plain = _config_with(tmp_path, method="platt")
    _, summary = run_v3(
        input_csv=CSV, config_path=plain, mode="vigilancia", p_cal="config"
    )
    assert summary.p_cal_method == "platt"

    both = _config_with(tmp_path, method="platt", p_cal="stub")
    _, summary = run_v3(
        input_csv=CSV, config_path=both, mode="vigilancia", p_cal="config"
    )
    assert summary.p_cal_method == "stub"

    # An explicit p_cal argument still wins over the YAML.
    _, summary = run_v3(
        input_csv=CSV, config_path=both, mode="vigilancia", p_cal="platt"
    )
    assert summary.p_cal_method == "platt"


def test_config_p_cal_ml_xgb_route(tiny_xgb, tmp_path):
    cfg = _config_with(tmp_path, method="platt", p_cal="ml_xgb")
    out, summary = run_v3(
        input_csv=CSV,
        config_path=cfg,
        mode="vigilancia",
        p_cal="config",
        split_by="refrec",
    )
    assert summary.p_cal_method == "ml_xgb"
    assert out["p_cal"].between(0, 1).all()


def test_config_p_cal_is_validated(tmp_path):
    with pytest.raises(ValueError, match="calibration.p_cal must be one of"):
        load_config(_config_with(tmp_path, p_cal="config"))


def test_rates_yaml_overrides_estimated_rates_path(monkeypatch):
    import gzcmd_record_linkage.config as config_module

    cfg = load_config(V31)
    assert cfg.llm_review.estimated_rates_path is not None
    sources = []

    def spy(fixed, *, mode, source=None, model=None):
        sources.append(source)
        return {band: LLMError(e_fp=0.1, e_fn=0.2, e_unsure=0.0) for band in fixed}

    monkeypatch.setattr(config_module, "load_band_rates", spy)
    build_engine_from_config(cfg, mode="vigilancia", policy_version="v3.1")
    build_engine_from_config(
        cfg, mode="vigilancia", policy_version="v3.1", rates_yaml="measured.yaml"
    )
    assert sources == [cfg.llm_review.estimated_rates_path, "measured.yaml"]
    # v3 never opens a rates file, with or without the override.
    build_engine_from_config(
        cfg, mode="vigilancia", policy_version="v3", rates_yaml="measured.yaml"
    )
    assert len(sources) == 2


def test_cli_parses_new_options():
    parser = cli.build_parser()
    base = ["run", "--input", "a.csv", "--config", "c.yaml", "--mode", "vigilancia"]
    default = parser.parse_args(base)
    assert default.p_cal == "fit_platt"
    assert default.ml_oof_path == ""
    assert default.reuse_ml_oof is False
    assert default.rates_yaml == ""
    for choice in (
        "ml_xgb",
        "platt",
        "anchor_platt",
        "config",
        "fit_ml_rf",
        "load_ml_rf",
        "fit_ml_xgb",
        "load_ml_xgb",
        "stub",
        "load_platt",
    ):
        assert parser.parse_args([*base, "--p-cal", choice]).p_cal == choice
    args = parser.parse_args(
        [
            *base,
            "--p-cal",
            "ml_xgb",
            "--ml-oof-path",
            "o.csv",
            "--reuse-ml-oof",
            "--rates-yaml",
            "r.yaml",
        ]
    )
    assert (args.ml_oof_path, args.reuse_ml_oof, args.rates_yaml) == (
        "o.csv",
        True,
        "r.yaml",
    )
    with pytest.raises(SystemExit):
        parser.parse_args([*base, "--p-cal", "nope"])


def test_cli_forwards_options_to_run_v3(monkeypatch):
    captured = {}

    def fake(**kwargs):
        captured.update(kwargs)
        summary = RunSummary(
            rows=0,
            llm_used=0,
            llm_max=0,
            actions={},
            guardrails={},
            review_requested=0,
            p_cal_method=kwargs["p_cal"],
            p_cal_params=None,
        )
        return pd.DataFrame(), summary

    monkeypatch.setattr(cli, "run_v3", fake)
    base = ["run", "--input", "a.csv", "--config", "c.yaml", "--mode", "vigilancia"]
    assert (
        cli.main(
            [
                *base,
                "--p-cal",
                "ml_xgb",
                "--ml-oof-path",
                "o.csv",
                "--reuse-ml-oof",
                "--rates-yaml",
                "r.yaml",
            ]
        )
        == 0
    )
    assert captured["p_cal"] == "ml_xgb"
    assert captured["ml_oof_path"] == "o.csv"
    assert captured["reuse_ml_oof"] is True
    assert captured["rates_yaml"] == "r.yaml"
    assert cli.main(base) == 0
    assert captured["p_cal"] == "fit_platt"
    assert captured["ml_oof_path"] is None
    assert captured["reuse_ml_oof"] is False
    assert captured["rates_yaml"] is None


@pytest.mark.skipif(
    not (THESIS_REPO / "gzcmd" / "ml_oof.py").exists(),
    reason="thesis repository (D:/git/phd-research) not available",
)
def test_parity_with_thesis_fit_xgb_oof(tmp_path):
    raw = pd.read_csv(CSV, sep=";", dtype=str, low_memory=False)
    sample = raw.sample(n=240, random_state=0).sort_index()
    subset = tmp_path / "subset.csv"
    sample.to_csv(subset, sep=";", index=False)

    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(THESIS_REPO))
    try:
        thesis = importlib.import_module("gzcmd.ml_oof")
    except ImportError as exc:  # pragma: no cover - depends on thesis environment
        pytest.skip(f"thesis gzcmd.ml_oof not importable: {exc}")
    finally:
        sys.path.remove(str(THESIS_REPO))
        sys.dont_write_bytecode = previous

    for split_by in ("row", "refrec"):
        theirs = thesis.fit_xgb_oof(
            subset, output_path=tmp_path / f"thesis_{split_by}.csv", split_by=split_by
        )
        ours = oof.fit_xgb_oof(
            subset, output_path=tmp_path / f"lib_{split_by}.csv", split_by=split_by
        )
        assert list(ours["pair_id"]) == list(theirs["pair_id"])
        np.testing.assert_array_equal(ours["fold"], theirs["fold"])
        np.testing.assert_allclose(ours["p_cal_ml"], theirs["p_cal_ml"], atol=1e-12)
