from dataclasses import replace
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from gzcmd_record_linkage.calibration import (
    compute_p_cal,
    fit_anchor_platt,
    fit_calibration_from_df,
    fit_platt_from_df,
    predict_platt,
)
from gzcmd_record_linkage.config import AnchorFitParams, CalibrationConfig, load_config
from gzcmd_record_linkage.loader import load_comparador_csv
from gzcmd_record_linkage.runner import run_v3


def _config_path():
    return files("gzcmd_record_linkage") / "gzcmd_v3_config.yaml"


@pytest.fixture
def synthetic_scores():
    rng = np.random.default_rng(42)
    scores = pd.Series(rng.uniform(0, 10, 10_000), name="nota_final")
    probabilities = 1 / (1 + np.exp(-(3 * scores - 21)))
    labels = pd.Series(rng.binomial(1, probabilities), name="TARGET")
    return pd.DataFrame({"nota_final": scores, "TARGET": labels})


def test_anchor_target_means_and_label_independence(synthetic_scores):
    df = synthetic_scores
    scores = df["nota_final"]
    # The known sigmoid gives high-purity anchors; labels never enter the fit.
    assert df.loc[scores >= 9, "TARGET"].mean() > 0.99
    assert df.loc[scores < 5, "TARGET"].mean() < 0.01
    model = fit_anchor_platt(scores)
    probabilities = predict_platt(scores, model=model)
    assert probabilities[scores >= 9].mean() == pytest.approx(0.992, abs=0.02)
    assert probabilities[scores < 5].mean() == pytest.approx(0.002, abs=0.02)
    config = CalibrationConfig(method="anchor_platt")
    assert fit_calibration_from_df(df.drop(columns="TARGET"), config=config) == model
    assert fit_calibration_from_df(
        df.assign(TARGET=1 - df["TARGET"]), config=config
    ) == model


def test_anchor_monotonicity(synthetic_scores):
    model = fit_anchor_platt(synthetic_scores["nota_final"])
    assert model.slope > 0
    scores = pd.Series(np.linspace(0, 10, 1001))
    assert np.all(np.diff(predict_platt(scores, model=model)) >= 0)


@pytest.mark.parametrize("positive_count,negative_count", [(29, 100), (100, 29)])
def test_insufficient_anchors_raise_and_log(positive_count, negative_count, caplog):
    scores = pd.Series([9.0] * positive_count + [4.9] * negative_count)
    with pytest.raises(ValueError, match="Insufficient anchors.*minimum 30"):
        fit_anchor_platt(scores)
    assert "anchor calibration not fitted" in caplog.text


def test_anchor_boundaries_and_minimum_are_inclusive():
    scores = pd.Series([9.0] * 30 + [4.9] * 30 + [5.0] * 100)
    assert fit_anchor_platt(scores).slope > 0
    with pytest.raises(ValueError, match="A-=0"):
        fit_anchor_platt(pd.Series([9.0] * 30 + [5.0] * 30))


def test_explicit_supervised_fallback(synthetic_scores, caplog):
    df = synthetic_scores.iloc[:100]
    config = CalibrationConfig(
        method="anchor_platt", fallback="platt",
        anchor_platt=AnchorFitParams(min_anchor_size=1000),
    )
    assert fit_calibration_from_df(df, config=config) == fit_platt_from_df(df)
    assert "falling back to supervised Platt" in caplog.text
    with pytest.raises(ValueError, match="Insufficient anchors"):
        fit_calibration_from_df(df.drop(columns="TARGET"), config=config)
    with pytest.raises(ValueError, match="Insufficient anchors"):
        fit_calibration_from_df(df, config=replace(config, fallback="error"))


def test_anchor_parameter_bounds(synthetic_scores):
    model = fit_anchor_platt(
        synthetic_scores["nota_final"], alpha_max=0.1, beta_max=0.5
    )
    assert abs(model.slope) <= 0.1
    assert abs(model.intercept) <= 0.5


def test_config_dispatch_and_runner_on_synthetic_csv(tmp_path):
    path = (
        Path(__file__).resolve().parents[1]
        / "data/synthetic/comparador_sintetico.csv"
    )
    config = load_config(_config_path())
    assert config.calibration.method == "anchor_platt"
    assert config.calibration.anchor_platt.r_plus == 0.992
    assert config.calibration.anchor_platt.r_minus == 0.002
    df = load_comparador_csv(path)
    model = fit_calibration_from_df(df, config=config.calibration)
    probabilities = compute_p_cal(df, method="anchor_platt", model=model)
    assert probabilities.notna().all()
    assert probabilities.between(0, 1).all()
    supervised = fit_calibration_from_df(
        df, config=replace(config.calibration, method="platt")
    )
    assert not np.allclose(
        probabilities, predict_platt(df["nota_final"], model=supervised)
    )

    anchored, summary = run_v3(
        input_csv=path, config_path=_config_path(), mode="vigilancia", p_cal="config"
    )
    np.testing.assert_allclose(anchored["p_cal"], probabilities)
    assert summary.p_cal_method == "anchor_platt"
    raw = yaml.safe_load(_config_path().read_text(encoding="utf-8"))
    raw["calibration"]["method"] = "platt"
    platt_config_path = tmp_path / "platt.yaml"
    platt_config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    platt, summary = run_v3(
        input_csv=path, config_path=platt_config_path, mode="vigilancia", p_cal="config"
    )
    assert summary.p_cal_method == "platt"
    assert not np.allclose(anchored["p_cal"], platt["p_cal"])
    legacy, _ = run_v3(
        input_csv=path, config_path=_config_path(), mode="vigilancia", p_cal="fit_platt"
    )
    np.testing.assert_allclose(legacy["p_cal"], platt["p_cal"])


def test_anchor_config_defaults_and_by_band(tmp_path, synthetic_scores):
    raw = yaml.safe_load(_config_path().read_text(encoding="utf-8"))
    raw.pop("anchors")
    raw["calibration"] = {"method": "anchor_platt", "by_band": True}
    path = tmp_path / "defaults.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_config(path).calibration
    assert config.anchor_platt == AnchorFitParams()
    assert fit_calibration_from_df(synthetic_scores, config=config) == (
        fit_calibration_from_df(synthetic_scores, config=replace(config, by_band=False))
    )


@pytest.mark.parametrize("parameters", [
    {"s_minus": 9.0}, {"r_plus": 1.0}, {"r_minus": -0.1},
    {"alpha_max": 0.0}, {"beta_max": float("inf")}, {"l2": -1.0},
    {"min_anchor_size": 0}, {"min_anchor_size": 1.5}, {"tol": 0.0},
])
def test_invalid_anchor_parameters(parameters):
    with pytest.raises(ValueError, match="anchor_platt"):
        fit_anchor_platt(pd.Series([0.0, 10.0]), **parameters)


@pytest.mark.parametrize("bad_score", [np.nan, np.inf, -np.inf, "invalid"])
def test_invalid_scores_raise(bad_score):
    with pytest.raises(ValueError, match="finite numeric"):
        fit_anchor_platt(pd.Series([0, 10, bad_score]))


@pytest.mark.parametrize("section,updates", [
    ("calibration", {"method": "unknown"}),
    ("calibration", {"fallback": "unknown"}),
    ("calibration", {"by_band": "yes"}),
    ("calibration", {"clip_min": 0.9, "clip_max": 0.1}),
    ("anchors", {"pos_anchor": {"rule": "nota_final > 9"}}),
    ("anchors", {"neg_anchor": {"target_mean_probability": 1.0}}),
])
def test_invalid_config(tmp_path, section, updates):
    raw = yaml.safe_load(_config_path().read_text(encoding="utf-8"))
    raw[section].update(updates)
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)
