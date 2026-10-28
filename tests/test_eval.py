from dataclasses import replace
from importlib.resources import files

import numpy as np
import pandas as pd
import pytest
import synthetic_data

from gzcmd_record_linkage import eval as eval_module
from gzcmd_record_linkage.classifier import ClassifierConfig
from gzcmd_record_linkage.config import load_config
from gzcmd_record_linkage.loader import LoadConfig, feature_engineer


@pytest.mark.parametrize(
    ("calibration", "classifier_type"),
    [("ml_rf", "random_forest"), ("ml_xgb", "xgboost")],
)
@pytest.mark.parametrize("custom", [False, True])
def test_evaluate_classifier_configuration(
    monkeypatch, calibration, classifier_type, custom
):
    seen = []

    class ClassifierSpy:
        def __init__(self, *, config):
            seen.append(config)

        def fit(self, df):
            assert "TARGET" in df
            return self

        def predict_proba(self, df):
            return np.full((len(df), 2), 0.5)

    monkeypatch.setattr(eval_module, "GZCMDClassifier", ClassifierSpy)
    config = ClassifierConfig(
        classifier_type="random_forest" if classifier_type == "xgboost" else "xgboost",
        n_estimators=7,
        xgb_n_estimators=9,
        xgb_device="cpu",
        random_state=123,
    )
    df = feature_engineer(
        synthetic_data.generate_comparador(seed=42, n_pairs=40).frame,
        cfg=LoadConfig(),
    )
    result = eval_module.evaluate_v3_dataframe(
        df,
        cfg=load_config(files("gzcmd_record_linkage") / "gzcmd_v3_config.yaml"),
        modes=["vigilancia"],
        split_by="row",
        seeds=[42],
        test_size=0.3,
        group_stratify=True,
        calibration=calibration,
        macd_enabled=True,
        **({"classifier_config": config} if custom else {}),
    )
    expected = (
        replace(config, classifier_type=classifier_type)
        if custom
        else ClassifierConfig(classifier_type=classifier_type)
    )
    assert seen == [expected]
    assert len(result) == 1
    assert config.classifier_type != classifier_type


def test_evaluate_csv_forwards_classifier_configuration(monkeypatch, tmp_path):
    csv_path = synthetic_data.to_comparador_csv(
        synthetic_data.generate_comparador(seed=42, n_pairs=40).frame,
        tmp_path / "comparador.csv",
    )
    config = ClassifierConfig(xgb_device="cpu", xgb_n_estimators=7)
    expected = pd.DataFrame({"forwarded": [True]})

    def evaluate_spy(df, **kwargs):
        assert kwargs["classifier_config"] is config
        assert "TARGET" in df
        return expected

    monkeypatch.setattr(eval_module, "evaluate_v3_dataframe", evaluate_spy)
    actual = eval_module.evaluate_v3_csv(
        input_csv=csv_path,
        config_path=files("gzcmd_record_linkage") / "gzcmd_v3_config.yaml",
        modes=["vigilancia"],
        split_by="row",
        seeds=[42],
        test_size=0.3,
        group_stratify=True,
        calibration="ml_xgb",
        macd_enabled=True,
        classifier_config=config,
    )
    assert actual is expected
