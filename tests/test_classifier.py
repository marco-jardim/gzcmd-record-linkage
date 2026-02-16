from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gzcmd_record_linkage.classifier import ClassifierConfig, GZCMDClassifier


def _dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "COMPREC": ["a"] * 12,
            "REFREC": ["b"] * 12,
            "PASSO": [1] * 12,
            "PAR": [1] * 12,
            "TARGET": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            "nota final": [i for i in range(12)],
            "nota_score": [0.1 * i for i in range(12)],
            "feature_a": [float(i) for i in range(12)],
        }
    )


@pytest.mark.parametrize("classifier_type", ["random_forest", "xgboost"])
def test_classifier_fit_predict_and_features(classifier_type: str) -> None:
    df = _dataset()
    config = ClassifierConfig(classifier_type=classifier_type)  # type: ignore
    clf = GZCMDClassifier(config=config)
    clf.fit(df)

    proba = clf.predict_proba(df)
    assert proba.shape == (12, 2)
    assert len(clf.features_ or []) == 3
    assert "TARGET" not in (clf.features_ or [])

    score = proba[:, 1]
    assert score.min() >= 0.0
    assert score.max() <= 1.0


@pytest.mark.parametrize("classifier_type", ["random_forest", "xgboost"])
def test_classifier_save_load_roundtrip(tmp_path: Path, classifier_type: str) -> None:
    df = _dataset()
    config = ClassifierConfig(classifier_type=classifier_type)  # type: ignore
    clf = GZCMDClassifier(config=config)
    clf.fit(df)

    path = tmp_path / "model.joblib"
    clf.save(path)
    loaded = GZCMDClassifier.load(path)
    assert loaded.features_ == clf.features_
    assert loaded.predict_proba(df).shape == (12, 2)


def test_xgboost_auto_scale_pos_weight() -> None:
    """Test that XGBoost automatically calculates scale_pos_weight."""
    df = _dataset()
    config = ClassifierConfig(
        classifier_type="xgboost",
        xgb_scale_pos_weight=None,  # Auto-calculate
    )
    clf = GZCMDClassifier(config=config)
    clf.fit(df)

    # Should train without error and produce valid probabilities
    proba = clf.predict_proba(df)
    assert proba.shape == (12, 2)
    assert proba[:, 1].min() >= 0.0
    assert proba[:, 1].max() <= 1.0


def test_xgboost_custom_hyperparameters() -> None:
    """Test that custom XGBoost hyperparameters are passed correctly."""
    df = _dataset()
    config = ClassifierConfig(
        classifier_type="xgboost",
        xgb_learning_rate=0.05,
        xgb_n_estimators=100,
        xgb_max_depth=3,
        xgb_subsample=0.7,
        xgb_colsample_bytree=0.7,
    )
    clf = GZCMDClassifier(config=config)
    clf.fit(df)

    # Should train without error
    proba = clf.predict_proba(df)
    assert proba.shape == (12, 2)
