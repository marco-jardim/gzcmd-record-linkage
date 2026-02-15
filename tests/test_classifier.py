from __future__ import annotations

from pathlib import Path

import pandas as pd

from gzcmd_record_linkage.classifier import GZCMDClassifier


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


def test_classifier_fit_predict_and_features() -> None:
    df = _dataset()
    clf = GZCMDClassifier()
    clf.fit(df)

    proba = clf.predict_proba(df)
    assert proba.shape == (12, 2)
    assert len(clf.features_ or []) == 3
    assert "TARGET" not in (clf.features_ or [])

    score = proba[:, 1]
    assert score.min() >= 0.0
    assert score.max() <= 1.0


def test_classifier_save_load_roundtrip(tmp_path: Path) -> None:
    df = _dataset()
    clf = GZCMDClassifier()
    clf.fit(df)

    path = tmp_path / "model.joblib"
    clf.save(path)
    loaded = GZCMDClassifier.load(path)
    assert loaded.features_ == clf.features_
    assert loaded.predict_proba(df).shape == (12, 2)
