from __future__ import annotations

import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from gzcmd_record_linkage.classifier import (
    ClassifierConfig,
    GZCMDClassifier,
    _detect_cuda_device,
)


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


def test_xgboost_gpu_device_auto_detect() -> None:
    """Test that XGBoost device auto-detection works (falls back to CPU if no CUDA)."""
    df = _dataset()
    config = ClassifierConfig(
        classifier_type="xgboost",
        xgb_device=None,  # Auto-detect
    )
    clf = GZCMDClassifier(config=config)
    clf.fit(df)

    # Should train without error regardless of CUDA availability
    proba = clf.predict_proba(df)
    assert proba.shape == (12, 2)
    assert proba[:, 1].min() >= 0.0
    assert proba[:, 1].max() <= 1.0


def test_xgboost_cpu_device_explicit() -> None:
    """Test that XGBoost can be forced to use CPU device."""
    df = _dataset()
    config = ClassifierConfig(
        classifier_type="xgboost",
        xgb_device="cpu",
    )
    clf = GZCMDClassifier(config=config)
    clf.fit(df)

    # Should train without error
    proba = clf.predict_proba(df)
    assert proba.shape == (12, 2)


def test_xgboost_default_hyperparameters() -> None:
    """Test that default XGBoost hyperparameters match task requirements."""
    config = ClassifierConfig(classifier_type="xgboost")

    assert config.xgb_n_estimators == 500
    assert config.xgb_max_depth == 6
    assert config.xgb_learning_rate == 0.1
    assert config.xgb_tree_method == "hist"


@pytest.fixture
def cuda_probe_without_torch(monkeypatch):
    """Isolate the cached XGBoost fallback from the host's torch/GPU state."""
    monkeypatch.setitem(sys.modules, "torch", None)
    _detect_cuda_device.cache_clear()
    yield
    _detect_cuda_device.cache_clear()


def test_cuda_detection_build_without_cuda(monkeypatch, cuda_probe_without_torch):
    import xgboost

    monkeypatch.setattr(xgboost, "build_info", lambda: {"USE_CUDA": False})

    def unexpected_classifier(**kwargs):
        pytest.fail("A CPU-only build must not attempt a GPU fit")

    monkeypatch.setattr(xgboost, "XGBClassifier", unexpected_classifier)
    assert _detect_cuda_device() == "cpu"


def test_cuda_detection_fit_failure(monkeypatch, cuda_probe_without_torch):
    import xgboost

    monkeypatch.setattr(xgboost, "build_info", lambda: {"USE_CUDA": True})

    def failed_fit(self, X, y):
        raise RuntimeError("GPU unavailable")

    monkeypatch.setattr(xgboost.XGBClassifier, "fit", failed_fit)
    assert _detect_cuda_device() == "cpu"


@pytest.mark.parametrize(
    "message",
    ["No visible GPU is found, setting device to CPU.", "Falling back to CPU."],
)
def test_cuda_detection_fallback_warning(
    monkeypatch, cuda_probe_without_torch, message
):
    import xgboost

    monkeypatch.setattr(xgboost, "build_info", lambda: {"USE_CUDA": True})

    def warned_fit(self, X, y):
        warnings.warn(message, UserWarning, stacklevel=2)
        return self

    monkeypatch.setattr(xgboost.XGBClassifier, "fit", warned_fit)
    assert _detect_cuda_device() == "cpu"


def test_cuda_detection_success_is_cached(monkeypatch, cuda_probe_without_torch):
    import xgboost

    calls = []
    monkeypatch.setattr(xgboost, "build_info", lambda: {"USE_CUDA": True})

    def successful_fit(self, X, y):
        calls.append(self.get_params())
        assert X.shape == (4, 2)
        assert set(y) == {0, 1}
        return self

    monkeypatch.setattr(xgboost.XGBClassifier, "fit", successful_fit)
    assert _detect_cuda_device() == "cuda"
    assert _detect_cuda_device() == "cuda"
    assert len(calls) == 1
    assert calls[0]["device"] == "cuda"
    assert calls[0]["tree_method"] == "hist"
    assert calls[0]["n_estimators"] == 1


def test_cuda_detection_torch_shortcut(monkeypatch, cuda_probe_without_torch):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
    )
    assert _detect_cuda_device() == "cuda"


def test_explicit_cpu_bypasses_cuda_detection(monkeypatch):
    import gzcmd_record_linkage.classifier as classifier_module

    def unexpected_detection():
        pytest.fail("An explicit device must bypass automatic detection")

    monkeypatch.setattr(classifier_module, "_detect_cuda_device", unexpected_detection)
    clf = GZCMDClassifier(
        ClassifierConfig(
            xgb_device="cpu", xgb_n_estimators=1, calibration_cv=2, n_jobs=1
        )
    ).fit(_dataset())
    params = clf.pipeline_.named_steps["classifier"].estimator.get_params()
    assert params["device"] == "cpu"
