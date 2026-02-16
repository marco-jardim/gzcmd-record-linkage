"""ML classifier integration for GZ-CMD.

Wraps a scikit-learn pipeline (XGBoost or RandomForest + Platt Calibration) to provide
probability estimates based on the full feature set (aggregates + subscores + MACD).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .loader import COL_TARGET

ClassifierType = Literal["random_forest", "xgboost"]


@dataclass(frozen=True)
class ClassifierConfig:
    classifier_type: ClassifierType = "xgboost"
    # Random Forest parameters
    n_estimators: int = 200
    max_depth: int = 10
    class_weight: str | dict | None = "balanced"
    # XGBoost parameters
    xgb_learning_rate: float = 0.1
    xgb_n_estimators: int = 300
    xgb_max_depth: int = 6
    xgb_scale_pos_weight: float | None = None
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    # Common parameters
    random_state: int = 42
    n_jobs: int = -1
    calibration_cv: int = 5


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Identify numeric feature columns for the classifier."""
    # Metadata and targets to exclude
    exclude = {
        "COMPREC",
        "REFREC",
        "PASSO",
        "PAR",
        "TARGET",
        "R_DTNASC",
        "C_DTNASC",
        "R_DTOBITO",
        "C_DTDIAG",
        "R_DTNASC_dt",
        "C_DTNASC_dt",
        "R_DTOBITO_dt",
        "C_DTDIAG_dt",
        "guardrail",
        "guardrail_reason",
        "band",
        "p_cal",
        "action",
        "base_choice",
        "base_loss",
        "loss_llm",
        "evr",
        "review_requested",
    }

    # Select numeric columns
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)

    return sorted(cols)


class GZCMDClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, config: ClassifierConfig = ClassifierConfig()):
        self.config = config
        self.features_: list[str] | None = None
        self.pipeline_: Pipeline | None = None

    def fit(self, df: pd.DataFrame, y: pd.Series | None = None) -> "GZCMDClassifier":
        if y is None:
            if COL_TARGET not in df.columns:
                raise ValueError(f"Target column '{COL_TARGET}' not found in DataFrame")
            y_raw = df[COL_TARGET]
            y = pd.Series(y_raw) if not isinstance(y_raw, pd.Series) else y_raw

        self.features_ = get_feature_columns(df)
        X = df[self.features_]

        # Select base classifier
        if self.config.classifier_type == "xgboost":
            # Lazy import to avoid hard dependency
            from xgboost import XGBClassifier

            # Auto-calculate scale_pos_weight if not provided
            scale_pos_weight = self.config.xgb_scale_pos_weight
            if scale_pos_weight is None:
                n_neg = int((y == 0).sum())
                n_pos = int((y == 1).sum())
                scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

            base_clf = XGBClassifier(
                learning_rate=self.config.xgb_learning_rate,
                n_estimators=self.config.xgb_n_estimators,
                max_depth=self.config.xgb_max_depth,
                scale_pos_weight=scale_pos_weight,
                subsample=self.config.xgb_subsample,
                colsample_bytree=self.config.xgb_colsample_bytree,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
                use_label_encoder=False,
                eval_metric="logloss",
            )
        else:  # random_forest
            base_clf = RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                class_weight=self.config.class_weight,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
            )

        # Pipeline: Impute -> Calibrated classifier
        # We use CalibratedClassifierCV to get probabilities calibrated on the training set
        # (via CV). CalibratedClassifierCV(method='sigmoid') applies Platt scaling.

        calibrated_clf = CalibratedClassifierCV(
            estimator=base_clf,
            method="sigmoid",  # Platt scaling
            cv=self.config.calibration_cv,
            n_jobs=self.config.n_jobs,
        )

        self.pipeline_ = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
                ("classifier", calibrated_clf),
            ]
        )

        self.pipeline_.fit(X, y)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.pipeline_ is None or self.features_ is None:
            raise RuntimeError("Classifier not fitted")

        # Ensure we have all expected features (fill missing with NaN/0)
        X = pd.DataFrame(index=df.index)
        for f in self.features_:
            if f in df.columns:
                X[f] = df[f]
            else:
                X[f] = np.nan

        return self.pipeline_.predict_proba(X)

    def save(self, path: str | Path) -> None:
        if self.pipeline_ is None:
            raise RuntimeError("Cannot save unfitted model")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "config": self.config,
            "features": self.features_,
            "pipeline": self.pipeline_,
        }
        joblib.dump(state, path)

    @classmethod
    def load(cls, path: str | Path) -> "GZCMDClassifier":
        state = joblib.load(path)
        clf = cls(config=state["config"])
        clf.features_ = state["features"]
        clf.pipeline_ = state["pipeline"]
        return clf
