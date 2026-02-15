from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


PCalMethod = Literal["stub", "platt"]


@dataclass(frozen=True)
class PlattModel:
    intercept: float
    slope: float


def get_nota_series(df: pd.DataFrame) -> pd.Series:
    if "nota final" in df.columns:
        return df["nota final"]
    if "nota_final" in df.columns:
        return df["nota_final"]
    raise KeyError("Missing score column: expected 'nota final' or 'nota_final'.")


def get_target_series(df: pd.DataFrame, *, col: str = "TARGET") -> pd.Series:
    if col not in df.columns:
        raise KeyError(f"Missing target column: expected '{col}'.")
    y = pd.to_numeric(df[col], errors="coerce")
    if y.isna().any():
        n_bad = int(y.isna().sum())
        raise ValueError(f"Found {n_bad} rows with missing/invalid {col}.")
    y_int = y.astype("int64")
    unique = set(y_int.unique().tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(
            f"{col} must be binary in {{0,1}}. Got values: {sorted(unique)}"
        )
    return y_int


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def fit_platt(
    nota: pd.Series,
    y: pd.Series,
    *,
    l2: float = 1e-3,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> PlattModel:
    x = pd.to_numeric(nota, errors="coerce").to_numpy(dtype=np.float64)
    yv = pd.to_numeric(y, errors="coerce").to_numpy(dtype=np.float64)

    if np.isnan(x).any():
        raise ValueError("nota contains NaN after numeric coercion")
    if np.isnan(yv).any():
        raise ValueError("y contains NaN after numeric coercion")
    if not set(np.unique(yv)).issubset({0.0, 1.0}):
        raise ValueError("y must be binary {0,1}")

    n = x.shape[0]
    if n < 10:
        raise ValueError("Need at least 10 rows to fit Platt calibration")

    # Initialize near base rate to avoid extreme intercepts.
    p0 = (yv.sum() + 0.5) / (n + 1.0)
    w0 = float(np.log(p0 / (1.0 - p0)))
    w1 = 0.0

    # Newton steps on NLL + 0.5*l2*w1^2 (do not regularize intercept).
    for _ in range(max_iter):
        z = w0 + w1 * x
        p = _sigmoid(z)
        w = p * (1.0 - p)

        g0 = float(np.sum(p - yv))
        g1 = float(np.sum((p - yv) * x) + l2 * w1)

        h00 = float(np.sum(w))
        h01 = float(np.sum(w * x))
        h11 = float(np.sum(w * x * x) + l2)

        H = np.array([[h00, h01], [h01, h11]], dtype=np.float64)
        g = np.array([g0, g1], dtype=np.float64)

        try:
            delta = np.linalg.solve(H, g)
        except np.linalg.LinAlgError as e:
            raise RuntimeError(
                "Failed to fit Platt calibration (singular Hessian)"
            ) from e

        w0 -= float(delta[0])
        w1 -= float(delta[1])

        if float(np.max(np.abs(delta))) < tol:
            break

    return PlattModel(intercept=w0, slope=w1)


def predict_platt(
    nota: pd.Series,
    *,
    model: PlattModel,
    clip_min: float = 1e-6,
    clip_max: float = 0.999999,
) -> pd.Series:
    x = pd.to_numeric(nota, errors="coerce")
    if x.isna().any():
        n_bad = int(x.isna().sum())
        raise ValueError(f"Found {n_bad} rows with missing/invalid nota final.")

    z = model.intercept + model.slope * x.astype("float64")
    p = 1.0 / (1.0 + np.exp(-z))
    return p.clip(lower=clip_min, upper=clip_max).astype("float64")


def predict_stub(
    nota: pd.Series,
    *,
    clip_min: float = 1e-6,
    clip_max: float = 0.999999,
) -> pd.Series:
    x = pd.to_numeric(nota, errors="coerce")
    if x.isna().any():
        n_bad = int(x.isna().sum())
        raise ValueError(f"Found {n_bad} rows with missing/invalid nota final.")

    p = (x.astype("float64") / 10.0).clip(lower=clip_min, upper=clip_max)
    return p.astype("float64")


def compute_p_cal(
    df: pd.DataFrame,
    *,
    method: PCalMethod,
    model: PlattModel | None = None,
    clip_min: float = 1e-6,
    clip_max: float = 0.999999,
) -> pd.Series:
    nota = get_nota_series(df)
    if method == "stub":
        return predict_stub(nota, clip_min=clip_min, clip_max=clip_max)
    if method == "platt":
        if model is None:
            raise ValueError("Platt method requires a fitted model")
        return predict_platt(nota, model=model, clip_min=clip_min, clip_max=clip_max)
    raise AssertionError(f"Unhandled method: {method}")


def fit_platt_from_df(
    df: pd.DataFrame,
    *,
    target_col: str = "TARGET",
    l2: float = 1e-3,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> PlattModel:
    nota = get_nota_series(df)
    y = get_target_series(df, col=target_col)
    return fit_platt(nota, y, l2=l2, max_iter=max_iter, tol=tol)


def save_platt_model(model: PlattModel, path: str | Path) -> None:
    out = {
        "method": "platt",
        **asdict(model),
    }
    Path(path).write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")


def load_platt_model(path: str | Path) -> PlattModel:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("Calibration file must be a JSON object")
    if data.get("method") != "platt":
        raise ValueError("Unsupported calibration method in file")
    return PlattModel(
        intercept=float(data["intercept"]),
        slope=float(data["slope"]),
    )
