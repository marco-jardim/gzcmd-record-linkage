from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConfusionCounts:
    tp: float
    fp: float
    fn: float
    tn: float

    @property
    def n(self) -> float:
        return float(self.tp + self.fp + self.fn + self.tn)

    @property
    def support_pos(self) -> float:
        return float(self.tp + self.fn)

    @property
    def support_neg(self) -> float:
        return float(self.fp + self.tn)


def _as_binary_1d(x: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")

    try:
        arr_int = arr.astype(np.int8)
    except Exception as e:  # pragma: no cover (defensive)
        raise ValueError(f"{name} must be coercible to int binary 0/1") from e

    unique = set(np.unique(arr_int).tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(f"{name} must be binary in {{0,1}}; got {sorted(unique)}")

    return arr_int


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> ConfusionCounts:
    yt = _as_binary_1d(y_true, name="y_true")
    yp = _as_binary_1d(y_pred, name="y_pred")
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    tp = float(np.sum((yt == 1) & (yp == 1)))
    fp = float(np.sum((yt == 0) & (yp == 1)))
    fn = float(np.sum((yt == 1) & (yp == 0)))
    tn = float(np.sum((yt == 0) & (yp == 0)))
    return ConfusionCounts(tp=tp, fp=fp, fn=fn, tn=tn)


def _safe_div(num: float, den: float, *, zero_division: float = 0.0) -> float:
    if den <= 0.0:
        return float(zero_division)
    return float(num / den)


def precision(counts: ConfusionCounts, *, zero_division: float = 0.0) -> float:
    return _safe_div(counts.tp, counts.tp + counts.fp, zero_division=zero_division)


def recall(counts: ConfusionCounts, *, zero_division: float = 0.0) -> float:
    return _safe_div(counts.tp, counts.tp + counts.fn, zero_division=zero_division)


def fbeta(counts: ConfusionCounts, *, beta: float, zero_division: float = 0.0) -> float:
    if beta <= 0.0:
        raise ValueError("beta must be > 0")
    b2 = float(beta * beta)
    num = (1.0 + b2) * counts.tp
    den = (1.0 + b2) * counts.tp + b2 * counts.fn + counts.fp
    return _safe_div(num, den, zero_division=zero_division)


def f1(counts: ConfusionCounts, *, zero_division: float = 0.0) -> float:
    return fbeta(counts, beta=1.0, zero_division=zero_division)


def metrics_dict(
    counts: ConfusionCounts,
    *,
    beta: float,
    prefix: str = "",
) -> dict[str, float]:
    return {
        f"{prefix}tp": float(counts.tp),
        f"{prefix}fp": float(counts.fp),
        f"{prefix}fn": float(counts.fn),
        f"{prefix}tn": float(counts.tn),
        f"{prefix}precision": precision(counts),
        f"{prefix}recall": recall(counts),
        f"{prefix}f1": f1(counts),
        f"{prefix}fbeta": fbeta(counts, beta=beta),
    }
