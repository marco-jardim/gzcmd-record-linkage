from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


SplitBy = Literal["row", "comprec", "refrec"]


@dataclass(frozen=True)
class SplitSpec:
    split_by: SplitBy = "row"
    test_size: float = 0.3
    seed: int = 42
    group_stratify: bool = True


def _validate_test_size(test_size: float) -> None:
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be in (0,1)")


def stratified_train_test_split_indices(
    y: np.ndarray,
    *,
    test_size: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    _validate_test_size(test_size)
    y_arr = np.asarray(y)
    if y_arr.ndim != 1:
        raise ValueError("y must be 1D")

    try:
        y_int = y_arr.astype(np.int8)
    except Exception as e:  # pragma: no cover
        raise ValueError("y must be coercible to int") from e

    unique = set(np.unique(y_int).tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(f"y must be binary in {{0,1}}; got {sorted(unique)}")

    n0 = int(np.sum(y_int == 0))
    n1 = int(np.sum(y_int == 1))
    if n0 < 2 or n1 < 2:
        raise ValueError("Need at least 2 samples per class for stratified split")

    idx = np.arange(len(y_int))
    test_parts: list[np.ndarray] = []
    train_parts: list[np.ndarray] = []
    for cls in (0, 1):
        cls_idx = idx[y_int == cls].copy()
        rng.shuffle(cls_idx)

        n_cls = len(cls_idx)
        n_test = int(round(n_cls * test_size))
        n_test = max(1, min(n_cls - 1, n_test))

        test_parts.append(cls_idx[:n_test])
        train_parts.append(cls_idx[n_test:])

    test_idx = np.concatenate(test_parts)
    train_idx = np.concatenate(train_parts)
    rng.shuffle(test_idx)
    rng.shuffle(train_idx)
    return train_idx, test_idx


def group_train_test_split_indices(
    groups: pd.Series,
    y: np.ndarray,
    *,
    test_size: float,
    rng: np.random.Generator,
    stratify: bool,
) -> tuple[np.ndarray, np.ndarray]:
    _validate_test_size(test_size)
    if groups.isna().any():
        n_bad = int(groups.isna().sum())
        raise ValueError(f"groups contains {n_bad} NA values")

    group_key = groups.astype(str)
    y_arr = np.asarray(y)
    if y_arr.ndim != 1 or len(y_arr) != len(group_key):
        raise ValueError("y must be 1D and aligned with groups")

    y_int = y_arr.astype(np.int8)
    unique = set(np.unique(y_int).tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(f"y must be binary in {{0,1}}; got {sorted(unique)}")

    tmp = pd.DataFrame({"group": group_key, "y": y_int})
    group_label = tmp.groupby("group", sort=False)["y"].max()
    group_ids = group_label.index.to_numpy()
    group_y = group_label.to_numpy(dtype=np.int8)

    if len(group_ids) < 2:
        raise ValueError("Need at least 2 groups for a group split")

    if stratify:
        train_gi, test_gi = stratified_train_test_split_indices(
            group_y, test_size=test_size, rng=rng
        )
    else:
        perm = rng.permutation(len(group_ids))
        n_test = int(round(len(group_ids) * test_size))
        n_test = max(1, min(len(group_ids) - 1, n_test))
        test_gi = perm[:n_test]
        train_gi = perm[n_test:]

    test_groups = set(group_ids[test_gi].tolist())
    in_test = group_key.isin(test_groups).to_numpy(dtype=bool)
    test_idx = np.flatnonzero(in_test)
    train_idx = np.flatnonzero(~in_test)
    return train_idx, test_idx


def split_train_test_indices(
    df: pd.DataFrame,
    y: pd.Series,
    *,
    spec: SplitSpec,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(spec.seed))
    split_by = str(spec.split_by).strip().lower()

    y_arr = pd.to_numeric(y, errors="coerce")
    if y_arr.isna().any():
        n_bad = int(y_arr.isna().sum())
        raise ValueError(f"Found {n_bad} NA values in y")

    if split_by == "row":
        return stratified_train_test_split_indices(
            y_arr.to_numpy(), test_size=spec.test_size, rng=rng
        )

    if split_by == "comprec":
        if "COMPREC" not in df.columns:
            raise KeyError("Missing COMPREC column required for comprec split")
        return group_train_test_split_indices(
            df["COMPREC"],
            y_arr.to_numpy(),
            test_size=spec.test_size,
            rng=rng,
            stratify=bool(spec.group_stratify),
        )

    if split_by == "refrec":
        if "REFREC" not in df.columns:
            raise KeyError("Missing REFREC column required for refrec split")
        return group_train_test_split_indices(
            df["REFREC"],
            y_arr.to_numpy(),
            test_size=spec.test_size,
            rng=rng,
            stratify=bool(spec.group_stratify),
        )

    raise ValueError("split_by must be one of: row, comprec, refrec")
