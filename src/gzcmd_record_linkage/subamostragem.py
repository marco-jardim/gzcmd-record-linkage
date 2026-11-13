"""Training-only negative undersampling around the nested OOF pipeline.

Faithful port of the generic parts of ``gzcmd/subamostragem.py`` from the thesis
research repository (exploratory, post hoc). Only negatives of the *training*
fits are removed; the inner prediction sets, the inner calibration labels and
the outer evaluation sets keep the real prevalence. The splitting, calibration
and classifier come unchanged from :mod:`gzcmd_record_linkage.oof`.

``evaluate`` and ``summarize`` carry study-specific choices in the thesis (top-k
list, minimum-recovery targets, top-96 sensitivity, step and sex labels, the
2522-row curve, the ``"original"`` reference row). Here each of them is a
keyword parameter whose default is a named ``THESIS_*`` constant, so the thesis
behaviour is reproduced exactly while other studies can override every value.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from .oof import N_SPLITS, SEED, nested_oof_probabilities, thesis_classifier

RATIOS = (1, 5, 20, 50)
"""Supported positive:negative ratios (negatives kept per positive)."""

REMOVAL_SEEDS = range(1000, 1010)
"""Thesis seeds of the negative removal (the model seed stays ``SEED``)."""

THESIS_TOP_K = (11, 17, 29, 58, 96, 121, 125, 440, 2522)
"""Queue depths whose cumulative positives are reported."""

THESIS_TARGETS = (9, 23, 36)
"""Positive counts whose minimum queue depth is reported."""

THESIS_SENSITIVITY_K = 96
"""Queue depth of the per-subgroup sensitivity."""

THESIS_STEP_LABELS = ("6", "9", "11")
"""Labels of the first subgroup dimension (linkage step)."""

THESIS_SEX_LABELS = ("F", "M", "MISSING")
"""Labels of the second subgroup dimension (sex)."""

THESIS_MISSING_LABEL = "MISSING"
"""Subgroup label that never gets a sensitivity value."""

THESIS_MIN_CELL = 30
"""Minimum subgroup size for a sensitivity value."""

THESIS_CURVE_MAX_K = 2522
"""Maximum queue depth of the cumulative curve."""

THESIS_REFERENCE_RATIO = "original"
"""``ratio`` value of the reference (no undersampling) rows in ``summarize``."""

THESIS_COUNT_NAME = "par2"
"""Name of the cumulative-positive metric/column."""

THESIS_STEP_NAME = "passo"
"""Name of the first subgroup dimension in metric names."""

THESIS_SEX_NAME = "sexo"
"""Name of the second subgroup dimension in metric names."""

SamplingAudit = Callable[[str, np.ndarray, np.ndarray], None]
"""``audit(stage, original_index, selected_index)``: ``stage`` is ``"fit"`` (the
index of the received training frame and of the sampled subset) or ``"predict"``
(the predicted index twice)."""


def sample_training_indices(
    y: ArrayLike, ratio: int | None, rng: np.random.Generator
) -> tuple[NDArray[np.int64], float]:
    """Keep every positive; sample at most ``ratio`` negatives per positive.

    Return the sorted kept positions and ``beta``, the fraction of negatives
    retained. ``ratio=None`` keeps everything (``beta == 1.0``).
    """
    labels = np.asarray(y, dtype=int)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Binary training labels required")
    if ratio is None:
        return np.arange(len(labels)), 1.0
    if ratio not in RATIOS:
        raise ValueError("Unsupported positive:negative ratio")
    positives, negatives = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    count = min(len(negatives), ratio * len(positives))
    selected = rng.choice(negatives, size=count, replace=False)
    return np.sort(np.concatenate([positives, selected])), count / len(negatives)


def correct_prior(p: ArrayLike, beta: ArrayLike) -> NDArray[np.float64]:
    """Dal Pozzolo et al. (2015), eq. (9), PDF p. 3; ``beta`` retains negatives.

    This assumes probabilities under the sampled prior. Applying it after an
    isotonic fit on real-prevalence labels is only a double-correction diagnostic.
    """
    probabilities, retention = np.asarray(p, dtype=float), np.asarray(beta, dtype=float)
    if (
        not np.isfinite(probabilities).all()
        or not np.isfinite(retention).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
        or ((retention <= 0) | (retention > 1)).any()
    ):
        raise ValueError("Probabilities in [0,1] and beta in (0,1] required")
    return retention * probabilities / (retention * probabilities + (1 - probabilities))


def undersampled_oof(
    x: pd.DataFrame,
    y: ArrayLike,
    groups: ArrayLike,
    *,
    ratio: int | None = None,
    removal_seed: int = 1000,
    model_factory: Callable[[np.ndarray, int], Any] | None = None,
    n_splits: int = N_SPLITS,
    audit: SamplingAudit | None = None,
) -> tuple[dict[str, NDArray[np.float64]], NDArray[np.int64], list[dict[str, Any]]]:
    """Delegate the unchanged ``refrec`` nested OOF to :func:`nested_oof_probabilities`.

    The factory's wrapper changes only fit membership. Inner prediction sets,
    inner calibration labels and outer evaluation sets are never sampled. The
    model seed stays ``SEED``; only the removal uses ``removal_seed``. The default
    classifier computes ``scale_pos_weight`` on the actual sampled training labels.

    Return ``({"isotonic", "isotonic_prior", "raw", "raw_prior"}, folds,
    effective)``: the calibrated and raw outer probabilities with and without the
    prior correction, the outer fold of each row and one record per fit (stage,
    sampled class counts and ``beta``).
    """
    factory = model_factory or thesis_classifier
    rng = np.random.default_rng(removal_seed)
    positions = pd.Series(np.arange(len(x)), index=x.index)
    if not x.index.is_unique:
        raise ValueError("Unique feature index required")
    outer_raw = np.full(len(x), np.nan)
    outer_beta = np.full(len(x), np.nan)
    fits = 0
    effective: list[dict[str, Any]] = []

    class SampledModel:
        def __init__(self, seed: int, is_outer: bool) -> None:
            self.seed, self.is_outer = seed, is_outer

        def fit(self, frame: pd.DataFrame, labels: np.ndarray) -> SampledModel:
            selected, self.beta = sample_training_indices(labels, ratio, rng)
            sampled_y = np.asarray(labels)[selected]
            self.model = factory(sampled_y, self.seed)
            self.model.fit(frame.iloc[selected], sampled_y)
            effective.append(
                {
                    "stage": "outer" if self.is_outer else "inner",
                    "n_positive": int(sampled_y.sum()),
                    "n_negative": int((sampled_y == 0).sum()),
                    "beta": self.beta,
                }
            )
            if audit is not None:
                audit(
                    "fit", frame.index.to_numpy(), frame.iloc[selected].index.to_numpy()
                )
            return self

        def predict_proba(self, frame: pd.DataFrame) -> NDArray[np.float64]:
            prediction = self.model.predict_proba(frame)
            if audit is not None:
                audit("predict", frame.index.to_numpy(), frame.index.to_numpy())
            if self.is_outer:
                idx = positions.loc[frame.index].to_numpy()
                outer_raw[idx], outer_beta[idx] = prediction[:, 1], self.beta
            return prediction

    def sampled_factory(labels: np.ndarray, seed: int) -> SampledModel:
        nonlocal fits
        outer = fits % (n_splits + 1) == n_splits
        fits += 1
        return SampledModel(seed, outer)

    isotonic, folds = nested_oof_probabilities(
        x,
        y,
        groups=groups,
        split_by="refrec",
        seed=SEED,
        n_splits=n_splits,
        model_factory=sampled_factory,
    )
    if not np.isfinite(outer_raw).all() or not np.isfinite(outer_beta).all():
        raise ValueError("Incomplete outer predictions")
    probabilities = {
        "isotonic": isotonic,
        "isotonic_prior": correct_prior(isotonic, outer_beta),
        "raw": outer_raw,
        "raw_prior": correct_prior(outer_raw, outer_beta),
    }
    return probabilities, folds, effective


def ece_quantile(y: ArrayLike, p: ArrayLike) -> float:
    """Expected calibration error over 10 probability-quantile bins.

    Ties stay together and duplicate edges drop; constant probabilities form one
    bin and empty intervals contribute zero.
    """
    labels, probabilities = np.asarray(y), np.asarray(p)
    edges = np.unique(np.quantile(probabilities, np.linspace(0, 1, 11)))
    bins = np.searchsorted(edges[1:-1], probabilities, side="left")
    return float(
        sum(
            np.mean(bins == b)
            * abs(labels[bins == b].mean() - probabilities[bins == b].mean())
            for b in np.unique(bins)
        )
    )


def evaluate(
    y: ArrayLike,
    probabilities: Mapping[str, NDArray[np.float64]],
    ids: ArrayLike,
    below: ArrayLike,
    steps: ArrayLike,
    sexes: ArrayLike,
    *,
    ratio: Any,
    removal_seed: int,
    top_k: Sequence[int] = THESIS_TOP_K,
    targets: Sequence[int] = THESIS_TARGETS,
    sensitivity_k: int = THESIS_SENSITIVITY_K,
    step_labels: Sequence[str] = THESIS_STEP_LABELS,
    sex_labels: Sequence[str] = THESIS_SEX_LABELS,
    missing_label: str = THESIS_MISSING_LABEL,
    min_cell: int = THESIS_MIN_CELL,
    curve_max_k: int = THESIS_CURVE_MAX_K,
    count_name: str = THESIS_COUNT_NAME,
    step_name: str = THESIS_STEP_NAME,
    sex_name: str = THESIS_SEX_NAME,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the long aggregate metrics and the aggregate cumulative curve.

    For each variant of ``probabilities`` and for the ``below`` and ``full``
    populations: AUC, AUC-PR, Brier, quantile ECE, size and positives. On the
    ``below`` population the queue is ordered by descending score, ties broken by
    ``ids``; it adds the cumulative positives at each ``top_k`` depth
    (``{count_name}_top_{k}``), the minimum depth reaching each of ``targets``
    (``k_min_{target}``), the sensitivity of the top ``sensitivity_k`` per
    subgroup of ``steps`` and ``sexes`` (``sens{k}_{dimension}_{label}`` with
    ``_den``/``_num``/``_n``; NaN for ``missing_label`` or cells under
    ``min_cell``) and the curve up to ``curve_max_k``.
    """
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    labels_all, below_mask = np.asarray(y), np.asarray(below, dtype=bool)
    records: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    for variant, p in probabilities.items():
        for population, mask in (
            ("below", below_mask),
            ("full", np.ones(len(labels_all), dtype=bool)),
        ):
            labels, scores = labels_all[mask], p[mask]
            values: dict[str, Any] = {
                "auc": roc_auc_score(labels, scores),
                "auc_pr": average_precision_score(labels, scores),
                "brier": brier_score_loss(labels, scores),
                "ece": ece_quantile(labels, scores),
                "n": len(labels),
                "positives": int(labels.sum()),
            }
            if population == "below":
                order = np.lexsort((np.asarray(ids)[mask], -scores))
                cumulative = labels[order].cumsum()
                for k in top_k:
                    if k <= len(order):
                        values[f"{count_name}_top_{k}"] = int(cumulative[k - 1])
                for target in targets:
                    values[f"k_min_{target}"] = (
                        int(np.searchsorted(cumulative, target) + 1)
                        if cumulative[-1] >= target
                        else np.nan
                    )
                selected = np.zeros(len(order), dtype=bool)
                selected[order[:sensitivity_k]] = True
                for dimension, groups, names in (
                    (step_name, np.asarray(steps)[mask], step_labels),
                    (sex_name, np.asarray(sexes)[mask], sex_labels),
                ):
                    for group in names:
                        cell = groups.astype(str) == group
                        den = int(labels[cell].sum())
                        num = int(labels[cell & selected].sum())
                        eligible = (
                            group != missing_label and int(cell.sum()) >= min_cell
                        )
                        prefix = f"sens{sensitivity_k}_{dimension}_{group}"
                        values[prefix] = num / den if den and eligible else np.nan
                        values[prefix + "_den"] = den
                        values[prefix + "_num"] = num
                        values[prefix + "_n"] = int(cell.sum())
                curves.extend(
                    {
                        "ratio": ratio,
                        "seed": removal_seed,
                        "variant": variant,
                        "k": k,
                        count_name: int(cumulative[k - 1]),
                    }
                    for k in range(1, min(curve_max_k, len(order)) + 1)
                )
            records.extend(
                {
                    "ratio": ratio,
                    "seed": removal_seed,
                    "variant": variant,
                    "population": population,
                    "metric": metric,
                    "value": float(value),
                }
                for metric, value in values.items()
            )
    return pd.DataFrame(records), pd.DataFrame(curves)


def summarize(
    metrics: pd.DataFrame,
    curves: pd.DataFrame,
    *,
    reference_ratio: Any = THESIS_REFERENCE_RATIO,
    count_name: str = THESIS_COUNT_NAME,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize across seeds per ratio and compare with the reference rows.

    ``metrics`` must contain rows whose ``ratio`` equals ``reference_ratio``.
    Return the per-(ratio, variant, population, metric) median/min/max with the
    deltas to the reference, and the per-``k`` mean/min/max of the curve.
    """
    keys = ["variant", "population", "metric"]
    reference = metrics.loc[metrics.ratio.eq(reference_ratio), [*keys, "value"]].rename(
        columns={"value": "reference"}
    )
    summary = (
        metrics.groupby(["ratio", *keys], sort=True, dropna=False)
        .value.agg(median="median", minimum="min", maximum="max", seeds="count")
        .reset_index()
    )
    summary = summary.merge(reference, on=keys, validate="many_to_one")
    for column in ("median", "minimum", "maximum"):
        summary[f"delta_{column}"] = summary[column] - summary.reference
    curve = (
        curves.groupby(["ratio", "variant", "k"], sort=True)[count_name]
        .agg(mean="mean", minimum="min", maximum="max")
        .reset_index()
    )
    return summary, curve
