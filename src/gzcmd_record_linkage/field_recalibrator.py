"""Multi-field logistic recalibrator fitted by MAP with a Gaussian prior.

The model is ``logit p = b0 + sum_j b_j x_j`` over the one-hot field-level
indicators of ``fellegi_sunter.design_matrix``. The prior means come from the
unsupervised Fellegi-Sunter fit; agreement coefficients are constrained to be
non-negative and, optionally, kept in a box around the prior means. The final
score (``nota final``) never enters this model.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize
from scipy.special import expit

from gzcmd_record_linkage.fellegi_sunter import FellegiSunterModel, agreement_mask

Status = Literal["prior", "fitted", "skipped_insufficient", "fallback_prior"]


@dataclass(frozen=True)
class FieldRecalibration:
    """Fitted (or inherited) logistic coefficients with provenance."""

    intercept: float
    coef: tuple[float, ...]
    feature_names: tuple[str, ...]
    status: Status
    n_labeled: int
    n_pos: int
    n_neg: int
    message: str = ""

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        matrix = np.asarray(X, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.coef):
            raise ValueError(
                f"X must be 2-D with {len(self.coef)} columns, got {matrix.shape}"
            )
        eta = self.intercept + matrix @ np.asarray(self.coef, dtype=np.float64)
        return np.asarray(expit(eta), dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        return {
            "intercept": float(self.intercept),
            "coef": [float(c) for c in self.coef],
            "feature_names": list(self.feature_names),
            "status": self.status,
            "n_labeled": int(self.n_labeled),
            "n_pos": int(self.n_pos),
            "n_neg": int(self.n_neg),
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FieldRecalibration:
        return cls(
            intercept=float(data["intercept"]),
            coef=tuple(float(c) for c in data["coef"]),
            feature_names=tuple(str(n) for n in data["feature_names"]),
            status=data["status"],
            n_labeled=int(data["n_labeled"]),
            n_pos=int(data["n_pos"]),
            n_neg=int(data["n_neg"]),
            message=str(data.get("message", "")),
        )


@dataclass(frozen=True)
class RecalibrationPrior:
    """Gaussian prior on (intercept, coefficients) plus optional trust box."""

    mean_intercept: float
    mean_coef: tuple[float, ...]
    sd_intercept: float
    sd_coef: tuple[float, ...]
    agreement_mask: tuple[bool, ...]
    trust_intercept: float | None = None
    trust_coef: float | None = None
    feature_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        k = len(self.mean_coef)
        if len(self.sd_coef) != k or len(self.agreement_mask) != k:
            raise ValueError("mean_coef, sd_coef and agreement_mask must align")
        if self.feature_names and len(self.feature_names) != k:
            raise ValueError("feature_names must align with mean_coef")
        sds = (self.sd_intercept, *self.sd_coef)
        if not all(math.isfinite(s) and s > 0 for s in sds):
            raise ValueError("Prior standard deviations must be finite and > 0")
        means = (self.mean_intercept, *self.mean_coef)
        if not all(math.isfinite(v) for v in means):
            raise ValueError("Prior means must be finite")
        for trust in (self.trust_intercept, self.trust_coef):
            if trust is not None and not (math.isfinite(trust) and trust >= 0):
                raise ValueError("Trust radii must be finite and >= 0")

    @classmethod
    def from_fellegi_sunter(
        cls,
        model: FellegiSunterModel,
        feature_names: Sequence[str],
        *,
        sd_intercept: float,
        sd_coef: float,
        trust_intercept: float | None = None,
        trust_coef: float | None = None,
    ) -> RecalibrationPrior:
        intercept, coef = model.logistic_coefficients()
        names = tuple(feature_names)
        if len(names) != len(coef):
            raise ValueError(
                f"feature_names has {len(names)} entries, model has {len(coef)}"
            )
        mask = agreement_mask(names)
        return cls(
            mean_intercept=float(intercept),
            mean_coef=tuple(float(c) for c in coef),
            sd_intercept=float(sd_intercept),
            sd_coef=tuple(float(sd_coef) for _ in coef),
            agreement_mask=tuple(bool(b) for b in mask),
            trust_intercept=trust_intercept,
            trust_coef=trust_coef,
            feature_names=names,
        )

    def as_recalibration(
        self, status: Status = "prior", message: str = ""
    ) -> FieldRecalibration:
        """The prior means as a model (used before any label exists).

        The means are projected onto the bounds (agreement coefficients >= 0 and
        the trust box), the same starting point the fit uses, so the model that
        orders a round never violates the constraints the fit enforces.
        """
        names = self.feature_names or tuple(f"x{j}" for j in range(len(self.mean_coef)))
        start = _bounds(self)[2]
        return FieldRecalibration(
            intercept=float(start[0]),
            coef=tuple(float(c) for c in start[1:]),
            feature_names=names,
            status=status,
            n_labeled=0,
            n_pos=0,
            n_neg=0,
            message=message,
        )


def _validate(
    X: ArrayLike, y: ArrayLike, weights: ArrayLike | None, n_features: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    matrix = np.asarray(X, dtype=np.float64)
    labels = np.asarray(y, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"X must be 2-D, got {matrix.ndim}-D")
    if labels.ndim != 1 or labels.shape[0] != matrix.shape[0]:
        raise ValueError("y must be 1-D and aligned with the rows of X")
    if matrix.shape[1] != n_features:
        raise ValueError(f"X has {matrix.shape[1]} columns, prior has {n_features}")
    if not np.isfinite(matrix).all():
        raise ValueError("X must be finite")
    if (
        not np.isfinite(labels).all()
        or labels.min(initial=0.0) < 0
        or (labels.max(initial=0.0) > 1)
    ):
        raise ValueError("y must be finite and within [0, 1]")
    if weights is None:
        w = np.ones(matrix.shape[0], dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != labels.shape:
            raise ValueError("weights must be 1-D and aligned with y")
        if not np.isfinite(w).all() or (w < 0).any():
            raise ValueError("weights must be finite and >= 0")
    return matrix, labels, w


def _bounds(prior: RecalibrationPrior) -> tuple[NDArray[np.float64], ...]:
    """Lower/upper bounds for ``[intercept, *coef]`` and the projected start."""
    mean = np.asarray([prior.mean_intercept, *prior.mean_coef], dtype=np.float64)
    lower = np.full(mean.shape, -np.inf)
    upper = np.full(mean.shape, np.inf)
    lower[1:] = np.where(np.asarray(prior.agreement_mask, dtype=bool), 0.0, -np.inf)
    if prior.trust_intercept is not None:
        lower[0] = max(lower[0], mean[0] - prior.trust_intercept)
        upper[0] = mean[0] + prior.trust_intercept
    if prior.trust_coef is not None:
        lower[1:] = np.maximum(lower[1:], mean[1:] - prior.trust_coef)
        upper[1:] = mean[1:] + prior.trust_coef
    upper = np.maximum(upper, lower)  # empty box after the >= 0 cut: pin at lower
    start = np.clip(mean, lower, upper)
    return lower, upper, start


def fit_field_recalibrator(
    X: ArrayLike,
    y: ArrayLike,
    prior: RecalibrationPrior,
    *,
    weights: ArrayLike | None = None,
    min_labeled: int = 20,
    min_per_class: int = 3,
    current: FieldRecalibration | None = None,
) -> FieldRecalibration:
    """MAP logistic fit with Platt targets, Gaussian prior and bounds.

    Objective: ``-sum w_i [t_i log s_i + (1-t_i) log(1-s_i)] + 0.5 sum
    ((b-mu)/sd)^2``. Hard labels get Platt targets (counts unweighted);
    fractional labels in (0, 1) are used as given. When the data are
    insufficient or the optimizer fails, ``current`` (or the prior model) is
    returned with status ``skipped_insufficient`` / ``fallback_prior``.
    """
    n_features = len(prior.mean_coef)
    matrix, labels, w = _validate(X, y, weights, n_features)
    n_labeled = int(labels.shape[0])
    n_pos = int((labels > 0.5).sum())
    n_neg = n_labeled - n_pos

    def fallback(status: Status, message: str) -> FieldRecalibration:
        base = current if current is not None else prior.as_recalibration()
        return replace(
            base,
            status=status,
            n_labeled=n_labeled,
            n_pos=n_pos,
            n_neg=n_neg,
            message=message,
        )

    if n_labeled < min_labeled or n_pos < min_per_class or n_neg < min_per_class:
        return fallback(
            "skipped_insufficient",
            f"insufficient labels: n={n_labeled} (min {min_labeled}), "
            f"pos={n_pos}, neg={n_neg} (min {min_per_class} each)",
        )

    hard_pos = labels == 1.0
    hard_neg = labels == 0.0
    n_hard_pos = int(hard_pos.sum())
    n_hard_neg = int(hard_neg.sum())
    target = labels.copy()
    target[hard_pos] = (n_hard_pos + 1) / (n_hard_pos + 2)
    target[hard_neg] = 1 / (n_hard_neg + 2)

    mean = np.asarray([prior.mean_intercept, *prior.mean_coef], dtype=np.float64)
    inv_var = 1.0 / np.asarray([prior.sd_intercept, *prior.sd_coef]) ** 2
    lower, upper, start = _bounds(prior)

    def objective(theta: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        eta = theta[0] + matrix @ theta[1:]
        # -[t log s + (1-t) log(1-s)] = softplus(eta) - t * eta
        loss = float(np.sum(w * (np.logaddexp(0.0, eta) - target * eta)))
        resid = w * (expit(eta) - target)
        grad = np.empty_like(theta)
        grad[0] = resid.sum()
        grad[1:] = matrix.T @ resid
        diff = theta - mean
        loss += 0.5 * float(np.sum(diff * diff * inv_var))
        grad += diff * inv_var
        return loss, grad

    bounds = list(zip(lower.tolist(), upper.tolist(), strict=True))
    try:
        result = minimize(
            objective,
            start,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
        )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        return fallback(
            "fallback_prior", f"optimizer raised {type(exc).__name__}: {exc}"
        )

    theta = np.asarray(result.x, dtype=np.float64)
    if not np.isfinite(theta).all() or not np.isfinite(result.fun):
        return fallback("fallback_prior", "optimizer returned non-finite values")
    if not result.success:
        # L-BFGS-B can flag line-search precision loss at the optimum itself.
        _, grad = objective(theta)
        free = ~(((theta <= lower) & (grad > 0)) | ((theta >= upper) & (grad < 0)))
        if float(np.max(np.abs(grad[free]), initial=0.0)) > 1e-4:
            return fallback(
                "fallback_prior", f"optimizer did not converge: {result.message}"
            )

    names = prior.feature_names or tuple(f"x{j}" for j in range(n_features))
    return FieldRecalibration(
        intercept=float(theta[0]),
        coef=tuple(float(c) for c in theta[1:]),
        feature_names=names,
        status="fitted",
        n_labeled=n_labeled,
        n_pos=n_pos,
        n_neg=n_neg,
        message="",
    )


def stratum_weight_parts(
    strata: Sequence[str],
    stratum_sizes: Mapping[str, float],
    *,
    max_ratio: float = 10.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Raw ``N_h / n_h`` and ratio-capped weights, one per item of ``strata``.

    ``n_h`` is the count of each stratum in ``strata`` (the caller passes *every*
    reviewed item, not only the labelled ones) and ``N_h`` comes from
    ``stratum_sizes``. The cap is ``max_ratio * min_h w_h`` over the strata that
    occur in ``strata``, so ``max / min <= max_ratio`` always holds.
    Strata and keys must be ``str``.
    """
    if not (math.isfinite(max_ratio) and max_ratio >= 1.0):
        raise ValueError("max_ratio must be finite and >= 1")
    labels = list(strata)
    if any(not isinstance(h, str) for h in labels) or any(
        not isinstance(k, str) for k in stratum_sizes
    ):
        raise ValueError("strata and stratum_sizes keys must be str")
    if not labels:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    counts = Counter(labels)
    missing = sorted(h for h in counts if h not in stratum_sizes)
    if missing:
        raise KeyError(f"Strata without a size: {missing}")
    per_stratum = {h: float(stratum_sizes[h]) / n for h, n in counts.items()}
    if not all(math.isfinite(w) and w > 0 for w in per_stratum.values()):
        raise ValueError("Stratum sizes must be finite and > 0")
    cap = max_ratio * min(per_stratum.values())
    raw = np.asarray([per_stratum[h] for h in labels], dtype=np.float64)
    return raw, np.minimum(raw, cap)


def stratum_weights(
    strata: Sequence[str],
    stratum_sizes: Mapping[str, float],
    *,
    max_ratio: float = 10.0,
    mask: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Inverse-inclusion weights with a bounded max/min ratio, mean 1.

    ``w_h = N_h / n_h`` capped at ``max_ratio * min_h w_h`` (see
    :func:`stratum_weight_parts`). When ``mask`` (boolean, aligned with
    ``strata``) is given, the weights are computed over all items but only the
    masked ones (the holdout) are returned, normalized to mean 1.
    """
    _, capped = stratum_weight_parts(strata, stratum_sizes, max_ratio=max_ratio)
    if mask is not None:
        keep = np.asarray(mask)
        if keep.dtype != bool or keep.shape != capped.shape:
            raise ValueError("mask must be a boolean array aligned with strata")
        capped = capped[keep]
    if capped.size == 0:
        return capped
    return np.asarray(capped / float(capped.mean()), dtype=np.float64)
