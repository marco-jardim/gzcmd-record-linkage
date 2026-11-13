"""Sequential arrangement (ASR): nested REFREC calibration below a fixed cut.

Faithful port of the generic parts of ``gzcmd/arranjo_corte.py`` of the thesis
research repository. ASR is the arrangement that *keeps the OpenRecLink (ORL)
cut* (pairs at or above the per-step cutoff stay accepted as ORL decided) and
decides the potential pairs *below* it with the v3.1 four-action policy applied
to a probability recalibrated on the below-cut pairs only (isotonic
regression, the principal arm ``P``). ``P-Platt`` (Platt 1999 smoothed targets
on clipped XGBoost logits) is a **sensitivity analysis**, not the principal
arm; ``A3`` (classifier restricted to below-cut pairs) and ``B1`` (isotonic on
the entire base) are the other registered arms.

Like ``oof`` this is a thesis evaluation artifact: it is supervised (labels
train the classifier and the calibrators), cross-fitted by REFREC and not a
deployable model. The thesis defaults are kept everywhere (cutoffs, partition
check, model, removed dossier fields); each is exposed as a parameter or as a
``THESIS_*`` constant. Reviewer (LLM) rates are estimated with the M3
estimator: UNSURE is marginal and never counted as FP or FN.

Left out on purpose (research-only): ``pending_cost`` (fixed token counts and
prices of the registered stop rule).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedGroupKFold

from gzcmd_record_linkage.calibration import PlattModel, predict_platt
from gzcmd_record_linkage.oof import N_SPLITS, SEED, thesis_classifier
from gzcmd_record_linkage.policy_engine import LLMError

THESIS_CUTOFFS: Mapping[int, float] = MappingProxyType(
    {6: 7.02408276169, 9: 7.84444860082, 11: 7.90191406713}
)
"""Preregistered ORL score cutoff of each ``PASSO`` (step) of the thesis."""

THESIS_PARTITION: tuple[int, int, int] = (211, 61485, 36)
"""Thesis partition under ``THESIS_CUTOFFS``: (accepted at/above the cut, below
the cut, below the cut with ``PAR == 2``). Pass it as ``expected`` to
``below_cutoff`` to assert it."""

THESIS_MODEL = "claude-haiku-5-5"
THESIS_PROMPT_VERSION = "v4"
THESIS_REMOVED_FIELDS: tuple[str, ...] = tuple(
    "model_outputs." + name
    for name in (
        "p_cal",
        "base_choice",
        "band",
        "evr",
        "guardrail",
        "guardrail_reason",
    )
)
"""Dossier fields removed from the reviewer's view in the thesis (cache key)."""

SUMMARY_COLUMNS = ["modo", "braco", "banda", "acao", "n"]
RATE_COLUMNS = ["modo", "fold", "banda", "e_fp", "e_fn", "e_unsure"]

GREY_BANDS = ("grey_low", "grey_mid", "grey_high", "near_high")

ModelFactory = Callable[[np.ndarray, int], Any]
Audit = Callable[[int, str, np.ndarray, np.ndarray, np.ndarray], None]


def sha256(path: str | Path) -> str:
    """Hex SHA-256 of the bytes of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def below_cutoff(
    raw: pd.DataFrame,
    cutoffs: Mapping[int, float] = THESIS_CUTOFFS,
    *,
    expected: tuple[int, int, int] | None = None,
) -> NDArray[np.bool_]:
    """Boolean mask of the pairs whose ``SCORE`` is below the cutoff of their ``PASSO``.

    ``raw`` holds the ORL columns ``PASSO``, ``SCORE`` (decimal comma accepted)
    and ``PAR``. ``cutoffs`` maps each step to its score cutoff (default
    ``THESIS_CUTOFFS``); a step without cutoff, or a missing score, raises
    ``ValueError``. With ``expected=(accepted, below, par2_below)`` the
    partition is asserted as the thesis does (default ``THESIS_PARTITION``
    counts are 211/61485/36): exactly ``accepted`` pairs at or above the cut,
    all of them (and only them) with ``PAR == 1``, and ``par2_below`` pairs
    with ``PAR == 2`` below the cut. ``None`` skips the check.
    """
    step = pd.to_numeric(raw["PASSO"], errors="raise")
    score = pd.to_numeric(
        raw["SCORE"].astype(str).str.replace(",", ".", regex=False), errors="raise"
    )
    cut = step.map(dict(cutoffs))
    if cut.isna().any() or score.isna().any():
        raise ValueError("Missing score or preregistered ORL step")
    below = score.lt(cut).to_numpy()
    if expected is not None:
        accepted, n_below, par2_below = expected
        par = pd.to_numeric(raw["PAR"], errors="coerce")
        if (
            int((~below).sum()) != accepted
            or int(below.sum()) != n_below
            or not par.loc[~below].eq(1).all()
            or not np.array_equal(par.eq(1).to_numpy(), ~below)
            or int(par.loc[below].eq(2).sum()) != par2_below
        ):
            raise ValueError(
                f"ORL partition differs from {accepted}/{n_below}/{par2_below}"
            )
    return below


class LogitPlatt:
    """Platt (1999) targets on clipped XGBoost logits; safeguarded optimization.

    ``P-Platt`` sensitivity calibrator (and the principal-arm fallback when a
    calibration set holds fewer than 10 ``PAR == 2``). Keeps the registered
    slope penalty .001, 100 iterations and tolerance 1e-10; the intercept is
    unpenalized. Newton with backtracking line search, falling back to BFGS:
    this changes the numerical optimization, not the logistic loss or its
    registered penalty. ``p`` is clipped to [1e-6, 1 - 1e-6] before the logit.
    """

    def __init__(self) -> None:
        self.model: PlattModel | None = None
        self.optimizer: str | None = None

    @staticmethod
    def design(p: ArrayLike) -> pd.Series:
        clipped = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return pd.Series(np.log(clipped / (1 - clipped)))

    def fit(self, p: ArrayLike, y: ArrayLike) -> LogitPlatt:
        from scipy.optimize import minimize
        from scipy.special import expit

        x = self.design(p).to_numpy()
        labels = np.asarray(y, dtype=int)
        if len(x) != len(labels) or set(np.unique(labels)) != {0, 1}:
            raise ValueError("Aligned binary labels required for Platt")
        if not np.isfinite(x).all():
            raise ValueError("Finite logits required for Platt")
        positive, negative = int(labels.sum()), int((labels == 0).sum())
        target = np.where(
            labels == 1, (positive + 1) / (positive + 2), 1 / (negative + 2)
        )
        design = np.column_stack([np.ones(len(x)), x])
        l2, max_iter, tol = 0.001, 100, 1e-10
        mean = float(target.mean())
        weights = np.array([np.log(mean / (1 - mean)), 0.0])

        def objective(w: np.ndarray) -> tuple[float, np.ndarray]:
            z = design @ w
            loss = float(np.sum(np.logaddexp(0, z) - target * z) + 0.5 * l2 * w[1] ** 2)
            gradient = design.T @ (expit(z) - target)
            gradient[1] += l2 * w[1]
            return loss, gradient

        def hessian(w: np.ndarray) -> np.ndarray:
            prediction = expit(design @ w)
            curvature = prediction * (1 - prediction)
            value = design.T @ (curvature[:, None] * design)
            value[1, 1] += l2
            return value

        converged = False
        for _ in range(max_iter):
            loss, gradient = objective(weights)
            if np.max(np.abs(gradient)) <= tol:
                converged = True
                break
            try:
                delta = np.linalg.solve(hessian(weights), gradient)
            except np.linalg.LinAlgError:
                break  # BFGS below does not invert a potentially singular Hessian.
            descent = float(gradient @ delta)
            if not np.isfinite(delta).all() or descent <= 0:
                break
            step = 1.0
            while step >= np.finfo(float).eps:
                candidate = weights - step * delta
                next_loss, _ = objective(candidate)
                if np.isfinite(next_loss) and next_loss <= loss - 1e-4 * step * descent:
                    weights = candidate
                    break
                step *= 0.5
            else:
                break
            if np.max(np.abs(step * delta)) <= tol:
                converged = True
                break
        self.optimizer = "newton_backtracking"
        if not converged:
            result = minimize(
                objective,
                weights,
                jac=True,
                method="BFGS",
                options={"gtol": tol, "maxiter": max_iter},
            )
            weights = result.x
            self.optimizer = "bfgs"
            # BFGS may report precision loss at the finite-precision optimum.
            # Accept that only if the Newton displacement meets the same tol.
            _, gradient = objective(weights)
            try:
                displacement = np.linalg.solve(hessian(weights), gradient)
            except np.linalg.LinAlgError as exc:
                raise RuntimeError(
                    "Platt optimizer has singular final curvature"
                ) from exc
            if not result.success and np.max(np.abs(displacement)) > tol:
                raise RuntimeError("Platt optimizer did not converge")
        if not np.isfinite(weights).all():
            raise RuntimeError("Platt optimizer returned nonfinite parameters")
        self.model = PlattModel(intercept=float(weights[0]), slope=float(weights[1]))
        return self

    def predict(self, p: ArrayLike) -> NDArray[np.float64]:
        if self.model is None:
            raise ValueError("Platt calibrator has not been fitted")
        return predict_platt(self.design(p), model=self.model).to_numpy()


def nested_cutoff_probabilities(
    x: pd.DataFrame,
    y: ArrayLike,
    below: ArrayLike,
    groups: ArrayLike,
    *,
    par2: ArrayLike | None = None,
    n_splits: int = N_SPLITS,
    seed: int = SEED,
    model_factory: ModelFactory | None = None,
    audit: Audit | None = None,
    include_a3: bool = True,
    include_platt: bool = True,
) -> tuple[dict[str, NDArray[np.float64]], NDArray[np.int64], list[dict[str, Any]]]:
    """Out-of-fold probabilities of the ASR arms, the outer fold and fold metadata.

    Reuses the full-data ``StratifiedGroupKFold`` 5x5 splits grouped by
    ``groups`` (REFREC); only the calibration membership changes for ``P``.
    ``y`` is binary, ``below`` flags the below-cutoff pairs (``below_cutoff``)
    and ``par2`` flags the ``PAR == 2`` pairs (default: ``y``). ``x`` is a
    DataFrame indexed positionally; ``model_factory(y_fit, seed)`` defaults to
    ``oof.thesis_classifier`` (the thesis XGBoost).

    Arms (keys of the returned mapping): ``P`` (principal: isotonic fitted on
    the below-cut inner OOF probabilities of the outer-training fold; Platt
    instead when the fold has fewer than 10 ``par2`` below the cut, recorded in
    the metadata ``principal``), ``B1`` (isotonic on the entire outer-training
    fold, i.e. the original REFREC probability), ``P-Platt`` (sensitivity,
    ``include_platt``) and ``A3`` (sensitivity: the classifier itself is fitted
    on below-cut pairs only, ``include_a3``; split construction is unchanged).
    With an all-true ``below`` and ``include_platt=False`` ``P`` equals
    ``oof.nested_oof_probabilities(split_by="refrec")`` exactly.

    ``audit(fold, stage, fit_indices, prediction_indices, outer_test)`` exposes
    the actual fitting boundary (stages ``inner``, ``A3_inner``, ``calibrator``,
    ``outer_and_calibrator``, ``A3_calibrator``).
    """
    factory = model_factory or thesis_classifier
    labels = np.asarray(y, dtype=int)
    below_mask = np.asarray(below, dtype=bool)
    group_values = np.asarray(groups)
    par2_mask = labels.astype(bool) if par2 is None else np.asarray(par2, dtype=bool)
    if not (
        len(x) == len(labels) == len(below_mask) == len(group_values) == len(par2_mask)
    ):
        raise ValueError("Aligned arrays required")
    if set(np.unique(labels)) != {0, 1} or pd.isna(group_values).any():
        raise ValueError("Binary labels and nonmissing groups required")
    arms = (
        ["P", "B1"]
        + (["P-Platt"] if include_platt else [])
        + (["A3"] if include_a3 else [])
    )
    probabilities = {arm: np.full(len(labels), np.nan) for arm in arms}
    folds = np.full(len(labels), -1, dtype=int)
    metadata: list[dict[str, Any]] = []
    outer = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
    for fold, (train, test) in enumerate(outer.split(x, labels, group_values)):
        if not set(group_values[train]).isdisjoint(group_values[test]):
            raise ValueError("Outer REFREC leakage")
        inner_probs = np.full(len(train), np.nan)
        a3_probs = np.full(len(train), np.nan)
        inner = StratifiedGroupKFold(
            n_splits, shuffle=True, random_state=seed + fold + 1
        )
        for inner_train, inner_test in inner.split(
            x.iloc[train], labels[train], group_values[train]
        ):
            fit_idx, pred_idx = train[inner_train], train[inner_test]
            if not set(group_values[fit_idx]).isdisjoint(group_values[pred_idx]):
                raise ValueError("Inner REFREC leakage")
            model = factory(labels[fit_idx], seed)
            model.fit(x.iloc[fit_idx], labels[fit_idx])
            inner_probs[inner_test] = model.predict_proba(x.iloc[pred_idx])[:, 1]
            if audit is not None:
                audit(fold, "inner", fit_idx.copy(), pred_idx.copy(), test.copy())
            if include_a3:
                restricted = fit_idx[below_mask[fit_idx]]
                model3 = factory(labels[restricted], seed)
                model3.fit(x.iloc[restricted], labels[restricted])
                a3_probs[inner_test] = model3.predict_proba(x.iloc[pred_idx])[:, 1]
                if audit is not None:
                    audit(
                        fold,
                        "A3_inner",
                        restricted.copy(),
                        pred_idx.copy(),
                        test.copy(),
                    )
        if not np.isfinite(inner_probs).all():
            raise ValueError("Incomplete inner OOF")
        eligible = below_mask[train]
        calibration = train[eligible]
        if len(np.unique(labels[calibration])) != 2:
            raise ValueError("Below-cutoff calibration requires both classes")
        fallback = int(par2_mask[calibration].sum()) < 10
        model = factory(labels[train], seed)
        model.fit(x.iloc[train], labels[train])
        raw_test = model.predict_proba(x.iloc[test])[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip").fit(
            inner_probs[eligible], labels[calibration]
        )
        platt = (
            LogitPlatt().fit(inner_probs[eligible], labels[calibration])
            if include_platt or fallback
            else None
        )
        if fallback:
            if platt is None:
                raise RuntimeError("Principal fallback requires Platt")
            probabilities["P"][test] = platt.predict(raw_test)
        else:
            probabilities["P"][test] = iso.predict(raw_test)
        if include_platt:
            if platt is None:
                raise RuntimeError("P-Platt requires its calibrator")
            probabilities["P-Platt"][test] = platt.predict(raw_test)
        probabilities["B1"][test] = (
            IsotonicRegression(out_of_bounds="clip")
            .fit(inner_probs, labels[train])
            .predict(raw_test)
        )
        if audit is not None:
            audit(fold, "calibrator", calibration.copy(), test.copy(), test.copy())
            audit(fold, "outer_and_calibrator", train.copy(), test.copy(), test.copy())
        if include_a3:
            model3 = factory(labels[calibration], seed)
            model3.fit(x.iloc[calibration], labels[calibration])
            raw3 = model3.predict_proba(x.iloc[test])[:, 1]
            cal3: Any = (
                LogitPlatt() if fallback else IsotonicRegression(out_of_bounds="clip")
            )
            probabilities["A3"][test] = cal3.fit(
                a3_probs[eligible], labels[calibration]
            ).predict(raw3)
            if audit is not None:
                audit(
                    fold, "A3_calibrator", calibration.copy(), test.copy(), test.copy()
                )
        folds[test] = fold
        metadata.append(
            {
                "fold": fold,
                "calibration_n": len(calibration),
                "calibration_par2": int(par2_mask[calibration].sum()),
                "principal": "platt" if fallback else "isotonic",
            }
        )
    if (folds < 0).any() or any(
        not np.isfinite(p).all() for p in probabilities.values()
    ):
        raise ValueError("Incomplete outer OOF")
    return probabilities, folds, metadata


def estimate_rates(
    frame: pd.DataFrame, fixed: Mapping[str, Mapping[str, float]]
) -> dict[str, LLMError]:
    """M3 estimator of the reviewer rates per band; UNSURE is never FP or FN.

    ``frame`` has one row per reviewed pair with ``band``, ``y`` (0/1) and
    ``decision`` (``MATCH``/``NONMATCH``/``UNSURE``). ``e_fp`` is MATCH over the
    negatives, ``e_fn`` NONMATCH over the positives and ``e_unsure`` the share
    of UNSURE. A band with fewer than 10 rows (or a missing denominator) takes
    the rate of ``fixed[band]`` (keys ``e_fp``, ``e_fn`` and optionally
    ``e_unsure``, default 0).
    """
    result = {}
    for band, fallback in fixed.items():
        cell = frame.loc[frame.band.eq(band)]
        negative, positive = cell.y.eq(0), cell.y.eq(1)
        enough = len(cell) >= 10
        result[band] = LLMError(
            float((cell.decision.eq("MATCH") & negative).sum() / negative.sum())
            if enough and negative.any()
            else float(fallback["e_fp"]),
            float((cell.decision.eq("NONMATCH") & positive).sum() / positive.sum())
            if enough and positive.any()
            else float(fallback["e_fn"]),
            float(cell.decision.eq("UNSURE").mean())
            if enough
            else float(fallback.get("e_unsure", 0.0)),
        )
    return result


def crossfit_rates(
    frame: pd.DataFrame,
    fixed: Mapping[str, Mapping[str, float]],
    folds: Iterable[int],
    *,
    par2_grey: bool = False,
    grey_bands: Iterable[str] = GREY_BANDS,
) -> dict[int, dict[str, LLMError]]:
    """Reviewer rates of each fold estimated on the other folds (``frame.fold``).

    A singleton ``folds`` means the pooled M3 check (no exclusion); production
    requires the five REFREC folds. With ``par2_grey`` (sensitivity
    ``P-PAR2-grey``) ``e_fn`` of the ``grey_bands`` is re-estimated pooling
    those bands and only the ``PAR == 2`` pairs (``frame.par``), when at least
    10 grey rows and one such positive exist.
    """
    fold_list = list(folds)
    grey = set(grey_bands)
    result = {}
    for fold in fold_list:
        train = frame if len(fold_list) == 1 else frame.loc[frame.fold.ne(fold)]
        rates = estimate_rates(train, fixed)
        if par2_grey:
            pooled = train.loc[train.band.isin(grey)]
            positives = pooled.par.eq(2)
            if len(pooled) >= 10 and positives.any():
                fn = float(
                    (pooled.decision.eq("NONMATCH") & positives).sum() / positives.sum()
                )
                rates = {
                    b: replace(r, e_fn=fn) if b in grey else r for b, r in rates.items()
                }
        result[int(fold)] = rates
    return result


def cache_matches(
    identity: Mapping[str, Any],
    run: Mapping[str, Any],
    *,
    mode: str,
    pair_id: str,
    prompt_sha256: str,
    dossier_sha256: str,
    model: str = THESIS_MODEL,
    removed_fields: Iterable[str] = THESIS_REMOVED_FIELDS,
    prompt_version: str = THESIS_PROMPT_VERSION,
) -> bool:
    """No cache hit without the complete identity, including removed fields.

    Both the journal ``identity`` and the ``run`` must agree on mode, model,
    prompt hash and the dossier hash of ``pair_id``; the run must be complete,
    carry the expected ``prompt_version`` and exactly the ``removed_fields``.
    Defaults are the thesis values (``THESIS_MODEL``, ``THESIS_PROMPT_VERSION``,
    ``THESIS_REMOVED_FIELDS``).
    """
    return bool(
        identity.get("mode") == run.get("mode") == mode
        and identity.get("model") == run.get("model") == model
        and identity.get("prompt_sha256") == run.get("prompt_sha256") == prompt_sha256
        and identity.get("dossier_sha256", {}).get(pair_id) == dossier_sha256
        and run.get("dossier_sha256", {}).get(pair_id) == dossier_sha256
        and set(run.get("removed_fields", [])) == set(removed_fields)
        and run.get("complete") is True
        and run.get("prompt_version") == prompt_version
    )


def summarize_actions(out: pd.DataFrame, mode: str, arm: str) -> pd.DataFrame:
    """Counts of ``action`` per ``band`` for one mode and arm.

    Allowlist output: deliberately excludes labels and correctness metrics, so
    it can be produced before any label-based evaluation.
    """
    summary = (
        out.groupby(["band", "action"], observed=True).size().reset_index(name="n")
    )
    summary = summary.rename(columns={"band": "banda", "action": "acao"})
    summary["modo"], summary["braco"] = mode, arm
    return summary[SUMMARY_COLUMNS]
