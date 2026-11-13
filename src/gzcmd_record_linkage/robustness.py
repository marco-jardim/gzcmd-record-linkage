"""Grouped-bootstrap robustness of queue fusions, with exact thesis parity.

This module generalises ``scripts/fusao_robustez.py`` of the research
repository. It answers: how stable is the review effort of a fusion (for
example reciprocal-rank fusion, F4 in the thesis) against its base queues when
the sample is perturbed?

Thesis procedure, reproduced draw for draw:

* one ``np.random.default_rng(seed)`` (PCG64, thesis seed 20261107) serves all
  replicates, and each replicate consumes it even if it is later excluded;
* the resampling unit is the group (REFREC in the thesis). Groups are sorted
  and coded (``pd.factorize(..., sort=True)``); a replicate draws ``G`` group
  codes with replacement (``rng.integers(0, G, size=G)``);
* every candidate receives the multiplicity of its group. Copies are *not*
  deduplicated: each copy is one observation and counts as one review;
* each base queue is re-ranked on the resampled data (copies of one candidate
  are adjacent, because they share the rank of the original queue), then the
  fusions are recomputed from the re-ranked queues. Models are not refit: the
  scores and labels are fixed, so the result is conditional on them;
* targets of recovered true pairs follow a hybrid rule, see
  ``replicate_targets``. Replicates without any true pair are excluded;
* the effort of every queue is ``searchsorted(cumulative true pairs, target) +
  1`` (reviews under a perfect reviewer), differences are ``fusion - base`` and
  reductions ``100 * (base - fusion) / base``; intervals are the linear 2.5 and
  97.5 percentiles.

Tie order. The thesis lays candidates out in ascending pair-id hash order and
breaks every tie (base queues, RRF) by position, so ties follow the hash. Here
that is ``tie_ids`` (see ``fusion``: ``llm_dossier.make_pair_id``): copies of
one candidate share its id and are ordered by position, which is immaterial
because copies are identical. Without ``tie_ids`` ties follow the item
position, exactly as in the thesis when the data are already in hash order.

Only deterministic fusions (``interleave`` and ``rrf``) are bootstrapped, as in
the thesis, which does not bootstrap the adaptive Thompson fusions A1 and A2.
The bootstrap does not measure training, review or post hoc selection
uncertainty; the thesis reading is exploratory.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from gzcmd_record_linkage.fusion import (
    THESIS_TARGETS,
    FusionSpec,
    fuse,
)
from gzcmd_record_linkage.ranking import Order, order_by_score

THESIS_SEED = 20261107
"""Seed of the thesis bootstrap."""

THESIS_FRACTIONS: tuple[float, ...] = (0.25, 0.64, 1.0)
"""Fractions of the replicate's true pairs used when it is too small for the
absolute targets (one fraction per target)."""

THESIS_COMPARISONS: tuple[tuple[str, str], ...] = (
    ("F4", "O1"),
    ("F4", "O3"),
    ("F3", "O1"),
    ("F3", "O3"),
)
"""Thesis comparisons, as ``(fusion, base)``: effort of fusion minus base."""

THESIS_N_BOOT = 2000
"""Number of replicates of the thesis bootstrap."""

_QUANTILES = [0.025, 0.5, 0.975]


@dataclass(frozen=True)
class BootstrapResult:
    """Outcome of ``bootstrap_fusion_differences``.

    ``replicates`` is a long table with one row per valid replicate, slot and
    queue: ``replicate`` (0-based draw index among all ``n_boot`` draws),
    ``n_items`` (resampled candidates, copies included), ``n_positives``,
    ``full_target`` (True when the absolute targets applied), ``slot`` (0-based
    position in ``targets``), ``nominal_target``, ``target`` (the target used
    in that replicate), ``queue`` and ``reviews``.

    ``summary`` has one row per comparison and slot, see
    ``bootstrap_fusion_differences`` for the columns. ``n_excluded`` counts the
    replicates dropped for having no true pair.
    """

    replicates: pd.DataFrame
    summary: pd.DataFrame
    n_boot: int
    n_valid: int
    n_excluded: int
    seed: int


def replicate_targets(
    n_positives: int,
    targets: Sequence[int] = THESIS_TARGETS,
    fractions: Sequence[float] = THESIS_FRACTIONS,
) -> tuple[tuple[int, ...], bool]:
    """Targets of a replicate with ``n_positives`` true pairs, and the rule used.

    Hybrid rule: when the replicate has at least ``max(targets)`` true pairs
    (thesis: 36, the full recovery of the original data) the absolute
    ``targets`` apply; otherwise each target is replaced by
    ``ceil(fraction * n_positives)``, one fraction per target (thesis: 25%,
    64% and 100% of the replicate's true pairs). The rule changes the estimand
    between replicates: it is not a fixed-sensitivity analysis. Returns the
    targets and whether the absolute ones were used. ``n_positives`` must be
    at least one.
    """
    _check_targets(targets, fractions)
    if n_positives < 1:
        raise ValueError("n_positives must be at least one")
    if n_positives >= max(targets):
        return tuple(int(t) for t in targets), True
    return (
        tuple(int(np.ceil(f * n_positives)) for f in fractions),
        False,
    )


def _check_targets(targets: Sequence[int], fractions: Sequence[float]) -> None:
    if len(targets) == 0:
        raise ValueError("targets must not be empty")
    if len(fractions) != len(targets):
        raise ValueError("fractions must have one entry per target")
    if any(isinstance(t, bool) or int(t) != t or t < 1 for t in targets):
        raise ValueError("targets must be integers of at least one")
    if any(not 0 < f <= 1 for f in fractions):
        raise ValueError("fractions must lie in (0, 1]")


def _check_fusions(
    fusions: Mapping[str, FusionSpec], base_names: Sequence[str]
) -> None:
    for key, spec in fusions.items():
        if key in base_names:
            raise ValueError(f"fusion name {key!r} collides with a base queue")
        if spec.method == "thompson":
            raise ValueError(
                f"fusion {key!r} is adaptive (thompson): only deterministic "
                "fusions (interleave, rrf) are bootstrapped"
            )
        missing = [arm for arm in spec.arms if arm not in base_names]
        if missing:
            raise ValueError(f"fusion {key!r} needs missing queues: {missing}")


def _base_orders(
    scores: Mapping[str, ArrayLike], tie_ids: ArrayLike | None, n: int
) -> dict[str, Order]:
    if not scores:
        raise ValueError("at least one base queue is required")
    orders: dict[str, Order] = {}
    for name, vector in scores.items():
        values = np.asarray(vector)
        if values.ndim != 1 or len(values) != n:
            raise ValueError(f"scores[{name!r}] must have one score per item")
        orders[name] = order_by_score(values, tie_ids=tie_ids)
    return orders


def _positives(labels: ArrayLike) -> NDArray[np.bool_]:
    values = np.asarray(labels)
    if values.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    return values > 0


def _resample(
    orders: Mapping[str, Order], multiplicity: NDArray[np.int64]
) -> tuple[dict[str, Order], NDArray[np.int64]]:
    """Expand multiplicities into contiguous copies and re-rank every queue.

    Each copy keeps the rank of its original in every queue (copies of one
    candidate tie and are ordered by position); copies are not deduplicated.
    Returns the re-ranked orders over the copies and, per copy, the original
    item.
    """
    original = np.repeat(np.arange(len(multiplicity)), multiplicity)
    result: dict[str, Order] = {}
    for name, order in orders.items():
        ranks = np.empty(len(order), dtype=np.int64)
        ranks[order] = np.arange(len(order))
        result[name] = np.argsort(ranks[original], kind="stable").astype(np.int64)
    return result, original


def _all_orders(
    base: Mapping[str, Order],
    fusions: Mapping[str, FusionSpec],
    tie_ids: ArrayLike | None,
) -> dict[str, Order]:
    orders = dict(base)
    for name, spec in fusions.items():
        orders[name] = fuse(base, spec, tie_ids=tie_ids)
    return orders


def _reviews(
    orders: Mapping[str, Order],
    positives: NDArray[np.bool_],
    targets: Sequence[int],
) -> dict[str, NDArray[np.int64]]:
    needed = np.asarray(targets)
    if len(needed) == 0 or needed.min() < 1 or needed.max() > int(positives.sum()):
        raise ValueError("target not reachable: above the true pairs available")
    return {
        name: np.searchsorted(np.cumsum(positives[order].astype(int)), needed) + 1
        for name, order in orders.items()
    }


def _replicate_orders(
    base: Mapping[str, Order],
    fusions: Mapping[str, FusionSpec],
    multiplicity: NDArray[np.int64],
    tie_ids: NDArray[np.generic] | None,
) -> tuple[dict[str, Order], NDArray[np.int64]]:
    sampled, original = _resample(base, multiplicity)
    ids = None if tie_ids is None else tie_ids[original]
    return _all_orders(sampled, fusions, ids), original


def _check_comparisons(
    comparisons: Sequence[tuple[str, str]], names: Sequence[str]
) -> None:
    if len(comparisons) == 0:
        raise ValueError("at least one comparison is required")
    for fused, base in comparisons:
        for name in (fused, base):
            if name not in names:
                raise ValueError(f"comparison uses unknown queue {name!r}")


def _tie_array(tie_ids: ArrayLike | None, n: int) -> NDArray[np.generic] | None:
    if tie_ids is None:
        return None
    ids = np.asarray(tie_ids)
    if ids.ndim != 1 or len(ids) != n:
        raise ValueError("tie_ids must have one id per item")
    return ids


def _summarise(
    efforts: Sequence[Mapping[str, NDArray[np.int64]]],
    comparisons: Sequence[tuple[str, str]],
    targets: Sequence[int],
    fractions: Sequence[float],
) -> pd.DataFrame:
    joint: dict[str, NDArray[np.float64]] = {}
    for fused in dict.fromkeys(f for f, _ in comparisons):
        bases = [b for f, b in comparisons if f == fused]
        lower = np.array(
            [np.logical_and.reduce([e[fused] < e[b] for b in bases]) for e in efforts]
        )
        joint[fused] = lower.mean(axis=0)
    rows = []
    for fused, base in comparisons:
        fv = np.array([e[fused] for e in efforts])
        bv = np.array([e[base] for e in efforts])
        diff = fv - bv
        reduction = 100 * (bv - fv) / bv
        for j, target in enumerate(targets):
            row: dict[str, object] = {
                "comparison": f"{fused}-{base}",
                "fusion": fused,
                "base": base,
                "slot": j,
                "nominal_target": int(target),
                "fraction_if_below_full": float(fractions[j]),
                "valid_replicates": len(efforts),
                "p_fusion_lower": float((diff[:, j] < 0).mean()),
            }
            for name, values in (
                ("difference", diff[:, j]),
                ("reduction_percent", reduction[:, j]),
            ):
                lo, median, hi = np.quantile(values, _QUANTILES, method="linear")
                row[f"{name}_median"] = float(median)
                row[f"{name}_ci95_lower"] = float(lo)
                row[f"{name}_ci95_upper"] = float(hi)
            row["p_fusion_lower_than_all_bases"] = float(joint[fused][j])
            rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_fusion_differences(
    scores: Mapping[str, ArrayLike],
    labels: ArrayLike,
    groups: ArrayLike,
    *,
    fusions: Mapping[str, FusionSpec],
    comparisons: Sequence[tuple[str, str]],
    n_boot: int = THESIS_N_BOOT,
    seed: int = THESIS_SEED,
    targets: Sequence[int] = THESIS_TARGETS,
    fractions: Sequence[float] = THESIS_FRACTIONS,
    tie_ids: ArrayLike | None = None,
) -> BootstrapResult:
    """Grouped bootstrap of the review effort of fusions against base queues.

    ``scores`` maps base queue names to score vectors (larger first), one entry
    per item; ``labels`` marks true pairs (above zero); ``groups`` gives the
    resampling unit of each item (thesis: REFREC). ``fusions`` maps names to
    deterministic ``FusionSpec`` (``interleave`` or ``rrf``) whose arms are base
    queue names; thompson specs raise ValueError. ``comparisons`` lists
    ``(queue, other)`` pairs of base or fusion names, measured as ``queue``
    minus ``other``. ``tie_ids`` orders ties as the thesis does (see the module
    docstring); the default is the item position.

    Each of ``n_boot`` replicates resamples groups with replacement, gives each
    item its group's multiplicity, re-ranks the queues, recomputes the fusions
    and measures the reviews to reach the replicate's targets
    (``replicate_targets``: absolute ``targets`` if it has at least
    ``max(targets)`` true pairs, else ``ceil(fraction * positives)``).
    Replicates with no true pair are excluded (ValueError if all are).

    ``summary`` columns: ``comparison`` (``"F4-O1"``), ``fusion``, ``base``,
    ``slot``, ``nominal_target``, ``fraction_if_below_full``,
    ``valid_replicates``, ``p_fusion_lower`` (share of replicates with fewer
    reviews), then ``difference_*`` (reviews, queue minus other) and
    ``reduction_percent_*`` (``100 * (other - queue) / other``) each with
    ``_median``, ``_ci95_lower`` and ``_ci95_upper`` (linear 2.5/97.5
    percentiles), and ``p_fusion_lower_than_all_bases``: share of replicates in
    which the queue needs fewer reviews than every ``other`` it is compared with
    (the thesis reports it for F4 only; the F4 values are the thesis ones).
    """
    if isinstance(n_boot, bool) or int(n_boot) != n_boot or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    _check_targets(targets, fractions)
    positives = _positives(labels)
    n = len(positives)
    group_values = np.asarray(groups)
    if group_values.ndim != 1 or len(group_values) != n:
        raise ValueError("groups must have one entry per item")
    codes, uniques = pd.factorize(group_values, sort=True)
    if (codes < 0).any():
        raise ValueError("groups must not contain missing values")
    group_count = len(uniques)
    ties = _tie_array(tie_ids, n)
    base = _base_orders(scores, ties, n)
    _check_fusions(fusions, list(base))
    _check_comparisons(comparisons, [*base, *fusions])

    rng = np.random.default_rng(seed)
    efforts: list[dict[str, NDArray[np.int64]]] = []
    records: list[dict[str, object]] = []
    for replicate in range(n_boot):
        draws = rng.integers(0, group_count, size=group_count)
        counts = np.bincount(draws, minlength=group_count)
        orders, original = _replicate_orders(base, fusions, counts[codes], ties)
        sampled = positives[original]
        total = int(sampled.sum())
        if not total:
            continue
        used, full = replicate_targets(total, targets, fractions)
        effort = _reviews(orders, sampled, used)
        efforts.append(effort)
        for name, values in effort.items():
            for j, value in enumerate(values):
                records.append(
                    {
                        "replicate": replicate,
                        "n_items": len(original),
                        "n_positives": total,
                        "full_target": full,
                        "slot": j,
                        "nominal_target": int(targets[j]),
                        "target": used[j],
                        "queue": name,
                        "reviews": int(value),
                    }
                )
    if not efforts:
        raise ValueError("no replicate has a true pair")
    return BootstrapResult(
        replicates=pd.DataFrame(records),
        summary=_summarise(efforts, comparisons, targets, fractions),
        n_boot=int(n_boot),
        n_valid=len(efforts),
        n_excluded=int(n_boot) - len(efforts),
        seed=int(seed),
    )


def subset_effort(
    scores: Mapping[str, ArrayLike],
    labels: ArrayLike,
    mask: ArrayLike,
    *,
    fusions: Mapping[str, FusionSpec],
    targets: Sequence[int] = THESIS_TARGETS,
    baselines: Sequence[str] = (),
    tie_ids: ArrayLike | None = None,
) -> pd.DataFrame:
    """Review effort of every queue and fusion recomputed inside a subset.

    Generalises the thesis band analysis (``faixa_table``): queues and fusions
    are recomputed only on the items where ``mask`` is true (every item keeps
    its rank from the full queue, then the fusions are recomputed on the
    subset), not filtered after fusing. The thesis band mask is the rounded
    score cut ``np.round(score, 3) >= -5.648`` (2,522 items and 36 true pairs
    on its data); the mask is the caller's, since the cut is specific to that
    data. Deterministic fusions only (thompson raises ValueError).

    Columns: ``queue``, ``target``, ``reviews``, and for each name in
    ``baselines`` (base queues or fusions) ``difference_<name>`` (reviews minus
    the baseline's) and ``reduction_percent_<name>``
    (``100 * (1 - reviews / baseline)``). ValueError if a target exceeds the
    subset's true pairs.
    """
    _check_targets(targets, (1.0,) * len(targets))
    positives = _positives(labels)
    n = len(positives)
    selected = np.asarray(mask)
    if selected.ndim != 1 or len(selected) != n:
        raise ValueError("mask must have one entry per item")
    ties = _tie_array(tie_ids, n)
    base = _base_orders(scores, ties, n)
    _check_fusions(fusions, list(base))
    for name in baselines:
        if name not in (*base, *fusions):
            raise ValueError(f"unknown baseline queue {name!r}")
    orders, original = _replicate_orders(
        base, fusions, (selected > 0).astype(np.int64), ties
    )
    effort = _reviews(orders, positives[original], targets)
    rows = []
    for name in orders:
        for j, target in enumerate(targets):
            row: dict[str, object] = {
                "queue": name,
                "target": int(target),
                "reviews": int(effort[name][j]),
            }
            for other in baselines:
                row[f"difference_{other}"] = int(effort[name][j] - effort[other][j])
                row[f"reduction_percent_{other}"] = float(
                    100 * (1 - effort[name][j] / effort[other][j])
                )
            rows.append(row)
    return pd.DataFrame(rows)
