"""Thesis queue fusions and queue-comparison metrics, with exact thesis parity.

This module reproduces ``scripts/fusao_filas.py`` of the research repository:

* ``THESIS_FUSIONS`` holds the named fusions F1-F4 (deterministic) and A1/A2
  (adaptive Thompson sampling) as ``FusionSpec`` values;
* ``fuse`` builds one fused order, and ``simulate_thompson`` replays a
  Thompson queue over known labels (a batch simulation, not an operational
  queue: in operation use ``ranking.ThompsonQueue`` with ``update()``);
* ``queue_metrics`` and ``compare_queues`` measure queues against known labels
  under a perfect-reviewer assumption.

Orders are complete permutations of positional item IDs ``0..n-1``, best first
(see ``ranking.validate_order``). Labels are positional too: an item is a true
pair when its label is above zero.

Tie order. The thesis breaks score ties (individual queues, RRF) by ascending
pair-id hash, the first 16 hex digits of SHA-256 of ``COMPREC|REFREC|PASSO``.
That hash is what ``llm_dossier.make_pair_id`` computes (the package and the
thesis implementations are identical), so ``tie_ids`` is built as::

    tie_ids = [make_pair_id(c, r, p) for c, r, p in zip(comprec, refrec, passo)]

Without ``tie_ids`` ties follow the item position.

None of the named fusions is a recommended default: they reproduce the thesis
definitions, and F4 was selected post hoc among six variants without
multiplicity correction (``estudo4_gzcmd.tex``, lines 202 and 225).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from gzcmd_record_linkage.ranking import (
    Order,
    ThompsonQueue,
    TieBreak,
    effort_curve,
    interleave,
    recovered_at_budget,
    reviews_to_target,
    rrf,
)

FusionMethod = Literal["interleave", "rrf", "thompson"]

THESIS_TARGETS: tuple[int, ...] = (9, 23, 36)
"""Recovery targets reported by the thesis (the script also computes 30)."""

THESIS_TOP_K: tuple[int, ...] = (17, 58, 96, 125, 440, 2522, 3040)
"""Review budgets reported by the thesis."""

THESIS_AUC_HORIZON = 3040
"""Horizon of the thesis recovery-AUC (reviews)."""

_METHODS = ("interleave", "rrf", "thompson")


@dataclass(frozen=True)
class FusionSpec:
    """Definition of a named queue fusion.

    ``arms`` lists the base queues by name, in order: the order is the
    interleaving order, the RRF summation order and, for Thompson sampling, the
    tie order between sampled arms (the first arm wins). ``k`` applies to
    ``rrf``; ``window`` and ``prior`` apply to ``thompson`` (``window=None``
    keeps every reward of an arm).
    """

    name: str
    method: FusionMethod
    arms: tuple[str, ...]
    k: int = 60
    window: int | None = 200
    prior: tuple[float, float] = (1.0, 1.0)

    def __post_init__(self) -> None:
        if self.method not in _METHODS:
            raise ValueError("method must be interleave, rrf or thompson")
        object.__setattr__(self, "arms", tuple(self.arms))
        object.__setattr__(self, "prior", (self.prior[0], self.prior[1]))
        if not self.arms:
            raise ValueError("a fusion needs at least one arm")


THESIS_FUSIONS: Mapping[str, FusionSpec] = MappingProxyType(
    {
        "F1": FusionSpec("F1", "interleave", ("O3", "O1")),
        "F2": FusionSpec("F2", "interleave", ("O3", "O1", "O3_MACD")),
        "F3": FusionSpec("F3", "rrf", ("O3", "O1", "O3_MACD"), k=60),
        "F4": FusionSpec("F4", "rrf", ("O3", "O1"), k=60),
        "A1": FusionSpec("A1", "thompson", ("O1", "O3", "O3_MACD"), window=200),
        "A2": FusionSpec("A2", "thompson", ("O1", "O3", "O3_MACD"), window=None),
    }
)
"""The thesis fusions, read-only.

They reproduce the thesis definitions (F1/F2 interleaving, F3/F4 reciprocal
rank fusion with k=60, A1/A2 Thompson sampling with Beta(1, 1), window 200 or
none). F4 was selected post hoc among six variants without multiplicity
correction (``estudo4_gzcmd.tex``, lines 202 and 225): none is a recommended
default. Arm order matters (it is the tie order) and is copied from the thesis.
"""


def _arm_orders(orders: Mapping[str, Order], spec: FusionSpec) -> list[Order]:
    missing = [arm for arm in spec.arms if arm not in orders]
    if missing:
        raise ValueError(f"fusion {spec.name} needs missing queues: {missing}")
    return [orders[arm] for arm in spec.arms]


def _positives(labels: ArrayLike) -> np.ndarray:
    values = np.asarray(labels)
    if values.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    return values > 0


def simulate_thompson(
    orders: Sequence[ArrayLike],
    labels: ArrayLike,
    *,
    window: int | None = 200,
    prior: tuple[float, float] = (1.0, 1.0),
    seed: int = 0,
) -> Order:
    """Replay Thompson sampling over known labels and return the review order.

    Reproduces the thesis ``thompson()`` draw for draw: one
    ``np.random.default_rng(seed)``; every step draws one Beta sample per arm,
    ``Beta(prior[0] + successes, prior[1] + failures)``; the arm with the
    largest sample is chosen (ties favour the first arm); that arm proposes its
    next item not yet reviewed by any arm; only the chosen arm receives the
    reward (label above zero); with ``window`` each arm keeps its last
    ``window`` rewards, otherwise all of them.

    It is built on ``ranking.ThompsonQueue`` (``next()`` then ``update()``),
    which consumes the random stream identically: the thesis also advances the
    other arms' pointers, but only over items already reviewed, and a pointer
    only moves past items it would skip anyway, so the draws and the chosen
    items coincide. Reading labels makes this a simulation or evaluation tool;
    in operation call ``ThompsonQueue.update`` with the reviewers' answers.
    """
    positives = _positives(labels)
    queue = ThompsonQueue(orders, window=window, prior=prior, seed=seed)
    n = len(positives)
    if len(orders[0]) != n:
        raise ValueError("labels must have one entry per item")
    result = np.empty(n, dtype=np.int64)
    for step in range(n):
        item = queue.next()
        queue.update(item, int(positives[item]))
        result[step] = item
    return result


def fuse(
    orders: Mapping[str, Order],
    spec: FusionSpec,
    *,
    labels: ArrayLike | None = None,
    seed: int = 0,
    tie_ids: ArrayLike | None = None,
    tie_break: TieBreak = "index",
) -> Order:
    """Build the order of one fusion from named base orders.

    ``interleave`` and ``rrf`` call ``ranking.interleave`` / ``ranking.rrf``
    with ``input_kind="orders"``; ``tie_ids`` and ``tie_break`` decide RRF ties
    (the thesis rule is ascending pair-id hash: see the module docstring).
    ``thompson`` needs ``labels`` (ValueError otherwise) and runs
    ``simulate_thompson`` with ``seed``; tie settings do not apply to it, since
    the base orders are consumed as given. Missing arms raise ValueError.
    """
    arms = _arm_orders(orders, spec)
    if spec.method == "interleave":
        return interleave(
            arms, seed=seed, input_kind="orders", tie_break=tie_break, tie_ids=tie_ids
        )
    if spec.method == "rrf":
        return rrf(
            arms,
            spec.k,
            seed=seed,
            input_kind="orders",
            tie_break=tie_break,
            tie_ids=tie_ids,
        )
    if labels is None:
        raise ValueError(f"fusion {spec.name} (thompson) requires labels")
    return simulate_thompson(
        arms, labels, window=spec.window, prior=spec.prior, seed=seed
    )


def _metric_names(targets: Sequence[int], top_k: Sequence[int]) -> list[str]:
    return (
        [f"reviews_to_{n}" for n in targets]
        + [f"recovered_at_{k}" for k in top_k]
        + ["auc_recovery"]
    )


def queue_metrics(
    order: ArrayLike,
    labels: ArrayLike,
    *,
    targets: Sequence[int] = THESIS_TARGETS,
    top_k: Sequence[int] = (),
    auc_horizon: int | None = None,
) -> dict[str, int | float | None]:
    """Effort metrics of one queue under a perfect reviewer.

    Keys, in this order:

    * ``reviews_to_{n}`` for each target n: ``searchsorted(C, n) + 1`` with C
      the cumulative true pairs, i.e. the smallest number of reviews that
      recovers n. If the queue never recovers n (n above the positives), the
      value is ``None``. The thesis formula would return ``len(order) + 1``
      there, but the thesis script rejects such curves (it requires the full
      recovery of the 36 true pairs), so no thesis value exists to match;
      ``None`` avoids a fake review count. A target below one raises ValueError;
    * ``recovered_at_{k}`` for each k in ``top_k``: C(k), true pairs after k
      reviews (a k beyond the queue gives all of them, the thesis would fail);
    * ``auc_recovery``: ``sum_{k=0}^{H-1} C(k) / (H * P)`` with C(0) = 0 and P
      the total of true pairs: the step-function integral of recovered
      fraction, exactly the thesis ``counts[:H-1].sum() / (H * total)``. H is
      ``auc_horizon`` (default: the queue length; at most the queue length).
      ValueError if there are no true pairs.
    """
    curve = effort_curve(order, labels).to_numpy()
    n = len(curve)
    total = int(curve[-1]) if n else 0
    if total == 0:
        raise ValueError("labels contain no true pairs")
    horizon = n if auc_horizon is None else auc_horizon
    if isinstance(horizon, bool) or not 1 <= horizon <= n:
        raise ValueError("auc_horizon must be between 1 and the queue length")
    result: dict[str, int | float | None] = {}
    for target in targets:
        result[f"reviews_to_{target}"] = (
            None if target > total else reviews_to_target(curve, target)
        )
    for budget in top_k:
        result[f"recovered_at_{budget}"] = recovered_at_budget(curve, budget)
    result["auc_recovery"] = float(curve[: horizon - 1].sum() / (horizon * total))
    return result


def _as_floats(
    metrics: Mapping[str, int | float | None], names: Sequence[str]
) -> list[float]:
    return [np.nan if metrics[name] is None else float(metrics[name]) for name in names]


def _row(
    queue: str,
    kind: str,
    method: str | None,
    runs: list[list[float]],
    names: Sequence[str],
) -> dict[str, object]:
    values = np.array(runs)
    row: dict[str, object] = {
        "queue": queue,
        "kind": kind,
        "method": method,
        "runs": len(runs),
    }
    mean = values.mean(axis=0)
    p10 = np.quantile(values, 0.1, axis=0, method="linear")
    p90 = np.quantile(values, 0.9, axis=0, method="linear")
    for j, name in enumerate(names):
        row[name] = float(mean[j])
        row[f"{name}_p10"] = float(p10[j])
        row[f"{name}_p90"] = float(p90[j])
    return row


def compare_queues(
    orders: Mapping[str, Order],
    labels: ArrayLike,
    *,
    fusions: Mapping[str, FusionSpec] | None = None,
    targets: Sequence[int] = THESIS_TARGETS,
    top_k: Sequence[int] = (),
    auc_horizon: int | None = None,
    adaptive_seeds: Sequence[int] = range(200),
    tie_ids: ArrayLike | None = None,
) -> pd.DataFrame:
    """Compare base queues and fusions on known labels, one row per queue.

    Rows: every base queue in ``orders`` (kind ``base``), then every fusion of
    ``fusions`` (default ``THESIS_FUSIONS``; kind ``fusion``). Deterministic
    fusions (interleave, rrf) are computed once, with ``tie_ids`` deciding RRF
    ties. Thompson fusions are simulated once per seed in ``adaptive_seeds``
    (thesis: 0..199; ValueError if empty) and summarised as in the thesis: mean
    and linear P10/P90 (``np.quantile``) of each metric over the runs, not
    metrics of an averaged curve. Fusions need labels for Thompson only; all
    rows need them for evaluation.

    Columns: ``queue``, ``kind``, ``method`` (``None`` for base queues),
    ``runs`` (1 for deterministic rows), then for each metric name of
    ``queue_metrics`` the mean as ``<name>``, plus ``<name>_p10`` and
    ``<name>_p90`` (all three equal for a single run). An unreachable target
    is NaN and propagates to the summaries of its row.
    """
    specs = THESIS_FUSIONS if fusions is None else fusions
    names = _metric_names(targets, top_k)

    def measure(order: ArrayLike) -> list[float]:
        return _as_floats(
            queue_metrics(
                order, labels, targets=targets, top_k=top_k, auc_horizon=auc_horizon
            ),
            names,
        )

    rows = [
        _row(name, "base", None, [measure(order)], names)
        for name, order in orders.items()
    ]
    for key, spec in specs.items():
        if spec.method == "thompson":
            seeds = list(adaptive_seeds)
            if not seeds:
                raise ValueError("adaptive_seeds must not be empty")
            arms = _arm_orders(orders, spec)
            runs = [
                measure(
                    simulate_thompson(
                        arms, labels, window=spec.window, prior=spec.prior, seed=seed
                    )
                )
                for seed in seeds
            ]
        else:
            runs = [measure(fuse(orders, spec, tie_ids=tie_ids))]
        rows.append(_row(key, "fusion", spec.method, runs, names))
    return pd.DataFrame(rows)
