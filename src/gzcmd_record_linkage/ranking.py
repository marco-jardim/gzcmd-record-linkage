"""Optional review-queue ordering, independent of policy and budget allocation.

Public helpers:

* ``order_by_score``, ``rrf``, ``interleave``, ``ThompsonQueue`` and
  ``review_queue`` order items for human review;
* ``top_k_cut`` marks which queue positions a reviewer examines under a budget;
* ``effort_curve``, ``reviews_to_target`` and ``recovered_at_budget`` measure a
  queue against known labels, under a perfect-reviewer assumption.

Score ties are broken by the thesis rule by default (``tie_break="index"``):
lower item position, or lower id when ``tie_ids`` is given, comes first. The
earlier seeded random tie permutation remains available as
``tie_break="random"``.

Nothing here is wired into the policy engine, configuration or CLI. No fusion
(for example the reciprocal-rank fusion called F4 in the thesis) is a
recommended default: the thesis selected it post hoc, on its own data.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

InputKind = Literal["orders", "scores"]
TieBreak = Literal["index", "random"]
QueueTieBreak = Literal["pair_id", "index", "random"]
Order = NDArray[np.int64]

EXAMINED = "examined"
NOT_EXAMINED = "not_examined"


def validate_order(order: ArrayLike, n: int) -> None:
    """Raise ValueError unless order is an integer permutation of range(n)."""
    values = np.asarray(order)
    if (
        not isinstance(n, (int, np.integer))
        or isinstance(n, (bool, np.bool_))
        or n < 0
        or values.ndim != 1
        or len(values) != n
        or (values.size and values.dtype.kind not in "iu")
        or not np.array_equal(np.sort(values), np.arange(n))
    ):
        raise ValueError("order must be a complete integer permutation of range(n)")


def _priority(
    n: int, tie_break: str, seed: int, tie_ids: ArrayLike | None
) -> NDArray[np.int64]:
    """Return per-item tie priority (lower first) for lexsort((priority, -score))."""
    if tie_break not in {"index", "random"}:
        raise ValueError("tie_break must be index or random")
    if tie_break == "random":
        if tie_ids is not None:
            raise ValueError("tie_ids requires tie_break='index'")
        return np.argsort(np.random.default_rng(seed).permutation(n))
    if tie_ids is None:
        return np.arange(n)
    ids = pd.Series(np.asarray(tie_ids))
    if ids.ndim != 1 or len(ids) != n:
        raise ValueError("tie_ids must have one id per item")
    if ids.isna().any():
        raise ValueError("tie_ids must not contain missing values")
    # Ranking handles numbers and strings alike; position resolves duplicate ids.
    ranks = ids.rank(method="min").to_numpy()
    return np.argsort(np.lexsort((np.arange(n), ranks)))


def _finite_scores(vector: ArrayLike) -> NDArray[np.float64]:
    scores = np.asarray(vector, dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    return scores


def order_by_score(
    scores: ArrayLike,
    *,
    tie_break: TieBreak = "index",
    seed: int = 42,
    tie_ids: ArrayLike | None = None,
) -> Order:
    """Return positional item IDs ordered from the largest to the smallest score.

    tie_break="index" (default, the thesis rule) orders equal scores by
    ascending position, as np.lexsort((np.arange(n), -score)); with tie_ids,
    equal scores follow ascending ids instead (position settles duplicate ids).
    tie_break="random" orders equal scores by a permutation fixed by seed and
    does not accept tie_ids. Neither rule looks at labels. Scores must be
    one-dimensional and finite.
    """
    values = np.asarray(scores)
    if values.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    finite = _finite_scores(values)
    priority = _priority(len(finite), tie_break, seed, tie_ids)
    return np.lexsort((priority, -finite)).astype(np.int64)


def _orders(
    rankings: Sequence[ArrayLike],
    input_kind: InputKind,
    seed: int,
    tie_break: TieBreak = "index",
    tie_ids: ArrayLike | None = None,
) -> list[Order]:
    if input_kind not in {"orders", "scores"}:
        raise ValueError("input_kind must be orders or scores")
    if len(rankings) == 0:
        raise ValueError("at least one ranking is required")
    vectors = [np.asarray(ranking) for ranking in rankings]
    if any(vector.ndim != 1 for vector in vectors):
        raise ValueError("rankings must be one-dimensional")
    n = len(vectors[0])
    if any(len(vector) != n for vector in vectors):
        raise ValueError("rankings must cover the same items")
    priority = _priority(n, tie_break, seed, tie_ids)
    orders = []
    for vector in vectors:
        if input_kind == "orders":
            validate_order(vector, n)
            orders.append(vector.astype(np.int64, copy=True))
        else:
            scores = _finite_scores(vector)
            orders.append(np.lexsort((priority, -scores)).astype(np.int64))
    return orders


def rrf(
    rankings: Sequence[ArrayLike],
    k: float = 60,
    *,
    seed: int = 42,
    input_kind: InputKind = "orders",
    tie_break: TieBreak = "index",
    tie_ids: ArrayLike | None = None,
) -> Order:
    """Fuse complete orders by sum(1 / (k + rank)), using one-based ranks.

    Orders contain positional item IDs 0..n-1, best first. For score vectors
    aligned to those IDs, pass input_kind="scores"; larger scores rank first.
    Explicit input_kind avoids ambiguity with integer-valued scores. Score and
    fusion ties follow tie_break: "index" (default, thesis rule) uses ascending
    item position, or ascending tie_ids when given; "random" uses item priority
    after a seed-fixed random permutation, never the labels. Reproducibility
    assumes fixed item IDs. k must be finite and nonnegative. Inputs are not
    modified.
    """
    if not np.isfinite(k) or k < 0:
        raise ValueError("k must be finite and nonnegative")
    orders = _orders(rankings, input_kind, seed, tie_break, tie_ids)
    if len(orders) == 1:
        # Preserve the exact input order even if very large k erases float gaps.
        return orders[0].copy()
    n = len(orders[0])
    fused = np.zeros(n)
    for order in orders:
        fused[order] += 1.0 / (k + np.arange(1, n + 1))
    priority = _priority(n, tie_break, seed, tie_ids)
    return np.lexsort((priority, -fused)).astype(np.int64)


def interleave(
    rankings: Sequence[ArrayLike],
    *,
    seed: int = 42,
    input_kind: InputKind = "orders",
    tie_break: TieBreak = "index",
    tie_ids: ArrayLike | None = None,
) -> Order:
    """Alternate arms in input order, skipping globally emitted items.

    Each arm emits its next unseen item on its turn. Input semantics and score
    tie-breaking (tie_break, tie_ids) match rrf; explicit orders preserve their
    supplied tie order.
    """
    orders = _orders(rankings, input_kind, seed, tie_break, tie_ids)
    n = len(orders[0])
    seen = np.zeros(n, dtype=bool)
    pointers = [0] * len(orders)
    result = []
    while len(result) < n:
        for arm, order in enumerate(orders):
            while pointers[arm] < n and seen[order[pointers[arm]]]:
                pointers[arm] += 1
            if pointers[arm] < n:
                item = int(order[pointers[arm]])
                seen[item] = True
                result.append(item)
                pointers[arm] += 1
    return np.asarray(result, dtype=np.int64)


class ThompsonQueue:
    """Stateful, globally deduplicated queue using Beta-Bernoulli sampling.

    next() reserves an unseen item from the sampled arm, or raises StopIteration.
    update(item, label) rewards only that item's selected arm, exactly once.
    label is binary (1 = confirmed link). Feedback may arrive out of order;
    window retains the last window received rewards *per arm*, or all rewards
    if None. Unlabelled reservations do not update the posterior. The queue
    never accesses a label column, performs I/O, or changes a policy decision.
    Input semantics match rrf, except that score ties always use the seeded
    random permutation (tie_break="random"). All randomness is local to the
    supplied seed.
    """

    def __init__(
        self,
        rankings: Sequence[ArrayLike],
        window: int | None = 200,
        prior: tuple[float, float] = (1, 1),
        seed: int = 42,
        *,
        input_kind: InputKind = "orders",
    ) -> None:
        if window is not None and (
            isinstance(window, bool) or not isinstance(window, int) or window <= 0
        ):
            raise ValueError("window must be a positive integer or None")
        if len(prior) != 2 or not np.isfinite(prior).all() or min(prior) <= 0:
            raise ValueError("prior must contain two finite positive parameters")
        self._orders = _orders(rankings, input_kind, seed, "random")
        self._rng = np.random.default_rng(seed)
        self._window = window
        self._prior = prior
        self._seen = np.zeros(len(self._orders[0]), dtype=bool)
        self._pointers = [0] * len(self._orders)
        self._pending: dict[int, int] = {}
        self._history: list[deque[int]] = [deque() for _ in self._orders]
        self._successes = np.zeros(len(self._orders), dtype=np.int64)
        self._trials = np.zeros(len(self._orders), dtype=np.int64)
        self._issued = 0

    def next(self) -> int:
        """Reserve the next unique positional item ID; raise on exhaustion."""
        if self._issued == len(self._seen):
            raise StopIteration
        samples = self._rng.beta(
            self._prior[0] + self._successes,
            self._prior[1] + self._trials - self._successes,
        )
        arm = int(np.argmax(samples))
        order = self._orders[arm]
        while self._seen[order[self._pointers[arm]]]:
            self._pointers[arm] += 1
        item = int(order[self._pointers[arm]])
        self._pointers[arm] += 1
        self._seen[item] = True
        self._issued += 1
        self._pending[item] = arm
        return item

    def update(self, item: int, label: int) -> None:
        """Apply binary feedback to a reserved item; reject duplicate updates."""
        if not isinstance(item, (int, np.integer)) or isinstance(item, bool):
            raise ValueError("item must be a reserved integer ID")
        if item not in self._pending:
            raise ValueError("item was not reserved or was already updated")
        if not isinstance(label, (int, np.integer, bool)) or label not in (0, 1):
            raise ValueError("label must be binary (0 or 1)")
        arm = self._pending.pop(item)
        reward = int(label)
        self._successes[arm] += reward
        self._trials[arm] += 1
        if self._window is not None:
            history = self._history[arm]
            history.append(reward)
            if len(history) > self._window:
                self._successes[arm] -= history.popleft()
                self._trials[arm] -= 1


def review_queue(
    df: pd.DataFrame,
    scores: Sequence[str] = (),
    method: Literal["evr", "rrf", "interleave"] = "rrf",
    *,
    seed: int = 42,
    k: float = 60,
    tie_break: QueueTieBreak = "pair_id",
    id_column: str = "pair_id",
) -> pd.DataFrame:
    """Return an ordered copy of an already selected human-review subset.

    scores names numeric columns (larger = earlier review). The caller filters
    below its cutoff; this function neither selects pairs nor changes actions,
    budgets, columns or index labels. Positional IDs also support duplicate
    DataFrame index labels. method="evr" is a passthrough: preserve the existing
    policy/caller order byte-for-byte on serialization, without recomputing EVR.

    Ties follow tie_break: "pair_id" (default, thesis rule) sorts by ascending
    id_column, falling back to row position when that column is absent; "index"
    uses row position; "random" uses a permutation fixed by seed.
    No configuration or CLI integration is implied by this standalone utility.
    """
    if tie_break not in {"pair_id", "index", "random"}:
        raise ValueError("tie_break must be pair_id, index or random")
    if method == "evr":
        return df.copy()
    if method not in {"rrf", "interleave"}:
        raise ValueError("method must be evr, rrf or interleave")
    rankings = [df[column].to_numpy(dtype=float) for column in scores]
    mode: TieBreak = "random" if tie_break == "random" else "index"
    ids = None
    if tie_break == "pair_id" and id_column in df.columns:
        ids = df[id_column].to_numpy()
    if method == "rrf":
        order = rrf(
            rankings, k=k, seed=seed, input_kind="scores", tie_break=mode, tie_ids=ids
        )
    else:
        order = interleave(
            rankings, seed=seed, input_kind="scores", tie_break=mode, tie_ids=ids
        )
    return df.iloc[order].copy()


def top_k_cut(
    order_or_df: ArrayLike | pd.DataFrame, k: int
) -> NDArray[np.object_] | pd.Series:
    """Label the first k queue positions "examined" and the rest "not_examined".

    The status is never a match decision: items below the stopping point were
    not examined, which is different from being non-matches, and must not be
    recorded as such. A k beyond the queue length examines everything.

    For an order (positional item IDs, best first) returns an array of statuses
    aligned to item IDs, so status[item] describes that item. For a DataFrame,
    taken as an already ordered queue, returns a Series aligned to its rows
    and index.
    """
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 0:
        raise ValueError("k must be a nonnegative integer")
    if isinstance(order_or_df, pd.DataFrame):
        status = np.full(len(order_or_df), NOT_EXAMINED, dtype=object)
        status[:k] = EXAMINED
        return pd.Series(status, index=order_or_df.index, name="review_status")
    order = np.asarray(order_or_df)
    validate_order(order, len(order))
    status = np.full(len(order), NOT_EXAMINED, dtype=object)
    if order.size:
        status[order[:k]] = EXAMINED
    return status


def effort_curve(order: ArrayLike, y: ArrayLike) -> pd.Series:
    """True pairs recovered after each review, following order.

    y is positional (values above zero count as true pairs). Assumes a perfect
    reviewer: every true pair reviewed is recovered. The Series index is the
    number of reviews (1, 2, ...).
    """
    labels = np.asarray(y) > 0
    order = np.asarray(order)
    validate_order(order, len(labels))
    cumulative = np.cumsum(labels[order].astype(int))
    index = pd.RangeIndex(1, len(labels) + 1, name="revisões")
    return pd.Series(cumulative, index=index, name="pares verdadeiros recuperados")


def reviews_to_target(curve: pd.Series | ArrayLike, target: int) -> int:
    """Smallest number of reviews that recovers target true pairs.

    Assumes a perfect reviewer. Raises ValueError if target is below one or
    the queue never recovers that many pairs.
    """
    values = np.asarray(curve)
    if target < 1:
        raise ValueError("a meta precisa ser de pelo menos um par verdadeiro")
    if len(values) == 0 or target > values[-1]:
        reach = int(values[-1]) if len(values) else 0
        raise ValueError(f"meta {target} inalcançável: a fila recupera {reach}")
    return int(np.searchsorted(values, target, side="left") + 1)


def recovered_at_budget(curve: pd.Series | ArrayLike, k: int) -> int:
    """True pairs recovered with k reviews (k beyond the queue: all of them).

    Assumes a perfect reviewer. Raises ValueError for a negative budget.
    """
    values = np.asarray(curve)
    if k < 0:
        raise ValueError("a carga não pode ser negativa")
    if k == 0 or len(values) == 0:
        return 0
    return int(values[min(k, len(values)) - 1])
