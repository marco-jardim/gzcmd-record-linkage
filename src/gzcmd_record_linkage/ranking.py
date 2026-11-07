"""Optional review-queue ordering, independent of policy and budget allocation."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

InputKind = Literal["orders", "scores"]
Order = NDArray[np.int64]


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


def _orders(
    rankings: Sequence[ArrayLike], input_kind: InputKind, seed: int
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
    tie_order = np.random.default_rng(seed).permutation(n)
    priority = np.argsort(tie_order)
    orders = []
    for vector in vectors:
        if input_kind == "orders":
            validate_order(vector, n)
            orders.append(vector.astype(np.int64, copy=True))
        else:
            scores = np.asarray(vector, dtype=float)
            if not np.isfinite(scores).all():
                raise ValueError("scores must be finite")
            orders.append(np.lexsort((priority, -scores)).astype(np.int64))
    return orders


def rrf(
    rankings: Sequence[ArrayLike],
    k: float = 60,
    *,
    seed: int = 42,
    input_kind: InputKind = "orders",
) -> Order:
    """Fuse complete orders by sum(1 / (k + rank)), using one-based ranks.

    Orders contain positional item IDs 0..n-1, best first. For score vectors
    aligned to those IDs, pass input_kind="scores"; larger scores rank first.
    Explicit input_kind avoids ambiguity with integer-valued scores. Score and
    fusion ties use item priority after a seed-fixed random permutation, never
    the original row order or labels. Reproducibility assumes fixed item IDs.
    k must be finite and nonnegative. Inputs are not modified.
    """
    if not np.isfinite(k) or k < 0:
        raise ValueError("k must be finite and nonnegative")
    orders = _orders(rankings, input_kind, seed)
    if len(orders) == 1:
        # Preserve the exact input order even if very large k erases float gaps.
        return orders[0].copy()
    n = len(orders[0])
    fused = np.zeros(n)
    for order in orders:
        fused[order] += 1.0 / (k + np.arange(1, n + 1))
    priority = np.argsort(np.random.default_rng(seed).permutation(n))
    return np.lexsort((priority, -fused)).astype(np.int64)


def interleave(
    rankings: Sequence[ArrayLike],
    *,
    seed: int = 42,
    input_kind: InputKind = "orders",
) -> Order:
    """Alternate arms in input order, skipping globally emitted items.

    Each arm emits its next unseen item on its turn. Input semantics and score
    tie-breaking match rrf; explicit orders preserve their supplied tie order.
    """
    orders = _orders(rankings, input_kind, seed)
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
    Input semantics match rrf. All randomness is local to the supplied seed.
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
        self._orders = _orders(rankings, input_kind, seed)
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
) -> pd.DataFrame:
    """Return an ordered copy of an already selected human-review subset.

    scores names numeric columns (larger = earlier review). The caller filters
    below its cutoff; this function neither selects pairs nor changes actions,
    budgets, columns or index labels. Positional IDs also support duplicate
    DataFrame index labels. method="evr" is a passthrough: preserve the existing
    policy/caller order byte-for-byte on serialization, without recomputing EVR.
    No configuration or CLI integration is implied by this standalone utility.
    """
    if method == "evr":
        return df.copy()
    if method not in {"rrf", "interleave"}:
        raise ValueError("method must be evr, rrf or interleave")
    rankings = [df[column].to_numpy(dtype=float) for column in scores]
    if method == "rrf":
        order = rrf(rankings, k=k, seed=seed, input_kind="scores")
    else:
        order = interleave(rankings, seed=seed, input_kind="scores")
    return df.iloc[order].copy()
