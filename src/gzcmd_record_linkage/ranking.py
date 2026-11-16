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
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

InputKind = Literal["orders", "scores"]
QueueInputKind = Literal["orders", "scores", "partial_orders"]
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


def _partial_orders(rankings: Sequence[ArrayLike], n_items: int) -> list[Order]:
    """Validate partial orders: unique integer item IDs in range(n_items) per arm."""
    if len(rankings) == 0:
        raise ValueError("at least one ranking is required")
    orders = []
    for arm, ranking in enumerate(rankings):
        vector = np.asarray(ranking)
        if vector.ndim != 1:
            raise ValueError(f"ranking {arm} must be one-dimensional")
        if vector.size == 0:
            raise ValueError(f"ranking {arm} must not be empty")
        if vector.dtype.kind not in "iu":
            raise ValueError(f"ranking {arm} must contain integer item IDs")
        if vector.min() < 0 or vector.max() >= n_items:
            raise ValueError(f"ranking {arm} contains item IDs outside [0, {n_items})")
        if np.unique(vector).size != vector.size:
            raise ValueError(f"ranking {arm} contains duplicate item IDs")
        orders.append(vector.astype(np.int64, copy=True))
    return orders


_STATE_VERSION = 1
_STATE_KEYS = (
    "version",
    "input_kind",
    "n_items",
    "window",
    "prior",
    "seed",
    "orders",
    "pointers",
    "history",
    "successes",
    "trials",
    "pending",
    "arm_by_item",
    "seen",
    "issued",
    "rng",
)


def _state_int(value: object, name: str, low: int, high: int) -> int:
    """Return value as int if it is a non-bool integer in [low, high]."""
    if (
        not isinstance(value, (int, np.integer))
        or isinstance(value, (bool, np.bool_))
        or not low <= value <= high
    ):
        raise ValueError(f"state {name} must be an integer in [{low}, {high}]")
    return int(value)


def _state_list(value: object, name: str, length: int | None = None) -> list:
    if not isinstance(value, list) or (length is not None and len(value) != length):
        suffix = "" if length is None else f" of length {length}"
        raise ValueError(f"state {name} must be a list{suffix}")
    return value


def _state_pairs(value: object, name: str, n_items: int, n_arms: int) -> dict[int, int]:
    """Parse a list of [item, arm] pairs into a dict."""
    pairs: dict[int, int] = {}
    for pair in _state_list(value, name):
        pair = _state_list(pair, f"{name} entry", 2)
        item = _state_int(pair[0], f"{name} item", 0, n_items - 1)
        if item in pairs:
            raise ValueError(f"state {name} repeats item {item}")
        pairs[item] = _state_int(pair[1], f"{name} arm", 0, n_arms - 1)
    return pairs


def _plain_scalar(value: Any) -> Any:
    """Python scalar for numpy scalars (``np.int64(5)`` -> ``5``); else unchanged."""
    return value.item() if isinstance(value, np.generic) else value


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

    input_kind="partial_orders" (requires n_items) accepts one distinct list of
    unique integer item IDs in [0, n_items) per arm. Arms may differ in length
    and may overlap. An item reserved through one arm is skipped by the others;
    an arm whose list has no unseen item left is exhausted and is never
    sampled. next() raises StopIteration only once every arm is exhausted. With
    complete permutations this mode consumes the RNG exactly like "orders".

    mark_seen(item) (additive) marks a not-pending item as seen without
    reserving it through any arm (for example one consumed by an outside
    exploration policy): every arm then skips it, no arm gets credit and it is
    never pending. It is idempotent and recorded in to_state().

    to_state() returns a JSON-compatible dict (including the RNG state) and
    from_state() restores a queue that continues the identical sequence.
    arm_stats(), exhausted() and arm_of(item) are read-only accessors.
    """

    def __init__(
        self,
        rankings: Sequence[ArrayLike],
        window: int | None = 200,
        prior: tuple[float, float] = (1, 1),
        seed: int = 42,
        *,
        input_kind: QueueInputKind = "orders",
        n_items: int | None = None,
    ) -> None:
        if window is not None and (
            isinstance(window, bool) or not isinstance(window, int) or window <= 0
        ):
            raise ValueError("window must be a positive integer or None")
        if len(prior) != 2 or not np.isfinite(prior).all() or min(prior) <= 0:
            raise ValueError("prior must contain two finite positive parameters")
        self._partial = input_kind == "partial_orders"
        if input_kind == "partial_orders":
            if (
                n_items is None
                or not isinstance(n_items, (int, np.integer))
                or isinstance(n_items, (bool, np.bool_))
                or n_items <= 0
            ):
                raise ValueError(
                    "n_items must be a positive integer for input_kind='partial_orders'"
                )
            self._orders = _partial_orders(rankings, int(n_items))
            n_seen = int(n_items)
        else:
            if n_items is not None:
                raise ValueError(
                    "n_items is only valid with input_kind='partial_orders'"
                )
            self._orders = _orders(rankings, input_kind, seed, "random")
            n_seen = len(self._orders[0])
        self._input_kind: str = input_kind
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._window = window
        self._prior = prior
        self._seen = np.zeros(n_seen, dtype=bool)
        self._pointers = [0] * len(self._orders)
        self._pending: dict[int, int] = {}
        self._marked: set[int] = set()
        self._arm_by_item: dict[int, int] = {}
        self._history: list[deque[int]] = [deque() for _ in self._orders]
        self._successes = np.zeros(len(self._orders), dtype=np.int64)
        self._trials = np.zeros(len(self._orders), dtype=np.int64)
        self._issued = 0

    def _first_unseen(self, arm: int) -> int:
        """Position of the first unseen item of arm at or after its pointer."""
        order = self._orders[arm]
        position = self._pointers[arm]
        while position < len(order) and self._seen[order[position]]:
            position += 1
        return position

    def exhausted(self) -> list[bool]:
        """Per arm: True when no unseen item is left in its list."""
        return [
            self._first_unseen(arm) >= len(order)
            for arm, order in enumerate(self._orders)
        ]

    def arm_stats(self) -> list[tuple[int, int]]:
        """Per arm: (successes, trials) over the current window."""
        return [
            (int(success), int(trial))
            for success, trial in zip(self._successes, self._trials, strict=True)
        ]

    def arm_of(self, item: int) -> int:
        """Arm that reserved item; valid while pending and after update.

        Raises KeyError if the item was never reserved.
        """
        try:
            return self._arm_by_item[item]
        except KeyError:
            raise KeyError(f"item {item!r} was never reserved") from None

    def _next_partial(self) -> int:
        positions = [self._first_unseen(arm) for arm in range(len(self._orders))]
        active = np.array(
            [
                position < len(order)
                for position, order in zip(positions, self._orders, strict=True)
            ]
        )
        if not active.any():
            raise StopIteration
        samples = self._rng.beta(
            self._prior[0] + self._successes,
            self._prior[1] + self._trials - self._successes,
        )
        arm = int(np.argmax(np.where(active, samples, -np.inf)))
        self._pointers = positions
        item = int(self._orders[arm][self._pointers[arm]])
        self._pointers[arm] += 1
        self._seen[item] = True
        self._issued += 1
        self._pending[item] = arm
        self._arm_by_item[item] = arm
        return item

    def next(self) -> int:
        """Reserve the next unique positional item ID; raise on exhaustion."""
        if self._partial:
            return self._next_partial()
        if self._issued + len(self._marked) == len(self._seen):
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
        self._arm_by_item[item] = arm
        return item

    def mark_seen(self, item: int) -> None:
        """Mark a not-pending item as seen so that every arm skips it.

        Idempotent. Raises ValueError for a non-integer or out-of-range item
        and for an item that is currently pending (reserved, not yet updated).
        """
        if not isinstance(item, (int, np.integer)) or isinstance(item, bool):
            raise ValueError("item must be an integer ID")
        if not 0 <= int(item) < len(self._seen):
            raise ValueError(f"item {item!r} is outside [0, {len(self._seen)})")
        key = int(item)
        if key in self._pending:
            raise ValueError("item is pending: update it before marking it seen")
        if self._seen[key]:
            return
        self._seen[key] = True
        self._marked.add(key)

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

    def to_state(self) -> dict:
        """Return the complete state as a JSON-compatible dict (RNG included)."""
        return {
            "version": _STATE_VERSION,
            "input_kind": self._input_kind,
            "n_items": len(self._seen),
            "window": _plain_scalar(self._window),
            "prior": [float(self._prior[0]), float(self._prior[1])],
            "seed": _plain_scalar(self._seed),
            "orders": [order.tolist() for order in self._orders],
            "pointers": [int(pointer) for pointer in self._pointers],
            "history": [[int(reward) for reward in h] for h in self._history],
            "successes": self._successes.tolist(),
            "trials": self._trials.tolist(),
            "pending": [[int(i), int(a)] for i, a in self._pending.items()],
            "arm_by_item": [[int(i), int(a)] for i, a in self._arm_by_item.items()],
            "seen": np.flatnonzero(self._seen).tolist(),
            "marked": sorted(self._marked),
            "issued": int(self._issued),
            "rng": self._rng.bit_generator.state,
        }

    @classmethod
    def from_state(cls, state: dict) -> ThompsonQueue:
        """Restore a queue from to_state() output (also after a JSON round trip)."""
        if not isinstance(state, dict):
            raise ValueError("state must be a dict")
        version = state.get("version")
        if isinstance(version, bool) or version != _STATE_VERSION:
            raise ValueError(f"unsupported ThompsonQueue state version: {version!r}")
        missing = [key for key in _STATE_KEYS if key not in state]
        if missing:
            raise ValueError(f"state is missing keys: {missing}")
        kind = state["input_kind"]
        if kind not in {"orders", "scores", "partial_orders"}:
            raise ValueError(f"state has an invalid input_kind: {kind!r}")
        prior = state["prior"]
        if not isinstance(prior, list) or len(prior) != 2:
            raise ValueError("state prior must be a list of two numbers")
        partial = kind == "partial_orders"
        orders = _state_list(state["orders"], "orders")
        # Stored scores-mode orders are already permutations: rebuild as orders.
        queue = cls(
            orders,
            state["window"],
            (prior[0], prior[1]),
            state["seed"],
            input_kind="partial_orders" if partial else "orders",
            n_items=state["n_items"] if partial else None,
        )
        queue._input_kind = kind
        n_items = len(queue._seen)
        n_arms = len(queue._orders)
        if _state_int(state["n_items"], "n_items", 0, n_items) != n_items:
            raise ValueError("state n_items does not match its orders")
        pointers = _state_list(state["pointers"], "pointers", n_arms)
        queue._pointers = [
            _state_int(p, "pointer", 0, len(queue._orders[arm]))
            for arm, p in enumerate(pointers)
        ]
        successes = _state_list(state["successes"], "successes", n_arms)
        trials = _state_list(state["trials"], "trials", n_arms)
        histories = _state_list(state["history"], "history", n_arms)
        for arm in range(n_arms):
            trial = _state_int(trials[arm], "trials entry", 0, n_items)
            success = _state_int(successes[arm], "successes entry", 0, trial)
            rewards = [
                _state_int(r, "history reward", 0, 1)
                for r in _state_list(histories[arm], "history entry")
            ]
            if queue._window is None:
                if rewards:
                    raise ValueError("state history must be empty without a window")
            elif (
                len(rewards) > queue._window
                or len(rewards) != trial
                or sum(rewards) != success
            ):
                raise ValueError("state history is inconsistent with its counters")
            queue._history[arm] = deque(rewards)
            queue._successes[arm] = success
            queue._trials[arm] = trial
        seen = _state_list(state["seen"], "seen")
        seen_items = {_state_int(i, "seen item", 0, n_items - 1) for i in seen}
        if len(seen_items) != len(seen):
            raise ValueError("state seen repeats an item")
        arm_by_item = _state_pairs(state["arm_by_item"], "arm_by_item", n_items, n_arms)
        pending = _state_pairs(state["pending"], "pending", n_items, n_arms)
        issued = _state_int(state["issued"], "issued", 0, n_items)
        marked_raw = _state_list(state.get("marked", []), "marked")
        marked = {_state_int(i, "marked item", 0, n_items - 1) for i in marked_raw}
        if len(marked) != len(marked_raw):
            raise ValueError("state marked repeats an item")
        if (
            set(arm_by_item) & marked
            or set(arm_by_item) | marked != seen_items
            or issued != len(arm_by_item)
        ):
            raise ValueError("state seen, marked, arm_by_item and issued disagree")
        for item, arm in arm_by_item.items():
            if not (queue._orders[arm] == item).any():
                raise ValueError(f"state item {item} is not in the list of arm {arm}")
        if any(arm_by_item.get(item) != arm for item, arm in pending.items()):
            raise ValueError("state pending items disagree with arm_by_item")
        queue._seen[sorted(seen_items)] = True
        queue._marked = marked
        queue._arm_by_item = arm_by_item
        queue._pending = pending
        queue._issued = issued
        rng_state = state["rng"]
        if not isinstance(rng_state, dict):
            raise ValueError("state rng must be a dict")
        try:
            queue._rng.bit_generator.state = rng_state
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"state rng is invalid: {exc}") from exc
        return queue


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
