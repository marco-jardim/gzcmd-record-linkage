"""ThompsonQueue partial orders, read-only accessors and state round trip."""

from __future__ import annotations

import json

import numpy as np
import pytest

from gzcmd_record_linkage.ranking import ThompsonQueue

N_ITEMS = 40


def label_of(item: int) -> int:
    return int(item % 3 == 0)


def make_queue(kind: str, window: int | None = 5, seed: int = 7) -> ThompsonQueue:
    rng = np.random.default_rng(123)
    if kind == "orders":
        rankings = [rng.permutation(30) for _ in range(3)]
        return ThompsonQueue(rankings, window=window, seed=seed)
    if kind == "scores":
        rankings = [rng.random(30) for _ in range(3)]
        return ThompsonQueue(rankings, window=window, seed=seed, input_kind="scores")
    rankings = [
        rng.permutation(N_ITEMS)[:12],
        rng.permutation(N_ITEMS)[:20],
        rng.permutation(N_ITEMS)[:30],
    ]
    return ThompsonQueue(
        rankings,
        window=window,
        seed=seed,
        input_kind="partial_orders",
        n_items=N_ITEMS,
    )


def drive(
    queue: ThompsonQueue, steps: int, held: int | None = None
) -> tuple[list[int], int | None]:
    """Run steps reservations; each item is labelled one step later.

    Returns the sequence and the item still pending when the run stops.
    """
    sequence = []
    for _ in range(steps):
        item = queue.next()
        sequence.append(item)
        if held is not None:
            queue.update(held, label_of(held))
        held = item
    return sequence, held


def drain(queue: ThompsonQueue) -> list[int]:
    items = []
    while True:
        try:
            item = queue.next()
        except StopIteration:
            return items
        items.append(item)
        queue.update(item, label_of(item))


@pytest.mark.parametrize("window", [5, None])
@pytest.mark.parametrize("kind", ["orders", "scores", "partial_orders"])
def test_resume_from_json_state_matches_uninterrupted_run(kind, window):
    reference = make_queue(kind, window)
    expected, _ = drive(reference, 25)

    queue = make_queue(kind, window)
    first, held = drive(queue, 10)
    assert held is not None  # one reservation is pending at the snapshot
    state = queue.to_state()
    assert state["version"] == 1
    restored_state = json.loads(json.dumps(state))
    assert restored_state == state

    resumed = ThompsonQueue.from_state(restored_state)
    second, _ = drive(resumed, 15, held)

    assert first + second == expected
    assert resumed.arm_stats() == reference.arm_stats()
    assert resumed.to_state() == reference.to_state()
    assert resumed.arm_of(held) == queue.arm_of(held)


def test_resumed_queue_keeps_pending_reservation():
    queue = make_queue("partial_orders")
    item = queue.next()
    resumed = ThompsonQueue.from_state(json.loads(json.dumps(queue.to_state())))
    assert resumed.arm_of(item) == queue.arm_of(item)
    resumed.update(item, 1)
    with pytest.raises(ValueError, match="already updated"):
        resumed.update(item, 1)
    queue.update(item, 1)
    assert resumed.arm_stats() == queue.arm_stats()


def test_disjoint_partial_orders_exhaust_arms_independently():
    rankings = [[2, 0, 1], [7, 3, 5, 4, 6], [15, 8, 9, 14, 10, 13, 11, 12]]
    queue = ThompsonQueue(rankings, seed=3, input_kind="partial_orders", n_items=16)
    issued = []
    while True:
        before = queue.exhausted()
        try:
            item = queue.next()
        except StopIteration:
            break
        assert not before[queue.arm_of(item)]
        issued.append(item)
        queue.update(item, label_of(item))
    assert sorted(issued) == list(range(16))
    assert queue.exhausted() == [True, True, True]
    with pytest.raises(StopIteration):
        queue.next()
    for arm, ranking in enumerate(rankings):
        from_arm = [i for i in issued if queue.arm_of(i) == arm]
        assert from_arm == ranking


def test_exhausted_arm_is_never_sampled_even_with_best_posterior():
    # Arm 0 is tiny but rewarded; once empty it must never come back.
    queue = ThompsonQueue(
        [[0], [1, 2, 3, 4, 5]],
        prior=(1, 1),
        seed=0,
        input_kind="partial_orders",
        n_items=6,
    )
    items = []
    for _ in range(6):
        item = queue.next()
        queue.update(item, 1 if queue.arm_of(item) == 0 else 0)
        items.append(item)
    assert sorted(items) == list(range(6))
    assert [queue.arm_of(i) for i in items].count(0) == 1
    with pytest.raises(StopIteration):
        queue.next()


def test_overlapping_arms_skip_items_seen_by_the_other_arm():
    rankings = [[0, 1, 2, 3], [3, 2, 1, 0, 4], [4, 0]]
    queue = ThompsonQueue(rankings, seed=11, input_kind="partial_orders", n_items=5)
    issued = []
    while True:
        try:
            item = queue.next()
        except StopIteration:
            break
        issued.append(item)
        union = set(issued)
        for arm, ranking in enumerate(rankings):
            # An arm is exhausted exactly when all its items were issued.
            assert queue.exhausted()[arm] == (set(ranking) <= union)
    assert sorted(issued) == [0, 1, 2, 3, 4]
    for arm, ranking in enumerate(rankings):
        from_arm = [i for i in issued if queue.arm_of(i) == arm]
        # Each arm emits its items in its own order, skipping the seen ones.
        assert from_arm == [i for i in ranking if i in from_arm]


@pytest.mark.parametrize(
    ("rankings", "n_items", "match"),
    [
        ([[0, 1]], None, "n_items"),
        ([[0, 1]], 0, "n_items"),
        ([[0, 1]], True, "n_items"),
        ([[0, 1, 1]], 3, "duplicate"),
        ([[0, 1], [2, 2]], 3, "ranking 1 contains duplicate"),
        ([[0, 3]], 3, "outside"),
        ([[-1, 0]], 3, "outside"),
        ([[]], 3, "empty"),
        ([[0.0, 1.0]], 3, "integer"),
        ([[0.5, 1]], 3, "integer"),
        ([["a", "b"]], 3, "integer"),
        ([[[0, 1], [1, 2]]], 3, "one-dimensional"),
        ([], 3, "at least one ranking"),
    ],
)
def test_partial_orders_validation(rankings, n_items, match):
    with pytest.raises(ValueError, match=match):
        ThompsonQueue(rankings, input_kind="partial_orders", n_items=n_items)


def test_n_items_is_rejected_for_complete_modes():
    with pytest.raises(ValueError, match="n_items"):
        ThompsonQueue([[0, 1]], n_items=2)


def test_arm_stats_respect_window():
    labels = [1, 1, 0, 1, 0]
    expected = [(1, 1), (2, 2), (2, 3), (2, 3), (1, 3)]
    queue = ThompsonQueue(
        [list(range(10))], window=3, input_kind="partial_orders", n_items=10
    )
    assert queue.arm_stats() == [(0, 0)]
    for label, stats in zip(labels, expected, strict=True):
        queue.update(queue.next(), label)
        assert queue.arm_stats() == [stats]

    unbounded = ThompsonQueue(
        [list(range(10))], window=None, input_kind="partial_orders", n_items=10
    )
    for label in labels:
        unbounded.update(unbounded.next(), label)
    assert unbounded.arm_stats() == [(3, 5)]


def test_arm_stats_only_count_the_sampled_arm_and_are_plain_ints():
    queue = make_queue("partial_orders", window=None)
    drive(queue, 12)
    stats = queue.arm_stats()
    assert all(type(s) is int and type(t) is int for s, t in stats)
    assert sum(t for _, t in stats) == 11  # twelfth reservation is pending


def test_arm_of_unknown_item_raises_key_error():
    queue = make_queue("orders")
    with pytest.raises(KeyError):
        queue.arm_of(0)
    item = queue.next()
    assert queue.arm_of(item) in range(3)
    queue.update(item, 0)
    assert queue.arm_of(item) in range(3)
    unseen = next(i for i in range(30) if i != item)
    with pytest.raises(KeyError):
        queue.arm_of(unseen)


def test_exhausted_in_complete_modes():
    queue = make_queue("orders")
    assert queue.exhausted() == [False, False, False]
    drain(queue)
    assert queue.exhausted() == [True, True, True]


def test_partial_orders_with_complete_permutations_matches_orders_mode():
    # Same permutations and seed must consume the RNG exactly alike.
    rng = np.random.default_rng(5)
    rankings = [rng.permutation(25) for _ in range(4)]
    original = ThompsonQueue(rankings, window=6, seed=99)
    partial = ThompsonQueue(
        rankings, window=6, seed=99, input_kind="partial_orders", n_items=25
    )
    sequence_a = drain(original)
    sequence_b = drain(partial)
    assert sorted(sequence_a) == list(range(25))
    assert sequence_a == sequence_b
    assert original.arm_stats() == partial.arm_stats()


def test_state_is_json_compatible_and_complete():
    queue = make_queue("partial_orders")
    drive(queue, 8)
    state = json.loads(json.dumps(queue.to_state()))
    assert state["input_kind"] == "partial_orders"
    assert state["n_items"] == N_ITEMS
    assert state["window"] == 5
    assert state["prior"] == [1.0, 1.0]
    assert state["seed"] == 7
    assert len(state["orders"]) == 3
    assert state["issued"] == 8
    assert len(state["pending"]) == 1
    assert state["rng"]["bit_generator"] == "PCG64"


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda s: s.update(version=2), "version"),
        (lambda s: s.pop("rng"), "missing keys"),
        (lambda s: s.update(input_kind="other"), "input_kind"),
        (lambda s: s.update(pointers=[0]), "pointers"),
        (lambda s: s.update(issued=0), "disagree"),
        (lambda s: s.update(rng="nope"), "rng"),
        (lambda s: s.update(rng={"bit_generator": "MT19937"}), "rng"),
    ],
)
def test_from_state_rejects_invalid_state(change, match):
    queue = make_queue("partial_orders")
    drive(queue, 4)
    state = json.loads(json.dumps(queue.to_state()))
    change(state)
    with pytest.raises(ValueError, match=match):
        ThompsonQueue.from_state(state)


def test_from_state_rejects_empty_state():
    with pytest.raises(ValueError, match="version"):
        ThompsonQueue.from_state({})


def test_state_with_numpy_scalars_is_json_serializable():
    queue = make_queue("orders", seed=np.int64(5))
    state = queue.to_state()
    assert type(state["seed"]) is int and state["seed"] == 5
    restored = ThompsonQueue.from_state(json.loads(json.dumps(state)))
    assert restored.next() == queue.next()


# --------------------------------------------------------------------------
# mark_seen
# --------------------------------------------------------------------------


def test_mark_seen_makes_every_arm_skip_the_item():
    queue = make_queue("partial_orders")
    target = int(queue._orders[2][0])
    queue.mark_seen(target)
    queue.mark_seen(target)  # idempotent
    assert queue.arm_stats() == [(0, 0)] * 3
    items = drain(queue)
    assert target not in items
    assert len(items) == len(set(items))
    with pytest.raises(KeyError):
        queue.arm_of(target)  # never reserved: no arm owns it
    assert queue.exhausted() == [True, True, True]


def test_mark_seen_is_idempotent_on_reserved_and_updated_items():
    queue = make_queue("partial_orders")
    item = queue.next()
    queue.update(item, 1)
    before = queue.to_state()
    queue.mark_seen(item)  # already seen through an arm: no-op
    assert queue.to_state() == before
    assert before["marked"] == []


def test_mark_seen_rejects_pending_and_out_of_range_items():
    queue = make_queue("partial_orders")
    item = queue.next()
    with pytest.raises(ValueError, match="pending"):
        queue.mark_seen(item)
    queue.update(item, 0)
    queue.mark_seen(item)
    for bad in (-1, N_ITEMS, 10**6):
        with pytest.raises(ValueError, match="outside"):
            queue.mark_seen(bad)
    for bad in (True, 1.5, "3"):
        with pytest.raises(ValueError, match="integer"):
            queue.mark_seen(bad)


def test_mark_seen_enters_the_state_and_resumes_identically():
    queue = make_queue("partial_orders")
    drive(queue, 5)
    for extra in (int(queue._orders[1][-1]), int(queue._orders[0][-1])):
        if not queue._seen[extra]:
            queue.mark_seen(extra)
    state = json.loads(json.dumps(queue.to_state()))
    assert state["marked"] == sorted(queue._marked) and state["marked"]
    restored = ThompsonQueue.from_state(state)
    assert restored.to_state() == queue.to_state()
    assert drain(restored) == drain(queue)


def test_mark_seen_in_orders_mode_ends_the_queue_and_state_stays_consistent():
    queue = make_queue("orders")
    queue.mark_seen(3)
    items = drain(queue)
    assert sorted(items + [3]) == list(range(30))
    restored = ThompsonQueue.from_state(json.loads(json.dumps(queue.to_state())))
    with pytest.raises(StopIteration):
        restored.next()


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s.update(marked=[0, 0]),
        lambda s: s.update(marked=[s["seen"][0]]),  # also reserved by an arm
        lambda s: s.update(marked=[N_ITEMS - 1, N_ITEMS + 5]),
        lambda s: s.update(
            marked=[next(i for i in range(N_ITEMS) if i not in s["seen"])]
        ),
    ],
)
def test_from_state_rejects_inconsistent_marked(change):
    queue = make_queue("partial_orders")
    drive(queue, 4)
    state = json.loads(json.dumps(queue.to_state()))
    change(state)
    with pytest.raises(ValueError):
        ThompsonQueue.from_state(state)


def test_state_without_marked_key_still_loads():
    queue = make_queue("partial_orders")
    drive(queue, 4)
    state = json.loads(json.dumps(queue.to_state()))
    del state["marked"]
    restored = ThompsonQueue.from_state(state)
    assert drain(restored) == drain(queue)
