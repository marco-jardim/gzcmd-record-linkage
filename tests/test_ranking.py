from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gzcmd_record_linkage.ranking import (
    ThompsonQueue,
    interleave,
    review_queue,
    rrf,
    validate_order,
)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("n", [0, 1, 7, 100])
def test_fusion_properties(seed, n):
    rng = np.random.default_rng(seed)
    orders = [rng.permutation(n) for _ in range(3)]
    before = [order.copy() for order in orders]
    for method in [rrf, interleave]:
        actual = method(orders, seed=seed)
        validate_order(actual, n)
        np.testing.assert_array_equal(actual, method(orders, seed=seed))
        np.testing.assert_array_equal(method(orders[:1]), orders[0])
    for actual, expected in zip(orders, before, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_rrf_common_top_and_formula():
    assert rrf([[2, 0, 1], [2, 1, 0]])[0] == 2
    orders = [[3, 0, 1, 2], [1, 3, 2, 0]]
    np.testing.assert_array_equal(rrf(orders, k=0), [3, 1, 0, 2])
    np.testing.assert_array_equal(rrf(orders[:1], k=1e300), orders[0])


def test_interleave_skips_shared_items():
    np.testing.assert_array_equal(
        interleave([[0, 1, 2, 3, 4], [0, 2, 1, 4, 3]]), [0, 2, 1, 4, 3]
    )


@pytest.mark.parametrize("method", [rrf, interleave])
def test_scores_and_shuffled_ties(method):
    scores = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    expected = np.random.default_rng(42).permutation(6)
    np.testing.assert_array_equal(
        method([scores], input_kind="scores"), expected
    )
    assert not np.array_equal(expected, np.arange(6))
    assert not np.array_equal(
        expected, method([scores], input_kind="scores", seed=43)
    )
    np.testing.assert_array_equal(
        method([[0, 2, 1]], input_kind="scores"), [1, 2, 0]
    )
    np.testing.assert_array_equal(method([[0, 2, 1]]), [0, 2, 1])


def test_rrf_fusion_ties_are_seeded():
    forward = list(range(6))
    actual = rrf([forward, forward[::-1]], seed=19)
    np.testing.assert_array_equal(actual, rrf([forward, forward[::-1]], seed=19))
    # Equal fused scores for items 0 and 5 follow shuffled, not row, priority.
    priority = list(np.random.default_rng(19).permutation(6))
    assert list(actual[:2]) == sorted([0, 5], key=priority.index)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("window", [None, 1, 3, 200])
def test_thompson_unique_exhaustion_and_determinism(seed, window):
    orders = [list(range(23)), list(reversed(range(23)))]
    queues = [ThompsonQueue(orders, seed=seed, window=window) for _ in range(2)]
    outputs = []
    for queue in queues:
        output = []
        for _ in range(23):
            item = queue.next()
            output.append(item)
            queue.update(item, item % 2)
        validate_order(output, 23)
        with pytest.raises(StopIteration):
            queue.next()
        outputs.append(output)
    assert outputs[0] == outputs[1]


def test_thompson_feedback_window_and_reservations():
    queue = ThompsonQueue([[0, 1, 2, 3]], window=2)
    items = [queue.next() for _ in range(4)]
    assert items == [0, 1, 2, 3]
    assert queue._trials.tolist() == [0]
    for item, label in [(3, 1), (0, 1), (2, 0)]:
        queue.update(item, label)
    assert queue._trials.tolist() == [2]
    assert queue._successes.tolist() == [1]
    for item, label in [(0, 1), (10, 1), (1, 2), (1, 0.5), (1.0, 1)]:
        with pytest.raises(ValueError):
            queue.update(item, label)
    queue.update(1, 0)  # Rejected feedback did not consume the reservation.
    assert queue._successes.tolist() == [0]


def test_thompson_rewards_only_selected_arm():
    queue = ThompsonQueue([[0, 1, 2], [2, 1, 0]], prior=(2, 3))
    item = queue.next()
    selected = queue._pending[item]
    queue.update(item, 1)
    assert queue._trials[selected] == queue._successes[selected] == 1
    assert queue._trials[1 - selected] == queue._successes[1 - selected] == 0


def test_thompson_scores_match_explicit_order():
    queue = ThompsonQueue([[0.2, 0.8, 0.5]], input_kind="scores")
    assert [queue.next() for _ in range(3)] == [1, 2, 0]


@pytest.mark.parametrize("window", [0, -1, True, 1.5])
def test_invalid_window(window):
    with pytest.raises(ValueError, match="window"):
        ThompsonQueue([[0]], window=window)


@pytest.mark.parametrize("prior", [(0, 1), (-1, 1), (1, np.inf), (1,)])
def test_invalid_prior(prior):
    with pytest.raises(ValueError, match="prior"):
        ThompsonQueue([[0]], prior=prior)


@pytest.mark.parametrize("order", [[0, 0], [0, 2], [-1, 0], [0.0, 1.0], [[0, 1]]])
def test_invalid_order(order):
    with pytest.raises(ValueError, match="permutation"):
        validate_order(order, 2)


@pytest.mark.parametrize("n", [-1, 1.5, True])
def test_invalid_universe_size(n):
    with pytest.raises(ValueError, match="permutation"):
        validate_order([], n)


@pytest.mark.parametrize("method", [rrf, interleave, ThompsonQueue])
@pytest.mark.parametrize("rankings", [[], [[0], [0, 1]], [[0, 0]], [[[0, 1]]]])
def test_invalid_rankings(method, rankings):
    with pytest.raises(ValueError):
        method(rankings)


@pytest.mark.parametrize("k", [-1, np.nan, np.inf])
def test_invalid_k(k):
    with pytest.raises(ValueError, match="k"):
        rrf([[0]], k=k)


@pytest.mark.parametrize("method", [rrf, interleave, ThompsonQueue])
def test_invalid_scores_and_kind(method):
    with pytest.raises(ValueError, match="finite"):
        method([[0, np.nan]], input_kind="scores")
    with pytest.raises(ValueError, match="input_kind"):
        method([[0]], input_kind="unknown")


@pytest.mark.parametrize("method", ["rrf", "interleave"])
def test_review_queue_copy_with_duplicate_index(method):
    frame = pd.DataFrame(
        {"s1": [0.1, 0.9, 0.4], "s2": [0.2, 0.8, 0.3], "action": ["DEFERRED"] * 3},
        index=["a", "a", "b"],
    )
    before = frame.copy(deep=True)
    result = review_queue(frame, scores=["s1", "s2"], method=method)
    pd.testing.assert_frame_equal(result, frame.iloc[[1, 2, 0]])
    pd.testing.assert_frame_equal(frame, before)
    assert result is not frame
    assert review_queue(frame, method="evr").to_csv() == frame.to_csv()


def test_empty_queue_and_bad_review_arguments():
    assert review_queue(pd.DataFrame({"s": []}), scores=["s"]).empty
    with pytest.raises(StopIteration):
        ThompsonQueue([[]]).next()
    with pytest.raises(ValueError):
        review_queue(pd.DataFrame({"s": [1]}))
    with pytest.raises(ValueError, match="method"):
        review_queue(pd.DataFrame(), method="unknown")
    with pytest.raises(KeyError):
        review_queue(pd.DataFrame(), scores=["missing"])
