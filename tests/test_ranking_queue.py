"""Thesis tie-break, pinned legacy random ties, top-k cut and effort metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gzcmd_record_linkage.ranking import (
    ThompsonQueue,
    effort_curve,
    interleave,
    order_by_score,
    recovered_at_budget,
    review_queue,
    reviews_to_target,
    rrf,
    top_k_cut,
)

TWO_SCORES = [[1, 1, 2, 2, 3, 3, 3, 0], [0.5, 0.5, 0.5, 0.1, 0.1, 0.9, 0.9, 0.9]]


def _thesis_rrf(orders):
    """Inline copy of the thesis fusion (k=60, ties by ascending position)."""
    n = len(orders[0])
    score = np.zeros(n)
    for order in orders:
        score[order] += 1.0 / (60 + np.arange(1, n + 1))
    return np.lexsort((np.arange(n), -score))


# ---------------------------------------------------------------------------
# Thesis tie-break ("index") and the pinned legacy "random" rule
# ---------------------------------------------------------------------------


def test_order_by_score_index_is_thesis_rule():
    scores = np.array([0.5, 0.9, 0.5, 0.9, 0.1, 0.5])
    expected = np.lexsort((np.arange(6), -scores))
    np.testing.assert_array_equal(order_by_score(scores), expected)
    np.testing.assert_array_equal(order_by_score(scores), [1, 3, 0, 2, 5, 4])
    np.testing.assert_array_equal(order_by_score([0.5] * 6), np.arange(6))
    assert order_by_score(scores).dtype == np.int64


def test_order_by_score_tie_ids_and_validation():
    scores = [0.5, 0.9, 0.5, 0.9]
    ids = [40, 30, 20, 10]
    np.testing.assert_array_equal(order_by_score(scores, tie_ids=ids), [3, 1, 2, 0])
    np.testing.assert_array_equal(
        order_by_score(scores, tie_ids=["d", "c", "b", "a"]), [3, 1, 2, 0]
    )
    # Duplicate ids fall back to position.
    np.testing.assert_array_equal(order_by_score(scores, tie_ids=[1] * 4), [1, 3, 0, 2])
    with pytest.raises(ValueError, match="tie_ids"):
        order_by_score(scores, tie_ids=[1, 2, 3])
    with pytest.raises(ValueError, match="tie_ids"):
        order_by_score(scores, tie_ids=[1, 2, np.nan, 4])
    with pytest.raises(ValueError, match="tie_ids"):
        order_by_score(scores, tie_break="random", tie_ids=ids)
    with pytest.raises(ValueError, match="tie_break"):
        order_by_score(scores, tie_break="stable")
    with pytest.raises(ValueError, match="finite"):
        order_by_score([0.1, np.inf])
    with pytest.raises(ValueError, match="one-dimensional"):
        order_by_score([[0.1, 0.2]])
    assert order_by_score([]).tolist() == []


def test_order_by_score_random_is_the_seeded_permutation():
    np.testing.assert_array_equal(
        order_by_score([0.5] * 6, tie_break="random"), [3, 2, 5, 4, 1, 0]
    )
    np.testing.assert_array_equal(
        order_by_score([0.5] * 8, tie_break="random", seed=7),
        [0, 6, 7, 2, 4, 5, 1, 3],
    )
    # Distinct scores are unaffected by the tie rule.
    np.testing.assert_array_equal(
        order_by_score([0.1, 0.9, 0.5], tie_break="random"), [1, 2, 0]
    )


def test_rrf_and_interleave_random_pinned_to_legacy_values():
    kw = {"input_kind": "scores", "tie_break": "random"}
    np.testing.assert_array_equal(rrf(TWO_SCORES, **kw), [6, 5, 4, 7, 2, 3, 1, 0])
    np.testing.assert_array_equal(
        rrf(TWO_SCORES, seed=5, **kw), [5, 6, 4, 7, 2, 1, 3, 0]
    )
    np.testing.assert_array_equal(
        interleave(TWO_SCORES, **kw), [4, 7, 6, 5, 3, 2, 1, 0]
    )
    np.testing.assert_array_equal(
        interleave(TWO_SCORES, seed=5, **kw), [4, 7, 5, 6, 2, 1, 3, 0]
    )
    forward = list(range(6))
    np.testing.assert_array_equal(
        rrf([forward, forward[::-1]], seed=19, tie_break="random"),
        [0, 5, 4, 1, 2, 3],
    )


def test_rrf_and_interleave_index_default_on_constructed_ties():
    kw = {"input_kind": "scores"}
    np.testing.assert_array_equal(rrf(TWO_SCORES, **kw), [5, 6, 4, 0, 2, 7, 1, 3])
    np.testing.assert_array_equal(
        interleave(TWO_SCORES, **kw), [4, 5, 6, 7, 2, 0, 3, 1]
    )
    np.testing.assert_array_equal(
        rrf(TWO_SCORES, tie_break="index", **kw), rrf(TWO_SCORES, **kw)
    )
    # The seed is irrelevant under the thesis rule.
    np.testing.assert_array_equal(rrf(TWO_SCORES, seed=5, **kw), rrf(TWO_SCORES, **kw))
    np.testing.assert_array_equal(rrf([[0.5] * 6], **kw), np.arange(6))
    forward = list(range(6))
    # Items 0/5, 1/4 and 2/3 tie in fused score: lower position first.
    np.testing.assert_array_equal(rrf([forward, forward[::-1]]), [0, 5, 1, 4, 2, 3])


def test_rrf_and_interleave_tie_ids():
    ids = [7, 6, 5, 4, 3, 2, 1, 0]
    kw = {"input_kind": "scores", "tie_ids": ids}
    np.testing.assert_array_equal(rrf([[0.5] * 8], **kw), np.arange(8)[::-1])
    np.testing.assert_array_equal(interleave([[0.5] * 8], **kw), np.arange(8)[::-1])
    with pytest.raises(ValueError, match="tie_ids"):
        rrf(TWO_SCORES, input_kind="scores", tie_break="random", tie_ids=ids)


@pytest.mark.parametrize("seed", range(5))
def test_rrf_matches_inline_thesis_implementation(seed):
    rng = np.random.default_rng(seed)
    n = 60
    orders = [rng.permutation(n) for _ in range(3)]
    np.testing.assert_array_equal(rrf(orders), _thesis_rrf(orders))
    scores = [rng.integers(0, 4, n).astype(float) for _ in range(2)]
    arms = [np.lexsort((np.arange(n), -s)) for s in scores]
    np.testing.assert_array_equal(rrf(scores, input_kind="scores"), _thesis_rrf(arms))


@pytest.mark.parametrize("method", [rrf, interleave])
def test_invalid_tie_break(method):
    with pytest.raises(ValueError, match="tie_break"):
        method([[0, 1]], tie_break="stable")


def test_thompson_queue_keeps_random_ties():
    queue = ThompsonQueue([[0.5] * 6], input_kind="scores")
    assert [queue.next() for _ in range(6)] == [3, 2, 5, 4, 1, 0]


# ---------------------------------------------------------------------------
# review_queue tie-breaking
# ---------------------------------------------------------------------------


def test_review_queue_ties_follow_pair_id_then_position():
    frame = pd.DataFrame(
        {"pair_id": [30, 10, 20, 40], "s": [0.5, 0.5, 0.5, 0.9]},
        index=["w", "x", "y", "z"],
    )
    by_id = ["z", "x", "y", "w"]
    by_position = ["z", "w", "x", "y"]
    result = review_queue(frame, scores=["s"])
    assert result.index.tolist() == by_id
    thesis = frame.sort_values(["s", "pair_id"], ascending=[False, True])
    pd.testing.assert_frame_equal(result, thesis)
    for method in ("rrf", "interleave"):
        assert review_queue(frame, ["s"], method).index.tolist() == by_id
    assert review_queue(frame, ["s"], tie_break="index").index.tolist() == by_position
    no_id = frame.drop(columns="pair_id")
    assert review_queue(no_id, ["s"]).index.tolist() == by_position
    renamed = frame.rename(columns={"pair_id": "id"})
    assert review_queue(renamed, ["s"], id_column="id").index.tolist() == by_id
    # Without the custom id column the fallback is position order.
    assert review_queue(renamed, ["s"]).index.tolist() == by_position
    text = frame.assign(pair_id=["c", "a", "b", "d"])
    assert review_queue(text, ["s"]).index.tolist() == by_id


def test_review_queue_random_pinned_to_legacy_values():
    frame = pd.DataFrame(
        {"a": TWO_SCORES[0], "b": TWO_SCORES[1], "pair_id": range(8, 0, -1)}
    )
    rrf_order = review_queue(frame, ["a", "b"], tie_break="random")
    assert rrf_order.index.tolist() == [6, 5, 4, 7, 2, 3, 1, 0]
    interleaved = review_queue(
        frame, ["a", "b"], "interleave", seed=3, tie_break="random"
    )
    assert interleaved.index.tolist() == [6, 7, 4, 5, 2, 1, 3, 0]
    with pytest.raises(ValueError, match="tie_break"):
        review_queue(frame, ["a"], tie_break="stable")


# ---------------------------------------------------------------------------
# top_k_cut and effort metrics
# ---------------------------------------------------------------------------


def test_top_k_cut_statuses_are_not_non_match():
    order = np.array([2, 0, 3, 1])
    status = top_k_cut(order, 2)
    assert status.tolist() == ["examined", "not_examined", "examined", "not_examined"]
    assert set(status) == {"examined", "not_examined"}
    assert top_k_cut(order, 0).tolist() == ["not_examined"] * 4
    assert top_k_cut(order, 99).tolist() == ["examined"] * 4
    assert top_k_cut([], 3).tolist() == []
    frame = pd.DataFrame({"x": [1, 2, 3]}, index=["a", "a", "b"])
    cut = top_k_cut(frame, 1)
    assert cut.tolist() == ["examined", "not_examined", "not_examined"]
    assert cut.index.tolist() == ["a", "a", "b"]
    for bad in (-1, 1.5, True):
        with pytest.raises(ValueError, match="k"):
            top_k_cut(order, bad)
    with pytest.raises(ValueError, match="permutation"):
        top_k_cut([0, 0], 1)


def test_effort_metrics_and_unreachable_target():
    curve = effort_curve([2, 0, 3, 1], [1, 0, 1, 0])
    assert curve.tolist() == [1, 2, 2, 2]
    assert curve.index.tolist() == [1, 2, 3, 4]
    assert reviews_to_target(curve, 1) == 1
    assert reviews_to_target(curve, 2) == 2
    assert reviews_to_target(curve.to_numpy(), 2) == 2
    assert recovered_at_budget(curve, 0) == 0
    assert recovered_at_budget(curve, 1) == 1
    assert recovered_at_budget(curve, 2) == 2
    assert recovered_at_budget(curve, 99) == 2
    assert recovered_at_budget([], 5) == 0
    with pytest.raises(ValueError, match="inalcançável"):
        reviews_to_target(curve, 3)
    with pytest.raises(ValueError, match="inalcançável"):
        reviews_to_target([], 1)
    with pytest.raises(ValueError, match="pelo menos"):
        reviews_to_target(curve, 0)
    with pytest.raises(ValueError, match="negativa"):
        recovered_at_budget(curve, -1)
    with pytest.raises(ValueError, match="permutation"):
        effort_curve([0, 0, 1, 2], [1, 0, 1, 0])
