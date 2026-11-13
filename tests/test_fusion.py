"""Thesis fusions: exact parity with inline copies of scripts/fusao_filas.py."""

from __future__ import annotations

from collections import deque
from types import MappingProxyType

import numpy as np
import pytest

from gzcmd_record_linkage.fusion import (
    THESIS_FUSIONS,
    FusionSpec,
    compare_queues,
    fuse,
    queue_metrics,
    simulate_thompson,
)
from gzcmd_record_linkage.ranking import ThompsonQueue

# ---------------------------------------------------------------------------
# Inline copies of the thesis code (scripts/fusao_filas.py), parameterised
# only where the thesis hard-codes the 61,485/36 universe.
# ---------------------------------------------------------------------------


def _validate(order, n):
    if len(order) != n or not np.array_equal(np.sort(order), np.arange(n)):
        raise ValueError("not a permutation")


def thesis_round_robin(queues):
    n = len(queues[0])
    seen = np.zeros(n, dtype=bool)
    pointers = [0] * len(queues)
    result = []
    while len(result) < n:
        for arm, q in enumerate(queues):
            p = pointers[arm]
            while p < n and seen[q[p]]:
                p += 1
            if p < n:
                item = q[p]
                seen[item] = True
                result.append(item)
                p += 1
            pointers[arm] = p
    order = np.asarray(result, dtype=np.int64)
    _validate(order, n)
    return order


def thesis_rrf(queues):
    n = len(queues[0])
    score = np.zeros(n)
    for q in queues:
        score[q] += 1.0 / (60 + np.arange(1, n + 1))
    order = np.lexsort((np.arange(n), -score))
    _validate(order, n)
    return order


def thesis_thompson(queues, labels, seed, window):
    rng = np.random.default_rng(seed)
    n, arms = len(labels), len(queues)
    seen = np.zeros(n, dtype=bool)
    pointers = [0] * arms
    successes = np.zeros(arms, dtype=np.int64)
    trials = np.zeros(arms, dtype=np.int64)
    history = [deque() for _ in range(arms)]
    order = np.empty(n, dtype=np.int64)
    for step in range(n):
        proposals = []
        for arm, q in enumerate(queues):
            p = pointers[arm]
            while p < n and seen[q[p]]:
                p += 1
            pointers[arm] = p
            if p == n:
                raise ValueError("exhausted arm")
            proposals.append(q[p])
        samples = rng.beta(1 + successes, 1 + trials - successes)
        arm = int(np.argmax(samples))
        item = proposals[arm]
        order[step] = item
        seen[item] = True
        reward = int(labels[item])
        successes[arm] += reward
        trials[arm] += 1
        if window is not None:
            history[arm].append(reward)
            if len(history[arm]) > window:
                successes[arm] -= history[arm].popleft()
                trials[arm] -= 1
    _validate(order, n)
    return order


def thesis_metrics(counts, targets, top_k, horizon):
    total = counts[-1]
    result = {f"reviews_to_{n}": float(np.searchsorted(counts, n) + 1) for n in targets}
    result.update({f"recovered_at_{k}": float(counts[k - 1]) for k in top_k})
    result["auc_recovery"] = float(counts[: horizon - 1].sum() / (horizon * total))
    return result


def thesis_aggregate(runs, names, targets, top_k, horizon):
    values = np.array(
        [[thesis_metrics(c, targets, top_k, horizon)[m] for m in names] for c in runs]
    )
    return (
        values.mean(axis=0),
        np.quantile(values, 0.1, axis=0, method="linear"),
        np.quantile(values, 0.9, axis=0, method="linear"),
    )


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


def make_data(n, seed, positives=None):
    """Three base orders (one the reverse of another: many RRF ties) and labels."""
    rng = np.random.default_rng(seed)
    first = rng.permutation(n).astype(np.int64)
    second = rng.permutation(n).astype(np.int64)
    third = first[::-1].copy()
    k = positives if positives is not None else max(3, n // 25)
    labels = np.zeros(n, dtype=bool)
    labels[rng.choice(n, size=k, replace=False)] = True
    # Make the queues informative so that rewards differ across arms.
    head = np.flatnonzero(labels)[: k // 2]
    second = np.concatenate([head, second[~np.isin(second, head)]]).astype(np.int64)
    return {"O1": first, "O3": second, "O3_MACD": third}, labels


CASES = [(30, 0), (120, 1), (400, 2), (900, 3)]


def metric_names(targets, top_k):
    return (
        [f"reviews_to_{n}" for n in targets]
        + [f"recovered_at_{k}" for k in top_k]
        + ["auc_recovery"]
    )


# ---------------------------------------------------------------------------
# Definitions
# ---------------------------------------------------------------------------


def test_thesis_fusions_content():
    assert list(THESIS_FUSIONS) == ["F1", "F2", "F3", "F4", "A1", "A2"]
    assert isinstance(THESIS_FUSIONS, MappingProxyType)
    with pytest.raises(TypeError):
        THESIS_FUSIONS["F5"] = FusionSpec("F5", "rrf", ("O1",))  # type: ignore[index]
    expected = {
        "F1": ("interleave", ("O3", "O1")),
        "F2": ("interleave", ("O3", "O1", "O3_MACD")),
        "F3": ("rrf", ("O3", "O1", "O3_MACD")),
        "F4": ("rrf", ("O3", "O1")),
        "A1": ("thompson", ("O1", "O3", "O3_MACD")),
        "A2": ("thompson", ("O1", "O3", "O3_MACD")),
    }
    for name, (method, arms) in expected.items():
        spec = THESIS_FUSIONS[name]
        assert (spec.name, spec.method, spec.arms) == (name, method, arms)
        assert spec.prior == (1.0, 1.0)
    assert THESIS_FUSIONS["F3"].k == THESIS_FUSIONS["F4"].k == 60
    assert THESIS_FUSIONS["A1"].window == 200
    assert THESIS_FUSIONS["A2"].window is None


def test_fusion_spec_validation_and_frozen():
    with pytest.raises(ValueError, match="method"):
        FusionSpec("X", "borda", ("O1",))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one arm"):
        FusionSpec("X", "rrf", ())
    spec = FusionSpec("X", "rrf", ["O1", "O3"])  # type: ignore[arg-type]
    assert spec.arms == ("O1", "O3")
    with pytest.raises(AttributeError):
        spec.k = 10  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Fusions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("n", "seed"), CASES)
def test_interleave_fusions_match_thesis_round_robin(n, seed):
    orders, _ = make_data(n, seed)
    for name in ("F1", "F2"):
        spec = THESIS_FUSIONS[name]
        got = fuse(orders, spec)
        want = thesis_round_robin([orders[a] for a in spec.arms])
        assert np.array_equal(got, want)


@pytest.mark.parametrize(("n", "seed"), CASES)
def test_rrf_fusions_match_thesis_with_ties(n, seed):
    orders, _ = make_data(n, seed)
    for name in ("F3", "F4"):
        spec = THESIS_FUSIONS[name]
        got = fuse(orders, spec)
        want = thesis_rrf([orders[a] for a in spec.arms])
        assert np.array_equal(got, want)
    # O3_MACD is the reverse of O1: F3 over (O1, O3_MACD) has exact RRF ties.
    pair = FusionSpec("P", "rrf", ("O1", "O3_MACD"))
    scores = np.zeros(n)
    for arm in pair.arms:
        scores[orders[arm]] += 1.0 / (60 + np.arange(1, n + 1))
    assert len(np.unique(scores)) < n
    assert np.array_equal(
        fuse(orders, pair), thesis_rrf([orders[a] for a in pair.arms])
    )


@pytest.mark.parametrize(("n", "seed"), CASES)
def test_rrf_ties_by_tie_ids_match_thesis_hash_space(n, seed):
    orders, _ = make_data(n, seed)
    rng = np.random.default_rng(100 + seed)
    ids = np.array([f"{v:016x}" for v in rng.permutation(10**6)[:n]])
    lexical_rank = np.argsort(np.argsort(ids))  # thesis index = hash rank
    pair = FusionSpec("P", "rrf", ("O1", "O3_MACD"))
    got = fuse(orders, pair, tie_ids=ids)
    thesis_queues = [lexical_rank[orders[a]] for a in pair.arms]
    assert np.array_equal(lexical_rank[got], thesis_rrf(thesis_queues))


@pytest.mark.parametrize(("n", "seed"), CASES)
@pytest.mark.parametrize("window", [None, 200, 5])
def test_simulate_thompson_matches_thesis(n, seed, window):
    orders, labels = make_data(n, seed)
    arms = [orders["O1"], orders["O3"], orders["O3_MACD"]]
    for run_seed in (0, 1, 7):
        got = simulate_thompson(arms, labels, window=window, seed=run_seed)
        want = thesis_thompson(arms, labels, run_seed, window)
        assert np.array_equal(got, want)


def test_simulate_thompson_equals_queue_replay_and_uses_thesis_defaults():
    orders, labels = make_data(300, 5)
    arms = [orders["O1"], orders["O3"], orders["O3_MACD"]]
    queue = ThompsonQueue(arms, window=200, prior=(1, 1), seed=0)
    replay = []
    for _ in range(len(labels)):
        item = queue.next()
        queue.update(item, int(labels[item]))
        replay.append(item)
    assert np.array_equal(simulate_thompson(arms, labels), np.array(replay))


def test_fuse_thompson_uses_spec_window_and_seed():
    orders, labels = make_data(400, 6)
    arms = [orders[a] for a in THESIS_FUSIONS["A1"].arms]
    for name, window in (("A1", 200), ("A2", None)):
        got = fuse(orders, THESIS_FUSIONS[name], labels=labels, seed=3)
        assert np.array_equal(got, thesis_thompson(arms, labels, 3, window))


def test_fuse_thompson_requires_labels():
    orders, _ = make_data(30, 0)
    with pytest.raises(ValueError, match="requires labels"):
        fuse(orders, THESIS_FUSIONS["A1"])


def test_fuse_rejects_missing_arm_and_wrong_label_length():
    orders, labels = make_data(30, 0)
    with pytest.raises(ValueError, match="missing queues"):
        fuse({"O1": orders["O1"]}, THESIS_FUSIONS["F4"])
    with pytest.raises(ValueError, match="one entry per item"):
        simulate_thompson([orders["O1"], orders["O3"]], labels[:-1])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("n", "seed"), CASES)
def test_queue_metrics_match_thesis(n, seed):
    orders, labels = make_data(n, seed)
    total = int(labels.sum())
    targets = (1, total // 2, total)
    top_k = (1, n // 4, n // 2, n)
    for order in orders.values():
        counts = labels[order].astype(int).cumsum()
        for horizon in (None, n, max(2, n // 3)):
            got = queue_metrics(
                order, labels, targets=targets, top_k=top_k, auc_horizon=horizon
            )
            want = thesis_metrics(counts, targets, top_k, horizon or n)
            assert list(got) == list(want)
            assert got == want


def test_queue_metrics_default_targets_and_auc_is_left_step():
    order = np.array([2, 0, 1, 3])
    labels = np.array([0, 1, 1, 0])
    got = queue_metrics(order, labels, targets=(1, 2), top_k=(2,), auc_horizon=3)
    # C = [1, 1, 2, 2]; sum C(0..2) = 0 + 1 + 1 = 2; H * P = 3 * 2
    assert got == {
        "reviews_to_1": 1,
        "reviews_to_2": 3,
        "recovered_at_2": 1,
        "auc_recovery": 2 / 6,
    }
    assert list(queue_metrics(order, labels)) == [
        "reviews_to_9",
        "reviews_to_23",
        "reviews_to_36",
        "auc_recovery",
    ]


def test_queue_metrics_unreachable_target_is_none():
    order = np.arange(10)
    labels = np.zeros(10, dtype=bool)
    labels[[3, 8]] = True
    got = queue_metrics(order, labels, targets=(2, 3))
    assert got["reviews_to_2"] == 9
    assert got["reviews_to_3"] is None
    with pytest.raises(ValueError):
        queue_metrics(order, labels, targets=(0,))
    with pytest.raises(ValueError, match="no true pairs"):
        queue_metrics(order, np.zeros(10))
    with pytest.raises(ValueError, match="auc_horizon"):
        queue_metrics(order, labels, auc_horizon=11)


# ---------------------------------------------------------------------------
# compare_queues
# ---------------------------------------------------------------------------


def test_compare_queues_shape_columns_and_deterministic_rows():
    orders, labels = make_data(400, 2)
    targets, top_k = (2, 5), (10, 100)
    seeds = range(4)
    table = compare_queues(
        orders,
        labels,
        targets=targets,
        top_k=top_k,
        auc_horizon=300,
        adaptive_seeds=seeds,
    )
    names = metric_names(targets, top_k)
    assert list(table.queue) == [
        "O1",
        "O3",
        "O3_MACD",
        "F1",
        "F2",
        "F3",
        "F4",
        "A1",
        "A2",
    ]
    assert list(table.kind) == ["base"] * 3 + ["fusion"] * 6
    assert list(table.runs) == [1] * 7 + [4, 4]
    assert table.method[:3].isna().all()
    assert list(table.method)[3:] == [
        "interleave", "interleave", "rrf", "rrf", "thompson", "thompson"
    ]  # fmt: skip
    expected_columns = ["queue", "kind", "method", "runs"] + [
        c for name in names for c in (name, f"{name}_p10", f"{name}_p90")
    ]
    assert list(table.columns) == expected_columns
    rows = table.set_index("queue")
    deterministic = {
        "O1": orders["O1"],
        "O3": orders["O3"],
        "O3_MACD": orders["O3_MACD"],
        "F1": thesis_round_robin([orders["O3"], orders["O1"]]),
        "F2": thesis_round_robin([orders["O3"], orders["O1"], orders["O3_MACD"]]),
        "F3": thesis_rrf([orders["O3"], orders["O1"], orders["O3_MACD"]]),
        "F4": thesis_rrf([orders["O3"], orders["O1"]]),
    }
    for name, order in deterministic.items():
        counts = labels[order].astype(int).cumsum()
        mean, p10, p90 = thesis_aggregate([counts], names, targets, top_k, 300)
        for j, metric in enumerate(names):
            assert rows.loc[name, metric] == mean[j]
            assert rows.loc[name, f"{metric}_p10"] == p10[j]
            assert rows.loc[name, f"{metric}_p90"] == p90[j]


@pytest.mark.parametrize("window_name", ["A1", "A2"])
def test_compare_queues_adaptive_rows_are_mean_p10_p90_over_seeds(window_name):
    orders, labels = make_data(400, 4)
    targets, top_k = (2, 4), (50,)
    seeds = range(6)
    spec = THESIS_FUSIONS[window_name]
    table = compare_queues(
        orders,
        labels,
        fusions={window_name: spec},
        targets=targets,
        top_k=top_k,
        adaptive_seeds=seeds,
    )
    assert list(table.queue) == ["O1", "O3", "O3_MACD", window_name]
    names = metric_names(targets, top_k)
    arms = [orders[a] for a in spec.arms]
    runs = [
        labels[thesis_thompson(arms, labels, s, spec.window)].astype(int).cumsum()
        for s in seeds
    ]
    mean, p10, p90 = thesis_aggregate(runs, names, targets, top_k, len(labels))
    row = table.set_index("queue").loc[window_name]
    for j, metric in enumerate(names):
        assert row[metric] == mean[j]
        assert row[f"{metric}_p10"] == p10[j]
        assert row[f"{metric}_p90"] == p90[j]


def test_compare_queues_unreachable_target_is_nan_and_empty_seeds_fail():
    orders, labels = make_data(60, 1, positives=3)
    table = compare_queues(
        orders, labels, fusions={"F4": THESIS_FUSIONS["F4"]}, targets=(3, 4)
    )
    assert table["reviews_to_4"].isna().all()
    assert table["reviews_to_3"].notna().all()
    with pytest.raises(ValueError, match="adaptive_seeds"):
        compare_queues(
            orders, labels, fusions={"A1": THESIS_FUSIONS["A1"]}, adaptive_seeds=[]
        )
