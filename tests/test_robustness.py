"""Tests of the grouped bootstrap of queue fusions against an inline thesis copy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gzcmd_record_linkage.fusion import THESIS_FUSIONS, FusionSpec
from gzcmd_record_linkage.ranking import order_by_score
from gzcmd_record_linkage.robustness import (
    _resample,
    bootstrap_fusion_differences,
    replicate_targets,
    subset_effort,
)

SEED = 20261107
N = 300
N_GROUPS = 60
N_BOOT = 50
FUSIONS = {name: THESIS_FUSIONS[name] for name in ("F3", "F4")}
COMPARISONS = (("F4", "O1"), ("F4", "O3"), ("F3", "O1"), ("F3", "O3"))
THESIS_ARMS = {"F4": ("O3", "O1"), "F3": ("O3", "O1", "O3_MACD")}


# ---- inline re-implementation of scripts/fusao_robustez.py -----------------


def thesis_rrf(queues):
    n = len(queues[0])
    score = np.zeros(n)
    for q in queues:
        score[q] += 1.0 / (60 + np.arange(1, n + 1))
    return np.lexsort((np.arange(n), -score))


def thesis_fused(queues):
    return {
        "O1": queues["O1"],
        "O3": queues["O3"],
        **{n: thesis_rrf([queues[a] for a in arms]) for n, arms in THESIS_ARMS.items()},
    }


def thesis_resampled(queues, multiplicity):
    original = np.repeat(np.arange(len(multiplicity)), multiplicity)
    result = {}
    for name, q in queues.items():
        ranks = np.empty(len(q), dtype=np.int64)
        ranks[q] = np.arange(len(q))
        result[name] = np.argsort(ranks[original], kind="stable")
    return result, original


def thesis_reviews(orders, labels, targets):
    if not targets or min(targets) < 1 or max(targets) > int(labels.sum()):
        raise ValueError("unreachable")
    return {
        name: np.searchsorted(labels[q].astype(int).cumsum(), targets) + 1
        for name, q in orders.items()
    }


def thesis_once(queues, labels, group_codes, group_count, rng):
    draws = rng.integers(0, group_count, size=group_count)
    counts = np.bincount(draws, minlength=group_count)
    sampled, original = thesis_resampled(queues, counts[group_codes])
    sampled_labels = labels[original]
    total = int(sampled_labels.sum())
    if not total:
        return None
    targets = (
        (9, 23, 36)
        if total >= 36
        else tuple(int(np.ceil(f * total)) for f in (0.25, 0.64, 1.0))
    )
    return (
        thesis_reviews(thesis_fused(sampled), sampled_labels, targets),
        total,
        len(original),
    )


def thesis_summary(runs):
    rows = []
    for fused, base in COMPARISONS:
        fv = np.array([r[0][fused] for r in runs])
        bv = np.array([r[0][base] for r in runs])
        diff = fv - bv
        reduction = 100 * (bv - fv) / bv
        for j, target in enumerate((9, 23, 36)):
            row = {
                "comparison": f"{fused}-{base}",
                "nominal_target": target,
                "valid_replicates": len(runs),
                "p_fusion_lower": float((diff[:, j] < 0).mean()),
            }
            for name, values in (
                ("difference", diff[:, j]),
                ("reduction_percent", reduction[:, j]),
            ):
                lo, median, hi = np.quantile(
                    values, [0.025, 0.5, 0.975], method="linear"
                )
                row.update(
                    {
                        f"{name}_median": float(median),
                        f"{name}_ci95_lower": float(lo),
                        f"{name}_ci95_upper": float(hi),
                    }
                )
            rows.append(row)
    joint = np.mean(
        [(r[0]["F4"] < r[0]["O1"]) & (r[0]["F4"] < r[0]["O3"]) for r in runs], axis=0
    )
    table = pd.DataFrame(rows)
    table["joint_f4"] = table.nominal_target.map(
        dict(zip((9, 23, 36), joint, strict=True))
    )
    return table


def thesis_bootstrap(scores, labels, groups, n_boot, seed=SEED):
    """Thesis flow on data already laid out in hash (tie) order."""
    queues = {name: np.lexsort((np.arange(len(s)), -s)) for name, s in scores.items()}
    codes, uniques = pd.factorize(groups, sort=True)
    rng = np.random.default_rng(seed)
    runs = [
        thesis_once(queues, labels, codes, len(uniques), rng) for _ in range(n_boot)
    ]
    return [r for r in runs if r is not None], n_boot - sum(r is not None for r in runs)


# ---- synthetic data --------------------------------------------------------


def make_data(n_positive_groups=None, seed=3, positive_rate=0.14):
    rng = np.random.default_rng(seed)
    groups = rng.integers(0, N_GROUPS, size=N)
    base = rng.normal(size=N)
    labels = rng.random(N) < positive_rate
    if n_positive_groups is not None:
        keep = np.isin(groups, np.arange(n_positive_groups))
        labels = labels & keep
        labels[np.flatnonzero(keep)[:3]] = True
    signal = labels * 1.5
    scores = {
        # Rounding creates score ties, which the tie order must settle.
        "O1": np.round(base + signal + rng.normal(size=N), 1),
        "O3": np.round(base + signal + rng.normal(size=N), 1),
        "O3_MACD": np.round(rng.normal(size=N) + signal, 1),
    }
    return scores, labels, groups


def lib_run(scores, labels, groups, **kwargs):
    fusions = {
        "F3": FusionSpec("F3", "rrf", ("O3", "O1", "O3_MACD")),
        "F4": FusionSpec("F4", "rrf", ("O3", "O1")),
    }
    return bootstrap_fusion_differences(
        scores,
        labels,
        groups,
        fusions=fusions,
        comparisons=COMPARISONS,
        n_boot=kwargs.pop("n_boot", N_BOOT),
        **kwargs,
    )


def assert_matches_thesis(result, runs, excluded):
    assert result.n_valid == len(runs)
    assert result.n_excluded == excluded
    reps = result.replicates
    assert reps["replicate"].nunique() == len(runs)
    for i, (effort, total, size) in enumerate(runs):
        rows = reps[reps["replicate"].eq(reps["replicate"].unique()[i])]
        assert (rows["n_positives"] == total).all()
        assert (rows["n_items"] == size).all()
        for name, values in effort.items():
            got = rows[rows["queue"].eq(name)].sort_values("slot")["reviews"].to_numpy()
            np.testing.assert_array_equal(got, values)
    expected = thesis_summary(runs)
    summary = result.summary
    assert list(summary["comparison"]) == list(expected["comparison"])
    assert list(summary["nominal_target"]) == list(expected["nominal_target"])
    for column in (
        "valid_replicates",
        "p_fusion_lower",
        "difference_median",
        "difference_ci95_lower",
        "difference_ci95_upper",
        "reduction_percent_median",
        "reduction_percent_ci95_lower",
        "reduction_percent_ci95_upper",
    ):
        np.testing.assert_array_equal(summary[column], expected[column])
    f4 = summary["fusion"].eq("F4")
    np.testing.assert_array_equal(
        summary.loc[f4, "p_fusion_lower_than_all_bases"], expected.loc[f4, "joint_f4"]
    )


# ---- tests ------------------------------------------------------------------


def test_matches_thesis_bootstrap_exactly():
    scores, labels, groups = make_data()
    result = lib_run(scores, labels, groups)
    runs, excluded = thesis_bootstrap(scores, labels, groups, N_BOOT)
    assert_matches_thesis(result, runs, excluded)
    assert result.seed == SEED
    assert result.n_boot == N_BOOT


def test_mixed_hybrid_target_rule_matches_thesis():
    scores, labels, groups = make_data(positive_rate=0.13)
    result = lib_run(scores, labels, groups, n_boot=200)
    runs, excluded = thesis_bootstrap(scores, labels, groups, 200)
    assert_matches_thesis(result, runs, excluded)
    full = result.replicates.drop_duplicates("replicate")["full_target"]
    assert full.any()
    assert not full.all()
    absolute = result.replicates[result.replicates["full_target"]]
    assert (absolute["target"] == absolute["nominal_target"]).all()
    assert (absolute["n_positives"] >= 36).all()
    fractional = result.replicates[~result.replicates["full_target"]]
    expected = np.ceil(
        fractional["slot"].map({0: 0.25, 1: 0.64, 2: 1.0}) * fractional["n_positives"]
    )
    np.testing.assert_array_equal(fractional["target"], expected.astype(int))
    assert (fractional["n_positives"] < 36).all()


def test_tie_ids_reproduce_thesis_order_on_shuffled_data():
    scores, labels, groups = make_data()
    ids = np.array([f"{i:08x}" for i in np.random.default_rng(5).permutation(N)])
    perm = np.random.default_rng(6).permutation(N)  # file order differs from hash order
    result = lib_run(
        {k: v[perm] for k, v in scores.items()},
        labels[perm],
        groups[perm],
        tie_ids=ids[perm],
    )
    hash_order = np.argsort(ids, kind="stable")
    runs, excluded = thesis_bootstrap(
        {k: v[hash_order] for k, v in scores.items()},
        labels[hash_order],
        groups[hash_order],
        N_BOOT,
    )
    assert_matches_thesis(result, runs, excluded)


def test_base_queue_order_matches_thesis_queue_with_copies():
    """Thesis ``queue()``: score descending, ties by ascending pair id."""
    scores, _, _ = make_data()
    ids = np.array([f"{i:08x}" for i in np.random.default_rng(9).permutation(N)])
    multiplicity = np.random.default_rng(10).integers(0, 3, size=N)
    for name, score in scores.items():
        thesis_queue = (
            pd.DataFrame({"id": ids, "score": score, "pos": np.arange(N)})
            .sort_values(["score", "id"], ascending=[False, True])
            .pos.to_numpy()
        )
        order = order_by_score(score, tie_ids=ids)
        np.testing.assert_array_equal(order, thesis_queue, err_msg=name)
        # Copies are contiguous, follow the rank of their original and keep the
        # thesis order (score descending, ties by ascending id).
        resampled, original = _resample({name: order}, multiplicity)
        np.testing.assert_array_equal(original, np.repeat(np.arange(N), multiplicity))
        ranks = np.empty(N, dtype=np.int64)
        ranks[thesis_queue] = np.arange(N)
        assert (np.diff(ranks[original[resampled[name]]]) >= 0).all()


def test_replicates_without_positives_are_excluded():
    scores, labels, groups = make_data(n_positive_groups=2, positive_rate=0.2)
    assert labels.sum() > 0
    result = lib_run(scores, labels, groups, n_boot=100)
    runs, excluded = thesis_bootstrap(scores, labels, groups, 100)
    assert excluded > 0
    assert result.n_excluded == excluded
    assert result.n_valid + result.n_excluded == 100
    assert_matches_thesis(result, runs, excluded)
    assert (result.replicates["n_positives"] > 0).all()
    assert (result.summary["valid_replicates"] == result.n_valid).all()


def test_all_replicates_excluded_raises():
    scores, labels, groups = make_data()
    with pytest.raises(ValueError, match="no replicate"):
        lib_run(scores, np.zeros(N, dtype=bool), groups, n_boot=5)


def test_hybrid_rule_function():
    assert replicate_targets(36) == ((9, 23, 36), True)
    assert replicate_targets(80) == ((9, 23, 36), True)
    assert replicate_targets(35) == ((9, 23, 35), False)
    assert replicate_targets(10) == ((3, 7, 10), False)
    assert replicate_targets(1) == ((1, 1, 1), False)
    assert replicate_targets(20, (5, 10), (0.5, 1.0)) == ((5, 10), True)
    assert replicate_targets(8, (5, 10), (0.5, 1.0)) == ((4, 8), False)
    with pytest.raises(ValueError):
        replicate_targets(0)
    with pytest.raises(ValueError):
        replicate_targets(5, (5, 10), (1.0,))


def test_thompson_fusions_are_rejected():
    scores, labels, groups = make_data()
    for name in ("A1", "A2"):
        with pytest.raises(ValueError, match="thompson"):
            bootstrap_fusion_differences(
                scores,
                labels,
                groups,
                fusions={name: THESIS_FUSIONS[name]},
                comparisons=[(name, "O1")],
                n_boot=2,
            )


def test_invalid_inputs_raise():
    scores, labels, groups = make_data()
    f4 = {"F4": THESIS_FUSIONS["F4"]}
    with pytest.raises(ValueError, match="unknown queue"):
        bootstrap_fusion_differences(
            scores, labels, groups, fusions=f4, comparisons=[("F4", "X")], n_boot=2
        )
    with pytest.raises(ValueError, match="missing queues"):
        bootstrap_fusion_differences(
            {"O1": scores["O1"]},
            labels,
            groups,
            fusions=f4,
            comparisons=[("F4", "O1")],
            n_boot=2,
        )
    with pytest.raises(ValueError, match="one entry per item"):
        bootstrap_fusion_differences(
            scores,
            labels,
            groups[:-1],
            fusions=f4,
            comparisons=[("F4", "O1")],
            n_boot=2,
        )
    with pytest.raises(ValueError, match="n_boot"):
        bootstrap_fusion_differences(
            scores, labels, groups, fusions=f4, comparisons=[("F4", "O1")], n_boot=0
        )


def test_deterministic_by_seed():
    scores, labels, groups = make_data()
    a = lib_run(scores, labels, groups, n_boot=20)
    b = lib_run(scores, labels, groups, n_boot=20)
    c = lib_run(scores, labels, groups, n_boot=20, seed=SEED + 1)
    pd.testing.assert_frame_equal(a.replicates, b.replicates)
    pd.testing.assert_frame_equal(a.summary, b.summary)
    assert not a.replicates["reviews"].equals(c.replicates["reviews"])
    assert c.seed == SEED + 1


def test_subset_effort_matches_thesis_band():
    scores, labels, groups = make_data(positive_rate=0.2)
    mask = np.round(scores["O1"], 1) >= -0.5
    assert labels[mask].sum() > 0
    targets = (3, 5, 9)
    assert labels[mask].sum() >= max(targets)
    queues = {n: np.lexsort((np.arange(N), -s)) for n, s in scores.items()}
    restricted, original = thesis_resampled(queues, mask.astype(int))
    effort = thesis_reviews(thesis_fused(restricted), labels[original], targets)
    table = subset_effort(
        scores,
        labels,
        mask,
        fusions=FUSIONS,
        targets=targets,
        baselines=("O1", "O3"),
    )
    assert list(table["queue"].unique()) == ["O1", "O3", "O3_MACD", "F3", "F4"]
    for name in ("O1", "O3", "F4", "F3"):
        rows = table[table["queue"].eq(name)]
        np.testing.assert_array_equal(rows["reviews"], effort[name])
        for base in ("O1", "O3"):
            np.testing.assert_array_equal(
                rows[f"difference_{base}"], effort[name] - effort[base]
            )
            np.testing.assert_array_equal(
                rows[f"reduction_percent_{base}"],
                100 * (1 - effort[name] / effort[base]),
            )
    with pytest.raises(ValueError, match="not reachable"):
        subset_effort(scores, labels, mask, fusions=FUSIONS, targets=(1000,))
