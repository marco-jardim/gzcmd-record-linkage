"""Context features and queues C / CM, F4_contexto."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from _thesis_repo import THESIS_ROOT, load_thesis, requires_thesis

from gzcmd_record_linkage.context import (
    CONTEXT_FEATURES,
    MACD_COLUMNS,
    THESIS_CONTEXT_FUSIONS,
    base_features,
    context_design_matrices,
    context_features,
    context_fusions,
    context_queues,
    macd_features,
    thesis_context_features,
)
from gzcmd_record_linkage.fusion import THESIS_FUSIONS, FusionSpec, fuse
from gzcmd_record_linkage.llm_dossier import make_pair_id
from gzcmd_record_linkage.loader import load_comparador_csv
from gzcmd_record_linkage.ranking import validate_order

FAST = {"n_estimators": 6, "max_depth": 3, "n_jobs": 1}


def reference_context(refrec, comprec, score):
    """Inline re-implementation: plain loops over the groups, no pandas."""
    out = {}
    for name, keys in (("refrec", refrec), ("comprec", comprec)):
        rows = {k: [] for k in keys}
        for k, s in zip(keys, score, strict=True):
            rows[k].append(s)
        cols = {
            f"ctx_{name}_{s}": []
            for s in (
                "n_candidatos",
                "posicao",
                "delta_melhor",
                "delta_segundo",
                "eh_melhor",
            )
        }
        for k, s in zip(keys, score, strict=True):
            group = sorted(rows[k], reverse=True)
            second = group[1] if len(group) > 1 else group[0]
            cols[f"ctx_{name}_n_candidatos"].append(float(len(group)))
            cols[f"ctx_{name}_posicao"].append(float(1 + sum(v > s for v in group)))
            cols[f"ctx_{name}_delta_melhor"].append(s - group[0])
            cols[f"ctx_{name}_delta_segundo"].append(s - second)
            cols[f"ctx_{name}_eh_melhor"].append(float(s == group[0]))
        out.update(cols)
    return pd.DataFrame(out)


def tie_frame():
    # REFREC R1 has a tie at the top; R2 is a singleton; R3 has a three-way tie.
    return pd.DataFrame(
        {
            "REFREC": ["R1", "R1", "R1", "R2", "R3", "R3", "R3", "R1"],
            "COMPREC": ["C1", "C2", "C1", "C1", "C3", "C4", "C3", "C5"],
            "nota final": ["9,5", "9.5", "7", "8", "6", "6", "6", "9.5"],
        },
        index=[10, 11, 12, 13, 14, 15, 16, 17],
    )


def test_context_features_match_reference_with_ties_and_singletons():
    df = tie_frame()
    got = context_features(df)
    score = [9.5, 9.5, 7.0, 8.0, 6.0, 6.0, 6.0, 9.5]
    want = reference_context(df["REFREC"], df["COMPREC"], score)
    assert list(got.columns) == list(CONTEXT_FEATURES) and len(CONTEXT_FEATURES) == 10
    assert (got.dtypes == "float64").all()
    assert list(got.index) == list(df.index)
    pd.testing.assert_frame_equal(got, want.set_axis(df.index), check_exact=True)
    # spot values: R2 singleton -> 0/0, best; R1 three-way tie at the top.
    assert got.loc[13, "ctx_refrec_delta_segundo"] == 0.0
    assert got.loc[13, "ctx_refrec_delta_melhor"] == 0.0
    assert got.loc[13, "ctx_refrec_posicao"] == 1.0
    assert got.loc[12, "ctx_refrec_posicao"] == 4.0  # "min" rank after 3 tied rows
    assert got.loc[10, "ctx_refrec_delta_segundo"] == 0.0  # tied best share the 2nd
    assert got.loc[12, "ctx_refrec_delta_segundo"] == 7.0 - 9.5
    assert got.loc[12, "ctx_refrec_eh_melhor"] == 0.0
    assert got.loc[14, "ctx_refrec_eh_melhor"] == 1.0


def test_context_features_random_vs_reference():
    rng = np.random.default_rng(3)
    n = 120
    refrec = [f"R{i}" for i in rng.integers(0, 25, n)]
    comprec = [f"C{i}" for i in rng.integers(0, 30, n)]
    score = np.round(rng.uniform(0, 10, n), 1)  # one decimal: many ties
    df = pd.DataFrame({"REFREC": refrec, "COMPREC": comprec, "nota final": score})
    want = reference_context(refrec, comprec, list(score))
    pd.testing.assert_frame_equal(
        context_features(df), want, check_exact=True, check_dtype=True
    )


def test_context_features_resolves_keys_alias_and_validates():
    df = pd.DataFrame(
        {
            "REFREC,x": [" a ", "a", "b"],
            "COMPREC": ["c", "c", "d"],
            "nota_final": [1, 2, 3],
        }
    )
    out = context_features(df)
    assert out["ctx_refrec_n_candidatos"].tolist() == [2.0, 2.0, 1.0]
    only = context_features(df, keys=("COMPREC",))
    assert list(only.columns) == [c for c in CONTEXT_FEATURES if "comprec" in c]
    with pytest.raises(ValueError, match="not found"):
        context_features(df.drop(columns="nota_final"))
    with pytest.raises(ValueError, match="grouping key"):
        context_features(df.drop(columns="COMPREC"))
    with pytest.raises(ValueError, match="blank"):
        context_features(df.assign(COMPREC=["c", " ", "d"]))
    with pytest.raises(ValueError, match="blank"):
        context_features(df.assign(COMPREC=["c", None, "d"]))
    with pytest.raises(ValueError, match="not finite"):
        context_features(df.assign(nota_final=["1", "inf", "3"]))


def test_context_features_zero_imputes_missing_score():
    df = pd.DataFrame(
        {"REFREC": ["a", "a"], "COMPREC": ["c", "d"], "nota final": [np.nan, 3.0]}
    )
    out = context_features(df)
    assert out["ctx_refrec_delta_melhor"].tolist() == [-3.0, 0.0]


def candidates(seed: int = 11, n_groups: int = 80, per_group: int = 4):
    rng = np.random.default_rng(seed)
    n = n_groups * per_group
    refrec = np.repeat([f"R{i:03d}" for i in range(n_groups)], per_group)
    comprec = [f"C{i}" for i in rng.integers(0, 150, n)]
    y = np.zeros(n, dtype=int)
    y[::per_group][rng.permutation(n_groups)[:30]] = 1
    nota = np.round(rng.uniform(0, 6, n) + 3.0 * y, 1)
    df = pd.DataFrame(
        {
            "COMPREC": comprec,
            "REFREC": refrec,
            "PASSO": ["1"] * n,
            "nota final": nota,
            "NOME sim": rng.uniform(size=n) + y,
            "DTNASC sim": rng.uniform(size=n),
        }
    )
    for name in MACD_COLUMNS:
        df[name] = rng.integers(0, 3, n).astype(float)
    df.loc[::17, MACD_COLUMNS[0]] = np.nan
    return df, y


def test_base_and_macd_features():
    df, _ = candidates()
    base = base_features(df)
    assert list(base.columns) == ["nota final", "NOME sim", "DTNASC sim"]
    macd = macd_features(df)
    assert list(macd.columns) == list(MACD_COLUMNS)
    assert macd[MACD_COLUMNS[0]].iloc[0] == -1.0 and np.isfinite(macd.to_numpy()).all()
    with pytest.raises(ValueError, match="9 MACD"):
        macd_features(df.drop(columns=MACD_COLUMNS[0]))
    with pytest.raises(ValueError, match="no comparator"):
        base_features(pd.DataFrame({"REFREC": ["a"]}))


def test_context_queues_both_variants_and_mask():
    df, y = candidates()
    mask = (df["nota final"] < 8.0).to_numpy()
    ids = np.array(
        [
            make_pair_id(c, r, p)
            for c, r, p in zip(df.COMPREC, df.REFREC, df.PASSO, strict=True)
        ]
    )
    res = context_queues(df, y, xgb_params=FAST, queue_mask=mask, tie_ids=ids)
    assert set(res.probabilities) == {"O3_contexto", "O3_contexto_MACD"}
    m = int(mask.sum())
    assert np.array_equal(res.rows, np.flatnonzero(mask))
    for variant, p in res.probabilities.items():
        assert p.shape == (len(df),) and np.all((p >= 0) & (p <= 1))
        validate_order(res.orders[variant], m)
        scores = p[res.rows][res.orders[variant]]
        assert np.all(np.diff(scores) <= 0)
        # ties follow ascending pair id
        sub_ids = ids[res.rows][res.orders[variant]]
        tied = np.diff(scores) == 0
        assert np.all(sub_ids[:-1][tied] <= sub_ids[1:][tied])
    assert res.folds.shape == (len(df),) and set(res.folds) == set(range(5))
    again = context_queues(df, y, xgb_params=FAST, queue_mask=mask, tie_ids=ids)
    for variant in res.probabilities:
        np.testing.assert_array_equal(
            res.probabilities[variant], again.probabilities[variant]
        )
        np.testing.assert_array_equal(res.orders[variant], again.orders[variant])
    # CM sees the MACD columns: different inputs, different probabilities.
    assert not np.array_equal(
        res.probabilities["O3_contexto"], res.probabilities["O3_contexto_MACD"]
    )


def test_context_queues_validation():
    df, y = candidates()
    with pytest.raises(ValueError, match="variants"):
        context_queues(df, y, variants=("O3",), xgb_params=FAST)
    with pytest.raises(ValueError, match="groups_col"):
        context_queues(df, y, groups_col="PASSO", xgb_params=FAST)
    with pytest.raises(ValueError, match="one label"):
        context_queues(df, y[:-1], xgb_params=FAST)
    with pytest.raises(ValueError, match="queue_mask"):
        context_queues(df, y, queue_mask=np.ones(len(df)), xgb_params=FAST)
    with pytest.raises(ValueError, match="tie_ids"):
        context_queues(df, y, tie_ids=np.arange(3), xgb_params=FAST)
    plain = df.drop(columns=list(MACD_COLUMNS))
    res = context_queues(plain, y, variants=("O3_contexto",), xgb_params=FAST)
    assert set(res.orders) == {"O3_contexto"}
    with pytest.raises(ValueError, match="9 MACD"):
        context_queues(plain, y, xgb_params=FAST)


def test_context_fusion_f4_contexto():
    spec = THESIS_CONTEXT_FUSIONS["F4_contexto"]
    assert spec == FusionSpec("F4_contexto", "rrf", ("O1", "O3_contexto"), k=60)
    assert "F4_contexto" not in THESIS_FUSIONS
    assert set(THESIS_FUSIONS) == {"F1", "F2", "F3", "F4", "A1", "A2"}
    rng = np.random.default_rng(5)
    n = 40
    o1 = rng.permutation(n)
    o3c = rng.permutation(n)
    fused = fuse({"O1": o1, "O3_contexto": o3c}, spec)
    # inline thesis RRF: sum 1/(60 + rank), ties by item position
    values = np.zeros(n)
    for order in (o1, o3c):
        values[order] += 1.0 / (60 + np.arange(1, n + 1))
    want = np.lexsort((np.arange(n), -values))
    np.testing.assert_array_equal(fused, want)
    via_helper = context_fusions({"O1": o1, "O3_contexto": o3c})
    assert list(via_helper) == ["F4_contexto"]
    np.testing.assert_array_equal(via_helper["F4_contexto"], want)
    with pytest.raises(ValueError, match="missing"):
        context_fusions({"O3_contexto": o3c})
    assert not hasattr(THESIS_CONTEXT_FUSIONS, "__setitem__")


def test_macd_order_is_the_thesis_order():
    assert MACD_COLUMNS == (
        "macd_nasc_close",
        "macd_nasc_day_match",
        "macd_nasc_diff_capped",
        "macd_nasc_month_match",
        "macd_nasc_partial_overlap",
        "macd_nasc_very_close",
        "macd_nasc_year_match",
        "macd_nome_perf_x_date_far",
        "macd_nome_perf_x_year_diff",
    )
    assert list(MACD_COLUMNS) == sorted(MACD_COLUMNS)
    df, _ = candidates()
    shuffled = df[[*reversed(MACD_COLUMNS), "REFREC", "COMPREC", "nota final"]]
    assert list(macd_features(shuffled).columns) == list(MACD_COLUMNS)


def test_base_features_keep_frame_order_and_design_matrix_layout():
    df, _ = candidates()
    flipped = df[["NOME sim", "DTNASC sim", "nota final", "REFREC", "COMPREC"]]
    assert list(base_features(flipped).columns) == [
        "NOME sim",
        "DTNASC sim",
        "nota final",
    ]
    mats = context_design_matrices(df)
    assert list(mats["O3_contexto"].columns) == [
        "nota final",
        "NOME sim",
        "DTNASC sim",
        *CONTEXT_FEATURES,
    ]
    assert list(mats["O3_contexto_MACD"].columns) == [
        "nota final",
        "NOME sim",
        "DTNASC sim",
        *MACD_COLUMNS,
        *CONTEXT_FEATURES,
    ]


def test_thesis_signature_agrees_with_dataframe_interface():
    df = tie_frame()
    x = pd.DataFrame(
        {"nota final": [9.5, 9.5, 7.0, 8.0, 6.0, 6.0, 6.0, 9.5]}, index=df.index
    )
    groups = {"refrec": df["REFREC"], "comprec": df["COMPREC"]}
    pd.testing.assert_frame_equal(
        thesis_context_features(x, groups), context_features(df), check_exact=True
    )
    with pytest.raises(ValueError, match="exactly one"):
        thesis_context_features(x.assign(nota_final=1.0), groups)
    with pytest.raises(ValueError, match="not finite"):
        thesis_context_features(x.assign(**{"nota final": np.nan}), groups)
    with pytest.raises(ValueError, match="misaligned"):
        thesis_context_features(
            x, {**groups, "refrec": groups["refrec"].reset_index(drop=True)}
        )


@requires_thesis
def test_parity_pinned_orders_with_thesis_files():
    thesis = load_thesis()
    data = thesis["data"]
    assert list(MACD_COLUMNS) == thesis["ablation_macd"]
    assert list(base_features(data["raw"]).columns) == thesis["original_features"]
    assert list(base_features(data["raw"]).columns) == list(
        data["x"]["O3_controle"].columns
    )
    assert list(data["x"]["O3_MACD"].columns) == [
        *data["x"]["O3_controle"].columns,
        *MACD_COLUMNS,
    ]


@requires_thesis
def test_parity_context_features_with_thesis_function_on_real_data():
    thesis = load_thesis()
    data = thesis["data"]
    resolve = thesis["sensibilidade"].resolve_prefixed_column
    groups = {
        name: data["raw"][resolve(data["raw"], name.upper())].str.strip()
        for name in ("refrec", "comprec")
    }
    want = thesis["contexto"].context_features(data["x"]["O3_controle"], groups)
    got = context_features(data["raw"])
    assert list(got.columns) == list(want.columns) == list(CONTEXT_FEATURES)
    pd.testing.assert_frame_equal(got, want, check_exact=True)
    np.testing.assert_array_equal(got.to_numpy(), want.to_numpy())
    via_thesis_signature = thesis_context_features(data["x"]["O3_controle"], groups)
    pd.testing.assert_frame_equal(via_thesis_signature, want, check_exact=True)


@requires_thesis
def test_parity_model_matrices_with_thesis_augment_on_real_data():
    thesis = load_thesis()
    data = thesis["data"]
    theirs = {
        "raw": data["raw"],
        "x": {name: frame.copy() for name, frame in data["x"].items()},
    }
    thesis["contexto"].augment(theirs)
    # The MACD indicators come from the thesis matrix (the library loader is not
    # exercised here); everything else is built by the library from the raw table.
    frame = data["raw"].copy()
    macd_source = data["x"]["O3_MACD"][list(MACD_COLUMNS)]
    frame[list(MACD_COLUMNS)] = macd_source
    ours = context_design_matrices(frame)
    for variant in ("O3_contexto", "O3_contexto_MACD"):
        want = theirs["x"][variant]
        assert list(ours[variant].columns) == list(want.columns)
        pd.testing.assert_frame_equal(ours[variant], want, check_exact=True)


@requires_thesis
def test_parity_macd_via_library_loader_on_real_data():
    # The library's own loading path (load_comparador_csv -> feature_engineer)
    # must derive the MACD block of the thesis O3_contexto_MACD matrix, exactly.
    thesis = load_thesis()
    data = thesis["data"]
    frame = load_comparador_csv(THESIS_ROOT / "data/COMPARADORSEMIDENT.csv")
    assert len(frame) == len(data["raw"])
    ours = macd_features(frame).reset_index(drop=True)
    theirs = data["x"]["O3_MACD"][list(MACD_COLUMNS)].reset_index(drop=True)
    assert list(ours.columns) == list(theirs.columns) == list(MACD_COLUMNS)
    pd.testing.assert_frame_equal(ours, theirs, check_exact=True)
