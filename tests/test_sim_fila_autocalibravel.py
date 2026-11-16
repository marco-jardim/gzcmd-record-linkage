"""Testes rápidos da simulação da fila auto-calibrável (dados sintéticos)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sim_fila_autocalibravel as sim

from gzcmd_record_linkage.calibration import get_nota_series
from gzcmd_record_linkage.loader import LoadConfig, load_comparador_csv

SYNTH = Path(__file__).resolve().parents[1] / "data" / "synthetic"
SYNTH_CSV = SYNTH / "comparador_sintetico.csv"


@pytest.fixture(scope="module")
def prep() -> sim.Prepared:
    raw = load_comparador_csv(SYNTH_CSV, cfg=LoadConfig(require_par=False))
    nota = pd.to_numeric(get_nota_series(raw), errors="coerce").to_numpy(dtype=float)
    return sim.prepare(raw, below=nota < 7.0, check_partition=False)


# --------------------------------------------------------------------------
# Taxas e roteamento
# --------------------------------------------------------------------------


def test_band_rates_laplace_and_low_fallback() -> None:
    rates = sim.build_scenario_rates("thesis_bands")[0]
    fp, fn, unsure = rates["grey_low"]
    # grey_low: (95 decisões, 2 pares, 93 não pares, 7 FP, 2 FN, 7 UNSURE)
    assert fp == pytest.approx(8 / 95)
    assert fn == pytest.approx(3 / 4)
    assert unsure == pytest.approx(8 / 97)
    assert rates["low"] == rates["grey_low"]
    assert set(rates) == set(sim.BAND_NAMES)
    # Faixa sem FP observado: Laplace mantém taxa positiva.
    assert rates["high"][0] == pytest.approx(1 / 2)


def test_perfect_scenario_is_all_zero() -> None:
    rates, disagreement = sim.build_scenario_rates("perfect")
    assert disagreement == 0.0
    assert all(r == (0.0, 0.0, 0.0) for r in rates.values())
    with pytest.raises(ValueError):
        sim.build_scenario_rates("inexistente")


@pytest.mark.parametrize(
    ("nota", "band"),
    [
        (0.0, "low"),
        (4.99, "low"),
        (5.0, "grey_low"),
        (5.99, "grey_low"),
        (6.0, "grey_mid"),
        (7.0, "grey_high"),
        (7.99, "grey_high"),
        (8.0, "near_high"),
        (8.99, "near_high"),
        (9.0, "high"),
        (12.5, "high"),
    ],
)
def test_banded_oracle_routes_by_band(nota: float, band: str) -> None:
    pairs = {"x": nota}
    truth = {"x": True}
    # Só a faixa alvo tem FN = 1: o veredito vira NONMATCH apenas nela.
    rates = {b: (0.0, 1.0 if b == band else 0.0, 0.0) for b in sim.BAND_NAMES}
    oracle = sim.BandedOracle(truth, pairs, rates, disagreement_rate=0.0, seed=3)
    assert oracle.verdict("x") == "NONMATCH"
    others = {b: (0.0, 0.0, 0.0) for b in sim.BAND_NAMES}
    clean = sim.BandedOracle(truth, pairs, others, disagreement_rate=0.0, seed=3)
    assert clean.verdict("x") == "MATCH"


def test_banded_oracle_is_deterministic_per_seed_and_pair() -> None:
    n = 400
    pids = [f"p{i}" for i in range(n)]
    truth = {p: i % 5 == 0 for i, p in enumerate(pids)}
    nota = {p: 4.0 + 6.0 * (i / n) for i, p in enumerate(pids)}
    rates = sim.build_scenario_rates("thesis_bands")[0]

    def verdicts(seed: int, order: list[str]) -> dict[str, str]:
        o = sim.BandedOracle(truth, nota, rates, disagreement_rate=0.1, seed=seed)
        return {p: o.verdict(p) for p in order}

    a = verdicts(1, pids)
    assert a == verdicts(1, list(reversed(pids)))  # independe da ordem
    assert a != verdicts(2, pids)  # semente muda a realização


# --------------------------------------------------------------------------
# Métricas e curvas
# --------------------------------------------------------------------------


def test_metrics_from_flags_and_reviews_to_k() -> None:
    match = np.array([1, 1, 0, 1, 1, 0, 1], dtype=bool)
    truth = np.array([1, 0, 1, 1, 1, 0, 1], dtype=bool)
    below = np.array([1, 1, 1, 0, 1, 0, 1], dtype=bool)
    par2 = below & truth
    m = sim.metrics_from_flags(match, truth, below, par2, 7)
    assert m["reviews"] == 7
    assert m["true_reviewed"] == 5
    assert m["found"] == 4  # MATCH ∧ verdade nas posições 0, 3, 4, 6
    assert m["found_below"] == 3  # posições 0, 4, 6
    assert m["fp_llm_match"] == 1  # posição 1
    assert m["reviews_to_k5"] != m["reviews_to_k5"]  # NaN: só 3 abaixo do corte
    m2 = sim.metrics_from_flags(match, truth, below, par2, 5)
    assert m2["reviews"] == 5
    assert m2["found"] == 3
    k = sim.metrics_from_flags(
        match, truth, below, par2, 100
    )  # orçamento acima do total é truncado
    assert k["reviews"] == 7


def test_first_k_position() -> None:
    flags = np.array([0, 0, 1, 0, 1, 1], dtype=bool)
    assert sim._first_k(flags, 1) == 3.0
    assert sim._first_k(flags, 3) == 6.0
    assert np.isnan(sim._first_k(flags, 4))


# --------------------------------------------------------------------------
# Taxas de um terceiro revisor (--rates-csv)
# --------------------------------------------------------------------------


def _write_rates(path: Path, rows: list[tuple[str, float, float, float]]) -> Path:
    pd.DataFrame(rows, columns=["band", "fp", "fn", "unsure"]).to_csv(path, index=False)
    return path


def test_load_rates_csv_roundtrip_and_low_fallback(tmp_path: Path) -> None:
    rows = [
        (b, 0.01 * (i + 1), 0.02 * (i + 1), 0.03)
        for i, b in enumerate(sim.BAND_NAMES)
        if b != "low"
    ]
    rates = sim.load_rates_csv(_write_rates(tmp_path / "r.csv", rows))
    assert set(rates) == set(sim.BAND_NAMES)
    assert rates["grey_mid"] == pytest.approx((0.03, 0.06, 0.03))
    assert rates["low"] == rates["grey_low"]  # a faixa low herda grey_low
    scen_rates, disagreement = sim.build_scenario_rates("rates_csv", tmp_path / "r.csv")
    assert scen_rates == rates
    assert disagreement == sim.DISAGREEMENT_THESIS


def test_load_rates_csv_validation(tmp_path: Path) -> None:
    full = [(b, 0.1, 0.1, 0.1) for b in sim.BAND_NAMES]
    with pytest.raises(ValueError, match="desconhecidas"):
        sim.load_rates_csv(_write_rates(tmp_path / "a.csv", [*full, ("zzz", 0, 0, 0)]))
    with pytest.raises(ValueError, match="duplicada"):
        sim.load_rates_csv(_write_rates(tmp_path / "b.csv", [*full, full[0]]))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        sim.load_rates_csv(_write_rates(tmp_path / "c.csv", [("grey_low", 1.5, 0, 0)]))
    with pytest.raises(ValueError, match="sem taxas"):
        sim.load_rates_csv(_write_rates(tmp_path / "d.csv", full[:3]))
    pd.DataFrame({"band": ["low"], "fp": [0.1]}).to_csv(tmp_path / "e.csv", index=False)
    with pytest.raises(ValueError, match="colunas ausentes"):
        sim.load_rates_csv(tmp_path / "e.csv")
    with pytest.raises(ValueError, match="exige"):
        sim.build_scenario_rates("rates_csv")


def test_cli_rates_csv_and_new_dimensions() -> None:
    base = ["--input", "x.csv", "--out-csv", "a.csv", "--out-md", "a.md"]
    args = sim.build_parser().parse_args(base)
    assert args.rates_csv is None
    assert args.field_sets == ["composites"]
    assert args.negative_labels == ["llm_nonmatch", "verified_only"]
    assert args.human_budgets == [25, 50, 100]
    args = sim.build_parser().parse_args(
        [
            *base,
            "--rates-csv",
            "taxas.csv",
            "--negative-labels",
            "verified_only",
            "--human-budgets",
            "25,100",
            "--scenarios",
            "perfect",
        ]
    )
    assert args.rates_csv == Path("taxas.csv")
    assert args.negative_labels == ["verified_only"]
    assert args.human_budgets == [25, 100]
    with pytest.raises(SystemExit):
        sim.build_parser().parse_args([*base, "--negative-labels", "zz"])
    with pytest.raises(SystemExit):
        sim.build_parser().parse_args([*base, "--human-budgets", "0"])
    # o cenário rates_csv sem --rates-csv é recusado antes de ler os dados
    with pytest.raises(SystemExit):
        sim.main([*base, "--scenarios", "rates_csv"])


# --------------------------------------------------------------------------
# Esforço humano: métricas e bases H0/H1
# --------------------------------------------------------------------------


def test_human_metrics_prefix_and_k() -> None:
    truth = np.array([0, 1, 0, 1, 1, 0, 1], dtype=bool)
    below = np.array([0, 1, 0, 0, 1, 0, 1], dtype=bool)
    llm = np.arange(1, 8, dtype=float) * 10
    m = sim.human_metrics(truth, below, llm, 5)
    assert m is not None
    assert m["human_decisions"] == 5
    assert m["verified"] == 3  # posições 1, 3, 4
    assert m["verified_below"] == 2  # posições 1 e 4
    assert m["llm_reviews"] == 50.0
    assert m["dec_per_verified"] == pytest.approx(5 / 3)
    assert m["hdec_to_k5"] != m["hdec_to_k5"]  # NaN: só 2 abaixo do corte
    full = sim.human_metrics(truth, below, llm, 7)
    assert full is not None and full["verified_below"] == 3
    assert sim.human_metrics(truth, below, llm, 8) is None  # não chegou a 8
    assert sim.human_metrics(truth, below, llm, 0) is None
    none = sim.human_metrics(np.zeros(3, dtype=bool), below[:3], llm[:3], 3)
    assert none is not None and "dec_per_verified" not in none


def _oracle(prep: sim.Prepared, scenario: str, seed: int) -> sim.BandedOracle:
    rates, disagreement = sim.build_scenario_rates(scenario)
    return sim.BandedOracle(
        dict(zip(prep.pair_ids.tolist(), prep.truth.tolist(), strict=True)),
        dict(zip(prep.pair_ids.tolist(), prep.nota.tolist(), strict=True)),
        rates,
        disagreement_rate=disagreement,
        seed=seed,
    )


def _effort_base() -> dict[str, str]:
    return {
        "pool": "all",
        "mode": "human",
        "reviewer": "perfect",
        "seed": 1,
        "variant": "default",
        "field_set": "composites",
        "negative_labels": "na",
        "human_budget": "na",
    }


def test_h0_h1_effort_baselines(prep: sim.Prepared) -> None:
    oracle = _oracle(prep, "perfect", 1)
    order = np.argsort(-prep.nota, kind="stable")
    base = _effort_base()
    h0 = pd.DataFrame(
        sim.effort_rows_baseline(
            "H0", base, order, prep.pair_ids, prep.truth, prep.below, oracle
        )
    ).set_index("budget")
    for b in sim.HUMAN_MARKS:
        assert h0.loc[f"B{b}", "human_decisions"] == b
        assert h0.loc[f"B{b}", "llm_reviews"] == 0  # humano sozinho: sem LLM
        assert h0.loc[f"B{b}", "verified"] == prep.truth[order[:b]].sum()
    assert h0.loc["run", "human_decisions"] == max(sim.HUMAN_MARKS)

    h1 = pd.DataFrame(
        sim.effort_rows_baseline(
            "H1", base, order, prep.pair_ids, prep.truth, prep.below, oracle
        )
    ).set_index("budget")
    # revisor perfeito: todo MATCH do LLM é verdadeiro, então cada decisão confirma;
    # o pool sintético tem só ~305 MATCH, logo B=400 não é alcançado por H1
    assert "B400" not in h1.index
    assert h1.loc["run", "human_decisions"] == int(prep.truth.sum())
    marks = [b for b in sim.HUMAN_MARKS if f"B{b}" in h1.index]
    assert marks == [50, 100, 200]
    for b in marks:
        assert h1.loc[f"B{b}", "verified"] == b
        assert h1.loc[f"B{b}", "dec_per_verified"] == 1.0
    # o LLM consome ao menos tantas revisões quanto decisões, e cresce com B
    llm = [h1.loc[f"B{b}", "llm_reviews"] for b in marks]
    assert all(x >= b for x, b in zip(llm, marks, strict=True))
    assert llm == sorted(llm)
    with pytest.raises(ValueError):
        sim.effort_rows_baseline(
            "H9", base, order, prep.pair_ids, prep.truth, prep.below, oracle
        )


def test_h1_with_errors_has_false_positives(prep: sim.Prepared) -> None:
    oracle = _oracle(prep, "thesis_bands", 1)
    order = np.argsort(-prep.nota, kind="stable")
    h1 = pd.DataFrame(
        sim.effort_rows_baseline(
            "H1", _effort_base(), order, prep.pair_ids, prep.truth, prep.below, oracle
        )
    ).set_index("budget")
    # com FP no revisor, nem toda decisão humana verifica um par
    last = max(h1.index[h1.index != "run"], key=lambda s: int(s[1:]))
    assert h1.loc[last, "verified"] <= int(last[1:])
    assert h1.loc[last, "dec_per_verified"] >= 1.0


# --------------------------------------------------------------------------
# Execução
# --------------------------------------------------------------------------


def test_prepare_hides_par_and_target(prep: sim.Prepared) -> None:
    assert "PAR" not in prep.df.columns
    assert "TARGET" not in prep.df.columns
    assert prep.truth.sum() > 0
    assert prep.truth.shape == (prep.n,)
    assert len(set(prep.pair_ids.tolist())) == prep.n


def _rows(
    prep: sim.Prepared,
    seed: int,
    *,
    pool: str = "all",
    scenario: str = "perfect",
    negative_labels: str = "llm_nonmatch",
    human_budget: int | None = 100,
    with_baselines: bool = True,
    with_effort: bool = True,
    max_reviews: int = 300,
) -> pd.DataFrame:
    rows = sim.run_one(
        prep,
        pool,
        "human",
        scenario,
        seed,
        negative_labels=negative_labels,
        human_budget=human_budget,
        max_reviews=max_reviews,
        fs_cache=None,
        with_baselines=with_baselines,
        with_effort=with_effort,
    )
    return pd.DataFrame(rows)


def test_run_one_is_deterministic_by_seed(prep: sim.Prepared) -> None:
    a = _rows(prep, 5)
    b = _rows(prep, 5)
    cols = [c for c in a.columns if c != "elapsed_s"]
    pd.testing.assert_frame_equal(a[cols], b[cols])


def test_human_decides_every_gate_item(prep: sim.Prepared) -> None:
    # se algum item ficasse sem decisão, a sessão levantaria ValueError
    df = _rows(prep, 3, scenario="thesis_bands", human_budget=100)
    run = df[(df["method"] == "session") & (df["budget"] == "run")].iloc[0]
    assert run["human_decisions"] > 0
    assert run["verified"] <= run["human_decisions"]
    # humano perfeito: tudo o que ele verificou é verdadeiro
    assert run["verified"] == run["verified_total"]
    # o orçamento 100 por rodada e os rótulos do humano
    assert run["human_decisions"] <= 100 * run["rounds"]
    perfect = _rows(prep, 3, human_budget=100)
    p_run = perfect[(perfect["method"] == "session") & (perfect["budget"] == "run")]
    assert p_run.iloc[0]["fp_llm_match"] == 0
    assert p_run.iloc[0]["fp_caught"] == 0


def test_over_budget_and_unverified_accounting(prep: sim.Prepared) -> None:
    df = _rows(prep, 4, scenario="thesis_bands", human_budget=10, max_reviews=400)
    run = df[(df["method"] == "session") & (df["budget"] == "run")].iloc[0]
    assert run["n_over_budget"] > 0  # orçamento de 10 por rodada estoura
    assert run["human_decisions"] <= 10 * run["rounds"]
    # todo FP do LLM foi detectado pelo humano ou ficou sem verificação
    assert run["fp_llm_match"] == run["fp_caught"] + run["fp_unverified"]
    assert run["n_llm_match_unverified"] == run["tp_unverified"] + run["fp_unverified"]


def test_session_human_budget_rows_and_budget_parity(prep: sim.Prepared) -> None:
    df = _rows(prep, 5)
    sess = df[df["method"] == "session"]
    # curvas por orçamento humano: B50 existe e verificados não decrescem com B
    b_rows = sess[sess["budget"].str.startswith("B")]
    assert "B50" in set(b_rows["budget"])
    b_rows = b_rows.assign(b=b_rows["budget"].str[1:].astype(int)).sort_values("b")
    assert b_rows["verified"].is_monotonic_increasing
    assert b_rows["llm_reviews"].is_monotonic_increasing
    # bases a orçamento de LLM igual ao da fila
    run = df[(df["budget"] == "run") & df["method"].isin(["session", *sim.BASELINES])]
    n_run = run.set_index("method").loc["session", "reviews"]
    assert (run["reviews"] == n_run).all()
    # H0/H1 saem uma vez, com chaves "na"
    effort = df[df["method"].isin(sim.EFFORT_BASELINES)]
    assert set(effort["method"]) == {"H0", "H1"}
    assert (effort["negative_labels"] == "na").all()
    assert (effort["human_budget"] == "na").all()
    # sem with_effort, nada de H0/H1
    assert not _rows(prep, 5, with_effort=False)["method"].isin(["H0", "H1"]).any()


def test_negative_labels_are_forwarded_and_keyed(prep: sim.Prepared) -> None:
    a = _rows(prep, 2, negative_labels="llm_nonmatch", with_baselines=False)
    b = _rows(prep, 2, negative_labels="verified_only", with_baselines=False)
    assert set(a["negative_labels"]) == {"llm_nonmatch"}
    assert set(b["negative_labels"]) == {"verified_only"}
    assert set(b["human_budget"]) == {"100"}


def test_run_one_pool_below_cut_restricts_pairs(prep: sim.Prepared) -> None:
    df = _rows(prep, 2, pool="below_cut")
    assert df[df["method"] == "session"]["reviews"].max() <= int(prep.below.sum())


def test_run_one_auto_has_no_human(prep: sim.Prepared) -> None:
    rows = sim.run_one(
        prep, "all", "auto", "thesis_bands", 3, max_reviews=100, human_budget=None
    )
    df = pd.DataFrame(rows)
    sess = df[(df["method"] == "session") & (df["budget"] == "run")].iloc[0]
    assert sess["human_decisions"] == 0
    assert sess["verified"] == 0
    assert sess["negative_labels"] == "na" and sess["human_budget"] == "na"
    assert (
        sess["n_llm_match_unverified"] == sess["tp_unverified"] + sess["fp_unverified"]
    )
    assert not df["method"].isin(["H0", "H1"]).any()
    assert any(c.startswith("finish_reason=") for c in df.columns)
    assert any(c.startswith("p0_method=") for c in df.columns)


def test_build_jobs_grid() -> None:
    seeds = [1, 2, 3]
    jobs = sim.build_jobs(seeds=seeds, scenarios=["perfect", "thesis_bands"])
    human = [j for j in jobs if j.mode == "human" and j.overrides is None]
    # pool × revisor × negative_labels × orçamento × sementes
    assert len(human) == 2 * 2 * 2 * 3 * 3
    assert {j.negative_labels for j in human} == set(sim.NEGATIVE_LABELS)
    assert {j.human_budget for j in human} == set(sim.HUMAN_BUDGETS)
    auto = [j for j in jobs if j.mode == "auto"]
    assert len(auto) == 2 * 2 * 3 and all(j.human_budget is None for j in auto)
    sens = [j for j in jobs if j.overrides is not None]
    assert len(sens) == 3
    assert all(j.pool == "below_cut" and j.overrides == sim.SENSITIVITY for j in sens)
    # H0/H1 uma vez por (pool, revisor, semente), só na configuração padrão
    effort = [j for j in jobs if j.effort]
    assert len(effort) == 2 * 2 * 3
    assert all(
        j.negative_labels == sim.DEFAULT_NEGATIVE
        and j.human_budget == sim.DEFAULT_HUMAN_BUDGET
        and j.overrides is None
        for j in effort
    )
    # sem sensibilidade e sem bloco automático
    slim = sim.build_jobs(
        seeds=seeds,
        scenarios=["perfect"],
        negative_labels=["llm_nonmatch"],
        human_budgets=[50],
        sensitivity=False,
        auto_block=False,
    )
    assert len(slim) == 2 * 3 and all(j.mode == "human" for j in slim)


def test_aggregate_by_human_budget(prep: sim.Prepared) -> None:
    runs = pd.concat(
        [_rows(prep, s, scenario="thesis_bands") for s in (1, 2)], ignore_index=True
    )
    agg = sim.aggregate(runs)
    assert {"mean", "min", "max", "n_valid", "n_seeds", "negative_labels"} <= set(
        agg.columns
    )
    assert (agg["min"] <= agg["mean"]).all()
    assert (agg["mean"] <= agg["max"]).all()
    h0 = agg[(agg["method"] == "H0") & (agg["budget"] == "B100")]
    v = h0[h0["metric"] == "verified"].iloc[0]
    assert v["n_seeds"] == 2 and v["n_valid"] == 2
    # a média por orçamento humano é a média dos runs
    direct = runs[(runs["method"] == "H0") & (runs["budget"] == "B100")]["verified"]
    assert v["mean"] == pytest.approx(direct.mean())
    assert not {"pair_id", "COMPREC", "REFREC"} & set(agg.columns)


def _meta() -> dict[str, object]:
    return {
        "rates": sim.build_scenario_rates("thesis_bands")[0],
        "partition": (1, 2, 3),
        "partition_ok": False,
        "seeds": [1, 2],
        "scenarios": ["perfect", "thesis_bands"],
        "negative_labels": ["llm_nonmatch"],
        "human_budgets": [100],
        "max_reviews": 300,
        "elapsed_sessions": 1.0,
        "command": "python notebooks/sim_fila_autocalibravel.py ...",
        "default_field_set": "composites",
        "rates_csv": None,
    }


def test_render_markdown_effort_sections_and_blocks(prep: sim.Prepared) -> None:
    parts = [_rows(prep, s, scenario="thesis_bands", human_budget=50) for s in (1, 2)]
    # o padrão (50 por rodada) é o que entra nas tabelas de esforço
    auto = [
        pd.DataFrame(
            sim.run_one(
                prep,
                "all",
                "auto",
                "thesis_bands",
                s,
                max_reviews=100,
                human_budget=None,
            )
        )
        for s in (1, 2)
    ]
    agg = sim.aggregate(pd.concat([*parts, *auto], ignore_index=True))
    old_cols = ["negative_labels", "human_budget"]
    v1 = agg[agg["field_set"] == "composites"].drop(columns=[*old_cols, "field_set"])
    v2 = agg.drop(columns=old_cols).assign(field_set="cdr_subscores")
    v2_comp = v2.assign(field_set="composites")
    first = sim.render_markdown(
        agg, meta=_meta(), v1=v1, v2=pd.concat([v2, v2_comp], ignore_index=True)
    )
    assert "Não é uma avaliação com LLM real" in first
    assert "## Esforço humano" in first
    assert "H0 humano sozinho" in first and "H1 nota + LLM tria" in first
    assert "H2 fila com portão revisto" in first
    assert "Viés de sobrevivência" in first
    assert "MATCH bruto do LLM" in first  # limitação declarada
    assert "## v1 × v2 × v3" in first and "→" in first
    assert "Regra de parada por posterior" in first
    edited = first.replace(
        "(texto ainda não escrito)", "texto manual"
    )  # os dois blocos viram texto manual
    again = sim.render_markdown(agg, meta=_meta(), v1=None, v2=None, previous=edited)
    assert again.count("texto manual") == 2
    assert "comparação omitida" in again


# --------------------------------------------------------------------------
# Revisores calibrados no Haiku real (taxas medidas e replay do fixture)
# --------------------------------------------------------------------------


def _counts_csv(path: Path) -> Path:
    """CSV de calibração mínimo (linhas ``todos`` por faixa, mais ruído)."""
    rows = [
        # faixa, lado, n_pares, n_nao_pares, fn, fp, unsure_pares, unsure_nao_pares
        ("low", "todos", 3, 100, 2, 0, 0, 5),
        ("grey_low", "todos", 17, 100, 5, 1, 4, 2),
        ("grey_mid", "todos", 57, 100, 4, 0, 5, 3),
        ("grey_high", "todos", 43, 100, 4, 1, 2, 2),
        ("near_high", "todos", 34, 68, 10, 0, 6, 5),
        ("high", "todos", 93, 2, 20, 0, 16, 0),
        ("high", "acima", 86, 0, 17, 0, 16, 0),  # ignorada: só lado = todos
        ("total", "todos", 247, 470, 45, 2, 33, 17),  # ignorada: não é faixa
    ]
    cols = [
        "faixa",
        "lado",
        "n_pares",
        "n_nao_pares",
        "fn",
        "fp",
        "unsure_pares",
        "unsure_nao_pares",
    ]
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)
    return path


def test_smoothed_rates_use_jeffreys_on_the_counts(tmp_path: Path) -> None:
    rates = sim.smoothed_rates_from_counts(_counts_csv(tmp_path / "c.csv"))
    assert set(rates) == set(sim.BAND_NAMES)
    fp, fn, unsure = rates["low"]
    assert fp == pytest.approx(0.5 / 101)  # (0 + 0,5) / (100 + 1)
    assert fn == pytest.approx(2.5 / 4)  # (2 + 0,5) / (3 + 1)
    assert unsure == pytest.approx(5.5 / 104)  # (0 + 5 + 0,5) / (103 + 1)
    # high tem só 2 não pares: a suavização evita FP = 0 e o puxa para cima
    assert rates["high"][0] == pytest.approx(0.5 / 3)
    bad = pd.read_csv(tmp_path / "c.csv").drop(columns=["fp"])
    bad.to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError, match="colunas ausentes"):
        sim.smoothed_rates_from_counts(tmp_path / "bad.csv")
    short = pd.read_csv(tmp_path / "c.csv").query("faixa != 'high'")
    short.to_csv(tmp_path / "short.csv", index=False)
    with pytest.raises(ValueError, match="sem contagens"):
        sim.smoothed_rates_from_counts(tmp_path / "short.csv")


def test_real_scenarios_rates(tmp_path: Path) -> None:
    measured = _write_rates(
        tmp_path / "m.csv", [(b, 0.0, 0.1, 0.05) for b in sim.BAND_NAMES]
    )
    inputs = sim.RealInputs(
        measured_rates_csv=measured, counts_csv=_counts_csv(tmp_path / "c.csv")
    )
    m, dm = sim.build_scenario_rates("measured", inputs=inputs)
    assert m["grey_mid"] == (0.0, 0.1, 0.05) and dm == sim.DISAGREEMENT_THESIS
    s, ds = sim.build_scenario_rates("measured_smoothed", inputs=inputs)
    assert s["low"][0] > 0  # Jeffreys nunca dá taxa zero
    f, df_ = sim.build_scenario_rates("fixture_replay", inputs=inputs)
    assert f == s and df_ == ds  # o fallback do replay é o measured_smoothed
    for scen in ("measured", "measured_smoothed", "fixture_replay"):
        with pytest.raises(ValueError, match="exige"):
            sim.build_scenario_rates(scen)
    assert set(sim.REAL_REVIEWERS) == {
        "measured",
        "measured_smoothed",
        "fixture_replay",
    }


def _fixture_file(path: Path, prep: sim.Prepared, n: int = 300) -> Path:
    """Fixture sintético: as ``n`` primeiras linhas, todas MATCH (consenso)."""
    import json

    import llm_real_eval as lre

    agent = {
        "decision": "MATCH",
        "confidence": 0.9,
        "reason_codes": ["X"],
        "tokens_prompt": 1,
        "tokens_completion": 1,
        "latency_s": 0.1,
    }
    rows = []
    for pid in prep.pair_ids[:n]:
        rec = {
            "pair_id": str(pid),
            "model": "m",
            "prompt_version": "v4",
            "remove_fields": ["features.nota_final"],
            "tokens_prompt": 2,
            "tokens_completion": 2,
            "result": {
                "decision": "MATCH",
                "protocol": "consensus",
                "confidence": 0.9,
                "reason_codes": ["X"],
                "total_latency_s": 0.2,
                "agent_a": agent,
                "agent_b": agent,
                "arbiter": None,
            },
        }
        rows.append(lre.fixture_row(rec))
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def _real_inputs(tmp_path: Path, prep: sim.Prepared) -> sim.RealInputs:
    return sim.RealInputs(
        measured_rates_csv=_write_rates(
            tmp_path / "m.csv", [(b, 0.0, 0.0, 0.0) for b in sim.BAND_NAMES]
        ),
        counts_csv=_counts_csv(tmp_path / "c.csv"),
        fixture_jsonl=_fixture_file(tmp_path / "fx.jsonl", prep),
    )


def test_replay_reviewer_replays_and_falls_back(
    prep: sim.Prepared, tmp_path: Path
) -> None:
    inputs = _real_inputs(tmp_path, prep)
    truth = dict(zip(prep.pair_ids.tolist(), prep.truth.tolist(), strict=True))
    nota = dict(zip(prep.pair_ids.tolist(), prep.nota.tolist(), strict=True))
    perfect = {b: (0.0, 0.0, 0.0) for b in sim.BAND_NAMES}
    assert inputs.fixture_jsonl is not None

    def build(seed: int) -> sim.ReplayReviewer:
        return sim.ReplayReviewer(
            inputs.fixture_jsonl,
            truth,
            nota,
            perfect,
            disagreement_rate=0.0,
            seed=seed,
        )

    a, b = build(1), build(2)
    inside = [str(p) for p in prep.pair_ids[:50]]
    outside = [str(p) for p in prep.pair_ids[300:350]]
    # presente no fixture: o veredito real (MATCH), fixo entre sementes
    verdicts_a = [a.verdict(p) for p in inside]
    assert set(verdicts_a) == {"MATCH"}
    assert verdicts_a == [b.verdict(p) for p in inside]
    assert a.n_replayed == 50 and a.n_missing == 0
    # ausente: o fallback por taxas (perfeito = verdade)
    got = [a.verdict(p) for p in outside]
    want = ["MATCH" if truth[p] else "NONMATCH" for p in outside]
    assert got == want
    assert a.n_missing == 50


def test_run_one_fixture_replay_reports_coverage(
    prep: sim.Prepared, tmp_path: Path
) -> None:
    inputs = _real_inputs(tmp_path, prep)
    rows = sim.run_one(
        prep,
        "all",
        "human",
        "fixture_replay",
        3,
        human_budget=100,
        max_reviews=300,
        with_effort=True,
        inputs=inputs,
    )
    df = pd.DataFrame(rows)
    run = df[(df["method"] == "session") & (df["budget"] == "run")].iloc[0]
    assert run["reviewer"] == "fixture_replay"
    assert run["replay_reviews"] + run["fallback_reviews"] == run["reviews"]
    assert 0.0 < run["replay_fraction"] < 1.0
    assert 0.0 <= run["replay_fraction_bases"] <= 1.0
    assert set(df[df["method"].isin(sim.EFFORT_BASELINES)]["method"]) == {"H0", "H1"}
    # sem ReplayReviewer a cobertura não existe
    plain = sim.run_one(prep, "all", "human", "perfect", 3, max_reviews=100)
    assert "replay_fraction" not in pd.DataFrame(plain).columns
    with pytest.raises(ValueError, match="exige"):
        sim.run_one(prep, "all", "human", "fixture_replay", 3, max_reviews=100)


def test_run_one_measured_scenarios(prep: sim.Prepared, tmp_path: Path) -> None:
    inputs = _real_inputs(tmp_path, prep)
    for scen in ("measured", "measured_smoothed"):
        df = pd.DataFrame(
            sim.run_one(
                prep,
                "all",
                "human",
                scen,
                2,
                human_budget=50,
                max_reviews=200,
                with_baselines=False,
                inputs=inputs,
            )
        )
        assert set(df["reviewer"]) == {scen}
        assert (df["human_decisions"].dropna() >= 0).all()


def test_read_agg_keeps_na_and_text_budgets(tmp_path: Path) -> None:
    agg = pd.DataFrame(
        {
            "negative_labels": ["na", "llm_nonmatch"],
            "human_budget": ["na", "50"],
            "reviewer": ["perfect", "measured"],
        }
    )
    agg.to_csv(tmp_path / "a.csv", index=False)
    back = sim.read_agg(tmp_path / "a.csv")
    assert back["negative_labels"].tolist() == ["na", "llm_nonmatch"]
    assert back["human_budget"].tolist() == ["na", "50"]  # 'na' não vira NaN


def test_reproduction_commands_and_new_cli_flags() -> None:
    base, real = sim.reproduction_commands(10, 2000)
    assert "--scenarios perfect,thesis_bands" in base
    assert "--scenarios measured,measured_smoothed,fixture_replay" in real
    assert "--append-csv" in real and "--no-auto-block" in real
    args = sim.build_parser().parse_args(
        [
            "--input",
            "x",
            "--out-csv",
            "a",
            "--out-md",
            "b",
            "--scenarios",
            "measured,measured_smoothed,fixture_replay",
            "--append-csv",
            "--no-sensitivity",
        ]
    )
    assert args.scenarios == ["measured", "measured_smoothed", "fixture_replay"]
    assert args.append_csv and args.no_sensitivity
    assert args.fixture_jsonl.name == "haiku-5-5_v4_sem-nota.jsonl"
    assert args.counts_csv.name == "llm_real_haiku_sem_nota.csv"


def test_render_markdown_real_section(prep: sim.Prepared, tmp_path: Path) -> None:
    inputs = _real_inputs(tmp_path, prep)
    parts = [
        pd.DataFrame(
            sim.run_one(
                prep,
                "all",
                "human",
                scen,
                s,
                human_budget=50,
                max_reviews=300,
                with_effort=True,
                inputs=inputs,
            )
        )
        for s in (1, 2)
        for scen in ("thesis_bands", "fixture_replay")
    ]
    agg = sim.aggregate(pd.concat(parts, ignore_index=True))
    meta = {
        **_meta(),
        "rates_measured": sim.load_rates_csv(inputs.measured_rates_csv),
        "rates_smoothed": sim.smoothed_rates_from_counts(inputs.counts_csv),
        "fixture": {
            "n_pairs": 300,
            "sha256": "abc",
            "origin": {"calibration_sample": {"n_pairs": 100}},
        },
    }
    md = sim.render_markdown(agg, meta=meta)
    assert "## Revisor calibrado no Haiku real" in md
    assert "fixo entre sementes" in md
    assert "Fração das revisões vinda do fixture" in md
    assert "fixture_replay" in md and "measured_smoothed" in md
    # sem runs reais, a seção não existe
    base_only = agg[~agg["reviewer"].isin(sim.REAL_REVIEWERS)]
    assert "Revisor calibrado no Haiku real" not in sim.render_markdown(
        base_only, meta=_meta()
    )
