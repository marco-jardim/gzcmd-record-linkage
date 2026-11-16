"""Testes rápidos e offline dos experimentos de engenharia de filas."""

from __future__ import annotations

import itertools
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import exp_engenharia_filas as exp
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from gzcmd_record_linkage.self_calibrating_queue import (
    SelfCalibratingConfig,
    SelfCalibratingSession,
)

N_SMALL = 3000


def _cell(**kw: Any) -> exp.Cell:
    return exp.Cell(n=N_SMALL, **kw)


def _auc(score: np.ndarray, y: np.ndarray) -> float:
    ranks = stats.rankdata(score)
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ---------------------------------------------------------------------------
# Gerador
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prevalence", [0.001, 0.01, 0.05, 0.2])
def test_generator_respects_prevalence_and_columns(prevalence: float) -> None:
    inst = exp.generate(exp.Cell(prevalence=prevalence, n=5000), seed=1)
    assert int(inst.y.sum()) == max(2, round(prevalence * 5000))
    needed = {
        "COMPREC",
        "REFREC",
        "PASSO",
        "nota final",
        "nome_score_total",
        "mae_score_total",
        "mae_missing",
        "dtnasc_score_total",
        "endereco_score_total",
        "municipio_score",
        "dtnasc_all_zero",
        "endereco_zero",
    }
    assert needed <= set(inst.df.columns)
    assert "PAR" not in inst.df.columns and "TARGET" not in inst.df.columns
    assert len(set(inst.pair_ids)) == inst.n
    assert inst.df["nota final"].between(0, 10).all()


def test_generator_is_deterministic_per_seed() -> None:
    a = exp.generate(_cell(prevalence=0.05), seed=3)
    b = exp.generate(_cell(prevalence=0.05), seed=3)
    c = exp.generate(_cell(prevalence=0.05), seed=4)
    pd.testing.assert_frame_equal(a.df, b.df)
    assert (a.llm_all == b.llm_all).all() and (a.human == b.human).all()
    assert not (a.y == c.y).all()


def test_signal_knob_makes_municipio_informative() -> None:
    cell = exp.Cell(prevalence=0.1, n=6000)
    off = exp.generate(cell, seed=2)
    on = exp.generate(exp.Cell(prevalence=0.1, n=6000, signal=True), seed=2)
    mun_off = off.df["municipio_score"].to_numpy()
    mun_on = on.df["municipio_score"].to_numpy()
    assert _auc(mun_off, off.y) == pytest.approx(0.5, abs=0.06)
    assert _auc(mun_on, on.y) > 0.7
    # a régua não usa o município: a nota é a mesma função dos campos
    assert _auc(on.nota, on.y) == pytest.approx(_auc(off.nota, off.y), abs=0.05)


def test_ruler_quality_knob_degrades_the_ruler() -> None:
    base = exp.generate(exp.Cell(prevalence=0.1, n=6000), seed=2)
    wrong = exp.generate(
        exp.Cell(prevalence=0.1, n=6000, ruler="wrong", degree=3), seed=2
    )
    noisy = exp.generate(
        exp.Cell(prevalence=0.1, n=6000, ruler="noise", degree=3), seed=2
    )
    auc_base = _auc(base.nota, base.y)
    assert _auc(wrong.nota, wrong.y) < auc_base - 0.02
    assert _auc(noisy.nota, noisy.y) < auc_base - 0.02


def test_hard_subgroup_and_nonmonotone_knobs() -> None:
    inst = exp.generate(exp.Cell(prevalence=0.1, n=6000, hard=True), seed=2)
    assert int(inst.hard.sum()) == round(exp.HARD_SHARE * inst.y.sum())
    assert (inst.hard <= inst.y).all()
    # difíceis ficam mais embaixo na régua
    assert inst.nota[inst.hard].mean() < inst.nota[inst.y & ~inst.hard].mean()
    nm = exp.generate(exp.Cell(prevalence=0.02, n=6000, nonmono=True), seed=2)
    base = exp.generate(exp.Cell(prevalence=0.02, n=6000), seed=2)
    lo, hi = (int(f * base.n) for f in exp.POCKET_RANK)
    srt = np.sort(base.nota)[::-1]
    high, low = srt[lo] + 0.01, srt[hi] - 0.01
    in_pocket_nm = (nm.nota <= high) & (nm.nota >= low)
    in_pocket_base = (base.nota <= high) & (base.nota >= low)
    gain = (nm.y & in_pocket_nm).sum() - (base.y & in_pocket_base).sum()
    assert gain > 0.3 * nm.y.sum()


def test_llm_rates_match_measured_and_concentration() -> None:
    n = 200_000
    rng = np.random.default_rng(0)
    y = rng.random(n) < 0.3
    nota = np.where(y, rng.normal(7.5, 0.4, n), rng.normal(2.0, 1.0, n))
    hard = y & (rng.random(n) < 0.3)
    u = rng.random((n, 2))
    llm = exp.simulate_llm(y, hard, nota, u)
    assert (llm[~y] == exp.MATCH).mean() == pytest.approx(exp.FP_LLM, abs=0.002)
    band = exp.band_index(nota)
    gh = y & (band == 3)  # grey_high
    expected_fn = exp.LLM_FN[3]
    nonmatch = (llm[gh] == exp.NONMATCH).mean()
    unsure = (llm[gh] == exp.UNSURE).mean()
    assert nonmatch == pytest.approx(expected_fn * (1 - exp.LLM_UNSURE[3]), abs=0.02)
    assert unsure == pytest.approx(exp.LLM_UNSURE[3], abs=0.02)
    conc = exp.simulate_llm(y, hard, nota, u, fn_concentrated=True)
    h = hard & (conc != exp.UNSURE)
    easy = y & ~hard & (conc != exp.UNSURE)
    assert (conc[h] == exp.NONMATCH).mean() == pytest.approx(exp.HARD_FN_LLM, abs=0.03)
    assert (conc[easy] == exp.NONMATCH).mean() < (llm[y & ~hard] == exp.NONMATCH).mean()


def test_human_simulation_rates() -> None:
    n = 100_000
    rng = np.random.default_rng(1)
    y = rng.random(n) < 0.5
    dec = exp.simulate_human(y, rng.random((n, 2)), fn=0.1, fp=0.02)
    assert (~dec[y]).mean() == pytest.approx(0.1, abs=0.01)
    assert dec[~y].mean() == pytest.approx(0.02, abs=0.005)
    perfect = exp.simulate_human(y, rng.random((n, 2)), fn=0.0, fp=0.0)
    assert (perfect == y).all()


# ---------------------------------------------------------------------------
# Estratégias
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def inst() -> exp.Instance:
    return exp.generate(_cell(prevalence=0.05, signal=True), seed=5)


def test_s0_is_a_permutation_ordered_by_ruler(inst: exp.Instance) -> None:
    order = exp.ruler_order(inst.nota)
    assert sorted(order.tolist()) == list(range(inst.n))
    assert (np.diff(inst.nota[order]) <= 0).all()


def test_s2_without_labels_orders_like_s0(inst: exp.Instance) -> None:
    spec = exp.build_spec(inst)
    eta = spec.eta(spec.initial())
    by_model = np.lexsort((np.arange(inst.n), -eta))
    assert (by_model == exp.ruler_order(inst.nota)).all()
    path = exp.adaptive_path(spec, inst.human, steps=exp.REFIT_EVERY)
    assert (path.chosen == exp.ruler_order(inst.nota)[: exp.REFIT_EVERY]).all()


def test_paths_are_unique_and_respect_budget(inst: exp.Instance) -> None:
    spec = exp.build_spec(inst)
    path = exp.adaptive_path(spec, inst.human, steps=120)
    assert len(path.chosen) == len(set(path.chosen.tolist())) == 120
    thompson = exp.thompson_path(inst.nota, inst.human, steps=120, seed=1)
    assert len(thompson) == len(set(thompson.tolist())) == 120
    spec3 = exp.build_spec(inst, llm_attribute=True)
    assert spec3.X.shape[1] > spec.X.shape[1]
    seen = np.flatnonzero(inst.llm_seen != exp.NONE)
    p4 = exp.adaptive_path(
        spec,
        inst.human,
        steps=60,
        extra_idx=seen,
        extra_y=(inst.llm_seen[seen] == exp.MATCH).astype(float),
    )
    assert len(set(p4.chosen.tolist())) == 60


def test_learning_moves_the_model_only_with_labels(inst: exp.Instance) -> None:
    spec = exp.build_spec(inst)
    path = exp.adaptive_path(spec, inst.human, steps=100)
    start, later = path.snapshots[0], path.snapshots[75]
    assert np.allclose(start, spec.initial())
    assert not np.allclose(later, start)
    # com prior forte (sigma ~ 0) o modelo não sai da régua
    tight = exp.build_spec(inst, sigma=1e-6)
    p_tight = exp.adaptive_path(tight, inst.human, steps=100)
    assert np.abs(p_tight.snapshots[75][2:]).max() < 1e-3


@pytest.mark.parametrize("budget", [50, 200])
def test_strategy_at_budget_uses_exactly_b_decisions(
    inst: exp.Instance, budget: int
) -> None:
    spec = exp.build_spec(inst)
    path = exp.adaptive_path(spec, inst.human, steps=budget)
    row = exp.strategy_at_budget(inst, path, spec, budget, seed=1)
    assert row["n_dec"] == budget
    assert row["tp"] <= min(budget, int(inst.y.sum()))
    assert 0.0 <= row["lo"] <= row["hi"]
    m = exp.sample_budget(budget)
    assert m == math.ceil(0.05 * budget)


def test_run_instance_pairs_strategies_and_budgets(inst: exp.Instance) -> None:
    out = exp.run_instance(inst, seed=1, budgets=(50, 100), with_s1=False)
    runs = pd.DataFrame(out["runs"])
    assert set(runs["strategy"]) == {"S0", "S2", "S2x", "S3", "S4", "S5"}
    assert (runs["n_dec"] == runs["B"]).all()
    assert (runs["tp"] <= runs["B"]).all()
    s0 = runs[runs["strategy"] == "S0"].set_index("B")["tp"]
    assert s0.loc[100] >= s0.loc[50]
    assert len(out["audit"]) == 2
    assert {r["rule"] for r in out["stops"]} == {"S2", "S6"}


def test_s1_runs_unchanged_session_on_the_generated_frame() -> None:
    small = exp.generate(exp.Cell(prevalence=0.05, n=1500), seed=2)
    rows = exp.run_s1(small, seed=2, budgets=(25, 50))
    by_b = {r["B"]: r for r in rows}
    assert by_b[25]["n_dec"] == 25
    assert by_b[0]["n_dec"] >= by_b[25]["n_dec"]
    cfg = SelfCalibratingConfig(mode="human", seed=2)
    session = SelfCalibratingSession(small.df, small.reviewer, cfg)
    assert session.config.human_budget_per_round == 50  # configuração padrão


# ---------------------------------------------------------------------------
# Amostra de estimação e Horvitz-Thompson
# ---------------------------------------------------------------------------


def test_allocation_sums_to_m_with_at_least_one_per_stratum() -> None:
    alloc = exp.allocate_sample(
        np.array([100.0, 900.0, 9000.0]), np.array([5, 3, 1.0]), 40
    )
    assert alloc.sum() == 40 and (alloc >= 1).all()
    tiny = exp.allocate_sample(
        np.array([100.0, 900.0, 9000.0]), np.array([5, 3, 1.0]), 2
    )
    assert tiny.sum() == 2
    capped = exp.allocate_sample(np.array([2.0, 3.0]), np.array([1.0, 1.0]), 50)
    assert capped.tolist() == [2, 3]


def test_draw_sample_has_known_inclusion_probabilities(inst: exp.Instance) -> None:
    pool = np.arange(inst.n)
    rng = np.random.default_rng(0)
    p_hat = np.full(inst.n, 0.05)
    sample = exp.draw_sample(pool, inst.nota, p_hat, 40, rng)
    assert sample.n_h.sum() == 40 == len(sample.idx)
    assert len(set(sample.idx.tolist())) == 40
    assert sample.N_h.sum() == inst.n
    for h in range(len(sample.N_h)):
        assert (sample.stratum == h).sum() == sample.n_h[h]


def test_horvitz_thompson_is_unbiased_on_a_known_population() -> None:
    strata = [np.array([1, 0, 0, 1]), np.array([0, 0, 1, 0, 0, 0])]
    sizes = (2, 3)
    errors = []
    combos = [
        list(itertools.combinations(range(len(s)), k))
        for s, k in zip(strata, sizes, strict=True)
    ]
    for pick in itertools.product(*combos):
        x = np.concatenate([strata[h][list(pick[h])] for h in range(2)]).astype(bool)
        lab = np.concatenate([np.full(sizes[h], h) for h in range(2)])
        sample = exp.Sample(
            idx=np.arange(x.size),
            N_h=np.array([4.0, 6.0]),
            n_h=np.array([2.0, 3.0]),
            stratum=lab,
        )
        est, lo, hi = exp.ht_remaining(sample, x)
        truth = sum(int(s.sum()) for s in strata) - int(x.sum())
        errors.append(est - truth)
        assert lo <= max(est, 0.0) <= hi + 1e-9
    assert np.mean(errors) == pytest.approx(0.0, abs=1e-12)


def test_ht_census_has_zero_width() -> None:
    sample = exp.Sample(
        idx=np.arange(4),
        N_h=np.array([4.0]),
        n_h=np.array([4.0]),
        stratum=np.zeros(4, dtype=int),
    )
    est, lo, hi = exp.ht_remaining(sample, np.array([True, False, True, False]))
    assert est == pytest.approx(0.0) and lo == hi == 0.0


# ---------------------------------------------------------------------------
# Critérios de decisão calculados à mão
# ---------------------------------------------------------------------------


def test_paired_ci_hand_example() -> None:
    mean, lo, hi = exp.paired_ci([1.0, 2.0, 3.0])
    half = stats.t.ppf(0.975, 2) / math.sqrt(3)
    assert mean == 2.0
    assert lo == pytest.approx(2.0 - half) and hi == pytest.approx(2.0 + half)
    assert exp.paired_ci([5.0, 5.0, 5.0]) == (5.0, 5.0, 5.0)
    assert math.isnan(exp.paired_ci([1.0])[1])
    assert exp.ci_includes_zero(-0.5, 4.0) and not exp.ci_includes_zero(0.1, 4.0)
    assert exp.ci_above_zero(0.1) and not exp.ci_above_zero(0.0)


def test_margin_is_max_of_one_pair_and_five_percent() -> None:
    assert exp.margin(10.0) == 1.0
    assert exp.margin(20.0) == 1.0
    assert exp.margin(100.0) == 5.0


def test_stop_rules_by_hand() -> None:
    # custo: rendimento esperado cai abaixo de c no 30o passo; nunca antes de 25
    pnext = np.concatenate([np.full(30, 0.5), np.full(20, 0.01)])
    assert exp.stop_cost(pnext, 0.05) == 30
    assert exp.stop_cost(np.full(40, 0.9), 0.05) == 40  # nunca para: usa o teto
    early = np.concatenate([np.full(5, 0.0), np.full(40, 0.9)])
    assert exp.stop_cost(early, 0.05) == 45  # a janela mínima de 25 evita parar cedo
    # tau: sem nenhum acerto, para quando (1 - tau)^(n + 1) < delta, ie n = 58
    n = exp.zero_needed = 58
    assert stats.beta.sf(exp.TAU, 1, 1 + n) < exp.DELTA_STOP
    assert stats.beta.sf(exp.TAU, 1, n) >= exp.DELTA_STOP
    assert exp.stop_tau(np.zeros(200, dtype=bool)) == n
    assert exp.stop_tau(np.ones(30, dtype=bool)) == 30  # nunca para: usa o teto


def test_regret_of_oracle_and_costs() -> None:
    n = 400
    y = np.zeros(n, dtype=bool)
    y[:10] = True
    nota = np.linspace(10, 0, n)
    inst = exp.Instance(
        df=pd.DataFrame({"x": np.zeros(n)}),
        y=y,
        hard=np.zeros(n, dtype=bool),
        nota=nota,
        llm_all=np.zeros(n, dtype=np.int8),
        llm_seen=np.zeros(n, dtype=np.int8),
        human=y.copy(),
        pair_ids=[str(i) for i in range(n)],
    )
    # caminho perfeito: os 10 positivos primeiro; S2 para quando o rendimento cai
    chosen = np.arange(n)
    pnext = np.where(np.arange(n) < 10, 0.9, 0.001)
    path = exp.PathResult(chosen, pnext, {0: np.zeros(1)})
    rows = {(r["cost"], r["rule"]): r for r in exp.regret_rows(inst, path, (0.05,))}
    s2 = rows[(0.05, "S2")]
    assert s2["k_stop"] == 25  # k_min = 25: antes disso não para
    assert s2["found"] == 10
    assert s2["oracle"] == pytest.approx(0.95 * 10)
    assert s2["regret"] == pytest.approx(0.95 * 10 - (10 - 0.05 * 25))
    s6 = rows[(0.05, "S6")]
    hit = (chosen < 10).astype(int)
    brute = next(
        k
        for k in range(1, n + 1)
        if k >= 1
        and stats.beta.sf(
            exp.TAU,
            1 + hit[max(0, k - 100) : k].sum(),
            1 + min(k, 100) - hit[max(0, k - 100) : k].sum(),
        )
        < exp.DELTA_STOP
    )
    assert s6["k_stop"] == brute == 109  # só quando 9 dos 10 acertos saem da janela
    assert s6["regret"] > s2["regret"]


def test_audit_row_by_hand() -> None:
    y = np.array([True, True, False, False, True, False])
    human = np.array([False, True, False, True, True, False])  # FN no 0; FP no 3
    llm = np.array(
        [exp.MATCH, exp.MATCH, exp.NONMATCH, exp.NONMATCH, exp.NONMATCH, exp.MATCH],
        dtype=np.int8,
    )
    inst = exp.Instance(
        df=pd.DataFrame({"x": np.zeros(6)}),
        y=y,
        hard=np.zeros(6, dtype=bool),
        nota=np.arange(6, 0, -1, dtype=float),
        llm_all=llm,
        llm_seen=llm,
        human=human,
        pair_ids=[str(i) for i in range(6)],
    )
    row = exp.audit_row(inst, exp.ruler_order(inst.nota), 6)
    # FN do humano: par 0. Re-revisão: LLM=MATCH e humano=NONMATCH: pares 0 e 5
    assert row == {"B": 6, "n_hfn": 1, "recovered": 1, "extras": 2}


def _runs(delta_s2: float, seeds: int = 4) -> pd.DataFrame:
    rows = []
    for seed in range(seeds):
        for b in exp.BUDGETS:
            for strat, shift in (("S0", 0.0), ("S2", delta_s2), ("S1", -3.0)):
                rows.append(
                    {
                        "cell_id": "base",
                        "seed": seed,
                        "strategy": strat,
                        "B": b,
                        "tp": 20.0 + seed + shift,
                        "hard_tp": 0.0,
                        "covered": 1.0,
                        "covered_hard": 1.0,
                        "lo": 0.0,
                        "hi": 5.0,
                    }
                )
    return pd.DataFrame(rows)


def _base_cells() -> pd.DataFrame:
    return pd.DataFrame([{**exp.asdict(exp.Cell()), "cell_id": "base"}])


def test_evaluate_h1_and_h6_hand_built() -> None:
    empty = pd.DataFrame()
    ok = {v["id"]: v for v in exp.evaluate(_runs(0.0), empty, empty, _base_cells())}
    assert ok["H1"]["status"] == "confirmada"  # S2-S0 = 0 e S1-S0 < 0
    assert ok["H6"]["status"] == "confirmada"
    assert ok["H12"]["status"] == "não testável"
    bad = {v["id"]: v for v in exp.evaluate(_runs(-5.0), empty, empty, _base_cells())}
    assert bad["H1"]["status"] == "refutada"  # IC de S2-S0 = -5 não inclui 0
    assert bad["H6"]["status"] == "refutada"  # -5 < -max(1, 5% de ~21)
    mild = {v["id"]: v for v in exp.evaluate(_runs(-0.5), empty, empty, _base_cells())}
    assert mild["H6"]["status"] == "confirmada"  # -0.5 >= -1


# ---------------------------------------------------------------------------
# Correções da auditoria: cobertura do LLM, régua usada, decisão humana
# ---------------------------------------------------------------------------

REAL = Path(r"D:\git\phd-research\data\COMPARADORSEMIDENT.csv")
FIXTURE = Path("data/llm_fixtures/haiku-5-5_v4_sem-nota.jsonl")
RATES = Path("data/llm_fixtures/haiku-5-5_v4_sem-nota.rates.csv")


def test_llm_coverage_depends_only_on_ruler_position() -> None:
    n = 40
    nota = np.linspace(9.0, 1.0, n)  # índice 0 = topo da régua
    y = np.zeros(n, dtype=bool)
    y[[2, 12, 20, 33]] = True
    # cobertura correlacionada com y: todo positivo coberto; negativos só no topo
    codes = np.full(n, exp.NONE, dtype=np.int8)
    codes[:5] = exp.NONMATCH
    codes[y] = exp.MATCH
    out = exp.restrict_to_window(nota, codes, window=8)
    assert (out[8:] == exp.NONE).all()  # nenhum positivo fora da janela entra
    assert (out[:8] == codes[:8]).all()
    assert int((out != exp.NONE).sum()) == int((codes[:8] != exp.NONE).sum())


def test_s3_and_s4_cannot_use_label_correlated_coverage() -> None:
    inst = exp.generate(exp.Cell(prevalence=0.05, n=1200), seed=1)
    window = 100
    leaky = np.full(inst.n, exp.NONE, dtype=np.int8)
    leaky[inst.y] = exp.MATCH  # cobertura = rótulo
    top = exp.ruler_order(inst.nota)[:window]
    leaky[top] = exp.NONMATCH
    safe = exp.restrict_to_window(inst.nota, leaky, window)
    outside = np.ones(inst.n, dtype=bool)
    outside[top] = False
    assert (safe[outside] == exp.NONE).all()
    spec = exp.build_spec(replace(inst, llm_seen=safe), llm_attribute=True)
    assert spec.feature_names[-1] == "llm:NONMATCH"  # só sobrou NONMATCH, na janela
    assert not spec.X[outside, -1].any()
    # S4: nenhum rótulo extra fora da janela
    extra = np.flatnonzero((safe == exp.MATCH) | (safe == exp.NONMATCH))
    assert set(extra.tolist()) <= set(top.tolist())


@pytest.mark.skipif(not REAL.exists(), reason="extrato real ausente")
def test_real_fixture_is_restricted_to_the_ruler_window() -> None:
    inst = exp.load_real_instance(REAL, FIXTURE, RATES, seed=0)
    covered = inst.llm_seen != exp.NONE
    window = np.zeros(inst.n, dtype=bool)
    window[exp.ruler_order(inst.nota)[: exp.REAL_LLM_WINDOW]] = True
    assert not (covered & ~window).any()
    # o fixture cobre os positivos de fora da janela, mas eles não entram
    assert int((inst.y & ~window & covered).sum()) == 0


def test_ruler_source_is_reported_and_stub_is_not_silent(
    inst: exp.Instance, caplog: pytest.LogCaptureFixture
) -> None:
    few = replace(inst, nota=np.clip(inst.nota, 0.0, 8.9))  # sem âncoras A+
    with caplog.at_level("WARNING"):
        offset, source = exp.ruler_offset_info(few.nota)
    assert source == "stub"
    assert any("Insufficient anchors" in r.getMessage() for r in caplog.records)
    assert (np.diff(offset[np.argsort(few.nota)]) >= 0).all()
    assert exp.build_spec(few).ruler_source == "stub"
    out = exp.run_instance(few, seed=1, budgets=(50,), with_s1=False)
    s2 = [r for r in out["runs"] if r["strategy"] == "S2"]
    assert s2 and all(r["ruler_source"] == "stub" for r in s2)
    full, src_full = exp.ruler_offset_info(few.nota, anchor_nota=inst.nota)
    assert src_full in {"anchor_platt_full_extract", "stub"}
    assert full.shape == few.nota.shape


def test_estimation_uses_only_the_human_decision_not_the_truth() -> None:
    base = exp.generate(exp.Cell(prevalence=0.05, n=600), seed=3)
    # humano diz MATCH para tudo; a verdade não tem nenhum par
    inst = replace(
        base,
        y=np.zeros(base.n, dtype=bool),
        hard=np.zeros(base.n, dtype=bool),
        human=np.ones(base.n, dtype=bool),
    )
    spec = exp.build_spec(inst)
    path = exp.adaptive_path(spec, inst.human, steps=50)
    row = exp.strategy_at_budget(inst, path, spec, 50, seed=1)
    # HT sobre a decisão humana: todo item amostrado é MATCH => total = tamanho
    # do pool; restante = pool - amostra (antes da correção dava 0, pois usava y)
    assert row["est"] == pytest.approx(inst.n - 50)
    assert row["tp"] == 0  # nenhum verdadeiro verificado


def test_stop_rule_sees_human_decisions_not_truth() -> None:
    n = 400
    human = np.zeros(n, dtype=bool)
    human[:10] = True  # 10 falsos positivos do humano nas 10 primeiras decisões
    inst = exp.Instance(
        df=pd.DataFrame({"x": np.zeros(n)}),
        y=np.zeros(n, dtype=bool),  # nenhum par verdadeiro
        hard=np.zeros(n, dtype=bool),
        nota=np.linspace(10, 0, n),
        llm_all=np.zeros(n, dtype=np.int8),
        llm_seen=np.zeros(n, dtype=np.int8),
        human=human,
        pair_ids=[str(i) for i in range(n)],
    )
    path = exp.PathResult(np.arange(n), np.full(n, 0.9), {0: np.zeros(1)})
    rows = {(r["cost"], r["rule"]): r for r in exp.regret_rows(inst, path, (0.05,))}
    # a regra vê 10 MATCH iniciais: para em 109 (com y teria parado em 58)
    assert rows[(0.05, "S6")]["k_stop"] == 109
    assert rows[(0.05, "S6")]["found"] == 0  # o valor obtido continua verdadeiro


# ---------------------------------------------------------------------------
# Replicação por passo de blocagem (seção 8)
# ---------------------------------------------------------------------------


def _passo_instance(n: int = 1800) -> exp.Instance:
    inst = exp.generate(exp.Cell(prevalence=0.05, n=n, signal=True), seed=4)
    df = inst.df.assign(PASSO=np.tile([6, 9, 11], n // 3))
    return replace(inst, df=df)


def test_folds_partition_pairs_and_positives_by_passo() -> None:
    inst = _passo_instance()
    passo = inst.df["PASSO"].to_numpy()
    spec = exp.build_spec(inst)
    pairs, positives = 0, 0
    for p in exp.PASSOS:
        rows, _ = exp.replicate_fold(inst, p, spec, (100,))
        assert rows[0]["pairs"] == int((passo == p).sum())
        assert rows[0]["positives"] == int((inst.y & (passo == p)).sum())
        pairs += rows[0]["pairs"]
        positives += rows[0]["positives"]
    assert pairs == inst.n and positives == int(inst.y.sum())


def test_transfer_uses_no_test_label() -> None:
    inst = _passo_instance()
    passo = inst.df["PASSO"].to_numpy()
    spec = exp.build_spec(inst)
    test = np.flatnonzero(passo == 6)
    train = np.flatnonzero(passo != 6)
    params = exp.fit_transfer_model(exp.subspec(spec, train), inst.human[train], 200)
    before = exp.order_with_model(exp.subspec(spec, test), params)
    # inverte todo rótulo do teste: o modelo e a ordem do teste não podem mudar
    flipped = inst.human.copy()
    flipped[test] = ~flipped[test]
    params2 = exp.fit_transfer_model(exp.subspec(spec, train), flipped[train], 200)
    after = exp.order_with_model(exp.subspec(spec, test), params2)
    assert np.array_equal(params, params2)
    assert np.array_equal(before, after)
    # as linhas de treino e de teste não se misturam
    assert not set(train.tolist()) & set(test.tolist())
    rows_a, _ = exp.replicate_fold(inst, 6, spec, (100,))
    rows_b, _ = exp.replicate_fold(replace(inst, human=flipped), 6, spec, (100,))
    assert rows_a[0]["S0"] != rows_b[0]["S0"] or rows_a[0]["T"] != rows_b[0]["T"]


def test_signals_use_documented_features() -> None:
    inst = _passo_instance()
    spec = exp.build_spec(inst)
    params = np.zeros(2 + spec.X.shape[1])
    names = list(spec.feature_names)
    params[2 + names.index("nome:L3")] = 0.7
    dt = [i for i, n in enumerate(names) if n.startswith("dtnasc:")]
    for i in dt:
        params[2 + i] = -0.3
    sig = exp.r4_signals(spec, params)
    assert sig["nome_L3"] == pytest.approx(0.7)
    assert sig["dtnasc_zero"] == pytest.approx(0.3)  # zero pesa mais que os demais
    assert sig["mae_missing"] == 0.0 or math.isnan(sig["mae_missing"])


def _found(rows: list[tuple]) -> pd.DataFrame:
    cols = ["fold", "ruler", "B", "positives", "S0", "T", "O"]
    return pd.DataFrame(rows, columns=cols)


def _sig(rows: list[tuple]) -> pd.DataFrame:
    cols = ["fold", "ruler", "positives", "mae_missing", "nome_L3", "dtnasc_zero"]
    return pd.DataFrame(rows, columns=cols)


def test_r1_r2_r3_hand_examples() -> None:
    def fold(f: int, pos: int, c: tuple, p: tuple) -> list[tuple]:
        # c, p = (S0, T, O) em B = 100, 200, 400 para as réguas C e P
        out = []
        for ruler, vals in (("C", c), ("P", p)):
            for b, (s0, t, o) in zip((100, 200, 400), vals, strict=True):
                out.append((f, ruler, b, pos, s0, t, o))
        return out

    # R1 confirmada (+3 em 400, >= 0 em todo B), R2 confirmada, R3 confirmada
    good = fold(
        6, 10, ((2, 2, 3), (3, 4, 4), (5, 8, 6)), ((2, 3, 3), (3, 6, 4), (5, 9, 6))
    )
    good += fold(
        9, 10, ((1, 1, 2), (2, 2, 3), (3, 4, 5)), ((1, 1, 2), (2, 3, 3), (3, 5, 5))
    )
    res = {r["id"]: r for r in exp.evaluate_replication(_found(good), _sig([]))}
    assert res["R1"]["status"] == "confirmada"  # soma em 400: (8-5)+(4-3) = +4
    assert res["R2"]["status"] == "confirmada"  # O-S0 = +1 e +2 em 400
    assert res["R3"]["status"] == "confirmada"  # P: 9+5 > C: 8+4 em 400; 6+3 > 4+2
    # R1 refutada: T < S0 em B = 400; R2 refutada: O < S0 numa dobra
    bad = fold(
        6, 10, ((2, 2, 3), (3, 2, 4), (5, 3, 4)), ((2, 2, 3), (3, 2, 4), (5, 3, 4))
    )
    res = {r["id"]: r for r in exp.evaluate_replication(_found(bad), _sig([]))}
    assert res["R1"]["status"] == "refutada"
    assert res["R2"]["status"] == "refutada"
    assert res["R3"]["status"] == "refutada"  # P = C em B = 200 e 400 (<=)
    # R1 inconclusiva: +1 em B = 400 (entre 0 e 3)
    mid = fold(
        6, 10, ((2, 2, 3), (3, 3, 4), (5, 6, 6)), ((2, 2, 3), (3, 3, 4), (5, 6, 6))
    )
    res = {r["id"]: r for r in exp.evaluate_replication(_found(mid), _sig([]))}
    assert res["R1"]["status"] == "inconclusiva"
    # dobra com menos de 5 positivos fica fora dos critérios
    tiny = fold(
        11, 4, ((0, 0, 0), (0, 0, 0), (9, 0, 0)), ((0, 0, 0), (0, 0, 0), (9, 0, 0))
    )
    res = {r["id"]: r for r in exp.evaluate_replication(_found(good + tiny), _sig([]))}
    assert res["R1"]["status"] == "confirmada"


def test_r4_hand_examples() -> None:
    found = _found([])
    yes = _sig(
        [(f, r, 10, 0.3, 0.2, -0.1) for f in (6, 9, 11) for r in ("P", "C")]
    )  # 2 de 3 a favor em todas
    no = _sig([(f, r, 10, -0.3, -0.2, 0.1) for f in (6, 9, 11) for r in ("P", "C")])
    mixed = _sig(
        [
            (6, "P", 10, 0.3, 0.2, 0.1),
            (9, "P", 10, -1.0, -1.0, 0.1),
            (11, "P", 10, 0.3, 0.2, 0.1),
        ]
        + [(f, "C", 10, 0.3, 0.2, 0.1) for f in (6, 9, 11)]
    )

    def get(s: pd.DataFrame) -> str:
        res = {r["id"]: r for r in exp.evaluate_replication(found, s)}
        return str(res["R4"]["status"])

    assert get(yes) == "confirmada"
    assert get(no) == "refutada"
    assert get(mixed).startswith("inconclusiva")
    # uma dobra não informativa (3 positivos) sai: sobra só 6 e 9, ambas a favor
    small = yes.copy()
    small.loc[small["fold"] == 11, "positives"] = 3
    assert get(small) == "confirmada"


def test_r4_reading_of_reference_level_is_flagged_as_ambiguous() -> None:
    rows = [(f, r, 10, 0.3, 0.2, -0.1) for f in (6, 9, 11) for r in ("P", "C")]
    sig = _sig(rows).assign(mae_missing_literal=math.nan)  # missing é a referência
    res = {r["id"]: r for r in exp.evaluate_replication(_found([]), sig)}
    # invariante: 2 de 3 a favor (confirmada); literal: só nome L3 a favor (1 de 3)
    assert res["R4"]["status"].startswith("inconclusiva — critério mal")
    both = sig.assign(mae_missing_literal=0.3)
    res = {r["id"]: r for r in exp.evaluate_replication(_found([]), both)}
    assert res["R4"]["status"] == "confirmada"


def test_r4_signals_missing_as_reference_uses_invariant_reading() -> None:
    inst = _passo_instance()
    spec = exp.build_spec(inst)
    names = list(spec.feature_names)
    params = np.zeros(2 + spec.X.shape[1])
    for i, n in enumerate(names):
        if n.startswith("mae:") and n != "mae:missing":
            params[2 + i] = -0.4
    if "mae:missing" in names:  # o sintético tem L0: o literal existe
        assert exp.r4_signals(spec, params)["mae_missing_literal"] == 0.0
    keep = [i for i, n in enumerate(names) if n != "mae:missing"]
    spec_ref = replace(spec, feature_names=tuple(names[i] for i in keep))
    p2 = np.concatenate([params[:2], params[2:][keep]])
    sig = exp.r4_signals(spec_ref, p2)
    assert math.isnan(sig["mae_missing_literal"])
    assert sig["mae_missing"] == pytest.approx(0.4)
