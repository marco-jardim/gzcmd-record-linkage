"""Testes rápidos e offline de ``notebooks/exp_filas_real.py`` (população à mão)."""

from __future__ import annotations

import exp_filas_real as efr
import numpy as np
import pandas as pd
import pytest


def test_tie_broken_order_descending_and_deterministic() -> None:
    nota = np.array([1.0, 3.0, 3.0, 2.0, 5.0])
    a = efr.tie_broken_order(nota, 7)
    b = efr.tie_broken_order(nota, 7)
    assert a.tolist() == b.tolist()
    assert nota[a].tolist() == [5.0, 3.0, 3.0, 2.0, 1.0]
    assert a[0] == 4 and a[-1] == 0


def test_band_bounds_cumulative_fractions() -> None:
    assert efr.band_bounds(1000) == [(0, 10), (10, 50), (50, 200), (200, 1000)]
    with pytest.raises(ValueError):
        efr.band_bounds(3)


def test_yield_by_bins_and_verified_at() -> None:
    truth = np.zeros(100, dtype=bool)
    truth[[0, 1, 5, 30, 99]] = True
    assert efr.yield_by_bins(truth, [0, 10, 50, 100]) == [(3, 10), (1, 40), (1, 50)]
    assert efr.verified_at(truth, [1, 2, 10, 100]) == {1: 1, 2: 2, 10: 3, 100: 5}


def test_global_order_vs_bandit_on_monotonic_yield() -> None:
    """Com rendimento monotônico, a ordem global não perde para o bandit."""
    n = 1000
    truth = np.zeros(n, dtype=bool)
    truth[:8] = True  # todos os verdadeiros no topo (faixa 0–1%)
    bounds = efr.band_bounds(n)
    budget = 50
    s0 = efr.verified_at(truth, [budget])[budget]
    gains = []
    for seed in range(10):
        items, arms = efr.bandit_pick(n, bounds, lambda r: int(truth[r]), budget, seed)
        assert len(items) == budget and len(set(items.tolist())) == budget
        assert set(arms.tolist()) <= {0, 1, 2, 3}
        gains.append(int(truth[items].sum()) - s0)
    assert max(gains) <= 0
    assert s0 == 8


def test_bandit_items_stay_in_their_band_top_down() -> None:
    n = 400
    bounds = efr.band_bounds(n)
    items, arms = efr.bandit_pick(n, bounds, lambda r: 0, 60, 3)
    for k, (a, b) in enumerate(bounds):
        mine = items[arms == k]
        assert ((mine >= a) & (mine < b)).all()
        assert mine.tolist() == sorted(mine.tolist())  # topo para baixo
        assert mine.tolist() == list(range(a, a + len(mine)))


def test_induce_nonmonotonic_moves_trues_into_destination() -> None:
    n = 200
    truth = np.zeros(n, dtype=bool)
    truth[[0, 1, 2, 7]] = True  # todos na faixa 0–5% (postos < 10)
    rng = np.random.default_rng(0)
    perm = efr.induce_nonmonotonic(truth, 10, 2, 10, rng)
    assert sorted(perm.tolist()) == list(range(n))  # é uma permutação
    moved = truth[perm]
    assert moved.sum() == 4
    assert not moved[:2].any()  # faixa 0–2 ficou sem verdadeiros
    assert moved[2:10].sum() == 4  # todos no destino
    assert perm[10:].tolist() == list(range(10, n))  # fora da janela não muda
    with pytest.raises(ValueError):
        efr.induce_nonmonotonic(truth, 10, 8, 10, rng)  # destino pequeno demais


def test_induced_nonmonotonicity_lets_bandit_beat_global_order() -> None:
    n = 10_000
    truth = np.zeros(n, dtype=bool)
    # faixa 0–1% (100 postos) sem verdadeiros; faixa 1–5% (400 postos) com 200
    truth[100:500:2] = True
    bounds = efr.band_bounds(n)
    budget = 60
    s0 = efr.verified_at(truth, [budget])[budget]
    assert s0 == 0
    s5 = []
    for seed in range(20):
        items, _ = efr.bandit_pick(n, bounds, lambda r: int(truth[r]), budget, seed)
        s5.append(int(truth[items].sum()))
    assert np.mean(s5) > s0


def test_human_error_is_deterministic_and_nested() -> None:
    assert efr.human_uniform(1, "x") == efr.human_uniform(1, "x")
    assert efr.human_uniform(1, "x") != efr.human_uniform(2, "x")
    u = np.array([efr.human_uniform(3, f"p{i}") for i in range(2000)])
    assert ((u >= 0) & (u < 1)).all()
    truth = np.array([True] * 1000 + [False] * 1000)
    h5 = efr.human_match(truth, u, 0.05, 0.0)
    h10 = efr.human_match(truth, u, 0.10, 0.01)
    assert not h5[1000:].any()  # FP = 0
    assert 0.02 < 1 - h5[:1000].mean() < 0.09  # FN ~ 5%
    assert (h10[:1000] <= h5[:1000]).all()  # erros aninhados: FN 10% ⊇ FN 5%
    assert h10[1000:].sum() == (u[1000:] < 0.01).sum()
    assert (efr.human_match(truth, u, 0.05, 0.0) == h5).all()


def test_audit_reviews_only_discordant_pairs() -> None:
    truth = np.array([1, 1, 1, 0, 0, 0, 0, 1], dtype=bool)
    human = np.array([1, 0, 0, 0, 0, 0, 1, 1], dtype=bool)  # FN: posições 1 e 2
    llm = np.array([1, 1, 0, 1, 0, 0, 1, 0], dtype=bool)
    m = efr.audit_metrics(truth, human, llm)
    # extras = LLM MATCH e humano NONMATCH: posições 1 e 3
    assert m["extras"] == 2
    assert m["recovered"] == 1  # posição 1 era par
    assert m["extras_llm_fp"] == 1  # posição 3 era não par
    assert m["human_fn"] == 2
    assert m["human_fp"] == 1
    assert m["human_found"] == 2
    # aleatória: 2 extras entre 5 NONMATCH do humano, 2 deles FN → 0,8
    assert m["random_recovered"] == pytest.approx(2 * 2 / 5)


def test_audit_without_discordance_has_no_extras() -> None:
    t = np.array([1, 0, 1], dtype=bool)
    m = efr.audit_metrics(t, t, t)
    assert m["extras"] == 0 and m["recovered"] == 0 and m["human_fn"] == 0


def test_summarize_and_ratio_boot() -> None:
    mean, lo, hi, n = efr.summarize([1.0, 2.0, 3.0, float("nan")])
    assert (mean, n) == (2.0, 3)
    assert lo < 2.0 < hi
    assert efr.summarize([4.0, 4.0])[1:3] == (4.0, 4.0)
    point, rlo, rhi = efr.ratio_boot([1, 0, 2, 1], [2, 0, 2, 2], n_boot=200)
    assert point == pytest.approx(4 / 6)
    assert rlo <= point <= rhi
    assert np.isnan(efr.ratio_boot([0, 0], [0, 0])[0])


def _diff(rows: list[tuple[str, str, int, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["scenario", "reward", "budget", "lo"])


def test_decide_h8_criteria() -> None:
    real_ok = [("real", "human", 50, -1.0), ("real", "llm", 50, 0.0)]
    ind_win = [("N1", "human", 50, 0.2)]
    ind_lose = [("N1", "human", 50, -0.1), ("N2", "llm", 50, 0.5)]
    assert efr.decide_h8(_diff(real_ok + ind_win), True)[0] == "confirmada"
    assert efr.decide_h8(_diff(real_ok + ind_lose), True)[0] == "inconclusiva"
    real_bad = [("real", "human", 50, 0.3)]
    assert efr.decide_h8(_diff(real_bad + ind_win), True)[0] == "refutada"
    assert efr.decide_h8(_diff(real_ok + ind_win), False)[0] == "inconclusiva"


def test_decide_h11_criteria() -> None:
    def cells(pairs: list[tuple[float, float]]) -> pd.DataFrame:
        return pd.DataFrame(pairs, columns=["recovery", "precision"])

    assert efr.decide_h11(cells([(0.6, 0.4), (0.5, 0.3)])) == ("confirmada", 2, 2)
    assert efr.decide_h11(cells([(0.6, 0.2), (0.4, 0.9)])) == ("refutada", 0, 2)
    assert efr.decide_h11(cells([(0.6, 0.4), (0.4, 0.9)])) == ("inconclusiva", 1, 2)
    assert efr.decide_h11(cells([(float("nan"), 0.9)]))[0] == "refutada"


def test_run_h8_and_h11_seed_on_tiny_population() -> None:
    class FakeLlm:
        def match(self, pid: str) -> bool:
            return pid.endswith("0")

        def covered(self, pid: str) -> bool:
            return True

    n = 400
    nota = np.linspace(10, 0, n)
    truth = np.zeros(n, dtype=bool)
    truth[[0, 1, 3, 20]] = True
    pids = np.array([f"p{i}" for i in range(n)], dtype=object)
    pool = efr.PoolData(pids, nota, truth)
    llm = FakeLlm()
    rows = efr.run_h8_seed(pool, llm, 1, budgets=(10, 40), scenarios=("real", "N1"))
    df = pd.DataFrame(rows)
    real = df[(df["scenario"] == "real") & (df["metric"] == "s0")]
    assert set(real["value"]) == {3.0, 4.0}  # B=10 e B=40 na ordem da nota
    assert {"s5", "diff", "arm1_share", "fixture_coverage"} <= set(df["metric"])
    h11 = pd.DataFrame(
        efr.run_h11_seed(pool, llm, 1, budgets=(10, 40), fns=(0.0,), fps=(0.0,))
    )
    assert len(h11) == 2
    assert (h11["human_fn"] == 0).all()  # humano perfeito: nada a recuperar
    assert (h11["recovered"] == 0).all()
