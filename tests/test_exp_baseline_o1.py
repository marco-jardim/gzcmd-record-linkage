"""Testes offline do script de comparação com a ordenação O1 (escore do OpenRecLink)."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import exp_baseline_o1 as o1
import numpy as np
import pandas as pd
import pytest

REAL = Path(r"D:\git\phd-research\data\COMPARADORSEMIDENT.csv")
FIXTURE = Path("data/llm_fixtures/haiku-5-5_v4_sem-nota.jsonl")


def test_resolve_score_legacy_header_and_decimal_comma() -> None:
    raw = pd.DataFrame(
        {"SCORE,C,15,0": [" 7,02", "-5,648", "15,04412094565"], "x": [1, 2, 3]}
    )
    out = o1.resolve_score(raw)
    assert out.tolist() == pytest.approx([7.02, -5.648, 15.04412094565])


def test_resolve_score_plain_numeric_and_errors() -> None:
    assert o1.resolve_score(pd.DataFrame({"SCORE": [1.5, 2.5]})).tolist() == [1.5, 2.5]
    # com as duas colunas vale a canônica
    both = pd.DataFrame({"SCORE": [1.0, 2.0], "SCORE,C,15,0": [9.0, 9.0]})
    assert o1.resolve_score(both).tolist() == [1.0, 2.0]
    with pytest.raises(KeyError):
        o1.resolve_score(pd.DataFrame({"nota final": [1.0]}))
    with pytest.raises(ValueError):
        o1.resolve_score(pd.DataFrame({"SCORE": ["abc", "1,0"]}))


def test_o1_order_is_descending_and_deterministic() -> None:
    score = np.array([1.0, 3.0, 3.0, 2.0, 3.0, 1.0, 0.0])
    a = o1.rank_order(score)
    assert (np.diff(score[a]) <= 0).all()
    assert sorted(a.tolist()) == list(range(len(score)))
    # arquivo: empates pela posição no arquivo
    assert a.tolist() == [1, 2, 4, 3, 0, 5, 6]
    # aleatório: mesma semente, mesma ordem; outra semente muda só os empates
    b1, b2, c = (
        o1.rank_order(score, 7),
        o1.rank_order(score, 7),
        o1.rank_order(score, 8),
    )
    assert (b1 == b2).all()
    assert (np.diff(score[c]) <= 0).all()
    many = {tuple(o1.rank_order(score, s)[:3].tolist()) for s in range(30)}
    assert len(many) > 1  # os três 3,0 aparecem em ordens diferentes
    assert all(set(t) == {1, 2, 4} for t in many)


def test_window_coverage_counts_hand_example() -> None:
    order = np.array([4, 2, 0, 1, 3, 5])
    codes = np.array(
        [o1.NONE, o1.MATCH, o1.NONMATCH, o1.NONE, o1.UNSURE, o1.NONE], dtype=np.int8
    )
    y = np.array([False, True, False, True, True, False])
    cov = o1.window_coverage(order, codes, y, window=4)  # pares 4, 2, 0, 1
    assert cov["window"] == 4
    assert cov["covered"] == 3  # 4 (UNSURE), 2 (NONMATCH), 1 não: ordem 4,2,0,1
    # pares 4, 2, 0, 1: veredito UNSURE, NONMATCH, NONE, MATCH
    assert cov["covered"] == 3 and cov["covered_fraction"] == pytest.approx(0.75)
    assert cov["positives"] == 2 and cov["positives_covered"] == 2  # 4 e 1


def test_llm_window_order_priorities_and_permutation() -> None:
    order = np.array([5, 4, 3, 2, 1, 0])
    codes = np.array(
        [o1.MATCH, o1.NONE, o1.UNSURE, o1.NONMATCH, o1.MATCH, o1.UNSURE], dtype=np.int8
    )
    out = o1.llm_window_order(order, codes, window=4)  # janela: 5, 4, 3, 2
    # MATCH (4), UNSURE (5, 2), resto (3: NONMATCH); depois fora da janela (1, 0)
    assert out.tolist() == [4, 5, 2, 3, 1, 0]
    assert sorted(out.tolist()) == list(range(6))
    # par sem veredito fica no resto, na ordem de base (não priorizado)
    none_codes = np.full(6, o1.NONE, dtype=np.int8)
    assert o1.llm_window_order(order, none_codes, 4).tolist() == order.tolist()


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture local ausente")
def test_fixture_codes_count_only_covered_pairs() -> None:
    ids = [
        json.loads(line)["pair_id"]
        for line in FIXTURE.read_text(encoding="utf-8").splitlines()[:25]
    ]
    probe = [*ids, "par-inexistente-1", "par-inexistente-2"]
    codes = o1.fixture_codes(probe, FIXTURE)
    assert int((codes != o1.NONE).sum()) == len(set(ids))
    assert (codes[-2:] == o1.NONE).all()
    assert set(codes[codes != o1.NONE].tolist()) <= {o1.MATCH, o1.NONMATCH, o1.UNSURE}


def test_llm_comparison_uses_no_synthetic_rates() -> None:
    assert set(inspect.signature(o1.fixture_codes).parameters) == {
        "pair_ids",
        "fixture",
    }
    assert set(inspect.signature(o1.llm_window_order).parameters) == {
        "order",
        "codes",
        "window",
    }
    src = inspect.getsource(o1.run_seed) + inspect.getsource(o1.fixture_codes)
    for banned in (
        "simulate_llm",
        "BandedOracle",
        "rates",
        "OracleReviewer",
        "fp_rate",
    ):
        assert banned not in src
    # fixture_codes usa o modo padrão (on_missing="error"), sem fallback por taxas
    assert 'on_missing="rates"' not in inspect.getsource(o1.fixture_codes)


@pytest.mark.skipif(not REAL.exists(), reason="extrato real ausente")
def test_o1_and_o2_reproduce_thesis_at_96_reviews() -> None:
    pool = o1.load_pool(REAL, "below")
    assert pool.n == 61485 and int(pool.y.sum()) == 36
    assert o1.found_at(pool.y, o1.rank_order(pool.score), 96) == o1.THESIS_O1_96
    assert o1.found_at(pool.y, o1.rank_order(pool.nota), 96) == o1.THESIS_O2_96
    # com desempate aleatório, a O1 em 96 revisões fica em 19 a 20 em todas as sementes
    vals = {o1.found_at(pool.y, o1.rank_order(pool.score, s), 96) for s in range(40)}
    assert vals <= {19, 20}
