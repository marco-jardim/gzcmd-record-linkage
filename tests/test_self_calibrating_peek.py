"""Testes de ``SelfCalibratingSession.peek_upcoming`` (só leitura)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import pytest

from gzcmd_record_linkage.llm_dossier import make_pair_id
from gzcmd_record_linkage.self_calibrating_queue import (
    EXPLORE_ARM,
    OracleReviewer,
    SelfCalibratingConfig,
    SelfCalibratingSession,
)

N = 900


def make_data(n: int = N, seed: int = 3) -> tuple[pd.DataFrame, dict[str, bool]]:
    rng = np.random.default_rng(seed)
    truth = rng.random(n) < 0.06

    def pick(m: list[float], u: list[float]) -> np.ndarray:
        return np.where(truth, rng.choice(m, n), rng.choice(u, n))

    nome = pick([1.0, 0.75, 0.4], [0.0, 0.3, 0.7, 1.0])
    mae = pick([1.0, 0.6, 0.0], [0.0, 0.4, 0.8])
    dtnasc = pick([1.0, 0.5, 0.0], [0.0, 0.5, 1.0])
    endereco = pick([1.0, 0.7, 0.2], [0.0, 0.3, 0.7])
    nota = np.clip(
        np.where(truth, rng.normal(8.8, 0.8, n), rng.normal(2.5, 1.8, n)), 0, 10
    )
    df = pd.DataFrame(
        {
            "COMPREC": np.arange(n),
            "REFREC": np.arange(n) + 100_000,
            "PASSO": rng.choice([6, 9, 11], size=n),
            "SCORE": np.round(nota * 0.95, 4),
            "nota final": nota,
            "nome_score_total": nome,
            "mae_score_total": mae,
            "mae_missing": (rng.random(n) < 0.05).astype(int),
            "dtnasc_score_total": dtnasc,
            "endereco_score_total": endereco,
            "municipio_score": pick([1.0, 0.0], [0.0, 1.0]),
            "dtnasc_all_zero": (dtnasc == 0).astype(int),
            "endereco_zero": (endereco == 0).astype(int),
        }
    )
    ids = [
        make_pair_id(c, r, p)
        for c, r, p in zip(df["COMPREC"], df["REFREC"], df["PASSO"], strict=True)
    ]
    return df, dict(zip(ids, (bool(t) for t in truth), strict=True))


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, dict[str, bool]]:
    return make_data()


def cfg(**kw: Any) -> SelfCalibratingConfig:
    base: dict[str, Any] = {
        "max_round_reviews": 60,
        "max_reviews": 150,
        "max_rounds": 3,
        "min_new_positives": 0,
        "explore_every": 5,
    }
    base.update(kw)
    return SelfCalibratingConfig(**base)


def human(truth: Mapping[str, bool], items: list[dict]) -> dict[str, str]:
    return {
        it["pair_id"]: "MATCH" if truth[it["pair_id"]] else "NONMATCH" for it in items
    }


def new_session(data, **kw: Any) -> SelfCalibratingSession:
    df, truth = data
    return SelfCalibratingSession(df, OracleReviewer(truth, seed=1), cfg(**kw))


def advance(session: SelfCalibratingSession, truth: Mapping[str, bool], steps: int):
    for _ in range(steps):
        if session.finished:
            break
        if session.awaiting_human:
            session.submit_human_decisions(human(truth, session.gate_items))
        else:
            session.step()


def test_state_is_unchanged_by_peek(data):
    _, truth = data
    session = new_session(data)
    for steps in (1, 3, 40, 80):
        advance(session, truth, steps)
        if session.phase != "reviewing":
            continue
        before = json.dumps(session.to_state(), sort_keys=True)
        first = session.peek_upcoming(8)
        second = session.peek_upcoming(8)
        assert first == second
        assert json.dumps(session.to_state(), sort_keys=True) == before
    assert first  # houve pelo menos um braço ativo


def test_peek_does_not_change_the_sequence(data):
    df, truth = data
    plain = new_session(data)
    peeked = new_session(data)
    events_a, events_b = [], []
    while not plain.finished:
        for s, out in ((plain, events_a), (peeked, events_b)):
            if s.awaiting_human:
                s.submit_human_decisions(human(truth, s.gate_items))
            else:
                if s is peeked:
                    s.peek_upcoming(5)
                out.append(s.step().to_dict())
    assert events_a == events_b


def test_next_presented_pair_is_first_of_its_arm(data):
    _, truth = data
    session = new_session(data)
    checked = {"bandit": 0, EXPLORE_ARM: 0}
    while not session.finished and sum(checked.values()) < 60:
        if session.awaiting_human:
            session.submit_human_decisions(human(truth, session.gate_items))
            continue
        peek = session.peek_upcoming(8)
        event = session.step()
        if event.kind == "pair_presented":
            label = event.payload["band_label"]
            pid = event.payload["pair_id"]
            assert label in peek, label
            assert peek[label][0] == pid
            assert pid in peek[label]
            checked["explore" if label == EXPLORE_ARM else "bandit"] += 1
    assert checked["bandit"] > 10 and checked[EXPLORE_ARM] >= 3


def test_peek_skips_reviewed_pending_and_respects_limits(data):
    _, truth = data
    session = new_session(data)
    advance(session, truth, 60)
    assert session.phase == "reviewing"
    reviewed = set(session.results()["pair_id"])
    peek = session.peek_upcoming(4)
    all_ids = [pid for ids in peek.values() for pid in ids]
    assert all(1 <= len(ids) <= 4 for ids in peek.values())
    assert not reviewed & set(all_ids)
    for ids in peek.values():
        assert len(ids) == len(set(ids))
    # entre a apresentação e a revisão, o par apresentado é "pendente" e não aparece
    while not any(e.kind == "pair_presented" for e in [session.step()]):
        pass
    presented = session.events[-1].payload["pair_id"]
    assert presented not in {
        p for ids in session.peek_upcoming(50).values() for p in ids
    }


def test_peek_after_resume_is_identical(data):
    df, truth = data
    session = new_session(data)
    advance(session, truth, 70)
    assert session.phase == "reviewing"
    peek = session.peek_upcoming(8)
    restored = SelfCalibratingSession.from_state(
        json.loads(json.dumps(session.to_state())), df, OracleReviewer(truth, seed=1)
    )
    assert restored.peek_upcoming(8) == peek


def test_peek_outside_reviewing_and_validation(data):
    _, truth = data
    session = new_session(data)
    assert session.peek_upcoming(3) == {}  # antes da primeira rodada
    # valores inválidos DE PROPÓSITO: a validação ocorre em tempo de execução
    invalid_values: tuple[Any, ...] = (0, -1, 1.5, True, "3")
    for bad in invalid_values:
        with pytest.raises(ValueError):
            session.peek_upcoming(bad)
    while not session.awaiting_human:
        session.step()
    assert session.peek_upcoming(3) == {}  # no portão


def test_no_explore_key_without_exploration(data):
    _, truth = data
    session = new_session(data, explore_every=0)
    advance(session, truth, 30)
    assert EXPLORE_ARM not in session.peek_upcoming(5)
    with_explore = new_session(data)
    advance(with_explore, truth, 30)
    assert EXPLORE_ARM in with_explore.peek_upcoming(5)
