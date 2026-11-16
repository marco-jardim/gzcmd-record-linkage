from __future__ import annotations

import json
import re
import warnings
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from scipy.stats import beta

from gzcmd_record_linkage import self_calibrating_queue as scq
from gzcmd_record_linkage.arranjo_corte import THESIS_CUTOFFS, below_cutoff
from gzcmd_record_linkage.calibration import _InsufficientAnchors, predict_stub
from gzcmd_record_linkage.fellegi_sunter import (
    CDR_SUBSCORE_COLUMNS,
    FellegiSunterError,
    design_matrix,
    discretize_fields,
    fit_fellegi_sunter,
)
from gzcmd_record_linkage.field_recalibrator import RecalibrationPrior
from gzcmd_record_linkage.llm_dossier import (
    _NAME_COLS,
    DossierView,
    build_dossier,
    make_pair_id,
)
from gzcmd_record_linkage.llm_review import LLMCallError
from gzcmd_record_linkage.self_calibrating_queue import (
    ExperimentalWarning,
    OracleReviewer,
    SelfCalibratingConfig,
    SelfCalibratingSession,
    SessionEvent,
    run_self_calibrating,
    stop_probability,
    zero_match_reviews_needed,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore::gzcmd_record_linkage.self_calibrating_queue.ExperimentalWarning"
)

N_PAIRS = 2400


# --------------------------------------------------------------------------
# Synthetic data
# --------------------------------------------------------------------------


def _pick(rng: np.random.Generator, values: list[float], probs: list[float], n: int):
    return rng.choice(values, size=n, p=probs)


def cdr_columns(truth: np.ndarray, seed: int = 11) -> dict[str, np.ndarray]:
    """Synthetic CdR sub-scores (own RNG, so the composites stay untouched)."""
    rng = np.random.default_rng(seed)
    n = truth.shape[0]
    out: dict[str, np.ndarray] = {}
    for name in CDR_SUBSCORE_COLUMNS:
        if name == "DTNASC dt inv ano":
            out[name] = np.zeros(n)  # constant column: only L0 occurs
        elif name.endswith("comuns"):  # penalties: only negative values
            values = rng.choice([-0.0, -0.25, -0.5, -0.75, -1.0], size=n)
            out[name] = np.where(truth, values * 0.2, values)
        elif name.endswith(("iguais", "raros", "muito parec", "abrev", "prox", "jacc")):
            up = rng.choice(
                [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], size=n, p=[0.1] + [0.18] * 5
            )
            down = rng.choice(
                [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                size=n,
                p=[0.6, 0.2, 0.1, 0.05, 0.03, 0.02],
            )
            out[name] = np.where(truth, up, down)
        else:  # binary agreement flags
            out[name] = np.where(
                truth, rng.random(n) < 0.9, rng.random(n) < 0.15
            ).astype(float)
    return out


def make_data(n: int = N_PAIRS, seed: int = 7) -> tuple[pd.DataFrame, dict[str, bool]]:
    rng = np.random.default_rng(seed)
    truth = rng.random(n) < 0.05
    pos = int(truth.sum())
    neg = n - pos

    def field(
        m_values: list[float],
        m_probs: list[float],
        u_values: list[float],
        u_probs: list[float],
    ) -> np.ndarray:
        out = np.empty(n)
        out[truth] = _pick(rng, m_values, m_probs, pos)
        out[~truth] = _pick(rng, u_values, u_probs, neg)
        return out

    nome = field(
        [1.0, 0.75, 0.4], [0.7, 0.2, 0.1], [0, 0.3, 0.7, 1], [0.5, 0.3, 0.15, 0.05]
    )
    mae = field(
        [1.0, 0.6, 0.0], [0.7, 0.2, 0.1], [0, 0.4, 0.8, 1], [0.6, 0.25, 0.1, 0.05]
    )
    dtnasc = field([1.0, 0.5, 0.0], [0.8, 0.1, 0.1], [0.0, 0.5, 1.0], [0.8, 0.15, 0.05])
    endereco = field(
        [1.0, 0.7, 0.2, 0.0],
        [0.5, 0.2, 0.2, 0.1],
        [0, 0.3, 0.7, 1],
        [0.6, 0.25, 0.1, 0.05],
    )
    municipio = field([1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.5, 0.5])
    nota = np.empty(n)
    nota[truth] = rng.normal(8.8, 0.8, pos)
    nota[~truth] = rng.normal(2.5, 1.8, neg)
    nota = np.clip(nota, 0.0, 10.0)
    passo = rng.choice([6, 9, 11], size=n)
    cdr = cdr_columns(truth)
    df = pd.DataFrame(
        {
            **cdr,
            "COMPREC": np.arange(n),
            "REFREC": np.arange(n) + 100_000,
            "PASSO": passo,
            "SCORE": np.round(nota * 0.95 + rng.normal(0, 0.3, n), 4),
            "nota final": nota,
            "nome_score_total": nome,
            "mae_score_total": mae,
            "mae_missing": (rng.random(n) < 0.05).astype(int),
            "dtnasc_score_total": dtnasc,
            "endereco_score_total": endereco,
            "municipio_score": municipio,
            "dtnasc_all_zero": (dtnasc == 0).astype(int),
            "endereco_zero": (endereco == 0).astype(int),
            "PAR": np.where(truth, 1, 0),
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


def legacy_cfg(**kwargs: Any) -> SelfCalibratingConfig:
    """Config of the pre-CdR tests: it FORCES non-default settings.

    Forced here (override with ``kwargs``): ``mode="auto"``, ``field_set="composites"``
    (the five loader composites), ``max_round_reviews=150``, ``max_reviews=500``,
    ``max_rounds=2``, ``min_new_positives=0`` and ``auto_max_low_conf_fraction=1.0``
    (guard off). The tests of the real default configuration use
    ``SelfCalibratingConfig()`` directly.
    """
    base: dict[str, Any] = {
        "mode": "auto",
        "max_round_reviews": 150,
        "max_reviews": 500,
        "max_rounds": 2,
        "min_new_positives": 0,
        # the pre-CdR-subscores tests run on the five loader composites
        "field_set": "composites",
        # the dominant-region guard has its own tests; keep the older ones unguarded
        "auto_max_low_conf_fraction": 1.0,
    }
    base.update(kwargs)
    return SelfCalibratingConfig(**base)


def fs_prior_model(df: pd.DataFrame, config: SelfCalibratingConfig):
    """Design matrix and the (projected) Fellegi-Sunter prior model of a session."""
    levels = discretize_fields(df, config.field_set)
    fs = fit_fellegi_sunter(levels)
    X, names = design_matrix(levels)
    prior = RecalibrationPrior.from_fellegi_sunter(
        fs,
        names,
        sd_intercept=config.effective_sd_intercept,
        sd_coef=config.effective_sd_coef,
        trust_intercept=config.effective_trust_intercept,
        trust_coef=config.effective_trust_coef,
    )
    return X, prior.as_recalibration()


def truth_callback(truth: Mapping[str, bool]) -> Callable[[list[dict]], dict[str, str]]:
    def callback(items: list[dict]) -> dict[str, str]:
        return {
            it["pair_id"]: "MATCH" if truth[it["pair_id"]] else "NONMATCH"
            for it in items
        }

    return callback


def drive(
    session: SelfCalibratingSession,
    callback: Callable[[list[dict]], Mapping[str, str]] | None = None,
) -> list[SessionEvent]:
    out: list[SessionEvent] = []
    while not session.finished:
        if session.awaiting_human:
            assert callback is not None
            session.submit_human_decisions(callback(session.gate_items))
        else:
            out.append(session.step())
    return out


def as_dicts(events: list[SessionEvent]) -> list[dict[str, Any]]:
    return [e.to_dict() for e in events]


def kinds(events: list[SessionEvent]) -> list[str]:
    return [e.kind for e in events]


GRAMMAR = re.compile(
    r"^O(?:(?:P(?:a{2,3}F|E))*SGH*BL*RD)*(?:P(?:a{2,3}F|E))*SGH*BL*RZ$"
)
LETTER = {
    "ordering": "O",
    "pair_presented": "P",
    "agent_verdict": "a",
    "final_verdict": "F",
    "review_error": "E",
    "round_stopped": "S",
    "gate_opened": "G",
    "human_decision": "H",
    "holdout_built": "B",
    "alarm": "L",
    "recalibration": "R",
    "reordering": "D",
    "run_finished": "Z",
}


def grammar_string(events: list[SessionEvent]) -> str:
    return "".join(LETTER[e.kind] for e in events)


# --------------------------------------------------------------------------
# 1. Full run and grammar
# --------------------------------------------------------------------------


def test_full_auto_run_follows_grammar(data):
    df, truth = data
    session = SelfCalibratingSession(df, OracleReviewer(truth), legacy_cfg())
    events = drive(session)
    assert session.finished
    assert GRAMMAR.match(grammar_string(events)), grammar_string(events)[:200]
    assert [e.seq for e in events] == list(range(len(events)))
    assert events == session.events
    assert events[-1].kind == "run_finished"
    with pytest.raises(StopIteration):
        session.step()
    # Every event is plain JSON.
    json.dumps(as_dicts(events))
    results = session.results()
    assert results["pair_id"].is_unique
    assert (results["llm_decision"] == "MATCH").sum() > 0
    assert results["llm_match_unverified"].equals(results["llm_decision"] == "MATCH")


def test_one_event_per_step(data):
    df, truth = data
    session = SelfCalibratingSession(
        df, OracleReviewer(truth), legacy_cfg(max_rounds=1)
    )
    for expected_seq in range(30):
        event = session.step()
        assert event.seq == expected_seq
        assert len(session.events) == expected_seq + 1


def test_run_self_calibrating_helper(data):
    df, truth = data
    results, summary, events = run_self_calibrating(
        df, OracleReviewer(truth), legacy_cfg(max_rounds=1)
    )
    assert len(results) == summary["n_reviewed"]
    assert events[-1].kind == "run_finished"
    with pytest.raises(ValueError, match="human_callback"):
        run_self_calibrating(df, OracleReviewer(truth), SelfCalibratingConfig())
    results_h, _, _ = run_self_calibrating(
        df,
        OracleReviewer(truth),
        legacy_cfg(mode="human", max_rounds=1),
        human_callback=truth_callback(truth),
    )
    assert len(results_h) > 0


# --------------------------------------------------------------------------
# 2. Determinism
# --------------------------------------------------------------------------


def test_determinism_and_seed_sensitivity(data):
    df, truth = data

    def run(seed: int) -> str:
        session = SelfCalibratingSession(
            df, OracleReviewer(truth), legacy_cfg(seed=seed)
        )
        return json.dumps(as_dicts(drive(session)), sort_keys=True)

    assert run(3) == run(3)
    assert run(3) != run(4)


# --------------------------------------------------------------------------
# 3. Resume
# --------------------------------------------------------------------------


def _restore(session: SelfCalibratingSession, df: pd.DataFrame, reviewer):
    state = json.loads(json.dumps(session.to_state()))
    assert state["version"] == 1
    assert "fingerprint" in state
    return SelfCalibratingSession.from_state(state, df, reviewer)


def test_resume_matches_uninterrupted_run_auto(data):
    df, truth = data
    config = legacy_cfg()
    reference = as_dicts(
        drive(SelfCalibratingSession(df, OracleReviewer(truth), config))
    )
    ks = {
        0,
        1,
        2,
        next(i for i, e in enumerate(reference) if e["kind"] == "pair_presented") + 1,
        next(i for i, e in enumerate(reference) if e["kind"] == "agent_verdict") + 1,
        next(i for i, e in enumerate(reference) if e["kind"] == "round_stopped") + 1,
        next(i for i, e in enumerate(reference) if e["kind"] == "holdout_built") + 1,
        next(i for i, e in enumerate(reference) if e["kind"] == "recalibration") + 1,
        next(i for i, e in enumerate(reference) if e["kind"] == "reordering") + 1,
    }
    for k in sorted(ks):
        session = SelfCalibratingSession(df, OracleReviewer(truth), config)
        for _ in range(k):
            session.step()
        restored = _restore(session, df, OracleReviewer(truth))
        assert as_dicts(drive(restored)) == reference[k:], f"cut at {k}"


def test_resume_while_awaiting_human(data):
    df, truth = data
    config = legacy_cfg(mode="human")
    reviewer_kwargs = {"fp_rate": 0.1, "unsure_rate": 0.1, "seed": 5}
    callback = truth_callback(truth)
    reference_session = SelfCalibratingSession(
        df, OracleReviewer(truth, **reviewer_kwargs), config
    )
    reference = as_dicts(drive(reference_session, callback))

    session = SelfCalibratingSession(
        df, OracleReviewer(truth, **reviewer_kwargs), config
    )
    while not session.awaiting_human:
        session.step()
    k = len(session.events)
    restored = _restore(session, df, OracleReviewer(truth, **reviewer_kwargs))
    assert restored.awaiting_human
    assert restored.gate_items == session.gate_items
    assert as_dicts(drive(restored, callback)) == reference[k:]


def test_from_state_rejects_different_dataframe(data):
    df, truth = data
    session = SelfCalibratingSession(df, OracleReviewer(truth), legacy_cfg())
    session.step()
    state = session.to_state()
    other = df.copy()
    other.loc[0, "nome_score_total"] = 0.123
    with pytest.raises(ValueError, match="does not match"):
        SelfCalibratingSession.from_state(state, other, OracleReviewer(truth))
    bad = dict(state, version=99)
    with pytest.raises(ValueError, match="version"):
        SelfCalibratingSession.from_state(bad, df, OracleReviewer(truth))


# --------------------------------------------------------------------------
# 4. Human gate
# --------------------------------------------------------------------------


def _human_cfg(**kwargs: Any) -> SelfCalibratingConfig:
    params: dict[str, Any] = {
        "mode": "human",
        "max_rounds": 1,
        "max_round_reviews": 300,
        **kwargs,
    }
    return legacy_cfg(**params)


def _until_awaiting(session: SelfCalibratingSession) -> list[SessionEvent]:
    out = []
    while not session.awaiting_human:
        out.append(session.step())
    return out


def _gate_session(df, truth, **kwargs: Any) -> SelfCalibratingSession:
    reviewer = OracleReviewer(truth, fp_rate=0.3, fn_rate=0.3, unsure_rate=0.1, seed=1)
    return SelfCalibratingSession(df, reviewer, _human_cfg(**kwargs))


def _priority(results: pd.DataFrame, decision: str) -> list[str]:
    sub = results[results["llm_decision"] == decision]
    return sorted(sub.index, key=lambda pid: (-sub.loc[pid, "p_cur_at_review"], pid))


def test_human_gate_follows_priority_and_the_per_round_budget(data):
    df, truth = data
    session = _gate_session(df, truth)  # human_budget_per_round=50 by default
    events = _until_awaiting(session)
    gate = events[-1].payload
    assert gate["awaiting"] is True
    with pytest.raises(RuntimeError, match="awaiting"):
        session.step()
    items = session.gate_items
    results = session.results().set_index("pair_id")
    matches = _priority(results, "MATCH")
    assert len(matches) > 50
    # only the first 50 MATCH (decreasing p_cur, ties by pair_id) fit the budget
    assert [it["pair_id"] for it in items] == matches[:50]
    assert {it["kind"] for it in items} == {"match"}
    n_unsure = int((results["llm_decision"] == "UNSURE").sum())
    by_kind = gate["candidates_by_kind"]
    assert by_kind["match"] == len(matches) and by_kind["unsure"] == n_unsure
    assert gate["n_candidates"] == sum(by_kind.values())
    assert gate["n_gate"] == gate["n_items"] == 50
    assert gate["n_over_budget"] == gate["n_candidates"] - 50
    assert gate["human_budget_per_round"] == 50
    assert gate["human_budget_remaining"] is None  # no total budget
    over = results[results["human_decision_source"] == "over_budget"]
    assert len(over) == gate["n_over_budget"]
    assert set(matches[50:]) <= set(over.index)
    assert over["human_decision"].isna().all()
    assert int(results["audited"].sum()) == 0  # the audit sample did not fit

    # no implicit confirmation: every item needs a decision
    with pytest.raises(ValueError, match="not in the gate"):
        session.submit_human_decisions({"nope": "MATCH"})
    invalid: dict[str, Any] = {items[0]["pair_id"]: "UNSURE"}
    with pytest.raises(ValueError, match="SKIP"):
        session.submit_human_decisions(invalid)
    with pytest.raises(ValueError, match="50 of 50 gate items have no decision"):
        session.submit_human_decisions({})
    everything = truth_callback(truth)(items)
    partial = {k: v for k, v in everything.items() if k != items[-1]["pair_id"]}
    with pytest.raises(ValueError, match="1 of 50 gate items") as caught:
        session.submit_human_decisions(partial)
    assert items[-1]["pair_id"] in str(caught.value)
    assert session.awaiting_human  # a rejected submission changes nothing

    session.submit_human_decisions(everything)
    rest = drive(session)
    human_events = [e for e in rest if e.kind == "human_decision"]
    assert len(human_events) == 50
    assert kinds(rest)[:50] == ["human_decision"] * 50
    final = session.results().set_index("pair_id")
    for it in items:
        row = final.loc[it["pair_id"]]
        assert row["human_decision_source"] == "human"
        assert row["human_decision"] == (
            "MATCH" if truth[it["pair_id"]] else "NONMATCH"
        )
        assert row["label"] == float(truth[it["pair_id"]])
        assert row["label_source"] == "human"
        assert not row["llm_match_unverified"]
    flagged = [
        pid
        for pid in final.index
        if final.loc[pid, "llm_decision"] == "MATCH" and not truth[pid]
    ]
    assert flagged, "expected some false positives from the oracle"
    # over-budget pairs have no label and stay unverified LLM MATCH
    over_final = final[final["human_decision_source"] == "over_budget"]
    assert over_final["label"].isna().all()
    assert (over_final["label_source"] == "excluded_over_budget").all()
    assert over_final.loc[
        over_final["llm_decision"] == "MATCH", "llm_match_unverified"
    ].all()
    # there is no positive label without a human MATCH
    assert (final.loc[final["label"] == 1.0, "label_source"] == "human").all()
    built = next(e for e in rest if e.kind == "holdout_built")
    assert built.payload["n_candidates"] == gate["n_candidates"]
    assert built.payload["n_gate"] == 50
    assert built.payload["n_over_budget"] == gate["n_over_budget"]
    assert built.payload["label_sources"]["excluded_over_budget"] == len(over_final)


def test_budget_priority_is_match_then_unsure_then_audit(data):
    df, truth = data
    probe = _gate_session(df, truth, human_budget_per_round=None)
    _until_awaiting(probe)
    res = probe.results().set_index("pair_id")
    n_match = int((res["llm_decision"] == "MATCH").sum())
    unsure = _priority(res, "UNSURE")
    assert len(unsure) > 3
    # the unlimited gate lists MATCH, then UNSURE, then the audit sample
    order = [it["kind"] for it in probe.gate_items]
    assert order == sorted(order, key=["match", "unsure", "audit"].index)
    assert 1 <= order.count("audit") <= 10  # nonmatch_audit_n default
    n_audit = order.count("audit")

    session = _gate_session(df, truth, human_budget_per_round=n_match + 2)
    events = _until_awaiting(session)
    gate = events[-1].payload
    items = session.gate_items
    assert [it["kind"] for it in items] == ["match"] * n_match + ["unsure"] * 2
    assert [it["pair_id"] for it in items if it["kind"] == "unsure"] == unsure[:2]
    assert gate["n_over_budget"] == len(unsure) - 2 + n_audit
    assert gate["over_budget_by_kind"] == {
        "unsure": len(unsure) - 2,
        "audit": n_audit,
    }
    final = session.results().set_index("pair_id")
    assert int(final["audited"].sum()) == 0
    assert (final.loc[unsure[2:], "human_decision_source"] == "over_budget").all()


def test_audit_gate_and_the_informative_sanity_alarm(data):
    df, truth = data
    session = _gate_session(df, truth, human_budget_per_round=None)
    events = _until_awaiting(session)
    items = session.gate_items
    results = session.results().set_index("pair_id")
    audits = [it for it in items if it["kind"] == "audit"]
    assert 0 < len(audits) <= 10
    for it in audits:
        assert it["llm_decision"] == "NONMATCH"
        assert results.loc[it["pair_id"], "rank_fraction_at_review"] < 0.05
        assert bool(results.loc[it["pair_id"], "audited"])
    assert int(results["audited"].sum()) == len(audits)
    assert events[-1].payload["n_over_budget"] == 0
    session.submit_human_decisions(truth_callback(truth)(items))
    rest = drive(session)
    alarms = [e for e in rest if e.kind == "alarm"]
    assert [a.payload["type"] for a in alarms] == ["audit_match_found"]
    payload = alarms[0].payload
    assert payload["n_audited"] == len(audits)
    assert payload["n_matches_found"] >= 1
    assert payload["rate"] == payload["n_matches_found"] / len(audits)
    assert "no statistical power" in payload["note"]
    assert "not an estimator" in payload["note"]
    # no audited match, no alarm
    clean = SelfCalibratingSession(
        df,
        OracleReviewer({pid: False for pid in truth}),
        _human_cfg(human_budget_per_round=None),
    )
    _until_awaiting(clean)
    audits_clean = [it for it in clean.gate_items if it["kind"] == "audit"]
    clean.submit_human_decisions({it["pair_id"]: "NONMATCH" for it in clean.gate_items})
    assert audits_clean
    assert not any(e.kind == "alarm" for e in drive(clean))


def test_audit_size_is_capped(data):
    df, truth = data
    session = SelfCalibratingSession(
        df,
        OracleReviewer(truth, fn_rate=0.5),
        _human_cfg(nonmatch_audit_n=5, human_budget_per_round=None),
    )
    _until_awaiting(session)
    assert sum(it["kind"] == "audit" for it in session.gate_items) == 5
    assert SelfCalibratingConfig().nonmatch_audit_n == 10


def test_skip_decisions_leave_the_item_out_of_the_holdout(data):
    df, truth = data
    session = _gate_session(df, truth, human_budget_per_round=None)
    _until_awaiting(session)
    items = session.gate_items
    decisions: dict[str, Any] = dict(truth_callback(truth)(items))
    skipped = [it["pair_id"] for it in items[:3]] + [
        it["pair_id"] for it in items if it["kind"] == "unsure"
    ][:2]
    for pid in skipped:
        decisions[pid] = "SKIP"
    session.submit_human_decisions(decisions)
    rest = drive(session)
    events = {
        e.payload["pair_id"]: e.payload for e in rest if e.kind == "human_decision"
    }
    final = session.results().set_index("pair_id")
    for pid in skipped:
        row = final.loc[pid]
        assert pd.isna(row["human_decision"])
        assert row["human_decision_source"] == "human_skip"
        assert pd.isna(row["label"]) and row["label_source"] == "excluded_human_skip"
        assert events[pid]["human_decision"] is None
        assert events[pid]["source"] == "human_skip" and events[pid]["changed"] is False
        assert row["llm_match_unverified"] == (row["llm_decision"] == "MATCH")
    built = next(e for e in rest if e.kind == "holdout_built")
    assert built.payload["label_sources"]["excluded_human_skip"] == len(skipped)
    assert not set(skipped) & set(session.holdout()["pair_id"])


def test_unsure_items_can_be_left_out_of_the_gate(data):
    df, truth = data
    session = _gate_session(
        df, truth, human_budget_per_round=None, gate_include_unsure=False
    )
    events = _until_awaiting(session)
    assert "unsure" not in {it["kind"] for it in session.gate_items}
    assert "unsure" not in events[-1].payload["candidates_by_kind"]
    session.submit_human_decisions(truth_callback(truth)(session.gate_items))
    drive(session)
    final = session.results()
    unsure = final[final["llm_decision"] == "UNSURE"]
    assert len(unsure) > 0
    assert (unsure["label_source"] == "excluded_unsure").all()
    assert unsure["label"].isna().all()


def test_human_unsure_resolved_as_match_is_a_positive(data):
    df, truth = data
    session = _gate_session(df, truth, human_budget_per_round=None)
    _until_awaiting(session)
    decisions = {
        it["pair_id"]: ("MATCH" if it["kind"] == "unsure" else "NONMATCH")
        for it in session.gate_items
    }
    session.submit_human_decisions(decisions)
    drive(session)
    final = session.results().set_index("pair_id")
    unsure = final[final["llm_decision"] == "UNSURE"]
    assert (unsure["label"] == 1.0).all() and (unsure["label_source"] == "human").all()
    assert unsure["llm_match_unverified"].eq(False).all()


@pytest.mark.parametrize("policy", ["llm_nonmatch", "verified_only"])
def test_negative_label_policies(data, policy):
    df, truth = data
    session = _gate_session(
        df, truth, human_budget_per_round=None, negative_labels=policy
    )
    _until_awaiting(session)
    session.submit_human_decisions(truth_callback(truth)(session.gate_items))
    drive(session)
    final = session.results()
    plain = final[(final["llm_decision"] == "NONMATCH") & ~final["audited"]]
    audited = final[final["audited"]]
    assert len(plain) > 0 and len(audited) > 0
    # an audited NONMATCH carries the human verdict under both policies
    assert (audited["label_source"] == "human").all()
    assert set(audited["label"]) <= {0.0, 1.0}
    if policy == "llm_nonmatch":
        assert (plain["label"] == 0.0).all()
        assert (plain["label_source"] == "llm_nonmatch").all()
    else:
        assert plain["label"].isna().all()
        assert (plain["label_source"] == "excluded_unverified_nonmatch").all()
    # a positive always comes from a human MATCH
    assert (final.loc[final["label"] == 1.0, "human_decision"] == "MATCH").all()
    holdout = session.holdout()
    assert set(holdout["label_source"]) <= {"human", "llm_nonmatch"}
    assert ("llm_nonmatch" in set(holdout["label_source"])) == (
        policy == "llm_nonmatch"
    )


def test_total_human_budget_ends_the_run_after_the_round(data):
    df, truth = data
    config = _human_cfg(
        human_budget_per_round=20,
        human_budget_total=30,
        max_rounds=4,
        max_round_reviews=120,
        max_reviews=5000,
        min_new_positives=0,
    )
    session = SelfCalibratingSession(df, OracleReviewer(truth, fp_rate=0.3), config)
    events = drive(session, truth_callback(truth))
    gates = [e for e in events if e.kind == "gate_opened"]
    assert [g.payload["n_gate"] for g in gates] == [20, 10]
    assert [g.payload["human_budget_remaining"] for g in gates] == [10, 0]
    assert gates[1].payload["n_over_budget"] > 0
    # the round in progress still gets its holdout and recalibration
    tail = kinds(events)[-4:]
    assert tail[-3:] == ["holdout_built", "recalibration", "run_finished"]
    assert events[-1].payload["reason"] == "human_budget"
    assert events[-1].round == 2
    assert session.summary()["human_used"] == 30
    assert session.summary()["finish_reason"] == "human_budget"
    # a total that is smaller than the first gate ends after round 1
    tiny = SelfCalibratingSession(
        df,
        OracleReviewer(truth, fp_rate=0.3),
        _human_cfg(human_budget_total=5, max_rounds=3, min_new_positives=0),
    )
    out = drive(tiny, truth_callback(truth))
    assert [e.payload["n_gate"] for e in out if e.kind == "gate_opened"] == [5]
    assert out[-1].payload["reason"] == "human_budget" and out[-1].round == 1
    # no limit: the budget never ends the run
    free = SelfCalibratingSession(
        df,
        OracleReviewer(truth, fp_rate=0.3),
        _human_cfg(human_budget_per_round=None, max_rounds=2, min_new_positives=0),
    )
    assert drive(free, truth_callback(truth))[-1].payload["reason"] != "human_budget"


def test_run_self_calibrating_needs_a_decision_for_every_item(data):
    df, truth = data
    full = truth_callback(truth)

    def forgetful(items: list[dict]) -> dict[str, str]:
        decisions = full(items)
        decisions.pop(items[0]["pair_id"])
        return decisions

    with pytest.raises(ValueError, match="no decision"):
        run_self_calibrating(
            df,
            OracleReviewer(truth),
            _human_cfg(),
            human_callback=forgetful,
        )

    def skipper(items: list[dict]) -> dict[str, str]:
        return {it["pair_id"]: "SKIP" for it in items}

    results, summary, _ = run_self_calibrating(
        df, OracleReviewer(truth), _human_cfg(), human_callback=skipper
    )
    assert results["label"].eq(1.0).sum() == 0  # nothing was confirmed
    assert summary["n_llm_match_unverified"] == int(
        (results["llm_decision"] == "MATCH").sum()
    )


# --------------------------------------------------------------------------
# 5. Stopping rule
# --------------------------------------------------------------------------


def test_stop_probability_closed_form():
    tau = 0.05
    for n in (0, 1, 10, 57, 58, 100):
        assert stop_probability(0, n, tau) == pytest.approx((1 - tau) ** (n + 1))
    assert zero_match_reviews_needed(0.05, 0.05) == 58
    assert stop_probability(0, 58, 0.05) < 0.05 <= stop_probability(0, 57, 0.05)
    assert stop_probability(10, 20, 0.05) > 0.99


def test_final_verdict_carries_stop_statistic_and_posterior_stop(data):
    df, truth = data
    nothing = {pid: False for pid in truth}
    session = SelfCalibratingSession(
        df,
        OracleReviewer(nothing),
        legacy_cfg(max_rounds=1, max_round_reviews=500, max_reviews=1000),
    )
    events = drive(session)
    finals = [e for e in events if e.kind == "final_verdict"]
    assert finals
    for e in finals[::25]:
        stat = e.payload["stop_statistic"]
        live = [a["p_gt_tau"] for a in stat["arms"] if not a["exhausted"]]
        assert stat["max_p_gt_tau"] == (max(live) if live else None)
    stopped = next(e for e in events if e.kind == "round_stopped")
    assert stopped.payload["reason"] == "posterior"
    assert stopped.payload["stop_statistic"]["max_p_gt_tau"] < 0.05
    live = [a for a in stopped.payload["stop_statistic"]["arms"] if not a["exhausted"]]
    assert live
    # every live arm needs at least n* = 58 reviews without MATCH to stop
    assert all(a["trials"] >= zero_match_reviews_needed(0.05, 0.05) for a in live)
    assert all(a["successes"] == 0 for a in live)


# --------------------------------------------------------------------------
# 6. PAR / TARGET are never read
# --------------------------------------------------------------------------


ALLOWED_COLUMNS = frozenset(
    {
        "COMPREC",
        "REFREC",
        "PASSO",
        "nota final",
        "SCORE",
        "auto_low_confidence",
        "nome_score_total",
        "mae_score_total",
        "mae_missing",
        "dtnasc_score_total",
        "endereco_score_total",
        "municipio_score",
        "dtnasc_all_zero",
        "endereco_zero",
    }
)


@pytest.mark.parametrize("field_set", ["composites", "cdr_subscores"])
def test_par_and_target_are_ignored(data, field_set):
    df, truth = data
    config = legacy_cfg(field_set=field_set)
    session = SelfCalibratingSession(df, OracleReviewer(truth), config)
    declared = set(scq._declared_columns(df, field_set))
    # the columns the session reads (and fingerprints) are a closed, known set
    assert declared <= ALLOWED_COLUMNS | set(CDR_SUBSCORE_COLUMNS)
    assert not declared & {"PAR", "TARGET"}
    assert set(session._used.columns) == declared
    base = json.dumps(as_dicts(drive(session)))
    shuffled = df.copy()
    shuffled["PAR"] = np.random.default_rng(0).permutation(shuffled["PAR"].to_numpy())
    shuffled["TARGET"] = np.random.default_rng(1).integers(0, 2, len(df))
    other = json.dumps(
        as_dicts(drive(SelfCalibratingSession(shuffled, OracleReviewer(truth), config)))
    )
    assert base == other
    no_par = df.drop(columns=["PAR"])
    assert base == json.dumps(
        as_dicts(drive(SelfCalibratingSession(no_par, OracleReviewer(truth), config)))
    )


# --------------------------------------------------------------------------
# 7. Score only in round 1
# --------------------------------------------------------------------------


def test_ordering_has_no_score_feature(data):
    df, truth = data
    session = SelfCalibratingSession(df, OracleReviewer(truth), legacy_cfg())
    first = session.step()
    assert first.kind == "ordering"
    names = first.payload["feature_names"]
    assert tuple(names) == design_matrix(discretize_fields(df))[1]
    assert not any("nota" in n for n in names)
    assert not any(n.endswith(":L0") for n in names)
    assert first.payload["p0_method"] in {"anchor_platt", "stub"}
    assert first.payload["low_confidence_source"] == "below_cutoff"


def grey_mask(df: pd.DataFrame) -> np.ndarray:
    nota = df["nota final"].to_numpy()
    return (nota >= 5) & (nota < 8)


def _row_index(df: pd.DataFrame) -> dict[str, int]:
    return {
        make_pair_id(c, r, p): i
        for i, (c, r, p) in enumerate(
            zip(df["COMPREC"], df["REFREC"], df["PASSO"], strict=True)
        )
    }


def test_round_two_uses_current_model(data):
    df, truth = data
    session = SelfCalibratingSession(
        df,
        OracleReviewer(truth),
        legacy_cfg(min_labeled=10),
        low_confidence=grey_mask(df),
    )
    events = drive(session)
    recal = next(e for e in events if e.kind == "recalibration" and e.round == 1)
    assert recal.payload["status"] == "fitted"
    X, _ = design_matrix(discretize_fields(df))
    expected = expit(recal.payload["intercept"] + X @ np.asarray(recal.payload["coef"]))
    results = session.results()
    round2 = results[results["round"] == 2]
    assert len(round2) > 0
    pos = _row_index(df)
    got = round2["p_cur_at_review"].to_numpy()
    want = np.array([expected[pos[pid]] for pid in round2["pair_id"]])
    np.testing.assert_allclose(got, want, rtol=1e-9)


def test_skipped_recalibration_keeps_fs_prior_for_round_two(data):
    df, truth = data
    session = SelfCalibratingSession(
        df,
        OracleReviewer(truth),
        legacy_cfg(min_labeled=10_000),
        low_confidence=grey_mask(df),
    )
    events = drive(session)
    recal = next(e for e in events if e.kind == "recalibration" and e.round == 1)
    assert recal.payload["status"] == "skipped_insufficient"
    X, prior_model = fs_prior_model(df, session.config)
    expected = prior_model.predict_proba(X)
    results = session.results()
    round2 = results[results["round"] == 2]
    assert len(round2) > 0
    pos = _row_index(df)
    want = np.array([expected[pos[pid]] for pid in round2["pair_id"]])
    np.testing.assert_allclose(round2["p_cur_at_review"].to_numpy(), want, rtol=1e-6)
    assert session.current_model is not None
    assert session.current_model.coef == pytest.approx(prior_model.coef, rel=1e-6)
    assert session.current_model.intercept == pytest.approx(
        prior_model.intercept, rel=1e-6
    )


# --------------------------------------------------------------------------
# 8. Automatic-mode safeguards
# --------------------------------------------------------------------------


def test_experimental_warning_only_in_auto(data):
    df, truth = data
    with pytest.warns(ExperimentalWarning):
        SelfCalibratingSession(df, OracleReviewer(truth), legacy_cfg())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SelfCalibratingSession(df, OracleReviewer(truth), legacy_cfg(mode="human"))
    assert not [w for w in caught if issubclass(w.category, ExperimentalWarning)]


def test_arbiter_match_is_excluded_from_holdout(data):
    df, truth = data
    session = SelfCalibratingSession(
        df,
        OracleReviewer(truth, disagreement_rate=1.0),
        legacy_cfg(max_rounds=1),
        low_confidence=np.zeros(len(df), dtype=bool),
    )
    drive(session)
    results = session.results()
    matches = results[results["llm_decision"] == "MATCH"]
    assert len(matches) > 0
    assert (matches["protocol"] == "arbiter").all()
    assert (matches["label_source"] == "excluded_arbiter").all()
    assert matches["label"].isna().all()
    assert (results["label"] == 1.0).sum() == 0
    assert (
        results["llm_match_unverified"] == (results["llm_decision"] == "MATCH")
    ).all()


def test_low_confidence_region_exclude_and_soft(data):
    df, truth = data
    mask = np.random.default_rng(3).random(len(df)) < 0.4
    exclude = SelfCalibratingSession(
        df, OracleReviewer(truth), legacy_cfg(max_rounds=1), low_confidence=mask
    )
    ordering = exclude.step()
    assert ordering.payload["low_confidence_source"] == "argument"
    assert ordering.payload["n_low_confidence"] == int(mask.sum())
    drive(exclude)
    res = exclude.results()
    low = res[res["low_confidence"]]
    assert len(low) > 0 and (low["llm_decision"] == "MATCH").any()
    assert low["label"].isna().all()
    assert (low["label_source"] == "excluded_low_conf").all()
    high = res[~res["low_confidence"] & (res["llm_decision"] == "NONMATCH")]
    assert (high["label"] == 0.0).all()

    soft = SelfCalibratingSession(
        df,
        OracleReviewer(truth),
        legacy_cfg(max_rounds=1, auto_low_conf_policy="soft"),
        low_confidence=mask,
    )
    drive(soft)
    res = soft.results()
    low = res[res["low_confidence"]]
    assert (low.loc[low["llm_decision"] == "MATCH", "label"] == 0.16).all()
    assert (low.loc[low["llm_decision"] == "NONMATCH", "label"] == 0.0).all()


def test_low_confidence_sources(data):
    df, truth = data
    with_col = df.copy()
    with_col["auto_low_confidence"] = np.arange(len(df)) % 5 == 0
    first = SelfCalibratingSession(with_col, OracleReviewer(truth), legacy_cfg()).step()
    assert first.payload["low_confidence_source"] == "column"
    assert first.payload["n_low_confidence"] == int(
        with_col["auto_low_confidence"].sum()
    )
    no_score = df.drop(columns=["SCORE"])
    first = SelfCalibratingSession(no_score, OracleReviewer(truth), legacy_cfg()).step()
    assert first.payload["low_confidence_source"] == "grey_bands_nota"
    nota = df["nota final"]
    assert first.payload["n_low_confidence"] == int(((nota >= 5) & (nota < 8)).sum())
    off = df.assign(PASSO=7)
    first = SelfCalibratingSession(off, OracleReviewer(truth), legacy_cfg()).step()
    assert first.payload["low_confidence_source"] == "grey_bands_nota"


def test_all_match_reviewer_triggers_anomaly(data):
    df, truth = data
    everything = {pid: True for pid in truth}
    session = SelfCalibratingSession(
        df, OracleReviewer(everything), legacy_cfg(max_rounds=3, max_round_reviews=120)
    )
    events = drive(session)
    alarm = next(e for e in events if e.kind == "alarm")
    assert alarm.payload["type"] == "auto_anomaly"
    assert alarm.payload["bands"]
    assert events[-1].kind == "run_finished"
    assert events[-1].payload["reason"] == "auto_anomaly"
    recal = next(e for e in events if e.kind == "recalibration")
    assert recal.payload["reverted"] is True
    # the preserved model keeps its own status instead of a fake fallback label
    assert recal.payload["status"] == recal.payload["previous_status"]
    assert recal.payload["status"] != "fallback_prior"
    _, prior_model = fs_prior_model(df, session.config)
    assert session.current_model is not None
    assert session.current_model.coef == pytest.approx(prior_model.coef, rel=1e-6)
    assert GRAMMAR.match(grammar_string(events))


# --------------------------------------------------------------------------
# 9. IPW weights
# --------------------------------------------------------------------------


def _labelled_strata(session: SelfCalibratingSession):
    results = session.results()
    members = results[results["label"].notna()]
    holdout = session.holdout().set_index("pair_id")
    return results, members, holdout


def test_stratum_ipw_weights_top_and_explore(data):
    df, truth = data
    session = SelfCalibratingSession(
        df,
        OracleReviewer(truth),
        legacy_cfg(max_rounds=2, weight_max_ratio=1e9),
        low_confidence=np.zeros(len(df), dtype=bool),
    )
    drive(session)
    results, members, holdout = _labelled_strata(session)
    weights = members["holdout_weight"].to_numpy()
    assert len(members) > 20
    assert weights.mean() == pytest.approx(1.0)
    assert (weights > 0).all()
    assert {"r1:top", "r1:explore"} <= set(holdout["stratum"])
    pool = {e["round"]: e["size"] for e in session.summary()["explore_sizes"]}
    explore = results["arm"] == "explore"
    n_e = results[explore].groupby("round").size().to_dict()
    raw = np.array(
        [
            pool[r] / n_e[r] if is_explore else 1.0
            for r, is_explore in zip(
                members["round"], explore[members.index], strict=True
            )
        ]
    )
    np.testing.assert_allclose(weights, raw / raw.mean())
    # the consumed prefix is a census: every top weight is the same
    top = members.loc[~explore[members.index], "holdout_weight"]
    assert top.nunique() == 1
    np.testing.assert_allclose(np.sort(holdout["weight"]), np.sort(weights))


def test_explore_weight_is_capped_by_the_ratio(data):
    df, truth = data
    session = SelfCalibratingSession(
        df,
        OracleReviewer(truth),
        legacy_cfg(max_rounds=2, weight_max_ratio=10.0),
        low_confidence=np.zeros(len(df), dtype=bool),
    )
    events = drive(session)
    results, members, holdout = _labelled_strata(session)
    w = holdout["weight"]
    top = w[holdout["stratum"].str.endswith(":top")]
    explore = w[holdout["stratum"].str.endswith(":explore")]
    assert len(top) and len(explore)
    assert w.max() / w.min() <= 10.0 + 1e-9
    # N / n_e is far above 10 here, so every explore weight sits on the cap
    np.testing.assert_allclose(explore.to_numpy() / top.iloc[0], 10.0)
    built = [e for e in events if e.kind == "holdout_built"][-1]
    assert built.payload["n_capped"] == len(explore)

    unweighted = SelfCalibratingSession(
        df, OracleReviewer(truth), legacy_cfg(max_rounds=1, weighting="none")
    )
    drive(unweighted)
    assert (unweighted.holdout()["weight"] == 1.0).all()


def test_without_exploration_there_is_a_single_top_stratum(data):
    df, truth = data
    session = SelfCalibratingSession(
        df, OracleReviewer(truth), legacy_cfg(max_rounds=1, explore_every=0)
    )
    events = drive(session)
    results = session.results()
    assert "explore" not in set(results["arm"])
    assert not any(
        e.kind == "pair_presented" and e.payload["arm"] == "explore" for e in events
    )
    holdout = session.holdout()
    assert set(holdout["stratum"]) == {"r1:top"}
    assert (holdout["weight"] == 1.0).all()


# --------------------------------------------------------------------------
# 9b. Exploration arm and top-down bands
# --------------------------------------------------------------------------


def _presented(events: list[SessionEvent]) -> list[SessionEvent]:
    return [e for e in events if e.kind == "pair_presented"]


def test_exploration_calendar_is_one_review_in_ten(data):
    df, truth = data
    session = SelfCalibratingSession(df, OracleReviewer(truth), legacy_cfg())
    events = drive(session)
    presented = _presented(events)
    assert len(presented) > 100
    for i, event in enumerate(presented, start=1):
        assert (event.payload["arm"] == "explore") == (i % 10 == 0), i
        if event.payload["arm"] == "explore":
            assert event.payload["band"] == -1
            assert event.payload["band_label"] == "explore"
    results = session.results()
    assert (results["arm"] == "explore").sum() == len(results) // 10
    assert results["pair_id"].is_unique
    # explored pairs come from all over the ranking, not from the top
    fractions = results.loc[results["arm"] == "explore", "rank_fraction_at_review"]
    assert fractions.max() > 0.5
    some = SelfCalibratingSession(
        df, OracleReviewer(truth), legacy_cfg(explore_every=4)
    )
    pres = _presented(drive(some))
    assert all(
        (e.payload["arm"] == "explore") == (i % 4 == 0)
        for i, e in enumerate(pres, start=1)
    )


def test_exploration_stays_outside_the_bandit_and_the_stopping_rule(data):
    df, truth = data
    everything = {pid: True for pid in truth}
    session = SelfCalibratingSession(
        df,
        OracleReviewer(everything),
        legacy_cfg(max_rounds=1, max_round_reviews=60, anomaly_min_matches=10_000),
    )
    events = drive(session)
    last: dict[str, Any] | None = None
    seen_explore = 0
    for e in events:
        if e.kind not in ("final_verdict", "review_error"):
            continue
        if e.payload["arm"] == "explore":
            seen_explore += 1
            assert e.payload["reward"] is None
            assert e.payload["band"] == -1
            assert last is not None
            # the posterior statistic is exactly the one before the review
            assert e.payload["stop_statistic"] == last
        else:
            assert e.payload["reward"] in (0, 1)
            last = e.payload["stop_statistic"]
    assert seen_explore >= 5
    assert all(
        isinstance(a["arm"], int)
        for e in events
        if e.kind == "final_verdict"
        for a in e.payload["stop_statistic"]["arms"]
    )
    # nothing in the stop statistic ever counts an explore review: the trials of
    # the bandit arms sum to the non-explore reviews (window is not reached)
    final = [e for e in events if e.kind == "final_verdict"][-1]
    n_bandit = int((session.results()["arm"] != "explore").sum())
    assert sum(a["trials"] for a in final.payload["stop_statistic"]["arms"]) == n_bandit
    assert all(
        a["successes"] == a["trials"] for a in final.payload["stop_statistic"]["arms"]
    )


def test_explore_review_errors_do_not_touch_the_bandit(data):
    df, truth = data
    probe = SelfCalibratingSession(df, OracleReviewer(truth), legacy_cfg(max_rounds=1))
    presented = _presented([probe.step() for _ in range(60)])
    explore_pid = next(
        e.payload["pair_id"] for e in presented if e.payload["arm"] == "explore"
    )
    session = SelfCalibratingSession(
        df,
        OracleReviewer(truth, error_pairs=frozenset({explore_pid})),
        legacy_cfg(max_rounds=1),
    )
    events = drive(session)
    errors = [e for e in events if e.kind == "review_error"]
    assert [e.payload["pair_id"] for e in errors] == [explore_pid]
    assert errors[0].payload["arm"] == "explore" and errors[0].payload["reward"] is None
    rec = session.results().set_index("pair_id").loc[explore_pid]
    assert rec["llm_decision"] == "ERROR" and rec["arm"] == "explore"


@pytest.mark.parametrize("explore_every", [0, 10])
def test_each_band_is_read_from_the_top_down(data, explore_every):
    df, truth = data
    session = SelfCalibratingSession(
        df,
        OracleReviewer(truth),
        legacy_cfg(max_rounds=1, max_round_reviews=300, explore_every=explore_every),
    )
    presented = _presented(drive(session))
    by_arm: dict[Any, list[tuple[float, str]]] = {}
    for e in presented:
        by_arm.setdefault(e.payload["arm"], []).append(
            (e.payload["p_cur"], e.payload["pair_id"])
        )
    assert {a for a in by_arm if a != "explore"} >= {0, 1}
    for arm, seq in by_arm.items():
        if arm == "explore":
            continue
        keys = [(-p, pid) for p, pid in seq]
        assert keys == sorted(keys), f"arm {arm} is not read top-down"
    if explore_every:
        explore = [p for p, _ in by_arm["explore"]]
        assert explore != sorted(explore, reverse=True)  # random, not sorted


def _expected_sizes(n: int, edges: tuple[float, ...]) -> list[int]:
    cuts = [0, *(int(np.ceil(e * n - 1e-9)) for e in edges), n]
    return [b - a for a, b in zip(cuts, cuts[1:], strict=False)]


def test_bands_are_rank_quantiles_with_the_stub(data, monkeypatch):
    df, truth = data

    def no_anchors(*args, **kwargs):
        raise _InsufficientAnchors("no anchors")

    monkeypatch.setattr(scq, "fit_anchor_platt", no_anchors)
    session = SelfCalibratingSession(
        df,
        OracleReviewer(truth),
        legacy_cfg(max_rounds=1, max_round_reviews=400, explore_every=0),
    )
    first = session.step()
    assert first.payload["p0_method"] == "stub"
    edges = (0.01, 0.05, 0.20)
    sizes = [b["size"] for b in first.payload["band_sizes"]]
    assert sizes == _expected_sizes(N_PAIRS, edges) == [24, 96, 360, 1920]
    assert [b["label"] for b in first.payload["band_sizes"]] == [
        "rank[0,1%)",
        "rank[1%,5%)",
        "rank[5%,20%)",
        "rank[20%,100%]",
    ]
    assert first.payload["explore"] == {"every": 0, "pool_size": N_PAIRS}
    # a band holds higher probabilities than the next one
    events = drive(session)
    by_band: dict[int, list[float]] = {}
    for e in _presented(events):
        by_band.setdefault(e.payload["band"], []).append(e.payload["p_cur"])
    for upper, lower in zip(sorted(by_band), sorted(by_band)[1:], strict=False):
        assert min(by_band[upper]) >= max(by_band[lower])


def test_bands_are_rank_quantiles_after_calibration_and_in_later_rounds(data):
    df, truth = data
    session = SelfCalibratingSession(
        df, OracleReviewer(truth), legacy_cfg(min_labeled=10)
    )
    events = drive(session)
    first = events[0]
    assert first.payload["p0_method"] == "anchor_platt"
    assert [b["size"] for b in first.payload["band_sizes"]] == [24, 96, 360, 1920]
    reorder = next(e for e in events if e.kind == "reordering")
    remaining = reorder.payload["n_remaining"]
    assert remaining < N_PAIRS
    assert [b["size"] for b in reorder.payload["band_sizes"]] == _expected_sizes(
        remaining, (0.01, 0.05, 0.20)
    )
    assert reorder.payload["explore_pool_size"] == remaining
    # round 2 ranks by the fitted model over the pairs still unreviewed
    recal = next(e for e in events if e.kind == "recalibration")
    assert recal.payload["status"] == "fitted"
    X, _ = design_matrix(discretize_fields(df))
    expected = expit(recal.payload["intercept"] + X @ np.asarray(recal.payload["coef"]))
    pos = _row_index(df)
    for e in _presented(events):
        if e.round == 2:
            assert e.payload["p_cur"] == pytest.approx(
                expected[pos[e.payload["pair_id"]]], rel=1e-9
            )


def test_audit_pool_follows_the_rank_at_round_start(data):
    df, truth = data
    config = _human_cfg(
        nonmatch_audit_n=1000, audit_rank_fraction=0.05, human_budget_per_round=None
    )
    session = SelfCalibratingSession(df, OracleReviewer(truth), config)
    events = _until_awaiting(session)
    results = session.results().set_index("pair_id")
    gate = events[-1].payload
    top = results[
        (results["llm_decision"] == "NONMATCH")
        & (results["rank_fraction_at_review"] < 0.05)
    ]
    assert gate["audit_pool_size"] == len(top) > 0
    assert gate["n_audit"] == len(top)  # n is large enough to take them all
    assert set(results.index[results["audited"]]) == set(top.index)
    # fraction 0 audits nothing
    none = SelfCalibratingSession(
        df, OracleReviewer(truth), _human_cfg(audit_rank_fraction=0.0)
    )
    gate0 = _until_awaiting(none)[-1].payload
    assert gate0["audit_pool_size"] == 0 and gate0["n_audit"] == 0


def test_resume_with_exploration_in_the_middle_is_identical(data):
    df, truth = data
    config = legacy_cfg(max_rounds=2)
    reference = as_dicts(
        drive(SelfCalibratingSession(df, OracleReviewer(truth), config))
    )
    explore_at = [
        i
        for i, e in enumerate(reference)
        if e["kind"] == "pair_presented" and e["payload"]["arm"] == "explore"
    ]
    assert len(explore_at) >= 10
    cuts = {explore_at[0] + 1, explore_at[0] + 2, explore_at[0] + 3, explore_at[-1] + 1}
    # also right before an exploration and across the round boundary
    cuts |= {
        explore_at[1] - 1,
        next(i for i, e in enumerate(reference) if e["kind"] == "reordering") + 1,
    }
    for k in sorted(cuts):
        session = SelfCalibratingSession(df, OracleReviewer(truth), config)
        for _ in range(k):
            session.step()
        restored = _restore(session, df, OracleReviewer(truth))
        assert restored.to_state() == session.to_state(), f"state at {k}"
        assert as_dicts(drive(restored)) == reference[k:], f"cut at {k}"


def test_exploration_is_deterministic_and_seeded(data):
    df, truth = data

    def explored(seed: int) -> list[str]:
        session = SelfCalibratingSession(
            df, OracleReviewer(truth), legacy_cfg(seed=seed, max_rounds=1)
        )
        drive(session)
        res = session.results()
        return res.loc[res["arm"] == "explore", "pair_id"].tolist()

    assert explored(3) == explored(3) != explored(4)


# --------------------------------------------------------------------------
# 10. LLM errors
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["auto", "human"])
def test_three_consecutive_llm_errors_end_the_run(data, mode):
    df, truth = data
    reviewer = OracleReviewer(truth, error_pairs=frozenset(truth))
    session = SelfCalibratingSession(df, reviewer, legacy_cfg(mode=mode, max_rounds=5))
    events = drive(session, truth_callback(truth))
    assert grammar_string(events) == "OPEPEPESGBRZ"
    assert events[-1].payload["reason"] == "llm_errors"
    stopped = next(e for e in events if e.kind == "round_stopped")
    assert stopped.payload["reason"] == "llm_errors"
    assert [
        e.payload["consecutive_errors"] for e in events if e.kind == "review_error"
    ] == [
        1,
        2,
        3,
    ]
    results = session.results()
    assert (results["llm_decision"] == "ERROR").all()
    assert (results["label_source"] == "excluded_error").all()
    gate = next(e for e in events if e.kind == "gate_opened")
    assert gate.payload["awaiting"] is False


def test_error_streak_resets_and_other_exceptions_propagate(data):
    df, truth = data
    probe = SelfCalibratingSession(df, OracleReviewer(truth), legacy_cfg())
    probe_events = []
    while len(probe_events) < 40:
        probe_events.append(probe.step())
    presented = [
        e.payload["pair_id"] for e in probe_events if e.kind == "pair_presented"
    ]
    failing = frozenset({presented[0], presented[2], presented[4]})
    session = SelfCalibratingSession(
        df, OracleReviewer(truth, error_pairs=failing), legacy_cfg(max_rounds=1)
    )
    events = drive(session)
    assert sum(e.kind == "review_error" for e in events) == 3
    assert events[-1].payload["reason"] != "llm_errors"

    class Broken:
        def review(self, dossier: DossierView):
            raise RuntimeError("boom")

    broken = SelfCalibratingSession(df, Broken(), legacy_cfg())
    broken.step()
    broken.step()
    with pytest.raises(RuntimeError, match="boom"):
        broken.step()
    assert broken.events[-1].kind == "pair_presented"  # nothing was recorded


# --------------------------------------------------------------------------
# 11. Dossier content
# --------------------------------------------------------------------------


def test_dossier_has_no_score_nor_model_outputs(data):
    df, truth = data
    reviewer = OracleReviewer(truth)
    session = SelfCalibratingSession(df, reviewer, legacy_cfg(max_rounds=1))
    for _ in range(25):
        session.step()
    assert reviewer.seen_payloads
    for payload in reviewer.seen_payloads:
        assert "model_outputs" not in payload
        assert "nota_final" not in payload["features"]
        assert "nota_final" not in json.dumps(payload)
        assert "PAR" not in json.dumps(payload)


# --------------------------------------------------------------------------
# Config and oracle
# --------------------------------------------------------------------------


def test_config_defaults_roundtrip_and_validation():
    config = SelfCalibratingConfig()
    assert config.mode == "human"
    assert config.field_set == "composites"
    assert (config.human_budget_per_round, config.human_budget_total) == (50, None)
    assert config.gate_include_unsure is True
    assert config.negative_labels == "llm_nonmatch"
    assert config.nonmatch_audit_n == 10
    assert (config.effective_sd_coef, config.effective_sd_intercept) == (1.0, 2.0)
    assert config.effective_trust_coef is None
    assert config.effective_trust_intercept is None
    auto = SelfCalibratingConfig(mode="auto")
    assert (auto.effective_sd_coef, auto.effective_sd_intercept) == (0.5, 1.0)
    assert (auto.effective_trust_coef, auto.effective_trust_intercept) == (1.0, 2.0)
    assert SelfCalibratingConfig(mode="auto", sd_coef=0.3).effective_sd_coef == 0.3
    again = SelfCalibratingConfig.from_dict(json.loads(json.dumps(auto.to_dict())))
    assert again == auto
    assert SelfCalibratingConfig().band_rank_edges == (0.01, 0.05, 0.20)
    assert SelfCalibratingConfig().audit_rank_fraction == 0.05
    assert SelfCalibratingConfig().explore_every == 10
    for bad in (
        {"mode": "x"},
        {"tau": 0.0},
        {"delta": 1.0},
        {"window": 0},
        {"band_rank_edges": (0.5, 0.1)},
        {"band_rank_edges": (0.1, 0.1)},
        {"band_rank_edges": (0.0, 0.5)},
        {"band_rank_edges": (0.1, 1.0)},
        {"band_rank_edges": ()},
        {"explore_every": -1},
        {"explore_every": 1.5},
        {"audit_rank_fraction": 1.5},
        {"weighting": "other"},
        {"auto_low_conf_policy": "x"},
        {"weight_max_ratio": 0.5},
        {"weight_max_ratio": float("inf")},
        {"window": 10},
        {"sd_coef": -1.0},
        {"field_set": "other"},
        {"human_budget_per_round": 0},
        {"human_budget_per_round": -3},
        {"human_budget_total": 0},
        {"human_budget_total": 1.5},
        {"human_budget_total": True},
        {"negative_labels": "all"},
        {"gate_include_unsure": 1},
        {"remove_fields": "features.nota_final"},
        {"remove_fields": ("features.nota_final", "features.nota_final")},
        {"remove_fields": ("features.nota_final", 3)},
    ):
        with pytest.raises(ValueError):
            SelfCalibratingConfig(**bad)
    with pytest.raises(ValueError, match="Unknown"):
        SelfCalibratingConfig.from_dict({"nope": 1})
    for removed in ("band_edges", "audit_p_threshold", "audit_alarm_rate"):
        with pytest.raises(ValueError, match="Unknown"):
            SelfCalibratingConfig.from_dict({removed: 0.1})
    assert SelfCalibratingConfig(explore_every=0).explore_every == 0
    rank = SelfCalibratingConfig(band_rank_edges=(0.02, 0.5), audit_rank_fraction=0.1)
    assert (
        SelfCalibratingConfig.from_dict(json.loads(json.dumps(rank.to_dict()))) == rank
    )


def test_oracle_reviewer_is_order_independent(data):
    df, truth = data
    rows = [df.iloc[i] for i in range(40)]
    ids = list(truth)[:40]
    dossiers = [
        DossierView(build_dossier(r, pair_id=pid), remove_model_outputs=True)
        for r, pid in zip(rows, ids, strict=True)
    ]
    kwargs = {
        "fp_rate": 0.3,
        "fn_rate": 0.3,
        "unsure_rate": 0.2,
        "disagreement_rate": 0.4,
    }
    a = OracleReviewer(truth, seed=9, **kwargs)
    b = OracleReviewer(truth, seed=9, **kwargs)
    forward = [a.review(d) for d in dossiers]
    backward = [b.review(d) for d in reversed(dossiers)][::-1]
    assert [r.decision for r in forward] == [r.decision for r in backward]
    assert [r.protocol for r in forward] == [r.protocol for r in backward]
    assert {r.protocol for r in forward} == {"consensus", "arbiter"}
    assert len(a.seen_payloads) == 40
    perfect = OracleReviewer(truth)
    for d in dossiers:
        result = perfect.review(d)
        assert result.decision == ("MATCH" if truth[d.pair_id] else "NONMATCH")
        assert result.protocol == "consensus" and result.arbiter is None
        assert result.total_latency_s == 0.0 and result.agent_a.tokens_prompt == 0
    failing = OracleReviewer(truth, error_pairs=frozenset({dossiers[0].pair_id}))
    with pytest.raises(LLMCallError):
        failing.review(dossiers[0])


def test_duplicate_pair_ids_rejected(data):
    df, truth = data
    dup = pd.concat([df, df.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="unique"):
        SelfCalibratingSession(dup, OracleReviewer(truth), legacy_cfg())


# --------------------------------------------------------------------------
# Review fixes: holdout snapshot, weights, dossier, mask, validation
# --------------------------------------------------------------------------


def _step_until_gate(session: SelfCalibratingSession) -> list[SessionEvent]:
    out: list[SessionEvent] = []
    while not session.awaiting_human and not session.finished:
        out.append(session.step())
    return out


def test_holdout_is_the_snapshot_of_the_last_holdout_built(data):
    df, truth = data
    session = SelfCalibratingSession(
        df,
        OracleReviewer(truth),
        legacy_cfg(mode="human", max_rounds=2, min_new_positives=0),
    )
    _step_until_gate(session)
    assert session.awaiting_human
    # open gate: reviews exist but no holdout has been built yet
    assert len(session.holdout()) == 0
    session.submit_human_decisions(truth_callback(truth)(session.gate_items))
    built = None
    while built is None:
        event = session.step()
        if event.kind == "holdout_built":
            built = event
    snapshot = session.holdout().copy()
    assert len(snapshot) == built.payload["n_holdout"] > 0
    assert snapshot["weight"].mean() == pytest.approx(1.0)
    # the snapshot survives a state round trip and is not recomputed afterwards
    restored = SelfCalibratingSession.from_state(
        session.to_state(), df, OracleReviewer(truth)
    )
    pd.testing.assert_frame_equal(restored.holdout(), snapshot)
    for _ in range(12):
        session.step()
    pd.testing.assert_frame_equal(session.holdout(), snapshot)


def test_holdout_built_reports_effective_size_cap_and_class_weights(data):
    df, truth = data
    session = SelfCalibratingSession(
        df, OracleReviewer(truth), legacy_cfg(max_rounds=1)
    )
    events = drive(session)
    built = next(e for e in events if e.kind == "holdout_built")
    holdout = session.holdout()
    w = holdout["weight"].to_numpy()
    assert built.payload["effective_n"] == pytest.approx(w.sum() ** 2 / (w**2).sum())
    assert 0 < built.payload["effective_n"] <= len(w) + 1e-9
    assert 0 <= built.payload["n_capped"] <= len(w)
    sums = built.payload["weight_sum_by_class"]
    assert sums["1"] == pytest.approx(w[holdout["y"] == 1.0].sum())
    assert sums["0"] == pytest.approx(w[holdout["y"] == 0.0].sum())
    assert sums["1"] + sums["0"] + sums["fractional"] == pytest.approx(w.sum())


def test_ratio_one_makes_every_weight_equal(data):
    df, truth = data
    session = SelfCalibratingSession(
        df, OracleReviewer(truth), legacy_cfg(max_rounds=1, weight_max_ratio=1.0)
    )
    events = drive(session)
    built = next(e for e in events if e.kind == "holdout_built")
    w = session.holdout()["weight"].to_numpy()
    np.testing.assert_allclose(w, 1.0)
    assert built.payload["effective_n"] == pytest.approx(len(w))


class _CapturingReviewer:
    def __init__(self, inner: OracleReviewer) -> None:
        self._inner = inner
        self.seen: dict[str, str] = {}

    def review(self, dossier: DossierView):
        self.seen[dossier.pair_id] = dossier.to_json()
        return self._inner.review(dossier)


def test_dossier_is_built_from_the_complete_row(data):
    df, truth = data
    rng = np.random.default_rng(3)
    extra = df.copy()
    for column in _NAME_COLS:
        extra[column] = np.round(rng.random(len(extra)), 4)
    # a column the dossier resolves only through its ",Int32" suffix variant
    renamed = _NAME_COLS[0]
    extra = extra.rename(columns={renamed: f"{renamed},Int32"})
    config = legacy_cfg(max_rounds=1)
    reviewer = _CapturingReviewer(OracleReviewer(truth))
    session = SelfCalibratingSession(extra, reviewer, config)
    for _ in range(40):
        session.step()
    index = _row_index(extra)
    assert reviewer.seen
    for pid, got in reviewer.seen.items():
        expected = DossierView(
            build_dossier(extra.iloc[index[pid]], pair_id=pid),
            remove_model_outputs=True,
            remove_fields=config.remove_fields,
        ).to_json()
        assert got == expected
    sample = json.loads(next(iter(reviewer.seen.values())))
    assert renamed in sample["features"]["name"]


def test_warns_when_the_score_stays_visible_to_the_reviewer(data):
    df, truth = data
    with pytest.warns(UserWarning, match="nota_final"):
        SelfCalibratingSession(df, OracleReviewer(truth), legacy_cfg(remove_fields=()))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SelfCalibratingSession(df, OracleReviewer(truth), legacy_cfg())
    assert not any("nota_final" in str(w.message) for w in caught)


def test_window_must_allow_the_posterior_stop():
    needed = zero_match_reviews_needed(0.05, 0.05)
    SelfCalibratingConfig(window=needed)
    with pytest.raises(ValueError, match="window"):
        SelfCalibratingConfig(window=needed - 1)
    needed_loose = zero_match_reviews_needed(0.2, 0.1)
    with pytest.raises(ValueError, match="window"):
        SelfCalibratingConfig(tau=0.2, delta=0.1, window=needed_loose - 1)
    SelfCalibratingConfig(tau=0.2, delta=0.1, window=needed_loose)


def test_low_confidence_is_decided_row_by_row(data):
    df, truth = data
    mixed = df.copy()
    mixed.loc[::2, "SCORE"] = np.nan
    mixed.loc[1::4, "PASSO"] = 7  # outside the thesis cutoffs
    first = SelfCalibratingSession(mixed, OracleReviewer(truth), legacy_cfg()).step()
    eligible = mixed["PASSO"].isin(list(THESIS_CUTOFFS)) & mixed["SCORE"].notna()
    assert 0 < eligible.sum() < len(mixed)
    cut = np.asarray(below_cutoff(mixed.loc[eligible, ["PASSO", "SCORE"]]), dtype=bool)
    nota = mixed["nota final"]
    grey = ((nota >= 5) & (nota < 8))[~eligible]
    assert first.payload["low_confidence_source"] == "mixed"
    assert first.payload["low_confidence_by_source"] == {
        "below_cutoff": int(cut.sum()),
        "grey_bands_nota": int(grey.sum()),
    }
    assert first.payload["n_low_confidence"] == int(cut.sum() + grey.sum())
    assert first.payload["low_confidence_note"]


def test_low_confidence_counts_by_source_for_pure_sources(data):
    df, truth = data
    first = SelfCalibratingSession(df, OracleReviewer(truth), legacy_cfg()).step()
    counts = first.payload["low_confidence_by_source"]
    assert first.payload["low_confidence_source"] == "below_cutoff"
    assert counts["grey_bands_nota"] == 0
    assert counts["below_cutoff"] == first.payload["n_low_confidence"]


@pytest.mark.parametrize(
    "bad",
    [
        np.array(["x"] * N_PAIRS),
        np.arange(N_PAIRS) % 2,
        np.array([True, None] * (N_PAIRS // 2), dtype=object),
    ],
)
def test_low_confidence_argument_must_be_bool(data, bad):
    df, truth = data
    with pytest.raises(ValueError, match="bool"):
        SelfCalibratingSession(
            df, OracleReviewer(truth), legacy_cfg(), low_confidence=bad
        )


@pytest.mark.parametrize("values", ["strings", "objects", "floats"])
def test_low_confidence_column_must_be_bool(data, values):
    df, truth = data
    column = {
        "strings": np.array(["yes", "no"] * (N_PAIRS // 2)),
        "objects": np.array([True, None] * (N_PAIRS // 2), dtype=object),
        "floats": np.arange(N_PAIRS) % 2 * 1.0,
    }[values]
    with pytest.raises(ValueError, match="bool"):
        SelfCalibratingSession(
            df.assign(auto_low_confidence=column), OracleReviewer(truth), legacy_cfg()
        )


def test_non_finite_score_is_rejected_at_start(data):
    df, truth = data
    for bad in (np.inf, -np.inf):
        broken = df.copy()
        broken.loc[3, "nota final"] = bad
        with pytest.raises(ValueError, match="non-finite"):
            SelfCalibratingSession(broken, OracleReviewer(truth), legacy_cfg())


def test_from_state_checks_the_fingerprint_before_indexing(data):
    df, truth = data
    mask = np.zeros(len(df), dtype=bool)
    mask[-1] = True  # the saved index would be out of range for a shorter frame
    session = SelfCalibratingSession(
        df, OracleReviewer(truth), legacy_cfg(), low_confidence=mask
    )
    state = session.to_state()
    assert state["phase"] == "init"
    with pytest.raises(ValueError, match="does not match"):
        SelfCalibratingSession.from_state(state, df.iloc[:100], OracleReviewer(truth))
    with pytest.raises(ValueError, match="does not match"):
        SelfCalibratingSession.from_state(
            state, df.drop(columns=["SCORE"]), OracleReviewer(truth)
        )


def test_posterior_stop_matches_beta_survival_on_window_counts(data):
    df, truth = data
    window = 60
    session = SelfCalibratingSession(
        df,
        OracleReviewer(truth),
        legacy_cfg(window=window, max_rounds=1, max_round_reviews=400, max_reviews=400),
    )
    events = drive(session)
    tau, delta = session.config.tau, session.config.delta
    history: dict[int, list[int]] = {}
    checked = 0
    for event in events:
        if event.kind in ("final_verdict", "review_error"):
            if event.payload["arm"] == "explore":
                continue  # outside the bandit: no reward, no trial
            history.setdefault(int(event.payload["arm"]), []).append(
                int(event.payload["reward"])
            )
            for arm in event.payload["stop_statistic"]["arms"]:
                recent = history.get(arm["arm"], [])[-window:]
                if arm["exhausted"]:
                    assert arm["p_gt_tau"] is None
                    continue
                assert arm["trials"] == len(recent)
                assert arm["successes"] == sum(recent)
                expected = beta.sf(tau, 1 + sum(recent), 1 + len(recent) - sum(recent))
                assert arm["p_gt_tau"] == pytest.approx(expected, rel=1e-9)
                checked += 1
    assert checked > 50
    stopped = next(e for e in events if e.kind == "round_stopped")
    stat = stopped.payload["stop_statistic"]
    live = [a for a in stat["arms"] if not a["exhausted"]]
    best = stat["max_p_gt_tau"]
    if stopped.payload["reason"] == "posterior":
        # an arm without trials has P(theta > tau) = 1 - tau >= delta: never stops
        assert live and all(a["trials"] > 0 and a["p_gt_tau"] < delta for a in live)
        assert best < delta
    else:
        # any other reason means the posterior rule had not fired (or no arm is left)
        assert best is None or best >= delta
        assert (best is None) == (not live)
    assert stop_probability(0, 0, tau) == pytest.approx(1 - tau)
    assert stop_probability(0, 0, tau) >= delta


def test_failed_fellegi_sunter_falls_back_to_the_prior_model(data, monkeypatch):
    df, truth = data

    def boom(*args, **kwargs):
        raise FellegiSunterError("forced failure")

    monkeypatch.setattr(scq, "fit_fellegi_sunter", boom)
    session = SelfCalibratingSession(df, OracleReviewer(truth), legacy_cfg())
    first = session.step()
    assert first.payload["fellegi_sunter"]["status"] == "failed"
    model = session.current_model
    assert model is not None and model.status == "prior"
    assert all(c == 0.0 for c in model.coef)
    m = min(max(first.payload["p0_mean"], 1e-6), 1 - 1e-6)
    assert model.intercept == pytest.approx(np.log(m / (1 - m)))


def test_p0_falls_back_to_the_stub_without_anchors(data, monkeypatch):
    df, truth = data

    def no_anchors(*args, **kwargs):
        raise _InsufficientAnchors("no anchors")

    monkeypatch.setattr(scq, "fit_anchor_platt", no_anchors)
    first = SelfCalibratingSession(df, OracleReviewer(truth), legacy_cfg()).step()
    assert first.payload["p0_method"] == "stub"
    assert "no anchors" in first.payload["p0_fallback_reason"]
    assert first.payload["p0_mean"] == pytest.approx(
        float(predict_stub(df["nota final"]).mean())
    )


# --------------------------------------------------------------------------
# Dominant low-confidence region guard (automatic mode, policy "exclude")
# --------------------------------------------------------------------------


def _dominant_cfg(**kwargs: Any) -> SelfCalibratingConfig:
    params: dict[str, Any] = {"auto_max_low_conf_fraction": 0.5, "max_rounds": 2}
    params.update(kwargs)
    return legacy_cfg(**params)


def _dominant_mask(df: pd.DataFrame) -> np.ndarray:
    mask = np.zeros(len(df), dtype=bool)
    mask[np.arange(len(df)) % 10 < 7] = True  # 70% of the pairs
    return mask


def test_dominant_region_disables_learning_for_the_whole_run(data):
    df, truth = data
    session = SelfCalibratingSession(
        df,
        OracleReviewer(truth),
        _dominant_cfg(),
        low_confidence=_dominant_mask(df),
    )
    events = drive(session)
    assert GRAMMAR.match(grammar_string(events))
    ordering = events[0]
    assert ordering.payload["low_conf_fraction"] == pytest.approx(0.7)
    assert ordering.payload["auto_max_low_conf_fraction"] == 0.5
    assert ordering.payload["auto_learning_enabled"] is False

    recals = [e for e in events if e.kind == "recalibration"]
    assert len(recals) == 2
    for event in recals:
        assert event.payload["status"] == "skipped_low_conf_dominant"
        assert event.payload["model_status"] == "prior"
        assert event.payload["low_conf_fraction"] == pytest.approx(0.7)
        assert event.payload["auto_max_low_conf_fraction"] == 0.5

    alarms = [
        e
        for e in events
        if e.kind == "alarm" and e.payload.get("reason") == "low_confidence_dominant"
    ]
    assert len(alarms) == 1
    assert alarms[0].round == 1
    first_round = [e.kind for e in events if e.round == 1]
    at = first_round.index("alarm")
    assert first_round[at - 1] == "holdout_built"
    assert first_round[at + 1] == "recalibration"

    # the holdout is still built and emitted every round
    assert len([e for e in events if e.kind == "holdout_built"]) == 2

    # round 2 is ordered by the Fellegi-Sunter prior, not by the score
    X, prior_model = fs_prior_model(df, session.config)
    expected = prior_model.predict_proba(X)
    index = _row_index(df)
    second = [e for e in events if e.kind == "pair_presented" and e.round == 2]
    assert second
    for e in second:
        assert e.payload["p_cur"] == pytest.approx(
            expected[index[e.payload["pair_id"]]], abs=1e-9
        )
    assert session.current_model is not None
    assert session.current_model.status == "prior"
    assert session.summary()["auto_learning_enabled"] is False


def test_default_mask_dominating_the_region_also_disables_learning(data):
    df, truth = data
    thesis_like = df.assign(SCORE=0.0)  # everything below the thesis cutoffs
    session = SelfCalibratingSession(
        thesis_like, OracleReviewer(truth), _dominant_cfg()
    )
    events = drive(session)
    ordering = events[0]
    assert ordering.payload["low_confidence_source"] == "below_cutoff"
    assert ordering.payload["low_conf_fraction"] > 0.5
    assert ordering.payload["auto_learning_enabled"] is False
    assert all(
        e.payload["status"] == "skipped_low_conf_dominant"
        for e in events
        if e.kind == "recalibration"
    )


def test_soft_policy_is_not_guarded(data):
    df, truth = data
    session = SelfCalibratingSession(
        df,
        OracleReviewer(truth),
        _dominant_cfg(auto_low_conf_policy="soft"),
        low_confidence=_dominant_mask(df),
    )
    events = drive(session)
    assert events[0].payload["auto_learning_enabled"] is True
    statuses = {e.payload["status"] for e in events if e.kind == "recalibration"}
    assert statuses
    assert statuses <= {"fitted", "skipped_insufficient"}
    assert not any(
        e.kind == "alarm" and e.payload.get("reason") == "low_confidence_dominant"
        for e in events
    )


def test_human_mode_is_not_guarded(data):
    df, truth = data
    session = SelfCalibratingSession(
        df,
        OracleReviewer(truth),
        legacy_cfg(mode="human", auto_max_low_conf_fraction=0.5, max_rounds=1),
        low_confidence=_dominant_mask(df),
    )
    events = drive(session, truth_callback(truth))
    assert events[0].payload["auto_learning_enabled"] is True
    assert not any(
        e.kind == "recalibration" and e.payload["status"] == "skipped_low_conf_dominant"
        for e in events
    )


def test_limit_one_keeps_the_previous_behaviour(data):
    df, truth = data
    session = SelfCalibratingSession(
        df,
        OracleReviewer(truth),
        _dominant_cfg(auto_max_low_conf_fraction=1.0),
        low_confidence=_dominant_mask(df),
    )
    events = drive(session)
    assert events[0].payload["auto_learning_enabled"] is True
    assert not any(
        e.kind == "alarm" and e.payload.get("reason") == "low_confidence_dominant"
        for e in events
    )
    assert all(
        e.payload["status"] != "skipped_low_conf_dominant"
        for e in events
        if e.kind == "recalibration"
    )
    assert GRAMMAR.match(grammar_string(events))


@pytest.mark.parametrize("stop_after", ["alarm", "recalibration", "reordering"])
def test_resume_in_the_dominant_scenario_is_identical(data, stop_after):
    df, truth = data
    mask = _dominant_mask(df)
    full = as_dicts(
        drive(
            SelfCalibratingSession(
                df, OracleReviewer(truth), _dominant_cfg(), low_confidence=mask
            )
        )
    )
    cut = next(i for i, e in enumerate(full) if e["kind"] == stop_after) + 1
    session = SelfCalibratingSession(
        df, OracleReviewer(truth), _dominant_cfg(), low_confidence=mask
    )
    head = [session.step().to_dict() for _ in range(cut)]
    state = json.loads(json.dumps(session.to_state()))
    restored = SelfCalibratingSession.from_state(state, df, OracleReviewer(truth))
    tail = as_dicts(drive(restored))
    assert head + tail == full


def test_dominant_fraction_validation_and_round_trip():
    assert SelfCalibratingConfig().auto_max_low_conf_fraction == 0.5
    for bad in (0.0, -0.1, 1.5, float("nan")):
        with pytest.raises(ValueError, match="auto_max_low_conf_fraction"):
            SelfCalibratingConfig(auto_max_low_conf_fraction=bad)
    config = SelfCalibratingConfig(mode="auto", auto_max_low_conf_fraction=0.25)
    again = SelfCalibratingConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    assert again == config and again.auto_max_low_conf_fraction == 0.25
    assert SelfCalibratingConfig(auto_max_low_conf_fraction=1.0)


# --------------------------------------------------------------------------
# 12. Field sets (CdR sub-scores by default, composites on request)
# --------------------------------------------------------------------------


def cdr_cfg(**kwargs: Any) -> SelfCalibratingConfig:
    return legacy_cfg(field_set="cdr_subscores", **kwargs)


def test_cdr_subscores_session_uses_the_subscore_design(data):
    df, truth = data
    session = SelfCalibratingSession(df, OracleReviewer(truth), cdr_cfg())
    first = session.step()
    names = first.payload["feature_names"]
    expected = design_matrix(discretize_fields(df, "cdr_subscores"))[1]
    assert first.payload["field_set"] == "cdr_subscores"
    assert tuple(names) == expected
    assert any(n.startswith("NOME prim frag igual:") for n in names)
    assert any(n == "mae_missing:missing" for n in names)
    assert not any(n.endswith(":L0") for n in names)
    assert len(names) > 29
    assert not any("nota" in n for n in names)
    assert first.payload["fellegi_sunter"]["status"] == "fitted"
    assert session.feature_names == expected
    # the composites run keeps its own (smaller) design
    comp = SelfCalibratingSession(df, OracleReviewer(truth), legacy_cfg()).step()
    assert comp.payload["field_set"] == "composites"
    assert (
        tuple(comp.payload["feature_names"]) == design_matrix(discretize_fields(df))[1]
    )
    assert len(comp.payload["feature_names"]) < len(names)


def test_cdr_subscores_full_run_and_resume(data):
    df, truth = data
    config = cdr_cfg(min_labeled=10)
    reference = as_dicts(
        drive(SelfCalibratingSession(df, OracleReviewer(truth), config))
    )
    assert GRAMMAR.match("".join(LETTER[e["kind"]] for e in reference))
    recal = next(e for e in reference if e["kind"] == "recalibration")
    assert recal["payload"]["status"] in {"fitted", "skipped_insufficient"}
    json.dumps(reference)
    for k in (1, 5, len(reference) // 2, len(reference) - 2):
        session = SelfCalibratingSession(df, OracleReviewer(truth), config)
        for _ in range(k):
            session.step()
        restored = _restore(session, df, OracleReviewer(truth))
        assert as_dicts(drive(restored)) == reference[k:], f"cut at {k}"


def test_cdr_subscores_fingerprint_and_suffix_resolution(data):
    df, truth = data
    session = SelfCalibratingSession(df, OracleReviewer(truth), cdr_cfg())
    session.step()
    state = session.to_state()
    changed = df.copy()
    changed.loc[0, "NOME prim frag igual"] = 0.5
    with pytest.raises(ValueError, match="does not match"):
        SelfCalibratingSession.from_state(state, changed, OracleReviewer(truth))
    # PAR is never read: changing it is fine
    other = df.copy()
    other.loc[0, "PAR"] = 1 - int(other.loc[0, "PAR"])
    SelfCalibratingSession.from_state(state, other, OracleReviewer(truth))
    # ",Type" suffix on a sub-score column is resolved like the dossier does
    renamed = df.rename(columns={"NOME ult frag igual": "NOME ult frag igual,Int32"})
    first = SelfCalibratingSession(renamed, OracleReviewer(truth), cdr_cfg()).step()
    base = SelfCalibratingSession(df, OracleReviewer(truth), cdr_cfg()).step()
    assert first.payload["feature_names"] == base.payload["feature_names"]
    missing = df.drop(columns=["NOME ult frag igual"])
    with pytest.raises(KeyError, match="NOME ult frag igual"):
        SelfCalibratingSession(missing, OracleReviewer(truth), cdr_cfg())


def test_composites_field_set_ignores_the_subscores(data):
    df, truth = data
    config = legacy_cfg()
    base = json.dumps(
        as_dicts(drive(SelfCalibratingSession(df, OracleReviewer(truth), config)))
    )
    only_composites = df.drop(columns=list(CDR_SUBSCORE_COLUMNS))
    again = json.dumps(
        as_dicts(
            drive(
                SelfCalibratingSession(only_composites, OracleReviewer(truth), config)
            )
        )
    )
    assert base == again


# --------------------------------------------------------------------------
# 13. Fingerprint, SCORE, remove_fields, config hygiene
# --------------------------------------------------------------------------


def test_fingerprint_covers_what_the_dossier_reads(data):
    df, truth = data
    for field_set in ("composites", "cdr_subscores"):
        session = SelfCalibratingSession(
            df, OracleReviewer(truth), legacy_cfg(field_set=field_set)
        )
        session.step()
        state = session.to_state()
        # composites and flags feed the LLM dossier: a change must be rejected
        for column in ("nome_score_total", "endereco_zero", "NOME prim frag igual"):
            changed = df.copy().astype({column: float})
            changed.loc[0, column] = 0.777
            with pytest.raises(ValueError, match="does not match"):
                SelfCalibratingSession.from_state(state, changed, OracleReviewer(truth))
    # a column neither the field set nor the dossier reads is not fingerprinted
    session = SelfCalibratingSession(df, OracleReviewer(truth), legacy_cfg())
    session.step()
    state = session.to_state()
    unread = df.copy()
    unread.loc[0, "CODMUNRES uf igual"] = 0.31  # cdr field, not a dossier column
    SelfCalibratingSession.from_state(state, unread, OracleReviewer(truth))
    reading = SelfCalibratingSession(df, OracleReviewer(truth), cdr_cfg())
    reading.step()
    with pytest.raises(ValueError, match="does not match"):
        SelfCalibratingSession.from_state(
            reading.to_state(), unread, OracleReviewer(truth)
        )
    # the suffix variant of a dossier column is fingerprinted by its resolved name
    renamed = df.rename(columns={"NOME qtd frag iguais": "NOME qtd frag iguais,Float"})
    assert "NOME qtd frag iguais,Float" in scq._declared_columns(renamed, "composites")


def _score_frame(df: pd.DataFrame) -> pd.DataFrame:
    """SCORE with the loader header and decimal comma, as in the ORL extract."""
    out = df.copy()
    out["SCORE,C,15,0"] = out["SCORE"].map(lambda x: f"{x:.4f}".replace(".", ","))
    return out.drop(columns=["SCORE"])


def test_score_column_with_type_suffix_and_decimal_comma(data):
    df, truth = data
    plain = SelfCalibratingSession(df, OracleReviewer(truth), legacy_cfg()).step()
    comma = _score_frame(df)
    assert comma["SCORE,C,15,0"].iloc[0].count(",") == 1
    session = SelfCalibratingSession(comma, OracleReviewer(truth), legacy_cfg())
    first = session.step()
    assert first.payload["low_confidence_source"] == "below_cutoff"
    assert first.payload["low_confidence_by_source"] == {
        "below_cutoff": plain.payload["n_low_confidence"],
        "grey_bands_nota": 0,
    }
    assert first.payload["n_low_confidence"] == plain.payload["n_low_confidence"] > 0
    assert "SCORE,C,15,0" in scq._declared_columns(comma, "composites")
    # the resolved column is fingerprinted
    state = session.to_state()
    changed = comma.copy()
    changed.loc[0, "SCORE,C,15,0"] = "0,0001"
    with pytest.raises(ValueError, match="does not match"):
        SelfCalibratingSession.from_state(state, changed, OracleReviewer(truth))
    SelfCalibratingSession.from_state(state, comma, OracleReviewer(truth))
    # a literal "7,02" is read as 7.02
    one = comma.iloc[:5].copy()
    one["PASSO"] = 6
    one["SCORE,C,15,0"] = ["7,02", "0,50", "9,99", "1,00", "8,00"]
    low = SelfCalibratingSession(one, OracleReviewer(truth), legacy_cfg())
    mask_event = low.step()
    cut = THESIS_CUTOFFS[6]
    expected = int(sum(float(v.replace(",", ".")) < cut for v in one["SCORE,C,15,0"]))
    assert mask_event.payload["n_low_confidence"] == expected


def test_remove_fields_are_validated_when_the_session_is_created(data):
    df, truth = data
    reviewer = OracleReviewer(truth)
    with pytest.raises(ValueError, match="not a string"):
        SelfCalibratingConfig(remove_fields="features.nota_final")
    with pytest.raises(ValueError, match="duplicate"):
        SelfCalibratingConfig(remove_fields=("features.a", "features.a"))
    for bad in (
        ("model_outputs.p_cal",),  # already removed by the session
        ("features.does_not_exist",),
        ("features.nota_final", "features.nope.deeper"),
        ("features..nota_final",),
    ):
        with pytest.raises(ValueError, match="Invalid remove_fields"):
            SelfCalibratingSession(df, reviewer, legacy_cfg(remove_fields=bad))
    assert reviewer.seen_payloads == []  # failed before any review
    ok = SelfCalibratingSession(
        df,
        reviewer,
        legacy_cfg(remove_fields=("features.nota_final", "features.composites")),
    )
    for _ in range(3):
        ok.step()
    assert reviewer.seen_payloads
    assert all("composites" not in p["features"] for p in reviewer.seen_payloads)


def test_zero_match_reviews_needed_is_exact():
    assert zero_match_reviews_needed(0.05, 0.05) == 58
    assert zero_match_reviews_needed(0.5, 0.25) == 2
    assert zero_match_reviews_needed(0.2, 0.1) == 10
    for tau in (0.01, 0.05, 0.1, 0.25, 0.5, 0.9):
        for delta in (0.01, 0.05, 0.1, 0.25, 0.5, 0.9):
            n = zero_match_reviews_needed(tau, delta)
            assert (1 - tau) ** (n + 1) < delta, (tau, delta)
            assert n == 0 or (1 - tau) ** n >= delta, (tau, delta)
            assert stop_probability(0, n, tau) < delta


def test_config_accepts_numpy_scalars_and_stays_json():
    config = SelfCalibratingConfig(
        seed=np.int64(5),
        window=np.int64(100),
        explore_every=np.int64(10),
        human_budget_total=np.int64(20),
        human_budget_per_round=np.int32(5),
        tau=np.float64(0.05),
        gate_include_unsure=np.bool_(True),
    )
    assert type(config.seed) is int and type(config.window) is int
    assert type(config.human_budget_total) is int and config.human_budget_total == 20
    assert type(config.tau) is float and config.gate_include_unsure is True
    again = SelfCalibratingConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    assert again == config
    assert config == SelfCalibratingConfig(
        seed=5, human_budget_total=20, human_budget_per_round=5
    )


# --------------------------------------------------------------------------
# 14. The real default configuration, end to end
# --------------------------------------------------------------------------


def _default_session(df, truth):
    reviewer = OracleReviewer(truth, fp_rate=0.0, fn_rate=0.3, unsure_rate=0.02, seed=3)
    return SelfCalibratingSession(df, reviewer, SelfCalibratingConfig())


def test_default_configuration_end_to_end_with_resume(data):
    df, truth = data
    config = SelfCalibratingConfig()  # human, composites, guard 0.5, budget 50
    assert (config.mode, config.field_set) == ("human", "composites")
    assert (config.human_budget_per_round, config.auto_max_low_conf_fraction) == (
        50,
        0.5,
    )
    callback = truth_callback(truth)
    reference = as_dicts(drive(_default_session(df, truth), callback))
    letters = "".join(LETTER[e["kind"]] for e in reference)
    assert GRAMMAR.match(letters), letters[:120]
    assert reference[-1]["payload"]["reason"] in {
        "max_rounds",
        "sterile_round",
        "max_reviews",
        "exhausted",
    }
    rounds = max(e["round"] for e in reference)
    assert rounds >= 2
    gates = [e for e in reference if e["kind"] == "gate_opened"]
    assert all(g["payload"]["n_gate"] <= 50 for g in gates)
    assert any(g["payload"]["n_audit"] > 0 for g in gates), "need an audit in the gate"
    assert any(g["payload"]["n_over_budget"] > 0 for g in gates)
    assert any(
        e["kind"] == "pair_presented" and e["payload"]["arm"] == "explore"
        for e in reference
    )
    json.dumps(reference)

    def first(kind: str, after: int = 0, **match: Any) -> int:
        return next(
            i
            for i, e in enumerate(reference)
            if i >= after
            and e["kind"] == kind
            and all(e["payload"].get(k) == v for k, v in match.items())
        )

    gate_at = first("gate_opened")
    reorder = first("reordering")
    cuts = {
        1,
        first("pair_presented") + 1,
        first("pair_presented", arm="explore") + 1,
        first("pair_presented", arm="explore") + 2,
        gate_at + 1,  # awaiting the human, audit items in the gate
        first("human_decision") + 1,
        first("holdout_built") + 1,
        first("recalibration") + 1,
        reorder + 1,
        reorder + 4,
        first("pair_presented", after=reorder, arm="explore") + 1,
        first("gate_opened", after=reorder) + 1,  # round >= 2 gate
        len(reference) - 2,
    }
    for k in sorted(cuts):
        session = _default_session(df, truth)
        taken: list[SessionEvent] = []
        while len(taken) < k:
            if session.awaiting_human:
                session.submit_human_decisions(callback(session.gate_items))
            else:
                taken.append(session.step())
        restored = _restore(
            session,
            df,
            OracleReviewer(truth, fp_rate=0.0, fn_rate=0.3, unsure_rate=0.02, seed=3),
        )
        assert restored.awaiting_human == session.awaiting_human
        tail = as_dicts(drive(restored, callback))
        assert as_dicts(taken) + tail == reference, f"cut at {k}"


# --------------------------------------------------------------------------
# 15. Why a round stops and why a run ends
# --------------------------------------------------------------------------


def _reasons(events: list[SessionEvent]) -> tuple[list[str], str]:
    stops = [e.payload["reason"] for e in events if e.kind == "round_stopped"]
    return stops, events[-1].payload["reason"]


def test_round_stop_max_round_reviews_and_run_end_max_rounds(data):
    df, truth = data
    session = SelfCalibratingSession(
        df,
        OracleReviewer({pid: True for pid in truth}),
        legacy_cfg(max_round_reviews=30, max_rounds=2, anomaly_min_matches=10_000),
    )
    stops, end = _reasons(drive(session))
    assert stops == ["max_round_reviews", "max_round_reviews"]
    assert end == "max_rounds"
    assert session.summary()["n_reviewed"] == 60


def test_round_stop_and_run_end_max_reviews(data):
    df, truth = data
    session = SelfCalibratingSession(
        df,
        OracleReviewer({pid: True for pid in truth}),
        legacy_cfg(max_reviews=40, max_round_reviews=500, anomaly_min_matches=10_000),
    )
    stops, end = _reasons(drive(session))
    assert stops == ["max_reviews"] and end == "max_reviews"
    assert session.summary()["n_reviewed"] == 40


def test_round_stop_and_run_end_exhausted():
    df, truth = make_data(n=120, seed=21)
    session = SelfCalibratingSession(
        df,
        OracleReviewer({pid: True for pid in truth}),
        legacy_cfg(max_round_reviews=500, max_reviews=500, anomaly_min_matches=10_000),
    )
    stops, end = _reasons(drive(session))
    assert stops == ["exhausted"] and end == "exhausted"
    assert session.summary()["n_reviewed"] == 120
    assert session.results()["pair_id"].is_unique


def test_run_end_sterile_round(data):
    df, truth = data
    session = SelfCalibratingSession(
        df,
        OracleReviewer({pid: False for pid in truth}),
        legacy_cfg(max_rounds=3, min_new_positives=1, max_round_reviews=500),
    )
    stops, end = _reasons(drive(session))
    assert stops == ["posterior"]
    assert end == "sterile_round"
    assert session.summary()["round"] == 1


def test_run_end_llm_errors_and_round_stop(data):
    df, truth = data
    session = SelfCalibratingSession(
        df,
        OracleReviewer(truth, error_pairs=frozenset(truth)),
        legacy_cfg(max_rounds=3),
    )
    stops, end = _reasons(drive(session))
    assert stops == ["llm_errors"] and end == "llm_errors"
