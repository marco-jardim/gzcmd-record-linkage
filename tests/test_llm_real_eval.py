"""Testes offline do harness de chamadas reais (cache, paralelismo, teto, chave)."""

from __future__ import annotations

import json
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import llm_real_eval as lre
import numpy as np
import pandas as pd
import pytest
from test_self_calibrating_peek import make_data

from gzcmd_record_linkage import llm_review
from gzcmd_record_linkage.llm_dossier import DossierView, build_dossier
from gzcmd_record_linkage.llm_review import AgentResponse, LLMCallError, ReviewResult
from gzcmd_record_linkage.self_calibrating_queue import OracleReviewer

SECRET = "sk-ant-SEGREDO-NAO-IMPRIMIR-123"


def row(i: int) -> pd.Series:
    return pd.Series(
        {
            "COMPREC": i,
            "REFREC": 1000 + i,
            "PASSO": 3,
            "nota final": 7.5,
            "NOME qtd frag iguais": 0.5,
            "DTNASC dt iguais": 1,
        }
    )


def views(n: int) -> list[DossierView]:
    return [lre.make_view(row(i)) for i in range(n)]


def agent(decision: str = "MATCH") -> AgentResponse:
    return AgentResponse(
        decision=decision,  # type: ignore[arg-type]
        confidence=0.8,
        reason_codes=["NAME_STRONG"],
        evidence_summary={
            "supports_match": [],
            "supports_nonmatch": [],
            "tie_breakers": [],
        },
        quality_flags={
            "pii_leak_detected": False,
            "insufficient_evidence": False,
            "inconsistent_input": False,
        },
        raw_json={"decision": decision},
        latency_s=0.1,
        tokens_prompt=10,
        tokens_completion=5,
    )


class FakeReviewer:
    """Revisor falso: conta chamadas e pode falhar por ``pair_id``."""

    instances: list[FakeReviewer] = []
    calls = 0
    lock = threading.Lock()
    fail: set[str] = set()

    def __init__(self) -> None:
        self.usage_log: list[dict[str, Any]] = []
        self.thread = threading.get_ident()
        with FakeReviewer.lock:
            FakeReviewer.instances.append(self)

    def review(self, dossier: DossierView) -> ReviewResult:
        with FakeReviewer.lock:
            FakeReviewer.calls += 1
        if dossier.pair_id in FakeReviewer.fail:
            raise LLMCallError("falha", model="m", pair_id=dossier.pair_id, attempts=1)
        a = agent("MATCH")
        self.usage_log.extend(
            [
                {"input_tokens": 10, "output_tokens": 5},
                {"input_tokens": 10, "output_tokens": 5},
            ]
        )
        return ReviewResult(
            pair_id=dossier.pair_id,
            decision="MATCH",
            confidence=0.8,
            reason_codes=a.reason_codes,
            evidence_summary=a.evidence_summary,
            quality_flags=a.quality_flags,
            protocol="consensus",
            agent_a=a,
            agent_b=agent("MATCH"),
            total_latency_s=0.2,
        )


@pytest.fixture(autouse=True)
def reset_fake(monkeypatch):
    FakeReviewer.instances = []
    FakeReviewer.calls = 0
    FakeReviewer.fail = set()

    def no_network(*args, **kwargs):
        raise AssertionError("rede proibida nos testes")

    monkeypatch.setattr(llm_review.requests, "post", no_network)


def make(tmp_path: Path, **kw: Any) -> lre.CachedParallelReviewer:
    return lre.CachedParallelReviewer(
        cache_dir=tmp_path, reviewer_factory=FakeReviewer, **kw
    )


# --- dossiê -----------------------------------------------------------------


def test_dossier_has_no_nota_final_nor_model_outputs():
    for v in views(3):
        js = v.to_json()
        assert "nota_final" not in js
        assert "model_outputs" not in js
    full = DossierView(build_dossier(row(1)))
    assert "nota_final" in full.to_json()


def test_review_refuses_dossier_with_forbidden_field(tmp_path):
    reviewer = make(tmp_path)
    leaky = DossierView(build_dossier(row(1)))  # ainda tem features.nota_final
    with pytest.raises(ValueError, match="proibido"):
        reviewer.review(leaky)
    assert FakeReviewer.calls == 0


# --- chave de cache -----------------------------------------------------------


def test_cache_key_depends_on_content():
    base = ("m", "v4", ("features.nota_final",), '{"a": 1}')
    k = lre.cache_key(*base)
    assert k == lre.cache_key(*base)
    assert k != lre.cache_key("m2", "v4", base[2], base[3])
    assert k != lre.cache_key("m", "v3", base[2], base[3])
    assert k != lre.cache_key("m", "v4", (), base[3])
    assert k != lre.cache_key("m", "v4", base[2], '{"a": 2}')
    # a ordem dos campos removidos não importa
    assert lre.cache_key("m", "v4", ("b", "a"), "x") == lre.cache_key(
        "m", "v4", ("a", "b"), "x"
    )


def test_key_for_same_content_same_key_and_different_content_differs(tmp_path):
    reviewer = make(tmp_path)
    a, b = views(2)
    assert reviewer.key_for(a) == reviewer.key_for(lre.make_view(row(0)))
    assert reviewer.key_for(a) != reviewer.key_for(b)


# --- cache hit / miss ---------------------------------------------------------


def test_cache_hit_miss_and_persistence(tmp_path):
    reviewer = make(tmp_path)
    v = views(1)[0]
    first = reviewer.review(v)
    assert FakeReviewer.calls == 1 and reviewer.api_reviews == 1
    second = reviewer.review(v)
    assert FakeReviewer.calls == 1  # hit
    assert second == first
    assert reviewer.stats()["cache_hits"] == 1

    # nova instância relê o JSONL: sem chamadas e sem consumir o teto
    again = make(tmp_path, max_api_reviews=0)
    assert len(again) == 1
    assert again.review(v) == first
    assert again.api_reviews == 0

    lines = reviewer.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["summary"]["decision"] == "MATCH"
    assert rec["tokens_prompt"] == 20 and rec["tokens_completion"] == 10
    assert rec["api_calls"] == 2 and rec["latency_s"] == 0.2


def test_truncated_last_line_is_ignored(tmp_path):
    reviewer = make(tmp_path)
    reviewer.review(views(1)[0])
    with reviewer.path.open("a", encoding="utf-8") as fh:
        fh.write('{"key": "abc", "resu')
    assert len(make(tmp_path)) == 1


# --- paralelismo ---------------------------------------------------------------


def test_prefetch_is_thread_safe_and_one_reviewer_per_thread(tmp_path):
    reviewer = make(tmp_path)
    vs = views(40)
    out = reviewer.prefetch(vs, max_workers=8)
    assert out == {"requested": 40, "done": 40, "errors": 0}
    assert FakeReviewer.calls == 40
    assert 1 <= len(FakeReviewer.instances) <= 8
    assert len({r.thread for r in FakeReviewer.instances}) == len(
        FakeReviewer.instances
    )
    lines = reviewer.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 40
    assert len({json.loads(line)["key"] for line in lines}) == 40
    # segunda passada: tudo em cache, nenhuma chamada
    assert reviewer.prefetch(vs, max_workers=8)["requested"] == 0
    assert FakeReviewer.calls == 40


def test_prefetch_deduplicates_equal_dossiers(tmp_path):
    reviewer = make(tmp_path)
    v = views(1)[0]
    assert reviewer.prefetch([v, v, v], max_workers=3)["requested"] == 1
    assert FakeReviewer.calls == 1


# --- teto ------------------------------------------------------------------------


def test_budget_stops_cleanly_and_keeps_what_was_saved(tmp_path):
    reviewer = make(tmp_path, max_api_reviews=5)
    with pytest.raises(lre.ReviewBudgetExceeded):
        reviewer.prefetch(views(30), max_workers=4)
    assert reviewer.api_reviews == 5
    assert FakeReviewer.calls <= 5
    saved = len(reviewer.path.read_text(encoding="utf-8").splitlines())
    assert saved == FakeReviewer.calls == len(reviewer)
    # review() direto também respeita o teto
    with pytest.raises(lre.ReviewBudgetExceeded):
        reviewer.review(views(31)[30])
    # o que foi gravado continua disponível sem custo
    reopened = make(tmp_path, max_api_reviews=0)
    assert len(reopened) == saved


def test_errors_are_counted_not_cached(tmp_path):
    vs = views(6)
    FakeReviewer.fail = {vs[1].pair_id, vs[4].pair_id}
    reviewer = make(tmp_path)
    out = reviewer.prefetch(vs, max_workers=3)
    assert out["errors"] == 2 and out["done"] == 4
    assert len(reviewer) == 4
    assert reviewer.errors["LLMCallError"] == 2
    assert reviewer.record_for(vs[1]) is None
    with pytest.raises(LLMCallError):
        reviewer.review(vs[1])
    err_lines = reviewer.error_path.read_text(encoding="utf-8").splitlines()
    assert len(err_lines) == 3 and "LLMCallError" in err_lines[0]


def test_too_many_errors_abort_the_batch(tmp_path):
    vs = views(20)
    FakeReviewer.fail = {v.pair_id for v in vs}
    reviewer = make(tmp_path)
    with pytest.raises(lre.ReviewAbort):
        reviewer.prefetch(vs, max_workers=1, max_errors=2)
    assert FakeReviewer.calls < 20
    assert len(reviewer) == 0


# --- transporte falso, uso, 429 e chave -------------------------------------------


class FakeStatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__("status")
        self.status_code = status_code


class FakeConnError(Exception):
    pass


PAYLOAD = {
    "decision": "MATCH",
    "confidence": 0.9,
    "reason_codes": ["NAME_STRONG"],
    "evidence_summary": {
        "supports_match": ["nome"],
        "supports_nonmatch": [],
        "tie_breakers": [],
    },
    "quality_flags": {
        "pii_leak_detected": False,
        "insufficient_evidence": False,
        "inconsistent_input": False,
    },
}


@pytest.fixture
def fake_anthropic(monkeypatch):
    state = SimpleNamespace(calls=0, fail_first=0, keys=[])
    lock = threading.Lock()

    class Client:
        def __init__(self, **kwargs: Any) -> None:
            with lock:
                state.keys.append(kwargs.get("api_key"))
            self.messages = SimpleNamespace(create=self._create)

        def _create(self, **kwargs: Any) -> Any:
            with lock:
                state.calls += 1
                n = state.calls
            if n <= state.fail_first:
                raise FakeStatusError(429)
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text=json.dumps(PAYLOAD))],
                stop_reason="end_turn",
                usage=SimpleNamespace(
                    input_tokens=12,
                    output_tokens=34,
                    cache_creation_input_tokens=100,
                    cache_read_input_tokens=200,
                ),
            )

    module = types.ModuleType("anthropic")
    module.Anthropic = Client
    module.APIStatusError = FakeStatusError
    module.APIConnectionError = FakeConnError
    monkeypatch.setitem(sys.modules, "anthropic", module)
    return state


def test_real_reviewer_path_records_usage_and_429(tmp_path, fake_anthropic):
    fake_anthropic.fail_first = 1
    reviewer = lre.CachedParallelReviewer(
        cache_dir=tmp_path, api_key="chave-falsa", retry_delay_s=0.0
    )
    reviewer.review(views(1)[0])
    rec = next(iter(reviewer._cache.values()))
    assert rec["api_calls"] == 2  # A e B concordam; a chamada com 429 não conta
    assert rec["http_errors"] == {"429": 1}
    assert rec["usage"]["input_tokens"] == 24
    assert rec["usage"]["output_tokens"] == 68
    assert rec["usage"]["cache_creation_input_tokens"] == 200
    assert rec["usage"]["cache_read_input_tokens"] == 400
    assert reviewer.http_errors["429"] == 1
    assert rec["summary"]["protocol"] == "consensus"


def test_prompt_and_model_come_from_v4_haiku(tmp_path, fake_anthropic):
    reviewer = lre.CachedParallelReviewer(cache_dir=tmp_path, api_key="chave-falsa")
    rev = reviewer._default_factory()
    assert rev.prompt_version == "v4"
    assert rev.model == "claude-haiku-5-5"


def test_missing_key_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv(lre.KEY_VAR, raising=False)
    with pytest.raises(lre.MissingApiKeyError, match=lre.KEY_VAR):
        lre.load_api_key(tmp_path / "nao_existe.env")


def test_key_loaded_from_env_file_is_never_printed(
    tmp_path, monkeypatch, capsys, caplog, fake_anthropic
):
    monkeypatch.delenv(lre.KEY_VAR, raising=False)
    env = tmp_path / ".env"
    env.write_text(f"{lre.KEY_VAR}={SECRET}\n", encoding="utf-8")
    key = lre.load_api_key(env)
    assert key == SECRET
    monkeypatch.delenv(lre.KEY_VAR, raising=False)

    cache_dir = tmp_path / "cache"
    reviewer = lre.CachedParallelReviewer(cache_dir=cache_dir, api_key=key)
    assert SECRET not in repr(reviewer) and SECRET not in str(reviewer)
    reviewer.prefetch(views(3), max_workers=2)
    assert SECRET in fake_anthropic.keys  # foi usada na chamada...
    out = capsys.readouterr()
    assert SECRET not in out.out + out.err + caplog.text  # ...e não vazou
    for path in cache_dir.iterdir():
        assert SECRET not in path.read_text(encoding="utf-8")


# --- amostra e estatística ----------------------------------------------------------


def test_band_of_nota_edges():
    nota = np.array([0.0, 4.99, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    assert lre.band_of_nota(nota).tolist() == [0, 0, 1, 2, 3, 4, 5, 5]


def test_select_sample_takes_all_pairs_and_caps_non_pairs():
    rng = np.random.default_rng(0)
    n = 3000
    nota = rng.uniform(0, 10, n)
    band = lre.band_of_nota(nota)
    truth = rng.random(n) < 0.05
    valid = np.ones(n, dtype=bool)
    s1 = lre.select_sample(truth, band, valid, seed=7, per_band=20)
    s2 = lre.select_sample(truth, band, valid, seed=7, per_band=20)
    assert s1.tolist() == s2.tolist()
    assert len(set(s1.tolist())) == len(s1)
    assert set(np.flatnonzero(truth)) <= set(s1.tolist())
    for b in range(len(lre.BAND_NAMES)):
        assert int((~truth[s1] & (band[s1] == b)).sum()) == 20
    pilot = lre.select_pilot(s1, truth, band, 20)
    assert len(pilot) == 20 and set(pilot.tolist()) <= set(s1.tolist())
    assert truth[pilot].any() and (~truth[pilot]).any()


def test_wilson_interval():
    p, lo, hi = lre.wilson(4, 25)
    assert p == pytest.approx(0.16)
    assert lo == pytest.approx(0.064, abs=0.001)
    assert hi == pytest.approx(0.347, abs=0.001)
    assert np.isnan(lre.wilson(0, 0)[0])


def test_vpp_population_reweights_false_positives():
    cell = lre.Cell(pos=10, neg=10, neg_pop=1000, tp=8, fn=2, fp=1, tn=9)
    point, lo, hi = lre.vpp_population([cell])
    # 8 VP contra ~100 FP esperados no extrato (não 1 na amostra)
    assert 8 / (8 + 150) < point < 8 / (8 + 60)
    assert lo < point < hi
    assert lre.wilson(8, 9)[0] > point  # o VPP bruto da amostra superestima


# --- fixture e FixtureReviewer -------------------------------------------------

FREE_TEXT = "TEXTO-LIVRE-NAO-EXPORTAR"


class RichFakeReviewer:
    """Revisor falso com texto livre, árbitro em metade dos pares e UNSURE."""

    def __init__(self) -> None:
        self.usage_log: list[dict[str, Any]] = []

    def review(self, dossier: DossierView) -> ReviewResult:
        k = int(dossier.pair_id, 16)
        decision = ("MATCH", "NONMATCH", "UNSURE")[k % 3]

        def ag(d: str, conf: float) -> AgentResponse:
            return AgentResponse(
                decision=d,  # type: ignore[arg-type]
                confidence=conf,
                reason_codes=["NAME_STRONG", "DOB_MATCH"],
                evidence_summary={
                    "supports_match": [FREE_TEXT],
                    "supports_nonmatch": [],
                    "tie_breakers": [],
                },
                quality_flags={"pii_leak_detected": False},
                raw_json={"free": FREE_TEXT},
                latency_s=1.5,
                tokens_prompt=100 + k % 7,
                tokens_completion=200 + k % 5,
            )

        arbiter = k % 2 == 0
        other = "NONMATCH" if decision == "MATCH" else "MATCH"
        a = ag(decision, 0.7)
        b = ag(other if arbiter else decision, 0.6)
        arb = ag(decision, 0.9) if arbiter else None
        self.usage_log.extend([{"input_tokens": 1, "output_tokens": 1}] * 2)
        return ReviewResult(
            pair_id=dossier.pair_id,
            decision=decision,  # type: ignore[arg-type]
            confidence=0.9 if arbiter else 0.7,
            reason_codes=["NAME_STRONG", "DOB_MATCH"],
            evidence_summary=a.evidence_summary,
            quality_flags=a.quality_flags,
            protocol="arbiter" if arbiter else "consensus",
            agent_a=a,
            agent_b=b,
            arbiter=arb,
            total_latency_s=3.25,
        )


def filled_cache(tmp_path: Path, n: int = 12):
    reviewer = lre.CachedParallelReviewer(
        cache_dir=tmp_path / "cache", reviewer_factory=RichFakeReviewer
    )
    vs = views(n)
    reviewer.prefetch(vs, max_workers=3)
    return reviewer, vs


def test_fixture_round_trip_is_identical(tmp_path):
    reviewer, vs = filled_cache(tmp_path)
    path, manifest = lre.export_fixture(
        list(reviewer.records_by_pair().values()), tmp_path / "fx"
    )
    fixture = lre.FixtureReviewer(path)
    assert len(fixture) == manifest["n_pairs"] == len(vs)
    protocols = set()
    for v in vs:
        original = reviewer.review(v)
        replay = fixture.review(v)
        protocols.add(replay.protocol)
        for field in ("pair_id", "decision", "confidence", "reason_codes", "protocol"):
            assert getattr(replay, field) == getattr(original, field)
        assert replay.total_latency_s == original.total_latency_s
        for name in ("agent_a", "agent_b", "arbiter"):
            ra, oa = getattr(replay, name), getattr(original, name)
            assert (ra is None) == (oa is None)
            if oa is not None:
                for field in (
                    "decision",
                    "confidence",
                    "reason_codes",
                    "tokens_prompt",
                    "tokens_completion",
                    "latency_s",
                ):
                    assert getattr(ra, field) == getattr(oa, field)
    assert protocols == {"consensus", "arbiter"}
    assert fixture.n_replayed == len(vs) and fixture.n_missing == 0


def test_fixture_has_no_free_text_and_a_manifest(tmp_path):
    reviewer, _ = filled_cache(tmp_path)
    assert FREE_TEXT in reviewer.path.read_text(encoding="utf-8")  # está no cache
    path, manifest = lre.export_fixture(
        list(reviewer.records_by_pair().values()),
        tmp_path / "fx",
        origin={"calibration_sample": {"seed": 1}},
    )
    text = path.read_text(encoding="utf-8")
    assert FREE_TEXT not in text
    for banned in ("raw_json", "evidence_summary", "quality_flags", "supports_match"):
        assert banned not in text
    for line in text.splitlines():
        assert set(json.loads(line)) == lre.FIXTURE_KEYS
    import hashlib

    assert manifest["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    saved = json.loads(
        (tmp_path / "fx" / f"{lre.FIXTURE_STEM}.manifest.json").read_text()
    )
    assert saved == manifest
    assert manifest["model"] == "claude-haiku-5-5"
    assert manifest["prompt_version"] == "v4"
    assert manifest["removed_fields"] == ["features.nota_final"]
    assert manifest["origin"] == {"calibration_sample": {"seed": 1}}
    # determinístico: exportar de novo dá o mesmo hash
    again, m2 = lre.export_fixture(
        list(reviewer.records_by_pair().values()), tmp_path / "fx2"
    )
    assert m2["sha256"] == manifest["sha256"]


def test_fixture_on_missing_error_and_rates(tmp_path):
    reviewer, vs = filled_cache(tmp_path, n=6)
    path, _ = lre.export_fixture(
        list(reviewer.records_by_pair().values()), tmp_path / "fx"
    )
    absent = lre.make_view(row(500))
    strict = lre.FixtureReviewer(path)  # padrão: error
    with pytest.raises(lre.FixtureMissingError):
        strict.review(absent)
    assert strict.n_missing == 1
    with pytest.raises(ValueError):
        lre.FixtureReviewer(path, on_missing="rates")  # faltam taxas/verdade/faixa
    with pytest.raises(ValueError):
        lre.FixtureReviewer(path, on_missing="outro")

    rates_csv = tmp_path / "rates.csv"
    pd.DataFrame(
        {
            "band": ["grey_mid", "high"],
            "fp": [0.0, 0.0],
            "fn": [0.0, 1.0],
            "unsure": [0.0, 0.0],
        }
    ).to_csv(rates_csv, index=False)
    rates = lre.load_rates_csv(rates_csv)
    assert rates["high"] == (0.0, 1.0, 0.0)
    pos = lre.make_view(row(501))
    neg = lre.make_view(row(502))
    fx = lre.FixtureReviewer(
        path,
        on_missing="rates",
        rates=rates,
        truth={pos.pair_id: True, neg.pair_id: False},
        band_by_pair={pos.pair_id: "high", neg.pair_id: "grey_mid"},
        seed=3,
    )
    assert fx.review(pos).decision == "NONMATCH"  # fn = 1 na faixa high
    assert fx.review(neg).decision == "NONMATCH"  # fp = 0
    assert fx.review(pos) == fx.review(pos)  # determinístico por pair_id
    assert fx.review(vs[0]).pair_id == vs[0].pair_id  # presente: replay
    assert fx.n_replayed == 1 and fx.n_missing == 4


def test_fixture_rejects_mixed_configurations(tmp_path):
    reviewer, _ = filled_cache(tmp_path, n=3)
    recs = list(reviewer.records_by_pair().values())
    recs[0] = {**recs[0], "prompt_version": "v3"}
    with pytest.raises(ValueError, match="configuração"):
        lre.export_fixture(recs, tmp_path / "fx")


def test_measured_rates_are_raw_proportions():
    cells = {(b, s): lre.Cell() for b in lre.BAND_NAMES for s in lre.SIDES}
    cells[("grey_mid", "abaixo")] = lre.Cell(
        pos=4, neg=10, neg_pop=100, tp=2, fn=1, unsure_pos=1, fp=1, tn=8, unsure_neg=1
    )
    rates = lre.measured_rates(cells)
    fp, fn, un = rates["grey_mid"]
    assert fp == pytest.approx(1 / 10)
    assert fn == pytest.approx(1 / 4)
    assert un == pytest.approx(2 / 14)
    assert np.isnan(rates["high"][0])


# --- fila ponta a ponta (offline) -----------------------------------------------


def synthetic_extract():
    df, truth = make_data(700, seed=5)
    nota = df["nota final"].to_numpy(dtype=float)
    ids = np.array(
        [
            lre.make_pair_id(c, r, p)
            for c, r, p in zip(df["COMPREC"], df["REFREC"], df["PASSO"], strict=True)
        ],
        dtype=object,
    )
    t = np.array([truth[i] for i in ids])
    ex = lre.Extract(df, nota, lre.band_of_nota(nota), t, nota < 7.0, ids)
    return ex, truth


def test_drive_queue_prefetch_resume_and_budget(tmp_path):
    ex, truth = synthetic_extract()

    def make_reviewer(cap: int = 10_000):
        return lre.CachedParallelReviewer(
            cache_dir=tmp_path / "cache",
            reviewer_factory=lambda: OracleReviewer(truth, seed=1),
            max_api_reviews=cap,
        )

    kw = dict(
        negative_labels="llm_nonmatch",
        workers=4,
        k_hot=6,
        k_cold=3,
        k_low=2,
        k_explore=3,
        max_minutes=None,
        out_dir=tmp_path / "out",
    )
    ledger = lre.BudgetLedger(tmp_path / "budget.json", cap=10_000)
    rev = make_reviewer()
    first = lre.drive_queue(ex, rev, ledger=ledger, **kw)
    assert first["status"] == "finished"
    t = first["totals"]
    assert 20 < t["llm_reviews"] <= rev.api_reviews == len(rev)
    assert (
        rev.api_reviews - t["llm_reviews"] < 0.2 * t["llm_reviews"]
    )  # sobra da pré-busca
    assert t["human_decisions"] > 0 and t["verified_true"] > 0
    assert t["verified_true"] <= t["true_total"]
    assert sum(r["llm_reviews"] for r in first["rounds"]) == t["llm_reviews"]
    assert ledger.used == rev.api_reviews
    assert json.loads((tmp_path / "budget.json").read_text())["used"] == ledger.used
    # nenhum identificador de par nos agregados
    blob = json.dumps(first)
    assert not any(str(pid) in blob for pid in list(ex.pair_ids)[:50])

    # nova execução: tudo vem do cache, sem API e com o mesmo resultado
    ledger2 = lre.BudgetLedger(tmp_path / "budget.json", cap=10_000)
    rev2 = make_reviewer(cap=0)
    again = lre.drive_queue(ex, rev2, ledger=ledger2, **kw)
    assert rev2.api_reviews == 0
    assert again["rounds"] == first["rounds"] and again["totals"] == t

    # teto: para limpo, salva o estado e mantém o que já foi gravado
    (tmp_path / "cache" / "claude-haiku-5-5__v4.jsonl").unlink()
    ledger3 = lre.BudgetLedger(tmp_path / "budget3.json", cap=15)
    rev3 = lre.CachedParallelReviewer(
        cache_dir=tmp_path / "cache3",
        reviewer_factory=lambda: OracleReviewer(truth, seed=1),
        max_api_reviews=ledger3.remaining,
    )
    cut = lre.drive_queue(
        ex, rev3, ledger=ledger3, **{**kw, "out_dir": tmp_path / "o3"}
    )
    assert cut["status"] == "ReviewBudgetExceeded"
    assert ledger3.used == rev3.api_reviews == 15
    assert len(rev3) == 15
    assert (tmp_path / "o3" / "queue_llm_nonmatch.state.json").exists()


def test_h1_h0_comparison_helpers():
    base = {
        "h1": {
            "match": [1, 0, 1, 1, 0, 1],
            "truth": [1, 0, 0, 1, 1, 1],
            "below": [0, 0, 0, 1, 1, 1],
        },
        "h0": {"truth": [1, 0, 1, 1], "below": [0, 0, 1, 1]},
    }
    h = lre.h1_at_human(base, 2)  # segundo MATCH está na posição 3
    assert h == {
        "human": 2,
        "llm_reviews": 3,
        "verified": 1,
        "below": 0,
        "truncated": False,
    }
    h = lre.h1_at_human(base, 10)  # base curta demais: valor ao fim, truncada
    assert h["truncated"] and h["human"] == 4 and h["verified"] == 3
    assert lre.h1_at_llm(base, 4) == {
        "human": 3,
        "llm_reviews": 4,
        "verified": 2,
        "below": 1,
        "truncated": False,
    }
    assert lre.h1_at_llm(base, 9)["truncated"]
    assert lre.h0_at_human(base, 3) == {"human": 3, "verified": 2, "below": 1}


# --- reordenação (desempate) -----------------------------------------------------

# nota (ordem N): 9.1 8.9 8.6 8.4 8.2 7.9 ; blocos de 0,5: 18 17 17 16 16 15
R_NOTA = np.array([9.1, 8.9, 8.6, 8.4, 8.2, 7.9])
R_GROUP = np.array([2, 0, 1, 0, 2, 0])  # NONMATCH MATCH UNSURE MATCH NONMATCH MATCH
R_CONF = np.array([0.9, 0.6, 0.5, 0.9, 0.4, 0.7])
R_TRUTH = np.array([0, 0, 0, 1, 0, 1])


def r_order(strategy: str, w: int = 6) -> list[int]:
    n = 6
    block = lre.score_block(R_NOTA[:w])
    return lre.full_order(strategy, n, R_GROUP[:w], R_CONF[:w], block).tolist()


def test_orderings_by_hand():
    assert r_order("N") == [0, 1, 2, 3, 4, 5]
    assert r_order("L1") == [1, 3, 5, 2, 0, 4]  # MATCH, UNSURE, NONMATCH
    assert r_order("L2") == [0, 1, 2, 3, 4, 5]  # blocos 18 | 17 | 16 | 15
    assert r_order("L3") == [3, 5, 1, 2, 0, 4]  # MATCH por confiança 0,9 0,7 0,6
    # janela menor: o resto vem depois, na ordem da nota
    assert r_order("L1", w=4) == [1, 3, 2, 0, 4, 5]
    assert r_order("L3", w=4) == [3, 1, 2, 0, 4, 5]
    with pytest.raises(ValueError):
        lre.window_order("L9", np.arange(2), R_GROUP[:2], R_CONF[:2], R_GROUP[:2])


def test_l2_blocks_break_ties_only_inside_each_block():
    nota = np.array([9.4, 9.3, 8.9, 8.6])  # blocos 18 18 17 17
    group = np.array([2, 0, 2, 0])
    conf = np.array([0.5, 0.5, 0.5, 0.5])
    order = lre.full_order("L2", 4, group, conf, lre.score_block(nota)).tolist()
    assert order == [1, 0, 3, 2]
    assert lre.full_order("L1", 4, group, conf, lre.score_block(nota)).tolist() == [
        1,
        3,
        0,
        2,
    ]


def test_every_strategy_is_a_permutation_nothing_discarded():
    rng = np.random.default_rng(4)
    n, w = 300, 120
    nota = np.sort(rng.uniform(3, 9, n))[::-1]
    group = rng.integers(0, 3, w)
    conf = rng.random(w)
    block = lre.score_block(nota[:w])
    for strategy in lre.STRATEGIES:
        order = lre.full_order(strategy, n, group, conf, block)
        assert sorted(order.tolist()) == list(range(n))
        assert sorted(order[:w].tolist()) == list(range(w))  # a janela fica na janela
        assert order[w:].tolist() == list(range(w, n))


def test_found_counts_at_budget():
    order = np.array(r_order("L1"))
    curve = lre.found_curve(order, R_TRUTH, 6)
    assert curve.tolist() == [0, 1, 2, 2, 2, 2]  # os verdadeiros são 3 e 5
    assert lre.found_at(order, R_TRUTH, 2) == 1
    assert lre.found_at(np.arange(6), R_TRUTH, 4) == 1
    assert lre.found_at(np.arange(6), R_TRUTH, 6) == 2


def test_base_order_ties_by_pair_id():
    nota = np.array([5.0, 7.0, 7.0, 6.0])
    pid = np.array(["b", "z", "a", "c"], dtype=object)
    assert lre.base_order(nota, pid).tolist() == [2, 1, 3, 0]


def boot_inputs(w: int = 80):
    rng = np.random.default_rng(1)
    truth = np.zeros(w, dtype=bool)
    truth[rng.choice(w, 8, replace=False)] = True
    group = np.where(truth, rng.choice([0, 0, 1, 2], w), rng.choice([0, 1, 2, 2, 2], w))
    return truth, group, rng.random(w), lre.score_block(np.linspace(9, 5, w))


def test_bootstrap_is_deterministic_and_zero_beyond_the_window():
    truth, group, conf, block = boot_inputs()
    kw = dict(reps=200, seed=7)
    a = lre.bootstrap_diff("L1", truth, group, conf, block, (10, 40, 80, 160), **kw)
    b = lre.bootstrap_diff("L1", truth, group, conf, block, (10, 40, 80, 160), **kw)
    assert a == b
    assert a != lre.bootstrap_diff(
        "L1", truth, group, conf, block, (10, 40, 80, 160), reps=200, seed=8
    )
    for budget in (80, 160):  # B >= W: mesmo conjunto revisado
        assert a[budget] == {"observed": 0.0, "boot_mean": 0.0, "lo": 0.0, "hi": 0.0}
    for v in a.values():
        assert v["lo"] <= v["hi"]
    # N contra N: diferença nula em toda réplica
    n_vs_n = lre.bootstrap_diff("N", truth, group, conf, block, (10, 40), reps=100)
    assert all(v["lo"] == v["hi"] == 0.0 for v in n_vs_n.values())


def test_bootstrap_detects_a_clear_gain():
    w = 100
    truth = np.zeros(w, dtype=bool)
    truth[[40, 55, 70, 85]] = True  # verdadeiros bem abaixo no topo da nota
    group = np.where(truth, 0, 2)  # o LLM acerta tudo
    conf = np.ones(w)
    block = lre.score_block(np.linspace(9, 5, w))
    out = lre.bootstrap_diff("L1", truth, group, conf, block, (10,), reps=300, seed=1)
    assert out[10]["observed"] == 4.0 and out[10]["lo"] > 0  # 4 contra 0 em B=10


def test_rerank_results_and_report_are_aggregates_only(tmp_path):
    n, w = 400, 120
    rng = np.random.default_rng(2)
    nota = np.sort(rng.uniform(4, 9, n))[::-1]
    truth = np.zeros(n, dtype=int)
    truth[rng.choice(n, 12, replace=False)] = 1
    recs = []
    for i in range(w):
        decision = ("MATCH", "UNSURE", "NONMATCH")[int(rng.integers(0, 3))]
        recs.append(
            {
                "pair_id": f"{i:016x}",
                "summary": {
                    "decision": decision,
                    "confidence": float(rng.random()),
                    "protocol": "consensus",
                },
                "api_calls": 2,
                "tokens_prompt": 10,
                "tokens_completion": 20,
            }
        )
    res = lre.rerank_results(
        nota, truth, recs, (60, 120), budgets=(10, 50, 100, 200, 400), reps=50
    )
    assert res["n_true"] == 12
    for w_key in ("60", "120"):
        win = res["windows"][w_key]
        assert win["api_calls"] == 2 * int(w_key)
        for strat, e in win["strategies"].items():
            assert e["diff"][400] == 0  # B = n: todos os verdadeiros em qualquer ordem
            assert e["found"][400] == 12
            if strat == "N":
                assert set(e["diff"].values()) == {0}
    md = lre.render_rerank_markdown(res, {"api": 5, "elapsed_s": 1.0}, 5, 2500)
    csv_rows = lre.rerank_csv_rows(res)
    blob = md + json.dumps(csv_rows)
    assert not any(r["pair_id"] in blob for r in recs)
    assert "Discussão" in md and "ruído" in md
    assert {r["tabela"] for r in csv_rows} >= {"janela", "encontrados", "curva"}
