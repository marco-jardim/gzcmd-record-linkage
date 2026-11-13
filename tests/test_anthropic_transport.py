"""Offline tests for the Anthropic transport. A fake client is used throughout."""

from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from gzcmd_record_linkage import anthropic_transport as transport
from gzcmd_record_linkage import llm_review
from gzcmd_record_linkage.anthropic_transport import AnthropicReviewer, caching_metadata
from gzcmd_record_linkage.llm_dossier import DossierView, build_dossier
from gzcmd_record_linkage.llm_review import LLMCallError, LLMReviewer


class FakeAPIStatusError(Exception):
    pass


class FakeAPIConnectionError(Exception):
    pass


class FakeClient:
    """Stands in for ``anthropic.Anthropic``; replays queued responses."""

    instances: list[FakeClient] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.queue: list[Any] = []
        self.calls: list[dict[str, Any]] = []
        self.messages = SimpleNamespace(create=self._create)
        FakeClient.instances.append(self)

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        item = self.queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture(autouse=True)
def fake_anthropic(monkeypatch):
    """Install a fake ``anthropic`` module and forbid real HTTP."""
    module = types.ModuleType("anthropic")
    module.Anthropic = FakeClient
    module.APIStatusError = FakeAPIStatusError
    module.APIConnectionError = FakeAPIConnectionError
    monkeypatch.setitem(sys.modules, "anthropic", module)
    FakeClient.instances = []

    def no_network(*args, **kwargs):
        raise AssertionError("network is forbidden in tests")

    monkeypatch.setattr(llm_review.requests, "post", no_network)
    return module


def response(
    payload: dict[str, Any] | None = None,
    stop_reason: str = "end_turn",
    text: str | None = None,
) -> SimpleNamespace:
    body = text if text is not None else json.dumps(payload or {"decision": "MATCH"})
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", text="ignored"),
            SimpleNamespace(type="text", text=body),
        ],
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=20, output_tokens=30),
    )


def reviewer_with(*queue: Any, **kwargs: Any) -> tuple[AnthropicReviewer, FakeClient]:
    reviewer = AnthropicReviewer(api_key="offline-not-a-key", **kwargs)
    client = reviewer._get_client()
    client.queue.extend(queue)
    return reviewer, client


def make_dossier():
    row = pd.Series(
        {
            "COMPREC": 1,
            "REFREC": 2,
            "PASSO": 3,
            "nota final": 7.5,
            "NOME qtd frag iguais": 0.5,
            "DTNASC dt iguais": 1,
        }
    )
    return build_dossier(row)


def valid_payload(pair_id: str, decision: str = "MATCH") -> dict[str, Any]:
    return {
        "pair_id": pair_id,
        "decision": decision,
        "confidence": 0.8,
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


def test_caching_metadata():
    assert caching_metadata() == {
        "enabled": True,
        "breakpoints": ["system[0]", "messages[0].content[0]"],
        "warmup": True,
    }
    assert caching_metadata(False)["warmup"] is False


def test_is_a_review_reviewer_without_new_hard_dependency():
    assert issubclass(AnthropicReviewer, LLMReviewer)
    reviewer = AnthropicReviewer(api_key="offline")
    assert reviewer.model == "claude-haiku-5-5"
    assert reviewer.max_tokens == 2500
    assert reviewer._client is None  # no client until the first request
    assert FakeClient.instances == []


def test_client_built_with_max_retries_zero_and_timeout():
    reviewer, client = reviewer_with(timeout_s=33.0)
    assert client.kwargs == {
        "api_key": "offline-not-a-key",
        "max_retries": 0,
        "timeout": 33.0,
    }
    assert reviewer._get_client() is client  # cached


@pytest.mark.parametrize("version", ["v3", "v4"])
@pytest.mark.parametrize("role", ["Agent-A", "Agent-B", "Arbiter"])
def test_blocks_preserve_bytes_and_mark_prefix(version, role):
    reviewer, client = reviewer_with(response(), prompt_version=version)
    dossier = make_dossier()
    if role == "Arbiter":
        messages = reviewer._build_arbiter_messages(
            dossier, {"decision": "MATCH"}, {"decision": "NONMATCH"}
        )
    else:
        messages = reviewer._build_messages(dossier, agent_role=role)

    parsed, latency, tokens_in, tokens_out = reviewer._call_api(messages)

    assert parsed == {"decision": "MATCH"}
    assert latency >= 0.0
    assert (tokens_in, tokens_out) == (20, 30)
    request = client.calls[0]
    system, user = request["system"], request["messages"][0]["content"]
    prompts = (
        llm_review._load_system_prompt(version),
        llm_review._load_user_template(version),
    )
    assert system[0]["text"] == prompts[0]
    assert user[0]["text"] == prompts[1]
    for blocks, message in [(system, messages[0]), (user, messages[1])]:
        assert len(blocks) == 2
        assert blocks[0]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in blocks[1]
        assert blocks[1]["text"].startswith("\n\n")
        joined = "".join(b["text"] for b in blocks)
        assert joined.encode("utf-8") == message["content"].encode("utf-8")
    if role == "Arbiter":
        text = user[1]["text"]
        assert text.index("### Dossiê original") < text.index("### Opinião Agent-A")
        assert text.index("### Opinião Agent-A") < text.index("### Opinião Agent-B")
    assert request["model"] == "claude-haiku-5-5"
    assert request["max_tokens"] == 2500
    for absent in ("temperature", "top_p", "top_k", "thinking"):
        assert absent not in request


def test_cache_blocks_rejects_messages_from_another_prompt():
    reviewer = AnthropicReviewer(api_key="offline")
    messages = [
        {"role": "system", "content": "other system prompt"},
        {"role": "user", "content": "other user prompt"},
    ]
    with pytest.raises(ValueError, match="do not match"):
        reviewer.cache_blocks(messages)


def test_explicit_model_overrides_default():
    reviewer, client = reviewer_with(response())
    reviewer._call_api(reviewer._build_messages(make_dossier()), model="claude-x")
    assert client.calls[0]["model"] == "claude-x"


def test_retries_with_exponential_backoff_then_succeeds(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(transport.time, "sleep", sleeps.append)
    reviewer, client = reviewer_with(
        FakeAPIConnectionError("boom"),
        FakeAPIStatusError("429"),
        response({"decision": "UNSURE"}),
        max_retries=3,
        retry_delay_s=2.0,
    )
    parsed, *_ = reviewer._call_api(reviewer._build_messages(make_dossier()))
    assert parsed == {"decision": "UNSURE"}
    assert len(client.calls) == 3
    assert sleeps == [2.0, 4.0]


def test_max_tokens_stop_reason_is_a_failed_attempt(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(transport.time, "sleep", sleeps.append)
    reviewer, client = reviewer_with(
        response(stop_reason="max_tokens"),
        response(stop_reason="max_tokens"),
        max_retries=2,
        retry_delay_s=1.0,
    )
    with pytest.raises(LLMCallError) as caught:
        reviewer._call_api(reviewer._build_messages(make_dossier()))
    assert caught.value.attempts == 2
    assert caught.value.model == "claude-haiku-5-5"
    assert "ValueError" in str(caught.value)
    assert len(client.calls) == 2
    assert sleeps == [1.0]  # no sleep after the last attempt


def test_unparseable_text_is_retried_and_exhausts(monkeypatch):
    monkeypatch.setattr(transport.time, "sleep", lambda _s: None)
    reviewer, client = reviewer_with(
        response(text="no json here"),
        response(text="still none"),
        max_retries=2,
    )
    with pytest.raises(LLMCallError):
        reviewer._call_api(reviewer._build_messages(make_dossier()))
    assert len(client.calls) == 2


def test_unexpected_errors_are_not_swallowed(monkeypatch):
    monkeypatch.setattr(transport.time, "sleep", lambda _s: None)
    reviewer, client = reviewer_with(RuntimeError("bug"), response())
    with pytest.raises(RuntimeError, match="bug"):
        reviewer._call_api(reviewer._build_messages(make_dossier()))
    assert len(client.calls) == 1


def test_missing_anthropic_gives_actionable_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)
    reviewer = AnthropicReviewer(api_key="offline")
    with pytest.raises(ImportError, match=r"gzcmd-record-linkage\[anthropic\]"):
        reviewer._get_client()
    with pytest.raises(ImportError, match=r"gzcmd-record-linkage\[anthropic\]"):
        reviewer._call_api(reviewer._build_messages(make_dossier()))


def test_from_env_reads_anthropic_api_key(monkeypatch):
    monkeypatch.setitem(sys.modules, "dotenv", None)  # never read a real .env
    monkeypatch.setenv("ANTHROPIC_API_KEY", "offline-not-a-key")
    reviewer = AnthropicReviewer.from_env(prompt_version="v4", max_tokens=1000)
    assert isinstance(reviewer, AnthropicReviewer)
    assert reviewer.api_key == "offline-not-a-key"
    assert reviewer.prompt_version == "v4"
    assert reviewer.max_tokens == 1000


def test_from_env_requires_key(monkeypatch):
    monkeypatch.setitem(sys.modules, "dotenv", None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
        AnthropicReviewer.from_env()


def test_full_two_agent_protocol_with_v4_and_dossier_without_model_outputs():
    dossier = make_dossier()
    view = DossierView(dossier, True)
    reviewer, client = reviewer_with(
        response(valid_payload(view.pair_id)),
        response(valid_payload(view.pair_id)),
        prompt_version="v4",
    )
    result = reviewer.review(view)
    assert result.decision == "MATCH"
    assert len(client.calls) == 2
    sent = client.calls[0]["messages"][0]["content"]
    sent_text = "".join(block["text"] for block in sent)
    assert "model_outputs" not in sent_text
    assert "p_cal" not in sent_text
    assert "GZ-CMD++ v4" not in sent_text  # title is not part of the messages
    system_text = client.calls[0]["system"][0]["text"]
    assert system_text == llm_review._load_system_prompt("v4")


def test_arbiter_called_on_disagreement():
    pid = make_dossier().pair_id
    reviewer, client = reviewer_with(
        response(valid_payload(pid, "MATCH")),
        response(valid_payload(pid, "NONMATCH")),
        response(valid_payload(pid, "UNSURE")),
    )
    result = reviewer.review(make_dossier())
    assert result.decision == "UNSURE"
    assert len(client.calls) == 3
    arbiter_user = client.calls[2]["messages"][0]["content"][1]["text"]
    assert "### Opinião Agent-B" in arbiter_user


# ---- prompt_version on the base reviewer / builders -----------------------


def test_default_prompt_version_is_v3_and_unchanged():
    assert LLMReviewer(api_key="k").prompt_version == "v3"
    system, user = LLMReviewer(api_key="k")._get_prompts()
    assert system == llm_review._load_system_prompt()
    assert user == llm_review._load_user_template()
    assert llm_review._resolve_prompt_path("v3") == llm_review._PROMPT_PATH


def test_v4_prompt_is_packaged_and_differs_from_v3():
    v4 = LLMReviewer(api_key="k", prompt_version="v4")._get_prompts()
    v3 = LLMReviewer(api_key="k")._get_prompts()
    assert v4 != v3
    assert llm_review._PROMPT_V4_PATH.is_file()
    assert "sem saídas do motor" in llm_review._PROMPT_V4_PATH.read_text("utf-8")


def test_unknown_prompt_version_is_rejected():
    with pytest.raises(ValueError, match="prompt_version"):
        LLMReviewer(api_key="k", prompt_version="v9")._get_prompts()
    with pytest.raises(ValueError, match="prompt_version"):
        llm_review.build_review_messages(make_dossier(), prompt_version="v9")


def test_public_builders_accept_prompt_version():
    dossier = make_dossier()
    v3 = llm_review.build_review_messages(dossier)
    v4 = llm_review.build_review_messages(dossier, prompt_version="v4")
    assert v3 != v4
    assert v4[0]["content"].startswith(llm_review._load_system_prompt("v4"))
    arbiter = llm_review.build_arbiter_messages(
        dossier, {"decision": "MATCH"}, {"decision": "NONMATCH"}, prompt_version="v4"
    )
    assert arbiter[0]["content"].startswith(llm_review._load_system_prompt("v4"))


def test_fireworks_reviewer_still_builds_its_request(monkeypatch):
    """The Fireworks transport is untouched (not blocked, unlike the thesis)."""
    seen: dict[str, Any] = {}

    def fake_post(url, headers, json, timeout):  # noqa: A002
        seen.update(url=url, payload=json)
        body = {
            "choices": [{"message": {"content": '{"decision": "MATCH"}'}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        }
        return SimpleNamespace(
            status_code=200, raise_for_status=lambda: None, json=lambda: body
        )

    monkeypatch.setattr(llm_review.requests, "post", fake_post)
    reviewer = LLMReviewer(api_key="offline")
    parsed, _latency, p_tokens, c_tokens = reviewer._call_api(
        reviewer._build_messages(make_dossier())
    )
    assert parsed == {"decision": "MATCH"}
    assert (p_tokens, c_tokens) == (1, 2)
    assert "fireworks" in seen["url"]
    assert seen["payload"]["temperature"] == 0.0
