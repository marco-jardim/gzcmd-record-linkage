"""LLM clerical review orchestrator — dual-agent + arbiter protocol.

Sends a dossier to two independent LLM agents (Agent-A, Agent-B) with the
same model but slightly varied system prompts. If they agree, the consensus
is accepted. If they disagree, an Arbiter agent receives both opinions and
makes the final call.

All calls go through the Fireworks REST API.

Usage::

    from gzcmd_record_linkage.llm_review import LLMReviewer, ReviewResult

    reviewer = LLMReviewer.from_env()              # reads .env
    result = reviewer.review(dossier)               # single dossier
    results = reviewer.review_batch(dossiers)       # list of dossiers
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import requests

from .llm_dossier import Dossier

__all__ = [
    "LLMReviewer",
    "ReviewResult",
    "ReviewVerdict",
    "LLMCallError",
]

logger = logging.getLogger(__name__)

ReviewVerdict = Literal["MATCH", "NONMATCH", "UNSURE"]

# Where to find the v3.1 prompt
_PROMPT_PATH = Path(__file__).parent / "gzcmd_v3_llm_prompt.md"

# Valid reason codes (must match schema)
VALID_REASON_CODES = frozenset(
    {
        "HIGH_SCORE_ANCHOR",
        "LOW_SCORE_ANCHOR",
        "NAME_STRONG",
        "NAME_WEAK",
        "DOB_STRONG",
        "DOB_WEAK",
        "MOTHER_STRONG",
        "MOTHER_MISSING",
        "ADDRESS_WEAK",
        "MUNICIPALITY_STRONG",
        "MODEL_HIGH_P",
        "MODEL_AMBIGUOUS",
        "CONFLICTING_SIGNALS",
        "INSUFFICIENT_EVIDENCE",
    }
)

VALID_DECISIONS = frozenset({"MATCH", "NONMATCH", "UNSURE"})

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class LLMCallError(Exception):
    """Raised when an LLM API call fails after all retries."""

    def __init__(self, message: str, *, model: str, pair_id: str, attempts: int):
        super().__init__(message)
        self.model = model
        self.pair_id = pair_id
        self.attempts = attempts


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentResponse:
    """Parsed response from a single LLM agent."""

    decision: ReviewVerdict
    confidence: float
    reason_codes: list[str]
    evidence_summary: dict[str, list[str]]
    quality_flags: dict[str, bool]
    raw_json: dict[str, Any]
    latency_s: float
    tokens_prompt: int
    tokens_completion: int


@dataclass(frozen=True)
class ReviewResult:
    """Final review output for one dossier."""

    pair_id: str
    decision: ReviewVerdict
    confidence: float
    reason_codes: list[str]
    evidence_summary: dict[str, list[str]]
    quality_flags: dict[str, bool]
    protocol: str  # "consensus" | "arbiter"
    agent_a: AgentResponse
    agent_b: AgentResponse
    arbiter: AgentResponse | None = None
    total_latency_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage/audit."""
        d: dict[str, Any] = {
            "pair_id": self.pair_id,
            "decision": self.decision,
            "confidence": self.confidence,
            "reason_codes": self.reason_codes,
            "evidence_summary": self.evidence_summary,
            "quality_flags": self.quality_flags,
            "protocol": self.protocol,
            "agent_a_decision": self.agent_a.decision,
            "agent_b_decision": self.agent_b.decision,
            "total_latency_s": round(self.total_latency_s, 2),
            "total_tokens": (
                self.agent_a.tokens_prompt
                + self.agent_a.tokens_completion
                + self.agent_b.tokens_prompt
                + self.agent_b.tokens_completion
            ),
        }
        if self.arbiter:
            d["arbiter_decision"] = self.arbiter.decision
            d["total_tokens"] += (
                self.arbiter.tokens_prompt + self.arbiter.tokens_completion
            )
        return d


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------


def _load_system_prompt() -> str:
    """Load the v3.1 system prompt from the markdown file.

    Extracts only the content under ``## Mensagem `system``` through
    the next ``---`` or ``## Mensagem `user````.
    """
    raw = _PROMPT_PATH.read_text(encoding="utf-8")

    # Extract system message section
    match = re.search(
        r"## Mensagem `system`\s*\n(.*?)(?=\n---|\n## Mensagem `user`)",
        raw,
        re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    # Fallback: use everything after the header line
    return raw.strip()


def _load_user_template() -> str:
    """Load the user message template from the prompt file."""
    raw = _PROMPT_PATH.read_text(encoding="utf-8")

    match = re.search(
        r"## Mensagem `user`.*?\n(.*?)(?=\n---|\n## Exemplo|\Z)",
        raw,
        re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    return (
        "Avalie o dossiê abaixo e retorne um JSON conforme o schema de saída.\n\n"
        "```json\n{dossier}\n```"
    )


# ── public message builders (for batch inference) ─────────────────────

_ROLE_INSTRUCTION: dict[str, str] = {
    "Agent-A": (
        "\n\nVocê é o Agent-A. Analise o dossiê de forma direta e objetiva. "
        "Priorize a evidência mais forte."
    ),
    "Agent-B": (
        "\n\nVocê é o Agent-B. Analise o dossiê com atenção especial a "
        "possíveis contradições e dados faltantes. Seja cético mas justo."
    ),
    "Arbiter": (
        "\n\nVocê é o Árbitro. Dois revisores divergiram na decisão. "
        "Analise o dossiê original e as duas opiniões. "
        "Dê a decisão final com justificativa."
    ),
}


def build_review_messages(
    dossier: "Dossier",
    agent_role: str = "Agent-A",
    batch_mode: bool = False,
) -> list[dict[str, str]]:
    """Build chat messages for a review agent (public, no class instance needed).

    Suitable for Fireworks batch inference JSONL generation.
    When *batch_mode* is True an extra instruction is appended to
    suppress chain-of-thought reasoning so the model outputs **only**
    the JSON object (avoids token-budget exhaustion in batch jobs).
    """
    system_prompt = _load_system_prompt()
    user_template = _load_user_template()
    full_system = system_prompt + _ROLE_INSTRUCTION.get(agent_role, "")
    if batch_mode:
        full_system += (
            "\n\nCRITICAL: Output ONLY the JSON object. "
            "Do NOT include any reasoning, analysis, chain-of-thought, "
            "or explanation before or after the JSON. "
            "Start your response with { and end with }."
        )
    dossier_json = dossier.to_json(indent=2)
    user_msg = f"{user_template}\n\n```json\n{dossier_json}\n```"
    return [
        {"role": "system", "content": full_system},
        {"role": "user", "content": user_msg},
    ]


def build_arbiter_messages(
    dossier: "Dossier",
    response_a: dict[str, Any],
    response_b: dict[str, Any],
) -> list[dict[str, str]]:
    """Build chat messages for the arbiter (public, no class instance needed)."""
    system_prompt = _load_system_prompt()
    user_template = _load_user_template()
    full_system = (
        system_prompt
        + "\n\nVocê é o Árbitro. Dois revisores independentes divergiram. "
        "Analise o dossiê original e as duas opiniões abaixo. "
        "Dê a decisão final. Não repita PII."
    )
    dossier_json = dossier.to_json(indent=2)
    user_msg = (
        f"{user_template}\n\n"
        f"### Dossiê original\n```json\n{dossier_json}\n```\n\n"
        f"### Opinião Agent-A\n```json\n{json.dumps(response_a, ensure_ascii=False, indent=2)}\n```\n\n"
        f"### Opinião Agent-B\n```json\n{json.dumps(response_b, ensure_ascii=False, indent=2)}\n```\n\n"
        "Analise as divergências e dê sua decisão final no formato JSON."
    )
    return [
        {"role": "system", "content": full_system},
        {"role": "user", "content": user_msg},
    ]


def extract_and_validate(raw_content: str, pair_id: str) -> dict[str, Any]:
    """Extract JSON from raw LLM output and validate against schema (public)."""
    return _validate_response(_extract_json(raw_content), pair_id)


# ---------------------------------------------------------------------------
# Response parsing & validation
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> dict[str, Any]:
    """Parse JSON from LLM output, handling markdown fences and <think> blocks."""
    # Strip <think>...</think> blocks (DeepSeek R1 style)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Strip untagged thinking text before the first '{' (DeepSeek V3 style)
    # These models sometimes output reasoning in natural language before the JSON
    first_brace = text.find("{")
    if first_brace > 0:
        text = text[first_brace:]

    # Try direct parse
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code fence
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding JSON starting with {"decision" (our expected schema key)
    decision_start = text.find('{"decision"')
    if decision_start == -1:
        decision_start = text.find('"decision"')
        if decision_start > 0:
            # Walk back to the opening brace
            candidate = text[:decision_start].rfind("{")
            if candidate != -1:
                decision_start = candidate
    if decision_start != -1:
        # Find matching closing brace via bracket counting
        depth = 0
        for i in range(decision_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[decision_start : i + 1])
                    except json.JSONDecodeError:
                        break

    # Fallback: outermost {...}
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start : brace_end + 1])
        except json.JSONDecodeError:
            pass

    msg = f"Could not extract valid JSON from LLM response: {text[:200]}"
    raise ValueError(msg)


def _validate_response(data: dict[str, Any], pair_id: str) -> dict[str, Any]:
    """Validate and normalize an LLM response dict against the output schema."""
    # Decision
    decision = str(data.get("decision", "UNSURE")).upper().strip()
    if decision not in VALID_DECISIONS:
        logger.warning(
            "Invalid decision '%s' for %s, defaulting to UNSURE", decision, pair_id
        )
        decision = "UNSURE"

    # Confidence
    try:
        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.5

    # Reason codes — filter to valid set
    raw_codes = data.get("reason_codes", [])
    if isinstance(raw_codes, list):
        codes = [c for c in raw_codes if c in VALID_REASON_CODES]
    else:
        codes = []
    if not codes:
        codes = ["INSUFFICIENT_EVIDENCE"]

    # Evidence summary
    ev = data.get("evidence_summary", {})
    if not isinstance(ev, dict):
        ev = {}
    evidence_summary = {
        "supports_match": _ensure_str_list(ev.get("supports_match", [])),
        "supports_nonmatch": _ensure_str_list(ev.get("supports_nonmatch", [])),
        "tie_breakers": _ensure_str_list(ev.get("tie_breakers", [])),
    }

    # Quality flags
    qf = data.get("quality_flags", {})
    if not isinstance(qf, dict):
        qf = {}
    quality_flags = {
        "pii_leak_detected": bool(qf.get("pii_leak_detected", False)),
        "insufficient_evidence": bool(qf.get("insufficient_evidence", False)),
        "inconsistent_input": bool(qf.get("inconsistent_input", False)),
    }

    return {
        "pair_id": pair_id,
        "decision": decision,
        "confidence": confidence,
        "reason_codes": codes,
        "evidence_summary": evidence_summary,
        "quality_flags": quality_flags,
    }


def _ensure_str_list(val: Any) -> list[str]:
    if isinstance(val, list):
        return [str(v) for v in val[:10]]
    return []


# ---------------------------------------------------------------------------
# LLM API caller
# ---------------------------------------------------------------------------


@dataclass
class LLMReviewer:
    """Orchestrates dual-agent + arbiter LLM clerical review.

    Parameters
    ----------
    api_key:
        Fireworks API key.
    base_url:
        Fireworks API base URL.
    model:
        Primary model ID (e.g. ``accounts/fireworks/models/kimi-k2p5``).
    fallback_model:
        Fallback model if primary fails.
    temperature:
        Sampling temperature (0.0 = deterministic).
    max_tokens:
        Max tokens per completion.
    max_retries:
        Max retry attempts per API call.
    retry_delay_s:
        Base delay between retries (doubles each attempt).
    timeout_s:
        HTTP request timeout in seconds.
    """

    api_key: str
    base_url: str = "https://api.fireworks.ai/inference/v1/chat/completions"
    model: str = "accounts/fireworks/models/kimi-k2p5"
    fallback_model: str = "accounts/fireworks/models/qwen3-235b-a22b"
    temperature: float = 0.0
    max_tokens: int = 800
    top_p: float = 1.0
    top_k: int = 40
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    stop: list[str] | None = None
    max_retries: int = 3
    retry_delay_s: float = 2.0
    timeout_s: float = 60.0

    # Cached prompts (loaded lazily)
    _system_prompt: str = field(default="", init=False, repr=False)
    _user_template: str = field(default="", init=False, repr=False)

    @classmethod
    def from_env(cls, **overrides: Any) -> LLMReviewer:
        """Create from environment variables (reads .env via dotenv if available)."""
        try:
            from dotenv import load_dotenv  # type: ignore[import-untyped]

            load_dotenv()
        except ImportError:
            pass

        api_key = os.environ.get("FIREWORKS_API_KEY", "")
        if not api_key:
            msg = "FIREWORKS_API_KEY not found in environment"
            raise EnvironmentError(msg)

        base_url = os.environ.get(
            "FIREWORKS_BASE_URL",
            "https://api.fireworks.ai/inference/v1/chat/completions",
        )
        return cls(api_key=api_key, base_url=base_url, **overrides)

    def _get_prompts(self) -> tuple[str, str]:
        if not self._system_prompt:
            self._system_prompt = _load_system_prompt()
            self._user_template = _load_user_template()
        return self._system_prompt, self._user_template

    # ------------------------------------------------------------------
    # Low-level API call
    # ------------------------------------------------------------------

    def _call_api(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
    ) -> tuple[dict[str, Any], float, int, int]:
        """Make a single API call with retry logic.

        Returns (parsed_json, latency_s, prompt_tokens, completion_tokens).
        """
        model = model or self.model
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "response_format": {"type": "json_object"},
        }
        if self.stop:
            payload["stop"] = self.stop

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            t0 = time.monotonic()
            try:
                resp = requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_s,
                )
                latency = time.monotonic() - t0

                if resp.status_code == 429:
                    # Rate limited — back off
                    delay = self.retry_delay_s * (2 ** (attempt - 1))
                    logger.warning(
                        "Rate limited (429), attempt %d/%d, sleeping %.1fs",
                        attempt,
                        self.max_retries,
                        delay,
                    )
                    time.sleep(delay)
                    continue

                resp.raise_for_status()
                body = resp.json()

                content = body["choices"][0]["message"]["content"]
                usage = body.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

                parsed = _extract_json(content)
                return parsed, latency, prompt_tokens, completion_tokens

            except (requests.RequestException, ValueError, KeyError) as exc:
                last_error = exc
                latency = time.monotonic() - t0
                logger.warning(
                    "API call failed (attempt %d/%d, %.1fs): %s",
                    attempt,
                    self.max_retries,
                    latency,
                    exc,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay_s * (2 ** (attempt - 1)))

        # All retries exhausted — try fallback model once
        if model != self.fallback_model:
            logger.warning(
                "Primary model %s failed, trying fallback %s",
                model,
                self.fallback_model,
            )
            return self._call_api(messages, model=self.fallback_model)

        msg = f"All {self.max_retries} attempts failed: {last_error}"
        raise LLMCallError(msg, model=model, pair_id="", attempts=self.max_retries)

    # ------------------------------------------------------------------
    # Agent calls
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        dossier: Dossier,
        *,
        agent_role: str = "Agent-A",
    ) -> list[dict[str, str]]:
        """Build chat messages for an agent call."""
        system_prompt, user_template = self._get_prompts()

        # Differentiate agents slightly for diversity
        role_instruction = {
            "Agent-A": (
                "\n\nVocê é o Agent-A. Analise o dossiê de forma direta e objetiva. "
                "Priorize a evidência mais forte."
            ),
            "Agent-B": (
                "\n\nVocê é o Agent-B. Analise o dossiê com atenção especial a "
                "possíveis contradições e dados faltantes. Seja cético mas justo."
            ),
            "Arbiter": (
                "\n\nVocê é o Árbitro. Dois revisores divergiram na decisão. "
                "Analise o dossiê original e as duas opiniões. "
                "Dê a decisão final com justificativa."
            ),
        }

        full_system = system_prompt + role_instruction.get(agent_role, "")
        dossier_json = dossier.to_json(indent=2)
        user_msg = f"{user_template}\n\n```json\n{dossier_json}\n```"

        return [
            {"role": "system", "content": full_system},
            {"role": "user", "content": user_msg},
        ]

    def _build_arbiter_messages(
        self,
        dossier: Dossier,
        response_a: dict[str, Any],
        response_b: dict[str, Any],
    ) -> list[dict[str, str]]:
        """Build messages for the arbiter, including both agent opinions."""
        system_prompt, user_template = self._get_prompts()

        full_system = (
            system_prompt
            + "\n\nVocê é o Árbitro. Dois revisores independentes divergiram. "
            "Analise o dossiê original e as duas opiniões abaixo. "
            "Dê a decisão final. Não repita PII."
        )

        dossier_json = dossier.to_json(indent=2)
        user_msg = (
            f"{user_template}\n\n"
            f"### Dossiê original\n```json\n{dossier_json}\n```\n\n"
            f"### Opinião Agent-A\n```json\n{json.dumps(response_a, ensure_ascii=False, indent=2)}\n```\n\n"
            f"### Opinião Agent-B\n```json\n{json.dumps(response_b, ensure_ascii=False, indent=2)}\n```\n\n"
            "Analise as divergências e dê sua decisão final no formato JSON."
        )

        return [
            {"role": "system", "content": full_system},
            {"role": "user", "content": user_msg},
        ]

    def _call_agent(
        self,
        dossier: Dossier,
        *,
        agent_role: str,
    ) -> AgentResponse:
        """Call a single agent and return parsed response."""
        if agent_role == "Arbiter":
            msg = "Use _call_arbiter for arbiter calls"
            raise ValueError(msg)

        messages = self._build_messages(dossier, agent_role=agent_role)
        raw, latency, pt, ct = self._call_api(messages)
        validated = _validate_response(raw, dossier.pair_id)

        return AgentResponse(
            decision=validated["decision"],
            confidence=validated["confidence"],
            reason_codes=validated["reason_codes"],
            evidence_summary=validated["evidence_summary"],
            quality_flags=validated["quality_flags"],
            raw_json=raw,
            latency_s=round(latency, 2),
            tokens_prompt=pt,
            tokens_completion=ct,
        )

    def _call_arbiter(
        self,
        dossier: Dossier,
        response_a: dict[str, Any],
        response_b: dict[str, Any],
    ) -> AgentResponse:
        """Call the arbiter with both agent opinions."""
        messages = self._build_arbiter_messages(dossier, response_a, response_b)
        raw, latency, pt, ct = self._call_api(messages)
        validated = _validate_response(raw, dossier.pair_id)

        return AgentResponse(
            decision=validated["decision"],
            confidence=validated["confidence"],
            reason_codes=validated["reason_codes"],
            evidence_summary=validated["evidence_summary"],
            quality_flags=validated["quality_flags"],
            raw_json=raw,
            latency_s=round(latency, 2),
            tokens_prompt=pt,
            tokens_completion=ct,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def review(self, dossier: Dossier) -> ReviewResult:
        """Review a single dossier using dual-agent + arbiter protocol.

        1. Agent-A evaluates the dossier.
        2. Agent-B evaluates the dossier independently.
        3. If they agree → consensus accepted.
        4. If they disagree → Arbiter breaks the tie.

        Returns
        -------
        ReviewResult
            Final decision with full audit trail.
        """
        t0 = time.monotonic()

        # Agent A
        logger.info("Calling Agent-A for pair %s", dossier.pair_id)
        agent_a = self._call_agent(dossier, agent_role="Agent-A")

        # Agent B
        logger.info("Calling Agent-B for pair %s", dossier.pair_id)
        agent_b = self._call_agent(dossier, agent_role="Agent-B")

        # Check consensus
        if agent_a.decision == agent_b.decision:
            # Consensus — use Agent-A's full response (higher confidence wins)
            winner = agent_a if agent_a.confidence >= agent_b.confidence else agent_b
            logger.info(
                "Consensus reached for %s: %s (A=%.2f, B=%.2f)",
                dossier.pair_id,
                winner.decision,
                agent_a.confidence,
                agent_b.confidence,
            )
            return ReviewResult(
                pair_id=dossier.pair_id,
                decision=winner.decision,
                confidence=max(agent_a.confidence, agent_b.confidence),
                reason_codes=winner.reason_codes,
                evidence_summary=winner.evidence_summary,
                quality_flags=winner.quality_flags,
                protocol="consensus",
                agent_a=agent_a,
                agent_b=agent_b,
                total_latency_s=round(time.monotonic() - t0, 2),
            )

        # Disagreement — call arbiter
        logger.info(
            "Disagreement for %s (A=%s, B=%s) — calling Arbiter",
            dossier.pair_id,
            agent_a.decision,
            agent_b.decision,
        )
        arbiter = self._call_arbiter(
            dossier,
            _validate_response(agent_a.raw_json, dossier.pair_id),
            _validate_response(agent_b.raw_json, dossier.pair_id),
        )

        return ReviewResult(
            pair_id=dossier.pair_id,
            decision=arbiter.decision,
            confidence=arbiter.confidence,
            reason_codes=arbiter.reason_codes,
            evidence_summary=arbiter.evidence_summary,
            quality_flags=arbiter.quality_flags,
            protocol="arbiter",
            agent_a=agent_a,
            agent_b=agent_b,
            arbiter=arbiter,
            total_latency_s=round(time.monotonic() - t0, 2),
        )

    def review_batch(
        self,
        dossiers: list[Dossier],
        *,
        stop_on_error: bool = False,
    ) -> list[ReviewResult | LLMCallError]:
        """Review multiple dossiers sequentially.

        Parameters
        ----------
        dossiers:
            List of dossiers to review.
        stop_on_error:
            If True, raise on first error. If False, collect errors.

        Returns
        -------
        list[ReviewResult | LLMCallError]
            One result per dossier (error objects for failures).
        """
        results: list[ReviewResult | LLMCallError] = []
        for i, dossier in enumerate(dossiers, 1):
            logger.info(
                "Reviewing dossier %d/%d (pair_id=%s)",
                i,
                len(dossiers),
                dossier.pair_id,
            )
            try:
                result = self.review(dossier)
                results.append(result)
                logger.info(
                    "  → %s (confidence=%.2f, protocol=%s, %.1fs)",
                    result.decision,
                    result.confidence,
                    result.protocol,
                    result.total_latency_s,
                )
            except LLMCallError as exc:
                if stop_on_error:
                    raise
                logger.error("  → FAILED: %s", exc)
                results.append(exc)
        return results
