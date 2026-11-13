"""Anthropic transport for the LLM clerical review, with prompt caching.

Ported from the thesis (``gzcmd.anthropic_transport``). It keeps the message
builders, the JSON parser and the dual-agent + arbiter protocol of
:class:`~gzcmd_record_linkage.llm_review.LLMReviewer` and changes only the wire
format: the invariant prompt prefix of the ``system`` and ``user`` messages is
sent as a text block marked ``cache_control: ephemeral``, and the variable
suffix (role instruction, dossier, opinions) follows as a second block.

The ``anthropic`` package is an optional dependency, imported lazily::

    pip install gzcmd-record-linkage[anthropic]

Usage (reproducing the thesis Claude Haiku runs: v4 instructions, dossier
without ``model_outputs``, ``max_tokens=2500``, no ``temperature``)::

    from gzcmd_record_linkage.anthropic_transport import AnthropicReviewer
    from gzcmd_record_linkage.llm_dossier import DossierView, build_dossier

    reviewer = AnthropicReviewer.from_env(prompt_version="v4")  # ANTHROPIC_API_KEY
    result = reviewer.review(DossierView(build_dossier(row), True))

No client is constructed until the first request, and provider text is never
logged.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

from .llm_review import LLMCallError, LLMReviewer, _extract_json

__all__ = ["AnthropicReviewer", "caching_metadata"]

_INSTALL_HINT = (
    "The 'anthropic' package is required for AnthropicReviewer. "
    "Install it with: pip install gzcmd-record-linkage[anthropic]"
)


def _import_anthropic() -> Any:
    """Import ``anthropic`` lazily, with an actionable error when missing."""
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    return anthropic


def caching_metadata(warmup: bool = True) -> dict[str, Any]:
    """Describe the prompt-cache breakpoints, for recording in run metadata."""
    return {
        "enabled": True,
        "breakpoints": ["system[0]", "messages[0].content[0]"],
        "warmup": warmup,
    }


@dataclass
class AnthropicReviewer(LLMReviewer):
    """:class:`LLMReviewer` over the Anthropic Messages API.

    Differences from the base class: no ``temperature``, ``top_p`` or other
    sampling parameter is sent (those fields are ignored); the client is built
    with ``max_retries=0`` and retries (``max_retries`` attempts, exponential
    backoff from ``retry_delay_s``) are handled here; a response with
    ``stop_reason == "max_tokens"`` is treated as a failed attempt.
    Defaults mirror the thesis Haiku runs: ``claude-haiku-5-5``,
    ``max_tokens=2500``, ``timeout_s=120``.
    """

    base_url: str = "https://api.anthropic.com"
    model: str = "claude-haiku-5-5"
    fallback_model: str = "claude-haiku-5-5"
    max_tokens: int = 2500
    timeout_s: float = 120.0

    _client: Any = field(default=None, init=False, repr=False)

    @classmethod
    def from_env(cls, **overrides: Any) -> AnthropicReviewer:
        """Create from ``ANTHROPIC_API_KEY`` (reads .env via dotenv if available).

        Not present in the thesis, which reads the variable in its runners.
        """
        try:
            from dotenv import load_dotenv  # type: ignore[import-untyped]

            load_dotenv()
        except ImportError:
            pass

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            msg = "ANTHROPIC_API_KEY not found in environment"
            raise OSError(msg)
        return cls(api_key=api_key, **overrides)

    def _get_client(self) -> Any:
        if self._client is None:
            anthropic = _import_anthropic()
            self._client = anthropic.Anthropic(
                api_key=self.api_key, max_retries=0, timeout=self.timeout_s
            )
        return self._client

    def cache_blocks(
        self, messages: list[dict[str, str]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Split chat messages into (system, user) blocks with a cached prefix.

        The text is preserved byte for byte: the concatenation of the blocks
        equals the original message content.
        """
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        user = "\n\n".join(m["content"] for m in messages if m["role"] == "user")
        system_prompt, user_template = self._get_prompts()
        if not system.startswith(system_prompt) or not user.startswith(user_template):
            raise ValueError("Messages do not match the selected prompt")

        def blocks(text: str, prefix: str) -> list[dict[str, Any]]:
            return [
                {
                    "type": "text",
                    "text": prefix,
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": text[len(prefix) :]},
            ]

        # Separators stay at the start of the variable suffix; text is never
        # stripped. The arbiter's dossier and opinions keep their order.
        return blocks(system, system_prompt), blocks(user, user_template)

    def _call_api(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
    ) -> tuple[dict[str, Any], float, int, int]:
        """Make a single API call with retry logic.

        Returns (parsed_json, latency_s, input_tokens, output_tokens).
        Raises :class:`LLMCallError` once all attempts are exhausted.
        """
        model_id = model or self.model
        anthropic = _import_anthropic()
        retryable = (
            anthropic.APIStatusError,
            anthropic.APIConnectionError,
            ValueError,
            KeyError,
        )
        system, user = self.cache_blocks(messages)
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            started = time.monotonic()
            try:
                response = self._get_client().messages.create(
                    model=model_id,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                    max_tokens=self.max_tokens,
                )
                text = "".join(
                    getattr(block, "text", "")
                    for block in response.content
                    if getattr(block, "type", "") == "text"
                )
                if response.stop_reason == "max_tokens":
                    raise ValueError("Truncated response")
                parsed = self._parse_response(text)
                return (
                    parsed,
                    time.monotonic() - started,
                    int(response.usage.input_tokens),
                    int(response.usage.output_tokens),
                )
            except retryable as exc:
                # Only the exception type is kept: provider text, dossiers
                # and credentials must not reach logs or messages.
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay_s * (2 ** (attempt - 1)))
        error_type = type(last_error).__name__ if last_error else "none"
        raise LLMCallError(
            f"Anthropic attempts exhausted (last error: {error_type})",
            model=model_id,
            pair_id="",
            attempts=self.max_retries,
        ) from last_error

    def _parse_response(self, text: str) -> dict[str, Any]:
        return _extract_json(text)
