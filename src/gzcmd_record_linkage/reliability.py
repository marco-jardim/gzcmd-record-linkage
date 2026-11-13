"""Measured per-band reviewer reliability (explicit fixed fallback for n<10).

Mirrors the thesis ``gzcmd/reliability.py``: rates come from the ``by_mode``
blocks of an estimated-rates YAML (e.g. ``error_rates_by_band_v4.yaml``) and
fall back to the fixed inline proportions for bands with fewer than 10
decisions or undefined estimates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from .policy_engine import LLMError

_HAIKU = "claude-haiku-5-5"


def reference_rates_model(source: str | Path | None) -> str:
    """Pick the reviewer block; v4's pooled block mixes in a nondiscriminating
    model, so v4 / sem_mo sources select Claude Haiku 5.5 explicitly."""
    if source is None:
        return "pooled"
    data = yaml.safe_load(Path(source).read_text(encoding="utf-8"))
    if data.get("prompt") == "v4" and data.get("dossie") == "sem_mo":
        blocks = data["by_mode"].values()
        if not all(_HAIKU in block.get("models", {}) for block in blocks):
            raise ValueError(
                "v4 reference requires the Haiku model block in every mode"
            )
        return _HAIKU
    return "pooled"


def load_band_rates(
    fixed: dict[str, dict[str, float]],
    *,
    mode: str,
    source: str | Path | None = None,
    model: str | None = None,
) -> dict[str, LLMError]:
    """Per-band reviewer rates for ``mode``.

    A band uses the measured estimate only when it has at least 10 decisions;
    otherwise ``fixed`` supplies ``e_fp``/``e_fn``/``e_unsure`` (``e_unsure``
    defaults to 0.0 there, not to the mode-level UNSURE rate). Within an
    eligible band, an undefined ``e_fp``/``e_fn`` (empty denominator) or a
    missing ``u`` falls back to ``fixed`` as well.
    """
    block: dict = {}
    if source is not None:
        data = yaml.safe_load(Path(source).read_text(encoding="utf-8"))
        model = model or reference_rates_model(source)
        by_mode = data["by_mode"]
        if mode not in by_mode:
            raise ValueError(f"Missing mode block in {source}: {mode}")
        mode_block = by_mode[mode]
        if model == "pooled":
            block = mode_block["pooled"]
        else:
            if model not in mode_block.get("models", {}):
                raise ValueError(f"Missing explicit reviewer block: {mode}/{model}")
            block = mode_block["models"][model]
    estimates = block.get("estimates_by_band", {})
    engine_rates = block.get("error_rates_by_band", {})
    result: dict[str, LLMError] = {}
    for band, fallback in fixed.items():
        estimate = estimates.get(band, {})
        enough = int(estimate.get("n", 0)) >= 10
        rates = engine_rates.get(band, {}) if enough else {}
        fp = rates.get("e_fp", fallback["e_fp"])
        fn = rates.get("e_fn", fallback["e_fn"])
        u = estimate.get("u") if enough else None
        values = [
            fallback["e_fp"] if fp is None else fp,
            fallback["e_fn"] if fn is None else fn,
            fallback.get("e_unsure", 0.0) if u is None else u,
        ]
        if not all(np.isfinite(v) and 0 <= v <= 1 for v in values):
            raise ValueError(f"Invalid reviewer probabilities: {mode}/{model}/{band}")
        result[band] = LLMError(*map(float, values))
    return result
