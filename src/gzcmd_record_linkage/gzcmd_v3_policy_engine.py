"""GZ-CMD++ v3 — Policy engine (expected-loss triage with LLM budget).

This module is intentionally self-contained and deterministic.
It assumes you already computed:
  - p_cal: calibrated probability of MATCH
  - band: one of the configured bands (e.g., 'grey_low', 'grey_mid', ...)
  - guardrail actions (optional): ALWAYS_MATCH / ALWAYS_NONMATCH / FORCE_REVIEW
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Tuple

import numpy as np
import pandas as pd

Decision = Literal["MATCH", "NONMATCH", "LLM_REVIEW"]


@dataclass(frozen=True)
class Costs:
    false_positive: float
    false_negative: float
    llm_review: float


@dataclass(frozen=True)
class LLMError:
    e_fp: float  # P(LLM says MATCH | true is NONMATCH) among reviewed
    e_fn: float  # P(LLM says NONMATCH | true is MATCH) among reviewed


@dataclass
class Budget:
    llm_max: int
    llm_used: int = 0


class PolicyEngineV3:
    def __init__(
        self,
        costs: Costs,
        llm_error_by_band: Dict[str, LLMError],
        budget: Budget,
        min_auto_match: float | None = None,
        max_auto_nonmatch: float | None = None,
    ) -> None:
        self.costs = costs
        self.llm_error_by_band = llm_error_by_band
        self.budget = budget
        self.min_auto_match = min_auto_match
        self.max_auto_nonmatch = max_auto_nonmatch

    def _losses(
        self, p: np.ndarray, bands: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        c = self.costs
        loss_match = (1.0 - p) * c.false_positive
        loss_non = p * c.false_negative

        e_fp = np.array([self.llm_error_by_band[b].e_fp for b in bands], dtype=float)
        e_fn = np.array([self.llm_error_by_band[b].e_fn for b in bands], dtype=float)

        loss_llm = (
            c.llm_review
            + (1.0 - p) * e_fp * c.false_positive
            + p * e_fn * c.false_negative
        )
        return loss_match, loss_non, loss_llm

    def triage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns df with columns: action, evr, base_choice, base_loss, loss_llm.

        Expects columns:
          - p_cal (float)
          - band (str)
          - guardrail (optional): 'ALWAYS_MATCH'|'ALWAYS_NONMATCH'|'FORCE_REVIEW'|None
        """
        out = df.copy()

        p = out["p_cal"].astype(float).to_numpy()
        bands = out["band"].astype(object).to_numpy()

        loss_match, loss_non, loss_llm = self._losses(p, bands)
        base_choice = np.where(loss_match <= loss_non, "MATCH", "NONMATCH").astype(
            object
        )
        base_loss = np.minimum(loss_match, loss_non)

        evr = base_loss - loss_llm  # expected value of reviewing

        out["base_choice"] = base_choice
        out["base_loss"] = base_loss
        out["loss_llm"] = loss_llm
        out["evr"] = evr

        # Default action: base choice
        out["action"] = out["base_choice"].astype(object)

        # Optional auto-decision caps (do not mutate probabilities).
        # Caps mark rows that should be reviewed if budget allows.
        cap_match_mask = np.zeros(len(out), dtype=bool)
        cap_non_mask = np.zeros(len(out), dtype=bool)
        if self.min_auto_match is not None:
            cap_match_mask = (base_choice == "MATCH") & (p < self.min_auto_match)
        if self.max_auto_nonmatch is not None:
            cap_non_mask = (base_choice == "NONMATCH") & (p > self.max_auto_nonmatch)
        cap_mask = cap_match_mask | cap_non_mask

        # Guardrails
        always_match_mask = np.zeros(len(out), dtype=bool)
        always_non_mask = np.zeros(len(out), dtype=bool)
        force_review_mask = np.zeros(len(out), dtype=bool)
        if "guardrail" in out.columns:
            guard_s = out["guardrail"]
            always_match_mask = guard_s.eq("ALWAYS_MATCH").fillna(False).to_numpy()
            always_non_mask = guard_s.eq("ALWAYS_NONMATCH").fillna(False).to_numpy()
            force_review_mask = guard_s.eq("FORCE_REVIEW").fillna(False).to_numpy()
            out.loc[always_match_mask, "action"] = "MATCH"
            out.loc[always_non_mask, "action"] = "NONMATCH"

        # Flag rows that requested review (budget may not cover all).
        review_requested = (evr > 0) | cap_mask | force_review_mask
        out["review_requested"] = review_requested.astype(bool)

        # Choose LLM review candidates within remaining budget.
        # Priority: FORCE_REVIEW first, then cap-triggered, then EVR>0.
        remaining = max(0, self.budget.llm_max - self.budget.llm_used)
        selected = np.zeros(len(out), dtype=bool)
        deterministic_mask = always_match_mask | always_non_mask

        def _select(mask: np.ndarray) -> None:
            nonlocal remaining, selected
            if remaining <= 0:
                return
            eligible = mask & ~selected & ~deterministic_mask
            if not eligible.any():
                return
            chosen = (
                out.loc[eligible]
                .sort_values("evr", ascending=False)
                .head(remaining)
                .index
            )
            selected[out.index.get_indexer(chosen)] = True
            remaining -= len(chosen)

        _select(force_review_mask)
        _select(cap_mask)
        _select(evr > 0)

        out.loc[selected, "action"] = "LLM_REVIEW"
        self.budget.llm_used += int(selected.sum())

        return out
