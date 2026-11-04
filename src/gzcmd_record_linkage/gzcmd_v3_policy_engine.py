"""GZ-CMD++ v3 — Policy engine (expected-loss triage with LLM budget).

This module is intentionally self-contained and deterministic.
It assumes you already computed:
  - p_cal: calibrated probability of MATCH
  - band: one of the configured bands (e.g., 'grey_low', 'grey_mid', ...)
  - guardrail actions (optional): ALWAYS_MATCH / ALWAYS_NONMATCH / FORCE_REVIEW
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np
import pandas as pd

Decision = Literal["MATCH", "NONMATCH", "LLM_REVIEW", "DEFERRED"]


@dataclass(frozen=True, init=False)
class Costs:
    false_positive: float
    false_negative: float
    llm_review: float
    # Keep the historical three-field dataclass serialization used by notebooks.
    _human_review: ClassVar[float] = 1.0

    def __init__(
        self,
        false_positive: float,
        false_negative: float,
        llm_review: float,
        human_review: float = 1.0,
    ) -> None:
        object.__setattr__(self, "false_positive", false_positive)
        object.__setattr__(self, "false_negative", false_negative)
        object.__setattr__(self, "llm_review", llm_review)
        object.__setattr__(self, "_human_review", human_review)

    @property
    def human_review(self) -> float:
        return self._human_review

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Costs):
            return NotImplemented
        return (
            self.false_positive,
            self.false_negative,
            self.llm_review,
            self.human_review,
        ) == (
            other.false_positive,
            other.false_negative,
            other.llm_review,
            other.human_review,
        )

    def total_loss(
        self,
        *,
        fp: int,
        fn: int,
        unsure: int,
        deferred: int,
        expected_human_error_loss: float = 0.0,
    ) -> float:
        """Observed decision loss; LLM API cost is reported separately from M6."""
        return (
            self.false_positive * fp
            + self.false_negative * fn
            + self.human_review * (unsure + deferred)
            + expected_human_error_loss
        )


@dataclass(frozen=True)
class HumanError:
    e_fp: float = 0.05
    e_fn: float = 0.05

    def __post_init__(self) -> None:
        if not all(np.isfinite(v) and 0 <= v <= 1 for v in (self.e_fp, self.e_fn)):
            raise ValueError("Human error rates must be finite probabilities")

    def error_loss(self, p, costs: Costs):
        return (
            p * self.e_fn * costs.false_negative
            + (1 - p) * self.e_fp * costs.false_positive
        )

    def loss(self, p, costs: Costs):
        return costs.human_review + self.error_loss(p, costs)


@dataclass
class HumanBudget:
    max_pairs: int | None = None
    used: int = 0

    def __post_init__(self) -> None:
        if self.max_pairs is not None and (
            not isinstance(self.max_pairs, int)
            or isinstance(self.max_pairs, bool)
            or self.max_pairs < 0
        ):
            raise ValueError("Human budget must be a nonnegative integer or None")
        if (
            not isinstance(self.used, int)
            or isinstance(self.used, bool)
            or self.used < 0
        ):
            raise ValueError("Human used must be a nonnegative integer")


@dataclass(frozen=True)
class LLMError:
    e_fp: float  # P(LLM says MATCH | true is NONMATCH) among reviewed
    e_fn: float  # P(LLM says NONMATCH | true is MATCH) among reviewed
    e_unsure: float = 0.0


@dataclass
class Budget:
    llm_max: int
    llm_used: int = 0


class PolicyEngineV3:
    def __init__(
        self,
        costs: Costs,
        llm_error_by_band: dict[str, LLMError],
        budget: Budget,
        min_auto_match: float | None = None,
        max_auto_nonmatch: float | None = None,
        policy_version: str = "v3",
        human_error: HumanError | None = None,
        human_budget: HumanBudget | None = None,
    ) -> None:
        self.costs = costs
        self.llm_error_by_band = llm_error_by_band
        self.budget = budget
        self.min_auto_match = min_auto_match
        self.max_auto_nonmatch = max_auto_nonmatch
        if policy_version not in {"v3", "v3.1"}:
            raise ValueError("policy_version must be v3 or v3.1")
        self.policy_version = policy_version
        self.human_error = (
            HumanError(0, 0)
            if policy_version == "v3"
            else (human_error or HumanError())
        )
        self.human_budget = human_budget or HumanBudget()

    def _losses(
        self, p: np.ndarray, bands: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        if self.policy_version == "v3.1":
            e_unsure = np.array(
                [self.llm_error_by_band[b].e_unsure for b in bands], dtype=float
            )
            loss_llm = loss_llm + e_unsure * self.human_error.loss(p, c)
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

        if self.policy_version == "v3.1" and not (
            np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        ):
            raise ValueError("p_cal must be finite and within [0,1]")

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

        if self.policy_version == "v3.1":
            out["loss_match"] = loss_match
            out["loss_nonmatch"] = loss_non
            return self._triage_bayes(out)

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

    def _triage_bayes(self, out: pd.DataFrame) -> pd.DataFrame:
        """Argmin over MATCH/NONMATCH/LLM/HUMAN, with explicit guard overrides.

        Ties prefer MATCH, NONMATCH, HUMAN, then LLM (avoid a needless API call).
        Budget is allocated by positive gain over the best non-LLM alternative;
        overflow executes that alternative, not a blind human referral. Legacy
        auto caps and FORCE_REVIEW do not override the four-action argmin.
        """
        score_col = "nota final" if "nota final" in out else "nota_final"
        nota = pd.to_numeric(out[score_col], errors="coerce")
        if not np.isfinite(nota.to_numpy(dtype=float)).all():
            raise ValueError("v3.1 requires a finite nota for every pair")
        if not out.index.is_unique:
            raise ValueError("v3.1 requires a unique row index")
        p = out.p_cal.to_numpy(dtype=float)
        if not (np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()):
            raise ValueError("p_cal must be finite and within [0,1]")
        guard = out.get("guardrail", pd.Series(pd.NA, index=out.index, dtype="string"))
        low = nota.lt(5)
        high_match = nota.ge(9) & guard.eq("ALWAYS_MATCH").fillna(False)
        if (guard.eq("ALWAYS_NONMATCH").fillna(False) & ~low).any():
            raise ValueError("v3.1 ALWAYS_NONMATCH is allowed only below 5")
        review = ~(low | high_match)
        out["grey_zone"] = nota.ge(5) & nota.lt(9)
        out["high_consistency_failed"] = nota.ge(9) & ~high_match
        out["review_area"] = review
        out["loss_human"] = self.human_error.loss(out.p_cal, self.costs)
        alternatives = out[["loss_match", "loss_nonmatch", "loss_human"]].to_numpy(
            dtype=float
        )
        choice = alternatives.argmin(axis=1)
        out["base_choice"] = np.array(["MATCH", "NONMATCH", "DEFERRED"], dtype=object)[
            choice
        ]
        out["base_loss"] = alternatives[np.arange(len(out)), choice]
        out["evr"] = out.base_loss - out.loss_llm
        eligible = review & out.evr.gt(0)
        out["review_requested"] = eligible
        out["action"] = out.base_choice
        out["decision_reason"] = "bayes_argmin"
        out.loc[low, "action"] = "NONMATCH"
        out.loc[high_match, "action"] = "MATCH"
        out.loc[low, "decision_reason"] = "guard_low"
        out.loc[high_match, "decision_reason"] = "guard_high_probability"
        out.loc[eligible, "decision_reason"] = "llm_budget_fallback"
        remaining = max(0, self.budget.llm_max - self.budget.llm_used)
        chosen = (
            out.loc[eligible]
            .sort_values("evr", ascending=False, kind="stable")
            .head(remaining)
            .index
        )
        out.loc[chosen, "action"] = "LLM_REVIEW"
        out.loc[chosen, "decision_reason"] = "llm_positive_gain"
        remaining -= len(chosen)

        # Nominal LLM winners have first claim on the LLM budget. Then allocate
        # direct human slots by gain over the best currently available alternative.
        # A rejected human request can use spare LLM capacity, else auto argmin.
        auto_loss = out[["loss_match", "loss_nonmatch"]].min(axis=1)
        auto_action = np.where(out.loss_match <= out.loss_nonmatch, "MATCH", "NONMATCH")
        unassigned = review & ~out.action.eq("LLM_REVIEW")
        human_alternative = (
            np.minimum(auto_loss, out.loss_llm) if remaining > 0 else auto_loss
        )
        out["human_gain"] = human_alternative - out.loss_human
        human_requested = (
            unassigned & out.loss_human.lt(auto_loss) & out.human_gain.ge(0)
        )
        out["human_requested"] = human_requested
        human_remaining = (
            len(out)
            if self.human_budget.max_pairs is None
            else max(0, self.human_budget.max_pairs - self.human_budget.used)
        )
        human_chosen = (
            out.loc[human_requested]
            .sort_values("human_gain", ascending=False, kind="stable")
            .head(human_remaining)
            .index
        )
        out.loc[unassigned, "action"] = auto_action[unassigned.to_numpy()]
        out.loc[human_chosen, "action"] = "DEFERRED"
        self.human_budget.used += len(human_chosen)
        rejected = human_requested & ~out.index.isin(human_chosen)
        out.loc[rejected, "decision_reason"] = "human_budget_fallback_auto"
        fallback = (
            unassigned & ~out.index.isin(human_chosen) & out.loss_llm.lt(auto_loss)
        )
        out["llm_fallback_gain"] = auto_loss - out.loss_llm
        extra = (
            out.loc[fallback]
            .sort_values("llm_fallback_gain", ascending=False, kind="stable")
            .head(remaining)
            .index
        )
        out.loc[extra, "action"] = "LLM_REVIEW"
        out.loc[extra, "decision_reason"] = "human_budget_fallback_llm"
        out["review_requested"] = eligible | fallback
        self.budget.llm_used += len(chosen) + len(extra)
        out["selected_loss"] = np.where(
            out.action.eq("DEFERRED"),
            out.loss_human,
            np.where(
                out.action.eq("LLM_REVIEW"),
                out.loss_llm,
                np.where(
                    out.action.eq("MATCH"),
                    (1 - out.p_cal) * self.costs.false_positive,
                    out.p_cal * self.costs.false_negative,
                ),
            ),
        )
        return out
