"""Safe pooled comparison tables, including explicit infeasible NP scenarios.

Only the generic comparison layer of the thesis is ported. The thesis'
``evaluate_birth_variant`` hard-codes thesis outcomes (``recuperados_dos_17``,
``adotada: False``) and is intentionally not part of the library; use
:func:`gzcmd_record_linkage.guardrails.high_guard_birth_variant` directly.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .cost_evaluation import expected_funnel_metrics
from .np_policy import NPInfeasibleError
from .policy_engine import (
    Budget,
    Costs,
    HumanBudget,
    HumanError,
    LLMError,
    PolicyEngineV3,
)
from .reliability import load_band_rates

NP_RATES = (0.0005, 0.001, 0.002, 0.005, 0.01)


def evaluate_np_point(
    frame: pd.DataFrame,
    cfg,
    rates: dict[str, LLMError],
    *,
    alpha: float,
    error: float | None = None,
    cap="config",
    kind: str = "np",
):
    """Triage ``frame`` under the ``confirmacao`` mode at one operating point.

    Returns ``(row, out)``: ``row`` is a flat dict (settings, feasibility and,
    when feasible, the expected funnel metrics plus the engine's NP diagnostics)
    and ``out`` is the triaged frame, or ``None`` when the NP constraint is
    infeasible (the diagnostics of ``NPInfeasibleError`` are put in ``row``;
    no metric is fabricated).

    ``alpha`` overrides ``max_fp_rate_auto``; ``error`` replaces both human error
    rates; ``cap="config"`` keeps the configured human budget, otherwise ``cap``
    is the budget (``None`` = unlimited). ``kind`` is ``"np"`` or ``"cost"``.

    The engine is built directly (like the thesis) rather than through
    ``runner.build_engine_from_config`` because that function does not accept
    injected ``rates`` or per-call human error/budget overrides.
    """
    c = cfg.decision_policy.modes["confirmacao"]
    human = (
        HumanError(c.human_e_fp, c.human_e_fn)
        if error is None
        else HumanError(error, error)
    )
    budget = c.human_max_pairs_per_window if cap == "config" else cap
    constraints = replace(c.np_policy, max_fp_rate_auto=alpha)
    row = {
        "modo": "confirmacao",
        "policy_kind": kind,
        "max_fp_rate_auto": alpha if kind == "np" else np.nan,
        "max_fp_expected_total": constraints.max_fp_expected_total,
        "max_fn_rate_auto": constraints.max_fn_rate_auto,
        "C_llm": c.llm_review_cost,
        "e_fp_h": human.e_fp,
        "e_fn_h": human.e_fn,
        "orcamento_humano": np.inf if budget is None else budget,
    }
    engine = PolicyEngineV3(
        Costs(
            c.false_positive_cost,
            c.false_negative_cost,
            c.llm_review_cost,
            c.human_review_cost,
        ),
        rates,
        Budget(c.llm_max_calls_per_window),
        policy_version="v3.1",
        policy_kind=kind,
        human_error=human,
        human_budget=HumanBudget(budget),
        np_constraints=constraints,
    )
    try:
        out = engine.triage(frame)
    except NPInfeasibleError as exc:
        row.update({"viavel": False, "status": "inviavel", **exc.diagnostics})
        return row, None
    row.update(
        {
            "viavel": True,
            "status": "viavel" if kind == "np" else "custo_sem_restricao",
            **expected_funnel_metrics(out, rates, human),
            **engine.np_diagnostics,
        }
    )
    return row, out


def compare_confirmation(frame: pd.DataFrame, cfg, rates_path) -> pd.DataFrame:
    """Cost policy at the configured alpha plus NP at alpha .001/.002/.01."""
    rates = load_band_rates(
        cfg.llm_review.error_rates_by_band, mode="confirmacao", source=rates_path
    )
    configured = cfg.decision_policy.modes["confirmacao"].np_policy.max_fp_rate_auto
    rows = []
    for kind, alpha in (
        ("cost", configured),
        ("np", 0.001),
        ("np", 0.002),
        ("np", 0.01),
    ):
        row, _ = evaluate_np_point(frame, cfg, rates, alpha=alpha, kind=kind)
        rows.append(row)
    return pd.DataFrame(rows)
