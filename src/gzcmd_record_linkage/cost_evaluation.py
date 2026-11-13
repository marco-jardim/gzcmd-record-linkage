"""Prospective pooled expectations; never substitutes simulated LLM responses."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .policy_engine import HumanError, LLMError


def expected_funnel_metrics(
    out: pd.DataFrame, rates: dict[str, LLMError], human_error: HumanError
) -> dict:
    """Gold labels evaluate recall, calibrated p evaluates expected decision loss.

    M3 u is marginal. Approximate its class-conditional value by that marginal,
    capped at 1-e_fn / 1-e_fp to keep the three outcomes coherent per class.
    """
    y = out["TARGET"] if "TARGET" in out else out["y"]
    positive = y.eq(1)
    llm = out.action.eq("LLM_REVIEW")
    human = out.action.eq("DEFERRED")
    match, non = out.action.eq("MATCH"), out.action.eq("NONMATCH")
    guard = out.guardrail.fillna("none")
    fp = out.band.map({b: r.e_fp for b, r in rates.items()}).astype(float)
    fn = out.band.map({b: r.e_fn for b, r in rates.items()}).astype(float)
    u = out.band.map({b: r.e_unsure for b, r in rates.items()}).astype(float)
    if fp.isna().any() or fn.isna().any() or u.isna().any():
        raise ValueError("Missing reliability band")
    u_pos = np.minimum(u, 1 - fn)
    u_neg = np.minimum(u, 1 - fp)
    auto_tp = int((match & positive).sum())
    expected_tp = auto_tp + float((1 - fn - u_pos)[llm & positive].sum())
    expected_tp_human = (
        expected_tp
        + int((human & positive).sum()) * (1 - human_error.e_fn)
        + float(u_pos[llm & positive].sum()) * (1 - human_error.e_fn)
    )
    auto_match = match & ~guard.eq("ALWAYS_MATCH")
    auto_non = non & ~guard.eq("ALWAYS_NONMATCH")
    negative_p = 1 - out.p_cal
    fp_auto_expected = float(negative_p[match].sum())
    fp_total_expected = (
        fp_auto_expected
        + float((negative_p * fp)[llm].sum())
        + float((negative_p * human_error.e_fp)[human].sum())
        + float((negative_p * u_neg * human_error.e_fp)[llm].sum())
    )
    score = out["nota final"] if "nota final" in out else out["nota"]
    n_pos = int(positive.sum())
    return {
        "lote_llm": int(llm.sum()),
        "humano_direto": int(human.sum()),
        "humano_via_llm_esperado": float(u[llm].sum()),
        "humano_total_esperado": int(human.sum()) + float(u[llm].sum()),
        "AUTO_MATCH": int(auto_match.sum()),
        "AUTO_NONMATCH": int(auto_non.sum()),
        "AUTO_MATCH_vinculos": int((auto_match & positive).sum()),
        "AUTO_MATCH_FP": int((auto_match & ~positive).sum()),
        "AUTO_NONMATCH_vinculos": int((auto_non & positive).sum()),
        "ALWAYS_MATCH": int(guard.eq("ALWAYS_MATCH").sum()),
        "ALWAYS_NONMATCH": int(guard.eq("ALWAYS_NONMATCH").sum()),
        "vinculos_perdidos_auto_nonmatch": int((auto_non & positive).sum()),
        "vinculos_perdidos_cinza": int(
            (auto_non & positive & score.ge(5) & score.lt(9)).sum()
        ),
        "vinculos_perdidos_total_nonmatch": int((non & positive).sum()),
        "FP_automaticos": int((match & ~positive).sum()),
        "FP_auto_match": int((auto_match & ~positive).sum()),
        "fp_esperado_auto_match_incl_guardas": fp_auto_expected,
        "taxa_fp_esperada_auto_match": (
            fp_auto_expected / int(match.sum()) if match.any() else 0.0
        ),
        "fp_esperado_total": fp_total_expected,
        "vinculos_esperados_auto_match_incl_guardas": float(out.p_cal[match].sum()),
        "sens_esperada_automatica": expected_tp / n_pos if n_pos else np.nan,
        "sens_esperada_com_humano": expected_tp_human / n_pos if n_pos else np.nan,
        "perda_esperada_total": float(out.selected_loss.sum()),
    }
