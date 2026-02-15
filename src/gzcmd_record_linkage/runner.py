from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .bands import BandAssigner
from .calibration import (
    PlattModel,
    compute_p_cal,
    fit_platt_from_df,
    get_nota_series,
    load_platt_model,
    save_platt_model,
)
from .classifier import GZCMDClassifier
from .config import GZCMDConfig, load_config
from .guardrails import apply_guardrails
from .loader import LoadConfig, load_comparador_csv
from .policy_engine import Budget, Costs, LLMError, PolicyEngineV3


@dataclass(frozen=True)
class RunSummary:
    rows: int
    llm_used: int
    llm_max: int
    actions: dict[str, int]
    guardrails: dict[str, int]
    review_requested: int
    p_cal_method: str
    p_cal_params: dict[str, float] | None


def build_engine_from_config(
    cfg: GZCMDConfig,
    *,
    mode: str,
    llm_used: int = 0,
) -> PolicyEngineV3:
    if mode not in cfg.decision_policy.modes:
        known = ", ".join(sorted(cfg.decision_policy.modes.keys()))
        raise KeyError(f"Unknown mode '{mode}'. Known modes: {known}")

    mode_cfg = cfg.decision_policy.modes[mode]

    costs = Costs(
        false_positive=float(mode_cfg.false_positive_cost),
        false_negative=float(mode_cfg.false_negative_cost),
        llm_review=float(mode_cfg.llm_review_cost),
    )

    if cfg.llm_review.enabled:
        missing: list[str] = []
        llm_error_by_band: dict[str, LLMError] = {}
        for band_def in cfg.bands.definitions:
            band = band_def.name
            rates = cfg.llm_review.error_rates_by_band.get(band)
            if rates is None:
                missing.append(band)
                continue
            llm_error_by_band[band] = LLMError(
                e_fp=float(rates["e_fp"]),
                e_fn=float(rates["e_fn"]),
            )

        if missing:
            raise ValueError(
                "Missing llm_review.reliability.error_rates_by_band for bands: "
                + ", ".join(missing)
            )

        budget = Budget(
            llm_max=int(mode_cfg.llm_max_calls_per_window),
            llm_used=int(llm_used),
        )
    else:
        llm_error_by_band = {
            band_def.name: LLMError(e_fp=0.0, e_fn=0.0)
            for band_def in cfg.bands.definitions
        }
        budget = Budget(llm_max=0, llm_used=int(llm_used))

    return PolicyEngineV3(
        costs=costs,
        llm_error_by_band=llm_error_by_band,
        budget=budget,
        min_auto_match=mode_cfg.min_auto_match_threshold,
        max_auto_nonmatch=mode_cfg.max_auto_nonmatch_threshold,
    )


def run_v3(
    *,
    input_csv: str | Path,
    config_path: str | Path,
    mode: str,
    macd_enabled: bool = True,
    llm_used: int = 0,
    p_cal: str = "fit_platt",
    platt_model_path: str | Path | None = None,
    save_platt_model_path: str | Path | None = None,
    ml_rf_model_path: str | Path | None = None,
    save_ml_rf_model_path: str | Path | None = None,
) -> tuple[pd.DataFrame, RunSummary]:
    cfg = load_config(config_path)
    df = load_comparador_csv(Path(input_csv), cfg=LoadConfig(macd_enabled=macd_enabled))

    nota_raw = pd.to_numeric(get_nota_series(df), errors="coerce")
    if isinstance(nota_raw, pd.Series):
        nota = nota_raw
    else:  # pragma: no cover (pd.to_numeric returns Series for Series input)
        nota = pd.Series(nota_raw, index=df.index)

    band_assigner = BandAssigner.from_config(cfg)
    df["band"] = band_assigner.assign(nota)
    if df["band"].isna().to_numpy(dtype=bool).any():
        n_bad = int(df["band"].isna().sum())
        raise ValueError(
            f"Unable to assign band for {n_bad} rows (nota final out of range?)."
        )

    p_cal_params: dict[str, float] | None = None
    clip_min = float(cfg.calibration.clip_min)
    clip_max = float(cfg.calibration.clip_max)
    if p_cal == "stub":
        df["p_cal"] = compute_p_cal(
            df, method="stub", clip_min=clip_min, clip_max=clip_max
        )
    elif p_cal == "fit_platt":
        model = fit_platt_from_df(
            df,
            l2=cfg.calibration.platt.l2,
            max_iter=cfg.calibration.platt.max_iter,
            tol=cfg.calibration.platt.tol,
        )
        p_cal_params = {
            "intercept": float(model.intercept),
            "slope": float(model.slope),
        }
        df["p_cal"] = compute_p_cal(
            df, method="platt", model=model, clip_min=clip_min, clip_max=clip_max
        )
        if save_platt_model_path is not None:
            save_platt_model(model, save_platt_model_path)
    elif p_cal == "load_platt":
        if platt_model_path is None:
            raise ValueError("platt_model_path is required when p_cal='load_platt'")
        model = load_platt_model(platt_model_path)
        p_cal_params = {
            "intercept": float(model.intercept),
            "slope": float(model.slope),
        }
        df["p_cal"] = compute_p_cal(
            df, method="platt", model=model, clip_min=clip_min, clip_max=clip_max
        )
    elif p_cal == "fit_ml_rf":
        clf = GZCMDClassifier()
        clf.fit(df)
        probs = clf.predict_proba(df)[:, 1]
        df["p_cal"] = probs
        if save_ml_rf_model_path is not None:
            clf.save(save_ml_rf_model_path)
    elif p_cal == "load_ml_rf":
        if ml_rf_model_path is None:
            raise ValueError("ml_rf_model_path is required when p_cal='load_ml_rf'")
        clf = GZCMDClassifier.load(ml_rf_model_path)
        probs = clf.predict_proba(df)[:, 1]
        df["p_cal"] = probs
    else:
        raise ValueError("p_cal must be one of: stub, fit_platt, load_platt, fit_ml_rf, load_ml_rf")

    g = apply_guardrails(
        df,
        temporal_days=cfg.guardrails.temporal_days,
        nota_always_match=cfg.guardrails.nota_always_match,
        nota_always_nonmatch=cfg.guardrails.nota_always_nonmatch,
        homonimia_min_nota=cfg.guardrails.homonimia_min_nota,
        homonimia_year_gap=cfg.guardrails.homonimia_year_gap,
    )
    df["guardrail"] = g.guardrail
    df["guardrail_reason"] = g.reason

    engine = build_engine_from_config(cfg, mode=mode, llm_used=llm_used)
    out = engine.triage(df)

    actions = out["action"].value_counts(dropna=False).to_dict()
    actions = {str(k): int(v) for k, v in actions.items()}

    guardrails = out["guardrail"].value_counts(dropna=False).to_dict()
    guardrails = {str(k): int(v) for k, v in guardrails.items()}

    review_requested = (
        int(out["review_requested"].sum()) if "review_requested" in out.columns else 0
    )

    summary = RunSummary(
        rows=int(len(out)),
        llm_used=int(engine.budget.llm_used),
        llm_max=int(engine.budget.llm_max),
        actions=actions,
        guardrails=guardrails,
        review_requested=review_requested,
        p_cal_method=str(p_cal),
        p_cal_params=p_cal_params,
    )
    return out, summary
