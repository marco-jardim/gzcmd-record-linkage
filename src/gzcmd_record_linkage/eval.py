from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence, cast

import numpy as np
import pandas as pd
from pandas import Series

from .bands import BandAssigner
from .calibration import (
    PlattModel,
    fit_platt,
    get_nota_series,
    predict_platt,
    predict_stub,
)
from .classifier import GZCMDClassifier
from .config import GZCMDConfig, load_config
from .guardrails import apply_guardrails
from .loader import LoadConfig, load_comparador_csv
from .metrics import ConfusionCounts, confusion_counts, metrics_dict
from .runner import build_engine_from_config
from .splitting import SplitBy, SplitSpec, split_train_test_indices


CalibrationMethod = Literal["platt", "stub", "ml_rf"]


@dataclass(frozen=True)
class EvalRun:
    mode: str
    split_by: str
    seed: int
    test_size: float
    group_stratify: bool
    macd_enabled: bool
    calibration: CalibrationMethod


def _beta_for_mode(mode: str) -> float:
    m = str(mode).strip().lower()
    if m == "vigilancia":
        return 2.0
    if m == "confirmacao":
        return 0.5
    return 1.0


def _coerce_nota(df: pd.DataFrame) -> Series:
    raw = pd.to_numeric(get_nota_series(df), errors="coerce")
    if isinstance(raw, Series):
        return raw
    return Series(raw, index=df.index)


def _expected_counts_after_llm(
    out: pd.DataFrame,
    y_true: np.ndarray,
    *,
    error_rates_by_band: dict[str, dict[str, float]],
) -> ConfusionCounts:
    action = out["action"].astype(str).to_numpy()
    band = out["band"].astype(str).to_numpy()
    yt = np.asarray(y_true).astype(np.int8)

    if yt.shape[0] != action.shape[0]:
        raise ValueError("y_true must align with out rows")

    match_mask = action == "MATCH"
    non_mask = action == "NONMATCH"
    review_mask = action == "LLM_REVIEW"

    tp = float(np.sum(match_mask & (yt == 1)))
    fp = float(np.sum(match_mask & (yt == 0)))
    fn = float(np.sum(non_mask & (yt == 1)))
    tn = float(np.sum(non_mask & (yt == 0)))

    if not np.any(review_mask):
        return ConfusionCounts(tp=tp, fp=fp, fn=fn, tn=tn)

    rb = band[review_mask]
    ry = yt[review_mask]
    for b in np.unique(rb):
        rates = error_rates_by_band.get(str(b))
        if rates is None:
            raise KeyError(f"Missing llm error rates for band '{b}'")
        e_fp = float(rates["e_fp"])
        e_fn = float(rates["e_fn"])

        mask_b = rb == b
        yb = ry[mask_b]
        n_pos = float(np.sum(yb == 1))
        n_neg = float(np.sum(yb == 0))

        tp += n_pos * (1.0 - e_fn)
        fn += n_pos * e_fn
        fp += n_neg * e_fp
        tn += n_neg * (1.0 - e_fp)

    return ConfusionCounts(tp=tp, fp=fp, fn=fn, tn=tn)


def _pred_from_action(out: pd.DataFrame) -> pd.Series:
    pred = pd.Series(pd.NA, index=out.index, dtype="Int64")
    pred[out["action"].eq("MATCH")] = 1
    pred[out["action"].eq("NONMATCH")] = 0
    return pred


def _ensure_binary_target(df: pd.DataFrame, *, col: str = "TARGET") -> pd.Series:
    if col not in df.columns:
        raise KeyError(f"Missing target column '{col}'")
    y_raw = pd.to_numeric(df[col], errors="coerce")
    if isinstance(y_raw, pd.Series):
        y = y_raw
    else:  # pragma: no cover
        y = pd.Series(y_raw, index=df.index)

    if y.isna().to_numpy(dtype=bool).any():
        n_bad = int(y.isna().sum())
        raise ValueError(f"Found {n_bad} NA values in {col}")
    y_int = y.astype("int64")
    unique = set(y_int.unique().tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(f"{col} must be binary in {{0,1}}; got {sorted(unique)}")
    return y_int


def evaluate_v3_dataframe(
    df: pd.DataFrame,
    *,
    cfg: GZCMDConfig,
    modes: Sequence[str],
    split_by: SplitBy,
    seeds: Sequence[int],
    test_size: float,
    group_stratify: bool,
    calibration: CalibrationMethod,
    macd_enabled: bool,
    guardrails_enabled: bool = True,
) -> pd.DataFrame:
    work = df.copy()
    y_all = _ensure_binary_target(work)

    nota = _coerce_nota(work)

    if nota.isna().to_numpy(dtype=bool).any():
        n_bad = int(nota.isna().sum())
        raise ValueError(f"Found {n_bad} rows with missing/invalid nota final")

    band_assigner = BandAssigner.from_config(cfg)
    work["band"] = band_assigner.assign(nota)
    if work["band"].isna().to_numpy(dtype=bool).any():
        n_bad = int(work["band"].isna().sum())
        raise ValueError(f"Unable to assign band for {n_bad} rows")

    if guardrails_enabled:
        g = apply_guardrails(
            work,
            temporal_days=cfg.guardrails.temporal_days,
            nota_always_match=cfg.guardrails.nota_always_match,
            nota_always_nonmatch=cfg.guardrails.nota_always_nonmatch,
            homonimia_min_nota=cfg.guardrails.homonimia_min_nota,
            homonimia_year_gap=cfg.guardrails.homonimia_year_gap,
        )
        work["guardrail"] = g.guardrail
    else:
        work["guardrail"] = pd.NA

    records: list[dict[str, float | int | str | bool]] = []
    for mode in modes:
        beta = _beta_for_mode(mode)
        for seed in seeds:
            spec = SplitSpec(
                split_by=split_by,
                test_size=test_size,
                seed=int(seed),
                group_stratify=group_stratify,
            )
            train_idx, test_idx = split_train_test_indices(work, y_all, spec=spec)
            train = work.iloc[train_idx]
            test = work.iloc[test_idx].copy()

            nota_train = _coerce_nota(train)
            y_train = _ensure_binary_target(train)
            nota_test = _coerce_nota(test)

            model: PlattModel | None = None
            if calibration == "platt":
                model = fit_platt(
                    nota_train,
                    y_train,
                    l2=cfg.calibration.platt.l2,
                    max_iter=cfg.calibration.platt.max_iter,
                    tol=cfg.calibration.platt.tol,
                )
                test["p_cal"] = predict_platt(
                    nota_test,
                    model=model,
                    clip_min=cfg.calibration.clip_min,
                    clip_max=cfg.calibration.clip_max,
                )
            elif calibration == "stub":
                test["p_cal"] = predict_stub(
                    nota_test,
                    clip_min=cfg.calibration.clip_min,
                    clip_max=cfg.calibration.clip_max,
                )
            elif calibration == "ml_rf":
                clf = GZCMDClassifier()
                clf.fit(train)
                probs = clf.predict_proba(test)[:, 1]
                test["p_cal"] = probs
            else:
                raise ValueError("calibration must be one of: platt, stub, ml_rf")

            engine = build_engine_from_config(cfg, mode=str(mode), llm_used=0)
            out = engine.triage(test)

            yt = _ensure_binary_target(out).to_numpy(dtype=np.int8)

            pred = _pred_from_action(out)
            decided_mask = np.asarray(pred.notna(), dtype=bool)
            n_test = int(len(out))
            n_decided = int(np.sum(decided_mask))
            auto_coverage = float(n_decided / n_test) if n_test else 0.0

            # Auto-only metrics are computed on decided subset.
            auto_pred = np.asarray(pred[decided_mask], dtype=np.int8)
            auto_counts = confusion_counts(yt[decided_mask], auto_pred)

            # Oracle: reviewed rows are always correct.
            oracle_pred = pred.copy()
            review_mask = np.asarray(out["action"].eq("LLM_REVIEW"), dtype=bool)
            oracle_pred.loc[review_mask] = yt[review_mask]
            oracle_counts = confusion_counts(yt, oracle_pred.to_numpy(dtype=np.int8))

            exp_counts = _expected_counts_after_llm(
                out, yt, error_rates_by_band=cfg.llm_review.error_rates_by_band
            )

            record: dict[str, float | int | str | bool] = {
                "mode": str(mode),
                "beta": float(beta),
                "split_by": str(split_by),
                "seed": int(seed),
                "test_size": float(test_size),
                "group_stratify": bool(group_stratify),
                "macd_enabled": bool(macd_enabled),
                "guardrails_enabled": bool(guardrails_enabled),
                "calibration": str(calibration),
                "n_train": int(len(train)),
                "n_test": n_test,
                "pos_train": int(y_train.sum()),
                "pos_test": int(np.sum(yt == 1)),
                "llm_max": int(engine.budget.llm_max),
                "llm_used": int(engine.budget.llm_used),
                "review_requested": (
                    int(out["review_requested"].sum())
                    if "review_requested" in out.columns
                    else 0
                ),
                "review_selected": int(np.sum(review_mask)),
                "auto_coverage": auto_coverage,
            }

            record.update(metrics_dict(auto_counts, beta=beta, prefix="auto_"))
            record.update(metrics_dict(oracle_counts, beta=beta, prefix="oracle_"))
            record.update(metrics_dict(exp_counts, beta=beta, prefix="exp_"))
            if model is not None:
                record["platt_intercept"] = float(model.intercept)
                record["platt_slope"] = float(model.slope)

            records.append(record)

    return pd.DataFrame(records)


def evaluate_v3_csv(
    *,
    input_csv: str | Path,
    config_path: str | Path,
    modes: Sequence[str] | None,
    split_by: str,
    seeds: Sequence[int],
    test_size: float,
    group_stratify: bool,
    calibration: CalibrationMethod,
    macd_enabled: bool,
    guardrails_enabled: bool = True,
) -> pd.DataFrame:
    cfg = load_config(config_path)
    df = load_comparador_csv(Path(input_csv), cfg=LoadConfig(macd_enabled=macd_enabled))

    if modes is None:
        modes_to_run = list(cfg.decision_policy.modes.keys())
    else:
        modes_to_run = [str(m) for m in modes]

    split_norm = str(split_by).strip().lower()
    if split_norm not in ("row", "comprec", "refrec"):
        raise ValueError("split_by must be one of: row, comprec, refrec")
    split_typed = cast(SplitBy, split_norm)

    return evaluate_v3_dataframe(
        df,
        cfg=cfg,
        modes=modes_to_run,
        split_by=split_typed,
        seeds=seeds,
        test_size=test_size,
        group_stratify=group_stratify,
        calibration=calibration,
        macd_enabled=macd_enabled,
        guardrails_enabled=guardrails_enabled,
    )
