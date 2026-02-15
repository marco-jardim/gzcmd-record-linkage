from __future__ import annotations

import argparse
from importlib.resources import files
from pathlib import Path

from .config import load_config
from .eval import evaluate_v3_csv
from .reporting import summarize_runs, summary_to_latex_table, write_csv, write_text
from .runner import run_v3


def _default_config_path() -> str:
    # Prefer package resource so defaults work from clean installs/sdist/wheels.
    resource = files("gzcmd_record_linkage") / "gzcmd_v3_config.yaml"
    if resource.is_file():
        return str(resource)

    legacy = Path("gzcmd") / "gzcmd_v3_config.yaml"
    if legacy.exists():
        return str(legacy)

    return str(Path("gzcmd_v3_config.yaml"))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gzcmd", description="GZ-CMD++ v3 runner")
    default_config = _default_config_path()
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run v3 policy pipeline")
    run.add_argument(
        "--input",
        default=str(Path("data") / "COMPARADORSEMIDENT.csv"),
        help="Input CSV path (default: data/COMPARADORSEMIDENT.csv)",
    )
    run.add_argument(
        "--config",
        default=default_config,
        help="Config YAML path (default: gzcmd_v3_config.yaml)",
    )
    run.add_argument(
        "--mode",
        required=True,
        help="Decision policy mode (e.g., vigilancia, confirmacao)",
    )
    run.add_argument(
        "--output",
        default="",
        help="Output CSV path (if empty, does not write a file)",
    )
    run.add_argument(
        "--no-macd",
        action="store_true",
        help="Disable MACD feature engineering",
    )
    run.add_argument(
        "--llm-used",
        type=int,
        default=0,
        help="Previously used LLM calls in the current window (default: 0)",
    )
    run.add_argument(
        "--p-cal",
        choices=["fit_platt", "load_platt", "stub", "fit_ml_rf", "load_ml_rf"],
        default="fit_platt",
        help="How to compute p_cal (default: fit_platt)",
    )
    run.add_argument(
        "--platt-model",
        default="",
        help="Path to a saved Platt model (required for --p-cal load_platt)",
    )
    run.add_argument(
        "--save-platt-model",
        default="",
        help="If set, save fitted Platt model to this path",
    )
    run.add_argument(
        "--ml-rf-model",
        default="",
        help="Path to a saved ML RF model (required for --p-cal load_ml_rf)",
    )
    run.add_argument(
        "--save-ml-rf-model",
        default="",
        help="If set, save fitted ML RF model to this path",
    )

    fit = sub.add_parser("fit-calibration", help="Fit and save Platt calibration")
    fit.add_argument(
        "--input",
        default=str(Path("data") / "COMPARADORSEMIDENT.csv"),
        help="Input CSV path (default: data/COMPARADORSEMIDENT.csv)",
    )
    fit.add_argument(
        "--config",
        default=default_config,
        help="Config YAML path (default: gzcmd_v3_config.yaml)",
    )
    fit.add_argument(
        "--output-model",
        required=True,
        help="Where to write the fitted calibration model (JSON)",
    )
    fit.add_argument(
        "--no-macd",
        action="store_true",
        help=(
            "Disable MACD feature engineering "
            "(not used by calibration, but keeps pipeline consistent)"
        ),
    )

    ev = sub.add_parser("eval", help="Evaluate v3 with train/test split")
    ev.add_argument(
        "--input",
        default=str(Path("data") / "COMPARADORSEMIDENT.csv"),
        help="Input CSV path (default: data/COMPARADORSEMIDENT.csv)",
    )
    ev.add_argument(
        "--config",
        default=default_config,
        help="Config YAML path (default: gzcmd_v3_config.yaml)",
    )
    ev.add_argument(
        "--modes",
        nargs="+",
        default=[],
        help="Modes to evaluate (default: all modes in config)",
    )
    ev.add_argument(
        "--split-by",
        choices=["row", "comprec", "refrec"],
        default="",
        help="Split strategy (default: config.evaluation.split_by)",
    )
    ev.add_argument(
        "--test-size",
        type=float,
        default=None,
        help="Test fraction in (0,1) (default: config.evaluation.test_size)",
    )
    ev.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[],
        help="Random seeds (default: config.evaluation.seeds)",
    )
    ev.add_argument(
        "--no-group-stratify",
        action="store_true",
        help="Disable stratifying groups by any-positive label",
    )
    ev.add_argument(
        "--calibration",
        choices=["platt", "stub", "ml_rf"],
        default="",
        help="Calibration method (default: platt)",
    )
    ev.add_argument(
        "--no-macd",
        action="store_true",
        help="Disable MACD feature engineering",
    )
    ev.add_argument(
        "--metrics-out",
        default="",
        help="Write per-run metrics CSV to this path (sep=';')",
    )
    ev.add_argument(
        "--summary-out",
        default="",
        help="Write aggregated summary CSV to this path (sep=';')",
    )
    ev.add_argument(
        "--latex-out",
        default="",
        help="Write a LaTeX table to this path",
    )
    ev.add_argument(
        "--latex-caption",
        default="GZ-CMD++ v3 evaluation summary",
        help="Caption for LaTeX table (default provided)",
    )
    ev.add_argument(
        "--latex-label",
        default="tab:gzcmd_v3_eval",
        help="Label for LaTeX table (default: tab:gzcmd_v3_eval)",
    )

    return p


def _print_summary(summary) -> None:
    print(f"rows={summary.rows}")
    print(f"llm_used={summary.llm_used}/{summary.llm_max}")
    print(f"review_requested={summary.review_requested}")
    print(f"guardrail_counts={summary.guardrails}")
    print(f"action_counts={summary.actions}")
    print(f"p_cal_method={summary.p_cal_method}")
    if summary.p_cal_params is not None:
        print(f"p_cal_params={summary.p_cal_params}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "run":
        out, summary = run_v3(
            input_csv=args.input,
            config_path=args.config,
            mode=args.mode,
            macd_enabled=not args.no_macd,
            llm_used=args.llm_used,
            p_cal=args.p_cal,
            platt_model_path=(args.platt_model or None),
            save_platt_model_path=(args.save_platt_model or None),
            ml_rf_model_path=(args.ml_rf_model or None),
            save_ml_rf_model_path=(args.save_ml_rf_model or None),
        )

        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(out_path, index=False, sep=";")

        _print_summary(summary)
        return 0

    if args.cmd == "fit-calibration":
        # Fit via the runner so we reuse the same loader + cleaning.
        _, summary = run_v3(
            input_csv=args.input,
            config_path=args.config,
            mode="vigilancia",
            macd_enabled=not args.no_macd,
            llm_used=0,
            p_cal="fit_platt",
            save_platt_model_path=args.output_model,
        )
        print(f"wrote_model={args.output_model}")
        print(f"p_cal_params={summary.p_cal_params}")
        return 0

    if args.cmd == "eval":
        cfg = load_config(args.config)

        modes = None if not args.modes else [str(m) for m in args.modes]
        split_by = str(args.split_by or cfg.evaluation.split_by).strip().lower()
        test_size = (
            float(cfg.evaluation.test_size)
            if args.test_size is None
            else float(args.test_size)
        )
        seeds = (
            list(cfg.evaluation.seeds)
            if not args.seeds
            else [int(s) for s in args.seeds]
        )
        group_stratify = (
            False if args.no_group_stratify else bool(cfg.evaluation.group_stratify)
        )
        calibration = str(args.calibration or "platt").strip().lower()

        runs = evaluate_v3_csv(
            input_csv=args.input,
            config_path=args.config,
            modes=modes,
            split_by=split_by,
            seeds=seeds,
            test_size=test_size,
            group_stratify=group_stratify,
            calibration=calibration,
            macd_enabled=not args.no_macd,
        )

        print(f"runs={len(runs)}")

        if args.metrics_out:
            write_csv(runs, args.metrics_out)
            print(f"wrote_metrics={args.metrics_out}")

        # Build a compact configuration label for summary tables.
        runs_for_summary = runs.copy()
        runs_for_summary["config"] = (
            runs_for_summary["mode"].astype(str)
            + " | split="
            + runs_for_summary["split_by"].astype(str)
            + " | cal="
            + runs_for_summary["calibration"].astype(str)
            + " | macd="
            + runs_for_summary["macd_enabled"].map(lambda x: "on" if bool(x) else "off")
        )

        metric_cols = [
            "exp_precision",
            "exp_recall",
            "exp_f1",
            "exp_fbeta",
            "auto_coverage",
            "llm_used",
        ]
        summary = summarize_runs(
            runs_for_summary, group_cols=["config"], metric_cols=metric_cols
        ).rename(columns={"config": "Config"})

        if args.summary_out:
            write_csv(summary, args.summary_out)
            print(f"wrote_summary={args.summary_out}")

        if args.latex_out:
            latex = summary_to_latex_table(
                summary,
                group_cols=["Config"],
                metrics=[
                    ("exp_precision", "Precision"),
                    ("exp_recall", "Recall"),
                    ("exp_f1", "F1"),
                    ("exp_fbeta", "Fbeta"),
                    ("auto_coverage", "Coverage"),
                    ("llm_used", "LLM"),
                ],
                caption=args.latex_caption,
                label=args.latex_label,
                decimals=3,
            )
            write_text(args.latex_out, latex)
            print(f"wrote_latex={args.latex_out}")

        return 0

    raise AssertionError(f"Unhandled command: {args.cmd}")
