from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


def write_csv(df: pd.DataFrame, path: str | Path, *, sep: str = ";") -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, sep=sep)


def summarize_runs(
    runs: pd.DataFrame,
    *,
    group_cols: list[str],
    metric_cols: list[str],
) -> pd.DataFrame:
    missing = [c for c in (group_cols + metric_cols) if c not in runs.columns]
    if missing:
        raise KeyError(f"Missing columns in runs df: {missing}")

    grp = runs.groupby(group_cols, dropna=False)
    summary = grp[metric_cols].agg(["mean", "std"]).reset_index()
    counts = grp.size().to_frame("n_runs").reset_index()

    # Flatten columns: (metric, agg) -> f"{metric}_{agg}".
    flat_cols: list[str] = []
    for c in summary.columns:
        if isinstance(c, tuple):
            name = str(c[0])
            suffix = "" if c[1] is None else str(c[1])
            if suffix == "":
                flat_cols.append(name)
            else:
                flat_cols.append(f"{name}_{suffix}")
        else:
            flat_cols.append(str(c))
    summary.columns = flat_cols

    out = summary.merge(counts, on=group_cols, how="left")
    return out


def _latex_escape(s: str) -> str:
    # Minimal escaping for our expected content.
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


def _fmt_pm(mean: float, std: float, *, decimals: int) -> str:
    fmt = f"{{:.{decimals}f}}"
    return f"${fmt.format(mean)} \\pm {fmt.format(std)}$"


def summary_to_latex_table(
    summary: pd.DataFrame,
    *,
    group_cols: list[str],
    metrics: list[tuple[str, str]],
    caption: str,
    label: str,
    decimals: int = 3,
) -> str:
    """Build a booktabs LaTeX table from a mean/std summary.

    metrics: list of (metric_name, header_label). Expects columns
      - f"{metric_name}_mean" and f"{metric_name}_std" in `summary`.

    ``caption`` and ``label`` are emitted verbatim (no escaping) so the
    caller may include LaTeX commands such as ``\\textsubscript{}``.
    """

    needed = list(group_cols)
    for metric, _ in metrics:
        needed.append(f"{metric}_mean")
        needed.append(f"{metric}_std")

    missing = [c for c in needed if c not in summary.columns]
    if missing:
        raise KeyError(f"Missing summary columns: {missing}")

    col_spec = "l" * len(group_cols) + "r" * len(metrics)
    header_cells = [_latex_escape(c) for c in group_cols] + [h for _, h in metrics]

    lines: list[str] = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\small")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(header_cells) + " \\\\")
    lines.append("\\midrule")

    for _, row in summary.iterrows():
        row_cells: list[str] = []
        for c in group_cols:
            row_cells.append(_latex_escape(str(row[c])))
        for metric, _ in metrics:
            m = float(row[f"{metric}_mean"])
            s_raw = row[f"{metric}_std"]
            try:
                s_float = float(s_raw)
            except (TypeError, ValueError):
                s_float = float("nan")

            s = 0.0 if math.isnan(s_float) else s_float
            row_cells.append(_fmt_pm(m, s, decimals=decimals))
        lines.append(" & ".join(row_cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines) + "\n"


def write_text(path: str | Path, content: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
