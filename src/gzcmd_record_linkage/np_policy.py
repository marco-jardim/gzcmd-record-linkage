"""Constrained automatic acceptance; mandatory guard MATCHs count in the FP budget.

Neyman-Pearson-style policy (thesis appendix): sort the MATCH candidates by
``p`` descending and take the largest prefix whose expected false positives,
guards included, satisfy ``sum(1 - p_i) <= alpha * |M|`` (or an absolute budget).
Infeasibility is reported through :class:`NPInfeasibleError`, never forced.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NPConstraints:
    max_fp_rate_auto: float = 0.002
    max_fp_expected_total: float | None = None
    max_fn_rate_auto: float = 0.02

    def __post_init__(self) -> None:
        for value in (self.max_fp_rate_auto, self.max_fn_rate_auto):
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("NP rates must be finite probabilities")
        if self.max_fp_expected_total is not None and (
            not np.isfinite(self.max_fp_expected_total)
            or self.max_fp_expected_total < 0
        ):
            raise ValueError("NP absolute FP budget must be finite and nonnegative")


class NPInfeasibleError(ValueError):
    """No decision set respects mandatory guards and the requested constraint."""

    def __init__(self, message: str, **diagnostics) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


def select_match_prefix(
    p: pd.Series, fixed_p: pd.Series, constraints: NPConstraints
) -> tuple[pd.Index, dict]:
    """Maximize sum(p) subject to expected FP, including compulsory guard MATCHs.

    For any cardinality, the descending-p prefix dominates every other subset
    in reward and FP cost. Check all prefixes: an initially infeasible guard
    rate can become feasible after adding p=1 rows. Never stop too early.
    """
    for values in (p, fixed_p):
        array = values.to_numpy(dtype=float)
        if not (np.isfinite(array).all() and ((array >= 0) & (array <= 1)).all()):
            raise ValueError("NP requires finite probabilities in [0,1]")
    ordered = p.sort_values(ascending=False, kind="stable")
    guard_fp = float((1 - fixed_p).sum())
    counts = len(fixed_p) + np.arange(len(ordered) + 1)
    fp = guard_fp + np.r_[0.0, (1 - ordered).cumsum().to_numpy()]
    rates = np.divide(fp, counts, out=np.zeros_like(fp), where=counts > 0)
    limits = (
        constraints.max_fp_rate_auto * counts
        if constraints.max_fp_expected_total is None
        else np.full(len(counts), constraints.max_fp_expected_total)
    )
    valid = fp <= limits + 1e-12
    best_rate_index = int(np.argmin(rates))
    diagnostics = {
        "fp_esperado_guardas": guard_fp,
        "n_guardas_match": len(fixed_p),
        "taxa_minima_atingivel": float(rates[best_rate_index]),
        "n_match_na_taxa_minima": int(counts[best_rate_index]),
        "max_fp_rate_auto": constraints.max_fp_rate_auto,
        "max_fp_expected_total": constraints.max_fp_expected_total,
    }
    if not valid.any():
        raise NPInfeasibleError(
            "Mandatory MATCH guards make the NP FP constraint infeasible",
            motivo="guardas_fp",
            **diagnostics,
        )
    k = int(np.flatnonzero(valid)[-1])
    diagnostics.update(
        {
            "np_fp_esperado_auto": float(fp[k]),
            "np_n_match_auto": int(counts[k]),
            "np_taxa_fp_auto": float(rates[k]),
            "np_fp_limite": float(limits[k]),
        }
    )
    return ordered.index[:k], diagnostics
