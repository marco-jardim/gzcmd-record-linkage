"""Drift monitoring module for GZ-CMD.

Calculates Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) statistics
to detect distribution shifts in model scores and features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .config import GZCMDConfig


@dataclass(frozen=True)
class DriftMetric:
    metric: Literal["psi", "ks"]
    value: float
    status: Literal["ok", "warning", "critical"]
    feature: str
    slice_name: str | None = None
    slice_value: str | None = None


def calculate_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Calculate Population Stability Index (PSI)."""

    def scale_range(input, min, max):
        input += -(np.min(input))
        input /= np.max(input) / (max - min)
        input += min
        return input

    breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    expected_percents = np.percentile(expected, breakpoints)

    # Handle unique values to avoid division by zero in bins
    if len(np.unique(expected_percents)) < len(breakpoints):
        # Fallback for low cardinality: use unique values as bins if few enough
        # For now, simple prevention of bin collapse
        expected_percents = np.unique(expected_percents)
        if len(expected_percents) < 2:
            return 0.0  # Cannot calc PSI for single value

    # Discretize
    # Use digitize to bin data
    # Note: this is a simplified PSI. A robust one handles bin alignment carefully.
    # For GZ-CMD scores (0-1 or 0-10), fixed bins might be better.
    # Let's use fixed bins for stability if range is known, otherwise quantiles.

    # Using fixed bins for 0-1 probabilities is safer for p_cal
    # Using quantiles of EXPECTED is standard for continuous features

    # Let's stick to standard quantile-based for generality
    # Recalculate bins based on expected

    # Define bins using expected distribution
    try:
        expected_percents = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    except IndexError:
        return 0.0

    # Make bins unique
    expected_percents = np.unique(expected_percents)

    expected_counts = np.histogram(expected, expected_percents)[0]
    actual_counts = np.histogram(actual, expected_percents)[0]

    # Add epsilon to avoid zero division
    expected_counts = expected_counts + 1e-6
    actual_counts = actual_counts + 1e-6

    expected_frac = expected_counts / len(expected)
    actual_frac = actual_counts / len(actual)

    psi_value = np.sum(
        (actual_frac - expected_frac) * np.log(actual_frac / expected_frac)
    )
    return float(psi_value)


def calculate_ks(expected: np.ndarray, actual: np.ndarray) -> float:
    """Calculate Kolmogorov-Smirnov statistic."""
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    stat, _ = ks_2samp(expected, actual)
    return float(stat)


class DriftMonitor:
    def __init__(self, config: GZCMDConfig):
        self.config = config.monitoring

    def check_drift(
        self, baseline_df: pd.DataFrame, current_df: pd.DataFrame
    ) -> list[DriftMetric]:
        """Check for drift between baseline and current dataframes."""

        results = []

        if not self.config.enabled:
            return results

        # 1. Global drift check
        for feature in self.config.monitored_fields:
            if feature not in baseline_df.columns or feature not in current_df.columns:
                continue

            # Drop NAs
            base_data = (
                pd.to_numeric(baseline_df[feature], errors="coerce").dropna().values
            )
            curr_data = (
                pd.to_numeric(current_df[feature], errors="coerce").dropna().values
            )

            # KS
            ks_thresholds = self.config.metrics.get("ks", {})
            ks_warn = ks_thresholds.get("threshold_warn", 0.1)
            ks_crit = ks_thresholds.get("threshold_critical", 0.2)

            ks_val = calculate_ks(base_data, curr_data)
            status = "ok"
            if ks_val >= ks_crit:
                status = "critical"
            elif ks_val >= ks_warn:
                status = "warning"

            results.append(
                DriftMetric(
                    metric="ks",
                    value=ks_val,
                    status=status,
                    feature=feature,
                    slice_name="global",
                    slice_value="all",
                )
            )

            # PSI
            psi_thresholds = self.config.metrics.get("psi", {})
            psi_warn = psi_thresholds.get("threshold_warn", 0.1)
            psi_crit = psi_thresholds.get("threshold_critical", 0.2)

            psi_val = calculate_psi(base_data, curr_data)
            status = "ok"
            if psi_val >= psi_crit:
                status = "critical"
            elif psi_val >= psi_warn:
                status = "warning"

            results.append(
                DriftMetric(
                    metric="psi",
                    value=psi_val,
                    status=status,
                    feature=feature,
                    slice_name="global",
                    slice_value="all",
                )
            )

        # 2. Sliced drift check (e.g. by band)
        # TODO: Implement slicing if needed. For thesis PoC, global is sufficient proof of concept.

        return results

    def get_alerts(self, metrics: list[DriftMetric]) -> list[str]:
        """Generate human-readable alerts from drift metrics."""
        alerts = []
        for m in metrics:
            if m.status in ("warning", "critical"):
                alerts.append(
                    f"Drift alert: {m.metric.upper()} for {m.feature} is {m.value:.3f} ({m.status})"
                )
        return alerts
