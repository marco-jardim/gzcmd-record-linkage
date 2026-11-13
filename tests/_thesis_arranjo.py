"""Read-only loader of the thesis ASR modules, for parity tests.

Same approach as ``_thesis_repo``: the thesis package is imported by path with
``sys.path`` extended only during the import, no bytecode is written and the
temporary module entries are removed afterwards. Every parity test using it
skips when the thesis repository or its data are absent.
"""

from __future__ import annotations

import importlib
import sys
from functools import lru_cache
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
from _thesis_repo import THESIS_AVAILABLE, THESIS_ROOT

RAW_CSV = THESIS_ROOT / "data" / "COMPARADORSEMIDENT.csv"
ASR_AVAILABLE = (
    THESIS_AVAILABLE
    and RAW_CSV.exists()
    and (THESIS_ROOT / "gzcmd" / "arranjo_corte.py").exists()
    and (THESIS_ROOT / "gzcmd" / "arranjo_robustez.py").exists()
)

requires_thesis_asr = pytest.mark.skipif(
    not ASR_AVAILABLE, reason="thesis ASR modules or data not found"
)

_NAMESPACES = ("gzcmd", "scripts")


def _owned(name: str) -> bool:
    return any(name == ns or name.startswith(ns + ".") for ns in _NAMESPACES)


@lru_cache(maxsize=1)
def load_asr() -> dict[str, ModuleType]:
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if _owned(k)}
    bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(THESIS_ROOT))
    try:
        return {
            "corte": importlib.import_module("gzcmd.arranjo_corte"),
            "robustez": importlib.import_module("gzcmd.arranjo_robustez"),
            "ml_oof": importlib.import_module("gzcmd.ml_oof"),
        }
    finally:
        sys.path.remove(str(THESIS_ROOT))
        sys.dont_write_bytecode = bytecode
        for name in [k for k in sys.modules if _owned(k)]:
            del sys.modules[name]
        sys.modules.update(saved)


@lru_cache(maxsize=1)
def load_raw() -> pd.DataFrame:
    """The raw comparator CSV as the thesis loads it (strings, header names only)."""
    raw = pd.read_csv(RAW_CSV, sep=";", dtype=str, low_memory=False)
    raw.columns = [c.split(",")[0] for c in raw.columns]
    return raw


def real_subsample(raw: pd.DataFrame, below: np.ndarray, seed: int = 0) -> np.ndarray:
    """Positional rows of a small deterministic subsample keeping whole REFREC groups.

    All groups holding a PAR=2 pair below the cut, 30 groups with PAR=1 pairs
    and 220 groups with only below-cut negatives.
    """
    par = pd.to_numeric(raw["PAR"], errors="coerce").to_numpy()
    groups = raw["REFREC"].str.strip().to_numpy()
    rng = np.random.default_rng(seed)
    par2_groups = np.unique(groups[below & (par == 2)])
    par1_groups = np.setdiff1d(np.unique(groups[par == 1]), par2_groups)
    positive_groups = np.union1d(par2_groups, par1_groups)
    negative_groups = np.setdiff1d(np.unique(groups[below]), positive_groups)
    chosen = np.concatenate(
        [
            par2_groups,
            rng.choice(par1_groups, min(30, len(par1_groups)), replace=False),
            rng.choice(negative_groups, 220, replace=False),
        ]
    )
    return np.flatnonzero(np.isin(groups, chosen))


def real_inputs():
    """(x, y, below, groups, par2) of the deterministic real subsample (lib helpers)."""
    from gzcmd_record_linkage.arranjo_corte import below_cutoff
    from gzcmd_record_linkage.oof import numeric_features

    raw = load_raw()
    below_all = below_cutoff(raw)
    rows = real_subsample(raw, below_all)
    part = raw.iloc[rows].reset_index(drop=True)
    par = pd.to_numeric(part["PAR"], errors="coerce")
    return (
        numeric_features(part),
        par.isin([1, 2]).astype(int).to_numpy(),
        below_all[rows],
        part["REFREC"].str.strip().to_numpy(),
        par.eq(2).to_numpy(),
    )
