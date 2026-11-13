"""Read-only access to the thesis research repository, for parity tests.

The thesis modules are loaded by path with importlib (``sys.path`` is extended
only while importing, bytecode is not written, and the temporary module entries
are removed afterwards), so the library gains no runtime dependency. Every
parity test skips when the repository or its data are absent.
"""

from __future__ import annotations

import importlib
import json
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

THESIS_ROOT = Path(r"D:\git\phd-research")
REQUIRED = (
    "data/COMPARADORSEMIDENT.csv",
    "data/pcal_ml_oof.csv",
    "data/pcal_ml_oof.json",
    "data/ablation_comprec_allowlist_metadata.json",
    "scripts/contexto_obito.py",
    "scripts/sensibilidade_o3_macd.py",
    "gzcmd/ml_oof.py",
)
THESIS_AVAILABLE = all((THESIS_ROOT / name).exists() for name in REQUIRED)

requires_thesis = pytest.mark.skipif(
    not THESIS_AVAILABLE, reason="thesis repository or its data not found"
)

_NAMESPACES = ("gzcmd", "scripts")


def _owned(name: str) -> bool:
    return any(name == ns or name.startswith(ns + ".") for ns in _NAMESPACES)


@lru_cache(maxsize=1)
def load_thesis() -> dict[str, Any]:
    """Import the thesis modules and run its data preparation (read-only)."""
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if _owned(k)}
    bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(THESIS_ROOT))
    try:
        modules: dict[str, ModuleType] = {
            "contexto": importlib.import_module("scripts.contexto_obito"),
            "sensibilidade": importlib.import_module("scripts.sensibilidade_o3_macd"),
            "ml_oof": importlib.import_module("gzcmd.ml_oof"),
        }
        data = modules["sensibilidade"].prepare(THESIS_ROOT)
    finally:
        sys.path.remove(str(THESIS_ROOT))
        sys.dont_write_bytecode = bytecode
        for name in [k for k in sys.modules if _owned(k)]:
            del sys.modules[name]
        sys.modules.update(saved)
    ablation = json.loads(
        (THESIS_ROOT / "data/ablation_comprec_allowlist_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    original = json.loads(
        (THESIS_ROOT / "data/pcal_ml_oof.json").read_text(encoding="utf-8")
    )
    return {
        **modules,
        "data": data,
        "ablation_macd": [
            c for c in ablation["features"]["macd"]["names"] if c.startswith("macd_")
        ],
        "original_features": original["features"],
    }
