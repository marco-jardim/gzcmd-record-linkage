"""Read-only import of the thesis evaluation modules, for parity tests.

Unlike :mod:`tests._thesis_repo` this needs no thesis *data*: only the plain
modules ``gzcmd.{np_evaluation,cost_evaluation,intervals,provenance,guardrails,
config}`` and the thesis v3.1/P config. They are loaded by path (``sys.path`` is
extended only while importing, bytecode is not written and the temporary module
entries are removed afterwards), so the library gains no dependency on the
thesis. Parity tests skip when those files are absent.
"""

from __future__ import annotations

import importlib
import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest
from _thesis_repo import THESIS_ROOT

THESIS_CONFIG = THESIS_ROOT / "gzcmd" / "gzcmd_v31_p_config.yaml"
_FILES = (
    "gzcmd/np_evaluation.py",
    "gzcmd/cost_evaluation.py",
    "gzcmd/intervals.py",
    "gzcmd/provenance.py",
    "gzcmd/guardrails.py",
    "gzcmd/config.py",
)
THESIS_EVAL_AVAILABLE = THESIS_CONFIG.exists() and all(
    (THESIS_ROOT / name).exists() for name in _FILES
)

requires_thesis_eval = pytest.mark.skipif(
    not THESIS_EVAL_AVAILABLE, reason="thesis evaluation modules not found"
)


def _owned(name: str) -> bool:
    return name == "gzcmd" or name.startswith("gzcmd.")


@lru_cache(maxsize=1)
def load_thesis_eval() -> SimpleNamespace:
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if _owned(k)}
    bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(THESIS_ROOT))
    try:
        namespace = SimpleNamespace(
            np_evaluation=importlib.import_module("gzcmd.np_evaluation"),
            cost_evaluation=importlib.import_module("gzcmd.cost_evaluation"),
            intervals=importlib.import_module("gzcmd.intervals"),
            provenance=importlib.import_module("gzcmd.provenance"),
            guardrails=importlib.import_module("gzcmd.guardrails"),
            config=importlib.import_module("gzcmd.config"),
            policy_engine=importlib.import_module("gzcmd.policy_engine"),
        )
    finally:
        sys.path.remove(str(THESIS_ROOT))
        sys.dont_write_bytecode = bytecode
        for name in [k for k in sys.modules if _owned(k)]:
            del sys.modules[name]
        sys.modules.update(saved)
    return namespace


def thesis_config(path: Path = THESIS_CONFIG):
    return load_thesis_eval().config.load_config(path)
