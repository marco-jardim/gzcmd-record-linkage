"""Configuração global do pytest, carregada antes de qualquer módulo de teste.

Limita a CPU usada pela suite (afinidade, prioridade baixa e threads BLAS); veja
``tests/_cpu_limit.py`` e a seção de desenvolvimento do README. Precisa rodar
antes de numpy/sklearn/xgboost serem importados (as variáveis de threads são
lidas na importação), por isso fica na raiz e não em ``tests/``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_HELPER = Path(__file__).resolve().parent / "tests" / "_cpu_limit.py"


def _load_cpu_limit():
    spec = importlib.util.spec_from_file_location("_cpu_limit", _HELPER)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_cpu_limit"] = module
    spec.loader.exec_module(module)
    return module


# Também roda em cada worker do xdist (cada um importa este conftest).
_load_cpu_limit().apply_cpu_limit()
