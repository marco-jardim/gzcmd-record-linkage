"""Caminhos compartilhados pelos testes do notebook (não é um módulo de testes)."""

from __future__ import annotations

from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return start


REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
NOTEBOOK_PATH = NOTEBOOKS_DIR / "gzcmd_passo_a_passo.ipynb"
DATA_SYNTHETIC_DIR = REPO_ROOT / "data" / "synthetic"
