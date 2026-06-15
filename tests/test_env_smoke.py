"""Smoke test de ambiente (TST0.1.a do plano).

Garante que **todos** os módulos públicos do pacote ``gzcmd_record_linkage``
importam sem erro no ambiente atual. É um teste de *fundação*: se algum módulo
quebra na importação (dependência faltando, erro de sintaxe, import circular),
todo o restante do notebook didático fica comprometido.

A descoberta de módulos é **dinâmica** (via ``pkgutil``): qualquer módulo novo
adicionado ao pacote é coberto automaticamente, sem precisar editar este teste.
O entry-point ``__main__`` é excluído de propósito — ele é exercitado pela CLI
em ``test_cli.py`` e sua importação dispara efeitos de linha de comando.

Não exercita comportamento — apenas a importabilidade. Comportamento é coberto
pelos testes de contrato (Fase 0.2) e demais fases.
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

import gzcmd_record_linkage

# Excluídos da varredura de importabilidade (com justificativa).
_EXCLUDED = {"__main__"}


def _public_modules() -> list[str]:
    """Descobre dinamicamente os submódulos públicos do pacote."""
    discovered = [
        f"{gzcmd_record_linkage.__name__}.{info.name}"
        for info in pkgutil.iter_modules(gzcmd_record_linkage.__path__)
        if info.name not in _EXCLUDED and not info.name.startswith("_")
    ]
    return sorted(discovered)


PUBLIC_MODULES = _public_modules()

# Módulos-âncora que DEVEM existir — protege contra uma varredura vazia
# (ex.: layout do pacote mudou e a descoberta falhou silenciosamente).
_EXPECTED_ANCHORS = {
    "gzcmd_record_linkage.bands",
    "gzcmd_record_linkage.calibration",
    "gzcmd_record_linkage.guardrails",
    "gzcmd_record_linkage.loader",
    "gzcmd_record_linkage.runner",
    "gzcmd_record_linkage.gzcmd_v3_policy_engine",
}


def test_descoberta_de_modulos_nao_vazia() -> None:
    """A varredura encontra os módulos-âncora esperados do pacote."""
    found = set(PUBLIC_MODULES)
    missing = _EXPECTED_ANCHORS - found
    assert not missing, f"Módulos-âncora ausentes na descoberta: {sorted(missing)}"


@pytest.mark.parametrize("module_name", PUBLIC_MODULES)
def test_public_module_importavel(module_name: str) -> None:
    """Cada módulo público importa sem lançar exceção."""
    module = importlib.import_module(module_name)
    assert module is not None


def test_versao_exposta() -> None:
    """O pacote expõe ``__version__`` como string não-vazia."""
    assert isinstance(gzcmd_record_linkage.__version__, str)
    assert gzcmd_record_linkage.__version__
