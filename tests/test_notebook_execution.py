"""Testes de execução do notebook (TST2.1.NBEXEC + TST2.1.a).

- **TST2.1.NBEXEC**: executa o notebook ponta-a-ponta de forma *headless* via
  ``nbclient``/``ExecutePreprocessor`` e falha se qualquer célula levantar erro.
  Este teste cresce com o notebook ao longo da Wave 2.
- **TST2.1.a**: faz *parse* do ``.ipynb`` via ``nbformat`` e verifica que as seções
  esperadas (cabeçalhos markdown) estão presentes.

O notebook contém uma célula de *bootstrap* que ajusta ``sys.path`` sozinha, então o
kernel filho não depende de ``PYTHONPATH`` herdado.
"""

from __future__ import annotations

import nbformat
import pytest
from _nb_paths import NOTEBOOK_PATH, REPO_ROOT

pytestmark = pytest.mark.notebook


def _read_notebook() -> nbformat.NotebookNode:
    assert NOTEBOOK_PATH.exists(), (
        f"Notebook não encontrado em {NOTEBOOK_PATH}. "
        "Gere-o com: python notebooks/build_notebook.py"
    )
    return nbformat.read(NOTEBOOK_PATH, as_version=4)


def test_tst_2_1_a_secoes_esperadas_presentes() -> None:
    """O notebook contém os cabeçalhos de seção esperados para a Fase 2.1."""
    nb = _read_notebook()
    markdown = "\n\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    esperadas = [
        "# GZ-CMD++ v3",
        "## Glossário",
        "## 1. Contexto",
        "## 2. Visão geral do pipeline",
        "## 3. Objetivos",
        "## 4. Setup",
        "## 5. Primeiro olhar nos dados",
        "exemplo-fio-condutor",
        "## 7. Carga dos dados e",
        "feature engineering",
        "## 8. Atribuição de bandas",
        "## 9. Calibração",
        "boletim da calibração",
        "## 10. Guardrails",
        "## 11. Política de decisão",
        "## 12. Reconciliação",
        "## 13. Avaliação held-out",
        "## 14. Revisão LLM",
        "## 15. Painel interativo",
        "## 16. Conclusões e limitações",
    ]
    faltando = [s for s in esperadas if s not in markdown]
    assert not faltando, f"Seções ausentes no notebook: {faltando}"


def test_tst_4_1_a_figuras_tem_titulo_e_legenda() -> None:
    """TST4.1.a — Heurística: toda célula de código que cria figura (``plt``) deve
    definir título (``set_title``/``plt.title``/``suptitle``) e rótulos de eixo
    (``set_xlabel``/``set_ylabel`` ou ``plt.xlabel``/``plt.ylabel``), para que cada
    figura responda a uma pergunta com eixos legíveis em PT-BR (DEC-09 / eixo 6 do QA).
    """
    nb = _read_notebook()
    sem_titulo: list[int] = []
    sem_eixos: list[int] = []
    for i, c in enumerate(nb.cells):
        if c.cell_type != "code":
            continue
        src = c.source
        # Considera célula de plotagem aquela que adiciona eixos/figura.
        cria_figura = "subplots(" in src or "plt.figure(" in src or ".plot(" in src
        if not cria_figura:
            continue
        tem_titulo = "set_title(" in src or "plt.title(" in src or "suptitle(" in src
        tem_eixos = ("set_xlabel(" in src and "set_ylabel(" in src) or (
            "plt.xlabel(" in src and "plt.ylabel(" in src
        )
        if not tem_titulo:
            sem_titulo.append(i)
        if not tem_eixos:
            sem_eixos.append(i)
    assert not sem_titulo, f"Figuras sem título nos índices de célula: {sem_titulo}"
    assert not sem_eixos, (
        f"Figuras sem rótulos de eixo nos índices de célula: {sem_eixos}"
    )


def test_tst_2_1_a_toda_celula_de_codigo_tem_fonte() -> None:
    """Nenhuma célula de código está vazia (higiene básica do builder)."""
    nb = _read_notebook()
    vazias = [
        i
        for i, c in enumerate(nb.cells)
        if c.cell_type == "code" and not c.source.strip()
    ]
    assert not vazias, f"Células de código vazias nos índices: {vazias}"


def test_tst_2_1_nbexec_executa_ponta_a_ponta() -> None:
    """Executa o notebook inteiro de forma headless; falha se alguma célula der erro."""
    nbclient = pytest.importorskip("nbclient")
    nb = _read_notebook()
    client = nbclient.NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(REPO_ROOT)}},
    )
    # Levanta CellExecutionError se qualquer célula falhar.
    client.execute()
