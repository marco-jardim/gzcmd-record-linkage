"""Testes de execução do notebook (TST2.1.NBEXEC + TST2.1.a).

- **TST2.1.NBEXEC**: executa o notebook ponta-a-ponta de forma *headless* via
  ``nbclient``/``ExecutePreprocessor`` e falha se qualquer célula levantar erro.
  Este teste cresce com o notebook ao longo da Wave 2.
- **TST2.1.a**: faz *parse* do ``.ipynb`` via ``nbformat`` e verifica que as seções
  esperadas (cabeçalhos markdown) estão presentes. As frases do texto que
  dependem de números da base sintética são conferidas na própria execução
  (função ``confere`` do notebook), de modo que a execução falha se o texto
  deixar de valer.

O notebook contém uma célula de *bootstrap* que ajusta ``sys.path`` sozinha, então o
kernel filho não depende de ``PYTHONPATH`` herdado.
"""

from __future__ import annotations

import pytest
from _nb_paths import NOTEBOOK_PATH, REPO_ROOT

nbformat = pytest.importorskip("nbformat")

pytestmark = pytest.mark.notebook


def _read_notebook() -> nbformat.NotebookNode:
    assert NOTEBOOK_PATH.exists(), (
        f"Notebook não encontrado em {NOTEBOOK_PATH}. "
        "Gere-o com: python notebooks/build_notebook.py"
    )
    return nbformat.read(NOTEBOOK_PATH, as_version=4)


def test_tst_2_1_a_secoes_esperadas_presentes() -> None:
    """O notebook contém os cabeçalhos do plano aprovado (seções 0 a 10)."""
    nb = _read_notebook()
    markdown = "\n\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    esperadas = [
        "# GZ-CMD passo a passo",
        "## Preparação do ambiente",
        "## 0. Glossário",
        "## 1. O problema",
        "## 2. Teoria do GZ-CMD",
        "### 2.4 Um exemplo resolvido",
        "## 3. Dados sintéticos",
        "## 4. Probabilidade calibrada fora da amostra",
        "validação cruzada aninhada 5 × 5",
        "## 5. Ordenações da fila",
        "### 5.1 A fusão por posição recíproca",
        "## 6. Curva de esforço",
        "## 7. Incerteza",
        "## 8. Contraste com a automação",
        "## 9. Linha de base histórica",
        "## 10. Fluxo recomendado e limitações",
        "### 10.3 Limitações",
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


def test_tst_2_1_nbexec_executa_ponta_a_ponta(monkeypatch) -> None:
    """Executa o notebook inteiro de forma headless; falha se alguma célula der erro."""
    monkeypatch.setenv("NBH_STRICT_WIDGETS", "1")
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
