"""Auditoria didática automatizada do notebook (DF-3 / TST4.3.a).

Este teste implementa, de forma estática (via ``nbformat``, sem executar o
notebook), a regra didática DF-3 do plano:

    "Antes de cada passo/célula de código existe uma célula *markdown* que
    explica, de forma compreensiva e em PT-BR, o que vai acontecer no passo
    seguinte. Não pode haver célula de código 'órfã' sem contexto prévio."

Interpretação operacional (faithful ao plano, Seção 5.2.1 / 4.3 / TST4.3.a):

* Uma célula de código é **substantiva** quando tem fonte não-vazia (células
  de código vazias já são proibidas por ``test_notebook_execution``).
* Toda célula de código substantiva deve ser **imediatamente precedida** por
  uma célula ``markdown`` **não-vazia** (o "o quê" e o "porquê" do passo).
* ``EXCECOES_ORFAS`` lista, de forma explícita e justificada, eventuais
  células de código autorizadas a não ter markdown imediatamente antes. Hoje
  está vazio: o notebook é gerado por ``build_notebook.py`` seguindo DF-3 e
  **todas** as células de código são precedidas por markdown.

Mantém-se *sem* a marca ``notebook`` porque é uma análise estática rápida
(parse do ``.ipynb``), não uma execução de kernel — deve rodar na suíte padrão.
"""

from __future__ import annotations

import nbformat
from _nb_paths import NOTEBOOK_PATH

# Conjunto de exceções explícitas: primeira linha (strip) de células de código
# autorizadas a NÃO ser precedidas por markdown. Cada entrada DEVE vir com uma
# justificativa em comentário. Atualmente vazio — nenhuma célula órfã existe.
EXCECOES_ORFAS: frozenset[str] = frozenset()


def _carregar_notebook() -> nbformat.NotebookNode:
    return nbformat.read(str(NOTEBOOK_PATH), as_version=4)


def test_tst_4_3_a_toda_celula_codigo_precedida_por_markdown() -> None:
    """DF-3: nenhuma célula de código substantiva é órfã (sem markdown antes)."""
    nb = _carregar_notebook()
    cells = nb.cells

    violacoes: list[str] = []
    for i, cell in enumerate(cells):
        if cell.cell_type != "code":
            continue
        fonte = (cell.source or "").strip()
        if not fonte:
            # Célula de código vazia: coberta por outro teste; ignorada aqui.
            continue
        primeira_linha = fonte.splitlines()[0].strip()
        if primeira_linha in EXCECOES_ORFAS:
            continue

        anterior = cells[i - 1] if i > 0 else None
        ok = (
            anterior is not None
            and anterior.cell_type == "markdown"
            and bool((anterior.source or "").strip())
        )
        if not ok:
            tipo_anterior = anterior.cell_type if anterior is not None else "<nenhuma>"
            violacoes.append(
                f"  célula {i} (code) órfã — anterior é '{tipo_anterior}'. "
                f"Primeira linha: {primeira_linha!r}"
            )

    assert not violacoes, (
        "Células de código sem markdown explicativo imediatamente antes "
        "(viola DF-3):\n" + "\n".join(violacoes)
    )


def test_tst_4_3_a_existe_markdown_antes_do_primeiro_codigo() -> None:
    """Sanidade: o notebook abre com narrativa (markdown), não com código."""
    nb = _carregar_notebook()
    primeiro_codigo = next(
        (i for i, c in enumerate(nb.cells) if c.cell_type == "code"), None
    )
    assert primeiro_codigo is not None, "Notebook não contém células de código."
    assert any(
        c.cell_type == "markdown" and (c.source or "").strip()
        for c in nb.cells[:primeiro_codigo]
    ), "Não há markdown explicativo antes da primeira célula de código."
