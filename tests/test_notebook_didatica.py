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

import ast
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from _nb_paths import NOTEBOOK_PATH

nbformat = pytest.importorskip("nbformat")
build_notebook = pytest.importorskip("build_notebook")

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


def test_sem_saidas_armazenadas() -> None:
    """O artefato versionado não guarda resultados nem contadores de execução."""
    for i, cell in enumerate(_carregar_notebook().cells):
        if cell.cell_type == "code":
            assert cell.outputs == [], f"Saídas armazenadas na célula {i}."
            assert cell.execution_count is None, f"Contador na célula {i}."


def test_builder_em_sincronia() -> None:
    """Construir em memória reproduz as fontes e os tipos do notebook entregue."""
    gerado = build_notebook.build()
    entregue = _carregar_notebook()
    assert [(c.cell_type, c.source, c.id, c.metadata) for c in gerado.cells] == [
        (c.cell_type, c.source, c.id, c.metadata) for c in entregue.cells
    ]
    assert gerado == build_notebook.build()


def test_seis_controles() -> None:
    codigo = "\n".join(
        _sem_comentarios(c.source) for c in _carregar_notebook().cells
        if c.cell_type == "code"
    )
    assert codigo.count("nbh.interagir(") == 6


def test_wrappers_sem_renderizacao_duplicada() -> None:
    for cell in _carregar_notebook().cells:
        if cell.cell_type != "code" or "nbh.interagir(" not in cell.source:
            continue
        tree = ast.parse(cell.source)
        assert "if not nbh.interagir(" in cell.source
        for forbidden in (".plot(", "subplots(", "plt.figure("):
            assert forbidden not in cell.source
        wrappers = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name.startswith("_ui_")]
        assert len(wrappers) == 1
        wrapper = ast.get_source_segment(cell.source, wrappers[0])
        assert "plt.show()\n    plt.close(fig)" in wrapper


def _celula_codigo_contendo(fragment):
    return next(cell.source for cell in build_notebook.build().cells
                if cell.cell_type == "code" and fragment in cell.source)


def test_bootstrap_local_nao_instala_sem_repositorio(monkeypatch, tmp_path):
    import subprocess

    source = _celula_codigo_contendo("def _find_repo_root(")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("Bootstrap local não pode clonar ou instalar pacotes.")

    monkeypatch.setattr(subprocess, "run", forbidden)
    with pytest.raises(RuntimeError, match="Raiz do repositório não encontrada"):
        exec(source, {})


def _proteger_modulos_do_projeto(monkeypatch):
    # O bootstrap esquece módulos já importados; restauramos no teardown.
    for nome in list(sys.modules):
        raiz = nome.split(".")[0]
        if raiz in {"nb_helpers", "synthetic_data", "gzcmd_record_linkage"}:
            monkeypatch.setitem(sys.modules, nome, sys.modules[nome])


def test_bootstrap_local_precede_src_e_notebooks(monkeypatch):
    root = Path(build_notebook.__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    monkeypatch.setattr(sys, "path", list(sys.path))
    _proteger_modulos_do_projeto(monkeypatch)
    exec(_celula_codigo_contendo("def _find_repo_root("), {})
    assert sys.path[:2] == [str(root / "notebooks"), str(root / "src")]


def test_secoes_avisam_quando_a_sessao_foi_reiniciada():
    cells = build_notebook.build().cells
    secao = None
    primeiras = {}
    for cell in cells:
        if cell.cell_type == "markdown":
            achou = re.search(r"^## (\d+)\.", cell.source, re.MULTILINE)
            if achou:
                secao = int(achou.group(1))
        elif secao is not None and secao >= 5 and secao not in primeiras:
            primeiras[secao] = cell.source
    # §15 e §16 são só texto; todas as seções com código de 5 a 14 têm guarda.
    assert set(primeiras) == set(range(5, 15))
    for numero, source in primeiras.items():
        guarda = build_notebook.GUARDA_SESSAO
        assert source.startswith(guarda), f"Seção {numero} sem guarda."
    with pytest.raises(AssertionError, match="Executar antes"):
        exec(build_notebook.GUARDA_SESSAO, {})
    exec(build_notebook.GUARDA_SESSAO, {"df_raw": object()})


def test_bootstrap_esquece_modulos_antigos(monkeypatch):
    root = Path(build_notebook.__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    monkeypatch.setattr(sys, "path", list(sys.path))
    _proteger_modulos_do_projeto(monkeypatch)
    antigo = SimpleNamespace(__name__="nb_helpers")
    monkeypatch.setitem(sys.modules, "nb_helpers", antigo)
    exec(_celula_codigo_contendo("def _find_repo_root("), {})
    assert "nb_helpers" not in sys.modules


@pytest.mark.parametrize("editados", ["", " M notebooks/nb_helpers.py"])
def test_bootstrap_colab_atualiza_copia_existente(
    monkeypatch, tmp_path, capsys, editados
):
    import subprocess

    # Cenário do Colab: cwd (/content) sem pyproject e uma cópia clonada antes.
    destino = tmp_path / "gzcmd-record-linkage"
    destino.mkdir()
    (destino / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "google.colab", SimpleNamespace())
    monkeypatch.setattr(sys, "path", list(sys.path))
    _proteger_modulos_do_projeto(monkeypatch)
    chamadas = []

    def fake_run(cmd, *args, **kwargs):
        chamadas.append(cmd)
        saida = editados if "status" in cmd else ""
        return subprocess.CompletedProcess(cmd, 0, stdout=saida, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    exec(_celula_codigo_contendo("def _find_repo_root("), {})
    git_fetch = [c for c in chamadas if "fetch" in c]
    git_reset = [c for c in chamadas if "reset" in c]
    assert not any("clone" in c for c in chamadas)
    assert any("pip" in c for c in chamadas)
    if editados:
        assert not git_fetch and not git_reset
        assert "NÃO foi atualizada" in capsys.readouterr().out
    else:
        assert git_fetch and git_reset
        assert git_reset[0][-1] == "FETCH_HEAD"


def test_celula_csv_confere_sem_sobrescrever(tmp_path, capsys):
    source = _celula_codigo_contendo("dataset = synthetic_data.generate_comparador(")
    frame = pd.DataFrame({"valor": [1, 2]})

    def write_csv(data, path):
        data.to_csv(path, index=False)

    generator = SimpleNamespace(
        generate_comparador=lambda **kw: SimpleNamespace(
            frame=frame, p_true=[0.1, 0.9],
            meta={"match_ratio_realized": 0.5, "true_slope": 1, "true_intercept": -1},
        ),
        to_comparador_csv=write_csv,
    )
    namespace = {"synthetic_data": generator, "SEED": 42, "DATA_DIR": tmp_path,
                 "REPO_ROOT": tmp_path, "Path": Path}
    exec(source, namespace)
    assert "arquivo criado" in capsys.readouterr().out
    path = tmp_path / "comparador_sintetico.csv"
    # A leitura de texto deve normalizar CRLF, sem tocar no arquivo existente.
    path.write_bytes(b"valor\r\n1\r\n2\r\n")
    before = path.read_bytes()
    exec(source, namespace)
    assert path.read_bytes() == before
    assert "conferido e preservado" in capsys.readouterr().out
    path.write_text("valor\n3\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="CSV versionado difere"):
        exec(source, namespace)
    assert path.read_text(encoding="utf-8") == "valor\n3\n"


def test_ipywidgets_nao_importado_no_notebook() -> None:
    """Imports opcionais ficam no helper, não nas células do notebook."""
    for i, cell in enumerate(_carregar_notebook().cells):
        if cell.cell_type == "code":
            assert not re.search(
                r"\b(?:import\s+ipywidgets\b|from\s+ipywidgets\b)", cell.source
            ), f"Import direto de ipywidgets na célula {i}."


def test_custos_nao_hardcoded() -> None:
    padroes = [
        r"CUSTO_FP\s*,\s*CUSTO_FN\s*=\s*\d", r'"false_positive"\s*:\s*\d',
        r"c_fp\s*=\s*\d", r"c_fn\s*=\s*\d", r"c_llm\s*=\s*\d",
        r"false_positive(_cost)?\s*=\s*\d", r"false_negative(_cost)?\s*=\s*\d",
        r"Costs\([^)]*\d",
    ]
    for i, cell in enumerate(_carregar_notebook().cells):
        if cell.cell_type == "code":
            for padrao in padroes:
                assert not re.search(padrao, _sem_comentarios(cell.source)), (
                    f"Custo fixado no código da célula {i}: {padrao}"
                )


def test_sem_travessao_no_markdown() -> None:
    for i, cell in enumerate(_carregar_notebook().cells):
        if cell.cell_type == "markdown":
            assert "—" not in cell.source, f"Travessão no markdown da célula {i}."


def test_details_com_summary_e_espaco_para_renderizar_markdown() -> None:
    """Blocos recolhíveis precisam de fechamento e respiro para Markdown/LaTeX."""
    for i, cell in enumerate(_carregar_notebook().cells):
        if (cell.cell_type != "markdown"
                or not re.search(r"<details(?:\s+open)?>", cell.source)):
            continue
        fonte = cell.source
        quantidade = len(re.findall(r"<details(?:\s+open)?>", fonte))
        assert quantidade == fonte.count("</details>"), f"Fechamento na célula {i}."
        assert quantidade == fonte.count("<summary>"), f"Resumo na célula {i}."
        assert quantidade == fonte.count("</summary>"), f"Fim do resumo na célula {i}."
        assert quantidade == len(re.findall(r"</summary>\r?\n[ \t]*\r?\n", fonte)), (
            f"Falta linha em branco após </summary> na célula {i}."
        )
        assert quantidade == len(re.findall(r"\r?\n[ \t]*\r?\n</details>", fonte)), (
            f"Falta linha em branco antes de </details> na célula {i}."
        )


def _sem_comentarios(source):
    return "\n".join(line for line in source.splitlines()
                     if not line.lstrip().startswith("#"))


def _conferir_blocos_simples(secao):
    assert secao.count("🧸") >= 8
    for subsection in re.split(r"(?m)^### ", secao)[1:]:
        if re.search(r"<details(?:\s+open)?>", subsection):
            assert re.search(r"(?m)^>\s*🧸", subsection), subsection.splitlines()[0]


def test_calibracao_tem_explicacoes_bem_simples() -> None:
    """A seção de calibração abre espaço explícito para quem começa do zero."""
    fontes: list[str] = []
    na_calibracao = False
    for cell in _carregar_notebook().cells:
        if cell.cell_type != "markdown":
            continue
        if "## 9. Calibração" in cell.source:
            na_calibracao = True
        if "## 10. " in cell.source:
            break
        if na_calibracao:
            fontes.append(cell.source)
    _conferir_blocos_simples("\n".join(fontes))


def test_avaliacao_held_out_tem_explicacoes_bem_simples() -> None:
    """A avaliação e o caminho ML também acolhem quem começa do zero."""
    markdown = "\n".join(
        cell.source for cell in _carregar_notebook().cells
        if cell.cell_type == "markdown"
    )
    secao_13 = markdown.split("## 13.", 1)[1].split("## 14.", 1)[0]
    _conferir_blocos_simples(secao_13)
