"""Auditoria didática automatizada do notebook (DF-3 / TST4.3.a).

Análise estática (via ``nbformat``, sem executar o notebook) do contrato de
``gzcmd_passo_a_passo.ipynb``, gerado por ``build_notebook.py``:

* toda célula de código substantiva vem logo depois de um markdown (DF-3);
* as seções aprovadas (0 a 10) aparecem em ordem, e cada seção com código tem
  ao menos uma célula ``**Leitura.**`` depois do primeiro código;
* a terminologia segue a tese (guia de estilo, seção 3.6);
* nenhum número medido na base sintética fica fixo no markdown, e os números
  da tese só aparecem na seção 10, como referência citada;
* o notebook usa a configuração recomendada pela tese e o perfil histórico.

Sem a marca ``notebook``: é um parse rápido e roda na suíte padrão.
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

# Primeira linha (strip) de células de código autorizadas a NÃO ser precedidas
# por markdown. Cada entrada precisa de justificativa. Hoje vazio.
EXCECOES_ORFAS: frozenset[str] = frozenset()

TITULOS_DAS_SECOES = {
    0: "Glossário",
    1: "O problema",
    2: "Teoria do GZ-CMD",
    3: "Dados sintéticos",
    4: "Probabilidade calibrada fora da amostra",
    5: "Ordenações da fila",
    6: "Curva de esforço",
    7: "Incerteza",
    8: "Contraste com a automação",
    9: "Linha de base histórica",
    10: "Fluxo recomendado e limitações",
}

_SECAO_RE = re.compile(r"^## (\d+)\. ", re.MULTILINE)


def _carregar_notebook() -> nbformat.NotebookNode:
    return nbformat.read(str(NOTEBOOK_PATH), as_version=4)


def _secoes(cells) -> dict[int | None, list]:
    """Agrupa as células pela seção numerada; ``None`` reúne as anteriores à 0."""
    secao = None
    grupos: dict[int | None, list] = {}
    for cell in cells:
        if cell.cell_type == "markdown":
            achou = _SECAO_RE.search(cell.source)
            if achou:
                secao = int(achou.group(1))
        grupos.setdefault(secao, []).append(cell)
    return grupos


def _markdown(cells=None) -> list[str]:
    cells = _carregar_notebook().cells if cells is None else cells
    return [c.source for c in cells if c.cell_type == "markdown"]


def _sem_codigo(fonte: str) -> str:
    """Tira do markdown os blocos e trechos de código (entre crases)."""
    fonte = re.sub(r"```.*?```", " ", fonte, flags=re.DOTALL)
    return re.sub(r"`[^`]*`", " ", fonte)


def _sem_comentarios(source):
    return "\n".join(line for line in source.splitlines()
                     if not line.lstrip().startswith("#"))


def test_tst_4_3_a_toda_celula_codigo_precedida_por_markdown() -> None:
    """DF-3: nenhuma célula de código substantiva é órfã (sem markdown antes)."""
    cells = _carregar_notebook().cells
    violacoes: list[str] = []
    for i, cell in enumerate(cells):
        if cell.cell_type != "code":
            continue
        fonte = (cell.source or "").strip()
        if not fonte:
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
                f"  célula {i} (code) órfã; anterior é '{tipo_anterior}'. "
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


def test_secoes_aprovadas_em_ordem() -> None:
    """As seções 0 a 10 do plano aprovado aparecem uma vez cada, em ordem."""
    cabecalhos = [
        linha
        for fonte in _markdown()
        for linha in fonte.splitlines()
        if _SECAO_RE.match(linha)
    ]
    numeros = [int(_SECAO_RE.match(linha).group(1)) for linha in cabecalhos]
    assert numeros == list(TITULOS_DAS_SECOES)
    for linha, (numero, titulo) in zip(
        cabecalhos, TITULOS_DAS_SECOES.items(), strict=True
    ):
        assert linha.startswith(f"## {numero}. {titulo}"), linha


def test_toda_secao_com_codigo_tem_leitura() -> None:
    """Depois do código de cada seção há uma célula que interpreta a saída."""
    for numero, cells in _secoes(_carregar_notebook().cells).items():
        tipos = [c.cell_type for c in cells]
        if numero is None or "code" not in tipos:
            continue
        depois_do_codigo = cells[tipos.index("code"):]
        assert any(
            c.cell_type == "markdown" and c.source.lstrip().startswith("**Leitura")
            for c in depois_do_codigo
        ), f"Seção {numero} sem célula de leitura depois do código."


def test_dois_controles() -> None:
    """Dois controles: perdas por modo e faixa (§2) e orçamento humano (§6)."""
    codigo = "\n".join(
        _sem_comentarios(c.source) for c in _carregar_notebook().cells
        if c.cell_type == "code"
    )
    assert codigo.count("nbh.interagir(") == 2


def test_controles_deixam_figura_estatica_sem_duplicar() -> None:
    """Cada controle desenha uma figura estática e só depois chama o widget.

    A figura estática fica na saída para o arquivo exportado e para quem não tem
    ``ipywidgets``; sem fallback em ``if not``, ela não sai em dobro.
    """
    for cell in _carregar_notebook().cells:
        if cell.cell_type != "code" or "nbh.interagir(" not in cell.source:
            continue
        tree = ast.parse(cell.source)
        assert "if not nbh.interagir(" not in cell.source
        for forbidden in (".plot(", "subplots(", "plt.figure("):
            assert forbidden not in cell.source
        wrappers = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name.startswith("_ui_")]
        assert len(wrappers) == 1
        wrapper = ast.get_source_segment(cell.source, wrappers[0])
        assert "plt.show()\n    plt.close(fig)" in wrapper
        chamadas = [
            ast.unparse(node.value.func) for node in tree.body
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        ]
        assert chamadas == [wrappers[0].name, "nbh.interagir"], chamadas


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
    projeto = {
        "nb_helpers", "nb_fila", "synthetic_data", "synthetic_linkage",
        "gzcmd_record_linkage",
    }
    for nome in list(sys.modules):
        if nome.split(".")[0] in projeto:
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
        elif secao is not None and secao >= 4 and secao not in primeiras:
            primeiras[secao] = cell.source
    # A §10 é só texto; todas as seções com código de 4 a 9 têm guarda, porque
    # dependem da base gerada na §3.
    assert set(primeiras) == set(range(4, 10))
    for numero, source in primeiras.items():
        assert source.startswith(build_notebook.GUARDA_SESSAO), f"Seção {numero}."
    with pytest.raises(AssertionError, match="Executar antes"):
        exec(build_notebook.GUARDA_SESSAO, {})
    exec(build_notebook.GUARDA_SESSAO, {"base": object()})


def test_bootstrap_esquece_modulos_antigos(monkeypatch):
    root = Path(build_notebook.__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    monkeypatch.setattr(sys, "path", list(sys.path))
    _proteger_modulos_do_projeto(monkeypatch)
    for nome in ("nb_helpers", "nb_fila"):
        monkeypatch.setitem(sys.modules, nome, SimpleNamespace(__name__=nome))
    exec(_celula_codigo_contendo("def _find_repo_root("), {})
    assert "nb_helpers" not in sys.modules
    assert "nb_fila" not in sys.modules


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
    """Custos vêm de ``nb_fila`` (os da tese), nunca de números nas células."""
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
            for traco in ("—", "–"):
                assert traco not in cell.source, f"Travessão no markdown {i}."


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


def test_perguntas_provaveis_recolhidas_no_fim_de_cada_secao() -> None:
    secoes = _secoes(_carregar_notebook().cells)
    for numero, perguntas in build_notebook.PERGUNTAS_PROVAVEIS.items():
        ultima = secoes[numero][-1]
        assert ultima.cell_type == "markdown", numero
        assert ultima.source.startswith("**❓ Perguntas prováveis."), numero
        assert ultima.source.count("<details>") == len(perguntas), numero
        assert "<details open" not in ultima.source, f"Pergunta aberta: seção {numero}."
        for pergunta, _ in perguntas:
            assert f"<summary>{pergunta}</summary>" in ultima.source, pergunta
    total = sum(len(p) for p in build_notebook.PERGUNTAS_PROVAVEIS.values())
    assert total >= 10


# Formas proibidas no corpo do texto (guia de estilo da tese, seção 3.6).
# Comparação sem distinguir maiúsculas, exceto siglas.
TERMOS_PROIBIDOS = (
    "zona cinzenta", "zona cinza", "grey zone", "gray area", "revisão clerical",
    "não vínculo", "não-vínculo", "não-par", "record linkage", "gold standard",
    "padrão-ouro", "padrão ouro", "floresta aleatória", "linkagem",
    "par candidato", "pares duvidosos", "bancos de dados",
    # Decisão de 2026-11-11: *expected loss* glosado como "custo esperado".
    "perda esperada", "perdas esperadas",
)
SIGLAS_PROIBIDAS = ("PPV", "SINAN", "SIH-SUS")


def test_terminologia_da_tese_no_markdown() -> None:
    for i, fonte in enumerate(_markdown()):
        texto = _sem_codigo(fonte)
        minusculo = texto.lower()
        for termo in TERMOS_PROIBIDOS:
            assert termo not in minusculo, f"'{termo}' no markdown {i}."
        for sigla in SIGLAS_PROIBIDAS:
            assert not re.search(rf"\b{sigla}\b", texto), f"'{sigla}' no markdown {i}."
        assert not re.search(r"\bRL\b", texto), f"Sigla RL no markdown {i}."
        # "precisão" só dentro de nomes consagrados de métrica.
        sem_metricas = re.sub(
            r"precisão média|curva precisão-sensibilidade|auc-pr", " ", minusculo
        )
        assert "precisão" not in sem_metricas, f"'precisão' como VPP no markdown {i}."
        # "área cinza" só dentro da glosa de grey area.
        assert texto.count("área cinza") == texto.count("*grey area* (área cinza)"), i
        # match/nonmatch só como rótulo de código (entre crases, em maiúsculas).
        assert not re.search(r"\b(?:match|matches|nonmatch)\b", texto), (
            f"match/nonmatch fora de código no markdown {i}."
        )
        # Em prosa, o método é *Platt scaling*; identificadores ficam no código.
        assert not re.search(r"\bPlatt\b(?! scaling)", texto), (
            f"'Platt' sem 'scaling' no markdown {i}."
        )


def test_codigo_sem_termos_obsoletos() -> None:
    for i, cell in enumerate(_carregar_notebook().cells):
        if cell.cell_type == "code":
            fonte = cell.source.lower()
            for termo in ("zona cinzenta", "revisão clerical"):
                assert termo not in fonte, f"'{termo}' na célula de código {i}."


TERMOS_DO_GLOSSARIO = (
    "*linkage* (relacionamento de registros)",
    "blocagem (*blocking*)",
    "*grey area* (área cinza)",
    "*expected loss* (custo esperado de cada ação)",
    "curva precisão-sensibilidade (*precision-recall*)",
    "mesma pessoa (verdade da simulação)",
    "Medidas Contínuas de Diferença de Datas",
    "*Platt scaling*",
    "ponto de corte",
    "potenciais pares",
    "escore",
    "nota final",
    "faixa intermediária da nota final",
    "referência operacional",
    "par verdadeiro",
    "não par",
    "vínculo",
    "revisão manual de origem",
    "valor preditivo positivo (VPP)",
    "sensibilidade",
    "curva de esforço",
    "MACD",
)


def test_glossario_define_os_termos_da_tese() -> None:
    glossario = "\n".join(_markdown(_secoes(_carregar_notebook().cells)[0]))
    faltando = [termo for termo in TERMOS_DO_GLOSSARIO if termo not in glossario]
    assert not faltando, f"Termos ausentes do glossário: {faltando}"
    abertura = "\n".join(_markdown(_secoes(_carregar_notebook().cells)[None]))
    for primeira_mencao in (
        "*linkage* (relacionamento de registros)",
        "Sistema de Informações sobre Mortalidade (SIM)",
        "Sistema de Informação de Agravos de Notificação (Sinan)",
    ):
        assert primeira_mencao in abertura, primeira_mencao


# Valores medidos na base sintética (semente 42) durante o desenvolvimento.
# Não podem aparecer fixos no markdown: o código os imprime.
NUMEROS_SINTETICOS_MEDIDOS = (
    "30.255", "30255", "30.006", "30006", "0,996", "0,114", "0,906", "0,236",
    "0,993", "0,338", "0,997", "0,351", "1.325", "19.758", "2.158", "22,7",
    "16,2", "−72", "-72", "−175", "-175", "−54", "-54", "−166", "-166",
)
INTEIROS_SINTETICOS_MEDIDOS = ("249", "570", "263", "225")
# Números com vírgula decimal ou separador de milhar admitidos fora das seções
# 0 e 10: parâmetros do método, não resultados.
DECIMAIS_PERMITIDOS = {"0,9", "2,5", "97,5"}
MILHARES_PERMITIDOS = {"2.000"}


def _isolado(valor: str) -> str:
    return rf"(?<![\d.,]){re.escape(valor)}(?!\d|[.,]\d)"


def test_sem_numeros_sinteticos_fixos_no_markdown() -> None:
    for numero, cells in _secoes(_carregar_notebook().cells).items():
        for fonte in _markdown(cells):
            for medido in NUMEROS_SINTETICOS_MEDIDOS:
                assert medido not in fonte, (numero, medido)
            for medido in INTEIROS_SINTETICOS_MEDIDOS:
                assert not re.search(_isolado(medido), fonte), (numero, medido)
            if numero in (0, 10):
                continue
            decimais = set(re.findall(r"(?<![\d.,])\d+,\d+", fonte))
            milhares = set(re.findall(r"(?<![\d.,])\d{1,3}(?:\.\d{3})+(?!\d)", fonte))
            assert decimais <= DECIMAIS_PERMITIDOS, (numero, decimais)
            assert milhares <= MILHARES_PERMITIDOS, (numero, milhares)


# Números da tese (conclusao.tex) que só podem aparecer na seção 10.
NUMEROS_DA_TESE = (
    "501", "125", "1.387", "2.521", "23.443", "1.777", "1.203", "16,0", "34,7",
    "6,4", "247", "61.485",
)
RELEASE_DA_TESE = (
    "https://github.com/marco-jardim/phd-research/releases/tag/tese-2026-11-11-7e91702"
)
ROTULOS_DA_CONCLUSAO = (
    "sec:concl-recomendacoes", "sec:concl-limitacoes", "sec:concl-sintese",
)


def test_numeros_da_tese_so_na_secao_10() -> None:
    secoes = _secoes(_carregar_notebook().cells)
    for numero, cells in secoes.items():
        if numero == 10:
            continue
        for fonte in _markdown(cells):
            for valor in NUMEROS_DA_TESE:
                assert not re.search(_isolado(valor), fonte), (numero, valor)
    secao_10 = "\n".join(_markdown(secoes[10]))
    for valor in ("125", "501", "29", "51", "1.387", "2.521", "23.443", "16,0"):
        assert re.search(_isolado(valor), secao_10), valor
    # A fonte é o rótulo da seção da conclusão e a release pública, não a linha.
    assert "conclusao.tex" in secao_10
    assert RELEASE_DA_TESE in secao_10
    for rotulo in ROTULOS_DA_CONCLUSAO:
        assert rotulo in secao_10, rotulo
    assert not re.search(r"\blinhas? \d", secao_10), "Fonte por número de linha."


def test_usa_a_configuracao_recomendada_e_o_perfil_historico() -> None:
    codigo = "\n".join(
        c.source for c in _carregar_notebook().cells if c.cell_type == "code"
    )
    for trecho in (
        "sl.gerar_base(", "nf.prob_calibrada_oof(",
        "nf.ordenacoes(", "nf.curva_esforco(", "nf.revisoes_para_meta(",
        "nf.bootstrap_diferenca(", "nf.politica_v31(", "nf.mesma_carga(",
        "rrf(", "run_v3(", 'p_cal="platt"', 'policy_version="v3"',
        "sl.sem_colunas_latentes(", "mesma pessoa (verdade da simulação)",
    ):
        assert trecho in codigo, trecho
    # A base sintética não tem canal de vazamento entre potenciais pares do
    # mesmo óbito: a partição por linha fica fora do notebook (só no texto).
    assert "prob_por_linha(" not in codigo
    assert "Claude Haiku 5.5" in "\n".join(_markdown())


def test_sem_dados_reais() -> None:
    for cell in _carregar_notebook().cells:
        assert "COMPARADORSEMIDENT" not in cell.source
