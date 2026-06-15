"""Gerador determinístico do notebook ``gzcmd_passo_a_passo.ipynb``.

Por que um *builder*?
---------------------
O notebook da apresentação cresce fase a fase (Wave 2 do plano) e precisa ser
**reprodutível**, **versionável com saídas limpas** e **auditável célula a célula**
(regra DF-3: toda célula de código "substantiva" é precedida por markdown
explicativo). Manter o ``.ipynb`` (JSON volumoso) à mão é frágil; descrevê-lo como
uma lista estruturada de células em Python torna o processo determinístico e
fácil de revisar em *diffs*.

O artefato versionado/entregue continua sendo o ``.ipynb`` — este script apenas o
(re)gera sem saídas. Para regenerar::

    python notebooks/build_notebook.py

As células ficam organizadas em blocos por fase do plano. Cada bloco é uma lista
de tuplas ``(tipo, texto)`` com ``tipo in {"md", "code"}``.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

# ---------------------------------------------------------------------------
# Bootstrap: célula de código que torna o notebook executável de qualquer cwd
# (papermill/nbconvert) localizando a raiz do repositório e ajustando sys.path.
# ---------------------------------------------------------------------------
BOOTSTRAP = """\
# Bootstrap de ambiente: localiza a raiz do repositório e configura os caminhos.
# (Necessário para que `import synthetic_data` funcione em execução headless.)
import sys
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return start


REPO_ROOT = _find_repo_root(Path.cwd().resolve())
NB_DIR = REPO_ROOT / "notebooks"
for _path in (str(REPO_ROOT), str(NB_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

DATA_DIR = REPO_ROOT / "data" / "synthetic"
DATA_DIR.mkdir(parents=True, exist_ok=True)
print(f"Raiz do repositório: {REPO_ROOT}")"""


# ===========================================================================
# FASE 2.1 — Estrutura, Narrativa e Setup
# ===========================================================================
FASE_2_1: list[tuple[str, str]] = [
    (
        "md",
        """\
# GZ-CMD++ v3 — Passo a Passo sobre um Dataset Sintético

> **Material didático (PT-BR)** para público técnico/acadêmico. Percorremos, estágio
> por estágio, o motor de decisão **GZ-CMD++ v3** de *record linkage* (vinculação de
> registros), usando um **dataset 100% sintético** gerado por código determinístico.
> Nenhum dado real ou pessoal (PII) é utilizado.

Ao longo do notebook você verá, para cada estágio do pipeline:

1. **Objetivos de aprendizagem** ("ao final você será capaz de…");
2. **Intuição antes do formalismo** — primeiro a pergunta de negócio, depois a matemática;
3. Um **exemplo-fio-condutor** (um par "herói") seguido do início ao fim;
4. **Código real** da biblioteca `gzcmd_record_linkage` (não reimplementações);
5. Uma **visualização** com propósito e um **recap** ao fim da seção.

---

## Glossário rápido

| Termo | Definição |
|-------|-----------|
| **Record linkage** | Decidir se dois registros (ex.: de bases distintas) referem-se à **mesma entidade** (pessoa). |
| **Par comparador** | Uma dupla (registro de referência × registro candidato) com subscores de similaridade. |
| **`nota_final`** | Escore agregado de similaridade do par (escala ~0–10) produzido pelo comparador OpenRecLink. |
| **Banda** | Faixa discreta de `nota_final` (ex.: `grey_mid`) usada para perfis de erro e monitoramento. |
| **Zona cinzenta** | Faixa intermediária de notas onde a decisão automática é arriscada. |
| **Calibração** | Transformar um escore em **probabilidade** confiável de match (`p_cal` ∈ [0,1]). |
| **Guardrail** | Regra de segurança determinística que força MATCH/NONMATCH/revisão. |
| **Triagem (*triage*)** | Decisão final por **custo esperado**: MATCH, NONMATCH ou LLM_REVIEW. |
| **ECE / Brier** | Métricas quantitativas de **qualidade de calibração** (definidas na seção de calibração). |

> Termos adicionais (ex.: *held-out*, *base rate*, *EVR*) são definidos no ponto em que aparecem.""",
    ),
    (
        "md",
        """\
## 1. Contexto: por que *record linkage* é difícil?

**Objetivos de aprendizagem.** Ao final desta seção você será capaz de:

- **explicar** o problema de vincular registros sem um identificador único confiável;
- **identificar** a "zona cinzenta" e por que ela exige tratamento especial;
- **descrever** em alto nível os cinco estágios do GZ-CMD++ v3.

**Intuição.** Imagine duas bases de dados de saúde. Em uma, um registro diz
*"Maria S. Souza, nascida em 12/03/1981"*; na outra, *"Maria Silva de Souza,
12/03/1981"*. É a **mesma pessoa**? Provavelmente — mas e se a data fosse
*13/03/1981*? Ou se o sobrenome fosse completamente diferente? Quando **não há
CPF confiável** ligando os registros, precisamos **decidir sob incerteza** a partir
de **subscores de similaridade** (nome, nome da mãe, data de nascimento, endereço…).

O comparador agrega esses subscores em uma **`nota_final`**. Notas muito altas são
quase certamente *matches*; muito baixas, quase certamente não. O problema mora no
**meio** — a **zona cinzenta** — onde aceitar ou rejeitar automaticamente custa caro
(falsos positivos e falsos negativos têm consequências distintas).""",
    ),
    (
        "md",
        """\
## 2. Visão geral do pipeline GZ-CMD++ v3

O motor é uma sequência de **cinco estágios**. Cada um será reproduzido isoladamente
nas próximas seções, usando as funções reais da biblioteca:

```
  CSV comparador (;/,)                      [loader.load_comparador_csv]
        │  feature engineering (agregados + flags + MACD opcional)
        ▼
  nota_final ──► BandAssigner.assign ──► band (low … high)     [bands]
        │
        ▼
  calibração  ──► compute_p_cal / fit_platt ──► p_cal ∈ [0,1]  [calibration]
        │
        ▼
  apply_guardrails ──► guardrail / reason                      [guardrails]
        │   (óbito×diagnóstico, homonímia, âncoras)
        ▼
  PolicyEngineV3.triage ──► action ∈ {MATCH, NONMATCH, LLM_REVIEW}
                            por CUSTO ESPERADO                  [policy engine]
```

Dois **modos** de operação calibram o apetite a risco:

- **`vigilancia`** — prioriza *recall* (recuperar o máximo de óbitos/matches);
- **`confirmacao`** — prioriza *precision* (listas de altíssima confiança).

> **Nota metodológica importante (será detalhada na seção de calibração):** a
> ferramenta, em `run_v3`, calibra **in-sample** (ajusta e pontua nas mesmas linhas).
> Isso é ótimo para *reproduzir* a ferramenta, mas **não** mede generalização. Por
> isso adotaremos **duas rotas** explícitas: **(A) reprodução fiel** e
> **(B) metodologia correta** (treino/teste *held-out*).""",
    ),
    (
        "md",
        """\
## 3. Objetivos da apresentação

Ao final deste notebook, teremos demonstrado, com números e gráficos honestos:

1. Como cada estágio do GZ-CMD++ v3 transforma os dados (load → bands → calibração → guardrails → triagem);
2. Como **calibrar** escores em probabilidades e como **medir** essa calibração (ECE, Brier) **sem vazamento**;
3. Como a **política de custo esperado** decide entre MATCH/NONMATCH/revisão nos dois modos;
4. Que o passo-a-passo manual **reconcilia** com a função de alto nível `run_v3` (rota A);
5. Como avaliar **generalização** corretamente (rota B, *held-out*, múltiplas seeds).

> **Honestidade científica.** Como os dados são sintéticos, conhecemos a
> **posterior verdadeira** `p*(x)` que gerou cada rótulo. Usaremos `p*` como
> *ground-truth* para **provar** (não apenas ilustrar) que a calibração funciona.""",
    ),
    (
        "md",
        """\
## 4. Setup do ambiente e geração do dataset

**O que vamos fazer a seguir.** A próxima célula prepara o ambiente: importa as
bibliotecas, configura os caminhos do repositório e fixa a *seed*. Em seguida,
geramos o dataset sintético e o salvamos como CSV no formato que o `loader` espera
(`;` como separador, `,` como decimal).""",
    ),
    ("code", BOOTSTRAP),
    (
        "md",
        """\
Agora importamos o `pandas` e o módulo `synthetic_data` (que gera os dados). Mais
adiante, à medida que precisarmos, traremos também `numpy`, `matplotlib` e o módulo
`nb_helpers` (utilitários de plot e métricas de calibração).""",
    ),
    (
        "code",
        """\
import pandas as pd

import synthetic_data
from gzcmd_record_linkage import __version__ as gzcmd_version

pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 140)

SEED = 42
print(f"gzcmd_record_linkage v{gzcmd_version} | seed global = {SEED}")""",
    ),
    (
        "md",
        """\
**Geração do dataset.** Chamamos `synthetic_data.generate_comparador(...)` com
`scenarios="all"`, que além de ~600 pares aleatórios injeta **7 cenários narrativos
rotulados** (match óbvio, não-match óbvio, homônimo, óbito-antes-do-diagnóstico,
nome-da-mãe-ausente, datas-invertidas e um caso clássico de zona cinzenta). O objeto
retornado expõe:

- `.frame` — o DataFrame no **schema cru** do comparador (entrada do pipeline);
- `.p_true` — a **posterior verdadeira** `p*(x)` de cada par (coluna de **validação**,
  jamais consumida pelo pipeline);
- `.meta` — metadados do gerador (seed, taxa de match, parâmetros da verdade-base).""",
    ),
    (
        "code",
        """\
dataset = synthetic_data.generate_comparador(seed=SEED, n_pairs=600, scenarios="all")
df_raw = dataset.frame
p_true = dataset.p_true
meta = dataset.meta

CSV_PATH = DATA_DIR / "comparador_sintetico.csv"
synthetic_data.to_comparador_csv(df_raw, CSV_PATH)

print(f"Pares gerados......: {len(df_raw)}")
print(f"Taxa de match (real): {meta['match_ratio_realized']:.3f}")
print(f"Verdade-base p*(s) = sigmoide({meta['true_slope']:.3f}*nota + ({meta['true_intercept']:.3f}))")
print(f"CSV salvo em.......: {CSV_PATH.relative_to(REPO_ROOT)}")""",
    ),
    (
        "md",
        """\
## 5. Primeiro olhar nos dados

**O que vamos fazer a seguir.** Antes de qualquer modelagem, olhamos o formato cru
dos dados: as primeiras linhas e um **dicionário de colunas** explicando cada grupo
de subscores. Isso ancora a intuição sobre *o que o comparador mediu*.""",
    ),
    (
        "code",
        """\
colunas_visao = [
    synthetic_data.COMPREC,
    synthetic_data.REFREC,
    "PASSO",
    "PAR",
    "nota final",
    "NOME qtd frag iguais",
    "NOMEMAE qtd frag iguais",
    "DTNASC dt iguais",
    "CODMUNRES local igual",
]
df_raw[colunas_visao].head(8)""",
    ),
    (
        "md",
        """\
### Dicionário de colunas (grupos principais)

| Coluna (crua) | Significado |
|---------------|-------------|
| `COMPREC,C,12,0` / `REFREC,C,12,0` | IDs do registro **comparador** e de **referência** (formam o par). |
| `PASSO` | Passo de *blocking* que originou o par. |
| `PAR` | Rótulo: `1`/`2` = match, `0` = não-match (origem do `TARGET`). |
| `nota final` | Escore agregado de similaridade (~0–10). É **o** observável que calibramos. |
| `NOME *` | Subscores de similaridade do **nome** (frações de fragmentos iguais). |
| `NOMEMAE *` | Subscores do **nome da mãe** (decisivos na zona cinzenta). |
| `DTNASC *` | Comparações de **data de nascimento** (iguais, aproximadas, invertidas). |
| `ENDERECO *` | Subscores de **endereço** (via, número, proximidade textual). |
| `CODMUNRES local igual` | Indicador de **município de residência** coincidente. |
| `R_DTNASC` / `C_DTNASC` | Datas de nascimento (YYYYMMDD) de cada registro. |
| `R_DTOBITO` / `C_DTDIAG` | Data de **óbito** (ref.) e de **diagnóstico** (comp.) — usadas no guardrail temporal. |

> A coluna de validação `p_true` **não** está no CSV de entrada — ela vive apenas no
> objeto `dataset` para checarmos a calibração mais adiante.""",
    ),
    (
        "md",
        """\
## 6. Apresentando o "herói": nosso exemplo-fio-condutor

**Intuição didática.** Para não nos perdermos em médias e agregados, vamos eleger
**um par específico** e segui-lo em **todos** os estágios. Escolhemos o cenário
`zona_cinzenta`: um par com `nota_final` no meio da escala — nem claramente match,
nem claramente não-match. É exatamente o tipo de caso que o GZ-CMD++ v3 foi
desenhado para tratar com cuidado (calibração + guardrails + triagem por custo).

A célula a seguir localiza o herói pelo mapa de cenários em `meta["scenarios"]` e
mostra seu estado inicial. Voltaremos a este "card" a cada estágio.""",
    ),
    (
        "code",
        '''\
HERO_SCENARIO = "zona_cinzenta"
hero_pos = meta["scenarios"][HERO_SCENARIO]
hero_idx = df_raw.index[hero_pos]


def card_heroi(df: pd.DataFrame, idx, colunas) -> pd.DataFrame:
    """Mostra o estado atual do par herói como uma coluna (transposto p/ leitura)."""
    presentes = [c for c in colunas if c in df.columns]
    return df.loc[[idx], presentes].T.rename(columns={idx: "herói"})


card_heroi(
    df_raw,
    hero_idx,
    [synthetic_data.COMPREC, synthetic_data.REFREC, "PASSO", "PAR", "nota final"],
)''',
    ),
    (
        "md",
        """\
**Recap da seção.** Motivamos o problema de *record linkage* e a zona cinzenta,
mapeamos os cinco estágios do GZ-CMD++ v3, geramos um dataset sintético válido no
schema do `loader` (com a posterior verdadeira `p*` reservada para validação) e
elegemos um par **herói** para acompanhar. **A seguir:** carregar o CSV pelo
`loader` e observar a *feature engineering*.""",
    ),
]


# ===========================================================================
# FASE 2.2 — Carga e Feature Engineering
# ===========================================================================
FASE_2_2: list[tuple[str, str]] = [
    (
        "md",
        """\
## 7. Carga dos dados e *feature engineering*

**Objetivos de aprendizagem.** Ao final desta seção você será capaz de:

- **explicar** o que o `loader` faz ao ler o CSV cru (resolução de colunas, *parsing* de datas);
- **identificar** os três grupos de colunas que ele produz: **agregados**, **flags** e **MACD**;
- **interpretar** a distribuição da `nota_final` por classe (match × não-match) e por que as classes **se sobrepõem**.

**Intuição.** O comparador entrega dezenas de subscores crus (frações de fragmentos
de nome iguais, comparações de data, proximidade de endereço…). Trabalhar com todos
eles, um a um, é inviável e ruidoso. O `loader` então **agrega** subscores correlatos
em poucos escores por dimensão (nome, data, endereço, município), cria **flags**
de ausência/risco e, opcionalmente, *features* **MACD** (interações entre nome
perfeito e distância temporal). Esse é o material sobre o qual o resto do pipeline opera.

**O que vamos fazer a seguir.** Carregar o CSV que salvamos com a função real
`load_comparador_csv` e inspecionar cada grupo de colunas que ela engenheira.""",
    ),
    (
        "code",
        """\
from gzcmd_record_linkage.loader import LoadConfig, load_comparador_csv

df = load_comparador_csv(CSV_PATH, cfg=LoadConfig(macd_enabled=True))
print(f"Linhas: {len(df)} | Colunas: {df.shape[1]}")
print(f"TARGET derivado de PAR -> positivos: {int(df['TARGET'].sum())} / {len(df)}")""",
    ),
    (
        "md",
        """\
### 7.1 Colunas **agregadas** (subscores resumidos por dimensão)

O `loader` combina os subscores crus em escores compactos no intervalo **[0, 1]**:

| Coluna agregada | Como é formada (resumo) |
|-----------------|--------------------------|
| `nome_score_total` | combinação ponderada dos fragmentos de **nome** iguais. |
| `mae_score_total` | idem para o **nome da mãe**. |
| `dtnasc_score_total` | média das comparações de **data de nascimento**. |
| `endereco_score_total` | média dos subscores de **endereço**. |
| `municipio_score` | indicador de **município** coincidente. |

A célula abaixo mostra esses agregados ao lado da `nota_final` e do `TARGET`.""",
    ),
    (
        "code",
        """\
colunas_agregadas = [
    "nota_final",
    "TARGET",
    "nome_score_total",
    "mae_score_total",
    "dtnasc_score_total",
    "endereco_score_total",
    "municipio_score",
]
df[colunas_agregadas].describe().T[["mean", "min", "max"]].round(3)""",
    ),
    (
        "md",
        """\
### 7.2 **Flags** de ausência/risco e *features* **MACD**

Além dos agregados, o `loader` cria **flags** binárias que sinalizam situações de
risco (consumidas adiante pelos *guardrails*):

| Flag | Significado |
|------|-------------|
| `mae_missing` | `1` quando **todos** os subscores do nome da mãe são zero (nome da mãe ausente). |
| `dtnasc_all_zero` | `True` quando nenhuma comparação de data de nascimento "pegou". |
| `endereco_zero` | `1` quando todos os subscores de endereço são zero. |
| `diff_ano` | diferença absoluta de **ano** de nascimento entre os registros. |

As colunas **MACD** (ativadas por `LoadConfig(macd_enabled=True)`) capturam
**interações** — por exemplo, `macd_nome_perf_x_date_far` marca o caso perigoso de
**nome perfeito porém datas distantes** (forte sinal de homonímia). A célula abaixo
lista as colunas MACD presentes.""",
    ),
    (
        "code",
        """\
flags = ["mae_missing", "dtnasc_all_zero", "endereco_zero", "diff_ano"]
macd_cols = sorted(c for c in df.columns if c.startswith("macd_"))
print("Flags (amostra de contagens):")
print(df[flags].apply(lambda s: s.astype(float)).agg(["mean", "max"]).round(3).T)
print(f"\\nColunas MACD presentes ({len(macd_cols)}):")
for c in macd_cols:
    print(f"  - {c}")""",
    ),
    (
        "md",
        """\
### 7.3 Visualização: distribuição da `nota_final` por classe

**Pergunta que a figura responde:** *as notas separam perfeitamente matches de
não-matches?* Se separassem, não haveria zona cinzenta — nem necessidade de
calibração ou de revisão. Esperamos ver **sobreposição** na faixa intermediária:
é exatamente onde a decisão é difícil (e onde o dataset sintético foi desenhado para
ter ambiguidade real, evitando circularidade).""",
    ),
    (
        "code",
        """\
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4.5))
bins = [i * 0.5 for i in range(0, 23)]  # 0.0 .. 11.0 em passos de 0.5
ax.hist(
    df.loc[df["TARGET"] == 0, "nota_final"],
    bins=bins,
    alpha=0.6,
    label="não-match (TARGET=0)",
    color="#4C72B0",
)
ax.hist(
    df.loc[df["TARGET"] == 1, "nota_final"],
    bins=bins,
    alpha=0.6,
    label="match (TARGET=1)",
    color="#C44E52",
)
ax.set_xlabel("nota_final (escore agregado de similaridade)")
ax.set_ylabel("frequência (nº de pares)")
ax.set_title("Distribuição da nota_final por classe — note a sobreposição na zona cinzenta")
ax.legend()
fig.tight_layout()
plt.show()""",
    ),
    (
        "md",
        """\
### 7.4 O herói após a *feature engineering*

Reencontramos nosso par **herói** (`zona_cinzenta`), agora com as colunas
engenheiradas. Repare na `nota_final` intermediária e nos escores parciais — é um
caso genuinamente ambíguo.""",
    ),
    (
        "code",
        """\
card_heroi(
    df,
    hero_idx,
    [
        "nota_final",
        "TARGET",
        "nome_score_total",
        "mae_score_total",
        "dtnasc_score_total",
        "municipio_score",
        "mae_missing",
        "diff_ano",
    ],
)""",
    ),
    (
        "md",
        """\
**Recap da seção.** Carregamos o CSV com o `loader` real e vimos como ele transforma
dezenas de subscores crus em **agregados** [0,1], **flags** de risco e *features*
**MACD**. A figura confirmou a **sobreposição** das classes na zona cinzenta — a
razão de ser de todo o pipeline. **A seguir:** transformar a `nota_final` contínua
em **bandas** discretas com o `BandAssigner`.""",
    ),
]


# ---------------------------------------------------------------------------
# Montagem do notebook
# ---------------------------------------------------------------------------
# A ordem das fases reflete a construção incremental do plano (Wave 2).
ALL_PHASES: list[list[tuple[str, str]]] = [
    FASE_2_1,
    FASE_2_2,
]


def build() -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
        "gzcmd_notebook": {
            "title": "GZ-CMD++ v3 — Passo a Passo",
            "generated_by": "notebooks/build_notebook.py",
        },
    }
    cells: list[nbformat.NotebookNode] = []
    for phase in ALL_PHASES:
        for kind, text in phase:
            if kind == "md":
                cells.append(new_markdown_cell(text))
            elif kind == "code":
                cells.append(new_code_cell(text))
            else:  # pragma: no cover - guarda defensiva
                raise ValueError(f"Tipo de célula desconhecido: {kind!r}")
    nb.cells = cells
    return nb


def main() -> Path:
    nb = build()
    out_path = Path(__file__).resolve().parent / "gzcmd_passo_a_passo.ipynb"
    nbformat.write(nb, out_path)
    print(f"Notebook gerado: {out_path} ({len(nb.cells)} células)")
    return out_path


if __name__ == "__main__":
    main()
