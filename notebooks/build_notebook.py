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
print(df[flags].astype(float).agg(["mean", "max"]).round(3).T)
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


# ===========================================================================
# FASE 2.3 — Atribuição de Bandas
# ===========================================================================
FASE_2_3: list[tuple[str, str]] = [
    (
        "md",
        """\
## 8. Atribuição de bandas

**Objetivos de aprendizagem.** Ao final desta seção você será capaz de:

- **explicar** por que discretizar a `nota_final` contínua em **bandas** nomeadas;
- **calcular** a banda de um par a partir das fronteiras definidas na configuração;
- **interpretar** o papel das bandas `grey_*` como a **zona de incerteza** do sistema.

**Intuição.** A `nota_final` é um número contínuo, mas a operação precisa de
**categorias acionáveis**: "isto é claramente alto", "isto está no meio", "isto é
baixo". As **bandas** fazem esse recorte. Elas servem a três propósitos no
GZ-CMD++ v3: (1) perfis de **erro do revisor LLM** por banda, (2) **monitoramento de
drift** por fatia e (3) leitura humana rápida. As bandas `grey_low`, `grey_mid` e
`grey_high` isolam justamente a **zona cinzenta** — onde a decisão automática é
arriscada e a calibração/triagem ganham importância.

**O que vamos fazer a seguir.** Carregar a **configuração real** da biblioteca
(o mesmo `gzcmd_v3_config.yaml` que `run_v3` usa), instanciar o `BandAssigner` a
partir dela e atribuir uma banda a cada par.""",
    ),
    (
        "code",
        """\
from importlib.resources import files

from gzcmd_record_linkage.bands import BandAssigner
from gzcmd_record_linkage.config import load_config

CONFIG_PATH = str(files("gzcmd_record_linkage") / "gzcmd_v3_config.yaml")
cfg = load_config(CONFIG_PATH)

assigner = BandAssigner.from_config(cfg)
df["band"] = assigner.assign(df["nota_final"])

print("Contagem de pares por banda:")
ordem_bandas = ["low", "grey_low", "grey_mid", "grey_high", "near_high", "high"]
print(df["band"].value_counts().reindex(ordem_bandas, fill_value=0).astype(int))""",
    ),
    (
        "md",
        """\
### 8.1 As fronteiras das bandas (lidas da configuração)

A tabela abaixo é construída **diretamente** a partir de `cfg.bands.definitions` —
ou seja, reflete fielmente o que o código usa. A semântica de cada faixa é
`min <= nota < max` (limite superior **exclusivo**), **exceto** a banda `high`, cujo
`inclusive_max=True` a torna `min <= nota <= max`. Notas fora de todas as faixas
recebem `<NA>` (não deveria ocorrer com `nota_final` em [0, ~10]).""",
    ),
    (
        "code",
        """\
fronteiras = pd.DataFrame(
    [
        {
            "banda": d.name,
            "min": d.min,
            "max": d.max,
            "inclui o max?": "sim" if d.inclusive_max else "não",
        }
        for d in cfg.bands.definitions
    ]
)
fronteiras""",
    ),
    (
        "md",
        """\
### 8.2 Visualização: a `nota_final` colorida por banda

**Pergunta que a figura responde:** *como as fronteiras recortam a distribuição de
notas?* Cada barra do histograma é colorida pela banda do seu intervalo. Repare como
as três bandas cinzentas (`grey_low`/`grey_mid`/`grey_high`) cobrem exatamente a
região central — a zona cinzenta que motivou todo o pipeline.""",
    ),
    (
        "code",
        """\
import numpy as np

band_order = ["low", "grey_low", "grey_mid", "grey_high", "near_high", "high"]
band_colors = {
    "low": "#4C72B0",
    "grey_low": "#8172B3",
    "grey_mid": "#CCB974",
    "grey_high": "#DD8452",
    "near_high": "#C44E52",
    "high": "#55A868",
}

edges = np.arange(0.0, 11.001, 0.25)
counts, _ = np.histogram(df["nota_final"].to_numpy(dtype=float), bins=edges)
centers = (edges[:-1] + edges[1:]) / 2.0
center_bands = assigner.assign(pd.Series(centers))

fig, ax = plt.subplots(figsize=(9, 4.5))
for b in band_order:
    mask = (center_bands == b).to_numpy(dtype=bool)
    if mask.any():
        ax.bar(centers[mask], counts[mask], width=0.24, color=band_colors[b], label=b)
ax.set_xlabel("nota_final (escore agregado de similaridade)")
ax.set_ylabel("frequência (nº de pares)")
ax.set_title("nota_final colorida pela banda atribuída")
ax.legend(title="banda", ncol=3)
fig.tight_layout()
plt.show()""",
    ),
    (
        "md",
        """\
### 8.3 O herói recebe sua banda

Nosso par **herói** (`zona_cinzenta`) agora carrega uma banda. Como esperado para um
caso ambíguo, ele cai numa das faixas `grey_*` — o território onde a decisão exige
calibração honesta e, possivelmente, revisão.""",
    ),
    (
        "code",
        """\
card_heroi(df, hero_idx, ["nota_final", "TARGET", "band"])""",
    ),
    (
        "md",
        """\
**Recap da seção.** Carregamos a configuração real, atribuímos bandas com o
`BandAssigner` (limite superior exclusivo, exceto `high`) e visualizamos como as
fronteiras recortam a distribuição. As bandas `grey_*` materializam a **zona de
incerteza**. **A seguir:** transformar a `nota_final` em uma **probabilidade
calibrada** `p_cal` e — criticamente — medir a qualidade dessa calibração **sem
vazamento** (rotas A e B).""",
    ),
]


# ===========================================================================
# FASE 2.4 — Calibração (Platt): rotas A (in-sample) e B (held-out)
# ===========================================================================
FASE_2_4: list[tuple[str, str]] = [
    (
        "md",
        """\
## 9. Calibração: de escore a **probabilidade** confiável

**Objetivos de aprendizagem.** Ao final desta seção você será capaz de:

- **explicar** o que significa uma probabilidade *calibrada* e por que a `nota_final` crua **não** é uma probabilidade;
- **derivar** o *Platt scaling* (`p = σ(a·s + b)`) e interpretar `a` (inclinação) e `b` (viés) geometricamente;
- **distinguir** avaliação **in-sample** (rota A, reproduz a ferramenta) de **held-out** (rota B, mede generalização);
- **medir** a qualidade da calibração com **ECE** e **Brier**, e **validar** o resultado contra a posterior verdadeira `p*`.

**Intuição.** A `nota_final` ordena os pares (quanto maior, mais provável o match),
mas seu valor numérico não é uma probabilidade: "nota 8" não quer dizer "80% de
chance de match". **Calibrar** é aprender a função monótona que converte o escore na
**probabilidade real** de match, `p_cal ∈ [0, 1]`. Com `p_cal` podemos, por exemplo,
afirmar honestamente "este par tem 92% de chance de ser a mesma pessoa" — e é sobre
`p_cal` que a política de custo (seção 11) tomará decisões.""",
    ),
    (
        "md",
        r"""\
### 9.1 Derivação do *Platt scaling*

Seja $s$ a `nota_final` de um par e $y \in \{0,1\}$ o rótulo verdadeiro (1 = match).
O Platt modela a probabilidade de match como uma **regressão logística 1-D** sobre o
escore:

$$ p(s) = \sigma(a\,s + b), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}. $$

- $a$ (**inclinação**, *slope*): controla **quão rápido** a probabilidade sobe com a
  nota. Quanto maior $a$, mais "abrupta" é a transição de não-match para match.
- $b$ (**viés**, *intercept*): **desloca** a curva. O ponto onde $p = 0.5$ ocorre em
  $s^\star = -b/a$ — a "nota de indiferença".

**Ajuste por máxima verossimilhança.** Estimamos $(a, b)$ minimizando a
**log-verossimilhança negativa** (NLL) com **regularização L2 apenas na inclinação**
(exatamente como a biblioteca faz):

$$ \mathcal{L}(a,b) = -\sum_{i} \Big[ y_i \log p(s_i) + (1-y_i)\log\big(1-p(s_i)\big) \Big] \;+\; \tfrac{1}{2}\,\lambda\, a^2. $$

A biblioteca resolve isso por **Newton–Raphson** (atualização $w \leftarrow w - H^{-1}g$,
com $g$ o gradiente e $H$ a Hessiana 2×2), inicializando $b$ no *log-odds* da taxa de
base e $a=0$. O procedimento é **determinístico** (sem aleatoriedade): mesma entrada,
mesma saída — fato que exploraremos na reconciliação com `run_v3` (seção 12).

> **Por que L2 só na inclinação?** Penalizar $a$ evita curvas absurdamente íngremes
> quando há poucos dados; deixar $b$ livre preserva a capacidade de acertar a
> **prevalência** (taxa de base) da amostra.""",
    ),
    (
        "md",
        """\
### 9.2 Rota A — reprodução **fiel** da ferramenta (in-sample)

**O que vamos fazer a seguir.** Ajustar o Platt em **todas** as linhas e pontuar
**essas mesmas linhas** — exatamente o que `run_v3(p_cal="fit_platt")` faz
internamente. Isso reproduz a ferramenta com fidelidade (usaremos isso na seção 12),
mas tem uma armadilha que tornamos explícita logo abaixo.""",
    ),
    (
        "code",
        """\
from gzcmd_record_linkage.calibration import compute_p_cal, fit_platt_from_df

# Rota A: ajuste GLOBAL in-sample (ajusta e pontua nas mesmas linhas).
platt_insample = fit_platt_from_df(df)
df["p_cal"] = compute_p_cal(df, method="platt", model=platt_insample)

print("Modelo Platt (in-sample):")
print(f"  inclinação a (slope)... = {platt_insample.slope:.4f}")
print(f"  viés b (intercept)..... = {platt_insample.intercept:.4f}")
print(f"  nota de indiferença s* = -b/a = {-platt_insample.intercept / platt_insample.slope:.3f}")
print(f"\\nVerdade-base do gerador: a*={meta['true_slope']:.4f}, b*={meta['true_intercept']:.4f}")
print(f"p_cal (in-sample) — min={df['p_cal'].min():.4f}, max={df['p_cal'].max():.4f}")""",
    ),
    (
        "md",
        """\
> ⚠️ **Por que a rota A NÃO mede generalização (vazamento — R-10).** Ajustamos os
> parâmetros usando os **mesmos** rótulos sobre os quais depois medimos o acerto. Um
> *reliability diagram* feito aqui seria **otimista por construção**: o modelo "já
> viu" cada ponto. Para medir calibração de verdade, precisamos de dados **não
> usados no ajuste** — é a rota B.""",
    ),
    (
        "md",
        """\
### 9.3 Rota B — metodologia **correta** (held-out, *group-aware*)

**O que vamos fazer a seguir.** Separar treino/teste de forma **group-aware** por
`COMPREC` (todas as linhas de um mesmo registro ficam do mesmo lado — evita o
vazamento por registro compartilhado, típico de *record linkage*). Ajustamos o Platt
**só no treino** e pontuamos **só no teste**. Toda métrica de calibração honesta vem
daqui.""",
    ),
    (
        "code",
        """\
from nb_helpers import brier_score, expected_calibration_error

train_idx, test_idx = synthetic_data.group_aware_split_indices(
    df, split_by="comprec", test_size=0.3, seed=SEED, group_stratify=True
)
df_train = df.iloc[train_idx]
df_test = df.iloc[test_idx]

# Ajuste SÓ no treino; previsão SÓ no teste.
platt_holdout = fit_platt_from_df(df_train)
p_cal_test = compute_p_cal(df_test, method="platt", model=platt_holdout)
y_test = df_test["TARGET"].to_numpy(dtype=float)

ece_holdout = expected_calibration_error(y_test, p_cal_test.to_numpy(), n_bins=10)
brier_holdout = brier_score(y_test, p_cal_test.to_numpy())

print(f"Treino: {len(df_train)} pares | Teste (held-out): {len(df_test)} pares")
print(f"Platt (treino): a={platt_holdout.slope:.4f}, b={platt_holdout.intercept:.4f}")
print(f"\\nMétricas de calibração NO TESTE (held-out):")
print(f"  ECE (Expected Calibration Error) = {ece_holdout:.4f}  (0 = perfeito)")
print(f"  Brier score....................  = {brier_holdout:.4f}  (menor é melhor)")""",
    ),
    (
        "md",
        """\
### 9.4 *Reliability diagram* (no teste held-out)

**Pergunta que a figura responde:** *quando o modelo diz "70%", a fração real de
matches é mesmo ~70%?* Agrupamos as previsões do **teste** em faixas e comparamos a
**confiança média** (eixo x) com a **acurácia empírica** (eixo y). Pontos sobre a
diagonal = perfeitamente calibrado.""",
    ),
    (
        "code",
        """\
def reliability_points(y_obs, p_hat, n_bins=10):
    \"\"\"Retorna (conf_média, acurácia_empírica, peso) por faixa não-vazia.\"\"\"
    p_hat = np.asarray(p_hat, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p_hat, edges[1:-1]), 0, n_bins - 1)
    conf, acc, wt = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() > 0:
            conf.append(p_hat[m].mean())
            acc.append(y_obs[m].mean())
            wt.append(int(m.sum()))
    return np.array(conf), np.array(acc), np.array(wt)


conf, acc, wt = reliability_points(y_test, p_cal_test.to_numpy(), n_bins=10)

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, 1], [0, 1], "--", color="gray", label="calibração perfeita")
ax.scatter(conf, acc, s=wt * 3, color="#C44E52", alpha=0.8, label="bins (área ∝ nº de pares)")
ax.set_xlabel("confiança média prevista (p_cal)")
ax.set_ylabel("fração empírica de matches (acurácia)")
ax.set_title(f"Reliability diagram — teste held-out\\nECE={ece_holdout:.3f} | Brier={brier_holdout:.3f}")
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.legend(loc="upper left")
fig.tight_layout()
plt.show()""",
    ),
    (
        "md",
        """\
### 9.5 Validação contra a **posterior verdadeira** `p*` (a prova, não a ilustração)

Como os dados são sintéticos, conhecemos `p_true = p*(s)` — a probabilidade real que
gerou cada rótulo. Podemos então perguntar algo que com dados reais é **impossível**:
*o Platt recuperou a verdade?* A figura sobrepõe, no **teste**, o `p_cal` estimado
contra `p*`. Se o método funciona, os pontos seguem a diagonal.

> **Anti-circularidade (DEC-06).** `p*` foi definida **antes** dos rótulos
> (`TARGET ~ Bernoulli(p*)`) e **nunca** entrou no pipeline. Recuperá-la é evidência
> genuína de que a calibração funciona — não uma tautologia.""",
    ),
    (
        "code",
        """\
# p_true está alinhada por posição às linhas do dataset (mesma ordem do CSV).
p_true_test = p_true.to_numpy(dtype=float)[test_idx]
mae_vs_ptrue = float(np.mean(np.abs(p_cal_test.to_numpy() - p_true_test)))

ordem = np.argsort(p_true_test)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot([0, 1], [0, 1], "--", color="gray", label="p_cal = p* (ideal)")
ax.scatter(p_true_test[ordem], p_cal_test.to_numpy()[ordem], s=14, alpha=0.5, color="#4C72B0", label="pares de teste")
ax.set_xlabel("posterior verdadeira p*(s)")
ax.set_ylabel("probabilidade calibrada p_cal (held-out)")
ax.set_title(f"O Platt recupera p*?  |  erro médio |p_cal − p*| = {mae_vs_ptrue:.4f}")
ax.legend(loc="upper left")
fig.tight_layout()
plt.show()

print(f"Erro absoluto médio entre p_cal (held-out) e a verdade p*: {mae_vs_ptrue:.4f}")
print("Interpretação: valores pequenos (~0.01–0.03) provam recuperação da verdade-base.")""",
    ),
    (
        "md",
        """\
### 9.6 Configuração × código: o que a *config* promete e o que o código faz (R-11)

A `gzcmd_v3_config.yaml` descreve a calibração como `method: anchor_platt` com
`by_band: true` (um Platt **por banda**, ancorado). **Porém**, o código realmente
executado por `run_v3` faz um **Platt global** (`fit_platt_from_df`), sem âncoras e
sem separação por banda. 

Tratamos isso com honestidade: **este notebook ensina o que o código faz** (Platt
global). A descrição da config é melhor entendida como **intenção de projeto /
roadmap**, não como comportamento atual. Sinalizar essa divergência é parte do rigor
científico — apresentar a config como se fosse a implementação seria enganoso.""",
    ),
    (
        "md",
        """\
### 9.7 Apêndice (opcional): calibração via modelo ML

A biblioteca também permite `p_cal` a partir de um classificador (`GZCMDClassifier`
com Random Forest/XGBoost), via `predict_proba(df)[:, 1]`. Para **este material
didático** mantemos o **Platt** como método principal (DEC-02): ele é 1-D,
visualizável e **determinístico**, o que o torna ideal para ensinar calibração e para
**reconciliar** com `run_v3` ao bit (seção 12). O caminho XGBoost é mencionado para
completude, mas **não é executado aqui** porque (a) introduz não-determinismo entre
threads/execuções (R-13) e (b) não acrescenta clareza conceitual sobre *calibração*.
A comparação ML × Platt é deixada como exercício para o leitor com `n_jobs=1` + seed
fixa.""",
    ),
    (
        "md",
        """\
### 9.8 O herói recebe sua probabilidade calibrada

Nosso par **herói** (`zona_cinzenta`) agora tem um `p_cal` (mostramos o da rota A,
que cobre todas as linhas). Como esperado para um caso ambíguo, sua probabilidade
**não** é próxima de 0 nem de 1 — fica no meio, sinalizando que a decisão automática
é arriscada e que guardrails/triagem (próximas seções) serão decisivos.""",
    ),
    (
        "code",
        """\
card_heroi(df, hero_idx, ["nota_final", "TARGET", "band", "p_cal"])""",
    ),
    (
        "md",
        """\
**Recap da seção.** Derivamos o *Platt scaling* (`p = σ(a·s + b)`), distinguimos
**rota A** (in-sample, reproduz `run_v3`, mas vaza) de **rota B** (held-out,
*group-aware*, honesta), medimos a calibração com **ECE** e **Brier** no teste e
**provamos** — contra a posterior verdadeira `p*` — que o Platt recupera a verdade.
Também declaramos a divergência **config × código** (R-11). **A seguir:** as regras de
segurança determinísticas — os **guardrails**.""",
    ),
]


# ===========================================================================
# FASE 2.5 — Guardrails determinísticos
# ===========================================================================
FASE_2_5: list[tuple[str, str]] = [
    (
        "md",
        """\
## 10. Guardrails

**Objetivos de aprendizagem.** Ao final desta seção você será capaz de:

- **explicar** por que guardrails determinísticos vêm antes da política de custo;
- **identificar** as regras que forçam `MATCH`, `NONMATCH` ou revisão humana/LLM;
- **relacionar** cada regra aos cenários sintéticos narrativos do dataset;
- **avaliar** quando um par segue sem guardrail para a triagem por custo esperado.

**Intuição.** Guardrails são **regras determinísticas de segurança** aplicadas
**ANTES** da política de custo. Eles não tentam otimizar o custo médio; tentam evitar
erros catastróficos e casos estruturalmente perigosos: datas impossíveis, notas
extremamente baixas/altas e homonímia forte. Só depois desse filtro a triagem decide
por custo esperado entre `MATCH`, `NONMATCH` e `LLM_REVIEW`.""",
    ),
    (
        "md",
        """\
**O que vamos fazer a seguir.** Chamaremos a função real
`apply_guardrails(df)`. Ela devolve duas `Series` alinhadas ao `df`: o guardrail
acionado (`ALWAYS_MATCH`, `ALWAYS_NONMATCH`, `FORCE_REVIEW` ou `<NA>`) e o motivo
determinístico (`reason`). A função **não muta** o DataFrame; aqui adicionamos as
colunas explicitamente para continuar a narrativa do notebook.""",
    ),
    (
        "code",
        """\
from gzcmd_record_linkage.guardrails import apply_guardrails

gout = apply_guardrails(df)
df["guardrail"] = gout.guardrail
df["guardrail_reason"] = gout.reason

df["guardrail"].value_counts(dropna=False)""",
    ),
    (
        "md",
        """\
### 10.1 Quatro regras, três ações possíveis

As regras implementadas hoje são:

1. **`temporal_filter` → `ALWAYS_NONMATCH`**: se o óbito da referência ocorre muito
   antes do diagnóstico do candidato, a vinculação é temporalmente impossível. No
   dataset, isso aparece no cenário `obito_antes_diag`.
2. **`nota_final_low` → `ALWAYS_NONMATCH`**: notas muito baixas são âncoras negativas;
   o cenário `nonmatch_obvio` foi desenhado para cair aqui.
3. **`homonimia_risk` → `FORCE_REVIEW`**: nome muito parecido com distância grande de
   ano de nascimento indica risco de homônimo; o cenário `homonimo` ilustra esse caso.
4. **`nota_final_high` → `ALWAYS_MATCH`**: notas extremamente altas são âncoras
   positivas; o cenário `match_obvio` representa esse caso.

> **R-11 (config × código).** A configuração menciona uma regra suave
> `grey_mother_missing`, mas ela **não está implementada no código atual**. Portanto,
> o cenário `mae_ausente` não deve ser interpretado como capturado por guardrail;
> ele segue para a etapa de triagem.""",
    ),
    (
        "code",
        """\
cols_guardrail = ["nota_final", "band", "guardrail", "guardrail_reason"]
exemplos_guardrail = (
    df.loc[df["guardrail"].notna(), cols_guardrail]
    .sort_values(["guardrail", "nota_final"], ascending=[True, False])
    .groupby("guardrail", dropna=True, group_keys=False)
    .head(3)
)
exemplos_guardrail""",
    ),
    (
        "md",
        """\
### 10.2 Mapa dos cenários narrativos

Os cenários sintéticos foram criados para deixar cada comportamento auditável:

| Cenário | Guardrail esperado | Motivo esperado |
|---------|--------------------|-----------------|
| `match_obvio` | `ALWAYS_MATCH` | `nota_final_high` |
| `nonmatch_obvio` | `ALWAYS_NONMATCH` | `nota_final_low` |
| `homonimo` | `FORCE_REVIEW` | `homonimia_risk` |
| `obito_antes_diag` | `ALWAYS_NONMATCH` | `temporal_filter` |
| `mae_ausente` | nenhum guardrail | regra suave da config não implementada |

Agora voltamos ao par herói (`zona_cinzenta`) para verificar se ele foi capturado por
alguma dessas regras determinísticas.""",
    ),
    (
        "code",
        """\
card_heroi(
    df,
    hero_idx,
    ["nota_final", "TARGET", "band", "p_cal", "guardrail", "guardrail_reason"],
)""",
    ),
    (
        "md",
        """\
**Recap da seção.** Aplicamos os guardrails reais do pacote e vimos que eles funcionam
como uma camada de segurança antes da decisão econômica: forçam match quando a nota é
altíssima, forçam não-match quando a nota é muito baixa ou há impossibilidade temporal,
e mandam homônimos perigosos para revisão.

O herói `zona_cinzenta` fica com `guardrail = <NA>`: ele **não** foi capturado por
nenhuma regra determinística e, portanto, segue para a etapa de triagem. **A seguir:**
a política de custo esperado decidirá a ação final usando `p_cal`, banda e guardrails.""",
    ),
]


# ---------------------------------------------------------------------------
# Montagem do notebook
# ---------------------------------------------------------------------------
# A ordem das fases reflete a construção incremental do plano (Wave 2).

FASE_2_6: list[tuple[str, str]] = [
    (
        "md",
        """
## 11. Política de decisão (triagem)

### Objetivos de aprendizagem

Ao final desta seção, você será capaz de:

- **explicar** por que a decisão final deve minimizar **custo esperado**, e não apenas
  comparar `nota_final` ou `p_cal` com um limiar fixo;
- **calcular** as perdas esperadas de `MATCH`, `NONMATCH` e `LLM_REVIEW` para um par;
- **comparar** os modos `vigilancia` e `confirmacao`, reconhecendo a assimetria de custos
  entre falso positivo e falso negativo;
- **interpretar** quando vale a pena revisar um caso ambíguo com LLM.

### Intuição

A política de decisão não pergunta “a probabilidade passou de 0,5?”. Ela pergunta:
**qual ação tem menor custo esperado para este modo de operação?**

Se errar custa muito, a política pode escolher uma ação conservadora. Se revisar custa menos
que errar, ela pode rotear o par para `LLM_REVIEW`. Isso é especialmente importante na zona
cinzenta: pares próximos aos limiares não são ruins; eles são justamente os pares em que
uma revisão adicional pode ter valor econômico.
""",
    ),
    (
        "md",
        r"""
### T2.6.2 — Custo esperado e valor esperado da revisão

Para cada par, usamos $p$ como a probabilidade calibrada de match (`p_cal`). A política compara
três perdas esperadas (usamos $\ell$ para "loss"/perda):

- $\ell_{\text{match}} = (1-p)\,c_{fp}$: custo esperado de aceitar como `MATCH`. Só pagamos esse
  custo quando o par era, na verdade, não-match; por isso aparece $(1-p)$.
- $\ell_{\text{non}} = p\,c_{fn}$: custo esperado de rejeitar como `NONMATCH`. Só pagamos esse
  custo quando o par era, na verdade, match; por isso aparece $p$.
- $\ell_{\text{llm}} = c_{llm} + (1-p)\,e_{fp}\,c_{fp} + p\,e_{fn}\,c_{fn}$: custo esperado de
  pedir revisão. Ele soma o custo direto da revisão ($c_{llm}$) com o erro residual esperado da
  LLM: $e_{fp}$ é a taxa residual de falso positivo após revisão e $e_{fn}$ é a taxa residual de
  falso negativo após revisão.

A melhor decisão automática sem revisão é:

- $\ell_{\min} = \min(\ell_{\text{match}}, \ell_{\text{non}})$ (a *base loss*);
- a *base choice* é $\operatorname{argmin}(\ell_{\text{match}}, \ell_{\text{non}})$, ou seja,
  `MATCH` ou `NONMATCH`.

O valor esperado da revisão (*expected value of review*) é:

- $\text{evr} = \ell_{\min} - \ell_{\text{llm}}$.

Se $evr > 0$ e ainda houver orçamento de revisão, revisar reduz o custo esperado; a ação final
pode ser `LLM_REVIEW`. Caso contrário, a ação fica com `base_choice`. Portanto,
`action ∈ {MATCH, NONMATCH, LLM_REVIEW}`.

Os dois modos mudam os custos e os limites de automação:

- **`vigilancia`**: $c_{fp}=10$, $c_{fn}=50$, `min_auto_match=0.85`,
  `max_auto_nonmatch=0.15`, orçamento 2000. Prioriza **recall**: perder um match custa caro.
- **`confirmacao`**: $c_{fp}=100$, $c_{fn}=20$, `min_auto_match=0.95`,
  `max_auto_nonmatch=0.10`, orçamento 1000. Prioriza **precisão**: confirmar um falso match custa caro.
""",
    ),
    (
        "md",
        """
### T2.6.1 — Aplicar a política nos dois modos

**Objetivo de aprendizagem.** Executar a triagem real do pacote, mantendo os dois modos
separados.

**Intuição.** O orçamento de revisão é estado interno do motor de política. Por isso usamos um
engine por modo e passamos uma cópia defensiva do `df`.

**Ação.** Construiremos `engine_vig` e `engine_conf` a partir da mesma configuração já carregada
em `cfg`, chamaremos `.triage(df.copy())` e inspecionaremos as colunas decisórias.

**Recap.** Depois desta célula teremos duas saídas alinhadas linha a linha: `out_vig` e
`out_conf`.
""",
    ),
    (
        "code",
        """from gzcmd_record_linkage.runner import build_engine_from_config

engine_vig = build_engine_from_config(cfg, mode="vigilancia")
out_vig = engine_vig.triage(df.copy())

engine_conf = build_engine_from_config(cfg, mode="confirmacao")
out_conf = engine_conf.triage(df.copy())

out_vig[["nota_final", "band", "p_cal", "base_choice", "evr", "action"]].head()""",
    ),
    (
        "md",
        """
### Como ler a saída da triagem

A política acrescenta colunas que explicam a decisão:

- `base_choice`: melhor ação automática antes da revisão (`MATCH` ou `NONMATCH`);
- `base_loss`: custo esperado dessa melhor ação automática;
- `loss_llm`: custo esperado de pedir revisão;
- `evr`: valor esperado da revisão (`base_loss - loss_llm`);
- `action`: ação final (`MATCH`, `NONMATCH` ou `LLM_REVIEW`);
- `review_requested`: booleano que indica se o par foi roteado para revisão.

**Recap.** `action` é a decisão operacional; as demais colunas explicam por que ela foi escolhida.
""",
    ),
    (
        "md",
        """
### T2.6.2 — Comparar a distribuição de ações

**Objetivo de aprendizagem.** Quantificar como a assimetria de custos muda o volume de decisões.

**Intuição.** Se `confirmacao` é mais conservador para falso positivo, esperamos menos `MATCH`
automáticos do que em `vigilancia`.

**Ação.** Contaremos `MATCH`, `NONMATCH` e `LLM_REVIEW` em cada modo e colocaremos tudo em uma
tabela única.

**Recap.** A tabela mostra a política como operação: quantos pares vão para cada rota.
""",
    ),
    (
        "code",
        """ordem_acoes = ["MATCH", "NONMATCH", "LLM_REVIEW"]
dist_acoes = (
    pd.DataFrame(
        {
            "vigilancia": out_vig["action"].value_counts(),
            "confirmacao": out_conf["action"].value_counts(),
        }
    )
    .reindex(ordem_acoes)
    .fillna(0)
    .astype(int)
)

dist_acoes""",
    ),
    (
        "md",
        """
### T2.6.3 — Visualizar a distribuição por modo

**Objetivo de aprendizagem.** Ler rapidamente a diferença operacional entre os modos.

**Intuição.** Barras agrupadas facilitam comparar a mesma ação sob custos diferentes.

**Ação.** Desenharemos um gráfico de barras com as contagens de `MATCH`, `NONMATCH` e
`LLM_REVIEW` para `vigilancia` e `confirmacao`.

**Recap.** A figura deve deixar visível que mudar custos e limiares muda a política de decisão.
""",
    ),
    (
        "code",
        """x = np.arange(len(ordem_acoes))
largura = 0.35

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(x - largura / 2, dist_acoes["vigilancia"], largura, label="vigilância", color="#4C72B0")
ax.bar(x + largura / 2, dist_acoes["confirmacao"], largura, label="confirmação", color="#DD8452")

ax.set_xticks(x)
ax.set_xticklabels(ordem_acoes)
ax.set_ylabel("número de pares")
ax.set_xlabel("ação final")
ax.set_title("Distribuição das ações de triagem por modo")
ax.legend(title="modo")
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()""",
    ),
    (
        "md",
        """
### Pares que mudam de decisão entre modos

**Objetivo de aprendizagem.** Identificar casos sensíveis à política, não apenas ao escore.

**Intuição.** Perto dos limiares, pequenas mudanças de custo esperado podem trocar a ação final.
Esses são pares de zona cinzenta: a evidência estatística é parecida, mas o contexto operacional
pede outra decisão.

**Ação.** Marcaremos as linhas em que `vigilancia` e `confirmacao` escolhem ações diferentes e
exibiremos uma amostra lado a lado.

**Recap.** Mudança de decisão não é inconsistência; é a política respondendo a objetivos diferentes.
""",
    ),
    (
        "code",
        """mudou = out_vig["action"] != out_conf["action"]
pares_mudam = pd.DataFrame(
    {
        "nota_final": df.loc[mudou, "nota_final"],
        "band": df.loc[mudou, "band"],
        "p_cal": df.loc[mudou, "p_cal"],
        "acao_vigilancia": out_vig.loc[mudou, "action"],
        "acao_confirmacao": out_conf.loc[mudou, "action"],
        "evr_vigilancia": out_vig.loc[mudou, "evr"],
        "evr_confirmacao": out_conf.loc[mudou, "evr"],
    }
)

pares_mudam.head(8)""",
    ),
    (
        "md",
        """
### Par herói: zona cinzenta sob duas políticas

**Objetivo de aprendizagem.** Conectar a decisão agregada ao caso narrativo acompanhado desde o
início do notebook.

**Intuição.** O par herói (`zona_cinzenta`) é o melhor exemplo de por que triagem não é limiar
fixo: ele pode ser tratado de forma diferente quando a organização prioriza recall ou precisão.

**Ação.** Reutilizaremos `card_heroi(df, idx, colunas)` e compararemos a ação do mesmo par nos
dois modos.

**Recap.** A mesma evidência pode gerar ações diferentes quando o custo de errar muda.
""",
    ),
    (
        "code",
        """cartao_heroi = card_heroi(
    df,
    hero_idx,
    ["nota_final", "TARGET", "band", "p_cal", "guardrail", "guardrail_reason"],
)
acoes_heroi = pd.DataFrame(
    {
        "modo": ["vigilancia", "confirmacao"],
        "action": [out_vig.loc[hero_idx, "action"], out_conf.loc[hero_idx, "action"]],
        "base_choice": [out_vig.loc[hero_idx, "base_choice"], out_conf.loc[hero_idx, "base_choice"]],
        "evr": [out_vig.loc[hero_idx, "evr"], out_conf.loc[hero_idx, "evr"]],
        "review_requested": [
            out_vig.loc[hero_idx, "review_requested"],
            out_conf.loc[hero_idx, "review_requested"],
        ],
    }
)

display(cartao_heroi)
acoes_heroi""",
    ),
    (
        "md",
        """
### Recap da seção e o que vem a seguir

A triagem fecha a Wave 2: agora o notebook tem bandas, calibração, guardrails e política de
decisão por custo esperado. Vimos que:

- `MATCH` e `NONMATCH` são comparados por perda esperada;
- `LLM_REVIEW` entra quando o valor esperado da revisão compensa o custo e há orçamento;
- `vigilancia` prioriza recall, enquanto `confirmacao` prioriza precisão;
- pares de zona cinzenta podem mudar de ação entre modos sem que o modelo esteja “errado”.

Na Wave 3, o próximo passo é reconciliar esta leitura didática com `run_v3` e medir desempenho
em held-out: métricas finais, auditoria de decisões e comparação ponta a ponta.
""",
    ),
]


FASE_3_1 = [
    (
        "md",
        """## 12. Reconciliação com `run_v3` (rota A, in-sample)

**Objetivos de aprendizagem.** Ao final desta seção, você deve conseguir **comparar** a rota manual com a função de produção `run_v3`, **verificar** igualdade coluna a coluna, **interpretar** o `RunSummary` e **distinguir** quando uma divergência é erro de implementação versus escolha metodológica.

**Intuição.** O passo a passo manual que construímos nas Fases 7–11 deve reproduzir **exatamente** o que a função de produção `run_v3` faz quando usamos os mesmos parâmetros: ajuste Platt determinístico, configuração igual aos defaults da função e modo `vigilancia`.

Há duas rotas conceituais:

- **Rota A (in-sample):** ajusta a calibração no mesmo conjunto usado na triagem manual. Esta rota deve reconciliar exatamente com `run_v3`.
- **Rota B (held-out, Fase 2.4):** separa treino/teste para medir generalização. Ela difere **por desenho**, porque o split muda os dados usados no ajuste e, portanto, os números esperados.

**Ação.** Vamos executar `run_v3` com `p_cal='fit_platt'` e comparar com `out_vig`, a triagem manual da rota A construída na Fase 11.

**Recap.** Se os parâmetros e o conjunto de ajuste são os mesmos, a reconciliação deve ser exata; se a rota é held-out, a diferença é esperada e correta.""",
    ),
    (
        "md",
        """Vamos rodar a função de produção `run_v3` com `p_cal='fit_platt'` e `mode='vigilancia'`. Em seguida, compararemos três saídas críticas com a rota A manual (`out_vig`):

1. `band`: faixa atribuída a partir de `nota_final`;
2. `p_cal`: probabilidade calibrada pelo Platt global;
3. `action`: decisão final da política de vigilância.

A comparação é coluna a coluna. Para `p_cal`, usamos a diferença máxima absoluta para capturar qualquer desvio numérico.""",
    ),
    (
        "code",
        """from gzcmd_record_linkage.runner import run_v3

out_run, summary = run_v3(
    input_csv=CSV_PATH,
    config_path=CONFIG_PATH,
    mode="vigilancia",
    macd_enabled=True,
    p_cal="fit_platt",
)

# Comparação coluna a coluna com a rota A manual (out_vig, da Fase 11)
import numpy as np

band_identico = bool((out_run["band"].to_numpy() == out_vig["band"].to_numpy()).all())
pcal_maxdiff = float(
    np.max(np.abs(out_run["p_cal"].to_numpy(float) - out_vig["p_cal"].to_numpy(float)))
)
acao_identica = bool((out_run["action"].to_numpy() == out_vig["action"].to_numpy()).all())

print(f"band idêntico:           {band_identico}")
print(f"p_cal diferença máxima:  {pcal_maxdiff:.2e}")
print(f"action idêntico:         {acao_identica}")""",
    ),
    (
        "md",
        """O objeto `RunSummary` resume a execução de produção. Ele registra quantas linhas foram processadas (`rows`), a distribuição das ações (`actions`), quais guardrails foram ativados (`guardrails`), quantos casos pediram revisão (`review_requested`) e como `p_cal` foi produzido (`p_cal_method` e `p_cal_params`).

Essa visão é útil para auditoria: além de saber que as colunas reconciliaram, conseguimos explicar o volume de decisões e os parâmetros efetivos da calibração.""",
    ),
    (
        "code",
        """resumo = {
    "linhas": summary.rows,
    "ações": summary.actions,
    "guardrails": summary.guardrails,
    "review_requested": summary.review_requested,
    "p_cal_method": summary.p_cal_method,
    "p_cal_params": summary.p_cal_params,
}
resumo""",
    ),
    (
        "md",
        """A reconciliação é exata porque o Platt global usado aqui é determinístico e porque os parâmetros da configuração são iguais aos defaults usados por `run_v3`. Assim, rota manual e função de produção leem os mesmos dados, aplicam a mesma calibração, atribuem as mesmas bandas, executam os mesmos guardrails e chegam à mesma ação.

Já a rota B, held-out, difere de propósito: ela faz split treino/teste para estimar desempenho fora da amostra. Como o ajuste é feito em outro subconjunto, os valores de `p_cal` e as métricas podem mudar. Isso não é bug; é exatamente o que queremos para medir generalização.

**R-13.** Para modelos como XGBoost ou Random Forest, a reconciliação deve ser tratada como qualitativa. Mesmo com seeds, pode haver não-determinismo entre threads, bibliotecas e execuções; portanto, verificamos propriedades agregadas e domínio das ações, não igualdade bit a bit.""",
    ),
    (
        "md",
        """**Recap da seção.** Nesta seção, executamos `run_v3` na rota A, com Platt ajustado in-sample, e comparamos a saída de produção com a triagem manual. A expectativa correta é igualdade exata para `band`, diferença numérica zero ou desprezível para `p_cal` e igualdade exata para `action`. Também vimos como ler o `RunSummary` e por que rotas held-out ou modelos ML não determinísticos pedem critérios de reconciliação diferentes.""",
    ),
]


FASE_3_2 = [
    (
        "md",
        """## 13. Avaliação held-out (rota B)

**Objetivos de aprendizagem.** Ao final desta seção, você deve conseguir **avaliar** métricas em teste held-out, **comparar** modos de triagem sob múltiplas sementes, **diagnosticar** vazamento por split ingênuo, **interpretar** curvas PR/ROC e **relacionar** limiares de política a custo esperado.

**Intuição.** Na seção 9 discutimos o risco R-10: medir desempenho no mesmo conjunto usado para ajustar a calibração produz uma leitura otimista. A rota B separa treino e teste por grupo (`COMPREC`) para que o Platt seja ajustado no treino e avaliado em pares não vistos no teste.

Também vamos acompanhar **cobertura automática**: a fração de pares que saem como decisão automática (`MATCH` ou `NONMATCH`) em vez de `LLM_REVIEW`. Cobertura alta reduz custo operacional, mas pode pressionar precisão ou recall dependendo do modo.

**Ação.** Vamos medir desempenho held-out em cinco sementes, resumir média±desvio, comparar splits com e sem vazamento, desenhar curvas de operação e traduzir FP/FN em custo.

**Recap.** A pergunta desta seção não é “o modelo acertou o treino?”, e sim “quão estável é a política quando avaliamos fora da amostra e sob diferentes cortes dos dados?”""",
    ),
    (
        "md",
        """### 13.1 Métricas held-out em múltiplas sementes

Uma única semente pode ser enganosa: ela mostra apenas uma realização do split treino/teste. Se a prevalência, a dificuldade dos pares ou os grupos `COMPREC` mudam entre sementes, precisão, recall, Fβ e cobertura também podem variar.

O próximo código chama `evaluate_v3_dataframe` na rota B. A função ajusta o Platt no treino, prediz no teste e calcula internamente bandas, guardrails, `p_cal`, métricas automáticas e metadados. Usamos os dois modos de política e agrupamos por modo para exibir média e desvio-padrão das métricas principais.""",
    ),
    (
        "code",
        """from gzcmd_record_linkage.eval import evaluate_v3_dataframe

SEEDS_32 = [42, 123, 456, 789, 2024]
metric_cols_32 = ["auto_precision", "auto_recall", "auto_fbeta", "auto_coverage"]

res = evaluate_v3_dataframe(
    df,
    cfg=cfg,
    modes=["vigilancia", "confirmacao"],
    split_by="comprec",
    seeds=SEEDS_32,
    test_size=0.3,
    group_stratify=True,
    calibration="platt",
    macd_enabled=True,
)

resumo_multi_seed = res.groupby("mode")[metric_cols_32].agg(["mean", "std"])
resumo_multi_seed""",
    ),
    (
        "md",
        """### 13.2 Barras com incerteza entre sementes

A tabela mostra os números exatos, mas um gráfico ajuda a enxergar a estabilidade relativa. As barras abaixo mostram a média por modo; as hastes de erro mostram o desvio-padrão entre as cinco sementes.

Se uma haste é grande, a conclusão depende mais do split. Se as hastes são pequenas, o comportamento do modo é mais estável para este conjunto sintético.""",
    ),
    (
        "code",
        """fig, ax = plt.subplots(figsize=(9, 5))

modes_32 = resumo_multi_seed.index.to_list()
x = np.arange(len(modes_32))
width = 0.22
metric_labels_32 = {
    "auto_precision": "Precisão automática",
    "auto_recall": "Recall automático",
    "auto_fbeta": "Fβ automático",
}

for i, metric in enumerate(metric_labels_32):
    means = resumo_multi_seed[(metric, "mean")].to_numpy(float)
    stds = resumo_multi_seed[(metric, "std")].fillna(0).to_numpy(float)
    ax.bar(
        x + (i - 1) * width,
        means,
        width,
        yerr=stds,
        capsize=4,
        label=metric_labels_32[metric],
    )

ax.set_xticks(x)
ax.set_xticklabels(modes_32)
ax.set_xlabel("Modo de operação")
ax.set_ylim(0, 1.05)
ax.set_ylabel("Métrica no teste held-out")
ax.set_title("Média ± desvio-padrão por modo (5 sementes, split por COMPREC)")
ax.legend(loc="lower right")
ax.grid(axis="y", alpha=0.25)
plt.show()""",
    ),
    (
        "md",
        """### 13.3 Demonstração de vazamento: `row` versus splits por grupo

Agora isolamos o modo `vigilancia` para manter o tempo de execução baixo e mudamos apenas a estratégia de split. O split `row` sorteia linhas independentemente; quando o **mesmo registro** (`COMPREC` ou `REFREC`) aparece em vários pares, ele pode cair em treino **e** em teste, gerando *vazamento por registro compartilhado* e métricas otimistas. Splits por `COMPREC`/`REFREC` mantêm grupos inteiros de um único lado do corte e evitam isso.

**Expectativa honesta para ESTE dataset.** A magnitude do vazamento depende de quão repetidos são os registros. No nosso gerador sintético a maioria dos `COMPREC`/`REFREC` é **única** (grupos de tamanho 1), então os três splits tendem a produzir métricas **quase idênticas** — o efeito aqui é pequeno por construção. A célula a seguir primeiro mede a multiplicidade dos grupos e depois compara a média do Fβ automático, recall e cobertura por tipo de split, para que o leitor julgue o efeito pelos números, não pela retórica.""",
    ),
    (
        "code",
        """# Multiplicidade dos grupos: fração de linhas cujo COMPREC/REFREC se repete.
_comprec = df["COMPREC"] if "COMPREC" in df.columns else df.filter(like="COMPREC").iloc[:, 0]
_refrec = df["REFREC"] if "REFREC" in df.columns else df.filter(like="REFREC").iloc[:, 0]
frac_comprec_repetido = (_comprec.map(_comprec.value_counts()) > 1).mean()
frac_refrec_repetido = (_refrec.map(_refrec.value_counts()) > 1).mean()
print(f"Fração de linhas com COMPREC repetido: {frac_comprec_repetido:.1%}")
print(f"Fração de linhas com REFREC repetido:  {frac_refrec_repetido:.1%}")

leakage_rows = []

for split_by in ["row", "comprec", "refrec"]:
    res_split = evaluate_v3_dataframe(
        df,
        cfg=cfg,
        modes=["vigilancia"],
        split_by=split_by,
        seeds=SEEDS_32,
        test_size=0.3,
        group_stratify=True,
        calibration="platt",
        macd_enabled=True,
    )
    leakage_rows.append(
        {
            "split_by": split_by,
            "auto_fbeta_mean": res_split["auto_fbeta"].mean(),
            "auto_recall_mean": res_split["auto_recall"].mean(),
            "auto_coverage_mean": res_split["auto_coverage"].mean(),
        }
    )

leakage_summary = pd.DataFrame(leakage_rows).set_index("split_by")
leakage_summary""",
    ),
    (
        "md",
        """**Lendo a tabela com honestidade.** Como antecipado, neste dataset sintético as três
estratégias entregam Fβ/recall/cobertura **praticamente iguais** (diferenças na casa do
milésimo) — coerente com a alta fração de grupos de tamanho 1 medida acima. Ou seja: aqui
o split `row` **não** infla materialmente as métricas, porque quase não há registro
compartilhado para vazar.

Isso **não** enfraquece a regra metodológica — apenas a contextualiza. Em dados reais de
*record linkage*, um mesmo indivíduo costuma participar de **muitos** pares candidatos
(blocking gera dezenas de comparações por registro). Nesse cenário o split `row` mistura o
mesmo registro entre treino e teste e **infla** o desempenho aparente; por isso a avaliação
correta usa split **group-aware** (`COMPREC`/`REFREC`), que é o default adotado em todo o
restante desta seção. A demonstração serve para você reconhecer o mecanismo e exigir o split
certo quando os grupos forem grandes.""",
    ),
    (
        "md",
        """### 13.4 Curvas PR e ROC da rota B

As métricas anteriores dependem dos limiares de política. Para ver todos os pontos de operação possíveis, reutilizamos `y_test` e `p_cal_test` já produzidos na seção 9.3 pela rota B held-out. Não recalculamos o split: aqui apenas varremos o limiar sobre a probabilidade calibrada.

A curva Precision-Recall é especialmente útil quando a prevalência de `MATCH` é baixa; a ROC mostra a troca entre taxa de falso positivo e verdadeiro positivo em outra escala.""",
    ),
    (
        "code",
        """from sklearn.metrics import auc, precision_recall_curve, roc_curve

y_test_arr = np.asarray(y_test, dtype=int)
p_cal_test_arr = np.asarray(p_cal_test, dtype=float)

precision_curve, recall_curve, _ = precision_recall_curve(y_test_arr, p_cal_test_arr)
fpr_curve, tpr_curve, _ = roc_curve(y_test_arr, p_cal_test_arr)

pr_auc = auc(recall_curve, precision_curve)
roc_auc = auc(fpr_curve, tpr_curve)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].plot(recall_curve, precision_curve, label=f"PR-AUC = {pr_auc:.3f}")
axes[0].set_xlabel("Recall")
axes[0].set_ylabel("Precisão")
axes[0].set_title("Curva Precision-Recall (rota B)")
axes[0].set_xlim(0, 1.02)
axes[0].set_ylim(0, 1.02)
axes[0].grid(alpha=0.25)
axes[0].legend(loc="lower left")

axes[1].plot(fpr_curve, tpr_curve, label=f"ROC-AUC = {roc_auc:.3f}")
axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray", label="aleatório")
axes[1].set_xlabel("Taxa de falso positivo")
axes[1].set_ylabel("Taxa de verdadeiro positivo")
axes[1].set_title("Curva ROC (rota B)")
axes[1].set_xlim(0, 1.02)
axes[1].set_ylim(0, 1.02)
axes[1].grid(alpha=0.25)
axes[1].legend(loc="lower right")

plt.tight_layout()
plt.show()

{"PR-AUC": pr_auc, "ROC-AUC": roc_auc}""",
    ),
    (
        "md",
        """### 13.5 Custo esperado versus limiar de MATCH

Na seção 11 discutimos como modos diferentes penalizam FP e FN de formas diferentes. Agora traduzimos isso em custo esperado: para cada limiar τ, predizemos `MATCH` quando `p_cal_test >= τ` e somamos `FP*c_fp + FN*c_fn`.

As linhas verticais mostram os limiares de auto-MATCH da política (`min_auto_match`). O ponto de menor custo da curva é um diagnóstico: ele não substitui a política, mas indica se o limiar operacional está conservador ou agressivo para a matriz de custos escolhida.""",
    ),
    (
        "code",
        """thresholds = np.linspace(0, 1, 101)
mode_costs_32 = {
    "vigilancia": {"false_positive": 10, "false_negative": 50, "min_auto_match": 0.85},
    "confirmacao": {"false_positive": 100, "false_negative": 20, "min_auto_match": 0.95},
}

cost_rows = []
for mode, costs in mode_costs_32.items():
    for tau in thresholds:
        pred_match = p_cal_test_arr >= tau
        fp = int(((pred_match == 1) & (y_test_arr == 0)).sum())
        fn = int(((pred_match == 0) & (y_test_arr == 1)).sum())
        total_cost = fp * costs["false_positive"] + fn * costs["false_negative"]
        cost_rows.append({"mode": mode, "tau": tau, "fp": fp, "fn": fn, "total_cost": total_cost})

cost_df = pd.DataFrame(cost_rows)
best_cost = cost_df.loc[cost_df.groupby("mode")["total_cost"].idxmin()].copy()

fig, ax = plt.subplots(figsize=(9, 5))
for mode, group in cost_df.groupby("mode"):
    ax.plot(group["tau"], group["total_cost"], label=f"custo — {mode}")
    policy_tau = mode_costs_32[mode]["min_auto_match"]
    ax.axvline(policy_tau, linestyle="--", alpha=0.65, label=f"limiar política — {mode}: {policy_tau:.2f}")
    best_row = best_cost.loc[best_cost["mode"] == mode].iloc[0]
    ax.scatter(best_row["tau"], best_row["total_cost"], s=55)
    ax.annotate(
        f"τ*={best_row['tau']:.2f}",
        xy=(best_row["tau"], best_row["total_cost"]),
        xytext=(5, 8),
        textcoords="offset points",
    )

ax.set_xlabel("Limiar τ para predizer MATCH")
ax.set_ylabel("Custo total no teste")
ax.set_title("Custo esperado por limiar e modo")
ax.grid(alpha=0.25)
ax.legend(loc="best")
plt.show()

best_cost[["mode", "tau", "fp", "fn", "total_cost"]]""",
    ),
    (
        "md",
        """### 13.6 Interpretação e próximos passos

O trade-off central é **precisão × recall × cobertura**. Em `vigilancia`, o custo de falso negativo é maior; por isso, esperamos uma política mais tolerante a revisar ou aceitar candidatos para proteger recall. Em `confirmacao`, falso positivo custa mais; a política tende a exigir evidência mais forte antes de auto-confirmar `MATCH`, o que pode reduzir cobertura ou recall automático.

A prevalência também importa. Quando a taxa-base de `MATCH` é baixa, pequenas mudanças em FP afetam muito a precisão; quando há poucos positivos, o recall fica sensível a poucos FN. Por isso a calibração Platt e o ponto de operação devem ser lidos junto com prevalência, matriz de custos e cobertura automática — não como métricas isoladas.

**Recap.** Avaliamos a rota B em múltiplas sementes, vimos por que split por linha pode inflar métricas, interpretamos PR/ROC e conectamos limiares a custo esperado. **O que vem a seguir:** na Fase 3.3, entraremos no stub de LLM para estudar como a revisão assistida pode atuar sobre os casos `LLM_REVIEW` sem contaminar a avaliação held-out.""",
    ),
]


FASE_3_3 = [
    (
        "md",
        """## 14. Revisão LLM (stub determinístico)

**Objetivos de aprendizagem.** Ao final desta seção, você deve conseguir **explicar** o papel da revisão clerical/LLM na zona cinzenta, **descrever** o protocolo `dual_agent_plus_arbiter`, **simular** a revisão de forma determinística e offline, e **medir** seu efeito sobre as decisões finais — sem confundir simulação com um LLM real.

**Intuição.** A política de triagem não decide tudo sozinha: os pares mais ambíguos saem como `LLM_REVIEW` para inspeção assistida. Numa operação real, um modelo de linguagem (ou um par de revisores humanos) examinaria o dossiê de cada par e devolveria `MATCH`/`NONMATCH`. Aqui **não chamamos nenhuma API** — isso quebraria a reprodutibilidade e exigiria rede na apresentação (R-05).

**O protocolo `dual_agent_plus_arbiter` (conceitual).** A config descreve um protocolo de consenso: dois agentes (A e B) revisam o mesmo dossiê de forma independente; se concordam, a decisão é aceita; se discordam, um terceiro agente **árbitro** desempata. A ideia espelha o esquema "2 revisores + consenso" e reduz a variância de um único revisor.

> **Honestidade científica (R-05 / CA-G6).** O que usamos abaixo é um **stub de simulação**, não um LLM. O stub "enxerga" o rótulo verdadeiro (`TARGET`) e devolve a decisão correta na maioria das vezes, **errando com as taxas por banda da própria config** (`e_fp`/`e_fn`). Serve para demonstrar, de forma determinística, o *efeito* da revisão sobre as métricas — jamais para afirmar desempenho de um LLM real.""",
    ),
    (
        "md",
        """### 14.1 Aplicando o stub aos casos `LLM_REVIEW`

Tomamos a triagem do modo `vigilancia` (rota A, seção 11), selecionamos os pares roteados a `LLM_REVIEW` e aplicamos `llm_review_stub`. As taxas de erro vêm de `cfg.llm_review.error_rates_by_band` — exatamente as que o motor usa para estimar o custo da revisão (`loss_llm`). A semente fixa garante que a simulação é reprodutível.""",
    ),
    (
        "code",
        """from gzcmd_record_linkage.metrics import confusion_counts, metrics_dict

from nb_helpers import llm_review_stub

error_rates = cfg.llm_review.error_rates_by_band
modo_revisao = "vigilancia"
out_modo = out_vig  # triagem rota A (in-sample), modo vigilancia

mask_review = out_modo["action"] == "LLM_REVIEW"
df_review = out_modo.loc[mask_review, ["band", "TARGET"]].copy()
decisoes_revisao = llm_review_stub(
    df_review, seed=SEED, error_rates_by_band=error_rates
)

acerto_stub = (
    decisoes_revisao.to_numpy() == out_modo.loc[mask_review, "TARGET"].map(
        {1: "MATCH", 0: "NONMATCH"}
    ).to_numpy()
).mean()

print(f"Pares roteados a LLM_REVIEW (modo {modo_revisao}): {int(mask_review.sum())}")
print("Distribuição das decisões simuladas:")
print(decisoes_revisao.value_counts())
print(f"Acurácia do revisor simulado (vs TARGET): {acerto_stub:.1%}")""",
    ),
    (
        "md",
        """### 14.2 Métricas finais incluindo a revisão simulada

Agora combinamos as decisões: pares automáticos mantêm sua ação (`MATCH`/`NONMATCH`); pares `LLM_REVIEW` recebem a decisão do stub. Com isso **todos** os pares ficam resolvidos, e podemos calcular precisão, recall e F1 finais sobre o conjunto inteiro (comparando com `TARGET`). Comparamos com a fotografia automática anterior, em que os `LLM_REVIEW` ainda estavam pendentes.""",
    ),
    (
        "code",
        """final_action = out_modo["action"].astype("object").copy()
final_action.loc[mask_review] = decisoes_revisao.to_numpy()

y_true = out_modo["TARGET"].astype(int).to_numpy()
y_pred_final = (final_action == "MATCH").astype(int).to_numpy()

counts_final = confusion_counts(y_true, y_pred_final)
metrics_final = metrics_dict(counts_final, beta=1.0)

cobertura_auto = float((out_modo["action"] != "LLM_REVIEW").mean())

tabela_final = pd.DataFrame(
    {
        "métrica": ["precisão", "recall", "F1", "cobertura automática (antes)"],
        "valor": [
            metrics_final["precision"],
            metrics_final["recall"],
            metrics_final["f1"],
            cobertura_auto,
        ],
    }
)
tabela_final.round(4)""",
    ),
    (
        "md",
        """### 14.3 O herói na revisão

Nosso par-fio-condutor (`zona_cinzenta`) é, por construção, um caso ambíguo. Vamos ver se a triagem o enviou para `LLM_REVIEW` e, em caso afirmativo, qual foi a decisão final simulada.""",
    ),
    (
        "code",
        """hero_action_vig = out_vig.loc[hero_idx, "action"]
if hero_action_vig == "LLM_REVIEW":
    hero_final = decisoes_revisao.loc[hero_idx]
    hero_msg = f"herói roteado a LLM_REVIEW → decisão simulada: {hero_final}"
else:
    hero_final = hero_action_vig
    hero_msg = f"herói decidido automaticamente: {hero_final}"

print(hero_msg)
card_heroi(out_vig, hero_idx, ["nota_final", "band", "p_cal", "action", "TARGET"])""",
    ),
    (
        "md",
        """**Recap.** Simulamos a etapa de revisão de forma determinística e transparente: o stub resolve os casos `LLM_REVIEW` com taxas de erro por banda vindas da config, sem qualquer chamada de rede. Medimos seu efeito sobre precisão/recall/F1 finais e acompanhamos o herói. Deixamos explícito que isto é uma **simulação** — um LLM real exigiria o protocolo `dual_agent_plus_arbiter` com dossiês e guardas de vazamento de PII.

**O que vem a seguir.** Com todos os estágios reproduzidos, fechamos o notebook com uma síntese do que foi demonstrado e — com igual destaque — das **limitações** que delimitam honestamente o que estes resultados significam.""",
    ),
]


FASE_INTERATIVO: list[tuple[str, str]] = [
    (
        "md",
        """## 15. Painel interativo (opcional)

**Objetivo de aprendizagem.** Ao final desta seção, você será capaz de **manipular** o limiar de decisão e a inclinação (*slope*) do Platt e **observar em tempo real** como precisão, recall e custo esperado se movem — conectando, de forma tátil, calibração → ponto de operação → custo.

**Intuição.** As seções anteriores fixaram limiares e *slope*. Aqui você "pega no volante": mover o limiar para a direita exige mais confiança para declarar `MATCH` (sobe a precisão, cai o recall); reescalar o *slope* torna a curva de calibração mais íngreme ou mais suave. O custo esperado (FP×10 + FN×50, modo `vigilancia`) reage a cada escolha. Reutilizamos o conjunto de **teste held-out** e o modelo Platt da seção 9.3 — ou seja, brincamos com a rota metodologicamente correta.

> **Headless-safe (DEC-10 / CA-G1).** Esta célula **sempre** renderiza uma figura **estática** (limiar 0,50, *slope* ×1,0), garantindo saída reprodutível na execução automatizada (`papermill`/`nbconvert`). Os **sliders** do `ipywidgets` só ativam quando há um frontend Jupyter ao vivo; em modo headless são ignorados sem erro (protegidos por `try/except`). Para interagir, abra o notebook no Jupyter Lab/Notebook e execute esta célula.""",
    ),
    (
        "code",
        """import matplotlib.pyplot as plt
import numpy as np  # noqa: F401  (garante np disponível mesmo se a seção 8 não rodou)

from gzcmd_record_linkage.calibration import PlattModel, predict_platt
from gzcmd_record_linkage.metrics import confusion_counts, f1, precision, recall

# Reutilizamos o conjunto de teste held-out e o modelo Platt ajustado no treino (seção 9.3).
nota_test = df_test["nota_final"].to_numpy(dtype=float)
y_test_int = df_test["TARGET"].astype(int).to_numpy()
CUSTO_FP, CUSTO_FN = 10.0, 50.0  # custos do modo "vigilancia"


def painel_operacao(limiar=0.50, escala_slope=1.0):
    \"\"\"Recalcula p_cal com o slope reescalado, aplica o limiar e mostra o efeito.\"\"\"
    modelo = PlattModel(
        intercept=platt_holdout.intercept,
        slope=platt_holdout.slope * escala_slope,
    )
    p = predict_platt(pd.Series(nota_test), model=modelo).to_numpy()
    pred = (p >= limiar).astype(int)
    cc = confusion_counts(y_test_int, pred)
    prec, rec, f = precision(cc), recall(cc), f1(cc)
    custo = cc.fp * CUSTO_FP + cc.fn * CUSTO_FN

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.hist(p[y_test_int == 1], bins=20, range=(0, 1), alpha=0.6, label="match (y=1)")
    ax1.hist(p[y_test_int == 0], bins=20, range=(0, 1), alpha=0.6, label="nao-match (y=0)")
    ax1.axvline(limiar, color="red", linestyle="--", label=f"limiar = {limiar:.2f}")
    ax1.set_title("Probabilidade calibrada por classe (teste held-out)")
    ax1.set_xlabel("p_cal")
    ax1.set_ylabel("contagem de pares")
    ax1.legend()

    ax2.bar(
        ["precisao", "recall", "F1"],
        [prec, rec, f],
        color=["#4C72B0", "#55A868", "#C44E52"],
    )
    ax2.set_ylim(0, 1.08)
    ax2.set_title(f"Custo esperado (FP x{CUSTO_FP:.0f} + FN x{CUSTO_FN:.0f}) = {custo:.0f}")
    ax2.set_xlabel("metrica")
    ax2.set_ylabel("valor")
    for i, v in enumerate([prec, rec, f]):
        ax2.text(i, v + 0.02, f"{v:.2f}", ha="center")
    fig.tight_layout()
    plt.show()


# Versao ESTATICA: sempre executa (garante saida no modo headless -- papermill/nbconvert).
painel_operacao(limiar=0.50, escala_slope=1.0)

# Versao INTERATIVA: so ativa quando ha um frontend de widgets (Jupyter ao vivo).
# Protegida por try/except para nunca quebrar a execucao headless (DEC-10 / CA-G1).
try:
    import ipywidgets as widgets
    from IPython.display import display

    painel = widgets.interactive(
        painel_operacao,
        limiar=widgets.FloatSlider(
            value=0.50, min=0.0, max=1.0, step=0.01, description="Limiar MATCH"
        ),
        escala_slope=widgets.FloatSlider(
            value=1.0, min=0.3, max=3.0, step=0.1, description="Escala slope"
        ),
    )
    display(painel)
    print("Sliders ativos: arraste para ver precisao/recall/custo ao vivo.")
except Exception as exc:  # pragma: no cover - fallback headless
    print(f"[modo nao-interativo: {type(exc).__name__}] use a figura estatica acima.")""",
    ),
    (
        "md",
        """**Recap da seção.** Você ajustou interativamente o limiar e o *slope* e viu o trade-off precisão × recall × custo se materializar. É exatamente esse trade-off que a política de custo esperado (seção 11) **automatiza** ao escolher o ponto de operação de cada modo. A seguir, fechamos com conclusões e limitações.""",
    ),
]


FASE_4_1: list[tuple[str, str]] = [
    (
        "md",
        """## 16. Conclusões e limitações

**Objetivo de aprendizagem.** Ao final desta seção, você será capaz de **resumir** o que cada
estágio do GZ-CMD++ v3 faz, **articular** por que a metodologia adotada é honesta e **enumerar**
as limitações que impedem leituras exageradas dos resultados.

### 16.1 O que demonstramos

Percorremos o pipeline inteiro sobre um dataset **100% sintético**, estágio a estágio:

1. **Carga e feature engineering** — dos subscores crus do comparador às features agregadas,
   flags e MACD que o pacote realmente consome.
2. **Bandas** — o binning de `nota_final` que organiza a zona cinzenta (`grey_*`) onde mora a
   incerteza.
3. **Calibração (Platt)** — derivamos a matemática e mostramos **duas rotas**: a rota A
   *in-sample* (que reproduz a ferramenta) e a rota B *held-out* (a única que mede
   generalização). Validamos a calibração contra a posterior verdadeira `p_true`, com **ECE** e
   **Brier** numéricos — não apenas o gráfico.
4. **Guardrails** — as regras determinísticas de segurança e os casos sintéticos que cada uma
   dispara.
5. **Triagem por custo esperado** — a decisão `MATCH`/`NONMATCH`/`LLM_REVIEW` nos modos
   `vigilancia` (recall) e `confirmacao` (precisão).
6. **Reconciliação com `run_v3`** — a rota A manual reproduz `run_v3` **bit a bit** para o Platt
   determinístico (banda idêntica, `p_cal` com `atol≤1e-9`, ação idêntica).
7. **Avaliação held-out multi-seed** — precisão/recall/F-beta e cobertura automática com
   variância entre seeds, curvas PR/ROC e a superfície de custo vs. limiar ligando a política ao
   ponto de operação ótimo.
8. **Revisão LLM (stub)** — uma simulação determinística e transparente do estágio clerical.

### 16.2 Limitações (leia antes de generalizar)

Estes resultados são uma **demonstração didática**, não evidência empírica sobre dados reais.
As limitações abaixo são parte essencial da leitura honesta:

- **Dados sintéticos.** Todo o dataset foi gerado por código (`synthetic_data.py`) a partir de uma
  posterior verdadeira conhecida `p_true`. Isso é ótimo para *provar* que a calibração recupera a
  verdade, mas as distribuições, a prevalência (`match_ratio`) e a estrutura de subscores foram
  escolhidas para a didática — **não** refletem uma base populacional real.
- **In-sample × held-out (R-10).** O `run_v3` calibra *in-sample* (ajusta e pontua as mesmas
  linhas). Um *reliability diagram* sobre esses mesmos dados é otimista por construção. **Só** a
  rota B (held-out, via `evaluate_v3_dataframe`) mede generalização — e foi nela que reportamos as
  métricas.
- **Config × código (R-11).** A config promete calibração `anchor_platt` **por banda** e uma regra
  `grey_mother_missing`; o código implementado faz **Platt global** e **não** aplica aquela regra.
  Ensinamos **o que o código faz**, tratando a config como intenção/roadmap.
- **Vazamento por grupo.** Demonstramos o split *group-aware*, mas neste sintético o efeito de
  vazamento é **negligenciável por construção** (os grupos `COMPREC`/`REFREC` são quase todos
  singletons). Em produção, com *blocking* e múltiplos pares por registro, o split por linha
  inflaria as métricas — por isso o split por grupo é a prática correta.
- **LLM simulado (R-05).** A etapa de revisão é um **stub determinístico** que usa o rótulo
  verdadeiro `TARGET` mais taxas de erro por banda. **Não** é um LLM real, não há chamada de rede,
  e a "acurácia" do stub é um artefato das taxas injetadas — não uma medida de um modelo real.
- **XGBoost não-determinístico (R-13).** O apêndice de ML é apenas **qualitativo**: a reconciliação
  exata vale para o Platt determinístico, não para o XGBoost.
- **Âncora de guardrail.** O `ALWAYS_MATCH` real exige `nota_final ≥ 10` **e** nome/data/município
  perfeitos — mais estrito do que a leitura ingênua de "`nota ≥ 9`".

### 16.3 Recap final

O valor deste notebook não é "o modelo acertou X%", e sim **mostrar como cada decisão é tomada,
por quê, e sob quais hipóteses**. A separação rota A × rota B, a validação contra `p_true` e a
transparência sobre simulação e divergências config×código são o que transformam uma demo bonita
em **material acadêmico defensável**.""",
    ),
]


ALL_PHASES: list[list[tuple[str, str]]] = [
    FASE_2_1,
    FASE_2_2,
    FASE_2_3,
    FASE_2_4,
    FASE_2_5,
    FASE_2_6,
    FASE_3_1,
    FASE_3_2,
    FASE_3_3,
    FASE_INTERATIVO,
    FASE_4_1,
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
