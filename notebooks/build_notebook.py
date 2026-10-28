"""Gerador determinístico do notebook ``gzcmd_passo_a_passo.ipynb``.

Por que um *builder*?
---------------------
O notebook da apresentação cresce fase a fase (Wave 2 do plano) e precisa ser
**reprodutível**, **versionável com saídas limpas** e **auditável célula a célula**
(regra DF-3: toda célula de código "substantiva" é precedida por markdown
explicativo). Manter o ``.ipynb`` (JSON volumoso) à mão é frágil; descrevê-lo como
uma lista estruturada de células em Python torna o processo determinístico e
fácil de revisar em *diffs*.

O artefato versionado/entregue continua sendo o ``.ipynb``. Este script apenas o
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
# Bootstrap de ambiente: torna o notebook executável tanto LOCALMENTE quanto no
# GOOGLE COLAB.
#
# - Localmente (repositório já clonado): localiza a raiz do repo procurando o
#   `pyproject.toml` nas pastas-pai e ajusta o `sys.path`.
# - No Google Colab (você abriu apenas o .ipynb, sem o repositório): clonamos o
#   repositório e instalamos a biblioteca. Assim a lib `gzcmd_record_linkage`, os
#   módulos do notebook (`synthetic_data`/`nb_helpers`) E os dados versionados
#   (`data/synthetic/comparador_sintetico.csv`) ficam disponíveis, é isto que
#   evita a "pasta de dados vazia".
import subprocess
import sys
from pathlib import Path

# Repositório público com a biblioteca, os módulos do notebook e os dados.
REPO_URL = "https://github.com/marco-jardim/gzcmd-record-linkage.git"


def _find_repo_root(start: Path) -> Path | None:
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return None


REPO_ROOT = _find_repo_root(Path.cwd().resolve())

if REPO_ROOT is None:
    if "google.colab" not in sys.modules:
        raise RuntimeError(
            "Raiz do repositório não encontrada. Abra o notebook dentro do "
            "repositório clonado; instalação automática só está disponível no Colab."
        )
    # Não há repositório no disco (cenário típico do Google Colab). Clonamos e
    # instalamos o pacote. A primeira execução leva ~1 min; as seguintes reusam.
    _dest = Path.cwd() / "gzcmd-record-linkage"
    if not (_dest / "pyproject.toml").exists():
        print(f"Clonando {REPO_URL} ...")
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, str(_dest)], check=True
        )
    print("Instalando a biblioteca gzcmd_record_linkage (pip)...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", str(_dest)], check=True
    )
    REPO_ROOT = _dest

NB_DIR = REPO_ROOT / "notebooks"
for _path in (str(REPO_ROOT / "src"), str(NB_DIR)):
    if _path in sys.path:
        sys.path.remove(_path)
    sys.path.insert(0, _path)

DATA_DIR = REPO_ROOT / "data" / "synthetic"
DATA_DIR.mkdir(parents=True, exist_ok=True)

_csv_versionado = DATA_DIR / "comparador_sintetico.csv"
print(f"Raiz do repositório: {REPO_ROOT}")
print(f"Pasta de dados.....: {DATA_DIR}")
print(f"CSV versionado já presente: {_csv_versionado.exists()}")"""


# ===========================================================================
# FASE 2.1: Estrutura, Narrativa e Setup
# ===========================================================================
FASE_2_1: list[tuple[str, str]] = [
    (
        "md",
        """\
# GZ-CMD++ v3: passo a passo sobre um dataset sintético

**Material didático em pt-BR**, para quem vem da saúde coletiva e da epidemiologia.
Relacionar SIM, SINAN ou SIH para encontrar a mesma pessoa é fazer *record linkage*.
Aqui usamos **somente dados sintéticos**, sem dados pessoais reais, para percorrer
o motor de decisão `gzcmd_record_linkage`.

**Como ler:** primeiro a intuição, depois a conta e o código. Um par "herói" será
nosso exemplo-fio-condutor. Seu `PASSAPORTE` guarda o que cada estágio acrescenta.
As figuras sempre têm uma pergunta; os controles convidam a testar hipóteses.

🧸 **Bem simples**: a ideia em duas frases · 🧒 **Um pouco mais**: o que o código faz · 📖 **Palavra nova** · 📐 **Para quem quer a matemática** (bloco recolhido; pode pular sem perder a história) · 🎯 **Para levar** · 🎛️ **controle** · ⚠️ **honestidade** · 🔎 **detalhe**.

## Glossário rápido

| Termo | Definição |
|-------|-----------|
| **Record linkage** | Decidir se dois registros (ex.: de bases distintas) referem-se à **mesma entidade** (pessoa). |
| **Par comparador** | Uma dupla (registro de referência × registro candidato) com subscores de similaridade. |
| **Nota final (`nota_final`)** | Placar de semelhança entre as fichas, dado pelo OpenRecLink, que não é uma probabilidade em porcentagem. |
| **Banda** | Faixa discreta de `nota_final` (ex.: `grey_mid`) usada para perfis de erro e monitoramento. |
| **Zona cinzenta** | Faixa intermediária de notas onde a decisão automática é arriscada. |
| **Calibração** | Ajustar a passagem da nota para uma probabilidade que combine com o que acontece nos pares. |
| **Probabilidade calibrada** | Probabilidade anunciada que, em grupos de pares parecidos, combina com a frequência de serem a mesma pessoa. |
| **Platt scaling** | Maneira de transformar notas em probabilidades ajustando a inclinação e o intercepto de uma curva. |
| **Sigmoide** | Curva em forma de S que mantém a probabilidade entre impossível e certo. |
| **Odds** | Probabilidade de acontecer dividida pela de não acontecer: p ÷ (1 − p). É a 'chance' da epidemiologia (razão de chances = OR). Neste material evitamos a palavra 'chance' e dizemos probabilidade. |
| **Log-odds** | Logaritmo das odds: uma régua sem limites, de menos infinito a mais infinito. |
| **Log-loss** | Multa por surpresa que cobra mais de uma previsão errada e muito confiante. |
| **Freio L2** | Pequena multa que desestimula uma curva inclinada demais. |
| **Passo de Newton** | Maneira de melhorar o ajuste usando a descida e a curvatura do caminho. |
| **Treino/teste** | Pares usados para aprender e pares separados para conferir o aprendizado. |
| **Vazamento** | Uso de informação da prova para ajudar a preparar a resposta. |
| **Guardrail** | Regra de segurança determinística que decide MATCH/NONMATCH direto ou coloca o par no topo da fila de revisão. |
| **Triagem (*triage*)** | Escolha base por menor custo esperado; a ação final ainda depende de guardrails, travas e vagas de revisão. |
| **TARGET** | Gabarito binário: 1 = mesma pessoa, 0 = pessoas distintas. |
| **ECE** | Erro médio entre a probabilidade prometida e a frequência observada, agrupando previsões parecidas. |
| **Brier** | Medida do erro das probabilidades par a par, que dá mais peso aos erros grandes. |
| **Diagrama de confiabilidade** | Desenho que compara a probabilidade prometida com a frequência observada em cada grupo. |
| `p*` | Probabilidade verdadeira escondida que conhecemos apenas porque inventamos estes dados, só nos 600 pares aleatórios; nos 7 cenários o rótulo foi escolhido à mão. |
| **Âncora** | Grupo de pares ao qual atribuímos uma probabilidade presumida para guiar o ajuste sem gabarito. |
| **Rótulo suave** | Resposta de quase certeza, em vez de declarar certeza absoluta. |
| **Trava** | Corte de probabilidade: MATCH base abaixo do corte de match, ou NONMATCH base acima do corte de não-match, pede revisão; sem vaga, mantém a escolha base. Não confundir com o limitador por janela (§9.8). |
| **Vaga de revisão** | Uma unidade do orçamento de revisões. |
| **Limitador por janela** | Regra que limita quanto cada controle da curva pode mudar entre períodos. |
| **Classificador** | Programa que aprende com exemplos respondidos a distinguir mesma pessoa de pessoas diferentes. |
| **`GZCMDClassifier`** | Programa da biblioteca que aprende com várias pistas e ajusta as probabilidades anunciadas (§13.7). |
| **Árvore de decisão** | Sequência de perguntas sobre as pistas que leva a uma resposta, como um jogo de adivinhação. |
| **Random Forest** | Comitê de árvores que veem partes diferentes dos dados e juntam seus votos. |
| **XGBoost** | Equipe de árvores em que cada nova árvore tenta corrigir os erros que ainda sobraram. |
| **Overfitting** | Decorar os exemplos de estudo em vez de aprender algo que ajude em fichas novas. |
| **Validação cruzada** | Revezar partes do treino entre estudar e conferir, sem usar a prova final. |
| **Importância de variável** | Crédito recebido por uma pista nas perguntas das árvores, sem provar que ela causa o resultado. |
| **Avaliação held-out** | Prova com pares reservados que o ajuste nunca viu. |
| **Semente (seed)** | Ponto de partida que permite repetir um sorteio do computador. |
| **Precisão** | Entre os pares aceitos como mesma pessoa, a parte que realmente é da mesma pessoa, como o VPP. Nas métricas automáticas, só entram os pares decididos, sem os pendentes de revisão. |
| **Recall automático** | Entre os pares da mesma pessoa decididos automaticamente, a parte aceita. Só equivale à sensibilidade do sistema inteiro quando não há pendentes. |
| **Fβ** | Nota que reúne precisão e recall, dando prioridade diferente a cada uma conforme o objetivo. Nas métricas automáticas, só entram os pares decididos, sem os pendentes de revisão. |
| **Cobertura automática** | Parte dos pares resolvida sem ficar pendente de revisão, não a parte que foi acertada. |
| **Curva ROC** | Desenho da sensibilidade contra 1 − especificidade ao mudar o corte de aceitação. |
| **Curva PR** | Desenho que acompanha precisão e sensibilidade ao mudar o corte de aceitação. |
| **AUC** | Área sob uma curva, usada para resumir seu desenho, com significado diferente na ROC e na PR. |
| **Rota A / rota B** | A: ajusta e confere nas mesmas fichas (confere o código). B: ajusta no treino e confere em fichas separadas (mede o desempenho). |
| **In-sample** | Previsão sobre uma ficha que já foi usada no ajuste. |
| **Corte (limiar)** | Valor de probabilidade a partir do qual o par é aceito. |
| **AP** | Precisão média ao longo da fila ordenada; resume a curva PR. |
| **MAE** | Distância média, sem sinal, entre duas probabilidades. |
| **Desvio-padrão** | Quanto os resultados se espalham em torno da média. |
| **Boosting** | Aprender em etapas, cada uma corrigindo o que sobrou das anteriores. |
| **Calibração interna** | Tradução de palpites em probabilidades feita dentro do classificador, em partes do treino. |

> Termos adicionais (ex.: *held-out*, *base rate*, *EVR* (ver 11.2)) são definidos no ponto em que aparecem.""",
    ),
    (
        "md",
        """\
## 1. Contexto: por que *record linkage* é difícil?

🧒 Imagine juntar duas listas de chamada de escolas diferentes sem um número de matrícula em comum. Uma ficha diz
*"Maria S. Souza, 12/03/1981"*; outra, *"Maria Silva de Souza, 13/03/1981"*.
É a mesma pessoa com um erro de digitação ou são duas pessoas diferentes?

O comparador mede **pistas de semelhança**: nome, mãe, nascimento e endereço.
Ele também entrega a **`nota_final`**, um placar, não uma probabilidade.
As pontas da escala costumam ser mais fáceis; a **zona cinzenta**, no meio,
exige combinar incerteza, regras de segurança e o preço dos erros.

Um **falso positivo** une pessoas distintas; um **falso negativo** deixa de unir
a mesma pessoa. Como num teste de rastreamento, esses erros não custam o mesmo.
É por isso que "passou de uma nota fixa?" não basta.""",
    ),
    (
        "md",
        """\
## 2. Visão geral do pipeline GZ-CMD++ v3

Seguiremos as funções reais da biblioteca, uma transformação por vez:

```
CSV comparador → loader (pistas organizadas)
    → BandAssigner (faixa da nota)
    → calibração Platt (nota em probabilidade)
    → apply_guardrails (regras de segurança)
    → PolicyEngineV3.triage (menor custo esperado)
    → MATCH / NONMATCH / LLM_REVIEW
```

Dois **modos** mudam o preço dos erros:

- **`vigilancia`**: prioriza *recall* (sensibilidade), tentando perder poucos matches.
- **`confirmacao`**: prioriza *precision* (valor preditivo positivo), tentando aceitar poucos pares errados.

⚠️ **Duas rotas, duas perguntas.** A rota **A** reproduz `run_v3`: ajusta e pontua
nas mesmas linhas (*in-sample*). Serve para conferir o código, não a generalização.
A rota **B** separa treino e teste (*held-out*), como uma coorte de validação.
Só ela mede o desempenho em pares que o ajuste não viu. Na §13.7, também
testamos um aprendiz que combina pistas, o `GZCMDClassifier` (Random Forest/XGBoost, explicados lá),
sem mudar a divisão treino/teste nem a política de custos.""",
    ),
    (
        "md",
        """\
## 3. Objetivos da apresentação

- Entender as transformações de CSV até decisão.
- Refazer as contas das features e da calibração.
- Comparar MATCH, NONMATCH e revisão pelo custo esperado.
- Reconciliar o passo a passo com `run_v3` (rota A).
- Avaliar no teste e reconhecer os limites do mundo sintético (rota B).""",
    ),
    (
        "md",
        """\
## 4. Setup do ambiente e geração do dataset

**No Colab:** o bootstrap clona o repositório e instala a biblioteca se necessário.
**Localmente:** apenas encontra a raiz e ajusta os caminhos.
Depois fixamos a semente e geramos o CSV (`;` como separador, `,` como decimal).""",
    ),
    ("code", BOOTSTRAP),
    (
        "md",
        """\
**Imports + configuração.** `cfg` vem do carregador da biblioteca; `ANCORAS` lê
também o bloco de âncoras da YAML crua, reservado para a demonstração didática.
Uma semente fixa torna os sorteios reproduzíveis.""",
    ),
    (
        "code",
        """\
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

import nb_helpers as nbh
import synthetic_data
from gzcmd_record_linkage import __version__ as gzcmd_version
from gzcmd_record_linkage.config import load_config

pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 140)
pd.set_option("display.float_format", lambda v: nbh.fmt_br(v, 4))


br = nbh.fmt_br


def confere(rotulo, valor, esperado, tol):
    assert abs(valor - esperado) <= tol, (
        f"O texto cita {esperado} para {rotulo}, mas o código mediu {valor}: atualize o texto."
    )
    print(f"✓ {rotulo}: medido={br(valor, 6)}; texto={br(esperado, 6)}; tolerância={br(tol, 6)}")


SEED = 42
CONFIG_PATH = str(REPO_ROOT / "src" / "gzcmd_record_linkage" / "gzcmd_v3_config.yaml")
cfg = load_config(CONFIG_PATH)
ANCORAS = nbh.load_anchor_spec(CONFIG_PATH)
print(f"gzcmd_record_linkage v{gzcmd_version} | seed global = {SEED}")""",
    ),
    (
        "md",
        """\
**Dataset:** 600 pares aleatórios mais 7 cenários narrativos, incluindo homônimo,
mãe ausente e zona cinzenta. `.frame` contém as pistas cruas; `.meta`, os metadados.
Guardamos `.p_true` (p*) à parte: é a probabilidade verdadeira que gerou o rótulo
dos 600 pares aleatórios; nos 7 cenários narrativos o rótulo foi escolhido à mão.
É usada só para validar, nunca como entrada do pipeline.""",
    ),
    (
        "code",
        """\
from nb_helpers import fmt_br as br

dataset = synthetic_data.generate_comparador(seed=SEED, n_pairs=600, scenarios="all")
df_raw = dataset.frame
p_true = dataset.p_true
meta = dataset.meta

CSV_PATH = DATA_DIR / "comparador_sintetico.csv"
if CSV_PATH.exists():
    import tempfile

    with tempfile.TemporaryDirectory() as pasta_temporaria:
        esperado = Path(pasta_temporaria) / CSV_PATH.name
        synthetic_data.to_comparador_csv(df_raw, esperado)
        assert esperado.read_text(encoding="utf-8") == CSV_PATH.read_text(encoding="utf-8"), (
            "O CSV versionado difere dos dados gerados. Confira a versão do gerador."
        )
    print("CSV versionado conferido e preservado.")
else:
    synthetic_data.to_comparador_csv(df_raw, CSV_PATH)
    print("CSV ausente: arquivo criado.")

print(f"Pares gerados......: {len(df_raw)}")
print(f"Taxa de match (real): {br(meta['match_ratio_realized'])}")
print(f"Verdade-base p*(s) = sigmoide({br(meta['true_slope'])}*nota + ({br(meta['true_intercept'])}))")
print(f"CSV salvo em.......: {CSV_PATH.relative_to(REPO_ROOT)}")""",
    ),
    (
        "md",
        """**Conferência automática do dataset.** Contamos as linhas e mostramos
p* ao lado do TARGET dos cenários narrativos: nesses casos o rótulo foi escolhido
à mão, não sorteado a partir de p*.""",
    ),
    (
        "code",
        """\
confere("pares aleatórios", len(df_raw) - len(meta["scenarios"]), 600, 0)
confere("cenários narrativos", len(meta["scenarios"]), 7, 0)
display(pd.DataFrame([
    {"cenário": nome, "p*": float(p_true.iloc[pos]),
     "TARGET (escolhido à mão)": int(df_raw.iloc[pos]["PAR"] in (1, 2))}
    for nome, pos in meta["scenarios"].items()
]).set_index("cenário"))""",
    ),
    (
        "md",
        """\
## 5. Primeiro olhar nos dados

Antes de modelar, veja cinco linhas e o que cada grupo mede:

| Colunas cruas | Leitura |
|---|---|
| `COMPREC,C,12,0` / `REFREC,C,12,0` | IDs do candidato e da referência. |
| `PASSO` / `PAR` | Blocagem: a rodada que escolheu quais pares comparar / rótulo (1 ou 2 = match; 0 = não-match). |
| `nota final` | Placar agregado do comparador, não uma probabilidade. |
| `NOME *` / `NOMEMAE *` | Semelhança dos fragmentos do nome e do nome da mãe. |
| `DTNASC *` / `ENDERECO *` | Comparações de nascimento e endereço. |
| `CODMUNRES local igual` | Município coincidente. |
| `R_DTNASC` / `C_DTNASC` | Nascimentos em YYYYMMDD. |
| `R_DTOBITO` / `C_DTDIAG` | Óbito e diagnóstico, para plausibilidade temporal. |

🔎 `p_true` fica fora do CSV de entrada, guardada no objeto `dataset`.""",
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
df_raw[colunas_visao].head(5)""",
    ),
    (
        "md",
        """\
## 6. Apresentando o "herói": nosso exemplo-fio-condutor

🧒 Vamos acompanhar um caso-índice: o cenário `zona_cinzenta`, nem um "sim"
óbvio, nem um "não" óbvio. Seu **`PASSAPORTE`** começa com os dados de entrada;
cada estágio acrescenta uma anotação, sem repetir um cartão inteiro.
Abaixo estão as pistas cruas que usaremos na conta à mão da seção 7.""",
    ),
    (
        "code",
        '''\
HERO_SCENARIO = "zona_cinzenta"
hero_pos = meta["scenarios"][HERO_SCENARIO]
hero_idx = df_raw.index[hero_pos]
hero_raw = df_raw.loc[hero_idx]
PASSAPORTE = {"6. Entrada": hero_raw[
    [synthetic_data.COMPREC, synthetic_data.REFREC, "PAR", "nota final"]
].to_dict()}
subscores_heroi = [c for c in df_raw.columns if c.startswith(("NOME", "DTNASC"))]
display(hero_raw[["nota final", *subscores_heroi]].to_frame("herói"))''',
    ),
]


# ===========================================================================
# FASE 2.2: Carga e Feature Engineering
# ===========================================================================
FASE_2_2: list[tuple[str, str]] = [
    (
        "md",
        """\
## 7. Carga dos dados e *feature engineering*

> 🧒 **Como para uma criança de 11 anos.** Imagine dois detetives comparando duas
> fichas. Um deles anota pistas: "os nomes batem em 6 de 10 pedaços", "as datas
> diferem por um dia", "a cidade é a mesma". O `loader` é o assistente que pega
> dezenas dessas anotações soltas e as organiza em poucas **notas de 0 a 1**, uma
> por assunto, e algumas **bandeirinhas de alerta**. Ele não decide nada; só
> arruma a mesa para quem vai decidir.

> ⚠️ **Uma honestidade logo de início.** A calibração Platt da biblioteca usa
> **apenas a `nota_final`**. As colunas desta seção alimentam os *guardrails*
> (seção 10), o aprendiz que combina pistas (`GZCMDClassifier`, seção 13.7), o dossiê
> do revisor (resumo das pistas para a segunda opinião) e o acompanhamento de mudanças nos dados (*drift*). Entender como elas nascem é entender
> o que esses estágios "enxergam".

Carregamos o CSV com MACD ativado (grupo de colunas `macd_*` que compara datas em dias) e guardamos os agregados do herói
no passaporte.""",
    ),
    (
        "code",
        """\
from gzcmd_record_linkage.loader import LoadConfig, load_comparador_csv

df = load_comparador_csv(CSV_PATH, cfg=LoadConfig(macd_enabled=True))
colunas_agregadas = [
    "nome_score_total", "mae_score_total", "dtnasc_score_total",
    "endereco_score_total", "municipio_score",
]
PASSAPORTE["7. Features"] = df.loc[hero_idx, colunas_agregadas].to_dict()
print(f"Linhas: {len(df)} | Colunas: {df.shape[1]}")
print(f"TARGET derivado de PAR -> positivos: {int(df['TARGET'].sum())} / {len(df)}")""",
    ),
    (
        "md",
        r"""### 7.1 A receita de cada nota, conta por conta

**🧒 Os três juízes do nome.** O primeiro juiz conta **quantos pedaços** do nome
batem e vale metade da nota. O segundo só olha se o **primeiro nome** bate e vale
um quarto. O terceiro só olha o **último sobrenome** e vale o outro quarto.
A nota do nome é a soma dos votos.

**📐 Passo a passo com o herói.** Pistas cruas: `NOME qtd frag iguais = 0,6`,
`NOME prim frag igual = 1`, `NOME ult frag igual = 0`.
1. Juiz da quantidade: 0,5 × 0,6 = 0,30
2. Juiz do primeiro nome: 0,25 × 1 = 0,25
3. Juiz do último sobrenome: 0,25 × 0 = 0,00
4. Soma: 0,30 + 0,25 + 0,00 = **0,55**
5. "Prender" em [0, 1] (`clip`): 0,55 já está dentro.

$$\text{nome\_score\_total}=\operatorname{clip}_{[0,1]}
\big(0{,}5\,q+0{,}25\,\mathbb 1_{\text{prim}}+0{,}25\,\mathbb 1_{\text{ult}}\big)$$

Como os pesos somam 1 e cada pista vive em [0, 1], o `clip` é só um cinto de
segurança contra dados fora do padrão. O nome da mãe usa **a mesma receita**:
no herói, 0,5 × 0,5 + 0,25 × 0 + 0,25 × 0 = **0,25**.

**Data de nascimento: a média de cinco perguntas de sim ou não** (igual?
diferente por um dígito? dia invertido? mês invertido? ano invertido?).
O herói só acende "diferente por um dígito": 1 ÷ 5 = **0,2**.

> 🔎 **Pegadinha que vale ouro.** Uma data **idêntica** costuma dar só 0,2, porque
> "igual" e "quase igual" nunca acendem juntos (só as inversões, raras, somariam mais).
> Nesta coluna, 0,2 não quer dizer
> "data ruim"; quer dizer "uma das cinco formas de concordar aconteceu".
> Por isso se lê a receita antes de interpretar o número.

**Endereço e município.** `endereco_score_total` é a soma dos seis subscores de
endereço dividida por 6; subscore ausente conta como 0. `municipio_score` vale 1
quando o município coincide.
**Bandeirinhas:** `mae_missing`, `endereco_zero` e `dtnasc_all_zero` acendem quando
todos os subscores do grupo são zero. Isso pode indicar campo vazio, não só
discordância, e merece indicador próprio.

No herói, os seis subscores de endereço valem 0,5: a média é **0,5**.
A seguir, conferimos as contas e comparamos as médias por `TARGET`.""",
    ),
    (
        "code",
        """\
hero_raw = df_raw.loc[hero_idx]
nome_manual = np.clip(
    0.5 * hero_raw["NOME qtd frag iguais"]
    + 0.25 * hero_raw["NOME prim frag igual"]
    + 0.25 * hero_raw["NOME ult frag igual"], 0, 1,
)
for rotulo, valor, esperado, tol in [
    ("peso da quantidade no loader", nbh.NOME_PESOS_LOADER[0], 0.5, 0.05),
    ("peso do primeiro nome no loader", nbh.NOME_PESOS_LOADER[1], 0.25, 0.005),
    ("peso do último nome no loader", nbh.NOME_PESOS_LOADER[2], 0.25, 0.005),
    ("voto da quantidade no herói", 0.5 * hero_raw["NOME qtd frag iguais"], 0.30, 0.005),
    ("voto do primeiro nome no herói", 0.25 * hero_raw["NOME prim frag igual"], 0.25, 0.005),
    ("voto do último nome no herói", 0.25 * hero_raw["NOME ult frag igual"], 0.00, 0.005),
]:
    confere(rotulo, valor, esperado, tol)
mae_manual = np.clip(
    0.5 * hero_raw["NOMEMAE qtd frag iguais"]
    + 0.25 * hero_raw["NOMEMAE prim frag igual"]
    + 0.25 * hero_raw["NOMEMAE ult frag igual"], 0, 1,
)
perguntas_data = [
    "DTNASC dt iguais", "DTNASC dt ap 1digi", "DTNASC dt inv dia",
    "DTNASC dt inv mes", "DTNASC dt inv ano",
]
data_manual = sum(float(hero_raw[c]) for c in perguntas_data) / 5
for coluna, esperado in {
    "NOME qtd frag iguais": 0.6, "NOME prim frag igual": 1,
    "NOME ult frag igual": 0, "NOMEMAE qtd frag iguais": 0.5,
    "NOMEMAE prim frag igual": 0, "NOMEMAE ult frag igual": 0,
    "DTNASC dt ap 1digi": 1,
}.items():
    confere(f"pista do herói: {coluna}", float(hero_raw[coluna]), esperado,
            0.05 if "qtd" in coluna else 0)
for coluna, esperado, tol in [
    ("nome_score_total", 0.55, 0.005), ("mae_score_total", 0.25, 0.005),
    ("dtnasc_score_total", 0.2, 0.05), ("endereco_score_total", 0.5, 0.05),
]:
    confere(f"nota do herói: {coluna}", float(df.loc[hero_idx, coluna]), esperado, tol)
for coluna in [c for c in df_raw if c.startswith("ENDERECO ")]:
    confere(f"endereço do herói: {coluna}", float(hero_raw[coluna]), 0.5, 0.05)
conta_manual = pd.Series({
    "nome_score_total": nome_manual,
    "mae_score_total": mae_manual,
    "dtnasc_score_total": data_manual,
}, name="conta à mão")
for coluna, valor in conta_manual.items():
    assert np.isclose(valor, df.loc[hero_idx, coluna]), coluna
display(pd.concat([conta_manual, df.loc[hero_idx, conta_manual.index]
                   .rename("loader")], axis=1))
display(df.groupby("TARGET")[colunas_agregadas].mean().round(3))""",
    ),
    (
        "md",
        """\
### 7.2 Por que a data ganha lupa própria (as *features* MACD)

**🧒 A régua de anos é grossa demais.** `diff_ano` compara só o **ano**.
31/12/1980 contra 01/01/1981 estão a um dia de distância, mas `diff_ano` diz 1.
Já 01/01/1980 contra 31/12/1980 dá `diff_ano = 0`, apesar de quase um ano de
distância. As colunas `macd_*` trocam essa régua por uma **régua em dias**.

**📐 As contas, uma por uma:**
1. `dias = |C_DTNASC − R_DTNASC|`. Herói: 15/03/1980 contra 16/03/1980, dias = 1.
2. `anos = dias ÷ 365,25` ≈ 0,003.
3. `macd_nasc_very_close = 1` se dias ≤ 7 (herói: acende).
4. `macd_nasc_close = 1` se dias ≤ 30 (herói: acende).
5. "Nome perfeito" = quantidade ≥ 0,95 **e** primeiro nome igual **e** último
   sobrenome igual (herói: 0,6 < 0,95, não é).
6. `macd_nome_perf_x_date_far = 1` se nome perfeito **e** anos ≥ 5: um alerta
   para o risco de **homônimo**, mesmo nome e gerações diferentes.
7. `macd_nome_perf_x_year_diff = (nome perfeito) × min(anos, 20)`: a mesma ideia
   em versão contínua, com teto de 20 anos.

**Por que isso importa.** Nome igual é uma pista forte e traiçoeira
("Maria da Silva" é comum). A data ajuda a distinguir a mesma pessoa de um homônimo.

> 🔎 **Dois avisos honestos.** (1) Neste gerador, datas discordantes diferem quase
> sempre por ±1 dia ou por 2 a 30 anos; quase não há diferenças entre 8 e 300 dias. Por isso
> as janelas de 7 e 30 dias mudam pouco aqui, e o controle 7.4 mostra isso.
> (2) O cenário `homonimo` tem quantidade 0,9, abaixo de 0,95, então o flag MACD
> **não** acende para ele; quem o pega é o guardrail de homonímia (seção 10).

A tabela mede a fração de pares com cada flag aceso, por classe, e exibe o homônimo.""",
    ),
    (
        "code",
        """\
flags = ["mae_missing", "dtnasc_all_zero", "endereco_zero", "diff_ano"]
macd_cols = sorted(c for c in df.columns if c.startswith("macd_"))
macd_flags_binarias = [
    "macd_nasc_year_match", "macd_nasc_month_match", "macd_nasc_day_match",
    "macd_nasc_very_close", "macd_nasc_close", "macd_nome_perf_x_date_far",
]
display(df.groupby("TARGET")[[*flags[:3], *macd_flags_binarias]].mean().round(3))
homonimo_idx = df.index[meta["scenarios"]["homonimo"]]
display(df.loc[[homonimo_idx], [
    "TARGET", "NOME qtd frag iguais", "NOME prim frag igual", "NOME ult frag igual",
    "diff_ano", *macd_cols,
]])
print("Datas do herói:", df.loc[hero_idx, "R_DTNASC_dt"],
      df.loc[hero_idx, "C_DTNASC_dt"])
dias_heroi = abs((df.loc[hero_idx, "C_DTNASC_dt"] - df.loc[hero_idx, "R_DTNASC_dt"]).days)
confere("dias entre nascimentos do herói", dias_heroi, 1, 0)
for coluna, esperado in [("R_DTNASC_dt", 19800315), ("C_DTNASC_dt", 19800316)]:
    confere(f"data do herói: {coluna}", int(df.loc[hero_idx, coluna].strftime("%Y%m%d")), esperado, 0)
confere("anos entre nascimentos do herói", dias_heroi / 365.25, 0.003, 0.0005)
confere("quantidade do nome do homônimo", df.loc[homonimo_idx, "NOME qtd frag iguais"], 0.9, 0.05)
for coluna in ["macd_nasc_very_close", "macd_nasc_close"]:
    confere(f"herói: {coluna}", df.loc[hero_idx, coluna], 1, 0)
confere("alerta MACD do homônimo", df.loc[homonimo_idx, "macd_nome_perf_x_date_far"], 0, 0)""",
    ),
    (
        "md",
        """\
### 7.3 Visualização: distribuição da `nota_final` por classe

**Pergunta:** as notas separam perfeitamente matches de não-matches?
A sobreposição na faixa intermediária mostra por que uma nota sozinha não resolve
todos os casos. A próxima seção recorta esse placar em faixas, sem eliminar a dúvida.""",
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
ax.set_title("Distribuição da nota_final por classe: a sobreposição na zona cinzenta")
ax.legend()
fig.tight_layout()
plt.show()""",
    ),
    (
        "md",
        """\
### 7.4 🎛️ W5: pesos do nome e janelas MACD

Mexa nos pesos do nome e nas janelas de dias. A figura separa a nota do nome por
classe e mostra o **AUC**: "sorteando um match e um não-match, qual a probabilidade de o
match ter nota maior?" (0,5 é cara ou coroa; 1,0 é separação perfeita; empate vale
meia vitória). Teste: (a) zerar o peso da quantidade derruba o AUC?
(b) com pesos somando mais que 1, o que o `clip` faz? (c) levar a janela de 30
para 300 dias muda algo neste dataset? Por quê?

As janelas MACD quase não mudam nada nestes dados sintéticos: as datas diferem
por ±1 dia ou por anos. Para ver uma mudança mais clara, mexa nos pesos do nome.

🔎 Os padrões vêm de **constantes do loader**, não da configuração YAML.
A conferência automática verifica que as funções auxiliares reproduzem suas colunas. O controle muda apenas
a exploração, sem alterar `df` nem as decisões posteriores. A visão estática
sempre roda; sem suporte a widgets, ela continua disponível.""",
    ),
    (
        "code",
        """\
pesos_nome = dict(zip(
    ("w_qtd", "w_prim", "w_ult"), nbh.NOME_PESOS_LOADER, strict=True,
))
defaults_features = {**pesos_nome, **nbh.MACD_JANELAS_LOADER}
assert np.allclose(nbh.nome_score(df, **pesos_nome), df["nome_score_total"])
macd_padrao = nbh.macd_flags(df, **nbh.MACD_JANELAS_LOADER)
for coluna in macd_padrao:
    assert np.allclose(macd_padrao[coluna], df[coluna], equal_nan=True), coluna


def _ui_feat(**kw):
    fig, resumo = nbh.plot_feature_explorer(df, **kw)
    plt.show()
    plt.close(fig)
    display(resumo)


if not nbh.interagir(_ui_feat, {
    "w_qtd": nbh.Deslizante(pesos_nome["w_qtd"], 0, 1, 0.05, "Peso quantidade"),
    "w_prim": nbh.Deslizante(pesos_nome["w_prim"], 0, 1, 0.05, "Peso primeiro"),
    "w_ult": nbh.Deslizante(pesos_nome["w_ult"], 0, 1, 0.05, "Peso último"),
    "very_close_days": nbh.Deslizante(
        defaults_features["very_close_days"], 0, 60, 1, "Dias muito perto", inteiro=True,
    ),
    "close_days": nbh.Deslizante(
        defaults_features["close_days"], 1, 365, 1, "Dias perto", inteiro=True,
    ),
    "far_years": nbh.Deslizante(
        defaults_features["far_years"], 1, 20, 0.5, "Anos longe",
    ),
}):
    _ui_feat(**defaults_features)""",
    ),
]


# ===========================================================================
# FASE 2.3: Atribuição de Bandas
# ===========================================================================
FASE_2_3: list[tuple[str, str]] = [
    (
        "md",
        """\
## 8. Atribuição de bandas

> 🧒 É como transformar a nota de uma prova em conceitos: em vez de "7,3",
> dizemos "faixa `grey_high`". O número continua igual; só ganha uma etiqueta.

As bandas `grey_low`, `grey_mid` e `grey_high` recortam a zona cinzenta.
As faixas servem para perfis de erro do revisor, monitoramento por fatia e
leitura humana; **não são uma decisão nem uma probabilidade**.

📐 Lemos as fronteiras de `cfg.bands.definitions`: `min <= nota < max`.
Na banda `high`, `inclusive_max=True` inclui também o máximo.
Uma nota fora de todas as faixas recebe `<NA>`. A célula aplica o `BandAssigner`
real, conta os pares e mostra os limites, sem copiá-los à mão.""",
    ),
    (
        "code",
        """\
from gzcmd_record_linkage.bands import BandAssigner

assigner = BandAssigner.from_config(cfg)
df["band"] = assigner.assign(df["nota_final"])
PASSAPORTE["8. Banda"] = {"band": df.loc[hero_idx, "band"]}

print("Contagem de pares por banda:")
ordem_bandas = [d.name for d in cfg.bands.definitions]
display(df["band"].value_counts().reindex(ordem_bandas, fill_value=0).astype(int))
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
display(fronteiras)""",
    ),
    (
        "md",
        """\
### 8.1 Visualização: a `nota_final` colorida por banda

**Pergunta:** como as fronteiras recortam a distribuição de notas?
Cada barra recebe a cor da banda do seu intervalo. As faixas `grey_*` destacam
o meio da escala. A seguir, vamos transformar o placar em probabilidade calibrada.""",
    ),
    (
        "code",
        """\
import numpy as np

band_order = ordem_bandas
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
]


# ===========================================================================
# FASE 2.4: Calibração (Platt), rotas A (in-sample) e B (held-out)
# ===========================================================================
FASE_2_4: list[tuple[str, str]] = [
    (
        "md",
        r"""## 9. Calibração: da nota a uma probabilidade confiável

> 🧸 **Bem simples.** Pense em alguém que avisa se vai chover. Você confia mais
> quando o aviso combina com o que costuma acontecer, não quando promete certeza.

> 🧒 **Um pouco mais.** Se a previsão diz 70% de chuva em muitos dias, esperamos
> chuva em cerca de 7 de cada 10 desses dias. O código busca isso para pares de fichas.

> 📖 **Palavra nova: probabilidade calibrada.** É uma probabilidade anunciada que combina
> com a frequência observada em grupos de pares com previsões parecidas.

**Exemplo pequeno.** Entre pares que recebem probabilidade 0,70, esperamos que cerca de
70% sejam a mesma pessoa; não é uma garantia sobre um par isolado.

> 🧸 **Bem simples.** Um placar de jogo conta pontos, não prevê o futuro.
> A nota das fichas também precisa de uma tradução antes de virar probabilidade.

> 🧒 **Um pouco mais.** Nota 8 não quer dizer 80% de probabilidade de ser a mesma pessoa.
> O código aprende essa tradução usando exemplos com resposta conhecida.

> 📖 **Palavra nova: placar (nota).** É uma medida de semelhança, não uma probabilidade.

<details>
<summary>📐 Para quem quer a matemática</summary>

Calibração perfeita: $P(\text{match}\mid p_{cal}=q)=q$ em todos os pontos.
Na epidemiologia, a mesma exigência vale para risco clínico: entre pacientes
com risco previsto de 10%, cerca de 10% adoecem.

</details>

> 🎯 **Para levar.** Uma nota alta e uma probabilidade confiável são coisas diferentes.""",
    ),
    (
        "md",
        r"""### 9.1 Do placar à probabilidade em quatro passos

> 🧸 **Bem simples.** Imagine traduzir o placar com uma régua que entorta nas
> pontas. Ela deixa a probabilidade chegar perto de impossível ou de certo, sem passar dali.

> 🧒 **Um pouco mais.** Em vez de trabalhar direto em porcentagens, o código
> usa uma régua intermediária, sem limites, e depois volta para as probabilidades.

> 📖 **Palavra nova: odds.** Probabilidade de acontecer dividida pela de não acontecer: é a "chance" da epidemiologia. A comparação entre odds é a razão de chances (OR). Fora desse contexto, dizemos probabilidade.
> 📖 **Palavra nova: log-odds.** É uma transformação das odds numa régua sem limites.

**Exemplo pequeno.** Probabilidade de 80% dá odds de 4 (quatro para um); 50% dá odds de 1.

> 🧸 **Bem simples.** A régua desenha uma estrada que lembra uma cobra se espreguiçando.
> No meio ela sobe depressa; nas pontas vai ficando quase deitada.

> 🧒 **Um pouco mais.** Essa curva em S devolve uma probabilidade entre 0% e 100%,
> mesmo quando a conta intermediária dá um número muito grande ou muito pequeno.

> 📖 **Palavra nova: sigmoide.** É a curva em S que faz essa volta para probabilidades.

> 🧸 **Bem simples.** Um parafuso deixa a subida mais forte e também arrasta o meio da curva;
> o outro sobe ou desce a curva inteira.

> 🧒 **Um pouco mais.** O código aprende esses dois controles: inclinação `a`
> e intercepto `b`, o valor da régua intermediária z quando a nota é zero.
> Como s* = −b/a, mudar a inclinação também muda a nota em que a probabilidade vira 50%.

> 📖 **Palavra nova: Platt scaling.** É aprender esses dois controles para traduzir notas em probabilidades.

**Exemplo pequeno.** Com inclinação 0,85 e intercepto −5,1 (valores só de exemplo), nota 6,5 vira cerca de 60%; nota 6,0 fica em 50%, o meio da subida.

<details>
<summary>📐 Para quem quer a matemática</summary>

**Por que não uma reta direto?** Uma reta que acompanha bem o meio da curva,
p = 0,5 + 0,21 × (nota − 6), daria 1,34 (134%) na nota 10 e −0,34 (−34%) na nota 2.
Precisamos de uma conta que aceite qualquer número e devolva sempre algo entre
0 e 1. O truque é passar pelo mundo onde retas são permitidas: os **log-odds**.

**📐 Passo 1: de probabilidade para *odds*.** odds = p ÷ (1 − p).
p = 0,8 dá 4 ("4 para 1"); p = 0,5 dá 1; p = 0,2 dá 0,25.

**Passo 2: logaritmo.** ln 4 ≈ +1,39; ln 1 = 0; ln 0,25 ≈ −1,39.
A escala fica simétrica e sem limites.

**Passo 3: uma reta nos log-odds.** $z=a\,s+b$, com s = `nota_final`.
`a` é a **inclinação** e `b` é o **intercepto**. Para a epidemiologia: $e^a$
é a razão de chances (OR) por ponto de nota. Com a = 0,85, cada ponto multiplica
as **odds** de match por $e^{0{,}85}\approx2{,}34$; a probabilidade **não** é multiplicada por 2,34.

**Passo 4: voltar com a sigmoide.** $p=\sigma(z)=1/(1+e^{-z})$.
Exemplo **ilustrativo**, a = 0,85, b = −5,1 e nota 6,5:
z = 0,85 × 6,5 − 5,1 = 0,425; $e^{-0{,}425}\approx0{,}654$;
p = 1 ÷ 1,654 ≈ **0,60**.

**Bônus: nota de indiferença.** p = 0,5 quando z = 0, portanto
$s^\star=-b/a$ = 5,1 ÷ 0,85 = **6,0** (se a ≠ 0).

> Convenção: `a` = `slope`, `b` = `intercept`, a mesma da config
> (`sigmoid(a*logit(p_base)+b)`). Na biblioteca: `PlattModel(intercept, slope)`.
> Em R: `glm(TARGET ~ nota_final, family = binomial)`.

</details>

> 🎯 **Para levar.** Platt aprende uma tradução com dois controles, não uma regra de dividir a nota por dez.""",
    ),
    (
        "md",
        r"""### 9.2 Como a máquina escolhe `a` e `b`: a multa por surpresa

> 🧸 **Bem simples.** Imagine pagar uma multa quando seu palpite surpreende você.
> Dizer que tem certeza e errar custa muito mais do que admitir dúvida.

> 🧒 **Um pouco mais.** O código testa os controles e soma as multas de cada par.
> Se anunciou 90% e era a mesma pessoa, a surpresa é pequena; se anunciou 10%, é grande.

> 📖 **Palavra nova: log-loss.** É a multa por surpresa usada para escolher os controles.

**Exemplo pequeno, com valores só de exemplo.** Quando o par é a mesma pessoa:

| probabilidade anunciada | multa por surpresa |
|---|---|
| 0,9 | 0,105 |
| 0,5 | 0,693 |
| 0,1 | 2,303 |

Errar com confiança custa cerca de **22 vezes** mais do que acertar com confiança.

> 🧸 **Bem simples.** Um freio de bicicleta evita ganhar velocidade demais.
> Aqui também colocamos um freio, para a subida da curva não ficar exagerada.

> 🧒 **Um pouco mais.** O código cobra uma pequena multa extra pela inclinação.
> Não cobra pelo intercepto: ele precisa poder acompanhar quantos pares são a mesma pessoa.

> 📖 **Palavra nova: freio L2.** É a multa extra que desestimula inclinações exageradas.

<details>
<summary>📐 Para quem quer a matemática</summary>

**A multa de um par:** $\ell_i=-[y_i\ln p_i+(1-y_i)\ln(1-p_i)]$.
Para y = 1, ela é $-\ln p$; a tabela acima usa essa conta.

**A multa por exagero (L2):**
$\mathcal L(a,b)=\sum_i\ell_i+\tfrac12\lambda a^2$, com λ = 0,001.
Se os dados fossem perfeitamente separáveis, sem esse freio a melhor curva
viraria um degrau, com `a` indo ao infinito. O L2 **não** multa `b`, porque
o intercepto permite ajustar a prevalência (taxa-base de matches).

</details>

> 🎯 **Para levar.** O ajuste busca pouca surpresa, sem exagerar na inclinação.""",
    ),
    (
        "md",
        r"""### 9.3 O passo de Newton, sem medo

> 🧸 **Bem simples.** Imagine andar no escuro sentindo a inclinação do chão.
> Você procura o fundo de uma tigela, onde uma bola acabaria parando.

> 🧒 **Um pouco mais.** O fundo representa controles com multas pequenas.
> O código sente a descida e o quanto o chão encurva para escolher o próximo passo;
> repete até os passos ficarem minúsculos.

> 📖 **Palavra nova: passo de Newton.** É uma atualização que usa a inclinação
> e a curvatura para procurar um ajuste melhor.

**Exemplo pequeno, com valores só de exemplo.** Notas 2 e 4 são de pessoas distintas; 6 e 8, da mesma pessoa.
Começamos dizendo 50% para todos. Após um passo, a subida fica centrada perto da
nota 5, entre os dois grupos. A tabela mostra os primeiros passos e o último.

<details>
<summary>📐 Para quem quer a matemática</summary>

O **gradiente** g indica a descida e a **curvatura** H descreve o formato local.
Newton aproxima a perda por uma tigela quadrática e procura o fundo dela.

**📐 Em uma dimensão.** f(x) = (x − 3)², em x = 0:
g = 2(x − 3) = −6; h = 2. Pulo δ = g ÷ h = −3;
novo ponto 0 − (−3) = **3**, o fundo exato em um passo.

**Em duas dimensões, com uma tabela 2 × 2:**
1. $p_i=\sigma(b+a s_i)$ e $w_i=p_i(1-p_i)$;
2. $g_b=\sum(p_i-y_i)$; $g_a=\sum(p_i-y_i)s_i+\lambda a$;
3. Matriz de curvatura:
$$H=\begin{pmatrix}\sum w_i & \sum w_i s_i\\ \sum w_i s_i & \sum w_i s_i^2+\lambda\end{pmatrix}$$
4. resolver $H\delta=g$: dividir o gradiente pela curvatura, versão matriz;
5. $b\leftarrow b-\delta_b$, $a\leftarrow a-\delta_a$;
   parar quando o maior pulo absoluto for menor que 10⁻¹⁰.

**Exemplo de bolso.** s = (2, 4, 6, 8), y = (0, 0, 1, 1).
Partida: a = 0, b = log-odds de (2 + 0,5) ÷ (4 + 1) = 0,5, logo b = 0.
1. Todo p = 0,5 e todo w = 0,25.
2. g_b = 0; g_a = 0,5·2 + 0,5·4 − 0,5·6 − 0,5·8 = **−4**.
3. Matriz e gradiente:
$$H=\begin{pmatrix}1&5\\5&30{,}001\end{pmatrix},\qquad g=(0;\,-4)$$
4. Linha 1: $\delta_b+5\delta_a=0$, logo $\delta_b=-5\delta_a$.
   Linha 2:
$$5\delta_b+30{,}001\delta_a=-4\Rightarrow5{,}001\delta_a=-4\Rightarrow\delta_a\approx-0{,}800,\qquad\delta_b\approx+4{,}000$$
5. Novo ponto: **a ≈ 0,80, b ≈ −4,00**; indiferença ≈ **5,0**,
   entre o último não-match (4) e o primeiro match (6).

O exemplo é separável: a inclinação continua crescendo, e só o L2 a segura.
A tabela mostra os primeiros passos e o último; os parâmetros e a perda são
**após** o passo, enquanto g, H e δ são os que produziram esse passo.
O traço completo guarda também g, H e δ; a tabela visível mostra só os controles e a multa.
Depois repetimos no treino real e conferimos o resultado com `fit_platt`
(a biblioteca exige ao menos 10 linhas, por isso não a chamamos no exemplo de 4).

</details>

> 🎯 **Para levar.** Newton é o jeito de procurar os controles, não outra forma de prever.""",
    ),
    (
        "code",
        """\
from gzcmd_record_linkage.calibration import (
    PlattModel, compute_p_cal, fit_platt, fit_platt_from_df,
)

traco_bolso = nbh.platt_newton_trace([2, 4, 6, 8], [0, 0, 1, 1])
print("Newton no exemplo de bolso: olhe 'multa total'; quanto menor, melhor (primeiros passos + último).")
display(pd.concat([traco_bolso.head(3), traco_bolso.tail(1)])[
    ["passo", "intercepto", "inclinacao", "perda"]
].rename(columns={"intercepto": "intercepto b", "inclinacao": "inclinação a", "perda": "multa total"}).round(4))

train_idx, test_idx = synthetic_data.group_aware_split_indices(
    df, split_by="comprec", test_size=0.3, seed=SEED, group_stratify=True,
)
traco_treino = nbh.platt_newton_trace(
    df.iloc[train_idx]["nota_final"], df.iloc[train_idx]["TARGET"],
)
print("Newton no treino real: olhe 'multa total'; quanto menor, melhor (primeiros passos + último).")
display(pd.concat([traco_treino.head(3), traco_treino.tail(1)])[
    ["passo", "intercepto", "inclinacao", "perda"]
].rename(columns={"intercepto": "intercepto b", "inclinacao": "inclinação a", "perda": "multa total"}).round(4))
referencia_newton = fit_platt(
    df.iloc[train_idx]["nota_final"], df.iloc[train_idx]["TARGET"],
)
assert np.allclose(
    traco_treino.iloc[-1][["intercepto", "inclinacao"]].to_numpy(dtype=float),
    [referencia_newton.intercept, referencia_newton.slope], atol=1e-10, rtol=0,
)
print("Último passo igual a fit_platt: atol=1e-10, rtol=0.")""",
    ),
    (
        "md",
        """\
### 9.4 Rotas A e B: duas perguntas, não dois vencedores

> 🧸 **Bem simples.** Estudar com a prova na mão facilita tirar uma nota bonita.
> Fazer outra prova, sem ter visto as respostas, mostra melhor o que você aprendeu.

> 🧒 **Um pouco mais.** A rota A aprende e confere nas mesmas fichas, como refazer
> o exercício estudado. A rota B reserva fichas para uma prova nova.

> 📖 **Palavra nova: treino/teste.** Treino ensina os controles; teste confere em pares separados.
> 📖 **Palavra nova: vazamento.** É deixar informação da prova ajudar a preparar a resposta.

**Exemplo pequeno.** Se uma ficha-candidata aparece em dois pares, os dois vão juntos para treino ou para teste. A ficha de referência ainda pode aparecer dos dois lados.

A rota A reproduz `run_v3(p_cal="fit_platt")` para conferir o código na seção 12,
mas sua avaliação é otimista. Ela continua em `df["p_cal"]`; a B fica em `p_cal_test`.
As colunas ECE e Brier serão explicadas a seguir: menores são melhores.

Medido aqui: a rota A tem a = **0,9008**, intercepto b = **−5,4311** e s* = **6,0291**.
A rota B tem a = **0,9096**, intercepto b = **−5,5296** e s* = **6,0789**.
s* é a nota em que a probabilidade vira 50%, não uma medida de desempenho.

> 🎯 **Para levar.** Não eleja uma rota vencedora por essa tabela: elas fazem provas diferentes.""",
    ),
    (
        "code",
        """\
from nb_helpers import brier_score, expected_calibration_error

platt_insample = fit_platt_from_df(df)
df["p_cal"] = compute_p_cal(df, method="platt", model=platt_insample)
df_train = df.iloc[train_idx]
df_test = df.iloc[test_idx]
platt_holdout = fit_platt_from_df(df_train)
p_cal_test = compute_p_cal(df_test, method="platt", model=platt_holdout)
y_test = df_test["TARGET"].to_numpy(dtype=float)
p_true_test = p_true.to_numpy(dtype=float)[test_idx]
nota_all = df["nota_final"].to_numpy(dtype=float)
nota_test = df_test["nota_final"].to_numpy(dtype=float)
ece_holdout = expected_calibration_error(y_test, p_cal_test.to_numpy(), n_bins=10)
brier_holdout = brier_score(y_test, p_cal_test.to_numpy())

print(f"Treino: {len(df_train)} pares | Teste (held-out): {len(df_test)} pares")
comparacao_rotas = pd.DataFrame([
    {"rota": "A: mesmas linhas", "n_ajuste": len(df), "n_avaliacao": len(df),
     "a": platt_insample.slope, "b": platt_insample.intercept,
     "s*": -platt_insample.intercept / platt_insample.slope,
     "ECE": expected_calibration_error(df["TARGET"], df["p_cal"]),
     "Brier": brier_score(df["TARGET"], df["p_cal"])},
    {"rota": "B: teste novo", "n_ajuste": len(df_train), "n_avaliacao": len(df_test),
     "a": platt_holdout.slope, "b": platt_holdout.intercept,
     "s*": -platt_holdout.intercept / platt_holdout.slope,
     "ECE": ece_holdout, "Brier": brier_holdout},
]).set_index("rota")
for nome, modelo, esperados in [
    ("rota A", platt_insample, (0.9008, -5.4311, 6.0291)),
    ("rota B", platt_holdout, (0.9096, -5.5296, 6.0789)),
]:
    for parametro, valor, esperado in zip(
        ("inclinação a", "intercepto b", "s*"),
        (modelo.slope, modelo.intercept, -modelo.intercept / modelo.slope), esperados, strict=True,
    ):
        confere(f"{nome}: {parametro}", valor, esperado, 0.00005)
print("Olhe ECE e Brier: quanto menor, melhor; as rotas usam avaliações diferentes.")
print(comparacao_rotas.rename(columns={
    "n_ajuste": "pares no ajuste", "n_avaliacao": "pares na avaliação",
    "a": "inclinação a", "b": "intercepto b", "s*": "nota que vira 50% (s*)",
    "ECE": "erro de promessa (ECE)", "Brier": "erro par a par (Brier)",
}).round(4).to_string())""",
    ),
    (
        "md",
        r"""### 9.5 O boletim da calibração: promessa, erro e desenho

> 🧸 **Bem simples.** Separe os avisos de chuva em gavetas de promessas parecidas.
> Confira se choveu tanto quanto foi prometido em cada gaveta.

> 🧒 **Um pouco mais.** O código faz 10 gavetas de probabilidades (no exemplo, só 2 gavetas) e mede a diferença
> entre promessa média e frequência de mesma pessoa; gavetas cheias pesam mais.

> 📖 **Palavra nova: ECE.** É o erro médio de promessa entre essas gavetas.

> 🧸 **Bem simples.** Agora confira cada aviso sozinho, em vez de juntar em gavetas.
> Um erro grande pesa mais no boletim do que um erro pequeno.

> 🧒 **Um pouco mais.** O código compara cada probabilidade com a resposta conhecida,
> multiplica cada diferença por ela mesma e tira a média.

> 📖 **Palavra nova: Brier.** É essa medida de erro par a par, que pesa mais os erros grandes.

**Exemplo pequeno, com valores só de exemplo.** Cinco probabilidades: 0,1; 0,2; 0,4; 0,6; 0,9.
Só os três últimos pares são a mesma pessoa. Na gaveta baixa, a promessa média
é 0,233 e acontece 0,333; na alta, promete 0,75 e acontece 1,0.
O ECE dá **0,16**; o Brier, **0,116**. Nos dois, menor é melhor.

> 🧸 **Bem simples.** Imagine um desenho que põe a promessa ao lado do que aconteceu.
> Quando combinam, a bolinha fica sobre uma estrada reta que cruza o desenho.

> 🧒 **Um pouco mais.** Leia a probabilidade prometida na horizontal e a frequência
> observada na vertical. Cada bolinha representa uma gaveta, não uma pessoa.

> 📖 **Palavra nova: diagrama de confiabilidade.** É esse desenho de promessa contra realidade.

Na diagonal: promessa cumprida. Abaixo: aconteceu menos que o prometido.
Acima: aconteceu mais. Bolinha pequena tem poucos pares e pode oscilar muito.
A figura e a tabela usam só o teste da rota B.

> 🧸 **Bem simples.** Mesmo sabendo a probabilidade certa, uma turma pequena pode
> ter mais ou menos acertos por sorte. A promessa certa não zera o boletim.
>
> 🧒 **Um pouco mais.** Com cerca de 183 pares em 10 gavetas, até p* pode ter
> ECE entre ~0,035 e ~0,09 só por sorte. Essa faixa é um nível, não um corte para diferenças.
> Isso explica ECE perto de 0,075 junto de distância até p* perto de 0,0099.
> A célula mede o ECE da própria p* e simula seu piso com novos rótulos sorteados.

<details>
<summary>📐 Para quem quer a matemática</summary>

**Exemplo completo:** p = (0,1; 0,2; 0,4; 0,6; 0,9); y = (0; 0; 1; 1; 1).
1. Gaveta [0; 0,5): promessa 0,233; aconteceu 1/3 = 0,333; distância 0,100.
2. Gaveta [0,5; 1]: promessa 0,75; aconteceu 1,0; distância 0,25.
3. ECE = (3/5) × 0,100 + (2/5) × 0,25 = **0,16**.
4. Brier = (0,01 + 0,04 + 0,36 + 0,16 + 0,01) ÷ 5 = **0,116**.

$$\text{ECE}=\sum_g\frac{n_g}{N}|\bar y_g-\bar p_g|
\qquad\text{Brier}=\frac1N\sum_i(p_i-y_i)^2$$

Régua: quem sempre diz a prevalência f tem Brier = f(1 − f), ou 0,25 se f = 0,5.

Excesso de confiança aparece como bolinhas abaixo da
diagonal à direita **e** acima dela à esquerda: previsões extremas demais.
Bolinha pequena tem poucos pares e balança muito.
ECE é a média ponderada das distâncias verticais, o espírito de Hosmer-Lemeshow
num número só, não o próprio teste estatístico. A figura e a tabela usam o conjunto
de teste da rota B (`p_cal_test`).

</details>

> 🎯 **Para levar.** O boletim mede os erros; o desenho ajuda você a ver onde estão.""",
    ),
    (
        "code",
        """\
fig, gavetas = nbh.plot_reliability(
    y_test, p_cal_test.to_numpy(), n_bins=10, titulo="Platt supervisionado no teste",
)
plt.show()
print("Olhe promessa média e frequência observada: quanto mais próximas, melhor.")
display(gavetas.rename(columns={
    "bin": "gaveta", "lo": "de", "hi": "até", "n": "pares",
    "conf": "promessa média", "acc": "frequência observada", "gap": "distância",
}).round(3))
piso_ece = nbh.ece_noise_floor(p_true_test)
ece_verdade = nbh.expected_calibration_error(y_test, p_true_test)
print(f"Até a probabilidade verdadeira tem ECE ≈ {br(ece_verdade, 4)} neste teste.")
print(f"Piso simulado: média={br(piso_ece['mean'], 4)}; "
      f"intervalo de 95%={br(piso_ece['ci_low'], 4)} a {br(piso_ece['ci_high'], 4)}.")
confere("pares no teste", len(y_test), 183, 0)
confere("ECE Platt no teste", ece_holdout, 0.075, 0.0005)
confere("MAE Platt vs p* no teste", np.mean(np.abs(p_cal_test.to_numpy() - p_true_test)), 0.0099, 0.00005)""",
    ),
    (
        "md",
        """\
### 9.6 🎛️ W1: experimente os dois parafusos do Platt

> 🧸 **Bem simples.** Agora você pode girar os botões do brinquedo.
> Veja a estrada mudar, mas não confunda brincar com fazer uma prova nova.

> 🧒 **Um pouco mais.** Os controles começam no ajuste aprendido no treino.
> Mude a inclinação e depois só o intercepto: ambos mudam a nota em que a probabilidade vira 50% (s*).

> 📖 **Palavra nova: `p*`.** É a probabilidade verdadeira escondida, conhecida aqui porque os dados são inventados.

**Exemplo pequeno.** Dobre a inclinação e olhe a curva. Depois mova só `b`.
À esquerda estão sua curva, a aprendida (tracejada), p* (pontilhada) e a frequência
de mesma pessoa por faixa. À direita, seu boletim no teste: ECE, Brier e log-loss.
A tabela mostra também a nota que vira 50% (`s*`).

> 🎯 **Para levar.** Escolher controles olhando o teste é exploração, não uma prova nova.

Isso não altera `df["p_cal"]`; sem controles interativos, a visão estática permanece.""",
    ),
    (
        "code",
        """\
def _ui_platt(inclinacao, intercepto):
    fig, resumo = nbh.plot_platt_explorer(
        nota_test, y_test, platt_holdout, ref_model=platt_holdout,
        p_true=p_true_test, inclinacao=inclinacao, intercepto=intercepto,
    )
    plt.show()
    plt.close(fig)
    print("Olhe ECE, Brier e log-loss: quanto menor, melhor; a e b são os controles.")
    display(resumo)


if not nbh.interagir(_ui_platt, {
    "inclinacao": nbh.Deslizante(
        platt_holdout.slope, 0, max(3, 3 * platt_holdout.slope), 0.01, "Inclinação a",
    ),
    "intercepto": nbh.Deslizante(
        platt_holdout.intercept, platt_holdout.intercept - 8,
        platt_holdout.intercept + 8, 0.05, "Intercepto b",
    ),
}):
    _ui_platt(platt_holdout.slope, platt_holdout.intercept)""",
    ),
    (
        "md",
        r"""
### 9.7 A prova com a verdade escondida, e o seu limite

> 🧸 **Bem simples.** Inventamos um jogo e guardamos a resposta num envelope fechado.
> Podemos conferir se o jogador a encontra, mas já escolhemos um jogo que ele sabe jogar.

> 🧒 **Um pouco mais.** O código aprende só com notas e respostas do treino,
> sem olhar p*. Depois comparamos suas probabilidades com essa resposta escondida no teste.

> 📖 **Palavra nova: verdade sintética.** É a regra que escolhemos para inventar os dados, não uma descoberta sobre pessoas reais.

**Exemplo pequeno.** A verdade tem inclinação **0,85** e nota de 50% perto de
**6,04**. A rota B aprende inclinação 0,91, intercepto −5,53, nota de 50% ≈ 6,08.
Na figura, ficar perto da diagonal significa
ficar perto de p*, não apenas acertar respostas de sim ou não.

> ⚠️ **O jogo foi feito para caber nessa curva.** Recuperar p* acontece em boa
> parte porque o gerador usa exatamente a forma que Platt sabe aprender.
> Isso confere o código neste cenário, não prova que ele funciona em dados reais.
> p* nunca entra no ajuste; não estamos validando numa população externa.

<details>
<summary>📐 Para quem quer a matemática</summary>

O gerador usa $p^*=\sigma(0{,}85\,(s-s_0))$, uma sigmoide de uma reta na nota
(`synthetic_data.py`); s₀ é o s* da verdade. A resposta está na família do modelo:
é especificação correta por construção. A recuperação também indica suficiência
do treino com cerca de 70% dos pares nesta simulação. Em dados reais, a curva
pode ser assimétrica, ter platôs ou depender de outras variáveis.

</details>

> 🎯 **Para levar.** Encontrar a resposta do jogo inventado não garante acertar fora dele.""",
    ),
    (
        "code",
        """\
mae_vs_ptrue = float(np.mean(np.abs(p_cal_test.to_numpy() - p_true_test)))
confere("inclinação da verdade", meta["true_slope"], 0.85, 0.005)
confere("s* da verdade", -meta["true_intercept"] / meta["true_slope"], 6.04, 0.005)
confere("inclinação da rota B", platt_holdout.slope, 0.91, 0.005)
confere("intercepto da rota B", platt_holdout.intercept, -5.53, 0.005)
confere("s* da rota B", -platt_holdout.intercept / platt_holdout.slope, 6.08, 0.005)
confere("fração usada no treino", len(df_train) / len(df), 0.70, 0.005)

ordem = np.argsort(p_true_test)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot([0, 1], [0, 1], "--", color="gray", label="p_cal = p* (ideal)")
ax.scatter(p_true_test[ordem], p_cal_test.to_numpy()[ordem], s=14, alpha=0.5, color="#4C72B0", label="pares de teste")
ax.set_xlabel("probabilidade verdadeira escondida (p*)")
ax.set_ylabel("probabilidade aprendida no teste (p_cal)")
ax.set_title(f"O Platt recupera p*?  |  erro médio |p_cal − p*| = {br(mae_vs_ptrue, 4)}")
ax.legend(loc="upper left")
fig.tight_layout()
plt.show()

print(f"Erro absoluto médio entre p_cal (held-out) e a verdade p*: {br(mae_vs_ptrue, 4)}")
print("Erro pequeno confirma o ajuste neste jogo, cuja regra foi feita para caber no Platt.")""",
    ),
    (
        "md",
        """\
### 9.8 Autocalibração por âncoras: calibrar sem gabarito (demonstração)

> 🧸 **Bem simples.** Imagine ajustar uma balança sem pesos-padrão para conferir.
> Você usa um saco que acha pesado e outro que acha leve; se achar errado, a balança engana.

> 🧒 **Um pouco mais.** Sem gabarito, o código separa fichas muito parecidas e
> pouco parecidas. Atribui probabilidades presumidas a esses grupos e ajusta a curva por elas.

> 📖 **Palavra nova: âncora.** É um grupo com probabilidade presumida que serve de referência ao ajuste.

**Exemplo pequeno.** A configuração presume quase certeza nas notas altas
e quase nenhuma probabilidade nas baixas. São suposições, não respostas conferidas:

| item | valor na config |
|---|---|
| âncora positiva | notas a partir de 9, média-alvo 0,98, mínimo 50 pares |
| âncora negativa | notas abaixo de 5, média-alvo 0,001, mínimo 200 pares |
| método | por âncoras, uma curva por faixa de nota (só no papel: ver aviso) |
| limitador por janela | mudança máxima de 0,5 em `a` e 0,5 em `b` |

> ⚠️ **É uma demonstração, não o que a biblioteca faz.** A biblioteca **não implementa**
> `anchor_platt`. `run_v3` usa Platt com gabarito (supervisionado); `fit_platt`
> só aceita respostas 0/1, e o carregador de configuração nem lê `anchors`.
> A mini-implementação está apenas em `notebooks/nb_helpers.py`.

Entre 5 e 9 a curva vem só da forma em S, não de dados: é extrapolação;
se a curva real não for uma sigmoide da nota, o meio erra sem aviso.

<details>
<summary>📐 Para quem quer os limites da demonstração</summary>

A config calibra `logit(p_base)` de um modelo-base; aqui usamos `nota_final`.
O ajuste é global: só há âncoras nas bandas `low` e `high`, não nas cinzentas,
então não reproduzimos `by_band`. As janelas são simuladas.
Usamos todas as notas, inclusive as do teste, mas nenhum rótulo no ajuste:
não há vazamento de **rótulo**, nem validação numa população externa.

</details>

> 🎯 **Para levar.** Sem gabarito, a qualidade da tradução depende das suposições nas pontas.""",
    ),
    (
        "md",
        r"""#### 9.8.1 Quase certeza e pares suficientes para ajustar

> 🧸 **Bem simples.** Dizer "quase certeza" deixa espaço para uma surpresa.
> É diferente de prometer que nunca vai errar.

> 🧒 **Um pouco mais.** Em vez de respostas duras de sim ou não, o código dá
> às âncoras as probabilidades presumidas 0,98 e 0,001. Ajusta só com esses pares.

> 📖 **Palavra nova: rótulo suave.** É uma resposta de quase certeza, não de certeza absoluta.

> 🧸 **Bem simples.** Não ajuste uma balança com sacos de menos para conferir.
> Se faltar material, deixe o ponteiro como estava.

> 🧒 **Um pouco mais.** Exigimos pares suficientes em cada grupo antes de mudar
> a curva. Também equilibramos os grupos para o mais cheio não mandar sozinho.

> 📖 **Palavra nova: congelar.** É manter a curva anterior quando falta o mínimo de pares.

**Exemplo pequeno, medido com seed 42.** Há **113/242** âncoras positivas/negativas
em todas as linhas e **81/171** só no treino. Como exigimos 200 negativas,
o treino sozinho congelaria. Usamos todas as notas, inclusive do teste, **sem seus rótulos**.
O p* médio nas âncoras é **0,9501** (cerca de **0,950**) e **0,0889**:
as suposições 0,98 e 0,001 não combinam com este mundo inventado.

| resultado medido | com gabarito do treino (rota B) | com âncoras |
|---|---|---|
| inclinação a | 0,9096 | 1,7530 |
| intercepto b | −5,5296 | −12,7238 |
| nota que vira 50% (s*) | 6,0789 (cerca de 6,08) | 7,2581 (cerca de 7,26) |
| probabilidade do herói (se estiver no treino, valor supervisionado é in-sample) | 0,5946 | 0,2093 |
| erro de promessa (ECE) no teste | 0,0751 | 0,1558 |
| erro par a par (Brier) no teste | 0,1087 | 0,1540 |
| distância média até p* (MAE) no teste | 0,0099 | 0,1214 |

O herói tem p* **0,5974**; a rota A lhe dá **0,6045**.
As âncoras pioraram os erros aqui. Isso não é uma regra para todos os dados.
O resultado ruim das âncoras decorre da inclinação escolhida no gerador (a = 0,85);
com caudas mais íngremes, a probabilidade verdadeira média nas âncoras ficaria
mais perto de 0,98/0,001 e o erro das âncoras tenderia a cair (não testado aqui).

<details>
<summary>📐 Para quem quer a matemática</summary>

Só entram os pares-âncora; cada um recebe rótulo suave t: 0,98 nas positivas
e 0,001 nas negativas. O resto segue 9.2 e 9.3.
1. **Escolher:** nota ≥ 9,0 ou nota < 5,0. Pares entre 5 e 9 ficam de fora.
2. **Conferir o mínimo:** com menos de 50 positivas ou 200 negativas, mantemos
   a curva anterior ("congelar"). Abaixo medimos todas as linhas e só o treino.
3. **Equilibrar:** com $N=n_++n_-$, cada positiva recebe peso $N/(2n_+)$ e
   cada negativa, $N/(2n_-)$, para os dois grupos puxarem com a mesma força.
4. **Multa suave:** $\mathcal L(a,b)=-\sum_{i\in\text{âncoras}}w_i
   [t_i\ln p_i+(1-t_i)\ln(1-p_i)]+\tfrac12\lambda a^2$.
5. **Newton igual ao anterior:** troque y por t e pese cada termo dos dados por w_i.

**Por que suave?** Com 0 e 1, as âncoras seriam perfeitamente separáveis e,
sem L2, a curva viraria degrau. Com 0,98 e 0,001, o ajuste tem solução finita.
Como as notas variam dentro dos grupos, a curva não consegue dar exatamente
o alvo a cada linha; encontra o melhor compromisso da multa ponderada.

**Conta de guardanapo, ilustrativa (também inspira o chute inicial do Newton).**
Se todo um grupo estivesse na sua nota média, teríamos duas equações:
- logit(0,98) = ln 49 ≈ **3,89** = a·s̄₊ + b;
- logit(0,001) ≈ **−6,91** = a·s̄₋ + b.

Com médias **ilustrativas** 9,5 e 3,0: a = 10,80 ÷ 6,5 ≈ **1,66**;
b = 3,89 − 1,66 × 9,5 ≈ **−11,9**; s* ≈ **7,2**.
Nesse exemplo de guardanapo, nota 6,5 daria p ≈ 0,25. **Não são os resultados
medidos:** a tabela seguinte traz o chute com as médias reais, o ajuste de
Newton, a verdade e as probabilidades do herói.

**Por que podem discordar?** Âncoras só são boas se os alvos forem verdadeiros.
Os alvos da tese que inspirou a config vieram de dados **reais**; neste mundo
sintético, o código mede o p* médio nas pontas e o compara a 0,98 e 0,001.
Âncora errada não avisa: calibra com confiança para o lugar errado.

</details>

> 🎯 **Para levar.** Quase certeza ainda é uma suposição: confira se as âncoras merecem confiança.""",
    ),
    (
        "code",
        """\
for campo, esperado in [
    ("pos_cut", 9), ("pos_target", 0.98), ("pos_min_count", 50),
    ("neg_cut", 5), ("neg_target", 0.001), ("neg_min_count", 200),
    ("max_abs_delta_a", 0.5), ("max_abs_delta_b", 0.5),
]:
    confere(f"âncoras: {campo}", getattr(ANCORAS, campo), esperado, 0)
assert ANCORAS.pos_op == ">=" and ANCORAS.neg_op == "<"
pos_all, neg_all = nbh.anchor_masks(nota_all, ANCORAS)
pos_train, neg_train = nbh.anchor_masks(df_train["nota_final"], ANCORAS)
contagem_ancoras = pd.DataFrame({
    "região": ["positiva", "negativa"],
    "todas as linhas": [int(pos_all.sum()), int(neg_all.sum())],
    "só treino": [int(pos_train.sum()), int(neg_train.sum())],
    "mínimo exigido": [ANCORAS.pos_min_count, ANCORAS.neg_min_count],
    "alvo configurado": [ANCORAS.pos_target, ANCORAS.neg_target],
    "p* médio verdadeiro": [float(p_true.to_numpy()[pos_all].mean()),
                            float(p_true.to_numpy()[neg_all].mean())],
}).set_index("região")
print("Compare contagens com mínimo exigido e p* médio verdadeiro com alvo configurado.")
print(contagem_ancoras.round(4).to_string())
for rotulo, valor, esperado in [
    ("âncoras positivas em todas", pos_all.sum(), 113),
    ("âncoras negativas em todas", neg_all.sum(), 242),
    ("âncoras positivas no treino", pos_train.sum(), 81),
    ("âncoras negativas no treino", neg_train.sum(), 171),
]:
    confere(rotulo, valor, esperado, 0)
confere("p* médio das âncoras positivas", p_true.to_numpy()[pos_all].mean(), 0.9501, 0.00005)
confere("p* médio das âncoras positivas (arredondado)", p_true.to_numpy()[pos_all].mean(), 0.950, 0.0005)
confere("p* médio das âncoras negativas", p_true.to_numpy()[neg_all].mean(), 0.0889, 0.00005)

chute_ancoras = nbh.anchor_two_point_guess(nota_all, ANCORAS)
ajuste_ancoras = nbh.fit_anchor_platt(nota_all, ANCORAS, fallback=platt_holdout)
platt_ancorado = ajuste_ancoras.model
assert platt_ancorado is not None
print(f"Ajuste ancorado: {ajuste_ancoras.status}; Newton: {ajuste_ancoras.n_iter} passos")
parametros_ancoras = pd.DataFrame([
    {"modelo": nome, "a": modelo.slope, "b": modelo.intercept,
     "s*": -modelo.intercept / modelo.slope}
    for nome, modelo in [
        ("Supervisionado (treino)", platt_holdout),
        ("Chute (médias reais)", chute_ancoras),
        ("Ancorado (todas)", platt_ancorado),
        ("Verdade sintética", PlattModel(
            intercept=meta["true_intercept"], slope=meta["true_slope"],
        )),
    ]
]).set_index("modelo")
print("Olhe a (inclinação), b (intercepto) e s* (nota que vira 50%): não são notas de desempenho.")
print(parametros_ancoras.round(4).to_string())
fig, comparacao_ancoras = nbh.plot_anchor_comparison(
    nota_all, nota_test, y_test, platt_holdout, ajuste_ancoras, ANCORAS,
    p_true_all=p_true.to_numpy(), p_true_test=p_true_test,
)
plt.show()
print("Boletim no mesmo teste: ECE, Brier e MAE vs p* são erros; quanto menor, melhor.")
print(comparacao_ancoras[["ECE", "Brier", "MAE vs p*"]].round(4).to_string())

hero_nota = float(df.loc[hero_idx, "nota_final"])
hero_sup = float(compute_p_cal(df.loc[[hero_idx]], method="platt", model=platt_holdout).iloc[0])
hero_anc = float(compute_p_cal(df.loc[[hero_idx]], method="platt", model=platt_ancorado).iloc[0])
hero_truth = float(p_true.loc[hero_idx])
hero_no_treino = hero_idx in df_train.index
rotulo_hero_sup = "supervisionado" + (" (herói está no treino)" if hero_no_treino else " (herói no teste)")
display(pd.DataFrame({
    rotulo_hero_sup: [hero_sup], "ancorado": [hero_anc], "p*": [hero_truth],
}, index=["probabilidade do herói"]))
for nome, modelo, esperados in [
    ("rota B", platt_holdout, (0.9096, -5.5296, 6.0789)),
    ("âncoras", platt_ancorado, (1.7530, -12.7238, 7.2581)),
]:
    for parametro, valor, esperado in zip(
        ("inclinação a", "intercepto b", "s*"),
        (modelo.slope, modelo.intercept, -modelo.intercept / modelo.slope), esperados, strict=True,
    ):
        confere(f"{nome}: {parametro}", valor, esperado, 0.00005)
for modelo, esperados in [
    ("Supervisionado", (0.0751, 0.1087, 0.0099)),
    ("Ancorado", (0.1558, 0.1540, 0.1214)),
]:
    for metrica, esperado in zip(("ECE", "Brier", "MAE vs p*"), esperados, strict=True):
        confere(f"{modelo}: {metrica}", comparacao_ancoras.loc[modelo, metrica], esperado, 0.00005)
for rotulo, valor, esperado in [
    ("herói supervisionado", hero_sup, 0.5946), ("herói ancorado", hero_anc, 0.2093),
    ("herói p*", hero_truth, 0.5974), ("herói rota A", df.loc[hero_idx, "p_cal"], 0.6045),
]:
    confere(rotulo, valor, esperado, 0.00005)
PASSAPORTE["9. Calibração"] = {
    "p_cal (rota A, pipeline)": float(df.loc[hero_idx, "p_cal"]),
    "p_cal supervisionado (treino)": hero_sup, "p_cal ancorado (demo)": hero_anc,
}
print(f"Herói, nota {br(hero_nota)}: supervisionado={br(hero_sup, 4)}, "
      f"ancorado={br(hero_anc, 4)}, p*={br(hero_truth, 4)}")""",
    ),
    (
        "md",
        """\
#### 🎛️ W3: e se as âncoras estiverem erradas?

> 🧸 **Bem simples.** Troque os sacos usados para conferir a balança.
> Veja como uma referência errada pode entortar a leitura inteira.

> 🧒 **Um pouco mais.** Mova os cortes e as probabilidades presumidas (alvos).
> As áreas sombreadas são os grupos; os quadrados mostram o que você presumiu.

**Exemplo pequeno.** Aproxime os alvos do p* médio verdadeiro escrito na figura.
O boletim melhora? Desligue "Equilibrar grupos": quem passa a puxar mais?
Exigir mínimos mostra a mensagem "congelado" quando faltam pares;
a curva exibida volta ao Platt com gabarito, não é um ajuste por âncoras.
Cortes cruzados mostram uma mensagem. A visão estática usa `ANCORAS`.

> 🎯 **Para levar.** Ajustar olhando o teste ajuda a explorar, não a validar.""",
    ),
    (
        "code",
        """\
from dataclasses import replace


def _ui_ancoras(corte_pos, corte_neg, alvo_pos, alvo_neg, balancear, exigir_minimo):
    if corte_neg >= corte_pos:
        print("O corte negativo deve ser menor que o positivo.")
        return
    if alvo_neg >= alvo_pos:
        print("O alvo negativo deve ser menor que o positivo.")
        return
    spec = replace(
        ANCORAS, pos_cut=corte_pos, neg_cut=corte_neg,
        pos_target=alvo_pos, neg_target=alvo_neg,
    )
    ajuste = nbh.fit_anchor_platt(
        nota_all, spec, balance=balancear, enforce_min_count=exigir_minimo,
        fallback=platt_holdout,
    )
    fig, resumo = nbh.plot_anchor_comparison(
        nota_all, nota_test, y_test, platt_holdout, ajuste, spec,
        p_true_all=p_true.to_numpy(), p_true_test=p_true_test,
    )
    plt.show()
    plt.close(fig)
    print("Olhe ECE, Brier e MAE vs p*: quanto menor, melhor (mas veja o piso de ruído); status informa o ajuste.")
    display(resumo)


if not nbh.interagir(_ui_ancoras, {
    "corte_pos": nbh.Deslizante(ANCORAS.pos_cut, 7, 10, 0.1, "Corte positivo"),
    "corte_neg": nbh.Deslizante(ANCORAS.neg_cut, 1, 7, 0.1, "Corte negativo"),
    "alvo_pos": nbh.Deslizante(ANCORAS.pos_target, 0.5, 0.999, 0.001, "Alvo positivo"),
    "alvo_neg": nbh.Deslizante(
        ANCORAS.neg_target, -4, -0.3, 0.1, "Alvo negativo", log=True,
    ),
    "balancear": nbh.Caixa(True, "Equilibrar grupos"),
    "exigir_minimo": nbh.Caixa(True, "Exigir mínimos"),
}):
    _ui_ancoras(ANCORAS.pos_cut, ANCORAS.neg_cut, ANCORAS.pos_target,
                ANCORAS.neg_target, True, True)""",
    ),
    (
        "md",
        r"""#### 🔒 Limitador por janela: o volante que não dá tranco

> 🧸 **Bem simples.** Um volante que vira de repente pode dar um tranco.
> Limitar o movimento ajuda, mas não garante que o carro siga o caminho desejado.

> 🧒 **Um pouco mais.** A cada período tentamos ajustar de novo. O código limita
> quanto cada controle pode mudar; se faltarem âncoras, mantém a curva já usada.

> 📖 **Palavra nova: limitador por janela.** É a regra que limita a mudança de cada controle entre períodos.

**Exemplo pequeno, ilustrativo.** Os controles começam em 0,85 e −5,1;
o ajuste pede 1,66 e −11,9. Cada controle pode andar só 0,5 e fica em 1,35 e −5,6.
A nota que vira 50% sai de 6,0 para **4,1**, embora o destino pedido fosse **7,2**.

> 🔎 **Efeito colateral.** Limitar os controles separadamente não limita com
> segurança a nota que vira 50%: ela pode até sair do caminho entre as duas curvas.

Limitar os parâmetros soltos é a regra da config, não a melhor escolha;
uma alternativa é limitar a nota de 50% (s*) e a inclinação, ou a mudança
da probabilidade numa grade de notas.

Na simulação, "no mês passado tínhamos gabarito; agora não temos".
Começamos no Platt do treino e criamos seis períodos só em memória.
O terceiro tem 300 pares aleatórios, sem os cenários narrativos: faltam âncoras
e a curva congela. A tabela confere isso.

<details>
<summary>📐 Para quem quer a matemática</summary>

A config limita a mudança a 0,5 em `a` e 0,5 em `b` por janela.

$$a_{\text{usado}}=a_{\text{ant}}+
\operatorname{clip}(a_{\text{novo}}-a_{\text{ant}},-0{,}5,+0{,}5)$$
(idem para b).

**Exemplo ilustrativo**, de (0,85; −5,1) para (1,66; −11,9):
1. Δa = 0,81 → +0,5 → a = 1,35.
2. Δb = −6,8 → −0,5 → b = −5,6.
3. s* = 5,6 ÷ 1,35 ≈ **4,1**.

</details>

> 🎯 **Para levar.** Suavizar os botões evita alguns trancos, mas também pode desviar a curva.""",
    ),
    (
        "code",
        """\
janelas_ancoras = [
    synthetic_data.generate_comparador(seed=SEED+k, n_pairs=n)
    .frame["nota final"].to_numpy()
    for k, n in enumerate([600, 600, 300, 600, 600, 600], 1)
]
historico_ancoras = nbh.simulate_anchor_windows(
    janelas_ancoras, ANCORAS, init=platt_holdout,
)
print("Compare s* bruto e usado (nota que vira 50%); status mostra quando a curva ficou congelada.")
print(historico_ancoras.assign(**{
    "s* bruto": lambda d: -d["b_bruto"] / d["a_bruto"],
}).rename(columns={
    "travou_a": "limitou_a", "travou_b": "limitou_b",
    "s_estrela": "s* usado", "n_pos": "âncoras +", "n_neg": "âncoras −",
}).round(4).to_string(index=False))
assert historico_ancoras.iloc[2]["status"] == "congelado"
assert np.allclose(
    historico_ancoras.iloc[2][["a_usado", "b_usado"]].to_numpy(dtype=float),
    historico_ancoras.iloc[1][["a_usado", "b_usado"]].to_numpy(dtype=float),
    atol=0, rtol=0,
)
print("Janela de 300 pares aleatórios: congelada; curva anterior preservada.")
confere("pares aleatórios na terceira janela", len(janelas_ancoras[2]), 300, 0)
confere("períodos simulados", len(janelas_ancoras), 6, 0)
fig, resumo_janelas = nbh.plot_anchor_windows(historico_ancoras)
plt.show()""",
    ),
    (
        "md",
        """\
### 9.9 Configuração não é implementação

> 🧸 **Bem simples.** Uma receita escrita no papel não é um bolo pronto.
> Precisamos conferir quais passos a cozinha realmente executa.

> 🧒 **Um pouco mais.** O arquivo YAML descreve âncoras por faixa, mas `run_v3`
> executa Platt com gabarito, numa curva única. A demonstração não muda `df["p_cal"]`.

> 📖 **Palavra nova: configuração.** É o arquivo de instruções, que só tem efeito quando o programa lê e executa cada opção.

**Exemplo pequeno.** Escrever `anchor_platt` no arquivo não faz a biblioteca
aprender por âncoras: essa parte existe apenas na demonstração do notebook.
Platt olha só a nota final, com resultado repetível para conferir a rota A.
O outro caminho, `GZCMDClassifier.predict_proba`, usa mais pistas e aparece
com Random Forest e XGBoost na §13.7, no mesmo teste separado.

> 🎯 **Para levar.** Qualquer caminho que anuncia probabilidades precisa ser conferido fora do treino.

A seguir: regras de segurança (*guardrails*), antes de comparar custos.""",
    ),
]


# ===========================================================================
# FASE 2.5: Guardrails determinísticos
# ===========================================================================
FASE_2_5: list[tuple[str, str]] = [
    (
        "md",
        """\
## 10. Guardrails

> 🧒 **O porteiro com a palavra final.** A calculadora faz a conta para todos;
> depois o porteiro manda em alguns casos. Quando ele pede segunda opinião
> e não há vaga, vale a conta.

**Quatro regras, na ordem do código:** a primeira que acende fica com o par.
1. `temporal_filter` → `ALWAYS_NONMATCH`: óbito da referência anterior ao
   diagnóstico do candidato menos a tolerância de `temporal_days` (180 dias).
   O descarte é definitivo: diagnósticos até 180 dias após o óbito passam,
   mas investigações reais tardias seriam perdidas. A regra deve ser validada
   por doença, não é uma impossibilidade universal.
2. `nota_final_low` → `ALWAYS_NONMATCH`: nota **< 3,0**.
3. `homonimia_risk` → `FORCE_REVIEW`: nota ≥ 7,0, os cinco subscores de nascimento
   zerados, diferença entre os **anos** de nascimento > 5 e todos os subscores de
   endereço zerados. A regra **não olha o nome**: a nota alta já faz esse papel.
4. `nota_final_high` → `ALWAYS_MATCH`: nota ≥ 10,0, quantidade do nome ≥ 0,95,
   primeiro/último fragmentos iguais, data de nascimento idêntica e município igual.

| Cenário | Guardrail esperado | Motivo esperado |
|---|---|---|
| `match_obvio` | `ALWAYS_MATCH` | `nota_final_high` |
| `nonmatch_obvio` | `ALWAYS_NONMATCH` | `nota_final_low` |
| `homonimo` | `FORCE_REVIEW` | `homonimia_risk` |
| `obito_antes_diag` | `ALWAYS_NONMATCH` | `temporal_filter` |
| Demais cenários (incluindo o herói e `mae_ausente`) | nenhum | nenhum |

> ⚠️ **Config × código.** A YAML lista `always_match: nota_final >= 9.0` e
> `always_nonmatch: nota_final <= 3.0`, mas o código usa os **`params`**:
> `ALWAYS_MATCH` exige nota ≥ **10,0** mais as pistas fortes acima;
> `ALWAYS_NONMATCH` usa nota **< 3,0**. As regras `risky_if`
> (`grey_mother_missing`, `low_name_and_low_dob`) **não estão implementadas**.
> No controle 10.1, baixe o corte para 9,0 e veja quantos pares mudariam.

Aplicamos `apply_guardrails` com os parâmetros de `cfg`, exatamente como `run_v3`.
A função devolve regra e motivo sem mudar `df`; nós acrescentamos as duas colunas.
A tabela seguinte confere **cada cenário**, em vez de escolher exemplos favoráveis.""",
    ),
    (
        "code",
        """\
from gzcmd_record_linkage.guardrails import apply_guardrails

for campo, esperado in [
    ("temporal_days", 180), ("nota_always_nonmatch", 3.0),
    ("homonimia_min_nota", 7.0), ("homonimia_year_gap", 5.0),
    ("nota_always_match", 10.0),
]:
    confere(f"guardrail: {campo}", getattr(cfg.guardrails, campo), esperado, 0)
# O corte do nome é uma constante do código, não um parâmetro de cfg.
sonda_nome = pd.DataFrame({
    "nota_final": [cfg.guardrails.nota_always_match] * 2,
    "NOME qtd frag iguais": [np.nextafter(0.95, 0), 0.95],
    "NOME prim frag igual": [1, 1], "NOME ult frag igual": [1, 1],
    "DTNASC dt iguais": [1, 1], "CODMUNRES local igual": [1, 1],
})
assert apply_guardrails(sonda_nome).guardrail.fillna("sem regra").tolist() == [
    "sem regra", "ALWAYS_MATCH",
], "O texto cita corte do nome ≥ 0,95; confira a constante no código."
gout = apply_guardrails(
    df,
    temporal_days=cfg.guardrails.temporal_days,
    nota_always_match=cfg.guardrails.nota_always_match,
    nota_always_nonmatch=cfg.guardrails.nota_always_nonmatch,
    homonimia_min_nota=cfg.guardrails.homonimia_min_nota,
    homonimia_year_gap=cfg.guardrails.homonimia_year_gap,
)
df["guardrail"] = gout.guardrail
df["guardrail_reason"] = gout.reason

print("Contagem de guardrails:")
display(df["guardrail"].fillna("sem regra").value_counts())
esperados_guardrail = {
    "match_obvio": ("ALWAYS_MATCH", "nota_final_high"),
    "nonmatch_obvio": ("ALWAYS_NONMATCH", "nota_final_low"),
    "homonimo": ("FORCE_REVIEW", "homonimia_risk"),
    "obito_antes_diag": ("ALWAYS_NONMATCH", "temporal_filter"),
}
checagens_cenarios = []
for cenario, posicao in meta["scenarios"].items():
    linha = df.iloc[posicao]
    esperado, motivo = esperados_guardrail.get(cenario, ("sem regra", "sem motivo"))
    regra_real = "sem regra" if pd.isna(linha["guardrail"]) else linha["guardrail"]
    motivo_real = ("sem motivo" if pd.isna(linha["guardrail_reason"])
                   else linha["guardrail_reason"])
    cenario_confere = regra_real == esperado and motivo_real == motivo
    checagens_cenarios.append({
        "cenário": cenario, "nota_final": linha["nota_final"],
        "esperado": esperado, "observado": regra_real,
        "motivo": motivo_real, "confere?": cenario_confere,
    })
checagem_guardrails = pd.DataFrame(checagens_cenarios).set_index("cenário")
display(checagem_guardrails)
assert checagem_guardrails["confere?"].all()
PASSAPORTE["10. Guardrails"] = df.loc[hero_idx, ["guardrail", "guardrail_reason"]].to_dict()""",
    ),
    (
        "md",
        """\
### 10.1 🎛️ W4: bandas e regras de segurança

Mova as cinco fronteiras da nota e os cortes dos guardrails. As linhas tracejadas
marcam as bandas originais; o mapa conta as transições entre bandas; as barras
comparam regras. O resumo conta mudanças de banda, regra e **ação** após nova
triagem em vigilância. Fronteiras precisam continuar em ordem crescente.
Se forem cruzadas, o controle mostra uma mensagem em vez de calcular.

Teste **`nota_always_match = 9.0`**: basta baixar a nota para todos virarem MATCH,
ou ainda são necessárias as pistas fortes? Mudar só uma banda altera a probabilidade
calibrada? Não: muda o perfil de erro usado para estimar a revisão.
O controle trabalha em cópias e não altera a rota A que vamos reconciliar.""",
    ),
    (
        "code",
        """\
fronteiras_w4 = [b.max for b in cfg.bands.definitions[:-1]]
defaults_w4 = {
    **{f"fronteira_{i}": valor for i, valor in enumerate(fronteiras_w4, 1)},
    "nota_always_match": cfg.guardrails.nota_always_match,
    "nota_always_nonmatch": cfg.guardrails.nota_always_nonmatch,
    "homonimia_min_nota": cfg.guardrails.homonimia_min_nota,
    "homonimia_year_gap": cfg.guardrails.homonimia_year_gap,
}


def _ui_bandas(fronteira_1, fronteira_2, fronteira_3, fronteira_4, fronteira_5,
               **guard_params):
    novas = [fronteira_1, fronteira_2, fronteira_3, fronteira_4, fronteira_5]
    if any(a >= b for a, b in zip(novas, novas[1:])):
        print("Mantenha as fronteiras em ordem estritamente crescente.")
        return
    fig, resumo = nbh.plot_bands_guardrails(
        df, cfg, edges=novas, guard_params=guard_params, mode="vigilancia",
    )
    plt.show()
    plt.close(fig)
    display(resumo)


if not nbh.interagir(_ui_bandas, {
    **{f"fronteira_{i}": nbh.Deslizante(valor, 0, 10, 0.1, f"Fronteira {i}")
       for i, valor in enumerate(fronteiras_w4, 1)},
    "nota_always_match": nbh.Deslizante(
        defaults_w4["nota_always_match"], 8, 10.5, 0.1, "Nota sempre match",
    ),
    "nota_always_nonmatch": nbh.Deslizante(
        defaults_w4["nota_always_nonmatch"], 0, 6, 0.1, "Nota não-match",
    ),
    "homonimia_min_nota": nbh.Deslizante(
        defaults_w4["homonimia_min_nota"], 5, 10, 0.1, "Nota homonímia",
    ),
    "homonimia_year_gap": nbh.Deslizante(
        int(defaults_w4["homonimia_year_gap"]), 0, 20, 1, "Anos homonímia", inteiro=True,
    ),
}):
    _ui_bandas(**defaults_w4)""",
    ),
]


# ---------------------------------------------------------------------------
# Montagem do notebook
# ---------------------------------------------------------------------------
# A ordem das fases reflete a construção incremental do plano (Wave 2).

FASE_2_6: list[tuple[str, str]] = [
    (
        "md",
        """\
## 11. Política de decisão (triagem)

> 🧒 **Como para uma criança de 11 anos.** Levar guarda-chuva custa pouco
> (carregar peso). Não levar e chover custa muito (molhar a mochila). Com 30%
> de probabilidade de chuva, vale levar? Depende de **quanto** cada erro dói.
> A triagem faz essa conta para cada par, com três opções: `MATCH`, `NONMATCH`
> ou pedir uma segunda opinião (`LLM_REVIEW`), que custa um pouco e ainda pode errar.

Primeiro calculamos o preço médio de cada opção; depois entram as regras de
segurança, as travas (ver 11.3) e as vagas disponíveis para revisão. Assim separamos
**a melhor escolha automática** da **ação operacional final**.""",
    ),
    (
        "md",
        r"""### 11.1 📐 As três perdas: probabilidade de errar × preço do erro

**Dizer MATCH:** só pago se não era match, com probabilidade (1 − p).
**Dizer NONMATCH:** só pago se era match, com probabilidade p.
**Pedir revisão:** pago uma taxa mais os erros que sobram depois da segunda opinião.
Aqui p é `p_cal`; $e_{fp}$ e $e_{fn}$ são taxas residuais do revisor por banda,
não as taxas de erro das decisões automáticas.

**Exemplo com p = 0,60**, banda `grey_mid` ($e_{fp}=0,06$, $e_{fn}=0,10$),
taxa de revisão 0,25. Os valores vêm da configuração atual; o código imprime
depois o p e a banda **reais do herói**, que podem diferir desta ilustração.

| Conta | Vigilância (FP 10 / FN 50) | Confirmação (FP 100 / FN 20) |
|---|---|---|
| MATCH | 0,40 × 10 = **4,00** | 0,40 × 100 = 40,00 |
| NONMATCH | 0,60 × 50 = 30,00 | 0,60 × 20 = **12,00** |
| Melhor automática | MATCH, 4,00 | NONMATCH, 12,00 |
| Revisão | 0,25 + 0,24 + 3,00 = **3,49** | 0,25 + 2,40 + 1,20 = **3,85** |
| EVR (ver 11.2) = base − revisão | **+0,51** | **+8,15** |

Com a mesma probabilidade, a melhor escolha **automática inverte** entre os modos;
nos dois, a revisão é mais barata. Não confunda essa escolha base com `action`:
se houver uma vaga de revisão, a ação pode ser `LLM_REVIEW` nos dois modos.

**Agora a receita formal:**
$$\ell_{match}=(1-p)c_{fp}\qquad\ell_{non}=p c_{fn}$$
$$\ell_{llm}=c_{llm}+(1-p)e_{fp}c_{fp}+p e_{fn}c_{fn}$$
A menor das duas primeiras é `base_loss`; sua opção é `base_choice`
(no empate, MATCH). Custos representam preferências institucionais, não
precisam estar em reais, mas devem usar a mesma unidade.

FP = 10, FN = 50 e revisão = 0,25 são valores da config, sem fonte empírica;
em uso real seriam definidos com a equipe de vigilância (quantos óbitos perdidos
equivalem a uma fusão errada). O W2 mostra a sensibilidade.""",
    ),
    (
        "md",
        r"""### 11.2 📐 Ponto de virada, janela de revisão e EVR

**🧒 A gangorra.** Conforme p cresce, "dizer MATCH" fica mais barato e
"dizer NONMATCH" mais caro. Onde os dois preços empatam?

**Passo a passo:**
1. $(1-p)c_{fp}=p c_{fn}$.
2. $c_{fp}-p c_{fp}=p c_{fn}$.
3. $c_{fp}=p(c_{fp}+c_{fn})$.
4. $p_{\text{virada}}=c_{fp}/(c_{fp}+c_{fn})$.

(Não confunda com p*, a probabilidade verdadeira da seção 9.)

Vigilância: 10 ÷ 60 ≈ **0,167**. Perder um match dói cinco vezes mais que aceitar
um falso match, então basta cerca de 17% para a escolha base ser MATCH.
Confirmação: 100 ÷ 120 ≈ **0,833**. O 0,5 só aparece com custos iguais.
Tudo isso depende de p **calibrada**: é aqui que a seção 9 paga a conta.

**A janela da revisão.** Para revisar ser mais barato que NONMATCH, rearranjamos
$\ell_{llm}<p c_{fn}$: $c_{llm}+e_{fp}c_{fp}<
p[c_{fn}(1-e_{fn})+e_{fp}c_{fp}]$. Isso dá o limite inferior.
Para vencer MATCH, rearranjamos $\ell_{llm}<(1-p)c_{fp}$:
$p[c_{fp}(1-e_{fp})+e_{fn}c_{fn}]<c_{fp}(1-e_{fp})-c_{llm}$.
Isso dá o superior:

$$p_{inf}=\frac{c_{llm}+e_{fp}c_{fp}}{c_{fn}(1-e_{fn})+e_{fp}c_{fp}}
\qquad p_{sup}=\frac{c_{fp}(1-e_{fp})-c_{llm}}{c_{fp}(1-e_{fp})+e_{fn}c_{fn}}$$

Em `grey_mid`, vigilância: aproximadamente **2%–64%** (0,019–0,635).
Em `high`, confirmação: aproximadamente **6%–99%**.
Em vigilância, dizer MATCH sozinho já fica mais barato que revisar acima de
~0,53 a ~0,85, conforme a banda (0,635 em `grey_mid`); em confirmação, dizer
NONMATCH sozinho fica mais barato abaixo de ~0,32 a ~0,38 nas bandas baixas.
A janela é larga e assimétrica, não restrita às pontas.
A revisão vence estritamente dentro do intervalo; a função auxiliar o intersecta com [0,1]
e marca intervalos vazios como NaN. São comparações de perdas para uma banda fixa,
não cortes na nota nem substitutos das travas (ver 11.3).

**EVR**, valor esperado da revisão (*expected value of review*), é quanto economizo:
$$EVR=\min(\ell_{match},\ell_{non})-\ell_{llm}$$
EVR > 0 significa economia esperada. A decisão automática pode ficar mais barata
longe das pontas, dependendo da banda. A tabela abaixo calcula os três pontos por
modo e banda **direto de `cfg`**, sem copiar custos no código.""",
    ),
    (
        "code",
        """\
limiares_decisao = nbh.thresholds_table(cfg)
display(limiares_decisao.round(4))
limiares_vig = limiares_decisao.loc[limiares_decisao["modo"].eq("vigilancia")]
confere("vigilância: menor llm_sup", limiares_vig["llm_sup"].min(), 0.530, 0.0005)
confere("vigilância: maior llm_sup", limiares_vig["llm_sup"].max(), 0.846, 0.0005)
for banda, esperado in [("low", 0.380), ("grey_low", 0.322)]:
    inferior = limiares_decisao.loc[
        limiares_decisao["modo"].eq("confirmacao") & limiares_decisao["banda"].eq(banda),
        "llm_inf",
    ].iloc[0]
    confere(f"confirmação/{banda}: llm_inf", inferior, esperado, 0.0005)
for modo, corte_match, corte_nonmatch in [
    ("vigilancia", 0.85, 0.15), ("confirmacao", 0.95, 0.10),
]:
    politica = cfg.decision_policy.modes[modo]
    confere(f"{modo}: trava MATCH", politica.min_auto_match_threshold, corte_match, 0)
    confere(f"{modo}: trava NONMATCH", politica.max_auto_nonmatch_threshold, corte_nonmatch, 0)
for taxa, esperado in [("e_fp", 0.06), ("e_fn", 0.10)]:
    confere(f"grey_mid: {taxa}", cfg.llm_review.error_rates_by_band["grey_mid"][taxa], esperado, 0)
politica_vig = cfg.decision_policy.modes["vigilancia"]
for rotulo, valor, esperado, tol in [
    ("custo FP da config", politica_vig.false_positive_cost, 10, 0),
    ("custo FN da config", politica_vig.false_negative_cost, 50, 0),
    ("custo revisão da config", politica_vig.llm_review_cost, 0.25, 0.005),
]:
    confere(rotulo, valor, esperado, tol)
for modo, banda, virada, inferior, superior, tol in [
    ("vigilancia", "grey_mid", 0.167, 0.019, 0.635, 0.0005),
    ("confirmacao", "high", 0.833, 0.06, 0.99, 0.005),
]:
    linha = limiares_decisao.loc[
        (limiares_decisao["modo"] == modo) & (limiares_decisao["banda"] == banda)
    ].iloc[0]
    confere(f"{modo}: ponto de virada", linha["virada"], virada, 0.0005)
    confere(f"{modo}/{banda}: início da janela", linha["llm_inf"], inferior, tol)
    confere(f"{modo}/{banda}: fim da janela", linha["llm_sup"], superior, tol)
    for coluna, esperado in [("llm_inf", round(100 * inferior)), ("llm_sup", round(100 * superior))]:
        confere(f"{modo}/{banda}: {coluna} em %", 100 * linha[coluna], esperado, 0.5)""",
    ),
    (
        "md",
        """\
### 11.3 Travas, orçamento e fila

**🧒 A fila do posto de saúde.** Primeiro os que a regra prioriza
(`FORCE_REVIEW`), depois os que caíram numa **trava**, por último os que
**valem a pena** (EVR > 0). Dentro de cada grupo, vem primeiro o maior EVR.
Acabaram as vagas de revisão, acabou a revisão.

**As travas só PEDEM revisão; não decidem.** Em vigilância, MATCH base com
p < 0,85 ou NONMATCH base com p > 0,15 pede revisão. Em confirmação os cortes
são 0,95 e 0,10. A trava não muda p nem `base_choice`: sem orçamento, as linhas
da fila mantêm a escolha base. Ela pode pedir revisão **mesmo com EVR < 0**
(por exemplo, p = 0,80 na vigilância): compra segurança, não economia.

O custo escolhe a melhor opção automática e ordena a fila. A trava é uma camada
de segurança contra mudança nos dados e calibração ruim. Na vigilância, todo
pedido com MATCH base e p < 0,85 é contado como "trava" pela prioridade,
mesmo quando também tem EVR > 0. A decomposição abaixo separa os pedidos
que só existem por causa da trava: **84 de 208** têm EVR ≤ 0; os outros **124**
também teriam EVR > 0. Na confirmação, são **75 de 354** só pela trava e **279**
também por EVR. Não é correto atribuir a maioria dos pedidos só à trava.
Ela pede revisão; sem vaga, fica com a escolha base. A tabela conta pedidos
elegíveis por prioridade, sem contar os guardrails que decidem diretamente.

> ⚠️ Sem vaga de revisão, até um `FORCE_REVIEW` fica com a escolha base:
> um homônimo com nota alta pode ser aceito sozinho.

**Ordem no código:**
1. `review_requested = EVR > 0 ou trava ou FORCE_REVIEW` marca pedidos, não atendimento.
2. Guardrails `ALWAYS_*` decidem diretamente e **nunca entram na fila**, mesmo
   quando sua coluna `review_requested` é True.
3. A seleção segue `FORCE_REVIEW > trava > EVR > 0`, cada grupo por EVR decrescente,
   até o orçamento acabar.
4. Só os selecionados recebem `action = LLM_REVIEW`.

> 🔎 **Aqui as vagas de revisão sobram:** os orçamentos são 2000 (vigilância) e 1000
> (confirmação), para apenas 607 pares. O controle 11.6 (W6) reduz as vagas de revisão para tornar a fila visível.

Construímos dois motores reais, cada um com **orçamento novo**, e usamos
`triage(df.copy())`. A tabela de ações resume a operação; a linha do herói
mostra as três perdas, EVR, escolha base, pedido e ação em cada modo.
`nbh.expected_losses` refaz as contas e as conferências automáticas verificam contra o motor.

Com o p real do herói (0,6045), o código imprime perdas MATCH / NONMATCH / revisão
de 3,955 / 30,22 / 3,510 e EVR 0,445 em vigilância;
39,55 / 12,09 / 3,832 e EVR 8,258 em confirmação.""",
    ),
    (
        "code",
        """\
engine_vig = nbh.build_engine(cfg, "vigilancia")
out_vig = engine_vig.triage(df.copy())
engine_conf = nbh.build_engine(cfg, "confirmacao")
out_conf = engine_conf.triage(df.copy())

ordem_acoes = ["MATCH", "NONMATCH", "LLM_REVIEW"]
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

print("Ações por modo:")
print(dist_acoes.to_string())
pedidos_por_prioridade = {}
decomposicao_trava = {}
for modo, saida in [("vigilancia", out_vig), ("confirmacao", out_conf)]:
    politica = cfg.decision_policy.modes[modo]
    status = nbh.review_status(
        saida, min_auto_match=politica.min_auto_match_threshold,
        max_auto_nonmatch=politica.max_auto_nonmatch_threshold,
    )
    elegiveis = status["status"].isin(["revisado", "de fora"])
    pedidos_por_prioridade[modo] = status.loc[elegiveis, "prioridade"].value_counts().reindex(
        ["force", "trava", "evr"], fill_value=0,
    )
    trava = elegiveis & status["prioridade"].eq("trava")
    decomposicao_trava[modo] = {
        "trava também com EVR > 0": int((trava & saida["evr"].gt(0)).sum()),
        "só trava (EVR ≤ 0)": int((trava & saida["evr"].le(0)).sum()),
        "total trava": int(trava.sum()),
    }
    confere(f"{modo}: decomposição da trava",
            sum(list(decomposicao_trava[modo].values())[:2]), int(trava.sum()), 0)
print("Pedidos elegíveis por prioridade (force / trava / evr):")
display(pd.DataFrame(pedidos_por_prioridade))
print("Trava por prioridade não significa pedido causado só pela trava:")
print(pd.DataFrame(decomposicao_trava).to_string())
for modo, com_evr, so_trava, total in [
    ("vigilancia", 124, 84, 208), ("confirmacao", 279, 75, 354),
]:
    for coluna, esperado in [
        ("trava também com EVR > 0", com_evr), ("só trava (EVR ≤ 0)", so_trava),
        ("total trava", total),
    ]:
        confere(f"{modo}: {coluna}", decomposicao_trava[modo][coluna], esperado, 0)
confere("pares para triagem", len(df), 607, 0)
for modo, esperado in [("vigilancia", 2000), ("confirmacao", 1000)]:
    confere(f"{modo}: vagas da config", cfg.decision_policy.modes[modo].llm_max_calls_per_window, esperado, 0)

hero_p_cal = float(df.loc[hero_idx, "p_cal"])
hero_band = df.loc[hero_idx, "band"]
print(f"Herói (rota A): p_cal={br(hero_p_cal, 4)}; banda={hero_band}")
perdas_heroi = []
for modo, saida in [("vigilancia", out_vig), ("confirmacao", out_conf)]:
    politica = cfg.decision_policy.modes[modo]
    perdas = nbh.expected_losses(
        [hero_p_cal], c_fp=politica.false_positive_cost,
        c_fn=politica.false_negative_cost, c_llm=politica.llm_review_cost,
        **cfg.llm_review.error_rates_by_band[hero_band],
    ).iloc[0]
    for coluna in ["base_loss", "loss_llm", "evr"]:
        assert np.isclose(perdas[coluna], saida.loc[hero_idx, coluna]), coluna
    assert perdas["base_choice"] == saida.loc[hero_idx, "base_choice"]
    perdas_heroi.append({
        "modo": modo, **perdas.to_dict(),
        "review_requested": bool(saida.loc[hero_idx, "review_requested"]),
        "action": saida.loc[hero_idx, "action"],
    })
acoes_heroi = pd.DataFrame(perdas_heroi).set_index("modo")
print(acoes_heroi.round(4).to_string())
confere("probabilidade do herói na triagem", hero_p_cal, 0.6045, 0.00005)
for modo, esperados, tolerancias in [
    ("vigilancia", (3.955, 30.22, 3.510, 0.445), (0.0005, 0.005, 0.0005, 0.0005)),
    ("confirmacao", (39.55, 12.09, 3.832, 8.258), (0.005, 0.005, 0.0005, 0.0005)),
]:
    for coluna, esperado, tol in zip(
        ("loss_match", "loss_non", "loss_llm", "evr"), esperados, tolerancias, strict=True,
    ):
        confere(f"herói {modo}: {coluna}", acoes_heroi.loc[modo, coluna], esperado, tol)
PASSAPORTE["11. Triagem"] = acoes_heroi.to_dict(orient="index")""",
    ),
    (
        "md",
        """\
### 11.4 🎛️ W2: o preço de cada erro

Escolha o modo, os custos de FP/FN/revisão e a banda. Trocar o modo repõe seus
custos da configuração. À esquerda estão as três retas de perda, a janela em
que revisar é mais barato, o ponto de virada e as travas pontilhadas.
À direita, comparamos ações do motor real no ajuste e na configuração original
do mesmo modo, sempre com orçamento novo.

Teste: aumentar o preço de FP move o ponto de virada para onde? Aumentar a taxa
da revisão encolhe sua janela? A banda escolhida muda as retas ilustradas;
na triagem cada par continua usando **sua própria banda**.
A visão estática começa nos custos de vigilância e não altera `out_vig`/`out_conf`.

> 🔎 Mudanças pequenas mudam pouco as ações; a partir de 2 vezes o custo FP
> padrão, 3 pares das bandas `near_high`/`high` pedem revisão; com 3 vezes, são 59.
> Com custo
> de revisão alto, os pedidos que vinham só do EVR viram decisão automática.
> Teste e explique. A tabela mede esses efeitos com orçamento novo; os guardrails
> diretos continuam mandando, independentemente dos custos.""",
    ),
    (
        "code",
        """\
presets_custos = {
    modo: {"c_fp": politica.false_positive_cost,
           "c_fn": politica.false_negative_cost, "c_llm": politica.llm_review_cost}
    for modo, politica in cfg.decision_policy.modes.items()
}
custos_w2 = presets_custos["vigilancia"]
base_fp = custos_w2["c_fp"]
base_llm = custos_w2["c_llm"]
prioridade_original_w2 = nbh.review_status(
    out_vig, min_auto_match=politica_vig.min_auto_match_threshold,
    max_auto_nonmatch=politica_vig.max_auto_nonmatch_threshold,
)["prioridade"]
sensibilidade_custos = []
for fator_fp, fator_llm in [(1, 1), (2, 1), (3, 1), (5, 1), (1, 20)]:
    sonda = nbh.build_engine(
        cfg, "vigilancia", c_fp=base_fp * fator_fp, c_llm=base_llm * fator_llm,
    ).triage(df.copy())
    revisao = sonda["action"].eq("LLM_REVIEW")
    sensibilidade_custos.append({
        "FP × padrão": fator_fp, "revisão × padrão": fator_llm,
        "custo FP": base_fp * fator_fp, "custo revisão": base_llm * fator_llm,
        "LLM_REVIEW": int(revisao.sum()),
        "near_high/high em revisão": int((revisao & sonda["band"].isin(["near_high", "high"])).sum()),
        "só EVR agora automático": int((
            out_vig["action"].eq("LLM_REVIEW") & sonda["action"].ne("LLM_REVIEW")
            & prioridade_original_w2.eq("evr")
        ).sum()),
    })
tabela_sensibilidade = pd.DataFrame(sensibilidade_custos)
print("Sensibilidade dos custos em vigilância (sempre orçamento novo):")
print(tabela_sensibilidade.to_string(index=False))
for posicao, total, bandas_altas in [(0, 290, 0), (1, 293, 3), (2, 332, 59), (3, 383, 149), (4, 217, 0)]:
    confere(f"W2 linha {posicao}: revisões", tabela_sensibilidade.loc[posicao, "LLM_REVIEW"], total, 0)
    confere(f"W2 linha {posicao}: near_high/high",
            tabela_sensibilidade.loc[posicao, "near_high/high em revisão"], bandas_altas, 0)
confere("W2 revisão cara: só EVR agora automático",
        tabela_sensibilidade.loc[4, "só EVR agora automático"], 73, 0)


def _ui_custos(modo, c_fp, c_fn, c_llm, banda):
    fig, resumo = nbh.plot_costs_explorer(
        df, cfg, mode=modo, c_fp=c_fp, c_fn=c_fn, c_llm=c_llm, band=banda,
    )
    plt.show()
    plt.close(fig)
    display(resumo)


if not nbh.interagir(_ui_custos, {
    "modo": nbh.Escolha(list(cfg.decision_policy.modes), "vigilancia", "Modo"),
    "c_fp": nbh.Deslizante(custos_w2["c_fp"], 1, 200, 1, "Custo FP"),
    "c_fn": nbh.Deslizante(custos_w2["c_fn"], 1, 200, 1, "Custo FN"),
    "c_llm": nbh.Deslizante(custos_w2["c_llm"], 0, 5, 0.05, "Custo revisão"),
    "banda": nbh.Escolha([b.name for b in cfg.bands.definitions], "grey_mid", "Banda"),
}, presets=presets_custos, preset_de="modo"):
    _ui_custos("vigilancia", **custos_w2, banda="grey_mid")""",
    ),
    (
        "md",
        """\
### 11.5 Pares que mudam de decisão entre modos

A evidência e a probabilidade são as mesmas; os preços e as travas mudam.
Mostramos os cinco primeiros pares com **ações finais** diferentes, não apenas escolhas base
diferentes. O herói pode ter escolhas base opostas e, ainda assim, revisão nos
dois modos. Mudança de ação não é inconsistência: responde ao objetivo operacional.""",
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

pares_mudam.head(5)""",
    ),
    (
        "md",
        """\
### 11.6 🎛️ W6: quando faltam vagas de revisão

Primeiro mostramos o orçamento da configuração, em que ninguém da fila fica
de fora. Depois usamos aproximadamente **25% dos pedidos elegíveis** em cada
modo: o gráfico passa a mostrar uma fila real. `ALWAYS_*` não conta como pedido
elegível, mesmo quando `review_requested` é True.

As cores separam revisado, pediu mas ficou de fora, não pediu e guardrail;
os marcadores distinguem FORCE_REVIEW, trava e EVR. As barras contam revisados
e excluídos por prioridade. O último EVR atendido pode ser menor que o primeiro
excluído de **outro grupo**, pois prioridade vem antes de economia.

O slider começa no menor entre orçamento da config e pedidos, e chega a zero.
Ao trocar o modo, repomos esse padrão; o máximo cobre os pedidos dos dois modos.
Teste cortar todas as vagas de revisão: quem mantém a escolha base? Os guardrails diretos
mudam? Cada chamada cria um orçamento novo, sem consumir as vagas de revisão da rota A.""",
    ),
    (
        "code",
        """\
pedidos_w6 = {}
for modo, saida in [("vigilancia", out_vig), ("confirmacao", out_conf)]:
    politica = cfg.decision_policy.modes[modo]
    status = nbh.review_status(
        saida, min_auto_match=politica.min_auto_match_threshold,
        max_auto_nonmatch=politica.max_auto_nonmatch_threshold,
    )
    pedidos_w6[modo] = int(status["status"].isin(["revisado", "de fora"]).sum())


def _ui_orcamento(modo, orcamento):
    fig, resumo = nbh.plot_budget_explorer(df, cfg, mode=modo, llm_max=orcamento)
    plt.show()
    plt.close(fig)
    contagens = resumo["status"].value_counts()
    print(f"{modo}: orçamento config="
          f"{cfg.decision_policy.modes[modo].llm_max_calls_per_window}; "
          f"explorado={orcamento}; pedidos elegíveis={pedidos_w6[modo]}; "
          f"revisados={contagens.get('revisado', 0)}; "
          f"de fora={contagens.get('de fora', 0)}")
    display(pd.crosstab(resumo["prioridade"], resumo["status"]))


for modo in pedidos_w6:
    _ui_orcamento(modo, int(round(0.25 * pedidos_w6[modo])))

presets_orcamento = {
    modo: {"orcamento": min(politica.llm_max_calls_per_window, pedidos_w6[modo])}
    for modo, politica in cfg.decision_policy.modes.items()
}
if not nbh.interagir(_ui_orcamento, {
    "modo": nbh.Escolha(list(cfg.decision_policy.modes), "vigilancia", "Modo"),
    "orcamento": nbh.Deslizante(
        presets_orcamento["vigilancia"]["orcamento"], 0, max(pedidos_w6.values()),
        1, "Vagas de revisão", inteiro=True,
    ),
}, presets=presets_orcamento, preset_de="modo"):
    _ui_orcamento("vigilancia", presets_orcamento["vigilancia"]["orcamento"])""",
    ),
]


FASE_3_1 = [
    (
        "md",
        """## 12. Reconciliação com `run_v3` (rota A, in-sample)

🧒 Conferimos a conta passo a passo com a calculadora oficial. Chamamos `run_v3`
com o mesmo CSV, config, MACD e Platt ajustado em todas as linhas da rota A.
Comparamos **exatamente** `band`, `p_cal`, `action`, `guardrail`, `evr`,
`base_choice` e `review_requested` com `out_vig`.
O `RunSummary` registra volume, regras, pedidos de revisão e parâmetros efetivos
da calibração: uma pequena trilha de auditoria, não uma avaliação held-out.""",
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

# Comparação coluna a coluna com a rota A manual (out_vig, da seção 11)
import numpy as np

band_identico = bool((out_run["band"].to_numpy() == out_vig["band"].to_numpy()).all())
pcal_maxdiff = float(
    np.max(np.abs(out_run["p_cal"].to_numpy(float) - out_vig["p_cal"].to_numpy(float)))
)
acao_identica = bool((out_run["action"].to_numpy() == out_vig["action"].to_numpy()).all())

print(f"band idêntico:           {band_identico}")
print(f"p_cal diferença máxima:  {br(pcal_maxdiff, 10)}")
print(f"action idêntico:         {acao_identica}")
assert band_identico and pcal_maxdiff == 0.0 and acao_identica
for coluna in ["guardrail", "evr", "base_choice", "review_requested"]:
    pd.testing.assert_series_equal(out_run[coluna], out_vig[coluna], check_exact=True)
    print(f"{coluna} idêntico: True (incluindo valores ausentes)")

resumo = {
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
        """**Por que bate exatamente?** Mesmos dados, parâmetros e ajuste Platt
determinístico produzem as mesmas bandas, probabilidades, guardrails e ações.
Isso verifica a implementação, não prova generalização. A rota B muda o conjunto
de ajuste de propósito; diferenças nela são esperadas. Modelos ML potencialmente
não determinísticos exigiriam outros critérios, não igualdade bit a bit.""",
    ),
]


FASE_3_2 = [
    (
        "md",
        r"""## 13. Avaliação held-out (rota B)

> 🧸 **Bem simples.** Imagine uma prova surpresa com questões que você nunca viu.
> Ela mostra melhor o que aprendeu do que repetir o exercício do caderno.

> 🧒 **Um pouco mais.** O código aprende nas fichas de treino e decide nas fichas
> reservadas para teste. As regras de segurança continuam valendo nessa prova.

> 📖 **Palavra nova: avaliação held-out.** É conferir o aprendizado em pares que o ajuste nunca viu.

> 🧸 **Bem simples.** Aceitar uma ficha errada e deixar uma ficha certa para trás
> são problemas diferentes. Pense numa peneira: você quer guardar o que procura sem guardar sujeira.

> 🧒 **Um pouco mais.** Precisão pergunta se os pares aceitos estavam certos, como o valor preditivo positivo. Recall pergunta quantos pares da mesma pessoa foram aceitos, como a sensibilidade. **Atenção:** nas tabelas desta seção, as duas contas olham só os pares que o programa decidiu sozinho; os que foram para revisão ficam de fora. Leia sempre a cobertura ao lado.

> 📖 **Palavra nova: precisão automática.** É a parte correta entre os pares aceitos como mesma pessoa (VPP), entre os decididos automaticamente.
> 📖 **Palavra nova: recall automático.** Entre os pares da mesma pessoa que o programa decidiu sozinho, a parte que ele aceitou. Só vira a sensibilidade do sistema inteiro quando nada fica pendente.

> 🧸 **Bem simples.** Uma peneira pode parecer perfeita porque outra regra já tirou
> os casos difíceis. Isso não prova que aprendeu a medir probabilidades.

> 🧒 **Um pouco mais.** Precisão, recall e Fβ entre os decididos medem sobretudo
> as travas e os guardrails. Em vigilância, aceitar sozinho exige p ≥ 0,85 pela
> trava, fora dos guardrails: com Platt crescente, equivale a um corte fixo na nota.
> O recall automático igual a 1, sem variação, vem das regras neste experimento,
> não demonstra estabilidade da calibração. Em confirmação, a curva nem alcança
> a saída superior da janela de revisão: a precisão 1,000 vem do `ALWAYS_MATCH`.

> 🧸 **Bem simples.** Um boletim pode dar mais atenção a perder encontros ou a aceitar
> enganos. É como escolher o que pesa mais na mochila antes de sair.

> 🧒 **Um pouco mais.** Fβ reúne precisão e recall numa nota. Em vigilância,
> usamos beta 2 para priorizar não perder pares da mesma pessoa; em confirmação, beta 0,5
> para priorizar aceitar pares corretos. São réguas diferentes, não um placar único.

> 📖 **Palavra nova: Fβ.** É uma nota conjunta com prioridade ajustável entre precisão e recall, entre os decididos automaticamente.

**Exemplo pequeno.** Aceitar 5 pares e acertar 4 dá precisão 0,8.
Se existiam 10 encontros verdadeiros entre os decididos automaticamente, encontrar 4 dá recall automático 0,4.

<details>
<summary>📐 Para quem quer a matemática</summary>

Precisão $P=TP/(TP+FP)$; recall $R=TP/(TP+FN)$.
$$F_\beta=\frac{(1+\beta^2)PR}{\beta^2P+R}$$
Em `auto_fbeta`, β = 2 em vigilância e β = 0,5 em confirmação.
O peso relativo do recall é β²: 4 no primeiro caso e 0,25 no segundo.
Com P = 0,8 e R = 0,6, F₂ ≈ 0,63 e F₀,₅ ≈ 0,75.
As métricas automáticas excluem as linhas pendentes como `LLM_REVIEW`.

</details>

> 🎯 **Para levar.** Uma prova nova e uma régua adequada ao objetivo vêm antes de uma nota bonita.""",
    ),
    (
        "md",
        r"""### 13.1 Métricas held-out em múltiplas sementes

> 🧸 **Bem simples.** Jogar o dado várias vezes ajuda a perceber a sorte de uma jogada.
> Refazer o sorteio da turma de estudo evita confiar numa prova especialmente fácil.

> 🧒 **Um pouco mais.** Usamos 5 sementes para mudar a divisão do mesmo conjunto,
> não para inventar dados novos. A média mostra o resultado central; o desvio-padrão,
> quanto os resultados se espalham. Na tabela, centro e variação aparecem em colunas lado a lado.

> 📖 **Palavra nova: semente (seed).** É o ponto de partida que permite repetir um sorteio.
> 📖 **Palavra nova: média e desvio-padrão.** São o centro dos resultados e uma medida de quanto variam.

> 🧸 **Bem simples.** Um ajudante pode acertar muito porque deixa os casos difíceis para você.
> Conte também quanto trabalho ele realmente resolveu sozinho.

> 🧒 **Um pouco mais.** A cobertura conta os pares que o programa resolveu sem mandar para revisão. Ela diz quanto trabalho sobrou, não quantos acertos houve.

> 📖 **Palavra nova: cobertura automática.** É a parte resolvida sem ficar pendente de revisão, não a parte acertada.

**Exemplo pequeno.** Resolver 6 de 10 pares dá cobertura de 60%, mesmo que haja erros.

<details>
<summary>📐 Para quem quer a matemática</summary>

Com k = 5 sorteios, $\bar x=\sum_j x_j/k$ e
$s=\sqrt{\sum_j(x_j-\bar x)^2/(k-1)}$.
Cobertura é o número de ações diferentes de `LLM_REVIEW` dividido pelo total.
`evaluate_v3_dataframe` refaz a calibração no treino e a triagem no teste de cada sorteio.
A variação entre sorteios deste sintético não é um intervalo de confiança populacional.

</details>

Os 5 sorteios reorganizam os mesmos 607 pares, não amostram uma população nova.
Também mostramos a **sensibilidade esperada do sistema inteiro, supondo as taxas
de erro do revisor da config**: ela inclui os pares revisados, ao contrário do recall automático.

> 🎯 **Para levar.** Leia regras, custo e trabalho resolvido juntos; a dispersão entre divisões não é incerteza populacional.""",
    ),
    (
        "code",
        """from time import perf_counter
from gzcmd_record_linkage.eval import evaluate_v3_dataframe

# Recarregamos as pistas originais com MACD, sem as anotações da aula,
# e conferimos abaixo que o Platt mantém seus resultados.
df_ml = load_comparador_csv(CSV_PATH, cfg=LoadConfig(macd_enabled=True))
pd.testing.assert_frame_equal(df_ml[["COMPREC", "REFREC"]], df[["COMPREC", "REFREC"]])
colunas_da_aula = set(df.columns) - set(df_ml.columns)

SEEDS_32 = [42, 123, 456, 789, 2024]
metric_cols_32 = ["auto_precision", "auto_recall", "auto_fbeta", "auto_coverage"]

args_avaliacao = dict(
    cfg=cfg,
    modes=["vigilancia", "confirmacao"],
    split_by="comprec",
    seeds=SEEDS_32,
    test_size=0.3,
    group_stratify=True,
    macd_enabled=True,
)
inicio_platt_politica = perf_counter()
res = evaluate_v3_dataframe(df_ml, calibration="platt", **args_avaliacao)
tempo_platt_politica = perf_counter() - inicio_platt_politica
res_platt_anterior = evaluate_v3_dataframe(df, calibration="platt", **args_avaliacao)
pd.testing.assert_frame_equal(res, res_platt_anterior)
print("Platt: frame limpo e frame da aula dão resultados idênticos.")

resumo_multi_seed = res.groupby("mode")[metric_cols_32].agg(["mean", "std"])
print("Olhe a coluna 'Resolve sozinho (cobertura)': acerto alto pode esconder muitos pendentes.")
display(resumo_multi_seed.rename(columns={
    "auto_precision": "Acerto entre aceitos e decididos (precisão automática)",
    "auto_recall": "Encontrados entre os decididos (recall automático)",
    "auto_fbeta": "Nota do modo entre os decididos (Fβ)",
    "auto_coverage": "Resolve sozinho (cobertura)",
    "mean": "Média", "std": "Variação (desvio-padrão)",
}).round(4))
confere("sorteios da avaliação", len(SEEDS_32), 5, 0)
confere("pares reorganizados", len(df_ml), 607, 0)
confere("trava de vigilância", cfg.decision_policy.modes["vigilancia"].min_auto_match_threshold, 0.85, 0)
confere("recall automático de vigilância (média)", res.loc[res["mode"].eq("vigilancia"), "auto_recall"].mean(), 1, 0)
confere("recall automático de vigilância (desvio)", res.loc[res["mode"].eq("vigilancia"), "auto_recall"].std(), 0, 0)
confere("precisão automática de confirmação", res.loc[res["mode"].eq("confirmacao"), "auto_precision"].mean(), 1, 0)
limiares_confirmacao = nbh.thresholds_table(cfg).query("modo == 'confirmacao' and banda == 'high'")
p_max_rota_b = float(1 / (1 + np.exp(-(platt_holdout.slope * 10 + platt_holdout.intercept))))
limite_superior_high = float(limiares_confirmacao["llm_sup"].iloc[0])
print("Confirmação, banda high: janela de revisão e máximo p da rota B na nota 10:")
print(limiares_confirmacao.assign(p_max_rota_b=p_max_rota_b).to_string(index=False))
assert p_max_rota_b < limite_superior_high
print("O Platt não sai por cima da janela: os MATCH automáticos vêm do ALWAYS_MATCH.")
metricas_sistema = ["exp_recall", "exp_fbeta", "oracle_recall"]
print("Sensibilidade esperada do sistema inteiro, supondo as taxas de erro do revisor da config (média ± desvio-padrão):")
print(res.groupby("mode")[metricas_sistema].agg(["mean", "std"]).rename(columns={
    "exp_recall": "Sensibilidade esperada", "exp_fbeta": "Fβ esperado",
    "oracle_recall": "Sensibilidade com revisor perfeito",
    "mean": "Média", "std": "Desvio-padrão",
}).to_string())""",
    ),
    (
        "md",
        """### 13.2 Barras com incerteza entre sementes

> 🧸 **Bem simples.** Imagine marcas de altura feitas em dias diferentes.
> Marcas muito espalhadas pedem mais cuidado ao contar a história.

> 🧒 **Um pouco mais.** A altura da barra é a média dos 5 sorteios.
> A haste mostra o desvio-padrão: grande significa maior dependência da divisão escolhida.

**Exemplo pequeno.** Resultados 0,7 e 0,9 espalham mais que dois resultados 0,8.

> 🎯 **Para levar.** Hastes pequenas indicam estabilidade aqui, não garantia em outra população.""",
    ),
    (
        "code",
        """fig, ax = plt.subplots(figsize=(9, 5))

modes_32 = resumo_multi_seed.index.to_list()
x = np.arange(len(modes_32))
width = 0.22
metric_labels_32 = {
    "auto_precision": "Acerto entre aceitos e decididos (precisão automática)",
    "auto_recall": "Encontrados entre os decididos (recall automático)",
    "auto_fbeta": "Nota do modo entre os decididos (Fβ)",
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
ax.set_title("Centro e variação (média ± desvio-padrão): 5 sorteios por candidato")
ax.legend(loc="lower right")
ax.grid(axis="y", alpha=0.25)
plt.show()""",
    ),
    (
        "md",
        """### 13.3 Sortear fichas soltas ou grupos de fichas

> 🧸 **Bem simples.** O mesmo aluno não pode estar na turma de estudo e na de prova.
> Se a mesma pessoa está nas duas turmas, a prova não é totalmente nova.

> 🧒 **Um pouco mais.** Sortear linhas (`row`) pode repetir um registro dos dois lados.
> Agrupar por `COMPREC` mantém todos os pares daquele candidato juntos; por `REFREC`,
> mantém os da referência. Nenhum dos dois garante separar toda pessoa nos dois lados.

> 📖 **Palavra nova: divisão por grupo.** É sortear blocos de fichas com o mesmo identificador, não fichas soltas.
> 📖 **Palavra nova: vazamento.** É deixar uma informação conhecida no estudo ajudar indevidamente na prova.

**Exemplo pequeno.** Os 2 pares de um candidato vão juntos para treino ou teste.
Contamos linhas com ID repetido e comparamos as médias dos 3 tipos de divisão.

> 🎯 **Para levar.** Escolha um grupo que represente quem precisa ficar realmente fora do estudo.""",
    ),
    (
        "code",
        """# Multiplicidade dos grupos: fração de linhas cujo COMPREC/REFREC se repete.
_comprec = df["COMPREC"] if "COMPREC" in df.columns else df.filter(like="COMPREC").iloc[:, 0]
_refrec = df["REFREC"] if "REFREC" in df.columns else df.filter(like="REFREC").iloc[:, 0]
frac_comprec_repetido = (_comprec.map(_comprec.value_counts()) > 1).mean()
frac_refrec_repetido = (_refrec.map(_refrec.value_counts()) > 1).mean()
print(f"Fração de linhas com COMPREC repetido: {br(100 * (frac_comprec_repetido), 1)}%".replace(".", ","))
print(f"Fração de linhas com REFREC repetido:  {br(100 * (frac_refrec_repetido), 1)}%".replace(".", ","))

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
delta_max_fbeta = float(
    leakage_summary["auto_fbeta_mean"].max() - leakage_summary["auto_fbeta_mean"].min()
)
print("Olhe a coluna 'Nota do modo entre os decididos (Fβ)': comparamos médias, não a variação entre sorteios.")
print(leakage_summary.rename(columns={
    "auto_fbeta_mean": "Nota do modo entre os decididos (Fβ)",
    "auto_recall_mean": "Encontrados entre os decididos (recall automático)",
    "auto_coverage_mean": "Resolve sozinho (cobertura)",
}).rename(index={"row": "fichas soltas", "comprec": "por candidato", "refrec": "por referência"})
    .rename_axis("Divisão").round(4).to_string())
print(f"Maior diferença de Fβ entre médias das divisões: {br(delta_max_fbeta)}")""",
    ),
    (
        "md",
        """**Lendo com honestidade.** `_make_ids` sorteia IDs independentemente de tudo:
vazamento via ID é impossível aqui. A diferença esperada entre tipos de divisão
é zero; as diferenças medidas de Fβ e cobertura são ruído de partição.
As frações de linhas com ID repetido decorrem dos ajustes `unique_share` do
gerador, não são descobertas sobre dados reais. A tabela mantém essa medição
e a cobertura para mostrar o efeito de mudar quem entra na prova.
Em bases reais, comparações da mesma pessoa podem compartilhar informação:
este sintético não testa esse problema.""",
    ),
    (
        "md",
        r"""### 13.4 Curvas PR e ROC da rota B

> 🧸 **Bem simples.** Imagine uma fila organizada dos encontros mais prováveis aos menos prováveis.
> Mover a porta de entrada deixa passar mais encontros, mas também mais enganos.

> 🧒 **Um pouco mais.** O código muda o corte de aceitação e desenha o que acontece.
> A ROC compara a sensibilidade com a fração de pares de pessoas diferentes aceitos por engano (1 − especificidade).
> A PR compara sensibilidade com precisão, útil sobretudo quando os encontros verdadeiros são raros.

> 📖 **Palavra nova: curva ROC.** É o desenho da sensibilidade contra 1 − especificidade ao mudar o corte de aceitação.
> 📖 **Palavra nova: curva PR.** É o desenho de precisão contra sensibilidade nesses cortes.

> 🧸 **Bem simples.** Sorteie uma ficha de encontro verdadeiro e outra de engano.
> Veja qual o ajudante põe na frente da fila.

> 🧒 **Um pouco mais.** A AUC-ROC resume essa capacidade de ordenar, não a honestidade
> das probabilidades anunciadas. Aqui usamos só o teste da §9.4, sem revisão nem regras de segurança.

> 📖 **Palavra nova: AUC.** É a área sob uma curva; na ROC, resume quem costuma ficar na frente.

**Exemplo pequeno.** AUC-ROC 0,9 indica ordem correta em cerca de 9 de 10 sorteios
desse tipo; empates contam pela metade. Não significa acertar 90% das decisões.

Na §13.7 aparece uma prima da área PR, a **AP** (*average precision*): a precisão média ao longo da fila, contada a cada novo encontro achado. Maior é melhor; não compare a AP com a área desta seção.

<details>
<summary>📐 Para quem quer a matemática</summary>

ROC: $\mathrm{TPR}=TP/(TP+FN)$ e $\mathrm{FPR}=FP/(FP+TN)$.
$$\mathrm{AUC}_{ROC}=P(p_+>p_-)+\tfrac12P(p_+=p_-)$$
AUC-ROC 0,5 equivale a ordem aleatória; 1 indica separação perfeita.
A área PR aqui usa integração trapezoidal da precisão em função do recall.
Já a §13.7 usa **average precision** (AP), média das precisões ponderada pelos
incrementos de recall: $AP=\sum_n(R_n-R_{n-1})P_n$.
As duas áreas PR não são a mesma conta e dependem da proporção de positivos.

</details>

> 🎯 **Para levar.** Ordenar bem não garante prometer probabilidades honestas nem operar bem a triagem.""",
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

axes[0].plot(recall_curve, precision_curve, label=f"Área sob a curva (PR-AUC) = {br(pr_auc)}")
axes[0].set_xlabel("Sensibilidade (recall)")
axes[0].set_ylabel("Precisão (VPP)")
axes[0].set_title("Precisão e sensibilidade (curva PR, rota B)")
axes[0].set_xlim(0, 1.02)
axes[0].set_ylim(0, 1.02)
axes[0].grid(alpha=0.25)
axes[0].legend(loc="lower left")

axes[1].plot(fpr_curve, tpr_curve, label=f"Ordenação (AUC-ROC) = {br(roc_auc)}")
axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray", label="aleatório")
axes[1].set_xlabel("Pessoas diferentes aceitas por engano (1 − especificidade)")
axes[1].set_ylabel("Encontros encontrados (sensibilidade)")
axes[1].set_title("Curva ROC (rota B)")
axes[1].set_xlim(0, 1.02)
axes[1].set_ylim(0, 1.02)
axes[1].grid(alpha=0.25)
axes[1].legend(loc="lower right")

plt.tight_layout()
plt.show()

pd.Series({"Área da curva PR (trapezoidal)": pr_auc, "Ordenação (AUC-ROC)": roc_auc}).to_frame("valor")""",
    ),
    (
        "md",
        r"""### 13.5 Custo observado versus limiar de MATCH

> 🧸 **Bem simples.** Imagine uma porta com um porteiro mais ou menos exigente.
> Deixar entrar quem não devia e barrar quem devia entrar têm preços diferentes.

> 🧒 **Um pouco mais.** Para cada corte, o código soma os erros do teste pelos
> preços de `cfg`, sem revisão nem regras de segurança. O ponto mais baixo mostra
> o menor gasto **observado aqui**, não o melhor corte garantido para amanhã.

> 📖 **Palavra nova: curva de custo por limiar.** É o gasto com os erros conforme você muda o corte de aceitação.

**Exemplo pequeno.** Com preço 10 por alarme falso, 2 alarmes custam 20.
A linha pontilhada é uma trava do modo: abaixo da linha o par pede revisão
(se a escolha base for MATCH); sem vaga, fica com a escolha base; acima,
ainda precisa de EVR ≤ 0 para ser aceito sozinho, fora dos guardrails.

<details>
<summary>📐 Para quem quer a matemática</summary>

Para cada τ, predizemos MATCH quando $p\geq\tau$ e calculamos
$C(\tau)=FP(\tau)c_{fp}+FN(\tau)c_{fn}$.
O ponto teórico $c_{fp}/(c_{fp}+c_{fn})$ minimiza a perda **esperada** para probabilidades
calibradas, não necessariamente o gasto observado num teste finito.
O mínimo observado é apenas diagnóstico; escolher o corte por ele exige outra prova independente.

</details>

> 🎯 **Para levar.** Use o teste para entender o gasto, não para escolher um corte e declarar vitória.""",
    ),
    (
        "code",
        """thresholds = np.linspace(0, 1, 101)
mode_costs = {
    mode: {
        "false_positive": policy.false_positive_cost,
        "false_negative": policy.false_negative_cost,
        "min_auto_match": policy.min_auto_match_threshold,
    }
    for mode, policy in cfg.decision_policy.modes.items()
}

cost_rows = []
for mode, costs in mode_costs.items():
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
    ax.plot(group["tau"], group["total_cost"], label=f"custo ({mode})")
    costs = mode_costs[mode]
    policy_tau = costs["min_auto_match"]
    if policy_tau is not None:
        ax.axvline(policy_tau, linestyle=":", alpha=0.65,
                   label=f"trava min_auto_match ({mode}): {br(policy_tau, 2)}")
    break_even = costs["false_positive"] / (costs["false_positive"] + costs["false_negative"])
    ax.axvline(break_even, linestyle="--", alpha=0.65,
               label=f"ponto de virada ({mode}): {br(break_even)}")
    best_row = best_cost.loc[best_cost["mode"] == mode].iloc[0]
    ax.scatter(best_row["tau"], best_row["total_cost"], s=55)
    ax.annotate(
        f"menor gasto: corte {br(best_row['tau'], 2)}",
        xy=(best_row["tau"], best_row["total_cost"]),
        xytext=(5, 8),
        textcoords="offset points",
    )

ax.set_xlabel("Corte de aceitação (limiar de MATCH)")
ax.set_ylabel("Custo total no teste")
ax.set_title("Custo observado no teste por limiar e modo")
ax.grid(alpha=0.25)
ax.legend(loc="best")
plt.show()

print("Olhe a coluna 'Gasto no teste': o menor gasto aqui não escolhe o corte de amanhã.")
display(best_cost[["mode", "tau", "fp", "fn", "total_cost"]].rename(columns={
    "mode": "Modo", "tau": "Corte (limiar)", "fp": "Alarmes falsos (FP)",
    "fn": "Encontros perdidos (FN)", "total_cost": "Gasto no teste",
}))""",
    ),
    (
        "md",
        """### 13.6 Interpretação e próximos passos

> 🧸 **Bem simples.** Não escolha o ajudante só pela nota do boletim.
> Veja também quanto trabalho deixou para você e quanto custaram os enganos.

> 🧒 **Um pouco mais.** Leia precisão, sensibilidade e cobertura junto com a
> proporção de encontros, a honestidade das probabilidades e os custos. Compare Fβ
> **dentro do mesmo modo**. A §13.7 põe outros aprendizes na mesma prova.

**Exemplo pequeno.** Acertar todos os 2 casos resolvidos pode esconder 8 pendentes.
Na §14 voltamos à rota A para revisão **simulada**: não é continuação desta prova
nem medida de capacidade de um LLM real.

> 🎯 **Para levar.** Acerto, pendências e custo contam partes diferentes da história.""",
    ),
    (
        "md",
        r"""### 13.7 Caminho ML: Random Forest e XGBoost contra o Platt na nota

> 🧸 **Bem simples.** Imagine um ajudante que aprende a separar fichas olhando exemplos respondidos.
> Depois ele tenta responder uma ficha nova sem espiar o gabarito.

> 🧒 **Um pouco mais.** O Platt olha só a nota final. O `GZCMDClassifier` aprende
> com as colunas numéricas selecionadas e depois ajusta suas probabilidades anunciadas.

> 📖 **Palavra nova: classificador.** É um programa que aprende a distinguir tipos de resposta usando exemplos.
> 📖 **Palavra nova: GZCMDClassifier.** É o aprendiz da biblioteca que combina pistas e calibra suas probabilidades.

> 🧸 **Bem simples.** Pense num jogo de adivinhação em que cada resposta escolhe a próxima pergunta.
> No fim do caminho, o ajudante dá seu palpite.

> 🧒 **Um pouco mais.** Uma árvore de decisão faz o jogo de 20 perguntas com pistas:
> "o nome é parecido?", depois "a data combina?". O treino escolhe perguntas e cortes.

> 📖 **Palavra nova: árvore de decisão.** É uma sequência de perguntas sobre pistas que leva a uma previsão.

> 🧸 **Bem simples.** Imagine um comitê de ajudantes que estudaram cadernos diferentes.
> Eles juntam seus votos em vez de depender de um palpite só.

> 🧒 **Um pouco mais.** Random Forest junta árvores; cada uma vê uma seleção
> sorteada de pares e considera partes diferentes das pistas ao perguntar.

> 📖 **Palavra nova: Random Forest.** É um comitê de árvores que combina seus votos.

> 🧸 **Bem simples.** Imagine um aluno que volta às questões que errou.
> Cada nova rodada de estudo tenta consertar o que ainda ficou errado.

> 🧒 **Um pouco mais.** XGBoost acrescenta árvores em sequência para corrigir
> erros das anteriores. Não é outro comitê de votos independentes.

> 📖 **Palavra nova: XGBoost e boosting.** XGBoost usa boosting, o aprendizado em etapas que corrigem os erros restantes.

> 🧸 **Bem simples.** Divida os exercícios em 5 montes. Estude com 4 e confira no que sobrou; troque o monte de conferência até todos terem servido uma vez. A prova final continua fechada.

> 🧒 **Um pouco mais.** O código faz isso com o treino: 5 rodadas, cada uma aprende com 4 partes e confere na quinta. Nessas conferências ele aprende a traduzir os palpites das árvores em probabilidades (um Platt por dentro). A probabilidade final é a média das 5 rodadas.

> 📖 **Palavra nova: validação cruzada.** É revezar partes do treino entre aprender e conferir sem usar o teste final.
> 📖 **Palavra nova: calibração interna.** É ajustar a tradução dos palpites em probabilidades dentro do próprio classificador.

**Exemplo concreto.** O código imprime as colunas selecionadas e o tamanho do treino.
Usamos as pistas sem as colunas criadas pela aula (§13.1), o treino da §9.3 e exatamente o mesmo teste.

<details>
<summary>📐 Para quem quer a matemática</summary>

⚠️ A regra que escolhe as colunas pega toda coluna numérica permitida, até cópias como `nota final` /
`nota_final` e as quatro datas cruas, como `C_DTNASC,C,8,0`, lidas como números.
Uma data escrita como número não é, por si, uma pista útil de semelhança.
Aliases dividem a importância. Uma alternativa mais justa usaria apenas os
escores agregados; não alteramos a regra da biblioteca aqui.

RF reamostra pares com reposição e sorteia subconjuntos de pistas em cada divisão.
XGBoost adiciona árvores à previsão: $f_t(x)=f_{t-1}(x)+\eta h_t(x)$;
a nova árvore aproxima correções da perda, não apenas uma lista literal de pares errados.
Na calibração interna, cada parte é conferida por uma árvore/comitê que não a treinou;
uma sigmoide aprende a converter os escores em probabilidades, e as previsões são combinadas.
Usamos `CalibratedClassifierCV(method="sigmoid", cv=5)`: para este estimador não congelado,
o padrão efetivo é `ensemble=True`, com média das previsões dos 5 ajustes calibrados.

Fixamos CPU em todos os ajustes de XGBoost para números reprodutíveis com estas versões das bibliotecas.
O tempo Platt inclui um novo ajuste conferido contra a §9.4; RF/XGB incluem a
calibração interna. A contagem é da lista selecionada: nesta versão do scikit-learn,
colunas inteiramente vazias no treino são descartadas pelo preenchimento,
também ausentes das importâncias. O código confere nomes e importâncias após preencher.

**A régua escondida tem limites.** O gerador sorteia TARGET pela sigmoide da nota.
Nas linhas aleatórias, as outras pistas vêm da nota e de ruído, sem depender
de TARGET: dada a nota, não acrescentam informação sobre a resposta.
Assim, p* é referência válida nesse mundo, mas não verdade universal condicionada
a todas as pistas. Nos cenários anexados, TARGET é escolhido à mão: p* continua
sendo a curva da nota, não a probabilidade verdadeira do cenário.

</details>

⚠️ **O jogo favorece a nota.** A probabilidade escondida p* depende só dela; as demais
pistas são inventadas com ruído a partir da nota, sem consultar a resposta.
Por isso não acrescentam informação dada a nota. "Erro contra p*" não é uma régua justa para
julgar ML em geral: num outro gerador, pistas extras poderiam revelar algo novo.
Os cenários com resposta escolhida à mão nem seguem esse sorteio.
A regra que escolhe as colunas também inclui cópias e datas cruas como números;
mais colunas não significam mais informação útil de semelhança.

> 🎯 **Para levar.** Mais pistas e um aprendiz mais complexo não garantem vantagem neste jogo inventado.""",
    ),
    (
        "code",
        """from gzcmd_record_linkage.classifier import ClassifierConfig, GZCMDClassifier

inicio_secao_ml = perf_counter()
inicio = perf_counter()
platt_cronometrado = fit_platt_from_df(df_ml.iloc[train_idx])
tempo_platt_split = perf_counter() - inicio
np.testing.assert_allclose(
    compute_p_cal(df_ml.iloc[test_idx], method="platt", model=platt_cronometrado),
    p_cal_test, atol=0, rtol=0,
)
configs_ml = {
    "Random Forest": ClassifierConfig(classifier_type="random_forest", xgb_device="cpu", random_state=SEED),
    "XGBoost": ClassifierConfig(classifier_type="xgboost", xgb_device="cpu", random_state=SEED),
}
classificadores = {}
previsoes_modelos = {"Platt (só a nota)": np.asarray(p_cal_test)}
tempos_ajuste = {"Platt (só a nota)": tempo_platt_split}
quantidades_features = {"Platt (só a nota)": 1}
print(f"Treino único: {len(train_idx)} pares; teste único: {len(test_idx)} pares")
for nome, config_ml in configs_ml.items():
    inicio = perf_counter()
    clf = GZCMDClassifier(config=config_ml).fit(df_ml.iloc[train_idx])
    tempos_ajuste[nome] = perf_counter() - inicio
    assert clf.features_ is not None
    proibidas = {"p_cal", "p_true", "TARGET", "PAR"} | colunas_da_aula
    assert not (set(clf.features_) & proibidas)
    assert set(clf.features_) <= set(df_ml.columns)
    print(f"{nome}: {len(clf.features_)} pistas selecionadas (features), sem gabarito/anotações da aula:")
    print(clf.features_)
    datas_cruas_ml = [coluna for coluna in clf.features_ if "," in coluna]
    print("Datas cruas lidas como números:", datas_cruas_ml)
    confere(f"{nome}: datas cruas selecionadas", len(datas_cruas_ml), 4, 0)
    confere(f"{nome}: partes da calibração interna", config_ml.calibration_cv, 5, 0)
    quantidades_features[nome] = len(clf.features_)
    vazias_treino = df_ml.iloc[train_idx][clf.features_].isna().all()
    print(f"{nome}: há coluna inteiramente NaN no treino? {bool(vazias_treino.any())}")
    nomes_preenchidos = clf.pipeline_["imputer"].get_feature_names_out(clf.features_)
    for parte in clf.pipeline_["classifier"].calibrated_classifiers_:
        assert len(nomes_preenchidos) == len(parte.estimator.feature_importances_), (
            "Nomes após preenchimento e importâncias devem ter o mesmo tamanho."
        )
    classificadores[nome] = clf
    previsoes_modelos[nome] = clf.predict_proba(df_ml.iloc[test_idx])[:, 1]

comparacao_modelos = nbh.model_comparison_metrics(
    y_test, previsoes_modelos, p_true=p_true_test, include_oracle=True,
)
comparacao_modelos.insert(0, "n features", pd.Series(quantidades_features))
comparacao_modelos.insert(1, "ajuste (s)", pd.Series(tempos_ajuste))
print("Olhe a coluna 'Ordena bem? (AUC-ROC)': maior é melhor; nos erros ECE/Brier/log-loss, menor é melhor (mas veja o piso de ruído).")
print(comparacao_modelos.rename(columns={
    "n features": "Pistas selecionadas (features)", "ajuste (s)": "Tempo de ajuste (s)",
    "AUC-ROC": "Ordena bem? (AUC-ROC)", "AUC-PR (AP)": "Precisão na fila (AP)",
    "ECE": "Erro de promessa (ECE)", "Brier": "Erro par a par (Brier)",
    "log-loss": "Multa por surpresa (log-loss)",
    "MAE vs p* (nota)": "Distância da regra inventada (MAE vs p*)",
}).round(4).to_string())
gavetas_ece = 10
piso_ece = nbh.ece_noise_floor(p_true_test, n_bins=gavetas_ece)
print("Piso amostral de ECE (novos rótulos, média e intervalo de 95%):")
print(pd.DataFrame([piso_ece]).rename(columns={
    "mean": "Média", "ci_low": "Limite inferior", "ci_high": "Limite superior",
}).to_string(index=False))
for nome, ece in comparacao_modelos["ECE"].items():
    na_faixa = piso_ece["ci_low"] <= ece <= piso_ece["ci_high"]
    print(f"{nome}: ECE={br(ece, 4)} dentro da faixa de p*? {na_faixa}")
    assert na_faixa, f"{nome}: ECE fora da faixa; atualize a leitura do piso de ruído."
for metrica in ["MAE vs p* (nota)", "Brier", "log-loss"]:
    candidatos = comparacao_modelos.loc[list(previsoes_modelos), metrica]
    assert candidatos.idxmin() == "Platt (só a nota)", (
        f"O texto favorece Platt em {metrica}; atualize a leitura."
    )
confere("pares no teste único", len(y_test), 183, 0)
confere("gavetas de ECE", gavetas_ece, 10, 0)
confere("ECE do oráculo p*", comparacao_modelos.loc["p* (oráculo)", "ECE"], 0.0754, 0.00005)
from sklearn.metrics import roc_auc_score

auc_nota_platt_oraculo = pd.Series({
    "nota": roc_auc_score(y_test, nota_test),
    "Platt": roc_auc_score(y_test, previsoes_modelos["Platt (só a nota)"]),
    "p*": roc_auc_score(y_test, p_true_test),
}, name="AUC-ROC")
print("AUC(nota) = AUC(Platt) = AUC(p*):")
print(auc_nota_platt_oraculo.to_string())
assert auc_nota_platt_oraculo.nunique() == 1
bootstrap_ml = pd.DataFrame([
    {"Comparação": f"Platt − {nome}", "Métrica": metrica,
     **nbh.paired_bootstrap_diff(
         y_test, previsoes_modelos["Platt (só a nota)"], previsoes_modelos[nome],
         metric=metrica, n_boot=2000, seed=SEED,
     )}
    for nome in classificadores for metrica in ["auc", "brier"]
])
print("Bootstrap pareado: diferença Platt menos concorrente, IC de 95%:")
print(bootstrap_ml.rename(columns={
    "diff": "Diferença", "ci_low": "Limite inferior", "ci_high": "Limite superior",
}).to_string(index=False))
intervalos_ml = bootstrap_ml.set_index(["Comparação", "Métrica"])
for metrica in ["auc", "brier"]:
    intervalo_rf = intervalos_ml.loc[("Platt − Random Forest", metrica)]
    assert intervalo_rf["ci_low"] <= 0 <= intervalo_rf["ci_high"], (
        f"O texto diz que o IC de {metrica} entre Platt e RF inclui zero; atualize a leitura."
    )
assert intervalos_ml.loc[("Platt − XGBoost", "auc"), "ci_low"] > 0, (
    "O texto diz que o IC de AUC exclui zero e favorece Platt contra XGB; atualize a leitura."
)
assert intervalos_ml.loc[("Platt − XGBoost", "brier"), "ci_high"] < 0, (
    "O texto diz que o IC de Brier exclui zero e favorece Platt contra XGB; atualize a leitura."
)
hero_pos = df_ml.index.get_loc(hero_idx)
assert hero_pos in train_idx, "O texto informa herói no treino; atualize a leitura desta divisão."
if hero_pos in test_idx:
    pos_teste = int(np.flatnonzero(test_idx == hero_pos)[0])
    probabilidades_heroi = {
        nome: float(p[pos_teste]) for nome, p in previsoes_modelos.items()
    }
    origem_heroi = "Prova nova (held-out): herói no teste"
else:
    origem_heroi = "Herói visto no treino (in-sample): não é prova nova (held-out)"
    probabilidades_heroi = {
        "Platt (só a nota)": float(compute_p_cal(
            df_ml.iloc[[hero_pos]], method="platt", model=platt_holdout,
        ).iloc[0]),
        **{nome: float(clf.predict_proba(df_ml.iloc[[hero_pos]])[0, 1])
           for nome, clf in classificadores.items()},
    }
print(origem_heroi)
display(pd.Series(probabilidades_heroi).to_frame("probabilidade (" + origem_heroi + ")"))
PASSAPORTE["13.7 ML"] = {"origem": origem_heroi, **probabilidades_heroi}""",
    ),
    (
        "md",
        """#### Promessa cumprida ou apenas boa ordenação?

> 🧸 **Bem simples.** Uma gaveta com poucas fichas pode parecer melhor só por sorte.
> Até quem conhece a probabilidade verdadeira vê diferenças entre promessa e resultado.

> 🧒 **Um pouco mais.** Com cerca de 183 pares e 10 gavetas, até a probabilidade
> verdadeira p* tem ECE ≈ 0,0754. O código simula novos rótulos para mostrar o
> nível desse ruído de sorteio (~0,035–0,091). Os ECEs de XGB, RF, Platt e do
> próprio p* caem todos na faixa simulada para p*: o ECE não ordena estes modelos.
> A régua sem ruído de rótulo deste mundo inventado, a distância até p*, mostra
> o Platt mais perto da verdade; Brier e log-loss concordam neste teste.

> 📖 **Palavra nova: piso amostral de ECE.** É a oscilação que ainda aparece
> ao comparar a probabilidade verdadeira com um conjunto finito de respostas sorteadas.

**Vantagem por construção vale para as probabilidades (AUC, Brier, distância até p*);
ela não garante o menor custo sob travas e revisor suposto.** A tabela e as
conferências de custo abaixo sustentam a leitura: em vigilância, Platt tem o menor
custo esperado por par; em confirmação, RF fica ligeiramente abaixo de Platt,
dentro da variação entre sorteios; XGB é o pior nos dois modos. Se o ranking mudar,
é preciso rever essa leitura. Não é um achado sobre tamanho do treino nem
inferioridade geral do ML. Dada a nota, as demais pistas não informam TARGET
nas linhas aleatórias. Ordenar pela nota já é a melhor ordenação possível nesse
modelo gerador: o código confere AUC(nota) = AUC(Platt) = AUC(p*).
Os cenários com rótulos escolhidos à mão são a ressalva a essa construção.

> 🧸 **Bem simples.** Refaça a prova sorteando as mesmas fichas para os dois ajudantes.
> Assim a comparação não mistura a vantagem do ajudante com a sorte da prova.

> 🧒 **Um pouco mais.** O bootstrap pareado imprime diferenças Platt menos RF/XGB
> e intervalos de 95%: positivo favorece Platt na AUC; negativo favorece Platt no
> Brier. Se o intervalo inclui zero, este teste não separa claramente os modelos
> nessa medida. Ele reamostra pares, não pessoas nem novas populações.

**Lendo estes intervalos.** RF inclui zero tanto na AUC quanto no Brier:
a diferença não é distinguível do ruído nesta prova. XGB exclui zero nas duas
medidas, com sinais que favorecem Platt: maior AUC e menor Brier neste conjunto.
O código confere os limites para que essa leitura falhe se os intervalos mudarem.

> 📖 **Palavra nova: bootstrap pareado.** É sortear novamente as linhas do teste,
> mantendo juntas as respostas dos modelos para cada par.

> 🧸 **Bem simples.** Uma resposta melhor no exercício já estudado não vale como prova nova.
> É como reconhecer a página do caderno sem saber resolver outra parecida.

> 🧒 **Um pouco mais.** Se o herói estiver no treino, e o código informa que está,
> suas probabilidades estão
> rotuladas como "herói visto no treino": não são evidência de que o mais
> confiante aprendeu melhor.

> 📖 **Palavra nova: visto no treino (in-sample).** É uma previsão sobre uma ficha que já ajudou a ensinar o modelo.

<details>
<summary>📐 Para quem quer a matemática</summary>

AP é average precision, não a área trapezoidal da §13.4.
ECE compara médias de gavetas; Brier e log-loss confrontam probabilidades com cada resposta.
Brier também reflete a capacidade de distinguir as classes.
Gavetas vazias são omitidas; gavetas pequenas oscilam. Não escolhemos os controles
do modelo (hiperparâmetros) olhando este teste.

</details>

> 🎯 **Para levar.** O herói ilustra um palpite; as fichas novas é que sustentam a comparação.""",
    ),
    (
        "code",
        """fig, boletim_modelos = nbh.plot_model_comparison(
    y_test, previsoes_modelos, p_true=p_true_test,
)
plt.show()""",
    ),
    (
        "md",
        r"""#### Quais pistas as árvores mais usaram?

> 🧸 **Bem simples.** Quem fala mais na reunião não é necessariamente quem decide.
> Contar participações não explica por que o resultado aconteceu.

> 🧒 **Um pouco mais.** As árvores dão crédito às pistas que ajudaram nas perguntas.
> Cópias como `nota final` e `nota_final` dividem esse crédito: uma pode aparecer
> pouco só porque a outra foi usada em seu lugar. Não são evidências independentes.

> 📖 **Palavra nova: importância de variável.** É o crédito de uma pista nas divisões das árvores, não prova de causa.

**Exemplo pequeno.** Dois colegas repetindo a mesma informação podem dividir o agradecimento.
O gráfico mostra as 10 pistas com mais crédito médio entre os 5 ajustes internos.

<details>
<summary>📐 Para quem quer a matemática</summary>

RF soma reduções de impureza ponderadas pelas amostras; para proporções de classes
$p_k$, a impureza de Gini é $I=1-\sum_k p_k^2$.
Uma divisão recebe $\Delta I=I_{pai}-(n_e I_e+n_d I_d)/n_{pai}$,
ponderada pela fração de amostras que chega ao nó.
XGBoost usa ganho na função objetivo: melhoria da perda com a penalização do modelo.
Calculamos a média das importâncias normalizadas dos cinco estimadores internos,
não da sigmoide calibradora. Pistas correlacionadas dividem crédito e muitas
divisões possíveis podem favorecer uma pista. Não é importância por permutação no teste.

</details>

> 🎯 **Para levar.** Crédito no algoritmo não é causa no mundo real.""",
    ),
    (
        "code",
        """fig, importancias_top10 = nbh.plot_classifier_importances(classificadores)
plt.show()
print("Olhe a coluna 'Crédito nas perguntas (importância)': não é efeito causal.")
print(importancias_top10.rename(columns={
    "modelo": "Modelo", "feature": "Pista (variável)",
    "importância média": "Crédito nas perguntas (importância)", "folds": "Partes internas (folds)",
}).round(4).to_string(index=False))
for nome, grupo in importancias_top10.groupby("modelo"):
    confere(f"{nome}: pistas no gráfico de importância", grupo["feature"].nunique(), 10, 0)
assert importancias_top10["folds"].eq(5).all()""",
    ),
    (
        "md",
        """#### E quando a probabilidade entra na política de custos?

> 🧸 **Bem simples.** Volte ao posto de saúde e conte quem foi atendido sem ajuda.
> Um boletim melhor pode esconder uma fila maior de casos esperando revisão.

> 🧒 **Um pouco mais.** Repetimos os 5 sorteios por `COMPREC`, nos dois modos,
> com as mesmas pistas e preços. A comparação principal é o **custo esperado por
> par em todas as linhas do teste**, junto da sensibilidade esperada do sistema
> inteiro e da cobertura. Os dois primeiros supõem as taxas de erro do revisor da config.

**Cuidado com a seleção.** Cada modelo deixa pares diferentes pendentes.
Adiar os casos difíceis pode aumentar precisão, recall ou Fβ entre os decididos
sem melhorar o sistema. Mantemos essas métricas como diagnóstico, não como ranking.

⚠️ **Separação incompleta por dentro.** O revezamento da calibração interna não
respeita grupos (`COMPREC`): o mesmo candidato pode participar do estudo e da
conferência internos. Neste gerador o efeito é nulo porque o ID não carrega
informação; em dados reais, não. A prova externa continua separada por `COMPREC`.

Fixamos CPU em todos os ajustes de XGBoost para números reprodutíveis com estas versões das bibliotecas.

<details>
<summary>📐 Para quem quer a matemática</summary>

Reutilizamos `res`, `SEEDS_32`, `split_by="comprec"`, `test_size=0.3`,
`group_stratify=True` e MACD ativo. Não reduzimos as árvores dos padrões reais
de `evaluate_v3_dataframe`. Média e desvio-padrão excluem os pendentes das métricas
de acerto automático, mas a cobertura usa todas as linhas do teste.
Os cinco folds internos de `CalibratedClassifierCV` não são group-aware.
O custo por par soma os custos esperados de FP e FN após revisão e o custo das
revisões selecionadas, dividindo por todo o teste. É uma conta sob hipóteses,
não gasto observado em serviço. A fórmula assume erros do revisor constantes
dentro de cada banda, independentemente de p.

`run_v3` com `fit_ml_rf`/`fit_ml_xgb` aprende e prevê nas mesmas linhas: é visto
no treino (*in-sample*), por isso não usamos suas previsões nesta prova.
A YAML cita `base_model` (`models/rf_smote.pkl`, SMOTE) e `logit(p_base)`,
mas esse caminho não está implementado. SMOTE criaria exemplos artificiais da
classe menos frequente; aqui não há esse passo. RF usa `class_weight="balanced"`
e XGB usa `scale_pos_weight`: dão pesos diferentes aos exemplos existentes.

</details>

⚠️ **Receita não executada.** A YAML descreve um modelo-base já treinado e uma forma de criar exemplos artificiais dos pares raros (SMOTE). Nada disso roda aqui. E rodar `run_v3` com ML aprende e responde nas mesmas fichas, então não serve de prova nova.

> 🎯 **Para levar.** Compare custo e sensibilidade do sistema inteiro; a vantagem probabilística da nota neste gerador não garante menor custo sob travas e revisor suposto. Confira o ranking medido por modo, não generalize para dados reais.""",
    ),
    (
        "code",
        """resultados_politica = [res]
tempos_politica = {"platt (reutilizado)": tempo_platt_politica}
for metodo in ["ml_rf", "ml_xgb"]:
    inicio = perf_counter()
    resultados_politica.append(evaluate_v3_dataframe(
        df_ml, calibration=metodo, **args_avaliacao,
        classifier_config=ClassifierConfig(xgb_device="cpu", random_state=SEED),
    ))
    tempos_politica[metodo] = perf_counter() - inicio
res_comparacao_ml = pd.concat(resultados_politica, ignore_index=True)
res_ml = nbh.custo_esperado_por_par(res_comparacao_ml, cfg)
metricas_politica_sistema = ["custo_esperado_por_par", "exp_recall", "exp_fbeta", "auto_coverage"]
resumo_sistema_ml = res_ml.groupby(["mode", "calibration"])[metricas_politica_sistema].agg(["mean", "std"])
custos_medios_ml = resumo_sistema_ml[("custo_esperado_por_par", "mean")]
for modo, ordem_esperada in [
    ("vigilancia", ["platt", "ml_rf", "ml_xgb"]),
    ("confirmacao", ["ml_rf", "platt", "ml_xgb"]),
]:
    ranking = custos_medios_ml.loc[modo].sort_values().index.tolist()
    print(f"{modo}: ranking de custo (menor primeiro): {ranking}")
    assert ranking == ordem_esperada, "Ranking mudou: atualize a leitura de custo em §13.7 e §16."
gap_platt_rf = custos_medios_ml.loc[("confirmacao", "platt")] - custos_medios_ml.loc[("confirmacao", "ml_rf")]
soma_desvios = sum(
    resumo_sistema_ml.loc[("confirmacao", metodo), ("custo_esperado_por_par", "std")]
    for metodo in ["platt", "ml_rf"]
)
print(f"Confirmação: custo Platt − RF={br(gap_platt_rf, 4)}; soma dos desvios={br(soma_desvios, 4)}")
assert 0 < gap_platt_rf < soma_desvios, "A diferença deve ser menor que a soma dos desvios dos sorteios."
tabela_sistema_ml = pd.DataFrame(index=resumo_sistema_ml.index)
for metrica in metricas_politica_sistema:
    tabela_sistema_ml[metrica] = [
        f"{br(media, 4)} ± {br(desvio, 4)}" for media, desvio in zip(
            resumo_sistema_ml[(metrica, "mean")], resumo_sistema_ml[(metrica, "std")], strict=True,
        )
    ]
nomes_modelos = {"platt": "Platt (só a nota)", "ml_rf": "Random Forest", "ml_xgb": "XGBoost"}
print("Comparação principal em TODO o teste (média ± desvio-padrão dos sorteios):")
print(tabela_sistema_ml.rename(columns={
    "custo_esperado_por_par": "Custo esperado por par", "exp_recall": "Sensibilidade esperada do sistema",
    "exp_fbeta": "Fβ esperado do sistema", "auto_coverage": "Cobertura automática",
}).rename(index=nomes_modelos, level="calibration").rename_axis(index=["Modo", "Modelo"]).to_string())
resumo_politica_ml = res_comparacao_ml.groupby(["mode", "calibration"])[
    metric_cols_32
].agg(["mean", "std"])
tabela_politica_ml = pd.DataFrame(index=resumo_politica_ml.index)
for metrica in metric_cols_32:
    tabela_politica_ml[metrica] = [
        f"{br(media, 3)} ± {br(desvio, 3)}".replace(".", ",") for media, desvio in zip(
            resumo_politica_ml[(metrica, "mean")], resumo_politica_ml[(metrica, "std")],
        )
    ]
print("Diagnóstico entre os decididos: subconjuntos diferentes, não um ranking justo de modelos.")
print(tabela_politica_ml.rename(columns={
    "auto_precision": "Acerto entre aceitos e decididos (precisão automática)",
    "auto_recall": "Encontrados entre os decididos (recall automático)",
    "auto_fbeta": "Nota do modo entre os decididos (Fβ)",
    "auto_coverage": "Resolve sozinho (cobertura)",
}).rename(index={"platt": "Platt (só a nota)", "ml_rf": "Random Forest", "ml_xgb": "XGBoost"}, level="calibration")
    .rename_axis(index=["Modo", "Modelo (calibração)"]).to_string())
print("Tempos da política (s):")
display(pd.Series(tempos_politica).to_frame("tempo (s)"))
for nome in classificadores:
    delta = comparacao_modelos.loc[nome] - comparacao_modelos.loc["Platt (só a nota)"]
    print((f"{nome} menos Platt no teste: ordenação (AUC-ROC) {delta['AUC-ROC']:+.4f}; "
           f"erro de promessa (ECE) {delta['ECE']:+.4f}; erro par a par (Brier) {delta['Brier']:+.4f}").replace(".", ","))
tempo_secao_ml = perf_counter() - inicio_secao_ml
print(f"Tempo adicional medido da §13.7: {br(tempo_secao_ml, 2)} s")""",
    ),
    (
        "md",
        """#### 🎯 Resumo da comparação

São duas perguntas, em duas tabelas pequenas. **Teste único (183 pares)** compara
probabilidades, incluindo p* como oráculo. **5 sorteios (média)** compara a política
em todos os pares do teste: custo esperado por par, sensibilidade esperada do sistema
e cobertura. Na segunda, cada célula traz vigilância / confirmação, nessa ordem.
Não misturamos a prova única com médias de divisões diferentes.
""",
    ),
    (
        "code",
        """print("🎯 Teste único (183 pares)")
resumo_teste_unico = comparacao_modelos[["AUC-ROC", "ECE", "Brier", "MAE vs p* (nota)"]].rename(columns={
    "MAE vs p* (nota)": "Distância até p*",
}).rename_axis("Modelo")
print(resumo_teste_unico.to_string())
print("🎯 5 sorteios (média); valores: vigilância / confirmação")
linhas_resumo_sorteios = []
for metodo, nome in nomes_modelos.items():
    linha = {"Modelo": nome}
    for metrica, rotulo in [
        ("custo_esperado_por_par", "Custo esperado por par (vigilância / confirmação)"),
        ("exp_recall", "Sensibilidade esperada do sistema (vigilância / confirmação)"),
        ("auto_coverage", "Cobertura (vigilância / confirmação)"),
    ]:
        linha[rotulo] = " / ".join(
            br(resumo_sistema_ml.loc[(modo, metodo), (metrica, "mean")], 4)
            for modo in ["vigilancia", "confirmacao"]
        )
    linhas_resumo_sorteios.append(linha)
resumo_cinco_sorteios = pd.DataFrame(linhas_resumo_sorteios).set_index("Modelo")
print(resumo_cinco_sorteios.to_string())
tempo_secao_ml = perf_counter() - inicio_secao_ml
print(f"Tempo total medido da §13.7: {br(tempo_secao_ml, 2)} s")""",
    ),
]


FASE_3_3 = [
    (
        "md",
        """## 14. Revisão LLM (simulador determinístico)

🧒 Quem ficou na fila recebe uma segunda opinião. Aqui ela é **simulada**, offline:
`llm_review_stub` vê `TARGET` e sorteia erros com as taxas por banda da config.
Semente fixa e ordem fixa das linhas tornam o resultado reproduzível, mas
**não medem a capacidade de um LLM**. Não há API nem inteligência artificial real.

A config descreve `dual_agent_plus_arbiter`: dois revisores independentes e um
árbitro para discordâncias. O simulador não implementa esse protocolo; só ilustra
seu efeito hipotético. Aplicamos a simulação aos `LLM_REVIEW` de `out_vig`,
na **rota A**, usando `cfg.llm_review.error_rates_by_band`.

⚠️ **É circular.** O acerto impresso só devolve, com ruído de sorteio, as taxas
de erro que nós configuramos. Não é uma descoberta sobre qualidade de revisão.
A conta esperada também supõe que, dentro de uma banda, o erro do revisor não
depende de p; uma revisão real pode errar mais justamente nos pares difíceis.""",
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
print(f"Acurácia do revisor simulado (vs TARGET): {br(100 * (acerto_stub), 1)}%")""",
    ),
    (
        "md",
        """### 14.1 Métricas finais incluindo a revisão simulada

Os automáticos mantêm a ação; os pendentes recebem a decisão simulada.
Calculamos precisão, recall e **F1** em todos os pares da rota A.
**F1 aqui só como resumo; o modo vigilância usaria F₂.** A cobertura
automática anterior informa quanto foi resolvido sem revisão; não é uma métrica
final de acerto. Esta tabela não é avaliação held-out nem validação externa.""",
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
print(tabela_final.round(4).to_string(index=False))""",
    ),
    (
        "md",
        """### 14.2 Passaporte final do herói

Reunimos uma vez as anotações feitas desde a entrada. Cada linha guarda o que
um estágio acrescentou, sem apagar as etapas anteriores. A última registra
a decisão da revisão **simulada** e o gabarito, não uma nova probabilidade.""",
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
PASSAPORTE["14. Revisão simulada"] = {
    "ação antes": hero_action_vig, "decisão final": hero_final,
    "TARGET": int(out_vig.loc[hero_idx, "TARGET"]),
}
passaporte_final = pd.json_normalize(PASSAPORTE, sep=" / ").T.rename(columns={0: "valor"})
passaporte_final.index.name = "etapa / campo"
print(passaporte_final.to_string())""",
    ),
]


FASE_INTERATIVO: list[tuple[str, str]] = [
    (
        "md",
        """## 15. Painel interativo

Os seis controles ficam ao lado do conceito, com visão estática sempre disponível.
Para interagir, use Jupyter com `ipywidgets` de `requirements/notebook.txt`.

| Controle | Seção | Que pergunta responde? |
|---|---|---|
| W5: pesos e janelas MACD | §7.4 | Os pesos do nome mudam o placar; por que as janelas MACD quase não mudam aqui? |
| W1: Platt (a, b) | §9.6 | Como inclinação e intercepto mudam a curva, o ponto s* e seu boletim? |
| W3: âncoras | §9.8 | O que acontece se os pontos de referência estiverem errados? |
| W4: bandas e guardrails | §10.1 | Mudar fronteiras e regras muda bandas, probabilidades ou ações? |
| W2: custos FP/FN/LLM | §11.4 | Quando aumentar FP traz novas revisões e encarecer a revisão devolve decisões automáticas? |
| W6: vagas de revisão | §11.6 | Quem recebe a última vaga de revisão e quem fica de fora? |

### Desafios para a aula

1. Dobre a inclinação no W1 sem mudar o intercepto. s* = −b/a também muda:
   explique por que o meio da curva não fica parado.
2. Aproxime os alvos do W3 das médias verdadeiras medidas. O boletim melhora?
3. Baixe `nota_always_match` para 9 no W4. Quais outras pistas ainda são exigidas?
4. Aumente FP para 2, 3 e 5 vezes o padrão no W2 e observe `near_high`/`high`.
   Encareça a revisão: quais pedidos só por EVR somem? Explique o papel das travas
   e das vagas. Depois zere as vagas de revisão no W6 e separe escolha
   base, pedido de revisão e ação final ao explicar o resultado.
5. Mova os pesos do nome no W5 e compare com mover só as janelas MACD:
   as janelas quase não importam neste gerador, com diferenças de datas em dias próximos ou anos.

⚠️ Ajustar olhando o teste é exploração: não transforme o melhor ajuste visto
nos controles em uma alegação de validação independente.""",
    ),
]


FASE_4_1: list[tuple[str, str]] = [
    (
        "md",
        """## 16. Conclusões e limitações

### 16.1 O que demonstramos

Seguimos pistas cruas → features → bandas → probabilidade → guardrails → triagem.
Refizemos contas do herói, derivamos Platt, comparamos perdas e reconciliamos
**exatamente** a rota A com `run_v3`. A rota B mediu calibração e desempenho fora
do treino; seis controles tornaram visíveis as hipóteses. O passaporte fechou
com uma revisão simulada, não com um LLM real.

### 16.2 Limitações (leia antes de generalizar)

Isto é **demonstração didática**, não evidência de desempenho em bases reais.

- **Dados sintéticos e verdade logística por construção.** A verdade é uma sigmoide
  de uma reta na nota, exatamente a família do Platt. Recuperar p* verifica o ajuste
  num modelo bem especificado, em grande parte por construção, não valida uso real.
  Os cenários narrativos têm rótulos escolhidos à mão, não sorteados segundo p*.
- **Rotas distintas.** `run_v3` ajusta e pontua nas mesmas linhas. Só a rota B estima
  desempenho fora do treino neste sintético. A revisão da §14 volta à rota A.
- **IDs compartilhados.** A §13.3 mede linhas com COMPREC/REFREC repetidos, não apenas
  IDs únicos. As frações vêm dos ajustes do gerador. Os IDs são independentes de
  tudo: vazamento via ID é impossível aqui e diferenças entre divisões são ruído
  de partição. Isso não reproduz comparações dependentes da mesma pessoa em bases reais.
- **Âncoras só no notebook.** A biblioteca não implementa `anchor_platt`; executa
  Platt supervisionado global. Os alvos da config não batem com as médias de p*
  medidas na §9.8. A demo não corrige essa hipótese.
- **Guardrails, config × código.** Os cortes descritos no texto da YAML não batem
  com os parâmetros executados, como conferimos na §10.
  `risky_if` (`grey_mother_missing`, `low_name_and_low_dob`) não está implementado.
- **Custos escolhidos, não estimados.** Os preços de FP, FN e revisão são escolhas
  da config sem fonte empírica. O custo esperado por par é condicionado a elas,
  não uma estimativa monetária validada de um serviço de saúde.
- **Métricas automáticas medem sobretudo travas e guardrails.** Precisão, recall
  e Fβ entre os decididos excluem pendentes e mudam de subconjunto entre modelos.
  Prefira a comparação por custo esperado por par, sensibilidade esperada do
  sistema inteiro e cobertura; as expectativas supõem o revisor da config.
- **ECE tem ruído amostral.** Mesmo p* tem ECE diferente de zero no teste finito.
  Diferenças abaixo dessa escala de ruído não sustentam uma promessa de melhora.
- **LLM simulado e circular.** O simulador usa TARGET e taxas de erro injetadas:
  o acerto só devolve o que configuramos, com ruído. A fórmula esperada assume
  erro independente de p dentro da banda. Não mede um LLM nem implementa dois
  agentes e árbitro. Uso real exigiria validação independente, dossiês protegidos
  e cuidado com dados pessoais.
- **ML fez esta prova, não uma prova em pessoas reais.** A §13.7 testa o aprendiz
  `GZCMDClassifier` com Random Forest/XGBoost sem as anotações da aula.
  Vantagem por construção vale para as probabilidades (AUC, Brier, distância até p*),
  não garante menor custo sob travas e revisor suposto. No ranking medido e
  conferido na §13.7, Platt tem o menor custo em vigilância; em confirmação,
  RF fica ligeiramente abaixo de Platt, dentro da variação entre sorteios;
  XGB é o pior nos dois modos. Se esse ranking mudar, reveja esta leitura.
  Dada a nota, as outras pistas não carregam informação adicional sobre TARGET nas linhas aleatórias.
  Nenhum modelo pode melhorar a probabilidade verdadeira p*; Platt já tem sua
  família correta. Não é um achado sobre pouco treino nem inferioridade geral do ML.
  A regra de features da biblioteca inclui datas cruas como números e aliases
  duplicados que dividem a importância; uma alternativa usaria só escores agregados.
  O revezamento interno não separa candidatos por grupo, embora o teste externo
  separe. Aqui o ID é independente; em bases reais essa falta de separação importa.
  Se o herói estiver no treino, e o código informa que está, ele não comprova
  desempenho fora do treino. A detecção de CUDA foi melhorada (há uma sonda do
  XGBoost); o notebook fixa CPU. Os números são reprodutíveis com estas versões das bibliotecas.
  A igualdade exata da §12 verifica apenas a rota Platt.
  O modelo-base (`base_model`), SMOTE e `logit(p_base)` da YAML não rodam aqui;
  `run_v3` com `fit_ml_*` aprende e responde nas mesmas fichas, não numa prova nova.

### 16.3 Recap final

O valor não é "acertou X%", mas **como cada decisão nasce, quanto custa e sob
quais hipóteses faz sentido**. Separar implementação, simulação e validação é
mais importante do que exibir uma métrica bonita.""",
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
            "title": "GZ-CMD++ v3: Passo a Passo",
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
    for index, cell in enumerate(cells):
        cell.id = f"c{index:03d}"
        cell.metadata = {}
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
