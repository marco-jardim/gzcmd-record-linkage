# Notebook didático — GZ-CMD Passo a Passo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marco-jardim/gzcmd-record-linkage/blob/master/notebooks/gzcmd_passo_a_passo.ipynb)

Material de apresentação (PT-BR, público técnico/acadêmico) que reproduz, passo a
passo, o pipeline do motor de decisão **GZ-CMD++ v3** (`gzcmd_record_linkage`)
sobre um **dataset 100% sintético** gerado para a apresentação.

O notebook percorre os oito estágios do pipeline — carga + feature engineering,
atribuição de bandas, calibração de Platt (in-sample × held-out), guardrails,
política de decisão por custo esperado nos dois modos (`vigilancia`/`confirmacao`),
reconciliação exata com `run_v3`, avaliação held-out multi-seed (PR/ROC + custo
vs. limiar) e revisão LLM simulada — cada um com explicação, matemática e
visualização.

## Conteúdo desta pasta

| Arquivo | Papel |
|---------|-------|
| `gzcmd_passo_a_passo.ipynb` | Notebook da apresentação. **Arquivo gerado** — não editar à mão. |
| `build_notebook.py` | **Fonte da verdade** do notebook: monta o `.ipynb` célula a célula via `nbformat`. Edite aqui e regenere. |
| `synthetic_data.py` | Gerador de dataset sintético (importável/testável), com posterior verdadeira `p_true` (DEC-06). |
| `nb_helpers.py` | Métricas de calibração (`expected_calibration_error`, `brier_score`) e stub determinístico de revisão LLM (`llm_review_stub`). |

O dataset gerado é salvo (e versionado) em
`../data/synthetic/comparador_sintetico.csv`.

## Rodar no Google Colab

Clique no selo **Open in Colab** acima (ou abra o `.ipynb` via *File → Open
notebook → GitHub*). **Não é preciso nenhum passo manual de setup:** a primeira
célula do notebook (*bootstrap*) detecta que está no Colab — onde o repositório
não está presente — e automaticamente:

1. **clona** o repositório (`git clone --depth 1`);
2. **instala** a biblioteca `gzcmd_record_linkage` via `pip`;
3. coloca os módulos auxiliares (`synthetic_data`, `nb_helpers`) no `sys.path`;
4. aponta a pasta de dados para `data/synthetic/`, que **já contém o CSV
   versionado**.

Depois é só executar as células de cima para baixo (*Runtime → Run all*).

> **Por que isso importa?** Se você abrir o `.ipynb` solto (sem o repositório), as
> bibliotecas e os dados não existem no ambiente do Colab — daí a clássica "pasta
> de dados vazia" e o erro `ModuleNotFoundError: synthetic_data`. O *bootstrap*
> resolve os dois problemas de uma vez. A primeira execução leva ~1 min (clone +
> `pip install`); execuções seguintes reutilizam o que já foi baixado.

## Pré-requisitos

- Python ≥ 3.10 (testado em 3.13).
- A biblioteca `gzcmd_record_linkage` instalada (editable recomendado).

## Instalação do ferramental (pwsh / Windows)

A biblioteca e o ferramental de notebook são instalados separadamente — o
ferramental **não** entra nas dependências do pacote publicado.

### Opção A — ambiente virtual dedicado (recomendado para reprodutibilidade)

```pwsh
# A partir da raiz do repositório (D:\git\gzcmd-record-linkage)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m pip install -r requirements/notebook.txt
```

### Opção B — ambiente já existente (conda/base)

Se a biblioteca já está instalada em modo editable no seu ambiente atual,
basta adicionar o ferramental de notebook:

```pwsh
python -m pip install -r requirements/notebook.txt
```

## Como abrir e executar o notebook

Para exploração interativa:

```pwsh
jupyter lab notebooks/gzcmd_passo_a_passo.ipynb
```

Para execução de ponta a ponta sem interface (headless), útil como teste de
reprodutibilidade e para gerar o artefato de evidência:

```pwsh
# Artefato executado (não versionado — ver DEC-03 / .gitignore)
jupyter nbconvert --to notebook --execute `
  notebooks/gzcmd_passo_a_passo.ipynb `
  --output ../docs/plans/qa/gzcmd_passo_a_passo.executed.ipynb

# ou via papermill
papermill notebooks/gzcmd_passo_a_passo.ipynb `
  docs/plans/qa/gzcmd_passo_a_passo.executed.ipynb
```

O notebook é **autossuficiente** quanto ao `sys.path` (a primeira célula localiza
a raiz do repositório), portanto não exige `PYTHONPATH` ao rodar em um kernel
headless.

## Como regenerar o notebook (após editar `build_notebook.py`)

O `.ipynb` é **gerado**; a fonte da verdade é `build_notebook.py`. Após qualquer
alteração de conteúdo:

```pwsh
python notebooks/build_notebook.py
```

Isso reescreve `notebooks/gzcmd_passo_a_passo.ipynb` a partir das fases em
`ALL_PHASES`.

## Como regenerar os dados sintéticos

O notebook regenera os dados na célula de setup. Para gerar fora do notebook
(requer `notebooks/` no `PYTHONPATH` — já configurado no pytest via
`pyproject.toml`):

```pwsh
$env:PYTHONPATH = "src;notebooks"
python -c "import synthetic_data as s; ds = s.generate_comparador(seed=42, n_pairs=600, scenarios='all'); s.to_comparador_csv(ds.frame, 'data/synthetic/comparador_sintetico.csv')"
```

## Testes relacionados

```pwsh
# Suíte completa
pytest -q

# Apenas os testes de execução do notebook (mais lentos)
pytest -q -m notebook

# Cobertura dos módulos de apoio do notebook
pytest -q --cov=synthetic_data --cov=nb_helpers --cov-report=term-missing
```
