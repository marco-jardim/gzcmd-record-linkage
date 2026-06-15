# Notebook didático — GZ-CMD Passo a Passo

Material de apresentação (PT-BR, público técnico/acadêmico) que reproduz, passo a
passo, o pipeline do motor de decisão **GZ-CMD++ v3** (`gzcmd_record_linkage`)
sobre um **dataset 100% sintético** gerado para a apresentação.

> Status: **em construção** (ver `docs/plans/notebook-gzcmd-passo-a-passo.md`).

## Conteúdo desta pasta

| Arquivo | Papel |
|---------|-------|
| `gzcmd_passo_a_passo.ipynb` | Notebook da apresentação (em construção). |
| `synthetic_data.py` | Gerador de dataset sintético (importável/testável). |
| `nb_helpers.py` | Helpers de plot, métricas de calibração (ECE/Brier) e stub determinístico de LLM. |

O dataset gerado é salvo em `../data/synthetic/comparador_sintetico.csv`.

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

## Como regenerar os dados sintéticos

O notebook regenera os dados na primeira célula de setup. Para gerar fora do
notebook:

```pwsh
python -c "from synthetic_data import generate_comparador; generate_comparador(seed=42)"
```

(Requer `notebooks/` no `PYTHONPATH`; ao rodar via pytest isso já está
configurado em `pyproject.toml`.)

## Como executar o notebook de ponta a ponta (headless)

```pwsh
papermill notebooks/gzcmd_passo_a_passo.ipynb docs/plans/qa/gzcmd_passo_a_passo.executed.ipynb
# ou
jupyter nbconvert --to notebook --execute notebooks/gzcmd_passo_a_passo.ipynb
```

## Testes relacionados

```pwsh
pytest -q
pytest -q --cov=synthetic_data --cov=nb_helpers --cov-report=term-missing
```
