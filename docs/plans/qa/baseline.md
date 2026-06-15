# Baseline do Ambiente — WAVE 0 / Fase 0.1

Registro do estado do repositório **antes** de iniciar a implementação do
notebook didático. Serve de referência para detectar regressões (PF-3).

## Ambiente

| Item | Valor |
|------|-------|
| Plataforma | win32 |
| Shell | pwsh (PowerShell) |
| Python | 3.13.11 (miniconda base) |
| pip | 26.0.1 |
| Repositório | `D:\git\gzcmd-record-linkage` |
| Branch | `master` |
| Último commit | `1c16351 feat: add XGBoost as default classifier with RF as alternative (v0.2.0)` |

## Pacote sob teste

- `gzcmd-record-linkage` **0.2.0**, instalado em modo **editable** apontando para
  `D:\git\gzcmd-record-linkage` (não foi necessário reinstalar).
- `import gzcmd_record_linkage` → OK.

## Ferramentas de desenvolvimento

| Ferramenta | Estado |
|-----------|--------|
| ruff | 0.15.14 ✅ |
| pytest | ✅ (8.x) |
| pytest-cov | ✅ |
| pyright | ❌ não instalado no ambiente (pacote pip exige Node). **Não-bloqueante**: a rubrica de QA (Seção 4, eixo 5) usa `ruff` como gate de lint/format. Documentado como item opcional. |
| matplotlib | 3.10.8 ✅ |
| papermill / nbformat / nbconvert | ❌ ausentes → instalados via `requirements/notebook.txt` nesta fase |

## CLIs

- `gzcmd --help` → OK (subcomandos: `run`, `fit-calibration`, `eval`).
- `python -m gzcmd_record_linkage --help` → OK (mesma saída).

## Baseline de testes (`pytest -q`)

- **17 testes — todos PASSARAM (verde).**
- Aviso não-fatal: `xgboost` emite `UserWarning` sobre fallback de device
  (CUDA→CPU / DMatrix). Não é falha; é diagnóstico de hardware.

### Módulos de teste existentes
- `tests/test_bands.py`
- `tests/test_classifier.py`
- `tests/test_cli.py`
- `tests/test_guardrails.py`
- `tests/test_loader.py`

## Mudanças não-commitadas pré-existentes (NÃO originadas por este plano)

No início da Fase 0.1 o working tree já continha modificações não-commitadas,
**alheias** ao plano do notebook:

- `src/gzcmd_record_linkage/classifier.py` (+33 linhas)
- `tests/test_classifier.py` (+42 linhas)

**Decisão (D1):** essas mudanças **não serão tocadas nem commitadas** por este
trabalho. Os commits do notebook stage **apenas** arquivos relacionados ao
plano. Diretórios `docs/` e `dist/` estavam *untracked* (`dist/` é artefato de
build e permanecerá ignorado).

## Decisão de ambiente registrada (D1)

**Não foi criada uma `.venv` dedicada.** A biblioteca `gzcmd` já está instalada
em modo editable no ambiente conda ativo (Python 3.13.11 ≥ 3.10), com
`matplotlib` presente. Criar uma venv nova forçaria reinstalação pesada de
numpy/pandas/scikit-learn/scipy/xgboost e a ativação não persiste entre chamadas
de shell não-interativas. O resultado funcional é idêntico. Os comandos de venv
ficam **documentados** em `notebooks/README.md` (Opção A) para o usuário final.
