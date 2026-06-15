# Senior QA Review — Fase 4.2 (Reprodutibilidade + README + artefato)

**Veredito:** ✅ APROVADA

## Escopo revisado
- T4.2.1 — Seeds fixas e reexecução limpa headless.
- T4.2.2 — `notebooks/README.md` finalizado.
- T4.2.3 — Artefato de evidência de execução.

## Evidências

| Item | Evidência |
|------|-----------|
| **Execução limpa ponta-a-ponta** | `jupyter nbconvert --to notebook --execute` retornou **exit 0**; 513.892 bytes escritos em `docs/plans/qa/gzcmd_passo_a_passo.executed.ipynb`. Nenhuma célula com erro. |
| **Determinismo / seeds** | `SEED=42` fixo no setup; `SEEDS_32=[42,123,456,789,2024]` na avaliação; gerador e stub LLM determinísticos (verificados em fases anteriores). Sem dependência de relógio/rede. |
| **README final** | Reescrito: papéis dos 4 arquivos (incl. `build_notebook.py` como fonte da verdade), instalação A/B, abrir/executar (jupyter lab / nbconvert / papermill), regenerar dados, **regenerar notebook**, comandos de teste (`pytest -q`, `-m notebook`, cobertura). |
| **Artefato (DEC-03)** | `gzcmd_passo_a_passo.executed.ipynb` (507.8 KB < 1 MB) em `docs/plans/qa/`; **não versionado** (`.gitignore` ignora `*.executed.ipynb`) — mantido como evidência local. |
| **TST4.2.b — ruff** | `ruff check` → All checks passed!; `ruff format --check` → 16 files already formatted. |
| **Suíte completa** | `pytest` → **99 passed**, 2 warnings (xgboost CUDA→CPU, benigno) em 48.69s. |

## Eixos da rubrica
- **Reprodutibilidade:** notebook executa do zero em ambiente headless sem erro; autossuficiente quanto a `sys.path`. ✅
- **Qualidade de código:** ruff limpo (check + format). ✅
- **Higiene de dados:** dataset 100% sintético; artefato executado fora do git. ✅
- **Não-poluição da lib:** nenhuma mudança em `src/`. ✅
- **Clareza didática:** README cobre instalar/executar/regenerar/testar. ✅

## Achados
Nenhum 🔴/🟡/🟢.

## Critérios de Aceitação
- **CA-G1** (executa ponta-a-ponta em ambiente limpo via nbconvert/papermill): ✅ satisfeito.
- **CA-G7** (ruff limpo; suíte verde): ✅ satisfeito (cobertura agregada validada no fechamento global, Fase 4.3).

**DoD da fase:** cumprido. Pronto para Fase 4.3 (QA Global + DoD final).
