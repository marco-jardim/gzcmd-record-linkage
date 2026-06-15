# Senior QA Review — WAVE 0 / Fase 0.1 (Preparação do Ambiente)

> Revisor: orquestrador `[self:opus]` (Opus faz o QA localmente, conforme D6).
> Data: execução contínua do plano.

## Escopo revisado
- `requirements/notebook.txt` (novo)
- `tests/test_env_smoke.py` (novo)
- `notebooks/README.md` (rascunho)
- `docs/plans/qa/baseline.md` (novo)
- Diretórios criados: `notebooks/`, `data/synthetic/`, `docs/plans/qa/`

## Avaliação por eixo da rubrica (Seção 4)

| Eixo | Resultado |
|------|-----------|
| 1. Correção funcional | ✅ Smoke test importa todos os módulos públicos; `requirements/notebook.txt` no formato correto e instalável. |
| 2. Cobertura de testes | ✅ (fase de ambiente) Smoke test cobre 100% dos módulos públicos por importação. Sem código de produção novo a cobrir. |
| 3. Determinismo | ✅ Teste de importação é determinístico; sem relógio/rede. |
| 4. Fidelidade científica | N/A nesta fase. |
| 5. Qualidade de código | ✅ `ruff check` + `ruff format --check` limpos no arquivo novo. |
| 6. Clareza didática | ✅ `README.md` em PT-BR explica instalação (venv e ambiente existente), regeneração e execução headless. |
| 7. Higiene de dados | ✅ Nenhum dado real; dataset sintético ainda não gerado. |
| 8. Não-poluição da lib | ✅ `pyproject.toml` **não** foi alterado; tooling de notebook isolado em `requirements/notebook.txt`. |
| 9. Rigor estatístico | N/A nesta fase. |
| 10. Andaime didático | N/A nesta fase (sem notebook ainda). |

## Achados e correções

- 🟡 **F-0.1-1 — Smoke test com lista de módulos hardcoded.** A primeira versão
  enumerava os módulos manualmente; um módulo novo no pacote não seria coberto.
  **Correção aplicada:** descoberta dinâmica via `pkgutil.iter_modules`, com
  exclusão justificada de `__main__` e um teste-âncora que falha se a varredura
  vier vazia/incompleta. Re-rodado: **20 passed**.
- 🟢 **F-0.1-2 — `pyright` ausente.** Pacote pip exige Node; não instalado.
  Não-bloqueante — a rubrica (eixo 5) usa `ruff` como gate. Registrado no
  `baseline.md` como opcional. (Backlog, não trivial → não corrigido agora.)

## Evidência
- `pytest -q` (suíte completa): **37 passed, 1 warning in ~30s** (17 originais + 20 smoke).
- `pytest tests/test_env_smoke.py`: **20 passed**.
- `ruff check` / `ruff format --check`: limpos.
- Instalação: `papermill 2.7.0`, `nbformat 5.10.4`, `nbconvert 7.17.1`, `ipywidgets 8.1.8`.

## Critério de Aceitação da fase
✅ `requirements/notebook.txt` instala sem erro; `pytest` verde; CLIs respondem;
`import matplotlib, papermill` OK.

## DoD da fase
✅ Comandos de setup documentados em `notebooks/README.md` (rascunho).
✅ Baseline registrado em `docs/plans/qa/baseline.md`.

**Veredito:** Fase 0.1 **APROVADA**. Sem 🔴/🟡 pendentes.
