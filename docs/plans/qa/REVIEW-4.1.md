# Senior QA Review — Fase 4.1 (Narrativa final + figuras + conclusões/limitações)

**Revisor:** Orquestrador (Opus, `[self:opus]`). **Veredito:** ✅ APROVADA.

## Escopo da fase
- T4.1.1 — Polimento de narrativa/figuras + correção do backlog cosmético 🟢 acumulado.
- T4.1.2 — Seção de **Conclusões e Limitações** (dados sintéticos, LLM simulado, in-sample×held-out, config×código).
- TST4.1.NBEXEC + TST4.1.a (figuras com título/eixos).

## Correções aplicadas
1. **LaTeX da seção 11 (custo esperado)** — `c\_{fp}`/`loss\_match` (que renderizavam literais com underscore escapado) reescritos com matemática correta: `\ell_{\text{match}}`, `\ell_{\text{non}}`, `\ell_{\text{llm}}`, `c_{fp}`, `c_{fn}`, `c_{llm}`, `e_{fp}`, `e_{fn}`, `\ell_{\min}`, `\text{evr}`. Subscritos agora renderizam no MathJax.
2. **Fase 2.3 (cosmético)** — `value_counts().reindex(ordem)` agora usa `fill_value=0` + `.astype(int)`; bandas vazias mostram `0` em vez de `NaN`.
3. **Fase 2.2 (cosmético)** — `df[flags].apply(lambda s: s.astype(float))` simplificado para `df[flags].astype(float)` (mesmo resultado, mais legível).

## Adições
- **Seção 15 — Conclusões e limitações** (markdown): objetivo de aprendizagem; 15.1 síntese dos 8 estágios; 15.2 limitações honestas enumeradas (dados sintéticos; in-sample×held-out R-10; config×código R-11; vazamento por grupo negligenciável-por-construção; LLM stub R-05; XGBoost não-determinístico R-13; estritura da âncora de guardrail); 15.3 recap final.
- **TST4.1.a** (`tests/test_notebook_execution.py::test_tst_4_1_a_figuras_tem_titulo_e_legenda`): heurística via `nbformat` — toda célula de plotagem deve ter título e rótulos de ambos os eixos.
- Correção decorrente: barra multi-seed (seção 13) ganhou `set_xlabel("Modo de operação")`.

## Eixos da rubrica (Seção 4)
- **Correção funcional:** fixes preservam comportamento; conclusões fiéis ao código. ✅
- **Cobertura:** +1 teste (figuras). ✅
- **Determinismo:** inalterado. ✅
- **Fidelidade científica:** limitações explícitas e quantificadas; sem afirmação enganosa. ✅
- **Qualidade de código:** `ruff check` + `format --check` limpos. ✅
- **Didática:** seção 15 com objetivo + recap; matemática da seção 11 agora renderiza. ✅
- **Não-poluição da lib:** apenas `notebooks/` e `tests/` alterados. ✅

## Evidência
- `pytest` (suíte completa): **99 passed** (era 98, +1). NBEXEC verde (103 células).
- `ruff check` / `ruff format --check`: limpos em `notebooks/build_notebook.py` + `tests/test_notebook_execution.py`.

## Achados
Nenhum 🔴/🟡. 🟢 backlog acumulado **zerado** nesta fase.
