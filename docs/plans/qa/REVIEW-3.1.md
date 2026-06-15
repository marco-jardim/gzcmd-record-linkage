# Senior QA Review — Fase 3.1 (Reconciliação manual rota A × `run_v3`)

**Revisor:** Orquestrador (Opus, self:opus). **Veredito: APROVADA.**

## Escopo
Seção 12 do notebook (`## 12. Reconciliação com run_v3`) + 3 testes (TST3.1.a/b/c) em `tests/test_notebook_pipeline.py`. Fecha CA-G3.

## Evidência objetiva (canário independente, seed=42, n=600, scenarios='all', 607 linhas)
- `band` idêntico: **True**
- `p_cal` diferença máxima: **0.0** (bit-a-bit; bem abaixo do atol≤1e-9 exigido)
- `action` idêntico: **True** (0 divergências)
- `RunSummary.rows`=607, `sum(actions)`=607 (fecha), actions={LLM_REVIEW:290, MATCH:194, NONMATCH:123}
- Platt manual == `p_cal_params` do `run_v3` (intercept=-5.4311, slope=0.9008)

## Eixos de revisão
1. **Correção funcional:** rota A reproduz `run_v3` exatamente — confirmado por execução real. ✅
2. **Determinismo:** Platt é Newton-Raphson sem RNG; params da config == defaults das funções; reconciliação exata e reprodutível. ✅
3. **Fidelidade científica:** distinção rota A (in-sample, reconcilia) × rota B (held-out, difere de propósito) explícita no markdown. R-13 (XGBoost não-determinístico → reconciliação só qualitativa) declarado. ✅
4. **Cobertura de testes:** TST3.1.a (igualdade exata band/p_cal/action), TST3.1.b (RunSummary consistente), TST3.1.c (XGBoost qualitativo, R-13). Todos verdes. ✅
5. **Qualidade de código:** `ruff check` + `ruff format --check` limpos nos 3 arquivos. ✅
6. **Didática (DF-3):** toda célula de código precedida de markdown explicativo (objetivo→intuição→ação→recap). ✅

## Achados
- 🔴 nenhum · 🟡 nenhum · 🟢 nenhum.

## Verificação
- Suíte completa: **88 passed** (era 85; +3).
- NBEXEC (notebook ponta-a-ponta) verde (~45s); notebook regenerado com 81 células.
- Aviso xgboost CUDA→CPU (`error_msg.cc`) é benigno (fallback de device), não é falha.
