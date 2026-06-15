# Senior QA Review — Fase 2.6 (Política de Decisão / Triage)

**Revisor:** Orquestrador (Opus, `[self:opus]`). **Veredito: APROVADA.**

## Escopo
Adição da seção `## 11. Política de decisão (triagem)` ao notebook, executando a triagem
real (`build_engine_from_config` + `PolicyEngineV3.triage`) nos modos `vigilancia` e
`confirmacao`, com explicação do custo esperado, tabela comparativa, gráfico de barras
agrupadas, tabela de pares que mudam de decisão e cartão do par herói. Fecha a WAVE 2.

## Eixos avaliados

1. **Correção funcional.** ✅ Usa `build_engine_from_config(cfg, mode=...)` e
   `engine.triage(df.copy())` exatamente conforme o contrato verificado (Fase 0.2). Um engine
   por modo (orçamento `llm_used` é estado interno do engine) + cópia defensiva do `df`. Colunas
   decisórias corretas (`base_choice`, `base_loss`, `loss_llm`, `evr`, `action`, `review_requested`).
2. **Cobertura de testes.** ✅ +2 testes (TST2.6.a conservadorismo direcional: `confirmacao`
   produz ≤ MATCH automáticos que `vigilancia`; TST2.6.b: ao menos um par muda de ação entre
   modos e `budget.llm_used ≤ llm_max`). Full suite 85 passed (era 83).
3. **Determinismo.** ✅ Triagem é determinística dada `p_cal`/`band`/`guardrail` fixos; sem RNG.
4. **Fidelidade científica.** ✅ Matemática do custo esperado correta e com todos os símbolos
   definidos: `loss_match=(1-p)c_fp`, `loss_non=p·c_fn`,
   `loss_llm=c_llm+(1-p)e_fp·c_fp+p·e_fn·c_fn`, `base_loss=min`, `evr=base_loss-loss_llm`.
   Assimetria de custos dos dois modos descrita honestamente.
5. **Qualidade de código.** ✅ `ruff check` + `ruff format --check` limpos nos 3 `.py`. Sem
   imports não usados (reusa `plt`/`np`/`pd`/`card_heroi`/`hero_idx`/`cfg` já em escopo).
6. **Clareza didática (PT-BR).** ✅ DF-3 respeitado — toda célula de código é precedida por
   markdown (objetivo → intuição → ação → recap). Par herói (`zona_cinzenta`) acompanhado.
7. **Higiene de dados.** ✅ Sem PII; dataset sintético.
8. **Não-poluição da lib.** ✅ Nada tocado em `src/gzcmd_record_linkage/`.
9. **Rigor estatístico.** ✅ Não há afirmação de generalização aqui (métricas held-out ficam na
   Fase 3.2); a seção é descritiva da política.
10. **Andaime didático (Bloom).** ✅ Objetivos com verbos (explicar/calcular/comparar/interpretar),
    intuição antes do formalismo, recap + "o que vem a seguir" (Wave 3).

## Verificação independente
- `ruff check` → All checks passed!
- `ruff format --check` → 3 files already formatted
- `pytest` (suite completa) → **85 passed**, 2 warnings (xgboost CUDA→CPU benigno) em ~40s
- NBEXEC (`test_tst_2_1_nbexec_executa_ponta_a_ponta`) → incluído na suíte, verde
- Notebook regenerado: 74 células

## Achados
- 🔴 Nenhum.
- 🟡 Nenhum.
- 🟢 (backlog Fase 4.1) No markdown de matemática, `c\_{fp}` renderiza como `c_fp` literal em
  vez de subscrito verdadeiro (`c_{\mathrm{fp}}`). Cosmético; não afeta correção nem execução.

## Gate da WAVE 2
✅ Todos os estágios (load → bands → calibração → guardrails → triage) reproduzidos
isoladamente, explicados em PT-BR e visualizados; notebook executa ponta a ponta. WAVE 2 fechada.
