# Senior QA Review — Fase 1.2 (Cenários Narrativos + Edge Cases)

**Revisor:** orquestrador (self:opus). **Estado:** 59 passed; ruff limpo; cobertura `synthetic_data` = **90%** (≥90% DoD).

## Escopo
- `synthetic_data.generate_comparador(scenarios=...)`: injeção determinística de 7 cenários nomeados (`SCENARIO_NAMES`), `meta["scenarios"]` mapeia nome→índice.
- Tests TST1.2.a (bandas), TST1.2.b (guardrails+reason), TST1.2.c (triage grey→LLM_REVIEW) + determinismo.

## Verificação independente end-to-end (via `run_v3`, mode=confirmacao, não confiei no auto-relato)
| Cenário | nota | band | guardrail | reason | action |
|---|---|---|---|---|---|
| match_obvio | 10.0 | high | ALWAYS_MATCH | nota_final_high | MATCH |
| nonmatch_obvio | 1.0 | low | ALWAYS_NONMATCH | nota_final_low | NONMATCH |
| homonimo | 7.5 | grey_high | FORCE_REVIEW | homonimia_risk | LLM_REVIEW |
| obito_antes_diag | 6.0 | grey_mid | ALWAYS_NONMATCH | temporal_filter | NONMATCH |
| mae_ausente | 6.5 | grey_mid | — | — | LLM_REVIEW |
| datas_invertidas | 6.8 | grey_mid | — | — | LLM_REVIEW |
| zona_cinzenta | 6.5 | grey_mid | — | — | LLM_REVIEW |

Todos batem com o projetado. `match_ratio` realizado = 0.49 (alvo 0.5) mesmo com injeção.

## Achados
- 🟢 Confusão de assinatura `load_comparador_csv(..., config=)` (correto: `cfg=`) detectada pelo subagente (parou após 2 falhas, conforme regra) e **corrigida** pelo orquestrador. Lição registrada para o router.
- 🟢 `mae_ausente`/`datas_invertidas` não disparam guardrail dedicado — correto: a regra `grey_mother_missing` da config **não está implementada** no código (R-11). Documentado no docstring; ambos caem em LLM_REVIEW por estarem na zona cinzenta sob `confirmacao`.
- 🟢 `PAR`/`TARGET` dos cenários é override didático deliberado (poucas linhas) — documentado; não afeta materialmente a calibração (match_ratio 0.49).
- Nenhum 🔴/🟡 pendente.

## Rubrica
Correção ✅ · Cobertura ✅ (90%) · Determinismo ✅ · Fidelidade ✅ · Código ✅ (ruff) · Didática ✅ · Higiene ✅ · Não-poluição ✅ · Rigor ✅

## Veredito
**Done.** Wave 1 concluída — dataset robusto, determinístico, com cenários verificados ponta-a-ponta. Pronto para Wave 2 (notebook).
