# Senior QA Review — Fase 2.5 (Guardrails)

**Revisor:** orquestrador (self:opus). **Veredito:** ✅ APROVADA.

## Escopo
Seção "## 10. Guardrails" do notebook (via `build_notebook.py` → `FASE_2_5`), teste
TST2.5.a em `tests/test_notebook_pipeline.py`, e lista de seções em
`tests/test_notebook_execution.py`.

## Eixos avaliados
1. **Correção funcional:** usa a API real `apply_guardrails(df) -> GuardrailOutput`
   com `.guardrail`/`.reason`; não muta o df (colunas adicionadas explicitamente).
   Valores e motivos conferem com o contrato (`docs/plans/qa/contrato-api.md`).
2. **Cobertura de testes:** TST2.5.a verifica que os 3 tipos de guardrail disparam e
   que os 4 motivos (`nota_final_high`, `nota_final_low`, `temporal_filter`,
   `homonimia_risk`) mapeiam para o guardrail correto. Inclui os dois motivos que
   levam a `ALWAYS_NONMATCH`.
3. **Determinismo:** opera sobre o CSV versionado (seed 42); sem RNG/rede.
4. **Fidelidade científica:** declara R-11 (regra `grey_mother_missing` da config NÃO
   implementada no código) — honesto; ensina o que o código faz.
5. **Qualidade de código:** ruff `check` + `format --check` limpos nos 3 arquivos.
6. **Clareza didática (DEC-09/DF-3):** objetivos de aprendizagem (verbos de Bloom),
   intuição antes da ação, toda célula de código precedida por markdown, tabela de
   cenários, card do herói (`zona_cinzenta`, guardrail=NA → segue para triagem), recap.
7. **Higiene de dados:** 100% sintético.
8. **Não-poluição da lib:** nenhuma alteração em `src/` ou deps da lib.

## Evidência
- `pytest tests/` → **83 passed**, 2 warnings (xgboost CUDA→CPU benigno).
- NBEXEC (`test_tst_2_1_nbexec_executa_ponta_a_ponta`) → passa (~7.4s).
- `ruff check` / `ruff format --check` → limpos.
- Notebook: 60 células; seção `## 10. Guardrails` presente.

## Achados
- 🔴: nenhum. 🟡: nenhum. 🟢: nenhum.

## DoD
- [x] Seção 10 pronta e executável. [x] Testes verdes. [x] Cobertura mantida.
- [x] QA registrado. [x] Pronto para commit.
