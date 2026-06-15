# Senior QA Review — Fase 2.2 (Carga e Feature Engineering)

**Revisor:** orquestrador (self:opus). **Veredito:** ✅ APROVADA.

## Escopo
Seção 7 do notebook (carga via `loader` real + *feature engineering*): agregados,
flags, MACD, visualização de sobreposição de classes e *hero card*. Builder
`notebooks/build_notebook.py` (constante `FASE_2_2`) + testes.

## Eixos avaliados
1. **Correção funcional** — `load_comparador_csv(CSV_PATH, cfg=LoadConfig(macd_enabled=True))`
   exatamente como no contrato verificado (`docs/plans/qa/contrato-api.md`). Nomes de
   colunas agregadas/flags/MACD conferem com o produzido pelo `loader`. ✅
2. **Cobertura de testes** — `tests/test_notebook_pipeline.py` (TST2.2.a) cobre:
   presença das colunas engenheiradas, *ranges* [0,1] dos agregados, `nota_final`
   plausível, `TARGET` binário e ambas as classes presentes. NBEXEC executa o
   notebook ponta-a-ponta com as novas células. ✅
3. **Determinismo** — testes operam sobre o CSV sintético versionado (fixo). ✅
4. **Fidelidade científica** — o histograma responde a UMA pergunta (as notas
   separam as classes?) e evidencia honestamente a **sobreposição** na zona
   cinzenta; texto deixa claro que `nota_final` é o observável a calibrar. ✅
5. **Qualidade de código** — `ruff check` limpo em `notebooks/*.py` e `tests/*`. ✅
6. **Clareza didática (PT-BR)** — objetivos de aprendizagem, intuição antes do
   formalismo, ação, *hero card* reaproveitado e recap. ✅
7. **Higiene de dados** — 100% sintético; sem PII. ✅
8. **Não-poluição da lib** — mudanças só em `notebooks/` e `tests/`; `src/` intacto. ✅
9. **Rigor estatístico** — calibração quantitativa virá na Fase 2.4; aqui a
   exigência é mostrar a sobreposição, atendida. ✅
10. **Andaime didático (Bloom)** — seção autocontida com objetivo→intuição→ação→recap. ✅

## Achados
- 🔴 nenhum.
- 🟡 nenhum.
- 🟢 (backlog, não-bloqueante) A célula de flags usa
  `df[flags].apply(lambda s: s.astype(float)).agg(["mean","max"])`; legível, porém
  poderia ser simplificada num polimento futuro (Fase 4.1).

## Evidência
- `pytest` (suíte completa): **65 passed** (era 62; +3 de TST2.2.a).
- `pytest tests/test_notebook_pipeline.py tests/test_notebook_execution.py`: 6 passed
  (inclui NBEXEC ponta-a-ponta).
- `ruff check notebooks/*.py tests/*`: All checks passed!

## Numeração
A seção 6 (herói) encerrou a Fase 2.1; a Fase 2.2 segue como **seção 7** (numeração
sequencial do notebook). A referência "seção 4" no plano é numeração lógica do plano,
não do notebook — sem impacto nos critérios.
