# Senior QA Review — Fase 3.2 (Avaliação held-out multi-seed + PR/ROC + custo vs limiar)

**Revisor:** Orquestrador (Opus, `[self:opus]`). **Veredito:** ✅ APROVADA (após correções de honestidade científica).

## Escopo revisado
Seção `## 13. Avaliação held-out (rota B)` do notebook (constante `FASE_3_2` em `build_notebook.py`), tests `test_notebook_pipeline.py` (TST3.2.a–d) e entrada de seção em `test_notebook_execution.py`.

## Eixos de QA

1. **Correção funcional:** `evaluate_v3_dataframe(..., calibration="platt")` usado exatamente conforme contrato (held-out: Platt no treino, prediz no teste; computa band/guardrail/p_cal internamente). Colunas reportadas (`auto_precision/recall/fbeta/coverage`) conferidas. ✅
2. **Cobertura de testes:** 4 testes novos determinísticos (consistência de métricas + fechamento de confusão `auto_total + llm_used == n_test`; direção de recall média sobre 5 seeds; row vs group-aware; colunas esperadas). Suíte 92 passed. ✅
3. **Determinismo:** seeds fixas `[42,123,456,789,2024]` (= `config.evaluation.seeds`); split group-aware com `seed`. ✅
4. **Fidelidade científica (eixo 9 — PhD):** held-out correto; PR-AUC=0.912, ROC-AUC=0.921 (não-trivial, classes sobrepostas). Variância entre seeds reportada (média±desvio + barras de erro). ✅
5. **Qualidade de código:** ruff check + format limpos (`build_notebook.py`, ambos os tests). ✅
6. **Clareza didática:** objetivos/intuição/ação/recap presentes; DF-3 respeitado (md antes de cada código). ✅

## 🔴/🟡 Achados e correções

- **🟡 [CORRIGIDO] Vazamento por split `row` NÃO é empiricamente demonstrável neste dataset.** Números observados (vigilancia, 5 seeds): `row` Fβ=0.9769, `comprec`=0.9803, `refrec`=0.9788 — `row` é até ligeiramente **menor**, não inflado. Causa: o gerador sintético produz grupos COMPREC/REFREC majoritariamente de **tamanho 1** (sem registro compartilhado para vazar). A narrativa original implicava inflação ("métricas otimistas") sem qualificar.
  - **Correção:** (a) markdown 13.3 reescrito com "expectativa honesta" — explica que o efeito é desprezível AQUI por construção e por quê; (b) célula de código agora **mede e imprime** a fração de COMPREC/REFREC repetidos antes da comparação; (c) novo markdown pós-tabela lê os números com honestidade e contextualiza o mecanismo (real em produção, com blocking gerando muitos pares por registro); (d) teste renomeado `test_tst_3_2_c_row_split_infla` → `test_tst_3_2_c_row_split_nao_deflaciona` com docstring explicando a tolerância e a razão empírica. Assertion tolerante (`row >= comprec - 0.02`) mantida — honesta, não afirma inflação forte.

- Nenhum 🔴. Nenhum 🟡 remanescente.

## 🟢 Backlog (Fase 4.1)
- Curvas PR/ROC reutilizam o split único da seção 9.3 (seed 42); poderia mostrar banda de variação multi-seed — nice-to-have, não bloqueante.

## Evidência
- `python notebooks/build_notebook.py` → 94 células.
- `ruff check`/`format --check` → limpos.
- `pytest` full → 92 passed; NBEXEC executa o notebook ponta-a-ponta (~45s).
- Números-chave (seed42/n600/scenarios=all): PR-AUC=0.912, ROC-AUC=0.921; recall vigilancia=1.00 ≥ confirmacao=0.80; cobertura vigilancia=0.54 vs confirmacao=0.29; COMPREC/REFREC repetidos ≈ baixa fração (grupos ~singleton).

CA-G5 satisfeito.
