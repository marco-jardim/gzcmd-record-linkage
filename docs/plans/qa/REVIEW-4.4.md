# Senior QA Review — Fase 4.4 (Interatividade · DEC-10)

> Incremento pós-release a pedido: implementar `ipywidgets` (DEC-10), antes
> tratado como backlog 🟢 opcional.

## O que foi implementado
Nova **seção 15 — "Painel interativo (opcional)"** no notebook (gerada por
`FASE_INTERATIVO` em `build_notebook.py`):
- Função `painel_operacao(limiar, escala_slope)` que, sobre o **conjunto de
  teste held-out** e o modelo Platt da seção 9.3, recalcula `p_cal` com o
  *slope* reescalado, aplica o limiar de `MATCH` e plota (i) a distribuição de
  `p_cal` por classe com a linha do limiar e (ii) precisão/recall/F1 com o
  **custo esperado** (FP×10 + FN×50, modo `vigilancia`) no título.
- **Sliders** `ipywidgets` (limiar ∈ [0,1]; escala do *slope* ∈ [0.3, 3.0]) via
  `widgets.interactive`.

## Headless-safe (CA-G1) — o ponto crítico
A célula **sempre** chama `painel_operacao(0.50, 1.0)` (figura estática) antes de
tentar os widgets. Os sliders ficam dentro de `try/except`: em execução
automatizada (`papermill`/`nbconvert`) sem frontend de widgets, nada quebra; em
Jupyter ao vivo, os sliders ativam. Conclusões renumeradas para **seção 16**.

## Verificação (evidência)
- `python notebooks/build_notebook.py` → **106 células** (era 103; +1 md, +1 code, +1 recap).
- `pytest` suíte completa: **101 passed**, 2 warnings (xgboost CUDA→CPU, benigno). O teste **NBEXEC** executa a nova célula de ponta a ponta **sem erro** (prova o fallback headless).
- `test_notebook_didatica` (DF-3): célula de código 103 precedida por markdown 102 ✅.
- `test_tst_4_1_a` (figuras): a nova figura tem título + rótulos de eixos ✅.
- `test_notebook_execution` seções: '## 15. Painel interativo' e '## 16. Conclusões e limitações' presentes ✅.
- `ruff check` + `format --check`: limpos.
- Artefato `*.executed.ipynb` regenerado (`nbconvert --execute`, exit 0).

## Achados
- 🔴/🟡: **nenhum**.
- 🟢: nenhum novo. (DEC-02 apêndice XGBoost segue markdown-only por R-13, decisão mantida.)

## Veredito
**APROVADO.** DEC-10 sai do backlog para ✅ implementado, sem comprometer a
reprodutibilidade headless. Docs de conformidade, auditoria didática, REVIEW-GLOBAL
e log de execução (Seção 10) atualizados.
