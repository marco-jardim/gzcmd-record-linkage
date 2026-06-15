# Senior QA Review — Fase 2.1 (Estrutura, Narrativa e Setup do Notebook)

**Revisor:** orquestrador (self:opus). **Data:** execução contínua do plano.
**Artefatos:** `notebooks/build_notebook.py`, `notebooks/gzcmd_passo_a_passo.ipynb`
(gerado), `tests/test_notebook_execution.py`, `tests/_nb_paths.py`.

## Decisão de arquitetura registrada
O notebook é **gerado** por `notebooks/build_notebook.py` (nbformat) a partir de uma
lista estruturada de células. Justificativa: o notebook cresce 6 fases (Wave 2);
manter JSON à mão é frágil. Benefícios: saídas sempre limpas (DEC-03), reprodutível,
*diff* legível, e a regra DF-3 (markdown antes de código) é estruturalmente garantida.
O `.ipynb` permanece o entregável versionado. Regenerar: `python notebooks/build_notebook.py`.

## Avaliação por eixo

| Eixo | Resultado |
|------|-----------|
| 1. Correção funcional | ✅ Usa API real (`synthetic_data.generate_comparador`, `to_comparador_csv`); diagrama do pipeline bate com o contrato verificado. |
| 2. Cobertura de testes | ✅ 3 testes: seções presentes (TST2.1.a), código não-vazio, execução headless ponta-a-ponta (TST2.1.NBEXEC via `nbclient`). |
| 3. Determinismo | ✅ `SEED=42`; gerador determinístico; bootstrap acha a raiz do repo sem caminho absoluto. |
| 4. Fidelidade científica | ✅ Nota metodológica in-sample×held-out já antecipada na visão geral; modos descritos corretamente (vigilancia=recall, confirmacao=precision). |
| 5. Qualidade de código | ✅ `ruff check`/`format` limpos em `notebooks/*.py` + `tests/*`. `.ipynb` excluído do ruff (escopo do plano em 4.2; validado por execução). |
| 6. Clareza didática (PT-BR) | ✅ Objetivos de aprendizagem (Seção 1), intuição antes do formalismo, glossário, exemplo-fio-condutor ("herói" = `zona_cinzenta`), recap. |
| 7. Higiene de dados | ✅ 100% sintético; `p_true` NÃO entra no CSV; formato `;`/`,`. |
| 8. Não-poluição da lib | ✅ Tudo em `notebooks/`; deps da lib (`pyproject [project]`) intactas. Mudanças em `pyproject`: só tooling (ruff exclude/per-file, marker pytest). |

## Achados e correções
- 🟡 **Imports não usados na célula de setup** (numpy/matplotlib/nb_helpers seriam
  importados sem uso na Fase 2.1). **Corrigido:** setup importa apenas `pandas` +
  `synthetic_data` + versão; os demais serão importados na fase de primeiro uso.
  Markdown ajustada para não prometer módulos ainda não importados.
- 🟡 **`tests/` não é pacote** → `from tests._nb_paths` falhou. **Corrigido:** import
  de topo `from _nb_paths import ...` (pytest insere `tests/` no `sys.path`).
- 🟢 **Marker `notebook`** registrado em `pyproject` para evitar warning.

## Veredito
**APROVADA.** 62 testes verdes (59 → 62), ruff limpo, notebook executa headless.
DF-3 satisfeita por construção. Pronta para Fase 2.2 (carga + feature engineering).
