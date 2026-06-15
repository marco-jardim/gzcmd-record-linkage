# Conformidade Final (DF-2) — Varredura Item a Item

> Fase 4.3 · T4.3.6 · critério **DF-2** (Seção 5.2.1).
> Legenda: ✅ implementado · 📝 implementado com desvio/justificativa registrada · ❌ omitido (nenhum).
> Evidência transversal: `pytest` **101 passed**; `nbconvert --execute` **exit 0** (507 KB);
> `ruff check`/`format --check` **limpos** em `notebooks/*.py` + `tests/*`.

## Critérios de Aceitação Global (CA-G1…CA-G9)

| ID | Critério | Status | Onde / evidência |
|----|----------|--------|------------------|
| CA-G1 | Notebook executa ponta-a-ponta em ambiente limpo | ✅ | `nbconvert --execute` exit 0 (Fase 4.2); `test_notebook_execution.py::...nbexec` |
| CA-G2 | Cada estágio isolado + PT-BR + ≥1 visualização | ✅ | Seções 7–11 (histogramas, barras, PR/ROC, custo×limiar) |
| CA-G3 | Rota A in-sample reconcilia com `run_v3` (atol≤1e-9 Platt) | ✅ | Fase 3.1: `band` idêntico, `p_cal` maxdiff=0.0, `action` 100% — `TST3.1.a` |
| CA-G4 | Dataset sintético determinístico, schema válido, edge cases, ambas classes, `p_true` | ✅ | Fase 1.1/1.2; `test_synthetic_data.py` (TST1.1.a–h, 1.2.a–c) |
| CA-G5 | Métricas held-out, ambos modos, variância ≥5 seeds, PR/ROC + custo×limiar | ✅ | Fase 3.2; seeds [42,123,456,789,2024]; `TST3.2.a–d` |
| CA-G6 | Estágio LLM por stub determinístico + protocolo explicado | ✅ | Fase 3.3; `llm_review_stub`; `dual_agent_plus_arbiter` (markdown) |
| CA-G7 | Suíte nova passa; cobertura ≥90% código novo; `ruff` limpo | ✅ | 101 passed; `synthetic_data` 90%, `nb_helpers` ≥90%; ruff limpo |
| CA-G8 | *Reliability diagram* held-out + ECE/Brier + overlay `p_true` + declara in-sample×held-out + R-11 | ✅ | Fase 2.4 (sec 9.3–9.6); `TST2.4.c/e` |
| CA-G9 | Didática: objetivos, intuição→formalismo, fio-condutor, recap | ✅ | Seções 1–15 + `auditoria-didatica.md` |

## Decisões de Projeto (DEC-01…DEC-10)

| ID | Decisão | Status | Nota |
|----|---------|--------|------|
| DEC-01 | Gerador em `notebooks/` + `pythonpath` do pytest | ✅ | `pyproject.toml` `pythonpath=["src","notebooks"]` |
| DEC-02 | Platt principal; XGBoost apêndice | 📝 | Apêndice (sec 9.7) **markdown-only**, não executado, por R-13 (não-determinismo) + escopo. Não bloqueia CA-G. |
| DEC-03 | `.ipynb` limpo versionado + artefato executado em `qa/` | ✅ | `*.executed.ipynb` (507 KB) gerado; **não** versionado (`.gitignore`) |
| DEC-04 | 300–1000 pares, seed 42 | ✅ | 600 aleatórios + 7 cenários; seed 42 |
| DEC-05 | Versionar CSV sintético | ✅ | `data/synthetic/comparador_sintetico.csv` commitado |
| DEC-06 | Posterior verdadeira `p*` anti-circularidade | ✅ | `p_true=σ(0.85·(nota−s0))`, `TARGET~Bernoulli(p*)`; coluna de validação |
| DEC-07 | Duas rotas explícitas (A fiel / B correta) | ✅ | Sec 9.2 (A) e 9.3 (B) |
| DEC-08 | ECE + Brier quantitativos | ✅ | `nb_helpers.expected_calibration_error`/`brier_score` + `test_nb_helpers.py` |
| DEC-09 | Andaime didático obrigatório | ✅ | objetivos/intuição/herói/recap por seção |
| DEC-10 | Interatividade `ipywidgets` (opcional) | 📝 | **Não implementado**. Plano marca como opcional e não-bloqueante (não pode ser pré-requisito da execução headless / CA-G1). Backlog. |

## Riscos e Mitigações (R-01…R-13)

| ID | Status | Mitigação aplicada |
|----|--------|--------------------|
| R-01 sem dataset | ✅ | Gerador sintético (Wave 1) |
| R-02 sem tooling notebook | ✅ | `requirements/notebook.txt` |
| R-03 `docs/plans/` ausente | ✅ | criado |
| R-04 mismatch config×loader | ✅ | de-para em `contrato-api.md`; gerador emite o que o loader consome |
| R-05 LLM exige API | ✅ | `llm_review_stub` determinístico, sem rede |
| R-06 fixtures desconhecidas | ✅ | inspecionado (não há `conftest.py`; fixtures inline) |
| R-07 `triage` muta df | ✅ | `triage` retorna cópia; usado `df.copy()` defensivo (sec 11) |
| R-08 Platt instável c/ poucos positivos | ✅ | `TST2.4.b` degradação elegante |
| R-09 notebook não-determinístico | ✅ | seeds fixas; execução limpa reproduzível |
| R-10 calibração in-sample (vazamento) | ✅ | Rotas A/B (DEC-07); métricas de generalização só held-out |
| R-11 config descreve mais que o código | ✅ | declarado no notebook sec 9.6 + `contrato-api.md` |
| R-12 circularidade do sintético | 📝 | `p*` + sobreposição (✅). Demo de vazamento por grupo (sec 13.3) é **negligenciável neste dataset por construção** (grupos majoritariamente singletons) — explicado com honestidade no notebook; mecanismo descrito para produção. |
| R-13 não-determinismo XGBoost | ✅ | reconciliação tight só p/ Platt; XGBoost qualitativo (`TST3.1.c`) / apêndice markdown |

## Desvios de contrato confirmados no código (vs. texto do plano)
- **Guardrail âncora `ALWAYS_MATCH`:** o código exige `nota_final ≥ 10` **e** nome/data/município perfeitos (não `≥ 9` como o texto da config sugeria). O notebook e o gerador ("match_obvio") respeitam o comportamento **real do código** (ground truth). 📝
- **`triage` não muta o df de entrada** (retorna cópia); o plano §2.2 dizia "muta o df". Documentado em `contrato-api.md`. ✅

## Tasks / Testes por fase (resumo)
Todas as Tasks `T*` e Testes `TST*` das Fases 0.1–4.3 foram implementados e
verificados (commits `chore(env)…` a `chore(release)…`). Detalhe por fase nos
respectivos `docs/plans/qa/REVIEW-<fase>.md`. Suíte: **101 testes** distribuídos
em `test_env_smoke`, `test_api_contract`, `test_scaffold_imports`,
`test_synthetic_data`, `test_nb_helpers`, `test_notebook_pipeline`,
`test_notebook_execution`, `test_notebook_didatica`.

## DoD Final (Seção 5.2.1)
- **DF-1** (rodar tudo) — ✅ `pytest` 101 passed + `nbconvert` exit 0; saídas coerentes com a narrativa (números de reconciliação, AUCs, taxas do stub conferidos).
- **DF-2** (conformidade) — ✅ este documento; **nenhum item omitido silenciosamente** (2 itens 📝 justificados: DEC-10 opcional, DEC-02 apêndice markdown).
- **DF-3** (explicação antes de cada etapa) — ✅ `auditoria-didatica.md` + `test_notebook_didatica.py`.

**Conclusão:** conformidade global satisfeita. Nenhum ❌.
