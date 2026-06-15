# Senior QA Review — GLOBAL (Fase 4.3 / T4.3.1)

> Revisor: orquestrador (Opus) em papel de **QA sênior cético**.
> Escopo: o conjunto do entregável (notebook + gerador + helpers + testes + docs).
> Gate final do plano (Seção 4, rubrica de 10 eixos; Seção 5 critérios globais).

## Evidência objetiva
- `pytest` (suíte completa): **101 passed**, 2 warnings (xgboost CUDA→CPU, benigno) — sem `skip`/`xfail`.
- `nbconvert --to notebook --execute`: **exit 0**, artefato 507 KB (`docs/plans/qa/gzcmd_passo_a_passo.executed.ipynb`, não versionado por DEC-03).
- `ruff check` + `ruff format --check`: **limpos** em `notebooks/*.py` e `tests/*`.
- Notebook: **103 células**, 15 seções, 37 células de código — todas precedidas por markdown (DF-3).

## Avaliação por eixo (rubrica Seção 4)

| # | Eixo | Veredito | Comentário |
|---|------|----------|------------|
| 1 | Correção funcional | ✅ | APIs reais usadas conforme `contrato-api.md`; reconciliação rota A × `run_v3` **exata** (maxdiff 0.0). |
| 2 | Cobertura de testes | ✅ | `synthetic_data` 90%, `nb_helpers` ≥90%; edge cases do catálogo 4.1 cobertos. |
| 3 | Determinismo/reprodutibilidade | ✅ | seeds fixas; `fit_platt` determinístico; stub determinístico; sem rede/relógio. |
| 4 | Fidelidade científica | ✅ | Platt derivado corretamente (NLL + L2 só no slope, Newton-Raphson); eixos/legendas PT-BR. |
| 5 | Qualidade de código | ✅ | ruff limpo; sem `# type: ignore`; paths via `pathlib`/`importlib.resources`. |
| 6 | Clareza didática (PT-BR) | ✅ | objetivos/intuição/herói/recap; glossário; sem "AI slop". |
| 7 | Higiene de dados | ✅ | 100% sintético; sem PII; CSV `;`/`,`; `p_true` nunca entra no pipeline. |
| 8 | Não-poluição da lib | ✅ | nada de apresentação em `src/`; `pyproject [project]` deps **intocadas** (só tooling/pytest config). |
| 9 | Rigor estatístico (PhD) | ✅ | sem vazamento treino/teste (rotas A/B); ECE+Brier numéricos; validação contra `p*`; variância multi-seed; "calibrado" sempre com número. |
| 10 | Andaime didático (Bloom) | ✅ | objetivos por seção; intuição antes do formalismo; fio-condutor; recap; símbolos definidos. |

## Pontos de honestidade científica que reforçam a qualidade
- A demonstração de **vazamento por split** (sec 13.3) é apresentada com honestidade: neste dataset o efeito é **negligenciável por construção** (grupos majoritariamente singletons). Em vez de forjar um resultado, o notebook **mede e explica** o mecanismo (relevante em produção com *blocking*). Este é o tipo de transparência exigido pelo eixo 9.
- A divergência **config × código** (R-11: `anchor_platt`/`by_band` não implementados) é declarada explicitamente (sec 9.6), ensinando "o que o código faz" e tratando a config como roadmap.
- A âncora de guardrail real (`nota ≥ 10` + perfeições) — mais estrita que o texto da config (`≥ 9`) — é respeitada conforme o **código** (ground truth), não conforme o texto.

## Achados
- 🔴 bloqueantes: **nenhum**.
- 🟡 importantes: **nenhum** (o backlog 🟡/🟢 das fases anteriores foi zerado na Fase 4.1).
- 🟢 nice-to-have (backlog, não-bloqueante):
  - DEC-10 (`ipywidgets` interativo) **não implementado** — opcional por plano; manteria a execução headless segura via `try/except`/flag se adicionado no futuro.
  - DEC-02 apêndice XGBoost permanece **markdown-only** (decisão consciente por R-13).

## Critérios globais
- **CA-G1…CA-G9:** todos satisfeitos (ver `conformidade-final.md`).
- **DoD Global (5.2):** DoDs de fase cumpridos e commitados; QA por fase em `REVIEW-*.md`; README do notebook presente; `.ipynb` limpo versionado + artefato executado como evidência; log de execução consolidado na Seção 10 do plano.
- **DF-1 / DF-2 / DF-3 (5.2.1):** satisfeitos.

## Veredito
**APROVADO — "Done" assinado.** O entregável passou de "demo bonita" a
**material acadêmico defensável**: fiel ao `gzcmd`, reprodutível, estatisticamente
honesto e didaticamente escalonado. Recomendação: publicar; tratar DEC-10 como
melhoria futura opcional.
