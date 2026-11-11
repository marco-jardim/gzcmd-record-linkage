# Auditoria Agent-Native — gzcmd-record-linkage

**Data:** 2026-07-07
**Papel:** Principal architect review
**Objetivo:** tornar o repositório *agent-native* — um agente de código deve conseguir pegar um bug report de usuário ou uma feature do roadmap e, sozinho, **reproduzir → implementar → testar → verificar** em build real, com o mínimo de intervenção humana.
**Plano de execução derivado:** `docs/plans/agent-native-plan.md`

---

## 0. Estado atual (baseline verificado)

| Dimensão | Estado |
| --- | --- |
| Linguagem/stack | Python ≥3.10, hatchling + hatch-vcs, sklearn/xgboost/pandas, PyYAML |
| Código | `src/gzcmd_record_linkage/` — 19 módulos, ~2.935 LOC |
| Entry point | CLI `gzcmd` (`cli.py:225`), subcomandos `run`, `fit-calibration`, `eval`. Sem servidor/GUI/jobs |
| Testes | 101 testes pytest, **100% offline e determinísticos**, sem mocks, sem conftest.py |
| CI | GitHub Actions (push/PR, Python 3.11/3.12): ruff parcial, pytest, coverage parcial, smoke CLI, build |
| Dados | `data/synthetic/comparador_sintetico.csv` (608 linhas, versionado — DEC-05); gerador determinístico `notebooks/synthetic_data.py` |
| Docs | `docs/plans/` maduro (plano do notebook, QA reviews por fase, `contrato-api.md` verificado contra código) |
| Convenções p/ agentes | **CLAUDE.md / AGENTS.md: inexistentes** |

**Pontos fortes já agent-native** (não perder): suite offline determinística; dataset sintético versionado + gerador com seed; `docs/plans/qa/contrato-api.md` é um contrato de API verificado contra o código (raro e valioso); cultura de QA por fase já estabelecida.

---

## 1. Chokepoints de julgamento humano

Cada item abaixo é um ponto onde hoje uma mudança exige julgamento pessoal ou conhecimento tribal, com a convenção escrita que o substitui.

| ID | Chokepoint (hoje) | Evidência | Convenção que resolve |
| --- | --- | --- | --- |
| **CP-01** | Não existe nenhum arquivo de convenções para agentes. Comandos canônicos, política de skip/xfail (DF-1), fronteira lib×notebooks (D7), idioma (docs PT-BR / código EN) vivem em um plano de 65KB e na cabeça do mantenedor | ausência de `CLAUDE.md`; diretrizes espalhadas em `docs/plans/notebook-gzcmd-passo-a-passo.md` | **CLAUDE.md** com: tabela de comandos, gates de qualidade, mapa de módulos, regras D1–D9, fluxo de release |
| **CP-02** | **Mapa de vida dos módulos é tribal.** ~1.158 linhas (39% do src) são código dormente inalcançável pela CLI: `llm_review.py` (697), `llm_dossier.py` (309), `drift_monitor.py` (143, zero importadores no repo inteiro), `runner_api.py` (9, zero importadores). Um agente recebendo "bug na revisão LLM" não sabe que o caminho vivo é o stub `nb_helpers.llm_review_stub` | grep de importadores; `runner.py`/`eval.py`/`cli.py` nunca importam esses módulos | Tabela **LIVE / DORMANT** no CLAUDE.md + decisão formal de disposição (subpacote `experimental/`) + banner no topo de cada módulo dormente |
| **CP-03** | **Barra de qualidade não-uniforme e não escrita.** Lint cobre 1/19 módulos (`ruff check src/gzcmd_record_linkage/cli.py tests`); coverage gate cobre 4/19 (loader, classifier, bands, guardrails); pyright é dev-dep declarada e **nunca invocada**. "Qual gate se aplica ao arquivo X?" = julgamento humano | `.github/workflows/ci.yml`; `pyproject.toml` (`fail_under = 0`) | Gates uniformes: `ruff check .`, `pyright`, coverage com ratchet documentado; tabela de gates no CLAUDE.md; pre-commit |
| **CP-04** | **Decisões de projeto espalhadas**: DEC-01…10 dentro do plano do notebook, DEC-05 num comentário do `.gitignore`, D7 num comentário de `requirements/notebook.txt`, R-04/R-11 no `contrato-api.md` | localizações citadas | **`docs/decisions.md`** (ADR-lite, append-only) consolidando DEC/R vigentes, com regra de numeração contínua |
| **CP-05** | **Verificação do notebook é eyeball humano**: rodar `nbconvert --execute` e conferir se números/figuras batem com a narrativa (DF-1/DF-3 são políticas manuais) | `docs/plans/qa/conformidade-final.md` registra execução manual como evidência | Asserções automatizadas de invariantes sobre o notebook executado (ranges de ECE/Brier, contagens de decisão) |
| **CP-06** | **Fluxo de release não documentado**: versão vem de git tag via hatch-vcs; ninguém escreveu como taguear/publicar | `pyproject.toml` (hatch-vcs) | Seção "Release" no CLAUDE.md |
| **CP-07** | **Semântica dos modos e flags da CLI** (`vigilancia`, `confirmacao`; `--p-cal` com 7 opções; `--mode` obrigatório sem default) só descobrível lendo código + YAML. Para reproduzir um bug, escolher modo/flags é julgamento | `cli.py` (tabela de flags na seção 3 do plano) | Exemplos canônicos executáveis no CLAUDE.md + `docs/decisions.md` explicando cada modo |
| **CP-08** | **Config aspiracional (R-11)**: `gzcmd_v3_config.yaml` declara `expected_columns` que o loader não produz; agente que "corrige" o loader para casar com a config quebra o sistema | `contrato-api.md` seção "De-para de features (config × loader)" | Comentário na config + regra explícita no CLAUDE.md: "contrato-api.md vence a config em caso de conflito" |

---

## 2. Lacunas de verificação (por subsistema)

O que falta para um agente **verificar a própria mudança de ponta a ponta**.

| Subsistema | Verificação hoje | Lacuna | Verificação-alvo |
| --- | --- | --- | --- |
| `loader.py` (304L) | 2 testes; gate 80% | Fixture de teste divergente do contrato real (R-04: nomes ENDERECO errados → `endereco_score_total=0` silencioso); formatos de data mistos (`R_DTOBITO` DDMMYYYY vs demais YYYYMMDD) sem testes dedicados | Fixture canônica derivada do `contrato-api.md`; testes de propriedade p/ datas, NA-values, fallback latin-1, colunas obrigatórias ausentes |
| `classifier.py` / `calibration.py` | 9 testes; gate 80% só no classifier | calibration sem gate; sem teste de round-trip save/load dos modelos (Platt JSON, RF, XGB) | Round-trip: fit → save → load → predições idênticas |
| `bands.py` / `guardrails.py` | 1 teste cada; gate 80% | Fronteiras exatas de banda (p == limite) e todas as regras de guardrail não são exercitadas individualmente | Testes table-driven de fronteira |
| **`gzcmd_v3_policy_engine.py` (125L) — coração do produto** | **import-only smoke** | Zero testes comportamentais do motor de decisão (expected-loss triage, budget de LLM, `LLMError`) | Tabela de decisão esperada: casos (p_cal, banda, custos, budget) → decisão; esgotamento de budget; empates |
| `runner.py` × `eval.py` (pipelines gêmeos) | testes indiretos via contrato/CLI | **Nenhum teste de paridade**: ambos re-fiam os mesmos 6 módulos de forma independente; mudança em `loader`/`classifier` pode satisfazer um caminho e quebrar o outro em silêncio | Teste de paridade: mesmos inputs → mesmos `p_cal`/bandas/decisões nos dois caminhos |
| `metrics.py` (73L), `splitting.py` (130L), `reporting.py` (102L) | import-only smoke | Zero comportamento verificado (divisão por zero em métricas, vazamento de grupo no split, formato do LaTeX/CSV) | Testes unitários + golden files para reporting |
| CLI e2e | `--help` smoke (rodado 2× no CI, redundante) | Nenhum teste roda `gzcmd run/eval/fit-calibration` de verdade sobre o CSV sintético e verifica artefatos de saída | **Golden e2e**: 3 subcomandos sobre `data/synthetic/comparador_sintetico.csv` com goldens versionados e tolerâncias |
| `llm_review.py` / `llm_dossier.py` | nada (dormentes) | Sem harness de mock HTTP (Fireworks); sem validação contra `gzcmd_v3_llm_output.schema.json`; requer `FIREWORKS_API_KEY` | Transport injetável + fake conformante ao schema + testes offline de contrato |
| `drift_monitor.py` | nada (órfão) | Zero referências no repo inteiro | Decidir: wire ou mover p/ experimental (ver SO-03) |
| Logging / observabilidade | **Zero logging no pipeline vivo** (único logger está no módulo dormente) | Log assertions impossíveis; stdout é a única evidência de execução | Logging estruturado com `run_id` + testes com `caplog` |
| Type-check | pyright instalado, nunca roda | Regressões de tipo invisíveis | `pyright` no CI (nível basic) |
| Notebook | teste de execução headless existe (`-m notebook`) | Verificação de narrativa×números é manual | Invariantes extraídos do `.executed.ipynb` com asserções e ranges |
| UI / screenshots | N/A (sem UI) | — | Equivalente: golden files para saídas CSV/LaTeX de `reporting.py` |

---

## 3. Caminhos de reprodução (bug report → reprodução autônoma)

Cenário típico: usuário roda `gzcmd run --mode vigilancia` sobre um `COMPARADORSEMIDENT.csv` real (não versionável, contém quasi-identificadores: datas de nascimento/óbito) e reporta "números estranhos" colando um trecho do stdout.

| ID | Lacuna | Detalhe | O que construir |
| --- | --- | --- | --- |
| **RP-01** | **Não existe diagnostic bundle** | CLI imprime resumo em stdout e não persiste nada; usuário só pode colar texto à mão | `--report run_report.json`: versão do pacote (hatch-vcs), python, versões de deps-chave, hash SHA-256 do input, path+hash+dump da config resolvida, flags efetivas, seeds, contagens (rows, guardrail_counts, action_counts), params de calibração, timings |
| **RP-02** | **Não existe replay** | Nada reconstrói uma execução a partir de um relato | `gzcmd replay run_report.json [--input CSV]`: re-executa com as mesmas flags/config, compara contagens e hashes, exit ≠ 0 com diff legível quando divergir |
| **RP-03** | **Primeiro `gzcmd run` de um agente falha** | Path default `data/COMPARADORSEMIDENT.csv` não existe no repo; o fixture sintético tem outro nome | Documentar exemplo canônico com `--input data/synthetic/comparador_sintetico.csv` no CLAUDE.md + fixture de conveniência |
| **RP-04** | **Dados reais não podem virar fixture** (PII/quasi-identificadores) | Subscores já são anônimos, mas datas brutas acompanham | `gzcmd anonymize`: shift consistente de datas + re-ids, preservando os subscores — transforma bundle de usuário em fixture compartilhável |
| **RP-05** | **Sem template de bug report** | `.github/ISSUE_TEMPLATE/` inexistente | Template exigindo: comando exato, `run_report.json`, amostra mínima (ou saída do anonymize), versão (`gzcmd --version` — a implementar) |
| **RP-06** | **Cenários do gerador não mapeados para classes de bug** | `synthetic_data.generate_comparador(seed, n_pairs, scenarios)` já gera homonímia/datas trocadas, mas nada documenta "para reproduzir bug de guardrail temporal, use cenário X" | Tabela cenário→classe-de-bug no CLAUDE.md + fixtures nomeadas por cenário no conftest |

---

## 4. Obstáculos estruturais

| ID | Obstáculo | Evidência | Fronteira proposta |
| --- | --- | --- | --- |
| **SO-01** | **Pipelines gêmeos**: `runner.py` (7 imports internos) e `eval.py` (9 imports internos — recordista) re-fiam independentemente `config+loader+classifier+calibration+bands+guardrails`. Todo change em módulo-base exige validar 2 call sites; nenhum sinal força isso | mapa de acoplamento (seção 3 do relatório de arquitetura); `eval.py:24` chama `runner.build_engine_from_config` mas re-fia o resto | Extrair **builder compartilhado** (`pipeline.py`): um único ponto que monta o pipeline; `runner` e `eval` consomem-no; teste de paridade trava a equivalência |
| **SO-02** | `config.py` importado por 5 módulos; schema ripple. Agrava-se com R-11 (config aspiracional) | importadores: bands, eval, cli, drift_monitor, runner | Testes de contrato do config; docstring de cada campo dizendo quem consome; regra "contrato-api.md vence" |
| **SO-03** | **39% do src é dormente** e indistinguível de código vivo — agente perde horas ou "conserta" o módulo errado | CP-02 | Mover `llm_review`, `llm_dossier`, `drift_monitor`, `runner_api` para `gzcmd_record_linkage/experimental/` com shims de import (deprecação) + banners; excluir do denominador de coverage; contrato mock p/ LLM |
| **SO-04** | `cli.py` (323L) mistura parsing, resolução de paths, orquestração e apresentação | 5 maiores arquivos | Handlers puros testáveis (parse → dict de args → handler); manter compat de flags |
| **SO-05** | **Fronteira lib×notebooks borrada**: `pythonpath=["src","notebooks"]` faz a suite da lib depender de `synthetic_data.py`/`nb_helpers.py` (camada de apresentação). Agente que muda o gerador pode quebrar testes da lib sem perceber | `pyproject.toml` pytest config | Congelar o contrato do gerador (testes de contrato já existem em parte) e documentar a fronteira; longo prazo: promover gerador a `gzcmd_record_linkage.testing` respeitando D7 |
| **SO-06** | **Sem `conftest.py`**: fixtures duplicadas inline por arquivo; R-04 mostrou fixture divergente do contrato real | ausência verificada | `tests/conftest.py` com fixture canônica única derivada do `contrato-api.md` |

---

## 5. Plano priorizado — atenção-humana-poupada ÷ esforço

| # | Item | Resolve | Esforço | Atenção poupada | Racional |
| --- | --- | --- | --- | --- | --- |
| **1** | **CLAUDE.md + `docs/decisions.md` + templates de issue/PR** | CP-01, CP-02, CP-04, CP-06, CP-07, CP-08, RP-05 | **S** | **Altíssima** | Cada interação com agente hoje começa re-explicando comandos, fronteiras e o que é código vivo. É o multiplicador de todo o resto |
| **2** | **Gates uniformes** (ruff full + format, pyright, coverage ratchet, pre-commit, CI) | CP-03 | **S–M** | **Alta** | Remove a pergunta "isso está bom o suficiente?" — o gate responde. Agente se auto-verifica sem revisor humano |
| **3** | **Fixture canônica + conftest + golden e2e da CLI** | VG (CLI, loader), RP-03, SO-06, R-04 | **M** | **Alta** | Dá ao agente o teste "rodei o produto de verdade e a saída bate". Hoje só `--help` é verificado |
| **4** | **Run manifest (`--report`) + `gzcmd replay` + template de bug report** | RP-01, RP-02, RP-05 | **M** | **Alta** | Fecha o ciclo bug-report→reprodução sem humano traduzindo relatos em invocações |
| **5** | **Dedup dos pipelines gêmeos (builder compartilhado) + teste de paridade** | SO-01 | **M** | **Média-alta** | Maior risco estrutural de mudança silenciosamente errada; depois disso, mudar módulo-base exige entender 1 caminho, não 2 |
| 6 | Testes comportamentais: policy engine core, metrics, splitting, reporting | VG core | M | Média-alta | O coração decisório sai de import-only para table-driven |
| 7 | Disposição dos dormentes (`experimental/`) + mock harness LLM | SO-03, CP-02 | M | Média | Elimina 39% de superfície falsa; LLM vira testável offline |
| 8 | Logging estruturado + log assertions | VG logging | S–M | Média | Evidência de execução vira artefato verificável (e alimenta o manifest) |
| 9 | Invariantes automatizados do notebook | CP-05 | M | Média | Substitui o eyeball QA por asserções |
| 10 | Property tests do loader + fix definitivo R-04 | VG loader | S | Média | Formatos de data mistos são armadilha clássica p/ agente |

**Especificação concreta dos top 5:** cada um está detalhado como fase executável (pre-flight, tasks, subtasks, testes, critérios de aceitação, DoD, QA review) em `docs/plans/agent-native-plan.md` — item 1 = Wave 0; item 2 = Wave 1; item 3 = Wave 2 (Fases 2.1–2.2); item 4 = Wave 3; item 5 = Wave 4 (Fase 4.1). Um agente pode iniciar qualquer um deles a partir apenas daquele documento.

---

## 6. Riscos da própria auditoria

- Goldens numéricos com sklearn/xgboost podem variar entre plataformas (Windows dev × ubuntu CI) — mitigação especificada no plano (arredondamento, tolerâncias, invariantes em vez de floats crus onde necessário).
- pyright pode revelar dívida de tipos grande — mitigação: nível `basic` + ratchet.
- Mover módulos dormentes pode quebrar consumidores externos desconhecidos — mitigação: shims de import com aviso de deprecação.
