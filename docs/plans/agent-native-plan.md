# Plano de Execução — Agent-Native Hardening do gzcmd-record-linkage

**Data:** 2026-07-07
**Origem:** `docs/agent-native-audit.md` (IDs CP-xx, VG, RP-xx, SO-xx referenciados abaixo)
**Meta:** ao final, um agente de código consegue pegar um bug report ou feature, e sozinho: reproduzir → implementar → testar → verificar em build real, guiado apenas pelos documentos deste repositório.

---

## 0. Diretrizes de Execução (LEIA ANTES DE COMEÇAR)

### D1 — Execução iterativa contínua (sem interrupções)
Execute as waves em ordem (0→5), iterando fase a fase **sem parar para pedir aprovação**. Só interrompa e consulte o humano se: (a) surgir ambiguidade que altere o resultado de forma material e não coberta por `docs/decisions.md` nem por este plano; (b) problema **crítico** (perda de dados, quebra irreversível de compatibilidade); (c) **bloqueio real** (dependência externa indisponível, gate impossível de satisfazer). Qualquer suposição assumida no lugar de uma pergunta deve ser registrada na Seção 10 (Log de Execução).

### D2 — Commit frequente (commit often)
Mínimo de 1 commit por subtask concluída e verificada. Conventional Commits: `feat:`, `fix:`, `test:`, `docs:`, `ci:`, `refactor:`, `style:`, `chore:`. Nunca misturar reformatação em massa com mudança lógica no mesmo commit (reformatação = commit `style:` isolado). Nunca commitar com suite vermelha, exceto commits `wip:` em branch de trabalho — proibidos na branch principal.

### D3 — Pre-flight check antes de cada fase
Executar o checklist padrão da Seção 3 + itens específicos da fase. Falhou pre-flight → resolver antes de iniciar a fase.

### D4 — Senior QA Review ao final de cada fase
Ao final de cada fase, um review com a rubrica da Seção 4, gravado em `docs/plans/qa/agent-native/REVIEW-<fase>.md`. **Todo achado 🔴 ou 🟡 deve ser corrigido antes de avançar** (correções entram como subtasks extras da própria fase, com commits próprios). Achados 🟢 são registrados. QA da fase: `[tier:medium]`. QA global (Fase 5.2): `[tier:heavy]`.

### D5 — Definition of Done dupla
Cada fase tem CA (Critério de Aceitação) + DoD próprios; o plano tem CA-Global e DoD-Global (Seção 5). Fase só está Done quando: CA da fase satisfeito + DoD da fase satisfeito + QA review sem 🔴/🟡 pendente.

### D6 — Roteamento de modelos (model-router)
Toda task é anotada com `[tier:fast|medium|heavy]`:
- `[tier:fast]` — leitura, inventário, verificação, coleta de evidência (subagente barato, read-only).
- `[tier:medium]` — implementação, testes, docs, correções (implementador padrão).
- `[tier:heavy]` — decisões arquiteturais, QA global, debugging após 2 falhas consecutivas.
Orquestração, síntese e vereditos ficam no orquestrador.

### D7 — Não poluir a biblioteca publicada (herdada do plano do notebook)
`src/gzcmd_record_linkage/` não recebe artefatos de apresentação. Dependências de runtime da lib não crescem sem DEC registrada. Exceção deliberada deste plano: novos módulos de produto (`run_report.py`, `replay` em `cli.py`, `pipeline.py`, subpacote `experimental/`) SÃO biblioteca — permitidos.

### D8 — Paralelismo seguro (ownership de arquivos)
Maximizar paralelismo **entre tasks com conjuntos de arquivos disjuntos**, orquestrando assim:
1. Cada task declara seu **ownership**: lista exata de arquivos que pode criar/editar (tabelas por fase). Um arquivo pertence a **no máximo 1 task ativa** por vez.
2. **Proibido**: dois agentes escreverem o mesmo arquivo em paralelo; um agente ler um arquivo que está listado no ownership de outra task ativa (estado potencialmente inconsistente). Se precisar ler, aguarde a task dona commitar.
3. Fases/tasks marcadas com `∥` podem rodar em paralelo (ownership disjunto verificado). Sem a marca, execução sequencial.
4. O orquestrador serializa a integração: ao final de cada task paralela, o subagente reporta, o orquestrador verifica (build/testes) e **commita antes** de despachar nova task que toque arquivos vizinhos.
5. `pyproject.toml`, `.github/workflows/ci.yml` e `CLAUDE.md` são **arquivos de integração**: só o orquestrador (ou uma única task designada) os edita em cada fase — nunca dois agentes ao mesmo tempo.

### D9 — Idioma e estilo
Docs e QA reviews: PT-BR (termos técnicos em inglês OK). Código, identificadores, mensagens de log e de commit: inglês. Segue `ruff` line-length 88, target py310.

---

## 1. Legenda do Model-Router

| Tag | Uso típico neste plano |
| --- | --- |
| `[tier:fast]` | Pre-flights, inventários, greps de verificação, coleta de números para goldens |
| `[tier:medium]` | Todas as implementações, testes novos, edição de docs, QA review de fase |
| `[tier:heavy]` | Fase 4.1 (design do builder), arbitragem de decisões DEC novas se ambíguas, QA Global (5.2) |

---

## 2. Estado atual e alvo

**Baseline (verificado na auditoria):** 101 testes verdes offline; lint só em `cli.py`+tests; coverage gate só em 4 módulos; pyright nunca roda; zero logging no pipeline vivo; sem CLAUDE.md; sem conftest; sem manifest/replay; pipelines gêmeos `runner.py`×`eval.py`; 4 módulos dormentes (~39% do src).

**Alvo:** ver CA-Global (Seção 5.1). Em uma frase: *tudo que o agente precisa saber está escrito; tudo que o agente muda é verificável por comando; todo bug report é reproduzível por comando.*

Comandos canônicos de verificação (usados em todos os pre-flights e DoDs):

```pwsh
python -m pip install -e .[dev]
pytest -q                                   # suite completa (rápida; exclui -m notebook por padrão? NÃO — roda tudo exceto se marcado)
ruff check .                                # após Wave 1: escopo total
ruff format --check .
pyright                                     # após Wave 1
python -m build
```

---

## 3. Pre-flight Check (Checklist Padrão)

Executar antes de CADA fase `[tier:fast]`:

1. `git status` limpo (working tree sem restos da fase anterior); branch de trabalho correta.
2. `python -m pip install -e .[dev]` sem erro (ambiente íntegro).
3. `pytest -q` **verde** (baseline da fase; anotar contagem de testes).
4. Gates já ativados até aqui passam (`ruff check` no escopo vigente; `pyright` após Wave 1).
5. Ler as seções deste plano da fase + entradas relevantes de `docs/decisions.md` + `CLAUDE.md` (após Wave 0).
6. **Ownership (D8):** confirmar que nenhum arquivo da fase está no ownership de outra task ativa.
7. Itens específicos da fase (listados em cada fase).

Registrar o resultado do pre-flight no topo do REVIEW da fase (1 linha por item: ok/desvio).

---

## 4. Senior QA Review (Rubrica Padrão)

Arquivo: `docs/plans/qa/agent-native/REVIEW-<fase>.md`. Formato house (igual aos REVIEWs existentes): `# Senior QA Review — Fase <X>` → `## Escopo revisado` → `## Achados` (🔴 bloqueante / 🟡 corrigir antes de avançar / 🟢 nota) → `## Eixos da rubrica` → `## Veredito` (Done / Rework).

Eixos:
1. **Correção funcional** — a mudança faz o que a task pede; casos de borda do catálogo (4.1) exercitados.
2. **Cobertura de testes** — novos testes cobrem happy path + edge cases; coverage não regride; módulos tocados entram no gate quando previsto.
3. **Determinismo/reprodutibilidade** — mesmo input → mesmo output (2 execuções consecutivas idênticas); seeds fixas; nada depende de rede/relógio (exceto timestamps isolados em manifest).
4. **Qualidade de código** — `ruff check`/`ruff format --check`/`pyright` limpos no escopo vigente.
5. **Docs sincronizadas** — CLAUDE.md, README, `docs/decisions.md`, `contrato-api.md` refletem a mudança.
6. **Fronteiras (D7/SO-05)** — lib não poluída; deps de runtime inalteradas (ou DEC registrada).
7. **Compatibilidade** — flags da CLI e imports públicos preservados (ou shim + DEC).
8. **Segurança/PII** — sem segredos/PII em fixtures, goldens, logs, manifests; `FIREWORKS_API_KEY` nunca logada/gravada.

### 4.1 Catálogo de Edge Cases (referência para todas as fases)
- **CSV**: vazio; só header; coluna obrigatória ausente (`COMPREC`,`REFREC`,`PASSO`,`PAR` → `KeyError` claro); `na_values` (`""`, `" "`, `"NA"`, `"N/A"`, `"null"`, `"None"`); decimal `,`; encoding latin-1 (fallback); datas `R_DTOBITO` em **DDMMYYYY** vs demais **YYYYMMDD**; colunas ENDERECO ausentes → `endereco_zero=1`.
- **Política/engine**: budget LLM = 0; budget esgotado no meio; `p_cal` ∈ {0.0, 1.0}; valores exatamente no limite de banda; empate de expected loss; modo inexistente → erro claro.
- **CLI**: `--p-cal load_platt` sem `--platt-model` → erro; `--output` em diretório inexistente → auto-create; flags conflitantes.
- **Métricas**: denominador zero (nenhum positivo/negativo); tudo-match; tudo-non-match.
- **Splitting**: grupo único; grupos menores que test_size; mesmo seed → mesmo split; grupos não vazam entre train/test.
- **Replay/manifest**: manifest corrompido/JSON inválido; hash de input divergente; versão de pacote divergente (warning, não erro); saída divergente → exit ≠ 0 com diff legível.
- **Windows×Linux**: paths com `Path` (nunca strings concatenadas); goldens de float com arredondamento (ver R-P1).

---

## 5. Critérios Globais

### 5.1 Critério de Aceitação Global (CA-G)

| ID | Critério |
| --- | --- |
| CA-G1 | Um agente novo, lendo **apenas** `CLAUDE.md`, consegue: instalar, rodar testes, lint, type-check, executar o e2e canônico da CLI e localizar o módulo dono de qualquer subsistema — sem perguntar nada |
| CA-G2 | `pytest -q` verde; suite inteira offline e determinística (2 execuções seguidas = mesmos resultados) |
| CA-G3 | `ruff check .` e `ruff format --check .` limpos (src + tests + notebooks/*.py) |
| CA-G4 | `pyright` sem erros no nível configurado, rodando no CI |
| CA-G5 | Coverage: módulos core (lista da Fase 1.3) ≥80%; gate global ≥75% após Wave 4 e ≥80% ao final, com `experimental/` omitido do denominador |
| CA-G6 | Golden e2e: `run`, `fit-calibration`, `eval` executam sobre o CSV sintético versionado com goldens estáveis |
| CA-G7 | Ensaio de reprodução: um bug report sintético (texto + `run_report.json`) é reproduzido de ponta a ponta via `gzcmd replay` por um agente, sem intervenção humana, documentado no REVIEW global |
| CA-G8 | Paridade `runner`×`eval` travada por teste automatizado |
| CA-G9 | Zero módulos fantasma: todo módulo em `src/` ou é vivo e testado comportamentalmente, ou está em `experimental/` com banner e teste de contrato |
| CA-G10 | Invariantes do notebook verificados por asserções (`-m notebook`) |
| CA-G11 | CI verde em Python 3.11 e 3.12 com todos os gates novos ativos |

### 5.2 Definition of Done Global (gate de encerramento)
1. Todas as fases com veredito **Done** e zero 🔴/🟡 pendentes.
2. CA-G1…CA-G11 verificados item a item com evidência, gravados em `docs/plans/qa/agent-native/conformidade-final.md` (tabela: ID | Critério | Status | Evidência).
3. `docs/agent-native-audit.md` atualizado: cada CP/VG/RP/SO com status (resolvido/parcial/aberto + fase).
4. QA Global (`[tier:heavy]`) executado, achados corrigidos, `REVIEW-GLOBAL.md` gravado.
5. Working tree limpo, tudo commitado, CI verde no último commit.

---

## 6. Decisões de Projeto (novas — registrar em `docs/decisions.md` na Fase 0.2)

| ID | Decisão | Default deste plano (usar se não houver objeção humana — D1) |
| --- | --- | --- |
| DEC-11 | Disposição dos módulos dormentes | Mover `llm_review.py`, `llm_dossier.py`, `drift_monitor.py`, `runner_api.py` para `gzcmd_record_linkage/experimental/`, deixando shims de import no local antigo com `DeprecationWarning` |
| DEC-12 | Estratégia de coverage | Ratchet por lista de módulos no gate + `fail_under` global progressivo (75→80); `experimental/` em `[tool.coverage.run] omit` |
| DEC-13 | Formato de logging | `logging` stdlib, formato `key=value` (ex.: `run_id=... stage=load rows=608`), logger raiz `gzcmd`, nível via `--log-level` (default WARNING; INFO em `--report`) |
| DEC-14 | Fixture canônica | `data/synthetic/comparador_sintetico.csv` é O input canônico de testes/exemplos; goldens derivam dele com seeds fixas |
| DEC-15 | Nível do pyright | `basic` em `src/gzcmd_record_linkage/`; `tests/` e `notebooks/` excluídos inicialmente (ratchet futuro) |
| DEC-16 | Goldens | `tests/golden/`; floats arredondados a 6 casas na serialização; regeneração via `GZCMD_UPDATE_GOLDEN=1 pytest <teste>`; diff de golden sempre revisado em commit próprio `test(golden):` |
| DEC-17 | Manifest/replay | `run_report.json` schema versionado (`report_version: 1`); replay compara contagens + hashes com igualdade exata e floats com `atol=1e-9` local / `1e-6` cross-platform |

---

## 7. Waves, Fases, Tasks e Subtasks

Ordem: Wave 0 → 1 → 2 → 3 → 4 → 5. Dentro de cada wave, `∥` marca paralelismo permitido (D8).

---

### 🌊 WAVE 0 — Convenções escritas (audit #1: CP-01, CP-02, CP-04, CP-06, CP-07, CP-08, RP-05)

Elimina o maior chokepoint: conhecimento tribal vira texto executável.

#### Fase 0.1 — CLAUDE.md `∥ com 0.2`

**Ownership:** `CLAUDE.md` (novo, raiz).
**Pre-flight extra:** confirmar números do baseline (contagem de testes, módulos) com `[tier:fast]`; reler `docs/agent-native-audit.md` seções 0–1.

**Tasks:**
1. `[tier:medium]` Criar `CLAUDE.md` com EXATAMENTE estas seções:
   1.1. **O que é** — 3 linhas (motor de política GZ-CMD++ v3 para triagem de revisão clerical em record linkage; lib+CLI; sem serviço/DB).
   1.2. **Comandos canônicos** — tabela: instalar (`python -m pip install -e .[dev]`), testar (`pytest -q`), testar notebook (`pytest -q -m notebook`), lint (`ruff check .`), format (`ruff format .`), types (`pyright`), build (`python -m build`), e2e exemplo (`gzcmd run --mode vigilancia --input data/synthetic/comparador_sintetico.csv --output out/run.csv`), eval exemplo, fit-calibration exemplo. Nota: até a Wave 1, lint/types têm escopo reduzido — a tabela nasce com o alvo final e um aviso "gates ativados progressivamente, ver plano".
   1.3. **Mapa de módulos** — tabela: módulo | status (**LIVE**/**DORMANT**) | responsabilidade (1 linha) | testes que o cobrem. Dados da auditoria (Seção 1, CP-02). Incluir aviso: *"llm_review/llm_dossier/drift_monitor/runner_api NÃO são alcançáveis pela CLI; o caminho vivo de review LLM é o stub `nb_helpers.llm_review_stub`"*.
   1.4. **Regras que vencem conflitos** — (a) `docs/plans/qa/contrato-api.md` vence a config YAML (R-11/CP-08); (b) formatos de data por coluna (R_DTOBITO=DDMMYYYY, demais=YYYYMMDD); (c) CSV `sep=";"`, `decimal=","`.
   1.5. **Modos e flags da CLI** — o que é `vigilancia` vs `confirmacao`; tabela mínima de flags com defaults (fonte: auditoria §CP-07); exemplo canônico executável de cada subcomando.
   1.6. **Fronteiras** — D7; `notebooks/` está no pythonpath de teste mas não é lib; mudar `synthetic_data.py` exige rodar também os testes da lib (SO-05); twin pipelines: *"mudou loader/classifier/calibration/bands/guardrails → rode testes de runner E eval"* (até Wave 4 remover a duplicação).
   1.7. **Quality gates** — tabela: gate | comando | escopo | quando roda (local/CI). Manter sincronizada a cada wave.
   1.8. **Fluxo de release** — versão via git tag (`hatch-vcs`): taguear `vX.Y.Z` → `python -m build` → artefatos em `dist/`.
   1.9. **Convenções** — D2 (commits), D9 (idioma), política de skip/xfail (proibido sem justificativa escrita no teste — herda DF-1), regra de goldens (DEC-16).
   1.10. **Reprodução de bugs** — como pedir/usar `run_report.json` + `gzcmd replay` (nasce como "planejado — Wave 3", atualizar quando entregue); tabela cenário-sintético→classe-de-bug (RP-06): homonímia→guardrails de nome; datas trocadas→guardrail temporal; etc. (extrair cenários reais de `notebooks/synthetic_data.py` com `[tier:fast]`).
2. `[tier:medium]` Validação: pedir a um subagente `[tier:fast]` que, lendo SÓ o CLAUDE.md, liste os comandos para (a) rodar testes, (b) rodar o e2e, (c) descobrir se `drift_monitor` é vivo. Corrigir lacunas encontradas.

**Novos testes da fase:** N/A (documento). Verificação substituta: task 2 (dry-run de leitura) + `pytest -q` inalterado.
**Critério de Aceitação:** CLAUDE.md existe com as 10 seções; dry-run da task 2 respondido corretamente pelo subagente; nenhum comando documentado falha quando executado (exceto os marcados "planejado").
**DoD:** CA satisfeito + commit `docs: add CLAUDE.md agent conventions` + QA review da fase gravado.

#### Fase 0.2 — `docs/decisions.md` + templates GitHub `∥ com 0.1`

**Ownership:** `docs/decisions.md` (novo), `.github/ISSUE_TEMPLATE/bug_report.md` (novo), `.github/ISSUE_TEMPLATE/feature_request.md` (novo), `.github/pull_request_template.md` (novo).
**Pre-flight extra:** `[tier:fast]` extrair DEC-01…DEC-10 e R-01…R-13 vigentes do plano do notebook e `contrato-api.md`.

**Tasks:**
1. `[tier:medium]` Criar `docs/decisions.md` (ADR-lite append-only): tabela por decisão (ID | data | decisão | racional | fonte original | status). Migrar DEC-01…10 (resumo 1 linha + link), registrar DEC-11…17 (Seção 6 deste plano), registrar riscos vivos relevantes (R-04 resolvido, R-10, R-11, R-12) com status.
2. `[tier:medium]` `bug_report.md` exigindo: comando exato executado; versão (`pip show gzcmd-record-linkage` até existir `--version`; atualizar na Wave 3); `run_report.json` anexado (marcar "disponível a partir da Wave 3"); amostra mínima de input ou saída do anonymize; stdout completo; SO/Python.
3. `[tier:medium]` `pull_request_template.md` com checklist espelhando os gates (pytest/ruff/pyright/coverage/docs sync/CLAUDE.md atualizado se convenção mudou).

**Novos testes da fase:** N/A (docs). Verificação: arquivos existem; DEC-11…17 registradas; links válidos (checagem manual `[tier:fast]`).
**Critério de Aceitação:** decisões consolidadas em arquivo único citável; templates ativos no GitHub (paths corretos `.github/ISSUE_TEMPLATE/`).
**DoD:** CA + commits (`docs: add decision log`, `chore: add github issue/pr templates`) + QA review.

**QA da Wave 0** (após 0.1 e 0.2): rubrica completa; foco no eixo 5 (docs) e 8 (nenhum segredo citado). Corrigir achados antes da Wave 1.

---

### 🌊 WAVE 1 — Gates uniformes (audit #2: CP-03)

Sequencial internamente (todas as fases tocam `pyproject.toml`/`src/` amplamente — D8 proíbe paralelo aqui).

#### Fase 1.1 — Ruff em tudo

**Ownership:** `pyproject.toml` (seção ruff), `.github/workflows/ci.yml`, `src/**/*.py`, `tests/**/*.py`, `notebooks/*.py` (correções de lint), `CLAUDE.md` (seção gates).
**Pre-flight extra:** `[tier:fast]` rodar `ruff check . --statistics` e `ruff format --check .` para inventariar violações ANTES (registrar contagem no REVIEW).

**Tasks:**
1. `[tier:medium]` Rodar `ruff format .` → commit isolado `style: apply ruff format repo-wide` (zero mudança lógica; suite verde antes e depois).
2. `[tier:medium]` Corrigir violações de `ruff check .` (regras E,F,I,UP,B,SIM já configuradas). Violações que exigirem mudança de comportamento → **não corrigir silenciosamente**: adicionar `# noqa: <regra>` + comentário + entrada no Log (Seção 10). Commit `fix: resolve ruff violations across src and tests`.
3. `[tier:medium]` Atualizar CI: `ruff check .` e `ruff format --check .` (substituindo o escopo `cli.py tests`). Remover a execução duplicada de `test_cli_module_help_smoke` (redundância apontada na auditoria). Commit `ci: lint full repo, drop redundant smoke rerun`.
4. `[tier:medium]` Atualizar CLAUDE.md §gates.

**Novos testes:** nenhum teste de produto; o gate É a verificação. Edge cases: garantir que `notebooks/build_notebook.py` (per-file-ignore E501 existente) continua contemplado.
**CA:** `ruff check .` e `ruff format --check .` limpos local e no CI; suite verde; diff de format sem mudança de comportamento (suite prova).
**DoD:** CA + CI verde + QA review (eixo 4 central) + achados corrigidos.

#### Fase 1.2 — Pyright ativo

**Ownership:** `pyproject.toml` (nova seção `[tool.pyright]`), `.github/workflows/ci.yml`, `src/**/*.py` (anotações/correções), `CLAUDE.md`.
**Pre-flight extra:** `[tier:fast]` rodar `pyright src/gzcmd_record_linkage` (nível basic) e inventariar erros por módulo.

**Tasks:**
1. `[tier:medium]` Configurar `[tool.pyright]`: `include=["src"]`, `typeCheckingMode="basic"`, `pythonVersion="3.10"` (DEC-15).
2. `[tier:medium]` Corrigir erros. Proibido `# type: ignore` sem comentário de justificativa; proibido afrouxar o nível para passar. Se um módulo dormente concentrar erros demais, permitido excluí-lo explicitamente em `exclude` com comentário `# DEC-11: dormant, moves to experimental in Wave 4` (registrar no Log).
3. `[tier:medium]` Adicionar step `pyright` no CI (após install). Commit `ci: enforce pyright basic on src`.
4. `[tier:medium]` CLAUDE.md §gates.

**Novos testes:** o gate é a verificação; adicionalmente `pytest -q` prova que anotações não mudaram comportamento.
**CA:** `pyright` exit 0 local e no CI; zero `type: ignore` sem justificativa.
**DoD:** CA + CI verde + QA review + correções.

#### Fase 1.3 — Coverage ratchet + pre-commit

**Ownership:** `pyproject.toml` (coverage), `.github/workflows/ci.yml`, `.pre-commit-config.yaml` (novo), `CLAUDE.md`, `README.md`.
**Pre-flight extra:** `[tier:fast]` medir coverage atual por módulo (`pytest --cov=gzcmd_record_linkage --cov-report=term-missing`) e registrar tabela no REVIEW.

**Tasks:**
1. `[tier:medium]` Mover o gate para fonte única em `pyproject.toml`: `[tool.coverage.report] fail_under = 60` (global inicial conservador — será ratcheted: 75 na Wave 4, 80 na Wave 5); manter lista core (loader, classifier, bands, guardrails) ≥80 via job CI dedicado (como hoje). Registrar plano de ratchet em `docs/decisions.md` (DEC-12).
2. `[tier:medium]` CI: substituir flags soltas por `pytest --cov=gzcmd_record_linkage --cov-report=term-missing` + gate core. Commit `ci: single-source coverage config with ratchet`.
3. `[tier:medium]` `.pre-commit-config.yaml`: hooks `ruff check --fix`, `ruff format`, `end-of-file-fixer`, `trailing-whitespace`. Documentar `pre-commit install` no CLAUDE.md e README (opcional local; CI é o gate soberano).
4. `[tier:medium]` Sincronizar README (comandos de teste/lint atuais).

**Novos testes:** N/A (infra). Verificação: rodar CI localmente equivalente (cada comando do workflow) — tudo verde.
**CA:** `fail_under` ativo e passando; core ≥80 mantido; pre-commit instala e roda limpo em um commit de teste.
**DoD:** CA + CI verde 3.11/3.12 + QA review da wave (eixos 2 e 4) + correções.

---

### 🌊 WAVE 2 — Verificação e2e + fixtures (audit #3 e #6: VG-CLI, VG-core, SO-06, R-04, RP-03)

#### Fase 2.1 — conftest + fixture canônica (fecha R-04)

**Ownership:** `tests/conftest.py` (novo), `tests/test_loader.py` (fixture corrigida), `CLAUDE.md` (§reprodução).
**Pre-flight extra:** `[tier:fast]` reler `contrato-api.md` §colunas RAW; confirmar assinatura de `generate_comparador` em `notebooks/synthetic_data.py`.

**Tasks:**
1. `[tier:medium]` Criar `tests/conftest.py`:
   - fixture `synthetic_csv_path` (session): retorna `Path("data/synthetic/comparador_sintetico.csv")` (DEC-14), com skip claro se ausente;
   - fixture `comparador_df_factory` (session): fábrica `(seed=42, n_pairs=..., scenarios=...) -> DataFrame` via `synthetic_data.generate_comparador`;
   - fixture `canonical_row_dict`: **um dicionário com os nomes EXATOS do contrato** (as 6 colunas ENDERECO corretas, datas nos formatos por coluna) — derivado de `contrato-api.md`, com comentário apontando para ele.
2. `[tier:medium]` Corrigir `tests/test_loader.py::_base_df` para usar os nomes ENDERECO do contrato (fecha R-04 em código, não só em doc). Adicionar asserção de que `endereco_score_total > 0` no fixture corrigido (prova de que a correção pegou).
3. `[tier:medium]` Novos testes de propriedade do loader (`tests/test_loader.py`): datas DDMMYYYY vs YYYYMMDD; `na_values`; coluna obrigatória ausente → `KeyError`; colunas ENDERECO ausentes → `endereco_zero=1`; decimal `,`.

**Novos testes da fase (mínimo):** 6+ testes (itens acima). Edge cases do catálogo §4.1-CSV cobertos.
**CA:** conftest importável pela suite toda; R-04 fechado (fixture correta + asserção); loader ≥85% coverage; suite verde.
**DoD:** CA + commits (`test: add canonical conftest fixtures`, `fix(test): align loader fixture with verified API contract (R-04)`, `test: loader edge cases`) + QA review.

#### Fase 2.2 — Golden e2e da CLI `∥ com 2.3`

**Ownership:** `tests/test_cli_e2e.py` (novo), `tests/golden/` (novo dir), `tests/golden/README.md`.
**Pre-flight extra:** `[tier:fast]` executar manualmente os 3 subcomandos sobre o CSV sintético e capturar as saídas (números para os goldens); registrar no REVIEW.

**Tasks:**
1. `[tier:medium]` `tests/test_cli_e2e.py` chamando `cli.main([...])` **in-process** (não subprocess — mais rápido e coberto pelo coverage) com `tmp_path`:
   - `run --mode vigilancia --input data/synthetic/comparador_sintetico.csv --output <tmp>/out.csv --p-cal fit_platt` → asserta: exit 0; arquivo existe; nº de linhas; colunas de saída; `guardrail_counts`/`action_counts` idênticos ao golden; stdout contém chaves do resumo;
   - o mesmo para o segundo modo do config (`confirmacao`);
   - `fit-calibration --output-model <tmp>/platt.json` → exit 0; JSON válido; chaves do modelo; params ≈ golden (`atol=1e-6`);
   - `eval --seeds 42 --metrics-out <tmp>/m.csv --summary-out <tmp>/s.csv --latex-out <tmp>/t.tex` → exit 0; CSVs parseáveis (`sep=";"`); métricas ≈ golden; LaTeX contém `\begin{tabular}` e o label default;
   - erro esperado: `run --p-cal load_platt` sem `--platt-model` → exit ≠ 0 com mensagem clara; CSV vazio → falha clara (não traceback críptico — se hoje for críptico, registrar 🟡 e melhorar mensagem em `loader.py` como subtask).
2. `[tier:medium]` Infra de golden (DEC-16): helper no conftest `golden_compare(name, payload)` que compara JSON serializado (floats arredondados a 6 casas) com `tests/golden/<name>.json`; `GZCMD_UPDATE_GOLDEN=1` regrava. `tests/golden/README.md` explica regeneração e política de revisão de diffs.
3. `[tier:medium]` Rodar a suite 2× seguidas → resultados idênticos (eixo 3).

**Novos testes:** 8+ testes e2e. Edge cases: §4.1-CLI completos.
**CA:** 3 subcomandos com goldens estáveis 2 execuções; falhas de flag com mensagens claras; coverage de `cli.py` ≥80%.
**DoD:** CA + commits + QA review (atenção especial R-P1: se CI ubuntu divergir dos goldens Windows, aplicar mitigação da Seção 9 ANTES do veredito).

#### Fase 2.3 — Testes comportamentais do core dormente-vivo `∥ com 2.2`

**Ownership:** `tests/test_policy_engine.py` (novo), `tests/test_metrics.py` (novo), `tests/test_splitting.py` (novo), `tests/test_reporting.py` (novo).
**Pre-flight extra:** `[tier:fast]` ler `gzcmd_v3_policy_engine.py` (125L), `metrics.py`, `splitting.py`, `reporting.py` na íntegra e extrair as tabelas de comportamento esperadas (assinaturas, invariantes) para o REVIEW.

**Tasks:**
1. `[tier:medium]` `test_policy_engine.py` — table-driven sobre `PolicyEngineV3`/`Budget`/`Costs`/`Decision`: matriz (p_cal, banda, custos, budget) → decisão esperada; budget=0; budget esgota no meio de um lote; p_cal ∈ {0,1}; empate de expected loss (documentar tie-break observado — vira contrato); `LLMError` propagação. Mínimo 10 casos.
2. `[tier:medium]` `test_metrics.py` — `confusion_counts`/`metrics_dict`: caso conhecido à mão; tudo-positivo; tudo-negativo; denominador zero (deve retornar 0/None documentado, não explodir). Mínimo 5 casos.
3. `[tier:medium]` `test_splitting.py` — determinismo por seed; group split não vaza `COMPREC`/`REFREC` entre folds; grupo único; test_size extremos. Mínimo 5 casos.
4. `[tier:medium]` `test_reporting.py` — golden do CSV de summary e do LaTeX (usa infra da 2.2 se disponível; senão golden local simples — coordenar com orquestrador para não editar conftest em paralelo: **conftest pertence à 2.1, já concluída**).

**Novos testes:** 20+ no total.
**CA:** os 4 módulos saem de import-only para comportamental; coverage de cada um ≥80%; adicioná-los à lista core do gate (edição de `pyproject.toml` feita PELO ORQUESTRADOR ao integrar — D8, arquivo de integração).
**DoD:** CA + commits por módulo + QA review da wave inteira (2.1+2.2+2.3) + correções.

---

### 🌊 WAVE 3 — Reprodução: manifest, replay, logging (audit #4 e #8: RP-01, RP-02, RP-04, VG-logging)

Sequencial (todas tocam `cli.py`).

#### Fase 3.1 — Run manifest (`--report`)

**Ownership:** `src/gzcmd_record_linkage/run_report.py` (novo), `src/gzcmd_record_linkage/cli.py`, `tests/test_run_report.py` (novo), `CLAUDE.md`, `.github/ISSUE_TEMPLATE/bug_report.md` (atualizar nota "Wave 3").
**Pre-flight extra:** padrão.

**Tasks:**
1. `[tier:medium]` `run_report.py`: dataclass `RunReport` + `build_report(...)` + `write_report(path)`. Campos (DEC-17): `report_version=1`, timestamp UTC, package version (`importlib.metadata`), python version, platform, versões de numpy/pandas/sklearn/xgboost, subcomando, flags efetivas (pós-parse, sem segredos), `input_path`+`input_sha256`+`input_rows`, `config_path`+`config_sha256`, seeds, resultados (rows, guardrail_counts, action_counts, p_cal_method/params, métricas quando `eval`), duração.
2. `[tier:medium]` `cli.py`: flag `--report <path>` nos 3 subcomandos; `--version` no parser raiz (fecha lacuna do template de bug). Sem `--report`, comportamento byte-idêntico ao atual (compat — eixo 7).
3. `[tier:medium]` `tests/test_run_report.py`: report gerado nos 3 subcomandos; schema completo; hash bate com hash independente do arquivo; **nenhuma chave de ambiente sensível presente** (grep por `FIREWORKS` no JSON); determinismo (2 runs → relatórios iguais exceto timestamp/duração).

**Novos testes:** 6+.
**CA:** `gzcmd run --mode vigilancia --input data/synthetic/comparador_sintetico.csv --report r.json` produz manifest completo e válido; `gzcmd --version` funciona; zero regressão.
**DoD:** CA + commits + QA review.

#### Fase 3.2 — `gzcmd replay`

**Ownership:** `src/gzcmd_record_linkage/replay.py` (novo), `src/gzcmd_record_linkage/cli.py`, `tests/test_replay.py` (novo), `CLAUDE.md` (§reprodução — remover "planejado"), `.github/ISSUE_TEMPLATE/bug_report.md`.

**Tasks:**
1. `[tier:medium]` `replay.py`: `load_report(path)` (valida `report_version`, JSON schema mínimo, erros claros); `replay(report, input_override=None)` → re-executa via `runner`/`eval` com flags/config do manifest → `ReplayResult` com comparação campo a campo (contagens exatas; floats `atol` DEC-17; hash do output quando houver).
2. `[tier:medium]` `cli.py`: subcomando `replay <report.json> [--input CSV]`; exit 0 = reproduziu igual; exit 2 = divergiu (imprime diff tabular legível: campo | esperado | obtido); exit 1 = erro de execução. Warning (não erro) quando versão do pacote difere.
3. `[tier:medium]` `tests/test_replay.py`: round-trip feliz (run --report → replay → exit 0); input adulterado → exit 2 com diff mencionando `input_sha256`; manifest corrompido → exit 1 mensagem clara; report de `eval` reproduz métricas; versão divergente → warning e prossegue.

**Novos testes:** 6+.
**CA:** ciclo completo demonstrado: `run --report` → editar nada → `replay` verde; adulterar 1 linha do CSV → `replay` vermelho com diff útil.
**DoD:** CA + commits + QA review.

#### Fase 3.3 — Logging estruturado + anonymize

**Ownership:** `src/gzcmd_record_linkage/runner.py`, `src/gzcmd_record_linkage/eval.py`, `src/gzcmd_record_linkage/cli.py` (`--log-level`), `src/gzcmd_record_linkage/anonymize.py` (novo), `tests/test_logging.py` (novo), `tests/test_anonymize.py` (novo), `CLAUDE.md`.

**Tasks:**
1. `[tier:medium]` Logging (DEC-13): logger `gzcmd.*`; eventos INFO por estágio no pipeline vivo (`run_id=<uuid8> stage=load rows=...`, `stage=calibrate method=...`, `stage=guardrails counts=...`, `stage=decide actions=...`); `--log-level` na CLI; default WARNING (stdout do usuário não muda — compat).
2. `[tier:medium]` `tests/test_logging.py` com `caplog`: run completo emite os 4 estágios com o MESMO `run_id`; nível respeitado; nenhuma linha com dado bruto de registro (só agregados — eixo 8).
3. `[tier:medium]` `anonymize.py` + subcomando `gzcmd anonymize --input real.csv --output anon.csv --date-shift-days N` (RP-04): re-id de `COMPREC`/`REFREC` (mapeamento determinístico por hash com salt), shift consistente de todas as datas (preservando deltas), subscores intocados. `tests/test_anonymize.py`: ids trocados, deltas de datas preservados, mesmo nº de linhas, roda no pipeline (`load_comparador_csv` aceita a saída), idempotência do mapeamento com mesmo salt.
4. `[tier:medium]` Integrar log level INFO automático quando `--report` presente (evidência rica no manifest workflow).

**Novos testes:** 8+.
**CA:** log assertions verdes; anonymize gera CSV que o loader aceita e o replay processa; stdout default inalterado.
**DoD:** CA + commits + **QA review da Wave 3 inteira** + correções.

---

### 🌊 WAVE 4 — Estrutura (audit #5 e #7: SO-01, SO-03, DEC-11)

#### Fase 4.1 — Builder compartilhado + teste de paridade

**Ownership:** `src/gzcmd_record_linkage/pipeline.py` (novo), `src/gzcmd_record_linkage/runner.py`, `src/gzcmd_record_linkage/eval.py`, `tests/test_pipeline_parity.py` (novo).
**Pre-flight extra:** `[tier:heavy]` (design): ler `runner.py` e `eval.py` na íntegra; produzir nota de design (no REVIEW): quais passos são idênticos, quais divergem legitimamente (ex.: split held-out da rota B), assinatura do builder. **Regra: refactor de extração pura — zero mudança de comportamento; goldens da Wave 2 são o detector.**

**Tasks:**
1. `[tier:heavy]` Design aprovado na nota de pre-flight (uma página: API de `pipeline.py`, ex.: `build_components(cfg, *, macd) -> PipelineComponents` + `score_frame(components, df, p_cal_source) -> ScoredFrame`).
2. `[tier:medium]` Implementar `pipeline.py`; refatorar `runner.py` e `eval.py` para consumi-lo. Imports públicos preservados (`run_v3`, `build_engine_from_config`, assinaturas intactas — eixo 7).
3. `[tier:medium]` `tests/test_pipeline_parity.py`: sobre o CSV sintético, para cada modo: `p_cal` (mesma fonte), bandas, flags de guardrail e decisões produzidos pelo caminho `runner` == caminho `eval` (rota in-sample), elemento a elemento. Caso de regressão: mudar um parâmetro de config e verificar que AMBOS mudam juntos.
4. `[tier:fast]` Verificação: goldens da Wave 2 intactos sem regeneração (prova de zero-mudança).

**Novos testes:** 3+ (paridade por modo + regressão conjunta).
**CA:** duplicação eliminada (`eval.py` não re-fia módulos-base diretamente — verificável por grep de imports); goldens intactos; paridade verde.
**DoD:** CA + commits (`refactor: extract shared pipeline builder`, `test: runner/eval parity lock`) + QA review.

#### Fase 4.2 — Quarentena dos dormentes (`experimental/`)

**Ownership:** `src/gzcmd_record_linkage/experimental/__init__.py` (novo), `src/gzcmd_record_linkage/experimental/{llm_review,llm_dossier,drift_monitor,runner_api}.py` (movidos), shims em `src/gzcmd_record_linkage/{llm_review,llm_dossier,drift_monitor,runner_api}.py`, `pyproject.toml` (coverage omit), `CLAUDE.md` (mapa de módulos), `tests/test_experimental_shims.py` (novo).
**Pre-flight extra:** `[tier:fast]` confirmar zero importadores internos dos 4 módulos (fora testes de smoke) — reconfirmação da auditoria pós-Waves 1–3.

**Tasks:**
1. `[tier:medium]` Mover os 4 módulos para `experimental/` (git mv, preserva história). Banner docstring em cada: status DORMANT, por que existe, o que falta para ativar, link p/ DEC-11.
2. `[tier:medium]` Shims no local antigo: `from .experimental.llm_review import *` + `DeprecationWarning` no import. `test_experimental_shims.py`: import antigo funciona e emite o warning; import novo funciona sem warning.
3. `[tier:medium]` `pyproject.toml`: `[tool.coverage.run] omit = ["*/experimental/*"]`; subir `fail_under` para **75** (ratchet DEC-12). Ajustar `test_env_smoke.py` se necessário (descoberta dinâmica de módulos deve continuar verde).
4. `[tier:medium]` Atualizar CLAUDE.md (mapa) e `docs/decisions.md` (DEC-11 status: aplicada).

**Novos testes:** 4+ (shims, warning, smoke pós-move).
**CA:** `src/` sem módulos fantasma no namespace principal; imports antigos compatíveis; `fail_under=75` verde; CA-G9 satisfeito para os 4 módulos.
**DoD:** CA + commits + QA review.

#### Fase 4.3 — Mock harness LLM (depois da 4.2)

**Ownership:** `src/gzcmd_record_linkage/experimental/llm_review.py` (injeção de transport), `tests/test_llm_review_contract.py` (novo), `tests/test_llm_dossier.py` (novo).
**Pre-flight extra:** `[tier:fast]` ler `llm_review.py` (697L) e mapear pontos de I/O HTTP (`requests.post` em ~:516) e leitura de env; ler `gzcmd_v3_llm_output.schema.json`.

**Tasks:**
1. `[tier:medium]` Injeção de dependência mínima: parâmetro `transport: Callable | None` (ou objeto com `.post`) na função/cliente que hoje chama `requests.post` diretamente; default preserva comportamento atual. Sem nova dependência de mock externa (D7) — fake caseiro nos testes.
2. `[tier:medium]` `tests/test_llm_review_contract.py`: fake transport devolvendo respostas conformes ao `gzcmd_v3_llm_output.schema.json` → fluxo dual-agent+arbiter completa offline; resposta malformada → erro tratado; rate-limit simulado → retry/backoff exercitado (com sleep monkeypatched — permitido monkeypatch de tempo); `FIREWORKS_API_KEY` ausente → `EnvironmentError` claro; chave NUNCA aparece em logs/erros.
3. `[tier:medium]` `tests/test_llm_dossier.py`: dossier de um par sintético não contém campos PII-like (lista negra de chaves), contém os subscores esperados; JSON serializável e validável.

**Novos testes:** 8+.
**CA:** caminho LLM 100% testável offline; schema validado em teste; zero rede na suite (verificável: rodar com rede desabilitada segue verde).
**DoD:** CA + commits + **QA review da Wave 4** + correções.

---

### 🌊 WAVE 5 — Notebook invariants + fechamento (audit #9: CP-05 + DoD global)

#### Fase 5.1 — Invariantes automatizados do notebook

**Ownership:** `tests/test_notebook_invariants.py` (novo), `notebooks/nb_helpers.py` (apenas se precisar expor helper de extração), `CLAUDE.md`.
**Pre-flight extra:** `[tier:fast]` inventariar quais números a QA manual conferia (ler `conformidade-final.md` e `auditoria-didatica.md`): ECE/Brier, contagens de decisão por banda, reconciliação manual×run_v3.

**Tasks:**
1. `[tier:medium]` `test_notebook_invariants.py` (marker `notebook`): executa o notebook via nbclient (reusando padrão de `test_notebook_execution.py`), extrai outputs das células-chave (por tag de célula ou parse de stdout) e asserta invariantes com ranges (não floats exatos): ECE dentro de [0, limite documentado]; Brier < limite; reconciliação manual×`run_v3` idêntica (esta é exata); contagem total de decisões == nº de pares; nenhum erro/warning inesperado no log de execução.
2. `[tier:medium]` Se necessário, adicionar tags nas células (via `build_notebook.py` — regenerar o `.ipynb` conforme fluxo documentado em `notebooks/README.md`).
3. `[tier:medium]` CI: job separado opcional `notebook` (manual/`workflow_dispatch` + semanal `schedule`) — NÃO no gate de PR (lento — R-P7). Documentar no CLAUDE.md.

**Novos testes:** 4+ invariantes.
**CA:** `pytest -q -m notebook` verde localmente com as novas asserções; eyeball QA substituído por asserções para os invariantes listados.
**DoD:** CA + commits + QA review.

#### Fase 5.2 — QA Global, conformidade e gate final

**Ownership:** `docs/plans/qa/agent-native/conformidade-final.md` (novo), `docs/plans/qa/agent-native/REVIEW-GLOBAL.md` (novo), `docs/agent-native-audit.md` (status update), `CLAUDE.md`/`README.md` (passada final).

**Tasks:**
1. `[tier:fast]` Coletar evidências: rodar TODOS os comandos canônicos (Seção 2) + CI verde no HEAD; 2 execuções da suite (determinismo); grep de módulos fantasma; imports públicos.
2. `[tier:heavy]` **Ensaio de reprodução (CA-G7)**: gerar um "bug report sintético" (com `anonymize` + `run --report` sobre um CSV de cenário com `scenarios` de homonímia); entregar a um subagente APENAS o texto do report + manifest + CLAUDE.md; ele deve reproduzir via `replay`, localizar o módulo responsável e propor o fix — sem nenhuma outra ajuda. Documentar transcript resumido no REVIEW-GLOBAL.
3. `[tier:heavy]` QA Global: rubrica completa sobre o repo inteiro; varredura CA-G1…11 → `conformidade-final.md` (tabela ID | Critério | Status | Evidência); varredura CP/VG/RP/SO da auditoria → atualizar status no `docs/agent-native-audit.md`.
4. `[tier:medium]` Corrigir TODOS os achados 🔴/🟡 do QA Global (subtasks + commits) e re-verificar.
5. `[tier:medium]` Ratchet final: `fail_under = 80`; passada final de sincronização de docs.

**CA:** CA-G1…CA-G11 todos ✅ com evidência.
**DoD:** DoD Global (Seção 5.2) integral. Veredito **Done** no REVIEW-GLOBAL.

---

## 8. Matriz de Rastreabilidade (Auditoria ↔ Fases)

| Achado da auditoria | Fase(s) |
| --- | --- |
| CP-01 (sem CLAUDE.md) | 0.1 |
| CP-02 (liveness tribal) | 0.1, 4.2 |
| CP-03 (gates não-uniformes) | 1.1, 1.2, 1.3, 5.2 |
| CP-04 (decisões espalhadas) | 0.2 |
| CP-05 (notebook eyeball) | 5.1 |
| CP-06 (release não documentado) | 0.1 |
| CP-07 (modos/flags tribais) | 0.1 |
| CP-08 / R-11 (config aspiracional) | 0.1 (regra), 2.1 (fixture) |
| VG loader/R-04 | 2.1 |
| VG CLI e2e | 2.2 |
| VG policy engine/metrics/splitting/reporting | 2.3 |
| VG llm mock harness | 4.3 |
| VG logging | 3.3 |
| VG pyright | 1.2 |
| RP-01 (manifest) | 3.1 |
| RP-02 (replay) | 3.2 |
| RP-03 (fixture canônica/exemplos) | 0.1, 2.1 |
| RP-04 (anonymize) | 3.3 |
| RP-05 (issue template) | 0.2, 3.1 |
| RP-06 (cenários→bugs) | 0.1 |
| SO-01 (twin pipelines) | 4.1 |
| SO-02 (config ripple) | 0.1 (regra), 2.3 (contratos) |
| SO-03 (dormentes) | 4.2, 4.3 |
| SO-04 (cli monolítica) | mitigada por 2.2/3.x (handlers testados); refactor pleno fora de escopo (registrar em decisions se necessário) |
| SO-05 (fronteira notebooks) | 0.1 (regra escrita); promoção a `testing/` fora de escopo — DEC futura |
| SO-06 (sem conftest) | 2.1 |

## 9. Registro de Riscos e Mitigações

| ID | Risco | Mitigação |
| --- | --- | --- |
| R-P1 | Goldens de float divergem Windows(dev)×ubuntu(CI) por BLAS/xgboost | Floats arredondados a 6 casas na serialização; comparação `atol=1e-6`; se persistir, converter o campo para invariante (range) e registrar no Log; goldens de contagens são exatos sempre |
| R-P2 | pyright revela dívida grande | Nível basic; exclusão explícita e comentada de dormentes (resolvidos na 4.2); nunca afrouxar global |
| R-P3 | Mover dormentes quebra consumidor externo desconhecido | Shims com `DeprecationWarning` por ≥1 minor release; testes de shim |
| R-P4 | Reformat em massa polui blame | Commit `style:` isolado; opcional `.git-blame-ignore-revs` |
| R-P5 | Agentes paralelos conflitam em arquivo | D8: ownership explícito por task; arquivos de integração só pelo orquestrador; commits serializados |
| R-P6 | Ratchet de coverage bloqueia wave | Gate global sobe só no fim das Waves 4 e 5, após os testes novos existirem; lista core cresce só quando módulo atinge 80 |
| R-P7 | Notebook lento demais p/ CI de PR | Job separado schedule/dispatch; marker `notebook` fora do gate de PR |
| R-P8 | `--report`/logging vazam segredo/PII | Testes explícitos (grep FIREWORKS no manifest; caplog sem dados brutos); eixo 8 em todo QA |

## 10. Log de Execução (preencher durante a execução)

### Suposições e desvios (D1)
| Data | Fase | Suposição/desvio | Racional |
| --- | --- | --- | --- |

### Decisões DEC resolvidas/alteradas
| DEC | Resolução | Data |
| --- | --- | --- |

### Achados de QA e correções
| Fase | Achado | Severidade | Correção (commit) |
| --- | --- | --- | --- |

### Gate final
- [ ] DoD Global integral (Seção 5.2)
- [ ] `conformidade-final.md` completo
- [ ] REVIEW-GLOBAL veredito Done
