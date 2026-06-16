# Plano de Execução — Notebook Didático "GZ-CMD Passo a Passo" sobre Dataset Sintético

> **Tipo de documento:** Plano de implementação e teste (modo planejamento).
> **Idioma do entregável:** Português do Brasil (PT-BR).
> **Público-alvo:** Técnico/acadêmico (mostra a matemática, curvas de calibração, custo esperado da política e métricas detalhadas).
> **Entregável principal:** Notebook Jupyter (`.ipynb`) que reproduz, passo a passo, o pipeline do `gzcmd-record-linkage` sobre um dataset **sintético** gerado para a apresentação.
> **Repositório:** `D:\git\gzcmd-record-linkage` · Plataforma: win32 · Shell: pwsh · Pacote: `gzcmd_record_linkage` (v0.2.0).

---

## 0. Diretrizes de Execução (LEIA ANTES DE COMEÇAR)

Estas diretrizes são **obrigatórias** e governam toda a execução deste plano.

### D1 — Execução iterativa contínua (sem interrupções)
Execute o plano **iterando wave a wave, fase a fase, sem parar para pedir confirmação** entre etapas. Só **PARE e consulte o humano** se ocorrer um destes três casos:
1. **Ambiguidade real:** uma decisão com múltiplas interpretações plausíveis e impacto significativo, sem default seguro.
2. **Problema crítico:** perda/corrupção de dados, quebra do ambiente, falha de segurança.
3. **Bloqueio duro:** dependência impossível de resolver (ex.: API obrigatória indisponível, incompatibilidade irreconciliável).
Fora desses casos, **continue** — registre suposições no log de execução e siga.

### D2 — Commit frequente (commit often)
- Faça commit ao final de **cada task relevante** e **obrigatoriamente** ao final de cada fase aprovada no QA.
- Use Conventional Commits em PT-BR/EN curto, escopo claro. Exemplos:
  - `feat(notebook): adiciona gerador de dataset sintético`
  - `test(notebook): cobre edge cases de guardrails no gerador`
  - `docs(plan): registra correções do QA da Fase 1.1`
  - `chore(deps): adiciona requirements/notebook.txt`
- **NUNCA** commitar segredos, dados reais ou `.env`. O dataset versionado é **100% sintético**.
- Não fazer `push`/PR sem solicitação explícita do humano (apenas commits locais).

### D3 — Pre-flight check antes de cada fase
Antes de iniciar **qualquer** fase, execute o **Pre-flight Check** (checklist padrão na Seção 3 + itens específicos da fase). Se um item do pre-flight falhar, **resolva antes de prosseguir**.

### D4 — Senior QA Review ao final de cada fase
Ao final de **cada fase**, execute uma **revisão de QA sênior** (critérios na Seção 4). **Corrija TODOS os pontos levantados** antes de marcar a fase como concluída. A correção é parte da fase, não opcional.

### D5 — Definition of Done dupla
Cada fase só é "Done" quando atende **(a)** seu DoD específico **e (b)** o DoD Global (Seção 5). Ao final do plano, validar o **Critério de Aceitação Global** e o **DoD Global**.

### D6 — Roteamento de modelos (model-router)
Respeite as anotações `[tier:*]` em cada task. Legenda na Seção 1. O orquestrador (Sisyphus/Opus) sintetiza e decide; execução read-only vai para `@fast`; implementação para `@medium`; arquitetura/QA sênior/reconciliação difícil para `@heavy` (ou local, se o orquestrador já for Opus).

### D7 — Não poluir a biblioteca publicada
Artefatos da apresentação (notebook, gerador, helpers, dados sintéticos, tooling de notebook) ficam **fora** de `src/gzcmd_record_linkage/`, salvo decisão explícita em contrário (ver Seção 6, Decisão DEC-01). A biblioteca `gzcmd` não deve ganhar dependências de apresentação.

---

## 1. Legenda do Model-Router

| Tag | Tier / Modelo | Quando usar nesta execução |
|-----|---------------|----------------------------|
| `[tier:fast]` | `@fast` (Haiku) — read-only | Buscar, ler, conferir assinaturas, listar arquivos, inspecionar `conftest.py`, validar schema. |
| `[tier:medium]` | `@medium` (GPT-5.x) — implementação | Escrever o gerador, células do notebook, testes, helpers, correções de QA, fixes de build/lint. |
| `[tier:heavy]` | `@heavy` (Opus/max) — arquitetura/QA | Revisão de QA sênior, design da reconciliação manual×`run_v3`, debugging após ≥2 falhas, análise de tradeoff. |
| `[self:opus]` | Orquestrador local | Síntese, decisões de escopo, gating entre fases, redação de relatórios de QA. **Se o orquestrador já é Opus, NÃO delega QA a `@heavy` — faz localmente.** |

> **Regra de custo:** trabalho read-only deve ir para `@fast` por padrão (≈20× mais barato que rodar a mesma busca no tier do orquestrador). Implementação para `@medium`. Antes de acionar `@heavy`, reúna contexto via `@fast`.

---

## 2. Sumário Executivo e Estado Atual

### 2.1 O que é o GZ-CMD (verificado no código)
Motor de decisão **GZ-CMD++ v3** para *record linkage* com triagem por custo e roteamento para revisão clerical/LLM. Contexto: deduplicação de registros pessoais (estilo CPF) e recuperação de óbitos. Pipeline real (`runner.py:94-228`):

```
load_comparador_csv()        # carrega pares "comparador" (sep=';', decimal=',', utf-8→latin-1)
   ↓  (feature engineering: agregados + flags + MACD opcional)
BandAssigner.assign(nota)    # binning de 'nota_final' → band (low … high)
   ↓
compute_p_cal()/Platt/XGB    # calibra probabilidade → coluna 'p_cal' ∈ [0,1]
   ↓
apply_guardrails(df)         # regras: óbito temporal, homonímia, âncoras → 'guardrail'/'guardrail_reason'
   ↓
PolicyEngineV3.triage(df)    # decisão por custo esperado → 'action' ∈ {MATCH, NONMATCH, LLM_REVIEW}
```

### 2.2 APIs reais confirmadas (ground truth — usar exatamente assim)
- `load_comparador_csv(path, *, cfg=LoadConfig(...))` — `loader.py`. CSV `;`/`,`. Colunas com prefixos OpenRecLink (ex.: `COMPREC,C,12,0`).
- `BandAssigner.from_config(cfg)` / `.assign(series) -> pd.Series` — `bands.py:18,21`. **Não muta** a entrada.
- `fit_platt_from_df(df, *, target_col="TARGET", ...) -> PlattModel` — `calibration.py:163`.
- `compute_p_cal(df, *, method, model=None, ...) -> pd.Series` — `calibration.py:145`. `method ∈ {"stub","platt"}`. **Retorna Series** (não grava no df).
- `GZCMDClassifier(config).fit(df)` / `.predict_proba(df) -> np.ndarray (n,2)` — `classifier.py:112,118,189`. Probabilidade de MATCH = `predict_proba(df)[:, 1]`. **Não existe `.predict()`.**
- `apply_guardrails(df, *, temporal_days=180, nota_always_match=10.0, nota_always_nonmatch=3.0, homonimia_min_nota=7.0, homonimia_year_gap=5.0) -> GuardrailOutput` — `guardrails.py:55`. Retorna dataclass com `.guardrail` e `.reason` (Series).
- `build_engine_from_config(cfg, *, mode, llm_used=0) -> PolicyEngineV3` — `runner.py:36-91`.
- `PolicyEngineV3.triage(df) -> pd.DataFrame` — `gzcmd_v3_policy_engine.py:72`. Lê `p_cal`, `band`, `guardrail` (opcional). **Muta o df** e adiciona: `action`, `evr`, `base_choice`, `base_loss`, `loss_llm`, `review_requested`. Atualiza `budget.llm_used`. **Não existe `.decide()`.**
- `run_v3(*, input_csv, config_path, mode, macd_enabled=True, llm_used=0, p_cal="fit_platt", ...) -> tuple[pd.DataFrame, RunSummary]` — `runner.py:94`.
- `RunSummary` (`runner.py:24-33`): `rows, llm_used, llm_max, actions, guardrails, review_requested, p_cal_method, p_cal_params`.
- Métricas: `confusion_counts`, `precision`, `recall`, `fbeta`, `f1`, `metrics_dict`, `ConfusionCounts` — `metrics.py`.
- Avaliação: `evaluate_v3_dataframe(...)` (`eval.py:131`), `evaluate_v3_csv(...)` (`eval.py:292`).
- Relatórios: `write_csv`, `summarize_runs`, `summary_to_latex_table`, `write_text` — `reporting.py`.

### 2.2.1 Nota metodológica CRÍTICA (verificada no código — base da revisão de calibração)
Dois fatos de implementação, confirmados linha a linha, **moldam toda a parte de calibração/avaliação do notebook**:

1. **`run_v3` calibra IN-SAMPLE (sem split).** Nas branches `fit_platt`/`fit_ml_rf`/`fit_ml_xgb` (`runner.py:135-180`), o modelo é **ajustado em TODAS as linhas e pontua as mesmas linhas** — não há treino/teste. `fit_platt` é determinístico (Newton-Raphson, sem RNG — `calibration.py:54-110`). Consequência: um *reliability diagram* feito sobre esses mesmos dados é **otimista por construção (vazamento)** e não mede generalização.
2. **`evaluate_v3_dataframe` faz o CERTO (held-out).** Ajusta o Platt no split de **treino** e prevê no **teste** (`eval.py:182-199`), com split potencialmente **group-aware** por `split_by ∈ {row, comprec, refrec}` (`splitting.py`). É a rota correta para estimar generalização.
3. **Config descreve MAIS do que o código faz.** A config diz calibração `anchor_platt` + `by_band: true`, mas **não há implementação** de calibração por banda/âncora no caminho do `run_v3` — é **Platt global** via `fit_platt_from_df` (`calibration.py:163`). O notebook deve ensinar **o que o código faz** e sinalizar a config como *roadmap/intenção de projeto*.

> **Implicação de design para o notebook (ver DEC-07):** separar claramente DUAS rotas — **(A) "reprodução fiel da ferramenta"** (in-sample, reconcilia exatamente com `run_v3`) e **(B) "metodologia ML correta"** (split treino/teste via `evaluate_v3_dataframe`, onde moram as métricas de generalização e o *reliability diagram* honesto). Misturar as duas é o erro metodológico central a evitar.

### 2.3 Schema de entrada (verificado em `tests/test_loader.py`)
Colunas obrigatórias (prefixo OpenRecLink ou canônico): `COMPREC,C,12,0`, `REFREC,C,12,0`, `PASSO`, `PAR` (1/2=match, 0=não-match), `nota final`, `R_DTNASC,C,8,0` (YYYYMMDD), `C_DTNASC,C,8,0` (YYYYMMDD), `R_DTOBITO,C,10,0`, `C_DTDIAG,C,10,0`, e subscores `NOME *`, `NOMEMAE *`, `DTNASC *`, `ENDERECO *`, `CODMUNRES local igual`. `TARGET` derivado de `PAR`.

### 2.4 Config (verificado em `gzcmd_v3_config.yaml`)
- Modos: **`vigilancia`** (alta sensibilidade; FP=10, FN=50, thr [0.85, 0.15], 2000 LLM/janela) e **`confirmacao`** (alta precisão; FP=100, FN=20, thr [0.95, 0.10], 1000 LLM/janela).
- Bandas: `low [0,5)`, `grey_low [5,6)`, `grey_mid [6,7)`, `grey_high [7,8)`, `near_high [8,9)`, `high [9, 999]`.
- Guardrails: `temporal_days=180`, `nota_always_match=10`, `nota_always_nonmatch=3`, `homonimia_min_nota=7`, `homonimia_year_gap=5`.

### 2.5 Lacunas/restrições conhecidas (impactam o plano)
- **R-01 (bloqueante p/ demo):** Não existe CSV de exemplo no repo (`data/COMPARADORSEMIDENT.csv` ausente). → Gerar dataset sintético é **obrigatório**.
- **R-02:** Não há `jupyter`, `matplotlib`, `papermill`, `nbval`, `nbconvert` no projeto. → Adicionar tooling de notebook em `requirements/notebook.txt`.
- **R-03:** `docs/plans/` não existia (criado por este plano).
- **R-04:** Possível mismatch entre features listadas na config (`nome_x_dtnasc`, `nome_perfeito`) e colunas produzidas pelo `loader` (`nome_score_total`). → Validar na Fase 0.2 (canary). *(Caso específico de R-11.)*
- **R-05:** Estágio LLM (`llm_review.py`) usa `gpt-5.2-pro` via API. → Para demo offline/reprodutível, **stub determinístico**; nunca depender de API ao vivo na apresentação.
- **R-06:** `conftest.py`/fixtures de teste ainda não inspecionados. → Pre-flight da Fase 0.2.
- **R-10 (metodológico, ALTA):** `run_v3` calibra **in-sample** (sem split) → *reliability diagram* sobre os mesmos dados é enganoso. → Notebook adota DEC-07 (rotas A/B); métricas de generalização **só** via `evaluate_v3_dataframe` (held-out). Ver 2.2.1.
- **R-11 (ALTA):** A config descreve comportamento (`anchor_platt`, `by_band`, listas de features) que **não está implementado** no código. → Ensinar **o que o código faz**; tratar a config como intenção/roadmap; canary 0.2 confirma o de-para real.
- **R-12 (metodológico, ALTA):** **Circularidade do sintético** — se `nota_final`/subscores e `TARGET` forem gerados pela mesma regra, a calibração fica trivialmente perfeita. → DEC-06: gerar a partir de uma **posterior verdadeira conhecida** `p*(x)`, com `TARGET ~ Bernoulli(p*)`, ruído e **sobreposição de classes**; validar calibração contra `p*`.
- **R-13:** XGBoost (`tree_method="hist"`, `n_jobs=-1`) pode ser **não-determinístico** entre execuções/threads. → Reconciliação tight **só** para Platt; XGBoost com `n_jobs=1`+seed e tolerância frouxa (ou apenas qualitativo). Ver R-09.

### 2.6 Layout de arquivos proposto (entregáveis)
```
docs/plans/notebook-gzcmd-passo-a-passo.md   # este plano
notebooks/
  gzcmd_passo_a_passo.ipynb                  # NOTEBOOK da apresentação (PT-BR)
  synthetic_data.py                          # gerador de dataset sintético (importável/testável)
  nb_helpers.py                              # helpers de plot + stub de LLM (determinístico)
data/synthetic/
  comparador_sintetico.csv                   # dataset gerado (sintético, pequeno, versionado)
tests/
  test_synthetic_data.py                     # testes do gerador (schema, determinismo, edge cases)
  test_notebook_pipeline.py                  # reconciliação manual × run_v3 + estágios
  test_notebook_execution.py                 # execução ponta-a-ponta do .ipynb (papermill/nbconvert)
requirements/
  notebook.txt                               # jupyter, matplotlib, papermill, nbformat, ipykernel...
```

---

## 3. Pre-flight Check (Checklist Padrão)

Executar **antes de cada fase** (`[tier:fast]` para inspeções; `[self:opus]` para o gating):

- [ ] **PF-1** `git status` limpo ou com apenas mudanças esperadas da fase anterior (já commitadas se aprovadas no QA).
- [ ] **PF-2** Ambiente virtual ativo e `gzcmd` importável (`python -c "import gzcmd_record_linkage"`).
- [ ] **PF-3** Suíte de testes **verde** no estado atual (`pytest -q`). Nenhuma regressão herdada.
- [ ] **PF-4** DoD da fase anterior cumprido e registrado.
- [ ] **PF-5** Dependências específicas da próxima fase disponíveis (ex.: `matplotlib` instalado antes da fase de gráficos).
- [ ] **PF-6** Espaço/escrita OK no diretório de trabalho; sem arquivos lockados.
- [ ] **PF-7** Itens de pre-flight específicos da fase (listados em cada fase) satisfeitos.

> Se PF-3 falhar por causa de algo herdado do repo (não causado por nós) → registrar como achado e tratar como **R-crítico** somente se bloquear; caso contrário, contornar e seguir (D1).

---

## 4. Senior QA Review (Rubrica Padrão)

Executar **ao final de cada fase** (`[tier:heavy]` ou `[self:opus]` se Opus). O revisor age como **QA sênior cético** e produz um relatório com **achados classificados** (🔴 bloqueante / 🟡 importante / 🟢 nice-to-have). **Todos os 🔴 e 🟡 devem ser corrigidos** antes do "Done"; 🟢 viram backlog se não triviais.

Eixos de revisão:
1. **Correção funcional:** o código faz o que diz? Bate com as APIs reais do `gzcmd` (Seção 2.2)?
2. **Cobertura de testes:** ≥ 90% de linhas/branches no código novo da fase; edge cases explicitados (Seção 4.1).
3. **Determinismo/reprodutibilidade:** seeds fixas; mesma entrada → mesma saída; sem dependência de relógio/rede.
4. **Fidelidade científica:** matemática correta (Platt, custo esperado), eixos/legendas corretos, sem afirmações enganosas.
5. **Qualidade de código:** passa `ruff check` (regras E,F,I,UP,B,SIM) e `ruff format --check`; sem `as any`/`# type: ignore`/`@ts-ignore`; sem caminhos hardcoded absolutos; paths via `pathlib`.
6. **Clareza didática (PT-BR):** narrativa explica o "porquê" de cada passo; termos definidos; público técnico/acadêmico atendido.
7. **Higiene de dados:** dataset 100% sintético; nada de PII real; CSV no formato `;`/`,`.
8. **Não-poluição da lib:** nenhuma dependência de apresentação vazou para `src/gzcmd_record_linkage/` (salvo DEC-01).
9. **Rigor estatístico (PhD):** sem vazamento treino/teste (R-10); calibração avaliada **held-out**; métricas quantitativas de calibração (**ECE** e **Brier score**), não só visual; validação contra a posterior verdadeira `p*(x)` quando aplicável (DEC-06); variância reportada via múltiplas seeds (intervalo/erro-padrão); afirmação de "calibrado" sempre acompanhada de número.
10. **Andaime didático (Bloom):** cada seção tem **objetivos de aprendizagem** explícitos ("ao final você será capaz de…"), intuição **antes** do formalismo, um **exemplo-fio-condutor** (par único seguido ponta-a-ponta) e um **recap** ao fim; carga cognitiva controlada; todo símbolo matemático é definido.

Saída do QA: relatório em `docs/plans/qa/REVIEW-<fase>.md` `[self:opus]` + lista de correções. Após corrigir, **re-rodar testes** e anexar evidência (output de `pytest`/`ruff`).

### 4.1 Catálogo de Edge Cases (referência para todas as fases)
- Notas exatamente nas **fronteiras de banda** (5.0, 6.0, 7.0, 8.0, 9.0) e `inclusive_max`.
- `p_cal` nos limites (≈0 e ≈1) e no **grey zone** dos thresholds de cada modo.
- **Nome da mãe ausente** (`mae_missing=1`; todos os `NOMEMAE *`=0).
- **Datas invertidas** (dia/mês/ano) e **aproximação de 1 dígito**.
- **Óbito antes do diagnóstico / janela temporal** (dispara guardrail temporal).
- **Homonímia:** `nota_final ≥ 7` porém `|diff_ano| ≥ 5` (dispara `FORCE_REVIEW`).
- **Âncoras:** `nota_final ≥ 9` (ALWAYS_MATCH) e `nota_final < 3`/`< 5` (ALWAYS_NONMATCH).
- **Classe única / desbalanceamento extremo** (calibração Platt deve degradar com elegância).
- **Encoding latin-1** e separador `;` no round-trip CSV.
- **MACD on/off** (colunas presentes/ausentes).
- **Modos divergentes:** mesmo par com decisão diferente em `vigilancia` vs `confirmacao`.
- **Vazamento por registro compartilhado:** mesmo `COMPREC`/`REFREC` em treino e teste (split `row` vaza; `comprec`/`refrec` não) — testar diferença de métricas entre `split_by`.
- **Prevalência ≠ taxa sintética:** `match_ratio` do gerador difere da prevalência real → discutir efeito da base rate sobre a calibração e o ponto de operação.
- **Posterior verdadeira nos extremos:** `p*(x)` ≈ 0 e ≈ 1 (clipping de `p_cal`), e na região de sobreposição (onde a calibração importa).

---

## 5. Critérios Globais

### 5.1 Critério de Aceitação Global
- **CA-G1:** O notebook `gzcmd_passo_a_passo.ipynb` executa **de ponta a ponta sem erros** em ambiente limpo (via `papermill`/`nbconvert --execute`), partindo apenas do repo + `requirements/notebook.txt`.
- **CA-G2:** Cada estágio do `gzcmd` (load → bands → calibração → guardrails → triage) é demonstrado **isoladamente**, com explicação PT-BR e **pelo menos uma visualização** pertinente.
- **CA-G3:** A saída do passo-a-passo manual na **rota A (in-sample)** **reconcilia** com `run_v3(...)` (mesmas colunas-chave: `p_cal`, `band`, `action`) dentro de tolerância documentada (`atol≤1e-9` para Platt determinístico; XGBoost tratado à parte — R-13).
- **CA-G4:** O dataset sintético é gerado por código determinístico (seed), válido no schema do `loader`, contém **todos** os cenários do catálogo de edge cases (Seção 4.1), ambas as classes **e expõe a posterior verdadeira `p*(x)`** (coluna de validação, não-entrada) conforme DEC-06.
- **CA-G5:** Métricas (precisão, recall, F-beta, cobertura da zona automática) são calculadas **na rota B (held-out)** para os dois modos, com **variância entre seeds** (≥5 seeds, erro-padrão/IC) e curvas **PR/ROC** + **superfície de custo vs. limiar**.
- **CA-G6:** Estágio LLM é **simulado por stub determinístico**, com explicação conceitual do protocolo `dual_agent_plus_arbiter`.
- **CA-G7:** Toda a suíte de testes nova passa; cobertura do código novo ≥ 90%; `ruff` limpo.
- **CA-G8 (calibração honesta):** O *reliability diagram* é feito **em teste held-out**, acompanhado de **ECE** e **Brier score** numéricos, e **sobreposto à posterior verdadeira `p*(x)`** demonstrando que o Platt a recupera (dentro de tolerância). O notebook **explicita** a diferença in-sample (`run_v3`) × held-out (`evaluate_v3_dataframe`) e a divergência config×código (R-11).
- **CA-G9 (didática):** Cada seção possui **objetivos de aprendizagem**, segue **intuição→formalismo**, usa um **exemplo-fio-condutor** rastreado em todos os estágios e termina com **recap**; a derivação do Platt define todos os símbolos.

### 5.2 Definition of Done Global
- [ ] Todos os DoD de fase cumpridos e commitados.
- [ ] CA-G1…CA-G9 satisfeitos com evidência (logs anexados em `docs/plans/qa/`).
- [ ] QA Global sênior executado e **todos os 🔴/🟡 corrigidos**.
- [ ] README curto de uso do notebook adicionado (como instalar tooling e executar).
- [ ] Notebook com saídas **limpas** versionado + uma versão executada (`*.executed.ipynb` ou HTML) como evidência (HTML pode ficar fora do git se grande — ver DEC-03).
- [ ] Plano atualizado com o **log de execução** (suposições, desvios, decisões).

#### 5.2.1 DoD FINAL (gate de encerramento — obrigatório)
Estes três critérios são a **checagem final** antes de declarar o trabalho concluído (verificados na Fase 4.3, tasks T4.3.5–T4.3.7):

- [ ] **DF-1 — Rodar tudo e funcionar como esperado.** Executar do zero, em ambiente limpo: **(a)** suíte completa `pytest -q` (toda verde, sem `skip`/`xfail` não justificados) **e (b)** o notebook **ponta-a-ponta** via `papermill`/`nbconvert --execute` sem erros. As saídas devem ser **coerentes com o que o texto descreve** (números, gráficos e decisões batem com a narrativa — não só "executou"). Evidência (logs + notebook executado) anexada em `docs/plans/qa/`.
- [ ] **DF-2 — Confirmar que o proposto foi implementado.** Varredura **item a item** do plano (todas as Tasks `T*`, Testes `TST*`, `CA-G1…CA-G9`, `DEC-01…DEC-10` e mitigações `R-01…R-13`): cada item está **implementado** ou tem **justificativa explícita** registrada no Log de Execução (Seção 10). **Nenhum item silenciosamente omitido.** Produzir um checklist de conformidade em `docs/plans/qa/conformidade-final.md`.
- [ ] **DF-3 — Explicação compreensiva antes de cada etapa.** Auditar o notebook célula a célula: **antes de cada passo/célula de código existe uma célula markdown** que explica, de forma compreensiva e em PT-BR, **o que vai acontecer no passo seguinte** (o "o quê" e o "porquê"), respeitando o andaime didático (Seção 6.5: objetivo de aprendizagem → intuição → ação → recap). **Não pode haver célula de código "órfã"** sem contexto prévio. Resultado da auditoria registrado em `docs/plans/qa/auditoria-didatica.md`.

---

## 6. Decisões de Projeto (abertas/registradas)

- **DEC-01 — Onde fica o gerador?** *Recomendado:* `notebooks/synthetic_data.py` (fora da lib). Para os testes importarem, adicionar `notebooks` ao `pythonpath` do pytest (`pyproject.toml [tool.pytest.ini_options] pythonpath = ["src", "notebooks"]`). Alternativa rejeitada: colocar em `src/gzcmd_record_linkage/examples/` (polui a lib publicada). *Default a seguir:* opção recomendada.
- **DEC-02 — Calibração principal:** **Platt** (didática, curva visualizável). XGBoost vira **apêndice opcional** no notebook.
- **DEC-03 — Evidência de execução:** versionar `.ipynb` com saídas limpas; gerar HTML/`.executed.ipynb` como artefato de QA (não versionar se > 1 MB; manter em `docs/plans/qa/`).
- **DEC-04 — Tamanho do dataset:** alvo **300–1000 pares** (suficiente para Platt e métricas estáveis, leve para git). Seed padrão `42`.
- **DEC-05 — Versionar o CSV sintético?** Sim (pequeno e reprodutível); ainda assim, o notebook **regenera** via `synthetic_data.py` para garantir reprodutibilidade.
- **DEC-06 — Modelo gerador com posterior verdadeira (anti-circularidade):** o gerador define explicitamente `p*(x) = σ(β₀ + βᵀ·features)` (posterior verdadeira de match), amostra `TARGET ~ Bernoulli(p*)`, e só então deriva `nota_final`/subscores com **ruído e sobreposição** de classes. `p*` é gravada como coluna de **validação** (`p_true`), **nunca** consumida pelo pipeline. *Justificativa:* permite validar a calibração contra a verdade-base (ground-truth posterior), tornando o *reliability diagram* uma prova, não uma tautologia. *(Resolve R-12.)*
- **DEC-07 — Duas rotas explícitas (reprodução × metodologia):** **Rota A "fiel"** = in-sample (reconcilia com `run_v3`); **Rota B "correta"** = split treino/teste via `evaluate_v3_dataframe` para todas as métricas de generalização e o *reliability diagram* honesto. As duas convivem no notebook, claramente rotuladas. *(Resolve R-10.)*
- **DEC-08 — Métricas quantitativas de calibração:** além do gráfico, computar **ECE** (Expected Calibration Error, binning explícito) e **Brier score** no teste held-out. Implementar em `nb_helpers.py` (`expected_calibration_error`, `brier_score`) com testes próprios.
- **DEC-09 — Andaime didático obrigatório:** objetivos de aprendizagem por seção, intuição→formalismo, **exemplo-fio-condutor** (um par "herói" sintético seguido em todos os estágios) e recap por seção. *(Atende público técnico/acadêmico e o eixo 10 do QA.)*
- **DEC-10 — Interatividade (opcional, alto valor):** se o tempo permitir, `ipywidgets` com sliders (limiar de decisão → precisão/recall/custo ao vivo; inclinação do Platt). **Não** pode ser pré-requisito da execução headless (papermill): proteger com `try/except`/flag para não quebrar CA-G1.

> Nenhuma DEC acima é bloqueante (há default seguro para todas) → seguir conforme D1.

---

## 6.5 Diretrizes Didáticas (PhD em ML + didática)

Princípios que o notebook **deve** seguir (auditados no eixo 10 do QA):

1. **Objetivos de aprendizagem por seção** — abrir cada estágio com "Ao final desta seção, você será capaz de…" (verbos de Bloom: *explicar, calcular, interpretar, comparar*).
2. **Intuição antes do formalismo** — primeiro um exemplo concreto e a pergunta de negócio; depois a matemática. Ex.: mostrar dois registros quase-iguais e perguntar "é a mesma pessoa?" antes de definir score/banda.
3. **Exemplo-fio-condutor ("herói")** — escolher 1 par sintético memorável e segui-lo em TODOS os estágios (nota → banda → `p_cal` → guardrail → ação), com um "card" recorrente mostrando seu estado. Reduz carga cognitiva e dá continuidade narrativa.
4. **Derivação completa do Platt** — definir todo símbolo: `p = σ(a·s + b)`, objetivo de **log-verossimilhança negativa** (NLL) regularizada por L2, por que é regressão logística 1-D sobre o score `s`, e o que `a` (inclinação) e `b` (viés) significam geometricamente.
5. **Honestidade científica visível** — toda afirmação quantitativa traz número e contexto; nunca apresentar resultado in-sample como evidência de generalização; sempre nomear limitações (sintético, prevalência, LLM simulado).
6. **Visual com propósito** — cada figura responde a UMA pergunta, com título/eixos/legenda em PT-BR; preferir poucos gráficos excelentes a muitos genéricos.
7. **Recap + "o que vem a seguir"** ao fim de cada seção, costurando o pipeline.
8. **Glossário** de termos (record linkage, blocking, zona cinzenta, calibração, ECE, Brier, guardrail, triagem) no início ou apêndice.

---

## 7. Waves, Fases, Tasks e Subtasks

> Cada fase segue o ciclo: **Pre-flight → Tasks/Subtasks → Testes da fase → Critério de Aceitação → DoD → Senior QA Review → Correções → Commit.**

---

### 🌊 WAVE 0 — Fundação, Ambiente e Contrato de API

**Objetivo da wave:** garantir ambiente reprodutível, validar (por execução real) as APIs e o schema, e montar o scaffolding. Sem isso, todo o resto é especulação.

#### Fase 0.1 — Preparação do Ambiente
**Pre-flight específico:** verificar versão do Python (≥3.10), `pip` disponível, estado do git. `[tier:fast]`

**Tasks:**
- T0.1.1 `[tier:medium]` Criar/ativar venv (`python -m venv .venv`); documentar comandos pwsh.
- T0.1.2 `[tier:medium]` `python -m pip install -e .[dev]` (instala `gzcmd` + ruff/pytest/pytest-cov/pyright).
- T0.1.3 `[tier:fast]` Rodar `pytest -q` e **registrar baseline** (nº de testes, verde/vermelho).
- T0.1.4 `[tier:fast]` `gzcmd --help` e `python -m gzcmd_record_linkage --help` funcionam.
- T0.1.5 `[tier:medium]` Criar `requirements/notebook.txt` (jupyter, ipykernel, matplotlib, papermill, nbformat, nbconvert, pandas já vem da lib) e instalar.

**Testes da fase:**
- TST0.1.a `[tier:medium]` Smoke test `tests/test_env_smoke.py`: importa todos os módulos públicos do pacote sem erro.
- TST0.1.b `[tier:fast]` Confirmar baseline `pytest` permanece verde após instalar tooling de notebook.

**Critério de Aceitação:** `pip install -e .[dev]` e `requirements/notebook.txt` instalam sem erro; `pytest` verde; CLIs respondem; `import matplotlib, papermill` OK.
**DoD:** comandos de setup documentados em `notebooks/README.md` (rascunho); baseline registrado em `docs/plans/qa/baseline.md`.
**Senior QA Review:** ambiente reprodutível? versões fixadas o suficiente? tooling de notebook isolado da lib (não entrou em `pyproject` da lib)? `[self:opus]`
**Commit:** `chore(env): setup venv + requirements/notebook.txt + smoke test`.

#### Fase 0.2 — Validação de Contrato de API (Canary/Probing)
**Pre-flight específico:** ler `tests/conftest.py` e fixtures existentes (R-06); ambiente da 0.1 OK. `[tier:fast]`

**Tasks:**
- T0.2.1 `[tier:fast]` Inspecionar `conftest.py`/fixtures; mapear helpers reutilizáveis para dados de teste.
- T0.2.2 `[tier:medium]` Script-canário (descartável, em `tests/test_api_contract.py`) que, a partir do DataFrame-fixture, executa **cada estágio uma vez** e captura as colunas produzidas: `BandAssigner.assign`, `fit_platt_from_df`+`compute_p_cal`, `apply_guardrails`, `build_engine_from_config`+`triage`.
- T0.2.3 `[tier:medium]` Round-trip CSV: escrever fixture como CSV (`;`,`,`), reler com `load_comparador_csv`, confirmar colunas cruas→engenheiradas (inclui MACD on/off).
- T0.2.4 `[tier:heavy]` Resolver **R-04** (mismatch config×loader): comparar `feature_columns` da config com colunas reais do `loader`; documentar o de-para. Se houver incompatibilidade que quebre `run_v3`, decidir mitigação (ajustar gerador para emitir o que o pipeline realmente consome).

**Testes da fase:**
- TST0.2.a `[tier:medium]` `test_api_contract.py`: asserts de que cada estágio produz exatamente as colunas esperadas (`band`, `p_cal`, `guardrail`, `action`, `evr`, …) com tipos corretos.
- TST0.2.b `[tier:medium]` Edge: `compute_p_cal(method="stub")` vs `"platt"` retornam Series em [0,1]; `triage` muta df e popula `action`.

**Critério de Aceitação:** I/O real de **todos** os estágios documentado e batendo com a Seção 2.2; R-04 resolvido (de-para escrito).
**DoD:** `docs/plans/qa/contrato-api.md` com o contrato verificado por execução; testes de contrato verdes.
**Senior QA Review:** algum estágio diverge do esperado? o de-para de features cobre `run_v3` ponta-a-ponta? riscos para o gerador? `[tier:heavy]`/`[self:opus]`
**Commit:** `test(contract): valida I/O real dos estágios do gzcmd + de-para de features`.

#### Fase 0.3 — Scaffolding do Projeto
**Pre-flight específico:** contrato 0.2 aprovado. `[tier:fast]`

**Tasks:**
- T0.3.1 `[tier:medium]` Criar diretórios `notebooks/`, `data/synthetic/`, `docs/plans/qa/`.
- T0.3.2 `[tier:medium]` Aplicar **DEC-01**: adicionar `notebooks` ao `pythonpath` do pytest em `pyproject.toml`.
- T0.3.3 `[tier:medium]` `.gitignore`: ignorar `.venv/`, `*.executed.ipynb` grandes, checkpoints do Jupyter; **não** ignorar `data/synthetic/*.csv`.
- T0.3.4 `[tier:medium]` Esqueleto de `notebooks/nb_helpers.py` (assinaturas vazias documentadas) e `notebooks/synthetic_data.py` (idem).

**Testes da fase:**
- TST0.3.a `[tier:fast]` `pytest` ainda coleta/roda com novo `pythonpath`; import de `synthetic_data` e `nb_helpers` funciona em teste trivial.

**Critério de Aceitação:** estrutura criada; pytest descobre `notebooks`; nada quebrado.
**DoD:** árvore de diretórios da Seção 2.6 existente; `pyproject` ajustado e testado.
**Senior QA Review:** `pythonpath` não causa import shadowing? `.gitignore` correto? `[self:opus]`
**Commit:** `chore(scaffold): estrutura de notebooks/dados/tests + pythonpath`.

**🔍 QA Review da WAVE 0 (gate de wave):** ambiente + contrato + scaffold sólidos; baseline verde; riscos R-01..R-06 endereçados ou com mitigação planejada. `[tier:heavy]`/`[self:opus]`

---

### 🌊 WAVE 1 — Gerador de Dataset Sintético

**Objetivo da wave:** produzir um dataset sintético rotulado, determinístico, **válido no schema do loader**, com correlação realista entre subscores↔`nota_final`↔`TARGET` (para a calibração ser significativa) e cobrindo **todos** os edge cases.

#### Fase 1.1 — Núcleo do Gerador
**Pre-flight específico:** contrato de API (0.2) e de-para de features disponíveis. `[tier:fast]`

**Tasks:**
- T1.1.1 `[tier:heavy]` **Design** do modelo gerador com **posterior verdadeira** (DEC-06): definir `p*(x) = σ(β₀ + βᵀ·z)` sobre fatores latentes `z` (semelhança de nome, data, endereço…); amostrar `TARGET ~ Bernoulli(p*)`; só então derivar subscores e `nota_final` **condicionados** a `z` com **ruído e sobreposição de classes** (matches e não-matches com regiões que se misturam, senão a calibração fica trivial). `[self:opus]` define `β`, ruído e grau de sobreposição. **Anti-circularidade:** `nota_final` não pode ser função determinística de `TARGET`.
- T1.1.2 `[tier:medium]` Implementar geração das colunas obrigatórias com os **nomes exatos** (prefixos OpenRecLink) da Seção 2.3; gravar coluna de validação `p_true` (= `p*`), marcada como **não-entrada** do pipeline.
- T1.1.3 `[tier:medium]` Subtask: gerar `R_DTNASC`/`C_DTNASC` (YYYYMMDD) coerentes com `z` (datas iguais/aproximadas em alta semelhança; divergentes em baixa); `R_DTOBITO`/`C_DTDIAG`.
- T1.1.4 `[tier:medium]` Subtask: derivar `nota_final` a partir dos subscores/`z` com ruído controlado, garantindo ambas as classes, grey zone populado e **sobreposição** (não separável perfeitamente).
- T1.1.5 `[tier:medium]` Função `to_comparador_csv(df, path)` que escreve no formato `;`/`,` (latin-1 e utf-8 testados); a coluna `p_true` **não** é escrita no CSV de entrada do pipeline (fica só no DataFrame de validação / arquivo separado).
- T1.1.6 `[tier:medium]` API de split reprodutível para a Rota B: helper que devolve índices treino/teste **group-aware** por `COMPREC`/`REFREC` (espelhando `split_by` do `eval.py`), para uso didático no notebook.

**Testes da fase (`tests/test_synthetic_data.py`):**
- TST1.1.a `[tier:medium]` **Schema:** todas as colunas obrigatórias presentes com nomes exatos; tipos corretos; CSV de entrada **não** contém `p_true`.
- TST1.1.b `[tier:medium]` **Determinismo:** mesma seed → DataFrame idêntico; seeds diferentes → diferentes.
- TST1.1.c `[tier:medium]` **Distribuição:** `match_ratio` respeitado dentro de tolerância; ambas as classes presentes; grey zone não-vazio.
- TST1.1.d `[tier:medium]` **Round-trip:** `to_comparador_csv` → `load_comparador_csv` sem erro, gera colunas engenheiradas; MACD on/off.
- TST1.1.e `[tier:medium]` **Edge:** fronteiras de banda (5/6/7/8/9) presentes; `nota_final` em [0, ~10+].
- TST1.1.f `[tier:medium]` **Anti-circularidade / sobreposição:** `nota_final` **não** é separável perfeitamente por `TARGET` (AUC < 1.0 e classes se sobrepõem numa faixa); `p_true ∈ (0,1)` com massa na região intermediária.
- TST1.1.g `[tier:medium]` **Posterior recuperável:** ordenando por `p_true`, a fração empírica de `TARGET=1` cresce monotonicamente por bins (a verdade-base é coerente) — garante que o Platt tem o que recuperar.
- TST1.1.h `[tier:medium]` **Split group-aware:** nenhum `COMPREC`/`REFREC` aparece simultaneamente em treino e teste quando `split_by ∈ {comprec, refrec}`.

**Critério de Aceitação:** CSV gerado é ingerido por `load_comparador_csv` sem erro; ambas as classes e grey zone presentes; determinístico; `p_true` coerente e classes **sobrepostas** (não triviais).
**DoD:** `synthetic_data.py` com docstrings PT-BR; testes verdes; cobertura ≥90% do módulo.
**Senior QA Review:** o modelo gerador é estatisticamente honesto (não circular, com sobreposição realista)? `p*` é recuperável pelo Platt? Platt terá sinal sem ser trivial? nomes de coluna 100% corretos? `[tier:heavy]`
**Commit:** `feat(notebook): gerador de dataset sintético (núcleo) + testes`.

#### Fase 1.2 — Cenários Narrativos e Edge Cases Rotulados
**Pre-flight específico:** núcleo 1.1 aprovado. `[tier:fast]`

**Tasks:**
- T1.2.1 `[tier:medium]` Injetar cenários nomeados (sintéticos, sem PII) para a narrativa: (a) match óbvio (âncora high), (b) não-match óbvio (âncora low), (c) gêmeos/homônimos (nota alta + gap de ano → `FORCE_REVIEW`), (d) óbito-antes-diagnóstico (guardrail temporal), (e) nome da mãe ausente, (f) datas invertidas, (g) caso clássico de zona cinzenta (vai a `LLM_REVIEW`).
- T1.2.2 `[tier:medium]` Parametrizar `scenarios=[...]` para forçar a inclusão desses pares com rótulo conhecido e nota controlada.

**Testes da fase:**
- TST1.2.a `[tier:medium]` Cada cenário nomeado cai na **banda** esperada.
- TST1.2.b `[tier:medium]` Cenários de guardrail disparam o `guardrail`/`reason` esperado ao passar por `apply_guardrails`.
- TST1.2.c `[tier:medium]` Cenário de zona cinzenta roteia para `LLM_REVIEW` no `triage` (modo `confirmacao`).

**Critério de Aceitação:** todos os 7 cenários presentes e validados ponta-a-ponta nos estágios relevantes.
**DoD:** cenários documentados (tabela PT-BR no docstring/`nb_helpers`); testes verdes; cobertura mantida ≥90%.
**Senior QA Review:** cenários são didáticos e inequívocos? algum cenário é instável (depende de ruído)? fixar para reprodutibilidade. `[tier:heavy]`
**Commit:** `feat(notebook): cenários narrativos + edge cases no gerador`.

**🔍 QA Review da WAVE 1 (gate):** dataset robusto, determinístico, com cenários — pronto para alimentar o notebook. `[tier:heavy]`/`[self:opus]`

---

### 🌊 WAVE 2 — Notebook: Pipeline Passo a Passo

**Objetivo da wave:** construir o notebook que percorre **cada estágio** com explicação técnica PT-BR, código real do `gzcmd` e visualização. **Cada fase adiciona seções e mantém o notebook executável de cima a baixo.**

> **Teste transversal da wave:** após cada fase, o notebook deve **executar ponta-a-ponta** via `papermill`/`nbconvert --execute` (TST-NBEXEC). Esse teste cresce com o notebook.

#### Fase 2.1 — Estrutura, Narrativa e Setup
**Pre-flight específico:** dataset sintético gerável (Wave 1); `papermill` instalado. `[tier:fast]`

**Tasks:**
- T2.1.1 `[tier:medium]` Criar `gzcmd_passo_a_passo.ipynb` com seções markdown PT-BR: contexto de record linkage, o problema da zona cinzenta, visão geral do GZ-CMD++ v3 (diagrama do pipeline), objetivos da apresentação.
- T2.1.2 `[tier:medium]` Célula de setup: imports, geração do dataset (`synthetic_data.generate_comparador(seed=42)`), salvar em `data/synthetic/`.
- T2.1.3 `[tier:medium]` Célula de "olhar os dados": `head`, dicionário de colunas (tabela PT-BR explicando cada subscore).

**Testes da fase:**
- TST2.1.NBEXEC `[tier:medium]` `tests/test_notebook_execution.py`: executa o notebook via papermill sem erro (estado atual).
- TST2.1.a `[tier:fast]` Notebook contém as seções esperadas (parse via `nbformat`).

**Critério de Aceitação:** notebook executa limpo; narrativa de abertura clara para público técnico.
**DoD:** seções 1–3 prontas; teste de execução verde.
**Senior QA Review:** abertura motiva o problema? termos definidos? sem "AI slop"? `[tier:heavy]`
**Commit:** `feat(notebook): estrutura + setup + visão geral do pipeline`.

#### Fase 2.2 — Carga e Feature Engineering
**Pre-flight:** 2.1 OK; CSV gerado disponível. `[tier:fast]`
**Tasks:**
- T2.2.1 `[tier:medium]` Célula: `load_comparador_csv(...)`; mostrar colunas cruas → agregadas (`nome_score_total`, `dtnasc_score_total`, …) → flags (`mae_missing`) → MACD.
- T2.2.2 `[tier:medium]` Explicação PT-BR de cada grupo de features + visualização (heatmap de correlação ou distribuição de `nota_final` por `TARGET`).
**Testes da fase:**
- TST2.2.a `[tier:medium]` Assert: colunas engenheiradas existem; ranges válidos (scores em [0,1]).
- TST2.2.NBEXEC `[tier:medium]` Execução ponta-a-ponta ainda verde.
**Aceitação:** estágio de carga reproduzido e explicado; visual presente.
**DoD:** seção 4 pronta; testes verdes. **QA:** features explicadas corretamente? MACD esclarecido? `[tier:heavy]` **Commit:** `feat(notebook): carga + feature engineering`.

#### Fase 2.3 — Atribuição de Bandas
**Pre-flight:** 2.2 OK. `[tier:fast]`
**Tasks:**
- T2.3.1 `[tier:medium]` `BandAssigner.from_config(cfg).assign(nota)`; histograma de `nota_final` colorido por banda; tabela de fronteiras.
- T2.3.2 `[tier:medium]` Explicar por que existem `grey_*` (zona de incerteza).
**Testes:**
- TST2.3.a `[tier:medium]` Bandas atribuídas batem com as fronteiras da config (incl. `inclusive_max`); edge nas fronteiras.
- TST2.3.NBEXEC `[tier:medium]` Execução verde.
**Aceitação/DoD:** seção 5 pronta; testes verdes. **QA:** fronteiras corretas? visual legível? `[tier:heavy]` **Commit:** `feat(notebook): atribuição de bandas + visualização`.

#### Fase 2.4 — Calibração (Platt) — **rotas A e B, sem vazamento**
**Objetivo de aprendizagem:** explicar o que é calibração de probabilidade, derivar o Platt, e **distinguir avaliação in-sample de held-out**; medir calibração com ECE/Brier e validar contra a posterior verdadeira.
**Pre-flight:** 2.3 OK; ambas as classes; `p_true` disponível; helper de split group-aware (T1.1.6) pronto. `[tier:fast]`
**Tasks:**
- T2.4.1 `[tier:medium]` **Derivação do Platt** (markdown, DEC-09 item 4): `p = σ(a·s + b)`, NLL regularizada por L2, interpretação de `a`/`b`. Intuição antes da fórmula (exemplo-fio-condutor).
- T2.4.2 `[tier:medium]` **Rota A (fiel à ferramenta):** `fit_platt_from_df(df)` → `PlattModel(intercept, slope)`; `compute_p_cal(df, method="platt", model=...)` **in-sample** — deixar explícito que reproduz o `run_v3` e **por que NÃO serve para avaliar generalização** (vazamento, R-10).
- T2.4.3 `[tier:medium]` **Rota B (metodologia correta):** split **group-aware** treino/teste (T1.1.6); `fit_platt` no treino; prever no teste; **reliability diagram no TESTE**; computar **ECE** e **Brier** (helpers DEC-08).
- T2.4.4 `[tier:medium]` **Validação contra a verdade-base:** sobrepor `p_cal` (teste) × `p_true` no diagrama; mostrar que o Platt recupera `p*` dentro de tolerância. Discutir **prevalência/base rate** (match_ratio ≠ real) e seu efeito.
- T2.4.5 `[tier:medium]` Sinalizar a **divergência config×código** (R-11): config promete `anchor_platt`/`by_band`; código faz Platt global. Apresentar como "intenção vs implementação".
- T2.4.6 `[tier:medium]` Apêndice opcional (DEC-02): XGBoost via `GZCMDClassifier`, `predict_proba(df)[:,1]` — comparar curvas de calibração (ML vs Platt). Fixar `n_jobs=1`+seed (R-13).
**Testes (`tests/test_notebook_pipeline.py` + helpers em `test_synthetic_data.py`/novo `test_nb_helpers.py`):**
- TST2.4.a `[tier:medium]` `p_cal ∈ [clip_min, clip_max]`; monotonicidade aproximada (s↑ ⇒ p_cal↑).
- TST2.4.b `[tier:medium]` Edge: classe única / poucos positivos não quebram (degradação elegante).
- TST2.4.c `[tier:medium]` **ECE/Brier:** helpers corretos contra casos fechados (ex.: previsões perfeitas → ECE≈0, Brier baixo; previsões constantes → valores conhecidos); binning robusto a bins vazios.
- TST2.4.d `[tier:medium]` **Sem vazamento:** o split da Rota B não compartilha `COMPREC`/`REFREC` entre treino/teste; calibração ajustada **só** no treino.
- TST2.4.e `[tier:medium]` **Recuperação de `p*`:** ECE de `p_cal` vs `p_true` no teste abaixo de um limiar tolerante (com seed fixa).
- TST2.4.NBEXEC `[tier:medium]` Execução verde.
**Aceitação/DoD:** seção 6 pronta; CA-G8 satisfeito; matemática correta; testes verdes. **QA:** Platt derivado corretamente? *reliability diagram* é held-out? ECE/Brier reportados? `p*` recuperada? vazamento evitado? divergência config×código declarada? `[tier:heavy]` **Commit:** `feat(notebook): calibração Platt (in-sample×held-out) + ECE/Brier + validação vs p_true`.

#### Fase 2.5 — Guardrails
**Pre-flight:** 2.4 OK; cenários de guardrail no dataset (Fase 1.2). `[tier:fast]`
**Tasks:**
- T2.5.1 `[tier:medium]` `apply_guardrails(df, ...)`; mostrar tabela de casos com `guardrail` ∈ {ALWAYS_MATCH, ALWAYS_NONMATCH, FORCE_REVIEW} e `reason`.
- T2.5.2 `[tier:medium]` Explicar cada regra (temporal de óbito, homonímia, âncoras) com o caso sintético correspondente.
**Testes:**
- TST2.5.a `[tier:medium]` Cada tipo de guardrail dispara ao menos uma vez; `reason` correto.
- TST2.5.NBEXEC `[tier:medium]` Execução verde.
**Aceitação/DoD:** seção 7 pronta; testes verdes. **QA:** regras corretas e bem ilustradas? `[tier:heavy]` **Commit:** `feat(notebook): guardrails + casos ilustrativos`.

#### Fase 2.6 — Política de Decisão (Triage) nos Dois Modos
**Pre-flight:** 2.5 OK; config carregada. `[tier:fast]`
**Tasks:**
- T2.6.1 `[tier:medium]` `build_engine_from_config(cfg, mode="vigilancia")` e `"confirmacao"`; `engine.triage(df.copy())` para cada modo (cuidado: `triage` muta — usar cópias).
- T2.6.2 `[tier:medium]` **Explicar o custo esperado** (FP/FN/LLM, thresholds, `evr`/`base_loss`/`loss_llm`); tabela comparando distribuição de `action` entre modos.
- T2.6.3 `[tier:medium]` Visual: barras de MATCH/NONMATCH/LLM_REVIEW por modo; destacar pares que mudam de decisão.
**Testes:**
- TST2.6.a `[tier:medium]` `action ∈ {MATCH,NONMATCH,LLM_REVIEW}`; `confirmacao` é mais conservador (mais NONMATCH/LLM_REVIEW que `vigilancia`) — assert direcional.
- TST2.6.b `[tier:medium]` Edge: par no grey zone muda de decisão entre modos; budget de LLM respeitado (`llm_used ≤ llm_max`).
- TST2.6.NBEXEC `[tier:medium]` Execução verde.
**Aceitação/DoD:** seção 8 pronta; testes verdes. **QA:** matemática do custo correta? comparação de modos honesta? mutação tratada? `[tier:heavy]` **Commit:** `feat(notebook): triage nos modos vigilancia/confirmacao + custo esperado`.

**🔍 QA Review da WAVE 2 (gate):** todos os estágios reproduzidos isoladamente, explicados e visualizados; notebook executa ponta-a-ponta. `[tier:heavy]`/`[self:opus]`

---

### 🌊 WAVE 3 — Integração End-to-End, Métricas e Revisão LLM (Stub)

#### Fase 3.1 — Reconciliação Manual × `run_v3` (joia da coroa) — **rota A (in-sample)**
**Pre-flight:** Wave 2 completa; CSV salvo. `[tier:fast]`
**Tasks:**
- T3.1.1 `[tier:heavy]` **Design** da reconciliação: a **rota A** (in-sample, `p_cal="fit_platt"`) deve reproduzir `run_v3(input_csv=..., config_path=..., mode="vigilancia")`. `[self:opus]` define tolerâncias e colunas. **Importante:** reconciliar contra `run_v3` exige que o manual também ajuste o Platt **em todas as linhas** (não usar o split da Rota B aqui). Como `fit_platt` é determinístico, espera-se igualdade quase-exata para Platt.
- T3.1.2 `[tier:medium]` Célula que roda `run_v3`, captura `(out_df, RunSummary)`, e **compara** `p_cal`, `band`, `action` com o resultado manual da rota A; exibir `RunSummary`.
- T3.1.3 `[tier:medium]` Explicar por que reconcilia (mesmo ajuste global determinístico) e **por que a Rota B difere de propósito** (split → números diferentes, e isso é correto). XGBoost/RF: reconciliação só qualitativa (R-13).
**Testes (`tests/test_notebook_pipeline.py`):**
- TST3.1.a `[tier:medium]` Reconciliação Platt: para o mesmo CSV/modo, `band` idêntico, `p_cal` com `atol≤1e-9`, e `action` **idêntico** (sem tolerância de linhas — deve bater 100% por ser determinístico; se não bater, é bug a investigar).
- TST3.1.b `[tier:medium]` `RunSummary.rows == len(df)`; soma de `actions` fecha com o total; `guardrails`/`review_requested` consistentes.
- TST3.1.c `[tier:medium]` XGBoost: reconciliação **tolerante** (ou marcada `xfail`/qualitativa) documentando não-determinismo (R-13).
**Aceitação:** CA-G3 satisfeito. **DoD:** seção 9 pronta; teste de reconciliação verde. **QA:** a reconciliação é genuína (rota A in-sample, não a rota B)? a distinção A/B está explícita no notebook? tolerâncias justificadas? `[tier:heavy]` **Commit:** `test(notebook): reconciliação manual (rota A) × run_v3`.

#### Fase 3.2 — Métricas e Avaliação — **rota B (held-out, com variância)**
**Objetivo de aprendizagem:** avaliar generalização corretamente; ler PR/ROC; ligar a política de custo a um ponto de operação ótimo.
**Pre-flight:** 3.1 OK; split group-aware disponível. `[tier:fast]`
**Tasks:**
- T3.2.1 `[tier:medium]` Métricas held-out via **`evaluate_v3_dataframe(...)`** (split treino/teste, calibração no treino) — **promovido a obrigatório** (não mais opcional). Usar `metrics.metrics_dict`/`confusion_counts`; reportar precisão, recall, F-beta, **cobertura automática** (fração não-`LLM_REVIEW`) por modo.
- T3.2.2 `[tier:medium]` **Variância entre seeds (≥5):** rodar múltiplas seeds; `summarize_runs` (média±desvio) + tabela; **barras de erro** nas figuras. Discutir por que ponto único engana.
- T3.2.3 `[tier:medium]` **Group-aware vs row split:** demonstrar empiricamente o **vazamento** comparando métricas `split_by=row` × `comprec`/`refrec` (mostra inflação otimista do `row`).
- T3.2.4 `[tier:medium]` **Curvas PR/ROC** sobre `p_cal` (teste) e **superfície de custo vs. limiar** por modo; mostrar que os thresholds `min_auto_match`/`max_auto_nonmatch` correspondem a um **ponto de operação por custo esperado** (ligar a `evr`/FP/FN da Fase 2.6).
- T3.2.5 `[tier:medium]` Interpretação PT-BR: trade-off precisão×recall×cobertura entre modos; efeito da prevalência.
**Testes:**
- TST3.2.a `[tier:medium]` Métricas consistentes (0≤p,r≤1); confusion fecha (`tp+fp+fn+tn==n`).
- TST3.2.b `[tier:medium]` `vigilancia` recall ≥ `confirmacao` recall **em média sobre seeds** (direcional, com tolerância — não em seed única, para evitar flakiness).
- TST3.2.c `[tier:medium]` `split_by=row` produz métricas **≥** group-aware (evidência de inflação por vazamento), com seed fixa.
- TST3.2.d `[tier:medium]` Resultado de `evaluate_v3_dataframe` tem as colunas esperadas (modes/seed/precision/recall/...).
- TST3.2.NBEXEC `[tier:medium]` Execução verde.
**Aceitação:** CA-G5. **DoD:** seção 10 pronta; testes verdes. **QA:** avaliação é held-out? variância reportada? vazamento por split demonstrado? PR/ROC e custo ligados à política? `[tier:heavy]` **Commit:** `feat(notebook): avaliação held-out multi-seed + PR/ROC + custo vs limiar`.

#### Fase 3.3 — Revisão LLM (Stub Determinístico)
**Pre-flight:** 3.2 OK. `[tier:fast]`
**Tasks:**
- T3.3.1 `[tier:medium]` Em `nb_helpers.py`, criar `llm_review_stub(df_review, seed=42, error_rates_by_band=...) -> pd.Series` determinístico, usando as taxas de erro por banda da config (R-05). **Sem chamadas de rede.**
- T3.3.2 `[tier:medium]` Célula: aplicar stub aos casos `LLM_REVIEW`; mostrar decisão final pós-revisão; explicar conceitualmente o protocolo `dual_agent_plus_arbiter` (markdown).
- T3.3.3 `[tier:medium]` Recalcular métricas finais incluindo a revisão simulada.
**Testes:**
- TST3.3.a `[tier:medium]` Stub determinístico (mesma seed → mesma saída); taxas de erro respeitadas dentro de tolerância.
- TST3.3.NBEXEC `[tier:medium]` Execução verde.
**Aceitação:** CA-G6. **DoD:** seção 11 pronta; testes verdes. **QA:** stub é honesto (não finge ser LLM real)? deixa claro que é simulação? `[tier:heavy]` **Commit:** `feat(notebook): stub determinístico de revisão LLM`.

**🔍 QA Review da WAVE 3 (gate):** ponta-a-ponta reconciliado, métricas sólidas, LLM simulado de forma transparente. `[tier:heavy]`/`[self:opus]`

---

### 🌊 WAVE 4 — Polimento, Reprodutibilidade e Entrega

#### Fase 4.1 — Narrativa Final, Figuras e Formatação
**Pre-flight:** Wave 3 completa. `[tier:fast]`
**Tasks:**
- T4.1.1 `[tier:medium]` Revisar toda a narrativa PT-BR (coesão, transições entre estágios, conclusão); padronizar estilo de figuras (títulos, eixos, legendas em PT-BR).
- T4.1.2 `[tier:medium]` Adicionar seção de **conclusões** e **limitações** (dados sintéticos, LLM simulado).
**Testes:** TST4.1.NBEXEC `[tier:medium]` execução verde; TST4.1.a `[tier:fast]` todas as figuras têm título/legenda (checagem heurística via nbformat se viável).
**Aceitação/DoD:** notebook coeso e apresentável. **QA:** fluxo didático impecável? `[tier:heavy]` **Commit:** `docs(notebook): narrativa final + figuras padronizadas`.

#### Fase 4.2 — Reprodutibilidade e Execução Limpa
**Pre-flight:** 4.1 OK. `[tier:fast]`
**Tasks:**
- T4.2.1 `[tier:medium]` Garantir seeds fixas em todo o notebook; limpar saídas; reexecutar do zero via `papermill` em venv limpa.
- T4.2.2 `[tier:medium]` `notebooks/README.md` final: como instalar (`pip install -e .` + `requirements/notebook.txt`), como executar/abrir, como regenerar dados.
- T4.2.3 `[tier:medium]` Gerar artefato de evidência: `*.executed.ipynb` ou HTML (DEC-03) em `docs/plans/qa/`.
**Testes:** TST4.2.a `[tier:medium]` execução em ambiente limpo (idealmente venv separada) sem erro; TST4.2.b `[tier:fast]` `ruff check` limpo em `notebooks/*.py` e `tests/*`.
**Aceitação:** CA-G1, CA-G7. **DoD:** README pronto; artefato de execução gerado. **QA:** realmente reprodutível do zero? `[tier:heavy]` **Commit:** `chore(notebook): reprodutibilidade + README + artefato de execução`.

#### Fase 4.3 — QA Global, DoD Global e Ensaio
**Pre-flight:** 4.2 OK; todos os DoD de fase cumpridos. `[tier:fast]`
**Tasks:**
- T4.3.1 `[tier:heavy]` **Senior QA Review Global** (rubrica Seção 4 sobre o conjunto): correção, cobertura agregada, reprodutibilidade, fidelidade científica, clareza, higiene de dados, não-poluição da lib. Produzir `docs/plans/qa/REVIEW-GLOBAL.md`. `[self:opus]`
- T4.3.2 `[tier:medium]` Corrigir **todos** os achados 🔴/🟡 do QA global; re-rodar suíte completa + execução do notebook.
- T4.3.3 `[tier:medium]` Validar **CA-G1…CA-G9** com evidências anexadas.
- T4.3.4 `[tier:fast]` Ensaio: verificar tempo de execução do notebook (apresentação) e que nada depende de rede.
- T4.3.5 `[tier:medium]` **DF-1 — Rodar tudo:** execução limpa de `pytest -q` (toda verde) + notebook ponta-a-ponta via `papermill`; conferir que as saídas batem com a narrativa; anexar logs e `*.executed.ipynb` em `docs/plans/qa/`.
- T4.3.6 `[tier:heavy]` **DF-2 — Conformidade:** varrer o plano item a item (Tasks/Testes/CA-G/DEC/R) e produzir `docs/plans/qa/conformidade-final.md` marcando implementado ✅ ou justificado 📝; **bloquear o "Done" se houver omissão sem justificativa**. `[self:opus]`
- T4.3.7 `[tier:medium]` **DF-3 — Auditoria didática:** percorrer o notebook garantindo que **toda célula de código é precedida por markdown explicativo** do passo seguinte; corrigir órfãs; registrar em `docs/plans/qa/auditoria-didatica.md`.
**Testes:** suíte completa `pytest -q` verde; execução ponta-a-ponta verde; `ruff` limpo.
- TST4.3.a `[tier:medium]` **Teste automatizado do DF-3** (`tests/test_notebook_didatica.py`): via `nbformat`, varrer as células e **falhar** se alguma célula `code` "substantiva" (que não seja import/setup trivial) **não** for precedida, dentro da mesma seção, por uma célula `markdown` não-vazia. Lista de exceções explícita e justificada.
**Aceitação:** **Critério de Aceitação Global (5.1)** satisfeito. **DoD:** **DoD Global (5.2)** satisfeito, **incluindo DF-1, DF-2 e DF-3 (Seção 5.2.1)**.
**Senior QA Review:** gate final — assinar o "Done". `[tier:heavy]`/`[self:opus]`
**Commit:** `chore(release): notebook gzcmd passo-a-passo concluído + QA global`.

**🔍 QA Review FINAL (gate global):** tudo verde, reprodutível, didático, fiel ao `gzcmd`. `[tier:heavy]`/`[self:opus]`

---

## 8. Matriz de Rastreabilidade (Critérios ↔ Fases)

| Critério Global | Onde é satisfeito |
|-----------------|-------------------|
| CA-G1 (executa ponta-a-ponta) | 2.1→4.2 (TST-NBEXEC), 4.2 |
| CA-G2 (estágios isolados + visual) | 2.2, 2.3, 2.4, 2.5, 2.6 |
| CA-G3 (reconciliação rota A in-sample) | 3.1 |
| CA-G4 (dataset sintético + `p_true` + edge cases) | 1.1, 1.2 |
| CA-G5 (métricas held-out + variância + PR/ROC) | 3.2 |
| CA-G6 (LLM stub) | 3.3 |
| CA-G7 (testes + cobertura + ruff) | todas as fases + 4.2, 4.3 |
| CA-G8 (calibração honesta: held-out + ECE/Brier + `p_true`) | 1.1, 2.4 |
| CA-G9 (didática: objetivos, fio-condutor, recap) | 2.1–2.6, 4.1 (+ Seção 6.5) |

---

## 9. Registro de Riscos e Mitigações

| ID | Risco | Severidade | Mitigação | Fase |
|----|-------|-----------|-----------|------|
| R-01 | Sem dataset de exemplo | Alta | Gerador sintético | Wave 1 |
| R-02 | Sem tooling de notebook | Média | `requirements/notebook.txt` | 0.1 |
| R-03 | `docs/plans/` ausente | Baixa | Criado | — |
| R-04 | Mismatch config×loader (features) | Média | Canary 0.2 + de-para; gerador emite o que o pipeline consome | 0.2, 1.1 |
| R-05 | LLM exige API | Média | Stub determinístico | 3.3 |
| R-06 | Fixtures de teste desconhecidas | Baixa | Inspeção no pre-flight 0.2 | 0.2 |
| R-07 | `triage` muta df | Média | Usar `df.copy()` por modo | 2.6, 3.1 |
| R-08 | Platt instável com poucos positivos | Média | Garantir balanço mínimo no gerador; teste de degradação | 1.1, 2.4 |
| R-09 | Notebook não-determinístico | Média | Seeds fixas; execução limpa | 4.2 |
| R-10 | Calibração in-sample no `run_v3` (vazamento) | Alta | Rotas A/B (DEC-07); métricas só held-out | 2.2.1, 2.4, 3.2 |
| R-11 | Config descreve mais do que o código implementa | Alta | Ensinar o código; canary de de-para; declarar no notebook | 0.2, 2.4 |
| R-12 | Circularidade do dataset sintético | Alta | Posterior verdadeira `p*` + sobreposição (DEC-06) | 1.1, 2.4 |
| R-13 | Não-determinismo do XGBoost | Média | `n_jobs=1`+seed; reconciliação tight só p/ Platt | 2.4, 3.1 |

---

## 10. Log de Execução (preencher durante a execução)

> Registrar aqui: suposições adotadas (D1), desvios do plano, decisões DEC resolvidas, achados de QA e suas correções, e qualquer parada por ambiguidade/bloqueio.

### Suposições e desvios (D1)
- **Fase 0.1 — sem `.venv` dedicada.** Usado o ambiente editável já instalado (miniconda base) em vez de criar `.venv`. Resultado funcional idêntico; shell não-interativo não persiste ativação. Comandos de `venv` documentados no `notebooks/README.md` para o usuário final. `pyproject [project]` **não** alterado (D7 respeitado).
- **`pyright` não instalado** (extra `dev` requer Node). Não-bloqueante: o `ruff` é o gate de lint do QA (eixo 5).
- **Contrato de API verificado por execução** (`docs/plans/qa/contrato-api.md`) — autoritativo. Desvios vs. texto do plano confirmados no código: (i) `triage` **retorna cópia** (não muta o df de entrada); (ii) guardrail `ALWAYS_MATCH` exige `nota_final ≥ 10` + nome/data/município perfeitos (não `≥ 9`). Notebook e gerador seguem o **código** (ground truth).

### Decisões DEC resolvidas
- **DEC-01** gerador em `notebooks/` + `pythonpath=["src","notebooks"]`. ✅
- **DEC-02** Platt principal; **apêndice XGBoost mantido markdown-only** (não executado) por R-13 (não-determinismo) + escopo. 📝
- **DEC-03** `.ipynb` limpo versionado; artefato `*.executed.ipynb` (507 KB) gerado em `docs/plans/qa/`, **não** versionado (`.gitignore`). ✅
- **DEC-06** posterior verdadeira `p*(x)=σ(0.85·(nota−s0))`, `TARGET~Bernoulli(p*)`; coluna de validação `p_true` nunca entra no pipeline. ✅ (Platt recupera `p*` com MAE ~0.009 in-sample / ECE pequeno held-out.)
- **DEC-07** rotas A (fiel/in-sample) e B (correta/held-out) explícitas. ✅
- **DEC-08** ECE + Brier em `nb_helpers` com testes fechados. ✅
- **DEC-10** `ipywidgets` interativo **implementado** (Fase 4.4, seção 15 "Painel interativo"): sliders de limiar + escala do *slope* → precisão/recall/custo ao vivo, com figura estática de fallback **headless-safe** (`try/except`) que não quebra `papermill`/`nbconvert` (CA-G1). ✅

### Achados de QA e correções
- **Fase 3.2 (eixo 9):** demo de vazamento `row` vs grupo mostrou efeito **negligenciável** (grupos majoritariamente singletons). Corrigido para **medir e explicar com honestidade** o mecanismo (sem forjar inflação). 📝 (R-12.)
- **Fase 4.1:** corrigido LaTeX da seção 11 (subscritos `c_{fp}` etc. renderizavam literalmente); `value_counts` de bandas com `reindex(fill_value=0)`; simplificação de agregação de flags. Backlog 🟢 zerado.
- **R-11** (config × código: `anchor_platt`/`by_band` não implementados) declarado no notebook (seção 9.6).
- **R-05** estágio LLM por **stub determinístico** (`llm_review_stub`), sem rede; honestidade explícita de que é simulação.

### Gate final
- **DF-1/DF-2/DF-3** satisfeitos (`pytest` 101 passed + `nbconvert` exit 0; `conformidade-final.md`; `auditoria-didatica.md` + `test_notebook_didatica.py`).
- **QA Global** assinado em `docs/plans/qa/REVIEW-GLOBAL.md` (sem 🔴/🟡; 🟢 = DEC-10/DEC-02 backlog).
- Nenhuma parada por ambiguidade/bloqueio crítico durante a execução.

---

## 11. Resumo dos Comandos-Chave (pwsh / win32)

```pwsh
# Setup
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .[dev]
python -m pip install -r requirements/notebook.txt

# Testes
pytest -q
pytest -q --cov=synthetic_data --cov=nb_helpers --cov-report=term-missing
# (nb_helpers inclui expected_calibration_error / brier_score / llm_review_stub)

# Lint/format
ruff check .
ruff format --check .

# Executar o notebook (reprodutibilidade / teste de execução)
papermill notebooks/gzcmd_passo_a_passo.ipynb docs/plans/qa/gzcmd_passo_a_passo.executed.ipynb
# ou
jupyter nbconvert --to notebook --execute notebooks/gzcmd_passo_a_passo.ipynb
```

---

## 12. Histórico de Revisão Técnica (PhD em ML + Didática)

Revisão realizada com verificação no código-fonte (não inferência). Achados classificados e onde foram endereçados:

| # | Achado | Severidade | Correção no plano |
|---|--------|-----------|-------------------|
| 1 | `run_v3` calibra **in-sample** (sem split) → vazamento; *reliability diagram* sobre os mesmos dados é enganoso. (`runner.py:135-180`) | 🔴 | 2.2.1, DEC-07 (rotas A/B), R-10, Fase 2.4 (rota B held-out), Fase 3.2 |
| 2 | **Circularidade do sintético:** rótulos derivados das mesmas features tornam a calibração trivial. | 🔴 | DEC-06 (`p*` conhecida + sobreposição), R-12, T1.1.1, TST1.1.f/g |
| 3 | **Faltavam métricas quantitativas** de calibração (só curva). | 🔴 | DEC-08 (ECE + Brier), CA-G8, TST2.4.c/e |
| 4 | **Config × código divergem** (`anchor_platt`/`by_band` não implementados; Platt é global). (`calibration.py:163`) | 🔴 | 2.2.1(3), R-11, T2.4.5 |
| 5 | **Vazamento por registro compartilhado** em record linkage (split por linha vaza). | 🟡 | Split group-aware (T1.1.6, `eval.py` `split_by`), TST1.1.h, T3.2.3/TST3.2.c |
| 6 | **Sem variância**: ponto único de métrica engana. | 🟡 | Multi-seed (≥5) + erro-padrão promovido a obrigatório (T3.2.2) |
| 7 | **Reconciliação ambígua** quanto a in-sample × held-out; XGBoost não-determinístico. | 🟡 | Fase 3.1 (rota A explícita; Platt 100%/atol≤1e-9; XGB tolerante), R-13, TST3.1.c |
| 8 | **Didática frágil:** sem objetivos de aprendizagem, sem fio-condutor, formalismo sem intuição. | 🟡 | DEC-09, Seção 6.5, eixo 10 do QA, CA-G9 |
| 9 | **Política desconectada** das curvas PR/ROC e do custo. | 🟡 | T3.2.4 (PR/ROC + superfície de custo ligada a `evr`/FP/FN) |
| 10 | **Prevalência/base rate** não discutida (afeta calibração e ponto de operação). | 🟢 | 4.1 edge cases, T2.4.4, T3.2.5 |
| 11 | **Derivação do Platt** superficial (símbolos indefinidos). | 🟢 | DEC-09 item 4, T2.4.1 |
| 12 | **Interatividade** ausente (alto valor para apresentação). | 🟢 | DEC-10 (`ipywidgets` opcional, protegido para headless) |

**Veredito da revisão:** o plano original era sólido em engenharia/processo, mas tinha **um furo metodológico central** (calibração in-sample apresentada como evidência) e **risco de circularidade** no sintético — ambos agora corrigidos com as rotas A/B e a posterior verdadeira `p*`. Com as adições (ECE/Brier, held-out, multi-seed, split group-aware, andaime didático), o notebook passa de "demo bonita" para **material acadêmico defensável**.

---

> **Próximo passo após aprovação deste plano:** iniciar **WAVE 0 / Fase 0.1**, seguindo D1 (execução iterativa contínua) e D2 (commit often), com pre-flight antes e QA sênior depois de cada fase.
