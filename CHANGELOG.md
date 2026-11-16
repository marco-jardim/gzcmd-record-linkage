# Changelog

## 0.7.0 (não publicada)

Método **experimental**: a fila de revisão auto-calibrável não existe na tese nem
na 0.6.1 e **não é recomendada para uso em produção**. Na simulação offline sobre
o extrato da tese ela não tem vantagem sobre a ordenação pela nota (abaixo do
corte, em 1.300 revisões: 7,2 contra 10,9 dos 36 pares PAR = 2). Com o Haiku real
(uma realização, humano simulado perfeito), a fila também não superou as bases:
com ~127 decisões humanas verificou 4 dos 36 pares abaixo do corte, contra 10 de
nota + LLM com humano confirmando os MATCH e 9 do humano sozinho na ordem da
nota. Todas as mudanças são aditivas: `run_v3` e todos os
defaults existentes permanecem inalterados.

Adicionado:

- `self_calibrating_queue.py`: revisão sem padrão-ouro (a sessão nunca lê `PAR`
  nem `TARGET`) sobre a saída ORL + CdR.
  - `SelfCalibratingSession`: sessão passo a passo (`step()` devolve exatamente
    um `SessionEvent`); `SelfCalibratingConfig`; eventos serializáveis em JSON
    (`ordering`, `pair_presented`, `agent_verdict`, `final_verdict`,
    `review_error`, `round_stopped`, `gate_opened`, `human_decision`,
    `holdout_built`, `alarm`, `recalibration`, `reordering`, `run_finished`);
    estado em `to_state()`/`from_state()`; `results()`, `summary()`,
    `holdout()`, `current_model`, `feature_names`. `run_self_calibrating`: laço
    de conveniência.
  - Ordem: faixas por **posto** de `p_cur` entre os pares não revisados no início
    da rodada (`band_rank_edges=(0.01, 0.05, 0.20)`), cada faixa consumida **do
    topo para baixo**, escolhida por um bandit de Thompson; braço de exploração
    fora do bandit e da regra de parada (`explore_every=10`, `0` desliga). A 1ª
    rodada ordena pela nota calibrada; depois, o modelo vigente (prior
    Fellegi-Sunter ajustado uma vez na inicialização e, com rótulos suficientes,
    a logística MAP).
  - Portão humano (`mode="human"`, padrão): a sessão pausa em `awaiting_human`
    até `submit_human_decisions(...)`, que exige uma decisão (`"MATCH"`,
    `"NONMATCH"` ou `"SKIP"`) para **todo** item; não há confirmação implícita.
    Orçamento humano (`human_budget_per_round=50`, `human_budget_total=None`),
    com prioridade MATCH do LLM (por `p_cur`), UNSURE (`gate_include_unsure`) e
    auditoria; o excedente fica `over_budget`, sem rótulo, e o orçamento total
    esgotado encerra o run (`human_budget`). Auditoria por posto
    (`audit_rank_fraction=0.05`, `nonmatch_audit_n=10`) com alarme informativo
    (`audit_match_found`, verificação de sanidade sem poder estatístico).
    `llm_match_unverified` vale para todo MATCH do LLM sem decisão humana. No
    modo humano um positivo exige MATCH humano; negativos por `negative_labels`
    (`"llm_nonmatch"` ou `"verified_only"`).
  - Holdout acumulado com pesos por estrato (`r{r}:top`, peso 1; `r{r}:explore`,
    peso N/n_e) e teto `weight_max_ratio`, que troca viés por variância.
  - Campos (`field_set`): os 5 compostos do loader (padrão) ou os 29 sub-escores
    do CdR + `mae_missing` (`"cdr_subscores"`, experimental: no extrato da tese o
    FS sem rótulos dá prevalência 0,49 e a simulação foi pior).
  - Modo automático (`mode="auto"`): exige opt-in explícito e emite
    `ExperimentalWarning`; a saída é marcada `llm_match_unverified`. Guarda de
    região dominante: com `auto_low_conf_policy="exclude"` e mais de
    `auto_max_low_conf_fraction` (padrão 0.5) dos pares na região de baixa
    confiança, o run não aprende (`recalibration` com
    `status="skipped_low_conf_dominant"`, ordem pelo prior Fellegi-Sunter, um
    `alarm` `low_confidence_dominant` na rodada 1).
  - Validações de entrada: `SCORE` resolvido com sufixo `,Tipo` e vírgula
    decimal; `remove_fields` rejeita `str` e duplicatas e é validado na criação
    da sessão (dossiê da 1ª linha); a impressão digital cobre as colunas que o
    dossiê lê (`DOSSIER_SOURCE_COLUMNS`); a configuração aceita inteiros numpy;
    `zero_match_reviews_needed` é exato.
  - `OracleReviewer`: revisor simulado determinístico, sem rede, para testes e
    simulações.
- `fellegi_sunter.py`: discretização dos campos (compostos ou sub-escores do CdR)
  e Fellegi-Sunter por EM, sem rótulos (prior da recalibração). Os sub-escores do
  mesmo campo são correlacionados: a independência condicional é violada e os
  pesos do prior ficam inflados.
- `field_recalibrator.py`: logística MAP multicampo, com prior, limites e pesos;
  o modelo-prior (`as_recalibration`) é projetado nos limites (coeficientes de
  concordância ≥ 0 e caixa de confiança), como o ponto inicial do ajuste.
- APIs aditivas em módulos existentes:
  - `ranking.ThompsonQueue(input_kind="partial_orders", n_items=...)`, mais
    `to_state`/`from_state`, `arm_stats`, `exhausted`, `arm_of` e
    `mark_seen(item)` (marca como visto um item não pendente, para os braços o
    pularem; idempotente; entra no estado);
  - `loader.LoadConfig(require_par=False)` (sem `PAR` não cria `TARGET`);
  - `llm_dossier.DossierView(remove_fields=...)` e a constante pública
    `DOSSIER_SOURCE_COLUMNS`.
- Plano de desenho e limitações conhecidas: `docs/plans/fila-autocalibravel.md`.
- Simulação offline da fila auto-calibrável sobre o extrato da tese, com revisor
  sintético por faixa de nota (`BandedOracle`) e humano simulado, sem rede e sem
  LLM real, comparada a bases de mesmo orçamento (nota, Fellegi-Sunter,
  aleatória): `notebooks/sim_fila_autocalibravel.py`,
  `tests/test_sim_fila_autocalibravel.py`, relatório
  `docs/sim_fila_autocalibravel.md` e agregados
  `docs/sim_fila_autocalibravel_resultados.csv` (só agregados, sem dado por par;
  não é avaliação com LLM real). A simulação v3 inclui as bases de esforço
  humano (H0 humano sozinho, H1 nota + LLM tria + humano confirma MATCH, H2 a
  fila), a grade de `negative_labels` × `human_budget_per_round` e o terceiro
  revisor opcional `--rates-csv`. Os resultados das versões anteriores do
  desenho ficam em `docs/sim_fila_autocalibravel_v2.md`,
  `docs/sim_fila_autocalibravel_v1.md` e nos CSVs `..._resultados_v2.csv` e
  `..._resultados_v1.csv`.
- `SelfCalibratingSession.peek_upcoming(n_per_arm)`: os próximos `pair_id` que
  cada braço apresentaria (faixas e exploração), só leitura (`to_state()` não
  muda); serve para pré-buscar revisões em paralelo.
- Harness de avaliação com LLM real (experimental, fora do `run_v3`; não é
  importado pelo pacote): `notebooks/llm_real_eval.py`, com os subcomandos
  `pilot`, `calibrate`, `report` (taxas por faixa), `queue-run` e `queue-report`
  (fila ponta a ponta), `baseline` (bases H1 e H0), `export-fixture` e
  `rerank-test` (LLM como desempate abaixo do corte), e `FixtureReviewer`, que
  reaplica sem rede as decisões gravadas. A chave é lida só pelo harness, de
  `ANTHROPIC_API_KEY_ALT` no `.env` (ignorado pelo git); as decisões reais ficam
  num fixture local em `data/llm_fixtures/` (ignorado pelo git).
- Relatórios agregados (sem dado por par) com o Haiku real, prompt v4 e sem a
  nota: `docs/llm_real_haiku_sem_nota.md` (taxas: FP 0,4% [0,1–1,5%], FN 18,2%
  [13,9–23,5%]; abaixo do corte, FN 44,4% [29,5–60,4%], 16 de 36),
  `docs/llm_real_fila_ponta_a_ponta.md` e
  `docs/llm_real_desempate_zona_cinza.md` (ganho só com ~400 decisões humanas:
  +5 e +6 de 36 nas janelas de 2.000 e 3.000; uma realização, evidência fraca).

Alterado:

- Versão do pacote: 0.7.0 (`pyproject.toml`). Nenhum comportamento existente
  muda; os padrões de `LoadConfig` e `DossierView` e todo o caminho de `run_v3`
  são os mesmos.

## 0.6.1 (2026-11-14)

Documentação: README atualizado para as capacidades da 0.5.0/0.6.0. Nenhuma
mudança de comportamento do código. Afirmações corrigidas:

- "Calibrated": o pacote não "só calibra por Platt"; também há a rota
  `p_cal="ml_xgb"` (isotônica OOF) e a recalibração isotônica do arranjo
  sequencial.
- Taxas empíricas: "não são carregadas automaticamente" deixou de valer;
  `llm_review.reliability.estimated_rates_path` as carrega na v3.1 e o perfil v3.1
  empacotado já as inclui.
- Reprodução v3.1: o perfil `packaged_config_path("v3.1")` (custo LLM 0,005,
  guardas 9/5, `p_cal: ml_xgb`) existe; deixou de ser necessário "fornecer seu
  YAML" para esses parâmetros.
- `eval`: não é mais descrito como avaliador somente v3; aceita
  `--policy-version {v3,v3.1}` (mantida a ressalva de que ações fora de `MATCH`,
  `NONMATCH` e `LLM_REVIEW` contam como indecisas).
- Configuração distribuída: são dois perfis (`gzcmd_v3_config.yaml` e
  `gzcmd_v31_config.yaml`), não só o v3; a descrição de `gpt-5.2-pro` vale para o
  perfil v3.
- Paridade: a frase sobre OOF e taxas empíricas "não incluídas" foi qualificada.
- Ordenação da fila: "Não há nova chave de configuração nem opção CLI" passou a
  se referir explicitamente à ordenação (`ranking`).
- Exemplo mínimo: rótulo "Saída na versão 0.3.0" atualizado (saída verificada na
  0.6.1, idêntica).
- Versões, instalação e citação atualizadas para 0.6.1; faltava linha em branco
  antes de "Linha de comando".
## 0.6.0 (2026-11-13)

A 0.5.0 nunca foi publicada no PyPI (a anterior publicada é a 0.4.2): esta versão
inclui tudo o que está descrito abaixo e na seção 0.5.0.

Alterado (**incompatível**):

- `run_v3(p_cal="config")` (e `gzcmd run --p-cal config`) passa a ler
  `calibration.p_cal` do YAML quando declarado e só depois `calibration.method`.
  O perfil v3.1 empacotado declara `p_cal: ml_xgb`, logo `p_cal="config"` com ele
  agora seleciona `ml_xgb` (antes: `anchor_platt`). O perfil v3 não muda.
- `calibration.p_cal` do YAML é validado contra `CALIBRATION_ROUTES`; valor
  desconhecido levanta `ValueError` ao carregar a configuração.

Adicionado:

- `np_evaluation.py` (`evaluate_np_point`, `compare_confirmation`, `NP_RATES`):
  avaliação NP da tese para `confirmacao` (política de custo e NP com alfa
  0,001, 0,002 e 0,01). Restrição inviável gera linha com `viavel=False` e os
  diagnósticos do `NPInfeasibleError`; nenhuma métrica é fabricada.
  `evaluate_birth_variant` não foi portado.
- `cost_evaluation.py` (`expected_funnel_metrics`): métricas esperadas do funil
  sobre as taxas dos revisores, sem respostas simuladas de LLM.
- `intervals.wilson95` e `provenance.py` (`sha256_normalized`,
  `sha256_file_normalized`, hash independente de final de linha).
- `guardrails.strong_birth_conflict` e `guardrails.high_guard_birth_variant`,
  variantes só de avaliação; `apply_guardrails` não as usa.
- `arranjo_corte.py`: arranjo sequencial (ASR) que mantém o corte do
  OpenRecLink (`below_cutoff`, `LogitPlatt`, `nested_cutoff_probabilities`,
  `estimate_rates`, `crossfit_rates`, `cache_matches`, `summarize_actions`,
  constantes `THESIS_*`). P-Platt é análise de sensibilidade; a checagem da
  partição da tese é opcional (`expected`).
- `arranjo_robustez.py`: grade de réplicas, `dominates`, `classify`,
  `action_stability`, `safety_stop` parametrizado, utilitários de *checkpoint*.
- `anthropic_transport.py` (`AnthropicReviewer`, `caching_metadata`) e extra
  `[anthropic]`: transporte da API Messages com prefixo do prompt em cache;
  `AnthropicReviewer.from_env` lê `ANTHROPIC_API_KEY`.
- `llm_review`: `prompt_version` (`"v3"` padrão, `"v4"`) em `LLMReviewer` e nos
  construtores de mensagens; prompt empacotado `gzcmd_v4_llm_prompt.md`.
- `llm_dossier.DossierView` (`remove_model_outputs`): dossiê `sem_mo` da tese;
  `Dossier` e `SCHEMA_VERSION` inalterados.
- `subamostragem.py`: subamostragem de negativos só no treinamento, com a
  correção de prior de Dal Pozzolo (`sample_training_indices`, `correct_prior`,
  `undersampled_oof`, `evaluate`, `summarize`); exploratório e *post hoc*.
- `oof.py`: `numeric_features`, `pair_ids` e `fit_xgb_oof` (XGBoost + isotônica
  aninhados, com tabela persistida, impressão digital e reuso validado).
- `runner.run_v3`: rota `p_cal="ml_xgb"` e argumentos `ml_oof_path`,
  `reuse_ml_oof`, `split_by` e `rates_yaml`; `build_engine_from_config` aceita
  `rates_yaml`. CLI: `run --p-cal {ml_xgb,platt,anchor_platt,config}`,
  `--ml-oof-path`, `--reuse-ml-oof`, `--rates-yaml`.
- `classifier`: `FeatureSet`, `COMPARATOR_FEATURES` (30 atributos),
  `feature_set` e `macd_enabled` em `get_feature_columns` e `GZCMDClassifier`
  (modelos salvos antes usam a seleção histórica). `eval` e a CLI:
  `--feature-allowlist` e `--policy-version {v3,v3.1}`.
- `config.CALIBRATION_ROUTES` e `CalibrationConfig.p_cal`;
  `LLMReviewConfig.error_rates_for_mode(mode, source=None)`.

Alterado:

- `eval` sem `--policy-version` mantém o comportamento v3 anterior da
  biblioteca (desvio deliberado da tese, que segue o YAML); com `v3` sobre YAML
  v3.1 restaura as guardas 10/3.
- `guardrails.py`: apenas reformatação de linhas longas, sem mudança de
  comportamento.

Corrigido:

- Mensagem de erro de `p_cal` inválido em `run_v3` passa a listar `ml_xgb`.

Ver `docs/PARIDADE_v31.md` para o que ficou fora do pacote por decisão.

Da rodada anterior (fusões, NP, *bootstrap* e filas de contexto; antes sob
"Não publicado"):

Adicionado:

- `fusion.py`: fusões de filas da tese com paridade exata de
  `scripts/fusao_filas.py`. `FusionSpec`, `THESIS_FUSIONS` (F1-F4, A1, A2),
  `fuse`, `simulate_thompson` (simulação em lote com rótulos, sobre
  `ThompsonQueue`), `queue_metrics` (revisões até a meta, recuperação em `k`,
  AUC de recuperação) e `compare_queues` (média e P10/P90 das fusões
  adaptativas sobre sementes). Nenhuma fusão é padrão recomendado: a F4 foi
  escolhida *post hoc*. Nenhuma alteração em `ranking.py`.
- `robustness.py`: *bootstrap* por `REFREC` das fusões (`fusao_robustez.py`):
  `bootstrap_fusion_differences`, `BootstrapResult`, `subset_effort`. Resultado
  condicional às probabilidades OOF e à referência fixas, sem incerteza de
  treinamento nem erro da revisão; escolha da F4 *post hoc*.
- `np_policy.py` (`NPConstraints`, `NPInfeasibleError`) e `policy_kind="np"`
  (motor, `config.py`, `runner.py`): política NP da tese, só para `confirmacao`
  e só na v3.1; sob a v3, em `run_v3`/`build_engine_from_config`, o NP é
  rebaixado silenciosamente para custo, como na tese (o motor direto levanta
  `ValueError`).
- `context.py` (`context_features`, `context_queues`, `THESIS_CONTEXT_FUSIONS`,
  `MACD_COLUMNS`) e `oof.py` (`nested_oof_probabilities`): filas de contexto
  C/CM e isotônica OOF aninhada, de `contexto_obito.py` e `ml_oof.py`. Uso
  exploratório; `COMPREC` pode cruzar partições de `REFREC`.

Corrigido:

- `loader.parse_date_yyyymmdd`/`parse_date_ddmmyyyy`: coluna de data
  inteiramente ausente (float/NaN no pandas) quebrava com `'float' object has no
  attribute 'zfill'`, impedindo `load_comparador_csv` de carregar a base da tese.
  Datas ausentes agora tratadas como na tese; teste de regressão e de paridade
  da derivação MACD.

## 0.5.0 (2026-11-13, nunca publicada; incluída na 0.6.0)

Adicionado:

- Perfil v3.1 empacotado `gzcmd_v31_config.yaml` e
  `packaged_config_path("v3"|"v3.1")`.
- Suporte a `estimated_rates_path` em `llm_review.reliability`: novo
  `reliability.py`, portabilidade de `gzcmd/reliability.py` da tese (bloco
  `claude-haiku-5-5` para o prompt v4/dossiê `sem_mo`; taxas medidas só quando
  n >= 10, senão os valores fixos inline, incluindo `e_unsure`).
- `error_rates_by_band_v4.yaml` empacotado (Claude Haiku 5.5; 1774 e 459
  decisões), cópia byte a byte do arquivo da tese.
- `ranking`: `order_by_score`, `top_k_cut` (rótulos `examined` e `not_examined`,
  nunca não vínculo), `effort_curve`, `reviews_to_target`,
  `recovered_at_budget` e `tie_ids`.

Alterado (**incompatível**):

- O desempate padrão em `rrf`, `interleave` e `order_by_score` passa a ser a
  regra da tese (maior escore primeiro, depois menor índice) e `review_queue`
  desempata por `pair_id` crescente. Para o comportamento anterior (permutação
  aleatória com semente), use `tie_break="random"`.
- Configurações v3.1 com `estimated_rates_path` passam a usar as taxas medidas
  por banda, em vez das taxas inline. A política v3 (padrão) não muda e nunca
  lê esse arquivo.
- Para bandas sem decisões, `e_unsure` segue a tese (valor inline), diferente do
  notebook tutorial (taxa do modo).

## 0.4.2 (2026-11-07)

- Adição de `ranking.py` como utilitário opt-in de ordenação de filas já
  selecionadas: RRF, intercalação com deduplicação, fila adaptativa de Thompson,
  validação de permutações e `review_queue` para DataFrames.
- Desempate de escores e RRF por embaralhamento com semente fixa; feedback
  binário de Thompson apenas para o braço selecionado, com janela por braço.
- Política, padrões, CLI e configuração preservados; o modo
  `evr` do utilitário mantém a ordem recebida sem recalcular decisões.

## 0.4.1 (2026-11-06)

- Correção do dossiê para não incluir colunas inexistentes no extrato histórico.
- Documentação de `consistency.conflict_margin_delta` como configuração reservada,
  sem cardinalidade N:1 nem resolução de conflitos por margem de probabilidade.
- Documentação dos defaults dos revisores no YAML e de sua distinção em relação
  aos parâmetros dos experimentos da tese.
- Nome histórico GZ-CMD e limites dos termos da expansão explicitados no README.

## 0.4.0 (no repositório, ainda não publicada)

- Política v3.1 (quatro ações) via `policy_version`; padrão permanece v3.
- API Python, YAML (`guardrails.policy_version`) e CLI (`--policy-version`).
- Revisão humana (`DEFERRED`), erro humano, abstenção LLM e orçamento humano
  configuráveis, com paridade sintética contra o motor de pesquisa.
- `RunSummary` acrescenta `human_used`, `human_max` e `policy_version`; a saída
  textual da CLI acrescenta as linhas `human_used=...` e `policy_version=...`.
- Override histórico v3 em YAML v3.1 restaura custo LLM 0,25 e guardas 10/3.
- `Costs.human_review` é campo real do dataclass, incluído em `asdict` e `replace`;
  o helper do notebook preserva esse custo ao explorar outros custos.

## 0.3.0 (2026-11-03)

- Calibração por âncoras (`fit_anchor_platt`) com alvos de prevalência 0,979/0,0001.
- `logit_mean_and_gradient` público (alias `_logit_mean_and_gradient` mantido).
