# Changelog

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
