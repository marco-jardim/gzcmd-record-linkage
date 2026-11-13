# Changelog

## Não publicado

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

## 0.5.0 (2026-11-13, no repositório, ainda não publicada)

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
