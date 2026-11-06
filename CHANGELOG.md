# Changelog

## 0.4.1 (no repositório, ainda não publicada)

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
