# Changelog

## 0.4.0 (no repositório, ainda não publicada)

- Política v3.1 (quatro ações) via `policy_version`; padrão permanece v3.
- API Python, YAML (`guardrails.policy_version`) e CLI (`--policy-version`).
- Revisão humana (`DEFERRED`), erro humano, abstenção LLM e orçamento humano
  configuráveis, com paridade sintética contra o motor de pesquisa.

## 0.3.0 (2026-11-03)

- Calibração por âncoras (`fit_anchor_platt`) com alvos de prevalência 0,979/0,0001.
- `logit_mean_and_gradient` público (alias `_logit_mean_and_gradient` mantido).
