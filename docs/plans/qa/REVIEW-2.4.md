# Senior QA Review — Fase 2.4 (Calibração Platt: rotas A/B + ECE/Brier + validação vs p_true)

**Revisor:** Orquestrador (Opus) — `[self:opus]`. **Veredito: APROVADA** (sem 🔴/🟡 pendentes).

A Fase 2.4 é a "joia metodológica" do plano (CA-G8). Revisão cética nos 10 eixos:

## Achados e resolução

| Eixo | Avaliação | Evidência |
|------|-----------|-----------|
| 1. Correção funcional | ✅ | ECE/Brier validados por casos fechados (`test_nb_helpers.py`, 11 testes). Rota A usa `fit_platt_from_df` + `compute_p_cal`; Rota B ajusta no treino, prevê no teste. |
| 2. Cobertura | ✅ | `nb_helpers` 98% (única linha não-coberta = `llm_review_stub`, NotImplemented, diferido p/ Fase 3.3). TST2.4.a/b/c/d/e implementados. |
| 3. Determinismo | ✅ | seed 42; `fit_platt` é Newton-Raphson sem RNG; split *group-aware* semeado. |
| 4. Fidelidade científica | ✅ | Vazamento (R-10) explicitado; *reliability diagram* **held-out**; ECE/Brier numéricos; recuperação de `p*` provada; divergência config×código (R-11) declarada. |
| 5. Qualidade de código | ✅ | `ruff check` limpo em `notebooks/` + `tests/`. |
| 6. Clareza didática | ✅ | Objetivos, intuição→formalismo, hero card, recap. |
| 7. Higiene de dados | ✅ | 100% sintético; `p_true` nunca no CSV de entrada. |
| 8. Não-poluição da lib | ✅ | ECE/Brier em `notebooks/nb_helpers.py` (fora de `src/`). |
| 9. Rigor estatístico | ✅ | Held-out + ECE + Brier + validação vs `p*` + sem vazamento (group-aware testado). Variância multi-seed é da Fase 3.2 (conforme plano). |
| 10. Andaime didático | ✅ | Derivação do Platt define todos os símbolos (`a`,`b`,`σ`,`s*=-b/a`, NLL, L2-on-slope); intuição antes da fórmula. |

## Achado 🐛 corrigido durante a fase
- **TST2.4.b (degradação elegante):** hipótese inicial errada (esperava que classe única "não quebrasse"). O código real levanta `RuntimeError` explícito (Hessiana singular) — isto **é** a degradação graciosa correta (falha alta e nomeada, não NaN silencioso). Teste reescrito para asseverar o `RuntimeError` na classe única e ajuste finito com poucos positivos (2/40).
- **TST do ECE (bins):** caso de "bin único" ajustado — pontos 0.54/0.55 cruzavam fronteira de bin em n_bins=20; corrigido para faixa única [0.50,0.55).

## Decisão de escopo registrada (D1 / DEC-02)
- **T2.4.6 (apêndice XGBoost):** implementado como **markdown conceitual**, sem execução ao vivo. Justificativa: (a) R-13 — XGBoost é não-determinístico entre threads/execuções, ameaçando CA-G1 (execução headless reprodutível); (b) não acrescenta clareza sobre *calibração* (foco da seção). DEC-02 já rebaixa XGBoost a "apêndice opcional". Caminho ML×Platt descrito com a ressalva `n_jobs=1`+seed. **Não bloqueia** nenhum CA-G.

## Verificação
- `pytest` completo: **82 passed** (was 67; +15). NBEXEC (notebook ponta-a-ponta) verde (~42s).
- `ruff check notebooks/ tests/`: All checks passed.
