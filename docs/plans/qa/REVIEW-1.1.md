# Senior QA Review — Fase 1.1 (Núcleo do Gerador Sintético)

**Revisor:** orquestrador (self:opus, equivale a [tier:heavy]). **Estado:** 55 passed; ruff lint+format limpos; cobertura `synthetic_data` = **95%** (≥90% DoD).

## Escopo
- `notebooks/synthetic_data.py`: `generate_comparador`, `to_comparador_csv`, `group_aware_split_indices`, `SyntheticDataset`.
- `tests/test_synthetic_data.py`: TST1.1.a–h + determinismo do split (8 testes).

## Verificações independentes executadas (não confiei no auto-relato do subagente)
1. **Testes:** 8 passed isolados; 55 passed na suíte completa (sem regressão). ✅
2. **Lint/format:** `ruff check` + `ruff format --check` limpos. ✅
3. **Cobertura:** 95% (linhas não cobertas são guardas defensivas: `n_pairs<=0`, ramo de data degenerada, branches do wrapper de split). ✅
4. **Recuperação da posterior (CRÍTICO p/ CA-G8):** ajustei o Platt real (`fit_platt_from_df`) ao dataset (seed 42, 600 pares):
   - `true_slope=0.85` → ajustado `0.902`; `true_intercept=-5.131` → ajustado `-5.438`.
   - **`mean|p_cal − p_true| = 0.0091`** (in-sample) — o Platt **recupera** a verdade-base dentro de ~1%.
   - `match_ratio` realizado = `0.502` (alvo 0.5). ✅

## Rubrica (Seção 4)
1. Correção funcional ✅ (schema exato; CSV carrega via `load_comparador_csv`).
2. Cobertura ✅ 95%; edge cases (fronteiras 5–9, mãe ausente, endereço zero, datas, grupos) testados.
3. Determinismo ✅ (RNG único `default_rng(seed)`; sem relógio/rede; `frame.equals` por seed).
4. Fidelidade científica ✅ **anti-circular** (`nota_final` gerada ANTES de `TARGET`; modelo bem-especificado sigmoid(a·nota−a·s0) recuperável; AUC∈(0.5,0.99) com sobreposição).
5. Qualidade de código ✅ ruff E,F,I,UP,B,SIM; sem `type: ignore`.
6. Didática ✅ docstrings PT-BR explicam o porquê do design anti-circular.
7. Higiene ✅ 100% sintético; `p_true` separado, NUNCA escrito no CSV de entrada (`to_comparador_csv` faz drop defensivo).
8. Não-poluição ✅ fora de `src/`.
9. Rigor estatístico ✅ posterior verdadeira `p*` validada; monotonicidade por bins (TST1.1.g).

## Achados
- 🟢 ALWAYS_MATCH (guardrail) exige `nota>=10` + nome/data/município perfeitos — o núcleo raramente satisfaz todos simultaneamente; **garantido pelos cenários da Fase 1.2** (documentado, não é defeito do núcleo).
- Nenhum 🔴/🟡.

## Veredito
**Done.** Pronto para Fase 1.2 (cenários narrativos + edge cases rotulados).
