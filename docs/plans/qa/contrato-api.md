# Contrato de API verificado — WAVE 0 / Fase 0.2

> Verificado **por leitura do código-fonte** (não inferência). Cada item abaixo
> tem a assinatura/coluna exata. Onde o código diverge da Seção 2.2 do plano ou
> da config, está marcado como ⚠️ **DIVERGÊNCIA**.

## 1. Carga e feature engineering (`loader.py`)

`load_comparador_csv(path=Path("data")/"COMPARADORSEMIDENT.csv", *, cfg=LoadConfig())`
- CSV: `sep=";"`, `decimal=","`, `encoding="utf-8"` com fallback `latin-1`.
- `LoadConfig(macd_enabled: bool = True)` (frozen).
- `na_values=["", " ", "NA", "N/A", "null", "None"]`.

### Colunas RAW consumidas (nomes EXATOS — o gerador deve emitir estas)
- Metadados: `COMPREC,C,12,0`, `REFREC,C,12,0`, `PASSO`, `PAR`, `nota final`.
- Datas: `R_DTNASC,C,8,0` (YYYYMMDD), `C_DTNASC,C,8,0` (YYYYMMDD),
  `R_DTOBITO,C,10,0` (**DDMMYYYY** — formato diferente!), `C_DTDIAG,C,10,0` (YYYYMMDD).
- NOME: `NOME qtd frag iguais`, `NOME prim frag igual`, `NOME ult frag igual`,
  `NOME prim ult frag igual`.
- NOMEMAE: `NOMEMAE qtd frag iguais`, `NOMEMAE prim frag igual`,
  `NOMEMAE ult frag igual`, `NOMEMAE prim ult frag igual`.
- DTNASC subscores (alimentam `dtnasc_score_total`/`dtnasc_all_zero`):
  `DTNASC dt iguais`, `DTNASC dt ap 1digi`, `DTNASC dt inv dia`,
  `DTNASC dt inv mes`, `DTNASC dt inv ano`.
- ENDERECO (os 6 EXATOS que alimentam `endereco_score_total`/`endereco_zero`):
  `ENDERECO via igual`, `ENDERECO via prox`, `ENDERECO numero igual`,
  `ENDERECO compl prox`, `ENDERECO texto prox`, `ENDERECO tokens jacc`.
- `CODMUNRES local igual` → `municipio_score`.

> ⚠️ **R-04 RESOLVIDO:** o `_base_df()` de `tests/test_loader.py` usa nomes de
> ENDERECO **diferentes** (`ENDERECO qtd frag iguais`, `prim frag igual`, …) que
> **não** são os agregados pelo loader → naquele fixture `endereco_score_total=0`
> e `endereco_zero=1`. **O gerador emitirá os 6 nomes corretos acima** para que
> `endereco_score_total` seja significativo.

### Colunas ENGENHEIRADAS produzidas (consumidas a jusante)
- `TARGET` = `1` se `PAR ∈ {1,2}` senão `0` (int8).
- `nota_final` (alias de `nota final`).
- `R_DTNASC_dt`, `C_DTNASC_dt`, `R_DTOBITO_dt`, `C_DTDIAG_dt` (datetime).
- `diff_ano` = `|ano(R_DTNASC) − ano(C_DTNASC)|` (float).
- `dtnasc_all_zero` (bool), `endereco_zero` (int8), `mae_missing` (int8).
- `nome_score_total` = `clip(0.5·qtd + 0.25·prim + 0.25·ult, 0, 1)`.
- `mae_score_total` = idem para NOMEMAE.
- `dtnasc_score_total` = `clip(mean(5 subscores DTNASC), 0, 1)`.
- `endereco_score_total` = `clip(mean(6 subscores ENDERECO), 0, 1)`.
- `municipio_score` = `clip(CODMUNRES local igual, 0, 1)`.
- `score_regras` = nº de regras disparadas entre
  {`NOME prim frag igual`, `DTNASC dt iguais`, `CODMUNRES local igual`} ≥ 1.0.
- MACD (se `macd_enabled`): `macd_nasc_diff_capped`, `macd_nasc_year_match`,
  `macd_nasc_month_match`, `macd_nasc_day_match`, `macd_nasc_partial_overlap`,
  `macd_nasc_close`, `macd_nasc_very_close`, `macd_nome_perf_x_date_far`,
  `macd_nome_perf_x_year_diff`.
- Requeridas (fail-fast `KeyError`): `COMPREC`, `REFREC`, `PASSO`, `PAR`.

## 2. Bandas (`bands.py`)
- `BandAssigner(definitions: tuple[BandDefinition, ...])` (frozen).
- `BandAssigner.from_config(cfg) -> BandAssigner`.
- `.assign(series: pd.Series) -> pd.Series` (dtype `string`); **não muta** entrada;
  coage numérico (`errors="coerce"`); fora de qualquer banda → `pd.NA`.
- `inclusive_max=True` ⇒ `s <= max`; `False` ⇒ `s < max`. Limite inferior sempre
  inclusivo (`s >= min`). Primeira definição que casa vence.

## 3. Calibração (`calibration.py`)
- `PlattModel(intercept: float, slope: float)` (frozen, **sem** clip/by-band).
- `fit_platt_from_df(df, *, target_col="TARGET", l2=1e-3, max_iter=100, tol=1e-10) -> PlattModel`.
  - Lê `nota final`/`nota_final` e `target_col`. Não muta df. Requer ≥10 linhas,
    `y ∈ {0,1}`.
  - **Math:** Newton-Raphson em regressão logística 1-D `p = σ(intercept + slope·nota)`;
    **L2 só no slope** (`+ 0.5·l2·slope²`); init pelo log-odds da base rate (Laplace);
    **determinístico** (sem RNG).
- `compute_p_cal(df, *, method, model=None, clip_min=1e-6, clip_max=0.999999) -> pd.Series`.
  - `method ∈ {"stub","platt"}`. `"stub"` = `clip(nota/10, clip_min, clip_max)`.
    `"platt"` = `predict_platt(...)` (exige `model`). **Retorna Series**, não grava no df.
- `predict_platt(nota, *, model, clip_min, clip_max)`: `σ(intercept + slope·nota)` e clip pós-sigmoide.

> ⚠️ **R-11 CONFIRMADO:** a config promete `method: anchor_platt`, `by_band: true`,
> `clip_min/clip_max`, `platt.l2/max_iter/tol`. O **código** implementa apenas
> **Platt global** (`fit_platt_from_df`), sem âncora e sem por-banda. O notebook
> ensina o que o código faz e trata a config como intenção/roadmap.

## 4. Guardrails (`guardrails.py`)
- `apply_guardrails(df, *, temporal_days=180, nota_always_match=10.0, nota_always_nonmatch=3.0, homonimia_min_nota=7.0, homonimia_year_gap=5.0) -> GuardrailOutput`.
- `GuardrailOutput(guardrail: pd.Series, reason: pd.Series)` (frozen, dtype `string`).
- `GuardrailDecision`: `"ALWAYS_MATCH"`, `"ALWAYS_NONMATCH"`, `"FORCE_REVIEW"`; sem guardrail ⇒ `pd.NA`.
- Regras (e `reason`):
  - `temporal_filter` → ALWAYS_NONMATCH: `R_DTOBITO_dt < C_DTDIAG_dt − temporal_days`.
  - `nota_final_low` → ALWAYS_NONMATCH: `nota_final < nota_always_nonmatch` (3.0, estrito).
  - `homonimia_risk` → FORCE_REVIEW: `nota ≥ 7` **E** `dtnasc_all_zero` **E** `diff_ano > 5` **E** `endereco_zero`.
  - `nota_final_high` → ALWAYS_MATCH: `nota ≥ 10` **E** `NOME qtd≥0.95` **E** `NOME prim≥1` **E** `NOME ult≥1` **E** `DTNASC dt iguais≥1` **E** `CODMUNRES local igual≥1`.

> ⚠️ **DIVERGÊNCIA (R-11):** o edge case do plano "nota ≥ 9 ⇒ ALWAYS_MATCH" e
> a config (`always_match: nota>=9`) **não** batem com o código, que exige
> `nota ≥ 10` + nome perfeito + data + município. O cenário "âncora high" do
> gerador deve satisfazer **todas** essas condições para disparar ALWAYS_MATCH.

## 5. Política de decisão (`gzcmd_v3_policy_engine.py`)
- `PolicyEngineV3(costs, llm_error_by_band, budget, min_auto_match=None, max_auto_nonmatch=None)`.
  - `Costs(false_positive, false_negative, llm_review)`.
  - `LLMError(e_fp, e_fn)`; `Budget(llm_max, llm_used=0)`.
- `.triage(df) -> pd.DataFrame`:
  - **Lê:** `p_cal`, `band`, `guardrail` (opcional).
  - **Adiciona:** `base_choice` {MATCH,NONMATCH}, `base_loss`, `loss_llm`, `evr`,
    `action` {MATCH,NONMATCH,LLM_REVIEW}, `review_requested` (bool).
  - ⚠️ **DIVERGÊNCIA (corrige plano §2.2):** `triage` faz `out = df.copy()` e
    **retorna a cópia** — **não muta** o df de entrada. Apenas `self.budget.llm_used`
    é incrementado (estado do engine). Mesmo assim, no notebook usaremos `df.copy()`
    por modo (boa prática + isolamento de `budget`).
  - Math: `loss_match=(1−p)·c_fp`; `loss_non=p·c_fn`;
    `loss_llm=c_llm+(1−p)·e_fp·c_fp+p·e_fn·c_fn`; `base_loss=min`; `evr=base_loss−loss_llm`.
  - Overrides: ALWAYS_MATCH→MATCH; ALWAYS_NONMATCH→NONMATCH; FORCE_REVIEW prioriza review.
  - Caps: `min_auto_match`, `max_auto_nonmatch` mandam para review se violados.

## 6. Runner (`runner.py`)
- `build_engine_from_config(cfg, *, mode, llm_used=0) -> PolicyEngineV3`.
- `run_v3(*, input_csv, config_path, mode, macd_enabled=True, llm_used=0, p_cal="fit_platt", platt_model_path=None, save_platt_model_path=None, ml_rf_model_path=None, save_ml_rf_model_path=None, ml_xgb_model_path=None, save_ml_xgb_model_path=None, classifier_config=None) -> tuple[pd.DataFrame, RunSummary]`.
  - `p_cal ∈ {"stub","fit_platt","load_platt","fit_ml_rf","load_ml_rf","fit_ml_xgb","load_ml_xgb"}`.
  - ⚠️ **R-10 CONFIRMADO:** `fit_platt`/`fit_ml_rf`/`fit_ml_xgb` ajustam **in-sample**
    (treina e pontua o MESMO df, sem split) — `runner.py:135-188`.
- DataFrame de saída: colunas do loader + `band`, `p_cal`, `guardrail`,
  `guardrail_reason`, `base_choice`, `base_loss`, `loss_llm`, `evr`, `action`,
  `review_requested`.
- `RunSummary(rows, llm_used, llm_max, actions: dict, guardrails: dict, review_requested, p_cal_method, p_cal_params: dict|None)` (frozen).

## 7. Avaliação held-out (`eval.py`) — Rota B
- `evaluate_v3_dataframe(df, *, cfg, modes, split_by, seeds, test_size, group_stratify, calibration, macd_enabled, guardrails_enabled=True) -> pd.DataFrame`.
  - **Held-out:** ajusta Platt no **treino**, prevê no **teste** (`eval.py:182-204`). ✓ correto.
  - Linhas por `(mode, seed)`. Colunas: `mode, beta, split_by, seed, test_size,
    group_stratify, macd_enabled, guardrails_enabled, calibration, n_train, n_test,
    pos_train, pos_test, llm_max, llm_used, review_requested, review_selected,
    auto_coverage`, `{auto_,oracle_,exp_}{tp,fp,fn,tn,precision,recall,f1,fbeta}`,
    `platt_intercept, platt_slope`.
  - Mapa de ação→predição: MATCH→1, NONMATCH→0, LLM_REVIEW→NA (fora das métricas `auto_`).
- `evaluate_v3_csv(*, input_csv, config_path, modes, split_by, seeds, test_size, group_stratify, calibration, macd_enabled, guardrails_enabled=True)`.
  - `split_by ∈ {"row","comprec","refrec"}`.
  - ⚠️ **A confirmar na Fase 3.2:** os valores aceitos do parâmetro `calibration`
    (`CalibrationMethod`). Hipótese: espelha `p_cal`. Verificar antes de usar.

## 8. Splitting (`splitting.py`)
- `SplitBy = Literal["row","comprec","refrec"]`.
- `SplitSpec(split_by="row", test_size=0.3, seed=42, group_stratify=True)` (frozen).
- `split_train_test_indices(df, y, *, spec) -> (train_idx, test_idx)` (arrays de índices).
  - `row`: split estratificado por linha. `comprec`/`refrec`: **group-aware** por
    `df["COMPREC"]`/`df["REFREC"]` — todas as linhas de um grupo vão juntas (sem vazamento).
  - RNG: `np.random.default_rng(spec.seed)`.

## 9. Métricas (`metrics.py`)
- `ConfusionCounts(tp, fp, fn, tn)` + props `n=tp+fp+fn+tn`, `support_pos=tp+fn`, `support_neg=fp+tn`.
- `confusion_counts(y_true, y_pred) -> ConfusionCounts` (arrays binários 0/1).
- `precision/recall(counts, zero_division=0.0)`; `fbeta(counts, beta, zero_division=0.0)`;
  `f1(counts, zero_division=0.0)`.
- `metrics_dict(counts, *, beta, prefix="") -> dict` com chaves
  `{prefix}{tp,fp,fn,tn,precision,recall,f1,fbeta}`.

---

## De-para de features (config × loader) — resumo R-04/R-11

| Config `expected_columns` | Produzida pelo loader? | Observação |
|---|---|---|
| `nota_final` | ✅ | alias de `nota final` |
| `step` | ❌ | loader usa `PASSO` |
| `nome_score_total`, `mae_score_total`, `dtnasc_score_total`, `endereco_score_total`, `municipio_score` | ✅ | agregados |
| `nome_x_dtnasc`, `nome_x_mae`, `mae_presente`, `dtnasc_perfeito`, `nome_perfeito` | ❌ | **não implementadas** (intenção/roadmap) |
| `mae_missing` | ✅ | |
| `nome_missing`, `dtnasc_missing`, `endereco_missing` | ❌ | só `mae_missing` existe |

**Conclusão (R-04):** `run_v3` ponta-a-ponta **não** depende das features ausentes
da config — Platt usa só `nota_final`; guardrails/triage usam as engenheiradas
acima. O gerador emitirá as RAW corretas → loader produz as engenheiradas reais.
A divergência é **documental** (config aspiracional), não quebra o pipeline.

---

## 10. API do objeto de configuração (`config.py`) — descoberto na 0.2

Confirmado por execução no teste de contrato:
- `load_config(path) -> GZCMDConfig`.
- Caminho do YAML empacotado: `importlib.resources.files("gzcmd_record_linkage") / "gzcmd_v3_config.yaml"`.
- `cfg.bands.definitions` → iterável de definições com `.name` (e min/max/inclusive_max).
- `cfg.calibration.clip_min`, `cfg.calibration.clip_max`.
- `cfg.calibration.platt.l2`, `.max_iter`, `.tol`.
- `build_engine_from_config(cfg, *, mode)` constrói o `PolicyEngineV3` com custos/budget do modo.

> Útil para as Fases 2.x/3.x: ler limites de clip e hiperparâmetros do Platt
> direto da config garante que o passo-a-passo manual use os MESMOS valores que
> `run_v3` (essencial para a reconciliação da Rota A, CA-G3).
