# Fila de revisão auto-calibrável (experimental, 0.7.0)

Status: **experimental, não avaliado na tese**. Desenho aprovado em 2026-11-15.
Mudança aditiva: `run_v3` e todos os defaults atuais ficam intactos.

## Objetivo

Revisar potenciais pares da saída ORL + CdR **sem padrão-ouro** (sem coluna
`PAR`). Um bandit de Thompson escolhe de qual faixa de posto consumir (cada
faixa é lida do topo para baixo, e um braço de exploração, fora do bandit,
revisa um par aleatório a cada 10 revisões); o revisor LLM
(dois agentes + árbitro, Haiku 5.5, prompt v4) julga par a par; ao fim de cada
rodada um portão (humano ou automático) fecha um holdout, um recalibrador
multicampo é reajustado e os pares restantes são reordenados.

## Módulos

| Arquivo | Conteúdo |
|---|---|
| `fellegi_sunter.py` (novo) | Discretização dos campos e Fellegi-Sunter por EM, sem rótulos |
| `field_recalibrator.py` (novo) | Logística MAP multicampo com prior, limites e pesos |
| `self_calibrating_queue.py` (novo) | Config, sessão passo a passo, eventos, `OracleReviewer` |
| `ranking.py` | Aditivo: `ThompsonQueue(input_kind="partial_orders", n_items=...)`, `to_state`/`from_state`, `arm_stats`, `exhausted`, `arm_of`, `mark_seen(item)` |
| `loader.py` | Aditivo: `LoadConfig(require_par: bool = True)` |
| `llm_dossier.py` | Aditivo: `DossierView(..., remove_fields=())` e a constante pública `DOSSIER_SOURCE_COLUMNS` |

## Entrada

DataFrame carregado por `load_comparador_csv(path, cfg=LoadConfig(require_par=False))`.
Com `require_par=False` o loader não exige `PAR` e, se `PAR` faltar, **não cria
`TARGET`** (hoje criaria `TARGET=0`, que se passaria por rótulo). Se `PAR`
existir, o comportamento é o atual. A sessão **nunca lê `PAR` nem `TARGET`**
(teste: embaralhar `PAR` não muda a sequência de eventos).

Colunas usadas pela sessão: `COMPREC`, `REFREC`, `PASSO`, `nota final`/`nota_final`
(só na 1ª rodada e na máscara de baixa confiança), as colunas do conjunto de
campos `field_set` (padrão `"composites"`: `nome_score_total`,
`mae_score_total`, `mae_missing`, `dtnasc_score_total`, `endereco_score_total`,
`municipio_score`; `"cdr_subscores"`, experimental: os 29 sub-escores do CdR e
`mae_missing`), opcionalmente `SCORE` e `auto_low_confidence`. As colunas são
resolvidas como o `llm_dossier` faz (tolerante ao sufixo `,Tipo`; `SCORE`
aceita `"SCORE,C,15,0"` e vírgula decimal, como `below_cutoff`). A impressão
digital do DataFrame cobre essas colunas **e todas as que o dossiê lê**
(`llm_dossier.DOSSIER_SOURCE_COLUMNS`, resolvidas com a mesma tolerância), pois
o dossiê é construído a partir da linha completa (`build_dossier` não lê `PAR`).
`remove_fields` rejeita `str` e caminhos repetidos e, na criação da sessão, os
caminhos são validados montando um `DossierView` da 1ª linha (erro claro antes de
qualquer revisão; `model_outputs.*` já é removido pela sessão). Se
`features.nota_final` não estiver em `remove_fields`, a criação da sessão emite
`UserWarning` (a nota fica visível ao LLM: protocolo diferente do aprovado).
Notas infinitas e `window` menor que `zero_match_reviews_needed(tau, delta)`
são rejeitadas (`ValueError`); `from_state` valida a impressão digital antes de
qualquer indexação.
`pair_id = make_pair_id(COMPREC, REFREC, PASSO)`.

## Campos e níveis (compartilhados por FS e recalibrador)

`field_set` (config) escolhe o conjunto: `"composites"` (padrão, a tabela
seguinte) ou `"cdr_subscores"` (experimental, seção **Campos: sub-escores do
CdR** abaixo; a simulação deu resultado pior com ele). `FieldLevels`, `FellegiSunterModel`, `design_matrix`,
`agreement_mask` e o prior do recalibrador funcionam para os dois, e a
equivalência exata FS ↔ logística vale nos dois.

| Campo | Coluna | Níveis (referência = L0, ver abaixo) |
|---|---|---|
| nome | `nome_score_total` | L0 `=0`; L1 `(0, 0,5]`; L2 `(0,5, 1)`; L3 `=1` |
| mae | `mae_score_total`, `mae_missing` | L0..L3 como acima; `missing` se `mae_missing==1` (prevalece) |
| dtnasc | `dtnasc_score_total` | L0..L3 |
| endereco | `endereco_score_total` | L0..L3 |
| municipio | `municipio_score` | L0 `=0`; L1 `(0, 1)`; L2 `=1` |

Igualdade com tolerância 1e-9. NaN → L0 (contado e reportado).

**Nível de referência.** Os níveis presentes de cada campo são determinados no
DataFrame inteiro. A referência é L0 se presente; senão `missing`, se presente;
senão o primeiro nível presente. Níveis que nunca ocorrem saem do suporte do EM
(sem massa de suavização, probabilidade 0 em m e u) e da matriz de desenho, e o
modelo guarda os níveis presentes e a referência (`to_dict`/`from_dict`). Sem
isso, no extrato da tese `mae_missing==1` coincide com `mae_score_total==0`, o
nível `mae:L0` fica vazio, as colunas de `mae` ficam colineares com o intercepto
e os μ do FS sofrem o deslocamento da suavização de célula vazia. Com a
referência real, o desenho dos compostos tem no máximo 15 indicadores +
intercepto (menos um por nível ausente), a restrição ≥ 0 vale para os níveis de concordância acima da
referência e `mae:missing`, quando não é a referência, fica livre. A
equivalência exata entre `predict_proba` e a forma logística é mantida.

### Campos: sub-escores do CdR (`field_set="cdr_subscores"`, experimental)

29 sub-escores: `NOME` e `NOMEMAE` × {`prim frag igual`, `ult frag igual`,
`qtd frag iguais`, `qtd frag raros`, `qtd frag comuns`, `qtd frag muito parec`,
`qtd frag abrev`}; `DTNASC` × {`dt iguais`, `dt ap 1digi`, `dt inv dia`,
`dt inv mes`, `dt inv ano`}; `CODMUNRES` × {`uf igual`, `uf prox`, `local igual`,
`local prox`}; `ENDERECO` × {`via igual`, `via prox`, `numero igual`,
`compl prox`, `texto prox`, `tokens jacc`}; mais `mae_missing` como campo
binário (níveis `L0`, `missing`; o nível `missing` é livre de sinal).

**Regra de níveis, sem rótulos**, calculada sobre o DataFrame inteiro e guardada
no modelo (`FellegiSunterModel.to_dict`, em `specs`): L0 = valor ≤ 0 ou NaN;
sobre os valores positivos, se houver ≤ 3 valores distintos, um nível por valor
(em ordem); senão, os tercis dos valores positivos (cortes guardados; níveis
vazios por empate saem do suporte). Os níveis acima de L0 são de concordância
(coeficiente ≥ 0 no recalibrador). A referência segue a regra de presença
acima.

Inspeção no extrato da tese (61 696 pares, só leitura): todos os sub-escores são
`float64` sem NaN, em [0, 1], **exceto** `NOME qtd frag comuns` e
`NOMEMAE qtd frag comuns`, que só têm valores em [−1, 0] (penalização por
fragmentos comuns; 10 e 11 valores distintos). Nesses dois a regra literal
(L0 = valor ≤ 0) os tornaria constantes. **Adaptação mínima:** quando uma coluna
não tem valor positivo mas tem negativos, a mesma regra vale sobre as
magnitudes, com níveis `P1..P3` (em vez de `L1..L3`) e **sem restrição de sinal**
(um fragmento comum a mais não é evidência de concordância: o sinal vem dos
dados e do recalibrador). `P<k>` e `missing` ficam fora da máscara de
concordância. Colunas sem nenhum valor positivo (por exemplo `DTNASC dt inv ano`,
100% zero) têm só L0 e não geram coluna de desenho.

**Limitação.** Os sub-escores do mesmo campo são fortemente correlacionados;
a independência condicional do FS é violada e os pesos do prior ficam
inflados (a mesma evidência é contada várias vezes). No extrato da tese o EM
sem rótulos converge para uma separação com prevalência ≈ 0,49, bem acima da
real: o prior FS é **inválido** nesse extrato. O recalibrador ajusta os pesos com
rótulos, e na simulação isso não bastou; por isso o padrão são os compostos.

## Fellegi-Sunter por EM (prior sem rótulos, sem nota)

Classe latente Y ∈ {0,1}, prevalência π, distribuições categóricas m_f (Y=1) e
u_f (Y=0), independência condicional. EM sobre a contagem de padrões únicos
(≤ 960 com os compostos). Inicialização determinística e agnóstica (m crescente nos níveis de
concordância, u decrescente; π=0,05), suavização de Dirichlet α=1, `tol=1e-8`
relativo do objetivo penalizado (log-verossimilhança + α·Σ log m, log u, o
log-prior de Dirichlet; a log-verossimilhança pura também é guardada),
`max_iter=500`. Identificação: se a classe 1
tiver menos concordância média que a 0, troca. Degenerado (π fora de
[1e-6, 1−1e-6] ou não finito) → `FellegiSunterError`.

Equivalência com a logística: intercepto = logit π + Σ_f log(m_f0/u_f0);
coeficiente do nível l = log(m_fl/u_fl) − log(m_f0/u_f0). Esses valores são as
**médias do prior** do recalibrador (o modelo-prior, `as_recalibration`, é
essas médias projetadas nos limites: coeficientes de concordância ≥ 0 e caixa de
confiança, o mesmo ponto inicial do ajuste). O FS é ajustado **uma vez**, na
inicialização. Se o EM falhar: médias 0 nos coeficientes e
intercepto = logit da média de p0, com evento.

## Calibração da 1ª rodada (única etapa com nota)

`p0 = predict_platt(nota, model=fit_anchor_platt(nota))`; se as âncoras forem
insuficientes, `predict_stub(nota)` e evento informando o fallback.

## Braços

**Faixas por posto.** No início de cada rodada, os pares **ainda não revisados**
são ordenados por `p_cur` decrescente (desempate por `pair_id`) e divididos
pelos quantis de posto `band_rank_edges=(0.01, 0.05, 0.20)`: topo 1%, 1–5%,
5–20% e o resto (o tamanho do corte é ⌈q·n⌉). Faixas vazias são descartadas.
Faixas por posto não dependem de a calibração da 1ª rodada funcionar: com o
pool restrito ao que está abaixo do corte da tese, o anchor-Platt falhou (A⁺ = 9
< 30 âncoras) e o stub nota/10 fazia da faixa de topo a zona cinzenta inteira (no
extrato inteiro A⁺ = 95, `apendice_gzcmd.tex:96`, e a calibração funciona).

**Cada braço de faixa é consumido do topo para baixo**, em ordem decrescente de
`p_cur` (desempate por `pair_id`). Bandit: `ThompsonQueue(partial_orders,
n_items=N, window=W, prior=(1,1), seed=derivada(seed, rodada))`, recriado a cada
rodada. Recompensa: 1 se MATCH; 0 se NONMATCH, UNSURE ou erro do LLM.

**Braço de exploração (fora do bandit).** Ordem aleatória (RNG derivado de
`(seed, rodada)`) sobre todos os pares não revisados no início da rodada. A cada
`explore_every=10` revisões do run (`0` desliga) a próxima revisão vem dele: é a
revisão de número `k` com `k` múltiplo de `explore_every`, contando todas as
revisões do run (calendário determinístico). O item sai por
`ThompsonQueue.mark_seen`, para que os braços de faixa o pulem. A exploração
**não entra na regra de parada nem na recompensa do bandit** (o `pair_presented`
diz `arm="explore"`, `band=-1`, e `final_verdict`/`review_error` trazem
`reward=null`); conta, sim, para `max_round_reviews`, `max_reviews` e para os
erros seguidos. Se a lista se esgota, a revisão vem do bandit. O motivo: a ordem
por probabilidade sozinha não diz nada sobre o que ficou abaixo do que foi
consumido, e os 5 compostos não carregam sinal abaixo do corte (na simulação, o
FS achou 0,9 dos 36 PAR=2 abaixo do corte, e a nota 12 em 100 revisões).

## Parada da rodada

Para cada braço k não esgotado, nas últimas `W` revisões do braço:
θ_k ~ Beta(1 + s_k, 1 + n_k − s_k). **A rodada para quando
max_k P(θ_k > τ) < δ** (cálculo exato, `scipy.stats.beta.sf`), sobre **todos**
os braços não esgotados. Outros motivos: todos os braços esgotados;
`max_round_reviews`; `max_reviews` do run; `max_consecutive_errors` erros
seguidos do LLM (`llm_errors`).

Revisões mínimas sem MATCH em **cada** braço não esgotado para a regra disparar:
n* = menor n com (1−τ)^(n+1) < δ (58 com os defaults; exato, sem arredondar
logaritmos).

## Portão

- **Humano** (`mode="human"`, padrão): o portão pausa a sessão com os itens
  candidatos da rodada, **em ordem de prioridade**:
  1. os MATCH do LLM da rodada, por `p_cur` decrescente (desempate por `pair_id`);
  2. os UNSURE do LLM (se `gate_include_unsure=True`), na mesma ordem;
  3. a auditoria: `nonmatch_audit_n=10` NONMATCH da rodada cujo par estava nos
     `audit_rank_fraction=0.05` superiores do posto de `p_cur` no início da
     rodada, sorteados com semente.

  **Orçamento humano.** Só os primeiros candidatos que cabem em
  `human_budget_per_round` (50; `None` = sem limite) e no que resta de
  `human_budget_total` (`None` = sem limite) vão ao portão. O excedente fica
  `human_decision_source="over_budget"`, sem rótulo e fora do holdout. Se o
  orçamento total se esgota, o run termina com motivo `human_budget`, depois do
  portão, do holdout e da recalibração da rodada corrente. Com orçamento
  apertado a auditoria é a primeira a ficar de fora (com os defaults, nas rodadas
  em que há mais de 40 MATCH + UNSURE ela não cabe).

  **Sem confirmação implícita.** `submit_human_decisions(decisions)` exige uma
  decisão para todo item do portão: `"MATCH"`, `"NONMATCH"` ou `"SKIP"`. Item
  faltando → `ValueError` (diz quantos e alguns `pair_id`); `"SKIP"` →
  `human_decision=None`, `human_decision_source="human_skip"`, fora do holdout.
  `run_self_calibrating`: o `human_callback` devolve decisões para todos os itens.

  **Auditoria e alarme.** A amostra é pequena demais para estimar a taxa de
  perda: o alarme `audit_match_found` dispara se pelo menos um auditado for par e
  seu payload diz que é verificação de sanidade, sem poder estatístico, não um
  estimador.
  `gate_opened` e `holdout_built` reportam nº de candidatos, nº no portão, nº
  `over_budget`, orçamento restante e as contagens por `label_source`. Com portão
  vazio (nenhum candidato, ou orçamento esgotado) `gate_opened` traz
  `awaiting=false` e a sessão segue direto ao holdout.
- **Automático** (`mode="auto"`, exige opt-in e emite `ExperimentalWarning`):
  sem pausa; regras do holdout abaixo.

Campo próprio: `llm_decision` (nunca sobrescrito), `human_decision ∈
{"MATCH","NONMATCH",None}`, `human_decision_source` (`human`, `human_skip`,
`over_budget` ou `None`), `label`, `label_source`. `llm_match_unverified` =
`llm_decision == "MATCH"` e `human_decision_source != "human"`.

## Holdout (acumulado de todas as rodadas)

| Situação | Rótulo |
|---|---|
| `human_decision` definido (MATCH, UNSURE ou auditoria) | o do humano (**positivo só com MATCH humano**) |
| Humano, `negative_labels="llm_nonmatch"` (padrão): NONMATCH do LLM não auditado | 0 (`llm_nonmatch`) |
| Humano, `negative_labels="verified_only"`: NONMATCH do LLM não auditado | excluído (`excluded_unverified_nonmatch`) |
| Humano: `human_skip` ou `over_budget` | excluído (`excluded_human_skip`, `excluded_over_budget`) |
| Humano: UNSURE sem decisão (fora do portão) ou MATCH do LLM sem decisão | excluído |
| Auto: MATCH por consenso fora da região de baixa confiança | 1 |
| Auto: MATCH pelo árbitro | excluído (`excluded_arbiter`) |
| Auto: NONMATCH fora da região de baixa confiança | 0 |
| Auto: qualquer veredito na região de baixa confiança | excluído (`excluded_low_conf`) |
| UNSURE / erro do LLM | excluído |

Opção desligada `auto_low_conf_policy="soft"`: na região de baixa confiança,
MATCH → y = `auto_low_conf_match_label` (0,16) e NONMATCH → 0.

**Região de baixa confiança** (só no modo automático), em ordem de precedência:
o argumento `low_confidence` (máscara bool) se informado; senão a coluna booleana
`auto_low_confidence`, se existir; senão `below_cutoff(df)` (corte da tese por PASSO
sobre `SCORE`) nas linhas cujo `PASSO` está nos cortes e cujo `SCORE` é
finito; nas demais linhas, faixas cinzentas do perfil v3.1 por nota, [5; 8). A
decisão é por linha (fonte `below_cutoff`, `grey_bands_nota` ou `mixed`) e o
evento `ordering` registra as contagens por fonte (`low_confidence_by_source`).
Máscara (argumento ou coluna) precisa ter dtype bool.

**Pesos**: `weighting="stratum_ipw"` (padrão), dois estratos por rodada
(`stratum_weights(..., max_ratio=weight_max_ratio)`):

- `r{r}:top`: as revisões do bandit na rodada; tamanho = nº dessas revisões, logo
  peso 1. O prefixo consumido de cada faixa é tratado como **censo** (π = 1);
- `r{r}:explore`: as revisões do braço de exploração; tamanho N = nº de pares
  não revisados no início da rodada, peso N / n_e. A exploração dá a **todos** os
  não revisados uma probabilidade de inclusão conhecida (n_e / N).

n_h conta todos os revisados do estrato (rotulados ou não). O teto
`weight_max_ratio=10`: w = min(w_h, R · min_h w_h) sobre todos os estratos
revisados, restritos ao holdout e normalizados para média 1. **O teto troca viés
por variância:** com N/n_e na casa de milhares, o peso da exploração fica
travado em R vezes o do topo, então a parte do universo não revisado é
subponderada (viés) em troca de uma variância e de um `effective_n` aceitáveis;
`weight_max_ratio` maior reduz o viés e aumenta a variância. `holdout_built`
reporta `effective_n` (Kish), `n_capped` e a soma de pesos por classe;
`holdout()` traz a coluna `stratum`. No modo automático, os pares da exploração
quase sempre caem na região de baixa confiança e saem do holdout
(`excluded_low_conf`); no humano entram (NONMATCH do LLM = 0). Opção `"none"`.

## Recalibrador

logit p̂ = β0 + Σ β_j x_j (x = indicadores dos níveis). Objetivo MAP:
−Σ w_i [ỹ_i log σ(η_i) + (1−ỹ_i) log(1−σ(η_i))] + ½ Σ_j ((β_j − μ_j)/s_j)²,
com μ = pesos do FS, alvos suavizados de Platt (ỹ+ = (N+ +1)/(N+ +2),
ỹ− = 1/(N− +2)). L-BFGS-B (scipy) com gradiente analítico.

| Parâmetro | Humano | Auto |
|---|---|---|
| sd do prior, coeficientes | 1,0 | 0,5 |
| sd do prior, intercepto | 2,0 | 1,0 |
| Limite de deslocamento (caixa em torno de μ) | — | 1,0 coef. / 2,0 intercepto |

Restrições: coeficientes dos níveis de concordância ≥ 0; `mae missing` e
intercepto livres. **A nota não entra.**

Fallback: menos de `min_labeled=20` rotulados ou menos de `min_per_class=3`
de alguma classe → mantém o modelo vigente (`skipped_insufficient`); não
convergência ou valor não finito → mantém o modelo vigente (`fallback_prior`).
Na anomalia o modelo vigente é preservado com o seu `status` e o evento
`recalibration` marca `reverted=true`.
Nota: o fallback para `LogitPlatt` do desenho inicial foi retirado porque
recalibraria sobre a nota, contrariando a decisão "nota só na 1ª rodada".

## Salvaguardas do modo automático

1. Opt-in explícito + `ExperimentalWarning`.
2. Só MATCH por consenso conta como positivo.
3. Região de baixa confiança fora do holdout (simétrico, sem viés de só-negativos).
   **Guarda de região dominante** (`auto_max_low_conf_fraction=0.5`, só no auto com
   `auto_low_conf_policy="exclude"`): se (# pares na região) / N passar do limite,
   o aprendizado fica desligado no run inteiro. O holdout continua sendo montado
   e emitido; a recalibração sai com `status="skipped_low_conf_dominant"`
   (`model_status` guarda o status do modelo vigente, com fração e limite no
   payload) e o modelo vigente permanece o prior FS, que ordena da rodada 2 em
   diante. `ordering` traz `low_conf_fraction`, `auto_max_low_conf_fraction` e
   `auto_learning_enabled`; um único `alarm` com `reason="low_confidence_dominant"`
   sai na rodada 1, entre `holdout_built` e `recalibration`. Com `"soft"` ou no
   modo humano a guarda não se aplica; `1.0` a desliga. O estado é derivado da
   máscara persistida e da configuração, então a retomada é idêntica.
4. Prior apertado + caixa de deslocamento.
5. Alarme de anomalia: numa faixa, MATCH da rodada > `anomaly_ratio=3.0` × o
   esperado por `p_cur` e pelo menos `anomaly_min_matches=5` → evento `alarm`,
   volta ao modelo anterior e encerra o run (`auto_anomaly`).
6. Saída marcada `llm_match_unverified` (todo MATCH do LLM sem decisão humana): o
   holdout calibra e ordena, nunca aceita vínculo.

## Término do run

Após o portão e a recalibração de cada rodada: `max_reviews` (2000);
`human_budget` (`human_budget_total` esgotado); fila esgotada; `max_rounds` (5); rodada estéril (positivos novos após o portão <
`min_new_positives=1`); alarme de anomalia; `max_consecutive_errors=3` erros
seguidos do LLM.

## Eventos (um por `step()`)

`SessionEvent(seq, round, kind, payload)`, serializável em JSON. Tipos:
`ordering`, `pair_presented` (sem chamar o LLM), `agent_verdict` (A, B, árbitro),
`final_verdict` (com estatística de parada), `review_error`, `round_stopped`,
`gate_opened` (`awaiting` true quando há itens no portão humano; false no auto e com
portão humano vazio), `human_decision` (um por item, inclusive `human_skip`), `holdout_built`, `alarm`, `recalibration`, `reordering`, `run_finished`.
A chamada ao LLM acontece no passo seguinte a `pair_presented`; os vereditos
ficam num buffer e saem um por passo. Com `awaiting_human`, `step()` levanta erro
até `submit_human_decisions(...)`.

## Máquina de estados

Fases: `init` → `reviewing` → `gate` → (`awaiting_human`) → `holdout` →
`recalibrate` → `terminate_check` → `reviewing` (próxima rodada) ou `finished`.

- `init`: calcula `p0`, ajusta o FS, monta a máscara de baixa confiança, emite
  `ordering` e prepara a rodada 1.
- `reviewing`: se houver evento no buffer, devolve-o. Senão verifica a parada
  (regra da posterior, `max_round_reviews`, `max_reviews`, erros seguidos); se
  parar, emite `round_stopped`. Senão escolhe o próximo par (exploração, se
  o calendário mandar; senão o topo do braço sorteado pelo bandit) e emite
  `pair_presented`. No passo seguinte chama o revisor, atualiza o bandit (menos
  numa revisão de exploração) e
  enfileira `agent_verdict` (A, B, árbitro se houver) e `final_verdict`, ou
  `review_error`.
- Ordem ao fim da rodada: `round_stopped` → `gate_opened` → `human_decision`×k →
  `holdout_built` → [`alarm`] → `recalibration` → `reordering` | `run_finished`.
  Mesmo quando o run termina por orçamento no meio de uma rodada, portão,
  holdout e recalibração acontecem antes de `run_finished`.
- `p_cur`: rodada 1 = `p0` (nota). Da rodada 2 em diante = modelo vigente sobre
  os campos; o modelo vigente começa como o prior FS (`status="prior"`), de modo
  que a nota nunca volta a ordenar, mesmo se a 1ª recalibração for pulada.
- Pesos: ver **Pesos** acima (estratos `top` e `explore` por rodada);
  `holdout()` devolve o holdout do último `holdout_built` (snapshot persistido
  no estado).
- Anomalia (auto): por grupo da rodada (faixa de posto ou exploração, `band=-1`),
  MATCH do LLM ≥ `anomaly_min_matches` e > `anomaly_ratio` × Σ `p_cur` dos
  revisados no grupo.

## Estado e determinismo

`to_state()` → dict JSON (versão, config, fase, rodada, contador de eventos,
buffer, registros, estado do bandit, modelo vigente, modelo FS, itens do portão,
impressão digital do DataFrame, tamanhos do conjunto de exploração por rodada,
itens do portão já usados do orçamento humano e estatísticas do portão,
estado do bandit com os itens `marked`). `from_state(state, df, reviewer)` valida
a impressão digital. `p_cur`, o posto de cada par e a ordem aleatória da
exploração são recalculados do DataFrame, do modelo vigente e do conjunto
revisado no início da rodada. Sem relógio nos eventos.
Uma semente controla tudo; desempates por `pair_id`.

## Revisor

Padrão de produção: `AnthropicReviewer(model="claude-haiku-5-5",
prompt_version="v4")` sobre `DossierView(build_dossier(row),
remove_model_outputs=True, remove_fields=("features.nota_final",))`. O prompt v4
trata `nota_final` como "se presente". As taxas de `error_rates_by_band_v4.yaml`
**não** valem para esta variante (v4 + sem_mo + sem nota).
`OracleReviewer(truth, fp_rate, fn_rate, unsure_rate, disagreement_rate, seed)`
para testes e simulações, sem rede.

## Avaliação com LLM real

Harness: `notebooks/llm_real_eval.py` (`pilot`, `calibrate`, `report`, `queue-run`,
`baseline`, `export-fixture`, `rerank-test`; `FixtureReviewer` reaplica as decisões
gravadas sem rede). Só ele lê a chave, de `ANTHROPIC_API_KEY_ALT` no `.env`
(ignorado pelo git); as decisões reais ficam num fixture local em
`data/llm_fixtures/` (ignorado pelo git). `SelfCalibratingSession.peek_upcoming`
serve à pré-busca paralela. Revisor: Haiku 5.5, prompt v4, sem `model_outputs` e
sem a nota. Relatórios agregados, sem dado por par:

1. [`docs/llm_real_haiku_sem_nota.md`](../llm_real_haiku_sem_nota.md): 717 pares
   (os 247 verdadeiros e 470 não pares). FP 0,4% [IC 95% 0,1–1,5%], FN 18,2%
   [13,9–23,5%]; abaixo do corte da tese, FN 44,4% [29,5–60,4%] (16 de 36). Vale
   para a variante sem nota; não substitui as taxas da tese.
2. [`docs/llm_real_fila_ponta_a_ponta.md`](../llm_real_fila_ponta_a_ponta.md):
   configuração padrão, humano simulado perfeito. Com 127 decisões humanas
   (`llm_nonmatch`; 128 em `verified_only`), a fila verificou 96 verdadeiros, 4
   deles dos 36 abaixo do corte; a base H1 (nota + LLM, humano confirma MATCH)
   verificou 10 abaixo do corte e H0 (humano sozinho na ordem da nota), 9. O run
   parou por `sterile_round` (3 rodadas, 1.443 revisões do LLM). A fila não
   superou as bases.
3. [`docs/llm_real_desempate_zona_cinza.md`](../llm_real_desempate_zona_cinza.md):
   o LLM reordena os pares abaixo do corte (o humano decide todo par que revisa).
   Ganho só com ~400 decisões humanas: +5 e +6 verdadeiros de 36 nas janelas de
   2.000 e 3.000 (L1 e L3). O IC por bootstrap exclui zero em 4 de 48
   combinações, só nesses casos.

Limitações: **uma realização** do LLM (temperatura padrão; o veredito fica
congelado no fixture), humano simulado **perfeito**, **36 positivos** abaixo do
corte (diferenças de 1–3 pares são ruído), taxas medidas numa amostra
estratificada e sem a nota, e o desempate não mede tempo por decisão. A simulação
com as taxas da vigilância (`docs/sim_fila_autocalibravel.md`) concluiu que a fila
com portão revisto não encurta o trabalho humano em relação à ordem da nota.

## Limitações conhecidas

O método é **experimental** e não recomendado para uso em produção. Os achados
abaixo foram identificados em revisão e **não foram corrigidos** (fora do escopo
desta versão; a política A' e a lógica de anomalia ficam como estão):

1. **Recompensa, parada e anomalia usam o MATCH bruto do LLM**, não o rótulo
   verificado: um LLM com falsos positivos infla as recompensas, atrasa a parada
   e pode disparar (ou deixar de disparar) a anomalia.
2. **A parada fica inerte com FP ≥ τ.** Se a taxa de MATCH do LLM em não-pares de
   um braço for pelo menos τ, P(θ > τ) nunca cai abaixo de δ e a rodada só termina
   por `max_round_reviews`, `max_reviews`, esgotamento ou erros seguidos.
3. **O bandit age sobre segmentos contíguos do ranking**, lidos do topo para baixo:
   a taxa de MATCH de um braço decresce por construção com a posição, e o modelo
   Beta-Bernoulli (amostras independentes) é só uma aproximação.
4. **A exploração rende pouco**: cerca de 0,06 par achado por exploração por run
   simulado (os PAR = 2 abaixo do corte são 36 em 61.485), então ela serve para
   dar probabilidade de inclusão conhecida ao universo não revisado, não para
   achar pares.
5. **Os pesos IPW contam a população por rodada** (os não revisados no início da
   rodada, não o universo) **e o teto de razão anula a correção**: com N/n_e na
   casa de milhares e `weight_max_ratio=10`, o estrato `explore` fica travado em
   10 vezes o `top`, muito abaixo do peso que corrigiria o viés de seleção.
6. **O prior Fellegi-Sunter pode ter prevalência implausível**: com os sub-escores
   do CdR o EM sem rótulos dá ≈ 0,49 no extrato da tese; com os compostos o
   prior é fraco abaixo do corte (a simulação achou 0,9 dos 36 PAR = 2).
7. **O recalibrador tem poucos eventos por parâmetro**: dezenas de positivos
   verificados para dezenas de coeficientes (46 colunas de desenho com os
   sub-escores), regularizado pelo prior, mas sem dado suficiente para corrigi-lo.
8. **Sem vantagem na simulação**: contra a ordenação pela nota, a fila não achou
   mais pares abaixo do corte com o mesmo orçamento (ver
   `docs/sim_fila_autocalibravel.md`).
