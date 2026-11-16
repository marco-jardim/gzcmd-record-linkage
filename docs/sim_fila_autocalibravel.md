# Simulação offline da fila auto-calibrável sobre o extrato da tese

> **Aviso.** Simulação com revisor **sintético**, calibrado nas taxas de erro por faixa da vigilância, medidas com a **nota visível** ao LLM; a variante simulada aqui **oculta a nota**. **Não é uma avaliação com LLM real.** O método é **experimental**, não existe na tese nem na 0.6.1, e estes números não o validam: mostram apenas o que a mecânica da fila faz sob as suposições declaradas.

Esta é a **versão 3** (portão humano com decisão explícita para todo item, orçamento humano por rodada, `negative_labels`, campos `composites`). A versão 2 está em [sim_fila_autocalibravel_v2.md](sim_fila_autocalibravel_v2.md) e a versão 1 em [sim_fila_autocalibravel_v1.md](sim_fila_autocalibravel_v1.md), com os agregados em `sim_fila_autocalibravel_resultados_v2.csv` e `..._v1.csv`.

## Método e suposições

- Dados: extrato `COMPARADORSEMIDENT.csv`, lido no lugar com `load_comparador_csv(..., LoadConfig(require_par=False))`. `PAR` fica numa série separada, usada só como verdade (`PAR ∈ {1, 2}`) do revisor e do humano simulados; a sessão recebe o DataFrame do extrato **sem `PAR` e sem `TARGET`**.
- Partição conferida com `below_cutoff(df, expected=THESIS_PARTITION)`: 211 acima do corte, 61485 abaixo, 36 com `PAR = 2` abaixo (igual à da tese: sim).
- Fila: a cada rodada os pares não revisados são ordenados por `p_cur` e divididos por posto em `band_rank_edges=(0,01; 0,05; 0,20)`; cada faixa é consumida do topo para baixo; um braço de exploração aleatório, fora do bandit, ocupa uma a cada `explore_every=10` revisões e não entra na parada nem na recompensa. Prior (Fellegi-Sunter por EM, sem rótulos) e recalibrador sobre `field_set=composites` (5 compostos).
- Portão humano (v3): o humano simulado é **perfeito** e decide **todos** os itens do portão pelo `PAR` (`MATCH` ou `NONMATCH`; não há confirmação implícita). Itens, por prioridade: MATCH do LLM (por `p_cur`), UNSURE e uma auditoria de `nonmatch_audit_n=10` NONMATCH do topo (5% do posto). O orçamento `human_budget_per_round` limita as decisões por rodada; o excedente é `over_budget`, sem rótulo. `llm_match_unverified` = MATCH do LLM sem decisão humana. `negative_labels` define os rótulos negativos do recalibrador: `llm_nonmatch` (NONMATCH do LLM vale 0) ou `verified_only` (só negativos verificados pelo humano).
- Revisor: `BandedOracle` roteia cada par, pela nota, para um `OracleReviewer` da faixa do perfil v3.1 (`low` [0, 5), `grey_low` [5, 6), `grey_mid` [6, 7), `grey_high` [7, 8), `near_high` [8, 9), `high` [9, ∞)). O `OracleReviewer` é determinístico por semente e identificador do par: o mesmo par recebe o mesmo veredito na fila e nas bases (comparação pareada).
- Cenários: `perfect` (taxas 0, sem desacordo) e `thesis_bands` (taxas abaixo, com `disagreement_rate = 0,10`, **suposição**: só afeta o protocolo, não o veredito).
- Taxas por faixa (vigilância, contagens agregadas de `notebooks/nb_fila.py`, com Laplace: fp = (FP+1)/(não pares+2), fn = (FN+1)/(pares+2), unsure = (UNSURE+1)/(decisões+2)). A faixa `low` não tem dados e usa as taxas de `grey_low` (**suposição**):

| faixa | FP | FN | UNSURE |
|---|---|---|---|
| grey_low | 8,4% | 75,0% | 8,2% |
| grey_mid | 2,7% | 6,7% | 6,0% |
| grey_high | 2,9% | 11,1% | 4,7% |
| near_high | 4,5% | 30,6% | 8,0% |
| high | 50,0% | 25,0% | 16,7% |
| low | 8,4% | 75,0% | 8,2% |

- Configuração: `SelfCalibratingConfig()` padrão (`tau=0,05`, `delta=0,05`, `window=100`, `max_round_reviews=500`, `max_rounds=5`, `field_set=composites`), `max_reviews=2000`. Com `human_budget_per_round=50`, a fila produz no máximo 250 decisões humanas por run. Sensibilidade: humano padrão × `below_cut` × `thesis_bands` com `tau=0,01` e `window=300`.
- Grade: modo humano × pool ∈ {`all`, `below_cut`} × revisor ∈ {`perfect`, `thesis_bands`} × `negative_labels` ∈ {`llm_nonmatch`, `verified_only`} × `human_budget_per_round` ∈ {25, 50, 100} × 10 sementes (1, 2, 3, 4, 5, 6, 7, 8, 9, 10); mais um bloco automático (pool × revisor × sementes), só de continuidade. Valores: média (mínimo–máximo) entre sementes.
- **Viés de sobrevivência.** `[n=k]` marca valores em que só k das sementes da configuração atingiram a marca (ou o orçamento) e entraram na média. Esses valores têm viés de sobrevivência: as sementes que chegam mais longe não são as mesmas que param antes (por exemplo, rodadas estéreis) e a média parcial não é comparável com a de uma marca alcançada por todas. “—” = nenhuma semente atingiu.
- **Limitação conhecida.** A recompensa do bandit, a regra de parada da rodada e a detecção de anomalia usam o **MATCH bruto do LLM**, não o MATCH verificado pelo humano, mesmo no modo humano: FP do LLM contam como recompensa.
- Linhas de base. A orçamento de LLM igual (continuidade com a v2): (a) nota decrescente; (b1) Fellegi-Sunter sobre os 5 compostos; (c) ordem aleatória, com a mesma realização do revisor, mesma semente e mesmo orçamento (revisões do run e marcas 100, 250, 500, 1000, 2000). **De esforço humano** (orçamento B ∈ {50, 100, 200, 400} decisões humanas; humano perfeito; mesma realização do revisor): **H0** o humano revisa sozinho os pares na ordem da nota (1 decisão por par); **H1** a ordem da nota, o LLM tria e o humano decide só os MATCH do LLM (1 decisão por MATCH; o LLM revisa na ordem até gerar B MATCH); **H2** é a fila com o portão revisto. “Verificado” = confirmado por decisão humana. H0 e H1 são calculados até 400 decisões; a fila, até o fim do run (no máximo 250 decisões com 50 por rodada).
- “Encontrado” (tabelas de LLM) = MATCH do LLM ∧ verdade. “FP” = MATCH do LLM sem verdade. “Abaixo do corte” = par com `PAR = 2` abaixo do corte da tese (36 no extrato).

## Esforço humano

Pergunta: com o humano revisando a fatia que o LLM marcou como MATCH (detectando FP) e a recalibração recebendo só rótulos verificados, a ferramenta encurta o trabalho humano? Comparação a **orçamento humano igual** B (decisões humanas), configuração padrão da fila (`llm_nonmatch`, 50 por rodada).

**Pares verdadeiros verificados**

| pool / revisor | método | B=50 | B=100 | B=200 | B=400 | run |
|---|---|---|---|---|---|---|
| all / perfect | H0 humano sozinho (ordem da nota) | 49,0 | 98,0 | 127,0 | 140,0 | 140,0 |
| all / perfect | H1 nota + LLM tria + humano confirma MATCH | 50,0 | 100,0 | 200,0 | — | 247,0 |
| all / perfect | H2 fila com portão revisto | 50,0 | 100,0 | — | — | 104,5 (103,0–105,0) |
| all / thesis_bands | H0 humano sozinho (ordem da nota) | 49,0 | 98,0 | 127,0 | 140,0 | 140,0 |
| all / thesis_bands | H1 nota + LLM tria + humano confirma MATCH | 48,9 (48,0–50,0) | 91,4 (83,0–95,0) | 133,6 (129,0–139,0) | 165,2 (159,0–170,0) | 165,2 (159,0–170,0) |
| all / thesis_bands | H2 fila com portão revisto | 48,9 (48,0–50,0) | 98,6 (97,0–100,0) | 103,5 (100,0–106,0) | — | 103,5 (100,0–106,0) |
| below_cut / perfect | H0 humano sozinho (ordem da nota) | 10,0 | 12,0 | 13,0 | 13,0 | 13,0 |
| below_cut / perfect | H1 nota + LLM tria + humano confirma MATCH | — | — | — | — | 36,0 |
| below_cut / perfect | H2 fila com portão revisto | 16,0 [n=1] | — | — | — | 15,0 (14,0–16,0) |
| below_cut / thesis_bands | H0 humano sozinho (ordem da nota) | 10,0 | 12,0 | 13,0 | 13,0 | 13,0 |
| below_cut / thesis_bands | H1 nota + LLM tria + humano confirma MATCH | 10,9 (8,0–14,0) | 13,2 (12,0–16,0) | 17,6 (16,0–20,0) | 18,6 (17,0–21,0) | 18,6 (17,0–21,0) |
| below_cut / thesis_bands | H2 fila com portão revisto | 8,3 (5,0–10,0) | 9,5 (5,0–12,0) | 13,0 [n=1] | — | 9,6 (5,0–13,0) |

**Pares abaixo do corte verificados (de 36)**

| pool / revisor | método | B=50 | B=100 | B=200 | B=400 | run |
|---|---|---|---|---|---|---|
| all / perfect | H0 humano sozinho (ordem da nota) | 5,0 | 8,7 (8,0–9,0) | 12,0 | 13,0 | 13,0 |
| all / perfect | H1 nota + LLM tria + humano confirma MATCH | 5,0 | 9,0 | 20,3 (20,0–21,0) | — | 36,0 |
| all / perfect | H2 fila com portão revisto | 5,0 | 5,0 | — | — | 7,6 (6,0–8,0) |
| all / thesis_bands | H0 humano sozinho (ordem da nota) | 5,0 | 8,7 (8,0–9,0) | 12,0 | 13,0 | 13,0 |
| all / thesis_bands | H1 nota + LLM tria + humano confirma MATCH | 2,1 (1,0–4,0) | 7,1 (5,0–10,0) | 12,7 (11,0–15,0) | 18,3 (17,0–20,0) | 18,3 (17,0–20,0) |
| all / thesis_bands | H2 fila com portão revisto | 2,1 (1,0–4,0) | 2,1 (1,0–4,0) | 4,1 (3,0–6,0) | — | 4,1 (3,0–6,0) |
| below_cut / perfect | H0 humano sozinho (ordem da nota) | 10,0 | 12,0 | 13,0 | 13,0 | 13,0 |
| below_cut / perfect | H1 nota + LLM tria + humano confirma MATCH | — | — | — | — | 36,0 |
| below_cut / perfect | H2 fila com portão revisto | 16,0 [n=1] | — | — | — | 15,0 (14,0–16,0) |
| below_cut / thesis_bands | H0 humano sozinho (ordem da nota) | 10,0 | 12,0 | 13,0 | 13,0 | 13,0 |
| below_cut / thesis_bands | H1 nota + LLM tria + humano confirma MATCH | 10,9 (8,0–14,0) | 13,2 (12,0–16,0) | 17,6 (16,0–20,0) | 18,6 (17,0–21,0) | 18,6 (17,0–21,0) |
| below_cut / thesis_bands | H2 fila com portão revisto | 8,3 (5,0–10,0) | 9,5 (5,0–12,0) | 13,0 [n=1] | — | 9,6 (5,0–13,0) |

**Decisões humanas por par verificado**

| pool / revisor | método | B=50 | B=100 | B=200 | B=400 | run |
|---|---|---|---|---|---|---|
| all / perfect | H0 humano sozinho (ordem da nota) | 1,0 | 1,0 | 1,6 | 2,9 | 2,9 |
| all / perfect | H1 nota + LLM tria + humano confirma MATCH | 1,0 | 1,0 | 1,0 | — | 1,0 |
| all / perfect | H2 fila com portão revisto | 1,0 | 1,0 | — | — | 1,3 (1,3–1,3) |
| all / thesis_bands | H0 humano sozinho (ordem da nota) | 1,0 | 1,0 | 1,6 | 2,9 | 2,9 |
| all / thesis_bands | H1 nota + LLM tria + humano confirma MATCH | 1,0 (1,0–1,0) | 1,1 (1,1–1,2) | 1,5 (1,4–1,6) | 2,4 (2,4–2,5) | 2,4 (2,4–2,5) |
| all / thesis_bands | H2 fila com portão revisto | 1,0 (1,0–1,0) | 1,0 (1,0–1,0) | 1,9 (1,9–2,0) | — | 1,9 (1,9–2,0) |
| below_cut / perfect | H0 humano sozinho (ordem da nota) | 5,0 | 8,3 | 15,4 | 30,8 | 30,8 |
| below_cut / perfect | H1 nota + LLM tria + humano confirma MATCH | — | — | — | — | 1,0 |
| below_cut / perfect | H2 fila com portão revisto | 3,1 [n=1] | — | — | — | 3,0 (2,4–3,5) |
| below_cut / thesis_bands | H0 humano sozinho (ordem da nota) | 5,0 | 8,3 | 15,4 | 30,8 | 30,8 |
| below_cut / thesis_bands | H1 nota + LLM tria + humano confirma MATCH | 4,7 (3,6–6,2) | 7,7 (6,2–8,3) | 11,4 (10,0–12,5) | 21,6 (19,0–23,5) | 21,6 (19,0–23,5) |
| below_cut / thesis_bands | H2 fila com portão revisto | 6,3 (5,0–10,0) | 11,3 (8,3–20,0) | 15,4 [n=1] | — | 14,4 (10,0–20,0) |

**Revisões de LLM consumidas**

| pool / revisor | método | B=50 | B=100 | B=200 | B=400 | run |
|---|---|---|---|---|---|---|
| all / perfect | H0 humano sozinho (ordem da nota) | 0 | 0 | 0 | 0 | 0 |
| all / perfect | H1 nota + LLM tria + humano confirma MATCH | 51 | 102 | 3518 (3513–3529) | — | 29193 (28947–29646) |
| all / perfect | H2 fila com portão revisto | 500 | 1000 | — | — | 1986 (1927–2000) |
| all / thesis_bands | H0 humano sozinho (ordem da nota) | 0 | 0 | 0 | 0 | 0 |
| all / thesis_bands | H1 nota + LLM tria + humano confirma MATCH | 79 (71–89) | 413 (272–541) | 2569 (2067–2966) | 8082 (7922–8503) | 8082 (7922–8503) |
| all / thesis_bands | H2 fila com portão revisto | 500 | 1000 | 2000 | — | 2000 |
| below_cut / perfect | H0 humano sozinho (ordem da nota) | 0 | 0 | 0 | 0 | 0 |
| below_cut / perfect | H1 nota + LLM tria + humano confirma MATCH | — | — | — | — | 24284 (24002–24523) |
| below_cut / perfect | H2 fila com portão revisto | 1294 [n=1] | — | — | — | 1026 (701–1294) |
| below_cut / thesis_bands | H0 humano sozinho (ordem da nota) | 0 | 0 | 0 | 0 | 0 |
| below_cut / thesis_bands | H1 nota + LLM tria + humano confirma MATCH | 1414 (1104–1548) | 3256 (2829–3699) | 7084 (6740–7531) | 9772 (9527–10017) | 9772 (9527–10017) |
| below_cut / thesis_bands | H2 fila com portão revisto | 500 | 1000 | 2000 [n=1] | — | 1350 (1000–2000) |

**Decisões humanas na sequência inteira (coluna run)**

| pool / revisor | método | B=50 | B=100 | B=200 | B=400 | run |
|---|---|---|---|---|---|---|
| all / perfect | H0 humano sozinho (ordem da nota) | 50 | 100 | 200 | 400 | 400 |
| all / perfect | H1 nota + LLM tria + humano confirma MATCH | 50 | 100 | 200 | — | 247 |
| all / perfect | H2 fila com portão revisto | 50 | 100 | — | — | 134 (133–135) |
| all / thesis_bands | H0 humano sozinho (ordem da nota) | 50 | 100 | 200 | 400 | 400 |
| all / thesis_bands | H1 nota + LLM tria + humano confirma MATCH | 50 | 100 | 200 | 400 | 400 |
| all / thesis_bands | H2 fila com portão revisto | 50 | 100 | 200 | — | 200 |
| below_cut / perfect | H0 humano sozinho (ordem da nota) | 50 | 100 | 200 | 400 | 400 |
| below_cut / perfect | H1 nota + LLM tria + humano confirma MATCH | — | — | — | — | 36 |
| below_cut / perfect | H2 fila com portão revisto | 50 [n=1] | — | — | — | 45 (34–56) |
| below_cut / thesis_bands | H0 humano sozinho (ordem da nota) | 50 | 100 | 200 | 400 | 400 |
| below_cut / thesis_bands | H1 nota + LLM tria + humano confirma MATCH | 50 | 100 | 200 | 400 | 400 |
| below_cut / thesis_bands | H2 fila com portão revisto | 50 | 100 | 200 [n=1] | — | 135 (100–200) |

`run` = a sequência inteira do método (H0/H1: até 400 decisões ou até acabarem os MATCH do LLM; fila: até o fim do run). Não é orçamento igual: serve para ver onde cada método termina. Colunas B com `—`: o método não chegou a B decisões; `[n=k]`: só k sementes chegaram (viés de sobrevivência).

**Decisões humanas até verificar k pares abaixo do corte**

Média (mínimo–máximo) das sementes que atingiram k, e entre colchetes a fração de sementes que atingiu. H0/H1: dentro de 400 decisões; fila: dentro do run.

| pool / revisor | método | k=5 | k=10 |
|---|---|---|---|
| all / perfect | H0 humano sozinho (ordem da nota) | 43 (43–44) [1,00] | 129 (129–130) [1,00] |
| all / perfect | H1 nota + LLM tria + humano confirma MATCH | 43 (43–43) [1,00] | 116 (115–116) [1,00] |
| all / perfect | H2 fila com portão revisto | 43 (43–43) [1,00] | — (0) |
| all / thesis_bands | H0 humano sozinho (ordem da nota) | 43 (43–44) [1,00] | 129 (129–130) [1,00] |
| all / thesis_bands | H1 nota + LLM tria + humano confirma MATCH | 74 (58–82) [1,00] | 124 (100–166) [1,00] |
| all / thesis_bands | H2 fila com portão revisto | 144 (127–160) [0,20] | — (0) |
| below_cut / perfect | H0 humano sozinho (ordem da nota) | 6 (5–6) [1,00] | 24 (23–24) [1,00] |
| below_cut / perfect | H1 nota + LLM tria + humano confirma MATCH | 5 (5–5) [1,00] | 10 (10–10) [1,00] |
| below_cut / perfect | H2 fila com portão revisto | 5 (5–5) [1,00] | 10 (10–10) [1,00] |
| below_cut / thesis_bands | H0 humano sozinho (ordem da nota) | 6 (5–6) [1,00] | 24 (23–24) [1,00] |
| below_cut / thesis_bands | H1 nota + LLM tria + humano confirma MATCH | 7 (5–9) [1,00] | 30 (20–55) [1,00] |
| below_cut / thesis_bands | H2 fila com portão revisto | 7 (5–9) [1,00] | 45 (30–54) [0,70] |

### Efeito de `negative_labels` e do orçamento por rodada (H2)

Fila no modo humano. “Verificados” = pares confirmados por decisão humana (verdade); “FP detectados” = MATCH do LLM que o humano rejeitou; “FP não verificados” = MATCH do LLM sem verdade e sem decisão humana (`over_budget`).

| configuração (pool / modo / revisor / negative_labels / por rodada) | revisões LLM | rodadas | decisões humanas | verificados | verificados abaixo do corte (de 36) | decisões por verificado | FP detectados | over_budget | llm_match_unverified | FP não verificados |
|---|---|---|---|---|---|---|---|---|---|---|
| all / human / perfect / llm_nonmatch / 25 | 1766,0 (1283,0–1996,0) | 4,4 (3,0–5,0) | 76,6 (60,0–84,0) | 52,6 (50,0–54,0) | 3,9 (2,0–5,0) | 1,5 (1,2–1,6) | 0,0 | 197,3 (197,0–198,0) | 177,3 (177,0–178,0) | 0,0 |
| all / human / perfect / llm_nonmatch / 50 | 1985,8 (1927,0–2000,0) | 5,0 | 134,5 (133,0–135,0) | 104,5 (103,0–105,0) | 7,6 (6,0–8,0) | 1,3 (1,3–1,3) | 0,0 | 147,3 (147,0–148,0) | 127,3 (127,0–128,0) | 0,0 |
| all / human / perfect / llm_nonmatch / 100 | 1909,7 (1617,0–1953,0) | 4,9 (4,0–5,0) | 230,9 (221,0–233,0) | 191,9 (191,0–193,0) | 17,0 (16,0–18,0) | 1,2 (1,2–1,2) | 0,0 | 49,1 (49,0–50,0) | 39,1 (39,0–40,0) | 0,0 |
| all / human / perfect / verified_only / 25 | 1885,7 (1571,0–1991,0) | 4,8 (4,0–5,0) | 81,7 (71,0–85,0) | 53,7 (51,0–55,0) | 3,8 (3,0–4,0) | 1,5 (1,4–1,6) | 0,0 | 196,9 (196,0–198,0) | 176,9 (176,0–178,0) | 0,0 |
| all / human / perfect / verified_only / 50 | 1877,2 (1571,0–2000,0) | 4,8 (4,0–5,0) | 131,5 (121,0–135,0) | 103,5 (101,0–105,0) | 6,6 (6,0–7,0) | 1,3 (1,2–1,3) | 0,0 | 146,9 (146,0–148,0) | 126,9 (126,0–128,0) | 0,0 |
| all / human / perfect / verified_only / 100 | 1882,6 (1627,0–2000,0) | 4,8 (4,0–5,0) | 230,0 (221,0–234,0) | 192,0 (191,0–194,0) | 17,2 (16,0–19,0) | 1,2 (1,2–1,2) | 0,0 | 49,1 (49,0–50,0) | 39,1 (39,0–40,0) | 0,0 |
| all / human / thesis_bands / llm_nonmatch / 25 | 1800,0 (1500,0–2000,0) | 3,6 (3,0–4,0) | 90,0 (75,0–100,0) | 51,9 (49,0–57,0) | 1,9 (0,0–5,0) | 1,7 (1,5–2,0) | 37,8 (25,0–50,0) | 332,4 (285,0–370,0) | 161,2 (147,0–177,0) | 52,7 (39,0–71,0) |
| all / human / thesis_bands / llm_nonmatch / 50 | 2000,0 | 4,0 | 200,0 | 103,5 (100,0–106,0) | 4,1 (3,0–6,0) | 1,9 (1,9–2,0) | 57,4 (41,0–69,0) | 246,9 (214,0–277,0) | 97,1 (91,0–105,0) | 37,5 (32,0–41,0) |
| all / human / thesis_bands / llm_nonmatch / 100 | 1950,0 (1500,0–2000,0) | 3,9 (3,0–4,0) | 342,1 (272,0–362,0) | 166,1 (159,0–175,0) | 13,9 (11,0–16,0) | 2,1 (1,7–2,2) | 87,4 (58,0–101,0) | 91,0 (78,0–111,0) | 4,6 (0,0–12,0) | 4,2 (0,0–11,0) |
| all / human / thesis_bands / verified_only / 25 | 1850,0 (1500,0–2000,0) | 3,7 (3,0–4,0) | 92,5 (75,0–100,0) | 52,6 (49,0–56,0) | 2,2 (1,0–4,0) | 1,8 (1,5–1,9) | 39,7 (25,0–48,0) | 339,1 (281,0–380,0) | 163,1 (151,0–179,0) | 54,2 (33,0–71,0) |
| all / human / thesis_bands / verified_only / 50 | 1900,0 (1500,0–2000,0) | 3,8 (3,0–4,0) | 190,0 (150,0–200,0) | 102,6 (98,0–108,0) | 3,6 (2,0–6,0) | 1,8 (1,5–2,0) | 56,9 (30,0–71,0) | 249,4 (206,0–281,0) | 98,6 (91,0–111,0) | 38,5 (27,0–52,0) |
| all / human / thesis_bands / verified_only / 100 | 1950,0 (1500,0–2000,0) | 3,9 (3,0–4,0) | 344,4 (277,0–371,0) | 166,4 (158,0–175,0) | 14,6 (13,0–17,0) | 2,1 (1,7–2,3) | 89,8 (66,0–115,0) | 92,5 (83,0–105,0) | 4,6 (0,0–12,0) | 4,2 (0,0–11,0) |
| below_cut / human / perfect / llm_nonmatch / 25 | 1026,3 (701,0–1294,0) | 3,0 (2,0–4,0) | 45,0 (34,0–56,0) | 15,0 (14,0–16,0) | 15,0 (14,0–16,0) | 3,0 (2,4–3,5) | 0,0 | 0,0 | 0,0 | 0,0 |
| below_cut / human / perfect / llm_nonmatch / 50 | 1026,3 (701,0–1294,0) | 3,0 (2,0–4,0) | 45,0 (34,0–56,0) | 15,0 (14,0–16,0) | 15,0 (14,0–16,0) | 3,0 (2,4–3,5) | 0,0 | 0,0 | 0,0 | 0,0 |
| below_cut / human / perfect / llm_nonmatch / 100 | 1026,3 (701,0–1294,0) | 3,0 (2,0–4,0) | 45,0 (34,0–56,0) | 15,0 (14,0–16,0) | 15,0 (14,0–16,0) | 3,0 (2,4–3,5) | 0,0 | 0,0 | 0,0 | 0,0 |
| below_cut / human / perfect / verified_only / 25 | 966,8 (662,0–1354,0) | 2,9 (2,0–4,0) | 43,2 (33,0–56,0) | 14,2 (13,0–16,0) | 14,2 (13,0–16,0) | 3,0 (2,4–3,7) | 0,0 | 0,0 | 0,0 | 0,0 |
| below_cut / human / perfect / verified_only / 50 | 966,8 (662,0–1354,0) | 2,9 (2,0–4,0) | 43,2 (33,0–56,0) | 14,2 (13,0–16,0) | 14,2 (13,0–16,0) | 3,0 (2,4–3,7) | 0,0 | 0,0 | 0,0 | 0,0 |
| below_cut / human / perfect / verified_only / 100 | 966,8 (662,0–1354,0) | 2,9 (2,0–4,0) | 43,2 (33,0–56,0) | 14,2 (13,0–16,0) | 14,2 (13,0–16,0) | 3,0 (2,4–3,7) | 0,0 | 0,0 | 0,0 | 0,0 |
| below_cut / human / thesis_bands / llm_nonmatch / 25 | 1350,0 (1000,0–2000,0) | 2,7 (2,0–4,0) | 67,5 (50,0–100,0) | 7,7 (5,0–10,0) | 7,7 (5,0–10,0) | 8,8 (6,2–12,5) | 58,3 (40,0–92,0) | 136,6 (90,0–204,0) | 14,5 (4,0–29,0) | 14,5 (4,0–29,0) |
| below_cut / human / thesis_bands / llm_nonmatch / 50 | 1350,0 (1000,0–2000,0) | 2,7 (2,0–4,0) | 135,0 (100,0–200,0) | 9,6 (5,0–13,0) | 9,6 (5,0–13,0) | 14,4 (10,0–20,0) | 68,0 (42,0–91,0) | 64,3 (47,0–87,0) | 0,0 | 0,0 |
| below_cut / human / thesis_bands / llm_nonmatch / 100 | 1450,0 (1000,0–2000,0) | 2,9 (2,0–4,0) | 217,4 (148,0–329,0) | 10,3 (5,0–14,0) | 10,3 (5,0–14,0) | 21,6 (13,7–30,8) | 75,9 (45,0–133,0) | 0,0 | 0,0 | 0,0 |
| below_cut / human / thesis_bands / verified_only / 25 | 1000,0 | 2,0 | 50,0 | 6,6 (5,0–8,0) | 6,6 (5,0–8,0) | 7,8 (6,2–10,0) | 41,2 (33,0–45,0) | 101,1 (86,0–122,0) | 9,5 (0,0–21,0) | 9,5 (0,0–21,0) |
| below_cut / human / thesis_bands / verified_only / 50 | 1300,0 (1000,0–2000,0) | 2,6 (2,0–4,0) | 130,0 (100,0–200,0) | 9,1 (5,0–12,0) | 9,1 (5,0–12,0) | 14,5 (10,0–20,0) | 68,8 (40,0–105,0) | 64,1 (38,0–104,0) | 0,0 | 0,0 |
| below_cut / human / thesis_bands / verified_only / 100 | 1100,0 (1000,0–1500,0) | 2,2 (2,0–3,0) | 167,6 (139,0–225,0) | 9,1 (5,0–11,0) | 9,1 (5,0–11,0) | 19,1 (13,9–30,2) | 58,4 (39,0–82,0) | 0,0 | 0,0 | 0,0 |

**Verificados abaixo do corte por orçamento humano B (fila)**

| configuração | B=50 | B=100 | B=200 | B=400 |
|---|---|---|---|---|
| all / human / perfect / llm_nonmatch / 25 | 2,0 | — | — | — |
| all / human / perfect / llm_nonmatch / 50 | 5,0 | 5,0 | — | — |
| all / human / perfect / llm_nonmatch / 100 | 5,0 | 9,0 | 16,1 (16,0–17,0) | — |
| all / human / perfect / verified_only / 25 | 2,0 | — | — | — |
| all / human / perfect / verified_only / 50 | 5,0 | 5,0 | — | — |
| all / human / perfect / verified_only / 100 | 5,0 | 9,0 | 16,9 (16,0–17,0) | — |
| all / human / thesis_bands / llm_nonmatch / 25 | 1,2 (0,0–2,0) | 2,2 (1,0–5,0) [n=6] | — | — |
| all / human / thesis_bands / llm_nonmatch / 50 | 2,1 (1,0–4,0) | 2,1 (1,0–4,0) | 4,1 (3,0–6,0) | — |
| all / human / thesis_bands / llm_nonmatch / 100 | 2,1 (1,0–4,0) | 6,9 (5,0–8,0) | 11,3 (9,0–13,0) | — |
| all / human / thesis_bands / verified_only / 25 | 1,2 (0,0–2,0) | 2,4 (1,0–4,0) [n=7] | — | — |
| all / human / thesis_bands / verified_only / 50 | 2,1 (1,0–4,0) | 2,1 (1,0–4,0) | 4,0 (2,0–6,0) [n=8] | — |
| all / human / thesis_bands / verified_only / 100 | 2,1 (1,0–4,0) | 6,9 (5,0–8,0) | 11,7 (10,0–14,0) | — |
| below_cut / human / perfect / llm_nonmatch / 25 | 16,0 [n=1] | — | — | — |
| below_cut / human / perfect / llm_nonmatch / 50 | 16,0 [n=1] | — | — | — |
| below_cut / human / perfect / llm_nonmatch / 100 | 16,0 [n=1] | — | — | — |
| below_cut / human / perfect / verified_only / 25 | 15,2 (15,0–16,0) [n=4] | — | — | — |
| below_cut / human / perfect / verified_only / 50 | 15,2 (15,0–16,0) [n=4] | — | — | — |
| below_cut / human / perfect / verified_only / 100 | 15,2 (15,0–16,0) [n=4] | — | — | — |
| below_cut / human / thesis_bands / llm_nonmatch / 25 | 7,5 (5,0–10,0) | 9,0 (8,0–10,0) [n=2] | — | — |
| below_cut / human / thesis_bands / llm_nonmatch / 50 | 8,3 (5,0–10,0) | 9,5 (5,0–12,0) | 13,0 [n=1] | — |
| below_cut / human / thesis_bands / llm_nonmatch / 100 | 8,3 (5,0–10,0) | 10,0 (5,0–12,0) | 11,3 (10,0–13,0) [n=7] | — |
| below_cut / human / thesis_bands / verified_only / 25 | 6,6 (5,0–8,0) | — | — | — |
| below_cut / human / thesis_bands / verified_only / 50 | 8,3 (5,0–10,0) | 8,9 (5,0–12,0) | 10,5 (10,0–11,0) [n=2] | — |
| below_cut / human / thesis_bands / verified_only / 100 | 8,3 (5,0–10,0) | 9,1 (5,0–11,0) | 11,0 [n=2] | — |

### Resposta

<!-- RESPOSTA:INICIO -->
**Resposta curta: nas condições simuladas, não.** Com o revisor `thesis_bands` e o humano perfeito, a fila (H2) não encurta o trabalho humano em relação a revisar sozinho na ordem da nota (H0) nem a deixar o LLM triar a ordem da nota e confirmar só os MATCH (H1). Só com revisor perfeito a fila e H1 encurtam. Os números, das tabelas acima:

- **Abaixo do corte, com `thesis_bands`, a orçamento humano igual** (verificados abaixo do corte, de 36): B=50: H0 10,0, H1 10,9 (8–14), H2 8,3 (5–10); B=100: H0 12,0, H1 13,2 (12–16), H2 9,5 (5–12). Em B=200, H1 chega a 17,6 (16–20) contra 13,0 de H0, mas à custa de 7084 (6740–7531) revisões de LLM; a fila padrão (50 por rodada) faz em média 135,0 decisões (100–200) no `below_cut` e só uma semente chega a 200 (`[n=1]`).
- **Decisões humanas até verificar k pares abaixo do corte, `thesis_bands`, `below_cut`:** k=5: H0 6 (5–6), H1 7 (5–9), H2 7 (5–9); k=10: H0 24 (23–24), H1 30 (20–55), H2 45 (30–54), este último atingido por só 70% das sementes.
- **Com revisor perfeito, no `below_cut`:** k=5 em 5 decisões (H1 e H2) contra 6 de H0, e k=10 em 10 decisões (H1 e H2) contra 24 de H0. Aqui o LLM encurta a revisão humana, porque todo MATCH é um par verdadeiro. A fila gasta, no total, 45,0 (34–56) decisões para verificar 15,0 (14–16) pares (3,0 decisões por verificado); o resto são itens de auditoria (10 por rodada em 3,0 rodadas).
- **No extrato inteiro (`all`)** a decisão humana quase sempre verifica um par (1,0 a 2,9 decisões por verificado), mas isso conta pares que o ORL já aceita acima do corte (211). Abaixo do corte, com `thesis_bands`: B=100: H0 8,7, H1 7,1 (5–10), H2 2,1 (1–4) com 50 por rodada e 6,9 (5–8) com 100 por rodada; B=200: H0 12,0, H1 12,7 (11–15), H2 4,1 (3–6) com 50 por rodada e 11,3 (9–13) com 100 por rodada.
- **O que o portão revisto faz:** o humano detecta os FP do LLM (FP detectados: 57,4 (41–69) no `all` e 68,0 (42–91) no `below_cut`, 50 por rodada, `thesis_bands`), então os MATCH aceitos ficam verificados. Mas, no `below_cut` com `thesis_bands`, a fila termina o run com 135,0 decisões a 14,4 (10–20) decisões por par verificado, contra 8,3 (H0) e 7,7 (6,2–8,3) (H1) em B=100, e 15,4 (H0) e 11,4 (10,0–12,5) (H1) em B=200.
<!-- RESPOSTA:FIM -->

## Resultados da fila (orçamento de LLM)

### Comportamento da fila (configuração padrão)

| configuração | revisões LLM | rodadas | decisões humanas | verdadeiros revisados | encontrados (MATCH do LLM ∧ verdade) | abaixo do corte (de 36) | FP | UNSURE |
|---|---|---|---|---|---|---|---|---|
| all / human / perfect | 1985,8 (1927,0–2000,0) | 5,0 | 134,5 (133,0–135,0) | 231,8 (231,0–232,0) | 231,8 (231,0–232,0) | 21,9 (21,0–22,0) | 0,0 | 0,0 |
| all / human / thesis_bands | 2000,0 | 4,0 | 200,0 | 231,4 (229,0–233,0) | 162,7 (154,0–169,0) | 14,4 (13,0–18,0) | 94,9 (78,0–107,0) | 149,3 (128,0–178,0) |
| below_cut / human / perfect | 1026,3 (701,0–1294,0) | 3,0 (2,0–4,0) | 45,0 (34,0–56,0) | 15,0 (14,0–16,0) | 15,0 (14,0–16,0) | 15,0 (14,0–16,0) | 0,0 | 0,0 |
| below_cut / human / thesis_bands | 1350,0 (1000,0–2000,0) | 2,7 (2,0–4,0) | 135,0 (100,0–200,0) | 14,1 (12,0–15,0) | 7,8 (5,0–10,0) | 7,8 (5,0–10,0) | 68,0 (42,0–91,0) | 96,5 (67,0–133,0) |
| all / auto / perfect | 1514,6 (1302,0–1672,0) | 3,6 (3,0–4,0) | 0,0 | 229,6 (229,0–230,0) | 229,6 (229,0–230,0) | 21,6 (21,0–22,0) | 0,0 | 0,0 |
| all / auto / thesis_bands | 1150,0 (500,0–2000,0) | 2,3 (1,0–4,0) | 0,0 | 192,6 (137,0–231,0) | 132,1 (86,0–168,0) | 10,9 (5,0–16,0) | 51,6 (14,0–99,0) | 87,4 (33,0–148,0) |
| below_cut / auto / perfect | 411,0 (393,0–439,0) | 1,0 | 0,0 | 13,1 (13,0–14,0) | 13,1 (13,0–14,0) | 13,1 (13,0–14,0) | 0,0 | 0,0 |
| below_cut / auto / thesis_bands | 500,0 | 1,0 | 0,0 | 12,7 (12,0–13,0) | 6,6 (5,0–8,0) | 6,6 (5,0–8,0) | 24,6 (17,0–31,0) | 38,2 (32,0–45,0) |
| below_cut / human / thesis_bands [tau=0,01, window=300] | 1550,0 (1000,0–2000,0) | 3,1 (2,0–4,0) | 155,0 (100,0–200,0) | 14,9 (14,0–15,0) | 8,6 (7,0–10,0) | 8,6 (7,0–10,0) | 77,2 (40,0–106,0) | 107,2 (78,0–145,0) |

### Motivos de parada, término e recalibração

Contagem média por run (parada de rodada, status de recalibração, método de `p0` na 1ª rodada) e fração de runs (término).

| configuração | término (fração de runs) | parada de rodada (média por run) | recalibração (média por run) | `p0` da 1ª rodada (fração de runs) |
|---|---|---|---|---|
| all / human / perfect / llm_nonmatch / 25 | max_rounds: 0,70; sterile_round: 0,30 | max_round_reviews: 2,00; posterior: 2,40 | fitted: 4,40 | anchor_platt: 1,00 |
| all / human / perfect / llm_nonmatch / 50 | max_reviews: 0,40; max_rounds: 0,60 | max_reviews: 0,40; max_round_reviews: 2,00; posterior: 2,60 | fitted: 5,00 | anchor_platt: 1,00 |
| all / human / perfect / llm_nonmatch / 100 | max_rounds: 0,90; sterile_round: 0,10 | max_round_reviews: 2,00; posterior: 2,90 | fitted: 4,90 | anchor_platt: 1,00 |
| all / human / perfect / verified_only / 25 | max_rounds: 0,80; sterile_round: 0,20 | max_round_reviews: 2,00; posterior: 2,80 | fitted: 2,80; skipped_insufficient: 2,00 | anchor_platt: 1,00 |
| all / human / perfect / verified_only / 50 | max_reviews: 0,10; max_rounds: 0,70; sterile_round: 0,20 | max_reviews: 0,10; max_round_reviews: 2,00; posterior: 2,70 | fitted: 2,80; skipped_insufficient: 2,00 | anchor_platt: 1,00 |
| all / human / perfect / verified_only / 100 | max_reviews: 0,10; max_rounds: 0,70; sterile_round: 0,20 | max_reviews: 0,10; max_round_reviews: 2,00; posterior: 2,70 | fitted: 3,80; skipped_insufficient: 1,00 | anchor_platt: 1,00 |
| all / human / thesis_bands / llm_nonmatch / 25 | max_reviews: 0,60; sterile_round: 0,40 | max_round_reviews: 3,60 | fitted: 3,60 | anchor_platt: 1,00 |
| all / human / thesis_bands / llm_nonmatch / 50 | max_reviews: 1,00 | max_round_reviews: 4,00 | fitted: 4,00 | anchor_platt: 1,00 |
| all / human / thesis_bands / llm_nonmatch / 100 | max_reviews: 0,90; sterile_round: 0,10 | max_round_reviews: 3,90 | fitted: 3,90 | anchor_platt: 1,00 |
| all / human / thesis_bands / verified_only / 25 | max_reviews: 0,70; sterile_round: 0,30 | max_round_reviews: 3,70 | fitted: 1,70; skipped_insufficient: 2,00 | anchor_platt: 1,00 |
| all / human / thesis_bands / verified_only / 50 | max_reviews: 0,80; sterile_round: 0,20 | max_round_reviews: 3,80 | fitted: 1,90; skipped_insufficient: 1,90 | anchor_platt: 1,00 |
| all / human / thesis_bands / verified_only / 100 | max_reviews: 0,90; sterile_round: 0,10 | max_round_reviews: 3,90 | fitted: 3,90 | anchor_platt: 1,00 |
| below_cut / human / perfect / llm_nonmatch / 25 | sterile_round: 1,00 | posterior: 3,00 | fitted: 3,00 | stub: 1,00 |
| below_cut / human / perfect / llm_nonmatch / 50 | sterile_round: 1,00 | posterior: 3,00 | fitted: 3,00 | stub: 1,00 |
| below_cut / human / perfect / llm_nonmatch / 100 | sterile_round: 1,00 | posterior: 3,00 | fitted: 3,00 | stub: 1,00 |
| below_cut / human / perfect / verified_only / 25 | sterile_round: 1,00 | posterior: 2,90 | fitted: 2,90 | stub: 1,00 |
| below_cut / human / perfect / verified_only / 50 | sterile_round: 1,00 | posterior: 2,90 | fitted: 2,90 | stub: 1,00 |
| below_cut / human / perfect / verified_only / 100 | sterile_round: 1,00 | posterior: 2,90 | fitted: 2,90 | stub: 1,00 |
| below_cut / human / thesis_bands / llm_nonmatch / 25 | max_reviews: 0,20; sterile_round: 0,80 | max_round_reviews: 2,70 | fitted: 2,70 | stub: 1,00 |
| below_cut / human / thesis_bands / llm_nonmatch / 50 | max_reviews: 0,10; sterile_round: 0,90 | max_round_reviews: 2,70 | fitted: 2,70 | stub: 1,00 |
| below_cut / human / thesis_bands / llm_nonmatch / 100 | max_reviews: 0,20; sterile_round: 0,80 | max_round_reviews: 2,90 | fitted: 2,90 | stub: 1,00 |
| below_cut / human / thesis_bands / verified_only / 25 | sterile_round: 1,00 | max_round_reviews: 2,00 | fitted: 2,00 | stub: 1,00 |
| below_cut / human / thesis_bands / verified_only / 50 | max_reviews: 0,20; sterile_round: 0,80 | max_round_reviews: 2,60 | fitted: 2,60 | stub: 1,00 |
| below_cut / human / thesis_bands / verified_only / 100 | sterile_round: 1,00 | max_round_reviews: 2,20 | fitted: 2,20 | stub: 1,00 |
| all / auto / perfect | sterile_round: 1,00 | max_round_reviews: 2,00; posterior: 1,60 | skipped_low_conf_dominant: 3,60 | anchor_platt: 1,00 |
| all / auto / thesis_bands | auto_anomaly: 0,60; max_reviews: 0,20; sterile_round: 0,20 | max_round_reviews: 2,30 | skipped_low_conf_dominant: 2,30 | anchor_platt: 1,00 |
| below_cut / auto / perfect | sterile_round: 1,00 | posterior: 1,00 | skipped_low_conf_dominant: 1,00 | stub: 1,00 |
| below_cut / auto / thesis_bands | sterile_round: 1,00 | max_round_reviews: 1,00 | skipped_low_conf_dominant: 1,00 | stub: 1,00 |
| below_cut / human / thesis_bands [tau=0,01, window=300] / llm_nonmatch / 50 | max_reviews: 0,20; sterile_round: 0,80 | max_round_reviews: 3,10 | fitted: 3,10 | stub: 1,00 |

### Regra de parada por posterior

A rodada só pára por `posterior` quando, em todos os braços ativos, P(θ > τ) < δ com τ = 0,05 e δ = 0,05, ou seja, quando a taxa de MATCH bruto do LLM em cada braço fica claramente abaixo de τ. Com `thesis_bands`, a taxa de FP da faixa `low` é 8,4% (tabela de taxas), acima de τ: mesmo em regiões sem nenhum par verdadeiro o LLM simulado marca MATCH com frequência maior que τ, e o posterior não cai abaixo de δ. A tabela mostra as paradas por motivo (configuração padrão, média por run).

| configuração | rodadas | paradas por `posterior` | paradas por `max_round_reviews` | paradas por `max_reviews` |
|---|---|---|---|---|
| all / human / perfect | 5,0 | 2,60 | 2,00 | 0,40 |
| all / human / thesis_bands | 4,0 | 0,00 | 4,00 | 0,00 |
| below_cut / human / perfect | 3,0 (2,0–4,0) | 3,00 | 0,00 | 0,00 |
| below_cut / human / thesis_bands | 2,7 (2,0–4,0) | 0,00 | 2,70 | 0,00 |
| all / auto / perfect | 3,6 (3,0–4,0) | 1,60 | 2,00 | 0,00 |
| all / auto / thesis_bands | 2,3 (1,0–4,0) | 0,00 | 2,30 | 0,00 |
| below_cut / auto / perfect | 1,0 | 1,00 | 0,00 | 0,00 |
| below_cut / auto / thesis_bands | 1,0 | 0,00 | 1,00 | 0,00 |
| below_cut / human / thesis_bands [tau=0,01, window=300] | 3,1 (2,0–4,0) | 0,00 | 3,10 | 0,00 |

### Comparação com as bases a orçamento de LLM igual (`run`)

**Pares verdadeiros encontrados (MATCH do LLM ∧ verdade)** (orçamento = nº de revisões da fila no run)

| configuração | revisões | fila | (a) nota | (b1) FS compostos | (c) aleatória |
|---|---|---|---|---|---|
| all / human / perfect | 1986 (1927–2000) | 231,8 (231,0–232,0) | 188,6 (186,0–189,0) | 235,2 (233,0–237,0) | 7,7 (4,0–15,0) |
| all / human / thesis_bands | 2000 | 162,7 (154,0–169,0) | 131,6 (122,0–139,0) | 162,5 (155,0–172,0) | 6,0 (2,0–13,0) |
| below_cut / human / perfect | 1026 (701–1294) | 15,0 (14,0–16,0) | 17,0 | 1,0 | 0,5 (0,0–1,0) |
| below_cut / human / thesis_bands | 1350 (1000–2000) | 7,8 (5,0–10,0) | 11,0 (9,0–14,0) | 0,9 (0,0–1,0) | 0,3 (0,0–1,0) |
| all / auto / perfect | 1515 (1302–1672) | 229,6 (229,0–230,0) | 173,0 (170,0–175,0) | 230,2 (229,0–232,0) | 5,9 (1,0–13,0) |
| all / auto / thesis_bands | 1150 (500–2000) | 132,1 (86,0–168,0) | 110,7 (89,0–134,0) | 159,5 (150,0–171,0) | 3,4 (0,0–13,0) |
| below_cut / auto / perfect | 411 (393–439) | 13,1 (13,0–14,0) | 14,0 (13,0–15,0) | 1,0 | 0,2 (0,0–1,0) |
| below_cut / auto / thesis_bands | 500 | 6,6 (5,0–8,0) | 8,8 (7,0–10,0) | 0,9 (0,0–1,0) | 0,2 (0,0–1,0) |
| below_cut / human / thesis_bands [tau=0,01, window=300] | 1550 (1000–2000) | 8,6 (7,0–10,0) | 11,4 (9,0–13,0) | 0,9 (0,0–1,0) | 0,5 (0,0–2,0) |

**Pares abaixo do corte encontrados (de 36)** (orçamento = nº de revisões da fila no run)

| configuração | revisões | fila | (a) nota | (b1) FS compostos | (c) aleatória |
|---|---|---|---|---|---|
| all / human / perfect | 1986 (1927–2000) | 21,9 (21,0–22,0) | 19,0 | 28,2 (26,0–30,0) | 0,9 (0,0–2,0) |
| all / human / thesis_bands | 2000 | 14,4 (13,0–18,0) | 12,0 (10,0–14,0) | 15,9 (13,0–18,0) | 0,8 (0,0–2,0) |
| below_cut / human / perfect | 1026 (701–1294) | 15,0 (14,0–16,0) | 17,0 | 1,0 | 0,5 (0,0–1,0) |
| below_cut / human / thesis_bands | 1350 (1000–2000) | 7,8 (5,0–10,0) | 11,0 (9,0–14,0) | 0,9 (0,0–1,0) | 0,3 (0,0–1,0) |
| all / auto / perfect | 1515 (1302–1672) | 21,6 (21,0–22,0) | 17,6 (17,0–18,0) | 23,2 (22,0–25,0) | 0,7 (0,0–1,0) |
| all / auto / thesis_bands | 1150 (500–2000) | 10,9 (5,0–16,0) | 9,4 (5,0–14,0) | 13,3 (9,0–18,0) | 0,3 (0,0–1,0) |
| below_cut / auto / perfect | 411 (393–439) | 13,1 (13,0–14,0) | 14,0 (13,0–15,0) | 1,0 | 0,2 (0,0–1,0) |
| below_cut / auto / thesis_bands | 500 | 6,6 (5,0–8,0) | 8,8 (7,0–10,0) | 0,9 (0,0–1,0) | 0,2 (0,0–1,0) |
| below_cut / human / thesis_bands [tau=0,01, window=300] | 1550 (1000–2000) | 8,6 (7,0–10,0) | 11,4 (9,0–13,0) | 0,9 (0,0–1,0) | 0,5 (0,0–2,0) |

**FP (MATCH do LLM sem verdade)** (orçamento = nº de revisões da fila no run)

| configuração | revisões | fila | (a) nota | (b1) FS compostos | (c) aleatória |
|---|---|---|---|---|---|
| all / human / perfect | 1986 (1927–2000) | 0,0 | 0,0 | 0,0 | 0,0 |
| all / human / thesis_bands | 2000 | 94,9 (78,0–107,0) | 50,3 (44,0–64,0) | 124,2 (114,0–145,0) | 145,8 (129,0–163,0) |
| below_cut / human / perfect | 1026 (701–1294) | 0,0 | 0,0 | 0,0 | 0,0 |
| below_cut / human / thesis_bands | 1350 (1000–2000) | 68,0 (42,0–91,0) | 37,1 (27,0–52,0) | 58,2 (31,0–92,0) | 98,5 (60,0–164,0) |
| all / auto / perfect | 1515 (1302–1672) | 0,0 | 0,0 | 0,0 | 0,0 |
| all / auto / thesis_bands | 1150 (500–2000) | 51,6 (14,0–99,0) | 28,3 (6,0–64,0) | 61,1 (12,0–127,0) | 80,4 (27,0–151,0) |
| below_cut / auto / perfect | 411 (393–439) | 0,0 | 0,0 | 0,0 | 0,0 |
| below_cut / auto / thesis_bands | 500 | 24,6 (17,0–31,0) | 13,9 (11,0–18,0) | 26,5 (19,0–35,0) | 36,2 (29,0–48,0) |
| below_cut / human / thesis_bands [tau=0,01, window=300] | 1550 (1000–2000) | 77,2 (40,0–106,0) | 42,3 (27,0–61,0) | 64,2 (43,0–96,0) | 112,6 (80,0–150,0) |

### Curvas: encontrados por nº de revisões de LLM

As bases não dependem do modo; aparecem uma vez por pool e revisor. Células com `[n=k]` têm viés de sobrevivência (ver o método).

**Verdadeiros encontrados**

| pool / revisor | método | 100 | 250 | 500 | 1000 | 2000 |
|---|---|---|---|---|---|---|
| all / perfect | fila auto-calibrável (human) | 83,2 (82,0–84,0) | 127,0 | 139,1 (139,0–140,0) | 227,3 (227,0–228,0) | 232,0 [n=4] |
| all / perfect | (a) nota decrescente | 98,0 | 133,0 | 145,0 | 166,6 (166,0–167,0) | 189,0 |
| all / perfect | (b1) FS sobre compostos | 89,6 (89,0–90,0) | 200,2 (200,0–201,0) | 222,9 (222,0–223,0) | 226,1 (225,0–228,0) | 235,5 (233,0–237,0) |
| all / perfect | (c) aleatória | 0,3 (0,0–2,0) | 0,6 (0,0–3,0) | 1,5 (0,0–4,0) | 3,0 (0,0–8,0) | 7,9 (4,0–16,0) |
| all / thesis_bands | fila auto-calibrável (human) | 47,2 (36,0–52,0) | 79,4 (68,0–87,0) | 88,5 (76,0–97,0) | 158,2 (150,0–164,0) | 162,7 (154,0–169,0) |
| all / thesis_bands | (a) nota decrescente | 61,6 (55,0–66,0) | 84,5 (72,0–93,0) | 94,2 (81,0–102,0) | 112,4 (101,0–121,0) | 131,6 (122,0–139,0) |
| all / thesis_bands | (b1) FS sobre compostos | 57,8 (52,0–64,0) | 140,3 (132,0–147,0) | 156,8 (148,0–166,0) | 159,1 (150,0–168,0) | 162,5 (155,0–172,0) |
| all / thesis_bands | (c) aleatória | 0,3 (0,0–2,0) | 0,6 (0,0–3,0) | 1,2 (0,0–4,0) | 2,4 (0,0–6,0) | 6,0 (2,0–13,0) |
| below_cut / perfect | fila auto-calibrável (human) | 10,0 | 12,0 | 13,3 (13,0–14,0) | 15,1 (15,0–16,0) [n=9] | — |
| below_cut / perfect | (a) nota decrescente | 12,0 | 13,0 | 15,0 | 17,0 | 19,0 |
| below_cut / perfect | (b1) FS sobre compostos | 0,0 | 0,0 | 1,0 | 1,0 | 1,0 |
| below_cut / perfect | (c) aleatória | 0,0 | 0,0 | 0,2 (0,0–1,0) | 0,4 (0,0–1,0) | 1,3 (0,0–3,0) |
| below_cut / thesis_bands | fila auto-calibrável (human) | 4,7 (4,0–6,0) | 6,0 (5,0–7,0) | 6,6 (5,0–8,0) | 7,7 (5,0–10,0) | 10,0 [n=1] |
| below_cut / thesis_bands | (a) nota decrescente | 6,0 (5,0–7,0) | 6,9 (5,0–8,0) | 8,8 (7,0–10,0) | 10,4 (8,0–12,0) | 12,0 (10,0–14,0) |
| below_cut / thesis_bands | (b1) FS sobre compostos | 0,0 | 0,0 | 0,9 (0,0–1,0) | 0,9 (0,0–1,0) | 0,9 (0,0–1,0) |
| below_cut / thesis_bands | (c) aleatória | 0,0 | 0,0 | 0,2 (0,0–1,0) | 0,2 (0,0–1,0) | 0,7 (0,0–2,0) |
| all / perfect | fila auto-calibrável (auto) | 83,2 (82,0–84,0) | 127,0 | 139,1 (139,0–140,0) | 226,9 (226,0–228,0) | — |
| all / thesis_bands | fila auto-calibrável (auto) | 47,2 (36,0–52,0) | 79,4 (68,0–87,0) | 88,5 (76,0–97,0) | 159,8 (152,0–168,0) [n=6] | 165,0 (164,0–166,0) [n=2] |
| below_cut / perfect | fila auto-calibrável (auto) | 10,0 | 12,0 | — | — | — |
| below_cut / thesis_bands | fila auto-calibrável (auto) | 4,7 (4,0–6,0) | 6,0 (5,0–7,0) | 6,6 (5,0–8,0) | — | — |

**Abaixo do corte encontrados (de 36)**

| pool / revisor | método | 100 | 250 | 500 | 1000 | 2000 |
|---|---|---|---|---|---|---|
| all / perfect | fila auto-calibrável (human) | 5,0 | 12,0 | 13,1 (13,0–14,0) | 19,3 (19,0–20,0) | 22,0 [n=4] |
| all / perfect | (a) nota decrescente | 8,7 (8,0–9,0) | 12,0 | 13,0 | 17,0 | 19,0 |
| all / perfect | (b1) FS sobre compostos | 6,0 | 7,2 (7,0–8,0) | 17,0 | 19,6 (18,0–21,0) | 28,5 (26,0–30,0) |
| all / perfect | (c) aleatória | 0,0 | 0,0 | 0,0 | 0,1 (0,0–1,0) | 0,9 (0,0–2,0) |
| all / thesis_bands | fila auto-calibrável (human) | 2,1 (1,0–4,0) | 6,0 (5,0–7,0) | 7,0 (5,0–9,0) | 12,6 (11,0–15,0) | 14,4 (13,0–18,0) |
| all / thesis_bands | (a) nota decrescente | 3,8 (3,0–5,0) | 6,0 (5,0–7,0) | 6,9 (5,0–8,0) | 10,4 (8,0–12,0) | 12,0 (10,0–14,0) |
| all / thesis_bands | (b1) FS sobre compostos | 2,6 (1,0–4,0) | 3,7 (2,0–5,0) | 11,0 (9,0–12,0) | 12,9 (10,0–15,0) | 15,9 (13,0–18,0) |
| all / thesis_bands | (c) aleatória | 0,0 | 0,0 | 0,0 | 0,1 (0,0–1,0) | 0,8 (0,0–2,0) |
| below_cut / perfect | fila auto-calibrável (human) | 10,0 | 12,0 | 13,3 (13,0–14,0) | 15,1 (15,0–16,0) [n=9] | — |
| below_cut / perfect | (a) nota decrescente | 12,0 | 13,0 | 15,0 | 17,0 | 19,0 |
| below_cut / perfect | (b1) FS sobre compostos | 0,0 | 0,0 | 1,0 | 1,0 | 1,0 |
| below_cut / perfect | (c) aleatória | 0,0 | 0,0 | 0,2 (0,0–1,0) | 0,4 (0,0–1,0) | 1,3 (0,0–3,0) |
| below_cut / thesis_bands | fila auto-calibrável (human) | 4,7 (4,0–6,0) | 6,0 (5,0–7,0) | 6,6 (5,0–8,0) | 7,7 (5,0–10,0) | 10,0 [n=1] |
| below_cut / thesis_bands | (a) nota decrescente | 6,0 (5,0–7,0) | 6,9 (5,0–8,0) | 8,8 (7,0–10,0) | 10,4 (8,0–12,0) | 12,0 (10,0–14,0) |
| below_cut / thesis_bands | (b1) FS sobre compostos | 0,0 | 0,0 | 0,9 (0,0–1,0) | 0,9 (0,0–1,0) | 0,9 (0,0–1,0) |
| below_cut / thesis_bands | (c) aleatória | 0,0 | 0,0 | 0,2 (0,0–1,0) | 0,2 (0,0–1,0) | 0,7 (0,0–2,0) |
| all / perfect | fila auto-calibrável (auto) | 5,0 | 12,0 | 13,1 (13,0–14,0) | 20,0 | — |
| all / thesis_bands | fila auto-calibrável (auto) | 2,1 (1,0–4,0) | 6,0 (5,0–7,0) | 7,0 (5,0–9,0) | 13,8 (13,0–15,0) [n=6] | 16,0 [n=2] |
| below_cut / perfect | fila auto-calibrável (auto) | 10,0 | 12,0 | — | — | — |
| below_cut / thesis_bands | fila auto-calibrável (auto) | 4,7 (4,0–6,0) | 6,0 (5,0–7,0) | 6,6 (5,0–8,0) | — | — |

### Revisões de LLM até achar k pares abaixo do corte

| configuração | método | k=5 | k=10 | k=20 |
|---|---|---|---|---|
| all / human / perfect | fila auto-calibrável | 53 (51–55) [1,00] | 148 (147–149) [1,00] | 1039 (968–1088) [1,00] |
| all / human / perfect | (a) nota decrescente | 43 (43–44) [1,00] | 129 (129–130) [1,00] | — (0) |
| all / human / perfect | (b1) FS sobre compostos | 63 (56–70) [1,00] | 312 (301–332) [1,00] | 964 (639–1206) [1,00] |
| all / human / perfect | (c) aleatória | — (0) | — (0) | — (0) |
| all / human / thesis_bands | fila auto-calibrável | 181 (119–248) [1,00] | 707 (647–861) [1,00] | — (0) |
| all / human / thesis_bands | (a) nota decrescente | 146 (94–197) [1,00] | 864 (540–1715) [1,00] | — (0) |
| all / human / thesis_bands | (b1) FS sobre compostos | 276 (238–329) [1,00] | 453 (359–672) [1,00] | — (0) |
| all / human / thesis_bands | (c) aleatória | — (0) | — (0) | — (0) |
| below_cut / human / perfect | fila auto-calibrável | 12 (8–15) [1,00] | 38 (35–41) [1,00] | — (0) |
| below_cut / human / perfect | (a) nota decrescente | 6 (5–6) [1,00] | 24 (23–24) [1,00] | — (0) |
| below_cut / human / perfect | (b1) FS sobre compostos | — (0) | — (0) | — (0) |
| below_cut / human / perfect | (c) aleatória | — (0) | — (0) | — (0) |
| below_cut / human / thesis_bands | fila auto-calibrável | 90 (21–196) [1,00] | 918 (798–1039) [0,20] | — (0) |
| below_cut / human / thesis_bands | (a) nota decrescente | 41 (9–80) [1,00] | 528 (408–603) [0,80] | — (0) |
| below_cut / human / thesis_bands | (b1) FS sobre compostos | — (0) | — (0) | — (0) |
| below_cut / human / thesis_bands | (c) aleatória | — (0) | — (0) | — (0) |
| all / auto / perfect | fila auto-calibrável | 53 (51–55) [1,00] | 148 (147–149) [1,00] | 883 (876–899) [1,00] |
| all / auto / perfect | (a) nota decrescente | 43 (43–44) [1,00] | 129 (129–130) [1,00] | — (0) |
| all / auto / perfect | (b1) FS sobre compostos | 63 (56–70) [1,00] | 312 (301–332) [1,00] | 964 (639–1206) [1,00] |
| all / auto / perfect | (c) aleatória | — (0) | — (0) | — (0) |
| all / auto / thesis_bands | fila auto-calibrável | 181 (119–248) [1,00] | 668 (619–689) [0,60] | — (0) |
| all / auto / thesis_bands | (a) nota decrescente | 146 (94–197) [1,00] | 638 (540–737) [0,60] | — (0) |
| all / auto / thesis_bands | (b1) FS sobre compostos | 276 (238–329) [1,00] | 429 (359–497) [0,90] | — (0) |
| all / auto / thesis_bands | (c) aleatória | — (0) | — (0) | — (0) |
| below_cut / auto / perfect | fila auto-calibrável | 12 (8–15) [1,00] | 38 (35–41) [1,00] | — (0) |
| below_cut / auto / perfect | (a) nota decrescente | 6 (5–6) [1,00] | 24 (23–24) [1,00] | — (0) |
| below_cut / auto / perfect | (b1) FS sobre compostos | — (0) | — (0) | — (0) |
| below_cut / auto / perfect | (c) aleatória | — (0) | — (0) | — (0) |
| below_cut / auto / thesis_bands | fila auto-calibrável | 90 (21–196) [1,00] | — (0) | — (0) |
| below_cut / auto / thesis_bands | (a) nota decrescente | 41 (9–80) [1,00] | 409 (408–409) [0,30] | — (0) |
| below_cut / auto / thesis_bands | (b1) FS sobre compostos | — (0) | — (0) | — (0) |
| below_cut / auto / thesis_bands | (c) aleatória | — (0) | — (0) | — (0) |
| below_cut / human / thesis_bands [tau=0,01, window=300] | fila auto-calibrável | 90 (21–196) [1,00] | 666 (639–693) [0,20] | — (0) |
| below_cut / human / thesis_bands [tau=0,01, window=300] | (a) nota decrescente | 41 (9–80) [1,00] | 631 (408–1456) [0,90] | — (0) |
| below_cut / human / thesis_bands [tau=0,01, window=300] | (b1) FS sobre compostos | — (0) | — (0) | — (0) |
| below_cut / human / thesis_bands [tau=0,01, window=300] | (c) aleatória | — (0) | — (0) | — (0) |

### Ganho das rodadas seguintes à 1ª

Pares abaixo do corte encontrados (MATCH do LLM ∧ verdade) entre as marcas de 500 e de 1.000 revisões de LLM, a fila contra a nota decrescente. A 1ª rodada (até 500 revisões) é ordenada pela nota; o que a fila acrescenta depois vem das rodadas ≥ 2. Médias; `[n=k]` = viés de sobrevivência.

| configuração | fila @500 | fila @1000 | Δ fila | nota @500 | nota @1000 | Δ nota |
|---|---|---|---|---|---|---|
| all / human / perfect | 13,1 (13,0–14,0) | 19,3 (19,0–20,0) | 6,2 | 13,0 | 17,0 | 4,0 |
| all / human / thesis_bands | 7,0 (5,0–9,0) | 12,6 (11,0–15,0) | 5,6 | 6,9 (5,0–8,0) | 10,4 (8,0–12,0) | 3,5 |
| below_cut / human / perfect | 13,3 (13,0–14,0) | 15,1 (15,0–16,0) [n=9] | 1,8 | 15,0 | 17,0 | 2,0 |
| below_cut / human / thesis_bands | 6,6 (5,0–8,0) | 7,7 (5,0–10,0) | 1,1 | 8,8 (7,0–10,0) | 10,4 (8,0–12,0) | 1,6 |

### Braço de exploração

| configuração | revisões LLM | exploração: revisões | exploração: encontrados | exploração: abaixo do corte |
|---|---|---|---|---|
| all / human / perfect | 1985,8 (1927,0–2000,0) | 198,2 (192,0–200,0) | 0,2 (0,0–1,0) | 0,2 (0,0–1,0) |
| all / human / thesis_bands | 2000,0 | 200,0 | 0,1 (0,0–1,0) | 0,1 (0,0–1,0) |
| below_cut / human / perfect | 1026,3 (701,0–1294,0) | 102,2 (70,0–129,0) | 0,2 (0,0–1,0) | 0,2 (0,0–1,0) |
| below_cut / human / thesis_bands | 1350,0 (1000,0–2000,0) | 135,0 (100,0–200,0) | 0,1 (0,0–1,0) | 0,1 (0,0–1,0) |
| all / auto / perfect | 1514,6 (1302,0–1672,0) | 151,0 (130,0–167,0) | 0,1 (0,0–1,0) | 0,1 (0,0–1,0) |
| all / auto / thesis_bands | 1150,0 (500,0–2000,0) | 115,0 (50,0–200,0) | 0,1 (0,0–1,0) | 0,1 (0,0–1,0) |
| below_cut / auto / perfect | 411,0 (393,0–439,0) | 40,6 (39,0–43,0) | 0,1 (0,0–1,0) | 0,1 (0,0–1,0) |
| below_cut / auto / thesis_bands | 500,0 | 50,0 | 0,1 (0,0–1,0) | 0,1 (0,0–1,0) |
| below_cut / human / thesis_bands [tau=0,01, window=300] | 1550,0 (1000,0–2000,0) | 155,0 (100,0–200,0) | 0,1 (0,0–1,0) | 0,1 (0,0–1,0) |

## O que o aprendizado (A') faz no modo automático

| configuração | aprendizado ligado (fração de runs) | recalibração (média por run) | `llm_match_unverified` |
|---|---|---|---|
| all / auto / perfect | 0,00 | skipped_low_conf_dominant: 3,60 | 229,6 (229,0–230,0) |
| all / auto / thesis_bands | 0,00 | skipped_low_conf_dominant: 2,30 | 183,7 (102,0–263,0) |
| below_cut / auto / perfect | 0,00 | skipped_low_conf_dominant: 1,00 | 13,1 (13,0–14,0) |
| below_cut / auto / thesis_bands | 0,00 | skipped_low_conf_dominant: 1,00 | 31,2 (24,0–38,0) |

## v1 × v2 × v3

Mesma grade de pool × modo × revisor e mesmas sementes; v3 = configuração padrão (humano: `llm_nonmatch`, 50 por rodada). Cada célula traz **v1 → v2 → v3** da fila. A v1 usava faixas por probabilidade e compostos; a v2, faixas por posto, exploração e sub-escores do CdR; a v3 volta aos compostos e muda o portão. O nº de revisões por run difere entre versões (regras de parada e portão diferentes), então a comparação a **orçamento igual** é a tabela por marcas. Células `[n=k]`: viés de sobrevivência.

**Encontrados (MATCH do LLM ∧ verdade), por run**

| configuração | v1 → v2 → v3 |
|---|---|
| all / human / perfect | 228,5 (227,0–230,0) → 153,0 (150,0–171,0) → 231,8 (231,0–232,0) |
| all / human / thesis_bands | 160,7 (153,0–169,0) → 98,7 (89,0–105,0) → 162,7 (154,0–169,0) |
| below_cut / human / perfect | 0,0 → 13,1 (13,0–14,0) → 15,0 (14,0–16,0) |
| below_cut / human / thesis_bands | 0,2 (0,0–1,0) → 7,2 (5,0–9,0) → 7,8 (5,0–10,0) |
| all / auto / perfect | 226,2 (224,0–229,0) → 139,2 (139,0–140,0) → 229,6 (229,0–230,0) |
| all / auto / thesis_bands | 89,6 (71,0–158,0) → 88,5 (76,0–97,0) → 132,1 (86,0–168,0) |
| below_cut / auto / perfect | 0,0 → 13,1 (13,0–14,0) → 13,1 (13,0–14,0) |
| below_cut / auto / thesis_bands | 0,2 (0,0–1,0) → 6,6 (5,0–8,0) → 6,6 (5,0–8,0) |

**Abaixo do corte (de 36), por run**

| configuração | v1 → v2 → v3 |
|---|---|
| all / human / perfect | 20,5 (19,0–23,0) → 17,8 (17,0–19,0) → 21,9 (21,0–22,0) |
| all / human / thesis_bands | 13,7 (12,0–16,0) → 10,5 (8,0–12,0) → 14,4 (13,0–18,0) |
| below_cut / human / perfect | 0,0 → 13,1 (13,0–14,0) → 15,0 (14,0–16,0) |
| below_cut / human / thesis_bands | 0,2 (0,0–1,0) → 7,2 (5,0–9,0) → 7,8 (5,0–10,0) |
| all / auto / perfect | 18,6 (16,0–20,0) → 13,2 (13,0–14,0) → 21,6 (21,0–22,0) |
| all / auto / thesis_bands | 6,1 (4,0–10,0) → 7,0 (5,0–9,0) → 10,9 (5,0–16,0) |
| below_cut / auto / perfect | 0,0 → 13,1 (13,0–14,0) → 13,1 (13,0–14,0) |
| below_cut / auto / thesis_bands | 0,2 (0,0–1,0) → 6,6 (5,0–8,0) → 6,6 (5,0–8,0) |

**FP, por run**

| configuração | v1 → v2 → v3 |
|---|---|
| all / human / perfect | 0,0 → 0,0 → 0,0 |
| all / human / thesis_bands | 107,0 (83,0–120,0) → 90,9 (77,0–103,0) → 94,9 (78,0–107,0) |
| below_cut / human / perfect | 0,0 → 0,0 → 0,0 |
| below_cut / human / thesis_bands | 43,5 (24,0–82,0) → 68,7 (46,0–122,0) → 68,0 (42,0–91,0) |
| all / auto / perfect | 0,0 → 0,0 → 0,0 |
| all / auto / thesis_bands | 21,1 (8,0–39,0) → 29,9 (14,0–46,0) → 51,6 (14,0–99,0) |
| below_cut / auto / perfect | 0,0 → 0,0 → 0,0 |
| below_cut / auto / thesis_bands | 35,5 (24,0–46,0) → 24,6 (17,0–31,0) → 24,6 (17,0–31,0) |

**Revisões de LLM por run**

| configuração | v1 → v2 → v3 |
|---|---|
| all / human / perfect | 1213 (1118–1473) → 1810 (1518–2000) → 1986 (1927–2000) |
| all / human / thesis_bands | 2000 → 2000 → 2000 |
| below_cut / human / perfect | 169 (165–179) → 686 (662–732) → 1026 (701–1294) |
| below_cut / human / thesis_bands | 600 (500–1000) → 1300 (1000–2000) → 1350 (1000–2000) |
| all / auto / perfect | 1107 (1035–1249) → 774 (766–792) → 1515 (1302–1672) |
| all / auto / thesis_bands | 550 (500–1000) → 800 (500–1000) → 1150 (500–2000) |
| below_cut / auto / perfect | 169 (165–179) → 411 (393–439) → 411 (393–439) |
| below_cut / auto / thesis_bands | 500 → 500 → 500 |

**Pares abaixo do corte encontrados por marca de revisões de LLM** (v1 → v2 → v3)

| configuração | 100 | 250 | 500 | 1000 | 2000 |
|---|---|---|---|---|---|
| all / human / perfect | 4,5 (3,0–6,0) → 5,0 → 5,0 | 10,4 (10,0–11,0) → 12,0 → 12,0 | 12,1 (10,0–14,0) → 13,1 (13,0–14,0) → 13,1 (13,0–14,0) | 20,0 (19,0–22,0) → 17,2 (17,0–18,0) → 19,3 (19,0–20,0) | — → 19,0 [n=1] → 22,0 [n=4] |
| all / human / thesis_bands | 1,8 (0,0–4,0) → 2,1 (1,0–4,0) → 2,1 (1,0–4,0) | 5,0 (4,0–6,0) → 6,0 (5,0–7,0) → 6,0 (5,0–7,0) | 5,8 (4,0–7,0) → 7,0 (5,0–9,0) → 7,0 (5,0–9,0) | 10,1 (7,0–13,0) → 8,7 (6,0–11,0) → 12,6 (11,0–15,0) | 13,7 (12,0–16,0) → 10,5 (8,0–12,0) → 14,4 (13,0–18,0) |
| below_cut / human / perfect | 0,0 → 10,0 → 10,0 | — → 12,0 → 12,0 | — → 13,1 (13,0–14,0) → 13,3 (13,0–14,0) | — → — → 15,1 (15,0–16,0) [n=9] | — → — → — |
| below_cut / human / thesis_bands | 0,0 → 4,7 (4,0–6,0) → 4,7 (4,0–6,0) | 0,2 (0,0–1,0) → 6,0 (5,0–7,0) → 6,0 (5,0–7,0) | 0,2 (0,0–1,0) → 6,6 (5,0–8,0) → 6,6 (5,0–8,0) | 1,0 [n=2] → 7,0 (5,0–8,0) → 7,7 (5,0–10,0) | — → 8,3 (8,0–9,0) [n=3] → 10,0 [n=1] |
| all / auto / perfect | 4,5 (3,0–6,0) → 5,0 → 5,0 | 10,4 (10,0–11,0) → 12,0 → 12,0 | 12,1 (10,0–14,0) → 13,1 (13,0–14,0) → 13,1 (13,0–14,0) | 18,0 (16,0–20,0) → — → 20,0 | — → — → — |
| all / auto / thesis_bands | 1,8 (0,0–4,0) → 2,1 (1,0–4,0) → 2,1 (1,0–4,0) | 5,0 (4,0–6,0) → 6,0 (5,0–7,0) → 6,0 (5,0–7,0) | 5,8 (4,0–7,0) → 7,0 (5,0–9,0) → 7,0 (5,0–9,0) | 10,0 [n=1] → 7,8 (7,0–9,0) [n=6] → 13,8 (13,0–15,0) [n=6] | — → — → 16,0 [n=2] |
| below_cut / auto / perfect | 0,0 → 10,0 → 10,0 | — → 12,0 → 12,0 | — → — → — | — → — → — | — → — → — |
| below_cut / auto / thesis_bands | 0,0 → 4,7 (4,0–6,0) → 4,7 (4,0–6,0) | 0,2 (0,0–1,0) → 6,0 (5,0–7,0) → 6,0 (5,0–7,0) | 0,2 (0,0–1,0) → 6,6 (5,0–8,0) → 6,6 (5,0–8,0) | — → — → — | — → — → — |

**Troca de campos na v2** (ablação da v2: compostos em vez de sub-escores, humano × `thesis_bands`), abaixo do corte por run:

| configuração | v1 (compostos) | v2 (sub-escores do CdR) | v2 ablação (compostos) | v3 (compostos) |
|---|---|---|---|---|
| all / human / thesis_bands | 13,7 (12,0–16,0) | 10,5 (8,0–12,0) | 14,2 (13,0–16,0) | 14,4 (13,0–18,0) |
| below_cut / human / thesis_bands | 0,2 (0,0–1,0) | 7,2 (5,0–9,0) | 8,0 (5,0–10,0) | 7,8 (5,0–10,0) |

## Revisor calibrado no Haiku real

Esta seção usa decisões **reais** do Claude Haiku 5.5 (prompt v4, nota oculta), gravadas antes em um fixture local e **reaplicadas sem nenhuma chamada à API**. O fixture e o `.rates.csv` ficam fora do git. Fixture: 4604 pares, sha256 `51521231ba9779b52ea68b3896b35cf728b851e016280d1bc9971b4609b931d5`; origem (precedência calibração > teste de reordenação > filas e bases): 717 da amostra de calibração, 2785 do teste de reordenação (pares abaixo do corte em ordem decrescente de nota) e 1102 de filas e bases. Os três revisores novos:
- `measured`: taxas por faixa do `.rates.csv` **como estão** (proporções brutas, sem suavização), com `disagreement_rate = 0,10` (suposição, como em `thesis_bands`).
- `measured_smoothed`: as mesmas contagens, de `docs/llm_real_haiku_sem_nota.csv` (linhas `lado = todos`), com **Jeffreys** (a + 0,5) / (n + 1): fp = (FP + 0,5) / (não pares + 1), fn = (FN + 0,5) / (pares + 1), unsure = (UNSURE + 0,5) / (decisões + 1).
- `fixture_replay`: o par presente no fixture volta com a decisão real gravada (`FixtureReviewer`); o par ausente usa o fallback por taxas `measured_smoothed`.
**Atenção ao replay.** No `fixture_replay` o veredito de um par presente no fixture é **fixo entre sementes**: as sementes só mudam a exploração, o desempate, o sorteio da auditoria e o fallback dos pares ausentes. A variação entre sementes é, portanto, menor que nos revisores por taxas, e os intervalos mínimo–máximo subestimam a incerteza do revisor real. As taxas medidas vêm de amostras pequenas (`low` tem 3 pares verdadeiros, `high` tem 2 não pares): tratar essas duas faixas como não medidas.

**Taxas por faixa dos três revisores** (FP, FN, UNSURE)

| faixa | thesis_bands FP | FN | UNSURE | measured FP | FN | UNSURE | measured_smoothed FP | FN | UNSURE |
|---|---|---|---|---|---|---|---|---|---|
| low | 8,4% | 75,0% | 8,2% | 0,0% | 66,7% | 4,9% | 0,5% | 62,5% | 5,3% |
| grey_low | 8,4% | 75,0% | 8,2% | 1,0% | 29,4% | 5,1% | 1,5% | 30,6% | 5,5% |
| grey_mid | 2,7% | 6,7% | 6,0% | 0,0% | 7,0% | 5,1% | 0,5% | 7,8% | 5,4% |
| grey_high | 2,9% | 11,1% | 4,7% | 1,0% | 9,3% | 2,8% | 1,5% | 10,2% | 3,1% |
| near_high | 4,5% | 30,6% | 8,0% | 0,0% | 29,4% | 10,8% | 0,7% | 30,0% | 11,2% |
| high | 50,0% | 25,0% | 16,7% | 0,0% | 21,5% | 16,8% | 16,7% | 21,8% | 17,2% |

### Fração das revisões vinda do fixture

Revisões da fila (`fixture_replay`) servidas pelo fixture (decisão real) e pelo fallback por taxas; e fração das revisões das bases (nota, FS, aleatória e H1) servidas pelo fixture.

| pool / negative_labels | revisões da fila | do fixture | do fallback | fração do fixture (fila) | fração do fixture (bases) |
|---|---|---|---|---|---|
| all / llm_nonmatch | 1453,8 (1406,0–1500,0) | 1292,2 (1262,0–1318,0) | 161,6 (134,0–203,0) | 88,9% | 31,8% |
| all / verified_only | 1426,8 (1387,0–1467,0) | 1278,7 (1255,0–1304,0) | 148,1 (128,0–180,0) | 89,6% | — |
| below_cut / llm_nonmatch | 926,1 (831,0–1000,0) | 496,3 (453,0–544,0) | 429,8 (339,0–534,0) | 53,8% | 20,5% |
| below_cut / verified_only | 918,5 (822,0–1000,0) | 515,8 (437,0–600,0) | 402,7 (344,0–459,0) | 56,1% | — |

### Esforço humano a orçamento B (decisões humanas)

Configuração padrão da fila (`llm_nonmatch`, 50 por rodada), humano perfeito; H0 não depende do revisor. Comparação com `thesis_bands`. `[n=k]` = viés de sobrevivência; `—` = não chegou a B.

**Pares abaixo do corte verificados (de 36)**

| pool / revisor | método | B=50 | B=100 | B=200 | B=400 | run |
|---|---|---|---|---|---|---|
| all | H0 humano sozinho (ordem da nota) | 5,0 | 8,7 (8,0–9,0) | 12,0 | 13,0 | 13,0 |
| all / thesis_bands | H1 nota + LLM tria + humano confirma MATCH | 2,1 (1,0–4,0) | 7,1 (5,0–10,0) | 12,7 (11,0–15,0) | 18,3 (17,0–20,0) | 18,3 (17,0–20,0) |
| all / thesis_bands | H2 fila com portão revisto | 2,1 (1,0–4,0) | 2,1 (1,0–4,0) | 4,1 (3,0–6,0) | — | 4,1 (3,0–6,0) |
| all / measured | H1 nota + LLM tria + humano confirma MATCH | 2,3 (1,0–4,0) | 8,0 (7,0–11,0) | 20,2 (17,0–23,0) | — | 25,0 (23,0–27,0) |
| all / measured | H2 fila com portão revisto | 2,3 (1,0–4,0) | 2,3 (1,0–4,0) | — | — | 3,8 (2,0–6,0) |
| all / measured_smoothed | H1 nota + LLM tria + humano confirma MATCH | 2,3 (1,0–4,0) | 7,7 (5,0–11,0) | 16,9 (15,0–20,0) | 24,1 (22,0–27,0) | 24,1 (22,0–27,0) |
| all / measured_smoothed | H2 fila com portão revisto | 2,3 (1,0–4,0) | 2,3 (1,0–4,0) | 3,0 [n=1] | — | 3,3 (2,0–7,0) |
| all / fixture_replay | H1 nota + LLM tria + humano confirma MATCH | 4,0 | 8,0 | 11,0 | 11,0 | 11,0 |
| all / fixture_replay | H2 fila com portão revisto | 4,0 | 4,0 | — | — | 4,0 |
| below_cut | H0 humano sozinho (ordem da nota) | 10,0 | 12,0 | 13,0 | 13,0 | 13,0 |
| below_cut / thesis_bands | H1 nota + LLM tria + humano confirma MATCH | 10,9 (8,0–14,0) | 13,2 (12,0–16,0) | 17,6 (16,0–20,0) | 18,6 (17,0–21,0) | 18,6 (17,0–21,0) |
| below_cut / thesis_bands | H2 fila com portão revisto | 8,3 (5,0–10,0) | 9,5 (5,0–12,0) | 13,0 [n=1] | — | 9,6 (5,0–13,0) |
| below_cut / measured | H1 nota + LLM tria + humano confirma MATCH | 20,8 (18,0–23,0) | 23,1 (20,0–26,0) | — | — | 25,0 (23,0–27,0) |
| below_cut / measured | H2 fila com portão revisto | 10,1 (8,0–13,0) | 11,8 (9,0–14,0) [n=6] | — | — | 11,0 (8,0–14,0) |
| below_cut / measured_smoothed | H1 nota + LLM tria + humano confirma MATCH | 16,2 (13,0–20,0) | 20,8 (19,0–23,0) | 23,7 (22,0–26,0) | — | 24,5 (23,0–27,0) |
| below_cut / measured_smoothed | H2 fila com portão revisto | 9,3 (7,0–11,0) | 10,9 (9,0–13,0) [n=8] | — | — | 10,6 (8,0–13,0) |
| below_cut / fixture_replay | H1 nota + LLM tria + humano confirma MATCH | 11,0 | 11,0 | 11,0 | 11,0 [n=7] | 11,0 |
| below_cut / fixture_replay | H2 fila com portão revisto | 6,1 (6,0–7,0) | — | — | — | 6,1 (6,0–7,0) |

**Pares verdadeiros verificados**

| pool / revisor | método | B=50 | B=100 | B=200 | B=400 | run |
|---|---|---|---|---|---|---|
| all | H0 humano sozinho (ordem da nota) | 49,0 | 98,0 | 127,0 | 140,0 | 140,0 |
| all / thesis_bands | H1 nota + LLM tria + humano confirma MATCH | 48,9 (48,0–50,0) | 91,4 (83,0–95,0) | 133,6 (129,0–139,0) | 165,2 (159,0–170,0) | 165,2 (159,0–170,0) |
| all / thesis_bands | H2 fila com portão revisto | 48,9 (48,0–50,0) | 98,6 (97,0–100,0) | 103,5 (100,0–106,0) | — | 103,5 (100,0–106,0) |
| all / measured | H1 nota + LLM tria + humano confirma MATCH | 50,0 | 97,3 (96,0–99,0) | 171,1 (166,0–176,0) | — | 181,6 (175,0–190,0) |
| all / measured | H2 fila com portão revisto | 50,0 | 99,8 (99,0–100,0) | — | — | 102,7 (100,0–105,0) |
| all / measured_smoothed | H1 nota + LLM tria + humano confirma MATCH | 49,7 (49,0–50,0) | 95,5 (92,0–98,0) | 162,5 (156,0–166,0) | 178,5 (172,0–188,0) | 178,5 (172,0–188,0) |
| all / measured_smoothed | H2 fila com portão revisto | 49,7 (49,0–50,0) | 99,5 (99,0–100,0) | 102,0 [n=1] | — | 102,2 (99,0–105,0) |
| all / fixture_replay | H1 nota + LLM tria + humano confirma MATCH | 50,0 | 90,0 | 138,4 (137,0–140,0) | 166,7 (166,0–167,0) | 166,7 (166,0–167,0) |
| all / fixture_replay | H2 fila com portão revisto | 50,0 | 96,0 | — | — | 96,0 |
| below_cut | H0 humano sozinho (ordem da nota) | 10,0 | 12,0 | 13,0 | 13,0 | 13,0 |
| below_cut / thesis_bands | H1 nota + LLM tria + humano confirma MATCH | 10,9 (8,0–14,0) | 13,2 (12,0–16,0) | 17,6 (16,0–20,0) | 18,6 (17,0–21,0) | 18,6 (17,0–21,0) |
| below_cut / thesis_bands | H2 fila com portão revisto | 8,3 (5,0–10,0) | 9,5 (5,0–12,0) | 13,0 [n=1] | — | 9,6 (5,0–13,0) |
| below_cut / measured | H1 nota + LLM tria + humano confirma MATCH | 20,8 (18,0–23,0) | 23,1 (20,0–26,0) | — | — | 25,0 (23,0–27,0) |
| below_cut / measured | H2 fila com portão revisto | 10,1 (8,0–13,0) | 11,8 (9,0–14,0) [n=6] | — | — | 11,0 (8,0–14,0) |
| below_cut / measured_smoothed | H1 nota + LLM tria + humano confirma MATCH | 16,2 (13,0–20,0) | 20,8 (19,0–23,0) | 23,7 (22,0–26,0) | — | 24,5 (23,0–27,0) |
| below_cut / measured_smoothed | H2 fila com portão revisto | 9,3 (7,0–11,0) | 10,9 (9,0–13,0) [n=8] | — | — | 10,6 (8,0–13,0) |
| below_cut / fixture_replay | H1 nota + LLM tria + humano confirma MATCH | 11,0 | 11,0 | 11,0 | 11,0 [n=7] | 11,0 |
| below_cut / fixture_replay | H2 fila com portão revisto | 6,1 (6,0–7,0) | — | — | — | 6,1 (6,0–7,0) |

**Decisões humanas por par verificado**

| pool / revisor | método | B=50 | B=100 | B=200 | B=400 | run |
|---|---|---|---|---|---|---|
| all | H0 humano sozinho (ordem da nota) | 1,0 | 1,0 | 1,6 | 2,9 | 2,9 |
| all / thesis_bands | H1 nota + LLM tria + humano confirma MATCH | 1,0 (1,0–1,0) | 1,1 (1,1–1,2) | 1,5 (1,4–1,6) | 2,4 (2,4–2,5) | 2,4 (2,4–2,5) |
| all / thesis_bands | H2 fila com portão revisto | 1,0 (1,0–1,0) | 1,0 (1,0–1,0) | 1,9 (1,9–2,0) | — | 1,9 (1,9–2,0) |
| all / measured | H1 nota + LLM tria + humano confirma MATCH | 1,0 | 1,0 (1,0–1,0) | 1,2 (1,1–1,2) | — | 1,8 (1,7–1,9) |
| all / measured | H2 fila com portão revisto | 1,0 | 1,0 (1,0–1,0) | — | — | 1,7 (1,2–1,9) |
| all / measured_smoothed | H1 nota + LLM tria + humano confirma MATCH | 1,0 (1,0–1,0) | 1,0 (1,0–1,1) | 1,2 (1,2–1,3) | 2,2 (2,1–2,3) | 2,2 (2,1–2,3) |
| all / measured_smoothed | H2 fila com portão revisto | 1,0 (1,0–1,0) | 1,0 (1,0–1,0) | 2,0 [n=1] | — | 1,8 (1,3–2,0) |
| all / fixture_replay | H1 nota + LLM tria + humano confirma MATCH | 1,0 | 1,1 | 1,4 (1,4–1,5) | 2,4 (2,4–2,4) | 2,4 (2,4–2,4) |
| all / fixture_replay | H2 fila com portão revisto | 1,0 | 1,0 | — | — | 1,3 (1,3–1,4) |
| below_cut | H0 humano sozinho (ordem da nota) | 5,0 | 8,3 | 15,4 | 30,8 | 30,8 |
| below_cut / thesis_bands | H1 nota + LLM tria + humano confirma MATCH | 4,7 (3,6–6,2) | 7,7 (6,2–8,3) | 11,4 (10,0–12,5) | 21,6 (19,0–23,5) | 21,6 (19,0–23,5) |
| below_cut / thesis_bands | H2 fila com portão revisto | 6,3 (5,0–10,0) | 11,3 (8,3–20,0) | 15,4 [n=1] | — | 14,4 (10,0–20,0) |
| below_cut / measured | H1 nota + LLM tria + humano confirma MATCH | 2,4 (2,2–2,8) | 4,4 (3,8–5,0) | — | — | 6,7 (5,8–7,3) |
| below_cut / measured | H2 fila com portão revisto | 5,1 (3,8–6,2) | 8,6 (7,1–11,1) [n=6] | — | — | 9,5 (8,2–11,4) |
| below_cut / measured_smoothed | H1 nota + LLM tria + humano confirma MATCH | 3,1 (2,5–3,8) | 4,8 (4,3–5,3) | 8,5 (7,7–9,1) | — | 12,5 (10,4–13,8) |
| below_cut / measured_smoothed | H2 fila com portão revisto | 5,5 (4,5–7,1) | 9,3 (7,7–11,1) [n=8] | — | — | 11,3 (7,0–16,3) |
| below_cut / fixture_replay | H1 nota + LLM tria + humano confirma MATCH | 4,5 | 9,1 | 18,2 | 36,4 [n=7] | 36,2 (35,2–36,4) |
| below_cut / fixture_replay | H2 fila com portão revisto | 8,2 (7,1–8,3) | — | — | — | 14,0 (12,8–15,8) |

**Revisões de LLM consumidas**

| pool / revisor | método | B=50 | B=100 | B=200 | B=400 | run |
|---|---|---|---|---|---|---|
| all | H0 humano sozinho (ordem da nota) | 0 | 0 | 0 | 0 | 0 |
| all / thesis_bands | H1 nota + LLM tria + humano confirma MATCH | 79 (71–89) | 413 (272–541) | 2569 (2067–2966) | 8082 (7922–8503) | 8082 (7922–8503) |
| all / thesis_bands | H2 fila com portão revisto | 500 | 1000 | 2000 | — | 2000 |
| all / measured | H1 nota + LLM tria + humano confirma MATCH | 78 (74–88) | 540 (390–766) | 9428 (8571–10477) | — | 25426 (21671–29646) |
| all / measured | H2 fila com portão revisto | 500 | 1000 | — | — | 1851 (1278–2000) |
| all / measured_smoothed | H1 nota + LLM tria + humano confirma MATCH | 78 (72–88) | 502 (390–734) | 5736 (4865–7934) | 20621 (18832–23968) | 20621 (18832–23968) |
| all / measured_smoothed | H2 fila com portão revisto | 500 | 1000 | 2000 [n=1] | — | 1875 (1317–2000) |
| all / fixture_replay | H1 nota + LLM tria + humano confirma MATCH | 84 | 540 (539–541) | 3474 (3348–3538) | 15263 (14153–16689) | 15263 (14153–16689) |
| all / fixture_replay | H2 fila com portão revisto | 500 | 1000 | — | — | 1454 (1406–1500) |
| below_cut | H0 humano sozinho (ordem da nota) | 0 | 0 | 0 | 0 | 0 |
| below_cut / thesis_bands | H1 nota + LLM tria + humano confirma MATCH | 1414 (1104–1548) | 3256 (2829–3699) | 7084 (6740–7531) | 9772 (9527–10017) | 9772 (9527–10017) |
| below_cut / thesis_bands | H2 fila com portão revisto | 500 | 1000 | 2000 [n=1] | — | 1350 (1000–2000) |
| below_cut / measured | H1 nota + LLM tria + humano confirma MATCH | 9287 (8671–10067) | 14354 (12658–15568) | — | — | 22560 (21291–24444) |
| below_cut / measured | H2 fila com portão revisto | 698 (496–955) | 1200 (1088–1309) [n=6] | — | — | 1176 (779–1608) |
| below_cut / measured_smoothed | H1 nota + LLM tria + humano confirma MATCH | 4755 (3885–7030) | 9796 (8808–10807) | 16854 (15063–18459) | — | 29679 (29032–29929) |
| below_cut / measured_smoothed | H2 fila com portão revisto | 609 (500–915) | 1302 (1132–1489) [n=8] | — | — | 1282 (775–1787) |
| below_cut / fixture_replay | H1 nota + LLM tria + humano confirma MATCH | 1618 (1610–1623) | 5644 (5537–5786) | 12315 (11191–13216) | 27755 (24906–29678) [n=7] | 28308 (24906–29708) |
| below_cut / fixture_replay | H2 fila com portão revisto | 498 (492–500) | — | — | — | 926 (831–1000) |

**Decisões humanas até verificar k pares abaixo do corte**

| pool / revisor | método | k=5 | k=10 |
|---|---|---|---|
| all | H0 humano sozinho (ordem da nota) | 43 (43–44) [1,00] | 129 (129–130) [1,00] |
| all / thesis_bands | H1 nota + LLM tria + humano confirma MATCH | 74 (58–82) [1,00] | 124 (100–166) [1,00] |
| all / thesis_bands | H2 fila com portão revisto | 144 (127–160) [0,20] | — (0) |
| all / measured | H1 nota + LLM tria + humano confirma MATCH | 69 (58–80) [1,00] | 108 (91–127) [1,00] |
| all / measured | H2 fila com portão revisto | 125 (102–140) [0,30] | — (0) |
| all / measured_smoothed | H1 nota + LLM tria + humano confirma MATCH | 70 (58–81) [1,00] | 110 (93–136) [1,00] |
| all / measured_smoothed | H2 fila com portão revisto | 102 (102–102) [0,10] | — (0) |
| all / fixture_replay | H1 nota + LLM tria + humano confirma MATCH | 59 (58–60) [1,00] | 112 (112–113) [1,00] |
| all / fixture_replay | H2 fila com portão revisto | — (0) | — (0) |
| below_cut | H0 humano sozinho (ordem da nota) | 6 (5–6) [1,00] | 24 (23–24) [1,00] |
| below_cut / thesis_bands | H1 nota + LLM tria + humano confirma MATCH | 7 (5–9) [1,00] | 30 (20–55) [1,00] |
| below_cut / thesis_bands | H2 fila com portão revisto | 7 (5–9) [1,00] | 45 (30–54) [0,70] |
| below_cut / measured | H1 nota + LLM tria + humano confirma MATCH | 5 (5–5) [1,00] | 14 (12–17) [1,00] |
| below_cut / measured | H2 fila com portão revisto | 5 (5–5) [1,00] | 31 (11–51) [0,80] |
| below_cut / measured_smoothed | H1 nota + LLM tria + humano confirma MATCH | 6 (5–6) [1,00] | 17 (12–25) [1,00] |
| below_cut / measured_smoothed | H2 fila com portão revisto | 6 (5–6) [1,00] | 36 (12–92) [0,70] |
| below_cut / fixture_replay | H1 nota + LLM tria + humano confirma MATCH | 5 (5–5) [1,00] | 28 (28–28) [1,00] |
| below_cut / fixture_replay | H2 fila com portão revisto | 5 (5–5) [1,00] | — (0) |

### Efeito de `negative_labels` (fila, 50 por rodada)

| configuração | revisões LLM | decisões humanas | verificados | verificados abaixo do corte | decisões por verificado | FP detectados | over_budget | FP não verificados |
|---|---|---|---|---|---|---|---|---|
| all / thesis_bands / llm_nonmatch | 2000,0 | 200,0 | 103,5 (100,0–106,0) | 4,1 (3,0–6,0) | 1,9 (1,9–2,0) | 57,4 (41,0–69,0) | 246,9 (214,0–277,0) | 37,5 (32,0–41,0) |
| all / thesis_bands / verified_only | 1900,0 (1500,0–2000,0) | 190,0 (150,0–200,0) | 102,6 (98,0–108,0) | 3,6 (2,0–6,0) | 1,8 (1,5–2,0) | 56,9 (30,0–71,0) | 249,4 (206,0–281,0) | 38,5 (27,0–52,0) |
| all / measured / llm_nonmatch | 1851,3 (1278,0–2000,0) | 174,2 (121,0–192,0) | 102,7 (100,0–105,0) | 3,8 (2,0–6,0) | 1,7 (1,2–1,9) | 2,2 (0,0–6,0) | 151,9 (141,0–169,0) | 3,5 (1,0–7,0) |
| all / measured / verified_only | 1885,6 (1601,0–2000,0) | 177,0 (155,0–198,0) | 103,6 (102,0–105,0) | 3,7 (2,0–5,0) | 1,7 (1,5–1,9) | 2,2 (1,0–5,0) | 152,6 (145,0–165,0) | 3,6 (1,0–7,0) |
| all / measured_smoothed / llm_nonmatch | 1875,3 (1317,0–2000,0) | 180,3 (124,0–200,0) | 102,2 (99,0–105,0) | 3,3 (2,0–7,0) | 1,8 (1,3–2,0) | 8,2 (2,0–15,0) | 158,6 (148,0–172,0) | 7,4 (3,0–14,0) |
| all / measured_smoothed / verified_only | 1875,5 (1365,0–2000,0) | 183,7 (130,0–215,0) | 102,7 (99,0–105,0) | 3,6 (2,0–5,0) | 1,8 (1,3–2,1) | 6,8 (3,0–13,0) | 158,1 (147,0–171,0) | 7,4 (3,0–14,0) |
| all / fixture_replay / llm_nonmatch | 1453,8 (1406,0–1500,0) | 128,9 (124,0–134,0) | 96,0 | 4,0 | 1,3 (1,3–1,4) | 7,1 (6,0–10,0) | 312,5 (310,0–316,0) | 112,7 (111,0–114,0) |
| all / fixture_replay / verified_only | 1426,8 (1387,0–1467,0) | 128,4 (125,0–133,0) | 96,0 | 4,0 | 1,3 (1,3–1,4) | 7,2 (7,0–9,0) | 312,9 (311,0–316,0) | 112,9 (112,0–114,0) |
| below_cut / thesis_bands / llm_nonmatch | 1350,0 (1000,0–2000,0) | 135,0 (100,0–200,0) | 9,6 (5,0–13,0) | 9,6 (5,0–13,0) | 14,4 (10,0–20,0) | 68,0 (42,0–91,0) | 64,3 (47,0–87,0) | 0,0 |
| below_cut / thesis_bands / verified_only | 1300,0 (1000,0–2000,0) | 130,0 (100,0–200,0) | 9,1 (5,0–12,0) | 9,1 (5,0–12,0) | 14,5 (10,0–20,0) | 68,8 (40,0–105,0) | 64,1 (38,0–104,0) | 0,0 |
| below_cut / measured / llm_nonmatch | 1176,4 (779,0–1608,0) | 104,9 (66,0–145,0) | 11,0 (8,0–14,0) | 11,0 (8,0–14,0) | 9,5 (8,2–11,4) | 5,8 (2,0–9,0) | 1,1 (0,0–4,0) | 0,0 |
| below_cut / measured / verified_only | 1205,8 (772,0–1631,0) | 106,3 (65,0–135,0) | 10,6 (8,0–13,0) | 10,6 (8,0–13,0) | 10,1 (6,5–13,2) | 5,7 (2,0–9,0) | 1,1 (0,0–4,0) | 0,0 |
| below_cut / measured_smoothed / llm_nonmatch | 1282,4 (775,0–1787,0) | 118,2 (75,0–166,0) | 10,6 (8,0–13,0) | 10,6 (8,0–13,0) | 11,3 (7,0–16,3) | 12,1 (2,0–21,0) | 4,1 (0,0–11,0) | 0,0 |
| below_cut / measured_smoothed / verified_only | 1331,7 (775,0–1964,0) | 121,0 (71,0–166,0) | 10,4 (7,0–13,0) | 10,4 (7,0–13,0) | 11,7 (7,0–16,6) | 13,6 (2,0–24,0) | 4,1 (0,0–11,0) | 0,0 |
| below_cut / fixture_replay / llm_nonmatch | 926,1 (831,0–1000,0) | 85,4 (77,0–95,0) | 6,1 (6,0–7,0) | 6,1 (6,0–7,0) | 14,0 (12,8–15,8) | 13,2 (9,0–17,0) | 7,1 (2,0–13,0) | 0,0 |
| below_cut / fixture_replay / verified_only | 918,5 (822,0–1000,0) | 86,2 (78,0–94,0) | 6,1 (6,0–7,0) | 6,1 (6,0–7,0) | 14,2 (12,0–15,7) | 13,3 (11,0–17,0) | 7,1 (2,0–13,0) | 0,0 |

### Fila × bases a orçamento de LLM igual (abaixo do corte, `run`)

| pool / revisor | revisões | fila | (a) nota | (b1) FS compostos | (c) aleatória | FP da fila | FP da nota |
|---|---|---|---|---|---|---|---|
| all / thesis_bands | 2000 | 14,4 (13,0–18,0) | 12,0 (10,0–14,0) | 15,9 (13,0–18,0) | 0,8 (0,0–2,0) | 94,9 (78,0–107,0) | 50,3 (44,0–64,0) |
| all / measured | 1851 (1278–2000) | 14,5 (13,0–16,0) | 12,5 (11,0–14,0) | 19,5 (14,0–23,0) | 0,8 (0,0–2,0) | 5,7 (2,0–10,0) | 11,2 (6,0–17,0) |
| all / measured_smoothed | 1875 (1317–2000) | 14,0 (12,0–16,0) | 12,1 (10,0–14,0) | 19,4 (14,0–23,0) | 0,8 (0,0–2,0) | 15,6 (9,0–26,0) | 19,4 (14,0–27,0) |
| all / fixture_replay | 1454 (1406–1500) | 11,0 | 10,0 | 11,0 | 0,1 (0,0–1,0) | 119,8 (118,0–121,0) | 26,7 (24,0–30,0) |
| below_cut / thesis_bands | 1350 (1000–2000) | 7,8 (5,0–10,0) | 11,0 (9,0–14,0) | 0,9 (0,0–1,0) | 0,3 (0,0–1,0) | 68,0 (42,0–91,0) | 37,1 (27,0–52,0) |
| below_cut / measured | 1176 (779–1608) | 8,8 (7,0–11,0) | 11,1 (9,0–14,0) | 0,9 (0,0–1,0) | 0,4 (0,0–1,0) | 5,8 (2,0–9,0) | 10,8 (6,0–17,0) |
| below_cut / measured_smoothed | 1282 (775–1787) | 8,6 (7,0–10,0) | 11,1 (8,0–13,0) | 0,9 (0,0–1,0) | 0,5 (0,0–1,0) | 12,1 (2,0–21,0) | 16,2 (9,0–23,0) |
| below_cut / fixture_replay | 926 (831–1000) | 6,0 | 10,0 | 0,0 | 0,0 | 13,2 (9,0–17,0) | 22,0 (21,0–23,0) |

## Discussão

<!-- DISCUSSAO:INICIO -->
Todos os números abaixo vêm das tabelas acima (média entre as 10 sementes; mínimo–máximo entre parênteses). “Abaixo do corte” são os 36 pares com `PAR = 2` abaixo do corte da tese. Valores com `[n=k]` têm viés de sobrevivência: só k sementes chegaram à marca ou ao orçamento, e a média parcial não é comparável com a das demais.

**O que os números mostram**

- **Esforço humano (a pergunta do usuário).** Com `thesis_bands`, a fila não encurta o trabalho humano em relação às bases. No `below_cut`, a orçamento humano igual, H2 verifica menos pares abaixo do corte que H0 e H1 (B=50: 8,3 contra 10,0 e 10,9; B=100: 9,5 contra 12,0 e 13,2). H1 só supera H0 a partir de B=200 (17,6 contra 13,0) e usa 7084 revisões de LLM para isso. Com revisor perfeito, H1 e H2 verificam k=10 pares abaixo do corte em 10 decisões contra 24 de H0: o ganho existe quando o LLM não erra, e some com as taxas de erro da vigilância (k=10: H2 45, H1 30, H0 24).
- **Orçamento humano por rodada.** No `below_cut` com `thesis_bands`, 25, 50 e 100 por rodada dão 7,7 (5–10), 9,6 (5–13) e 10,3 (5–14) pares abaixo do corte verificados, com 8,8, 14,4 e 21,6 decisões por par verificado: mais orçamento verifica um pouco mais e custa proporcionalmente mais por par. No `all` o orçamento muda quem fica sem verificação: com 50 por rodada sobram 246,9 itens `over_budget` e 37,5 FP não verificados; com 100, 91,0 e 4,2. Em B=200 no `all`, só com 100 por rodada H2 se aproxima das bases (11,3 contra 12,0 de H0 e 12,7 de H1).
- **`negative_labels`.** `llm_nonmatch` e `verified_only` dão resultados parecidos, com intervalos sobrepostos (por exemplo, `all` com `thesis_bands` e 100 por rodada: 13,9 (11–16) e 14,6 (13–17); `below_cut` com 50 por rodada: 9,6 e 9,1). Com `verified_only` a recalibração é mais vezes `skipped_insufficient` (`all`, revisor perfeito, 50 por rodada: 2,00 por run, contra 0 com `llm_nonmatch`), ou seja, faltam negativos verificados; isso não se converteu em perda visível de pares verificados nestes dados.
- **Regra de parada por posterior.** Com `thesis_bands`, a parada por `posterior` não disparou em nenhuma rodada de nenhuma configuração (0,00 por run em todas as linhas); com revisor perfeito disparou (2,60 e 3,00 por run no `all` e no `below_cut`). A taxa de FP da faixa `low` na tabela de taxas é 8,4%, acima de τ = 0,05: o LLM simulado marca MATCH com frequência maior que τ mesmo onde não há pares, então P(θ > τ) não cai abaixo de δ. As rodadas terminam por `max_round_reviews`, e o run, por `sterile_round` ou `max_reviews`.
- **Sensibilidade.** Com `tau=0,01` e `window=300` (humano, `below_cut`, `thesis_bands`) a parada por posterior também não disparou (0,00); o que mudou foi a janela do Thompson (e o `tau` só entra nessa parada, que não disparou). Resultado: 8,6 (7–10) pares abaixo do corte em 1550 revisões (1000–2000), contra 7,8 (5–10) em 1350 na configuração padrão, com intervalos sobrepostos; a diferença não pode ser atribuída à parada.
- **v1 → v2 → v3, por run.** A v2 regrediu com revisor perfeito: encontrados no `all` de 226,2 (auto) para 139,2 e de 228,5 (humano) para 153,0, e a v3 volta a 229,6 e 231,8. Abaixo do corte (humano, perfeito): 20,5 → 17,8 → 21,9; no `below_cut` os pares passaram de 0,0 (v1) para 13,1 (v2) e 15,0 (v3).
- **A perda tardia v1 → v2 no `all` vem da troca de campos.** Abaixo do corte, `thesis_bands`, humano: v1 13,7 (12–16), v2 com sub-escores 10,5 (8–12), v2 com compostos (ablação) 14,2 (13–16) e v3 com compostos 14,4 (13–18). O resultado com compostos da v2 (14,2) é ≥ o da v1 (13,7), e a v3 não muda isso de forma material.
- **A orçamento igual (marcas), abaixo do corte.** `all`/`thesis_bands`: em 500 revisões, v1 5,8, v2 7,0, v3 7,0; em 1000, 10,1, 8,7 e 12,6; em 2000, 13,7, 10,5 e 14,4. `below_cut`/`thesis_bands`: em 500, 0,2, 6,6 e 6,6; em 1000, 1,0 `[n=2]`, 7,0 e 7,7. Comparar o resultado final de cada run (600 revisões na v1, 1300 na v2) não é comparar o mesmo orçamento; esta tabela é a comparação correta.
- **Parte do ganho abaixo do corte vem da 1ª rodada ordenada pela nota.** No `below_cut`/`thesis_bands`, de 500 a 1000 revisões a fila acrescenta +1,1 (6,6 → 7,7) e a nota +1,6 (8,8 → 10,4); com revisor perfeito, +1,8 `[n=9]` contra +2,0. As rodadas ≥ 2 acrescentam pouco ali. No `all` elas acrescentam mais que a nota (+5,6 contra +3,5 com `thesis_bands`; +6,2 contra +4,0 com revisor perfeito).
- **Anchor-Platt.** A tabela de `p0` mostra `stub` em 100% dos runs do `below_cut` e `anchor_platt` em 100% dos runs do `all`: a falha de calibração por âncoras (A⁺ = 9 < 30, que vem do log da execução) é do pool abaixo do corte; no extrato inteiro as âncoras bastam (A⁺ = 95, informado pela equipe, não calculado aqui).
- **Fila × bases a orçamento de LLM igual (abaixo do corte, `thesis_bands`, humano).** `all`: fila 14,4 (13–18), nota 12,0 (10–14), FS sobre compostos 15,9 (13–18); `below_cut`: fila 7,8 (5–10), nota 11,0 (9–14), FS 0,9 (0–1). FP no `below_cut`: fila 68,0 (42–91), nota 37,1 (27–52).
- **Exploração e modo automático.** O braço de exploração (10% das revisões) acha 0,1 par abaixo do corte por run em `thesis_bands`. No modo automático o aprendizado fica desligado nas quatro configurações (0,00; `skipped_low_conf_dominant`).

- **Revisor calibrado no Haiku real (replay): esforço humano.** No `below_cut`, a orçamento humano B=50, a fila (H2) verifica 6,1 (6,0–7,0) pares abaixo do corte, contra 10,0 de H0 e 11,0 de H1. A fila faz em média 85,4 (77–95) decisões no run, a 14,0 decisões por par verificado, e não chega a B=100 em nenhuma semente; para k=10 pares abaixo do corte, nenhuma semente da fila chegou lá, enquanto H0 precisa de 24 decisões (23–24) e H1 de 28 (28–28). Para k=5, H1 e H2 levam 5 decisões (5–5) e H0 6 (5–6). H1 fica em 11,0 verificados de B=50 a B=400 (`[n=7]` em B=400), contra 13,0 de H0 em B=400. No `all`, B=50: H0 5,0, H1 4,0, H2 4,0; B=100: H0 8,7, H1 8,0, H2 4,0.
- **Cobertura do fixture.** No `fixture_replay`, 88,9% das revisões da fila vêm do fixture no `all` e 53,8% no `below_cut` (o resto é o fallback por `measured_smoothed`); nas bases (nota, FS, aleatória e H1), 31,8% e 20,5%. Os resultados do replay, sobretudo os de H1 e os do `below_cut`, misturam, portanto, decisões reais e taxas.
- **FP: o replay fica acima do que as taxas medidas preveem.** A fila tem 119,8 (118–121) MATCH do LLM sem verdade no `all` com `fixture_replay`, contra 5,7 (2–10) com `measured`; a nota, 26,7 (24–30) contra 11,2 (6–17). No `below_cut`: fila 13,2 (9–17) contra 5,8 (2–9); nota 22,0 (21–23) contra 10,8 (6–17). (Hipótese não testada aqui: o fixture inclui, segundo o manifesto, os pares abaixo do corte em ordem decrescente de nota, que são os mais difíceis.)
- **Taxas medidas (`measured`, `measured_smoothed`).** Com FP de 0,0% a 1,5% nas faixas centrais (e `high` suavizada a 16,7%), a triagem do LLM aparece muito melhor que com `thesis_bands`: no `below_cut`, H1 verifica 20,8 (18–23) e 16,2 (13–20) pares abaixo do corte em B=50, contra 10,9 (8–14) com `thesis_bands` e 10,0 de H0; k=10 em 14 (12–17) e 17 (12–25) decisões, contra 30 (20–55) e 24. A fila não acompanha: 10,1 (8–13) e 9,3 (7–11) pares em B=50 (H0: 10,0), e k=10 em 31 (11–51) [0,80] e 36 (12–92) [0,70] decisões.
- **A orçamento de LLM igual (abaixo do corte, `below_cut`).** `fixture_replay`: fila 6,0, nota 10,0, FS 0,0; `measured`: fila 8,8 (7–11), nota 11,1 (9–14); `measured_smoothed`: fila 8,6 (7–10), nota 11,1 (8–13).
- **`negative_labels` nos revisores reais.** Sem diferença material: `below_cut`/`measured` 11,0 (8–14) com `llm_nonmatch` e 10,6 (8–13) com `verified_only`; `fixture_replay`, 6,1 (6,0–7,0) nos dois.

**O que os números não mostram**

- Não mostram desempenho com LLM real: o revisor é sintético, as taxas vêm de contagens pequenas da vigilância (a faixa `grey_low` tem só 2 pares verdadeiros e a `high` 0 não pares, o que, com Laplace, dá FN de 75,0% e FP de 50,0%), a faixa `low` herdou as de `grey_low` por suposição e a nota estava visível ao LLM na vigilância; aqui está oculta.
- O humano é perfeito e decide pelo `PAR`; erro humano, custo e ordem de trabalho reais não entram. A “decisão” de H0 e a de H2 custam o mesmo aqui; na prática, confirmar um MATCH já proposto pode custar menos que decidir do zero.
- A fila padrão (50 por rodada, no máximo 5 rodadas) faz no máximo 250 decisões; B=200 e B=400 só são alcançados por parte das sementes ou pelas configurações de 100 por rodada, e várias células de H2 têm `[n=k]` ou `—`. Não há comparação a B=400 com a fila.
- No `all`, parte dos pares verificados já seria aceita pelo ORL (211 acima do corte): decisões por verificado perto de 1,0 ali não indicam economia. O pool `below_cut` é o que corresponde ao trabalho pós-ORL.
- A recompensa do bandit, a regra de parada e a detecção de anomalia usam o MATCH bruto do LLM, não o MATCH verificado pelo humano (limitação conhecida da implementação). Isto afeta o que a fila persegue e explica por que os FP não saem do bandit.
- Há um só extrato e só 36 positivos abaixo do corte: diferenças de 1 a 3 pares são ruído, e as 10 sementes só variam o revisor e a ordem aleatória, não os dados.
- H1 é calculado sem limite de revisões de LLM (até 400 MATCH, no máximo 30.000 revisões); a fila tem `max_reviews=2000`. As comparações de esforço humano não são a orçamento de LLM igual: ver a tabela de revisões de LLM consumidas.
- No `fixture_replay`, o veredito de um par presente no fixture é fixo entre sementes: as 10 sementes só variam exploração, desempate, auditoria e fallback. Os intervalos mínimo–máximo do replay (por exemplo, 11,0 sem variação para H1 no `below_cut`) subestimam a incerteza do revisor real.
- As taxas medidas vêm de uma amostra pequena (`low` tem 3 pares verdadeiros, `high` tem 2 não pares); o replay cobre só parte dos pares que a fila e as bases revisam (53,8% e 20,5% no `below_cut`), e o resto é taxa suavizada. Não há medida do Haiku real para os pares que ficam fora do fixture.
- O replay mostra o que o fixture contém; não mostra o que o LLM faria nos pares não revisados.

**Conclusão.** Com as taxas de erro da vigilância (`thesis_bands`) e com o replay do Haiku real, a fila com portão revisto não reduz as decisões humanas por par verificado abaixo do corte em relação a revisar na ordem da nota: no `below_cut` com o replay, B=50 dá 6,1 pares verificados na fila contra 10,0 (H0) e 11,0 (H1), e a fila não chega a k=10. A triagem do LLM só compensa claramente com revisor perfeito (k=10 em 10 decisões contra 24) ou com as taxas medidas, que o replay não confirma: o replay tem FP muito acima do que essas taxas preveem (119,8 contra 5,7 no `all`) e H1 não passa de 11,0 verificados abaixo do corte até B=400. O método permanece experimental e não validado; o fixture cobre só parte dos pares revisados.
<!-- DISCUSSAO:FIM -->

## Reprodução

```powershell
$env:PYTHONPATH = 'src'; python notebooks/sim_fila_autocalibravel.py --input ../phd-research/data/COMPARADORSEMIDENT.csv --seeds 10 --max-reviews 2000 --scenarios perfect,thesis_bands --field-sets composites --negative-labels llm_nonmatch,verified_only --human-budgets 25,50,100 --out-csv docs/sim_fila_autocalibravel_resultados.csv --out-md docs/sim_fila_autocalibravel.md
$env:PYTHONPATH = 'src'; python notebooks/sim_fila_autocalibravel.py --input ../phd-research/data/COMPARADORSEMIDENT.csv --seeds 10 --max-reviews 2000 --scenarios measured,measured_smoothed,fixture_replay --field-sets composites --negative-labels llm_nonmatch,verified_only --human-budgets 50 --no-auto-block --no-sensitivity --append-csv --out-csv docs/sim_fila_autocalibravel_resultados.csv --out-md docs/sim_fila_autocalibravel.md
```

O cenário opcional `rates_csv` (`--rates-csv`, colunas `band, fp, fn, unsure`) não foi usado. O segundo comando acima acrescenta ao CSV (sem apagar as linhas anteriores) os revisores calibrados no Haiku real; ele precisa dos insumos locais em `data/llm_fixtures/` e do relatório `docs/llm_real_haiku_sem_nota.csv`.

Soma dos tempos das sessões (sem as bases): 1518 s. Nenhum identificador de par é gravado: só agregados.
