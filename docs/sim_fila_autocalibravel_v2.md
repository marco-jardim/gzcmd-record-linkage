> Versão 2 do desenho (faixas por posto, topo para baixo, exploração, sub-escores do CdR); substituída pela v3 em docs/sim_fila_autocalibravel.md. Agregados da v2 em docs/sim_fila_autocalibravel_resultados_v2.csv. O comando de reprodução abaixo é histórico: não reproduz a v2 com o código atual e sobrescreveria artefatos atuais.

# Simulação offline da fila auto-calibrável sobre o extrato da tese

> **Aviso.** Simulação com revisor **sintético**, calibrado nas taxas de erro por faixa da vigilância, medidas com a **nota visível** ao LLM; a variante simulada aqui **oculta a nota**. **Não é uma avaliação com LLM real.** O método é **experimental**, não existe na tese nem na 0.6.1, e estes números não o validam: mostram apenas o que a mecânica da fila faz sob as suposições declaradas.

Esta é a **versão 2** do desenho (faixas por posto, consumo do topo para baixo, exploração aleatória, campos do CdR). A versão 1 (faixas por probabilidade, ordem aleatória dentro da faixa, 5 compostos) está em [sim_fila_autocalibravel_v1.md](sim_fila_autocalibravel_v1.md), com os agregados em `sim_fila_autocalibravel_resultados_v1.csv`.

## Método e suposições

- Dados: extrato `COMPARADORSEMIDENT.csv`, lido no lugar com `load_comparador_csv(..., LoadConfig(require_par=False))`. `PAR` fica numa série separada, usada só como verdade (`PAR ∈ {1, 2}`) do revisor e do humano simulados; a sessão recebe o DataFrame do extrato, com os 29 sub-escores do CdR, **sem `PAR` e sem `TARGET`**.
- Partição conferida com `below_cutoff(df, expected=THESIS_PARTITION)`: 211 acima do corte, 61485 abaixo, 36 com `PAR = 2` abaixo (igual à da tese: sim).
- Fila (v2): (A) a cada rodada os pares não revisados são ordenados por `p_cur` e divididos por posto em `band_rank_edges=(0,01; 0,05; 0,20)` (topo 1%, 1–5%, 5–20% e o resto); cada faixa é consumida **do topo para baixo**; um braço de **exploração** aleatório, fora do bandit, ocupa uma a cada `explore_every=10` revisões e não entra na parada nem na recompensa do bandit. (B) A auditoria de NONMATCH do portão humano usa os 5% superiores do posto (`audit_rank_fraction=0,05`, 30 itens). (C) O prior (Fellegi-Sunter por EM, sem rótulos) e o recalibrador usam os 29 sub-escores do CdR e `mae_missing` (`field_set="cdr_subscores"`); a ablação usa os 5 compostos (`field_set="composites"`).
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

- Humano simulado: perfeito; decide pelo `PAR` todos os itens do portão (MATCH, UNSURE e a auditoria de NONMATCH) e conta decisões humanas.
- Configuração: `SelfCalibratingConfig()` padrão (`tau=0,05`, `delta=0,05`, `window=100`, `band_rank_edges`, `explore_every=10`, `audit_rank_fraction=0,05`, `field_set=cdr_subscores`), `max_reviews=2000`, salvo `mode`. Sensibilidade: humano × `below_cut` × `thesis_bands` com `tau=0,01` e `window=300`.
- Grade: pool ∈ {`all`, `below_cut`} × modo ∈ {`human`, `auto`} × revisor ∈ {`perfect`, `thesis_bands`} × 10 sementes (1, 2, 3, 4, 5, 6, 7, 8, 9, 10). Ablação: humano × {`all`, `below_cut`} × `thesis_bands` com `field_set="composites"` (isola o efeito dos campos do CdR em relação às mudanças A e B). Valores: média (mínimo–máximo) entre sementes; `[n=k]` quando só k sementes atingiram a marca (a média é só sobre elas); “—” quando nenhuma atingiu.
- Linhas de base, com a **mesma realização do revisor**, mesma semente e mesmo orçamento: (a) nota decrescente; (b1) Fellegi-Sunter decrescente sobre os 5 compostos; (b2) Fellegi-Sunter decrescente sobre os sub-escores do CdR (`fit_fellegi_sunter` sobre `discretize_fields`, sem rótulos); (c) ordem aleatória. Orçamento = nº de revisões do run (`run`) e as marcas 100, 250, 500, 1000 e 2000.
- “Encontrado” = MATCH do LLM ∧ verdade (no modo humano, equivale a confirmado). “FP” = MATCH do LLM sem verdade: no humano, enviado ao portão; no automático, aceito como `llm_match_unverified`. “Abaixo do corte” = par com `PAR = 2` abaixo do corte da tese (36 no extrato).

## Resultados

### Comportamento da fila

| configuração | revisões LLM | rodadas | decisões humanas | verdadeiros revisados | encontrados | abaixo do corte (de 36) | FP | UNSURE |
|---|---|---|---|---|---|---|---|---|
| all / auto / perfect | 774,0 (766,0–792,0) | 2,0 | 0,0 | 139,2 (139,0–140,0) | 139,2 (139,0–140,0) | 13,2 (13,0–14,0) | 0,0 | 0,0 |
| all / auto / thesis_bands | 800,0 (500,0–1000,0) | 1,6 (1,0–2,0) | 0,0 | 137,9 (137,0–140,0) | 88,5 (76,0–97,0) | 7,0 (5,0–9,0) | 29,9 (14,0–46,0) | 60,2 (33,0–87,0) |
| all / human / perfect | 1809,6 (1518,0–2000,0) | 4,8 (4,0–5,0) | 297,0 (271,0–321,0) | 153,0 (150,0–171,0) | 153,0 (150,0–171,0) | 17,8 (17,0–19,0) | 0,0 | 0,0 |
| all / human / thesis_bands | 2000,0 | 4,0 | 459,3 (440,0–480,0) | 149,7 (149,0–151,0) | 98,7 (89,0–105,0) | 10,5 (8,0–12,0) | 90,9 (77,0–103,0) | 149,7 (134,0–169,0) |
| below_cut / auto / perfect | 411,0 (393,0–439,0) | 1,0 | 0,0 | 13,1 (13,0–14,0) | 13,1 (13,0–14,0) | 13,1 (13,0–14,0) | 0,0 | 0,0 |
| below_cut / auto / thesis_bands | 500,0 | 1,0 | 0,0 | 12,7 (12,0–13,0) | 6,6 (5,0–8,0) | 6,6 (5,0–8,0) | 24,6 (17,0–31,0) | 38,2 (32,0–45,0) |
| below_cut / human / perfect | 685,8 (662,0–732,0) | 2,0 | 73,1 (73,0–74,0) | 13,1 (13,0–14,0) | 13,1 (13,0–14,0) | 13,1 (13,0–14,0) | 0,0 | 0,0 |
| below_cut / human / thesis_bands | 1300,0 (1000,0–2000,0) | 2,6 (2,0–4,0) | 249,9 (176,0–403,0) | 13,4 (13,0–15,0) | 7,2 (5,0–9,0) | 7,2 (5,0–9,0) | 68,7 (46,0–122,0) | 96,0 (65,0–153,0) |
| below_cut / human / thesis_bands [tau=0,01, window=300] | 1150,0 (1000,0–2000,0) | 2,3 (2,0–4,0) | 218,4 (182,0–352,0) | 13,4 (13,0–15,0) | 7,3 (6,0–10,0) | 7,3 (6,0–10,0) | 59,5 (46,0–103,0) | 82,6 (66,0–119,0) |

### Motivos de parada, término e recalibração

Contagem média por run (parada de rodada e status de recalibração) e fração de runs (término).

| configuração | término (fração de runs) | parada de rodada (média por run) | recalibração (média por run) | aprendizado ligado (fração) |
|---|---|---|---|---|
| all / auto / perfect | sterile_round: 1,00 | max_round_reviews: 1,00; posterior: 1,00 | skipped_low_conf_dominant: 2,00 | 0,00 |
| all / auto / thesis_bands | auto_anomaly: 0,40; sterile_round: 0,60 | max_round_reviews: 1,60 | skipped_low_conf_dominant: 1,60 | 0,00 |
| all / human / perfect | max_reviews: 0,10; max_rounds: 0,70; sterile_round: 0,20 | max_reviews: 0,10; max_round_reviews: 1,10; posterior: 3,60 | fitted: 4,80 | 1,00 |
| all / human / thesis_bands | max_reviews: 1,00 | max_round_reviews: 4,00 | fitted: 4,00 | 1,00 |
| below_cut / auto / perfect | sterile_round: 1,00 | posterior: 1,00 | skipped_low_conf_dominant: 1,00 | 0,00 |
| below_cut / auto / thesis_bands | sterile_round: 1,00 | max_round_reviews: 1,00 | skipped_low_conf_dominant: 1,00 | 0,00 |
| below_cut / human / perfect | sterile_round: 1,00 | posterior: 2,00 | fitted: 2,00 | 1,00 |
| below_cut / human / thesis_bands | max_reviews: 0,30; sterile_round: 0,70 | max_round_reviews: 2,60 | fitted: 2,60 | 1,00 |
| below_cut / human / thesis_bands [tau=0,01, window=300] | max_reviews: 0,10; sterile_round: 0,90 | max_round_reviews: 2,30 | fitted: 2,30 | 1,00 |

### Comparação com as bases a orçamento igual (`run`)

**Pares verdadeiros encontrados** (orçamento = nº de revisões da fila no run)

| configuração | revisões | fila | (a) nota | (b1) FS compostos | (b2) FS CdR | (c) aleatória |
|---|---|---|---|---|---|---|
| all / auto / perfect | 774 (766–792) | 139,2 (139,0–140,0) | 154,6 (154,0–155,0) | 225,1 (224,0–226,0) | 14,2 (11,0–18,0) | 2,2 (0,0–5,0) |
| all / auto / thesis_bands | 800 (500–1000) | 88,5 (76,0–97,0) | 105,1 (89,0–121,0) | 158,2 (150,0–168,0) | 9,5 (6,0–15,0) | 1,9 (0,0–6,0) |
| all / human / perfect | 1810 (1518–2000) | 153,0 (150,0–171,0) | 181,9 (174,0–189,0) | 233,4 (230,0–237,0) | 32,5 (21,0–40,0) | 6,9 (3,0–14,0) |
| all / human / thesis_bands | 2000 | 98,7 (89,0–105,0) | 131,6 (122,0–139,0) | 162,5 (155,0–172,0) | 24,1 (20,0–30,0) | 6,0 (2,0–13,0) |
| below_cut / auto / perfect | 411 (393–439) | 13,1 (13,0–14,0) | 14,0 (13,0–15,0) | 1,0 | 1,0 (0,0–3,0) | 0,2 (0,0–1,0) |
| below_cut / auto / thesis_bands | 500 | 6,6 (5,0–8,0) | 8,8 (7,0–10,0) | 0,9 (0,0–1,0) | 0,7 (0,0–2,0) | 0,2 (0,0–1,0) |
| below_cut / human / perfect | 686 (662–732) | 13,1 (13,0–14,0) | 17,0 | 1,0 | 2,1 (1,0–4,0) | 0,3 (0,0–1,0) |
| below_cut / human / thesis_bands | 1300 (1000–2000) | 7,2 (5,0–9,0) | 10,9 (8,0–14,0) | 0,9 (0,0–1,0) | 1,6 (0,0–4,0) | 0,2 (0,0–1,0) |
| below_cut / human / thesis_bands [tau=0,01, window=300] | 1150 (1000–2000) | 7,3 (6,0–10,0) | 10,7 (9,0–14,0) | 0,9 (0,0–1,0) | 1,3 (0,0–3,0) | 0,2 (0,0–1,0) |

**Pares abaixo do corte encontrados (de 36)** (orçamento = nº de revisões da fila no run)

| configuração | revisões | fila | (a) nota | (b1) FS compostos | (b2) FS CdR | (c) aleatória |
|---|---|---|---|---|---|---|
| all / auto / perfect | 774 (766–792) | 13,2 (13,0–14,0) | 17,0 | 18,8 (18,0–20,0) | 1,9 (0,0–3,0) | 0,0 |
| all / auto / thesis_bands | 800 (500–1000) | 7,0 (5,0–9,0) | 9,0 (5,0–12,0) | 12,2 (9,0–15,0) | 1,3 (0,0–3,0) | 0,1 (0,0–1,0) |
| all / human / perfect | 1810 (1518–2000) | 17,8 (17,0–19,0) | 18,7 (17,0–19,0) | 26,4 (23,0–30,0) | 4,5 (3,0–7,0) | 0,8 (0,0–2,0) |
| all / human / thesis_bands | 2000 | 10,5 (8,0–12,0) | 12,0 (10,0–14,0) | 15,9 (13,0–18,0) | 3,3 (2,0–6,0) | 0,8 (0,0–2,0) |
| below_cut / auto / perfect | 411 (393–439) | 13,1 (13,0–14,0) | 14,0 (13,0–15,0) | 1,0 | 1,0 (0,0–3,0) | 0,2 (0,0–1,0) |
| below_cut / auto / thesis_bands | 500 | 6,6 (5,0–8,0) | 8,8 (7,0–10,0) | 0,9 (0,0–1,0) | 0,7 (0,0–2,0) | 0,2 (0,0–1,0) |
| below_cut / human / perfect | 686 (662–732) | 13,1 (13,0–14,0) | 17,0 | 1,0 | 2,1 (1,0–4,0) | 0,3 (0,0–1,0) |
| below_cut / human / thesis_bands | 1300 (1000–2000) | 7,2 (5,0–9,0) | 10,9 (8,0–14,0) | 0,9 (0,0–1,0) | 1,6 (0,0–4,0) | 0,2 (0,0–1,0) |
| below_cut / human / thesis_bands [tau=0,01, window=300] | 1150 (1000–2000) | 7,3 (6,0–10,0) | 10,7 (9,0–14,0) | 0,9 (0,0–1,0) | 1,3 (0,0–3,0) | 0,2 (0,0–1,0) |

**FP (MATCH do LLM sem verdade)** (orçamento = nº de revisões da fila no run)

| configuração | revisões | fila | (a) nota | (b1) FS compostos | (b2) FS CdR | (c) aleatória |
|---|---|---|---|---|---|---|
| all / auto / perfect | 774 (766–792) | 0,0 | 0,0 | 0,0 | 0,0 | 0,0 |
| all / auto / thesis_bands | 800 (500–1000) | 29,9 (14,0–46,0) | 18,5 (6,0–28,0) | 40,3 (12,0–66,0) | 28,7 (14,0–45,0) | 56,2 (27,0–81,0) |
| all / human / perfect | 1810 (1518–2000) | 0,0 | 0,0 | 0,0 | 0,0 | 0,0 |
| all / human / thesis_bands | 2000 | 90,9 (77,0–103,0) | 50,3 (44,0–64,0) | 124,2 (114,0–145,0) | 73,3 (56,0–92,0) | 145,8 (129,0–163,0) |
| below_cut / auto / perfect | 411 (393–439) | 0,0 | 0,0 | 0,0 | 0,0 | 0,0 |
| below_cut / auto / thesis_bands | 500 | 24,6 (17,0–31,0) | 13,9 (11,0–18,0) | 26,5 (19,0–35,0) | 19,9 (15,0–25,0) | 36,2 (29,0–48,0) |
| below_cut / human / perfect | 686 (662–732) | 0,0 | 0,0 | 0,0 | 0,0 | 0,0 |
| below_cut / human / thesis_bands | 1300 (1000–2000) | 68,7 (46,0–122,0) | 38,3 (27,0–61,0) | 55,5 (31,0–92,0) | 52,0 (30,0–100,0) | 95,0 (60,0–164,0) |
| below_cut / human / thesis_bands [tau=0,01, window=300] | 1150 (1000–2000) | 59,5 (46,0–103,0) | 33,0 (24,0–52,0) | 50,7 (31,0–92,0) | 45,7 (30,0–79,0) | 83,5 (60,0–164,0) |

### Curvas: encontrados por nº de revisões

As bases não dependem do modo (mesma ordem e mesmo revisor); aparecem uma vez por pool e revisor. A fila aparece nas marcas que alcançou.

**Verdadeiros encontrados**

| pool / revisor | método | 100 | 250 | 500 | 1000 | 2000 |
|---|---|---|---|---|---|---|
| all / perfect | fila auto-calibrável (auto) | 83,2 (82,0–84,0) | 127,0 | 139,1 (139,0–140,0) | — | — |
| all / thesis_bands | fila auto-calibrável (auto) | 47,2 (36,0–52,0) | 79,4 (68,0–87,0) | 88,5 (76,0–97,0) | 89,2 (76,0–97,0) [n=6] | — |
| all / perfect | fila auto-calibrável (human) | 83,2 (82,0–84,0) | 127,0 | 139,1 (139,0–140,0) | 149,1 (148,0–150,0) | 171,0 [n=1] |
| all / perfect | (a) nota decrescente | 98,0 | 133,0 | 145,0 | 166,6 (166,0–167,0) | 189,0 |
| all / perfect | (b1) FS sobre compostos | 89,6 (89,0–90,0) | 200,2 (200,0–201,0) | 222,9 (222,0–223,0) | 226,1 (225,0–228,0) | 235,5 (233,0–237,0) |
| all / perfect | (b2) FS sobre sub-escores do CdR | 2,2 (0,0–7,0) | 5,5 (3,0–8,0) | 10,1 (7,0–14,0) | 18,3 (15,0–24,0) | 36,3 (32,0–40,0) |
| all / perfect | (c) aleatória | 0,3 (0,0–2,0) | 0,6 (0,0–3,0) | 1,5 (0,0–4,0) | 3,0 (0,0–8,0) | 7,9 (4,0–16,0) |
| all / thesis_bands | fila auto-calibrável (human) | 47,2 (36,0–52,0) | 79,4 (68,0–87,0) | 88,5 (76,0–97,0) | 95,9 (84,0–103,0) | 98,7 (89,0–105,0) |
| all / thesis_bands | (a) nota decrescente | 61,6 (55,0–66,0) | 84,5 (72,0–93,0) | 94,2 (81,0–102,0) | 112,4 (101,0–121,0) | 131,6 (122,0–139,0) |
| all / thesis_bands | (b1) FS sobre compostos | 57,8 (52,0–64,0) | 140,3 (132,0–147,0) | 156,8 (148,0–166,0) | 159,1 (150,0–168,0) | 162,5 (155,0–172,0) |
| all / thesis_bands | (b2) FS sobre sub-escores do CdR | 1,3 (0,0–3,0) | 3,2 (1,0–7,0) | 6,4 (3,0–8,0) | 12,0 (9,0–15,0) | 24,1 (20,0–30,0) |
| all / thesis_bands | (c) aleatória | 0,3 (0,0–2,0) | 0,6 (0,0–3,0) | 1,2 (0,0–4,0) | 2,4 (0,0–6,0) | 6,0 (2,0–13,0) |
| below_cut / perfect | fila auto-calibrável (auto) | 10,0 | 12,0 | — | — | — |
| below_cut / thesis_bands | fila auto-calibrável (auto) | 4,7 (4,0–6,0) | 6,0 (5,0–7,0) | 6,6 (5,0–8,0) | — | — |
| below_cut / perfect | fila auto-calibrável (human) | 10,0 | 12,0 | 13,1 (13,0–14,0) | — | — |
| below_cut / perfect | (a) nota decrescente | 12,0 | 13,0 | 15,0 | 17,0 | 19,0 |
| below_cut / perfect | (b1) FS sobre compostos | 0,0 | 0,0 | 1,0 | 1,0 | 1,0 |
| below_cut / perfect | (b2) FS sobre sub-escores do CdR | 0,6 (0,0–1,0) | 1,0 (0,0–3,0) | 1,4 (0,0–3,0) | 2,5 (1,0–5,0) | 5,2 (3,0–8,0) |
| below_cut / perfect | (c) aleatória | 0,0 | 0,0 | 0,2 (0,0–1,0) | 0,4 (0,0–1,0) | 1,3 (0,0–3,0) |
| below_cut / thesis_bands | fila auto-calibrável (human) | 4,7 (4,0–6,0) | 6,0 (5,0–7,0) | 6,6 (5,0–8,0) | 7,0 (5,0–8,0) | 8,3 (8,0–9,0) [n=3] |
| below_cut / thesis_bands | (a) nota decrescente | 6,0 (5,0–7,0) | 6,9 (5,0–8,0) | 8,8 (7,0–10,0) | 10,4 (8,0–12,0) | 12,0 (10,0–14,0) |
| below_cut / thesis_bands | (b1) FS sobre compostos | 0,0 | 0,0 | 0,9 (0,0–1,0) | 0,9 (0,0–1,0) | 0,9 (0,0–1,0) |
| below_cut / thesis_bands | (b2) FS sobre sub-escores do CdR | 0,3 (0,0–1,0) | 0,5 (0,0–2,0) | 0,7 (0,0–2,0) | 1,3 (0,0–3,0) | 2,7 (2,0–4,0) |
| below_cut / thesis_bands | (c) aleatória | 0,0 | 0,0 | 0,2 (0,0–1,0) | 0,2 (0,0–1,0) | 0,7 (0,0–2,0) |

**Abaixo do corte encontrados (de 36)**

| pool / revisor | método | 100 | 250 | 500 | 1000 | 2000 |
|---|---|---|---|---|---|---|
| all / perfect | fila auto-calibrável (auto) | 5,0 | 12,0 | 13,1 (13,0–14,0) | — | — |
| all / thesis_bands | fila auto-calibrável (auto) | 2,1 (1,0–4,0) | 6,0 (5,0–7,0) | 7,0 (5,0–9,0) | 7,8 (7,0–9,0) [n=6] | — |
| all / perfect | fila auto-calibrável (human) | 5,0 | 12,0 | 13,1 (13,0–14,0) | 17,2 (17,0–18,0) | 19,0 [n=1] |
| all / perfect | (a) nota decrescente | 8,7 (8,0–9,0) | 12,0 | 13,0 | 17,0 | 19,0 |
| all / perfect | (b1) FS sobre compostos | 6,0 | 7,2 (7,0–8,0) | 17,0 | 19,6 (18,0–21,0) | 28,5 (26,0–30,0) |
| all / perfect | (b2) FS sobre sub-escores do CdR | 0,2 (0,0–1,0) | 0,9 (0,0–3,0) | 1,2 (0,0–3,0) | 2,7 (0,0–5,0) | 5,6 (3,0–8,0) |
| all / perfect | (c) aleatória | 0,0 | 0,0 | 0,0 | 0,1 (0,0–1,0) | 0,9 (0,0–2,0) |
| all / thesis_bands | fila auto-calibrável (human) | 2,1 (1,0–4,0) | 6,0 (5,0–7,0) | 7,0 (5,0–9,0) | 8,7 (6,0–11,0) | 10,5 (8,0–12,0) |
| all / thesis_bands | (a) nota decrescente | 3,8 (3,0–5,0) | 6,0 (5,0–7,0) | 6,9 (5,0–8,0) | 10,4 (8,0–12,0) | 12,0 (10,0–14,0) |
| all / thesis_bands | (b1) FS sobre compostos | 2,6 (1,0–4,0) | 3,7 (2,0–5,0) | 11,0 (9,0–12,0) | 12,9 (10,0–15,0) | 15,9 (13,0–18,0) |
| all / thesis_bands | (b2) FS sobre sub-escores do CdR | 0,1 (0,0–1,0) | 0,7 (0,0–2,0) | 0,8 (0,0–2,0) | 1,5 (0,0–3,0) | 3,3 (2,0–6,0) |
| all / thesis_bands | (c) aleatória | 0,0 | 0,0 | 0,0 | 0,1 (0,0–1,0) | 0,8 (0,0–2,0) |
| below_cut / perfect | fila auto-calibrável (auto) | 10,0 | 12,0 | — | — | — |
| below_cut / thesis_bands | fila auto-calibrável (auto) | 4,7 (4,0–6,0) | 6,0 (5,0–7,0) | 6,6 (5,0–8,0) | — | — |
| below_cut / perfect | fila auto-calibrável (human) | 10,0 | 12,0 | 13,1 (13,0–14,0) | — | — |
| below_cut / perfect | (a) nota decrescente | 12,0 | 13,0 | 15,0 | 17,0 | 19,0 |
| below_cut / perfect | (b1) FS sobre compostos | 0,0 | 0,0 | 1,0 | 1,0 | 1,0 |
| below_cut / perfect | (b2) FS sobre sub-escores do CdR | 0,6 (0,0–1,0) | 1,0 (0,0–3,0) | 1,4 (0,0–3,0) | 2,5 (1,0–5,0) | 5,2 (3,0–8,0) |
| below_cut / perfect | (c) aleatória | 0,0 | 0,0 | 0,2 (0,0–1,0) | 0,4 (0,0–1,0) | 1,3 (0,0–3,0) |
| below_cut / thesis_bands | fila auto-calibrável (human) | 4,7 (4,0–6,0) | 6,0 (5,0–7,0) | 6,6 (5,0–8,0) | 7,0 (5,0–8,0) | 8,3 (8,0–9,0) [n=3] |
| below_cut / thesis_bands | (a) nota decrescente | 6,0 (5,0–7,0) | 6,9 (5,0–8,0) | 8,8 (7,0–10,0) | 10,4 (8,0–12,0) | 12,0 (10,0–14,0) |
| below_cut / thesis_bands | (b1) FS sobre compostos | 0,0 | 0,0 | 0,9 (0,0–1,0) | 0,9 (0,0–1,0) | 0,9 (0,0–1,0) |
| below_cut / thesis_bands | (b2) FS sobre sub-escores do CdR | 0,3 (0,0–1,0) | 0,5 (0,0–2,0) | 0,7 (0,0–2,0) | 1,3 (0,0–3,0) | 2,7 (2,0–4,0) |
| below_cut / thesis_bands | (c) aleatória | 0,0 | 0,0 | 0,2 (0,0–1,0) | 0,2 (0,0–1,0) | 0,7 (0,0–2,0) |

### Revisões até achar k pares abaixo do corte

Dentro do orçamento do run; entre parênteses, mínimo–máximo das sementes que atingiram k. Entre colchetes, a fração de sementes que atingiu.

| configuração | método | k=5 | k=10 | k=20 |
|---|---|---|---|---|
| all / auto / perfect | fila auto-calibrável | 53 (51–55) [1,00] | 148 (147–149) [1,00] | — (0) |
| all / auto / perfect | (a) nota decrescente | 43 (43–44) [1,00] | 129 (129–130) [1,00] | — (0) |
| all / auto / perfect | (b1) FS sobre compostos | 63 (56–70) [1,00] | 312 (301–332) [1,00] | 639 (639–639) [0,10] |
| all / auto / perfect | (b2) FS sobre sub-escores do CdR | — (0) | — (0) | — (0) |
| all / auto / perfect | (c) aleatória | — (0) | — (0) | — (0) |
| all / auto / thesis_bands | fila auto-calibrável | 181 (119–248) [1,00] | — (0) | — (0) |
| all / auto / thesis_bands | (a) nota decrescente | 146 (94–197) [1,00] | 638 (540–737) [0,60] | — (0) |
| all / auto / thesis_bands | (b1) FS sobre compostos | 276 (238–329) [1,00] | 429 (359–497) [0,90] | — (0) |
| all / auto / thesis_bands | (b2) FS sobre sub-escores do CdR | — (0) | — (0) | — (0) |
| all / auto / thesis_bands | (c) aleatória | — (0) | — (0) | — (0) |
| all / human / perfect | fila auto-calibrável | 53 (51–55) [1,00] | 148 (147–149) [1,00] | — (0) |
| all / human / perfect | (a) nota decrescente | 43 (43–44) [1,00] | 129 (129–130) [1,00] | — (0) |
| all / human / perfect | (b1) FS sobre compostos | 63 (56–70) [1,00] | 312 (301–332) [1,00] | 964 (639–1206) [1,00] |
| all / human / perfect | (b2) FS sobre sub-escores do CdR | 1208 (864–1759) [0,40] | — (0) | — (0) |
| all / human / perfect | (c) aleatória | — (0) | — (0) | — (0) |
| all / human / thesis_bands | fila auto-calibrável | 181 (119–248) [1,00] | 854 (509–1307) [0,70] | — (0) |
| all / human / thesis_bands | (a) nota decrescente | 146 (94–197) [1,00] | 864 (540–1715) [1,00] | — (0) |
| all / human / thesis_bands | (b1) FS sobre compostos | 276 (238–329) [1,00] | 453 (359–672) [1,00] | — (0) |
| all / human / thesis_bands | (b2) FS sobre sub-escores do CdR | 1458 (1363–1552) [0,20] | — (0) | — (0) |
| all / human / thesis_bands | (c) aleatória | — (0) | — (0) | — (0) |
| below_cut / auto / perfect | fila auto-calibrável | 12 (8–15) [1,00] | 38 (35–41) [1,00] | — (0) |
| below_cut / auto / perfect | (a) nota decrescente | 6 (5–6) [1,00] | 24 (23–24) [1,00] | — (0) |
| below_cut / auto / perfect | (b1) FS sobre compostos | — (0) | — (0) | — (0) |
| below_cut / auto / perfect | (b2) FS sobre sub-escores do CdR | — (0) | — (0) | — (0) |
| below_cut / auto / perfect | (c) aleatória | — (0) | — (0) | — (0) |
| below_cut / auto / thesis_bands | fila auto-calibrável | 90 (21–196) [1,00] | — (0) | — (0) |
| below_cut / auto / thesis_bands | (a) nota decrescente | 41 (9–80) [1,00] | 409 (408–409) [0,30] | — (0) |
| below_cut / auto / thesis_bands | (b1) FS sobre compostos | — (0) | — (0) | — (0) |
| below_cut / auto / thesis_bands | (b2) FS sobre sub-escores do CdR | — (0) | — (0) | — (0) |
| below_cut / auto / thesis_bands | (c) aleatória | — (0) | — (0) | — (0) |
| below_cut / human / perfect | fila auto-calibrável | 12 (8–15) [1,00] | 38 (35–41) [1,00] | — (0) |
| below_cut / human / perfect | (a) nota decrescente | 6 (5–6) [1,00] | 24 (23–24) [1,00] | — (0) |
| below_cut / human / perfect | (b1) FS sobre compostos | — (0) | — (0) | — (0) |
| below_cut / human / perfect | (b2) FS sobre sub-escores do CdR | — (0) | — (0) | — (0) |
| below_cut / human / perfect | (c) aleatória | — (0) | — (0) | — (0) |
| below_cut / human / thesis_bands | fila auto-calibrável | 90 (21–196) [1,00] | — (0) | — (0) |
| below_cut / human / thesis_bands | (a) nota decrescente | 41 (9–80) [1,00] | 528 (408–603) [0,80] | — (0) |
| below_cut / human / thesis_bands | (b1) FS sobre compostos | — (0) | — (0) | — (0) |
| below_cut / human / thesis_bands | (b2) FS sobre sub-escores do CdR | — (0) | — (0) | — (0) |
| below_cut / human / thesis_bands | (c) aleatória | — (0) | — (0) | — (0) |
| below_cut / human / thesis_bands [tau=0,01, window=300] | fila auto-calibrável | 90 (21–196) [1,00] | 1037 (1037–1037) [0,10] | — (0) |
| below_cut / human / thesis_bands [tau=0,01, window=300] | (a) nota decrescente | 41 (9–80) [1,00] | 528 (408–603) [0,80] | — (0) |
| below_cut / human / thesis_bands [tau=0,01, window=300] | (b1) FS sobre compostos | — (0) | — (0) | — (0) |
| below_cut / human / thesis_bands [tau=0,01, window=300] | (b2) FS sobre sub-escores do CdR | — (0) | — (0) | — (0) |
| below_cut / human / thesis_bands [tau=0,01, window=300] | (c) aleatória | — (0) | — (0) | — (0) |

### Braço de exploração

Revisões do braço de exploração (fora do bandit) e o que elas encontraram dentro do run.

| configuração | revisões LLM | exploração: revisões | exploração: encontrados | exploração: abaixo do corte |
|---|---|---|---|---|
| all / auto / perfect | 774,0 (766,0–792,0) | 76,8 (76,0–79,0) | 0,2 (0,0–1,0) | 0,2 (0,0–1,0) |
| all / auto / thesis_bands | 800,0 (500,0–1000,0) | 80,0 (50,0–100,0) | 0,1 (0,0–1,0) | 0,1 (0,0–1,0) |
| all / human / perfect | 1809,6 (1518,0–2000,0) | 180,5 (151,0–200,0) | 0,5 (0,0–1,0) | 0,3 (0,0–1,0) |
| all / human / thesis_bands | 2000,0 | 200,0 | 0,3 (0,0–1,0) | 0,1 (0,0–1,0) |
| below_cut / auto / perfect | 411,0 (393,0–439,0) | 40,6 (39,0–43,0) | 0,1 (0,0–1,0) | 0,1 (0,0–1,0) |
| below_cut / auto / thesis_bands | 500,0 | 50,0 | 0,1 (0,0–1,0) | 0,1 (0,0–1,0) |
| below_cut / human / perfect | 685,8 (662,0–732,0) | 68,2 (66,0–73,0) | 0,1 (0,0–1,0) | 0,1 (0,0–1,0) |
| below_cut / human / thesis_bands | 1300,0 (1000,0–2000,0) | 130,0 (100,0–200,0) | 0,1 (0,0–1,0) | 0,1 (0,0–1,0) |
| below_cut / human / thesis_bands [tau=0,01, window=300] | 1150,0 (1000,0–2000,0) | 115,0 (100,0–200,0) | 0,1 (0,0–1,0) | 0,1 (0,0–1,0) |

## O que o aprendizado (A') faz no modo automático

| configuração | aprendizado ligado (fração de runs) | recalibração (média por run) | `llm_match_unverified` |
|---|---|---|---|
| all / auto / perfect | 0,00 | skipped_low_conf_dominant: 2,00 | 139,2 (139,0–140,0) |
| all / auto / thesis_bands | 0,00 | skipped_low_conf_dominant: 1,60 | 118,4 (102,0–143,0) |
| below_cut / auto / perfect | 0,00 | skipped_low_conf_dominant: 1,00 | 13,1 (13,0–14,0) |
| below_cut / auto / thesis_bands | 0,00 | skipped_low_conf_dominant: 1,00 | 31,2 (24,0–38,0) |

## v1 × v2

Mesma grade, mesmas sementes e mesmo revisor. Cada célula traz **v1 → v2** da fila (campos do CdR e demais mudanças A+B+C na v2). Só as configurações presentes nos dois CSVs. O `[n=k]` da v1 usa o n de sementes da configuração (recalculado ao ler o CSV da v1).

| configuração | revisões | encontrados | abaixo do corte (de 36) | FP |
|---|---|---|---|---|
| all / auto / perfect | 1106,8 (1035,0–1249,0) → 774,0 (766,0–792,0) | 226,2 (224,0–229,0) → 139,2 (139,0–140,0) | 18,6 (16,0–20,0) → 13,2 (13,0–14,0) | 0,0 → 0,0 |
| all / auto / thesis_bands | 550,0 (500,0–1000,0) → 800,0 (500,0–1000,0) | 89,6 (71,0–158,0) → 88,5 (76,0–97,0) | 6,1 (4,0–10,0) → 7,0 (5,0–9,0) | 21,1 (8,0–39,0) → 29,9 (14,0–46,0) |
| all / human / perfect | 1212,6 (1118,0–1473,0) → 1809,6 (1518,0–2000,0) | 228,5 (227,0–230,0) → 153,0 (150,0–171,0) | 20,5 (19,0–23,0) → 17,8 (17,0–19,0) | 0,0 → 0,0 |
| all / human / thesis_bands | 2000,0 → 2000,0 | 160,7 (153,0–169,0) → 98,7 (89,0–105,0) | 13,7 (12,0–16,0) → 10,5 (8,0–12,0) | 107,0 (83,0–120,0) → 90,9 (77,0–103,0) |
| below_cut / auto / perfect | 169,0 (165,0–179,0) → 411,0 (393,0–439,0) | 0,0 → 13,1 (13,0–14,0) | 0,0 → 13,1 (13,0–14,0) | 0,0 → 0,0 |
| below_cut / auto / thesis_bands | 500,0 → 500,0 | 0,2 (0,0–1,0) → 6,6 (5,0–8,0) | 0,2 (0,0–1,0) → 6,6 (5,0–8,0) | 35,5 (24,0–46,0) → 24,6 (17,0–31,0) |
| below_cut / human / perfect | 169,0 (165,0–179,0) → 685,8 (662,0–732,0) | 0,0 → 13,1 (13,0–14,0) | 0,0 → 13,1 (13,0–14,0) | 0,0 → 0,0 |
| below_cut / human / thesis_bands | 600,0 (500,0–1000,0) → 1300,0 (1000,0–2000,0) | 0,2 (0,0–1,0) → 7,2 (5,0–9,0) | 0,2 (0,0–1,0) → 7,2 (5,0–9,0) | 43,5 (24,0–82,0) → 68,7 (46,0–122,0) |

Revisões até achar k pares abaixo do corte (v1 → v2), com a fração que atingiu entre colchetes:

| configuração | k=5 | k=10 |
|---|---|---|
| all / auto / perfect | 103 (70–133) [1,00] → 53 (51–55) [1,00] | 174 (150–181) [1,00] → 148 (147–149) [1,00] |
| all / auto / thesis_bands | 211 (107–433) [0,90] → 181 (119–248) [1,00] | 938 (938–938) [0,10] → — (0) |
| all / human / perfect | 103 (70–133) [1,00] → 53 (51–55) [1,00] | 174 (150–181) [1,00] → 148 (147–149) [1,00] |
| all / human / thesis_bands | 254 (107–638) [1,00] → 181 (119–248) [1,00] | 1011 (640–1707) [1,00] → 854 (509–1307) [0,70] |
| below_cut / auto / perfect | — (0) → 12 (8–15) [1,00] | — (0) → 38 (35–41) [1,00] |
| below_cut / auto / thesis_bands | — (0) → 90 (21–196) [1,00] | — (0) → — (0) |
| below_cut / human / perfect | — (0) → 12 (8–15) [1,00] | — (0) → 38 (35–41) [1,00] |
| below_cut / human / thesis_bands | — (0) → 90 (21–196) [1,00] | — (0) → — (0) |

Pares abaixo do corte encontrados nas marcas de revisões (v1 → v2), modo humano; mesmo nº de revisões nas duas versões:

| configuração | 100 | 250 | 500 | 1000 | 2000 |
|---|---|---|---|---|---|
| all / human / perfect | 4,5 (3,0–6,0) → 5,0 | 10,4 (10,0–11,0) → 12,0 | 12,1 (10,0–14,0) → 13,1 (13,0–14,0) | 20,0 (19,0–22,0) → 17,2 (17,0–18,0) | — → 19,0 [n=1] |
| all / human / thesis_bands | 1,8 (0,0–4,0) → 2,1 (1,0–4,0) | 5,0 (4,0–6,0) → 6,0 (5,0–7,0) | 5,8 (4,0–7,0) → 7,0 (5,0–9,0) | 10,1 (7,0–13,0) → 8,7 (6,0–11,0) | 13,7 (12,0–16,0) → 10,5 (8,0–12,0) |
| below_cut / human / perfect | 0,0 → 10,0 | — → 12,0 | — → 13,1 (13,0–14,0) | — → — | — → — |
| below_cut / human / thesis_bands | 0,0 → 4,7 (4,0–6,0) | 0,2 (0,0–1,0) → 6,0 (5,0–7,0) | 0,2 (0,0–1,0) → 6,6 (5,0–8,0) | 1,0 [n=2] → 7,0 (5,0–8,0) | — → 8,3 (8,0–9,0) [n=3] |

## Ablação: campos do CdR × 5 compostos

Humano × `thesis_bands`, com as mudanças A e B em ambas as linhas; só muda o conjunto de campos do prior e do recalibrador. “Prevalência FS” é a prevalência estimada pelo EM na 1ª rodada.

| pool / campos | revisões | encontrados | abaixo do corte (de 36) | FP | k=5 | k=10 | prevalência FS | recalibração (média por run) |
|---|---|---|---|---|---|---|---|---|
| all / sub-escores do CdR | 2000,0 | 98,7 (89,0–105,0) | 10,5 (8,0–12,0) | 90,9 (77,0–103,0) | 181 (119–248) [1,00] | 854 (509–1307) [0,70] | 0,493 | fitted: 4,00 |
| all / 5 compostos | 2000,0 | 162,6 (153,0–169,0) | 14,2 (13,0–16,0) | 97,5 (89,0–109,0) | 181 (119–248) [1,00] | 795 (647–1132) [1,00] | 0,007 | fitted: 4,00 |
| below_cut / sub-escores do CdR | 1300,0 (1000,0–2000,0) | 7,2 (5,0–9,0) | 7,2 (5,0–9,0) | 68,7 (46,0–122,0) | 90 (21–196) [1,00] | — (0) | 0,493 | fitted: 2,60 |
| below_cut / 5 compostos | 1400,0 (1000,0–2000,0) | 8,0 (5,0–10,0) | 8,0 (5,0–10,0) | 70,4 (46,0–91,0) | 90 (21–196) [1,00] | 918 (798–1039) [0,20] | 0,091 | fitted: 2,80 |

## Discussão

<!-- DISCUSSAO:INICIO -->
Todos os números abaixo vêm das tabelas acima (média entre as 10 sementes; intervalo mínimo–máximo entre parênteses). “Abaixo do corte” são os 36 pares com `PAR = 2` abaixo do corte da tese. As revisões por run diferem entre v1 e v2 (regras de parada diferentes); por isso a comparação justa entre versões é a tabela por marcas de revisões, não a coluna `run`.

**O que os números mostram**

- **Abaixo do corte, a v2 passou a achar pares; a v1 não achava.** No pool `below_cut` com `thesis_bands` e portão humano, a v2 encontrou 7,2 pares (5–9) em 1.300 revisões (1.000–2.000), contra 0,2 (0–1) em 600 na v1. Nas mesmas marcas de revisões, a v1 tinha 0,2 em 500 e 1,0 em 1.000 contra 6,6 (5–8) e 7,0 (5–8) da v2. Com revisor perfeito, a v2 achou 13,1 (13–14) em 686 revisões (662–732) e 10,0 já nas primeiras 100; a v1 parava em 169 revisões com 0 pares.
- **Mas a nota decrescente continua à frente no `below_cut`, com o mesmo orçamento e o mesmo revisor.** Com `thesis_bands`: 10,9 (8–14) da nota contra 7,2 (5–9) da fila em 1.300 revisões; nas marcas, 8,8 (7–10) contra 6,6 (5–8) em 500, 10,4 (8–12) contra 7,0 (5–8) em 1.000 e 12,0 (10–14) contra 8,3 (8–9) em 2.000 (esta última marca só com 3 sementes na fila, `[n=3]`). Com revisor perfeito: 17,0 da nota contra 13,1 da fila em 686 revisões. Para k=5, a nota precisa de 41 revisões (9–80) e a fila de 90 (21–196), ambas atingindo k=5 em todas as sementes; para k=10 a fila não chegou lá com `thesis_bands` (nem com `tau=0,01` e `window=300`, salvo 1 semente), enquanto a nota chegou em 528 (408–603) em 80% das sementes. Os FP também são maiores na fila: 68,7 (46–122) contra 38,3 (27–61) da nota, ou seja, um VPP aproximado do LLM como aceitador (encontrados ÷ (encontrados + FP), das médias) de cerca de 9,5% contra cerca de 22%.
- **Fellegi-Sunter sozinho, com qualquer conjunto de campos, é fraco abaixo do corte.** No `below_cut` com `thesis_bands`: 0,9 (0–1) para os 5 compostos (b1) e 1,6 (0–4) para os sub-escores do CdR (b2) em 1.300 revisões, contra 0,2 (0–1) da ordem aleatória. Os sub-escores do CdR tornam o FS pior no pool completo: em `all` com `thesis_bands` e portão humano, o (b2) encontra 24,1 (20–30) pares em 2.000 revisões contra 162,5 (155–172) do (b1). A prevalência estimada pelo EM com os sub-escores foi 0,493, e com os compostos 0,007 (pool `all`) e 0,091 (pool `below_cut`): o achado do implementador (prevalência de 0,49 com independência condicional violada) se reproduz.
- **Ablação (efeito de C isolado de A+B).** Com compostos em vez de sub-escores, no pool `all` com `thesis_bands` e portão humano, em 2.000 revisões: 14,2 (13–16) pares abaixo do corte contra 10,5 (8–12) com os sub-escores, e 162,6 (153–169) pares encontrados contra 98,7 (89–105); os intervalos não se sobrepõem. No `below_cut` a diferença é pequena e os intervalos se sobrepõem: 8,0 (5–10) em 1.400 revisões contra 7,2 (5–9) em 1.300. Os valores de k=5 são idênticos nas duas linhas (181 revisões, 119–248 no pool `all`; 90, 21–196 no `below_cut`): a 1ª rodada é igual nos dois conjuntos de campos, pois só usa a nota (`p0`); a diferença de campos só atua da rodada 2 em diante. Em v1 × v2 no pool `all`, a v2 está à frente até ~500 revisões (por exemplo 7,0 contra 5,8 em 500, `thesis_bands`) e atrás depois (10,5 contra 13,7 em 2.000); a ablação mostra que o conjunto de campos explica pelo menos parte dessa perda tardia, sem que a tabela permita isolar mais que isso.
- **Pool completo (`all`): a fila fica abaixo da nota e do FS sobre compostos.** Com `thesis_bands` e portão humano, em 2.000 revisões: 10,5 (8–12) pares abaixo do corte da fila, 12,0 (10–14) da nota, 15,9 (13–18) do (b1) e 3,3 (2–6) do (b2). Com revisor perfeito e 1.810 revisões (1.518–2.000): 17,8 (17–19), 18,7 (17–19), 26,4 (23–30) e 4,5 (3–7). Em FP, a fila (90,9; 77–103) fica entre a nota (50,3; 44–64) e o (b1) (124,2; 114–145).
- **Exploração.** O braço de exploração ocupou 10% das revisões (200,0 em `all`, 130,0 em `below_cut`, com `thesis_bands` e portão humano) e achou 0,1 (0–1) pares abaixo do corte em cada caso (0,3 encontrados no total em `all`): com 36 positivos em 61.485 pares, a exploração aleatória quase não acha nada, e o ganho da v2 vem de percorrer o topo das faixas, não da exploração.
- **Custo humano.** No portão humano: 459,3 decisões (440–480) em 2.000 revisões no pool `all` com `thesis_bands`, 297,0 (271–321) em 1.809,6 revisões com revisor perfeito e 249,9 (176–403) em 1.300 revisões no `below_cut` com `thesis_bands`. É o custo do desenho do portão (MATCH, UNSURE e auditoria de NONMATCH), não uma medida de economia de revisão manual.
- **Modo automático.** O aprendizado ficou desligado nas quatro configurações (`aprendizado ligado` = 0,00; recalibração `skipped_low_conf_dominant`), como esperado no extrato da tese. Mesmo sem aprendizado, a v2 acha 7,0 (5–9) pares abaixo do corte em 800 revisões (500–1.000) no pool `all` e 6,6 (5–8) em 500 no `below_cut`, com `thesis_bands`, porque o ganho vem da ordenação da rodada 1 e do consumo do topo para baixo. Os pares aceitos como `llm_match_unverified` foram 118,4 (102–143) no pool `all` e 31,2 (24–38) no `below_cut`.
- **Sensibilidade.** Com `tau=0,01` e `window=300` (humano, `below_cut`, `thesis_bands`): 7,3 (6–10) pares encontrados em 1.150 revisões (1.000–2.000), contra 7,2 em 1.300 com os valores padrão; a parada mais tardia não muda o resultado de forma material.

**O que os números não mostram**

- Não mostram desempenho com LLM real: o revisor é sintético, as taxas vêm de contagens pequenas da vigilância (a faixa `grey_low` tem só 2 pares verdadeiros e a `high` 0 não pares, o que, com Laplace, dá FN de 75,0% e FP de 50,0%) e a faixa `low` herdou as de `grey_low` por suposição. Na vigilância a nota estava visível ao LLM; aqui está oculta, e o efeito disso nas taxas é desconhecido.
- O humano é perfeito e decide pelo `PAR`; erro humano e custo de revisão não entram.
- Há um só extrato e só 36 positivos abaixo do corte: diferenças de 1 a 3 pares são ruído, e as 10 sementes só variam o revisor e a ordem aleatória, não os dados. As comparações v1 × v2 usam as mesmas sementes, mas a v2 muda três coisas ao mesmo tempo (A, B e C); só C foi isolada (ablação), e A e B não foram separadas entre si.
- A 1ª rodada usa a nota para `p0` (na execução, a calibração por âncoras informou âncoras insuficientes, A+=9 contra o mínimo de 30, e a sessão usou o fallback do plano; isso vem do log da execução, não do CSV). Parte do desempenho da v2 abaixo do corte vem, portanto, de uma ordenação que já usa a nota; a nota como base (a) ainda é melhor no `below_cut` nas mesmas marcas.
- “Encontrado” conta o MATCH do LLM com verdade; não mede a economia de revisão manual nem a utilidade clínica. As bases (a), (b1), (b2) e (c) não têm a amostragem de exploração nem o holdout ponderado da fila, cuja finalidade (calibração com probabilidade de inclusão conhecida) não aparece nestas métricas.
- A explicação de que o EM com sub-escores piora a ordenação por violar a independência condicional é compatível com a prevalência de 0,493 e com o desempenho fraco do (b2), mas esta simulação não a testa diretamente.

**Conclusão.** A v2 corrigiu o problema central da v1 no `below_cut` (de 0,2 para 7,2 pares abaixo do corte com `thesis_bands`), mas não superou a ordenação pela nota abaixo do corte da tese, e no pool completo ficou atrás da nota e do Fellegi-Sunter sobre compostos; os sub-escores do CdR como campos do prior pioraram o pool completo em relação aos compostos. Isso não contradiz a conclusão da tese de que a automação não reduziu a revisão manual além da ordenação. O método permanece experimental e não validado.
<!-- DISCUSSAO:FIM -->

## Reprodução

```powershell
$env:PYTHONPATH = 'src'; python notebooks/sim_fila_autocalibravel.py --input ../phd-research/data/COMPARADORSEMIDENT.csv --seeds 10 --max-reviews 2000 --scenarios perfect,thesis_bands --field-sets cdr_subscores --out-csv docs/sim_fila_autocalibravel_resultados.csv --out-md docs/sim_fila_autocalibravel.md
```

Soma dos tempos das sessões (sem as bases): 358 s. Nenhum identificador de par é gravado: só agregados.
