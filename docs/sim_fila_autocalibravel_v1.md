> Versão 1 do desenho (faixas por probabilidade, ordem aleatória dentro da faixa, 5 compostos); substituída pela v2 (docs/sim_fila_autocalibravel_v2.md) e pela v3 (docs/sim_fila_autocalibravel.md). Valores em marcas não atingidas por todas as sementes aparecem sem [n=k] (bug de agregação da v1; ver a tabela v1 × v2 da v2/v3). O comando de reprodução abaixo é histórico: não reproduz a v1 com o código atual e sobrescreveria artefatos atuais.

# Simulação offline da fila auto-calibrável sobre o extrato da tese

> **Aviso.** Simulação com revisor **sintético**, calibrado nas taxas de erro por faixa da vigilância, medidas com a **nota visível** ao LLM; a variante simulada aqui **oculta a nota**. **Não é uma avaliação com LLM real.** O método é **experimental**, não existe na tese nem na 0.6.1, e estes números não o validam: mostram apenas o que a mecânica da fila faz sob as suposições declaradas.

## Método e suposições

- Dados: extrato `COMPARADORSEMIDENT.csv`, lido no lugar com `load_comparador_csv(..., LoadConfig(require_par=False))`. `PAR` fica numa série separada, usada só como verdade (`PAR ∈ {1, 2}`) do revisor e do humano simulados; a sessão recebe o DataFrame **sem `PAR` e sem `TARGET`**.
- Partição conferida com `below_cutoff(df, expected=THESIS_PARTITION)`: 211 acima do corte, 61485 abaixo, 36 com `PAR = 2` abaixo (igual à da tese: sim).
- Revisor: `BandedOracle` roteia cada par, pela nota, para um `OracleReviewer` da faixa do perfil v3.1 (`low` [0, 5), `grey_low` [5, 6), `grey_mid` [6, 7), `grey_high` [7, 8), `near_high` [8, 9), `high` [9, ∞)). O `OracleReviewer` é determinístico por `(semente, pair_id)`: o mesmo par recebe o mesmo veredito na fila e nas bases (comparação pareada).
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
- Configuração: `SelfCalibratingConfig()` padrão (`tau=0,05`, `delta=0,05`, `window=100`, `band_edges=(0,5; 0,1; 0,01)`), `max_reviews=2000`, salvo `mode`. Sensibilidade: humano × `below_cut` × `thesis_bands` com `tau=0,01` e `window=300`.
- Grade: pool ∈ {`all`, `below_cut`} × modo ∈ {`human`, `auto`} × revisor ∈ {`perfect`, `thesis_bands`} × 10 sementes (1, 2, 3, 4, 5, 6, 7, 8, 9, 10). Valores: média (mínimo–máximo) entre sementes; `[n=k]` quando só k sementes atingiram a marca (a média é só sobre elas); “—” quando nenhuma atingiu.
- Linhas de base, com a **mesma realização do revisor**, mesma semente e mesmo orçamento: (a) nota decrescente; (b) Fellegi-Sunter decrescente (`fit_fellegi_sunter` sobre `discretize_fields`, sem rótulos); (c) ordem aleatória. Orçamento = nº de revisões do run (`run`) e as marcas 100, 250, 500, 1000 e 2000.
- “Encontrado” = MATCH do LLM ∧ verdade (no modo humano, equivale a confirmado). “FP” = MATCH do LLM sem verdade: no humano, enviado ao portão; no automático, aceito como `llm_match_unverified`. “Abaixo do corte” = par com `PAR = 2` abaixo do corte da tese (36 no extrato).

## Resultados

### Comportamento da fila

| configuração | revisões LLM | rodadas | decisões humanas | verdadeiros revisados | encontrados | abaixo do corte (de 36) | FP | UNSURE |
|---|---|---|---|---|---|---|---|---|
| all / auto / perfect | 1106,8 (1035,0–1249,0) | 3,1 (3,0–4,0) | 0,0 | 226,2 (224,0–229,0) | 226,2 (224,0–229,0) | 18,6 (16,0–20,0) | 0,0 | 0,0 |
| all / auto / thesis_bands | 550,0 (500,0–1000,0) | 1,1 (1,0–2,0) | 0,0 | 138,1 (123,0–218,0) | 89,6 (71,0–158,0) | 6,1 (4,0–10,0) | 21,1 (8,0–39,0) | 46,8 (36,0–87,0) |
| all / human / perfect | 1212,6 (1118,0–1473,0) | 3,6 (3,0–5,0) | 297,4 (287,0–319,0) | 228,5 (227,0–230,0) | 228,5 (227,0–230,0) | 20,5 (19,0–23,0) | 0,0 | 0,0 |
| all / human / thesis_bands | 2000,0 | 4,1 (4,0–5,0) | 493,1 (468,0–538,0) | 229,1 (224,0–231,0) | 160,7 (153,0–169,0) | 13,7 (12,0–16,0) | 107,0 (83,0–120,0) | 154,7 (139,0–174,0) |
| below_cut / auto / perfect | 169,0 (165,0–179,0) | 1,0 | 0,0 | 0,0 | 0,0 | 0,0 | 0,0 | 0,0 |
| below_cut / auto / thesis_bands | 500,0 | 1,0 | 0,0 | 0,2 (0,0–1,0) | 0,2 (0,0–1,0) | 0,2 (0,0–1,0) | 35,5 (24,0–46,0) | 39,0 (30,0–45,0) |
| below_cut / human / perfect | 169,0 (165,0–179,0) | 1,0 | 30,0 | 0,0 | 0,0 | 0,0 | 0,0 | 0,0 |
| below_cut / human / thesis_bands | 600,0 (500,0–1000,0) | 1,2 (1,0–2,0) | 125,7 (95,0–219,0) | 0,2 (0,0–1,0) | 0,2 (0,0–1,0) | 0,2 (0,0–1,0) | 43,5 (24,0–82,0) | 46,0 (30,0–84,0) |
| below_cut / human / thesis_bands [tau=0,01, window=300] | 600,0 (500,0–1000,0) | 1,2 (1,0–2,0) | 125,2 (95,0–215,0) | 0,2 (0,0–1,0) | 0,2 (0,0–1,0) | 0,2 (0,0–1,0) | 42,0 (25,0–69,0) | 47,0 (30,0–85,0) |

### Motivos de parada, término e recalibração

Contagem média por run (parada de rodada e status de recalibração) e fração de runs (término).

| configuração | término (fração de runs) | parada de rodada (média por run) | recalibração (média por run) | aprendizado ligado (fração) |
|---|---|---|---|---|
| all / auto / perfect | auto_anomaly: 0,00; max_reviews: 0,00; max_rounds: 0,00; sterile_round: 1,00 | max_round_reviews: 1,00; posterior: 2,10 | skipped_low_conf_dominant: 3,10 | 0,00 |
| all / auto / thesis_bands | auto_anomaly: 1,00; max_reviews: 0,00; max_rounds: 0,00; sterile_round: 0,00 | max_round_reviews: 1,10 | skipped_low_conf_dominant: 1,10 | 0,00 |
| all / human / perfect | auto_anomaly: 0,00; max_reviews: 0,00; max_rounds: 0,10; sterile_round: 0,90 | max_round_reviews: 1,80; posterior: 1,80 | fitted: 3,60 | 1,00 |
| all / human / thesis_bands | auto_anomaly: 0,00; max_reviews: 1,00; max_rounds: 0,00; sterile_round: 0,00 | max_reviews: 0,10; max_round_reviews: 3,90; posterior: 0,10 | fitted: 4,10 | 1,00 |
| below_cut / auto / perfect | auto_anomaly: 0,00; max_reviews: 0,00; max_rounds: 0,00; sterile_round: 1,00 | posterior: 1,00 | skipped_low_conf_dominant: 1,00 | 0,00 |
| below_cut / auto / thesis_bands | auto_anomaly: 0,00; max_reviews: 0,00; max_rounds: 0,00; sterile_round: 1,00 | max_round_reviews: 1,00 | skipped_low_conf_dominant: 1,00 | 0,00 |
| below_cut / human / perfect | auto_anomaly: 0,00; max_reviews: 0,00; max_rounds: 0,00; sterile_round: 1,00 | posterior: 1,00 | skipped_insufficient: 1,00 | 1,00 |
| below_cut / human / thesis_bands | auto_anomaly: 0,00; max_reviews: 0,00; max_rounds: 0,00; sterile_round: 1,00 | max_round_reviews: 1,20 | skipped_insufficient: 1,20 | 1,00 |
| below_cut / human / thesis_bands [tau=0,01, window=300] | auto_anomaly: 0,00; max_reviews: 0,00; max_rounds: 0,00; sterile_round: 1,00 | max_round_reviews: 1,20 | skipped_insufficient: 1,20 | 1,00 |

### Comparação com as bases a orçamento igual (`run`)

**Pares verdadeiros encontrados** (orçamento = nº de revisões da fila no run)

| configuração | revisões | fila | (a) nota | (b) FS | (c) aleatória |
|---|---|---|---|---|---|
| all / auto / perfect | 1107 (1035–1249) | 226,2 (224,0–229,0) | 168,0 (167,0–170,0) | 226,4 (225,0–228,0) | 3,3 (0,0–8,0) |
| all / auto / thesis_bands | 550 (500–1000) | 89,6 (71,0–158,0) | 95,8 (81,0–113,0) | 157,0 (148,0–168,0) | 1,3 (0,0–4,0) |
| all / human / perfect | 1213 (1118–1473) | 228,5 (227,0–230,0) | 169,4 (168,0–174,0) | 227,5 (226,0–229,0) | 4,0 (1,0–9,0) |
| all / human / thesis_bands | 2000 | 160,7 (153,0–169,0) | 131,6 (122,0–139,0) | 162,5 (155,0–172,0) | 6,0 (2,0–13,0) |
| below_cut / auto / perfect | 169 (165–179) | 0,0 | 13,0 | 0,0 | 0,0 |
| below_cut / auto / thesis_bands | 500 | 0,2 (0,0–1,0) | 8,8 (7,0–10,0) | 0,9 (0,0–1,0) | 0,2 (0,0–1,0) |
| below_cut / human / perfect | 169 (165–179) | 0,0 | 13,0 | 0,0 | 0,0 |
| below_cut / human / thesis_bands | 600 (500–1000) | 0,2 (0,0–1,0) | 9,1 (7,0–11,0) | 0,9 (0,0–1,0) | 0,2 (0,0–1,0) |
| below_cut / human / thesis_bands [tau=0,01, window=300] | 600 (500–1000) | 0,2 (0,0–1,0) | 9,1 (7,0–11,0) | 0,9 (0,0–1,0) | 0,2 (0,0–1,0) |

**Pares abaixo do corte encontrados (de 36)** (orçamento = nº de revisões da fila no run)

| configuração | revisões | fila | (a) nota | (b) FS | (c) aleatória |
|---|---|---|---|---|---|
| all / auto / perfect | 1107 (1035–1249) | 18,6 (16,0–20,0) | 17,0 | 19,9 (19,0–21,0) | 0,1 (0,0–1,0) |
| all / auto / thesis_bands | 550 (500–1000) | 6,1 (4,0–10,0) | 7,2 (5,0–10,0) | 11,2 (9,0–14,0) | 0,0 |
| all / human / perfect | 1213 (1118–1473) | 20,5 (19,0–23,0) | 17,0 | 20,6 (19,0–22,0) | 0,3 (0,0–1,0) |
| all / human / thesis_bands | 2000 | 13,7 (12,0–16,0) | 12,0 (10,0–14,0) | 15,9 (13,0–18,0) | 0,8 (0,0–2,0) |
| below_cut / auto / perfect | 169 (165–179) | 0,0 | 13,0 | 0,0 | 0,0 |
| below_cut / auto / thesis_bands | 500 | 0,2 (0,0–1,0) | 8,8 (7,0–10,0) | 0,9 (0,0–1,0) | 0,2 (0,0–1,0) |
| below_cut / human / perfect | 169 (165–179) | 0,0 | 13,0 | 0,0 | 0,0 |
| below_cut / human / thesis_bands | 600 (500–1000) | 0,2 (0,0–1,0) | 9,1 (7,0–11,0) | 0,9 (0,0–1,0) | 0,2 (0,0–1,0) |
| below_cut / human / thesis_bands [tau=0,01, window=300] | 600 (500–1000) | 0,2 (0,0–1,0) | 9,1 (7,0–11,0) | 0,9 (0,0–1,0) | 0,2 (0,0–1,0) |

**FP (MATCH do LLM sem verdade)** (orçamento = nº de revisões da fila no run)

| configuração | revisões | fila | (a) nota | (b) FS | (c) aleatória |
|---|---|---|---|---|---|
| all / auto / perfect | 1107 (1035–1249) | 0,0 | 0,0 | 0,0 | 0,0 |
| all / auto / thesis_bands | 550 (500–1000) | 21,1 (8,0–39,0) | 12,3 (6,0–26,0) | 20,7 (12,0–60,0) | 37,0 (27,0–60,0) |
| all / human / perfect | 1213 (1118–1473) | 0,0 | 0,0 | 0,0 | 0,0 |
| all / human / thesis_bands | 2000 | 107,0 (83,0–120,0) | 50,3 (44,0–64,0) | 124,2 (114,0–145,0) | 145,8 (129,0–163,0) |
| below_cut / auto / perfect | 169 (165–179) | 0,0 | 0,0 | 0,0 | 0,0 |
| below_cut / auto / thesis_bands | 500 | 35,5 (24,0–46,0) | 13,9 (11,0–18,0) | 26,5 (19,0–35,0) | 36,2 (29,0–48,0) |
| below_cut / human / perfect | 169 (165–179) | 0,0 | 0,0 | 0,0 | 0,0 |
| below_cut / human / thesis_bands | 600 (500–1000) | 43,5 (24,0–82,0) | 16,8 (11,0–31,0) | 28,9 (19,0–43,0) | 45,0 (29,0–80,0) |
| below_cut / human / thesis_bands [tau=0,01, window=300] | 600 (500–1000) | 42,0 (25,0–69,0) | 16,8 (11,0–31,0) | 28,9 (19,0–43,0) | 45,0 (29,0–80,0) |

### Curvas: encontrados por nº de revisões

As bases não dependem do modo (mesma ordem e mesmo revisor); aparecem uma vez por pool e revisor. A fila aparece nas marcas que alcançou.

**Verdadeiros encontrados**

| pool / revisor | método | 100 | 250 | 500 | 1000 | 2000 |
|---|---|---|---|---|---|---|
| all / perfect | fila auto-calibrável (auto) | 65,6 (60,0–70,0) | 123,9 (122,0–128,0) | 133,3 (130,0–137,0) | 225,5 (224,0–227,0) | — |
| all / thesis_bands | fila auto-calibrável (auto) | 35,0 (30,0–42,0) | 77,2 (65,0–85,0) | 82,7 (71,0–95,0) | 158,0 | — |
| all / perfect | fila auto-calibrável (human) | 65,6 (60,0–70,0) | 123,9 (122,0–128,0) | 133,3 (130,0–137,0) | 227,6 (226,0–230,0) | — |
| all / perfect | (a) nota decrescente | 98,0 | 133,0 | 145,0 | 166,6 (166,0–167,0) | 189,0 |
| all / perfect | (b) Fellegi-Sunter | 89,6 (89,0–90,0) | 200,2 (200,0–201,0) | 222,9 (222,0–223,0) | 226,1 (225,0–228,0) | 235,5 (233,0–237,0) |
| all / perfect | (c) aleatória | 0,3 (0,0–2,0) | 0,6 (0,0–3,0) | 1,5 (0,0–4,0) | 3,0 (0,0–8,0) | 7,9 (4,0–16,0) |
| all / thesis_bands | fila auto-calibrável (human) | 35,0 (30,0–42,0) | 77,2 (65,0–85,0) | 82,7 (71,0–95,0) | 154,3 (144,0–162,0) | 160,7 (153,0–169,0) |
| all / thesis_bands | (a) nota decrescente | 61,6 (55,0–66,0) | 84,5 (72,0–93,0) | 94,2 (81,0–102,0) | 112,4 (101,0–121,0) | 131,6 (122,0–139,0) |
| all / thesis_bands | (b) Fellegi-Sunter | 57,8 (52,0–64,0) | 140,3 (132,0–147,0) | 156,8 (148,0–166,0) | 159,1 (150,0–168,0) | 162,5 (155,0–172,0) |
| all / thesis_bands | (c) aleatória | 0,3 (0,0–2,0) | 0,6 (0,0–3,0) | 1,2 (0,0–4,0) | 2,4 (0,0–6,0) | 6,0 (2,0–13,0) |
| below_cut / perfect | fila auto-calibrável (auto) | 0,0 | — | — | — | — |
| below_cut / thesis_bands | fila auto-calibrável (auto) | 0,0 | 0,2 (0,0–1,0) | 0,2 (0,0–1,0) | — | — |
| below_cut / perfect | fila auto-calibrável (human) | 0,0 | — | — | — | — |
| below_cut / perfect | (a) nota decrescente | 12,0 | 13,0 | 15,0 | 17,0 | 19,0 |
| below_cut / perfect | (b) Fellegi-Sunter | 0,0 | 0,0 | 1,0 | 1,0 | 1,0 |
| below_cut / perfect | (c) aleatória | 0,0 | 0,0 | 0,2 (0,0–1,0) | 0,4 (0,0–1,0) | 1,3 (0,0–3,0) |
| below_cut / thesis_bands | fila auto-calibrável (human) | 0,0 | 0,2 (0,0–1,0) | 0,2 (0,0–1,0) | 1,0 | — |
| below_cut / thesis_bands | (a) nota decrescente | 6,0 (5,0–7,0) | 6,9 (5,0–8,0) | 8,8 (7,0–10,0) | 10,4 (8,0–12,0) | 12,0 (10,0–14,0) |
| below_cut / thesis_bands | (b) Fellegi-Sunter | 0,0 | 0,0 | 0,9 (0,0–1,0) | 0,9 (0,0–1,0) | 0,9 (0,0–1,0) |
| below_cut / thesis_bands | (c) aleatória | 0,0 | 0,0 | 0,2 (0,0–1,0) | 0,2 (0,0–1,0) | 0,7 (0,0–2,0) |

**Abaixo do corte encontrados (de 36)**

| pool / revisor | método | 100 | 250 | 500 | 1000 | 2000 |
|---|---|---|---|---|---|---|
| all / perfect | fila auto-calibrável (auto) | 4,5 (3,0–6,0) | 10,4 (10,0–11,0) | 12,1 (10,0–14,0) | 18,0 (16,0–20,0) | — |
| all / thesis_bands | fila auto-calibrável (auto) | 1,8 (0,0–4,0) | 5,0 (4,0–6,0) | 5,8 (4,0–7,0) | 10,0 | — |
| all / perfect | fila auto-calibrável (human) | 4,5 (3,0–6,0) | 10,4 (10,0–11,0) | 12,1 (10,0–14,0) | 20,0 (19,0–22,0) | — |
| all / perfect | (a) nota decrescente | 8,7 (8,0–9,0) | 12,0 | 13,0 | 17,0 | 19,0 |
| all / perfect | (b) Fellegi-Sunter | 6,0 | 7,2 (7,0–8,0) | 17,0 | 19,6 (18,0–21,0) | 28,5 (26,0–30,0) |
| all / perfect | (c) aleatória | 0,0 | 0,0 | 0,0 | 0,1 (0,0–1,0) | 0,9 (0,0–2,0) |
| all / thesis_bands | fila auto-calibrável (human) | 1,8 (0,0–4,0) | 5,0 (4,0–6,0) | 5,8 (4,0–7,0) | 10,1 (7,0–13,0) | 13,7 (12,0–16,0) |
| all / thesis_bands | (a) nota decrescente | 3,8 (3,0–5,0) | 6,0 (5,0–7,0) | 6,9 (5,0–8,0) | 10,4 (8,0–12,0) | 12,0 (10,0–14,0) |
| all / thesis_bands | (b) Fellegi-Sunter | 2,6 (1,0–4,0) | 3,7 (2,0–5,0) | 11,0 (9,0–12,0) | 12,9 (10,0–15,0) | 15,9 (13,0–18,0) |
| all / thesis_bands | (c) aleatória | 0,0 | 0,0 | 0,0 | 0,1 (0,0–1,0) | 0,8 (0,0–2,0) |
| below_cut / perfect | fila auto-calibrável (auto) | 0,0 | — | — | — | — |
| below_cut / thesis_bands | fila auto-calibrável (auto) | 0,0 | 0,2 (0,0–1,0) | 0,2 (0,0–1,0) | — | — |
| below_cut / perfect | fila auto-calibrável (human) | 0,0 | — | — | — | — |
| below_cut / perfect | (a) nota decrescente | 12,0 | 13,0 | 15,0 | 17,0 | 19,0 |
| below_cut / perfect | (b) Fellegi-Sunter | 0,0 | 0,0 | 1,0 | 1,0 | 1,0 |
| below_cut / perfect | (c) aleatória | 0,0 | 0,0 | 0,2 (0,0–1,0) | 0,4 (0,0–1,0) | 1,3 (0,0–3,0) |
| below_cut / thesis_bands | fila auto-calibrável (human) | 0,0 | 0,2 (0,0–1,0) | 0,2 (0,0–1,0) | 1,0 | — |
| below_cut / thesis_bands | (a) nota decrescente | 6,0 (5,0–7,0) | 6,9 (5,0–8,0) | 8,8 (7,0–10,0) | 10,4 (8,0–12,0) | 12,0 (10,0–14,0) |
| below_cut / thesis_bands | (b) Fellegi-Sunter | 0,0 | 0,0 | 0,9 (0,0–1,0) | 0,9 (0,0–1,0) | 0,9 (0,0–1,0) |
| below_cut / thesis_bands | (c) aleatória | 0,0 | 0,0 | 0,2 (0,0–1,0) | 0,2 (0,0–1,0) | 0,7 (0,0–2,0) |

### Revisões até achar k pares abaixo do corte

Dentro do orçamento do run; entre parênteses, mínimo–máximo das sementes que atingiram k. A coluna “atingiu” é a fração de sementes.

| configuração | método | k=5 [atingiu] | k=10 [atingiu] | k=20 [atingiu] |
|---|---|---|---|---|
| all / auto / perfect | fila auto-calibrável | 103 (70–133) [1,00] | 174 (150–181) [1,00] | 922 (763–1018) [n=4] [0,40] |
| all / auto / perfect | (a) nota decrescente | 43 (43–44) [1,00] | 129 (129–130) [1,00] | — (0) |
| all / auto / perfect | (b) Fellegi-Sunter | 63 (56–70) [1,00] | 312 (301–332) [1,00] | 882 (639–1056) [n=7] [0,70] |
| all / auto / perfect | (c) aleatória | — (0) | — (0) | — (0) |
| all / auto / thesis_bands | fila auto-calibrável | 211 (107–433) [n=9] [0,90] | 938 [n=1] [0,10] | — (0) |
| all / auto / thesis_bands | (a) nota decrescente | 146 (94–197) [1,00] | 732 [n=1] [0,10] | — (0) |
| all / auto / thesis_bands | (b) Fellegi-Sunter | 276 (238–329) [1,00] | 429 (359–497) [n=9] [0,90] | — (0) |
| all / auto / thesis_bands | (c) aleatória | — (0) | — (0) | — (0) |
| all / human / perfect | fila auto-calibrável | 103 (70–133) [1,00] | 174 (150–181) [1,00] | 958 (919–1008) [n=6] [0,60] |
| all / human / perfect | (a) nota decrescente | 43 (43–44) [1,00] | 129 (129–130) [1,00] | — (0) |
| all / human / perfect | (b) Fellegi-Sunter | 63 (56–70) [1,00] | 312 (301–332) [1,00] | 937 (639–1143) [n=9] [0,90] |
| all / human / perfect | (c) aleatória | — (0) | — (0) | — (0) |
| all / human / thesis_bands | fila auto-calibrável | 254 (107–638) [1,00] | 1011 (640–1707) [1,00] | — (0) |
| all / human / thesis_bands | (a) nota decrescente | 146 (94–197) [1,00] | 864 (540–1715) [1,00] | — (0) |
| all / human / thesis_bands | (b) Fellegi-Sunter | 276 (238–329) [1,00] | 453 (359–672) [1,00] | — (0) |
| all / human / thesis_bands | (c) aleatória | — (0) | — (0) | — (0) |
| below_cut / auto / perfect | fila auto-calibrável | — (0) | — (0) | — (0) |
| below_cut / auto / perfect | (a) nota decrescente | 6 (5–6) [1,00] | 24 (23–24) [1,00] | — (0) |
| below_cut / auto / perfect | (b) Fellegi-Sunter | — (0) | — (0) | — (0) |
| below_cut / auto / perfect | (c) aleatória | — (0) | — (0) | — (0) |
| below_cut / auto / thesis_bands | fila auto-calibrável | — (0) | — (0) | — (0) |
| below_cut / auto / thesis_bands | (a) nota decrescente | 41 (9–80) [1,00] | 409 (408–409) [n=3] [0,30] | — (0) |
| below_cut / auto / thesis_bands | (b) Fellegi-Sunter | — (0) | — (0) | — (0) |
| below_cut / auto / thesis_bands | (c) aleatória | — (0) | — (0) | — (0) |
| below_cut / human / perfect | fila auto-calibrável | — (0) | — (0) | — (0) |
| below_cut / human / perfect | (a) nota decrescente | 6 (5–6) [1,00] | 24 (23–24) [1,00] | — (0) |
| below_cut / human / perfect | (b) Fellegi-Sunter | — (0) | — (0) | — (0) |
| below_cut / human / perfect | (c) aleatória | — (0) | — (0) | — (0) |
| below_cut / human / thesis_bands | fila auto-calibrável | — (0) | — (0) | — (0) |
| below_cut / human / thesis_bands | (a) nota decrescente | 41 (9–80) [1,00] | 456 (408–599) [n=4] [0,40] | — (0) |
| below_cut / human / thesis_bands | (b) Fellegi-Sunter | — (0) | — (0) | — (0) |
| below_cut / human / thesis_bands | (c) aleatória | — (0) | — (0) | — (0) |
| below_cut / human / thesis_bands [tau=0,01, window=300] | fila auto-calibrável | — (0) | — (0) | — (0) |
| below_cut / human / thesis_bands [tau=0,01, window=300] | (a) nota decrescente | 41 (9–80) [1,00] | 456 (408–599) [n=4] [0,40] | — (0) |
| below_cut / human / thesis_bands [tau=0,01, window=300] | (b) Fellegi-Sunter | — (0) | — (0) | — (0) |
| below_cut / human / thesis_bands [tau=0,01, window=300] | (c) aleatória | — (0) | — (0) | — (0) |

## O que o aprendizado (A') faz no modo automático

| configuração | aprendizado ligado (fração de runs) | `llm_match_unverified` |
|---|---|---|
| all / auto / perfect | 0,00 | 226,2 (224,0–229,0) |
| all / auto / thesis_bands | 0,00 | 110,7 (91,0–197,0) |
| below_cut / auto / perfect | 0,00 | 0,0 |
| below_cut / auto / thesis_bands | 0,00 | 35,7 (24,0–46,0) |

## Discussão

<!-- DISCUSSAO:INICIO -->
Todos os números abaixo vêm das tabelas acima (média entre as 10 sementes; intervalo entre parênteses).

**O que os números mostram**

- **Abaixo do corte, a fila não achou os pares.** No pool `below_cut` (61.485 pares, 36 com `PAR = 2`), com revisor perfeito, a fila gastou 169 revisões (165–179) e encontrou 0 pares; parou por `posterior` na primeira rodada e a recalibração foi `skipped_insufficient` (holdout sem positivos suficientes). A nota decrescente, com o mesmo orçamento e o mesmo revisor, encontrou 13, e levou 6 revisões (5–6) para chegar a k=5. Com `thesis_bands`, a fila usou 600 revisões (500–1000) e encontrou 0,2 par, contra 9,1 da nota, 0,9 do Fellegi-Sunter e 0,2 da ordem aleatória; os FP enviados ao portão foram 43,5 (fila), 16,8 (nota), 28,9 (FS) e 45,0 (aleatória). A ordem dentro de cada faixa é aleatória por desenho (probabilidade de inclusão conhecida), e com 36 positivos entre 61.485 pares a taxa por faixa é baixa demais para o bandit achar algo: o comportamento se parece com o da ordem aleatória.
- **No pool completo, a fila fica no nível do Fellegi-Sunter, sem superá-lo de forma clara.** Com revisor perfeito e portão humano, a fila achou 20,5 pares abaixo do corte (19–23) em 1.213 revisões (1.118–1.473), contra 17,0 da nota e 20,6 do FS. Com `thesis_bands` e portão humano, em 2.000 revisões: 13,7 (12–16) da fila, 12,0 (10–14) da nota e 15,9 (13–18) do FS; os intervalos se sobrepõem. A fila fica ligeiramente acima da nota e ligeiramente abaixo do FS (o prior da própria fila), com intervalos sobrepostos.
- **Curva e tempo até k.** Nas marcas iniciais a fila é mais lenta que o FS: com `thesis_bands` e portão humano, 77,2 verdadeiros encontrados em 250 revisões, contra 140,3 do FS e 84,5 da nota. Para achar k=5 pares abaixo do corte, com `thesis_bands`, humano e pool `all`: 254 revisões (107–638) da fila, 146 (94–197) da nota e 276 (238–329) do FS; para k=10, 1.011 (640–1.707), 864 (540–1.715) e 453 (359–672). Com k=20, nenhum método chegou a 20 dentro do orçamento com `thesis_bands`.
- **FP.** Com `thesis_bands` e portão humano no pool `all`, o orçamento de 2.000 revisões produziu 107,0 MATCH do LLM sem verdade (83–120) na fila, 50,3 na nota, 124,2 no FS e 145,8 na ordem aleatória; a nota produz menos FP, provavelmente porque revisa antes os pares de nota alta (hipótese, não testada aqui), mas encontra menos pares abaixo do corte que a fila e o FS. No pool `below_cut`, a nota teve 9,1 pares encontrados para 16,8 FP (cerca de 35% de VPP do LLM como aceitador, com as taxas supostas), a fila 0,2 para 43,5.
- **Custo humano.** No modo humano, o portão recebeu 297,4 decisões (287–319) em 1.212,6 revisões com revisor perfeito e 493,1 (468–538) em 2.000 revisões com `thesis_bands` (pool `all`). Esses itens são MATCH, UNSURE e a auditoria de NONMATCH; o número é do desenho do portão e não mede economia de revisão manual.
- **Modo automático.** O aprendizado ficou **desligado em todas as quatro configurações automáticas** (`aprendizado ligado` = 0,00; recalibração `skipped_low_conf_dominant`), como esperado no extrato da tese: quase todos os pares estão abaixo do corte, portanto a região de baixa confiança passa de 50% e a ordem fica pelo prior Fellegi-Sunter. A recalibração só rodou (`fitted`) nos runs humanos do pool `all`. No pool `all` com `thesis_bands`, todos os runs terminaram por `auto_anomaly`; no automático com `thesis_bands` houve 110,7 `llm_match_unverified` (91–197) no pool `all` e 35,7 (24–46) no `below_cut`, com 0,2 par abaixo do corte (e nenhum no `perfect`) neste último. O A' (aprendizado) não contribui: o resultado do modo automático é o do prior mais a ordem aleatória por faixa.
- **Sensibilidade.** Com `tau=0,01` e `window=300` (humano, `below_cut`, `thesis_bands`) o resultado não mudou de forma material: 600 revisões (500–1000), 0,2 par encontrado, 42,0 FP (25–69).

**O que os números não mostram**

- Não mostram desempenho com LLM real: o revisor é sintético, as taxas vêm de contagens pequenas da vigilância (a faixa `grey_low` tem só 2 pares verdadeiros e a `high` 0 não pares, o que, com Laplace, dá FN de 75,0% e FP de 50,0%) e a faixa `low` herdou as de `grey_low` por suposição. Na vigilância a nota estava visível ao LLM; aqui está oculta, e o efeito disso nas taxas é desconhecido.
- O humano é perfeito e decide pelo `PAR`; erro humano, cansaço e custo de revisão não entram.
- Há um só extrato e só 36 positivos abaixo do corte: diferenças de 1 a 3 pares são ruído, e as 10 sementes só variam o revisor e a ordem aleatória, não os dados.
- “Encontrado” conta o MATCH do LLM com verdade; não mede a economia de revisão manual nem a utilidade clínica. A ordem da fila foi comparada a bases de mesmo orçamento, mas as bases (a) e (b) usam a nota e os campos sem a etapa de amostragem estratificada da fila, cuja finalidade (holdout com probabilidade de inclusão conhecida, para calibração) não aparece nestas métricas.
- No log da execução, a calibração por âncoras da 1ª rodada informou âncoras insuficientes para o extrato (A+=9, mínimo 30) e a sessão usou o fallback do plano (`predict_stub`); isso vem da mensagem de log da execução, não do CSV.

**Conclusão.** Nesta simulação, a fila auto-calibrável não superou a ordenação pela nota abaixo do corte da tese e ficou, no pool completo, no nível do Fellegi-Sunter; o ganho de aprendizado não aparece (a recalibração só ajustou o modelo no modo humano do pool `all`, sem vantagem visível sobre o prior). Isso não contradiz a conclusão da tese de que a automação não reduziu a revisão manual além da ordenação. O método permanece experimental e não validado.
<!-- DISCUSSAO:FIM -->

## Reprodução

```powershell
$env:PYTHONPATH = 'src'; python notebooks/sim_fila_autocalibravel.py --input ../phd-research/data/COMPARADORSEMIDENT.csv --seeds 10 --max-reviews 2000 --scenarios perfect,thesis_bands --out-csv docs/sim_fila_autocalibravel_resultados.csv --out-md docs/sim_fila_autocalibravel.md
```

Soma dos tempos das sessões (sem as bases): 169 s. Nenhum identificador de par é gravado: só agregados.
