# Achados sobre filas de revisão e uso de LLM na zona cinza: base para atualizar a tese

**Data:** 2026-11-15.
**Status:** documento vivo; achados parciais marcados. Cada linha traz o seu status e a fonte. **Todos os experimentos planejados foram concluídos** (e, depois, refeitos contra a O1: seção J) (inclusive a replicação por passo de blocagem, seção E.5, e a simulação com taxas medidas do Haiku e replay do fixture, seção F), e o documento está **pronto para servir de base à atualização da tese**. As pendências restantes são as que exigem dados externos ou revisores reais (H12; replicação externa com outro extrato ou mais positivos; estudo com revisores humanos), listadas na seção I.

> **Aviso.** Nada do que está aqui foi avaliado na tese original. Os experimentos usam o método experimental `self_calibrating_queue` (0.7.0), que não existe na tese nem na 0.6.1, e o extrato único da tese (61.696 pares; 247 verdadeiros, 211 aceitos pelo corte e 36 abaixo dele). Os revisores humanos são **simulados e perfeitos**; os vereditos do LLM são, em geral, **uma única realização** (temperatura padrão, congelada num fixture). **Referência das comparações:** todos os experimentos das seções A a I compararam as estratégias com a ordem pela nota final do CdR (a O2 da tese, 12 verdadeiros com 96 revisões); a tese também tem a O1, a ordem pelo escore do OpenRecLink (19 com 96 revisões), que também não usa rótulos. A seção J refaz as comparações principais contra a O1 e muda várias leituras. Apenas agregados; nenhum identificador de par, nenhuma chave. Este documento não edita a tese: a seção H apenas sugere onde e como cada achado poderia entrar.

**Legenda de status**

- **Confirmado**: critério pré-registrado atendido ou medição direta.
- **Parcial/Exploratório**: indício sem pré-registro, ou dependente de escolhas do analista (inclui resultados "inconclusivos" por critério mal especificado).
- **Refutado**: critério pré-registrado violado, ou hipótese contrariada pela medição.
- **Pendente**: sem resultado ainda, ou não testável offline.

Onde um status "pela letra" difere do que o critério pretendia medir, isso está dito na linha.

## Tabela-resumo

| # | Achado | Status | Número-chave | Fonte | Impacto na tese |
|---|---|---|---|---|---|
| A1 | Taxas do Haiku 5.5 (prompt v4, dossiê sem saídas do motor e sem nota) em amostra estratificada | **Confirmado** (medição direta) | FP 0,4% [0,1–1,5] (2/470); FN 18,2% [13,9–23,5] (45/247); FN acima do corte 13,7% [9,7–19,0], abaixo 44,4% [29,5–60,4] (16/36) | `llm_real_haiku_sem_nota.md` | Estende o exame do dossiê (nota oculta); refina a leitura das taxas do revisor |
| A2 | VPP projetado do LLM abaixo do corte, nota oculta | **Parcial/Exploratório** (reescala e seleção diferem da tese) | 2,9% [0,8–12,1] (todas as faixas); 4,8% [1,5–20,9] (sem `low`) contra 16,0% [6,4–34,7] (4/25) da tese | idem | Reforça "o LLM não deve aceitar sozinho"; comparação só de ordem de grandeza |
| B1 | A fila auto-calibrável padrão não supera as bases a orçamento humano igual | **Confirmado** (direção; medição direta, sintética e real) | Real, ~127 decisões humanas, abaixo do corte (de 36): fila 4, H1 10, H0 9 | `sim_fila_autocalibravel.md`, `llm_real_fila_ponta_a_ponta.md` | Consistente com a tese (redução vem da ordenação); método não validado |
| B2 | Razões do fracasso D1–D6 | **Parcial/Exploratório** (D1, D2, D5 medidas; D3, D4, D6 são hipóteses) | D5: FN 44,4% abaixo do corte; D2: prevalência 0,06% | `engenharia_de_filas.md` §2 | Nota de limitação |
| C1 | LLM como desempate na zona cinza (reordenar a janela) | **Parcial/Exploratório** | L1 em B=400: +2 (W=500) a +6 (W=3000) de 36; IC exclui 0 em 4 de 48 combinações; misto em B ≤ 100 (−3 a +2) | `llm_real_desempate_zona_cinza.md` | Qualifica "usos subordinados não reduzem a carga" |
| D1 | H1–H12 sintético (39 células × 10 sementes) | Misto (ver §D) | 4 confirmadas, 6 refutadas pela letra, 1 inconclusiva, 1 não testável | `engenharia_de_filas.md` §6 | Princípios de desenho de filas; poucos pontos da tese |
| D2 | H8 real (bandit sobre faixas contíguas) | **Confirmado** | S5−S0 de −2,00 a 0,00 no real; premissa de monotonia verificada (2,76%; 0,12%; 0,11%; 0,01%) | `engenharia_de_filas_real_h8_h11.md` | Qualifica a menção à fila adaptativa (Thompson) em trabalhos futuros |
| D3 | H11 real (LLM audita FN do humano) | **Refutado** (0 de 16 células) | recuperação 26,1%–47,9%; precisão 2,6%–100% (100% só em B=100, com 0,3–0,8 par) | idem | Estende os cenários de sensibilidade humana |
| E1 | S2 (régua como offset, rótulos verificados) supera a régua no extrato real | **Parcial/Exploratório** (rebaixado após a replicação: o ganho grande dependia da régua plana) | Na amostra, B=800: 33 contra 17 (régua plana) e 20 contra 17 (calibrada); placebo: o ganho some. **Fora da amostra** (replicação por passo, régua calibrada, passos 9 e 11): +2, +4, +1 pares sobre a régua em B=100/200/400, nunca negativo; com a régua plana, +2, +4, +7 | `engenharia_de_filas.md` §6.7 e §8.1 | Estende a recomendação de ordenação; ganho pequeno e dependente da escala da régua |
| E2 | Perfil "dados ausentes" abaixo do corte | **Parcial/Exploratório** (sustentado só em parte: mãe ausente) | Na amostra: 12 em 102 (11,8%) na faixa de nota 5,47–6,90, contra 0,04% sem o perfil. Fora da amostra: mãe ausente com peso a favor de par nos 6 modelos de treino (+0,12 a +0,48); nome em nível máximo misto (−0,25 a +0,35); data de nascimento zerada **contra** (−0,47 a −1,19) | `engenharia_de_filas.md` §6.7 e §8.1 | Hipótese restrita à mãe ausente; precisa de outro extrato |
| E3 | LLM como atributo (S3) no real | **Parcial/Exploratório** (uma realização; IC degenerado) | B=400: S3 +16, S2 +6 sobre a régua (29 e 19 contra 13) | idem | Qualifica "LLM não reduz a revisão além da ordenação" |
| E4 | Replicação por passo de blocagem (R1–R4) | **Parcial/Exploratório** (R1–R4 todas inconclusivas; R4 por critério mal especificado) | Positivos por passo 6/9/11 = 4/7/25 (passo 6 não informativo). R1: T−S0 = +2, +4, +1 (exigido ≥ +3 em B=400); R2: O−S0 em B=400 = +0 (passo 9) e +1 (passo 11); R3: P×C em T = 19×19 (B=200) e 25×19 (B=400) | `engenharia_de_filas.md` §8.1; `engenharia_de_filas_replicacao_passo.csv` | Rebaixa a evidência de E1/E2; não confirma o ganho fora da amostra |
| F1 | Simulação com taxas medidas do Haiku e replay do fixture | **Parcial/Exploratório** (concluída; cobertura parcial do fixture) | Abaixo do corte, B=50: fila 6,1; H0 10,0; H1 11,0; a fila não chega a k=10 | `sim_fila_autocalibravel.md`, "Revisor calibrado no Haiku real" | Mesma leitura de B1, com revisor real |
| J1 | O1 (escore do OpenRecLink) contra O2 (nota) com humano sozinho; reproduz a tese em 96 revisões | **Confirmado** (medição direta) | 96 revisões: O1 19, O2 12. Com desempate aleatório: O1 = 8,7 [5–12], 19,0, 19,1, 19,8, 21,4, 25,0 e O2 = 10, 12, 12, 13, 13, 17 em B = 50, 96, 100, 200, 400, 800 | `comparacao_o1.md` §2, §4.a | A referência correta dos achados é a O1; ordem do arquivo favorece a O1 |
| J2 | S2 contra a O1 | **Parcial/Exploratório** (reduz E1) | Sem ganho em B = 96 a 400 (−7,1 a −1,6 com régua O1; −4,1 a −2,2 com a régua plana); em B = 800, +2,8 (régua O1) e +8,0 (régua plana), contra +16 sobre a O2 | `comparacao_o1.md` §4.b | O ganho de S2 sobre a referência real quase desaparece |
| J3 | Bandit sobre faixas de O1 contra O1 | **Confirmado** (direção) | S5−O1 = −5,6; −10,1; −9,1; +0,4; −0,6; −3,6 em B = 50, 96, 100, 200, 400, 800 | `comparacao_o1.md` §4.c | Mantém D2 |
| J4 | LLM como desempate sobre janelas da O1 (decisões reais; cobertura limitada) | **Parcial/Exploratório** (cobertura de 8% a 18% da janela) | −3,5 a +5,7 pares (até +6), inconsistente; −3,5 a +2,9 em B = 96; o fixture bruto (100% dos positivos cobertos) inflaria o ganho (+11,4 em B = 50) | `comparacao_o1.md` §4.d | Qualifica C1: ganho incerto com a O1 |
| J5 | Fila auto-calibrável existente contra a curva da O1 | **Confirmado** (direção) | Perde da O1 em B ≥ 100 em todo revisor (11,8 contra 19,1 em B=100, taxas medidas; S1 real 7,2); em B = 50 só o LLM perfeito supera (16,0 contra 8,7 a 12) | `comparacao_o1.md` §4.e | Reforça B1 com margem maior |
| J6 | H0 pela O1 na execução ponta a ponta (pool `all`, ~127 decisões) | **Confirmado** (medição direta) | 127 verificados e 0 abaixo do corte em B = 127; no pool abaixo do corte a O1 recupera 19,1 (B=100) | `comparacao_o1.md` §4.f | O pool `all` não serve para comparar a O1 abaixo do corte |
| J7 | Erro E1 da revisão factual (1,95 contra 17,00 do H8 real) | **Corrigido** | Em B=800: N1: S0 1,25 e S5 1,95; N2: S0 0 e S5 0,50; real: S0 17 e S5 15 | `comparacao_o1.md` §6; D (H8 real) | Corrige a leitura do ganho do bandit |
| I1 | H12: o LLM reduz o tempo por decisão humana | **Pendente** (não testável offline) | — | `engenharia_de_filas.md` §4 | Requer estudo com revisores reais |

## A. Medições do LLM real (Haiku 5.5, prompt v4, sem saídas do motor e sem nota)

**Fonte:** `docs/llm_real_haiku_sem_nota.md`. **Status: Confirmado** como medição direta; **A2 Parcial/Exploratório**.

**Protocolo.** Modelo `claude-haiku-5-5`, prompt `v4`, agente A, agente B e árbitro só na divergência, `max_tokens=2500`, sem temperatura. O dossiê tem só escores numéricos, sem `model_outputs` e sem `features.nota_final` (a nota está **oculta** ao LLM). Amostra estratificada (semente 20261115): todos os 247 pares verdadeiros (211 acima e 36 abaixo do corte) e até 100 não pares por faixa (470 no total), ou seja, 717 pares. Verdade: PAR ∈ {1, 2}. FP = MATCH entre não pares; FN = NONMATCH entre pares; UNSURE fica nos denominadores. IC de Wilson a 95%.

**Taxas totais.**

| medida | todos | acima do corte | abaixo do corte |
|---|---|---|---|
| Pares / não pares | 247 / 470 | 211 / 0 | 36 / 470 |
| FP | 2 de 470 = 0,4% [0,1–1,5] | n/d | 2 de 470 = 0,4% [0,1–1,5] |
| FN | 45 de 247 = 18,2% [13,9–23,5] | 29 de 211 = 13,7% [9,7–19,0] | 16 de 36 = 44,4% [29,5–60,4] |
| UNSURE | 7,0% [5,3–9,1] | 11,4% [7,8–16,4] | 5,1% [3,5–7,4] |
| Decisões com árbitro | 10,5% | 14,2% | 8,9% |

Os dois FP estão abaixo do corte (um em `grey_low`, um em `grey_high`).

**Por faixa (lado = todos).** Faixas pela nota v3.1: `low` [0, 5), `grey_low` [5, 6), `grey_mid` [6, 7), `grey_high` [7, 8), `near_high` [8, 9), `high` [9, ∞).

| faixa | pares | não pares | FP % [IC] | FN % [IC] | UNSURE % |
|---|--:|--:|---|---|--:|
| low | 3 | 100 | 0,0 [0,0–3,7] | 66,7 [20,8–93,9] (2/3) | 4,9 |
| grey_low | 17 | 100 | 1,0 [0,2–5,4] | 29,4 [13,3–53,1] (5/17) | 5,1 |
| grey_mid | 57 | 100 | 0,0 [0,0–3,7] | 7,0 [2,8–16,7] (4/57) | 5,1 |
| grey_high | 43 | 100 | 1,0 [0,2–5,4] | 9,3 [3,7–21,6] (4/43) | 2,8 |
| near_high | 34 | 68 | 0,0 [0,0–5,3] | 29,4 [16,8–46,2] (10/34) | 10,8 |
| high | 93 | 2 | 0,0 [0,0–65,8] | 21,5 [14,4–30,9] (20/93) | 16,8 |

**Abaixo do corte, por faixa, com o VPP projetado ao extrato.** VPP (extrato) reescala os não pares da célula (faixa × lado) ao total do extrato (mediana a posteriori e IC de Jeffreys, 20 mil sorteios).

| faixa | pares | não pares | FP | FN % [IC] | VPP (amostra) % [IC] | VPP (extrato) % [IC] |
|---|--:|--:|--:|---|---|---|
| low | 1 | 100 | 0 | 100,0 [20,7–100,0] (1/1) | n/d | 0,0 [0,0–0,0] |
| grey_low | 9 | 100 | 1 | 55,6 [26,7–81,1] (5/9) | 0,0 [0,0–79,3] | 0,0 [0,0–0,0] |
| grey_mid | 9 | 100 | 0 | 33,3 [12,1–64,6] (3/9) | 100,0 [20,7–100,0] | 6,6 [0,6–96,6] |
| grey_high | 5 | 100 | 1 | 20,0 [3,6–62,4] (1/5) | 80,0 [37,6–96,4] | 24,5 [7,7–78,9] |
| near_high | 5 | 68 | 0 | 60,0 [23,1–88,2] (3/5) | 100,0 [34,2–100,0] | 89,7 [44,5–100,0] |
| high | 7 | 2 | 0 | 42,9 [15,8–75,0] (3/7) | 100,0 [51,0–100,0] | 95,4 [75,0–100,0] |
| **total abaixo** | 36 | 470 | 2 | 44,4 [29,5–60,4] (16/36) | 84,6 [57,8–95,7] | **2,9 [0,8–12,1]** |
| **total abaixo sem `low`** | 35 | 370 | 2 | 42,9 [28,0–59,1] (15/35) | 84,6 [57,8–95,7] | **4,8 [1,5–20,9]** |

**Comparação com a tese.** A tese (nota **visível**) tem VPP de 16,0% (4 pares verdadeiros em 25 aceitações), IC 95% de 6,4% a 34,7%. Aqui (nota **oculta**), abaixo do corte e reescalado ao extrato, o VPP é 2,9% [0,8–12,1] com todas as faixas e 4,8% [1,5–20,9] sem `low`. Os intervalos se sobrepõem em parte (6,4%–12,1% e 6,4%–20,9%).

**Comparação com as taxas sintéticas da simulação** (Laplace sobre a vigilância da tese, nota visível; `low` herdou `grey_low`): FP sintético 8,4% / 8,4% / 2,7% / 2,9% / 4,5% / 50,0% contra o medido 0,0% / 1,0% / 0,0% / 1,0% / 0,0% / 0,0% (`low` … `high`); FN sintético 75,0% / 75,0% / 6,7% / 11,1% / 30,6% / 25,0% contra o medido 66,7% / 29,4% / 7,0% / 9,3% / 29,4% / 21,5%. Os FN concordam bem de `grey_mid` em diante; as taxas de FP da vigilância ficaram bem acima das medidas aqui.

**Custo da medição.** 717 pares revisados (árbitro em 75); 1.542 chamadas; 1.135.128 tokens de entrada (sem cache) e 2.685.026 de saída; 672 s em 3 execuções; nenhum erro HTTP; 1 erro de revisão registrado.

**Ressalvas.**

1. **A amostra é estratificada e não representa a fila.** Só até 100 não pares por faixa entram; o VPP da amostra é superestimado e o VPP do extrato depende da reescala. A tese mediu pares escolhidos pela política, não uma amostra estratificada: a comparação com 16,0% é de ordem de grandeza, não de igualdade.
2. **Dois fatores se misturam.** Aqui mudam, ao mesmo tempo, a ocultação da nota e a população. Não há comparação pareada nota visível × nota oculta nos mesmos pares, de modo que a diferença para a tese não pode ser atribuída à nota.
3. **Faixas com poucos dados.** `low` tem 3 pares verdadeiros (1 abaixo do corte) e `high`, 2 não pares: tratar as duas faixas como não medidas. Os IC do VPP do extrato por faixa são muito largos.
4. **Uma realização do Haiku.** A própria tese registra variabilidade entre execuções do modelo (concordância de 72,2%, 13 de 18 pares; `estudo4_gzcmd.tex:95`).
5. **Hipótese não testada.** FP de 0,4% na amostra estratificada e 21 FP em 25 aceitações na tese (`estudo4_gzcmd.tex:311`) podem ser compatíveis se os FP se concentrarem nos não pares de maior nota dentro de cada faixa, que são os que a fila e a política selecionam. O replay (seção F) aponta nessa direção (FP bem acima do previsto pelas taxas medidas), mas o fixture não permite separar as explicações.

## B. Fila auto-calibrável (método 0.7.0)

**Fontes:** `docs/sim_fila_autocalibravel.md` (v3), `_v2.md`, `_v1.md`; `docs/llm_real_fila_ponta_a_ponta.md`; `docs/engenharia_de_filas.md` §1–2; desenho em `docs/plans/fila-autocalibravel.md`. **Status: B1 Confirmado** (direção); **B2 Parcial/Exploratório**.

### B.1 Simulação (revisor sintético calibrado na vigilância, nota oculta; 10 sementes)

O revisor simulado usa as taxas da vigilância da tese (nota visível), não as medidas da seção A; é cenário `thesis_bands`, com humano perfeito. Valores: média (mínimo–máximo) entre sementes.

**Pares abaixo do corte (de 36) encontrados por run, v1 → v2 → v3 (humano, fila padrão).**

| pool / revisor | v1 → v2 → v3 |
|---|---|
| `all` / perfeito | 20,5 → 17,8 → 21,9 |
| `all` / `thesis_bands` | 13,7 → 10,5 → 14,4 |
| `below_cut` / perfeito | 0,0 → 13,1 → 15,0 |
| `below_cut` / `thesis_bands` | 0,2 → 7,2 → 7,8 |

- v1 (faixas por probabilidade, 5 compostos): não superou a nota abaixo do corte; no pool completo ficou no nível do Fellegi-Sunter (`sim_fila_autocalibravel_v1.md:235`).
- v2 (faixas por posto, exploração, sub-escores do CdR): corrigiu o problema central da v1 no `below_cut` (0,2 → 7,2), mas não superou a nota; os sub-escores pioraram o pool completo (`sim_fila_autocalibravel_v2.md:322`). A ablação mostra que a perda tardia vem da troca de campos: 14,2 com compostos contra 10,5 com sub-escores.
- v3 (portão humano com decisão explícita, orçamento humano por rodada, `negative_labels`): ver abaixo.

**Esforço humano a orçamento igual (v3, `thesis_bands`, abaixo do corte, pares abaixo do corte verificados de 36).**

| método | B=50 | B=100 | B=200 | k=10 (decisões) |
|---|---|---|---|---|
| H0: humano sozinho, ordem da nota | 10,0 | 12,0 | 13,0 | 24 |
| H1: nota + LLM tria + humano confirma MATCH | 10,9 (8–14) | 13,2 (12–16) | 17,6 (16–20), com 7.084 revisões de LLM | 30 (20–55) |
| H2: fila com portão revisto | 8,3 (5–10) | 9,5 (5–12) | 13,0 [n=1] | 45 (30–54), só 70% das sementes |

Com revisor perfeito, H1 e H2 verificam k=10 pares abaixo do corte em 10 decisões contra 24 de H0: o ganho existe quando o LLM não erra e some com as taxas de erro da vigilância. A parada por `posterior` não disparou em nenhuma rodada com `thesis_bands` (a taxa de FP da faixa `low`, 8,4%, supera τ = 0,05). A conclusão da simulação v3 é que, nas condições simuladas, a fila com portão revisto não encurta o trabalho humano em relação à ordem da nota nem a H1 (`sim_fila_autocalibravel.md`, bloco "Resposta").

**Efeito de `negative_labels`.** `llm_nonmatch` e `verified_only` dão resultados parecidos, com intervalos sobrepostos (por exemplo, `below_cut`, 50 por rodada: 9,6 e 9,1 pares abaixo do corte). Com `verified_only`, a recalibração é mais vezes `skipped_insufficient`, sem perda visível de pares verificados.

### B.2 Execução de ponta a ponta com o Haiku real (humano simulado perfeito)

Pool `all` (61.696 pares; 247 verdadeiros), configuração padrão (50 decisões humanas por rodada, `max_reviews`=2000, `max_rounds`=5), uma execução. O run parou por `sterile_round` após 3 rodadas, sem atingir `max_reviews`.

| política de negativos | revisões do LLM | MATCH do LLM | decisões humanas | verdadeiros verificados | abaixo do corte (de 36) | FP do LLM pegos | over_budget |
|---|--:|--:|--:|--:|--:|--:|--:|
| `llm_nonmatch` | 1.443 | 287 | 127 | 96 | 4 | 6 | 310 |
| `verified_only` | 1.406 | 288 | 128 | 96 | 4 | 7 | 310 |

**A decisões humanas iguais** (pares abaixo do corte verificados, de 36):

| decisões humanas | fila (`llm_nonmatch`) | H1 | H0 |
|--:|--:|--:|--:|
| 50 | 4 | 4 | 5 |
| 100 | 4 | 8 | 9 |
| 127 | 4 | 10 | 9 |

Com as mesmas 1.443 revisões do LLM, H1 verifica 115 (10 abaixo do corte) gastando 142 decisões humanas. Recalibração: com `llm_nonmatch`, a rodada 1 ajusta o modelo com 421 rótulos (50 positivos, 371 negativos; Kish 140,4); com `verified_only`, a rodada 1 é `skipped_insufficient` (50 positivos, 0 negativos). Custo: 3.105 chamadas (`llm_nonmatch`), 2.252.301 tokens de entrada e 5.333.905 de saída; 2.375 das 3.000 novas revisões permitidas pelo teto global.

### B.3 Por que a fila perdeu (D1–D6)

Fonte: `engenharia_de_filas.md` §2.

| id | diagnóstico | evidência | status |
|---|---|---|---|
| D1 | A régua (nota) já é forte | A nota acha 12 dos 36 nas primeiras 100 revisões (`sim_fila_autocalibravel.md`, curva da nota abaixo do corte: 12,0 em 100 revisões, perfeito) | Confirmado (medição direta) |
| D2 | Poucos positivos para aprender | Prevalência 0,06% (36 em 61.485); ~10 positivos em centenas de revisões contra 15 a 53 coeficientes | Confirmado (contagem) |
| D3 | Positivos difíceis diferem dos fáceis | Coerente com o perfil "dados ausentes" (seção E), não isolado | Parcial/Exploratório |
| D4 | Objetivos misturados (explorar e calibrar no mesmo controle) | Não isolado em experimento; motivou S2 e a separação do princípio 4 | Parcial/Exploratório |
| D5 | Rótulo ruidoso do LLM | FN de 44,4% abaixo do corte (seção A) | Confirmado (medição direta) |
| D6 | A régua ficou fora do modelo | Plausível; S2 (régua como offset) recupera ganho no real (seção E), mas não houve ablação isolada | Parcial/Exploratório |

**Limitações conhecidas do método** (`plans/fila-autocalibravel.md`, "Limitações conhecidas"): recompensa, parada e anomalia usam o MATCH bruto do LLM, não o verificado; a parada fica inerte com FP ≥ τ; o bandit age sobre segmentos contíguos do ranking. O método é experimental e não é recomendado para produção.

## C. LLM como desempate na zona cinza

**Fonte:** `docs/llm_real_desempate_zona_cinza.md`. **Status: Parcial/Exploratório** (36 positivos; uma realização do LLM; humano perfeito; não mede tempo por decisão).

**Desenho.** Pool: os 61.485 pares abaixo do corte (36 verdadeiros). Ordem base N: nota decrescente. O LLM dá veredito aos W primeiros de N (W = 500, 1.000, 2.000, 3.000) e **não decide nada**: cada estratégia é uma permutação do pool e o humano decide todo par que revisa. L1: na janela, MATCH, depois UNSURE, depois NONMATCH (cada grupo na ordem da nota); L2: por blocos de nota de largura 0,5; L3: como L1, com os MATCH por confiança do LLM. Orçamentos B = 25 … 1.600; bootstrap pareado (2.000 réplicas).

**O LLM na janela.**

| W | verdadeiros na janela | MATCH / UNSURE / NONMATCH entre os verdadeiros | FP (não pares com MATCH) |
|--:|--:|---|--:|
| 500 | 15 | 8 / 0 / 7 | 12 |
| 1.000 | 17 | 10 / 0 / 7 | 23 |
| 2.000 | 19 | 11 / 0 / 8 | 48 |
| 3.000 | 20 | 11 / 1 / 8 | 58 |

**Verdadeiros abaixo do corte achados (de 36).** N (nota): 10, 10, 12, 13, 13, 17, 19 em B = 25, 50, 100, 200, 400, 800, 1.600.

| estratégia / W | B=25 | B=50 | B=100 | B=200 | B=400 | B=800 |
|---|--:|--:|--:|--:|--:|--:|
| L1, W=500 | 8 (−2) | 12 (+2) | 14 (+2) | 15 (+2) | 15 (+2) | 17 (0) |
| L1, W=1.000 | 8 (−2) | 10 (0) | 14 (+2) | 17 (+4) | 17 (+4) | 17 (0) |
| L1, W=2.000 | 8 (−2) | 11 (+1) | 11 (−1) | 17 (+4) | 18 (+5) | 18 (+1) |
| L1, W=3.000 | 8 (−2) | 11 (+1) | 11 (−1) | 16 (+3) | 19 (+6) | 19 (+2) |
| L2, W=3.000 | 10 (0) | 10 (0) | 12 (0) | 13 (0) | 17 (+4) | 17 (0) |

Com B ≥ W, L e N revisam o mesmo conjunto (diferença 0 por construção). L1 e L3 acharam o mesmo número em 25 de 28 células.

**Bootstrap pareado (L − N).** O IC de 95% exclui zero em **4 de 48** combinações (estratégia, W, B ∈ {50, 100, 200, 400} com B < W): por exemplo, L1 com W=3.000 e B=400: +6 [+1; +9]; L1 com W=2.000 e B=400: +5 [+1; +8]. Com B=50 e 100, os IC incluem zero (por exemplo, L1, W=2.000, B=100: −1 [−7; +6]).

**Custo.** 1.055 (W=500) a 6.411 chamadas (W=3.000) por janela; o menor custo por par extra entre B=50 e 400 é 528 chamadas (L1, W=500, B=50); em B=400 e W=3.000, 1.068 chamadas por par extra. O humano decide o mesmo número B de pares em todas as estratégias.

**Leitura.** O ganho mais claro aparece com ~200 a 400 decisões humanas (+2 a +6 de 36, maior com janelas maiores) e custa milhares de chamadas ao LLM; com 25 a 100 decisões, os resultados são mistos (de −3 a +2) e os IC incluem zero; com B ≥ 800 a diferença volta a ~0 a +2. Com 36 positivos, diferenças de 1–3 pares são ruído. A leitura é coerente com H11 real refutada (seção D) e com a medição de FN (seção A): o LLM deixa passar 44,4% dos pares abaixo do corte e dá MATCH a dezenas de não pares na janela.

## D. Hipóteses de engenharia de filas H1–H12

**Fontes:** `docs/engenharia_de_filas.md` §4 (hipóteses pré-registradas em 2026-11-15, antes dos experimentos), §6–7 (resultados e revisões); `docs/engenharia_de_filas_real_h8_h11.md`. Notação: S0 = régua (nota); S1 = fila auto-calibrável padrão; S2 = fila com régua como offset e rótulos verificados, sem LLM; S3 = S2 com o veredito do LLM como atributo; S4 = S2 com o veredito como rótulo; S5 = bandit de Thompson sobre faixas contíguas.

**Execução.** 39 células sintéticas × 10 sementes (N = 20.000 candidatos por célula); no real, 10 sementes (61.485 pares abaixo do corte; 36 verdadeiros). O status vem da regra da seção 4 aplicada **à letra**; onde a letra e a intenção divergem, está dito.

| Id | Sintético | Extrato real (36 positivos) | Números principais e ressalva |
|---|---|---|---|
| H1 (aprender não supera a régua bem especificada) | **Refutado** (pela letra) | **Refutado** (S2 > S0) | Sintético: S2−S0 = −2,9 … −1,6 com IC excluindo 0, por causa do custo da amostra de estimação (5% de B); sem ela (S2x−S0) fica entre −1,1 e 0,0; S1−S0 = +11,1; +4,0; −11,1; −39,4; −62,7 (B = 50…800). Real: S2−S0 = +4, +3, +4, +6, +16; S1−S0 < 0 em todo B (−3,8 a −9,8). A previsão qualitativa se mantém no sintético; a letra falha |
| H2 (o ganho cresce com a prevalência) | **Refutado** | — | Ganho relativo médio de S2: −0,042; −0,021; −0,034; −0,023 (0,1% a 20%); nunca positivo |
| H3 (régua fraca deixa espaço para aprender) | **Confirmado** | — | Pesos errados (graus 0–3): S2−S0 médio −2,08; −0,58; +3,24; +7,30; ruído: −2,08; +3,08; +19,30; +29,48 |
| H4 (só há ganho se há sinal fora da régua) | **Refutado** (pela letra) | — | Com prevalência de 5%, a S2 perde com e sem sinal pelo custo da amostra; sem esse custo (S2x), o padrão aparece só em B grande (+8,3 em B=800 com sinal) |
| H5 (positivos difíceis limitam o aprendizado) | **Refutado** | — | S2 recupera mais difíceis que a régua com sinal (+2,9 em B=800 a 5%), contrário à previsão |
| H6 (o offset garante não inferioridade) | **Refutado** | **Confirmado** | Sintético: 49 de 200 (célula, B) violam, 26 em B=50; real: 0 de 5, pior folga +4,0 |
| H7 (LLM como atributo > LLM como rótulo) | **Refutado** (pela letra) | **Confirmado** (evidência fraca, após corrigir o vazamento) | S3−S4 ≥ 0 sempre, mas S3−S0 viola −δ em 9 de 200 (B=50, 5%). Real: S3−S0 = +4; +3; +6; +16; +15; S3−S4 = +2; +2; +3; +10; 0 (S0, S3, S4 determinísticas; uma realização do LLM) |
| H8 (bandit sobre faixas contíguas não ajuda com rendimento monotônico) | **Confirmado** | **Confirmado** (ambos os lados) | Sintético: S5−S0 ≤ −0,6 nas monotônicas; com bolso de positivos, +25,0 (B=400) a +130,3. Real: ver abaixo |
| H9 (estimação separada funciona com orçamento pequeno) | **Confirmado** (IC inútil) | — | Cobertura S2 0,990, estimador da S1 0,128; largura média do IC de milhares para centenas de restantes |
| H10 (parada por custo > τ fixo) | **Parcial/Exploratório** (inconclusivo: critério mal especificado) | — | S2 < S6 em 1 de 4 níveis; em 5% e 20% nenhuma regra para (empate por construção); onde divergem, S2 para antes (k = 98 contra 215 em 0,1%) |
| H11 (LLM audita decisões humanas) | **Confirmado** | **Refutado** | Sintético: recuperação 0,59–0,78, precisão 0,86–0,99. Real: ver abaixo |
| H12 (o LLM reduz o tempo por decisão humana) | **Pendente** (não testável offline) | **Pendente** | Exige estudo com revisores reais |

**H8 no real** (`engenharia_de_filas_real_h8_h11.md`; 20 sementes; recompensa = par verdadeiro ou MATCH do LLM). Premissa de monotonia verificada: taxa por faixa 2,76% (0–1%), 0,12% (1–5%), 0,11% (5–20%), 0,01% (20–100%). Cenário `real`: S5−S0 de −2,00 (B=100 e B=800) a 0,00, nunca com limite inferior > 0 (S0 = 10, 12, 13, 13, 17 e S5 = 10, 10, 12,2, 13, 15 em B = 50…800). Com não monotonia induzida (verdadeiros do topo permutados para 1–5%, N1, ou 5–20%, N2): S5−S0 = +0,15 a +0,80 (N1) e +0,05 a +0,50 (N2), com IC acima de 0 em B ≥ 100 (N1) e B ≥ 400 (N2). Ressalva: os ganhos são pequenos em valor absoluto e medidos contra uma S0 que a indução prejudicou. **Correção (erro E1 da revisão factual):** em B=800, com a permutação N1 a ordem pela nota (S0) cai para 1,25 e o bandit S5 tem 1,95 (+0,70); com N2, S0 = 0 e S5 = 0,50; os 17,00 são da S0 **sem** indução (cenário real, em que S5 = 15). O texto anterior, que contrapunha o 1,95 do bandit com N1 aos 17,00 da S0 sem indução, estava errado. N1 e N2 são construções do analista.

**H11 no real** (16 células de FN humano × FP humano × B). **0 de 16** satisfazem recuperação ≥ 50% e precisão ≥ 30%: recuperação de 26,1% a 47,9%; precisão de 100% em B=100 (0,3 a 0,8 par recuperado por semente), 9,1%–21,6% em B=200, 3,0%–7,7% em B=400 e 2,6%–5,2% em B=800, contra 0,2%–3,4% de uma auditoria aleatória do mesmo tamanho (15 a 30 vezes melhor que o acaso, mas longe de 30%). Na grade principal (`exp_engenharia_filas.py`, B=400, FP humano de 1%, 200 realizações do erro humano, fixture do LLM real), com FN humano de 5%: 134 FN, 61 recuperados (0,46), 2.041 re-revisões, precisão 0,03; com 10%: 257, 113 (0,44), 2.093, 0,05.

**Ressalvas gerais de D** (`engenharia_de_filas.md` §6.6): o gerador sintético é escolha do analista, a régua usa a evidência latente (mais fina que os compostos do modelo) e o LLM simulado erra só por faixa de nota e nunca por par; a conclusão "S3 muito melhor que tudo" no sintético depende de o LLM revisar de graça os 2.000 primeiros da régua; no real, S0 e S2 são determinísticas, os "IC" degeneram em ponto e diferenças de 3 a 16 pares são evidência fraca. Comparações múltiplas: H6 e H7 exigem o critério em todas as 200 combinações (célula, B).

**Revisões após os resultados** (§6.7 e §7): (A) vazamento de cobertura do fixture no real, corrigido (a cobertura passou a depender da posição na régua, não do rótulo; os números de S3/S4 e o veredito de H7 mudaram); (B) a régua do real é um *stub* (`nota/10`), porque abaixo do corte há 9 pares com nota ≥ 9 (< 30 âncoras) e o ajuste por âncoras falha; (C) o estimador HT e a regra de parada passaram a ver só a decisão humana (nenhum status sintético mudou).

**Recomendação revisada** (§7, parte sem experimento marcada pelo documento de origem): ordenar pela melhor régua; usar S2 só com razão para duvidar da régua, reportando a régua efetiva e checando placebo; usar o LLM como atributo e não como rótulo nem como auditor nesta prevalência; não gastar 5% de B em amostra uniforme; parar por custo esperado; bandit só com indício de não monotonia; não usar a sessão padrão atual (para em ~190 decisões e perde da régua de B=200 em diante).

## E. Achado principal parcial: S2 no extrato real (régua como ponto de partida, rótulos verificados)

**Fonte:** `docs/engenharia_de_filas.md` §6.3, §6.7, §7, §8.1; `docs/engenharia_de_filas_replicacao_passo.csv`. **Status: Parcial/Exploratório, reclassificado após a replicação por passo (E.5).** Os números da §6.7 não são critérios pré-registrados; uma semente determinística; humano perfeito; 36 positivos. **Resumo da reclassificação:** o ganho grande de S2 no extrato (33 contra 17 em B=800) dependia da régua plana (*stub*); fora da amostra, com a régua calibrada, o ganho é pequeno (+1 a +4 pares somados nos passos 9 e 11) e nunca negativo, e nenhum critério R1–R4 foi confirmado nem refutado. O perfil "dados ausentes" só se sustenta, **parcialmente, para mãe ausente**.

### E.1 Rendimento no real (verdadeiros abaixo do corte verificados, de 36)

| estratégia | B=50 | B=100 | B=200 | B=400 | B=800 |
|---|--:|--:|--:|--:|--:|
| S0 (régua) | 10 | 12 | 13 | 13 | 17 |
| S1 (sessão atual, ~106 decisões) | 6,2 | 7,2 | 7,2 | 7,2 | 7,2 |
| S2 (= S2x, régua plana) | 14 | 15 | 17 | 19 | 33 |
| S3 (LLM como atributo; veredito só na janela dos 3.000 primeiros) | 14 | 15 | 19 | 29 | 32 |
| S4 (LLM como rótulo, idem) | 12 | 13 | 16 | 19 | 32 |
| S5 (bandit) | 10 | 10 | 12 | 13 | 13 |

### E.2 O ganho de S2 vem dos compostos, e em boa parte da escala da régua

- **Placebo** (linhas dos 8 compostos embaralhadas juntas, nota e verdade fixas; 10 embaralhamentos): S2x embaralhada = 10,1 / 11,1 / 12,3 / 13,5 / 16,2 (máximo 11 / 13 / 13 / 14 / 17) contra S2x = 14 / 15 / 17 / 19 / 33. **O ganho some**: o que S2 usa está nos compostos, não em artefato do método.
- **Escala da régua.** Com a régua plana (*stub*), S2 tem 33 contra 17 de S0 em B=800 (+16); trocando pelo Platt por âncoras ajustado no extrato inteiro (A⁺ = 95), S2 tem **20 contra 17** (+3; +1 com `sd_gamma` = 0,1), e em B=400, 18 contra 13 (+5; o *stub* dava +6). `sd_gamma` não é a alavanca (32 a 33 em B=800). Conclusão do próprio documento: a maior parte do ganho de +16 vem da escala plana do *stub* (13 de 16 pares em B=800); o que sobrevive (+5 em B=400) vem dos compostos. A replicação por passo (E.5) confirma essa leitura fora da amostra: com a régua calibrada o ganho de transferência é de +1 a +4 e, com a plana, +2, +4 e +7.
- **Coeficientes** (σ = 0,5): dominantes `dtnasc:L1` (+0,43 a +0,58) até k=200 e `mae:L2` (−0,85) em k=775; todos pequenos.

### E.3 Perfil "dados ausentes" (exploratório, lido nos próprios positivos)

Em B=800, a régua S0 alcança 17 dos 36; dos 19 que não alcança, S2 alcança 16. Entre os 19: `mae_missing` 15, `nome` L3 13, DTNASC zerada 17; o perfil triplo (`mae_missing` e `nome` L3 e DTNASC zerada) em 12. Rendimento do perfil triplo: 12 positivos em 1.476 pares do pool; **12 em 102 (11,8%) na faixa de nota 5,47–6,90**, contra 5 em 13.389 (0,04%) nessa faixa sem o perfil e 1,0% na margem de S0 (postos 400–800). **Ressalva do documento de origem: o perfil foi lido nos próprios positivos que ele explica (em amostra, descoberto depois); não é teste de hipótese e precisa de positivos novos.** Leitura original: a régua parece sub-ponderar o padrão "dados ausentes" em notas de 5,5 a 6,9, inferência de 12 pares.

**Após a replicação por passo (E.5).** Os modelos treinados sem o passo de teste dão peso a favor de par à **mãe ausente** nos seis casos (3 dobras × 2 réguas; +0,12 a +0,48, leitura invariante à referência), mas o nome em nível máximo é misto (−0,25 a +0,35) e a data de nascimento zerada pesa **contra** par (−0,47 a −1,19). O perfil triplo, portanto, **não se sustenta fora da amostra**; o que resta é uma hipótese sobre mãe ausente, ainda sem teste independente (dobras de 7 e 25 positivos).

### E.4 LLM como atributo (S3) no real

S3−S0 = +4; +3; +6; +16; +15 e S3−S4 = +2; +2; +3; +10; 0 (B = 50…800). Em B=400: S3 +16 e S2 +6 sobre a régua (29 e 19 contra 13); em B=800, S3, S4 e S2 empatam (32, 32 e 33). Evidência de **uma realização do LLM** e 36 positivos. S3 supõe o LLM gratuito sobre a janela dos 3.000 primeiros pela nota (2.992 cobertos); o vazamento de cobertura corrigido mostra que o fixture precisa de cobertura independente do rótulo. No sintético, S3 é muito melhor que as demais (146,2 contra 90,3 de S0 em B=200), mas isso depende de o LLM simulado (FP 0,4%) ser um filtro de custo humano zero.

### E.5 Replicação por passo de blocagem: concluída; R1–R4 inconclusivas

Pré-registrada em 2026-11-15 em `docs/engenharia_de_filas.md` §8, **antes** de rodar, depois de ver os resultados reais de H1/H7 e as confirmações da §6.7. Executada em 2026-11-15 (`python notebooks/exp_engenharia_filas.py replicate-passo ...`; 2 s; determinística; humano perfeito); resultados em §8.1 e em `docs/engenharia_de_filas_replicacao_passo.csv`. O critério e o texto do pré-registro não foram alterados. **Resultado: nenhum dos quatro critérios foi confirmado nem refutado.**

- **Dobras.** Pares abaixo do corte, separados por `PASSO` (6, 9, 11); PAR só como verdade; uma dobra = um passo de teste, os outros dois são treino; dobra com menos de 5 positivos abaixo do corte é "não informativa" e fica fora dos critérios (reportada mesmo assim). Pares e positivos por passo (conferidos no CSV): passo 6, 2.573 pares e **4 positivos (não informativa)**; passo 9, 1.329 pares e 7 positivos; passo 11, 57.583 pares e 25 positivos (total 61.485 e 36). A tese informa os mesmos 4, 7 e 25 pares recuperados por revisão manual (`apendice_linkage_origem.tex:34–36`). Só os passos 9 e 11 entram em R1–R4.
- **Variantes de régua:** (P) plana (`nota/10`, como no real); (C) calibrada (Platt por âncoras ajustado no extrato inteiro). Procedimentos: **T** (transferência: S2 treina nos outros dois passos com B=800 e ordena o passo de teste sem rótulo do teste) e **O** (online: S2 só dentro do passo de teste). B ∈ {100, 200, 400}.
- **Critérios (somas sobre as dobras informativas; execução determinística, sem IC):**
  - **R1.** A correção aprendida transfere entre passos (variante C): confirmada se ganho(T) − S0 somado ≥ +3 em B=400 e ≥ 0 em todo B; refutada se < 0 em B=400; senão, inconclusiva.
  - **R2.** O ganho online se repete dentro de cada passo (variante C): confirmada se ganho(O) − S0 > 0 em B=400 em todas as dobras informativas; refutada se < 0 em alguma; senão, inconclusiva.
  - **R3.** A régua plana supera a calibrada fora da amostra: confirmada se, em T, a soma de encontrados com P > com C em B=400 e em B=200; refutada se ≤ em ambos.
  - **R4.** O padrão "dados ausentes" é aprendido sem ver o passo de teste: confirmada se, nas três dobras, o modelo de treino dá peso a favor de par a pelo menos dois de `mae:missing`, concordância máxima de nome e nível zero de data de nascimento; refutada se em nenhuma dobra.
- **Ressalva registrada antes:** o perfil foi identificado com todos os 36 positivos, inclusive os de cada passo de teste; R4 testa se o **modelo** aprende o padrão sem esses rótulos, não se o perfil foi descoberto de forma independente.

**Verdadeiros verificados** (S0 = ordem pela nota no passo; T = modelo treinado nos outros dois passos, sem rótulo do teste; O = online dentro do passo; B = 100 / 200 / 400). Todos conferidos contra o CSV.

| passo (pares; positivos) | régua | S0 | T | O |
|---|---|---|---|---|
| 6 (2.573; 4), não informativa | P (plana) | 3 / 3 / 3 | 4 / 4 / 4 | 4 / 4 / 4 |
| 6 | C (calibrada) | 3 / 3 / 3 | 3 / 3 / 3 | 3 / 3 / 3 |
| 9 (1.329; 7) | P | 5 / 6 / 7 | 7 / 7 / 7 | 7 / 7 / 7 |
| 9 | C | 5 / 6 / 7 | 5 / 7 / 7 | 5 / 7 / 7 |
| 11 (57.583; 25) | P | 9 / 9 / 11 | 9 / 12 / 18 | 12 / 14 / 14 |
| 11 | C | 9 / 9 / 11 | 11 / 12 / 12 | 9 / 9 / 12 |

**Status** (critérios exatamente como no pré-registro):

| critério | status | números | leitura |
|---|---|---|---|
| R1 (transferência, régua C) | **Inconclusiva** | Soma T−S0 nos passos 9 e 11: +2 (B=100), +4 (B=200), +1 (B=400) | Nunca negativo, mas abaixo do +3 exigido em B=400: a correção aprendida transfere pouco com a régua calibrada |
| R2 (online, régua C) | **Inconclusiva** | O−S0 em B=400: passo 9, +0; passo 11, +1 | O ganho online não aparece no passo 9 e é de 1 par no 11; nunca negativo |
| R3 (plana > calibrada fora da amostra) | **Inconclusiva** | Encontrados em T somados, P contra C: 19 contra 19 (B=200); 25 contra 19 (B=400) | A plana não perde da calibrada e ganha em B=400, mas empata em B=200, então a letra (maior em ambos) falha |
| R4 (perfil "dados ausentes" aprendido sem ver o teste) | **Inconclusiva — critério mal especificado** | Veredito por (leitura de `mae:missing`, régua) nas dobras informativas: literal/C refutada, literal/P refutada, invariante/C inconclusiva, invariante/P confirmada | O critério não diz a régua nem o que fazer quando `missing` é a categoria de referência da codificação (sem coluna; na leitura literal fica indefinido) |

**R4, pesos do modelo de treino** (coeficiente relativo à referência da codificação; `mae:missing` na leitura invariante = menos a média dos demais níveis de `mae`; `dtnasc zero` = menos a média dos níveis de data de nascimento acima de zero, positivo quando o nível zero pesa mais a favor de par):

| passo de teste | régua | mae:missing (invariante) | nome L3 | dtnasc zero |
|---|---|---|---|---|
| 6 (não informativa) | P | +0,48 | +0,14 | −0,47 |
| 9 | P | +0,35 | +0,16 | −0,52 |
| 11 | P | +0,30 | +0,35 | −0,56 |
| 6 (não informativa) | C | +0,25 | −0,25 | −1,19 |
| 9 | C | +0,12 | −0,12 | −0,95 |
| 11 | C | +0,22 | +0,15 | −0,70 |

Leitura: mãe ausente é favorecida em todos os modelos (na leitura invariante); nome em nível máximo é misto; data de nascimento zerada pesa **contra**. O perfil triplo da §6.7 não se reproduz como um todo.

**Fora dos critérios (régua plana):** a transferência soma +2, +4 e +7 sobre S0 em B=100, 200 e 400 (passos 9 e 11), e o online no passo 11 soma +3, +5 e +3 (12/14/14 contra 9/9/11): o ganho está na régua plana, não na calibrada.

**Ressalvas.** Poucos positivos por passo (4, 7 e 25): uma diferença de 1 par muda o status; execução determinística, sem IC. Humano perfeito. O treino do passo 11 vem de dois passos pequenos (3.902 pares, 11 positivos) e o dos demais, de ~58.000 pares. O perfil foi derivado com os 36 positivos, inclusive os de cada passo de teste; R4 mede só se o **modelo** o aprende sem os rótulos do teste. As definições operacionais de R4 (duas leituras de `mae:missing`, nível zero da data de nascimento, decisão por todas as leituras e réguas) foram fixadas ao implementar, antes dos números finais; o literal `mae:missing` só foi visto como indefinido ao rodar. A replicação por passo **não** é coorte nova nem substitui o critério prospectivo da tese: é replicação interna, no mesmo extrato.

**Consequências para o achado principal.** (i) O ganho de S2 não foi reproduzido em escala relevante fora da amostra com a régua calibrada (+1 a +4 pares somados, nunca negativo), de modo que o +16 de B=800 deve ser lido como efeito da escala plana da régua; (ii) a hipótese de que a régua sub-pondera o perfil triplo continua **não testada fora da amostra** e, nos pesos aprendidos, só mãe ausente a sustenta, em parte; (iii) a recomendação de usar S2 só com régua duvidosa, reportando a régua efetiva (`engenharia_de_filas.md` §7), permanece.

## F. Simulação com taxas medidas do Haiku e replay do fixture

**Fonte:** `docs/sim_fila_autocalibravel.md`, seção "Revisor calibrado no Haiku real" (**existe**; por isso a linha não é Pendente). **Status: Parcial/Exploratório** (execução concluída, mas o fixture cobre só parte dos pares revisados e o veredito é fixo entre sementes).

**Revisores.** `measured` (taxas por faixa do `.rates.csv` como estão, `disagreement_rate` = 0,10, suposição), `measured_smoothed` (as mesmas contagens com Jeffreys) e `fixture_replay` (o par presente no fixture volta com a decisão real gravada; o ausente usa o fallback `measured_smoothed`). O fixture tem 4.604 pares (717 da amostra de calibração, 2.785 do teste de reordenação e 1.102 de filas e bases); o replay não faz chamada à API.

**Esforço humano no pool `below_cut` (configuração padrão da fila, humano perfeito; pares abaixo do corte verificados, de 36; para comparar com `thesis_bands`, ver a seção B).**

Pool `below_cut` (conferido contra a versão final de `sim_fila_autocalibravel.md`, tabelas "Pares abaixo do corte verificados" e "Decisões humanas até verificar k pares abaixo do corte"; `—` = a marca não foi atingida; `[n=k]` = só k sementes chegaram, com viés de sobrevivência; entre parênteses, mínimo–máximo):

| revisor / método | B=50 | B=100 | B=200 | B=400 | k=5 (decisões) | k=10 (decisões) |
|---|---|---|---|---|---|---|
| H0 (humano, ordem da nota) | 10,0 | 12,0 | 13,0 | 13,0 | 6 (5–6) | 24 (23–24) |
| `fixture_replay` / H1 | 11,0 | 11,0 | 11,0 | 11,0 [n=7] | 5 (5–5) | 28 (28–28) |
| `fixture_replay` / H2 (fila) | 6,1 (6,0–7,0) | — | — | — | 5 (5–5) | — (nenhuma semente) |
| `measured` / H1 | 20,8 (18–23) | 23,1 (20–26) | — | — | 5 (5–5) | 14 (12–17) |
| `measured` / H2 | 10,1 (8–13) | 11,8 (9–14) [n=6] | — | — | 5 (5–5) | 31 (11–51) [0,80] |
| `measured_smoothed` / H1 | 16,2 (13–20) | 20,8 (19–23) | 23,7 (22–26) | — | 6 (5–6) | 17 (12–25) |
| `measured_smoothed` / H2 | 9,3 (7–11) | 10,9 (9–13) [n=8] | — | — | 6 (5–6) | 36 (12–92) [0,70] |

Pool `all` (inclui os 211 pares que o corte já aceita; decisões por verificado perto de 1,0 ali não indicam economia):

| revisor / método | B=50 | B=100 | B=200 | B=400 | k=5 (decisões) | k=10 (decisões) |
|---|---|---|---|---|---|---|
| H0 (humano, ordem da nota) | 5,0 | 8,7 (8,0–9,0) | 12,0 | 13,0 | 43 (43–44) | 129 (129–130) |
| `fixture_replay` / H1 | 4,0 | 8,0 | 11,0 | 11,0 | 59 (58–60) | 112 (112–113) |
| `fixture_replay` / H2 (fila) | 4,0 | 4,0 | — | — | — (0) | — (0) |
| `measured` / H1 | 2,3 (1–4) | 8,0 (7–11) | 20,2 (17–23) | — | 69 (58–80) | 108 (91–127) |
| `measured` / H2 | 2,3 (1–4) | 2,3 (1–4) | — | — | 125 (102–140) [0,30] | — (0) |
| `measured_smoothed` / H1 | 2,3 (1–4) | 7,7 (5–11) | 16,9 (15–20) | 24,1 (22–27) | 70 (58–81) | 110 (93–136) |
| `measured_smoothed` / H2 | 2,3 (1–4) | 2,3 (1–4) | 3,0 [n=1] | — | 102 (102–102) [0,10] | — (0) |

No `fixture_replay` a fila faz em média 85,4 decisões por run (14,0 decisões por par verificado) e não chega a B=100 em nenhuma semente. A orçamento de LLM igual no `below_cut`: `fixture_replay` fila 6,0, nota 10,0; `measured` fila 8,8, nota 11,1; `measured_smoothed` fila 8,6, nota 11,1.

**Pontos de atenção.**

- Com as taxas **medidas**, a triagem do LLM (H1) aparece muito melhor que com `thesis_bands` (20,8 contra 10,9 em B=50), mas o **replay não confirma** essa vantagem: H1 fica em 11,0 verificados abaixo do corte de B=50 a B=400.
- **FP: o replay fica acima do que as taxas medidas preveem.** No `all`, a fila tem 119,8 (118–121) MATCH sem verdade no `fixture_replay`, contra 5,7 (2–10) com `measured`; no `below_cut`, 13,2 (9–17) contra 5,8 (2–9). Hipótese **não testada**: o fixture inclui pares abaixo do corte em ordem decrescente de nota (os mais difíceis), que a amostra estratificada subamostra (ver ressalva 5 da seção A).
- **Cobertura do fixture.** Fração das revisões da fila servida pelo fixture: 88,9% (`all`) e 53,8% (`below_cut`); nas bases (nota, FS, aleatória e H1): 31,8% e 20,5%. Os resultados do replay misturam, portanto, decisões reais e taxas.
- O veredito de um par presente é **fixo entre sementes**: as sementes só variam exploração, desempate, auditoria e fallback; os intervalos mínimo–máximo subestimam a incerteza do revisor real. As taxas medidas vêm de amostras pequenas (`low`: 3 verdadeiros; `high`: 2 não pares).
- Conclusão do documento de origem: com `thesis_bands` e com o replay do Haiku real, a fila com portão revisto não reduz as decisões humanas por par verificado abaixo do corte em relação a revisar na ordem da nota; o método permanece experimental e não validado.

## G. Infraestrutura e reprodutibilidade

**Código (todos em `notebooks/`, ainda não commitados no momento da escrita).**

- `llm_real_eval.py`: harness de medição e de fila com LLM real. Subcomandos documentados no plano: `pilot`, `calibrate`, `report`, `queue-run`, `baseline`, `export-fixture`, `rerank-test`. `FixtureReviewer` reaplica as decisões gravadas, sem rede. Exemplos de uso no cabeçalho do script: `python notebooks/llm_real_eval.py pilot --n 20 --workers 8`; `... calibrate --workers 24 --max-api-reviews 1200`; `... report`. Os argumentos de `queue-run`, `baseline`, `export-fixture` e `rerank-test` não foram conferidos aqui.
- `sim_fila_autocalibravel.py`: simulação v3 e revisores `measured`, `measured_smoothed`, `fixture_replay`. Comandos de reprodução na seção "Reprodução" de `sim_fila_autocalibravel.md` (primeiro bloco: `thesis_bands` e `perfect`; segundo bloco, com `--append-csv`: revisores calibrados no Haiku real, que exigem os insumos locais abaixo).
- `exp_engenharia_filas.py`: H1–H12 sintético e real (`python notebooks/exp_engenharia_filas.py --seeds 10 --workers 14 --real-input ../phd-research/data/COMPARADORSEMIDENT.csv --out-csv docs/engenharia_de_filas_resultados.csv`; sensibilidade a σ com `--subset sensibilidade --no-s1 --sigma 0.25`, `1`, `2`).
- `exp_filas_real.py`: H8 e H11 no extrato real (`$env:PYTHONPATH = 'src'; python notebooks/exp_filas_real.py --input ../phd-research/data/COMPARADORSEMIDENT.csv --seeds 20 --out-csv docs/engenharia_de_filas_real_h8_h11.csv --out-md docs/engenharia_de_filas_real_h8_h11.md`).
- Testes: `tests/test_llm_real_eval.py`, `test_exp_engenharia_filas.py`, `test_exp_filas_real.py`, `test_sim_fila_autocalibravel.py`, entre outros.

**Fixture local (não versionado).** `data/llm_fixtures/haiku-5-5_v4_sem-nota.jsonl`: 4.604 pares de decisões reais do Haiku 5.5 (prompt v4, nota oculta), sha256 `51521231ba9779b52ea68b3896b35cf728b851e016280d1bc9971b4609b931d5`. Origem (precedência calibração > teste de reordenação > filas e bases): 717 da amostra de calibração, 2.785 do teste de reordenação e 1.102 de filas e bases. O mesmo sha256 consta em `sim_fila_autocalibravel.md` e em `engenharia_de_filas_real_h8_h11.md`.

**O que não está versionado e por quê.**

| item | onde | motivo |
|---|---|---|
| Fixture de decisões do LLM | `data/llm_fixtures/` (`.gitignore`) | Contém decisões por par; a tese registra que o `pair_id` histórico, sem sal, não garante anonimização (`capitulo_dados.tex:123`) |
| Cache de respostas do LLM | `data/llm_cache/` (`.gitignore`) | Idem; ocupa espaço e é regenerável |
| Chave da API | `.env` (`.gitignore`); só o harness a lê, e ela nunca é impressa nem gravada | Segredo |
| `.rates.csv` local | fora do git (segundo `sim_fila_autocalibravel.md`) | Derivado do fixture |
| Extrato `COMPARADORSEMIDENT.csv` | `../phd-research/data/` | Pseudonimizado, com quase-identificadores; acesso restrito (`capitulo_dados.tex:29`) |

Os relatórios `docs/*.md` e os CSV agregados (`docs/*.csv`) contêm só agregados e nenhum identificador de par; no momento da escrita, constam como não rastreados (`git status`) e **ainda não foram commitados**. Para reproduzir sem o fixture, é preciso refazer as chamadas à API (cache vazio), com a ressalva de que o Haiku é estocástico (a tese registra 13 de 18 decisões repetidas, `estudo4_gzcmd.tex:95`).

**Custos registrados.** Medição das taxas: 1.542 chamadas, 672 s. Fila ponta a ponta: 3.105 chamadas (`llm_nonmatch`) e 3.038 (`verified_only`); desempate: 1.514 revisões novas (teto 2.500), 1.460 s. Teto global de novas revisões na ponta a ponta: 2.375 de 3.000.

## H. Implicações para a tese

Mapeamento por achado, com `arquivo:linha` em `D:\git\phd-research\text\latex\` (somente leitura; só cito o que li). Tipos de alteração: **nova subseção**, **nota de rodapé**, **revisão de afirmação**, **trabalho futuro**. A força da evidência reflete o que a seção de origem permite (36 positivos, uma realização do LLM, humano perfeito).

| # | Achado | Ponto da tese | Relação | Alteração sugerida | Força da evidência |
|---|---|---|---|---|---|
| 1 | A1: taxas do Haiku com nota oculta (FP 0,4%; FN 44,4% abaixo do corte) | `estudo4_gzcmd.tex:89` (o dossiê "reúne também a nota final"; saídas decisórias retiradas); `apendice_gzcmd.tex:281–293` (ablação do dossiê: retirada de `model_outputs`, 18 pares) | **Estende**: é uma segunda ablação (a nota final), em amostra estratificada | Nova subseção no apêndice, após a ablação do dossiê, ou nota de rodapé na seção do revisor | Moderada para as taxas (medição direta, IC reportados); fraca para qualquer inferência sobre o efeito da nota (sem comparação pareada) |
| 2 | A2: VPP projetado 2,9% [0,8–12,1] e 4,8% [1,5–20,9] contra 16,0% [6,4–34,7] | `estudo4_gzcmd.tex:311` (VPP 16,0%, 4/25, IC 6,4–34,7%; previstos 9,06 FP, observados 21); `conclusao.tex:27` e `:93` | **Refina**: mesma direção (VPP baixo), ordem de grandeza compatível, ICs parcialmente sobrepostos; reforça "o LLM não deve aceitar vínculos sozinho" | Nota de rodapé ou parágrafo na discussão: "medição posterior, com nota oculta e amostra estratificada, não sugere VPP maior". Não alterar o 16,0% | Fraca a moderada (reescala dependente da seleção; populações diferentes) |
| 3 | A (ressalva 5) e F: FP concentrados nos não pares de maior nota; o replay tem FP acima das taxas medidas | `estudo4_gzcmd.tex:311` ("proporções de erro ... estimadas em coortes com pares mais fáceis"); `:313` | **Refina** (hipótese): sugere que as proporções de erro por faixa não são homogêneas dentro da faixa | Trabalho futuro: medir FP do LLM por posição da nota dentro da faixa, com amostra independente | Fraca (hipótese não testada) |
| 4 | A: FN do LLM de 44,4% abaixo do corte (13,7% acima) | `estudo4_gzcmd.tex:271` (filtro de NONMATCH descartou 9 dos 22 PAR=2 do topo; VPN 87,8%); `conclusao.tex:27` (6 pares perdidos por NONMATCH) | **Confirma** a direção (o LLM deixa passar pares verdadeiros abaixo do corte) e **estende** com FN por faixa | Nota de rodapé na seção do filtro, citando a taxa medida | Moderada (16/36; IC 29,5–60,4%) |
| 5 | B1: a fila auto-calibrável perde das bases a orçamento humano igual (simulada e real) | `estudo4_gzcmd.tex:307` e `conclusao.tex:5`, `:25`, `:81` ("a redução veio da ordenação, não da automação"); `conclusao.tex:101` (fila adaptativa de Thompson) | **Confirma** (consistente; não contradiz) | Se mencionada, nota em trabalhos futuros: método experimental 0.7.0 **não avaliado na tese**; resultado negativo a orçamento igual. Sem alterar as conclusões | Moderada para a direção (várias configurações, simulação e execução real); fraca para magnitudes |
| 6 | D2: H8 real (S5 ≤ S0 com rendimento monotônico; ganhos só com não monotonia induzida) | `conclusao.tex:101` ("A fila adaptativa ... tem precedente ... nesta coorte exigiu em média 96 revisões para recuperar 23 pares verdadeiros, contra 125 de O3, em exploração posterior ao registro e sem validação") | **Refina/qualifica**: com a premissa de monotonia verificada no extrato, o bandit sobre faixas contíguas não supera a ordenação | Revisão de afirmação (ressalva): "no extrato real, o bandit sobre faixas contíguas não superou a ordenação por nota quando o rendimento cai com o posto" | Moderada a fraca (20 sementes; 36 positivos; indução do analista no lado B). Os desenhos da tese e do experimento diferem; não comparar diretamente os números |
| 7 | D3: H11 real refutada (0 de 16) | `estudo4_gzcmd.tex:279` (cenários de sensibilidade humana: "perdas de cobertura, não de erro humano"); `conclusao.tex:77` (o revisor humano não foi observado) | **Estende**: com erro humano simulado, o LLM real não audita bem (recuperação 26%–48%) | Trabalho futuro: o uso do LLM como auditor de decisões humanas fica sem apoio nesta prevalência | Moderada a fraca: a refutação vale no real (0 de 16 células), mas com humano simulado cujo erro é independente por par e uma realização do LLM |
| 8 | C1 e E3: LLM como desempate e como atributo (+4 a +6 em B=400; S3 +16 no real) | `conclusao.tex:53` ("a tese não sustenta ... que seus usos subordinados ao revisor humano reduzam a carga"); `estudo4_gzcmd.tex:271` (o filtro só supera O3 com H=22) e `:285` (custos, C_LLM) | **Refina**: o enunciado "não sustenta" continua correto, mas há indício exploratório de ganho com ~200–400 decisões, e custo de milhares de chamadas | Trabalho futuro (hipótese a validar); eventualmente nota de rodapé em `conclusao.tex:53`. Não reverter a conclusão | Fraca (uma realização; IC exclui 0 em 4 de 48; S3 supõe LLM gratuito) |
| 9 | E1: S2 (régua como offset + rótulos verificados) supera a régua no real | `conclusao.tex:81` e `:83` (recomendação: O3 ou F4); `estudo4_gzcmd.tex:327` | **Estende**: outra forma de aprender dos compostos (online, com rótulos verificados) | Trabalho futuro, com uma frase de ressalva; subseção exploratória no apêndice só se for útil mostrar o placebo e a replicação. Sem mudar a recomendação | **Fraca (rebaixada após a replicação por passo):** determinística; 36 positivos; o +16 reduz-se a +3 com a régua calibrada e, fora da amostra (passos 9 e 11), o ganho é de +1 a +4 pares somados, nunca negativo; R1–R3 inconclusivas |
| 10 | E2: a escala da régua explica 13 de 16 pares; Platt do extrato inteiro corta o ganho | `apendice_gzcmd.tex:96` (A⁺ com 95 pares, 97,9%; A⁻ com 39.981, três verdadeiros) e `:100–108` (calibração por âncoras e efeito sobre a decisão); `conclusao.tex:56` (a calibração por âncoras não é absoluta) | **Confirma/estende** a ressalva: a escala da calibração altera o que o aprendizado faz | Nota de rodapé nas âncoras: abaixo do corte há só 9 pares com nota ≥ 9 (< 30 âncoras), e a régua do experimento foi um *stub* | Moderada, e reforçada pela replicação por passo (com a régua plana a transferência rende +2, +4 e +7; com a calibrada, +2, +4 e +1); num único extrato |
| 11 | E3 (perfil "dados ausentes") | `capitulo_dados.tex:77` e `estudo4_gzcmd.tex:361` (revisão de origem por revisora única; hipótese de que o CdR precedeu a revisão, com viés de verificação e de incorporação) | **Estende e adverte**: o perfil foi lido nos próprios positivos e pode refletir como a revisora decidiu, não uma propriedade dos pares | Trabalho futuro: validar em positivos novos e em revisão cega (`estudo4_gzcmd.tex:341`). Não afirmar o perfil triplo: só a mãe ausente teve peso a favor de par fora da amostra (+0,12 a +0,48), enquanto a data de nascimento zerada pesou contra (−0,47 a −1,19) | **Muito fraca (rebaixada após a replicação por passo):** 12 pares em amostra; R4 inconclusiva por critério mal especificado; perfil triplo não reproduzido |
| 12 | E4: replicação por passo (R1–R4) | `capitulo_dados.tex:29` (REFREC e COMPREC agrupam as partições da tese: partição por óbito, distinta da por PASSO); `apendice_linkage_origem.tex:34–36` (passos 6, 9 e 11); `conclusao.tex:97–99` (critério prospectivo, coorte nova) | **R1–R4 inconclusivas** (positivos por passo 6/9/11 = 4/7/25; passo 6 não informativo) | Nenhuma alteração na tese além de, se S2 for mencionado, dizer que o ganho não se replicou fora da amostra com régua calibrada. É replicação interna por passo, não coorte nova, e não substitui o critério prospectivo | Fraca (4, 7 e 25 positivos; um par muda o status; determinística, sem IC) |
| 13 | S0 (nota decrescente) ≈ O2 | `estudo4_gzcmd.tex:331` (O2 alcança os 9 PAR=2 da confirmação com 11 revisões); `capitulo_dados.tex:75` (os 36 estão entre os 2.522 de maior escore) | **Consistente**: S0 acha 10 dos 36 em B=25 e 17 em B=800 | Nenhuma alteração; verificação cruzada de plausibilidade (pools e metas não são idênticos; não reanalisado) | Fraca, só plausibilidade |
| 14 | Variabilidade e uma realização do LLM | `estudo4_gzcmd.tex:95` (13 de 18 decisões repetidas pelo Haiku) | **Confirma** a limitação | Citar na ressalva de qualquer menção aos novos resultados | Moderada |
| 15 | H12 e revisores reais | `conclusao.tex:77` e `:101` ("Resta substituir o revisor simulado por revisão humana observada") | **Confirma a necessidade** | Trabalho futuro já presente; reforçar com o ponto "o ganho esperado do LLM é em tempo por decisão, não em número de decisões" (princípio 6 do documento de engenharia) | n/a (não testável offline) |

**Pontos em que a tese poderia ganhar uma frase, sem mudar conclusões.**

- Em `conclusao.tex:5`, `:25` e `:81`, a afirmação "a redução veio da ordenação" não é contrariada por nenhum dos achados; B1 e F1 a reforçam em outras condições.
- Em `conclusao.tex:53`, o "não sustenta" sobre usos subordinados do LLM pode ser qualificado por C1/E3, como hipótese exploratória.
- Em `conclusao.tex:101`, a referência à fila adaptativa de Thompson merece a ressalva do achado D2.

## I. Limitações gerais e próximos passos

**Limitações gerais.**

1. **36 positivos abaixo do corte.** Diferenças de 1–3 pares são ruído; os IC por semente cobrem só o desempate, o bandit e o erro humano simulado, não a incerteza de amostragem dos pares. S0, S2 e S3 são determinísticas no real e seus "IC" degeneram em ponto.
2. **Uma realização do LLM.** O fixture congela uma decisão por par; as sementes não a reamostram. O Haiku é estocástico (`estudo4_gzcmd.tex:95`).
3. **Humano perfeito e simulado.** Erro correlacionado com a dificuldade, fadiga, tempo e custo por decisão não entram. O humano que decide do zero e o que confirma um MATCH proposto custam o mesmo na simulação.
4. **Um só extrato e a referência operacional.** Revisora única, ordem e parada não documentadas, hipótese não verificada de que o CdR precedeu a revisão (`capitulo_dados.tex:75–77`). O que o aprendizado "descobre" pode ser, em parte, o que a revisora fez.
5. **Dados sintéticos escolhidos pelo analista.** Gerador, atributos fora da régua e subgrupo difícil são construções; o resultado "S3 muito melhor" depende de o LLM revisar de graça.
6. **Cobertura do fixture.** 53,8% das revisões da fila no `below_cut` e 20,5% das bases vêm do fixture; o resto é taxa suavizada.
7. **A amostra de calibração do LLM é estratificada**, com faixas muito pequenas (`low`, `high`).
8. **Parte dos achados é pós-resultado.** Os itens de §6.7 e §7 de `engenharia_de_filas.md` (placebo, régua mais íngreme, perfil) são exploratórios, não pré-registrados. A replicação por passo (§8) foi pré-registrada, mas o critério R4 estava mal especificado (régua e leitura de `mae:missing`), e as dobras têm 4, 7 e 25 positivos.
9. **Ainda não commitado.** Código, relatórios e CSV estão como não rastreados no momento da escrita.

**Próximos passos.**

- **Concluídos (nada mais a rodar com os dados atuais):** replicação por passo (seção E.5; R1–R4 inconclusivas), simulação com taxas medidas e replay do fixture (seção F), H1–H12 no sintético e no real (seção D). Reabrir E.5 só com um critério R4 reespecificado (régua e leitura de `mae:missing` fixadas antes) e, de preferência, com dados novos.
- **Replicação externa com outro extrato ou mais positivos abaixo do corte** (outra coorte, período ou par de bases): é o que falta para decidir se o ganho de S2 com régua calibrada (+1 a +4 pares) e o peso a favor de par da mãe ausente existem fora de um extrato com 36 positivos. Com 4, 7 e 25 positivos por passo, nenhuma replicação interna decide.
- **Estudo com revisores humanos reais:** medir tempo por decisão (H12), erro humano e concordância, e o efeito de ver ou não a nota e o veredito do LLM; sem isso, todos os números de esforço continuam condicionais a humano perfeito.
- **Perfil "dados ausentes" restrito à mãe ausente:** testar em positivos novos (e em revisão cega, `estudo4_gzcmd.tex:341`) se a mãe ausente é de fato sub-ponderada pela régua; descartar o perfil triplo, cuja data de nascimento zerada pesou contra par fora da amostra. Repetir H11 com humano cujo erro dependa da dificuldade; variar o tamanho da amostra de estimação (testar amostra menor, só no fim, ou estimador assistido por modelo).
- **Comparação pareada nota visível × nota oculta** nos mesmos pares, para separar o efeito da nota do efeito da população (seção A).
- **Várias realizações do LLM** (reamostrar decisões) para dar incerteza ao revisor real; fixture com cobertura independente do rótulo.
- **H12 (não testável offline):** estudo com revisores reais, medindo tempo por decisão, erro humano e concordância, com dupla revisão e referência cega (`estudo4_gzcmd.tex:341`, `conclusao.tex:101`).
- **Replicação externa no sentido da tese:** coorte ou período não usados para fixar cortes e regras, com ordenação e orçamento fixados antes dos desfechos e referência cega independente (`conclusao.tex:97–99`); a replicação por passo (E.5) não a substitui.

## J. Comparação com O1 (escore do OpenRecLink)

Detalhes, tabelas e reprodução: `docs/comparacao_o1.md` e `docs/comparacao_o1.csv` (`notebooks/exp_baseline_o1.py`; testes em `tests/test_exp_baseline_o1.py`). **Escopo:** offline (nenhum identificador de par foi enviado à API), 36 positivos abaixo do corte, uma realização do LLM (decisões reais do fixture; sem taxas sintéticas nas comparações com o LLM), humano perfeito.

**O que a O1 é aqui.** Ordem decrescente de `SCORE` (cabeçalho `SCORE,C,15,0`, vírgula decimal resolvida), sem rótulos, universo abaixo do corte (61.485). Reproduz a tese em 96 revisões: **O1 = 19, O2 = 12**. O `SCORE` tem só 82 valores distintos: o desempate muda o resultado (B = 50: 5 a 12; B = 200 e 400: 19 a 22 e 19 a 23), e a ordem do arquivo do extrato favorece a O1 (positivos mais cedo). Usa-se o desempate aleatório por semente (média [mín–máx]); a ordem do arquivo fica como variante.

**Tabela-síntese** (verdadeiros abaixo do corte, de 36; B = 50 / 96 / 100 / 200 / 400 / 800):

| estratégia | B=50 | B=96 | B=100 | B=200 | B=400 | B=800 |
|---|---|---|---|---|---|---|
| O1, ordem do arquivo (tese) | 12 | 19 | 19 | 21 | 23 | 25 |
| O1, desempate aleatório (média) | 8,7 | 19,0 | 19,1 | 19,8 | 21,4 | 25,0 |
| O2, nota final do CdR | 10 | 12 | 12 | 13 | 13 | 17 |
| S2x, régua O1 (Platt do extrato inteiro) | 9,8 | 12,0 | 12,1 | 13,8 | 19,6 | 27,8 |
| S2x, régua O2 plana | 14,0 | 15,0 | 15,0 | 17,0 | 19,0 | 33,0 |
| S5 (bandit) sobre faixas de O1 | 3,0 | 9,0 | 10,0 | 20,0 | 20,6 | 21,4 |
| LLM L1 em O1, W = 1000 (cobertura 12,5%) | 13,9 | 22,0 | 22,1 | 22,5 | 23,5 | 27,0 |

**Consequência para as conclusões (o que muda com a referência O1):**

1. **A referência é mais forte** (19 contra 12 em 96 revisões; 25 contra 17 em 800). Nenhuma das margens "sobre a régua" das seções A a I era sobre a O1.
2. **E1 (S2 supera a régua no real) enfraquece muito.** Contra a O1, S2 fica 1,6 a 7,1 pares abaixo em B = 96 a 400 (régua O1) e 2,2 a 4,1 (régua plana); só em B = 800 aparece ganho: +2,8 (régua O1) e +8,0 (régua plana; era +16 contra a O2). A S2 não foi ajustada à escala do `SCORE`.
3. **B1 (a fila padrão não supera as bases) se reforça**: a fila perde da O1 por 6,8 a 9,6 pares em B = 100 e 200 (taxas medidas e da tese) contra 0 a 2,5 da O2; S1 real 7,2 contra 19 a 25. Só o LLM perfeito em B = 50 (16,0) supera a O1.
4. **D2 (bandit) se mantém** contra a O1 (S5 ≤ O1 em quase todo B).
5. **C1/E3 (LLM como desempate/atributo)**: com a O1 a comparação é **limitada** pela cobertura do fixture (8% a 18% da janela, na variante limpa); o efeito vai de −3,5 a +5,7 pares (até +6), inconsistente (negativo em B = 96 e 100 nas janelas de 2.000 e 3.000), e em B = 96 (−3,5 a +2,9). Usar o fixture inteiro daria ganho inflado (+11,4 em B = 50), porque a cobertura depende do rótulo (100% dos positivos cobertos contra 19% a 32% dos pares).
6. **Ponta a ponta (B ≈ 127):** H0 pela O1 no pool `all` verifica 127 e revisa 0 pares abaixo do corte (a O1 põe antes os 211 acima do corte), por isso a métrica de verificados no pool `all` não serve; no pool abaixo do corte a O1 recupera 19,1 em B = 100, contra 4 da fila.
7. **Em B = 50 a O1 não é melhor que a O2** (8,7 contra 10 em média), o que limita a leitura de "a O1 domina".
8. **Implicação para a tese:** a recomendação da tese de ordenar abaixo do corte por O3/F4 não é afetada (não comparável aqui, sem padrão-ouro); mas qualquer menção aos achados de fila deve citar a O1 como referência sem rótulos e dizer que os ganhos de S2/LLM frente à O2 encolhem ou somem frente à O1 e dependem do desempate. Nada mais foi alterado na seção H.

**Ressalvas.** Uma realização do LLM; 36 positivos; S2 e S5 não ajustadas à O1; valores das filas existentes lidos dos CSVs (sem rodar de novo; pontos de B diferem entre as fontes); a regra de desempate aleatório é uma escolha do analista.
