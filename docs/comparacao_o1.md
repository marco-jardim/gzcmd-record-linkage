# Comparação das estratégias com a ordenação O1 (escore do OpenRecLink)

> **Aviso de escopo.** Análise **offline** (sem chamadas à API; nenhum identificador de par foi enviado a serviço externo). Extrato único da tese, pares abaixo do corte (61.485; **36 verdadeiros**); `PAR` só como verdade; humano **perfeito**; os vereditos do LLM vêm **só** das decisões reais do fixture local (uma realização congelada; sem taxas sintéticas na comparação com o LLM). Exploratório e pós-resultado; apenas agregados; nada em nível de par. Código: `notebooks/exp_baseline_o1.py`; dados: `docs/comparacao_o1.csv`; testes: `tests/test_exp_baseline_o1.py`.

## 1. Por que esta comparação

Todos os experimentos anteriores deste repositório compararam as estratégias com a ordem pela **nota final do CdR** (O2 na tese), que recupera 12 dos 36 pares verdadeiros abaixo do corte com 96 revisões. A tese tem também a **O1**, a ordem pelo escore do OpenRecLink (`SCORE`), igualmente **sem rótulos** (`glossario.tex:86`; seção `sec:e4-mesma-carga` de `estudo4_gzcmd.tex`), que recupera 19 com 96 revisões. A O3 (XGBoost fora da amostra, treinado com `PAR`) e as fusões não são comparáveis no cenário sem padrão-ouro. Aqui as comparações principais são refeitas contra a O1.

## 2. Reprodução dos números da tese e o desempate

`SCORE` é lido do cabeçalho `SCORE,C,15,0` (vírgula decimal resolvida pelo carregador) no universo abaixo do corte (`below_cutoff` com a partição esperada 211/61.485/36). **Reproduz a tese em 96 revisões: O1 = 19 e O2 = 12** (ordem do arquivo como desempate).

**O desempate importa para a O1.** O `SCORE` tem só 82 valores distintos nos 61.485 pares (a nota tem 703): a O1 tem blocos grandes de empate. A ordem do arquivo do extrato **não é neutra** (os positivos aparecem mais cedo que os demais), então desempatar pela posição no arquivo é otimista. O desempate principal deste documento é **aleatório e determinístico por semente** (permutação das linhas; 30 sementes nas comparações com S2/S5/LLM e 300 na tabela abaixo); a ordem do arquivo fica como variante rotulada. A O2 não depende do desempate (mínimo = máximo).

| estratégia | B=50 | B=96 | B=100 | B=200 | B=400 | B=800 |
|---|---|---|---|---|---|---|
| O1, ordem do arquivo (reproduz a tese) | 12 | 19 | 19 | 21 | 23 | 25 |
| O1, desempate aleatório: média [mín–máx], 300 sorteios | 8,7 [5–12] | 19,0 [19–20] | 19,1 [19–20] | 19,8 [19–22] | 21,4 [19–23] | 25,0 [25–25] |
| O2 (nota final do CdR) | 10 | 12 | 12 | 13 | 13 | 17 |

Em 96 e 100 revisões a O1 é estável (19 a 20 pares); em B = 50 o desempate decide (5 a 12, média 8,7, contra 12 pela ordem do arquivo) e em B = 200 e 400 também (19 a 22 e 19 a 23).

## 3. Tabela-síntese: estratégia × B (verdadeiros abaixo do corte recuperados, de 36)

O1 e O2 lado a lado; as demais linhas são médias entre as 30 sementes de desempate da O1 (a ordem do arquivo não se aplica a S2/S5). `S2x` = exploração (sem a amostra de estimação); `S2` = com 5% de B em amostra. Cobertura do LLM nas linhas L1: ver 4.d.

| estratégia | B=50 | B=96 | B=100 | B=200 | B=400 | B=800 |
|---|---|---|---|---|---|---|
| **O1** arquivo (tese) | 12 | 19 | 19 | 21 | 23 | 25 |
| **O1** aleatório, média [mín–máx] | 8,7 [5–12] | 19,0 [19–20] | 19,1 [19–20] | 19,8 [19–22] | 21,4 [19–23] | 25,0 [25–25] |
| **O2** nota final | 10 | 12 | 12 | 13 | 13 | 17 |
| S2x, régua O1 (Platt do extrato inteiro) | 9,8 | 12,0 | 12,1 | 13,8 | 19,6 | 27,8 |
| S2x, régua O1 (identidade: SCORE como logit) | 4,0 | 12,1 | 12,1 | 14,0 | 19,3 | 27,8 |
| S2x, régua O2 plana (stub nota/10) | 14,0 | 15,0 | 15,0 | 17,0 | 19,0 | 33,0 |
| S2x, régua O2 calibrada (Platt do extrato inteiro) | 10,0 | 12,0 | 12,0 | 13,0 | 18,0 | 20,0 |
| S2 (com amostra), régua O1 (Platt) | 8,8 | 12,0 | 12,0 | 13,6 | 18,3 | 27,8 |
| S2 (com amostra), régua O2 plana | 14,0 | 15,0 | 15,0 | 17,0 | 19,0 | 33,0 |
| S5 bandit sobre faixas de O1 | 3,0 | 9,0 | 10,0 | 20,0 | 20,6 | 21,4 |
| S5 bandit sobre faixas de O2 | 10,0 | 10,0 | 10,0 | 12,0 | 13,0 | 13,0 |
| LLM L1 em O1, W=500 (cobertura ~18%) | 12,3 | 19,8 | 19,9 | 20,3 | 21,3 | 25,0 |
| LLM L1 em O1, W=1000 (cobertura ~12%) | 13,9 | 22,0 | 22,1 | 22,5 | 23,5 | 27,0 |
| LLM L1 em O1, W=2000 (cobertura ~10%) | 12,0 | 17,6 | 18,7 | 25,3 | 26,4 | 30,0 |
| LLM L1 em O1, W=3000 (cobertura ~8%) | 11,0 | 15,6 | 16,0 | 25,3 | 26,3 | 30,0 |

Diferença média contra a O1 do **mesmo desempate** (pareada por semente):

| estratégia | B=50 | B=96 | B=100 | B=200 | B=400 | B=800 |
|---|---|---|---|---|---|---|
| O2 | +1,4 | -7,1 | -7,1 | -6,6 | -8,2 | -8,0 |
| S2x, régua O1 (Platt) | +1,2 | -7,1 | -7,0 | -5,8 | -1,6 | +2,8 |
| S2x, régua O1 (identidade) | -4,6 | -7,0 | -7,0 | -5,6 | -1,9 | +2,8 |
| S2x, régua O2 plana | +5,4 | -4,1 | -4,1 | -2,6 | -2,2 | +8,0 |
| S2x, régua O2 calibrada | +1,4 | -7,1 | -7,1 | -6,6 | -3,2 | -5,0 |
| S5 sobre O1 | -5,6 | -10,1 | -9,1 | +0,4 | -0,6 | -3,6 |
| L1 em O1, W=1000 | +5,3 | +2,9 | +3,0 | +2,9 | +2,4 | +2,0 |
| L1 em O1, W=2000 | +3,4 | -1,5 | -0,4 | +5,7 | +5,2 | +5,0 |
| L1 em O1, W=3000 | +2,4 | -3,5 | -3,1 | +5,7 | +5,2 | +5,0 |

## 4. Resultados por item

### 4.a Humano sozinho: O1 contra O2

Com o desempate aleatório a O1 supera a O2 por 7 pares em 96 e 100 revisões (19,0 contra 12), 6 a 8 em B = 200 a 800 (19,8 contra 13; 21,4 contra 13; 25 contra 17) e **fica abaixo da O2 em B = 50** (8,7 contra 10; faixa 5 a 12 conforme o desempate). A conclusão da tese (a O1 recupera mais que a O2 nas cargas de 96 em diante) se mantém.

### 4.b Fila com aprendizado (S2) com a régua O1 e com a O2

Régua O1: `offset = a + b·SCORE`, com (a, b) do `fit_anchor_platt` no `SCORE` do extrato inteiro (A⁺ = SCORE ≥ 9 com média 0,979; A⁻ = SCORE < 5 com média 1e-4; mesmos parâmetros da nota; intercepto −5,57 e inclinação 0,843 por ponto de SCORE), ou a identidade (SCORE tomado como logit). Os demais coeficientes são os de S2 (σ = 0,5; reajuste a cada 25 decisões). **Contra a O1, S2 não ganha em B de 96 a 400 em nenhuma régua**: −7,1 a −1,6 pares (régua O1) e −4,1 a −2,2 (régua O2 plana, a melhor contra a O2); só em B = 800 aparece ganho, de +2,8 (régua O1) e +8,0 (régua O2 plana, a que dava +16 contra a O2). Em B = 50 a S2 com régua O1-identidade perde 4,6 (o intercepto da identidade está muito acima da taxa real) e com Platt empata (+0,2). A S2 não foi ajustada à escala do `SCORE` (prior do intercepto e da inclinação fixados para a nota), e a O1 tem blocos de empate que a S2 reordena pelos compostos: o resultado vale para S2 como definida.

### 4.c Bandit de Thompson sobre faixas de posição da O1

Faixas 1%, 5%, 20% e resto da posição na O1. S5 fica **abaixo da O1** em quase todo B (diferença média −5,6 em B = 50; −10,1 em 96; −9,1 em 100; +0,4 em 200; −0,6 em 400; −3,6 em 800): como com a O2 (H8), com rendimento que cai com a posição o bandit não ajuda; o lado "só com não monotonia" não foi refeito aqui.

### 4.d LLM como desempate sobre janelas da O1 (decisões reais do fixture)

O fixture cobre pares guiados pela verdade (a amostra de calibração e as execuções anteriores cobrem **todos** os 36 positivos), de modo que usá-lo em toda a janela da O1 vazaria o rótulo pela cobertura. A variante **limpa** (a que vale) só aceita veredito de pares dos 3.000 primeiros da O2, cobertura que depende da nota e não do rótulo; os demais pares da janela ficam "sem veredito" e são tratados como não priorizados (L1: MATCH, depois UNSURE, depois o resto na ordem da O1). Nenhuma taxa sintética é usada.

| janela W da O1 | pares cobertos (limpa) | % da janela | positivos na janela | positivos cobertos (limpa) | pares cobertos (fixture bruto) | positivos cobertos (bruto) |
|---|---|---|---|---|---|---|
| 500 | 89 | 17,8% | 22 | 9,8 | 160 (32,1%) | 22 |
| 1000 | 125 | 12,5% | 28 | 12,0 | 224 (22,4%) | 28 |
| 2000 | 190 | 9,5% | 31 | 15,0 | 468 (23,4%) | 31 |
| 3000 | 253 | 8,4% | 36 | 20,0 | 576 (19,2%) | 36 |

**A cobertura é baixa (8% a 18% da janela)**, então a comparação é **limitada**: o LLM só pode reordenar a fração coberta. No fixture bruto 100% dos positivos da janela estão cobertos contra 19% a 32% dos pares: a cobertura depende do rótulo. Resultado limpo, diferença contra a O1 (pares): W=500 +3,7 (B=50), +0,8 (96 e 100), +0,7, +0,2 e 0,0; W=1000 +5,3, +2,9, +3,0, +2,9, +2,4, +2,0; W=2000 +3,4, −1,5, −0,4, +5,7, +5,2, +5,0; W=3000 +2,4, −3,5, −3,1, +5,7, +5,2, +5,0 (B = 50, 96, 100, 200, 400, 800). O fixture bruto dá +11,4 e +5,9 (W=1000, B = 50 e 96), números inflados pela cobertura e **não** devem ser usados. Com a O1, o efeito do LLM vai de −3,5 a +5,7 pares (até +6) conforme W e B, inconsistente: negativo em B = 96 e 100 com W = 2.000 e 3.000, não consistente em B = 96 (−3,5 a +2,9) e dependente de poucos pares cobertos.

### 4.e A fila auto-calibrável existente contra a curva da O1

Leitura dos CSVs existentes (`sim_fila_autocalibravel_resultados.csv`, `engenharia_de_filas_resultados.csv`), abaixo do corte, humano perfeito; pontos de B em que cada fonte reporta (— = não reportado).

| estratégia | B=50 | B=96 | B=100 | B=200 | B=400 | B=800 |
|---|---|---|---|---|---|---|
| O1 arquivo | 12 | 19 | 19 | 21 | 23 | 25 |
| O1 aleatório (média) | 8,7 | 19,0 | 19,1 | 19,8 | 21,4 | 25,0 |
| O2 (H0 da simulação) | 10 | 12 | 12 | 13 | 13 | 17 |
| Fila (sim.), replay do fixture | 6,1 | — | — | — | — | — |
| Fila (sim.), LLM nas taxas medidas | 10,1 | — | 11,8 | — | — | — |
| Fila (sim.), LLM nas taxas da tese | 8,3 | — | 9,5 | 13,0 | — | — |
| Fila (sim.), LLM perfeito | 16,0 | — | — | — | — | — |
| S1 real (engenharia de filas) | 6,2 | — | 7,2 | 7,2 | 7,2 | 7,2 |

A fila perde da O1 em todo B reportado a partir de 100, com qualquer revisor simulado ou real (ex.: 11,8 em B = 100 com taxas medidas contra 19,1; 13,0 em B = 200 com taxas da tese contra 19,8; S1 real 7,2 em B = 100 a 800 contra 19 a 25). Em B = 50, onde a O1 é instável (5 a 12; média 8,7), a fila com o replay do fixture (6,1) fica abaixo, a com taxas medidas (10,1) entre a média e a ordem do arquivo, e só o revisor **perfeito** (16,0) a supera com folga, o que depende de o LLM não errar nunca.

### 4.f Execução real ponta a ponta (~127 decisões) com H0 pela O1

A execução real (pool `all`, 61.696 pares, 247 verdadeiros, 36 abaixo do corte) teve, com ~127 decisões humanas, fila 96 verificados (4 abaixo do corte), H1 106 (10) e H0 pela O2 113 (9). H0 pela **O1** no mesmo pool:

| H0 | B=50 verif. / abaixo | B=100 | B=127 | B=200 |
|---|---|---|---|---|
| O1, ordem do arquivo | 50 / 0 | 100 / 0 | 127 / 0 | 193 / 2 |
| O1, desempate aleatório (média de 300) | 50 / 0 | 100 / 0 | 127 / 0 | 193 / 2 |
| O2, ordem do arquivo | 49 / 5 | 98 / 9 | 113 / 9 | 127 / 12 |
| O2, desempate aleatório (média de 300) | 49 / 5 | 98 / 8,7 | 113,8 / 9,2 | 127 / 12 |

No pool `all` a O1 põe primeiro os 211 pares acima do corte (todos verdadeiros, já aceitos pela regra), de modo que H0 pela O1 "verifica" 127 em 127 decisões e **ainda não revisa nenhum par abaixo do corte** (0 até B = 127; 2 em B = 200): a métrica de verificados no pool `all` recompensa re-verificar o que o corte já aceita. A comparação que importa para a tese é a do pool abaixo do corte (4.a): a O1 recupera 19,1 em B = 100 e 19,8 em B = 200, contra 4 da fila no pool `all` com 127 decisões e 9 da H0 pela O2.

## 5. O que muda quando a referência é a O1

1. **A referência é mais forte que a O2**: 19 contra 12 em 96 revisões e 25 contra 17 em B = 800. Toda margem sobre a O2 encolhe ou desaparece.
2. **O ganho de S2 do extrato real some até B = 400 e encolhe em 800** (de +16 contra a O2 para +8 com a régua plana e +2,8 com a régua O1). Com a O1 o ganho que sobrevive é pequeno, só em B alto e dependente de a S2 usar a escala plana.
3. **A fila auto-calibrável perde da O1 com margem maior**: 6,8 a 9,6 pares em B = 100 e 200 (taxas medidas e da tese) contra 0 a 2,5 da O2.
4. **O bandit e as hipóteses sobre rendimento monotônico se mantêm** (S5 ≤ O1).
5. **O LLM como desempate só pode ser avaliado com cobertura limitada** (8% a 18% da janela, limpa) e o efeito vai de −3,5 a +5,7 (até +6), inconsistente (negativo em B = 96 e 100 nas janelas de 2.000 e 3.000); o fixture bruto inflaria o ganho (+11 em B = 50).
6. **A O1 é sensível ao desempate** (B = 50: 5 a 12; B = 200 e 400: 19 a 23), e a ordem do arquivo favorece a O1: relatar sempre a média e a faixa.
7. **Em B = 50 a O1 não supera a O2** (8,7 contra 10 em média), onde a S2 com régua plana e o revisor perfeito ainda ganham (14; 16).

## 6. Ressalvas e correções

- Uma realização do LLM congelada; 36 positivos (diferenças de 1 a 3 pares são ruído); humano perfeito; extrato único; a O1 é determinística exceto pelo desempate, que é o que as sementes variam (o resto é determinístico ou varia pelo bandit).
- A S2 com a régua O1 usa um offset construído aqui (Platt por âncoras no `SCORE`) e priors fixados para a nota; pode haver configuração melhor. A S2 e a S5 não são ajustadas para a O1.
- Os valores das filas existentes são lidos dos CSVs (sem rodar de novo); os pontos de B diferem entre as fontes.
- **Erratum E1 (revisão factual da tese).** No relatório do H8 real (`engenharia_de_filas_real_h8_h11.md`, linhas 70, 80 e 151) e em `engenharia_de_filas.md` §6.4 aparece "1,95 contra 17,00 verdadeiros em B = 800". O 1,95 é do **bandit S5 com a permutação N1** em B = 800; a ordem pela nota (S0) com a mesma permutação teve **1,25** (N1) e **0** (N2); os 17,00 são da S0 **sem** indução (cenário real). Os números certos em B = 800: N1: S0 1,25 e S5 1,95 (+0,70); N2: S0 0 e S5 0,50; cenário real: S0 17 e S5 15. O texto de `achados_filas_llm.md` foi corrigido; os dois outros arquivos não foram alterados nesta tarefa.

## 7. Reprodução

```
$env:PYTHONPATH = 'src;notebooks'
python notebooks/exp_baseline_o1.py --real-input ../phd-research/data/COMPARADORSEMIDENT.csv --seeds 30 --workers 10
python -m pytest tests/test_exp_baseline_o1.py -p no:warnings
```

Tempo: ~1 min (30 sementes, 10 processos). Saída: `docs/comparacao_o1.csv` (tabelas `reproducao`, `comparacao`, `cobertura_fixture`, `ponta_a_ponta_h0`, `fila_existente_vs_o1`, `transformacao_o1`).
