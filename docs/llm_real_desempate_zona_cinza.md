# O LLM como desempate na zona cinza (abaixo do corte)

> **Aviso:** os vereditos do LLM são **uma única realização** (temperatura padrão); humano simulado **perfeito** (decide pelo PAR todo par que revisa); o experimento **não mede tempo por decisão**; exploratório, com apenas 36 verdadeiros. Apenas agregados; nada em nível de par.

## Desenho

- Pool: os 61.485 pares abaixo do corte, 36 verdadeiros (PAR ∈ {1, 2}, usado só como verdade). Ordem base **N**: nota decrescente, empate por `pair_id`.
- Janela W: os W primeiros de N recebem veredito do Haiku (prompt v4, sem `model_outputs` e sem `features.nota_final`). O LLM não decide nada e nenhum par é descartado: cada estratégia é uma permutação do pool e o humano decide cada par que revisa.
- **N**: ordem da nota. **L1**: na janela, MATCH, depois UNSURE, depois NONMATCH, cada grupo na ordem da nota; depois o resto em N. **L2**: blocos de nota de largura 0,5 (decrescentes); em cada bloco, MATCH → UNSURE → NONMATCH na ordem da nota; depois o resto em N. **L3**: como L1, mas os MATCH por confiança do LLM decrescente e depois pela nota.
- Janelas: 500, 1000, 2000, 3000; orçamentos de decisões humanas B: 25, 50, 100, 200, 400, 800, 1600. Com B ≥ W, L e N revisam o mesmo conjunto (diferença 0 por construção).
- Incerteza: bootstrap pareado da diferença L − N (2000 réplicas, semente 20261115), reamostrando os pares da janela com reposição, estratificado por verdade; IC percentil de 95%. Com 36 positivos, diferenças de 1–3 pares são ruído.

## O LLM na janela

| W | verdadeiros na janela | LLM MATCH / UNSURE / NONMATCH entre os verdadeiros | não pares com MATCH (FP) | não pares com UNSURE | MATCH totais | árbitro |
|--:|--:|---|--:|--:|--:|--:|
| 500 | 15 | 8 / 0 / 7 | 12 | 14 | 20 | 43 |
| 1000 | 17 | 10 / 0 / 7 | 23 | 36 | 33 | 95 |
| 2000 | 19 | 11 / 0 / 8 | 48 | 57 | 59 | 204 |
| 3000 | 20 | 11 / 1 / 8 | 58 | 90 | 69 | 323 |

Verdadeiros por quartil de posição na janela (MATCH / UNSURE / NONMATCH):

| W | Q1 (topo) | Q2 | Q3 | Q4 |
|--:|---|---|---|---|
| 500 | 6 / 0 / 6 | 0 / 0 / 1 | 0 / 0 / 0 | 2 / 0 / 0 |
| 1000 | 6 / 0 / 7 | 2 / 0 / 0 | 2 / 0 / 0 | 0 / 0 / 0 |
| 2000 | 8 / 0 / 7 | 2 / 0 / 0 | 1 / 0 / 0 | 0 / 0 / 1 |
| 3000 | 10 / 0 / 7 | 1 / 0 / 0 | 0 / 1 / 1 | 0 / 0 / 0 |

## Verdadeiros abaixo do corte encontrados (de 36) a B decisões

**W = 500** (valor; diferença contra N entre parênteses)

| estratégia | B=25 | B=50 | B=100 | B=200 | B=400 | B=800 | B=1600 |
|---|--:|--:|--:|--:|--:|--:|--:|
| N | 10 | 10 | 12 | 13 | 13 | 17 | 19 |
| L1 | 8 (-2) | 12 (+2) | 14 (+2) | 15 (+2) | 15 (+2) | 17 (+0) | 19 (+0) |
| L2 | 10 (+0) | 10 (+0) | 12 (+0) | 13 (+0) | 15 (+2) | 17 (+0) | 19 (+0) |
| L3 | 8 (-2) | 12 (+2) | 14 (+2) | 15 (+2) | 15 (+2) | 17 (+0) | 19 (+0) |

**W = 1000** (valor; diferença contra N entre parênteses)

| estratégia | B=25 | B=50 | B=100 | B=200 | B=400 | B=800 | B=1600 |
|---|--:|--:|--:|--:|--:|--:|--:|
| N | 10 | 10 | 12 | 13 | 13 | 17 | 19 |
| L1 | 8 (-2) | 10 (+0) | 14 (+2) | 17 (+4) | 17 (+4) | 17 (+0) | 19 (+0) |
| L2 | 10 (+0) | 10 (+0) | 12 (+0) | 13 (+0) | 17 (+4) | 17 (+0) | 19 (+0) |
| L3 | 10 (+0) | 10 (+0) | 14 (+2) | 17 (+4) | 17 (+4) | 17 (+0) | 19 (+0) |

**W = 2000** (valor; diferença contra N entre parênteses)

| estratégia | B=25 | B=50 | B=100 | B=200 | B=400 | B=800 | B=1600 |
|---|--:|--:|--:|--:|--:|--:|--:|
| N | 10 | 10 | 12 | 13 | 13 | 17 | 19 |
| L1 | 8 (-2) | 11 (+1) | 11 (-1) | 17 (+4) | 18 (+5) | 18 (+1) | 19 (+0) |
| L2 | 10 (+0) | 10 (+0) | 12 (+0) | 13 (+0) | 17 (+4) | 17 (+0) | 19 (+0) |
| L3 | 7 (-3) | 11 (+1) | 11 (-1) | 17 (+4) | 18 (+5) | 18 (+1) | 19 (+0) |

**W = 3000** (valor; diferença contra N entre parênteses)

| estratégia | B=25 | B=50 | B=100 | B=200 | B=400 | B=800 | B=1600 |
|---|--:|--:|--:|--:|--:|--:|--:|
| N | 10 | 10 | 12 | 13 | 13 | 17 | 19 |
| L1 | 8 (-2) | 11 (+1) | 11 (-1) | 16 (+3) | 19 (+6) | 19 (+2) | 19 (+0) |
| L2 | 10 (+0) | 10 (+0) | 12 (+0) | 13 (+0) | 17 (+4) | 17 (+0) | 19 (+0) |
| L3 | 7 (-3) | 11 (+1) | 11 (-1) | 16 (+3) | 19 (+6) | 19 (+2) | 19 (+0) |

Decisões humanas por verdadeiro encontrado (B / achados):

| W | estratégia | B=25 | B=50 | B=100 | B=200 | B=400 | B=800 | B=1600 |
|--:|---|--:|--:|--:|--:|--:|--:|--:|
| 500 | N | 2,5 | 5,0 | 8,3 | 15,4 | 30,8 | 47,1 | 84,2 |
| 500 | L1 | 3,1 | 4,2 | 7,1 | 13,3 | 26,7 | 47,1 | 84,2 |
| 500 | L2 | 2,5 | 5,0 | 8,3 | 15,4 | 26,7 | 47,1 | 84,2 |
| 500 | L3 | 3,1 | 4,2 | 7,1 | 13,3 | 26,7 | 47,1 | 84,2 |
| 1000 | N | 2,5 | 5,0 | 8,3 | 15,4 | 30,8 | 47,1 | 84,2 |
| 1000 | L1 | 3,1 | 5,0 | 7,1 | 11,8 | 23,5 | 47,1 | 84,2 |
| 1000 | L2 | 2,5 | 5,0 | 8,3 | 15,4 | 23,5 | 47,1 | 84,2 |
| 1000 | L3 | 2,5 | 5,0 | 7,1 | 11,8 | 23,5 | 47,1 | 84,2 |
| 2000 | N | 2,5 | 5,0 | 8,3 | 15,4 | 30,8 | 47,1 | 84,2 |
| 2000 | L1 | 3,1 | 4,5 | 9,1 | 11,8 | 22,2 | 44,4 | 84,2 |
| 2000 | L2 | 2,5 | 5,0 | 8,3 | 15,4 | 23,5 | 47,1 | 84,2 |
| 2000 | L3 | 3,6 | 4,5 | 9,1 | 11,8 | 22,2 | 44,4 | 84,2 |
| 3000 | N | 2,5 | 5,0 | 8,3 | 15,4 | 30,8 | 47,1 | 84,2 |
| 3000 | L1 | 3,1 | 4,5 | 9,1 | 12,5 | 21,1 | 42,1 | 84,2 |
| 3000 | L2 | 2,5 | 5,0 | 8,3 | 15,4 | 23,5 | 47,1 | 84,2 |
| 3000 | L3 | 3,6 | 4,5 | 9,1 | 12,5 | 21,1 | 42,1 | 84,2 |

## Bootstrap pareado da diferença L − N

| W | estratégia | B=50 | B=100 | B=200 | B=400 |
|--:|---|---|---|---|---|
| 500 | L1 | +2 [-2; +5] | +2 [-3; +4] | +2 [+0; +5] | +2 [+0; +5] |
| 500 | L2 | +0 [+0; +0] | +0 [+0; +0] | +0 [+0; +0] | +2 [+0; +5] |
| 500 | L3 | +2 [-2; +5] | +2 [-3; +4] | +2 [+0; +5] | +2 [+0; +5] |
| 1000 | L1 | +0 [-6; +5] | +2 [-3; +6] | +4 [+0; +7] | +4 [+0; +7] |
| 1000 | L2 | +0 [+0; +0] | +0 [+0; +0] | +0 [+0; +0] | +4 [+0; +7] |
| 1000 | L3 | +0 [-6; +5] | +2 [-3; +6] | +4 [+0; +7] | +4 [+0; +7] |
| 2000 | L1 | +1 [-5; +7] | -1 [-7; +6] | +4 [-1; +8] | +5 [+1; +8] |
| 2000 | L2 | +0 [+0; +0] | +0 [+0; +0] | +0 [+0; +0] | +4 [+0; +7] |
| 2000 | L3 | +1 [-5; +7] | -1 [-7; +6] | +4 [-1; +8] | +5 [+1; +8] |
| 3000 | L1 | +1 [-5; +7] | -1 [-7; +5] | +3 [-3; +9] | +6 [+1; +9] |
| 3000 | L2 | +0 [+0; +0] | +0 [+0; +0] | +0 [+0; +0] | +4 [+0; +7] |
| 3000 | L3 | +1 [-5; +6] | -1 [-7; +5] | +3 [-3; +9] | +6 [+1; +9] |

Cada célula: diferença observada [IC 95% do bootstrap]. A curva completa (achados × B, de 10 em 10) está no CSV, tabela `curva`.

## Custo

| W | revisões do LLM | chamadas à API | tokens de entrada | tokens de saída | chamadas por par extra (L1 / L2 / L3) em B=100 e B=400 |
|--:|--:|--:|--:|--:|---|
| 500 | 500 | 1.055 | 775.828 | 1.838.761 | B=100: 528 / n/d / 528; B=400: 528 / 528 / 528 |
| 1000 | 1.000 | 2.117 | 1.566.592 | 3.711.273 | B=100: 1058 / n/d / 1058; B=400: 529 / 529 / 529 |
| 2000 | 2.000 | 4.253 | 3.156.575 | 7.491.542 | B=100: n/d / n/d / n/d; B=400: 851 / 1063 / 851 |
| 3000 | 3.000 | 6.411 | 4.763.885 | 11.336.520 | B=100: n/d / n/d / n/d; B=400: 1068 / 1603 / 1068 |

`n/d` = a estratégia não achou mais verdadeiros que N nesse B. As chamadas contam todas as feitas para a janela (incluindo as reaproveitadas do cache de execuções anteriores, que foram pagas antes).

Revisões novas na API neste teste: 1.514 (teto 2500; usadas 1514); tempo de relógio 1.460 s; workers 24; HTTP 429/erros: nenhum.

## Discussão

- W=500: 15 dos 36 verdadeiros estão na janela; o LLM marcou 8 como MATCH, 0 como UNSURE e 7 como NONMATCH, e deu MATCH a 12 não pares.
  - L1: diferença contra N em B=50/100/200/400 = +2/+2/+2/+2.
  - L2: diferença contra N em B=50/100/200/400 = +0/+0/+0/+2.
  - L3: diferença contra N em B=50/100/200/400 = +2/+2/+2/+2.
- W=1000: 17 dos 36 verdadeiros estão na janela; o LLM marcou 10 como MATCH, 0 como UNSURE e 7 como NONMATCH, e deu MATCH a 23 não pares.
  - L1: diferença contra N em B=50/100/200/400 = +0/+2/+4/+4.
  - L2: diferença contra N em B=50/100/200/400 = +0/+0/+0/+4.
  - L3: diferença contra N em B=50/100/200/400 = +0/+2/+4/+4.
- W=2000: 19 dos 36 verdadeiros estão na janela; o LLM marcou 11 como MATCH, 0 como UNSURE e 8 como NONMATCH, e deu MATCH a 48 não pares.
  - L1: diferença contra N em B=50/100/200/400 = +1/-1/+4/+5.
  - L2: diferença contra N em B=50/100/200/400 = +0/+0/+0/+4.
  - L3: diferença contra N em B=50/100/200/400 = +1/-1/+4/+5.
- W=3000: 20 dos 36 verdadeiros estão na janela; o LLM marcou 11 como MATCH, 1 como UNSURE e 8 como NONMATCH, e deu MATCH a 58 não pares.
  - L1: diferença contra N em B=50/100/200/400 = +1/-1/+3/+6.
  - L2: diferença contra N em B=50/100/200/400 = +0/+0/+0/+4.
  - L3: diferença contra N em B=50/100/200/400 = +1/-1/+3/+6.
- Faixa das diferenças contra N entre as janelas (mín a máx):
  - L1: B=25: -2 a -2; B=50: +0 a +2; B=100: -1 a +2; B=200: +2 a +4; B=400: +2 a +6; B=800: +0 a +2.
  - L2: B=25: +0 a +0; B=50: +0 a +0; B=100: +0 a +0; B=200: +0 a +0; B=400: +2 a +4; B=800: +0 a +0.
  - L3: B=25: -3 a +0; B=50: +0 a +2; B=100: -1 a +2; B=200: +2 a +4; B=400: +2 a +6; B=800: +0 a +2.
- L1 e L3 acharam o mesmo número de verdadeiros em 25 de 28 células (janela × B).
- Menor custo por par extra entre B=50..400: 528 chamadas (L1, W=500, B=50).
- Em 4 de 48 combinações (estratégia, W, B ∈ {50,100,200,400} com B < W) o IC 95% do bootstrap exclui zero.
- Maior diferença observada: +6 verdadeiros (L1, W=3000, B=400), em 36 positivos: diferenças de 1–3 pares são ruído.
- Custo (chamadas à API por janela): W=500: 1.055 chamadas, W=1000: 2.117 chamadas, W=2000: 4.253 chamadas, W=3000: 6.411 chamadas; o humano decide o mesmo número B de pares em todas as estratégias.

