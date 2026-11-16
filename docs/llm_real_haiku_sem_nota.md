# Revisor LLM real (Claude Haiku, sem a nota) — taxas por faixa

> **Aviso:** medição com LLM real numa amostra estratificada; **não é avaliação da fila**. O dossiê tem só escores numéricos (mesmo protocolo da tese), sem `model_outputs` e sem `features.nota_final`; a nota está oculta ao LLM. Apenas agregados; nada em nível de par.

## Protocolo

- Modelo `claude-haiku-5-5`, prompt `v4`, Agente A + Agente B e árbitro só na divergência, `max_tokens=2500`, sem temperatura.
- Amostra (semente 20261115): **todos** os pares verdadeiros (247: 211 acima e 36 abaixo do corte) e até 100 não pares por faixa (470 no total); 717 pares.
- Faixas pela nota v3.1: low [0,5), grey_low [5,6), grey_mid [6,7), grey_high [7,8), near_high [8,9), high [9,∞). Lado do corte: `SCORE` contra o corte da tese por `PASSO`.
- Verdade: PAR ∈ {1, 2}. FP = MATCH entre não pares; FN = NONMATCH entre pares (UNSURE fica nos denominadores); UNSURE e árbitro sobre as decisões. IC de Wilson 95%.
- VPP do LLM como aceitador (MATCH = aceita): **VPP (amostra)** = VP/(VP+FP) na amostra, com IC de Wilson, superestimado porque só uma fração dos não pares entrou; **VPP (extrato)** reescala os não pares da célula (faixa × lado) para o total do extrato (mediana a posteriori e IC de Jeffreys por simulação, 20 mil sorteios).

## Taxas por faixa e lado do corte

| faixa | lado | pares | não pares | FP | FP % [IC] | FN | FN % [IC] | UNSURE % [IC] | árbitro % | VPP amostra % [IC] | VPP extrato % [IC] |
|---|---|--:|--:|--:|---|--:|---|---|---|---|---|
| low | todos | 3 | 100 | 0 | 0,0% [0,0–3,7] | 2 | 66,7% [20,8–93,9] | 4,9% [2,1–10,9] | 8,7% | 100,0% [20,7–100,0] | 1,1% [0,1–81,5] |
| low | acima | 2 | 0 | 0 | n/d | 1 | 50,0% [9,5–90,5] | 0,0% [0,0–65,8] | 0,0% | 100,0% [20,7–100,0] | 100,0% [100,0–100,0] |
| low | abaixo | 1 | 100 | 0 | 0,0% [0,0–3,7] | 1 | 100,0% [20,7–100,0] | 5,0% [2,1–11,1] | 8,9% | n/d | 0,0% [0,0–0,0] |
| grey_low | todos | 17 | 100 | 1 | 1,0% [0,2–5,4] | 5 | 29,4% [13,3–53,1] | 5,1% [2,4–10,7] | 4,3% | 88,9% [56,5–98,0] | 4,5% [1,2–35,2] |
| grey_low | acima | 8 | 0 | 0 | n/d | 0 | 0,0% [0,0–32,4] | 0,0% [0,0–32,4] | 0,0% | 100,0% [67,6–100,0] | 100,0% [100,0–100,0] |
| grey_low | abaixo | 9 | 100 | 1 | 1,0% [0,2–5,4] | 5 | 55,6% [26,7–81,1] | 5,5% [2,5–11,5] | 4,6% | 0,0% [0,0–79,3] | 0,0% [0,0–0,0] |
| grey_mid | todos | 57 | 100 | 0 | 0,0% [0,0–3,7] | 4 | 7,0% [2,8–16,7] | 5,1% [2,6–9,7] | 8,9% | 100,0% [92,6–100,0] | 77,4% [23,8–99,9] |
| grey_mid | acima | 48 | 0 | 0 | n/d | 1 | 2,1% [0,4–10,9] | 0,0% [0,0–7,4] | 6,2% | 100,0% [92,4–100,0] | 100,0% [100,0–100,0] |
| grey_mid | abaixo | 9 | 100 | 0 | 0,0% [0,0–3,7] | 3 | 33,3% [12,1–64,6] | 7,3% [3,8–13,8] | 10,1% | 100,0% [20,7–100,0] | 6,6% [0,6–96,6] |
| grey_high | todos | 43 | 100 | 1 | 1,0% [0,2–5,4] | 4 | 9,3% [3,7–21,6] | 2,8% [1,1–7,0] | 13,3% | 97,4% [86,5–99,5] | 75,0% [43,5–97,2] |
| grey_high | acima | 38 | 0 | 0 | n/d | 3 | 7,9% [2,7–20,8] | 5,3% [1,5–17,3] | 5,3% | 100,0% [89,6–100,0] | 100,0% [100,0–100,0] |
| grey_high | abaixo | 5 | 100 | 1 | 1,0% [0,2–5,4] | 1 | 20,0% [3,6–62,4] | 1,9% [0,5–6,7] | 16,2% | 80,0% [37,6–96,4] | 24,5% [7,7–78,9] |
| near_high | todos | 34 | 68 | 0 | 0,0% [0,0–5,3] | 10 | 29,4% [16,8–46,2] | 10,8% [6,1–18,3] | 5,9% | 100,0% [82,4–100,0] | 98,7% [87,8–100,0] |
| near_high | acima | 29 | 0 | 0 | n/d | 7 | 24,1% [12,2–42,1] | 20,7% [9,8–38,4] | 10,3% | 100,0% [80,6–100,0] | 100,0% [100,0–100,0] |
| near_high | abaixo | 5 | 68 | 0 | 0,0% [0,0–5,3] | 3 | 60,0% [23,1–88,2] | 6,8% [3,0–15,1] | 4,1% | 100,0% [34,2–100,0] | 89,7% [44,5–100,0] |
| high | todos | 93 | 2 | 0 | 0,0% [0,0–65,8] | 20 | 21,5% [14,4–30,9] | 16,8% [10,6–25,6] | 23,2% | 100,0% [93,7–100,0] | 99,7% [97,7–100,0] |
| high | acima | 86 | 0 | 0 | n/d | 17 | 19,8% [12,7–29,4] | 18,6% [11,8–28,1] | 25,6% | 100,0% [93,2–100,0] | 100,0% [100,0–100,0] |
| high | abaixo | 7 | 2 | 0 | 0,0% [0,0–65,8] | 3 | 42,9% [15,8–75,0] | 0,0% [0,0–29,9] | 0,0% | 100,0% [51,0–100,0] | 95,4% [75,0–100,0] |
| total | todos | 247 | 470 | 2 | 0,4% [0,1–1,5] | 45 | 18,2% [13,9–23,5] | 7,0% [5,3–9,1] | 10,5% | 98,8% [95,8–99,7] | 31,2% [11,2–67,8] |
| total | acima | 211 | 0 | 0 | n/d | 29 | 13,7% [9,7–19,0] | 11,4% [7,8–16,4] | 14,2% | 100,0% [97,6–100,0] | 100,0% [100,0–100,0] |
| total | abaixo | 36 | 470 | 2 | 0,4% [0,1–1,5] | 16 | 44,4% [29,5–60,4] | 5,1% [3,5–7,4] | 8,9% | 84,6% [57,8–95,7] | 2,9% [0,8–12,1] |
| total_sem_low | todos | 244 | 370 | 2 | 0,5% [0,1–1,9] | 43 | 17,6% [13,4–22,9] | 7,3% [5,5–9,7] | 10,7% | 98,8% [95,8–99,7] | 43,6% [19,0–80,1] |
| total_sem_low | acima | 209 | 0 | 0 | n/d | 28 | 13,4% [9,4–18,7] | 11,5% [7,8–16,5] | 14,4% | 100,0% [97,6–100,0] | 100,0% [100,0–100,0] |
| total_sem_low | abaixo | 35 | 370 | 2 | 0,5% [0,1–1,9] | 15 | 42,9% [28,0–59,1] | 5,2% [3,4–7,8] | 8,9% | 84,6% [57,8–95,7] | 4,8% [1,5–20,9] |

## Comparação com as taxas sintéticas da simulação

Taxas sintéticas: Laplace sobre a vigilância da tese (nota **visível**); `low` herdou `grey_low` (nunca medido). Medido: faixa inteira (os dois lados), nota **oculta**.

| faixa | FP sint. | FP medido [IC] | FN sint. | FN medido [IC] | UNSURE sint. | UNSURE medido [IC] |
|---|--:|---|--:|---|--:|---|
| low | 8,4% | 0,0% [0,0–3,7] | 75,0% | 66,7% [20,8–93,9] | 8,2% | 4,9% [2,1–10,9] |
| grey_low | 8,4% | 1,0% [0,2–5,4] | 75,0% | 29,4% [13,3–53,1] | 8,2% | 5,1% [2,4–10,7] |
| grey_mid | 2,7% | 0,0% [0,0–3,7] | 6,7% | 7,0% [2,8–16,7] | 6,0% | 5,1% [2,6–9,7] |
| grey_high | 2,9% | 1,0% [0,2–5,4] | 11,1% | 9,3% [3,7–21,6] | 4,7% | 2,8% [1,1–7,0] |
| near_high | 4,5% | 0,0% [0,0–5,3] | 30,6% | 29,4% [16,8–46,2] | 8,0% | 10,8% [6,1–18,3] |
| high | 50,0% | 0,0% [0,0–65,8] | 25,0% | 21,5% [14,4–30,9] | 16,7% | 16,8% [10,6–25,6] |

## Comparação com a tese

- Tese (nota **visível**): VPP 16,0%, 4/25, IC 6,4%–34,7%.
- Aqui (nota **oculta**), abaixo do corte, VPP reescalado ao extrato: 2,9% [0,8–12,1] com todas as faixas e 4,8% [1,5–20,9] sem a faixa `low`. Populações e seleção diferem (a tese mediu pares escolhidos pela fila, não uma amostra estratificada), então a comparação é de ordem de grandeza, não de igualdade.

## Custo da medição

- Pares revisados: 717 (árbitro em 75); chamadas à API: 1.542 (33 a mais que o mínimo A+B(+árbitro), por nova tentativa de resposta truncada ou inválida); erros de revisão registrados (par sem veredito na tentativa): 1.
- Tokens de entrada (sem cache): 1.135.128; de saída: 2.685.026; escrita no cache de prompt: 20.736; leitura do cache de prompt: 3.726.102.
- Tempo de relógio (soma das execuções, piloto incluído): 672 s em 3 execuções; erros HTTP (429 etc.): nenhum.

