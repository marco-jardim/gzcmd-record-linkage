# H8 e H11 no extrato real da tese

Status: **resultado experimental**, sem alterar as hipóteses nem os critérios pré-registrados em `docs/engenharia_de_filas.md` (seção 4). Só agregados; nenhum identificador nem dado em nível de par. Gerado por `notebooks/exp_filas_real.py`.

## Resumo

| Hipótese | Status no extrato real | Base |
|---|---|---|
| H8 | **confirmada** | premissa de monotonia: sim; lado A (S5 ≤ S0 no real): sim; lado B (S5 > S0 com não monotonia induzida): sim |
| H11 | **refutada** | 0 de 16 células satisfazem recuperação ≥ 50% e precisão ≥ 30% (metade real; o sintético é outra tarefa) |

## Desenho

- **Dados.** Extrato da tese, pool = pares abaixo do corte: 61.485 pares, 36 verdadeiros (PAR ∈ {1, 2} só como verdade). Régua = `nota final`. Empates da nota quebrados por permutação da semente (mesma ordem para S0 e S5 numa semente).
- **Sementes.** 20. Só o bandit, o desempate dos empates, o erro do humano simulado e o fallback do LLM variam; as decisões do LLM presentes no fixture são **uma realização congelada**.
- **S0.** Ordem global decrescente da nota; B decisões humanas (humano perfeito): verificados = verdadeiros entre os B primeiros.
- **S5.** `ThompsonQueue(input_kind="partial_orders")`, prior Beta(1, 1), sem janela, 4 braços = faixas contíguas por posto com frações acumuladas 1%, 5%, 20% e o resto (0–1%, 1–5%, 5–20%, 20–100%; leitura adotada para "1%, 5%, 20%, resto"), cada faixa do topo para baixo. Cada par escolhido é decidido pelo humano; verificados = verdadeiros entre os B escolhidos. Recompensa do bandit: `human` = par verdadeiro (humano perfeito, testa o bandit em si) ou `llm` = MATCH do fixture (fallback por taxas por faixa para pares ausentes, `disagreement_rate` = 0,10 como em `sim_fila_autocalibravel`).
- **Não monotonia induzida.** `N1`: os verdadeiros do top-5% (janela de postos 0–5%) são permutados para postos sorteados da faixa 1–5%, de modo que a faixa 0–1% fica sem verdadeiros e a 1–5% concentra todos. `N2`: o mesmo com janela 0–20% e destino a faixa 5–20%. A ordem alterada vale para S0 e S5.
- **Definição operacional dos critérios de H8** (fixada antes dos números; a tabela diz só "ambos os lados se verificam"). Lado A: no cenário `real`, para toda recompensa e todo B, o limite inferior do IC 95% (t, entre sementes) de S5 − S0 não é positivo. Lado B: em `N1` ou `N2`, com recompensa `human`, algum B tem limite inferior de S5 − S0 > 0. Confirmada se A e B; refutada se A falha; inconclusiva se a premissa de monotonia falha ou se A vale e B não (a escolha das induções é minha).
- **H11.** Humano simulado revisa os B primeiros da ordem da nota; `u` ~ U[0,1) determinístico por (semente, par), com a mesma `u` para todas as taxas: par é perdido se `u < FN`, não par vira MATCH se `u < FP`. Auditoria: humano perfeito revê só os pares com LLM = MATCH e humano = NONMATCH. Recuperação = FN humanos recuperados / FN humanos; precisão = recuperados / revisões extras; ambas **agregadas entre sementes** (Σ/Σ, IC 95% por bootstrap das sementes, 2000 reamostras), porque por semente o denominador é 0 em várias células. Auditoria aleatória do mesmo tamanho = esperança exata entre os NONMATCH do humano. Célula satisfaz se recuperação ≥ 50% e precisão ≥ 30%; confirmada se todas, refutada se nenhuma, inconclusiva se mista.

## H8

### Premissa: o rendimento cai com o posto?

Média entre sementes (só o desempate dos empates varia). Os 36 verdadeiros no pool.

| faixa do bandit | pares | verdadeiros | taxa |
|---|---|---|---|
| 0–1% | 615 | 17,0 | 2,76% |
| 1–5% | 2459 | 3,0 | 0,12% |
| 5–20% | 9223 | 10,0 | 0,11% |
| 20–100% | 49188 | 6,0 | 0,01% |

| decil de posto | pares | verdadeiros | taxa |
|---|---|---|---|
| 1 | 6148 | 24,0 | 0,39% |
| 2 | 6149 | 6,0 | 0,10% |
| 3 | 6149 | 4,0 | 0,07% |
| 4 | 6148 | 1,9 | 0,03% |
| 5 | 6148 | 0,1 | 0,00% |
| 6 | 6149 | 0,0 | 0,00% |
| 7 | 6149 | 0,0 | 0,00% |
| 8 | 6148 | 0,0 | 0,00% |
| 9 | 6148 | 0,0 | 0,00% |
| 10 | 6149 | 0,0 | 0,00% |

Premissa (taxa por faixa do bandit não crescente): **verificada**. Com 36 positivos, as taxas por decil são ruidosas (poucas unidades).

### Verdadeiros verificados e S5 − S0: recompensa = par verdadeiro (humano perfeito)

**cenário `real` (régua como está)**

| B | S0 | S5 [IC 95%] | S5 − S0 [IC 95%] | uso dos braços (0–1 / 1–5 / 5–20 / 20–100) |
|---|---|---|---|---|
| 50 | 10,00 | 10,00 [10,00; 10,00] | 0,00 [0,00; 0,00] | 69,2% / 10,5% / 10,1% / 10,2% |
| 100 | 12,00 | 10,00 [10,00; 10,00] | -2,00 [-2,00; -2,00] | 66,6% / 11,3% / 10,6% / 11,4% |
| 200 | 13,00 | 12,20 [12,01; 12,39] | -0,80 [-0,99; -0,61] | 67,8% / 11,0% / 10,8% / 10,3% |
| 400 | 13,00 | 13,00 [13,00; 13,00] | 0,00 [0,00; 0,00] | 68,5% / 10,5% / 10,7% / 10,4% |
| 800 | 17,00 | 15,00 [15,00; 15,00] | -2,00 [-2,00; -2,00] | 69,4% / 10,3% / 10,2% / 10,1% |

**cenário `N1` (verdadeiros concentrados em 1–5%)**

| B | S0 | S5 [IC 95%] | S5 − S0 [IC 95%] | uso dos braços (0–1 / 1–5 / 5–20 / 20–100) |
|---|---|---|---|---|
| 50 | 0,00 | 0,15 [-0,02; 0,32] | 0,15 [-0,02; 0,32] | 24,8% / 26,6% / 24,7% / 23,9% |
| 100 | 0,00 | 0,20 [0,01; 0,39] | 0,20 [0,01; 0,39] | 24,4% / 26,5% / 24,2% / 24,9% |
| 200 | 0,00 | 0,40 [0,12; 0,68] | 0,40 [0,12; 0,68] | 24,1% / 27,5% / 23,6% / 24,8% |
| 400 | 0,00 | 0,80 [0,31; 1,29] | 0,80 [0,31; 1,29] | 23,1% / 29,9% / 23,2% / 23,8% |
| 800 | 1,25 | 1,95 [1,00; 2,90] | 0,70 [0,19; 1,21] | 21,6% / 34,6% / 21,9% / 21,9% |

**cenário `N2` (verdadeiros concentrados em 5–20%)**

| B | S0 | S5 [IC 95%] | S5 − S0 [IC 95%] | uso dos braços (0–1 / 1–5 / 5–20 / 20–100) |
|---|---|---|---|---|
| 50 | 0,00 | 0,05 [-0,05; 0,15] | 0,05 [-0,05; 0,15] | 25,0% / 25,2% / 25,7% / 24,1% |
| 100 | 0,00 | 0,15 [-0,08; 0,38] | 0,15 [-0,08; 0,38] | 24,9% / 24,9% / 25,3% / 24,9% |
| 200 | 0,00 | 0,20 [-0,04; 0,44] | 0,20 [-0,04; 0,44] | 24,1% / 24,7% / 26,1% / 25,1% |
| 400 | 0,00 | 0,30 [0,03; 0,57] | 0,30 [0,03; 0,57] | 24,0% / 24,3% / 27,3% / 24,4% |
| 800 | 0,00 | 0,50 [0,14; 0,86] | 0,50 [0,14; 0,86] | 23,8% / 23,7% / 28,9% / 23,6% |

### Verdadeiros verificados e S5 − S0: recompensa = MATCH do LLM (fixture)

**cenário `real` (régua como está)**

| B | S0 | S5 [IC 95%] | S5 − S0 [IC 95%] | uso dos braços (0–1 / 1–5 / 5–20 / 20–100) |
|---|---|---|---|---|
| 50 | 10,00 | 10,00 [10,00; 10,00] | 0,00 [0,00; 0,00] | 58,5% / 16,1% / 11,8% / 13,6% |
| 100 | 12,00 | 10,00 [10,00; 10,00] | -2,00 [-2,00; -2,00] | 55,2% / 17,3% / 13,3% / 14,2% |
| 200 | 13,00 | 12,00 [12,00; 12,00] | -1,00 [-1,00; -1,00] | 49,4% / 22,9% / 13,1% / 14,6% |
| 400 | 13,00 | 13,00 [13,00; 13,00] | 0,00 [0,00; 0,00] | 53,5% / 22,0% / 11,2% / 13,3% |
| 800 | 17,00 | 15,00 [15,00; 15,00] | -2,00 [-2,00; -2,00] | 59,0% / 19,8% / 9,0% / 12,2% |

**cenário `N1` (verdadeiros concentrados em 1–5%)**

| B | S0 | S5 [IC 95%] | S5 − S0 [IC 95%] | uso dos braços (0–1 / 1–5 / 5–20 / 20–100) |
|---|---|---|---|---|
| 50 | 0,00 | 0,15 [-0,02; 0,32] | 0,15 [-0,02; 0,32] | 22,2% / 32,1% / 22,4% / 23,3% |
| 100 | 0,00 | 0,30 [0,08; 0,52] | 0,30 [0,08; 0,52] | 20,6% / 34,7% / 21,5% / 23,4% |
| 200 | 0,00 | 0,60 [0,25; 0,95] | 0,60 [0,25; 0,95] | 19,6% / 36,2% / 20,4% / 23,8% |
| 400 | 0,00 | 1,15 [0,50; 1,80] | 1,15 [0,50; 1,80] | 18,0% / 40,1% / 18,1% / 23,8% |
| 800 | 1,25 | 2,25 [1,29; 3,21] | 1,00 [0,39; 1,61] | 26,0% / 37,7% / 14,1% / 22,3% |

**cenário `N2` (verdadeiros concentrados em 5–20%)**

| B | S0 | S5 [IC 95%] | S5 − S0 [IC 95%] | uso dos braços (0–1 / 1–5 / 5–20 / 20–100) |
|---|---|---|---|---|
| 50 | 0,00 | 0,05 [-0,05; 0,15] | 0,05 [-0,05; 0,15] | 22,4% / 30,9% / 23,3% / 23,4% |
| 100 | 0,00 | 0,15 [-0,08; 0,38] | 0,15 [-0,08; 0,38] | 20,9% / 32,6% / 22,7% / 23,8% |
| 200 | 0,00 | 0,20 [-0,04; 0,44] | 0,20 [-0,04; 0,44] | 20,0% / 32,9% / 22,8% / 24,4% |
| 400 | 0,00 | 0,25 [-0,01; 0,51] | 0,25 [-0,01; 0,51] | 18,7% / 34,7% / 21,6% / 24,9% |
| 800 | 0,00 | 0,40 [0,05; 0,75] | 0,40 [0,05; 0,75] | 29,5% / 29,3% / 17,3% / 23,9% |

### Cobertura do fixture nas escolhas do bandit (recompensa LLM)

| cenário | B=50 | B=100 | B=200 | B=400 | B=800 |
|---|---|---|---|---|---|
| real | 76,2% | 73,9% | 73,7% | 76,6% | 79,9% |
| N1 | 57,1% | 57,5% | 57,9% | 60,2% | 65,2% |
| N2 | 55,3% | 55,8% | 55,5% | 55,6% | 60,4% |

Fração das escolhas servida pelo fixture; o resto veio do fallback.

## H11

Cobertura do fixture nos pares revisados pelo humano (os B primeiros da nota): B=100: 100,0%; B=200: 100,0%; B=400: 100,0%; B=800: 100,0% (média 100,0%); o restante usa o fallback por taxas.

Recuperação e precisão são razões agregadas (IC 95% bootstrap entre sementes). "Aleatória" = auditoria aleatória do mesmo tamanho entre os NONMATCH do humano. Médias por semente: FN humanos, revisões extras, recuperados e extras que eram não pares (falsos positivos do LLM).

| FN | FP | B | FN hum. | extras | recup. | extras não pares | recuperação | recup. aleatória | precisão | prec. aleatória | satisfaz |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5,0% | 0,0% | 100 | 0,95 | 0,30 | 0,30 | 0,00 | 31,6% [16,7%; 43,8%] | 0,7% [0,4%; 1,0%] | 100,0% [100,0%; 100,0%] | 2,2% [2,2%; 2,2%] | não |
| 5,0% | 0,0% | 200 | 1,15 | 3,30 | 0,30 | 3,00 | 26,1% [13,3%; 37,5%] | 1,9% [1,7%; 2,0%] | 9,1% [4,7%; 14,3%] | 0,7% [0,5%; 0,8%] | não |
| 5,0% | 0,0% | 400 | 1,15 | 9,95 | 0,30 | 9,65 | 26,1% [13,3%; 37,5%] | 2,6% [2,5%; 2,7%] | 3,0% [1,5%; 5,0%] | 0,3% [0,2%; 0,4%] | não |
| 5,0% | 0,0% | 800 | 1,40 | 21,55 | 0,55 | 21,00 | 39,3% [22,2%; 53,6%] | 2,8% [2,7%; 2,9%] | 2,6% [1,2%; 4,1%] | 0,2% [0,1%; 0,2%] | não |
| 5,0% | 1,0% | 100 | 0,95 | 0,30 | 0,30 | 0,00 | 31,6% [16,7%; 43,8%] | 0,7% [0,4%; 1,0%] | 100,0% [100,0%; 100,0%] | 2,3% [2,2%; 2,3%] | não |
| 5,0% | 1,0% | 200 | 1,15 | 3,20 | 0,30 | 2,90 | 26,1% [13,3%; 37,5%] | 1,9% [1,8%; 2,0%] | 9,4% [4,7%; 14,7%] | 0,7% [0,5%; 0,8%] | não |
| 5,0% | 1,0% | 400 | 1,15 | 9,85 | 0,30 | 9,55 | 26,1% [13,3%; 37,5%] | 2,6% [2,6%; 2,7%] | 3,0% [1,5%; 5,1%] | 0,3% [0,2%; 0,4%] | não |
| 5,0% | 1,0% | 800 | 1,40 | 21,35 | 0,55 | 20,80 | 39,3% [22,2%; 53,6%] | 2,8% [2,7%; 2,9%] | 2,6% [1,2%; 4,1%] | 0,2% [0,1%; 0,2%] | não |
| 10,0% | 0,0% | 100 | 1,85 | 0,80 | 0,80 | 0,00 | 43,2% [32,1%; 53,7%] | 1,5% [1,1%; 1,8%] | 100,0% [100,0%; 100,0%] | 3,4% [2,7%; 3,9%] | não |
| 10,0% | 0,0% | 200 | 2,05 | 3,80 | 0,80 | 3,00 | 39,0% [27,6%; 50,0%] | 2,2% [2,0%; 2,4%] | 21,1% [13,0%; 27,7%] | 1,2% [0,9%; 1,5%] | não |
| 10,0% | 0,0% | 400 | 2,05 | 10,45 | 0,80 | 9,65 | 39,0% [27,6%; 50,0%] | 2,8% [2,7%; 2,9%] | 7,7% [4,4%; 10,7%] | 0,5% [0,4%; 0,7%] | não |
| 10,0% | 0,0% | 800 | 2,40 | 22,15 | 1,15 | 21,00 | 47,9% [36,1%; 59,2%] | 2,9% [2,8%; 2,9%] | 5,2% [3,4%; 6,9%] | 0,3% [0,2%; 0,4%] | não |
| 10,0% | 1,0% | 100 | 1,85 | 0,80 | 0,80 | 0,00 | 43,2% [32,1%; 53,7%] | 1,5% [1,1%; 1,8%] | 100,0% [100,0%; 100,0%] | 3,4% [2,7%; 3,9%] | não |
| 10,0% | 1,0% | 200 | 2,05 | 3,70 | 0,80 | 2,90 | 39,0% [27,6%; 50,0%] | 2,2% [2,0%; 2,4%] | 21,6% [13,2%; 28,6%] | 1,2% [0,9%; 1,5%] | não |
| 10,0% | 1,0% | 400 | 2,05 | 10,35 | 0,80 | 9,55 | 39,0% [27,6%; 50,0%] | 2,8% [2,7%; 2,9%] | 7,7% [4,5%; 10,8%] | 0,6% [0,4%; 0,7%] | não |
| 10,0% | 1,0% | 800 | 2,40 | 21,95 | 1,15 | 20,80 | 47,9% [36,1%; 59,2%] | 2,9% [2,8%; 2,9%] | 5,2% [3,5%; 7,0%] | 0,3% [0,2%; 0,4%] | não |

## Leitura

**H8: confirmada.** Diferenças S5 − S0 (verdadeiros verificados): `real`/`human`: S5 − S0 de -2,00 (B=100) a 0,00 (B=50); maior limite inferior do IC 0,00; `real`/`llm`: S5 − S0 de -2,00 (B=100) a 0,00 (B=50); maior limite inferior do IC 0,00; `N1`/`human`: S5 − S0 de 0,15 (B=50) a 0,80 (B=400); maior limite inferior do IC 0,31; `N1`/`llm`: S5 − S0 de 0,15 (B=50) a 1,15 (B=400); maior limite inferior do IC 0,50; `N2`/`human`: S5 − S0 de 0,05 (B=50) a 0,50 (B=800); maior limite inferior do IC 0,14; `N2`/`llm`: S5 − S0 de 0,05 (B=50) a 0,40 (B=800); maior limite inferior do IC 0,05. O ganho do bandit nas induções é pequeno em valor absoluto e é medido contra uma S0 que a própria indução prejudicou (os verdadeiros saíram do topo): com B=800, o melhor S5 induzido tem 1,95 verdadeiros contra 17,00 da S0 na régua real. No cenário real o bandit gasta cerca de 30% do orçamento nos braços de baixo rendimento e nunca supera a ordem da nota.

**H11: refutada** (0 de 16 células). Melhor recuperação: 47,9% (FN 10,0%, FP 0,0%, B=800), com precisão 5,2% contra 0,3% da auditoria aleatória. As revisões extras incluem todo falso positivo do LLM (não par com MATCH que o humano, corretamente, recusou), o que derruba a precisão quando o LLM tem FP > 0 e há poucos FN humanos para recuperar.

## Ressalvas

- **36 positivos.** Qualquer taxa por faixa ou por decil repousa em poucas unidades; os IC entre sementes cobrem só o desempate, o bandit e o erro do humano simulado, **não** a incerteza de amostragem dos 36 pares.
- **Humano simulado.** O erro é sorteado por (semente, par), independente do par; humanos reais erram de forma correlacionada com a dificuldade (e com o próprio LLM).
- **Uma realização do LLM.** O fixture congela uma decisão por par; as sementes não a reamostram. A cobertura do fixture e a parcela vinda do fallback (taxas por faixa, `disagreement_rate` = 0,10, suposição) estão acima.
- **Escopo de B.** B ≤ 800 é ≈ 1,3% do pool, bem menos que os braços 2–4; o bandit gasta parte do orçamento explorando braços de baixo rendimento (uso dos braços nas tabelas) e N2 mede o custo de achar uma faixa intermediária distante.
- **Induções.** N1 e N2 são construções minhas (permutação dos verdadeiros na janela); outras induções poderiam dar outro resultado no lado B.
- **Sintético.** H11 também exige o resultado no sintético (`notebooks/exp_engenharia_filas.py`); aqui está só a metade real.

## Execução

- Pool: 61485 pares, 36 verdadeiros; fixture: 4604 pares (sha256 `51521231ba9779b52ea68b3896b35cf728b851e016280d1bc9971b4609b931d5`).
- Tempo total: 9 s; pico de memória (working set): 261 MB.
