# Engenharia de filas de revisão de potenciais pares

Status: **documento de trabalho**. Princípios e hipóteses **pré-registrados em
2026-11-15, antes dos experimentos**. A seção "Resultados" é preenchida depois,
sem alterar as hipóteses nem os critérios de decisão. Nada aqui foi avaliado na
tese.

## 1. De onde vem este documento

A fila auto-calibrável (`self_calibrating_queue`, 0.7.0, experimental) foi
avaliada em três versões de simulação, numa execução de ponta a ponta com o
Claude Haiku real e num teste de desempate na zona cinza
(`docs/sim_fila_autocalibravel.md`, `docs/llm_real_fila_ponta_a_ponta.md`,
`docs/llm_real_desempate_zona_cinza.md`). Em todas as comparações a orçamento
humano igual, ela ficou atrás da ordenação pela nota. Exemplo, execução real com
~127 decisões humanas, pares abaixo do corte encontrados (de 36): fila 4; nota
com LLM e humano confirmando os MATCH 10; humano sozinho na ordem da nota 9.

Este documento separa o que é específico do extrato da tese do que pode valer
para filas de revisão em geral, e transforma isso em hipóteses testáveis.

## 2. Diagnóstico: por que a fila perdeu neste extrato

- **D1. Régua forte.** A nota acha 12 dos 36 pares abaixo do corte nas primeiras
  100 revisões; ela já resume os campos que o modelo aprenderia.
- **D2. Poucos positivos para aprender.** Com prevalência de 0,06%, centenas de
  revisões rendem ~10 positivos, contra 15 a 53 coeficientes a ajustar.
- **D3. Positivos difíceis diferentes dos fáceis.** O modelo aprende com o topo
  (pares óbvios) e procura "mais do mesmo"; os do fundo têm outros padrões.
- **D4. Objetivos misturados.** Achar pares agora (explorar o melhor escore) e
  calibrar (amostra com probabilidade conhecida) foram postos no mesmo controle
  (bandit + exploração + pesos com teto); nenhum dos dois ficou bem servido.
- **D5. Rótulo ruidoso.** O veredito do LLM como rótulo tem FN de 44,4% abaixo
  do corte (Haiku real, nota oculta).
- **D6. A régua ficou fora do modelo.** Ao retirar a nota do recalibrador, a
  seleção pela nota deixou de ser ignorável e o modelo partiu de um prior pior
  (Fellegi-Sunter com prevalência implausível).

## 3. Princípios propostos

1. **Comece pela melhor régua sem rótulos** e revise do topo para baixo
   (princípio de ordenação por probabilidade).
2. **Aprenda só a correção.** Modelo com a régua como *offset* (ou covariável
   com prior centrado no valor sem rótulos) e encolhimento forte dos demais
   coeficientes para zero: sem dados, ordena como a régua.
3. **Só rótulo verificado treina.** Decisão humana é rótulo; veredito do LLM é
   **atributo**.
4. **Separe estimação de exploração.** Uma amostra probabilística pequena e
   explícita (com π conhecido) estima positivos restantes e cobertura; ela não
   controla a ordem.
5. **Pare por custo.** Pare quando o número esperado de positivos por próxima
   decisão humana cair abaixo do custo relativo da decisão, ou ao atingir uma
   cobertura-alvo com intervalo de confiança.
6. **Use o LLM para reduzir o tempo por decisão** e para auditar decisões
   humanas, não para reduzir o número de decisões nem para decidir.
7. **Avalie a orçamento humano igual**, com comparações pareadas, várias
   sementes e fixtures reaplicáveis de decisões reais.

## 4. Hipóteses pré-registradas

Notação: S0 = régua (ordem pela nota/escore base); S1 = fila auto-calibrável
atual (configuração padrão, modo humano); S2 = fila com régua como offset e
rótulos verificados (princípios 2–4), sem LLM; S3 = S2 com o veredito do LLM como
atributo; S4 = S2 com o veredito do LLM como rótulo; S5 = bandit de Thompson
sobre faixas contíguas da régua; S6 = S2 com parada por τ fixo (regra atual).
"Ganho" = pares verdadeiros verificados a mais que S0 com o mesmo número B de
decisões humanas. "Ruído" = IC 95% pareado (sementes) incluindo 0.

| Id | Hipótese | Previsão falsificável | Critério de decisão |
|---|---|---|---|
| H1 | Com régua bem especificada, aprender não supera a régua | Cenário base: ganho de S2 é ruído; S1 < S0 | Confirmada se o IC de S2−S0 inclui 0 em todo B e S1−S0 < 0 em algum B |
| H2 | O ganho de aprender cresce com a prevalência | Ganho de S2 (com sinal fora da régua) cresce de 0,1% → 1% → 5% → 20% | Confirmada se o ganho relativo médio é monotônico não decrescente e > 0 a partir de 5% |
| H3 | Régua fraca ou importada deixa espaço para aprender | Com pesos errados/ruído na régua, S2 > S0, crescente com o grau de erro | Confirmada se o IC do ganho exclui 0 no nível mais degradado e o ganho cresce com o grau |
| H4 | Só há ganho se existe sinal fora da régua | Com atributo informativo fora da régua S2 > S0; sem ele, ruído | Confirmada se ambos os lados se verificam no cenário base de prevalência 5% |
| H5 | Positivos difíceis com outro perfil limitam o aprendizado | Com subgrupo difícil no fundo, o ganho de S2 encolhe e esse subgrupo é sub-recuperado; a amostra probabilística é o que o detecta | Confirmada se a recuperação do subgrupo por S2 ≤ a de S0 e a estimativa de restantes da amostra cobre esse subgrupo |
| H6 | O offset garante não inferioridade | S2 nunca fica pior que S0 além de δ = max(1 par, 5% de S0) | Confirmada se, em todas as células e B, o limite inferior do IC de S2−S0 ≥ −δ |
| H7 | LLM como atributo > LLM como rótulo | S3 ≥ S4 e S3 ≥ S0; S4 pode ficar < S0 com FN alto | Confirmada se S3−S4 ≥ 0 com IC excluindo 0 em algum B, e S3−S0 ≥ −δ em todos; checada também no extrato real com o fixture |
| H8 | Bandit sobre faixas contíguas não ajuda com rendimento monotônico | S5 ≤ S0 quando o rendimento cai com o posto; S5 > S0 só com não monotonia induzida | Confirmada se ambos os lados se verificam |
| H9 | Estimação separada funciona com orçamento pequeno | Amostra de 5% do orçamento humano dá IC 95% de positivos restantes com cobertura ≥ 90%; os pesos com teto da fila atual não | Confirmada se a cobertura empírica (sementes × células) ≥ 90% para S2 e < 90% para o estimador da S1 |
| H10 | Parada por custo é melhor que τ fixo | Para um custo relativo c, o arrependimento (valor do oráculo − valor obtido) de S2 é menor que o de S6 | Confirmada se o arrependimento médio de S2 < S6 em ≥ 3 de 4 níveis de prevalência |
| H11 | O LLM serve para auditar decisões humanas | Com humano errando (FN 5–10%), revisar de novo só os pares em que LLM = MATCH e humano = NONMATCH recupera a maioria dos FN humanos com poucas revisões extras | Confirmada se recupera ≥ 50% dos FN humanos com precisão ≥ 30% (extras que eram de fato pares), no sintético e no extrato real com o fixture |
| H12 | O LLM reduz o tempo por decisão humana | — | **Não testável offline**: exige estudo com revisores reais |

## 5. Protocolo experimental

- **Gerador sintético** com botões independentes: prevalência entre candidatos
  (0,1%, 1%, 5%, 20%); qualidade da régua (correta; pesos errados; ruído);
  atributo informativo fora da régua (sim/não); subgrupo de positivos difíceis
  com perfil distinto (sim/não); não monotonia induzida na régua (para H8);
  LLM simulado com as taxas medidas do Haiku (FP 0,4%, FN por faixa), com opção
  de FN concentrado nos difíceis; humano perfeito ou com FN/FP (para H11).
  Os dados têm as colunas que a sessão exige, para que S1 rode sem adaptação.
- **Desenho**: um cenário base (prevalência 1%, régua correta, sem sinal fora,
  sem subgrupo difícil, rótulos verificados), variação de um fator por vez, e a
  interação prevalência × qualidade da régua × sinal fora da régua.
- **Orçamentos humanos** B ∈ {50, 100, 200, 400, 800}; 10 sementes por célula;
  comparações pareadas (mesma realização do LLM e do humano por par).
- **Dados reais**: no extrato da tese abaixo do corte, com o fixture do Haiku
  real, checar H1, H6, H7, H8 e H11 (36 positivos: evidência fraca, declarada).
- **Métricas**: verdadeiros verificados por B; ganho e IC pareado; recuperação
  do subgrupo difícil; cobertura dos IC de positivos restantes; arrependimento
  da parada; para H11, FN humanos recuperados e precisão da auditoria.

## 6. Resultados

Executado em 2026-11-15 com `notebooks/exp_engenharia_filas.py` (testes em
`tests/test_exp_engenharia_filas.py`); agregados em
`docs/engenharia_de_filas_resultados.csv` (coluna `table`: `yield`, `paired`,
`coverage`, `s1_estimate`, `regret`, `audit`, `hypothesis`) e, para a
sensibilidade a σ, em `docs/engenharia_de_filas_resultados_sensibilidade.csv`.
As seções 1 a 5 não foram alteradas. O status vem da regra de decisão da
seção 4, aplicada **à letra**; onde a letra e o que ela pretendia medir
divergem, isso está dito em cada linha.

Reprodução:

```
python notebooks/exp_engenharia_filas.py --seeds 10 --workers 14 \
    --real-input ../phd-research/data/COMPARADORSEMIDENT.csv \
    --out-csv docs/engenharia_de_filas_resultados.csv
python notebooks/exp_engenharia_filas.py --seeds 10 --workers 14 \
    --subset sensibilidade --no-s1 --sigma 0.25   # e 1, 2
```

### 6.1 Execução

- Grade: 39 células × 10 sementes = 390 execuções sintéticas, N = 20.000
  candidatos por célula (**sem redução**), mais 10 sementes no extrato real
  (61.485 pares abaixo do corte, 36 verdadeiros) e 200 realizações do erro do
  humano para H11 no real. Relógio da grade: **307 s** com 10 processos (219 s com 14 na primeira execução, antes das correções da seção 6.7)
  (a S1 sozinha leva ~6 s por execução sintética). Sensibilidade a σ: 3 grades de
  5 células, ~10 s cada.
- Células: base (prevalência 1%, régua correta, sem sinal fora dela, sem
  subgrupo difícil); um fator por vez (prevalência 0,1/5/20%; pesos errados
  graus 1-3; ruído graus 1-3; sinal fora da régua; subgrupo difícil; FN do LLM
  concentrado nos difíceis; não monotonia; humano com FN 5%/10% e FP 1%); e a
  interação prevalência × régua {correta, pesos errados grau 2, ruído grau 2} ×
  sinal {não, sim}.
- Gerador: campos latentes gaussianos por campo (nome, mãe, nascimento,
  endereço, município) → compostos em [0, 1] com os níveis do
  `fellegi_sunter`; a régua é a soma ponderada da evidência **latente**
  (mais fina que os compostos que o modelo vê), reescalada para 0-10. "Sinal
  fora da régua" = o município é informativo e a régua não o usa. O LLM
  simulado tem FP 0,4% e FN/UNSURE por faixa de nota do `*.rates.csv` e revê os
  2.000 primeiros da régua (sem custo humano); o humano e o LLM são sorteados
  uma vez por par e reaplicados a todas as estratégias (pareamento).
  Calibração do gerador, feita só olhando a S0 antes dos experimentos: no
  cenário base a régua recupera ~71% dos 200 positivos em B = 800 (há espaço
  para aprender).
- Definições operacionais fixadas no código **antes** de ver qualquer
  resultado: IC pareado = t de Student a 95% sobre as 10 sementes; "ganho" =
  verdadeiros verificados (decisão humana MATCH e par verdadeiro); S2 gasta 5% de
  B (arredondado para cima: 3, 5, 10, 20, 40) numa amostra de estimação **dentro
  de B**, tirada ao fim da exploração; **S2x** é um diagnóstico (S2 sem essa
  amostra, ou seja, todas as B decisões na exploração) e não entra nos critérios;
  σ = 0,5 para os coeficientes de campo, 2 para as categorias do LLM em S3, prior
  N(1; 0,5²) para a inclinação da régua e N(0; 2²) para o intercepto; a S1 é a
  sessão padrão (para quando suas regras mandam: ~180 decisões no sintético, 85
  no real) e seu valor fica constante depois de parar; H2 usa o ganho relativo =
  média sobre B de (S2−S0)/S0; H3 é aplicado à série de pesos errados (a de ruído
  é reportada); H7 é avaliado no cenário base e nas células com FN concentrado;
  H8 lado "não ajuda" = nenhum IC de S5−S0 acima de 0 nas células monotônicas;
  H10 usa c = 0,05 e S6 percorre o mesmo caminho de S2 (só muda a regra de
  parada; sem a amostra de estimação em ambos).

### 6.2 Tabela-resumo

| Id | Sintético | Extrato real (36 positivos) |
|---|---|---|
| H1 | refutada (pela letra) | refutada (S2 > S0) |
| H2 | refutada | — |
| H3 | confirmada | — |
| H4 | refutada (pela letra) | — |
| H5 | refutada | — |
| H6 | refutada | confirmada |
| H7 | refutada (pela letra) | confirmada (evidência fraca; após corrigir o vazamento) |
| H8 | confirmada | confirmada (ambos os lados, agente paralelo) |
| H9 | confirmada (IC inútil) | — |
| H10 | inconclusiva — critério mal especificado | — |
| H11 | confirmada | refutada |
| H12 | não testável | não testável |

### 6.3 Rendimento (verdadeiros verificados, média de 10 sementes)

Cenário base sintético (200 positivos em 20.000):

| estratégia | B=50 | B=100 | B=200 | B=400 | B=800 |
|---|---|---|---|---|---|
| S0 régua | 38,8 | 61,6 | 90,3 | 118,6 | 141,9 |
| S1 sessão atual | 49,9 | 65,6 | 79,2 | 79,2 | 79,2 |
| S2 | 35,9 | 60,0 | 87,8 | 116,9 | 140,2 |
| S2x (diagnóstico) | 37,7 | 61,4 | 90,0 | 118,5 | 141,5 |
| S3 LLM atributo | 43,6 | 88,9 | 146,2 | 160,0 | 165,9 |
| S4 LLM rótulo | 37,1 | 60,3 | 89,4 | 117,8 | 140,0 |
| S5 bandit | 36,5 | 58,0 | 85,8 | 114,2 | 138,3 |

Extrato real abaixo do corte (36 positivos; humano perfeito; régua de S2/S3/S4 = stub, ver 6.7):

| estratégia | B=50 | B=100 | B=200 | B=400 | B=800 |
|---|---|---|---|---|---|
| S0 régua | 10 | 12 | 13 | 13 | 17 |
| S1 sessão atual (~106 decisões) | 6,2 | 7,2 | 7,2 | 7,2 | 7,2 |
| S2 (= S2x: a amostra não achou positivos) | 14 | 15 | 17 | 19 | 33 |
| S3 (veredito só na janela dos 3.000 primeiros) | 14 | 15 | 19 | 29 | 32 |
| S4 (idem) | 12 | 13 | 16 | 19 | 32 |
| S5 | 10 | 10 | 12 | 13 | 13 |

### 6.4 Hipótese por hipótese

**H1 — refutada pela letra (sintético) e refutada (real).** Cenário base:
S2−S0 = −2,9 [−5,4; −0,4], −1,6 [−2,7; −0,5], −2,5 [−4,5; −0,5], −1,7
[−2,9; −0,5], −1,7 [−3,1; −0,3] em B = 50…800: o IC exclui 0, então o critério
"ganho é ruído" falha. A causa é o custo da amostra de estimação (5% de B
sem exploração): sem ela (S2x−S0) a diferença fica entre −1,1 e 0,0. S1−S0 =
+11,1; +4,0; −11,1; −39,4; −62,7: a S1 *ganha* de S0 em B ≤ 100 (o LLM, de graça,
põe os MATCH na frente do portão) e perde de B = 200 em diante, porque a sessão
pára em ~190 decisões. Em suma: aprender não superou a régua (previsão
qualitativa mantida), mas a letra do critério não vale porque ignora o custo da
estimação. No real a régua não está "bem especificada": S2−S0 = +4, +3, +4, +6,
+16 (S0 é determinística e a S2 também, o IC degenera em ponto) e S1−S0 < 0 em
todo B (−3,8 a −9,8; a S1 pára em ~106 decisões).

**H2 — refutada.** Ganho relativo médio de S2 (com sinal fora da régua) por
prevalência 0,1% / 1% / 5% / 20%: −0,042; −0,021; −0,034; −0,023: nunca positivo
e não monotônico. Mesmo sem o custo da amostra (S2x): −0,025; +0,004; +0,003;
−0,000 (ruído em torno de 0). Neste gerador, com N fixo e B ≤ 800, o teto da
régua em prevalências altas (B é pequeno frente a 1.000-4.000 positivos) deixa
pouco para aprender; o ganho absoluto de S2x no 5% com sinal só aparece em
B = 800 (+8,3).

**H3 — confirmada.** Pesos errados (grau 0 a 3; prevalência 1%, sem sinal),
S2−S0 médio sobre B: −2,08; −0,58; +3,24; +7,30, com IC excluindo 0 em algum B
nos graus 2 e 3; cresce com o grau. Ruído: −2,08; +3,08; +19,30; +29,48 (o ganho
vem de S2 reordenar pelos campos quando a régua perde informação). Em
B = 800 e grau 3: +11,1 (pesos) e +57,2 (ruído).

**H4 — refutada pela letra.** Prevalência 5%: com sinal, S2−S0 = −2,6; −4,1;
−5,4; −12,1; −6,5 em B = 50…800 (nenhum IC acima de 0); sem sinal: −2,6; −3,6;
−5,8; −14,4; −16,3 (IC exclui 0 para baixo). A letra exige "ganho com sinal"
e "ruído sem sinal"; a S2 perde nos dois por causa do custo de 5% de B numa
amostra aleatória quando a prevalência é alta (S2−S2x = −12,6 com sinal e −12,7 sem sinal em B = 400).
Sem esse custo o padrão aparece: S2x−S0 com sinal = 0,0; −0,1; +0,4; +0,5;
+8,3 contra sem sinal 0,0; 0,0; +0,4; −1,7; −2,3. A diferença sinal × sem sinal
existe mas só em B grande e só depois de descontado o custo da estimação.

**H5 — refutada.** Células com subgrupo difícil (30% dos positivos, nome/mãe
fracos): S2−S0 em difíceis recuperados, média sobre B: −0,12 (1%, sem sinal),
+0,32 (1%, com sinal), +0,78 (5%, com sinal); com sinal a S2 recupera *mais*
difíceis que a régua (+2,9 em B = 800 no 5%), contrário à previsão. A cobertura
do IC de difíceis restantes é 1,000, mas com IC enormes (ver H9). Não houve
"encolhimento do ganho" mensurável pelo subgrupo.

**H6 — refutada.** 49 de 200 (célula, B) violam IC inferior ≥ −δ
(δ = max(1; 5% de S0)); 26 em B = 50, 10 em B = 100, 9 em 200, 2 em 400, 2 em
800; pior folga −3,80 em p1%, humano FN 5%/FP 1%, B = 50. Sem a amostra de
estimação (S2x) ainda há 17 de 200 violações (mínimo de média −2,4), e a S1
viola em 81 de 200. Concentram-se em B pequeno (a amostra de 3 a 5 itens custa
até 2-3 pares), em humano com erro e em não monotonia. No real: 0 de 5
violações, pior folga +4,0 (confirmada).

**H7 — refutada pela letra (sintético); confirmada, com evidência fraca (real, após corrigir o vazamento).** S3−S4 ≥ 0 sempre
(média mínima +0,6) e com IC > 0 em 14 de 15 (célula, B) do cenário base e das
células de FN concentrado: o lado "atributo ≥ rótulo" se sustenta; S4−S0 é ~
−1 a −2 (e até +7,9 no 5% com sinal e FN concentrado, B = 800). Mas S3−S0 viola
−δ em 9 de 200 (célula, B), todos em B = 50, prevalência 5% (−1,9 a −2,9 frente
a δ ≈ 2,2-2,5): é o custo da amostra, de novo. S3 é, de longe, a melhor
estratégia no sintético (cenário base: 146,2 vs 90,3 em B = 200), porque o LLM
simulado (FP 0,4%) é um filtro quase sem custo humano. No real, depois de restringir o veredito do fixture à janela dos 3.000 primeiros pela nota (seção 6.7): S3−S0 = +4; +3; +6; +16; +15 e S3−S4 = +2; +2; +3; +10; 0 (S3 e S4 batem em 32 em B = 800): S3−S4 ≥ 0 em todo B e S3−S0 sem violação, então a letra é satisfeita; mas S0, S3 e S4 são determinísticas no real (IC degenerados em ponto, uma só realização do LLM, 36 positivos). Os números anteriores à correção (S3 = 14, 14, 15, 18, 20; S4 = 10, 12, 15, 19, 22; veredito "refutada") usavam a cobertura do fixture, que depende do rótulo, e foram descartados.

**H8 — confirmada (sintético) e lado testável confirmado (real).** Células
monotônicas: máx S5−S0 = −0,6, nenhum IC acima de 0 em 20. Com não monotonia
(bolso de positivos no posto 5-7%): S5−S0 = +25,0 (B = 400) e +72,4 (B = 800)
no 1%; +27,9 e +130,3 no 5%, IC acima de 0 em 4 de 10. No real S5−S0 ≤ 0 em todo B. O agente paralelo (`docs/engenharia_de_filas_real_h8_h11.md`, 20 sementes, humano perfeito ou LLM como recompensa) testa os dois lados no real e **confirma H8**: premissa de monotonia verificada (taxa por faixa 2,76%; 0,12%; 0,11%; 0,01%); cenário `real`: S5−S0 de −2,00 a 0,00, nunca com limite inferior > 0; com não monotonia induzida (os verdadeiros do topo permutados para 1–5% ou 5–20%) S5−S0 = +0,15 a +0,80 (N1) e +0,05 a +0,50 (N2), IC acima de 0 em B ≥ 100 (N1) e B ≥ 400 (N2); os ganhos são pequenos em valor absoluto e medidos contra uma S0 que a indução prejudicou (1,95 vs 17,00 verdadeiros em B = 800).

**H9 — confirmada pela letra, com ressalva grave.** Cobertura empírica do IC
95% de positivos restantes: S2 0,990 (n = 1.950 células × sementes × B; 0,963
a 1,000 por prevalência); estimador da S1 (amostra de exploração da última
rodada, pesos com teto 10): 0,128 (n = 390 execuções); sem o teto, 0,850. A cobertura da S2 é alta porque o IC é
muito largo: largura média 5.400 a 14.700 (por B) para um valor verdadeiro de
~650-930 restantes (média entre células); estima-se a ordem de grandeza e não
mais. A comparação com a S1 também não é de mesma natureza (a S1 estima os
positivos não revistos no início da última rodada).

**H10 — inconclusiva — critério mal especificado.** Arrependimento médio com
c = 0,05, S2 vs S6, por prevalência 0,1% / 1% / 5% / 20%: 14,9 vs 18,7; 89,8 vs
88,6; 236,9 vs 236,9; 8,6 vs 8,6. S2 < S6 em 1 de 4 níveis. Em 5% e 20% nenhuma
das duas regras pára (o rendimento esperado fica acima de c até B = 800), então
os dois níveis empatam por construção para qualquer c da grade e "≥ 3 de 4"
é inalcançável; pela letra daria "refutada". Onde as regras divergem, S2 pára
bem antes que S6 (k médio 98 vs 215 no 0,1%; 476 vs 793 no 1%) e ganha com
c ≥ 0,05 no 0,1% (14,9 vs 18,7; c = 0,10: 16,8 vs 28,5) e no 1% com c = 0,10
(104,6 vs 118,2), mas perde de leve com c = 0,05 no 1% e empata com c = 0,02.

**H11 — confirmada (sintético) e refutada (real).** B = 400, humano FP 1%.
Sintético: FN 5% / 1%: 69 FN humanos, 54 recuperados (0,78), 63 re-revisões,
precisão 0,86; FN 10% / 1%: 119, 87 (0,73), 96, 0,91; FN 5% / 5%: 185, 109
(0,59), 112, 0,97; FN 10% / 5%: 356, 215 (0,60), 218, 0,99 (critério: ≥ 50% e
≥ 30%). No real (LLM real do fixture; 200 realizações), FN humano 5%: 134 FN,
61 recuperados (0,46), 2.041 re-revisões, precisão 0,03; FN 10%: 257, 113 (0,44),
2.093, 0,05. O LLM real marca MATCH em ~10 não pares por realização entre os
400 primeiros e deixa passar 54-56% dos pares abaixo do corte: a auditoria
custa 18 a 33 re-revisões por par recuperado e nem recupera metade. O agente paralelo chega à mesma conclusão no real (`docs/engenharia_de_filas_real_h8_h11.md`, 20 sementes, 16 células de FN × FP × B): **0 de 16** satisfazem recuperação ≥ 50% e precisão ≥ 30%; recuperação de 26,1% a 47,9%; precisão de 100% em B = 100 (0,3 a 0,8 par recuperado por semente), 9,1% a 21,6% em B = 200, 3,0% a 7,7% em B = 400 e 2,6% a 5,2% em B = 800, contra 0,2% a 3,4% de uma auditoria aleatória do mesmo tamanho (a auditoria é 15 a 30 vezes melhor que o acaso, mas continua longe de 30% de precisão fora de B = 100).

**H12 — não testável** offline.

### 6.5 Sensibilidade a σ (exploratória, fora dos critérios)

S2−S0 / S2x−S0 médios em B = 50, 100, 200, 400, 800, σ ∈ {0,25; 0,5; 1; 2}
(`docs/engenharia_de_filas_resultados_sensibilidade.csv`):

- 1%, sem sinal: S2x−S0 entre −3,5 e +0,3, piora com σ maior (−3,5 a −0,2 em
  σ = 2): sem sinal a régua não tem o que aprender.
- 1%, com sinal: S2x−S0 em B = 400 = +1,2 (0,25), +2,8 (0,5), +3,1 (1), +4,2 (2);
  a S2 (com amostra) só passa de 0 em B ≥ 400 para σ = 0,5 e já em
  B = 200 para σ ≥ 1.
- 5%, com sinal, B = 800: S2x−S0 = +4,0; +8,3; +14,4; +17,1 e S2−S0 = −12,2;
  −6,5; −2,5; +0,5.
- 1%, pesos errados grau 3, B = 800: S2−S0 = +7,1; +11,1; +15,1; +16,7.

Com σ maior o ganho de aprender cresce nas células em que existe o que aprender
e a perda cresce onde não existe; o custo da amostra de 5% não depende de σ.

### 6.6 Ressalvas

- **Sintético ≠ real.** O gerador foi escolhido por mim, a régua usa a
  evidência latente (mais fina que os compostos do modelo), o LLM simulado erra
  só por faixa de nota e nunca por par (FN independentes entre pares), e a
  conclusão "S3 muito melhor que tudo" depende de o LLM revisar de graça os 2.000
  primeiros da régua. As prevalências altas são medidas com N fixo e B ≤ 800: o
  teto de B, não só a prevalência, limita o ganho.
- **36 positivos no real.** S0 e S2 são determinísticas no real; os "IC" de
  S2−S0 degeneram em ponto (todas as sementes iguais) e as diferenças de 3 a
  16 pares são evidência fraca. O fixture cobre ~4.400 dos 61.485 pares, mas só os 3.000 primeiros pela nota (2.992 cobertos) entram em S3/S4 e na S1; fora da janela o veredito é "sem veredito" (S3/S4) ou simulado por faixa (S1). A cobertura fora da janela depende do rótulo (cobre os 16 positivos de fora e 1.385 dos ~58.000 não pares) e por isso é descartada.
- **Humano simulado.** Decide exatamente como mandam as taxas e a verdade; o
  humano real erra por par e por fadiga. O custo por decisão e o tempo
  (H12) não são medidos.
- **O custo da amostra de estimação** domina vários critérios (H1, H4, H6, H7):
  a S2x mostra o desempenho sem ele, mas a decisão foi tomada pela S2 como
  definida (a amostra faz parte do orçamento). A amostra também não tem
  utilidade prática com IC de milhares de pares (H9): precisaria de estimador
  assistido por modelo ou de amostras maiores.
- **Comparações múltiplas.** H6 e H7 exigem o critério em *todas* as células e B
  (200 comparações com 10 sementes): algumas violações são esperadas por acaso
  mesmo com um custo real pequeno; a lista por B está no CSV (`paired`).
- A S1 tem como estimador de restantes o da última rodada; sua definição
  (explore, teto 10) foi feita aqui, e não existe na sessão.

### 6.7 Revisão após a auditoria: correções e confirmações exploratórias

**Correções.**
(A) *Vazamento no real (corrigido).* O fixture cobre todos os 247 pares verdadeiros
da amostra de calibração e das execuções guiadas pela verdade: fora da janela dos
3.000 primeiros pela nota ele cobria os 16 positivos de fora e 1.385 não pares.
`llm_seen` passou a valer só na janela (`restrict_to_window`; cobertura em função da
posição na régua, não do rótulo; 2.992 dos 3.000 cobertos, 8 empatados na fronteira
sem veredito), vale para S3, S4 e para a S1 (`WindowedFixtureReviewer`). O
sintético já usava cobertura por posição. Efeito: as linhas S3/S4 do real e o
veredito real de H7 mudaram (6.3 e H7).
(B) *Régua do real = stub.* Abaixo do corte há 9 pares com nota ≥ 9 (< 30 âncoras),
o `fit_anchor_platt` falha e S2/S3/S4 usam `logit(nota / 10)`. O aviso do
`calibration` agora **não é silenciado** e a coluna `ruler_source` (CSV, tabela
`yield`) registra a régua usada: `anchor_platt` em 37 células sintéticas,
`anchor_platt+stub` nas duas de prevalência 0,1% (algumas sementes sem âncoras) e
`stub` no extrato real.
(C) *Verdade fora das decisões (corrigido).* O estimador HT (`ht_remaining`) e a
regra de parada (`stop_tau`, `regret_rows`) passaram a ver só a decisão humana. Só
as células com erro humano mudam; **nenhum status sintético mudou** (H9: largura
média 8.813,7 → 8.828,6, cobertura S2 0,990 e S1 0,128 iguais). Testes novos
falhariam antes (A, B, C).

**Confirmações do ganho de S2 no real (exploratórias; uma semente determinística; não
são critérios pré-registrados).**

1. *Coeficientes* (`feature_names`, prior σ = 0,5; γ = inclinação da régua, δ =
intercepto):

| k | γ | δ | 5 maiores \|θ\| (feature: valor) |
|---|---|---|---|
| 25 | 0,49 | −1,97 | dtnasc:L1 +0,43; endereco:L1 +0,22; municipio:L2 −0,12; mae:L2 −0,11; mae:L1 −0,07 |
| 50 | 0,52 | −2,26 | dtnasc:L1 +0,51; nome:L3 −0,24; mae:L1 −0,17; endereco:L1 +0,15; mae:L2 −0,13 |
| 200 | 0,79 | −3,65 | dtnasc:L1 +0,58; mae:L2 −0,39; mae:L1 −0,18; endereco:L1 +0,17; municipio:L2 −0,13 |
| 775 | 0,82 | −3,39 | mae:L2 −0,85; mae:L1 −0,66; dtnasc:L1 +0,48; nome:L2 −0,25; nome:L3 +0,20 |

   Feature dominante: `dtnasc:L1` (+0,43 a +0,58) até k = 200 e `mae:L2` (−0,85) em
   k = 775, quando 10 coeficientes passam de 0,1 (4 a 5 antes). Todos pequenos; o
   efeito é grande porque a régua-stub é plana (ver 3).
2. *Placebo* (linhas dos 8 compostos embaralhadas juntas, nota e verdade fixas; 10
   embaralhamentos), verdadeiros verificados em B = 50 / 100 / 200 / 400 / 800:
   S0 = 10 / 12 / 13 / 13 / 17; S2x = 14 / 15 / 17 / 19 / 33; S2x embaralhada, média
   = 10,1 / 11,1 / 12,3 / 13,5 / 16,2 (máximo 11 / 13 / 13 / 14 / 17). **O ganho
   some**: o que S2 usa está nos compostos, não em artefato da régua ou do método.
3. *Régua mais íngreme* (S2x, B = 50 / 100 / 200 / 400 / 800):

| régua de S2 | 50 | 100 | 200 | 400 | 800 |
|---|---|---|---|---|---|
| stub, sd_gamma = 0,5 (protocolo) | 14 | 15 | 17 | 19 | 33 |
| stub, sd_gamma = 0,1 | 10 | 15 | 17 | 21 | 33 |
| stub, sd_gamma = 0,02 | 10 | 15 | 17 | 19 | 32 |
| stub, sd_gamma = 2 | 15 | 16 | 17 | 19 | 33 |
| Platt do extrato inteiro (A⁺ = 95), sd_gamma = 0,5 | 10 | 12 | 13 | 18 | 20 |
| Platt do extrato inteiro, sd_gamma = 0,1 | 10 | 12 | 13 | 18 | 18 |

   `sd_gamma` não é a alavanca (32 a 33 em B = 800). Trocar o stub pelo Platt
   ajustado no extrato inteiro e aplicado ao pool corta o ganho sobre S0 de +16 para
   +3 (B = 800; +1 com `sd_gamma` = 0,1) e de +6 para +5 em B = 400: **a maior parte
   do ganho de +16 vem da escala plana do stub** (a mesma σ = 0,5 pesa muito mais
   contra uma régua de inclinação ~0,2-0,4 por ponto de nota que contra uma de ~3);
   o que sobrevive (+5 em B = 400) vem dos compostos.
4. *Grupo explicativo* (agregados; B = 800): a régua S0 alcança 17 dos 36; dos 19 que
   não alcança, S2 alcança 16. Entre os 19: `mae_missing` 15, `nome` L3 13, DTNASC
   zerada 17, endereço L0 13 e L1 6, o perfil triplo (`mae_missing` e `nome` L3 e
   DTNASC zerada) em 12. Entre os 16 alcançados por S2: 13, 12, 14, endereço L0 10 e
   L1 6, perfil triplo 12 (todos); nota de 5,47 a 6,90. Entre os 17 que S0 alcança:
   `mae_missing` 2, `nome` L3 5, DTNASC zerada 7, perfil triplo 0. Os 3 que S2
   também não alcança: `mae_missing` 2, `nome` L3 1, DTNASC zerada 3. Rendimento do
   perfil triplo: 12 positivos em 1.476 pares no pool; **12 em 102 (11,8%) na faixa
   de nota 5,47-6,90**, contra 5 em 13.389 (0,04%) nessa faixa sem o perfil e 1,0% na
   margem de S0 (postos 400-800). **Ressalva: o perfil foi lido nos próprios
   positivos que ele explica (em amostra, descoberto depois); não é um teste de
   hipótese e precisa de positivos novos.**

Leitura: o ganho de S2 no real é (i) real no sentido de vir dos compostos (placebo),
(ii) em boa parte (13 de 16 em B = 800) efeito da escala do stub e (iii) coerente com
a régua sub-ponderar o padrão "dados ausentes" (`mae_missing`, nome L3, DTNASC
zerada) em notas 5,5 a 6,9; (iii) é só uma inferência de 12 pares.

## 7. Revisões após os resultados

Status: **leitura posterior aos experimentos**; as seções 1 a 5 (princípios,
hipóteses, critérios) não foram alteradas e os números abaixo vêm das tabelas da
seção 6 e do CSV. Itens marcados *(exploratório)* não eram hipóteses
pré-registradas.

**O que os dados mostraram sobre os princípios.**

- *Princípio 1 (começar pela régua):* sustentado. No cenário base S2x−S0 fica entre
  −1,1 e 0,0 e a S1 (padrão) perde da régua de B = 200 em diante (S1−S0 = −11,1;
  −39,4; −62,7 em B = 200, 400, 800), porque pára em ~190 decisões; no real também
  (−3,8 a −9,8).
- *Princípio 2 (aprender só a correção):* ganha onde a régua é fraca e não custa
  caro onde é boa. Pesos errados: S2−S0 médio −2,08; −0,58; +3,24; +7,30 nos graus 0
  a 3 (até +16,7 em B = 800 com σ = 2, *exploratório*); régua com ruído: −2,08;
  +3,08; +19,30; +29,48 (até +57,2 em B = 800). Com régua correta e sem sinal, σ
  maior só piora (S2x−S0 até −3,5 em σ = 2, *exploratório*). No real S2 vence a
  régua (+3 a +16), mas 13 dos 16 pares de B = 800 dependem da escala do stub (6.7).
- *Princípio 3 (só rótulo verificado treina; LLM é atributo):* sustentado no
  sintético: S3−S4 ≥ +0,6 e chega a +56,8 em B = 200 no cenário base, S4 ≈ S0
  (−0,8 a −1,9). No real o LLM como atributo (S3) rende +16 sobre S0 em B = 400 e o
  como rótulo (S4) alcança S3 em B = 800 (32 e 32): evidência de uma realização do
  LLM e 36 positivos. O vazamento de cobertura (6.7 A) mostra que o fixture precisa
  de cobertura independente do rótulo.
- *Princípio 4 (estimar separado):* a amostra uniforme de 5% de B **custa mais do
  que informa**. Sem ela (S2x) a diferença frente a S0 vai de −2,4 a +8,3 conforme a célula (−1,1
  a 0,0 no base); a amostra custa 1,3 a 2,2 pares em prevalência 1%, até 14,8 em 5% e
  18,9 em 20% (S2−S2x, B ≤ 800), o que derruba H1, H4, H6 e H7 à letra. Os IC do HT cobrem (0,990) por serem largos: largura média 8.829
  para centenas de positivos restantes.
- *Princípio 5 (parar por custo):* H10 ficou inconclusivo (empate por construção em
  5% e 20%). Onde as regras divergem S2 pára mais cedo (k = 98 vs 215 em 0,1%; 476
  vs 793 em 1%) com menos arrependimento em 0,1% (14,9 vs 18,7) e igual em 1%
  (89,8 vs 88,6).
- *Princípio 6 (LLM):* o LLM real **não serve como auditor** nesta prevalência
  (H11 real: recuperação 26% a 48%; precisão 100% só em B = 100 e 2,6% a 5,2% em B = 800,
  15 a 30 vezes melhor que o acaso mas longe de 30%); no sintético serviu (recuperação
  0,59 a 0,78, precisão 0,86 a 0,99) por ter FP de 0,4% e FN humanos suficientes. O
  LLM como atributo de ordenação aumenta o rendimento por decisão (S3 = 146,2 vs
  S0 = 90,3 em B = 200 no sintético; 29 vs 13 em B = 400 no real), o que contraria a
  parte "não reduzir decisões" do princípio 6 enquanto o LLM for gratuito.
- *Bandit:* só ajuda com régua não monotônica (H8: S5−S0 ≤ −0,6 nas células
  monotônicas; +25,0 a +130,3 com o bolso de positivos em B ≥ 400; no real
  +0,05 a +0,80 nas induções). *Subgrupo difícil:* S2 não perde os difíceis (S2−S0
  de −0,12 a +0,78 em média).
- *Régua do extrato real (exploratório):* parece sub-ponderar o padrão "dados
  ausentes" (`mae_missing`, nome L3, DTNASC zerada): 12 dos 19 positivos que S0 não
  alcança têm o perfil, rendimento 11,8% (12/102) na faixa de nota 5,47-6,90 contra
  1,0% na margem de S0. Só vale como hipótese: em amostra e 12 pares.

**Recomendação revisada de engenharia de filas** (as partes sem experimento estão
marcadas).

1. Ordenar pela melhor régua e revisar do topo para baixo; só trocar a ordem com
   evidência de que ela perde algo.
2. Se houver razão para duvidar da régua, usar a correção aprendida com a régua como
   offset (S2) e **reportar a régua efetiva** (`ruler_source`): com menos de 30
   âncoras o stub achata a régua e dá peso grande aos campos. Escolher σ (≥ 1 se a
   régua for fraca) antes de ver os dados, e checar o ganho com um placebo
   (embaralhar os compostos) antes de acreditar nele.
3. Usar o LLM como **atributo** (S3), com cobertura independente do rótulo e
   conhecida; não como rótulo e não como auditor de humanos nesta prevalência.
4. Não gastar 5% de B em amostra uniforme para estimar restantes: o custo é de 1,3 a
   18,9 pares em B ≤ 800 e o IC é inútil. Alternativas **não testadas**: amostra
   menor ou só no fim da campanha, estimador assistido por modelo, estimar restantes
   pela calibração do modelo.
5. Parar por custo esperado com o modelo calibrado (S2) em vez de τ fixo; testar
   com custos que separem as regras (c = 0,10 deu 16,8 vs 28,5 em 0,1%).
6. Bandit sobre faixas só com indício de rendimento não monotônico.
7. Não usar a sessão auto-calibrável atual com as configurações padrão: ela pára em
   ~190 decisões e perde da régua de B = 200 em diante.
8. Próximos experimentos (não feitos): testar o perfil "dados ausentes" em positivos
   novos (validação cruzada dos 36); variar o tamanho da amostra de estimação;
   repetir H11 com um humano cujo erro dependa da dificuldade.

**Após a replicação por passo:** a replicação pré-registrada (seção 8.1) **não
reproduziu** o ganho de S2 fora dos pares em que ele foi observado, com a régua
calibrada: o modelo treinado em dois passos ganha +2, +4 e +1 pares sobre S0 nas
dobras informativas em B = 100, 200 e 400 (R1 inconclusiva) e o S2 online ganha +0 e
+1 em B = 400 (R2 inconclusiva). Com a régua plana (stub) a transferência rende +2,
+4 e +7 (exploratório, fora de R1), o que reforça a leitura da seção 6.7 de que o
ganho depende da escala da régua; R3 ficou inconclusiva (19 vs 19 em B = 200; 25
vs 19 em B = 400). Sobre o padrão "dados ausentes": os modelos treinados sem o
passo de teste dão peso a favor de par à mãe ausente (+0,12 a +0,48, leitura
invariante à referência) nos seis modelos (3 dobras × 2 réguas), mas o nome em
nível máximo é misto (−0,25 a +0,35) e a data de nascimento zerada pesa **contra**
(−0,47 a −1,19); R4 ficou inconclusiva por critério mal especificado. A
afirmação (iii) da seção 6.7 deve ser lida como sustentada só para "mãe ausente" e
ainda sem teste independente (dobras de 7 e 25 positivos; 4 na dobra do passo 6).
A recomendação 2 (usar S2 só com régua duvidosa e reportar a régua efetiva) fica;
a hipótese de que a régua da tese sub-pondera o perfil triplo continua não
testada fora da amostra.

## 8. Replicação por passo de blocagem (pré-registro, 2026-11-15)

Registrada **antes** de rodar, depois de ver os resultados reais de H1/H7 e as
confirmações da seção 6.7. Objetivo: saber se o ganho de S2 no extrato real
(33 × 17 em B = 800 com régua plana; 20 × 17 com régua calibrada) e o perfil
"dados ausentes" se sustentam fora dos pares em que foram observados.

**Dados.** Pares abaixo do corte do extrato da tese, separados por `PASSO`
(6, 9, 11). PAR só como verdade. Dobra = um passo de teste; os outros dois são
de treino. Uma dobra com menos de 5 positivos abaixo do corte é declarada não
informativa e fica fora dos critérios (reportada mesmo assim).

**Variantes de régua.** (P) plana = `predict_stub` (nota/10), como no real;
(C) calibrada = Platt por âncoras ajustado no extrato inteiro e aplicado aos
pares abaixo do corte.

**Procedimentos por dobra.**
- T (transferência): S2 roda no treino com orçamento humano de 800 decisões;
  o modelo final (pesos dos campos e da régua) ordena o passo de teste **sem
  nenhum rótulo do teste**; comparado com S0 (nota) no teste.
- O (online): S2 roda só dentro do passo de teste, aprendendo com os rótulos
  verificados do próprio passo; comparado com S0 no mesmo passo.
- B ∈ {100, 200, 400} decisões humanas por dobra (limitado ao tamanho do passo).

**Hipóteses e critérios** (somas sobre as dobras informativas; humano
perfeito; execução determinística, sem IC).

| Id | Hipótese | Critério de decisão |
|---|---|---|
| R1 | A correção aprendida transfere entre passos (variante C) | Confirmada se ganho(T) − S0 somado ≥ +3 em B = 400 e ≥ 0 em todo B; refutada se < 0 em B = 400; senão inconclusiva |
| R2 | O ganho online se repete dentro de cada passo (variante C) | Confirmada se ganho(O) − S0 > 0 em B = 400 em todas as dobras informativas; refutada se < 0 em alguma; senão inconclusiva |
| R3 | A régua plana supera a calibrada fora da amostra | Confirmada se, em T, a soma de encontrados com P > com C em B = 400 e em B = 200; refutada se ≤ em ambos |
| R4 | O padrão "dados ausentes" é aprendido sem ver o passo de teste | Confirmada se, nas três dobras, o modelo de treino dá peso a favor de par (sinal positivo em relação à referência) a pelo menos dois de: `mae:missing`, nível de concordância máxima de nome, nível zero de data de nascimento contra o esperado pela régua; refutada se em nenhuma dobra |

Ressalva registrada antes: o perfil da seção 6.7 foi identificado com todos os
36 positivos, inclusive os de cada passo de teste; R4 testa se o **modelo**
aprende o padrão sem esses rótulos, não se o perfil foi descoberto de forma
independente.

### 8.1 Resultados

Executado em 2026-11-15 com `python notebooks/exp_engenharia_filas.py
replicate-passo --real-input ../phd-research/data/COMPARADORSEMIDENT.csv`
(2 s; determinístico, humano perfeito; agregados em
`docs/engenharia_de_filas_replicacao_passo.csv`; testes em
`tests/test_exp_engenharia_filas.py`). O critério e o texto da seção 8 não foram
alterados.

**Pares e positivos por passo** (abaixo do corte): passo 6: 2.573 pares, 4
positivos (**não informativa**, fora dos critérios); passo 9: 1.329 pares, 7
positivos; passo 11: 57.583 pares, 25 positivos (total 61.485 e 36). Só as dobras
dos passos 9 e 11 entram em R1-R4.

**Procedimentos como implementados.** T: treino nos outros dois passos com 800
decisões humanas (exploração de S2, com reajuste a cada 25), reajuste final com as
800 e ordenação do passo de teste pelo modelo; nenhum rótulo do teste entra (teste
automatizado: inverter todos os rótulos do teste não muda o modelo nem a ordem). O:
S2 só dentro do passo de teste, também por exploração (a amostra de estimação não
conta; incluí-la dá os mesmos números, coluna `O_com_amostra`). As colunas e os
níveis dos campos vêm do pool inteiro (só atributos, sem rótulos). Régua P =
stub `nota/10` (`ruler_source = stub`); C = Platt por âncoras ajustado no extrato
inteiro e aplicado ao pool (`anchor_platt_full_extract`).

**Verdadeiros verificados** (S0 = ordem pela nota no passo; T = modelo de treino;
O = online):

| passo (pares; positivos) | régua | B | S0 | T | O |
|---|---|---|---|---|---|
| 6 (2.573; 4) não informativa | P | 100 / 200 / 400 | 3 / 3 / 3 | 4 / 4 / 4 | 4 / 4 / 4 |
| 6 | C | 100 / 200 / 400 | 3 / 3 / 3 | 3 / 3 / 3 | 3 / 3 / 3 |
| 9 (1.329; 7) | P | 100 / 200 / 400 | 5 / 6 / 7 | 7 / 7 / 7 | 7 / 7 / 7 |
| 9 | C | 100 / 200 / 400 | 5 / 6 / 7 | 5 / 7 / 7 | 5 / 7 / 7 |
| 11 (57.583; 25) | P | 100 / 200 / 400 | 9 / 9 / 11 | 9 / 12 / 18 | 12 / 14 / 14 |
| 11 | C | 100 / 200 / 400 | 9 / 9 / 11 | 11 / 12 / 12 | 9 / 9 / 12 |

**R4 — pesos do modelo de treino** (coeficiente relativo à referência da
codificação; `mae:missing` em duas leituras porque nesta codificação `missing` é a
própria referência de `mae`, sem coluna: *literal* = não definido; *invariante* =
menos a média dos demais níveis de `mae`; `dtnasc zero` = menos a média dos níveis
de data de nascimento acima de zero, ou seja, positivo quando o nível zero pesa
mais a favor de par):

| passo de teste | régua | mae:missing (invariante) | nome L3 | dtnasc zero |
|---|---|---|---|---|
| 6 (não informativa) | P | +0,48 | +0,14 | −0,47 |
| 9 | P | +0,35 | +0,16 | −0,52 |
| 11 | P | +0,30 | +0,35 | −0,56 |
| 6 (não informativa) | C | +0,25 | −0,25 | −1,19 |
| 9 | C | +0,12 | −0,13 | −0,95 |
| 11 | C | +0,22 | +0,15 | −0,70 |

**Status** (critérios exatamente como na tabela da seção 8):

| Id | Status | Números | Leitura |
|---|---|---|---|
| R1 | **inconclusiva** | soma T−S0 (C, passos 9 e 11): +2 (B=100), +4 (B=200), +1 (B=400) | nunca abaixo de 0, mas fica abaixo do +3 exigido em B = 400: a correção aprendida transfere pouco com a régua calibrada |
| R2 | **inconclusiva** | O−S0 em B=400 (C): passo 9: +0; passo 11: +1 | o ganho online não aparece no passo 9 e é de 1 par no 11; nunca negativo |
| R3 | **inconclusiva** | T somado, P vs C: B=200: 19 vs 19; B=400: 25 vs 19 | a régua plana não perde da calibrada e ganha em B = 400, mas empata em B = 200, então a letra (maior em ambos) falha |
| R4 | **inconclusiva — critério mal especificado** | veredito por (leitura, régua) nas dobras informativas: literal/C refutada, literal/P refutada, invariante/C inconclusiva, invariante/P confirmada | o critério não diz a régua nem o que fazer quando `missing` é a referência; mãe ausente é favorecida em todos os modelos, nome L3 é misto e data de nascimento zerada pesa contra |

Observação fora dos critérios: com a régua plana P a transferência soma +2, +4 e
+7 sobre S0 em B = 100, 200 e 400 (passos 9 e 11), e o online no passo 11 soma +3,
+5 e +3: o ganho está na régua plana, não na calibrada.

**Ressalvas.**
- Poucos positivos por passo (4, 7 e 25): uma diferença de 1 par muda o status; a
  execução é determinística e não há IC.
- Humano perfeito; o treino do passo 11 vem de dois passos pequenos (3.902 pares,
  11 positivos) e o treino dos outros passos vem de ~58.000 pares.
- O perfil "dados ausentes" foi derivado com todos os 36 positivos (também os de
  cada passo de teste); R4 mede só se o modelo o aprende sem os rótulos do teste.
- As definições operacionais de R4 (duas leituras de `mae:missing`, nível zero da
  data de nascimento como menos a média dos demais níveis, decisão por todas as
  leituras e réguas) foram fixadas ao implementar, antes dos números finais de
  R4; o literal `mae:missing` só foi visto como indefinido ao rodar.
- B limitado ao tamanho do passo (não houve caso: o menor passo tem 1.329 pares).
