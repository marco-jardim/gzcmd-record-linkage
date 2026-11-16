# Fila auto-calibrável ponta a ponta com o Haiku real (humano simulado)

> **Aviso:** uma única execução com LLM real (temperatura padrão; o veredito estocástico fica congelado no fixture); humano simulado **perfeito** (decide todo item do portão pelo PAR); método experimental. Não é avaliação final da fila. Apenas agregados; nada em nível de par.

## Protocolo

- Pool `all` do extrato (61.696 pares; 247 verdadeiros, 36 deles abaixo do corte). A sessão não vê `PAR` nem `TARGET`.
- Sessão com a configuração padrão: modo `human`, `field_set` `composites`, 50 decisões humanas por rodada, `max_reviews`=2000, `max_rounds`=5, semente 20261115.
- Revisor: `claude-haiku-5-5`, prompt `v4`, sem `model_outputs` e sem `features.nota_final`; cache em disco e pré-busca paralela dos próximos pares de cada braço (`peek_upcoming`).
- Rótulos da recalibração: decisões humanas (verificadas) mais, conforme a política, os NONMATCH do LLM (`llm_nonmatch`) ou nenhum negativo sem verificação (`verified_only`).
- H1: ordem da nota decrescente, LLM revisa, humano confirma só os MATCH. H0: humano sozinho na ordem da nota.

## Rodadas — `llm_nonmatch`

Status: **finished** (motivo de parada do run: `sterile_round`); 1.443 revisões do LLM.

| rodada | revisões LLM | MATCH LLM | decisões humanas | over_budget | verdadeiros verificados | FP pegos | abaixo do corte (de 36) | parada da rodada |
|--:|--:|--:|--:|--:|--:|--:|--:|---|
| 1 | 500 | 86 | 50 | 79 | 50 | 0 | 4 | max_round_reviews |
| 2 | 500 | 199 | 50 | 231 | 46 | 4 | 0 | max_round_reviews |
| 3 | 443 | 2 | 27 | 0 | 0 | 2 | 0 | posterior |
| **total** | 1443 | 287 | 127 | 310 | 96 | 6 | 4 | sterile_round |

Recalibração ao fim de cada rodada (rótulos verificados por humano):

| rodada | status | rotulados | pos | neg | Kish | ‖Δcoef‖₂ vs prior | máx abs Δcoef | Δintercepto |
|--:|---|--:|--:|--:|--:|--:|--:|--:|
| 1 | fitted | 421 | 50 | 371 | 140,4 | 3,05 | 2,35 | -3,25 |
| 2 | fitted | 690 | 96 | 594 | 236,5 | 2,66 | 2,42 | -4,89 |
| 3 | fitted | 1133 | 96 | 1037 | 381,6 | 2,62 | 2,25 | -5,19 |

## Rodadas — `verified_only`

Status: **finished** (motivo de parada do run: `sterile_round`); 1.406 revisões do LLM.

| rodada | revisões LLM | MATCH LLM | decisões humanas | over_budget | verdadeiros verificados | FP pegos | abaixo do corte (de 36) | parada da rodada |
|--:|--:|--:|--:|--:|--:|--:|--:|---|
| 1 | 500 | 86 | 50 | 79 | 50 | 0 | 4 | max_round_reviews |
| 2 | 500 | 199 | 50 | 231 | 46 | 4 | 0 | max_round_reviews |
| 3 | 406 | 3 | 28 | 0 | 0 | 3 | 0 | posterior |
| **total** | 1406 | 288 | 128 | 310 | 96 | 7 | 4 | sterile_round |

Recalibração ao fim de cada rodada (rótulos verificados por humano):

| rodada | status | rotulados | pos | neg | Kish | ‖Δcoef‖₂ vs prior | máx abs Δcoef | Δintercepto |
|--:|---|--:|--:|--:|--:|--:|--:|--:|
| 1 | skipped_insufficient | 50 | 50 | 0 | 50,0 | 0,00 | 0,00 | 0,00 |
| 2 | fitted | 100 | 96 | 4 | 59,7 | 0,81 | 0,73 | -0,90 |
| 3 | fitted | 128 | 96 | 32 | 56,5 | 1,13 | 1,08 | -0,77 |

## Comparação com H1 e H0

**A decisões humanas iguais** (a fila decide até 50 itens por rodada; os pontos são os acumulados ao fim de cada rodada; H1 gasta uma decisão por MATCH do LLM):

| política | rodada | decisões humanas | fila: verificados | fila: abaixo | H1: verificados | H1: abaixo | H1: revisões LLM | H0: verificados | H0: abaixo |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| llm_nonmatch | 1 | 50 | 50 | 4 | 50 | 4 | 84 | 49 | 5 |
| llm_nonmatch | 2 | 100 | 96 | 4 | 90 | 8 | 541 | 98 | 9 |
| llm_nonmatch | 3 | 127 | 96 | 4 | 106 | 10 | 929 | 113 | 9 |
| verified_only | 1 | 50 | 50 | 4 | 50 | 4 | 84 | 49 | 5 |
| verified_only | 2 | 100 | 96 | 4 | 90 | 8 | 541 | 98 | 9 |
| verified_only | 3 | 128 | 96 | 4 | 107 | 10 | 935 | 114 | 9 |

**Orçamentos de referência das bases** (50/100/200/400 decisões humanas; a fila, com 50 por rodada e no máximo 5 rodadas, não passa de 250):

| decisões humanas | H1: verificados | H1: abaixo | H1: revisões LLM | H0: verificados | H0: abaixo |
|--:|--:|--:|--:|--:|--:|
| 50 | 50 | 4 | 84 | 49 | 5 |
| 100 | 90 | 8 | 541 | 98 | 9 |
| 200 | 115* | 10* | 1443 | 127 | 12 |
| 400 | 115* | 10* | 1443 | 140 | 13 |

H1 foi calculada sobre 1.443 revisões do LLM (pedidas: 1.443; status `finished`). `*` = a base não chegou a esse nº de decisões humanas dentro das revisões disponíveis (valor ao fim delas).

**A revisões de LLM iguais** (o humano confirma todo MATCH do H1):

| política | rodada | revisões LLM | fila: verificados | fila: abaixo | fila: decisões humanas | H1: verificados | H1: abaixo | H1: decisões humanas |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| llm_nonmatch | 1 | 500 | 50 | 4 | 50 | 88 | 6 | 96 |
| llm_nonmatch | 2 | 1000 | 96 | 4 | 100 | 108 | 10 | 129 |
| llm_nonmatch | 3 | 1443 | 96 | 4 | 127 | 115 | 10 | 142 |
| verified_only | 1 | 500 | 50 | 4 | 50 | 88 | 6 | 96 |
| verified_only | 2 | 1000 | 96 | 4 | 100 | 108 | 10 | 129 |
| verified_only | 3 | 1406 | 96 | 4 | 128 | 114 | 10 | 138 |

## Políticas de negativos

| métrica | `llm_nonmatch` | `verified_only` |
|---|--:|--:|
| revisões do LLM | 1443 | 1406 |
| decisões humanas | 127 | 128 |
| verdadeiros verificados | 96 | 96 |
| abaixo do corte verificados | 4 | 4 |
| FP do LLM pegos pelo humano | 6 | 7 |
| over_budget | 310 | 310 |
| motivo de parada | `sterile_round` | `sterile_round` |

## Discussão

- `llm_nonmatch`: 96 verdadeiros verificados (4 de 36 abaixo do corte) com 127 decisões humanas e 1.443 revisões do LLM. Com as mesmas 127 decisões, H1 verifica 106 (10 abaixo) e H0 verifica 113 (9 abaixo); com as mesmas 1.443 revisões do LLM, H1 verifica 115 (10 abaixo) gastando 142 decisões humanas.
- `llm_nonmatch`: 3 rodadas de no máximo 5; a última (3) parou por `posterior` e trouxe 0 verdadeiros novos; motivo de parada do run: `sterile_round` (`max_reviews`=2000 não foi atingido: 1.443 revisões). O LLM marcou 287 MATCH; 310 candidatos do portão (MATCH, UNSURE ou auditoria) ficaram sem decisão humana por falta de orçamento (50 por rodada).
- `verified_only`: 96 verdadeiros verificados (4 de 36 abaixo do corte) com 128 decisões humanas e 1.406 revisões do LLM. Com as mesmas 128 decisões, H1 verifica 107 (10 abaixo) e H0 verifica 114 (9 abaixo); com as mesmas 1.406 revisões do LLM, H1 verifica 114 (10 abaixo) gastando 138 decisões humanas.
- `verified_only`: 3 rodadas de no máximo 5; a última (3) parou por `posterior` e trouxe 0 verdadeiros novos; motivo de parada do run: `sterile_round` (`max_reviews`=2000 não foi atingido: 1.406 revisões). O LLM marcou 288 MATCH; 310 candidatos do portão (MATCH, UNSURE ou auditoria) ficaram sem decisão humana por falta de orçamento (50 por rodada).
- Políticas de negativos: na rodada 1, `llm_nonmatch` recalibra com 421 rótulos (50 pos, 371 neg; Kish 140,4; status `fitted`) e `verified_only` com 50 (50 pos, 0 neg; status `skipped_insufficient`). Ao fim, ‖Δcoef‖₂ vs prior é 2,62 em `llm_nonmatch` e 1,13 em `verified_only`; verdadeiros verificados: 96 e 96; abaixo do corte: 4 e 4.

## Custo

| etapa | pares com veredito | chamadas | tokens entrada | tokens saída | escrita cache | leitura cache | revisões novas na API | relógio (s) | workers | HTTP 429/erros |
|---|--:|--:|--:|--:|--:|--:|--:|--:|---|---|
| fila `llm_nonmatch` | 1.443 | 3.105 | 2.252.301 | 5.333.905 | 78.509 | 7.465.920 | 1.268 | 3.292 | 24, 32 | {'429': 44} |
| fila `verified_only` | 1.406 | 3.038 | 2.218.268 | 5.239.867 | 78.509 | 7.303.329 | 302 | 595 | 24 | nenhum |
| base H1 | 1.443 | | | | | | 805 | 818 | 24 | nenhum |

A primeira execução da fila `llm_nonmatch` usou 32 workers e teve as respostas 429 e os 2 erros de revisão registrados acima; as demais usaram 24 workers, sem 429.

Teto global de novas revisões de par na API: 2375 usadas de 3000. As execuções são retomáveis: refazer a fila reaplica o que está no cache sem custo, então o relógio soma só o tempo útil de cada execução.

