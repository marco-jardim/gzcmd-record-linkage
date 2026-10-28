# Notebook didático v2: reescrita de `gzcmd_passo_a_passo.ipynb`

Fonte da verdade do notebook: `notebooks/build_notebook.py` (`ALL_PHASES`, ~L2095–2107; `nbformat.write` ~L2137). Nunca editar o `.ipynb` à mão: ele é regenerado com `python notebooks/build_notebook.py`.

Decisões do usuário:
- Reescrever o notebook existente (mesmo caminho/nome; testes continuam valendo).
- Autocalibração por âncoras: explicar o Platt supervisionado que existe e implementar uma **mini-demo didática só em `notebooks/nb_helpers.py`** (nada em `src/`), comparando lado a lado e deixando claro que a biblioteca não faz isso.
- Seis controles interativos (ipywidgets) com fallback estático: W1 Platt (a, b); W2 custos FP/FN/LLM; W3 âncoras; W4 bandas e guardrails; W5 pesos das features e janelas MACD; W6 orçamento de LLM.
- O `.ipynb` gerado continua **sem saídas armazenadas**.
- Toda prosa do notebook em pt-BR com acentos corretos, nível ELI11 nos pontos difíceis, **sem travessões (—)** (regra herdada do commit d246b44).
- Sem novas dependências pesadas: só matplotlib + ipywidgets (já em `requirements/notebook.txt`).

`C#` abaixo = índice da célula no notebook atual (derivado da ordem de `ALL_PHASES`).

---

## 0. Decisões de design

1. **Nomes dos parâmetros de Platt.** `a` = inclinação (slope), `b` = intercepto: `p = σ(a·s + b)`. É a convenção já usada no notebook (build_notebook.py:752–775) e na YAML (`sigmoid(a*logit(p_base)+b)`, gzcmd_v3_config.yaml:102). A trava de 0,5 em `a`/`b` (YAML L118–119) depende dessa convenção fixa.
2. **Como a demo de âncoras funciona.** Platt supervisionado de novo, mas ajustado **só nas linhas-âncora** com **rótulos suaves** t = 0,98 / 0,001, com os dois grupos de peso igual. O loop de Newton é o mesmo da biblioteca, então a matemática da §9 se reaproveita, e rótulos suaves nunca são perfeitamente separáveis, então o ajuste sempre tem resposta finita.
   - `fit_platt` da biblioteca não serve: rejeita alvo diferente de 0/1 (calibration.py:69–70).
   - A solução "duas equações em log-odds" é só a conta de guardanapo da prosa e o ponto de partida do Newton, não um segundo método.
   - `config.py` não lê `anchors` nem `calibration.safeguards` (`GZCMDConfig`, config.py:84–93), então o helper lê a YAML crua.
3. **Onde as âncoras são ajustadas.** Em **todas as linhas** (o ajuste não usa rótulos, então não há vazamento de rótulo); avaliado no mesmo teste held-out do Platt supervisionado. O notebook também imprime que, só no treino, a âncora negativa cairia abaixo de `min_count`.
4. **Widgets ficam ao lado do conceito:** W5 na §7, W1 e W3 na §9, W4 na §10, W2 e W6 na §11. A §15 vira um guia em markdown dos seis controles. O `painel_operacao` antigo (C103) sai; W1 e W2 cobrem o que ele fazia.
5. **Imports em `nb_helpers.py`:** matplotlib e ipywidgets importados **dentro** das funções. `requirements/dev.txt` não tem matplotlib (dev.txt:1–8) e `test_scaffold_imports` importa `nb_helpers` no ambiente dev (test_scaffold_imports.py:19–22).
6. **Sem travessões** na prosa; propor teste que garanta isso.

### Achados para a prosa (todos do código)

| # | Achado | Evidência |
|---|---|---|
| F1 | Data de nascimento **idêntica** dá `dtnasc_score_total = 0.2`, não 1.0. "Igual" e "diferente por um dígito" são exclusivos e as inversões são raras, então a média de 5 raramente passa de 0,4. | loader.py:231–235; synthetic_data.py:410–414 |
| F2 | No sintético, datas discordantes diferem por 0/±1 dia ou por 2–30 anos. As janelas MACD de 7 e 30 dias quase não importam. | synthetic_data.py:388–395 |
| F3 | Cenário `homonimo` tem `NOME qtd = 0.9` < 0,95 ("nome perfeito"), então `macd_nome_perf_x_date_far` = 0 para ele. Quem o pega é o guardrail `homonimia_risk`. | synthetic_data.py:138; loader.py:281–283; guardrails.py:134–142 |
| F4 | **Bug fora de escopo:** linhas de cenário usam a coluna `"ENDERECO tokens jaccard"`, mas as aleatórias e o loader usam `"ENDERECO tokens jacc"`. Linhas de cenário (inclusive o herói) têm média de endereço com 5 de 6 subscores (herói ≈ 0,417). **Não corrigir** nesta reescrita (mudaria o CSV versionado); deixar o endereço do herói fora da conta à mão. | synthetic_data.py:49 vs 445; loader.py:243 |
| F5 | Verdade sintética estimada nas âncoras (s0 ≈ 6,0): p* médio ≈ 0,94 com nota ≥ 9 e ≈ 0,08–0,10 com nota < 5, contra alvos 0,98 e 0,001. O ajuste ancorado sai ~2x mais íngreme e deslocado à direita. O notebook **mede**, não afirma. | synthetic_data.py:231–256, 333–335 |
| F6 | Contagens estimadas: ~108 linhas com nota ≥ 9 e ~238 com nota < 5 nas 607. No treino de 70%, negativas ≈ 167 < min_count 200. | synthetic_data.py:231–233 |
| F7 | Travar `a` e `b` separadamente pode pôr o ponto de 50% (s*) onde nenhuma das curvas estava. Exemplo em B.3. | YAML:118–119 |
| F8 | Orçamentos (2000 vigilância / 1000 confirmação) excedem as 607 linhas; o orçamento nunca aperta aqui. W6 precisa permitir valores bem abaixo da config. | YAML:160–172 |
| F9 | Travas podem pedir revisão mesmo com EVR < 0 (ex.: p = 0,80 em vigilância): compram segurança, não economia. Linhas decididas por guardrail podem ter `review_requested=True` mas nunca são selecionadas. | policy_engine.py:103–109, 124, 131, 137 |
| F10 | ~54% das linhas compartilham COMPREC e ~44% compartilham REFREC (de `unique_share` 0,70/0,76). Os splits ainda saem quase iguais porque as features de cada linha são sorteadas independentemente do ID. | synthetic_data.py:288–294, 332–446, 418–419 |
| F11 | O Platt da biblioteca vê **só** `nota_final`. As features alimentam guardrails, classificador ML, dossiê do LLM e monitoramento de drift. | calibration.py:21–26; classifier.py:4, 72; YAML:237–246 |

Números marcados como estimativa (s0 ≈ 6,0, F5, F6, F10) **não foram medidos**: cada um tem uma célula de código que imprime o valor real, e a prosa os rotula "ilustrativo" ou "o código mede".

---

## A. Nova estrutura

Fio do herói vira um dict `PASSAPORTE`: §6 cria, a célula principal de cada seção acrescenta uma linha, §14 imprime uma vez. Todas as células de cartão do herói separadas (C25, C34, C50, C58, C72, C100) saem.

| § | Propósito | Novas células, em ordem | Cortado/mesclado | Tamanho |
|---|---|---|---|---|
| Título/Glossário | Orientação | md: título, "como ler" com legenda de ícones (🧒 ELI11 · 📐 matemática passo a passo · 🎛️ controle · ⚠️ honestidade), glossário | C0, absorvendo C3 | ~35 linhas |
| 1 | Contexto | md | C1 cortado ~30% | ~15 |
| 2 | Pipeline | md: diagrama, os dois modos, rotas A/B | C2 cortado | ~25 |
| 3 | Objetivos | md: 5 bullets curtos | C3 → ~6 linhas (o título fica) | ~6 |
| 4 | Setup | md (nota Colab, 4 linhas) → code BOOTSTRAP (lógica idêntica, comentários enxutos) → md → code (imports, `cfg`, `CONFIG_PATH`, `ANCORAS = nbh.load_anchor_spec(...)`, `SEED`) → md → code (gera e salva o CSV, **mesma chamada** de L273) | C4–C9; C6+C7 numa célula "imports + config". Carregar `cfg` vem para cá (saindo de C28) para que todo número venha da config. | ~6 células |
| 5 | Primeiro olhar | md (§5 + dicionário de colunas condensado) → code (`head(5)`) | C10, C11, C12 | 2 |
| 6 | Herói | md ("exemplo-fio-condutor", ideia do passaporte) → code (acha o herói, cria `PASSAPORTE`, mostra subscores crus do herói, usados na conta à mão da §7) | C13, C14; recap C15 sai | 2 |
| 7 | Carga e features | md 7.0 (🧒 + honestidade F11) → code (carga e resumo) → md 7.1 (📐 receitas, exemplo do herói, F1) → code (refaz as somas do herói à mão e `assert` contra o loader; médias de cada agregado por TARGET) → md 7.2 (diff_ano vs dias, MACD, F2/F3) → code (taxas dos flags MACD por TARGET + a linha homonimo) → md 7.3 (pergunta da figura) → code (histograma da nota, igual C23) → md 7.4 🎛️ W5 → code W5 | C16–C26; C19+C21 na tabela por TARGET; C24–C26 saem | 10 |
| 8 | Bandas | md (🧒 + semântica das bandas) → code (assign, contagens e tabela de fronteiras numa célula) → md (figura) → code (histograma colorido) | C29 vai para o md; C30 mesclado em C28; C33–C35 saem | 4 |
| 9 | Calibração | md 9.0 (🧒 + "probabilidade calibrada") → md 9.1 (📐 odds → log-odds → reta → sigmoide) → md 9.2 (📐 log-loss + L2) → md 9.3 (📐 Newton) → code (traço do Newton: exemplo de bolso e dados reais, comparado com `fit_platt`) → md 9.4 (rotas A/B condensadas) → code (A e B juntas, tabela comparativa) → md 9.5 (📐 ECE/Brier exemplo, leitura do diagrama) → code (reliability via `nbh.plot_reliability`) → md 9.6 🎛️ W1 → code W1 → md 9.7 (p* + honestidade (i)) → code (scatter p*) → md 9.8 (âncoras: 🧒, tabela da config, ⚠️) → md 9.8b (📐 rótulo suave, conta de guardanapo) → code (ajuste âncora vs supervisionado vs verdade) → md 🎛️ W3 → code W3 → md 🔒 trava → code (simulação de janelas e gráfico) → md 9.9 (config vs código, nota ML) | C36–C51. C38, C40, C41 → 9.4; binning duplicado de C44 trocado pelo helper; C47+C48 → 9.9 (~6 linhas); C49–C51 saem | ~21 |
| 10 | Guardrails | md (🧒, 4 regras, tabela de cenários, ⚠️ fix (v)) → code (`apply_guardrails` **com `cfg.guardrails.*`**, como runner.py:194–201; contagens; checagem por cenário substituindo a tabela "3 exemplos") → md 🎛️ W4 → code W4 | C52, C53, C55, C57 → 1 md; C54+C56 → 1 code; C58, C59 saem | 4 |
| 11 | Triagem | md 11.0 (🧒 guarda-chuva) → md 11.1 (📐 três perdas, números do herói) → md 11.2 (📐 ponto de virada, janela de revisão, EVR) → code (`nbh.thresholds_table(cfg)`) → md 11.3 (travas, fila, orçamento, F8/F9) → code (dois engines, tabela de ações, PASSAPORTE) → md 🎛️ W2 → code W2 → md (pares que mudam entre modos) → code (`head(5)`) → md 🎛️ W6 → code W6 | C60 condensado; `<details>` de C61 aberto em 11.1/11.2; C64 vai para o md; C65–C68 (barras) substituídos pelo estático do W2; C71–C73 saem | 12 |
| 12 | Reconciliação | md (C74+C75) → code (`run_v3`, comparação e dict do summary numa célula) → md ("por que bate exatamente", C79+C80) | C74–C80, C78 mesclado em C76 | 3 |
| 13 | Held-out | md (intro + **Fβ, fix (iv)**) → code (eval multi-seed) → md → code (barras rotuladas "F₂ (vig.) / F₀,₅ (conf.)") → md 13.3 (medir, não afirmar: fix (ii)) → code (imprime frações de IDs repetidos e o gap máximo de Fβ) → md (leitura: F10) → md 13.4 → code PR/ROC → md 13.5 → code (curva de custo com **custos do cfg**, fix (iii), + linha teórica do ponto de virada, trava rotulada como trava) → md 13.6 (curto) | C81–C93; C88 reescrito; C93 pela metade | 12 |
| 14 | LLM | md (C94+C95) → code (stub) → md → code (métricas finais) → md ("passaporte final") → code (tabela do passaporte e resultado da revisão do herói) | C99/C100 viram o passaporte; C101 sai | 6 |
| 15 | Painel interativo | md: tabela ligando os 6 controles a seção e pergunta + "Desafios para a aula" | C102–C104 substituídos | 1 |
| 16 | Conclusões | md, limitações reescritas (ver D) | C105 | 1 |

Total ≈ 87–88 células.

**Contrato de variáveis** (slices posteriores dependem delas): `df_raw, p_true, meta, SEED, CSV_PATH, CONFIG_PATH, cfg, ANCORAS, df, hero_idx, PASSAPORTE, platt_insample, df["p_cal"], train_idx, test_idx, df_train, df_test, platt_holdout, p_cal_test, y_test, p_true_test, ece_holdout, brier_holdout, df["band"], df["guardrail"], df["guardrail_reason"], engine_vig, engine_conf, out_vig, out_conf, out_run, summary, res, SEEDS_32, decisoes_revisao`.

---

## B. Rascunhos de prosa (pt-BR) para as partes difíceis

### B.1 §7 feature engineering

**7.0 (md)**
> ## 7. Carga dos dados e *feature engineering*
>
> > 🧒 **Como para uma criança de 11 anos.** Imagine dois detetives comparando duas fichas. Um deles anota pistas: "os nomes batem em 6 de 10 pedaços", "as datas diferem por um dia", "a cidade é a mesma". O `loader` é o assistente que pega dezenas dessas anotações soltas e as organiza em poucas **notas de 0 a 1**, uma por assunto, e algumas **bandeirinhas de alerta**. Ele não decide nada; só arruma a mesa para quem vai decidir.
>
> > ⚠️ **Uma honestidade logo de início.** A calibração Platt da biblioteca usa **apenas a `nota_final`**. As colunas desta seção alimentam os *guardrails* (seção 10), o caminho opcional de *machine learning* (`GZCMDClassifier`), o dossiê do revisor e o monitoramento de *drift*. Entender como elas nascem é entender o que esses estágios "enxergam".

**7.1 (md, r-string)**
> ### 7.1 A receita de cada nota, conta por conta
>
> **🧒 Os três juízes do nome.** O primeiro juiz conta **quantos pedaços** do nome batem e vale metade da nota. O segundo só olha se o **primeiro nome** bate e vale um quarto. O terceiro só olha o **último sobrenome** e vale o outro quarto. A nota do nome é a soma dos votos.
>
> **📐 Passo a passo com o herói.** Pistas cruas: `NOME qtd frag iguais = 0,6`, `NOME prim frag igual = 1`, `NOME ult frag igual = 0`.
> 1. Juiz da quantidade: 0,5 × 0,6 = 0,30
> 2. Juiz do primeiro nome: 0,25 × 1 = 0,25
> 3. Juiz do último sobrenome: 0,25 × 0 = 0,00
> 4. Soma: 0,30 + 0,25 + 0,00 = **0,55**
> 5. "Prender" em [0, 1] (`clip`): 0,55 já está dentro.
>
> $$\text{nome\_score\_total}=\operatorname{clip}_{[0,1]}\big(0{,}5\,q+0{,}25\,\mathbb 1_{\text{prim}}+0{,}25\,\mathbb 1_{\text{ult}}\big)$$
>
> Como os pesos somam 1 e cada pista vive em [0, 1], o `clip` é só um cinto de segurança contra dados fora do padrão. O nome da mãe usa **a mesma receita**: no herói, 0,5 × 0,5 + 0 + 0 = **0,25**.
>
> **Data de nascimento: a média de cinco perguntas de sim ou não** (igual? diferente por um dígito? dia invertido? mês invertido? ano invertido?). O herói só acende "diferente por um dígito": 1 ÷ 5 = **0,2**.
>
> > 🔎 **Pegadinha que vale ouro.** Uma data **idêntica** também dá só 0,2, porque "igual" e "quase igual" nunca acendem juntos. Nesta coluna, 0,2 não quer dizer "data ruim"; quer dizer "uma das cinco formas de concordar aconteceu". Por isso se lê a receita antes de interpretar o número.
>
> **Endereço e município.** `endereco_score_total` é a média de seis subscores de endereço; `municipio_score` vale 1 quando o município coincide. **Bandeirinhas:** `mae_missing`, `endereco_zero` e `dtnasc_all_zero` acendem quando *todos* os subscores do grupo são zero. Zero em tudo costuma significar campo vazio, não discordância, e por isso ganha indicador próprio, como fazemos com dados ausentes antes de ajustar um modelo.

(Os valores crus do herói citados acima devem ser conferidos contra o CSV pelo código da célula seguinte; se divergirem, a prosa usa os valores reais.)

**7.2 (md)**
> ### 7.2 Por que a data ganha lupa própria (as *features* MACD)
>
> **🧒 A régua de anos é grossa demais.** `diff_ano` compara só o **ano**. 31/12/1980 contra 01/01/1981 estão a um dia de distância, mas `diff_ano` diz 1. Já 01/01/1980 contra 31/12/1980 dá `diff_ano = 0`, apesar de quase um ano de distância. As colunas `macd_*` trocam essa régua por uma **régua em dias**.
>
> **📐 As contas, uma por uma:**
> 1. `dias = |C_DTNASC − R_DTNASC|`. Herói: 15/03/1980 contra 16/03/1980, dias = 1.
> 2. `anos = dias ÷ 365,25` ≈ 0,003.
> 3. `macd_nasc_very_close = 1` se dias ≤ 7 (herói: acende).
> 4. `macd_nasc_close = 1` se dias ≤ 30 (herói: acende).
> 5. "Nome perfeito" = quantidade ≥ 0,95 **e** primeiro nome igual **e** último sobrenome igual (herói: 0,6 < 0,95, não é).
> 6. `macd_nome_perf_x_date_far = 1` se nome perfeito **e** anos ≥ 5: o retrato do **homônimo**, mesmo nome e gerações diferentes.
> 7. `macd_nome_perf_x_year_diff = (nome perfeito) × min(anos, 20)`: a mesma ideia em versão contínua, com teto de 20 anos.
>
> **Por que isso importa.** Nome igual é a pista mais forte e a mais traiçoeira ("Maria da Silva" é comum). O que separa a mesma pessoa de um homônimo costuma ser a data.
>
> > 🔎 **Dois avisos honestos.** (1) Neste gerador, datas discordantes diferem por ±1 dia ou por 2 a 30 anos; quase não há diferenças entre 8 e 300 dias. Por isso as janelas de 7 e 30 dias mudam pouco aqui, e o controle 7.4 mostra isso. (2) O cenário `homonimo` tem quantidade 0,9, abaixo de 0,95, então o flag MACD **não** acende para ele; quem o pega é o guardrail de homonímia (seção 10).

**7.4 🎛️ (md)**
> Mexa nos pesos do nome e nas janelas de dias. A figura separa a nota do nome por classe e mostra o **AUC**: "sorteando um match e um não-match, qual a chance de o match ter nota maior?" (0,5 é cara ou coroa; 1,0 é separação perfeita). Teste: (a) zerar o peso da quantidade derruba o AUC? (b) com pesos somando mais que 1, o que o `clip` faz? (c) levar a janela de 30 para 300 dias muda algo neste dataset? Por quê?

### B.2 §9 calibração

**9.0 (md)**
> ## 9. Calibração: de escore a **probabilidade** confiável
>
> > 🧒 **Como para uma criança de 11 anos.** A moça da previsão do tempo diz "70% de chance de chuva". Ela é confiável se, juntando todos os dias em que disse 70%, choveu em mais ou menos 7 de cada 10. Não precisa acertar todo dia; o número dela precisa **significar** o que diz.
>
> **O que é uma probabilidade calibrada.** Pegue todos os pares com `p_cal ≈ 0,70`. Se cerca de 70% deles forem a mesma pessoa, o sistema está calibrado nesse ponto. Calibração perfeita é isso em todos os pontos: $P(\text{match}\mid p_{cal}=q)=q$. A `nota_final` não tem essa propriedade: nota 8 não significa 80%. É um placar, sem unidade de probabilidade. Na epidemiologia, é a exigência dos escores de risco clínico: entre os pacientes com risco previsto de 10%, cerca de 10% adoecem.

**9.1 (md, r-string)**
> ### 9.1 Do placar à probabilidade em quatro passos
>
> **🧒 Por que não uma reta direto?** Com `p = 0,1 × nota`, a nota 12 daria 120%. Precisamos de uma conta que aceite qualquer número e devolva sempre algo entre 0 e 1. O truque é passar por um mundo onde retas são permitidas: o dos **log-odds**.
>
> **📐 Passo 1: de chance para *odds*.** odds = p ÷ (1 − p). p = 0,8 dá 4 ("4 para 1"); p = 0,5 dá 1; p = 0,2 dá 0,25.
>
> **Passo 2: logaritmo.** ln 4 ≈ +1,39; ln 1 = 0; ln 0,25 ≈ −1,39. A escala fica simétrica e sem limites.
>
> **Passo 3: uma reta no mundo dos log-odds.** $z = a\,s + b$. Aqui `a` é a **inclinação** e `b` é o **intercepto**. Para a epidemiologia: $e^{a}$ é uma **razão de chances por ponto de nota**. Com a = 0,85, cada ponto a mais multiplica as chances de match por $e^{0{,}85}\approx 2{,}34$.
>
> **Passo 4: voltar com a sigmoide.** $p=\sigma(z)=\dfrac{1}{1+e^{-z}}$. Com valores redondos parecidos com os do nosso mundo sintético (a = 0,85, b = −5,1) e nota 6,5: z = 0,85 × 6,5 − 5,1 = 0,425; $e^{-0{,}425}\approx 0{,}654$; p = 1 ÷ 1,654 ≈ **0,60**.
>
> **Bônus: a nota de indiferença.** p = 0,5 quando z = 0, ou seja, $s^\star=-b/a$ = 5,1 ÷ 0,85 = **6,0**.
>
> > Convenção: `a` = `slope`, `b` = `intercept`, a mesma da config (`sigmoid(a*logit(p_base)+b)`). Em R: `glm(TARGET ~ nota_final, family = binomial)`.

**9.2 (md, r-string)**
> ### 9.2 Como a máquina escolhe `a` e `b`: a multa por surpresa
>
> **🧒** Para cada par, o modelo paga uma multa proporcional ao susto que leva ao ver a resposta. Disse 90% e era match: susto pequeno. Disse 10% e era match: susto enorme. Ajustar é achar o `a` e o `b` com a **menor soma de multas**.
>
> **📐 A multa de um par (log-loss):** $\ell_i=-\big[y_i\ln p_i+(1-y_i)\ln(1-p_i)\big]$. Com y = 1:
>
> | p | multa = −ln p |
> |---|---|
> | 0,9 | 0,105 |
> | 0,5 | 0,693 |
> | 0,1 | 2,303 |
>
> Errar com confiança custa **22 vezes** mais do que acertar com confiança.
>
> **A multa por exagero (L2):** $\mathcal L(a,b)=\sum_i \ell_i+\tfrac12\lambda a^2$, com λ = 0,001. Se os dados fossem perfeitamente separáveis, a melhor curva seria um degrau, com `a` indo ao infinito; o L2 é o freio. Ele **não** multa o `b`, porque `b` carrega a prevalência (a taxa-base de matches).

(Evidência: calibration.py:81, 88, 92.)

**9.3 (md, r-string)**
> ### 9.3 O passo de Newton, sem medo
>
> **🧒 A bola na tigela.** A soma das multas forma uma tigela, e no fundo estão os melhores `a` e `b`. De olhos vendados, sentimos duas coisas: o **gradiente** `g` (para que lado desce e quão inclinado é) e a **curvatura** `H` (quão fechada é a tigela). Newton imagina uma tigela perfeita com essa inclinação e essa curvatura e **pula direto para o fundo dela**.
>
> **📐 Em uma dimensão.** f(x) = (x − 3)², em x = 0: g = 2(x − 3) = −6; h = 2. Pulo δ = g ÷ h = −3; novo ponto 0 − (−3) = **3**, o fundo exato em um passo.
>
> **Em duas dimensões, com uma tabela 2 × 2:**
> 1. $p_i=\sigma(b+a s_i)$ e $w_i=p_i(1-p_i)$;
> 2. $g_b=\sum(p_i-y_i)$; $g_a=\sum(p_i-y_i)s_i+\lambda a$;
> 3. $H=\begin{pmatrix}\sum w_i & \sum w_i s_i\\ \sum w_i s_i & \sum w_i s_i^2+\lambda\end{pmatrix}$;
> 4. resolver $H\delta=g$ (o "δ = H⁻¹g": dividir o gradiente pela curvatura, versão matriz);
> 5. $b\leftarrow b-\delta_b$, $a\leftarrow a-\delta_a$; parar quando o pulo for menor que 10⁻¹⁰.
>
> **Exemplo de bolso.** s = (2, 4, 6, 8), y = (0, 0, 1, 1). Partida: a = 0, b = log-odds de (2 + 0,5) ÷ (4 + 1) = 0,5, logo b = 0.
> 1. Todo p = 0,5 e todo w = 0,25.
> 2. $g_b$ = 0; $g_a$ = 0,5·2 + 0,5·4 − 0,5·6 − 0,5·8 = **−4**.
> 3. H = [[1; 5], [5; 30,001]].
> 4. Resolvendo: $\delta_a$ = −4 ÷ 5,001 ≈ −0,800; $\delta_b$ = −5 $\delta_a$ ≈ +4,000.
> 5. Novo ponto: **a ≈ 0,80, b ≈ −4,00**; indiferença 4,00 ÷ 0,80 = **5,0**, exatamente entre o último não-match (4) e o primeiro match (6).
>
> Como este exemplo é separável, se deixarmos o Newton continuar a inclinação cresce muito, e só o L2 segura (a tabela abaixo mostra). Depois rodamos o mesmo traço nos dados reais e conferimos que o último passo bate com `fit_platt`.

(Números conferidos: δ = (3.9992, −0.79984). Evidência: calibration.py:77–108.)

**9.5 (md, r-string)**
> ### 9.5 O boletim da calibração: ECE, Brier e o *reliability diagram*
>
> **🧒** Separamos as previsões em 10 gavetas pela confiança anunciada. Em cada gaveta comparamos a **promessa média** com o que **aconteceu**. ECE é a distância média entre promessa e realidade, pesada pelo tamanho da gaveta. Brier é o erro quadrático médio entre probabilidade e resultado.
>
> **📐 Exemplo com 5 pares e 2 gavetas** (o mesmo dos testes automáticos). p = (0,1; 0,2; 0,4; 0,6; 0,9); y = (0; 0; 1; 1; 1).
> *ECE:*
> 1. Gaveta [0; 0,5): promessa média 0,233, aconteceu 1/3 = 0,333, distância 0,100.
> 2. Gaveta [0,5; 1]: promessa 0,75, aconteceu 1,0, distância 0,25.
> 3. (3/5) × 0,100 + (2/5) × 0,25 = 0,06 + 0,10 = **0,16**.
>
> *Brier:* 0,01 + 0,04 + 0,36 + 0,16 + 0,01 = 0,58; 0,58 ÷ 5 = **0,116**.
>
> $$\text{ECE}=\sum_g\frac{n_g}{N}\big|\bar y_g-\bar p_g\big|\qquad \text{Brier}=\frac1N\sum_i(p_i-y_i)^2$$
>
> Régua: quem sempre diz a prevalência f tem Brier = f(1 − f), que dá 0,25 com f = 0,5.
>
> **Como ler o diagrama.** Bolinha na diagonal: promessa cumprida. **Abaixo** da diagonal: prometeu mais do que entregou (excesso de confiança). **Acima**: entregou mais do que prometeu (timidez). Bolinha pequena tem poucos pares e balança muito. O ECE é a média ponderada das distâncias verticais até a diagonal: o espírito de Hosmer-Lemeshow num número só.

(Evidência: test_nb_helpers.py:43–50; nb_helpers.py:113–114.)

**9.6 🎛️ (md)**
> Os controles começam no Platt aprendido no treino. À esquerda: a sua curva, a aprendida (tracejada), a verdade `p*` (pontilhada) e a frequência real de matches por faixa de nota. À direita: o boletim da **sua** curva no teste, com ECE, Brier e log-loss. Teste: (a) dobre a inclinação: para onde vão as bolinhas? (b) mova só o intercepto: para onde vai −b/a? (c) se achar uma curva com ECE menor que a aprendida, lembre que você a escolheu olhando o teste, o mesmo pecado da rota A.

**9.7 (md), honestidade (i)**
> ### 9.7 A prova com a verdade escondida, e o seu limite
>
> Conhecemos a chance verdadeira `p*` de cada par. O Platt, vendo só notas e rótulos do treino, redescobre `p*` no teste?
>
> > ⚠️ **Leia esta prova com a lupa certa.** O gerador cria a verdade **exatamente no formato que o Platt usa**: p* = σ(0,85 × (nota − s₀)), uma sigmoide de uma reta na nota (`synthetic_data.py`, linhas 333 a 335). A resposta certa está garantidamente "no cardápio". Recuperar `p*` prova que o ajuste e o código estão corretos e que cerca de 70% dos pares bastam para o treino. **Não** prova que o Platt funcionaria em dados reais, onde a curva verdadeira pode ser assimétrica, ter platôs ou depender de outras variáveis. Em linguagem epidemiológica: é especificação correta por construção, não validação externa.

O print de C46 (L973, "provam recuperação") passa a "confirmam que o ajuste funciona quando o modelo está bem especificado".

### B.3 Autocalibração por âncoras (nova)

**9.8 (md)**
> ### 9.8 Autocalibração por âncoras: calibrar sem gabarito (demonstração)
>
> > 🧒 **Como para uma criança de 11 anos.** Calibrar uma balança sem pesos-padrão: você só sabe que "o saco de 5 kg deveria marcar 5" e que "a balança vazia deveria marcar 0". Com esses dois pontos de referência, ajusta o ponteiro. As **âncoras** fazem isso com probabilidades: "nota ≥ 9 é quase sempre a mesma pessoa (~98%)" e "nota < 5 quase nunca é (~0,1%)". Com dois pontos, ajustamos a curva **sem rótulos**.
>
> | item | valor na config |
> |---|---|
> | âncora positiva | `nota_final >= 9.0`, média-alvo 0,98, mínimo 50 pares |
> | âncora negativa | `nota_final < 5.0`, média-alvo 0,001, mínimo 200 pares |
> | método | `anchor_platt`, `by_band: true` |
> | trava por janela | mudança máxima de 0,5 em `a` e 0,5 em `b` |
>
> > ⚠️ **Honestidade em primeiro lugar.** A biblioteca **não implementa** `anchor_platt`. O que roda em `run_v3` é o Platt supervisionado, que exige rótulos: `fit_platt` recusa alvos que não sejam 0 ou 1, e o carregador de config nem lê o bloco `anchors`. O que segue é uma **mini-implementação didática** em `notebooks/nb_helpers.py`, para mostrar a **intenção** da config, não o comportamento da biblioteca. Três simplificações declaradas: (1) a config calibra `logit(p_base)` de um modelo-base; aqui usamos a `nota_final`; (2) o ajuste é **global**, porque `by_band` não funciona literalmente: as âncoras moram só nas bandas `low` (< 5) e `high` (≥ 9), e as bandas cinzentas não têm **nenhum** par-âncora; (3) as janelas são simuladas. Como o ajuste não usa rótulos, podemos usar todos os pares sem vazamento de rótulo; avaliamos no mesmo teste da rota B.

**9.8b (md, r-string)**
> #### 📐 A conta: o mesmo Platt, com "rótulos de confiança"
>
> Só os pares-âncora entram, e cada um recebe um **rótulo suave** t: 0,98 nas positivas e 0,001 nas negativas. O resto é idêntico a 9.2 e 9.3.
> 1. **Escolher as âncoras:** nota ≥ 9,0 (positivas) e nota < 5,0 (negativas). Os pares entre 5 e 9 ficam de fora.
> 2. **Conferir o mínimo:** com menos de 50 positivas ou menos de 200 negativas, não ajustamos e mantemos a curva anterior ("congelar").
> 3. **Equilibrar:** peso $w=N/(2n_{+})$ para cada positiva e $N/(2n_{-})$ para cada negativa, para que os dois pontos de referência puxem com a mesma força.
> 4. **Multa com rótulo suave:** $\mathcal L(a,b)=-\sum_{i\in\text{âncoras}}w_i\big[t_i\ln p_i+(1-t_i)\ln(1-p_i)\big]+\tfrac12\lambda a^2$.
> 5. **Newton igual ao de antes:** troque y por t e multiplique cada termo por $w_i$.
>
> **Por que rótulo suave?** Com 0 e 1, as âncoras ficariam perfeitamente separadas (toda positiva tem nota maior que toda negativa) e a curva viraria degrau. Com 0,98 e 0,001, a melhor curva é finita.
>
> **Conta de guardanapo (é também o ponto de partida do Newton).** Se cada grupo estivesse todo na sua nota média, bastariam duas equações:
> - logit(0,98) = ln 49 ≈ **3,89** = a·s̄₊ + b
> - logit(0,001) ≈ **−6,91** = a·s̄₋ + b
>
> Com médias **ilustrativas** s̄₊ = 9,5 e s̄₋ = 3,0: a = 10,80 ÷ 6,5 ≈ **1,66**; b = 3,89 − 1,66 × 9,5 ≈ **−11,9**; indiferença ≈ **7,2**. Contra a verdade sintética (a* = 0,85, indiferença ≈ 6,0), a curva ancorada sai **duas vezes mais íngreme e deslocada para a direita**. No herói (nota 6,5): σ(−1,1) ≈ 0,25, contra p* ≈ 0,60.
>
> **Por que a diferença?** Âncoras só são boas se os alvos forem verdadeiros. A tese que inspirou a config observou ~98% e ~0% nas pontas de dados **reais**; no nosso mundo sintético a fração de matches com nota < 5 é bem maior que 0,1% (o código mede). Âncora errada não avisa: ela calibra com confiança para o lugar errado.

**🔒 Trava (md)**
> **🧒 O volante com limitador.** A cada janela a calibração seria refeita; uma semana esquisita poderia dar um tranco. A config limita: de uma janela para a outra, `a` e `b` mudam no máximo 0,5 cada.
>
> $$a_{\text{usado}}=a_{\text{ant}}+\operatorname{clip}(a_{\text{novo}}-a_{\text{ant}},\,-0{,}5,\,+0{,}5)$$ (idem para b).
>
> Exemplo: do supervisionado (0,85; −5,1) para o ancorado pedido (1,66; −11,9):
> 1. Δa = 0,81 → +0,5 → a = 1,35
> 2. Δb = −6,8 → −0,5 → b = −5,6
> 3. indiferença 5,6 ÷ 1,35 ≈ **4,1**
>
> > 🔎 **Efeito colateral.** A curva antiga tinha indiferença em 6,0 e a pedida em 7,2; travando `a` e `b` separadamente, a intermediária foi para **4,1**, onde nenhuma das duas estava. E 0,5 é muito para `a` (quase dobra a inclinação) e pouco para `b` (que precisa andar 6,8). A simulação mostra várias janelas, incluindo uma "semana fraca" com poucas âncoras, em que a calibração congela.

Dados das janelas (no notebook): `[synthetic_data.generate_comparador(seed=SEED+k, n_pairs=n).frame["nota final"].to_numpy() for k, n in enumerate([600, 600, 300, 600, 600, 600], 1)]`. A janela de 300 pares deve cair abaixo do min_count negativo e congelar. `init=platt_holdout`, com a história "no mês passado tínhamos rótulos; daqui em diante não".

### B.4 §11 triagem por custo esperado

**11.0 (md)**
> ## 11. Política de decisão (triagem)
>
> > 🧒 **Como para uma criança de 11 anos.** Levar guarda-chuva custa pouco (carregar peso). Não levar e chover custa muito (molhar a mochila). Com 30% de chance de chuva, vale levar? Depende de **quanto** cada erro dói. A triagem faz essa conta para cada par, com três opções: `MATCH`, `NONMATCH` ou pedir uma segunda opinião (`LLM_REVIEW`), que custa um pouquinho e erra menos.

**11.1 (md, r-string)**
> ### 11.1 As três contas, com o nosso herói
>
> **Dizer MATCH:** só pago se o par não era match: $\text{perda}_{\text{match}}=(1-p)\,c_{fp}$.
>
> **Dizer NONMATCH:** só pago se era match: $\text{perda}_{\text{non}}=p\,c_{fn}$.
>
> **Pedir revisão:** pago a taxa $c_{llm}$ mais o erro que sobra depois da revisão (taxas por banda): $\text{perda}_{\text{llm}}=c_{llm}+(1-p)\,e_{fp}\,c_{fp}+p\,e_{fn}\,c_{fn}$.
>
> Exemplo com p = 0,60, banda `grey_mid` (e_fp = 0,06; e_fn = 0,10) e c_llm = 0,25. O valor exato do herói é impresso pelo código.
>
> | conta | vigilância (10/50) | confirmação (100/20) |
> |---|---|---|
> | MATCH | 0,40 × 10 = **4,00** | 0,40 × 100 = 40,00 |
> | NONMATCH | 0,60 × 50 = 30,00 | 0,60 × 20 = **12,00** |
> | melhor automática | MATCH, 4,00 | NONMATCH, 12,00 |
> | revisão | 0,25 + 0,24 + 3,00 = **3,49** | 0,25 + 2,40 + 1,20 = **3,85** |
> | EVR = base − revisão | **+0,51** | **+8,15** |
>
> Com a **mesma** probabilidade, a melhor decisão automática **inverte** entre os modos, e nos dois a revisão compensa.

**11.2 (md, r-string)**
> ### 11.2 O ponto de virada, e por que a revisão só "paga" no meio
>
> **🧒 A gangorra.** Conforme p cresce, o prato "dizer MATCH" fica mais leve e o prato "dizer NONMATCH" mais pesado. Em algum p a gangorra fica reta.
>
> **📐 Passo a passo:**
> 1. (1 − p) c_fp = p c_fn
> 2. c_fp − p c_fp = p c_fn
> 3. c_fp = p (c_fp + c_fn)
> 4. $p^\star=\dfrac{c_{fp}}{c_{fp}+c_{fn}}$
>
> **Vigilância:** 10 ÷ 60 ≈ **0,167**. Basta 17% de chance para compensar dizer MATCH, porque perder um match dói 5 vezes mais. **Confirmação:** 100 ÷ 120 ≈ **0,833**. O "0,5" só aparece quando c_fp = c_fn, e tudo isto **só vale se p for calibrada**: é aqui que a seção 9 paga a conta.
>
> **A janela da revisão.** As três perdas são retas em p, então a revisão vence nas duas comparações dentro de um intervalo:
> $$p_{\inf}=\frac{c_{llm}+e_{fp}c_{fp}}{c_{fn}(1-e_{fn})+e_{fp}c_{fp}}\qquad p_{\sup}=\frac{c_{fp}(1-e_{fp})-c_{llm}}{c_{fp}(1-e_{fp})+e_{fn}c_{fn}}$$
> - Em `grey_mid`, vigilância: 0,85 ÷ 45,6 ≈ **0,019** e 9,15 ÷ 14,4 ≈ **0,635**.
> - Em `grey_mid`, confirmação: 6,25 ÷ 24 ≈ **0,260** e 93,75 ÷ 96 ≈ **0,977**.
>
> **Por que só no meio.** Nas pontas, a melhor decisão automática já é quase de graça: com p = 0,02, dizer NONMATCH custa 0,02 × 50 = 1,0, enquanto revisar custa 1,38 na banda `low`. No meio, a decisão automática é cara e a revisão a barateia. A tabela abaixo calcula tudo isso para todas as bandas, direto da config.

**11.3 (md)**
> ### 11.3 Travas, orçamento e fila
>
> **🧒 A fila do posto de saúde.** Há mais pedidos que fichas. Primeiro os que a regra **manda** atender (`FORCE_REVIEW`), depois os que caíram numa **trava**, por último os que **valem a pena** (EVR > 0). Dentro de cada grupo, atende primeiro quem tem maior EVR. Acabaram as fichas, acabou a revisão.
>
> **As travas só pedem; não decidem.** Em vigilância, um MATCH automático com p < 0,85 e um NONMATCH automático com p > 0,15 são marcados para revisão (confirmação: 0,95 e 0,10). A trava não muda a decisão base nem a probabilidade; sem orçamento, o par volta para a decisão automática. E uma trava pode pedir revisão **mesmo com EVR negativo**: ela compra segurança, não economia.
>
> **Ordem no código:** (1) `review_requested = EVR > 0 ou trava ou FORCE_REVIEW`; (2) pares com `ALWAYS_*` nunca entram na fila; (3) seleciona `FORCE_REVIEW`, depois travas, depois EVR > 0, cada grupo por EVR decrescente, até esgotar o orçamento; (4) quem foi selecionado vira `LLM_REVIEW`.
>
> > 🔎 Neste dataset o orçamento nunca aperta (2000 e 1000 revisões para cerca de 600 pares). O controle 11.6 deixa você apertá-lo.

(Evidência: policy_engine.py:101–154.)

---

## C. Controles interativos

**Padrão comum**, sempre dentro de **uma** célula de código (mantém a regra md-antes-de-code):

```python
def _ui_x(**kw):                      # wrapper fino, no notebook
    fig, resumo = nbh.plot_x(..., **kw); plt.show(); display(resumo)
_ui_x(**DEFAULTS_DA_CONFIG)           # visão estática: sempre roda, inclusive headless
nbh.interagir(_ui_x, {...controles...}) # retorna False e imprime nota se não houver frontend de widgets
```

Cola de UI em `nb_helpers`:
- `Deslizante(valor, minimo, maximo, passo, descricao, inteiro=False, log=False)`, `Escolha(opcoes, valor, descricao)` e `Caixa(valor, descricao)`, dataclasses congeladas.
- `interagir(func, controles, *, fixos=None, presets=None, preset_de=None) -> bool`. Importa `ipywidgets` preguiçosamente dentro de `try`; no Colab chama `google.colab.output.enable_custom_widget_manager()` num `try` aninhado; `continuous_update=False`; `presets` reajusta os sliders quando um dropdown de modo muda; qualquer exceção imprime `[modo não interativo: …]` e retorna False.

As funções `plot_*` retornam `(Figure, resumo)`, nunca chamam `plt.show()`, e sempre definem título e rótulos x/y em pt-BR (mantém a garantia TST4.1.a; ver E). Células wrapper não devem conter `.plot(`.

| # | Função em nb_helpers | Controles (default ← fonte; faixa) | O que desenha | Fallback estático |
|---|---|---|---|---|
| W1 (§9.6) | `plot_platt_explorer(nota, y, model, *, ref_model, p_true, n_bins=10)` | `inclinacao` ← `platt_holdout.slope`, [0, max(3, 3â)] passo 0.01; `intercepto` ← `platt_holdout.intercept`, [b̂−8, b̂+8] passo 0.05 | Esq.: taxa empírica de match por faixa de nota (pontos com tamanho por n), curva do usuário, curva ajustada (tracejada), p* (pontilhada), linha vertical em s* = −b/a. Dir.: `plot_reliability` no teste. Título: ECE, Brier, log-loss. | `_ui_platt(platt_holdout.slope, platt_holdout.intercept)` |
| W2 (§11) | `plot_costs_explorer(df, cfg, *, mode, c_fp, c_fn, c_llm, band)` | `modo` Escolha (vigilancia/confirmacao, presets de `cfg.decision_policy.modes[m]`); `c_fp`, `c_fn` [1, 200] passo 1; `c_llm` [0, 5] passo 0.05; `banda` Escolha ← nomes de `cfg.bands`, default `grey_mid` | Esq.: as três retas de perda em p∈[0,1] para a banda, janela de revisão sombreada, ponto de virada vertical, travas pontilhadas. Dir.: contagem de ações, ajuste do usuário vs baseline da config (`PolicyEngineV3` real, `Budget` novo a cada chamada). | Chamada com os valores de vigilância do cfg |
| W3 (§9.8) | `plot_anchor_comparison(nota_all, nota_test, y_test, sup_model, fit, spec, *, p_true_all, p_true_test)` + `fit_anchor_platt` | `corte_pos` ← `ANCORAS.pos_cut` (9.0), [7, 10]; `corte_neg` ← 5.0, [1, 7]; `alvo_pos` ← 0.98, [0.5, 0.999]; `alvo_neg` FloatLogSlider expoente [−4, −0.3] ← log10(0.001); `balancear` Caixa True; `exigir_minimo` Caixa True | Esq.: supervisionado vs ancorado vs p*, regiões-âncora sombreadas, quadrados nos alvos, anotado com o **p* médio verdadeiro** em cada região. Dir.: dois reliability diagrams no teste. Tabela: n_pos/n_neg vs min_count, status, a, b, s*, ECE, Brier, MAE vs p*. Se `corte_neg ≥ corte_pos`, mensagem no lugar do gráfico. | Chamada com os valores da config |
| W4 (§10) | `plot_bands_guardrails(df, cfg, *, edges, guard_params, mode="vigilancia")` | 5 fronteiras `fronteira_1..5` ← `cfg.bands.definitions` (5–9), [0, 10] passo 0.1; `nota_always_match` ← `cfg.guardrails` 10.0, [8, 10.5]; `nota_always_nonmatch` ← 3.0, [0, 6]; `homonimia_min_nota` ← 7.0, [5, 10]; `homonimia_year_gap` ← 5, [0, 20] int | Histograma com fronteiras antigas (tracejadas) e novas; heatmap de transição banda antiga→nova com contagens; contagem de guardrails baseline vs novo. Imprime quantos pares mudaram de banda, regra e ação (re-triagem, vigilância). | Chamada com a config; o md convida a testar `nota_always_match = 9.0` |
| W5 (§7.4) | `plot_feature_explorer(df, *, w_qtd, w_prim, w_ult, very_close_days, close_days, far_years)` | pesos ← `NOME_PESOS_LOADER` (0.5/0.25/0.25), [0, 1] passo 0.05; dias ← `MACD_JANELAS_LOADER` (7, 30) int [0, 60] e [1, 365]; `far_years` ← 5, [1, 20] | Histograma de `nome_score` por TARGET com AUC (novo vs default); taxas dos flags por TARGET; histograma de \|dias\| por TARGET com eixo x log e linhas das janelas (torna F2 visível) | Chamada com os defaults do loader; um `assert` no notebook mostra que reproduzem as colunas do loader |
| W6 (§11.6) | `plot_budget_explorer(df, cfg, *, mode, llm_max)` + `review_status` | `modo` Escolha; `orcamento` IntSlider [0, n_pedidos(mode)], default `min(orçamento cfg, n_pedidos)`; imprime o valor da config | Dispersão p_cal vs EVR, cores: revisado / pediu mas ficou de fora / não pediu / guardrail; marcador por prioridade (force X, trava ▲, evr ●); linha em EVR=0. Barras por prioridade: revisados vs de fora. Título com o último EVR revisado e o primeiro de fora. | Chamada no orçamento da config ("ninguém fica de fora") e depois uma segunda estática em 25% dos pedidos, para o headless mostrar uma fila real |

Defaults dos pesos e janelas MACD são **constantes do loader**, não da config (loader.py:211–213, 278–283). `nb_helpers` os espelha como constantes nomeadas e um teste unitário confere que batem com o loader. O texto diz isso abertamente.

---

## D. Correções de honestidade

| Fix | Onde | Mudança |
|---|---|---|
| (i) recuperação de p* por construção | md §9.7 + print de C46 | Rascunho ⚠️ em B.2; suavizar o print de L973. |
| (ii) "maioria dos IDs única" | C86, C88, C105 (L2073–2076); docstring em test_notebook_pipeline.py:446–451 | Remover a afirmação. O código imprime as frações (medição de C87 em L1662–1668 já existe; manter) + `Δmax Fβ`. O md seguinte explica a independência (F10). |
| (iii) custos hard-coded | C92 `mode_costs_32` (L1770–1773); C103 `CUSTO_FP` (L1956, sai com a §15); números da §11 | Montar `mode_costs` de `cfg.decision_policy.modes[m].false_positive_cost / false_negative_cost / min_auto_match_threshold`. Acrescentar a linha do ponto de virada `c_fp/(c_fp+c_fn)`. Renomear "limiar política" para "trava min_auto_match". Números da §11 também vêm de `nbh.thresholds_table(cfg)`. |
| (iv) Fβ | intro da §13 | "**Qual F?** `auto_fbeta` usa β = 2 em vigilância e β = 0,5 em confirmação. $F_\beta=\frac{(1+\beta^2)PR}{\beta^2P+R}$; β = 2 faz o recall pesar 4 vezes mais. Exemplo P = 0,8, R = 0,6: F₂ ≈ 0,63; F₀,₅ = 0,75. Comparar `auto_fbeta` entre modos mistura duas réguas." (eval.py:42–48; metrics.py:72–78). Re-rotular as barras de C85. |
| (v) guardrails config vs código | md §10 | "> ⚠️ **Config × código.** O YAML lista `always_match: nota_final >= 9.0` e `always_nonmatch: nota_final <= 3.0`, mas o código usa os `params`: `ALWAYS_MATCH` exige nota ≥ **10,0** e nome, data e município perfeitos; `ALWAYS_NONMATCH` usa nota **< 3,0**. As regras `risky_if` (`grey_mother_missing`, `low_name_and_low_dob`) **não estão implementadas**. No controle 10.2, baixe o corte para 9,0 e veja quantos pares mudariam." (YAML:124–145; guardrails.py:99–104, 151–163; config.py:232–238) |

Limitações da §16: trocar o bullet "quase todos singletons" por F10; acrescentar "verdade logística por construção"; acrescentar "âncoras: demonstração no notebook, alvos da tese ≠ verdade sintética"; manter o bullet do stub de LLM.

---

## E. Impacto nos testes

**Restrições que precisam continuar valendo:**
1. `test_notebook_didatica.py`: toda célula de código não vazia vem **logo depois** de uma célula markdown não vazia (L40–73; `EXCECOES_ORFAS` vazio em L33), e a primeira célula de código é precedida por markdown (L76–86). Chamada estática e interativa na mesma célula; nunca duas células de código adjacentes.
2. `test_notebook_execution.py`:
   - Estas 20 substrings precisam aparecer no markdown (L34–55): `# GZ-CMD++ v3`, `## Glossário`, `## 1. Contexto`, `## 2. Visão geral do pipeline`, `## 3. Objetivos`, `## 4. Setup`, `## 5. Primeiro olhar nos dados`, `exemplo-fio-condutor`, `## 7. Carga dos dados e`, `feature engineering`, `## 8. Atribuição de bandas`, `## 9. Calibração`, `boletim da calibração`, `## 10. Guardrails`, `## 11. Política de decisão`, `## 12. Reconciliação`, `## 13. Avaliação held-out`, `## 14. Revisão LLM`, `## 15. Painel interativo`, `## 16. Conclusões e limitações`.
   - Célula cujo source contenha `subplots(`, `plt.figure(` ou `.plot(` precisa de título e rótulos x/y na mesma célula (L60–88). Wrappers não podem conter `.plot(`.
   - Nenhuma célula de código vazia (L91–99).
   - Execução completa via nbclient, 600 s por célula, cwd em `REPO_ROOT` (L102–113).
3. `_nb_paths.py`:17 fixa caminho e nome do notebook.
4. `test_nb_helpers.py`: comportamento e exceções existentes de ECE, Brier e stub (L15–165). Refatorar ECE via `reliability_points` precisa manter `digitize(right=False)` nas bordas internas, o clipping e pular gavetas vazias (nb_helpers.py:92–103).
5. `test_scaffold_imports.py`:19–22: os três nomes existentes ficam; `import nb_helpers` funciona sem matplotlib nem ipywidgets.
6. `test_notebook_pipeline.py` usa o **CSV versionado** (L26–35), que o notebook reescreve (build_notebook.py:273–279). Manter a chamada do gerador exatamente igual; `git diff data/` deve ficar vazio.
7. Nenhum teste checa contagem de células, tags ou saídas; cortar células é seguro.

**Testes a acrescentar/atualizar:**
- `test_notebook_execution`: acrescentar `"Autocalibração por âncoras"` e `"Fβ"` à lista esperada.
- `test_notebook_didatica`, novos:
  - `test_sem_saidas_armazenadas`: toda célula de código com `outputs == []` e `execution_count is None`.
  - `test_builder_em_sincronia`: sources de `build_notebook.build()` iguais aos do `.ipynb`.
  - `test_seis_controles`: exatamente 6 ocorrências de `nbh.interagir(`.
  - `test_ipywidgets_nao_importado_no_notebook`: nenhum `import ipywidgets` / `from ipywidgets` nas células.
  - `test_custos_nao_hardcoded`: regex `CUSTO_FP\s*,\s*CUSTO_FN\s*=\s*\d` e `"false_positive"\s*:\s*\d` não podem casar.
  - `test_sem_travessao_no_markdown`: nenhum "—" em células markdown.
- Novo `tests/test_nb_helpers_didatica.py`:
  - `reliability_points` dá conf (0.2333, 0.75), acc (1/3, 1) e n (3, 2) no exemplo de 5 pares; ECE continua 0.16.
  - `log_loss([1], [0.9])` ≈ 0.10536.
  - Traço de Newton: primeiro passo do exemplo de bolso dá b ≈ −3.99920, a ≈ 0.79984 (atol 1e-6); a última linha bate com `fit_platt` nos dados sintéticos (atol 1e-10).
  - `load_anchor_spec(yaml do pacote)` retorna (>=, 9.0, 0.98, 50, <, 5.0, 0.001, 200, 0.5, 0.5).
  - Chute de dois pontos em grupos constantes (9.5 / 3.0) é exato.
  - `fit_anchor_platt` com l2=0 em dois grupos constantes: médias ajustadas iguais aos alvos (1e-6). Falha de min_count retorna status `"congelado"` e o modelo de fallback.
  - `clamp_update((0.85, −5.1) → (1.66, −11.9))` retorna (1.35, −5.6, True, True).
  - `simulate_anchor_windows`: uma janela fraca congela.
  - `expected_losses` bate com `base_loss`, `loss_llm` e `evr` do engine.
  - `decision_thresholds` para grey_mid: vigilância 1/6, 0.85/45.6, 9.15/14.4; confirmação 100/120, 6.25/24, 93.75/96.
  - Invariantes de `review_status` contra o engine real para `llm_max` ∈ {0, 5, 20, grande}: revisado ⇔ `LLM_REVIEW`; nenhuma prioridade menor revisada enquanto uma maior fica de fora; dentro de uma prioridade, o menor EVR revisado ≥ o maior EVR de fora.
  - `nome_score` e `macd_flags` nos defaults iguais às colunas do loader no CSV versionado.
  - `auc_separacao` dá 1 para perfeito, 0 para invertido, 0.5 para empates.
  - `assign_bands_with_edges` com as fronteiras da config igual a `BandAssigner`; fronteiras não crescentes levantam `ValueError`.
  - `guardrail_diff` com parâmetros idênticos dá zero mudanças.
  - `interagir` com `sys.modules["ipywidgets"]=None` retorna False e imprime.
  - Todo `plot_*` nos defaults (backend Agg, `importorskip("matplotlib")`) tem título e rótulos x/y não vazios em todo eixo.
- `test_scaffold_imports`: acrescentar os novos nomes públicos.
- `test_notebook_pipeline.py`:446–451: corrigir só a docstring.

---

## F. Fatias de implementação (sequenciais)

Toda fatia termina com `python notebooks/build_notebook.py` (quando mexe no builder), `pytest -m "not notebook"`, `pytest -m notebook` (fatias 3–6) e `ruff check notebooks tests`. `git diff data/` vazio.

| Fatia | Arquivos | ALL_PHASES | Aceite |
|---|---|---|---|
| 1. Helpers: núcleo de cálculo + cola de UI | `notebooks/nb_helpers.py`, `tests/test_nb_helpers_didatica.py`, `tests/test_scaffold_imports.py` | nenhum | Tudo do apêndice exceto `plot_*`, com imports opcionais preguiçosos; testes novos passam; os 19 testes de helper existentes intactos e passando. |
| 2. Helpers: figuras | `nb_helpers.py` (`plot_reliability`, 6 `plot_*`, `plot_anchor_windows`), testes | nenhum | Testes de figura passam; nenhum `plt.show`; cada plot < 1 s em 607 linhas. |
| 3. §1–§8 | `build_notebook.py` | `FASE_2_1`, `FASE_2_2`, `FASE_2_3` | `cfg`, `CONFIG_PATH`, `ANCORAS` e `PASSAPORTE` existem a partir de §4/§6; W5 dentro; `card_heroi` de C14 e chamadas C25/C34/C50/C58/C72/C100 removidas; ≈28 células; testes de seção passam. |
| 4. §9 | `build_notebook.py` | `FASE_2_4` | Variáveis do contrato mantidas; traço de Newton, W1, W3 e demo da trava dentro; binning inline de C44 removido; honestidade (i) e caixa ⚠️ "não é comportamento da biblioteca" presentes. |
| 5. §10–§11 | `build_notebook.py` | `FASE_2_5`, `FASE_2_6` | Guardrails chamados com `cfg.guardrails.*`; fix (v); `thresholds_table`; W4, W2, W6; `out_vig` e `out_conf` inalterados (§12 precisa continuar batendo exatamente). |
| 6. §12–§16 + testes de nível notebook | `build_notebook.py`, `tests/test_notebook_didatica.py`, `tests/test_notebook_execution.py`, `tests/test_notebook_pipeline.py` (docstring) | `FASE_3_1`, `FASE_3_2`, `FASE_3_3`, `FASE_INTERATIVO`, `FASE_4_1` | Fixes (ii), (iii), (iv); guia da §15; §16 atualizada; testes estáticos novos passam; execução headless completa passa; ~87 células; `.ipynb` sem saídas. |

---

## Apêndice: API proposta de `nb_helpers` (tudo novo, só notebook, nada em `src/`)

```python
# calibração
reliability_points(y_true, p_pred, *, n_bins=10) -> pd.DataFrame        # bin, lo, hi, n, conf, acc, gap
log_loss(y_true, p_pred, *, eps=1e-15) -> float                         # y pode ser suave em [0,1]
platt_newton_trace(nota, y, *, l2=1e-3, max_iter=100, tol=1e-10, weights=None, init=None) -> pd.DataFrame
    # passo, intercepto, inclinacao, g_b, g_a, h_bb, h_ba, h_aa, delta_b, delta_a, perda
    # compartilha um _newton_logistic(x, t, w, l2, ...) privado com fit_anchor_platt
# âncoras
@dataclass(frozen=True) class AnchorSpec: pos_op, pos_cut, pos_target, pos_min_count, neg_op, neg_cut, neg_target, neg_min_count, max_abs_delta_a, max_abs_delta_b
load_anchor_spec(config_path) -> AnchorSpec        # yaml.safe_load; regex r"^\s*nota_final\s*(>=|>|<=|<)\s*([0-9.]+)\s*$"
anchor_masks(nota, spec) -> tuple[np.ndarray, np.ndarray]
anchor_two_point_guess(nota, spec) -> PlattModel
@dataclass(frozen=True) class AnchorFit: model: PlattModel | None, status: str, n_pos: int, n_neg: int, mean_s_pos: float, mean_s_neg: float, fitted_mean_pos: float, fitted_mean_neg: float, n_iter: int
fit_anchor_platt(nota, spec, *, l2=1e-3, max_iter=100, tol=1e-10, balance=True, enforce_min_count=True, fallback=None) -> AnchorFit
clamp_update(prev, new, *, max_abs_delta_a, max_abs_delta_b) -> tuple[PlattModel, bool, bool]
simulate_anchor_windows(windows, spec, *, init, l2=1e-3) -> pd.DataFrame   # janela, n_pos, n_neg, status, a_bruto, b_bruto, a_usado, b_usado, s_estrela, travou_a, travou_b
# política
expected_losses(p, *, c_fp, c_fn, c_llm, e_fp, e_fn) -> pd.DataFrame
decision_thresholds(*, c_fp, c_fn, c_llm, e_fp, e_fn) -> dict[str, float]  # virada, llm_inf, llm_sup (NaN se vazio)
thresholds_table(cfg) -> pd.DataFrame                                      # modo × banda
build_engine(cfg, mode, *, c_fp=None, c_fn=None, c_llm=None, llm_max=None) -> PolicyEngineV3  # Budget novo a cada chamada
review_status(out, *, min_auto_match, max_auto_nonmatch) -> pd.DataFrame   # prioridade, status (rotula a escolha do engine real; não a re-prediz)
# features
NOME_PESOS_LOADER = (0.5, 0.25, 0.25); MACD_JANELAS_LOADER = {"very_close_days": 7, "close_days": 30, "far_years": 5.0}
nome_score(df, *, w_qtd, w_prim, w_ult) -> pd.Series
macd_flags(df, *, very_close_days, close_days, far_years) -> pd.DataFrame
auc_separacao(x, y) -> float                     # sklearn.metrics.roc_auc_score (dependência base)
# bandas / guardrails
assign_bands_with_edges(nota, edges, names) -> pd.Series   # monta BandDefinition + BandAssigner; ValueError se não crescente
band_transitions(old, new, order) -> pd.DataFrame
guardrail_diff(df, base_params, new_params) -> pd.DataFrame
# figuras (matplotlib preguiçoso) e UI (ipywidgets preguiçoso): ver C
plot_reliability(...), plot_platt_explorer(...), plot_costs_explorer(...), plot_anchor_comparison(...),
plot_bands_guardrails(...), plot_feature_explorer(...), plot_budget_explorer(...), plot_anchor_windows(...)
Deslizante, Escolha, Caixa, interagir(...)
```
