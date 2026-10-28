# Roteiro de demonstração ao vivo

Notebook: `notebooks/gzcmd_passo_a_passo.ipynb`.
Fonte do conteúdo: `notebooks/build_notebook.py`.
Público: professora e colegas de saúde coletiva, sem formação em estatística ou aprendizado de máquina.
Duração: cerca de 19 minutos, com aproximadamente 17 minutos de fala e 2 de navegação.
Use as falas como apoio. Não leia código nem todas as colunas das tabelas.

## Antes de começar

- Execute o notebook inteiro antes: **Kernel → Restart & Run All**. Aguarde terminar e deixe figuras e tabelas prontas. Ao vivo, role pelas saídas; execute apenas a célula de um controle se precisar reativá-lo.
- No ambiente do kernel, instale `python -m pip install -r requirements/notebook.txt`. Abra em Jupyter com suporte a `ipywidgets` e teste os controles antes da apresentação.
- Tempo total medido nesta máquina, por execução headless das células com `nbclient`: **50,00 segundos**, incluindo início do kernel. A cópia executada em `$env:TEMP` foi apagada depois. Instalação de dependências não entra nesse tempo.
- **Não execute novamente a §13.7 ao vivo.** Tempo total impresso pela seção: **36,06 segundos** nesta execução. Mostre as duas tabelas em “🎯 Resumo da comparação”, já calculadas.
- O notebook fixa CPU nos ajustes de XGBoost e usa sementes fixas para repetir os números. CPU fixa não significa duração fixa: outra máquina pode levar mais ou menos tempo.
- Para resetar um controle, execute novamente **somente sua célula de código**. Os valores iniciais voltam; esses controles exploram cópias e não substituem os resultados principais. Em W6, trocar o modo também repõe suas vagas padrão.

## Roteiro

1. **1. Contexto: por que *record linkage* é difícil?** + **2. Visão geral do pipeline GZ-CMD++ v3** | **1 min 30 s**
   - **Dizer:** “Estas duas Marias podem ser a mesma pessoa com uma data digitada errada. Unir pessoas diferentes é um erro; deixar separada a mesma pessoa é outro, e os dois podem ter preços diferentes.”
   - **Mostrar/clicar:** as duas fichas, o diagrama CSV → decisão e o aviso das rotas A/B. Diga que A confere o código e B confere fichas separadas; pule o glossário.
   - **O que a plateia deve perceber:** nota de semelhança não é probabilidade; conferir o exercício estudado não é fazer uma prova nova.

2. **5. Primeiro olhar nos dados** + **6. Apresentando o "herói": nosso exemplo-fio-condutor** | **45 s**
   - **Dizer:** “Cada linha compara duas fichas, não representa uma pessoa sozinha. Vamos seguir o mesmo par, chamado `zona_cinzenta`, para não trocar de exemplo a cada conta.”
   - **Mostrar/clicar:** uma linha da tabela e as pistas do herói. Aponte o `PASSAPORTE`, que vai guardar as etapas.
   - **O que a plateia deve perceber:** o herói é um caso intermediário e serve para explicar, não para provar desempenho.

3. **7.1 A receita de cada nota, conta por conta** + **7.3 Visualização: distribuição da `nota_final` por classe** | **1 min 15 s**
   - **Dizer:** “Três juízes dão a nota do nome: quantidade de pedaços iguais, primeiro nome e último sobrenome. No herói, os votos somam 0,30 + 0,25 + 0,00 = 0,55; isso ainda não é probabilidade.”
   - **Mostrar/clicar:** a conta conferida com o loader e o histograma. Aponte onde as duas cores se sobrepõem no meio.
   - **O que a plateia deve perceber:** organizar pistas ajuda, mas a mesma nota pode aparecer em pares certos e errados.

4. **8. Atribuição de bandas** | **20 s**
   - **Dizer:** “As bandas são etiquetas para faixas da nota. Etiquetar não decide se as fichas são da mesma pessoa.”
   - **Mostrar/clicar:** nomes das bandas e tabela de fronteiras; não leia cada contagem.
   - **O que a plateia deve perceber:** banda não é decisão nem probabilidade; também ajuda a representar erros da revisão por faixa.

5. **9. Calibração: da nota a uma probabilidade confiável** + **9.1 Do placar à probabilidade em quatro passos** | **1 min 15 s**
   - **Dizer:** “Uma previsão de chuva é confiável quando combina com o que acontece em muitos dias parecidos. Aqui traduzimos a nota para probabilidade: 80% dá odds de 4, e a curva em S mantém a resposta entre impossível e certo.”
   - **Mostrar/clicar:** o exemplo das odds e a explicação da sigmoide. Nomeie inclinação `a` e intercepto `b`; pule os blocos matemáticos e siga para §9.4.
   - **O que a plateia deve perceber:** nota alta não vira porcentagem por simples divisão; a tradução precisa ser aprendida.

6. **9.4 Rotas A e B: duas perguntas, não dois vencedores** + **9.5 O boletim da calibração: promessa, erro e desenho** | **1 min 15 s**
   - **Dizer:** “Não vamos eleger a rota com menor erro, pois as provas são diferentes. No desenho, cada bolinha reúne pares com promessas parecidas; perto da diagonal, promessa e frequência combinam.”
   - **Mostrar/clicar:** tabela das rotas e diagrama de confiabilidade. Aponte ECE do Platt de 0,0751 e distância até p* de 0,0099; até p* tem ECE de 0,0754 neste teste.
   - **O que a plateia deve perceber:** o ECE tem ruído de amostra; uma promessa correta não precisa coincidir exatamente com um pequeno grupo sorteado.

7. **9.6 🎛️ W1: experimente os dois parafusos do Platt** | **1 min 15 s**
   - **Dizer:** “Vou dobrar a inclinação sem mexer no intercepto b: a subida muda e seu meio também muda. Depois volto ao início e movo só o intercepto b para ver outra maneira de deslocar a curva.”
   - **Mostrar/clicar:** em “Inclinação a”, passe de aproximadamente 0,91 para 1,82 e olhe `s*`. Solte o mouse para atualizar; reexecute só a célula para resetar, depois mova “Intercepto b” e compare com a curva tracejada.
   - **O que a plateia deve perceber:** os dois controles mudam a nota que vira 50%; brincar olhando o teste não é validação independente.

8. **9.7 A prova com a verdade escondida, e o seu limite** | **30 s**
   - **Dizer:** “Como inventamos os dados, guardamos a probabilidade verdadeira, chamada p*. O jogo já foi feito para caber no Platt: chegar perto dessa resposta confere este jogo, não uma população real.”
   - **Mostrar/clicar:** pontos próximos da diagonal e distância média de 0,0099. Avise que p* não foi usada para ensinar o ajuste.
   - **O que a plateia deve perceber:** conhecer a regra escondida é uma vantagem da simulação e também um limite da demonstração.

9. **9.8 Autocalibração por âncoras: calibrar sem gabarito (demonstração)** + **🎛️ W3: e se as âncoras estiverem erradas?** | **1 min 45 s**
   - **Dizer:** “Sem respostas conhecidas, usamos grupos que presumimos muito positivos ou muito negativos. Ter pares suficientes não torna essa suposição verdadeira: o alvo precisa combinar com o grupo.”
   - **Mostrar/clicar:** em “9.8.1 Quase certeza e pares suficientes para ajustar”, compare contagens: todas as linhas têm 113 positivas e 242 negativas; só o treino tem 81 e 171, abaixo do mínimo de 200 negativas. Compare alvos 0,98/0,001 com p* médio 0,9501/0,0889.
   - **Mostrar/clicar:** em W3, mantenha cortes e caixas; mude “Alvo positivo” para perto de 0,95 e “Alvo negativo” para perto de 0,09. O negativo é logarítmico: use o passo próximo de 0,10; com positivo 0,95, ECE cai de 0,1558 para 0,0687 e MAE de 0,1214 para 0,0179.
   - **O que a plateia deve perceber:** referências erradas entortam a tradução; ajustar alvos usando p* é uma exploração privilegiada deste sintético, não uma solução sem gabarito para bases reais.

10. **10. Guardrails** + **10.1 🎛️ W4: bandas e regras de segurança** | **45 s**
    - **Dizer:** “Estas regras funcionam como um porteiro: algumas decidem diretamente e outra pede revisão. Baixar a nota de entrada não apaga as outras exigências.”
    - **Mostrar/clicar:** cenários `match_obvio`, `nonmatch_obvio`, `homonimo` e `obito_antes_diag`. Em W4, mude **só** “Nota sempre match” (`nota_always_match`) de 10 para 9; mantenha fronteiras e demais controles.
    - **O que a plateia deve perceber:** nome, nascimento e município ainda são exigidos para `ALWAYS_MATCH`; o controle não muda a rota A original.

11. **11.1 📐 As três perdas: probabilidade de errar × preço do erro** + **11.3 Travas, orçamento e fila** | **1 min 45 s**
    - **Dizer:** “Com probabilidade 0,60, a melhor escolha automática é MATCH na vigilância e NONMATCH na confirmação, porque os preços dos erros mudam. Revisar custa menos nos dois, mas pedir revisão ainda não é receber uma vaga.”
    - **Mostrar/clicar:** tabela ilustrativa: automática 4,00 contra revisão 3,49 na vigilância; automática 12,00 contra revisão 3,85 na confirmação. Não confunda esses valores com os do herói, cuja probabilidade real é 0,6045.
    - **Mostrar/clicar:** tabela de pedidos elegíveis por prioridade. Na vigilância, dos 208 pedidos rotulados como trava, 84 só existem pela trava (EVR ≤ 0), e 124 também têm EVR > 0; há mais 81 apenas por EVR. Na confirmação, a decomposição da trava é 75 só pela trava e 279 também por EVR. Explique EVR como economia esperada da revisão.
    - **O que a plateia deve perceber:** a trava pede revisão mesmo sem economia; prioridade é FORCE_REVIEW, depois trava, depois EVR.

12. **11.6 🎛️ W6: quando faltam vagas de revisão** | **1 min 15 s**
    - **Dizer:** “Agora o posto fica sem vagas de revisão. Quem pediu e não foi atendido fica com a escolha base, inclusive quem recebeu FORCE_REVIEW.”
    - **Mostrar/clicar:** selecione “Modo” = `vigilancia` e leve “Vagas de revisão” a zero. Aponte os grupos “de fora”, inclusive prioridade `force`; os guardrails `ALWAYS_*` continuam decidindo diretamente.
    - **O que a plateia deve perceber:** FORCE_REVIEW não garante atendimento; até um homônimo pode virar MATCH sem segunda opinião. Trocar o modo repõe as vagas padrão.

13. **12. Reconciliação com `run_v3` (rota A, in-sample)** | **20 s**
    - **Dizer:** “A conta que abrimos em etapas bate exatamente com a execução oficial. Isso testa se explicamos o programa certo, não se ele funciona em outra população.”
    - **Mostrar/clicar:** linhas “idêntico: True” e diferença máxima de `p_cal` igual a zero. Não reexecute.
    - **O que a plateia deve perceber:** conferência do código e avaliação de desempenho são perguntas separadas.

14. **13.1 Métricas held-out em múltiplas sementes** + **13.7 Caminho ML: Random Forest e XGBoost contra o Platt na nota** | **2 min**
    - **Dizer:** “O recall automático (entre os decididos) deixa os pendentes de fora e mede muito as travas; não é a sensibilidade do sistema inteiro. Neste jogo, pistas extras não trazem informação nova depois da nota, então o modelo mais complexo não tem vantagem garantida.”
    - **Mostrar/clicar:** em §13.1, aponte cobertura junto do recall. Em §13.7, vá direto a “🎯 Resumo da comparação”: primeiro teste único, depois cinco sorteios; não execute treinamento nem percorra importâncias.
    - **Mostrar/clicar:** na tabela dos sorteios, compare “Custo esperado por par (vigilância / confirmação)”. Leia o custo com sensibilidade esperada e cobertura; ele inclui todo o teste e supõe os erros do revisor configurado.
    - **Cola da tabela de custo esperado por par**, média dos cinco sorteios:

      | Modelo | Vigilância | Confirmação |
      |---|---:|---:|
      | Platt (só a nota) | 1,6388 | 2,4945 |
      | Random Forest | 1,7013 | 2,4166 |
      | XGBoost | 1,7465 | 3,2434 |

    - **O que a plateia deve perceber:** no teste único, Platt fica mais perto de p* e leia a linha "Leitura desta execução" do bootstrap (na referência, RF não se separa do Platt e XGB perde em AUC/Brier). Vantagem probabilística por construção não garante menor custo sob travas e revisor suposto: a diferença de custo entre Platt e RF fica dentro da variação entre sorteios, e a ordem exata entre os três pode mudar de uma máquina para outra. Isso não prova inferioridade geral do aprendizado de máquina.

15. **16. Conclusões e limitações** | **1 min**
    - **Dizer:** “Esta aula mostra como a decisão nasce, não promete desempenho em serviço. Há três limites centrais: dados sintéticos feitos com uma curva logística, âncoras implementadas só no notebook e revisor LLM apenas simulado.”
    - **Mostrar/clicar:** em “16.2 Limitações (leia antes de generalizar)”, aponte esses três itens; termine em “16.3 Recap final”.
    - **O que a plateia deve perceber:** aplicar em saúde exige conferir dados reais, custos escolhidos pela equipe e qualidade independente da revisão.

## Pular ao vivo

- **4. Setup do ambiente e geração do dataset:** já executado; instalação e geração desviam da história.
- **7.2 Por que a data ganha lupa própria (as *features* MACD):** detalhes de datas ficam para perguntas.
- **7.4 🎛️ W5: pesos do nome e janelas MACD:** é outra exploração; as janelas quase não mudam neste gerador.
- **9.2 Como a máquina escolhe `a` e `b`: a multa por surpresa:** a conta do ajuste pode ficar para leitura posterior.
- **9.3 O passo de Newton, sem medo:** não é necessário explicar o algoritmo de busca para entender a curva.
- **🔒 Limitador por janela: o volante que não dá tranco:** efeito técnico importante, mas fora do fio principal.
- **11.4 🎛️ W2: o preço de cada erro:** com 2 vezes o FP padrão, 3 pares de `near_high`/`high` pedem revisão; com 3 vezes, são 59. Revisão a 20 vezes a taxa padrão devolve decisão automática a 73 pedidos que vinham só do EVR. A tabela de sensibilidade confere esses efeitos.
- **11.2 📐 Ponto de virada, janela de revisão e EVR:** deixe a derivação para a banca; use apenas a ideia de economia esperada.
- **13.3 Sortear fichas soltas ou grupos de fichas:** divisão por candidato já foi apresentada; IDs deste sintético não carregam informação.
- **13.4 Curvas PR e ROC da rota B:** ordenar bem é diferente de calibrar bem; não abra outra frente na fala principal.
- **13.5 Custo observado versus limiar de MATCH:** escolher o melhor corte olhando o teste exigiria outra prova.
- **14. Revisão LLM (simulador determinístico):** apenas reproduz hipóteses de erro, não demonstra um LLM real.
- **15. Painel interativo:** é um índice dos controles já visitados, não uma etapa adicional.

## Perguntas prováveis da banca

1. **Como ECE de 0,075 combina com distância até p* de 0,0099?** São comparações diferentes: ECE usa respostas sorteadas em grupos; distância até p* usa a regra escondida. Até p* tem ECE de 0,0754; veja §9.5 e “Promessa cumprida ou apenas boa ordenação?” na §13.7.
2. **RF ou XGB calibram melhor porque algum ECE é menor?** Os ECEs de XGB, RF, Platt e p* caem todos na faixa simulada para p* (~0,035–0,091): o ECE não ordena estes modelos. É um nível de ruído, não um corte para diferenças. Brier, log-loss e distância até p* favorecem Platt aqui; o bootstrap de AUC/Brier não separa RF do Platt e favorece Platt contra XGB. Veja §13.7.
3. **Para que serve o ponto de virada se a trava exige 0,85?** Ele escolhe a opção automática de menor custo e ajuda a calcular a economia da revisão. A trava de vigilância pede revisão abaixo de 0,85 quando a base é MATCH; sem vaga, vale a base. Veja §11.2, §11.3 e W6.
4. **FORCE_REVIEW sem vaga faz o quê?** Mantém a escolha base, não bloqueia o par obrigatoriamente. `ALWAYS_*` é diferente: decide diretamente e não disputa vaga. Veja §11.3 e W6.
5. **De onde vem a curva entre notas 5 e 9, sem âncoras ali?** Da forma em S escolhida, sem observações-âncora no meio. O notebook chama isso de extrapolação; uma forma real diferente pode errar sem aviso. Veja §9.8.
6. **Por que limitar `a` e `b` separadamente?** É a regra descrita na configuração, não uma recomendação de melhor método. Pode deslocar o meio da curva para fora do caminho esperado; limitar `s*` e inclinação, ou a mudança das probabilidades, é uma alternativa. Veja “🔒 Limitador por janela: o volante que não dá tranco”.
7. **De onde vêm os custos 10, 50 e 0,25?** Da configuração de vigilância, sem estimativa empírica. São preferências na mesma unidade, não necessariamente reais; uma equipe de saúde teria de justificá-las. Veja §11.1 e W2 (§11.4).
8. **O revisor simulado é circular?** Sim. Ele consulta TARGET e injeta erros configurados; o acerto devolve essas hipóteses com ruído. Não mede um LLM real nem implementa dois agentes e árbitro. Veja §14 e §16.2.
9. **Onde está a sensibilidade do sistema inteiro?** Na “Sensibilidade esperada do sistema”, incluindo revisados sob as taxas de erro presumidas. Não confunda com recall automático (entre os decididos); confira também cobertura e custo. Veja §13.1 e as tabelas finais da §13.7.
10. **Cinco sorteios dão incerteza real sobre uma população?** Não. Reorganizam os mesmos pares e mostram dependência da divisão, não um intervalo de confiança populacional; o bootstrap pareado também reamostra pares deste teste, não novas populações. Veja §13.1 e §13.7.

## Se algo der errado ao vivo

- **O controle mostra uma mensagem de erro:** pare a exploração e use a figura estática já renderizada acima ou a saída inicial preservada. Ela continua válida para os parâmetros originais; não descreva a tentativa inválida como resultado.
- **W4 reclama das fronteiras:** mantenha todas em ordem estritamente crescente. Para a demonstração, não mexa nelas; altere só “Nota sempre match”.
- **W3 mostra “congelado”:** faltou o mínimo de pares. Nesse controle, a curva exibida volta ao Platt supervisionado de referência; não é uma nova curva por âncoras. Restaure cortes e mínimos originais para continuar.
- **O kernel reiniciou:** as figuras podem continuar visíveis, mas as variáveis sumiram. Execute as células em ordem até a seção necessária, sem pular dependências; não tente recuperar a §13.7 ao vivo, use as saídas previamente preparadas.
- **Widgets não aparecem:** siga com tabelas e figuras estáticas. Explique a mudança que faria e mantenha a conclusão nos valores originais, sem inventar um resultado interativo.
