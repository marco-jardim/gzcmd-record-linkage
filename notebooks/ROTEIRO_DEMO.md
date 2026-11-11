# Roteiro de demonstração ao vivo

Notebook: `notebooks/gzcmd_passo_a_passo.ipynb`.
Fonte do conteúdo: `notebooks/build_notebook.py`.
Público: professora e colegas de saúde coletiva, sem formação em estatística ou aprendizado de máquina.
Duração: cerca de 18 minutos, com aproximadamente 16 minutos de fala e 2 de navegação.
Use as falas como apoio. Não leia código nem todas as colunas das tabelas.

**Números.** Os números da base sintética não estão neste roteiro de propósito: leia-os na saída impressa pelo notebook, que é a fonte. As frases marcadas com ✓ na saída foram conferidas pela própria execução. Os números da tese só aparecem na seção 10, com o rótulo da seção de `conclusao.tex` e a release pública da tese; cite-os de lá.

## Antes de começar

- Execute o notebook inteiro antes: **Kernel → Restart & Run All**. Espere terminar e deixe figuras e tabelas prontas. Ao vivo, role pelas saídas; execute apenas a célula de um controle se precisar reativá-lo.
- No ambiente do kernel, instale `python -m pip install -r requirements/notebook.txt`. Abra em Jupyter com suporte a `ipywidgets` e teste os dois controles antes da apresentação. Cada controle desenha primeiro uma figura estática; o widget fica logo abaixo dela.
- A execução completa leva de um a três minutos. A seção 4 concentra o tempo (validação cruzada aninhada). **Não execute de novo a seção 4 ao vivo.**
- O notebook usa sementes fixas e o XGBoost em CPU: na mesma máquina e com as mesmas versões de bibliotecas, os números se repetem. Em outra máquina, podem variar um pouco; as frases conferidas com ✓ continuam valendo, ou a execução para e avisa.
- Para resetar um controle, execute de novo **somente a célula dele**.

## Roteiro

1. **Abertura** + **1. O problema: o que fica abaixo do ponto de corte** | **2 min**
   - **Dizer:** “O OpenRecLink aceita como vínculo o que passa do ponto de corte. Abaixo dele, na *grey area*, há dezenas de milhares de potenciais pares e poucas dezenas de pares verdadeiros: óbitos de pessoas com tuberculose notificada que o corte deixou passar. Revisar tudo é inviável; a pergunta é em que ordem revisar.”
   - **Mostrar/clicar:** o aviso de dados sintéticos e a figura da fila de brinquedo. Aponte que as duas filas acham os mesmos 3 pares verdadeiros, mas a fila A acha a maior parte nas primeiras revisões.
   - **O que a plateia deve perceber:** com orçamento humano limitado, a ordem da fila decide quantos óbitos serão vinculados.

2. **2.1 a 2.3 Teoria: a política de quatro ações** | **2 min 30 s**
   - **Dizer:** “Esta política foi avaliada na tese e não reduziu a revisão manual além da ordenação; ela vem primeiro porque introduz duas ideias: probabilidade calibrada e custo dos erros. Para cada potencial par, a política compara quatro custos esperados: aceitar, rejeitar, mandar ao revisor humano ou consultar o LLM, e escolhe o menor. Na vigilância, perder um óbito custa mais; na confirmação, unir pessoas diferentes custa mais.”
   - **Mostrar/clicar:** a tabela dos quatro custos esperados (leia só as linhas de aceitar e rejeitar em voz alta) e a tabela de custos impressa.
   - **O que a plateia deve perceber:** a evidência (p) fica separada da preferência do serviço (os custos); a decisão combina as duas de forma explícita.

3. **2.4 Um exemplo resolvido** + **2.5 controle dos custos** | **2 min**
   - **Dizer:** “Um par com 80% de probabilidade, com nota final de 8 a 9. A probabilidade é a mesma nos dois modos, e as taxas do LLM nessa faixa são parecidas. Na vigilância, aceitar sai mais barato; na confirmação, a política manda à revisão manual. A terceira linha usa os custos da confirmação com as taxas da vigilância: a ação não muda, então quem decide são os custos.”
   - **Mostrar/clicar:** as linhas ✓ da conferência com o motor. Depois, a figura estática da vigilância com nota final de 6 a 7 e as duas tabelas de janelas impressas (de 5 a 6 e de 6 a 7).
   - **O que a plateia deve perceber:** revisar vence numa janela larga de p, não só no meio; onde o LLM errou pouco, ele substitui a revisão manual; onde ele rejeitou todos os pares verdadeiros que viu, a revisão manual vence.

4. **3. Dados sintéticos** | **2 min**
   - **Dizer:** “A base imita a dificuldade da tese, não os seus números: registros precários, homônimos, um passo de blocagem fraco e o viés de verificação da revisão manual de origem. O que se vê aqui demonstra o método; não é evidência.”
   - **Mostrar/clicar:** a tabela por passo (leia o total abaixo do corte e o número de PAR = 2 impressos), a figura dos histogramas e as contagens de PAR = 0 e PAR = 2 acima da mediana dos PAR = 2.
   - **O que a plateia deve perceber:** os PAR = 2 são raríssimos abaixo do corte, e nem o escore nem a nota final os separam bem sozinhos; a referência favorece os pares de escore alto (viés de verificação).

5. **4. Probabilidade calibrada fora da amostra** | **2 min 30 s**
   - **Dizer:** “O XGBoost combina os 29 subescores e a nota final. Três cuidados: cada par recebe a probabilidade de um modelo que não o viu; todos os pares de um mesmo óbito ficam do mesmo lado; e a probabilidade é calibrada, também fora da amostra. Para a fila, só a ordem importa; a calibração serve para ler p como probabilidade e para a política.”
   - **Mostrar/clicar:** o diagrama da validação cruzada aninhada, os dois painéis de confiabilidade (base inteira e abaixo do corte, com os grupos escassos listados) e a tabela de AUC e precisão média abaixo do corte.
   - **O que a plateia deve perceber:** a AUC é alta para todos porque há muitos não pares fáceis; a precisão média, que olha o topo da fila, é maior para a probabilidade calibrada. Em dados reais, partir por potencial par em vez de por óbito seria vazamento; esta base sintética não tem por onde vazar, e o notebook só explica isso em texto.

6. **5. Ordenações da fila** | **1 min 30 s**
   - **Dizer:** “Quatro filas: pelo escore do OpenRecLink, pela nota final, pela probabilidade calibrada e pela fusão das duas últimas por posição recíproca. A fusão mistura posições, não valores, porque o escore e a probabilidade estão em escalas diferentes.”
   - **Mostrar/clicar:** o exemplo de brinquedo (A, primeiro numa fila e quarto na outra; a fusão impressa em letras) e as linhas que dizem qual fila lidera com 10 e com 100 revisões.
   - **O que a plateia deve perceber:** as quatro filas fazem melhor que o sorteio, mas não igualmente; F4 foi escolhida na tese entre seis variantes, sem pré-registro.

7. **6. Curva de esforço** + **controle do orçamento humano** | **2 min 30 s**
   - **Dizer:** “Com revisor perfeito simulado, contamos quantas revisões cada fila exige para recuperar 25%, 64% e 100% dos PAR = 2. Nenhuma fila vence em todas as metas: a escolha depende do orçamento humano.”
   - **Mostrar/clicar:** a tabela de revisões por meta (com os empates indicados), a tabela de mesma carga e a figura em escala logarítmica. No controle, compare um orçamento de algumas dezenas de revisões com um de algumas centenas.
   - **O que a plateia deve perceber:** a recomendação não é uma fila fixa; é escolher entre O3 e F4 pela curva de esforço, dado o orçamento. Na recuperação integral, os degraus da calibração isotônica podem atrasar O3; parte da vantagem de F4 e O1 no fim da fila pode vir da própria referência.

8. **7. Incerteza** | **1 min**
   - **Dizer:** “Sorteamos óbitos com reposição duas mil vezes e recontamos as revisões. Se o intervalo de 95% fica inteiro de um lado do zero, a mesma fila venceu em pelo menos 97,5% das réplicas; se inclui o zero, o *bootstrap* não distingue a diferença do acaso.”
   - **Mostrar/clicar:** as três linhas de leitura impressas (O3 com O1, F4 com O1, F4 com O3).
   - **O que a plateia deve perceber:** com poucos pares verdadeiros, os intervalos são largos; e o intervalo não inclui a incerteza do treinamento nem a da referência.

9. **8. Contraste com a automação** | **2 min**
   - **Dizer:** “Agora a alternativa: deixar a política decidir abaixo do corte, com um LLM simulado com as taxas de erro do revisor de referência da tese. Comparamos na mesma carga humana, com uma simplificação que favorece a política.”
   - **Mostrar/clicar:** as leituras impressas da comparação na mesma carga (falsos positivos do LLM e total) e a linha do VPP do `MATCH` do LLM.
   - **O que a plateia deve perceber:** na vigilância, a automação recupera no máximo um par verdadeiro a mais que a fila O3 e acrescenta falsos positivos; o `MATCH` do LLM tem VPP baixo, por isso o LLM só apoia, e o `MATCH` dele vai ao revisor humano.

10. **10. Fluxo recomendado e limitações** | **1 min 30 s**
    - **Dizer:** “Nas palavras da tese: preservar o corte e, abaixo dele, ordenar a revisão manual pela probabilidade calibrada, isolada ou fundida ao escore, escolhendo pela curva de esforço. Os números da tabela são da coorte real e vêm da conclusão, seção a seção. E o que fica além do orçamento é ‘não examinado’, não ‘não par’: ausência de vínculo por falta de revisão não é óbito não notificado.”
    - **Mostrar/clicar:** as duas citações e a tabela de números da tese; termine em “10.4 Para levar”.
    - **O que a plateia deve perceber:** os limites centrais são revisor perfeito simulado, análise *post hoc* com F4 escolhida entre variantes, referência produzida por revisora única (com viés de verificação e risco de viés de incorporação) e LLM simulado.

## Pular ao vivo

- **Preparação do ambiente:** já executada.
- **0. Glossário:** volte a ele só se surgir uma dúvida de termo.
- **Taxas de erro do LLM por faixa (seção 2.3):** é a tabela de parâmetros; mencione só que vêm do revisor de referência da tese.
- **Parte da escala de p em que cada ação vence (fim da seção 2):** detalhe técnico.
- **O que a referência não vê (fim da seção 3):** guarde para a pergunta sobre viés de verificação.
- **As partições, antes de ajustar (seção 4):** mostre só se perguntarem como se evita o vazamento.
- **Os degraus da calibração isotônica (seção 4):** retome se perguntarem por que O3 vai mal na recuperação integral.
- **9. Linha de base histórica:** seção técnica; confere que o perfil v3 continua reproduzível e não faz parte da recomendação.

## Perguntas prováveis da banca

1. **Os resultados sintéticos confirmam a tese?** Não. A base foi ajustada para imitar a dificuldade do problema, não os resultados. O notebook demonstra o método; os números da tese estão na seção 10, com a fonte.
2. **Por que agrupar por óbito na validação cruzada?** Em dados reais, potenciais pares do mesmo óbito repetem nome, nascimento e mãe do falecido; se caírem no treino e no teste, o modelo é avaliado num óbito que já viu, e a avaliação fica otimista. Esta base sintética não tem esse canal de vazamento, por isso o notebook não compara as partições (seção 4).
3. **Por que a AUC é alta para tudo e a precisão média não?** A *grey area* tem muitos não pares fáceis, que inflam a AUC. A precisão média olha o topo da fila, onde está o orçamento humano. Veja a seção 4.
4. **Por que não escolher sempre O3?** Nenhuma fila vence em todas as metas. O3 tende a ir bem nas metas menores; na recuperação integral, os degraus da calibração isotônica podem atrasá-la, e F4 desempata pelo escore. Parte dessa vantagem pode vir do viés de verificação da referência. Veja as seções 4 e 6.
5. **O que significa um IC que exclui zero aqui?** Que, nas réplicas do *bootstrap* agrupado por óbito, a mesma fila exigiu menos revisões em pelo menos 97,5% delas. O IC é condicional: modelo e filas fixos, referência tomada como certa; neste notebook, a fusão não é refeita nas réplicas, ao contrário da tese. Veja a seção 7.
6. **Por que o LLM não pode aceitar vínculos sozinho?** Mesmo com taxa de falso positivo pequena, ele é aplicado a muitos não pares, e o VPP do `MATCH` fica baixo. Veja a seção 8 e, na tese, `sec:concl-recomendacoes`.
7. **O LLM do notebook é real?** Não. As respostas são sorteadas com as taxas de erro por faixa do revisor de referência da tese (Claude Haiku 5.5). Nenhuma chamada é feita.
8. **De onde vêm os custos de vigilância e confirmação?** São os da tese, lidos do módulo `nb_fila`. Representam a importância relativa dos erros para o serviço, na unidade de revisões humanas equivalentes.
9. **O que é o perfil histórico e por que ele está no notebook?** É o `run_v3` com *Platt scaling* e política v3. A tese compara antes e depois, e o pacote precisa reproduzi-lo; a seção 9 confere a reprodução, não o desempenho.
10. **O que fazer com os potenciais pares que ficam além do orçamento?** Registrá-los como não examinados, não como não pares: a ausência de vínculo por esgotamento do orçamento não equivale a óbito não notificado (`sec:concl-recomendacoes`).

## Se algo der errado ao vivo

- **Uma célula para com a mensagem “O texto afirma ...”:** uma frase do texto deixou de valer com os números desta execução (outra versão de biblioteca, por exemplo). Não improvise: diga que o notebook confere o próprio texto, siga com a saída preparada antes e anote para corrigir o texto.
- **O controle mostra uma mensagem de erro:** pare a exploração e use a figura estática desenhada acima do controle. Não descreva a tentativa inválida como resultado.
- **O kernel reiniciou:** as figuras podem continuar visíveis, mas as variáveis sumiram. As seções 4 a 9 avisam; use *Executar antes* (Ctrl+F8) na célula em que parou. Não reexecute a seção 4 ao vivo se der para seguir com as saídas preparadas.
- **Widgets não aparecem:** siga com as figuras estáticas, que já estão na saída, e explique a mudança que faria, sem inventar um resultado interativo.
