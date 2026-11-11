# Notebook didático: GZ-CMD passo a passo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marco-jardim/gzcmd-record-linkage/blob/master/notebooks/gzcmd_passo_a_passo.ipynb)

Material de apresentação em português, para profissionais e estudantes de saúde
coletiva, sobre o *linkage* (relacionamento de registros) entre o Sistema de
Informações sobre Mortalidade (SIM) e o Sistema de Informação de Agravos de
Notificação (Sinan), nas notificações de tuberculose.

O notebook ilustra, numa base sintética, a configuração recomendada pela tese:

1. preservar os vínculos aceitos pelo ponto de corte do OpenRecLink;
2. na *grey area* (região abaixo do ponto de corte), estimar a probabilidade
   calibrada fora da amostra de cada potencial par: XGBoost sobre os 29
   subescores e a nota final do Comparador de Registros (CdR), calibração
   isotônica, validação cruzada aninhada 5 × 5 agrupada por óbito (`REFREC`);
3. ordenar a fila de revisão manual por essa probabilidade, isolada (O3) ou
   fundida ao escore do OpenRecLink por posição recíproca (F4), e escolher entre
   elas pela curva de esforço, com a incerteza dada pelo *bootstrap* agrupado;
4. usar o LLM, se houver, só como apoio: o `MATCH` dele segue ao revisor humano
   (variante que a tese indica caso se use o LLM, ainda como hipótese
   exploratória).

A política v3.1 de quatro ações com LLM simulado aparece como contraste (na
tese, ela não reduziu a revisão manual além da ordenação), e o perfil histórico
do pacote (`run_v3` com `p_cal="platt"` e `policy_version="v3"`) como linha de
base reproduzível, não como recomendação.

**Todos os dados são sintéticos.** A base imita a dificuldade do problema da
tese (pares verdadeiros raros abaixo do corte, concentração no passo 11, escore
e nota final discordantes), não os seus números. Os resultados do notebook
demonstram o método; não são evidência sobre ele. Os números da tese aparecem
só na seção 10, como referência citada, com o rótulo da seção de
`conclusao.tex` e a release pública da tese
([tese-2026-11-11-7e91702](https://github.com/marco-jardim/phd-research/releases/tag/tese-2026-11-11-7e91702)).

## Seções

| Seção | Conteúdo |
|---|---|
| Preparação do ambiente | Localiza o repositório (ou clona e instala, no Colab) e importa os módulos. |
| 0. Glossário | Terminologia da tese, em linguagem simples. |
| 1. O problema | O ponto de corte, a *grey area*, o custo da revisão manual e uma fila de brinquedo. |
| 2. Teoria do GZ-CMD: a política de quatro ações | Probabilidade calibrada, custos esperados (*expected loss*) das quatro ações, regra de decisão, guardas, orçamentos e modos; exemplo resolvido que separa o efeito dos custos do efeito das taxas do LLM, conferido com o motor do pacote; controle dos custos em função de p. |
| 3. Dados sintéticos | O que `gerar_base` simula e por quê; contagens por passo; sobreposição de PAR = 0 e PAR = 2; viés de verificação pela verdade da simulação. |
| 4. Probabilidade calibrada fora da amostra | Validação cruzada aninhada agrupada por óbito, diagrama de confiabilidade na base inteira e abaixo do corte, degraus da calibração isotônica, por que partir por óbito, AUC e precisão média abaixo do corte. |
| 5. Ordenações da fila | O1, O2, O3 e F4; a fusão por posição recíproca num exemplo de brinquedo, conferida com `ranking.rrf`. |
| 6. Curva de esforço | Revisões para as metas de 25%, 64% e 100% dos PAR = 2; mesma carga; controle do orçamento humano. |
| 7. Incerteza | *Bootstrap* agrupado por óbito de O3 com O1, F4 com O1 e F4 com O3. |
| 8. Contraste com a automação | Política v3.1 nos dois modos com LLM simulado; comparação na mesma carga humana; VPP do `MATCH` do LLM. |
| 9. Linha de base histórica | Seção técnica: perfil v3 reproduzível sobre a base antiga, reconciliado passo a passo com `run_v3`. |
| 10. Fluxo recomendado e limitações | O fluxo nas palavras da tese, onde está cada função, números da tese citados com a fonte e limitações. |

Dois controles interativos acompanham o texto: os custos esperados por modo e
faixa da nota final (seção 2) e o orçamento humano na curva de esforço (seção
6). Cada um desenha primeiro uma figura estática, que fica na saída e no arquivo
exportado, e depois o widget, quando o Jupyter tem `ipywidgets`.

## Conteúdo desta pasta

| Arquivo | Papel |
|---------|-------|
| `gzcmd_passo_a_passo.ipynb` | Notebook da apresentação. **Arquivo gerado**, sem saídas: não editar à mão. |
| `build_notebook.py` | **Fonte do notebook**: monta o `.ipynb` célula a célula via `nbformat`, a partir das listas em `TODAS_AS_SECOES`. Edite aqui e regenere. |
| `synthetic_linkage.py` | Gerador da base sintética da fila abaixo do corte (potenciais pares, 29 subescores, nota final, escore do OpenRecLink, ponto de corte, `PAR`). |
| `nb_fila.py` | Probabilidade calibrada fora da amostra, filas O1 a F4, curva de esforço, *bootstrap* agrupado, política v3.1 e revisor LLM simulado. Inclui `prob_por_linha` (partição por potencial par), que o notebook não usa: a base sintética não tem canal de vazamento entre potenciais pares do mesmo óbito. |
| `synthetic_data.py` | Gerador da base sintética antiga, usada só na seção 9 (perfil histórico). |
| `nb_helpers.py` | Controles interativos, métricas de apoio e diagrama de confiabilidade. |
| `ROTEIRO_DEMO.md` | Roteiro para apresentação ao vivo. |

No pacote publicado, a fusão de filas está em `gzcmd_record_linkage.ranking`
(`rrf` e `review_queue(df, scores=[...], method="rrf")`). Os empates são
desfeitos por uma permutação aleatória com semente fixa sobre a posição das
linhas: mudar a semente ou a ordem das linhas muda a ordem dentro dos empates.

A base antiga é gravada (e versionada) em
`../data/synthetic/comparador_sintetico.csv`; o notebook confere o arquivo sem
sobrescrevê-lo. A base nova é gerada em memória a cada execução, com semente
fixa.

## Rodar no Google Colab

Clique no selo **Open in Colab** acima (ou abra o `.ipynb` em *File → Open
notebook → GitHub*). A primeira célula de código detecta o Colab e:

1. clona o repositório (`git clone --depth 1`), ou atualiza uma cópia anterior
   que não tenha arquivos editados;
2. instala a biblioteca `gzcmd_record_linkage` via `pip`;
3. coloca os módulos do notebook (`synthetic_linkage`, `nb_fila`, `nb_helpers`,
   `synthetic_data`) no `sys.path`.

Depois, execute as células de cima para baixo (*Runtime → Run all*). O Colab
clona o ramo padrão do repositório: o notebook só funciona lá quando os módulos
desta pasta estiverem nesse ramo.

## Pré-requisitos e instalação (pwsh)

- Python 3.10 ou mais recente.
- A biblioteca `gzcmd_record_linkage` instalada (modo editável recomendado).

A biblioteca e o ferramental de notebook são instalados separadamente; o
ferramental não entra nas dependências do pacote publicado.

```pwsh
# A partir da raiz do repositório
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m pip install -r requirements/notebook.txt
```

Com a biblioteca já instalada no ambiente atual, basta o ferramental:

```pwsh
python -m pip install -r requirements/notebook.txt
```

## Abrir e executar

```pwsh
jupyter lab notebooks/gzcmd_passo_a_passo.ipynb
```

Execução sem interface (teste de reprodutibilidade), com a cópia executada fora
do repositório:

```pwsh
jupyter nbconvert --to notebook --execute `
  --ExecutePreprocessor.timeout=600 `
  --output-dir "$env:TEMP" --output gzcmd_passo_a_passo.executed `
  notebooks/gzcmd_passo_a_passo.ipynb
```

A execução completa leva de um a três minutos num computador portátil. A seção
4 concentra o tempo (validação cruzada aninhada com 30 ajustes do XGBoost); as
demais seções levam segundos.

Se a sessão for reiniciada, as seções 4 a 9 avisam que as variáveis anteriores
sumiram; use *Executar antes* (Ctrl+F8) na célula em que parou.

## Regenerar o notebook

O `.ipynb` é gerado; a fonte é `build_notebook.py`. Depois de qualquer
alteração:

```pwsh
python notebooks/build_notebook.py
```

## Regras de conteúdo

Conferidas pelos testes em `tests/test_notebook_didatica.py`:

- toda célula de código vem depois de um markdown que explica o que ela faz e
  por quê; cada seção com código tem uma célula **Leitura** depois das saídas;
- números da base sintética nunca ficam fixos no markdown: o código os imprime,
  e a função `confere` testa, na execução, as frases que dependem deles;
- números da tese só na seção 10, com o rótulo da seção de `conclusao.tex`
  (`sec:concl-sintese`, `sec:concl-limitacoes`, `sec:concl-recomendacoes`) e a
  release pública da tese, sem números de linha;
- terminologia da tese (*grey area*, potenciais pares, par verdadeiro e não par,
  vínculo só para a decisão, revisão manual, referência operacional, VPP,
  *expected loss* glosado como custo esperado, *Platt scaling*); a verdade
  latente da simulação se chama "mesma pessoa (verdade da simulação)";
- custos lidos de `nb_fila`, nunca digitados nas células;
- os controles desenham uma figura estática antes do widget.

## Testes

```pwsh
# Suíte rápida (inclui a auditoria estática do notebook)
python -m pytest -q -m "not notebook"

# Execução do notebook de ponta a ponta (mais lenta)
python -m pytest -q -m notebook

# Cobertura dos módulos de apoio do notebook
python -m pytest -q --cov=synthetic_linkage --cov=nb_fila --cov=nb_helpers --cov-report=term-missing
```
