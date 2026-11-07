# GZ-CMD-Record-Linkage

Componentes Python para pós-processamento de potenciais pares de *linkage* e
encaminhamento da área cinza para revisão. O pacote recebe escores produzidos por
um comparador; não executa blocagem nem gera potenciais pares a partir de duas bases.

## Nome e origem

GZ-CMD tem como nome histórico "Grey-Zone: Calibrated Mixture-of-experts +
Conformal Deferral + Constrained Matching". Os termos não correspondem todos a
mecanismos implementados:

- **Calibrated**: calibração de probabilidades, com Platt no perfil histórico e
  calibração isotônica no perfil v3.1.
- **Mixture-of-experts**: uso livre do termo para a articulação de regras,
  classificador e revisor sob uma política, sem mecanismo aprendido de seleção
  entre especialistas.
- **Conformal Deferral**: o encaminhamento à revisão humana decorre de
  probabilidades calibradas, custos e orçamento, não de predição conformal;
  não há quantis de não conformidade nem garantia de cobertura.
- **Constrained Matching**: guardas determinísticas e limites de revisão, sem
  cardinalidade N:1 nem resolução de conflitos por margem de probabilidade.

## Versão e escopo

- Distribuição PyPI: `gzcmd-record-linkage==0.4.0`
- Importação Python: `gzcmd_record_linkage`
- Licença: [MIT](LICENSE)
- Python: 3.10 ou superior
- Versão no repositório: **0.4.1** (ainda não publicada)
- Versão no PyPI: **0.4.0**
- Políticas: **v3** (padrão, três ações) e **v3.1** (quatro ações, desde 0.4.0)

A versão do pacote, a versão da política (v3 ou v3.1) e a versão do depósito
de dados (3.0.0) não são intercambiáveis. A instalação PyPI abaixo fornece
0.4.0, com suporte às políticas v3 e v3.1. A paridade da política não inclui
automaticamente os modelos, as probabilidades OOF nem as taxas empíricas dos
experimentos da tese.

## Instalação

```console
python -m pip install gzcmd-record-linkage==0.4.0
```

Para desenvolvimento, a partir deste repositório:

```console
python -m pip install -e ".[dev]"
```

A versão é declarada em `pyproject.toml`; não depende de `hatch-vcs`.

## Exemplo mínimo sem dados individuais nem serviços externos

O exemplo exercita somente a triagem v3. As probabilidades, a banda, os custos e
as taxas de erro são sintéticos e fornecidos diretamente. Não há treinamento,
calibração ou chamada a um modelo de linguagem neste exemplo.

```python
import pandas as pd
from gzcmd_record_linkage.policy_engine import (
    Budget, Costs, LLMError, PolicyEngineV3,
)

pares = pd.DataFrame({
    "p_cal": [0.01, 0.50, 0.99],
    "band": ["grey_mid"] * 3,
})
motor = PolicyEngineV3(
    costs=Costs(false_positive=10, false_negative=50,
                llm_review=0.25),
    llm_error_by_band={"grey_mid": LLMError(0.06, 0.10)},
    budget=Budget(llm_max=1),
)
saida = motor.triage(pares)
print(saida[["p_cal", "action", "review_requested"]]
      .to_string(index=False))
```

Saída na versão 0.3.0:

```text
 p_cal     action  review_requested
  0.01   NONMATCH             False
  0.50 LLM_REVIEW              True
  0.99      MATCH             False
```

`LLM_REVIEW` é encaminhamento, não resposta final de um revisor. O orçamento
contabiliza os encaminhamentos; usar o mesmo motor novamente mantém `llm_used`.

## Fluxo completo e entrada

`gzcmd_record_linkage.runner_api.run_v3` combina carregamento, atributos derivados,
bandas, calibração, guardas e triagem por *expected loss*. Retorna um `DataFrame`
e um `RunSummary`. O arquivo de configuração incluído na distribuição é
`gzcmd_v3_config.yaml`, acessível com `importlib.resources.files`.

O carregador espera um CSV do comparador, separado por ponto e vírgula, com vírgula
decimal. No protocolo da tese, os potenciais pares vêm do OpenRecLink e os
subescores vêm do Comparador de Registros. A adaptação a outros geradores de pares
exige preservar a semântica dos campos, não apenas seus nomes.

- `nota final` (com alias `nota_final`) determina as bandas e a calibração do escore.
- Os subescores do CdR abrangem nome, nome da mãe, nascimento, endereço e município.
  Exemplos de nomes consumidos: `NOME qtd frag iguais`, `NOMEMAE prim frag igual`,
  `DTNASC dt iguais`, `ENDERECO via igual` e `CODMUNRES local igual`. O exemplo
  acima entra diretamente no motor, depois da preparação e da calibração.
- `COMPREC`, `REFREC` e `PASSO` são obrigatórios no carregador e permitem
  rastrear os potenciais pares.
- `PAR` também é obrigatório no carregador e contém o gabarito para
  treinamento/avaliação: valores 1 e 2 indicam vínculo; os demais são convertidos
  em classe negativa. O ajuste supervisionado exige gabarito válido, com ambas
  as classes; a presença da coluna, por si só, não garante essa validade.
- `R_DTNASC` e `C_DTNASC` usam `yyyymmdd`; `R_DTOBITO` usa `ddmmyyyy` e
  `C_DTDIAG` usa `yyyymmdd`. As datas alimentam atributos e guardas temporais.
  Dados individuais só devem ser tratados em ambiente protegido.

O carregador aceita várias colunas ausentes, mas isso altera os atributos e a
aplicação das guardas. Campo ausente não é evidência de discordância nem de
concordância. Uma execução sem erro não valida o esquema nem a adequação da entrada.

### Configuração efetiva e interpretação

O YAML define bandas, custos por modo, limites de decisão automática, orçamento,
taxas de erro por banda e parâmetros de guardas. `vigilancia` atribui maior custo
ao falso negativo; `confirmacao`, ao falso positivo. São modos de decisão, não
classificadores separados. As guardas implementadas incluem incompatibilidade
temporal, nota baixa, risco de homonímia e concordâncias fortes.

Nem todo bloco descritivo do YAML é executado. Expressões textuais em
`guardrails.always_match`, `always_nonmatch` e `risky_if` não constituem um
interpretador de regras: o runner chama `apply_guardrails` com `guardrails.params`.
`calibration.by_band` não produz ajuste por banda nos métodos Platt.

O bloco `consistency`, inclusive `conflict_margin_delta`, é reservado e não
implementado. Não impõe cardinalidade N:1 nem resolve conflitos por margem de
probabilidade. A presença desses campos no YAML não significa que estejam ativos.

### Dossiê histórico e parâmetros do experimento

O construtor do dossiê mantém a fronteira histórica de 14 dos 29 subescores do
extrato: três de nome, três de nome da mãe, um de nascimento, seis de endereço e
um municipal, além dos compostos, indicadores e nota final. O municipal é enviado
como `municipality`. Não entram subescores de raridade, frequência, abreviação e
fragmentos muito parecidos, aproximações/inversões de data ou UF/proximidade
municipal. Componentes separados de ano/mês/dia e o campo composto de primeiro e
último fragmentos não existem nesse extrato; não são entradas adicionais do
construtor. Campos nominais são excluídos, mas isso não torna os subescores e o
pseudônimo de par dados anônimos nem autoriza seu envio a terceiros.

O YAML distribuído é um padrão da biblioteca, não a configuração dos runs da
tese. Os três revisores estão descritos nele com `gpt-5.2-pro`, `temperature: 0.0`
e `max_tokens: 500`; esses defaults foram preservados. O experimento da tese com
Claude Haiku 5.5 omitiu `temperature` e usou `max_tokens: 2500`, com instruções v4
e sem `model_outputs` no dossiê. Reproduzir esse experimento exige seus runners,
instruções, coortes, calibração e taxas empíricas; não basta carregar o YAML do
pacote ou trocar o nome do modelo.

A saída acrescenta `band`, `p_cal`, `guardrail`, `guardrail_reason`, `action`,
`base_choice`, `base_loss`, `loss_llm`, `evr` e `review_requested`. Na v3.1 também
inclui `loss_human`, `decision_reason`, `human_gain`, `selected_loss`, `loss_match`
e `loss_nonmatch`. `evr` compara a melhor alternativa não LLM (incluindo humano
na v3.1) com a perda LLM. `review_requested=True` não
garante encaminhamento quando o orçamento se esgota; confira `action` e o resumo.
O runner não executa automaticamente a revisão externa dos pares encaminhados.

## Ordenação da fila de revisão

A ordenação padrão da política continua por `evr` (e, na v3.1, pelos ganhos
humanos e de fallback). Essa ordenação distribui orçamento, não é substituída
pela fusão. O perfil histórico `p_cal="platt"`, `policy_version="v3"` permanece
inalterado. Não há nova chave de configuração nem opção CLI: o módulo
`gzcmd_record_linkage.ranking` é um utilitário opt-in para uma fila de revisão
humana já selecionada pelo chamador, por exemplo, os pares abaixo de seu corte.

RRF (*Reciprocal Rank Fusion*) soma `1 / (k + posição)` em cada ordenação,
com posições iniciadas em 1 e `k=60`. Pode ser usada para combinar prioridades
de escores em escalas diferentes, sem recalibrá-los. Não há garantia de redução
do trabalho de revisão; avalie a ordenação em dados independentes.

```python
from gzcmd_record_linkage.ranking import review_queue

# O chamador define o corte e a população elegível; maior escore vem primeiro.
abaixo_do_corte = pares.loc[pares["nota_final"] < corte]
fila = review_queue(
    abaixo_do_corte, scores=["nota_final", "p_cal"], method="rrf", seed=42,
)
```

`review_queue` devolve uma cópia reordenada, sem alterar ações, colunas, índices
ou orçamento. `method="interleave"` alterna as filas, pulando itens já emitidos;
`method="evr"` preserva exatamente a ordem recebida, sem recalcular EVR nem
classificar novamente a saída da política. O padrão do utilitário é `rrf`,
mas ele não é chamado automaticamente pelo motor ou pelo runner.

`rrf` e `interleave` recebem ordens completas de IDs posicionais `0..n-1`;
para vetores de escores alinhados aos mesmos itens, use `input_kind="scores"`.
O parâmetro explícito evita confundir escores inteiros com permutações.
Empates de escores e de RRF seguem a prioridade dos IDs após embaralhamento
com `seed=42` por padrão, não a ordem original das linhas nem os rótulos.
A reprodução exige IDs posicionais e sementes iguais. `validate_order(order, n)`
rejeita omissões, duplicações e IDs inválidos.

`ThompsonQueue(ordens, window=200, prior=(1, 1), seed=42)` oferece uma fila
adaptativa: chame `next()` e depois `update(item, label)`, com rótulo binário
(1 para vínculo confirmado). Somente o braço sorteado recebe a recompensa;
`window` limita as últimas respostas recebidas por braço (`None` mantém todas).
Itens reservados não se repetem globalmente, mesmo antes do feedback;
`next()` lança `StopIteration` ao esgotar a fila. Respostas podem chegar fora
de ordem, mas cada item só pode ser atualizado uma vez. A classe não lê
gabaritos nem executa revisão externa. Diferentemente das funções de fusão,
ela mantém estado local entre chamadas, sem I/O.

## Linha de comando

A CLI existe na versão 0.3.0 e oferece `run`, `fit-calibration` e `eval`:

```console
gzcmd --help
python -m gzcmd_record_linkage --help
gzcmd run --help
gzcmd eval --help
```

Sintaxe para uma entrada já preparada e rotulada, usando ajuste supervisionado de
Platt (substitua os caminhos pelos seus arquivos):

```console
gzcmd run --input pares.csv --config minha_config.yaml --mode vigilancia --p-cal fit_platt --output decisoes.csv
```

Esse comando ajusta e aplica a calibração na entrada; **não é avaliação
independente**. Para avaliação, use `gzcmd eval`, separando treinamento e teste e
explicitando o agrupamento, por exemplo `--split-by refrec`. O modo `stub` é apenas
uma aproximação para desenvolvimento, não calibração validada.

## Reprodução das políticas da tese

Desde 0.4.0, a API aceita `policy_version="v3"` ou `policy_version="v3.1"`
em `PolicyEngineV3`, `build_engine_from_config` e `run_v3`. Sem override, o
runner e o construtor por configuração usam `guardrails.policy_version` do YAML;
quando a chave é omitida, o padrão é **v3**. O motor direto também usa **v3**.
O perfil histórico `run_v3(..., p_cal="platt", policy_version="v3")` conserva
as decisões anteriores. A opção explícita prevalece sobre o YAML. Se o YAML
declara v3.1 e a execução pede v3, o override histórico restaura custo LLM 0,25
e guardas de nota 10/3, mesmo que o YAML configure custo 0,005 e guardas 9/5.
Um YAML que já declara v3 continua respeitando seus próprios custos e guardas.

```python
from gzcmd_record_linkage.runner_api import run_v3

saida, resumo = run_v3(
    input_csv="pares.csv", config_path="minha_config.yaml",
    mode="vigilancia", p_cal="load_platt", platt_model_path="platt.json",
    policy_version="v3.1",
)
```

O modelo salvo deve ter sido ajustado em dados independentes da entrada, se o
objetivo for avaliação. A política consome `p_cal`; selecioná-la não cria
probabilidades out-of-fold nem valida a calibração.

```console
gzcmd run --input pares.csv --mode vigilancia --p-cal load_platt --platt-model platt.json --policy-version v3.1 --output decisoes.csv
```

### Política v3.1 de quatro ações

As ações são `MATCH`, `NONMATCH`, `LLM_REVIEW` e `DEFERRED` (encaminhamento direto
ao revisor humano, não descarte). Nota < 5 determina `NONMATCH`; nota >= 9 com
`p_cal >= 0.9` determina `MATCH`. Nota alta sem essa consistência volta à
comparação de perdas. Filtros temporais e de homonímia pertencem somente à v3.

Fora das guardas, a v3.1 compara as perdas esperadas das quatro ações. Empates
preferem `MATCH`, `NONMATCH`, revisão humana e, por último, LLM. As vagas LLM
são distribuídas por ganho positivo sobre a melhor alternativa não LLM, com
ordenação estável. A falta de vagas humanas ou LLM aciona o fallback documentado
em [PARIDADE_v31.md](docs/PARIDADE_v31.md), não encaminhamento humano cego.
Os caps legados e `FORCE_REVIEW` não substituem o argmin v3.1.

O YAML aceita `decision_policy.modes.<modo>.costs.human_review` (padrão 1),
`human_error: {e_fp: 0.05, e_fn: 0.05}` e
`human_budget: {max_pairs_per_window: null}` (capacidade ilimitada).
`llm_review.reliability.error_rates_by_band.<banda>.e_unsure` (padrão zero)
acrescenta à perda LLM a carga humana esperada após abstenção. As taxas de erro
humano são hipóteses, não concordância medida. `human_used` permite continuar
o orçamento entre lotes; `RunSummary` informa consumo humano e versão da política.

O YAML distribuído conserva os custos e taxas fixas da v3, inclusive custo LLM
0,25, para não alterar o perfil histórico. Para reproduzir um cenário v3.1 da
tese, forneça seu YAML com custo LLM 0,005, taxas por banda, probabilidades OOF,
modelos e particionamento correspondentes. Taxas empíricas externas não são
carregadas automaticamente. O comando `eval` permanece um avaliador **v3** de
três ações; não apresenta métricas v3.1 como se fossem avaliação completa.

Preserve versões das dependências, sementes e grupos de treinamento/avaliação.
Custos e erros dos revisores são hipóteses operacionais; não substituem auditoria.
Nenhum resultado sintético deste README estima desempenho no serviço.

## Licença, depósito de dados e citação

O código é distribuído sob [licença MIT](LICENSE), sem garantia.

O depósito associado é o **Benchmark de Similaridade para Record Linkage em
Tuberculose (SINAN)**. Os DOI identificam o depósito de dados, não uma versão do
pacote:

- Versão 3.0.0: [10.5281/zenodo.23928226](https://doi.org/10.5281/zenodo.23928226)
- DOI conceito: [10.5281/zenodo.18652286](https://doi.org/10.5281/zenodo.18652286)

Ao utilizar o pacote, cite a versão efetivamente usada (0.4.0 publicada ou
outra versão identificada, com seu commit quando instalada do repositório) e o
[repositório de software](https://github.com/marco-jardim/gzcmd-record-linkage).
Para o método e os experimentos, cite também a tese de Marco Elisio Oliveira
Jardim, no Programa de Pós-Graduação em Saúde Coletiva, IESC/UFRJ, e identifique o
commit consultado no
[repositório da tese](https://github.com/marco-jardim/phd-research/tree/pos-qualificacao).
Ao reutilizar o benchmark, cite separadamente o DOI da versão utilizada.

O pacote não substitui o OpenRecLink, o CdR nem a validação em outra coorte.
Privacidade, calibração, orçamento de revisão e confirmação dos vínculos precisam
ser avaliados antes da incorporação dos resultados à rotina do serviço.
