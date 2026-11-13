# GZ-CMD-Record-Linkage

Componentes Python para pós-processamento de potenciais pares de *linkage* e
encaminhamento da área cinza para revisão. O pacote recebe escores produzidos por
um comparador; não executa blocagem nem gera potenciais pares a partir de duas bases.

## Nome e origem

GZ-CMD tem como nome histórico "Grey-Zone: Calibrated Mixture-of-experts +
Conformal Deferral + Constrained Matching". Os termos não correspondem todos a
mecanismos implementados:

- **Calibrated**: calibração de probabilidades por Platt (`fit_platt`,
  `fit_anchor_platt`) ou pela calibração sigmoide interna do `GZCMDClassifier`.
  A probabilidade isotônica fora da amostra (OOF) usada na tese é portada por
  `gzcmd_record_linkage.oof` (`nested_oof_probabilities`, ver "Filas de
  contexto (C/CM)") e passada ao motor de política (`PolicyEngineV3`) na coluna
  `p_cal`; fora desse módulo, o pacote só calibra por Platt (acima).
- **Mixture-of-experts**: uso livre do termo para a articulação de regras,
  classificador e revisor sob uma política, sem mecanismo aprendido de seleção
  entre especialistas.
- **Conformal Deferral**: o encaminhamento à revisão humana decorre de
  probabilidades calibradas, custos e orçamento, não de predição conformal;
  não há quantis de não conformidade nem garantia de cobertura.
- **Constrained Matching**: guardas determinísticas e limites de revisão, sem
  cardinalidade N:1 nem resolução de conflitos por margem de probabilidade.

## Versão e escopo

- Distribuição PyPI: `gzcmd-record-linkage==0.4.2`
- Importação Python: `gzcmd_record_linkage`
- Licença: [MIT](LICENSE)
- Python: 3.10 ou superior
- Versão no repositório: **0.5.0** (ainda não publicada)
- Versão mais recente no PyPI: **0.4.2** (publicadas: 0.1.0, 0.3.0, 0.4.0, 0.4.1 e 0.4.2)
- Políticas: **v3** (padrão, três ações) e **v3.1** (quatro ações, desde 0.4.0)

A versão do pacote, a versão da política (v3 ou v3.1) e a versão do depósito
de dados (3.0.0) não são intercambiáveis. A instalação PyPI abaixo fornece
0.4.2, com suporte às políticas v3 e v3.1. A paridade da política não inclui
automaticamente os modelos, as probabilidades OOF nem as taxas empíricas dos
experimentos da tese.

## Instalação

```console
python -m pip install gzcmd-record-linkage==0.4.2
```

A versão 0.5.0 (perfil v3.1 empacotado, `estimated_rates_path` e utilitários de
fila; ver `CHANGELOG.md`) está apenas no repositório e ainda não foi publicada no
PyPI; até a publicação, instale-a a partir do código-fonte.

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
Desde 0.5.0, o desempate padrão (`tie_break="index"`) segue a regra da tese:
maior escore primeiro e, nos empates, o menor índice posicional (em
`review_queue`, o menor `pair_id`); nunca os rótulos. Para o comportamento
anterior, a permutação aleatória com semente fixa (`seed=42` por padrão), passe
`tie_break="random"`. A reprodução exige IDs posicionais e, no modo aleatório,
sementes iguais. `validate_order(order, n)`
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

## Filas de revisão

Além de `rrf`, `interleave`, `review_queue` e `ThompsonQueue`, o módulo
`gzcmd_record_linkage.ranking` oferece auxiliares para ordenar e medir filas:

- `order_by_score(escores)`: ordem de IDs posicionais, maior escore primeiro;
- `top_k_cut(ordem_ou_df, k)`: rotula os `k` primeiros como `examined` e o
  restante como `not_examined`. Os itens **não examinados** não são decisões de
  não vínculo e não devem ser registrados como tal;
- `effort_curve(ordem, y)`, `reviews_to_target(curva, meta)` e
  `recovered_at_budget(curva, k)`: vínculos recuperados por revisão, revisões
  necessárias para uma meta e vínculos recuperados com orçamento `k`;
- `tie_ids`: IDs estáveis (por exemplo, `pair_id`) usados no desempate.

O desempate padrão é a regra da tese (maior escore, depois menor índice ou
`pair_id`); use `tie_break="random"` para a permutação com semente fixa.

Nenhuma fusão é recomendada como padrão: a tese escolheu a F4 depois de ver os
resultados (*post hoc*), e as curvas de esforço devem ser avaliadas em dados
independentes. Ficam no código do chamador as questões operacionais: quem
revisa, a capacidade por turno, a persistência da fila, a coleta de rótulos para
`ThompsonQueue.update` e a escolha entre O3, F4 e F3 para cada coorte.

### Fusão de filas

O módulo `gzcmd_record_linkage.fusion` reproduz, com paridade exata, as fusões
da tese (`scripts/fusao_filas.py`): `THESIS_FUSIONS` (somente leitura) define
F1/F2 (intercalação), F3/F4 (RRF, `k=60`) e A1/A2 (Thompson com
`Beta(1, 1)`, janela de 200 revisões ou sem janela). A ordem dos braços é a
ordem de desempate e foi copiada da tese. `fuse` monta uma fila fundida,
`simulate_thompson` repete uma fila Thompson sobre rótulos conhecidos,
`queue_metrics` mede uma fila e `compare_queues` devolve uma linha por fila base
e por fusão.

```python
from gzcmd_record_linkage.fusion import THESIS_FUSIONS, compare_queues, fuse
from gzcmd_record_linkage.llm_dossier import make_pair_id

# Ordens completas de IDs posicionais 0..n-1 (melhor primeiro), uma por fila.
filas = {"O1": ordem_o1, "O3": ordem_o3, "O3_MACD": ordem_macd}
# Desempate da tese: hash do par crescente (SHA-256 de COMPREC|REFREC|PASSO).
tie_ids = [make_pair_id(c, r, p) for c, r, p in zip(comprec, refrec, passo)]

fila_f4 = fuse(filas, THESIS_FUSIONS["F4"], tie_ids=tie_ids)
tabela = compare_queues(
    filas, rotulos, targets=(9, 23, 36), top_k=(125, 3040),
    auc_horizon=3040, tie_ids=tie_ids,
)
```

`compare_queues` calcula uma vez as fusões determinísticas; A1 e A2 rodam uma
simulação por semente (`adaptive_seeds`, padrão `0..199`) e informam média e
P10/P90 (quantil linear) de cada métrica, não as métricas de uma curva média.
Colunas: `queue`, `kind`, `method`, `runs` e, para cada métrica
(`reviews_to_<meta>`, `recovered_at_<k>`, `auc_recovery`), a média em `<métrica>`
e `<métrica>_p10`/`<métrica>_p90` (iguais à média quando há uma só execução).
Uma meta inalcançável vale `None` (NaN na tabela). `auc_recovery` é a integral
em degraus de `C(k)/total` para `k=0..H-1`, com `C(0)=0`.

Ressalvas:

- A F4 foi escolhida *post hoc* entre seis variantes, sem correção para
  múltiplas comparações; nenhuma fusão é padrão recomendado.
- A1 e A2 leem os rótulos: servem só para simulação e avaliação. Em operação,
  use `ThompsonQueue` com `update()` à medida que os revisores respondem.
- A robustez por *bootstrap* tem ressalvas próprias (próxima seção).

### Robustez por bootstrap (REFREC)

`gzcmd_record_linkage.robustness` porta `scripts/fusao_robustez.py`.
`bootstrap_fusion_differences` reamostra os grupos `REFREC` com reposição,
expande as multiplicidades em cópias contíguas, reordena todas as filas base
(maior escore primeiro; empate pelo `tie_ids`, ou pela posição se ele for
omitido, como em `queue()` da tese) e refaz as fusões antes de medir as
revisões até as metas. A chamada é
`bootstrap_fusion_differences(scores, labels, groups, *, fusions, comparisons,
n_boot=2000, seed=20261107, ...)`, com `fusions` um mapa de `FusionSpec`
determinísticas e `comparisons` uma lista de pares `(fusão, base)`. Devolve um
`BootstrapResult` com as réplicas (`replicates`), o resumo (`summary`:
diferença mediana, IC95%, redução percentual e fração de réplicas em que a
fusão vence), `n_boot`, `n_valid`, `n_excluded` e `seed`. Réplicas sem positivos
são excluídas e contadas (`n_excluded`); só as fusões determinísticas
(`interleave` e `rrf`) são aceitas, e as Thompson são recusadas com
`ValueError`. As metas de cada réplica seguem `replicate_targets` (regra
híbrida, ver as ressalvas). `subset_effort` mede o esforço
em um subconjunto (por exemplo, uma faixa de escore), como a tabela por faixa da
tese.

Ressalvas (tese, `estudo4_gzcmd.tex:225`): o resultado é condicional às
probabilidades OOF e à referência fixas; não inclui a incerteza do treinamento
nem o erro da revisão; e a F4 foi escolhida *post hoc*. O intervalo descreve a
variação entre grupos `REFREC`, não uma garantia sobre dados futuros.

### Filas de contexto (C/CM)

`gzcmd_record_linkage.context` porta as variantes de `scripts/contexto_obito.py`:
`context_features` deriva os atributos de contexto do grupo `REFREC`/`COMPREC`,
`context_queues(df, y, ...)` treina e ordena as filas `O3_contexto` (C) e
`O3_contexto_MACD` (CM) e devolve um `ContextQueues` (`probabilities`, `folds`,
`orders`, `rows`). `THESIS_CONTEXT_FUSIONS` contém só `F4_contexto` (RRF, k=60,
de `O1` e `O3_contexto`), aplicada por `context_fusions` sobre as ordens, entre
as quais a fila `O1` (por escore) é fornecida pelo chamador.
`gzcmd_record_linkage.oof.nested_oof_probabilities(x, y, ...)` porta
`gzcmd/ml_oof.py` (probabilidade isotônica OOF aninhada; devolve as
probabilidades e a dobra externa de cada linha). É um artefato de avaliação
supervisionado, que usa os rótulos, não um modelo implantável. A ordem das nove
colunas `macd_*` (`MACD_COLUMNS`) segue a tese (alfabética) e a derivação pelo
carregador do pacote reproduz a matriz da tese.

Uso **exploratório**, não é padrão: o contexto de `COMPREC` pode cruzar as
partições de `REFREC` na validação, de modo que o desempenho pode estar
otimista; nenhuma fila C/CM é recomendada sem avaliação em dados independentes.

## Política NP (Neyman-Pearson)

Alternativa à política de custo, portada de `np_policy.py` e `_triage_np` da
tese. Seleção: `policy_kind="np"` (argumento de `run_v3`/`build_engine_from_config`
ou `policy_kind: np` no modo do YAML), apenas no modo `confirmacao`, e apenas
sob a política v3.1; outros modos recusam NP (`ValueError`). Os limites vêm de
`NPConstraints` (argumento `np_constraints` de `run_v3`/`build_engine_from_config`
ou YAML em `np:` do modo): `max_fp_rate_auto=0.002`,
`max_fp_expected_total=None` (sem limite absoluto) e `max_fn_rate_auto=0.02`
(limite de `p_cal` para o NONMATCH automático). O motor levanta
`NPInfeasibleError` quando os MATCH obrigatórios das guardas tornam a restrição
de falsos positivos inviável (`motivo="guardas_fp"`) ou quando a capacidade de
revisão humana não cobre os pares pendentes (`motivo="orcamento_humano"`).

Como na tese, o NP é **rebaixado silenciosamente para custo** quando a versão
efetiva da política é v3 (inclusive o override `policy_version="v3"` sobre um
YAML v3.1) em `run_v3` e `build_engine_from_config`: nenhum aviso nem erro é
emitido e a execução usa a política de custo. Já `PolicyEngineV3` construído
diretamente com `policy_version="v3"` e `policy_kind="np"` levanta `ValueError`.
Confira a versão efetiva antes de interpretar um resultado como NP.
`np_evaluation.compare_confirmation` da tese não foi portado.

## Perfil v3.1 empacotado

`packaged_config_path("v3.1")` devolve o caminho do perfil v3.1 distribuído com o
pacote (`gzcmd_v31_config.yaml`); `packaged_config_path("v3")` devolve o perfil
v3 padrão. Esse perfil declara `llm_review.reliability.estimated_rates_path`,
apontando para `error_rates_by_band_v4.yaml` (também empacotado, Claude Haiku
5.5, 1774 e 459 decisões por modo). Com v3.1, as taxas por banda passam a ser as
medidas, apenas quando a banda tem n >= 10 decisões; caso contrário valem os
valores fixos inline, incluindo `e_unsure`. A política v3 (padrão) e o override
histórico `policy_version="v3"` usam sempre as taxas inline e nunca leem esse
arquivo.

```python
from gzcmd_record_linkage.config import load_config, packaged_config_path

cfg = load_config(packaged_config_path("v3.1"))
```

Mudança incompatível em 0.5.0: configurações v3.1 com `estimated_rates_path`
agora usam as taxas medidas por banda em vez das taxas inline. Consulte
`docs/PARIDADE_v31.md` e `CHANGELOG.md`.
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

Ao utilizar o pacote, cite a versão efetivamente usada (por exemplo, 0.4.2 publicada, ou
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
