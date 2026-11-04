# Paridade da política v3.1 no pacote 0.4.0

## Fonte e escopo

Portabilidade da política Bayesiana de quatro ações do motor de pesquisa
`D:/git/phd-research/gzcmd/`, consultado em 04/11/2026, HEAD
`59272a7edeaffb2606bd48e412fb865c708f9020`. Nenhum arquivo-fonte desse repositório
foi editado para esta portabilidade.

SHA-256 dos arquivos de referência consultados (incluindo finais de linha):

| Arquivo | SHA-256 |
| :--- | :--- |
| `gzcmd_v3_policy_engine.py` | `c9a35191db3d4ec689720b3a91722b93f5a65bc8cb8affd9f90dc84e22f4a666` |
| `guardrails.py` | `c23d5d5dc3087daeca7b5ade3245b57f19b04209b3d75e47d84092ad4210a850` |

Os contratos foram conferidos em `tests/test_m1_m2_m6.py` da referência e em
`M1_M2_M6.md`: quatro perdas, prioridade estável por ganho, limites humanos e
LLM, guardas probabilísticas e preservação histórica. A alternativa NP de
`test_np_policy.py`, os modelos OOF e o carregamento de taxas empíricas externas
não integram esta portabilidade. Paridade de política exige as mesmas entradas,
custos e taxas, não significa reproduzir toda a cadeia experimental da tese.

## Mapeamento função a função

| Referência em `gzcmd/` | Pacote em `src/gzcmd_record_linkage/` | Contrato |
| :--- | :--- | :--- |
| `gzcmd_v3_policy_engine.Costs` | `gzcmd_v3_policy_engine.Costs` | Custos FP, FN, LLM e humano; `total_loss` |
| `HumanError.error_loss`, `HumanError.loss` | Mesmos nomes | Erro humano esperado e custo direto |
| `HumanBudget`, `Budget`, `LLMError` | Mesmos nomes, também exportados por `policy_engine` | Consumo persistente e taxa de abstenção `e_unsure` |
| `PolicyEngineV3.__init__` | Mesmo nome | `policy_version` validado; padrão v3; erro humano v3 zero |
| `PolicyEngineV3._losses` | Mesmo nome | Mesmas quatro parcelas na v3.1; v3 mantém a expressão histórica sem abstenção |
| `PolicyEngineV3.triage` | Mesmo nome | Despacho por versão; colunas históricas v3 mantidas |
| `PolicyEngineV3._triage_bayes` | Mesmo nome | Portabilidade direta da alocação e fallback, sem NP |
| `guardrails.apply_guardrails` | Mesmo nome | Retorno antecipado v3.1 antes de regras temporais/homonímia |
| `bands.BandAssigner.assign` | Mesmo nome, já existente | Bandas e inclusividade iguais; nenhuma mudança |
| `config.load_config` | Mesmo nome | `guardrails.policy_version` omitido implica v3, não v3.1 |
| `runner.build_engine_from_config` | Mesmo nome | Custos, erros, dois orçamentos; v3 explícita em YAML v3.1 restaura custo LLM 0,25 |
| `runner.run_v3` | Mesmo nome; alias em `runner_api` | Encaminha versão; v3 explícita em YAML v3.1 restaura guardas 10/3; resumo inclui versão e orçamento humano |
| `cli.build_parser`, `cli.main` | Mesmos nomes | `run --policy-version {v3,v3.1}`; omissão consulta YAML |

Validação de probabilidades v3.1 antecede o cálculo de perdas no pacote para
evitar avisos numéricos com infinito; a referência rejeita a mesma entrada no
início de `_triage_bayes`. A saída para probabilidades válidas não é alterada.
`Costs.human_review` é um campo real do dataclass, como na referência; participa
de `asdict`, `replace`, igualdade e representação. O helper `nb_helpers.build_engine`
preserva esse custo ao explorar outros custos. A única alteração em teste
preexistente inclui o quarto campo na asserção de `asdict`, acompanhando o
contrato público 0.4.0, sem mudar as ações históricas.

A referência atual soma `e_unsure*L_h(p)` também na v3 (com erro humano zero,
mas custo humano ainda presente). O pacote deliberadamente ignora `e_unsure`
na v3 para preservar a expressão histórica da versão 0.3.0. Por isso a paridade
v3 usa abstenção zero; a comparação v3.1 cobre também abstenção 0,2. Ambas as
entradas do pacote, YAML e construtor `LLMError`, validam `e_unsure` em [0,1].
Outra diferença intencional é a ordem das colunas: a referência insere
`loss_match` e `loss_nonmatch` antes de `loss_llm`; o pacote só as acrescenta
ao entrar na v3.1, após `evr`, preservando as colunas históricas v3. A comparação
externa alinha as colunas pelo nome antes de conferir os valores.

## Bandas, guardas e decisão

As bandas continuam `[0,5)`, `[5,6)`, `[6,7)`, `[7,8)`, `[8,9)` e `[9,999]`.
Na v3.1, nota < 5 recebe `ALWAYS_NONMATCH`; nota >= 9 e p >= 0,9 recebe
`ALWAYS_MATCH`, mesmo sem nascimento ou município exatos. Nota alta com p < 0,9
fica sem guarda e recebe motivo `high_probability_inconsistent`. Temporalidade
e homonímia só são aplicadas na v3. Nenhum par da área cinza recebe uma dessas
guardas v3.1. Nota não finita, índice duplicado, p fora de [0,1] e
`ALWAYS_NONMATCH` fora da banda baixa são recusados pelo motor v3.1.

Fora das guardas, com p = `p_cal`:

- MATCH: `(1-p)*C_fp`;
- NONMATCH: `p*C_fn`;
- DEFERRED: `C_h + p*e_fn_h*C_fn + (1-p)*e_fp_h*C_fp`;
- LLM_REVIEW: `C_llm + (1-p)*e_fp*C_fp + p*e_fn*C_fn + e_unsure*L_h(p)`.

Empates preferem MATCH, NONMATCH, humano, LLM. Revisão LLM só vence com ganho
estritamente positivo sobre a melhor alternativa não LLM. A ordenação é
estável. LLM nominal tem prioridade; depois, vagas humanas são distribuídas
por ganho contra as alternativas ainda disponíveis. Sem capacidade humana,
usa LLM se houver ganho e vaga, senão a melhor ação automática. Caps legados e
FORCE_REVIEW não sobrepõem o argmin. Trata-se da mesma alocação gulosa da
referência, não de otimização global conjunta. O orçamento humano limita
DEFERRED direto; não reserva antecipadamente vagas para futuras abstenções LLM.

## Configuração e limites da paridade

O YAML distribuído conserva `llm_review: 0.25` e as taxas fixas da versão 0.3.0.
Assim, selecionar v3.1 muda a política, mas não troca silenciosamente custos,
calibrador ou taxas. O cenário atual da tese usa custo LLM 0,005, humano 1 e
erros humanos 0,05. Para compará-lo, forneça os mesmos custos e taxas no YAML;
as taxas externas da referência não são buscadas pelo pacote. A configuração
de abstenção usa `e_unsure` dentro de cada taxa por banda.
Quando o YAML declara v3.1 e há override explícito para v3, o runner restaura
custo LLM 0,25 e guardas 10/3, como o `legacy_override` da referência. Esse
override não altera os demais custos, taxas ou calibração. Um YAML que já
declara v3 conserva seus custos e guardas configurados.

O runner mantém seus métodos de calibração anteriores. Treinar Platt na entrada
não constitui avaliação independente, embora seja mantido para reproduzir o
perfil histórico. Esta entrega não porta o treinamento OOF aninhado. `eval`
continua explicitamente v3 porque suas métricas esperadas têm três ações.

## Evidência reproduzível

```console
python -m pytest tests/test_policy_v31.py -o addopts= -q
python -m pytest -q
```

`GZCMD_REFERENCE_ROOT` pode apontar para outra cópia do repositório de pesquisa;
o padrão é `D:/git/phd-research`. A fixture importa via `sys.path`. Sem essa
cópia, somente a comparação externa é pulada, com motivo explícito. Regressões,
guardas, CLI e quatro ações continuam testados em qualquer instalação.

O teste de paridade cruza 128 combinações de nota e p em quatro contextos de
subescores/datas (512 pares por cenário), seis bandas,
ambos os modos, quatro combinações de orçamentos e abstenção 0/0,2. Compara
bandas, guardas, motivos, todas as colunas comuns e consumo dos dois orçamentos.
Há 32 cenários parametrizados, dos quais 16 exercitam v3.1; v3 sempre usa
abstenção zero. Além das datas incompatíveis, há datas compatíveis com
homonímia, concordância forte e subescores neutros. As asserções exigem a
presença das guardas altas/baixas, temporalidade, homonímia, caps e seleção LLM;
FORCE_REVIEW também é injetado fora dos caps e com EVR negativo.

Os YAML em `tests/fixtures/` são snapshots: configuração do tag `v0.3.0`
(`ad823eb`) e configuração v3.1 distribuída pelo motor de referência na data
desta revisão. O teste de override compara todas as colunas e perdas,
exatamente, tanto na API quanto na CLI, para os dois modos, sem depender de
histórico Git ou arquivos externos na CI.

As regressões locais fixam SHA-256 da sequência ordenada de ações do CSV
versionado `data/synthetic/comparador_sintetico.csv` (607 potenciais pares),
com Platt e configuração distribuída. São testes sintéticos, não resultados
de desempenho no serviço:

| Modo | Política | MATCH | NONMATCH | LLM_REVIEW | DEFERRED |
| :--- | :--- | ---: | ---: | ---: | ---: |
| vigilancia | v3 | 194 | 123 | 290 | 0 |
| confirmacao | v3 | 10 | 173 | 424 | 0 |
| vigilancia | v3.1 | 254 | 242 | 0 | 111 |
| confirmacao | v3.1 | 113 | 242 | 143 | 109 |

Além das regressões, nesta portabilidade o motor do commit base `089d914` foi
carregado separadamente e comparado ao novo motor no mesmo fluxo Platt do CSV,
com `pandas.testing.assert_frame_equal(check_exact=True)`: todas as colunas
históricas permaneceram exatamente iguais nos dois modos. A única alteração
em teste existente foi a asserção de serialização dos quatro custos descrita
acima; nenhum teste foi removido ou pulado para contornar regressões.
