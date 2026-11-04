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
| `runner.build_engine_from_config` | Mesmo nome | Custos, erros, abstenção, dois orçamentos e override explícito |
| `runner.run_v3` | Mesmo nome; alias em `runner_api` | Encaminha versão às guardas e motor; resumo inclui versão e orçamento humano |
| `cli.build_parser`, `cli.main` | Mesmos nomes | `run --policy-version {v3,v3.1}`; omissão consulta YAML |

Validação de probabilidades v3.1 antecede o cálculo de perdas no pacote para
evitar avisos numéricos com infinito; a referência rejeita a mesma entrada no
início de `_triage_bayes`. A saída para probabilidades válidas não é alterada.
`Costs` aceita custo humano no construtor e o expõe por `human_review`, mas
preserva a serialização histórica de três campos por `dataclasses.asdict`,
exigida pelos testes e exemplos do notebook. Para serializar os quatro custos,
inclua explicitamente a propriedade `human_review`.

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

O teste de paridade cruza 128 combinações de nota e p por cenário, seis bandas,
ambos os modos, quatro combinações de orçamentos e abstenção 0/0,2. Compara
bandas, guardas, motivos, todas as colunas comuns e consumo dos dois orçamentos.
Há 32 cenários parametrizados, dos quais 16 exercitam v3.1; v3 sempre usa
abstenção zero, pois a versão histórica não tinha esse parâmetro.

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
históricas permaneceram exatamente iguais nos dois modos. Os testes existentes
não foram alterados.
