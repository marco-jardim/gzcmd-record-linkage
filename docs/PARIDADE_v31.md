# Paridade da política v3.1 no pacote 0.5.0

## Fonte e escopo

Portabilidade da política Bayesiana de quatro ações do motor de pesquisa
`D:/git/phd-research/gzcmd/`. A política (motor, guardas, orçamentos) foi
portada em 0.4.0 a partir do HEAD `59272a7edeaffb2606bd48e412fb865c708f9020`
(consulta de 04/11/2026). Para a versão 0.5.0 a referência foi atualizada para o
commit `08befb1` (ramo `pq/slides-defesa-rev2`), consultado em 13/11/2026.
Nenhum arquivo-fonte desse repositório foi editado para esta portabilidade.

SHA-256 dos arquivos de referência consultados (incluindo finais de linha):

| Arquivo | SHA-256 |
| :--- | :--- |
| `gzcmd_v3_policy_engine.py` | `c9a35191db3d4ec689720b3a91722b93f5a65bc8cb8affd9f90dc84e22f4a666` |
| `guardrails.py` | `c23d5d5dc3087daeca7b5ade3245b57f19b04209b3d75e47d84092ad4210a850` |
| `reliability.py` (commit `08befb1`) | `EC08FC224D0AD66177135EF1C9BCC115FF9FA86655C90A07AE68DE0515AD6D31` |
| `error_rates_by_band_v4.yaml` (commit `08befb1`) | `0C48024B94DE9BD2C3299FD0C21640A1E8A7F0733A424C59D6250BA30AE778B1` |

Os contratos foram conferidos em `tests/test_m1_m2_m6.py` da referência e em
`M1_M2_M6.md`: quatro perdas, prioridade estável por ganho, limites humanos e
LLM, guardas probabilísticas e preservação histórica. A alternativa NP de
`test_np_policy.py`, os modelos OOF e o restante da cadeia experimental
não integram a portabilidade de 0.4.0 (o carregamento de taxas medidas veio em 0.5.0, ver abaixo). Paridade de política exige as mesmas entradas,
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
| `reliability.py` (`load_band_rates`) | `reliability.py` | Porte do bloco `claude-haiku-5-5` para prompt v4/dossiê `sem_mo`: taxas medidas só com n >= 10, senão valores inline fixos, incluindo `e_unsure` |
| `error_rates_by_band_v4.yaml` | `error_rates_by_band_v4.yaml` | Cópia byte a byte (Claude Haiku 5.5; 1774 e 459 decisões), sem cabeçalho; SHA-256 na tabela acima; teste `test_packaged_v4_yaml_is_verbatim_thesis_copy` compara os bytes quando a tese está disponível |
| `gzcmd_v31_p_config.yaml` | `gzcmd_v31_config.yaml` | Perfil v3.1/P empacotado, acessível por `packaged_config_path("v3.1")`; declara `estimated_rates_path` |
| `scripts/fusao_filas.py` (desempate de `rrf` e `round_robin`) | `ranking.py` | Desempate padrão da tese: maior escore e, nos empates, menor índice (`review_queue`: menor `pair_id`); `tie_break="random"` preserva a permutação com semente |
| `runner.build_engine_from_config` (taxas medidas) | Mesmo nome | v3.1 com `estimated_rates_path` usa as taxas medidas (mudança incompatível em 0.5.0); v3 e o override histórico nunca leem o arquivo |

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

## Itens da referência não portados

Conferidos na referência `08befb1` e deliberadamente fora do pacote 0.5.0:

- política NP (`np_policy.py` e `_triage_np`);
- `arranjo_corte.py` (ASR);
- calibração isotônica OOF;
- lista de atributos permitidos (*allowlist*) e `--policy-version` do commit
  `f566368` da tese;
- `subamostragem.py`;
- `anthropic_transport.py`.

Para bandas sem decisões, `e_unsure` segue a tese (valor inline), diferente do
notebook tutorial, que usa a taxa do modo.

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
as taxas medidas só são lidas quando o YAML v3.1 declara `estimated_rates_path` (perfil `gzcmd_v31_config.yaml`). A configuração
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
