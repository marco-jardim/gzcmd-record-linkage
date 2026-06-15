# Senior QA Review — WAVE 0 / Fase 0.2 (Validação de Contrato de API)

> Revisor: orquestrador `[self:opus]` (Opus faz o QA localmente, D6).

## Escopo revisado
- `tests/test_api_contract.py` (novo, 642 linhas, 7 testes)
- `docs/plans/qa/contrato-api.md` (contrato verificado por leitura de código + execução)

## Método
Contrato extraído por **leitura linha-a-linha** de `loader.py`, `bands.py`,
`calibration.py`, `guardrails.py`, `gzcmd_v3_policy_engine.py`, `runner.py`,
`eval.py`, `splitting.py`, `metrics.py` e validado por **execução real** no
canário (`test_api_contract.py`).

## Avaliação por eixo

| Eixo | Resultado |
|------|-----------|
| 1. Correção funcional | ✅ Cada estágio (load→bands→calibração→guardrails→triage) exercitado uma vez; colunas/tipos batem com o contrato. |
| 2. Cobertura | ✅ 7 testes cobrindo loader (engenheiradas + MACD on/off), bands (sem mutação), calibração (stub+platt, sem mutação), guardrails, triage (cópia), p_cal em [0,1]. |
| 3. Determinismo | ✅ Fixture fixo de 12 linhas; Platt determinístico; sem rede/relógio. |
| 4. Fidelidade científica | ✅ Usa `clip_min/clip_max` e hiperparâmetros do Platt vindos da config real. |
| 5. Qualidade de código | ✅ `ruff check` + `format --check` limpos; tipagem; `pathlib`; sem `type: ignore`. |
| 7. Higiene de dados | ✅ Fixture sintético em memória, IDs fictícios; CSV `;`/`,`. |
| 8. Não-poluição da lib | ✅ `src/` e `pyproject.toml` intactos. |

## Achados e correções

- ⚠️ **F-0.2-1 — Divergências código×plano/config detectadas e documentadas**
  (não são bugs nossos; são fatos a ensinar no notebook):
  1. `triage` **retorna cópia** (`out=df.copy()`), não muta o df de entrada
     (plano §2.2 dizia "muta"). → corrigido na doc; notebook usará `df.copy()` por modo.
  2. Guardrail `ALWAYS_MATCH` exige `nota≥10` + nome perfeito + `DTNASC dt iguais≥1`
     + `CODMUNRES≥1` (não só `nota≥9`). → cenário "âncora high" do gerador deve
     satisfazer todas as condições (anotado para Fase 1.2).
  3. R-11 confirmado: `anchor_platt`/`by_band` da config **não** implementados;
     Platt é global. → ensinar o código; config = roadmap.
  Todas registradas em `contrato-api.md` (seções ⚠️).
- 🟢 **F-0.2-2 — Canário não asserta disparo de guardrail.** O teste checa apenas
  o subconjunto de valores válidos. O disparo efetivo (ALWAYS_MATCH/temporal/low/
  homonímia) é validado nas Fases 1.2/2.5 com cenários nomeados. **Aceito** como
  escopo correto da fase (contrato = colunas/tipos). Backlog coberto adiante.
- 🟢 **F-0.2-3 — `CalibrationMethod` (valores aceitos por `evaluate_v3_*`) ainda
  não confirmado.** Anotado em `contrato-api.md §7` para verificação na Fase 3.2,
  antes do uso da Rota B. Não-bloqueante agora.

## R-04 (de-para config×loader)
✅ **RESOLVIDO.** De-para escrito em `contrato-api.md`. `run_v3` ponta-a-ponta não
depende das features ausentes da config (Platt usa só `nota_final`). O gerador
emitirá as RAW corretas (incl. os 6 nomes corretos de ENDERECO).

## Evidência
- `pytest tests/test_api_contract.py`: **7 passed**.
- Suíte completa: **44 passed, 1 warning** (37 anteriores + 7 contrato).
- `ruff check`/`format --check`: limpos.

## Critério de Aceitação / DoD da fase
✅ I/O real de todos os estágios documentado e batendo com o contrato; R-04 resolvido;
`docs/plans/qa/contrato-api.md` produzido; testes de contrato verdes.

**Veredito:** Fase 0.2 **APROVADA**. Sem 🔴/🟡 pendentes (divergências são
documentais e viram material didático; 🟢 endereçados em fases indicadas).
