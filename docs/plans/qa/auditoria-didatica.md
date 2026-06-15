# Auditoria Didática (DF-3) — `gzcmd_passo_a_passo.ipynb`

> Fase 4.3 · T4.3.7 · critério **DF-3** (Seção 5.2.1 do plano).
> Verificação automatizada: `tests/test_notebook_didatica.py` (TST4.3.a).

## Regra auditada
Antes de **cada** célula de código existe uma célula *markdown* não-vazia que
explica, em PT-BR, **o quê** e **o porquê** do passo seguinte. Nenhuma célula
de código pode ser "órfã" (sem contexto prévio).

## Método
Análise **estática** via `nbformat` (sem executar o kernel): percorre as 103
células do notebook na ordem e verifica, para cada célula `code` substantiva
(fonte não-vazia), que a célula **imediatamente anterior** é `markdown`
não-vazia. Conjunto de exceções explícito (`EXCECOES_ORFAS`) — atualmente
**vazio**.

## Resultado
**APROVADO.** As 37 células de código do notebook são **todas** imediatamente
precedidas por uma célula markdown explicativa. **Zero** células órfãs; **zero**
ocorrências de código consecutivo. Não foi necessária nenhuma exceção.

Mapeamento (índice da célula de código → markdown imediatamente antes):

| Seção | Células de código | Cobertura |
|-------|-------------------|-----------|
| 4. Setup | 5, 7, 9 | md 4/6/8 ✅ |
| 5. Primeiro olhar | 11 | md 10 ✅ |
| 6. Herói | 14 | md 13 ✅ |
| 7. Carga + FE | 17, 19, 21, 23, 25 | md 16/18/20/22/24 ✅ |
| 8. Bandas | 28, 30, 32, 34 | md 27/29/31/33 ✅ |
| 9. Calibração | 39, 42, 44, 46, 50 | md 38/41/43/45/49 ✅ |
| 10. Guardrails | 54, 56, 58 | md 53/55/57 ✅ |
| 11. Triagem | 63, 66, 68, 70, 72 | md 62/65/67/69/71 ✅ |
| 12. Reconciliação | 76, 78 | md 75/77 ✅ |
| 13. Avaliação held-out | 83, 85, 87, 90, 92 | md 82/84/86/89/91 ✅ |
| 14. Revisão LLM | 96, 98, 100 | md 95/97/99 ✅ |

Além disso, o notebook **abre** com narrativa markdown (células 0–4) antes da
primeira célula de código (verificado por teste de sanidade dedicado).

## Andaime didático (Seção 6.5 / CA-G9)
Cada seção segue o ciclo **objetivo de aprendizagem → intuição → ação (código)
→ recap**, com o **exemplo-fio-condutor** ("herói" `zona_cinzenta`) reaparecendo
em todos os estágios (cards nas seções 6, 7.4, 8.3, 9.8, 10, 11, 14.3).

## Conclusão
DF-3 **satisfeito** e protegido por teste automatizado que roda na suíte padrão
(`pytest`), prevenindo regressão em futuras regenerações do notebook.
