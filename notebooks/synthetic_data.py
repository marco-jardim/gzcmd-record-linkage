"""Gerador de dataset sintético para o notebook didático "GZ-CMD Passo a Passo".

Este módulo vive **fora** de ``src/gzcmd_record_linkage/`` (diretriz D7 / DEC-01):
ele é um artefato de *apresentação*, não faz parte da biblioteca publicada. Os
testes o importam graças a ``pythonpath = ["src", "notebooks"]`` no ``pyproject.toml``.

Objetivo
--------
Produzir um CSV no formato exato que ``load_comparador_csv`` ingere (colunas com
prefixo OpenRecLink, ``sep=';'``, ``decimal=','``), rotulado e **determinístico**,
cobrindo todos os edge cases do plano (Seção 4.1).

Anti-circularidade (DEC-06)
---------------------------
A posterior verdadeira ``p*(x)`` é definida **explicitamente** sobre ``nota_final``
(o observável 1-D que o Platt do código realmente usa) e o rótulo é amostrado como
``TARGET ~ Bernoulli(p*)``. Assim ``nota_final`` é gerada ANTES de ``TARGET`` (não é
função determinística do rótulo), o que garante sobreposição de classes (AUC < 1) e
torna a recuperação de ``p*`` pelo Platt uma *prova*, não uma tautologia. A coluna
``p_true`` é de **validação** e NUNCA é escrita no CSV de entrada do pipeline.

Status: ESQUELETO (Fase 0.3). Implementação na Fase 1.1.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

__all__ = [
    "SyntheticDataset",
    "generate_comparador",
    "to_comparador_csv",
    "group_aware_split_indices",
]


@dataclass(frozen=True)
class SyntheticDataset:
    """Resultado do gerador.

    Attributes
    ----------
    frame:
        DataFrame com as colunas cruas (entrada do pipeline) — formato do loader.
    p_true:
        Série com a posterior verdadeira ``p*(x)`` por par (coluna de validação,
        NÃO-entrada do pipeline; alinhada por índice ao ``frame``).
    meta:
        Metadados do gerador (seed, n_pairs, match_ratio efetivo, coef. da p*, etc.).
    """

    frame: pd.DataFrame
    p_true: pd.Series
    meta: dict


def generate_comparador(
    *,
    seed: int = 42,
    n_pairs: int = 600,
    match_ratio: float = 0.5,
    scenarios: list | None = None,
) -> SyntheticDataset:
    """Gera o dataset sintético (núcleo). Ver Fase 1.1.

    Parameters
    ----------
    seed:
        Semente para reprodutibilidade total.
    n_pairs:
        Número de pares (DEC-04: alvo 300–1000).
    match_ratio:
        Fração-alvo de positivos amostrada via Bernoulli(p*) (aproximada).
    scenarios:
        Cenários narrativos nomeados a injetar (Fase 1.2).
    """
    raise NotImplementedError("Implementado na Fase 1.1.")


def to_comparador_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    """Escreve ``frame`` no formato do loader (``sep=';'``, ``decimal=','``).

    A coluna de validação ``p_true`` (se presente) é removida antes de escrever:
    ela nunca entra no CSV consumido pelo pipeline.
    """
    raise NotImplementedError("Implementado na Fase 1.1.")


def group_aware_split_indices(
    frame: pd.DataFrame,
    *,
    split_by: str = "comprec",
    test_size: float = 0.3,
    seed: int = 42,
    group_stratify: bool = True,
):
    """Split treino/teste reprodutível e *group-aware* (T1.1.6; espelha ``splitting``).

    Returns ``(train_idx, test_idx)`` como arrays numpy de índices.
    """
    raise NotImplementedError("Implementado na Fase 1.1.")
