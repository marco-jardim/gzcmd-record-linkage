"""Helpers do notebook didático: métricas de calibração, plots e stub de LLM.

Vive **fora** da biblioteca publicada (D7 / DEC-01). Importável nos testes via
``pythonpath`` configurado no ``pyproject.toml``.

Conteúdo planejado
-------------------
- ``expected_calibration_error`` / ``brier_score`` (DEC-08): métricas quantitativas
  de calibração, com testes próprios (Fase 2.4).
- ``llm_review_stub`` (R-05 / Fase 3.3): simulação determinística da revisão LLM,
  usando as taxas de erro por banda da config. SEM chamadas de rede.
- Funções de plotagem (matplotlib) com títulos/eixos/legendas em PT-BR.

Status: ESQUELETO (Fase 0.3). Implementação nas Fases 2.4 e 3.3.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "expected_calibration_error",
    "brier_score",
    "llm_review_stub",
]


def _as_pair(
    y_true: Sequence[float] | np.ndarray | pd.Series,
    p_pred: Sequence[float] | np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Converte entradas para arrays float 1-D alinhados e validados.

    Aceita listas, ``np.ndarray`` ou ``pd.Series``. Garante mesmo tamanho,
    não-vazio e ausência de NaN — pré-condições das métricas de calibração.
    """
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(p_pred, dtype=float).ravel()
    if y.shape != p.shape:
        raise ValueError(
            f"y_true e p_pred devem ter o mesmo tamanho: {y.shape} != {p.shape}"
        )
    if y.size == 0:
        raise ValueError("y_true e p_pred não podem ser vazios.")
    if np.isnan(y).any() or np.isnan(p).any():
        raise ValueError("y_true e p_pred não podem conter NaN.")
    return y, p


def expected_calibration_error(
    y_true: Sequence[float] | np.ndarray | pd.Series,
    p_pred: Sequence[float] | np.ndarray | pd.Series,
    *,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (binning explícito). Ver DEC-08 / Fase 2.4.

    Mede o desalinhamento médio entre a *confiança* prevista e a *acurácia*
    observada, agrupando as previsões em ``n_bins`` faixas de largura igual em
    ``[0, 1]``:

    ``ECE = soma_b (|B_b| / N) * |acc(B_b) - conf(B_b)|``

    onde, para cada bin ``b``, ``conf(B_b)`` é a média de ``p_pred`` e
    ``acc(B_b)`` é a fração empírica de positivos (``y_true``). Bins vazios são
    ignorados (contribuição zero), o que torna a métrica robusta.

    Parameters
    ----------
    y_true:
        Rótulos binários em ``{0, 1}``.
    p_pred:
        Probabilidades previstas em ``[0, 1]``.
    n_bins:
        Número de faixas de largura igual (padrão 10). Deve ser >= 1.

    Returns
    -------
    float
        ECE em ``[0, 1]`` (0 = perfeitamente calibrado).
    """
    if n_bins < 1:
        raise ValueError("n_bins deve ser >= 1.")
    y, p = _as_pair(y_true, p_pred)
    n_total = y.size
    # Arestas das faixas; usamos apenas as internas no digitize para obter
    # índices em [0, n_bins - 1]. Clip garante robustez a p ligeiramente
    # fora de [0, 1] (ex.: clip_max=0.999999 ou ruído numérico).
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        acc = float(y[mask].mean())
        conf = float(p[mask].mean())
        ece += (count / n_total) * abs(acc - conf)
    return float(ece)


def brier_score(
    y_true: Sequence[float] | np.ndarray | pd.Series,
    p_pred: Sequence[float] | np.ndarray | pd.Series,
) -> float:
    """Brier score = média((p - y)^2). Ver DEC-08 / Fase 2.4.

    É o erro quadrático médio entre a probabilidade prevista e o rótulo. Para
    um previsor constante ``p = c`` num conjunto com prevalência ``f`` de
    positivos, vale ``f*(c-1)^2 + (1-f)*c^2``; com ``c = f`` reduz a ``f*(1-f)``.
    """
    y, p = _as_pair(y_true, p_pred)
    return float(np.mean((p - y) ** 2))


#: Taxas de erro de fallback (banda ausente em ``error_rates_by_band``).
#: Espelham a banda mais pessimista da config (``low``): e_fp=0.10, e_fn=0.15.
_FALLBACK_ERROR_RATES: dict[str, float] = {"e_fp": 0.10, "e_fn": 0.15}


def llm_review_stub(
    df_review: pd.DataFrame,
    *,
    seed: int = 42,
    error_rates_by_band: Mapping[str, Mapping[str, float]] | None = None,
    target_col: str = "TARGET",
    band_col: str = "band",
) -> pd.Series:
    """Stub determinístico da revisão clerical/LLM (R-05 / Fase 3.3).

    **Isto é uma SIMULAÇÃO, não um LLM real.** Não há chamada de rede nem modelo
    de linguagem. O stub representa um revisor *quase-oráculo*: ele "enxerga" o
    rótulo verdadeiro (``target_col``) e devolve a decisão correta na maioria das
    vezes, mas **erra com taxas dependentes da banda** — exatamente o perfil de
    confiabilidade que a config atribui ao revisor (``e_fp``/``e_fn`` por banda).
    Serve para demonstrar, de forma reprodutível e offline, o *efeito* da etapa
    de revisão sobre as métricas finais, sem depender de uma API ao vivo.

    Mecânica (determinística por ``seed``):

    - Sorteia-se ``u ~ Uniforme(0, 1)`` para cada linha, **em ordem de linha**,
      com ``np.random.default_rng(seed)`` (uma única sequência → reprodutível).
    - Para uma banda ``b``, usa ``rates = error_rates_by_band[b]`` (ou o fallback
      ``{e_fp: 0.10, e_fn: 0.15}`` se a banda não estiver no mapa).
    - Se o rótulo verdadeiro é MATCH (``target == 1``): o revisor erra (diz
      ``NONMATCH``) quando ``u < e_fn``; senão acerta (``MATCH``).
    - Se o rótulo verdadeiro é NÃO-MATCH (``target == 0``): o revisor erra (diz
      ``MATCH``) quando ``u < e_fp``; senão acerta (``NONMATCH``).

    Parameters
    ----------
    df_review:
        Subconjunto de pares roteados a ``LLM_REVIEW``. Precisa conter
        ``target_col`` (rótulo verdadeiro, ``{0, 1}``) e ``band_col`` (nome da
        banda). A ordem das linhas determina o sorteio (reprodutibilidade).
    seed:
        Semente do gerador. Mesma semente + mesmo ``df_review`` → mesma saída.
    error_rates_by_band:
        Mapa ``{banda: {"e_fp": float, "e_fn": float}}`` (tipicamente
        ``cfg.llm_review.error_rates_by_band``). Se ``None``, usa o fallback
        para todas as linhas.
    target_col, band_col:
        Nomes das colunas de rótulo verdadeiro e banda.

    Returns
    -------
    pd.Series
        Decisões em ``{"MATCH", "NONMATCH"}``, dtype ``string``, alinhadas ao
        ``df_review.index``. Série vazia se ``df_review`` for vazio.
    """
    if target_col not in df_review.columns:
        raise KeyError(f"df_review precisa da coluna de rótulo '{target_col}'.")
    if band_col not in df_review.columns:
        raise KeyError(f"df_review precisa da coluna de banda '{band_col}'.")

    rates_by_band = error_rates_by_band or {}
    n = len(df_review)
    if n == 0:
        return pd.Series([], dtype="string", index=df_review.index)

    rng = np.random.default_rng(seed)
    u = rng.random(n)  # uma sequência, em ordem de linha → determinística

    targets = df_review[target_col].to_numpy()
    bands = df_review[band_col].to_numpy()

    decisions: list[str] = []
    for i in range(n):
        rates = rates_by_band.get(str(bands[i]), _FALLBACK_ERROR_RATES)
        e_fp = float(rates["e_fp"])
        e_fn = float(rates["e_fn"])
        if int(targets[i]) == 1:
            decisions.append("NONMATCH" if u[i] < e_fn else "MATCH")
        else:
            decisions.append("MATCH" if u[i] < e_fp else "NONMATCH")

    return pd.Series(decisions, index=df_review.index, dtype="string")
