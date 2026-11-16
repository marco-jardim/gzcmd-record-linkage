"""Experimentos pré-registrados de engenharia de filas de revisão.

Implementa a seção 4 e 5 de ``docs/engenharia_de_filas.md`` (hipóteses H1-H12,
critérios de decisão e protocolo). Experimental e offline: sem rede e sem
chamadas de API. O humano e o LLM são simulados (sintético) ou reaplicados de um
fixture local (dados reais). ``PAR`` entra **apenas** como verdade do humano
simulado e das métricas; nunca como atributo de nenhuma estratégia. Só
agregados são escritos.

Estratégias (``B`` = nº de decisões humanas):

* S0: ordem da régua (``nota final``), do topo para baixo.
* S1: ``SelfCalibratingSession`` atual (configuração padrão, modo humano; o
  humano simulado decide todos os itens do portão).
* S2: protótipo (princípios 2-5): regressão logística com a régua como
  covariável (``logit p0`` com prior centrado no valor sem rótulos) e demais
  coeficientes ~ N(0, sigma^2); reajuste a cada 25 decisões humanas só com
  rótulos verificados; exploração = topo global do escore; estimação = amostra
  probabilística de 5% de B (estratificada pela régua, pi conhecido) com
  estimador de Horvitz-Thompson de positivos restantes e IC 95%.
* S3: S2 com o veredito do LLM como atributo; S4: como rótulo.
* S5: bandit de Thompson (``ranking.ThompsonQueue``) sobre faixas contíguas.
* S6: S2 com parada por tau fixo (regra atual); só entra na hipótese H10.

Uso (a partir da raiz do repositório)::

    python notebooks/exp_engenharia_filas.py --seeds 10 --workers 14 \\
        --real-input ../phd-research/data/COMPARADORSEMIDENT.csv \\
        --out-csv docs/engenharia_de_filas_resultados.csv
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from collections.abc import Mapping, Sequence  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from dataclasses import asdict, dataclass, replace  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from typing import Any, cast  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.special import expit  # noqa: E402

from gzcmd_record_linkage.arranjo_corte import (  # noqa: E402
    THESIS_PARTITION,
    below_cutoff,
)
from gzcmd_record_linkage.calibration import (  # noqa: E402
    _InsufficientAnchors,
    fit_anchor_platt,
    get_nota_series,
)
from gzcmd_record_linkage.fellegi_sunter import (  # noqa: E402
    design_matrix,
    discretize_fields,
)
from gzcmd_record_linkage.llm_dossier import make_pair_id  # noqa: E402
from gzcmd_record_linkage.llm_review import (  # noqa: E402
    AgentResponse,
    ReviewResult,
)
from gzcmd_record_linkage.loader import LoadConfig, load_comparador_csv  # noqa: E402
from gzcmd_record_linkage.ranking import ThompsonQueue  # noqa: E402
from gzcmd_record_linkage.self_calibrating_queue import (  # noqa: E402
    ExperimentalWarning,
    SelfCalibratingConfig,
    SelfCalibratingSession,
)

# ---------------------------------------------------------------------------
# Constantes do protocolo (pré-registradas)
# ---------------------------------------------------------------------------

BUDGETS: tuple[int, ...] = (50, 100, 200, 400, 800)
N_CANDIDATES = 20_000
SAMPLE_FRACTION = 0.05
REFIT_EVERY = 25
SIGMA_COEF = 0.5  # sd do prior dos coeficientes de campo (encolhimento a zero)
SIGMA_LLM = 2.0  # sd do prior das categorias do veredito do LLM (S3)
SD_GAMMA = 0.5  # sd do prior da inclinação da régua (centrada em 1)
SD_DELTA = 2.0  # sd do prior do intercepto (centrado em 0)
REAL_LLM_WINDOW = 3_000  # janela da régua em que o fixture é usado
LLM_REVIEWED = 2_000  # o LLM revê os L primeiros da régua (livre de custo humano)
STRATA_EDGES = (0.05, 0.25)  # estratos da amostra de estimação (posto na régua)
THOMPSON_EDGES = (0.01, 0.05, 0.20)  # faixas contíguas do bandit (como a sessão)
THOMPSON_WINDOW = 100
COSTS: tuple[float, ...] = (0.02, 0.05, 0.10)
PRIMARY_COST = 0.05
TAU, DELTA_STOP, WINDOW_STOP = 0.05, 0.05, 100
MAX_PATH = max(BUDGETS)
LOOKAHEAD = 10  # nº de próximos itens que estimam o rendimento esperado
HARD_SHARE = 0.30
POCKET_SHARE = 0.50  # fração dos positivos "fáceis" realocados (não monotonia)
POCKET_RANK = (0.05, 0.07)  # faixa de posto onde fica o bolso de positivos
D_RULER = (1.6, 1.2, 1.0, 0.9)  # separação por campo (nome, mae, dtnasc, endereço)
D_MUN = 1.2  # separação do município quando há sinal fora da régua
HARD_FACTOR = (0.3, 0.3, 1.0, 1.0, 1.0)
WRONG_STEP = (0.4, -0.3, 0.4, -0.3)  # erro dos pesos da régua por grau
NOISE_STEP = 1.5  # sd do ruído somado à régua por grau
NOTA_SCALE = 4.0
FIELD_NAMES = ("nome", "mae", "dtnasc", "endereco", "municipio")
FP_LLM = 0.004
HARD_FN_LLM = 0.80
BAND_EDGES = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
BAND_NAMES = ("low", "grey_low", "grey_mid", "grey_high", "near_high", "high")
# FN e UNSURE por faixa do Haiku real (nota oculta), ver ``*.rates.csv``
LLM_FN = (0.666667, 0.294118, 0.070175, 0.093023, 0.294118, 0.215054)
LLM_UNSURE = (0.048544, 0.051282, 0.050955, 0.027972, 0.107843, 0.168421)
MATCH, NONMATCH, UNSURE, NONE = 1, 0, 2, -1  # códigos do veredito do LLM
T_CRIT_95 = 1.96

# ---------------------------------------------------------------------------
# Células e instâncias
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """Botões do gerador sintético (seção 5)."""

    prevalence: float = 0.01
    ruler: str = "correct"  # correct | wrong | noise
    degree: int = 0
    signal: bool = False
    hard: bool = False
    nonmono: bool = False
    fn_concentrated: bool = False
    human_fn: float = 0.0
    human_fp: float = 0.0
    n: int = N_CANDIDATES

    @property
    def cell_id(self) -> str:
        parts = [f"p{self.prevalence:g}", f"r-{self.ruler}{self.degree}"]
        if self.signal:
            parts.append("sig")
        if self.hard:
            parts.append("hard")
        if self.nonmono:
            parts.append("nm")
        if self.fn_concentrated:
            parts.append("fnc")
        if self.human_fn or self.human_fp:
            parts.append(f"h{self.human_fn:g}-{self.human_fp:g}")
        return "_".join(parts)


@dataclass
class Instance:
    """Uma realização: dados da sessão e verdade/ruído simulados."""

    df: pd.DataFrame
    y: NDArray[np.bool_]
    hard: NDArray[np.bool_]
    nota: NDArray[np.float64]
    llm_all: NDArray[np.int8]  # veredito de todo par (sintético) ou NONE
    llm_seen: NDArray[np.int8]  # veredito disponível às estratégias S3/S4
    human: NDArray[np.bool_]  # decisão do humano (True = MATCH) por par
    pair_ids: list[str]
    kind: str = "synthetic"
    reviewer: Any = None
    nota_all: NDArray[np.float64] | None = None  # nota do extrato inteiro (real)

    @property
    def n(self) -> int:
        return len(self.y)


def band_index(nota: NDArray[np.float64]) -> NDArray[np.int64]:
    """Faixa (0..5) de cada nota, com limites [a, b)."""
    return np.searchsorted(BAND_EDGES, nota, side="right").astype(np.int64)


def simulate_llm(
    y: NDArray[np.bool_],
    hard: NDArray[np.bool_],
    nota: NDArray[np.float64],
    u: NDArray[np.float64],
    *,
    fn_concentrated: bool = False,
) -> NDArray[np.int8]:
    """Veredito do LLM por par: FP 0,4%, FN e UNSURE por faixa de nota.

    ``u`` tem 3 colunas uniformes por par (FN/FP, UNSURE). Com
    ``fn_concentrated`` os difíceis têm FN 0,8 e os fáceis metade da taxa da
    faixa. A mesma realização vale para todas as estratégias (pareamento).
    """
    band = band_index(nota)
    fn = np.asarray(LLM_FN)[band]
    if fn_concentrated:
        fn = np.where(hard, HARD_FN_LLM, 0.5 * fn)
    base = np.where(
        y,
        np.where(u[:, 0] < fn, NONMATCH, MATCH),
        np.where(u[:, 0] < FP_LLM, MATCH, NONMATCH),
    ).astype(np.int8)
    unsure = u[:, 1] < np.asarray(LLM_UNSURE)[band]
    return np.where(unsure, UNSURE, base).astype(np.int8)


def simulate_human(
    y: NDArray[np.bool_], u: NDArray[np.float64], fn: float, fp: float
) -> NDArray[np.bool_]:
    """Decisão do humano (True = MATCH): FN ``fn`` nos pares, FP ``fp`` nos demais."""
    return np.where(y, u[:, 0] >= fn, u[:, 1] < fp).astype(bool)


def ruler_order(nota: NDArray[np.float64]) -> NDArray[np.int64]:
    """Ordem da régua: nota decrescente, empates pelo índice."""
    return np.lexsort((np.arange(len(nota)), -nota)).astype(np.int64)


def restrict_to_window(
    nota: NDArray[np.float64], codes: NDArray[np.int8], window: int
) -> NDArray[np.int8]:
    """Veredito só para os ``window`` primeiros da régua; ``NONE`` fora dela.

    A cobertura do LLM tem de ser função da régua e nunca do rótulo: um fixture
    pode cobrir pares por razões ligadas à verdade (amostra de calibração,
    execuções guiadas por rótulo). Aqui só a posição na régua decide.
    """
    out = np.full(len(nota), NONE, dtype=np.int8)
    top = ruler_order(nota)[: min(window, len(nota))]
    out[top] = codes[top]
    return out


def generate(cell: Cell, seed: int) -> Instance:
    """Gera o DataFrame da sessão e a verdade, com os botões de ``cell``."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    n = cell.n
    n_pos = max(2, round(cell.prevalence * n))
    y = np.zeros(n, dtype=bool)
    y[rng.choice(n, size=n_pos, replace=False)] = True
    pos = np.flatnonzero(y)
    hard = np.zeros(n, dtype=bool)
    if cell.hard:
        hard[rng.choice(pos, size=max(1, round(HARD_SHARE * n_pos)), replace=False)] = (
            True
        )
    d = np.array([*D_RULER, D_MUN if cell.signal else 0.0])
    sep = np.where(y[:, None], d[None, :], 0.0)
    sep = np.where(hard[:, None], sep * np.asarray(HARD_FACTOR)[None, :], sep)
    g = rng.standard_normal((n, 5)) + sep
    z = np.clip(0.22 + 0.28 * g, 0.0, 1.0)
    mae_missing = rng.random(n) < 0.05
    z[mae_missing, 1] = 0.0
    # régua: soma ponderada da evidência latente centrada (LLR com os pesos dela)
    w = np.asarray(D_RULER, dtype=float)
    if cell.ruler == "wrong":
        w = w * (1.0 + cell.degree * np.asarray(WRONG_STEP))
    raw = ((g[:, :4] - 0.5 * np.asarray(D_RULER)[None, :]) * w[None, :]).sum(axis=1)
    if cell.ruler == "noise":
        raw = raw + rng.normal(0.0, NOISE_STEP * cell.degree, n)
    nota = 5.0 + 5.0 * np.tanh(raw / NOTA_SCALE)
    if cell.nonmono:
        easy = np.flatnonzero(y & ~hard)
        moved = rng.choice(
            easy, size=max(1, round(POCKET_SHARE * easy.size)), replace=False
        )
        srt = np.sort(nota)[::-1]
        lo, hi = (int(f * n) for f in POCKET_RANK)
        nota[moved] = rng.choice(srt[lo:hi], size=moved.size) + rng.normal(
            0.0, 1e-3, moved.size
        )
    passo = rng.choice([6, 9, 11], size=n)
    df = pd.DataFrame(
        {
            "COMPREC": np.arange(n),
            "REFREC": np.arange(n) + 1_000_000,
            "PASSO": passo,
            "SCORE": np.round(nota * 0.95, 4),
            "nota final": nota,
            "nome_score_total": z[:, 0],
            "mae_score_total": z[:, 1],
            "mae_missing": mae_missing.astype(int),
            "dtnasc_score_total": z[:, 2],
            "endereco_score_total": z[:, 3],
            "municipio_score": z[:, 4],
            "dtnasc_all_zero": (z[:, 2] == 0).astype(int),
            "endereco_zero": (z[:, 3] == 0).astype(int),
        }
    )
    pair_ids = [
        make_pair_id(c, r, p)
        for c, r, p in zip(df["COMPREC"], df["REFREC"], df["PASSO"], strict=True)
    ]
    u_llm = np.random.default_rng(np.random.SeedSequence([seed, 2])).random((n, 2))
    u_h = np.random.default_rng(np.random.SeedSequence([seed, 3])).random((n, 2))
    llm_all = simulate_llm(y, hard, nota, u_llm, fn_concentrated=cell.fn_concentrated)
    llm_seen = restrict_to_window(nota, llm_all, LLM_REVIEWED)
    human = simulate_human(y, u_h, cell.human_fn, cell.human_fp)
    return Instance(
        df=df,
        y=y,
        hard=hard,
        nota=nota,
        llm_all=llm_all,
        llm_seen=llm_seen,
        human=human,
        pair_ids=pair_ids,
        reviewer=VerdictReviewer(pair_ids, llm_all),
    )


class VerdictReviewer:
    """``PairReviewer`` que devolve um veredito pré-sorteado por ``pair_id``."""

    def __init__(self, pair_ids: Sequence[str], codes: NDArray[np.int8]) -> None:
        self._code = dict(zip(pair_ids, codes.tolist(), strict=True))

    @staticmethod
    def _agent(decision: str) -> AgentResponse:
        return AgentResponse(
            decision=cast(Any, decision),
            confidence=0.9,
            reason_codes=[],
            evidence_summary={},
            quality_flags={},
            raw_json={},
            latency_s=0.0,
            tokens_prompt=0,
            tokens_completion=0,
        )

    def review(self, dossier: Any) -> ReviewResult:
        name = {MATCH: "MATCH", NONMATCH: "NONMATCH", UNSURE: "UNSURE"}[
            self._code[dossier.pair_id]
        ]
        agent = self._agent(name)
        return ReviewResult(
            pair_id=dossier.pair_id,
            decision=cast(Any, name),
            confidence=0.9,
            reason_codes=[],
            evidence_summary={},
            quality_flags={},
            protocol="consensus",
            agent_a=agent,
            agent_b=agent,
            arbiter=None,
        )


# ---------------------------------------------------------------------------
# Modelo de S2 (régua como offset, encolhimento forte, rótulo verificado)
# ---------------------------------------------------------------------------


def ruler_offset_info(
    nota: NDArray[np.float64], anchor_nota: NDArray[np.float64] | None = None
) -> tuple[NDArray[np.float64], str]:
    """``logit p0`` e a fonte usada: Platt por âncoras (como a sessão) ou o stub.

    O Platt por âncoras (``fit_anchor_platt``) exige >= 30 pares com nota >= 9 e
    >= 30 com nota < 5. Sem isso o ajuste falha (``_InsufficientAnchors``; o
    aviso do ``calibration`` **não é silenciado**) e a régua passa a ser o stub
    ``logit(nota / 10)``: é o que acontece no extrato real abaixo do corte (só 9
    pares com nota >= 9). ``anchor_nota`` ajusta o Platt em outra população (por
    exemplo o extrato inteiro) e o aplica a ``nota``. A fonte devolvida é
    ``anchor_platt``, ``anchor_platt_full_extract`` ou ``stub``. Um termo
    minúsculo em ``nota`` garante ordem estritamente monótona.
    """
    reference = nota if anchor_nota is None else anchor_nota
    try:
        model = fit_anchor_platt(pd.Series(reference))
        out = model.intercept + model.slope * nota
        source = "anchor_platt" if anchor_nota is None else "anchor_platt_full_extract"
    except _InsufficientAnchors:
        p = np.clip(nota / 10.0, 1e-3, 1.0 - 1e-3)
        out = np.log(p / (1.0 - p))
        source = "stub"
    return np.asarray(out + 1e-9 * nota, dtype=np.float64), source


def ruler_offset(nota: NDArray[np.float64]) -> NDArray[np.float64]:
    """Só o offset de :func:`ruler_offset_info`."""
    return ruler_offset_info(nota)[0]


@dataclass
class ModelSpec:
    """Entradas do ajuste: ``eta = gamma*offset + delta + X theta``."""

    X: NDArray[np.float64]
    offset: NDArray[np.float64]
    prior_sd: NDArray[np.float64]  # sd por coluna de X
    sd_gamma: float = SD_GAMMA
    sd_delta: float = SD_DELTA
    feature_names: tuple[str, ...] = ()
    ruler_source: str = "anchor_platt"  # anchor_platt | stub | ..._full_extract

    def eta(self, params: NDArray[np.float64]) -> NDArray[np.float64]:
        return params[0] * self.offset + params[1] + self.X @ params[2:]

    def initial(self) -> NDArray[np.float64]:
        p = np.zeros(2 + self.X.shape[1])
        p[0] = 1.0
        return p


def fit_offset_logistic(
    spec: ModelSpec,
    idx: NDArray[np.int64],
    y: NDArray[np.float64],
    start: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """MAP da logística: régua como covariável (prior em 1), theta ~ N(0, sd^2)."""
    params0 = spec.initial() if start is None else start.copy()
    if idx.size == 0:
        return params0
    o, X = spec.offset[idx], spec.X[idx]
    var_theta = np.square(spec.prior_sd)

    def objective(p: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        eta = p[0] * o + p[1] + X @ p[2:]
        loss = float(np.sum(np.logaddexp(0.0, eta) - y * eta))
        loss += 0.5 * ((p[0] - 1.0) / spec.sd_gamma) ** 2
        loss += 0.5 * (p[1] / spec.sd_delta) ** 2
        loss += 0.5 * float(np.sum(np.square(p[2:]) / var_theta))
        r = expit(eta) - y
        grad = np.empty_like(p)
        grad[0] = float(r @ o) + (p[0] - 1.0) / spec.sd_gamma**2
        grad[1] = float(r.sum()) + p[1] / spec.sd_delta**2
        grad[2:] = X.T @ r + p[2:] / var_theta
        return loss, grad

    res = minimize(objective, params0, jac=True, method="L-BFGS-B")
    return np.asarray(res.x, dtype=np.float64)


@dataclass
class PathResult:
    chosen: NDArray[np.int64]
    pnext: NDArray[np.float64]  # rendimento esperado dos próximos itens, antes de cada
    snapshots: dict[int, NDArray[np.float64]]


def adaptive_path(
    spec: ModelSpec,
    human_match: NDArray[np.bool_],
    steps: int,
    *,
    refit_every: int = REFIT_EVERY,
    extra_idx: NDArray[np.int64] | None = None,
    extra_y: NDArray[np.float64] | None = None,
) -> PathResult:
    """Sequência de decisões humanas pelo topo global do escore do modelo.

    A cada ``refit_every`` decisões o modelo é reajustado só com rótulos
    verificados (decisões humanas); ``extra_*`` são rótulos adicionais (S4: o
    veredito do LLM), que perdem para a decisão humana do mesmo par.
    """
    n = len(spec.offset)
    steps = min(steps, n)
    reviewed = np.zeros(n, dtype=bool)
    chosen = np.empty(steps, dtype=np.int64)
    pnext = np.empty(steps, dtype=np.float64)
    labels = np.zeros(n, dtype=np.float64)
    has_extra = extra_idx is not None and extra_idx.size > 0
    params = spec.initial()
    snapshots: dict[int, NDArray[np.float64]] = {}
    idx_all = np.arange(n)
    order = np.empty(0, dtype=np.int64)
    p_sorted = np.empty(0, dtype=np.float64)
    ptr = 0

    def refit(k: int) -> NDArray[np.float64]:
        done = chosen[:k]
        ids, ys = done, labels[done]
        if has_extra:
            assert extra_idx is not None and extra_y is not None
            keep = ~reviewed[extra_idx]
            ids = np.concatenate([done, extra_idx[keep]])
            ys = np.concatenate([ys, extra_y[keep]])
        return fit_offset_logistic(spec, ids.astype(np.int64), ys, params)

    for k in range(steps):
        if k == 0 and has_extra:
            params = refit(0)
        elif k > 0 and k % refit_every == 0:
            params = refit(k)
        if k % refit_every == 0:
            snapshots[k] = params.copy()
            eta = spec.eta(params)
            full = np.lexsort((idx_all, -eta))
            order = full[~reviewed[full]]
            p_sorted = expit(eta[order])
            ptr = 0
        item = int(order[ptr])
        pnext[k] = float(p_sorted[ptr : ptr + LOOKAHEAD].mean())
        ptr += 1
        chosen[k] = item
        reviewed[item] = True
        labels[item] = 1.0 if human_match[item] else 0.0
    return PathResult(chosen, pnext, snapshots)


def model_at(path: PathResult, k: int) -> NDArray[np.float64]:
    """Parâmetros do último reajuste feito até ``k`` decisões."""
    return path.snapshots[(k // REFIT_EVERY) * REFIT_EVERY]


def build_spec(
    inst: Instance,
    *,
    llm_attribute: bool = False,
    sigma: float = SIGMA_COEF,
    sd_gamma: float = SD_GAMMA,
    ruler: str = "pool",
) -> ModelSpec:
    """Atributos (níveis dos campos; veredito do LLM em S3) e offset da régua.

    ``ruler="full"`` ajusta o Platt por âncoras no extrato inteiro (confirmação
    exploratória no real); o padrão ajusta no próprio pool, como a sessão.
    """
    levels = discretize_fields(inst.df, "composites")
    X, names = design_matrix(levels)
    sd = np.full(X.shape[1], sigma)
    names = tuple(names)
    if llm_attribute:
        cols = [
            (inst.llm_seen == c).astype(np.float64) for c in (MATCH, UNSURE, NONMATCH)
        ]
        label = {0: "llm:MATCH", 1: "llm:UNSURE", 2: "llm:NONMATCH"}
        keep = [(label[k], c) for k, c in enumerate(cols) if c.any()]
        X = np.column_stack([X, *(c for _, c in keep)])
        sd = np.concatenate([sd, np.full(len(keep), SIGMA_LLM)])
        names = (*names, *(n for n, _ in keep))
    if ruler == "full":
        if inst.nota_all is None:
            raise ValueError("ruler='full' exige a nota do extrato inteiro")
        offset, source = ruler_offset_info(inst.nota, inst.nota_all)
    else:
        offset, source = ruler_offset_info(inst.nota)
    return ModelSpec(
        X=X,
        offset=offset,
        prior_sd=sd,
        sd_gamma=sd_gamma,
        feature_names=names,
        ruler_source=source,
    )


# ---------------------------------------------------------------------------
# Amostra de estimação e Horvitz-Thompson
# ---------------------------------------------------------------------------


def allocate_sample(
    sizes: NDArray[np.float64], weights: NDArray[np.float64], m: int
) -> NDArray[np.int64]:
    """Aloca ``m`` unidades aos estratos ~ ``weights`` (mínimo 1; máximo o tamanho)."""
    h = len(sizes)
    m = int(min(m, sizes.sum()))
    alloc = np.minimum(np.ones(h, dtype=np.int64), sizes.astype(np.int64))
    if m < alloc.sum():
        out = np.zeros(h, dtype=np.int64)
        out[np.argsort(-weights)[:m]] = 1
        return out
    w = np.maximum(weights, 1e-12)
    while alloc.sum() < m:
        room = alloc < sizes
        if not room.any():
            break
        target = np.where(room, w / w[room].sum() * (m - alloc.sum()) + alloc, -np.inf)
        gap = np.where(room, target - alloc, -np.inf)
        alloc[int(np.argmax(gap))] += 1
    return alloc


@dataclass
class Sample:
    idx: NDArray[np.int64]
    N_h: NDArray[np.float64]
    n_h: NDArray[np.float64]
    stratum: NDArray[np.int64]  # estrato de cada item amostrado


def draw_sample(
    pool: NDArray[np.int64],
    nota: NDArray[np.float64],
    p_hat: NDArray[np.float64],
    m: int,
    rng: np.random.Generator,
    edges: Sequence[float] = STRATA_EDGES,
) -> Sample:
    """Amostra estratificada pelo posto na régua dentro de ``pool`` (pi conhecido).

    Alocação de Neyman assistida pelo modelo: ``N_h * sqrt(q_h (1 - q_h))`` com
    ``q_h`` a probabilidade média do modelo no estrato (limitada a [1e-3, .5]).
    """
    ordered = pool[np.lexsort((pool, -nota[pool]))]
    size = len(ordered)
    cuts = [math.ceil(e * size - 1e-9) for e in edges]
    bounds = [0, *cuts, size]
    strata = [ordered[a:b] for a, b in zip(bounds[:-1], bounds[1:], strict=True)]
    strata = [s for s in strata if s.size]
    sizes = np.array([s.size for s in strata], dtype=np.float64)
    q = np.array([np.clip(p_hat[s].mean(), 1e-3, 0.5) for s in strata])
    weights = sizes * np.sqrt(q * (1.0 - q))
    alloc = allocate_sample(sizes, weights, m)
    picked: list[NDArray[np.int64]] = []
    labels: list[NDArray[np.int64]] = []
    for h, (s, a) in enumerate(zip(strata, alloc, strict=True)):
        if a > 0:
            chosen = rng.choice(s, size=int(a), replace=False)
            picked.append(np.asarray(chosen, dtype=np.int64))
            labels.append(np.full(int(a), h, dtype=np.int64))
    idx = np.concatenate(picked) if picked else np.empty(0, dtype=np.int64)
    lab = np.concatenate(labels) if labels else np.empty(0, dtype=np.int64)
    return Sample(idx, sizes, alloc.astype(np.float64), lab)


def ht_remaining(
    sample: Sample, observed: NDArray[np.bool_], z: float = T_CRIT_95
) -> tuple[float, float, float]:
    """Positivos restantes (não revistos fora da amostra) e IC 95%.

    ``R_hat = sum_h N_h x_h / n_h - sum_h x_h``: estimador de Horvitz-Thompson
    do total do pool menos os positivos verificados na própria amostra. O erro é
    o do HT (``R_hat - R = T_hat - T``), sem viés. Variância estratificada com
    ``p~ = (x + 1/2) / (n + 1)`` (evita variância nula com x = 0); limite
    inferior truncado em 0. ``observed`` é a decisão por item amostrado.
    """
    k = len(sample.N_h)
    x = np.array(
        [observed[sample.stratum == h].sum() for h in range(k)], dtype=np.float64
    )
    n_h, N_h = sample.n_h, sample.N_h
    safe = np.maximum(n_h, 1.0)
    total = float(np.sum(np.where(n_h > 0, N_h * x / safe, 0.0)))
    p = (x + 0.5) / (n_h + 1.0)
    var = np.where(
        n_h > 0, N_h**2 * np.maximum(1.0 - n_h / N_h, 0.0) * p * (1 - p) / safe, 0.0
    )
    est = total - float(x.sum())
    half = z * math.sqrt(float(var.sum()))
    return est, max(0.0, est - half), est + half


# ---------------------------------------------------------------------------
# Estratégias
# ---------------------------------------------------------------------------


def verified(inst: Instance, decided: NDArray[np.int64]) -> tuple[int, int]:
    """(pares verdadeiros verificados, difíceis verificados) entre os decididos."""
    hit = inst.human[decided] & inst.y[decided]
    return int(hit.sum()), int((hit & inst.hard[decided]).sum())


def thompson_path(
    nota: NDArray[np.float64],
    human_match: NDArray[np.bool_],
    steps: int,
    seed: int,
    *,
    edges: Sequence[float] = THOMPSON_EDGES,
    window: int = THOMPSON_WINDOW,
) -> NDArray[np.int64]:
    """S5: bandit de Thompson sobre faixas contíguas da régua (recompensa = MATCH)."""
    order = ruler_order(nota)
    n = len(order)
    cuts = [math.ceil(e * n - 1e-9) for e in edges]
    band = np.searchsorted(cuts, np.arange(n), side="right")
    arms = [order[band == b] for b in range(len(edges) + 1) if (band == b).any()]
    queue = ThompsonQueue(
        arms,
        window=window,
        prior=(1, 1),
        seed=seed,
        input_kind="partial_orders",
        n_items=n,
    )
    out: list[int] = []
    for _ in range(min(steps, n)):
        try:
            item = queue.next()
        except StopIteration:
            break
        queue.update(item, int(human_match[item]))
        out.append(item)
    return np.asarray(out, dtype=np.int64)


def sample_budget(b: int) -> int:
    """Tamanho da amostra de estimação: 5% de ``B`` (arredondado para cima)."""
    return max(1, math.ceil(SAMPLE_FRACTION * b))


def strategy_at_budget(
    inst: Instance,
    path: PathResult,
    spec: ModelSpec,
    budget: int,
    seed: int,
) -> dict[str, Any]:
    """S2/S3/S4 com orçamento ``B``: B-m decisões de exploração + amostra de m."""
    m = sample_budget(budget)
    e = budget - m
    explored = path.chosen[:e]
    params = model_at(path, e)
    pool = np.setdiff1d(np.arange(inst.n), explored)
    p_hat = expit(spec.eta(params))
    rng = np.random.default_rng(np.random.SeedSequence([seed, 5, budget]))
    sample = draw_sample(pool, inst.nota, p_hat, m, rng)
    decided = np.concatenate([explored, sample.idx])
    tp, hard_tp = verified(inst, decided)
    in_sample = np.zeros(inst.n, dtype=bool)
    in_sample[sample.idx] = True
    never = ~np.isin(np.arange(inst.n), decided)
    truth = float((inst.y & never).sum())
    est, lo, hi = ht_remaining(sample, inst.human[sample.idx])
    truth_h = float((inst.y & inst.hard & never).sum())
    est_h, lo_h, hi_h = ht_remaining(
        sample, inst.human[sample.idx] & inst.hard[sample.idx]
    )
    return {
        "tp": tp,
        "hard_tp": hard_tp,
        "n_dec": int(decided.size),
        "est": est,
        "lo": lo,
        "hi": hi,
        "truth": truth,
        "covered": float(lo <= truth <= hi),
        "est_hard": est_h,
        "lo_hard": lo_h,
        "hi_hard": hi_h,
        "truth_hard": truth_h,
        "covered_hard": float(lo_h <= truth_h <= hi_h),
    }


def s0_at_budget(
    inst: Instance, order: NDArray[np.int64], budget: int
) -> dict[str, Any]:
    decided = order[:budget]
    tp, hard_tp = verified(inst, decided)
    return {"tp": tp, "hard_tp": hard_tp, "n_dec": int(decided.size)}


def paths_value(
    inst: Instance, chosen: NDArray[np.int64], budget: int
) -> dict[str, Any]:
    decided = chosen[:budget]
    tp, hard_tp = verified(inst, decided)
    return {"tp": tp, "hard_tp": hard_tp, "n_dec": int(decided.size)}


# ---------------------------------------------------------------------------
# S1: a sessão atual
# ---------------------------------------------------------------------------


def run_s1(inst: Instance, seed: int, budgets: Sequence[int]) -> list[dict[str, Any]]:
    """Roda a ``SelfCalibratingSession`` padrão (humano decide todo item do portão)."""
    index = {pid: i for i, pid in enumerate(inst.pair_ids)}
    cfg = SelfCalibratingConfig(mode="human", seed=seed)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        session = SelfCalibratingSession(inst.df, inst.reviewer, cfg)
        gate_order: list[int] = []
        while not session.finished:
            if session.awaiting_human:
                items = session.gate_items
                gate_order += [index[str(it["pair_id"])] for it in items]
                session.submit_human_decisions(
                    {
                        str(it["pair_id"]): (
                            "MATCH"
                            if inst.human[index[str(it["pair_id"])]]
                            else "NONMATCH"
                        )
                        for it in items
                    }
                )
            else:
                session.step()
    elapsed = time.perf_counter() - t0
    decided = np.asarray(gate_order, dtype=np.int64)
    rows = []
    for b in budgets:
        tp, hard_tp = verified(inst, decided[:b])
        rows.append(
            {
                "strategy": "S1",
                "B": b,
                "tp": tp,
                "hard_tp": hard_tp,
                "n_dec": int(min(b, decided.size)),
            }
        )
    rows.append(s1_estimate(inst, session, index, int(decided.size), elapsed))
    return rows


def s1_estimate(
    inst: Instance,
    session: SelfCalibratingSession,
    index: Mapping[str, int],
    n_decisions: int,
    elapsed: float,
) -> dict[str, Any]:
    """Estimador de positivos restantes da S1: amostra de exploração, pesos com teto.

    Última rodada ``r``: ``N_r`` pares não revistos no início, ``n_e`` revisões de
    exploração (a cada 10ª revisão), ``x`` delas com MATCH humano. Peso
    ``min(N_r / n_e, 10)`` (teto ``weight_max_ratio`` sobre o peso 1 do estrato
    ``top``). Alvo: positivos entre os pares não revistos no início de ``r``.
    """
    res = session.results()
    summary = session.summary()
    last = int(res["round"].max())
    sizes = {int(r["round"]): int(r["size"]) for r in summary["explore_sizes"]}
    n_r = float(sizes[last])
    before = np.array([index[p] for p in res.loc[res["round"] < last, "pair_id"]])
    target = float(inst.y.sum() - (inst.y[before].sum() if before.size else 0))
    exp = res[(res["round"] == last) & (res["arm"].astype(str) == "explore")]
    n_e = len(exp)
    x = float((exp["human_decision"] == "MATCH").sum())
    out: dict[str, Any] = {
        "strategy": "S1",
        "B": 0,
        "tp": math.nan,
        "hard_tp": math.nan,
        "n_dec": n_decisions,
        "elapsed_s": elapsed,
        "truth": target,
        "rounds": float(last),
    }
    if n_e == 0:
        out.update(
            {"est": math.nan, "lo": math.nan, "hi": math.nan, "covered": math.nan}
        )
        return out
    raw_w = n_r / n_e
    for tag, w in (("", min(raw_w, 10.0)), ("_uncapped", raw_w)):
        p = (x + 0.5) / (n_e + 1.0)
        half = T_CRIT_95 * w * math.sqrt(n_e * p * (1.0 - p))
        est = w * x
        lo, hi = max(0.0, est - half), est + half
        out[f"est{tag}"] = est
        out[f"lo{tag}"] = lo
        out[f"hi{tag}"] = hi
        out[f"covered{tag}"] = float(lo <= target <= hi)
    return out


# ---------------------------------------------------------------------------
# Parada por custo (H10) e auditoria (H11)
# ---------------------------------------------------------------------------


def stop_cost(pnext: NDArray[np.float64], cost: float, k_min: int = REFIT_EVERY) -> int:
    """Nº de decisões ao parar: 1o ``k >= k_min`` com rendimento esperado < ``cost``."""
    hits = np.flatnonzero(pnext[k_min:] < cost)
    return int(hits[0] + k_min) if hits.size else int(len(pnext))


def stop_tau(
    hit: NDArray[np.bool_],
    tau: float = TAU,
    delta: float = DELTA_STOP,
    window: int = WINDOW_STOP,
) -> int:
    """Regra atual: para quando ``P(theta > tau) < delta`` na janela recente.

    ``hit`` são as decisões humanas MATCH (o que o operador vê), nunca a verdade.
    """
    cum = np.concatenate([[0], np.cumsum(hit.astype(np.int64))])
    k = np.arange(1, len(hit) + 1)
    s = cum[k] - cum[np.maximum(k - window, 0)]
    n = np.minimum(k, window)
    p = stats.beta.sf(tau, 1 + s, 1 + n - s)
    hits = np.flatnonzero(p < delta)
    return int(hits[0] + 1) if hits.size else int(len(hit))


def regret_rows(
    inst: Instance, path: PathResult, costs: Sequence[float] = COSTS
) -> list[dict[str, Any]]:
    """Arrependimento = valor do oráculo - valor obtido, para S2 (custo) e S6 (tau)."""
    steps = len(path.chosen)
    decided = inst.human[path.chosen]  # só o que o operador observa
    truth_hit = (decided & inst.y[path.chosen]).astype(np.int64)
    cum = np.concatenate([[0], np.cumsum(truth_hit)])  # valor verdadeiro obtido
    total = int(inst.y.sum())
    k6 = stop_tau(decided)
    rows = []
    for c in costs:
        oracle = (1.0 - c) * min(total, steps)
        for rule, k in (("S2", stop_cost(path.pnext, c)), ("S6", k6)):
            value = float(cum[k]) - c * k
            rows.append(
                {
                    "cost": c,
                    "rule": rule,
                    "k_stop": k,
                    "found": int(cum[k]),
                    "value": value,
                    "oracle": oracle,
                    "regret": oracle - value,
                }
            )
    return rows


def audit_row(inst: Instance, order: NDArray[np.int64], budget: int) -> dict[str, Any]:
    """H11: re-revisar só pares com LLM = MATCH e humano = NONMATCH (S0, topo B)."""
    top = order[:budget]
    missed = inst.y[top] & ~inst.human[top]
    flagged = (inst.llm_seen[top] == MATCH) & ~inst.human[top]
    return {
        "B": budget,
        "n_hfn": int(missed.sum()),
        "recovered": int((flagged & inst.y[top]).sum()),
        "extras": int(flagged.sum()),
    }


# ---------------------------------------------------------------------------
# Execução de uma (célula, semente)
# ---------------------------------------------------------------------------


def run_instance(
    inst: Instance,
    seed: int,
    *,
    budgets: Sequence[int] = BUDGETS,
    with_s1: bool = True,
    sigma: float = SIGMA_COEF,
) -> dict[str, list[dict[str, Any]]]:
    """Todas as estratégias numa realização, com o mesmo LLM e humano (pareado)."""
    steps = min(max(budgets), inst.n)
    order = ruler_order(inst.nota)
    runs: list[dict[str, Any]] = []
    for b in budgets:
        runs.append({"strategy": "S0", "B": b, **s0_at_budget(inst, order, b)})
    spec2 = build_spec(inst, sigma=sigma)
    path2 = adaptive_path(spec2, inst.human, steps)
    spec3 = build_spec(inst, llm_attribute=True, sigma=sigma)
    path3 = adaptive_path(spec3, inst.human, steps)
    seen = np.flatnonzero((inst.llm_seen == MATCH) | (inst.llm_seen == NONMATCH))
    path4 = adaptive_path(
        spec2,
        inst.human,
        steps,
        extra_idx=seen.astype(np.int64),
        extra_y=(inst.llm_seen[seen] == MATCH).astype(np.float64),
    )
    chosen5 = thompson_path(inst.nota, inst.human, steps, seed=seed + 7)
    for b in budgets:
        runs.append(
            {
                "strategy": "S2x",
                "B": b,
                **paths_value(inst, path2.chosen, b),
                "ruler_source": spec2.ruler_source,
            }
        )
        runs.append(
            {
                "strategy": "S2",
                "B": b,
                **strategy_at_budget(inst, path2, spec2, b, seed),
                "ruler_source": spec2.ruler_source,
            }
        )
        runs.append(
            {
                "strategy": "S3",
                "B": b,
                **strategy_at_budget(inst, path3, spec3, b, seed),
                "ruler_source": spec3.ruler_source,
            }
        )
        runs.append(
            {
                "strategy": "S4",
                "B": b,
                **strategy_at_budget(inst, path4, spec2, b, seed),
                "ruler_source": spec2.ruler_source,
            }
        )
        runs.append({"strategy": "S5", "B": b, **paths_value(inst, chosen5, b)})
    if with_s1 and inst.reviewer is not None:
        runs += run_s1(inst, seed, budgets)
    return {
        "runs": runs,
        "stops": regret_rows(inst, path2),
        "audit": [audit_row(inst, order, b) for b in budgets],
    }


def tag_rows(
    rows: list[dict[str, Any]], meta: Mapping[str, Any]
) -> list[dict[str, Any]]:
    return [{**meta, **r} for r in rows]


def run_cell_seed(task: tuple[Cell, int, bool, float]) -> dict[str, Any]:
    """Tarefa de um processo: gera, roda as estratégias e devolve linhas marcadas."""
    cell, seed, with_s1, sigma = task
    t0 = time.perf_counter()
    inst = generate(cell, seed)
    out = run_instance(inst, seed, with_s1=with_s1, sigma=sigma)
    meta = {
        **asdict(cell),
        "cell_id": cell.cell_id,
        "seed": seed,
        "source": "synthetic",
    }
    return {
        "runs": tag_rows(out["runs"], meta),
        "stops": tag_rows(out["stops"], meta),
        "audit": tag_rows(out["audit"], meta),
        "elapsed": time.perf_counter() - t0,
        "cell_id": cell.cell_id,
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# Dados reais (extrato da tese abaixo do corte + fixture do Haiku)
# ---------------------------------------------------------------------------


def band_names(nota: NDArray[np.float64]) -> list[str]:
    names = np.asarray(BAND_NAMES, dtype=object)
    return names[band_index(nota)].tolist()


class WindowedFixtureReviewer:
    """Fixture real só para os pares da janela; fora dela, taxas por faixa.

    A S1 também não pode se beneficiar de uma cobertura do fixture que dependa
    do rótulo (calibração, execuções guiadas pela verdade).
    """

    def __init__(
        self,
        real: Any,
        fallback: Any,
        index: Mapping[str, int],
        use_real: NDArray[np.bool_],
    ) -> None:
        self._real, self._fallback = real, fallback
        self._index, self._use = index, use_real

    def review(self, dossier: Any) -> ReviewResult:
        if self._use[self._index[dossier.pair_id]]:
            return cast(ReviewResult, self._real.review(dossier))
        return cast(ReviewResult, self._fallback.review(dossier))


def load_real_instance(
    path: Path,
    fixture: Path,
    rates_csv: Path,
    seed: int,
    *,
    human_fn: float = 0.0,
    human_fp: float = 0.0,
    window: int = REAL_LLM_WINDOW,
) -> Instance:
    """Extrato abaixo do corte, com o veredito do fixture; PAR só como verdade.

    O veredito só vale para os ``window`` primeiros pela nota (cobertura que
    não depende do rótulo): o fixture também cobre a amostra de calibração e
    execuções guiadas pela verdade, e usá-lo como atributo vazaria ``y``.
    """
    import llm_real_eval  # lazy: módulo pesado; não é editado aqui

    raw = load_comparador_csv(path, cfg=LoadConfig(require_par=False))
    if "SCORE" not in raw.columns:
        legacy = [c for c in raw.columns if str(c).upper().startswith("SCORE,")]
        if legacy:
            raw = raw.assign(SCORE=pd.to_numeric(raw[legacy[0]], errors="coerce"))
    par = pd.to_numeric(raw["PAR"], errors="coerce")
    below = np.asarray(below_cutoff(raw, expected=THESIS_PARTITION), dtype=bool)
    keep = raw.drop(columns=[c for c in ("PAR", "TARGET") if c in raw.columns])
    df = keep.loc[below].reset_index(drop=True)
    y = par.isin([1, 2]).to_numpy()[below]
    nota = pd.to_numeric(get_nota_series(df), errors="coerce").to_numpy(dtype=float)
    pair_ids = [
        make_pair_id(c, r, p)
        for c, r, p in zip(df["COMPREC"], df["REFREC"], df["PASSO"], strict=True)
    ]
    truth = dict(zip(pair_ids, y.tolist(), strict=True))
    rates = llm_real_eval.load_rates_csv(rates_csv)
    band_by_pair = dict(zip(pair_ids, band_names(nota), strict=True))
    empty = Path(tempfile.gettempdir()) / "exp_filas_fixture_vazio.jsonl"
    empty.touch(exist_ok=True)
    fallback = llm_real_eval.FixtureReviewer(
        empty,
        on_missing="rates",
        rates=rates,
        truth=truth,
        band_by_pair=band_by_pair,
        seed=seed,
        disagreement_rate=0.0,
    )
    plain = llm_real_eval.FixtureReviewer(fixture)
    code = {"MATCH": MATCH, "NONMATCH": NONMATCH, "UNSURE": UNSURE}
    covered = np.full(len(pair_ids), NONE, dtype=np.int8)
    for i, pid in enumerate(pair_ids):
        if pid in plain:
            stub = cast(Any, SimpleNamespace(pair_id=pid, to_dict=dict))
            covered[i] = code[str(plain.review(stub).decision)]
    seen = restrict_to_window(nota, covered, window)
    reviewer = WindowedFixtureReviewer(
        plain, fallback, {p: i for i, p in enumerate(pair_ids)}, seen != NONE
    )
    u_h = np.random.default_rng(np.random.SeedSequence([seed, 3])).random((len(y), 2))
    return Instance(
        df=df,
        y=y,
        hard=np.zeros(len(y), dtype=bool),
        nota=nota,
        llm_all=seen,
        llm_seen=seen,
        human=simulate_human(y, u_h, human_fn, human_fp),
        pair_ids=pair_ids,
        kind="real",
        reviewer=reviewer,
        nota_all=pd.to_numeric(get_nota_series(keep), errors="coerce").to_numpy(
            dtype=float
        ),
    )


_REAL_ARGS: dict[str, Any] = {}


def run_real_seed(task: tuple[int, bool]) -> dict[str, Any]:
    """Tarefa de um processo: estratégias no extrato real (humano perfeito)."""
    seed, with_s1 = task
    t0 = time.perf_counter()
    inst = load_real_instance(
        _REAL_ARGS["path"], _REAL_ARGS["fixture"], _REAL_ARGS["rates"], seed
    )
    out = run_instance(inst, seed, with_s1=with_s1)
    meta = {
        "cell_id": "real_below_cutoff",
        "seed": seed,
        "source": "real",
        "prevalence": float(inst.y.mean()),
        "n": inst.n,
    }
    return {
        "runs": tag_rows(out["runs"], meta),
        "stops": tag_rows(out["stops"], meta),
        "audit": [],
        "elapsed": time.perf_counter() - t0,
        "cell_id": "real_below_cutoff",
        "seed": seed,
    }


def real_audit_rows(
    path: Path, fixture: Path, rates_csv: Path, n_draws: int, human_fn: Sequence[float]
) -> list[dict[str, Any]]:
    """H11 no extrato real: muitas realizações do erro do humano (a régua é fixa)."""
    base = load_real_instance(path, fixture, rates_csv, 0)
    order = ruler_order(base.nota)
    rows = []
    for fn in human_fn:
        for s in range(n_draws):
            u_h = np.random.default_rng(np.random.SeedSequence([s, 33])).random(
                (base.n, 2)
            )
            inst = replace(base, human=simulate_human(base.y, u_h, fn, 0.01))
            for b in BUDGETS:
                rows.append(
                    {
                        "cell_id": f"real_h{fn:g}",
                        "seed": s,
                        "source": "real",
                        "human_fn": fn,
                        **audit_row(inst, order, b),
                    }
                )
    return rows


# ---------------------------------------------------------------------------
# Grade
# ---------------------------------------------------------------------------

PREVALENCES = (0.001, 0.01, 0.05, 0.20)


def build_cells(n: int = N_CANDIDATES) -> list[Cell]:
    """Cenário base + um fator por vez + interação (prevalência x régua x sinal)."""
    base = Cell(n=n)
    cells: list[Cell] = [base]
    for p in PREVALENCES:
        for sig in (False, True):
            cells.append(replace(base, prevalence=p, signal=sig))
    for kind in ("wrong", "noise"):
        for deg in (1, 2, 3):
            cells.append(replace(base, ruler=kind, degree=deg))
    for kind in ("wrong", "noise"):
        for p in PREVALENCES:
            for sig in (False, True):
                cells.append(
                    replace(base, ruler=kind, degree=2, prevalence=p, signal=sig)
                )
    cells += [
        replace(base, hard=True),
        replace(base, hard=True, signal=True),
        replace(base, hard=True, signal=True, prevalence=0.05),
        replace(base, hard=True, fn_concentrated=True),
        replace(base, hard=True, signal=True, prevalence=0.05, fn_concentrated=True),
        replace(base, nonmono=True),
        replace(base, nonmono=True, prevalence=0.05),
    ]
    for p in (0.01, 0.05):
        for fn in (0.05, 0.10):
            cells.append(replace(base, prevalence=p, human_fn=fn, human_fp=0.01))
    seen: dict[str, Cell] = {}
    for c in cells:
        seen.setdefault(c.cell_id, c)
    return list(seen.values())


# ---------------------------------------------------------------------------
# Análise: IC pareado e critérios de decisão
# ---------------------------------------------------------------------------


def paired_ci(d: Sequence[float] | NDArray[np.float64]) -> tuple[float, float, float]:
    """Média e IC 95% (t de Student) das diferenças pareadas entre sementes."""
    arr = np.asarray(d, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan, math.nan, math.nan
    mean = float(arr.mean())
    if arr.size < 2:
        return mean, math.nan, math.nan
    sd = float(arr.std(ddof=1))
    if sd == 0.0:
        return mean, mean, mean
    half = float(stats.t.ppf(0.975, arr.size - 1)) * sd / math.sqrt(arr.size)
    return mean, mean - half, mean + half


def margin(s0_mean: float) -> float:
    """Margem de não inferioridade: ``max(1 par, 5% de S0)``."""
    return max(1.0, 0.05 * s0_mean)


def ci_includes_zero(lo: float, hi: float) -> bool:
    """Ruído = IC 95% pareado incluindo 0."""
    return lo <= 0.0 <= hi


def ci_above_zero(lo: float) -> bool:
    return lo > 0.0


def paired_table(
    runs: pd.DataFrame, a: str, b: str, metric: str = "tp"
) -> pd.DataFrame:
    """Diferença pareada por semente ``a - b`` em cada (célula, B)."""
    sub = runs[runs["strategy"].isin([a, b]) & (runs["B"] > 0)]
    if sub["strategy"].nunique() < 2:
        cols = ["cell_id", "B", "comparison", "metric", "mean", "lo", "hi"]
        return pd.DataFrame(columns=[*cols, "ref_mean", "a_mean", "n_seeds"])
    wide = sub.pivot_table(
        index=["cell_id", "B", "seed"],
        columns="strategy",
        values=metric,
        aggfunc="first",
    ).reset_index()
    rows = []
    for (cell, bud), grp in wide.groupby(["cell_id", "B"]):
        mean, lo, hi = paired_ci((grp[a] - grp[b]).to_numpy())
        rows.append(
            {
                "cell_id": cell,
                "B": int(cast(Any, bud)),
                "comparison": f"{a}-{b}",
                "metric": metric,
                "mean": mean,
                "lo": lo,
                "hi": hi,
                "ref_mean": float(grp[b].mean()),
                "a_mean": float(grp[a].mean()),
                "n_seeds": int(len(grp)),
            }
        )
    cols = ["cell_id", "B", "comparison", "metric", "mean", "lo", "hi"]
    cols += ["ref_mean", "a_mean", "n_seeds"]
    return pd.DataFrame(rows, columns=cols)


def is_base(c: Mapping[str, Any]) -> bool:
    return (
        math.isclose(float(c["prevalence"]), 0.01)
        and c["ruler"] == "correct"
        and not c["signal"]
        and not c["hard"]
        and not c["nonmono"]
        and not c["fn_concentrated"]
        and not c["human_fn"]
    )


def _status(ok: bool | None) -> str:
    if ok is None:
        return "inconclusiva"
    return "confirmada" if ok else "refutada"


def _with_cells(df: pd.DataFrame, info: pd.DataFrame) -> pd.DataFrame:
    """Junta os botões da célula só nas colunas que ainda faltam."""
    extra = info.drop(columns=[c for c in info.columns if c in df.columns])
    return df.join(extra, on="cell_id")


def evaluate(
    runs: pd.DataFrame,
    stops: pd.DataFrame,
    audit: pd.DataFrame,
    cells: pd.DataFrame,
    *,
    scope: str = "sintético",
    base_ids: frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    """Aplica os critérios de decisão pré-registrados (H1-H11) e devolve o veredito.

    ``cells`` tem uma linha por ``cell_id`` com os botões da célula. Cada item
    devolvido traz ``id``, ``status`` e ``detail`` (números que o sustentam).
    """
    info = cells.set_index("cell_id")
    p20 = paired_table(runs, "S2", "S0")
    p10 = paired_table(runs, "S1", "S0")
    p20 = p20.join(info, on="cell_id")
    p10 = p10.join(info, on="cell_id")
    out: list[dict[str, Any]] = []

    def sel(df: pd.DataFrame, **kw: Any) -> pd.DataFrame:
        m = np.ones(len(df), dtype=bool)
        for k, v in kw.items():
            col = df[k]
            if isinstance(v, float):
                m &= np.isclose(col.to_numpy(dtype=float), v)
            else:
                m &= (col == v).to_numpy()
        return df[m]

    # H1 -----------------------------------------------------------------
    def isb(r: Mapping[str, Any]) -> bool:
        return r["cell_id"] in base_ids if base_ids is not None else is_base(r)

    base_mask = np.array([isb(r) for r in p20.to_dict("records")], dtype=bool)
    b20, b10 = (
        p20[base_mask],
        p10[np.array([isb(r) for r in p10.to_dict("records")], dtype=bool)],
    )
    if len(b20):
        zero_all = all(ci_includes_zero(r.lo, r.hi) for r in b20.itertuples())
        s1_neg = any(r.mean < 0 for r in b10.itertuples())
        detail = "; ".join(
            f"B={r.B}: S2-S0={r.mean:+.1f} [{r.lo:+.1f}, {r.hi:+.1f}]"
            for r in b20.itertuples()
        )
        detail += " | S1-S0: " + ", ".join(
            f"B={r.B}: {r.mean:+.1f}" for r in b10.itertuples()
        )
        out.append(
            {
                "id": "H1",
                "status": _status(zero_all and s1_neg),
                "detail": detail,
                "scope": scope,
            }
        )
    # H2 -----------------------------------------------------------------
    rel = []
    for p in PREVALENCES:
        sub = sel(
            p20[
                (p20["signal"])
                & (p20["ruler"] == "correct")
                & (~p20["hard"])
                & (~p20["nonmono"])
                & (p20["human_fn"] == 0)
            ],
            prevalence=float(p),
        )
        if len(sub):
            rel.append(
                (p, float(np.mean(sub["mean"] / np.maximum(sub["ref_mean"], 1e-9))))
            )
    if len(rel) == len(PREVALENCES):
        vals = [v for _, v in rel]
        mono = all(b >= a for a, b in zip(vals, vals[1:], strict=False))
        pos = all(v > 0 for p, v in rel if p >= 0.05)
        out.append(
            {
                "id": "H2",
                "status": _status(mono and pos),
                "detail": "ganho relativo médio por prevalência: "
                + ", ".join(f"{p:g}: {v:+.3f}" for p, v in rel),
                "scope": scope,
            }
        )
    # H3 -----------------------------------------------------------------
    series = {}
    for kind in ("wrong", "noise"):
        row = []
        for deg in range(4):
            cond = (
                (p20["ruler"] == ("correct" if deg == 0 else kind))
                & (p20["degree"] == deg)
                & (~p20["signal"])
                & (~p20["hard"])
                & (~p20["nonmono"])
                & (p20["human_fn"] == 0)
                & np.isclose(p20["prevalence"].astype(float), 0.01)
            )
            sub = p20[cond]
            if len(sub):
                row.append(
                    (deg, float(sub["mean"].mean()), bool((sub["lo"] > 0).any()))
                )
        series[kind] = row
    if all(len(v) == 4 for v in series.values()):
        w = series["wrong"]
        growing = (
            all(b[1] >= a[1] for a, b in zip(w, w[1:], strict=False))
            and w[-1][1] > w[0][1]
        )
        ok = w[-1][2] and growing
        detail = "; ".join(
            f"{k}: " + ", ".join(f"g{d}: {m:+.2f}{'*' if s else ''}" for d, m, s in v)
            for k, v in series.items()
        )
        out.append(
            {
                "id": "H3",
                "status": _status(ok),
                "detail": detail
                + " (* = IC exclui 0 em algum B; critério na série de pesos errados)",
                "scope": scope,
            }
        )
    # H4 -----------------------------------------------------------------
    s1c = sel(
        p20[
            (p20["ruler"] == "correct")
            & (~p20["hard"])
            & (~p20["nonmono"])
            & (p20["human_fn"] == 0)
        ],
        prevalence=0.05,
    )
    with_sig, no_sig = s1c[s1c["signal"]], s1c[~s1c["signal"]]
    if len(with_sig) and len(no_sig):
        up = bool((with_sig["lo"] > 0).any())
        noise = all(ci_includes_zero(r.lo, r.hi) for r in no_sig.itertuples())
        detail = (
            "com sinal: "
            + ", ".join(
                f"B={r.B}: {r.mean:+.1f} [{r.lo:+.1f}, {r.hi:+.1f}]"
                for r in with_sig.itertuples()
            )
            + " | sem sinal: "
            + ", ".join(
                f"B={r.B}: {r.mean:+.1f} [{r.lo:+.1f}, {r.hi:+.1f}]"
                for r in no_sig.itertuples()
            )
        )
        out.append(
            {
                "id": "H4",
                "status": _status(up and noise),
                "detail": detail,
                "scope": scope,
            }
        )
    # H5 -----------------------------------------------------------------
    hard_cells = info.index[info["hard"] & ~info["fn_concentrated"]].tolist()
    if hard_cells:
        ph = paired_table(runs, "S2", "S0", "hard_tp")
        ph = ph[ph["cell_id"].isin(hard_cells)]
        rec_ok = all(float(g["mean"].mean()) <= 0 for _, g in ph.groupby("cell_id"))
        cov = runs[(runs["strategy"] == "S2") & runs["cell_id"].isin(hard_cells)][
            "covered_hard"
        ].mean()
        detail = (
            "S2-S0 em difíceis recuperados (média sobre B): "
            + ", ".join(
                f"{c}: {g['mean'].mean():+.2f}" for c, g in ph.groupby("cell_id")
            )
            + f" | cobertura do IC de difíceis restantes: {cov:.3f}"
        )
        out.append(
            {
                "id": "H5",
                "status": _status(rec_ok and cov >= 0.9),
                "detail": detail,
                "scope": scope,
            }
        )
    # H6 -----------------------------------------------------------------
    viol = [
        (r.cell_id, r.B, r.lo, margin(r.ref_mean))
        for r in p20.itertuples()
        if r.lo < -margin(r.ref_mean)
    ]
    worst = min(
        ((r.lo + margin(r.ref_mean), r.cell_id, r.B) for r in p20.itertuples()),
        default=(math.nan, "", 0),
    )
    out.append(
        {
            "id": "H6",
            "status": _status(len(viol) == 0),
            "detail": (
                f"{len(viol)} de {len(p20)} (célula, B) violam IC inferior >= "
                f"-delta; pior folga {worst[0]:+.2f} em {worst[1]} B={worst[2]}"
            ),
            "scope": scope,
        }
    )
    # H7 -----------------------------------------------------------------
    p34 = paired_table(runs, "S3", "S4").join(info, on="cell_id")
    p30 = paired_table(runs, "S3", "S0").join(info, on="cell_id")
    if len(p34):
        key = np.array(
            [isb(r) or bool(r["fn_concentrated"]) for r in p34.to_dict("records")]
        )
        sub = p34[key]
        nonneg = bool((sub["mean"] >= 0).all())
        sig = bool((sub["lo"] > 0).any())
        ni = [(r.cell_id, r.B) for r in p30.itertuples() if r.lo < -margin(r.ref_mean)]
        detail = (
            f"S3-S4 (base e FN concentrado): média mínima {sub['mean'].min():+.2f}, "
            f"IC>0 em {int((sub['lo'] > 0).sum())} de {len(sub)}; "
            f"S3-S0 viola -delta em {len(ni)} de {len(p30)}"
        )
        out.append(
            {
                "id": "H7",
                "status": _status(nonneg and sig and not ni),
                "detail": detail,
                "scope": scope,
            }
        )
    # H8 -----------------------------------------------------------------
    p50 = paired_table(runs, "S5", "S0").join(info, on="cell_id")
    if len(p50):
        mono = p50[
            (p50["ruler"] == "correct")
            & (~p50["signal"])
            & (~p50["hard"])
            & (~p50["nonmono"])
            & (p50["human_fn"] == 0)
        ]
        nm = p50[p50["nonmono"]]
        a_ok = not bool((mono["lo"] > 0).any())
        b_ok = bool((nm["lo"] > 0).any()) if len(nm) else None
        n_up = int((mono["lo"] > 0).sum())
        detail = (
            f"monotônico: máx S5-S0 {mono['mean'].max():+.1f}, "
            f"IC>0 em {n_up} de {len(mono)}"
        )
        if len(nm):
            detail += (
                f" | não monotônico: S5-S0 máx {nm['mean'].max():+.1f}, "
                f"IC>0 em {int((nm['lo'] > 0).sum())} de {len(nm)}"
            )
        ok = None if b_ok is None else (a_ok and b_ok)
        if scope == "real":
            ok = a_ok  # só o lado monotônico é testável no extrato
            detail += " (só o lado 'S5 não supera S0' é testável no real)"
        out.append(
            {"id": "H8", "status": _status(ok), "detail": detail, "scope": scope}
        )
    # H9 -----------------------------------------------------------------
    s2 = runs[(runs["strategy"] == "S2") & (runs["B"] > 0)]
    s1 = runs[(runs["strategy"] == "S1") & (runs["B"] == 0)]
    if len(s2):
        c2 = float(s2["covered"].mean())
        c1 = (
            float(s1["covered"].mean())
            if len(s1) and s1["covered"].notna().any()
            else math.nan
        )
        c1u = (
            float(s1["covered_uncapped"].mean())
            if "covered_uncapped" in s1 and s1["covered_uncapped"].notna().any()
            else math.nan
        )
        wid = float((s2["hi"] - s2["lo"]).mean())
        detail = (
            f"cobertura S2 {c2:.3f} (n={len(s2)}), largura média {wid:.1f}; "
            f"S1 com teto {c1:.3f} (n={int(s1['covered'].notna().sum())}); "
            f"S1 sem teto {c1u:.3f}"
        )
        ok = None if math.isnan(c1) else (c2 >= 0.9 and c1 < 0.9)
        out.append(
            {"id": "H9", "status": _status(ok), "detail": detail, "scope": scope}
        )
    # H10 ----------------------------------------------------------------
    if len(stops) == 0:
        stops = pd.DataFrame({"cost": [], "cell_id": []})
    st = _with_cells(stops[stops["cost"].round(4) == PRIMARY_COST], info)
    st = st[
        (st["ruler"] == "correct")
        & (~st["signal"])
        & (~st["hard"])
        & (~st["nonmono"])
        & (st["human_fn"] == 0)
    ]
    if len(st):
        wins, lines = 0, []
        for p in PREVALENCES:
            g = sel(st, prevalence=float(p))
            if not len(g):
                continue
            r2 = float(g[g["rule"] == "S2"]["regret"].mean())
            r6 = float(g[g["rule"] == "S6"]["regret"].mean())
            wins += int(r2 < r6)
            lines.append(f"{p:g}: S2 {r2:.1f} vs S6 {r6:.1f}")
        out.append(
            {
                "id": "H10",
                "status": _status(wins >= 3),
                "detail": f"c={PRIMARY_COST}; arrependimento médio "
                + "; ".join(lines)
                + f"; S2 < S6 em {wins} de {len(lines)}",
                "scope": scope,
            }
        )
    # H11 ----------------------------------------------------------------
    if len(audit):
        a = audit[(audit["B"] == 400) & (audit["human_fn"] > 0)]
        if len(a):
            lines, ok = [], True
            for cid, g in a.groupby("cell_id"):
                fn_tot, rec, ext = (
                    g["n_hfn"].sum(),
                    g["recovered"].sum(),
                    g["extras"].sum(),
                )
                recall = rec / fn_tot if fn_tot else math.nan
                prec = rec / ext if ext else math.nan
                ok &= bool(recall >= 0.5 and prec >= 0.3)
                lines.append(
                    f"{cid}: FN humanos {int(fn_tot)}, recuperados {int(rec)} "
                    f"({recall:.2f}), extras {int(ext)}, precisão {prec:.2f}"
                )
            out.append(
                {
                    "id": "H11",
                    "status": _status(ok),
                    "detail": "B=400; " + "; ".join(lines),
                    "scope": scope,
                }
            )
    out.append(
        {
            "id": "H12",
            "status": "não testável",
            "detail": "exige estudo com revisores reais",
            "scope": scope,
        }
    )
    return out


# ---------------------------------------------------------------------------
# Agregação e CSV (só agregados)
# ---------------------------------------------------------------------------


def aggregate_runs(runs: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """Médias por (célula, estratégia, B) das métricas de rendimento."""
    metrics = ["tp", "hard_tp", "n_dec"]
    sub = runs[runs["B"] > 0]
    agg = sub.groupby(["cell_id", "strategy", "B"])[metrics].agg(
        ["mean", "std", "count"]
    )
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    out = agg.reset_index()
    if "ruler_source" in sub.columns:
        src = sub.groupby(["cell_id", "strategy"])["ruler_source"].agg(
            lambda s: "+".join(sorted(set(s.dropna().astype(str))))
        )
        out = out.merge(src.reset_index(), on=["cell_id", "strategy"], how="left")
    return out.merge(cells, on="cell_id", how="left")


def build_csv(
    runs: pd.DataFrame,
    stops: pd.DataFrame,
    audit: pd.DataFrame,
    cells: pd.DataFrame,
    verdicts: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """CSV longo único (``table``) com os agregados; sem dados em nível de par."""
    parts: list[pd.DataFrame] = []
    cell_cols = [
        "prevalence",
        "ruler",
        "degree",
        "signal",
        "hard",
        "nonmono",
        "fn_concentrated",
        "human_fn",
        "human_fp",
    ]
    agg = aggregate_runs(runs, cells)
    y = agg.rename(columns={"tp_mean": "mean", "tp_std": "sd", "tp_count": "n_seeds"})
    y = y.assign(table="yield", metric="tp")[
        [
            "table",
            "cell_id",
            *cell_cols,
            "strategy",
            "B",
            "metric",
            "mean",
            "sd",
            "n_seeds",
            "hard_tp_mean",
            "n_dec_mean",
            *(["ruler_source"] if "ruler_source" in agg else []),
        ]
    ]
    parts.append(y)
    for a, b in (
        ("S2", "S0"),
        ("S1", "S0"),
        ("S3", "S0"),
        ("S3", "S4"),
        ("S4", "S0"),
        ("S5", "S0"),
        ("S2x", "S0"),
        ("S2", "S2x"),
    ):
        for metric in ("tp", "hard_tp"):
            pt = paired_table(runs, a, b, metric)
            if len(pt):
                pt = (
                    pt.assign(table="paired", strategy=f"{a}-{b}")
                    .rename(columns={"lo": "ci_lo", "hi": "ci_hi"})
                    .drop(columns=["comparison"])
                    .merge(cells, on="cell_id", how="left")
                )
                parts.append(pt)
    cov = runs[(runs["strategy"].isin(["S2", "S3", "S4"])) & (runs["B"] > 0)]
    if len(cov):
        g = (
            cov.groupby(["cell_id", "strategy", "B"])
            .agg(
                coverage=("covered", "mean"),
                coverage_hard=("covered_hard", "mean"),
                width=("hi", "mean"),
                lo_mean=("lo", "mean"),
                est=("est", "mean"),
                truth=("truth", "mean"),
                n_seeds=("covered", "count"),
            )
            .reset_index()
        )
        g["width"] = g["width"] - g["lo_mean"]
        parts.append(
            g.drop(columns=["lo_mean"])
            .assign(table="coverage")
            .merge(cells, on="cell_id", how="left")
        )
    s1 = runs[(runs["strategy"] == "S1") & (runs["B"] == 0)]
    if len(s1):
        cols = [
            c
            for c in (
                "covered",
                "covered_uncapped",
                "est",
                "est_uncapped",
                "truth",
                "n_dec",
                "rounds",
                "elapsed_s",
            )
            if c in s1
        ]
        g = s1.groupby("cell_id")[cols].mean().reset_index()
        g["n_seeds"] = s1.groupby("cell_id").size().to_numpy()
        g = g.rename(
            columns={"covered": "coverage", "covered_uncapped": "coverage_uncapped"}
        )
        parts.append(
            g.assign(table="s1_estimate", strategy="S1").merge(
                cells, on="cell_id", how="left"
            )
        )
    if len(stops):
        g = (
            stops.groupby(["cell_id", "cost", "rule"])
            .agg(
                regret=("regret", "mean"),
                k_stop=("k_stop", "mean"),
                found=("found", "mean"),
                value=("value", "mean"),
                n_seeds=("regret", "count"),
            )
            .reset_index()
        )
        parts.append(g.assign(table="regret").merge(cells, on="cell_id", how="left"))
    if len(audit):
        g = (
            audit.groupby(["cell_id", "B"])
            .agg(
                n_hfn=("n_hfn", "sum"),
                recovered=("recovered", "sum"),
                extras=("extras", "sum"),
                n_seeds=("seed", "nunique"),
            )
            .reset_index()
        )
        g["recovery"] = g["recovered"] / g["n_hfn"].replace(0, np.nan)
        g["precision"] = g["recovered"] / g["extras"].replace(0, np.nan)
        parts.append(
            g.assign(table="audit").merge(
                cells, on="cell_id", how="left", suffixes=("_sum", "")
            )
        )
    v = pd.DataFrame(list(verdicts)).rename(columns={"id": "hypothesis"})
    if len(v):
        parts.append(v.assign(table="hypothesis"))
    out = pd.concat(parts, ignore_index=True, sort=False)
    first = ["table", "scope", "hypothesis", "status", "cell_id", "strategy", "B"]
    cols = [c for c in first if c in out] + [c for c in out.columns if c not in first]
    return out[cols]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


COMPOSITE_COLUMNS = (
    "nome_score_total",
    "mae_score_total",
    "mae_missing",
    "dtnasc_score_total",
    "endereco_score_total",
    "municipio_score",
    "dtnasc_all_zero",
    "endereco_zero",
)
RULER_VARIANTS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("stub, sd_gamma=0.5 (S2 do protocolo)", {}),
    ("stub, sd_gamma=0.1", {"sd_gamma": 0.1}),
    ("stub, sd_gamma=0.02", {"sd_gamma": 0.02}),
    ("stub, sd_gamma=2", {"sd_gamma": 2.0}),
    ("Platt do extrato inteiro, sd_gamma=0.5", {"ruler": "full"}),
    ("Platt do extrato inteiro, sd_gamma=0.1", {"ruler": "full", "sd_gamma": 0.1}),
)


def _row(
    test: str, variant: str, metric: str, value: float, b: int = 0
) -> dict[str, Any]:
    return {
        "table": "real_confirm",
        "scope": "real",
        "test": test,
        "variant": variant,
        "B": b,
        "metric": metric,
        "value": float(value),
    }


def real_confirmations(inst: Instance, placebo_seeds: int = 10) -> list[dict[str, Any]]:
    """Confirmações EXPLORATÓRIAS do ganho de S2 no extrato real (só agregados).

    1. coeficientes do modelo em k = 25, 50, 200, 775 decisões;
    2. placebo: embaralha as linhas dos compostos (a nota e a verdade ficam);
    3. régua mais íngreme (Platt do extrato inteiro) e/ou ``sd_gamma`` menor;
    4. perfil dos positivos que a régua não alcança em B = 800 e que S2 alcança.
    """
    rows: list[dict[str, Any]] = []
    steps = min(MAX_PATH, inst.n)
    order = ruler_order(inst.nota)
    spec = build_spec(inst)
    path = adaptive_path(spec, inst.human, steps)
    rows.append(
        _row(
            "ruler",
            spec.ruler_source,
            "stub=1_platt=0",
            float(spec.ruler_source == "stub"),
        )
    )
    # 1. coeficientes ------------------------------------------------------
    for k in (25, 50, 200, 775):
        params = model_at(path, k)
        theta = params[2:]
        top = np.argsort(-np.abs(theta))[:5]
        rows.append(_row("coef", f"k={k}", "gamma", params[0]))
        rows.append(_row("coef", f"k={k}", "delta", params[1]))
        rows.append(
            _row(
                "coef",
                f"k={k}",
                "n_coef_abs_gt_0.1",
                float((np.abs(theta) > 0.1).sum()),
            )
        )
        for r, j in enumerate(top, start=1):
            rows.append(
                _row("coef", f"k={k}", f"rank{r}:{spec.feature_names[j]}", theta[j])
            )
    # 2. placebo -----------------------------------------------------------
    for b in BUDGETS:
        rows.append(_row("placebo", "S0", "tp", verified(inst, order[:b])[0], b))
        rows.append(
            _row(
                "placebo",
                "S2x sem embaralhar",
                "tp",
                verified(inst, path.chosen[:b])[0],
                b,
            )
        )
    tp_shuffled: dict[int, list[int]] = {b: [] for b in BUDGETS}
    for s in range(placebo_seeds):
        perm = np.random.default_rng(np.random.SeedSequence([s, 99])).permutation(
            inst.n
        )
        df2 = inst.df.copy()
        cols = list(COMPOSITE_COLUMNS)
        df2[cols] = inst.df[cols].to_numpy()[perm]
        shuffled = replace(inst, df=df2)
        sp = build_spec(shuffled)
        pt = adaptive_path(sp, inst.human, steps)
        for b in BUDGETS:
            tp_shuffled[b].append(verified(inst, pt.chosen[:b])[0])
    for b in BUDGETS:
        v = np.asarray(tp_shuffled[b], dtype=float)
        rows.append(
            _row("placebo", "S2x compostos embaralhados (media)", "tp", v.mean(), b)
        )
        rows.append(
            _row("placebo", "S2x compostos embaralhados (max)", "tp", v.max(), b)
        )
    # 3. régua mais íngreme ------------------------------------------------
    for name, kw in RULER_VARIANTS:
        sp = build_spec(inst, **kw)
        pt = adaptive_path(sp, inst.human, steps)
        for b in BUDGETS:
            rows.append(
                _row(
                    "ruler_variant",
                    f"{name} [{sp.ruler_source}]",
                    "tp",
                    verified(inst, pt.chosen[:b])[0],
                    b,
                )
            )
    # 4. grupo explicativo -------------------------------------------------
    b = max(BUDGETS)
    levels = discretize_fields(inst.df, "composites")
    s0_set = np.zeros(inst.n, dtype=bool)
    s0_set[order[:b]] = True
    s2_set = np.zeros(inst.n, dtype=bool)
    s2_set[path.chosen[:b]] = True
    groups = {
        "S0 alcanca": inst.y & s0_set,
        "S0 nao alcanca": inst.y & ~s0_set,
        "S0 nao alcanca, S2 alcanca": inst.y & ~s0_set & s2_set,
        "S0 nao alcanca, S2 nao alcanca": inst.y & ~s0_set & ~s2_set,
        "S2 alcanca": inst.y & s2_set,
    }
    mae_missing = inst.df["mae_missing"].to_numpy() == 1
    nome_l3 = levels.codes["nome"] == 3
    dt_zero = levels.codes["dtnasc"] == 0
    profile = mae_missing & nome_l3 & dt_zero
    for name, mask in groups.items():
        rows.append(_row("grupo", name, "n", mask.sum()))
        rows.append(_row("grupo", name, "mae_missing", (mask & mae_missing).sum()))
        rows.append(_row("grupo", name, "nome_L3", (mask & nome_l3).sum()))
        rows.append(_row("grupo", name, "dtnasc_zerada", (mask & dt_zero).sum()))
        rows.append(
            _row(
                "grupo",
                name,
                "perfil_mae_missing_nomeL3_dtnasc_zerada",
                (mask & profile).sum(),
            )
        )
        for code in range(4):
            rows.append(
                _row(
                    "grupo",
                    name,
                    f"endereco_L{code}",
                    (mask & (levels.codes["endereco"] == code)).sum(),
                )
            )
        if mask.any():
            rows.append(_row("grupo", name, "nota_min", inst.nota[mask].min()))
            rows.append(_row("grupo", name, "nota_max", inst.nota[mask].max()))
    reached = groups["S0 nao alcanca, S2 alcanca"]
    if reached.any():
        lo, hi = float(inst.nota[reached].min()), float(inst.nota[reached].max())
        band = (inst.nota >= lo) & (inst.nota <= hi)
        for name, mask in (
            ("perfil no pool todo", profile),
            ("perfil na faixa de nota do grupo S2", profile & band),
            ("faixa de nota do grupo S2 (sem perfil)", band & ~profile),
        ):
            rows.append(_row("perfil_rendimento", name, "pares", mask.sum()))
            rows.append(
                _row("perfil_rendimento", name, "positivos", (mask & inst.y).sum())
            )
            rows.append(
                _row(
                    "perfil_rendimento",
                    name,
                    "rendimento",
                    float((mask & inst.y).sum()) / max(1, int(mask.sum())),
                )
            )
        marg = order[400:b]
        rows.append(
            _row(
                "perfil_rendimento",
                "margem de S0 (postos 400-800)",
                "rendimento",
                float(inst.y[marg].mean()),
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Replicação por passo de blocagem (seção 8 do documento; pré-registrada)
# ---------------------------------------------------------------------------

PASSOS: tuple[int, ...] = (6, 9, 11)
REPLICATION_BUDGETS: tuple[int, ...] = (100, 200, 400)
TRAIN_DECISIONS = 800
MIN_POSITIVES = 5  # dobra com menos positivos é não informativa
RULERS = {"P": "pool", "C": "full"}  # P = stub (nota/10); C = Platt do extrato inteiro


def subspec(spec: ModelSpec, idx: NDArray[np.int64]) -> ModelSpec:
    """Mesmo modelo restrito às linhas ``idx`` (colunas e régua iguais)."""
    return replace(spec, X=spec.X[idx], offset=spec.offset[idx])


def fit_transfer_model(
    train_spec: ModelSpec, human_train: NDArray[np.bool_], steps: int = TRAIN_DECISIONS
) -> NDArray[np.float64]:
    """T, treino: S2 com ``steps`` decisões humanas; devolve o modelo final.

    Só recebe a especificação e as decisões do **treino**; o teste não entra.
    O modelo final é reajustado com todas as decisões feitas.
    """
    path = adaptive_path(train_spec, human_train, steps)
    labels = human_train[path.chosen].astype(np.float64)
    start = path.snapshots[max(path.snapshots)]
    return fit_offset_logistic(train_spec, path.chosen, labels, start)


def order_with_model(
    test_spec: ModelSpec, params: NDArray[np.float64]
) -> NDArray[np.int64]:
    """T, teste: ordem do passo de teste pelo modelo; sem nenhum rótulo."""
    eta = test_spec.eta(params)
    return np.lexsort((np.arange(len(eta)), -eta)).astype(np.int64)


def found_in(
    y: NDArray[np.bool_], human: NDArray[np.bool_], order: NDArray[np.int64], b: int
) -> int:
    """Verdadeiros verificados entre as ``b`` primeiras decisões da ordem."""
    top = order[:b]
    return int((human[top] & y[top]).sum())


def r4_signals(spec: ModelSpec, params: NDArray[np.float64]) -> dict[str, float]:
    """Pesos do modelo de treino para o padrão "dados ausentes" (R4).

    * ``nome_L3``: coeficiente de ``nome:L3`` (relativo à referência);
    * ``dtnasc_zero``: menos a média dos coeficientes dos níveis presentes de data
      de nascimento acima de zero (positivo quando o nível zero pesa mais a favor
      de par que os demais): o nível zero é a referência da codificação;
    * ``mae_missing_literal``: coeficiente de ``mae:missing``; é ``NaN`` quando
      ``missing`` é a própria referência da codificação (não há L0 em ``mae``);
    * ``mae_missing``: o mesmo, mas, se ``missing`` for a referência, menos a média
      dos demais níveis de ``mae`` (leitura invariante à referência, como em
      ``dtnasc_zero``).

    A diferença entre as duas leituras de ``mae:missing`` é uma ambiguidade do
    critério R4; ambas são reportadas e ``evaluate_replication`` a trata.
    """
    theta = dict(zip(spec.feature_names, params[2:], strict=True))
    dt = [v for k, v in theta.items() if k.startswith("dtnasc:")]
    mae = [v for k, v in theta.items() if k.startswith("mae:")]
    literal = float(theta["mae:missing"]) if "mae:missing" in theta else math.nan
    invariant = literal
    if math.isnan(literal) and mae:
        invariant = float(-np.mean(mae))
    return {
        "mae_missing_literal": literal,
        "mae_missing": invariant,
        "nome_L3": float(theta.get("nome:L3", math.nan)),
        "dtnasc_zero": float(-np.mean(dt)) if dt else math.nan,
    }


def replicate_fold(
    inst: Instance, test_passo: int, spec: ModelSpec, budgets: Sequence[int]
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """T, O e S0 numa dobra (passo de teste); humano perfeito, determinístico."""
    passo = pd.to_numeric(inst.df["PASSO"], errors="coerce").to_numpy()
    test = np.flatnonzero(passo == test_passo).astype(np.int64)
    train = np.flatnonzero(passo != test_passo).astype(np.int64)
    y_test, h_test = inst.y[test], inst.human[test]
    base: dict[str, Any] = {
        "fold": test_passo,
        "pairs": int(test.size),
        "positives": int(y_test.sum()),
        "ruler_source": spec.ruler_source,
    }
    s0_order = ruler_order(inst.nota[test])
    params = fit_transfer_model(subspec(spec, train), inst.human[train])
    signals = r4_signals(spec, params)
    t_order = order_with_model(subspec(spec, test), params)
    test_inst = Instance(
        df=inst.df.iloc[test].reset_index(drop=True),
        y=y_test,
        hard=np.zeros(test.size, dtype=bool),
        nota=inst.nota[test],
        llm_all=np.full(test.size, NONE, dtype=np.int8),
        llm_seen=np.full(test.size, NONE, dtype=np.int8),
        human=h_test,
        pair_ids=[inst.pair_ids[i] for i in test],
    )
    o_spec = subspec(spec, test)
    steps = min(max(budgets), int(test.size))
    o_path = adaptive_path(o_spec, h_test, steps)
    rows = []
    for b in budgets:
        bb = min(b, int(test.size))
        o_s2 = strategy_at_budget(test_inst, o_path, o_spec, bb, seed=0)["tp"]
        rows.append(
            {
                **base,
                "B": b,
                "S0": found_in(y_test, h_test, s0_order, bb),
                "T": found_in(y_test, h_test, t_order, bb),
                "O": found_in(y_test, h_test, o_path.chosen, bb),
                "O_com_amostra": int(o_s2),
            }
        )
    return rows, signals


def run_replication(
    inst: Instance, budgets: Sequence[int] = REPLICATION_BUDGETS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Todas as dobras e as duas réguas: ``found`` por B e sinais de R4."""
    found: list[dict[str, Any]] = []
    sig: list[dict[str, Any]] = []
    for variant, ruler in RULERS.items():
        spec = build_spec(inst, ruler=ruler)
        for passo in PASSOS:
            rows, signals = replicate_fold(inst, passo, spec, budgets)
            found += [{**r, "ruler": variant} for r in rows]
            sig.append(
                {
                    "fold": passo,
                    "ruler": variant,
                    "ruler_source": spec.ruler_source,
                    "positives": rows[0]["positives"],
                    **signals,
                }
            )
    return pd.DataFrame(found), pd.DataFrame(sig)


def evaluate_replication(
    found: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    min_positives: int = MIN_POSITIVES,
) -> list[dict[str, Any]]:
    """Critérios R1-R4 da seção 8, exatamente como pré-registrados."""
    inf = found[found["positives"] >= min_positives]
    sg = signals[signals["positives"] >= min_positives]
    out: list[dict[str, Any]] = []

    def total(ruler: str, b: int, col: str) -> int:
        sub = inf[(inf["ruler"] == ruler) & (inf["B"] == b)]
        return int(sub[col].sum())

    # R1
    gain = {b: total("C", b, "T") - total("C", b, "S0") for b in REPLICATION_BUDGETS}
    if gain[400] < 0:
        r1 = "refutada"
    elif gain[400] >= 3 and all(g >= 0 for g in gain.values()):
        r1 = "confirmada"
    else:
        r1 = "inconclusiva"
    out.append(
        {
            "id": "R1",
            "status": r1,
            "detail": "soma T-S0 (C, dobras informativas): "
            + ", ".join(f"B={b}: {g:+d}" for b, g in gain.items()),
        }
    )
    # R2
    c400 = inf[(inf["ruler"] == "C") & (inf["B"] == 400)]
    d = (c400["O"] - c400["S0"]).to_numpy()
    if len(d) == 0:
        r2 = "inconclusiva"
    elif (d < 0).any():
        r2 = "refutada"
    elif (d > 0).all():
        r2 = "confirmada"
    else:
        r2 = "inconclusiva"
    out.append(
        {
            "id": "R2",
            "status": r2,
            "detail": "O-S0 em B=400 por dobra (C): "
            + ", ".join(
                f"passo {int(f)}: {int(x):+d}"
                for f, x in zip(c400["fold"], d, strict=True)
            ),
        }
    )
    # R3
    p = {b: total("P", b, "T") for b in (200, 400)}
    c = {b: total("C", b, "T") for b in (200, 400)}
    if p[400] > c[400] and p[200] > c[200]:
        r3 = "confirmada"
    elif p[400] <= c[400] and p[200] <= c[200]:
        r3 = "refutada"
    else:
        r3 = "inconclusiva"
    out.append(
        {
            "id": "R3",
            "status": r3,
            "detail": "encontrados em T somados, P vs C: "
            + ", ".join(f"B={b}: {p[b]} vs {c[b]}" for b in (200, 400)),
        }
    )
    # R4: o texto não diz a régua nem como ler `mae:missing` quando é a
    # referência; avalia-se cada régua e cada leitura e só se decide se todas
    # concordam
    lines = []
    verdicts = []
    for reading, col in (
        ("literal", "mae_missing_literal"),
        ("invariante", "mae_missing"),
    ):
        if col not in sg.columns:
            col = "mae_missing"
        for ruler, g in sg.groupby("ruler"):
            flags = []
            for r in g.to_dict("records"):
                k = sum(
                    1
                    for v in (r[col], r["nome_L3"], r["dtnasc_zero"])
                    if not math.isnan(v) and v > 0
                )
                flags.append(k >= 2)
                lines.append(
                    f"{ruler}/{reading}/passo {int(r['fold'])}: mae:missing "
                    f"{r[col]:+.2f}, nome:L3 {r['nome_L3']:+.2f}, dtnasc zero "
                    f"{r['dtnasc_zero']:+.2f} ({k} de 3 a favor)"
                )
            if flags and all(flags):
                verdicts.append("confirmada")
            elif flags and not any(flags):
                verdicts.append("refutada")
            else:
                verdicts.append("inconclusiva")
    if verdicts and len(set(verdicts)) == 1:
        r4 = verdicts[0]
    else:
        r4 = "inconclusiva — critério mal especificado"
    out.append(
        {
            "id": "R4",
            "status": r4,
            "detail": "veredito por (leitura, régua): "
            + ", ".join(verdicts)
            + " | "
            + "; ".join(lines),
        }
    )
    return out


def replication_csv(
    found: pd.DataFrame, signals: pd.DataFrame, verdicts: Sequence[Mapping[str, Any]]
) -> pd.DataFrame:
    """CSV longo só com agregados (por dobra, régua e B)."""
    f = found.assign(table="found")
    s = signals.assign(table="r4_signals")
    v = pd.DataFrame(list(verdicts)).rename(columns={"id": "criterion"})
    return pd.concat([f, s, v.assign(table="criteria")], ignore_index=True, sort=False)


def replicate_main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(prog="replicate-passo")
    ap.add_argument("--real-input", type=Path, required=True)
    ap.add_argument(
        "--fixture",
        type=Path,
        default=Path("data/llm_fixtures/haiku-5-5_v4_sem-nota.jsonl"),
    )
    ap.add_argument(
        "--rates-csv",
        type=Path,
        default=Path("data/llm_fixtures/haiku-5-5_v4_sem-nota.rates.csv"),
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path("docs/engenharia_de_filas_replicacao_passo.csv"),
    )
    args = ap.parse_args(list(argv))
    inst = load_real_instance(args.real_input, args.fixture, args.rates_csv, 0)
    passo = pd.to_numeric(inst.df["PASSO"], errors="coerce")
    for pv in PASSOS:
        m = (passo == pv).to_numpy()
        print(f"passo {pv}: {int(m.sum())} pares, {int((inst.y & m).sum())} positivos")
    t0 = time.perf_counter()
    found, signals = run_replication(inst)
    verdicts = evaluate_replication(found, signals)
    table = replication_csv(found, signals, verdicts)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_csv, index=False, float_format="%.6g")
    print(found.to_string(index=False))
    print(signals.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    for v in verdicts:
        print(f"{v['id']} {v['status']} | {v['detail']}")
    log(f"replicação em {time.perf_counter() - t0:.0f}s -> {args.out_csv}")
    return 0


def render_confirm(rows: Sequence[Mapping[str, Any]]) -> str:
    df = pd.DataFrame(list(rows))
    return df.to_string(index=False, float_format=lambda v: f"{v:.3g}")


def cells_frame(cells: Sequence[Cell]) -> pd.DataFrame:
    rows = [{**asdict(c), "cell_id": c.cell_id} for c in cells]
    return pd.DataFrame(rows)


def collect(
    results: Sequence[Mapping[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def frame(key: str) -> pd.DataFrame:
        rows = [r for res in results for r in res[key]]
        return pd.DataFrame(rows)

    return frame("runs"), frame("stops"), frame("audit")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--n", type=int, default=N_CANDIDATES)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    p.add_argument(
        "--smoke", action="store_true", help="grade mínima (3 células, 2 sementes)"
    )
    p.add_argument("--sigma", type=float, default=SIGMA_COEF)
    p.add_argument(
        "--subset", choices=("completo", "sensibilidade"), default="completo"
    )
    p.add_argument("--no-s1", action="store_true", help="não roda a S1 (sintético)")
    p.add_argument("--real-input", type=Path, default=None)
    p.add_argument(
        "--fixture",
        type=Path,
        default=Path("data/llm_fixtures/haiku-5-5_v4_sem-nota.jsonl"),
    )
    p.add_argument(
        "--rates-csv",
        type=Path,
        default=Path("data/llm_fixtures/haiku-5-5_v4_sem-nota.rates.csv"),
    )
    p.add_argument("--real-seeds", type=int, default=10)
    p.add_argument("--real-audit-draws", type=int, default=200)
    p.add_argument(
        "--real-confirm",
        action="store_true",
        help="confirmações exploratórias de S2 no real (coeficientes, placebo, régua)",
    )
    p.add_argument(
        "--out-csv", type=Path, default=Path("docs/engenharia_de_filas_resultados.csv")
    )
    p.add_argument(
        "--out-md", type=Path, default=None, help="fragmento markdown com as tabelas"
    )
    return p


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] == "replicate-passo":
        return replicate_main(raw[1:])
    args = build_parser().parse_args(argv)
    cells = build_cells(args.n)
    seeds = list(range(args.seeds))
    if args.smoke:
        cells = [
            c
            for c in cells
            if c.cell_id in {build_cells(args.n)[i].cell_id for i in (0, 2, 4)}
        ]
        seeds = seeds[:2]
    if args.subset == "sensibilidade":
        keep_ids = {
            c.cell_id
            for c in (
                Cell(n=args.n),
                Cell(n=args.n, prevalence=0.05),
                Cell(n=args.n, prevalence=0.05, signal=True),
                Cell(n=args.n, signal=True),
                Cell(n=args.n, ruler="wrong", degree=3),
            )
        }
        cells = [c for c in cells if c.cell_id in keep_ids]
    tasks: list[Any] = [
        (c, s, not args.no_s1, args.sigma) for c in cells for s in seeds
    ]
    t0 = time.perf_counter()
    results: list[dict[str, Any]] = []
    real_results: list[dict[str, Any]] = []
    _REAL_ARGS.update(path=args.real_input, fixture=args.fixture, rates=args.rates_csv)
    log(
        f"{len(cells)} células x {len(seeds)} sementes = {len(tasks)} tarefas; "
        f"{args.workers} processos"
    )
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init_real, initargs=(dict(_REAL_ARGS),)
    ) as pool:
        futs = []
        if args.real_input is not None:
            futs += [
                pool.submit(run_real_seed, (s, True)) for s in range(args.real_seeds)
            ]
        futs += [pool.submit(run_cell_seed, t) for t in tasks]
        for done, fut in enumerate(as_completed(futs), start=1):
            res = fut.result()
            is_real = res["cell_id"] == "real_below_cutoff"
            (real_results if is_real else results).append(res)
            if done % 20 == 0 or done == len(futs):
                log(f"{done}/{len(futs)} tarefas ({time.perf_counter() - t0:.0f}s)")
    log(f"grade concluída em {time.perf_counter() - t0:.0f}s")
    runs, stops, audit = collect(results)
    cframe = cells_frame(cells)
    verdicts = evaluate(runs, stops, audit, cframe)
    if real_results:
        rruns, rstops, _ = collect(real_results)
        raudit = pd.DataFrame(
            real_audit_rows(
                args.real_input,
                args.fixture,
                args.rates_csv,
                args.real_audit_draws,
                (0.05, 0.10),
            )
        )
        rcells = pd.DataFrame(
            [
                {
                    "cell_id": "real_below_cutoff",
                    "prevalence": float(rruns["prevalence"].iloc[0]),
                    "ruler": "correct",
                    "degree": 0,
                    "signal": False,
                    "hard": False,
                    "nonmono": False,
                    "fn_concentrated": False,
                    "human_fn": 0.0,
                    "human_fp": 0.0,
                    "n": int(rruns["n"].iloc[0]),
                }
            ]
        )
        rcells = pd.concat(
            [
                rcells,
                pd.DataFrame(
                    [
                        {
                            "cell_id": c,
                            "prevalence": rcells["prevalence"].iloc[0],
                            "ruler": "correct",
                            "degree": 0,
                            "signal": False,
                            "hard": False,
                            "nonmono": False,
                            "fn_concentrated": False,
                            "human_fn": f,
                            "human_fp": 0.01,
                            "n": rcells["n"].iloc[0],
                        }
                        for c, f in (("real_h0.05", 0.05), ("real_h0.1", 0.1))
                    ]
                ),
            ],
            ignore_index=True,
        )
        real_v = evaluate_real(rruns, rstops, raudit, rcells)
        runs = pd.concat([runs, rruns], ignore_index=True)
        stops = pd.concat([stops, rstops], ignore_index=True)
        audit = pd.concat([audit, raudit], ignore_index=True)
        cframe = pd.concat([cframe, rcells], ignore_index=True)
        verdicts = [*verdicts, *real_v]
    table = build_csv(runs, stops, audit, cframe, verdicts)
    if args.real_confirm and args.real_input is not None:
        base = load_real_instance(args.real_input, args.fixture, args.rates_csv, 0)
        confirm = real_confirmations(base)
        print(render_confirm(confirm))
        table = pd.concat([table, pd.DataFrame(confirm)], ignore_index=True, sort=False)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_csv, index=False, float_format="%.6g")
    log(f"escrito {args.out_csv} ({len(table)} linhas)")
    for v in verdicts:
        print(f"{v['scope']:10s} {v['id']:4s} {v['status']:14s} {v['detail']}")
    if args.out_md is not None:
        args.out_md.write_text(
            render_tables(runs, stops, audit, cframe, verdicts), encoding="utf-8"
        )
    return 0


def _init_real(args: dict[str, Any]) -> None:
    _REAL_ARGS.update(args)


def evaluate_real(
    runs: pd.DataFrame, stops: pd.DataFrame, audit: pd.DataFrame, cells: pd.DataFrame
) -> list[dict[str, Any]]:
    """Critérios do extrato real: H1, H6, H7, H8 (lado testável) e H11."""
    keep = {"H1", "H6", "H7", "H8"}
    base = cells[cells["cell_id"] == "real_below_cutoff"]
    out = [
        v
        for v in evaluate(
            runs,
            stops,
            pd.DataFrame(),
            base,
            scope="real",
            base_ids=frozenset({"real_below_cutoff"}),
        )
        if v["id"] in keep
    ]
    h11 = [
        v for v in evaluate(runs, stops, audit, cells, scope="real") if v["id"] == "H11"
    ]
    return [*out, *h11]


def render_tables(
    runs: pd.DataFrame,
    stops: pd.DataFrame,
    audit: pd.DataFrame,
    cells: pd.DataFrame,
    verdicts: Sequence[Mapping[str, Any]],
) -> str:
    """Fragmento markdown com o rendimento médio por estratégia e B (agregados)."""
    agg = aggregate_runs(runs, cells)
    lines = [
        "| célula | estratégia | " + " | ".join(f"B={b}" for b in BUDGETS) + " |",
        "|---|---|" + "---|" * len(BUDGETS),
    ]
    for cid, g in agg.groupby("cell_id", sort=False):
        for strat in ("S0", "S1", "S2", "S2x", "S3", "S4", "S5"):
            h = g[g["strategy"] == strat].set_index("B")["tp_mean"]
            if h.empty:
                continue
            lines.append(
                f"| {cid} | {strat} | "
                + " | ".join(f"{h.get(b, math.nan):.1f}" for b in BUDGETS)
                + " |"
            )
    lines += ["", "| escopo | hipótese | status | números |", "|---|---|---|---|"]
    lines += [
        f"| {v['scope']} | {v['id']} | {v['status']} | {v['detail']} |"
        for v in verdicts
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
