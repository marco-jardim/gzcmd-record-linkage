"""Simulação offline da fila de revisão auto-calibrável sobre o extrato da tese.

Experimental. Sem rede e sem LLM real: o revisor é sintético
(``BandedOracle``) e o humano do portão é simulado. ``PAR`` entra **apenas** como
verdade do revisor simulado e do humano simulado; a sessão recebe um DataFrame
sem ``PAR`` e sem ``TARGET``. O portão humano simulado decide todos os itens
pelo ``PAR``. Nada em nível de par é escrito: só agregados.

Uso (a partir da raiz do repositório)::

    python notebooks/sim_fila_autocalibravel.py \\
        --input ../phd-research/data/COMPARADORSEMIDENT.csv \\
        --seeds 10 --max-reviews 2000 --scenarios perfect,thesis_bands \\
        --field-sets composites \\
        --out-csv docs/sim_fila_autocalibravel_resultados.csv \\
        --out-md docs/sim_fila_autocalibravel.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple, cast

import numpy as np
import pandas as pd
from nb_fila import CONTAGENS_LLM_HAIKU_V4

from gzcmd_record_linkage.arranjo_corte import THESIS_PARTITION, below_cutoff
from gzcmd_record_linkage.calibration import get_nota_series
from gzcmd_record_linkage.fellegi_sunter import (
    FIELD_SETS,
    FellegiSunterError,
    discretize_fields,
    fit_fellegi_sunter,
)
from gzcmd_record_linkage.llm_dossier import DossierView, make_pair_id
from gzcmd_record_linkage.llm_review import ReviewResult
from gzcmd_record_linkage.loader import LoadConfig, load_comparador_csv
from gzcmd_record_linkage.self_calibrating_queue import (
    ExperimentalWarning,
    OracleReviewer,
    SelfCalibratingConfig,
    SelfCalibratingSession,
)

__all__ = [
    "BANDS",
    "BASELINES",
    "REAL_REVIEWERS",
    "RealInputs",
    "ReplayReviewer",
    "smoothed_rates_from_counts",
    "EFFORT_BASELINES",
    "Job",
    "load_rates_csv",
    "BandedOracle",
    "Prepared",
    "SCENARIOS",
    "aggregate",
    "build_jobs",
    "effort_rows_baseline",
    "human_metrics",
    "baseline_orders",
    "band_rates",
    "build_scenario_rates",
    "metrics_from_flags",
    "prepare",
    "run_grid",
    "run_one",
    "render_markdown",
    "main",
]

# Faixas de nota do perfil v3.1: (nome, limite inferior, limite superior).
BANDS: tuple[tuple[str, float, float], ...] = (
    ("low", -np.inf, 5.0),
    ("grey_low", 5.0, 6.0),
    ("grey_mid", 6.0, 7.0),
    ("grey_high", 7.0, 8.0),
    ("near_high", 8.0, 9.0),
    ("high", 9.0, np.inf),
)
BAND_NAMES: tuple[str, ...] = tuple(b[0] for b in BANDS)
_EDGES = np.array([b[1] for b in BANDS[1:]], dtype=float)

SCENARIOS: tuple[str, ...] = ("perfect", "thesis_bands")
# revisores calibrados no Haiku real (taxas medidas e replay do fixture)
REAL_REVIEWERS: tuple[str, ...] = ("measured", "measured_smoothed", "fixture_replay")
DISAGREEMENT_THESIS = 0.10
MARKS: tuple[int, ...] = (100, 250, 500, 1000, 2000)
K_TARGETS: tuple[int, ...] = (5, 10, 20)
POOLS: tuple[str, ...] = ("all", "below_cut")
SENSITIVITY: dict[str, Any] = {"tau": 0.01, "window": 300}
DEFAULT_FIELD_SET = "composites"
NEGATIVE_LABELS: tuple[str, ...] = ("llm_nonmatch", "verified_only")
HUMAN_BUDGETS: tuple[int, ...] = (25, 50, 100)
DEFAULT_NEGATIVE = "llm_nonmatch"
DEFAULT_HUMAN_BUDGET = 50
# orçamentos de decisões humanas das comparações de esforço humano
HUMAN_MARKS: tuple[int, ...] = (50, 100, 200, 400)
HUMAN_K: tuple[int, ...] = (5, 10)
# H1 varre a ordem da nota até gerar este nº de MATCH (ou esgotar o pool)
H1_SCAN_CAP = 30_000

Rates = tuple[float, float, float]  # (fp, fn, unsure)


# ---------------------------------------------------------------------------
# Taxas por faixa
# ---------------------------------------------------------------------------


def band_rates(
    counts: Mapping[str, tuple[int, int, int, int, int, int]],
) -> dict[str, Rates]:
    """Taxas com suavização de Laplace por faixa, a partir das contagens.

    ``counts[faixa] = (decisões, pares, não pares, FP, FN, UNSURE)``:
    fp = (FP+1)/(não pares+2), fn = (FN+1)/(pares+2),
    unsure = (UNSURE+1)/(decisões+2). A faixa ``low`` sem dados herda as taxas
    de ``grey_low`` (suposição declarada).
    """
    out: dict[str, Rates] = {}
    for name, (n, pos, neg, fp, fn, unsure) in counts.items():
        out[name] = ((fp + 1) / (neg + 2), (fn + 1) / (pos + 2), (unsure + 1) / (n + 2))
    if "low" not in out:
        out["low"] = out["grey_low"]
    missing = [b for b in BAND_NAMES if b not in out]
    if missing:
        raise ValueError(f"faixas sem taxas: {missing}")
    return out


def load_rates_csv(path: Path) -> dict[str, Rates]:
    """Taxas por faixa de um CSV com colunas ``band, fp, fn, unsure``.

    Taxas em [0, 1]; a faixa ``low`` sem linha herda ``grey_low``; faixa
    desconhecida, duplicada ou taxa fora de [0, 1] levanta ``ValueError``.
    """
    frame = pd.read_csv(path)
    missing = {"band", "fp", "fn", "unsure"} - set(frame.columns)
    if missing:
        raise ValueError(f"colunas ausentes em {path}: {sorted(missing)}")
    bands = frame["band"].astype(str).str.strip()
    unknown = sorted(set(bands) - set(BAND_NAMES))
    if unknown:
        raise ValueError(f"faixas desconhecidas: {unknown}")
    if bands.duplicated().any():
        raise ValueError("faixa duplicada no CSV de taxas")
    out: dict[str, Rates] = {}
    for name, fp, fn, unsure in zip(
        bands, frame["fp"], frame["fn"], frame["unsure"], strict=True
    ):
        rate = (float(fp), float(fn), float(unsure))
        if not all(0.0 <= r <= 1.0 for r in rate):
            raise ValueError(f"taxas fora de [0, 1] na faixa {name}")
        out[str(name)] = rate
    if "low" not in out and "grey_low" in out:
        out["low"] = out["grey_low"]
    absent = [b for b in BAND_NAMES if b not in out]
    if absent:
        raise ValueError(f"faixas sem taxas: {absent}")
    return out


@dataclass(frozen=True)
class RealInputs:
    """Insumos locais (fora do git) dos revisores calibrados no Haiku real."""

    measured_rates_csv: Path | None = None  # band,fp,fn,unsure (sem suavização)
    counts_csv: Path | None = None  # relatório de calibração (contagens por faixa)
    fixture_jsonl: Path | None = None  # decisões reais gravadas (replay)


def smoothed_rates_from_counts(path: Path) -> dict[str, Rates]:
    """Taxas por faixa com Jeffreys, a partir das contagens da calibração real.

    Lê as linhas ``lado = todos`` (os dois lados do corte juntos) do CSV de
    calibração e aplica (a + 0,5) / (n + 1): fp = (FP + 0,5) / (não pares + 1),
    fn = (FN + 0,5) / (pares + 1), unsure = (UNSURE_pares + UNSURE_não_pares
    + 0,5) / (pares + não pares + 1). Faixas com poucos casos (``low`` com 3
    pares verdadeiros, ``high`` com 2 não pares) ficam bem suavizadas.
    """
    frame = pd.read_csv(path)
    need = {
        "faixa",
        "lado",
        "n_pares",
        "n_nao_pares",
        "fn",
        "fp",
        "unsure_pares",
        "unsure_nao_pares",
    }
    missing = need - set(frame.columns)
    if missing:
        raise ValueError(f"colunas ausentes em {path}: {sorted(missing)}")
    out: dict[str, Rates] = {}
    rows = frame[frame["lado"].astype(str).str.strip() == "todos"]
    for r in rows.itertuples(index=False):
        band = str(r.faixa).strip()
        if band not in BAND_NAMES:
            continue
        n_pos, n_neg = int(r.n_pares), int(r.n_nao_pares)
        unsure = int(r.unsure_pares) + int(r.unsure_nao_pares)
        out[band] = (
            (int(r.fp) + 0.5) / (n_neg + 1),
            (int(r.fn) + 0.5) / (n_pos + 1),
            (unsure + 0.5) / (n_pos + n_neg + 1),
        )
    absent = [b for b in BAND_NAMES if b not in out]
    if absent:
        raise ValueError(f"faixas sem contagens: {absent}")
    return out


def build_scenario_rates(
    scenario: str,
    rates_csv: Path | None = None,
    inputs: RealInputs | None = None,
) -> tuple[dict[str, Rates], float]:
    """Taxas por faixa e ``disagreement_rate`` de um cenário de revisor.

    ``rates_csv`` é o cenário ``rates_csv`` (taxas por faixa de um CSV).
    ``measured``: o ``.rates.csv`` da calibração real, como está.
    ``measured_smoothed``: as mesmas contagens com Jeffreys. ``fixture_replay``
    usa estas últimas só para os pares ausentes do fixture (fallback).
    ``disagreement_rate`` = 0,10 nos cenários com erro (suposição).
    """
    if scenario == "rates_csv":
        if rates_csv is None:
            raise ValueError("cenário rates_csv exige --rates-csv")
        return load_rates_csv(rates_csv), DISAGREEMENT_THESIS
    if scenario == "measured":
        if inputs is None or inputs.measured_rates_csv is None:
            raise ValueError("cenário measured exige --measured-rates-csv")
        return load_rates_csv(inputs.measured_rates_csv), DISAGREEMENT_THESIS
    if scenario in ("measured_smoothed", "fixture_replay"):
        if inputs is None or inputs.counts_csv is None:
            raise ValueError(f"cenário {scenario} exige --counts-csv")
        return smoothed_rates_from_counts(inputs.counts_csv), DISAGREEMENT_THESIS
    if scenario == "perfect":
        return {b: (0.0, 0.0, 0.0) for b in BAND_NAMES}, 0.0
    if scenario == "thesis_bands":
        return band_rates(CONTAGENS_LLM_HAIKU_V4["vigilancia"]), DISAGREEMENT_THESIS
    raise ValueError(f"cenário desconhecido: {scenario!r}")


def band_of_nota(nota: np.ndarray) -> np.ndarray:
    """Índice da faixa (0..5) de cada nota, com limites [a, b)."""
    return np.searchsorted(_EDGES, nota, side="right")


# ---------------------------------------------------------------------------
# Revisor simulado por faixa
# ---------------------------------------------------------------------------


class BandedOracle:
    """``PairReviewer`` que roteia para um ``OracleReviewer`` por faixa da nota.

    A nota fica oculta ao LLM da sessão (``features.nota_final`` removido), mas o
    revisor simulado a conhece por ``pair_id`` para escolher as taxas da faixa.
    Cada ``OracleReviewer`` é determinístico por ``(seed, pair_id)``: o mesmo par
    recebe o mesmo veredito em qualquer método (comparação pareada).
    """

    def __init__(
        self,
        truth: Mapping[str, bool],
        nota_by_pair: Mapping[str, float],
        rates: Mapping[str, Rates],
        *,
        disagreement_rate: float,
        seed: int,
    ) -> None:
        self._nota = nota_by_pair
        self._by_band = [
            OracleReviewer(
                truth,
                fp_rate=rates[name][0],
                fn_rate=rates[name][1],
                unsure_rate=rates[name][2],
                disagreement_rate=disagreement_rate,
                seed=seed,
            )
            for name in BAND_NAMES
        ]
        self.band_counts: Counter[str] = Counter()

    def _band(self, pair_id: str) -> int:
        return int(band_of_nota(np.asarray([self._nota[pair_id]]))[0])

    def review(self, dossier: DossierView) -> ReviewResult:
        band = self._band(dossier.pair_id)
        self.band_counts[BAND_NAMES[band]] += 1
        return self._by_band[band].review(dossier)

    def verdict(self, pair_id: str) -> str:
        """Veredito (MATCH/NONMATCH/UNSURE) sem montar o dossiê (para as bases)."""
        stub = cast(DossierView, SimpleNamespace(pair_id=pair_id, to_dict=dict))
        band = self._band(pair_id)
        result = self._by_band[band].review(stub)
        self._by_band[band].seen_payloads.clear()
        return str(result.decision)


class ReplayReviewer:
    """Replay das decisões reais do fixture, com fallback por taxas por faixa.

    Envolve ``llm_real_eval.FixtureReviewer`` (``on_missing="rates"``): um par
    presente no fixture volta com a decisão real gravada (fixa entre sementes);
    um par ausente é julgado por um ``OracleReviewer`` da faixa (taxas do
    cenário, determinístico por semente e par). ``n_replayed`` e ``n_missing``
    contam as revisões vindas do fixture e do fallback.
    """

    def __init__(
        self,
        fixture_jsonl: Path,
        truth: Mapping[str, bool],
        nota_by_pair: Mapping[str, float],
        rates: Mapping[str, Rates],
        *,
        disagreement_rate: float,
        seed: int,
    ) -> None:
        import llm_real_eval  # lazy: o módulo é pesado e só o replay o usa

        names = np.asarray(BAND_NAMES, dtype=object)
        keys = list(nota_by_pair)
        bands = names[band_of_nota(np.asarray([nota_by_pair[k] for k in keys]))]
        band_by_pair = dict(zip(keys, bands.tolist(), strict=True))
        self._fixture = llm_real_eval.FixtureReviewer(
            fixture_jsonl,
            on_missing="rates",
            rates=dict(rates),
            truth=truth,
            band_by_pair=band_by_pair,
            seed=seed,
            disagreement_rate=disagreement_rate,
        )

    @property
    def n_replayed(self) -> int:
        return int(self._fixture.n_replayed)

    @property
    def n_missing(self) -> int:
        return int(self._fixture.n_missing)

    def review(self, dossier: DossierView) -> ReviewResult:
        return cast(ReviewResult, self._fixture.review(dossier))

    def verdict(self, pair_id: str) -> str:
        """Veredito (MATCH/NONMATCH/UNSURE) sem montar o dossiê (para as bases)."""
        stub = cast(DossierView, SimpleNamespace(pair_id=pair_id, to_dict=dict))
        return str(self._fixture.review(stub).decision)


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------


@dataclass
class Prepared:
    """Dados da simulação. ``df`` não tem ``PAR`` nem ``TARGET``."""

    df: pd.DataFrame
    pair_ids: np.ndarray
    nota: np.ndarray
    truth: np.ndarray  # PAR in {1, 2}; só para o revisor/humano simulados
    par2: np.ndarray  # PAR == 2
    below: np.ndarray  # abaixo do corte da tese
    partition: tuple[int, int, int]

    @property
    def n(self) -> int:
        return len(self.df)


def prepare(
    raw: pd.DataFrame,
    *,
    below: np.ndarray | None = None,
    check_partition: bool = True,
) -> Prepared:
    """Separa a verdade (``PAR``) do DataFrame entregue à sessão."""
    if "SCORE" not in raw.columns:
        legacy = [c for c in raw.columns if str(c).upper().startswith("SCORE,")]
        if legacy:  # cabeçalho dBase do extrato: ``SCORE,C,15,0``
            raw = raw.assign(SCORE=pd.to_numeric(raw[legacy[0]], errors="coerce"))
    par = pd.to_numeric(raw["PAR"], errors="coerce")
    truth = par.isin([1, 2]).to_numpy()
    par2 = par.eq(2).to_numpy()
    if below is None:
        expected = THESIS_PARTITION if check_partition else None
        below = below_cutoff(raw, expected=expected)
    below = np.asarray(below, dtype=bool)
    df = raw.drop(columns=[c for c in ("PAR", "TARGET") if c in raw.columns])
    df = df.reset_index(drop=True)
    nota = pd.to_numeric(get_nota_series(df), errors="coerce").to_numpy(dtype=float)
    pair_ids = np.array(
        [
            make_pair_id(c, r, p)
            for c, r, p in zip(df["COMPREC"], df["REFREC"], df["PASSO"], strict=True)
        ],
        dtype=object,
    )
    if len(set(pair_ids.tolist())) != len(pair_ids):
        raise ValueError("pair_id duplicado")
    partition = (
        int((~below).sum()),
        int(below.sum()),
        int((par2 & below).sum()),
    )
    return Prepared(df, pair_ids, nota, truth, par2, below, partition)


def load_prepared(path: Path, *, check_partition: bool = True) -> Prepared:
    raw = load_comparador_csv(path, cfg=LoadConfig(require_par=False))
    return prepare(raw, check_partition=check_partition)


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------


def _first_k(flags: np.ndarray, k: int) -> float:
    idx = np.flatnonzero(flags)
    return float(idx[k - 1] + 1) if len(idx) >= k else float("nan")


def metrics_from_flags(
    llm_match: np.ndarray,
    truth: np.ndarray,
    below: np.ndarray,
    par2: np.ndarray,
    budget: int,
) -> dict[str, float]:
    """Métricas nas primeiras ``budget`` revisões (arrays na ordem de revisão)."""
    b = min(int(budget), len(llm_match))
    m, t, lo, p2 = llm_match[:b], truth[:b], below[:b], par2[:b]
    found = m & t
    found_below = found & lo
    out: dict[str, float] = {
        "reviews": float(b),
        "true_reviewed": float(t.sum()),
        "par2_below_reviewed": float((t & lo & p2).sum()),
        "found": float(found.sum()),
        "found_below": float(found_below.sum()),
        "par2_below_found": float((found_below & p2).sum()),
        "fp_llm_match": float((m & ~t).sum()),
    }
    for k in K_TARGETS:
        out[f"reviews_to_k{k}"] = _first_k(found_below, k)
    return out


# ---------------------------------------------------------------------------
# Linhas de base
# ---------------------------------------------------------------------------

# bases a orçamento de LLM igual (continuidade com a v2)
BASELINES: tuple[str, ...] = ("nota", "fs_comp", "random")
# bases de esforço humano (orçamento = decisões humanas)
EFFORT_BASELINES: tuple[str, ...] = ("H0", "H1")


def _tie_broken_desc(score: np.ndarray, seed: int) -> np.ndarray:
    perm = np.random.default_rng(seed).permutation(len(score))
    return perm[np.argsort(-score[perm], kind="stable")]


def baseline_orders(
    df_pool: pd.DataFrame,
    nota_pool: np.ndarray,
    seed: int,
    fs_scores: Mapping[str, np.ndarray | None],
) -> dict[str, np.ndarray]:
    """Ordens (índices posicionais no pool) das bases a orçamento de LLM igual.

    ``fs_scores["fs_comp"]`` é o posterior do FS sobre os compostos; um FS que
    não ajustou (``None``) fica de fora.
    """
    out: dict[str, np.ndarray] = {"nota": _tie_broken_desc(nota_pool, seed)}
    scores = fs_scores.get("fs_comp")
    if scores is not None:
        out["fs_comp"] = _tie_broken_desc(scores, seed)
    out["random"] = np.random.default_rng(seed + 7919).permutation(len(df_pool))
    return out


def fs_posterior(df_pool: pd.DataFrame, field_set: str) -> np.ndarray | None:
    """Posterior do FS (EM sem rótulos); ``None`` se o EM degenerar."""
    levels = discretize_fields(df_pool, field_set)
    try:
        model = fit_fellegi_sunter(levels)
    except FellegiSunterError:
        return None
    return np.asarray(model.predict_proba(levels), dtype=float)


# ---------------------------------------------------------------------------
# Esforço humano
# ---------------------------------------------------------------------------


def human_metrics(
    seq_truth: np.ndarray,
    seq_below: np.ndarray,
    seq_llm: np.ndarray,
    budget: int,
) -> dict[str, float] | None:
    """Métricas nas primeiras ``budget`` decisões humanas.

    As três sequências estão na ordem das decisões: se o par é verdadeiro (o
    humano perfeito confirma exatamente os verdadeiros), se está abaixo do corte
    e quantas revisões de LLM já tinham sido consumidas. Devolve ``None`` se o
    método não chegou a ``budget`` decisões.
    """
    n = len(seq_truth)
    if budget < 1 or budget > n:
        return None
    t = seq_truth[:budget]
    vb = t & seq_below[:budget]
    verified = float(t.sum())
    out: dict[str, float] = {
        "human_decisions": float(budget),
        "verified": verified,
        "verified_below": float(vb.sum()),
        "llm_reviews": float(seq_llm[budget - 1]),
    }
    if verified > 0:
        out["dec_per_verified"] = budget / verified
    for k in HUMAN_K:
        out[f"hdec_to_k{k}"] = _first_k(vb, k)
    return out


def _human_rows(
    base: Mapping[str, Any],
    method: str,
    seq_truth: np.ndarray,
    seq_below: np.ndarray,
    seq_llm: np.ndarray,
) -> list[dict[str, Any]]:
    """Linhas por orçamento humano ``B{b}`` e ``run`` (a sequência inteira)."""
    rows: list[dict[str, Any]] = []
    for b in HUMAN_MARKS:
        m = human_metrics(seq_truth, seq_below, seq_llm, b)
        if m is not None:
            rows.append({**base, "method": method, "budget": f"B{b}", **m})
    return rows


def effort_rows_baseline(
    method: str,
    base: Mapping[str, Any],
    order: np.ndarray,
    pids: np.ndarray,
    truth: np.ndarray,
    below: np.ndarray,
    oracle: BandedOracle | ReplayReviewer,
    *,
    horizon: int = max(HUMAN_MARKS),
) -> list[dict[str, Any]]:
    """Bases de esforço humano sobre a ordem da nota (humano perfeito).

    H0: o humano revisa sozinho, 1 decisão por par, na ordem da nota.
    H1: o LLM tria na ordem da nota e o humano decide só os MATCH do LLM (1
    decisão por MATCH); o LLM revisa até gerar ``horizon`` MATCH (ou esgotar o
    pool, ou ``H1_SCAN_CAP`` revisões).
    """
    if method == "H0":
        top = order[:horizon]
        seq_llm = np.zeros(len(top))
    elif method == "H1":
        picked: list[int] = []
        used: list[int] = []
        for j, i in enumerate(order[:H1_SCAN_CAP]):
            if oracle.verdict(str(pids[i])) == "MATCH":
                picked.append(int(i))
                used.append(j + 1)
                if len(picked) >= horizon:
                    break
        top = np.asarray(picked, dtype=int)
        seq_llm = np.asarray(used, dtype=float)
    else:
        raise ValueError(f"base de esforço desconhecida: {method!r}")
    s_truth, s_below = truth[top], below[top]
    rows = _human_rows(base, method, s_truth, s_below, seq_llm)
    full = human_metrics(s_truth, s_below, seq_llm, len(top)) if len(top) else None
    if full is not None:
        rows.append({**base, "method": method, "budget": "run", **full})
    return rows


# ---------------------------------------------------------------------------
# Uma execução
# ---------------------------------------------------------------------------


def _pool_index(prep: Prepared, pool: str) -> np.ndarray:
    if pool == "all":
        return np.arange(prep.n)
    if pool == "below_cut":
        return np.flatnonzero(prep.below)
    raise ValueError(f"pool desconhecido: {pool!r}")


def _marks_for(n_run: int, n_pool: int) -> list[tuple[str, int]]:
    budgets: list[tuple[str, int]] = [("run", n_run)]
    budgets += [(str(m), m) for m in MARKS if m <= n_pool]
    return budgets


def run_one(
    prep: Prepared,
    pool: str,
    mode: str,
    scenario: str,
    seed: int,
    *,
    field_set: str = DEFAULT_FIELD_SET,
    negative_labels: str = DEFAULT_NEGATIVE,
    human_budget: int | None = DEFAULT_HUMAN_BUDGET,
    overrides: Mapping[str, Any] | None = None,
    fs_cache: dict[tuple[str, str], np.ndarray | None] | None = None,
    max_reviews: int = 2000,
    with_baselines: bool = True,
    with_effort: bool = False,
    rates_csv: Path | None = None,
    inputs: RealInputs | None = None,
) -> list[dict[str, Any]]:
    """Executa a sessão e as bases; devolve linhas de métricas agregáveis.

    No modo humano o humano simulado é perfeito e decide **todos** os itens do
    portão pelo ``PAR``. ``with_effort`` acrescenta as bases H0/H1.
    """
    idx = _pool_index(prep, pool)
    df_pool = prep.df.iloc[idx].reset_index(drop=True)
    pids = prep.pair_ids[idx]
    truth = prep.truth[idx]
    below = prep.below[idx]
    par2 = prep.par2[idx]
    nota = prep.nota[idx]
    truth_by_pid = dict(zip(pids.tolist(), truth.tolist(), strict=True))
    nota_by_pid = dict(zip(pids.tolist(), nota.tolist(), strict=True))
    rates, disagreement = build_scenario_rates(scenario, rates_csv, inputs)
    oracle: BandedOracle | ReplayReviewer
    if scenario == "fixture_replay":
        if inputs is None or inputs.fixture_jsonl is None:
            raise ValueError("cenário fixture_replay exige --fixture-jsonl")
        oracle = ReplayReviewer(
            inputs.fixture_jsonl,
            truth_by_pid,
            nota_by_pid,
            rates,
            disagreement_rate=disagreement,
            seed=seed,
        )
    else:
        oracle = BandedOracle(
            truth_by_pid,
            nota_by_pid,
            rates,
            disagreement_rate=disagreement,
            seed=seed,
        )
    cfg_kwargs: dict[str, Any] = {
        "max_reviews": max_reviews,
        "field_set": field_set,
    }
    if mode == "human":
        cfg_kwargs["negative_labels"] = negative_labels
        cfg_kwargs["human_budget_per_round"] = human_budget
    cfg_kwargs.update(overrides or {})
    cfg = SelfCalibratingConfig(mode=cast(Any, mode), seed=seed, **cfg_kwargs)

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        session = SelfCalibratingSession(df_pool, oracle, cfg)
        while not session.finished:
            if session.awaiting_human:
                session.submit_human_decisions(
                    {
                        str(it["pair_id"]): (
                            "MATCH" if truth_by_pid[str(it["pair_id"])] else "NONMATCH"
                        )
                        for it in session.gate_items
                    }
                )
            else:
                session.step()
    elapsed = time.perf_counter() - t0
    replay_stats: dict[str, float] = {}
    if isinstance(oracle, ReplayReviewer):
        n_rep, n_fb = oracle.n_replayed, oracle.n_missing
        replay_stats = {
            "replay_reviews": float(n_rep),
            "fallback_reviews": float(n_fb),
            "replay_fraction": n_rep / (n_rep + n_fb) if n_rep + n_fb else 0.0,
        }
    res = session.results()
    summary = session.summary()
    events = session.events

    pos = pd.Index(pids).get_indexer(pd.Index(res["pair_id"].to_numpy()))
    if (pos < 0).any():
        raise RuntimeError("pair_id da sessão fora do pool")
    llm_match = (res["llm_decision"] == "MATCH").to_numpy()
    explore = (res["arm"].astype(str) == "explore").to_numpy()
    s_truth, s_below, s_par2 = truth[pos], below[pos], par2[pos]
    n_run = len(res)

    # decisões humanas na ordem do portão (para as curvas por orçamento humano)
    seq_pids: list[str] = []
    seq_round: list[int] = []
    for ev in events:
        if ev.kind == "gate_opened":
            for it in ev.payload.get("items", []):
                seq_pids.append(str(it["pair_id"]))
                seq_round.append(int(ev.round))
    seq_pos = pd.Index(pids).get_indexer(pd.Index(seq_pids)) if seq_pids else None
    if seq_pos is not None and (seq_pos < 0).any():
        raise RuntimeError("item do portão fora do pool")
    n_dec = len(seq_pids)
    if mode == "human" and n_dec != int(summary["human_used"]):
        raise RuntimeError("decisões humanas divergem de summary['human_used']")
    cum_reviews = res.groupby("round").size().sort_index().cumsum()
    if seq_pos is not None:
        h_truth = truth[seq_pos]
        h_below = below[seq_pos]
        h_llm = np.array(
            [float(cum_reviews.loc[r]) for r in seq_round],
            dtype=float,
        )
    else:
        h_truth = np.zeros(0, dtype=bool)
        h_below = np.zeros(0, dtype=bool)
        h_llm = np.zeros(0, dtype=float)

    source = res["human_decision_source"].astype(str)
    unverified = (llm_match & (source != "human")).astype(bool)
    verified = (res["human_decision"] == "MATCH").to_numpy()
    cat: dict[str, float] = {}
    cat[f"finish_reason={summary['finish_reason']}"] = 1.0
    for rnd in summary["rounds"]:
        key = f"round_stop={rnd['stop_reason']}"
        cat[key] = cat.get(key, 0.0) + 1.0
    fs_prev: float | None = None
    for ev in events:
        if ev.kind == "recalibration":
            key = f"recal={ev.payload.get('status')}"
            cat[key] = cat.get(key, 0.0) + 1.0
        elif ev.kind == "ordering" and fs_prev is None:
            fs_info = ev.payload.get("fellegi_sunter")
            if isinstance(fs_info, dict):
                cat[f"fs_status={fs_info.get('status')}"] = 1.0
                if "prevalence" in fs_info:
                    fs_prev = float(fs_info["prevalence"])
            cat[f"p0_method={ev.payload.get('p0_method')}"] = 1.0
    found_all = llm_match & s_truth
    run_level = {
        "rounds": float(len(summary["rounds"])),
        "llm_errors": float(summary["llm_counts"]["ERROR"]),
        "llm_unsure": float(summary["llm_counts"]["UNSURE"]),
        "n_llm_match_unverified": float(unverified.sum()),
        "tp_unverified": float((unverified & s_truth).sum()),
        "fp_unverified": float((unverified & ~s_truth).sum()),
        "fp_caught": float(
            (llm_match & (res["human_decision"] == "NONMATCH").to_numpy()).sum()
        ),
        "n_over_budget": float((source == "over_budget").sum()),
        "verified_total": float(verified.sum()),
        "auto_learning_enabled": float(bool(summary["auto_learning_enabled"])),
        "explore_reviews": float(explore.sum()),
        "explore_found": float((explore & found_all).sum()),
        "explore_found_below": float((explore & found_all & s_below).sum()),
        "elapsed_s": elapsed,
    }
    if fs_prev is not None:
        run_level["fs_prevalence_round1"] = fs_prev
    run_level.update(replay_stats)

    is_human = mode == "human"
    base = {
        "pool": pool,
        "mode": mode,
        "reviewer": scenario,
        "seed": seed,
        "variant": "default" if not overrides else "sensibilidade",
        "field_set": field_set,
        "negative_labels": negative_labels if is_human else "na",
        "human_budget": str(human_budget) if is_human else "na",
    }
    rows: list[dict[str, Any]] = []
    run_row: dict[str, Any] = {}
    for label, budget in _marks_for(n_run, len(pids)):
        if budget > n_run:
            continue
        row = {**base, "method": "session", "budget": label}
        row.update(metrics_from_flags(llm_match, s_truth, s_below, s_par2, budget))
        if label == "run":
            row.update(run_level)
            row.update(cat)
            full = human_metrics(h_truth, h_below, h_llm, n_dec) if n_dec else None
            if full is not None:
                row.update(full)
            else:
                row.update(
                    {"human_decisions": 0.0, "verified": 0.0, "verified_below": 0.0}
                )
            run_row = row
        rows.append(row)
    rows += _human_rows(base, "session", h_truth, h_below, h_llm)
    if not with_baselines:
        return rows

    fs_scores: dict[str, np.ndarray | None] = {}
    key = (pool, "composites")
    if fs_cache is not None and key in fs_cache:
        fs_scores["fs_comp"] = fs_cache[key]
    else:
        scores = fs_posterior(df_pool, "composites")
        if fs_cache is not None:
            fs_cache[key] = scores
        fs_scores["fs_comp"] = scores
    orders = baseline_orders(df_pool, nota, seed, fs_scores)
    replay_before = (
        (oracle.n_replayed, oracle.n_missing)
        if isinstance(oracle, ReplayReviewer)
        else None
    )
    cap = max([n_run, *[m for m in MARKS if m <= len(pids)]])
    for method, order in orders.items():
        top = order[:cap]
        verdict = np.array([oracle.verdict(str(pids[i])) for i in top])
        b_match = verdict == "MATCH"
        for label, budget in _marks_for(n_run, len(pids)):
            row = {**base, "method": method, "budget": label}
            row.update(
                metrics_from_flags(b_match, truth[top], below[top], par2[top], budget)
            )
            rows.append(row)
    if with_effort and is_human:
        effort_base = {
            **base,
            "variant": "default",
            "negative_labels": "na",
            "human_budget": "na",
        }
        for method in EFFORT_BASELINES:
            rows += effort_rows_baseline(
                method, effort_base, orders["nota"], pids, truth, below, oracle
            )
    if replay_before is not None and isinstance(oracle, ReplayReviewer):
        d_rep = oracle.n_replayed - replay_before[0]
        d_fb = oracle.n_missing - replay_before[1]
        # fração das revisões das bases (nota, FS, aleatória, H1) que vieram do fixture
        run_row["replay_fraction_bases"] = (
            d_rep / (d_rep + d_fb) if d_rep + d_fb else 0.0
        )
    return rows


# ---------------------------------------------------------------------------
# Grade e agregação
# ---------------------------------------------------------------------------


class Job(NamedTuple):
    """Uma execução da grade."""

    pool: str
    mode: str
    scenario: str
    seed: int
    field_set: str
    negative_labels: str
    human_budget: int | None
    overrides: dict[str, Any] | None
    bases: bool
    effort: bool


def build_jobs(
    *,
    seeds: Sequence[int],
    scenarios: Sequence[str],
    pools: Sequence[str] = POOLS,
    field_sets: Sequence[str] = (DEFAULT_FIELD_SET,),
    negative_labels: Sequence[str] = NEGATIVE_LABELS,
    human_budgets: Sequence[int] = HUMAN_BUDGETS,
    sensitivity: bool = True,
    auto_block: bool = True,
) -> list[Job]:
    """Lista de execuções da grade.

    Humano: pool × revisor × ``negative_labels`` × ``human_budget_per_round`` ×
    sementes (× conjuntos de campos). As bases só são calculadas para o primeiro
    conjunto de campos e para a configuração padrão (``llm_nonmatch``, 50 por
    rodada); H0/H1 saem uma vez por (pool, revisor, semente). O bloco automático
    (pool × revisor × sementes) é só de continuidade; a sensibilidade usa o
    humano padrão com ``tau`` e ``window`` alterados.
    """
    jobs: list[Job] = []
    first = field_sets[0]
    for fset in field_sets:
        for pool in pools:
            for scenario in scenarios:
                for neg in negative_labels:
                    for hb in human_budgets:
                        for seed in seeds:
                            default = (
                                fset == first
                                and neg == DEFAULT_NEGATIVE
                                and hb == DEFAULT_HUMAN_BUDGET
                            )
                            jobs.append(
                                Job(
                                    pool,
                                    "human",
                                    scenario,
                                    seed,
                                    fset,
                                    neg,
                                    hb,
                                    None,
                                    default,
                                    default,
                                )
                            )
    if auto_block:
        for pool in pools:
            for scenario in scenarios:
                for seed in seeds:
                    jobs.append(
                        Job(
                            pool,
                            "auto",
                            scenario,
                            seed,
                            first,
                            DEFAULT_NEGATIVE,
                            None,
                            None,
                            True,
                            False,
                        )
                    )
    if sensitivity and "thesis_bands" in scenarios and "below_cut" in pools:
        for seed in seeds:
            jobs.append(
                Job(
                    "below_cut",
                    "human",
                    "thesis_bands",
                    seed,
                    first,
                    DEFAULT_NEGATIVE,
                    DEFAULT_HUMAN_BUDGET,
                    SENSITIVITY,
                    True,
                    False,
                )
            )
    return jobs


def run_grid(
    prep: Prepared,
    *,
    seeds: Sequence[int],
    scenarios: Sequence[str],
    max_reviews: int,
    sensitivity: bool = True,
    auto_block: bool = True,
    field_sets: Sequence[str] = (DEFAULT_FIELD_SET,),
    pools: Sequence[str] = POOLS,
    negative_labels: Sequence[str] = NEGATIVE_LABELS,
    human_budgets: Sequence[int] = HUMAN_BUDGETS,
    rates_csv: Path | None = None,
    inputs: RealInputs | None = None,
    log: Any = None,
) -> pd.DataFrame:
    jobs = build_jobs(
        seeds=seeds,
        scenarios=scenarios,
        pools=pools,
        field_sets=field_sets,
        negative_labels=negative_labels,
        human_budgets=human_budgets,
        sensitivity=sensitivity,
        auto_block=auto_block,
    )
    fs_cache: dict[tuple[str, str], np.ndarray | None] = {}
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for i, job in enumerate(jobs, 1):
        t1 = time.perf_counter()
        rows += run_one(
            prep,
            job.pool,
            job.mode,
            job.scenario,
            job.seed,
            field_set=job.field_set,
            negative_labels=job.negative_labels,
            human_budget=job.human_budget,
            overrides=job.overrides,
            fs_cache=fs_cache,
            max_reviews=max_reviews,
            with_baselines=job.bases,
            with_effort=job.effort,
            rates_csv=rates_csv,
            inputs=inputs,
        )
        if log is not None:
            tag = f"{job.negative_labels}/hb={job.human_budget}"
            log(
                f"[{i}/{len(jobs)}] {job.pool}/{job.mode}/{job.scenario}/"
                f"{job.field_set}/{tag}/seed={job.seed}"
                f"{'/sens' if job.overrides else ''}: "
                f"{time.perf_counter() - t1:.1f}s "
                f"(total {time.perf_counter() - t0:.0f}s)"
            )
    runs = pd.DataFrame(rows)
    cat_cols = [c for c in runs.columns if "=" in c]
    sess = runs["method"] == "session"
    is_run = runs["budget"] == "run"
    runs.loc[sess & is_run, cat_cols] = runs.loc[sess & is_run, cat_cols].fillna(0.0)
    return runs


KEYS = [
    "pool",
    "mode",
    "reviewer",
    "variant",
    "field_set",
    "negative_labels",
    "human_budget",
    "method",
    "budget",
]


def aggregate(runs: pd.DataFrame) -> pd.DataFrame:
    """Formato longo: média, mínimo, máximo e n de sementes válidas por métrica."""
    metrics = [c for c in runs.columns if c not in (*KEYS, "seed")]
    long = runs.melt(id_vars=[*KEYS, "seed"], value_vars=metrics, var_name="metric")
    long = long.dropna(subset=["value"])
    g = long.groupby([*KEYS, "metric"], sort=True)["value"]
    out = g.agg(mean="mean", min="min", max="max", n_valid="count").reset_index()
    # n de sementes da configuração (não da marca): uma marca que só algumas
    # sementes alcançaram deve aparecer como parcial no relatório
    cfg_keys = [k for k in KEYS if k != "budget"]
    n_seeds = runs.groupby(cfg_keys)["seed"].nunique().rename("n_seeds").reset_index()
    return out.merge(n_seeds, on=cfg_keys, how="left")


# ---------------------------------------------------------------------------
# Relatório
# ---------------------------------------------------------------------------


def _num(x: float, digits: int = 1) -> str:
    if x != x:
        return "—"
    s = f"{x:.{digits}f}"
    return s.replace(".", ",")


def _cell(agg: pd.DataFrame, metric: str, digits: int = 1) -> str:
    r = agg[agg["metric"] == metric]
    if r.empty:
        return "—"
    r0 = r.iloc[0]
    partial = f" [n={int(r0['n_valid'])}]" if r0["n_valid"] < r0["n_seeds"] else ""
    if r0["min"] == r0["max"]:
        return _num(r0["mean"], digits) + partial
    spread = f"{_num(r0['min'], digits)}–{_num(r0['max'], digits)}"
    return f"{_num(r0['mean'], digits)} ({spread}){partial}"


def _cell_k(agg: pd.DataFrame, metric: str) -> str:
    """Média (mín–máx) [fração de sementes que atingiu] de uma métrica "até k"."""
    m = agg[agg["metric"] == metric]
    if m.empty:
        return "— (0)"
    frac = float(m.iloc[0]["n_valid"]) / float(m.iloc[0]["n_seeds"])
    spread = f"{_num(m.iloc[0]['min'], 0)}–{_num(m.iloc[0]['max'], 0)}"
    return f"{_num(m.iloc[0]['mean'], 0)} ({spread}) [{_num(frac, 2)}]"


def _mean(agg: pd.DataFrame, metric: str) -> float:
    r = agg[agg["metric"] == metric]
    return float(r.iloc[0]["mean"]) if not r.empty else float("nan")


def _sel(agg: pd.DataFrame, **kw: str) -> pd.DataFrame:
    out = agg
    for k, v in kw.items():
        out = out[out[k] == v]
    return out


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


_METHOD_LABEL = {
    "session": "fila auto-calibrável",
    "nota": "(a) nota decrescente",
    "fs_comp": "(b1) FS sobre compostos",
    "random": "(c) aleatória",
    "H0": "H0 humano sozinho (ordem da nota)",
    "H1": "H1 nota + LLM tria + humano confirma MATCH",
    "H2": "H2 fila com portão revisto",
}
_BLOCKS = {
    "resposta": ("<!-- RESPOSTA:INICIO -->", "<!-- RESPOSTA:FIM -->"),
    "discussao": ("<!-- DISCUSSAO:INICIO -->", "<!-- DISCUSSAO:FIM -->"),
}
_PLACEHOLDER = "(texto ainda não escrito)"

# pool, modo, revisor, variante, campos, negative_labels, human_budget
Config = tuple[str, str, str, str, str, str, str]


def _hb_key(hb: str) -> int:
    return int(hb) if hb.isdigit() else -1


def _configs(agg: pd.DataFrame) -> list[Config]:
    cols = [
        "pool",
        "mode",
        "reviewer",
        "variant",
        "field_set",
        "negative_labels",
        "human_budget",
    ]
    frame = agg[agg["method"] == "session"][cols].drop_duplicates()
    out = [tuple(str(v) for v in row) for row in frame.itertuples(index=False)]
    out = [(a, b, c, d, e, f, g) for a, b, c, d, e, f, g in out]
    return sorted(
        out,
        key=lambda c: (c[3], c[1] != "human", c[0], c[2], c[5], _hb_key(c[6])),
    )


def _is_default(c: Config) -> bool:
    return c[1] == "auto" or (
        c[5] == DEFAULT_NEGATIVE and c[6] == str(DEFAULT_HUMAN_BUDGET)
    )


def _name(c: Config, *, full: bool = False) -> str:
    pool, mode, scen, var, _, neg, hb = c
    out = f"{pool} / {mode} / {scen}"
    if var == "sensibilidade":
        out += " [tau=0,01, window=300]"
    if full and mode == "human":
        out += f" / {neg} / {hb}"
    return out


def _sess(
    agg: pd.DataFrame, c: Config, method: str = "session", budget: str = "run"
) -> pd.DataFrame:
    pool, mode, scen, var, fset, neg, hb = c
    return _sel(
        agg,
        pool=pool,
        mode=mode,
        reviewer=scen,
        variant=var,
        field_set=fset,
        negative_labels=neg,
        human_budget=hb,
        method=method,
        budget=budget,
    )


def _effort(agg: pd.DataFrame, c: Config, method: str, budget: str) -> pd.DataFrame:
    """Linhas de H0/H1 (uma por pool/revisor) ou de H2 (a fila da config)."""
    if method in ("session", "H2"):
        return _sess(agg, c, "session", budget)
    return _sel(
        agg,
        pool=c[0],
        mode="human",
        reviewer=c[2],
        variant="default",
        field_set=c[4],
        negative_labels="na",
        human_budget="na",
        method=method,
        budget=budget,
    )


def _old(v: pd.DataFrame | None, c: Config, budget: str = "run", **extra: str):
    """Linhas ``session`` da v1/v2 da configuração equivalente (ou vazio)."""
    if v is None:
        return pd.DataFrame(columns=["metric"])
    out = _sel(
        v,
        pool=c[0],
        mode=c[1],
        reviewer=c[2],
        variant=c[3],
        method="session",
        budget=budget,
    )
    for k, val in extra.items():
        if k in out.columns:
            out = out[out[k] == val]
    return out


def _cats(a: pd.DataFrame, prefix: str, *, digits: int = 2) -> list[str]:
    return [
        f"{m.split('=', 1)[1]}: {_num(float(v), digits)}"
        for m, v in zip(a["metric"], a["mean"], strict=True)
        if m.startswith(prefix) and v > 0
    ]


def _keep_block(previous: str | None, name: str) -> str:
    start, end = _BLOCKS[name]
    if previous and start in previous and end in previous:
        return previous.split(start, 1)[1].split(end, 1)[0].strip()
    return _PLACEHOLDER


def _fix_v1_seeds(v1: pd.DataFrame | None) -> pd.DataFrame | None:
    """A v1 gravou ``n_seeds`` por marca; o n da configuração é o máximo."""
    if v1 is None:
        return None
    grp = [
        k
        for k in ("pool", "mode", "reviewer", "variant", "field_set", "method")
        if k in v1.columns
    ]
    return v1.assign(n_seeds=v1.groupby(grp)["n_seeds"].transform("max"))


def _fraction(x: float) -> str:
    return _num(100 * x, 1) + "%" if x == x else "—"


def _real_cfg(
    pool: str, rev: str, neg: str, fset: str, hb: int | None = None
) -> Config:
    hb_val = DEFAULT_HUMAN_BUDGET if hb is None else hb
    return (pool, "human", rev, "default", fset, neg, str(hb_val))


def _real_section(agg: pd.DataFrame, meta: Mapping[str, Any], fset: str) -> list[str]:
    """Seção "Revisor calibrado no Haiku real" (vazia se não houver esses runs)."""
    present = {c[2] for c in _configs(agg) if c[4] == fset}
    real = [r for r in REAL_REVIEWERS if r in present]
    if not real:
        return []
    revs = ["thesis_bands", *real]
    out: list[str] = []
    w = out.append
    w("## Revisor calibrado no Haiku real\n")
    fx = meta.get("fixture") or {}
    origin = fx.get("origin", {})
    w(
        "Esta seção usa decisões **reais** do Claude Haiku 5.5 (prompt v4, nota "
        "oculta), gravadas antes em um fixture local e **reaplicadas sem nenhuma "
        "chamada à API**. O fixture e o `.rates.csv` ficam fora do git. "
        + (
            f"Fixture: {fx.get('n_pairs')} pares, sha256 `{fx.get('sha256')}`; "
            "origem (precedência calibração > teste de reordenação > filas e "
            f"bases): {origin.get('calibration_sample', {}).get('n_pairs')} da "
            "amostra de calibração, "
            f"{origin.get('rerank_test', {}).get('n_pairs')} do teste de "
            "reordenação (pares abaixo do corte em ordem decrescente de nota) e "
            f"{origin.get('queue_runs_and_baselines', {}).get('n_pairs')} de "
            "filas e bases. "
            if fx
            else ""
        )
        + "Os três revisores novos:"
    )
    w(
        "- `measured`: taxas por faixa do `.rates.csv` **como estão** (proporções "
        "brutas, sem suavização), com `disagreement_rate = 0,10` (suposição, como "
        "em `thesis_bands`)."
    )
    w(
        "- `measured_smoothed`: as mesmas contagens, de "
        "`docs/llm_real_haiku_sem_nota.csv` (linhas `lado = todos`), com "
        "**Jeffreys** (a + 0,5) / (n + 1): fp = (FP + 0,5) / (não pares + 1), "
        "fn = (FN + 0,5) / (pares + 1), unsure = (UNSURE + 0,5) / (decisões + 1)."
    )
    w(
        "- `fixture_replay`: o par presente no fixture volta com a decisão real "
        "gravada (`FixtureReviewer`); o par ausente usa o fallback por taxas "
        "`measured_smoothed`."
    )
    w(
        "**Atenção ao replay.** No `fixture_replay` o veredito de um par presente "
        "no fixture é **fixo entre sementes**: as sementes só mudam a "
        "exploração, o desempate, o sorteio da auditoria e o fallback dos pares "
        "ausentes. A variação entre sementes é, portanto, menor que nos "
        "revisores por taxas, e os intervalos mínimo–máximo subestimam a "
        "incerteza do revisor real. As taxas medidas vêm de amostras pequenas "
        "(`low` tem 3 pares verdadeiros, `high` tem 2 não pares): tratar essas "
        "duas faixas como não medidas.\n"
    )

    # taxas
    r_meas = meta.get("rates_measured") or {}
    r_smooth = meta.get("rates_smoothed") or {}
    r_thesis = meta["rates"]
    rows = []
    for band in BAND_NAMES:
        row = [band]
        for src in (r_thesis, r_meas, r_smooth):
            row += (
                [_num(100 * v, 1) + "%" for v in src[band]]
                if band in src
                else ["—"] * 3
            )
        rows.append(row)
    w("**Taxas por faixa dos três revisores** (FP, FN, UNSURE)\n")
    w(
        _table(
            [
                "faixa",
                "thesis_bands FP",
                "FN",
                "UNSURE",
                "measured FP",
                "FN",
                "UNSURE",
                "measured_smoothed FP",
                "FN",
                "UNSURE",
            ],
            rows,
        )
    )
    w("")

    # cobertura do fixture
    w("### Fração das revisões vinda do fixture\n")
    w(
        "Revisões da fila (`fixture_replay`) servidas pelo fixture (decisão real) e "
        "pelo fallback por taxas; e fração das revisões das bases (nota, FS, "
        "aleatória e H1) servidas pelo fixture.\n"
    )
    rows = []
    for pool in POOLS:
        for neg in NEGATIVE_LABELS:
            c = _real_cfg(pool, "fixture_replay", neg, fset)
            a = _sess(agg, c)
            if a.empty:
                continue
            rows.append(
                [
                    f"{pool} / {neg}",
                    _cell(a, "reviews"),
                    _cell(a, "replay_reviews"),
                    _cell(a, "fallback_reviews"),
                    _fraction(_mean(a, "replay_fraction")),
                    _fraction(_mean(a, "replay_fraction_bases")),
                ]
            )
    w(
        _table(
            [
                "pool / negative_labels",
                "revisões da fila",
                "do fixture",
                "do fallback",
                "fração do fixture (fila)",
                "fração do fixture (bases)",
            ],
            rows,
        )
    )
    w("")

    # esforço humano
    w("### Esforço humano a orçamento B (decisões humanas)\n")
    w(
        "Configuração padrão da fila (`llm_nonmatch`, 50 por rodada), humano "
        "perfeito; H0 não depende do revisor. Comparação com `thesis_bands`. "
        "`[n=k]` = viés de sobrevivência; `—` = não chegou a B.\n"
    )
    for metric, title, digits in (
        ("verified_below", "Pares abaixo do corte verificados (de 36)", 1),
        ("verified", "Pares verdadeiros verificados", 1),
        ("dec_per_verified", "Decisões humanas por par verificado", 1),
        ("llm_reviews", "Revisões de LLM consumidas", 0),
    ):
        w(f"**{title}**\n")
        rows = []
        for pool in POOLS:
            c0 = _real_cfg(pool, "thesis_bands", DEFAULT_NEGATIVE, fset)
            r0 = [f"{pool}", _METHOD_LABEL["H0"]]
            for b in HUMAN_MARKS:
                r0.append(_cell(_effort(agg, c0, "H0", f"B{b}"), metric, digits))
            r0.append(_cell(_effort(agg, c0, "H0", "run"), metric, digits))
            rows.append(r0)
            for rev in revs:
                c = _real_cfg(pool, rev, DEFAULT_NEGATIVE, fset)
                for method in ("H1", "H2"):
                    r = [f"{pool} / {rev}", _METHOD_LABEL[method]]
                    for b in HUMAN_MARKS:
                        r.append(
                            _cell(_effort(agg, c, method, f"B{b}"), metric, digits)
                        )
                    r.append(_cell(_effort(agg, c, method, "run"), metric, digits))
                    rows.append(r)
        header = ["pool / revisor", "método", *[f"B={b}" for b in HUMAN_MARKS], "run"]
        w(_table(header, rows))
        w("")
    w("**Decisões humanas até verificar k pares abaixo do corte**\n")
    rows = []
    for pool in POOLS:
        c0 = _real_cfg(pool, "thesis_bands", DEFAULT_NEGATIVE, fset)
        rows.append(
            [pool, _METHOD_LABEL["H0"]]
            + [_cell_k(_effort(agg, c0, "H0", "run"), f"hdec_to_k{k}") for k in HUMAN_K]
        )
        for rev in revs:
            c = _real_cfg(pool, rev, DEFAULT_NEGATIVE, fset)
            for method in ("H1", "H2"):
                rows.append(
                    [f"{pool} / {rev}", _METHOD_LABEL[method]]
                    + [
                        _cell_k(_effort(agg, c, method, "run"), f"hdec_to_k{k}")
                        for k in HUMAN_K
                    ]
                )
    w(_table(["pool / revisor", "método", *[f"k={k}" for k in HUMAN_K]], rows))
    w("")

    # negative_labels
    w("### Efeito de `negative_labels` (fila, 50 por rodada)\n")
    rows = []
    for pool in POOLS:
        for rev in revs:
            for neg in NEGATIVE_LABELS:
                a = _sess(agg, _real_cfg(pool, rev, neg, fset))
                if a.empty:
                    continue
                rows.append(
                    [
                        f"{pool} / {rev} / {neg}",
                        _cell(a, "reviews"),
                        _cell(a, "human_decisions"),
                        _cell(a, "verified"),
                        _cell(a, "verified_below"),
                        _cell(a, "dec_per_verified"),
                        _cell(a, "fp_caught"),
                        _cell(a, "n_over_budget"),
                        _cell(a, "fp_unverified"),
                    ]
                )
    w(
        _table(
            [
                "configuração",
                "revisões LLM",
                "decisões humanas",
                "verificados",
                "verificados abaixo do corte",
                "decisões por verificado",
                "FP detectados",
                "over_budget",
                "FP não verificados",
            ],
            rows,
        )
    )
    w("")

    # orçamento de LLM igual
    w("### Fila × bases a orçamento de LLM igual (abaixo do corte, `run`)\n")
    rows = []
    for pool in POOLS:
        for rev in revs:
            c = _real_cfg(pool, rev, DEFAULT_NEGATIVE, fset)
            a = _sess(agg, c)
            if a.empty:
                continue
            rows.append(
                [f"{pool} / {rev}", _cell(a, "reviews", 0)]
                + [
                    _cell(_sess(agg, c, m), "found_below")
                    for m in ("session", "nota", "fs_comp", "random")
                ]
                + [
                    _cell(_sess(agg, c, "session"), "fp_llm_match"),
                    _cell(_sess(agg, c, "nota"), "fp_llm_match"),
                ]
            )
    w(
        _table(
            [
                "pool / revisor",
                "revisões",
                "fila",
                "(a) nota",
                "(b1) FS compostos",
                "(c) aleatória",
                "FP da fila",
                "FP da nota",
            ],
            rows,
        )
    )
    w("")
    return out


def render_markdown(
    agg: pd.DataFrame,
    *,
    meta: Mapping[str, Any],
    v1: pd.DataFrame | None = None,
    v2: pd.DataFrame | None = None,
    previous: str | None = None,
) -> str:
    """Relatório em português; todos os números vêm de ``agg``, ``v1``, ``v2``."""
    v1 = _fix_v1_seeds(v1)
    v2 = _fix_v1_seeds(v2)
    if v2 is not None and "field_set" in v2.columns:
        v2_main = v2[v2["field_set"] == "cdr_subscores"]
        v2_comp = v2[v2["field_set"] == "composites"]
    else:
        v2_main, v2_comp = v2, None
    default_fs = str(meta["default_field_set"])
    rates_tbl = [
        [name, *[_num(100 * v, 1) + "%" for v in rate]]
        for name, rate in meta["rates"].items()
    ]
    cfgs = [
        c for c in _configs(agg) if c[4] == default_fs and c[2] not in REAL_REVIEWERS
    ]
    dflt = [c for c in cfgs if _is_default(c)]
    human_grid = [c for c in cfgs if c[1] == "human" and c[3] == "default"]
    human_dflt = [c for c in human_grid if _is_default(c)]
    auto = [c for c in cfgs if c[1] == "auto"]
    low_fp = float(meta["rates"]["low"][0])

    out: list[str] = []
    w = out.append
    w("# Simulação offline da fila auto-calibrável sobre o extrato da tese\n")
    w(
        "> **Aviso.** Simulação com revisor **sintético**, calibrado nas taxas de "
        "erro por faixa da vigilância, medidas com a **nota visível** ao LLM; a "
        "variante simulada aqui **oculta a nota**. **Não é uma avaliação com LLM "
        "real.** O método é **experimental**, não existe na tese nem na 0.6.1, e "
        "estes números não o validam: mostram apenas o que a mecânica da fila faz "
        "sob as suposições declaradas.\n"
    )
    w(
        "Esta é a **versão 3** (portão humano com decisão explícita para todo "
        "item, orçamento humano por rodada, `negative_labels`, campos "
        "`composites`). A versão 2 está em "
        "[sim_fila_autocalibravel_v2.md](sim_fila_autocalibravel_v2.md) e a "
        "versão 1 em [sim_fila_autocalibravel_v1.md](sim_fila_autocalibravel_v1.md), "
        "com os agregados em `sim_fila_autocalibravel_resultados_v2.csv` e "
        "`..._v1.csv`.\n"
    )
    w("## Método e suposições\n")
    w(
        "- Dados: extrato `COMPARADORSEMIDENT.csv`, lido no lugar com "
        "`load_comparador_csv(..., LoadConfig(require_par=False))`. `PAR` fica numa "
        "série separada, usada só como verdade (`PAR ∈ {1, 2}`) do revisor e do "
        "humano simulados; a sessão recebe o DataFrame do extrato **sem `PAR` e "
        "sem `TARGET`**."
    )
    acc, below, p2 = meta["partition"]
    w(
        f"- Partição conferida com `below_cutoff(df, expected=THESIS_PARTITION)`: "
        f"{acc} acima do corte, {below} abaixo, {p2} com `PAR = 2` abaixo "
        f"(igual à da tese: {'sim' if meta['partition_ok'] else 'não'})."
    )
    w(
        "- Fila: a cada rodada os pares não revisados são ordenados por `p_cur` e "
        "divididos por posto em `band_rank_edges=(0,01; 0,05; 0,20)`; cada faixa é "
        "consumida do topo para baixo; um braço de exploração aleatório, fora do "
        "bandit, ocupa uma a cada `explore_every=10` revisões e não entra na "
        "parada nem na recompensa. Prior (Fellegi-Sunter por EM, sem rótulos) e "
        f"recalibrador sobre `field_set={default_fs}` (5 compostos)."
    )
    w(
        "- Portão humano (v3): o humano simulado é **perfeito** e decide **todos** "
        "os itens do portão pelo `PAR` (`MATCH` ou `NONMATCH`; não há "
        "confirmação implícita). Itens, por prioridade: MATCH do LLM (por "
        "`p_cur`), UNSURE e uma auditoria de `nonmatch_audit_n=10` NONMATCH do "
        "topo (5% do posto). O orçamento `human_budget_per_round` limita as "
        "decisões por rodada; o excedente é `over_budget`, sem rótulo. "
        "`llm_match_unverified` = MATCH do LLM sem decisão humana. "
        "`negative_labels` define os rótulos negativos do recalibrador: "
        "`llm_nonmatch` (NONMATCH do LLM vale 0) ou `verified_only` (só "
        "negativos verificados pelo humano)."
    )
    w(
        "- Revisor: `BandedOracle` roteia cada par, pela nota, para um "
        "`OracleReviewer` da faixa do perfil v3.1 (`low` [0, 5), `grey_low` [5, 6), "
        "`grey_mid` [6, 7), `grey_high` [7, 8), `near_high` [8, 9), `high` [9, ∞)). "
        "O `OracleReviewer` é determinístico por semente e identificador do par: "
        "o mesmo par recebe o mesmo veredito na fila e nas bases (comparação "
        "pareada)."
    )
    w(
        "- Cenários: `perfect` (taxas 0, sem desacordo) e `thesis_bands` (taxas "
        "abaixo, com `disagreement_rate = 0,10`, **suposição**: só afeta o "
        "protocolo, não o veredito)."
        + (
            " Terceiro revisor `rates_csv`: taxas por faixa lidas de "
            f"`{meta['rates_csv']}`."
            if meta.get("rates_csv")
            else ""
        )
    )
    w(
        "- Taxas por faixa (vigilância, contagens agregadas de "
        "`notebooks/nb_fila.py`, com Laplace: fp = (FP+1)/(não pares+2), "
        "fn = (FN+1)/(pares+2), unsure = (UNSURE+1)/(decisões+2)). A faixa `low` "
        "não tem dados e usa as taxas de `grey_low` (**suposição**):\n"
    )
    w(_table(["faixa", "FP", "FN", "UNSURE"], rates_tbl))
    w("")
    w(
        "- Configuração: `SelfCalibratingConfig()` padrão (`tau=0,05`, "
        "`delta=0,05`, `window=100`, `max_round_reviews=500`, `max_rounds=5`, "
        f"`field_set={default_fs}`), `max_reviews={meta['max_reviews']}`. Com "
        "`human_budget_per_round=50`, a fila produz no máximo 250 decisões "
        "humanas por run. Sensibilidade: humano padrão × `below_cut` × "
        "`thesis_bands` com `tau=0,01` e `window=300`."
    )
    scen_txt = ", ".join(f"`{s}`" for s in meta["scenarios"])
    neg_txt = ", ".join(f"`{s}`" for s in meta["negative_labels"])
    hb_txt = ", ".join(str(b) for b in meta["human_budgets"])
    seed_txt = ", ".join(str(s) for s in meta["seeds"])
    w(
        "- Grade: modo humano × pool ∈ {`all`, `below_cut`} × revisor ∈ "
        f"{{{scen_txt}}} × `negative_labels` ∈ {{{neg_txt}}} × "
        f"`human_budget_per_round` ∈ {{{hb_txt}}} × {len(meta['seeds'])} "
        f"sementes ({seed_txt}); mais um bloco automático (pool × revisor × "
        "sementes), só de continuidade. Valores: média (mínimo–máximo) entre "
        "sementes."
    )
    w(
        "- **Viés de sobrevivência.** `[n=k]` marca valores em que só k das "
        "sementes da configuração atingiram a marca (ou o orçamento) e entraram "
        "na média. Esses valores têm viés de sobrevivência: as sementes que "
        "chegam mais longe não são as mesmas que param antes (por exemplo, "
        "rodadas estéreis) e a média parcial não é comparável com a de uma marca "
        "alcançada por todas. “—” = nenhuma semente atingiu."
    )
    w(
        "- **Limitação conhecida.** A recompensa do bandit, a regra de parada "
        "da rodada e a detecção de anomalia usam o **MATCH bruto do LLM**, não "
        "o MATCH verificado pelo humano, mesmo no modo humano: FP do LLM contam "
        "como recompensa."
    )
    w(
        "- Linhas de base. A orçamento de LLM igual (continuidade com a v2): "
        "(a) nota decrescente; (b1) Fellegi-Sunter sobre os 5 compostos; "
        "(c) ordem aleatória, com a mesma realização do revisor, mesma semente e "
        "mesmo orçamento (revisões do run e marcas 100, 250, 500, 1000, 2000). "
        "**De esforço humano** (orçamento B ∈ {50, 100, 200, 400} decisões "
        "humanas; humano perfeito; mesma realização do revisor): **H0** o humano "
        "revisa sozinho os pares na ordem da nota (1 decisão por par); **H1** a "
        "ordem da nota, o LLM tria e o humano decide só os MATCH do LLM (1 "
        "decisão por MATCH; o LLM revisa na ordem até gerar B MATCH); **H2** é a "
        "fila com o portão revisto. “Verificado” = confirmado por decisão "
        "humana. H0 e H1 são calculados até 400 decisões; a fila, até o fim do "
        "run (no máximo 250 decisões com 50 por rodada)."
    )
    w(
        "- “Encontrado” (tabelas de LLM) = MATCH do LLM ∧ verdade. “FP” = MATCH "
        "do LLM sem verdade. “Abaixo do corte” = par com `PAR = 2` abaixo do "
        "corte da tese (36 no extrato).\n"
    )

    # ------------------------------------------------------------- esforço
    w("## Esforço humano\n")
    w(
        "Pergunta: com o humano revisando a fatia que o LLM marcou como MATCH "
        "(detectando FP) e a recalibração recebendo só rótulos verificados, a "
        "ferramenta encurta o trabalho humano? Comparação a **orçamento humano "
        "igual** B (decisões humanas), configuração padrão da fila "
        "(`llm_nonmatch`, 50 por rodada).\n"
    )
    for metric, title, digits in (
        ("verified", "Pares verdadeiros verificados", 1),
        ("verified_below", "Pares abaixo do corte verificados (de 36)", 1),
        ("dec_per_verified", "Decisões humanas por par verificado", 1),
        ("llm_reviews", "Revisões de LLM consumidas", 0),
        ("human_decisions", "Decisões humanas na sequência inteira (coluna run)", 0),
    ):
        w(f"**{title}**\n")
        rows = []
        for c in human_dflt:
            if c[3] != "default":
                continue
            for method in ("H0", "H1", "H2"):
                r = [f"{c[0]} / {c[2]}", _METHOD_LABEL[method]]
                r += [
                    _cell(_effort(agg, c, method, f"B{b}"), metric, digits)
                    for b in HUMAN_MARKS
                ]
                r.append(_cell(_effort(agg, c, method, "run"), metric, digits))
                rows.append(r)
        header = ["pool / revisor", "método", *[f"B={b}" for b in HUMAN_MARKS], "run"]
        w(_table(header, rows))
        w("")
    w(
        "`run` = a sequência inteira do método (H0/H1: até 400 decisões ou até "
        "acabarem os MATCH do LLM; fila: até o fim do run). Não é orçamento "
        "igual: serve para ver onde cada método termina. Colunas B com `—`: o "
        "método não chegou a B decisões; `[n=k]`: só k sementes chegaram (viés "
        "de sobrevivência).\n"
    )
    w("**Decisões humanas até verificar k pares abaixo do corte**\n")
    w(
        "Média (mínimo–máximo) das sementes que atingiram k, e entre colchetes a "
        "fração de sementes que atingiu. H0/H1: dentro de 400 decisões; fila: "
        "dentro do run.\n"
    )
    rows = []
    for c in human_dflt:
        for method in ("H0", "H1", "H2"):
            a = _effort(agg, c, method, "run")
            rows.append(
                [f"{c[0]} / {c[2]}", _METHOD_LABEL[method]]
                + [_cell_k(a, f"hdec_to_k{k}") for k in HUMAN_K]
            )
    w(_table(["pool / revisor", "método", *[f"k={k}" for k in HUMAN_K]], rows))
    w("")

    w("### Efeito de `negative_labels` e do orçamento por rodada (H2)\n")
    w(
        "Fila no modo humano. “Verificados” = pares confirmados por decisão "
        "humana (verdade); “FP detectados” = MATCH do LLM que o humano rejeitou; "
        "“FP não verificados” = MATCH do LLM sem verdade e sem decisão humana "
        "(`over_budget`).\n"
    )
    rows = []
    for c in human_grid:
        a = _sess(agg, c)
        rows.append(
            [
                _name(c, full=True),
                _cell(a, "reviews"),
                _cell(a, "rounds"),
                _cell(a, "human_decisions"),
                _cell(a, "verified"),
                _cell(a, "verified_below"),
                _cell(a, "dec_per_verified"),
                _cell(a, "fp_caught"),
                _cell(a, "n_over_budget"),
                _cell(a, "n_llm_match_unverified"),
                _cell(a, "fp_unverified"),
            ]
        )
    w(
        _table(
            [
                "configuração (pool / modo / revisor / negative_labels / por rodada)",
                "revisões LLM",
                "rodadas",
                "decisões humanas",
                "verificados",
                "verificados abaixo do corte (de 36)",
                "decisões por verificado",
                "FP detectados",
                "over_budget",
                "llm_match_unverified",
                "FP não verificados",
            ],
            rows,
        )
    )
    w("")
    w("**Verificados abaixo do corte por orçamento humano B (fila)**\n")
    rows = []
    for c in human_grid:
        rows.append(
            [_name(c, full=True)]
            + [
                _cell(_sess(agg, c, "session", f"B{b}"), "verified_below")
                for b in HUMAN_MARKS
            ]
        )
    w(_table(["configuração", *[f"B={b}" for b in HUMAN_MARKS]], rows))
    w("")

    w("### Resposta\n")
    w(_BLOCKS["resposta"][0])
    w(_keep_block(previous, "resposta"))
    w(_BLOCKS["resposta"][1])
    w("")

    # ------------------------------------------------------------- resultados
    w("## Resultados da fila (orçamento de LLM)\n")
    w("### Comportamento da fila (configuração padrão)\n")
    rows = []
    for c in dflt:
        a = _sess(agg, c)
        rows.append(
            [
                _name(c),
                _cell(a, "reviews"),
                _cell(a, "rounds"),
                _cell(a, "human_decisions"),
                _cell(a, "true_reviewed"),
                _cell(a, "found"),
                _cell(a, "found_below"),
                _cell(a, "fp_llm_match"),
                _cell(a, "llm_unsure"),
            ]
        )
    w(
        _table(
            [
                "configuração",
                "revisões LLM",
                "rodadas",
                "decisões humanas",
                "verdadeiros revisados",
                "encontrados (MATCH do LLM ∧ verdade)",
                "abaixo do corte (de 36)",
                "FP",
                "UNSURE",
            ],
            rows,
        )
    )
    w("")

    w("### Motivos de parada, término e recalibração\n")
    w(
        "Contagem média por run (parada de rodada, status de recalibração, método "
        "de `p0` na 1ª rodada) e fração de runs (término).\n"
    )
    rows = []
    for c in [*human_grid, *auto, *[x for x in cfgs if x[3] == "sensibilidade"]]:
        a = _sess(agg, c)
        rows.append(
            [
                _name(c, full=True),
                "; ".join(_cats(a, "finish_reason=")) or "—",
                "; ".join(_cats(a, "round_stop=")) or "—",
                "; ".join(_cats(a, "recal=")) or "—",
                "; ".join(_cats(a, "p0_method=")) or "—",
            ]
        )
    w(
        _table(
            [
                "configuração",
                "término (fração de runs)",
                "parada de rodada (média por run)",
                "recalibração (média por run)",
                "`p0` da 1ª rodada (fração de runs)",
            ],
            rows,
        )
    )
    w("")

    w("### Regra de parada por posterior\n")
    w(
        "A rodada só pára por `posterior` quando, em todos os braços ativos, "
        "P(θ > τ) < δ com τ = 0,05 e δ = 0,05, ou seja, quando a taxa de MATCH "
        "bruto do LLM em cada braço fica claramente abaixo de τ. Com "
        f"`thesis_bands`, a taxa de FP da faixa `low` é {_num(100 * low_fp, 1)}% "
        "(tabela de taxas), acima de τ: mesmo em regiões sem nenhum par "
        "verdadeiro o LLM simulado marca MATCH com frequência maior que τ, e o "
        "posterior não cai abaixo de δ. A tabela mostra as paradas por motivo "
        "(configuração padrão, média por run).\n"
    )
    rows = []
    for c in dflt:
        a = _sess(agg, c)
        rows.append(
            [
                _name(c),
                _cell(a, "rounds"),
                _num(_mean(a, "round_stop=posterior"), 2),
                _num(_mean(a, "round_stop=max_round_reviews"), 2),
                _num(_mean(a, "round_stop=max_reviews"), 2),
            ]
        )
    w(
        _table(
            [
                "configuração",
                "rodadas",
                "paradas por `posterior`",
                "paradas por `max_round_reviews`",
                "paradas por `max_reviews`",
            ],
            rows,
        )
    )
    w("")

    w("### Comparação com as bases a orçamento de LLM igual (`run`)\n")
    methods = ("session", "nota", "fs_comp", "random")
    for metric, title in (
        ("found", "Pares verdadeiros encontrados (MATCH do LLM ∧ verdade)"),
        ("found_below", "Pares abaixo do corte encontrados (de 36)"),
        ("fp_llm_match", "FP (MATCH do LLM sem verdade)"),
    ):
        w(f"**{title}** (orçamento = nº de revisões da fila no run)\n")
        rows = []
        for c in dflt:
            r = [_name(c), _cell(_sess(agg, c), "reviews", 0)]
            r += [_cell(_sess(agg, c, m), metric) for m in methods]
            rows.append(r)
        w(
            _table(
                [
                    "configuração",
                    "revisões",
                    "fila",
                    "(a) nota",
                    "(b1) FS compostos",
                    "(c) aleatória",
                ],
                rows,
            )
        )
        w("")

    w("### Curvas: encontrados por nº de revisões de LLM\n")
    w(
        "As bases não dependem do modo; aparecem uma vez por pool e revisor. "
        "Células com `[n=k]` têm viés de sobrevivência (ver o método).\n"
    )
    for metric, title in (
        ("found", "Verdadeiros encontrados"),
        ("found_below", "Abaixo do corte encontrados (de 36)"),
    ):
        w(f"**{title}**\n")
        rows = []
        for c in [x for x in dflt if x[3] == "default"]:
            for method in methods:
                if method != "session" and c[1] != "human":
                    continue
                label = _METHOD_LABEL[method]
                if method == "session":
                    label += f" ({c[1]})"
                r = [f"{c[0]} / {c[2]}", label]
                r += [_cell(_sess(agg, c, method, str(m)), metric) for m in MARKS]
                rows.append(r)
        w(_table(["pool / revisor", "método", *[str(m) for m in MARKS]], rows))
        w("")

    w("### Revisões de LLM até achar k pares abaixo do corte\n")
    rows = []
    for c in dflt:
        for method in methods:
            r = [_name(c), _METHOD_LABEL[method]]
            a = _sess(agg, c, method)
            r += [_cell_k(a, f"reviews_to_k{k}") for k in K_TARGETS]
            rows.append(r)
    w(_table(["configuração", "método", *[f"k={k}" for k in K_TARGETS]], rows))
    w("")

    w("### Ganho das rodadas seguintes à 1ª\n")
    w(
        "Pares abaixo do corte encontrados (MATCH do LLM ∧ verdade) entre as "
        "marcas de 500 e de 1.000 revisões de LLM, a fila contra a nota "
        "decrescente. A 1ª rodada (até 500 revisões) é ordenada pela nota; o "
        "que a fila acrescenta depois vem das rodadas ≥ 2. Médias; `[n=k]` = "
        "viés de sobrevivência.\n"
    )
    rows = []
    for c in [x for x in dflt if x[1] == "human" and x[3] == "default"]:
        f5 = _sess(agg, c, "session", "500")
        f10 = _sess(agg, c, "session", "1000")
        n5 = _sess(agg, c, "nota", "500")
        n10 = _sess(agg, c, "nota", "1000")
        d_f = _mean(f10, "found_below") - _mean(f5, "found_below")
        d_n = _mean(n10, "found_below") - _mean(n5, "found_below")
        rows.append(
            [
                _name(c),
                _cell(f5, "found_below"),
                _cell(f10, "found_below"),
                _num(d_f, 1),
                _cell(n5, "found_below"),
                _cell(n10, "found_below"),
                _num(d_n, 1),
            ]
        )
    w(
        _table(
            [
                "configuração",
                "fila @500",
                "fila @1000",
                "Δ fila",
                "nota @500",
                "nota @1000",
                "Δ nota",
            ],
            rows,
        )
    )
    w("")

    w("### Braço de exploração\n")
    rows = []
    for c in dflt:
        a = _sess(agg, c)
        rows.append(
            [
                _name(c),
                _cell(a, "reviews"),
                _cell(a, "explore_reviews"),
                _cell(a, "explore_found"),
                _cell(a, "explore_found_below"),
            ]
        )
    w(
        _table(
            [
                "configuração",
                "revisões LLM",
                "exploração: revisões",
                "exploração: encontrados",
                "exploração: abaixo do corte",
            ],
            rows,
        )
    )
    w("")

    w("## O que o aprendizado (A') faz no modo automático\n")
    rows = []
    for c in auto:
        a = _sess(agg, c)
        rows.append(
            [
                _name(c),
                _cell(a, "auto_learning_enabled", 2),
                "; ".join(_cats(a, "recal=")) or "—",
                _cell(a, "n_llm_match_unverified"),
            ]
        )
    w(
        _table(
            [
                "configuração",
                "aprendizado ligado (fração de runs)",
                "recalibração (média por run)",
                "`llm_match_unverified`",
            ],
            rows,
        )
    )
    w("")

    # ------------------------------------------------------------- v1 v2 v3
    w("## v1 × v2 × v3\n")
    if v1 is None and v2 is None:
        w("(CSVs das versões anteriores não encontrados; comparação omitida.)\n")
    else:
        w(
            "Mesma grade de pool × modo × revisor e mesmas sementes; v3 = "
            "configuração padrão (humano: `llm_nonmatch`, 50 por rodada). "
            "Cada célula traz **v1 → v2 → v3** da fila. A v1 usava faixas por "
            "probabilidade e compostos; a v2, faixas por posto, exploração e "
            "sub-escores do CdR; a v3 volta aos compostos e muda o portão. O "
            "nº de revisões por run difere entre versões (regras de parada e "
            "portão diferentes), então a comparação a **orçamento igual** é a "
            "tabela por marcas. Células `[n=k]`: viés de sobrevivência.\n"
        )
        cmp_cfg = [c for c in dflt if c[3] == "default"]
        for metric, title in (
            ("found", "Encontrados (MATCH do LLM ∧ verdade), por run"),
            ("found_below", "Abaixo do corte (de 36), por run"),
            ("fp_llm_match", "FP, por run"),
            ("reviews", "Revisões de LLM por run"),
        ):
            w(f"**{title}**\n")
            rows = []
            for c in cmp_cfg:
                r = [_name(c)]
                a1 = _old(v1, c)
                a2 = _old(v2_main, c, field_set="cdr_subscores")
                a3 = _sess(agg, c)
                r.append(
                    " → ".join(
                        _cell(a, metric, 1 if metric != "reviews" else 0)
                        for a in (a1, a2, a3)
                    )
                )
                rows.append(r)
            w(_table(["configuração", "v1 → v2 → v3"], rows))
            w("")
        w(
            "**Pares abaixo do corte encontrados por marca de revisões de LLM** "
            "(v1 → v2 → v3)\n"
        )
        rows = []
        for c in cmp_cfg:
            row = [_name(c)]
            for m in MARKS:
                cells = [
                    _cell(_old(v1, c, str(m)), "found_below"),
                    _cell(
                        _old(v2_main, c, str(m), field_set="cdr_subscores"),
                        "found_below",
                    ),
                    _cell(_sess(agg, c, "session", str(m)), "found_below"),
                ]
                row.append(" → ".join(cells))
            rows.append(row)
        w(_table(["configuração", *[str(m) for m in MARKS]], rows))
        w("")
        if v2_comp is not None and not v2_comp.empty:
            w(
                "**Troca de campos na v2** (ablação da v2: compostos em vez de "
                "sub-escores, humano × `thesis_bands`), abaixo do corte por run:\n"
            )
            rows = []
            for c in [x for x in cmp_cfg if x[1] == "human" and x[2] == "thesis_bands"]:
                a_c = _old(v2_comp, c, field_set="composites")
                rows.append(
                    [
                        _name(c),
                        _cell(_old(v1, c), "found_below"),
                        _cell(
                            _old(v2_main, c, field_set="cdr_subscores"), "found_below"
                        ),
                        _cell(a_c, "found_below"),
                        _cell(_sess(agg, c), "found_below"),
                    ]
                )
            w(
                _table(
                    [
                        "configuração",
                        "v1 (compostos)",
                        "v2 (sub-escores do CdR)",
                        "v2 ablação (compostos)",
                        "v3 (compostos)",
                    ],
                    rows,
                )
            )
            w("")

    out += _real_section(agg, meta, default_fs)

    w("## Discussão\n")
    w(_BLOCKS["discussao"][0])
    w(_keep_block(previous, "discussao"))
    w(_BLOCKS["discussao"][1])
    w("")
    w("## Reprodução\n")
    w("```powershell")
    for command in meta.get("commands") or [meta["command"]]:
        w(command)
    w("```\n")
    w(
        "O cenário opcional `rates_csv` (`--rates-csv`, colunas `band, fp, fn, "
        "unsure`) não foi usado. O segundo comando acima acrescenta ao CSV (sem "
        "apagar as linhas anteriores) os revisores calibrados no Haiku real; "
        "ele precisa dos insumos locais em `data/llm_fixtures/` e do relatório "
        "`docs/llm_real_haiku_sem_nota.csv`."
    )
    w("")
    w(
        "Soma dos tempos das sessões (sem as bases): "
        f"{_num(meta['elapsed_sessions'], 0)} s. "
        "Nenhum identificador de par é gravado: só agregados."
    )
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_seeds(text: str) -> list[int]:
    if "," in text:
        return [int(s) for s in text.split(",") if s.strip()]
    n = int(text)
    if n < 1:
        raise argparse.ArgumentTypeError("--seeds deve ser >= 1")
    return list(range(1, n + 1))


def _parse_scenarios(text: str) -> list[str]:
    out = [s.strip() for s in text.split(",") if s.strip()]
    bad = [s for s in out if s not in (*SCENARIOS, "rates_csv", *REAL_REVIEWERS)]
    if bad:
        raise argparse.ArgumentTypeError(f"cenários desconhecidos: {bad}")
    return out


def _parse_field_sets(text: str) -> list[str]:
    out = [s.strip() for s in text.split(",") if s.strip()]
    bad = [s for s in out if s not in FIELD_SETS]
    if bad or not out:
        raise argparse.ArgumentTypeError(f"conjuntos de campos inválidos: {bad}")
    return out


def _parse_negative(text: str) -> list[str]:
    out = [s.strip() for s in text.split(",") if s.strip()]
    bad = [s for s in out if s not in NEGATIVE_LABELS]
    if bad or not out:
        raise argparse.ArgumentTypeError(f"negative_labels inválidos: {bad}")
    return out


def _parse_budgets(text: str) -> list[int]:
    try:
        out = [int(s) for s in text.split(",") if s.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not out or any(b < 1 for b in out):
        raise argparse.ArgumentTypeError("orçamentos devem ser inteiros >= 1")
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--input", type=Path, required=True, help="COMPARADORSEMIDENT.csv")
    p.add_argument(
        "--seeds", type=_parse_seeds, default=_parse_seeds("3"), help="N ou lista a,b,c"
    )
    p.add_argument("--max-reviews", type=int, default=2000)
    p.add_argument(
        "--scenarios", type=_parse_scenarios, default=list(SCENARIOS), help="lista"
    )
    p.add_argument(
        "--rates-csv",
        type=Path,
        default=None,
        help="terceiro revisor: CSV band,fp,fn,unsure (acrescenta o cenário rates_csv)",
    )
    p.add_argument(
        "--field-sets",
        type=_parse_field_sets,
        default=[DEFAULT_FIELD_SET],
        help="conjuntos de campos da fila (padrão: composites)",
    )
    p.add_argument(
        "--negative-labels",
        type=_parse_negative,
        default=list(NEGATIVE_LABELS),
        help="negative_labels do modo humano (padrão: llm_nonmatch,verified_only)",
    )
    p.add_argument(
        "--human-budgets",
        type=_parse_budgets,
        default=list(HUMAN_BUDGETS),
        help="human_budget_per_round do modo humano (padrão: 25,50,100)",
    )
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--out-md", type=Path, required=True)
    p.add_argument(
        "--v1-csv",
        type=Path,
        default=Path("docs/sim_fila_autocalibravel_resultados_v1.csv"),
        help="agregados da v1 para a tabela v1 × v2 × v3 (omitida se não existir)",
    )
    p.add_argument(
        "--v2-csv",
        type=Path,
        default=Path("docs/sim_fila_autocalibravel_resultados_v2.csv"),
        help="agregados da v2 para a tabela v1 × v2 × v3 (omitida se não existir)",
    )
    p.add_argument(
        "--measured-rates-csv",
        type=Path,
        default=Path("data/llm_fixtures/haiku-5-5_v4_sem-nota.rates.csv"),
        help="taxas medidas por faixa (cenário measured)",
    )
    p.add_argument(
        "--counts-csv",
        type=Path,
        default=Path("docs/llm_real_haiku_sem_nota.csv"),
        help="contagens da calibração real (cenários measured_smoothed e replay)",
    )
    p.add_argument(
        "--fixture-jsonl",
        type=Path,
        default=Path("data/llm_fixtures/haiku-5-5_v4_sem-nota.jsonl"),
        help="decisões reais gravadas (cenário fixture_replay)",
    )
    p.add_argument(
        "--append-csv",
        action="store_true",
        help="acrescenta as linhas novas ao --out-csv (troca só os mesmos revisores)",
    )
    p.add_argument(
        "--no-sensitivity",
        action="store_true",
        help="não roda o bloco de sensibilidade (tau, window)",
    )
    p.add_argument(
        "--render-only",
        action="store_true",
        help="regenera o relatório a partir de --out-csv, sem rodar a grade",
    )
    p.add_argument(
        "--no-auto-block",
        action="store_true",
        help="não roda o bloco automático de continuidade",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="1 semente, max-reviews 300, config padrão, sem sensibilidade (fumaça)",
    )
    p.add_argument(
        "--skip-partition-check",
        action="store_true",
        help="não exige a partição 211/61485/36 da tese",
    )
    return p


def read_agg(path: Path) -> pd.DataFrame:
    """Lê o CSV agregado preservando ``na`` e ``human_budget`` como texto."""
    return pd.read_csv(path, keep_default_na=False, dtype={"human_budget": str})


def reproduction_commands(n_seeds: int, max_reviews: int) -> list[str]:
    """Comandos que geram o CSV: a grade v3 e o acréscimo do Haiku real."""
    head = (
        "$env:PYTHONPATH = 'src'; python notebooks/sim_fila_autocalibravel.py "
        "--input ../phd-research/data/COMPARADORSEMIDENT.csv "
        f"--seeds {n_seeds} --max-reviews {max_reviews} "
    )
    out_args = (
        "--out-csv docs/sim_fila_autocalibravel_resultados.csv "
        "--out-md docs/sim_fila_autocalibravel.md"
    )
    return [
        head
        + "--scenarios perfect,thesis_bands --field-sets composites "
        + "--negative-labels llm_nonmatch,verified_only --human-budgets 25,50,100 "
        + out_args,
        head
        + "--scenarios measured,measured_smoothed,fixture_replay "
        + "--field-sets composites --negative-labels llm_nonmatch,verified_only "
        + "--human-budgets 50 --no-auto-block --no-sensitivity --append-csv "
        + out_args,
    ]


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    seeds = args.seeds
    max_reviews = args.max_reviews
    sensitivity = not args.no_sensitivity
    negative_labels = args.negative_labels
    human_budgets = args.human_budgets
    scenarios = list(args.scenarios)
    if args.rates_csv is not None and "rates_csv" not in scenarios:
        scenarios.append("rates_csv")
    if "rates_csv" in scenarios and args.rates_csv is None:
        raise SystemExit("o cenário rates_csv exige --rates-csv")
    if args.quick:
        seeds = seeds[:1]
        max_reviews = min(max_reviews, 300)
        sensitivity = False
        negative_labels = [DEFAULT_NEGATIVE]
        human_budgets = [DEFAULT_HUMAN_BUDGET]

    def log(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    t0 = time.perf_counter()
    prep = load_prepared(args.input, check_partition=not args.skip_partition_check)
    log(f"dados: {prep.n} pares; partição {prep.partition}")
    inputs = RealInputs(
        measured_rates_csv=args.measured_rates_csv,
        counts_csv=args.counts_csv,
        fixture_jsonl=args.fixture_jsonl,
    )
    if not args.render_only:
        runs = run_grid(
            prep,
            seeds=seeds,
            scenarios=scenarios,
            max_reviews=max_reviews,
            sensitivity=sensitivity,
            auto_block=not args.no_auto_block,
            field_sets=args.field_sets,
            negative_labels=negative_labels,
            human_budgets=human_budgets,
            rates_csv=args.rates_csv,
            inputs=inputs,
            log=log,
        )
        new = aggregate(runs)
        if args.append_csv and args.out_csv.exists():
            old = read_agg(args.out_csv)
            keep = ~old["reviewer"].isin(set(new["reviewer"]))
            new = pd.concat([old[keep], new], ignore_index=True)
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        new.to_csv(args.out_csv, index=False, float_format="%.6g")
    agg = read_agg(args.out_csv)
    el = agg[
        (agg["method"] == "session")
        & (agg["budget"] == "run")
        & (agg["metric"] == "elapsed_s")
    ]
    sess_elapsed = float((el["mean"] * el["n_valid"]).sum())

    rates, _ = build_scenario_rates("thesis_bands")
    reviewers = list(dict.fromkeys(agg["reviewer"].astype(str)))
    base_scen = [r for r in reviewers if r not in REAL_REVIEWERS]
    human = agg[agg["negative_labels"] != "na"]
    meta: dict[str, Any] = {
        "rates": rates,
        "partition": prep.partition,
        "partition_ok": prep.partition == THESIS_PARTITION,
        "seeds": seeds,
        "scenarios": base_scen,
        "negative_labels": sorted(set(human["negative_labels"].astype(str))),
        "human_budgets": sorted(
            {int(x) for x in human["human_budget"].astype(str) if x.isdigit()}
        ),
        "max_reviews": max_reviews,
        "elapsed_sessions": sess_elapsed,
        "commands": reproduction_commands(len(seeds), max_reviews),
        "default_field_set": args.field_sets[0],
        "rates_csv": str(args.rates_csv) if args.rates_csv is not None else None,
    }
    if args.measured_rates_csv.exists():
        meta["rates_measured"] = load_rates_csv(args.measured_rates_csv)
    if args.counts_csv.exists():
        meta["rates_smoothed"] = smoothed_rates_from_counts(args.counts_csv)
    manifest = args.fixture_jsonl.with_name(args.fixture_jsonl.stem + ".manifest.json")
    if manifest.exists():
        meta["fixture"] = json.loads(manifest.read_text(encoding="utf-8"))
    v1 = pd.read_csv(args.v1_csv) if args.v1_csv.exists() else None
    v2 = pd.read_csv(args.v2_csv) if args.v2_csv.exists() else None
    previous = args.out_md.read_text(encoding="utf-8") if args.out_md.exists() else None
    md = render_markdown(agg, meta=meta, v1=v1, v2=v2, previous=previous)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md, encoding="utf-8", newline="\n")
    log(f"ok em {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
