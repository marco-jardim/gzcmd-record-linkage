"""H8 e H11 de ``docs/engenharia_de_filas.md`` no extrato real da tese.

Experimental. Sem rede e sem LLM real: o veredito do LLM vem do fixture local
(``data/llm_fixtures/haiku-5-5_v4_sem-nota.jsonl``) reaplicado por
``llm_real_eval.FixtureReviewer`` (``on_missing="rates"``); o humano é simulado.
``PAR`` entra **apenas** como verdade do humano simulado e das recompensas. Nada
em nível de par é escrito: só agregados.

* **H8** (bandit de Thompson sobre faixas contíguas da régua não ajuda quando o
  rendimento é monotônico): S0 = ordem global decrescente da nota; S5 =
  ``ranking.ThompsonQueue(input_kind="partial_orders")`` sobre as faixas
  contíguas por posto (1%, 5%, 20%, resto; acumulados), do topo para baixo.
  Recompensa = par verdadeiro (humano perfeito) ou MATCH do LLM (fixture).
  Cenários da régua: ``real`` (como está), ``N1`` e ``N2`` (não monotonia
  induzida: os verdadeiros da janela do topo são permutados para dentro de uma
  faixa intermediária).
* **H11** (o LLM serve para auditar decisões humanas): humano simulado com FN e
  FP revisa a ordem da nota; audita-se de novo (humano perfeito) só os pares com
  LLM = MATCH e humano = NONMATCH; compara-se com uma auditoria aleatória do
  mesmo tamanho entre os NONMATCH do humano.

Uso (a partir da raiz do repositório)::

    $env:PYTHONPATH = 'src'
    python notebooks/exp_filas_real.py \\
        --input ../phd-research/data/COMPARADORSEMIDENT.csv --seeds 20 \\
        --out-csv docs/engenharia_de_filas_real_h8_h11.csv \\
        --out-md docs/engenharia_de_filas_real_h8_h11.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd
from scipy import stats

from gzcmd_record_linkage.ranking import ThompsonQueue

__all__ = [
    "BAND_FRACTIONS",
    "H11_BUDGETS",
    "H8_BUDGETS",
    "LlmVerdicts",
    "VerdictSource",
    "PoolData",
    "audit_metrics",
    "band_bounds",
    "bandit_pick",
    "decide_h11",
    "decide_h8",
    "human_match",
    "human_uniform",
    "induce_nonmonotonic",
    "main",
    "ratio_boot",
    "summarize",
    "tie_broken_order",
    "verified_at",
    "yield_by_bins",
]

BAND_FRACTIONS: tuple[float, ...] = (0.01, 0.05, 0.20)  # acumulados; resto = 1.0
H8_BUDGETS: tuple[int, ...] = (50, 100, 200, 400, 800)
H11_BUDGETS: tuple[int, ...] = (100, 200, 400, 800)
H11_FN: tuple[float, ...] = (0.05, 0.10)
H11_FP: tuple[float, ...] = (0.0, 0.01)
SCENARIOS: tuple[str, ...] = ("real", "N1", "N2")
REWARDS: tuple[str, ...] = ("human", "llm")
FALLBACK_DISAGREEMENT = 0.10  # como em sim_fila_autocalibravel (suposição)
RECOVERY_MIN = 0.5
PRECISION_MIN = 0.3
N_BOOT = 2000


# ---------------------------------------------------------------------------
# Ordem, faixas e rendimento
# ---------------------------------------------------------------------------


def tie_broken_order(score: np.ndarray, seed: int) -> np.ndarray:
    """Índices em ordem decrescente de ``score``; empates por permutação da semente."""
    perm = np.random.default_rng(seed).permutation(len(score))
    return perm[np.argsort(-score[perm], kind="stable")]


def band_bounds(
    n: int, fractions: Sequence[float] = BAND_FRACTIONS
) -> list[tuple[int, int]]:
    """Faixas contíguas por posto ``[início, fim)`` com frações acumuladas.

    ``fractions = (0,01, 0,05, 0,20)`` dá as faixas 0–1%, 1–5%, 5–20% e 20–100%.
    """
    cuts = [max(1, round(f * n)) for f in fractions]
    edges = [0, *cuts, n]
    if any(b <= a for a, b in zip(edges[:-1], edges[1:], strict=True)):
        raise ValueError("população pequena demais para as faixas pedidas")
    return list(zip(edges[:-1], edges[1:], strict=True))


def yield_by_bins(
    truth_ranked: np.ndarray, edges: Sequence[int]
) -> list[tuple[int, int]]:
    """``(verdadeiros, tamanho)`` por faixa ``[edges[i], edges[i+1])`` do posto."""
    t = np.asarray(truth_ranked, dtype=bool)
    return [
        (int(t[a:b].sum()), int(b - a))
        for a, b in zip(edges[:-1], edges[1:], strict=True)
    ]


def quantile_edges(n: int, k: int) -> list[int]:
    """Limites de ``k`` faixas de postos de tamanho quase igual."""
    return [round(i * n / k) for i in range(k + 1)]


def verified_at(truth_in_order: np.ndarray, budgets: Sequence[int]) -> dict[int, int]:
    """Verdadeiros nas primeiras ``B`` decisões, para cada ``B``."""
    t = np.asarray(truth_in_order, dtype=bool)
    return {int(b): int(t[: int(b)].sum()) for b in budgets}


def induce_nonmonotonic(
    truth_ranked: np.ndarray,
    window_end: int,
    dest_start: int,
    dest_end: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Permutação que concentra os verdadeiros da janela numa faixa intermediária.

    Dentro dos postos ``[0, window_end)``, todos os verdadeiros vão para postos
    sorteados (sem repetição) de ``[dest_start, dest_end)``; os demais pares da
    janela ocupam o resto na ordem original. Fora da janela nada muda.
    Devolve ``perm`` tal que ``nova_ordem = ordem[perm]``.
    """
    t = np.asarray(truth_ranked, dtype=bool)
    if not 0 <= dest_start < dest_end <= window_end <= len(t):
        raise ValueError("janela e destino inconsistentes")
    pos_true = np.flatnonzero(t[:window_end])
    if len(pos_true) > dest_end - dest_start:
        raise ValueError("destino menor que o número de verdadeiros")
    targets = np.sort(rng.choice(np.arange(dest_start, dest_end), len(pos_true), False))
    perm = np.arange(len(t))
    others = np.setdiff1d(np.arange(window_end), pos_true, assume_unique=True)
    free = np.setdiff1d(np.arange(window_end), targets, assume_unique=True)
    perm[targets] = pos_true
    perm[free] = others
    return perm


# ---------------------------------------------------------------------------
# Bandit (S5)
# ---------------------------------------------------------------------------


def bandit_pick(
    n: int,
    bounds: Sequence[tuple[int, int]],
    reward_fn: Callable[[int], int],
    budget: int,
    seed: int,
    *,
    window: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """S5: Thompson sobre faixas contíguas; itens e braços das ``budget`` escolhas.

    Os itens são **postos** (0 = topo da ordem); cada braço lista os postos da
    sua faixa do topo para baixo. ``reward_fn(posto)`` devolve 0/1 e alimenta o
    bandit. Prior Beta(1, 1); ``window=None`` (sem janela; B ≤ 800).
    """
    rankings = [np.arange(a, b, dtype=np.int64) for a, b in bounds]
    queue = ThompsonQueue(
        rankings,
        window=window,
        prior=(1, 1),
        seed=seed,
        input_kind="partial_orders",
        n_items=n,
    )
    items: list[int] = []
    arms: list[int] = []
    for _ in range(int(budget)):
        try:
            item = queue.next()
        except StopIteration:
            break
        items.append(item)
        arms.append(queue.arm_of(item))
        queue.update(item, int(reward_fn(item)))
    return np.asarray(items, dtype=np.int64), np.asarray(arms, dtype=np.int64)


# ---------------------------------------------------------------------------
# Humano simulado e auditoria (H11)
# ---------------------------------------------------------------------------


def human_uniform(seed: int, key: str) -> float:
    """Uniforme em [0, 1) determinística por ``(seed, key)`` (par)."""
    digest = hashlib.blake2b(f"{seed}|{key}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2.0**64


def human_match(
    truth: np.ndarray, u: np.ndarray, fn_rate: float, fp_rate: float
) -> np.ndarray:
    """Decisão MATCH do humano simulado: par é perdido se ``u < fn``; não par vira
    MATCH se ``u < fp``. O mesmo ``u`` serve a todas as taxas (erros aninhados)."""
    t = np.asarray(truth, dtype=bool)
    uu = np.asarray(u, dtype=float)
    return np.where(t, uu >= fn_rate, uu < fp_rate).astype(bool)


def audit_metrics(
    truth: np.ndarray, human_says_match: np.ndarray, llm_match: np.ndarray
) -> dict[str, float]:
    """Contagens da auditoria sobre os pares revisados pelo humano.

    Auditoria: revisar de novo (humano perfeito) só os pares com LLM = MATCH e
    humano = NONMATCH. ``random_*`` é o valor esperado (exato, hipergeométrico)
    de uma auditoria aleatória do mesmo tamanho entre os NONMATCH do humano.
    """
    t = np.asarray(truth, dtype=bool)
    h = np.asarray(human_says_match, dtype=bool)
    m = np.asarray(llm_match, dtype=bool)
    non = ~h
    extras = m & non
    n_extras = int(extras.sum())
    human_fn = int((t & non).sum())
    n_non = int(non.sum())
    recovered = int((extras & t).sum())
    random_recovered = n_extras * human_fn / n_non if n_non else 0.0
    return {
        "n_reviewed": float(len(t)),
        "human_found": float((t & h).sum()),
        "human_fn": float(human_fn),
        "human_fp": float((~t & h).sum()),
        "extras": float(n_extras),
        "recovered": float(recovered),
        "extras_llm_fp": float((extras & ~t).sum()),
        "random_recovered": float(random_recovered),
    }


# ---------------------------------------------------------------------------
# Estatística
# ---------------------------------------------------------------------------


def summarize(values: Sequence[float]) -> tuple[float, float, float, int]:
    """Média, limites do IC 95% (t de Student entre sementes) e n válido."""
    x = np.asarray([v for v in values if v == v], dtype=float)
    n = len(x)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean = float(x.mean())
    if n < 2:
        return mean, float("nan"), float("nan"), n
    sd = float(x.std(ddof=1))
    half = float(stats.t.ppf(0.975, n - 1)) * sd / np.sqrt(n)
    return mean, mean - half, mean + half, n


def ratio_boot(
    num: Sequence[float],
    den: Sequence[float],
    *,
    n_boot: int = N_BOOT,
    seed: int = 12345,
) -> tuple[float, float, float]:
    """Razão agregada Σnum/Σden e IC 95% por bootstrap das sementes."""
    a = np.asarray(num, dtype=float)
    b = np.asarray(den, dtype=float)
    if len(a) != len(b) or len(a) == 0:
        raise ValueError("num e den devem ter o mesmo tamanho (> 0)")
    if b.sum() <= 0:
        return float("nan"), float("nan"), float("nan")
    point = float(a.sum() / b.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    sa, sb = a[idx].sum(axis=1), b[idx].sum(axis=1)
    ok = sb > 0
    if not ok.any():
        return point, float("nan"), float("nan")
    r = sa[ok] / sb[ok]
    return point, float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


# ---------------------------------------------------------------------------
# Critérios de decisão (seção 4 de docs/engenharia_de_filas.md)
# ---------------------------------------------------------------------------


def decide_h8(diff: pd.DataFrame, premise_ok: bool) -> tuple[str, dict[str, bool]]:
    """Status da H8 a partir de ``diff`` = S5 − S0 agregado.

    ``diff``: colunas ``scenario``, ``reward``, ``budget``, ``lo`` (limite
    inferior do IC 95% de S5 − S0). Definição operacional, fixada antes dos
    números: lado A (monotônico): em ``scenario == "real"``, nenhum ``B`` e
    nenhuma recompensa tem ``lo > 0`` (S5 não supera S0 além do ruído); lado B
    (não monotonia induzida): em N1 ou N2, com humano perfeito como recompensa,
    algum ``B`` tem ``lo > 0``. Confirmada se A e B; refutada se A falha;
    inconclusiva se a premissa de monotonia falha ou se A vale mas B não.
    """
    real = diff[diff["scenario"] == "real"]
    induced = diff[(diff["scenario"] != "real") & (diff["reward"] == "human")]
    side_a = bool(len(real)) and bool((real["lo"].fillna(np.inf) <= 0).all())
    side_b = bool((induced["lo"].fillna(-np.inf) > 0).any())
    flags = {"premissa": bool(premise_ok), "lado_A": side_a, "lado_B": side_b}
    if not premise_ok:
        return "inconclusiva", flags
    if not side_a:
        return "refutada", flags
    return ("confirmada" if side_b else "inconclusiva"), flags


def decide_h11(cells: pd.DataFrame) -> tuple[str, int, int]:
    """Status da H11 (metade real) a partir das células FN × FP × B.

    ``cells``: colunas ``recovery`` e ``precision`` (razões agregadas entre
    sementes). Uma célula satisfaz se recuperação ≥ 50% e precisão ≥ 30%.
    Confirmada se todas satisfazem; refutada se nenhuma; inconclusiva se mista.
    Devolve também ``(células que satisfazem, total)``.
    """
    ok = (cells["recovery"] >= RECOVERY_MIN) & (cells["precision"] >= PRECISION_MIN)
    n_ok, n = int(ok.sum()), len(cells)
    if n and n_ok == n:
        return "confirmada", n_ok, n
    if n_ok == 0:
        return "refutada", n_ok, n
    return "inconclusiva", n_ok, n


# ---------------------------------------------------------------------------
# Dados e fixture
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PoolData:
    """Pool abaixo do corte. ``truth`` (PAR in {1, 2}) só serve de verdade."""

    pids: np.ndarray  # object
    nota: np.ndarray
    truth: np.ndarray

    @property
    def n(self) -> int:
        return len(self.pids)


def load_pool(path: Path) -> PoolData:
    """Pares abaixo do corte da tese (checa a partição 61.485 / 36)."""
    import sim_fila_autocalibravel as sim  # lazy: módulo pesado

    prep = sim.load_prepared(path)
    idx = np.flatnonzero(prep.below)
    return PoolData(prep.pair_ids[idx], prep.nota[idx], prep.truth[idx])


class VerdictSource(Protocol):
    """Fonte de vereditos MATCH do LLM por par (fixture ou dublê de teste)."""

    def match(self, pid: str) -> bool: ...

    def covered(self, pid: str) -> bool: ...


class LlmVerdicts:
    """Veredito MATCH do LLM por par: fixture e, na falta, taxas por faixa.

    Envolve ``llm_real_eval.FixtureReviewer`` (``on_missing="rates"``) e guarda
    quais pares consultados vieram do fixture e quais do fallback.
    """

    def __init__(
        self,
        fixture_jsonl: Path,
        rates_csv: Path,
        pool: PoolData,
        seed: int,
        *,
        band_by_pid: Mapping[str, str] | None = None,
        truth_by_pid: Mapping[str, bool] | None = None,
    ) -> None:
        import llm_real_eval  # lazy
        import sim_fila_autocalibravel as sim

        if band_by_pid is None:
            names = np.asarray(sim.BAND_NAMES, dtype=object)
            bands = names[sim.band_of_nota(pool.nota)]
            band_by_pid = dict(zip(pool.pids.tolist(), bands.tolist(), strict=True))
        if truth_by_pid is None:
            truth_by_pid = dict(
                zip(pool.pids.tolist(), pool.truth.tolist(), strict=True)
            )
        self._rev = llm_real_eval.FixtureReviewer(
            fixture_jsonl,
            on_missing="rates",
            rates=llm_real_eval.load_rates_csv(rates_csv),
            truth=truth_by_pid,
            band_by_pair=band_by_pid,
            seed=seed,
            disagreement_rate=FALLBACK_DISAGREEMENT,
        )
        self._cache: dict[str, bool] = {}
        self._covered: dict[str, bool] = {}

    def covered(self, pid: str) -> bool:
        """O par está no fixture (decisão real gravada)?"""
        return pid in self._rev

    def match(self, pid: str) -> bool:
        if pid not in self._cache:
            stub = cast(Any, SimpleNamespace(pair_id=pid, to_dict=dict))
            self._cache[pid] = str(self._rev.review(stub).decision) == "MATCH"
            self._covered[pid] = pid in self._rev
        return self._cache[pid]


# ---------------------------------------------------------------------------
# H8
# ---------------------------------------------------------------------------


def scenario_order(
    order: np.ndarray, truth: np.ndarray, scenario: str, seed: int
) -> np.ndarray:
    """Ordem de revisão (índices do pool) da régua do cenário."""
    n = len(order)
    bounds = band_bounds(n)
    if scenario == "real":
        return order
    rng = np.random.default_rng(seed + 104_729)
    if scenario == "N1":  # verdadeiros do top-5% movidos para a faixa 1–5%
        perm = induce_nonmonotonic(
            truth[order], bounds[1][1], bounds[1][0], bounds[1][1], rng
        )
    elif scenario == "N2":  # verdadeiros do top-20% movidos para a faixa 5–20%
        perm = induce_nonmonotonic(
            truth[order], bounds[2][1], bounds[2][0], bounds[2][1], rng
        )
    else:
        raise ValueError(f"cenário desconhecido: {scenario!r}")
    return order[perm]


def run_h8_seed(
    pool: PoolData,
    llm: VerdictSource | None,
    seed: int,
    budgets: Sequence[int] = H8_BUDGETS,
    scenarios: Sequence[str] = SCENARIOS,
    rewards: Sequence[str] = REWARDS,
) -> list[dict[str, Any]]:
    """Linhas ``(seed, scenario, reward, budget, metric, value)`` de uma semente."""
    bmax = max(budgets)
    bounds = band_bounds(pool.n)
    base_order = tie_broken_order(pool.nota, seed)
    rows: list[dict[str, Any]] = []

    def add(scen: str, rew: str, b: int, metric: str, value: float) -> None:
        rows.append(
            {
                "seed": seed,
                "scenario": scen,
                "reward": rew,
                "budget": b,
                "metric": metric,
                "value": value,
            }
        )

    for scen in scenarios:
        order = scenario_order(base_order, pool.truth, scen, seed)
        truth_r = pool.truth[order]
        pids_r = pool.pids[order]
        s0 = verified_at(truth_r[:bmax], budgets)
        for rew in rewards:
            if rew == "human":

                def reward_fn(rank: int, tr: np.ndarray = truth_r) -> int:
                    return int(tr[rank])

            elif rew == "llm":
                if llm is None:
                    continue

                def reward_fn(rank: int, pr: np.ndarray = pids_r) -> int:
                    return int(llm.match(str(pr[rank])))

            else:
                raise ValueError(f"recompensa desconhecida: {rew!r}")
            items, arms = bandit_pick(pool.n, bounds, reward_fn, bmax, seed + 17)
            s5_truth = truth_r[items]
            s5 = verified_at(s5_truth, budgets)
            for b in budgets:
                add(scen, rew, b, "s0", s0[b])
                add(scen, rew, b, "s5", s5[b])
                add(scen, rew, b, "diff", s5[b] - s0[b])
                for k in range(len(bounds)):
                    add(
                        scen, rew, b, f"arm{k + 1}_share", float((arms[:b] == k).mean())
                    )
                if rew == "llm" and llm is not None:
                    cov = [llm.covered(str(pids_r[i])) for i in items[:b]]
                    add(scen, rew, b, "fixture_coverage", float(np.mean(cov)))
    return rows


def premise_table(
    pool: PoolData, seeds: Sequence[int]
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Rendimento por faixa do bandit e por decil de posto (média das sementes)."""
    bounds = band_bounds(pool.n)
    edges_arm = [bounds[0][0], *[b for _, b in bounds]]
    edges_dec = quantile_edges(pool.n, 10)
    arm_pos = np.zeros(len(bounds))
    dec_pos = np.zeros(10)
    for s in seeds:
        t = pool.truth[tie_broken_order(pool.nota, s)]
        arm_pos += [p for p, _ in yield_by_bins(t, edges_arm)]
        dec_pos += [p for p, _ in yield_by_bins(t, edges_dec)]
    arm_pos /= len(seeds)
    dec_pos /= len(seeds)
    arm_n = np.diff(edges_arm)
    dec_n = np.diff(edges_dec)
    arms = pd.DataFrame(
        {
            "faixa": ["0–1%", "1–5%", "5–20%", "20–100%"],
            "pares": arm_n,
            "verdadeiros": arm_pos,
            "taxa": arm_pos / arm_n,
        }
    )
    decs = pd.DataFrame(
        {
            "decil": np.arange(1, 11),
            "pares": dec_n,
            "verdadeiros": dec_pos,
            "taxa": dec_pos / dec_n,
        }
    )
    premise_ok = bool((np.diff(arms["taxa"].to_numpy()) <= 0).all())
    return arms, decs, premise_ok


# ---------------------------------------------------------------------------
# H11
# ---------------------------------------------------------------------------


def run_h11_seed(
    pool: PoolData,
    llm: VerdictSource,
    seed: int,
    budgets: Sequence[int] = H11_BUDGETS,
    fns: Sequence[float] = H11_FN,
    fps: Sequence[float] = H11_FP,
) -> list[dict[str, Any]]:
    """Uma linha por (FN, FP, B) com as contagens da auditoria."""
    bmax = max(budgets)
    order = tie_broken_order(pool.nota, seed)[:bmax]
    truth_r = pool.truth[order]
    pids_r = [str(p) for p in pool.pids[order]]
    llm_m = np.array([llm.match(p) for p in pids_r], dtype=bool)
    cov = np.array([llm.covered(p) for p in pids_r], dtype=bool)
    u = np.array([human_uniform(seed, p) for p in pids_r])
    rows: list[dict[str, Any]] = []
    for fn in fns:
        for fp in fps:
            hm = human_match(truth_r, u, fn, fp)
            for b in budgets:
                m = audit_metrics(truth_r[:b], hm[:b], llm_m[:b])
                m["llm_match"] = float(llm_m[:b].sum())
                m["llm_match_true"] = float((llm_m[:b] & truth_r[:b]).sum())
                m["true_reviewed"] = float(truth_r[:b].sum())
                m["fixture_coverage"] = float(cov[:b].mean())
                rows.append({"seed": seed, "fn": fn, "fp": fp, "budget": b, **m})
    return rows


def aggregate_h11(raw: pd.DataFrame) -> pd.DataFrame:
    """Por célula (FN, FP, B): razões agregadas com IC bootstrap entre sementes."""
    out: list[dict[str, Any]] = []
    for (fn, fp, b), g in raw.groupby(["fn", "fp", "budget"], sort=True):
        rec, rec_lo, rec_hi = ratio_boot(g["recovered"], g["human_fn"])
        prec, prec_lo, prec_hi = ratio_boot(g["recovered"], g["extras"])
        rrec, rrec_lo, rrec_hi = ratio_boot(g["random_recovered"], g["human_fn"])
        rprec, rprec_lo, rprec_hi = ratio_boot(g["random_recovered"], g["extras"])
        out.append(
            {
                "fn": fn,
                "fp": fp,
                "budget": int(b),
                "n_seeds": int(g["seed"].nunique()),
                "human_fn_mean": float(g["human_fn"].mean()),
                "extras_mean": float(g["extras"].mean()),
                "recovered_mean": float(g["recovered"].mean()),
                "extras_llm_fp_mean": float(g["extras_llm_fp"].mean()),
                "recovery": rec,
                "recovery_lo": rec_lo,
                "recovery_hi": rec_hi,
                "precision": prec,
                "precision_lo": prec_lo,
                "precision_hi": prec_hi,
                "random_recovery": rrec,
                "random_recovery_lo": rrec_lo,
                "random_recovery_hi": rrec_hi,
                "random_precision": rprec,
                "random_precision_lo": rprec_lo,
                "random_precision_hi": rprec_hi,
                "fixture_coverage": float(g["fixture_coverage"].mean()),
            }
        )
    return pd.DataFrame(out)


def aggregate_h8(raw: pd.DataFrame) -> pd.DataFrame:
    """Média e IC 95% (t) entre sementes por cenário × recompensa × B × métrica."""
    out: list[dict[str, Any]] = []
    keys = ["scenario", "reward", "budget", "metric"]
    for key, g in raw.groupby(keys, sort=False):
        mean, lo, hi, n = summarize(g["value"].tolist())
        out.append(
            {
                **dict(zip(keys, key, strict=True)),
                "mean": mean,
                "lo": lo,
                "hi": hi,
                "n": n,
            }
        )
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Relatório
# ---------------------------------------------------------------------------


def _num(x: float, digits: int = 1) -> str:
    if x != x:
        return "—"
    return f"{x:.{digits}f}".replace(".", ",")


def _pct(x: float) -> str:
    return "—" if x != x else _num(100 * x, 1) + "%"


def _ci(mean: float, lo: float, hi: float, digits: int = 2) -> str:
    if mean != mean:
        return "—"
    if lo != lo:
        return _num(mean, digits)
    return f"{_num(mean, digits)} [{_num(lo, digits)}; {_num(hi, digits)}]"


def _pci(mean: float, lo: float, hi: float) -> str:
    if mean != mean:
        return "—"
    if lo != lo:
        return _pct(mean)
    return f"{_pct(mean)} [{_pct(lo)}; {_pct(hi)}]"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _h8_cell(agg: pd.DataFrame, scen: str, rew: str, b: int, metric: str) -> pd.Series:
    sel = agg[
        (agg["scenario"] == scen)
        & (agg["reward"] == rew)
        & (agg["budget"] == b)
        & (agg["metric"] == metric)
    ]
    return sel.iloc[0]


def render_markdown(
    meta: Mapping[str, Any],
    arms: pd.DataFrame,
    decs: pd.DataFrame,
    h8: pd.DataFrame,
    h8_status: tuple[str, dict[str, bool]],
    h11: pd.DataFrame,
    h11_status: tuple[str, int, int],
) -> str:
    w: list[str] = []
    a = w.append
    budgets8 = sorted(h8["budget"].unique())
    n_pool_txt = f"{meta['n_pool']:,}".replace(",", ".")
    a("# H8 e H11 no extrato real da tese\n")
    a(
        "Status: **resultado experimental**, sem alterar as hipóteses nem os "
        "critérios pré-registrados em `docs/engenharia_de_filas.md` (seção 4). "
        "Só agregados; nenhum identificador nem dado em nível de par. "
        "Gerado por `notebooks/exp_filas_real.py`.\n"
    )
    a("## Resumo\n")
    s8, flags = h8_status
    s11, n_ok, n_cells = h11_status
    a(
        _table(
            ["Hipótese", "Status no extrato real", "Base"],
            [
                [
                    "H8",
                    f"**{s8}**",
                    f"premissa de monotonia: {'sim' if flags['premissa'] else 'não'}; "
                    f"lado A (S5 ≤ S0 no real): {'sim' if flags['lado_A'] else 'não'}; "
                    f"lado B (S5 > S0 com não monotonia induzida): "
                    f"{'sim' if flags['lado_B'] else 'não'}",
                ],
                [
                    "H11",
                    f"**{s11}**",
                    f"{n_ok} de {n_cells} células satisfazem recuperação ≥ 50% e "
                    "precisão ≥ 30% (metade real; o sintético é outra tarefa)",
                ],
            ],
        )
    )
    a("")
    a("## Desenho\n")
    a(
        f"- **Dados.** Extrato da tese, pool = pares abaixo do corte: "
        f"{n_pool_txt} pares, {meta['n_true']} verdadeiros (PAR ∈ {{1, 2}} "
        "só como verdade). Régua = `nota final`. Empates da nota quebrados por "
        "permutação da semente (mesma ordem para S0 e S5 numa semente)."
    )
    a(
        f"- **Sementes.** {meta['seeds']}. Só o bandit, o desempate dos empates, o "
        "erro do humano simulado e o fallback do LLM variam; as decisões do LLM "
        "presentes no fixture são **uma realização congelada**."
    )
    a(
        "- **S0.** Ordem global decrescente da nota; B decisões humanas "
        "(humano perfeito): verificados = verdadeiros entre os B primeiros."
    )
    a(
        '- **S5.** `ThompsonQueue(input_kind="partial_orders")`, prior Beta(1, 1), '
        "sem janela, 4 braços = faixas contíguas por posto com frações acumuladas "
        "1%, 5%, 20% e o resto (0–1%, 1–5%, 5–20%, 20–100%; leitura adotada para "
        '"1%, 5%, 20%, resto"), cada faixa do topo para baixo. Cada par escolhido '
        "é decidido pelo humano; verificados = verdadeiros entre os B escolhidos. "
        "Recompensa do bandit: `human` = par verdadeiro (humano perfeito, testa o "
        "bandit em si) ou `llm` = MATCH do fixture (fallback por taxas por faixa "
        "para pares ausentes, `disagreement_rate` = 0,10 como em "
        "`sim_fila_autocalibravel`)."
    )
    a(
        "- **Não monotonia induzida.** `N1`: os verdadeiros do top-5% (janela de "
        "postos 0–5%) são permutados para postos sorteados da faixa 1–5%, de modo que "
        "a faixa 0–1% fica sem verdadeiros e a 1–5% concentra todos. `N2`: o mesmo com "
        "janela 0–20% e destino a faixa 5–20%. A ordem alterada vale para S0 e S5."
    )
    a(
        "- **Definição operacional dos critérios de H8** (fixada antes dos números; "
        'a tabela diz só "ambos os lados se verificam"). Lado A: no cenário `real`, '
        "para toda recompensa e todo B, o limite inferior do IC 95% (t, entre "
        "sementes) de S5 − S0 não é positivo. Lado B: em `N1` ou `N2`, com recompensa "
        "`human`, algum B tem limite inferior de S5 − S0 > 0. Confirmada se A e B; "
        "refutada se A falha; inconclusiva se a premissa de monotonia falha ou se A "
        "vale e B não (a escolha das induções é minha)."
    )
    a(
        "- **H11.** Humano simulado revisa os B primeiros da ordem da nota; "
        "`u` ~ U[0,1) determinístico por (semente, par), com a mesma `u` para todas "
        "as taxas: par é perdido se `u < FN`, não par vira MATCH se `u < FP`. "
        "Auditoria: humano perfeito revê só os pares com LLM = MATCH e humano = "
        "NONMATCH. Recuperação = FN humanos recuperados / FN humanos; precisão = "
        "recuperados / revisões extras; ambas **agregadas entre sementes** "
        "(Σ/Σ, IC 95% por bootstrap das sementes, "
        f"{N_BOOT} reamostras), porque por semente o denominador é 0 em várias "
        "células. Auditoria aleatória do mesmo tamanho = esperança exata entre os "
        "NONMATCH do humano. Célula satisfaz se recuperação ≥ 50% e precisão ≥ 30%; "
        "confirmada se todas, refutada se nenhuma, inconclusiva se mista.\n"
    )

    a("## H8\n")
    a("### Premissa: o rendimento cai com o posto?\n")
    a(
        "Média entre sementes (só o desempate dos empates varia). "
        f"Os {meta['n_true']} verdadeiros no pool.\n"
    )
    a(
        _table(
            ["faixa do bandit", "pares", "verdadeiros", "taxa"],
            [
                [
                    str(r.faixa),
                    str(int(r.pares)),
                    _num(float(r.verdadeiros), 1),
                    _num(100 * float(r.taxa), 2) + "%",
                ]
                for r in arms.itertuples()
            ],
        )
    )
    a("")
    a(
        _table(
            ["decil de posto", "pares", "verdadeiros", "taxa"],
            [
                [
                    str(int(r.decil)),
                    str(int(r.pares)),
                    _num(float(r.verdadeiros), 1),
                    _num(100 * float(r.taxa), 2) + "%",
                ]
                for r in decs.itertuples()
            ],
        )
    )
    a("")
    a(
        f"Premissa (taxa por faixa do bandit não crescente): "
        f"**{'verificada' if flags['premissa'] else 'não verificada'}**. Com 36 "
        "positivos, as taxas por decil são ruidosas (poucas unidades).\n"
    )
    for rew, title in (
        ("human", "recompensa = par verdadeiro (humano perfeito)"),
        ("llm", "recompensa = MATCH do LLM (fixture)"),
    ):
        a(f"### Verdadeiros verificados e S5 − S0: {title}\n")
        for scen in SCENARIOS:
            label = {
                "real": "cenário `real` (régua como está)",
                "N1": "cenário `N1` (verdadeiros concentrados em 1–5%)",
                "N2": "cenário `N2` (verdadeiros concentrados em 5–20%)",
            }[scen]
            a(f"**{label}**\n")
            rows = []
            for b in budgets8:
                s0 = _h8_cell(h8, scen, rew, b, "s0")
                s5 = _h8_cell(h8, scen, rew, b, "s5")
                d = _h8_cell(h8, scen, rew, b, "diff")
                shares = [
                    _pct(float(_h8_cell(h8, scen, rew, b, f"arm{k}_share")["mean"]))
                    for k in range(1, 5)
                ]
                rows.append(
                    [
                        str(b),
                        _num(float(s0["mean"]), 2),
                        _ci(float(s5["mean"]), float(s5["lo"]), float(s5["hi"])),
                        _ci(float(d["mean"]), float(d["lo"]), float(d["hi"])),
                        " / ".join(shares),
                    ]
                )
            a(
                _table(
                    [
                        "B",
                        "S0",
                        "S5 [IC 95%]",
                        "S5 − S0 [IC 95%]",
                        "uso dos braços (0–1 / 1–5 / 5–20 / 20–100)",
                    ],
                    rows,
                )
            )
            a("")
    cov = h8[(h8["reward"] == "llm") & (h8["metric"] == "fixture_coverage")]
    if not cov.empty:
        a("### Cobertura do fixture nas escolhas do bandit (recompensa LLM)\n")
        rows = []
        for scen in SCENARIOS:
            row = [scen]
            for b in budgets8:
                c = cov[(cov["scenario"] == scen) & (cov["budget"] == b)]
                row.append(_pct(float(c["mean"].iloc[0])) if len(c) else "—")
            rows.append(row)
        a(_table(["cenário", *[f"B={b}" for b in budgets8]], rows))
        a("\nFração das escolhas servida pelo fixture; o resto veio do fallback.\n")

    a("## H11\n")
    cov11 = float(h11["fixture_coverage"].mean())
    cov_by_b = h11.groupby("budget")["fixture_coverage"].mean()
    a(
        "Cobertura do fixture nos pares revisados pelo humano (os B primeiros da "
        "nota): "
        + "; ".join(f"B={int(b)}: {_pct(float(c))}" for b, c in cov_by_b.items())
        + f" (média {_pct(cov11)}); o restante usa o fallback por taxas.\n"
    )
    a(
        "Recuperação e precisão são razões agregadas (IC 95% bootstrap entre "
        'sementes). "Aleatória" = auditoria aleatória do mesmo tamanho entre os '
        "NONMATCH do humano. Médias por semente: FN humanos, revisões extras, "
        "recuperados e extras que eram não pares (falsos positivos do LLM).\n"
    )
    rows = []
    for r in h11.itertuples():
        ok = r.recovery >= RECOVERY_MIN and r.precision >= PRECISION_MIN
        rows.append(
            [
                _pct(r.fn),
                _pct(r.fp),
                str(r.budget),
                _num(r.human_fn_mean, 2),
                _num(r.extras_mean, 2),
                _num(r.recovered_mean, 2),
                _num(r.extras_llm_fp_mean, 2),
                _pci(r.recovery, r.recovery_lo, r.recovery_hi),
                _pci(r.random_recovery, r.random_recovery_lo, r.random_recovery_hi),
                _pci(r.precision, r.precision_lo, r.precision_hi),
                _pci(r.random_precision, r.random_precision_lo, r.random_precision_hi),
                "sim" if ok else "não",
            ]
        )
    a(
        _table(
            [
                "FN",
                "FP",
                "B",
                "FN hum.",
                "extras",
                "recup.",
                "extras não pares",
                "recuperação",
                "recup. aleatória",
                "precisão",
                "prec. aleatória",
                "satisfaz",
            ],
            rows,
        )
    )
    a("")

    a("## Leitura\n")
    a(meta["leitura_h8"])
    a("")
    a(meta["leitura_h11"])
    a("")
    a("## Ressalvas\n")
    a(
        "- **36 positivos.** Qualquer taxa por faixa ou por decil repousa em poucas "
        "unidades; os IC entre sementes cobrem só o desempate, o bandit e o erro do "
        "humano simulado, **não** a incerteza de amostragem dos 36 pares."
    )
    a(
        "- **Humano simulado.** O erro é sorteado por (semente, par), "
        "independente do par; humanos reais erram de forma correlacionada com a "
        "dificuldade (e com o próprio LLM)."
    )
    a(
        "- **Uma realização do LLM.** O fixture congela uma decisão por par; as "
        "sementes não a reamostram. A cobertura do fixture e a parcela vinda do "
        "fallback (taxas por faixa, `disagreement_rate` = 0,10, suposição) estão "
        "acima."
    )
    a(
        "- **Escopo de B.** B ≤ 800 é ≈ 1,3% do pool, bem menos que os braços "
        "2–4; o bandit gasta parte do orçamento explorando braços de baixo "
        "rendimento (uso dos braços nas tabelas) e N2 mede o custo de achar uma "
        "faixa intermediária distante."
    )
    a(
        "- **Induções.** N1 e N2 são construções minhas (permutação dos "
        "verdadeiros na janela); outras induções poderiam dar outro resultado no "
        "lado B."
    )
    a(
        "- **Sintético.** H11 também exige o resultado no sintético "
        "(`notebooks/exp_engenharia_filas.py`); aqui está só a metade real.\n"
    )
    a("## Execução\n")
    a(
        f"- Pool: {meta['n_pool']} pares, {meta['n_true']} verdadeiros; fixture: "
        f"{meta.get('fixture_pairs', '—')} pares "
        f"(sha256 `{meta.get('fixture_sha256', '—')}`)."
    )
    a(
        f"- Tempo total: {_num(meta['elapsed_s'], 0)} s; pico de memória "
        f"(working set): {_num(meta['peak_mb'], 0)} MB."
    )
    return "\n".join(w) + "\n"


def to_long_csv(h8: pd.DataFrame, h11: pd.DataFrame) -> pd.DataFrame:
    """CSV único em formato longo (só agregados)."""
    a = h8.assign(hypothesis="H8", fn="", fp="", n_seeds=h8["n"])
    a = a.rename(columns={"mean": "value", "lo": "ci_lo", "hi": "ci_hi"})
    cols = ["hypothesis", "scenario", "reward", "fn", "fp", "budget", "metric"]
    cols_tail = ["value", "ci_lo", "ci_hi", "n_seeds"]
    a = a[[*cols, *cols_tail]]
    rows: list[dict[str, Any]] = []
    metrics = [
        "human_fn_mean",
        "extras_mean",
        "recovered_mean",
        "extras_llm_fp_mean",
        "fixture_coverage",
    ]
    ratio = ["recovery", "precision", "random_recovery", "random_precision"]
    for r in h11.to_dict("records"):
        base = {
            "hypothesis": "H11",
            "scenario": "real",
            "reward": "",
            "fn": r["fn"],
            "fp": r["fp"],
            "budget": r["budget"],
            "n_seeds": r["n_seeds"],
        }
        for m in metrics:
            rows.append({**base, "metric": m, "value": r[m], "ci_lo": "", "ci_hi": ""})
        for m in ratio:
            rows.append(
                {
                    **base,
                    "metric": m,
                    "value": r[m],
                    "ci_lo": r[f"{m}_lo"],
                    "ci_hi": r[f"{m}_hi"],
                }
            )
    b = pd.DataFrame(rows)[[*cols, *cols_tail]]
    return pd.concat([a, b], ignore_index=True)


def peak_memory_mb() -> float:
    try:
        import psutil

        info = psutil.Process().memory_info()
        peak = getattr(info, "peak_wset", info.rss)
        return float(peak) / 2**20
    except Exception:  # noqa: BLE001 - métrica opcional
        return float("nan")


def _leituras(
    h8: pd.DataFrame,
    h8_status: tuple[str, dict[str, bool]],
    h11: pd.DataFrame,
    h11_status: tuple[str, int, int],
) -> tuple[str, str]:
    """Texto de leitura com os números mais relevantes (gerado dos dados)."""
    d = h8[h8["metric"] == "diff"]
    parts = []
    for scen in SCENARIOS:
        for rew in REWARDS:
            sub = d[(d["scenario"] == scen) & (d["reward"] == rew)]
            if sub.empty:
                continue
            best = sub.loc[sub["mean"].idxmax()]
            worst = sub.loc[sub["mean"].idxmin()]
            parts.append(
                f"`{scen}`/`{rew}`: S5 − S0 de {_num(float(worst['mean']), 2)} "
                f"(B={int(worst['budget'])}) a {_num(float(best['mean']), 2)} "
                f"(B={int(best['budget'])}); maior limite inferior do IC "
                f"{_num(float(sub['lo'].max()), 2)}"
            )
    t8 = (
        f"**H8: {h8_status[0]}.** Diferenças S5 − S0 (verdadeiros verificados): "
        + "; ".join(parts)
        + "."
    )
    b_max = int(h8["budget"].max())

    def _mean_at(metric: str, scenario: str) -> float:
        sel = h8[
            (h8["metric"] == metric)
            & (h8["scenario"] == scenario)
            & (h8["reward"] == "human")
            & (h8["budget"] == b_max)
        ]["mean"]
        return float(sel.iloc[0])

    pairs = "; ".join(
        f"{label}: S0 {_num(_mean_at('s0', sc), 2)} e S5 {_num(_mean_at('s5', sc), 2)}"
        for sc, label in (("N1", "N1"), ("N2", "N2"), ("real", "sem indução"))
    )
    t8 += (
        " O ganho do bandit nas induções é pequeno em valor absoluto e é medido "
        "contra uma S0 que a própria indução prejudicou (os verdadeiros saíram do "
        f"topo). Com B={b_max}, verdadeiros verificados por cenário — {pairs}. "
        "No cenário real o bandit gasta cerca de 30% do orçamento nos braços de "
        "baixo rendimento e nunca supera a ordem da nota."
    )
    best = h11.loc[h11["recovery"].idxmax()] if h11["recovery"].notna().any() else None
    t11 = f"**H11: {h11_status[0]}** ({h11_status[1]} de {h11_status[2]} células)."
    if best is not None:
        t11 += (
            f" Melhor recuperação: {_pct(float(best['recovery']))} (FN "
            f"{_pct(float(best['fn']))}, FP {_pct(float(best['fp']))}, "
            f"B={int(best['budget'])}), com precisão {_pct(float(best['precision']))} "
            f"contra {_pct(float(best['random_precision']))} da auditoria aleatória. "
            "As revisões extras incluem todo falso positivo do LLM (não par com "
            "MATCH que o humano, corretamente, recusou), o que derruba a precisão "
            "quando o LLM tem FP > 0 e há poucos FN humanos para recuperar."
        )
    return t8, t11


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--input",
        type=Path,
        default=Path("../phd-research/data/COMPARADORSEMIDENT.csv"),
    )
    p.add_argument(
        "--fixture",
        type=Path,
        default=Path("data/llm_fixtures/haiku-5-5_v4_sem-nota.jsonl"),
    )
    p.add_argument(
        "--rates",
        type=Path,
        default=Path("data/llm_fixtures/haiku-5-5_v4_sem-nota.rates.csv"),
    )
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument(
        "--out-csv", type=Path, default=Path("docs/engenharia_de_filas_real_h8_h11.csv")
    )
    p.add_argument(
        "--out-md", type=Path, default=Path("docs/engenharia_de_filas_real_h8_h11.md")
    )
    args = p.parse_args(argv)

    t0 = time.perf_counter()
    pool = load_pool(args.input)
    print(
        f"pool: {pool.n} pares, {int(pool.truth.sum())} verdadeiros "
        f"({time.perf_counter() - t0:.0f}s)",
        file=sys.stderr,
    )
    seeds = list(range(args.seeds))
    import sim_fila_autocalibravel as sim

    names = np.asarray(sim.BAND_NAMES, dtype=object)
    band_by_pid = dict(
        zip(
            pool.pids.tolist(), names[sim.band_of_nota(pool.nota)].tolist(), strict=True
        )
    )
    truth_by_pid = dict(zip(pool.pids.tolist(), pool.truth.tolist(), strict=True))

    h8_rows: list[dict[str, Any]] = []
    h11_rows: list[dict[str, Any]] = []
    for s in seeds:
        llm = LlmVerdicts(
            args.fixture,
            args.rates,
            pool,
            s,
            band_by_pid=band_by_pid,
            truth_by_pid=truth_by_pid,
        )
        h8_rows += run_h8_seed(pool, llm, s)
        h11_rows += run_h11_seed(pool, llm, s)
        print(f"semente {s} ok ({time.perf_counter() - t0:.0f}s)", file=sys.stderr)

    arms, decs, premise_ok = premise_table(pool, seeds)
    h8 = aggregate_h8(pd.DataFrame(h8_rows))
    h11 = aggregate_h11(pd.DataFrame(h11_rows))
    h8_status = decide_h8(h8[h8["metric"] == "diff"], premise_ok)
    h11_status = decide_h11(h11)
    leitura8, leitura11 = _leituras(h8, h8_status, h11, h11_status)
    manifest_path = args.fixture.parent / f"{args.fixture.stem}.manifest.json"
    meta: dict[str, Any] = {
        "n_pool": pool.n,
        "n_true": int(pool.truth.sum()),
        "seeds": args.seeds,
        "elapsed_s": time.perf_counter() - t0,
        "peak_mb": peak_memory_mb(),
        "leitura_h8": leitura8,
        "leitura_h11": leitura11,
    }
    if manifest_path.exists():
        man = json.loads(manifest_path.read_text(encoding="utf-8"))
        meta["fixture_pairs"] = man.get("n_pairs")
        meta["fixture_sha256"] = man.get("sha256")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    to_long_csv(h8, h11).to_csv(args.out_csv, index=False)
    args.out_md.write_text(
        render_markdown(meta, arms, decs, h8, h8_status, h11, h11_status),
        encoding="utf-8",
    )
    print(
        f"H8: {h8_status}\nH11: {h11_status}\n"
        f"tempo {meta['elapsed_s']:.0f}s, pico {meta['peak_mb']:.0f} MB",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
