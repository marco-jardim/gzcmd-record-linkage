"""Comparações a orçamento humano igual contra a ordenação O1 (escore do OpenRecLink).

Todos os experimentos anteriores usaram como referência a ordem pela nota final
do CdR (O2 da tese). A tese também tem a O1, a ordem pelo escore do OpenRecLink
(``SCORE``), igualmente **sem rótulos**; ela recupera 19 pares verdadeiros abaixo
do corte com 96 revisões, contra 12 da O2. Este script refaz as comparações
principais com a O1. Offline: nenhuma chamada de API; o veredito do LLM vem só do
fixture local de decisões reais. ``PAR`` entra apenas como verdade (humano
perfeito). Só agregados são escritos.

Empates. ``SCORE`` tem só 82 valores distintos nos 61.485 pares abaixo do corte,
então a O1 tem blocos grandes de empate e o desempate muda o resultado. A ordem
do arquivo do extrato **não** é neutra (os positivos aparecem mais cedo que os
demais): desempatar pela posição no arquivo ("arquivo", que reproduz os 19 e 12
da tese em 96 revisões) é otimista. O desempate principal aqui é **aleatório e
determinístico por semente** (permutação das linhas); relatam-se média, mínimo e
máximo sobre as sementes, e a ordem do arquivo como variante rotulada.

Uso (a partir da raiz do repositório)::

    python notebooks/exp_baseline_o1.py --real-input \\
        ../phd-research/data/COMPARADORSEMIDENT.csv
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from collections.abc import Mapping, Sequence  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from dataclasses import dataclass, replace  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from typing import Any, cast  # noqa: E402

import exp_engenharia_filas as exp  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

from gzcmd_record_linkage.arranjo_corte import (  # noqa: E402
    THESIS_PARTITION,
    below_cutoff,
)
from gzcmd_record_linkage.calibration import get_nota_series  # noqa: E402
from gzcmd_record_linkage.llm_dossier import make_pair_id  # noqa: E402
from gzcmd_record_linkage.loader import LoadConfig, load_comparador_csv  # noqa: E402

BUDGETS: tuple[int, ...] = (50, 96, 100, 200, 400, 800)
WINDOWS: tuple[int, ...] = (500, 1000, 2000, 3000)
END_TO_END_BUDGETS: tuple[int, ...] = (50, 100, 127, 200)
THESIS_O1_96, THESIS_O2_96 = 19, 12  # verdadeiros abaixo do corte em 96 revisões
MATCH, NONMATCH, UNSURE, NONE = exp.MATCH, exp.NONMATCH, exp.UNSURE, exp.NONE


# ---------------------------------------------------------------------------
# Escore, ordens e dados
# ---------------------------------------------------------------------------


def resolve_score(raw: pd.DataFrame) -> NDArray[np.float64]:
    """``SCORE`` numérico: aceita ``SCORE`` ou o cabeçalho ``SCORE,C,15,0``.

    Valores em texto aceitam espaços e vírgula decimal (``" 7,02"``).
    """
    cols = [c for c in raw.columns if str(c).split(",")[0].strip() == "SCORE"]
    if not cols:
        raise KeyError("sem coluna SCORE (nem 'SCORE,C,15,0')")
    col = "SCORE" if "SCORE" in cols else cols[0]
    values = raw[col]
    if not pd.api.types.is_numeric_dtype(values):
        values = values.astype(str).str.strip().str.replace(",", ".", regex=False)
    out = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(out).all():
        raise ValueError("SCORE com valores ausentes ou não numéricos")
    return out


def rank_order(
    score: NDArray[np.float64], tie_seed: int | None = None
) -> NDArray[np.int64]:
    """Ordem decrescente do escore, com desempate documentado.

    ``tie_seed=None``: pela posição no arquivo ("arquivo"; otimista, ver o
    módulo). Com semente: desempate aleatório, determinístico por semente.
    """
    n = len(score)
    key = (
        np.arange(n) if tie_seed is None else np.random.default_rng(tie_seed).random(n)
    )
    return np.lexsort((key, -score)).astype(np.int64)


@dataclass
class Pool:
    """Pares do extrato (abaixo do corte ou todos); ``y`` só como verdade."""

    df: pd.DataFrame
    y: NDArray[np.bool_]
    below: NDArray[np.bool_]  # abaixo do corte (no pool "all" marca os 36 de interesse)
    nota: NDArray[np.float64]  # O2
    score: NDArray[np.float64]  # O1
    pair_ids: list[str]
    nota_all: NDArray[np.float64]
    score_all: NDArray[np.float64]

    @property
    def n(self) -> int:
        return len(self.y)


def load_pool(path: Path, which: str = "below") -> Pool:
    """Extrato da tese: ``which`` = ``below`` (61.485) ou ``all`` (61.696)."""
    raw = load_comparador_csv(path, cfg=LoadConfig(require_par=False))
    score_all = resolve_score(raw)
    raw = raw.assign(SCORE=score_all)
    par = pd.to_numeric(raw["PAR"], errors="coerce")
    below_all = np.asarray(below_cutoff(raw, expected=THESIS_PARTITION), dtype=bool)
    keep = raw.drop(columns=[c for c in ("PAR", "TARGET") if c in raw.columns])
    nota_all = pd.to_numeric(get_nota_series(keep), errors="coerce").to_numpy(float)
    sel = below_all if which == "below" else np.ones(len(raw), dtype=bool)
    df = keep.loc[sel].reset_index(drop=True)
    pair_ids = [
        make_pair_id(c, r, p)
        for c, r, p in zip(df["COMPREC"], df["REFREC"], df["PASSO"], strict=True)
    ]
    return Pool(
        df=df,
        y=par.isin([1, 2]).to_numpy()[sel],
        below=below_all[sel],
        nota=nota_all[sel],
        score=score_all[sel],
        pair_ids=pair_ids,
        nota_all=nota_all,
        score_all=score_all,
    )


def fixture_codes(pair_ids: Sequence[str], fixture: Path) -> NDArray[np.int8]:
    """Veredito **real** do fixture por par; ``NONE`` onde o fixture não cobre.

    Só decisões gravadas; nenhuma taxa sintética (``FixtureReviewer`` no modo
    padrão ``on_missing="error"``, consultado apenas para pares presentes).
    """
    import llm_real_eval  # lazy: módulo pesado; não é editado aqui

    plain = llm_real_eval.FixtureReviewer(fixture)
    code = {"MATCH": MATCH, "NONMATCH": NONMATCH, "UNSURE": UNSURE}
    out = np.full(len(pair_ids), NONE, dtype=np.int8)
    for i, pid in enumerate(pair_ids):
        if pid in plain:
            stub = cast(Any, SimpleNamespace(pair_id=pid, to_dict=dict))
            out[i] = code[str(plain.review(stub).decision)]
    return out


def found_at(y: NDArray[np.bool_], order: NDArray[np.int64], b: int) -> int:
    """Verdadeiros entre as ``b`` primeiras posições (humano perfeito)."""
    return int(y[order[:b]].sum())


def window_coverage(
    order: NDArray[np.int64],
    codes: NDArray[np.int8],
    y: NDArray[np.bool_],
    window: int,
) -> dict[str, float]:
    """Cobertura do fixture nos ``window`` primeiros da ordem."""
    top = order[:window]
    covered = codes[top] != NONE
    return {
        "window": float(min(window, len(order))),
        "covered": float(covered.sum()),
        "covered_fraction": float(covered.mean()),
        "positives": float(y[top].sum()),
        "positives_covered": float((y[top] & covered).sum()),
    }


def llm_window_order(
    order: NDArray[np.int64], codes: NDArray[np.int8], window: int
) -> NDArray[np.int64]:
    """Desempate pelo LLM na janela (L1): MATCH, depois UNSURE, depois o resto.

    Usa **só** o veredito real do fixture (``codes``). Par da janela sem veredito
    fica no "resto", junto com os NONMATCH, na ordem de base (não priorizado).
    Fora da janela vale a ordem de base. É uma permutação de ``order``.
    """
    top = order[:window]
    c = codes[top]
    return np.concatenate(
        [
            top[c == MATCH],
            top[c == UNSURE],
            top[(c != MATCH) & (c != UNSURE)],
            order[window:],
        ]
    ).astype(np.int64)


def o1_offset(
    pool: Pool, kind: str
) -> tuple[NDArray[np.float64], str, tuple[float, float] | None]:
    """Offset (logit) da O1 para S2: ``identity`` (SCORE) ou Platt por âncoras.

    ``platt``: ``fit_anchor_platt`` no ``SCORE`` do **extrato inteiro** (âncoras
    A+ = SCORE >= 9 com média 0,979; A- = SCORE < 5 com média 1e-4, os mesmos
    parâmetros usados para a nota) e aplicado aos pares do pool.
    """
    if kind == "identity":
        return pool.score.astype(float), "identity", None
    out, source = exp.ruler_offset_info(pool.score, anchor_nota=pool.score_all)
    if source == "stub":
        raise RuntimeError("O1: âncoras insuficientes; o stub não vale para SCORE")
    i = int(np.argmax(pool.score))
    j = int(np.argmin(pool.score))
    slope = float((out[i] - out[j]) / (pool.score[i] - pool.score[j]) - 1e-9)
    intercept = float(out[i] - (slope + 1e-9) * pool.score[i])
    return out, source, (intercept, slope)


# ---------------------------------------------------------------------------
# Uma semente (desempate aleatório da O1) com todas as estratégias
# ---------------------------------------------------------------------------

_STATE: dict[str, Any] = {}


def _init(path: Path, fixture: Path) -> None:
    pool = load_pool(path, "below")
    _STATE["pool"] = pool
    _STATE["codes"] = fixture_codes(pool.pair_ids, fixture)


def _permute(
    pool: Pool, codes: NDArray[np.int8], seed: int
) -> tuple[Pool, NDArray[np.int8]]:
    perm = np.random.default_rng(np.random.SeedSequence([seed, 11])).permutation(pool.n)
    new = replace(
        pool,
        df=pool.df.iloc[perm].reset_index(drop=True),
        y=pool.y[perm],
        below=pool.below[perm],
        nota=pool.nota[perm],
        score=pool.score[perm],
        pair_ids=[pool.pair_ids[i] for i in perm],
    )
    return new, codes[perm]


def run_seed(seed: int) -> list[dict[str, Any]]:
    """Todas as estratégias numa permutação (= desempate aleatório) da O1."""
    pool, codes = _permute(_STATE["pool"], _STATE["codes"], seed)
    y = pool.y
    n = pool.n
    rows: list[dict[str, Any]] = []

    def add(strategy: str, b: int, value: float, **extra: Any) -> None:
        rows.append(
            {"seed": seed, "strategy": strategy, "B": b, "found": value, **extra}
        )

    o1 = rank_order(pool.score)  # empates pelo índice = permutação aleatória
    o2 = rank_order(pool.nota)
    for b in BUDGETS:
        add("O1", b, found_at(y, o1, b))
        add("O2", b, found_at(y, o2, b))
    # (d) LLM como desempate sobre janelas de O1 (e de O2, para comparar)
    # "limpo": veredito só para pares dos 3.000 primeiros da O2 (cobertura que
    # depende só da nota); "bruto": todo o fixture (cobre 100% dos positivos, ou
    # seja, a cobertura depende do rótulo e a comparação favorece o LLM)
    clean = exp.restrict_to_window(pool.nota, codes, exp.REAL_LLM_WINDOW)
    for w in WINDOWS:
        for tag, base in (("O1", o1), ("O2", o2)):
            for kind, cd in (("L1", clean), ("L1bruto", codes)):
                ordered = llm_window_order(base, cd, w)
                for b in BUDGETS:
                    add(f"{kind}[{tag},W={w}]", b, found_at(y, ordered, b))
            for kind, cd in (("cobertura", clean), ("cobertura_bruta", codes)):
                cov = window_coverage(base, cd, y, w)
                rows.append(
                    {"seed": seed, "strategy": f"{kind}[{tag},W={w}]", "B": 0, **cov}
                )
    # (b) S2 com régua O1 (identidade e Platt) e O2 (plana P e calibrada C)
    base_inst = exp.Instance(
        df=pool.df,
        y=y,
        hard=np.zeros(n, dtype=bool),
        nota=pool.nota,
        llm_all=np.full(n, NONE, dtype=np.int8),
        llm_seen=np.full(n, NONE, dtype=np.int8),
        human=y.copy(),
        pair_ids=pool.pair_ids,
        nota_all=pool.nota_all,
    )
    spec_p = exp.build_spec(base_inst, ruler="pool")
    spec_c = exp.build_spec(base_inst, ruler="full")
    variants: list[tuple[str, exp.ModelSpec, exp.Instance]] = [
        ("O2-P", spec_p, base_inst),
        ("O2-C", spec_c, base_inst),
    ]
    for kind in ("identity", "platt"):
        off, source, _ = o1_offset(pool, kind)
        spec = replace(spec_p, offset=off, ruler_source=f"O1:{source}")
        variants.append((f"O1-{kind}", spec, replace(base_inst, nota=pool.score)))
    steps = min(max(BUDGETS), n)
    for name, spec, inst in variants:
        path = exp.adaptive_path(spec, inst.human, steps)
        for b in BUDGETS:
            add(
                f"S2x[{name}]",
                b,
                exp.verified(inst, path.chosen[:b])[0],
                ruler=spec.ruler_source,
            )
            tp = exp.strategy_at_budget(inst, path, spec, b, seed)["tp"]
            add(f"S2[{name}]", b, tp, ruler=spec.ruler_source)
    # (c) bandit de Thompson sobre faixas de posição da O1 e da O2
    for tag, arr in (("O1", pool.score), ("O2", pool.nota)):
        chosen = exp.thompson_path(arr, y.copy(), steps, seed=seed + 7)
        for b in BUDGETS:
            add(f"S5[{tag}]", b, found_at(y, chosen, b))
    return rows


# ---------------------------------------------------------------------------
# Ordem do arquivo, reprodução da tese e ponta a ponta
# ---------------------------------------------------------------------------


def reproduction(pool: Pool, draws: int = 300) -> list[dict[str, Any]]:
    """O1/O2 em 96 revisões: arquivo (reprodução) e desempate aleatório."""
    rows: list[dict[str, Any]] = []
    for tag, arr in (("O1", pool.score), ("O2", pool.nota)):
        file_order = rank_order(arr)
        rnd = np.array(
            [
                [found_at(pool.y, rank_order(arr, s), b) for b in BUDGETS]
                for s in range(draws)
            ]
        )
        for j, b in enumerate(BUDGETS):
            rows.append(
                {
                    "strategy": f"{tag}-arquivo",
                    "B": b,
                    "found": float(found_at(pool.y, file_order, b)),
                }
            )
            rows.append(
                {
                    "strategy": f"{tag}-aleatorio",
                    "B": b,
                    "found": float(rnd[:, j].mean()),
                    "min": float(rnd[:, j].min()),
                    "max": float(rnd[:, j].max()),
                    "n": draws,
                }
            )
    return rows


def end_to_end_h0(path: Path, draws: int = 300) -> list[dict[str, Any]]:
    """H0 (humano sozinho) com O1 e O2 no pool ``all`` da execução ponta a ponta.

    Verificados = verdadeiros (PAR em {1, 2}); "abaixo" = os 36 abaixo do corte.
    """
    pool = load_pool(path, "all")
    rows: list[dict[str, Any]] = []
    for tag, arr in (("O1", pool.score), ("O2", pool.nota)):
        for label, tie in (("arquivo", None), ("aleatorio", draws)):
            seeds = [None] if tie is None else list(range(draws))
            res = np.array(
                [
                    [
                        [
                            int(pool.y[rank_order(arr, s)[:b]].sum()),
                            int((pool.y & pool.below)[rank_order(arr, s)[:b]].sum()),
                        ]
                        for b in END_TO_END_BUDGETS
                    ]
                    for s in seeds
                ],
                dtype=float,
            )
            for j, b in enumerate(END_TO_END_BUDGETS):
                for k, metric in enumerate(("verified", "verified_below")):
                    rows.append(
                        {
                            "strategy": f"H0[{tag}-{label}]",
                            "B": b,
                            "metric": metric,
                            "found": float(res[:, j, k].mean()),
                            "min": float(res[:, j, k].min()),
                            "max": float(res[:, j, k].max()),
                            "n": len(seeds),
                        }
                    )
    return rows


def existing_queue_vs_o1(
    sim_csv: Path,
    eng_csv: Path,
    o1_stable: Mapping[int, int],
    o1_mean: Mapping[int, float],
) -> list[dict[str, Any]]:
    """Leitura dos CSVs existentes (fila auto-calibrável) ao lado da curva de O1."""
    rows: list[dict[str, Any]] = []
    d = pd.read_csv(sim_csv)
    d["budget"] = d["budget"].astype(str)
    q = d[
        d["method"].isin(["H0", "session"])
        & d["budget"].isin(["B50", "B100", "B200", "B400"])
        & (d["metric"] == "verified_below")
        & (d["mode"] == "human")
        & (d["variant"] == "default")
        & (d["pool"] == "below_cut")
    ]
    for key, g in q.groupby(
        ["reviewer", "method", "negative_labels", "human_budget", "budget"],
        dropna=False,
    ):
        reviewer, method, neg, hb, budget = key
        rows.append(
            {
                "fonte": "sim_fila_autocalibravel",
                "reviewer": reviewer,
                "strategy": method,
                "negative_labels": neg,
                "human_budget": hb,
                "B": int(str(budget)[1:]),
                "found": float(g["mean"].mean()),
                "n_seeds": float(g["n_seeds"].max()),
            }
        )
    e = pd.read_csv(eng_csv, low_memory=False)
    y = e[(e["table"] == "yield") & (e["cell_id"] == "real_below_cutoff")]
    for r in y.itertuples():
        rows.append(
            {
                "fonte": "engenharia_de_filas",
                "reviewer": "real",
                "strategy": r.strategy,
                "B": int(r.B),
                "found": float(r.mean),
                "n_seeds": float(r.n_seeds),
            }
        )
    for b, v in o1_stable.items():
        rows.append(
            {"fonte": "O1", "strategy": "O1-arquivo", "B": b, "found": float(v)}
        )
    for b, v in o1_mean.items():
        rows.append(
            {"fonte": "O1", "strategy": "O1-aleatorio", "B": b, "found": float(v)}
        )
    return rows


# ---------------------------------------------------------------------------
# Agregação e CLI
# ---------------------------------------------------------------------------


def aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    """Média, mínimo e máximo entre sementes e diferença pareada contra O1."""
    sub = rows[rows["B"] > 0]
    g = sub.groupby(["strategy", "B"])["found"].agg(["mean", "min", "max", "count"])
    g = g.rename(columns={"count": "n"}).reset_index()
    wide = sub.pivot_table(index=["seed", "B"], columns="strategy", values="found")
    ref = wide["O1"]
    diffs = []
    for s in wide.columns:
        d = (wide[s] - ref).groupby(level="B")
        diffs.append(
            pd.DataFrame(
                {
                    "strategy": s,
                    "B": d.mean().index,
                    "diff_vs_O1_mean": d.mean().to_numpy(),
                    "diff_vs_O1_min": d.min().to_numpy(),
                    "diff_vs_O1_max": d.max().to_numpy(),
                }
            )
        )
    return g.merge(pd.concat(diffs), on=["strategy", "B"], how="left")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real-input", type=Path, required=True)
    ap.add_argument(
        "--fixture",
        type=Path,
        default=Path("data/llm_fixtures/haiku-5-5_v4_sem-nota.jsonl"),
    )
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument(
        "--sim-csv",
        type=Path,
        default=Path("docs/sim_fila_autocalibravel_resultados.csv"),
    )
    ap.add_argument(
        "--eng-csv", type=Path, default=Path("docs/engenharia_de_filas_resultados.csv")
    )
    ap.add_argument("--out-csv", type=Path, default=Path("docs/comparacao_o1.csv"))
    args = ap.parse_args(list(sys.argv[1:] if argv is None else argv))
    warnings.simplefilter("ignore")
    pool = load_pool(args.real_input, "below")
    log(
        f"pool: {pool.n} pares, {int(pool.y.sum())} verdadeiros; "
        f"SCORE com {len(np.unique(pool.score))} valores distintos"
    )
    rep = reproduction(pool)
    f96 = {(r["strategy"], r["B"]): r for r in rep}
    o1f, o2f = f96[("O1-arquivo", 96)]["found"], f96[("O2-arquivo", 96)]["found"]
    log(
        f"reprodução em 96 (ordem do arquivo): O1 = {o1f:.0f} (tese {THESIS_O1_96}), "
        f"O2 = {o2f:.0f} (tese {THESIS_O2_96})"
    )
    if (round(o1f), round(o2f)) != (THESIS_O1_96, THESIS_O2_96):
        log("NÃO reproduz a tese: pare e reporte (universo/desempate)")
        return 2
    t0 = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init,
        initargs=(args.real_input, args.fixture),
    ) as ex:
        out = list(ex.map(run_seed, range(args.seeds)))
    log(f"{args.seeds} sementes em {time.perf_counter() - t0:.0f}s")
    rows = pd.DataFrame([r for part in out for r in part])
    agg = aggregate(rows)
    cov = rows[rows["strategy"].str.startswith("cobertura")]
    cov_agg = cov.groupby("strategy")[
        ["window", "covered", "covered_fraction", "positives", "positives_covered"]
    ].agg(["mean", "min", "max"])
    cov_agg.columns = [f"{a}_{b}" for a, b in cov_agg.columns]
    e2e = end_to_end_h0(args.real_input)
    o1_stable = {r["B"]: int(r["found"]) for r in rep if r["strategy"] == "O1-arquivo"}
    o1_mean = {r["B"]: r["found"] for r in rep if r["strategy"] == "O1-aleatorio"}
    existing = existing_queue_vs_o1(args.sim_csv, args.eng_csv, o1_stable, o1_mean)
    parts = [
        pd.DataFrame(rep).assign(table="reproducao"),
        agg.assign(table="comparacao"),
        cov_agg.reset_index().assign(table="cobertura_fixture"),
        pd.DataFrame(e2e).assign(table="ponta_a_ponta_h0"),
        pd.DataFrame(existing).assign(table="fila_existente_vs_o1"),
    ]
    off = [o1_offset(pool, k) for k in ("platt",)][0][2]
    parts.append(
        pd.DataFrame(
            [
                {
                    "strategy": "O1-platt",
                    "metric": "intercepto",
                    "found": off[0] if off else math.nan,
                },
                {
                    "strategy": "O1-platt",
                    "metric": "inclinacao",
                    "found": off[1] if off else math.nan,
                },
            ]
        ).assign(table="transformacao_o1")
    )
    table = pd.concat(parts, ignore_index=True, sort=False)
    first = ["table", "strategy", "B"]
    table = table[first + [c for c in table.columns if c not in first]]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_csv, index=False, float_format="%.6g")
    log(f"escrito {args.out_csv} ({len(table)} linhas)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
