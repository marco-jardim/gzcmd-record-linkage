"""Non-confirmatory robustness analysis of the sequential arrangement (ASR).

Faithful port of the generic parts of ``gzcmd/arranjo_robustez.py`` of the
thesis research repository. The analysis re-trains the ASR (see
``arranjo_corte``: the OpenRecLink cut is kept and the pairs below it are
decided with the v3.1 policy over an isotonic recalibration, the principal arm
``P``) varying one factor at a time, and asks whether ``P-Platt`` (a
*sensitivity analysis*, never the principal arm) keeps weakly dominating ``P``:
block N varies the XGBoost seed and ``n_jobs`` with fixed folds, block P varies
the REFREC fold seed. It does not confirm ``P-Platt``, does not measure
sampling variability and does not cover LLM stochasticity.

The thesis grid (``grid()``), the dominance vector and the classification
thresholds (``classify``) are the defaults; all are parameters. Note that
XGBoost ``hist`` results depend on ``n_jobs`` (the thesis found different
probabilities for ``n_jobs=2`` versus ``8``), which is why ``n_jobs`` is a
studied factor.

Left out on purpose (research-only): ``verify_prereg`` (pins sklearn==1.8.0,
xgboost==3.2.0 and the SHA-256 of thesis files), ``pending_cost`` (fixed token
counts and prices) and ``resolve_input`` (thesis data paths).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.stats import beta

from gzcmd_record_linkage.arranjo_corte import (
    ModelFactory,
    cache_matches,
    nested_cutoff_probabilities,
)
from gzcmd_record_linkage.oof import thesis_classifier

THESIS_MAXIMIZE: tuple[str, ...] = ("aceitos",)
THESIS_MINIMIZE: tuple[str, ...] = (
    "fp_abaixo",
    "perdidos",
    "pares_humano",
    "pares_llm",
)
"""Dominance vector of the thesis: accepted up; FP below the cut, lost pairs,
pairs to the human and pairs to the LLM down."""

THESIS_XGB_SEEDS: tuple[int, ...] = (42, *range(101, 110))
THESIS_N_JOBS: tuple[int, ...] = (8, 2)
THESIS_FOLD_SEEDS: tuple[int, ...] = tuple(range(1000, 1200, 10))


@dataclass(frozen=True)
class Replica:
    replica: str
    bloco: str
    fold_seed: int
    xgb_seed: int
    n_jobs: int


def grid(
    *,
    xgb_seeds: Sequence[int] = THESIS_XGB_SEEDS,
    n_jobs: Sequence[int] = THESIS_N_JOBS,
    fold_seeds: Sequence[int] = THESIS_FOLD_SEEDS,
    base_fold_seed: int = 42,
    base_xgb_seed: int = 42,
    base_n_jobs: int = 8,
) -> list[Replica]:
    """Replicas of the thesis design: block N then block P (one factor at a time).

    Block N (``N00``...): ``base_fold_seed`` with every ``xgb_seeds`` x ``n_jobs``
    pair (seed-major); block P (``P00``...): ``base_xgb_seed`` and ``base_n_jobs``
    with every fold seed of ``fold_seeds``. Defaults give the thesis 20 + 20
    replicas, the first of block N being the reference R0 (42, 42, 8).
    """
    block_n = [
        Replica(f"N{i:02d}", "N", base_fold_seed, seed, jobs)
        for i, (seed, jobs) in enumerate((s, j) for s in xgb_seeds for j in n_jobs)
    ]
    block_p = [
        Replica(f"P{i:02d}", "P", seed, base_xgb_seed, base_n_jobs)
        for i, seed in enumerate(fold_seeds)
    ]
    return block_n + block_p


def replica_factory(xgb_seed: int, n_jobs: int) -> ModelFactory:
    """Model factory that changes only ``random_state`` and ``n_jobs``.

    The ``seed`` that ``nested_cutoff_probabilities`` passes (the fold seed) is
    ignored: the classifier seed is ``xgb_seed`` in every fit.
    """
    return lambda y, _seed: thesis_classifier(y, xgb_seed).set_params(n_jobs=n_jobs)


def validate_dossiers(
    path: str | Path,
    run: Mapping[str, Any],
    mode: str,
    views: Mapping[str, Any],
    *,
    prompt_sha256: str,
    **identity_defaults: Any,
) -> None:
    """Validate the complete cache identity of every result of ``run``.

    ``views[pair_id]`` is a pandas object whose ``to_json(sort_keys=True)``
    hashes to the dossier hash. The journal identity is read from
    ``<stem>_journal/identity.json`` next to ``path`` when present, otherwise
    ``run`` itself is the identity. ``identity_defaults`` (``model``,
    ``removed_fields``, ``prompt_version``) go to ``cache_matches``. Raises
    ``ValueError`` on the first mismatch.
    """
    path = Path(path)
    identity_path = path.with_name(path.stem + "_journal") / "identity.json"
    identity = (
        json.loads(identity_path.read_text(encoding="utf-8"))
        if identity_path.exists()
        else run
    )
    for result in run["results"]:
        pid = result["pair_id"]
        digest = hashlib.sha256(views[pid].to_json(sort_keys=True).encode()).hexdigest()
        if not cache_matches(
            identity,
            run,
            mode=mode,
            pair_id=pid,
            prompt_sha256=prompt_sha256,
            dossier_sha256=digest,
            **identity_defaults,
        ):
            raise ValueError(f"Dossier cache identity mismatch: {mode}/{pid}")


def run_replica(
    x: pd.DataFrame,
    y: ArrayLike,
    below: ArrayLike,
    groups: ArrayLike,
    *,
    par2: ArrayLike | None,
    replica: Replica,
    include_a3: bool = False,
    audit: Any = None,
) -> Any:
    """``nested_cutoff_probabilities`` for one replica (A3 off by default)."""
    return nested_cutoff_probabilities(
        x,
        y,
        below,
        groups,
        par2=par2,
        seed=replica.fold_seed,
        model_factory=replica_factory(replica.xgb_seed, replica.n_jobs),
        include_a3=include_a3,
        audit=audit,
    )


def dominates(
    a: Mapping[str, float],
    b: Mapping[str, float],
    *,
    maximize: Iterable[str] = THESIS_MAXIMIZE,
    minimize: Iterable[str] = THESIS_MINIMIZE,
) -> bool:
    """``a`` weakly Pareto-dominates ``b``: tolerance zero, one strict inequality.

    Every ``maximize`` key is at least as large and every ``minimize`` key at
    least as small in ``a``, and at least one of them strictly better. The
    defaults are the thesis vector (accepted up; FP below the cut, lost pairs,
    pairs to the human and to the LLM down).
    """
    delta = np.array(
        [a[k] - b[k] for k in maximize] + [b[k] - a[k] for k in minimize], dtype=float
    )
    return bool(np.all(delta >= 0) and np.any(delta > 0))


def clopper_pearson(k: int, n: int = 20, alpha: float = 0.05) -> tuple[float, float]:
    """Exact (Clopper-Pearson) two-sided ``1 - alpha`` interval of ``k`` of ``n``."""
    if not 0 <= k <= n or n <= 0:
        raise ValueError("Invalid binomial counts")
    return (
        0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1)),
        1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k)),
    )


def classify(
    counts: Sequence[int],
    inverse: Sequence[int],
    *,
    stable_min: int = 18,
    stable_inverse_max: int = 1,
    unsustained_below: int = 10,
    unsustained_inverse_min: int = 2,
) -> str:
    """Thesis verdict of one mode from the two blocks (N, P), with 20 replicas each.

    ``counts`` are the replicas where P-Platt dominates P, ``inverse`` those
    where P dominates P-Platt (one value per block). ``'estável'``: both
    ``counts >= stable_min`` and both ``inverse <= stable_inverse_max``
    (thesis: >= 18/20 and <= 1/20); ``'não sustentada'``: both
    ``counts < unsustained_below`` or any ``inverse >= unsustained_inverse_min``
    (thesis: < 10/20, >= 2/20); otherwise ``'mista'``. Invalid replicas stay in
    the denominator as "does not dominate".
    """
    if len(counts) != 2 or len(inverse) != 2:
        raise ValueError("Both fixed blocks required")
    if all(k >= stable_min for k in counts) and all(
        k <= stable_inverse_max for k in inverse
    ):
        return "estável"
    if all(k < unsustained_below for k in counts) or any(
        k >= unsustained_inverse_min for k in inverse
    ):
        return "não sustentada"
    return "mista"


def safety_stop(
    n_new: int,
    *,
    cost: float,
    projected_seconds: float,
    max_pairs: int,
    max_cost: float,
    max_seconds: float,
) -> None:
    """Raise ``RuntimeError`` when a new run exceeds any of the given ceilings.

    Fully parameterised (no money or time defaults): stops when ``n_new >
    max_pairs``, ``cost > max_cost`` or ``projected_seconds > max_seconds``.
    """
    if n_new > max_pairs or cost > max_cost or projected_seconds > max_seconds:
        raise RuntimeError(
            f"STOP: pares={n_new}, custo_USD={cost:.8f}, "
            f"projeção_s={projected_seconds:.3f}"
        )


def replace_backoff(source: str | Path, destination: str | Path) -> None:
    """``os.replace`` with exponential backoff (Windows holds files briefly)."""
    for attempt in range(8):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.1 * 2**attempt)


def atomic_json(path: str | Path, value: Any) -> None:
    """Write ``value`` as indented UTF-8 JSON via a temporary file and replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    replace_backoff(temporary, path)


def checkpoint_matches(path: str | Path, identity: Any) -> bool:
    """True when ``<path>.npz`` and ``<path>.json`` exist and the JSON identity matches.

    A replica must be recomputed when the identity (seeds, ``n_jobs``, hashes,
    versions) diverges.
    """
    path = Path(path)
    return (
        path.with_suffix(".npz").exists()
        and path.with_suffix(".json").exists()
        and json.loads(path.with_suffix(".json").read_text(encoding="utf-8")).get(
            "identity"
        )
        == identity
    )


def action_stability(
    current: pd.DataFrame,
    reference: pd.DataFrame,
    below_ids: ArrayLike,
    *,
    tolerance: float = 1e-14,
    review_action: str = "LLM_REVIEW",
) -> dict[str, float | int]:
    """Label-free stability of a replica against the reference over ``below_ids``.

    ``current``/``reference`` have ``pair_id``, ``p_cal`` and ``action``. Returns
    the number of pairs whose ``p_cal`` moved by more than ``tolerance``, the
    maximum movement, the share of changed actions and the Jaccard index of the
    ``review_action`` sets (1 when both are empty).
    """
    current = current.set_index("pair_id").loc[below_ids]
    reference = reference.set_index("pair_id").loc[below_ids]
    a = set(current.index[current.action.eq(review_action)])
    b = set(reference.index[reference.action.eq(review_action)])
    delta = np.abs(current.p_cal.to_numpy() - reference.p_cal.to_numpy())
    return {
        "p_alterados": int((delta > tolerance).sum()),
        "delta_p_max": float(delta.max()),
        "proporcao_acao_alterada": float(current.action.ne(reference.action).mean()),
        "jaccard_llm": len(a & b) / len(a | b) if a | b else 1.0,
    }
