"""Self-calibrating review queue (experimental, no gold standard).

A Thompson-sampling bandit picks which rank band to consume; each band is read
from the top down (decreasing current probability). Bands are quantiles of the
rank of the current probability among the pairs not yet reviewed at the start
of the round. Outside the bandit, an exploration arm reviews a random pair
every ``explore_every`` reviews. An LLM reviewer (two agents plus an arbiter)
judges pair by pair; at the end of each round a gate (human or automatic)
closes a holdout, a multi-field logistic recalibrator is refitted and the
remaining pairs are reordered.

The session is a step-by-step state machine: every :meth:`SelfCalibratingSession.step`
returns exactly one :class:`SessionEvent`. State is JSON-serializable
(:meth:`SelfCalibratingSession.to_state`) and a session restored with
:meth:`SelfCalibratingSession.from_state` continues the identical sequence.

The session never reads ``PAR`` or ``TARGET``. The score (``nota final``) is
used only to compute ``p0`` in round 1 and, in automatic mode, to build the
low-confidence mask when no better source is available.

Experimental and not evaluated in the thesis: ``run_v3`` and every existing
default are untouched.
"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from typing import Any, Literal, Protocol, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import beta as beta_dist

from .arranjo_corte import THESIS_CUTOFFS, below_cutoff
from .calibration import (
    _InsufficientAnchors,
    fit_anchor_platt,
    get_nota_series,
    predict_platt,
    predict_stub,
)
from .fellegi_sunter import (
    FIELD_SETS,
    FellegiSunterError,
    FellegiSunterModel,
    FieldLevels,
    agreement_mask,
    design_matrix,
    discretize_fields,
    field_columns,
    fit_fellegi_sunter,
    resolve_column,
)
from .field_recalibrator import (
    FieldRecalibration,
    RecalibrationPrior,
    fit_field_recalibrator,
    stratum_weight_parts,
    stratum_weights,
)
from .llm_dossier import (
    DOSSIER_SOURCE_COLUMNS,
    DossierView,
    build_dossier,
    make_pair_id,
)
from .llm_review import AgentResponse, LLMCallError, ReviewResult, ReviewVerdict
from .ranking import ThompsonQueue

STATE_VERSION = 1

Mode = Literal["human", "auto"]
Decision = Literal["MATCH", "NONMATCH", "SKIP"]
EventKind = Literal[
    "ordering",
    "pair_presented",
    "agent_verdict",
    "final_verdict",
    "review_error",
    "round_stopped",
    "gate_opened",
    "human_decision",
    "holdout_built",
    "alarm",
    "recalibration",
    "reordering",
    "run_finished",
]

_KEY_COLUMNS = ("COMPREC", "REFREC", "PASSO")
_NOTA_COLUMNS = ("nota final", "nota_final")
_EXTRA_COLUMNS = ("auto_low_confidence",)
# Columns the session itself reads (and fingerprints) besides the ones that
# depend on the data (``SCORE`` with a possible ``",Type"`` suffix, the
# ``field_set`` columns and the dossier source columns; see ``_declared_columns``).
_BASE_COLUMNS: tuple[str, ...] = tuple(
    dict.fromkeys((*_KEY_COLUMNS, *_NOTA_COLUMNS, *_EXTRA_COLUMNS))
)
_NOTA_FIELD = "features.nota_final"
_GREY_LOW, _GREY_HIGH = 5.0, 8.0
_LOW_CONF_LABEL_DEFAULT = 0.16
_HUMAN_VALUES = ("MATCH", "NONMATCH", "SKIP")


class ExperimentalWarning(UserWarning):
    """Emitted when the experimental automatic mode is used."""


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SelfCalibratingConfig:
    """All parameters of a self-calibrating session (defaults from the plan)."""

    mode: Literal["human", "auto"] = "human"
    seed: int = 42
    field_set: Literal["cdr_subscores", "composites"] = "composites"
    tau: float = 0.05
    delta: float = 0.05
    window: int = 100
    band_rank_edges: tuple[float, ...] = (0.01, 0.05, 0.20)
    explore_every: int = 10
    max_round_reviews: int = 500
    max_reviews: int = 2000
    max_rounds: int = 5
    min_new_positives: int = 1
    nonmatch_audit_n: int = 10
    audit_rank_fraction: float = 0.05
    human_budget_per_round: int | None = 50
    human_budget_total: int | None = None
    gate_include_unsure: bool = True
    negative_labels: Literal["llm_nonmatch", "verified_only"] = "llm_nonmatch"
    weighting: Literal["stratum_ipw", "none"] = "stratum_ipw"
    weight_max_ratio: float = 10.0
    min_labeled: int = 20
    min_per_class: int = 3
    sd_coef: float | None = None
    sd_intercept: float | None = None
    trust_coef: float | None = None
    trust_intercept: float | None = None
    auto_low_conf_policy: Literal["exclude", "soft"] = "exclude"
    auto_low_conf_match_label: float = _LOW_CONF_LABEL_DEFAULT
    auto_max_low_conf_fraction: float = 0.5
    anomaly_ratio: float = 3.0
    anomaly_min_matches: int = 5
    max_consecutive_errors: int = 3
    remove_fields: tuple[str, ...] = ("features.nota_final",)
    include_macd: bool = True
    keep_event_log: bool = True

    def __post_init__(self) -> None:
        # numpy scalars (for example ``np.int64`` from a grid) become Python ones
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, np.bool_):
                object.__setattr__(self, f.name, bool(value))
            elif isinstance(value, np.integer):
                object.__setattr__(self, f.name, int(value))
            elif isinstance(value, np.floating):
                object.__setattr__(self, f.name, float(value))
        if self.mode not in ("human", "auto"):
            raise ValueError(f"mode must be 'human' or 'auto', got {self.mode!r}")
        if self.field_set not in FIELD_SETS:
            raise ValueError(
                f"field_set must be 'cdr_subscores' or 'composites', "
                f"got {self.field_set!r}"
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if self.seed < 0:
            raise ValueError("seed must be >= 0")
        for name in ("tau", "delta"):
            if not 0.0 < getattr(self, name) < 1.0:
                raise ValueError(f"{name} must be in (0, 1)")
        for name in (
            "window",
            "max_round_reviews",
            "max_reviews",
            "max_rounds",
            "min_labeled",
            "min_per_class",
            "anomaly_min_matches",
            "max_consecutive_errors",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an integer >= 1")
        for name in ("min_new_positives", "nonmatch_audit_n", "explore_every"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be an integer >= 0")
        needed = zero_match_reviews_needed(self.tau, self.delta)
        if self.window < needed:
            raise ValueError(
                f"window={self.window} is smaller than the {needed} match-free "
                f"reviews needed for P(theta > tau) < delta (tau={self.tau}, "
                f"delta={self.delta}): the posterior stop could never fire"
            )
        edges = tuple(float(e) for e in self.band_rank_edges)
        if not edges or any(not 0.0 < e < 1.0 for e in edges):
            raise ValueError("band_rank_edges must be non-empty values in (0, 1)")
        if any(a >= b for a, b in zip(edges, edges[1:], strict=False)):
            raise ValueError("band_rank_edges must be strictly increasing")
        object.__setattr__(self, "band_rank_edges", edges)
        if not 0.0 <= self.audit_rank_fraction <= 1.0:
            raise ValueError("audit_rank_fraction must be in [0, 1]")
        for name in ("human_budget_per_round", "human_budget_total"):
            budget = getattr(self, name)
            if budget is not None and (
                isinstance(budget, bool) or not isinstance(budget, int) or budget < 1
            ):
                raise ValueError(f"{name} must be an integer > 0 (or None)")
        if not isinstance(self.gate_include_unsure, bool):
            raise ValueError("gate_include_unsure must be a bool")
        if self.negative_labels not in ("llm_nonmatch", "verified_only"):
            raise ValueError(
                "negative_labels must be 'llm_nonmatch' or 'verified_only'"
            )
        if self.weighting not in ("stratum_ipw", "none"):
            raise ValueError("weighting must be 'stratum_ipw' or 'none'")
        if not (math.isfinite(self.weight_max_ratio) and self.weight_max_ratio >= 1.0):
            raise ValueError("weight_max_ratio must be finite and >= 1")
        for name in ("sd_coef", "sd_intercept"):
            value = getattr(self, name)
            if value is not None and not (math.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be finite and > 0 (or None)")
        for name in ("trust_coef", "trust_intercept"):
            value = getattr(self, name)
            if value is not None and not (math.isfinite(value) and value >= 0):
                raise ValueError(f"{name} must be finite and >= 0 (or None)")
        if self.auto_low_conf_policy not in ("exclude", "soft"):
            raise ValueError("auto_low_conf_policy must be 'exclude' or 'soft'")
        if not 0.0 < self.auto_low_conf_match_label < 1.0:
            raise ValueError("auto_low_conf_match_label must be in (0, 1)")
        if not 0.0 < self.auto_max_low_conf_fraction <= 1.0:
            raise ValueError("auto_max_low_conf_fraction must be in (0, 1]")
        if not self.anomaly_ratio > 0:
            raise ValueError("anomaly_ratio must be > 0")
        if isinstance(self.remove_fields, str):
            raise ValueError(
                "remove_fields must be a sequence of dotted paths, not a string: "
                f"use ({self.remove_fields!r},)"
            )
        paths = tuple(self.remove_fields)
        if any(not isinstance(path, str) for path in paths):
            raise ValueError("remove_fields must contain only strings")
        repeated = sorted({path for path in paths if paths.count(path) > 1})
        if repeated:
            raise ValueError(f"remove_fields has duplicate paths: {repeated}")
        object.__setattr__(self, "remove_fields", paths)

    @property
    def effective_sd_coef(self) -> float:
        if self.sd_coef is not None:
            return float(self.sd_coef)
        return 1.0 if self.mode == "human" else 0.5

    @property
    def effective_sd_intercept(self) -> float:
        if self.sd_intercept is not None:
            return float(self.sd_intercept)
        return 2.0 if self.mode == "human" else 1.0

    @property
    def effective_trust_coef(self) -> float | None:
        if self.trust_coef is not None:
            return float(self.trust_coef)
        return None if self.mode == "human" else 1.0

    @property
    def effective_trust_intercept(self) -> float | None:
        if self.trust_intercept is not None:
            return float(self.trust_intercept)
        return None if self.mode == "human" else 2.0

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            out[f.name] = list(value) if isinstance(value, tuple) else value
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SelfCalibratingConfig:
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"Unknown config keys: {unknown}")
        kwargs = dict(data)
        for key in ("band_rank_edges", "remove_fields"):
            if key in kwargs:
                kwargs[key] = tuple(kwargs[key])
        return cls(**kwargs)


# --------------------------------------------------------------------------
# Events and protocols
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionEvent:
    """One step of the session. ``payload`` holds JSON types only."""

    seq: int
    round: int
    kind: EventKind
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "round": self.round,
            "kind": self.kind,
            "payload": self.payload,
        }


class PairReviewer(Protocol):
    """Anything that judges a dossier (LLM, oracle, ...)."""

    def review(self, dossier: DossierView) -> ReviewResult: ...


# --------------------------------------------------------------------------
# Stopping rule
# --------------------------------------------------------------------------


def stop_probability(successes: int, trials: int, tau: float) -> float:
    """``P(theta > tau)`` for ``theta ~ Beta(1 + s, 1 + n - s)`` (exact)."""
    return float(beta_dist.sf(tau, 1 + successes, 1 + trials - successes))


def zero_match_reviews_needed(tau: float, delta: float) -> int:
    """Smallest ``n`` with ``(1 - tau) ** (n + 1) < delta`` (exact).

    That is the minimum number of reviews without MATCH after which
    ``P(theta > tau) < delta`` for ``theta ~ Beta(1, 1 + n)``.
    """
    n = max(0, math.floor(math.log(delta) / math.log1p(-tau)) - 1)
    while (1.0 - tau) ** (n + 1) >= delta:
        n += 1
    while n > 0 and (1.0 - tau) ** n < delta:
        n -= 1
    return n


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _clean(obj: Any) -> Any:
    """Convert numpy/tuple values to plain JSON types (non-finite -> None)."""
    if isinstance(obj, Mapping):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return [_clean(v) for v in obj.tolist()]
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    return obj


def _derive_seed(*parts: int) -> int:
    state = np.random.SeedSequence([int(p) for p in parts]).generate_state(
        2, dtype=np.uint32
    )
    return int(state[0]) | (int(state[1]) << 31)


def _rng(*parts: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(p) for p in parts]))


def _declared_columns(df: pd.DataFrame, field_set: str) -> list[str]:
    """Columns read by the session (and fingerprinted).

    Base columns, ``SCORE``, the ``field_set`` columns and every column the
    dossier is built from. Names are resolved tolerating a ``,Type`` suffix, as
    ``llm_dossier`` does, so the fingerprint covers what the reviewer sees.
    """
    columns = list(df.columns)
    base = [c for c in _BASE_COLUMNS if c in columns]
    fields = field_columns(columns, field_set)
    resolved = (
        resolve_column(columns, name) for name in ("SCORE", *DOSSIER_SOURCE_COLUMNS)
    )
    extra = [c for c in resolved if c is not None]
    return sorted(dict.fromkeys((*base, *fields, *extra)))


def _bool_mask(values: Any, name: str) -> NDArray[np.bool_]:
    """Boolean array from a bool-dtype array/Series (ValueError otherwise)."""
    series = pd.Series(values)
    if not pd.api.types.is_bool_dtype(series.dtype):
        raise ValueError(f"{name} must have dtype bool, got {series.dtype}")
    return series.fillna(False).to_numpy(dtype=bool)


def _fingerprint(df: pd.DataFrame, columns: Sequence[str]) -> str:
    digest = hashlib.sha256()
    digest.update(str(len(df)).encode())
    for column in columns:
        digest.update(b"\x00" + column.encode() + b"\x00")
        hashed = pd.util.hash_pandas_object(df[column], index=False)
        digest.update(hashed.to_numpy().tobytes())
    return digest.hexdigest()


EXPLORE_ARM = "explore"
_EXPLORE_BAND = -1


def _band_label(edges: Sequence[float], band: int) -> str:
    """Label of a rank band (``-1`` is the exploration arm)."""
    if band < 0:
        return EXPLORE_ARM

    def pct(edge: float) -> str:
        return f"{edge * 100:g}%"

    if band == 0:
        return f"rank[0,{pct(edges[0])})"
    if band == len(edges):
        return f"rank[{pct(edges[-1])},100%]"
    return f"rank[{pct(edges[band - 1])},{pct(edges[band])})"


def _rank_bands(n: int, edges: Sequence[float]) -> NDArray[np.int64]:
    """Band of each rank position ``0..n-1`` (position 0 is the highest p).

    Band ``b`` holds the positions below ``ceil(edges[b] * n)`` that are not in
    an earlier band; the last band is the rest.
    """
    cuts = [math.ceil(edge * n - 1e-9) for edge in edges]
    return np.searchsorted(cuts, np.arange(n), side="right").astype(np.int64)


def _logit_mean(p: NDArray[np.float64]) -> float:
    m = float(np.clip(np.mean(p), 1e-6, 1 - 1e-6))
    return math.log(m / (1.0 - m))


# --------------------------------------------------------------------------
# Session
# --------------------------------------------------------------------------


class SelfCalibratingSession:
    """Step-by-step self-calibrating review queue (see module docstring)."""

    def __init__(
        self,
        df: pd.DataFrame,
        reviewer: PairReviewer,
        config: SelfCalibratingConfig | None = None,
        *,
        low_confidence: Sequence[bool] | NDArray[np.bool_] | None = None,
    ) -> None:
        self._cfg = config if config is not None else SelfCalibratingConfig()
        self._reviewer = reviewer
        missing = [c for c in _KEY_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if not any(c in df.columns for c in _NOTA_COLUMNS):
            raise ValueError("Missing score column: 'nota final' or 'nota_final'")
        if len(df) == 0:
            raise ValueError("df is empty")
        present = _declared_columns(df, self._cfg.field_set)
        self._used = df[present].reset_index(drop=True)
        self._source = df.reset_index(drop=True)
        self._n = len(self._used)
        nota = pd.to_numeric(get_nota_series(self._used), errors="coerce")
        if bool(np.isinf(nota.to_numpy(dtype=np.float64)).any()):
            raise ValueError("The score column contains non-finite (inf) values")
        self._pair_ids = [
            make_pair_id(c, r, p)
            for c, r, p in zip(
                self._used["COMPREC"],
                self._used["REFREC"],
                self._used["PASSO"],
                strict=True,
            )
        ]
        if len(set(self._pair_ids)) != self._n:
            raise ValueError("pair_id (COMPREC, REFREC, PASSO) must be unique")
        self._pid_array = np.array(self._pair_ids)
        self._pid_rank = np.empty(self._n, dtype=np.int64)
        self._pid_rank[np.argsort(self._pid_array, kind="stable")] = np.arange(self._n)
        self._levels: FieldLevels = discretize_fields(self._used, self._cfg.field_set)
        self._X, self._feature_names = design_matrix(self._levels)
        self._fingerprint = _fingerprint(self._used, present)

        self._low_conf_arg: NDArray[np.bool_] | None = None
        if low_confidence is not None:
            arg = np.asarray(low_confidence)
            if arg.dtype != np.bool_:
                raise ValueError(
                    f"low_confidence must have dtype bool, got {arg.dtype}"
                )
            if arg.shape != (self._n,):
                raise ValueError("low_confidence must be a bool array of length n")
            self._low_conf_arg = arg
        elif self._cfg.mode == "auto" and "auto_low_confidence" in self._used.columns:
            _bool_mask(self._used["auto_low_confidence"], "auto_low_confidence")

        self._phase = "init"
        self._round = 0
        self._seq = 0
        self._buffer: list[dict[str, Any]] = []
        self._records: list[dict[str, Any]] = []
        self._rec_by_pid: dict[str, dict[str, Any]] = {}
        self._reviewed = np.zeros(self._n, dtype=bool)
        self._queue: ThompsonQueue | None = None
        self._arm_bands: list[int] = []
        self._band_sizes: dict[tuple[int, int], int] = {}
        self._explore_sizes: dict[int, int] = {}
        self._explore_order: NDArray[np.int64] = np.empty(0, dtype=np.int64)
        self._explore_ptr = 0
        self._rank_frac: NDArray[np.float64] | None = None
        self._presented: dict[str, Any] | None = None
        self._p_round: NDArray[np.float64] | None = None
        self._model: FieldRecalibration | None = None
        self._prior: RecalibrationPrior | None = None
        self._fs: FellegiSunterModel | None = None
        self._gate_items: list[dict[str, Any]] = []
        self._holdout: list[dict[str, Any]] = []
        self._low_conf: NDArray[np.bool_] = np.zeros(self._n, dtype=bool)
        self._low_conf_source = "unresolved"
        self._consecutive_errors = 0
        self._round_reviews = 0
        self._round_new_positives = 0
        self._anomaly = False
        self._finish_reason: str | None = None
        self._round_log: list[dict[str, Any]] = []
        self._human_used = 0
        self._gate_stats: dict[str, Any] = {}
        self._events: list[SessionEvent] = []

        self._validate_remove_fields()
        if _NOTA_FIELD not in self._cfg.remove_fields:
            warnings.warn(
                f"{_NOTA_FIELD!r} is not in remove_fields: the score is visible to "
                "the LLM reviewer, a different protocol from the approved one.",
                UserWarning,
                stacklevel=2,
            )
        if self._cfg.mode == "auto":
            warnings.warn(
                "Automatic mode of the self-calibrating queue is experimental and "
                "not evaluated; its output is never a confirmed link.",
                ExperimentalWarning,
                stacklevel=2,
            )

    def _validate_remove_fields(self) -> None:
        """Build a dossier view of the first row so a bad path fails at creation."""
        cfg = self._cfg
        try:
            DossierView(
                build_dossier(
                    self._source.iloc[0],
                    pair_id=self._pair_ids[0],
                    include_macd=cfg.include_macd,
                ),
                remove_model_outputs=True,
                remove_fields=cfg.remove_fields,
            )
        except (KeyError, ValueError) as exc:
            detail = exc.args[0] if exc.args else exc
            raise ValueError(
                f"Invalid remove_fields {list(cfg.remove_fields)}: {detail}. "
                "Paths are relative to the dossier payload; 'model_outputs.*' is "
                "already removed by the session"
            ) from exc

    # ----------------------------------------------------------------------
    # Public properties
    # ----------------------------------------------------------------------

    @property
    def config(self) -> SelfCalibratingConfig:
        return self._cfg

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def finished(self) -> bool:
        return self._phase == "finished" and not self._buffer

    @property
    def awaiting_human(self) -> bool:
        return self._phase == "awaiting_human" and not self._buffer

    @property
    def gate_items(self) -> list[dict[str, Any]]:
        if not self.awaiting_human:
            return []
        return json.loads(json.dumps(self._gate_items))

    @property
    def events(self) -> list[SessionEvent]:
        return list(self._events)

    @property
    def current_model(self) -> FieldRecalibration | None:
        return self._model

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(self._feature_names)

    def peek_upcoming(self, n_per_arm: int) -> dict[str, list[str]]:
        """Next pair_ids each arm would present (read-only; additive API).

        For every active band arm (key = its band label) and for the exploration
        arm (key ``"explore"``, only when ``explore_every > 0``), the next at
        most ``n_per_arm`` pair_ids that arm would present, skipping pairs
        already reviewed or pending. It consumes no randomness and changes no
        state (``to_state()`` is identical before and after): the bandit chooses
        the arm, and the pair that arm presents next is the first element of its
        list. Arms with nothing left are omitted. Outside the ``reviewing``
        phase (before the first round, at the gate, after the end) the result is
        empty. Meant for prefetching reviews in parallel.
        """
        if isinstance(n_per_arm, bool) or not isinstance(n_per_arm, int):
            raise ValueError("n_per_arm must be an integer >= 1")
        if n_per_arm < 1:
            raise ValueError("n_per_arm must be an integer >= 1")
        queue = self._queue
        if self._phase != "reviewing" or queue is None:
            return {}
        # Read-only look at the queue's lists, pointers and seen mask.
        seen = queue._seen
        out: dict[str, list[str]] = {}
        for arm, band in enumerate(self._arm_bands):
            order = queue._orders[arm]
            position = queue._pointers[arm]
            found: list[str] = []
            while position < len(order) and len(found) < n_per_arm:
                item = int(order[position])
                position += 1
                if not seen[item] and not self._reviewed[item]:
                    found.append(self._pair_ids[item])
            if found:
                out[_band_label(self._cfg.band_rank_edges, band)] = found
        if self._cfg.explore_every > 0:
            order = self._explore_order
            position = self._explore_ptr
            pending = None if self._presented is None else int(self._presented["idx"])
            found = []
            while position < order.size and len(found) < n_per_arm:
                item = int(order[position])
                position += 1
                if not self._reviewed[item] and item != pending and not seen[item]:
                    found.append(self._pair_ids[item])
            if found:
                out[EXPLORE_ARM] = found
        return out

    # ----------------------------------------------------------------------
    # Stepping
    # ----------------------------------------------------------------------

    def step(self) -> SessionEvent:
        """Advance the session and return exactly one event."""
        if self.finished:
            raise StopIteration("session finished")
        if not self._buffer:
            if self._phase == "awaiting_human":
                raise RuntimeError(
                    "Session is awaiting human decisions: call "
                    "submit_human_decisions() before step()."
                )
            self._advance()
        return self._pop()

    def _pop(self) -> SessionEvent:
        item = self._buffer.pop(0)
        event = SessionEvent(
            seq=self._seq,
            round=int(item["round"]),
            kind=item["kind"],
            payload=item["payload"],
        )
        self._seq += 1
        if self._cfg.keep_event_log:
            self._events.append(event)
        return event

    def _emit(self, kind: EventKind, payload: Mapping[str, Any]) -> None:
        self._buffer.append(
            {"round": self._round, "kind": kind, "payload": _clean(payload)}
        )

    def _advance(self) -> None:
        phase = self._phase
        if phase == "init":
            self._do_init()
        elif phase == "reviewing":
            self._do_reviewing()
        elif phase == "gate":
            self._do_gate()
        elif phase == "holdout":
            self._do_holdout()
        elif phase == "recalibrate":
            self._do_recalibrate()
        elif phase == "terminate_check":
            self._do_terminate_check()
        else:
            raise RuntimeError(f"Cannot advance from phase {phase!r}")

    # ----------------------------------------------------------------------
    # init
    # ----------------------------------------------------------------------

    def _compute_p0(self) -> tuple[NDArray[np.float64], str, str | None, int]:
        nota = pd.to_numeric(get_nota_series(self._used), errors="coerce")
        n_nan = int(nota.isna().sum())
        nota = nota.fillna(0.0)
        try:
            model = fit_anchor_platt(nota)
        except _InsufficientAnchors as exc:
            p = predict_stub(nota)
            return p.to_numpy(dtype=np.float64), "stub", str(exc), n_nan
        p = predict_platt(nota, model=model)
        return p.to_numpy(dtype=np.float64), "anchor_platt", None, n_nan

    def _resolve_low_confidence(
        self,
    ) -> tuple[NDArray[np.bool_], str, str | None, dict[str, int]]:
        """Mask, source, note and per-source counts of low-confidence pairs.

        Without an explicit mask, automatic mode decides row by row: rows whose
        PASSO is a thesis step and whose SCORE is finite use ``below_cutoff``;
        every other row falls back to the grey band of the score (5 <= nota < 8).
        """
        used = self._used
        n = self._n
        if self._low_conf_arg is not None:
            mask = self._low_conf_arg.copy()
            return mask, "argument", None, {"argument": int(mask.sum())}
        if self._cfg.mode == "human":
            return np.zeros(n, dtype=bool), "not_used_human_mode", None, {}
        if "auto_low_confidence" in used.columns:
            mask = _bool_mask(used["auto_low_confidence"], "auto_low_confidence")
            return mask, "column", None, {"column": int(mask.sum())}
        nota = pd.to_numeric(get_nota_series(used), errors="coerce").fillna(0.0)
        mask = ((nota >= _GREY_LOW) & (nota < _GREY_HIGH)).to_numpy(dtype=bool).copy()
        eligible = np.zeros(n, dtype=bool)
        note: str | None = None
        score_column = resolve_column(list(used.columns), "SCORE")
        if "PASSO" in used.columns and score_column is not None:
            steps = pd.to_numeric(used["PASSO"], errors="coerce")
            # decimal comma accepted, as below_cutoff does ("7,02")
            score = pd.to_numeric(
                used[score_column].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )
            eligible = steps.isin(list(THESIS_CUTOFFS)).to_numpy(
                dtype=bool
            ) & np.isfinite(score.to_numpy(dtype=np.float64))
            if eligible.any():
                try:
                    cut = below_cutoff(
                        pd.DataFrame(
                            {
                                "PASSO": used.loc[eligible, "PASSO"],
                                "SCORE": score[eligible],
                            }
                        )
                    )
                except ValueError as exc:
                    note = f"below_cutoff failed: {exc}"
                    eligible = np.zeros(n, dtype=bool)
                else:
                    mask[eligible] = np.asarray(cut, dtype=bool)
            if not eligible.all():
                note = note or (
                    f"{int((~eligible).sum())} rows without a thesis PASSO or finite "
                    "SCORE use the grey band of the score"
                )
        else:
            note = "PASSO/SCORE not available"
        counts = {
            "below_cutoff": int(mask[eligible].sum()),
            "grey_bands_nota": int(mask[~eligible].sum()),
        }
        if eligible.all():
            source = "below_cutoff"
        elif not eligible.any():
            source = "grey_bands_nota"
        else:
            source = "mixed"
        return mask, source, note, counts

    def _low_conf_fraction(self) -> float:
        return float(self._low_conf.sum()) / self._n

    def _auto_learning_enabled(self) -> bool:
        """False when, in automatic mode with policy ``exclude``, the low-confidence
        region is larger than ``auto_max_low_conf_fraction`` (the run then never
        refits the recalibrator). Derived from the mask, so it survives a resume.
        """
        cfg = self._cfg
        if cfg.mode != "auto" or cfg.auto_low_conf_policy != "exclude":
            return True
        return self._low_conf_fraction() <= cfg.auto_max_low_conf_fraction

    def _make_prior(
        self, mean_intercept: float, mean_coef: Sequence[float]
    ) -> RecalibrationPrior:
        cfg = self._cfg
        k = len(mean_coef)
        return RecalibrationPrior(
            mean_intercept=float(mean_intercept),
            mean_coef=tuple(float(c) for c in mean_coef),
            sd_intercept=cfg.effective_sd_intercept,
            sd_coef=tuple(cfg.effective_sd_coef for _ in range(k)),
            agreement_mask=tuple(bool(b) for b in agreement_mask(self._feature_names)),
            trust_intercept=cfg.effective_trust_intercept,
            trust_coef=cfg.effective_trust_coef,
            feature_names=tuple(self._feature_names),
        )

    def _do_init(self) -> None:
        cfg = self._cfg
        p0, method, p0_note, nota_nan = self._compute_p0()
        fs_info: dict[str, Any]
        try:
            fs = fit_fellegi_sunter(self._levels)
        except FellegiSunterError as exc:
            self._fs = None
            self._prior = self._make_prior(
                _logit_mean(p0), [0.0] * len(self._feature_names)
            )
            fs_info = {"status": "failed", "error": str(exc)}
        else:
            self._fs = fs
            prior = RecalibrationPrior.from_fellegi_sunter(
                fs,
                self._feature_names,
                sd_intercept=cfg.effective_sd_intercept,
                sd_coef=cfg.effective_sd_coef,
                trust_intercept=cfg.effective_trust_intercept,
                trust_coef=cfg.effective_trust_coef,
            )
            self._prior = prior
            fs_info = {
                "status": "fitted",
                "prevalence": fs.prevalence,
                "n_iter": fs.n_iter,
                "converged": fs.converged,
                "log_likelihood": fs.log_likelihood,
            }
        self._model = self._prior.as_recalibration("prior")
        mask, source, mask_note, mask_counts = self._resolve_low_confidence()
        self._low_conf = mask
        self._low_conf_source = source
        self._p_round = p0
        self._round = 1
        band_sizes = self._start_round(p0)
        self._phase = "reviewing"
        self._emit(
            "ordering",
            {
                "mode": cfg.mode,
                "seed": cfg.seed,
                "n_pairs": self._n,
                "field_set": cfg.field_set,
                "p0_method": method,
                "p0_fallback_reason": p0_note,
                "nota_nan_filled": nota_nan,
                "p0_mean": float(np.mean(p0)),
                "feature_names": list(self._feature_names),
                "fellegi_sunter": fs_info,
                "prior_status": "prior",
                "n_nan_by_field": dict(self._levels.n_nan),
                "low_confidence_source": source,
                "low_confidence_note": mask_note,
                "low_confidence_by_source": mask_counts,
                "n_low_confidence": int(mask.sum()),
                "low_conf_fraction": self._low_conf_fraction(),
                "auto_max_low_conf_fraction": cfg.auto_max_low_conf_fraction,
                "auto_learning_enabled": self._auto_learning_enabled(),
                "band_sizes": band_sizes,
                "arm_bands": list(self._arm_bands),
                "explore": {
                    "every": cfg.explore_every,
                    "pool_size": self._explore_sizes[self._round],
                },
            },
        )

    # ----------------------------------------------------------------------
    # Rounds
    # ----------------------------------------------------------------------

    def _round_layout(
        self, p: NDArray[np.float64], members: NDArray[np.int64]
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """``members`` by decreasing ``p`` (ties by pair_id) and the rank fraction.

        The rank fraction of a member is its 0-based position in that order
        divided by the number of members (NaN for every other pair).
        """
        order = members[np.lexsort((self._pid_rank[members], -p[members]))]
        order = order.astype(np.int64)
        fraction = np.full(self._n, np.nan, dtype=np.float64)
        fraction[order] = np.arange(order.size, dtype=np.float64) / order.size
        return order, fraction

    def _set_explore(self, members: NDArray[np.int64]) -> None:
        """Random order of the exploration arm for the current round."""
        by_pid = members[np.argsort(self._pid_rank[members], kind="stable")]
        perm = _rng(self._cfg.seed, self._round, 9000).permutation(by_pid.size)
        self._explore_order = by_pid[perm].astype(np.int64)
        self._explore_ptr = 0

    def _next_explore(self) -> int | None:
        """Next not yet reviewed pair of the exploration order, if any."""
        order = self._explore_order
        while (
            self._explore_ptr < order.size and self._reviewed[order[self._explore_ptr]]
        ):
            self._explore_ptr += 1
        if self._explore_ptr >= order.size:
            return None
        item = int(order[self._explore_ptr])
        self._explore_ptr += 1
        return item

    def _start_round(self, p: NDArray[np.float64]) -> list[dict[str, Any]]:
        cfg = self._cfg
        members = np.flatnonzero(~self._reviewed).astype(np.int64)
        if members.size == 0:
            raise RuntimeError("No unreviewed pairs left to start a round")
        order, fraction = self._round_layout(p, members)
        positions = _rank_bands(order.size, cfg.band_rank_edges)
        orders: list[NDArray[np.int64]] = []
        arm_bands: list[int] = []
        info: list[dict[str, Any]] = []
        for b in range(len(cfg.band_rank_edges) + 1):
            segment = order[positions == b]
            if segment.size == 0:
                continue
            orders.append(segment)
            arm_bands.append(b)
            self._band_sizes[(self._round, b)] = int(segment.size)
            info.append(
                {
                    "arm": len(arm_bands) - 1,
                    "band": b,
                    "label": _band_label(cfg.band_rank_edges, b),
                    "size": int(segment.size),
                }
            )
        self._arm_bands = arm_bands
        self._queue = ThompsonQueue(
            orders,
            window=cfg.window,
            prior=(1, 1),
            seed=_derive_seed(cfg.seed, self._round, 1),
            input_kind="partial_orders",
            n_items=self._n,
        )
        self._p_round = p
        self._rank_frac = fraction
        self._explore_sizes[self._round] = int(members.size)
        self._set_explore(members)
        self._round_reviews = 0
        self._round_new_positives = 0
        return info

    def _stop_statistic(self) -> dict[str, Any]:
        queue = self._queue
        if queue is None:
            raise RuntimeError("No active round")
        stats = queue.arm_stats()
        exhausted = queue.exhausted()
        arms: list[dict[str, Any]] = []
        best: float | None = None
        for arm, ((s, n), done) in enumerate(zip(stats, exhausted, strict=True)):
            p = None if done else stop_probability(s, n, self._cfg.tau)
            if p is not None and (best is None or p > best):
                best = p
            arms.append(
                {
                    "arm": arm,
                    "band": self._arm_bands[arm],
                    "successes": s,
                    "trials": n,
                    "exhausted": bool(done),
                    "p_gt_tau": p,
                }
            )
        return {
            "tau": self._cfg.tau,
            "delta": self._cfg.delta,
            "max_p_gt_tau": best,
            "arms": arms,
        }

    def _round_stop_reason(self) -> str | None:
        cfg = self._cfg
        stat = self._stop_statistic()
        best = stat["max_p_gt_tau"]
        if best is None:
            return "exhausted"
        if best < cfg.delta:
            return "posterior"
        if self._round_reviews >= cfg.max_round_reviews:
            return "max_round_reviews"
        if len(self._records) >= cfg.max_reviews:
            return "max_reviews"
        if self._consecutive_errors >= cfg.max_consecutive_errors:
            return "llm_errors"
        return None

    # ----------------------------------------------------------------------
    # reviewing
    # ----------------------------------------------------------------------

    def _do_reviewing(self) -> None:
        if self._presented is not None:
            self._review_presented()
            return
        reason = self._round_stop_reason()
        if reason is not None:
            self._stop_round(reason)
            return
        queue = self._queue
        p_round = self._p_round
        if queue is None or p_round is None:
            raise RuntimeError("No active round")
        rank_frac = self._rank_frac
        if rank_frac is None:
            raise RuntimeError("No active round")
        cfg = self._cfg
        explored: int | None = None
        if cfg.explore_every > 0 and (len(self._records) + 1) % cfg.explore_every == 0:
            explored = self._next_explore()
        arm: int | str
        if explored is not None:
            # outside the bandit: no reward, no stopping statistic
            item = explored
            queue.mark_seen(item)
            arm, band = EXPLORE_ARM, _EXPLORE_BAND
        else:
            item = queue.next()
            arm = queue.arm_of(item)
            band = self._arm_bands[arm]
        p_cur = float(p_round[item])
        fraction = float(rank_frac[item])
        self._presented = {
            "idx": int(item),
            "pair_id": self._pair_ids[item],
            "arm": arm,
            "band": int(band),
            "p_cur": p_cur,
            "rank_frac": fraction,
        }
        self._emit(
            "pair_presented",
            {
                "pair_id": self._pair_ids[item],
                "arm": arm,
                "band": band,
                "band_label": _band_label(cfg.band_rank_edges, band),
                "p_cur": p_cur,
                "rank_fraction": fraction,
                "round_review": self._round_reviews + 1,
            },
        )

    def _stop_round(self, reason: str) -> None:
        recs = [r for r in self._records if r["round"] == self._round]
        counts = Counter(r["llm_decision"] for r in recs)
        stat = self._stop_statistic()
        self._round_log.append({"round": self._round, "stop_reason": reason})
        self._phase = "gate"
        self._emit(
            "round_stopped",
            {
                "reason": reason,
                "n_reviews": len(recs),
                "n_match": counts.get("MATCH", 0),
                "n_nonmatch": counts.get("NONMATCH", 0),
                "n_unsure": counts.get("UNSURE", 0),
                "n_errors": counts.get("ERROR", 0),
                "stop_statistic": stat,
            },
        )

    def _review_presented(self) -> None:
        pres = self._presented
        queue = self._queue
        if pres is None or queue is None:
            raise RuntimeError("Nothing presented")
        idx = int(pres["idx"])
        pid = str(pres["pair_id"])
        dossier = DossierView(
            build_dossier(
                self._source.iloc[idx],
                pair_id=pid,
                include_macd=self._cfg.include_macd,
            ),
            remove_model_outputs=True,
            remove_fields=self._cfg.remove_fields,
        )
        record: dict[str, Any] = {
            "pair_id": pid,
            "idx": idx,
            "round": self._round,
            "arm": pres["arm"],
            "band": int(pres["band"]),
            "p_cur_at_review": float(pres["p_cur"]),
            "rank_frac": float(pres["rank_frac"]),
            "llm_decision": "ERROR",
            "confidence": None,
            "protocol": None,
            "agent_a_decision": None,
            "agent_b_decision": None,
            "arbiter_decision": None,
            "error": None,
            "human_decision": None,
            "human_decision_source": None,
            "label": None,
            "label_source": None,
            "low_confidence": bool(self._low_conf[idx]),
            "audited": False,
        }
        explore = pres["arm"] == EXPLORE_ARM
        try:
            result = self._reviewer.review(dossier)
        except LLMCallError as exc:
            record["error"] = str(exc)
            self._register(record)
            if not explore:
                queue.update(idx, 0)
            self._consecutive_errors += 1
            self._presented = None
            self._emit(
                "review_error",
                {
                    "pair_id": pid,
                    "error": str(exc),
                    "model": exc.model,
                    "attempts": exc.attempts,
                    "reward": None if explore else 0,
                    "consecutive_errors": self._consecutive_errors,
                    "arm": pres["arm"],
                    "band": pres["band"],
                    "p_cur": pres["p_cur"],
                    "stop_statistic": self._stop_statistic(),
                },
            )
            return
        record.update(
            {
                "llm_decision": result.decision,
                "confidence": float(result.confidence),
                "protocol": result.protocol,
                "agent_a_decision": result.agent_a.decision,
                "agent_b_decision": result.agent_b.decision,
                "arbiter_decision": (
                    result.arbiter.decision if result.arbiter is not None else None
                ),
            }
        )
        self._register(record)
        reward = 1 if result.decision == "MATCH" else 0
        if not explore:
            queue.update(idx, reward)
        self._consecutive_errors = 0
        self._presented = None
        agents: list[tuple[str, AgentResponse]] = [
            ("A", result.agent_a),
            ("B", result.agent_b),
        ]
        if result.arbiter is not None:
            agents.append(("ARBITER", result.arbiter))
        for name, resp in agents:
            self._emit(
                "agent_verdict",
                {
                    "pair_id": pid,
                    "agent": name,
                    "decision": resp.decision,
                    "confidence": resp.confidence,
                    "reason_codes": list(resp.reason_codes),
                },
            )
        self._emit(
            "final_verdict",
            {
                "pair_id": pid,
                "decision": result.decision,
                "confidence": result.confidence,
                "protocol": result.protocol,
                "reward": None if explore else reward,
                "arm": pres["arm"],
                "band": pres["band"],
                "p_cur": pres["p_cur"],
                "reason_codes": list(result.reason_codes),
                "stop_statistic": self._stop_statistic(),
            },
        )

    def _register(self, record: dict[str, Any]) -> None:
        self._records.append(record)
        self._rec_by_pid[record["pair_id"]] = record
        self._reviewed[record["idx"]] = True
        self._round_reviews += 1

    # ----------------------------------------------------------------------
    # gate
    # ----------------------------------------------------------------------

    def _gate_capacity(self, n_candidates: int) -> int:
        """How many candidates fit in this round's human budget."""
        cfg = self._cfg
        capacity = n_candidates
        if cfg.human_budget_per_round is not None:
            capacity = min(capacity, cfg.human_budget_per_round)
        if cfg.human_budget_total is not None:
            capacity = min(capacity, max(0, cfg.human_budget_total - self._human_used))
        return capacity

    def _do_gate(self) -> None:
        cfg = self._cfg
        recs = [r for r in self._records if r["round"] == self._round]
        items: list[dict[str, Any]] = []
        candidates: list[tuple[dict[str, Any], str]] = []
        pool_size = 0
        if cfg.mode == "human":

            def priority(r: Mapping[str, Any]) -> tuple[float, str]:
                return (-float(r["p_cur_at_review"]), str(r["pair_id"]))

            candidates += [
                (r, "match")
                for r in sorted(
                    (r for r in recs if r["llm_decision"] == "MATCH"), key=priority
                )
            ]
            if cfg.gate_include_unsure:
                candidates += [
                    (r, "unsure")
                    for r in sorted(
                        (r for r in recs if r["llm_decision"] == "UNSURE"),
                        key=priority,
                    )
                ]
            pool = sorted(
                (
                    r
                    for r in recs
                    if r["llm_decision"] == "NONMATCH"
                    and r["rank_frac"] < cfg.audit_rank_fraction
                ),
                key=lambda r: r["pair_id"],
            )
            pool_size = len(pool)
            k = min(cfg.nonmatch_audit_n, pool_size)
            if k > 0:
                chosen = _rng(cfg.seed, self._round, 7000).choice(
                    pool_size, size=k, replace=False
                )
                candidates += [
                    (pool[j], "audit") for j in sorted(int(c) for c in chosen)
                ]
            capacity = self._gate_capacity(len(candidates))
            for rec, kind in candidates[:capacity]:
                if kind == "audit":
                    rec["audited"] = True
                items.append(self._gate_item(rec, kind))
            for rec, _ in candidates[capacity:]:
                rec["human_decision"] = None
                rec["human_decision_source"] = "over_budget"
            self._human_used += len(items)
        over = candidates[len(items) :]
        total = cfg.human_budget_total
        self._gate_stats = {
            "n_candidates": len(candidates),
            "n_gate": len(items),
            "n_over_budget": len(over),
            "candidates_by_kind": dict(Counter(kind for _, kind in candidates)),
            "over_budget_by_kind": dict(Counter(kind for _, kind in over)),
            "human_used": self._human_used,
            "human_budget_remaining": (
                None if total is None else max(0, total - self._human_used)
            ),
            "human_budget_per_round": cfg.human_budget_per_round,
        }
        self._gate_items = items
        awaiting = bool(items)
        self._phase = "awaiting_human" if awaiting else "holdout"
        kinds = Counter(it["kind"] for it in items)
        self._emit(
            "gate_opened",
            {
                "mode": cfg.mode,
                "awaiting": awaiting,
                "n_items": len(items),
                "n_match": kinds.get("match", 0),
                "n_unsure": kinds.get("unsure", 0),
                "n_audit": kinds.get("audit", 0),
                "audit_pool_size": pool_size,
                **self._gate_stats,
                "items": items,
            },
        )

    @staticmethod
    def _gate_item(record: Mapping[str, Any], kind: str) -> dict[str, Any]:
        return {
            "pair_id": record["pair_id"],
            "kind": kind,
            "llm_decision": record["llm_decision"],
            "protocol": record["protocol"],
            "p_cur": record["p_cur_at_review"],
            "band": record["band"],
        }

    def submit_human_decisions(self, decisions: Mapping[str, Decision]) -> None:
        """Close the human gate: every item needs a decision.

        Values are ``"MATCH"``, ``"NONMATCH"`` or ``"SKIP"`` (the item stays
        undecided and out of the holdout). There is no implicit confirmation:
        an item without an entry raises ``ValueError``.
        """
        if not self.awaiting_human:
            raise RuntimeError("Session is not awaiting human decisions")
        ids = [it["pair_id"] for it in self._gate_items]
        unknown = sorted(set(decisions) - set(ids))
        if unknown:
            raise ValueError(f"pair_ids not in the gate: {unknown}")
        bad = sorted(
            f"{k}={v!r}" for k, v in decisions.items() if v not in _HUMAN_VALUES
        )
        if bad:
            raise ValueError(f"Decisions must be 'MATCH', 'NONMATCH' or 'SKIP': {bad}")
        missing = [pid for pid in ids if pid not in decisions]
        if missing:
            raise ValueError(
                f"{len(missing)} of {len(ids)} gate items have no decision "
                f"(every item needs 'MATCH', 'NONMATCH' or 'SKIP'), "
                f"e.g. {missing[:5]}"
            )
        events: list[dict[str, Any]] = []
        for item in self._gate_items:
            record = self._rec_by_pid[item["pair_id"]]
            pid = item["pair_id"]
            choice = decisions[pid]
            human: str | None
            if choice == "SKIP":
                human, source = None, "human_skip"
            else:
                human, source = choice, "human"
            record["human_decision"] = human
            record["human_decision_source"] = source
            events.append(
                {
                    "pair_id": pid,
                    "item_kind": item["kind"],
                    "llm_decision": record["llm_decision"],
                    "human_decision": human,
                    "source": source,
                    "changed": human is not None and human != record["llm_decision"],
                }
            )
        self._gate_items = []
        self._phase = "holdout"
        for payload in events:
            self._emit("human_decision", payload)

    # ----------------------------------------------------------------------
    # holdout
    # ----------------------------------------------------------------------

    def _label(self, rec: Mapping[str, Any]) -> tuple[float | None, str]:
        cfg = self._cfg
        human = rec["human_decision"]
        if human is not None:
            return (1.0 if human == "MATCH" else 0.0), str(rec["human_decision_source"])
        llm = rec["llm_decision"]
        if llm == "ERROR":
            return None, "excluded_error"
        if cfg.mode == "human":
            # a positive label needs a human MATCH (handled above)
            source = rec["human_decision_source"]
            if source in ("human_skip", "over_budget"):
                return None, f"excluded_{source}"
            if llm == "UNSURE":
                return None, "excluded_unsure"
            if llm == "NONMATCH":
                if cfg.negative_labels == "llm_nonmatch":
                    return 0.0, "llm_nonmatch"
                return None, "excluded_unverified_nonmatch"
            return None, "excluded_unconfirmed_match"
        if llm == "UNSURE":
            return None, "excluded_unsure"
        if rec["low_confidence"]:
            if cfg.auto_low_conf_policy == "soft":
                value = cfg.auto_low_conf_match_label if llm == "MATCH" else 0.0
                return value, "auto_low_conf_soft"
            return None, "excluded_low_conf"
        if llm == "MATCH":
            if rec["protocol"] == "arbiter":
                return None, "excluded_arbiter"
            return 1.0, "auto_consensus_match"
        return 0.0, "auto_nonmatch"

    @staticmethod
    def _stratum(rec: Mapping[str, Any]) -> str:
        """``r{round}:top`` (bandit reviews) or ``r{round}:explore``."""
        kind = "explore" if rec["arm"] == EXPLORE_ARM else "top"
        return f"r{rec['round']}:{kind}"

    def _weights(
        self, members: Sequence[Mapping[str, Any]]
    ) -> tuple[NDArray[np.float64], int]:
        """Holdout weights (mean 1) and how many of them hit the ratio cap.

        Per round there are two strata. ``top`` (the bandit reviews, each band
        read from the top down) has size equal to its number of reviews, so its
        weight is 1: the consumed prefix is treated as a census (pi = 1).
        ``explore`` (the random arm) has size ``N``, the pairs not yet reviewed
        at the start of the round, so its weight is ``N / n_e`` (every
        unreviewed pair has a known inclusion probability ``n_e / N``).
        ``n_h`` counts every reviewed pair of the stratum (labelled or not).
        Weights are capped at ``weight_max_ratio * min_h w_h`` and restricted
        to ``members``: the cap trades bias for variance.
        """
        cfg = self._cfg
        if not members:
            return np.empty(0, dtype=np.float64), 0
        if cfg.weighting == "none":
            return np.ones(len(members), dtype=np.float64), 0
        strata = [self._stratum(r) for r in self._records]
        counts = Counter(strata)
        sizes: dict[str, float] = {}
        for rec, stratum in zip(self._records, strata, strict=True):
            if stratum not in sizes:
                sizes[stratum] = (
                    float(self._explore_sizes[int(rec["round"])])
                    if rec["arm"] == EXPLORE_ARM
                    else float(counts[stratum])
                )
        ids = {r["pair_id"] for r in members}
        keep = np.array([r["pair_id"] in ids for r in self._records], dtype=bool)
        weights = stratum_weights(
            strata, sizes, max_ratio=cfg.weight_max_ratio, mask=keep
        )
        raw, capped = stratum_weight_parts(
            strata, sizes, max_ratio=cfg.weight_max_ratio
        )
        n_capped = int((raw[keep] > capped[keep]).sum())
        return weights, n_capped

    def _relabel(self) -> list[dict[str, Any]]:
        for rec in self._records:
            label, source = self._label(rec)
            rec["label"] = label
            rec["label_source"] = source
        return [r for r in self._records if r["label"] is not None]

    def holdout(self) -> pd.DataFrame:
        """Holdout of the last ``holdout_built``: label, weight and stratum."""
        return pd.DataFrame(
            self._holdout,
            columns=[
                "pair_id",
                "round",
                "band",
                "stratum",
                "y",
                "label_source",
                "weight",
            ],
        )

    def _do_holdout(self) -> None:
        cfg = self._cfg
        members = self._relabel()
        weights, n_capped = self._weights(members)
        self._holdout = [
            {
                "pair_id": r["pair_id"],
                "round": r["round"],
                "band": r["band"],
                "stratum": self._stratum(r),
                "y": r["label"],
                "label_source": r["label_source"],
                "weight": float(w),
            }
            for r, w in zip(members, weights, strict=True)
        ]
        recs = [r for r in self._records if r["round"] == self._round]
        self._round_new_positives = sum(1 for r in recs if r["label"] == 1.0)
        sources = Counter(r["label_source"] for r in self._records)
        y = np.asarray([r["label"] for r in members], dtype=np.float64)
        fractional = (y > 0.0) & (y < 1.0)
        effective_n = (
            float(weights.sum() ** 2 / np.square(weights).sum())
            if weights.size
            else 0.0
        )
        self._emit(
            "holdout_built",
            {
                "n_holdout": len(members),
                "n_label_1": int((y == 1.0).sum()),
                "n_label_0": int((y == 0.0).sum()),
                "n_fractional": int(fractional.sum()),
                "new_positives": self._round_new_positives,
                "label_sources": dict(sources),
                "weighting": cfg.weighting,
                "weight_mean": float(weights.mean()) if weights.size else None,
                "weight_min": float(weights.min()) if weights.size else None,
                "weight_max": float(weights.max()) if weights.size else None,
                "effective_n": effective_n,
                "n_capped": n_capped,
                "n_candidates": self._gate_stats.get("n_candidates", 0),
                "n_gate": self._gate_stats.get("n_gate", 0),
                "n_over_budget": self._gate_stats.get("n_over_budget", 0),
                "human_budget_remaining": self._gate_stats.get(
                    "human_budget_remaining"
                ),
                "label_sources_round": dict(Counter(r["label_source"] for r in recs)),
                "weight_sum_by_class": {
                    "1": float(weights[y == 1.0].sum()),
                    "0": float(weights[y == 0.0].sum()),
                    "fractional": float(weights[fractional].sum()),
                },
            },
        )
        if self._round == 1 and not self._auto_learning_enabled():
            self._emit(
                "alarm",
                {
                    "type": "low_confidence_dominant",
                    "reason": "low_confidence_dominant",
                    "low_conf_fraction": self._low_conf_fraction(),
                    "auto_max_low_conf_fraction": cfg.auto_max_low_conf_fraction,
                    "action": "learning_disabled",
                },
            )
        audited = [r for r in recs if r["audited"] and r["human_decision"] is not None]
        n_miss = sum(1 for r in audited if r["human_decision"] == "MATCH")
        if n_miss >= 1:
            self._emit(
                "alarm",
                {
                    "type": "audit_match_found",
                    "n_audited": len(audited),
                    "n_matches_found": n_miss,
                    "rate": n_miss / len(audited),
                    "note": (
                        "sanity check only: the audit sample is too small to "
                        "estimate the miss rate (no statistical power); this "
                        "alarm is not an estimator"
                    ),
                },
            )
        if cfg.mode == "auto":
            flagged = self._anomalous_bands(recs)
            if flagged:
                self._anomaly = True
                self._emit(
                    "alarm",
                    {
                        "type": "auto_anomaly",
                        "bands": flagged,
                        "action": "revert_model_and_stop",
                    },
                )
        self._phase = "recalibrate"

    def _anomalous_bands(
        self, recs: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        cfg = self._cfg
        flagged: list[dict[str, Any]] = []
        for band in sorted({r["band"] for r in recs}):
            in_band = [r for r in recs if r["band"] == band]
            matches = sum(1 for r in in_band if r["llm_decision"] == "MATCH")
            expected = float(sum(r["p_cur_at_review"] for r in in_band))
            if matches >= cfg.anomaly_min_matches and matches > (
                cfg.anomaly_ratio * expected
            ):
                flagged.append(
                    {
                        "band": band,
                        "matches": matches,
                        "expected": expected,
                        "n_reviewed": len(in_band),
                    }
                )
        return flagged

    # ----------------------------------------------------------------------
    # recalibrate / terminate
    # ----------------------------------------------------------------------

    def _do_recalibrate(self) -> None:
        cfg = self._cfg
        model, prior = self._model, self._prior
        if model is None or prior is None:
            raise RuntimeError("Recalibrator not initialised")
        previous = model.status
        if not self._auto_learning_enabled():
            self._phase = "terminate_check"
            self._emit(
                "recalibration",
                {
                    "status": "skipped_low_conf_dominant",
                    "previous_status": previous,
                    "model_status": model.status,
                    "n_labeled": model.n_labeled,
                    "n_pos": model.n_pos,
                    "n_neg": model.n_neg,
                    "intercept": model.intercept,
                    "coef": list(model.coef),
                    "message": "low-confidence region dominant: model not refitted",
                    "reverted": self._anomaly,
                    "low_conf_fraction": self._low_conf_fraction(),
                    "auto_max_low_conf_fraction": cfg.auto_max_low_conf_fraction,
                },
            )
            return
        members = [r for r in self._records if r["label"] is not None]
        if self._anomaly:
            new = replace(model, message="automatic anomaly: previous model kept")
        elif not members:
            new = replace(
                model,
                status="skipped_insufficient",
                n_labeled=0,
                n_pos=0,
                n_neg=0,
                message="empty holdout",
            )
        else:
            idx = np.asarray([r["idx"] for r in members], dtype=np.int64)
            weight_by_pid = {row["pair_id"]: row["weight"] for row in self._holdout}
            new = fit_field_recalibrator(
                self._X[idx],
                np.asarray([r["label"] for r in members], dtype=np.float64),
                prior,
                weights=np.asarray(
                    [weight_by_pid[r["pair_id"]] for r in members], dtype=np.float64
                ),
                min_labeled=cfg.min_labeled,
                min_per_class=cfg.min_per_class,
                current=model,
            )
        self._model = new
        self._phase = "terminate_check"
        self._emit(
            "recalibration",
            {
                "status": new.status,
                "previous_status": previous,
                "n_labeled": new.n_labeled,
                "n_pos": new.n_pos,
                "n_neg": new.n_neg,
                "intercept": new.intercept,
                "coef": list(new.coef),
                "message": new.message,
                "reverted": self._anomaly,
            },
        )

    def _finish_reason_now(self) -> str | None:
        cfg = self._cfg
        if self._anomaly:
            return "auto_anomaly"
        if self._consecutive_errors >= cfg.max_consecutive_errors:
            return "llm_errors"
        if len(self._records) >= cfg.max_reviews:
            return "max_reviews"
        if (
            cfg.human_budget_total is not None
            and self._human_used >= cfg.human_budget_total
        ):
            return "human_budget"
        if bool(self._reviewed.all()):
            return "exhausted"
        if self._round >= cfg.max_rounds:
            return "max_rounds"
        if self._round_new_positives < cfg.min_new_positives:
            return "sterile_round"
        return None

    def _do_terminate_check(self) -> None:
        reason = self._finish_reason_now()
        if reason is not None:
            self._finish_reason = reason
            self._phase = "finished"
            self._emit(
                "run_finished",
                {"reason": reason, "summary": self.summary()},
            )
            return
        model = self._model
        if model is None:
            raise RuntimeError("Model not initialised")
        finished_round = self._round
        self._round += 1
        p = model.predict_proba(self._X)
        info = self._start_round(p)
        self._phase = "reviewing"
        self._emit(
            "reordering",
            {
                "previous_round": finished_round,
                "model_status": model.status,
                "n_remaining": int((~self._reviewed).sum()),
                "band_sizes": info,
                "explore_pool_size": self._explore_sizes[self._round],
                "p_source": "field_model",
            },
        )

    # ----------------------------------------------------------------------
    # Outputs
    # ----------------------------------------------------------------------

    def results(self) -> pd.DataFrame:
        """One row per reviewed pair (``llm_decision`` is never overwritten)."""
        columns = [
            "pair_id",
            "COMPREC",
            "REFREC",
            "PASSO",
            "round",
            "arm",
            "band",
            "band_index",
            "p_cur_at_review",
            "rank_fraction_at_review",
            "llm_decision",
            "confidence",
            "protocol",
            "agent_a_decision",
            "agent_b_decision",
            "arbiter_decision",
            "human_decision",
            "human_decision_source",
            "label",
            "label_source",
            "holdout_weight",
            "low_confidence",
            "audited",
            "llm_match_unverified",
            "error",
        ]
        weight = {row["pair_id"]: row["weight"] for row in self._holdout}
        rows: list[dict[str, Any]] = []
        for r in self._records:
            idx = r["idx"]
            rows.append(
                {
                    "pair_id": r["pair_id"],
                    "COMPREC": self._used["COMPREC"].iloc[idx],
                    "REFREC": self._used["REFREC"].iloc[idx],
                    "PASSO": self._used["PASSO"].iloc[idx],
                    "round": r["round"],
                    "arm": r["arm"],
                    "band": _band_label(self._cfg.band_rank_edges, r["band"]),
                    "band_index": r["band"],
                    "p_cur_at_review": r["p_cur_at_review"],
                    "rank_fraction_at_review": r["rank_frac"],
                    "llm_decision": r["llm_decision"],
                    "confidence": r["confidence"],
                    "protocol": r["protocol"],
                    "agent_a_decision": r["agent_a_decision"],
                    "agent_b_decision": r["agent_b_decision"],
                    "arbiter_decision": r["arbiter_decision"],
                    "human_decision": r["human_decision"],
                    "human_decision_source": r["human_decision_source"],
                    "label": r["label"],
                    "label_source": r["label_source"],
                    "holdout_weight": weight.get(r["pair_id"]),
                    "low_confidence": r["low_confidence"],
                    "audited": r["audited"],
                    "llm_match_unverified": (
                        r["llm_decision"] == "MATCH"
                        and r["human_decision_source"] != "human"
                    ),
                    "error": r["error"],
                }
            )
        out = pd.DataFrame(rows, columns=columns)
        out["label"] = pd.to_numeric(out["label"], errors="coerce")
        out["holdout_weight"] = pd.to_numeric(out["holdout_weight"], errors="coerce")
        return out

    def summary(self) -> dict[str, Any]:
        counts = Counter(r["llm_decision"] for r in self._records)
        members = [r for r in self._records if r["label"] is not None]
        audited = [r for r in self._records if r["audited"]]
        model = self._model
        return _clean(
            {
                "mode": self._cfg.mode,
                "phase": self._phase,
                "finished": self._phase == "finished",
                "finish_reason": self._finish_reason,
                "round": self._round,
                "n_events": self._seq,
                "n_reviewed": len(self._records),
                "llm_counts": {
                    k: counts.get(k, 0)
                    for k in ("MATCH", "NONMATCH", "UNSURE", "ERROR")
                },
                "n_audited": len(audited),
                "n_audit_matches": sum(
                    1 for r in audited if r["human_decision"] == "MATCH"
                ),
                "n_holdout": len(members),
                "n_holdout_pos": sum(1 for r in members if r["label"] == 1.0),
                "n_holdout_neg": sum(1 for r in members if r["label"] == 0.0),
                "n_llm_match_unverified": sum(
                    1
                    for r in self._records
                    if r["llm_decision"] == "MATCH"
                    and r["human_decision_source"] != "human"
                ),
                "human_used": self._human_used,
                "human_budget_total": self._cfg.human_budget_total,
                "model_status": None if model is None else model.status,
                "low_confidence_source": self._low_conf_source,
                "n_low_confidence": int(self._low_conf.sum()),
                "low_conf_fraction": self._low_conf_fraction(),
                "auto_learning_enabled": self._auto_learning_enabled(),
                "band_sizes": [
                    {"round": r, "band": b, "size": n}
                    for (r, b), n in sorted(self._band_sizes.items())
                ],
                "explore_sizes": [
                    {"round": r, "size": n}
                    for r, n in sorted(self._explore_sizes.items())
                ],
                "rounds": list(self._round_log),
            }
        )

    # ----------------------------------------------------------------------
    # State
    # ----------------------------------------------------------------------

    def to_state(self) -> dict[str, Any]:
        """JSON-compatible snapshot (the event log is not included)."""
        low_conf: dict[str, Any] | None
        if self._phase == "init":
            arg = self._low_conf_arg
            low_conf = (
                None
                if arg is None
                else {"source": "argument", "indices": np.flatnonzero(arg).tolist()}
            )
        else:
            low_conf = {
                "source": self._low_conf_source,
                "indices": np.flatnonzero(self._low_conf).tolist(),
            }
        state: dict[str, Any] = {
            "version": STATE_VERSION,
            "config": self._cfg.to_dict(),
            "fingerprint": self._fingerprint,
            "phase": self._phase,
            "round": self._round,
            "seq": self._seq,
            "buffer": self._buffer,
            "records": self._records,
            "presented": self._presented,
            "bandit": None if self._queue is None else self._queue.to_state(),
            "arm_bands": self._arm_bands,
            "band_sizes": [[r, b, n] for (r, b), n in sorted(self._band_sizes.items())],
            "explore_sizes": [[r, n] for r, n in sorted(self._explore_sizes.items())],
            "model": None if self._model is None else self._model.to_dict(),
            "prior": (
                None
                if self._prior is None
                else {
                    "mean_intercept": self._prior.mean_intercept,
                    "mean_coef": list(self._prior.mean_coef),
                }
            ),
            "fs_model": None if self._fs is None else self._fs.to_dict(),
            "gate_items": self._gate_items,
            "holdout": self._holdout,
            "low_confidence": low_conf,
            "consecutive_errors": self._consecutive_errors,
            "round_new_positives": self._round_new_positives,
            "anomaly": self._anomaly,
            "finish_reason": self._finish_reason,
            "round_log": self._round_log,
            "human_used": self._human_used,
            "gate_stats": self._gate_stats,
        }
        return json.loads(json.dumps(state))

    @classmethod
    def from_state(
        cls,
        state: Mapping[str, Any],
        df: pd.DataFrame,
        reviewer: PairReviewer,
    ) -> SelfCalibratingSession:
        """Restore a session; the DataFrame fingerprint must match."""
        if state.get("version") != STATE_VERSION:
            raise ValueError(f"Unsupported state version {state.get('version')!r}")
        cfg = SelfCalibratingConfig.from_dict(state["config"])
        columns = _declared_columns(df, cfg.field_set)
        if (
            _fingerprint(df[columns].reset_index(drop=True), columns)
            != state["fingerprint"]
        ):
            raise ValueError("DataFrame does not match the saved session state")
        low = state.get("low_confidence")
        arg: NDArray[np.bool_] | None = None
        if state["phase"] == "init" and low is not None:
            arg = np.zeros(len(df), dtype=bool)
            arg[np.asarray(low["indices"], dtype=np.int64)] = True
        session = cls(df, reviewer, cfg, low_confidence=arg)
        data = json.loads(json.dumps(state))
        session._phase = data["phase"]
        session._round = int(data["round"])
        session._seq = int(data["seq"])
        session._buffer = data["buffer"]
        session._records = data["records"]
        session._rec_by_pid = {r["pair_id"]: r for r in session._records}
        session._reviewed = np.zeros(session._n, dtype=bool)
        for r in session._records:
            session._reviewed[r["idx"]] = True
        session._round_reviews = sum(
            1 for r in session._records if r["round"] == session._round
        )
        session._presented = data["presented"]
        if data["bandit"] is not None:
            session._queue = ThompsonQueue.from_state(data["bandit"])
        session._arm_bands = [int(b) for b in data["arm_bands"]]
        session._band_sizes = {
            (int(r), int(b)): int(n) for r, b, n in data["band_sizes"]
        }
        session._explore_sizes = {
            int(r): int(n) for r, n in data.get("explore_sizes", [])
        }
        if data["model"] is not None:
            session._model = FieldRecalibration.from_dict(data["model"])
        if data["prior"] is not None:
            session._prior = session._make_prior(
                data["prior"]["mean_intercept"], data["prior"]["mean_coef"]
            )
        if data["fs_model"] is not None:
            session._fs = FellegiSunterModel.from_dict(data["fs_model"])
        session._gate_items = data["gate_items"]
        session._holdout = data.get("holdout", [])
        if session._phase != "init" and low is not None:
            session._low_conf = np.zeros(session._n, dtype=bool)
            session._low_conf[np.asarray(low["indices"], dtype=np.int64)] = True
            session._low_conf_source = str(low["source"])
        session._consecutive_errors = int(data["consecutive_errors"])
        session._round_new_positives = int(data["round_new_positives"])
        session._anomaly = bool(data["anomaly"])
        session._finish_reason = data["finish_reason"]
        session._round_log = data["round_log"]
        session._human_used = int(data.get("human_used", 0))
        session._gate_stats = dict(data.get("gate_stats", {}))
        if session._phase == "reviewing":
            p_round: NDArray[np.float64] | None = None
            if session._round == 1:
                p_round = session._compute_p0()[0]
            elif session._model is not None:
                p_round = session._model.predict_proba(session._X)
            session._p_round = p_round
            if p_round is not None:
                # The round layout (rank fractions, exploration order) is a pure
                # function of the round-start probabilities and reviewed set.
                before = np.zeros(session._n, dtype=bool)
                for r in session._records:
                    if r["round"] < session._round:
                        before[r["idx"]] = True
                members = np.flatnonzero(~before).astype(np.int64)
                session._rank_frac = session._round_layout(p_round, members)[1]
                session._set_explore(members)
        return session


# --------------------------------------------------------------------------
# Oracle reviewer (tests and simulations, no network)
# --------------------------------------------------------------------------

_DECISIONS = ("MATCH", "NONMATCH", "UNSURE")


class OracleReviewer:
    """Deterministic simulated reviewer driven by a truth vector.

    Errors depend only on ``(seed, pair_id)``, so the order of calls never
    matters. ``fp_rate``/``fn_rate`` flip the shared judgement,
    ``unsure_rate`` turns it into UNSURE and ``disagreement_rate`` makes agent B
    disagree with A, which sends the pair to the arbiter (whose answer is the
    shared judgement). Pairs in ``error_pairs`` raise :class:`LLMCallError`.
    """

    def __init__(
        self,
        truth: Mapping[str, bool],
        *,
        fp_rate: float = 0.0,
        fn_rate: float = 0.0,
        unsure_rate: float = 0.0,
        disagreement_rate: float = 0.0,
        error_pairs: frozenset[str] = frozenset(),
        seed: int = 0,
    ) -> None:
        for name, value in (
            ("fp_rate", fp_rate),
            ("fn_rate", fn_rate),
            ("unsure_rate", unsure_rate),
            ("disagreement_rate", disagreement_rate),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        self._truth = truth
        self._fp = fp_rate
        self._fn = fn_rate
        self._unsure = unsure_rate
        self._disagree = disagreement_rate
        self._error_pairs = frozenset(error_pairs)
        self._seed = seed
        self.seen_payloads: list[dict[str, Any]] = []

    @staticmethod
    def _agent(decision: str) -> AgentResponse:
        return AgentResponse(
            decision=cast(ReviewVerdict, decision),
            confidence=0.9,
            reason_codes=[],
            evidence_summary={},
            quality_flags={},
            raw_json={},
            latency_s=0.0,
            tokens_prompt=0,
            tokens_completion=0,
        )

    def review(self, dossier: DossierView) -> ReviewResult:
        pair_id = dossier.pair_id
        self.seen_payloads.append(dossier.to_dict())
        if pair_id in self._error_pairs:
            raise LLMCallError(
                f"oracle error for {pair_id}",
                model="oracle",
                pair_id=pair_id,
                attempts=1,
            )
        is_match = bool(self._truth[pair_id])
        key = int(hashlib.sha256(pair_id.encode()).hexdigest()[:16], 16)
        u = np.random.default_rng(np.random.SeedSequence([self._seed, key])).random(4)
        if is_match:
            base = "NONMATCH" if u[0] < self._fn else "MATCH"
        else:
            base = "MATCH" if u[0] < self._fp else "NONMATCH"
        if u[1] < self._unsure:
            base = "UNSURE"
        other = base
        if u[2] < self._disagree:
            alternatives = [d for d in _DECISIONS[:2] if d != base]
            other = alternatives[0]
        agent_a = self._agent(base)
        agent_b = self._agent(other)
        arbiter = None if other == base else self._agent(base)
        return ReviewResult(
            pair_id=pair_id,
            decision=cast(ReviewVerdict, base),
            confidence=0.9,
            reason_codes=[],
            evidence_summary={},
            quality_flags={},
            protocol="consensus" if arbiter is None else "arbiter",
            agent_a=agent_a,
            agent_b=agent_b,
            arbiter=arbiter,
        )


# --------------------------------------------------------------------------
# Convenience loop
# --------------------------------------------------------------------------


def run_self_calibrating(
    df: pd.DataFrame,
    reviewer: PairReviewer,
    config: SelfCalibratingConfig | None = None,
    *,
    human_callback: Callable[[list[dict[str, Any]]], Mapping[str, Decision]]
    | None = None,
    low_confidence: Sequence[bool] | NDArray[np.bool_] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], list[SessionEvent]]:
    """Run a session to the end; human mode needs ``human_callback``.

    ``human_callback(gate_items)`` returns a mapping ``pair_id -> decision``.
    """
    cfg = config if config is not None else SelfCalibratingConfig()
    if cfg.mode == "human" and human_callback is None:
        raise ValueError("human mode requires a human_callback")
    session = SelfCalibratingSession(df, reviewer, cfg, low_confidence=low_confidence)
    while not session.finished:
        if session.awaiting_human:
            session.submit_human_decisions(human_callback(session.gate_items))
        else:
            session.step()
    return session.results(), session.summary(), session.events
