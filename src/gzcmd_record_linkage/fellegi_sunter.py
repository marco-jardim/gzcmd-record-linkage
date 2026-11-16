"""Field discretization and unsupervised Fellegi-Sunter model fitted by EM.

Two field sets are supported (``field_set``):

* ``"composites"``: the five comparison composites produced by the loader, all
  in [0, 1]; each is discretized into a few fixed ordinal agreement levels.
* ``"cdr_subscores"`` (**experimental**; not the default): the 29 raw CdR
  sub-scores (name, mother's name, birth date, municipality and address
  fragments) plus the binary ``mae_missing``. On the thesis extract the unsupervised
  fit is not valid as a prior (prevalence about 0.49), see the limitation below.
  Levels come from a label-free rule computed on the whole frame: ``L0`` is
  ``value <= 0`` or NaN; the positive values get one level per distinct value
  when there are at most three of them, otherwise the terciles of the positive
  values. A column that has only negative values (the "common fragments"
  penalties) applies the same rule to the magnitudes, with levels named
  ``P1..P3``; those penalty levels carry no sign constraint. The cut points
  are stored on the :class:`FieldSpec` and serialized with the model.

In both cases a latent-class model with conditional independence is fitted on
the unique level patterns without any label (``L0`` is the reference level of
the one-hot design). The fitted model has an exact logistic representation over
the one-hot level indicators, which seeds the prior of the supervised
recalibrator in ``field_recalibrator``.

Limitation: the sub-scores of one field are strongly correlated, so the
conditional independence of the model is violated and the prior weights are
inflated (the same evidence is counted several times). The recalibrator fits
the weights with labels; in the simulation on the thesis extract that did not
suffice.

The final score (``nota final``) is intentionally never used here.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.special import expit, logit

TOLERANCE = 1e-9
_MISSING_LEVEL = "missing"
_PENALTY_LEVEL = re.compile(r"^P\d+$")

FieldSetName = Literal["cdr_subscores", "composites"]
FIELD_SETS: tuple[str, ...] = ("cdr_subscores", "composites")
FieldKind = Literal["composite", "values", "terciles", "flag"]


class FellegiSunterError(RuntimeError):
    """The EM fit degenerated (prevalence out of range or non-finite values)."""


@dataclass(frozen=True)
class FieldSpec:
    """One compared field: source column(s) and the names of its levels.

    ``level_names`` lists the ordinal agreement levels (``L0``, ``L1``, ...),
    optionally followed by the ``missing`` level. Level codes are positions in
    this tuple.
    """

    name: str
    column: str
    level_names: tuple[str, ...]
    missing_column: str | None = None
    kind: FieldKind = "composite"
    cuts: tuple[float, ...] = ()
    sign: int = 1

    @property
    def n_levels(self) -> int:
        return len(self.level_names)

    @property
    def n_agreement(self) -> int:
        return sum(1 for n in self.level_names if n != _MISSING_LEVEL)

    @property
    def is_penalty(self) -> bool:
        """True for columns whose non-zero values are all negative."""
        return self.sign < 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "column": self.column,
            "level_names": list(self.level_names),
            "missing_column": self.missing_column,
            "kind": self.kind,
            "cuts": [float(c) for c in self.cuts],
            "sign": int(self.sign),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FieldSpec:
        missing = data.get("missing_column")
        return cls(
            name=str(data["name"]),
            column=str(data["column"]),
            level_names=tuple(str(n) for n in data["level_names"]),
            missing_column=None if missing is None else str(missing),
            kind=data.get("kind", "composite"),
            cuts=tuple(float(c) for c in data.get("cuts", ())),
            sign=int(data.get("sign", 1)),
        )


FIELD_SPECS: tuple[FieldSpec, ...] = (
    FieldSpec("nome", "nome_score_total", ("L0", "L1", "L2", "L3")),
    FieldSpec(
        "mae",
        "mae_score_total",
        ("L0", "L1", "L2", "L3", _MISSING_LEVEL),
        missing_column="mae_missing",
    ),
    FieldSpec("dtnasc", "dtnasc_score_total", ("L0", "L1", "L2", "L3")),
    FieldSpec("endereco", "endereco_score_total", ("L0", "L1", "L2", "L3")),
    FieldSpec("municipio", "municipio_score", ("L0", "L1", "L2")),
)


@dataclass(frozen=True)
class FieldLevels:
    """Discretized fields: level code per row and field, NaN counts per field.

    ``present`` lists, for each field, the level codes that occur in the data
    and ``reference`` is the level used as the baseline of the one-hot design:
    ``L0`` when present, else ``missing`` when present, else the first present
    level. Levels that never occur are outside the model support.
    """

    codes: dict[str, NDArray[np.int64]]
    n_nan: dict[str, int]
    n_rows: int
    specs: tuple[FieldSpec, ...] = FIELD_SPECS
    field_set: str = "composites"
    present: dict[str, tuple[int, ...]] = field(init=False)
    reference: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        present: dict[str, tuple[int, ...]] = {}
        reference: dict[str, int] = {}
        for spec in self.specs:
            levels = tuple(int(c) for c in np.unique(self.codes[spec.name]))
            present[spec.name] = levels
            reference[spec.name] = _reference_level(spec, levels)
        object.__setattr__(self, "present", present)
        object.__setattr__(self, "reference", reference)


def _reference_level(spec: FieldSpec, present: tuple[int, ...]) -> int:
    """L0 if present, else ``missing`` if present, else the first present level."""
    if not present or 0 in present:
        return 0
    if _MISSING_LEVEL in spec.level_names:
        missing = spec.level_names.index(_MISSING_LEVEL)
        if missing in present:
            return missing
    return present[0]


def _grade(x: NDArray[np.float64], n_agreement: int) -> NDArray[np.int64]:
    """Ordinal level of scores in [0, 1]; NaN falls in L0.

    ``=0`` and ``=1`` use a 1e-9 tolerance; interior cut points are exact:
    L1 is ``(0, 0.5]`` and L2 is ``(0.5, 1)`` for four-level fields.
    """
    top = x >= 1.0 - TOLERANCE
    positive = x > TOLERANCE
    if n_agreement == 4:
        codes = np.where(top, 3, np.where(x > 0.5, 2, np.where(positive, 1, 0)))
    else:
        codes = np.where(top, n_agreement - 1, np.where(positive, 1, 0))
    return codes.astype(np.int64)


CDR_SUBSCORE_COLUMNS: tuple[str, ...] = (
    *(
        f"{prefix} {suffix}"
        for prefix in ("NOME", "NOMEMAE")
        for suffix in (
            "prim frag igual",
            "ult frag igual",
            "qtd frag iguais",
            "qtd frag raros",
            "qtd frag comuns",
            "qtd frag muito parec",
            "qtd frag abrev",
        )
    ),
    *(
        f"DTNASC {suffix}"
        for suffix in (
            "dt iguais",
            "dt ap 1digi",
            "dt inv dia",
            "dt inv mes",
            "dt inv ano",
        )
    ),
    *(
        f"CODMUNRES {suffix}"
        for suffix in ("uf igual", "uf prox", "local igual", "local prox")
    ),
    *(
        f"ENDERECO {suffix}"
        for suffix in (
            "via igual",
            "via prox",
            "numero igual",
            "compl prox",
            "texto prox",
            "tokens jacc",
        )
    ),
)
CDR_MISSING_COLUMN = "mae_missing"


def resolve_column(columns: Sequence[str], canonical: str) -> str | None:
    """Column named ``canonical``, or one that is it with a ``,Type`` suffix."""
    if canonical in columns:
        return canonical
    for col in columns:
        if str(col).split(",")[0].strip() == canonical:
            return str(col)
    return None


def _resolve_all(
    columns: Sequence[str], field_set: str
) -> tuple[dict[str, str], list[str]]:
    """Actual column per canonical name, and the canonical names not found."""
    if field_set == "composites":
        names = [s.column for s in FIELD_SPECS] + [
            s.missing_column for s in FIELD_SPECS if s.missing_column is not None
        ]
        exact = set(columns)
        return {n: n for n in names if n in exact}, [n for n in names if n not in exact]
    if field_set == "cdr_subscores":
        resolved: dict[str, str] = {}
        absent: list[str] = []
        for name in (*CDR_SUBSCORE_COLUMNS, CDR_MISSING_COLUMN):
            actual = resolve_column(columns, name)
            if actual is None:
                absent.append(name)
            else:
                resolved[name] = actual
        return resolved, absent
    raise ValueError(f"field_set must be one of {FIELD_SETS}, got {field_set!r}")


def field_columns(columns: Sequence[str], field_set: str) -> tuple[str, ...]:
    """Actual DataFrame columns the field set reads (those that exist)."""
    resolved, _ = _resolve_all(columns, field_set)
    return tuple(dict.fromkeys(resolved.values()))


def _graded_spec(name: str, column: str, x: NDArray[np.float64]) -> FieldSpec:
    """Label-free level rule of one CdR sub-score (see the module docstring)."""
    positive = x[x > 0.0]
    negative = -x[x < 0.0]
    if positive.size:
        sign, mags = 1, positive
    elif negative.size:
        sign, mags = -1, negative
    else:
        return FieldSpec(name, column, ("L0",), kind="values")
    distinct = np.unique(mags)
    prefix = "L" if sign > 0 else "P"
    if distinct.size <= 3:
        kind: FieldKind = "values"
        cuts = tuple(float(v) for v in distinct)
        n_levels = len(cuts)
    else:
        kind = "terciles"
        cuts = tuple(float(v) for v in np.quantile(mags, [1.0 / 3.0, 2.0 / 3.0]))
        n_levels = 3
    names = ("L0", *(f"{prefix}{i}" for i in range(1, n_levels + 1)))
    return FieldSpec(name, column, names, kind=kind, cuts=cuts, sign=sign)


def _graded_codes(spec: FieldSpec, x: NDArray[np.float64]) -> NDArray[np.int64]:
    """Level codes of ``x`` under the stored cuts (NaN and ``<= 0`` are L0)."""
    signed = x * spec.sign
    active = signed > 0.0
    codes = np.zeros(x.shape[0], dtype=np.int64)
    if spec.n_levels > 1 and active.any():
        idx = np.searchsorted(np.asarray(spec.cuts), signed[active], side="left")
        codes[active] = np.minimum(idx + 1, spec.n_levels - 1)
    return codes


def _discretize_cdr(df: pd.DataFrame) -> FieldLevels:
    resolved, absent = _resolve_all(list(df.columns), "cdr_subscores")
    if absent:
        raise KeyError(f"Missing required columns for discretization: {absent}")
    specs: list[FieldSpec] = []
    codes: dict[str, NDArray[np.int64]] = {}
    n_nan: dict[str, int] = {}
    for name in CDR_SUBSCORE_COLUMNS:
        column = resolved[name]
        x = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=np.float64)
        spec = _graded_spec(name, column, x)
        specs.append(spec)
        codes[name] = _graded_codes(spec, x)
        n_nan[name] = int(np.isnan(x).sum())
    column = resolved[CDR_MISSING_COLUMN]
    flag = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=np.float64)
    specs.append(
        FieldSpec(CDR_MISSING_COLUMN, column, ("L0", _MISSING_LEVEL), kind="flag")
    )
    codes[CDR_MISSING_COLUMN] = (np.nan_to_num(flag, nan=0.0) == 1.0).astype(np.int64)
    n_nan[CDR_MISSING_COLUMN] = int(np.isnan(flag).sum())
    return FieldLevels(
        codes=codes,
        n_nan=n_nan,
        n_rows=len(df),
        specs=tuple(specs),
        field_set="cdr_subscores",
    )


def discretize_fields(df: pd.DataFrame, field_set: str = "composites") -> FieldLevels:
    """Map the frame to level codes for the chosen ``field_set``.

    ``"composites"`` (the default of this function) uses ``FIELD_SPECS``;
    ``"cdr_subscores"`` derives the levels of the 29 sub-scores from ``df``.
    """
    if field_set == "cdr_subscores":
        return _discretize_cdr(df)
    _, absent = _resolve_all(list(df.columns), field_set)
    if absent:
        raise KeyError(f"Missing required columns for discretization: {absent}")
    codes: dict[str, NDArray[np.int64]] = {}
    n_nan: dict[str, int] = {}
    for spec in FIELD_SPECS:
        x = df[spec.column].to_numpy(dtype=np.float64)
        field_codes = _grade(x, spec.n_agreement)
        nan = np.isnan(x)
        if spec.missing_column is not None:
            flag = df[spec.missing_column].to_numpy(dtype=np.float64)
            missing = np.nan_to_num(flag, nan=0.0) == 1.0
            field_codes = np.where(
                missing, spec.level_names.index(_MISSING_LEVEL), field_codes
            ).astype(np.int64)
            nan = nan & ~missing
        codes[spec.name] = field_codes
        n_nan[spec.name] = int(nan.sum())
    return FieldLevels(codes=codes, n_nan=n_nan, n_rows=len(df))


def _design_levels(
    spec: FieldSpec, present: tuple[int, ...], reference: int
) -> list[int]:
    """Level codes with a design column: present levels except the reference."""
    return [code for code in present if code != reference]


def feature_names(levels: FieldLevels) -> tuple[str, ...]:
    """Names (``field:level``) of the design columns for these ``levels``."""
    return tuple(
        f"{spec.name}:{spec.level_names[code]}"
        for spec in levels.specs
        for code in _design_levels(
            spec, levels.present[spec.name], levels.reference[spec.name]
        )
    )


def design_matrix(
    levels: FieldLevels,
) -> tuple[NDArray[np.float64], tuple[str, ...]]:
    """One-hot indicators of the present levels, without each field's reference.

    The reference is ``L0`` when it occurs (15 columns with all levels), else
    ``missing`` when it occurs, else the first present level; absent levels get
    no column, so the design never contains an all-zero or collinear column.
    """
    columns: list[NDArray[np.float64]] = []
    for spec in levels.specs:
        field_codes = levels.codes[spec.name]
        for code in _design_levels(
            spec, levels.present[spec.name], levels.reference[spec.name]
        ):
            columns.append((field_codes == code).astype(np.float64))
    matrix = (
        np.column_stack(columns) if columns else np.empty((levels.n_rows, 0))
    ).astype(np.float64)
    return matrix, feature_names(levels)


def agreement_mask(feature_names: tuple[str, ...] | list[str]) -> NDArray[np.bool_]:
    """True for agreement levels (every feature except ``*:missing``/``*:P<k>``).

    Design columns are the present levels other than the reference, so every
    agreement column is a level above the reference (coefficient >= 0);
    ``missing`` levels and the ``P<k>`` penalty levels of CdR columns that only
    take negative values are free unless they are the reference itself.
    """
    return np.array(
        [
            not (
                name.endswith(f":{_MISSING_LEVEL}")
                or _PENALTY_LEVEL.match(name.rsplit(":", 1)[-1])
            )
            for name in feature_names
        ],
        dtype=bool,
    )


@dataclass(frozen=True)
class FellegiSunterModel:
    """Latent-class parameters: prevalence, and level distributions per class.

    ``m``/``u`` have one entry per level of the field with probability 0 at
    levels outside ``present`` (they never occurred in the fitted data).
    ``log_likelihood`` is the pure log-likelihood; ``penalized_objective`` adds
    the Dirichlet log-prior and is what the EM maximizes and converges on.
    """

    prevalence: float
    m: dict[str, tuple[float, ...]]
    u: dict[str, tuple[float, ...]]
    log_likelihood: float
    n_iter: int
    converged: bool
    present: dict[str, tuple[int, ...]] = field(default_factory=dict)
    reference: dict[str, int] = field(default_factory=dict)
    penalized_objective: float | None = None
    specs: tuple[FieldSpec, ...] = FIELD_SPECS
    field_set: str = "composites"

    def _present(self, spec: FieldSpec) -> tuple[int, ...]:
        return self.present.get(spec.name, tuple(range(spec.n_levels)))

    def _reference(self, spec: FieldSpec) -> int:
        return self.reference.get(spec.name, 0)

    def _log_ratio(self, spec: FieldSpec) -> NDArray[np.float64]:
        """``log m - log u`` per level; 0 (neutral) at levels outside support."""
        m = np.asarray(self.m[spec.name], dtype=np.float64)
        u = np.asarray(self.u[spec.name], dtype=np.float64)
        ratio = np.zeros(spec.n_levels, dtype=np.float64)
        idx = list(self._present(spec))
        ratio[idx] = np.log(m[idx]) - np.log(u[idx])
        return ratio

    def logistic_coefficients(self) -> tuple[float, NDArray[np.float64]]:
        """Intercept and coefficients aligned with ``design_matrix`` columns."""
        intercept = float(logit(self.prevalence))
        coef: list[float] = []
        for spec in self.specs:
            ratio = self._log_ratio(spec)
            ref = self._reference(spec)
            intercept += float(ratio[ref])
            coef.extend(
                float(ratio[code] - ratio[ref])
                for code in _design_levels(spec, self._present(spec), ref)
            )
        return intercept, np.asarray(coef, dtype=np.float64)

    def predict_proba(self, levels: FieldLevels) -> NDArray[np.float64]:
        """Posterior match probability from the latent-class likelihoods."""
        eta = np.full(levels.n_rows, float(logit(self.prevalence)))
        for spec in self.specs:
            eta += self._log_ratio(spec)[levels.codes[spec.name]]
        return np.asarray(expit(eta), dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prevalence": float(self.prevalence),
            "m": {k: [float(v) for v in vs] for k, vs in self.m.items()},
            "u": {k: [float(v) for v in vs] for k, vs in self.u.items()},
            "log_likelihood": float(self.log_likelihood),
            "penalized_objective": (
                None
                if self.penalized_objective is None
                else float(self.penalized_objective)
            ),
            "n_iter": int(self.n_iter),
            "converged": bool(self.converged),
            "present": {k: [int(c) for c in v] for k, v in self.present.items()},
            "reference": {k: int(v) for k, v in self.reference.items()},
            "field_set": self.field_set,
            "specs": [spec.to_dict() for spec in self.specs],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FellegiSunterModel:
        penalized = data.get("penalized_objective")
        specs = (
            tuple(FieldSpec.from_dict(d) for d in data["specs"])
            if "specs" in data
            else FIELD_SPECS
        )
        return cls(
            prevalence=float(data["prevalence"]),
            m={k: tuple(float(v) for v in vs) for k, vs in data["m"].items()},
            u={k: tuple(float(v) for v in vs) for k, vs in data["u"].items()},
            log_likelihood=float(data["log_likelihood"]),
            n_iter=int(data["n_iter"]),
            converged=bool(data["converged"]),
            present={
                k: tuple(int(c) for c in v) for k, v in data.get("present", {}).items()
            },
            reference={k: int(v) for k, v in data.get("reference", {}).items()},
            penalized_objective=None if penalized is None else float(penalized),
            specs=specs,
            field_set=str(data.get("field_set", "composites")),
        )


def _initial_tables(
    spec: FieldSpec, present: tuple[int, ...]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """m increasing and u decreasing in the agreement level; ``missing`` equal.

    Levels outside ``present`` get probability 0 and the rest is renormalized.
    Penalty columns (only negative values) start with ``m == u``: the sign of
    their evidence is left to the other fields and to the data.
    """
    k = spec.n_agreement
    ranks = np.arange(1, k + 1, dtype=np.float64)
    m = list(ranks)
    u = list(ranks[::-1])
    if spec.is_penalty:
        m = [1.0] * k
        u = [1.0] * k
    if spec.n_levels > k:
        m.append((k + 1) / 2)
        u.append((k + 1) / 2)
    keep = np.zeros(spec.n_levels, dtype=bool)
    keep[list(present)] = True
    m_arr = np.where(keep, np.asarray(m, dtype=np.float64), 0.0)
    u_arr = np.where(keep, np.asarray(u, dtype=np.float64), 0.0)
    return m_arr / m_arr.sum(), u_arr / u_arr.sum()


def _expected_agreement(
    tables: dict[str, NDArray[np.float64]], specs: tuple[FieldSpec, ...]
) -> float:
    total = 0.0
    for spec in specs:
        ordinal = np.arange(spec.n_levels, dtype=np.float64)
        ordinal[spec.n_agreement :] = 0.0
        if spec.is_penalty:
            ordinal[:] = 0.0
        total += float(tables[spec.name] @ ordinal)
    return total


def orient_classes(
    m_tab: dict[str, NDArray[np.float64]],
    u_tab: dict[str, NDArray[np.float64]],
    prevalence: float,
    specs: tuple[FieldSpec, ...] = FIELD_SPECS,
) -> tuple[dict[str, NDArray[np.float64]], dict[str, NDArray[np.float64]], float]:
    """Make class 1 the one with higher expected agreement (swap if needed)."""
    if _expected_agreement(m_tab, specs) < _expected_agreement(u_tab, specs):
        return u_tab, m_tab, 1.0 - prevalence
    return m_tab, u_tab, prevalence


def _safe_log(table: NDArray[np.float64]) -> NDArray[np.float64]:
    """Log of a level table; levels with probability 0 map to 0 (unused)."""
    return np.log(np.where(table > 0, table, 1.0))


def fit_fellegi_sunter(
    levels: FieldLevels,
    *,
    initial_prevalence: float = 0.05,
    alpha: float = 1.0,
    tol: float = 1e-8,
    max_iter: int = 500,
) -> FellegiSunterModel:
    """Fit the latent-class Fellegi-Sunter model by EM on unique patterns.

    Deterministic start (m increasing and u decreasing in agreement), Dirichlet
    smoothing ``alpha`` on the m/u updates over the *present* levels only
    (absent levels carry no mass), stop on the relative change of the penalized
    objective (log-likelihood plus ``alpha`` times the log level probabilities,
    the Dirichlet log-prior); when ``alpha == 0`` it is the log-likelihood. If
    class 1 ends with lower expected agreement than class 0 the classes are
    swapped. Raises ``FellegiSunterError`` on degenerate fits.
    """
    if levels.n_rows == 0:
        raise ValueError("Cannot fit Fellegi-Sunter on zero rows")
    if not 0.0 < initial_prevalence < 1.0:
        raise ValueError("initial_prevalence must be in (0, 1)")
    if alpha < 0 or tol <= 0 or max_iter < 1:
        raise ValueError("alpha >= 0, tol > 0 and max_iter >= 1 are required")

    specs = levels.specs
    stacked = np.column_stack([levels.codes[s.name] for s in specs])
    patterns, counts = np.unique(stacked, axis=0, return_counts=True)
    n = counts.astype(np.float64)
    total = float(n.sum())

    m_tab: dict[str, NDArray[np.float64]] = {}
    u_tab: dict[str, NDArray[np.float64]] = {}
    for spec in specs:
        m_tab[spec.name], u_tab[spec.name] = _initial_tables(
            spec, levels.present[spec.name]
        )
    prevalence = float(initial_prevalence)

    def log_prior() -> float:
        if alpha == 0:
            return 0.0
        return float(
            alpha
            * sum(
                _safe_log(m_tab[s.name])[list(levels.present[s.name])].sum()
                + _safe_log(u_tab[s.name])[list(levels.present[s.name])].sum()
                for s in specs
            )
        )

    def e_step(pi: float) -> tuple[float, NDArray[np.float64]]:
        log_m = np.zeros(len(n))
        log_u = np.zeros(len(n))
        for j, spec in enumerate(specs):
            log_m += _safe_log(m_tab[spec.name])[patterns[:, j]]
            log_u += _safe_log(u_tab[spec.name])[patterns[:, j]]
        a = np.log(pi) + log_m
        b = np.log1p(-pi) + log_u
        ll = float(np.sum(n * np.logaddexp(a, b)))
        return ll, np.asarray(expit(a - b), dtype=np.float64)

    def check(pi: float, objective: float) -> None:
        if not np.isfinite(pi) or not np.isfinite(objective):
            raise FellegiSunterError("Non-finite value during EM")
        if pi < 1e-6 or pi > 1.0 - 1e-6:
            raise FellegiSunterError(f"Prevalence {pi:.3g} left [1e-6, 1-1e-6]")

    converged = False
    n_iter = 0
    previous = -np.inf
    for n_iter in range(1, max_iter + 1):
        ll, resp = e_step(prevalence)
        penalized = ll + log_prior()
        check(prevalence, penalized)
        if np.isfinite(previous) and abs(penalized - previous) <= tol * abs(penalized):
            converged = True
            n_iter -= 1
            break
        previous = penalized
        w1 = n * resp
        w0 = n - w1
        prevalence = float(w1.sum() / total)
        for j, spec in enumerate(specs):
            idx = list(levels.present[spec.name])
            c1 = np.bincount(patterns[:, j], weights=w1, minlength=spec.n_levels)
            c0 = np.bincount(patterns[:, j], weights=w0, minlength=spec.n_levels)
            m_new = np.zeros(spec.n_levels)
            u_new = np.zeros(spec.n_levels)
            m_new[idx] = (c1[idx] + alpha) / (c1[idx].sum() + alpha * len(idx))
            u_new[idx] = (c0[idx] + alpha) / (c0[idx].sum() + alpha * len(idx))
            m_tab[spec.name] = m_new
            u_tab[spec.name] = u_new
        if not (np.isfinite(prevalence)):
            raise FellegiSunterError("Non-finite prevalence during EM")
    ll, _ = e_step(prevalence)
    penalized = ll + log_prior()
    check(prevalence, penalized)
    for spec in specs:
        idx = list(levels.present[spec.name])
        if not (
            np.isfinite(m_tab[spec.name]).all() and np.isfinite(u_tab[spec.name]).all()
        ):
            raise FellegiSunterError("Non-finite level probabilities")
        if (m_tab[spec.name][idx] <= 0).any() or (u_tab[spec.name][idx] <= 0).any():
            raise FellegiSunterError(
                f"Zero level probability for field {spec.name!r}; use alpha > 0"
            )

    m_tab, u_tab, prevalence = orient_classes(m_tab, u_tab, prevalence, specs)
    check(prevalence, penalized)

    return FellegiSunterModel(
        prevalence=prevalence,
        m={s.name: tuple(float(v) for v in m_tab[s.name]) for s in specs},
        u={s.name: tuple(float(v) for v in u_tab[s.name]) for s in specs},
        log_likelihood=ll,
        n_iter=n_iter,
        converged=converged,
        present=dict(levels.present),
        reference=dict(levels.reference),
        penalized_objective=penalized,
        specs=levels.specs,
        field_set=levels.field_set,
    )
