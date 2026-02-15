from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class BandDefinition:
    name: str
    min: float | None
    max: float | None
    inclusive_max: bool = False


@dataclass(frozen=True)
class BandsConfig:
    definitions: tuple[BandDefinition, ...]


@dataclass(frozen=True)
class ModePolicyConfig:
    false_positive_cost: float
    false_negative_cost: float
    llm_review_cost: float
    min_auto_match_threshold: float | None
    max_auto_nonmatch_threshold: float | None
    llm_max_calls_per_window: int


@dataclass(frozen=True)
class DecisionPolicyConfig:
    modes: dict[str, ModePolicyConfig]


@dataclass(frozen=True)
class LLMReviewConfig:
    enabled: bool
    error_rates_by_band: dict[str, dict[str, float]]


@dataclass(frozen=True)
class GuardrailsParams:
    temporal_days: int = 180
    nota_always_match: float = 10.0
    nota_always_nonmatch: float = 3.0
    homonimia_min_nota: float = 7.0
    homonimia_year_gap: float = 5.0


@dataclass(frozen=True)
class PlattFitParams:
    l2: float = 1e-3
    max_iter: int = 100
    tol: float = 1e-10


@dataclass(frozen=True)
class CalibrationConfig:
    clip_min: float = 1e-6
    clip_max: float = 0.999999
    platt: PlattFitParams = field(default_factory=PlattFitParams)


@dataclass(frozen=True)
class EvaluationDefaults:
    test_size: float = 0.3
    seeds: tuple[int, ...] = (42,)
    split_by: str = "row"
    group_stratify: bool = True


@dataclass(frozen=True)
class MonitoringConfig:
    enabled: bool
    metrics: dict[str, dict[str, float]]
    slices: list[str]
    monitored_fields: list[str]
    actions_on_critical_drift: list[str]


@dataclass(frozen=True)
class GZCMDConfig:
    version: str
    bands: BandsConfig
    decision_policy: DecisionPolicyConfig
    llm_review: LLMReviewConfig
    guardrails: GuardrailsParams = field(default_factory=GuardrailsParams)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    evaluation: EvaluationDefaults = field(default_factory=EvaluationDefaults)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)


def _as_mapping(x: Any, *, field: str) -> Mapping[str, Any]:
    if x is None:
        return {}
    if not isinstance(x, Mapping):
        raise TypeError(f"{field} must be a mapping, got {type(x).__name__}")
    return x


def _load_band_definitions(data: Mapping[str, Any]) -> tuple[BandDefinition, ...]:
    raw_defs = _as_mapping(data.get("bands"), field="bands").get("definitions")
    if raw_defs is None:
        raise ValueError("Missing required config: bands.definitions")

    # Support both styles:
    #  - list of mappings with an explicit `name`
    #  - mapping of {band_name: {min/max/...}}
    items: list[tuple[str, Mapping[str, Any]]] = []
    if isinstance(raw_defs, Mapping):
        for name, cfg in raw_defs.items():
            items.append(
                (str(name), _as_mapping(cfg, field=f"bands.definitions[{name}]"))
            )
    elif isinstance(raw_defs, list):
        if not raw_defs:
            raise ValueError("bands.definitions must be non-empty")
        for i, item in enumerate(raw_defs):
            if not isinstance(item, Mapping):
                raise TypeError(f"bands.definitions[{i}] must be a mapping")
            name = str(item.get("name", "")).strip()
            if not name:
                raise ValueError(f"bands.definitions[{i}].name must be non-empty")
            items.append((name, item))
    else:
        raise TypeError("bands.definitions must be a mapping or a list")

    out: list[BandDefinition] = []
    seen: set[str] = set()
    inclusive_max_count = 0

    for i, (name, item) in enumerate(items):
        name = str(name).strip()
        if not name:
            raise ValueError(f"bands.definitions[{i}] band name must be non-empty")
        if name in seen:
            raise ValueError(f"Duplicate band name: {name!r}")
        seen.add(name)

        min_v = item.get("min")
        max_v = item.get("max")
        min_f = None if min_v is None else float(min_v)
        max_f = None if max_v is None else float(max_v)
        inclusive_max = bool(item.get("inclusive_max", False))
        if inclusive_max:
            inclusive_max_count += 1

        if (min_f is not None) and (max_f is not None) and not (min_f < max_f):
            raise ValueError(
                f"Invalid band range for {name!r}: expected min < max, got {min_f} >= {max_f}"
            )

        out.append(
            BandDefinition(
                name=name,
                min=min_f,
                max=max_f,
                inclusive_max=inclusive_max,
            )
        )

    if inclusive_max_count > 1:
        raise ValueError("Only one band may set inclusive_max: true")

    return tuple(out)


def _load_decision_policy(data: Mapping[str, Any]) -> DecisionPolicyConfig:
    modes = _as_mapping(
        _as_mapping(data.get("decision_policy"), field="decision_policy").get("modes"),
        field="decision_policy.modes",
    )
    if not modes:
        raise ValueError("Missing required config: decision_policy.modes")

    parsed: dict[str, ModePolicyConfig] = {}
    for mode_name, mode_cfg in modes.items():
        cfg = _as_mapping(mode_cfg, field=f"decision_policy.modes[{mode_name}]")

        costs = _as_mapping(
            cfg.get("costs"), field=f"decision_policy.modes[{mode_name}].costs"
        )
        auto = _as_mapping(
            cfg.get("auto"), field=f"decision_policy.modes[{mode_name}].auto"
        )
        llm_budget = _as_mapping(
            cfg.get("llm_budget"),
            field=f"decision_policy.modes[{mode_name}].llm_budget",
        )

        min_auto_raw = auto.get("min_auto_match_threshold")
        min_auto_match_threshold = None if min_auto_raw is None else float(min_auto_raw)
        max_auto_raw = auto.get("max_auto_nonmatch_threshold")
        max_auto_nonmatch_threshold = (
            None if max_auto_raw is None else float(max_auto_raw)
        )

        parsed[mode_name] = ModePolicyConfig(
            false_positive_cost=float(costs["false_positive"]),
            false_negative_cost=float(costs["false_negative"]),
            llm_review_cost=float(costs["llm_review"]),
            min_auto_match_threshold=min_auto_match_threshold,
            max_auto_nonmatch_threshold=max_auto_nonmatch_threshold,
            llm_max_calls_per_window=int(llm_budget["max_calls_per_window"]),
        )

    return DecisionPolicyConfig(modes=parsed)


def _load_llm_review(data: Mapping[str, Any]) -> LLMReviewConfig:
    llm = _as_mapping(data.get("llm_review"), field="llm_review")
    enabled = bool(llm.get("enabled", False))

    reliability = _as_mapping(llm.get("reliability"), field="llm_review.reliability")
    raw_rates = reliability.get("error_rates_by_band") or {}
    if not isinstance(raw_rates, Mapping):
        raise TypeError("llm_review.reliability.error_rates_by_band must be a mapping")

    rates: dict[str, dict[str, float]] = {}
    for band, rate_cfg in raw_rates.items():
        cfg = _as_mapping(
            rate_cfg, field=f"llm_review.reliability.error_rates_by_band[{band}]"
        )
        rates[str(band)] = {"e_fp": float(cfg["e_fp"]), "e_fn": float(cfg["e_fn"])}

    return LLMReviewConfig(enabled=enabled, error_rates_by_band=rates)


def _load_guardrails_params(data: Mapping[str, Any]) -> GuardrailsParams:
    guardrails = _as_mapping(data.get("guardrails"), field="guardrails")
    params = guardrails.get("params")
    params_map = (
        _as_mapping(params, field="guardrails.params")
        if params is not None
        else guardrails
    )

    temporal_days_raw = params_map.get("temporal_days", 180)
    temporal_days = 180 if temporal_days_raw is None else int(temporal_days_raw)

    nota_always_match_raw = params_map.get("nota_always_match", 10.0)
    nota_always_match = (
        10.0 if nota_always_match_raw is None else float(nota_always_match_raw)
    )

    nota_always_nonmatch_raw = params_map.get("nota_always_nonmatch", 3.0)
    nota_always_nonmatch = (
        3.0 if nota_always_nonmatch_raw is None else float(nota_always_nonmatch_raw)
    )

    homonimia_min_nota_raw = params_map.get("homonimia_min_nota", 7.0)
    homonimia_min_nota = (
        7.0 if homonimia_min_nota_raw is None else float(homonimia_min_nota_raw)
    )

    homonimia_year_gap_raw = params_map.get("homonimia_year_gap", 5.0)
    homonimia_year_gap = (
        5.0 if homonimia_year_gap_raw is None else float(homonimia_year_gap_raw)
    )

    return GuardrailsParams(
        temporal_days=temporal_days,
        nota_always_match=nota_always_match,
        nota_always_nonmatch=nota_always_nonmatch,
        homonimia_min_nota=homonimia_min_nota,
        homonimia_year_gap=homonimia_year_gap,
    )


def _load_calibration_config(data: Mapping[str, Any]) -> CalibrationConfig:
    cal = _as_mapping(data.get("calibration"), field="calibration")
    platt = _as_mapping(cal.get("platt"), field="calibration.platt")

    clip_min_raw = cal.get("clip_min", 1e-6)
    clip_min = 1e-6 if clip_min_raw is None else float(clip_min_raw)
    clip_max_raw = cal.get("clip_max", 0.999999)
    clip_max = 0.999999 if clip_max_raw is None else float(clip_max_raw)

    l2_raw = platt.get("l2", 1e-3)
    max_iter_raw = platt.get("max_iter", 100)
    tol_raw = platt.get("tol", 1e-10)

    return CalibrationConfig(
        clip_min=clip_min,
        clip_max=clip_max,
        platt=PlattFitParams(
            l2=1e-3 if l2_raw is None else float(l2_raw),
            max_iter=100 if max_iter_raw is None else int(max_iter_raw),
            tol=1e-10 if tol_raw is None else float(tol_raw),
        ),
    )


def _load_evaluation_defaults(data: Mapping[str, Any]) -> EvaluationDefaults:
    ev = _as_mapping(data.get("evaluation"), field="evaluation")

    raw_seeds = ev.get("seeds", (42,))
    seeds_list: list[int]
    if isinstance(raw_seeds, int):
        seeds_list = [int(raw_seeds)]
    elif isinstance(raw_seeds, list):
        seeds_list = [int(x) for x in raw_seeds]
    elif isinstance(raw_seeds, tuple):
        seeds_list = [int(x) for x in raw_seeds]
    else:
        raise TypeError("evaluation.seeds must be an int or a list of ints")

    if not seeds_list:
        raise ValueError("evaluation.seeds must be non-empty")

    split_by = str(ev.get("split_by", "row")).strip().lower()
    group_stratify = bool(ev.get("group_stratify", True))

    test_size = float(ev.get("test_size", 0.3))
    if not (0.0 < test_size < 1.0):
        raise ValueError("evaluation.test_size must be in (0,1)")

    return EvaluationDefaults(
        test_size=test_size,
        seeds=tuple(seeds_list),
        split_by=split_by,
        group_stratify=group_stratify,
    )


def _load_monitoring_config(data: Mapping[str, Any]) -> MonitoringConfig:
    mon = _as_mapping(data.get("monitoring"), field="monitoring")
    drift = _as_mapping(mon.get("drift"), field="monitoring.drift")

    if not drift.get("enabled", False):
        return MonitoringConfig(enabled=False)

    metrics = _as_mapping(drift.get("metrics"), field="monitoring.drift.metrics")
    safe_metrics: dict[str, dict[str, float]] = {}
    for k, v in metrics.items():
        v_map = _as_mapping(v, field=f"monitoring.drift.metrics[{k}]")
        safe_metrics[k] = {sk: float(sv) for sk, sv in v_map.items()}

    slices = [str(x) for x in drift.get("slices", [])]
    fields = [str(x) for x in drift.get("monitored_fields", [])]
    actions = [str(x) for x in drift.get("actions_on_critical_drift", [])]

    return MonitoringConfig(
        enabled=True,
        metrics=safe_metrics,
        slices=slices,
        monitored_fields=fields,
        actions_on_critical_drift=actions,
    )


def load_config(path: str | Path) -> GZCMDConfig:
    """Load `gzcmd_v3_config.yaml`-style config into typed dataclasses."""

    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, Mapping):
        raise TypeError("Config root must be a mapping")

    version = str(data.get("version", "")).strip()
    if not version:
        raise ValueError("Missing required config: version")

    bands = BandsConfig(definitions=_load_band_definitions(data))
    decision_policy = _load_decision_policy(data)
    llm_review = _load_llm_review(data)
    guardrails = _load_guardrails_params(data)
    calibration = _load_calibration_config(data)
    evaluation = _load_evaluation_defaults(data)
    monitoring = _load_monitoring_config(data)

    return GZCMDConfig(
        version=version,
        bands=bands,
        decision_policy=decision_policy,
        llm_review=llm_review,
        guardrails=guardrails,
        calibration=calibration,
        evaluation=evaluation,
        monitoring=monitoring,
    )
