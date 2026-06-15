from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

COMPREC = "COMPREC,C,12,0"
REFREC = "REFREC,C,12,0"
R_DTNASC = "R_DTNASC,C,8,0"
C_DTNASC = "C_DTNASC,C,8,0"
R_DTOBITO = "R_DTOBITO,C,10,0"
C_DTDIAG = "C_DTDIAG,C,10,0"
SCENARIO_NAMES = (
    "match_obvio",
    "nonmatch_obvio",
    "homonimo",
    "obito_antes_diag",
    "mae_ausente",
    "datas_invertidas",
    "zona_cinzenta",
)


_BASE_SCENARIO_VALUES: dict[str, object] = {
    "PASSO": 1,
    "NOME qtd frag iguais": 0.5,
    "NOME prim frag igual": 0,
    "NOME ult frag igual": 0,
    "NOME prim ult frag igual": 0,
    "NOMEMAE qtd frag iguais": 0.5,
    "NOMEMAE prim frag igual": 0,
    "NOMEMAE ult frag igual": 0,
    "NOMEMAE prim ult frag igual": 0,
    "DTNASC dt iguais": 0,
    "DTNASC dt ap 1digi": 0,
    "DTNASC dt inv dia": 0,
    "DTNASC dt inv mes": 0,
    "DTNASC dt inv ano": 0,
    "ENDERECO via igual": 0.5,
    "ENDERECO via prox": 0.5,
    "ENDERECO numero igual": 0.5,
    "ENDERECO compl prox": 0.5,
    "ENDERECO texto prox": 0.5,
    "ENDERECO tokens jaccard": 0.5,
    "CODMUNRES local igual": 0,
    R_DTNASC: "19800510",
    C_DTNASC: "19800510",
    R_DTOBITO: "01012022",
    C_DTDIAG: "20200101",
}


def _resolve_scenarios(scenarios: object | None) -> tuple[str, ...]:
    if scenarios is None:
        return ()
    if scenarios == "all":
        return SCENARIO_NAMES
    if isinstance(scenarios, str):
        names = (scenarios,)
    else:
        try:
            names = tuple(iter(scenarios))
        except TypeError as exc:
            msg = "scenarios deve ser None, 'all' ou iterável de nomes"
            raise ValueError(msg) from exc

    unknown = sorted(set(names) - set(SCENARIO_NAMES))
    if unknown:
        raise ValueError(f"cenários desconhecidos: {', '.join(unknown)}")
    return names


def _scenario_row(name: str) -> dict[str, object]:
    row = dict(_BASE_SCENARIO_VALUES)
    row.update({COMPREC: f"SCEN-{name}", REFREC: f"SREF-{name}"})

    if name == "match_obvio":
        row.update(
            {
                "PAR": 1,
                "TARGET": 1,
                "nota final": 10.0,
                "NOME qtd frag iguais": 1.0,
                "NOME prim frag igual": 1,
                "NOME ult frag igual": 1,
                "NOME prim ult frag igual": 1,
                "DTNASC dt iguais": 1,
                "CODMUNRES local igual": 1,
                "NOMEMAE qtd frag iguais": 1.0,
                "NOMEMAE prim frag igual": 1,
                "NOMEMAE ult frag igual": 1,
                "NOMEMAE prim ult frag igual": 1,
                "ENDERECO via igual": 0.95,
                "ENDERECO via prox": 0.95,
                "ENDERECO numero igual": 0.95,
                "ENDERECO compl prox": 0.95,
                "ENDERECO texto prox": 0.95,
                "ENDERECO tokens jaccard": 0.95,
            }
        )
    elif name == "nonmatch_obvio":
        row.update(
            {
                "PAR": 0,
                "TARGET": 0,
                "nota final": 1.0,
                R_DTNASC: "19800510",
                C_DTNASC: "19550510",
                "NOME qtd frag iguais": 0.0,
                "NOME prim frag igual": 0,
                "NOME ult frag igual": 0,
                "NOME prim ult frag igual": 0,
                "NOMEMAE qtd frag iguais": 0.0,
                "NOMEMAE prim frag igual": 0,
                "NOMEMAE ult frag igual": 0,
                "NOMEMAE prim ult frag igual": 0,
                "ENDERECO via igual": 0.0,
                "ENDERECO via prox": 0.0,
                "ENDERECO numero igual": 0.0,
                "ENDERECO compl prox": 0.0,
                "ENDERECO texto prox": 0.0,
                "ENDERECO tokens jaccard": 0.0,
            }
        )
    elif name == "homonimo":
        row.update(
            {
                "PAR": 0,
                "TARGET": 0,
                "nota final": 7.5,
                R_DTNASC: "19800315",
                C_DTNASC: "19710802",
                "NOME qtd frag iguais": 0.9,
                "NOME prim frag igual": 1,
                "NOME ult frag igual": 1,
                "NOMEMAE qtd frag iguais": 0.6,
                "NOMEMAE prim frag igual": 1,
                "NOMEMAE ult frag igual": 0,
                "ENDERECO via igual": 0.0,
                "ENDERECO via prox": 0.0,
                "ENDERECO numero igual": 0.0,
                "ENDERECO compl prox": 0.0,
                "ENDERECO texto prox": 0.0,
                "ENDERECO tokens jaccard": 0.0,
            }
        )
    elif name == "obito_antes_diag":
        row.update(
            {
                "PAR": 0,
                "TARGET": 0,
                "nota final": 6.0,
                "DTNASC dt iguais": 1,
                R_DTOBITO: "01012010",
                C_DTDIAG: "20200101",
            }
        )
    elif name == "mae_ausente":
        row.update(
            {
                "PAR": 1,
                "TARGET": 1,
                "nota final": 6.5,
                "NOME qtd frag iguais": 0.8,
                "NOME prim frag igual": 1,
                "DTNASC dt iguais": 1,
                "NOMEMAE qtd frag iguais": 0.0,
                "NOMEMAE prim frag igual": 0,
                "NOMEMAE ult frag igual": 0,
                "NOMEMAE prim ult frag igual": 0,
                "CODMUNRES local igual": 1,
            }
        )
    elif name == "datas_invertidas":
        row.update(
            {
                "PAR": 1,
                "TARGET": 1,
                "nota final": 6.8,
                R_DTNASC: "19800312",
                C_DTNASC: "19801203",
                "NOME qtd frag iguais": 0.9,
                "NOME prim frag igual": 1,
                "NOME ult frag igual": 1,
                "NOMEMAE qtd frag iguais": 0.6,
                "NOMEMAE prim frag igual": 1,
                "DTNASC dt inv dia": 1,
                "DTNASC dt inv mes": 1,
                "CODMUNRES local igual": 1,
            }
        )
    elif name == "zona_cinzenta":
        row.update(
            {
                "PAR": 1,
                "TARGET": 1,
                "nota final": 6.5,
                R_DTNASC: "19800315",
                C_DTNASC: "19800316",
                "NOME qtd frag iguais": 0.6,
                "NOME prim frag igual": 1,
                "DTNASC dt ap 1digi": 1,
                "CODMUNRES local igual": 1,
            }
        )
    else:
        raise ValueError(f"cenário desconhecido: {name}")

    return row


@dataclass(frozen=True)
class SyntheticDataset:
    """Conjunto sintético bruto e posterior verdadeiro usado nos notebooks."""

    frame: pd.DataFrame
    p_true: pd.Series
    meta: dict[str, Any]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _sample_scores(rng: np.random.Generator, n_pairs: int) -> np.ndarray:
    means = np.array([3.0, 6.5, 9.0])
    sds = np.array([1.3, 1.1, 1.0])
    weights = np.array([0.40, 0.25, 0.35])
    clusters = rng.choice(len(means), size=n_pairs, p=weights)
    scores = rng.normal(means[clusters], sds[clusters])
    scores = np.clip(scores, 0.0, 10.0)

    for idx, value in enumerate([5.0, 6.0, 7.0, 8.0, 9.0]):
        if idx < n_pairs:
            scores[idx] = value

    return np.round(scores, 2)


def _solve_s0(scores: np.ndarray, match_ratio: float, a_true: float) -> float:
    target = float(np.clip(match_ratio, 0.0, 1.0))
    lo = 0.0
    hi = 10.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        mean_p = float(_sigmoid(a_true * (scores - mid)).mean())
        if mean_p > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _fmt_yyyymmdd(value: date) -> str:
    return value.strftime("%Y%m%d")


def _fmt_ddmmyyyy(value: date) -> str:
    return value.strftime("%d%m%Y")


def _random_birthdate(rng: np.random.Generator) -> date:
    year = int(rng.integers(1940, 2011))
    month = int(rng.integers(1, 13))
    max_day = calendar.monthrange(year, month)[1]
    day = int(rng.integers(1, max_day + 1))
    return date(year, month, day)


def _shift_years(base: date, years: int) -> date:
    year = int(np.clip(base.year + years, 1940, 2010))
    day = min(base.day, calendar.monthrange(year, base.month)[1])
    return date(year, base.month, day)


def _make_ids(
    rng: np.random.Generator,
    n_pairs: int,
    *,
    prefix: str,
    unique_share: float,
) -> list[str]:
    n_unique = max(1, int(round(n_pairs * unique_share)))
    ids = [f"{prefix}{idx:06d}" for idx in range(n_unique)]
    values = ids.copy()
    while len(values) < n_pairs:
        values.append(str(rng.choice(ids)))
    rng.shuffle(values)
    return values[:n_pairs]


def _clip01(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0.0, 1.0)


def _bernoulli(rng: np.random.Generator, probability: np.ndarray) -> np.ndarray:
    return (rng.random(len(probability)) < _clip01(probability)).astype(int)


def generate_comparador(
    *,
    seed: int = 42,
    n_pairs: int = 600,
    match_ratio: float = 0.5,
    scenarios: object | None = None,
) -> SyntheticDataset:
    """Gera pares sintéticos no formato bruto do Comparador.

    O escore ``nota final`` é amostrado antes do rótulo e define o posterior
    verdadeiro por ``sigmoid(a_true * (nota_final - s0))``. Assim, a calibração
    Platt usada pela biblioteca é recuperável sem circularidade.

    Quando ``scenarios`` é informado, linhas narrativas nomeadas são anexadas
    após os pares aleatórios. O ``s0`` continua ajustado só nos pares aleatórios,
    mas o ``p_true`` é calculado pela mesma sigmoide para todas as linhas. Para
    fins didáticos, ``PAR``/``TARGET`` das linhas narrativas recebem rótulos
    fixos calibrados para exercitar guardrails e triagem; em especial,
    ``mae_ausente`` documenta a flag de mãe ausente e a regra cinza de mãe
    ausente do config que ainda não foi implementada no código (R-11).
    """
    scenario_names = _resolve_scenarios(scenarios)

    if n_pairs <= 0:
        raise ValueError("n_pairs deve ser positivo")

    rng = np.random.default_rng(seed)
    scores = _sample_scores(rng, n_pairs)
    a_true = 0.85
    s0 = _solve_s0(scores, match_ratio, a_true)
    p_values = np.clip(_sigmoid(a_true * (scores - s0)), 1e-6, 1.0 - 1e-6)
    target = (rng.random(n_pairs) < p_values).astype(int)
    q = _clip01((scores / 10.0) + rng.normal(0.0, 0.08, n_pairs))

    nome_qtd = _clip01(q + rng.normal(0.0, 0.12, n_pairs))
    nome_prim = _bernoulli(rng, q)
    nome_ult = _bernoulli(rng, q)
    nome_prim_ult = (
        (nome_prim == 1) & (nome_ult == 1) & (rng.random(n_pairs) < 0.9)
    ).astype(int)

    mae_missing = rng.random(n_pairs) < 0.12
    mae_qtd = _clip01(q + rng.normal(0.0, 0.15, n_pairs))
    mae_prim = _bernoulli(rng, q * 0.95)
    mae_ult = _bernoulli(rng, q * 0.95)
    mae_prim_ult = (
        (mae_prim == 1) & (mae_ult == 1) & (rng.random(n_pairs) < 0.85)
    ).astype(int)
    mae_qtd[mae_missing] = 0.0
    mae_prim[mae_missing] = 0
    mae_ult[mae_missing] = 0
    mae_prim_ult[mae_missing] = 0

    endereco_missing = rng.random(n_pairs) < 0.10
    end_via_igual = _clip01(q + rng.normal(0.0, 0.18, n_pairs))
    end_via_prox = _clip01(q + rng.normal(0.0, 0.15, n_pairs))
    end_numero_igual = _clip01(q + rng.normal(0.0, 0.20, n_pairs))
    end_compl_prox = _clip01(q + rng.normal(0.0, 0.22, n_pairs))
    end_texto_prox = _clip01(q + rng.normal(0.0, 0.14, n_pairs))
    end_tokens_jacc = _clip01(q + rng.normal(0.0, 0.12, n_pairs))
    for values in (
        end_via_igual,
        end_via_prox,
        end_numero_igual,
        end_compl_prox,
        end_texto_prox,
        end_tokens_jacc,
    ):
        values[endereco_missing] = 0.0

    r_dtnasc: list[str] = []
    c_dtnasc: list[str] = []
    r_dtobito: list[str] = []
    c_dtdiag: list[str] = []
    dt_iguais: list[int] = []
    dt_ap_1digi: list[int] = []
    dt_inv_dia: list[int] = []
    dt_inv_mes: list[int] = []
    dt_inv_ano: list[int] = []

    for quality in q:
        birth = _random_birthdate(rng)
        similar_date = rng.random() < quality
        if similar_date:
            if rng.random() < 0.82:
                comp_birth = birth
            else:
                comp_birth = birth + timedelta(days=int(rng.choice([-1, 1])))
        else:
            delta_years = int(rng.integers(2, 31)) * int(rng.choice([-1, 1]))
            comp_birth = _shift_years(birth, delta_years)

        death_age_days = int(rng.integers(45 * 365, 90 * 365))
        death = birth + timedelta(days=death_age_days)
        if death.year > 2024:
            death = date(2024, int(rng.integers(1, 13)), int(rng.integers(1, 29)))
        diag = death - timedelta(days=int(rng.integers(0, 121)))
        if diag <= birth:
            diag = birth + timedelta(days=365)

        day_distance = abs((comp_birth - birth).days)
        r_dtnasc.append(_fmt_yyyymmdd(birth))
        c_dtnasc.append(_fmt_yyyymmdd(comp_birth))
        r_dtobito.append(_fmt_ddmmyyyy(death))
        c_dtdiag.append(_fmt_yyyymmdd(diag))
        dt_iguais.append(int(comp_birth == birth))
        dt_ap_1digi.append(int(comp_birth != birth and day_distance <= 9))
        dt_inv_dia.append(int(rng.random() < 0.015 + 0.020 * quality))
        dt_inv_mes.append(int(rng.random() < 0.010 + 0.015 * quality))
        dt_inv_ano.append(int(rng.random() < 0.008 + 0.012 * quality))

    frame = pd.DataFrame(
        {
            COMPREC: _make_ids(rng, n_pairs, prefix="C", unique_share=0.70),
            REFREC: _make_ids(rng, n_pairs, prefix="R", unique_share=0.76),
            "PASSO": np.ones(n_pairs, dtype=int),
            "PAR": target.astype(int),
            "nota final": scores.astype(float),
            R_DTNASC: r_dtnasc,
            C_DTNASC: c_dtnasc,
            R_DTOBITO: r_dtobito,
            C_DTDIAG: c_dtdiag,
            "NOME qtd frag iguais": np.round(nome_qtd, 4),
            "NOME prim frag igual": nome_prim,
            "NOME ult frag igual": nome_ult,
            "NOME prim ult frag igual": nome_prim_ult,
            "NOMEMAE qtd frag iguais": np.round(mae_qtd, 4),
            "NOMEMAE prim frag igual": mae_prim,
            "NOMEMAE ult frag igual": mae_ult,
            "NOMEMAE prim ult frag igual": mae_prim_ult,
            "DTNASC dt iguais": dt_iguais,
            "DTNASC dt ap 1digi": dt_ap_1digi,
            "DTNASC dt inv dia": dt_inv_dia,
            "DTNASC dt inv mes": dt_inv_mes,
            "DTNASC dt inv ano": dt_inv_ano,
            "ENDERECO via igual": np.round(end_via_igual, 4),
            "ENDERECO via prox": np.round(end_via_prox, 4),
            "ENDERECO numero igual": np.round(end_numero_igual, 4),
            "ENDERECO compl prox": np.round(end_compl_prox, 4),
            "ENDERECO texto prox": np.round(end_texto_prox, 4),
            "ENDERECO tokens jacc": np.round(end_tokens_jacc, 4),
            "CODMUNRES local igual": _bernoulli(rng, q),
        }
    )
    scenario_positions: dict[str, int] = {}
    if scenario_names:
        scenario_frame = pd.DataFrame([_scenario_row(name) for name in scenario_names])
        frame = pd.concat([frame, scenario_frame], ignore_index=True)
        scenario_positions = {
            name: n_pairs + offset for offset, name in enumerate(scenario_names)
        }

    all_scores = frame["nota final"].to_numpy(dtype=float)
    all_p_values = np.clip(_sigmoid(a_true * (all_scores - s0)), 1e-6, 1.0 - 1e-6)
    p_true = pd.Series(all_p_values, index=frame.index, name="p_true")
    meta = {
        "seed": seed,
        "n_pairs": n_pairs,
        "match_ratio_target": match_ratio,
        "match_ratio_realized": float(target.mean()),
        "a_true": a_true,
        "s0": float(s0),
        "true_intercept": float(-a_true * s0),
        "true_slope": a_true,
    }
    meta["scenarios"] = scenario_positions
    return SyntheticDataset(frame=frame, p_true=p_true, meta=meta)


def to_comparador_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    """Escreve o DataFrame bruto no CSV esperado pelo carregador."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_frame = frame.drop(columns=["p_true"], errors="ignore")
    csv_frame.to_csv(output, sep=";", decimal=",", encoding="utf-8", index=False)
    return output


def group_aware_split_indices(
    frame: pd.DataFrame,
    *,
    split_by: str = "comprec",
    test_size: float = 0.3,
    seed: int = 42,
    group_stratify: bool = True,
):
    """Delega a separação treino/teste ao splitter real da biblioteca."""
    from gzcmd_record_linkage.splitting import SplitSpec, split_train_test_indices

    split_frame = frame.copy()
    if "COMPREC" not in split_frame.columns and COMPREC in split_frame.columns:
        split_frame["COMPREC"] = split_frame[COMPREC]
    if "REFREC" not in split_frame.columns and REFREC in split_frame.columns:
        split_frame["REFREC"] = split_frame[REFREC]
    if "TARGET" in frame.columns:
        y = frame["TARGET"].astype(int)
    else:
        y = frame["PAR"].isin([1, 2]).astype(int)
    spec = SplitSpec(
        split_by=split_by,
        test_size=test_size,
        seed=seed,
        group_stratify=group_stratify,
    )
    return split_train_test_indices(split_frame, y, spec=spec)
