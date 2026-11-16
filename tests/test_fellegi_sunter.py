from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from gzcmd_record_linkage.fellegi_sunter import (
    CDR_SUBSCORE_COLUMNS,
    FIELD_SPECS,
    FellegiSunterError,
    FellegiSunterModel,
    FieldLevels,
    agreement_mask,
    design_matrix,
    discretize_fields,
    field_columns,
    fit_fellegi_sunter,
    orient_classes,
)

TRUE_PREVALENCE = 0.1
TRUE_M = {
    "nome": (0.02, 0.08, 0.20, 0.70),
    "mae": (0.05, 0.10, 0.15, 0.60, 0.10),
    "dtnasc": (0.03, 0.07, 0.20, 0.70),
    "endereco": (0.10, 0.20, 0.30, 0.40),
    "municipio": (0.05, 0.15, 0.80),
}
TRUE_U = {
    "nome": (0.60, 0.30, 0.09, 0.01),
    "mae": (0.50, 0.25, 0.10, 0.05, 0.10),
    "dtnasc": (0.70, 0.20, 0.08, 0.02),
    "endereco": (0.55, 0.25, 0.15, 0.05),
    "municipio": (0.30, 0.30, 0.40),
}


def _frame(**overrides: list[float]) -> pd.DataFrame:
    n = len(next(iter(overrides.values())))
    base: dict[str, list[float]] = {
        "nome_score_total": [0.0] * n,
        "mae_score_total": [0.0] * n,
        "mae_missing": [0.0] * n,
        "dtnasc_score_total": [0.0] * n,
        "endereco_score_total": [0.0] * n,
        "municipio_score": [0.0] * n,
    }
    base.update(overrides)
    return pd.DataFrame(base)


def _synthetic(
    n: int = 20000,
    seed: int = 7,
    prevalence: float = TRUE_PREVALENCE,
    m: dict[str, tuple[float, ...]] = TRUE_M,
    u: dict[str, tuple[float, ...]] = TRUE_U,
) -> FieldLevels:
    rng = np.random.default_rng(seed)
    cls = rng.random(n) < prevalence
    codes: dict[str, np.ndarray] = {}
    for spec in FIELD_SPECS:
        draw_m = rng.choice(spec.n_levels, size=n, p=m[spec.name])
        draw_u = rng.choice(spec.n_levels, size=n, p=u[spec.name])
        codes[spec.name] = np.where(cls, draw_m, draw_u).astype(np.int64)
    return FieldLevels(codes=codes, n_nan={s.name: 0 for s in FIELD_SPECS}, n_rows=n)


# --- discretization -------------------------------------------------------


def test_discretize_boundaries_four_levels() -> None:
    values = [0.0, 1e-12, 1e-6, 0.5, 0.5 + 1e-9, 0.9, 1 - 1e-12, 1.0, np.nan]
    levels = discretize_fields(_frame(nome_score_total=values))
    assert levels.codes["nome"].tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 0]
    assert levels.n_nan["nome"] == 1
    assert levels.n_rows == len(values)
    assert levels.codes["nome"].dtype == np.int64


def test_discretize_other_four_level_fields_share_rules() -> None:
    values = [0.0, 0.3, 0.7, 1.0]
    levels = discretize_fields(
        _frame(dtnasc_score_total=values, endereco_score_total=values)
    )
    assert levels.codes["dtnasc"].tolist() == [0, 1, 2, 3]
    assert levels.codes["endereco"].tolist() == [0, 1, 2, 3]


def test_discretize_municipio_three_levels() -> None:
    values = [0.0, 1e-12, 0.3, 0.5, 0.999, 1.0, np.nan]
    levels = discretize_fields(_frame(municipio_score=values))
    assert levels.codes["municipio"].tolist() == [0, 0, 1, 1, 1, 2, 0]
    assert levels.n_nan["municipio"] == 1


def test_discretize_mae_missing_takes_precedence() -> None:
    levels = discretize_fields(
        _frame(
            mae_score_total=[1.0, 0.7, np.nan, np.nan, 0.0],
            mae_missing=[1.0, 1.0, 1.0, 0.0, np.nan],
        )
    )
    assert levels.codes["mae"].tolist() == [4, 4, 4, 0, 0]
    # NaN under mae_missing==1 is explained by the flag, not counted.
    assert levels.n_nan["mae"] == 1


def test_discretize_missing_column_raises_key_error() -> None:
    df = _frame(nome_score_total=[0.1]).drop(columns=["municipio_score"])
    with pytest.raises(KeyError, match="municipio_score"):
        discretize_fields(df)


def test_discretize_requires_mae_missing_column() -> None:
    df = _frame(nome_score_total=[0.1]).drop(columns=["mae_missing"])
    with pytest.raises(KeyError, match="mae_missing"):
        discretize_fields(df)


# --- design matrix --------------------------------------------------------


def test_design_matrix_shape_names_and_mask() -> None:
    df = _frame(
        nome_score_total=[0.0, 0.3, 0.7, 1.0, 0.0, 0.0],
        mae_score_total=[0.0, 0.3, 0.7, 1.0, 0.0, 0.0],
        mae_missing=[0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        dtnasc_score_total=[0.0, 0.3, 0.7, 1.0, 0.0, 0.0],
        endereco_score_total=[0.0, 0.3, 0.7, 1.0, 0.0, 0.0],
        municipio_score=[0.0, 0.5, 1.0, 1.0, 0.0, 0.0],
    )
    matrix, names = design_matrix(discretize_fields(df))
    assert matrix.shape == (6, 15)
    assert matrix.dtype == np.float64
    assert len(set(names)) == 15
    assert names[0] == "nome:L1"
    assert "mae:missing" in names
    assert "municipio:L2" in names and "municipio:L3" not in names
    assert not any(n.endswith(":L0") for n in names)
    col = {n: matrix[:, i] for i, n in enumerate(names)}
    assert col["nome:L1"].tolist() == [0, 1, 0, 0, 0, 0]
    assert col["nome:L3"].tolist() == [0, 0, 0, 1, 0, 0]
    assert col["mae:missing"].tolist() == [0, 0, 0, 0, 1, 0]
    assert col["municipio:L2"].tolist() == [0, 0, 1, 1, 0, 0]
    # at most one indicator per field and row
    assert matrix.sum(axis=1).max() <= len(FIELD_SPECS)

    mask = agreement_mask(names)
    assert mask.dtype == bool
    assert mask.sum() == 14
    assert not mask[names.index("mae:missing")]


def test_design_drops_absent_levels_from_a_short_frame() -> None:
    df = _frame(
        nome_score_total=[0.0, 0.3, 0.7, 1.0],
        mae_missing=[0.0, 0.0, 0.0, 1.0],
        municipio_score=[0.0, 0.0, 1.0, 1.0],
    )
    levels = discretize_fields(df)
    matrix, names = design_matrix(levels)
    assert matrix.shape[1] == len(names)
    assert "nome:L1" in names and "municipio:L1" not in names
    assert all(matrix.sum(axis=0) > 0)  # no all-zero column


def _mae_l0_never_occurs(n: int = 4000, seed: int = 21) -> pd.DataFrame:
    """Thesis-like extract: every mae score of 0 is flagged ``mae_missing``."""
    rng = np.random.default_rng(seed)
    truth = rng.random(n) < 0.15

    def draw(values: list[float], p_m: list[float], p_u: list[float]) -> np.ndarray:
        out = rng.choice(values, size=n, p=p_u)
        out[truth] = rng.choice(values, size=int(truth.sum()), p=p_m)
        return out

    four = [0.0, 0.3, 0.7, 1.0]
    mae = draw(four, [0.1, 0.1, 0.2, 0.6], [0.5, 0.2, 0.2, 0.1])
    return pd.DataFrame(
        {
            "nome_score_total": draw(
                four, [0.05, 0.1, 0.25, 0.6], [0.5, 0.3, 0.15, 0.05]
            ),
            "mae_score_total": mae,
            "mae_missing": (mae == 0.0).astype(int),
            "dtnasc_score_total": draw(
                four, [0.1, 0.1, 0.2, 0.6], [0.6, 0.2, 0.15, 0.05]
            ),
            "endereco_score_total": draw(
                four, [0.1, 0.2, 0.3, 0.4], [0.5, 0.3, 0.1, 0.1]
            ),
            "municipio_score": draw([0.0, 0.5, 1.0], [0.1, 0.2, 0.7], [0.4, 0.3, 0.3]),
        }
    )


def test_empty_l0_level_uses_missing_as_reference() -> None:
    levels = discretize_fields(_mae_l0_never_occurs())
    mae_missing_code = 4
    assert 0 not in levels.present["mae"]
    assert levels.reference["mae"] == mae_missing_code
    assert levels.reference["nome"] == 0

    matrix, names = design_matrix(levels)
    assert "mae:missing" not in names and "mae:L0" not in names
    assert {"mae:L1", "mae:L2", "mae:L3"} <= set(names)
    design = np.column_stack([np.ones(len(matrix)), matrix])
    assert np.linalg.matrix_rank(design) == design.shape[1]  # no collinearity
    assert agreement_mask(names).all()  # every remaining level is above reference


def test_empty_l0_level_has_no_smoothing_mass_and_exact_equivalence() -> None:
    levels = discretize_fields(_mae_l0_never_occurs())
    model = fit_fellegi_sunter(levels, alpha=1.0)
    assert model.m["mae"][0] == 0.0 and model.u["mae"][0] == 0.0
    assert sum(model.m["mae"]) == pytest.approx(1.0)
    assert sum(model.u["mae"]) == pytest.approx(1.0)
    assert all(
        np.isfinite(v) for tab in (model.m, model.u) for t in tab.values() for v in t
    )
    assert model.reference["mae"] == 4

    matrix, names = design_matrix(levels)
    intercept, coef = model.logistic_coefficients()
    assert coef.shape == (len(names),)
    np.testing.assert_allclose(
        model.predict_proba(levels), expit(intercept + matrix @ coef), atol=1e-10
    )
    restored = FellegiSunterModel.from_dict(model.to_dict())
    assert restored == model
    assert restored.logistic_coefficients()[0] == pytest.approx(intercept)


def test_absent_level_does_not_shift_the_other_level_estimates() -> None:
    # Smoothing an empty cell would move mass to it: with alpha large the present
    # levels must still add to 1 and the absent one must stay at exactly 0.
    levels = discretize_fields(_mae_l0_never_occurs())
    model = fit_fellegi_sunter(levels, alpha=50.0)
    assert model.m["mae"][0] == 0.0
    assert sum(model.m["mae"][1:]) == pytest.approx(1.0)


# --- EM -------------------------------------------------------------------


def test_em_recovers_synthetic_parameters() -> None:
    levels = _synthetic()
    model = fit_fellegi_sunter(levels)
    assert model.converged
    assert model.prevalence == pytest.approx(TRUE_PREVALENCE, abs=0.02)
    for name in TRUE_M:
        assert np.allclose(model.m[name], TRUE_M[name], atol=0.06), name
        assert np.allclose(model.u[name], TRUE_U[name], atol=0.03), name
        assert sum(model.m[name]) == pytest.approx(1.0)
        assert sum(model.u[name]) == pytest.approx(1.0)


def test_em_swaps_classes_when_prevalence_is_high() -> None:
    # Match class is the majority: the identification rule must still label the
    # class with higher agreement as class 1.
    levels = _synthetic(prevalence=0.7, seed=11)
    model = fit_fellegi_sunter(levels)
    assert model.prevalence == pytest.approx(0.7, abs=0.03)
    assert model.m["nome"][3] > model.u["nome"][3]
    assert model.m["nome"][0] < model.u["nome"][0]


def test_em_swaps_when_data_generated_with_inverted_roles() -> None:
    levels = _synthetic(prevalence=0.9, seed=3, m=TRUE_U, u=TRUE_M)
    model = fit_fellegi_sunter(levels)
    # generating class 1 has the low-agreement tables, so after identification
    # the fitted class 1 is the minority class with high agreement.
    assert model.prevalence == pytest.approx(0.1, abs=0.03)
    assert np.allclose(model.m["nome"], TRUE_M["nome"], atol=0.06)


def test_em_is_deterministic() -> None:
    levels = _synthetic(n=5000, seed=1)
    a = fit_fellegi_sunter(levels)
    b = fit_fellegi_sunter(levels)
    assert a == b


def test_em_respects_max_iter() -> None:
    model = fit_fellegi_sunter(_synthetic(n=5000), max_iter=2)
    assert model.n_iter == 2
    assert not model.converged


def test_em_initial_prevalence_out_of_range_raises() -> None:
    n = 200
    codes = {s.name: np.zeros(n, dtype=np.int64) for s in FIELD_SPECS}
    levels = FieldLevels(codes=codes, n_nan={s.name: 0 for s in FIELD_SPECS}, n_rows=n)
    # A start below 1e-6 is rejected by the EM guard on the first iteration.
    with pytest.raises(FellegiSunterError, match="Prevalence"):
        fit_fellegi_sunter(levels, initial_prevalence=1e-7, alpha=0.0)


def test_em_rejects_empty_and_bad_arguments() -> None:
    empty = FieldLevels(
        codes={s.name: np.empty(0, dtype=np.int64) for s in FIELD_SPECS},
        n_nan={s.name: 0 for s in FIELD_SPECS},
        n_rows=0,
    )
    with pytest.raises(ValueError):
        fit_fellegi_sunter(empty)
    with pytest.raises(ValueError):
        fit_fellegi_sunter(_synthetic(n=100), initial_prevalence=1.5)


# --- logistic equivalence -------------------------------------------------


def test_predict_proba_equals_logistic_form() -> None:
    levels = _synthetic(n=3000, seed=5)
    model = fit_fellegi_sunter(levels)
    matrix, names = design_matrix(levels)
    intercept, coef = model.logistic_coefficients()
    assert coef.shape == (len(names),)
    logistic = expit(intercept + matrix @ coef)
    np.testing.assert_allclose(model.predict_proba(levels), logistic, atol=1e-10)


def test_predict_proba_matches_direct_bayes_posterior() -> None:
    levels = _synthetic(n=50, seed=2)
    model = fit_fellegi_sunter(levels)
    proba = model.predict_proba(levels)
    for i in range(5):
        num = model.prevalence
        den = 1 - model.prevalence
        for spec in FIELD_SPECS:
            code = int(levels.codes[spec.name][i])
            num *= model.m[spec.name][code]
            den *= model.u[spec.name][code]
        assert proba[i] == pytest.approx(num / (num + den), abs=1e-12)


def test_coefficients_follow_documented_formula() -> None:
    model = FellegiSunterModel(
        prevalence=0.2,
        m={
            "nome": (0.1, 0.1, 0.2, 0.6),
            "mae": (0.1, 0.1, 0.2, 0.4, 0.2),
            "dtnasc": (0.1, 0.1, 0.2, 0.6),
            "endereco": (0.1, 0.1, 0.2, 0.6),
            "municipio": (0.2, 0.3, 0.5),
        },
        u={
            "nome": (0.5, 0.2, 0.2, 0.1),
            "mae": (0.4, 0.2, 0.1, 0.1, 0.2),
            "dtnasc": (0.5, 0.2, 0.2, 0.1),
            "endereco": (0.5, 0.2, 0.2, 0.1),
            "municipio": (0.5, 0.3, 0.2),
        },
        log_likelihood=0.0,
        n_iter=0,
        converged=True,
    )
    intercept, coef = model.logistic_coefficients()
    r0 = {k: np.log(model.m[k][0] / model.u[k][0]) for k in model.m}
    assert intercept == pytest.approx(np.log(0.2 / 0.8) + sum(r0.values()))
    assert coef[0] == pytest.approx(np.log(0.1 / 0.2) - r0["nome"])  # nome:L1
    assert coef[2] == pytest.approx(np.log(0.6 / 0.1) - r0["nome"])  # nome:L3
    assert coef[-1] == pytest.approx(np.log(0.5 / 0.2) - r0["municipio"])


def test_model_dict_round_trip_is_json_compatible() -> None:
    import json

    model = fit_fellegi_sunter(_synthetic(n=3000, seed=9))
    payload = json.loads(json.dumps(model.to_dict()))
    assert FellegiSunterModel.from_dict(payload) == model


# --- class orientation and penalized objective ----------------------------


def _tables(high_agreement_in_m: bool) -> tuple[dict, dict]:
    high = {s.name: np.eye(s.n_levels)[s.n_agreement - 1] for s in FIELD_SPECS}
    low = {s.name: np.eye(s.n_levels)[0] for s in FIELD_SPECS}
    return (high, low) if high_agreement_in_m else (low, high)


def test_orient_classes_swaps_when_class_one_has_lower_agreement() -> None:
    m, u = _tables(high_agreement_in_m=False)
    new_m, new_u, prevalence = orient_classes(m, u, 0.8)
    assert new_m is u and new_u is m
    assert prevalence == pytest.approx(0.2)


def test_orient_classes_keeps_correct_orientation() -> None:
    m, u = _tables(high_agreement_in_m=True)
    new_m, new_u, prevalence = orient_classes(m, u, 0.3)
    assert new_m is m and new_u is u
    assert prevalence == 0.3


def _direct_log_likelihood(model: FellegiSunterModel, levels: FieldLevels) -> float:
    num = np.full(levels.n_rows, model.prevalence)
    den = np.full(levels.n_rows, 1.0 - model.prevalence)
    for spec in FIELD_SPECS:
        num = num * np.asarray(model.m[spec.name])[levels.codes[spec.name]]
        den = den * np.asarray(model.u[spec.name])[levels.codes[spec.name]]
    return float(np.log(num + den).sum())


@pytest.mark.parametrize("alpha", [0.0, 1.0, 5.0])
def test_penalized_objective_adds_dirichlet_log_prior(alpha: float) -> None:
    levels = _synthetic(n=4000, seed=13)
    model = fit_fellegi_sunter(levels, alpha=alpha)
    assert model.log_likelihood == pytest.approx(
        _direct_log_likelihood(model, levels), rel=1e-9
    )
    log_prior = alpha * sum(
        np.log(np.asarray(t)).sum()
        for tabs in (model.m, model.u)
        for t in tabs.values()
    )
    assert model.penalized_objective == pytest.approx(
        model.log_likelihood + log_prior, rel=1e-9
    )
    if alpha == 0.0:
        assert model.penalized_objective == pytest.approx(model.log_likelihood)


# --- CdR sub-score field set ----------------------------------------------


def _cdr_frame(n: int = 3000, seed: int = 3) -> pd.DataFrame:
    """All 29 sub-scores plus mae_missing with mixed value structures."""
    rng = np.random.default_rng(seed)
    cls = rng.random(n) < 0.15
    out: dict[str, np.ndarray] = {}
    for name in CDR_SUBSCORE_COLUMNS:
        agree = np.where(cls, rng.random(n) < 0.85, rng.random(n) < 0.15)
        out[name] = agree.astype(float)
    out["NOME qtd frag iguais"] = np.where(
        cls,
        rng.choice([0.0, 0.25, 0.5, 0.75, 1.0], size=n, p=[0.05, 0.1, 0.15, 0.3, 0.4]),
        rng.choice([0.0, 0.25, 0.5, 0.75, 1.0], size=n, p=[0.6, 0.2, 0.1, 0.07, 0.03]),
    )
    out["NOME qtd frag comuns"] = np.where(
        cls,
        rng.choice([-0.0, -0.25, -0.5, -1.0], size=n, p=[0.7, 0.1, 0.1, 0.1]),
        rng.choice([-0.0, -0.25, -0.5, -1.0], size=n, p=[0.3, 0.3, 0.2, 0.2]),
    )
    out["mae_missing"] = (rng.random(n) < 0.1).astype(float)
    return pd.DataFrame(out)


def test_cdr_level_rule_values_terciles_nan_and_penalties() -> None:
    df = _cdr_frame()
    df["NOME prim frag igual"] = np.r_[[np.nan, 0.0, -2.0], [1.0] * (len(df) - 3)]
    levels = discretize_fields(df, "cdr_subscores")
    assert levels.field_set == "cdr_subscores"
    assert [s.name for s in levels.specs] == [*CDR_SUBSCORE_COLUMNS, "mae_missing"]
    by_name = {s.name: s for s in levels.specs}
    # NaN, 0 and a negative value fall in L0; only 1.0 is positive: one level
    flag = by_name["NOME prim frag igual"]
    assert flag.kind == "values" and flag.level_names == ("L0", "L1")
    assert levels.codes[flag.name][:3].tolist() == [0, 0, 0]
    assert levels.n_nan[flag.name] == 1
    # <= 3 distinct positive values: one level per value, sorted
    few = by_name["NOME ult frag igual"]
    assert few.kind == "values" and few.cuts == (1.0,)
    # > 3 distinct positive values: terciles of the positive values
    many = by_name["NOME qtd frag iguais"]
    x = df["NOME qtd frag iguais"].to_numpy()
    q1, q2 = np.quantile(x[x > 0], [1 / 3, 2 / 3])
    assert many.kind == "terciles" and many.cuts == (q1, q2)
    expected = np.where(x <= 0, 0, np.where(x <= q1, 1, np.where(x <= q2, 2, 3)))
    np.testing.assert_array_equal(levels.codes[many.name], expected)
    # only-negative column: levels on the magnitudes, named P<k>, free sign
    pen = by_name["NOME qtd frag comuns"]
    assert pen.is_penalty and pen.level_names[0] == "L0"
    assert all(n.startswith("P") for n in pen.level_names[1:])
    neg = df["NOME qtd frag comuns"].to_numpy()
    assert (levels.codes[pen.name][neg == 0] == 0).all()
    assert (levels.codes[pen.name][neg < 0] > 0).all()
    # binary mae_missing flag keeps the free-sign "missing" level
    mae = by_name["mae_missing"]
    assert mae.kind == "flag" and mae.level_names == ("L0", "missing")
    np.testing.assert_array_equal(
        levels.codes["mae_missing"], df["mae_missing"].to_numpy().astype(int)
    )
    _, names = design_matrix(levels)
    mask = agreement_mask(names)
    for name, is_agreement in zip(names, mask, strict=True):
        if name == "mae_missing:missing" or name.startswith("NOME qtd frag comuns:P"):
            assert not is_agreement
        else:
            assert is_agreement
    assert not any(n.endswith(":L0") for n in names)


def test_cdr_reference_level_when_l0_never_occurs() -> None:
    df = _cdr_frame()
    df["DTNASC dt iguais"] = 0.5  # constant positive: L0 absent, single level
    df["NOME qtd frag muito parec"] = np.tile([0.2, 0.4, 0.6, 0.8], len(df) // 4)
    levels = discretize_fields(df, "cdr_subscores")
    assert levels.present["DTNASC dt iguais"] == (1,)
    assert levels.reference["DTNASC dt iguais"] == 1
    assert levels.present["NOME qtd frag muito parec"] == (1, 2, 3)
    assert levels.reference["NOME qtd frag muito parec"] == 1
    matrix, names = design_matrix(levels)
    assert matrix.shape[1] == len(names)
    assert not any(n.startswith("DTNASC dt iguais:") for n in names)
    assert not any(n.startswith("NOME qtd frag muito parec:L1") for n in names)
    assert (matrix.std(axis=0) > 0).all()


def test_cdr_missing_column_suffix_and_errors() -> None:
    df = _cdr_frame().rename(columns={"ENDERECO via prox": "ENDERECO via prox,Float"})
    levels = discretize_fields(df, "cdr_subscores")
    spec = next(s for s in levels.specs if s.name == "ENDERECO via prox")
    assert spec.column == "ENDERECO via prox,Float"
    with pytest.raises(KeyError, match="DTNASC dt inv mes"):
        discretize_fields(df.drop(columns=["DTNASC dt inv mes"]), "cdr_subscores")
    with pytest.raises(ValueError, match="field_set"):
        discretize_fields(df, "other")
    assert field_columns(list(df.columns), "cdr_subscores").count("mae_missing") == 1


def test_cdr_fit_is_exactly_logistic_and_round_trips() -> None:
    levels = discretize_fields(_cdr_frame(), "cdr_subscores")
    model = fit_fellegi_sunter(levels)
    matrix, names = design_matrix(levels)
    intercept, coef = model.logistic_coefficients()
    assert len(coef) == len(names)
    np.testing.assert_allclose(
        expit(intercept + matrix @ coef), model.predict_proba(levels), atol=1e-9
    )
    again = FellegiSunterModel.from_dict(json.loads(json.dumps(model.to_dict())))
    assert again.field_set == "cdr_subscores"
    assert again.specs == model.specs
    np.testing.assert_allclose(again.predict_proba(levels), model.predict_proba(levels))
    assert again.logistic_coefficients()[0] == pytest.approx(intercept)
    # deterministic
    assert fit_fellegi_sunter(levels).to_dict() == model.to_dict()


def test_composites_model_dict_roundtrip_keeps_default_specs() -> None:
    model = fit_fellegi_sunter(_synthetic(n=3000))
    again = FellegiSunterModel.from_dict(model.to_dict())
    assert again.specs == FIELD_SPECS and again.field_set == "composites"
    legacy = model.to_dict()
    del legacy["specs"], legacy["field_set"]
    assert FellegiSunterModel.from_dict(legacy).specs == FIELD_SPECS
