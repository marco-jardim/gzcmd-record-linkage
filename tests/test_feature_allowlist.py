"""Feature allowlist selection and eval policy-version override."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from gzcmd_record_linkage import cli
from gzcmd_record_linkage import eval as evaluation
from gzcmd_record_linkage.classifier import (
    COMPARATOR_FEATURES,
    ClassifierConfig,
    GZCMDClassifier,
    get_feature_columns,
)
from gzcmd_record_linkage.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data/synthetic/comparador_sintetico.csv"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
CONFIG = str(files("gzcmd_record_linkage") / "gzcmd_v3_config.yaml")
V31_YAML = FIXTURES / "gzcmd_reference_config_v31.yaml"
HISTORICAL_YAML = FIXTURES / "gzcmd_config_0.3.0.yaml"

# Independent snapshot of the historical numeric names (raw-field look-alikes included).
HISTORICAL_OFF = """CODMUNRES local igual
CODMUNRES local prox
CODMUNRES uf igual
CODMUNRES uf prox
CODMUNRES_local_igual
CODMUNRES_local_prox
CODMUNRES_uf_igual
CODMUNRES_uf_prox
COMPREC,C,12,0
C_CODMUNRES,C,6,0
C_DTDIAG,C,10,0
C_DTNASC,C,8,0
C_DTNASC2,C,8,0
C_DTNOTI,C,10,0
C_IDLINHA,C,10,0
C_IDMUNI2,C,6,0
C_IDMUNIAT,C,6,0
C_IDPESSOA,C,10,0
C_IDUNID,C,15,0
C_IDUNIDAT,C,15,0
C_NOVOIDPESSOA,C,10,0
C_SITUENCE,C,10,0
DTNASC dt ap 1digi
DTNASC dt iguais
DTNASC dt inv ano
DTNASC dt inv dia
DTNASC dt inv mes
DTNASC_dt_ap_1digi
DTNASC_dt_iguais
DTNASC_dt_inv_ano
DTNASC_dt_inv_dia
DTNASC_dt_inv_mes
ENDERECO compl prox
ENDERECO numero igual
ENDERECO texto prox
ENDERECO tokens jacc
ENDERECO via igual
ENDERECO via prox
ENDERECO_compl_prox
ENDERECO_numero_igual
ENDERECO_texto_prox
ENDERECO_tokens_jacc
ENDERECO_via_igual
ENDERECO_via_prox
NOME prim frag igual
NOME qtd frag abrev
NOME qtd frag comuns
NOME qtd frag iguais
NOME qtd frag muito parec
NOME qtd frag raros
NOME ult frag igual
NOMEMAE prim frag igual
NOMEMAE qtd frag abrev
NOMEMAE qtd frag comuns
NOMEMAE qtd frag iguais
NOMEMAE qtd frag muito parec
NOMEMAE qtd frag raros
NOMEMAE ult frag igual
NOMEMAE_prim_frag_igual
NOMEMAE_qtd_frag_abrev
NOMEMAE_qtd_frag_comuns
NOMEMAE_qtd_frag_iguais
NOMEMAE_qtd_frag_muito_parec
NOMEMAE_qtd_frag_raros
NOMEMAE_ult_frag_igual
NOME_prim_frag_igual
NOME_qtd_frag_abrev
NOME_qtd_frag_comuns
NOME_qtd_frag_iguais
NOME_qtd_frag_muito_parec
NOME_qtd_frag_raros
NOME_ult_frag_igual
REFREC,C,12,0
R_CODMUNRES,C,6,0
R_DTNASC,C,8,0
R_DTNASC2,C,8,0
R_DTOBITO,C,10,0
R_IDLINHA,C,10,0
SCORE,C,15,0
diff_ano
dtnasc_all_zero
dtnasc_score_total
endereco_score_total
endereco_zero
mae_missing
mae_score_total
municipio_score
nome_score_total
nota final
nota_final
score_regras""".splitlines()
MACD = [
    "macd_nasc_close",
    "macd_nasc_day_match",
    "macd_nasc_diff_capped",
    "macd_nasc_month_match",
    "macd_nasc_partial_overlap",
    "macd_nasc_very_close",
    "macd_nasc_year_match",
    "macd_nome_perf_x_date_far",
    "macd_nome_perf_x_year_diff",
]
RF = ClassifierConfig(
    classifier_type="random_forest",
    n_estimators=3,
    max_depth=2,
    calibration_cv=2,
    n_jobs=1,
)


def frame(enabled: bool = True) -> pd.DataFrame:
    names = HISTORICAL_OFF + (MACD if enabled else [])
    out = pd.DataFrame({name: np.arange(20, dtype=float) for name in names})
    for name in ["COMPREC", "REFREC", "PAR", "TARGET", "PASSO", "guardrail", "p_cal"]:
        out[name] = np.arange(20) % 2
    out["text_column"] = "texto sintético"
    return out


@pytest.mark.parametrize("enabled,count", [(True, 100), (False, 91)])
def test_historical_snapshot_and_default_unchanged(enabled, count):
    expected = sorted(HISTORICAL_OFF + (MACD if enabled else []))
    selected = get_feature_columns(frame(enabled))
    assert len(selected) == count
    assert selected == expected
    assert selected == get_feature_columns(frame(enabled), feature_set="historical")
    # macd_enabled only affects the allowlist selection.
    assert selected == get_feature_columns(frame(enabled), macd_enabled=False)


@pytest.mark.parametrize("enabled,count", [(True, 39), (False, 30)])
def test_allowlist_exact_names_and_no_raw_fields(enabled, count):
    # MACD exists in the frame in both cases: the opt-out must remove it.
    selected = get_feature_columns(
        frame(), feature_set="allowlist", macd_enabled=enabled
    )
    assert len(COMPARATOR_FEATURES) == 30
    assert len(set(COMPARATOR_FEATURES)) == 30
    assert selected == sorted(list(COMPARATOR_FEATURES) + (MACD if enabled else []))
    assert len(selected) == count
    prohibited = ("SCORE", "COMPREC", "REFREC", "TARGET", "C_SITUENCE", "ID")
    assert not any(t in name.upper() for name in selected for t in prohibited)
    assert not any(name.split(",", 1)[0] == "PAR" for name in selected)
    assert not any(name.startswith(("C_", "R_")) for name in selected)
    assert "nota_final" not in selected


def test_allowlist_numeric_only_and_invalid_selection():
    data = frame()
    data["nota final"] = "não numérica"
    assert "nota final" not in get_feature_columns(data, feature_set="allowlist")
    with pytest.raises(ValueError, match="feature_set"):
        get_feature_columns(data, **{"feature_set": "invalid"})


def test_classifier_fit_and_persistence_opt_in(tmp_path):
    clf = GZCMDClassifier(RF, feature_set="allowlist", macd_enabled=False).fit(frame())
    assert clf.features_ == sorted(COMPARATOR_FEATURES)
    path = tmp_path / "classifier.joblib"
    clf.save(path)
    loaded = GZCMDClassifier.load(path)
    assert loaded.feature_set == "allowlist"
    assert loaded.macd_enabled is False
    assert loaded.config == RF
    np.testing.assert_allclose(
        clf.predict_proba(frame()), loaded.predict_proba(frame())
    )

    # Models saved before these fields existed load with the historical defaults.
    state = joblib.load(path)
    del state["feature_set"], state["macd_enabled"]
    joblib.dump(state, path)
    legacy = GZCMDClassifier.load(path)
    assert legacy.feature_set == "historical"
    assert legacy.macd_enabled is True
    np.testing.assert_allclose(
        clf.predict_proba(frame()), legacy.predict_proba(frame())
    )


def test_classifier_defaults_are_historical():
    clf = GZCMDClassifier(RF).fit(frame())
    assert clf.feature_set == "historical"
    assert clf.macd_enabled is True
    assert clf.features_ == get_feature_columns(frame())


def test_csv_evaluation_passes_allowlist_and_macd(monkeypatch):
    captured = {}
    data = frame()
    monkeypatch.setattr(evaluation, "load_comparador_csv", lambda *a, **kw: data)

    def capture(df, **kwargs):
        captured.update(kwargs)
        assert df is data
        return pd.DataFrame()

    monkeypatch.setattr(evaluation, "evaluate_v3_dataframe", capture)
    evaluation.evaluate_v3_csv(
        input_csv="unused.csv",
        config_path=CONFIG,
        modes=None,
        split_by="comprec",
        seeds=[42],
        test_size=0.3,
        group_stratify=True,
        calibration="ml_rf",
        macd_enabled=False,
        feature_set="allowlist",
        policy_version="v3",
    )
    assert captured["feature_set"] == "allowlist"
    assert captured["macd_enabled"] is False
    assert captured["policy_version"] == "v3"


@pytest.mark.parametrize(
    "flags,feature_set,policy_version",
    [
        ([], "historical", None),
        (["--feature-allowlist"], "allowlist", None),
        (["--policy-version", "v3"], "historical", "v3"),
        (["--feature-allowlist", "--policy-version", "v3.1"], "allowlist", "v3.1"),
    ],
)
def test_cli_eval_feature_and_policy_options(
    monkeypatch, flags, feature_set, policy_version
):
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame(
            {
                "mode": ["vigilancia"],
                "split_by": ["comprec"],
                "calibration": ["ml_rf"],
                "macd_enabled": [False],
                "exp_precision": [1.0],
                "exp_recall": [0.9],
                "exp_f1": [0.95],
                "exp_fbeta": [0.92],
                "auto_coverage": [0.9],
                "llm_used": [10],
            }
        )

    monkeypatch.setattr(cli, "evaluate_v3_csv", capture)
    argv = ["eval", "--config", CONFIG, "--calibration", "ml_rf", "--no-macd", *flags]
    assert cli.main(argv) == 0
    assert captured["feature_set"] == feature_set
    assert captured["macd_enabled"] is False
    assert captured["policy_version"] == policy_version


def test_cli_eval_rejects_unknown_policy_version():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["eval", "--policy-version", "v4"])


def _spy_eval(monkeypatch, cfg, **kwargs):
    captured: dict = {}
    data = frame()
    data["nota final"] = 6.0
    original_builder = evaluation.build_engine_from_config

    def guards(df, **kw):
        captured["guards"] = kw
        return pd.DataFrame({"guardrail": pd.Series(pd.NA, index=df.index)})

    def builder(c, **kw):
        captured["engine"] = kw
        return original_builder(c, **kw)

    class ClassifierSpy:
        def __init__(self, **kw):
            captured["classifier"] = kw

        def fit(self, train):
            captured["training_indices"] = train.index.tolist()

        def predict_proba(self, test):
            assert set(test.index).isdisjoint(captured["training_indices"])
            return np.tile([0.9, 0.1], (len(test), 1))

    monkeypatch.setattr(evaluation, "apply_guardrails", guards)
    monkeypatch.setattr(evaluation, "build_engine_from_config", builder)
    monkeypatch.setattr(evaluation, "GZCMDClassifier", ClassifierSpy)
    monkeypatch.setattr(
        evaluation,
        "split_train_test_indices",
        lambda *a, **kw: (np.arange(10), np.arange(10, 20)),
    )
    result = evaluation.evaluate_v3_dataframe(
        data,
        cfg=cfg,
        modes=["vigilancia"],
        split_by="comprec",
        seeds=[42],
        test_size=0.3,
        group_stratify=True,
        calibration="ml_rf",
        macd_enabled=False,
        **kwargs,
    )
    assert len(result) == 1
    return captured


def test_eval_v3_override_and_rf_selection_reach_actual_paths(monkeypatch):
    cfg = load_config(V31_YAML)
    assert cfg.guardrails.policy_version == "v3.1"
    assert (cfg.guardrails.nota_always_match, cfg.guardrails.nota_always_nonmatch) == (
        9,
        5,
    )
    captured = _spy_eval(monkeypatch, cfg, feature_set="allowlist", policy_version="v3")
    assert captured["guards"]["nota_always_match"] == 10.0
    assert captured["guards"]["nota_always_nonmatch"] == 3.0
    assert captured["guards"]["policy_version"] == "v3"
    assert captured["engine"]["policy_version"] == "v3"
    assert captured["classifier"]["feature_set"] == "allowlist"
    assert captured["classifier"]["macd_enabled"] is False
    assert captured["classifier"]["config"].classifier_type == "random_forest"


def test_eval_default_policy_keeps_yaml_thresholds(monkeypatch):
    cfg = load_config(V31_YAML)
    captured = _spy_eval(monkeypatch, cfg)
    # No override requested: thresholds, engine policy and selection are unchanged.
    assert captured["guards"]["nota_always_match"] == 9
    assert captured["guards"]["nota_always_nonmatch"] == 5
    assert captured["guards"]["policy_version"] == "v3"
    assert captured["engine"]["policy_version"] == "v3"
    assert captured["classifier"]["feature_set"] == "historical"


def test_eval_v3_override_noop_on_v3_yaml(monkeypatch):
    cfg = load_config(CONFIG)
    assert cfg.guardrails.policy_version == "v3"
    captured = _spy_eval(monkeypatch, cfg, policy_version="v3")
    assert captured["guards"]["nota_always_match"] == cfg.guardrails.nota_always_match
    assert (
        captured["guards"]["nota_always_nonmatch"]
        == cfg.guardrails.nota_always_nonmatch
    )


def test_eval_rejects_invalid_selectors():
    cfg = load_config(CONFIG)
    data = frame()
    kwargs = dict(
        cfg=cfg,
        modes=["vigilancia"],
        split_by="comprec",
        seeds=[42],
        test_size=0.3,
        group_stratify=True,
        calibration="stub",
        macd_enabled=False,
    )
    with pytest.raises(ValueError, match="policy_version"):
        evaluation.evaluate_v3_dataframe(data, policy_version="v4", **kwargs)
    with pytest.raises(ValueError, match="feature_set"):
        evaluation.evaluate_v3_dataframe(data, **kwargs, **{"feature_set": "bad"})


@pytest.mark.parametrize("calibration", ["platt", "stub"])
def test_eval_v31_yaml_legacy_override_matches_030(calibration):
    common = dict(
        input_csv=CSV,
        modes=None,
        split_by="row",
        seeds=[42],
        test_size=0.3,
        group_stratify=True,
        calibration=calibration,
        macd_enabled=False,
    )
    expected = evaluation.evaluate_v3_csv(config_path=HISTORICAL_YAML, **common)
    actual = evaluation.evaluate_v3_csv(
        config_path=V31_YAML, policy_version="v3", **common
    )
    assert len(actual) > 0
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
