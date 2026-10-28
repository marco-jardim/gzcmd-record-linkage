"""Contratos numéricos e cola de UI do notebook, sem depender de figuras."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import nb_helpers as nbh
import numpy as np
import pandas as pd
import pytest
import yaml

from gzcmd_record_linkage.bands import BandAssigner
from gzcmd_record_linkage.calibration import PlattModel, fit_platt
from gzcmd_record_linkage.classifier import ClassifierConfig, GZCMDClassifier
from gzcmd_record_linkage.config import load_config
from gzcmd_record_linkage.guardrails import apply_guardrails
from gzcmd_record_linkage.loader import load_comparador_csv
from gzcmd_record_linkage.runner import build_engine_from_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "src/gzcmd_record_linkage/gzcmd_v3_config.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def df():
    return load_comparador_csv(ROOT / "data/synthetic/comparador_sintetico.csv")


@pytest.fixture(scope="module")
def spec():
    return nbh.load_anchor_spec(CONFIG_PATH)


def test_reliability_exemplo_e_ece():
    y, p = [0, 0, 1, 1, 1], [0.1, 0.2, 0.4, 0.6, 0.9]
    points = nbh.reliability_points(y, p, n_bins=2)
    np.testing.assert_allclose(points["conf"], [0.7 / 3, 0.75])
    np.testing.assert_allclose(points["acc"], [1 / 3, 1])
    assert points["n"].tolist() == [3, 2]
    assert points["bin"].tolist() == [0, 1]
    np.testing.assert_allclose(points[["lo", "hi"]], [[0, 0.5], [0.5, 1]])
    assert nbh.expected_calibration_error(y, p, n_bins=2) == pytest.approx(0.16)


@pytest.fixture(scope="module", params=["random_forest", "xgboost"])
def small_classifier(request, df):
    config = ClassifierConfig(
        classifier_type=request.param, n_estimators=8, xgb_n_estimators=8,
        xgb_device="cpu", n_jobs=1, random_state=42,
    )
    return GZCMDClassifier(config=config).fit(df.iloc[:100])


def test_importancias_media_cinco_estimadores(small_classifier):
    clf = small_classifier
    summary = nbh.classifier_importances(clf)
    folds = clf.pipeline_["classifier"].calibrated_classifiers_
    names = clf.pipeline_["imputer"].get_feature_names_out(clf.features_)
    expected = np.mean([fold.estimator.feature_importances_ for fold in folds], axis=0)
    np.testing.assert_allclose(
        summary.set_index("feature").loc[names, "importância média"], expected,
    )
    assert summary["folds"].eq(5).all()
    assert summary["importância média"].is_monotonic_decreasing
    assert not set(summary["feature"]) & {"PAR", "TARGET", "p_cal", "COMPREC"}
    with pytest.raises(ValueError, match="não ajustado"):
        nbh.classifier_importances(GZCMDClassifier())


def test_boletim_comparacao_mesmo_teste():
    summary = nbh.model_comparison_metrics(
        [0, 0, 1, 1], {"perfeito": [0, 0, 1, 1]}, p_true=[0, 0, 1, 1],
    )
    assert summary.loc["perfeito", "AUC-ROC"] == 1
    assert summary.loc["perfeito", "AUC-PR (AP)"] == 1
    assert summary.loc["perfeito", "ECE"] == 0
    assert summary.loc["perfeito", "Brier"] == 0
    assert summary.loc["perfeito", "MAE vs p* (nota)"] == 0
    assert np.isfinite(summary.to_numpy()).all()
    with pytest.raises(ValueError, match="mesmo tamanho"):
        nbh.model_comparison_metrics([0, 1], {"ruim": [0.1]}, p_true=[0.1, 0.9])


def test_figuras_comparacao_e_importancias(plt, small_classifier):
    y = [0, 0, 1, 1]
    predictions = {"Platt-nota": [0.1, 0.2, 0.7, 0.8], "RF": [0.2, 0.3, 0.6, 0.9]}
    fig, summary = nbh.plot_model_comparison(y, predictions, p_true=y)
    assert len(fig.axes) == 3
    pd.testing.assert_frame_equal(
        summary, nbh.model_comparison_metrics(y, predictions, p_true=y),
    )
    for ax in fig.axes:
        assert ax.get_title() and ax.get_xlabel() and ax.get_ylabel()
    plt.close(fig)
    fig, summary = nbh.plot_classifier_importances({"modelo": small_classifier})
    assert len(summary) == 10
    assert len(fig.axes[0].patches) == 10
    for ax in fig.axes:
        assert ax.get_title() and ax.get_xlabel() and ax.get_ylabel()
    plt.close(fig)


def test_reliability_bordas_clipping_e_bins_vazios():
    points = nbh.reliability_points([0, 1, 1, 1], [-0.1, 0.5, 1, 1.1], n_bins=4)
    assert points["bin"].tolist() == [0, 2, 3]
    np.testing.assert_allclose(points["conf"], [-0.1, 0.5, 1.05])
    assert points["n"].tolist() == [1, 1, 2]


def test_log_loss_exemplo_e_alvos_suaves():
    assert nbh.log_loss([1], [0.9]) == pytest.approx(0.10536051565782628)
    assert nbh.log_loss([0.25], [0.5]) == pytest.approx(np.log(2))
    assert np.isfinite(nbh.log_loss([0, 1], [1, 0]))
    with pytest.raises(ValueError):
        nbh.log_loss([1.1], [0.5])


def test_newton_primeiro_passo():
    trace = nbh.platt_newton_trace([2, 4, 6, 8], [0, 0, 1, 1])
    first = trace.iloc[0]
    assert first["intercepto"] == pytest.approx(-3.99920, abs=1e-6)
    assert first["inclinacao"] == pytest.approx(0.79984, abs=1e-6)
    np.testing.assert_allclose(first[["g_b", "g_a"]], [0, -4])
    np.testing.assert_allclose(first[["h_bb", "h_ba", "h_aa"]], [1, 5, 30.001])
    assert (trace["perda"].diff().dropna() <= 1e-12).all()


def test_newton_igual_biblioteca_no_csv(df):
    model = fit_platt(df["nota final"], df["TARGET"])
    last = nbh.platt_newton_trace(df["nota final"], df["TARGET"]).iloc[-1]
    np.testing.assert_allclose(
        last[["intercepto", "inclinacao"]], [model.intercept, model.slope],
        atol=1e-10, rtol=0,
    )


def test_newton_suave_ponderado_equivale_a_repetir_linhas():
    x, t, w = np.array([1, 3, 7, 9]), np.array([0.01, 0.2, 0.7, 0.98]), [1, 2, 3, 4]
    weighted = nbh.platt_newton_trace(x, t, weights=w)
    repeated = nbh.platt_newton_trace(np.repeat(x, w), np.repeat(t, w))
    np.testing.assert_allclose(weighted.iloc[-1], repeated.iloc[-1], atol=1e-10)
    with pytest.raises(ValueError, match="pesos"):
        nbh.platt_newton_trace(x, t, weights=[1, -1, 1, 1])


def test_load_anchor_spec(spec):
    assert tuple(asdict(spec).values()) == (
        ">=", 9.0, 0.98, 50, "<", 5.0, 0.001, 200, 0.5, 0.5,
    )


def test_load_anchor_spec_rejeita_regra_arbitraria(tmp_path):
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["anchors"]["pos_anchor"]["rule"] = "nota_final >= 9 or TARGET == 1"
    path = tmp_path / "ancoras.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="Regra"):
        nbh.load_anchor_spec(path)


def test_anchor_masks_fronteiras_e_sobreposicao(spec):
    pos, neg = nbh.anchor_masks([4.9, 5, 8.9, 9, 10], spec)
    assert pos.tolist() == [False, False, False, True, True]
    assert neg.tolist() == [True, False, False, False, False]
    strict = replace(spec, pos_op=">", neg_op="<=")
    pos, neg = nbh.anchor_masks([5, 9], strict)
    assert pos.tolist() == [False, False]
    assert neg.tolist() == [True, False]
    with pytest.raises(ValueError, match="sobrepor"):
        nbh.anchor_masks([8], replace(spec, pos_cut=7, neg_cut=9))


def test_chute_de_dois_pontos(spec):
    model = nbh.anchor_two_point_guess([9.5, 9.5, 3, 3], spec)
    logit_pos = np.log(0.98 / 0.02)
    logit_neg = np.log(0.001 / 0.999)
    assert model.slope == pytest.approx((logit_pos - logit_neg) / 6.5)
    assert model.intercept == pytest.approx(logit_pos - model.slope * 9.5)


def test_ajuste_ancoras_constantes_e_congelamento(spec):
    nota = np.r_[np.full(50, 9.5), np.full(200, 3.0)]
    fit = nbh.fit_anchor_platt(nota, spec, l2=0)
    assert fit.status == "ajustado"
    assert fit.converged
    assert (fit.n_pos, fit.n_neg) == (50, 200)
    assert fit.fitted_mean_pos == pytest.approx(spec.pos_target, abs=1e-6)
    assert fit.fitted_mean_neg == pytest.approx(spec.neg_target, abs=1e-6)
    assert fit.n_iter >= 1
    fallback = PlattModel(intercept=-5.1, slope=0.85)
    frozen = nbh.fit_anchor_platt([9.5, 3], spec, fallback=fallback)
    assert frozen.status == "congelado"
    assert frozen.model is fallback
    assert frozen.n_iter == 0
    assert nbh.fit_anchor_platt([9.5, 3], spec).model is None
    assert nbh.fit_anchor_platt(
        [9.5, 3], spec, enforce_min_count=False,
    ).status == "ajustado"


def test_ancoras_pesos_balanceados(spec):
    nota = np.r_[np.linspace(9, 10, 50), np.linspace(1, 4.9, 200)]
    targets = np.r_[np.full(50, spec.pos_target), np.full(200, spec.neg_target)]
    for balance in (True, False):
        fit = nbh.fit_anchor_platt(nota, spec, balance=balance)
        repeats = (np.r_[np.full(50, 20), np.full(200, 5)]
                   if balance else np.ones(250, dtype=int))
        trace = nbh.platt_newton_trace(
            np.repeat(nota, repeats), np.repeat(targets, repeats),
            l2=8e-3 if balance else 1e-3,
        )
        np.testing.assert_allclose(
            trace.iloc[-1][["intercepto", "inclinacao"]],
            [fit.model.intercept, fit.model.slope], atol=1e-10, rtol=0,
        )


def test_clamp_update():
    prev, new = PlattModel(-5.1, 0.85), PlattModel(-11.9, 1.66)
    used, a, b = nbh.clamp_update(
        prev, new, max_abs_delta_a=0.5, max_abs_delta_b=0.5,
    )
    assert used.slope == pytest.approx(1.35)
    assert used.intercept == pytest.approx(-5.6)
    assert a is True and b is True
    assert nbh.clamp_update(
        prev, prev, max_abs_delta_a=0.5, max_abs_delta_b=0.5,
    ) == (prev, False, False)


def test_janela_fraca_congela(spec):
    strong = np.r_[np.full(50, 9.5), np.full(200, 3)]
    history = nbh.simulate_anchor_windows(
        [strong, [9.5, 3], strong], spec, init=PlattModel(-5.1, 0.85),
    )
    assert history["status"].tolist() == ["ajustado", "congelado", "ajustado"]
    np.testing.assert_array_equal(
        history.loc[0, ["a_usado", "b_usado"]],
        history.loc[1, ["a_usado", "b_usado"]],
    )
    assert not history.loc[1, ["travou_a", "travou_b"]].any()
    assert history["a_usado"].diff().abs().dropna().max() <= 0.5
    assert history["b_usado"].diff().abs().dropna().max() <= 0.5


@pytest.mark.parametrize("mode", ["vigilancia", "confirmacao"])
def test_expected_losses_igual_engine(cfg, mode):
    engine = nbh.build_engine(cfg, mode)
    p = pd.Series(np.linspace(0, 1, 101), index=np.arange(101) + 200)
    bands = np.resize([b.name for b in cfg.bands.definitions], len(p))
    frame = pd.DataFrame({"p_cal": p, "band": bands})
    out = engine.triage(frame)
    losses = nbh.expected_losses(
        p, c_fp=engine.costs.false_positive, c_fn=engine.costs.false_negative,
        c_llm=engine.costs.llm_review,
        e_fp=[engine.llm_error_by_band[b].e_fp for b in bands],
        e_fn=[engine.llm_error_by_band[b].e_fn for b in bands],
    )
    pd.testing.assert_frame_equal(
        losses[["base_choice", "base_loss", "loss_llm", "evr"]],
        out[["base_choice", "base_loss", "loss_llm", "evr"]],
    )


@pytest.mark.parametrize("c_fp,c_fn,expected", [
    (10, 50, [1 / 6, 0.85 / 45.6, 9.15 / 14.4]),
    (100, 20, [100 / 120, 6.25 / 24, 93.75 / 96]),
])
def test_decision_thresholds(c_fp, c_fn, expected):
    result = nbh.decision_thresholds(
        c_fp=c_fp, c_fn=c_fn, c_llm=0.25, e_fp=0.06, e_fn=0.10,
    )
    np.testing.assert_allclose(list(result.values()), expected)
    empty = nbh.decision_thresholds(
        c_fp=c_fp, c_fn=c_fn, c_llm=1000, e_fp=0.06, e_fn=0.10,
    )
    assert np.isnan(empty["llm_inf"]) and np.isnan(empty["llm_sup"])


def test_thresholds_table(cfg):
    table = nbh.thresholds_table(cfg)
    assert len(table) == len(cfg.decision_policy.modes) * len(cfg.bands.definitions)
    assert not table.duplicated(["modo", "banda"]).any()
    assert set(table["banda"]) == {b.name for b in cfg.bands.definitions}


def test_build_engine_orcamento_novo_e_equivalencia(cfg):
    first = nbh.build_engine(cfg, "vigilancia")
    other = nbh.build_engine(cfg, "vigilancia")
    reference = build_engine_from_config(cfg, mode="vigilancia")
    assert first.costs == reference.costs
    assert first.llm_error_by_band == reference.llm_error_by_band
    assert first.min_auto_match == reference.min_auto_match
    assert first.max_auto_nonmatch == reference.max_auto_nonmatch
    assert first.budget == reference.budget
    first.budget.llm_used = 17
    assert other.budget.llm_used == 0
    changed = nbh.build_engine(cfg, "vigilancia", c_fp=1, c_fn=2, c_llm=0, llm_max=5)
    assert asdict(changed.costs) == {
        "false_positive": 1, "false_negative": 2, "llm_review": 0,
    }
    assert changed.budget.llm_max == 5
    disabled = replace(cfg, llm_review=replace(cfg.llm_review, enabled=False))
    assert nbh.build_engine(disabled, "vigilancia", llm_max=50).budget.llm_max == 0


@pytest.mark.parametrize("llm_max", [0, 5, 20, 10000])
@pytest.mark.parametrize("mode", ["vigilancia", "confirmacao"])
def test_review_status_invariantes_engine_real(cfg, mode, llm_max):
    # Bandas variadas exercitam FORCE_REVIEW, travas e candidatos somente EVR.
    frame = pd.DataFrame({
        "p_cal": np.tile(np.linspace(0.001, 0.999, 101), 6),
        "band": np.repeat([b.name for b in cfg.bands.definitions], 101),
        "guardrail": pd.Series(pd.NA, index=range(606), dtype="string"),
    })
    frame.loc[[10, 100, 200], "guardrail"] = "FORCE_REVIEW"
    frame.loc[20, "guardrail"] = "ALWAYS_MATCH"
    frame.loc[80, "guardrail"] = "ALWAYS_NONMATCH"
    engine = nbh.build_engine(cfg, mode, llm_max=llm_max)
    out = engine.triage(frame)
    before = out.copy(deep=True)
    labels = nbh.review_status(
        out, min_auto_match=engine.min_auto_match,
        max_auto_nonmatch=engine.max_auto_nonmatch,
    )
    pd.testing.assert_frame_equal(out, before)
    reviewed = labels["status"].eq("revisado")
    np.testing.assert_array_equal(reviewed, out["action"].eq("LLM_REVIEW"))
    assert labels.loc[[20, 80], "status"].eq("guardrail").all()
    assert not reviewed.loc[[20, 80]].any()
    assert reviewed.sum() == min(llm_max, (out["review_requested"] & ~(
        out["guardrail"].isin(["ALWAYS_MATCH", "ALWAYS_NONMATCH"])
    )).sum())
    ranks = labels["prioridade"].map({"force": 0, "trava": 1, "evr": 2})
    excluded = labels["status"].eq("de fora")
    if reviewed.any() and excluded.any():
        assert ranks[reviewed].max() <= ranks[excluded].min()
    for priority in ("force", "trava", "evr"):
        selected = reviewed & labels["prioridade"].eq(priority)
        outside = excluded & labels["prioridade"].eq(priority)
        if selected.any() and outside.any():
            assert out.loc[selected, "evr"].min() >= out.loc[outside, "evr"].max()


def test_features_defaults_iguais_loader(df):
    score = nbh.nome_score(df, **dict(zip(
        ["w_qtd", "w_prim", "w_ult"], nbh.NOME_PESOS_LOADER, strict=True,
    )))
    pd.testing.assert_series_equal(score, df["nome_score_total"])
    flags = nbh.macd_flags(df, **nbh.MACD_JANELAS_LOADER)
    pd.testing.assert_frame_equal(flags, df[flags.columns])


@pytest.mark.parametrize("x,expected", [([0, 1], 1), ([1, 0], 0), ([0.5, 0.5], 0.5)])
def test_auc_separacao(x, expected):
    assert nbh.auc_separacao(x, [0, 1]) == expected


def test_bandas_fronteiras_config_e_invalidas(df, cfg):
    names = [band.name for band in cfg.bands.definitions]
    edges = [band.max for band in cfg.bands.definitions[:-1]]
    nota = pd.concat([
        df["nota final"], pd.Series([-1, 0, 5, 6, 7, 8, 9, 999, 1000, np.nan]),
    ])
    pd.testing.assert_series_equal(
        nbh.assign_bands_with_edges(nota, edges, names),
        BandAssigner.from_config(cfg).assign(nota),
    )
    for invalid in ([5, 6, 6, 8, 9], [5, 7, 6, 8, 9], [5, 6, np.nan, 8, 9]):
        with pytest.raises(ValueError, match="crescentes"):
            nbh.assign_bands_with_edges(nota, invalid, names)


def test_band_transitions():
    table = nbh.band_transitions(["a", "a", "b"], ["a", "b", "a"], ["a", "b", "c"])
    np.testing.assert_array_equal(table, [[1, 1, 0], [1, 0, 0], [0, 0, 0]])


def test_guardrail_diff_parametros_identicos(df, cfg):
    diff = nbh.guardrail_diff(df, cfg.guardrails, asdict(cfg.guardrails))
    assert not diff[["mudou_regra", "mudou_motivo"]].any().any()
    expected = apply_guardrails(df, **asdict(cfg.guardrails))
    pd.testing.assert_series_equal(diff["guardrail_base"], expected.guardrail,
                                   check_names=False)


def test_interagir_sem_ipywidgets(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "ipywidgets", None)
    assert nbh.interagir(lambda: None, {}) is False
    printed = capsys.readouterr().out
    assert "[modo não interativo:" in printed
    assert "ipywidgets" in printed


def test_import_sem_dependencias_de_apresentacao():
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        [str(ROOT / "src"), str(ROOT / "notebooks")],
    )}
    result = subprocess.run(
        [sys.executable, "-c", "import sys; sys.modules['matplotlib'] = None; "
         "sys.modules['ipywidgets'] = None; import nb_helpers; "
         "assert nb_helpers.AnchorSpec is not None"],
        env=env, cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture
def plt():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib import pyplot

    yield pyplot
    pyplot.close("all")


@pytest.fixture(scope="module")
def plot_data(df):
    indices = np.random.default_rng(42).permutation(len(df))
    train, test = indices[:int(0.7 * len(df))], indices[int(0.7 * len(df)):]
    nota = df["nota final"].to_numpy()
    y = df["TARGET"].to_numpy()
    model = fit_platt(df["nota final"].iloc[train], df["TARGET"].iloc[train])
    frame = df.copy()
    frame["p_cal"] = 1 / (1 + np.exp(-(model.slope * nota + model.intercept)))
    # Curva de referência ilustrativa para exercitar a entrada opcional p*.
    truth = 1 / (1 + np.exp(-0.85 * (nota - 6)))
    return frame, nota, y, test, model, truth


def _check_figure(plt, fig, summary, started):
    from matplotlib.figure import Figure

    try:
        assert isinstance(fig, Figure)
        assert isinstance(summary, pd.DataFrame)
        assert len(summary)
        assert fig.axes
        for ax in fig.axes:
            assert ax.get_title()
            assert ax.get_xlabel()
            assert ax.get_ylabel()
            for text in [ax.title, ax.xaxis.label, ax.yaxis.label, *ax.texts]:
                assert "—" not in text.get_text()
        fig.canvas.draw()
    finally:
        plt.close(fig)
    assert not plt.fignum_exists(fig.number)


@pytest.mark.parametrize("name", [
    "reliability", "platt_explorer", "costs_explorer", "anchor_comparison",
    "bands_guardrails", "feature_explorer", "budget_explorer", "anchor_windows",
])
def test_figuras_defaults_com_titulos_rotulos_e_tempo(plt, plot_data, cfg, spec, name):
    frame, nota, y, test, model, truth = plot_data
    assert len(frame) == 607
    fit = nbh.fit_anchor_platt(nota, spec, fallback=model)
    history = nbh.simulate_anchor_windows([nota, nota[:10], nota], spec, init=model)
    calls = {
        "reliability": lambda: nbh.plot_reliability(y[test], frame["p_cal"].iloc[test]),
        "platt_explorer": lambda: nbh.plot_platt_explorer(
            nota[test], y[test], model, ref_model=model, p_true=truth[test],
        ),
        "costs_explorer": lambda: nbh.plot_costs_explorer(frame, cfg),
        "anchor_comparison": lambda: nbh.plot_anchor_comparison(
            nota, nota[test], y[test], model, fit, spec,
            p_true_all=truth, p_true_test=truth[test],
        ),
        "bands_guardrails": lambda: nbh.plot_bands_guardrails(frame, cfg),
        "feature_explorer": lambda: nbh.plot_feature_explorer(frame),
        "budget_explorer": lambda: nbh.plot_budget_explorer(frame, cfg),
        "anchor_windows": lambda: nbh.plot_anchor_windows(history),
    }
    started = perf_counter()
    fig, summary = calls[name]()
    _check_figure(plt, fig, summary, started)


@pytest.mark.parametrize("cut", [9, 10])
def test_figura_ancoras_cortes_cruzados_informa_sem_levantar(plt, plot_data, spec, cut):
    _, nota, y, test, model, _ = plot_data
    before = plt.get_fignums()
    fig, summary = nbh.plot_anchor_comparison(
        nota, nota[test], y[test], model, spec=replace(spec, neg_cut=cut),
    )
    assert fig is None
    assert "corte negativo" in summary["mensagem"]
    assert plt.get_fignums() == before


@pytest.mark.parametrize("mode", ["vigilancia", "confirmacao"])
def test_figura_custos_usa_engine_config_e_nao_muta(plt, plot_data, cfg, mode):
    frame = plot_data[0]
    before = frame.copy(deep=True)
    fig, counts = nbh.plot_costs_explorer(frame, cfg, mode=mode, c_fp=37, c_llm=1.2)
    assert counts.attrs["custos"]["c_fp"] == 37
    policy = cfg.decision_policy.modes[mode]
    assert counts.attrs["custos"]["c_fn"] == policy.false_negative_cost
    rates = cfg.llm_review.error_rates_by_band["grey_mid"]
    assert counts.attrs["custos"]["e_fp"] == rates["e_fp"]
    guard = apply_guardrails(frame, **asdict(cfg.guardrails))
    ready = frame.assign(band=BandAssigner.from_config(cfg).assign(frame["nota final"]),
                         guardrail=guard.guardrail)
    out = nbh.build_engine(cfg, mode, c_fp=37, c_llm=1.2).triage(ready)
    np.testing.assert_array_equal(
        counts["Ajuste"],
        out["action"].value_counts().reindex(counts.index, fill_value=0),
    )
    assert counts.sum().eq(len(frame)).all()
    pd.testing.assert_frame_equal(frame, before)
    plt.close(fig)


@pytest.mark.parametrize("budget", [0, 5, 10000])
def test_figura_orcamento_reflete_engine_e_orcamento_novo(plt, plot_data, cfg, budget):
    for _ in range(2):
        fig, summary = nbh.plot_budget_explorer(plot_data[0], cfg, llm_max=budget)
        reviewed = summary["status"].eq("revisado")
        np.testing.assert_array_equal(
            reviewed.to_numpy(dtype=bool),
            summary["action"].eq("LLM_REVIEW").to_numpy(dtype=bool),
        )
        requests = summary["status"].isin(["revisado", "de fora"]).sum()
        assert summary["status"].eq("revisado").sum() == min(budget, requests)
        assert summary.attrs["orçamento explorado"] == budget
        assert any(np.array_equal(line.get_ydata(), [0, 0])
                   for line in fig.axes[0].lines)
        plt.close(fig)


def test_figura_bandas_retriam_e_contam_transicoes(plt, plot_data, cfg):
    frame = plot_data[0]
    fig, summary = nbh.plot_bands_guardrails(frame, cfg)
    changes = ["mudaram de banda", "mudaram de regra", "mudaram de ação"]
    assert summary[changes].eq(0).all().all()
    plt.close(fig)
    fig, summary = nbh.plot_bands_guardrails(
        frame, cfg, edges=[4, 5, 6, 7, 8], guard_params={"nota_always_match": 9},
    )
    transitions = summary.attrs["transições"]
    assert transitions.to_numpy().sum() == len(frame)
    assert summary["mudaram de banda"].iloc[0] == len(frame) - np.trace(transitions)
    plt.close(fig)


def test_figura_features_log_zero_e_janelas(plt, df):
    fig, summary = nbh.plot_feature_explorer(df)
    assert fig.axes[2].get_xscale() == "log"
    windows = [line.get_xdata()[0] for line in fig.axes[2].lines]
    assert windows == [8, 31, 5 * 365.25 + 1]
    assert min(patch.get_x() for patch in fig.axes[2].patches) == 1
    np.testing.assert_allclose(summary["AUC novo"], summary["AUC padrão"])
    plt.close(fig)


def test_figura_janelas_destaca_congelamento(plt, plot_data, spec):
    _, nota, _, _, model, _ = plot_data
    history = nbh.simulate_anchor_windows([nota, nota[:10], nota], spec, init=model)
    fig, summary = nbh.plot_anchor_windows(history)
    expected = nbh.fit_anchor_platt(nota, spec, fallback=model).model
    assert summary["status"].tolist() == ["ajustado", "congelado", "ajustado"]
    assert summary.loc[0, "a_bruto"] == pytest.approx(expected.slope)
    assert summary.loc[0, "b_bruto"] == pytest.approx(expected.intercept)
    assert summary.loc[1, "a_usado"] == summary.loc[0, "a_usado"]
    assert summary.loc[1, "b_usado"] == summary.loc[0, "b_usado"]
    assert all(len(ax.patches) == 1 for ax in fig.axes)
    np.testing.assert_array_equal(fig.axes[0].lines[0].get_ydata(), history["a_bruto"])
    np.testing.assert_array_equal(fig.axes[1].lines[1].get_ydata(), history["b_usado"])
    plt.close(fig)


def test_reliability_bordas_decimais():
    points = nbh.reliability_points([1, 1, 1], [0.3, 0.6, 0.7])
    assert points["bin"].tolist() == [3, 6, 7]


@pytest.mark.parametrize("mode", ["vigilancia", "confirmacao"])
def test_evr_positivo_somente_dentro_da_janela(cfg, mode):
    policy = cfg.decision_policy.modes[mode]
    for band in cfg.bands.definitions:
        rates = cfg.llm_review.error_rates_by_band[band.name]
        kwargs = dict(c_fp=policy.false_positive_cost, c_fn=policy.false_negative_cost,
                      c_llm=policy.llm_review_cost,
                      e_fp=rates["e_fp"], e_fn=rates["e_fn"])
        limits = nbh.decision_thresholds(**kwargs)
        p = np.linspace(0, 1, 1001)
        losses = nbh.expected_losses(p, **kwargs)
        inside = (p > limits["llm_inf"]) & (p < limits["llm_sup"])
        np.testing.assert_array_equal(losses["evr"].gt(0), inside)


def test_newton_nao_converge_e_fallback(spec, monkeypatch):
    nota = np.r_[np.linspace(9, 10, 50), np.linspace(1, 4.9, 200)]
    trace = nbh.platt_newton_trace(nota, nota / 10, max_iter=1)
    assert trace.attrs["convergiu"] is False
    assert trace.attrs["status"] == "não convergiu"
    fit = nbh.fit_anchor_platt(nota, spec, max_iter=1)
    assert fit.status == "não convergiu" and not fit.converged
    assert fit.model is None
    fallback = PlattModel(-5, 1)
    fit = nbh.fit_anchor_platt(nota, spec, max_iter=1, fallback=fallback)
    assert fit.model is fallback
    monkeypatch.setattr(np.linalg, "solve", lambda h, g: np.array([np.nan, np.nan]))
    fit = nbh.fit_anchor_platt(nota, spec, fallback=fallback)
    assert fit.status == "não convergiu" and fit.model is fallback
    assert not fit.converged


def test_janela_inicial_congelada_sem_modelo(spec):
    strong = np.r_[np.full(50, 9.5), np.full(200, 3)]
    history = nbh.simulate_anchor_windows([[9.5, 3], strong], spec, init=None)
    assert history.loc[0, "status"] == "congelado"
    parameters = ["a_bruto", "b_bruto", "a_usado", "b_usado", "s_estrela"]
    assert history.loc[0, parameters].isna().all()
    assert history.loc[1, "status"] == "ajustado"


def test_figura_bandas_invalidas_informa(plt, plot_data, cfg):
    fig, summary = nbh.plot_bands_guardrails(plot_data[0], cfg, edges=[5, 6, 6, 8, 9])
    message = "As fronteiras precisam ser crescentes:"
    assert message in summary.loc[0, "mensagem"]
    assert any(message in text.get_text() for ax in fig.axes for text in ax.texts)
    plt.close(fig)


def test_taxas_com_chave_extra(plt, plot_data, cfg):
    rates = {name: {**values, "nota": "metadado"}
             for name, values in cfg.llm_review.error_rates_by_band.items()}
    extra = replace(cfg, llm_review=replace(cfg.llm_review, error_rates_by_band=rates))
    pd.testing.assert_frame_equal(
        nbh.thresholds_table(extra), nbh.thresholds_table(cfg),
    )
    fig, summary = nbh.plot_costs_explorer(plot_data[0], extra)
    assert "nota" not in summary.attrs["custos"]
    plt.close(fig)


@pytest.mark.parametrize("status", ["congelado", "não convergiu"])
def test_figura_ancoras_fallback_explicito(plt, plot_data, spec, status):
    _, nota, y, test, model, _ = plot_data
    fit = replace(nbh.fit_anchor_platt([9.5, 3], spec, fallback=model), status=status)
    fig, summary = nbh.plot_anchor_comparison(
        nota, nota[test], y[test], model, fit, spec,
    )
    message = (
        "Congelado: poucas âncoras; a curva mostrada é o Platt com gabarito."
        if status == "congelado" else
        "Não convergiu: a curva mostrada é o Platt com gabarito."
    )
    assert message in fig.axes[0].get_title()
    assert summary["mensagem"].eq(message).all()
    assert "Ancorado (fallback: Platt com gabarito)" in summary.index
    assert "Ancorado" not in summary.index
    assert status in fig.axes[0].get_title()
    plt.close(fig)


def test_controles_reais_criacao_mudanca_e_preset(monkeypatch):
    pytest.importorskip("ipywidgets")
    ipy_display = pytest.importorskip("IPython.display")

    displayed, calls = [], []
    monkeypatch.setattr(ipy_display, "display", displayed.append)
    controls = {
        "modo": nbh.Escolha(["a", "b"], "a", "Modo"),
        "x": nbh.Deslizante(1, 0, 10, 1, "X"),
        "y": nbh.Deslizante(2, 0, 10, 1, "Y"),
    }
    assert nbh.interagir(lambda **kw: calls.append(kw), controls,
                        presets={"b": {"x": 5, "y": 6}}, preset_de="modo")
    assert calls == [{"modo": "a", "x": 1, "y": 2}]
    modo, x, _, _ = displayed[0].children
    x.value = 3
    assert len(calls) == 2 and calls[-1]["x"] == 3
    modo.value = "b"
    assert len(calls) == 3
    assert calls[-1] == {"modo": "b", "x": 5, "y": 6}


def test_controle_callback_falha_e_strict(monkeypatch, capsys):
    pytest.importorskip("ipywidgets")
    pytest.importorskip("IPython.display")

    def broken(x):
        raise ZeroDivisionError("divisão de demonstração")

    controls = {"x": nbh.Deslizante(1, 0, 2, 1, "X")}
    monkeypatch.delenv("NBH_STRICT_WIDGETS", raising=False)
    assert nbh.interagir(broken, controls) is False
    assert "falha no controle: divisão de demonstração" in capsys.readouterr().out
    monkeypatch.setenv("NBH_STRICT_WIDGETS", "1")
    with pytest.raises(ZeroDivisionError, match="demonstração"):
        nbh.interagir(broken, controls)
    monkeypatch.setitem(sys.modules, "ipywidgets", None)
    assert nbh.interagir(broken, controls) is False


def test_oraculo_opcional():
    y, truth = [0, 0, 1, 1], [0.1, 0.3, 0.6, 0.9]
    predictions = {"modelo": [0.2, 0.4, 0.5, 0.8]}
    baseline = nbh.model_comparison_metrics(y, predictions, p_true=truth)
    oracle = nbh.model_comparison_metrics(
        y, predictions, p_true=truth, include_oracle=True,
    )
    pd.testing.assert_frame_equal(oracle.loc[["modelo"]], baseline)
    assert oracle.loc["p* (oráculo)", "MAE vs p* (nota)"] == 0
    assert oracle.loc["p* (oráculo)", "Brier"] == pytest.approx(
        np.mean((np.array(y) - truth) ** 2),
    )
    assert list(predictions) == ["modelo"]


def test_piso_ece_sorteios_independentes():
    p = np.linspace(0.05, 0.95, 25)
    rng = np.random.default_rng(7)
    values = [nbh.expected_calibration_error(rng.binomial(1, p), p, n_bins=5)
              for _ in range(30)]
    actual = nbh.ece_noise_floor(p, n_bins=5, n_sim=30, seed=7)
    assert actual["mean"] == pytest.approx(np.mean(values))
    np.testing.assert_allclose(
        [actual["ci_low"], actual["ci_high"]], np.quantile(values, [0.025, 0.975]),
    )
    assert nbh.ece_noise_floor([0, 1], n_sim=5) == {
        "mean": 0, "ci_low": 0, "ci_high": 0,
    }


@pytest.mark.parametrize("metric", ["auc", "brier", "logloss", "ece"])
def test_bootstrap_pareado(metric):
    y = np.tile([0, 1], 20)
    a, b = 0.1 + 0.8 * y, np.full(len(y), 0.5)
    score = {"auc": nbh.auc_separacao, "brier": nbh.brier_score,
             "logloss": nbh.log_loss, "ece": nbh.expected_calibration_error}[metric]
    point = (score(a, y) - score(b, y) if metric == "auc"
             else score(y, a) - score(y, b))
    result = nbh.paired_bootstrap_diff(y, a, b, metric=metric, n_boot=30, seed=4)
    assert result["diff"] == pytest.approx(point)
    assert result["ci_low"] <= result["diff"] + 1e-12
    assert result["ci_high"] >= result["diff"] - 1e-12
    assert result == nbh.paired_bootstrap_diff(
        y, a, b, metric=metric, n_boot=30, seed=4,
    )
    assert nbh.paired_bootstrap_diff(y, a, a, metric=metric, n_boot=30) == {
        "diff": 0, "ci_low": 0, "ci_high": 0,
    }


def test_bootstrap_confere_sorteio_conjunto():
    y = np.array([0, 1, 0, 1, 0, 1])
    a = np.array([0.1, 0.7, 0.3, 0.9, 0.2, 0.8])
    b = np.array([0.2, 0.6, 0.4, 0.8, 0.3, 0.7])
    rng = np.random.default_rng(8)
    values = []
    for _ in range(50):
        ix = rng.integers(0, len(y), size=len(y))
        values.append(np.mean((a[ix] - y[ix]) ** 2 - (b[ix] - y[ix]) ** 2))
    result = nbh.paired_bootstrap_diff(y, a, b, metric="brier", n_boot=50, seed=8)
    np.testing.assert_allclose(
        [result["ci_low"], result["ci_high"]], np.quantile(values, [0.025, 0.975]),
    )


def test_custo_por_par_colunas_reais(df, cfg):
    from gzcmd_record_linkage.eval import evaluate_v3_dataframe

    res = evaluate_v3_dataframe(
        df, cfg=cfg, modes=list(cfg.decision_policy.modes), split_by="comprec",
        seeds=[42], test_size=0.3, group_stratify=True,
        calibration="platt", macd_enabled=True,
    )
    before = res.copy(deep=True)
    summary = nbh.custo_esperado_por_par(res, cfg)
    for i, row in res.iterrows():
        policy = cfg.decision_policy.modes[row["mode"]]
        expected = (row["exp_fp"] * policy.false_positive_cost
                    + row["exp_fn"] * policy.false_negative_cost
                    + row["review_selected"] * policy.llm_review_cost) / row["n_test"]
        assert summary.loc[i, "custo_esperado_por_par"] == pytest.approx(expected)
    pd.testing.assert_frame_equal(res, before)


def test_formatacao_br():
    assert nbh.fmt_br(0.12345) == "0,123"
    assert nbh.fmt_br(-12.345, 2) == "-12,35"
