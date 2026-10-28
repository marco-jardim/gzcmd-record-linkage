"""TST0.3.a — o pythonpath de notebooks/ funciona e os esqueletos importam.

Garante que ``pythonpath = ["src", "notebooks"]`` (DEC-01) está ativo e que os
módulos de apresentação são descobríveis pelo pytest, sem sombrear a lib.
"""

from __future__ import annotations

import importlib


def test_import_synthetic_data() -> None:
    mod = importlib.import_module("synthetic_data")
    expected = ("generate_comparador", "to_comparador_csv", "group_aware_split_indices")
    for name in expected:
        assert hasattr(mod, name), f"esperado {name} em synthetic_data"


def test_import_nb_helpers() -> None:
    mod = importlib.import_module("nb_helpers")
    expected = (
        "expected_calibration_error", "brier_score", "llm_review_stub",
        "reliability_points", "log_loss", "platt_newton_trace", "AnchorSpec",
        "load_anchor_spec", "anchor_masks", "anchor_two_point_guess", "AnchorFit",
        "fit_anchor_platt", "clamp_update", "simulate_anchor_windows",
        "expected_losses", "decision_thresholds", "thresholds_table", "build_engine",
        "review_status", "NOME_PESOS_LOADER", "MACD_JANELAS_LOADER", "nome_score",
        "macd_flags", "auc_separacao", "assign_bands_with_edges", "band_transitions",
        "guardrail_diff", "Deslizante", "Escolha", "Caixa", "interagir",
        "plot_reliability", "plot_platt_explorer", "plot_costs_explorer",
        "plot_anchor_comparison", "plot_bands_guardrails", "plot_feature_explorer",
        "plot_budget_explorer", "plot_anchor_windows",
        "classifier_importances", "model_comparison_metrics",
        "plot_model_comparison", "plot_classifier_importances",
        "ece_noise_floor", "paired_bootstrap_diff", "custo_esperado_por_par", "fmt_br",
    )
    for name in expected:
        assert hasattr(mod, name), f"esperado {name} em nb_helpers"


def test_library_not_shadowed() -> None:
    """notebooks/ no pythonpath não deve sombrear o pacote publicado."""
    gz = importlib.import_module("gzcmd_record_linkage")
    assert gz.__name__ == "gzcmd_record_linkage"
