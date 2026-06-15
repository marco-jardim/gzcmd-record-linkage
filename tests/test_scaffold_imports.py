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
    for name in ("expected_calibration_error", "brier_score", "llm_review_stub"):
        assert hasattr(mod, name), f"esperado {name} em nb_helpers"


def test_library_not_shadowed() -> None:
    """notebooks/ no pythonpath não deve sombrear o pacote publicado."""
    gz = importlib.import_module("gzcmd_record_linkage")
    assert gz.__name__ == "gzcmd_record_linkage"
