"""Packaged v3.1 profile and ``estimated_rates_path`` (measured reviewer rates)."""

from __future__ import annotations

import os
from pathlib import Path

import nb_fila
import pytest
import yaml

from gzcmd_record_linkage.config import load_config, packaged_config_path
from gzcmd_record_linkage.reliability import load_band_rates
from gzcmd_record_linkage.runner import build_engine_from_config

PACKAGE_DIR = Path(__file__).resolve().parents[1] / "src" / "gzcmd_record_linkage"
FIXTURES = Path(__file__).parent / "fixtures"
MODES = ("vigilancia", "confirmacao")
BANDS = ("low", "grey_low", "grey_mid", "grey_high", "near_high", "high")


@pytest.fixture(scope="module")
def v31_cfg():
    return load_config(packaged_config_path("v3.1"))


# --------------------------------------------------------------------------- #
# Packaged profile
# --------------------------------------------------------------------------- #


def test_packaged_v31_profile_matches_thesis_p_config(v31_cfg):
    cfg = v31_cfg
    assert packaged_config_path("v3.1").parent.resolve() == PACKAGE_DIR.resolve()
    assert cfg.guardrails.policy_version == "v3.1"
    assert cfg.guardrails.nota_always_match == 9.0
    assert cfg.guardrails.nota_always_nonmatch == 5.0
    assert [b.name for b in cfg.bands.definitions] == list(BANDS)
    assert cfg.calibration.anchor_platt.r_plus == 0.979
    assert cfg.calibration.anchor_platt.r_minus == 0.0001
    assert cfg.evaluation.seeds == (42,)

    vig, conf = (cfg.decision_policy.modes[m] for m in MODES)
    assert (vig.false_positive_cost, vig.false_negative_cost) == (10.0, 50.0)
    assert (conf.false_positive_cost, conf.false_negative_cost) == (100.0, 20.0)
    for mode_cfg in (vig, conf):
        assert mode_cfg.llm_review_cost == 0.005
        assert mode_cfg.human_review_cost == 1.0
        assert (mode_cfg.human_e_fp, mode_cfg.human_e_fn) == (0.05, 0.05)
    assert (vig.llm_max_calls_per_window, conf.llm_max_calls_per_window) == (2000, 1000)

    raw = yaml.safe_load(packaged_config_path("v3.1").read_text(encoding="utf-8"))
    for agent in raw["llm_review"]["model"].values():
        assert agent["max_tokens"] == 2500
        assert "temperature" not in agent


def test_packaged_profile_points_to_packaged_v4_rates(v31_cfg):
    path = Path(v31_cfg.llm_review.estimated_rates_path)
    assert path.is_absolute()
    assert path == (PACKAGE_DIR / "error_rates_by_band_v4.yaml").resolve()
    assert path.is_file()


def test_default_profile_is_unchanged_v3():
    default = load_config(packaged_config_path("v3"))
    assert packaged_config_path("v3").name == "gzcmd_v3_config.yaml"
    assert default.guardrails.policy_version == "v3"
    assert default.llm_review.estimated_rates_path is None
    assert default.llm_review.error_rates_by_band["grey_mid"] == {
        "e_fp": 0.06,
        "e_fn": 0.10,
    }
    with pytest.raises(ValueError, match="Unknown packaged config profile"):
        packaged_config_path("v9")


# --------------------------------------------------------------------------- #
# Measured Haiku v4 rates
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", MODES)
def test_measured_rates_match_notebook_counts(v31_cfg, mode):
    """Bands with decisions: lib (from the YAML) == notebook (from the counts)."""
    rates = v31_cfg.llm_review.error_rates_for_mode(mode)
    notebook = nb_fila.taxas_llm_por_faixa(mode)
    with_decisions = nb_fila.CONTAGENS_LLM_HAIKU_V4[mode]
    assert set(rates) == set(BANDS)
    assert set(with_decisions) <= set(BANDS)
    for band in with_decisions:
        for key in ("e_fp", "e_fn", "e_unsure"):
            assert rates[band][key] == pytest.approx(notebook[band][key], abs=1e-12), (
                mode,
                band,
                key,
            )


@pytest.mark.parametrize("mode", MODES)
def test_fixed_fallback_for_bands_without_decisions(v31_cfg, mode):
    """Bands with an empty denominator keep the fixed e_fp/e_fn (notebook agrees)."""
    rates = v31_cfg.llm_review.error_rates_for_mode(mode)
    fixed = v31_cfg.llm_review.error_rates_by_band
    notebook = nb_fila.taxas_llm_por_faixa(mode)
    for band in set(BANDS) - set(nb_fila.CONTAGENS_LLM_HAIKU_V4[mode]):
        assert rates[band]["e_fp"] == fixed[band]["e_fp"] == notebook[band]["e_fp"]
        assert rates[band]["e_fn"] == fixed[band]["e_fn"] == notebook[band]["e_fn"]


def test_known_values(v31_cfg):
    vig = v31_cfg.llm_review.error_rates_for_mode("vigilancia")
    conf = v31_cfg.llm_review.error_rates_for_mode("confirmacao")
    # vigilancia/high has no non-pairs: e_fp falls back to the fixed 0.01.
    assert vig["high"]["e_fp"] == 0.01
    assert vig["high"]["e_fn"] == pytest.approx(5 / 22)
    # vigilancia/grey_low: FN = 2/2 is kept as measured, not smoothed.
    assert vig["grey_low"]["e_fn"] == 1.0
    assert vig["grey_mid"]["e_fp"] == pytest.approx(14 / 553)
    # confirmacao/grey_mid has no decisions: fixed 0.06 / 0.10.
    assert (conf["grey_mid"]["e_fp"], conf["grey_mid"]["e_fn"]) == (0.06, 0.10)
    assert (conf["grey_low"]["e_fp"], conf["grey_low"]["e_fn"]) == (0.08, 0.12)
    assert conf["high"]["e_fp"] == 0.0  # 0 FP out of 2 non-pairs (denominator > 0)
    assert conf["near_high"]["e_fn"] == pytest.approx(8 / 34)


def test_e_unsure_without_decisions_follows_thesis_not_notebook(v31_cfg):
    """Thesis rule (reliability.py): bands with n<10 use the fixed e_unsure,
    which defaults to 0.0. The notebook instead uses the mode-wide UNSURE
    rate. This documents the intentional difference."""
    conf = v31_cfg.llm_review.error_rates_for_mode("confirmacao")
    notebook = nb_fila.taxas_llm_por_faixa("confirmacao")
    mode_rate = 26 / 459  # (9 + 8 + 9) UNSURE over 459 decisions
    assert notebook["grey_mid"]["e_unsure"] == pytest.approx(mode_rate)
    assert conf["grey_mid"]["e_unsure"] == 0.0
    assert conf["low"]["e_unsure"] == 0.0
    vig = v31_cfg.llm_review.error_rates_for_mode("vigilancia")
    assert vig["low"]["e_unsure"] == 0.0


def test_runner_uses_measured_rates_for_v31_only(v31_cfg):
    engine = build_engine_from_config(v31_cfg, mode="confirmacao")
    assert engine.llm_error_by_band["grey_mid"].e_fp == 0.06
    assert engine.llm_error_by_band["high"].e_fn == pytest.approx(11 / 61)
    assert engine.llm_error_by_band["high"].e_unsure == pytest.approx(9 / 63)
    # Legacy v3 override reproduces v3: inline priors, no measured file read.
    legacy = build_engine_from_config(v31_cfg, mode="confirmacao", policy_version="v3")
    assert legacy.llm_error_by_band["high"].e_fn == 0.03
    assert legacy.llm_error_by_band["high"].e_unsure == 0.0


# --------------------------------------------------------------------------- #
# Loader semantics
# --------------------------------------------------------------------------- #


def _write_profile(directory: Path, rates_ref: str | None) -> Path:
    data = yaml.safe_load(packaged_config_path("v3.1").read_text(encoding="utf-8"))
    reliability = data["llm_review"]["reliability"]
    if rates_ref is None:
        del reliability["estimated_rates_path"]
    else:
        reliability["estimated_rates_path"] = rates_ref
    path = directory / "profile.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _write_rates(path: Path, *, n: int, e_fp: float, u: float) -> None:
    block = {
        "estimates_by_band": {"grey_mid": {"n": n, "u": u}},
        "error_rates_by_band": {"grey_mid": {"e_fp": e_fp, "e_fn": 0.5}},
    }
    data = {"by_mode": {m: {"pooled": block} for m in MODES}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_relative_path_resolves_against_yaml_directory(tmp_path, monkeypatch):
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    _write_rates(config_dir / "sub" / "rates.yaml", n=40, e_fp=0.2, u=0.3)
    profile = _write_profile(config_dir, "sub/rates.yaml")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)  # must not matter

    cfg = load_config(profile)

    expected = (config_dir / "sub" / "rates.yaml").resolve()
    assert Path(cfg.llm_review.estimated_rates_path) == expected
    rates = cfg.llm_review.error_rates_for_mode("vigilancia")["grey_mid"]
    assert rates == {"e_fp": 0.2, "e_fn": 0.5, "e_unsure": 0.3}


def test_small_band_falls_back_to_fixed_values(tmp_path):
    _write_rates(tmp_path / "rates.yaml", n=9, e_fp=0.2, u=0.3)
    cfg = load_config(_write_profile(tmp_path, "rates.yaml"))
    rates = cfg.llm_review.error_rates_for_mode("confirmacao")["grey_mid"]
    assert rates == {"e_fp": 0.06, "e_fn": 0.10, "e_unsure": 0.0}

    fixed = {"grey_mid": {"e_fp": 0.06, "e_fn": 0.10, "e_unsure": 0.02}}
    fallback = load_band_rates(
        fixed, mode="vigilancia", source=tmp_path / "rates.yaml"
    )["grey_mid"]
    assert (fallback.e_fp, fallback.e_fn, fallback.e_unsure) == (0.06, 0.10, 0.02)


def test_unknown_mode_and_invalid_rates_raise(tmp_path):
    rates_path = tmp_path / "rates.yaml"
    _write_rates(rates_path, n=40, e_fp=1.5, u=0.3)
    cfg = load_config(_write_profile(tmp_path, "rates.yaml"))
    with pytest.raises(ValueError, match="Invalid reviewer probabilities"):
        cfg.llm_review.error_rates_for_mode("vigilancia")
    with pytest.raises(ValueError, match="Missing mode block"):
        cfg.llm_review.error_rates_for_mode("inexistente")


def test_config_without_key_behaves_as_before(tmp_path):
    cfg = load_config(_write_profile(tmp_path, None))
    assert cfg.llm_review.estimated_rates_path is None
    inline = cfg.llm_review.error_rates_by_band
    assert inline["grey_mid"] == {"e_fp": 0.06, "e_fn": 0.10}
    for mode in MODES:
        assert cfg.llm_review.error_rates_for_mode(mode) == inline
    engine = build_engine_from_config(cfg, mode="vigilancia")
    assert engine.llm_error_by_band["high"].e_fn == 0.03
    assert engine.llm_error_by_band["high"].e_unsure == 0.0


def test_reference_fixture_with_dangling_path_still_loads_inline():
    """The v3.1 reference fixture names a rates file that is not shipped with
    it; loading stays lazy so only consumers of the measured rates need it."""
    cfg = load_config(FIXTURES / "gzcmd_reference_config_v31.yaml")
    assert cfg.llm_review.estimated_rates_path is not None
    assert cfg.llm_review.error_rates_by_band["grey_low"]["e_fp"] == 0.08


def test_packaged_v4_yaml_is_verbatim_thesis_copy():
    default_dir = PACKAGE_DIR.parents[2] / "phd-research"  # sibling checkout
    thesis_dir = Path(os.environ.get("GZCMD_THESIS_DIR", default_dir))
    source = thesis_dir / "gzcmd" / "error_rates_by_band_v4.yaml"
    if not source.is_file():
        pytest.skip(f"thesis repository not available: {source}")
    packaged = PACKAGE_DIR / "error_rates_by_band_v4.yaml"
    assert packaged.read_bytes() == source.read_bytes()
