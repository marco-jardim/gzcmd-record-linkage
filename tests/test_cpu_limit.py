"""Testes do limite de CPU (conftest.py da raiz + tests/_cpu_limit.py)."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

_HELPER = Path(__file__).resolve().parent / "_cpu_limit.py"


def _module():
    mod = sys.modules.get("_cpu_limit")
    if mod is None:
        spec = importlib.util.spec_from_file_location("_cpu_limit", _HELPER)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_cpu_limit"] = mod
        spec.loader.exec_module(mod)
    return mod


cpu = _module()

_DISABLED = cpu.is_disabled()
_CI = cpu.is_ci()
_AFFINITY_SUPPORTED = cpu.current_affinity_count() is not None
_PRIORITY_SUPPORTED = cpu.is_below_normal_priority() is not None


def test_resolve_max_cpus_default_is_half_of_logical_cpus():
    assert cpu.resolve_max_cpus({}) == max(1, (os.cpu_count() or 1) // 2)


def test_resolve_max_cpus_env_override():
    assert cpu.resolve_max_cpus({"PYTEST_MAX_CPUS": "3"}) == 3


@pytest.mark.parametrize("bad", ["0", "-2", "abc"])
def test_resolve_max_cpus_invalid_warns_and_falls_back(bad):
    with pytest.warns(RuntimeWarning, match="PYTEST_MAX_CPUS"):
        n = cpu.resolve_max_cpus({"PYTEST_MAX_CPUS": bad})
    assert n == max(1, (os.cpu_count() or 1) // 2)


@pytest.mark.parametrize("value", ["0", "false", "off", "OFF"])
def test_opt_out_leaves_environ_untouched(value):
    env = {"PYTEST_CPU_LIMIT": value}
    report = cpu.apply_cpu_limit(env)
    assert report["disabled"] is True
    assert env == {"PYTEST_CPU_LIMIT": value}


def test_ci_applies_only_thread_vars():
    env = {"CI": "true", "PYTEST_MAX_CPUS": "2"}
    report = cpu.apply_cpu_limit(env)
    assert report["ci"] is True
    assert report["priority"] is False and report["affinity"] is False
    assert all(env[name] == "2" for name in cpu.THREAD_ENV_VARS)


def test_thread_vars_do_not_override_user_values():
    env = {"CI": "1", "PYTEST_MAX_CPUS": "2", "OMP_NUM_THREADS": "7"}
    cpu.apply_cpu_limit(env)
    assert env["OMP_NUM_THREADS"] == "7"
    assert env["MKL_NUM_THREADS"] == "2"


@pytest.mark.skipif(_DISABLED, reason="PYTEST_CPU_LIMIT desativa o limite")
def test_thread_env_vars_are_set():
    for name in cpu.THREAD_ENV_VARS:
        assert os.environ.get(name, "").isdigit(), name


@pytest.mark.skipif(
    _DISABLED or _CI, reason="limite de afinidade não se aplica (opt-out ou CI)"
)
@pytest.mark.skipif(
    not _AFFINITY_SUPPORTED, reason="sem API de afinidade nesta plataforma"
)
def test_process_affinity_is_capped():
    n = cpu.resolve_max_cpus()
    count = cpu.current_affinity_count()
    print(f"affinity: {count} CPU(s) permitidas (cap={n}, logical={os.cpu_count()})")
    assert 1 <= count <= n


@pytest.mark.skipif(_DISABLED or _CI, reason="prioridade não se aplica (opt-out ou CI)")
@pytest.mark.skipif(not _PRIORITY_SUPPORTED, reason="sem controle de prioridade")
def test_process_priority_is_below_normal():
    assert cpu.is_below_normal_priority() is True
    if sys.platform == "win32":
        k32 = cpu._win_kernel32()
        print(f"priority class: {k32.GetPriorityClass(k32.GetCurrentProcess()):#x}")
    else:
        print(f"nice: {os.nice(0)}")


_PROBE = """
import importlib.util, json, os, sys
spec = importlib.util.spec_from_file_location("_cpu_limit", {helper!r})
m = importlib.util.module_from_spec(spec)
sys.modules["_cpu_limit"] = m
spec.loader.exec_module(m)
report = m.apply_cpu_limit()
print(json.dumps({{
    "report": report,
    "affinity": m.current_affinity_count(),
    "below_normal": m.is_below_normal_priority(),
    "omp": os.environ.get("OMP_NUM_THREADS"),
}}))
"""


def _probe(**overrides):
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {*cpu.THREAD_ENV_VARS, "CI", "PYTEST_CPU_LIMIT", "PYTEST_MAX_CPUS"}
    }
    env.update(overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = subprocess.run(
            [sys.executable, "-c", _PROBE.format(helper=str(_HELPER))],
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_opt_out_in_subprocess_does_nothing():
    result = _probe(PYTEST_CPU_LIMIT="0", PYTEST_MAX_CPUS="1")
    assert result["report"]["disabled"] is True
    assert result["omp"] is None
    # Nada alterado: a afinidade é só a herdada deste processo.
    assert result["affinity"] == cpu.current_affinity_count()


@pytest.mark.skipif(
    not _AFFINITY_SUPPORTED, reason="sem API de afinidade nesta plataforma"
)
@pytest.mark.skipif((os.cpu_count() or 1) < 2, reason="precisa de >= 2 CPUs")
def test_enabled_in_subprocess_caps_affinity_and_priority():
    result = _probe(PYTEST_MAX_CPUS="1")
    assert result["omp"] == "1"
    assert result["report"]["max_cpus"] == 1
    assert result["affinity"] == 1
    if _PRIORITY_SUPPORTED:
        assert result["below_normal"] is True
