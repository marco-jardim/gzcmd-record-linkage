"""Limite de CPU para a suite de testes (somente biblioteca padrão).

Carregado por ``conftest.py`` (raiz do repositório) antes de qualquer módulo de
teste importar numpy, scikit-learn ou xgboost. Não altera nenhum padrão da
biblioteca (``n_jobs``, ``THESIS_XGB_PARAMS``, ``ClassifierConfig``): afinidade e
prioridade limitam o uso de CPU sem mudar a numérica.

Variáveis de ambiente:

* ``PYTEST_CPU_LIMIT`` = ``0``/``false``/``off``: desativa tudo.
* ``PYTEST_MAX_CPUS`` = inteiro >= 1: CPUs lógicas permitidas
  (padrão: ``max(1, os.cpu_count() // 2)``).
* ``CI`` definida: aplica só as variáveis de threads (runners em velocidade total).
"""

from __future__ import annotations

import os
import sys
import warnings
from collections.abc import MutableMapping

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_DISABLED_VALUES = {"0", "false", "off"}
_BELOW_NORMAL_PRIORITY_CLASS = 0x4000
_IDLE_PRIORITY_CLASS = 0x40
_NORMAL_OR_ABOVE = (0x20, 0x8000, 0x80, 0x100)  # normal, above, high, realtime
_POSIX_NICE_TARGET = 10


def _warn(message: str) -> None:
    warnings.warn(f"gzcmd cpu_limit: {message}", RuntimeWarning, stacklevel=3)


def is_disabled(environ: MutableMapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return env.get("PYTEST_CPU_LIMIT", "").strip().lower() in _DISABLED_VALUES


def is_ci(environ: MutableMapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return bool(env.get("CI"))


def resolve_max_cpus(environ: MutableMapping[str, str] | None = None) -> int:
    """N de ``PYTEST_MAX_CPUS`` (>= 1) ou metade das CPUs lógicas (mínimo 1)."""
    env = os.environ if environ is None else environ
    default = max(1, (os.cpu_count() or 1) // 2)
    raw = env.get("PYTEST_MAX_CPUS")
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < 1:
        _warn(
            f"PYTEST_MAX_CPUS={raw!r} inválido (esperado inteiro >= 1); "
            f"usando {default}"
        )
        return default
    return value


# --------------------------------------------------------------------- Windows
def _win_kernel32():
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.GetCurrentProcess.argtypes = []
    k32.GetPriorityClass.restype = wintypes.DWORD
    k32.GetPriorityClass.argtypes = [wintypes.HANDLE]
    k32.SetPriorityClass.restype = wintypes.BOOL
    k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    k32.GetProcessAffinityMask.restype = wintypes.BOOL
    k32.GetProcessAffinityMask.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    k32.SetProcessAffinityMask.restype = wintypes.BOOL
    k32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    return k32


def _win_affinity_masks(k32) -> tuple[int, int] | None:
    import ctypes

    proc = ctypes.c_size_t(0)
    system = ctypes.c_size_t(0)
    if not k32.GetProcessAffinityMask(
        k32.GetCurrentProcess(), ctypes.byref(proc), ctypes.byref(system)
    ):
        _warn(f"GetProcessAffinityMask falhou (erro Windows {ctypes.get_last_error()})")
        return None
    return proc.value, system.value


def _lowest_bits(mask: int, n: int) -> int:
    out = 0
    for bit in range(mask.bit_length()):
        if mask & (1 << bit):
            out |= 1 << bit
            n -= 1
            if n == 0:
                break
    return out


def _set_priority_windows() -> bool:
    import ctypes

    k32 = _win_kernel32()
    handle = k32.GetCurrentProcess()
    current = k32.GetPriorityClass(handle)
    if current == 0:
        _warn(f"GetPriorityClass falhou (erro Windows {ctypes.get_last_error()})")
        return False
    if current not in _NORMAL_OR_ABOVE:
        return False  # já é BelowNormal/Idle (ou erro tratado abaixo)
    if not k32.SetPriorityClass(handle, _BELOW_NORMAL_PRIORITY_CLASS):
        _warn(f"SetPriorityClass falhou (erro Windows {ctypes.get_last_error()})")
        return False
    return True


def _set_affinity_windows(n: int) -> bool:
    import ctypes

    k32 = _win_kernel32()
    masks = _win_affinity_masks(k32)
    if masks is None:
        return False
    proc_mask, system_mask = masks
    # Parte da máscara atual do processo (= a do sistema, salvo se o pai já a
    # restringiu); assim a aplicação é idempotente e nunca amplia o conjunto.
    base = (proc_mask & system_mask) or system_mask
    if bin(base).count("1") <= n:
        return False
    if not k32.SetProcessAffinityMask(k32.GetCurrentProcess(), _lowest_bits(base, n)):
        _warn(f"SetProcessAffinityMask falhou (erro Windows {ctypes.get_last_error()})")
        return False
    return True


# ----------------------------------------------------------------- consultas
def current_affinity_count() -> int | None:
    """Nº de CPUs permitidas ao processo atual; ``None`` onde não há API."""
    if sys.platform == "win32":
        k32 = _win_kernel32()
        masks = _win_affinity_masks(k32)
        if masks is None:
            return None
        return bin(masks[0]).count("1")
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return None


def is_below_normal_priority() -> bool | None:
    """Prioridade abaixo do normal (Windows) ou nice >= 10 (POSIX)."""
    if sys.platform == "win32":
        k32 = _win_kernel32()
        value = k32.GetPriorityClass(k32.GetCurrentProcess())
        return value in (_BELOW_NORMAL_PRIORITY_CLASS, _IDLE_PRIORITY_CLASS)
    if hasattr(os, "nice"):
        return os.nice(0) >= _POSIX_NICE_TARGET
    return None


# ------------------------------------------------------------------ aplicação
def apply_cpu_limit(
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, object]:
    """Aplica o limite; devolve um relato do que foi feito.

    Afinidade é do processo inteiro: workers do xdist herdam a máscara do pai e
    reaplicam o mesmo corte (idempotente), então juntos permanecem nas mesmas N CPUs.
    """
    env = os.environ if environ is None else environ
    report: dict[str, object] = {
        "disabled": False,
        "ci": False,
        "max_cpus": None,
        "priority": False,
        "affinity": False,
    }
    if is_disabled(env):
        report["disabled"] = True
        return report
    n = resolve_max_cpus(env)
    report["max_cpus"] = n
    for name in THREAD_ENV_VARS:
        env.setdefault(name, str(n))  # setdefault: o override do usuário vence
    if is_ci(env):
        report["ci"] = True
        return report

    try:
        if sys.platform == "win32":
            report["priority"] = _set_priority_windows()
        elif hasattr(os, "nice"):
            current = os.nice(0)
            if current < _POSIX_NICE_TARGET:
                os.nice(_POSIX_NICE_TARGET - current)
                report["priority"] = True
    except (OSError, AttributeError, ValueError) as exc:
        _warn(f"não foi possível baixar a prioridade: {exc!r}")

    try:
        if sys.platform == "win32":
            report["affinity"] = _set_affinity_windows(n)
        elif hasattr(os, "sched_setaffinity"):
            cpus = sorted(os.sched_getaffinity(0))
            if len(cpus) > n:
                os.sched_setaffinity(0, set(cpus[:n]))
                report["affinity"] = True
        # macOS: sem API de afinidade; nada a fazer.
    except (OSError, AttributeError, ValueError) as exc:
        _warn(f"não foi possível restringir a afinidade: {exc!r}")
    return report
