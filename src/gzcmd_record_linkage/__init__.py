"""GZ-CMD++ research implementation.

This package contains a small, deterministic reference implementation of the
"grey-zone clerical review" policy described in the thesis work.
"""

from __future__ import annotations

from importlib import metadata

__all__ = ["__version__"]


def _package_version() -> str:
    try:
        return metadata.version("gzcmd-record-linkage")
    except metadata.PackageNotFoundError:
        return "0.1.0"


__version__ = _package_version()
