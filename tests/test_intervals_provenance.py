"""Wilson interval and normalised provenance hash: reference values and parity."""

from __future__ import annotations

import hashlib
from math import sqrt
from pathlib import Path

import pytest
from _thesis_eval import load_thesis_eval, requires_thesis_eval

from gzcmd_record_linkage.intervals import wilson95
from gzcmd_record_linkage.provenance import sha256_file_normalized, sha256_normalized


def _reference_wilson(k: int, n: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def test_wilson_reference_values() -> None:
    assert wilson95(50, 100) == pytest.approx((0.4038315303659956, 0.5961684696340044))
    assert wilson95(203, 247) == pytest.approx((0.769327, 0.864540), abs=1e-6)
    for k, n in ((1, 2), (3, 10), (7, 9), (120, 1000)):
        assert wilson95(k, n) == pytest.approx(_reference_wilson(k, n))


def test_wilson_boundaries_are_exact_and_inside_unit_interval() -> None:
    assert wilson95(0, 36)[0] == 0
    assert wilson95(36, 36)[1] == 1
    low, high = wilson95(0, 36)
    assert 0 < high < 1
    low, high = wilson95(36, 36)
    assert 0 < low < 1


@pytest.mark.parametrize("successes,total", [(1, 0), (0, 0), (37, 36), (-1, 5)])
def test_wilson_rejects_invalid_counts(successes: int, total: int) -> None:
    with pytest.raises(ValueError, match="Wilson interval requires"):
        wilson95(successes, total)


def test_sha256_normalizes_crlf_to_lf_only() -> None:
    lf = b'{"a": 1}\n{"b": 2}\n'
    crlf = lf.replace(b"\n", b"\r\n")
    assert sha256_normalized(crlf) == sha256_normalized(lf)
    assert sha256_normalized(lf) == hashlib.sha256(lf).hexdigest()
    # A lone CR is not a CRLF and is preserved; a CRLF is rewritten once.
    assert sha256_normalized(b"a\rb") == hashlib.sha256(b"a\rb").hexdigest()
    assert sha256_normalized(b"a\r\r\nb") == hashlib.sha256(b"a\r\nb").hexdigest()
    assert sha256_normalized(b"") == hashlib.sha256(b"").hexdigest()


def test_sha256_file_normalized_is_line_ending_independent(tmp_path: Path) -> None:
    lf, crlf = tmp_path / "lf.csv", tmp_path / "crlf.csv"
    lf.write_bytes(b"x,y\n1,2\n")
    crlf.write_bytes(b"x,y\r\n1,2\r\n")
    assert sha256_file_normalized(lf) == sha256_file_normalized(crlf)
    assert sha256_file_normalized(str(crlf)) == sha256_normalized(b"x,y\n1,2\n")
    with pytest.raises(FileNotFoundError):
        sha256_file_normalized(tmp_path / "absent.csv")


@requires_thesis_eval
def test_thesis_parity(tmp_path: Path) -> None:
    thesis = load_thesis_eval()
    for n in (1, 2, 36, 247, 1000):
        for k in sorted({0, 1, n // 3, n // 2, n - 1, n}):
            assert wilson95(k, n) == thesis.intervals.wilson95(k, n)
    for data in (b"", b"a\r\nb\r\n", b"a\rb\n", b"\r\n\r\n", b"x" * 1000):
        assert sha256_normalized(data) == thesis.provenance.sha256_normalized(data)
    path = tmp_path / "artifact.json"
    path.write_bytes(b'{"k": 1}\r\n')
    assert sha256_file_normalized(path) == thesis.provenance.sha256_file_normalized(
        path
    )
