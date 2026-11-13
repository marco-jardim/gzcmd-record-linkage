"""Platform-independent provenance hash.

JSON/CSV artefacts are normalised by git to LF at commit time, but on Windows
they may be CRLF on disk. The hash recorded in the estimated-rates YAML must be
the same in both cases, so it is computed over the content with line
terminators normalised to LF (only ``CRLF`` is rewritten; a lone ``CR`` is kept).
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_normalized(data: bytes) -> str:
    """SHA-256 hex digest of ``data`` after replacing ``CRLF`` with ``LF``."""
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def sha256_file_normalized(path: str | Path) -> str:
    """:func:`sha256_normalized` of the bytes of the file at ``path``."""
    return sha256_normalized(Path(path).read_bytes())
