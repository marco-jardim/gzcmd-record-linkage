from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from gzcmd_record_linkage import cli


def test_build_parser_includes_core_commands() -> None:
    parser = cli.build_parser()
    help_text = parser.format_help()

    assert "run" in help_text
    assert "fit-calibration" in help_text
    assert "eval" in help_text


def test_default_config_path_is_packaged_file() -> None:
    cfg = Path(cli._default_config_path())
    assert cfg.name == "gzcmd_v3_config.yaml"
    assert cfg.exists()


def test_cli_module_help_smoke() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "gzcmd_record_linkage", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "GZ-CMD++ v3 runner" in result.stdout


def test_entrypoint_script_exists() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "gzcmd_record_linkage", "run", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--input" in result.stdout
