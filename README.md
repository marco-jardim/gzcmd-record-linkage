# GZ-CMD-Record-Linkage

Reference implementation of the GZ-CMD++ v3 policy engine, packaged as a reusable
Python package with `src` layout and CLI entry points.

## Package

- PyPI package name: `gzcmd-record-linkage`
- Import package: `gzcmd_record_linkage`

## Install

- Editable install for development:

```bash
python -m pip install -e .[dev]
```

- Runtime only:

```bash
python -m pip install -e .
```

The package version is derived from git tags using `hatch-vcs`.

## Development

- Lint:

```bash
ruff check src/gzcmd_record_linkage/cli.py tests
```

- Tests with coverage target:

```bash
pytest --cov=gzcmd_record_linkage.loader \
  --cov=gzcmd_record_linkage.classifier \
  --cov=gzcmd_record_linkage.bands \
  --cov=gzcmd_record_linkage.guardrails \
  --cov-fail-under=80
```

- Build:

```bash
python -m build
```

## CLI

The package exposes a `gzcmd` CLI entry point.

```bash
gzcmd --help
```

You can also invoke the module directly:

```bash
python -m gzcmd_record_linkage --help
```
