"""Tests for version consistency."""

import tomllib
from pathlib import Path

import pytest


def test_version_matches_pyproject() -> None:
    """elspeth.__version__ must match pyproject.toml — single source of truth.

    Bug: elspeth-1b77a953e7 — hard-coded __version__ drifted from pyproject.toml.
    Fix: __version__ now reads from importlib.metadata at import time.
    """
    import elspeth

    pyproject = Path(__file__).parents[2] / "pyproject.toml"
    with open(pyproject, "rb") as f:
        data = tomllib.load(f)

    expected = data["project"]["version"]
    if elspeth.__version__ == "UNKNOWN-dev-uninstalled":
        pytest.skip(
            "Package not installed (importlib.metadata fallback active). Run 'uv pip install -e .' to enable version consistency check."
        )
    assert elspeth.__version__ == expected
