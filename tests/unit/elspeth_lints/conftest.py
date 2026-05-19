"""Shared fixtures for elspeth-lints unit tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture
def elspeth_lints_subprocess_env() -> dict[str, str]:
    """Environment for subprocesses that invoke the elspeth-lints CLI."""
    project_root = Path(__file__).resolve().parents[3]
    pythonpath_entries = [
        str(project_root / "src"),
        str(project_root / "elspeth-lints" / "src"),
    ]
    existing_pythonpath = os.environ.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(pythonpath_entries),
    }
