"""Tests for orchestrator facade import behavior."""

from __future__ import annotations

import subprocess
import sys


def test_orchestrator_facade_type_import_does_not_load_runtime_core() -> None:
    """Importing type exports from the facade should not bootstrap runtime modules."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys

from elspeth.engine.orchestrator import PipelineConfig

blocked_modules = [
    name
    for name in (
        "elspeth.engine.orchestrator.core",
        "elspeth.engine.executors.declaration_contract_bootstrap",
    )
    if name in sys.modules
]
if blocked_modules:
    print(f"FAIL: runtime modules loaded: {blocked_modules}")
    sys.exit(1)
if PipelineConfig.__module__ != "elspeth.engine.orchestrator.types":
    print(f"FAIL: unexpected PipelineConfig module: {PipelineConfig.__module__}")
    sys.exit(1)
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Facade import loaded runtime modules:\n{result.stdout}\n{result.stderr}"
