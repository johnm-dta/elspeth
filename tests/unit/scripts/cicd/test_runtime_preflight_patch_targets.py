"""Persistent guard for runtime-preflight test patch targets."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
EXECUTION_SERVICE_TEST = PROJECT_ROOT / "tests" / "unit" / "web" / "execution" / "test_service.py"
STALE_RUNTIME_GRAPH_PATCH_TARGETS = (
    "elspeth.web.execution.service.ExecutionGraph",
    "elspeth.web.execution.service.instantiate_plugins_from_config",
)


def test_execution_service_tests_do_not_patch_old_runtime_graph_symbols() -> None:
    text = EXECUTION_SERVICE_TEST.read_text(encoding="utf-8")
    stale_targets = [target for target in STALE_RUNTIME_GRAPH_PATCH_TARGETS if target in text]

    assert stale_targets == [], (
        f"Patch runtime graph setup through elspeth.web.execution.preflight, not elspeth.web.execution.service: {stale_targets}"
    )
