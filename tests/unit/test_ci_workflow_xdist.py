"""CI workflow invariants for pytest parallel execution."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from elspeth.testing.pytest_xdist_auto import pytest_cmdline_main

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yaml"


def _ci_workflow() -> dict[str, Any]:
    raw = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(raw, dict), "CI workflow YAML root must be a mapping"
    return raw


def _step_run(job: dict[str, Any], step_name: str) -> str:
    for step in job["steps"]:
        if step.get("name") == step_name:
            run = step.get("run")
            assert isinstance(run, str), f"{step_name!r} must have a shell run block"
            return run
    raise AssertionError(f"Missing CI step {step_name!r}")


def test_python_matrix_ci_uses_xdist_for_remote_test_runtime() -> None:
    """Remote Python test lanes must leave xdist available instead of disabling it."""
    workflow = _ci_workflow()
    test_job = workflow["jobs"]["test"]

    coverage_run = _step_run(test_job, "Run tests with coverage")
    no_coverage_run = _step_run(test_job, "Run tests without coverage")

    assert "-n0" not in coverage_run
    assert "--numprocesses=0" not in coverage_run
    assert "-n0" not in no_coverage_run
    assert "--numprocesses=0" not in no_coverage_run


def test_xdist_auto_defaults_to_parallel_in_ci_controller(monkeypatch: pytest.MonkeyPatch) -> None:
    """CI controllers should get the same xdist default as local runs."""
    monkeypatch.setenv("CI", "true")
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    config = SimpleNamespace(option=SimpleNamespace(numprocesses=None))

    pytest_cmdline_main(config)  # type: ignore[arg-type]

    assert config.option.numprocesses == "auto"


def test_xdist_auto_preserves_explicit_process_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit pytest ``-n`` choices remain authoritative."""
    monkeypatch.setenv("CI", "true")
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    config = SimpleNamespace(option=SimpleNamespace(numprocesses=0))

    pytest_cmdline_main(config)  # type: ignore[arg-type]

    assert config.option.numprocesses == 0


def test_xdist_auto_stays_sequential_for_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Coverage jobs stay sequential because pytest-cov owns worker coordination."""
    monkeypatch.setenv("CI", "true")
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    config = SimpleNamespace(option=SimpleNamespace(cov_source=["src/elspeth"], numprocesses=None))

    pytest_cmdline_main(config)  # type: ignore[arg-type]

    assert config.option.numprocesses is None


def test_xdist_auto_noops_inside_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Workers must not recursively auto-enable xdist."""
    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw0")
    config = SimpleNamespace(option=SimpleNamespace(numprocesses=None))

    pytest_cmdline_main(config)  # type: ignore[arg-type]

    assert config.option.numprocesses is None


def test_postgres_testcontainer_ci_stays_sequential() -> None:
    """The Docker-backed lane is intentionally not fanned out across xdist workers."""
    workflow = _ci_workflow()
    testcontainer_job = workflow["jobs"]["test-postgres-testcontainer"]

    run = _step_run(testcontainer_job, "Run testcontainer tests")

    assert "-n auto" not in run
