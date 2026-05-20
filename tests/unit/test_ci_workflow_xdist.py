"""CI workflow invariants for pytest parallel execution."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from elspeth.testing.pytest_xdist_auto import pytest_cmdline_main


def test_xdist_auto_defaults_to_parallel_locally(monkeypatch: pytest.MonkeyPatch) -> None:
    """Local pytest runs default to xdist when no process count is explicit."""
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    config = SimpleNamespace(option=SimpleNamespace(numprocesses=None))

    pytest_cmdline_main(config)  # type: ignore[arg-type]

    assert config.option.numprocesses == "auto"


def test_xdist_auto_noops_in_ci(monkeypatch: pytest.MonkeyPatch) -> None:
    """CI controllers stay sequential for clearer failure output."""
    monkeypatch.setenv("CI", "1")
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    config = SimpleNamespace(option=SimpleNamespace(numprocesses=None))

    pytest_cmdline_main(config)  # type: ignore[arg-type]

    assert config.option.numprocesses is None


def test_xdist_auto_noops_inside_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Workers must not recursively auto-enable xdist."""
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw0")
    config = SimpleNamespace(option=SimpleNamespace(numprocesses=None))

    pytest_cmdline_main(config)  # type: ignore[arg-type]

    assert config.option.numprocesses is None


def test_xdist_auto_preserves_explicit_process_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit pytest ``-n`` choices remain authoritative."""
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    config = SimpleNamespace(option=SimpleNamespace(numprocesses=4))

    pytest_cmdline_main(config)  # type: ignore[arg-type]

    assert config.option.numprocesses == 4
