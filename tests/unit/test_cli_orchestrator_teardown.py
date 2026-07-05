"""Regression tests for _close_orchestrator_resources teardown masking.

elspeth-5310f58a2b: _orchestrator_context() always closed rate_limit_registry
and telemetry_manager in its finally block, unguarded. RateLimiter.close()
crosses pyrate-limiter/SQLite internals and TelemetryManager.close() re-raises
non-transport exporter close errors, so a teardown exception raised from that
finally replaced the primary exception from orchestrator.run()/resume() — the
CLI reported a cleanup failure instead of the real pipeline failure.

The teardown logic is extracted into _close_orchestrator_resources so the
masking-preservation behaviour is unit-testable without building the whole
orchestrator context.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

import elspeth.contracts.errors as contract_errors
from elspeth.cli import _close_orchestrator_resources, app


class _CloseRaises:
    def __init__(self, exc: BaseException) -> None:
        self._exc = exc
        self.closed = False

    def close(self) -> None:
        self.closed = True
        raise self._exc


class _CloseOk:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


@dataclass(slots=True)
class _LandscapeDBFactory:
    db: object

    def from_url(self, *_args: object, **_kwargs: object) -> object:
        return self.db


@dataclass(slots=True)
class _FakePayloadStore:
    base_path: Path


class _FakeOrchestrator:
    def __init__(self, run_result: object | BaseException) -> None:
        self._run_result = run_result

    def run(self, *_args: object, **_kwargs: object) -> object:
        if isinstance(self._run_result, BaseException):
            raise self._run_result
        return self._run_result


@dataclass(slots=True)
class _FakeContextManager:
    value: object

    def __enter__(self) -> object:
        return self.value

    def __exit__(self, *_exc_info: object) -> bool:
        return False


def _interactive_config() -> SimpleNamespace:
    return SimpleNamespace(
        landscape=SimpleNamespace(
            url="sqlite:///audit.db",
            dump_to_jsonl=False,
            dump_to_jsonl_path=None,
            dump_to_jsonl_fail_on_error=False,
            dump_to_jsonl_include_payloads=False,
            dump_to_jsonl_payload_base_path=None,
        ),
        payload_store=SimpleNamespace(
            backend="filesystem",
            base_path=Path(".elspeth/payloads"),
        ),
    )


@contextmanager
def _patched_interactive_execution(db: object, run_result: object | BaseException):
    ctx = SimpleNamespace(
        pipeline_config=SimpleNamespace(),
        orchestrator=_FakeOrchestrator(run_result),
    )

    with (
        patch("elspeth.core.landscape.LandscapeDB", new=_LandscapeDBFactory(db)),
        patch("elspeth.core.payload_store.FilesystemPayloadStore", new=_FakePayloadStore),
        patch("elspeth.cli._orchestrator_context", new=lambda *_args, **_kwargs: _FakeContextManager(ctx)),
        patch("elspeth.plugins.infrastructure.runtime_factory.make_sink_factory", new=lambda _config: object()),
        patch(
            "elspeth.plugins.transforms.llm.model_catalog.read_openrouter_catalog_snapshot_id",
            new=lambda: ("sha256", "test"),
        ),
    ):
        yield


def test_close_failure_does_not_mask_pending_pipeline_exception():
    """With a pipeline exception pending, close failures are suppressed (logged), not raised."""
    rate_limit_registry = _CloseRaises(RuntimeError("rate limiter teardown boom"))
    telemetry_manager = _CloseRaises(RuntimeError("telemetry close boom"))

    # Must not raise — the pending pipeline exception (param) outranks teardown errors.
    _close_orchestrator_resources(
        rate_limit_registry,
        telemetry_manager,
        pending_exc=ValueError("real pipeline failure"),
    )

    # Both resources attempted, even though the first close() raised.
    assert rate_limit_registry.closed and telemetry_manager.closed


def test_close_failure_surfaces_when_no_pending_exception():
    """With no pipeline exception pending, a teardown close failure must surface."""
    rate_limit_registry = _CloseOk()
    telemetry_manager = _CloseRaises(RuntimeError("telemetry close boom"))

    with pytest.raises(RuntimeError, match="telemetry close boom"):
        _close_orchestrator_resources(rate_limit_registry, telemetry_manager, pending_exc=None)

    # Both still attempted before the surfaced error.
    assert rate_limit_registry.closed and telemetry_manager.closed


def test_tier1_error_propagates_even_when_pipeline_exception_pending():
    """Audit-integrity errors during teardown outrank even the pending pipeline exception."""
    rate_limit_registry = _CloseRaises(contract_errors.AuditIntegrityError("audit corruption during close"))
    telemetry_manager = _CloseOk()

    with pytest.raises(contract_errors.AuditIntegrityError):
        _close_orchestrator_resources(
            rate_limit_registry,
            telemetry_manager,
            pending_exc=ValueError("real pipeline failure"),
        )


def test_none_resources_are_skipped():
    """A None resource (construction failed before assignment) is simply skipped."""
    _close_orchestrator_resources(None, None, pending_exc=None)  # must not raise


def test_interactive_pipeline_failure_is_not_masked_by_db_close_failure():
    """The interactive run wrapper must preserve the primary orchestrator error."""
    from elspeth.cli import _execute_pipeline_with_instances

    db = _CloseRaises(RuntimeError("close failed"))
    with _patched_interactive_execution(db, ValueError("pipeline failed")), pytest.raises(ValueError, match="pipeline failed"):
        _execute_pipeline_with_instances(
            _interactive_config(),
            graph=object(),
            plugins=object(),
        )


def test_interactive_success_propagates_db_close_failure():
    """A clean interactive run must not report success when db.close() failed."""
    from elspeth.cli import _execute_pipeline_with_instances

    db = _CloseRaises(RuntimeError("close failed"))
    run_result = SimpleNamespace(run_id="run-1", status="completed", rows_processed=1)
    with _patched_interactive_execution(db, run_result), pytest.raises(RuntimeError, match="close failed"):
        _execute_pipeline_with_instances(
            _interactive_config(),
            graph=object(),
            plugins=object(),
        )


def test_explain_exit_not_masked_by_db_close_failure():
    """The explain command's intended CLI exit must survive database close failure."""
    db = _CloseRaises(RuntimeError("close failed"))

    with (
        patch("elspeth.cli_helpers.resolve_database_url", new=lambda _database, _settings_path: ("sqlite:///audit.db", None)),
        patch("elspeth.cli_helpers.resolve_audit_passphrase", new=lambda _landscape_settings: None),
        patch("elspeth.cli_helpers.resolve_run_id", new=lambda _run_id, _factory: None),
        patch("elspeth.core.landscape.LandscapeDB", new=_LandscapeDBFactory(db)),
        patch("elspeth.core.landscape.factory.RecorderFactory", new=lambda _db: SimpleNamespace()),
    ):
        result = CliRunner().invoke(
            app,
            ["explain", "--run", "latest", "--row", "row-1", "--no-tui", "--database", "audit.db"],
        )

    assert result.exit_code == 1
    assert getattr(result.exception, "code", None) == 1
    assert "No runs found in database" in result.output
    assert "close failed" not in str(result.exception)


def test_cli_db_commands_use_failure_preserving_close_helper():
    """Resume, explain, and purge must not use bare db.close() teardown."""
    import elspeth.cli as cli

    for command in (cli.resume, cli.explain, cli.purge):
        source = textwrap.dedent(inspect.getsource(command))
        assert "_close_landscape_db" in source, command.__name__
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "close"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "db"
            ):
                pytest.fail(f"{command.__name__} uses bare db.close() instead of _close_landscape_db()")
