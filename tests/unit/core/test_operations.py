"""Unit tests for track_operation lifecycle behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pytest

from elspeth.core.landscape.execution_repository import ExecutionRepository
from elspeth.core.operations import track_operation
from tests.fixtures.factories import make_context

_REDACTED = "<redacted-secret>"


@dataclass
class _Operation:
    operation_id: str


class _FakeFactory:
    def __init__(self, *, complete_error: Exception | None = None) -> None:
        self.begin_calls: list[dict[str, Any]] = []
        self.complete_calls: list[dict[str, Any]] = []
        self._complete_error = complete_error

    def begin_operation(
        self,
        *,
        run_id: str,
        node_id: str,
        operation_type: str,
        input_data: dict[str, Any] | None = None,
    ) -> _Operation:
        self.begin_calls.append(
            {
                "run_id": run_id,
                "node_id": node_id,
                "operation_type": operation_type,
                "input_data": input_data,
            }
        )
        return _Operation(operation_id="op-001")

    def complete_operation(
        self,
        *,
        operation_id: str,
        status: str,
        output_data: dict[str, Any] | None,
        error: str | None,
        duration_ms: float,
    ) -> None:
        self.complete_calls.append(
            {
                "operation_id": operation_id,
                "status": status,
                "output_data": output_data,
                "error": error,
                "duration_ms": duration_ms,
            }
        )
        if self._complete_error is not None:
            raise self._complete_error


def test_track_operation_records_completed_status_and_output_data() -> None:
    factory = _FakeFactory()
    ctx = make_context()
    assert ctx.operation_id is None

    with track_operation(
        recorder=cast(ExecutionRepository, factory),
        run_id="run-001",
        node_id="node-001",
        operation_type="source_load",
        ctx=ctx,
        input_data={"source": "csv"},
    ) as handle:
        assert ctx.operation_id == "op-001"
        handle.output_data = {"rows_loaded": 3}

    assert ctx.operation_id is None
    assert factory.begin_calls[0]["input_data"] == {"source": "csv"}
    assert factory.complete_calls[0]["status"] == "completed"
    assert factory.complete_calls[0]["output_data"] == {"rows_loaded": 3}


def test_track_operation_scrubs_structured_input_metadata_before_persisting() -> None:
    factory = _FakeFactory()
    ctx = make_context()
    secret = "sk-or-v1-" + ("B" * 24)

    with track_operation(
        recorder=cast(ExecutionRepository, factory),
        run_id="run-001",
        node_id="node-001",
        operation_type="source_load",
        ctx=ctx,
        input_data={
            "source": "api",
            "headers": {"Authorization": f"Bearer {secret}"},
            "nested": [{"password": "operator-password"}],
        },
    ):
        pass

    persisted_input = factory.begin_calls[0]["input_data"]
    assert secret not in repr(persisted_input)
    assert "operator-password" not in repr(persisted_input)
    assert persisted_input == {
        "source": "api",
        "headers": {"Authorization": _REDACTED},
        "nested": [{"password": _REDACTED}],
    }


def test_track_operation_scrubs_structured_output_metadata_before_persisting() -> None:
    factory = _FakeFactory()
    ctx = make_context()
    token = "sk-" + ("C" * 24)

    with track_operation(
        recorder=cast(ExecutionRepository, factory),
        run_id="run-001",
        node_id="node-001",
        operation_type="sink_write",
        ctx=ctx,
    ) as handle:
        handle.output_data = {
            "artifact_uri": f"https://user:{token}@storage.example.com/out.csv",
            "rows_written": 3,
        }

    persisted_output = factory.complete_calls[0]["output_data"]
    assert token not in repr(persisted_output)
    assert persisted_output == {
        "artifact_uri": _REDACTED,
        "rows_written": 3,
    }


def test_track_operation_marks_failed_for_exception() -> None:
    factory = _FakeFactory()
    ctx = make_context()

    with (
        pytest.raises(ValueError, match="boom"),
        track_operation(
            recorder=cast(ExecutionRepository, factory),
            run_id="run-001",
            node_id="node-001",
            operation_type="source_load",
            ctx=ctx,
        ),
    ):
        raise ValueError("boom")

    assert factory.complete_calls[0]["status"] == "failed"
    assert factory.complete_calls[0]["error"] == "boom"
    assert ctx.operation_id is None


def test_track_operation_marks_failed_for_base_exception() -> None:
    class _Fatal(BaseException):
        pass

    factory = _FakeFactory()
    ctx = make_context()

    with (
        pytest.raises(_Fatal, match="stop-now"),
        track_operation(
            recorder=cast(ExecutionRepository, factory),
            run_id="run-001",
            node_id="node-001",
            operation_type="source_load",
            ctx=ctx,
        ),
    ):
        raise _Fatal("stop-now")

    assert factory.complete_calls[0]["status"] == "failed"
    assert factory.complete_calls[0]["error"] == "stop-now"
    assert ctx.operation_id is None


@pytest.mark.parametrize(
    ("exc", "expected_error"),
    [
        (KeyboardInterrupt(), "KeyboardInterrupt"),
        (SystemExit(), "SystemExit"),
    ],
)
def test_track_operation_renders_informative_error_for_blank_base_exception_strings(
    exc: BaseException,
    expected_error: str,
) -> None:
    factory = _FakeFactory()
    ctx = make_context()

    with (
        pytest.raises(type(exc)),
        track_operation(
            recorder=cast(ExecutionRepository, factory),
            run_id="run-001",
            node_id="node-001",
            operation_type="source_load",
            ctx=ctx,
        ),
    ):
        raise exc

    assert factory.complete_calls[0]["status"] == "failed"
    assert factory.complete_calls[0]["error"] == expected_error
    assert ctx.operation_id is None


def test_track_operation_falls_back_for_broken_exception_str_override() -> None:
    class _BrokenStrError(Exception):
        def __str__(self) -> str:
            raise RuntimeError("boom")

    factory = _FakeFactory()
    ctx = make_context()

    with (
        pytest.raises(_BrokenStrError),
        track_operation(
            recorder=cast(ExecutionRepository, factory),
            run_id="run-001",
            node_id="node-001",
            operation_type="source_load",
            ctx=ctx,
        ),
    ):
        raise _BrokenStrError("original message")

    assert factory.complete_calls[0]["status"] == "failed"
    assert factory.complete_calls[0]["error"] == "_BrokenStrError"
    assert ctx.operation_id is None


def test_track_operation_scrubs_secret_bearing_error_before_persisting() -> None:
    """A provider exception that embeds an API key must not reach the audit trail.

    ``track_operation`` persists ``operations.error_message`` from the string it
    hands to ``complete_operation``. Runtime-preflight failures interpolate the
    underlying client error (which can carry a bearer/API key) into their
    message, so that string MUST pass through the audit scrubber here — at the
    single ``_render_exception`` chokepoint — not merely in the structured
    ``to_audit_dict()`` payload (which this persistence path never consults).
    """
    from elspeth.contracts.errors import RuntimePreflightFailedError

    factory = _FakeFactory()
    ctx = make_context()
    # Build the key-shaped value at runtime so the source carries no literal
    # secret for the repo secret-scan hook to flag (cf. tests/unit/contracts/
    # test_errors.py); the runtime value still matches an audit-scrub pattern.
    secret = "sk-or-v1-" + ("A" * 24)
    cause = RuntimeError(f"401 unauthorized (Authorization: Bearer {secret})")
    preflight_error = RuntimePreflightFailedError(plugin_name="llm", provider="openrouter", cause=cause)
    # Sanity: the raw, unscrubbed message really does carry the secret.
    assert secret in str(preflight_error)

    with (
        pytest.raises(RuntimePreflightFailedError),
        track_operation(
            recorder=cast(ExecutionRepository, factory),
            run_id="run-001",
            node_id="node-001",
            operation_type="runtime_preflight",
            ctx=ctx,
        ),
    ):
        raise preflight_error

    persisted_error = factory.complete_calls[0]["error"]
    assert factory.complete_calls[0]["status"] == "failed"
    assert secret not in persisted_error
    assert persisted_error == "<redacted-secret>"


def test_track_operation_raises_db_error_if_completion_fails_after_success() -> None:
    factory = _FakeFactory(complete_error=RuntimeError("db write failed"))
    ctx = make_context()

    with (
        pytest.raises(RuntimeError, match="db write failed"),
        track_operation(
            recorder=cast(ExecutionRepository, factory),
            run_id="run-001",
            node_id="node-001",
            operation_type="sink_write",
            ctx=ctx,
        ),
    ):
        pass

    assert factory.complete_calls[0]["status"] == "completed"
    assert ctx.operation_id is None


def test_track_operation_does_not_mask_original_exception_when_completion_fails() -> None:
    factory = _FakeFactory(complete_error=RuntimeError("db write failed"))
    ctx = make_context()

    with (
        pytest.raises(ValueError, match="original failure"),
        track_operation(
            recorder=cast(ExecutionRepository, factory),
            run_id="run-001",
            node_id="node-001",
            operation_type="source_load",
            ctx=ctx,
        ),
    ):
        raise ValueError("original failure")

    assert factory.complete_calls[0]["status"] == "failed"
    assert factory.complete_calls[0]["error"] == "original failure"
    assert ctx.operation_id is None


def test_track_operation_reraises_framework_bug_error_even_with_original_exception() -> None:
    """FrameworkBugError from complete_operation() must supersede any original exception.

    Tier 1 violations indicate audit corruption — categorically worse than
    whatever the operation body was doing.
    """
    from elspeth.contracts import FrameworkBugError

    factory = _FakeFactory(complete_error=FrameworkBugError("audit corruption"))
    ctx = make_context()

    with (
        pytest.raises(FrameworkBugError, match="audit corruption"),
        track_operation(
            recorder=cast(ExecutionRepository, factory),
            run_id="run-001",
            node_id="node-001",
            operation_type="source_load",
            ctx=ctx,
        ),
    ):
        raise ValueError("original failure")

    assert ctx.operation_id is None


def test_track_operation_reraises_audit_integrity_error_even_with_original_exception() -> None:
    """AuditIntegrityError from complete_operation() must supersede any original exception."""
    from elspeth.contracts.errors import AuditIntegrityError

    factory = _FakeFactory(complete_error=AuditIntegrityError("DB corrupted"))
    ctx = make_context()

    with (
        pytest.raises(AuditIntegrityError, match="DB corrupted"),
        track_operation(
            recorder=cast(ExecutionRepository, factory),
            run_id="run-001",
            node_id="node-001",
            operation_type="sink_write",
            ctx=ctx,
        ),
    ):
        raise RuntimeError("operation error")

    assert ctx.operation_id is None


def test_track_operation_tier1_error_chains_original_exception() -> None:
    """When Tier 1 error supersedes, the original exception is chained via __cause__."""
    from elspeth.contracts import FrameworkBugError

    factory = _FakeFactory(complete_error=FrameworkBugError("corruption"))
    ctx = make_context()

    with (
        pytest.raises(FrameworkBugError) as exc_info,
        track_operation(
            recorder=cast(ExecutionRepository, factory),
            run_id="run-001",
            node_id="node-001",
            operation_type="source_load",
            ctx=ctx,
        ),
    ):
        raise ValueError("original cause")

    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert str(exc_info.value.__cause__) == "original cause"
