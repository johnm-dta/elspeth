"""Tests for error/reason schema contracts.

Tests for:
- ExecutionError frozen dataclass (exception, exception_type, traceback, phase)
- CoalesceFailureReason frozen dataclass (failure_reason, expected_branches, etc.)
- RoutingReason TypedDict (rule, matched_value, threshold fields)
- TransformReason TypedDict (action, fields_modified fields)
"""

import dataclasses
from collections.abc import Mapping

import pytest


class TestExecutionError:
    """Tests for ExecutionError frozen dataclass — construction, immutability, serialization."""

    def test_execution_error_is_frozen_dataclass(self) -> None:
        """ExecutionError is a frozen dataclass (immutable after construction)."""
        from elspeth.contracts import ExecutionError

        assert dataclasses.is_dataclass(ExecutionError)
        error = ExecutionError(exception="test", exception_type="ValueError")
        with pytest.raises(dataclasses.FrozenInstanceError):
            error.exception = "modified"  # type: ignore[misc]

    def test_execution_error_to_dict_required_only(self) -> None:
        """to_dict() serializes exception_type as 'type' and omits None fields."""
        from elspeth.contracts import ExecutionError

        error = ExecutionError(exception="boom", exception_type="RuntimeError")
        d = error.to_dict()
        assert d == {"exception": "boom", "type": "RuntimeError"}
        assert "traceback" not in d
        assert "phase" not in d

    def test_execution_error_to_dict_with_optionals(self) -> None:
        """to_dict() includes optional fields when set."""
        from elspeth.contracts import ExecutionError

        error = ExecutionError(
            exception="boom",
            exception_type="RuntimeError",
            traceback="Traceback ...",
            phase="flush",
        )
        d = error.to_dict()
        assert d == {
            "exception": "boom",
            "type": "RuntimeError",
            "traceback": "Traceback ...",
            "phase": "flush",
        }

    def test_execution_error_audit_payload_is_structured(self) -> None:
        """ExecutionError serializes to the stable audit payload shape."""
        from elspeth.contracts import ExecutionError

        error = ExecutionError(
            exception="Another error",
            exception_type="RuntimeError",
            traceback="Traceback (most recent call last):\n  File ...",
        )

        payload = error.to_dict()
        assert payload == {
            "exception": "Another error",
            "type": "RuntimeError",
            "traceback": "Traceback (most recent call last):\n  File ...",
        }
        assert "exception_type" not in payload

    def test_execution_error_scrubs_exception_text_for_audit(self) -> None:
        """Freeform exception text must not persist secret-bearing messages."""
        from elspeth.contracts import ExecutionError

        secret = "sk-" + ("a" * 32)
        error = ExecutionError(
            exception=f"provider failed with Authorization: Bearer {secret}",
            exception_type="RuntimeError",
        )

        assert error.exception == "<redacted-secret>"
        assert secret not in str(error.to_dict())

    def test_execution_error_scrubs_context_payload_for_audit(self) -> None:
        """Structured context must be self-scrubbed by the DTO."""
        from elspeth.contracts import ExecutionError

        secret = "sk-" + ("a" * 32)
        error = ExecutionError(
            exception="boom",
            exception_type="RuntimeError",
            context={
                "password": secret,
                "nested": {"api_key": secret},
                "tokens": [secret],
                "safe": "operator diagnostic",
            },
        )

        assert error.context is not None
        nested = error.context["nested"]
        assert isinstance(nested, Mapping)
        assert error.context["password"] == "<redacted-secret>"
        assert nested["api_key"] == "<redacted-secret>"
        assert error.context["tokens"] == ("<redacted-secret>",)
        assert error.context["safe"] == "operator diagnostic"
        assert secret not in repr(error.context)
        assert secret not in str(error.to_dict())

    def test_execution_error_traceback_redacts_only_secret_lines(self) -> None:
        """Traceback scrubbing preserves safe frame diagnostics around a secret line."""
        from elspeth.contracts import ExecutionError

        secret = "sk-" + ("a" * 32)
        traceback = (
            "Traceback (most recent call last):\n"
            '  File "worker.py", line 10, in run\n'
            "    run_step()\n"
            '  File "plugin.py", line 20, in run_step\n'
            f"    raise RuntimeError('Authorization: Bearer {secret}')\n"
            "RuntimeError: failed\n"
        )

        error = ExecutionError(
            exception="boom",
            exception_type="RuntimeError",
            traceback=traceback,
        )

        assert error.traceback == (
            "Traceback (most recent call last):\n"
            '  File "worker.py", line 10, in run\n'
            "    run_step()\n"
            '  File "plugin.py", line 20, in run_step\n'
            "<redacted-secret>\n"
            "RuntimeError: failed\n"
        )
        assert secret not in error.traceback
        assert secret not in str(error.to_dict())

    def test_execution_error_traceback_scrubs_embedded_secret_for_audit(self) -> None:
        """Traceback text must not persist secret-bearing messages in audit payloads."""
        from elspeth.contracts import ExecutionError

        secret = "sk-" + ("a" * 32)
        error = ExecutionError(
            exception="boom",
            exception_type="RuntimeError",
            traceback=f"RuntimeError: Authorization: Bearer {secret}",
        )

        assert error.traceback == "<redacted-secret>"
        assert error.to_dict()["traceback"] == "<redacted-secret>"
        assert secret not in str(error.to_dict())

    def test_execution_error_traceback_scrubs_secret_split_across_message_lines(self) -> None:
        """Traceback scrubbing does not treat newlines as secret-evasion boundaries."""
        from elspeth.contracts import ExecutionError

        traceback = (
            'Traceback (most recent call last):\n  File "worker.py", line 10, in run\nRuntimeError: Authorization:\nBearer opaque-token\n'
        )

        error = ExecutionError(
            exception="boom",
            exception_type="RuntimeError",
            traceback=traceback,
        )

        assert error.traceback == (
            'Traceback (most recent call last):\n  File "worker.py", line 10, in run\n<redacted-secret>\n<redacted-secret>\n'
        )
        assert "Authorization" not in error.traceback
        assert "opaque-token" not in error.traceback
        assert "opaque-token" not in str(error.to_dict())

    def test_execution_error_traceback_scrubs_secret_continuation_after_redacted_message_line(self) -> None:
        """Continuation lines after a redacted exception message tail are redacted."""
        from elspeth.contracts import ExecutionError

        traceback = (
            'Traceback (most recent call last):\n  File "worker.py", line 10, in run\nRuntimeError: Authorization: Bearer\nopaque-token\n'
        )

        error = ExecutionError(
            exception="boom",
            exception_type="RuntimeError",
            traceback=traceback,
        )

        assert error.traceback == (
            'Traceback (most recent call last):\n  File "worker.py", line 10, in run\n<redacted-secret>\n<redacted-secret>\n'
        )
        assert "Authorization" not in error.traceback
        assert "opaque-token" not in error.traceback
        assert "opaque-token" not in str(error.to_dict())


class TestRuntimePreflightFailedError:
    """Tests for runtime-preflight audit payloads."""

    def test_to_audit_dict_scrubs_external_provider_exception_message(self) -> None:
        from elspeth.contracts.errors import RuntimePreflightFailedError

        secret = "sk-" + ("a" * 32)
        err = RuntimePreflightFailedError(
            plugin_name="llm_transform",
            provider="openrouter",
            cause=RuntimeError(f"provider rejected request with bearer {secret}"),
        )

        assert secret in str(err), "live exception text remains useful for local debugging"
        audit = err.to_audit_dict()

        assert audit["message"] == "<redacted-secret>"
        assert secret not in audit["message"]
        assert audit["plugin_name"] == "llm_transform"
        assert audit["provider"] == "openrouter"
        assert audit["cause_type"] == "RuntimeError"


class TestWriteLockHeldError:
    """Tests for write-lock contention diagnostics."""

    def test_default_message_omits_worker_forensics(self) -> None:
        from elspeth.contracts.coordination import RegisteredWorker
        from elspeth.contracts.errors import WriteLockHeldError

        err = WriteLockHeldError(
            run_id="run-sensitive",
            workers=(
                RegisteredWorker(
                    worker_id="worker-secret",
                    role="leader",
                    status="active",
                    pid=4242,
                    hostname="build-host.internal",
                ),
            ),
        )

        message = str(err)
        assert err.workers[0].worker_id == "worker-secret"
        assert "1 registered worker" in message
        assert "worker-secret" not in message
        assert "leader" not in message
        assert "active" not in message
        assert "4242" not in message
        assert "build-host.internal" not in message


class TestRoutingReasonSchema:
    """Tests for RoutingReason union type schema introspection."""

    pass


class TestRoutingReason:
    """Tests for RoutingReason union type usage."""

    pass


class TestTransformSuccessReason:
    """Tests for TransformSuccessReason TypedDict — construction and Literal values."""

    pass


class TestRoutingReasonUsage:
    """Tests for constructing valid RoutingReason variants."""

    pass


class TestTransformErrorReasonContract:
    """Tests for TransformErrorReason TypedDict contract — Literal values and optional fields."""

    pass


class TestTransformErrorReasonUsage:
    """Tests for constructing valid TransformErrorReason values."""

    pass


class TestNestedTypeDicts:
    """Tests for nested TypedDict structures."""

    pass


class TestQueryFailureDetailUsage:
    """Tests for constructing valid QueryFailureDetail values."""

    pass


class TestErrorDetailUsage:
    """Tests for constructing valid ErrorDetail values."""

    pass


class TestFailedQueriesFieldType:
    """Tests for failed_queries field with union type."""

    pass


class TestErrorsFieldType:
    """Tests for errors field with union type."""

    pass


class TestCoalesceFailureReasonSchema:
    """Tests for CoalesceFailureReason frozen dataclass schema."""

    def test_is_frozen_dataclass(self) -> None:
        """CoalesceFailureReason is a frozen dataclass (immutable after construction)."""
        from elspeth.contracts import CoalesceFailureReason

        assert dataclasses.is_dataclass(CoalesceFailureReason)
        error = CoalesceFailureReason(
            failure_reason="quorum_not_met",
            expected_branches=("a", "b"),
            branches_arrived=("a",),
            merge_policy="union",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            error.failure_reason = "modified"  # type: ignore[misc]

    def test_has_slots(self) -> None:
        """CoalesceFailureReason uses __slots__ for memory efficiency — no instance __dict__."""
        from elspeth.contracts import CoalesceFailureReason

        instance = CoalesceFailureReason(
            failure_reason="quorum_not_met",
            expected_branches=("a", "b"),
            branches_arrived=("a",),
            merge_policy="union",
        )
        with pytest.raises(TypeError, match="vars\\(\\) argument must have __dict__ attribute"):
            vars(instance)

    def test_to_dict_required_only(self) -> None:
        """to_dict() omits None-valued optional fields."""
        from elspeth.contracts import CoalesceFailureReason

        error = CoalesceFailureReason(
            failure_reason="incomplete_branches",
            expected_branches=("path_a", "path_b"),
            branches_arrived=("path_a",),
            merge_policy="union",
        )
        d = error.to_dict()
        assert d == {
            "failure_reason": "incomplete_branches",
            "expected_branches": ["path_a", "path_b"],
            "branches_arrived": ["path_a"],
            "merge_policy": "union",
        }
        assert "timeout_ms" not in d
        assert "select_branch" not in d

    def test_to_dict_with_timeout(self) -> None:
        """to_dict() includes timeout_ms when set."""
        from elspeth.contracts import CoalesceFailureReason

        error = CoalesceFailureReason(
            failure_reason="quorum_not_met_at_timeout",
            expected_branches=("a", "b", "c"),
            branches_arrived=("a",),
            merge_policy="nested",
            timeout_ms=30000,
        )
        d = error.to_dict()
        assert d["timeout_ms"] == 30000
        assert "select_branch" not in d

    def test_to_dict_with_select_branch(self) -> None:
        """to_dict() includes select_branch when set."""
        from elspeth.contracts import CoalesceFailureReason

        error = CoalesceFailureReason(
            failure_reason="select_branch_not_arrived",
            expected_branches=("fast", "slow"),
            branches_arrived=("slow",),
            merge_policy="select",
            select_branch="fast",
        )
        d = error.to_dict()
        assert d["select_branch"] == "fast"
        assert "timeout_ms" not in d

    def test_to_dict_with_all_optionals(self) -> None:
        """to_dict() includes all fields when all are set."""
        from elspeth.contracts import CoalesceFailureReason

        error = CoalesceFailureReason(
            failure_reason="select_branch_not_arrived",
            expected_branches=("a", "b"),
            branches_arrived=("b",),
            merge_policy="select",
            timeout_ms=5000,
            select_branch="a",
        )
        d = error.to_dict()
        assert d == {
            "failure_reason": "select_branch_not_arrived",
            "expected_branches": ["a", "b"],
            "branches_arrived": ["b"],
            "merge_policy": "select",
            "timeout_ms": 5000,
            "select_branch": "a",
        }

    def test_late_arrival_has_empty_branches_arrived(self) -> None:
        """Late arrival failures have empty branches_arrived list."""
        from elspeth.contracts import CoalesceFailureReason

        error = CoalesceFailureReason(
            failure_reason="late_arrival_after_merge",
            expected_branches=("a", "b"),
            branches_arrived=(),
            merge_policy="union",
        )
        assert error.branches_arrived == ()
        assert error.to_dict()["branches_arrived"] == []


class TestExecutionErrorPostInit:
    """Tests for ExecutionError __post_init__ validation."""

    def test_rejects_empty_exception(self) -> None:
        from elspeth.contracts import ExecutionError

        with pytest.raises(ValueError, match="exception must not be empty"):
            ExecutionError(exception="", exception_type="ValueError")

    def test_rejects_empty_exception_type(self) -> None:
        from elspeth.contracts import ExecutionError

        with pytest.raises(ValueError, match="exception_type must not be empty"):
            ExecutionError(exception="boom", exception_type="")

    def test_accepts_valid_construction(self) -> None:
        from elspeth.contracts import ExecutionError

        error = ExecutionError(exception="boom", exception_type="RuntimeError")
        assert error.exception == "boom"

    def test_rejects_non_mapping_context(self) -> None:
        from elspeth.contracts import ExecutionError

        with pytest.raises(TypeError, match="context must be a mapping, got list"):
            ExecutionError(
                exception="boom",
                exception_type="RuntimeError",
                context=["bad"],  # type: ignore[arg-type]
            )

    def test_rejects_context_with_non_string_keys(self) -> None:
        from elspeth.contracts import ExecutionError

        with pytest.raises(TypeError, match="context keys must be strings"):
            ExecutionError(
                exception="boom",
                exception_type="RuntimeError",
                context={1: "bad"},  # type: ignore[dict-item]
            )


class TestCoalesceFailureReasonPostInit:
    """Tests for CoalesceFailureReason __post_init__ validation."""

    def test_rejects_empty_failure_reason(self) -> None:
        from elspeth.contracts import CoalesceFailureReason

        with pytest.raises(ValueError, match="failure_reason must not be empty"):
            CoalesceFailureReason(
                failure_reason="",
                expected_branches=("a",),
                branches_arrived=(),
                merge_policy="union",
            )

    def test_rejects_empty_merge_policy(self) -> None:
        from elspeth.contracts import CoalesceFailureReason

        with pytest.raises(ValueError, match="merge_policy must not be empty"):
            CoalesceFailureReason(
                failure_reason="quorum_not_met",
                expected_branches=("a",),
                branches_arrived=(),
                merge_policy="",
            )

    def test_rejects_empty_expected_branches(self) -> None:
        from elspeth.contracts import CoalesceFailureReason

        with pytest.raises(ValueError, match="expected_branches must not be empty"):
            CoalesceFailureReason(
                failure_reason="quorum_not_met",
                expected_branches=(),
                branches_arrived=(),
                merge_policy="union",
            )

    def test_rejects_negative_timeout_ms(self) -> None:
        from elspeth.contracts import CoalesceFailureReason

        with pytest.raises(ValueError, match="timeout_ms must be non-negative"):
            CoalesceFailureReason(
                failure_reason="timeout",
                expected_branches=("a",),
                branches_arrived=(),
                merge_policy="union",
                timeout_ms=-1,
            )

    def test_to_dict_serializes_tuples_as_lists(self) -> None:
        """to_dict() converts tuple fields to lists for JSON compatibility."""
        from elspeth.contracts import CoalesceFailureReason

        error = CoalesceFailureReason(
            failure_reason="quorum_not_met",
            expected_branches=("a", "b"),
            branches_arrived=("a",),
            merge_policy="union",
        )
        d = error.to_dict()
        assert isinstance(d["expected_branches"], list)
        assert isinstance(d["branches_arrived"], list)
        assert d["expected_branches"] == ["a", "b"]
        assert d["branches_arrived"] == ["a"]


class TestCoalesceFailureReasonDeepFreeze:
    """Branch fields must be deeply frozen on direct construction."""

    def test_expected_branches_frozen(self) -> None:
        from elspeth.contracts import CoalesceFailureReason

        branches: list[str] = ["a", "b"]
        reason = CoalesceFailureReason(
            failure_reason="quorum_not_met",
            expected_branches=branches,  # type: ignore[arg-type]
            branches_arrived=("a",),
            merge_policy="union",
        )
        branches.append("mutated")
        assert isinstance(reason.expected_branches, tuple)
        assert "mutated" not in reason.expected_branches

    def test_branches_arrived_frozen(self) -> None:
        from elspeth.contracts import CoalesceFailureReason

        arrived: list[str] = ["a"]
        reason = CoalesceFailureReason(
            failure_reason="quorum_not_met",
            expected_branches=("a", "b"),
            branches_arrived=arrived,  # type: ignore[arg-type]
            merge_policy="union",
        )
        arrived.append("mutated")
        assert isinstance(reason.branches_arrived, tuple)
        assert "mutated" not in reason.branches_arrived
