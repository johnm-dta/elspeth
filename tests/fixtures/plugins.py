# tests/fixtures/plugins.py
"""Consolidated test plugins — one canonical definition per plugin.

Eliminates the 3 duplicate ListSource/CollectSink definitions from v1.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, ClassVar

from pydantic import ConfigDict

from elspeth.contracts import ArtifactDescriptor, Determinism, PluginSchema, SourceRow
from elspeth.contracts.diversion import RowDiversion, SinkWriteResult
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.results import TransformResult
from tests.fixtures.base_classes import _TestSchema, _TestSinkBase, _TestSourceBase


class _EngineTestSchema(PluginSchema):
    """Dynamic schema for engine-level test plugins."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")


class ListSource(_TestSourceBase):
    """Source that yields rows from a list.

    Usage:
        source = ListSource([{"value": 1}, {"value": 2}])
        source = ListSource([{"value": 1}], name="my_source")
    """

    determinism = Determinism.IO_READ
    output_schema = _EngineTestSchema

    def __init__(self, data: list[dict[str, Any]], name: str = "list_source", on_success: str = "default") -> None:
        super().__init__()
        self._data = data
        self.name = name
        self.on_success = on_success

    def on_start(self, ctx: Any) -> None:
        pass

    def load(self, ctx: Any) -> Iterator[SourceRow]:
        yield from self.wrap_rows(self._data)

    def close(self) -> None:
        pass


class CollectSink(_TestSinkBase):
    """Sink that collects results into a list.

    Usage:
        sink = CollectSink()
        sink = CollectSink("output_sink")
        sink = CollectSink("output", node_id="sink_node_123")
    """

    def __init__(self, name: str = "collect", *, node_id: str | None = None) -> None:
        super().__init__()
        self.name = name
        self.node_id = node_id
        self.results: list[dict[str, Any]] = []
        self._artifact_counter = 0

    @property
    def rows_written(self) -> list[dict[str, Any]]:
        """Alias for results — some tests use this name."""
        return self.results

    def on_start(self, ctx: Any) -> None:
        pass

    def on_complete(self, ctx: Any) -> None:
        pass

    def write(self, rows: Any, ctx: Any) -> SinkWriteResult:
        self.results.extend(rows)
        self._artifact_counter += 1
        return SinkWriteResult(
            artifact=ArtifactDescriptor.for_file(
                path=f"memory://{self.name}_{self._artifact_counter}",
                size_bytes=len(str(rows)),
                content_hash=f"hash_{self._artifact_counter}",
            )
        )

    def close(self) -> None:
        pass


class FailingSink(_TestSinkBase):
    """Sink whose write() always raises RuntimeError.

    For testing error handling in orchestrator, executors, and outcome recording.

    Usage:
        sink = FailingSink()
        sink = FailingSink("broken_sink")
        sink = FailingSink(error_message="Custom error")
    """

    def __init__(
        self,
        name: str = "failing_sink",
        *,
        node_id: str | None = None,
        error_message: str = "Sink write failed",
    ) -> None:
        super().__init__()
        self.name = name
        self.node_id = node_id
        self._error_message = error_message

    def on_start(self, ctx: Any) -> None:
        pass

    def on_complete(self, ctx: Any) -> None:
        pass

    def write(self, rows: Any, ctx: Any) -> SinkWriteResult:
        raise RuntimeError(self._error_message)

    def close(self) -> None:
        pass


class DivertingSink(_TestSinkBase):
    """Sink that diverts the first configured rows and writes the rest.

    This exercises the production ``SinkExecutor`` discard branch: when no
    failsink is configured for this sink, every returned ``RowDiversion`` is
    recorded as ``(FAILURE, SINK_DISCARDED)`` with the discard sentinel sink.
    """

    name = "diverting_sink"

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        name: str | None = None,
        divert_count: int | None = None,
    ) -> None:
        super().__init__()
        options = dict(config or {})
        configured_name = name if name is not None else options.get("name", self.name)
        self.name = str(configured_name)
        configured_divert_count = divert_count if divert_count is not None else options.get("divert_count")
        if configured_divert_count is None:
            self._divert_count: int | None = None
        else:
            self._divert_count = int(configured_divert_count)
        self.results: list[dict[str, Any]] = []
        self._artifact_counter = 0

    def on_start(self, ctx: Any) -> None:
        pass

    def on_complete(self, ctx: Any) -> None:
        pass

    def write(self, rows: Any, ctx: Any) -> SinkWriteResult:
        del ctx
        row_list = list(rows)
        limit = self._divert_count if self._divert_count is not None else len(row_list)
        diversions = tuple(
            RowDiversion(
                row_index=idx,
                reason="diverting_sink: forced divert for ADR-019 test",
                row_data=dict(row),
            )
            for idx, row in enumerate(row_list)
            if idx < limit
        )
        primary_rows = [dict(row) for idx, row in enumerate(row_list) if idx >= limit]
        self.results.extend(primary_rows)
        self._artifact_counter += 1
        return SinkWriteResult(
            artifact=ArtifactDescriptor.for_file(
                path=f"memory://{self.name}_{self._artifact_counter}",
                size_bytes=len(str(primary_rows)),
                content_hash=f"hash_{self._artifact_counter}",
            ),
            diversions=diversions,
        )

    def close(self) -> None:
        pass


class FailingSource(ListSource):
    """Source whose load() always raises RuntimeError.

    For testing error handling during source loading (orchestrator cleanup,
    export partial semantics, etc.).

    Usage:
        source = FailingSource()
        source = FailingSource(error_message="Custom load failure")
    """

    determinism = Determinism.IO_READ

    def __init__(
        self,
        *,
        name: str = "failing_source",
        error_message: str = "Source failed intentionally",
    ) -> None:
        super().__init__(data=[], name=name)
        self._error_message = error_message

    def load(self, ctx: Any) -> Iterator[SourceRow]:
        raise RuntimeError(self._error_message)


class PassTransform(BaseTransform):
    """Identity transform — passes rows through unchanged."""

    name = "pass_transform"
    determinism = Determinism.DETERMINISTIC
    input_schema: type[PluginSchema] = _TestSchema
    output_schema: type[PluginSchema] = _TestSchema

    def __init__(
        self,
        *,
        name: str | None = None,
        input_connection: str | None = None,
        on_success: str | None = None,
        on_error: str | None = None,
    ) -> None:
        super().__init__({"schema": {"mode": "observed"}})
        if name is not None:
            self.name = name
        if input_connection is not None:
            self.input = input_connection
        if on_success is not None:
            self.on_success = on_success
        if on_error is not None:
            self.on_error = on_error

    def process(self, row: Any, ctx: Any) -> TransformResult:
        return TransformResult.success(row, success_reason={"action": "passthrough"})


class FailTransform(BaseTransform):
    """Transform that always returns an error result."""

    name = "fail_transform"
    determinism = Determinism.DETERMINISTIC
    input_schema: type[PluginSchema] = _TestSchema
    output_schema: type[PluginSchema] = _TestSchema
    on_error = "discard"

    def __init__(
        self,
        error_reason: str = "always_fail",
        *,
        name: str | None = None,
        input_connection: str | None = None,
        on_error: str | None = None,
    ) -> None:
        super().__init__({"schema": {"mode": "observed"}})
        if name is not None:
            self.name = name
        if input_connection is not None:
            self.input = input_connection
        if on_error is not None:
            self.on_error = on_error
        self._error_reason = error_reason

    def process(self, row: Any, ctx: Any) -> TransformResult:
        return TransformResult.error({"reason": "deliberate_failure", "error": self._error_reason})


class ConditionalErrorTransform(BaseTransform):
    """Transform that errors on rows where 'fail' key is truthy."""

    name = "conditional_error"
    determinism = Determinism.DETERMINISTIC
    input_schema: type[PluginSchema] = _TestSchema
    output_schema: type[PluginSchema] = _TestSchema
    on_error = "discard"

    def __init__(
        self,
        *,
        name: str | None = None,
        input_connection: str | None = None,
        on_success: str | None = None,
        on_error: str | None = None,
    ) -> None:
        super().__init__({"schema": {"mode": "observed"}})
        if name is not None:
            self.name = name
        if input_connection is not None:
            self.input = input_connection
        if on_success is not None:
            self.on_success = on_success
        if on_error is not None:
            self.on_error = on_error

    def process(self, row: Any, ctx: Any) -> TransformResult:
        if row["fail"]:
            return TransformResult.error({"reason": "test_error"})
        return TransformResult.success(row, success_reason={"action": "test"})


class CountingTransform(BaseTransform):
    """Transform that counts invocations (for retry testing)."""

    name = "counting_transform"
    determinism = Determinism.DETERMINISTIC
    input_schema: type[PluginSchema] = _TestSchema
    output_schema: type[PluginSchema] = _TestSchema

    def __init__(
        self,
        *,
        name: str | None = None,
        input_connection: str | None = None,
        on_success: str | None = None,
        on_error: str | None = None,
    ) -> None:
        super().__init__({"schema": {"mode": "observed"}})
        if name is not None:
            self.name = name
        if input_connection is not None:
            self.input = input_connection
        if on_success is not None:
            self.on_success = on_success
        if on_error is not None:
            self.on_error = on_error
        self.call_count = 0

    def process(self, row: Any, ctx: Any) -> TransformResult:
        self.call_count += 1
        return TransformResult.success(row, success_reason={"action": "counted"})


class SlowTransform(BaseTransform):
    """Transform with configurable delay (for timeout testing)."""

    name = "slow_transform"
    determinism = Determinism.DETERMINISTIC
    input_schema: type[PluginSchema] = _TestSchema
    output_schema: type[PluginSchema] = _TestSchema

    def __init__(
        self,
        delay_seconds: float = 0.1,
        *,
        name: str | None = None,
        input_connection: str | None = None,
        on_success: str | None = None,
        on_error: str | None = None,
    ) -> None:
        super().__init__({"schema": {"mode": "observed"}})
        if name is not None:
            self.name = name
        if input_connection is not None:
            self.input = input_connection
        if on_success is not None:
            self.on_success = on_success
        if on_error is not None:
            self.on_error = on_error
        self._delay = delay_seconds

    def process(self, row: Any, ctx: Any) -> TransformResult:
        import time

        time.sleep(self._delay)
        return TransformResult.success(row, success_reason={"action": "delayed"})


class ErrorOnNthTransform(BaseTransform):
    """Transform that errors on the Nth invocation (for retry integration)."""

    name = "error_on_nth"
    determinism = Determinism.DETERMINISTIC
    input_schema: type[PluginSchema] = _TestSchema
    output_schema: type[PluginSchema] = _TestSchema
    on_error = "discard"

    def __init__(
        self,
        error_on: int = 1,
        *,
        name: str | None = None,
        input_connection: str | None = None,
        on_success: str | None = None,
        on_error: str | None = None,
    ) -> None:
        super().__init__({"schema": {"mode": "observed"}})
        if name is not None:
            self.name = name
        if input_connection is not None:
            self.input = input_connection
        if on_success is not None:
            self.on_success = on_success
        if on_error is not None:
            self.on_error = on_error
        self._error_on = error_on
        self._call_count = 0

    def process(self, row: Any, ctx: Any) -> TransformResult:
        self._call_count += 1
        if self._call_count == self._error_on:
            return TransformResult.error({"reason": "simulated_failure", "error": f"nth_error_{self._error_on}"}, retryable=True)
        return TransformResult.success(row, success_reason={"action": "passed"})


class CallRecordingTransform(BaseTransform):
    """Transform that records a single HTTP operation_call via ctx.record_call().

    Used by F1 Cell-2 regression: proves re-driven transforms RE-FIRE their
    recorded calls (at-least-once) and that the re-fired call's node_state
    carries resume_checkpoint_id (attributability invariant, ADDENDUM 2.C).

    Each invocation of process() records one HTTP call with a synthetic
    request payload.  The call is state-parented (calls.state_id → node_states)
    so it inherits the resume_checkpoint_id from its parent node_state.
    """

    name = "call_recording_transform"
    determinism = Determinism.NON_DETERMINISTIC  # has external call side-effects
    input_schema: type[PluginSchema] = _TestSchema
    output_schema: type[PluginSchema] = _TestSchema

    def __init__(
        self,
        *,
        name: str | None = None,
        input_connection: str | None = None,
        on_success: str | None = None,
        on_error: str | None = None,
    ) -> None:
        super().__init__({"schema": {"mode": "observed"}})
        if name is not None:
            self.name = name
        if input_connection is not None:
            self.input = input_connection
        if on_success is not None:
            self.on_success = on_success
        if on_error is not None:
            self.on_error = on_error

    def process(self, row: Any, ctx: Any) -> TransformResult:
        from elspeth.contracts.enums import CallStatus, CallType

        # Record a synthetic HTTP call so the audit trail contains a calls row
        # linked to this node_state (state-parented: calls.state_id is set).
        # On resume, the re-drive writes a new node_state at attempt=max+1 with
        # resume_checkpoint_id set, and records another calls row linked there.
        ctx.record_call(
            call_type=CallType.HTTP,
            status=CallStatus.SUCCESS,
            request_data={"url": "http://test.internal/probe", "row_value": row["value"] if "value" in row else 0},
            response_data={"status": 200},
        )
        return TransformResult.success(row, success_reason={"action": "call_recorded"})
