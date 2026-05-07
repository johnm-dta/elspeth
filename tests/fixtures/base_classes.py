# tests/fixtures/base_classes.py
"""Protocol-compliant test base classes.

Single canonical definition of base classes for test sources, sinks,
and transforms. Migrated from tests/conftest.py with no behavioral changes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import TYPE_CHECKING, Any, cast

from elspeth.contracts import Determinism, PluginSchema, SourceRow
from elspeth.plugins.infrastructure.base import BaseTransform

if TYPE_CHECKING:
    from elspeth.contracts import (
        BatchTransformProtocol,
        SinkProtocol,
        SourceProtocol,
        TransformProtocol,
        TransformResult,
    )
    from elspeth.contracts.schema_contract import SchemaContract


class _TestSchema(PluginSchema):
    """Minimal schema for test fixtures."""

    pass


class _TestSourceBase:
    """Base class for test sources implementing SourceProtocol.

    Provides all required Protocol attributes and lifecycle methods.
    Child classes must provide: name, output_schema, load(ctx).
    """

    name: str
    output_schema: type[PluginSchema]
    node_id: str | None = None
    determinism = Determinism.DETERMINISTIC
    plugin_version = "1.0.0"
    source_file_hash: str | None = None
    _on_validation_failure: str = "discard"
    on_success: str = "default"
    declared_guaranteed_fields: frozenset[str] = frozenset()

    def __init__(self) -> None:
        self.config: dict[str, Any] = {"schema": {"mode": "observed"}}
        self._schema_contract: SchemaContract | None = None

    def wrap_rows(self, rows: list[dict[str, Any]]) -> Iterator[SourceRow]:
        """Wrap plain dicts in SourceRow.valid() as required by source protocol."""
        from elspeth.contracts.schema_contract import FieldContract, SchemaContract

        for row in rows:
            fields = tuple(
                FieldContract(
                    normalized_name=key,
                    original_name=key,
                    python_type=object,
                    required=False,
                    source="inferred",
                )
                for key in row
            )
            contract = SchemaContract(mode="OBSERVED", fields=fields, locked=True)
            if self._schema_contract is None:
                self._schema_contract = contract
            yield SourceRow.valid(row, contract=contract)

    def on_start(self, ctx: Any) -> None:
        pass

    def on_complete(self, ctx: Any) -> None:
        pass

    def close(self) -> None:
        pass

    def get_field_resolution(self) -> tuple[Mapping[str, str], str | None] | None:
        return None

    def get_schema_contract(self) -> SchemaContract | None:
        return self._schema_contract


class CallbackSource(_TestSourceBase):
    """Source with callbacks for deterministic MockClock testing.

    Enables tests to advance a MockClock between row yields.
    """

    name: str = "callback_source"
    output_schema: type[PluginSchema] = _TestSchema

    def __init__(
        self,
        rows: list[dict[str, Any]],
        output_schema: type[PluginSchema] | None = None,
        after_yield_callback: Callable[[int], None] | None = None,
        source_name: str = "callback_source",
        on_success: str = "default",
    ) -> None:
        super().__init__()
        self._rows = rows
        self._after_yield_callback = after_yield_callback
        self.name = source_name
        self.on_success = on_success
        if output_schema is not None:
            self.output_schema = output_schema

    def load(self, ctx: Any) -> Iterator[SourceRow]:
        from elspeth.contracts.schema_contract import FieldContract, SchemaContract

        for i, row in enumerate(self._rows):
            fields = tuple(
                FieldContract(
                    normalized_name=key,
                    original_name=key,
                    python_type=object,
                    required=False,
                    source="inferred",
                )
                for key in row
            )
            contract = SchemaContract(mode="OBSERVED", fields=fields, locked=True)
            yield SourceRow.valid(row, contract=contract)
            if self._after_yield_callback is not None:
                self._after_yield_callback(i)


class _TestSinkBase:
    """Base class for test sinks implementing SinkProtocol."""

    name: str
    input_schema: type[PluginSchema] = _TestSchema
    idempotent: bool = True
    node_id: str | None = None
    determinism = Determinism.DETERMINISTIC
    plugin_version = "1.0.0"
    source_file_hash: str | None = None
    declared_required_fields: frozenset[str] = frozenset()
    _on_write_failure: str | None = "discard"
    supports_resume: bool = False

    def __init__(self) -> None:
        self.config: dict[str, Any] = {"schema": {"mode": "observed"}}
        self._diversion_log: list[Any] = []

    def _reset_diversion_log(self) -> None:
        self._diversion_log = []

    def on_start(self, ctx: Any) -> None:
        pass

    def on_complete(self, ctx: Any) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass

    def configure_for_resume(self) -> None:
        raise NotImplementedError("Test sinks do not support resume")

    def validate_output_target(self) -> Any:
        from elspeth.contracts.sink import OutputValidationResult

        return OutputValidationResult.success()

    @property
    def needs_resume_field_resolution(self) -> bool:
        return False

    def set_resume_field_resolution(self, resolution_mapping: dict[str, str]) -> None:
        pass

    @classmethod
    def get_config_model(cls, config: dict[str, Any] | None = None) -> None:
        return None

    @classmethod
    def get_config_schema(cls) -> dict[str, Any]:
        """Minimal schema stub — production sinks return a JSON schema,
        test fixtures only need the attribute to satisfy SinkProtocol.
        """
        return {"type": "object", "additionalProperties": True}


class _TestTransformBase(BaseTransform):
    """Base class for test transforms inheriting production BaseTransform.

    Inherits lifecycle methods (on_start, on_complete, close) and the
    _on_start_called lifecycle guard from BaseTransform. This ensures test
    transforms automatically track any future BaseTransform attributes,
    preventing silent drift between test fixtures and production code.
    """

    name: str
    input_schema: type[PluginSchema] = _TestSchema
    output_schema: type[PluginSchema] = _TestSchema
    plugin_version = "1.0.0"

    def __init__(self) -> None:
        super().__init__({"schema": {"mode": "observed"}})


# ─────────────────────────────────────────────────────────────────────────
# Type Cast Helpers
# ─────────────────────────────────────────────────────────────────────────


def as_source(source: Any) -> SourceProtocol:
    """Cast a test source to SourceProtocol."""
    return cast("SourceProtocol", source)


def as_transform(transform: Any) -> TransformProtocol:
    """Cast a test transform to TransformProtocol."""
    return cast("TransformProtocol", transform)


def as_batch_transform(transform: Any) -> BatchTransformProtocol:
    """Cast a test batch transform to BatchTransformProtocol."""
    return cast("BatchTransformProtocol", transform)


def inject_write_failure[S](sink: S, value: str = "discard") -> S:
    """Inject _on_write_failure on a production sink instance.

    Production code injects this via cli_helpers from SinkSettings.
    Tests that construct sinks directly bypass that path. Call this
    on any production sink (CSVSink, JSONSink, etc.) after construction.

    Returns the same sink for call-chaining.
    """
    # Access via Any — S is always a concrete sink with _on_write_failure,
    # but the generic type parameter can't express the SinkProtocol bound.
    s: Any = sink
    if s._on_write_failure is None:
        s._on_write_failure = value
    return sink


def as_sink(sink: Any) -> SinkProtocol:
    """Cast a test sink to SinkProtocol.

    Also ensures _on_write_failure is set if not already — production code
    injects this via cli_helpers, but tests that construct sinks directly
    bypass that path.
    """
    inject_write_failure(sink)
    return cast("SinkProtocol", sink)


def create_observed_contract(row: dict[str, Any]) -> SchemaContract:
    """Create an OBSERVED schema contract from a row."""
    from elspeth.contracts.schema_contract import FieldContract, SchemaContract

    fields = tuple(
        FieldContract(
            normalized_name=key,
            original_name=key,
            python_type=object,
            required=False,
            source="inferred",
        )
        for key in row
    )
    return SchemaContract(mode="OBSERVED", fields=fields, locked=True)


def as_transform_result(result: Any) -> TransformResult:
    """Assert and cast a result to TransformResult."""
    from elspeth.engine.batch_adapter import ExceptionResult

    if isinstance(result, ExceptionResult):
        raise AssertionError(f"Expected TransformResult but got ExceptionResult: {result.exception}\n{result.traceback}")
    return cast("TransformResult", result)
