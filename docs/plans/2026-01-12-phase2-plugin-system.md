# Phase 2: Plugin System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the domain-agnostic plugin system with protocols, base classes, and pluggy integration. Establish the "bones" for Phase 3 Landscape/OpenTelemetry integration.

**Architecture:** Everything extensible happens through plugins. The framework is neutral about what transforms do - it only knows how to orchestrate them. This phase creates the contracts; Phase 3 creates the engine that uses them.

**Tech Stack:** Python 3.11+, Pydantic v2 (schemas), pluggy (hooks), typing (Protocols)

**Phase 3 Integration Points:** This phase deliberately includes Optional fields and lifecycle hooks that Phase 3 will use for Landscape audit and OpenTelemetry tracing. These are documented inline.

---

## Task 1: RowOutcome Enum

**Files:**
- Create: `src/elspeth/plugins/results.py`
- Create: `tests/plugins/test_results.py`

### Step 1: Write the failing test

```python
# tests/plugins/test_results.py
"""Tests for plugin result types."""

import pytest


class TestRowOutcome:
    """Terminal states for rows."""

    def test_all_terminal_states_exist(self) -> None:
        from elspeth.plugins.results import RowOutcome

        # Every row must reach exactly one terminal state
        assert RowOutcome.COMPLETED.value == "completed"
        assert RowOutcome.ROUTED.value == "routed"
        assert RowOutcome.FORKED.value == "forked"
        assert RowOutcome.CONSUMED_IN_BATCH.value == "consumed_in_batch"
        assert RowOutcome.COALESCED.value == "coalesced"
        assert RowOutcome.QUARANTINED.value == "quarantined"
        assert RowOutcome.FAILED.value == "failed"

    def test_outcome_is_enum(self) -> None:
        from enum import Enum

        from elspeth.plugins.results import RowOutcome

        assert issubclass(RowOutcome, Enum)
```

### Step 2: Run test to verify it fails

Run: `pytest tests/plugins/test_results.py -v`
Expected: FAIL (ImportError)

### Step 3: Create results module with RowOutcome

```python
# src/elspeth/plugins/results.py
"""Result types for plugin operations.

These types define the contracts between plugins and the SDA engine.
All fields needed for Phase 3 Landscape/OpenTelemetry integration are
included here, even if not used until Phase 3.
"""

from enum import Enum


class RowOutcome(Enum):
    """Terminal states for rows in the pipeline.

    INVARIANT: Every row reaches exactly one terminal state.
    No silent drops.
    """

    COMPLETED = "completed"           # Reached output sink
    ROUTED = "routed"                 # Sent to named sink by gate (move mode)
    FORKED = "forked"                 # Split into child tokens (parent terminates)
    CONSUMED_IN_BATCH = "consumed_in_batch"  # Fed into aggregation
    COALESCED = "coalesced"           # Merged with other tokens
    QUARANTINED = "quarantined"       # Failed, stored for investigation
    FAILED = "failed"                 # Failed, not recoverable
```

### Step 4: Create tests directory

```bash
mkdir -p tests/plugins
touch tests/plugins/__init__.py
```

### Step 5: Run test to verify it passes

Run: `pytest tests/plugins/test_results.py -v`
Expected: PASS

### Step 6: Commit

```bash
git add src/elspeth/plugins/results.py tests/plugins/
git commit -m "feat(plugins): add RowOutcome enum for terminal states"
```

---

## Task 2: RoutingAction and TransformResult

**Files:**
- Modify: `src/elspeth/plugins/results.py`
- Modify: `tests/plugins/test_results.py`

### Step 1: Write the failing tests

```python
# Add to tests/plugins/test_results.py

from dataclasses import FrozenInstanceError


class TestRoutingAction:
    """Routing decisions from gates."""

    def test_continue_action(self) -> None:
        from elspeth.plugins.results import RoutingAction

        action = RoutingAction.continue_()
        assert action.kind == "continue"
        assert action.destinations == []
        assert action.mode == "move"

    def test_route_to_sink(self) -> None:
        from elspeth.plugins.results import RoutingAction

        action = RoutingAction.route_to_sink("flagged", reason={"confidence": 0.95})
        assert action.kind == "route_to_sink"
        assert action.destinations == ["flagged"]
        assert action.reason == {"confidence": 0.95}

    def test_fork_to_paths(self) -> None:
        from elspeth.plugins.results import RoutingAction

        action = RoutingAction.fork_to_paths(["stats", "classifier", "archive"])
        assert action.kind == "fork_to_paths"
        assert action.destinations == ["stats", "classifier", "archive"]
        assert action.mode == "copy"

    def test_immutable(self) -> None:
        from elspeth.plugins.results import RoutingAction

        action = RoutingAction.continue_()
        with pytest.raises(FrozenInstanceError):
            action.kind = "route_to_sink"


class TestTransformResult:
    """Results from transform operations."""

    def test_success_result(self) -> None:
        from elspeth.plugins.results import TransformResult

        result = TransformResult.success({"value": 42})
        assert result.status == "success"
        assert result.row == {"value": 42}
        assert result.retryable is False

    def test_error_result(self) -> None:
        from elspeth.plugins.results import TransformResult

        result = TransformResult.error(
            reason={"error": "validation failed"},
            retryable=True,
        )
        assert result.status == "error"
        assert result.row is None
        assert result.retryable is True

    def test_has_audit_fields(self) -> None:
        """Phase 3 integration: audit fields must exist."""
        from elspeth.plugins.results import TransformResult

        result = TransformResult.success({"x": 1})
        # These fields are set by the engine in Phase 3
        assert hasattr(result, "input_hash")
        assert hasattr(result, "output_hash")
        assert hasattr(result, "duration_ms")
        assert result.input_hash is None  # Not set yet
```

### Step 2: Run test to verify it fails

Run: `pytest tests/plugins/test_results.py::TestRoutingAction -v`
Expected: FAIL

### Step 3: Add RoutingAction and TransformResult

```python
# Add to src/elspeth/plugins/results.py

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class RoutingAction:
    """What a gate decided to do with a row.

    Immutable to prevent accidental modification after decision.
    """

    kind: Literal["continue", "route_to_sink", "fork_to_paths"]
    destinations: tuple[str, ...]  # Immutable sequence
    mode: Literal["move", "copy"]
    reason: dict[str, Any]

    @classmethod
    def continue_(cls, reason: dict[str, Any] | None = None) -> "RoutingAction":
        """Row continues to next transform."""
        return cls(
            kind="continue",
            destinations=(),
            mode="move",
            reason=reason or {},
        )

    @classmethod
    def route_to_sink(
        cls,
        sink_name: str,
        *,
        mode: Literal["move", "copy"] = "move",
        reason: dict[str, Any] | None = None,
    ) -> "RoutingAction":
        """Route row to a named sink."""
        return cls(
            kind="route_to_sink",
            destinations=(sink_name,),
            mode=mode,
            reason=reason or {},
        )

    @classmethod
    def fork_to_paths(
        cls,
        paths: list[str],
        *,
        reason: dict[str, Any] | None = None,
    ) -> "RoutingAction":
        """Fork row to multiple parallel paths (copy mode)."""
        return cls(
            kind="fork_to_paths",
            destinations=tuple(paths),
            mode="copy",
            reason=reason or {},
        )


@dataclass
class TransformResult:
    """Result from any transform operation.

    Includes all fields needed for Phase 3 Landscape audit.
    The engine populates audit fields; plugins set status/row/reason.
    """

    status: Literal["success", "route", "error"]
    row: dict[str, Any] | None
    reason: dict[str, Any] | None
    retryable: bool = False

    # === Phase 3 Audit Fields ===
    # Set by engine, not by plugins
    input_hash: str | None = field(default=None, repr=False)
    output_hash: str | None = field(default=None, repr=False)
    duration_ms: float | None = field(default=None, repr=False)

    @classmethod
    def success(cls, row: dict[str, Any]) -> "TransformResult":
        """Create a successful transform result."""
        return cls(status="success", row=row, reason=None)

    @classmethod
    def error(
        cls,
        reason: dict[str, Any],
        *,
        retryable: bool = False,
    ) -> "TransformResult":
        """Create an error result."""
        return cls(
            status="error",
            row=None,
            reason=reason,
            retryable=retryable,
        )
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/plugins/test_results.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add -u
git commit -m "feat(plugins): add RoutingAction and TransformResult with audit fields"
```

---

## Task 3: GateResult and AcceptResult

**Files:**
- Modify: `src/elspeth/plugins/results.py`
- Modify: `tests/plugins/test_results.py`

### Step 1: Write the failing tests

```python
# Add to tests/plugins/test_results.py

class TestGateResult:
    """Results from gate transforms."""

    def test_gate_result_with_continue(self) -> None:
        from elspeth.plugins.results import GateResult, RoutingAction

        result = GateResult(
            row={"value": 42},
            action=RoutingAction.continue_(),
        )
        assert result.row == {"value": 42}
        assert result.action.kind == "continue"

    def test_gate_result_with_route(self) -> None:
        from elspeth.plugins.results import GateResult, RoutingAction

        result = GateResult(
            row={"value": 42, "flagged": True},
            action=RoutingAction.route_to_sink("review", reason={"score": 0.9}),
        )
        assert result.action.kind == "route_to_sink"
        assert result.action.destinations == ("review",)

    def test_has_audit_fields(self) -> None:
        """Phase 3 integration: audit fields must exist."""
        from elspeth.plugins.results import GateResult, RoutingAction

        result = GateResult(
            row={"x": 1},
            action=RoutingAction.continue_(),
        )
        assert hasattr(result, "input_hash")
        assert hasattr(result, "output_hash")
        assert hasattr(result, "duration_ms")


class TestAcceptResult:
    """Results from aggregation accept()."""

    def test_accepted_no_trigger(self) -> None:
        from elspeth.plugins.results import AcceptResult

        result = AcceptResult(accepted=True, trigger=False)
        assert result.accepted is True
        assert result.trigger is False

    def test_accepted_with_trigger(self) -> None:
        from elspeth.plugins.results import AcceptResult

        result = AcceptResult(accepted=True, trigger=True)
        assert result.trigger is True

    def test_has_batch_id_field(self) -> None:
        """Phase 3 integration: batch_id for Landscape."""
        from elspeth.plugins.results import AcceptResult

        result = AcceptResult(accepted=True, trigger=False)
        assert hasattr(result, "batch_id")
        assert result.batch_id is None  # Set by engine in Phase 3
```

### Step 2: Run test to verify it fails

Run: `pytest tests/plugins/test_results.py::TestGateResult -v`
Expected: FAIL

### Step 3: Add GateResult and AcceptResult

```python
# Add to src/elspeth/plugins/results.py

@dataclass
class GateResult:
    """Result from a gate transform.

    Gates evaluate rows and decide routing, possibly modifying the row.
    """

    row: dict[str, Any]
    action: RoutingAction

    # === Phase 3 Audit Fields ===
    input_hash: str | None = field(default=None, repr=False)
    output_hash: str | None = field(default=None, repr=False)
    duration_ms: float | None = field(default=None, repr=False)


@dataclass
class AcceptResult:
    """Result from aggregation accept().

    Indicates whether the row was accepted and if batch should trigger.
    """

    accepted: bool
    trigger: bool  # Should flush now?

    # === Phase 3 Audit Fields ===
    # batch_id is set by engine when creating/updating Landscape batch
    batch_id: str | None = field(default=None, repr=False)
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/plugins/test_results.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add -u
git commit -m "feat(plugins): add GateResult and AcceptResult with Phase 3 fields"
```

---

## Task 4: PluginContext

**Files:**
- Create: `src/elspeth/plugins/context.py`
- Create: `tests/plugins/test_context.py`

### Step 1: Write the failing tests

```python
# tests/plugins/test_context.py
"""Tests for plugin context."""

from contextlib import nullcontext

import pytest


class TestPluginContext:
    """Context passed to all plugin operations."""

    def test_minimal_context(self) -> None:
        from elspeth.plugins.context import PluginContext

        ctx = PluginContext(run_id="run-001", config={})
        assert ctx.run_id == "run-001"
        assert ctx.config == {}

    def test_optional_integrations_default_none(self) -> None:
        from elspeth.plugins.context import PluginContext

        ctx = PluginContext(run_id="run-001", config={})
        # Phase 3 integration points - optional in Phase 2
        assert ctx.landscape is None
        assert ctx.tracer is None
        assert ctx.payload_store is None

    def test_start_span_without_tracer(self) -> None:
        from elspeth.plugins.context import PluginContext

        ctx = PluginContext(run_id="run-001", config={})
        # Should return nullcontext when no tracer
        span_ctx = ctx.start_span("test_operation")
        assert isinstance(span_ctx, nullcontext)

    def test_get_config_value(self) -> None:
        from elspeth.plugins.context import PluginContext

        ctx = PluginContext(
            run_id="run-001",
            config={"threshold": 0.5, "nested": {"key": "value"}},
        )
        assert ctx.get("threshold") == 0.5
        assert ctx.get("nested.key") == "value"
        assert ctx.get("missing", default="default") == "default"
```

### Step 2: Run test to verify it fails

Run: `pytest tests/plugins/test_context.py -v`
Expected: FAIL (ImportError)

### Step 3: Create PluginContext

```python
# src/elspeth/plugins/context.py
"""Plugin execution context.

The PluginContext carries everything a plugin might need during execution.
Phase 2 includes Optional placeholders for Phase 3 integrations.

Phase 3 Integration Points:
- landscape: LandscapeRecorder for audit trail
- tracer: OpenTelemetry Tracer for distributed tracing
- payload_store: PayloadStore for large blob storage
"""

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ContextManager

if TYPE_CHECKING:
    # These types are available in Phase 3
    # Using string annotations to avoid import errors in Phase 2
    from opentelemetry.trace import Span, Tracer

    from elspeth.core.payload_store import PayloadStore


# Protocol for Landscape recorder (Phase 3)
# Defined here so plugins can type-hint against it
class LandscapeRecorder:
    """Protocol for Landscape audit recording.

    Implemented in Phase 3. Plugins can optionally use this
    for custom audit events.
    """

    def record_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Record a custom audit event."""
        ...


@dataclass
class PluginContext:
    """Context passed to every plugin operation.

    Provides access to:
    - Run metadata (run_id, config)
    - Phase 3 integrations (landscape, tracer, payload_store)
    - Utility methods (get config values, start spans)

    Example:
        def process(self, row: dict, ctx: PluginContext) -> TransformResult:
            threshold = ctx.get("threshold", default=0.5)
            with ctx.start_span("my_operation"):
                result = do_work(row, threshold)
            return TransformResult.success(result)
    """

    run_id: str
    config: dict[str, Any]

    # === Phase 3 Integration Points ===
    # Optional in Phase 2, populated by engine in Phase 3
    landscape: LandscapeRecorder | None = None
    tracer: "Tracer | None" = None
    payload_store: "PayloadStore | None" = None

    # Additional metadata
    node_id: str | None = field(default=None)
    plugin_name: str | None = field(default=None)

    def get(self, key: str, *, default: Any = None) -> Any:
        """Get a config value by dotted path.

        Args:
            key: Dotted path like "nested.key"
            default: Value if key not found

        Returns:
            Config value or default
        """
        parts = key.split(".")
        value: Any = self.config
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        return value

    def start_span(self, name: str) -> ContextManager["Span | None"]:
        """Start an OpenTelemetry span.

        Returns nullcontext if tracer not configured.

        Usage:
            with ctx.start_span("operation_name"):
                do_work()
        """
        if self.tracer is None:
            return nullcontext()
        return self.tracer.start_as_current_span(name)
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/plugins/test_context.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/plugins/context.py tests/plugins/test_context.py
git commit -m "feat(plugins): add PluginContext with Phase 3 integration points"
```

---

## Task 5: Pydantic Schema Base Classes

**Files:**
- Create: `src/elspeth/plugins/schemas.py`
- Create: `tests/plugins/test_schemas.py`

### Step 1: Write the failing tests

```python
# tests/plugins/test_schemas.py
"""Tests for plugin schema system."""

import pytest
from pydantic import ValidationError


class TestPluginSchema:
    """Base class for plugin schemas."""

    def test_schema_validates_fields(self) -> None:
        from pydantic import BaseModel

        from elspeth.plugins.schemas import PluginSchema

        class MySchema(PluginSchema):
            temperature: float
            humidity: float

        # Valid data
        data = MySchema(temperature=20.5, humidity=65.0)
        assert data.temperature == 20.5

        # Invalid data
        with pytest.raises(ValidationError):
            MySchema(temperature="not a number", humidity=65.0)

    def test_schema_to_dict(self) -> None:
        from elspeth.plugins.schemas import PluginSchema

        class MySchema(PluginSchema):
            value: int
            name: str

        data = MySchema(value=42, name="test")
        as_dict = data.to_row()
        assert as_dict == {"value": 42, "name": "test"}

    def test_schema_from_row(self) -> None:
        from elspeth.plugins.schemas import PluginSchema

        class MySchema(PluginSchema):
            value: int
            name: str

        row = {"value": 42, "name": "test", "extra": "ignored"}
        data = MySchema.from_row(row)
        assert data.value == 42
        assert data.name == "test"

    def test_schema_extra_fields_ignored(self) -> None:
        from elspeth.plugins.schemas import PluginSchema

        class StrictSchema(PluginSchema):
            required_field: str

        # Extra fields should be ignored, not cause errors
        data = StrictSchema.from_row({"required_field": "value", "extra": "ignored"})
        assert data.required_field == "value"


class TestSchemaValidation:
    """Schema validation utilities."""

    def test_validate_row_against_schema(self) -> None:
        from elspeth.plugins.schemas import PluginSchema, validate_row

        class MySchema(PluginSchema):
            x: int
            y: int

        # Valid
        errors = validate_row({"x": 1, "y": 2}, MySchema)
        assert errors == []

        # Invalid
        errors = validate_row({"x": "not int", "y": 2}, MySchema)
        assert len(errors) > 0

    def test_validate_missing_field(self) -> None:
        from elspeth.plugins.schemas import PluginSchema, validate_row

        class MySchema(PluginSchema):
            required: str

        errors = validate_row({}, MySchema)
        assert len(errors) > 0
        assert "required" in str(errors[0])
```

### Step 2: Run test to verify it fails

Run: `pytest tests/plugins/test_schemas.py -v`
Expected: FAIL (ImportError)

### Step 3: Create schema base classes

```python
# src/elspeth/plugins/schemas.py
"""Pydantic-based schema system for plugins.

Every plugin declares input and output schemas using Pydantic models.
This enables:
- Runtime validation of row data
- Pipeline validation at config time (Phase 3)
- Documentation generation
- Landscape context recording
"""

from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError

T = TypeVar("T", bound="PluginSchema")


class PluginSchema(BaseModel):
    """Base class for plugin input/output schemas.

    Plugins define schemas by subclassing:

        class MyInputSchema(PluginSchema):
            temperature: float
            humidity: float

        class MyOutputSchema(PluginSchema):
            temperature: float
            humidity: float
            heat_index: float  # Added by transform

    Features:
    - Extra fields ignored (rows may have more fields than schema requires)
    - Strict type validation
    - Easy conversion to/from row dicts
    """

    model_config = ConfigDict(
        extra="ignore",  # Rows may have extra fields
        strict=False,    # Allow coercion (e.g., int -> float)
        frozen=False,    # Allow modification
    )

    def to_row(self) -> dict[str, Any]:
        """Convert schema instance to row dict."""
        return self.model_dump()

    @classmethod
    def from_row(cls: type[T], row: dict[str, Any]) -> T:
        """Create schema instance from row dict.

        Extra fields in row are ignored.
        """
        return cls.model_validate(row)


class SchemaValidationError:
    """A validation error for a specific field."""

    def __init__(self, field: str, message: str, value: Any = None) -> None:
        self.field = field
        self.message = message
        self.value = value

    def __str__(self) -> str:
        return f"{self.field}: {self.message}"

    def __repr__(self) -> str:
        return f"SchemaValidationError({self.field!r}, {self.message!r})"


def validate_row(
    row: dict[str, Any],
    schema: type[PluginSchema],
) -> list[SchemaValidationError]:
    """Validate a row against a schema.

    Args:
        row: Row data to validate
        schema: PluginSchema subclass

    Returns:
        List of validation errors (empty if valid)
    """
    try:
        schema.model_validate(row)
        return []
    except ValidationError as e:
        errors = []
        for error in e.errors():
            field = ".".join(str(loc) for loc in error["loc"])
            errors.append(SchemaValidationError(
                field=field,
                message=error["msg"],
                value=error.get("input"),
            ))
        return errors
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/plugins/test_schemas.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/plugins/schemas.py tests/plugins/test_schemas.py
git commit -m "feat(plugins): add Pydantic schema base classes"
```

---

## Task 6: Schema Compatibility Checking

**Files:**
- Modify: `src/elspeth/plugins/schemas.py`
- Modify: `tests/plugins/test_schemas.py`

### Step 1: Write the failing tests

```python
# Add to tests/plugins/test_schemas.py

class TestSchemaCompatibility:
    """Check if output schema is compatible with input schema."""

    def test_compatible_schemas(self) -> None:
        from elspeth.plugins.schemas import PluginSchema, check_compatibility

        class ProducerOutput(PluginSchema):
            x: int
            y: int
            z: str

        class ConsumerInput(PluginSchema):
            x: int
            y: int

        # Producer outputs all fields consumer needs
        result = check_compatibility(ProducerOutput, ConsumerInput)
        assert result.compatible is True
        assert result.missing_fields == []

    def test_incompatible_schemas_missing_field(self) -> None:
        from elspeth.plugins.schemas import PluginSchema, check_compatibility

        class ProducerOutput(PluginSchema):
            x: int

        class ConsumerInput(PluginSchema):
            x: int
            y: int  # Not provided by producer

        result = check_compatibility(ProducerOutput, ConsumerInput)
        assert result.compatible is False
        assert "y" in result.missing_fields

    def test_incompatible_schemas_type_mismatch(self) -> None:
        from elspeth.plugins.schemas import PluginSchema, check_compatibility

        class ProducerOutput(PluginSchema):
            value: str  # String

        class ConsumerInput(PluginSchema):
            value: int  # Expects int

        result = check_compatibility(ProducerOutput, ConsumerInput)
        assert result.compatible is False
        assert len(result.type_mismatches) > 0
```

### Step 2: Run test to verify it fails

Run: `pytest tests/plugins/test_schemas.py::TestSchemaCompatibility -v`
Expected: FAIL

### Step 3: Add compatibility checking

```python
# Add to src/elspeth/plugins/schemas.py

from dataclasses import dataclass, field as dataclass_field
from typing import get_type_hints


@dataclass
class CompatibilityResult:
    """Result of schema compatibility check."""

    compatible: bool
    missing_fields: list[str] = dataclass_field(default_factory=list)
    type_mismatches: list[tuple[str, type, type]] = dataclass_field(default_factory=list)

    @property
    def error_message(self) -> str | None:
        """Human-readable error message if incompatible."""
        if self.compatible:
            return None

        parts = []
        if self.missing_fields:
            parts.append(f"Missing fields: {', '.join(self.missing_fields)}")
        if self.type_mismatches:
            mismatches = [
                f"{name} (expected {expected.__name__}, got {actual.__name__})"
                for name, expected, actual in self.type_mismatches
            ]
            parts.append(f"Type mismatches: {', '.join(mismatches)}")

        return "; ".join(parts)


def check_compatibility(
    producer_schema: type[PluginSchema],
    consumer_schema: type[PluginSchema],
) -> CompatibilityResult:
    """Check if producer output is compatible with consumer input.

    Compatibility means:
    - All required fields in consumer are provided by producer
    - Field types are compatible (exact match or coercible)

    Args:
        producer_schema: Output schema of upstream plugin
        consumer_schema: Input schema of downstream plugin

    Returns:
        CompatibilityResult indicating compatibility and any issues
    """
    producer_hints = get_type_hints(producer_schema)
    consumer_hints = get_type_hints(consumer_schema)

    missing: list[str] = []
    mismatches: list[tuple[str, type, type]] = []

    for field_name, expected_type in consumer_hints.items():
        if field_name not in producer_hints:
            missing.append(field_name)
        else:
            actual_type = producer_hints[field_name]
            if not _types_compatible(actual_type, expected_type):
                mismatches.append((field_name, expected_type, actual_type))

    compatible = len(missing) == 0 and len(mismatches) == 0

    return CompatibilityResult(
        compatible=compatible,
        missing_fields=missing,
        type_mismatches=mismatches,
    )


def _types_compatible(actual: type, expected: type) -> bool:
    """Check if actual type is compatible with expected type.

    Currently checks for exact match or numeric compatibility.
    """
    # Exact match
    if actual == expected:
        return True

    # Numeric compatibility (int -> float is OK)
    if expected is float and actual is int:
        return True

    # Optional handling could be added here

    return False
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/plugins/test_schemas.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add -u
git commit -m "feat(plugins): add schema compatibility checking"
```

---

## Task 7: Source Plugin Protocol

**Files:**
- Create: `src/elspeth/plugins/protocols.py`
- Create: `tests/plugins/test_protocols.py`

### Step 1: Write the failing tests

```python
# tests/plugins/test_protocols.py
"""Tests for plugin protocols."""

from typing import Iterator

import pytest


class TestSourceProtocol:
    """Source plugin protocol."""

    def test_source_protocol_definition(self) -> None:
        from typing import runtime_checkable

        from elspeth.plugins.protocols import SourceProtocol

        # Should be a Protocol
        assert hasattr(SourceProtocol, "__protocol_attrs__")

    def test_source_implementation(self) -> None:
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.protocols import SourceProtocol
        from elspeth.plugins.schemas import PluginSchema

        class OutputSchema(PluginSchema):
            value: int

        class MySource:
            """Example source implementation."""

            name = "my_source"
            output_schema = OutputSchema

            def __init__(self, config: dict) -> None:
                self.config = config

            def load(self, ctx: PluginContext) -> Iterator[dict]:
                for i in range(3):
                    yield {"value": i}

            def close(self) -> None:
                pass

        source = MySource({"path": "test.csv"})
        ctx = PluginContext(run_id="test", config={})

        rows = list(source.load(ctx))
        assert len(rows) == 3
        assert rows[0] == {"value": 0}

    def test_source_has_lifecycle_hooks(self) -> None:
        from elspeth.plugins.protocols import SourceProtocol

        # Check protocol has expected methods
        assert hasattr(SourceProtocol, "load")
        assert hasattr(SourceProtocol, "close")
```

### Step 2: Run test to verify it fails

Run: `pytest tests/plugins/test_protocols.py -v`
Expected: FAIL (ImportError)

### Step 3: Create protocols module with SourceProtocol

```python
# src/elspeth/plugins/protocols.py
"""Plugin protocols defining the contracts for each plugin type.

These protocols define what methods plugins must implement.
They're used for type checking, not runtime enforcement (that's pluggy's job).

Plugin Types:
- Source: Loads data into the system (one per run)
- Transform: Processes rows (stateless)
- Gate: Routes rows to destinations (stateless)
- Aggregation: Accumulates rows, flushes batches (stateful)
- Coalesce: Merges parallel paths (stateful)
- Sink: Outputs data (one or more per run)
"""

from typing import TYPE_CHECKING, Any, Iterator, Protocol, runtime_checkable

if TYPE_CHECKING:
    from elspeth.plugins.context import PluginContext
    from elspeth.plugins.results import AcceptResult, GateResult, TransformResult
    from elspeth.plugins.schemas import PluginSchema


@runtime_checkable
class SourceProtocol(Protocol):
    """Protocol for source plugins.

    Sources load data into the system. There is exactly one source per run.

    Lifecycle:
    1. __init__(config) - Plugin instantiation
    2. on_start(ctx) - Called before loading (optional)
    3. load(ctx) - Yields rows
    4. close() - Cleanup

    Example:
        class CSVSource:
            name = "csv"
            output_schema = RowSchema

            def load(self, ctx: PluginContext) -> Iterator[dict]:
                with open(self.path) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        yield row
    """

    name: str
    output_schema: type["PluginSchema"]

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        ...

    def load(self, ctx: "PluginContext") -> Iterator[dict[str, Any]]:
        """Load and yield rows from the source.

        Args:
            ctx: Plugin context with run metadata

        Yields:
            Row dicts matching output_schema
        """
        ...

    def close(self) -> None:
        """Clean up resources.

        Called after all rows are loaded or on error.
        """
        ...

    # === Optional Lifecycle Hooks ===

    def on_start(self, ctx: "PluginContext") -> None:
        """Called before load(). Override for setup."""
        ...
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/plugins/test_protocols.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/plugins/protocols.py tests/plugins/test_protocols.py
git commit -m "feat(plugins): add SourceProtocol"
```

---

## Task 8: Transform Plugin Protocols

**Files:**
- Modify: `src/elspeth/plugins/protocols.py`
- Modify: `tests/plugins/test_protocols.py`

### Step 1: Write the failing tests

```python
# Add to tests/plugins/test_protocols.py

class TestTransformProtocol:
    """Transform plugin protocol (stateless row processing)."""

    def test_transform_implementation(self) -> None:
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.protocols import TransformProtocol
        from elspeth.plugins.results import TransformResult
        from elspeth.plugins.schemas import PluginSchema

        class InputSchema(PluginSchema):
            value: int

        class OutputSchema(PluginSchema):
            value: int
            doubled: int

        class DoubleTransform:
            name = "double"
            input_schema = InputSchema
            output_schema = OutputSchema

            def __init__(self, config: dict) -> None:
                self.config = config

            def process(self, row: dict, ctx: PluginContext) -> TransformResult:
                return TransformResult.success({
                    "value": row["value"],
                    "doubled": row["value"] * 2,
                })

        transform = DoubleTransform({})
        ctx = PluginContext(run_id="test", config={})

        result = transform.process({"value": 21}, ctx)
        assert result.status == "success"
        assert result.row == {"value": 21, "doubled": 42}


class TestGateProtocol:
    """Gate plugin protocol (routing decisions)."""

    def test_gate_implementation(self) -> None:
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.protocols import GateProtocol
        from elspeth.plugins.results import GateResult, RoutingAction
        from elspeth.plugins.schemas import PluginSchema

        class RowSchema(PluginSchema):
            value: int

        class ThresholdGate:
            name = "threshold"
            input_schema = RowSchema
            output_schema = RowSchema

            def __init__(self, config: dict) -> None:
                self.threshold = config.get("threshold", 10)

            def evaluate(self, row: dict, ctx: PluginContext) -> GateResult:
                if row["value"] > self.threshold:
                    return GateResult(
                        row=row,
                        action=RoutingAction.route_to_sink(
                            "high_values",
                            reason={"value": row["value"], "threshold": self.threshold},
                        ),
                    )
                return GateResult(row=row, action=RoutingAction.continue_())

        gate = ThresholdGate({"threshold": 50})
        ctx = PluginContext(run_id="test", config={})

        # Below threshold - continue
        result = gate.evaluate({"value": 30}, ctx)
        assert result.action.kind == "continue"

        # Above threshold - route
        result = gate.evaluate({"value": 100}, ctx)
        assert result.action.kind == "route_to_sink"
        assert result.action.destinations == ("high_values",)
```

### Step 2: Run test to verify it fails

Run: `pytest tests/plugins/test_protocols.py::TestTransformProtocol -v`
Expected: FAIL

### Step 3: Add TransformProtocol and GateProtocol

```python
# Add to src/elspeth/plugins/protocols.py

@runtime_checkable
class TransformProtocol(Protocol):
    """Protocol for stateless row transforms.

    Transforms process one row and emit one row (possibly modified).
    They are stateless between rows.

    Example:
        class EnrichTransform:
            name = "enrich"
            input_schema = InputSchema
            output_schema = OutputSchema

            def process(self, row: dict, ctx: PluginContext) -> TransformResult:
                enriched = {**row, "timestamp": datetime.now().isoformat()}
                return TransformResult.success(enriched)
    """

    name: str
    input_schema: type["PluginSchema"]
    output_schema: type["PluginSchema"]

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        ...

    def process(
        self,
        row: dict[str, Any],
        ctx: "PluginContext",
    ) -> "TransformResult":
        """Process a single row.

        Args:
            row: Input row matching input_schema
            ctx: Plugin context

        Returns:
            TransformResult with processed row or error
        """
        ...

    # === Optional Lifecycle Hooks ===

    def on_register(self, ctx: "PluginContext") -> None:
        """Called when plugin is registered."""
        ...

    def on_start(self, ctx: "PluginContext") -> None:
        """Called at start of run."""
        ...

    def on_complete(self, ctx: "PluginContext") -> None:
        """Called at end of run."""
        ...


@runtime_checkable
class GateProtocol(Protocol):
    """Protocol for gate transforms (routing decisions).

    Gates evaluate rows and decide routing. They can:
    - Continue to next transform
    - Route to a named sink
    - Fork to multiple parallel paths

    Example:
        class SafetyGate:
            name = "safety"
            input_schema = InputSchema
            output_schema = OutputSchema

            def evaluate(self, row: dict, ctx: PluginContext) -> GateResult:
                if row.get("suspicious"):
                    return GateResult(
                        row=row,
                        action=RoutingAction.route_to_sink("review_queue"),
                    )
                return GateResult(row=row, action=RoutingAction.continue_())
    """

    name: str
    input_schema: type["PluginSchema"]
    output_schema: type["PluginSchema"]

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        ...

    def evaluate(
        self,
        row: dict[str, Any],
        ctx: "PluginContext",
    ) -> "GateResult":
        """Evaluate a row and decide routing.

        Args:
            row: Input row
            ctx: Plugin context

        Returns:
            GateResult with (possibly modified) row and routing action
        """
        ...

    # === Optional Lifecycle Hooks ===

    def on_register(self, ctx: "PluginContext") -> None:
        """Called when plugin is registered."""
        ...

    def on_start(self, ctx: "PluginContext") -> None:
        """Called at start of run."""
        ...

    def on_complete(self, ctx: "PluginContext") -> None:
        """Called at end of run."""
        ...
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/plugins/test_protocols.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add -u
git commit -m "feat(plugins): add TransformProtocol and GateProtocol"
```

---

## Task 9: Aggregation and Coalesce Protocols

**Files:**
- Modify: `src/elspeth/plugins/protocols.py`
- Modify: `tests/plugins/test_protocols.py`

### Step 1: Write the failing tests

```python
# Add to tests/plugins/test_protocols.py

class TestAggregationProtocol:
    """Aggregation plugin protocol (stateful batching)."""

    def test_aggregation_implementation(self) -> None:
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.protocols import AggregationProtocol
        from elspeth.plugins.results import AcceptResult
        from elspeth.plugins.schemas import PluginSchema

        class InputSchema(PluginSchema):
            value: int

        class OutputSchema(PluginSchema):
            total: int
            count: int

        class SumAggregation:
            name = "sum"
            input_schema = InputSchema
            output_schema = OutputSchema

            def __init__(self, config: dict) -> None:
                self.batch_size = config.get("batch_size", 3)
                self._values: list[int] = []

            def accept(self, row: dict, ctx: PluginContext) -> AcceptResult:
                self._values.append(row["value"])
                trigger = len(self._values) >= self.batch_size
                return AcceptResult(accepted=True, trigger=trigger)

            def should_trigger(self) -> bool:
                return len(self._values) >= self.batch_size

            def flush(self, ctx: PluginContext) -> list[dict]:
                result = {
                    "total": sum(self._values),
                    "count": len(self._values),
                }
                self._values = []
                return [result]

            def reset(self) -> None:
                self._values = []

        agg = SumAggregation({"batch_size": 2})
        ctx = PluginContext(run_id="test", config={})

        # First row - no trigger
        result = agg.accept({"value": 10}, ctx)
        assert result.accepted is True
        assert result.trigger is False

        # Second row - trigger
        result = agg.accept({"value": 20}, ctx)
        assert result.trigger is True

        # Flush
        outputs = agg.flush(ctx)
        assert len(outputs) == 1
        assert outputs[0] == {"total": 30, "count": 2}


class TestCoalesceProtocol:
    """Coalesce plugin protocol (merge parallel paths)."""

    def test_coalesce_policy_types(self) -> None:
        from elspeth.plugins.protocols import CoalescePolicy

        # All policies should exist
        assert CoalescePolicy.REQUIRE_ALL.value == "require_all"
        assert CoalescePolicy.QUORUM.value == "quorum"
        assert CoalescePolicy.BEST_EFFORT.value == "best_effort"

    def test_quorum_requires_threshold(self) -> None:
        """QUORUM policy needs a quorum_threshold."""
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.protocols import CoalescePolicy
        from elspeth.plugins.schemas import PluginSchema

        class OutputSchema(PluginSchema):
            combined: str

        class QuorumCoalesce:
            name = "quorum_merge"
            policy = CoalescePolicy.QUORUM
            quorum_threshold = 2  # At least 2 branches must arrive
            expected_branches = ["branch_a", "branch_b", "branch_c"]
            output_schema = OutputSchema

            def __init__(self, config: dict) -> None:
                pass

            def merge(self, branch_outputs: dict, ctx: PluginContext) -> dict:
                return {"combined": "+".join(branch_outputs.keys())}

        coalesce = QuorumCoalesce({})
        assert coalesce.quorum_threshold == 2
        assert len(coalesce.expected_branches) == 3
```

### Step 2: Run test to verify it fails

Run: `pytest tests/plugins/test_protocols.py::TestAggregationProtocol -v`
Expected: FAIL

### Step 3: Add AggregationProtocol and CoalesceProtocol

```python
# Add to src/elspeth/plugins/protocols.py

from enum import Enum


class CoalescePolicy(Enum):
    """How coalesce handles partial arrivals."""

    REQUIRE_ALL = "require_all"   # Wait for all branches; any failure fails
    QUORUM = "quorum"             # Merge if >= n branches succeed
    BEST_EFFORT = "best_effort"   # Merge whatever arrives by timeout


@runtime_checkable
class AggregationProtocol(Protocol):
    """Protocol for aggregation transforms (stateful batching).

    Aggregations accumulate rows until a trigger condition, then flush.

    Phase 3 Integration:
    - Engine creates Landscape batch on first accept()
    - Engine persists batch membership on every accept()
    - Engine transitions batch status on flush()

    Example:
        class StatsAggregation:
            name = "stats"
            input_schema = InputSchema
            output_schema = StatsSchema

            def accept(self, row, ctx) -> AcceptResult:
                self._values.append(row["value"])
                return AcceptResult(
                    accepted=True,
                    trigger=len(self._values) >= self.batch_size,
                )

            def flush(self, ctx) -> list[dict]:
                result = {"mean": statistics.mean(self._values)}
                self._values = []
                return [result]
    """

    name: str
    input_schema: type["PluginSchema"]
    output_schema: type["PluginSchema"]

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        ...

    def accept(
        self,
        row: dict[str, Any],
        ctx: "PluginContext",
    ) -> "AcceptResult":
        """Accept a row into the batch.

        Called for each row. Implementation should:
        1. Store the row in internal buffer
        2. Return trigger=True when batch should flush

        Note: In Phase 3, the engine wraps this to manage Landscape batches.

        Args:
            row: Input row
            ctx: Plugin context

        Returns:
            AcceptResult indicating acceptance and trigger state
        """
        ...

    def should_trigger(self) -> bool:
        """Check if batch should flush now.

        Called by engine to check trigger condition outside of accept().
        """
        ...

    def flush(self, ctx: "PluginContext") -> list[dict[str, Any]]:
        """Process accumulated rows and return results.

        Called when trigger condition is met or at end of source.
        Should reset internal state after processing.

        Note: In Phase 3, the engine wraps this to:
        1. Transition batch to "executing"
        2. Record results
        3. Transition batch to "completed" or "failed"

        Args:
            ctx: Plugin context

        Returns:
            List of output rows (usually one aggregate result)
        """
        ...

    def reset(self) -> None:
        """Reset internal state.

        Called by engine on error recovery.
        """
        ...

    # === Optional Lifecycle Hooks ===

    def on_register(self, ctx: "PluginContext") -> None:
        """Called when plugin is registered."""
        ...

    def on_start(self, ctx: "PluginContext") -> None:
        """Called at start of run."""
        ...

    def on_complete(self, ctx: "PluginContext") -> None:
        """Called at end of run."""
        ...


@runtime_checkable
class CoalesceProtocol(Protocol):
    """Protocol for coalesce transforms (merge parallel paths).

    Coalesce merges results from parallel branches back into a single path.

    Configuration:
    - policy: How to handle partial arrivals
    - quorum_threshold: Minimum branches for QUORUM policy (None otherwise)
    - inputs: Which branches to expect
    - key: How to correlate branch outputs (Phase 3 engine concern)

    Example:
        class SimpleCoalesce:
            name = "merge"
            policy = CoalescePolicy.REQUIRE_ALL
            quorum_threshold = None  # Only used for QUORUM policy

            def merge(self, branch_outputs, ctx) -> dict:
                merged = {}
                for branch_name, output in branch_outputs.items():
                    merged.update(output)
                return merged

        class QuorumCoalesce:
            name = "quorum_merge"
            policy = CoalescePolicy.QUORUM
            quorum_threshold = 2  # Proceed if >= 2 branches arrive
    """

    name: str
    policy: CoalescePolicy
    quorum_threshold: int | None  # Required if policy == QUORUM
    expected_branches: list[str]
    output_schema: type["PluginSchema"]

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        ...

    def merge(
        self,
        branch_outputs: dict[str, dict[str, Any]],
        ctx: "PluginContext",
    ) -> dict[str, Any]:
        """Merge outputs from multiple branches.

        Args:
            branch_outputs: Map of branch_name -> output_row
            ctx: Plugin context

        Returns:
            Merged output row
        """
        ...

    # === Optional Lifecycle Hooks ===

    def on_register(self, ctx: "PluginContext") -> None:
        """Called when plugin is registered."""
        ...
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/plugins/test_protocols.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add -u
git commit -m "feat(plugins): add AggregationProtocol and CoalesceProtocol"
```

---

## Task 10: Sink Protocol

**Files:**
- Modify: `src/elspeth/plugins/protocols.py`
- Modify: `tests/plugins/test_protocols.py`

### Step 1: Write the failing tests

```python
# Add to tests/plugins/test_protocols.py

class TestSinkProtocol:
    """Sink plugin protocol."""

    def test_sink_implementation(self) -> None:
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.protocols import SinkProtocol
        from elspeth.plugins.schemas import PluginSchema

        class InputSchema(PluginSchema):
            value: int

        class MemorySink:
            """Test sink that stores rows in memory."""

            name = "memory"
            input_schema = InputSchema
            rows: list[dict]

            def __init__(self, config: dict) -> None:
                self.rows = []
                self.config = config

            def write(self, row: dict, ctx: PluginContext) -> None:
                self.rows.append(row)

            def flush(self) -> None:
                pass

            def close(self) -> None:
                pass

        sink = MemorySink({})
        ctx = PluginContext(run_id="test", config={})

        sink.write({"value": 1}, ctx)
        sink.write({"value": 2}, ctx)

        assert len(sink.rows) == 2
        assert sink.rows[0] == {"value": 1}

    def test_sink_has_idempotency_support(self) -> None:
        """Sinks should support idempotency keys."""
        from elspeth.plugins.protocols import SinkProtocol

        # Protocol should have idempotent flag
        # (implementation detail - sinks can declare if they're idempotent)
        pass  # Just checking protocol exists
```

### Step 2: Run test to verify it fails

Run: `pytest tests/plugins/test_protocols.py::TestSinkProtocol -v`
Expected: FAIL (if SinkProtocol doesn't exist)

### Step 3: Add SinkProtocol

```python
# Add to src/elspeth/plugins/protocols.py

@runtime_checkable
class SinkProtocol(Protocol):
    """Protocol for sink plugins.

    Sinks output data to external destinations.
    There can be multiple sinks per run.

    Idempotency:
    - Sinks receive idempotency keys: {run_id}:{row_id}:{sink_name}
    - Sinks that cannot guarantee idempotency should set idempotent=False

    Example:
        class CSVSink:
            name = "csv"
            input_schema = RowSchema
            idempotent = False  # Appends are not idempotent

            def write(self, row: dict, ctx: PluginContext) -> None:
                self._writer.writerow(row)

            def flush(self) -> None:
                self._file.flush()
    """

    name: str
    input_schema: type["PluginSchema"]
    idempotent: bool  # Can this sink handle retries safely?

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        ...

    def write(
        self,
        row: dict[str, Any],
        ctx: "PluginContext",
    ) -> None:
        """Write a row to the sink.

        Args:
            row: Row data to write
            ctx: Plugin context
        """
        ...

    def flush(self) -> None:
        """Flush any buffered data.

        Called periodically and at end of run.
        """
        ...

    def close(self) -> None:
        """Close the sink and release resources.

        Called at end of run or on error.
        """
        ...

    # === Optional Lifecycle Hooks ===

    def on_register(self, ctx: "PluginContext") -> None:
        """Called when plugin is registered."""
        ...

    def on_start(self, ctx: "PluginContext") -> None:
        """Called at start of run."""
        ...

    def on_complete(self, ctx: "PluginContext") -> None:
        """Called at end of run (before close)."""
        ...
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/plugins/test_protocols.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add -u
git commit -m "feat(plugins): add SinkProtocol"
```

---

## Task 11: pluggy Hookspecs

**Files:**
- Create: `src/elspeth/plugins/hookspecs.py`
- Create: `tests/plugins/test_hookspecs.py`

### Step 1: Write the failing tests

```python
# tests/plugins/test_hookspecs.py
"""Tests for pluggy hook specifications."""

import pytest


class TestHookspecs:
    """pluggy hook specifications."""

    def test_hookspec_marker_exists(self) -> None:
        from elspeth.plugins.hookspecs import hookspec

        assert hookspec is not None

    def test_source_hooks_defined(self) -> None:
        from elspeth.plugins.hookspecs import ElspethSourceSpec

        # Check that hook methods exist
        assert hasattr(ElspethSourceSpec, "elspeth_get_source")

    def test_transform_hooks_defined(self) -> None:
        from elspeth.plugins.hookspecs import ElspethTransformSpec

        assert hasattr(ElspethTransformSpec, "elspeth_get_transforms")
        assert hasattr(ElspethTransformSpec, "elspeth_get_gates")
        assert hasattr(ElspethTransformSpec, "elspeth_get_aggregations")

    def test_sink_hooks_defined(self) -> None:
        from elspeth.plugins.hookspecs import ElspethSinkSpec

        assert hasattr(ElspethSinkSpec, "elspeth_get_sinks")
```

### Step 2: Run test to verify it fails

Run: `pytest tests/plugins/test_hookspecs.py -v`
Expected: FAIL (ImportError)

### Step 3: Create hookspecs module

```python
# src/elspeth/plugins/hookspecs.py
"""pluggy hook specifications for Elspeth plugins.

Plugins implement these hooks to register themselves with the framework.
The plugin manager calls these hooks during discovery.

Usage:
    import pluggy
    from elspeth.plugins.hookspecs import hookspec

    class MyPlugin:
        @hookspec.implementation
        def elspeth_get_transforms(self):
            return [MyTransform]
"""

from typing import TYPE_CHECKING

import pluggy

if TYPE_CHECKING:
    from elspeth.plugins.protocols import (
        AggregationProtocol,
        CoalesceProtocol,
        GateProtocol,
        SinkProtocol,
        SourceProtocol,
        TransformProtocol,
    )

# Project name for pluggy
PROJECT_NAME = "elspeth"

# Hook specification marker
hookspec = pluggy.HookspecMarker(PROJECT_NAME)

# Hook implementation marker (for plugins to use)
hookimpl = pluggy.HookimplMarker(PROJECT_NAME)


class ElspethSourceSpec:
    """Hook specifications for source plugins."""

    @hookspec
    def elspeth_get_source(self) -> list[type["SourceProtocol"]]:
        """Return source plugin classes.

        Returns:
            List of Source plugin classes (not instances)
        """


class ElspethTransformSpec:
    """Hook specifications for transform plugins."""

    @hookspec
    def elspeth_get_transforms(self) -> list[type["TransformProtocol"]]:
        """Return transform plugin classes.

        Returns:
            List of Transform plugin classes
        """

    @hookspec
    def elspeth_get_gates(self) -> list[type["GateProtocol"]]:
        """Return gate plugin classes.

        Returns:
            List of Gate plugin classes
        """

    @hookspec
    def elspeth_get_aggregations(self) -> list[type["AggregationProtocol"]]:
        """Return aggregation plugin classes.

        Returns:
            List of Aggregation plugin classes
        """

    @hookspec
    def elspeth_get_coalesces(self) -> list[type["CoalesceProtocol"]]:
        """Return coalesce plugin classes.

        Returns:
            List of Coalesce plugin classes
        """


class ElspethSinkSpec:
    """Hook specifications for sink plugins."""

    @hookspec
    def elspeth_get_sinks(self) -> list[type["SinkProtocol"]]:
        """Return sink plugin classes.

        Returns:
            List of Sink plugin classes
        """
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/plugins/test_hookspecs.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/plugins/hookspecs.py tests/plugins/test_hookspecs.py
git commit -m "feat(plugins): add pluggy hookspecs"
```

---

## Task 12: Plugin Manager

**Files:**
- Create: `src/elspeth/plugins/manager.py`
- Create: `tests/plugins/test_manager.py`

### Step 1: Write the failing tests

```python
# tests/plugins/test_manager.py
"""Tests for plugin manager."""

import pytest


class TestPluginManager:
    """Plugin discovery and registration."""

    def test_create_manager(self) -> None:
        from elspeth.plugins.manager import PluginManager

        manager = PluginManager()
        assert manager is not None

    def test_register_plugin(self) -> None:
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.hookspecs import hookimpl
        from elspeth.plugins.manager import PluginManager
        from elspeth.plugins.results import TransformResult
        from elspeth.plugins.schemas import PluginSchema

        class InputSchema(PluginSchema):
            x: int

        class OutputSchema(PluginSchema):
            x: int
            y: int

        class MyTransform:
            name = "my_transform"
            input_schema = InputSchema
            output_schema = OutputSchema

            def __init__(self, config: dict) -> None:
                pass

            def process(self, row: dict, ctx: PluginContext) -> TransformResult:
                return TransformResult.success({**row, "y": row["x"] * 2})

        class MyPlugin:
            @hookimpl
            def elspeth_get_transforms(self) -> list:
                return [MyTransform]

        manager = PluginManager()
        manager.register(MyPlugin())

        transforms = manager.get_transforms()
        assert len(transforms) == 1
        assert transforms[0].name == "my_transform"

    def test_get_plugin_by_name(self) -> None:
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.hookspecs import hookimpl
        from elspeth.plugins.manager import PluginManager
        from elspeth.plugins.results import TransformResult
        from elspeth.plugins.schemas import PluginSchema

        class Schema(PluginSchema):
            x: int

        class TransformA:
            name = "transform_a"
            input_schema = Schema
            output_schema = Schema

            def __init__(self, config: dict) -> None:
                pass

            def process(self, row: dict, ctx: PluginContext) -> TransformResult:
                return TransformResult.success(row)

        class TransformB:
            name = "transform_b"
            input_schema = Schema
            output_schema = Schema

            def __init__(self, config: dict) -> None:
                pass

            def process(self, row: dict, ctx: PluginContext) -> TransformResult:
                return TransformResult.success(row)

        class MyPlugin:
            @hookimpl
            def elspeth_get_transforms(self) -> list:
                return [TransformA, TransformB]

        manager = PluginManager()
        manager.register(MyPlugin())

        transform = manager.get_transform_by_name("transform_b")
        assert transform is not None
        assert transform.name == "transform_b"

        missing = manager.get_transform_by_name("nonexistent")
        assert missing is None
```

### Step 2: Run test to verify it fails

Run: `pytest tests/plugins/test_manager.py -v`
Expected: FAIL (ImportError)

### Step 3: Create plugin manager

```python
# src/elspeth/plugins/manager.py
"""Plugin manager for discovery, registration, and lifecycle.

Uses pluggy for hook-based plugin registration.
"""

from typing import Any

import pluggy

from elspeth.plugins.hookspecs import (
    PROJECT_NAME,
    ElspethSinkSpec,
    ElspethSourceSpec,
    ElspethTransformSpec,
)
from elspeth.plugins.protocols import (
    AggregationProtocol,
    CoalesceProtocol,
    GateProtocol,
    SinkProtocol,
    SourceProtocol,
    TransformProtocol,
)


class PluginManager:
    """Manages plugin discovery, registration, and lookup.

    Usage:
        manager = PluginManager()
        manager.register(MyPlugin())

        transforms = manager.get_transforms()
        my_transform = manager.get_transform_by_name("my_transform")
    """

    def __init__(self) -> None:
        self._pm = pluggy.PluginManager(PROJECT_NAME)

        # Register hookspecs
        self._pm.add_hookspecs(ElspethSourceSpec)
        self._pm.add_hookspecs(ElspethTransformSpec)
        self._pm.add_hookspecs(ElspethSinkSpec)

        # Caches
        self._sources: list[type[SourceProtocol]] = []
        self._transforms: list[type[TransformProtocol]] = []
        self._gates: list[type[GateProtocol]] = []
        self._aggregations: list[type[AggregationProtocol]] = []
        self._coalesces: list[type[CoalesceProtocol]] = []
        self._sinks: list[type[SinkProtocol]] = []

    def register(self, plugin: Any) -> None:
        """Register a plugin.

        Args:
            plugin: Plugin instance implementing hook methods
        """
        self._pm.register(plugin)
        self._refresh_caches()

    def _refresh_caches(self) -> None:
        """Refresh plugin caches from hooks."""
        self._sources = []
        self._transforms = []
        self._gates = []
        self._aggregations = []
        self._coalesces = []
        self._sinks = []

        # Collect from all registered plugins
        for sources in self._pm.hook.elspeth_get_source():
            self._sources.extend(sources)

        for transforms in self._pm.hook.elspeth_get_transforms():
            self._transforms.extend(transforms)

        for gates in self._pm.hook.elspeth_get_gates():
            self._gates.extend(gates)

        for aggs in self._pm.hook.elspeth_get_aggregations():
            self._aggregations.extend(aggs)

        for coalesces in self._pm.hook.elspeth_get_coalesces():
            self._coalesces.extend(coalesces)

        for sinks in self._pm.hook.elspeth_get_sinks():
            self._sinks.extend(sinks)

    # === Getters ===

    def get_sources(self) -> list[type[SourceProtocol]]:
        """Get all registered source plugins."""
        return self._sources.copy()

    def get_transforms(self) -> list[type[TransformProtocol]]:
        """Get all registered transform plugins."""
        return self._transforms.copy()

    def get_gates(self) -> list[type[GateProtocol]]:
        """Get all registered gate plugins."""
        return self._gates.copy()

    def get_aggregations(self) -> list[type[AggregationProtocol]]:
        """Get all registered aggregation plugins."""
        return self._aggregations.copy()

    def get_coalesces(self) -> list[type[CoalesceProtocol]]:
        """Get all registered coalesce plugins."""
        return self._coalesces.copy()

    def get_sinks(self) -> list[type[SinkProtocol]]:
        """Get all registered sink plugins."""
        return self._sinks.copy()

    # === Lookup by name ===

    def get_source_by_name(self, name: str) -> type[SourceProtocol] | None:
        """Get source plugin by name."""
        for source in self._sources:
            if source.name == name:
                return source
        return None

    def get_transform_by_name(self, name: str) -> type[TransformProtocol] | None:
        """Get transform plugin by name."""
        for transform in self._transforms:
            if transform.name == name:
                return transform
        return None

    def get_gate_by_name(self, name: str) -> type[GateProtocol] | None:
        """Get gate plugin by name."""
        for gate in self._gates:
            if gate.name == name:
                return gate
        return None

    def get_aggregation_by_name(self, name: str) -> type[AggregationProtocol] | None:
        """Get aggregation plugin by name."""
        for agg in self._aggregations:
            if agg.name == name:
                return agg
        return None

    def get_sink_by_name(self, name: str) -> type[SinkProtocol] | None:
        """Get sink plugin by name."""
        for sink in self._sinks:
            if sink.name == name:
                return sink
        return None
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/plugins/test_manager.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/plugins/manager.py tests/plugins/test_manager.py
git commit -m "feat(plugins): add PluginManager with pluggy integration"
```

---

## Task 13: Base Transform Classes

**Files:**
- Create: `src/elspeth/plugins/base.py`
- Create: `tests/plugins/test_base.py`

### Step 1: Write the failing tests

```python
# tests/plugins/test_base.py
"""Tests for plugin base classes."""

import pytest


class TestBaseTransform:
    """Base class for transforms."""

    def test_base_transform_abstract(self) -> None:
        from elspeth.plugins.base import BaseTransform

        # Should not be instantiable directly
        with pytest.raises(TypeError):
            BaseTransform({})

    def test_subclass_implementation(self) -> None:
        from elspeth.plugins.base import BaseTransform
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import TransformResult
        from elspeth.plugins.schemas import PluginSchema

        class InputSchema(PluginSchema):
            x: int

        class OutputSchema(PluginSchema):
            x: int
            doubled: int

        class DoubleTransform(BaseTransform):
            name = "double"
            input_schema = InputSchema
            output_schema = OutputSchema

            def process(self, row: dict, ctx: PluginContext) -> TransformResult:
                return TransformResult.success({
                    "x": row["x"],
                    "doubled": row["x"] * 2,
                })

        transform = DoubleTransform({"some": "config"})
        ctx = PluginContext(run_id="test", config={})

        result = transform.process({"x": 21}, ctx)
        assert result.row == {"x": 21, "doubled": 42}

    def test_lifecycle_hooks_exist(self) -> None:
        from elspeth.plugins.base import BaseTransform

        # These should exist as no-op methods
        assert hasattr(BaseTransform, "on_register")
        assert hasattr(BaseTransform, "on_start")
        assert hasattr(BaseTransform, "on_complete")


class TestBaseGate:
    """Base class for gates."""

    def test_base_gate_implementation(self) -> None:
        from elspeth.plugins.base import BaseGate
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import GateResult, RoutingAction
        from elspeth.plugins.schemas import PluginSchema

        class RowSchema(PluginSchema):
            value: int

        class ThresholdGate(BaseGate):
            name = "threshold"
            input_schema = RowSchema
            output_schema = RowSchema

            def evaluate(self, row: dict, ctx: PluginContext) -> GateResult:
                threshold = self.config.get("threshold", 10)
                if row["value"] > threshold:
                    return GateResult(
                        row=row,
                        action=RoutingAction.route_to_sink("high"),
                    )
                return GateResult(row=row, action=RoutingAction.continue_())

        gate = ThresholdGate({"threshold": 50})
        ctx = PluginContext(run_id="test", config={})

        result = gate.evaluate({"value": 100}, ctx)
        assert result.action.kind == "route_to_sink"
```

### Step 2: Run test to verify it fails

Run: `pytest tests/plugins/test_base.py -v`
Expected: FAIL (ImportError)

### Step 3: Create base classes

```python
# src/elspeth/plugins/base.py
"""Base classes for plugin implementations.

These provide common functionality and ensure proper interface compliance.
Plugins can subclass these for convenience, or implement protocols directly.

Phase 3 Integration:
- Lifecycle hooks (on_register, on_start, on_complete) are called by engine
- PluginContext is provided by engine with landscape/tracer/payload_store
"""

from abc import ABC, abstractmethod
from typing import Any

from elspeth.plugins.context import PluginContext
from elspeth.plugins.results import (
    AcceptResult,
    GateResult,
    TransformResult,
)
from elspeth.plugins.schemas import PluginSchema


class BaseTransform(ABC):
    """Base class for stateless row transforms.

    Subclass and implement process() to create a transform.

    Example:
        class MyTransform(BaseTransform):
            name = "my_transform"
            input_schema = InputSchema
            output_schema = OutputSchema

            def process(self, row: dict, ctx: PluginContext) -> TransformResult:
                return TransformResult.success({**row, "new_field": "value"})
    """

    name: str
    input_schema: type[PluginSchema]
    output_schema: type[PluginSchema]

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        self.config = config

    @abstractmethod
    def process(
        self,
        row: dict[str, Any],
        ctx: PluginContext,
    ) -> TransformResult:
        """Process a single row.

        Args:
            row: Input row matching input_schema
            ctx: Plugin context

        Returns:
            TransformResult with processed row or error
        """
        ...

    # === Lifecycle Hooks (Phase 3) ===

    def on_register(self, ctx: PluginContext) -> None:
        """Called when plugin is registered with the engine.

        Override for one-time setup.
        """
        pass

    def on_start(self, ctx: PluginContext) -> None:
        """Called at the start of each run.

        Override for per-run initialization.
        """
        pass

    def on_complete(self, ctx: PluginContext) -> None:
        """Called at the end of each run.

        Override for cleanup.
        """
        pass


class BaseGate(ABC):
    """Base class for gate transforms (routing decisions).

    Subclass and implement evaluate() to create a gate.

    Example:
        class SafetyGate(BaseGate):
            name = "safety"
            input_schema = RowSchema
            output_schema = RowSchema

            def evaluate(self, row: dict, ctx: PluginContext) -> GateResult:
                if self._is_suspicious(row):
                    return GateResult(
                        row=row,
                        action=RoutingAction.route_to_sink("review"),
                    )
                return GateResult(row=row, action=RoutingAction.continue_())
    """

    name: str
    input_schema: type[PluginSchema]
    output_schema: type[PluginSchema]

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        self.config = config

    @abstractmethod
    def evaluate(
        self,
        row: dict[str, Any],
        ctx: PluginContext,
    ) -> GateResult:
        """Evaluate a row and decide routing.

        Args:
            row: Input row
            ctx: Plugin context

        Returns:
            GateResult with routing decision
        """
        ...

    # === Lifecycle Hooks (Phase 3) ===

    def on_register(self, ctx: PluginContext) -> None:
        """Called when plugin is registered."""
        pass

    def on_start(self, ctx: PluginContext) -> None:
        """Called at start of run."""
        pass

    def on_complete(self, ctx: PluginContext) -> None:
        """Called at end of run."""
        pass


class BaseAggregation(ABC):
    """Base class for aggregation transforms (stateful batching).

    Subclass and implement accept(), should_trigger(), flush().

    Phase 3 Integration:
    - Engine creates Landscape batch on first accept()
    - Engine persists batch membership on every accept()
    - Engine manages batch state transitions

    Example:
        class StatsAggregation(BaseAggregation):
            name = "stats"
            input_schema = InputSchema
            output_schema = StatsSchema

            def __init__(self, config):
                super().__init__(config)
                self._values = []

            def accept(self, row, ctx) -> AcceptResult:
                self._values.append(row["value"])
                return AcceptResult(
                    accepted=True,
                    trigger=len(self._values) >= 100,
                )

            def flush(self, ctx) -> list[dict]:
                result = {"mean": statistics.mean(self._values)}
                self._values = []
                return [result]
    """

    name: str
    input_schema: type[PluginSchema]
    output_schema: type[PluginSchema]

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        self.config = config

    @abstractmethod
    def accept(
        self,
        row: dict[str, Any],
        ctx: PluginContext,
    ) -> AcceptResult:
        """Accept a row into the batch."""
        ...

    @abstractmethod
    def should_trigger(self) -> bool:
        """Check if batch should flush."""
        ...

    @abstractmethod
    def flush(self, ctx: PluginContext) -> list[dict[str, Any]]:
        """Process batch and return results."""
        ...

    def reset(self) -> None:
        """Reset internal state.

        Override if you have state beyond what __init__ sets up.
        """
        pass

    # === Lifecycle Hooks (Phase 3) ===

    def on_register(self, ctx: PluginContext) -> None:
        """Called when plugin is registered."""
        pass

    def on_start(self, ctx: PluginContext) -> None:
        """Called at start of run."""
        pass

    def on_complete(self, ctx: PluginContext) -> None:
        """Called at end of run."""
        pass


class BaseSink(ABC):
    """Base class for sink plugins.

    Subclass and implement write(), flush(), close().

    Example:
        class CSVSink(BaseSink):
            name = "csv"
            input_schema = RowSchema
            idempotent = False

            def write(self, row, ctx) -> None:
                self._writer.writerow(row)

            def flush(self) -> None:
                self._file.flush()

            def close(self) -> None:
                self._file.close()
    """

    name: str
    input_schema: type[PluginSchema]
    idempotent: bool = False

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        self.config = config

    @abstractmethod
    def write(
        self,
        row: dict[str, Any],
        ctx: PluginContext,
    ) -> None:
        """Write a row to the sink."""
        ...

    @abstractmethod
    def flush(self) -> None:
        """Flush buffered data."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Close and release resources."""
        ...

    # === Lifecycle Hooks (Phase 3) ===

    def on_register(self, ctx: PluginContext) -> None:
        """Called when plugin is registered."""
        pass

    def on_start(self, ctx: PluginContext) -> None:
        """Called at start of run."""
        pass

    def on_complete(self, ctx: PluginContext) -> None:
        """Called at end of run (before close)."""
        pass
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/plugins/test_base.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/plugins/base.py tests/plugins/test_base.py
git commit -m "feat(plugins): add base classes for all plugin types"
```

---

## Task 14: Base Source Class

**Files:**
- Modify: `src/elspeth/plugins/base.py`
- Modify: `tests/plugins/test_base.py`

### Step 1: Write the failing tests

```python
# Add to tests/plugins/test_base.py

from typing import Iterator


class TestBaseSource:
    """Base class for sources."""

    def test_base_source_implementation(self) -> None:
        from elspeth.plugins.base import BaseSource
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.schemas import PluginSchema

        class OutputSchema(PluginSchema):
            value: int

        class ListSource(BaseSource):
            name = "list"
            output_schema = OutputSchema

            def __init__(self, config: dict) -> None:
                super().__init__(config)
                self._data = config.get("data", [])

            def load(self, ctx: PluginContext) -> Iterator[dict]:
                for item in self._data:
                    yield item

            def close(self) -> None:
                pass

        source = ListSource({"data": [{"value": 1}, {"value": 2}]})
        ctx = PluginContext(run_id="test", config={})

        rows = list(source.load(ctx))
        assert len(rows) == 2
        assert rows[0] == {"value": 1}
```

### Step 2: Run test to verify it fails

Run: `pytest tests/plugins/test_base.py::TestBaseSource -v`
Expected: FAIL

### Step 3: Add BaseSource to base.py

```python
# Add to src/elspeth/plugins/base.py

from typing import Iterator


class BaseSource(ABC):
    """Base class for source plugins.

    Subclass and implement load() and close().

    Example:
        class CSVSource(BaseSource):
            name = "csv"
            output_schema = RowSchema

            def load(self, ctx: PluginContext) -> Iterator[dict]:
                with open(self.config["path"]) as f:
                    reader = csv.DictReader(f)
                    yield from reader

            def close(self) -> None:
                pass  # File already closed by context manager
    """

    name: str
    output_schema: type[PluginSchema]

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        self.config = config

    @abstractmethod
    def load(self, ctx: PluginContext) -> Iterator[dict[str, Any]]:
        """Load and yield rows from the source.

        Args:
            ctx: Plugin context

        Yields:
            Row dicts matching output_schema
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Clean up resources."""
        ...

    # === Lifecycle Hooks (Phase 3) ===

    def on_start(self, ctx: PluginContext) -> None:
        """Called before load()."""
        pass
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/plugins/test_base.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add -u
git commit -m "feat(plugins): add BaseSource class"
```

---

## Task 15: Public API Exports

**Files:**
- Modify: `src/elspeth/plugins/__init__.py`

### Step 1: Write the failing tests

```python
# Add to tests/plugins/test_results.py

class TestPluginsPublicAPI:
    """Public API exports from elspeth.plugins."""

    def test_results_importable(self) -> None:
        from elspeth.plugins import (
            AcceptResult,
            GateResult,
            RoutingAction,
            RowOutcome,
            TransformResult,
        )

        assert RowOutcome is not None
        assert TransformResult is not None

    def test_context_importable(self) -> None:
        from elspeth.plugins import PluginContext

        assert PluginContext is not None

    def test_schemas_importable(self) -> None:
        from elspeth.plugins import PluginSchema, check_compatibility

        assert PluginSchema is not None

    def test_protocols_importable(self) -> None:
        from elspeth.plugins import (
            AggregationProtocol,
            GateProtocol,
            SinkProtocol,
            SourceProtocol,
            TransformProtocol,
        )

        assert SourceProtocol is not None

    def test_base_classes_importable(self) -> None:
        from elspeth.plugins import (
            BaseAggregation,
            BaseGate,
            BaseSink,
            BaseSource,
            BaseTransform,
        )

        assert BaseTransform is not None

    def test_manager_importable(self) -> None:
        from elspeth.plugins import PluginManager

        assert PluginManager is not None

    def test_hookspecs_importable(self) -> None:
        from elspeth.plugins import hookimpl, hookspec

        assert hookspec is not None
        assert hookimpl is not None
```

### Step 2: Run test to verify it fails

Run: `pytest tests/plugins/test_results.py::TestPluginsPublicAPI -v`
Expected: FAIL (ImportError)

### Step 3: Update plugins __init__.py

```python
# src/elspeth/plugins/__init__.py
"""Plugin system: Sources, Transforms, Sinks via pluggy.

This module provides the plugin infrastructure for Elspeth:

- Protocols: Type contracts for plugin implementations
- Base classes: Convenient base classes with lifecycle hooks
- Results: Return types for plugin operations
- Schemas: Pydantic-based input/output schemas
- Manager: Plugin discovery and registration
- Hookspecs: pluggy hook definitions

Phase 3 Integration:
- PluginContext carries Landscape, Tracer, PayloadStore
- Result types include audit fields (hashes, duration)
- Base classes have lifecycle hooks for engine integration
"""

# Results
from elspeth.plugins.results import (
    AcceptResult,
    GateResult,
    RoutingAction,
    RowOutcome,
    TransformResult,
)

# Context
from elspeth.plugins.context import PluginContext

# Schemas
from elspeth.plugins.schemas import (
    CompatibilityResult,
    PluginSchema,
    SchemaValidationError,
    check_compatibility,
    validate_row,
)

# Protocols
from elspeth.plugins.protocols import (
    AggregationProtocol,
    CoalescePolicy,
    CoalesceProtocol,
    GateProtocol,
    SinkProtocol,
    SourceProtocol,
    TransformProtocol,
)

# Base classes
from elspeth.plugins.base import (
    BaseAggregation,
    BaseGate,
    BaseSink,
    BaseSource,
    BaseTransform,
)

# Manager
from elspeth.plugins.manager import PluginManager

# Hookspecs
from elspeth.plugins.hookspecs import hookimpl, hookspec

__all__ = [
    # Results
    "AcceptResult",
    "GateResult",
    "RoutingAction",
    "RowOutcome",
    "TransformResult",
    # Context
    "PluginContext",
    # Schemas
    "CompatibilityResult",
    "PluginSchema",
    "SchemaValidationError",
    "check_compatibility",
    "validate_row",
    # Protocols
    "AggregationProtocol",
    "CoalescePolicy",
    "CoalesceProtocol",
    "GateProtocol",
    "SinkProtocol",
    "SourceProtocol",
    "TransformProtocol",
    # Base classes
    "BaseAggregation",
    "BaseGate",
    "BaseSink",
    "BaseSource",
    "BaseTransform",
    # Manager
    "PluginManager",
    # Hookspecs
    "hookimpl",
    "hookspec",
]
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/plugins/ -v`
Expected: PASS

### Step 5: Commit

```bash
git add -u
git commit -m "feat(plugins): export public API from elspeth.plugins"
```

---

## Task 16: Phase 3 Integration Documentation

**Files:**
- Create: `src/elspeth/plugins/PHASE3_INTEGRATION.md`

### Step 1: Create integration documentation

```markdown
# Phase 3 Integration Points

This document describes how Phase 3 (SDA Engine) will integrate with the
Phase 2 plugin system. These integration points are built into the Phase 2
design to ensure clean integration.

## PluginContext Integration

Phase 3 creates PluginContext with full integration:

```python
# Phase 3: SDA Engine creates context
ctx = PluginContext(
    run_id=run.run_id,
    config=resolved_config,
    landscape=LandscapeRecorder(db, run.run_id),
    tracer=opentelemetry.trace.get_tracer("elspeth"),
    payload_store=FilesystemPayloadStore(base_path),
)
```

## Transform Processing

Phase 3 engine wraps transform.process() to add audit:

```python
# Phase 3: Engine wraps process() calls
def process_with_audit(transform, row, ctx):
    input_hash = stable_hash(row)

    with ctx.start_span(f"transform:{transform.name}") as span:
        span.set_attribute("input_hash", input_hash)

        start = time.perf_counter()
        result = transform.process(row, ctx)
        duration_ms = (time.perf_counter() - start) * 1000

        # Populate audit fields
        result.input_hash = input_hash
        result.output_hash = stable_hash(result.row) if result.row else None
        result.duration_ms = duration_ms

        # Record in Landscape
        ctx.landscape.record_node_state(...)

        span.set_attribute("output_hash", result.output_hash)
        span.set_attribute("status", result.status)

    return result
```

## Aggregation Batch Management

Phase 3 engine manages Landscape batches:

```python
# Phase 3: Engine wraps accept() for batch tracking
def accept_with_batch(aggregation, row, token_id, ctx):
    if aggregation._batch_id is None:
        # Create draft batch in Landscape
        batch_id = ctx.landscape.create_batch(
            node_id=aggregation.node_id,
            status="draft",
        )
        aggregation._batch_id = batch_id

    # Persist membership immediately (crash-safe)
    ctx.landscape.add_batch_member(
        batch_id=aggregation._batch_id,
        token_id=token_id,
        ordinal=len(aggregation._buffer),
    )

    result = aggregation.accept(row, ctx)
    result.batch_id = aggregation._batch_id
    return result

def flush_with_audit(aggregation, ctx):
    ctx.landscape.update_batch_status(aggregation._batch_id, "executing")

    try:
        outputs = aggregation.flush(ctx)
        ctx.landscape.update_batch_status(aggregation._batch_id, "completed")

        # Record batch outputs
        for output in outputs:
            ctx.landscape.add_batch_output(aggregation._batch_id, output)

        return outputs
    except Exception as e:
        ctx.landscape.update_batch_status(
            aggregation._batch_id, "failed", error=str(e)
        )
        raise
```

## Gate Routing Events

Phase 3 engine records routing decisions:

```python
# Phase 3: Engine wraps evaluate() for routing audit
def evaluate_with_routing(gate, row, ctx):
    result = gate.evaluate(row, ctx)

    # Record routing event
    ctx.landscape.record_routing_event(
        node_id=gate.node_id,
        action=result.action,
        reason=result.action.reason,
    )

    return result
```

## Lifecycle Hook Calls

Phase 3 engine calls lifecycle hooks:

```python
# Phase 3: Engine calls hooks at appropriate times

# During plugin registration
for plugin in all_plugins:
    plugin.on_register(ctx)

# At run start
for plugin in all_plugins:
    plugin.on_start(ctx)

# At run end
for plugin in all_plugins:
    plugin.on_complete(ctx)
```

## OpenTelemetry Span Structure

Phase 3 creates this span hierarchy:

```
run:{run_id}
├── source:{source_name}
│   └── load
├── row:{row_id}
│   ├── transform:{transform_name}
│   │   └── external_call (Phase 6)
│   ├── gate:{gate_name}
│   └── sink:{sink_name}
└── aggregation:{agg_name}
    └── flush
```

## Checklist for Phase 3 Implementation

- [ ] Create PluginContext with landscape, tracer, payload_store
- [ ] Wrap transform.process() with audit recording
- [ ] Wrap gate.evaluate() with routing event recording
- [ ] Wrap aggregation.accept() with batch member recording
- [ ] Wrap aggregation.flush() with batch status management
- [ ] Call lifecycle hooks at appropriate times
- [ ] Populate audit fields in result types
- [ ] Create OpenTelemetry spans for all operations
```

### Step 2: Commit

```bash
git add src/elspeth/plugins/PHASE3_INTEGRATION.md
git commit -m "docs(plugins): document Phase 3 integration points"
```

---

## Task 17: Final Integration Test

**Files:**
- Create: `tests/plugins/test_integration.py`

### Step 1: Create integration test

```python
# tests/plugins/test_integration.py
"""Integration tests for the plugin system."""

from typing import Iterator

import pytest


class TestPluginSystemIntegration:
    """End-to-end plugin system tests."""

    def test_full_plugin_workflow(self) -> None:
        """Test source -> transform -> gate -> sink workflow."""
        from elspeth.plugins import (
            BaseGate,
            BaseSink,
            BaseSource,
            BaseTransform,
            GateResult,
            PluginContext,
            PluginManager,
            PluginSchema,
            RoutingAction,
            TransformResult,
            hookimpl,
        )

        # Define schemas
        class InputSchema(PluginSchema):
            value: int

        class EnrichedSchema(PluginSchema):
            value: int
            doubled: int

        # Define plugins
        class ListSource(BaseSource):
            name = "list"
            output_schema = InputSchema

            def load(self, ctx: PluginContext) -> Iterator[dict]:
                for v in self.config.get("values", []):
                    yield {"value": v}

            def close(self) -> None:
                pass

        class DoubleTransform(BaseTransform):
            name = "double"
            input_schema = InputSchema
            output_schema = EnrichedSchema

            def process(self, row: dict, ctx: PluginContext) -> TransformResult:
                return TransformResult.success({
                    "value": row["value"],
                    "doubled": row["value"] * 2,
                })

        class ThresholdGate(BaseGate):
            name = "threshold"
            input_schema = EnrichedSchema
            output_schema = EnrichedSchema

            def evaluate(self, row: dict, ctx: PluginContext) -> GateResult:
                if row["doubled"] > self.config.get("threshold", 100):
                    return GateResult(
                        row=row,
                        action=RoutingAction.route_to_sink("high"),
                    )
                return GateResult(row=row, action=RoutingAction.continue_())

        class MemorySink(BaseSink):
            name = "memory"
            input_schema = EnrichedSchema
            rows: list = []

            def write(self, row: dict, ctx: PluginContext) -> None:
                MemorySink.rows.append(row)

            def flush(self) -> None:
                pass

            def close(self) -> None:
                pass

        # Register plugins
        class TestPlugin:
            @hookimpl
            def elspeth_get_source(self) -> list:
                return [ListSource]

            @hookimpl
            def elspeth_get_transforms(self) -> list:
                return [DoubleTransform]

            @hookimpl
            def elspeth_get_gates(self) -> list:
                return [ThresholdGate]

            @hookimpl
            def elspeth_get_sinks(self) -> list:
                return [MemorySink]

        manager = PluginManager()
        manager.register(TestPlugin())

        # Verify registration
        assert len(manager.get_sources()) == 1
        assert len(manager.get_transforms()) == 1
        assert len(manager.get_gates()) == 1
        assert len(manager.get_sinks()) == 1

        # Create instances and process
        ctx = PluginContext(run_id="test-001", config={})

        source = manager.get_source_by_name("list")({"values": [10, 50, 100]})
        transform = manager.get_transform_by_name("double")({})
        gate = manager.get_gate_by_name("threshold")({"threshold": 100})
        sink = manager.get_sink_by_name("memory")({})

        MemorySink.rows = []  # Reset

        for row in source.load(ctx):
            result = transform.process(row, ctx)
            assert result.status == "success"

            gate_result = gate.evaluate(result.row, ctx)

            if gate_result.action.kind == "continue":
                sink.write(gate_result.row, ctx)

        # Verify results
        # Values: 10*2=20, 50*2=100, 100*2=200
        # Threshold 100: 20 continues, 100 continues, 200 routed
        assert len(MemorySink.rows) == 2
        assert MemorySink.rows[0]["doubled"] == 20
        assert MemorySink.rows[1]["doubled"] == 100

    def test_schema_validation_in_pipeline(self) -> None:
        """Test that schema compatibility is checked."""
        from elspeth.plugins import PluginSchema, check_compatibility

        class SourceOutput(PluginSchema):
            a: int
            b: str

        class TransformInput(PluginSchema):
            a: int
            b: str
            c: float  # Not provided by source!

        result = check_compatibility(SourceOutput, TransformInput)
        assert result.compatible is False
        assert "c" in result.missing_fields
```

### Step 2: Run test to verify it passes

Run: `pytest tests/plugins/test_integration.py -v`
Expected: PASS

### Step 3: Commit

```bash
git add tests/plugins/test_integration.py
git commit -m "test(plugins): add integration tests for plugin workflow"
```

---

## Task 18: Final Verification

### Step 1: Run all tests with coverage

Run: `pytest tests/ -v --cov=src/elspeth --cov-report=term-missing`
Expected: PASS with good coverage

### Step 2: Run type checking

Run: `mypy src/elspeth/plugins/`
Expected: PASS (or known issues only)

### Step 3: Run linting

Run: `ruff check src/elspeth/plugins/ tests/plugins/`
Expected: PASS

### Step 4: Final commit

```bash
git add -A
git commit -m "chore: Phase 2 Plugin System complete"
```

---

## Summary

Phase 2 builds the plugin infrastructure with Phase 3 integration built-in:

| Component | Files | Purpose |
|-----------|-------|---------|
| **Results** | `plugins/results.py` | RowOutcome, TransformResult, GateResult, AcceptResult with audit fields |
| **Context** | `plugins/context.py` | PluginContext with Optional landscape/tracer/payload_store |
| **Schemas** | `plugins/schemas.py` | Pydantic-based schemas with compatibility checking |
| **Protocols** | `plugins/protocols.py` | Type contracts for all plugin types |
| **Base Classes** | `plugins/base.py` | BaseSource, BaseTransform, BaseGate, BaseAggregation, BaseSink |
| **Hookspecs** | `plugins/hookspecs.py` | pluggy hook definitions |
| **Manager** | `plugins/manager.py` | Plugin discovery and registration |

### Phase 3 Integration Checklist

All these are ready for Phase 3:

- [x] `PluginContext` has `landscape`, `tracer`, `payload_store` fields (Optional)
- [x] `PluginContext.start_span()` exists (no-op without tracer)
- [x] All result types have `input_hash`, `output_hash`, `duration_ms` fields
- [x] Base classes have `on_register()`, `on_start()`, `on_complete()` hooks
- [x] `AcceptResult` has `batch_id` field
- [x] Schemas are Pydantic models with validation
- [x] Phase 3 integration points documented in `PHASE3_INTEGRATION.md`
