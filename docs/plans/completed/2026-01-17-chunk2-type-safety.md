# Chunk 2: Type Safety & Contracts Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate stringly-typed APIs and consolidate data contracts for type-safe cross-subsystem communication.

**Architecture:** Move PluginSchema implementation to contracts (single source of truth), convert model type hints to enums (repositories already convert at runtime), wire up schema validation in orchestrator, and replace all string literal comparisons with enum members.

**Tech Stack:** Pydantic v2, Python dataclasses, TypedDict, str-Enum pattern

---

## Task 1: Consolidate PluginSchema to Contracts

**Files:**
- Modify: `src/elspeth/contracts/data.py`
- Modify: `src/elspeth/plugins/schemas.py`
- Modify: `src/elspeth/engine/schema_validator.py` (update import path)
- Test: `tests/contracts/test_plugin_schema.py` (CREATE)

### ⚠️ IMPORTANT: Trust Boundary Context

**Current `contracts/data.py` has WRONG defaults** that will be replaced:
```python
# WRONG (current stub - treats plugin data as "Our Data"):
model_config = {"frozen": True, "extra": "forbid"}

# CORRECT (this plan adopts - treats plugin data as "Their Data"):
model_config = ConfigDict(extra="ignore", strict=False, frozen=False)
```

**Why the permissive version is correct per Data Manifesto:**

PluginSchema validates **"Their Data"** (user rows from sources, transform outputs) - NOT "Our Data" (audit trail). Per CLAUDE.md:

> **Their Data** - Can be literal trash. We don't control what users feed us.
> Validate at the boundary, record what we got, continue processing.

Therefore:
- `extra="ignore"` - CSVs may have 50 columns; transform only needs 3. Don't reject.
- `strict=False` - JSON has no integer type. Allow `1.0` → `int` coercion.
- `frozen=False` - Transforms modify row data. Schemas must be mutable.

The strict version in `contracts/data.py` was an **error during centralization** - a stub created with defensive defaults appropriate for internal data, not trust boundaries. This task corrects that mistake.

### Step 1: Write failing test for PluginSchema in contracts

```python
# tests/contracts/test_plugin_schema.py
"""Tests for PluginSchema in contracts module."""

import pytest
from pydantic import ValidationError


class TestPluginSchemaLocation:
    """Verify PluginSchema is importable from contracts."""

    def test_plugin_schema_importable_from_contracts(self) -> None:
        """PluginSchema should be importable from elspeth.contracts."""
        from elspeth.contracts import PluginSchema

        # Should have the correct config for "Their Data" trust boundary
        # (permissive validation, not strict "Our Data" settings)
        assert PluginSchema.model_config.get("extra") == "ignore"
        assert PluginSchema.model_config.get("frozen") is False

    def test_schema_validation_error_importable_from_contracts(self) -> None:
        """SchemaValidationError should be importable from contracts."""
        from elspeth.contracts import SchemaValidationError

        error = SchemaValidationError("field", "message", "value")
        assert error.field == "field"
        assert error.message == "message"

    def test_compatibility_result_importable_from_contracts(self) -> None:
        """CompatibilityResult should be importable from contracts."""
        from elspeth.contracts import CompatibilityResult

        result = CompatibilityResult(compatible=True)
        assert result.compatible is True

    def test_validate_row_importable_from_contracts(self) -> None:
        """validate_row should be importable from contracts."""
        from elspeth.contracts import PluginSchema, validate_row

        class TestSchema(PluginSchema):
            name: str

        errors = validate_row({"name": "test"}, TestSchema)
        assert errors == []

    def test_check_compatibility_importable_from_contracts(self) -> None:
        """check_compatibility should be importable from contracts."""
        from elspeth.contracts import PluginSchema, check_compatibility

        class SchemaA(PluginSchema):
            name: str

        class SchemaB(PluginSchema):
            name: str

        result = check_compatibility(SchemaA, SchemaB)
        assert result.compatible is True


class TestPluginSchemaReExport:
    """Verify plugins/schemas.py re-exports from contracts."""

    def test_plugin_schema_same_class(self) -> None:
        """Both imports should return the same class."""
        from elspeth.contracts import PluginSchema as ContractsSchema
        from elspeth.plugins.schemas import PluginSchema as PluginsSchema

        assert ContractsSchema is PluginsSchema
```

### Step 2: Run test to verify it fails

Run: `pytest tests/contracts/test_plugin_schema.py -v`
Expected: FAIL - imports will fail or config won't match

### Step 3: Move full PluginSchema implementation to contracts/data.py

Replace the entire contents of `src/elspeth/contracts/data.py`:

```python
# src/elspeth/contracts/data.py
"""Plugin data schema contracts.

PluginSchema is the base class for plugin input/output schemas.
Plugins declare their expected data shape by subclassing this.

This is the CANONICAL location - plugins/schemas.py re-exports from here.
"""

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from types import UnionType
from typing import Any, TypeVar, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, ValidationError

T = TypeVar("T", bound="PluginSchema")


class PluginSchema(BaseModel):
    """Base class for plugin input/output schemas.

    TRUST BOUNDARY: "Their Data"
    ============================
    PluginSchema validates user data entering the system (source rows,
    transform outputs) - NOT audit trail data. Per the Data Manifesto:

    - "Their Data" can be literal trash - validate, record, continue
    - "Our Data" (audit trail) must be pristine - crash on anomaly

    Therefore this schema uses PERMISSIVE settings:
    - extra="ignore": CSVs may have extra columns the plugin doesn't need
    - strict=False: Allow coercion (JSON's 1.0 → int is fine)
    - frozen=False: Transforms need to modify row data

    Usage:
        class MyInputSchema(PluginSchema):
            temperature: float
            humidity: float
    """

    model_config = ConfigDict(
        # PERMISSIVE settings for "Their Data" trust boundary:
        extra="ignore",  # Rows may have extra fields - don't reject
        strict=False,  # Allow type coercion (int -> float, etc.)
        frozen=False,  # Allow modification by transforms
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
            errors.append(
                SchemaValidationError(
                    field=field,
                    message=error["msg"],
                    value=error.get("input"),
                )
            )
        return errors


@dataclass
class CompatibilityResult:
    """Result of schema compatibility check."""

    compatible: bool
    missing_fields: list[str] = dataclass_field(default_factory=list)
    type_mismatches: list[tuple[str, str, str]] = dataclass_field(default_factory=list)

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
                f"{name} (expected {expected}, got {actual})"
                for name, expected, actual in self.type_mismatches
            ]
            parts.append(f"Type mismatches: {', '.join(mismatches)}")

        return "; ".join(parts)


def check_compatibility(
    producer_schema: type[PluginSchema],
    consumer_schema: type[PluginSchema],
) -> CompatibilityResult:
    """Check if producer output is compatible with consumer input.

    Uses Pydantic model_fields metadata for accurate compatibility checking.
    This handles optional fields, unions, constrained types, and defaults.

    Compatibility means:
    - All REQUIRED fields in consumer are provided by producer
    - Fields with defaults in consumer are optional
    - Field types are compatible (exact match or coercible)

    Args:
        producer_schema: Output schema of upstream plugin
        consumer_schema: Input schema of downstream plugin

    Returns:
        CompatibilityResult indicating compatibility and any issues
    """
    # Use Pydantic v2 model_fields for accurate field introspection
    producer_fields = producer_schema.model_fields
    consumer_fields = consumer_schema.model_fields

    missing: list[str] = []
    mismatches: list[tuple[str, str, str]] = []

    for field_name, consumer_field in consumer_fields.items():
        # Check if field is required (no default value)
        is_required = consumer_field.is_required()

        if field_name not in producer_fields:
            # Missing field - only a problem if required
            if is_required:
                missing.append(field_name)
        else:
            producer_field = producer_fields[field_name]
            if not _types_compatible(
                producer_field.annotation, consumer_field.annotation
            ):
                mismatches.append(
                    (
                        field_name,
                        _type_name(consumer_field.annotation),
                        _type_name(producer_field.annotation),
                    )
                )

    compatible = len(missing) == 0 and len(mismatches) == 0

    return CompatibilityResult(
        compatible=compatible,
        missing_fields=missing,
        type_mismatches=mismatches,
    )


def _type_name(t: Any) -> str:
    """Get readable name for a type annotation."""
    if hasattr(t, "__name__"):
        return str(t.__name__)
    return str(t)


def _is_union_type(t: Any) -> bool:
    """Check if type is a Union (typing.Union or types.UnionType)."""
    origin = get_origin(t)
    return origin is Union or isinstance(t, UnionType)


def _types_compatible(actual: Any, expected: Any) -> bool:
    """Check if actual type is compatible with expected type.

    Handles:
    - Exact matches
    - Numeric compatibility (int -> float)
    - Optional[X] on consumer side (producer can send X or X | None)
    - Union types (both typing.Union and X | Y syntax)
    """
    # Exact match
    if actual == expected:
        return True

    # Numeric compatibility (int -> float is OK)
    if expected is float and actual is int:
        return True

    # Handle Optional/Union types (both typing.Union and types.UnionType)
    if _is_union_type(expected):
        expected_args = get_args(expected)
        # Check if actual type matches any of the union members
        if actual in expected_args:
            return True
        # Check if actual is a Union that's a subset
        if _is_union_type(actual):
            actual_args = get_args(actual)
            return all(a in expected_args for a in actual_args)

    return False
```

### Step 4: Update contracts/__init__.py to export new items

Add to `src/elspeth/contracts/__init__.py`:

```python
from elspeth.contracts.data import (
    CompatibilityResult,
    PluginSchema,
    SchemaValidationError,
    check_compatibility,
    validate_row,
)

# Add to __all__:
__all__ = [
    # ... existing exports ...
    "CompatibilityResult",
    "PluginSchema",
    "SchemaValidationError",
    "check_compatibility",
    "validate_row",
]
```

### Step 5: Convert plugins/schemas.py to re-export shim

Replace entire contents of `src/elspeth/plugins/schemas.py`:

```python
# src/elspeth/plugins/schemas.py
"""Re-export schema types from contracts for backwards compatibility.

The canonical location is elspeth.contracts.data.
This module exists for import compatibility with existing code.
"""

from elspeth.contracts.data import (
    CompatibilityResult,
    PluginSchema,
    SchemaValidationError,
    check_compatibility,
    validate_row,
)

__all__ = [
    "CompatibilityResult",
    "PluginSchema",
    "SchemaValidationError",
    "check_compatibility",
    "validate_row",
]
```

### Step 6: Update schema_validator.py to import from contracts

Update `src/elspeth/engine/schema_validator.py` line 11:

```python
# Change from:
from elspeth.plugins.schemas import PluginSchema

# To:
from elspeth.contracts import PluginSchema
```

This ensures the engine imports directly from contracts (the canonical location) rather than through the re-export shim.

### Step 7: Run test to verify it passes

Run: `pytest tests/contracts/test_plugin_schema.py -v`
Expected: PASS

### Step 8: Run full test suite to verify no regressions

Run: `pytest tests/ -v --tb=short`
Expected: All existing tests pass

### Step 9: Commit

```bash
git add src/elspeth/contracts/data.py src/elspeth/contracts/__init__.py src/elspeth/plugins/schemas.py src/elspeth/engine/schema_validator.py tests/contracts/test_plugin_schema.py
git commit -m "$(cat <<'EOF'
refactor(contracts): consolidate PluginSchema to contracts module

Move full PluginSchema implementation from plugins/schemas.py to
contracts/data.py. The plugins/schemas.py now re-exports from contracts
for backwards compatibility.

TRUST BOUNDARY: PluginSchema validates "Their Data" (user rows), so uses
permissive settings (extra="ignore", strict=False, frozen=False) per
Data Manifesto. The previous contracts/data.py stub had incorrect strict
settings that have been corrected.

Includes: PluginSchema, SchemaValidationError, CompatibilityResult,
validate_row(), check_compatibility()

Also updates schema_validator.py to import from contracts directly.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Convert Models to Use Enum Types

**Files:**
- Modify: `src/elspeth/core/landscape/models.py`
- Test: `tests/core/landscape/test_models_enums.py` (CREATE)

### ⚠️ NOTE: Dual Model Definitions

There are currently TWO sets of model definitions:
- `core/landscape/models.py` - stringly-typed (Run, Node, Edge, etc.)
- `contracts/audit.py` - enum-typed (already has proper types)

**Why both exist:** `contracts/audit.py` was created during centralization as the "correct" version, but `core/landscape/models.py` wasn't deleted or updated. The repositories use `contracts/audit.py` internally.

**This task:** Updates `core/landscape/models.py` type hints to match `contracts/audit.py`. A future Chunk 3 task should consider consolidating to a single location.

### Step 1: Write failing test for enum-typed models

```python
# tests/core/landscape/test_models_enums.py
"""Tests for enum-typed model fields."""

from datetime import datetime

import pytest

from elspeth.contracts.enums import (
    Determinism,
    ExportStatus,
    NodeType,
    RoutingMode,
)
from elspeth.core.landscape.models import Edge, Node, Run


class TestModelEnumTypes:
    """Verify model fields use enum types, not strings."""

    def test_run_export_status_accepts_enum(self) -> None:
        """Run.export_status should accept ExportStatus enum."""
        run = Run(
            run_id="run-1",
            started_at=datetime.now(),
            config_hash="abc",
            settings_json="{}",
            canonical_version="v1",
            status="completed",
            export_status=ExportStatus.PENDING,
        )
        assert run.export_status == ExportStatus.PENDING

    def test_node_type_accepts_enum(self) -> None:
        """Node.node_type should accept NodeType enum."""
        node = Node(
            node_id="node-1",
            run_id="run-1",
            plugin_name="test",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            determinism=Determinism.DETERMINISTIC,
            config_hash="abc",
            config_json="{}",
            registered_at=datetime.now(),
        )
        assert node.node_type == NodeType.TRANSFORM

    def test_node_determinism_accepts_enum(self) -> None:
        """Node.determinism should accept Determinism enum."""
        node = Node(
            node_id="node-1",
            run_id="run-1",
            plugin_name="test",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            determinism=Determinism.IO_READ,
            config_hash="abc",
            config_json="{}",
            registered_at=datetime.now(),
        )
        assert node.determinism == Determinism.IO_READ

    def test_edge_default_mode_accepts_enum(self) -> None:
        """Edge.default_mode should accept RoutingMode enum."""
        edge = Edge(
            edge_id="edge-1",
            run_id="run-1",
            from_node_id="node-1",
            to_node_id="node-2",
            label="continue",
            default_mode=RoutingMode.MOVE,
            created_at=datetime.now(),
        )
        assert edge.default_mode == RoutingMode.MOVE
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/landscape/test_models_enums.py -v`
Expected: FAIL - type errors because fields are typed as str

### Step 3: Update models.py to use enum types

Update `src/elspeth/core/landscape/models.py`:

```python
# Add imports at top:
from elspeth.contracts import (
    Determinism,
    ExportStatus,
    NodeType,
    RoutingMode,
    NodeStateStatus,
    RunStatus,
)

# Update Run dataclass (line ~36):
    export_status: ExportStatus | None = None

# Update Node dataclass (lines ~50, ~52):
    node_type: NodeType
    # ...
    determinism: Determinism

# Update Edge dataclass (line ~69):
    default_mode: RoutingMode
```

### Step 4: Run test to verify it passes

Run: `pytest tests/core/landscape/test_models_enums.py -v`
Expected: PASS

### Step 5: Run full test suite

Run: `pytest tests/ -v --tb=short`
Expected: All tests pass (repositories already convert strings to enums)

### Step 6: Commit

```bash
git add src/elspeth/core/landscape/models.py tests/core/landscape/test_models_enums.py
git commit -m "$(cat <<'EOF'
refactor(models): use enum types instead of strings

Convert Node.node_type to NodeType, Node.determinism to Determinism,
Edge.default_mode to RoutingMode, Run.export_status to ExportStatus.

Repositories already convert strings to enums when loading from DB,
so this is a type hint alignment change only.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Create TypedDicts for Update Schemas

**Files:**
- Modify: `src/elspeth/contracts/audit.py`
- Test: `tests/contracts/test_update_schemas.py` (CREATE)

### Step 1: Write failing test for update TypedDicts

```python
# tests/contracts/test_update_schemas.py
"""Tests for update schema TypedDicts."""

from datetime import datetime

import pytest


class TestUpdateSchemas:
    """Verify update TypedDicts are importable and usable."""

    def test_export_status_update_importable(self) -> None:
        """ExportStatusUpdate should be importable from contracts."""
        from elspeth.contracts import ExportStatusUpdate

        # Should accept all valid fields
        update: ExportStatusUpdate = {
            "export_status": "completed",
            "exported_at": datetime.now(),
        }
        assert "export_status" in update

    def test_batch_status_update_importable(self) -> None:
        """BatchStatusUpdate should be importable from contracts."""
        from elspeth.contracts import BatchStatusUpdate

        update: BatchStatusUpdate = {
            "status": "executing",
            "trigger_reason": "count_reached",
        }
        assert "status" in update

    def test_export_status_update_partial(self) -> None:
        """ExportStatusUpdate should allow partial updates."""
        from elspeth.contracts import ExportStatusUpdate

        # Only status field
        update: ExportStatusUpdate = {"export_status": "pending"}
        assert len(update) == 1

    def test_batch_status_update_with_state_id(self) -> None:
        """BatchStatusUpdate should accept state_id for aggregation linking."""
        from elspeth.contracts import BatchStatusUpdate

        update: BatchStatusUpdate = {
            "status": "completed",
            "completed_at": datetime.now(),
            "state_id": "state-123",
        }
        assert "state_id" in update
```

### Step 2: Run test to verify it fails

Run: `pytest tests/contracts/test_update_schemas.py -v`
Expected: FAIL - imports will fail

### Step 3: Add TypedDicts to contracts/audit.py

Add to `src/elspeth/contracts/audit.py`:

```python
from typing import TypedDict
from datetime import datetime

class ExportStatusUpdate(TypedDict, total=False):
    """Schema for export status updates in recorder."""

    export_status: str
    exported_at: datetime
    export_error: str
    export_format: str
    export_sink: str


class BatchStatusUpdate(TypedDict, total=False):
    """Schema for batch status updates in recorder."""

    status: str
    completed_at: datetime
    trigger_reason: str
    state_id: str
```

### Step 4: Export from contracts/__init__.py

Add to `src/elspeth/contracts/__init__.py`:

```python
from elspeth.contracts.audit import (
    # ... existing ...
    BatchStatusUpdate,
    ExportStatusUpdate,
)

__all__ = [
    # ... existing ...
    "BatchStatusUpdate",
    "ExportStatusUpdate",
]
```

### Step 5: Run test to verify it passes

Run: `pytest tests/contracts/test_update_schemas.py -v`
Expected: PASS

### Step 6: Commit

```bash
git add src/elspeth/contracts/audit.py src/elspeth/contracts/__init__.py tests/contracts/test_update_schemas.py
git commit -m "$(cat <<'EOF'
feat(contracts): add TypedDicts for update schemas

Add ExportStatusUpdate and BatchStatusUpdate TypedDicts to formalize
the dict shapes passed to recorder update methods.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Integrate Schema Validator in Orchestrator

**Files:**
- Modify: `src/elspeth/engine/orchestrator.py`
- Test: `tests/engine/test_orchestrator_schema_validation.py` (CREATE)

### Step 1: Write failing test for schema validation integration

```python
# tests/engine/test_orchestrator_schema_validation.py
"""Tests for schema validation in orchestrator."""

from unittest.mock import MagicMock, patch

import pytest

from elspeth.contracts import PluginSchema
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape import LandscapeDB
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig


class InputSchema(PluginSchema):
    """Test input schema."""

    name: str
    value: int


class OutputSchema(PluginSchema):
    """Test output schema."""

    name: str
    result: float


class IncompatibleSchema(PluginSchema):
    """Schema missing required field."""

    missing_field: str


class TestOrchestratorSchemaValidation:
    """Test that orchestrator calls schema validator."""

    def test_schema_validation_called_on_run(self) -> None:
        """Orchestrator should call validate_pipeline_schemas."""
        with patch(
            "elspeth.engine.orchestrator.validate_pipeline_schemas"
        ) as mock_validate:
            mock_validate.return_value = []  # No errors

            # Create minimal mocks
            db = MagicMock(spec=LandscapeDB)
            orchestrator = Orchestrator(db)

            # Create mock config with schemas
            source = MagicMock()
            source.name = "test_source"
            source.output_schema = OutputSchema
            source.node_id = None

            config = PipelineConfig(
                source=source,
                transforms=[],
                sinks={},
            )

            graph = MagicMock(spec=ExecutionGraph)
            graph.topological_order.return_value = ["source"]
            graph.get_source.return_value = "source"
            graph.get_transform_id_map.return_value = {}
            graph.get_sink_id_map.return_value = {}
            graph.get_edges.return_value = []
            graph.get_route_resolution_map.return_value = {}
            graph.get_output_sink.return_value = "output"
            graph.get_node_info.return_value = MagicMock(
                plugin_name="test", node_type="source", config={}
            )

            # This will fail for other reasons, but we want to verify
            # validate_pipeline_schemas was called
            try:
                orchestrator.run(config, graph=graph)
            except Exception:
                pass  # Expected to fail - we're testing the call

            # Verify schema validation was called
            mock_validate.assert_called_once()

    def test_schema_validation_errors_raise(self) -> None:
        """Schema validation errors should raise before processing."""
        with patch(
            "elspeth.engine.orchestrator.validate_pipeline_schemas"
        ) as mock_validate:
            mock_validate.return_value = [
                "Source output missing fields required by transform[0]: {'missing_field'}"
            ]

            db = MagicMock(spec=LandscapeDB)
            orchestrator = Orchestrator(db)

            source = MagicMock()
            source.name = "test_source"
            source.output_schema = OutputSchema

            config = PipelineConfig(
                source=source,
                transforms=[],
                sinks={},
            )

            graph = MagicMock(spec=ExecutionGraph)

            with pytest.raises(ValueError, match="schema incompatibility"):
                orchestrator.run(config, graph=graph)
```

### Step 2: Run test to verify it fails

Run: `pytest tests/engine/test_orchestrator_schema_validation.py -v`
Expected: FAIL - validate_pipeline_schemas not called

### Step 3: Add schema validation to orchestrator

Add import and validation call in `src/elspeth/engine/orchestrator.py`:

```python
# Add import near top:
from elspeth.engine.schema_validator import validate_pipeline_schemas

# In run() method, after graph validation (around line 237), add:
        # Validate schema compatibility (opt-in, skip if schemas not defined)
        source_output = getattr(config.source, "output_schema", None)
        transform_inputs = [
            getattr(t, "input_schema", None) for t in config.transforms
        ]
        transform_outputs = [
            getattr(t, "output_schema", None) for t in config.transforms
        ]
        sink_inputs = [
            getattr(s, "input_schema", None) for s in config.sinks.values()
        ]

        schema_errors = validate_pipeline_schemas(
            source_output=source_output,
            transform_inputs=transform_inputs,
            transform_outputs=transform_outputs,
            sink_inputs=sink_inputs,
        )
        if schema_errors:
            raise ValueError(
                f"Pipeline schema incompatibility: {'; '.join(schema_errors)}"
            )
```

### Step 4: Run test to verify it passes

Run: `pytest tests/engine/test_orchestrator_schema_validation.py -v`
Expected: PASS

### Step 5: Run full test suite

Run: `pytest tests/ -v --tb=short`
Expected: All tests pass

### Step 6: Commit

```bash
git add src/elspeth/engine/orchestrator.py tests/engine/test_orchestrator_schema_validation.py
git commit -m "$(cat <<'EOF'
feat(engine): integrate schema validator in orchestrator

Call validate_pipeline_schemas() after graph validation but before
processing. Schema errors now raise ValueError with details before
any rows are processed.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Fix Stringly-Typed Routing Decisions

**Files:**
- Modify: `src/elspeth/engine/executors.py`
- Modify: `src/elspeth/engine/processor.py`
- Test: `tests/engine/test_routing_enums.py` (CREATE)

### Step 1: Write failing test for enum-based routing

```python
# tests/engine/test_routing_enums.py
"""Tests for enum-based routing comparisons."""

import pytest

from elspeth.contracts.enums import RoutingKind


class TestRoutingKindUsage:
    """Verify RoutingKind enum is used for comparisons."""

    def test_routing_kind_continue(self) -> None:
        """CONTINUE should be comparable to action.kind."""
        kind = RoutingKind.CONTINUE
        assert kind == RoutingKind.CONTINUE
        assert kind.value == "continue"

    def test_routing_kind_route(self) -> None:
        """ROUTE should be comparable to action.kind."""
        kind = RoutingKind.ROUTE
        assert kind == RoutingKind.ROUTE
        assert kind.value == "route"

    def test_routing_kind_fork(self) -> None:
        """FORK_TO_PATHS should be comparable to action.kind."""
        kind = RoutingKind.FORK_TO_PATHS
        assert kind == RoutingKind.FORK_TO_PATHS
        assert kind.value == "fork_to_paths"

    def test_routing_action_uses_enum(self) -> None:
        """RoutingAction.kind should return RoutingKind enum."""
        from elspeth.contracts import RoutingAction

        action = RoutingAction.continue_()  # Note: method is continue_() not continue_processing()
        assert isinstance(action.kind, RoutingKind)
        assert action.kind is RoutingKind.CONTINUE  # Use 'is' for enum identity

        action = RoutingAction.route("sink1")
        assert isinstance(action.kind, RoutingKind)
        assert action.kind is RoutingKind.ROUTE
```

### Step 2: Run test to verify behavior

Run: `pytest tests/engine/test_routing_enums.py -v`
Expected: May pass if RoutingAction already returns enum, otherwise FAIL

### Step 3: Update executors.py to use RoutingKind enum

In `src/elspeth/engine/executors.py`, update:

```python
# Add import:
from elspeth.contracts.enums import RoutingKind

# Line 334: Change from:
if action.kind == "continue":
# To:
if action.kind == RoutingKind.CONTINUE:

# Line 338: Change from:
elif action.kind == "route":
# To:
elif action.kind == RoutingKind.ROUTE:

# Line 360: Change from:
elif action.kind == "fork_to_paths":
# To:
elif action.kind == RoutingKind.FORK_TO_PATHS:
```

### Step 4: Update processor.py to use RoutingKind enum

In `src/elspeth/engine/processor.py`, update:

```python
# Add import:
from elspeth.contracts.enums import RoutingKind

# Line 136: Change from:
elif outcome.result.action.kind == "fork_to_paths":
# To:
elif outcome.result.action.kind == RoutingKind.FORK_TO_PATHS:
```

### Step 5: Run tests

Run: `pytest tests/engine/test_routing_enums.py tests/engine/ -v --tb=short`
Expected: PASS

### Step 6: Run full test suite

Run: `pytest tests/ -v --tb=short`
Expected: All tests pass

### Step 7: Commit

```bash
git add src/elspeth/engine/executors.py src/elspeth/engine/processor.py tests/engine/test_routing_enums.py
git commit -m "$(cat <<'EOF'
refactor(engine): use RoutingKind enum instead of string literals

Replace string comparisons like `action.kind == "continue"` with
enum comparisons like `action.kind == RoutingKind.CONTINUE`.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Make node_id Assignment Explicit

**Files:**
- Modify: `src/elspeth/engine/orchestrator.py`
- Test: `tests/engine/test_node_id_assignment.py` (CREATE)

### Step 1: Write failing test for explicit node_id validation

```python
# tests/engine/test_node_id_assignment.py
"""Tests for explicit node_id assignment with validation."""

from unittest.mock import MagicMock

import pytest

from elspeth.engine.orchestrator import Orchestrator


class TestNodeIdAssignment:
    """Test node_id assignment validation."""

    def test_assign_plugin_node_ids_validates_source(self) -> None:
        """Should raise if source lacks node_id attribute."""
        db = MagicMock()
        orchestrator = Orchestrator(db)

        # Source without node_id attribute
        source = object()  # Plain object, no node_id

        with pytest.raises(AttributeError):
            orchestrator._assign_plugin_node_ids(
                source=source,
                transforms=[],
                sinks={},
                source_id="source-1",
                transform_id_map={},
                sink_id_map={},
            )

    def test_assign_plugin_node_ids_sets_source_id(self) -> None:
        """Should set node_id on source plugin."""
        db = MagicMock()
        orchestrator = Orchestrator(db)

        source = MagicMock()
        source.node_id = None

        orchestrator._assign_plugin_node_ids(
            source=source,
            transforms=[],
            sinks={},
            source_id="source-1",
            transform_id_map={},
            sink_id_map={},
        )

        assert source.node_id == "source-1"

    def test_assign_plugin_node_ids_sets_transform_ids(self) -> None:
        """Should set node_id on all transforms."""
        db = MagicMock()
        orchestrator = Orchestrator(db)

        source = MagicMock()
        source.node_id = None

        t1 = MagicMock()
        t1.node_id = None
        t2 = MagicMock()
        t2.node_id = None

        orchestrator._assign_plugin_node_ids(
            source=source,
            transforms=[t1, t2],
            sinks={},
            source_id="source-1",
            transform_id_map={0: "transform-0", 1: "transform-1"},
            sink_id_map={},
        )

        assert t1.node_id == "transform-0"
        assert t2.node_id == "transform-1"

    def test_assign_plugin_node_ids_sets_sink_ids(self) -> None:
        """Should set node_id on all sinks."""
        db = MagicMock()
        orchestrator = Orchestrator(db)

        source = MagicMock()
        source.node_id = None

        sink1 = MagicMock()
        sink1.node_id = None
        sink2 = MagicMock()
        sink2.node_id = None

        orchestrator._assign_plugin_node_ids(
            source=source,
            transforms=[],
            sinks={"output": sink1, "errors": sink2},
            source_id="source-1",
            transform_id_map={},
            sink_id_map={"output": "sink-output", "errors": "sink-errors"},
        )

        assert sink1.node_id == "sink-output"
        assert sink2.node_id == "sink-errors"
```

### Step 2: Run test to verify it fails

Run: `pytest tests/engine/test_node_id_assignment.py -v`
Expected: FAIL - method doesn't exist

### Step 3: Extract node_id assignment to explicit method

Add method to `src/elspeth/engine/orchestrator.py`:

```python
    def _assign_plugin_node_ids(
        self,
        source: SourceProtocol,
        transforms: list[TransformLike],
        sinks: dict[str, SinkProtocol],
        source_id: str,
        transform_id_map: dict[int, str],
        sink_id_map: dict[str, str],
    ) -> None:
        """Explicitly assign node_id to all plugins with validation.

        This is part of the plugin protocol contract - all plugins define
        node_id: str | None and the orchestrator populates it.

        Args:
            source: Source plugin instance
            transforms: List of transform plugins
            sinks: Dict of sink_name -> sink plugin
            source_id: Node ID for source
            transform_id_map: Maps transform sequence -> node_id
            sink_id_map: Maps sink_name -> node_id

        Raises:
            ValueError: If transform/sink not in ID map
        """
        # Set node_id on source
        source.node_id = source_id

        # Set node_id on transforms
        for seq, transform in enumerate(transforms):
            if seq not in transform_id_map:
                raise ValueError(
                    f"Transform at sequence {seq} not found in graph. "
                    f"Graph has mappings for sequences: {list(transform_id_map.keys())}"
                )
            transform.node_id = transform_id_map[seq]

        # Set node_id on sinks
        for sink_name, sink in sinks.items():
            if sink_name not in sink_id_map:
                raise ValueError(
                    f"Sink '{sink_name}' not found in graph. "
                    f"Available sinks: {list(sink_id_map.keys())}"
                )
            sink.node_id = sink_id_map[sink_name]
```

### Step 4: Update _execute_run to call the new method

In `_execute_run()`, replace lines 405-425 with:

```python
        # Assign node_ids to all plugins
        self._assign_plugin_node_ids(
            source=config.source,
            transforms=config.transforms,
            sinks=config.sinks,
            source_id=source_id,
            transform_id_map=transform_id_map,
            sink_id_map=sink_id_map,
        )
```

### Step 5: Run tests

Run: `pytest tests/engine/test_node_id_assignment.py -v`
Expected: PASS

### Step 6: Run full test suite

Run: `pytest tests/ -v --tb=short`
Expected: All tests pass

### Step 7: Commit

```bash
git add src/elspeth/engine/orchestrator.py tests/engine/test_node_id_assignment.py
git commit -m "$(cat <<'EOF'
refactor(orchestrator): extract explicit node_id assignment method

Move node_id assignment to _assign_plugin_node_ids() for clarity
and explicit validation. No functional change.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Create RetryPolicy TypedDict

**Files:**
- Modify: `src/elspeth/contracts/engine.py` (CREATE)
- Modify: `src/elspeth/engine/retry.py`
- Test: `tests/engine/test_retry_policy.py` (CREATE)

### Step 1: Write failing test for RetryPolicy TypedDict

```python
# tests/engine/test_retry_policy.py
"""Tests for RetryPolicy TypedDict."""

import pytest


class TestRetryPolicy:
    """Verify RetryPolicy TypedDict works correctly."""

    def test_retry_policy_importable(self) -> None:
        """RetryPolicy should be importable from contracts."""
        from elspeth.contracts import RetryPolicy

        policy: RetryPolicy = {
            "max_attempts": 3,
            "base_delay": 1.0,
        }
        assert policy["max_attempts"] == 3

    def test_retry_config_from_policy_with_typed_dict(self) -> None:
        """RetryConfig.from_policy should accept RetryPolicy."""
        from elspeth.contracts import RetryPolicy
        from elspeth.engine.retry import RetryConfig

        policy: RetryPolicy = {
            "max_attempts": 5,
            "base_delay": 2.0,
            "max_delay": 120.0,
            "jitter": 0.5,
        }

        config = RetryConfig.from_policy(policy)
        assert config.max_attempts == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 120.0
        assert config.jitter == 0.5

    def test_retry_policy_partial(self) -> None:
        """RetryPolicy should allow partial specification."""
        from elspeth.contracts import RetryPolicy
        from elspeth.engine.retry import RetryConfig

        # Only specify some fields
        policy: RetryPolicy = {"max_attempts": 10}
        config = RetryConfig.from_policy(policy)
        assert config.max_attempts == 10
        # Defaults for unspecified fields
        assert config.base_delay == 1.0
```

### Step 2: Run test to verify it fails

Run: `pytest tests/engine/test_retry_policy.py -v`
Expected: FAIL - RetryPolicy not defined

### Step 3: Create contracts/engine.py with RetryPolicy

Create `src/elspeth/contracts/engine.py`:

```python
# src/elspeth/contracts/engine.py
"""Engine-related type contracts."""

from typing import TypedDict


class RetryPolicy(TypedDict, total=False):
    """Schema for retry configuration from plugin policies.

    All fields are optional - from_policy() applies defaults.
    """

    max_attempts: int
    base_delay: float
    max_delay: float
    jitter: float
    retry_on: list[str]
```

### Step 4: Export from contracts/__init__.py

Add to `src/elspeth/contracts/__init__.py`:

```python
from elspeth.contracts.engine import RetryPolicy

__all__ = [
    # ... existing ...
    "RetryPolicy",
]
```

### Step 5: Update retry.py type hint (optional - for documentation)

In `src/elspeth/engine/retry.py`, update `from_policy` signature:

```python
from elspeth.contracts import RetryPolicy

@classmethod
def from_policy(cls, policy: RetryPolicy | None) -> "RetryConfig":
```

### Step 6: Run tests

Run: `pytest tests/engine/test_retry_policy.py -v`
Expected: PASS

### Step 7: Commit

```bash
git add src/elspeth/contracts/engine.py src/elspeth/contracts/__init__.py src/elspeth/engine/retry.py tests/engine/test_retry_policy.py
git commit -m "$(cat <<'EOF'
feat(contracts): add RetryPolicy TypedDict

Add typed schema for retry configuration used by RetryConfig.from_policy().

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Create ExecutionResult TypedDict

**Files:**
- Modify: `src/elspeth/contracts/cli.py` (CREATE)
- Modify: `src/elspeth/cli.py`
- Test: `tests/cli/test_execution_result.py` (CREATE)

### NOTE: TypedDict is Intentionally Broader Than Current Usage

The current `_execute_pipeline()` returns only `run_id`, `status`, `rows_processed`.
The TypedDict includes additional fields (`rows_succeeded`, `rows_failed`, `duration_seconds`)
that will be populated as the CLI matures. Using `total=False` allows partial returns now
while documenting the full contract for future expansion.

### Step 1: Write failing test for ExecutionResult

```python
# tests/cli/test_execution_result.py
"""Tests for ExecutionResult TypedDict."""

import pytest


class TestExecutionResult:
    """Verify ExecutionResult TypedDict works correctly."""

    def test_execution_result_importable(self) -> None:
        """ExecutionResult should be importable from contracts."""
        from elspeth.contracts import ExecutionResult

        result: ExecutionResult = {
            "run_id": "run-123",
            "status": "completed",
            "rows_processed": 100,
        }
        assert result["run_id"] == "run-123"

    def test_execution_result_full(self) -> None:
        """ExecutionResult should accept all fields."""
        from elspeth.contracts import ExecutionResult

        result: ExecutionResult = {
            "run_id": "run-456",
            "status": "completed",
            "rows_processed": 1000,
            "rows_succeeded": 990,
            "rows_failed": 10,
            "duration_seconds": 45.5,
        }
        assert result["rows_succeeded"] == 990
```

### Step 2: Run test to verify it fails

Run: `pytest tests/cli/test_execution_result.py -v`
Expected: FAIL - ExecutionResult not defined

### Step 3: Create contracts/cli.py with ExecutionResult

Create `src/elspeth/contracts/cli.py`:

```python
# src/elspeth/contracts/cli.py
"""CLI-related type contracts."""

from typing import TypedDict


class ExecutionResult(TypedDict, total=False):
    """Result from pipeline execution.

    Returned by _execute_pipeline() in cli.py.
    """

    run_id: str
    status: str
    rows_processed: int
    rows_succeeded: int
    rows_failed: int
    duration_seconds: float
```

### Step 4: Export from contracts/__init__.py

Add to `src/elspeth/contracts/__init__.py`:

```python
from elspeth.contracts.cli import ExecutionResult

__all__ = [
    # ... existing ...
    "ExecutionResult",
]
```

### Step 5: Update cli.py return type

In `src/elspeth/cli.py`, update `_execute_pipeline`:

```python
from elspeth.contracts import ExecutionResult

def _execute_pipeline(config: ElspethSettings, verbose: bool = False) -> ExecutionResult:
    """Execute a pipeline from configuration.

    Args:
        config: Validated ElspethSettings instance.
        verbose: Show detailed output.

    Returns:
        ExecutionResult with run_id, status, rows_processed.
    """
```

### Step 6: Run tests

Run: `pytest tests/cli/test_execution_result.py -v`
Expected: PASS

### Step 7: Run full test suite

Run: `pytest tests/ -v --tb=short`
Expected: All tests pass

### Step 8: Commit

```bash
git add src/elspeth/contracts/cli.py src/elspeth/contracts/__init__.py src/elspeth/cli.py tests/cli/test_execution_result.py
git commit -m "$(cat <<'EOF'
feat(contracts): add ExecutionResult TypedDict

Add typed schema for pipeline execution results from _execute_pipeline().

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Final Verification

### Run full test suite

```bash
pytest tests/ -v
```

### Run type checker

```bash
mypy src/elspeth/ --ignore-missing-imports
```

### Run linter

```bash
ruff check src/elspeth/
```

---

## Summary

| Task | Description | Files Modified | Tests Added |
|------|-------------|----------------|-------------|
| 1 | Consolidate PluginSchema to contracts | contracts/data.py, plugins/schemas.py | test_plugin_schema.py |
| 2 | Convert models to use enum types | models.py | test_models_enums.py |
| 3 | Create update schema TypedDicts | contracts/audit.py | test_update_schemas.py |
| 4 | Integrate schema validator | orchestrator.py | test_orchestrator_schema_validation.py |
| 5 | Fix stringly-typed routing | executors.py, processor.py | test_routing_enums.py |
| 6 | Extract node_id assignment | orchestrator.py | test_node_id_assignment.py |
| 7 | Create RetryPolicy TypedDict | contracts/engine.py, retry.py | test_retry_policy.py |
| 8 | Create ExecutionResult TypedDict | contracts/cli.py, cli.py | test_execution_result.py |
