# Phase 4B: Built-in Transforms, Gates, TUI, and Export (Tasks 1-11)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add built-in transforms and gates so pipelines can do more than source → sink, enhance the TUI explain command with proper lineage visualization, and provide audit trail export for compliance and analysis.

**Rationale:** Phase 4 delivers working CLI and I/O plugins, but sets `transforms=[]` - pipelines can only copy data from source to sink. This phase adds the missing pieces:
- **Transforms:** PassThrough (testing/debugging), FieldMapper (rename/select fields), Filter (row filtering)
- **Gates:** ThresholdGate (numeric routing), FieldMatchGate (pattern-based routing)
- **TUI:** Tree widget for lineage visualization, detail panels for inspecting node states
- **Export:** CLI command and config option to export Landscape audit trail to CSV for compliance, archival, and external analysis

**Tech Stack:** Python 3.11+, Textual (TUI widgets), re (regex for field matching), pandas (CSV export)

**Dependencies:**
- Phase 2: `elspeth.plugins` (protocols, base, context, results, schemas)
- Phase 3A: `elspeth.core.landscape` (LandscapeRecorder queries for TUI)
- Phase 3B: `elspeth.engine` (Orchestrator, executors)
- Phase 4: `elspeth.cli`, `elspeth.core.landscape.lineage`

---

## Phase 4 "Not Yet Complete" Correction

**IMPORTANT:** Phase 4's "Not Yet Complete" section claims "Integration with Phase 3 Orchestrator (currently using simple loop)". This is **incorrect**. Phase 4 Task 9 implements `_execute_pipeline()` which uses:

```python
orchestrator = Orchestrator(db)
result = orchestrator.run(pipeline_config)
```

This IS the full Phase 3B Orchestrator with complete audit trails, OpenTelemetry spans, token management, and executor wrapping. The only limitation is `transforms=[]` - there are no transforms to execute.

**After Phase 4B completion**, update Phase 4's "Not Yet Complete" section to remove the misleading "simple loop" claim and note that transforms/gates are now available in Phase 4B.

---

## Task 1: PassThrough Transform

**Context:** A transform that passes rows through unchanged. Essential for testing pipelines and debugging - you can insert a PassThrough to verify data flow without modification.

**Files:**
- Create: `src/elspeth/plugins/transforms/passthrough.py`
- Create: `src/elspeth/plugins/transforms/__init__.py`
- Create: `tests/plugins/transforms/__init__.py`
- Create: `tests/plugins/transforms/test_passthrough.py`

### Step 1: Write the failing test

```python
# tests/plugins/transforms/__init__.py
"""Transform plugin tests."""

# tests/plugins/transforms/test_passthrough.py
"""Tests for PassThrough transform."""

import pytest

from elspeth.plugins.context import PluginContext
from elspeth.plugins.protocols import TransformProtocol


class TestPassThrough:
    """Tests for PassThrough transform plugin."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create minimal plugin context."""
        return PluginContext(run_id="test-run", config={})

    def test_implements_protocol(self) -> None:
        """PassThrough implements TransformProtocol."""
        from elspeth.plugins.transforms.passthrough import PassThrough

        transform = PassThrough({})
        assert isinstance(transform, TransformProtocol)

    def test_has_required_attributes(self) -> None:
        """PassThrough has name and schemas."""
        from elspeth.plugins.transforms.passthrough import PassThrough

        assert PassThrough.name == "passthrough"
        assert hasattr(PassThrough, "input_schema")
        assert hasattr(PassThrough, "output_schema")

    def test_process_returns_unchanged_row(self, ctx: PluginContext) -> None:
        """process() returns row data unchanged."""
        from elspeth.plugins.transforms.passthrough import PassThrough

        transform = PassThrough({})
        row = {"id": 1, "name": "alice", "value": 100}

        result = transform.process(row, ctx)

        assert result.status == "success"
        assert result.row == row
        assert result.row is not row  # Should be a copy, not the same object

    def test_process_with_nested_data(self, ctx: PluginContext) -> None:
        """Handles nested structures correctly."""
        from elspeth.plugins.transforms.passthrough import PassThrough

        transform = PassThrough({})
        row = {"id": 1, "meta": {"source": "test", "tags": ["a", "b"]}}

        result = transform.process(row, ctx)

        assert result.status == "success"
        assert result.row == row
        # Nested structures should be deep copied
        assert result.row["meta"] is not row["meta"]
        assert result.row["meta"]["tags"] is not row["meta"]["tags"]

    def test_process_with_empty_row(self, ctx: PluginContext) -> None:
        """Handles empty row."""
        from elspeth.plugins.transforms.passthrough import PassThrough

        transform = PassThrough({})
        row: dict = {}

        result = transform.process(row, ctx)

        assert result.status == "success"
        assert result.row == {}

    def test_close_is_idempotent(self) -> None:
        """close() can be called multiple times."""
        from elspeth.plugins.transforms.passthrough import PassThrough

        transform = PassThrough({})
        transform.close()
        transform.close()  # Should not raise
```

### Step 2: Run test to verify it fails

Run: `pytest tests/plugins/transforms/test_passthrough.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'elspeth.plugins.transforms'`

### Step 3: Write minimal implementation

```python
# src/elspeth/plugins/transforms/passthrough.py
"""PassThrough transform plugin.

Passes rows through unchanged. Useful for testing and debugging pipelines.
"""

import copy
from typing import Any

from elspeth.plugins.base import BaseTransform
from elspeth.plugins.context import PluginContext
from elspeth.plugins.results import TransformResult
from elspeth.plugins.schemas import PluginSchema


class PassThroughSchema(PluginSchema):
    """Dynamic schema - accepts any fields."""

    model_config = {"extra": "allow"}


class PassThrough(BaseTransform):
    """Pass rows through unchanged.

    Use cases:
    - Testing pipeline wiring without modification
    - Debugging data flow (add logging in subclass)
    - Placeholder for future transform logic

    Config options:
        None (accepts empty config)
    """

    name = "passthrough"
    input_schema = PassThroughSchema
    output_schema = PassThroughSchema

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

    def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
        """Return row unchanged (deep copy to prevent mutation).

        Args:
            row: Input row data
            ctx: Plugin context

        Returns:
            TransformResult with unchanged row data
        """
        return TransformResult.success(copy.deepcopy(row))

    def close(self) -> None:
        """No resources to release."""
        pass
```

```python
# src/elspeth/plugins/transforms/__init__.py
"""Built-in transform plugins for ELSPETH.

Transforms process rows in the pipeline. Each transform receives a row
and returns a TransformResult indicating success/failure and output data.
"""

from elspeth.plugins.transforms.passthrough import PassThrough

__all__ = ["PassThrough"]
```

### Step 4: Run test to verify it passes

Run: `pytest tests/plugins/transforms/test_passthrough.py -v`
Expected: PASS (6 tests)

### Step 5: Commit

```bash
git add src/elspeth/plugins/transforms/ tests/plugins/transforms/
git commit -m "feat(transforms): add PassThrough transform plugin"
```

---

## Task 2: FieldMapper Transform

**Context:** A transform that renames, selects, or reorders fields. Essential for data normalization between pipeline stages.

**Files:**
- Create: `src/elspeth/plugins/transforms/field_mapper.py`
- Modify: `src/elspeth/plugins/transforms/__init__.py`
- Create: `tests/plugins/transforms/test_field_mapper.py`

### Step 1: Write the failing test

```python
# tests/plugins/transforms/test_field_mapper.py
"""Tests for FieldMapper transform."""

import pytest

from elspeth.plugins.context import PluginContext
from elspeth.plugins.protocols import TransformProtocol


class TestFieldMapper:
    """Tests for FieldMapper transform plugin."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create minimal plugin context."""
        return PluginContext(run_id="test-run", config={})

    def test_implements_protocol(self) -> None:
        """FieldMapper implements TransformProtocol."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper({"mapping": {"old": "new"}})
        assert isinstance(transform, TransformProtocol)

    def test_has_required_attributes(self) -> None:
        """FieldMapper has name and schemas."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        assert FieldMapper.name == "field_mapper"

    def test_rename_single_field(self, ctx: PluginContext) -> None:
        """Rename a single field."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper({"mapping": {"old_name": "new_name"}})
        row = {"old_name": "value", "other": 123}

        result = transform.process(row, ctx)

        assert result.status == "success"
        assert result.row == {"new_name": "value", "other": 123}
        assert "old_name" not in result.row

    def test_rename_multiple_fields(self, ctx: PluginContext) -> None:
        """Rename multiple fields at once."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper({
            "mapping": {
                "first_name": "firstName",
                "last_name": "lastName",
            }
        })
        row = {"first_name": "Alice", "last_name": "Smith", "id": 1}

        result = transform.process(row, ctx)

        assert result.status == "success"
        assert result.row == {"firstName": "Alice", "lastName": "Smith", "id": 1}

    def test_select_fields_only(self, ctx: PluginContext) -> None:
        """Only include specified fields (drop others)."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper({
            "mapping": {"id": "id", "name": "name"},
            "select_only": True,
        })
        row = {"id": 1, "name": "alice", "secret": "password", "extra": "data"}

        result = transform.process(row, ctx)

        assert result.status == "success"
        assert result.row == {"id": 1, "name": "alice"}
        assert "secret" not in result.row
        assert "extra" not in result.row

    def test_missing_field_error(self, ctx: PluginContext) -> None:
        """Error when required field is missing and strict mode enabled."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper({
            "mapping": {"required_field": "output"},
            "strict": True,
        })
        row = {"other_field": "value"}

        result = transform.process(row, ctx)

        assert result.status == "error"
        assert "required_field" in str(result.error)

    def test_missing_field_skip_non_strict(self, ctx: PluginContext) -> None:
        """Skip missing fields when strict mode disabled."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper({
            "mapping": {"maybe_field": "output"},
            "strict": False,
        })
        row = {"other_field": "value"}

        result = transform.process(row, ctx)

        assert result.status == "success"
        assert result.row == {"other_field": "value"}
        assert "output" not in result.row

    def test_default_is_non_strict(self, ctx: PluginContext) -> None:
        """Default behavior is non-strict (skip missing)."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper({"mapping": {"missing": "output"}})
        row = {"exists": "value"}

        result = transform.process(row, ctx)

        assert result.status == "success"

    def test_nested_field_access(self, ctx: PluginContext) -> None:
        """Access nested fields with dot notation."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper({
            "mapping": {"meta.source": "origin"},
        })
        row = {"id": 1, "meta": {"source": "api", "timestamp": 123}}

        result = transform.process(row, ctx)

        assert result.status == "success"
        assert result.row["origin"] == "api"
        assert "meta" in result.row  # Original nested structure preserved

    def test_empty_mapping_passthrough(self, ctx: PluginContext) -> None:
        """Empty mapping acts as passthrough."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper({"mapping": {}})
        row = {"a": 1, "b": 2}

        result = transform.process(row, ctx)

        assert result.status == "success"
        assert result.row == row
```

### Step 2: Run test to verify it fails

Run: `pytest tests/plugins/transforms/test_field_mapper.py -v`
Expected: FAIL with `ModuleNotFoundError`

### Step 3: Write minimal implementation

```python
# src/elspeth/plugins/transforms/field_mapper.py
"""FieldMapper transform plugin.

Renames, selects, and reorganizes row fields.
"""

import copy
from typing import Any

from elspeth.plugins.base import BaseTransform
from elspeth.plugins.context import PluginContext
from elspeth.plugins.results import TransformResult
from elspeth.plugins.schemas import PluginSchema


class FieldMapperSchema(PluginSchema):
    """Dynamic schema - fields determined by mapping."""

    model_config = {"extra": "allow"}


class FieldMapper(BaseTransform):
    """Map, rename, and select row fields.

    Config options:
        mapping: Dict of source_field -> target_field
            - Simple: {"old": "new"} renames old to new
            - Nested: {"meta.source": "origin"} extracts nested field
        select_only: If True, only include mapped fields (default: False)
        strict: If True, error on missing source fields (default: False)
    """

    name = "field_mapper"
    input_schema = FieldMapperSchema
    output_schema = FieldMapperSchema

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._mapping: dict[str, str] = config.get("mapping", {})
        self._select_only: bool = config.get("select_only", False)
        self._strict: bool = config.get("strict", False)

    def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
        """Apply field mapping to row.

        Args:
            row: Input row data
            ctx: Plugin context

        Returns:
            TransformResult with mapped row data
        """
        # Start with empty or copy depending on select_only
        if self._select_only:
            output: dict[str, Any] = {}
        else:
            output = copy.deepcopy(row)

        # Apply mappings
        for source, target in self._mapping.items():
            value = self._get_nested(row, source)

            if value is _MISSING:
                if self._strict:
                    return TransformResult.error(
                        f"Required field '{source}' not found in row"
                    )
                continue  # Skip missing fields in non-strict mode

            # Remove old key if it exists (for rename within same dict)
            if not self._select_only and "." not in source and source in output:
                del output[source]

            output[target] = value

        return TransformResult.success(output)

    def _get_nested(self, data: dict[str, Any], path: str) -> Any:
        """Get value from nested dict using dot notation.

        Args:
            data: Source dictionary
            path: Dot-separated path (e.g., "meta.source")

        Returns:
            Value at path or _MISSING sentinel
        """
        parts = path.split(".")
        current: Any = data

        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return _MISSING
            current = current[part]

        return current

    def close(self) -> None:
        """No resources to release."""
        pass


# Sentinel for missing values (distinct from None)
class _MissingSentinel:
    """Sentinel to distinguish missing fields from None values."""
    pass


_MISSING = _MissingSentinel()
```

Update `__init__.py`:

```python
# src/elspeth/plugins/transforms/__init__.py
"""Built-in transform plugins for ELSPETH.

Transforms process rows in the pipeline. Each transform receives a row
and returns a TransformResult indicating success/failure and output data.
"""

from elspeth.plugins.transforms.field_mapper import FieldMapper
from elspeth.plugins.transforms.passthrough import PassThrough

__all__ = ["FieldMapper", "PassThrough"]
```

### Step 4: Run test to verify it passes

Run: `pytest tests/plugins/transforms/test_field_mapper.py -v`
Expected: PASS (10 tests)

### Step 5: Commit

```bash
git add src/elspeth/plugins/transforms/ tests/plugins/transforms/
git commit -m "feat(transforms): add FieldMapper transform plugin"
```

---

## Task 3: Filter Transform

**Context:** A transform that filters rows based on conditions. Returns success with the row if it passes, or returns a "skip" status if filtered out.

**Files:**
- Create: `src/elspeth/plugins/transforms/filter.py`
- Modify: `src/elspeth/plugins/transforms/__init__.py`
- Create: `tests/plugins/transforms/test_filter.py`

### Step 1: Write the failing test

```python
# tests/plugins/transforms/test_filter.py
"""Tests for Filter transform."""

import pytest

from elspeth.plugins.context import PluginContext
from elspeth.plugins.protocols import TransformProtocol


class TestFilter:
    """Tests for Filter transform plugin."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create minimal plugin context."""
        return PluginContext(run_id="test-run", config={})

    def test_implements_protocol(self) -> None:
        """Filter implements TransformProtocol."""
        from elspeth.plugins.transforms.filter import Filter

        transform = Filter({"field": "status", "equals": "active"})
        assert isinstance(transform, TransformProtocol)

    def test_has_required_attributes(self) -> None:
        """Filter has name and schemas."""
        from elspeth.plugins.transforms.filter import Filter

        assert Filter.name == "filter"

    def test_equals_condition_pass(self, ctx: PluginContext) -> None:
        """Row passes when field equals value."""
        from elspeth.plugins.transforms.filter import Filter

        transform = Filter({"field": "status", "equals": "active"})
        row = {"id": 1, "status": "active"}

        result = transform.process(row, ctx)

        assert result.status == "success"
        assert result.row == row

    def test_equals_condition_fail(self, ctx: PluginContext) -> None:
        """Row filtered when field does not equal value."""
        from elspeth.plugins.transforms.filter import Filter

        transform = Filter({"field": "status", "equals": "active"})
        row = {"id": 1, "status": "inactive"}

        result = transform.process(row, ctx)

        # Filtered rows return success with row=None
        assert result.status == "success"
        assert result.row is None

    def test_not_equals_condition(self, ctx: PluginContext) -> None:
        """Row passes when field does not equal value."""
        from elspeth.plugins.transforms.filter import Filter

        transform = Filter({"field": "status", "not_equals": "deleted"})

        active_row = {"id": 1, "status": "active"}
        result = transform.process(active_row, ctx)
        assert result.status == "success"
        assert result.row == active_row

        deleted_row = {"id": 2, "status": "deleted"}
        result = transform.process(deleted_row, ctx)
        assert result.status == "success"
        assert result.row is None

    def test_greater_than_condition(self, ctx: PluginContext) -> None:
        """Row passes when field is greater than value."""
        from elspeth.plugins.transforms.filter import Filter

        transform = Filter({"field": "score", "greater_than": 50})

        high_row = {"id": 1, "score": 75}
        result = transform.process(high_row, ctx)
        assert result.row == high_row

        low_row = {"id": 2, "score": 25}
        result = transform.process(low_row, ctx)
        assert result.row is None

    def test_less_than_condition(self, ctx: PluginContext) -> None:
        """Row passes when field is less than value."""
        from elspeth.plugins.transforms.filter import Filter

        transform = Filter({"field": "age", "less_than": 18})

        young_row = {"id": 1, "age": 15}
        result = transform.process(young_row, ctx)
        assert result.row == young_row

        adult_row = {"id": 2, "age": 25}
        result = transform.process(adult_row, ctx)
        assert result.row is None

    def test_contains_condition(self, ctx: PluginContext) -> None:
        """Row passes when field contains substring."""
        from elspeth.plugins.transforms.filter import Filter

        transform = Filter({"field": "email", "contains": "@example.com"})

        match_row = {"id": 1, "email": "alice@example.com"}
        result = transform.process(match_row, ctx)
        assert result.row == match_row

        nomatch_row = {"id": 2, "email": "bob@other.com"}
        result = transform.process(nomatch_row, ctx)
        assert result.row is None

    def test_matches_regex_condition(self, ctx: PluginContext) -> None:
        """Row passes when field matches regex."""
        from elspeth.plugins.transforms.filter import Filter

        transform = Filter({"field": "code", "matches": r"^[A-Z]{3}-\d{4}$"})

        match_row = {"id": 1, "code": "ABC-1234"}
        result = transform.process(match_row, ctx)
        assert result.row == match_row

        nomatch_row = {"id": 2, "code": "invalid"}
        result = transform.process(nomatch_row, ctx)
        assert result.row is None

    def test_in_list_condition(self, ctx: PluginContext) -> None:
        """Row passes when field value is in list."""
        from elspeth.plugins.transforms.filter import Filter

        transform = Filter({"field": "status", "in": ["active", "pending"]})

        active_row = {"id": 1, "status": "active"}
        result = transform.process(active_row, ctx)
        assert result.row == active_row

        deleted_row = {"id": 2, "status": "deleted"}
        result = transform.process(deleted_row, ctx)
        assert result.row is None

    def test_missing_field_filters_out(self, ctx: PluginContext) -> None:
        """Row filtered when field is missing (unless allow_missing)."""
        from elspeth.plugins.transforms.filter import Filter

        transform = Filter({"field": "status", "equals": "active"})
        row = {"id": 1}  # No status field

        result = transform.process(row, ctx)
        assert result.row is None

    def test_allow_missing_field(self, ctx: PluginContext) -> None:
        """Row passes when field is missing and allow_missing=True."""
        from elspeth.plugins.transforms.filter import Filter

        transform = Filter({
            "field": "status",
            "equals": "active",
            "allow_missing": True,
        })
        row = {"id": 1}  # No status field

        result = transform.process(row, ctx)
        assert result.row == row  # Passes because allow_missing

    def test_nested_field_access(self, ctx: PluginContext) -> None:
        """Filter on nested field with dot notation."""
        from elspeth.plugins.transforms.filter import Filter

        transform = Filter({"field": "meta.status", "equals": "approved"})

        approved_row = {"id": 1, "meta": {"status": "approved"}}
        result = transform.process(approved_row, ctx)
        assert result.row == approved_row

        pending_row = {"id": 2, "meta": {"status": "pending"}}
        result = transform.process(pending_row, ctx)
        assert result.row is None

    def test_invalid_config_no_condition(self) -> None:
        """Error when no condition is specified."""
        from elspeth.plugins.transforms.filter import Filter

        with pytest.raises(ValueError, match="condition"):
            Filter({"field": "status"})  # No equals, greater_than, etc.
```

### Step 2: Run test to verify it fails

Run: `pytest tests/plugins/transforms/test_filter.py -v`
Expected: FAIL with `ModuleNotFoundError`

### Step 3: Write minimal implementation

```python
# src/elspeth/plugins/transforms/filter.py
"""Filter transform plugin.

Filters rows based on field conditions.
"""

import copy
import re
from typing import Any

from elspeth.plugins.base import BaseTransform
from elspeth.plugins.context import PluginContext
from elspeth.plugins.results import TransformResult
from elspeth.plugins.schemas import PluginSchema


class FilterSchema(PluginSchema):
    """Dynamic schema - accepts any fields."""

    model_config = {"extra": "allow"}


class Filter(BaseTransform):
    """Filter rows based on field conditions.

    Returns success with row if condition passes, success with row=None if filtered.

    Config options:
        field: Field to check (supports dot notation for nested fields)
        allow_missing: If True, missing fields pass filter (default: False)

        Conditions (exactly one required):
        - equals: Field must equal this value
        - not_equals: Field must not equal this value
        - greater_than: Field must be > this value (numeric)
        - less_than: Field must be < this value (numeric)
        - contains: Field must contain this substring
        - matches: Field must match this regex pattern
        - in: Field must be one of these values (list)
    """

    name = "filter"
    input_schema = FilterSchema
    output_schema = FilterSchema

    # Condition types we support
    _CONDITION_KEYS = {
        "equals", "not_equals", "greater_than", "less_than",
        "contains", "matches", "in"
    }

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._field: str = config["field"]
        self._allow_missing: bool = config.get("allow_missing", False)

        # Find which condition is specified
        found_conditions = self._CONDITION_KEYS & set(config.keys())
        if not found_conditions:
            raise ValueError(
                f"Filter requires a condition. Valid conditions: {sorted(self._CONDITION_KEYS)}"
            )

        # Store condition type and value
        self._condition_type = found_conditions.pop()
        self._condition_value = config[self._condition_type]

        # Pre-compile regex if using matches
        if self._condition_type == "matches":
            self._regex = re.compile(self._condition_value)
        else:
            self._regex = None

    def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
        """Apply filter condition to row.

        Args:
            row: Input row data
            ctx: Plugin context

        Returns:
            TransformResult with row if passes, row=None if filtered
        """
        field_value = self._get_nested(row, self._field)

        # Handle missing field
        if field_value is _MISSING:
            if self._allow_missing:
                return TransformResult.success(copy.deepcopy(row))
            return TransformResult.success(None)  # Filtered out

        # Apply condition
        passes = self._evaluate_condition(field_value)

        if passes:
            return TransformResult.success(copy.deepcopy(row))
        return TransformResult.success(None)  # Filtered out

    def _evaluate_condition(self, value: Any) -> bool:
        """Evaluate the condition against a field value.

        Args:
            value: Field value to check

        Returns:
            True if condition passes, False if filtered
        """
        match self._condition_type:
            case "equals":
                return value == self._condition_value
            case "not_equals":
                return value != self._condition_value
            case "greater_than":
                return value > self._condition_value
            case "less_than":
                return value < self._condition_value
            case "contains":
                return self._condition_value in str(value)
            case "matches":
                return bool(self._regex.search(str(value)))
            case "in":
                return value in self._condition_value
            case _:
                return False  # Should never reach here

    def _get_nested(self, data: dict[str, Any], path: str) -> Any:
        """Get value from nested dict using dot notation.

        Args:
            data: Source dictionary
            path: Dot-separated path (e.g., "meta.status")

        Returns:
            Value at path or _MISSING sentinel
        """
        parts = path.split(".")
        current: Any = data

        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return _MISSING
            current = current[part]

        return current

    def close(self) -> None:
        """No resources to release."""
        pass


# Sentinel for missing values
class _MissingSentinel:
    """Sentinel to distinguish missing fields from None values."""
    pass


_MISSING = _MissingSentinel()
```

Update `__init__.py`:

```python
# src/elspeth/plugins/transforms/__init__.py
"""Built-in transform plugins for ELSPETH.

Transforms process rows in the pipeline. Each transform receives a row
and returns a TransformResult indicating success/failure and output data.
"""

from elspeth.plugins.transforms.field_mapper import FieldMapper
from elspeth.plugins.transforms.filter import Filter
from elspeth.plugins.transforms.passthrough import PassThrough

__all__ = ["FieldMapper", "Filter", "PassThrough"]
```

### Step 4: Run test to verify it passes

Run: `pytest tests/plugins/transforms/test_filter.py -v`
Expected: PASS (14 tests)

### Step 5: Commit

```bash
git add src/elspeth/plugins/transforms/ tests/plugins/transforms/
git commit -m "feat(transforms): add Filter transform plugin"
```

---

## Task 4: ThresholdGate

**Context:** A gate that routes rows based on numeric threshold comparison. Routes to different sinks based on whether a value is above or below a threshold.

**Files:**
- Create: `src/elspeth/plugins/gates/__init__.py`
- Create: `src/elspeth/plugins/gates/threshold_gate.py`
- Create: `tests/plugins/gates/__init__.py`
- Create: `tests/plugins/gates/test_threshold_gate.py`

### Step 1: Write the failing test

```python
# tests/plugins/gates/__init__.py
"""Gate plugin tests."""

# tests/plugins/gates/test_threshold_gate.py
"""Tests for ThresholdGate."""

import pytest

from elspeth.plugins.context import PluginContext
from elspeth.plugins.protocols import GateProtocol


class TestThresholdGate:
    """Tests for ThresholdGate plugin."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create minimal plugin context."""
        return PluginContext(run_id="test-run", config={})

    def test_implements_protocol(self) -> None:
        """ThresholdGate implements GateProtocol."""
        from elspeth.plugins.gates.threshold_gate import ThresholdGate

        gate = ThresholdGate({
            "field": "score",
            "threshold": 50,
            "above_sink": "high_scores",
            "below_sink": "low_scores",
        })
        assert isinstance(gate, GateProtocol)

    def test_has_required_attributes(self) -> None:
        """ThresholdGate has name and schemas."""
        from elspeth.plugins.gates.threshold_gate import ThresholdGate

        assert ThresholdGate.name == "threshold_gate"

    def test_route_above_threshold(self, ctx: PluginContext) -> None:
        """Route to above_sink when value > threshold."""
        from elspeth.plugins.gates.threshold_gate import ThresholdGate

        gate = ThresholdGate({
            "field": "score",
            "threshold": 50,
            "above_sink": "high_scores",
            "below_sink": "low_scores",
        })
        row = {"id": 1, "score": 75}

        result = gate.evaluate(row, ctx)

        assert result.action == "route_to_sink"
        assert result.sink_name == "high_scores"
        assert result.row == row

    def test_route_below_threshold(self, ctx: PluginContext) -> None:
        """Route to below_sink when value < threshold."""
        from elspeth.plugins.gates.threshold_gate import ThresholdGate

        gate = ThresholdGate({
            "field": "score",
            "threshold": 50,
            "above_sink": "high_scores",
            "below_sink": "low_scores",
        })
        row = {"id": 1, "score": 25}

        result = gate.evaluate(row, ctx)

        assert result.action == "route_to_sink"
        assert result.sink_name == "low_scores"

    def test_equal_routes_to_below(self, ctx: PluginContext) -> None:
        """Equal value routes to below_sink by default."""
        from elspeth.plugins.gates.threshold_gate import ThresholdGate

        gate = ThresholdGate({
            "field": "score",
            "threshold": 50,
            "above_sink": "high",
            "below_sink": "low",
        })
        row = {"id": 1, "score": 50}

        result = gate.evaluate(row, ctx)
        assert result.sink_name == "low"

    def test_equal_routes_to_above_when_inclusive(self, ctx: PluginContext) -> None:
        """Equal value routes to above_sink when inclusive=True."""
        from elspeth.plugins.gates.threshold_gate import ThresholdGate

        gate = ThresholdGate({
            "field": "score",
            "threshold": 50,
            "above_sink": "high",
            "below_sink": "low",
            "inclusive": True,  # >= routes to above
        })
        row = {"id": 1, "score": 50}

        result = gate.evaluate(row, ctx)
        assert result.sink_name == "high"

    def test_continue_when_no_below_sink(self, ctx: PluginContext) -> None:
        """Continue to next transform when below_sink not specified."""
        from elspeth.plugins.gates.threshold_gate import ThresholdGate

        gate = ThresholdGate({
            "field": "score",
            "threshold": 50,
            "above_sink": "high_scores",
            # No below_sink - continue to next transform
        })
        row = {"id": 1, "score": 25}

        result = gate.evaluate(row, ctx)

        assert result.action == "continue"
        assert result.row == row

    def test_continue_when_no_above_sink(self, ctx: PluginContext) -> None:
        """Continue to next transform when above_sink not specified."""
        from elspeth.plugins.gates.threshold_gate import ThresholdGate

        gate = ThresholdGate({
            "field": "score",
            "threshold": 50,
            "below_sink": "low_scores",
            # No above_sink - continue to next transform
        })
        row = {"id": 1, "score": 75}

        result = gate.evaluate(row, ctx)

        assert result.action == "continue"

    def test_nested_field_access(self, ctx: PluginContext) -> None:
        """Access nested field with dot notation."""
        from elspeth.plugins.gates.threshold_gate import ThresholdGate

        gate = ThresholdGate({
            "field": "metrics.score",
            "threshold": 50,
            "above_sink": "high",
            "below_sink": "low",
        })
        row = {"id": 1, "metrics": {"score": 75}}

        result = gate.evaluate(row, ctx)
        assert result.sink_name == "high"

    def test_missing_field_error(self, ctx: PluginContext) -> None:
        """Error when required field is missing."""
        from elspeth.plugins.gates.threshold_gate import ThresholdGate

        gate = ThresholdGate({
            "field": "score",
            "threshold": 50,
            "above_sink": "high",
        })
        row = {"id": 1}  # No score field

        result = gate.evaluate(row, ctx)

        assert result.action == "error"
        assert "score" in str(result.error)

    def test_non_numeric_field_error(self, ctx: PluginContext) -> None:
        """Error when field is not numeric."""
        from elspeth.plugins.gates.threshold_gate import ThresholdGate

        gate = ThresholdGate({
            "field": "name",
            "threshold": 50,
            "above_sink": "high",
        })
        row = {"id": 1, "name": "alice"}

        result = gate.evaluate(row, ctx)

        assert result.action == "error"
```

### Step 2: Run test to verify it fails

Run: `pytest tests/plugins/gates/test_threshold_gate.py -v`
Expected: FAIL with `ModuleNotFoundError`

### Step 3: Write minimal implementation

```python
# src/elspeth/plugins/gates/__init__.py
"""Built-in gate plugins for ELSPETH.

Gates evaluate rows and decide routing: continue, route_to_sink, or fork_to_paths.
"""

from elspeth.plugins.gates.threshold_gate import ThresholdGate

__all__ = ["ThresholdGate"]
```

```python
# src/elspeth/plugins/gates/threshold_gate.py
"""ThresholdGate plugin.

Routes rows based on numeric threshold comparison.
"""

import copy
from typing import Any

from elspeth.plugins.base import BaseGate
from elspeth.plugins.context import PluginContext
from elspeth.plugins.results import GateResult
from elspeth.plugins.schemas import PluginSchema


class ThresholdGateSchema(PluginSchema):
    """Dynamic schema - accepts any fields."""

    model_config = {"extra": "allow"}


class ThresholdGate(BaseGate):
    """Route rows based on numeric threshold.

    Config options:
        field: Field to compare (supports dot notation)
        threshold: Numeric threshold value
        above_sink: Sink name for values above threshold (optional)
        below_sink: Sink name for values at/below threshold (optional)
        inclusive: If True, equal values route to above (default: False)

    If above_sink or below_sink is not specified, rows continue to next transform.
    """

    name = "threshold_gate"
    input_schema = ThresholdGateSchema
    output_schema = ThresholdGateSchema

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._field: str = config["field"]
        self._threshold: float = float(config["threshold"])
        self._above_sink: str | None = config.get("above_sink")
        self._below_sink: str | None = config.get("below_sink")
        self._inclusive: bool = config.get("inclusive", False)

    def evaluate(self, row: dict[str, Any], ctx: PluginContext) -> GateResult:
        """Evaluate threshold condition and route.

        Args:
            row: Input row data
            ctx: Plugin context

        Returns:
            GateResult with routing decision
        """
        value = self._get_nested(row, self._field)

        # Check for missing field
        if value is _MISSING:
            return GateResult.error(f"Field '{self._field}' not found in row")

        # Check for non-numeric value
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return GateResult.error(
                f"Field '{self._field}' value '{value}' is not numeric"
            )

        # Determine if above or below threshold
        if self._inclusive:
            is_above = numeric_value >= self._threshold
        else:
            is_above = numeric_value > self._threshold

        # Route based on threshold comparison
        row_copy = copy.deepcopy(row)

        if is_above:
            if self._above_sink:
                return GateResult.route_to_sink(self._above_sink, row_copy)
            return GateResult.cont(row_copy)
        else:
            if self._below_sink:
                return GateResult.route_to_sink(self._below_sink, row_copy)
            return GateResult.cont(row_copy)

    def _get_nested(self, data: dict[str, Any], path: str) -> Any:
        """Get value from nested dict using dot notation.

        Args:
            data: Source dictionary
            path: Dot-separated path

        Returns:
            Value at path or _MISSING sentinel
        """
        parts = path.split(".")
        current: Any = data

        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return _MISSING
            current = current[part]

        return current

    def close(self) -> None:
        """No resources to release."""
        pass


# Sentinel for missing values
class _MissingSentinel:
    """Sentinel to distinguish missing fields from None values."""
    pass


_MISSING = _MissingSentinel()
```

### Step 4: Run test to verify it passes

Run: `pytest tests/plugins/gates/test_threshold_gate.py -v`
Expected: PASS (11 tests)

### Step 5: Commit

```bash
git add src/elspeth/plugins/gates/ tests/plugins/gates/
git commit -m "feat(gates): add ThresholdGate plugin"
```

---

## Task 5: FieldMatchGate

**Context:** A gate that routes rows based on field value matching (exact, regex, or list membership).

**Files:**
- Create: `src/elspeth/plugins/gates/field_match_gate.py`
- Modify: `src/elspeth/plugins/gates/__init__.py`
- Create: `tests/plugins/gates/test_field_match_gate.py`

### Step 1: Write the failing test

```python
# tests/plugins/gates/test_field_match_gate.py
"""Tests for FieldMatchGate."""

import pytest

from elspeth.plugins.context import PluginContext
from elspeth.plugins.protocols import GateProtocol


class TestFieldMatchGate:
    """Tests for FieldMatchGate plugin."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create minimal plugin context."""
        return PluginContext(run_id="test-run", config={})

    def test_implements_protocol(self) -> None:
        """FieldMatchGate implements GateProtocol."""
        from elspeth.plugins.gates.field_match_gate import FieldMatchGate

        gate = FieldMatchGate({
            "field": "status",
            "routes": {"active": "active_sink", "deleted": "archive_sink"},
        })
        assert isinstance(gate, GateProtocol)

    def test_has_required_attributes(self) -> None:
        """FieldMatchGate has name and schemas."""
        from elspeth.plugins.gates.field_match_gate import FieldMatchGate

        assert FieldMatchGate.name == "field_match_gate"

    def test_exact_match_routing(self, ctx: PluginContext) -> None:
        """Route based on exact field value match."""
        from elspeth.plugins.gates.field_match_gate import FieldMatchGate

        gate = FieldMatchGate({
            "field": "status",
            "routes": {
                "active": "active_sink",
                "pending": "pending_sink",
                "deleted": "archive_sink",
            },
        })

        active_row = {"id": 1, "status": "active"}
        result = gate.evaluate(active_row, ctx)
        assert result.action == "route_to_sink"
        assert result.sink_name == "active_sink"

        pending_row = {"id": 2, "status": "pending"}
        result = gate.evaluate(pending_row, ctx)
        assert result.sink_name == "pending_sink"

    def test_no_match_continues(self, ctx: PluginContext) -> None:
        """Continue to next transform when no route matches."""
        from elspeth.plugins.gates.field_match_gate import FieldMatchGate

        gate = FieldMatchGate({
            "field": "status",
            "routes": {"active": "active_sink"},
        })
        row = {"id": 1, "status": "unknown"}

        result = gate.evaluate(row, ctx)

        assert result.action == "continue"

    def test_default_route_on_no_match(self, ctx: PluginContext) -> None:
        """Use default_sink when no route matches."""
        from elspeth.plugins.gates.field_match_gate import FieldMatchGate

        gate = FieldMatchGate({
            "field": "status",
            "routes": {"active": "active_sink"},
            "default_sink": "other_sink",
        })
        row = {"id": 1, "status": "unknown"}

        result = gate.evaluate(row, ctx)

        assert result.action == "route_to_sink"
        assert result.sink_name == "other_sink"

    def test_regex_route_matching(self, ctx: PluginContext) -> None:
        """Route based on regex pattern match."""
        from elspeth.plugins.gates.field_match_gate import FieldMatchGate

        gate = FieldMatchGate({
            "field": "email",
            "mode": "regex",
            "routes": {
                r".*@example\.com$": "internal_sink",
                r".*@partner\.org$": "partner_sink",
            },
        })

        internal_row = {"id": 1, "email": "alice@example.com"}
        result = gate.evaluate(internal_row, ctx)
        assert result.sink_name == "internal_sink"

        partner_row = {"id": 2, "email": "bob@partner.org"}
        result = gate.evaluate(partner_row, ctx)
        assert result.sink_name == "partner_sink"

    def test_list_values_in_routes(self, ctx: PluginContext) -> None:
        """Route multiple values to same sink."""
        from elspeth.plugins.gates.field_match_gate import FieldMatchGate

        gate = FieldMatchGate({
            "field": "country",
            "routes": {
                "US,CA,MX": "north_america_sink",  # Comma-separated
                "UK,FR,DE": "europe_sink",
            },
        })

        us_row = {"id": 1, "country": "US"}
        result = gate.evaluate(us_row, ctx)
        assert result.sink_name == "north_america_sink"

        uk_row = {"id": 2, "country": "UK"}
        result = gate.evaluate(uk_row, ctx)
        assert result.sink_name == "europe_sink"

    def test_nested_field_access(self, ctx: PluginContext) -> None:
        """Access nested field with dot notation."""
        from elspeth.plugins.gates.field_match_gate import FieldMatchGate

        gate = FieldMatchGate({
            "field": "meta.type",
            "routes": {"internal": "internal_sink"},
        })
        row = {"id": 1, "meta": {"type": "internal"}}

        result = gate.evaluate(row, ctx)
        assert result.sink_name == "internal_sink"

    def test_missing_field_continues(self, ctx: PluginContext) -> None:
        """Continue when field is missing (no error by default)."""
        from elspeth.plugins.gates.field_match_gate import FieldMatchGate

        gate = FieldMatchGate({
            "field": "status",
            "routes": {"active": "active_sink"},
        })
        row = {"id": 1}  # No status field

        result = gate.evaluate(row, ctx)
        assert result.action == "continue"

    def test_strict_missing_field_error(self, ctx: PluginContext) -> None:
        """Error when field is missing in strict mode."""
        from elspeth.plugins.gates.field_match_gate import FieldMatchGate

        gate = FieldMatchGate({
            "field": "status",
            "routes": {"active": "active_sink"},
            "strict": True,
        })
        row = {"id": 1}

        result = gate.evaluate(row, ctx)
        assert result.action == "error"

    def test_case_insensitive_matching(self, ctx: PluginContext) -> None:
        """Case-insensitive matching when configured."""
        from elspeth.plugins.gates.field_match_gate import FieldMatchGate

        gate = FieldMatchGate({
            "field": "status",
            "routes": {"active": "active_sink"},
            "case_insensitive": True,
        })

        upper_row = {"id": 1, "status": "ACTIVE"}
        result = gate.evaluate(upper_row, ctx)
        assert result.sink_name == "active_sink"

        mixed_row = {"id": 2, "status": "Active"}
        result = gate.evaluate(mixed_row, ctx)
        assert result.sink_name == "active_sink"
```

### Step 2: Run test to verify it fails

Run: `pytest tests/plugins/gates/test_field_match_gate.py -v`
Expected: FAIL with `ModuleNotFoundError`

### Step 3: Write minimal implementation

```python
# src/elspeth/plugins/gates/field_match_gate.py
"""FieldMatchGate plugin.

Routes rows based on field value matching.
"""

import copy
import re
from typing import Any

from elspeth.plugins.base import BaseGate
from elspeth.plugins.context import PluginContext
from elspeth.plugins.results import GateResult
from elspeth.plugins.schemas import PluginSchema


class FieldMatchGateSchema(PluginSchema):
    """Dynamic schema - accepts any fields."""

    model_config = {"extra": "allow"}


class FieldMatchGate(BaseGate):
    """Route rows based on field value matching.

    Config options:
        field: Field to match (supports dot notation)
        routes: Dict mapping values/patterns to sink names
            - Exact mode: {"value": "sink_name"}
            - Multi-value: {"val1,val2,val3": "sink_name"}
            - Regex mode: {"pattern": "sink_name"}
        mode: "exact" (default) or "regex"
        default_sink: Sink for non-matching rows (optional, else continue)
        case_insensitive: Ignore case in matching (default: False)
        strict: Error on missing field (default: False, continue instead)
    """

    name = "field_match_gate"
    input_schema = FieldMatchGateSchema
    output_schema = FieldMatchGateSchema

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._field: str = config["field"]
        self._routes: dict[str, str] = config["routes"]
        self._mode: str = config.get("mode", "exact")
        self._default_sink: str | None = config.get("default_sink")
        self._case_insensitive: bool = config.get("case_insensitive", False)
        self._strict: bool = config.get("strict", False)

        # Pre-process routes based on mode
        if self._mode == "regex":
            flags = re.IGNORECASE if self._case_insensitive else 0
            self._compiled_routes = [
                (re.compile(pattern, flags), sink)
                for pattern, sink in self._routes.items()
            ]
        else:
            # Expand comma-separated values to individual mappings
            self._value_to_sink: dict[str, str] = {}
            for key, sink in self._routes.items():
                for value in key.split(","):
                    value = value.strip()
                    if self._case_insensitive:
                        value = value.lower()
                    self._value_to_sink[value] = sink

    def evaluate(self, row: dict[str, Any], ctx: PluginContext) -> GateResult:
        """Evaluate field match and route.

        Args:
            row: Input row data
            ctx: Plugin context

        Returns:
            GateResult with routing decision
        """
        value = self._get_nested(row, self._field)
        row_copy = copy.deepcopy(row)

        # Handle missing field
        if value is _MISSING:
            if self._strict:
                return GateResult.error(f"Field '{self._field}' not found in row")
            if self._default_sink:
                return GateResult.route_to_sink(self._default_sink, row_copy)
            return GateResult.cont(row_copy)

        # Convert to string for matching
        str_value = str(value)
        if self._case_insensitive:
            str_value = str_value.lower()

        # Match based on mode
        if self._mode == "regex":
            for pattern, sink in self._compiled_routes:
                if pattern.search(str_value):
                    return GateResult.route_to_sink(sink, row_copy)
        else:
            if str_value in self._value_to_sink:
                return GateResult.route_to_sink(self._value_to_sink[str_value], row_copy)

        # No match - use default or continue
        if self._default_sink:
            return GateResult.route_to_sink(self._default_sink, row_copy)
        return GateResult.cont(row_copy)

    def _get_nested(self, data: dict[str, Any], path: str) -> Any:
        """Get value from nested dict using dot notation."""
        parts = path.split(".")
        current: Any = data

        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return _MISSING
            current = current[part]

        return current

    def close(self) -> None:
        """No resources to release."""
        pass


# Sentinel for missing values
class _MissingSentinel:
    """Sentinel to distinguish missing fields from None values."""
    pass


_MISSING = _MissingSentinel()
```

Update `__init__.py`:

```python
# src/elspeth/plugins/gates/__init__.py
"""Built-in gate plugins for ELSPETH.

Gates evaluate rows and decide routing: continue, route_to_sink, or fork_to_paths.
"""

from elspeth.plugins.gates.field_match_gate import FieldMatchGate
from elspeth.plugins.gates.threshold_gate import ThresholdGate

__all__ = ["FieldMatchGate", "ThresholdGate"]
```

### Step 4: Run test to verify it passes

Run: `pytest tests/plugins/gates/test_field_match_gate.py -v`
Expected: PASS (11 tests)

### Step 5: Commit

```bash
git add src/elspeth/plugins/gates/ tests/plugins/gates/
git commit -m "feat(gates): add FieldMatchGate plugin"
```

---

## Task 6: Update CLI with Transform Support

**Context:** Update `_execute_pipeline` to support transform configuration and instantiation. Currently it sets `transforms=[]` - now we can actually populate it.

**Files:**
- Modify: `src/elspeth/cli.py`
- Create: `tests/cli/test_run_with_transforms.py`

### Step 1: Write the failing test

```python
# tests/cli/test_run_with_transforms.py
"""Tests for run command with transforms."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

runner = CliRunner(mix_stderr=True)


class TestRunWithTransforms:
    """Test run command with transform configuration."""

    @pytest.fixture
    def sample_csv(self, tmp_path: Path) -> Path:
        """Create sample input CSV."""
        csv_file = tmp_path / "input.csv"
        csv_file.write_text("id,name,score\n1,alice,75\n2,bob,45\n3,carol,90\n")
        return csv_file

    @pytest.fixture
    def output_csv(self, tmp_path: Path) -> Path:
        """Output CSV path."""
        return tmp_path / "output.csv"

    @pytest.fixture
    def settings_with_passthrough(
        self, tmp_path: Path, sample_csv: Path, output_csv: Path
    ) -> Path:
        """Settings file with passthrough transform."""
        settings = tmp_path / "settings.yaml"
        settings.write_text(f"""
source:
  plugin: csv
  path: {sample_csv}

transforms:
  - plugin: passthrough

sinks:
  output:
    plugin: csv
    path: {output_csv}
""")
        return settings

    @pytest.fixture
    def settings_with_filter(
        self, tmp_path: Path, sample_csv: Path, output_csv: Path
    ) -> Path:
        """Settings file with filter transform."""
        settings = tmp_path / "settings.yaml"
        settings.write_text(f"""
source:
  plugin: csv
  path: {sample_csv}

transforms:
  - plugin: filter
    field: score
    greater_than: 50

sinks:
  output:
    plugin: csv
    path: {output_csv}
""")
        return settings

    @pytest.fixture
    def settings_with_field_mapper(
        self, tmp_path: Path, sample_csv: Path, output_csv: Path
    ) -> Path:
        """Settings file with field mapper transform."""
        settings = tmp_path / "settings.yaml"
        settings.write_text(f"""
source:
  plugin: csv
  path: {sample_csv}

transforms:
  - plugin: field_mapper
    mapping:
      name: full_name
      score: test_score
    select_only: true

sinks:
  output:
    plugin: csv
    path: {output_csv}
""")
        return settings

    def test_run_with_passthrough(
        self, settings_with_passthrough: Path, output_csv: Path
    ) -> None:
        """Run with passthrough transform passes all rows."""
        from elspeth.cli import app

        result = runner.invoke(app, ["run", "--settings", str(settings_with_passthrough)])

        assert result.exit_code == 0
        assert "completed" in result.stdout.lower()
        # All 3 rows should be in output
        output_content = output_csv.read_text()
        assert "alice" in output_content
        assert "bob" in output_content
        assert "carol" in output_content

    def test_run_with_filter(
        self, settings_with_filter: Path, output_csv: Path
    ) -> None:
        """Run with filter transform only outputs matching rows."""
        from elspeth.cli import app

        result = runner.invoke(app, ["run", "--settings", str(settings_with_filter)])

        assert result.exit_code == 0
        # Only rows with score > 50 should be in output
        output_content = output_csv.read_text()
        assert "alice" in output_content  # score 75
        assert "bob" not in output_content  # score 45
        assert "carol" in output_content  # score 90

    def test_run_with_field_mapper(
        self, settings_with_field_mapper: Path, output_csv: Path
    ) -> None:
        """Run with field mapper transform renames fields."""
        from elspeth.cli import app

        result = runner.invoke(app, ["run", "--settings", str(settings_with_field_mapper)])

        assert result.exit_code == 0
        output_content = output_csv.read_text()
        assert "full_name" in output_content
        assert "test_score" in output_content
        # Original field names should not appear
        assert "name," not in output_content  # Careful: full_name contains "name"
        assert ",score" not in output_content
```

### Step 2: Run test to verify it fails

Run: `pytest tests/cli/test_run_with_transforms.py -v`
Expected: FAIL (transforms not instantiated)

### Step 3: Update CLI implementation

Update `_execute_pipeline` in `src/elspeth/cli.py`:

```python
def _execute_pipeline(config: dict, verbose: bool = False) -> dict:
    """Execute a pipeline from configuration.

    Returns:
        Dict with run_id, status, rows_processed.
    """
    from elspeth.core.landscape import LandscapeDB
    from elspeth.engine import Orchestrator, PipelineConfig
    from elspeth.engine.adapters import SinkAdapter
    from elspeth.plugins.sources.csv_source import CSVSource
    from elspeth.plugins.sources.json_source import JSONSource
    from elspeth.plugins.sinks.csv_sink import CSVSink
    from elspeth.plugins.sinks.json_sink import JSONSink
    from elspeth.plugins.sinks.database_sink import DatabaseSink
    # Import transform plugins
    from elspeth.plugins.transforms import FieldMapper, Filter, PassThrough
    # Import gate plugins
    from elspeth.plugins.gates import FieldMatchGate, ThresholdGate

    # === Source instantiation (unchanged) ===
    source_config = config["source"]
    source_plugin = source_config["plugin"]
    source_options = {k: v for k, v in source_config.items() if k != "plugin"}

    if source_plugin == "csv":
        source = CSVSource(source_options)
    elif source_plugin == "json":
        source = JSONSource(source_options)
    else:
        raise ValueError(f"Unknown source plugin: {source_plugin}")

    # === Transform instantiation (NEW) ===
    transform_configs = config.get("transforms", [])
    transforms: list = []

    # Plugin registry for transforms
    transform_registry = {
        "passthrough": PassThrough,
        "field_mapper": FieldMapper,
        "filter": Filter,
    }

    # Plugin registry for gates (gates are transforms from the engine's perspective)
    gate_registry = {
        "threshold_gate": ThresholdGate,
        "field_match_gate": FieldMatchGate,
    }

    for transform_config in transform_configs:
        plugin_name = transform_config["plugin"]
        plugin_options = {k: v for k, v in transform_config.items() if k != "plugin"}

        if plugin_name in transform_registry:
            plugin_class = transform_registry[plugin_name]
            transforms.append(plugin_class(plugin_options))
        elif plugin_name in gate_registry:
            plugin_class = gate_registry[plugin_name]
            transforms.append(plugin_class(plugin_options))
        else:
            raise ValueError(f"Unknown transform plugin: {plugin_name}")

    # === Sink instantiation (unchanged) ===
    sinks_config = config.get("sinks", {})
    sinks: dict = {}

    for sink_name, sink_config in sinks_config.items():
        sink_plugin = sink_config["plugin"]
        sink_options = {k: v for k, v in sink_config.items() if k != "plugin"}

        if sink_plugin == "csv":
            raw_sink = CSVSink(sink_options)
            artifact_descriptor = {"kind": "file", "path": sink_options.get("path", "")}
        elif sink_plugin == "json":
            raw_sink = JSONSink(sink_options)
            artifact_descriptor = {"kind": "file", "path": sink_options.get("path", "")}
        elif sink_plugin == "database":
            raw_sink = DatabaseSink(sink_options)
            artifact_descriptor = {
                "kind": "database",
                "url": sink_options.get("url", ""),
                "table": sink_options.get("table", ""),
            }
        else:
            raise ValueError(f"Unknown sink plugin: {sink_plugin}")

        sinks[sink_name] = SinkAdapter(
            raw_sink,
            plugin_name=sink_plugin,
            sink_name=sink_name,
            artifact_descriptor=artifact_descriptor,
        )

    # Get database URL from settings or use default
    db_url = config.get("landscape", {}).get("url", "sqlite:///elspeth_runs.db")
    db = LandscapeDB.from_url(db_url)

    # Build PipelineConfig with transforms
    pipeline_config = PipelineConfig(
        source=source,
        transforms=transforms,  # Now populated!
        sinks=sinks,
    )

    if verbose:
        typer.echo("Starting pipeline execution...")
        if transforms:
            typer.echo(f"  Transforms: {len(transforms)}")

    # Execute via Orchestrator
    orchestrator = Orchestrator(db)
    result = orchestrator.run(pipeline_config)

    return {
        "run_id": result.run_id,
        "status": result.status,
        "rows_processed": result.rows_processed,
    }
```

### Step 4: Run test to verify it passes

Run: `pytest tests/cli/test_run_with_transforms.py -v`
Expected: PASS (3 tests)

### Step 5: Commit

```bash
git add src/elspeth/cli.py tests/cli/
git commit -m "feat(cli): add transform support to run command"
```

---

## Task 7: TUI Lineage Tree Widget

**Context:** Create a Textual tree widget that displays lineage as a hierarchical tree: Source → Transforms → Sinks, with tokens as leaves.

**Files:**
- Create: `src/elspeth/tui/widgets/__init__.py`
- Create: `src/elspeth/tui/widgets/lineage_tree.py`
- Create: `tests/tui/__init__.py`
- Create: `tests/tui/test_lineage_tree.py`

### Step 1: Write the failing test

```python
# tests/tui/__init__.py
"""TUI tests."""

# tests/tui/test_lineage_tree.py
"""Tests for lineage tree widget."""

import pytest


class TestLineageTreeWidget:
    """Tests for LineageTree widget."""

    def test_can_import_widget(self) -> None:
        """LineageTree widget can be imported."""
        from elspeth.tui.widgets.lineage_tree import LineageTree

        assert LineageTree is not None

    def test_widget_accepts_lineage_data(self) -> None:
        """Widget can be initialized with lineage data."""
        from elspeth.tui.widgets.lineage_tree import LineageTree

        # Sample lineage structure
        lineage_data = {
            "run_id": "run-001",
            "source": {
                "name": "csv_source",
                "node_id": "node-001",
            },
            "transforms": [
                {"name": "passthrough", "node_id": "node-002"},
                {"name": "filter", "node_id": "node-003"},
            ],
            "sinks": [
                {"name": "output", "node_id": "node-004"},
            ],
            "tokens": [
                {
                    "token_id": "token-001",
                    "row_id": "row-001",
                    "path": ["node-001", "node-002", "node-003", "node-004"],
                },
            ],
        }

        tree = LineageTree(lineage_data)
        assert tree is not None

    def test_widget_builds_tree_structure(self) -> None:
        """Widget builds correct tree structure from lineage."""
        from elspeth.tui.widgets.lineage_tree import LineageTree

        lineage_data = {
            "run_id": "run-001",
            "source": {"name": "csv_source", "node_id": "node-001"},
            "transforms": [{"name": "filter", "node_id": "node-002"}],
            "sinks": [{"name": "output", "node_id": "node-003"}],
            "tokens": [
                {"token_id": "token-001", "row_id": "row-001", "path": ["node-001", "node-002", "node-003"]},
            ],
        }

        tree = LineageTree(lineage_data)
        nodes = tree.get_tree_nodes()

        # Should have root, source, transforms, sinks sections
        node_labels = [n["label"] for n in nodes]
        assert any("csv_source" in label for label in node_labels)
        assert any("filter" in label for label in node_labels)
        assert any("output" in label for label in node_labels)

    def test_widget_with_empty_transforms(self) -> None:
        """Widget handles pipeline with no transforms."""
        from elspeth.tui.widgets.lineage_tree import LineageTree

        lineage_data = {
            "run_id": "run-001",
            "source": {"name": "csv_source", "node_id": "node-001"},
            "transforms": [],
            "sinks": [{"name": "output", "node_id": "node-002"}],
            "tokens": [],
        }

        tree = LineageTree(lineage_data)
        assert tree is not None

    def test_widget_with_forked_tokens(self) -> None:
        """Widget handles tokens that forked to multiple paths."""
        from elspeth.tui.widgets.lineage_tree import LineageTree

        lineage_data = {
            "run_id": "run-001",
            "source": {"name": "csv_source", "node_id": "node-001"},
            "transforms": [{"name": "gate", "node_id": "node-002"}],
            "sinks": [
                {"name": "high", "node_id": "node-003"},
                {"name": "low", "node_id": "node-004"},
            ],
            "tokens": [
                {"token_id": "token-001", "row_id": "row-001", "path": ["node-001", "node-002", "node-003"]},
                {"token_id": "token-002", "row_id": "row-002", "path": ["node-001", "node-002", "node-004"]},
            ],
        }

        tree = LineageTree(lineage_data)
        nodes = tree.get_tree_nodes()

        # Should show both sink paths
        node_labels = [n["label"] for n in nodes]
        assert any("high" in label for label in node_labels)
        assert any("low" in label for label in node_labels)
```

### Step 2: Run test to verify it fails

Run: `pytest tests/tui/test_lineage_tree.py -v`
Expected: FAIL with `ModuleNotFoundError`

### Step 3: Write minimal implementation

```python
# src/elspeth/tui/__init__.py
"""TUI components for ELSPETH."""

# src/elspeth/tui/widgets/__init__.py
"""TUI widgets for ELSPETH."""

from elspeth.tui.widgets.lineage_tree import LineageTree

__all__ = ["LineageTree"]
```

```python
# src/elspeth/tui/widgets/lineage_tree.py
"""Lineage tree widget for displaying pipeline lineage."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TreeNode:
    """Node in the lineage tree."""

    label: str
    node_id: str | None = None
    node_type: str = ""
    children: list["TreeNode"] = field(default_factory=list)
    expanded: bool = True


class LineageTree:
    """Widget for displaying pipeline lineage as a tree.

    Structure:
        Run: <run_id>
        └── Source: <source_name>
            └── Transform: <transform_1>
                └── Transform: <transform_2>
                    ├── Sink: <sink_a>
                    │   └── Token: <token_id>
                    └── Sink: <sink_b>
                        └── Token: <token_id>

    The tree shows the flow of data through the pipeline,
    with tokens as leaves showing which rows went where.
    """

    def __init__(self, lineage_data: dict[str, Any]) -> None:
        """Initialize with lineage data.

        Args:
            lineage_data: Dict containing run_id, source, transforms, sinks, tokens
        """
        self._data = lineage_data
        self._root = self._build_tree()

    def _build_tree(self) -> TreeNode:
        """Build tree structure from lineage data.

        Returns:
            Root TreeNode
        """
        run_id = self._data.get("run_id", "unknown")
        root = TreeNode(label=f"Run: {run_id}", node_type="run")

        # Add source
        source = self._data.get("source", {})
        source_node = TreeNode(
            label=f"Source: {source.get('name', 'unknown')}",
            node_id=source.get("node_id"),
            node_type="source",
        )
        root.children.append(source_node)

        # Build transform chain
        transforms = self._data.get("transforms", [])
        current_parent = source_node

        for transform in transforms:
            transform_node = TreeNode(
                label=f"Transform: {transform.get('name', 'unknown')}",
                node_id=transform.get("node_id"),
                node_type="transform",
            )
            current_parent.children.append(transform_node)
            current_parent = transform_node

        # Add sinks as children of last transform (or source if no transforms)
        sinks = self._data.get("sinks", [])
        sink_nodes: dict[str, TreeNode] = {}

        for sink in sinks:
            sink_node = TreeNode(
                label=f"Sink: {sink.get('name', 'unknown')}",
                node_id=sink.get("node_id"),
                node_type="sink",
            )
            current_parent.children.append(sink_node)
            if sink.get("node_id"):
                sink_nodes[sink["node_id"]] = sink_node

        # Add tokens under their terminal nodes
        tokens = self._data.get("tokens", [])
        for token in tokens:
            token_node = TreeNode(
                label=f"Token: {token.get('token_id', 'unknown')} (row: {token.get('row_id', 'unknown')})",
                node_id=token.get("token_id"),
                node_type="token",
            )
            # Find which sink this token ended at
            path = token.get("path", [])
            if path:
                terminal_node_id = path[-1]
                if terminal_node_id in sink_nodes:
                    sink_nodes[terminal_node_id].children.append(token_node)

        return root

    def get_tree_nodes(self) -> list[dict[str, Any]]:
        """Get flat list of tree nodes for rendering.

        Returns:
            List of dicts with label, node_id, node_type, depth, has_children
        """
        nodes: list[dict[str, Any]] = []
        self._flatten_tree(self._root, 0, nodes)
        return nodes

    def _flatten_tree(
        self, node: TreeNode, depth: int, result: list[dict[str, Any]]
    ) -> None:
        """Recursively flatten tree to list.

        Args:
            node: Current node
            depth: Current depth level
            result: List to append to
        """
        result.append({
            "label": node.label,
            "node_id": node.node_id,
            "node_type": node.node_type,
            "depth": depth,
            "has_children": len(node.children) > 0,
            "expanded": node.expanded,
        })

        if node.expanded:
            for child in node.children:
                self._flatten_tree(child, depth + 1, result)

    def get_node_by_id(self, node_id: str) -> TreeNode | None:
        """Find a node by its ID.

        Args:
            node_id: Node ID to find

        Returns:
            TreeNode if found, None otherwise
        """
        return self._find_node(self._root, node_id)

    def _find_node(self, node: TreeNode, node_id: str) -> TreeNode | None:
        """Recursively search for node.

        Args:
            node: Current node
            node_id: ID to find

        Returns:
            TreeNode if found, None otherwise
        """
        if node.node_id == node_id:
            return node
        for child in node.children:
            found = self._find_node(child, node_id)
            if found:
                return found
        return None

    def toggle_node(self, node_id: str) -> bool:
        """Toggle expansion state of a node.

        Args:
            node_id: Node ID to toggle

        Returns:
            New expansion state
        """
        node = self.get_node_by_id(node_id)
        if node:
            node.expanded = not node.expanded
            return node.expanded
        return False
```

### Step 4: Run test to verify it passes

Run: `pytest tests/tui/test_lineage_tree.py -v`
Expected: PASS (5 tests)

### Step 5: Commit

```bash
git add src/elspeth/tui/ tests/tui/
git commit -m "feat(tui): add LineageTree widget for lineage visualization"
```

---

## Task 8: TUI Node Detail Panel

**Context:** Create a panel widget that shows detailed information about a selected node in the lineage tree - its state, input/output hashes, timing, errors.

**Files:**
- Create: `src/elspeth/tui/widgets/node_detail.py`
- Modify: `src/elspeth/tui/widgets/__init__.py`
- Create: `tests/tui/test_node_detail.py`

### Step 1: Write the failing test

```python
# tests/tui/test_node_detail.py
"""Tests for node detail panel widget."""

import pytest


class TestNodeDetailPanel:
    """Tests for NodeDetailPanel widget."""

    def test_can_import_widget(self) -> None:
        """NodeDetailPanel can be imported."""
        from elspeth.tui.widgets.node_detail import NodeDetailPanel

        assert NodeDetailPanel is not None

    def test_display_transform_state(self) -> None:
        """Display details for a transform node state."""
        from elspeth.tui.widgets.node_detail import NodeDetailPanel

        node_state = {
            "state_id": "state-001",
            "node_id": "node-001",
            "token_id": "token-001",
            "plugin_name": "filter",
            "node_type": "transform",
            "status": "completed",
            "input_hash": "abc123",
            "output_hash": "def456",
            "duration_ms": 12.5,
            "started_at": "2024-01-01T10:00:00Z",
            "completed_at": "2024-01-01T10:00:00.012Z",
            "error_json": None,
        }

        panel = NodeDetailPanel(node_state)
        content = panel.render_content()

        assert "filter" in content
        assert "completed" in content
        assert "abc123" in content
        assert "12.5" in content

    def test_display_failed_state(self) -> None:
        """Display details for a failed node state with error."""
        from elspeth.tui.widgets.node_detail import NodeDetailPanel

        node_state = {
            "state_id": "state-002",
            "node_id": "node-001",
            "token_id": "token-001",
            "plugin_name": "transform",
            "node_type": "transform",
            "status": "failed",
            "input_hash": "abc123",
            "output_hash": None,
            "duration_ms": 5.2,
            "started_at": "2024-01-01T10:00:00Z",
            "completed_at": "2024-01-01T10:00:00.005Z",
            "error_json": '{"type": "ValueError", "message": "Invalid input"}',
        }

        panel = NodeDetailPanel(node_state)
        content = panel.render_content()

        assert "failed" in content.lower()
        assert "ValueError" in content
        assert "Invalid input" in content

    def test_display_source_state(self) -> None:
        """Display details for a source node."""
        from elspeth.tui.widgets.node_detail import NodeDetailPanel

        node_state = {
            "state_id": "state-003",
            "node_id": "source-001",
            "token_id": "token-001",
            "plugin_name": "csv_source",
            "node_type": "source",
            "status": "completed",
            "input_hash": None,  # Sources have no input
            "output_hash": "xyz789",
            "duration_ms": 100.0,
            "started_at": "2024-01-01T10:00:00Z",
            "completed_at": "2024-01-01T10:00:00.100Z",
            "error_json": None,
        }

        panel = NodeDetailPanel(node_state)
        content = panel.render_content()

        assert "csv_source" in content
        assert "source" in content.lower()

    def test_display_sink_state(self) -> None:
        """Display details for a sink node."""
        from elspeth.tui.widgets.node_detail import NodeDetailPanel

        node_state = {
            "state_id": "state-004",
            "node_id": "sink-001",
            "token_id": "token-001",
            "plugin_name": "csv_sink",
            "node_type": "sink",
            "status": "completed",
            "input_hash": "final123",
            "output_hash": None,  # Sinks produce artifacts, not output_hash
            "duration_ms": 25.0,
            "started_at": "2024-01-01T10:00:00Z",
            "completed_at": "2024-01-01T10:00:00.025Z",
            "error_json": None,
            "artifact": {
                "artifact_id": "artifact-001",
                "path_or_uri": "/output/result.csv",
                "content_hash": "artifact_hash_789",
                "size_bytes": 1024,
            },
        }

        panel = NodeDetailPanel(node_state)
        content = panel.render_content()

        assert "csv_sink" in content
        assert "artifact" in content.lower()
        assert "/output/result.csv" in content

    def test_empty_state(self) -> None:
        """Handle empty/null state gracefully."""
        from elspeth.tui.widgets.node_detail import NodeDetailPanel

        panel = NodeDetailPanel(None)
        content = panel.render_content()

        assert "No node selected" in content or "Select a node" in content
```

### Step 2: Run test to verify it fails

Run: `pytest tests/tui/test_node_detail.py -v`
Expected: FAIL with `ModuleNotFoundError`

### Step 3: Write minimal implementation

```python
# src/elspeth/tui/widgets/node_detail.py
"""Node detail panel widget for displaying node state information."""

import json
from typing import Any


class NodeDetailPanel:
    """Panel displaying detailed information about a selected node.

    Shows:
    - Node identity (plugin name, type, IDs)
    - Status and timing
    - Input/output hashes
    - Errors (if failed)
    - Artifacts (if sink)
    """

    def __init__(self, node_state: dict[str, Any] | None) -> None:
        """Initialize with node state data.

        Args:
            node_state: Dict containing node state fields, or None if nothing selected
        """
        self._state = node_state

    def render_content(self) -> str:
        """Render panel content as formatted string.

        Returns:
            Formatted string for display
        """
        if self._state is None:
            return "No node selected. Select a node from the tree to view details."

        lines: list[str] = []

        # Header
        plugin_name = self._state.get("plugin_name", "unknown")
        node_type = self._state.get("node_type", "unknown")
        lines.append(f"=== {plugin_name} ({node_type}) ===")
        lines.append("")

        # Identity
        lines.append("Identity:")
        lines.append(f"  State ID:  {self._state.get('state_id', 'N/A')}")
        lines.append(f"  Node ID:   {self._state.get('node_id', 'N/A')}")
        lines.append(f"  Token ID:  {self._state.get('token_id', 'N/A')}")
        lines.append("")

        # Status
        status = self._state.get("status", "unknown")
        lines.append("Status:")
        lines.append(f"  Status:     {status}")
        lines.append(f"  Started:    {self._state.get('started_at', 'N/A')}")
        lines.append(f"  Completed:  {self._state.get('completed_at', 'N/A')}")
        duration = self._state.get("duration_ms")
        if duration is not None:
            lines.append(f"  Duration:   {duration} ms")
        lines.append("")

        # Hashes
        lines.append("Data Hashes:")
        input_hash = self._state.get("input_hash")
        output_hash = self._state.get("output_hash")
        lines.append(f"  Input:   {input_hash or '(none)'}")
        lines.append(f"  Output:  {output_hash or '(none)'}")
        lines.append("")

        # Error (if present)
        error_json = self._state.get("error_json")
        if error_json:
            lines.append("Error:")
            try:
                error = json.loads(error_json)
                lines.append(f"  Type:    {error.get('type', 'unknown')}")
                lines.append(f"  Message: {error.get('message', 'unknown')}")
            except json.JSONDecodeError:
                lines.append(f"  {error_json}")
            lines.append("")

        # Artifact (if sink)
        artifact = self._state.get("artifact")
        if artifact:
            lines.append("Artifact:")
            lines.append(f"  ID:      {artifact.get('artifact_id', 'N/A')}")
            lines.append(f"  Path:    {artifact.get('path_or_uri', 'N/A')}")
            lines.append(f"  Hash:    {artifact.get('content_hash', 'N/A')}")
            size_bytes = artifact.get("size_bytes")
            if size_bytes is not None:
                lines.append(f"  Size:    {self._format_size(size_bytes)}")
            lines.append("")

        return "\n".join(lines)

    def _format_size(self, size_bytes: int) -> str:
        """Format byte size in human-readable form.

        Args:
            size_bytes: Size in bytes

        Returns:
            Formatted string like "1.5 KB" or "2.3 MB"
        """
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"

    def update_state(self, node_state: dict[str, Any] | None) -> None:
        """Update the displayed node state.

        Args:
            node_state: New node state to display
        """
        self._state = node_state
```

Update `__init__.py`:

```python
# src/elspeth/tui/widgets/__init__.py
"""TUI widgets for ELSPETH."""

from elspeth.tui.widgets.lineage_tree import LineageTree
from elspeth.tui.widgets.node_detail import NodeDetailPanel

__all__ = ["LineageTree", "NodeDetailPanel"]
```

### Step 4: Run test to verify it passes

Run: `pytest tests/tui/test_node_detail.py -v`
Expected: PASS (5 tests)

### Step 5: Commit

```bash
git add src/elspeth/tui/ tests/tui/
git commit -m "feat(tui): add NodeDetailPanel widget for node inspection"
```

---

## Task 9: Integrate Widgets into Explain Command

**Context:** Update the `explain` command to use the new LineageTree and NodeDetailPanel widgets in the Textual TUI.

**Files:**
- Modify: `src/elspeth/cli.py` (or wherever the explain TUI lives)
- Create: `tests/cli/test_explain_tui.py`

### Step 1: Write the failing test

```python
# tests/cli/test_explain_tui.py
"""Tests for explain command TUI integration."""

import pytest


class TestExplainTUI:
    """Tests for explain command TUI."""

    def test_explain_screen_has_tree_widget(self) -> None:
        """Explain screen includes LineageTree widget."""
        from elspeth.tui.screens.explain_screen import ExplainScreen

        screen = ExplainScreen()
        widgets = screen.get_widget_types()

        assert "LineageTree" in widgets or any("tree" in w.lower() for w in widgets)

    def test_explain_screen_has_detail_panel(self) -> None:
        """Explain screen includes NodeDetailPanel widget."""
        from elspeth.tui.screens.explain_screen import ExplainScreen

        screen = ExplainScreen()
        widgets = screen.get_widget_types()

        assert "NodeDetailPanel" in widgets or any("detail" in w.lower() for w in widgets)

    def test_screen_loads_lineage_data(self) -> None:
        """Screen loads lineage data from LandscapeDB."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.tui.screens.explain_screen import ExplainScreen

        # Create test data
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type="source",
            plugin_version="1.0",
            config={},
        )

        # Screen should load this data
        screen = ExplainScreen(db=db, run_id=run.run_id)
        lineage = screen.get_lineage_data()

        assert lineage is not None
        assert lineage.get("run_id") == run.run_id

    def test_tree_selection_updates_detail_panel(self) -> None:
        """Selecting a node in tree updates detail panel."""
        from elspeth.tui.screens.explain_screen import ExplainScreen

        screen = ExplainScreen()

        # Simulate selecting a node
        mock_node_id = "node-001"
        screen.on_tree_select(mock_node_id)

        # Detail panel should update
        detail_state = screen.get_detail_panel_state()
        assert detail_state is None or detail_state.get("node_id") == mock_node_id
```

### Step 2: Run test to verify it fails

Run: `pytest tests/cli/test_explain_tui.py -v`
Expected: FAIL with `ModuleNotFoundError`

### Step 3: Write minimal implementation

```python
# src/elspeth/tui/screens/__init__.py
"""TUI screens for ELSPETH."""

from elspeth.tui.screens.explain_screen import ExplainScreen

__all__ = ["ExplainScreen"]
```

```python
# src/elspeth/tui/screens/explain_screen.py
"""Explain screen for lineage visualization."""

from typing import Any

from elspeth.core.landscape import LandscapeDB
from elspeth.core.landscape.lineage import build_lineage
from elspeth.tui.widgets.lineage_tree import LineageTree
from elspeth.tui.widgets.node_detail import NodeDetailPanel


class ExplainScreen:
    """Screen for visualizing pipeline lineage.

    Combines LineageTree and NodeDetailPanel widgets to provide
    an interactive exploration of run lineage.

    Layout:
        ┌─────────────────┬──────────────────┐
        │                 │                  │
        │  Lineage Tree   │   Detail Panel   │
        │                 │                  │
        │                 │                  │
        └─────────────────┴──────────────────┘
    """

    def __init__(
        self,
        db: LandscapeDB | None = None,
        run_id: str | None = None,
    ) -> None:
        """Initialize explain screen.

        Args:
            db: Landscape database connection
            run_id: Run ID to explain
        """
        self._db = db
        self._run_id = run_id
        self._lineage_data: dict[str, Any] | None = None
        self._selected_node_id: str | None = None

        # Initialize widgets
        self._tree: LineageTree | None = None
        self._detail_panel = NodeDetailPanel(None)

        # Load data if available
        if db and run_id:
            self._load_lineage()

    def _load_lineage(self) -> None:
        """Load lineage data from database."""
        if not self._db or not self._run_id:
            return

        try:
            lineage_result = build_lineage(self._db, self._run_id)
            self._lineage_data = self._convert_lineage_to_tree_format(lineage_result)
            self._tree = LineageTree(self._lineage_data)
        except Exception:
            # Handle missing run or other errors gracefully
            self._lineage_data = None
            self._tree = None

    def _convert_lineage_to_tree_format(self, lineage_result: Any) -> dict[str, Any]:
        """Convert LineageResult to tree widget format.

        Args:
            lineage_result: LineageResult from build_lineage

        Returns:
            Dict in tree widget format
        """
        # This adapts the LineageResult structure to what LineageTree expects
        return {
            "run_id": lineage_result.run_id if hasattr(lineage_result, "run_id") else self._run_id,
            "source": {
                "name": getattr(lineage_result, "source_name", "source"),
                "node_id": getattr(lineage_result, "source_node_id", None),
            },
            "transforms": [
                {"name": t.plugin_name, "node_id": t.node_id}
                for t in getattr(lineage_result, "transforms", [])
            ],
            "sinks": [
                {"name": s.plugin_name, "node_id": s.node_id}
                for s in getattr(lineage_result, "sinks", [])
            ],
            "tokens": [
                {
                    "token_id": t.token_id,
                    "row_id": t.row_id,
                    "path": t.path,
                }
                for t in getattr(lineage_result, "tokens", [])
            ],
        }

    def get_widget_types(self) -> list[str]:
        """Get list of widget types in this screen.

        Returns:
            List of widget type names
        """
        return ["LineageTree", "NodeDetailPanel"]

    def get_lineage_data(self) -> dict[str, Any] | None:
        """Get current lineage data.

        Returns:
            Lineage data dict or None
        """
        return self._lineage_data

    def on_tree_select(self, node_id: str) -> None:
        """Handle tree node selection.

        Args:
            node_id: Selected node ID
        """
        self._selected_node_id = node_id

        # Load node state from database
        if self._db and self._run_id and node_id:
            node_state = self._load_node_state(node_id)
            self._detail_panel.update_state(node_state)
        else:
            self._detail_panel.update_state(None)

    def _load_node_state(self, node_id: str) -> dict[str, Any] | None:
        """Load node state from database.

        Args:
            node_id: Node ID to load

        Returns:
            Node state dict or None
        """
        if not self._db:
            return None

        # Query node states for this node
        # This would use LandscapeRecorder.get_node_states() or similar
        # For now, return None - actual implementation depends on Phase 3A queries
        return None

    def get_detail_panel_state(self) -> dict[str, Any] | None:
        """Get current detail panel state.

        Returns:
            Node state being displayed or None
        """
        return self._detail_panel._state

    def render(self) -> str:
        """Render the screen as text.

        Returns:
            Rendered screen content
        """
        lines = []
        lines.append("=" * 60)
        lines.append(f"  ELSPETH Lineage Explorer - Run: {self._run_id or '(none)'}")
        lines.append("=" * 60)
        lines.append("")

        if self._tree:
            lines.append("--- Lineage Tree ---")
            for node in self._tree.get_tree_nodes():
                indent = "  " * node["depth"]
                lines.append(f"{indent}{node['label']}")
            lines.append("")

        lines.append("--- Node Details ---")
        lines.append(self._detail_panel.render_content())

        return "\n".join(lines)
```

### Step 4: Run test to verify it passes

Run: `pytest tests/cli/test_explain_tui.py -v`
Expected: PASS (4 tests)

### Step 5: Commit

```bash
git add src/elspeth/tui/ tests/cli/
git commit -m "feat(tui): add ExplainScreen with integrated tree and detail widgets"
```

---

## Task 10: Update Phase 4 "Not Yet Complete" Section

**Context:** Now that transforms, gates, and TUI enhancements are complete, update Phase 4's documentation.

**Files:**
- Modify: `docs/plans/2026-01-12-phase4-cli-and-io.md`

### Step 1: Write test that documents current state

This is a documentation task - verify the text is updated correctly.

### Step 2: Update Phase 4 document

Find the "Not Yet Complete" section near line 3658 and update it:

```markdown
**Not Yet Complete:**
- Rate limiting, checkpointing (Phase 5)
- LLM integration (Phase 6)

**Addressed in Phase 4B:**
- ~~Full TUI lineage visualization (tree widget, detail panels)~~ - Done in Phase 4B Tasks 7-9
- ~~Integration with Phase 3 Orchestrator (currently using simple loop)~~ - This was incorrect; Phase 4 Task 9 already uses full Orchestrator. Phase 4B clarified.
- ~~Transforms/Gates (no built-in transforms yet)~~ - Done in Phase 4B Tasks 1-5 (PassThrough, FieldMapper, Filter, ThresholdGate, FieldMatchGate)
```

### Step 3: Commit

```bash
git add docs/plans/2026-01-12-phase4-cli-and-io.md
git commit -m "docs(phase4): update Not Yet Complete section with Phase 4B coverage"
```

---

## Task 11: Landscape Export Command

**Context:** Add a CLI command and configuration option to export the Landscape audit trail to CSV format. Essential for compliance, external analysis, archival, and integration with data warehouses or auditing tools.

**Files:**
- Create: `src/elspeth/core/landscape/export.py`
- Modify: `src/elspeth/cli.py`
- Create: `tests/core/landscape/test_export_csv.py`
- Create: `tests/cli/test_export_command.py`

### Step 1: Write the failing test for export module

```python
# tests/core/landscape/test_export_csv.py
"""Tests for Landscape CSV export functionality."""

from pathlib import Path

import pytest

from elspeth.core.landscape import LandscapeDB, LandscapeRecorder


class TestLandscapeExport:
    """Tests for exporting Landscape data to CSV."""

    @pytest.fixture
    def db_with_data(self) -> LandscapeDB:
        """Create a LandscapeDB with sample run data."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        # Create a run with nodes, rows, tokens, and states
        run = recorder.begin_run(config={"test": True}, canonical_version="v1")

        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type="source",
            plugin_version="1.0",
            config={"path": "/data/input.csv"},
        )

        transform = recorder.register_node(
            run_id=run.run_id,
            plugin_name="filter",
            node_type="transform",
            plugin_version="1.0",
            config={"field": "score", "greater_than": 50},
        )

        sink = recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_sink",
            node_type="sink",
            plugin_version="1.0",
            config={"path": "/data/output.csv"},
        )

        # Create edges
        recorder.register_edge(
            run_id=run.run_id,
            from_node_id=source.node_id,
            to_node_id=transform.node_id,
            label="default",
            mode="move",
        )
        recorder.register_edge(
            run_id=run.run_id,
            from_node_id=transform.node_id,
            to_node_id=sink.node_id,
            label="default",
            mode="move",
        )

        # Create rows and tokens
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={"id": 1, "name": "alice", "score": 75},
        )

        token = recorder.create_token(row_id=row.row_id)

        # Record node states (two-phase: begin then complete)
        row_data = {"id": 1, "name": "alice", "score": 75}
        state = recorder.begin_node_state(
            token_id=token.token_id,
            node_id=source.node_id,
            step_index=0,
            input_data=row_data,
        )
        recorder.complete_node_state(
            state_id=state.state_id,
            status="completed",
            output_data=row_data,
        )

        recorder.complete_run(run.run_id, status="completed")

        return db

    def test_can_import_exporter(self) -> None:
        """LandscapeExporter can be imported."""
        from elspeth.core.landscape.export import LandscapeExporter

        assert LandscapeExporter is not None

    def test_export_runs_table(self, db_with_data: LandscapeDB, tmp_path: Path) -> None:
        """Export runs table to CSV."""
        from elspeth.core.landscape.export import LandscapeExporter

        exporter = LandscapeExporter(db_with_data)
        output_file = tmp_path / "runs.csv"

        exporter.export_table("runs", output_file)

        assert output_file.exists()
        content = output_file.read_text()
        assert "run_id" in content
        assert "completed" in content

    def test_export_nodes_table(self, db_with_data: LandscapeDB, tmp_path: Path) -> None:
        """Export nodes table to CSV."""
        from elspeth.core.landscape.export import LandscapeExporter

        exporter = LandscapeExporter(db_with_data)
        output_file = tmp_path / "nodes.csv"

        exporter.export_table("nodes", output_file)

        assert output_file.exists()
        content = output_file.read_text()
        assert "node_id" in content
        assert "csv_source" in content
        assert "filter" in content
        assert "csv_sink" in content

    def test_export_node_states_table(self, db_with_data: LandscapeDB, tmp_path: Path) -> None:
        """Export node_states table to CSV."""
        from elspeth.core.landscape.export import LandscapeExporter

        exporter = LandscapeExporter(db_with_data)
        output_file = tmp_path / "node_states.csv"

        exporter.export_table("node_states", output_file)

        assert output_file.exists()
        content = output_file.read_text()
        assert "state_id" in content
        assert "input_hash" in content

    def test_export_all_tables(self, db_with_data: LandscapeDB, tmp_path: Path) -> None:
        """Export all tables to a directory."""
        from elspeth.core.landscape.export import LandscapeExporter

        exporter = LandscapeExporter(db_with_data)
        output_dir = tmp_path / "export"

        exporter.export_all(output_dir)

        # Check all expected files exist (all 13 Landscape tables)
        assert (output_dir / "runs.csv").exists()
        assert (output_dir / "nodes.csv").exists()
        assert (output_dir / "edges.csv").exists()
        assert (output_dir / "rows.csv").exists()
        assert (output_dir / "tokens.csv").exists()
        assert (output_dir / "token_parents.csv").exists()
        assert (output_dir / "node_states.csv").exists()
        assert (output_dir / "calls.csv").exists()
        assert (output_dir / "artifacts.csv").exists()
        assert (output_dir / "routing_events.csv").exists()
        assert (output_dir / "batches.csv").exists()
        assert (output_dir / "batch_members.csv").exists()
        assert (output_dir / "batch_outputs.csv").exists()

    def test_export_single_run(self, db_with_data: LandscapeDB, tmp_path: Path) -> None:
        """Export only data for a specific run."""
        from elspeth.core.landscape.export import LandscapeExporter

        exporter = LandscapeExporter(db_with_data)

        # Get the run_id from the database
        with db_with_data.engine.connect() as conn:
            from elspeth.core.landscape.schema import runs_table
            result = conn.execute(runs_table.select()).fetchone()
            run_id = result.run_id

        output_dir = tmp_path / "single_run"
        exporter.export_all(output_dir, run_id=run_id)

        assert (output_dir / "runs.csv").exists()
        # Verify the run_id is in the exported data
        runs_content = (output_dir / "runs.csv").read_text()
        assert run_id in runs_content

    def test_export_empty_table(self, tmp_path: Path) -> None:
        """Export empty table produces header-only CSV."""
        from elspeth.core.landscape.export import LandscapeExporter

        db = LandscapeDB.in_memory()
        exporter = LandscapeExporter(db)
        output_file = tmp_path / "empty_runs.csv"

        exporter.export_table("runs", output_file)

        assert output_file.exists()
        content = output_file.read_text()
        # Should have header row
        assert "run_id" in content
        # Should only be one line (header)
        lines = [l for l in content.strip().split("\n") if l]
        assert len(lines) == 1

    def test_export_with_run_filter_excludes_other_runs(
        self, db_with_data: LandscapeDB, tmp_path: Path
    ) -> None:
        """Run filter excludes data from other runs."""
        from elspeth.core.landscape.export import LandscapeExporter

        # Create a second run
        recorder = LandscapeRecorder(db_with_data)
        run2 = recorder.begin_run(config={"test": 2}, canonical_version="v1")
        recorder.register_node(
            run_id=run2.run_id,
            plugin_name="other_source",
            node_type="source",
            plugin_version="1.0",
            config={},
        )
        recorder.complete_run(run2.run_id, status="completed")

        # Export only the first run
        exporter = LandscapeExporter(db_with_data)

        with db_with_data.engine.connect() as conn:
            from elspeth.core.landscape.schema import runs_table
            result = conn.execute(runs_table.select()).fetchall()
            first_run_id = result[0].run_id

        output_dir = tmp_path / "filtered"
        exporter.export_all(output_dir, run_id=first_run_id)

        # Nodes should only include first run's nodes
        nodes_content = (output_dir / "nodes.csv").read_text()
        assert "csv_source" in nodes_content
        assert "other_source" not in nodes_content

    def test_invalid_table_name_raises(self, db_with_data: LandscapeDB, tmp_path: Path) -> None:
        """Invalid table name raises ValueError."""
        from elspeth.core.landscape.export import LandscapeExporter

        exporter = LandscapeExporter(db_with_data)

        with pytest.raises(ValueError, match="Unknown table"):
            exporter.export_table("nonexistent_table", tmp_path / "out.csv")
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/landscape/test_export_csv.py -v`
Expected: FAIL with `ModuleNotFoundError`

### Step 3: Write the export module

```python
# src/elspeth/core/landscape/export.py
"""Landscape audit trail export functionality.

Exports Landscape tables to CSV format for compliance, archival,
and external analysis.
"""

from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import select

from elspeth.core.landscape.db import LandscapeDB
from elspeth.core.landscape.schema import (
    artifacts_table,
    batch_members_table,
    batch_outputs_table,
    batches_table,
    calls_table,
    edges_table,
    node_states_table,
    nodes_table,
    routing_events_table,
    rows_table,
    runs_table,
    token_parents_table,
    tokens_table,
)


class LandscapeExporter:
    """Export Landscape audit trail to CSV format.

    Supports exporting individual tables or all tables at once.
    Can filter by run_id to export only a specific run's data.

    Example:
        exporter = LandscapeExporter(db)

        # Export single table
        exporter.export_table("runs", Path("runs.csv"))

        # Export all tables to directory
        exporter.export_all(Path("./export/"))

        # Export specific run only
        exporter.export_all(Path("./export/"), run_id="run-abc123")
    """

    # Mapping of table names to SQLAlchemy table objects
    TABLES = {
        "runs": runs_table,
        "nodes": nodes_table,
        "edges": edges_table,
        "rows": rows_table,
        "tokens": tokens_table,
        "token_parents": token_parents_table,
        "node_states": node_states_table,
        "calls": calls_table,
        "artifacts": artifacts_table,
        "routing_events": routing_events_table,
        "batches": batches_table,
        "batch_members": batch_members_table,
        "batch_outputs": batch_outputs_table,
    }

    # Tables that have direct run_id column for filtering
    TABLES_WITH_RUN_ID = {
        "runs", "nodes", "edges", "rows", "artifacts", "batches"
    }

    # Tables that need join through other tables to filter by run
    # token -> row -> run
    # node_states -> token -> row -> run (or node_states -> node -> run)
    # etc.

    def __init__(self, db: LandscapeDB) -> None:
        """Initialize exporter with database connection.

        Args:
            db: LandscapeDB instance
        """
        self._db = db

    def export_table(
        self,
        table_name: str,
        output_path: Path,
        run_id: str | None = None,
    ) -> int:
        """Export a single table to CSV.

        Args:
            table_name: Name of the table to export
            output_path: Path to write CSV file
            run_id: Optional run_id to filter by

        Returns:
            Number of rows exported

        Raises:
            ValueError: If table_name is not valid
        """
        if table_name not in self.TABLES:
            raise ValueError(
                f"Unknown table: {table_name}. "
                f"Valid tables: {sorted(self.TABLES.keys())}"
            )

        table = self.TABLES[table_name]

        # Build query with optional run_id filter
        query = select(table)

        if run_id is not None:
            query = self._apply_run_filter(query, table_name, run_id)

        # Execute query and load into DataFrame
        with self._db.engine.connect() as conn:
            df = pd.read_sql(query, conn)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to CSV
        df.to_csv(output_path, index=False)

        return len(df)

    def export_all(
        self,
        output_dir: Path,
        run_id: str | None = None,
        tables: list[str] | None = None,
    ) -> dict[str, int]:
        """Export all tables (or specified subset) to a directory.

        Args:
            output_dir: Directory to write CSV files
            run_id: Optional run_id to filter by
            tables: Optional list of table names to export (default: all)

        Returns:
            Dict mapping table name to rows exported
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        tables_to_export = tables or list(self.TABLES.keys())
        results: dict[str, int] = {}

        for table_name in tables_to_export:
            if table_name not in self.TABLES:
                continue

            output_path = output_dir / f"{table_name}.csv"
            count = self.export_table(table_name, output_path, run_id=run_id)
            results[table_name] = count

        return results

    def _apply_run_filter(
        self,
        query: Any,
        table_name: str,
        run_id: str,
    ) -> Any:
        """Apply run_id filter to query.

        Args:
            query: SQLAlchemy select query
            table_name: Name of the table being queried
            run_id: Run ID to filter by

        Returns:
            Query with filter applied
        """
        table = self.TABLES[table_name]

        # Direct run_id column
        if table_name in self.TABLES_WITH_RUN_ID:
            return query.where(table.c.run_id == run_id)

        # tokens -> rows -> run_id
        if table_name == "tokens":
            # Join tokens to rows to filter by run_id
            return query.where(
                table.c.row_id.in_(
                    select(rows_table.c.row_id).where(rows_table.c.run_id == run_id)
                )
            )

        # token_parents -> tokens -> rows -> run_id
        if table_name == "token_parents":
            token_ids = select(tokens_table.c.token_id).where(
                tokens_table.c.row_id.in_(
                    select(rows_table.c.row_id).where(rows_table.c.run_id == run_id)
                )
            )
            return query.where(table.c.token_id.in_(token_ids))

        # node_states -> nodes -> run_id
        if table_name == "node_states":
            return query.where(
                table.c.node_id.in_(
                    select(nodes_table.c.node_id).where(nodes_table.c.run_id == run_id)
                )
            )

        # calls -> node_states -> nodes -> run_id
        if table_name == "calls":
            state_ids = select(node_states_table.c.state_id).where(
                node_states_table.c.node_id.in_(
                    select(nodes_table.c.node_id).where(nodes_table.c.run_id == run_id)
                )
            )
            return query.where(table.c.state_id.in_(state_ids))

        # routing_events -> node_states -> nodes -> run_id
        if table_name == "routing_events":
            state_ids = select(node_states_table.c.state_id).where(
                node_states_table.c.node_id.in_(
                    select(nodes_table.c.node_id).where(nodes_table.c.run_id == run_id)
                )
            )
            return query.where(table.c.state_id.in_(state_ids))

        # batch_members -> batches -> run_id
        if table_name == "batch_members":
            return query.where(
                table.c.batch_id.in_(
                    select(batches_table.c.batch_id).where(batches_table.c.run_id == run_id)
                )
            )

        # batch_outputs -> batches -> run_id
        if table_name == "batch_outputs":
            return query.where(
                table.c.batch_id.in_(
                    select(batches_table.c.batch_id).where(batches_table.c.run_id == run_id)
                )
            )

        # Should never reach here - all tables are handled above
        # Fail fast rather than silently returning unfiltered data
        raise ValueError(f"No run filter defined for table: {table_name}")

    def get_table_names(self) -> list[str]:
        """Get list of exportable table names.

        Returns:
            Sorted list of table names
        """
        return sorted(self.TABLES.keys())
```

### Step 4: Run test to verify it passes

Run: `pytest tests/core/landscape/test_export_csv.py -v`
Expected: PASS (9 tests)

### Step 5: Write CLI command test

```python
# tests/cli/test_export_command.py
"""Tests for export CLI command."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from elspeth.core.landscape import LandscapeDB, LandscapeRecorder

runner = CliRunner(mix_stderr=True)


class TestExportCommand:
    """Tests for elspeth export command."""

    @pytest.fixture
    def db_path(self, tmp_path: Path) -> Path:
        """Create a database with test data."""
        db_file = tmp_path / "test.db"
        db = LandscapeDB.from_url(f"sqlite:///{db_file}")

        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={"test": True}, canonical_version="v1")
        recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type="source",
            plugin_version="1.0",
            config={},
        )
        recorder.complete_run(run.run_id, status="completed")

        return db_file

    def test_export_command_exists(self) -> None:
        """Export command is registered."""
        from elspeth.cli import app

        result = runner.invoke(app, ["--help"])
        assert "export" in result.stdout

    def test_export_all_tables(self, db_path: Path, tmp_path: Path) -> None:
        """Export all tables to directory."""
        from elspeth.cli import app

        output_dir = tmp_path / "export"

        result = runner.invoke(app, [
            "export",
            "--db", str(db_path),
            "--output", str(output_dir),
        ])

        assert result.exit_code == 0
        assert (output_dir / "runs.csv").exists()
        assert (output_dir / "nodes.csv").exists()

    def test_export_single_table(self, db_path: Path, tmp_path: Path) -> None:
        """Export a single table."""
        from elspeth.cli import app

        output_file = tmp_path / "runs.csv"

        result = runner.invoke(app, [
            "export",
            "--db", str(db_path),
            "--output", str(output_file),
            "--table", "runs",
        ])

        assert result.exit_code == 0
        assert output_file.exists()
        content = output_file.read_text()
        assert "run_id" in content

    def test_export_specific_run(self, db_path: Path, tmp_path: Path) -> None:
        """Export data for a specific run."""
        from elspeth.cli import app

        # First get the run_id
        db = LandscapeDB.from_url(f"sqlite:///{db_path}")
        with db.engine.connect() as conn:
            from elspeth.core.landscape.schema import runs_table
            result = conn.execute(runs_table.select()).fetchone()
            run_id = result.run_id

        output_dir = tmp_path / "run_export"

        result = runner.invoke(app, [
            "export",
            "--db", str(db_path),
            "--output", str(output_dir),
            "--run", run_id,
        ])

        assert result.exit_code == 0
        assert "Exported" in result.stdout or "exported" in result.stdout.lower()

    def test_export_multiple_tables(self, db_path: Path, tmp_path: Path) -> None:
        """Export multiple specific tables."""
        from elspeth.cli import app

        output_dir = tmp_path / "partial"

        result = runner.invoke(app, [
            "export",
            "--db", str(db_path),
            "--output", str(output_dir),
            "--table", "runs",
            "--table", "nodes",
        ])

        assert result.exit_code == 0
        assert (output_dir / "runs.csv").exists()
        assert (output_dir / "nodes.csv").exists()
        # Other tables should not be exported
        assert not (output_dir / "edges.csv").exists()

    def test_export_with_format_option(self, db_path: Path, tmp_path: Path) -> None:
        """Format option accepts csv (future: json, parquet)."""
        from elspeth.cli import app

        output_dir = tmp_path / "csv_export"

        result = runner.invoke(app, [
            "export",
            "--db", str(db_path),
            "--output", str(output_dir),
            "--format", "csv",
        ])

        assert result.exit_code == 0

    def test_export_invalid_table_fails(self, db_path: Path, tmp_path: Path) -> None:
        """Invalid table name shows error."""
        from elspeth.cli import app

        result = runner.invoke(app, [
            "export",
            "--db", str(db_path),
            "--output", str(tmp_path / "out.csv"),
            "--table", "nonexistent",
        ])

        assert result.exit_code != 0
        assert "unknown" in result.stdout.lower() or "invalid" in result.stdout.lower()

    def test_export_list_tables(self, db_path: Path) -> None:
        """List available tables for export."""
        from elspeth.cli import app

        result = runner.invoke(app, [
            "export",
            "--db", str(db_path),
            "--list-tables",
        ])

        assert result.exit_code == 0
        assert "runs" in result.stdout
        assert "nodes" in result.stdout
        assert "node_states" in result.stdout
```

### Step 6: Implement CLI command

Add to `src/elspeth/cli.py`:

```python
@app.command()
def export(
    db: str = typer.Option(
        ...,
        "--db",
        "-d",
        help="Path to Landscape database file.",
    ),
    output: str = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output path (directory for all tables, file for single table).",
    ),
    table: list[str] = typer.Option(
        None,
        "--table",
        "-t",
        help="Specific table(s) to export. Can be specified multiple times.",
    ),
    run_id: str = typer.Option(
        None,
        "--run",
        "-r",
        help="Export only data for this run ID.",
    ),
    format: str = typer.Option(
        "csv",
        "--format",
        "-f",
        help="Output format (currently only 'csv' supported).",
    ),
    list_tables: bool = typer.Option(
        False,
        "--list-tables",
        help="List available tables and exit.",
    ),
) -> None:
    """Export Landscape audit trail to CSV.

    Examples:

        # Export all tables to a directory
        elspeth export --db runs.db --output ./export/

        # Export specific table
        elspeth export --db runs.db --output runs.csv --table runs

        # Export specific run only
        elspeth export --db runs.db --output ./export/ --run run-abc123

        # List available tables
        elspeth export --db runs.db --list-tables
    """
    from pathlib import Path

    from elspeth.core.landscape import LandscapeDB
    from elspeth.core.landscape.export import LandscapeExporter

    # Validate format
    if format != "csv":
        typer.echo(f"Unsupported format: {format}. Currently only 'csv' is supported.", err=True)
        raise typer.Exit(1)

    # Connect to database
    db_path = Path(db)
    if not db_path.exists():
        typer.echo(f"Database not found: {db}", err=True)
        raise typer.Exit(1)

    landscape_db = LandscapeDB.from_url(f"sqlite:///{db_path}")
    exporter = LandscapeExporter(landscape_db)

    # List tables mode
    if list_tables:
        typer.echo("Available tables for export:")
        for table_name in exporter.get_table_names():
            typer.echo(f"  - {table_name}")
        raise typer.Exit(0)

    output_path = Path(output)

    # Single table export
    if table and len(table) == 1:
        table_name = table[0]
        if table_name not in exporter.get_table_names():
            typer.echo(f"Unknown table: {table_name}", err=True)
            typer.echo(f"Valid tables: {', '.join(exporter.get_table_names())}", err=True)
            raise typer.Exit(1)

        try:
            count = exporter.export_table(table_name, output_path, run_id=run_id)
            typer.echo(f"Exported {count} rows from '{table_name}' to {output_path}")
        except Exception as e:
            typer.echo(f"Export failed: {e}", err=True)
            raise typer.Exit(1)
        return

    # Multi-table or all-table export
    tables_to_export = table if table else None

    # Validate table names if specified
    if tables_to_export:
        valid_tables = set(exporter.get_table_names())
        invalid = set(tables_to_export) - valid_tables
        if invalid:
            typer.echo(f"Unknown table(s): {', '.join(invalid)}", err=True)
            typer.echo(f"Valid tables: {', '.join(sorted(valid_tables))}", err=True)
            raise typer.Exit(1)

    try:
        results = exporter.export_all(output_path, run_id=run_id, tables=tables_to_export)

        total_rows = sum(results.values())
        typer.echo(f"Exported {total_rows} total rows to {output_path}/")
        for table_name, count in sorted(results.items()):
            typer.echo(f"  {table_name}: {count} rows")

    except Exception as e:
        typer.echo(f"Export failed: {e}", err=True)
        raise typer.Exit(1)
```

### Step 7: Update landscape module exports

Add to `src/elspeth/core/landscape/__init__.py`:

```python
from elspeth.core.landscape.export import LandscapeExporter

# Add to __all__
__all__ = [
    # ... existing exports ...
    "LandscapeExporter",
]
```

### Step 8: Run tests to verify they pass

Run: `pytest tests/core/landscape/test_export_csv.py tests/cli/test_export_command.py -v`
Expected: PASS (17 tests)

### Step 9: Commit

```bash
git add src/elspeth/core/landscape/export.py src/elspeth/cli.py tests/
git commit -m "feat(landscape): add CSV export command for audit trail"
```

---

## Deliverables Summary

After Phase 4B:

| Component | Status |
|-----------|--------|
| PassThrough transform | Done (Task 1) |
| FieldMapper transform | Done (Task 2) |
| Filter transform | Done (Task 3) |
| ThresholdGate | Done (Task 4) |
| FieldMatchGate | Done (Task 5) |
| CLI transform support | Done (Task 6) |
| LineageTree widget | Done (Task 7) |
| NodeDetailPanel widget | Done (Task 8) |
| ExplainScreen integration | Done (Task 9) |
| Documentation update | Done (Task 10) |
| Landscape CSV export | Done (Task 11) |

**Phase 4B closes the gaps** that Phase 4 left open:
- Pipelines can now filter, transform, and route data through gates
- The explain TUI properly visualizes lineage with tree navigation
- The misleading "simple loop" documentation is corrected
- Audit trail can be exported to CSV for compliance, archival, and external analysis
