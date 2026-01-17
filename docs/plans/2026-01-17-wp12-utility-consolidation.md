# WP-12: Utility Consolidation

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extract duplicated utility code to a shared module, reducing copy-paste and ensuring consistent behavior.

**Architecture:** Create `src/elspeth/plugins/utils.py` containing:
1. `get_nested_field()` - Dot-notation field access with MISSING sentinel support
2. `DynamicSchema` - Factory/class for "accept any fields" schemas (replaces 3 identical classes)

**Tech Stack:** Python 3.12, Pydantic, typing

**Scope Note:** Gate files (`filter_gate.py`, `field_match_gate.py`, `threshold_gate.py`) also contain `_get_nested()` but will be deleted in WP-02. Only update `field_mapper.py`.

---

## Task 1: Create utils.py with get_nested_field()

**Files:**
- Create: `src/elspeth/plugins/utils.py`
- Test: `tests/plugins/test_utils.py`

**Step 1: Write the failing test**

Create `tests/plugins/test_utils.py`:

```python
"""Tests for plugin utilities."""

from typing import Any

import pytest


class TestGetNestedField:
    """Tests for get_nested_field utility."""

    def test_get_nested_field_exists(self) -> None:
        """get_nested_field can be imported."""
        from elspeth.plugins.utils import get_nested_field

        assert get_nested_field is not None

    def test_simple_field_access(self) -> None:
        """Access top-level field."""
        from elspeth.plugins.utils import get_nested_field

        data = {"name": "Alice", "age": 30}
        assert get_nested_field(data, "name") == "Alice"
        assert get_nested_field(data, "age") == 30

    def test_nested_field_access(self) -> None:
        """Access nested field with dot notation."""
        from elspeth.plugins.utils import get_nested_field

        data = {"user": {"name": "Bob", "profile": {"city": "NYC"}}}
        assert get_nested_field(data, "user.name") == "Bob"
        assert get_nested_field(data, "user.profile.city") == "NYC"

    def test_missing_field_returns_sentinel(self) -> None:
        """Missing field returns MISSING sentinel."""
        from elspeth.plugins.sentinels import MISSING
        from elspeth.plugins.utils import get_nested_field

        data = {"name": "Alice"}
        result = get_nested_field(data, "age")
        assert result is MISSING

    def test_missing_nested_field_returns_sentinel(self) -> None:
        """Missing nested field returns MISSING sentinel."""
        from elspeth.plugins.sentinels import MISSING
        from elspeth.plugins.utils import get_nested_field

        data = {"user": {"name": "Alice"}}
        result = get_nested_field(data, "user.email")
        assert result is MISSING

    def test_missing_intermediate_returns_sentinel(self) -> None:
        """Missing intermediate path returns MISSING sentinel."""
        from elspeth.plugins.sentinels import MISSING
        from elspeth.plugins.utils import get_nested_field

        data = {"user": {"name": "Alice"}}
        result = get_nested_field(data, "user.profile.city")
        assert result is MISSING

    def test_custom_default(self) -> None:
        """Custom default value for missing fields."""
        from elspeth.plugins.utils import get_nested_field

        data = {"name": "Alice"}
        result = get_nested_field(data, "age", default=0)
        assert result == 0

    def test_none_value_not_missing(self) -> None:
        """Explicit None is returned, not treated as missing."""
        from elspeth.plugins.sentinels import MISSING
        from elspeth.plugins.utils import get_nested_field

        data = {"value": None}
        result = get_nested_field(data, "value")
        assert result is None
        assert result is not MISSING

    def test_non_dict_intermediate_returns_sentinel(self) -> None:
        """Non-dict intermediate value returns MISSING."""
        from elspeth.plugins.sentinels import MISSING
        from elspeth.plugins.utils import get_nested_field

        data = {"user": "string_not_dict"}
        result = get_nested_field(data, "user.name")
        assert result is MISSING
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/plugins/test_utils.py::TestGetNestedField::test_get_nested_field_exists -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'elspeth.plugins.utils'`

**Step 3: Implement get_nested_field**

Create `src/elspeth/plugins/utils.py`:

```python
"""Shared utilities for the plugin system.

This module provides common functions used across multiple plugin types
to avoid code duplication and ensure consistent behavior.
"""

from typing import Any

from elspeth.plugins.sentinels import MISSING, MissingSentinel


def get_nested_field(
    data: dict[str, Any],
    path: str,
    default: Any = MISSING,
) -> Any:
    """Get value from nested dict using dot notation.

    Traverses a nested dictionary structure using a dot-separated path.
    Returns the MISSING sentinel (or custom default) if the path doesn't exist.

    Args:
        data: Source dictionary to traverse
        path: Dot-separated path (e.g., "user.profile.name")
        default: Value to return if path not found (default: MISSING sentinel)

    Returns:
        Value at path, or default if not found

    Examples:
        >>> data = {"user": {"name": "Alice", "age": 30}}
        >>> get_nested_field(data, "user.name")
        'Alice'
        >>> get_nested_field(data, "user.email")
        <MISSING>
        >>> get_nested_field(data, "user.email", default="unknown")
        'unknown'
    """
    parts = path.split(".")
    current: Any = data

    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]

    return current
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/plugins/test_utils.py::TestGetNestedField -v`

Expected: All 9 tests pass

**Step 5: Commit**

```
git add -A && git commit -m "feat(plugins): add get_nested_field utility

Extracts the common nested field access pattern to a shared utility.
Supports dot notation paths, MISSING sentinel, and custom defaults.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: Add DynamicSchema factory

**Files:**
- Modify: `src/elspeth/plugins/utils.py`
- Test: `tests/plugins/test_utils.py`

**Step 1: Write the failing test**

Add to `tests/plugins/test_utils.py`:

```python
class TestDynamicSchema:
    """Tests for DynamicSchema utility."""

    def test_dynamic_schema_exists(self) -> None:
        """DynamicSchema can be imported."""
        from elspeth.plugins.utils import DynamicSchema

        assert DynamicSchema is not None

    def test_dynamic_schema_accepts_any_fields(self) -> None:
        """DynamicSchema accepts any fields without validation errors."""
        from elspeth.plugins.utils import DynamicSchema

        # Should not raise
        instance = DynamicSchema(foo="bar", count=42, nested={"a": 1})

        assert instance.foo == "bar"
        assert instance.count == 42
        assert instance.nested == {"a": 1}

    def test_dynamic_schema_is_plugin_schema(self) -> None:
        """DynamicSchema is a valid PluginSchema."""
        from elspeth.contracts import PluginSchema
        from elspeth.plugins.utils import DynamicSchema

        assert issubclass(DynamicSchema, PluginSchema)

    def test_dynamic_schema_empty_is_valid(self) -> None:
        """DynamicSchema accepts empty initialization."""
        from elspeth.plugins.utils import DynamicSchema

        instance = DynamicSchema()
        assert instance is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/plugins/test_utils.py::TestDynamicSchema::test_dynamic_schema_exists -v`

Expected: FAIL with `ImportError: cannot import name 'DynamicSchema'`

**Step 3: Implement DynamicSchema**

Add to `src/elspeth/plugins/utils.py`:

```python
from elspeth.contracts import PluginSchema


class DynamicSchema(PluginSchema):
    """A PluginSchema that accepts any fields.

    Use this as input_schema for plugins that accept arbitrary row structures
    (e.g., sinks that write whatever they receive).

    This replaces the pattern of creating multiple identical classes:
        class CSVInputSchema(PluginSchema):
            model_config = {"extra": "allow"}

        class JSONInputSchema(PluginSchema):
            model_config = {"extra": "allow"}

    Now just use:
        input_schema = DynamicSchema

    Attributes are accessible directly on instances:
        schema = DynamicSchema(name="Alice", age=30)
        assert schema.name == "Alice"
    """

    model_config = {"extra": "allow"}  # noqa: RUF012 - Pydantic pattern
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/plugins/test_utils.py::TestDynamicSchema -v`

Expected: All 4 tests pass

**Step 5: Commit**

```
git add -A && git commit -m "feat(plugins): add DynamicSchema for arbitrary field acceptance

Provides a reusable PluginSchema subclass that accepts any fields.
Replaces the duplicated CSVInputSchema, JSONInputSchema, DatabaseInputSchema pattern.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: Update field_mapper.py to use get_nested_field

**Files:**
- Modify: `src/elspeth/plugins/transforms/field_mapper.py`
- Test: `tests/plugins/transforms/test_field_mapper.py`

**Step 1: Read current implementation**

The current `field_mapper.py` has `_get_nested()` as an instance method (lines 90-108).

**Step 2: Replace with import**

Change:

```python
# REMOVE this method from the class:
def _get_nested(self, data: dict[str, Any], path: str) -> Any:
    """Get value from nested dict using dot notation.
    ...
    """
    parts = path.split(".")
    current: Any = data

    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return MISSING
        current = current[part]

    return current
```

To:

```python
# ADD import at top of file:
from elspeth.plugins.utils import get_nested_field

# UPDATE call sites from self._get_nested() to get_nested_field()
```

**Step 3: Find and update call sites**

In `field_mapper.py`, find all uses of `self._get_nested(` and replace with `get_nested_field(`.

**Step 4: Run existing tests**

Run: `pytest tests/plugins/transforms/test_field_mapper.py -v`

Expected: All tests pass (behavior unchanged)

**Step 5: Commit**

```
git add -A && git commit -m "refactor(field_mapper): use shared get_nested_field utility

Removes duplicated _get_nested method in favor of shared utility.
Behavior is identical - this is a pure consolidation.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: Update sinks to use DynamicSchema

**Files:**
- Modify: `src/elspeth/plugins/sinks/csv_sink.py`
- Modify: `src/elspeth/plugins/sinks/json_sink.py`
- Modify: `src/elspeth/plugins/sinks/database_sink.py`

**Rationale:** Three identical schema classes violate DRY. Since we're consolidating utilities, we consolidate fully.

**Step 1: Update CSVSink**

In `csv_sink.py`, replace:

```python
class CSVInputSchema(PluginSchema):
    """Dynamic schema - accepts any row structure."""

    model_config = {"extra": "allow"}  # noqa: RUF012 - Pydantic pattern
```

With:

```python
from elspeth.plugins.utils import DynamicSchema

# Remove CSVInputSchema class entirely
# Update class attribute:
class CSVSink(BaseSink):
    ...
    input_schema = DynamicSchema
```

**Step 2: Update JSONSink**

Same pattern - remove `JSONInputSchema`, use `DynamicSchema`.

**Step 3: Update DatabaseSink**

Same pattern - remove `DatabaseInputSchema`, use `DynamicSchema`.

**Step 4: Run sink tests**

Run: `pytest tests/plugins/sinks/ -v`

Expected: All 41 tests pass

**Step 5: Commit**

```
git add -A && git commit -m "refactor(sinks): use shared DynamicSchema for input schemas

Removes three identical schema classes in favor of the shared DynamicSchema.
Reduces duplication and ensures consistent behavior.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 5: Run full verification

**Step 1: Run mypy**

```bash
mypy src/elspeth/plugins/utils.py src/elspeth/plugins/transforms/field_mapper.py --strict
```

Expected: No errors

**Step 2: Run all plugin tests**

```bash
pytest tests/plugins/ -v
```

Expected: All tests pass

**Step 3: Verify no duplicates remain**

```bash
# Should only find utils.py and the deleted gate files (if WP-02 not done yet)
grep -r "_get_nested\|def get_nested" src/elspeth/plugins/ --include="*.py"
```

Expected: Only `utils.py` contains `get_nested_field`

**Step 4: Final commit**

```
git add -A && git commit -m "chore: verify WP-12 utility consolidation complete

- get_nested_field extracted to utils.py
- DynamicSchema available for arbitrary-field schemas
- field_mapper.py updated to use shared utility
- All tests pass

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Verification Checklist

- [ ] `src/elspeth/plugins/utils.py` exists with `get_nested_field()` and `DynamicSchema`
- [ ] `get_nested_field()` has 9 passing tests
- [ ] `DynamicSchema` has 4 passing tests
- [ ] `field_mapper.py` imports from utils, no local `_get_nested`
- [ ] All sinks use `DynamicSchema` instead of local schema classes
- [ ] `mypy --strict` passes on utils.py and field_mapper.py
- [ ] All plugin tests pass

---

## Files Changed Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `src/elspeth/plugins/utils.py` | CREATE | New utility module |
| `tests/plugins/test_utils.py` | CREATE | Tests for utilities |
| `src/elspeth/plugins/transforms/field_mapper.py` | MODIFY | Use get_nested_field import |
| `src/elspeth/plugins/sinks/csv_sink.py` | MODIFY | Use DynamicSchema |
| `src/elspeth/plugins/sinks/json_sink.py` | MODIFY | Use DynamicSchema |
| `src/elspeth/plugins/sinks/database_sink.py` | MODIFY | Use DynamicSchema |

---

## Dependency Notes

- **Depends on:** Nothing (but recommended after WP-02 so gate files are already deleted)
- **Unlocks:** Nothing (pure cleanup)
- **Risk:** Low - pure refactoring with no behavior change
