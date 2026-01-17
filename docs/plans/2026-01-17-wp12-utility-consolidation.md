# WP-12: Utility Consolidation

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Extract `get_nested_field()` utility to shared module.

**Architecture:** Create `src/elspeth/plugins/utils.py` containing the `get_nested_field()` function currently duplicated in multiple files.

**Tech Stack:** Python 3.12, typing

---

## Key Architectural Decisions

| Decision | Rationale |
|----------|-----------|
| No coercion in utility | Utility handles pipeline data (elevated trust) - retrieval only |
| MISSING sentinel for absent fields | Distinguishes "field missing" from "field is None" |
| Default parameter support | Allows caller to specify fallback without sentinel checking |
| isinstance() check is legitimate | Traversing nested structures requires type checking |

---

## Scope

**In scope:**
- Extract `_get_nested()` from `field_mapper.py` to shared `utils.py`
- Rename to `get_nested_field()` for clearer public API

**Out of scope:**
- Schema consolidation (handled by WP-11.99's `create_schema_from_config()`)
- Gate files (`filter_gate.py`, `field_match_gate.py`, `threshold_gate.py`) contain `_get_nested()` but are deleted in WP-02

**Depends on:** WP-11.99 (establishes plugin module structure)
**Risk:** Low - pure refactoring with identical behavior

---

## Task 1: Create utils.py with get_nested_field()

**Files:**
- Create: `src/elspeth/plugins/utils.py`
- Test: `tests/plugins/test_utils.py`

**Step 1: Write the failing test**

Create `tests/plugins/test_utils.py`:

```python
"""Tests for plugin utilities."""

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

from elspeth.plugins.sentinels import MISSING


def get_nested_field(
    data: dict[str, Any],
    path: str,
    default: Any = MISSING,
) -> Any:
    """Get value from nested dict using dot notation.

    Traverses a nested dictionary structure using a dot-separated path.
    Returns the MISSING sentinel (or custom default) if the path doesn't exist.

    This function does NOT coerce values - it returns exactly what is found
    or the default if the path is missing. This is appropriate for pipeline
    data which has elevated trust (validated at source boundaries).

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
        >>> from elspeth.plugins.sentinels import MISSING
        >>> get_nested_field(data, "user.email") is MISSING
        True
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

```bash
git add src/elspeth/plugins/utils.py tests/plugins/test_utils.py
git commit -m "$(cat <<'EOF'
feat(plugins): add get_nested_field utility

Extracts the common nested field access pattern to a shared utility.
Supports dot notation paths, MISSING sentinel, and custom defaults.

Part of WP-12: Utility Consolidation

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Update field_mapper.py to use get_nested_field

**Files:**
- Modify: `src/elspeth/plugins/transforms/field_mapper.py`
- Test: `tests/plugins/transforms/test_field_mapper.py`

**Step 1: Read current implementation**

The current `field_mapper.py` has `_get_nested()` as an instance method (lines 115-133).

**Step 2: Add import and remove method**

At top of file, add:
```python
from elspeth.plugins.utils import get_nested_field
```

**Delete the entire `_get_nested()` method** (lines 115-133). Do not comment it out or rename it - per the no legacy code policy, old code is deleted completely.

**Step 3: Update call sites**

Find all uses of `self._get_nested(` and replace with `get_nested_field(`:

```python
# OLD:
value = self._get_nested(row, source)

# NEW:
value = get_nested_field(row, source)
```

**Step 4: Run existing tests**

Run: `pytest tests/plugins/transforms/test_field_mapper.py -v`

Expected: All tests pass (behavior unchanged)

**Step 5: Commit**

```bash
git add src/elspeth/plugins/transforms/field_mapper.py
git commit -m "$(cat <<'EOF'
refactor(field_mapper): use shared get_nested_field utility

Removes duplicated _get_nested method in favor of shared utility.
Behavior is identical - this is a pure consolidation.

Part of WP-12: Utility Consolidation

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Final Verification

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

**Step 3: Verify no duplicates remain in field_mapper**

```bash
grep -n "_get_nested" src/elspeth/plugins/transforms/field_mapper.py
```

Expected: No results (method deleted)

**Note:** Gate files (`filter_gate.py`, `field_match_gate.py`, `threshold_gate.py`) still contain `_get_nested()` - they are deleted in WP-02, not updated here.

---

## Verification Checklist

- [ ] `src/elspeth/plugins/utils.py` exists with `get_nested_field()`
- [ ] `get_nested_field()` has 9 passing tests
- [ ] `field_mapper.py` imports from utils, no local `_get_nested`
- [ ] `mypy --strict` passes on utils.py and field_mapper.py
- [ ] All plugin tests pass

---

## Files Changed Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `src/elspeth/plugins/utils.py` | CREATE | New utility module with `get_nested_field()` |
| `tests/plugins/test_utils.py` | CREATE | Tests for utilities |
| `src/elspeth/plugins/transforms/field_mapper.py` | MODIFY | Use shared utility, delete `_get_nested()` |

---

## Dependency Notes

- **Depends on:** WP-11.99 (establishes plugin module patterns)
- **Unlocks:** Nothing (pure cleanup)
- **Risk:** Low - pure refactoring with no behavior change
- **Estimated Effort:** 30 minutes
