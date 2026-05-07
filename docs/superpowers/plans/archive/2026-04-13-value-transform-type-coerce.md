# Value Transform and Type Coerce Plugins Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement two deterministic transform plugins: `type_coerce` for explicit type normalization and `value_transform` for expression-based field computation.

**Architecture:** Both plugins follow the same pattern as existing transforms (`truncate`, `field_mapper`): Pydantic config class, `BaseTransform` subclass with `process()` method, atomic row semantics (any failure returns original row to error sink). `value_transform` reuses the existing `ExpressionParser` with no modifications.

**Tech Stack:** Python, Pydantic, pytest, existing ELSPETH infrastructure (`BaseTransform`, `TransformDataConfig`, `TransformResult`, `ExpressionParser`)

**Spec:** `docs/superpowers/specs/2026-04-13-value-transform-type-coerce-design.md`

---

## File Structure

```
src/elspeth/plugins/transforms/
├── type_coerce.py              # New: TypeCoerceConfig + TypeCoerce plugin
└── value_transform.py          # New: ValueTransformConfig + ValueTransform plugin

tests/unit/plugins/transforms/
├── test_type_coerce.py         # New: Config validation + conversion behavior
└── test_value_transform.py     # New: Config validation + expression behavior
```

---

## Task 1: Type Coerce — Conversion Functions

**Files:**
- Create: `src/elspeth/plugins/transforms/type_coerce.py`
- Test: `tests/unit/plugins/transforms/test_type_coerce.py`

The conversion logic is the core complexity. We'll implement standalone conversion functions first, then wrap them in the plugin class.

- [ ] **Step 1: Create test file with int conversion tests**

```python
"""Tests for TypeCoerce transform — behavioral unit tests."""

import math
import pytest


class TestCoerceToInt:
    """Test coerce_to_int conversion function."""

    def test_int_unchanged(self) -> None:
        from elspeth.plugins.transforms.type_coerce import coerce_to_int
        assert coerce_to_int(42) == 42
        assert coerce_to_int(-7) == -7
        assert coerce_to_int(0) == 0

    def test_float_no_fractional_part(self) -> None:
        from elspeth.plugins.transforms.type_coerce import coerce_to_int
        assert coerce_to_int(3.0) == 3
        assert coerce_to_int(-5.0) == -5
        assert coerce_to_int(0.0) == 0

    def test_float_with_fractional_part_errors(self) -> None:
        from elspeth.plugins.transforms.type_coerce import coerce_to_int, CoercionError
        with pytest.raises(CoercionError, match="fractional"):
            coerce_to_int(3.9)
        with pytest.raises(CoercionError, match="fractional"):
            coerce_to_int(3.1)
        with pytest.raises(CoercionError, match="fractional"):
            coerce_to_int(-2.5)

    def test_string_integer(self) -> None:
        from elspeth.plugins.transforms.type_coerce import coerce_to_int
        assert coerce_to_int("42") == 42
        assert coerce_to_int("-7") == -7
        assert coerce_to_int("+42") == 42
        assert coerce_to_int("0") == 0

    def test_string_with_whitespace(self) -> None:
        from elspeth.plugins.transforms.type_coerce import coerce_to_int
        assert coerce_to_int(" 42 ") == 42
        assert coerce_to_int("  -7  ") == -7

    def test_string_decimal_errors(self) -> None:
        from elspeth.plugins.transforms.type_coerce import coerce_to_int, CoercionError
        with pytest.raises(CoercionError, match="not a valid integer"):
            coerce_to_int("3.5")
        with pytest.raises(CoercionError, match="not a valid integer"):
            coerce_to_int("3.0")  # String "3.0" is not valid int

    def test_string_scientific_notation_errors(self) -> None:
        from elspeth.plugins.transforms.type_coerce import coerce_to_int, CoercionError
        with pytest.raises(CoercionError, match="not a valid integer"):
            coerce_to_int("1e3")

    def test_empty_string_errors(self) -> None:
        from elspeth.plugins.transforms.type_coerce import coerce_to_int, CoercionError
        with pytest.raises(CoercionError, match="empty"):
            coerce_to_int("")
        with pytest.raises(CoercionError, match="empty"):
            coerce_to_int("   ")

    def test_bool_errors(self) -> None:
        from elspeth.plugins.transforms.type_coerce import coerce_to_int, CoercionError
        with pytest.raises(CoercionError, match="bool"):
            coerce_to_int(True)
        with pytest.raises(CoercionError, match="bool"):
            coerce_to_int(False)

    def test_none_errors(self) -> None:
        from elspeth.plugins.transforms.type_coerce import coerce_to_int, CoercionError
        with pytest.raises(CoercionError, match="None"):
            coerce_to_int(None)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/plugins/transforms/test_type_coerce.py::TestCoerceToInt -v`
Expected: FAIL with "cannot import name 'coerce_to_int'"

- [ ] **Step 3: Implement coerce_to_int**

```python
"""TypeCoerce transform plugin.

Performs explicit, strict, per-field type normalization.

IMPORTANT: Transforms use allow_coercion=False to catch upstream bugs.
If the source outputs wrong types, the transform crashes immediately.
"""

from __future__ import annotations

import math
from typing import Any


class CoercionError(Exception):
    """Raised when type coercion fails."""

    def __init__(self, value: Any, target_type: str, reason: str) -> None:
        self.value = value
        self.target_type = target_type
        self.reason = reason
        super().__init__(f"Cannot coerce {type(value).__name__} to {target_type}: {reason}")


def coerce_to_int(value: Any) -> int:
    """Coerce value to int with strict rules.

    Accepts:
        - int (unchanged)
        - float with no fractional part (3.0 -> 3)
        - string of integer after trim ("42", " -7 ")

    Rejects:
        - float with fractional part (3.9 -> error)
        - string with decimal ("3.5" -> error)
        - scientific notation string ("1e3" -> error)
        - empty/whitespace string
        - bool (True/False are technically ints but rejected)
        - None
    """
    # Reject None first
    if value is None:
        raise CoercionError(value, "int", "None cannot be converted to int")

    # Reject bool explicitly (before int check, since bool is subclass of int)
    if type(value) is bool:
        raise CoercionError(value, "int", "bool cannot be converted to int")

    # int passes through
    if type(value) is int:
        return value

    # float: only if no fractional part
    if type(value) is float:
        if not math.isfinite(value):
            raise CoercionError(value, "int", "non-finite float cannot be converted to int")
        if value != int(value):
            raise CoercionError(value, "int", f"float {value} has fractional part")
        return int(value)

    # string: parse as integer
    if type(value) is str:
        trimmed = value.strip()
        if not trimmed:
            raise CoercionError(value, "int", "empty string cannot be converted to int")
        try:
            return int(trimmed)
        except ValueError:
            raise CoercionError(value, "int", f"'{trimmed}' is not a valid integer string")

    raise CoercionError(value, "int", f"unsupported type {type(value).__name__}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/plugins/transforms/test_type_coerce.py::TestCoerceToInt -v`
Expected: All PASS

- [ ] **Step 5: Add float conversion tests**

Add to `tests/unit/plugins/transforms/test_type_coerce.py`:

```python
class TestCoerceToFloat:
    """Test coerce_to_float conversion function."""

    def test_float_unchanged(self) -> None:
        from elspeth.plugins.transforms.type_coerce import coerce_to_float
        assert coerce_to_float(3.14) == 3.14
        assert coerce_to_float(-2.5) == -2.5
        assert coerce_to_float(0.0) == 0.0

    def test_int_to_float(self) -> None:
        from elspeth.plugins.transforms.type_coerce import coerce_to_float
        assert coerce_to_float(42) == 42.0
        assert coerce_to_float(-7) == -7.0
        assert coerce_to_float(0) == 0.0

    def test_string_numeric(self) -> None:
        from elspeth.plugins.transforms.type_coerce import coerce_to_float
        assert coerce_to_float("12.5") == 12.5
        assert coerce_to_float("-3.14") == -3.14
        assert coerce_to_float("+2.5") == 2.5
        assert coerce_to_float("42") == 42.0

    def test_string_with_whitespace(self) -> None:
        from elspeth.plugins.transforms.type_coerce import coerce_to_float
        assert coerce_to_float(" 12.5 ") == 12.5
        assert coerce_to_float("  -3.14  ") == -3.14

    def test_scientific_notation(self) -> None:
        from elspeth.plugins.transforms.type_coerce import coerce_to_float
        assert coerce_to_float("1e3") == 1000.0
        assert coerce_to_float("2.5e-4") == 0.00025

    def test_nan_inf_rejected(self) -> None:
        from elspeth.plugins.transforms.type_coerce import coerce_to_float, CoercionError
        with pytest.raises(CoercionError, match="non-finite"):
            coerce_to_float("nan")
        with pytest.raises(CoercionError, match="non-finite"):
            coerce_to_float("inf")
        with pytest.raises(CoercionError, match="non-finite"):
            coerce_to_float("-inf")
        with pytest.raises(CoercionError, match="non-finite"):
            coerce_to_float(float("nan"))
        with pytest.raises(CoercionError, match="non-finite"):
            coerce_to_float(float("inf"))

    def test_empty_string_errors(self) -> None:
        from elspeth.plugins.transforms.type_coerce import coerce_to_float, CoercionError
        with pytest.raises(CoercionError, match="empty"):
            coerce_to_float("")
        with pytest.raises(CoercionError, match="empty"):
            coerce_to_float("   ")

    def test_bool_errors(self) -> None:
        from elspeth.plugins.transforms.type_coerce import coerce_to_float, CoercionError
        with pytest.raises(CoercionError, match="bool"):
            coerce_to_float(True)

    def test_none_errors(self) -> None:
        from elspeth.plugins.transforms.type_coerce import coerce_to_float, CoercionError
        with pytest.raises(CoercionError, match="None"):
            coerce_to_float(None)
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `pytest tests/unit/plugins/transforms/test_type_coerce.py::TestCoerceToFloat -v`
Expected: FAIL with "cannot import name 'coerce_to_float'"

- [ ] **Step 7: Implement coerce_to_float**

Add to `src/elspeth/plugins/transforms/type_coerce.py`:

```python
def coerce_to_float(value: Any) -> float:
    """Coerce value to float with strict rules.

    Accepts:
        - float (unchanged, must be finite)
        - int -> float
        - numeric string after trim ("12.5", "1e3")

    Rejects:
        - non-finite floats (NaN, inf, -inf)
        - empty/whitespace string
        - bool
        - None
    """
    # Reject None first
    if value is None:
        raise CoercionError(value, "float", "None cannot be converted to float")

    # Reject bool explicitly
    if type(value) is bool:
        raise CoercionError(value, "float", "bool cannot be converted to float")

    # float: check finite
    if type(value) is float:
        if not math.isfinite(value):
            raise CoercionError(value, "float", "non-finite float values are not allowed")
        return value

    # int -> float
    if type(value) is int:
        return float(value)

    # string: parse as float
    if type(value) is str:
        trimmed = value.strip()
        if not trimmed:
            raise CoercionError(value, "float", "empty string cannot be converted to float")
        try:
            result = float(trimmed)
        except ValueError:
            raise CoercionError(value, "float", f"'{trimmed}' is not a valid numeric string")
        if not math.isfinite(result):
            raise CoercionError(value, "float", f"'{trimmed}' produces non-finite value")
        return result

    raise CoercionError(value, "float", f"unsupported type {type(value).__name__}")
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `pytest tests/unit/plugins/transforms/test_type_coerce.py::TestCoerceToFloat -v`
Expected: All PASS

- [ ] **Step 9: Add bool conversion tests**

Add to `tests/unit/plugins/transforms/test_type_coerce.py`:

```python
class TestCoerceToBool:
    """Test coerce_to_bool conversion function."""

    def test_bool_unchanged(self) -> None:
        from elspeth.plugins.transforms.type_coerce import coerce_to_bool
        assert coerce_to_bool(True) is True
        assert coerce_to_bool(False) is False

    def test_int_zero_one(self) -> None:
        from elspeth.plugins.transforms.type_coerce import coerce_to_bool
        assert coerce_to_bool(0) is False
        assert coerce_to_bool(1) is True

    def test_int_other_errors(self) -> None:
        from elspeth.plugins.transforms.type_coerce import coerce_to_bool, CoercionError
        with pytest.raises(CoercionError, match="only 0 and 1"):
            coerce_to_bool(2)
        with pytest.raises(CoercionError, match="only 0 and 1"):
            coerce_to_bool(-1)

    def test_string_true_set(self) -> None:
        from elspeth.plugins.transforms.type_coerce import coerce_to_bool
        for val in ["true", "TRUE", "True", "1", "yes", "YES", "y", "Y", "on", "ON"]:
            assert coerce_to_bool(val) is True, f"Expected True for {val!r}"

    def test_string_false_set(self) -> None:
        from elspeth.plugins.transforms.type_coerce import coerce_to_bool
        for val in ["false", "FALSE", "False", "0", "no", "NO", "n", "N", "off", "OFF", ""]:
            assert coerce_to_bool(val) is False, f"Expected False for {val!r}"

    def test_string_with_whitespace(self) -> None:
        from elspeth.plugins.transforms.type_coerce import coerce_to_bool
        assert coerce_to_bool(" true ") is True
        assert coerce_to_bool("  false  ") is False
        assert coerce_to_bool("  ") is False  # whitespace-only = empty = false

    def test_string_invalid_errors(self) -> None:
        from elspeth.plugins.transforms.type_coerce import coerce_to_bool, CoercionError
        with pytest.raises(CoercionError, match="not a valid boolean"):
            coerce_to_bool("maybe")
        with pytest.raises(CoercionError, match="not a valid boolean"):
            coerce_to_bool("oui")

    def test_float_errors(self) -> None:
        from elspeth.plugins.transforms.type_coerce import coerce_to_bool, CoercionError
        with pytest.raises(CoercionError, match="float"):
            coerce_to_bool(1.0)

    def test_none_errors(self) -> None:
        from elspeth.plugins.transforms.type_coerce import coerce_to_bool, CoercionError
        with pytest.raises(CoercionError, match="None"):
            coerce_to_bool(None)
```

- [ ] **Step 10: Run tests to verify they fail**

Run: `pytest tests/unit/plugins/transforms/test_type_coerce.py::TestCoerceToBool -v`
Expected: FAIL with "cannot import name 'coerce_to_bool'"

- [ ] **Step 11: Implement coerce_to_bool**

Add to `src/elspeth/plugins/transforms/type_coerce.py`:

```python
# Boolean string mappings (case-insensitive after trim)
_BOOL_TRUE_STRINGS: frozenset[str] = frozenset({"true", "1", "yes", "y", "on"})
_BOOL_FALSE_STRINGS: frozenset[str] = frozenset({"false", "0", "no", "n", "off", ""})


def coerce_to_bool(value: Any) -> bool:
    """Coerce value to bool with strict rules.

    Accepts:
        - bool (unchanged)
        - int 0 -> False, int 1 -> True
        - string true set (case-insensitive): true, 1, yes, y, on
        - string false set (case-insensitive): false, 0, no, n, off, ""

    Rejects:
        - other integers (2, -1, etc.)
        - other strings
        - float
        - None
    """
    # Reject None first
    if value is None:
        raise CoercionError(value, "bool", "None cannot be converted to bool")

    # bool passes through
    if type(value) is bool:
        return value

    # int: only 0 and 1
    if type(value) is int:
        if value == 0:
            return False
        if value == 1:
            return True
        raise CoercionError(value, "bool", f"only 0 and 1 can be converted to bool, got {value}")

    # float: reject
    if type(value) is float:
        raise CoercionError(value, "bool", "float cannot be converted to bool")

    # string: check against true/false sets
    if type(value) is str:
        normalized = value.strip().lower()
        if normalized in _BOOL_TRUE_STRINGS:
            return True
        if normalized in _BOOL_FALSE_STRINGS:
            return False
        raise CoercionError(value, "bool", f"'{value}' is not a valid boolean string")

    raise CoercionError(value, "bool", f"unsupported type {type(value).__name__}")
```

- [ ] **Step 12: Run tests to verify they pass**

Run: `pytest tests/unit/plugins/transforms/test_type_coerce.py::TestCoerceToBool -v`
Expected: All PASS

- [ ] **Step 13: Add str conversion tests**

Add to `tests/unit/plugins/transforms/test_type_coerce.py`:

```python
class TestCoerceToStr:
    """Test coerce_to_str conversion function."""

    def test_str_unchanged(self) -> None:
        from elspeth.plugins.transforms.type_coerce import coerce_to_str
        assert coerce_to_str("hello") == "hello"
        assert coerce_to_str("") == ""

    def test_int_to_str(self) -> None:
        from elspeth.plugins.transforms.type_coerce import coerce_to_str
        assert coerce_to_str(42) == "42"
        assert coerce_to_str(-7) == "-7"
        assert coerce_to_str(0) == "0"

    def test_float_to_str(self) -> None:
        from elspeth.plugins.transforms.type_coerce import coerce_to_str
        assert coerce_to_str(3.14) == "3.14"
        assert coerce_to_str(-2.5) == "-2.5"

    def test_bool_to_str(self) -> None:
        from elspeth.plugins.transforms.type_coerce import coerce_to_str
        assert coerce_to_str(True) == "True"
        assert coerce_to_str(False) == "False"

    def test_list_errors(self) -> None:
        from elspeth.plugins.transforms.type_coerce import coerce_to_str, CoercionError
        with pytest.raises(CoercionError, match="not a scalar"):
            coerce_to_str([1, 2, 3])

    def test_dict_errors(self) -> None:
        from elspeth.plugins.transforms.type_coerce import coerce_to_str, CoercionError
        with pytest.raises(CoercionError, match="not a scalar"):
            coerce_to_str({"a": 1})

    def test_none_errors(self) -> None:
        from elspeth.plugins.transforms.type_coerce import coerce_to_str, CoercionError
        with pytest.raises(CoercionError, match="None"):
            coerce_to_str(None)
```

- [ ] **Step 14: Run tests to verify they fail**

Run: `pytest tests/unit/plugins/transforms/test_type_coerce.py::TestCoerceToStr -v`
Expected: FAIL with "cannot import name 'coerce_to_str'"

- [ ] **Step 15: Implement coerce_to_str**

Add to `src/elspeth/plugins/transforms/type_coerce.py`:

```python
# Scalar types accepted for string conversion
_SCALAR_TYPES: tuple[type, ...] = (str, int, float, bool)


def coerce_to_str(value: Any) -> str:
    """Coerce value to str with strict rules.

    Accepts:
        - str (unchanged)
        - int, float, bool -> Python str()

    Rejects:
        - list, dict, objects, bytes (not scalars)
        - None
    """
    # Reject None first
    if value is None:
        raise CoercionError(value, "str", "None cannot be converted to str")

    # Only accept scalar types
    if not isinstance(value, _SCALAR_TYPES):
        raise CoercionError(value, "str", f"{type(value).__name__} is not a scalar type")

    return str(value)
```

- [ ] **Step 16: Run tests to verify they pass**

Run: `pytest tests/unit/plugins/transforms/test_type_coerce.py::TestCoerceToStr -v`
Expected: All PASS

- [ ] **Step 17: Commit conversion functions**

```bash
git add src/elspeth/plugins/transforms/type_coerce.py tests/unit/plugins/transforms/test_type_coerce.py
git commit -m "feat(type_coerce): add strict type conversion functions

Implements coerce_to_int, coerce_to_float, coerce_to_bool, coerce_to_str
with explicit rejection of edge cases (NaN/inf, Python truthiness, etc.)

Part of value_transform/type_coerce plugin implementation."
```

---

## Task 2: Type Coerce — Config Class

**Files:**
- Modify: `src/elspeth/plugins/transforms/type_coerce.py`
- Modify: `tests/unit/plugins/transforms/test_type_coerce.py`

- [ ] **Step 1: Add config validation tests**

Add to `tests/unit/plugins/transforms/test_type_coerce.py`:

```python
from pydantic import ValidationError

from elspeth.contracts.schema import SchemaConfig

OBSERVED_SCHEMA_CONFIG = SchemaConfig.from_dict({"mode": "observed"})


class TestTypeCoerceConfig:
    """Pydantic config validation for TypeCoerceConfig."""

    def test_valid_config(self) -> None:
        from elspeth.plugins.transforms.type_coerce import TypeCoerceConfig
        cfg = TypeCoerceConfig(
            conversions=[
                {"field": "price", "to": "float"},
                {"field": "quantity", "to": "int"},
            ],
            schema_config=OBSERVED_SCHEMA_CONFIG,
        )
        assert len(cfg.conversions) == 2
        assert cfg.conversions[0].field == "price"
        assert cfg.conversions[0].to == "float"

    def test_rejects_empty_conversions(self) -> None:
        from elspeth.plugins.transforms.type_coerce import TypeCoerceConfig
        with pytest.raises(ValidationError, match="at least one"):
            TypeCoerceConfig(conversions=[], schema_config=OBSERVED_SCHEMA_CONFIG)

    def test_rejects_invalid_target_type(self) -> None:
        from elspeth.plugins.transforms.type_coerce import TypeCoerceConfig
        with pytest.raises(ValidationError, match="invalid target type"):
            TypeCoerceConfig(
                conversions=[{"field": "x", "to": "datetime"}],
                schema_config=OBSERVED_SCHEMA_CONFIG,
            )

    def test_rejects_empty_field_name(self) -> None:
        from elspeth.plugins.transforms.type_coerce import TypeCoerceConfig
        with pytest.raises(ValidationError, match="field name"):
            TypeCoerceConfig(
                conversions=[{"field": "", "to": "int"}],
                schema_config=OBSERVED_SCHEMA_CONFIG,
            )

    def test_from_dict_factory(self) -> None:
        from elspeth.plugins.transforms.type_coerce import TypeCoerceConfig
        cfg = TypeCoerceConfig.from_dict({
            "schema": {"mode": "observed"},
            "conversions": [{"field": "x", "to": "int"}],
        })
        assert cfg.conversions[0].field == "x"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/plugins/transforms/test_type_coerce.py::TestTypeCoerceConfig -v`
Expected: FAIL with "cannot import name 'TypeCoerceConfig'"

- [ ] **Step 3: Implement TypeCoerceConfig**

Add to `src/elspeth/plugins/transforms/type_coerce.py` (after the conversion functions):

```python
from typing import Literal

from pydantic import Field, field_validator, model_validator

from elspeth.plugins.infrastructure.config_base import TransformDataConfig


class ConversionSpec(BaseModel):
    """Single field conversion specification."""

    model_config = {"extra": "forbid", "frozen": True}

    field: str
    to: Literal["int", "float", "bool", "str"]

    @field_validator("field")
    @classmethod
    def _validate_field_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("field name must not be empty")
        return v

    @field_validator("to")
    @classmethod
    def _validate_target_type(cls, v: str) -> str:
        valid_types = {"int", "float", "bool", "str"}
        if v not in valid_types:
            raise ValueError(f"invalid target type '{v}', must be one of {valid_types}")
        return v


class TypeCoerceConfig(TransformDataConfig):
    """Configuration for type coercion transform.

    Requires 'schema' in config to define input/output expectations.
    Use 'schema: {mode: observed}' for dynamic field handling.
    """

    conversions: list[ConversionSpec] = Field(
        ...,
        description="List of field type conversions to apply",
    )

    @model_validator(mode="after")
    def _validate_conversions_not_empty(self) -> "TypeCoerceConfig":
        if not self.conversions:
            raise ValueError("conversions must contain at least one conversion")
        return self
```

Also add the import at the top:

```python
from pydantic import BaseModel, Field, field_validator, model_validator
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/plugins/transforms/test_type_coerce.py::TestTypeCoerceConfig -v`
Expected: All PASS

- [ ] **Step 5: Commit config class**

```bash
git add src/elspeth/plugins/transforms/type_coerce.py tests/unit/plugins/transforms/test_type_coerce.py
git commit -m "feat(type_coerce): add TypeCoerceConfig with validation

Pydantic config class with ConversionSpec for field/to pairs.
Validates non-empty conversions and valid target types."
```

---

## Task 3: Type Coerce — Plugin Class

**Files:**
- Modify: `src/elspeth/plugins/transforms/type_coerce.py`
- Modify: `tests/unit/plugins/transforms/test_type_coerce.py`

- [ ] **Step 1: Add plugin behavior tests**

Add to `tests/unit/plugins/transforms/test_type_coerce.py`:

```python
from elspeth.contracts.plugin_context import PluginContext
from elspeth.testing import make_pipeline_row
from tests.fixtures.factories import make_source_context

DYNAMIC_SCHEMA = {"mode": "observed"}


class TestTypeCoerceBehavior:
    """Core type coercion mechanics."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        return make_source_context()

    def test_coerces_string_to_int(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.type_coerce import TypeCoerce
        transform = TypeCoerce({
            "schema": DYNAMIC_SCHEMA,
            "conversions": [{"field": "quantity", "to": "int"}],
        })
        row = make_pipeline_row({"quantity": "42"})
        result = transform.process(row, ctx)
        assert result.status == "success"
        assert result.row is not None
        assert result.row["quantity"] == 42
        assert type(result.row["quantity"]) is int

    def test_coerces_string_to_float(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.type_coerce import TypeCoerce
        transform = TypeCoerce({
            "schema": DYNAMIC_SCHEMA,
            "conversions": [{"field": "price", "to": "float"}],
        })
        row = make_pipeline_row({"price": " 12.50 "})
        result = transform.process(row, ctx)
        assert result.status == "success"
        assert result.row is not None
        assert result.row["price"] == 12.5

    def test_coerces_string_to_bool(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.type_coerce import TypeCoerce
        transform = TypeCoerce({
            "schema": DYNAMIC_SCHEMA,
            "conversions": [{"field": "active", "to": "bool"}],
        })
        row = make_pipeline_row({"active": "false"})
        result = transform.process(row, ctx)
        assert result.status == "success"
        assert result.row is not None
        assert result.row["active"] is False

    def test_coerces_int_to_str(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.type_coerce import TypeCoerce
        transform = TypeCoerce({
            "schema": DYNAMIC_SCHEMA,
            "conversions": [{"field": "user_id", "to": "str"}],
        })
        row = make_pipeline_row({"user_id": 42})
        result = transform.process(row, ctx)
        assert result.status == "success"
        assert result.row is not None
        assert result.row["user_id"] == "42"

    def test_multiple_conversions(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.type_coerce import TypeCoerce
        transform = TypeCoerce({
            "schema": DYNAMIC_SCHEMA,
            "conversions": [
                {"field": "price", "to": "float"},
                {"field": "quantity", "to": "int"},
                {"field": "active", "to": "bool"},
            ],
        })
        row = make_pipeline_row({"price": "12.50", "quantity": "3", "active": "yes"})
        result = transform.process(row, ctx)
        assert result.status == "success"
        assert result.row is not None
        assert result.row["price"] == 12.5
        assert result.row["quantity"] == 3
        assert result.row["active"] is True

    def test_already_correct_type_unchanged(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.type_coerce import TypeCoerce
        transform = TypeCoerce({
            "schema": DYNAMIC_SCHEMA,
            "conversions": [{"field": "quantity", "to": "int"}],
        })
        row = make_pipeline_row({"quantity": 42})
        result = transform.process(row, ctx)
        assert result.status == "success"
        assert result.row is not None
        assert result.row["quantity"] == 42
        # Check audit shows unchanged
        assert "quantity" in result.success_reason.get("fields_unchanged", [])

    def test_missing_field_errors(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.type_coerce import TypeCoerce
        transform = TypeCoerce({
            "schema": DYNAMIC_SCHEMA,
            "conversions": [{"field": "missing", "to": "int"}],
        })
        row = make_pipeline_row({"other": 42})
        result = transform.process(row, ctx)
        assert result.status == "error"
        assert "missing" in result.error_reason.get("reason", "")

    def test_conversion_failure_errors(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.type_coerce import TypeCoerce
        transform = TypeCoerce({
            "schema": DYNAMIC_SCHEMA,
            "conversions": [{"field": "active", "to": "bool"}],
        })
        row = make_pipeline_row({"active": "maybe"})
        result = transform.process(row, ctx)
        assert result.status == "error"
        assert "maybe" in result.error_reason.get("reason", "")

    def test_atomic_failure_no_partial_mutation(self, ctx: PluginContext) -> None:
        """If second conversion fails, first conversion should not be applied."""
        from elspeth.plugins.transforms.type_coerce import TypeCoerce
        transform = TypeCoerce({
            "schema": DYNAMIC_SCHEMA,
            "conversions": [
                {"field": "price", "to": "float"},  # Would succeed
                {"field": "active", "to": "bool"},  # Will fail
            ],
        })
        row = make_pipeline_row({"price": "12.50", "active": "maybe"})
        result = transform.process(row, ctx)
        assert result.status == "error"
        # Original row should be unchanged (error path gets original row)

    def test_audit_trail_fields_coerced(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.type_coerce import TypeCoerce
        transform = TypeCoerce({
            "schema": DYNAMIC_SCHEMA,
            "conversions": [
                {"field": "price", "to": "float"},
                {"field": "quantity", "to": "int"},
            ],
        })
        row = make_pipeline_row({"price": "12.50", "quantity": 3})  # quantity already int
        result = transform.process(row, ctx)
        assert result.status == "success"
        assert result.success_reason["action"] == "coerced"
        assert result.success_reason["fields_coerced"] == ["price"]
        assert result.success_reason["fields_unchanged"] == ["quantity"]
        assert result.success_reason["rules_evaluated"] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/plugins/transforms/test_type_coerce.py::TestTypeCoerceBehavior -v`
Expected: FAIL with "cannot import name 'TypeCoerce'"

- [ ] **Step 3: Implement TypeCoerce plugin class**

Add to `src/elspeth/plugins/transforms/type_coerce.py`:

```python
import copy
from typing import Any, Callable

from elspeth.contracts.contexts import TransformContext
from elspeth.contracts.schema_contract import PipelineRow
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.results import TransformResult


# Conversion function dispatch table
_COERCION_FUNCS: dict[str, Callable[[Any], Any]] = {
    "int": coerce_to_int,
    "float": coerce_to_float,
    "bool": coerce_to_bool,
    "str": coerce_to_str,
}

# Target type checks for idempotency
_TARGET_TYPES: dict[str, type] = {
    "int": int,
    "float": float,
    "bool": bool,
    "str": str,
}


class TypeCoerce(BaseTransform):
    """Perform explicit, strict, per-field type normalization.

    Conversions are evaluated in order on a working copy of the row.
    If all conversions succeed, the updated row is emitted.
    If any conversion fails, the original row is returned as an error
    and no partial changes are emitted on the success path.

    Config options:
        schema: Required. Schema for input/output (use {mode: observed} for any fields)
        conversions: List of {field, to} specs defining type conversions
    """

    name = "type_coerce"
    plugin_version = "1.0.0"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = TypeCoerceConfig.from_dict(config)
        self._conversions = cfg.conversions
        self._schema_config = cfg.schema_config

        self.input_schema, self.output_schema = self._create_schemas(
            cfg.schema_config,
            "TypeCoerce",
        )

    def process(self, row: PipelineRow, ctx: TransformContext) -> TransformResult:
        """Apply type conversions to row fields.

        Args:
            row: Input row data
            ctx: Plugin context

        Returns:
            TransformResult with converted field values, or error if any conversion fails
        """
        # Work on a copy to support atomic rollback
        output = copy.deepcopy(row.to_dict())
        fields_coerced: list[str] = []
        fields_unchanged: list[str] = []

        for spec in self._conversions:
            field_name = spec.field
            target_type_name = spec.to

            # Check field exists
            if field_name not in row:
                return TransformResult.error({
                    "action": "error",
                    "plugin": "type_coerce",
                    "field": field_name,
                    "target_type": target_type_name,
                    "reason": f"Field '{field_name}' not found in row",
                })

            value = row[field_name]

            # Check for None
            if value is None:
                return TransformResult.error({
                    "action": "error",
                    "plugin": "type_coerce",
                    "field": field_name,
                    "target_type": target_type_name,
                    "reason": f"Field '{field_name}' is None",
                })

            # Check if already correct type (idempotent)
            target_type = _TARGET_TYPES[target_type_name]
            # Use type() not isinstance() to avoid bool matching int
            if type(value) is target_type:
                fields_unchanged.append(field_name)
                continue

            # Apply conversion
            coerce_func = _COERCION_FUNCS[target_type_name]
            try:
                converted = coerce_func(value)
            except CoercionError as e:
                return TransformResult.error({
                    "action": "error",
                    "plugin": "type_coerce",
                    "field": field_name,
                    "target_type": target_type_name,
                    "reason": e.reason,
                })

            output[field_name] = converted
            fields_coerced.append(field_name)

        return TransformResult.success(
            PipelineRow(output, row.contract),
            success_reason={
                "action": "coerced",
                "fields_coerced": fields_coerced,
                "fields_unchanged": fields_unchanged,
                "rules_evaluated": len(self._conversions),
            },
        )

    def close(self) -> None:
        """No resources to release."""
        pass
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/plugins/transforms/test_type_coerce.py::TestTypeCoerceBehavior -v`
Expected: All PASS

- [ ] **Step 5: Run all type_coerce tests**

Run: `pytest tests/unit/plugins/transforms/test_type_coerce.py -v`
Expected: All PASS

- [ ] **Step 6: Commit plugin class**

```bash
git add src/elspeth/plugins/transforms/type_coerce.py tests/unit/plugins/transforms/test_type_coerce.py
git commit -m "feat(type_coerce): add TypeCoerce transform plugin

Implements atomic per-row type normalization with strict conversion rules.
Audit trail tracks fields_coerced vs fields_unchanged."
```

---

## Task 4: Value Transform — Config Class

**Files:**
- Create: `src/elspeth/plugins/transforms/value_transform.py`
- Create: `tests/unit/plugins/transforms/test_value_transform.py`

- [ ] **Step 1: Create test file with config validation tests**

```python
"""Tests for ValueTransform transform — behavioral unit tests."""

import pytest
from pydantic import ValidationError

from elspeth.contracts.schema import SchemaConfig

OBSERVED_SCHEMA_CONFIG = SchemaConfig.from_dict({"mode": "observed"})


class TestValueTransformConfig:
    """Pydantic config validation for ValueTransformConfig."""

    def test_valid_config(self) -> None:
        from elspeth.plugins.transforms.value_transform import ValueTransformConfig
        cfg = ValueTransformConfig(
            operations=[
                {"target": "total", "expression": "row['price'] * row['quantity']"},
            ],
            schema_config=OBSERVED_SCHEMA_CONFIG,
        )
        assert len(cfg.operations) == 1
        assert cfg.operations[0].target == "total"

    def test_rejects_empty_operations(self) -> None:
        from elspeth.plugins.transforms.value_transform import ValueTransformConfig
        with pytest.raises(ValidationError, match="at least one"):
            ValueTransformConfig(operations=[], schema_config=OBSERVED_SCHEMA_CONFIG)

    def test_rejects_empty_target(self) -> None:
        from elspeth.plugins.transforms.value_transform import ValueTransformConfig
        with pytest.raises(ValidationError, match="target"):
            ValueTransformConfig(
                operations=[{"target": "", "expression": "row['x']"}],
                schema_config=OBSERVED_SCHEMA_CONFIG,
            )

    def test_rejects_empty_expression(self) -> None:
        from elspeth.plugins.transforms.value_transform import ValueTransformConfig
        with pytest.raises(ValidationError, match="expression"):
            ValueTransformConfig(
                operations=[{"target": "x", "expression": ""}],
                schema_config=OBSERVED_SCHEMA_CONFIG,
            )

    def test_rejects_invalid_expression_syntax(self) -> None:
        from elspeth.plugins.transforms.value_transform import ValueTransformConfig
        with pytest.raises(ValidationError, match="syntax|parse"):
            ValueTransformConfig(
                operations=[{"target": "x", "expression": "row['x'"}],  # Missing ]
                schema_config=OBSERVED_SCHEMA_CONFIG,
            )

    def test_rejects_forbidden_expression_constructs(self) -> None:
        from elspeth.plugins.transforms.value_transform import ValueTransformConfig
        # Lambda is forbidden by ExpressionParser
        with pytest.raises(ValidationError, match="Lambda|forbidden"):
            ValueTransformConfig(
                operations=[{"target": "x", "expression": "lambda: 1"}],
                schema_config=OBSERVED_SCHEMA_CONFIG,
            )

    def test_allows_duplicate_targets(self) -> None:
        from elspeth.plugins.transforms.value_transform import ValueTransformConfig
        cfg = ValueTransformConfig(
            operations=[
                {"target": "x", "expression": "row['x'] + 1"},
                {"target": "x", "expression": "row['x'] * 2"},  # Same target
            ],
            schema_config=OBSERVED_SCHEMA_CONFIG,
        )
        assert len(cfg.operations) == 2

    def test_from_dict_factory(self) -> None:
        from elspeth.plugins.transforms.value_transform import ValueTransformConfig
        cfg = ValueTransformConfig.from_dict({
            "schema": {"mode": "observed"},
            "operations": [{"target": "x", "expression": "row['a'] + row['b']"}],
        })
        assert cfg.operations[0].target == "x"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/plugins/transforms/test_value_transform.py::TestValueTransformConfig -v`
Expected: FAIL with "cannot import name 'ValueTransformConfig'"

- [ ] **Step 3: Implement ValueTransformConfig**

```python
"""ValueTransform transform plugin.

Applies expressions to compute new or modified field values.

IMPORTANT: Transforms use allow_coercion=False to catch upstream bugs.
If the source outputs wrong types, the transform crashes immediately.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from elspeth.core.expression_parser import (
    ExpressionParser,
    ExpressionSecurityError,
    ExpressionSyntaxError,
)
from elspeth.plugins.infrastructure.config_base import TransformDataConfig


class OperationSpec(BaseModel):
    """Single value transform operation specification."""

    model_config = {"extra": "forbid", "frozen": True}

    target: str
    expression: str
    # Parsed expression stored after validation
    _parsed_expression: ExpressionParser | None = None

    @field_validator("target")
    @classmethod
    def _validate_target(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("target field name must not be empty")
        return v

    @field_validator("expression")
    @classmethod
    def _validate_expression(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("expression must not be empty")
        return v

    @model_validator(mode="after")
    def _parse_expression(self) -> "OperationSpec":
        """Parse and validate expression at config time."""
        try:
            parser = ExpressionParser(self.expression)
            # Store the parsed expression for later use
            object.__setattr__(self, "_parsed_expression", parser)
        except ExpressionSyntaxError as e:
            raise ValueError(f"Expression syntax error: {e}") from e
        except ExpressionSecurityError as e:
            raise ValueError(f"Expression contains forbidden constructs: {e}") from e
        return self

    def get_parser(self) -> ExpressionParser:
        """Get the pre-parsed expression parser."""
        if self._parsed_expression is None:
            # Re-parse if needed (shouldn't happen after validation)
            return ExpressionParser(self.expression)
        return self._parsed_expression


class ValueTransformConfig(TransformDataConfig):
    """Configuration for value transform.

    Requires 'schema' in config to define input/output expectations.
    Use 'schema: {mode: observed}' for dynamic field handling.
    """

    operations: list[OperationSpec] = Field(
        ...,
        description="List of operations to apply (target + expression pairs)",
    )

    @model_validator(mode="after")
    def _validate_operations_not_empty(self) -> "ValueTransformConfig":
        if not self.operations:
            raise ValueError("operations must contain at least one operation")
        return self
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/plugins/transforms/test_value_transform.py::TestValueTransformConfig -v`
Expected: All PASS

- [ ] **Step 5: Commit config class**

```bash
git add src/elspeth/plugins/transforms/value_transform.py tests/unit/plugins/transforms/test_value_transform.py
git commit -m "feat(value_transform): add ValueTransformConfig with expression validation

Pydantic config class with OperationSpec for target/expression pairs.
Expressions are parsed at config time via ExpressionParser."
```

---

## Task 5: Value Transform — Plugin Class

**Files:**
- Modify: `src/elspeth/plugins/transforms/value_transform.py`
- Modify: `tests/unit/plugins/transforms/test_value_transform.py`

- [ ] **Step 1: Add plugin behavior tests**

Add to `tests/unit/plugins/transforms/test_value_transform.py`:

```python
from elspeth.contracts.plugin_context import PluginContext
from elspeth.testing import make_pipeline_row
from tests.fixtures.factories import make_source_context

DYNAMIC_SCHEMA = {"mode": "observed"}


class TestValueTransformBehavior:
    """Core value transformation mechanics."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        return make_source_context()

    def test_arithmetic_expression(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.value_transform import ValueTransform
        transform = ValueTransform({
            "schema": DYNAMIC_SCHEMA,
            "operations": [{"target": "total", "expression": "row['price'] * row['quantity']"}],
        })
        row = make_pipeline_row({"price": 10, "quantity": 2})
        result = transform.process(row, ctx)
        assert result.status == "success"
        assert result.row is not None
        assert result.row["total"] == 20

    def test_string_concatenation(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.value_transform import ValueTransform
        transform = ValueTransform({
            "schema": DYNAMIC_SCHEMA,
            "operations": [{"target": "line", "expression": "row['line'] + ' World'"}],
        })
        row = make_pipeline_row({"line": "Hello"})
        result = transform.process(row, ctx)
        assert result.status == "success"
        assert result.row is not None
        assert result.row["line"] == "Hello World"

    def test_multiple_operations_sequential(self, ctx: PluginContext) -> None:
        """Operations see results of prior operations."""
        from elspeth.plugins.transforms.value_transform import ValueTransform
        transform = ValueTransform({
            "schema": DYNAMIC_SCHEMA,
            "operations": [
                {"target": "subtotal", "expression": "row['price'] * row['quantity']"},
                {"target": "tax", "expression": "row['subtotal'] * 0.2"},
                {"target": "total", "expression": "row['subtotal'] + row['tax']"},
            ],
        })
        row = make_pipeline_row({"price": 100, "quantity": 2})
        result = transform.process(row, ctx)
        assert result.status == "success"
        assert result.row is not None
        assert result.row["subtotal"] == 200
        assert result.row["tax"] == 40.0
        assert result.row["total"] == 240.0

    def test_self_reference_overwrite(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.value_transform import ValueTransform
        transform = ValueTransform({
            "schema": DYNAMIC_SCHEMA,
            "operations": [{"target": "price", "expression": "row['price'] * 1.1"}],
        })
        row = make_pipeline_row({"price": 100})
        result = transform.process(row, ctx)
        assert result.status == "success"
        assert result.row is not None
        assert result.row["price"] == 110.0

    def test_duplicate_targets_sequential_rewrite(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.value_transform import ValueTransform
        transform = ValueTransform({
            "schema": DYNAMIC_SCHEMA,
            "operations": [
                {"target": "x", "expression": "row['x'] + 1"},
                {"target": "x", "expression": "row['x'] * 2"},
            ],
        })
        row = make_pipeline_row({"x": 5})
        result = transform.process(row, ctx)
        assert result.status == "success"
        assert result.row is not None
        # (5 + 1) * 2 = 12
        assert result.row["x"] == 12

    def test_creates_new_field(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.value_transform import ValueTransform
        transform = ValueTransform({
            "schema": DYNAMIC_SCHEMA,
            "operations": [{"target": "new_field", "expression": "row['x'] + 100"}],
        })
        row = make_pipeline_row({"x": 5})
        result = transform.process(row, ctx)
        assert result.status == "success"
        assert result.row is not None
        assert result.row["new_field"] == 105
        assert result.row["x"] == 5  # Original preserved

    def test_ternary_expression(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.value_transform import ValueTransform
        transform = ValueTransform({
            "schema": DYNAMIC_SCHEMA,
            "operations": [
                {"target": "discount", "expression": "row['price'] * 0.1 if row['price'] > 50 else 0"},
            ],
        })
        row = make_pipeline_row({"price": 100})
        result = transform.process(row, ctx)
        assert result.status == "success"
        assert result.row is not None
        assert result.row["discount"] == 10.0

    def test_missing_field_errors(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.value_transform import ValueTransform
        transform = ValueTransform({
            "schema": DYNAMIC_SCHEMA,
            "operations": [{"target": "total", "expression": "row['missing'] * 2"}],
        })
        row = make_pipeline_row({"other": 42})
        result = transform.process(row, ctx)
        assert result.status == "error"
        assert "missing" in result.error_reason.get("reason", "").lower()

    def test_type_error_in_expression(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.value_transform import ValueTransform
        transform = ValueTransform({
            "schema": DYNAMIC_SCHEMA,
            "operations": [{"target": "result", "expression": "row['text'] * row['num']"}],
        })
        # Can't multiply string by string (would need int)
        row = make_pipeline_row({"text": "hello", "num": "world"})
        result = transform.process(row, ctx)
        assert result.status == "error"

    def test_atomic_failure_no_partial_mutation(self, ctx: PluginContext) -> None:
        """If second operation fails, first operation should not be applied."""
        from elspeth.plugins.transforms.value_transform import ValueTransform
        transform = ValueTransform({
            "schema": DYNAMIC_SCHEMA,
            "operations": [
                {"target": "first", "expression": "row['x'] + 1"},  # Would succeed
                {"target": "second", "expression": "row['missing'] * 2"},  # Will fail
            ],
        })
        row = make_pipeline_row({"x": 5})
        result = transform.process(row, ctx)
        assert result.status == "error"
        # Original row should be unchanged (error path gets original row)

    def test_audit_trail(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.value_transform import ValueTransform
        transform = ValueTransform({
            "schema": DYNAMIC_SCHEMA,
            "operations": [
                {"target": "total", "expression": "row['price'] * row['quantity']"},
                {"target": "line", "expression": "row['line'] + ' modified'"},
            ],
        })
        row = make_pipeline_row({"price": 10, "quantity": 2, "line": "Hello"})
        result = transform.process(row, ctx)
        assert result.status == "success"
        assert result.success_reason["action"] == "transformed"
        assert "total" in result.success_reason["fields_added"]
        assert "line" in result.success_reason["fields_modified"]
        assert result.success_reason["operations_applied"] == 2

    def test_row_get_with_none_handling(self, ctx: PluginContext) -> None:
        """row.get() returning None in expression that handles it."""
        from elspeth.plugins.transforms.value_transform import ValueTransform
        transform = ValueTransform({
            "schema": DYNAMIC_SCHEMA,
            "operations": [
                {"target": "result", "expression": "row.get('optional') if row.get('optional') is not None else 0"},
            ],
        })
        row = make_pipeline_row({"other": 42})  # 'optional' is missing
        result = transform.process(row, ctx)
        assert result.status == "success"
        assert result.row is not None
        assert result.row["result"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/plugins/transforms/test_value_transform.py::TestValueTransformBehavior -v`
Expected: FAIL with "cannot import name 'ValueTransform'"

- [ ] **Step 3: Implement ValueTransform plugin class**

Add to `src/elspeth/plugins/transforms/value_transform.py`:

```python
import copy
from typing import Any

from elspeth.contracts.contexts import TransformContext
from elspeth.contracts.schema_contract import PipelineRow
from elspeth.core.expression_parser import ExpressionEvaluationError
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.results import TransformResult


class ValueTransform(BaseTransform):
    """Apply expressions to compute new or modified field values.

    Operations are evaluated in order on a working copy of the row.
    Each operation sees the results of prior operations (sequential visibility).
    If all operations succeed, the updated row is emitted.
    If any operation fails, the original row is returned as an error
    and no partial changes are emitted on the success path.

    Config options:
        schema: Required. Schema for input/output (use {mode: observed} for any fields)
        operations: List of {target, expression} specs defining field computations
    """

    name = "value_transform"
    plugin_version = "1.0.0"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = ValueTransformConfig.from_dict(config)
        self._operations = cfg.operations
        self._schema_config = cfg.schema_config

        # declared_output_fields for contract propagation
        self.declared_output_fields = frozenset(op.target for op in cfg.operations)

        self.input_schema, self.output_schema = self._create_schemas(
            cfg.schema_config,
            "ValueTransform",
            adds_fields=True,
        )

    def process(self, row: PipelineRow, ctx: TransformContext) -> TransformResult:
        """Apply expression operations to row.

        Args:
            row: Input row data
            ctx: Plugin context

        Returns:
            TransformResult with computed field values, or error if any operation fails
        """
        # Work on a copy to support atomic rollback
        working_data = copy.deepcopy(row.to_dict())
        fields_modified: list[str] = []
        fields_added: list[str] = []
        original_fields = set(row.to_dict().keys())

        for op in self._operations:
            target = op.target
            parser = op.get_parser()

            try:
                result = parser.evaluate(working_data)
            except ExpressionEvaluationError as e:
                return TransformResult.error({
                    "action": "error",
                    "plugin": "value_transform",
                    "target": target,
                    "expression": op.expression,
                    "reason": str(e),
                })
            except Exception as e:
                # Catch-all for unexpected evaluation errors
                return TransformResult.error({
                    "action": "error",
                    "plugin": "value_transform",
                    "target": target,
                    "expression": op.expression,
                    "reason": f"Unexpected error: {type(e).__name__}: {e}",
                })

            # Track field changes
            if target in original_fields:
                if target not in fields_modified:
                    fields_modified.append(target)
            else:
                if target not in fields_added:
                    fields_added.append(target)

            # Write result to working copy
            working_data[target] = result

        return TransformResult.success(
            PipelineRow(working_data, row.contract),
            success_reason={
                "action": "transformed",
                "fields_modified": fields_modified,
                "fields_added": fields_added,
                "operations_applied": len(self._operations),
            },
        )

    def close(self) -> None:
        """No resources to release."""
        pass
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/plugins/transforms/test_value_transform.py::TestValueTransformBehavior -v`
Expected: All PASS

- [ ] **Step 5: Run all value_transform tests**

Run: `pytest tests/unit/plugins/transforms/test_value_transform.py -v`
Expected: All PASS

- [ ] **Step 6: Commit plugin class**

```bash
git add src/elspeth/plugins/transforms/value_transform.py tests/unit/plugins/transforms/test_value_transform.py
git commit -m "feat(value_transform): add ValueTransform plugin

Implements expression-based field computation using ExpressionParser.
Sequential operation visibility, atomic per-row failure semantics."
```

---

## Task 6: Plugin Registration

**Files:**
- Modify: `src/elspeth/plugins/transforms/__init__.py`

- [ ] **Step 1: Check current plugin exports**

Run: `cat src/elspeth/plugins/transforms/__init__.py`

- [ ] **Step 2: Add new plugins to exports**

Add the new imports to `src/elspeth/plugins/transforms/__init__.py`:

```python
from elspeth.plugins.transforms.type_coerce import TypeCoerce
from elspeth.plugins.transforms.value_transform import ValueTransform
```

And add to `__all__` if it exists:

```python
__all__ = [
    # ... existing exports ...
    "TypeCoerce",
    "ValueTransform",
]
```

- [ ] **Step 3: Verify imports work**

Run: `python -c "from elspeth.plugins.transforms import TypeCoerce, ValueTransform; print('OK')"`
Expected: "OK"

- [ ] **Step 4: Commit registration**

```bash
git add src/elspeth/plugins/transforms/__init__.py
git commit -m "feat: register TypeCoerce and ValueTransform plugins

Adds new transforms to plugin exports."
```

---

## Task 7: Integration Test — Combined Pipeline

**Files:**
- Create: `tests/integration/plugins/test_type_coerce_value_transform_pipeline.py`

- [ ] **Step 1: Create integration test**

```python
"""Integration test: type_coerce + value_transform in a pipeline.

Tests the recommended pattern: normalize types first, then compute values.
"""

import pytest

from elspeth.testing import make_pipeline_row
from tests.fixtures.factories import make_source_context


class TestTypeCoerceValueTransformPipeline:
    """Test type_coerce followed by value_transform."""

    @pytest.fixture
    def ctx(self):
        return make_source_context()

    def test_typical_pipeline_pattern(self, ctx) -> None:
        """Normalize types, then compute derived values."""
        from elspeth.plugins.transforms.type_coerce import TypeCoerce
        from elspeth.plugins.transforms.value_transform import ValueTransform

        # Step 1: Normalize types
        type_coerce = TypeCoerce({
            "schema": {"mode": "observed"},
            "conversions": [
                {"field": "price", "to": "float"},
                {"field": "quantity", "to": "int"},
            ],
        })

        # Step 2: Compute derived values
        value_transform = ValueTransform({
            "schema": {"mode": "observed"},
            "operations": [
                {"target": "subtotal", "expression": "row['price'] * row['quantity']"},
                {"target": "tax", "expression": "row['subtotal'] * 0.2"},
                {"target": "total", "expression": "row['subtotal'] + row['tax']"},
            ],
        })

        # Input with string types (typical from CSV/API)
        row = make_pipeline_row({
            "price": " 12.50 ",
            "quantity": "3",
            "description": "Widget",
        })

        # Apply type coercion
        result1 = type_coerce.process(row, ctx)
        assert result1.status == "success"
        assert result1.row is not None
        assert result1.row["price"] == 12.5
        assert result1.row["quantity"] == 3

        # Apply value transform
        result2 = value_transform.process(result1.row, ctx)
        assert result2.status == "success"
        assert result2.row is not None
        assert result2.row["subtotal"] == 37.5
        assert result2.row["tax"] == 7.5
        assert result2.row["total"] == 45.0
        # Original fields preserved
        assert result2.row["description"] == "Widget"

    def test_type_error_without_coercion(self, ctx) -> None:
        """Show what happens if you skip type_coerce with string data."""
        from elspeth.plugins.transforms.value_transform import ValueTransform

        value_transform = ValueTransform({
            "schema": {"mode": "observed"},
            "operations": [
                {"target": "total", "expression": "row['price'] * row['quantity']"},
            ],
        })

        # String types will fail multiplication
        row = make_pipeline_row({"price": "12.50", "quantity": "3"})
        result = value_transform.process(row, ctx)
        assert result.status == "error"
        # This demonstrates why type_coerce should come first
```

- [ ] **Step 2: Run integration test**

Run: `pytest tests/integration/plugins/test_type_coerce_value_transform_pipeline.py -v`
Expected: All PASS

- [ ] **Step 3: Commit integration test**

```bash
git add tests/integration/plugins/test_type_coerce_value_transform_pipeline.py
git commit -m "test: add integration test for type_coerce + value_transform pipeline

Demonstrates the recommended pattern: normalize types first, then compute."
```

---

## Task 8: Final Verification

- [ ] **Step 1: Run all new tests**

Run: `pytest tests/unit/plugins/transforms/test_type_coerce.py tests/unit/plugins/transforms/test_value_transform.py tests/integration/plugins/test_type_coerce_value_transform_pipeline.py -v`
Expected: All PASS

- [ ] **Step 2: Run type checking**

Run: `mypy src/elspeth/plugins/transforms/type_coerce.py src/elspeth/plugins/transforms/value_transform.py`
Expected: No errors

- [ ] **Step 3: Run linting**

Run: `ruff check src/elspeth/plugins/transforms/type_coerce.py src/elspeth/plugins/transforms/value_transform.py`
Expected: No errors

- [ ] **Step 4: Run full test suite to check for regressions**

Run: `pytest tests/unit/ -x -q`
Expected: All PASS

- [ ] **Step 5: Final commit if any fixes were needed**

```bash
git status
# If there are uncommitted changes from fixes:
git add -A
git commit -m "fix: address type/lint issues in type_coerce and value_transform"
```
