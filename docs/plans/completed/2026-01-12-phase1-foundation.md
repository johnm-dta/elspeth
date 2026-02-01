# Phase 1: Foundation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the foundational infrastructure for Elspeth - canonical JSON hashing, configuration system, payload storage, Landscape audit core, and DAG validation.

**Architecture:** Phase 1 proves the audit infrastructure works with deterministic operations before any external calls (LLMs, APIs) are added. Every component is designed for testability and follows the "no silent drops" principle. The Landscape is the source of truth; OpenTelemetry is a view.

**Tech Stack:** Python 3.11+, rfc8785 (canonical JSON), SQLAlchemy Core (not ORM), NetworkX (DAG), Dynaconf + Pydantic (config), pytest + hypothesis (testing)

---

---

## Implementation Summary

**Status:** Completed
**Commits:** See git history for this feature
**Notes:** Phase 1 foundation implemented including canonical JSON hashing (RFC 8785), Dynaconf+Pydantic configuration, payload storage, Landscape audit core, and DAG validation with NetworkX.

---

## Task 1: Canonical JSON - Core Normalization

**Files:**
- Create: `src/elspeth/core/canonical.py`
- Create: `tests/core/test_canonical.py`

### Step 1: Write the failing test for basic normalization

```python
# tests/core/test_canonical.py
"""Tests for canonical JSON serialization and hashing."""

import pytest


class TestNormalizeValue:
    """Test _normalize_value handles Python primitives."""

    def test_string_passthrough(self) -> None:
        from elspeth.core.canonical import _normalize_value

        assert _normalize_value("hello") == "hello"

    def test_int_passthrough(self) -> None:
        from elspeth.core.canonical import _normalize_value

        assert _normalize_value(42) == 42

    def test_float_passthrough(self) -> None:
        from elspeth.core.canonical import _normalize_value

        assert _normalize_value(3.14) == 3.14

    def test_none_passthrough(self) -> None:
        from elspeth.core.canonical import _normalize_value

        assert _normalize_value(None) is None

    def test_bool_passthrough(self) -> None:
        from elspeth.core.canonical import _normalize_value

        assert _normalize_value(True) is True
        assert _normalize_value(False) is False
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/test_canonical.py -v`
Expected: FAIL with "ModuleNotFoundError" or "ImportError"

### Step 3: Write minimal implementation

```python
# src/elspeth/core/canonical.py
"""
Canonical JSON serialization for deterministic hashing.

Two-phase approach:
1. Normalize: Convert pandas/numpy types to JSON-safe primitives (our code)
2. Serialize: Produce deterministic JSON per RFC 8785/JCS (rfc8785 package)

IMPORTANT: NaN and Infinity are strictly REJECTED, not silently converted.
This is defense-in-depth for audit integrity.
"""

from typing import Any

# Version string stored with every run for hash verification
CANONICAL_VERSION = "sha256-rfc8785-v1"


def _normalize_value(obj: Any) -> Any:
    """Convert a single value to JSON-safe primitive.

    Args:
        obj: Any Python value

    Returns:
        JSON-serializable primitive

    Raises:
        ValueError: If value contains NaN or Infinity
    """
    # Primitives pass through unchanged
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj

    if isinstance(obj, float):
        return obj

    return obj
```

### Step 4: Run test to verify it passes

Run: `pytest tests/core/test_canonical.py -v`
Expected: PASS (5 tests)

### Step 5: Commit

```bash
git add tests/core/test_canonical.py src/elspeth/core/canonical.py
git commit -m "feat(canonical): add basic value normalization"
```

---

## Task 2: Canonical JSON - Reject NaN and Infinity

**Files:**
- Modify: `src/elspeth/core/canonical.py`
- Modify: `tests/core/test_canonical.py`

### Step 1: Write the failing tests for NaN/Infinity rejection

```python
# Add to tests/core/test_canonical.py

import math


class TestNanInfinityRejection:
    """NaN and Infinity must be rejected, not silently converted."""

    def test_nan_raises_value_error(self) -> None:
        from elspeth.core.canonical import _normalize_value

        with pytest.raises(ValueError, match="non-finite"):
            _normalize_value(float("nan"))

    def test_positive_infinity_raises_value_error(self) -> None:
        from elspeth.core.canonical import _normalize_value

        with pytest.raises(ValueError, match="non-finite"):
            _normalize_value(float("inf"))

    def test_negative_infinity_raises_value_error(self) -> None:
        from elspeth.core.canonical import _normalize_value

        with pytest.raises(ValueError, match="non-finite"):
            _normalize_value(float("-inf"))

    def test_normal_float_allowed(self) -> None:
        from elspeth.core.canonical import _normalize_value

        # These should NOT raise
        assert _normalize_value(0.0) == 0.0
        assert _normalize_value(-0.0) == -0.0
        assert _normalize_value(1e308) == 1e308
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/test_canonical.py::TestNanInfinityRejection -v`
Expected: FAIL (3 tests fail - no ValueError raised)

### Step 3: Update implementation to reject non-finite floats

```python
# Update _normalize_value in src/elspeth/core/canonical.py
import math
from typing import Any

CANONICAL_VERSION = "sha256-rfc8785-v1"


def _normalize_value(obj: Any) -> Any:
    """Convert a single value to JSON-safe primitive.

    Args:
        obj: Any Python value

    Returns:
        JSON-serializable primitive

    Raises:
        ValueError: If value contains NaN or Infinity
    """
    # Primitives pass through unchanged
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj

    # Floats: check for non-finite values FIRST
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            raise ValueError(
                f"Cannot canonicalize non-finite float: {obj}. "
                "Use None for missing values, not NaN."
            )
        return obj

    return obj
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/core/test_canonical.py -v`
Expected: PASS (all tests)

### Step 5: Commit

```bash
git add -u
git commit -m "feat(canonical): reject NaN and Infinity with clear error"
```

---

## Task 3: Canonical JSON - NumPy Type Handling

**Files:**
- Modify: `src/elspeth/core/canonical.py`
- Modify: `tests/core/test_canonical.py`

### Step 1: Write the failing tests for numpy types

```python
# Add to tests/core/test_canonical.py

import numpy as np


class TestNumpyTypeConversion:
    """NumPy types must be converted to Python primitives."""

    def test_numpy_int64_converts_to_int(self) -> None:
        from elspeth.core.canonical import _normalize_value

        result = _normalize_value(np.int64(42))
        assert result == 42
        assert type(result) is int

    def test_numpy_float64_converts_to_float(self) -> None:
        from elspeth.core.canonical import _normalize_value

        result = _normalize_value(np.float64(3.14))
        assert result == 3.14
        assert type(result) is float

    def test_numpy_float64_nan_raises(self) -> None:
        from elspeth.core.canonical import _normalize_value

        with pytest.raises(ValueError, match="non-finite"):
            _normalize_value(np.float64("nan"))

    def test_numpy_float64_inf_raises(self) -> None:
        from elspeth.core.canonical import _normalize_value

        with pytest.raises(ValueError, match="non-finite"):
            _normalize_value(np.float64("inf"))

    def test_numpy_bool_converts_to_bool(self) -> None:
        from elspeth.core.canonical import _normalize_value

        result = _normalize_value(np.bool_(True))
        assert result is True
        assert type(result) is bool

    def test_numpy_array_converts_to_list(self) -> None:
        from elspeth.core.canonical import _normalize_value

        result = _normalize_value(np.array([1, 2, 3]))
        assert result == [1, 2, 3]
        assert type(result) is list
        assert all(type(x) is int for x in result)
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/test_canonical.py::TestNumpyTypeConversion -v`
Expected: FAIL (types not converted)

### Step 3: Update implementation for numpy types

```python
# Update src/elspeth/core/canonical.py
import math
from typing import Any

import numpy as np

CANONICAL_VERSION = "sha256-rfc8785-v1"


def _normalize_value(obj: Any) -> Any:
    """Convert a single value to JSON-safe primitive.

    Handles pandas and numpy types that appear in real pipeline data.

    NaN Policy: STRICT REJECTION
    - NaN and Infinity are invalid input states, not "missing"
    - Use None/pd.NA/NaT for intentional missing values
    - This prevents silent data corruption in audit records

    Args:
        obj: Any Python value

    Returns:
        JSON-serializable primitive

    Raises:
        ValueError: If value contains NaN or Infinity
    """
    # Check for NaN/Infinity FIRST (before type coercion)
    if isinstance(obj, (float, np.floating)):
        if math.isnan(obj) or math.isinf(obj):
            raise ValueError(
                f"Cannot canonicalize non-finite float: {obj}. "
                "Use None for missing values, not NaN."
            )
        if isinstance(obj, np.floating):
            return float(obj)
        return obj

    # Primitives pass through unchanged
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj

    # NumPy scalar types
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_normalize_value(x) for x in obj.tolist()]

    return obj
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/core/test_canonical.py -v`
Expected: PASS (all tests)

### Step 5: Commit

```bash
git add -u
git commit -m "feat(canonical): convert numpy types to Python primitives"
```

---

## Task 4: Canonical JSON - Pandas Type Handling

**Files:**
- Modify: `src/elspeth/core/canonical.py`
- Modify: `tests/core/test_canonical.py`

### Step 1: Write the failing tests for pandas types

```python
# Add to tests/core/test_canonical.py

import pandas as pd


class TestPandasTypeConversion:
    """Pandas types must be converted to JSON-safe primitives."""

    def test_pandas_timestamp_naive_to_utc_iso(self) -> None:
        from elspeth.core.canonical import _normalize_value

        ts = pd.Timestamp("2026-01-12 10:30:00")
        result = _normalize_value(ts)
        assert result == "2026-01-12T10:30:00+00:00"
        assert type(result) is str

    def test_pandas_timestamp_aware_to_utc_iso(self) -> None:
        from elspeth.core.canonical import _normalize_value

        ts = pd.Timestamp("2026-01-12 10:30:00", tz="US/Eastern")
        result = _normalize_value(ts)
        # Should be converted to UTC
        assert "+00:00" in result or "Z" in result
        assert type(result) is str

    def test_pandas_nat_to_none(self) -> None:
        from elspeth.core.canonical import _normalize_value

        result = _normalize_value(pd.NaT)
        assert result is None

    def test_pandas_na_to_none(self) -> None:
        from elspeth.core.canonical import _normalize_value

        result = _normalize_value(pd.NA)
        assert result is None
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/test_canonical.py::TestPandasTypeConversion -v`
Expected: FAIL (types not converted)

### Step 3: Update implementation for pandas types

```python
# Update src/elspeth/core/canonical.py - add pandas handling
import math
from typing import Any

import numpy as np
import pandas as pd

CANONICAL_VERSION = "sha256-rfc8785-v1"


def _normalize_value(obj: Any) -> Any:
    """Convert a single value to JSON-safe primitive.

    Handles pandas and numpy types that appear in real pipeline data.

    NaN Policy: STRICT REJECTION
    - NaN and Infinity are invalid input states, not "missing"
    - Use None/pd.NA/NaT for intentional missing values
    - This prevents silent data corruption in audit records

    Args:
        obj: Any Python value

    Returns:
        JSON-serializable primitive

    Raises:
        ValueError: If value contains NaN or Infinity
    """
    # Check for NaN/Infinity FIRST (before type coercion)
    if isinstance(obj, (float, np.floating)):
        if math.isnan(obj) or math.isinf(obj):
            raise ValueError(
                f"Cannot canonicalize non-finite float: {obj}. "
                "Use None for missing values, not NaN."
            )
        if isinstance(obj, np.floating):
            return float(obj)
        return obj

    # Primitives pass through unchanged
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj

    # NumPy scalar types
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_normalize_value(x) for x in obj.tolist()]

    # Pandas types
    if isinstance(obj, pd.Timestamp):
        # Naive timestamps assumed UTC (explicit policy)
        if obj.tz is None:
            return obj.tz_localize("UTC").isoformat()
        return obj.tz_convert("UTC").isoformat()

    # Intentional missing values (NOT NaN - that's rejected above)
    if obj is pd.NA or (isinstance(obj, type(pd.NaT)) and obj is pd.NaT):
        return None

    return obj
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/core/test_canonical.py -v`
Expected: PASS (all tests)

### Step 5: Commit

```bash
git add -u
git commit -m "feat(canonical): convert pandas types (Timestamp, NaT, NA)"
```

---

## Task 5: Canonical JSON - Datetime and Special Types

**Files:**
- Modify: `src/elspeth/core/canonical.py`
- Modify: `tests/core/test_canonical.py`

### Step 1: Write the failing tests for datetime, bytes, Decimal

```python
# Add to tests/core/test_canonical.py

import base64
from datetime import datetime, timezone
from decimal import Decimal


class TestSpecialTypeConversion:
    """Special Python types must be converted consistently."""

    def test_datetime_naive_to_utc_iso(self) -> None:
        from elspeth.core.canonical import _normalize_value

        dt = datetime(2026, 1, 12, 10, 30, 0)
        result = _normalize_value(dt)
        assert result == "2026-01-12T10:30:00+00:00"

    def test_datetime_aware_to_utc_iso(self) -> None:
        from elspeth.core.canonical import _normalize_value

        dt = datetime(2026, 1, 12, 10, 30, 0, tzinfo=timezone.utc)
        result = _normalize_value(dt)
        assert result == "2026-01-12T10:30:00+00:00"

    def test_bytes_to_base64_wrapper(self) -> None:
        from elspeth.core.canonical import _normalize_value

        data = b"hello world"
        result = _normalize_value(data)
        assert result == {"__bytes__": base64.b64encode(data).decode("ascii")}

    def test_decimal_to_string(self) -> None:
        from elspeth.core.canonical import _normalize_value

        # Decimal preserves precision as string
        result = _normalize_value(Decimal("123.456789012345"))
        assert result == "123.456789012345"
        assert type(result) is str
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/test_canonical.py::TestSpecialTypeConversion -v`
Expected: FAIL

### Step 3: Update implementation for special types

```python
# Update src/elspeth/core/canonical.py - add datetime, bytes, Decimal
import base64
import math
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

CANONICAL_VERSION = "sha256-rfc8785-v1"


def _normalize_value(obj: Any) -> Any:
    """Convert a single value to JSON-safe primitive.

    Handles pandas and numpy types that appear in real pipeline data.

    NaN Policy: STRICT REJECTION
    - NaN and Infinity are invalid input states, not "missing"
    - Use None/pd.NA/NaT for intentional missing values
    - This prevents silent data corruption in audit records

    Args:
        obj: Any Python value

    Returns:
        JSON-serializable primitive

    Raises:
        ValueError: If value contains NaN or Infinity
    """
    # Check for NaN/Infinity FIRST (before type coercion)
    if isinstance(obj, (float, np.floating)):
        if math.isnan(obj) or math.isinf(obj):
            raise ValueError(
                f"Cannot canonicalize non-finite float: {obj}. "
                "Use None for missing values, not NaN."
            )
        if isinstance(obj, np.floating):
            return float(obj)
        return obj

    # Primitives pass through unchanged
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj

    # NumPy scalar types
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_normalize_value(x) for x in obj.tolist()]

    # Pandas types
    if isinstance(obj, pd.Timestamp):
        if obj.tz is None:
            return obj.tz_localize("UTC").isoformat()
        return obj.tz_convert("UTC").isoformat()

    if obj is pd.NA or (isinstance(obj, type(pd.NaT)) and obj is pd.NaT):
        return None

    # Standard library types
    if isinstance(obj, datetime):
        if obj.tzinfo is None:
            obj = obj.replace(tzinfo=timezone.utc)
        return obj.astimezone(timezone.utc).isoformat()

    if isinstance(obj, bytes):
        return {"__bytes__": base64.b64encode(obj).decode("ascii")}

    if isinstance(obj, Decimal):
        return str(obj)

    return obj
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/core/test_canonical.py -v`
Expected: PASS (all tests)

### Step 5: Commit

```bash
git add -u
git commit -m "feat(canonical): handle datetime, bytes, Decimal types"
```

---

## Task 6: Canonical JSON - Recursive Normalization

**Files:**
- Modify: `src/elspeth/core/canonical.py`
- Modify: `tests/core/test_canonical.py`

### Step 1: Write the failing tests for recursive structures

```python
# Add to tests/core/test_canonical.py

class TestRecursiveNormalization:
    """Nested structures must be normalized recursively."""

    def test_dict_with_numpy_values(self) -> None:
        from elspeth.core.canonical import _normalize_for_canonical

        data = {"count": np.int64(42), "rate": np.float64(3.14)}
        result = _normalize_for_canonical(data)
        assert result == {"count": 42, "rate": 3.14}
        assert type(result["count"]) is int
        assert type(result["rate"]) is float

    def test_list_with_mixed_types(self) -> None:
        from elspeth.core.canonical import _normalize_for_canonical

        data = [np.int64(1), pd.Timestamp("2026-01-12"), None]
        result = _normalize_for_canonical(data)
        assert result[0] == 1
        assert "2026-01-12" in result[1]
        assert result[2] is None

    def test_nested_dict(self) -> None:
        from elspeth.core.canonical import _normalize_for_canonical

        data = {
            "outer": {
                "inner": np.int64(42),
                "list": [np.float64(1.0), np.float64(2.0)],
            }
        }
        result = _normalize_for_canonical(data)
        assert result["outer"]["inner"] == 42
        assert result["outer"]["list"] == [1.0, 2.0]

    def test_tuple_converts_to_list(self) -> None:
        from elspeth.core.canonical import _normalize_for_canonical

        data = (1, 2, 3)
        result = _normalize_for_canonical(data)
        assert result == [1, 2, 3]
        assert type(result) is list

    def test_nan_in_nested_raises(self) -> None:
        from elspeth.core.canonical import _normalize_for_canonical

        data = {"values": [1.0, float("nan"), 3.0]}
        with pytest.raises(ValueError, match="non-finite"):
            _normalize_for_canonical(data)
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/test_canonical.py::TestRecursiveNormalization -v`
Expected: FAIL (function doesn't exist)

### Step 3: Add recursive normalization function

```python
# Add to src/elspeth/core/canonical.py after _normalize_value

def _normalize_for_canonical(data: Any) -> Any:
    """Recursively normalize a data structure for canonical JSON.

    Converts pandas/numpy types to JSON-safe primitives.

    Args:
        data: Any data structure (dict, list, primitive)

    Returns:
        Normalized data structure with only JSON-safe types

    Raises:
        ValueError: If data contains NaN, Infinity, or other non-serializable values
    """
    if isinstance(data, dict):
        return {k: _normalize_for_canonical(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [_normalize_for_canonical(v) for v in data]
    return _normalize_value(data)
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/core/test_canonical.py -v`
Expected: PASS (all tests)

### Step 5: Commit

```bash
git add -u
git commit -m "feat(canonical): add recursive structure normalization"
```

---

## Task 7: Canonical JSON - RFC 8785 Serialization and Hashing

**Files:**
- Modify: `src/elspeth/core/canonical.py`
- Modify: `tests/core/test_canonical.py`

### Step 1: Write the failing tests for canonical_json and stable_hash

```python
# Add to tests/core/test_canonical.py

import hashlib


class TestCanonicalJsonSerialization:
    """RFC 8785 canonical JSON serialization."""

    def test_canonical_json_sorts_keys(self) -> None:
        from elspeth.core.canonical import canonical_json

        # Keys should be sorted alphabetically
        data = {"z": 1, "a": 2, "m": 3}
        result = canonical_json(data)
        assert result == '{"a":2,"m":3,"z":1}'

    def test_canonical_json_no_whitespace(self) -> None:
        from elspeth.core.canonical import canonical_json

        data = {"key": "value", "list": [1, 2, 3]}
        result = canonical_json(data)
        assert " " not in result
        assert "\n" not in result

    def test_canonical_json_handles_numpy(self) -> None:
        from elspeth.core.canonical import canonical_json

        data = {"count": np.int64(42)}
        result = canonical_json(data)
        assert result == '{"count":42}'


class TestStableHash:
    """Stable hashing with versioned algorithm."""

    def test_stable_hash_returns_hex_string(self) -> None:
        from elspeth.core.canonical import stable_hash

        result = stable_hash({"key": "value"})
        assert isinstance(result, str)
        assert len(result) == 64  # SHA-256 hex

    def test_stable_hash_deterministic(self) -> None:
        from elspeth.core.canonical import stable_hash

        data = {"a": 1, "b": [2, 3]}
        hash1 = stable_hash(data)
        hash2 = stable_hash(data)
        assert hash1 == hash2

    def test_stable_hash_key_order_independent(self) -> None:
        from elspeth.core.canonical import stable_hash

        # Different key order should produce same hash
        hash1 = stable_hash({"a": 1, "b": 2})
        hash2 = stable_hash({"b": 2, "a": 1})
        assert hash1 == hash2

    def test_stable_hash_verifiable(self) -> None:
        from elspeth.core.canonical import canonical_json, stable_hash

        data = {"test": "data"}
        hash_result = stable_hash(data)
        # Should match manual computation
        canonical = canonical_json(data)
        expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert hash_result == expected
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/test_canonical.py::TestCanonicalJsonSerialization -v`
Expected: FAIL (function doesn't exist)

### Step 3: Add canonical_json and stable_hash functions

```python
# Add to src/elspeth/core/canonical.py

import hashlib

import rfc8785


def canonical_json(obj: Any) -> str:
    """Produce canonical JSON for hashing.

    Two-phase approach:
    1. Normalize pandas/numpy types to JSON-safe primitives (our code)
    2. Serialize per RFC 8785/JCS standard (rfc8785 package)

    Args:
        obj: Data structure to serialize

    Returns:
        Canonical JSON string (no whitespace, sorted keys)

    Raises:
        ValueError: If data contains NaN, Infinity, or other non-finite values
        TypeError: If data contains types that cannot be serialized
    """
    normalized = _normalize_for_canonical(obj)
    return rfc8785.dumps(normalized)


def stable_hash(obj: Any, version: str = CANONICAL_VERSION) -> str:
    """Compute stable hash of object.

    Args:
        obj: Data structure to hash
        version: Hash algorithm version (stored with runs for verification)

    Returns:
        SHA-256 hex digest of canonical JSON
    """
    canonical = canonical_json(obj)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/core/test_canonical.py -v`
Expected: PASS (all tests)

### Step 5: Commit

```bash
git add -u
git commit -m "feat(canonical): add RFC 8785 serialization and stable hashing"
```

---

## Task 8: Canonical JSON - Cross-Process Stability Test

**Files:**
- Create: `tests/core/test_canonical_stability.py`

### Step 1: Write the cross-process hash stability test

```python
# tests/core/test_canonical_stability.py
"""Cross-process hash stability tests.

These tests verify that hashes are stable across different Python processes,
which is critical for audit integrity.
"""

import subprocess
import sys

import pytest


class TestCrossProcessStability:
    """Verify hashes are stable across Python processes."""

    def test_hash_stable_across_subprocess(self) -> None:
        """Hash computed in subprocess must match hash computed here."""
        from elspeth.core.canonical import stable_hash

        test_data = {"name": "test", "values": [1, 2, 3], "nested": {"a": 1}}

        # Compute hash in current process
        local_hash = stable_hash(test_data)

        # Compute hash in subprocess
        code = """
import json
import sys
sys.path.insert(0, 'src')
from elspeth.core.canonical import stable_hash
data = {"name": "test", "values": [1, 2, 3], "nested": {"a": 1}}
print(stable_hash(data))
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
        )
        subprocess_hash = result.stdout.strip()

        assert local_hash == subprocess_hash, (
            f"Hash mismatch across processes!\n"
            f"Local: {local_hash}\n"
            f"Subprocess: {subprocess_hash}"
        )

    def test_version_string_format(self) -> None:
        """Version string should identify algorithm components."""
        from elspeth.core.canonical import CANONICAL_VERSION

        assert "sha256" in CANONICAL_VERSION
        assert "rfc8785" in CANONICAL_VERSION
        assert "v1" in CANONICAL_VERSION
```

### Step 2: Run test to verify it passes

Run: `pytest tests/core/test_canonical_stability.py -v`
Expected: PASS

### Step 3: Commit

```bash
git add tests/core/test_canonical_stability.py
git commit -m "test(canonical): add cross-process hash stability test"
```

---

## Task 9: Canonical JSON - Public API Exports

**Files:**
- Modify: `src/elspeth/core/__init__.py`
- Modify: `src/elspeth/core/canonical.py`

### Step 1: Write the failing test for public API

```python
# Add to tests/core/test_canonical.py

class TestPublicAPI:
    """Public API should be importable from elspeth.core."""

    def test_canonical_json_importable(self) -> None:
        from elspeth.core import canonical_json

        assert callable(canonical_json)

    def test_stable_hash_importable(self) -> None:
        from elspeth.core import stable_hash

        assert callable(stable_hash)

    def test_canonical_version_importable(self) -> None:
        from elspeth.core import CANONICAL_VERSION

        assert isinstance(CANONICAL_VERSION, str)
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/test_canonical.py::TestPublicAPI -v`
Expected: FAIL (ImportError)

### Step 3: Update __init__.py to export public API

```python
# src/elspeth/core/__init__.py
"""Core infrastructure: Landscape, Canonical, Configuration."""

from elspeth.core.canonical import (
    CANONICAL_VERSION,
    canonical_json,
    stable_hash,
)

__all__ = [
    "CANONICAL_VERSION",
    "canonical_json",
    "stable_hash",
]
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/core/test_canonical.py -v`
Expected: PASS (all tests)

### Step 5: Commit

```bash
git add -u
git commit -m "feat(core): export canonical JSON public API"
```

---

## Task 10: Configuration - Settings Schema

**Files:**
- Create: `src/elspeth/core/config.py`
- Create: `tests/core/test_config.py`

### Step 1: Write the failing test for settings schema

```python
# tests/core/test_config.py
"""Tests for configuration loading and validation."""

import pytest
from pydantic import ValidationError


class TestLandscapeSettings:
    """Landscape configuration settings."""

    def test_landscape_settings_defaults(self) -> None:
        from elspeth.core.config import LandscapeSettings

        settings = LandscapeSettings()
        assert settings.enabled is True
        assert settings.backend == "sqlite"
        assert settings.path == "./runs/landscape.db"

    def test_landscape_settings_custom(self) -> None:
        from elspeth.core.config import LandscapeSettings

        settings = LandscapeSettings(
            enabled=False,
            backend="postgresql",
            path="postgresql://localhost/elspeth",
        )
        assert settings.enabled is False
        assert settings.backend == "postgresql"


class TestRetentionSettings:
    """Retention policy settings."""

    def test_retention_defaults(self) -> None:
        from elspeth.core.config import RetentionSettings

        settings = RetentionSettings()
        assert settings.row_payloads_days == 90
        assert settings.call_payloads_days == 90

    def test_retention_custom(self) -> None:
        from elspeth.core.config import RetentionSettings

        settings = RetentionSettings(row_payloads_days=30, call_payloads_days=7)
        assert settings.row_payloads_days == 30
        assert settings.call_payloads_days == 7


class TestConcurrencySettings:
    """Concurrency settings."""

    def test_concurrency_defaults(self) -> None:
        from elspeth.core.config import ConcurrencySettings

        settings = ConcurrencySettings()
        assert settings.max_workers == 4

    def test_concurrency_validation(self) -> None:
        from elspeth.core.config import ConcurrencySettings

        with pytest.raises(ValidationError):
            ConcurrencySettings(max_workers=0)

        with pytest.raises(ValidationError):
            ConcurrencySettings(max_workers=-1)
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/test_config.py -v`
Expected: FAIL (ImportError)

### Step 3: Create configuration schema

```python
# src/elspeth/core/config.py
"""Configuration loading and validation.

Uses Pydantic for schema validation with sensible defaults.
Dynaconf handles multi-source loading (files, env vars, etc.)
"""

from pydantic import BaseModel, Field


class RetentionSettings(BaseModel):
    """Payload retention policy."""

    row_payloads_days: int = Field(default=90, ge=1)
    call_payloads_days: int = Field(default=90, ge=1)
    compress_after_days: int = Field(default=7, ge=1)


class LandscapeSettings(BaseModel):
    """Landscape audit system configuration."""

    enabled: bool = True
    backend: str = "sqlite"
    path: str = "./runs/landscape.db"
    retention: RetentionSettings = Field(default_factory=RetentionSettings)


class ConcurrencySettings(BaseModel):
    """Concurrency and parallelism settings."""

    max_workers: int = Field(default=4, ge=1)
    batch_size: int = Field(default=10, ge=1)


class ElspethSettings(BaseModel):
    """Root configuration schema for Elspeth."""

    landscape: LandscapeSettings = Field(default_factory=LandscapeSettings)
    concurrency: ConcurrencySettings = Field(default_factory=ConcurrencySettings)
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/core/test_config.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/core/config.py tests/core/test_config.py
git commit -m "feat(config): add Pydantic settings schema"
```

---

## Task 11: Configuration - Dynaconf Loading

**Files:**
- Modify: `src/elspeth/core/config.py`
- Modify: `tests/core/test_config.py`

### Step 1: Write the failing test for settings loading

```python
# Add to tests/core/test_config.py

import tempfile
from pathlib import Path


class TestSettingsLoading:
    """Settings loading from YAML files."""

    def test_load_from_yaml(self, tmp_path: Path) -> None:
        from elspeth.core.config import load_settings

        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
default:
  landscape:
    enabled: true
    backend: sqlite
    path: ./test.db
  concurrency:
    max_workers: 8
""")
        settings = load_settings(str(config_file))
        assert settings.landscape.path == "./test.db"
        assert settings.concurrency.max_workers == 8

    def test_load_with_env_override(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from elspeth.core.config import load_settings

        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
default:
  landscape:
    path: ./default.db
""")
        # Environment variable should override file
        monkeypatch.setenv("ELSPETH_LANDSCAPE__PATH", "./from_env.db")

        settings = load_settings(str(config_file))
        assert settings.landscape.path == "./from_env.db"

    def test_load_missing_file_raises(self) -> None:
        from elspeth.core.config import load_settings

        with pytest.raises(FileNotFoundError):
            load_settings("/nonexistent/path/settings.yaml")
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/test_config.py::TestSettingsLoading -v`
Expected: FAIL (function doesn't exist)

### Step 3: Add settings loading function

```python
# Add to src/elspeth/core/config.py

from pathlib import Path
from typing import Any

from dynaconf import Dynaconf


def load_settings(
    settings_file: str,
    profile: str | None = None,
) -> ElspethSettings:
    """Load and validate settings from file.

    Args:
        settings_file: Path to YAML settings file
        profile: Optional profile name (e.g., "production")

    Returns:
        Validated ElspethSettings instance

    Raises:
        FileNotFoundError: If settings file doesn't exist
        ValidationError: If settings fail validation
    """
    path = Path(settings_file)
    if not path.exists():
        raise FileNotFoundError(f"Settings file not found: {settings_file}")

    # Load with Dynaconf
    dynaconf_settings = Dynaconf(
        settings_files=[str(path)],
        environments=True,
        env_prefix="ELSPETH",
        env_switcher="ELSPETH_ENV",
        envvar_prefix="ELSPETH",
        env=profile or "default",
        merge_enabled=True,
    )

    # Convert to dict and validate with Pydantic
    settings_dict = dynaconf_settings.as_dict()

    return ElspethSettings(**settings_dict)
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/core/test_config.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add -u
git commit -m "feat(config): add Dynaconf-based settings loading"
```

---

## Task 12: Payload Store - Protocol and Filesystem Implementation

**Files:**
- Create: `src/elspeth/core/payload_store.py`
- Create: `tests/core/test_payload_store.py`

### Step 1: Write the failing tests for PayloadStore

```python
# tests/core/test_payload_store.py
"""Tests for payload storage abstraction."""

from pathlib import Path

import pytest


class TestPayloadRef:
    """PayloadRef represents a reference to stored data."""

    def test_payload_ref_creation(self) -> None:
        from elspeth.core.payload_store import PayloadRef

        ref = PayloadRef(
            store_id="fs",
            content_hash="abc123",
            content_type="application/json",
            size_bytes=1024,
        )
        assert ref.store_id == "fs"
        assert ref.content_hash == "abc123"
        assert ref.size_bytes == 1024


class TestFilesystemPayloadStore:
    """Filesystem-based payload storage."""

    def test_put_returns_ref(self, tmp_path: Path) -> None:
        from elspeth.core.payload_store import FilesystemPayloadStore

        store = FilesystemPayloadStore(tmp_path)
        ref = store.put(b"hello world", "text/plain")

        assert ref.store_id == "filesystem"
        assert ref.content_type == "text/plain"
        assert ref.size_bytes == 11

    def test_get_returns_data(self, tmp_path: Path) -> None:
        from elspeth.core.payload_store import FilesystemPayloadStore

        store = FilesystemPayloadStore(tmp_path)
        ref = store.put(b"hello world", "text/plain")
        data = store.get(ref)

        assert data == b"hello world"

    def test_exists_true_for_stored(self, tmp_path: Path) -> None:
        from elspeth.core.payload_store import FilesystemPayloadStore

        store = FilesystemPayloadStore(tmp_path)
        ref = store.put(b"test data", "text/plain")

        assert store.exists(ref) is True

    def test_exists_false_for_missing(self, tmp_path: Path) -> None:
        from elspeth.core.payload_store import FilesystemPayloadStore, PayloadRef

        store = FilesystemPayloadStore(tmp_path)
        fake_ref = PayloadRef(
            store_id="filesystem",
            content_hash="nonexistent",
            content_type="text/plain",
            size_bytes=0,
        )

        assert store.exists(fake_ref) is False

    def test_get_missing_returns_none(self, tmp_path: Path) -> None:
        from elspeth.core.payload_store import FilesystemPayloadStore, PayloadRef

        store = FilesystemPayloadStore(tmp_path)
        fake_ref = PayloadRef(
            store_id="filesystem",
            content_hash="nonexistent",
            content_type="text/plain",
            size_bytes=0,
        )

        assert store.get(fake_ref) is None

    def test_content_addressable(self, tmp_path: Path) -> None:
        from elspeth.core.payload_store import FilesystemPayloadStore

        store = FilesystemPayloadStore(tmp_path)
        # Same content should produce same hash
        ref1 = store.put(b"identical", "text/plain")
        ref2 = store.put(b"identical", "text/plain")

        assert ref1.content_hash == ref2.content_hash
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/test_payload_store.py -v`
Expected: FAIL (ImportError)

### Step 3: Create payload store implementation

```python
# src/elspeth/core/payload_store.py
"""Payload store abstraction for large blob storage.

Separates large payloads (LLM responses, row data) from audit tables
to enable retention policies and storage optimization.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class PayloadRef:
    """Reference to a stored payload."""

    store_id: str
    content_hash: str
    content_type: str
    size_bytes: int


class PayloadStore(Protocol):
    """Protocol for payload storage backends."""

    def put(self, data: bytes, content_type: str) -> PayloadRef:
        """Store data and return a reference."""
        ...

    def get(self, ref: PayloadRef) -> bytes | None:
        """Retrieve data by reference. Returns None if not found."""
        ...

    def exists(self, ref: PayloadRef) -> bool:
        """Check if payload exists."""
        ...


class FilesystemPayloadStore:
    """Filesystem-based payload storage.

    Uses content-addressable storage where files are named by their SHA-256 hash.
    This provides natural deduplication.
    """

    store_id = "filesystem"

    def __init__(self, base_path: Path) -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _hash_content(self, data: bytes) -> str:
        """Compute SHA-256 hash of content."""
        return hashlib.sha256(data).hexdigest()

    def _path_for_hash(self, content_hash: str) -> Path:
        """Get filesystem path for a content hash.

        Uses two-level directory structure to avoid too many files in one dir.
        """
        return self.base_path / content_hash[:2] / content_hash[2:4] / content_hash

    def put(self, data: bytes, content_type: str) -> PayloadRef:
        """Store data and return a reference."""
        content_hash = self._hash_content(data)
        path = self._path_for_hash(content_hash)

        # Only write if not already present (content-addressable)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

        return PayloadRef(
            store_id=self.store_id,
            content_hash=content_hash,
            content_type=content_type,
            size_bytes=len(data),
        )

    def get(self, ref: PayloadRef) -> bytes | None:
        """Retrieve data by reference."""
        path = self._path_for_hash(ref.content_hash)
        if not path.exists():
            return None
        return path.read_bytes()

    def exists(self, ref: PayloadRef) -> bool:
        """Check if payload exists."""
        path = self._path_for_hash(ref.content_hash)
        return path.exists()
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/core/test_payload_store.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/core/payload_store.py tests/core/test_payload_store.py
git commit -m "feat(payload): add filesystem payload store with content-addressable storage"
```

---

## Task 13: DAG Validation - Graph Builder

**Files:**
- Create: `src/elspeth/core/dag.py`
- Create: `tests/core/test_dag.py`

### Step 1: Write the failing tests for DAG building

```python
# tests/core/test_dag.py
"""Tests for DAG validation and operations."""

import pytest


class TestDAGBuilder:
    """Building execution graphs from configuration."""

    def test_empty_dag(self) -> None:
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        assert graph.node_count == 0
        assert graph.edge_count == 0

    def test_add_node(self) -> None:
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("source", node_type="source", plugin_name="csv")

        assert graph.node_count == 1
        assert graph.has_node("source")

    def test_add_edge(self) -> None:
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("source", node_type="source", plugin_name="csv")
        graph.add_node("transform", node_type="transform", plugin_name="validate")
        graph.add_edge("source", "transform", label="continue")

        assert graph.edge_count == 1

    def test_linear_pipeline(self) -> None:
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("source", node_type="source", plugin_name="csv")
        graph.add_node("t1", node_type="transform", plugin_name="enrich")
        graph.add_node("t2", node_type="transform", plugin_name="classify")
        graph.add_node("sink", node_type="sink", plugin_name="csv")

        graph.add_edge("source", "t1", label="continue")
        graph.add_edge("t1", "t2", label="continue")
        graph.add_edge("t2", "sink", label="continue")

        assert graph.node_count == 4
        assert graph.edge_count == 3
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/test_dag.py::TestDAGBuilder -v`
Expected: FAIL (ImportError)

### Step 3: Create DAG builder

```python
# src/elspeth/core/dag.py
"""DAG (Directed Acyclic Graph) operations for execution planning.

Uses NetworkX for graph operations including:
- Acyclicity validation
- Topological sorting
- Path finding for lineage queries
"""

from dataclasses import dataclass, field
from typing import Any

import networkx as nx


@dataclass
class NodeInfo:
    """Information about a node in the execution graph."""

    node_id: str
    node_type: str  # source, transform, gate, aggregation, coalesce, sink
    plugin_name: str
    config: dict[str, Any] = field(default_factory=dict)


class ExecutionGraph:
    """Execution graph for pipeline configuration.

    Wraps NetworkX DiGraph with domain-specific operations.
    """

    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()

    @property
    def node_count(self) -> int:
        """Number of nodes in the graph."""
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        """Number of edges in the graph."""
        return self._graph.number_of_edges()

    def has_node(self, node_id: str) -> bool:
        """Check if node exists."""
        return self._graph.has_node(node_id)

    def add_node(
        self,
        node_id: str,
        *,
        node_type: str,
        plugin_name: str,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Add a node to the execution graph."""
        info = NodeInfo(
            node_id=node_id,
            node_type=node_type,
            plugin_name=plugin_name,
            config=config or {},
        )
        self._graph.add_node(node_id, info=info)

    def add_edge(
        self,
        from_node: str,
        to_node: str,
        *,
        label: str,
        mode: str = "move",
    ) -> None:
        """Add an edge between nodes.

        Args:
            from_node: Source node ID
            to_node: Target node ID
            label: Edge label (e.g., "continue", "suspicious")
            mode: Routing mode ("move" or "copy")
        """
        self._graph.add_edge(from_node, to_node, label=label, mode=mode)
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/core/test_dag.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/core/dag.py tests/core/test_dag.py
git commit -m "feat(dag): add ExecutionGraph with NetworkX backend"
```

---

## Task 14: DAG Validation - Acyclicity and Topological Sort

**Files:**
- Modify: `src/elspeth/core/dag.py`
- Modify: `tests/core/test_dag.py`

### Step 1: Write the failing tests for validation

```python
# Add to tests/core/test_dag.py

class TestDAGValidation:
    """Validation of execution graphs."""

    def test_is_valid_for_acyclic(self) -> None:
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("a", node_type="source", plugin_name="csv")
        graph.add_node("b", node_type="transform", plugin_name="x")
        graph.add_node("c", node_type="sink", plugin_name="csv")
        graph.add_edge("a", "b", label="continue")
        graph.add_edge("b", "c", label="continue")

        assert graph.is_acyclic() is True

    def test_is_invalid_for_cycle(self) -> None:
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("a", node_type="transform", plugin_name="x")
        graph.add_node("b", node_type="transform", plugin_name="y")
        graph.add_edge("a", "b", label="continue")
        graph.add_edge("b", "a", label="continue")  # Creates cycle!

        assert graph.is_acyclic() is False

    def test_validate_raises_on_cycle(self) -> None:
        from elspeth.core.dag import ExecutionGraph, GraphValidationError

        graph = ExecutionGraph()
        graph.add_node("a", node_type="transform", plugin_name="x")
        graph.add_node("b", node_type="transform", plugin_name="y")
        graph.add_edge("a", "b", label="continue")
        graph.add_edge("b", "a", label="continue")

        with pytest.raises(GraphValidationError, match="cycle"):
            graph.validate()

    def test_topological_order(self) -> None:
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("source", node_type="source", plugin_name="csv")
        graph.add_node("t1", node_type="transform", plugin_name="a")
        graph.add_node("t2", node_type="transform", plugin_name="b")
        graph.add_node("sink", node_type="sink", plugin_name="csv")

        graph.add_edge("source", "t1", label="continue")
        graph.add_edge("t1", "t2", label="continue")
        graph.add_edge("t2", "sink", label="continue")

        order = graph.topological_order()

        # Source must come first, sink must come last
        assert order[0] == "source"
        assert order[-1] == "sink"
        # t1 must come before t2
        assert order.index("t1") < order.index("t2")
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/test_dag.py::TestDAGValidation -v`
Expected: FAIL (methods don't exist)

### Step 3: Add validation methods

```python
# Add to src/elspeth/core/dag.py


class GraphValidationError(Exception):
    """Raised when graph validation fails."""

    pass


# Add methods to ExecutionGraph class:

    def is_acyclic(self) -> bool:
        """Check if the graph is acyclic (a valid DAG)."""
        return nx.is_directed_acyclic_graph(self._graph)

    def validate(self) -> None:
        """Validate the execution graph.

        Raises:
            GraphValidationError: If validation fails
        """
        if not self.is_acyclic():
            # Find the cycle for error message
            try:
                cycle = nx.find_cycle(self._graph)
                cycle_str = " -> ".join(f"{u}" for u, v in cycle)
                raise GraphValidationError(
                    f"Graph contains a cycle: {cycle_str}"
                )
            except nx.NetworkXNoCycle:
                # Shouldn't happen if is_acyclic() returned False
                raise GraphValidationError("Graph contains a cycle")

    def topological_order(self) -> list[str]:
        """Return nodes in topological order.

        Returns:
            List of node IDs in execution order

        Raises:
            GraphValidationError: If graph has cycles
        """
        try:
            return list(nx.topological_sort(self._graph))
        except nx.NetworkXUnfeasible as e:
            raise GraphValidationError(f"Cannot sort graph: {e}") from e
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/core/test_dag.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add -u
git commit -m "feat(dag): add acyclicity validation and topological sort"
```

---

## Task 15: DAG Validation - Source and Sink Validation

**Files:**
- Modify: `src/elspeth/core/dag.py`
- Modify: `tests/core/test_dag.py`

### Step 1: Write the failing tests for source/sink validation

```python
# Add to tests/core/test_dag.py

class TestSourceSinkValidation:
    """Validation of source and sink constraints."""

    def test_validate_requires_exactly_one_source(self) -> None:
        from elspeth.core.dag import ExecutionGraph, GraphValidationError

        graph = ExecutionGraph()
        graph.add_node("t1", node_type="transform", plugin_name="x")
        graph.add_node("sink", node_type="sink", plugin_name="csv")
        graph.add_edge("t1", "sink", label="continue")

        with pytest.raises(GraphValidationError, match="exactly one source"):
            graph.validate()

    def test_validate_requires_at_least_one_sink(self) -> None:
        from elspeth.core.dag import ExecutionGraph, GraphValidationError

        graph = ExecutionGraph()
        graph.add_node("source", node_type="source", plugin_name="csv")
        graph.add_node("t1", node_type="transform", plugin_name="x")
        graph.add_edge("source", "t1", label="continue")

        with pytest.raises(GraphValidationError, match="at least one sink"):
            graph.validate()

    def test_validate_multiple_sinks_allowed(self) -> None:
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("source", node_type="source", plugin_name="csv")
        graph.add_node("gate", node_type="gate", plugin_name="classifier")
        graph.add_node("sink1", node_type="sink", plugin_name="csv")
        graph.add_node("sink2", node_type="sink", plugin_name="csv")

        graph.add_edge("source", "gate", label="continue")
        graph.add_edge("gate", "sink1", label="normal")
        graph.add_edge("gate", "sink2", label="flagged")

        # Should not raise
        graph.validate()

    def test_get_source_node(self) -> None:
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("my_source", node_type="source", plugin_name="csv")
        graph.add_node("sink", node_type="sink", plugin_name="csv")
        graph.add_edge("my_source", "sink", label="continue")

        assert graph.get_source() == "my_source"

    def test_get_sink_nodes(self) -> None:
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("source", node_type="source", plugin_name="csv")
        graph.add_node("sink1", node_type="sink", plugin_name="csv")
        graph.add_node("sink2", node_type="sink", plugin_name="json")
        graph.add_edge("source", "sink1", label="continue")
        graph.add_edge("source", "sink2", label="continue")

        sinks = graph.get_sinks()
        assert set(sinks) == {"sink1", "sink2"}
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/test_dag.py::TestSourceSinkValidation -v`
Expected: FAIL

### Step 3: Add source/sink validation

```python
# Add methods to ExecutionGraph class in src/elspeth/core/dag.py

    def get_source(self) -> str | None:
        """Get the source node ID."""
        sources = [
            node_id
            for node_id, data in self._graph.nodes(data=True)
            if data.get("info") and data["info"].node_type == "source"
        ]
        return sources[0] if len(sources) == 1 else None

    def get_sinks(self) -> list[str]:
        """Get all sink node IDs."""
        return [
            node_id
            for node_id, data in self._graph.nodes(data=True)
            if data.get("info") and data["info"].node_type == "sink"
        ]

    def validate(self) -> None:
        """Validate the execution graph.

        Raises:
            GraphValidationError: If validation fails
        """
        # Check for cycles
        if not self.is_acyclic():
            try:
                cycle = nx.find_cycle(self._graph)
                cycle_str = " -> ".join(f"{u}" for u, v in cycle)
                raise GraphValidationError(
                    f"Graph contains a cycle: {cycle_str}"
                )
            except nx.NetworkXNoCycle:
                raise GraphValidationError("Graph contains a cycle")

        # Check for exactly one source
        sources = [
            node_id
            for node_id, data in self._graph.nodes(data=True)
            if data.get("info") and data["info"].node_type == "source"
        ]
        if len(sources) != 1:
            raise GraphValidationError(
                f"Graph must have exactly one source, found {len(sources)}"
            )

        # Check for at least one sink
        sinks = self.get_sinks()
        if len(sinks) < 1:
            raise GraphValidationError(
                "Graph must have at least one sink"
            )
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/core/test_dag.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add -u
git commit -m "feat(dag): validate source/sink constraints"
```

---

## Task 16: Landscape - Database Schema Models

**Files:**
- Create: `src/elspeth/core/landscape/models.py`
- Create: `tests/core/landscape/test_models.py`

### Step 1: Write the failing tests for schema models

```python
# tests/core/landscape/test_models.py
"""Tests for Landscape database models."""

from datetime import datetime, timezone

import pytest


class TestRunModel:
    """Run table model."""

    def test_create_run(self) -> None:
        from elspeth.core.landscape.models import Run

        run = Run(
            run_id="run-001",
            started_at=datetime.now(timezone.utc),
            config_hash="abc123",
            settings_json="{}",
            canonical_version="sha256-rfc8785-v1",
            status="running",
        )
        assert run.run_id == "run-001"
        assert run.status == "running"


class TestNodeModel:
    """Node table model."""

    def test_create_node(self) -> None:
        from elspeth.core.landscape.models import Node

        node = Node(
            node_id="node-001",
            run_id="run-001",
            plugin_name="csv",
            node_type="source",
            plugin_version="1.0.0",
            config_hash="def456",
            config_json="{}",
            registered_at=datetime.now(timezone.utc),
        )
        assert node.node_type == "source"


class TestRowModel:
    """Row table model."""

    def test_create_row(self) -> None:
        from elspeth.core.landscape.models import Row

        row = Row(
            row_id="row-001",
            run_id="run-001",
            source_node_id="source-001",
            row_index=0,
            source_data_hash="ghi789",
            created_at=datetime.now(timezone.utc),
        )
        assert row.row_index == 0


class TestTokenModel:
    """Token table model."""

    def test_create_token(self) -> None:
        from elspeth.core.landscape.models import Token

        token = Token(
            token_id="token-001",
            row_id="row-001",
            created_at=datetime.now(timezone.utc),
        )
        assert token.token_id == "token-001"
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/landscape/test_models.py -v`
Expected: FAIL (ImportError)

### Step 3: Create tests directory and models

```bash
mkdir -p tests/core/landscape
touch tests/core/landscape/__init__.py
```

```python
# src/elspeth/core/landscape/models.py
"""SQLAlchemy models for Landscape audit tables.

These models define the schema for tracking:
- Runs and their configuration
- Nodes (plugin instances) in the execution graph
- Rows loaded from sources
- Tokens (row instances flowing through DAG paths)
- Node states (what happened at each node for each token)
- External calls
- Artifacts produced by sinks
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Run:
    """A single execution of a pipeline."""

    run_id: str
    started_at: datetime
    config_hash: str
    settings_json: str
    canonical_version: str
    status: str  # running, completed, failed
    completed_at: datetime | None = None
    reproducibility_grade: str | None = None


@dataclass
class Node:
    """A node (plugin instance) in the execution graph."""

    node_id: str
    run_id: str
    plugin_name: str
    node_type: str  # source, transform, gate, aggregation, coalesce, sink
    plugin_version: str
    config_hash: str
    config_json: str
    registered_at: datetime
    schema_hash: str | None = None
    sequence_in_pipeline: int | None = None


@dataclass
class Edge:
    """An edge in the execution graph."""

    edge_id: str
    run_id: str
    from_node_id: str
    to_node_id: str
    label: str  # "continue", route name, etc.
    default_mode: str  # "move" or "copy"
    created_at: datetime


@dataclass
class Row:
    """A source row loaded into the system."""

    row_id: str
    run_id: str
    source_node_id: str
    row_index: int
    source_data_hash: str
    created_at: datetime
    source_data_ref: str | None = None  # Payload store reference


@dataclass
class Token:
    """A row instance flowing through a specific DAG path."""

    token_id: str
    row_id: str
    created_at: datetime
    fork_group_id: str | None = None
    join_group_id: str | None = None
    branch_name: str | None = None


@dataclass
class TokenParent:
    """Parent relationship for tokens (supports multi-parent joins)."""

    token_id: str
    parent_token_id: str
    ordinal: int


@dataclass
class NodeState:
    """What happened when a token visited a node."""

    state_id: str
    token_id: str
    node_id: str
    step_index: int
    attempt: int
    status: str  # open, completed, failed
    input_hash: str
    started_at: datetime
    output_hash: str | None = None
    context_before_json: str | None = None
    context_after_json: str | None = None
    duration_ms: float | None = None
    error_json: str | None = None
    completed_at: datetime | None = None


@dataclass
class Call:
    """An external call made during node processing."""

    call_id: str
    state_id: str
    call_index: int
    call_type: str  # llm, http, sql, filesystem
    status: str  # success, error
    request_hash: str
    created_at: datetime
    request_ref: str | None = None
    response_hash: str | None = None
    response_ref: str | None = None
    error_json: str | None = None
    latency_ms: float | None = None


@dataclass
class Artifact:
    """An artifact produced by a sink."""

    artifact_id: str
    run_id: str
    produced_by_state_id: str
    sink_node_id: str
    artifact_type: str
    path_or_uri: str
    content_hash: str
    size_bytes: int
    created_at: datetime
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/core/landscape/test_models.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/core/landscape/models.py tests/core/landscape/
git commit -m "feat(landscape): add dataclass models for audit tables"
```

---

## Task 17: Landscape - SQLAlchemy Table Definitions

**Files:**
- Create: `src/elspeth/core/landscape/schema.py`
- Create: `tests/core/landscape/test_schema.py`

### Step 1: Write the failing tests for SQLAlchemy schema

```python
# tests/core/landscape/test_schema.py
"""Tests for Landscape SQLAlchemy schema."""

import pytest
from sqlalchemy import inspect


class TestSchemaDefinition:
    """SQLAlchemy table definitions."""

    def test_runs_table_exists(self) -> None:
        from elspeth.core.landscape.schema import metadata, runs_table

        assert runs_table.name == "runs"
        assert "run_id" in [c.name for c in runs_table.columns]

    def test_nodes_table_exists(self) -> None:
        from elspeth.core.landscape.schema import nodes_table

        assert nodes_table.name == "nodes"
        assert "node_id" in [c.name for c in nodes_table.columns]

    def test_rows_table_exists(self) -> None:
        from elspeth.core.landscape.schema import rows_table

        assert rows_table.name == "rows"
        assert "row_id" in [c.name for c in rows_table.columns]

    def test_tokens_table_exists(self) -> None:
        from elspeth.core.landscape.schema import tokens_table

        assert tokens_table.name == "tokens"

    def test_node_states_table_exists(self) -> None:
        from elspeth.core.landscape.schema import node_states_table

        assert node_states_table.name == "node_states"


class TestSchemaCreation:
    """Creating tables in a database."""

    def test_create_all_tables(self, tmp_path) -> None:
        from sqlalchemy import create_engine

        from elspeth.core.landscape.schema import metadata

        db_path = tmp_path / "test.db"
        engine = create_engine(f"sqlite:///{db_path}")

        metadata.create_all(engine)

        # Verify tables exist
        inspector = inspect(engine)
        tables = inspector.get_table_names()

        assert "runs" in tables
        assert "nodes" in tables
        assert "rows" in tables
        assert "tokens" in tables
        assert "node_states" in tables
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/landscape/test_schema.py -v`
Expected: FAIL (ImportError)

### Step 3: Create SQLAlchemy schema

```python
# src/elspeth/core/landscape/schema.py
"""SQLAlchemy table definitions for Landscape.

Uses SQLAlchemy Core (not ORM) for explicit control over queries
and compatibility with multiple database backends.
"""

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
)

# Shared metadata for all tables
metadata = MetaData()

# === Runs and Configuration ===

runs_table = Table(
    "runs",
    metadata,
    Column("run_id", String(64), primary_key=True),
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("completed_at", DateTime(timezone=True)),
    Column("config_hash", String(64), nullable=False),
    Column("settings_json", Text, nullable=False),
    Column("reproducibility_grade", String(32)),
    Column("canonical_version", String(64), nullable=False),
    Column("status", String(32), nullable=False),
)

# === Nodes (Plugin Instances) ===

nodes_table = Table(
    "nodes",
    metadata,
    Column("node_id", String(64), primary_key=True),
    Column("run_id", String(64), ForeignKey("runs.run_id"), nullable=False),
    Column("plugin_name", String(128), nullable=False),
    Column("node_type", String(32), nullable=False),
    Column("plugin_version", String(32), nullable=False),
    Column("config_hash", String(64), nullable=False),
    Column("config_json", Text, nullable=False),
    Column("schema_hash", String(64)),
    Column("sequence_in_pipeline", Integer),
    Column("registered_at", DateTime(timezone=True), nullable=False),
)

# === Edges ===

edges_table = Table(
    "edges",
    metadata,
    Column("edge_id", String(64), primary_key=True),
    Column("run_id", String(64), ForeignKey("runs.run_id"), nullable=False),
    Column("from_node_id", String(64), ForeignKey("nodes.node_id"), nullable=False),
    Column("to_node_id", String(64), ForeignKey("nodes.node_id"), nullable=False),
    Column("label", String(64), nullable=False),
    Column("default_mode", String(16), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("run_id", "from_node_id", "label"),
)

# === Source Rows ===

rows_table = Table(
    "rows",
    metadata,
    Column("row_id", String(64), primary_key=True),
    Column("run_id", String(64), ForeignKey("runs.run_id"), nullable=False),
    Column("source_node_id", String(64), ForeignKey("nodes.node_id"), nullable=False),
    Column("row_index", Integer, nullable=False),
    Column("source_data_hash", String(64), nullable=False),
    Column("source_data_ref", String(256)),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("run_id", "row_index"),
)

# === Tokens ===

tokens_table = Table(
    "tokens",
    metadata,
    Column("token_id", String(64), primary_key=True),
    Column("row_id", String(64), ForeignKey("rows.row_id"), nullable=False),
    Column("fork_group_id", String(64)),
    Column("join_group_id", String(64)),
    Column("branch_name", String(64)),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

# === Token Parents (for multi-parent joins) ===

token_parents_table = Table(
    "token_parents",
    metadata,
    Column(
        "token_id", String(64), ForeignKey("tokens.token_id"), primary_key=True
    ),
    Column(
        "parent_token_id",
        String(64),
        ForeignKey("tokens.token_id"),
        primary_key=True,
    ),
    Column("ordinal", Integer, nullable=False),
    UniqueConstraint("token_id", "ordinal"),
)

# === Node States ===

node_states_table = Table(
    "node_states",
    metadata,
    Column("state_id", String(64), primary_key=True),
    Column("token_id", String(64), ForeignKey("tokens.token_id"), nullable=False),
    Column("node_id", String(64), ForeignKey("nodes.node_id"), nullable=False),
    Column("step_index", Integer, nullable=False),
    Column("attempt", Integer, nullable=False, default=0),
    Column("status", String(32), nullable=False),
    Column("input_hash", String(64), nullable=False),
    Column("output_hash", String(64)),
    Column("context_before_json", Text),
    Column("context_after_json", Text),
    Column("duration_ms", Float),
    Column("error_json", Text),
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("completed_at", DateTime(timezone=True)),
    UniqueConstraint("token_id", "node_id", "attempt"),
    UniqueConstraint("token_id", "step_index"),
)

# === External Calls ===

calls_table = Table(
    "calls",
    metadata,
    Column("call_id", String(64), primary_key=True),
    Column(
        "state_id", String(64), ForeignKey("node_states.state_id"), nullable=False
    ),
    Column("call_index", Integer, nullable=False),
    Column("call_type", String(32), nullable=False),
    Column("status", String(32), nullable=False),
    Column("request_hash", String(64), nullable=False),
    Column("request_ref", String(256)),
    Column("response_hash", String(64)),
    Column("response_ref", String(256)),
    Column("error_json", Text),
    Column("latency_ms", Float),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("state_id", "call_index"),
)

# === Artifacts ===

artifacts_table = Table(
    "artifacts",
    metadata,
    Column("artifact_id", String(64), primary_key=True),
    Column("run_id", String(64), ForeignKey("runs.run_id"), nullable=False),
    Column(
        "produced_by_state_id",
        String(64),
        ForeignKey("node_states.state_id"),
        nullable=False,
    ),
    Column("sink_node_id", String(64), ForeignKey("nodes.node_id"), nullable=False),
    Column("artifact_type", String(64), nullable=False),
    Column("path_or_uri", String(512), nullable=False),
    Column("content_hash", String(64), nullable=False),
    Column("size_bytes", Integer, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/core/landscape/test_schema.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/core/landscape/schema.py tests/core/landscape/test_schema.py
git commit -m "feat(landscape): add SQLAlchemy table definitions"
```

---

## Task 18: Landscape - Database Connection Management

**Files:**
- Create: `src/elspeth/core/landscape/database.py`
- Create: `tests/core/landscape/test_database.py`

### Step 1: Write the failing tests for database connection

```python
# tests/core/landscape/test_database.py
"""Tests for Landscape database connection management."""

from pathlib import Path

import pytest


class TestDatabaseConnection:
    """Database connection and initialization."""

    def test_connect_creates_tables(self, tmp_path: Path) -> None:
        from sqlalchemy import inspect

        from elspeth.core.landscape.database import LandscapeDB

        db_path = tmp_path / "landscape.db"
        db = LandscapeDB(f"sqlite:///{db_path}")

        # Tables should be created
        inspector = inspect(db.engine)
        tables = inspector.get_table_names()

        assert "runs" in tables
        assert "nodes" in tables

    def test_sqlite_wal_mode(self, tmp_path: Path) -> None:
        from elspeth.core.landscape.database import LandscapeDB

        db_path = tmp_path / "landscape.db"
        db = LandscapeDB(f"sqlite:///{db_path}")

        # Check WAL mode is enabled
        with db.engine.connect() as conn:
            result = conn.execute(
                db.engine.dialect.compiler(db.engine.dialect, None)
                ._literal_execute_expanding_bind_from_text("PRAGMA journal_mode")
                if hasattr(db.engine.dialect.compiler, "_literal_execute_expanding_bind_from_text")
                else conn.exec_driver_sql("PRAGMA journal_mode")
            )
            mode = result.scalar()
            # WAL mode should be enabled for SQLite
            assert mode == "wal"

    def test_context_manager(self, tmp_path: Path) -> None:
        from elspeth.core.landscape.database import LandscapeDB

        db_path = tmp_path / "landscape.db"

        with LandscapeDB(f"sqlite:///{db_path}") as db:
            assert db.engine is not None
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/landscape/test_database.py -v`
Expected: FAIL (ImportError)

### Step 3: Create database connection management

```python
# src/elspeth/core/landscape/database.py
"""Database connection management for Landscape.

Handles SQLite (development) and PostgreSQL (production) backends
with appropriate settings for each.
"""

from typing import Self

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine

from elspeth.core.landscape.schema import metadata


class LandscapeDB:
    """Landscape database connection manager."""

    def __init__(self, connection_string: str) -> None:
        """Initialize database connection.

        Args:
            connection_string: SQLAlchemy connection string
                e.g., "sqlite:///./runs/landscape.db"
                      "postgresql://user:pass@host/dbname"
        """
        self.connection_string = connection_string
        self._engine: Engine | None = None
        self._setup_engine()
        self._create_tables()

    def _setup_engine(self) -> None:
        """Create and configure the database engine."""
        self._engine = create_engine(
            self.connection_string,
            echo=False,  # Set True for SQL debugging
        )

        # SQLite-specific configuration
        if self.connection_string.startswith("sqlite"):
            self._configure_sqlite()

    def _configure_sqlite(self) -> None:
        """Configure SQLite for reliability."""

        @event.listens_for(self._engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):  # type: ignore[no-untyped-def]
            cursor = dbapi_connection.cursor()
            # Enable WAL mode for better concurrency
            cursor.execute("PRAGMA journal_mode=WAL")
            # Enable foreign key enforcement
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    def _create_tables(self) -> None:
        """Create all tables if they don't exist."""
        metadata.create_all(self.engine)

    @property
    def engine(self) -> Engine:
        """Get the SQLAlchemy engine."""
        if self._engine is None:
            raise RuntimeError("Database not initialized")
        return self._engine

    def close(self) -> None:
        """Close database connection."""
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[no-untyped-def]
        self.close()
```

### Step 4: Update test to use correct SQLite pragma check

```python
# Update tests/core/landscape/test_database.py

class TestDatabaseConnection:
    """Database connection and initialization."""

    def test_connect_creates_tables(self, tmp_path: Path) -> None:
        from sqlalchemy import inspect

        from elspeth.core.landscape.database import LandscapeDB

        db_path = tmp_path / "landscape.db"
        db = LandscapeDB(f"sqlite:///{db_path}")

        inspector = inspect(db.engine)
        tables = inspector.get_table_names()

        assert "runs" in tables
        assert "nodes" in tables

    def test_sqlite_wal_mode(self, tmp_path: Path) -> None:
        from sqlalchemy import text

        from elspeth.core.landscape.database import LandscapeDB

        db_path = tmp_path / "landscape.db"
        db = LandscapeDB(f"sqlite:///{db_path}")

        with db.engine.connect() as conn:
            result = conn.execute(text("PRAGMA journal_mode"))
            mode = result.scalar()
            assert mode == "wal"

    def test_context_manager(self, tmp_path: Path) -> None:
        from elspeth.core.landscape.database import LandscapeDB

        db_path = tmp_path / "landscape.db"

        with LandscapeDB(f"sqlite:///{db_path}") as db:
            assert db.engine is not None
```

### Step 5: Run tests to verify they pass

Run: `pytest tests/core/landscape/test_database.py -v`
Expected: PASS

### Step 6: Commit

```bash
git add src/elspeth/core/landscape/database.py tests/core/landscape/test_database.py
git commit -m "feat(landscape): add database connection with WAL mode"
```

---

## Task 19: Final Integration - Core Exports

**Files:**
- Modify: `src/elspeth/core/__init__.py`
- Modify: `src/elspeth/core/landscape/__init__.py`

### Step 1: Write the failing test for public API

```python
# Add to tests/core/test_canonical.py

class TestCoreIntegration:
    """Core module integration."""

    def test_dag_importable_from_core(self) -> None:
        from elspeth.core import ExecutionGraph, GraphValidationError

        assert ExecutionGraph is not None
        assert GraphValidationError is not None

    def test_config_importable_from_core(self) -> None:
        from elspeth.core import ElspethSettings, load_settings

        assert ElspethSettings is not None
        assert callable(load_settings)

    def test_payload_store_importable_from_core(self) -> None:
        from elspeth.core import FilesystemPayloadStore, PayloadRef

        assert FilesystemPayloadStore is not None
        assert PayloadRef is not None
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/test_canonical.py::TestCoreIntegration -v`
Expected: FAIL (ImportError)

### Step 3: Update core exports

```python
# src/elspeth/core/__init__.py
"""Core infrastructure: Landscape, Canonical, Configuration, DAG."""

from elspeth.core.canonical import (
    CANONICAL_VERSION,
    canonical_json,
    stable_hash,
)
from elspeth.core.config import (
    ConcurrencySettings,
    ElspethSettings,
    LandscapeSettings,
    RetentionSettings,
    load_settings,
)
from elspeth.core.dag import (
    ExecutionGraph,
    GraphValidationError,
    NodeInfo,
)
from elspeth.core.payload_store import (
    FilesystemPayloadStore,
    PayloadRef,
    PayloadStore,
)

__all__ = [
    # Canonical
    "CANONICAL_VERSION",
    "canonical_json",
    "stable_hash",
    # Config
    "ConcurrencySettings",
    "ElspethSettings",
    "LandscapeSettings",
    "RetentionSettings",
    "load_settings",
    # DAG
    "ExecutionGraph",
    "GraphValidationError",
    "NodeInfo",
    # Payload Store
    "FilesystemPayloadStore",
    "PayloadRef",
    "PayloadStore",
]
```

```python
# src/elspeth/core/landscape/__init__.py
"""Landscape: The audit backbone for complete traceability."""

from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.models import (
    Artifact,
    Call,
    Edge,
    Node,
    NodeState,
    Row,
    Run,
    Token,
    TokenParent,
)
from elspeth.core.landscape.schema import metadata

__all__ = [
    # Database
    "LandscapeDB",
    "metadata",
    # Models
    "Artifact",
    "Call",
    "Edge",
    "Node",
    "NodeState",
    "Row",
    "Run",
    "Token",
    "TokenParent",
]
```

### Step 4: Run all tests to verify they pass

Run: `pytest tests/ -v`
Expected: PASS (all tests)

### Step 5: Commit

```bash
git add -u
git commit -m "feat(core): export all Phase 1 components from core module"
```

---

## Task 20: Final Verification

### Step 1: Run all tests with coverage

Run: `pytest tests/ -v --cov=src/elspeth --cov-report=term-missing`
Expected: PASS with good coverage

### Step 2: Run type checking

Run: `mypy src/elspeth/`
Expected: PASS (or known issues only)

### Step 3: Run linting

Run: `ruff check src/ tests/`
Expected: PASS

### Step 4: Final commit

```bash
git add -u
git commit -m "chore: Phase 1 Foundation complete"
```

---

## Summary

Phase 1 implements the foundational infrastructure:

| Component | Files | Purpose |
|-----------|-------|---------|
| **Canonical JSON** | `core/canonical.py` | Deterministic hashing with RFC 8785 |
| **Configuration** | `core/config.py` | Dynaconf + Pydantic settings |
| **Payload Store** | `core/payload_store.py` | Content-addressable blob storage |
| **DAG** | `core/dag.py` | NetworkX-based graph validation |
| **Landscape** | `core/landscape/` | Audit database schema and connection |

All components are tested, typed, and exported from `elspeth.core`.
