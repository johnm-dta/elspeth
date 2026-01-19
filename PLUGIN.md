# ELSPETH Plugin Development Guide

This document describes how to develop plugins for ELSPETH and how to verify they honor their protocol contracts.

## Plugin Types

ELSPETH has three core plugin types that form the Sense/Decide/Act pipeline:

| Plugin Type | Protocol | Purpose |
|-------------|----------|---------|
| **Source** | `SourceProtocol` | Load data into the system (exactly 1 per run) |
| **Transform** | `TransformProtocol` | Process/classify rows (0+ per run) |
| **Sink** | `SinkProtocol` | Output results (1+ per run) |

```
SENSE (Source) → DECIDE (Transform/Gate) → ACT (Sink)
```

## The Trust Model and Plugins

**CRITICAL**: All plugins are **system-owned code**, not user-provided extensions. This means:

- Plugin bugs are **system bugs** that crash immediately (not silently handled)
- Plugins follow the Three-Tier Trust Model:
  - **Sources** receive external data (Tier 3 - Zero Trust) and may coerce types
  - **Transforms/Sinks** receive pipeline data (Tier 2 - Elevated Trust) and must NOT coerce

| Plugin Type | Coercion Allowed? | Why |
|-------------|-------------------|-----|
| **Source** | ✅ Yes | Normalizes external data at ingestion boundary |
| **Transform** | ❌ No | Wrong types = upstream bug → crash |
| **Sink** | ❌ No | Wrong types = upstream bug → crash |

## Contract Testing for Plugins

Every plugin MUST pass its protocol contract tests. The contract test framework verifies interface guarantees automatically.

### Quick Start: Testing a New Plugin

```python
# tests/contracts/source_contracts/test_my_source_contract.py
from pathlib import Path
import pytest
from .test_source_protocol import SourceContractPropertyTestBase
from elspeth.plugins.sources.my_source import MySource

class TestMySourceContract(SourceContractPropertyTestBase):
    """Contract tests for MySource plugin."""

    @pytest.fixture
    def source_data(self, tmp_path: Path) -> Path:
        """Create test data for the source."""
        data_file = tmp_path / "test_data.json"
        data_file.write_text('{"id": 1, "name": "test"}')
        return data_file

    @pytest.fixture
    def source(self, source_data: Path):
        """Create a configured source instance."""
        return MySource({
            "path": str(source_data),
            "schema": {"fields": "dynamic"},
            "on_validation_failure": "discard",
        })

    # All protocol contract tests are inherited automatically!
    # Add plugin-specific tests below if needed.
```

That's it. Your plugin now has **14+ contract tests** inherited from the base class.

### Contract Test Base Classes

| Base Class | Location | Inherited Tests |
|------------|----------|-----------------|
| `SourceContractTestBase` | `tests/contracts/source_contracts/test_source_protocol.py` | 14 tests |
| `SourceContractPropertyTestBase` | Same file | 14 + property tests |
| `TransformContractTestBase` | `tests/contracts/transform_contracts/test_transform_protocol.py` | 15 tests |
| `TransformContractPropertyTestBase` | Same file | 15 + property tests |
| `SinkContractTestBase` | `tests/contracts/sink_contracts/test_sink_protocol.py` | 17 tests |
| `SinkDeterminismContractTestBase` | Same file | 17 + determinism tests |

### Required Fixtures by Plugin Type

#### Source Plugins

```python
class TestMySourceContract(SourceContractPropertyTestBase):

    @pytest.fixture
    def source(self) -> SourceProtocol:
        """REQUIRED: Return a configured source instance."""
        return MySource({...})

    # Optional: ctx fixture is provided by base class
    # Override if you need custom context
```

#### Transform Plugins

```python
class TestMyTransformContract(TransformContractPropertyTestBase):

    @pytest.fixture
    def transform(self) -> TransformProtocol:
        """REQUIRED: Return a configured transform instance."""
        return MyTransform({...})

    @pytest.fixture
    def valid_input(self) -> dict:
        """REQUIRED: Return input that should process successfully."""
        return {"id": 1, "name": "test"}

    # Optional: ctx fixture is provided by base class
```

#### Sink Plugins

```python
class TestMySinkContract(SinkDeterminismContractTestBase):

    @pytest.fixture
    def sink(self, tmp_path: Path) -> SinkProtocol:
        """REQUIRED: Return a configured sink instance."""
        return MySink({
            "path": str(tmp_path / "output.csv"),
            ...
        })

    @pytest.fixture
    def sample_rows(self) -> list[dict]:
        """REQUIRED: Return sample rows to write."""
        return [{"id": 1}, {"id": 2}]

    # Optional: ctx fixture is provided by base class
```

## Protocol Contracts

### Source Protocol Contracts

A valid source implementation MUST:

| Contract | Verified By |
|----------|-------------|
| Have `name` attribute (non-empty string) | `test_source_has_name` |
| Have `output_schema` attribute (class type) | `test_source_has_output_schema` |
| Have `determinism` attribute (Determinism enum) | `test_source_has_determinism` |
| Have `plugin_version` attribute (string) | `test_source_has_plugin_version` |
| `load()` returns an iterator | `test_load_returns_iterator` |
| `load()` yields `SourceRow` objects only | `test_load_yields_source_rows` |
| Valid `SourceRow` has non-None `.row` dict | `test_valid_rows_have_data` |
| Quarantined `SourceRow` has `.quarantine_error` | `test_quarantined_rows_have_error` |
| Quarantined `SourceRow` has `.quarantine_destination` | `test_quarantined_rows_have_destination` |
| `close()` is idempotent (safe to call multiple times) | `test_close_is_idempotent` |
| `on_start()` does not raise | `test_on_start_does_not_raise` |
| `on_complete()` does not raise | `test_on_complete_does_not_raise` |

**Common Mistakes**:
```python
# WRONG - yielding raw dict
def load(self, ctx):
    for row in data:
        yield row  # ❌ Must be SourceRow!

# CORRECT - wrapping in SourceRow
def load(self, ctx):
    for row in data:
        yield SourceRow.valid(row)  # ✅
```

### Transform Protocol Contracts

A valid transform implementation MUST:

| Contract | Verified By |
|----------|-------------|
| Have `name` attribute | `test_transform_has_name` |
| Have `input_schema` attribute | `test_transform_has_input_schema` |
| Have `output_schema` attribute | `test_transform_has_output_schema` |
| Have `determinism` attribute | `test_transform_has_determinism` |
| Have `plugin_version` attribute | `test_transform_has_plugin_version` |
| Have `is_batch_aware` attribute (bool) | `test_transform_has_batch_awareness_flag` |
| Have `creates_tokens` attribute (bool) | `test_transform_has_creates_tokens_flag` |
| `process()` returns `TransformResult` | `test_process_returns_transform_result` |
| `TransformResult` has `.status` ("success" or "error") | `test_success_result_has_status` |
| Success result has output data (`.row` or `.rows`) | `test_success_result_has_output_data` |
| Success `.row` is a dict | `test_success_single_row_is_dict` |
| Success `.rows` is a list of dicts | `test_success_multi_row_is_list` |
| `close()` is idempotent | `test_close_is_idempotent` |

**Common Mistakes**:
```python
# WRONG - returning raw dict
def process(self, row, ctx):
    return {"result": row["value"] * 2}  # ❌ Must be TransformResult!

# CORRECT - using factory method
def process(self, row, ctx):
    return TransformResult.success({"result": row["value"] * 2})  # ✅
```

### Sink Protocol Contracts

A valid sink implementation MUST:

| Contract | Verified By |
|----------|-------------|
| Have `name` attribute | `test_sink_has_name` |
| Have `input_schema` attribute | `test_sink_has_input_schema` |
| Have `determinism` attribute | `test_sink_has_determinism` |
| Have `plugin_version` attribute | `test_sink_has_plugin_version` |
| Have `idempotent` attribute (bool) | `test_sink_has_idempotent_flag` |
| `write()` returns `ArtifactDescriptor` | `test_write_returns_artifact_descriptor` |
| `ArtifactDescriptor.content_hash` is not None | `test_artifact_has_content_hash` |
| `content_hash` is valid SHA-256 (64 hex chars) | `test_content_hash_is_sha256_hex` |
| `ArtifactDescriptor.size_bytes` is not None | `test_artifact_has_size_bytes` |
| `ArtifactDescriptor.artifact_type` is valid | `test_artifact_has_artifact_type` |
| `ArtifactDescriptor.path_or_uri` is not empty | `test_artifact_has_path_or_uri` |
| `write([])` returns valid descriptor | `test_write_empty_batch_returns_descriptor` |
| `flush()` is idempotent | `test_flush_is_idempotent` |
| `close()` is idempotent | `test_close_is_idempotent` |
| Same data produces same `content_hash` | `test_same_data_same_hash` |

**Critical for Audit Integrity**: The `content_hash` contract is non-negotiable. If the same data produces different hashes, the audit trail cannot be verified.

```python
# WRONG - no content hash
def write(self, rows, ctx):
    self._write_rows(rows)
    return ArtifactDescriptor(...)  # ❌ Missing content_hash!

# CORRECT - always compute content hash
def write(self, rows, ctx):
    self._write_rows(rows)
    content_hash = self._compute_sha256()
    return ArtifactDescriptor.for_file(
        path=str(self._path),
        content_hash=content_hash,  # ✅ REQUIRED
        size_bytes=self._path.stat().st_size,
    )
```

## Running Contract Tests

```bash
# Run all contract tests
.venv/bin/python -m pytest tests/contracts/source_contracts/ \
    tests/contracts/transform_contracts/ \
    tests/contracts/sink_contracts/ -v

# Run tests for a specific plugin
.venv/bin/python -m pytest tests/contracts/source_contracts/test_csv_source_contract.py -v

# Run with Hypothesis nightly profile (more examples)
HYPOTHESIS_PROFILE=nightly .venv/bin/python -m pytest tests/contracts/ -v
```

## Adding Plugin-Specific Tests

After inheriting the base contract tests, add tests for behavior specific to your plugin:

```python
class TestMySourceContract(SourceContractPropertyTestBase):
    # ... fixtures ...

    # Plugin-specific tests
    def test_my_source_handles_encoding(self, tmp_path):
        """MySource MUST handle UTF-8 encoding correctly."""
        # Create test file with special characters
        data_file = tmp_path / "unicode.csv"
        data_file.write_text("name\n日本語\némoji 🎉\n", encoding="utf-8")

        source = MySource({"path": str(data_file), ...})
        ctx = PluginContext(run_id="test", config={})

        rows = list(source.load(ctx))
        assert len(rows) == 2
        assert rows[0].row["name"] == "日本語"

    def test_my_source_respects_config_option(self, source, ctx):
        """MySource MUST respect the 'skip_rows' config option."""
        # Test plugin-specific configuration
        ...
```

## Property-Based Contract Tests

Some contract tests use Hypothesis for property-based verification:

```python
class TestMyTransformContract(TransformContractPropertyTestBase):
    # ... fixtures ...

    @given(
        data=st.dictionaries(
            keys=st.text(min_size=1, max_size=20),
            values=st.integers(),
            min_size=1,
        )
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_my_transform_handles_any_dict(self, transform, ctx, data):
        """Property: MyTransform handles any valid dict input."""
        result = transform.process(data, ctx)
        assert isinstance(result, TransformResult)
```

**Note**: When using `@given` with pytest fixtures, add `suppress_health_check=[HealthCheck.function_scoped_fixture]` to the `@settings` decorator.

## Checklist for New Plugins

Before submitting a new plugin:

- [ ] Plugin has all required protocol attributes (`name`, `*_schema`, `determinism`, `plugin_version`)
- [ ] Contract test class created inheriting appropriate base class
- [ ] All inherited contract tests pass
- [ ] Plugin-specific behavior has additional tests
- [ ] Sources yield `SourceRow.valid()` or `SourceRow.quarantined()`
- [ ] Transforms return `TransformResult.success()` or `TransformResult.error()`
- [ ] Sinks return `ArtifactDescriptor` with valid `content_hash`
- [ ] `close()` method is idempotent
- [ ] No type coercion in transforms/sinks (sources only)

## Example: Complete Plugin Test File

```python
# tests/contracts/source_contracts/test_json_source_contract.py
"""Contract tests for JSONSource plugin."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from elspeth.plugins.sources.json_source import JSONSource

from .test_source_protocol import SourceContractPropertyTestBase

if TYPE_CHECKING:
    from elspeth.plugins.protocols import SourceProtocol


class TestJSONSourceContract(SourceContractPropertyTestBase):
    """Contract tests for JSONSource."""

    @pytest.fixture
    def source_data(self, tmp_path: Path) -> Path:
        """Create a test JSON file."""
        json_file = tmp_path / "test_data.json"
        json_file.write_text('[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]')
        return json_file

    @pytest.fixture
    def source(self, source_data: Path) -> "SourceProtocol":
        """Create a JSONSource instance."""
        return JSONSource({
            "path": str(source_data),
            "schema": {"fields": "dynamic"},
            "on_validation_failure": "discard",
        })

    # Inherits 14+ contract tests from SourceContractPropertyTestBase

    # Plugin-specific tests
    def test_json_source_handles_nested_objects(self, tmp_path: Path) -> None:
        """JSONSource MUST handle nested JSON objects."""
        from elspeth.plugins.context import PluginContext

        nested_file = tmp_path / "nested.json"
        nested_file.write_text('[{"id": 1, "meta": {"created": "2024-01-01"}}]')

        source = JSONSource({
            "path": str(nested_file),
            "schema": {"fields": "dynamic"},
            "on_validation_failure": "discard",
        })
        ctx = PluginContext(run_id="test", config={})

        rows = list(source.load(ctx))
        assert len(rows) == 1
        assert rows[0].row["meta"]["created"] == "2024-01-01"
```

## Further Reading

- `TEST_SYSTEM.md` - Complete test system documentation
- `CLAUDE.md` - Project overview and Three-Tier Trust Model
- `docs/plans/2026-01-20-world-class-test-regime.md` - Test regime proposal
- `src/elspeth/plugins/protocols.py` - Protocol definitions
- `src/elspeth/contracts/results.py` - Result type definitions
