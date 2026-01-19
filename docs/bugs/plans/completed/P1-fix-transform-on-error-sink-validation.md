# Implementation Plan: Validate Transform on_error Sink Destinations

**Bug:** P1-2026-01-19-transform-on-error-sink-not-validated.md
**Estimated Time:** 2-3 hours
**Complexity:** Low
**Risk:** Low (fail-fast improvement, follows existing pattern)

## Summary

Transforms can specify `on_error: "sink_name"` to route errors, but unlike gate routes, this destination is not validated at pipeline startup. If the sink doesn't exist, you get a `KeyError` mid-run. This plan adds validation that mirrors the existing `_validate_route_destinations()` pattern.

## Root Cause

- Gate route destinations are validated via `_validate_route_destinations()` at startup
- Transform `on_error` destinations are only checked when an error actually occurs (in `TransformExecutor`)
- The orchestrator indexes `pending_tokens[result.sink_name]` without checking if it exists

## Implementation Steps

### Step 1: Add transform error sink validation method

**File:** `src/elspeth/engine/orchestrator.py`

**Location:** After `_validate_route_destinations()` method (around line 252)

```python
def _validate_transform_error_sinks(
    self,
    transforms: list[RowPlugin],
    available_sinks: set[str],
    transform_id_map: dict[int, str],
) -> None:
    """Validate all transform on_error destinations reference existing sinks.

    Called at pipeline initialization, BEFORE any rows are processed.
    This catches config errors early instead of failing mid-run with KeyError.

    Args:
        transforms: List of transform plugins
        available_sinks: Set of sink names from PipelineConfig
        transform_id_map: Maps transform sequence -> node_id

    Raises:
        RouteValidationError: If any transform on_error references a non-existent sink
    """
    for seq, transform in enumerate(transforms):
        # Access _on_error from the transform (part of TransformProtocol)
        on_error = getattr(transform, "_on_error", None)

        if on_error is None:
            # No error routing configured - that's fine
            continue

        if on_error == "discard":
            # "discard" is a special value, not a sink name
            continue

        # on_error should reference an existing sink
        if on_error not in available_sinks:
            raise RouteValidationError(
                f"Transform '{transform.name}' has on_error='{on_error}' "
                f"but no sink named '{on_error}' exists. "
                f"Available sinks: {sorted(available_sinks)}. "
                f"Use 'discard' to drop error rows without routing."
            )
```

### Step 2: Call validation during pipeline setup

**File:** `src/elspeth/engine/orchestrator.py`

**Location:** In `run()` method, after `_validate_route_destinations()` call (around line 520)

Find the existing call:
```python
self._validate_route_destinations(
    route_resolution_map=route_resolution_map,
    available_sinks=available_sinks,
    transform_id_map=transform_id_map,
    transforms=transforms,
    config_gate_id_map=config_gate_id_map,
    config_gates=config.gates,
)
```

Add immediately after:
```python
# Validate transform error sink destinations
self._validate_transform_error_sinks(
    transforms=transforms,
    available_sinks=available_sinks,
    transform_id_map=transform_id_map,
)
```

### Step 3: Also validate in `resume()` method if it exists

Check if there's a `resume()` method that also sets up the pipeline. If so, add the same validation there.

**Search pattern:** `def resume(`

### Step 4: Add unit tests

**File:** `tests/engine/test_orchestrator_validation.py` (new file or add to existing)

```python
"""Tests for pipeline validation at startup."""

import pytest

from elspeth.contracts.results import TransformResult
from elspeth.engine import RouteValidationError
from elspeth.engine.orchestrator import Orchestrator
from elspeth.plugins.base import BaseRowTransform
from elspeth.plugins.config_base import TransformDataConfig


class ErrorProneTransform(BaseRowTransform):
    """Test transform that can error with configurable on_error."""

    name = "error_prone"
    version = "1.0.0"

    def __init__(self, config: TransformDataConfig) -> None:
        super().__init__(config)
        self._on_error = config.on_error

    def process(self, row: dict, ctx) -> TransformResult:
        # Always error for testing
        return TransformResult.error({"reason": "test error"})


class TestTransformErrorSinkValidation:
    """Tests for on_error sink validation."""

    def test_invalid_on_error_sink_fails_at_startup(self, tmp_path):
        """Transform with on_error pointing to non-existent sink fails before processing."""
        from elspeth.core.config import PipelineConfig, SinkSettings, SourceSettings
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.plugins.sources.csv_source import CSVSource

        # Create minimal CSV
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("id,value\n1,test\n")

        # Create config with transform that has invalid on_error
        config = PipelineConfig(
            datasource=SourceSettings(
                plugin="csv",
                location=str(csv_file),
            ),
            sinks={
                "output": SinkSettings(
                    plugin="csv",
                    location=str(tmp_path / "output.csv"),
                ),
                # Note: "error_sink" is NOT defined
            },
            output_sink="output",
            row_plugins=[
                {
                    "plugin": "error_prone",
                    "on_error": "nonexistent_sink",  # This sink doesn't exist!
                }
            ],
        )

        db = LandscapeDB.in_memory()

        with pytest.raises(RouteValidationError) as exc_info:
            orchestrator = Orchestrator(db)
            # The error should occur during run() setup, before processing rows
            orchestrator.run(config)

        error_msg = str(exc_info.value)
        assert "error_prone" in error_msg
        assert "nonexistent_sink" in error_msg
        assert "output" in error_msg  # Should list available sinks

    def test_valid_on_error_sink_passes_validation(self, tmp_path):
        """Transform with on_error pointing to existing sink passes validation."""
        # Similar setup but with valid sink - should not raise during setup
        ...

    def test_on_error_discard_passes_validation(self, tmp_path):
        """Transform with on_error='discard' passes validation."""
        # "discard" is special and doesn't need a sink
        ...

    def test_on_error_none_passes_validation(self, tmp_path):
        """Transform without on_error configured passes validation."""
        # None means "crash on error" which is valid config
        ...
```

### Step 5: Update source quarantine validation (related)

The bug report mentions source quarantine validation is tracked separately, but the same pattern should apply. Check if `on_validation_failure` for sources also needs this validation.

**File to check:** Source config validation

## Testing Checklist

- [ ] Invalid `on_error` sink name raises `RouteValidationError` at startup
- [ ] Error message includes transform name and available sinks
- [ ] `on_error: "discard"` passes validation
- [ ] `on_error: null` (not set) passes validation
- [ ] Valid `on_error` sink name passes validation
- [ ] Error occurs BEFORE any rows are processed
- [ ] Existing gate route validation still works

## Run Tests

```bash
# Run new tests
.venv/bin/python -m pytest tests/engine/test_orchestrator_validation.py -v

# Run all orchestrator tests
.venv/bin/python -m pytest tests/engine/test_orchestrator*.py -v

# Run full test suite
.venv/bin/python -m pytest tests/ -v
```

## Acceptance Criteria

1. ✅ Pipeline with invalid `on_error` sink fails before processing rows
2. ✅ Error message is clear and actionable (shows transform name, invalid sink, available sinks)
3. ✅ `"discard"` and `None` are valid values that don't require sink lookup
4. ✅ Validation follows existing `_validate_route_destinations()` pattern
5. ✅ No regression in gate route validation

## Notes

**Why use `RouteValidationError`:**
Reusing the existing exception type keeps error handling consistent. The error message makes it clear this is about transform error routing, not gate routing.

**Why validate at startup:**
LLM transforms fail frequently (rate limits, content filters, timeouts). If error routing is misconfigured, you want to know immediately, not after processing 1000 rows.
