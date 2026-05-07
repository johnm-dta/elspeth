# Sink Failsink Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-row write failure routing to sinks via a configurable failsink (CSV/JSON/XML), where the plugin decides crash-vs-divert in its `write()` exception handling.

**Architecture:** BaseSink gains `_divert_row()` infrastructure. Plugins call it when per-row writes fail. SinkExecutor reads the diversion log after `write()` returns, writes diverted rows to the failsink, and records `DIVERTED` outcomes. `on_write_failure` is a mandatory field on SinkSettings (parallel to source `on_validation_failure`).

**Tech Stack:** Python dataclasses, Pydantic (config), pluggy (plugin system), SQLAlchemy Core (audit trail), pytest + Hypothesis (testing).

**Spec:** `docs/superpowers/specs/2026-03-26-sink-failsink-design.md`

**Key reference files:**
- `src/elspeth/contracts/enums.py` — RowOutcome enum (line 147)
- `src/elspeth/contracts/sink.py` — OutputValidationResult (existing sink contracts)
- `src/elspeth/contracts/run_result.py` — RunResult (line 19)
- `src/elspeth/contracts/plugin_protocols.py` — SinkProtocol (line 382)
- `src/elspeth/contracts/engine.py` — PendingOutcome (line 47)
- `src/elspeth/plugins/infrastructure/base.py` — BaseSink (line 339)
- `src/elspeth/engine/executors/sink.py` — SinkExecutor (line 30)
- `src/elspeth/engine/orchestrator/types.py` — ExecutionCounters (line 141), RunResult (line 199)
- `src/elspeth/engine/orchestrator/outcomes.py` — accumulate_row_outcomes (line 73)
- `src/elspeth/engine/orchestrator/core.py` — _drain_pending_tokens_to_sinks (line 514)
- `src/elspeth/core/config.py` — SinkSettings (line 916)
- `src/elspeth/core/dag/builder.py` — DIVERT edge creation (line 725)
- `src/elspeth/cli_helpers.py` — instantiate_plugins_from_config (line 47)
- `src/elspeth/engine/orchestrator/validation.py` — validate_transform_error_sinks (line 94)

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `src/elspeth/contracts/diversion.py` | `RowDiversion` and `SinkWriteResult` frozen dataclasses |
| Modify | `src/elspeth/contracts/enums.py:168` | Add `DIVERTED` to `RowOutcome` |
| Modify | `src/elspeth/contracts/run_result.py:19` | Add `rows_diverted` to `RunResult` |
| Modify | `src/elspeth/contracts/sink.py` | Re-export `SinkWriteResult` for convenience |
| Modify | `src/elspeth/contracts/plugin_protocols.py:451` | Update `SinkProtocol.write()` return type |
| Modify | `src/elspeth/contracts/__init__.py` | Export new types |
| Modify | `src/elspeth/plugins/infrastructure/base.py:339` | Add `_divert_row()`, `_diversion_log`, `_on_write_failure` to `BaseSink` |
| Modify | `src/elspeth/core/config.py:916` | Add `on_write_failure` to `SinkSettings` (mandatory) |
| Modify | `src/elspeth/engine/orchestrator/validation.py:94` | Add `validate_sink_failsink_destinations()` |
| Modify | `src/elspeth/core/dag/builder.py:725` | Add `__failsink__` DIVERT edges |
| Modify | `src/elspeth/engine/orchestrator/types.py:141` | Add `rows_diverted` to `ExecutionCounters` |
| Modify | `src/elspeth/engine/orchestrator/outcomes.py:73` | Add `DIVERTED` branch to `accumulate_row_outcomes()` |
| Modify | `src/elspeth/engine/executors/sink.py:94` | Handle `SinkWriteResult` diversions in `SinkExecutor.write()` |
| Modify | `src/elspeth/engine/orchestrator/core.py:514` | Pass failsink to `SinkExecutor.write()` in `_drain_pending_tokens_to_sinks()` |
| Modify | `src/elspeth/cli_helpers.py:107` | Inject `_on_write_failure` on sinks during instantiation |
| Modify | `src/elspeth/plugins/sinks/csv_sink.py` | Update `write()` return type to `SinkWriteResult` |
| Modify | `src/elspeth/plugins/sinks/json_sink.py` | Update `write()` return type to `SinkWriteResult` |
| Modify | `src/elspeth/plugins/sinks/database_sink.py` | Update `write()` return type to `SinkWriteResult` |
| Modify | `src/elspeth/plugins/sinks/azure_blob_sink.py` | Update `write()` return type to `SinkWriteResult` |
| Modify | `src/elspeth/plugins/sinks/chroma_sink.py` | Migrate inline filtering to `_divert_row()`, return `SinkWriteResult` |
| Create | `tests/unit/contracts/test_diversion.py` | Unit tests for `RowDiversion`, `SinkWriteResult` |
| Create | `tests/unit/engine/test_sink_executor_diversion.py` | Unit tests for SinkExecutor failsink routing |
| Modify | `tests/unit/plugins/sinks/test_chroma_sink.py` | Update tests for `SinkWriteResult` + `_divert_row()` migration |

---

### Task 1: RowDiversion and SinkWriteResult contracts

**Files:**
- Create: `src/elspeth/contracts/diversion.py`
- Create: `tests/unit/contracts/test_diversion.py`
- Modify: `src/elspeth/contracts/__init__.py`

- [ ] **Step 1: Write tests for RowDiversion and SinkWriteResult**

```python
# tests/unit/contracts/test_diversion.py
"""Tests for sink diversion contracts."""
from __future__ import annotations

from elspeth.contracts.diversion import RowDiversion, SinkWriteResult
from elspeth.contracts.results import ArtifactDescriptor


class TestRowDiversion:
    def test_create_with_required_fields(self) -> None:
        d = RowDiversion(row_index=0, reason="bad metadata", row_data={"a": 1})
        assert d.row_index == 0
        assert d.reason == "bad metadata"
        assert d.row_data == {"a": 1}

    def test_frozen(self) -> None:
        d = RowDiversion(row_index=0, reason="test", row_data={"a": 1})
        import pytest
        with pytest.raises(AttributeError):
            d.row_index = 1  # type: ignore[misc]

    def test_row_data_deep_frozen(self) -> None:
        d = RowDiversion(row_index=0, reason="test", row_data={"a": {"nested": 1}})
        import pytest
        with pytest.raises(TypeError):
            d.row_data["b"] = 2  # type: ignore[index]


class TestSinkWriteResult:
    def test_no_diversions(self) -> None:
        artifact = ArtifactDescriptor.for_file(
            path="/tmp/test.csv",
            content_hash="abc123",
            size_bytes=100,
        )
        result = SinkWriteResult(artifact=artifact)
        assert result.diversions == ()
        assert result.artifact is artifact

    def test_with_diversions(self) -> None:
        artifact = ArtifactDescriptor.for_file(
            path="/tmp/test.csv",
            content_hash="abc123",
            size_bytes=100,
        )
        divs = (
            RowDiversion(row_index=1, reason="bad type", row_data={"x": 1}),
            RowDiversion(row_index=3, reason="too long", row_data={"x": 2}),
        )
        result = SinkWriteResult(artifact=artifact, diversions=divs)
        assert len(result.diversions) == 2
        assert result.diversions[0].row_index == 1
        assert result.diversions[1].row_index == 3

    def test_frozen(self) -> None:
        artifact = ArtifactDescriptor.for_file(
            path="/tmp/test.csv",
            content_hash="abc123",
            size_bytes=100,
        )
        result = SinkWriteResult(artifact=artifact)
        import pytest
        with pytest.raises(AttributeError):
            result.artifact = artifact  # type: ignore[misc]
```

- [ ] **Step 2: Run tests — expect FAIL (module not found)**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_diversion.py -v`
Expected: `ModuleNotFoundError: No module named 'elspeth.contracts.diversion'`

- [ ] **Step 3: Implement RowDiversion and SinkWriteResult**

```python
# src/elspeth/contracts/diversion.py
"""Sink diversion contracts — per-row write failure routing types.

These types support the failsink pattern: when a sink can't write a specific
row (value-level failure at the Tier 2 → External boundary), it diverts the
row to a failsink. These contracts carry diversion information from the plugin's
write() method back to SinkExecutor for audit trail recording.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from elspeth.contracts.freeze import freeze_fields
from elspeth.contracts.results import ArtifactDescriptor


@dataclass(frozen=True, slots=True)
class RowDiversion:
    """Record of a single row diverted to failsink during write().

    Created by BaseSink._divert_row() and accumulated in _diversion_log.
    Read by SinkExecutor after write() returns to record per-token outcomes.

    Attributes:
        row_index: Index in the original batch passed to write().
        reason: Why the external system rejected this row.
        row_data: The row that was diverted (for failsink write).
    """

    row_index: int
    reason: str
    row_data: Mapping[str, Any]

    def __post_init__(self) -> None:
        freeze_fields(self, "row_data")


@dataclass(frozen=True, slots=True)
class SinkWriteResult:
    """Result of a sink write() call with optional diversion information.

    Replaces ArtifactDescriptor as the return type of write().
    Sinks with no diversions return SinkWriteResult(artifact=..., diversions=()).
    """

    artifact: ArtifactDescriptor
    diversions: tuple[RowDiversion, ...] = ()
```

- [ ] **Step 4: Add exports to contracts __init__.py**

Add to `src/elspeth/contracts/__init__.py`:
```python
from elspeth.contracts.diversion import RowDiversion, SinkWriteResult
```

Find the existing import block and add these alongside the other contract imports.

- [ ] **Step 5: Run tests — expect PASS**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_diversion.py -v`
Expected: All 5 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/contracts/diversion.py tests/unit/contracts/test_diversion.py src/elspeth/contracts/__init__.py
git commit -m "feat(contracts): add RowDiversion and SinkWriteResult for failsink pattern"
```

---

### Task 2: DIVERTED RowOutcome + RunResult counter

**Files:**
- Modify: `src/elspeth/contracts/enums.py:168`
- Modify: `src/elspeth/contracts/run_result.py:19`

- [ ] **Step 1: Write test for DIVERTED outcome**

Add to an existing enums test file or create inline:

```python
# Quick verification — run in pytest
def test_diverted_is_terminal() -> None:
    from elspeth.contracts.enums import RowOutcome
    assert RowOutcome.DIVERTED == "diverted"
    assert RowOutcome.DIVERTED.is_terminal is True
```

- [ ] **Step 2: Add DIVERTED to RowOutcome**

In `src/elspeth/contracts/enums.py`, after line 173 (`QUARANTINED = "quarantined"`), add:

```python
    DIVERTED = "diverted"
```

Update the docstring (lines 154-162) to include:
```
    - DIVERTED: Sink write failed for this row, diverted to failsink
```

- [ ] **Step 3: Add rows_diverted to RunResult**

In `src/elspeth/contracts/run_result.py`, after line 33 (`rows_buffered: int = 0`), add:

```python
    rows_diverted: int = 0  # Rows diverted to failsink during sink write
```

In `__post_init__`, after line 48 (`require_int(self.rows_buffered, ...)`), add:

```python
        require_int(self.rows_diverted, "rows_diverted", min_value=0)
```

- [ ] **Step 4: Run full test suite to check for breakage**

Run: `.venv/bin/python -m pytest tests/unit/contracts/ -v --tb=short`
Expected: All PASS. The new enum value and new field with default=0 should not break anything.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/contracts/enums.py src/elspeth/contracts/run_result.py
git commit -m "feat(contracts): add DIVERTED outcome and rows_diverted counter"
```

---

### Task 3: ExecutionCounters + accumulate_row_outcomes DIVERTED branch

**Files:**
- Modify: `src/elspeth/engine/orchestrator/types.py:141`
- Modify: `src/elspeth/engine/orchestrator/outcomes.py:73`

- [ ] **Step 1: Write test for DIVERTED in accumulate_row_outcomes**

```python
# tests/unit/engine/test_outcomes_diverted.py
"""Test DIVERTED outcome branch in accumulate_row_outcomes."""
from __future__ import annotations

from unittest.mock import MagicMock

from elspeth.contracts.enums import RowOutcome
from elspeth.engine.orchestrator.outcomes import accumulate_row_outcomes
from elspeth.engine.orchestrator.types import ExecutionCounters


def _make_result(outcome: RowOutcome, sink_name: str | None = None) -> MagicMock:
    result = MagicMock()
    result.outcome = outcome
    result.sink_name = sink_name
    result.token = MagicMock()
    return result


class TestAccumulateDiverted:
    def test_diverted_increments_counter(self) -> None:
        counters = ExecutionCounters()
        pending_tokens: dict[str, list] = {"output": []}
        results = [_make_result(RowOutcome.DIVERTED)]
        accumulate_row_outcomes(results, counters, {"output": MagicMock()}, pending_tokens)
        assert counters.rows_diverted == 1

    def test_diverted_does_not_route_to_pending(self) -> None:
        """DIVERTED tokens are already written by SinkExecutor — no pending routing."""
        counters = ExecutionCounters()
        pending_tokens: dict[str, list] = {"output": []}
        results = [_make_result(RowOutcome.DIVERTED)]
        accumulate_row_outcomes(results, counters, {"output": MagicMock()}, pending_tokens)
        assert pending_tokens["output"] == []

    def test_mixed_completed_and_diverted(self) -> None:
        counters = ExecutionCounters()
        pending_tokens: dict[str, list] = {"output": []}
        results = [
            _make_result(RowOutcome.COMPLETED, sink_name="output"),
            _make_result(RowOutcome.DIVERTED),
            _make_result(RowOutcome.COMPLETED, sink_name="output"),
        ]
        accumulate_row_outcomes(results, counters, {"output": MagicMock()}, pending_tokens)
        assert counters.rows_succeeded == 2
        assert counters.rows_diverted == 1
        assert len(pending_tokens["output"]) == 2
```

- [ ] **Step 2: Run test — expect FAIL**

Run: `.venv/bin/python -m pytest tests/unit/engine/test_outcomes_diverted.py -v`
Expected: `AttributeError: 'ExecutionCounters' object has no attribute 'rows_diverted'`

- [ ] **Step 3: Add rows_diverted to ExecutionCounters**

In `src/elspeth/engine/orchestrator/types.py`, after line 163 (`routed_destinations: Counter[str] = ...`), add:

```python
    rows_diverted: int = 0
```

In `to_run_result()` (line 199), add to the RunResult constructor call after `rows_buffered`:

```python
            rows_diverted=self.rows_diverted,
```

In `accumulate_flush_result()` (line 165), add after `rows_buffered`:

```python
        self.rows_diverted += result.rows_diverted
```

In `to_flush_result()` (line 182), add after `rows_buffered`:

```python
            rows_diverted=self.rows_diverted,
```

Also add `rows_diverted: int = 0` to `AggregationFlushResult` (line 104) and update its `__add__` method.

- [ ] **Step 4: Add DIVERTED branch to accumulate_row_outcomes**

In `src/elspeth/engine/orchestrator/outcomes.py`, after the `EXPANDED` branch (line 125) and before the `BUFFERED` branch (line 126), add:

```python
        elif result.outcome == RowOutcome.DIVERTED:
            counters.rows_diverted += 1
            # DIVERTED tokens are already written to failsink by SinkExecutor.
            # They are NOT appended to pending_tokens — no further routing needed.
```

- [ ] **Step 5: Run tests — expect PASS**

Run: `.venv/bin/python -m pytest tests/unit/engine/test_outcomes_diverted.py -v`
Expected: All 3 tests PASS.

- [ ] **Step 6: Run broader orchestrator tests for regression**

Run: `.venv/bin/python -m pytest tests/unit/engine/ -v --tb=short -q`
Expected: All PASS.

- [ ] **Step 7: Commit**

```bash
git add src/elspeth/engine/orchestrator/types.py src/elspeth/engine/orchestrator/outcomes.py tests/unit/engine/test_outcomes_diverted.py
git commit -m "feat(engine): add DIVERTED branch to outcome accumulation and counters"
```

---

### Task 4: SinkSettings.on_write_failure (mandatory config field)

**Files:**
- Modify: `src/elspeth/core/config.py:916`

- [ ] **Step 1: Write test for on_write_failure validation**

```python
# tests/unit/core/test_sink_settings_on_write_failure.py
"""Test SinkSettings.on_write_failure mandatory field."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from elspeth.core.config import SinkSettings


class TestSinkSettingsOnWriteFailure:
    def test_on_write_failure_required(self) -> None:
        """SinkSettings must have on_write_failure — no default."""
        with pytest.raises(ValidationError, match="on_write_failure"):
            SinkSettings(plugin="csv")

    def test_on_write_failure_discard(self) -> None:
        s = SinkSettings(plugin="csv", on_write_failure="discard")
        assert s.on_write_failure == "discard"

    def test_on_write_failure_sink_name(self) -> None:
        s = SinkSettings(plugin="csv", on_write_failure="csv_failsink")
        assert s.on_write_failure == "csv_failsink"

    def test_on_write_failure_empty_rejected(self) -> None:
        with pytest.raises(ValidationError, match="on_write_failure"):
            SinkSettings(plugin="csv", on_write_failure="")

    def test_on_write_failure_whitespace_stripped(self) -> None:
        s = SinkSettings(plugin="csv", on_write_failure="  discard  ")
        assert s.on_write_failure == "discard"
```

- [ ] **Step 2: Run test — expect FAIL**

Run: `.venv/bin/python -m pytest tests/unit/core/test_sink_settings_on_write_failure.py -v`
Expected: FAIL — `on_write_failure` field doesn't exist yet.

- [ ] **Step 3: Add on_write_failure to SinkSettings**

In `src/elspeth/core/config.py`, replace the `SinkSettings` class (lines 916-925):

```python
class SinkSettings(BaseModel):
    """Sink plugin configuration per architecture."""

    model_config = {"frozen": True, "extra": "forbid"}

    plugin: str = Field(description="Plugin name (csv, json, database, webhook, etc.)")
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Plugin-specific configuration options",
    )
    on_write_failure: str = Field(
        ...,  # Required — no default
        description=(
            "Per-row write failure handling. Required — pipeline author must decide: "
            "'discard' to drop with audit record, or a sink name to divert to failsink "
            "(must be csv, json, or xml plugin)."
        ),
    )

    @field_validator("on_write_failure")
    @classmethod
    def validate_on_write_failure(cls, v: str) -> str:
        """Ensure on_write_failure is not empty."""
        if not v or not v.strip():
            raise ValueError("on_write_failure must be a sink name or 'discard'")
        value = v.strip()
        if value == "discard":
            return value
        return _validate_connection_or_sink_name(value, field_label="Sink on_write_failure sink name")
```

Check that `_validate_connection_or_sink_name` is already imported/available in config.py (it is — used by TransformSettings). Check that `field_validator` is imported from pydantic (it is — used elsewhere in config.py).

- [ ] **Step 4: Update ALL existing pipeline YAML fixtures and test configs**

Search for all YAML/dict fixtures that create SinkSettings without `on_write_failure` and add `on_write_failure: discard` to each. This is the most labor-intensive step.

Run: `grep -r "plugin.*csv\|plugin.*json\|plugin.*chroma\|plugin.*database" tests/ --include="*.py" --include="*.yaml" -l` to find files.

For each test that constructs `SinkSettings(plugin=..., options=...)`, add `on_write_failure="discard"`.

For each YAML fixture with sink config, add `on_write_failure: discard`.

- [ ] **Step 5: Run tests — expect PASS**

Run: `.venv/bin/python -m pytest tests/unit/core/test_sink_settings_on_write_failure.py -v`
Expected: All 5 PASS.

Run: `.venv/bin/python -m pytest tests/ -x --tb=short -q`
Expected: All PASS (after fixture updates).

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/core/config.py tests/
git commit -m "feat(config): add mandatory on_write_failure to SinkSettings"
```

---

### Task 5: BaseSink._divert_row() infrastructure

**Files:**
- Modify: `src/elspeth/plugins/infrastructure/base.py:339`

- [ ] **Step 1: Write tests for _divert_row()**

```python
# tests/unit/plugins/infrastructure/test_base_sink_divert.py
"""Tests for BaseSink._divert_row() failsink infrastructure."""
from __future__ import annotations

from typing import Any

import pytest

from elspeth.contracts.diversion import RowDiversion, SinkWriteResult
from elspeth.contracts.errors import FrameworkBugError
from elspeth.contracts.results import ArtifactDescriptor
from elspeth.plugins.infrastructure.base import BaseSink


class StubSink(BaseSink):
    """Minimal concrete sink for testing BaseSink infrastructure."""
    name = "stub_sink"
    input_schema = None  # type: ignore[assignment]

    def write(self, rows: list[dict[str, Any]], ctx: Any) -> SinkWriteResult:
        artifact = ArtifactDescriptor.for_file(path="/tmp/stub", content_hash="x", size_bytes=0)
        return SinkWriteResult(artifact=artifact, diversions=self._get_diversions())

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class TestDivertRow:
    def test_divert_accumulates_to_log(self) -> None:
        sink = StubSink({})
        sink._on_write_failure = "csv_failsink"
        sink._divert_row({"a": 1}, row_index=0, reason="bad value")
        sink._divert_row({"b": 2}, row_index=2, reason="too long")
        diversions = sink._get_diversions()
        assert len(diversions) == 2
        assert diversions[0].row_index == 0
        assert diversions[1].row_index == 2

    def test_divert_without_on_write_failure_raises(self) -> None:
        sink = StubSink({})
        # _on_write_failure not set — calling _divert_row is a plugin bug
        with pytest.raises(FrameworkBugError, match="on_write_failure"):
            sink._divert_row({"a": 1}, row_index=0, reason="test")

    def test_reset_clears_log(self) -> None:
        sink = StubSink({})
        sink._on_write_failure = "discard"
        sink._divert_row({"a": 1}, row_index=0, reason="test")
        assert len(sink._get_diversions()) == 1
        sink._reset_diversion_log()
        assert len(sink._get_diversions()) == 0

    def test_divert_with_discard_mode(self) -> None:
        """Discard mode still accumulates — executor decides what to do."""
        sink = StubSink({})
        sink._on_write_failure = "discard"
        sink._divert_row({"a": 1}, row_index=0, reason="intentional drop")
        diversions = sink._get_diversions()
        assert len(diversions) == 1
        assert diversions[0].reason == "intentional drop"

    def test_get_diversions_returns_tuple(self) -> None:
        sink = StubSink({})
        sink._on_write_failure = "discard"
        sink._divert_row({"a": 1}, row_index=0, reason="test")
        result = sink._get_diversions()
        assert isinstance(result, tuple)
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `.venv/bin/python -m pytest tests/unit/plugins/infrastructure/test_base_sink_divert.py -v`
Expected: FAIL — `_on_write_failure`, `_divert_row` not on BaseSink.

- [ ] **Step 3: Add _divert_row() infrastructure to BaseSink**

In `src/elspeth/plugins/infrastructure/base.py`, add to the `BaseSink` class:

After the existing class attributes (around line 407), add:

```python
    # Failsink infrastructure — set by orchestrator from SinkSettings.on_write_failure
    _on_write_failure: str | None = None  # None until injected; "discard" or sink name at runtime
```

In `__init__` (line 488), add after `self._needs_resume_field_resolution = False`:

```python
        self._diversion_log: list[RowDiversion] = []
```

Add the import at the top of the file (in the existing imports section):
```python
from elspeth.contracts.diversion import RowDiversion, SinkWriteResult
from elspeth.contracts.errors import FrameworkBugError
```

Add the three methods after `set_output_contract()` (around line 544):

```python
    def _divert_row(self, row: dict[str, Any], row_index: int, reason: str) -> None:
        """Divert a row to the failsink or discard. Called by plugin write() on per-row failure.

        This is the sink-side equivalent of SourceRow.quarantined(). The plugin
        catches a per-row exception from the external system and calls this
        instead of re-raising.

        Args:
            row: The row dict that couldn't be written.
            row_index: Index in the original batch (for token correlation).
            reason: Human-readable reason for the diversion.

        Raises:
            FrameworkBugError: If _on_write_failure has not been set (plugin bug —
                calling _divert_row before orchestrator injection).
        """
        if self._on_write_failure is None:
            raise FrameworkBugError(
                f"Sink '{self.name}' called _divert_row() but _on_write_failure is not set. "
                f"Configure on_write_failure in pipeline YAML or re-raise "
                f"the exception to crash the pipeline."
            )
        self._diversion_log.append(RowDiversion(
            row_index=row_index,
            reason=reason,
            row_data=row,
        ))

    def _reset_diversion_log(self) -> None:
        """Clear diversion log before each write() call. Called by SinkExecutor."""
        self._diversion_log = []

    def _get_diversions(self) -> tuple[RowDiversion, ...]:
        """Return accumulated diversions from the last write() call."""
        return tuple(self._diversion_log)
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `.venv/bin/python -m pytest tests/unit/plugins/infrastructure/test_base_sink_divert.py -v`
Expected: All 5 tests PASS.

- [ ] **Step 5: Run existing sink tests for regression**

Run: `.venv/bin/python -m pytest tests/unit/plugins/sinks/ -v --tb=short -q`
Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/plugins/infrastructure/base.py tests/unit/plugins/infrastructure/test_base_sink_divert.py
git commit -m "feat(base): add _divert_row() infrastructure to BaseSink"
```

---

### Task 6: Update SinkProtocol.write() return type

**Files:**
- Modify: `src/elspeth/contracts/plugin_protocols.py:451`

- [ ] **Step 1: Update SinkProtocol.write() return type**

In `src/elspeth/contracts/plugin_protocols.py`, change the `write()` method (line 451):

From:
```python
    def write(
        self,
        rows: list[dict[str, Any]],
        ctx: "SinkContext",
    ) -> "ArtifactDescriptor":
```

To:
```python
    def write(
        self,
        rows: list[dict[str, Any]],
        ctx: "SinkContext",
    ) -> "SinkWriteResult":
```

Add `SinkWriteResult` to the `TYPE_CHECKING` imports at the top of the file:
```python
    from elspeth.contracts.diversion import SinkWriteResult
```

Update the docstring return type description accordingly.

- [ ] **Step 2: Run mypy to check type consistency**

Run: `.venv/bin/python -m mypy src/elspeth/contracts/plugin_protocols.py`
Expected: PASS (or only pre-existing issues).

- [ ] **Step 3: Commit**

```bash
git add src/elspeth/contracts/plugin_protocols.py
git commit -m "feat(protocol): update SinkProtocol.write() return type to SinkWriteResult"
```

---

### Task 7: Update all sink write() return types to SinkWriteResult

**Files:**
- Modify: `src/elspeth/plugins/sinks/csv_sink.py`
- Modify: `src/elspeth/plugins/sinks/json_sink.py`
- Modify: `src/elspeth/plugins/sinks/database_sink.py`
- Modify: `src/elspeth/plugins/sinks/azure_blob_sink.py`
- Modify: `src/elspeth/plugins/infrastructure/base.py` (abstract method signature)

For each sink, the change is mechanical:

1. Add import: `from elspeth.contracts.diversion import SinkWriteResult`
2. Change `write()` return type from `ArtifactDescriptor` to `SinkWriteResult`
3. Wrap the returned `ArtifactDescriptor` in `SinkWriteResult(artifact=...)`

- [ ] **Step 1: Update BaseSink abstract write() signature**

In `src/elspeth/plugins/infrastructure/base.py`, change the abstract `write()` method (line 492):

From:
```python
    @abstractmethod
    def write(
        self,
        rows: list[dict[str, Any]],
        ctx: SinkContext,
    ) -> ArtifactDescriptor:
```

To:
```python
    @abstractmethod
    def write(
        self,
        rows: list[dict[str, Any]],
        ctx: SinkContext,
    ) -> SinkWriteResult:
```

- [ ] **Step 2: Update CSVSink**

In `src/elspeth/plugins/sinks/csv_sink.py`:
- Add import: `from elspeth.contracts.diversion import SinkWriteResult`
- Change `write()` return type annotation to `SinkWriteResult`
- At each `return ArtifactDescriptor.for_file(...)` statement, wrap: `return SinkWriteResult(artifact=ArtifactDescriptor.for_file(...))`

- [ ] **Step 3: Update JSONSink**

Same mechanical change as CSVSink in `src/elspeth/plugins/sinks/json_sink.py`.

- [ ] **Step 4: Update DatabaseSink**

Same mechanical change in `src/elspeth/plugins/sinks/database_sink.py`.

- [ ] **Step 5: Update AzureBlobSink**

Same mechanical change in `src/elspeth/plugins/sinks/azure_blob_sink.py`.

- [ ] **Step 6: Run all sink tests**

Run: `.venv/bin/python -m pytest tests/unit/plugins/sinks/ -v --tb=short -q`
Expected: Some failures — tests that assert `isinstance(result, ArtifactDescriptor)` need to change to check `result.artifact` instead.

- [ ] **Step 7: Fix test assertions**

Update tests that directly check the return value of `write()`:
- Change `assert isinstance(result, ArtifactDescriptor)` to `assert isinstance(result, SinkWriteResult)`
- Change `result.content_hash` to `result.artifact.content_hash`
- Change `result.path_or_uri` to `result.artifact.path_or_uri`

- [ ] **Step 8: Run all sink tests — expect PASS**

Run: `.venv/bin/python -m pytest tests/unit/plugins/sinks/ -v --tb=short -q`
Expected: All PASS.

- [ ] **Step 9: Run mypy**

Run: `.venv/bin/python -m mypy src/elspeth/plugins/sinks/`
Expected: PASS (or only pre-existing issues).

- [ ] **Step 10: Commit**

```bash
git add src/elspeth/plugins/infrastructure/base.py src/elspeth/plugins/sinks/ tests/unit/plugins/sinks/
git commit -m "refactor(sinks): update all sink write() return types to SinkWriteResult"
```

---

### Task 8: SinkExecutor failsink routing

**Files:**
- Modify: `src/elspeth/engine/executors/sink.py:94`
- Create: `tests/unit/engine/test_sink_executor_diversion.py`

This is the most critical task — it handles per-token outcome recording for diverted rows.

- [ ] **Step 1: Write tests for SinkExecutor diversion handling**

```python
# tests/unit/engine/test_sink_executor_diversion.py
"""Tests for SinkExecutor failsink routing."""
from __future__ import annotations

import hashlib
from typing import Any
from unittest.mock import MagicMock, call

import pytest

from elspeth.contracts import PendingOutcome, RowOutcome, TokenInfo
from elspeth.contracts.diversion import RowDiversion, SinkWriteResult
from elspeth.contracts.enums import NodeStateStatus
from elspeth.contracts.results import ArtifactDescriptor
from elspeth.engine.executors.sink import SinkExecutor


def _make_token(token_id: str = "tok-1", row_data: dict | None = None) -> TokenInfo:
    """Create a minimal TokenInfo mock."""
    token = MagicMock(spec=TokenInfo)
    token.token_id = token_id
    token.row_id = f"row-{token_id}"
    mock_row = MagicMock()
    mock_row.to_dict.return_value = row_data or {"field": "value"}
    mock_row.contract = MagicMock()
    mock_row.contract.merge.return_value = mock_row.contract
    token.row_data = mock_row
    return token


def _make_artifact(path: str = "/tmp/test") -> ArtifactDescriptor:
    return ArtifactDescriptor.for_file(path=path, content_hash="abc", size_bytes=100)


def _make_sink(
    name: str = "primary",
    node_id: str = "node-primary",
    diversions: tuple[RowDiversion, ...] = (),
) -> MagicMock:
    sink = MagicMock()
    sink.name = name
    sink.node_id = node_id
    sink.validate_input = False
    sink.declared_required_fields = frozenset()
    sink.write.return_value = SinkWriteResult(
        artifact=_make_artifact(),
        diversions=diversions,
    )
    sink._reset_diversion_log = MagicMock()
    sink._on_write_failure = "discard"
    return sink


def _make_executor() -> tuple[SinkExecutor, MagicMock, MagicMock]:
    recorder = MagicMock()
    recorder.begin_node_state.return_value = MagicMock(state_id="state-1")
    recorder.allocate_operation_call_index = MagicMock(return_value=0)
    spans = MagicMock()
    spans.sink_span.return_value.__enter__ = MagicMock(return_value=None)
    spans.sink_span.return_value.__exit__ = MagicMock(return_value=False)
    executor = SinkExecutor(recorder, spans, "run-1")
    return executor, recorder, spans


class TestSinkExecutorNoDiversions:
    """Verify existing behavior is preserved when no diversions occur."""

    def test_no_diversions_records_all_completed(self) -> None:
        executor, recorder, _ = _make_executor()
        sink = _make_sink()
        tokens = [_make_token("t1"), _make_token("t2")]
        executor.write(
            sink=sink,
            tokens=tokens,
            ctx=MagicMock(run_id="run-1"),
            step_in_pipeline=5,
            sink_name="primary",
            pending_outcome=PendingOutcome(RowOutcome.COMPLETED),
        )
        # All tokens should get COMPLETED outcome
        outcome_calls = recorder.record_token_outcome.call_args_list
        assert len(outcome_calls) == 2
        for c in outcome_calls:
            assert c.kwargs["outcome"] == RowOutcome.COMPLETED


class TestSinkExecutorWithDiversions:
    def test_diverted_tokens_get_diverted_outcome(self) -> None:
        executor, recorder, _ = _make_executor()
        diversions = (
            RowDiversion(row_index=1, reason="bad metadata", row_data={"x": 1}),
        )
        sink = _make_sink(diversions=diversions)
        # failsink for discard mode — no actual failsink write needed
        tokens = [_make_token("t0"), _make_token("t1")]
        executor.write(
            sink=sink,
            tokens=tokens,
            ctx=MagicMock(run_id="run-1"),
            step_in_pipeline=5,
            sink_name="primary",
            pending_outcome=PendingOutcome(RowOutcome.COMPLETED),
        )
        outcome_calls = recorder.record_token_outcome.call_args_list
        # t0 should be COMPLETED, t1 should be DIVERTED
        assert outcome_calls[0].kwargs["outcome"] == RowOutcome.COMPLETED
        assert outcome_calls[1].kwargs["outcome"] == RowOutcome.DIVERTED

    def test_all_diverted_no_primary_artifact(self) -> None:
        """When all rows are diverted, primary artifact still exists (empty write)."""
        executor, recorder, _ = _make_executor()
        diversions = (
            RowDiversion(row_index=0, reason="bad", row_data={"x": 1}),
        )
        sink = _make_sink(diversions=diversions)
        tokens = [_make_token("t0")]
        executor.write(
            sink=sink,
            tokens=tokens,
            ctx=MagicMock(run_id="run-1"),
            step_in_pipeline=5,
            sink_name="primary",
            pending_outcome=PendingOutcome(RowOutcome.COMPLETED),
        )
        outcome_calls = recorder.record_token_outcome.call_args_list
        assert len(outcome_calls) == 1
        assert outcome_calls[0].kwargs["outcome"] == RowOutcome.DIVERTED
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `.venv/bin/python -m pytest tests/unit/engine/test_sink_executor_diversion.py -v`
Expected: FAIL — SinkExecutor doesn't handle SinkWriteResult yet.

- [ ] **Step 3: Modify SinkExecutor.write() to handle SinkWriteResult**

In `src/elspeth/engine/executors/sink.py`, the key changes are:

1. Add import: `from elspeth.contracts.diversion import SinkWriteResult, RowDiversion`
2. Add `failsink` parameter to `write()` signature (optional, `SinkProtocol | None = None`)
3. After `sink.write()` returns, check if it's a `SinkWriteResult` (it always will be after Task 7, but handle `ArtifactDescriptor` for safety during migration)
4. Build a set of diverted `row_index` values from `result.diversions`
5. In the token outcome recording loop (line 367), check if the token's index is in the diverted set:
   - If yes: record `DIVERTED` outcome with `error_hash` computed from reason
   - If no: record the original `pending_outcome`
6. If there are diversions AND `_on_write_failure != "discard"` AND `failsink is not None`: write diverted rows to failsink

The critical invariant: **primary token states are completed BEFORE failsink writes**. If failsink write fails, only failsink token states are marked FAILED. Primary states remain COMPLETED.

Read the full `SinkExecutor.write()` method carefully before modifying. The changes integrate into the existing error handling structure. Key insertion points:

- After `artifact_info = sink.write(rows, ctx)` (line 278): extract `SinkWriteResult`, build diverted index set
- After `sink.flush()` (line 299): if diversions exist, handle failsink write
- In the outcome recording loop (line 367): check diverted set for per-token outcome

**Failsink write details (non-discard mode):**
When writing diverted rows to the failsink, enrich each row with diversion metadata:
```python
enriched_row = {
    **diversion.row_data,
    "__diversion_reason": diversion.reason,
    "__diverted_from": sink_name,
    "__diversion_timestamp": datetime.now(UTC).isoformat(),
}
```

**Routing_event recording:**
Before opening failsink node_states, record a DIVERT routing_event for each diverted token. This requires looking up the `__failsink__` edge_id from the DAG edge_map. The edge_map is accessible via the orchestrator context — pass it through to the executor or resolve the edge_id in the orchestrator before calling write. Use the same pattern as source quarantine routing_events in `_handle_quarantined_row()` (orchestrator/core.py:1799).

- [ ] **Step 4: Run tests — expect PASS**

Run: `.venv/bin/python -m pytest tests/unit/engine/test_sink_executor_diversion.py -v`
Expected: All PASS.

- [ ] **Step 5: Run existing SinkExecutor tests for regression**

Run: `.venv/bin/python -m pytest tests/unit/engine/test_executors.py -v --tb=short -q`
Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/engine/executors/sink.py tests/unit/engine/test_sink_executor_diversion.py
git commit -m "feat(executor): add failsink routing to SinkExecutor.write()"
```

---

### Task 9: Config validation + DAG builder __failsink__ edges

**Files:**
- Modify: `src/elspeth/engine/orchestrator/validation.py`
- Modify: `src/elspeth/core/dag/builder.py:725`

- [ ] **Step 1: Write test for failsink config validation**

```python
# tests/unit/engine/test_failsink_validation.py
"""Tests for sink failsink destination validation."""
from __future__ import annotations

import pytest

from elspeth.engine.orchestrator.types import RouteValidationError
from elspeth.engine.orchestrator.validation import validate_sink_failsink_destinations


class TestValidateSinkFailsinkDestinations:
    def test_discard_always_valid(self) -> None:
        """on_write_failure='discard' needs no target sink."""
        validate_sink_failsink_destinations(
            sink_configs={"output": _stub_settings("discard")},
            available_sinks={"output"},
            allowed_failsink_plugins={"csv", "json", "xml"},
            sink_plugins={"output": "chroma_sink"},
        )  # No error

    def test_valid_failsink_reference(self) -> None:
        validate_sink_failsink_destinations(
            sink_configs={"output": _stub_settings("csv_failsink"), "csv_failsink": _stub_settings("discard")},
            available_sinks={"output", "csv_failsink"},
            allowed_failsink_plugins={"csv", "json", "xml"},
            sink_plugins={"output": "chroma_sink", "csv_failsink": "csv"},
        )  # No error

    def test_unknown_failsink_raises(self) -> None:
        with pytest.raises(RouteValidationError, match="nonexistent"):
            validate_sink_failsink_destinations(
                sink_configs={"output": _stub_settings("nonexistent")},
                available_sinks={"output"},
                allowed_failsink_plugins={"csv", "json", "xml"},
                sink_plugins={"output": "chroma_sink"},
            )

    def test_non_file_failsink_raises(self) -> None:
        """Failsink must be csv, json, or xml."""
        with pytest.raises(RouteValidationError, match="csv, json, or xml"):
            validate_sink_failsink_destinations(
                sink_configs={"output": _stub_settings("db_sink"), "db_sink": _stub_settings("discard")},
                available_sinks={"output", "db_sink"},
                allowed_failsink_plugins={"csv", "json", "xml"},
                sink_plugins={"output": "chroma_sink", "db_sink": "database"},
            )

    def test_failsink_chaining_raises(self) -> None:
        """Failsink targets must have on_write_failure='discard'."""
        with pytest.raises(RouteValidationError, match="discard"):
            validate_sink_failsink_destinations(
                sink_configs={
                    "output": _stub_settings("failsink1"),
                    "failsink1": _stub_settings("failsink2"),
                    "failsink2": _stub_settings("discard"),
                },
                available_sinks={"output", "failsink1", "failsink2"},
                allowed_failsink_plugins={"csv", "json", "xml"},
                sink_plugins={"output": "chroma_sink", "failsink1": "csv", "failsink2": "csv"},
            )


def _stub_settings(on_write_failure: str) -> object:
    """Minimal stub with on_write_failure attribute."""
    from types import SimpleNamespace
    return SimpleNamespace(on_write_failure=on_write_failure)
```

- [ ] **Step 2: Run test — expect FAIL**

Run: `.venv/bin/python -m pytest tests/unit/engine/test_failsink_validation.py -v`
Expected: `ImportError: cannot import name 'validate_sink_failsink_destinations'`

- [ ] **Step 3: Implement validate_sink_failsink_destinations()**

Add to `src/elspeth/engine/orchestrator/validation.py`, after `validate_source_quarantine_destination()`:

```python
_ALLOWED_FAILSINK_PLUGINS = frozenset({"csv", "json", "xml"})


def validate_sink_failsink_destinations(
    sink_configs: Mapping[str, Any],
    available_sinks: set[str],
    allowed_failsink_plugins: frozenset[str] = _ALLOWED_FAILSINK_PLUGINS,
    sink_plugins: Mapping[str, str] | None = None,
) -> None:
    """Validate all sink on_write_failure destinations.

    Rules:
    1. 'discard' is always valid
    2. Sink name must exist in available_sinks
    3. Target sink must use csv, json, or xml plugin
    4. Target sink must have on_write_failure='discard' (no chains)
    """
    for sink_name, config in sink_configs.items():
        dest = config.on_write_failure
        if dest == "discard":
            continue

        if dest not in available_sinks:
            raise RouteValidationError(
                f"Sink '{sink_name}' on_write_failure references unknown sink '{dest}'. "
                f"Available sinks: {sorted(available_sinks)}."
            )

        if sink_plugins and dest in sink_plugins:
            plugin_type = sink_plugins[dest]
            if plugin_type not in allowed_failsink_plugins:
                raise RouteValidationError(
                    f"Sink '{sink_name}' on_write_failure references '{dest}' "
                    f"(plugin='{plugin_type}'), but failsinks must use csv, json, or xml plugins."
                )

        if dest in sink_configs:
            target_dest = sink_configs[dest].on_write_failure
            if target_dest != "discard":
                raise RouteValidationError(
                    f"Sink '{sink_name}' on_write_failure references '{dest}', "
                    f"but '{dest}' has on_write_failure='{target_dest}'. "
                    f"Failsink targets must have on_write_failure='discard' (no chains)."
                )
```

- [ ] **Step 4: Add __failsink__ DIVERT edges to DAG builder**

In `src/elspeth/core/dag/builder.py`, after the transform error edges block (around line 763), add:

```python
    # Sink failsink edges
    for sink_name_str, sink_node_id in sink_ids.items():
        sink_name_key = str(sink_name_str)
        if sink_name_key in sink_settings_map:
            on_write_failure = sink_settings_map[sink_name_key].on_write_failure
            if on_write_failure != "discard" and SinkName(on_write_failure) in sink_ids:
                graph.add_edge(
                    sink_node_id,
                    sink_ids[SinkName(on_write_failure)],
                    label="__failsink__",
                    mode=RoutingMode.DIVERT,
                )
```

Note: The `sink_settings_map` will need to be passed into `from_plugin_instances()`. Check the current signature and add the parameter. This is a `Mapping[str, SinkSettings]` from the pipeline config.

- [ ] **Step 5: Run tests — expect PASS**

Run: `.venv/bin/python -m pytest tests/unit/engine/test_failsink_validation.py -v`
Expected: All 5 PASS.

- [ ] **Step 6: Run DAG builder tests for regression**

Run: `.venv/bin/python -m pytest tests/unit/core/dag/ -v --tb=short -q`
Expected: All PASS.

- [ ] **Step 7: Commit**

```bash
git add src/elspeth/engine/orchestrator/validation.py src/elspeth/core/dag/builder.py tests/unit/engine/test_failsink_validation.py
git commit -m "feat(validation): add failsink config validation and DAG DIVERT edges"
```

---

### Task 10: Wire orchestrator — inject _on_write_failure + pass failsink to executor

**Files:**
- Modify: `src/elspeth/cli_helpers.py:107`
- Modify: `src/elspeth/engine/orchestrator/core.py:514`

- [ ] **Step 1: Inject _on_write_failure in cli_helpers.py**

In `src/elspeth/cli_helpers.py`, in the sink instantiation loop (line 107-111), add after `sinks[sink_name] = sink_cls(dict(sink_config.options))`:

```python
        # Bridge: inject on_write_failure from settings level
        sinks[sink_name]._on_write_failure = sink_config.on_write_failure
```

- [ ] **Step 2: Resolve failsink references in _drain_pending_tokens_to_sinks()**

In `src/elspeth/engine/orchestrator/core.py`, in `_drain_pending_tokens_to_sinks()` (line 514), modify the `sink_executor.write()` call (line 577) to pass the resolved failsink:

```python
                # Resolve failsink reference (if configured and not 'discard')
                failsink: SinkProtocol | None = None
                if hasattr(sink, '_on_write_failure') and sink._on_write_failure not in (None, 'discard'):
                    failsink_name = sink._on_write_failure
                    if failsink_name in config.sinks:
                        failsink = config.sinks[failsink_name]

                sink_executor.write(
                    sink=sink,
                    tokens=group_tokens,
                    ctx=ctx,
                    step_in_pipeline=step,
                    sink_name=sink_name,
                    pending_outcome=pending_outcome,
                    failsink=failsink,
                    on_token_written=on_token_written,
                )
```

- [ ] **Step 3: Run integration tests**

Run: `.venv/bin/python -m pytest tests/integration/ -v --tb=short -q`
Expected: All PASS (or existing failures only).

- [ ] **Step 4: Commit**

```bash
git add src/elspeth/cli_helpers.py src/elspeth/engine/orchestrator/core.py
git commit -m "feat(orchestrator): wire on_write_failure injection and failsink resolution"
```

---

### Task 11: ChromaSink migration — replace inline filtering with _divert_row()

**Files:**
- Modify: `src/elspeth/plugins/sinks/chroma_sink.py:164`
- Modify: `tests/unit/plugins/sinks/test_chroma_sink.py`

- [ ] **Step 1: Update ChromaSink.write() to use _divert_row()**

Replace the inline metadata filtering block (lines 191-252 of `chroma_sink.py`) with calls to `self._divert_row()`. The ChromaSink already identifies bad metadata types — instead of building `rejected_metadata` and filtering `valid_indices`, call `_divert_row()` for each bad row and `continue`.

Change the return type from `ArtifactDescriptor` to `SinkWriteResult`.

The pattern is shown in the spec's ChromaSink example (Section 2: Plugin Author Ergonomics).

- [ ] **Step 2: Update ChromaSink tests**

In `tests/unit/plugins/sinks/test_chroma_sink.py`:
- Update tests that check `ArtifactDescriptor` returns to check `SinkWriteResult.artifact`
- Update `TestChromaSinkMetadataTypeValidation` tests to assert `_get_diversions()` instead of checking response_data for rejected_metadata
- Add test: ChromaSink with `_on_write_failure` not set raises `FrameworkBugError` when metadata validation fails

- [ ] **Step 3: Run ChromaSink tests**

Run: `.venv/bin/python -m pytest tests/unit/plugins/sinks/test_chroma_sink.py -v`
Expected: All PASS.

- [ ] **Step 4: Commit**

```bash
git add src/elspeth/plugins/sinks/chroma_sink.py tests/unit/plugins/sinks/test_chroma_sink.py
git commit -m "feat(chroma): migrate inline metadata filtering to _divert_row() pattern"
```

---

### Task 12: Full test suite + mypy + ruff

- [ ] **Step 1: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -v --tb=short -q`
Expected: All PASS.

- [ ] **Step 2: Run mypy**

Run: `.venv/bin/python -m mypy src/`
Expected: PASS (or only pre-existing issues).

- [ ] **Step 3: Run ruff**

Run: `.venv/bin/python -m ruff check src/`
Expected: PASS.

- [ ] **Step 4: Run tier model enforcement**

Run: `.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model`
Expected: PASS — `contracts/diversion.py` is L0, imports only from L0.

- [ ] **Step 5: Run config contracts check**

Run: `.venv/bin/python -m scripts.check_contracts`
Expected: PASS.

- [ ] **Step 6: Final commit (if any fixes needed)**

```bash
git add -A
git commit -m "fix: address lint/type issues from failsink implementation"
```
