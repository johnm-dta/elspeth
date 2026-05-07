# Sink Failsink Part 1: Contracts & Infrastructure

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the foundational types, enums, config fields, and BaseSink plumbing for the failsink pattern. After this plan, all contracts exist, all sinks return `SinkWriteResult`, and `BaseSink._divert_row()` is available for plugin authors.

**Architecture:** New `RowDiversion` and `SinkWriteResult` frozen dataclasses in `contracts/diversion.py`. `DIVERTED` added to `RowOutcome`. `on_write_failure` becomes a mandatory field on `SinkSettings`. `BaseSink` gains `_divert_row()` infrastructure. All sink `write()` return types change from `ArtifactDescriptor` to `SinkWriteResult`.

**Tech Stack:** Python dataclasses, Pydantic (config), pytest (testing).

**Spec:** `docs/superpowers/specs/2026-03-26-sink-failsink-design.md`

**Depends on:** Nothing (this is the foundation).
**Blocks:** Part 2 (engine + wiring) depends on everything here.

---

### Task 1: RowDiversion and SinkWriteResult contracts

**Files:**
- Create: `src/elspeth/contracts/diversion.py`
- Create: `tests/unit/contracts/test_diversion.py`
- Modify: `src/elspeth/contracts/__init__.py:236-244`

- [ ] **Step 1: Write tests for RowDiversion and SinkWriteResult**

```python
# tests/unit/contracts/test_diversion.py
"""Tests for sink diversion contracts."""
from __future__ import annotations

from typing import Any

import pytest

from elspeth.contracts.diversion import RowDiversion, SinkWriteResult
from elspeth.contracts.results import ArtifactDescriptor


class TestRowDiversion:
    def test_create_with_required_fields(self) -> None:
        d = RowDiversion(row_index=0, reason="bad metadata", row_data={"a": 1})
        assert d.row_index == 0
        assert d.reason == "bad metadata"
        assert d.row_data["a"] == 1

    def test_frozen(self) -> None:
        d = RowDiversion(row_index=0, reason="test", row_data={"a": 1})
        with pytest.raises(AttributeError):
            d.row_index = 1  # type: ignore[misc]

    def test_row_data_deep_frozen(self) -> None:
        """row_data uses freeze_fields — nested dicts become MappingProxyType."""
        d = RowDiversion(row_index=0, reason="test", row_data={"a": {"nested": 1}})
        with pytest.raises(TypeError):
            d.row_data["b"] = 2  # type: ignore[index]

    def test_row_data_nested_frozen(self) -> None:
        """Nested dicts inside row_data are also frozen."""
        d = RowDiversion(row_index=0, reason="test", row_data={"a": {"nested": 1}})
        with pytest.raises(TypeError):
            d.row_data["a"]["new_key"] = 99  # type: ignore[index]


class TestSinkWriteResult:
    def _make_artifact(self) -> ArtifactDescriptor:
        return ArtifactDescriptor.for_file(
            path="/tmp/test.csv",
            content_hash="abc123def456",
            size_bytes=100,
        )

    def test_no_diversions_default(self) -> None:
        result = SinkWriteResult(artifact=self._make_artifact())
        assert result.diversions == ()
        assert result.artifact.path_or_uri == "/tmp/test.csv"

    def test_with_diversions(self) -> None:
        divs = (
            RowDiversion(row_index=1, reason="bad type", row_data={"x": 1}),
            RowDiversion(row_index=3, reason="too long", row_data={"x": 2}),
        )
        result = SinkWriteResult(artifact=self._make_artifact(), diversions=divs)
        assert len(result.diversions) == 2
        assert result.diversions[0].row_index == 1
        assert result.diversions[1].row_index == 3

    def test_frozen(self) -> None:
        result = SinkWriteResult(artifact=self._make_artifact())
        with pytest.raises(AttributeError):
            result.artifact = self._make_artifact()  # type: ignore[misc]

    def test_diversions_tuple_not_list(self) -> None:
        """Diversions must be a tuple (immutable), not a list."""
        result = SinkWriteResult(artifact=self._make_artifact())
        assert isinstance(result.diversions, tuple)
```

- [ ] **Step 2: Run tests — expect FAIL (module not found)**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_diversion.py -v`
Expected: `ModuleNotFoundError: No module named 'elspeth.contracts.diversion'`

- [ ] **Step 3: Implement RowDiversion and SinkWriteResult**

```python
# src/elspeth/contracts/diversion.py
"""Sink diversion contracts — per-row write failure routing types.

These types support the failsink pattern: when a sink can't write a specific
row (value-level failure at the Tier 2 -> External boundary), it diverts the
row to a failsink. These contracts carry diversion information from the plugin's
write() method back to SinkExecutor for audit trail recording.

Layer: L0 (contracts). Imports only from L0 and stdlib.
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

    Replaces ArtifactDescriptor as the return type of BaseSink.write().
    Sinks with no diversions return SinkWriteResult(artifact=..., diversions=()).

    Attributes:
        artifact: ArtifactDescriptor for the primary write (may represent zero rows
            if all were diverted).
        diversions: Tuple of RowDiversion records for rows diverted during write().
            Empty tuple if no diversions occurred.
    """

    artifact: ArtifactDescriptor
    diversions: tuple[RowDiversion, ...] = ()
```

- [ ] **Step 4: Add exports to contracts/__init__.py**

In `src/elspeth/contracts/__init__.py`, find the results import block (lines 236-244):

```python
from elspeth.contracts.results import (
    ArtifactDescriptor,
    ExceptionResult,
    FailureInfo,
    GateResult,
    RowResult,
    SourceRow,
    TransformResult,
)
```

Add after this block:

```python
from elspeth.contracts.diversion import RowDiversion, SinkWriteResult
```

- [ ] **Step 5: Run tests — expect PASS**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_diversion.py -v`
Expected: All 7 tests PASS.

- [ ] **Step 6: Run mypy on the new module**

Run: `.venv/bin/python -m mypy src/elspeth/contracts/diversion.py`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/elspeth/contracts/diversion.py tests/unit/contracts/test_diversion.py src/elspeth/contracts/__init__.py
git commit -m "feat(contracts): add RowDiversion and SinkWriteResult for failsink pattern"
```

---

### Task 2: DIVERTED RowOutcome + RunResult.rows_diverted

**Files:**
- Modify: `src/elspeth/contracts/enums.py:147-179`
- Modify: `src/elspeth/contracts/run_result.py:19-49`

- [ ] **Step 1: Write test for DIVERTED outcome**

```python
# tests/unit/contracts/test_diverted_outcome.py
"""Test DIVERTED RowOutcome and RunResult.rows_diverted."""
from __future__ import annotations

import pytest

from elspeth.contracts.enums import RowOutcome, RunStatus
from elspeth.contracts.run_result import RunResult


class TestDivertedOutcome:
    def test_diverted_value(self) -> None:
        assert RowOutcome.DIVERTED == "diverted"
        assert RowOutcome.DIVERTED.value == "diverted"

    def test_diverted_is_terminal(self) -> None:
        assert RowOutcome.DIVERTED.is_terminal is True

    def test_diverted_in_terminal_set(self) -> None:
        """DIVERTED should be in the same category as QUARANTINED and ROUTED."""
        terminal_outcomes = [o for o in RowOutcome if o.is_terminal]
        assert RowOutcome.DIVERTED in terminal_outcomes


class TestRunResultDiverted:
    def test_rows_diverted_default_zero(self) -> None:
        result = RunResult(
            run_id="test-1",
            status=RunStatus.COMPLETED,
            rows_processed=10,
            rows_succeeded=8,
            rows_failed=2,
            rows_routed=0,
        )
        assert result.rows_diverted == 0

    def test_rows_diverted_explicit(self) -> None:
        result = RunResult(
            run_id="test-1",
            status=RunStatus.COMPLETED,
            rows_processed=10,
            rows_succeeded=7,
            rows_failed=0,
            rows_routed=0,
            rows_diverted=3,
        )
        assert result.rows_diverted == 3

    def test_rows_diverted_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="rows_diverted"):
            RunResult(
                run_id="test-1",
                status=RunStatus.COMPLETED,
                rows_processed=10,
                rows_succeeded=10,
                rows_failed=0,
                rows_routed=0,
                rows_diverted=-1,
            )
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_diverted_outcome.py -v`
Expected: `AttributeError: DIVERTED` (not on RowOutcome yet).

- [ ] **Step 3: Add DIVERTED to RowOutcome**

In `src/elspeth/contracts/enums.py`, find the terminal outcomes block (lines 168-176):

```python
    # Terminal outcomes
    COMPLETED = "completed"
    ROUTED = "routed"
    FORKED = "forked"
    FAILED = "failed"
    QUARANTINED = "quarantined"
    CONSUMED_IN_BATCH = "consumed_in_batch"
    COALESCED = "coalesced"
    EXPANDED = "expanded"
```

Add after `QUARANTINED = "quarantined"` (line 173):

```python
    DIVERTED = "diverted"
```

Update the docstring (lines 148-166). Find:
```
    - QUARANTINED: Failed validation, stored for investigation
```

Add after it:
```
    - DIVERTED: Sink write failed for this row, diverted to failsink
```

- [ ] **Step 4: Add rows_diverted to RunResult**

In `src/elspeth/contracts/run_result.py`, find (lines 28-33):

```python
    rows_quarantined: int = 0
    rows_forked: int = 0
    rows_coalesced: int = 0
    rows_coalesce_failed: int = 0  # Coalesce failures (quorum_not_met, incomplete_branches)
    rows_expanded: int = 0  # Deaggregation parent tokens
    rows_buffered: int = 0  # Passthrough mode buffered tokens
```

Add after `rows_buffered: int = 0` (line 33):

```python
    rows_diverted: int = 0  # Rows diverted to failsink during sink write
```

In `__post_init__`, find (line 48):

```python
        require_int(self.rows_buffered, "rows_buffered", min_value=0)
```

Add after it:

```python
        require_int(self.rows_diverted, "rows_diverted", min_value=0)
```

- [ ] **Step 5: Run tests — expect PASS**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_diverted_outcome.py -v`
Expected: All 6 tests PASS.

- [ ] **Step 6: Run existing contracts tests for regression**

Run: `.venv/bin/python -m pytest tests/unit/contracts/ -v --tb=short -q`
Expected: All PASS. New enum member + new field with default=0 is backwards-compatible.

- [ ] **Step 7: Commit**

```bash
git add src/elspeth/contracts/enums.py src/elspeth/contracts/run_result.py tests/unit/contracts/test_diverted_outcome.py
git commit -m "feat(contracts): add DIVERTED outcome and rows_diverted counter"
```

---

### Task 3: SinkSettings.on_write_failure (mandatory config field)

**Files:**
- Modify: `src/elspeth/core/config.py:916-926`
- Create: `tests/unit/core/test_sink_settings_on_write_failure.py`

- [ ] **Step 1: Write tests for on_write_failure**

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

    def test_on_write_failure_whitespace_only_rejected(self) -> None:
        with pytest.raises(ValidationError, match="on_write_failure"):
            SinkSettings(plugin="csv", on_write_failure="   ")

    def test_on_write_failure_whitespace_stripped(self) -> None:
        s = SinkSettings(plugin="csv", on_write_failure="  discard  ")
        assert s.on_write_failure == "discard"

    def test_on_write_failure_with_options(self) -> None:
        s = SinkSettings(
            plugin="chroma_sink",
            on_write_failure="csv_failsink",
            options={"collection": "test"},
        )
        assert s.on_write_failure == "csv_failsink"
        assert s.options == {"collection": "test"}
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `.venv/bin/python -m pytest tests/unit/core/test_sink_settings_on_write_failure.py -v`
Expected: FAIL — `on_write_failure` doesn't exist on SinkSettings yet. The `test_on_write_failure_required` test will unexpectedly pass (no field = no error from pydantic for missing field), but `test_on_write_failure_discard` will fail with `unexpected keyword argument`.

- [ ] **Step 3: Add on_write_failure to SinkSettings**

In `src/elspeth/core/config.py`, find the SinkSettings class (lines 916-926):

```python
class SinkSettings(BaseModel):
    """Sink plugin configuration per architecture."""

    model_config = {"frozen": True, "extra": "forbid"}

    plugin: str = Field(description="Plugin name (csv, json, database, webhook, etc.)")
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Plugin-specific configuration options",
    )
```

Replace with:

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

Verify that `field_validator` is imported from pydantic (it is — used by `TransformSettings` at line 896) and `_validate_connection_or_sink_name` exists (it does — used at lines 901 and 913).

- [ ] **Step 4: Run new tests — expect PASS**

Run: `.venv/bin/python -m pytest tests/unit/core/test_sink_settings_on_write_failure.py -v`
Expected: All 7 tests PASS.

- [ ] **Step 5: Enumerate and fix all broken SinkSettings usages**

The new required field will break every existing `SinkSettings(plugin=..., options=...)` call that doesn't include `on_write_failure`. Before running tests, enumerate the blast radius:

Run: `grep -rn "SinkSettings(" tests/ src/ --include="*.py" | grep -v "on_write_failure" | grep -v "test_sink_settings_on_write_failure"`

This shows every `SinkSettings(` construction that is missing `on_write_failure`. Also search for YAML sink blocks:

Run: `grep -rn "plugin:" tests/ examples/ --include="*.yaml" --include="*.yml" -B1 -A2 | grep -v "on_write_failure"`

For each Python call site: add `on_write_failure="discard"`.
For each YAML sink block: add `on_write_failure: discard`.

Treat this as a bounded checklist — every match from the grep must be fixed before running the test suite. Do not use an iterative "run tests, find next failure" loop.

- [ ] **Step 6: Run full test suite — expect PASS**

Run: `.venv/bin/python -m pytest tests/ -x --tb=short -q`
Expected: All PASS.

- [ ] **Step 7: Commit**

```bash
git add src/elspeth/core/config.py tests/
git commit -m "feat(config): add mandatory on_write_failure to SinkSettings"
```

---

### Task 4: BaseSink._divert_row() infrastructure

**Files:**
- Modify: `src/elspeth/plugins/infrastructure/base.py:339-566`
- Create: `tests/unit/plugins/infrastructure/test_base_sink_divert.py`

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
        for i, row in enumerate(rows):
            if "__should_divert" in row and row["__should_divert"]:
                self._divert_row(row, row_index=i, reason="test diversion")
        artifact = ArtifactDescriptor.for_file(
            path="/tmp/stub", content_hash="x" * 64, size_bytes=0,
        )
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
        assert diversions[0].reason == "bad value"
        assert diversions[1].row_index == 2
        assert diversions[1].reason == "too long"

    def test_divert_without_on_write_failure_raises(self) -> None:
        """Calling _divert_row when _on_write_failure is None is a plugin bug."""
        sink = StubSink({})
        # _on_write_failure defaults to None — not set by orchestrator
        with pytest.raises(FrameworkBugError, match="on_write_failure"):
            sink._divert_row({"a": 1}, row_index=0, reason="test")

    def test_reset_clears_log(self) -> None:
        sink = StubSink({})
        sink._on_write_failure = "discard"
        sink._divert_row({"a": 1}, row_index=0, reason="test")
        assert len(sink._get_diversions()) == 1
        sink._reset_diversion_log()
        assert len(sink._get_diversions()) == 0

    def test_divert_with_discard_mode_accumulates(self) -> None:
        """Discard mode still accumulates to log — executor decides outcome."""
        sink = StubSink({})
        sink._on_write_failure = "discard"
        sink._divert_row({"a": 1}, row_index=0, reason="intentional drop")
        diversions = sink._get_diversions()
        assert len(diversions) == 1
        assert diversions[0].reason == "intentional drop"

    def test_get_diversions_returns_tuple(self) -> None:
        """Diversions are returned as an immutable tuple."""
        sink = StubSink({})
        sink._on_write_failure = "csv_failsink"
        sink._divert_row({"a": 1}, row_index=0, reason="test")
        result = sink._get_diversions()
        assert isinstance(result, tuple)

    def test_row_data_preserved_in_diversion(self) -> None:
        """The original row data is captured in the RowDiversion."""
        sink = StubSink({})
        sink._on_write_failure = "csv_failsink"
        original_row = {"field_a": "value", "field_b": 42}
        sink._divert_row(original_row, row_index=0, reason="test")
        diversions = sink._get_diversions()
        assert diversions[0].row_data["field_a"] == "value"
        assert diversions[0].row_data["field_b"] == 42


class TestStubSinkWriteWithDiversion:
    def test_write_returns_diversions_from_log(self) -> None:
        """Integration: write() uses _divert_row and _get_diversions correctly."""
        sink = StubSink({})
        sink._on_write_failure = "csv_failsink"
        rows = [
            {"value": "ok"},
            {"value": "bad", "__should_divert": True},
            {"value": "ok2"},
        ]
        result = sink.write(rows, ctx=None)
        assert isinstance(result, SinkWriteResult)
        assert len(result.diversions) == 1
        assert result.diversions[0].row_index == 1
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `.venv/bin/python -m pytest tests/unit/plugins/infrastructure/test_base_sink_divert.py -v`
Expected: FAIL — `_on_write_failure`, `_divert_row` don't exist on BaseSink.

- [ ] **Step 3: Add _divert_row() infrastructure to BaseSink**

In `src/elspeth/plugins/infrastructure/base.py`, add imports at the top. Find the existing import block (around lines 40-51):

```python
from elspeth.contracts import ArtifactDescriptor, Determinism, PluginSchema, SourceRow
```

Add after it:

```python
from elspeth.contracts.diversion import RowDiversion, SinkWriteResult
from elspeth.contracts.errors import FrameworkBugError
```

In the `BaseSink` class, add after the existing class attributes (around line 407, after `validate_input: bool = False`):

```python
    # Failsink infrastructure — set by orchestrator from SinkSettings.on_write_failure
    # None until injected at pipeline startup; "discard" or sink name at runtime.
    _on_write_failure: str | None = None
```

In `__init__` (line 488), add after `self._needs_resume_field_resolution = False` (line 490):

```python
        self._diversion_log: list[RowDiversion] = []
```

Add the three methods after `set_output_contract()` (around line 544), before the lifecycle hooks section:

```python
    # === Failsink Infrastructure ===

    def _divert_row(self, row: dict[str, Any], row_index: int, reason: str) -> None:
        """Divert a row to the failsink or discard. Called by plugin write() on per-row failure.

        This is the sink-side equivalent of SourceRow.quarantined(). The plugin
        catches a per-row exception from the external system and calls this
        instead of re-raising.

        Both "discard" and failsink modes accumulate to _diversion_log.
        The executor reads the log after write() returns and handles the
        actual discard-vs-write decision.

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
Expected: All 8 tests PASS.

- [ ] **Step 5: Run existing BaseSink / sink tests for regression**

Run: `.venv/bin/python -m pytest tests/unit/plugins/ -v --tb=short -q`
Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/plugins/infrastructure/base.py tests/unit/plugins/infrastructure/test_base_sink_divert.py
git commit -m "feat(base): add _divert_row() infrastructure to BaseSink"
```

---

### Task 5: Update SinkProtocol.write() return type

**Files:**
- Modify: `src/elspeth/contracts/plugin_protocols.py:451-465`

- [ ] **Step 1: Update SinkProtocol.write() return type**

In `src/elspeth/contracts/plugin_protocols.py`, find the TYPE_CHECKING imports (around line 22):

```python
    from elspeth.contracts.results import ArtifactDescriptor, SourceRow, TransformResult
```

Add to this block:

```python
    from elspeth.contracts.diversion import SinkWriteResult
```

Find the `write()` method on SinkProtocol (lines 451-465):

```python
    def write(
        self,
        rows: list[dict[str, Any]],
        ctx: "SinkContext",
    ) -> "ArtifactDescriptor":
        """Write a batch of rows to the sink.

        Args:
            rows: List of row dicts to write
            ctx: Sink context with run identity and recording methods

        Returns:
            ArtifactDescriptor with content_hash and size_bytes (REQUIRED for audit)
        """
        ...
```

Change the return type and docstring:

```python
    def write(
        self,
        rows: list[dict[str, Any]],
        ctx: "SinkContext",
    ) -> "SinkWriteResult":
        """Write a batch of rows to the sink.

        Args:
            rows: List of row dicts to write
            ctx: Sink context with run identity and recording methods

        Returns:
            SinkWriteResult containing ArtifactDescriptor and any RowDiversions.
            Sinks with no diversions return SinkWriteResult(artifact=...).
        """
        ...
```

Also update the example in the SinkProtocol class docstring (around line 411):

```python
            def write(self, rows: list[dict], ctx: SinkContext) -> ArtifactDescriptor:
```

Change to:

```python
            def write(self, rows: list[dict], ctx: SinkContext) -> SinkWriteResult:
```

- [ ] **Step 2: Update BaseSink.write() abstract method**

In `src/elspeth/plugins/infrastructure/base.py`, find the abstract write() (around line 492):

```python
    @abstractmethod
    def write(
        self,
        rows: list[dict[str, Any]],
        ctx: SinkContext,
    ) -> ArtifactDescriptor:
```

Change return type to `SinkWriteResult`.

- [ ] **Step 3: Run mypy on protocol and base**

Run: `.venv/bin/python -m mypy src/elspeth/contracts/plugin_protocols.py src/elspeth/plugins/infrastructure/base.py`
Expected: Errors about sink implementations not matching the new return type. This is expected — we'll fix them in Task 6.

- [ ] **Step 4: Commit**

```bash
git add src/elspeth/contracts/plugin_protocols.py src/elspeth/plugins/infrastructure/base.py
git commit -m "feat(protocol): update SinkProtocol.write() return type to SinkWriteResult"
```

---

### Task 6: Migrate all sink write() return types to SinkWriteResult

**Files:**
- Modify: `src/elspeth/plugins/sinks/csv_sink.py:20,241,257,323`
- Modify: `src/elspeth/plugins/sinks/json_sink.py:17,245,261,296`
- Modify: `src/elspeth/plugins/sinks/database_sink.py:25,445,474,553`
- Modify: `src/elspeth/plugins/sinks/azure_blob_sink.py:28,534,552,666`
- Modify: `src/elspeth/plugins/sinks/chroma_sink.py:164,248,376`

Each sink gets the same mechanical change:
1. Add import: `from elspeth.contracts.diversion import SinkWriteResult`
2. Change `write()` return annotation from `ArtifactDescriptor` to `SinkWriteResult`
3. Wrap each `return ArtifactDescriptor.for_*(...)` with `return SinkWriteResult(artifact=ArtifactDescriptor.for_*(...))`

- [ ] **Step 1: Update CSVSink**

In `src/elspeth/plugins/sinks/csv_sink.py`:

Add import (after line 20 `from elspeth.contracts import ArtifactDescriptor, PluginSchema`):
```python
from elspeth.contracts.diversion import SinkWriteResult
```

Change write() signature (line 241):
```python
    def write(self, rows: list[dict[str, Any]], ctx: SinkContext) -> SinkWriteResult:
```

Change first return (line 257-261, empty batch):
```python
        return SinkWriteResult(artifact=ArtifactDescriptor.for_file(
            path=str(self._path),
            content_hash=content_hash,
            size_bytes=0,
        ))
```

Change second return (line 323-327, normal):
```python
        return SinkWriteResult(artifact=ArtifactDescriptor.for_file(
            path=str(self._path),
            content_hash=content_hash,
            size_bytes=size_bytes,
        ))
```

- [ ] **Step 2: Update JSONSink**

Same pattern in `src/elspeth/plugins/sinks/json_sink.py`:

Add import after line 17: `from elspeth.contracts.diversion import SinkWriteResult`

Change write() signature (line 245) return type to `SinkWriteResult`.

Wrap returns at lines 261-265 and 296-300.

- [ ] **Step 3: Update DatabaseSink**

Same pattern in `src/elspeth/plugins/sinks/database_sink.py`:

Add import after line 25: `from elspeth.contracts.diversion import SinkWriteResult`

Change write() signature (line 445) return type to `SinkWriteResult`.

Wrap returns at lines 474-480 and 553-559.

- [ ] **Step 4: Update AzureBlobSink**

Same pattern in `src/elspeth/plugins/sinks/azure_blob_sink.py`:

Add import after line 28: `from elspeth.contracts.diversion import SinkWriteResult`

Change write() signature (line 534) return type to `SinkWriteResult`.

Wrap returns at lines 552-557 and 666-671.

- [ ] **Step 5: Update ChromaSink (return type only — migration is Part 2)**

In `src/elspeth/plugins/sinks/chroma_sink.py`:

Add import (after line 23 `from elspeth.contracts.results import ArtifactDescriptor`):
```python
from elspeth.contracts.diversion import SinkWriteResult
```

Change write() signature (line 164) return type to `SinkWriteResult`.

Wrap returns at lines 248-254 (all-rejected) and 376-382 (normal). **Do NOT change the inline metadata filtering yet** — that's Part 2, Task 11.

- [ ] **Step 6: Fix test assertions that check write() return type**

Run the test suite to find failures:

Run: `.venv/bin/python -m pytest tests/unit/plugins/sinks/ -x --tb=line -q 2>&1 | head -20`

For each failure, update the test to expect `SinkWriteResult` instead of `ArtifactDescriptor`:

- `result.content_hash` → `result.artifact.content_hash`
- `result.path_or_uri` → `result.artifact.path_or_uri`
- `result.size_bytes` → `result.artifact.size_bytes`
- `result.artifact_type` → `result.artifact.artifact_type`
- `isinstance(result, ArtifactDescriptor)` → `isinstance(result, SinkWriteResult)`

Repeat until all sink tests pass.

- [ ] **Step 7: Run full sink test suite — expect PASS**

Run: `.venv/bin/python -m pytest tests/unit/plugins/sinks/ -v --tb=short -q`
Expected: All PASS.

- [ ] **Step 8: Run mypy on all sinks**

Run: `.venv/bin/python -m mypy src/elspeth/plugins/sinks/`
Expected: PASS (or only pre-existing issues).

- [ ] **Step 9: Commit**

```bash
git add src/elspeth/plugins/sinks/ tests/unit/plugins/sinks/
git commit -m "refactor(sinks): update all sink write() return types to SinkWriteResult"
```

---

### Task 7: ExecutionCounters.rows_diverted + accumulate_row_outcomes DIVERTED branch

**Files:**
- Modify: `src/elspeth/engine/orchestrator/types.py:104-220`
- Modify: `src/elspeth/engine/orchestrator/outcomes.py:73-130`
- Create: `tests/unit/engine/test_outcomes_diverted.py`

- [ ] **Step 1: Write tests for DIVERTED in accumulate_row_outcomes**

```python
# tests/unit/engine/test_outcomes_diverted.py
"""Test DIVERTED outcome branch in accumulate_row_outcomes."""
from __future__ import annotations

from unittest.mock import MagicMock

from elspeth.contracts.enums import RowOutcome
from elspeth.engine.orchestrator.outcomes import accumulate_row_outcomes
from elspeth.engine.orchestrator.types import ExecutionCounters


def _make_result(outcome: RowOutcome, sink_name: str | None = None) -> MagicMock:
    """Create a mock RowResult with the given outcome."""
    result = MagicMock()
    result.outcome = outcome
    result.sink_name = sink_name
    result.token = MagicMock()
    result.token.token_id = f"tok-{id(result)}"
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

    def test_multiple_diverted(self) -> None:
        counters = ExecutionCounters()
        pending_tokens: dict[str, list] = {"output": []}
        results = [
            _make_result(RowOutcome.DIVERTED),
            _make_result(RowOutcome.DIVERTED),
            _make_result(RowOutcome.DIVERTED),
        ]
        accumulate_row_outcomes(results, counters, {"output": MagicMock()}, pending_tokens)
        assert counters.rows_diverted == 3

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

    def test_diverted_in_run_result(self) -> None:
        """rows_diverted flows through to RunResult via to_run_result()."""
        from elspeth.contracts.enums import RunStatus
        counters = ExecutionCounters()
        counters.rows_diverted = 5
        counters.rows_processed = 10
        counters.rows_succeeded = 5
        result = counters.to_run_result(run_id="test-1", status=RunStatus.COMPLETED)
        assert result.rows_diverted == 5
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `.venv/bin/python -m pytest tests/unit/engine/test_outcomes_diverted.py -v`
Expected: `AttributeError: 'ExecutionCounters' object has no attribute 'rows_diverted'`

- [ ] **Step 3: Add rows_diverted to ExecutionCounters and AggregationFlushResult**

In `src/elspeth/engine/orchestrator/types.py`:

**AggregationFlushResult** (line 103-138): Add `rows_diverted: int = 0` after `rows_buffered` (line 118). Update `__add__` (line 124):
```python
            rows_diverted=self.rows_diverted + other.rows_diverted,
```

**ExecutionCounters** (line 141-220): Add `rows_diverted: int = 0` after `routed_destinations` (line 163).

Update `accumulate_flush_result()` (line 165): Add after `rows_buffered`:
```python
        self.rows_diverted += result.rows_diverted
```

Update `to_flush_result()` (line 182): Add after `rows_buffered`:
```python
            rows_diverted=self.rows_diverted,
```

Update `to_run_result()` (line 199): Add after `rows_buffered`:
```python
            rows_diverted=self.rows_diverted,
```

- [ ] **Step 4: Add DIVERTED branch to accumulate_row_outcomes()**

In `src/elspeth/engine/orchestrator/outcomes.py`, find (lines 123-129):

```python
        elif result.outcome == RowOutcome.EXPANDED:
            # Deaggregation parent token - children counted separately
            counters.rows_expanded += 1
        elif result.outcome == RowOutcome.BUFFERED:
            # Passthrough mode buffered token
            counters.rows_buffered += 1
        else:
```

Add before the `BUFFERED` branch:

```python
        elif result.outcome == RowOutcome.DIVERTED:
            counters.rows_diverted += 1
            # DIVERTED tokens are already written to failsink by SinkExecutor.
            # They are NOT appended to pending_tokens — no further routing needed.
```

**IMPORTANT — Counter increment path:** In normal pipeline execution, the `DIVERTED` branch in `accumulate_row_outcomes()` will NOT fire — diversions are determined inside `SinkExecutor.write()`, after the processor has already emitted `COMPLETED` outcomes for those tokens. The `rows_diverted` counter is primarily incremented by `_write_pending_to_sinks()` in Part 2, which reads the diversion count from `SinkExecutor.write()`'s return value and adds it to `counters.rows_diverted`. The branch here exists as a guard for future code paths that might emit `DIVERTED` from the processor level, and to prevent `OrchestrationInvariantError` on unrecognised outcomes.

- [ ] **Step 5: Run tests — expect PASS**

Run: `.venv/bin/python -m pytest tests/unit/engine/test_outcomes_diverted.py -v`
Expected: All 5 tests PASS.

- [ ] **Step 6: Run broader orchestrator tests for regression**

Run: `.venv/bin/python -m pytest tests/unit/engine/ -v --tb=short -q`
Expected: All PASS.

- [ ] **Step 7: Commit**

```bash
git add src/elspeth/engine/orchestrator/types.py src/elspeth/engine/orchestrator/outcomes.py tests/unit/engine/test_outcomes_diverted.py
git commit -m "feat(engine): add DIVERTED branch to outcome accumulation and counters"
```

---

### Task 8: Part 1 verification — full suite + CI checks

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

- [ ] **Step 6: Fix any issues and commit**

```bash
git add -A
git commit -m "fix: address lint/type issues from failsink Part 1"
```
