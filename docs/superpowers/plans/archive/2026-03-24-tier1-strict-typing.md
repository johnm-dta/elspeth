# Tier 1 Strict Typing — Burn Down Loose Types in Audit Paths

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate `dict[str, Any]`, bare `str`, and missing freeze guards across all Tier 1 (Landscape/audit) code paths, so that mypy catches type errors at development time rather than at runtime.

**Architecture:** Three layers of work: (1) introduce missing enums and TypedDicts in the contracts layer, (2) fix freeze guard violations in existing dataclasses, (3) propagate strict types through the repository/recorder facade and serialization code. Each task is independently testable and commitable.

**Tech Stack:** Python 3.12, mypy (strict on modified files), existing `contracts/enums.py` StrEnum patterns, `contracts/freeze.py` for deep immutability, TypedDict for export record shapes.

---

## File Map

### New Files
- `src/elspeth/contracts/coalesce_enums.py` — `CoalescePolicy` and `MergeStrategy` StrEnums
- `src/elspeth/contracts/export_records.py` — TypedDicts for exporter record shapes
- `tests/unit/contracts/test_coalesce_enums.py` — Enum exhaustiveness tests
- `tests/unit/contracts/test_export_records.py` — TypedDict construction tests

### Modified Files
- `src/elspeth/contracts/coalesce_metadata.py` — Replace `str` fields with enums, add `freeze_fields`
- `src/elspeth/contracts/__init__.py` — Re-export new enums
- `src/elspeth/core/config.py:712-718` — Use enum values in `Literal` (or replace with enum)
- `src/elspeth/core/landscape/exporter.py` — Use TypedDict return types for record builders
- `src/elspeth/core/landscape/formatters.py` — Narrow `Any` parameters where possible
- `src/elspeth/core/landscape/recorder.py` — Narrow `dict[str, Any]` parameters on write methods
- `src/elspeth/core/landscape/execution_repository.py` — Same narrowing
- `src/elspeth/core/landscape/data_flow_repository.py` — Same narrowing
- `tests/unit/contracts/test_coalesce_metadata.py` — Update for enum fields + freeze validation

---

## Task 1: CoalescePolicy and MergeStrategy enums

**Rationale:** `CoalesceMetadata.policy` and `.merge_strategy` are bare `str` but the domain has exactly 4 policies (`require_all`, `quorum`, `best_effort`, `first`) and 3 merge strategies (`union`, `nested`, `select`). These are already `Literal` types in `CoalesceSettings` (config.py:712-718). Extracting enums means mypy catches typos and the audit trail records validated values, not arbitrary strings.

**Files:**
- Create: `src/elspeth/contracts/coalesce_enums.py`
- Create: `tests/unit/contracts/test_coalesce_enums.py`
- Modify: `src/elspeth/contracts/__init__.py`

- [ ] **Step 1: Write failing tests for the new enums**

```python
# tests/unit/contracts/test_coalesce_enums.py
"""Tests for CoalescePolicy and MergeStrategy enums."""

from elspeth.contracts.coalesce_enums import CoalescePolicy, MergeStrategy


class TestCoalescePolicy:
    def test_members(self) -> None:
        assert set(CoalescePolicy) == {
            CoalescePolicy.REQUIRE_ALL,
            CoalescePolicy.QUORUM,
            CoalescePolicy.BEST_EFFORT,
            CoalescePolicy.FIRST,
        }

    def test_values_match_config_literals(self) -> None:
        """Values must match the Literal strings in CoalesceSettings.policy."""
        assert CoalescePolicy.REQUIRE_ALL.value == "require_all"
        assert CoalescePolicy.QUORUM.value == "quorum"
        assert CoalescePolicy.BEST_EFFORT.value == "best_effort"
        assert CoalescePolicy.FIRST.value == "first"

    def test_round_trip_from_string(self) -> None:
        for member in CoalescePolicy:
            assert CoalescePolicy(member.value) is member

    def test_invalid_value_raises(self) -> None:
        import pytest
        with pytest.raises(ValueError):
            CoalescePolicy("nonexistent")


class TestMergeStrategy:
    def test_members(self) -> None:
        assert set(MergeStrategy) == {
            MergeStrategy.UNION,
            MergeStrategy.NESTED,
            MergeStrategy.SELECT,
        }

    def test_values_match_config_literals(self) -> None:
        assert MergeStrategy.UNION.value == "union"
        assert MergeStrategy.NESTED.value == "nested"
        assert MergeStrategy.SELECT.value == "select"

    def test_invalid_value_raises(self) -> None:
        import pytest
        with pytest.raises(ValueError):
            MergeStrategy("nonexistent")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_coalesce_enums.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'elspeth.contracts.coalesce_enums'`

- [ ] **Step 3: Implement the enums**

```python
# src/elspeth/contracts/coalesce_enums.py
"""Coalesce policy and merge strategy enums for the audit trail.

These replace bare ``str`` in CoalesceMetadata so that mypy catches
invalid policy/strategy values at development time. Values match the
Literal strings in ``CoalesceSettings`` (core/config.py).
"""

from enum import StrEnum


class CoalescePolicy(StrEnum):
    """How a coalesce point handles partial branch arrivals."""

    REQUIRE_ALL = "require_all"
    QUORUM = "quorum"
    BEST_EFFORT = "best_effort"
    FIRST = "first"


class MergeStrategy(StrEnum):
    """How a coalesce point combines row data from branches."""

    UNION = "union"
    NESTED = "nested"
    SELECT = "select"
```

- [ ] **Step 4: Add re-exports to `contracts/__init__.py`**

Add `CoalescePolicy` and `MergeStrategy` to the imports and `__all__` in `src/elspeth/contracts/__init__.py`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_coalesce_enums.py -v`
Expected: PASS — all 7 tests green

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/contracts/coalesce_enums.py tests/unit/contracts/test_coalesce_enums.py src/elspeth/contracts/__init__.py
git commit -m "feat: add CoalescePolicy and MergeStrategy StrEnums for Tier 1 typing"
```

---

## Task 2: Fix CoalesceMetadata — enums + freeze_fields

**Rationale:** `CoalesceMetadata` has two issues: (1) `policy: str` and `merge_strategy: str | None` should use the new enums, and (2) `__post_init__` doesn't call `freeze_fields` despite having `MappingProxyType` container fields. The factory methods already pre-wrap in `MappingProxyType`, but the contract requires `freeze_fields` for consistency and safety (a caller could construct directly and pass a raw dict).

**⚠️ Reviewer note (N4):** Existing tests in `test_coalesce_metadata.py` use `policy="wait_all"` which is NOT a valid policy. These tests will break when the field becomes `CoalescePolicy`. Fix all `"wait_all"` occurrences → `CoalescePolicy.REQUIRE_ALL` (or whichever policy the test intends).

**Files:**
- Modify: `src/elspeth/contracts/coalesce_metadata.py`
- Modify: `tests/unit/contracts/test_coalesce_metadata.py` — update `"wait_all"` → valid enum values
- Modify: `src/elspeth/engine/coalesce_executor.py` — update construction sites to pass enums

- [ ] **Step 1: Write failing tests for enum fields and freeze guards**

```python
# In test_coalesce_metadata.py — add these tests

import pytest
from types import MappingProxyType
from elspeth.contracts.coalesce_enums import CoalescePolicy, MergeStrategy
from elspeth.contracts.coalesce_metadata import CoalesceMetadata


class TestCoalesceMetadataEnumFields:
    def test_policy_is_enum(self) -> None:
        meta = CoalesceMetadata(policy=CoalescePolicy.REQUIRE_ALL)
        assert isinstance(meta.policy, CoalescePolicy)

    def test_merge_strategy_is_enum(self) -> None:
        meta = CoalesceMetadata(
            policy=CoalescePolicy.REQUIRE_ALL,
            merge_strategy=MergeStrategy.UNION,
        )
        assert isinstance(meta.merge_strategy, MergeStrategy)

    def test_factory_for_late_arrival_uses_enum(self) -> None:
        meta = CoalesceMetadata.for_late_arrival(
            policy=CoalescePolicy.REQUIRE_ALL,
            reason="test",
        )
        assert meta.policy is CoalescePolicy.REQUIRE_ALL


class TestCoalesceMetadataFreezeGuards:
    def test_direct_construction_with_raw_dict_freezes(self) -> None:
        """Even if someone bypasses factories, freeze_fields catches it."""
        meta = CoalesceMetadata(
            policy=CoalescePolicy.REQUIRE_ALL,
            branches_lost={"a": "timeout"},  # type: ignore[arg-type] — raw dict
        )
        # freeze_fields should have converted to MappingProxyType
        assert isinstance(meta.branches_lost, MappingProxyType)

    def test_direct_construction_with_raw_collisions_freezes(self) -> None:
        meta = CoalesceMetadata(
            policy=CoalescePolicy.REQUIRE_ALL,
            union_field_collisions={"x": ("a", "b")},  # type: ignore[arg-type]
        )
        assert isinstance(meta.union_field_collisions, MappingProxyType)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_coalesce_metadata.py -v -k "Enum or Freeze"`
Expected: FAIL — `policy` is `str`, not `CoalescePolicy`; raw dicts not frozen

- [ ] **Step 3: Update CoalesceMetadata to use enums and freeze_fields**

In `src/elspeth/contracts/coalesce_metadata.py`:

1. Change `policy: str` → `policy: CoalescePolicy`
2. Change `merge_strategy: str | None` → `merge_strategy: MergeStrategy | None`
3. Add `freeze_fields` call in `__post_init__` for `branches_lost` and `union_field_collisions`
4. Update `to_dict()` to emit `.value` for enum fields
5. Update factory method type annotations

```python
# Key changes in __post_init__:
def __post_init__(self) -> None:
    if not self.policy:
        raise ValueError("CoalesceMetadata.policy must not be empty")
    # Freeze container fields — catches direct construction with raw dicts
    fields_to_freeze = []
    if self.branches_lost is not None:
        fields_to_freeze.append("branches_lost")
    if self.union_field_collisions is not None:
        fields_to_freeze.append("union_field_collisions")
    if fields_to_freeze:
        freeze_fields(self, *fields_to_freeze)
```

- [ ] **Step 4: Update coalesce_executor.py construction sites**

Replace string literals with enum members at all 4 factory call sites:
- `policy=settings.policy` → `policy=CoalescePolicy(settings.policy)` (settings.policy is a Literal str from Pydantic)
- `merge_strategy=settings.merge` → `merge_strategy=MergeStrategy(settings.merge)`

- [ ] **Step 5: Run full coalesce test suite**

Run: `.venv/bin/python -m pytest tests/ -k "coalesce" -v`
Expected: ALL PASS

- [ ] **Step 6: Run mypy on modified files**

Run: `.venv/bin/python -m mypy src/elspeth/contracts/coalesce_metadata.py src/elspeth/engine/coalesce_executor.py`
Expected: No errors

- [ ] **Step 7: Commit**

```bash
git add src/elspeth/contracts/coalesce_metadata.py src/elspeth/engine/coalesce_executor.py tests/
git commit -m "fix: CoalesceMetadata — enum fields + freeze_fields in __post_init__"
```

---

## Task 3: Exporter TypedDicts — typed record shapes

**Rationale:** `LandscapeExporter._iter_records()` yields `dict[str, Any]` for 15 different record types. Each record type has a fixed schema. TypedDicts make these schemas visible to mypy and serve as documentation for downstream consumers (CSV/JSON export, compliance review).

**Files:**
- Create: `src/elspeth/contracts/export_records.py`
- Create: `tests/unit/contracts/test_export_records.py`
- Modify: `src/elspeth/core/landscape/exporter.py` — annotate record construction

- [ ] **Step 1: Write tests for TypedDict constructability**

```python
# tests/unit/contracts/test_export_records.py
"""Verify export record TypedDicts can be constructed with expected fields."""

from elspeth.contracts.export_records import (
    RunExportRecord,
    NodeExportRecord,
    EdgeExportRecord,
    RowExportRecord,
    TokenExportRecord,
    TokenParentExportRecord,
    TokenOutcomeExportRecord,
    NodeStateExportRecord,
    RoutingEventExportRecord,
    CallExportRecord,
    BatchExportRecord,
    BatchMemberExportRecord,
    ArtifactExportRecord,
    OperationExportRecord,
    SecretResolutionExportRecord,
)


class TestRunExportRecord:
    def test_construction(self) -> None:
        rec: RunExportRecord = {
            "record_type": "run",
            "run_id": "r1",
            "status": "completed",
            "started_at": "2026-01-01T00:00:00",
            "completed_at": "2026-01-01T01:00:00",
            "canonical_version": "1.0",
            "config_hash": "abc123",
            "settings": {"key": "value"},
            "reproducibility_grade": "deterministic",
        }
        assert rec["record_type"] == "run"
```

Each TypedDict should have a minimal construction test verifying the required fields. Keep these as smoke tests — the exporter integration tests verify full correctness.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_export_records.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement the TypedDicts**

```python
# src/elspeth/contracts/export_records.py
"""TypedDict definitions for Landscape export records.

Each TypedDict defines the exact shape of one record type yielded by
``LandscapeExporter._iter_records()``. Replaces ``dict[str, Any]``
so that mypy can verify field names and types at construction sites.
"""

from __future__ import annotations

from typing import Any, TypedDict


class RunExportRecord(TypedDict):
    record_type: str  # Literal["run"] — but kept as str for union compatibility
    run_id: str
    status: str
    started_at: str | None
    completed_at: str | None
    canonical_version: str
    config_hash: str
    settings: Any  # Resolved config — structure varies by pipeline
    reproducibility_grade: str | None


class NodeExportRecord(TypedDict):
    record_type: str
    run_id: str
    node_id: str
    plugin_name: str
    node_type: str
    plugin_version: str
    determinism: str
    config_hash: str
    config: Any  # Resolved plugin config — structure varies by plugin
    schema_hash: str | None
    schema_mode: str | None
    schema_fields: list[dict[str, object]] | None
    sequence_in_pipeline: int


# ... (one TypedDict per record type, ~15 total)
# Pattern: each mirrors the dict literal in exporter.py exactly
```

Note: the full file will have ~15 TypedDicts. The implementer should create one TypedDict per record type by reading the corresponding `yield {...}` block in `exporter.py` lines 198-567.

- [ ] **Step 4: Annotate exporter record construction**

In `exporter.py`, change `_iter_records` return from `Iterator[dict[str, Any]]` to `Iterator[ExportRecord]` where `ExportRecord = RunExportRecord | NodeExportRecord | ...`. Annotate each `yield` block with the specific TypedDict. This gives mypy visibility into field correctness at each construction site.

The public `export_run()` and `export_run_grouped()` can keep `dict[str, Any]` returns since they're the external API and TypedDict is structurally compatible.

- [ ] **Step 5: Run exporter tests**

Run: `.venv/bin/python -m pytest tests/ -k "export" -v`
Expected: ALL PASS

- [ ] **Step 6: Run mypy**

Run: `.venv/bin/python -m mypy src/elspeth/contracts/export_records.py src/elspeth/core/landscape/exporter.py`
Expected: No errors

- [ ] **Step 7: Commit**

```bash
git add src/elspeth/contracts/export_records.py tests/unit/contracts/test_export_records.py src/elspeth/core/landscape/exporter.py
git commit -m "feat: TypedDict export records — typed shapes for all 15 Landscape record types"
```

---

## Task 4: Narrow formatters.py — reduce Any surface

**Rationale:** `formatters.py` has `serialize_datetime(obj: Any) -> Any` and `dataclass_to_dict(obj: Any) -> Any`. These are inherently recursive over heterogeneous structures, so full type elimination isn't practical. But we can: (1) add `@overload` signatures for the common concrete cases, and (2) change `ExportFormatter.format()` to accept the new TypedDict union instead of `dict[str, Any]`.

**Files:**
- Modify: `src/elspeth/core/landscape/formatters.py`

- [ ] **Step 1: Add overload signatures to serialize_datetime**

```python
from typing import overload

@overload
def serialize_datetime(obj: dict[str, object]) -> dict[str, object]: ...
@overload
def serialize_datetime(obj: list[object]) -> list[object]: ...
@overload
def serialize_datetime(obj: float) -> float: ...
@overload
def serialize_datetime(obj: datetime) -> str: ...
@overload
def serialize_datetime(obj: Any) -> Any: ...

def serialize_datetime(obj: Any) -> Any:
    # existing implementation unchanged
```

- [ ] **Step 2: Update ExportFormatter protocol**

Change `ExportFormatter.format()` parameter from `dict[str, Any]` to use the TypedDict union from Task 3.

- [ ] **Step 3: Run tests**

Run: `.venv/bin/python -m pytest tests/ -k "format" -v`
Expected: ALL PASS

- [ ] **Step 4: Run mypy**

Run: `.venv/bin/python -m mypy src/elspeth/core/landscape/formatters.py`
Expected: No errors

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/core/landscape/formatters.py
git commit -m "refactor: add overload signatures to serialize_datetime, narrow formatter types"
```

---

## Task 5: Narrow recorder/repository dict[str, Any] parameters

**Rationale:** The `recorder.py` facade and its backing repositories (`execution_repository.py`, `data_flow_repository.py`) accept `dict[str, Any]` for pipeline data parameters (`input_data`, `output_data`, `row_data`, `context`, `config`). These parameters carry **Tier 2 pipeline data** (post-source, type-valid), not arbitrary dicts. The correct narrowing is `dict[str, Any]` → `Mapping[str, object]` — this:

1. Makes the read-only intent clear (Mapping vs dict)
2. Replaces `Any` values with `object` (forces explicit narrowing at use sites)
3. Accepts both `dict` and `PipelineRow` (which implements Mapping)

**Important:** This is a signature-only change. The internal logic hashes these dicts via `canonical_json()` which already accepts `Any`. No implementation changes needed.

**Files:**
- Modify: `src/elspeth/core/landscape/recorder.py`
- Modify: `src/elspeth/core/landscape/execution_repository.py`
- Modify: `src/elspeth/core/landscape/data_flow_repository.py`

- [ ] **Step 1: Change parameter types in execution_repository.py**

For every `input_data: dict[str, Any]`, `output_data: dict[str, Any]`, and similar parameters, change to `Mapping[str, object]`. For `output_data` that can also be a list, use `Mapping[str, object] | list[Mapping[str, object]]`.

- [ ] **Step 2: Change parameter types in data_flow_repository.py**

Same pattern. `row_data: dict[str, Any]` → `Mapping[str, object]`, `context: dict[str, Any]` → `Mapping[str, object]`, `config: dict[str, Any]` → `Mapping[str, object]`.

- [ ] **Step 3: Change parameter types in recorder.py facade**

Mirror the repository changes in the facade methods.

- [ ] **Step 4: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -x`
Expected: ALL PASS — `Mapping[str, object]` is a supertype of `dict[str, Any]`, so all existing callers are compatible.

- [ ] **Step 5: Run mypy on all three files**

Run: `.venv/bin/python -m mypy src/elspeth/core/landscape/recorder.py src/elspeth/core/landscape/execution_repository.py src/elspeth/core/landscape/data_flow_repository.py`
Expected: No new errors. Note: some callers passing `dict[str, Any]` may trigger warnings — those are downstream issues to fix later, not blockers.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/core/landscape/recorder.py src/elspeth/core/landscape/execution_repository.py src/elspeth/core/landscape/data_flow_repository.py
git commit -m "refactor: narrow Tier 1 write paths — dict[str, Any] → Mapping[str, object]"
```

---

## Task 6: Narrow query_repository.py return type

**Rationale:** `query_repository.py` has `_retrieve_and_parse_payload() -> dict[str, Any]` which deserializes stored JSON payloads. This is Tier 1 data (our own payload store), so the return could be `dict[str, object]` at minimum.

**Files:**
- Modify: `src/elspeth/core/landscape/query_repository.py`

- [ ] **Step 1: Change return type**

`_retrieve_and_parse_payload(self, row_id: str, source_data_ref: str) -> dict[str, object]`

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/ -k "query" -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add src/elspeth/core/landscape/query_repository.py
git commit -m "refactor: narrow _retrieve_and_parse_payload return to dict[str, object]"
```

---

## Task 7: Final mypy sweep and CI verification

**Rationale:** After all changes, run mypy in strict mode on the modified files to catch any regressions, then run the full test suite and CI checks.

**Files:** All modified files from Tasks 1-6.

- [ ] **Step 1: Run mypy on all Tier 1 files**

```bash
.venv/bin/python -m mypy \
  src/elspeth/contracts/coalesce_enums.py \
  src/elspeth/contracts/coalesce_metadata.py \
  src/elspeth/contracts/export_records.py \
  src/elspeth/core/landscape/recorder.py \
  src/elspeth/core/landscape/execution_repository.py \
  src/elspeth/core/landscape/data_flow_repository.py \
  src/elspeth/core/landscape/query_repository.py \
  src/elspeth/core/landscape/formatters.py \
  src/elspeth/core/landscape/exporter.py
```

Expected: No errors

- [ ] **Step 2: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -x`
Expected: ALL PASS

- [ ] **Step 3: Run ruff**

Run: `.venv/bin/python -m ruff check src/elspeth/contracts/ src/elspeth/core/landscape/`
Expected: No new violations

- [ ] **Step 4: Run tier model enforcer**

Run: `.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model`
Expected: PASS — new enums are in contracts (L0), no upward imports

- [ ] **Step 5: Commit any fixups**

If any issues found in steps 1-4, fix and commit.

---

## Scope Notes

### What this plan covers
- All `dict[str, Any]` in Tier 1 write paths (recorder, repositories)
- Missing freeze guards (CoalesceMetadata)
- Bare `str` where enums exist (coalesce policy/strategy)
- Untyped export record shapes (exporter)
- Loose `Any` in formatters

### What this plan does NOT cover (future work)
- **`checkpoint/serialization.py`** — `Any` params are inherent to recursive JSON ser/de. Already well-guarded with NaN/Infinity rejection. Low value to narrow.
- **`journal.py`** — `Any` in SQLAlchemy event handler signatures. These are dictated by SQLAlchemy's callback protocol. Cannot narrow without wrapper types.
- **`CoalesceSettings` config.py Literal → enum migration** — The Pydantic model uses `Literal["require_all", ...]` which works with StrEnum values. Could migrate to use the enum directly, but that's a config-layer change (different concern).
- **Exporter `to_dict()` methods on audit dataclasses** — Many `to_dict() -> dict[str, Any]` on contracts. These are serialization boundaries with legitimately variable shapes. TypedDict per class is possible but high-volume low-value.
- **`PipelineRow` integration** — `row_data: dict[str, Any] | PipelineRow` could be unified to a protocol. Separate effort.
