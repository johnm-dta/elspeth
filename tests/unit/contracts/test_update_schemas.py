"""Structural drift checks for recorder update TypedDicts.

``ExportStatusUpdate`` and ``BatchStatusUpdate`` are partial-update DTOs for
the Tier-1 audit records ``Run`` and ``Batch``. They are exported from
``elspeth.contracts`` (public surface) and describe what fields recorder
methods are permitted to write when updating those records.

These tests guard the *contract*, not Python's dict-literal syntax. They
detect:

1. Drift in the set of keys (additions, removals, renames).
2. Drift in the annotation type of each key.
3. Drift in the ``__total__`` flag (partial-update semantics).
4. Drift between the update DTO and the parent dataclass: every update key
   must exist as a field on the parent dataclass, and the update annotation
   must equal the parent's non-``None`` payload type.

If a developer adds, removes, renames, or retypes a field on ``Run`` or
``Batch`` without making the matching change to the update TypedDict (or
vice versa), exactly one of these tests fails with attribution to the key
that drifted.
"""

from __future__ import annotations

import dataclasses
import types
from datetime import datetime
from typing import Union, get_args, get_origin, get_type_hints

import pytest

from elspeth.contracts import (
    BatchStatus,
    BatchStatusUpdate,
    ExportStatus,
    ExportStatusUpdate,
)
from elspeth.contracts.audit import Batch, Run


def _strip_optional(annotation: object) -> object:
    """Return the non-``None`` payload of ``T | None`` / ``Optional[T]``.

    Returns ``annotation`` unchanged if it is not a union with ``None``.
    Crashes if a union has more than one non-``None`` member, since the
    update TypedDicts are designed to mirror simple optional fields.
    """
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        non_none = tuple(a for a in get_args(annotation) if a is not type(None))
        if len(non_none) == 1:
            return non_none[0]
        raise AssertionError(f"Cannot strip Optional from multi-arm union: {annotation!r}")
    return annotation


# ---------------------------------------------------------------------------
# ExportStatusUpdate ↔ Run
# ---------------------------------------------------------------------------

EXPECTED_EXPORT_KEYS: dict[str, type] = {
    "export_status": ExportStatus,
    "exported_at": datetime,
    "export_error": str,
    "export_format": str,
    "export_sink": str,
}


class TestExportStatusUpdateContract:
    """Drift guard for ``ExportStatusUpdate`` against ``Run``."""

    def test_keys_match_expected_set(self) -> None:
        """Key set must match exactly — no additions, no removals."""
        actual = set(get_type_hints(ExportStatusUpdate).keys())
        expected = set(EXPECTED_EXPORT_KEYS.keys())
        missing = expected - actual
        added = actual - expected
        assert not missing and not added, f"ExportStatusUpdate key drift: missing={sorted(missing)} added={sorted(added)}"

    @pytest.mark.parametrize(
        ("key", "expected_type"),
        sorted(EXPECTED_EXPORT_KEYS.items()),
    )
    def test_each_annotation_type(self, key: str, expected_type: type) -> None:
        """Each key's annotation must match the expected type exactly."""
        hints = get_type_hints(ExportStatusUpdate)
        assert hints[key] is expected_type, f"ExportStatusUpdate.{key} annotation drifted: expected {expected_type!r}, got {hints[key]!r}"

    def test_total_is_false_for_partial_updates(self) -> None:
        """``total=False`` is load-bearing — recorder issues partial updates."""
        assert ExportStatusUpdate.__total__ is False, (
            "ExportStatusUpdate.__total__ must be False; partial updates are required by the recorder contract."
        )

    def test_keys_are_subset_of_run_fields(self) -> None:
        """Every update key must name a real field on ``Run``."""
        run_fields = {f.name for f in dataclasses.fields(Run)}
        update_keys = set(get_type_hints(ExportStatusUpdate).keys())
        orphans = update_keys - run_fields
        assert not orphans, (
            f"ExportStatusUpdate keys not present on Run: {sorted(orphans)} — update DTO references fields the audit record cannot store."
        )

    @pytest.mark.parametrize("key", sorted(EXPECTED_EXPORT_KEYS.keys()))
    def test_update_type_matches_run_field_payload(self, key: str) -> None:
        """Update annotation must equal the parent field's non-``None`` type."""
        run_hints = get_type_hints(Run)
        update_hints = get_type_hints(ExportStatusUpdate)
        run_payload = _strip_optional(run_hints[key])
        assert update_hints[key] is run_payload, (
            f"ExportStatusUpdate.{key}={update_hints[key]!r} drifted from "
            f"Run.{key} payload={run_payload!r} (full Run field: "
            f"{run_hints[key]!r})"
        )


# ---------------------------------------------------------------------------
# BatchStatusUpdate ↔ Batch
# ---------------------------------------------------------------------------

EXPECTED_BATCH_KEYS: dict[str, type] = {
    "status": BatchStatus,
    "completed_at": datetime,
    "trigger_reason": str,
    "aggregation_state_id": str,
}


class TestBatchStatusUpdateContract:
    """Drift guard for ``BatchStatusUpdate`` against ``Batch``."""

    def test_keys_match_expected_set(self) -> None:
        """Key set must match exactly — no additions, no removals."""
        actual = set(get_type_hints(BatchStatusUpdate).keys())
        expected = set(EXPECTED_BATCH_KEYS.keys())
        missing = expected - actual
        added = actual - expected
        assert not missing and not added, f"BatchStatusUpdate key drift: missing={sorted(missing)} added={sorted(added)}"

    @pytest.mark.parametrize(
        ("key", "expected_type"),
        sorted(EXPECTED_BATCH_KEYS.items()),
    )
    def test_each_annotation_type(self, key: str, expected_type: type) -> None:
        """Each key's annotation must match the expected type exactly."""
        hints = get_type_hints(BatchStatusUpdate)
        assert hints[key] is expected_type, f"BatchStatusUpdate.{key} annotation drifted: expected {expected_type!r}, got {hints[key]!r}"

    def test_total_is_false_for_partial_updates(self) -> None:
        """``total=False`` is load-bearing — recorder issues partial updates."""
        assert BatchStatusUpdate.__total__ is False, (
            "BatchStatusUpdate.__total__ must be False; partial updates are required by the recorder contract."
        )

    def test_keys_are_subset_of_batch_fields(self) -> None:
        """Every update key must name a real field on ``Batch``."""
        batch_fields = {f.name for f in dataclasses.fields(Batch)}
        update_keys = set(get_type_hints(BatchStatusUpdate).keys())
        orphans = update_keys - batch_fields
        assert not orphans, (
            f"BatchStatusUpdate keys not present on Batch: {sorted(orphans)} — update DTO references fields the audit record cannot store."
        )

    @pytest.mark.parametrize("key", sorted(EXPECTED_BATCH_KEYS.keys()))
    def test_update_type_matches_batch_field_payload(self, key: str) -> None:
        """Update annotation must equal the parent field's non-``None`` type."""
        batch_hints = get_type_hints(Batch)
        update_hints = get_type_hints(BatchStatusUpdate)
        batch_payload = _strip_optional(batch_hints[key])
        assert update_hints[key] is batch_payload, (
            f"BatchStatusUpdate.{key}={update_hints[key]!r} drifted from "
            f"Batch.{key} payload={batch_payload!r} (full Batch field: "
            f"{batch_hints[key]!r})"
        )
