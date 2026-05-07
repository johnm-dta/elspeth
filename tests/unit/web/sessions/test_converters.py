"""Tests for state_from_record — DB record to domain object conversion.

Verifies:
- Round-trip: record → CompositionState preserves all fields
- Tier 1 crash: None metadata_ raises ValueError (database corruption)
- None nodes/edges/outputs → empty sequences (legitimate initial state)
- Source None → source is None (no source configured yet)
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any
from uuid import uuid4

import pytest

from elspeth.web.sessions.converters import state_from_record
from elspeth.web.sessions.protocol import CompositionStateRecord


def _make_record(**overrides: Any) -> CompositionStateRecord:
    """Build a CompositionStateRecord with sensible defaults."""
    defaults: dict[str, Any] = {
        "id": uuid4(),
        "session_id": uuid4(),
        "version": 1,
        "source": {
            "plugin": "csv",
            "on_success": "output",
            "options": {"path": "test.csv", "schema": {"mode": "observed"}},
            "on_validation_failure": "quarantine",
        },
        "nodes": [],
        "edges": [],
        "outputs": [
            {
                "name": "output",
                "plugin": "csv",
                "options": {"path": "out.csv", "schema": {"mode": "observed"}},
                "on_write_failure": "discard",
            }
        ],
        "metadata_": {"name": "Test Pipeline", "description": "A test"},
        "is_valid": True,
        "validation_errors": None,
        "created_at": datetime.now(UTC),
        "derived_from_state_id": None,
    }
    defaults.update(overrides)
    return CompositionStateRecord(**defaults)


class TestStateFromRecord:
    """Round-trip conversion preserves domain semantics."""

    def test_basic_roundtrip(self) -> None:
        """Record with all fields populated converts cleanly."""
        record = _make_record()
        state = state_from_record(record)

        assert state.source is not None
        assert state.source.plugin == "csv"
        assert state.source.on_success == "output"
        assert state.version == 1
        assert state.metadata.name == "Test Pipeline"

    def test_source_none_preserved(self) -> None:
        """None source → CompositionState.source is None."""
        record = _make_record(source=None)
        state = state_from_record(record)
        assert state.source is None

    def test_none_nodes_becomes_empty_tuple(self) -> None:
        """None nodes → empty tuple (legitimate initial state)."""
        record = _make_record(nodes=None)
        state = state_from_record(record)
        assert state.nodes == ()

    def test_none_edges_becomes_empty_tuple(self) -> None:
        """None edges → empty tuple."""
        record = _make_record(edges=None)
        state = state_from_record(record)
        assert state.edges == ()

    def test_none_outputs_becomes_empty_tuple(self) -> None:
        """None outputs → empty tuple."""
        record = _make_record(outputs=None)
        state = state_from_record(record)
        assert state.outputs == ()

    def test_all_collections_none(self) -> None:
        """All optional collections None → valid empty state."""
        record = _make_record(source=None, nodes=None, edges=None, outputs=None)
        state = state_from_record(record)
        assert state.source is None
        assert state.nodes == ()
        assert state.edges == ()
        assert state.outputs == ()

    def test_version_preserved(self) -> None:
        record = _make_record(version=42)
        state = state_from_record(record)
        assert state.version == 42

    def test_frozen_fields_thawed_for_from_dict(self) -> None:
        """Frozen MappingProxyType fields are thawed before from_dict."""
        # CompositionStateRecord freezes its container fields. The converter
        # must deep_thaw them before passing to CompositionState.from_dict().
        record = _make_record()
        # Verify the record's fields are actually frozen
        assert isinstance(record.metadata_, MappingProxyType)
        # Conversion should still work despite frozen fields
        state = state_from_record(record)
        assert state.metadata.name == "Test Pipeline"


class TestTier1MetadataCrash:
    """Tier 1: None metadata_ is database corruption — crash immediately."""

    def test_none_metadata_raises_valueerror(self) -> None:
        record = _make_record(metadata_=None)
        with pytest.raises(ValueError, match="None metadata_"):
            state_from_record(record)
