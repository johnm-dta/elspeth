"""Tests for the Phase-7A extension of PluginSummary."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from elspeth.contracts.enums import AuditCharacteristic
from elspeth.web.catalog.schemas import PluginSummary


def test_plugin_summary_accepts_all_new_fields() -> None:
    """A summary populated with every new field round-trips cleanly."""
    summary = PluginSummary(
        name="csv",
        description="Read rows from a CSV file.",
        plugin_type="source",
        config_fields=[],
        usage_when_to_use="When you have a reasonably large dataset already in a file.",
        usage_when_not_to_use="Small inline data — use 'Inline data from chat' instead.",
        example_use="source:\n  plugin: csv\n  options:\n    path: data/input.csv",
        capability_tags=("csv", "file", "batch"),
        audit_characteristics=(
            AuditCharacteristic.COERCE,
            AuditCharacteristic.IO_READ,
            AuditCharacteristic.QUARANTINE,
        ),
    )
    assert summary.capability_tags == ("csv", "file", "batch")
    assert AuditCharacteristic.IO_READ in summary.audit_characteristics


def test_plugin_summary_defaults_for_unfilled_plugin() -> None:
    """A summary with no reference content uses the documented defaults."""
    summary = PluginSummary(
        name="azure_blob",
        description="Read blobs from Azure storage.",
        plugin_type="source",
        config_fields=[],
    )
    assert summary.usage_when_to_use is None
    assert summary.usage_when_not_to_use is None
    assert summary.example_use is None
    assert summary.capability_tags == ()
    assert summary.audit_characteristics == ()


def test_plugin_summary_rejects_extra_fields() -> None:
    """_StrictResponse uses extra='forbid'; unknown fields crash."""
    with pytest.raises(ValidationError):
        PluginSummary(
            name="csv",
            description="...",
            plugin_type="source",
            config_fields=[],
            mystery_field="surprise",  # type: ignore[call-arg]
        )


def test_audit_characteristics_serializes_as_list_for_json() -> None:
    """Pydantic emits tuple as a list in JSON. The derivation helper sorts
    before constructing the tuple, so the wire order is deterministic and
    matches `sorted(...)`; the frontend's `string[]` typing reads this
    directly.  AuditCharacteristic is a StrEnum so members serialise as
    their string value on the wire."""
    summary = PluginSummary(
        name="csv",
        description="...",
        plugin_type="source",
        config_fields=[],
        audit_characteristics=(AuditCharacteristic.IO_READ, AuditCharacteristic.QUARANTINE),
    )
    payload = summary.model_dump(mode="json")
    assert isinstance(payload["audit_characteristics"], list)
    assert payload["audit_characteristics"] == ["io_read", "quarantine"]
