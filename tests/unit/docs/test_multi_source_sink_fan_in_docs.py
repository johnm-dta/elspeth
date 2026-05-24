"""Regression guards for multi-source sink fan-in contract documentation."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_system_operations_documents_direct_sink_fan_in_policy() -> None:
    """The system-operations contract must explain the SINK fan-in exemption."""
    system_operations = (PROJECT_ROOT / "docs/contracts/system-operations.md").read_text()

    assert "Direct multi-source fan-in to a sink is allowed without an explicit queue." in system_operations
    assert "Sinks are terminal write boundaries, not ordinary processing nodes." in system_operations
    assert "`ingest_sequence` remains the ordering authority for direct sink fan-in." in system_operations
    assert "Transforms, aggregations, and gates still require an explicit `queue`" in system_operations
