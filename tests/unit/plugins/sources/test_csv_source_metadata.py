"""Tests asserting csv_source.py is the canonical Phase-7A example."""

from __future__ import annotations

from elspeth.plugins.sources.csv_source import CSVSource


def test_csv_source_has_when_to_use() -> None:
    assert CSVSource.usage_when_to_use is not None
    # Sanity-check that the prose is actually content, not a stub.
    assert len(CSVSource.usage_when_to_use) > 40


def test_csv_source_has_when_not_to_use() -> None:
    assert CSVSource.usage_when_not_to_use is not None
    assert len(CSVSource.usage_when_not_to_use) > 40


def test_csv_source_has_example_use() -> None:
    assert CSVSource.example_use is not None
    assert "csv" in CSVSource.example_use


def test_csv_source_has_capability_tags() -> None:
    tags = CSVSource.capability_tags
    assert "csv" in tags
    assert "file" in tags


def test_csv_source_declared_audit_characteristics_includes_coerce() -> None:
    """CSV source coerces external string data to typed columns per
    Tier-3 boundary rules; that's a notable audit trait worth declaring."""
    assert "coerce" in CSVSource.audit_characteristics
