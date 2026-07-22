"""Tests for CSVSink append mode."""

from pathlib import Path

import pytest

from elspeth.contracts.plugin_context import PluginContext
from elspeth.plugins.sinks.csv_sink import CSVSink
from tests.fixtures.base_classes import inject_write_failure
from tests.fixtures.factories import make_context
from tests.fixtures.landscape import make_factory

# Strict schema config for tests - CSVSink requires fixed columns
STRICT_SCHEMA = {"mode": "fixed", "fields": ["id: int", "value: str"]}


@pytest.fixture
def ctx() -> PluginContext:
    """Create test context."""
    factory = make_factory()
    return make_context(landscape=factory.plugin_audit_writer())


class TestCSVSinkAppendMode:
    """Tests for CSVSink append mode."""

    def test_append_mode_declared_required_fields_set(self, tmp_path: Path) -> None:
        """Append-mode sink declares required fields for executor enforcement.

        Required-field enforcement is centralized in SinkExecutor. This test
        verifies the sink correctly populates declared_required_fields so the
        executor can reject rows with missing fields before they reach write().
        """
        output_path = tmp_path / "output.csv"

        sink = inject_write_failure(
            CSVSink(
                {
                    "path": str(output_path),
                    "schema": STRICT_SCHEMA,
                    "mode": "append",
                }
            )
        )

        assert sink.declared_required_fields == frozenset({"id", "value"})
