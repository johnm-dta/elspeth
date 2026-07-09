"""Tests for sink failsink destination validation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.contracts.sink import FAILSINK_ELIGIBLE_PLUGIN_TEXT, FAILSINK_ELIGIBLE_SINK_PLUGINS
from elspeth.engine.orchestrator.types import RouteValidationError
from elspeth.engine.orchestrator.validation import validate_sink_failsink_destinations


def _stub(on_write_failure: str | None) -> SimpleNamespace:
    """Minimal stub with on_write_failure attribute."""
    return SimpleNamespace(on_write_failure=on_write_failure)


class TestValidateSinkFailsinkDestinations:
    def test_discard_always_valid(self) -> None:
        """on_write_failure='discard' needs no target sink."""
        validate_sink_failsink_destinations(
            sink_configs={"output": _stub("discard")},
            available_sinks={"output"},
            sink_plugins={"output": "chroma_sink"},
        )  # No error raised

    def test_valid_failsink_reference(self) -> None:
        validate_sink_failsink_destinations(
            sink_configs={
                "output": _stub("csv_failsink"),
                "csv_failsink": _stub("discard"),
            },
            available_sinks={"output", "csv_failsink"},
            sink_plugins={"output": "chroma_sink", "csv_failsink": "csv"},
        )  # No error raised

    def test_json_failsink_valid(self) -> None:
        validate_sink_failsink_destinations(
            sink_configs={
                "output": _stub("json_failsink"),
                "json_failsink": _stub("discard"),
            },
            available_sinks={"output", "json_failsink"},
            sink_plugins={"output": "database", "json_failsink": "json"},
        )  # No error raised

    @pytest.mark.parametrize("plugin_name", sorted(FAILSINK_ELIGIBLE_SINK_PLUGINS))
    def test_shared_failsink_capability_plugins_are_valid(self, plugin_name: str) -> None:
        failsink_name = f"{plugin_name}_failsink"
        validate_sink_failsink_destinations(
            sink_configs={
                "output": _stub(failsink_name),
                failsink_name: _stub("discard"),
            },
            available_sinks={"output", failsink_name},
            sink_plugins={"output": "database", failsink_name: plugin_name},
        )  # No error raised

    def test_unknown_failsink_raises(self) -> None:
        with pytest.raises(RouteValidationError, match="nonexistent"):
            validate_sink_failsink_destinations(
                sink_configs={"output": _stub("nonexistent")},
                available_sinks={"output"},
                sink_plugins={"output": "chroma_sink"},
            )

    def test_non_file_failsink_raises(self) -> None:
        """Failsink must use a shared failsink-capable plugin."""
        with pytest.raises(RouteValidationError, match=FAILSINK_ELIGIBLE_PLUGIN_TEXT):
            validate_sink_failsink_destinations(
                sink_configs={
                    "output": _stub("db_sink"),
                    "db_sink": _stub("discard"),
                },
                available_sinks={"output", "db_sink"},
                sink_plugins={"output": "chroma_sink", "db_sink": "database"},
            )

    def test_chroma_as_failsink_raises(self) -> None:
        """ChromaSink is not a valid failsink plugin type."""
        with pytest.raises(RouteValidationError, match=FAILSINK_ELIGIBLE_PLUGIN_TEXT):
            validate_sink_failsink_destinations(
                sink_configs={
                    "output": _stub("chroma_backup"),
                    "chroma_backup": _stub("discard"),
                },
                available_sinks={"output", "chroma_backup"},
                sink_plugins={"output": "csv", "chroma_backup": "chroma_sink"},
            )

    def test_failsink_chaining_raises(self) -> None:
        """Failsink targets must have on_write_failure='discard' (no chains)."""
        with pytest.raises(RouteValidationError, match="discard"):
            validate_sink_failsink_destinations(
                sink_configs={
                    "output": _stub("failsink1"),
                    "failsink1": _stub("failsink2"),
                    "failsink2": _stub("discard"),
                },
                available_sinks={"output", "failsink1", "failsink2"},
                sink_plugins={"output": "chroma_sink", "failsink1": "csv", "failsink2": "csv"},
            )

    def test_self_reference_raises(self) -> None:
        """A sink cannot reference itself as failsink."""
        with pytest.raises(RouteValidationError, match="itself"):
            validate_sink_failsink_destinations(
                sink_configs={"output": _stub("output")},
                available_sinks={"output"},
                sink_plugins={"output": "csv"},
            )

    def test_multiple_sinks_mixed_valid(self) -> None:
        """Multiple sinks: some with failsink, some with discard."""
        validate_sink_failsink_destinations(
            sink_configs={
                "chroma_out": _stub("csv_fail"),
                "db_out": _stub("discard"),
                "csv_fail": _stub("discard"),
            },
            available_sinks={"chroma_out", "db_out", "csv_fail"},
            sink_plugins={"chroma_out": "chroma_sink", "db_out": "database", "csv_fail": "csv"},
        )  # No error raised

    def test_none_on_write_failure_raises(self) -> None:
        """on_write_failure=None (pre-injection default) is a framework bug."""
        with pytest.raises(OrchestrationInvariantError, match="injection was skipped"):
            validate_sink_failsink_destinations(
                sink_configs={"output": _stub(None)},
                available_sinks={"output"},
                sink_plugins={"output": "chroma_sink"},
            )

    def test_sink_plugins_missing_key_crashes(self) -> None:
        """If sink_plugins is missing a key that available_sinks has, it's a framework bug.

        Rule 2 guarantees dest exists in available_sinks. Rule 4 must use direct
        access into sink_plugins — defensive `if dest in sink_plugins` would silently
        skip the plugin type check on map inconsistency.
        """
        with pytest.raises(KeyError, match="csv_failsink"):
            validate_sink_failsink_destinations(
                sink_configs={
                    "output": _stub("csv_failsink"),
                    "csv_failsink": _stub("discard"),
                },
                available_sinks={"output", "csv_failsink"},
                # Bug: csv_failsink missing from sink_plugins
                sink_plugins={"output": "chroma_sink"},
            )

    def test_all_discard(self) -> None:
        """All sinks using discard — valid."""
        validate_sink_failsink_destinations(
            sink_configs={
                "sink_a": _stub("discard"),
                "sink_b": _stub("discard"),
            },
            available_sinks={"sink_a", "sink_b"},
            sink_plugins={"sink_a": "csv", "sink_b": "json"},
        )  # No error raised
