"""Tests for CSV sink plugin."""

from pathlib import Path

import pytest

from elspeth.contracts import Determinism
from elspeth.contracts.plugin_context import PluginContext
from tests.fixtures.base_classes import inject_write_failure
from tests.fixtures.factories import make_context
from tests.fixtures.landscape import make_factory

# Strict schema config for tests - PathConfig now requires schema
# CSVSink requires fixed-column structure, so we use strict mode
# Tests that need specific fields define their own schema
STRICT_SCHEMA = {"mode": "fixed", "fields": ["id: str", "name: str"]}


class TestCSVSink:
    """Tests for CSVSink plugin."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create a minimal plugin context."""
        factory = make_factory()
        return make_context(landscape=factory.plugin_audit_writer())

    def test_fail_if_exists_collision_policy_refuses_existing_write_target(
        self,
        tmp_path: Path,
    ) -> None:
        """Explicit fail-if-exists mode must not truncate a taken output file."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        output_file.write_text("existing\n")
        with pytest.raises(FileExistsError, match="already exists"):
            CSVSink(
                {
                    "path": str(output_file),
                    "schema": STRICT_SCHEMA,
                    "collision_policy": "fail_if_exists",
                }
            )

        assert output_file.read_text() == "existing\n"

    def test_declared_required_fields_set_from_strict_schema(self, tmp_path: Path) -> None:
        """CSVSink populates declared_required_fields from fixed-mode schema.

        Required-field enforcement is centralized in SinkExecutor (not in sink.write()).
        This test verifies the sink correctly declares which fields are required,
        so the executor can enforce them.
        """
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        sink = inject_write_failure(CSVSink({"path": str(output_file), "schema": STRICT_SCHEMA}))

        # Both 'id' and 'name' are required (no '?' suffix in STRICT_SCHEMA)
        assert sink.declared_required_fields == frozenset({"id", "name"})

    def test_declared_required_fields_excludes_optional(self, tmp_path: Path) -> None:
        """Optional fields (with '?' suffix) are not in declared_required_fields."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        optional_schema = {"mode": "fixed", "fields": ["id: str", "name: str?"]}
        output_file = tmp_path / "output.csv"
        sink = inject_write_failure(CSVSink({"path": str(output_file), "schema": optional_schema}))

        # Only 'id' is required; 'name' has '?' suffix so it's optional
        assert sink.declared_required_fields == frozenset({"id"})
        assert "name" not in sink.declared_required_fields

    def test_has_plugin_version(self) -> None:
        """CSVSink has plugin_version attribute."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        sink = inject_write_failure(CSVSink({"path": "/tmp/test.csv", "schema": STRICT_SCHEMA}))
        assert hasattr(sink, "plugin_version")
        assert sink.plugin_version == "1.0.0"

    def test_has_determinism(self) -> None:
        """CSVSink has determinism attribute."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        sink = inject_write_failure(CSVSink({"path": "/tmp/test.csv", "schema": STRICT_SCHEMA}))
        assert sink.determinism == Determinism.IO_WRITE

    def test_invalid_mode_rejected(self, tmp_path: Path) -> None:
        """Invalid mode values should be rejected at config time.

        This prevents typos like 'apend' from silently truncating files.
        """
        from elspeth.plugins.infrastructure.config_base import PluginConfigError
        from elspeth.plugins.sinks.csv_sink import CSVSinkConfig

        with pytest.raises(PluginConfigError, match=r"'write'.*'append'"):
            CSVSinkConfig.from_dict(
                {
                    "path": str(tmp_path / "output.csv"),
                    "schema": STRICT_SCHEMA,
                    "mode": "apend",  # Typo - should be "append"
                }
            )

    def test_valid_modes_accepted(self, tmp_path: Path) -> None:
        """Valid mode values 'write' and 'append' should be accepted."""
        from elspeth.plugins.sinks.csv_sink import CSVSinkConfig

        # Both valid values should work without error
        write_config = CSVSinkConfig.from_dict(
            {
                "path": str(tmp_path / "write_output.csv"),
                "schema": STRICT_SCHEMA,
                "mode": "write",
            }
        )
        assert write_config.mode == "write"

        append_config = CSVSinkConfig.from_dict(
            {
                "path": str(tmp_path / "append_output.csv"),
                "schema": STRICT_SCHEMA,
                "mode": "append",
            }
        )
        assert append_config.mode == "append"


class TestCSVSinkSchemaValidation:
    """Tests for CSVSink schema modes using infer-and-lock pattern.

    CSVSink supports all schema modes:
    - strict: columns from config, extras rejected at write time
    - free: declared columns + extras from first row, then locked
    - dynamic: columns from first row, then locked

    The "lock" is enforced by DictWriter's extrasaction='raise' default.
    """

    def test_accepts_strict_mode_schema(self, tmp_path: Path) -> None:
        """CSVSink accepts strict mode - columns from config."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        strict_schema = {"mode": "fixed", "fields": ["id: int", "name: str"]}

        sink = inject_write_failure(CSVSink({"path": str(tmp_path / "output.csv"), "schema": strict_schema}))
        assert sink is not None

    def test_accepts_free_mode_schema(self, tmp_path: Path) -> None:
        """CSVSink accepts free mode - declared + first-row extras, then locked."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        free_schema = {"mode": "flexible", "fields": ["id: int"]}

        sink = inject_write_failure(CSVSink({"path": str(tmp_path / "output.csv"), "schema": free_schema}))
        assert sink is not None

    def test_accepts_dynamic_schema(self, tmp_path: Path) -> None:
        """CSVSink accepts dynamic mode - columns from first row, then locked."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        dynamic_schema = {"mode": "observed"}

        sink = inject_write_failure(CSVSink({"path": str(tmp_path / "output.csv"), "schema": dynamic_schema}))
        assert sink is not None


class _UnstringifiableValue:
    """A value whose str()/repr() raise — a broken object, not Tier-2 data.

    Used to document the boundary: csv staging coerces values via str(), and a
    value whose str() raises an arbitrary exception is an upstream-bug-shaped
    fault, NOT operation-unsafe data. The narrow (ValueError, csv.Error) catch
    deliberately does not divert it — it propagates and crashes (Plugin
    Ownership). This mirrors the json_sink reference, which catches only
    (ValueError, TypeError).
    """

    def __str__(self) -> str:
        raise RuntimeError("value cannot be stringified")

    def __repr__(self) -> str:
        raise RuntimeError("value cannot be reprified")


class _ValueErrorStringificationValue:
    """A broken object whose str() raises ValueError specifically."""

    def __str__(self) -> str:
        raise ValueError("value cannot be stringified")

    def __repr__(self) -> str:
        return "_ValueErrorStringificationValue()"


class TestCSVSinkPerRowDiversion:
    """Per-row write faults are diverted, not batch-aborting.

    A failure attributable to one row's CSV shape or target encoding is a
    per-row Tier-2 data fault. With on_write_failure configured the offending
    row is diverted (recorded + routed) and the remaining rows are written,
    rather than aborting the whole batch. With no on_write_failure configured
    the sink fails closed (the framework refuses to guess the operator's
    discard-vs-preserve intent). A value whose own str() raises is an upstream
    bug and must propagate instead of being diverted.

    Batch-integrity failures (file open/permission, disk-full, rollback) remain
    raises — they are not attributable to a single row.
    """

    @pytest.fixture
    def ctx(self) -> PluginContext:
        factory = make_factory()
        return make_context(landscape=factory.plugin_audit_writer())

    # ------------------------------------------------------------------
    # B3.2-csv: per-row codec encoding faults must be diverted, not batch-aborting
    # ------------------------------------------------------------------
